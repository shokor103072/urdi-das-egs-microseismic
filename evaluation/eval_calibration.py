"""
NLL and Brier Score Calibration Metrics
TNNLS Reviewer Request: calibration metrics beyond ECE

Metrics computed on Utah FORGE 3-2417 (zero-shot):
  - NLL  : Negative Log-Likelihood = -mean(y*log(p) + (1-y)*log(1-p))
  - Brier: mean squared error between probabilities and labels
  - ECE  : Expected Calibration Error (existing, recomputed for consistency)
  - MCE  : Maximum Calibration Error
  - AUROC: Area under ROC curve for error detection
           (how well uncertainty predicts misclassification)

Uses saved URDI model weights from multiseed run.
Reports mean +/- std across 5 seeds for baseline and URDI lambda=10.

Outputs:
  data/calibration_metrics_report.txt
  data/figures/paper_calibration_extended.png
"""

import os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NPY_X2    = Path("./Dataset/X_forge.npy")
NPY_Y2    = Path("./Dataset/y_forge.npy")
MODEL_DIR = Path("./Model")
DATA_DIR  = Path("./data")
FIG_DIR   = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS     = [42, 7, 13, 99, 2024]
BATCH     = 32
MC_SAMPLES = 20   # for MC-Dropout ECE


# ── SE-ResNet ──────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch,ch//r), nn.ReLU(True),
            nn.Linear(ch//r,ch), nn.Sigmoid())
    def forward(self, x): return x * self.se(x).view(x.size(0),-1,1,1)

class SEResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch),
            nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch))
        self.se = SEBlock(ch); self.act = nn.GELU()
    def forward(self, x): return self.act(self.se(self.block(x)) + x)

class SEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        def dn(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci,co,3,stride=2,padding=1,bias=False),
                nn.BatchNorm2d(co), nn.GELU())
        self.stem = nn.Sequential(
            nn.Conv2d(1,32,(7,15),stride=(2,4),padding=(3,7),bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((3,5),stride=(2,2),padding=(1,2)))
        self.body = nn.Sequential(
            SEResBlock(32),SEResBlock(32),dn(32,64),
            SEResBlock(64),SEResBlock(64),dn(64,128),
            SEResBlock(128),SEResBlock(128),dn(128,256),
            SEResBlock(256),SEResBlock(256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128,2))
    def forward(self, x): return self.head(self.body(self.stem(x)))


# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════
class QuickDS(torch.utils.data.Dataset):
    def __init__(self, X_f16, y):
        self.X = X_f16
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i].astype(np.float32)), self.y[i]


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════
def get_probs(model, loader, device):
    """Standard softmax probabilities."""
    model.eval(); ps = []
    with torch.no_grad():
        for xb, _ in loader:
            ps.append(F.softmax(model(xb.to(device)).float(),
                                dim=1)[:,1].cpu().numpy())
    return np.concatenate(ps)

def get_mc_probs(model, loader, device, n=MC_SAMPLES):
    """MC-Dropout mean probabilities."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)): m.train()
    ps = []
    with torch.no_grad():
        for xb, _ in loader:
            xb_t = xb.repeat_interleave(n, 0).to(device)
            pr   = F.softmax(model(xb_t).float(), dim=1)[:,1]
            ps.append(pr.view(-1, n).mean(1).cpu().numpy())
    return np.concatenate(ps)

def nll(p_event, labels, eps=1e-7):
    """Binary NLL (lower is better)."""
    p = np.clip(p_event, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

def brier(p_event, labels):
    """Brier score = MSE between probs and binary labels (lower is better)."""
    return float(np.mean((p_event - labels) ** 2))

def ece(p_event, labels, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        mask = (p_event >= bins[i]) & (
            p_event <= bins[i+1] if i == n_bins-1 else p_event < bins[i+1])
        if mask.sum() > 0:
            e += mask.sum()/len(labels) * abs(labels[mask].mean() - p_event[mask].mean())
    return float(e)

def mce(p_event, labels, n_bins=10):
    """Maximum Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1); max_e = 0.0
    for i in range(n_bins):
        mask = (p_event >= bins[i]) & (
            p_event <= bins[i+1] if i == n_bins-1 else p_event < bins[i+1])
        if mask.sum() > 0:
            diff = abs(labels[mask].mean() - p_event[mask].mean())
            max_e = max(max_e, diff)
    return float(max_e)

def uncertainty_auroc(p_event, labels):
    """
    AUROC for using uncertainty (|p - 0.5|, inverted) to detect errors.
    High uncertainty (p near 0.5) should correlate with misclassifications.
    """
    pred   = (p_event >= 0.5).astype(int)
    errors = (pred != labels).astype(int)  # 1 = error, 0 = correct
    if errors.sum() == 0 or errors.sum() == len(errors):
        return float("nan")
    uncertainty = 1 - 2 * np.abs(p_event - 0.5)  # high when p near 0.5
    try:
        return float(roc_auc_score(errors, uncertainty))
    except Exception:
        return float("nan")

def all_metrics(p_event, labels):
    return {
        "nll":        nll(p_event, labels),
        "brier":      brier(p_event, labels),
        "ece":        ece(p_event, labels),
        "mce":        mce(p_event, labels),
        "unc_auroc":  uncertainty_auroc(p_event, labels),
    }


# ══════════════════════════════════════════════════════════════
# RELIABILITY DIAGRAM
# ══════════════════════════════════════════════════════════════
def reliability_bins(p_event, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    conf_means, acc_means, counts = [], [], []
    for i in range(n_bins):
        mask = (p_event >= bins[i]) & (
            p_event <= bins[i+1] if i == n_bins-1 else p_event < bins[i+1])
        if mask.sum() > 0:
            conf_means.append(p_event[mask].mean())
            acc_means.append(labels[mask].mean())
            counts.append(mask.sum())
        else:
            conf_means.append((bins[i]+bins[i+1])/2)
            acc_means.append(float("nan"))
            counts.append(0)
    return np.array(conf_means), np.array(acc_means), np.array(counts)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
print("="*65)
print("Extended Calibration Metrics: NLL, Brier, ECE, MCE, AUROC")
print("="*65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\nLoading Utah FORGE 3-2417...")
X2 = np.load(NPY_X2).astype(np.float16)
y2 = np.load(NPY_Y2)
print(f"  {X2.shape}  events={(y2==1).sum()}  noise={(y2==0).sum()}")

ds     = QuickDS(X2, y2)
loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                    num_workers=0, pin_memory=False)

# Models to evaluate: baseline (lam=0) and URDI lam=10
CONFIGS = [
    ("Baseline (CE)", "urdi_ms_lam0p0_s{}.pth"),
    ("URDI λ=0.01",   "urdi_ms_lam0p01_s{}.pth"),
    ("URDI λ=0.1",    "urdi_ms_lam0p1_s{}.pth"),
    ("URDI λ=1.0",    "urdi_ms_lam1p0_s{}.pth"),
    ("URDI λ=10",     "urdi_ms_lam10p0_s{}.pth"),
]

all_rows    = []
rdiag_data  = {}   # for reliability diagram

for label, weight_pattern in CONFIGS:
    print(f"\n{'='*50}")
    print(f"Config: {label}")
    print(f"{'='*50}")

    seed_metrics = []

    for seed in SEEDS:
        wpath = MODEL_DIR / weight_pattern.format(seed)
        if not wpath.exists():
            print(f"  seed={seed}: {wpath.name} not found — skip")
            continue

        model = SEResNet().to(device)
        try:
            model.load_state_dict(torch.load(str(wpath), map_location=device,
                                              weights_only=True))
        except Exception as e:
            print(f"  seed={seed}: load error {e}")
            del model; continue

        # Softmax probs
        p_soft = get_probs(model, loader, device)
        m_soft = all_metrics(p_soft, y2)

        # MC-Dropout probs
        p_mc   = get_mc_probs(model, loader, device)
        m_mc   = all_metrics(p_mc, y2)

        print(f"  seed={seed}:")
        print(f"    Softmax: NLL={m_soft['nll']:.4f}  "
              f"Brier={m_soft['brier']:.4f}  ECE={m_soft['ece']:.4f}  "
              f"MCE={m_soft['mce']:.4f}  UncAUROC={m_soft['unc_auroc']:.4f}")
        print(f"    MC-Drop: NLL={m_mc['nll']:.4f}  "
              f"Brier={m_mc['brier']:.4f}  ECE={m_mc['ece']:.4f}  "
              f"MCE={m_mc['mce']:.4f}  UncAUROC={m_mc['unc_auroc']:.4f}")

        seed_metrics.append({"seed": seed,
                              "soft": m_soft, "mc": m_mc,
                              "p_soft": p_soft, "p_mc": p_mc})

        del model; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    if not seed_metrics:
        continue

    # Aggregate across seeds
    for mode in ("soft", "mc"):
        for metric in ("nll", "brier", "ece", "mce", "unc_auroc"):
            vals = [s[mode][metric] for s in seed_metrics
                    if not np.isnan(s[mode][metric])]
            all_rows.append({
                "config":  label,
                "mode":    "Softmax" if mode == "soft" else "MC-Dropout",
                "metric":  metric,
                "mean":    float(np.mean(vals)),
                "std":     float(np.std(vals)),
                "n_seeds": len(vals),
            })

    # Store mean reliability diagram data (softmax)
    p_mean = np.mean([s["p_soft"] for s in seed_metrics], axis=0)
    conf_b, acc_b, cnt_b = reliability_bins(p_mean, y2)
    rdiag_data[label] = (conf_b, acc_b, cnt_b)


# ── Summary table ──────────────────────────────────────────────
df = pd.DataFrame(all_rows)
df.to_csv(DATA_DIR / "calibration_metrics_full.csv", index=False)

# Pivot for clean display
pivot_soft = df[df["mode"] == "Softmax"].pivot_table(
    index="config", columns="metric",
    values=["mean", "std"], aggfunc="first")
pivot_mc   = df[df["mode"] == "MC-Dropout"].pivot_table(
    index="config", columns="metric",
    values=["mean", "std"], aggfunc="first")

print("\n" + "="*80)
print("EXTENDED CALIBRATION METRICS — SOFTMAX (mean +/- std, 5 seeds)")
print("="*80)
for config in [c for c, _ in CONFIGS]:
    sub = df[(df["config"]==config) & (df["mode"]=="Softmax")]
    if len(sub) == 0: continue
    row = {r["metric"]: (r["mean"], r["std"]) for _, r in sub.iterrows()}
    print(f"\n  {config}")
    for m in ("nll", "brier", "ece", "mce", "unc_auroc"):
        if m in row:
            print(f"    {m.upper():<12}: {row[m][0]:.4f} +/- {row[m][1]:.4f}")


# ── Figure: reliability diagrams + metric comparison ──────────
print("\nGenerating figure...")
configs_plot = [c for c, _ in CONFIGS if c in rdiag_data]
n_configs    = len(configs_plot)

fig = plt.figure(figsize=(16, 10))
# Top row: reliability diagrams
# Bottom row: metric bar charts

from matplotlib.gridspec import GridSpec
gs = GridSpec(2, max(n_configs, 3), figure=fig,
              hspace=0.45, wspace=0.35)

colors = ["#B4B2A9","#1D9E75","#7F77DD","#EF9F27","#E24B4A"]

# Reliability diagrams (top row)
for ci, config in enumerate(configs_plot):
    ax = fig.add_subplot(gs[0, ci])
    conf_b, acc_b, _ = rdiag_data[config]
    valid = ~np.isnan(acc_b)
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Perfect", alpha=0.5)
    ax.bar(conf_b[valid], acc_b[valid], width=0.08,
           color=colors[ci], alpha=0.7, edgecolor="white")
    ax.plot(conf_b[valid], acc_b[valid], "o-",
            color=colors[ci], markersize=4, linewidth=1.5)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence", fontsize=8)
    if ci == 0: ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_title(config, fontsize=8, fontweight="bold",
                 color=colors[ci])
    ax.tick_params(labelsize=7)

    # Add ECE text
    sub = df[(df["config"]==config) & (df["mode"]=="Softmax") &
             (df["metric"]=="ece")]
    if len(sub) > 0:
        ax.text(0.05, 0.92,
                f"ECE={sub['mean'].values[0]:.3f}",
                transform=ax.transAxes, fontsize=7,
                color=colors[ci], fontweight="bold")

# Bottom row: NLL, Brier, MCE bars
metrics_plot = [("nll","NLL\n(lower=better)"),
                ("brier","Brier Score\n(lower=better)"),
                ("mce","MCE\n(lower=better)")]

for mi, (metric, title) in enumerate(metrics_plot):
    ax = fig.add_subplot(gs[1, mi])
    sub = df[(df["mode"]=="Softmax") & (df["metric"]==metric)]
    x   = np.arange(len(configs_plot))
    vals= []; errs = []
    for c in configs_plot:
        r = sub[sub["config"]==c]
        vals.append(r["mean"].values[0] if len(r)>0 else 0)
        errs.append(r["std"].values[0]  if len(r)>0 else 0)

    bars = ax.bar(x, vals, color=colors[:len(configs_plot)],
                  edgecolor="white", alpha=0.9)
    ax.errorbar(x, vals, yerr=errs, fmt="none",
                color="black", capsize=4, linewidth=1.5)
    base_val = vals[0]
    ax.axhline(base_val, color="#888", linestyle="--",
               linewidth=1, label=f"Baseline={base_val:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ","\\n") for c in configs_plot],
                       rotation=30, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                v+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.02,
                f"{v:.3f}", ha="center", fontsize=7)

fig.suptitle("Extended Calibration Analysis — Utah FORGE 3-2417 (Zero-Shot)\n"
             "Top: Reliability diagrams (mean over 5 seeds)  "
             "Bottom: NLL, Brier, MCE metrics",
             fontsize=10, fontweight="bold")

plt.savefig(FIG_DIR/"paper_calibration_extended.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: data/figures/paper_calibration_extended.png")


# ── Text report ────────────────────────────────────────────────
with open(DATA_DIR/"calibration_metrics_report.txt","w",
          encoding="utf-8") as f:
    f.write("EXTENDED CALIBRATION METRICS\n"+"="*70+"\n\n")
    f.write("Dataset: Utah FORGE 3-2417 (zero-shot)\n")
    f.write(f"Seeds  : {SEEDS}\n\n")
    f.write("Metrics:\n")
    f.write("  NLL       : Negative Log-Likelihood (lower = better)\n")
    f.write("  Brier     : Mean squared prob error (lower = better)\n")
    f.write("  ECE       : Expected Calibration Error (lower = better)\n")
    f.write("  MCE       : Maximum Calibration Error (lower = better)\n")
    f.write("  UncAUROC  : AUROC for uncertainty predicting errors (higher = better)\n\n")

    for mode_label in ("Softmax", "MC-Dropout"):
        f.write(f"\n--- {mode_label} ---\n")
        f.write(f"{'Config':<20} {'NLL':>10} {'Brier':>10} "
                f"{'ECE':>10} {'MCE':>10} {'UncAUROC':>12}\n")
        f.write("-"*74+"\n")
        for config, _ in CONFIGS:
            sub = df[(df["config"]==config) & (df["mode"]==mode_label)]
            if len(sub) == 0: continue
            row = {r["metric"]: (r["mean"],r["std"]) for _,r in sub.iterrows()}
            def fmt(m):
                if m in row: return f"{row[m][0]:>6.4f}+/-{row[m][1]:.4f}"
                return "    ---     "
            f.write(f"  {config:<18} {fmt('nll'):>14} {fmt('brier'):>14} "
                    f"{fmt('ece'):>14} {fmt('mce'):>14} {fmt('unc_auroc'):>14}\n")

print("  Saved: data/calibration_metrics_report.txt")
print("\n"+"="*65)
print("Done. Share data/calibration_metrics_report.txt")
print("="*65)
