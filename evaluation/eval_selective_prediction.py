"""
Selective Prediction / Risk-Coverage Curves
TNNLS: Shows that URDI uncertainty is actionable for selective prediction

CONCEPT:
  A model can "abstain" on high-uncertainty predictions.
  Coverage = fraction of samples the model answers (1 - abstain rate)
  Risk     = error rate on the answered samples

  A good uncertainty model: as coverage decreases (abstain more),
  risk drops faster — uncertainty correctly identifies which samples
  the model will get wrong.

  AUROC of this curve = area under risk-coverage curve (lower = better,
  since we want risk to drop fast as we abstain more).

EXPERIMENT:
  For each URDI lambda:
    - Use uncertainty = 1 - 2*|p - 0.5|  (high when p near 0.5)
    - Sort FORGE samples by uncertainty (ascending = most confident first)
    - Compute error rate on top-k% most confident samples (coverage sweep)
    - Plot risk vs coverage for baseline vs URDI lambda=1 vs URDI lambda=10

  A model whose uncertainty correctly identifies errors will show
  steep risk reduction as coverage decreases.

Outputs:
  data/figures/paper_selective_prediction.png
  data/selective_pred_report.txt
"""

import os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

SEEDS      = [42, 7, 13, 99, 2024]
BATCH      = 32
MC_SAMPLES = 10   # reduced — 10 passes is sufficient for uncertainty ranking

# Configs to plot: baseline and the two best lambdas
PLOT_CONFIGS = [
    ("Baseline (CE)",  0.0,  "#B4B2A9", "--"),
    ("URDI λ=1.0",    1.0,  "#1D9E75", "-"),
    ("URDI λ=10",     10.0, "#E24B4A", "-"),
]


# ── SE-ResNet ──────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch,ch//r), nn.ReLU(True), nn.Linear(ch//r,ch), nn.Sigmoid())
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
        self.stem   = nn.Sequential(
            nn.Conv2d(1,32,(7,15),stride=(2,4),padding=(3,7),bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((3,5),stride=(2,2),padding=(1,2)))
        self.layer1 = nn.Sequential(SEResBlock(32), SEResBlock(32))
        self.down1  = dn(32, 64)
        self.layer2 = nn.Sequential(SEResBlock(64), SEResBlock(64))
        self.down2  = dn(64, 128)
        self.layer3 = nn.Sequential(SEResBlock(128), SEResBlock(128))
        self.down3  = dn(128, 256)
        self.layer4 = nn.Sequential(SEResBlock(256), SEResBlock(256))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128,2))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.down1(x)
        x = self.layer2(x); x = self.down2(x)
        x = self.layer3(x); x = self.down3(x)
        x = self.layer4(x)
        return self.head(x)


# ── Dataset ────────────────────────────────────────────────────
class QuickDS(torch.utils.data.Dataset):
    def __init__(self, X_f16, y):
        self.X=X_f16; self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i].astype(np.float32)), self.y[i]


# ── Weight finder ──────────────────────────────────────────────
def find_weight(lam, seed):
    tag = str(lam).replace(".","p")
    for f in [
        MODEL_DIR / f"urdi_ms_lam{tag}_s{seed}.pth",
        MODEL_DIR / f"urdi_ms_lam{tag}_{seed}.pth",
        MODEL_DIR / f"best_seresnet_urdi_lam{tag}.pth",
    ]:
        if f.exists(): return f
    for f in MODEL_DIR.glob("*.pth"):
        n = f.name.lower()
        if tag.lower() in n and str(seed) in n:
            return f
    return None


# ── Inference ──────────────────────────────────────────────────
def get_mc_uncertainty(model, loader, device, n=MC_SAMPLES):
    """
    Returns (mean_prob, uncertainty, predictions, labels) for all FORGE samples.
    Runs MC passes ONE AT A TIME to avoid GPU OOM.
    uncertainty = 1 - 2*|mean_prob - 0.5|  (high = uncertain = near 0.5)
    """
    model.eval()
    # Enable dropout for MC sampling
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)): m.train()

    # Collect all batches' probs across n passes
    # Shape: (n_passes, n_samples)
    all_pass_probs = []

    for pass_i in range(n):
        pass_probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                p  = F.softmax(model(xb).float(), dim=1)[:, 1]
                pass_probs.append(p.cpu().numpy())
                del xb
        all_pass_probs.append(np.concatenate(pass_probs))

    probs_matrix = np.stack(all_pass_probs, axis=0)  # (n, N)
    mean_p  = probs_matrix.mean(axis=0)               # (N,)
    preds   = (mean_p >= 0.5).astype(int)
    unc     = 1.0 - 2.0 * np.abs(mean_p - 0.5)

    # Labels
    labels = []
    for _, yb in loader:
        labels.append(yb.numpy())
    labels = np.concatenate(labels)

    return mean_p, unc, preds, labels


# ── Risk-coverage curve ────────────────────────────────────────
def risk_coverage_curve(unc, preds, labels, n_thresholds=50):
    """
    Sorts by uncertainty ascending (most confident first).
    For each coverage level c, report error rate on top-c fraction.
    Returns (coverage_array, risk_array).
    """
    # Sort by uncertainty ascending = most confident first
    order = np.argsort(unc)
    preds_sorted  = preds[order]
    labels_sorted = labels[order]

    n = len(labels)
    coverages = np.linspace(0.1, 1.0, n_thresholds)
    risks     = []

    for cov in coverages:
        k       = max(1, int(cov * n))
        # Take the k most confident samples
        p_sub   = preds_sorted[:k]
        l_sub   = labels_sorted[:k]
        error_r = float((p_sub != l_sub).mean())
        risks.append(error_r)

    return coverages, np.array(risks)


def area_under_rc(coverages, risks):
    """Area under risk-coverage curve (trapezoidal). Lower = better."""
    return float(np.trapz(risks, coverages))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
print("="*65)
print("Selective Prediction — Risk-Coverage Curves")
print("="*65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\nLoading Utah FORGE 3-2417...")
X2 = np.load(NPY_X2).astype(np.float16)
y2 = np.load(NPY_Y2)
print(f"  {X2.shape}  ev={(y2==1).sum()}  no={(y2==0).sum()}")

loader = DataLoader(QuickDS(X2, y2), batch_size=BATCH,
                    shuffle=False, num_workers=0, pin_memory=False)

# Collect curves across seeds for each config
results = {}   # label -> list of (coverages, risks) per seed

for label, lam, color, ls in PLOT_CONFIGS:
    print(f"\n{'='*50}\n{label}\n{'='*50}")
    seed_curves = []
    aurc_vals   = []

    for seed in SEEDS:
        wpath = find_weight(lam, seed)
        if wpath is None:
            print(f"  seed={seed}: weight not found — skip")
            continue

        model = SEResNet().to(device)
        try:
            model.load_state_dict(torch.load(str(wpath), map_location=device,
                                              weights_only=True))
        except Exception as e:
            print(f"  seed={seed}: {e}")
            del model; continue

        _, unc, preds, labels_arr = get_mc_uncertainty(model, loader, device)

        full_err = float((preds != labels_arr).mean())
        covs, risks = risk_coverage_curve(unc, preds, labels_arr)
        aurc = area_under_rc(covs, risks)

        print(f"  seed={seed}: full_err={full_err:.4f}  AURC={aurc:.4f}")
        seed_curves.append((covs, risks))
        aurc_vals.append(aurc)

        del model; gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()

    if seed_curves:
        # Mean risk curve across seeds (same coverage grid)
        all_risks = np.array([r for _, r in seed_curves])
        mean_risk = all_risks.mean(axis=0)
        std_risk  = all_risks.std(axis=0)
        covs      = seed_curves[0][0]
        mean_aurc = float(np.mean(aurc_vals))
        std_aurc  = float(np.std(aurc_vals))
        print(f"  Mean AURC = {mean_aurc:.4f} +/- {std_aurc:.4f}")
        results[label] = {
            "color": color, "ls": ls,
            "covs": covs,
            "mean_risk": mean_risk, "std_risk": std_risk,
            "mean_aurc": mean_aurc, "std_aurc": std_aurc,
        }


# ── Figure ─────────────────────────────────────────────────────
print("\nGenerating risk-coverage figure...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: risk-coverage curves
ax = axes[0]
for label, r in results.items():
    covs      = r["covs"]
    mean_risk = r["mean_risk"]
    std_risk  = r["std_risk"]
    ax.plot(covs * 100, mean_risk * 100,
            color=r["color"], linestyle=r["ls"],
            linewidth=2.2, label=f"{label}  (AURC={r['mean_aurc']:.4f})")
    ax.fill_between(covs * 100,
                    (mean_risk - std_risk) * 100,
                    (mean_risk + std_risk) * 100,
                    color=r["color"], alpha=0.12)

# Reference: random uncertainty (flat line at full error rate)
if results:
    first = list(results.values())[0]
    full_err = first["mean_risk"][-1] * 100
    ax.axhline(full_err, color="#AAAAAA", linestyle=":",
               linewidth=1.2, label=f"No abstention (full risk={full_err:.1f}%)")

ax.set_xlabel("Coverage (%)\n(fraction of FORGE samples answered)", fontsize=10)
ax.set_ylabel("Risk (% error rate on answered samples)", fontsize=10)
ax.set_title("Risk-Coverage Curves\nUtah FORGE 3-2417 (zero-shot)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(10, 100)

# Right: AURC bar chart
ax2  = axes[1]
lbls = list(results.keys())
aurcs_m = [results[l]["mean_aurc"] for l in lbls]
aurcs_s = [results[l]["std_aurc"]  for l in lbls]
cols    = [results[l]["color"]     for l in lbls]

bars = ax2.bar(range(len(lbls)), aurcs_m,
               color=cols, edgecolor="white", alpha=0.9, width=0.5)
ax2.errorbar(range(len(lbls)), aurcs_m, yerr=aurcs_s,
             fmt="none", color="black", capsize=6, linewidth=1.8)
ax2.set_xticks(range(len(lbls)))
ax2.set_xticklabels(lbls, rotation=15, ha="right", fontsize=9)
ax2.set_ylabel("AURC (lower = better)", fontsize=10)
ax2.set_title("Area Under Risk-Coverage Curve\nmean +/- std (5 seeds)",
              fontsize=10, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)
for bar, v, s in zip(bars, aurcs_m, aurcs_s):
    ax2.text(bar.get_x() + bar.get_width()/2,
             v + s + 0.001,
             f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Selective Prediction Analysis — URDI\n"
             "Does uncertainty correctly identify which FORGE samples are wrong?",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "paper_selective_prediction.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: data/figures/paper_selective_prediction.png")


# ── Report ─────────────────────────────────────────────────────
with open(DATA_DIR/"selective_pred_report.txt","w",encoding="utf-8") as f:
    f.write("SELECTIVE PREDICTION — RISK-COVERAGE ANALYSIS\n")
    f.write("="*60+"\n\n")
    f.write("Dataset: Utah FORGE 3-2417 (zero-shot, 494 samples)\n")
    f.write(f"Seeds  : {SEEDS}\n")
    f.write(f"MC samples for uncertainty: {MC_SAMPLES}\n\n")
    f.write("Uncertainty measure: 1 - 2*|p_event - 0.5|\n")
    f.write("  High uncertainty = p near 0.5 = model unsure\n\n")
    f.write(f"{'Config':<20} {'AURC mean':>12} {'AURC std':>10} {'Interpretation'}\n")
    f.write("-"*70+"\n")
    for label, r in results.items():
        interp = "worse" if r["mean_aurc"] == max(d["mean_aurc"] for d in results.values()) \
                 else ("best" if r["mean_aurc"] == min(d["mean_aurc"] for d in results.values()) \
                 else "")
        f.write(f"  {label:<18} {r['mean_aurc']:>12.4f} "
                f"{r['std_aurc']:>10.4f}  {interp}\n")
    f.write("\nAURC interpretation:\n")
    f.write("  Lower AURC = uncertainty more accurately identifies errors\n")
    f.write("  = model abstains on the right samples\n")
    f.write("  = uncertainty is more actionable for deployment\n")

print("  Saved: data/selective_pred_report.txt")
print("\n"+"="*65)
print("Done. Share data/selective_pred_report.txt")
print("="*65)
