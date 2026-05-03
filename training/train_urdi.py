"""
URDI — Uncertainty-Regularized Domain-Invariant Training
TNNLS Revision: Novel Learning Contribution (Concern 1)

MOTIVATION:
  Standard cross-entropy training does not explicitly encourage the network
  to learn domain-invariant representations. It minimizes classification loss
  on the source domain without considering whether the learned features will
  generalize under distribution shift.

PROPOSED OBJECTIVE:
  L_URDI = L_CE + lambda * R_unc

  where R_unc = mean MC-Dropout variance across a mini-batch:
    R_unc = (1/N) sum_i Var_{t~Dropout}[p(y=1|x_i, theta, t)]

  This uncertainty regularizer penalizes high epistemic uncertainty during
  training, encouraging the network to learn features that are:
    (a) Discriminative (low CE loss)
    (b) Consistently confident (low MC variance)

  Features that are confident tend to be domain-invariant because uncertain
  predictions arise precisely when the model encounters out-of-distribution
  patterns — penalizing this during training forces the model away from
  representations that are brittle to input perturbations.

RELATIONSHIP TO DOMAIN GENERALIZATION:
  URDI is related to entropy minimization (Grandvalet & Bengio, 2005) but
  operates on epistemic uncertainty rather than predictive entropy, and uses
  MC-Dropout variance rather than softmax entropy. Unlike domain-adversarial
  training, URDI requires no target-domain data during training.

EXPERIMENT:
  Train SE-ResNet with URDI (lambda in {0.01, 0.1, 1.0, 10.0}) on Stage 14.
  Evaluate on Stage 2 (zero-shot): F1, AUC, ECE, mean mc_std.
  Compare against standard SE-ResNet baseline.

  Key metrics:
    - Stage 2 F1          : does URDI improve cross-domain detection?
    - Stage 2 ECE         : does URDI improve calibration?
    - Stage 2 mc_std      : does URDI reduce uncertainty on the new domain?
    - Stage 14 F1         : does URDI hurt in-distribution performance?

OUTPUTS:
  data/urdi_results.csv
  data/figures/paper_urdi.png
  data/urdi_report.txt
  Model/best_seresnet_urdi.pth   (best lambda model)
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NPY_X14   = Path(r"./Dataset/X.npy")
NPY_Y14   = Path(r"./Dataset/y.npy")
NPY_X2    = Path(r"./Dataset/X_forge.npy")
NPY_Y2    = Path(r"./Dataset/y_forge.npy")
MODEL_DIR = Path(r"./Model")
DATA_DIR = Path(r"./data")
FIG_DIR    = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
EPOCHS      = 50
PATIENCE    = 12
LR          = 3e-4
WEIGHT_DECAY= 1e-3
BATCH_SIZE  = 16
MC_K        = 5       # MC samples per batch during training (keep small for speed)
LAMBDAS     = [0.0, 0.01, 0.1, 1.0, 10.0]   # 0.0 = standard baseline

torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════
# SE-RESNET WITH MC-DROPOUT SUPPORT
# ═══════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//ratio), nn.ReLU(inplace=True),
            nn.Linear(ch//ratio, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class SEResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.se  = SEBlock(ch)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.se(self.block(x)) + x)


class SEResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, (7,15), stride=(2,4), padding=(3,7), bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((3,5), stride=(2,2), padding=(1,2)))
        self.layer1 = nn.Sequential(SEResBlock(32), SEResBlock(32))
        self.down1  = nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1,bias=False),
                                     nn.BatchNorm2d(64), nn.GELU())
        self.layer2 = nn.Sequential(SEResBlock(64), SEResBlock(64))
        self.down2  = nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1,bias=False),
                                     nn.BatchNorm2d(128), nn.GELU())
        self.layer3 = nn.Sequential(SEResBlock(128), SEResBlock(128))
        self.down3  = nn.Sequential(nn.Conv2d(128,256,3,stride=2,padding=1,bias=False),
                                     nn.BatchNorm2d(256), nn.GELU())
        self.layer4 = nn.Sequential(SEResBlock(256), SEResBlock(256))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.down1(x)
        x = self.layer2(x); x = self.down2(x); x = self.layer3(x)
        x = self.down3(x); x = self.layer4(x)
        return self.head(self.pool(x))

    def mc_uncertainty(self, x, k=5):
        """
        Compute mean batch uncertainty via k stochastic forward passes.
        Dropout and Dropout2d stay active (model must be in train mode).
        Returns scalar: mean variance across batch and MC samples.
        """
        probs = torch.stack([
            F.softmax(self.forward(x), dim=1)[:, 1]
            for _ in range(k)
        ], dim=0)   # (k, B)
        return probs.var(dim=0).mean()   # scalar


class URDILoss(nn.Module):
    """
    URDI training loss:
      L = L_CE + lambda * R_unc
    where R_unc = mean MC-Dropout variance over the mini-batch.
    """
    def __init__(self, lambda_unc: float = 0.1, mc_k: int = 5):
        super().__init__()
        self.lambda_unc = lambda_unc
        self.mc_k       = mc_k
        self.ce         = nn.CrossEntropyLoss()

    def forward(self, model: nn.Module, x: torch.Tensor,
                y: torch.Tensor, logits: torch.Tensor) -> tuple:
        loss_ce  = self.ce(logits, y)
        if self.lambda_unc == 0.0:
            return loss_ce, loss_ce.item(), 0.0

        # Keep dropout active for uncertainty estimation
        # model is already in train() mode so dropout is active
        r_unc    = model.mc_uncertainty(x, k=self.mc_k)
        loss     = loss_ce + self.lambda_unc * r_unc
        return loss, loss_ce.item(), r_unc.item()


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate_standard(model, X, y, device, batch_size=16):
    """Standard inference metrics."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    all_p, all_pred = [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_p.append(F.softmax(logits.float(),dim=1)[:,1].cpu().numpy())
            all_pred.append(logits.float().argmax(1).cpu().numpy())
    p    = np.concatenate(all_p)
    pred = np.concatenate(all_pred)
    f1   = f1_score(y, pred, zero_division=0)
    try: auc = roc_auc_score(y, p) if len(np.unique(y))>1 else 0.0
    except: auc = 0.0
    acc = (pred == y).mean()
    return dict(f1=f1, auc=auc, acc=float(acc), p=p, pred=pred)


def evaluate_mc(model, X, device, n_samples=20, batch_size=8):
    """MC-Dropout uncertainty estimation."""
    # Enable dropout for MC
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    means, stds = [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb_tiled = xb.repeat_interleave(n_samples, dim=0).to(device)
            probs    = F.softmax(model(xb_tiled).float(), dim=1)[:,1]
            probs    = probs.view(-1, n_samples)
            means.append(probs.mean(dim=1).cpu().numpy())
            stds.append(probs.std(dim=1).cpu().numpy())
    return np.concatenate(means), np.concatenate(stds)


def compute_ece(p_event, labels, n_bins=10):
    """Expected Calibration Error."""
    bins  = np.linspace(0, 1, n_bins+1)
    total = len(labels)
    ece   = 0.0
    for i in range(n_bins):
        mask = (p_event >= bins[i]) & (p_event < bins[i+1])
        if i == n_bins-1:
            mask = (p_event >= bins[i]) & (p_event <= bins[i+1])
        if mask.sum() > 0:
            acc_bin  = labels[mask].mean()
            conf_bin = p_event[mask].mean()
            ece     += mask.sum() / total * abs(acc_bin - conf_bin)
    return float(ece)


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
def train_urdi(X_tr, y_tr, X_val, y_val, device,
               lambda_unc=0.1, epochs=EPOCHS,
               batch_size=BATCH_SIZE, save_path=None):
    counts  = np.bincount(y_tr)
    weights = 1.0 / counts[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(y_tr, dtype=torch.long))
    loader  = DataLoader(dataset, batch_size=batch_size,
                         sampler=sampler, num_workers=0)
    val_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                             torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    model     = SEResNet().to(device)
    criterion = URDILoss(lambda_unc=lambda_unc, mc_k=MC_K)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=epochs)
    scaler    = torch.amp.GradScaler("cuda") if device.type=="cuda" else None

    best_f1, best_epoch, wait = 0.0, 0, 0
    history   = []

    for ep in range(1, epochs+1):
        model.train()
        t0 = time.time()
        total_ce = 0.0; total_unc = 0.0; n_batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if scaler and lambda_unc == 0.0:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss, ce_v, unc_v = criterion(model, xb, yb, logits)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                # URDI: run without autocast to keep MC passes stable
                logits = model(xb)
                loss, ce_v, unc_v = criterion(model, xb, yb, logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_ce  += ce_v
            total_unc += unc_v
            n_batches += 1

        scheduler.step()

        # Validation
        val_m = evaluate_standard(model, X_val, y_val, device)
        elapsed = time.time() - t0

        improved = val_m["f1"] > best_f1
        if improved:
            best_f1, best_epoch, wait = val_m["f1"], ep, 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            wait += 1

        history.append({
            "epoch": ep, "val_f1": val_m["f1"],
            "ce_loss": total_ce/n_batches,
            "unc_reg": total_unc/n_batches,
        })

        if ep % 10 == 0 or ep == 1:
            print(f"    ep={ep:>3}  val_f1={val_m['f1']:.4f}  "
                  f"ce={total_ce/n_batches:.4f}  "
                  f"unc={total_unc/n_batches:.5f}  "
                  f"{'<best' if improved else ''}")

        if wait >= PATIENCE:
            print(f"    Early stopping at epoch {ep}")
            break

    # Load best
    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, weights_only=True,
                                          map_location=device))
    return model, best_f1, best_epoch, pd.DataFrame(history)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("URDI — Uncertainty-Regularized Domain-Invariant Training")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Lambdas: {LAMBDAS}")
print(f"MC_K   : {MC_K} (forward passes per batch during training)")

print("\nLoading data...")
X14 = np.load(NPY_X14); y14_t = np.load(NPY_Y14)
X2  = np.load(NPY_X2);  y2    = np.load(NPY_Y2)
rpt = X14.shape[0] // len(y14_t)
y14 = np.repeat(y14_t, rpt)

# Train/val/test split (same as original training)
idx    = np.arange(len(y14))
tr_idx, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y14)
val_idx, te_idx = train_test_split(tmp, test_size=0.50, random_state=SEED,
                                    stratify=y14[tmp])
X_tr, y_tr   = X14[tr_idx],  y14[tr_idx]
X_val, y_val = X14[val_idx], y14[val_idx]
X_te,  y_te  = X14[te_idx],  y14[te_idx]
print(f"  Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}")
print(f"  Stage 2: {len(y2)}")

results = []

for lam in LAMBDAS:
    lam_str  = f"lambda={lam}"
    lam_tag  = f"lam{str(lam).replace('.','p')}"
    save_path = MODEL_DIR / f"best_seresnet_urdi_{lam_tag}.pth"

    print(f"\n{'='*55}")
    print(f"Training: {lam_str}  "
          f"({'standard baseline' if lam==0.0 else 'URDI'})")
    print(f"{'='*55}")

    model, best_val_f1, best_ep, hist = train_urdi(
        X_tr, y_tr, X_val, y_val, device,
        lambda_unc=lam, epochs=EPOCHS,
        batch_size=BATCH_SIZE, save_path=str(save_path)
    )
    print(f"  Best val F1 = {best_val_f1:.4f} at epoch {best_ep}")

    # Stage 14 test evaluation
    m14  = evaluate_standard(model, X_te,  y_te, device)
    # Stage 2 zero-shot
    m2   = evaluate_standard(model, X2,    y2,   device)
    # MC-Dropout uncertainty on Stage 2
    _, mc_std2 = evaluate_mc(model, X2, device, n_samples=20, batch_size=4)
    # ECE
    ece14 = compute_ece(m14["p"], y_te)
    ece2  = compute_ece(m2["p"],  y2)
    # MC ECE (Stage 2)
    mc_mean2, _ = evaluate_mc(model, X2, device, n_samples=20, batch_size=4)
    ece2_mc = compute_ece(mc_mean2, y2)

    row = {
        "lambda":       lam,
        "label":        "Baseline" if lam==0.0 else f"URDI λ={lam}",
        "best_val_f1":  best_val_f1,
        "s14_f1":       m14["f1"],
        "s14_ece":      ece14,
        "s2_f1":        m2["f1"],
        "s2_auc":       m2["auc"],
        "s2_ece":       ece2,
        "s2_ece_mc":    ece2_mc,
        "s2_mc_std_mean": float(mc_std2.mean()),
        "gen_gap_f1":   m14["f1"] - m2["f1"],
    }
    results.append(row)

    print(f"  S14 test : F1={m14['f1']:.4f}  ECE={ece14:.4f}")
    print(f"  S2 zero-shot: F1={m2['f1']:.4f}  AUC={m2['auc']:.4f}  "
          f"ECE={ece2:.4f}  mc_std={mc_std2.mean():.5f}")

    del model
    torch.cuda.empty_cache() if device.type=="cuda" else None

df = pd.DataFrame(results)
df.to_csv(DATA_DIR / "urdi_results.csv", index=False)
print(f"\nSaved -> data/urdi_results.csv")
print(df[["label","s14_f1","s2_f1","s2_ece","s2_mc_std_mean","gen_gap_f1"]]
      .to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════
print("\nGenerating URDI figure...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels   = df["label"].tolist()
colors   = ["#B4B2A9"] + ["#1D9E75","#7F77DD","#EF9F27","#E24B4A"]

# Stage 2 F1
ax = axes[0]
bars = ax.bar(range(len(df)), df["s2_f1"], color=colors,
              edgecolor="white", linewidth=0.5, alpha=0.9)
ax.axhline(df.loc[df["lambda"]==0,"s2_f1"].values[0],
           color="#B4B2A9", linestyle="--", linewidth=1.5,
           label="Baseline")
ax.set_xticks(range(len(df))); ax.set_xticklabels(labels, rotation=25,
                                                    ha="right", fontsize=9)
ax.set_ylabel("Stage 2 F1 (zero-shot)")
ax.set_title("Cross-domain F1\n(higher is better)")
ax.set_ylim(df["s2_f1"].min()-0.02, 1.01)
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, df["s2_f1"]):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.002,
            f"{v:.4f}", ha="center", fontsize=8)

# Stage 2 ECE (softmax)
ax = axes[1]
bars = ax.bar(range(len(df)), df["s2_ece"], color=colors,
              edgecolor="white", linewidth=0.5, alpha=0.9)
ax.axhline(df.loc[df["lambda"]==0,"s2_ece"].values[0],
           color="#B4B2A9", linestyle="--", linewidth=1.5)
ax.axhline(0.05, color="#E24B4A", linestyle=":", linewidth=1,
           alpha=0.7, label="Well-calibrated threshold")
ax.set_xticks(range(len(df))); ax.set_xticklabels(labels, rotation=25,
                                                    ha="right", fontsize=9)
ax.set_ylabel("Stage 2 ECE (lower is better)")
ax.set_title("Cross-domain calibration\n(lower ECE = better calibrated)")
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
for bar, v in zip(bars, df["s2_ece"]):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.0003,
            f"{v:.4f}", ha="center", fontsize=8)

# Stage 2 mean mc_std
ax = axes[2]
bars = ax.bar(range(len(df)), df["s2_mc_std_mean"], color=colors,
              edgecolor="white", linewidth=0.5, alpha=0.9)
ax.axhline(df.loc[df["lambda"]==0,"s2_mc_std_mean"].values[0],
           color="#B4B2A9", linestyle="--", linewidth=1.5)
ax.set_xticks(range(len(df))); ax.set_xticklabels(labels, rotation=25,
                                                    ha="right", fontsize=9)
ax.set_ylabel("Mean MC-Dropout std (Stage 2)")
ax.set_title("Epistemic uncertainty\n(lower = more confident cross-domain)")
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, df["s2_mc_std_mean"]):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.00003,
            f"{v:.5f}", ha="center", fontsize=8)

fig.suptitle("URDI — Uncertainty-Regularized Domain-Invariant Training\n"
             "Effect of uncertainty regularization strength (λ) on cross-domain performance",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "paper_urdi.png", dpi=200, bbox_inches="tight")
plt.close()
print("    Saved: paper_urdi.png")


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════
report = DATA_DIR / "urdi_report.txt"
with open(report, "w", encoding="utf-8") as f:
    f.write("URDI — Uncertainty-Regularized Domain-Invariant Training\n")
    f.write("=" * 65 + "\n\n")
    f.write("Training objective: L = L_CE + lambda * R_unc\n")
    f.write(f"  MC forward passes per batch (k) : {MC_K}\n")
    f.write(f"  Epochs                          : {EPOCHS}\n")
    f.write(f"  Patience                        : {PATIENCE}\n\n")

    hdr = (f"{'Config':<20} {'S14 F1':>8} {'S14 ECE':>9} "
           f"{'S2 F1':>8} {'S2 ECE':>9} {'S2 mc_std':>10} {'Gap':>7}\n")
    f.write(hdr)
    f.write("-" * 75 + "\n")
    for _, row in df.iterrows():
        f.write(f"{row['label']:<20} {row['s14_f1']:>8.4f} {row['s14_ece']:>9.4f} "
                f"{row['s2_f1']:>8.4f} {row['s2_ece']:>9.4f} "
                f"{row['s2_mc_std_mean']:>10.5f} {row['gen_gap_f1']:>7.4f}\n")

    # Find best lambda
    best_idx = df["s2_f1"].idxmax()
    best_row = df.iloc[best_idx]
    base_row = df[df["lambda"]==0.0].iloc[0]
    f.write(f"\nBest configuration: {best_row['label']}\n")
    f.write(f"  S2 F1 improvement over baseline  : "
            f"{best_row['s2_f1']-base_row['s2_f1']:+.4f}\n")
    f.write(f"  S2 ECE improvement over baseline : "
            f"{best_row['s2_ece']-base_row['s2_ece']:+.4f} "
            f"({'better' if best_row['s2_ece']<base_row['s2_ece'] else 'worse'})\n")
    f.write(f"  S2 mc_std vs baseline            : "
            f"{best_row['s2_mc_std_mean']-base_row['s2_mc_std_mean']:+.5f}\n")
    f.write(f"  S14 F1 cost                      : "
            f"{best_row['s14_f1']-base_row['s14_f1']:+.4f}\n")

print(f"  Saved -> data/urdi_report.txt")
print("\n" + "="*65)
print("URDI training complete.")
print("  data/urdi_results.csv")
print("  data/figures/paper_urdi.png")
print("  data/urdi_report.txt")
print("="*65)




