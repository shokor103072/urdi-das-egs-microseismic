"""
Phase 1 — Perception Module (v2 — Corrected)
Agentic DAS Microseismic Monitoring System

ROOT CAUSE FIX:
  SE-ResNet was trained on 2400-timestep windows (full X.npy rows).
  Previous version incorrectly fed 256-step sub-windows → 37% accuracy.
  This version feeds full 2400-step windows → expected ~97% accuracy.

Input:
  Stage 14: E:\\Events_14\\Events_14\\Data\\Dataset\\X.npy   (35766, 1, 361, 2400)
            E:\\Events_14\\Events_14\\Data\\Dataset\\y.npy   (3974,) one label per trigger
            Labels expanded: trigger i → rows [i*9 : (i+1)*9] share label y[i]

  Stage 2:  data/stage2_X_segmented.npy  (18144, 1, 361, 256)
            NOTE: Stage 2 has 2400-step windows too — use X_forge.npy directly.
            data/stage2_event_log.pkl has source_sample index for alignment.

Output saved to data/:
  stage14_perception.pkl   (35766 rows) — one row per X14 row
  stage2_perception.pkl    (2016 rows)  — one row per X_forge row
  perception_report.txt
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = Path(r"E:\Events_14\Events_14\Data")
MODEL_PATH = BASE_DIR / "Model" / "best_seresnet.pth"
NPY_X14    = BASE_DIR / "Dataset" / "X.npy"
NPY_Y14    = BASE_DIR / "Dataset" / "y.npy"
NPY_X2     = BASE_DIR / "Dataset" / "X_forge.npy"
NPY_Y2     = BASE_DIR / "Dataset" / "y_forge.npy"
DATA_DIR   = Path("./data")

MC_SAMPLES    = 20
BATCH_SIZE    = 32   # standard inference batch size
MC_BATCH_SIZE = 2    # MC-Dropout batch size (tiled x20, so 2x20=40 on GPU at once)


# ═══════════════════════════════════════════════════════════════
# SE-RESNET ARCHITECTURE  (exact copy from train_comparison.py)
# ═══════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // ratio), nn.ReLU(inplace=True),
            nn.Linear(ch // ratio, ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class SEResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.se  = SEBlock(ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.block(x)) + x)


class SEResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, (7, 15), stride=(2, 4), padding=(3, 7), bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((3, 5), stride=(2, 2), padding=(1, 2)),
        )
        self.layer1 = nn.Sequential(SEResBlock(32), SEResBlock(32))
        self.down1  = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64), nn.GELU())
        self.layer2 = nn.Sequential(SEResBlock(64), SEResBlock(64))
        self.down2  = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(128), nn.GELU())
        self.layer3 = nn.Sequential(SEResBlock(128), SEResBlock(128))
        self.down3  = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(256), nn.GELU())
        self.layer4 = nn.Sequential(SEResBlock(256), SEResBlock(256))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.down1(x)
        x = self.layer2(x); x = self.down2(x)
        x = self.layer3(x); x = self.down3(x)
        x = self.layer4(x)
        return self.head(self.pool(x))


def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════
def standard_inference(model, loader, device):
    model.eval()
    all_p, all_pred = [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
            probs = F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            preds = logits.float().argmax(dim=1).cpu().numpy()
            all_p.append(probs)
            all_pred.append(preds)
    return np.concatenate(all_p), np.concatenate(all_pred)


def mc_dropout_inference(model, loader, device, n_samples=20):
    """
    Batched MC-Dropout: tile each batch n_samples times along batch dim.
    Single forward pass per batch instead of n_samples loops.
    """
    enable_mc_dropout(model)
    all_mean, all_std = [], []
    with torch.no_grad():
        for (xb,) in loader:
            # (B, 1, C, T) -> (B*n, 1, C, T)
            xb_tiled = xb.repeat_interleave(n_samples, dim=0).to(device)
            logits   = model(xb_tiled)
            probs    = F.softmax(logits.float(), dim=1)[:, 1]
            probs    = probs.view(-1, n_samples)   # (B, n_samples)
            all_mean.append(probs.mean(dim=1).cpu().numpy())
            all_std.append(probs.std(dim=1).cpu().numpy())
    return np.concatenate(all_mean), np.concatenate(all_std)


def run_perception(name, X, y, model, device):
    """Run full perception pipeline on pre-loaded X, y arrays."""
    print(f"\n  Input shape : {X.shape}  Labels: {y.shape}")
    print(f"  Label dist  : {{1: {(y==1).sum()}, 0: {(y==0).sum()}}}")

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

    print(f"  Standard inference...")
    p_event, pred_label = standard_inference(model, loader, device)

    # Separate smaller loader for MC-Dropout to avoid GPU OOM
    mc_loader = DataLoader(dataset, batch_size=MC_BATCH_SIZE,
                           shuffle=False, num_workers=0)
    print(f"  MC-Dropout inference ({MC_SAMPLES} passes, batch={MC_BATCH_SIZE}×{MC_SAMPLES}={MC_BATCH_SIZE*MC_SAMPLES})...")
    mc_mean, mc_std = mc_dropout_inference(model, mc_loader, device, MC_SAMPLES)

    confidence = 1.0 - mc_std

    df = pd.DataFrame({
        "label":      y.astype(int),
        "label_name": ["event" if v == 1 else "noise" for v in y],
        "p_event":    p_event,
        "pred_label": pred_label.astype(int),
        "mc_mean":    mc_mean,
        "mc_std":     mc_std,
        "confidence": confidence,
        "correct":    (pred_label == y).astype(int),
    })

    acc = df["correct"].mean()
    n_unc = (df["mc_std"] > 0.15).sum()
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Mean confidence  : {confidence.mean():.4f}")
    print(f"  High uncertainty : {n_unc} windows (mc_std > 0.15)")

    return df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1 — Perception Module  (v2 corrected)")
print("=" * 60)

print(f"\nPyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("\n  [WARNING] CUDA not detected — running on CPU.")
    print("  Fix: pip install torch torchvision "
          "--index-url https://download.pytorch.org/whl/cu121 "
          "--force-reinstall --no-deps")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")
if device.type == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")

# Load model
print(f"\nLoading SE-ResNet from: {MODEL_PATH}")
model = SEResNet(num_classes=2).to(device)
state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
print("  Loaded OK")


# ── Stage 14 ──────────────────────────────────────────────────
print("\n── Stage 14 ─────────────────────────────────────────────")
print("Loading X14 (full 2400-step windows)...")
X14 = np.load(NPY_X14)          # (35766, 1, 361, 2400)
y14_trigger = np.load(NPY_Y14)  # (3974,) one per trigger

# Expand labels: each trigger has 9 consecutive X rows
n_rows14   = X14.shape[0]
n_trig14   = len(y14_trigger)
rows_per_t = n_rows14 // n_trig14   # = 9
y14 = np.repeat(y14_trigger, rows_per_t)   # (35766,)
print(f"  Labels expanded: {n_trig14} triggers × {rows_per_t} = {len(y14)} rows")

# Merge event log (chronological ordering) with X14 row indices
df_log14 = pd.read_pickle(DATA_DIR / "stage14_event_log.pkl")

df14 = run_perception("Stage 14", X14, y14, model, device)

# Attach trigger metadata from event log
# event log npy_index maps to original X14 row before chronological reorder
# Here we run on X14 in its native order, so index = X14 row directly
df14["npy_index"] = np.arange(len(df14))

# Save
df14.to_pickle(DATA_DIR / "stage14_perception.pkl")
print(f"  Saved -> data/stage14_perception.pkl  ({len(df14)} rows)")
del X14


# ── Stage 2 ───────────────────────────────────────────────────
print("\n── Stage 2 (FORGE) ──────────────────────────────────────")
print("Loading X_forge (full 2400-step windows)...")
X2 = np.load(NPY_X2)   # (2016, 1, 361, 2400)
y2 = np.load(NPY_Y2)   # (2016,)

df2 = run_perception("Stage 2 (FORGE)", X2, y2, model, device)
df2["source_sample"] = np.arange(len(df2))

df2.to_pickle(DATA_DIR / "stage2_perception.pkl")
print(f"  Saved -> data/stage2_perception.pkl  ({len(df2)} rows)")
del X2


# ── Report ────────────────────────────────────────────────────
report = DATA_DIR / "perception_report.txt"
with open(report, "w") as f:
    f.write("PHASE 1 PERCEPTION REPORT (v2 — full 2400-step inputs)\n")
    f.write("=" * 60 + "\n\n")
    for stage_name, df in [("Stage 14", df14), ("Stage 2 (FORGE)", df2)]:
        tp = int(((df["pred_label"]==1) & (df["label"]==1)).sum())
        tn = int(((df["pred_label"]==0) & (df["label"]==0)).sum())
        fp = int(((df["pred_label"]==1) & (df["label"]==0)).sum())
        fn = int(((df["pred_label"]==0) & (df["label"]==1)).sum())
        acc    = (tp+tn)/len(df)
        prec   = tp/(tp+fp) if (tp+fp) > 0 else 0
        recall = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1     = 2*prec*recall/(prec+recall) if (prec+recall) > 0 else 0
        far    = fp/(fp+tn) if (fp+tn) > 0 else 0

        f.write(f"{stage_name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total windows         : {len(df)}\n")
        f.write(f"Accuracy              : {acc:.4f}\n")
        f.write(f"F1 Score              : {f1:.4f}\n")
        f.write(f"Precision             : {prec:.4f}\n")
        f.write(f"Recall                : {recall:.4f}\n")
        f.write(f"False Alarm Rate      : {far:.4f}\n")
        f.write(f"Confusion matrix      : TP={tp} TN={tn} FP={fp} FN={fn}\n")
        f.write(f"Mean p_event          : {df['p_event'].mean():.4f}\n")
        f.write(f"Mean MC-Dropout mean  : {df['mc_mean'].mean():.4f}\n")
        f.write(f"Mean MC-Dropout std   : {df['mc_std'].mean():.4f}\n")
        f.write(f"Mean confidence       : {df['confidence'].mean():.4f}\n")
        f.write(f"High unc windows      : {(df['mc_std']>0.15).sum()} "
                f"(mc_std > 0.15)\n\n")

    f.write("Output columns:\n")
    f.write("  label       — ground truth (0/1)\n")
    f.write("  p_event     — standard softmax P(event)\n")
    f.write("  pred_label  — argmax prediction\n")
    f.write("  mc_mean     — MC-Dropout mean P(event)\n")
    f.write("  mc_std      — epistemic uncertainty\n")
    f.write("  confidence  — 1 - mc_std\n")
    f.write("  correct     — 1 if pred_label == label\n")

print(f"  Saved -> data/perception_report.txt")
print("\n" + "=" * 60)
print("Phase 1 complete.")
print("Next -> Phase 2: state_tracker.py")
print("=" * 60)
