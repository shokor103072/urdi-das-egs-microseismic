"""
URDI Baselines Comparison — RAM-safe version
TNNLS Concern 4: compare URDI against calibration/regularization baselines

Baselines:
  1. Temperature Scaling    — post-hoc calibration, no retraining
  2. Entropy Regularization — maximize predictive entropy during training
  3. Confidence Penalty     — penalize overconfident predictions
  4. Deep Ensemble (3x)     — 3 independently trained SE-ResNets

RAM fix: mmap loading + NumpyDataset (batches converted, not full array).
Peak RAM ~16 GB instead of ~32 GB.
"""

import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from pathlib import Path
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

NPY_X14   = Path("./Dataset/X.npy")
NPY_Y14   = Path("./Dataset/y.npy")
NPY_X2    = Path("./Dataset/X_forge.npy")
NPY_Y2    = Path("./Dataset/y_forge.npy")
MODEL_DIR = Path("./Model")
DATA_DIR  = Path("./data")
FIG_DIR   = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED           = 42
EPOCHS         = 15
PATIENCE       = 5
LR             = 3e-4
BATCH_SIZE     = 32
N_ENSEMBLE     = 3
ENSEMBLE_SEEDS = [42, 7, 13]

torch.manual_seed(SEED); np.random.seed(SEED)


# ── Architecture ───────────────────────────────────────────────
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
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch),
            nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch))
        self.se = SEBlock(ch); self.act = nn.GELU()
    def forward(self, x): return self.act(self.se(self.block(x)) + x)

class SEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1,32,(7,15),stride=(2,4),padding=(3,7),bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((3,5),stride=(2,2),padding=(1,2)))
        self.layer1 = nn.Sequential(SEResBlock(32), SEResBlock(32))
        self.down1  = nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(64),nn.GELU())
        self.layer2 = nn.Sequential(SEResBlock(64), SEResBlock(64))
        self.down2  = nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(128),nn.GELU())
        self.layer3 = nn.Sequential(SEResBlock(128), SEResBlock(128))
        self.down3  = nn.Sequential(nn.Conv2d(128,256,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(256),nn.GELU())
        self.layer4 = nn.Sequential(SEResBlock(256), SEResBlock(256))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(nn.Flatten(),nn.Dropout(0.5),
            nn.Linear(256,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,2))
    def forward(self, x):
        x=self.stem(x); x=self.layer1(x); x=self.down1(x)
        x=self.layer2(x); x=self.down2(x); x=self.layer3(x)
        x=self.down3(x); x=self.layer4(x)
        return self.head(self.pool(x))


# ── RAM-safe Dataset: converts batches not full array ──────────
class NumpyDataset(Dataset):
    """Wraps a numpy array — only converts to tensor batch-by-batch."""
    def __init__(self, X, y=None):
        self.X = X   # numpy array, may be mmap
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.tensor(np.array(self.X[i]), dtype=torch.float32)
        if self.y is not None:
            return x, torch.tensor(int(self.y[i]), dtype=torch.long)
        return (x,)


# ── Helpers ────────────────────────────────────────────────────
def infer(model, X, device, batch_size=32):
    model.eval()
    loader = DataLoader(NumpyDataset(X), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    ps, preds = [], []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            ps.append(F.softmax(logits.float(),dim=1)[:,1].cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(ps), np.concatenate(preds)

def ece(p, y, n=10):
    bins = np.linspace(0,1,n+1)
    e = 0.0
    for i in range(n):
        m = (p>=bins[i]) & (p<=(bins[i+1] if i==n-1 else bins[i+1]))
        if m.sum()>0:
            e += m.sum()/len(y)*abs(y[m].mean()-p[m].mean())
    return float(e)

def metrics(p, pred, y):
    f1 = f1_score(y, pred, zero_division=0)
    try: auc = roc_auc_score(y,p) if len(np.unique(y))>1 else 0.0
    except: auc=0.0
    return dict(f1=float(f1), auc=float(auc), ece=ece(p,y))

def train_model(X_tr, y_tr, X_val, y_val, device, criterion_fn,
                save_path, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    counts  = np.bincount(y_tr); weights = 1.0/counts[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    # NumpyDataset: no full-array tensor copy — only batches
    loader  = DataLoader(NumpyDataset(X_tr, y_tr),
                         batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    model = SEResNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_f1, wait = 0.0, 0
    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        p_val, pred_val = infer(model, X_val, device)
        vf1 = f1_score(y_val, pred_val, zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1; wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
        if ep % 5 == 0:
            print(f"    ep={ep:>3}  val_f1={vf1:.4f}", flush=True)
        if wait >= PATIENCE: break
    model.load_state_dict(torch.load(save_path, weights_only=True,
                                      map_location=device))
    return model


# ── Main ───────────────────────────────────────────────────────
print("="*65)
print("URDI Baselines Comparison  (RAM-safe)")
print("="*65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── RAM-safe data loading ──────────────────────────────────────
print("Loading labels...", flush=True)
y14 = np.load(NPY_Y14)            # 32 KB — trivial
y2  = np.load(NPY_Y2)             # 4 KB  — trivial

print("Computing split indices...", flush=True)
idx = np.arange(len(y14))
tr_idx, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y14)
val_idx, te_idx = train_test_split(tmp, test_size=0.50, random_state=SEED,
                                    stratify=y14[tmp])
y_tr  = y14[tr_idx]
y_val = y14[val_idx]
y_te  = y14[te_idx]

# Load X14 once as mmap (file stays on disk, pages loaded on demand)
print("Memory-mapping X.npy (13 GB stays on disk)...", flush=True)
X14_mmap = np.load(NPY_X14, mmap_mode='r')

# Extract splits as contiguous float32 copies, then release mmap
print("Extracting train split (~9 GB copy)...", flush=True)
X_tr  = np.array(X14_mmap[tr_idx],  dtype=np.float32)
print("Extracting val split (~2 GB copy)...", flush=True)
X_val = np.array(X14_mmap[val_idx], dtype=np.float32)
print("Extracting test split (~2 GB copy)...", flush=True)
X_te  = np.array(X14_mmap[te_idx],  dtype=np.float32)

# Release mmap — frees 13 GB virtual mapping
del X14_mmap
import gc; gc.collect()
print("Mmap released. Loading FORGE data...", flush=True)

X2 = np.load(NPY_X2).astype(np.float32)   # 1.6 GB — fits fine
print(f"Train:{len(y_tr)} Val:{len(y_val)} Test:{len(y_te)} FORGE:{len(y2)}")
print(f"RAM in use after load: X_tr={X_tr.nbytes/1e9:.1f}GB  "
      f"X_val={X_val.nbytes/1e9:.1f}GB  X2={X2.nbytes/1e9:.1f}GB", flush=True)

results = []


# ── 0. URDI λ=10 (pre-trained) ────────────────────────────────
print("\n── URDI λ=10 (pre-trained) ─────────────────────────────")
urdi_path = MODEL_DIR / "best_seresnet_urdi_lam10p0.pth"
if urdi_path.exists():
    m = SEResNet().to(device)
    m.load_state_dict(torch.load(str(urdi_path), map_location=device, weights_only=True))
    p2, pred2 = infer(m, X2, device)
    r = metrics(p2, pred2, y2)
    print(f"  FORGE F1={r['f1']:.4f}  ECE={r['ece']:.4f}")
    results.append({"method":"URDI λ=10 (proposed)", "forge_f1":r["f1"],
                    "forge_ece":r["ece"], "forge_auc":r["auc"]})
    del m; torch.cuda.empty_cache()
else:
    print("  WARNING: best_seresnet_urdi_lam10p0.pth not found — skipping")


# ── 1. Baseline CE (pre-trained) ──────────────────────────────
print("\n── Baseline (standard CE) ──────────────────────────────")
base_path = MODEL_DIR / "best_seresnet_urdi_lam0p0.pth"
if not base_path.exists():
    base_path = MODEL_DIR / "best_seresnet.pth"
m_base = SEResNet().to(device)
m_base.load_state_dict(torch.load(str(base_path), map_location=device, weights_only=True))
p2_base, pred2_base = infer(m_base, X2, device)
r_base = metrics(p2_base, pred2_base, y2)
print(f"  FORGE F1={r_base['f1']:.4f}  ECE={r_base['ece']:.4f}")
results.append({"method":"Baseline (CE only)", "forge_f1":r_base["f1"],
                "forge_ece":r_base["ece"], "forge_auc":r_base["auc"]})


# ── 2. Temperature Scaling (post-hoc, no retraining) ──────────
torch.cuda.empty_cache(); time.sleep(3)
print("\n── Temperature Scaling (post-hoc) ──────────────────────")
logits_val = []
m_base.eval()
with torch.no_grad():
    ldr = DataLoader(NumpyDataset(X_val), batch_size=32, shuffle=False, num_workers=0)
    for (xb,) in ldr:
        logits_val.append(m_base(xb.to(device)).float().cpu())
logits_val = torch.cat(logits_val)
y_val_t    = torch.tensor(y_val, dtype=torch.long)

def nll_at_temp(T):
    return F.cross_entropy(logits_val / T, y_val_t).item()

result_T = minimize_scalar(nll_at_temp, bounds=(0.1, 10.0), method="bounded")
T_star   = float(result_T.x)
print(f"  Optimal temperature T* = {T_star:.4f}")

logits_forge = []
with torch.no_grad():
    ldr2 = DataLoader(NumpyDataset(X2), batch_size=32, shuffle=False, num_workers=0)
    for (xb,) in ldr2:
        logits_forge.append(m_base(xb.to(device)).float().cpu())
logits_forge = torch.cat(logits_forge)
p_ts    = F.softmax(logits_forge / T_star, dim=1)[:,1].numpy()
pred_ts = (p_ts >= 0.5).astype(int)
r_ts    = metrics(p_ts, pred_ts, y2)
print(f"  FORGE F1={r_ts['f1']:.4f}  ECE={r_ts['ece']:.4f}  T={T_star:.3f}")
results.append({"method":f"Temperature Scaling (T={T_star:.2f})", "forge_f1":r_ts["f1"],
                "forge_ece":r_ts["ece"], "forge_auc":r_ts["auc"]})
del m_base; torch.cuda.empty_cache()


# ── 3. Entropy Regularization ─────────────────────────────────
torch.cuda.empty_cache(); time.sleep(3)
print("\n── Entropy Regularization ──────────────────────────────")
BETA_ENT = 0.1

def criterion_entropy(logits, labels, beta=BETA_ENT):
    p       = F.softmax(logits, dim=1)
    entropy = -(p * (p + 1e-8).log()).sum(dim=1).mean()
    return F.cross_entropy(logits, labels) - beta * entropy

ent_path = str(MODEL_DIR / "best_seresnet_entropy_reg.pth")
print(f"  Training (beta={BETA_ENT})...")
m_ent = train_model(X_tr, y_tr, X_val, y_val, device,
                    lambda l,y: criterion_entropy(l,y), ent_path)
p2_ent, pred2_ent = infer(m_ent, X2, device)
r_ent = metrics(p2_ent, pred2_ent, y2)
print(f"  FORGE F1={r_ent['f1']:.4f}  ECE={r_ent['ece']:.4f}")
results.append({"method":"Entropy Regularization", "forge_f1":r_ent["f1"],
                "forge_ece":r_ent["ece"], "forge_auc":r_ent["auc"]})
del m_ent; torch.cuda.empty_cache()


# ── 4. Confidence Penalty ─────────────────────────────────────
torch.cuda.empty_cache(); time.sleep(3)
print("\n── Confidence Penalty ──────────────────────────────────")
GAMMA = 0.5; CONF_THRESH = 0.9

def criterion_conf_penalty(logits, labels):
    p_max   = F.softmax(logits, dim=1).max(dim=1).values
    penalty = F.relu(p_max - CONF_THRESH).mean()
    return F.cross_entropy(logits, labels) + GAMMA * penalty

conf_path = str(MODEL_DIR / "best_seresnet_conf_penalty.pth")
print(f"  Training (gamma={GAMMA}, thresh={CONF_THRESH})...")
m_conf = train_model(X_tr, y_tr, X_val, y_val, device,
                     criterion_conf_penalty, conf_path)
p2_conf, pred2_conf = infer(m_conf, X2, device)
r_conf = metrics(p2_conf, pred2_conf, y2)
print(f"  FORGE F1={r_conf['f1']:.4f}  ECE={r_conf['ece']:.4f}")
results.append({"method":"Confidence Penalty", "forge_f1":r_conf["f1"],
                "forge_ece":r_conf["ece"], "forge_auc":r_conf["auc"]})
del m_conf; torch.cuda.empty_cache()


# ── 5. Deep Ensemble ──────────────────────────────────────────
torch.cuda.empty_cache(); time.sleep(3)
print(f"\n── Deep Ensemble ({N_ENSEMBLE}x SE-ResNet) ─────────────────────")
ensemble_probs = []
for i, seed_i in enumerate(ENSEMBLE_SEEDS):
    ens_path = str(MODEL_DIR / f"best_seresnet_ensemble_{seed_i}.pth")
    print(f"  Training member {i+1}/{N_ENSEMBLE} (seed={seed_i})...")
    m_ens = train_model(X_tr, y_tr, X_val, y_val, device,
                        nn.CrossEntropyLoss(), ens_path, seed=seed_i)
    p2_e, _ = infer(m_ens, X2, device)
    ensemble_probs.append(p2_e)
    del m_ens; torch.cuda.empty_cache()

p_ens    = np.stack(ensemble_probs).mean(axis=0)
pred_ens = (p_ens >= 0.5).astype(int)
r_ens    = metrics(p_ens, pred_ens, y2)
print(f"  FORGE F1={r_ens['f1']:.4f}  ECE={r_ens['ece']:.4f}")
results.append({"method":f"Deep Ensemble ({N_ENSEMBLE}x)", "forge_f1":r_ens["f1"],
                "forge_ece":r_ens["ece"], "forge_auc":r_ens["auc"]})


# ── Results table ──────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(DATA_DIR / "urdi_baselines.csv", index=False)
print(f"\n{'Method':<38} {'F1':>8} {'ECE':>8} {'AUC':>8}")
print("-"*63)
for _, r in df.iterrows():
    marker = " <-- PROPOSED" if "URDI" in r["method"] else ""
    print(f"  {r['method']:<36} {r['forge_f1']:>8.4f} "
          f"{r['forge_ece']:>8.4f} {r['forge_auc']:>8.4f}{marker}")


# ── Figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors  = ["#E24B4A","#B4B2A9","#EF9F27","#7F77DD","#1a5c3a","#1D9E75"]
x       = range(len(df))

for ax, col, title in [
    (axes[0], "forge_f1",  "Utah FORGE 3-2417 F1\n(higher is better)"),
    (axes[1], "forge_ece", "FORGE ECE\n(lower is better)"),
]:
    bars = ax.bar(x, df[col], color=colors[:len(df)],
                  edgecolor="white", linewidth=0.5, alpha=0.9)
    base_val = df[df["method"].str.contains("Baseline")][col].values[0]
    ax.axhline(base_val, color="#B4B2A9", linestyle="--",
               linewidth=1.5, label=f"Baseline ({base_val:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"].tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    for bar, v in zip(bars, df[col]):
        ax.text(bar.get_x()+bar.get_width()/2,
                v+(max(df[col])-min(df[col]))*0.02,
                f"{v:.4f}", ha="center", fontsize=7)

fig.suptitle("URDI vs Calibration Baselines — Utah FORGE 3-2417",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "paper_urdi_baselines.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: data/figures/paper_urdi_baselines.png")

with open(DATA_DIR / "urdi_baselines_report.txt", "w", encoding="utf-8") as f:
    f.write("URDI Baselines Comparison\n")
    f.write("="*55 + "\n\n")
    f.write(f"{'Method':<38} {'F1':>8} {'ECE':>8}\n")
    f.write("-"*55 + "\n")
    for _, r in df.iterrows():
        f.write(f"  {r['method']:<36} {r['forge_f1']:>8.4f} {r['forge_ece']:>8.4f}\n")

print("Saved: data/urdi_baselines_report.txt")
print("\n" + "="*65)
print("Done. Share data/urdi_baselines_report.txt for paper update.")
print("="*65)
