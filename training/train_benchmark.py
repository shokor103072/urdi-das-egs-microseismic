"""
Multi-Seed Architecture Benchmark — Memory Efficient Version
Fixes: PC hanging due to RAM overflow causing disk swapping

ROOT CAUSE:
  X14 = (3974, 1, 361, 2400) float32 = 13.8 GB
  + DataLoader copies during training = 30+ GB total
  -> Windows pagefile (swap) kicks in -> hard disk thrashing -> hang

FIXES APPLIED:
  1. Load X14 with mmap_mode='r' then extract splits as float16 (half size)
  2. Delete mmap reference immediately after splitting
  3. Custom Dataset converts float16->float32 per-batch only (never full array)
  4. pin_memory=False, num_workers=0 (no extra RAM copies)
  5. gc.collect() + cuda.empty_cache() after every model+seed run
  6. One model in RAM at a time

RAM usage with this script:
  X_tr  (float16): ~4.8 GB
  X_val (float16): ~1.0 GB
  X_te  (float16): ~1.0 GB
  X2    (float16): ~0.9 GB
  Model weights  : ~0.05 GB
  TOTAL          : ~7.8 GB  (fits in 16 GB RAM)
"""

import os, gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True

NPY_X14   = Path("./Dataset/X.npy")
NPY_Y14   = Path("./Dataset/y.npy")
NPY_X2    = Path("./Dataset/X_forge.npy")
NPY_Y2    = Path("./Dataset/y_forge.npy")
MODEL_DIR = Path("./Model")
DATA_DIR  = Path("./data")
FIG_DIR   = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS         = [42, 7, 13, 99, 2024]
TARGET_MODELS = ["SE-ResNet", "ResNet", "CNN-GRU", "ConvNeXt", "Conformer"]
EPOCHS        = 30
PATIENCE      = 8
LR            = 3e-4
WEIGHT_DECAY  = 1e-3
BATCH_SIZE    = 16       # small batch = less GPU memory per step
N_BOOTSTRAP   = 500
PARTIAL_CSV   = DATA_DIR / "multiseed_partial.csv"


# ══════════════════════════════════════════════════════════════
# MEMORY-EFFICIENT DATASET
# float16 stored in RAM, converted to float32 one batch at a time
# ══════════════════════════════════════════════════════════════
class HalfDataset(torch.utils.data.Dataset):
    def __init__(self, X_f16: np.ndarray, y: np.ndarray):
        self.X = X_f16                              # float16 in RAM
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        # float16 -> float32 for one sample only
        return torch.from_numpy(self.X[i].astype(np.float32)), self.y[i]


def make_loader(X_f16, y, batch_size, weighted=False):
    ds = HalfDataset(X_f16, y)
    if weighted:
        counts  = np.bincount(y)
        w       = torch.tensor(1.0 / counts[y], dtype=torch.float32)
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0, pin_memory=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)


# ══════════════════════════════════════════════════════════════
# ARCHITECTURES
# ══════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r), nn.ReLU(inplace=True),
            nn.Linear(ch//r, ch), nn.Sigmoid())
    def forward(self, x): return x * self.se(x).view(x.size(0), -1, 1, 1)

class SEResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch),
            nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch))
        self.se = SEBlock(ch); self.act = nn.GELU()
    def forward(self, x): return self.act(self.se(self.block(x)) + x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch),
            nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch))
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.block(x) + x)

def _stem():
    return nn.Sequential(
        nn.Conv2d(1,32,(7,15),stride=(2,4),padding=(3,7),bias=False),
        nn.BatchNorm2d(32), nn.GELU(),
        nn.MaxPool2d((3,5),stride=(2,2),padding=(1,2)))

def _down(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci,co,3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(co), nn.GELU())

def _head(ch):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.5), nn.Linear(ch,128), nn.GELU(),
        nn.Dropout(0.3), nn.Linear(128,2))

class SEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _stem(),
            SEResBlock(32), SEResBlock(32), _down(32,64),
            SEResBlock(64), SEResBlock(64), _down(64,128),
            SEResBlock(128), SEResBlock(128), _down(128,256),
            SEResBlock(256), SEResBlock(256))
        self.head = _head(256)
    def forward(self, x): return self.head(self.net(x))

class DASResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _stem(),
            ResBlock(32), ResBlock(32), _down(32,64),
            ResBlock(64), ResBlock(64), _down(64,128),
            ResBlock(128), ResBlock(128), _down(128,256),
            ResBlock(256), ResBlock(256))
        self.head = _head(256)
    def forward(self, x): return self.head(self.net(x))

class CNNGRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,(7,15),stride=(2,4),padding=(3,7),bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32,64,(3,7),stride=(2,2),padding=(1,3),bias=False),
            nn.BatchNorm2d(64), nn.GELU(), nn.AdaptiveAvgPool2d((1,None)))
        self.gru  = nn.GRU(64, 128, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=False)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128,64),
                                   nn.GELU(), nn.Linear(64,2))
    def forward(self, x):
        x = self.cnn(x).squeeze(2).transpose(1,2)
        out, _ = self.gru(x); return self.head(out[:,-1,:])

class ConvNeXtBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch,ch,7,padding=3,groups=ch,bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.pw = nn.Sequential(nn.Conv2d(ch,ch*4,1), nn.GELU(),
                                 nn.Conv2d(ch*4,ch,1))
        self.g  = nn.Parameter(torch.ones(ch,1,1)*1e-6)
    def forward(self, x):
        return x + self.pw(self.bn(self.dw(x))) * self.g

class ConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.s0 = nn.Sequential(nn.Conv2d(1,32,(4,8),stride=(2,4),bias=False),
                                  nn.BatchNorm2d(32))
        self.s1 = nn.Sequential(ConvNeXtBlock(32), ConvNeXtBlock(32))
        self.d1 = nn.Sequential(nn.BatchNorm2d(32),
                                  nn.Conv2d(32,64,2,stride=2,bias=False))
        self.s2 = nn.Sequential(*[ConvNeXtBlock(64) for _ in range(3)])
        self.d2 = nn.Sequential(nn.BatchNorm2d(64),
                                  nn.Conv2d(64,128,2,stride=2,bias=False))
        self.s3 = nn.Sequential(*[ConvNeXtBlock(128) for _ in range(3)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.LayerNorm(128), nn.Linear(128,64), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(64,2))
    def forward(self, x):
        x = self.s1(self.s0(x))
        x = self.s2(self.d1(x))
        x = self.s3(self.d2(x))
        return self.head(x)

class ConformerBlock(nn.Module):
    def __init__(self, d, h=4, ff=4, dr=0.1):
        super().__init__()
        self.ff1  = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,d*ff),
            nn.GELU(), nn.Dropout(dr), nn.Linear(d*ff,d), nn.Dropout(dr))
        self.attn = nn.MultiheadAttention(d, h, dropout=dr, batch_first=True)
        self.na   = nn.LayerNorm(d)
        self.conv = nn.Sequential(nn.LayerNorm(d), nn.Conv1d(d,d*2,1),
            nn.GLU(dim=1), nn.Conv1d(d,d,31,padding=15,groups=d),
            nn.BatchNorm1d(d), nn.SiLU(), nn.Conv1d(d,d,1), nn.Dropout(dr))
        self.ff2  = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,d*ff),
            nn.GELU(), nn.Dropout(dr), nn.Linear(d*ff,d), nn.Dropout(dr))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        x = x + 0.5*self.ff1(x)
        r, _ = self.attn(self.na(x), self.na(x), self.na(x)); x = x + r
        x = x + self.conv(x.transpose(1,2)).transpose(1,2)
        return self.norm(x + 0.5*self.ff2(x))

class Conformer(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,(7,15),stride=(2,4),padding=(3,7),bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32,d,(3,3),stride=(2,2),padding=1,bias=False),
            nn.BatchNorm2d(d), nn.GELU())
        self.proj   = nn.Linear(d, d)
        self.blocks = nn.Sequential(ConformerBlock(d), ConformerBlock(d))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(d,32), nn.GELU(), nn.Linear(32,2))
    def forward(self, x):
        x = self.cnn(x); B,C,H,W = x.shape
        x = self.proj(x.mean(2).transpose(1,2))
        x = self.blocks(x)
        return self.head(x.transpose(1,2))

MODEL_CLASSES = {
    "SE-ResNet": SEResNet,
    "ResNet":    DASResNet,
    "CNN-GRU":   CNNGRUModel,
    "ConvNeXt":  ConvNeXt,
    "Conformer": Conformer,
}


# ══════════════════════════════════════════════════════════════
# EVAL + BOOTSTRAP
# ══════════════════════════════════════════════════════════════
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(1).cpu())
            labels.append(yb)
    pred  = torch.cat(preds).numpy()
    label = torch.cat(labels).numpy()
    return float(f1_score(label, pred, zero_division=0)), pred, label

def bootstrap_ci(y_true, y_pred, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    ns  = len(y_true)
    f1s = [f1_score(y_true[rng.integers(0,ns,ns)],
                     y_pred[rng.integers(0,ns,ns)],
                     zero_division=0) for _ in range(n)]
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════
def train_one(tr_loader, val_loader, ModelClass, device, save_path, seed):
    set_seed(seed)
    model = ModelClass().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.CrossEntropyLoss()
    best_f1, wait = 0.0, 0

    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vf1, _, _ = evaluate(model, val_loader, device)
        if vf1 > best_f1:
            best_f1 = vf1; wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep={ep:>3}  val_f1={vf1:.4f}  best={best_f1:.4f}",
                  flush=True)
        if wait >= PATIENCE:
            print(f"    Early stop ep={ep}"); break

    model.load_state_dict(torch.load(save_path, weights_only=True,
                                      map_location=device))
    return model


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
print("="*65)
print("Multi-Seed Benchmark — Memory Efficient")
print(f"  Models: {TARGET_MODELS}")
print(f"  Seeds : {SEEDS}")
print("="*65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU    : {torch.cuda.get_device_name(0)}  ({total_vram:.1f} GB VRAM)")

# Load labels (tiny)
y14_t = np.load(NPY_Y14); y2 = np.load(NPY_Y2)
y14   = np.repeat(y14_t, 1)

# Fixed split indices
idx = np.arange(len(y14))
tr_idx, tmp = train_test_split(idx, test_size=0.30,
                                random_state=42, stratify=y14)
val_idx, te_idx = train_test_split(tmp, test_size=0.50,
                                    random_state=42, stratify=y14[tmp])
y_tr  = y14[tr_idx]; y_val = y14[val_idx]; y_te = y14[te_idx]
print(f"\nSplit: train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

# Load X14 via mmap, extract splits as float16, release mmap
print("\nExtracting data splits as float16 (saves ~7 GB RAM)...")
t0 = time.time()
X14m  = np.load(NPY_X14, mmap_mode='r')
X_tr  = X14m[tr_idx].astype(np.float16);  print(f"  X_tr  {X_tr.nbytes/1e9:.2f} GB")
X_val = X14m[val_idx].astype(np.float16); print(f"  X_val {X_val.nbytes/1e9:.2f} GB")
X_te  = X14m[te_idx].astype(np.float16);  print(f"  X_te  {X_te.nbytes/1e9:.2f} GB")
del X14m; free_memory()   # release mmap
X2    = np.load(NPY_X2).astype(np.float16)
print(f"  X2    {X2.nbytes/1e9:.2f} GB")
total = (X_tr.nbytes+X_val.nbytes+X_te.nbytes+X2.nbytes)/1e9
print(f"  Total in RAM: {total:.2f} GB  ({time.time()-t0:.1f}s)")

# Build loaders (reused across seeds)
tr_loader    = make_loader(X_tr,  y_tr,  BATCH_SIZE, weighted=True)
val_loader   = make_loader(X_val, y_val, BATCH_SIZE)
te_loader    = make_loader(X_te,  y_te,  BATCH_SIZE)
forge_loader = make_loader(X2,    y2,    BATCH_SIZE)

# Crash recovery
rows, completed = [], set()
if PARTIAL_CSV.exists():
    df_p = pd.read_csv(PARTIAL_CSV)
    rows = df_p.to_dict("records")
    for r in rows: completed.add((r["model"], int(r["seed"])))
    print(f"\nRecovered {len(rows)} results from previous run.")

# Training loop
total_runs = len(TARGET_MODELS) * len(SEEDS)
run_idx    = 0

for name in TARGET_MODELS:
    Cls = MODEL_CLASSES[name]
    print(f"\n{'='*55}\nModel: {name}\n{'='*55}")

    for seed in SEEDS:
        run_idx += 1
        if (name, seed) in completed:
            print(f"  [SKIP] seed={seed}"); continue

        print(f"\n  [{run_idx}/{total_runs}] seed={seed}", flush=True)
        save = str(MODEL_DIR / f"ms_{name.lower().replace('-','_')}_{seed}.pth")
        t0   = time.time()

        try:
            model = train_one(tr_loader, val_loader, Cls, device, save, seed)
            f1_14, _, _        = evaluate(model, te_loader,    device)
            f1_2,  p2, l2      = evaluate(model, forge_loader, device)
            ci_lo, ci_hi       = bootstrap_ci(l2, p2)
            print(f"  S14={f1_14:.4f}  FORGE={f1_2:.4f}  "
                  f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  t={time.time()-t0:.0f}s")
            row = {"model":name,"seed":seed,"s14_f1":f1_14,"forge_f1":f1_2,
                   "gap":f1_14-f1_2,"ci_lo":ci_lo,"ci_hi":ci_hi}
            rows.append(row); completed.add((name, seed))
            pd.DataFrame(rows).to_csv(PARTIAL_CSV, index=False)
        except Exception as e:
            print(f"  [ERROR] {e}")
        finally:
            if "model" in dir(): del model
            free_memory()

# Summary
df = pd.DataFrame(rows)
df.to_csv(DATA_DIR / "multiseed_results.csv", index=False)

print("\n" + "="*65)
print("SUMMARY")
print("="*65)
summary = []
for name in TARGET_MODELS:
    sub = df[df["model"]==name]
    if len(sub)==0: continue
    r = {"model":name, "n_seeds":len(sub),
         "s14_mean":sub.s14_f1.mean(), "s14_std":sub.s14_f1.std(),
         "forge_mean":sub.forge_f1.mean(), "forge_std":sub.forge_f1.std(),
         "gap_mean":sub.gap.mean(), "gap_std":sub.gap.std(),
         "ci_lo":sub.ci_lo.mean(), "ci_hi":sub.ci_hi.mean()}
    summary.append(r)
    print(f"\n{name} (n={r['n_seeds']})")
    print(f"  S14 F1  : {r['s14_mean']:.4f} +/- {r['s14_std']:.4f}")
    print(f"  FORGE F1: {r['forge_mean']:.4f} +/- {r['forge_std']:.4f}  "
          f"95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    print(f"  Gap     : {r['gap_mean']:.4f} +/- {r['gap_std']:.4f}")

df_sum = pd.DataFrame(summary)
df_sum.to_csv(DATA_DIR/"multiseed_summary.csv", index=False)

# Figure
if len(df_sum) > 0:
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    colors = ["#1D9E75","#7F77DD","#EF9F27","#E24B4A","#1a5c3a"]
    x = np.arange(len(df_sum))
    for ax,(cm,cs,title) in zip(axes,[
        ("s14_mean","s14_std","Frisco-2-P F1 (in-distribution)"),
        ("forge_mean","forge_std","Utah FORGE 3-2417 F1 (zero-shot)")]):
        bars = ax.bar(x, df_sum[cm], color=colors[:len(df_sum)],
                      edgecolor="white", alpha=0.9, width=0.5)
        ax.errorbar(x, df_sum[cm], yerr=df_sum[cs],
                    fmt="none", color="black", capsize=5, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sum["model"], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("F1"); ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(max(0, df_sum[cm].min()-0.12), 1.02)
        for bar,m,s in zip(bars,df_sum[cm],df_sum[cs]):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+0.008,
                    f"{m:.3f}\n+/-{s:.3f}", ha="center", fontsize=7.5)
    fig.suptitle(f"Multi-Seed Benchmark (n={len(SEEDS)} seeds: {SEEDS})\nMean +/- std",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"paper_multiseed.png", dpi=200, bbox_inches="tight")
    plt.close(); print("\nSaved: data/figures/paper_multiseed.png")

with open(DATA_DIR/"multiseed_report.txt","w",encoding="utf-8") as f:
    f.write("MULTI-SEED ARCHITECTURE BENCHMARK\n"+"="*70+"\n\n")
    f.write(f"Seeds:{SEEDS}  Epochs:{EPOCHS}  Patience:{PATIENCE}  Batch:{BATCH_SIZE}\n\n")
    f.write(f"{'Model':<14}{'n':>3}{'S14 F1':>15}{'FORGE F1':>17}"
            f"{'Gap':>15}{'95% CI':>20}\n"+"-"*80+"\n")
    for _,r in df_sum.iterrows():
        f.write(f"  {r['model']:<12}{r['n_seeds']:>3} "
                f"{r['s14_mean']:>7.4f}+/-{r['s14_std']:.4f} "
                f"{r['forge_mean']:>8.4f}+/-{r['forge_std']:.4f} "
                f"{r['gap_mean']:>6.4f}+/-{r['gap_std']:.4f} "
                f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]\n")

print("Saved: data/multiseed_report.txt")
print("\n"+"="*65+"\nDone. Share data/multiseed_report.txt\n"+"="*65)
