"""
DG Baselines — RandConv and Mixup
5 seeds each, same SE-ResNet backbone + training protocol as URDI.
Evaluated on Utah FORGE 3-2417 (zero-shot).

Produces:
  data/dg_baselines_report.txt   — mean±std table for paper
  data/dg_baselines_results.csv  — per-seed raw results
  Model/dg_{method}_s{seed}.pth  — saved weights

Run time: ~2 hrs total on RTX 4060 (4 methods × 5 seeds × ~25 min/run)
Crash recovery: resumes from dg_baselines_partial.csv
"""

import os, gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True

# ── Paths ──────────────────────────────────────────────────────
NPY_X14      = Path("./Dataset/X.npy")
NPY_Y14      = Path("./Dataset/y.npy")
NPY_X2       = Path("./Dataset/X_forge.npy")
NPY_Y2       = Path("./Dataset/y_forge.npy")
MODEL_DIR    = Path("./Model")
DATA_DIR     = Path("./data")
FIG_DIR      = DATA_DIR / "figures"
PARTIAL_CSV  = DATA_DIR / "dg_baselines_partial.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (identical to URDI multi-seed) ─────────────
SEEDS        = [42, 7, 13, 99, 2024]
EPOCHS       = 30
PATIENCE     = 8
LR           = 3e-4
WEIGHT_DECAY = 1e-3
BATCH        = 16
MC_PASSES    = 20     # for uncertainty eval
N_BOOTSTRAP  = 2000

# RandConv: probability of randomizing stem conv per batch
RANDCONV_P   = 0.5
# Mixup: alpha parameter for Beta distribution
MIXUP_ALPHA  = 0.4


# ── Dataset ────────────────────────────────────────────────────
class HalfDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i].astype(np.float32)), self.y[i]

def make_loader(X, y, batch, weighted=False):
    ds = HalfDS(X, y)
    if weighted:
        c = np.bincount(y)
        w = torch.tensor(1.0/c[y], dtype=torch.float32)
        s = WeightedRandomSampler(w, len(w), replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=s, num_workers=0)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

def free():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)


# ── SE-ResNet (identical to URDI script) ──────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
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
        def dn(ci,co):
            return nn.Sequential(
                nn.Conv2d(ci,co,3,stride=2,padding=1,bias=False),
                nn.BatchNorm2d(co), nn.GELU())
        self.stem = nn.Sequential(
            nn.Conv2d(1,32,(7,15),stride=(2,4),padding=(3,7),bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((3,5),stride=(2,2),padding=(1,2)))
        self.body = nn.Sequential(
            SEResBlock(32), SEResBlock(32), dn(32,64),
            SEResBlock(64), SEResBlock(64), dn(64,128),
            SEResBlock(128),SEResBlock(128),dn(128,256),
            SEResBlock(256),SEResBlock(256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256,128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128,2))

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# ── RandConv augmentation ─────────────────────────────────────
class RandConvStem(nn.Module):
    """
    Randomized first-layer convolution (Xu et al. 2021).
    At each training batch (with prob p), replace the stem conv weights
    with random Gaussian weights (same shape), normalized to preserve
    mean activation magnitude.
    At inference: use the learned weights normally.
    """
    def __init__(self, base_stem, p=RANDCONV_P):
        super().__init__()
        self.base_stem = base_stem   # original SEResNet stem
        self.p = p
        # Save original conv weight shape
        conv = base_stem[0]
        self.weight_shape = conv.weight.shape

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.p:
            # Create random kernel same shape as stem conv
            rand_w = torch.randn(self.weight_shape, device=x.device, dtype=x.dtype)
            # Normalize: scale so that std of output matches original
            rand_w = rand_w / (rand_w.norm() + 1e-8) * self.base_stem[0].weight.norm()
            # Apply random conv then rest of stem (BN, GELU, Pool)
            out = F.conv2d(x, rand_w,
                           bias=None,
                           stride=self.base_stem[0].stride,
                           padding=self.base_stem[0].padding)
            # BN + GELU + Pool
            out = self.base_stem[1](out)   # BN
            out = self.base_stem[2](out)   # GELU
            out = self.base_stem[3](out)   # MaxPool
            return out
        else:
            return self.base_stem(x)

class SEResNetRandConv(SEResNet):
    """SE-ResNet with RandConv applied to stem at train time."""
    def __init__(self):
        super().__init__()
        self.rand_stem = RandConvStem(self.stem)

    def forward(self, x):
        x = self.rand_stem(x)
        return self.head(self.body(x))


# ── Metrics ────────────────────────────────────────────────────
def ece_metric(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p <= hi if i == n_bins-1 else p < hi)
        if mask.sum() > 0:
            e += mask.sum()/len(y) * abs(y[mask].mean() - p[mask].mean())
    return float(e)

def mce_metric(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    worst = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p <= hi if i == n_bins-1 else p < hi)
        if mask.sum() > 0:
            worst = max(worst, abs(y[mask].mean() - p[mask].mean()))
    return float(worst)

def aurc_metric(p, y, n_mc=MC_PASSES, model=None, loader=None, device=None):
    """
    AURC via MC-Dropout uncertainty = 1 - 2|p_mc - 0.5|.
    Lower AURC = uncertainty better identifies errors.
    """
    if model is None:
        unc = 1.0 - 2.0*np.abs(p - 0.5)
    else:
        # Run MC passes
        model.eval()
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)): m.train()
        all_passes = []
        for _ in range(n_mc):
            pp = []
            with torch.no_grad():
                for xb, _ in loader:
                    pr = F.softmax(model(xb.to(device)).float(), dim=1)[:,1]
                    pp.append(pr.cpu().numpy())
            all_passes.append(np.concatenate(pp))
        mat    = np.stack(all_passes, 0)
        mean_p = mat.mean(0)
        unc    = 1.0 - 2.0*np.abs(mean_p - 0.5)
        p      = mean_p

    pred  = (p >= 0.5).astype(int)
    error = (pred != y).astype(float)
    # Sort by uncertainty descending; build risk-coverage curve
    order = np.argsort(unc)[::-1]   # most uncertain first → abstain first
    n     = len(y)
    risks, coverages = [], []
    for k in range(1, n+1):
        answered = order[k-1:]     # least uncertain = most confident
        if len(answered) == 0: continue
        risks.append(error[answered].mean())
        coverages.append(len(answered)/n)
    # AURC = trapezoidal integral of risk over coverage
    risks     = np.array(risks[::-1])    # increasing coverage
    coverages = np.array(coverages[::-1])
    return float(np.trapz(risks, coverages))

def evaluate(model, loader, device):
    model.eval(); ps, preds, labs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            ps.append(F.softmax(logits.float(), dim=1)[:,1].cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            labs.append(yb.numpy())
    p    = np.concatenate(ps)
    pred = np.concatenate(preds)
    lab  = np.concatenate(labs)
    return float(f1_score(lab, pred, zero_division=0)), p, pred, lab


# ── Training loops ─────────────────────────────────────────────
def train_standard(tr_ldr, val_ldr, device, save_path, seed,
                   model_cls=SEResNet, criterion=None):
    set_seed(seed)
    model = model_cls().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce    = nn.CrossEntropyLoss()
    if criterion is None: criterion = lambda logits, y: ce(logits, y)
    best_f1, wait = 0.0, 0

    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tr_ldr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        vf1, _, _, _ = evaluate(model, val_ldr, device)
        if vf1 > best_f1:
            best_f1 = vf1; wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep={ep:>3}  val_f1={vf1:.4f}  best={best_f1:.4f}", flush=True)
        if wait >= PATIENCE:
            print(f"    Early stop ep={ep}"); break

    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    return model


def train_mixup(tr_ldr, val_ldr, device, save_path, seed, alpha=MIXUP_ALPHA):
    """Mixup training (Zhang et al. 2018). Uses soft CE for mixed labels."""
    set_seed(seed)
    model = SEResNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_f1, wait = 0.0, 0

    def mixup_batch(xb, yb):
        lam  = np.random.beta(alpha, alpha)
        idx  = torch.randperm(xb.size(0), device=xb.device)
        xm   = lam*xb + (1-lam)*xb[idx]
        ya   = F.one_hot(yb, 2).float()
        ym   = lam*ya + (1-lam)*ya[idx]    # soft labels
        return xm, ym

    def soft_ce(logits, ym):
        log_p = F.log_softmax(logits.float(), dim=1)
        return -(ym * log_p).sum(1).mean()

    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tr_ldr:
            xb, yb = xb.to(device), yb.to(device)
            xm, ym = mixup_batch(xb, yb)
            opt.zero_grad()
            loss = soft_ce(model(xm), ym)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Evaluate on un-mixed val data
        vf1, _, _, _ = evaluate(model, val_ldr, device)
        if vf1 > best_f1:
            best_f1 = vf1; wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep={ep:>3}  val_f1={vf1:.4f}  best={best_f1:.4f}", flush=True)
        if wait >= PATIENCE:
            print(f"    Early stop ep={ep}"); break

    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))
    return model


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
print("="*65)
print("DG Baselines: RandConv + Mixup")
print(f"  2 methods × {len(SEEDS)} seeds = {2*len(SEEDS)} runs")
print("="*65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load labels
y14 = np.load(NPY_Y14)
y2  = np.load(NPY_Y2)

# Same split as URDI (random_state=42, stratified)
idx                = np.arange(len(y14))
tr_idx, tmp        = train_test_split(idx, test_size=0.30, random_state=42, stratify=y14)
val_idx, te_idx    = train_test_split(tmp, test_size=0.50, random_state=42, stratify=y14[tmp])
y_tr = y14[tr_idx]; y_val = y14[val_idx]; y_te = y14[te_idx]
print(f"Split: train={len(y_tr)} val={len(y_val)} test={len(y_te)} forge={len(y2)}")

# Load data float16
print("\nLoading data (float16)...")
t0 = time.time()
X14m  = np.load(NPY_X14, mmap_mode='r')
X_tr  = X14m[tr_idx].astype(np.float16)
X_val = X14m[val_idx].astype(np.float16)
X_te  = X14m[te_idx].astype(np.float16)
del X14m; free()
X2 = np.load(NPY_X2).astype(np.float16)
gb = (X_tr.nbytes+X_val.nbytes+X_te.nbytes+X2.nbytes)/1e9
print(f"  {gb:.2f} GB RAM  ({time.time()-t0:.1f}s)")

tr_ldr    = make_loader(X_tr,  y_tr,  BATCH, weighted=True)
val_ldr   = make_loader(X_val, y_val, BATCH)
te_ldr    = make_loader(X_te,  y_te,  BATCH)
forge_ldr = make_loader(X2,    y2,    BATCH)

# Crash recovery
rows, done = [], set()
if PARTIAL_CSV.exists():
    df_p = pd.read_csv(PARTIAL_CSV)
    rows = df_p.to_dict("records")
    for r in rows: done.add((r["method"], int(r["seed"])))
    print(f"Recovered {len(rows)} runs.")

# ── Method configs ─────────────────────────────────────────────
METHODS = [
    ("Baseline CE",   SEResNet,         "ce",      "standard"),
    ("Mixup",         SEResNet,         "mixup",   "mixup"),
    ("RandConv",      SEResNetRandConv, "randconv","standard"),
]

for method_name, model_cls, tag, train_mode in METHODS:
    print(f"\n{'='*50}\n{method_name}\n{'='*50}")
    for seed in SEEDS:
        key = (method_name, seed)
        if key in done:
            print(f"  [SKIP] seed={seed}"); continue

        print(f"\n  seed={seed}  ({method_name})", flush=True)
        save = str(MODEL_DIR / f"dg_{tag}_s{seed}.pth")
        t0   = time.time()

        try:
            if train_mode == "mixup":
                model = train_mixup(tr_ldr, val_ldr, device, save, seed)
            else:
                model = train_standard(tr_ldr, val_ldr, device, save, seed,
                                       model_cls=model_cls)

            # ── Evaluate ──────────────────────────────────────
            # Frisco-2-P in-distribution
            f1_s14, _, _, _ = evaluate(model, te_ldr, device)

            # FORGE zero-shot — softmax
            f1_f, p_f, pred_f, lab_f = evaluate(model, forge_ldr, device)
            ece_f = ece_metric(p_f, lab_f)
            mce_f = mce_metric(p_f, lab_f)

            # AURC via MC-Dropout
            aurc_f = aurc_metric(None, lab_f,
                                  model=model, loader=forge_ldr, device=device)

            t1 = time.time()
            print(f"  → S14 F1={f1_s14:.4f}  FORGE F1={f1_f:.4f}  "
                  f"ECE={ece_f:.4f}  MCE={mce_f:.4f}  "
                  f"AURC={aurc_f:.4f}  t={t1-t0:.0f}s")

            rows.append({
                "method": method_name, "seed": seed,
                "s14_f1": f1_s14,
                "forge_f1": f1_f, "forge_ece": ece_f,
                "forge_mce": mce_f, "forge_aurc": aurc_f,
            })
            done.add(key)
            pd.DataFrame(rows).to_csv(PARTIAL_CSV, index=False)

        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            if "model" in dir(): del model
            free()

# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
df = pd.DataFrame(rows)
df.to_csv(DATA_DIR / "dg_baselines_results.csv", index=False)

print("\n" + "="*65 + "\nSUMMARY\n" + "="*65)
method_order = ["Baseline CE", "Mixup", "RandConv"]
summary_rows = []
for m in method_order:
    sub = df[df["method"] == m]
    if len(sub) == 0: continue
    r = {
        "method":    m,
        "n":         len(sub),
        "s14_mean":  sub.s14_f1.mean(),    "s14_std":  sub.s14_f1.std(),
        "f1_mean":   sub.forge_f1.mean(),  "f1_std":   sub.forge_f1.std(),
        "ece_mean":  sub.forge_ece.mean(), "ece_std":  sub.forge_ece.std(),
        "mce_mean":  sub.forge_mce.mean(), "mce_std":  sub.forge_mce.std(),
        "aurc_mean": sub.forge_aurc.mean(),"aurc_std": sub.forge_aurc.std(),
    }
    summary_rows.append(r)
    print(f"  {m:<15} n={r['n']}  "
          f"FORGE F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}  "
          f"ECE={r['ece_mean']:.4f}±{r['ece_std']:.4f}  "
          f"MCE={r['mce_mean']:.4f}±{r['mce_std']:.4f}  "
          f"AURC={r['aurc_mean']:.4f}±{r['aurc_std']:.4f}")

with open(DATA_DIR / "dg_baselines_report.txt", "w", encoding="utf-8") as f:
    f.write("DG BASELINES — UTAH FORGE 3-2417 (ZERO-SHOT)\n" + "="*70 + "\n")
    f.write(f"Seeds: {SEEDS}  Backbone: SE-ResNet  Protocol: same as URDI\n\n")
    f.write(f"{'Method':<16}{'n':>3}{'S14 F1':>12}{'FORGE F1':>16}"
            f"{'ECE':>14}{'MCE':>14}{'AURC':>14}\n")
    f.write("-"*75 + "\n")
    for r in summary_rows:
        f.write(f"  {r['method']:<14}{r['n']:>3} "
                f"{r['s14_mean']:>7.4f}±{r['s14_std']:.4f} "
                f"{r['f1_mean']:>7.4f}±{r['f1_std']:.4f} "
                f"{r['ece_mean']:>7.4f}±{r['ece_std']:.4f} "
                f"{r['mce_mean']:>7.4f}±{r['mce_std']:.4f} "
                f"{r['aurc_mean']:>7.4f}±{r['aurc_std']:.4f}\n")

print("\nSaved: data/dg_baselines_report.txt")
print("="*65 + "\nDone.\n" + "="*65)
