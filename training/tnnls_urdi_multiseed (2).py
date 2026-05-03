"""
Multi-Seed URDI Training
5 seeds x 5 lambdas = 25 runs total
Memory efficient: float16 in RAM, float32 only per-batch on GPU
Crash recovery: saves after every run to urdi_multiseed_partial.csv
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

SEEDS        = [42, 7, 13, 99, 2024]
LAMBDAS      = [0.0, 0.01, 0.1, 1.0, 10.0]
EPOCHS       = 30
PATIENCE     = 8
LR           = 3e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE   = 16
MC_K         = 2
N_BOOTSTRAP  = 2000
PARTIAL_CSV  = DATA_DIR / "urdi_multiseed_partial.csv"


# ── Float16 dataset ────────────────────────────────────────────
class HalfDataset(torch.utils.data.Dataset):
    def __init__(self, X_f16, y):
        self.X = X_f16
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i].astype(np.float32)), self.y[i]

def make_loader(X_f16, y, batch_size, weighted=False):
    ds = HalfDataset(X_f16, y)
    if weighted:
        counts = np.bincount(y)
        w = torch.tensor(1.0/counts[y], dtype=torch.float32)
        s = WeightedRandomSampler(w, len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=s,
                          num_workers=0, pin_memory=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)


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
        def dn(ci,co):
            return nn.Sequential(nn.Conv2d(ci,co,3,stride=2,padding=1,bias=False),
                                   nn.BatchNorm2d(co),nn.GELU())
        self.stem = nn.Sequential(
            nn.Conv2d(1,32,(7,15),stride=(2,4),padding=(3,7),bias=False),
            nn.BatchNorm2d(32),nn.GELU(),
            nn.MaxPool2d((3,5),stride=(2,2),padding=(1,2)))
        self.body = nn.Sequential(
            SEResBlock(32),SEResBlock(32),dn(32,64),
            SEResBlock(64),SEResBlock(64),dn(64,128),
            SEResBlock(128),SEResBlock(128),dn(128,256),
            SEResBlock(256),SEResBlock(256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.5),
            nn.Linear(256,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,2))

    def forward(self, x): return self.head(self.body(self.stem(x)))

    def mc_var(self, x, k=MC_K):
        probs = torch.stack(
            [F.softmax(self.forward(x),dim=1)[:,1] for _ in range(k)], 0)
        return probs.var(0).mean()


# ── Helpers ────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval(); ps, preds, labs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            ps.append(F.softmax(logits.float(),dim=1)[:,1].cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            labs.append(yb.numpy())
    p    = np.concatenate(ps)
    pred = np.concatenate(preds)
    lab  = np.concatenate(labs)
    return float(f1_score(lab,pred,zero_division=0)), p, pred, lab

def ece(p, y, n=10):
    bins = np.linspace(0,1,n+1); e = 0.0
    for i in range(n):
        m = (p>=bins[i]) & (p<=(bins[i+1] if i==n-1 else bins[i+1]))
        if m.sum()>0: e+=m.sum()/len(y)*abs(y[m].mean()-p[m].mean())
    return float(e)

def mc_std_eval(model, loader, device, n_mc=20):
    model.eval()
    for m in model.modules():
        if isinstance(m,(nn.Dropout,nn.Dropout2d)): m.train()
    stds=[]
    with torch.no_grad():
        for xb,_ in loader:
            xb_t=xb.repeat_interleave(n_mc,0).to(device)
            pr=F.softmax(model(xb_t).float(),dim=1)[:,1].view(-1,n_mc)
            stds.append(pr.std(1).cpu().numpy())
    return float(np.concatenate(stds).mean())

def boot_ci(y_true, y_pred, n=N_BOOTSTRAP, seed=0):
    rng=np.random.default_rng(seed); ns=len(y_true)
    f1s=[]
    for _ in range(n):
        idx=rng.integers(0,ns,ns)
        f1s.append(f1_score(y_true[idx],y_pred[idx],zero_division=0))
    return float(np.percentile(f1s,2.5)), float(np.percentile(f1s,97.5))


# ── Training ───────────────────────────────────────────────────
def train_one(tr_loader, val_loader, device, lam, save_path, seed):
    set_seed(seed)
    model = SEResNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    ce    = nn.CrossEntropyLoss()
    best_f1, wait = 0.0, 0

    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tr_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            if lam == 0.0:
                loss = ce(logits, yb)
            else:
                loss = ce(logits, yb) + lam * model.mc_var(xb, k=MC_K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        vf1, _, _, _ = evaluate(model, val_loader, device)
        if vf1 > best_f1:
            best_f1=vf1; wait=0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep={ep:>3}  val={vf1:.4f}  best={best_f1:.4f}",
                  flush=True)
        if wait >= PATIENCE:
            print(f"    Early stop ep={ep}"); break

    model.load_state_dict(torch.load(save_path,weights_only=True,
                                      map_location=device))
    return model


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
print("="*65)
print("Multi-Seed URDI")
print(f"  {len(LAMBDAS)} lambdas x {len(SEEDS)} seeds = {len(LAMBDAS)*len(SEEDS)} runs")
print("="*65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type=="cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

y14_t = np.load(NPY_Y14); y2 = np.load(NPY_Y2)
y14   = np.repeat(y14_t, 1)

idx = np.arange(len(y14))
tr_idx,tmp    = train_test_split(idx,test_size=0.30,random_state=42,stratify=y14)
val_idx,te_idx= train_test_split(tmp,test_size=0.50,random_state=42,stratify=y14[tmp])
y_tr=y14[tr_idx]; y_val=y14[val_idx]; y_te=y14[te_idx]
print(f"Split: train={len(y_tr)} val={len(y_val)} test={len(y_te)} forge={len(y2)}")

print("\nLoading data as float16...")
t0=time.time()
X14m=np.load(NPY_X14,mmap_mode='r')
X_tr =X14m[tr_idx].astype(np.float16)
X_val=X14m[val_idx].astype(np.float16)
X_te =X14m[te_idx].astype(np.float16)
del X14m; free_memory()
X2=np.load(NPY_X2).astype(np.float16)
gb=(X_tr.nbytes+X_val.nbytes+X_te.nbytes+X2.nbytes)/1e9
print(f"  RAM: {gb:.2f} GB  ({time.time()-t0:.1f}s)")

tr_loader    = make_loader(X_tr, y_tr,  BATCH_SIZE, weighted=True)
val_loader   = make_loader(X_val,y_val, BATCH_SIZE)
te_loader    = make_loader(X_te, y_te,  BATCH_SIZE)
forge_loader = make_loader(X2,   y2,    BATCH_SIZE)

rows, completed = [], set()
if PARTIAL_CSV.exists():
    df_p=pd.read_csv(PARTIAL_CSV)
    rows=df_p.to_dict("records")
    for r in rows: completed.add((float(r["lambda"]),int(r["seed"])))
    print(f"\nRecovered {len(rows)} runs.")

total=len(LAMBDAS)*len(SEEDS); run_idx=0

for lam in LAMBDAS:
    tag   = str(lam).replace('.','p')
    label = "Baseline" if lam==0.0 else f"URDI lam={lam}"
    print(f"\n{'='*55}\n{label}\n{'='*55}")

    for seed in SEEDS:
        run_idx += 1
        key = (float(lam), int(seed))
        if key in completed:
            print(f"  [SKIP] seed={seed}"); continue

        print(f"\n  [{run_idx}/{total}] lam={lam}  seed={seed}", flush=True)
        save = str(MODEL_DIR / f"urdi_ms_lam{tag}_s{seed}.pth")
        t0   = time.time()

        try:
            model = train_one(tr_loader,val_loader,device,lam,save,seed)

            f1_14,_,_,_        = evaluate(model,te_loader,   device)
            f1_2, p2,pred2,lab2= evaluate(model,forge_loader,device)
            ci_lo, ci_hi       = boot_ci(lab2,pred2,seed=seed)
            ece2               = ece(p2, y2)
            mc2                = mc_std_eval(model,forge_loader,device)

            print(f"  S14={f1_14:.4f}  FORGE={f1_2:.4f}  "
                  f"ECE={ece2:.4f}  mc={mc2:.5f}  "
                  f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  t={time.time()-t0:.0f}s")

            row={"lambda":lam,"seed":seed,
                 "s14_f1":f1_14,"forge_f1":f1_2,
                 "forge_ece":ece2,"forge_mc_std":mc2,
                 "gap":f1_14-f1_2,"ci_lo":ci_lo,"ci_hi":ci_hi}
            rows.append(row); completed.add(key)
            pd.DataFrame(rows).to_csv(PARTIAL_CSV,index=False)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
        finally:
            if "model" in dir(): del model
            free_memory()


# ── Summary ────────────────────────────────────────────────────
df=pd.DataFrame(rows)
df.to_csv(DATA_DIR/"urdi_multiseed_results.csv",index=False)

print("\n"+"="*65+"\nSUMMARY\n"+"="*65)
summary=[]
for lam in LAMBDAS:
    sub=df[df["lambda"]==lam]
    if len(sub)==0: continue
    lbl="Baseline (CE)" if lam==0.0 else f"URDI lam={lam}"
    r={"lambda":lam,"label":lbl,"n":len(sub),
       "s14_mean":sub.s14_f1.mean(),"s14_std":sub.s14_f1.std(),
       "forge_mean":sub.forge_f1.mean(),"forge_std":sub.forge_f1.std(),
       "ece_mean":sub.forge_ece.mean(),"ece_std":sub.forge_ece.std(),
       "mc_mean":sub.forge_mc_std.mean(),
       "gap_mean":sub.gap.mean(),"gap_std":sub.gap.std(),
       "ci_lo":sub.ci_lo.mean(),"ci_hi":sub.ci_hi.mean()}
    summary.append(r)
    print(f"\n{lbl}  (n={r['n']})")
    print(f"  S14 : {r['s14_mean']:.4f}+/-{r['s14_std']:.4f}")
    print(f"  FORGE: {r['forge_mean']:.4f}+/-{r['forge_std']:.4f}  "
          f"CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
    print(f"  ECE : {r['ece_mean']:.4f}+/-{r['ece_std']:.4f}")
    print(f"  mc  : {r['mc_mean']:.5f}")

df_sum=pd.DataFrame(summary)
df_sum.to_csv(DATA_DIR/"urdi_multiseed_summary.csv",index=False)

# Figure
fig,axes=plt.subplots(1,3,figsize=(15,5))
colors=["#B4B2A9","#1D9E75","#7F77DD","#EF9F27","#E24B4A"]
x=np.arange(len(df_sum))
base_row=df_sum[df_sum["lambda"]==0.0].iloc[0]

for ax,(col_m,col_s,title) in zip(axes,[
    ("forge_mean","forge_std","FORGE F1  mean+/-std\nhigher is better"),
    ("ece_mean","ece_std","FORGE ECE  mean+/-std\nlower is better"),
    ("mc_mean",None,"Mean MC-Dropout std"),
]):
    vals=df_sum[col_m].values
    bars=ax.bar(x,vals,color=colors[:len(df_sum)],edgecolor="white",alpha=0.9)
    if col_s:
        ax.errorbar(x,vals,yerr=df_sum[col_s].values,fmt="none",
                    color="black",capsize=5,linewidth=1.5)
    bv=base_row[col_m]
    ax.axhline(bv,color="#888",linestyle="--",linewidth=1.5,
               label=f"Baseline={bv:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(df_sum["label"],rotation=25,
                                          ha="right",fontsize=8)
    ax.set_title(title,fontsize=9,fontweight="bold")
    ax.grid(axis="y",alpha=0.3); ax.legend(fontsize=7)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                v+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.015,
                f"{v:.4f}",ha="center",fontsize=7.5)

fig.suptitle(f"Multi-Seed URDI  (n={len(SEEDS)} seeds)\nMean+/-std across {SEEDS}",
             fontsize=11,fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR/"paper_urdi_multiseed.png",dpi=200,bbox_inches="tight")
plt.close(); print("\nSaved: paper_urdi_multiseed.png")

# Report
with open(DATA_DIR/"urdi_multiseed_report.txt","w",encoding="utf-8") as f:
    f.write("MULTI-SEED URDI REPORT\n"+"="*70+"\n\n")
    f.write(f"Seeds:{SEEDS}  Epochs:{EPOCHS}  Patience:{PATIENCE}  "
            f"Batch:{BATCH_SIZE}  MC_K:{MC_K}\n\n")
    f.write(f"{'Config':<20}{'n':>3}{'S14 F1':>15}{'FORGE F1':>17}"
            f"{'ECE':>12}{'Gap':>13}{'95%CI':>20}\n"+"-"*100+"\n")
    for r in summary:
        f.write(f"  {r['label']:<18}{r['n']:>3} "
                f"{r['s14_mean']:>7.4f}+/-{r['s14_std']:.4f} "
                f"{r['forge_mean']:>8.4f}+/-{r['forge_std']:.4f} "
                f"{r['ece_mean']:>6.4f}+/-{r['ece_std']:.4f} "
                f"{r['gap_mean']:>5.4f}+/-{r['gap_std']:.4f} "
                f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]\n")

print("Saved: data/urdi_multiseed_report.txt")
print("\n"+"="*65+"\nDone. Share data/urdi_multiseed_report.txt\n"+"="*65)
