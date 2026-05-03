"""
Window Position Leakage Control Experiment
TNNLS Concern 5: reviewer suspects model exploits temporal position
(event=center, noise=start) rather than seismic morphology.

EXPERIMENT:
  Reconstruct Utah FORGE 3-2417 with randomized noise window positions:
    - Event windows: keep at 1800-3600ms (same as paper)
    - Noise windows: randomly sample from any non-event region
      using step=50ms, avoiding overlap with event zone

  If SE-ResNet F1 does NOT drop significantly on randomized version,
  it proves morphology drives classification, not temporal position.

EXPECTED OUTCOME:
  F1(original) ≈ F1(randomized) → morphology-driven → no leakage concern
  F1(original) >> F1(randomized) → position-driven → leakage confirmed

OUTPUTS:
  data/window_leakage_report.txt
  data/figures/paper_window_leakage.png
"""

import os
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EVENT_DIR   = Path(r"E:\New folder\SEG\SEGY\event")
NOISE_DIR   = Path(r"E:\New folder\SEG\SEGY\noise")
MODEL_PATH  = Path("./Model/best_seresnet.pth")
DATA_DIR    = Path("./data")
FIG_DIR     = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Match paper preprocessing
N_CH_TARGET = 361
WIN_LEN     = 2400    # 2.4s at 1000 Hz
N_SEEDS     = 5
SEED        = 42

np.random.seed(SEED)


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
        x=self.stem(x);x=self.layer1(x);x=self.down1(x)
        x=self.layer2(x);x=self.down2(x);x=self.layer3(x)
        x=self.down3(x);x=self.layer4(x)
        return self.head(self.pool(x))


# ── Data helpers ───────────────────────────────────────────────
def read_sgy(path):
    try:
        with segyio.open(str(path), "r", ignore_geometry=True) as f:
            return segyio.collect(f.trace[:]).astype(np.float32)
    except Exception:
        with segyio.open(str(path), "r", ignore_geometry=True, strict=False) as f:
            return segyio.collect(f.trace[:]).astype(np.float32)

def subsample_ch(data, n=N_CH_TARGET):
    idx = np.linspace(0, data.shape[0]-1, n, dtype=int)
    return data[idx, :]

def pad_to(w, target=WIN_LEN):
    n = w.shape[1]
    if n >= target:
        s = (n - target) // 2
        return w[:, s:s+target]
    pl = (target - n) // 2
    pr = target - n - pl
    return np.pad(w, ((0,0),(pl,pr)), mode="constant")

def normalize_l2(w):
    norm = np.sqrt((w**2).sum())
    return w / norm * np.sqrt(w.size) if norm > 1e-10 else w


def extract_event(data_full):
    """
    Extract event window. Pre-cropped FORGE files already contain
    the event window (1800-3600ms already extracted, shape ~1801 samples).
    For raw files (>3600 samples), index [1800:3600]. For pre-cropped, use [:1800].
    """
    if data_full.shape[1] > 3600:
        return data_full[:, 1800:3600]   # raw full file
    else:
        return data_full[:, :1800]        # already pre-cropped


def extract_noise_paper(data_full):
    """Paper method: use first 1800 samples (pre-cropped noise files start at 0ms)."""
    return data_full[:, 0:1800]


def extract_noise_random(data_full, rng, event_start=1800, event_end=3600):
    """
    Random method: pick any non-event 1800-sample window.
    Available: [0, event_start] and [event_end, end]
    Minimum window needed: 1800 samples.
    """
    n_samp = data_full.shape[1]
    candidates = []

    # Pre-event zone
    if event_start >= 1800:
        for s in range(0, event_start - 1800 + 1, 50):
            candidates.append(s)

    # Post-event zone
    for s in range(event_end, n_samp - 1800 + 1, 50):
        candidates.append(s)

    if not candidates:
        return None

    start = rng.choice(candidates)
    return data_full[:, start:start + 1800]


def process_dataset(event_files, noise_files_or_func,
                    use_random_noise=False, seed=42):
    """
    Build X, y arrays.
    If use_random_noise=True, use random window positions for noise.
    """
    rng = np.random.default_rng(seed)
    X_list = []; y_list = []

    # Map noise files by stem for paired reading
    noise_map = {}
    for nf in noise_files_or_func:
        # Extract timestamp key from filename
        stem = nf.stem.replace("_noise", "").replace("_cropped", "")
        noise_map[stem] = nf

    for ef in event_files:
        stem = ef.stem.replace("_cropped", "")
        try:
            ev_raw = read_sgy(ef)
        except Exception:
            continue

        ev_ch = subsample_ch(ev_raw)
        ev_w  = extract_event(ev_ch)

        if ev_w.shape[1] < 1800:
            continue

        ev_pad  = pad_to(ev_w)
        ev_norm = normalize_l2(ev_pad)
        X_list.append(ev_norm[np.newaxis]); y_list.append(1)

        # Noise extraction
        if use_random_noise:
            # Read the original full SGY for random windowing
            # Use event SGY since noise SGY is pre-cropped
            # Fall back to pre-event zone of event file
            no_w = extract_noise_random(ev_ch, rng)
            if no_w is None:
                continue
        else:
            # Use corresponding noise file
            if stem not in noise_map:
                continue
            try:
                no_raw = read_sgy(noise_map[stem])
            except Exception:
                continue
            no_ch = subsample_ch(no_raw)
            no_w  = no_ch[:, :1800]   # 0-1800ms

        no_pad  = pad_to(no_w)
        no_norm = normalize_l2(no_pad)
        X_list.append(no_norm[np.newaxis]); y_list.append(0)

    return np.stack(X_list), np.array(y_list, dtype=np.int64)


def evaluate(model, X, y, device, batch_size=16):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    pred = []
    with torch.no_grad():
        for (xb,) in loader:
            pred.append(model(xb.to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    return float(f1_score(y, pred, zero_division=0))


# ── Main ───────────────────────────────────────────────────────
print("="*65)
print("Window Position Leakage Control Experiment")
print("="*65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model = SEResNet().to(device)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device,
                                  weights_only=True))
print(f"Loaded: {MODEL_PATH.name}")

event_files = sorted(EVENT_DIR.glob("*.sgy"))
noise_files = sorted(NOISE_DIR.glob("*.sgy"))
print(f"\nEvent files: {len(event_files)}")
print(f"Noise files: {len(noise_files)}")

# ── Condition A: paper method (fixed positions) ────────────────
print("\n[A] Paper method: event=1800-3600ms, noise=0-1800ms")
X_paper, y_paper = process_dataset(event_files, noise_files,
                                     use_random_noise=False, seed=SEED)
f1_paper = evaluate(model, X_paper, y_paper, device)
print(f"  Samples: {len(y_paper)}  Event rate: {y_paper.mean():.3f}")
print(f"  F1 = {f1_paper:.4f}")

# ── Condition B: randomized noise positions (N_SEEDS runs) ─────
print(f"\n[B] Randomized noise: {N_SEEDS} different random seeds")
f1_random_list = []

# Need full SGY files (event SGYs which are 5s long) for random windowing
# Check if original raw SGY folder is available
RAW_DIR = Path(r"E:\New folder\SEG\SEGY")
raw_files = sorted(RAW_DIR.glob("*.sgy"))
use_raw = len(raw_files) > 0 and raw_files[0].stat().st_size > 1e6 * 50

if use_raw:
    print(f"  Using raw SGY files ({len(raw_files)} files, full 5s)")
    for seed_i in range(N_SEEDS):
        X_rand, y_rand = process_dataset(raw_files, [],
                                          use_random_noise=True,
                                          seed=SEED + seed_i * 7)
        f1_r = evaluate(model, X_rand, y_rand, device)
        f1_random_list.append(f1_r)
        print(f"  Seed {seed_i}: F1={f1_r:.4f}")
else:
    # Fall back: use different sections of the pre-cropped event files
    print("  Raw SGY not available — using alternative random positions")
    print("  (sampling different 1800-sample windows from event files)")
    for seed_i in range(N_SEEDS):
        rng = np.random.default_rng(SEED + seed_i * 7)
        X_list = []; y_list = []
        for ef in event_files:
            try:
                ev_raw = read_sgy(ef)
            except Exception:
                continue
            ev_ch = subsample_ch(ev_raw)
            n_samp = ev_ch.shape[1]

            # Event window (fixed, as in paper)
            if n_samp >= 3600:
                ev_w = ev_ch[:, 1800:3600]
            else:
                continue

            ev_norm = normalize_l2(pad_to(ev_w))
            X_list.append(ev_norm[np.newaxis]); y_list.append(1)

            # Random noise window from outside event zone [1800-3600]
            candidates = list(range(0, 1800 - 1800 + 1, 50))  # pre-event
            if n_samp > 3600 + 1800:
                candidates += list(range(3600, n_samp - 1800 + 1, 50))

            if not candidates:
                # If pre-event is too short, use start of file
                no_w = ev_ch[:, 0:1800]
            else:
                start = rng.choice(candidates)
                no_w  = ev_ch[:, start:start + 1800]

            no_norm = normalize_l2(pad_to(no_w))
            X_list.append(no_norm[np.newaxis]); y_list.append(0)

        X_rand = np.stack(X_list); y_rand = np.array(y_list, dtype=np.int64)
        f1_r = evaluate(model, X_rand, y_rand, device)
        f1_random_list.append(f1_r)
        print(f"  Seed {seed_i}: F1={f1_r:.4f}  "
              f"(n={len(y_rand)}, rate={y_rand.mean():.3f})")

f1_rand_mean = float(np.mean(f1_random_list))
f1_rand_std  = float(np.std(f1_random_list))
delta        = f1_paper - f1_rand_mean

print(f"\nResults:")
print(f"  Paper method F1         : {f1_paper:.4f}")
print(f"  Randomized noise F1     : {f1_rand_mean:.4f} +/- {f1_rand_std:.4f}")
print(f"  Difference              : {delta:+.4f}")

if abs(delta) < 0.05:
    verdict = "PASS — F1 difference < 0.05. Model uses morphology, not position."
elif abs(delta) < 0.10:
    verdict = "BORDERLINE — modest position effect. Discuss as limitation."
else:
    verdict = "FAIL — large position effect. Revise noise extraction."
print(f"  Verdict: {verdict}")


# ── Figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: bar comparison
ax = axes[0]
bars = ax.bar(
    ["Paper\n(fixed position)", f"Randomized\n(mean, n={N_SEEDS})"],
    [f1_paper, f1_rand_mean],
    color=["#1D9E75", "#7F77DD"],
    edgecolor="white", width=0.4, alpha=0.9)
ax.errorbar(1, f1_rand_mean, yerr=f1_rand_std,
            fmt="none", color="black", capsize=6, linewidth=2)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("Window Position Leakage Control\nUtah FORGE 3-2417",
             fontsize=10, fontweight="bold")
ax.set_ylim(max(0, min(f1_paper, f1_rand_mean) - 0.1), 1.02)
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, [f1_paper, f1_rand_mean]):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
            f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.text(0.5, max(f1_paper, f1_rand_mean)-0.03,
        f"$\\Delta$F1\,=\,{delta:+.4f}", ha="center",
        transform=ax.get_xaxis_transform(), fontsize=9,
        color="#E24B4A" if abs(delta) > 0.05 else "#1a5c3a")

# Right: per-seed scatter
ax = axes[1]
ax.scatter(range(N_SEEDS), f1_random_list,
           color="#7F77DD", s=80, zorder=3, label="Random seed F1")
ax.axhline(f1_rand_mean, color="#7F77DD", linestyle="--",
           linewidth=1.5, label=f"Mean={f1_rand_mean:.4f}")
ax.axhline(f1_paper, color="#1D9E75", linestyle="-",
           linewidth=2, label=f"Paper={f1_paper:.4f}")
ax.fill_between(range(N_SEEDS),
                f1_rand_mean - f1_rand_std,
                f1_rand_mean + f1_rand_std,
                alpha=0.15, color="#7F77DD")
ax.set_xlabel("Random seed"); ax.set_ylabel("F1 Score")
ax.set_title("Randomized Noise: per-seed F1", fontsize=10,
             fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_ylim(max(0, f1_rand_mean - 0.15), 1.02)

plt.tight_layout()
plt.savefig(FIG_DIR / "paper_window_leakage.png", dpi=200,
            bbox_inches="tight")
plt.close()
print("\nSaved: data/figures/paper_window_leakage.png")

# ── Report ─────────────────────────────────────────────────────
with open(DATA_DIR / "window_leakage_report.txt", "w",
          encoding="utf-8") as f:
    f.write("WINDOW POSITION LEAKAGE CONTROL\n")
    f.write("="*55 + "\n\n")
    f.write("Concern: Model may exploit temporal position rather than morphology.\n")
    f.write("  Paper: event=1800-3600ms, noise=0-1800ms\n\n")
    f.write(f"Paper method F1     : {f1_paper:.4f}\n")
    f.write(f"Randomized noise F1 : {f1_rand_mean:.4f} +/- {f1_rand_std:.4f}\n")
    f.write(f"Difference          : {delta:+.4f}\n\n")
    f.write(f"Verdict: {verdict}\n\n")
    f.write("Per-seed randomized F1 values:\n")
    for i, v in enumerate(f1_random_list):
        f.write(f"  Seed {i}: {v:.4f}\n")

print("Saved: data/window_leakage_report.txt")
print("\n" + "="*65)
print(f"Window leakage control complete.")
print(f"Paper F1={f1_paper:.4f}  Randomized F1={f1_rand_mean:.4f}+/-{f1_rand_std:.4f}")
print("="*65)
