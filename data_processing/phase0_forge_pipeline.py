"""
Utah FORGE 3-2417 Data Pipeline
Reads pre-cropped SGY files:
  Event : event folder  (1800-3600 ms, ch 1300-3200)
  Noise : noise folder  (0-1800 ms,    ch 1300-3200)

Each file: ~1900 channels x 1800 samples
Processing:
  1. Subsample 1900 -> 361 channels (equidistant)
  2. Zero-pad  1800 -> 2400 samples (center-pad to match Frisco-2-P)
  3. L2 normalize per window
  4. Save X_forge.npy (N, 1, 361, 2400)  y_forge.npy (N,)

Dataset naming:
  Source (training) : Cape EGS Frisco-2-P  (Stage 14, June 2024)
  Target (eval)     : Utah FORGE 3-2417    (July 2023)
"""

import os
import numpy as np
import segyio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

EVENT_DIR = Path(r"E:\New folder\SEG\SEGY\event")
NOISE_DIR = Path(r"E:\New folder\SEG\SEGY\noise")
OUT_DIR   = Path(r"E:\Events_14\Events_14\Data\Dataset")
DATA_DIR  = Path("./data")
FIG_DIR   = DATA_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_CH_TARGET = 361
WIN_LEN     = 2400   # Frisco-2-P window length — pad FORGE to match
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def read_sgy(path):
    try:
        with segyio.open(str(path), "r", ignore_geometry=True) as f:
            return segyio.collect(f.trace[:]).astype(np.float32)
    except Exception:
        with segyio.open(str(path), "r",
                         ignore_geometry=True, strict=False) as f:
            return segyio.collect(f.trace[:]).astype(np.float32)


def subsample_channels(data, n_target=N_CH_TARGET):
    """Equidistant subsample n_target channels from full array."""
    n_ch = data.shape[0]
    idx  = np.linspace(0, n_ch - 1, n_target, dtype=int)
    return data[idx, :]


def pad_to_length(data_ch, target_len=WIN_LEN):
    """
    Center-pad time axis to target_len with zeros.
    data_ch: (n_ch, n_samples) — n_samples <= target_len
    """
    n_ch, n_samples = data_ch.shape
    if n_samples >= target_len:
        # Crop center if somehow too long
        start = (n_samples - target_len) // 2
        return data_ch[:, start:start + target_len]
    pad_total = target_len - n_samples
    pad_left  = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(data_ch, ((0, 0), (pad_left, pad_right)), mode="constant")


def normalize_l2(w):
    """L2 normalization to match Frisco-2-P preprocessing."""
    norm = np.sqrt((w ** 2).sum())
    return w / norm * np.sqrt(w.size) if norm > 1e-10 else w


def process_folder(folder, label, label_name):
    """Read all SGY files from folder, return (X_list, n_ok, n_fail)."""
    files  = sorted(folder.glob("*.sgy"))
    X_list = []
    n_ok   = 0; n_fail = 0

    print(f"\n  Processing {len(files)} {label_name} files from:")
    print(f"    {folder}")

    for i, fpath in enumerate(files):
        try:
            data = read_sgy(fpath)          # (n_ch, n_samples)
        except Exception as e:
            print(f"    [SKIP] {fpath.name}: {e}")
            n_fail += 1
            continue

        data_ch = subsample_channels(data, N_CH_TARGET)  # (361, ~1800)
        data_pad = pad_to_length(data_ch, WIN_LEN)       # (361, 2400)
        data_norm = normalize_l2(data_pad)                # (361, 2400)
        X_list.append(data_norm[np.newaxis])              # (1, 361, 2400)
        n_ok += 1

        if i < 3 or i % 50 == 0:
            print(f"    [{i:>3}] {fpath.name[:45]}  "
                  f"raw={data.shape}  → (361, 2400) ✓")

    return X_list, n_ok, n_fail


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("="*65)
print("Utah FORGE 3-2417 — Data Pipeline")
print("Cape EGS Frisco-2-P → Utah FORGE 3-2417 (cross-domain)")
print("="*65)

# Inspect one file first
sample_file = sorted(EVENT_DIR.glob("*.sgy"))[0]
print(f"\nInspecting: {sample_file.name}")
d_sample = read_sgy(sample_file)
n_ch_raw, n_samp_raw = d_sample.shape
print(f"  Raw shape     : {d_sample.shape}")
print(f"  After subsample: ({N_CH_TARGET}, {n_samp_raw})")
print(f"  After pad      : ({N_CH_TARGET}, {WIN_LEN})")
del d_sample

# Process events
X_ev, n_ev_ok, n_ev_fail = process_folder(EVENT_DIR, 1, "EVENT")

# Process noise
X_no, n_no_ok, n_no_fail = process_folder(NOISE_DIR, 0, "NOISE")

# Assemble
X_forge = np.concatenate(
    [np.stack(X_ev)] + [np.stack(X_no)], axis=0
)  # events first, then noise — consistent with Frisco-2-P structure

n_events = len(X_ev)
n_noise  = len(X_no)
y_forge  = np.array([1]*n_events + [0]*n_noise, dtype=np.int64)

print(f"\nAssembly:")
print(f"  Events : {n_events}  (failed: {n_ev_fail})")
print(f"  Noise  : {n_noise}   (failed: {n_no_fail})")
print(f"  Total  : {len(y_forge)}")
print(f"  Shape  : {X_forge.shape}")
print(f"  Event rate: {y_forge.mean():.4f}")

# Save
np.save(str(OUT_DIR   / "X_forge.npy"), X_forge)
np.save(str(OUT_DIR   / "y_forge.npy"), y_forge)
np.save(str(DATA_DIR  / "X_forge.npy"), X_forge)
np.save(str(DATA_DIR  / "y_forge.npy"), y_forge)
print(f"\n  Saved → {OUT_DIR}/X_forge.npy")
print(f"  Saved → {OUT_DIR}/y_forge.npy")

# ── QC figure ─────────────────────────────────────────────────
print("\nGenerating QC figure...")
fig, axes = plt.subplots(3, 2, figsize=(12, 9))
for col, (label, title, idx_arr) in enumerate([
    (1, "Event (Utah FORGE 3-2417)", np.where(y_forge==1)[0]),
    (0, "Noise (Utah FORGE 3-2417)", np.where(y_forge==0)[0]),
]):
    for row in range(3):
        ax = axes[row][col]
        if row >= len(idx_arr):
            ax.axis("off"); continue
        w    = X_forge[idx_arr[row], 0]   # (361, 2400)
        vmax = max(float(np.abs(w).max()), 0.01)
        ax.imshow(w, aspect="auto", cmap="seismic",
                  vmin=-vmax*0.3, vmax=vmax*0.3,
                  extent=[0, WIN_LEN, N_CH_TARGET, 0])
        ax.set_title(f"{title} [{idx_arr[row]}]", fontsize=8)
        ax.set_xlabel("Time (samples)"); ax.set_ylabel("Channel")

fig.suptitle("Utah FORGE 3-2417 — Processed Windows QC\n"
             f"Channels: 361 subsampled | Time: 2400 samples (center-padded from 1800)",
             fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "forge_qc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: data/figures/forge_qc.png")

# ── Energy distribution ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
E_ev = (X_forge[y_forge==1]**2).sum(axis=(1,2,3))
E_no = (X_forge[y_forge==0]**2).sum(axis=(1,2,3))
ax.hist(E_ev, bins=30, alpha=0.7, color="#1D9E75",
        label=f"Event (n={n_events})")
ax.hist(E_no, bins=30, alpha=0.7, color="#7F77DD",
        label=f"Noise (n={n_noise})")
ax.set_xlabel("Window energy (L2-normalized)")
ax.set_ylabel("Count")
ax.set_title("Utah FORGE 3-2417 — Energy distribution by class")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "forge_energy.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: data/figures/forge_energy.png")

# ── Report ─────────────────────────────────────────────────────
with open(DATA_DIR / "forge_pipeline_report.txt", "w") as f:
    f.write("UTAH FORGE 3-2417 DATA PIPELINE REPORT\n")
    f.write("="*60 + "\n\n")
    f.write("Dataset naming:\n")
    f.write("  Source (training) : Cape EGS Frisco-2-P (Stage 14, June 2024)\n")
    f.write("  Target (eval)     : Utah FORGE 3-2417   (July 2023)\n\n")
    f.write("Pre-processing (external):\n")
    f.write(f"  Event window : 1800–3600 ms, channels 1300–3200\n")
    f.write(f"  Noise window : 0–1800 ms,    channels 1300–3200\n")
    f.write(f"  Raw shape    : ({n_ch_raw}, {n_samp_raw})\n\n")
    f.write("Pipeline processing:\n")
    f.write(f"  Channel subsample : {n_ch_raw} → {N_CH_TARGET} (equidistant)\n")
    f.write(f"  Time padding      : {n_samp_raw} → {WIN_LEN} (center zero-pad)\n")
    f.write(f"  Normalization     : L2 per window\n\n")
    f.write(f"Output:\n")
    f.write(f"  Shape      : {X_forge.shape}\n")
    f.write(f"  Events     : {n_events}\n")
    f.write(f"  Noise      : {n_noise}\n")
    f.write(f"  Event rate : {y_forge.mean():.4f}\n\n")
    f.write("Independence from Frisco-2-P:\n")
    f.write("  Different EGS site (Utah vs Nevada)\n")
    f.write("  Different year (July 2023 vs June 2024)\n")
    f.write("  Different acquisition system (Silixa iDAS)\n")
    f.write("  Different geology and injection zone\n")

print(f"\n  Saved → data/forge_pipeline_report.txt")
print("\n" + "="*65)
print(f"Done: {n_events} events + {n_noise} noise = {len(y_forge)} samples")
print(f"Event rate: {y_forge.mean():.4f}")
print("\nNext: python analysis_arch_comparison.py")
print("="*65)
