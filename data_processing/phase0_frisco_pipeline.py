"""
Phase 0 — Data Pipeline (v4 — Final)
Agentic DAS Microseismic Monitoring System
Cape EGS Frisco-2-P (Stage 14) + Utah FORGE 3-2417 (Stage 2)

Corrected dataset structure:
  Stage 14:
    X.npy  shape  : (35766, 1, 361, 2400) — 35766 windows, 2400 timesteps each
    y.npy  shape  : (3974,)               — one label per SGY trigger (NOT per X row)
    SGY files     : 3112 event + 862 noise = 3974 triggers
    X ordering    : first 28008 rows = events (3112 × 9 rows), next 7758 = noise (862 × 9)
    y ordering    : first 3112 entries = 1, next 862 = 0

  Stage 2 (FORGE):
    X_forge.npy   : (2016, 1, 361, 2400) — one label per row, y_forge aligns 1:1
    y_forge.npy   : (2016,)

Both stages share the same 2400-timestep window length.
Both are segmented into 256-step non-overlapping sub-windows for SE-ResNet compatibility.
  2400 // 256 = 9 sub-windows per original window (96 trailing timesteps discarded)

Outputs saved to ./data/:
  stage14_event_log.pkl       chronological log, one row per 256-step sub-window (321894 rows)
  stage14_X_segmented.npy     (321894, 1, 361, 256)
  stage14_y_segmented.npy     (321894,)
  stage2_event_log.pkl        log for Stage 2 sub-windows (18144 rows)
  stage2_X_segmented.npy      (18144, 1, 361, 256)
  stage2_y_segmented.npy      (18144,)
  pipeline_report.txt
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = Path(r"E:\Events_14\Events_14\Data")
EVENT_DIR   = BASE_DIR / "Event"
NOISE_DIR   = BASE_DIR / "Noise"
DATASET_DIR = BASE_DIR / "Dataset"
MODEL_DIR   = BASE_DIR / "Model"

NPY_X14  = DATASET_DIR / "X.npy"
NPY_Y14  = DATASET_DIR / "y.npy"
NPY_X2   = DATASET_DIR / "X_forge.npy"
NPY_Y2   = DATASET_DIR / "y_forge.npy"

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

TRIGGER_PATTERN = re.compile(r"_(\d+)\.sgy$", re.IGNORECASE)
WIN_SIZE = 256    # timesteps SE-ResNet was trained on
ROWS_PER_TRIGGER = 9   # X14 rows per SGY trigger


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def scan_sgy_folder(folder: Path, label: int) -> pd.DataFrame:
    records = []
    for fname in sorted(folder.glob("*.sgy")):
        m = TRIGGER_PATTERN.search(fname.name)
        if m:
            records.append({
                "filename":   fname.name,
                "trigger_id": int(m.group(1)),
                "label":      label,
                "label_name": "event" if label == 1 else "noise",
            })
    return pd.DataFrame(records).sort_values("trigger_id").reset_index(drop=True)


def segment_array(X: np.ndarray, y_trigger: np.ndarray,
                  win_size: int, rows_per_trigger: int = 1):
    """
    Segments X of shape (N, 1, C, T) into sub-windows of win_size.
    y_trigger: one label per trigger (length = N // rows_per_trigger).

    Returns:
      X_seg : (N * segs_per_row, 1, C, win_size)
      y_seg : (N * segs_per_row,)
      records: list of dicts for building the event log
    """
    n_total, _, n_ch, n_ts = X.shape
    segs_per_row = n_ts // WIN_SIZE
    n_triggers   = n_total // rows_per_trigger
    n_out        = n_total * segs_per_row

    X_seg   = np.empty((n_out, 1, n_ch, win_size), dtype=X.dtype)
    y_seg   = np.empty(n_out, dtype=np.int64)
    records = []

    out_idx = 0
    for x_row in range(n_total):
        trigger_idx = x_row // rows_per_trigger
        row_in_trigger = x_row % rows_per_trigger
        label = int(y_trigger[trigger_idx])
        for s in range(segs_per_row):
            t0 = s * win_size
            X_seg[out_idx]  = X[x_row, :, :, t0:t0 + win_size]
            y_seg[out_idx]  = label
            records.append({
                "npy_index":       out_idx,
                "source_x_row":    x_row,
                "trigger_idx":     trigger_idx,
                "row_in_trigger":  row_in_trigger,
                "segment":         s,
                "t_start":         t0,
                "label":           label,
                "label_name":      "event" if label == 1 else "noise",
            })
            out_idx += 1

    return X_seg, y_seg, records


# ═══════════════════════════════════════════════════════════════
# STAGE 14
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 0 — Data Pipeline  |  Stage 14 + Stage 2 (FORGE)")
print("=" * 60)

print("\n── STAGE 14 ─────────────────────────────────────────────")

print("\n[S14 1/4] Scanning SGY folders...")
df_ev14 = scan_sgy_folder(EVENT_DIR, label=1)
df_no14 = scan_sgy_folder(NOISE_DIR, label=0)
n_ev_files = len(df_ev14)
n_no_files = len(df_no14)
print(f"  Event SGY triggers : {n_ev_files}")
print(f"  Noise SGY triggers : {n_no_files}")
print(f"  Total triggers     : {n_ev_files + n_no_files}")

print("\n[S14 2/4] Loading numpy arrays...")
X14 = np.load(NPY_X14, mmap_mode="r")
y14 = np.load(NPY_Y14)
print(f"  X14 shape : {X14.shape}  dtype: {X14.dtype}")
print(f"  y14 shape : {y14.shape}  dtype: {y14.dtype}")

n_rows14, _, n_ch14, n_ts14 = X14.shape
n_triggers14 = len(y14)
rows_per_t14 = n_rows14 // n_triggers14
segs_per_row14 = n_ts14 // WIN_SIZE
leftover14 = n_ts14 % WIN_SIZE

print(f"  Rows per trigger   : {rows_per_t14}  ({n_rows14} / {n_triggers14})")
print(f"  Segs per row       : {segs_per_row14}  ({WIN_SIZE}-step windows from {n_ts14} timesteps)")
print(f"  Trailing timesteps : {leftover14} (discarded)")

assert n_rows14 % n_triggers14 == 0, \
    f"X14 rows {n_rows14} not divisible by trigger count {n_triggers14}"
assert n_ev_files + n_no_files == n_triggers14, \
    f"SGY file count {n_ev_files+n_no_files} != y14 length {n_triggers14}"
print("  Alignment verified OK")

# Build per-trigger y array aligned to X14's row ordering
# X14 ordering: events first (sorted by trigger ID), then noise
# y14 ordering: first n_ev_files entries = 1, remaining = 0
# We derive labels from folder structure (more reliable than y14 values)
y14_per_trigger = np.array(
    [1] * n_ev_files + [0] * n_no_files, dtype=np.int64
)

# Verify against y14
n_ev_y14 = int((y14 == 1).sum())
n_no_y14 = int((y14 == 0).sum())
print(f"  y14 label dist     : {{1: {n_ev_y14}, 0: {n_no_y14}}}")
assert n_ev_y14 == n_ev_files and n_no_y14 == n_no_files, \
    f"Label count mismatch: SGY has {n_ev_files}E/{n_no_files}N, y14 has {n_ev_y14}E/{n_no_y14}N"
print("  Label counts match y14 OK")

print("\n[S14 3/4] Segmenting Stage 14 into 256-step windows (loading full array)...")
X14_full = np.load(NPY_X14)
X14_seg, y14_seg, records14 = segment_array(
    X14_full, y14_per_trigger, WIN_SIZE, rows_per_t14
)
del X14_full
print(f"  Output shape       : {X14_seg.shape}")

# Build chronological event log
df14 = pd.DataFrame(records14)

# Map trigger_idx to trigger_id (events 0..3111, noise 3112..3973)
trigger_ids_ordered = (
    df_ev14["trigger_id"].tolist() + df_no14["trigger_id"].tolist()
)
filenames_ordered = (
    df_ev14["filename"].tolist() + df_no14["filename"].tolist()
)
df14["trigger_id"] = df14["trigger_idx"].map(
    dict(enumerate(trigger_ids_ordered))
)
df14["filename"] = df14["trigger_idx"].map(
    dict(enumerate(filenames_ordered))
)
df14["stage"] = 14

# Sort chronologically by trigger_id, then row_in_trigger, then segment
df14 = df14.sort_values(
    ["trigger_id", "row_in_trigger", "segment"]
).reset_index(drop=True)
df14["seq_index"]     = df14.index
df14["relative_time"] = df14["trigger_id"] - df14["trigger_id"].min()

# Trigger gap
uniq14 = df14.drop_duplicates("trigger_id")[["trigger_id"]].copy()
uniq14["trigger_gap"] = uniq14["trigger_id"].diff().fillna(0).astype(int)
gap_map14 = uniq14.set_index("trigger_id")["trigger_gap"]
df14["trigger_gap"]    = df14["trigger_id"].map(gap_map14).fillna(0).astype(int)
df14["is_quiet_period"] = df14["trigger_gap"] > 10

# Rolling event rate
df14["rolling_event_rate"] = (
    df14["label"].rolling(window=50, min_periods=1).mean().values
)

df14 = df14[[
    "seq_index", "trigger_id", "trigger_idx", "row_in_trigger", "segment",
    "relative_time", "trigger_gap", "is_quiet_period",
    "label", "label_name", "stage", "rolling_event_rate",
    "npy_index", "filename",
]]

print(f"  Total sub-windows  : {len(df14)}")
print(f"  Event rate         : {df14['label'].mean():.4f}")
print(f"  Quiet windows      : {df14['is_quiet_period'].sum()}")

print("\n[S14 4/4] Saving Stage 14 outputs...")
df14.to_pickle(OUTPUT_DIR / "stage14_event_log.pkl")
np.save(OUTPUT_DIR / "stage14_X_segmented.npy", X14_seg)
np.save(OUTPUT_DIR / "stage14_y_segmented.npy", y14_seg)
print(f"  Saved event log    -> data/stage14_event_log.pkl")
print(f"  Saved X14_seg      -> {X14_seg.shape}")
print(f"  Saved y14_seg      -> {y14_seg.shape}")
del X14_seg


# ═══════════════════════════════════════════════════════════════
# STAGE 2 (FORGE)
# ═══════════════════════════════════════════════════════════════
print("\n── STAGE 2 (FORGE) ──────────────────────────────────────")

print("\n[S2 1/3] Loading Stage 2 arrays...")
X2_full = np.load(NPY_X2)
y2      = np.load(NPY_Y2)
print(f"  X2 shape : {X2_full.shape}  dtype: {X2_full.dtype}")
print(f"  y2 shape : {y2.shape}  dtype: {y2.dtype}")

n_samp2, _, n_ch2, n_ts2 = X2_full.shape
n_ev2  = int((y2 == 1).sum())
n_no2  = int((y2 == 0).sum())
segs2  = n_ts2 // WIN_SIZE
print(f"  Events   : {n_ev2}  Noise : {n_no2}")
print(f"  Segs per sample : {segs2}  (trailing {n_ts2 % WIN_SIZE} timesteps discarded)")

# Stage 2: y2 aligns 1:1 with X2 rows (no rows_per_trigger issue)
print("\n[S2 2/3] Segmenting Stage 2 into 256-step windows...")
X2_seg, y2_seg, records2 = segment_array(X2_full, y2, WIN_SIZE, rows_per_trigger=1)
del X2_full
print(f"  Output shape : {X2_seg.shape}")

df2 = pd.DataFrame(records2)
df2 = df2.rename(columns={"trigger_idx": "source_sample"})
df2["stage"]         = 2
df2["seq_index"]     = df2.index
df2["relative_time"] = df2["source_sample"]
df2["rolling_event_rate"] = (
    df2["label"].rolling(window=50, min_periods=1).mean().values
)

df2 = df2[[
    "seq_index", "source_sample", "segment", "relative_time",
    "label", "label_name", "stage", "rolling_event_rate", "npy_index",
]]

print(f"  Total sub-windows : {len(df2)}")
print(f"  Event rate        : {df2['label'].mean():.4f}")

print("\n[S2 3/3] Saving Stage 2 outputs...")
df2.to_pickle(OUTPUT_DIR / "stage2_event_log.pkl")
np.save(OUTPUT_DIR / "stage2_X_segmented.npy", X2_seg)
np.save(OUTPUT_DIR / "stage2_y_segmented.npy", y2_seg)
print(f"  Saved event log   -> data/stage2_event_log.pkl")
print(f"  Saved X2_seg      -> {X2_seg.shape}")
print(f"  Saved y2_seg      -> {y2_seg.shape}")
del X2_seg


# ═══════════════════════════════════════════════════════════════
# PIPELINE REPORT
# ═══════════════════════════════════════════════════════════════
report = OUTPUT_DIR / "pipeline_report.txt"
with open(report, "w") as f:
    f.write("PHASE 0 PIPELINE REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("STAGE 14 (Cape EGS Frisco-2-P)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Event SGY triggers       : {n_ev_files}\n")
    f.write(f"Noise SGY triggers       : {n_no_files}\n")
    f.write(f"X14 rows per trigger     : {rows_per_t14}\n")
    f.write(f"Segs per row (256-step)  : {segs_per_row14}\n")
    f.write(f"Total sub-windows        : {len(df14)}\n")
    f.write(f"X14_seg shape            : {len(df14), 1, n_ch14, WIN_SIZE}\n")
    f.write(f"Trigger ID range         : {df14['trigger_id'].min()} -> {df14['trigger_id'].max()}\n")
    f.write(f"Overall event rate       : {df14['label'].mean():.4f}\n")
    f.write(f"Quiet period windows     : {df14['is_quiet_period'].sum()}\n\n")

    f.write("STAGE 2 (Utah FORGE 3-2417, July 2023)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Original samples         : {n_samp2}\n")
    f.write(f"Original timesteps       : {n_ts2}\n")
    f.write(f"Segs per sample (256-step): {segs2}\n")
    f.write(f"Total sub-windows        : {len(df2)}\n")
    f.write(f"X2_seg shape             : {len(df2), 1, n_ch2, WIN_SIZE}\n")
    f.write(f"Events (original)        : {n_ev2}\n")
    f.write(f"Noise  (original)        : {n_no2}\n")
    f.write(f"Overall event rate       : {df2['label'].mean():.4f}\n\n")

    f.write("MODEL WEIGHTS\n")
    f.write("-" * 40 + "\n")
    for p in sorted(MODEL_DIR.glob("*.pth")):
        f.write(f"  {p.name:<30} {p.stat().st_size//1024:>8} KB\n")

    f.write("\nStage 14 event log — first 10 rows:\n")
    f.write(df14.head(10).to_string(index=False))
    f.write("\n\nStage 2 event log — first 10 rows:\n")
    f.write(df2.head(10).to_string(index=False))

print(f"\n  Saved report      -> data/pipeline_report.txt")

print("\n" + "=" * 60)
print("Phase 0 complete.")
print("")
print("data/ contents:")
print(f"  stage14_event_log.pkl      {len(df14)} rows (chronological sub-windows)")
print(f"  stage14_X_segmented.npy    ({len(df14)}, 1, {n_ch14}, {WIN_SIZE})")
print(f"  stage14_y_segmented.npy    ({len(df14)},)")
print(f"  stage2_event_log.pkl       {len(df2)} rows")
print(f"  stage2_X_segmented.npy     ({len(df2)}, 1, {n_ch2}, {WIN_SIZE})")
print(f"  stage2_y_segmented.npy     ({len(df2)},)")
print("")
print("Next -> Phase 1: perception.py")
print("=" * 60)
