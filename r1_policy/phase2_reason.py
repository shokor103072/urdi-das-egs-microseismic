"""
Phase 2 — Reason Layer
Agentic DAS Microseismic Monitoring System

Three components built on top of Phase 1 perception outputs:

  2a. State Tracker
      Rolling event-rate over configurable windows (Δt = 50, 100, 200 triggers).
      Maintains SeismicState per timestep: {event_rate, mean_confidence,
      mean_uncertainty, active_trigger_count}.
      Sensitivity analysis: compares alert lag across window sizes.

  2b. Uncertainty Profiler
      Analyses MC-Dropout std distribution.
      Flags high-uncertainty clusters (bursts of uncertain predictions).
      Calibration check: does high uncertainty correlate with wrong predictions?

  2c. Spatiotemporal Anomaly Scorer
      Detects sudden spikes in event rate (rate-of-change anomaly).
      Detects uncertainty bursts (consecutive high-uncertainty windows).
      Produces per-trigger anomaly_score in [0, 1].

Inputs:
  data/stage14_perception.pkl   (3974 rows, one per trigger)
  data/stage2_perception.pkl    (2016 rows, one per sample)

Outputs:
  data/stage14_reason.pkl       event log + state + anomaly columns
  data/stage2_reason.pkl        same for Stage 2
  data/reason_report.txt        full analysis summary
  data/figures/                 plots for the paper
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_DIR = Path("./data")
FIG_DIR  = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Window sizes to test for state tracker sensitivity analysis
WINDOW_SIZES = [25, 50, 100, 200]

# Thresholds
HIGH_UNC_THRESHOLD   = 0.15   # mc_std above this = high uncertainty
ALERT_RATE_THRESHOLD = 0.70   # event_rate above this triggers Watch
ANOMALY_SPIKE_FACTOR = 2.0    # rate spike > 2x local mean = anomaly


# ═══════════════════════════════════════════════════════════════
# 2a. STATE TRACKER
# ═══════════════════════════════════════════════════════════════
def compute_state_tracker(df: pd.DataFrame,
                           window_sizes: list) -> pd.DataFrame:
    """
    Computes rolling statistics for each window size.
    Uses PREDICTED labels (pred_label) as the agent would at runtime —
    not ground truth labels.

    Returns df with new columns per window size:
      rate_W{w}         rolling event rate (pred_label)
      conf_W{w}         rolling mean confidence
      unc_W{w}          rolling mean uncertainty (mc_std)
    """
    df = df.copy().reset_index(drop=True)

    for w in window_sizes:
        df[f"rate_W{w}"] = (
            df["pred_label"]
            .rolling(window=w, min_periods=1)
            .mean()
            .values
        )
        df[f"conf_W{w}"] = (
            df["confidence"]
            .rolling(window=w, min_periods=1)
            .mean()
            .values
        )
        df[f"unc_W{w}"] = (
            df["mc_std"]
            .rolling(window=w, min_periods=1)
            .mean()
            .values
        )

    return df


def alert_lag_analysis(df: pd.DataFrame,
                        window_sizes: list,
                        threshold: float = ALERT_RATE_THRESHOLD) -> dict:
    """
    For each window size, find the first timestep where the rolling
    event rate exceeds the alert threshold.
    Measures how quickly the state tracker can detect a high-activity period.
    """
    results = {}
    for w in window_sizes:
        col = f"rate_W{w}"
        above = df[col] >= threshold
        first_alert = above.idxmax() if above.any() else None
        results[w] = {
            "first_alert_idx":  int(first_alert) if first_alert is not None else None,
            "pct_above":        float(above.mean()),
            "max_rate":         float(df[col].max()),
            "mean_rate":        float(df[col].mean()),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# 2b. UNCERTAINTY PROFILER
# ═══════════════════════════════════════════════════════════════
def uncertainty_profile(df: pd.DataFrame) -> dict:
    """
    Analyses the MC-Dropout uncertainty distribution.
    Key question: does high uncertainty predict wrong predictions?
    """
    high_unc  = df["mc_std"] > HIGH_UNC_THRESHOLD
    n_high    = high_unc.sum()
    n_total   = len(df)

    # Accuracy split: high uncertainty vs low uncertainty
    acc_high = df.loc[high_unc, "correct"].mean() if n_high > 0 else None
    acc_low  = df.loc[~high_unc, "correct"].mean()

    # Among wrong predictions, what fraction had high uncertainty?
    wrong     = df["correct"] == 0
    n_wrong   = wrong.sum()
    unc_wrong = (high_unc & wrong).sum()
    pct_wrong_with_high_unc = float(unc_wrong / n_wrong) if n_wrong > 0 else 0

    # Uncertainty burst detection: consecutive high-unc windows
    unc_arr    = (df["mc_std"] > HIGH_UNC_THRESHOLD).astype(int).values
    bursts     = []
    in_burst   = False
    burst_start = 0
    for i, v in enumerate(unc_arr):
        if v == 1 and not in_burst:
            in_burst    = True
            burst_start = i
        elif v == 0 and in_burst:
            in_burst = False
            bursts.append((burst_start, i - 1, i - burst_start))
    if in_burst:
        bursts.append((burst_start, len(unc_arr)-1, len(unc_arr)-burst_start))

    return {
        "n_high_unc":               int(n_high),
        "pct_high_unc":             float(n_high / n_total),
        "acc_high_unc":             float(acc_high) if acc_high is not None else None,
        "acc_low_unc":              float(acc_low),
        "pct_wrong_with_high_unc":  pct_wrong_with_high_unc,
        "n_bursts":                 len(bursts),
        "bursts":                   bursts,
        "max_burst_len":            max((b[2] for b in bursts), default=0),
        "mean_mc_std":              float(df["mc_std"].mean()),
        "p95_mc_std":               float(np.percentile(df["mc_std"], 95)),
        "p99_mc_std":               float(np.percentile(df["mc_std"], 99)),
    }


# ═══════════════════════════════════════════════════════════════
# 2c. SPATIOTEMPORAL ANOMALY SCORER
# ═══════════════════════════════════════════════════════════════
def compute_anomaly_scores(df: pd.DataFrame,
                            base_window: int = 50) -> pd.DataFrame:
    """
    Computes a composite anomaly score in [0, 1] per timestep.

    Components:
      1. Rate spike score   : how much current rate exceeds local mean
      2. Uncertainty score  : normalised mc_std
      3. FAR spike score    : sudden increase in false alarm density

    Final anomaly_score = weighted combination of the three.
    """
    df = df.copy()

    # --- 1. Rate spike score ---
    # Smooth rate with short and long windows; spike = short/long ratio
    rate_pred = df["pred_label"].values.astype(float)
    rate_short = uniform_filter1d(rate_pred, size=max(5, base_window//10))
    rate_long  = uniform_filter1d(rate_pred, size=base_window)
    # Avoid division by zero
    spike_ratio = np.where(rate_long > 0.05, rate_short / rate_long, 1.0)
    # Normalise: spike > 2× local mean → score approaches 1
    rate_spike_score = np.clip((spike_ratio - 1.0) / (ANOMALY_SPIKE_FACTOR - 1.0),
                                0.0, 1.0)

    # --- 2. Uncertainty score ---
    unc_vals  = df["mc_std"].values
    unc_score = np.clip(unc_vals / HIGH_UNC_THRESHOLD, 0.0, 1.0)

    # --- 3. Incorrect prediction burst score ---
    # Smooth error signal over short window
    error_signal = (1 - df["correct"].values).astype(float)
    error_smooth  = uniform_filter1d(error_signal, size=max(5, base_window//10))
    error_score   = np.clip(error_smooth / 0.5, 0.0, 1.0)

    # Composite score (weighted sum)
    W_RATE  = 0.5
    W_UNC   = 0.3
    W_ERROR = 0.2
    anomaly_score = (W_RATE  * rate_spike_score +
                     W_UNC   * unc_score         +
                     W_ERROR * error_score)
    anomaly_score = np.clip(anomaly_score, 0.0, 1.0)

    df["rate_spike_score"] = rate_spike_score
    df["unc_score"]        = unc_score
    df["error_score"]      = error_score
    df["anomaly_score"]    = anomaly_score

    # Flag high-anomaly windows
    df["is_anomaly"] = anomaly_score > 0.5

    return df


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
def plot_state_tracker(df: pd.DataFrame, window_sizes: list,
                        stage: str, save_path: Path):
    fig, axes = plt.subplots(len(window_sizes), 1,
                              figsize=(14, 3 * len(window_sizes)),
                              sharex=True)
    if len(window_sizes) == 1:
        axes = [axes]

    x = np.arange(len(df))
    for ax, w in zip(axes, window_sizes):
        rate = df[f"rate_W{w}"].values
        ax.plot(x, rate, linewidth=0.8, color="#1D9E75", label=f"Event rate W={w}")
        ax.axhline(ALERT_RATE_THRESHOLD, color="#E24B4A",
                   linestyle="--", linewidth=1, label=f"Alert threshold ({ALERT_RATE_THRESHOLD})")
        ax.fill_between(x, rate, alpha=0.15, color="#1D9E75")
        ax.set_ylabel(f"Rate W={w}", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Trigger index (chronological)", fontsize=10)
    fig.suptitle(f"State Tracker — Rolling Event Rate  |  {stage}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_uncertainty_distribution(df: pd.DataFrame, stage: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of mc_std
    axes[0].hist(df["mc_std"], bins=60, color="#7F77DD",
                 edgecolor="white", linewidth=0.3)
    axes[0].axvline(HIGH_UNC_THRESHOLD, color="#E24B4A",
                    linestyle="--", label=f"Threshold ({HIGH_UNC_THRESHOLD})")
    axes[0].set_xlabel("MC-Dropout std (epistemic uncertainty)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Uncertainty distribution")
    axes[0].legend()

    # Uncertainty over time
    axes[1].plot(np.arange(len(df)), df["mc_std"].values,
                 linewidth=0.5, color="#7F77DD", alpha=0.7)
    axes[1].axhline(HIGH_UNC_THRESHOLD, color="#E24B4A",
                    linestyle="--", linewidth=1)
    axes[1].set_xlabel("Trigger index (chronological)")
    axes[1].set_ylabel("MC-Dropout std")
    axes[1].set_title("Uncertainty over time")

    fig.suptitle(f"Uncertainty Profile  |  {stage}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_anomaly_scores(df: pd.DataFrame, stage: str, save_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(df))

    # Event rate (W=50)
    w = 50 if f"rate_W50" in df.columns else WINDOW_SIZES[1]
    axes[0].plot(x, df[f"rate_W{w}"].values, color="#1D9E75", linewidth=0.8)
    axes[0].set_ylabel("Event rate")
    axes[0].set_title("Rolling event rate")
    axes[0].grid(alpha=0.3)

    # Uncertainty
    axes[1].plot(x, df["mc_std"].values, color="#7F77DD",
                 linewidth=0.5, alpha=0.8)
    axes[1].axhline(HIGH_UNC_THRESHOLD, color="#E24B4A",
                    linestyle="--", linewidth=1)
    axes[1].set_ylabel("MC-Dropout std")
    axes[1].set_title("Epistemic uncertainty")
    axes[1].grid(alpha=0.3)

    # Anomaly score
    anom = df["anomaly_score"].values
    axes[2].plot(x, anom, color="#D85A30", linewidth=0.8)
    axes[2].fill_between(x, anom, alpha=0.2, color="#D85A30")
    axes[2].axhline(0.5, color="#E24B4A", linestyle="--",
                    linewidth=1, label="Anomaly threshold (0.5)")
    axes[2].set_ylabel("Anomaly score")
    axes[2].set_xlabel("Trigger index (chronological)")
    axes[2].set_title("Composite anomaly score")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Reason Layer Output  |  {stage}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_window_sensitivity(lag_results_14: dict, lag_results_2: dict,
                             save_path: Path):
    ws    = list(lag_results_14.keys())
    lags  = [lag_results_14[w]["first_alert_idx"] or 0 for w in ws]
    rates = [lag_results_14[w]["pct_above"] * 100 for w in ws]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar([str(w) for w in ws], lags,
                color=["#1D9E75", "#7F77DD", "#EF9F27", "#D85A30"])
    axes[0].set_xlabel("Window size (Δt)")
    axes[0].set_ylabel("First alert trigger index")
    axes[0].set_title("Alert lag vs window size — Stage 14")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(lags):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=9)

    pct14 = [lag_results_14[w]["pct_above"] * 100 for w in ws]
    pct2  = [lag_results_2[w]["pct_above"]  * 100 for w in ws]
    x_pos = np.arange(len(ws))
    axes[1].bar(x_pos - 0.2, pct14, width=0.35,
                label="Stage 14", color="#1D9E75")
    axes[1].bar(x_pos + 0.2, pct2,  width=0.35,
                label="Stage 2",  color="#7F77DD")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([str(w) for w in ws])
    axes[1].set_xlabel("Window size (Δt)")
    axes[1].set_ylabel("% timesteps above alert threshold")
    axes[1].set_title("Time above alert threshold by stage")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("State Tracker Sensitivity Analysis", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 2 — Reason Layer")
print("=" * 60)

# ── Load perception outputs ────────────────────────────────────
print("\nLoading Phase 1 outputs...")
df14 = pd.read_pickle(DATA_DIR / "stage14_perception.pkl")
df2  = pd.read_pickle(DATA_DIR / "stage2_perception.pkl")
print(f"  Stage 14 : {len(df14)} triggers")
print(f"  Stage 2  : {len(df2)} samples")

# Add seq_index if not present
if "seq_index" not in df14.columns:
    df14["seq_index"] = np.arange(len(df14))
if "seq_index" not in df2.columns:
    df2["seq_index"] = np.arange(len(df2))


# ── 2a. State Tracker ─────────────────────────────────────────
print("\n[2a] State Tracker...")
df14 = compute_state_tracker(df14, WINDOW_SIZES)
df2  = compute_state_tracker(df2,  WINDOW_SIZES)

lag14 = alert_lag_analysis(df14, WINDOW_SIZES)
lag2  = alert_lag_analysis(df2,  WINDOW_SIZES)

print("\n  Stage 14 — Alert lag by window size:")
for w, res in lag14.items():
    print(f"    W={w:>4} : first_alert={res['first_alert_idx']}  "
          f"pct_above={res['pct_above']*100:.1f}%  "
          f"max_rate={res['max_rate']:.3f}")

print("\n  Stage 2 — Alert lag by window size:")
for w, res in lag2.items():
    print(f"    W={w:>4} : first_alert={res['first_alert_idx']}  "
          f"pct_above={res['pct_above']*100:.1f}%  "
          f"max_rate={res['max_rate']:.3f}")


# ── 2b. Uncertainty Profiler ───────────────────────────────────
print("\n[2b] Uncertainty Profiler...")
unc14 = uncertainty_profile(df14)
unc2  = uncertainty_profile(df2)

print(f"\n  Stage 14:")
print(f"    High-uncertainty windows : {unc14['n_high_unc']} "
      f"({unc14['pct_high_unc']*100:.2f}%)")
print(f"    Accuracy (high unc)      : {unc14['acc_high_unc']}")
print(f"    Accuracy (low unc)       : {unc14['acc_low_unc']:.4f}")
print(f"    Wrong preds with high unc: {unc14['pct_wrong_with_high_unc']*100:.1f}%")
print(f"    Uncertainty bursts       : {unc14['n_bursts']}")
print(f"    Max burst length         : {unc14['max_burst_len']}")
print(f"    p95 mc_std               : {unc14['p95_mc_std']:.4f}")

print(f"\n  Stage 2 (FORGE):")
print(f"    High-uncertainty windows : {unc2['n_high_unc']} "
      f"({unc2['pct_high_unc']*100:.2f}%)")
print(f"    Accuracy (high unc)      : {unc2['acc_high_unc']}")
print(f"    Accuracy (low unc)       : {unc2['acc_low_unc']:.4f}")
print(f"    Wrong preds with high unc: {unc2['pct_wrong_with_high_unc']*100:.1f}%")
print(f"    Uncertainty bursts       : {unc2['n_bursts']}")
print(f"    Max burst length         : {unc2['max_burst_len']}")
print(f"    p95 mc_std               : {unc2['p95_mc_std']:.4f}")


# ── 2c. Anomaly Scorer ────────────────────────────────────────
print("\n[2c] Anomaly Scorer...")
df14 = compute_anomaly_scores(df14, base_window=50)
df2  = compute_anomaly_scores(df2,  base_window=50)

n_anom14 = df14["is_anomaly"].sum()
n_anom2  = df2["is_anomaly"].sum()
print(f"  Stage 14 anomalies  : {n_anom14} / {len(df14)} "
      f"({n_anom14/len(df14)*100:.2f}%)")
print(f"  Stage 2  anomalies  : {n_anom2} / {len(df2)} "
      f"({n_anom2/len(df2)*100:.2f}%)")
print(f"  Stage 14 max score  : {df14['anomaly_score'].max():.4f}")
print(f"  Stage 2  max score  : {df2['anomaly_score'].max():.4f}")


# ── Plots ──────────────────────────────────────────────────────
print("\nGenerating figures...")

plot_state_tracker(df14, WINDOW_SIZES, "Stage 14",
                   FIG_DIR / "state_tracker_stage14.png")
plot_state_tracker(df2,  WINDOW_SIZES, "Stage 2 (FORGE)",
                   FIG_DIR / "state_tracker_stage2.png")
plot_uncertainty_distribution(df14, "Stage 14",
                               FIG_DIR / "uncertainty_stage14.png")
plot_uncertainty_distribution(df2,  "Stage 2 (FORGE)",
                               FIG_DIR / "uncertainty_stage2.png")
plot_anomaly_scores(df14, "Stage 14",
                    FIG_DIR / "anomaly_stage14.png")
plot_anomaly_scores(df2,  "Stage 2 (FORGE)",
                    FIG_DIR / "anomaly_stage2.png")
plot_window_sensitivity(lag14, lag2,
                        FIG_DIR / "window_sensitivity.png")


# ── Save outputs ──────────────────────────────────────────────
print("\nSaving outputs...")

# Final column selection
reason_cols14 = (
    list(df14.columns[:df14.columns.get_loc("p_event") + 7]) +
    [f"rate_W{w}" for w in WINDOW_SIZES] +
    [f"conf_W{w}" for w in WINDOW_SIZES] +
    [f"unc_W{w}"  for w in WINDOW_SIZES] +
    ["rate_spike_score", "unc_score", "error_score",
     "anomaly_score", "is_anomaly"]
)
# Keep only valid columns
reason_cols14 = [c for c in reason_cols14 if c in df14.columns]
reason_cols2  = [c for c in reason_cols14 if c in df2.columns]

df14[reason_cols14].to_pickle(DATA_DIR / "stage14_reason.pkl")
df2[reason_cols2].to_pickle(DATA_DIR / "stage2_reason.pkl")
print(f"  Saved -> data/stage14_reason.pkl  ({len(df14)} rows)")
print(f"  Saved -> data/stage2_reason.pkl   ({len(df2)} rows)")


# ── Reason report ──────────────────────────────────────────────
report = DATA_DIR / "reason_report.txt"
with open(report, "w") as f:
    f.write("PHASE 2 REASON LAYER REPORT\n")
    f.write("=" * 60 + "\n\n")

    for stage_name, lag_res, unc_res, df in [
        ("Stage 14",       lag14, unc14, df14),
        ("Stage 2 (FORGE)", lag2,  unc2,  df2),
    ]:
        f.write(f"{stage_name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total triggers        : {len(df)}\n\n")

        f.write("State Tracker (alert lag analysis):\n")
        for w, res in lag_res.items():
            f.write(f"  W={w:>4}  first_alert={res['first_alert_idx']}  "
                    f"pct_above={res['pct_above']*100:.1f}%  "
                    f"max_rate={res['max_rate']:.3f}\n")
        f.write("\n")

        f.write("Uncertainty Profile:\n")
        f.write(f"  High-unc windows     : {unc_res['n_high_unc']} "
                f"({unc_res['pct_high_unc']*100:.2f}%)\n")
        f.write(f"  Acc (high unc)       : {unc_res['acc_high_unc']}\n")
        f.write(f"  Acc (low unc)        : {unc_res['acc_low_unc']:.4f}\n")
        f.write(f"  % wrong w/ high unc  : {unc_res['pct_wrong_with_high_unc']*100:.1f}%\n")
        f.write(f"  Uncertainty bursts   : {unc_res['n_bursts']}\n")
        f.write(f"  Max burst length     : {unc_res['max_burst_len']}\n")
        f.write(f"  Mean mc_std          : {unc_res['mean_mc_std']:.5f}\n")
        f.write(f"  p95 mc_std           : {unc_res['p95_mc_std']:.5f}\n")
        f.write(f"  p99 mc_std           : {unc_res['p99_mc_std']:.5f}\n\n")

        f.write("Anomaly Scorer:\n")
        n_anom = df["is_anomaly"].sum()
        f.write(f"  Anomalous windows    : {n_anom} ({n_anom/len(df)*100:.2f}%)\n")
        f.write(f"  Max anomaly score    : {df['anomaly_score'].max():.4f}\n")
        f.write(f"  Mean anomaly score   : {df['anomaly_score'].mean():.4f}\n\n")

    f.write("Figures saved to data/figures/:\n")
    for fig in sorted(FIG_DIR.glob("*.png")):
        f.write(f"  {fig.name}\n")

print(f"  Saved -> data/reason_report.txt")

print("\n" + "=" * 60)
print("Phase 2 complete.")
print("Outputs: stage14_reason.pkl | stage2_reason.pkl")
print("Figures: data/figures/")
print("Next -> Phase 3: policy_rule.py + policy_rl.py")
print("=" * 60)
