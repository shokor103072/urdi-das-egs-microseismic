"""
Phase 4 — Act Layer
Agentic DAS Microseismic Monitoring System

Three components that convert policy decisions into operational outputs:

  4a. Alert Dispatcher
      Maps {Watch, Caution, Halt} → structured alert with:
        - Hazard level (LOW / MODERATE / HIGH)
        - Recommended operator action
        - Justification string (rate, uncertainty, anomaly values)
        - Confidence of the alert decision

  4b. Injection Advisor
      Maps policy action + seismic state → injection rate recommendation:
        Watch   → MAINTAIN current rate
        Caution → REDUCE by 20%
        Halt    → SUSPEND injection

  4c. Feedback & Selective Retraining Flag
      Simulates operator confirmation loop:
        - High-uncertainty windows → flagged for operator review
        - If operator overrides decision → logged as correction
        - If correction rate > threshold → triggers retraining flag
      Uses MC-Dropout uncertainty as the triage signal.

Inputs:
  data/stage14_plan.pkl
  data/stage2_plan.pkl

Outputs:
  data/stage14_act.pkl          full agent output log
  data/stage2_act.pkl
  data/act_report.txt           operational summary
  data/figures/act_*.png        alert timeline figures
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_DIR = Path("./data")
FIG_DIR  = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
HIGH_UNC_THRESHOLD    = 0.15   # mc_std above this → flagged for review
CORRECTION_RATE_LIMIT = 0.05   # >5% correction rate → trigger retraining flag

# Injection rate recommendation multipliers
INJECTION_RATES = {
    "MAINTAIN":  1.00,
    "REDUCE":    0.80,
    "SUSPEND":   0.00,
}

# Alert level mapping
ALERT_LEVELS = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH",
}

ALERT_COLORS = {
    "LOW":      "#1D9E75",
    "MODERATE": "#EF9F27",
    "HIGH":     "#E24B4A",
}


# ═══════════════════════════════════════════════════════════════
# 4a. ALERT DISPATCHER
# ═══════════════════════════════════════════════════════════════
def build_alert(action: int, rate: float, unc: float,
                anomaly: float, confidence: float) -> dict:
    """
    Builds a structured alert record for one trigger window.
    Uses the RL policy action as primary decision.
    """
    level    = ALERT_LEVELS[action]
    inj_rec  = ["MAINTAIN", "REDUCE", "SUSPEND"][action]

    # Justification string
    parts = []
    if rate >= 0.70:
        parts.append(f"event_rate={rate:.2f} (HIGH)")
    elif rate >= 0.40:
        parts.append(f"event_rate={rate:.2f} (MODERATE)")
    else:
        parts.append(f"event_rate={rate:.2f} (LOW)")
    if unc >= HIGH_UNC_THRESHOLD:
        parts.append(f"uncertainty={unc:.3f} (ELEVATED)")
    if anomaly >= 0.50:
        parts.append(f"anomaly_score={anomaly:.3f} (ANOMALY DETECTED)")
    justification = " | ".join(parts)

    return {
        "hazard_level":     level,
        "injection_rec":    inj_rec,
        "inj_multiplier":   INJECTION_RATES[inj_rec],
        "justification":    justification,
        "alert_confidence": float(confidence),
    }


def dispatch_alerts(df: pd.DataFrame,
                     action_col: str = "action_rl",
                     window: int = 50) -> pd.DataFrame:
    """Applies alert dispatcher to every row in df."""
    rate_col = f"rate_W{window}" if f"rate_W{window}" in df.columns else "rate_W50"

    alerts = []
    for _, row in df.iterrows():
        alert = build_alert(
            action    = int(row[action_col]),
            rate      = float(row[rate_col]),
            unc       = float(row["mc_std"]),
            anomaly   = float(row["anomaly_score"]),
            confidence= float(row["confidence"]),
        )
        alerts.append(alert)

    df_alert = pd.DataFrame(alerts)
    return pd.concat([df.reset_index(drop=True),
                      df_alert.reset_index(drop=True)], axis=1)


# ═══════════════════════════════════════════════════════════════
# 4b. INJECTION ADVISOR — summary statistics
# ═══════════════════════════════════════════════════════════════
def injection_advisory_summary(df: pd.DataFrame) -> dict:
    """
    Computes what fraction of the operational sequence each
    injection recommendation is active.
    """
    counts = df["injection_rec"].value_counts()
    total  = len(df)
    return {
        rec: {
            "count":   int(counts.get(rec, 0)),
            "pct":     float(counts.get(rec, 0) / total * 100),
        }
        for rec in ["MAINTAIN", "REDUCE", "SUSPEND"]
    }


def compute_simulated_injection_rate(df: pd.DataFrame) -> np.ndarray:
    """
    Simulates a normalised injection rate timeline based on agent recommendations.
    Starts at 1.0, multiplied by inj_multiplier each step.
    Rate recovers gradually when not suspended (operator resumes).
    """
    mults = df["inj_multiplier"].values
    rate  = np.zeros(len(mults))
    cur   = 1.0
    for i, m in enumerate(mults):
        if m == 0.0:
            cur = 0.0
        elif m == 0.80:
            cur = max(cur * 0.80, 0.20)
        else:
            cur = min(cur * 1.05, 1.0)  # gradual recovery
        rate[i] = cur
    return rate


# ═══════════════════════════════════════════════════════════════
# 4c. FEEDBACK & SELECTIVE RETRAINING FLAG
# ═══════════════════════════════════════════════════════════════
def simulate_operator_feedback(df: pd.DataFrame,
                                 action_col: str = "action_rl",
                                 unc_col: str = "mc_std") -> pd.DataFrame:
    """
    Simulates the operator confirmation loop.

    Logic:
      1. Flag windows where mc_std > HIGH_UNC_THRESHOLD for operator review.
      2. For flagged windows: if prediction is also wrong (correct==0),
         operator overrides → logged as correction.
      3. If running correction rate > CORRECTION_RATE_LIMIT → retrain_flag=True.

    Returns df with feedback columns added.
    """
    df = df.copy()

    flagged        = (df[unc_col] > HIGH_UNC_THRESHOLD).values
    is_wrong       = (df["correct"] == 0).values if "correct" in df.columns else \
                     (df["rl_correct"] == 0).values
    corrections    = flagged & is_wrong
    retrain_flags  = np.zeros(len(df), dtype=bool)

    # Compute running correction rate over a 100-trigger window
    corr_arr    = corrections.astype(float)
    running_rate = pd.Series(corr_arr).rolling(100, min_periods=1).mean().values
    retrain_flags = running_rate > CORRECTION_RATE_LIMIT

    df["flagged_for_review"]   = flagged
    df["operator_correction"]  = corrections
    df["running_correction_rate"] = running_rate
    df["retrain_flag"]         = retrain_flags

    return df


def feedback_summary(df: pd.DataFrame) -> dict:
    n        = len(df)
    flagged  = int(df["flagged_for_review"].sum())
    corrected= int(df["operator_correction"].sum())
    retrain  = int(df["retrain_flag"].sum())
    return {
        "total_triggers":       n,
        "flagged_for_review":   flagged,
        "pct_flagged":          float(flagged / n * 100),
        "operator_corrections": corrected,
        "pct_corrected":        float(corrected / n * 100),
        "retrain_flag_triggers":retrain,
        "pct_retrain":          float(retrain / n * 100),
        "max_correction_rate":  float(df["running_correction_rate"].max()),
    }


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
def plot_alert_timeline(df: pd.DataFrame, stage: str, save_path: Path,
                         window: int = 50):
    rate_col = f"rate_W{window}" if f"rate_W{window}" in df.columns else "rate_W50"
    x        = np.arange(len(df))
    inj_sim  = compute_simulated_injection_rate(df)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Alert level timeline (colour coded)
    level_map = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    for t, level in enumerate(df["hazard_level"]):
        axes[0].axvspan(t, t+1, alpha=0.6,
                        color=ALERT_COLORS[level], linewidth=0)
    axes[0].set_ylabel("Hazard level")
    axes[0].set_title("Alert dispatcher output")
    axes[0].set_yticks([])
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=ALERT_COLORS[l], label=l)
                    for l in ["LOW", "MODERATE", "HIGH"]]
    axes[0].legend(handles=legend_elems, loc="upper right", fontsize=9)

    # Event rate vs simulated injection rate
    axes[1].plot(x, df[rate_col].values, color="#1D9E75",
                 linewidth=0.8, label="Event rate")
    axes[1].plot(x, inj_sim, color="#7F77DD",
                 linewidth=1.0, linestyle="--", label="Simulated injection rate")
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Event rate vs agent-controlled injection rate")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    # Operator review flags
    axes[2].fill_between(x,
                          df["flagged_for_review"].astype(int),
                          color="#EF9F27", alpha=0.5, label="Flagged for review")
    axes[2].fill_between(x,
                          df["operator_correction"].astype(int),
                          color="#E24B4A", alpha=0.7, label="Operator correction")
    axes[2].plot(x, df["running_correction_rate"].values,
                 color="#E24B4A", linewidth=0.8, linestyle="--",
                 label="Running correction rate")
    axes[2].axhline(CORRECTION_RATE_LIMIT, color="#E24B4A",
                    linestyle=":", linewidth=1,
                    label=f"Retrain threshold ({CORRECTION_RATE_LIMIT})")
    axes[2].set_ylabel("Flag / rate")
    axes[2].set_xlabel("Trigger index (chronological)")
    axes[2].set_title("Operator feedback & retraining signal")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Act Layer Output  |  {stage}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_injection_advisory(df14: pd.DataFrame, df2: pd.DataFrame,
                              save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, stage in [(axes[0], df14, "Stage 14"),
                           (axes[1], df2,  "Stage 2 (FORGE)")]:
        summary = injection_advisory_summary(df)
        recs    = ["MAINTAIN", "REDUCE", "SUSPEND"]
        counts  = [summary[r]["count"] for r in recs]
        pcts    = [summary[r]["pct"]   for r in recs]
        colors  = ["#1D9E75", "#EF9F27", "#E24B4A"]
        bars = ax.bar(recs, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(stage)
        ax.set_ylabel("Trigger count")
        ax.grid(axis="y", alpha=0.3)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 5,
                    f"{pct:.1f}%", ha="center", fontsize=10)

    fig.suptitle("Injection Advisory Distribution — Agent Recommendations",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_confidence_vs_alert(df: pd.DataFrame, stage: str, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confidence distribution by hazard level
    for level, color in ALERT_COLORS.items():
        mask = df["hazard_level"] == level
        if mask.sum() > 0:
            axes[0].hist(df.loc[mask, "alert_confidence"],
                         bins=40, alpha=0.6, color=color, label=level)
    axes[0].set_xlabel("Alert confidence (1 - mc_std)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence distribution by hazard level")
    axes[0].legend()

    # Anomaly score vs alert level (scatter)
    level_num = df["hazard_level"].map({"LOW": 0, "MODERATE": 1, "HIGH": 2})
    axes[1].scatter(df["anomaly_score"], level_num,
                    c=df["hazard_level"].map(ALERT_COLORS),
                    alpha=0.3, s=5)
    axes[1].set_xlabel("Anomaly score")
    axes[1].set_ylabel("Hazard level (0=LOW, 1=MOD, 2=HIGH)")
    axes[1].set_title("Anomaly score vs hazard level")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["LOW", "MODERATE", "HIGH"])
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Alert Confidence Analysis  |  {stage}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 4 — Act Layer")
print("=" * 60)

print("\nLoading Phase 3 outputs...")
df14 = pd.read_pickle(DATA_DIR / "stage14_plan.pkl")
df2  = pd.read_pickle(DATA_DIR / "stage2_plan.pkl")
print(f"  Stage 14 : {len(df14)} triggers")
print(f"  Stage 2  : {len(df2)} samples")

# ── 4a. Alert Dispatcher ──────────────────────────────────────
print("\n[4a] Alert Dispatcher...")
df14 = dispatch_alerts(df14, action_col="action_rl", window=50)
df2  = dispatch_alerts(df2,  action_col="action_rl", window=50)

for name, df in [("Stage 14", df14), ("Stage 2", df2)]:
    dist = df["hazard_level"].value_counts().to_dict()
    print(f"  {name} alert distribution : {dist}")

# ── 4b. Injection Advisor ─────────────────────────────────────
print("\n[4b] Injection Advisor...")
for name, df in [("Stage 14", df14), ("Stage 2", df2)]:
    summary = injection_advisory_summary(df)
    print(f"  {name}:")
    for rec, vals in summary.items():
        print(f"    {rec:<10} : {vals['count']:>5} triggers  ({vals['pct']:.1f}%)")

# ── 4c. Feedback & Retraining ─────────────────────────────────
print("\n[4c] Operator Feedback & Retraining Flag...")

# Use rl_correct if available, else correct
correct_col = "rl_correct" if "rl_correct" in df14.columns else "correct"
df14 = simulate_operator_feedback(df14, action_col="action_rl", unc_col="mc_std")
df2  = simulate_operator_feedback(df2,  action_col="action_rl", unc_col="mc_std")

fb14 = feedback_summary(df14)
fb2  = feedback_summary(df2)

for name, fb in [("Stage 14", fb14), ("Stage 2", fb2)]:
    print(f"  {name}:")
    print(f"    Flagged for review     : {fb['flagged_for_review']} "
          f"({fb['pct_flagged']:.2f}%)")
    print(f"    Operator corrections   : {fb['operator_corrections']} "
          f"({fb['pct_corrected']:.2f}%)")
    print(f"    Retrain flag triggers  : {fb['retrain_flag_triggers']} "
          f"({fb['pct_retrain']:.2f}%)")
    print(f"    Max correction rate    : {fb['max_correction_rate']:.4f}")

# ── Plots ──────────────────────────────────────────────────────
print("\nGenerating figures...")
plot_alert_timeline(df14, "Stage 14",
                    FIG_DIR / "act_alert_timeline_stage14.png")
plot_alert_timeline(df2,  "Stage 2 (FORGE)",
                    FIG_DIR / "act_alert_timeline_stage2.png")
plot_injection_advisory(df14, df2,
                        FIG_DIR / "act_injection_advisory.png")
plot_confidence_vs_alert(df14, "Stage 14",
                          FIG_DIR / "act_confidence_stage14.png")
plot_confidence_vs_alert(df2,  "Stage 2 (FORGE)",
                          FIG_DIR / "act_confidence_stage2.png")

# ── Save outputs ──────────────────────────────────────────────
print("\nSaving outputs...")
df14.to_pickle(DATA_DIR / "stage14_act.pkl")
df2.to_pickle(DATA_DIR  / "stage2_act.pkl")
print(f"  Saved -> data/stage14_act.pkl  ({len(df14)} rows)")
print(f"  Saved -> data/stage2_act.pkl   ({len(df2)} rows)")

# ── Act report ────────────────────────────────────────────────
report = DATA_DIR / "act_report.txt"
with open(report, "w") as f:
    f.write("PHASE 4 ACT LAYER REPORT\n")
    f.write("=" * 60 + "\n\n")

    for name, df, fb in [("Stage 14",        df14, fb14),
                          ("Stage 2 (FORGE)", df2,  fb2)]:
        f.write(f"{name}\n")
        f.write("-" * 40 + "\n")

        f.write("Alert distribution:\n")
        dist = df["hazard_level"].value_counts().to_dict()
        for lvl in ["HIGH", "MODERATE", "LOW"]:
            n   = dist.get(lvl, 0)
            pct = n / len(df) * 100
            f.write(f"  {lvl:<10} : {n:>5} ({pct:.1f}%)\n")

        f.write("\nInjection advisory:\n")
        inj = injection_advisory_summary(df)
        for rec in ["MAINTAIN", "REDUCE", "SUSPEND"]:
            f.write(f"  {rec:<10} : {inj[rec]['count']:>5} "
                    f"({inj[rec]['pct']:.1f}%)\n")

        f.write("\nOperator feedback:\n")
        f.write(f"  Flagged for review     : {fb['flagged_for_review']} "
                f"({fb['pct_flagged']:.2f}%)\n")
        f.write(f"  Operator corrections   : {fb['operator_corrections']} "
                f"({fb['pct_corrected']:.2f}%)\n")
        f.write(f"  Retrain flag triggers  : {fb['retrain_flag_triggers']} "
                f"({fb['pct_retrain']:.2f}%)\n")
        f.write(f"  Max correction rate    : {fb['max_correction_rate']:.4f}\n\n")

        f.write("Sample alert records (first 5):\n")
        cols = ["hazard_level", "injection_rec", "alert_confidence", "justification"]
        cols = [c for c in cols if c in df.columns]
        f.write(df[cols].head(5).to_string(index=False))
        f.write("\n\n")

print(f"  Saved -> data/act_report.txt")

print("\n" + "=" * 60)
print("Phase 4 complete.")
print("Outputs : stage14_act.pkl | stage2_act.pkl")
print("Figures : data/figures/act_*.png")
print("Next -> Phase 5: agent.py (full integration)")
print("=" * 60)
