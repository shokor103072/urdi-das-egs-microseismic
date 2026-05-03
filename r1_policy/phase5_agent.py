"""
Phase 5 — Full Agent Integration + Ablation Study
Agentic DAS Microseismic Monitoring System

Integrates all four layers into a single EGSMonitoringAgent class
and runs a formal ablation study comparing:

  Config A: Perception only        (SE-ResNet detection, no reasoning)
  Config B: Perception + Reason    (+ state tracker + uncertainty)
  Config C: Perception + Reason + Rule Policy   (full agent, rule-based)
  Config D: Perception + Reason + RL Policy     (full agent, RL)  ← proposed

Each config evaluated on Stage 14 (in-distribution) across:
  - Decision accuracy      vs ground truth policy actions
  - Missed Halt rate       (safety-critical metric)
  - Unnecessary Halt rate  (operational efficiency metric)
  - Mean alert confidence
  - Mean uncertainty flagging rate
  - Autonomous operation rate  (% triggers requiring no operator input)

Inputs:
  data/stage14_act.pkl
  data/stage2_act.pkl
  data/policy_rl_qtable.npy

Outputs:
  data/ablation_results.csv     ablation table (all configs x all metrics)
  data/stage14_agent.pkl        full agent output for Stage 14
  data/agent_report.txt         comprehensive evaluation report
  data/figures/ablation_*.png   ablation figures
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

ALERT_COLORS = {"LOW": "#1D9E75", "MODERATE": "#EF9F27", "HIGH": "#E24B4A"}
W = 50   # window size used throughout


# ═══════════════════════════════════════════════════════════════
# GROUND TRUTH (same definition as Phase 3)
# ═══════════════════════════════════════════════════════════════
def assign_gt_action(rate, anomaly, unc):
    if rate >= 0.70 or anomaly >= 0.50:
        return 2
    elif rate >= 0.40 or unc >= 0.05:
        return 1
    else:
        return 0


# ═══════════════════════════════════════════════════════════════
# ABLATION CONFIGS
# ═══════════════════════════════════════════════════════════════
def config_A_perception_only(df: pd.DataFrame) -> np.ndarray:
    """
    Perception only: maps raw p_event to an action.
    No state tracker, no uncertainty, no anomaly.
    p_event >= 0.70 → Halt, >= 0.40 → Caution, else Watch.
    """
    p = df["p_event"].values
    actions = np.where(p >= 0.70, 2,
              np.where(p >= 0.40, 1, 0))
    return actions


def config_B_perception_reason(df: pd.DataFrame) -> np.ndarray:
    """
    Perception + Reason: uses rolling event rate and uncertainty
    but no policy optimisation — simple fixed thresholds.
    """
    rate_col = f"rate_W{W}" if f"rate_W{W}" in df.columns else "rate_W50"
    rate = df[rate_col].values
    unc  = df["mc_std"].values
    actions = np.where((rate >= 0.70),                   2,
              np.where((rate >= 0.40) | (unc >= 0.05),   1, 0))
    return actions


def config_C_full_rule(df: pd.DataFrame) -> np.ndarray:
    """Full agent with rule-based policy (from Phase 3)."""
    return df["action_rule"].values.copy()


def config_D_full_rl(df: pd.DataFrame) -> np.ndarray:
    """Full agent with RL policy (from Phase 3) — proposed system."""
    return df["action_rl"].values.copy()


# ═══════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════
def evaluate_config(pred_actions: np.ndarray, gt_actions: np.ndarray,
                     df: pd.DataFrame, config_name: str) -> dict:
    n = len(pred_actions)

    overall_acc  = (pred_actions == gt_actions).mean()
    halt_mask    = gt_actions == 2
    caution_mask = gt_actions == 1
    watch_mask   = gt_actions == 0

    halt_acc    = (pred_actions[halt_mask]    == 2).mean() if halt_mask.sum() > 0 else None
    caution_acc = (pred_actions[caution_mask] == 1).mean() if caution_mask.sum() > 0 else None
    watch_acc   = (pred_actions[watch_mask]   == 0).mean() if watch_mask.sum() > 0 else None

    missed_halts      = int(((pred_actions != 2) & halt_mask).sum())
    missed_halt_rate  = missed_halts / halt_mask.sum() if halt_mask.sum() > 0 else 0

    unnecessary_halts = int(((pred_actions == 2) & watch_mask).sum())
    unneeded_halt_rate= unnecessary_halts / watch_mask.sum() if watch_mask.sum() > 0 else 0

    # Autonomous operation rate: triggers NOT flagged for operator review
    if "flagged_for_review" in df.columns:
        autonomous_rate = 1.0 - df["flagged_for_review"].mean()
    else:
        autonomous_rate = 1.0 - (df["mc_std"] > 0.15).mean()

    mean_confidence = df["confidence"].mean() if "confidence" in df.columns \
                      else df["alert_confidence"].mean() if "alert_confidence" in df.columns \
                      else None

    # Action distribution
    action_dist = {
        "Watch_pct":   float((pred_actions == 0).mean() * 100),
        "Caution_pct": float((pred_actions == 1).mean() * 100),
        "Halt_pct":    float((pred_actions == 2).mean() * 100),
    }

    return {
        "config":               config_name,
        "overall_accuracy":     float(overall_acc),
        "halt_accuracy":        float(halt_acc)    if halt_acc    is not None else None,
        "caution_accuracy":     float(caution_acc) if caution_acc is not None else None,
        "watch_accuracy":       float(watch_acc)   if watch_acc   is not None else None,
        "missed_halt_rate":     float(missed_halt_rate),
        "missed_halts":         missed_halts,
        "unnecessary_halt_rate":float(unneeded_halt_rate),
        "unnecessary_halts":    unnecessary_halts,
        "autonomous_rate":      float(autonomous_rate),
        "mean_confidence":      float(mean_confidence) if mean_confidence else None,
        **action_dist,
    }


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
def plot_ablation_bars(results: pd.DataFrame, save_path: Path):
    configs = results["config"].tolist()
    metrics = [
        ("overall_accuracy",      "Overall accuracy"),
        ("halt_accuracy",         "Halt accuracy"),
        ("missed_halt_rate",      "Missed Halt rate (lower=better)"),
        ("unnecessary_halt_rate", "Unnecessary Halt rate (lower=better)"),
        ("autonomous_rate",       "Autonomous operation rate"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    colors = ["#D3D1C7", "#B4B2A9", "#7F77DD", "#1D9E75"]  # A, B, C, D

    for ax, (col, label) in zip(axes, metrics):
        vals = [results.loc[results["config"]==c, col].values[0]
                if col in results.columns else 0
                for c in configs]
        vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(range(len(configs)), vals, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.split(" ")[0] for c in configs],
                           fontsize=9, rotation=15)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Ablation Study — Stage 14 (in-distribution)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_action_distribution_comparison(results: pd.DataFrame, save_path: Path):
    configs = results["config"].tolist()
    x       = np.arange(len(configs))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, results["Watch_pct"],   width, label="Watch",
           color="#1D9E75", edgecolor="white")
    ax.bar(x,          results["Caution_pct"], width, label="Caution",
           color="#EF9F27", edgecolor="white")
    ax.bar(x + width,  results["Halt_pct"],   width, label="Halt",
           color="#E24B4A", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=10, fontsize=9)
    ax.set_ylabel("% of triggers")
    ax.set_title("Action distribution by agent configuration — Stage 14")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_agent_timeline(df: pd.DataFrame, gt_actions: np.ndarray,
                         config_actions: dict, save_path: Path):
    """Overlay timeline of GT and all 4 configs."""
    rate_col = f"rate_W{W}" if f"rate_W{W}" in df.columns else "rate_W50"
    x = np.arange(len(df))
    n_configs = len(config_actions)

    fig, axes = plt.subplots(n_configs + 2, 1,
                              figsize=(16, 3 * (n_configs + 2)),
                              sharex=True)

    # Event rate
    axes[0].plot(x, df[rate_col].values, color="#1D9E75", linewidth=0.8)
    axes[0].set_ylabel("Event rate")
    axes[0].set_title(f"Rolling event rate (W={W})")
    axes[0].grid(alpha=0.3)

    # Ground truth
    colors_map = {0: "#1D9E75", 1: "#EF9F27", 2: "#E24B4A"}
    for t, a in enumerate(gt_actions):
        axes[1].axvspan(t, t+1, alpha=0.6, color=colors_map[a], linewidth=0)
    axes[1].set_title("Ground truth")
    axes[1].set_yticks([])

    # Each config
    for i, (name, acts) in enumerate(config_actions.items()):
        ax = axes[i + 2]
        acc = (acts == gt_actions).mean()
        for t, a in enumerate(acts):
            ax.axvspan(t, t+1, alpha=0.6, color=colors_map[a], linewidth=0)
        ax.set_title(f"{name}  (acc={acc:.4f})")
        ax.set_yticks([])

    axes[-1].set_xlabel("Trigger index (chronological)")

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=colors_map[i], label=l)
                    for i, l in enumerate(["Watch", "Caution", "Halt"])]
    axes[0].legend(handles=legend_elems, loc="upper right", fontsize=9)

    fig.suptitle("Agent Ablation — Decision Timeline  |  Stage 14", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_safety_efficiency(results: pd.DataFrame, save_path: Path):
    """Safety vs efficiency scatter: missed_halt_rate vs unnecessary_halt_rate."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#D3D1C7", "#B4B2A9", "#7F77DD", "#1D9E75"]

    for i, row in results.iterrows():
        ax.scatter(row["unnecessary_halt_rate"], row["missed_halt_rate"],
                   color=colors[i], s=200, zorder=5)
        ax.annotate(row["config"].split("(")[0].strip(),
                    (row["unnecessary_halt_rate"], row["missed_halt_rate"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Unnecessary Halt rate (operational cost →)")
    ax.set_ylabel("Missed Halt rate (safety risk ↑)")
    ax.set_title("Safety vs Efficiency Trade-off — Ablation Configurations")
    ax.grid(alpha=0.3)
    # Origin is ideal
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.text(0.01, 0.01, "Ideal", fontsize=9, color="gray",
            transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 5 — Full Agent Integration & Ablation")
print("=" * 60)

print("\nLoading Phase 4 outputs...")
df14 = pd.read_pickle(DATA_DIR / "stage14_act.pkl")
df2  = pd.read_pickle(DATA_DIR / "stage2_act.pkl")
print(f"  Stage 14 : {len(df14)} triggers")
print(f"  Stage 2  : {len(df2)} samples")

# Ground truth actions
rate_col = f"rate_W{W}" if f"rate_W{W}" in df14.columns else "rate_W50"
gt14 = np.array([assign_gt_action(row[rate_col], row["anomaly_score"],
                                   row["mc_std"]) for _, row in df14.iterrows()])
gt2  = np.array([assign_gt_action(row[rate_col] if rate_col in df2.columns
                                   else row.get("rate_W50", 0.5),
                                   row["anomaly_score"], row["mc_std"])
                  for _, row in df2.iterrows()])


# ── Ablation on Stage 14 ──────────────────────────────────────
print("\n── Ablation Study — Stage 14 (in-distribution) ─────────")

configs = {
    "A: Perception only":       config_A_perception_only(df14),
    "B: + Reason":              config_B_perception_reason(df14),
    "C: + Rule policy":         config_C_full_rule(df14),
    "D: + RL policy (proposed)":config_D_full_rl(df14),
}

ablation_rows = []
for name, acts in configs.items():
    result = evaluate_config(acts, gt14, df14, name)
    ablation_rows.append(result)
    print(f"\n  {name}")
    print(f"    Overall acc    : {result['overall_accuracy']:.4f}")
    print(f"    Halt acc       : {result['halt_accuracy']}")
    print(f"    Missed Halts   : {result['missed_halts']} "
          f"(rate={result['missed_halt_rate']:.4f})")
    print(f"    Unneeded Halts : {result['unnecessary_halts']} "
          f"(rate={result['unnecessary_halt_rate']:.4f})")
    print(f"    Autonomous rate: {result['autonomous_rate']:.4f}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(DATA_DIR / "ablation_results.csv", index=False)
print(f"\n  Saved -> data/ablation_results.csv")


# ── Cross-stage evaluation (Stage 2) ─────────────────────────
print("\n── Cross-Stage Evaluation — Stage 2 (zero-shot) ────────")

configs2 = {
    "A: Perception only":       config_A_perception_only(df2),
    "B: + Reason":              config_B_perception_reason(df2),
    "C: + Rule policy":         config_C_full_rule(df2),
    "D: + RL policy (proposed)":config_D_full_rl(df2),
}

ablation_rows2 = []
for name, acts in configs2.items():
    result = evaluate_config(acts, gt2, df2, name)
    ablation_rows2.append(result)
    print(f"\n  {name}")
    print(f"    Overall acc    : {result['overall_accuracy']:.4f}")
    print(f"    Missed Halts   : {result['missed_halts']} "
          f"(rate={result['missed_halt_rate']:.4f})")
    print(f"    Unneeded Halts : {result['unnecessary_halts']} "
          f"(rate={result['unnecessary_halt_rate']:.4f})")

ablation_df2 = pd.DataFrame(ablation_rows2)
ablation_df2.to_csv(DATA_DIR / "ablation_results_stage2.csv", index=False)
print(f"\n  Saved -> data/ablation_results_stage2.csv")


# ── Plots ──────────────────────────────────────────────────────
print("\nGenerating figures...")
plot_ablation_bars(ablation_df,
                   FIG_DIR / "ablation_bars_stage14.png")
plot_action_distribution_comparison(ablation_df,
                                     FIG_DIR / "ablation_action_dist.png")
plot_agent_timeline(df14, gt14, configs,
                    FIG_DIR / "ablation_timeline_stage14.png")
plot_safety_efficiency(ablation_df,
                       FIG_DIR / "ablation_safety_efficiency.png")


# ── Save full agent output ────────────────────────────────────
print("\nSaving full agent outputs...")
df14["gt_action_p5"]     = gt14
df14["config_A_action"]  = configs["A: Perception only"]
df14["config_B_action"]  = configs["B: + Reason"]
df14["config_C_action"]  = configs["C: + Rule policy"]
df14["config_D_action"]  = configs["D: + RL policy (proposed)"]

df14.to_pickle(DATA_DIR / "stage14_agent.pkl")
print(f"  Saved -> data/stage14_agent.pkl")


# ── Agent report ──────────────────────────────────────────────
report = DATA_DIR / "agent_report.txt"
with open(report, "w") as f:
    f.write("PHASE 5 AGENT REPORT — ABLATION STUDY\n")
    f.write("=" * 70 + "\n\n")

    header = (f"{'Config':<35} {'OvAcc':>7} {'HaltAcc':>8} "
              f"{'MissHalt':>9} {'UnnHalt':>8} {'AutRate':>8}")
    f.write("Stage 14 (in-distribution)\n")
    f.write("-" * 70 + "\n")
    f.write(header + "\n")
    f.write("-" * 70 + "\n")
    for row in ablation_rows:
        f.write(
            f"{row['config']:<35} "
            f"{row['overall_accuracy']:>7.4f} "
            f"{str(round(row['halt_accuracy'],4) if row['halt_accuracy'] else 'N/A'):>8} "
            f"{row['missed_halt_rate']:>9.4f} "
            f"{row['unnecessary_halt_rate']:>8.4f} "
            f"{row['autonomous_rate']:>8.4f}\n"
        )

    f.write("\nStage 2 — FORGE (zero-shot cross-stage)\n")
    f.write("-" * 70 + "\n")
    f.write(header + "\n")
    f.write("-" * 70 + "\n")
    for row in ablation_rows2:
        f.write(
            f"{row['config']:<35} "
            f"{row['overall_accuracy']:>7.4f} "
            f"{str(round(row['halt_accuracy'],4) if row['halt_accuracy'] else 'N/A'):>8} "
            f"{row['missed_halt_rate']:>9.4f} "
            f"{row['unnecessary_halt_rate']:>8.4f} "
            f"{row['autonomous_rate']:>8.4f}\n"
        )

    f.write("\nAction distribution — Stage 14:\n")
    f.write(f"{'Config':<35} {'Watch%':>8} {'Caution%':>10} {'Halt%':>7}\n")
    f.write("-" * 65 + "\n")
    for row in ablation_rows:
        f.write(f"{row['config']:<35} "
                f"{row['Watch_pct']:>7.1f}% "
                f"{row['Caution_pct']:>9.1f}% "
                f"{row['Halt_pct']:>6.1f}%\n")

print(f"  Saved -> data/agent_report.txt")

print("\n" + "=" * 60)
print("Phase 5 complete.")
print("Outputs : stage14_agent.pkl")
print("          ablation_results.csv | ablation_results_stage2.csv")
print("Figures : data/figures/ablation_*.png")
print("Next -> Phase 6: evaluate_stage2.py (cross-stage deployment)")
print("=" * 60)
