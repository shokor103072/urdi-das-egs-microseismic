"""
Phase 3 — Plan Layer
Agentic DAS Microseismic Monitoring System

Two policy modules trained/tuned on Stage 14, then evaluated on Stage 2:

  3a. Rule-Based Policy (baseline)
      Threshold table on {event_rate, uncertainty, anomaly_score}.
      Actions: Watch (0), Caution (1), Halt (2).
      Thresholds tuned by grid search on Stage 14.

  3b. Q-Learning RL Policy
      State space (discretised):
        s = (rate_bin, unc_bin, anomaly_bin)  — 4×3×2 = 24 states
      Action space: {Watch=0, Caution=1, Halt=2}
      Reward function:
        +1.0  correct Watch    (true noise or low event rate)
        +1.0  correct Caution  (moderate event rate)
        +2.0  correct Halt     (high event rate / anomaly detected)
        -0.5  unnecessary Halt when quiet
        -1.0  missed Halt during anomaly
        -0.3  wrong action generally
      Trained on Stage 14 operational sequence.

  Cross-stage evaluation:
      Both policies deployed on Stage 2 WITHOUT retraining.
      Measures autonomous adaptation quality.

Inputs:
  data/stage14_reason.pkl
  data/stage2_reason.pkl

Outputs:
  data/stage14_plan.pkl         reason df + policy decisions
  data/stage2_plan.pkl          same for Stage 2
  data/policy_rl_qtable.npy    trained Q-table
  data/plan_report.txt          full evaluation report
  data/figures/                 policy decision plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_DIR = Path("./data")
FIG_DIR  = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Action labels
ACTIONS     = {0: "Watch", 1: "Caution", 2: "Halt"}
ACTION_COLS = ["action_rule", "action_rl"]

# State space bins
RATE_BINS  = [0.0, 0.40, 0.70, 1.01]   # 3 bins: low / moderate / high
UNC_BINS   = [0.0, 0.05, 0.15, 1.01]   # 3 bins: confident / moderate / uncertain
ANOM_BINS  = [0.0, 0.50, 1.01]          # 2 bins: normal / anomalous

N_RATE  = len(RATE_BINS)  - 1   # 3
N_UNC   = len(UNC_BINS)   - 1   # 3
N_ANOM  = len(ANOM_BINS)  - 1   # 2
N_STATES  = N_RATE * N_UNC * N_ANOM   # 18
N_ACTIONS = 3


# ═══════════════════════════════════════════════════════════════
# GROUND TRUTH POLICY LABEL
# ═══════════════════════════════════════════════════════════════
def assign_gt_action(rate, anomaly_score, unc):
    """
    Ground truth action based on true operational state.
    Used for reward computation and policy evaluation.
      rate >= 0.70 OR anomaly_score >= 0.50 → Halt (2)
      rate >= 0.40 OR unc >= 0.05           → Caution (1)
      otherwise                             → Watch (0)
    """
    if rate >= 0.70 or anomaly_score >= 0.50:
        return 2  # Halt
    elif rate >= 0.40 or unc >= 0.05:
        return 1  # Caution
    else:
        return 0  # Watch


def assign_gt_actions(df: pd.DataFrame, window: int = 50) -> np.ndarray:
    rate_col = f"rate_W{window}" if f"rate_W{window}" in df.columns else "rate_W50"
    return np.array([
        assign_gt_action(row[rate_col], row["anomaly_score"], row["mc_std"])
        for _, row in df.iterrows()
    ])


# ═══════════════════════════════════════════════════════════════
# 3a. RULE-BASED POLICY
# ═══════════════════════════════════════════════════════════════
def rule_policy(rate: float, unc: float, anomaly: float,
                alpha: float, beta: float,
                unc_thresh: float, anom_thresh: float) -> int:
    """
    Threshold rule:
      rate >= beta  OR anomaly >= anom_thresh → Halt
      rate >= alpha OR unc >= unc_thresh      → Caution
      otherwise                              → Watch
    """
    if rate >= beta or anomaly >= anom_thresh:
        return 2
    elif rate >= alpha or unc >= unc_thresh:
        return 1
    else:
        return 0


def grid_search_rule_policy(df: pd.DataFrame,
                              gt_actions: np.ndarray,
                              window: int = 50) -> dict:
    """
    Grid search over (alpha, beta, unc_thresh, anom_thresh).
    Maximises accuracy of action prediction vs gt_actions.
    """
    rate_col = f"rate_W{window}" if f"rate_W{window}" in df.columns else "rate_W50"
    rates    = df[rate_col].values
    uncs     = df["mc_std"].values
    anoms    = df["anomaly_score"].values

    alphas      = [0.30, 0.40, 0.50]
    betas       = [0.60, 0.70, 0.80]
    unc_threshs = [0.03, 0.05, 0.10]
    anom_threshs= [0.30, 0.50, 0.70]

    best_acc, best_params = 0.0, {}
    for alpha, beta, unc_t, anom_t in product(alphas, betas, unc_threshs, anom_threshs):
        if alpha >= beta:
            continue
        preds = np.array([rule_policy(r, u, a, alpha, beta, unc_t, anom_t)
                          for r, u, a in zip(rates, uncs, anoms)])
        acc = (preds == gt_actions).mean()
        if acc > best_acc:
            best_acc    = acc
            best_params = dict(alpha=alpha, beta=beta,
                               unc_thresh=unc_t, anom_thresh=anom_t)

    return {"best_acc": best_acc, "params": best_params}


def apply_rule_policy(df: pd.DataFrame, params: dict,
                       window: int = 50) -> np.ndarray:
    rate_col = f"rate_W{window}" if f"rate_W{window}" in df.columns else "rate_W50"
    return np.array([
        rule_policy(row[rate_col], row["mc_std"], row["anomaly_score"], **params)
        for _, row in df.iterrows()
    ])


# ═══════════════════════════════════════════════════════════════
# 3b. Q-LEARNING RL POLICY
# ═══════════════════════════════════════════════════════════════
def discretise_state(rate: float, unc: float, anomaly: float) -> int:
    """Maps continuous (rate, unc, anomaly) to a single state index."""
    r = min(int(np.digitize(rate,  RATE_BINS[1:]) ), N_RATE  - 1)
    u = min(int(np.digitize(unc,   UNC_BINS[1:])  ), N_UNC   - 1)
    a = min(int(np.digitize(anomaly, ANOM_BINS[1:])), N_ANOM  - 1)
    return r * (N_UNC * N_ANOM) + u * N_ANOM + a


def compute_reward(action: int, gt_action: int,
                   rate: float, anomaly: float) -> float:
    """
    Shaped reward function encouraging correct hazard decisions.
    Penalises missed Halts (safety-critical) more than unnecessary Halts.
    """
    if action == gt_action:
        if action == 2:   return  2.0   # correct Halt — high reward
        if action == 1:   return  1.0   # correct Caution
        return  1.0                      # correct Watch
    else:
        # Missed Halt during anomaly — most dangerous
        if gt_action == 2 and action != 2:
            return -1.5
        # Unnecessary Halt when quiet
        if action == 2 and gt_action == 0:
            return -0.5
        # Other wrong actions
        return -0.3


def train_q_learning(df: pd.DataFrame, gt_actions: np.ndarray,
                      window: int = 50,
                      n_episodes: int = 300,
                      lr: float = 0.1,
                      gamma: float = 0.95,
                      eps_start: float = 1.0,
                      eps_end: float = 0.05,
                      eps_decay: float = 0.97) -> tuple:
    """
    Tabular Q-learning on Stage 14 operational sequence.
    Each episode = one pass through the full trigger sequence.
    Returns (Q_table, training_history).
    """
    rate_col = f"rate_W{window}" if f"rate_W{window}" in df.columns else "rate_W50"
    rates  = df[rate_col].values
    uncs   = df["mc_std"].values
    anoms  = df["anomaly_score"].values
    T      = len(df)

    Q      = np.zeros((N_STATES, N_ACTIONS))
    eps    = eps_start
    history = []   # (episode, total_reward, accuracy)

    for ep in range(n_episodes):
        total_reward = 0.0
        correct      = 0

        for t in range(T - 1):
            s  = discretise_state(rates[t], uncs[t], anoms[t])
            s_ = discretise_state(rates[t+1], uncs[t+1], anoms[t+1])
            gt = gt_actions[t]

            # Epsilon-greedy action selection
            if np.random.rand() < eps:
                a = np.random.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[s]))

            r = compute_reward(a, gt, rates[t], anoms[t])
            total_reward += r
            correct      += int(a == gt)

            # Q-update
            Q[s, a] += lr * (r + gamma * np.max(Q[s_]) - Q[s, a])

        eps = max(eps_end, eps * eps_decay)
        acc = correct / (T - 1)

        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"    Ep {ep+1:>4}/{n_episodes}  "
                  f"reward={total_reward:>8.1f}  acc={acc:.4f}  eps={eps:.3f}")
        history.append({"episode": ep+1, "reward": total_reward,
                         "accuracy": acc, "epsilon": eps})

    return Q, pd.DataFrame(history)


def apply_rl_policy(df: pd.DataFrame, Q: np.ndarray,
                     window: int = 50) -> np.ndarray:
    rate_col = f"rate_W{window}" if f"rate_W{window}" in df.columns else "rate_W50"
    actions = []
    for _, row in df.iterrows():
        s = discretise_state(row[rate_col], row["mc_std"], row["anomaly_score"])
        actions.append(int(np.argmax(Q[s])))
    return np.array(actions)


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate_policy(pred_actions: np.ndarray,
                     gt_actions: np.ndarray,
                     name: str) -> dict:
    """Per-action accuracy and overall policy quality metrics."""
    acc = (pred_actions == gt_actions).mean()
    results = {"name": name, "overall_accuracy": float(acc)}

    for a_idx, a_name in ACTIONS.items():
        mask     = gt_actions == a_idx
        if mask.sum() == 0:
            results[f"{a_name}_accuracy"] = None
            continue
        a_acc    = (pred_actions[mask] == a_idx).mean()
        results[f"{a_name}_accuracy"] = float(a_acc)
        results[f"{a_name}_count"]    = int(mask.sum())

    # Safety metric: missed Halts (most dangerous error)
    halt_mask    = gt_actions == 2
    missed_halts = ((pred_actions != 2) & halt_mask).sum()
    results["missed_halts"]     = int(missed_halts)
    results["missed_halt_rate"] = float(missed_halts / halt_mask.sum()
                                        if halt_mask.sum() > 0 else 0)

    # Unnecessary Halts (operational cost)
    unneeded_halts = ((pred_actions == 2) & (gt_actions == 0)).sum()
    results["unnecessary_halts"] = int(unneeded_halts)

    return results


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
ACTION_COLORS = {0: "#1D9E75", 1: "#EF9F27", 2: "#E24B4A"}

def plot_policy_decisions(df: pd.DataFrame,
                           rule_actions: np.ndarray,
                           rl_actions: np.ndarray,
                           gt_actions: np.ndarray,
                           stage: str, save_path: Path,
                           window: int = 50):
    rate_col = f"rate_W{window}" if f"rate_W{window}" in df.columns else "rate_W50"
    x    = np.arange(len(df))
    rate = df[rate_col].values

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Event rate
    axes[0].plot(x, rate, color="#1D9E75", linewidth=0.8)
    axes[0].set_ylabel("Event rate")
    axes[0].set_title(f"Rolling event rate (W={window})")
    axes[0].grid(alpha=0.3)

    # Ground truth
    for t, a in enumerate(gt_actions):
        axes[1].axvspan(t, t+1, alpha=0.6, color=ACTION_COLORS[a], linewidth=0)
    axes[1].set_ylabel("GT action")
    axes[1].set_title("Ground truth policy")
    axes[1].set_yticks([])

    # Rule-based
    for t, a in enumerate(rule_actions):
        axes[2].axvspan(t, t+1, alpha=0.6, color=ACTION_COLORS[a], linewidth=0)
    rule_acc = (rule_actions == gt_actions).mean()
    axes[2].set_ylabel("Rule action")
    axes[2].set_title(f"Rule-based policy  (acc={rule_acc:.4f})")
    axes[2].set_yticks([])

    # RL
    for t, a in enumerate(rl_actions):
        axes[3].axvspan(t, t+1, alpha=0.6, color=ACTION_COLORS[a], linewidth=0)
    rl_acc = (rl_actions == gt_actions).mean()
    axes[3].set_ylabel("RL action")
    axes[3].set_xlabel("Trigger index (chronological)")
    axes[3].set_title(f"RL Q-learning policy  (acc={rl_acc:.4f})")
    axes[3].set_yticks([])

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=ACTION_COLORS[i], label=ACTIONS[i])
                    for i in range(3)]
    axes[0].legend(handles=legend_elems, loc="upper right",
                   fontsize=9, title="Action")

    fig.suptitle(f"Policy Decisions  |  {stage}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_training_curve(history: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["episode"], history["reward"],
                 color="#7F77DD", linewidth=0.8)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].set_title("RL training — cumulative reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["episode"], history["accuracy"],
                 color="#1D9E75", linewidth=0.8)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Action accuracy")
    axes[1].set_title("RL training — policy accuracy")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Q-Learning Training Curve — Stage 14", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


def plot_cross_stage_comparison(eval14_rule, eval14_rl,
                                 eval2_rule,  eval2_rl,
                                 save_path: Path):
    metrics  = ["overall_accuracy", "Watch_accuracy",
                "Caution_accuracy", "Halt_accuracy"]
    labels   = ["Overall", "Watch", "Caution", "Halt"]
    x        = np.arange(len(metrics))
    width    = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, stage_label, rule_eval, rl_eval in [
        (axes[0], "Stage 14 (trained)", eval14_rule, eval14_rl),
        (axes[1], "Stage 2  (zero-shot)", eval2_rule,  eval2_rl),
    ]:
        rule_vals = [rule_eval.get(m) or 0 for m in metrics]
        rl_vals   = [rl_eval.get(m)   or 0 for m in metrics]

        ax.bar(x - width/2, rule_vals, width, label="Rule-based", color="#7F77DD")
        ax.bar(x + width/2, rl_vals,   width, label="RL policy",  color="#1D9E75")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy")
        ax.set_title(stage_label)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for i, (rv, lv) in enumerate(zip(rule_vals, rl_vals)):
            ax.text(i - width/2, rv + 0.01, f"{rv:.3f}", ha="center",
                    fontsize=7, color="#7F77DD")
            ax.text(i + width/2, lv + 0.01, f"{lv:.3f}", ha="center",
                    fontsize=7, color="#1D9E75")

    fig.suptitle("Policy Evaluation — Stage 14 vs Stage 2 (cross-stage)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
np.random.seed(42)

print("=" * 60)
print("PHASE 3 — Plan Layer")
print("=" * 60)

print("\nLoading Phase 2 outputs...")
df14 = pd.read_pickle(DATA_DIR / "stage14_reason.pkl")
df2  = pd.read_pickle(DATA_DIR / "stage2_reason.pkl")
print(f"  Stage 14 : {len(df14)} triggers")
print(f"  Stage 2  : {len(df2)} samples")

# Determine which rate column exists
W = 50
rate_col = f"rate_W{W}" if f"rate_W{W}" in df14.columns else "rate_W50"

# Ground truth actions
print("\nAssigning ground truth actions...")
gt14 = assign_gt_actions(df14, window=W)
gt2  = assign_gt_actions(df2,  window=W)

for name, gt in [("Stage 14", gt14), ("Stage 2", gt2)]:
    counts = {ACTIONS[a]: (gt==a).sum() for a in range(3)}
    print(f"  {name} GT distribution: {counts}")


# ── 3a. Rule-Based Policy ─────────────────────────────────────
print("\n[3a] Rule-Based Policy — grid search on Stage 14...")
gs_result = grid_search_rule_policy(df14, gt14, window=W)
best_params = gs_result["params"]
print(f"  Best params : {best_params}")
print(f"  Best acc    : {gs_result['best_acc']:.4f}")

# Apply to both stages
rule14 = apply_rule_policy(df14, best_params, window=W)
rule2  = apply_rule_policy(df2,  best_params, window=W)

eval14_rule = evaluate_policy(rule14, gt14, "Rule Stage 14")
eval2_rule  = evaluate_policy(rule2,  gt2,  "Rule Stage 2")

print(f"\n  Rule policy — Stage 14 : acc={eval14_rule['overall_accuracy']:.4f}  "
      f"missed_halts={eval14_rule['missed_halts']}")
print(f"  Rule policy — Stage 2  : acc={eval2_rule['overall_accuracy']:.4f}  "
      f"missed_halts={eval2_rule['missed_halts']}")


# ── 3b. Q-Learning RL Policy ──────────────────────────────────
print("\n[3b] Q-Learning RL Policy — training on Stage 14...")
Q, history = train_q_learning(df14, gt14, window=W,
                               n_episodes=300,
                               lr=0.1, gamma=0.95,
                               eps_start=1.0, eps_end=0.05,
                               eps_decay=0.97)

np.save(DATA_DIR / "policy_rl_qtable.npy", Q)
history.to_csv(DATA_DIR / "rl_training_history.csv", index=False)
print(f"  Q-table saved -> data/policy_rl_qtable.npy")
print(f"  Final episode accuracy: {history['accuracy'].iloc[-1]:.4f}")
print(f"  Q-table shape: {Q.shape}  ({N_STATES} states x {N_ACTIONS} actions)")

# Apply to both stages
rl14 = apply_rl_policy(df14, Q, window=W)
rl2  = apply_rl_policy(df2,  Q, window=W)

eval14_rl = evaluate_policy(rl14, gt14, "RL Stage 14")
eval2_rl  = evaluate_policy(rl2,  gt2,  "RL Stage 2")

print(f"\n  RL policy — Stage 14 : acc={eval14_rl['overall_accuracy']:.4f}  "
      f"missed_halts={eval14_rl['missed_halts']}")
print(f"  RL policy — Stage 2  : acc={eval2_rl['overall_accuracy']:.4f}  "
      f"missed_halts={eval2_rl['missed_halts']}")


# ── Attach decisions to dataframes ────────────────────────────
df14["gt_action"]      = gt14
df14["gt_action_name"] = [ACTIONS[a] for a in gt14]
df14["action_rule"]    = rule14
df14["action_rl"]      = rl14
df14["rule_correct"]   = (rule14 == gt14).astype(int)
df14["rl_correct"]     = (rl14   == gt14).astype(int)

df2["gt_action"]       = gt2
df2["gt_action_name"]  = [ACTIONS[a] for a in gt2]
df2["action_rule"]     = rule2
df2["action_rl"]       = rl2
df2["rule_correct"]    = (rule2 == gt2).astype(int)
df2["rl_correct"]      = (rl2   == gt2).astype(int)


# ── Plots ──────────────────────────────────────────────────────
print("\nGenerating figures...")
plot_policy_decisions(df14, rule14, rl14, gt14,
                      "Stage 14", FIG_DIR / "policy_decisions_stage14.png")
plot_policy_decisions(df2, rule2, rl2, gt2,
                      "Stage 2 (FORGE)", FIG_DIR / "policy_decisions_stage2.png")
plot_training_curve(history, FIG_DIR / "rl_training_curve.png")
plot_cross_stage_comparison(eval14_rule, eval14_rl,
                             eval2_rule,  eval2_rl,
                             FIG_DIR / "cross_stage_policy_comparison.png")


# ── Save outputs ──────────────────────────────────────────────
print("\nSaving outputs...")
df14.to_pickle(DATA_DIR / "stage14_plan.pkl")
df2.to_pickle(DATA_DIR  / "stage2_plan.pkl")
print(f"  Saved -> data/stage14_plan.pkl  ({len(df14)} rows)")
print(f"  Saved -> data/stage2_plan.pkl   ({len(df2)} rows)")


# ── Plan report ───────────────────────────────────────────────
report = DATA_DIR / "plan_report.txt"
with open(report, "w") as f:
    f.write("PHASE 3 PLAN LAYER REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("Rule-Based Policy Parameters (tuned on Stage 14):\n")
    for k, v in best_params.items():
        f.write(f"  {k:<20}: {v}\n")
    f.write(f"  Grid search accuracy  : {gs_result['best_acc']:.4f}\n\n")

    f.write("Q-Learning Configuration:\n")
    f.write(f"  State space     : {N_STATES} states "
            f"({N_RATE} rate x {N_UNC} unc x {N_ANOM} anomaly bins)\n")
    f.write(f"  Action space    : {N_ACTIONS} actions (Watch/Caution/Halt)\n")
    f.write(f"  Episodes        : 300\n")
    f.write(f"  Learning rate   : 0.1\n")
    f.write(f"  Gamma           : 0.95\n")
    f.write(f"  Final epsilon   : {history['epsilon'].iloc[-1]:.4f}\n\n")

    f.write("Policy Evaluation:\n")
    f.write(f"{'Metric':<35} {'Rule S14':>10} {'RL S14':>10} "
            f"{'Rule S2':>10} {'RL S2':>10}\n")
    f.write("-" * 75 + "\n")

    rows = [
        ("Overall accuracy",   "overall_accuracy"),
        ("Watch accuracy",     "Watch_accuracy"),
        ("Caution accuracy",   "Caution_accuracy"),
        ("Halt accuracy",      "Halt_accuracy"),
        ("Missed Halt rate",   "missed_halt_rate"),
        ("Unnecessary Halts",  "unnecessary_halts"),
    ]
    for label, key in rows:
        def fmt(v):
            if v is None: return "     N/A"
            if isinstance(v, float): return f"{v:>10.4f}"
            return f"{v:>10}"
        f.write(f"{label:<35} "
                f"{fmt(eval14_rule.get(key))} "
                f"{fmt(eval14_rl.get(key))} "
                f"{fmt(eval2_rule.get(key))} "
                f"{fmt(eval2_rl.get(key))}\n")

    f.write("\nGround truth action distributions:\n")
    for name, gt in [("Stage 14", gt14), ("Stage 2", gt2)]:
        counts = {ACTIONS[a]: int((gt==a).sum()) for a in range(3)}
        f.write(f"  {name}: {counts}\n")

    f.write("\nQ-table (rows=states, cols=actions):\n")
    f.write("  Rows: state index 0.." + str(N_STATES-1) + "\n")
    f.write("  Cols: Watch | Caution | Halt\n")
    np.set_printoptions(precision=3, suppress=True)
    f.write(str(Q) + "\n")

print(f"  Saved -> data/plan_report.txt")

print("\n" + "=" * 60)
print("Phase 3 complete.")
print("Outputs : stage14_plan.pkl | stage2_plan.pkl")
print("          policy_rl_qtable.npy | rl_training_history.csv")
print("Next -> Phase 4: act_layer.py")
print("=" * 60)
