# Uncertainty-Regularized Domain Generalization for Cross-Site DAS Microseismic Detection in Enhanced Geothermal Systems

**Paper:** "Uncertainty-Regularized Domain Generalization for Cross-Site DAS Microseismic Detection in Enhanced Geothermal Systems"  
**Authors:** Shokor, Jafreezal Bin Jaafar (Senior Member, IEEE), and Irving Vitra Paputungan  
**Submitted to:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

---

## Overview

This repository contains the full implementation of:

1. **URDI** (Uncertainty-Regularized Domain-Invariant Training) — a novel training objective that penalizes MC-Dropout epistemic variance to improve cross-domain calibration without target-domain data.
2. **10-architecture benchmark** across six families (CNN, CNN-RNN, RNN, SSM/Mamba, GNN, Transformer) evaluated on two EGS datasets with 5-seed multi-seed statistics.
3. **Uncertainty-aware RL hazard policy** — tabular Q-learning agent for three-tier (Watch/Caution/Halt) operational decisions.
4. **Full evaluation pipeline** — calibration metrics (ECE, MCE, NLL, Brier), selective prediction (AURC), domain shift analysis (MMD, t-SNE), and DG baseline comparison.

### Key Results

| Method | FORGE F1 | ECE | MCE | AURC |
|---|---|---|---|---|
| CE Baseline | 0.697±0.022 | 0.442±0.041 | 0.822±0.087 | 0.558±0.003 |
| Mixup | 0.676±0.135† | **0.290±0.089** | **0.616±0.161** | 0.378±0.106† |
| RandConv | 0.695±0.030 | 0.440±0.064 | 0.730±0.164 | 0.450±0.077 |
| **URDI λ=1.0 (ours)** | **0.701±0.026** | 0.429±0.061 | 0.663±0.186 | **0.305±0.003** |
| URDI λ=10 (ours) | 0.692±0.024 | 0.437±0.059 | **0.647±0.195** | 0.506±0.003 |

†Mixup collapses on 2/5 seeds (F1 std=0.135, 5× URDI). URDI achieves the best AURC (p=0.001, paired bootstrap) — the most deployment-relevant metric — with the lowest seed variance.

---

## Repository Structure

```
urdi-das-egs-microseismic/
├── README.md
├── requirements.txt
├── LICENSE
│
├── data_processing/
│   ├── phase0_frisco_pipeline.py   # Cape EGS Frisco-2-P: SGY → numpy arrays
│   └── phase0_forge_pipeline.py    # Utah FORGE 3-2417: SGY → numpy arrays (public)
│
├── architectures/
│   └── models.py                   # All 10 architectures (SE-ResNet, Conformer, GRU, ViT, Mamba...)
│
├── training/
│   ├── train_benchmark.py          # 10-architecture × 5-seed benchmark
│   ├── train_urdi.py               # URDI training (5 seeds × 5 λ values = 25 runs)
│   └── train_dg_baselines.py       # DG baselines: Mixup, RandConv (5 seeds each)
│
├── evaluation/
│   ├── eval_calibration.py         # NLL, Brier, ECE, MCE, UncAUROC
│   ├── eval_selective_prediction.py # AURC, risk-coverage curves
│   ├── eval_temporal_split.py      # Three-domain cross-domain evaluation
│   ├── eval_window_leakage.py      # Window position validity check
│   ├── eval_urdi_baselines.py      # Calibration baselines (temp scaling, entropy reg...)
│   └── eval_rl_vs_supervised.py    # RL vs supervised classifiers on same state features
│
├── rl_policy/
│   ├── phase1_perception.py        # SE-ResNet detection + MC-Dropout uncertainty
│   ├── phase2_reason.py            # Anomaly scoring + rolling event rate
│   ├── phase3_plan.py              # State discretization (18 states)
│   ├── phase4_act.py               # Q-learning agent training (300 episodes)
│   └── phase5_agent.py             # Full pipeline + evaluation vs STA/LTA GT
│
├── figures/
│   └── regenerate_all_figures.py   # Reproduce all paper figures from saved results
│
└── paper/
    ├── tnnls_paper_r1.tex           # LaTeX source (R1 revision)
    └── references_r1.bib            # Bibliography (48 entries)
```

---

## Datasets

| Dataset | Status | DOI | Notes |
|---|---|---|---|
| **Utah FORGE 3-2417** | ✅ Public | [10.15121/1838538](https://doi.org/10.15121/1838538) | 494 samples, zero-shot eval |
| **Cape EGS Frisco-2-P** | 🔒 Embargoed until Sept 2026 | [10.15121/2479174](https://doi.org/10.15121/2479174) | 3,974 samples, training |

The Utah FORGE 3-2417 processing pipeline (`data_processing/phase0_forge_pipeline.py`) is fully reproducible from the public DOE repository. All preprocessing scripts for Frisco-2-P will be released upon dataset embargo expiry.

**Reproducible without Frisco-2-P:** The FORGE processing pipeline, all evaluation scripts on FORGE, and figure regeneration scripts.

---

## Installation

```bash
git clone https://github.com/shokor103072/urdi-das-egs-microseismic.git
cd urdi-das-egs-microseismic
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- PyTorch ≥ 2.0 with CUDA (tested on CUDA 12.1)
- See `requirements.txt` for full list

---

## Quick Start

### 1. Process Utah FORGE 3-2417 (public data)
```bash
# Download from DOE: https://doi.org/10.15121/1838538
python data_processing/phase0_forge_pipeline.py \
    --sgy_dir ./raw_forge/ \
    --out_dir ./Dataset/
# Produces: Dataset/X_forge.npy (494, 1, 361, 2400), Dataset/y_forge.npy
```

### 2. Run URDI training (requires Frisco-2-P)
```bash
python training/train_urdi.py \
    --data_dir ./Dataset/ \
    --model_dir ./Model/ \
    --seeds 42 7 13 99 2024 \
    --lambdas 0.0 0.01 0.1 1.0 10.0
# 5 seeds × 5 lambdas = 25 runs (~8 hrs on RTX 4060)
```

### 3. Evaluate on Utah FORGE 3-2417
```bash
# Calibration metrics (ECE, MCE, NLL, Brier)
python evaluation/eval_calibration.py --model_dir ./Model/ --data_dir ./Dataset/

# Selective prediction (AURC)
python evaluation/eval_selective_prediction.py --model_dir ./Model/ --data_dir ./Dataset/
```

### 4. Run DG baseline comparison
```bash
python training/train_dg_baselines.py \
    --data_dir ./Dataset/ \
    --model_dir ./Model/ \
    --methods baseline mixup randconv
```

### 5. Reproduce all figures
```bash
python figures/regenerate_all_figures.py --data_dir ./data/ --fig_dir ./data/figures/
```

---

## Reproducibility

| Experiment | Seeds | Runs | GPU Time | Crash-safe |
|---|---|---|---|---|
| Architecture benchmark | 5 | 5 models × 5 seeds | ~4 hrs (RTX 4060) | ✅ |
| URDI training | 5 | 25 runs (5×5) | ~8 hrs | ✅ |
| DG baselines | 5 | 15 runs (3×5) | ~2 hrs | ✅ |
| Calibration eval | — | inference only | ~10 min | ✅ |
| RL policy | 5 | 5 seeds | ~30 min | ✅ |

All training scripts support crash recovery via partial CSV files. Multi-seed results use paired bootstrap CIs (n=2,000 resamples).

**Statistical testing:**
- Architecture comparison: corrected paired bootstrap CI (n=2,000)
- URDI vs baseline: paired bootstrap p-values (F1 p=0.31, ECE p=0.24, MCE p=0.032✓, AURC p=0.001✓)
- RL policy: McNemar's test with Bonferroni correction (α=0.0125, K=3 comparisons)

---

## Citation

If you use this code or the URDI method, please cite:

```bibtex
@article{shokor2025urdi,
  title     = {Uncertainty-Regularized Domain Generalization for Cross-Site
               {DAS} Microseismic Detection in Enhanced Geothermal Systems},
  author    = {Shokor and Jaafar, Jafreezal Bin and Paputungan, Irving Vitra},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2025},
  note      = {Under review}
}
```

---

## Advanced Gather Viewer

The open-source seismic labeling tool used for dataset inspection is available at:  
[github.com/shokor103072/Advanced-Gather-Viewer](https://github.com/shokor103072/Advanced-Gather-Viewer)

---

## License

MIT License — see `LICENSE` for details.

The Cape EGS Frisco-2-P dataset is under embargo until September 2026 (Fervo Energy). Researchers wishing to reproduce Frisco-2-P results may contact the corresponding author for preprocessing pipeline scripts.

---

## Contact

**Shokor** — Universiti Teknologi PETRONAS, Malaysia  
Email: shokor@utp.edu.my  
GitHub: [shokor103072](https://github.com/shokor103072)
