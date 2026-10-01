# SIT307 11.1HD: Reproduction and Leakage-Free Re-evaluation of a Stacking Ensemble for Heart Disease Prediction

This package reproduces and critically re-evaluates:

> M. Bhagat, A. Sharma, P. Agarwal, "An efficient stacking-based ensemble technique for early heart attack prediction," *Multimedia Tools and Applications*, vol. 84, pp. 36351–36375, 2025, doi: 10.1007/s11042-024-19293-7.

- **Part 1** reproduces the paper's pipeline (six classifiers + 5-fold stacking, random 80/20 split) and compares the results with Table 11 of the paper.
- **Part 2** proposes and evaluates a leakage-free protocol (record-level deduplication, repeated stratified CV, nested hyperparameter tuning, corrected statistical tests).

## 1. Folder structure

```
sit307_11_1HD/
├── README.md
├── requirements.txt
├── data/
│   ├── heart.csv                      Kaggle Heart Disease Dataset used by the paper (1025 rows)
│   └── processed.cleveland.data       original UCI Cleveland file, used for the cross-check in Part 1 (Section 7.1)
├── src/
│   ├── heart_utils.py                 preprocessing, models, stacking, metrics (Part 1 and 2)
│   └── proposed.py                    nested CV, tuning grids, statistical tests (Part 2)
├── notebooks/
│   ├── part1_reproduction.ipynb       Part 1: reproduction of the paper
│   └── part2_proposed.ipynb           Part 2: proposed leakage-free protocol
├── figures/                           created when the notebooks run
└── results/                           created when the notebooks run (CSV tables)
```

The notebooks locate the project root automatically (the folder that contains `src/`), so they can be run from the project root or from `notebooks/`.

## 2. Installation

Tested with Python 3.11. Using conda:

```bash
conda create -n heart python=3.11
conda activate heart
pip install -r requirements.txt
```

`requirements.txt` pins the exact library versions used to produce the reported results.

## 3. How to reproduce the results

Run the notebooks **in this order**, each with *Restart Kernel and Run All*:

1. `notebooks/part1_reproduction.ipynb` (about 1–2 minutes)
2. `notebooks/part2_proposed.ipynb` (about 6–20 minutes depending on the number of CPU cores)

Part 2 reads two files written by Part 1 (`results/paper_table11.csv` and `results/part1_multiseed_raw.csv`), so Part 1 must be run first.

All random seeds are fixed in the notebooks, so re-running produces the same numbers.

### Resuming Part 2

Part 2 saves each outer fold of the nested cross-validation to `results/cache/`. If a run is interrupted, running the notebook again resumes from the last finished fold. **Delete `results/cache/` to recompute everything from scratch.** This package is distributed without the cache, so a first run always computes all results.

## 4. Expected key results

| Result | Value |
|---|---|
| Part 1: stacking test accuracy (seed 42, paper protocol) | 0.9854 (paper: 0.9853) |
| Part 1: unique records in the 1025-row file | 302 |
| Part 1: mean share of test rows with an identical copy in the training set (20 splits) | 96.6% |
| Part 2, Exp. 1: optimism of the paper protocol (internal estimate − held-out accuracy) | +0.207 |
| Part 2, Exp. 1: optimism of the proposed protocol | +0.004 |
| Part 2, Exp. 2: tuned stacking accuracy, nested CV (50 outer folds) | 0.835 |

Reproducibility check: both notebooks were executed independently on Windows (xgboost 3.2.0) and Linux (xgboost 3.4.1). All accuracies, selected hyperparameters and statistical conclusions were identical; the only difference was an AUC change of 0.0001 for the default stacking model, caused by the different xgboost version. The numbers in the report were produced with the versions in `requirements.txt`.

## 5. Outputs

- `figures/`: all figures used in the report (`p1_*.png` for Part 1, `p2_*.png` for Part 2).
- `results/`: CSV tables behind every number in the report, for example `part1_multiseed_accuracy_summary.csv`, `part1_full_metric_comparison.csv`, `part1_tree_depth_sweep_summary.csv`, `part1_stacking_limited_trees.csv`, `part1_missing_code_sensitivity_summary.csv`, `part2_exp1_summary.csv`, `part2_exp1b_duplicate_training_effect.csv`, `part2_exp2_summary.csv`, `part2_exp2_tests.csv`, `part2_exp2_test_sensitivity.csv`, `final_comparison.csv`.

## 6. Dataset notes

- Source used by the paper: Kaggle Heart Disease Dataset, https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset (file `heart.csv`, included in `data/`).
- The file contains 1025 rows but only 302 unique records. Part 1 keeps all rows to reproduce the paper; Part 2 removes exact duplicates before any split.
- The label `target` is used exactly as provided in the file, following the paper.
- `processed.cleveland.data` is the processed Cleveland file of the UCI Heart Disease repository (A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, UCI Machine Learning Repository, 1989, doi: 10.24432/C52P4X), redistributed under its CC BY 4.0 licence. Part 1 (Section 7.1) uses it to show that all 302 unique Kaggle records match the Cleveland database and that the Kaggle label and several category codes are renumbered relative to the original data.

## 7. Use of Generative AI

See the "Acknowledgement of GenAI use" section of the technical report.
