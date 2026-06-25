# 01 — ML Foundations: Iris Classifier + Linear Regression from Scratch

First project of the 120-day AI Engineer track. It began as a small, clean, end-to-end supervised classification pipeline (Day 1) and grew a second foundational notebook (Day 2) that builds linear regression and gradient descent from scratch in NumPy, then circles back to evaluate the Day 1 model more rigorously. Both notebooks live here together because they're small, tightly related ML-foundations exercises — not separate projects.

**Status:** Day 1 + Day 2 complete. Intentionally kept simple — cross-validation lands on Day 3, feature-scaling pipelines on Day 5. See [Known limitations & next steps](#known-limitations--next-steps).

---

## Notebooks

| Notebook | Day | What it covers |
|---|---|---|
| `01_eda_split_and_baseline.ipynb` | Day 1 | Iris EDA → stratified split → `LogisticRegression` baseline → confusion matrix, coefficients, 5-NN comparison |
| `02_linear_regression_scratch.ipynb` | Day 2 | Linear regression + gradient descent from scratch (NumPy), verified against the closed-form solution and `sklearn`; batch vs. mini-batch vs. SGD; L1 vs. L2 regularization; richer evaluation (precision/recall/F1/ROC-AUC) of the Day 1 Iris model |

---

## Problem statement

Primary task (Day 1): classify an iris flower into one of three species — *setosa*, *versicolor*, or *virginica* — from four physical measurements (sepal length, sepal width, petal length, petal width).

Day 2 adds a separate from-scratch learning exercise: fit a linear regression model with gradient descent on synthetic data, and reuse the Iris classifier as the subject of a deeper evaluation pass.

## Data source

- **Iris (Day 1):** scikit-learn's built-in `load_iris()` (the classic UCI Iris dataset). 150 samples, 4 numeric features, 3 classes, perfectly balanced (50/50/50), no missing values.
- **Synthetic regression (Day 2):** a generated linear dataset (known true `weights`/`bias` plus noise) so the from-scratch model's recovered parameters can be checked against ground truth.

## Target definition

- `target`: integer label (0 = setosa, 1 = versicolor, 2 = virginica)
- `species`: human-readable string, derived from `target` via `iris.target_names`
- Day 2 synthetic data: a continuous numeric target generated from a known linear relationship.

## Success metric

Day 1 / Iris: accuracy, supported by a full `classification_report` (precision/recall/F1 per class) and a confusion matrix — accuracy alone is reasonable here only because the classes are exactly balanced and there's no asymmetric cost between the three species. Day 2 extends this same Iris evaluation with macro-averaged ROC-AUC.

Day 2 / from-scratch regression: agreement of the recovered `weights`/`bias` with both the closed-form solution and `sklearn.LinearRegression`, to 4 decimal places.

## Validation strategy

A single stratified holdout split: `test_size=0.2`, `random_state=42`, `stratify=y` → 120 train / 30 test, with class balance preserved in both. **This is a placeholder validation strategy** — a single split on 150 rows isn't robust enough to declare a "winner" between models. Proper k-fold cross-validation is deliberately scheduled for Day 3, not added here.

## Baseline

`LogisticRegression(max_iter=200)`, trained on the training split only, evaluated once on the held-out test split.

---

## Repo structure

```
01-ml-foundations-iris/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_split_and_baseline.ipynb
│   └── 02_linear_regression_scratch.ipynb
├── src/
└── images/
    ├── petal_scatter_by_species.png
    ├── linear_data.png
    ├── linear_fit.png
    ├── gd_cost_curve.png
    └── gd_variants.png
```

## Run instructions

```bash
conda create -n aieng python=3.11 -y
conda activate aieng
pip install -r requirements.txt
jupyter notebook
```
Open either notebook under `notebooks/` and run all cells top to bottom — no external data download needed. `load_iris()` ships with scikit-learn, and the Day 2 regression data is generated in-notebook.

---

## Results

### Day 1 — Iris baseline

| Check | Result |
|---|---|
| Dataset shape | 150 rows × 4 features |
| Class balance | 50 / 50 / 50 (setosa / versicolor / virginica) |
| Missing values | 0 |
| Train / test split | 120 / 30 (stratified, `random_state=42`) |
| **Baseline accuracy (LogisticRegression)** | **0.9667** (29/30 correct) |
| Confusion matrix | 1 miss: versicolor → virginica; setosa perfect |

**Plot:** `images/petal_scatter_by_species.png` — petal length vs. petal width, colored by species. Setosa forms a fully separate cluster; versicolor and virginica overlap slightly at the boundary, which lines up exactly with the model's one misclassification.

**Coefficient check:** `model.coef_` shows petal length/width carrying far larger magnitude than the sepal measurements — consistent with the plot, the model is relying on the features that visually separate the classes.

**Secondary comparison (not a verdict):** A 5-NN classifier scored 1.0 on this same split. Noted for context only — a single 30-row test set can't reliably distinguish two models this close; that comparison only becomes meaningful once cross-validation is in place (Day 3).

### Day 2 — Linear regression from scratch + deeper Iris evaluation

| Check | Result |
|---|---|
| From-scratch GD vs. closed-form vs. `sklearn` | All three agree to 4 decimals (bias ≈ 4.0842, weight ≈ 2.9688) |
| Batch vs. mini-batch vs. SGD | Same solution; convergence gets noisier as the batch shrinks (SGD noisiest) |
| L1 (Lasso) vs. L2 (Ridge) | Lasso zeroed 6 of 10 known-zero synthetic weights; L2 shrank but kept all |
| Iris re-evaluation (Day 1 model) | precision / recall / F1 per class + **macro ROC-AUC = 1.0** |

**Parameter convention:** the from-scratch model keeps `weights` and `bias` as separate named attributes (not a single combined vector with a bias column appended to `X`) — matching scikit-learn / PyTorch and the NumPy MLP coming later in the track.

**Verification standard:** the from-scratch implementation is cross-checked against `sklearn` to confirm numerical agreement before any conclusions are drawn — the same discipline that carries into every later from-scratch build.

**Plots:**
- `images/linear_data.png` — the synthetic dataset (true rule: y = 4 + 3x + noise).
- `images/linear_fit.png` — the from-scratch model's fitted line over that data.
- `images/gd_cost_curve.png` — average error (cost) shrinking as gradient descent runs.
- `images/gd_variants.png` — batch vs. mini-batch vs. SGD under the same learning rate.

The L1-vs-L2 regularization and the Iris ROC-AUC re-evaluation live in the notebook as tables/printouts — they weren't exported as image files.

---

## Known limitations & next steps

- No cross-validation yet — single split only. → Day 3
- No feature scaling / preprocessing pipeline — logistic regression would likely benefit from it. → Day 5 (`Pipeline` + `ColumnTransformer`)
- No hyperparameter tuning.
- KNN-vs-LogReg comparison above is *not* a statistically meaningful result given the dataset size — flagged deliberately rather than overstated.
- Day 2's linear regression uses synthetic data by design — the point is to verify a from-scratch implementation against known ground truth, not to model a real dataset.
- Iris dataset is small and exceptionally clean (no missing data, perfectly balanced) — not representative of real-world data quality.

## What this project is for

Project 1 in the AI-engineer portfolio — establishes the repo conventions (structure, README format, reproducibility via `random_state`, train/test discipline, from-scratch-vs-library verification) that every later project in this track reuses.
