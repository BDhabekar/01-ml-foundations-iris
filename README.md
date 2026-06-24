# 01 — ML Foundations: Iris Classifier

First project of the 120-day AI Engineer track. A small, clean, end-to-end supervised classification pipeline — built to practice the full workflow (load → EDA → split → baseline model → evaluation) before adding any complexity.

**Status:** Day 1 baseline complete. Intentionally kept simple — cross-validation lands on Day 3, feature-scaling pipelines on Day 5. See [Known limitations & next steps](#known-limitations--next-steps).

---

## Problem statement

Classify an iris flower into one of three species — *setosa*, *versicolor*, or *virginica* — based on four physical measurements (sepal length, sepal width, petal length, petal width).

## Data source

scikit-learn's built-in `load_iris()` (the classic UCI Iris dataset). 150 samples, 4 numeric features, 3 classes, perfectly balanced (50/50/50), no missing values.

## Target definition

- `target`: integer label (0 = setosa, 1 = versicolor, 2 = virginica)
- `species`: human-readable string, derived from `target` via `iris.target_names`

## Success metric

Accuracy, supported by a full `classification_report` (precision/recall/F1 per class) and a confusion matrix — accuracy alone is reasonable here only because the classes are exactly balanced and there's no asymmetric cost between the three species.

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
│   └── 01_eda_split_and_baseline.ipynb
├── src/
└── images/
    └── petal_scatter_by_species.png
```

## Run instructions

```bash
conda create -n aieng python=3.11 -y
conda activate aieng
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_and_split.ipynb
```
Run all cells top to bottom — no external data download needed, `load_iris()` ships with scikit-learn.

---

## Results

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

---

## Known limitations & next steps

- No cross-validation yet — single split only. → Day 3
- No feature scaling / preprocessing pipeline — logistic regression would likely benefit from it. → Day 5 (`Pipeline` + `ColumnTransformer`)
- No hyperparameter tuning.
- KNN-vs-LogReg comparison above is *not* a statistically meaningful result given the dataset size — flagged deliberately rather than overstated.
- Dataset is small and exceptionally clean (no missing data, perfectly balanced) — not representative of real-world data quality.

## What this project is for

Project 1 in the AI-engineer portfolio — establishes the repo conventions (structure, README format, reproducibility via `random_state`, train/test discipline) that every later project in this track will reuse.