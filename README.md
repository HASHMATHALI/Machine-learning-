# Two small machine-learning projects, with their real scores

Two self-contained scikit-learn scripts. Each runs from the command line, saves
its outputs next to itself, and the numbers below are what the scripts print.

| Project | Task | Data | Result |
|---|---|---|---|
| [diabetes/](diabetes) | Classification | Pima Indians Diabetes, 768 rows | ROC AUC 0.81 on 154 held-out rows; finds about half of the diabetic patients |
| [clustering/](clustering) | Segmentation | Mall Customers, 200 rows | 6 segments, silhouette 0.43 |

## Diabetes prediction

Trains Logistic Regression, Random Forest and an SVM, each tuned with a 5-fold
grid search, and saves the best one.

| Model | Cross-validated ROC AUC | Test ROC AUC | Test accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| SVM (linear, C=10), selected | 0.845 | 0.814 | 0.701 | 0.591 | 0.481 |
| Logistic Regression (C=0.1) | 0.844 | 0.810 | 0.701 | 0.583 | 0.519 |
| Random Forest | 0.836 | 0.810 | 0.740 | 0.667 | 0.519 |

What this means:

- The three models are indistinguishable on this data; the differences are well inside the noise of a 154-row test set.
- At the default threshold the selected model catches only 48% of the diabetic
  patients in the test set. That is too low for screening; lowering the
  decision threshold would trade precision for recall.
- The model is chosen by cross-validation on the training split. The test set
  is used only to report scores.
- Zeros in glucose, blood pressure, skin thickness, insulin and BMI are treated
  as "not recorded" and imputed with the median, both in training and when
  scoring new rows.

```bash
cd diabetes
```

```bash
python diabetes_ml.py
```

```bash
python diabetes_ml.py --predict new_rows.csv
```

Outputs: `metrics_summary.json`, one `metrics_<model>.json` per model,
`best_model.joblib`, `correlation_heatmap.png`, `predictions.csv`.

## Customer segmentation

Clusters customers on age, annual income and spending score with K-Means. The
number of clusters is chosen by silhouette score over k = 2 to 10 rather than
fixed in advance; the best is k = 6 (0.428, against 0.417 for k = 5).

| Segment | Customers | Mean age | Income (k$) | Spending score |
|---|---:|---:|---:|---:|
| Older, mid income, mid spend | 45 | 56.3 | 54.3 | 49.1 |
| Young, mid income, mid spend | 39 | 26.8 | 57.1 | 48.1 |
| High income, high spend | 39 | 32.7 | 86.5 | 82.1 |
| High income, low spend | 33 | 41.9 | 88.9 | 17.0 |
| Young, low income, high spend | 23 | 25.0 | 25.3 | 77.6 |
| Low income, low spend | 21 | 45.5 | 26.3 | 19.4 |

A silhouette of 0.43 means the groups overlap; treat them as a useful
description, not as sharply separate types.

```bash
cd clustering
```

```bash
python cust.py
```

Outputs: `elbow_silhouette.png`, `clusters.png`, `cluster_profiles.csv`,
`segments.csv`. Use `--k 5` to force a number of clusters.

## Install

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Not in this repository

An earlier version of this README also described a car-price regression and a
separate exploratory-analysis project. Their code was never added here, so they
have been removed from the description.
