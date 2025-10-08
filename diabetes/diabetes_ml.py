import pandas as pd
import numpy as np
import argparse
import json
import joblib
import os
from typing import List

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------- utilities --------------------

PIMA_ZERO_TO_NAN = {"glucose", "bloodpressure", "skinthickness", "insulin", "bmi"}

def load_data(path):
    """Loads CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    print("\n✅ Data Loaded Successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    return df

def pima_fix(df: pd.DataFrame) -> pd.DataFrame:
    """Convert zeros to NaN for typical Pima features where zero is invalid."""
    d = df.copy()
    for c in d.columns:
        if c.lower() in PIMA_ZERO_TO_NAN and pd.api.types.is_numeric_dtype(d[c]):
            d.loc[d[c] == 0, c] = np.nan
    return d

def eda_summary(df):
    """Performs basic EDA and saves correlation heatmap."""
    print("📊 Basic EDA Summary:\n")
    print(df.describe(include="all"))
    print("\nMissing Values:\n", df.isnull().sum())
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()
        print("\n🖼️ Correlation heatmap saved as 'correlation_heatmap.png'\n")

def evaluate(y_true, y_pred, y_proba=None):
    """Computes all classification metrics."""
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    if y_proba is not None:
        try:
            results["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            results["roc_auc"] = None
    print("\n📈 Evaluation Metrics:\n", json.dumps(results, indent=2))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return results

def save_feature_signature(columns: List[str], path: str = "feature_order.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": columns}, f, indent=2)

def load_feature_signature(path: str = "feature_order.json") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["feature_columns"]

def align_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    d = df[[c for c in df.columns if c in feature_cols]].copy()
    for c in feature_cols:
        if c not in d.columns:
            d[c] = np.nan
    return d[feature_cols]


# -------------------- training & prediction --------------------

def train_models(X_train, X_test, y_train, y_test, model_out: str):
    """Trains Logistic Regression, Random Forest, and SVM; picks best by ROC AUC."""
    common_steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    models = {
        "LogisticRegression": (Pipeline(common_steps + [('clf', LogisticRegression(max_iter=2000, solver="liblinear", random_state=42))]),
                               {'clf__C': [0.01, 0.1, 1, 10]}),
        "RandomForest": (Pipeline(common_steps + [('clf', RandomForestClassifier(random_state=42, n_jobs=-1))]),
                         {'clf__n_estimators': [200, 400], 'clf__max_depth': [None, 6, 12], 'clf__min_samples_split': [2, 5]}),
        "SVC": (Pipeline(common_steps + [('clf', SVC(probability=True, random_state=42))]),
                {'clf__C': [0.1, 1, 10], 'clf__kernel': ['rbf', 'linear']})
    }

    best_model, best_name, best_auc = None, None, -1.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (pipe, params) in models.items():
        print(f"\n🚀 Training {name} ...")
        gs = GridSearchCV(pipe, params, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        y_pred = gs.best_estimator_.predict(X_test)
        try:
            y_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
        metrics = evaluate(y_test, y_pred, y_proba)

        # save metrics per model
        with open(f"metrics_{name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        auc = metrics.get("roc_auc", 0.0) or 0.0
        if auc > best_auc:
            best_auc = auc
            best_model = gs.best_estimator_
            best_name = name

    print(f"\n🏆 Best Model: {best_name} with ROC AUC: {best_auc:.3f}")
    joblib.dump(best_model, model_out)
    print(f"💾 Model saved as {model_out}\n")
    return best_model

def predict_new(model_path, input_csv, feature_sig_path: str = "feature_order.json"):
    """Predicts diabetes outcomes for new data."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model = joblib.load(model_path)
    new_data = pd.read_csv(input_csv)

    # align to training features if signature exists
    if os.path.exists(feature_sig_path):
        cols = load_feature_signature(feature_sig_path)
        new_data = align_columns(new_data, cols)

    preds = model.predict(new_data)
    proba = None
    try:
        proba = model.predict_proba(new_data)[:, 1]
    except Exception:
        pass

    out = new_data.copy()
    out["Prediction"] = preds
    if proba is not None:
        out["Probability"] = proba
    out.to_csv("predictions.csv", index=False)
    print("✅ Predictions saved to predictions.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes.csv", help="Path to diabetes dataset")
    parser.add_argument("--target_col", type=str, default="Outcome", help="Target column name")
    parser.add_argument("--predict", type=str, default=None, help="CSV file for prediction")
    parser.add_argument("--model_out", type=str, default="best_model.joblib", help="Where to save the trained model")
    parser.add_argument("--disable_pima_fix", action="store_true", help="Disable zero→NaN fix for Pima columns")
    args = parser.parse_args()

    if args.predict:
        predict_new(args.model_out, args.predict)
    else:
        df = load_data(args.data_path)
        if not args.disable_pima_fix:
            df = pima_fix(df)
        eda_summary(df)

        if args.target_col not in df.columns:
            raise ValueError(f"Target column '{args.target_col}' not found in dataset.")
        X = df.drop(columns=[args.target_col])
        y = df[args.target_col]
        if not pd.api.types.is_numeric_dtype(y):
            y = y.astype('category').cat.codes  # handle Yes/No

        # Save raw feature order so predict CSVs can be aligned later
        save_feature_signature(X.columns.tolist())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        train_models(X_train, X_test, y_train, y_test, model_out=args.model_out)


if __name__ == "__main__":
    main()
