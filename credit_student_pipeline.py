#!/usr/bin/env python3
"""
Quick-and-dirty student-style script for credit default prediction with
RandomForest, SVM and a lightweight ANFIS-like classifier.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore")


class ChillANFIS(BaseEstimator, ClassifierMixin):
    """Loosely inspired ANFIS classifier using Gaussian rules + logistic head."""

    def __init__(self, n_rules=6, fuzz_sigma=1.0, max_iter=400, random_state=42):
        self.n_rules = n_rules
        self.fuzz_sigma = fuzz_sigma
        self.max_iter = max_iter
        self.random_state = random_state
        self._rule_centers = None
        self._head_clf = None

    def _gauss_feats(self, X):
        if self._rule_centers is None:
            raise RuntimeError("Model not fitted")
        dmat = np.square(X[:, None, :] - self._rule_centers[None, :, :]).sum(axis=2)
        phi = np.exp(-dmat / (2.0 * (self.fuzz_sigma**2)))
        return np.hstack([phi, np.ones((phi.shape[0], 1))])

    def fit(self, X, y):
        km = MiniBatchKMeans(
            n_clusters=self.n_rules,
            random_state=self.random_state,
            batch_size=512,
            max_iter=200,
        )
        km.fit(X)
        self._rule_centers = km.cluster_centers_
        feats = self._gauss_feats(X)
        self._head_clf = LogisticRegression(
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self._head_clf.fit(feats, y)
        return self

    def predict_proba(self, X):
        feats = self._gauss_feats(X)
        return self._head_clf.predict_proba(feats)

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5


def clean_up_dataframe(raw_df):
    df_tmp = raw_df.copy()
    if "ID" in df_tmp.columns:
        df_tmp = df_tmp.drop(columns=["ID"])
    num_columns = df_tmp.select_dtypes(include=[np.number]).columns.tolist()
    for goofy_col in num_columns:
        low_q = df_tmp[goofy_col].quantile(0.01)
        high_q = df_tmp[goofy_col].quantile(0.99)
        df_tmp[goofy_col] = df_tmp[goofy_col].clip(lower=low_q, upper=high_q)
    return df_tmp


def prep_features(df_clean, target_col):
    y_blob = df_clean[target_col].astype(int)
    X_blob = df_clean.drop(columns=[target_col])
    maybe_cat = ["SEX", "EDUCATION", "MARRIAGE"]
    catty_cols = [c for c in maybe_cat if c in X_blob.columns]
    numstuff_cols = [c for c in X_blob.columns if c not in catty_cols]
    column_magician = ColumnTransformer(
        transformers=[
            ("nums_part", MinMaxScaler(), numstuff_cols),
            ("cats_part", OneHotEncoder(handle_unknown="ignore"), catty_cols),
        ]
    )
    return X_blob, y_blob, column_magician


def run_grid(model, params, Xfit, yfit):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring="f1",
        n_jobs=-1,
        cv=folds,
        verbose=0,
    )
    gs.fit(Xfit, yfit)
    return gs


def evaluate_model(model_name, fitted_model, Xte, yte):
    preds = fitted_model.predict(Xte)
    proba = (
        fitted_model.predict_proba(Xte)[:, 1]
        if hasattr(fitted_model, "predict_proba")
        else None
    )
    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds)
    rec = recall_score(yte, preds)
    auc = roc_auc_score(yte, proba) if proba is not None else float("nan")
    cm = confusion_matrix(yte, preds)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Recall (defaut): {rec:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("Confusion matrix:")
    print(cm)
    if proba is not None:
        fpr, tpr, _ = roc_curve(yte, proba)
        print(f"ROC curve points (first 5): {list(zip(fpr[:5], tpr[:5]))}")


def main():
    data_path = Path("default of credit card clients.xls")
    if not data_path.exists():
        raise FileNotFoundError("Dataset file is missing next to this script.")

    raw_df = pd.read_excel(data_path, header=1)
    target_col = "target"

    tidy_df = clean_up_dataframe(raw_df)
    X_raw, y_raw, column_prep = prep_features(tidy_df, target_col)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=0.2,
        stratify=y_raw,
        random_state=42,
    )

    column_prep.fit(X_train_raw)
    X_train_ready = column_prep.transform(X_train_raw)
    X_test_ready = column_prep.transform(X_test_raw)

    smo = SMOTE(random_state=42)
    X_balanced, y_balanced = smo.fit_resample(X_train_ready, y_train_raw)

    rf_params = {
        "n_estimators": [200, 400],
        "max_depth": [8, 14],
        "min_samples_leaf": [1, 4],
    }
    rf_grid = run_grid(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        rf_params,
        X_balanced,
        y_balanced,
    )

    svm_params = {
        "C": [1.0, 5.0],
        "gamma": ["scale", 0.01],
    }
    svm_grid = run_grid(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        svm_params,
        X_balanced,
        y_balanced,
    )

    anfis_params = {
        "n_rules": [4, 6],
        "fuzz_sigma": [0.5, 1.0],
    }
    anfis_grid = run_grid(
        ChillANFIS(random_state=42),
        anfis_params,
        X_balanced,
        y_balanced,
    )

    best_rf = rf_grid.best_estimator_
    best_svm = svm_grid.best_estimator_
    best_anfis = anfis_grid.best_estimator_

    evaluate_model("RandomForest", best_rf, X_test_ready, y_test_raw)
    evaluate_model("SVM RBF", best_svm, X_test_ready, y_test_raw)
    evaluate_model("ChillANFIS", best_anfis, X_test_ready, y_test_raw)


if __name__ == "__main__":
    main()
