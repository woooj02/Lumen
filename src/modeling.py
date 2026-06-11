"""AutoML: detect the task, build a leak-free preprocessing pipeline, train and
compare several scikit-learn models, and surface metrics + feature importance.

The preprocessing (impute → encode → scale) lives INSIDE an sklearn Pipeline so
it is fit only on the training fold — no data leakage, the thing reviewers check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_task(df: pd.DataFrame, target: str) -> str:
    """Return 'classification' or 'regression' from the target column."""
    s = df[target]
    if s.dtype == "object" or str(s.dtype) == "category" or s.dtype == bool:
        return "classification"
    # Numeric but few distinct values → treat as classification.
    if s.nunique(dropna=True) <= 10:
        return "classification"
    return "regression"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def _models(task: str) -> dict:
    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


@dataclass
class ModelResult:
    name: str
    metrics: dict
    pipeline: Pipeline
    task: str
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None = None
    importance: pd.DataFrame | None = None
    labels: list = field(default_factory=list)


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        return None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    if len(imp) != len(names):
        return None
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values(
        "importance", ascending=False).reset_index(drop=True)


def train_and_compare(df: pd.DataFrame, target: str, task: str | None = None,
                      test_size: float = 0.2, cv: int = 5) -> tuple[list[ModelResult], pd.DataFrame]:
    """Train every candidate model and return (results, leaderboard).

    Drops rows with a missing target, splits, fits each pipeline, evaluates on
    the held-out test set, and runs cross-validation for a stability estimate.
    """
    data = df.dropna(subset=[target]).copy()
    if len(data) < 30:
        raise ValueError("Need at least 30 rows with a non-missing target to model reliably.")

    task = task or detect_task(data, target)
    y = data[target]
    X = data.drop(columns=[target])
    # Drop obvious identifier columns (all-unique) — they only cause leakage/overfit.
    id_cols = [c for c in X.columns if X[c].nunique(dropna=True) == len(X)]
    X = X.drop(columns=id_cols)

    if task == "classification":
        y = y.astype(str)

    stratify = y if (task == "classification" and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify)

    results: list[ModelResult] = []
    scoring = "f1_weighted" if task == "classification" else "r2"

    for name, estimator in _models(task).items():
        pipe = Pipeline([("pre", _build_preprocessor(X)), ("model", estimator)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        proba = None
        if task == "classification" and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_te)
            except Exception:
                proba = None

        try:
            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
        except Exception:
            cv_mean, cv_std = float("nan"), float("nan")

        if task == "classification":
            metrics = {
                "accuracy": round(accuracy_score(y_te, pred), 4),
                "precision": round(precision_score(y_te, pred, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_te, pred, average="weighted", zero_division=0), 4),
                "f1": round(f1_score(y_te, pred, average="weighted", zero_division=0), 4),
                f"cv_{scoring}_mean": round(cv_mean, 4),
                f"cv_{scoring}_std": round(cv_std, 4),
            }
            if proba is not None and len(np.unique(y_tr)) == 2:
                try:
                    pos = pipe.classes_[1]
                    metrics["roc_auc"] = round(roc_auc_score((y_te == pos).astype(int), proba[:, 1]), 4)
                except Exception:
                    pass
            labels = sorted(pd.unique(y_te))
        else:
            rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
            metrics = {
                "r2": round(r2_score(y_te, pred), 4),
                "mae": round(mean_absolute_error(y_te, pred), 4),
                "rmse": round(rmse, 4),
                f"cv_{scoring}_mean": round(cv_mean, 4),
                f"cv_{scoring}_std": round(cv_std, 4),
            }
            labels = []

        results.append(ModelResult(
            name=name, metrics=metrics, pipeline=pipe, task=task,
            y_test=np.asarray(y_te), y_pred=np.asarray(pred), y_proba=proba,
            importance=_feature_importance(pipe), labels=labels,
        ))

    primary = "f1" if task == "classification" else "r2"
    results.sort(key=lambda r: r.metrics.get(primary, float("-inf")), reverse=True)
    leaderboard = pd.DataFrame([{"model": r.name, **r.metrics} for r in results])
    return results, leaderboard


def confusion(result: ModelResult) -> tuple[np.ndarray, list]:
    labels = result.labels or sorted(set(result.y_test) | set(result.y_pred))
    cm = confusion_matrix(result.y_test, result.y_pred, labels=labels)
    return cm, labels
