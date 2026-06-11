"""Anomaly / "records to review" detection.

Generic, dataset-agnostic outlier detection using an Isolation Forest over all
usable columns (numeric + low-cardinality categoricals, scaled & encoded). This
is the honest, general form of "fraud detection": it surfaces records that look
statistically unlike the rest — exactly what you'd triage first in claims,
transactions, or billing data — without hard-coding rules for any one domain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_anomalies(df: pd.DataFrame, sensitivity: float = 0.03, max_categories: int = 25):
    """Flag unusual rows.

    `sensitivity` is the expected share of anomalies (0.01–0.15 is sensible).
    Returns (scored_df, flagged_df, summary):
      scored_df  – original data + 'anomaly_score' (higher = more unusual) + 'flagged'
      flagged_df – just the flagged rows, most-unusual first
      summary    – dict with counts + which fields most separate flagged vs normal
    """
    work = df.copy()
    # Identifier-like columns (all unique) carry no anomaly signal — drop them.
    id_cols = [c for c in work.columns if work[c].nunique(dropna=True) == len(work) and len(work) > 0]
    feat = work.drop(columns=id_cols, errors="ignore")

    num = feat.select_dtypes(include="number").columns.tolist()
    cat = [c for c in feat.select_dtypes(exclude="number").columns
           if 1 < feat[c].nunique(dropna=True) <= max_categories]

    if not num and not cat:
        raise ValueError("No usable feature columns for anomaly detection.")

    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    X = pre.fit_transform(feat[num + cat])

    model = IsolationForest(contamination=sensitivity, n_estimators=200, random_state=42, n_jobs=-1)
    labels = model.fit_predict(X)            # -1 = anomaly, 1 = normal
    scores = model.score_samples(X)          # higher = more normal

    scored = df.copy()
    scored["anomaly_score"] = np.round(-scores, 4)   # flip so higher = more unusual
    scored["flagged"] = labels == -1
    flagged = scored[scored["flagged"]].sort_values("anomaly_score", ascending=False)

    summary = {
        "records": int(len(scored)),
        "flagged": int(scored["flagged"].sum()),
        "flagged_pct": round(scored["flagged"].mean() * 100, 2),
        "drivers": _top_drivers(scored, num),
        "features_used": num + cat,
    }
    return scored, flagged, summary


def _top_drivers(scored: pd.DataFrame, num_cols: list[str], top: int = 3) -> list[str]:
    """Plain-English note on which numeric fields differ most for flagged rows."""
    if not scored["flagged"].any():
        return []
    drivers = []
    norm = scored[~scored["flagged"]]
    flag = scored[scored["flagged"]]
    diffs = []
    for c in num_cols:
        n_mean, f_mean, n_std = norm[c].mean(), flag[c].mean(), norm[c].std(ddof=0)
        if n_std and not np.isnan(f_mean):
            diffs.append((abs(f_mean - n_mean) / n_std, c, n_mean, f_mean))
    for _, c, n_mean, f_mean in sorted(diffs, reverse=True)[:top]:
        direction = "higher" if f_mean > n_mean else "lower"
        drivers.append(f"Flagged records have {direction} {c.replace('_', ' ')} "
                       f"(avg {f_mean:,.1f} vs {n_mean:,.1f}).")
    return drivers
