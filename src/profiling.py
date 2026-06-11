"""Automated dataset profiling — the numbers behind the "Profile" page.

Pure pandas/numpy so it is fast, dependency-light, and unit-testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def overview(df: pd.DataFrame) -> dict:
    """High-level dataset health metrics."""
    n_cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing,
        "missing_pct": round((missing / n_cells * 100) if n_cells else 0.0, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": int(df.select_dtypes(include="number").shape[1]),
        "categorical_cols": int(df.select_dtypes(exclude="number").shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }


def _first_valid(series: pd.Series):
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else None


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile table: dtype, missing, cardinality, example value."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_%": round(s.isna().mean() * 100, 2),
                "unique": int(s.nunique(dropna=True)),
                "unique_%": round(s.nunique(dropna=True) / n * 100, 2) if n else 0.0,
                "example": _first_valid(s),
            }
        )
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """describe() for numeric columns plus skewness — empty frame if none."""
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc["skew"] = num.skew(numeric_only=True)
    desc["missing"] = num.isna().sum()
    return desc.round(3)


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Top values + cardinality for non-numeric columns."""
    cat = df.select_dtypes(exclude="number")
    rows = []
    for col in cat.columns:
        vc = cat[col].value_counts(dropna=True)
        top = ", ".join(f"{idx} ({cnt})" for idx, cnt in vc.head(top_n).items())
        rows.append({"column": col, "unique": int(cat[col].nunique()), "top_values": top})
    return pd.DataFrame(rows)


def quality_flags(df: pd.DataFrame) -> list[str]:
    """Heuristic data-quality warnings shown as actionable callouts."""
    flags = []
    dup = int(df.duplicated().sum())
    if dup:
        flags.append(f"{dup} duplicate row(s) detected — consider dropping them.")

    high_missing = df.columns[df.isna().mean() > 0.4].tolist()
    if high_missing:
        flags.append(f"{len(high_missing)} column(s) >40% missing: {', '.join(high_missing[:6])}")

    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant:
        flags.append(f"Constant/empty column(s) carry no signal: {', '.join(constant[:6])}")

    id_like = [
        c for c in df.columns
        if df[c].nunique(dropna=True) == len(df) and len(df) > 0
    ]
    if id_like:
        flags.append(f"ID-like column(s) (all unique): {', '.join(id_like[:6])} — exclude from modelling.")

    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna()
        if len(s) > 10 and abs(s.skew()) > 2:
            flags.append(f"'{c}' is highly skewed (skew={s.skew():.1f}) — a log transform may help.")
    return flags
