"""Reusable data-cleaning transforms.

Every function is pure: it takes a DataFrame + params and returns
`(new_df, message, code)` where `code` is the equivalent pandas one-liner.
Collecting those `code` strings lets the app EXPORT a reproducible cleaning
script — turning point-and-click cleaning into a real, version-controllable
pipeline (the differentiator vs. a throwaway notebook).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Result = tuple[pd.DataFrame, str, str]


def drop_duplicates(df: pd.DataFrame) -> Result:
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(out)
    return out, f"Dropped {removed} duplicate row(s).", "df = df.drop_duplicates().reset_index(drop=True)"


def drop_columns(df: pd.DataFrame, columns: list[str]) -> Result:
    out = df.drop(columns=columns)
    return out, f"Dropped column(s): {', '.join(columns)}.", f"df = df.drop(columns={columns!r})"


def rename_column(df: pd.DataFrame, old: str, new: str) -> Result:
    out = df.rename(columns={old: new})
    return out, f"Renamed '{old}' → '{new}'.", f"df = df.rename(columns={{{old!r}: {new!r}}})"


def impute_missing(df: pd.DataFrame, columns: list[str], strategy: str, fill_value=None) -> Result:
    """strategy: 'mean' | 'median' | 'mode' | 'constant' | 'drop_rows'."""
    out = df.copy()
    if strategy == "drop_rows":
        before = len(out)
        out = out.dropna(subset=columns).reset_index(drop=True)
        msg = f"Dropped {before - len(out)} row(s) with missing values in {columns}."
        return out, msg, f"df = df.dropna(subset={columns!r}).reset_index(drop=True)"

    code_lines = []
    for col in columns:
        if strategy == "mean":
            val = out[col].mean()
        elif strategy == "median":
            val = out[col].median()
        elif strategy == "mode":
            mode = out[col].mode(dropna=True)
            val = mode.iloc[0] if not mode.empty else fill_value
        elif strategy == "constant":
            val = fill_value
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        out[col] = out[col].fillna(val)
        code_lines.append(f"df[{col!r}] = df[{col!r}].fillna({val!r})")
    return out, f"Imputed missing in {columns} using '{strategy}'.", "\n".join(code_lines)


def convert_dtype(df: pd.DataFrame, column: str, dtype: str) -> Result:
    """dtype: 'int' | 'float' | 'string' | 'category' | 'datetime' | 'bool'."""
    out = df.copy()
    if dtype == "datetime":
        out[column] = pd.to_datetime(out[column], errors="coerce")
        code = f"df[{column!r}] = pd.to_datetime(df[{column!r}], errors='coerce')"
    elif dtype == "int":
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
        code = f"df[{column!r}] = pd.to_numeric(df[{column!r}], errors='coerce').astype('Int64')"
    elif dtype == "float":
        out[column] = pd.to_numeric(out[column], errors="coerce")
        code = f"df[{column!r}] = pd.to_numeric(df[{column!r}], errors='coerce')"
    else:
        out[column] = out[column].astype(dtype)
        code = f"df[{column!r}] = df[{column!r}].astype({dtype!r})"
    return out, f"Converted '{column}' → {dtype}.", code


def handle_outliers(df: pd.DataFrame, columns: list[str], method: str = "iqr", action: str = "clip") -> Result:
    """method: 'iqr' | 'zscore'; action: 'clip' | 'remove'."""
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    code_lines = []
    for col in columns:
        s = out[col]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:  # zscore
            mean, std = s.mean(), s.std(ddof=0)
            low, high = mean - 3 * std, mean + 3 * std
        if action == "clip":
            out[col] = s.clip(low, high)
            code_lines.append(f"df[{col!r}] = df[{col!r}].clip({low:.4g}, {high:.4g})")
        else:
            mask &= s.between(low, high) | s.isna()
            code_lines.append(f"df = df[df[{col!r}].between({low:.4g}, {high:.4g})]")
    if action == "remove":
        before = len(out)
        out = out[mask].reset_index(drop=True)
        msg = f"Removed {before - len(out)} outlier row(s) across {columns} ({method})."
    else:
        msg = f"Clipped outliers in {columns} ({method})."
    return out, msg, "\n".join(code_lines)


def encode_categorical(df: pd.DataFrame, columns: list[str], method: str = "onehot") -> Result:
    """method: 'onehot' | 'label'."""
    out = df.copy()
    if method == "onehot":
        out = pd.get_dummies(out, columns=columns, drop_first=False)
        return out, f"One-hot encoded {columns}.", f"df = pd.get_dummies(df, columns={columns!r})"
    code_lines = []
    for col in columns:
        out[col] = out[col].astype("category").cat.codes
        code_lines.append(f"df[{col!r}] = df[{col!r}].astype('category').cat.codes")
    return out, f"Label encoded {columns}.", "\n".join(code_lines)


def scale_numeric(df: pd.DataFrame, columns: list[str], method: str = "standard") -> Result:
    """method: 'standard' (z-score) | 'minmax'. Transparent formulas (exportable)."""
    out = df.copy()
    code_lines = []
    for col in columns:
        s = out[col]
        if method == "standard":
            mean, std = s.mean(), s.std(ddof=0)
            std = std or 1.0
            out[col] = (s - mean) / std
            code_lines.append(f"df[{col!r}] = (df[{col!r}] - {mean:.6g}) / {std:.6g}")
        else:  # minmax
            lo, hi = s.min(), s.max()
            rng = (hi - lo) or 1.0
            out[col] = (s - lo) / rng
            code_lines.append(f"df[{col!r}] = (df[{col!r}] - {lo:.6g}) / {rng:.6g}")
    return out, f"Scaled {columns} ({method}).", "\n".join(code_lines)


def filter_rows(df: pd.DataFrame, query: str) -> Result:
    before = len(df)
    out = df.query(query).reset_index(drop=True)
    return out, f"Filtered rows where `{query}` — kept {len(out)}/{before}.", f"df = df.query({query!r}).reset_index(drop=True)"
