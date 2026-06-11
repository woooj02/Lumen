"""Statistical analysis helpers — correlations, group-by, and hypothesis tests.

Wraps scipy so the "Analyze" page can offer real inferential statistics
(t-test, ANOVA, chi-square) and not just descriptive numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """method: 'pearson' | 'spearman' | 'kendall'."""
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method=method).round(3)


def top_correlations(df: pd.DataFrame, method: str = "pearson", n: int = 10) -> pd.DataFrame:
    """Strongest absolute pairwise correlations, de-duplicated."""
    corr = correlation_matrix(df, method)
    if corr.empty:
        return pd.DataFrame()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = ["feature_a", "feature_b", "correlation"]
    pairs["abs"] = pairs["correlation"].abs()
    return pairs.sort_values("abs", ascending=False).drop(columns="abs").head(n).reset_index(drop=True)


def group_aggregate(df: pd.DataFrame, group_by: list[str], value: str, agg: str = "mean") -> pd.DataFrame:
    """agg: any pandas agg name ('mean','sum','count','median','min','max','std')."""
    result = df.groupby(group_by, dropna=False)[value].agg(agg).reset_index()
    return result.sort_values(value, ascending=False).reset_index(drop=True)


def t_test(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict:
    """Independent two-sample Welch t-test between the two largest groups."""
    groups = df[group_col].dropna().value_counts().head(2).index.tolist()
    if len(groups) < 2:
        raise ValueError("Need at least two groups for a t-test.")
    a = df.loc[df[group_col] == groups[0], numeric_col].dropna()
    b = df.loc[df[group_col] == groups[1], numeric_col].dropna()
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "test": "Welch two-sample t-test",
        "groups": [str(groups[0]), str(groups[1])],
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "t_statistic": round(float(stat), 4),
        "p_value": float(p),
        "significant_0.05": bool(p < 0.05),
    }


def anova(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict:
    """One-way ANOVA across all groups of `group_col`."""
    samples = [g[numeric_col].dropna().values for _, g in df.groupby(group_col, dropna=True)]
    samples = [s for s in samples if len(s) > 1]
    if len(samples) < 2:
        raise ValueError("Need at least two groups with >1 observation.")
    stat, p = stats.f_oneway(*samples)
    return {
        "test": "One-way ANOVA",
        "n_groups": len(samples),
        "f_statistic": round(float(stat), 4),
        "p_value": float(p),
        "significant_0.05": bool(p < 0.05),
    }


def chi_square(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Chi-square test of independence between two categorical columns."""
    table = pd.crosstab(df[col_a], df[col_b])
    stat, p, dof, _ = stats.chi2_contingency(table)
    return {
        "test": "Chi-square test of independence",
        "chi2": round(float(stat), 4),
        "dof": int(dof),
        "p_value": float(p),
        "significant_0.05": bool(p < 0.05),
    }
