"""Auto-insight engine — turns a freshly loaded DataFrame into the things a
non-technical user actually wants to see: headline numbers, plain-English
findings, and a smart set of charts chosen automatically from the data.

No configuration required — the dashboard calls these and renders the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import analysis, viz


def _interesting_categoricals(df: pd.DataFrame, lo: int = 2, hi: int = 20) -> list[str]:
    """Categorical columns with a useful number of distinct values (not IDs)."""
    out = []
    for c in df.select_dtypes(exclude="number").columns:
        nun = df[c].nunique(dropna=True)
        if lo <= nun <= hi and nun < len(df):
            out.append(c)
    return out


def _continuous_numerics(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are genuinely continuous (not 0/1 flags or IDs)."""
    out = []
    for c in df.select_dtypes(include="number").columns:
        nun = df[c].nunique(dropna=True)
        if nun > 10 and nun < len(df):
            out.append(c)
    return out


def kpi_cards(df: pd.DataFrame) -> list[dict]:
    """Headline metrics for the top of the dashboard."""
    completeness = 100 - df.isna().mean().mean() * 100
    cards = [
        {"label": "Records", "value": f"{len(df):,}"},
        {"label": "Fields", "value": f"{df.shape[1]}"},
        {"label": "Data completeness", "value": f"{completeness:.0f}%"},
        {"label": "Duplicate rows", "value": f"{int(df.duplicated().sum()):,}"},
    ]
    # If there's an obvious money/amount column, headline its total.
    money = [c for c in df.select_dtypes(include="number").columns
             if any(k in c.lower() for k in ("amount", "charge", "price", "cost", "revenue", "total", "paid", "claim"))]
    if money:
        c = money[0]
        cards.append({"label": f"Total {c.replace('_', ' ')}", "value": f"{df[c].sum():,.0f}"})
    return cards


def key_findings(df: pd.DataFrame) -> list[str]:
    """Plain-English bullet points summarising the dataset."""
    f = []
    n, m = df.shape
    f.append(f"This dataset has **{n:,} records** across **{m} fields**.")

    overall_missing = df.isna().mean().mean() * 100
    if overall_missing < 1:
        f.append("The data is **almost complete** — barely any missing values.")
    else:
        worst = df.isna().mean().idxmax()
        wpct = df.isna().mean().max() * 100
        f.append(f"About **{overall_missing:.1f}%** of values are missing overall; "
                 f"*{worst}* is the most incomplete ({wpct:.0f}% blank).")

    dups = int(df.duplicated().sum())
    if dups:
        f.append(f"There are **{dups:,} duplicate rows** that may need removing.")

    cats = _interesting_categoricals(df)
    if cats:
        c = cats[0]
        vc = df[c].value_counts(normalize=True)
        f.append(f"The most common **{c.replace('_', ' ')}** is "
                 f"*{vc.index[0]}* ({vc.iloc[0] * 100:.0f}% of records).")

    nums = _continuous_numerics(df)
    if nums:
        c = nums[0]
        s = df[c]
        f.append(f"**{c.replace('_', ' ').title()}** ranges from {s.min():,.0f} to "
                 f"{s.max():,.0f}, averaging {s.mean():,.1f}.")

    top = analysis.top_correlations(df, n=1)
    if not top.empty and abs(top.iloc[0]["correlation"]) >= 0.4:
        r = top.iloc[0]
        rel = "rise together" if r["correlation"] > 0 else "move in opposite directions"
        f.append(f"**{r['feature_a']}** and **{r['feature_b']}** {rel} "
                 f"(correlation {r['correlation']:.2f}).")

    dt = df.select_dtypes(include="datetime").columns.tolist()
    if dt:
        c = dt[0]
        f.append(f"Records span **{df[c].min():%Y-%m-%d}** to **{df[c].max():%Y-%m-%d}**.")

    return f


def auto_charts(df: pd.DataFrame, max_charts: int = 6) -> list[tuple[str, object]]:
    """Pick a sensible set of charts automatically and return [(title, fig)]."""
    charts: list[tuple[str, object]] = []
    cats = _interesting_categoricals(df)
    nums = _continuous_numerics(df)
    dates = df.select_dtypes(include="datetime").columns.tolist()

    # 1) Trend over time, if a date column exists.
    if dates and nums:
        d, y = dates[0], nums[0]
        tmp = df.dropna(subset=[d]).copy()
        tmp["_period"] = tmp[d].dt.to_period("M").dt.to_timestamp()
        agg = tmp.groupby("_period")[y].sum().reset_index()
        charts.append((f"{y.replace('_', ' ').title()} over time", viz.line(agg, "_period", y)))

    # 2) Counts for the top categorical fields.
    for c in cats[:2]:
        charts.append((f"Breakdown by {c.replace('_', ' ')}", viz.bar(df, c)))

    # 3) Distributions for the top numeric fields.
    for c in nums[:2]:
        charts.append((f"Distribution of {c.replace('_', ' ')}", viz.histogram(df, c)))

    # 4) Correlation heatmap if there's enough numeric breadth.
    corr = analysis.correlation_matrix(df)
    if corr.shape[0] >= 3:
        charts.append(("How the numbers relate", viz.correlation_heatmap(corr)))

    return charts[:max_charts]
