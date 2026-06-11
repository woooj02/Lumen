"""Unit tests for the analytical core (no Streamlit needed).

Run: pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import (analysis, anomaly, cleaning, db, insights, modeling,
                 profiling, report)
from src.sample_data import make_customers


@pytest.fixture(scope="module")
def df():
    return make_customers(n=400, seed=1)


def test_sample_shape(df):
    assert df.shape[0] >= 400          # includes injected duplicate rows
    assert "churn" in df.columns
    assert df.isna().sum().sum() > 0   # missing values were seeded


def test_overview_and_profile(df):
    ov = profiling.overview(df)
    assert ov["rows"] == len(df)
    assert ov["duplicate_rows"] > 0
    prof = profiling.column_profile(df)
    assert set(["column", "dtype", "missing", "unique"]).issubset(prof.columns)
    assert len(profiling.quality_flags(df)) > 0


def test_cleaning_pipeline(df):
    out, msg, code = cleaning.drop_duplicates(df)
    assert out.duplicated().sum() == 0
    assert "drop_duplicates" in code

    out2, _, _ = cleaning.impute_missing(df, ["total_charges"], "median")
    assert out2["total_charges"].isna().sum() == 0

    out3, _, _ = cleaning.handle_outliers(df, ["monthly_charges"], "iqr", "clip")
    assert out3["monthly_charges"].max() <= df["monthly_charges"].max()

    out4, _, _ = cleaning.encode_categorical(df, ["gender"], "onehot")
    assert any(c.startswith("gender_") for c in out4.columns)


def test_analysis(df):
    corr = analysis.correlation_matrix(df)
    assert not corr.empty
    top = analysis.top_correlations(df)
    assert "correlation" in top.columns
    res = analysis.t_test(df, "monthly_charges", "churn")
    assert "p_value" in res and 0 <= res["p_value"] <= 1
    chi = analysis.chi_square(df, "contract_type", "churn")
    assert chi["p_value"] <= 1


def test_sql(df):
    out = db.run_query(df, "SELECT contract_type, COUNT(*) AS n FROM data GROUP BY contract_type")
    assert "n" in out.columns and out["n"].sum() == len(df)
    with pytest.raises(ValueError):
        db.run_query(df, "DROP TABLE data")


def test_modeling_classification(df):
    results, leaderboard = modeling.train_and_compare(df, "churn", "classification", cv=3)
    assert len(results) == 3
    assert "accuracy" in leaderboard.columns
    assert results[0].metrics["accuracy"] > 0.5   # beats nothing-learned baseline
    cm, labels = modeling.confusion(results[0])
    assert cm.shape == (len(labels), len(labels))


def test_modeling_regression(df):
    results, leaderboard = modeling.train_and_compare(df, "monthly_charges", "regression", cv=3)
    assert "r2" in leaderboard.columns
    assert results[0].importance is not None


def test_anomaly_detection(df):
    scored, flagged, summary = anomaly.detect_anomalies(df, sensitivity=0.05)
    assert "anomaly_score" in scored.columns and "flagged" in scored.columns
    assert len(scored) == len(df)
    assert summary["flagged"] == int(scored["flagged"].sum())
    assert 0 < summary["flagged_pct"] < 20      # ~5% flagged, with slack
    assert len(flagged) == summary["flagged"]


def test_insights_engine(df):
    cards = insights.kpi_cards(df)
    assert any(c["label"] == "Records" for c in cards)
    findings = insights.key_findings(df)
    assert findings and all(isinstance(s, str) for s in findings)
    charts = insights.auto_charts(df)
    assert charts and all(hasattr(fig, "to_html") for _, fig in charts)


def test_report_builds(df):
    html = report.build_report(
        title="Test",
        cards=insights.kpi_cards(df),
        profile_df=profiling.column_profile(df),
        findings=insights.key_findings(df),
        anomaly={"records": 400, "flagged": 20, "flagged_pct": 5.0, "drivers": ["x higher"]},
        quality_flags=profiling.quality_flags(df),
        cleaning_steps=["dropped dupes"],
    )
    assert "<html" in html and "Test" in html and "Records to review" in html
