"""Plotly chart builders used by the Visualize page and the HTML report.

Each returns a `plotly.graph_objects.Figure` so it can be shown in Streamlit
(`st.plotly_chart`) or serialised into the report (`fig.to_html`).
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_TEMPLATE = "plotly_white"


def histogram(df, x, color=None, nbins=30):
    fig = px.histogram(df, x=x, color=color, nbins=nbins, marginal="box",
                       template=_TEMPLATE, title=f"Distribution of {x}")
    return fig


def bar(df, x, y=None, color=None, agg="sum"):
    if y is None:
        data = df[x].value_counts().reset_index()
        data.columns = [x, "count"]
        fig = px.bar(data, x=x, y="count", template=_TEMPLATE, title=f"Count by {x}")
    else:
        data = df.groupby(x, dropna=False)[y].agg(agg).reset_index()
        fig = px.bar(data, x=x, y=y, color=color, template=_TEMPLATE,
                     title=f"{agg.title()} of {y} by {x}")
    return fig


def line(df, x, y, color=None):
    fig = px.line(df.sort_values(x), x=x, y=y, color=color,
                  template=_TEMPLATE, title=f"{y} over {x}")
    return fig


def scatter(df, x, y, color=None, size=None, trendline=True):
    fig = px.scatter(df, x=x, y=y, color=color, size=size,
                     trendline="ols" if trendline else None,
                     template=_TEMPLATE, title=f"{y} vs {x}")
    return fig


def box(df, y, x=None, color=None):
    fig = px.box(df, x=x, y=y, color=color, template=_TEMPLATE,
                 title=f"Spread of {y}" + (f" by {x}" if x else ""))
    return fig


def pie(df, names, values=None):
    if values is None:
        data = df[names].value_counts().reset_index()
        data.columns = [names, "count"]
        fig = px.pie(data, names=names, values="count", template=_TEMPLATE,
                     title=f"Share of {names}", hole=0.4)
    else:
        fig = px.pie(df, names=names, values=values, template=_TEMPLATE,
                     title=f"{values} by {names}", hole=0.4)
    return fig


def correlation_heatmap(corr: pd.DataFrame):
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, template=_TEMPLATE, title="Correlation heatmap")
    return fig


def confusion_matrix_fig(cm, labels):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=[str(l) for l in labels], y=[str(l) for l in labels],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    template=_TEMPLATE, title="Confusion matrix")
    return fig


def residuals_fig(y_true, y_pred):
    resid = pd.Series(y_true).reset_index(drop=True) - pd.Series(y_pred).reset_index(drop=True)
    fig = px.scatter(x=y_pred, y=resid, template=_TEMPLATE,
                     labels={"x": "Predicted", "y": "Residual"}, title="Residuals vs predicted")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


def feature_importance_fig(importance_df: pd.DataFrame):
    data = importance_df.sort_values("importance").tail(20)
    fig = px.bar(data, x="importance", y="feature", orientation="h",
                 template=_TEMPLATE, title="Feature importance")
    return fig
