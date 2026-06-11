"""Shared Streamlit session-state helpers so every page reads/writes the same
working DataFrame, original snapshot, and cleaning recipe.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def set_data(df: pd.DataFrame, name: str = "dataset") -> None:
    st.session_state["df"] = df
    st.session_state["df_original"] = df.copy()
    st.session_state["dataset_name"] = name
    st.session_state["recipe"] = []      # list of {"message","code"}
    st.session_state["model_results"] = None


def get_df() -> pd.DataFrame | None:
    return st.session_state.get("df")


def update_df(df: pd.DataFrame) -> None:
    st.session_state["df"] = df


def reset_df() -> None:
    if "df_original" in st.session_state:
        st.session_state["df"] = st.session_state["df_original"].copy()
        st.session_state["recipe"] = []


def log_step(message: str, code: str) -> None:
    st.session_state.setdefault("recipe", []).append({"message": message, "code": code})


def get_recipe() -> list[dict]:
    return st.session_state.get("recipe", [])


def require_data() -> pd.DataFrame:
    """Return the working df or stop the page with a friendly prompt."""
    df = get_df()
    if df is None:
        st.warning("No dataset loaded yet. Go to the **Home** page to upload a file or load the sample data.")
        st.stop()
    return df


def recipe_to_script() -> str:
    """Render the cleaning recipe as a runnable pandas script."""
    steps = get_recipe()
    header = [
        "# Reproducible cleaning pipeline exported by InsightForge",
        "import pandas as pd",
        "",
        f"df = pd.read_csv('{st.session_state.get('dataset_name', 'data')}.csv')",
        "",
    ]
    body = []
    for step in steps:
        body.append(f"# {step['message']}")
        body.append(step["code"])
        body.append("")
    return "\n".join(header + body)
