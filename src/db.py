"""SQL layer — register the working DataFrame in an in-memory SQLite database
and run arbitrary SELECT queries against it via SQLAlchemy + pandas.

Lets analysts query their data in real SQL (joins, GROUP BY, window functions)
without leaving the app — and demonstrates the SQL/SQLAlchemy skills every
data-analyst posting asks for.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace", "attach")


def run_query(df: pd.DataFrame, query: str, table_name: str = "data") -> pd.DataFrame:
    """Execute a read-only SQL query treating the DataFrame as table `data`.

    Raises ValueError on anything that isn't a single SELECT/WITH statement.
    """
    stripped = query.strip().rstrip(";").lower()
    if not (stripped.startswith("select") or stripped.startswith("with")):
        raise ValueError("Only SELECT / WITH queries are allowed.")
    if any(f" {kw} " in f" {stripped} " for kw in _FORBIDDEN):
        raise ValueError("Write/DDL statements are not permitted in this sandbox.")

    engine = create_engine("sqlite://")  # in-memory; lives only for this call
    try:
        df.to_sql(table_name, engine, index=False, if_exists="replace")
        with engine.connect() as conn:
            return pd.read_sql(text(query.rstrip(";")), conn)
    finally:
        engine.dispose()


def schema_preview(df: pd.DataFrame, table_name: str = "data") -> pd.DataFrame:
    """Column → SQLite type mapping, so the user knows what they can query."""
    type_map = {"int64": "INTEGER", "Int64": "INTEGER", "float64": "REAL",
                "bool": "INTEGER", "datetime64[ns]": "TIMESTAMP"}
    return pd.DataFrame({
        "column": df.columns,
        "sql_type": [type_map.get(str(t), "TEXT") for t in df.dtypes],
    })
