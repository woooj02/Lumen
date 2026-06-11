"""Data ingestion: read CSV / Excel / JSON / Parquet / TSV into a DataFrame.

Keeps loading logic in one place so every page (and the test-suite) shares the
same behaviour. Also exposes a lightweight memory optimiser used after upload.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet"}


def load_file(file, filename: str | None = None) -> pd.DataFrame:
    """Load an uploaded file-like object (or a path) into a DataFrame.

    `file` may be a Streamlit UploadedFile, an open binary stream, or a path str.
    `filename` overrides the name used for extension sniffing (UploadedFile has .name).
    """
    name = filename or getattr(file, "name", str(file))
    ext = Path(name).suffix.lower()

    if ext in (".csv", ".txt"):
        return pd.read_csv(file)
    if ext == ".tsv":
        return pd.read_csv(file, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file)
    if ext == ".json":
        return pd.read_json(file)
    if ext == ".parquet":
        return pd.read_parquet(file)

    # Unknown extension — make a best effort as delimited text.
    return pd.read_csv(file)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert low-cardinality strings to category.

    Shrinks memory footprint without changing values — handy on large uploads.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["object"]).columns:
        # Convert to category when fewer than 50% of values are unique.
        if out[col].nunique(dropna=True) / max(len(out), 1) < 0.5:
            out[col] = out[col].astype("category")
    return out


def coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: parse object columns that look like dates into datetime64."""
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        sample = out[col].dropna().astype(str).head(25)
        if sample.empty:
            continue
        looks_dateish = sample.str.contains(r"\d{4}|\d{1,2}[/-]\d{1,2}", regex=True).mean() > 0.7
        if looks_dateish:
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().mean() > 0.8:  # only keep if most rows parsed
                out[col] = parsed
    return out
