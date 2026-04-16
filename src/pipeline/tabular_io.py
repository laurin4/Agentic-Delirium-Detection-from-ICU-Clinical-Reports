"""Centralized tabular file loading (CSV / Excel) for pipeline inputs."""

from pathlib import Path

import pandas as pd


def read_tabular(path: Path) -> pd.DataFrame:
    """
    Load a single table file. Supports .csv, .xlsx, .xls.
    Does not assume a specific schema; callers normalize columns.
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    raise ValueError(f"Unsupported tabular format for path: {path} (suffix={suffix!r})")
