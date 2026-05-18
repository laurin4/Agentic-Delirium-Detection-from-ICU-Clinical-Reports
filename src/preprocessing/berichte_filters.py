"""
Berichte.csv processing filters (raw rows are never deleted on disk).
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

DOKUMENTATIONSBLATT_BERTYP = "Dokumentationsblatt"

REPORT_TYPES_FOR_MATRIX = (
    "Verlaufseintrag",
    "Verlegungsbericht",
    "Austrittsbericht",
)


def normalize_bertyp(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s


def is_dokumentationsblatt(bertyp: object) -> bool:
    """True when bertyp equals Dokumentationsblatt (exact match after strip)."""
    return normalize_bertyp(bertyp) == DOKUMENTATIONSBLATT_BERTYP


def exclude_dokumentationsblatt(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Return a copy without Dokumentationsblatt rows and the excluded row count.

    If ``bertyp`` is missing, no rows are excluded (warning logged).
    """
    if df.empty:
        return df.copy(), 0
    out = df.copy()
    if "bertyp" not in out.columns:
        LOGGER.warning("Berichte dataframe has no 'bertyp' column; Dokumentationsblatt exclusion skipped.")
        return out, 0
    mask = out["bertyp"].map(is_dokumentationsblatt)
    excluded = int(mask.sum())
    if excluded:
        LOGGER.info("excluded_dokumentationsblatt_count=%d", excluded)
    return out.loc[~mask].copy(), excluded
