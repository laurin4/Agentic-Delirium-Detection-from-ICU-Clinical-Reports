"""
Build patient-level report text from anonymized hospital reports (Berichte.csv).

Berichte.csv is external (not committed). Paths come from paths.BERICHTE_INPUT_PATH.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.paths import BERICHTE_INPUT_PATH

LOGGER = logging.getLogger(__name__)

OPTIONAL_COLUMNS = ["berdat", "bertyp", "diag", "epikrise", "jetziges_leiden", "prozedere"]

_SECTION_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("diag", "[Diagnosen]"),
    ("epikrise", "[Epikrise]"),
    ("jetziges_leiden", "[Jetziges Leiden]"),
    ("prozedere", "[Prozedere]"),
)

_CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def _normalize_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none"):
        return ""
    return s


def _read_berichte_csv(path: Path) -> pd.DataFrame:
    last_err: Optional[BaseException] = None
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(path, sep=";", dtype=str, encoding=enc)
        except UnicodeDecodeError as exc:
            last_err = exc
        except Exception as exc:
            last_err = exc
            LOGGER.warning("Failed reading Berichte.csv with encoding %s: %s", enc, exc)
            continue
    raise ValueError(f"Berichte.csv could not be read: {path}") from last_err


def load_berichte_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Berichte.csv from the centralized path (or an explicit path).

    Raises FileNotFoundError if the file is missing (e.g. not deployed on a dev machine).
    """
    resolved = path if path is not None else BERICHTE_INPUT_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Berichte input missing: {resolved}. "
            "Expected external anonymized CSV (semicolon-separated) at this path."
        )
    df = _read_berichte_csv(resolved)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _row_blocks(row: Dict[str, str]) -> Optional[str]:
    parts: List[str] = []
    for col, heading in _SECTION_FIELDS:
        text = _normalize_str(row.get(col, ""))
        if text:
            parts.append(f"{heading}\n{text}")
    if not parts:
        return None
    return "\n\n".join(parts)


def build_patient_level_berichte_reports(input_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Berichte.csv (semicolon-separated) and return one row per PatientenID:

    Columns: PatientenID, bericht, report_text
    """
    csv_path = input_path if input_path is not None else BERICHTE_INPUT_PATH

    df = load_berichte_dataframe(csv_path)

    if "PatientID" not in df.columns:
        raise ValueError(f"Berichte.csv must contain column 'PatientID'. Found columns: {list(df.columns)}")

    for name in OPTIONAL_COLUMNS:
        if name not in df.columns:
            LOGGER.warning(
                "Berichte.csv missing optional column '%s'. Treating values as empty (path=%s).",
                name,
                csv_path,
            )
            df[name] = ""

    if "PatientID" in df.columns:
        df["PatientID"] = df["PatientID"].astype(str).str.strip()

    if "berdat" in df.columns:
        df["_sort_datum"] = pd.to_datetime(df["berdat"], errors="coerce")
    else:
        df["_sort_datum"] = pd.NaT

    rows_out: List[dict] = []

    grouped = df.groupby("PatientID", dropna=False, sort=False)
    for pid, sub in grouped:
        pid_clean = _normalize_str(pid)
        if not pid_clean or pid_clean.lower() == "nan":
            LOGGER.warning("Skipping row group with empty PatientID.")
            continue

        sub = sub.sort_values("_sort_datum", kind="stable", na_position="last")

        block_strings: List[str] = []
        for _, row in sub.iterrows():
            row_dict = {c: row.get(c, "") for c in sub.columns}
            blk = _row_blocks(row_dict)
            if blk:
                block_strings.append(blk)

        report_text = "\n\n".join(block_strings)

        bericht_name = f"berichte_{pid_clean}.txt"
        rows_out.append(
            {
                "PatientenID": pid_clean,
                "bericht": bericht_name,
                "report_text": report_text,
            }
        )

    return pd.DataFrame(rows_out, columns=["PatientenID", "bericht", "report_text"])


def build_patient_level_berichte_report_records(input_path: Optional[Path] = None) -> List[dict]:
    df = build_patient_level_berichte_reports(input_path)
    return df.to_dict(orient="records")
