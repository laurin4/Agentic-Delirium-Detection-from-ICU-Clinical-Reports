"""
Build patient-level report text from anonymized hospital reports (Berichte.csv).

Berichte.csv is external (not committed). Paths come from paths.BERICHTE_INPUT_PATH.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.paths import BERICHTE_INPUT_PATH
from src.preprocessing.berichte_filters import exclude_dokumentationsblatt, normalize_bertyp
from src.preprocessing.report_identity import (
    SOURCE_REPORT_ROW_ID_COL,
    assign_source_report_row_ids,
    compute_pipeline_bericht_id,
)

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


def read_berichte_csv_robust(
    path: Path,
    *,
    log_context: str = "Berichte load",
) -> pd.DataFrame:
    """
    Load semicolon-separated Berichte.csv, skipping malformed rows instead of failing.

    Uses ``engine="python"`` and ``on_bad_lines`` so free-text delimiter issues do not
    abort downstream exports. Logs a warning per skipped row (and a summary count).
    """
    skipped_count = 0

    def _on_bad_line(_bad_line: list[str]) -> None:
        nonlocal skipped_count
        skipped_count += 1
        LOGGER.warning(
            "[WARN] skipped malformed Berichte.csv row during %s",
            log_context,
        )
        return None

    last_err: Optional[BaseException] = None
    for enc in _CSV_ENCODINGS:
        try:
            try:
                df = pd.read_csv(
                    path,
                    sep=";",
                    dtype=str,
                    encoding=enc,
                    engine="python",
                    on_bad_lines=_on_bad_line,
                )
            except TypeError:
                # Older pandas: no callable on_bad_lines
                df = pd.read_csv(
                    path,
                    sep=";",
                    dtype=str,
                    encoding=enc,
                    engine="python",
                    on_bad_lines="warn",
                )
            if skipped_count:
                LOGGER.warning(
                    "[WARN] skipped %d malformed Berichte.csv row(s) during %s",
                    skipped_count,
                    log_context,
                )
            return df
        except UnicodeDecodeError as exc:
            last_err = exc
            skipped_count = 0
        except pd.errors.ParserError as exc:
            last_err = exc
            LOGGER.warning(
                "ParserError reading Berichte.csv with encoding %s (%s): %s",
                enc,
                log_context,
                exc,
            )
            skipped_count = 0
        except Exception as exc:
            last_err = exc
            LOGGER.warning("Failed reading Berichte.csv with encoding %s: %s", enc, exc)
            skipped_count = 0
            continue
    raise ValueError(f"Berichte.csv could not be read: {path}") from last_err


def _read_berichte_csv(path: Path) -> pd.DataFrame:
    return read_berichte_csv_robust(path, log_context="Berichte load")


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
    df, _ = exclude_dokumentationsblatt(df)

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


def build_report_level_berichte_records(
    input_path: Optional[Path] = None,
    *,
    apply_dokumentationsblatt_exclusion: bool = True,
) -> Tuple[List[dict], int]:
    """
    One pipeline record per Berichte.csv row (report-level prediction).

    Excludes ``bertyp == Dokumentationsblatt`` when *apply_dokumentationsblatt_exclusion* is True.
    Returns (records, excluded_dokumentationsblatt_count).
    """
    csv_path = input_path if input_path is not None else BERICHTE_INPUT_PATH
    df = load_berichte_dataframe(csv_path)

    if "PatientID" not in df.columns:
        raise ValueError(f"Berichte.csv must contain column 'PatientID'. Found columns: {list(df.columns)}")

    df = assign_source_report_row_ids(df)

    excluded_count = 0
    if apply_dokumentationsblatt_exclusion:
        df, excluded_count = exclude_dokumentationsblatt(df)

    for name in OPTIONAL_COLUMNS:
        if name not in df.columns:
            df[name] = ""

    df["PatientID"] = df["PatientID"].astype(str).str.strip()
    df["PatientenID"] = df["PatientID"]
    df["bertyp"] = df["bertyp"].map(normalize_bertyp) if "bertyp" in df.columns else ""

    records: List[dict] = []
    for idx, row in df.iterrows():
        pid = _normalize_str(row.get("PatientID", ""))
        if not pid or pid.lower() == "nan":
            continue
        row_dict = {c: row.get(c, "") for c in df.columns}
        blk = _row_blocks(row_dict)
        if not blk:
            continue
        bericht_id = compute_pipeline_bericht_id(row)
        berdat = _normalize_str(row.get("berdat", ""))

        records.append(
            {
                "PatientenID": pid,
                "bericht": bericht_id,
                "bertyp": normalize_bertyp(row.get("bertyp", "")),
                "berdat": berdat,
                SOURCE_REPORT_ROW_ID_COL: str(row.get(SOURCE_REPORT_ROW_ID_COL, "")),
                "report_text": blk,
            }
        )

    return records, excluded_count
