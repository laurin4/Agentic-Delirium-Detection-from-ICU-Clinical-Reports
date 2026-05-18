"""
Patient-level cohort counts: Berichte.csv vs structured_baseline.csv.

Single source of truth for data-coverage metrics and plots (no hardcoded sizes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from src.pipeline.paths import BERICHTE_INPUT_PATH, STRUCTURED_BASELINE_PATH
from src.pipeline.schema_normalize import clean_patient_id_value, normalize_patient_id_column
from src.preprocessing.berichte_filters import exclude_dokumentationsblatt

CURRENT_COHORT_METRIC_NAMES: Tuple[str, ...] = (
    "berichte_rows",
    "berichte_unique_patientids",
    "structured_baseline_rows",
    "structured_baseline_unique_patientids",
    "overlap_patientids",
    "berichte_without_baseline",
    "baseline_without_berichte",
)

PLOT_TITLE_SUFFIX = (
    "\n(Current final data: Berichte.csv vs structured_baseline.csv, patient-level unique IDs)"
)


def normalize_patient_id_series(series: pd.Series) -> pd.Series:
    return series.map(clean_patient_id_value)


def _filter_valid_ids(series: pd.Series) -> pd.Series:
    s = normalize_patient_id_series(series)
    return s[(s.str.len() > 0) & (s.str.lower() != "nan")]


def load_berichte_rows(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Berichte.csv; require PatientID; return rows with normalized PatientID."""
    resolved = path if path is not None else BERICHTE_INPUT_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Berichte.csv missing: {resolved}")

    last_err: Optional[BaseException] = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(resolved, sep=";", dtype=str, encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
        except Exception as exc:
            last_err = exc
            continue
    else:
        raise ValueError(f"Berichte.csv could not be read: {resolved}") from last_err

    df.columns = [str(c).strip() for c in df.columns]
    if "PatientID" not in df.columns:
        raise ValueError(f"Berichte.csv must contain 'PatientID'. Found: {list(df.columns)}")
    out = df.copy()
    out["PatientID"] = _filter_valid_ids(out["PatientID"])
    out, excluded = exclude_dokumentationsblatt(out)
    if excluded:
        import logging

        logging.getLogger(__name__).info("excluded_dokumentationsblatt_count=%d", excluded)
    return out


def load_structured_baseline_rows(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load outputs/baseline/structured_baseline.csv (one row per patient expected).

    Deduplicates on PatientenID (keeps first) so stale multi-row baselines cannot
    inflate unique-ID counts.
    """
    resolved = path if path is not None else STRUCTURED_BASELINE_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"structured_baseline.csv missing: {resolved}. "
            "Run: python -m src.pipeline.prepare_structured_data"
        )

    out = normalize_patient_id_column(pd.read_csv(resolved))
    if "PatientenID" not in out.columns:
        raise ValueError(f"structured_baseline.csv must contain 'PatientenID'. Found: {list(out.columns)}")
    out["PatientenID"] = _filter_valid_ids(out["PatientenID"])
    if out["PatientenID"].duplicated().any():
        out = out.drop_duplicates(subset=["PatientenID"], keep="first")
    return out


def berichte_patient_id_set(berichte: pd.DataFrame) -> Set[str]:
    return set(berichte["PatientID"].unique())


def baseline_patient_id_set(baseline: pd.DataFrame) -> Set[str]:
    return set(baseline["PatientenID"].unique())


def compute_current_cohort_counts(
    berichte: pd.DataFrame,
    baseline: pd.DataFrame,
) -> Dict[str, int]:
    b_ids = berichte_patient_id_set(berichte)
    m_ids = baseline_patient_id_set(baseline)
    inter = b_ids & m_ids
    return {
        "berichte_rows": int(len(berichte)),
        "berichte_unique_patientids": int(len(b_ids)),
        "structured_baseline_rows": int(len(baseline)),
        "structured_baseline_unique_patientids": int(len(m_ids)),
        "overlap_patientids": int(len(inter)),
        "berichte_without_baseline": int(len(b_ids - m_ids)),
        "baseline_without_berichte": int(len(m_ids - b_ids)),
    }


def current_cohort_counts_dataframe(counts: Dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": name, "value": int(counts[name])} for name in CURRENT_COHORT_METRIC_NAMES]
    )


def load_and_compute_current_cohort_counts(
    berichte_path: Optional[Path] = None,
    baseline_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Path, Path]:
    b_path = berichte_path if berichte_path is not None else BERICHTE_INPUT_PATH
    m_path = baseline_path if baseline_path is not None else STRUCTURED_BASELINE_PATH
    berichte = load_berichte_rows(b_path)
    baseline = load_structured_baseline_rows(m_path)
    counts = compute_current_cohort_counts(berichte, baseline)
    return berichte, baseline, counts, b_path, m_path


def print_current_cohort_counts(
    counts: Dict[str, int],
    *,
    berichte_path: Path,
    baseline_path: Path,
) -> None:
    print("")
    print("=== Current cohort counts (Berichte vs structured_baseline) ===")
    print(f"  Berichte path:   {berichte_path.resolve()}")
    print(f"  Baseline path:   {baseline_path.resolve()}")
    for name in CURRENT_COHORT_METRIC_NAMES:
        print(f"  {name}: {counts[name]}")
    print("=" * 62)
    print("")
