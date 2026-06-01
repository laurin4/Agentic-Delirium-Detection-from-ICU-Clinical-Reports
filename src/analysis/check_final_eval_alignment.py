"""
Strict empirical alignment check: frozen cohort vs validation_cohort_predictions.

Read-only on inputs; writes only final_eval_alignment_check.txt.
Does not use Berichte.csv or source_report_row_id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.paths import (
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    MANUAL_VALIDATION_DIR,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.pipeline.prompt_run_paths import (
    get_versioned_final_eval_alignment_path,
    resolve_validation_predictions_path,
)
from src.pipeline.validation_report_identity import VALIDATION_REPORT_ID_COL, _norm_id
from src.preprocessing.berichte_filters import normalize_bertyp

LOGGER = logging.getLogger(__name__)

FINAL_EVAL_ALIGNMENT_CHECK_PATH = (
    MANUAL_VALIDATION_DIR / "final_eval_alignment_check.txt"
)

COMPARE_FIELDS: tuple[str, ...] = ("PatientenID", "bertyp", "berdat")
MAX_EXAMPLE_ROWS = 20


@dataclass
class AlignmentCheckResult:
    cohort_rows: int = 0
    prediction_rows: int = 0
    cohort_unique_ids: int = 0
    prediction_unique_ids: int = 0
    cohort_duplicate_ids: int = 0
    prediction_duplicate_ids: int = 0
    missing_in_predictions: List[str] = field(default_factory=list)
    extra_in_predictions: List[str] = field(default_factory=list)
    patient_id_mismatch: int = 0
    bertyp_mismatch: int = 0
    berdat_mismatch: int = 0
    berdat_format_only_mismatch: int = 0
    mismatch_examples: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    verdict: str = "FAIL"

    def to_report_lines(self) -> List[str]:
        lines = [
            "Final evaluation alignment check (validation_report_id)",
            "=" * 56,
            "",
            "Inputs",
            "-" * 56,
            f"cohort_rows={self.cohort_rows}",
            f"prediction_rows={self.prediction_rows}",
            "",
            "1. Row count equality",
            "-" * 56,
            f"equal={self.cohort_rows == self.prediction_rows}",
            "",
            "2. validation_report_id uniqueness",
            "-" * 56,
            f"cohort_unique_ids={self.cohort_unique_ids}",
            f"prediction_unique_ids={self.prediction_unique_ids}",
            f"cohort_duplicate_id_rows={self.cohort_duplicate_ids}",
            f"prediction_duplicate_id_rows={self.prediction_duplicate_ids}",
            "",
            "3. validation_report_id set equality",
            "-" * 56,
            f"missing_in_predictions={len(self.missing_in_predictions)}",
            f"extra_in_predictions={len(self.extra_in_predictions)}",
        ]
        if self.missing_in_predictions:
            lines.append(f"  missing_sample={self.missing_in_predictions[:MAX_EXAMPLE_ROWS]}")
        if self.extra_in_predictions:
            lines.append(f"  extra_sample={self.extra_in_predictions[:MAX_EXAMPLE_ROWS]}")
        lines.extend(
            [
                "",
                "4–6. Per-ID field mismatches (cohort vs predictions)",
                "-" * 56,
                f"patient_id_mismatch={self.patient_id_mismatch}",
                f"bertyp_mismatch={self.bertyp_mismatch}",
                f"berdat_mismatch={self.berdat_mismatch}",
                f"berdat_format_only_mismatch={self.berdat_format_only_mismatch}",
                "",
            ]
        )
        if self.errors:
            lines.append("Errors")
            lines.append("-" * 56)
            for err in self.errors:
                lines.append(f"  - {err}")
            lines.append("")

        if self.mismatch_examples:
            lines.append("Mismatch examples")
            lines.append("-" * 56)
            for ex in self.mismatch_examples[:MAX_EXAMPLE_ROWS]:
                lines.append(f"  validation_report_id={ex.get('validation_report_id', '')}")
                for key in ("issue", "cohort_value", "prediction_value"):
                    if key in ex:
                        lines.append(f"    {key}={ex[key]}")
            if len(self.mismatch_examples) > MAX_EXAMPLE_ROWS:
                lines.append(
                    f"  ... and {len(self.mismatch_examples) - MAX_EXAMPLE_ROWS} more"
                )
            lines.append("")

        lines.extend(
            [
                "7. Verdict",
                "-" * 56,
                f"VERDICT: {self.verdict}",
                "",
                "Interpretation",
                "-" * 56,
                "PASS: safe to trust final_manual_validation_evaluation merge by validation_report_id.",
                "WARNING: ID sets align; only berdat string formatting differs (same calendar date).",
                "FAIL: row/ID mismatch or PatientenID/bertyp/berdat identity disagreement.",
            ]
        )
        return lines


def _norm_patient_id(value: object) -> str:
    return _norm_id(value)


def _norm_bertyp(value: object) -> str:
    return normalize_bertyp(value) if _norm_id(value) else ""


def _norm_berdat_string(value: object) -> str:
    return _norm_id(value)


def _parse_berdat(value: object) -> Optional[pd.Timestamp]:
    s = _norm_berdat_string(value)
    if not s:
        return None
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return None
    return ts.normalize()


def _berdat_mismatch_kind(cohort_val: object, pred_val: object) -> Optional[str]:
    """
    Return None if berdat matches; ``format_only`` if same date different strings;
    ``mismatch`` if truly different.
    """
    cs = _norm_berdat_string(cohort_val)
    ps = _norm_berdat_string(pred_val)
    if cs == ps:
        return None
    if not cs and not ps:
        return None
    if not cs or not ps:
        return "mismatch"
    c_ts = _parse_berdat(cs)
    p_ts = _parse_berdat(ps)
    if c_ts is not None and p_ts is not None and c_ts == p_ts:
        return "format_only"
    return "mismatch"


def run_final_eval_alignment_check(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
) -> AlignmentCheckResult:
    result = AlignmentCheckResult()

    if not cohort_path.exists():
        result.errors.append(f"cohort missing: {cohort_path}")
        return result
    if not predictions_path.exists():
        result.errors.append(f"predictions missing: {predictions_path}")
        return result

    cohort = pd.read_csv(cohort_path)
    preds = pd.read_csv(predictions_path)

    if VALIDATION_REPORT_ID_COL not in cohort.columns:
        result.errors.append(f"cohort missing column {VALIDATION_REPORT_ID_COL}")
        return result
    if VALIDATION_REPORT_ID_COL not in preds.columns:
        result.errors.append(f"predictions missing column {VALIDATION_REPORT_ID_COL}")
        return result

    for field in COMPARE_FIELDS:
        if field not in cohort.columns:
            result.errors.append(f"cohort missing column {field}")
        if field not in preds.columns:
            result.errors.append(f"predictions missing column {field}")
    if result.errors:
        return result

    result.cohort_rows = len(cohort)
    result.prediction_rows = len(preds)

    c_ids = cohort[VALIDATION_REPORT_ID_COL].astype(str).map(_norm_id)
    p_ids = preds[VALIDATION_REPORT_ID_COL].astype(str).map(_norm_id)

    result.cohort_duplicate_ids = int(c_ids.duplicated(keep=False).sum())
    result.prediction_duplicate_ids = int(p_ids.duplicated(keep=False).sum())
    result.cohort_unique_ids = int(c_ids[c_ids != ""].nunique())
    result.prediction_unique_ids = int(p_ids[p_ids != ""].nunique())

    c_set = set(c_ids[c_ids != ""])
    p_set = set(p_ids[p_ids != ""])

    result.missing_in_predictions = sorted(c_set - p_set)
    result.extra_in_predictions = sorted(p_set - c_set)

    cohort_by_id = (
        cohort.assign(_vid=c_ids)
        .loc[c_ids != ""]
        .drop_duplicates("_vid", keep="first")
        .set_index("_vid")
    )
    pred_by_id = (
        preds.assign(_vid=p_ids)
        .loc[p_ids != ""]
        .drop_duplicates("_vid", keep="first")
        .set_index("_vid")
    )

    common_ids = sorted(c_set & p_set)
    for vid in common_ids:
        crow = cohort_by_id.loc[vid]
        prow = pred_by_id.loc[vid]

        c_pid = _norm_patient_id(crow.get("PatientenID"))
        p_pid = _norm_patient_id(prow.get("PatientenID"))
        if c_pid != p_pid:
            result.patient_id_mismatch += 1
            result.mismatch_examples.append(
                {
                    "validation_report_id": vid,
                    "issue": "patient_id_mismatch",
                    "cohort_value": c_pid,
                    "prediction_value": p_pid,
                }
            )

        c_bt = _norm_bertyp(crow.get("bertyp"))
        p_bt = _norm_bertyp(prow.get("bertyp"))
        if c_bt != p_bt:
            result.bertyp_mismatch += 1
            result.mismatch_examples.append(
                {
                    "validation_report_id": vid,
                    "issue": "bertyp_mismatch",
                    "cohort_value": c_bt,
                    "prediction_value": p_bt,
                }
            )

        berdat_kind = _berdat_mismatch_kind(crow.get("berdat"), prow.get("berdat"))
        if berdat_kind == "mismatch":
            result.berdat_mismatch += 1
            result.mismatch_examples.append(
                {
                    "validation_report_id": vid,
                    "issue": "berdat_mismatch",
                    "cohort_value": _norm_berdat_string(crow.get("berdat")),
                    "prediction_value": _norm_berdat_string(prow.get("berdat")),
                }
            )
        elif berdat_kind == "format_only":
            result.berdat_format_only_mismatch += 1
            result.mismatch_examples.append(
                {
                    "validation_report_id": vid,
                    "issue": "berdat_format_only_mismatch",
                    "cohort_value": _norm_berdat_string(crow.get("berdat")),
                    "prediction_value": _norm_berdat_string(prow.get("berdat")),
                }
            )

    result.verdict = compute_alignment_verdict(result)
    return result


def compute_alignment_verdict(result: AlignmentCheckResult) -> str:
    if result.errors:
        return "FAIL"
    if result.cohort_rows != result.prediction_rows:
        return "FAIL"
    if result.cohort_duplicate_ids or result.prediction_duplicate_ids:
        return "FAIL"
    if result.missing_in_predictions or result.extra_in_predictions:
        return "FAIL"
    if result.patient_id_mismatch or result.bertyp_mismatch:
        return "FAIL"
    if result.berdat_mismatch:
        return "FAIL"
    if result.berdat_format_only_mismatch:
        return "WARNING"
    return "PASS"


def format_alignment_report(result: AlignmentCheckResult) -> str:
    return "\n".join(result.to_report_lines()) + "\n"


def write_final_eval_alignment_check(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    predictions_path: Path | None = None,
    output_path: Path | None = None,
) -> AlignmentCheckResult:
    resolved_predictions = (
        predictions_path
        if predictions_path is not None
        else resolve_validation_predictions_path()
    )
    resolved_output = (
        output_path
        if output_path is not None
        else get_versioned_final_eval_alignment_path()
    )
    result = run_final_eval_alignment_check(cohort_path, resolved_predictions)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(format_alignment_report(result), encoding="utf-8")
    LOGGER.info(
        "Wrote final eval alignment check: %s (verdict=%s)",
        resolved_output,
        result.verdict,
    )
    return result


def main() -> None:
    result = write_final_eval_alignment_check()
    print(format_alignment_report(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
