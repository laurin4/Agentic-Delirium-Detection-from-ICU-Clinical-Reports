"""
LEGACY / DEPRECATED — use ``export_patient_validation_cohort`` as the PRIMARY workflow.

Export ~100-patient mixed manual validation sample from patient_reporttype_matrix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.pipeline.schema_normalize import (
    assert_patientenid_column,
    log_patientenid_dtype_if_debug,
    normalize_patient_id_column,
)
from src.pipeline.paths import (
    MANUAL_VALIDATION_DIR,
    MANUAL_VALIDATION_SAMPLE_PATH,
    PATIENT_REPORTTYPE_MATRIX_PATH,
    PREDICTIONS_DIR,
)

LOGGER = logging.getLogger(__name__)

TARGET_SAMPLE_SIZE = 100
PER_CATEGORY_CAP = 25

PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"


def _assign_validation_category(row: pd.Series) -> str:
    base = int(row.get("baseline_composite", 0) or 0)
    model = int(row.get("model_patient_positive", 0) or 0)
    if model == 1 and base == 1:
        cat = "TP_composite"
    elif model == 0 and base == 0:
        cat = "TN_composite"
    elif model == 1 and base == 0:
        cat = "FP_composite"
    elif model == 0 and base == 1:
        cat = "FN_composite"
    else:
        cat = "other"

    if int(row.get("any_manual_review_candidate", 0) or 0) == 1:
        if int(row.get("any_direct_delir_evidence", 0) or 0) == 0 and int(
            row.get("any_indirect_delir_evidence", 0) or 0
        ) == 1:
            return "indirect_symptoms_only"
        return "borderline_manual_review"
    return cat


def _pick(df: pd.DataFrame, mask: pd.Series, n: int, seen: set) -> List[str]:
    candidates = df.loc[mask, "PatientenID"].astype(str).tolist()
    out: List[str] = []
    for pid in candidates:
        if pid in seen:
            continue
        out.append(pid)
        seen.add(pid)
        if len(out) >= n:
            break
    return out


def build_validation_sample(
    matrix: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_size: int = TARGET_SAMPLE_SIZE,
) -> pd.DataFrame:
    m = normalize_patient_id_column(matrix)
    assert_patientenid_column(m, "patient_reporttype_matrix")
    log_patientenid_dtype_if_debug(m, "matrix")

    m["validation_sampling_category"] = m.apply(_assign_validation_category, axis=1)

    priority = [
        "FP_composite",
        "FN_composite",
        "TP_composite",
        "TN_composite",
        "borderline_manual_review",
        "indirect_symptoms_only",
    ]
    seen: set = set()
    selected_ids: List[str] = []
    per_cat = max(5, target_size // len(priority))

    for cat in priority:
        mask = m["validation_sampling_category"] == cat
        picked = _pick(m, mask, per_cat, seen)
        selected_ids.extend(picked)

    if len(selected_ids) < target_size:
        for pid in m["PatientenID"].astype(str).tolist():
            if pid not in seen:
                selected_ids.append(pid)
                seen.add(pid)
            if len(selected_ids) >= target_size:
                break

    selected_ids = selected_ids[:target_size]
    sample = m[m["PatientenID"].isin(selected_ids)].copy()
    sample = normalize_patient_id_column(sample)

    if predictions.empty or "PatientenID" not in predictions.columns:
        sample["evidence_snippets"] = ""
        return sample

    pred = normalize_patient_id_column(predictions)
    assert_patientenid_column(pred, "predictions")
    log_patientenid_dtype_if_debug(pred, "predictions")

    if "evidence_snippets" in pred.columns:
        ev = (
            pred.groupby("PatientenID")["evidence_snippets"]
            .apply(lambda s: " || ".join(str(x) for x in s if str(x).strip()))
            .reset_index(name="evidence_snippets")
        )
        ev = normalize_patient_id_column(ev)
        log_patientenid_dtype_if_debug(ev, "evidence_aggregate")
        n_before = len(sample)
        sample = sample.merge(ev, on="PatientenID", how="left")
        if len(sample) != n_before:
            LOGGER.warning(
                "Evidence merge changed row count %d -> %d; check duplicate PatientenID keys.",
                n_before,
                len(sample),
            )
    else:
        sample["evidence_snippets"] = ""

    return sample


def main(
    matrix_path: Path = PATIENT_REPORTTYPE_MATRIX_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    output_path: Path = MANUAL_VALIDATION_SAMPLE_PATH,
) -> None:
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Patient matrix missing: {matrix_path}. "
            "Run python -m src.analysis.create_patient_reporttype_matrix first."
        )
    matrix = normalize_patient_id_column(pd.read_csv(matrix_path))
    preds = (
        normalize_patient_id_column(pd.read_csv(predictions_path))
        if predictions_path.exists()
        else pd.DataFrame()
    )

    sample = build_validation_sample(matrix, preds)
    MANUAL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

    print(f"Wrote manual validation sample: {output_path}")
    print(f"sample_size={len(sample)}")
    if "validation_sampling_category" in sample.columns:
        print(sample["validation_sampling_category"].value_counts().to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
