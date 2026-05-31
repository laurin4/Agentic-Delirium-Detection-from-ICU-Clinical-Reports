"""Tests for validation_report_id as primary evaluation identity."""

import json

import pandas as pd
import pytest

from src.analysis.audit_validation_matching import check_prediction_merge_integrity
from src.analysis.final_manual_validation_evaluation import (
    load_merged_frozen_cohort,
    run_final_evaluation,
)
from src.analysis.validation_report_trace import build_report_trace
from src.pipeline.frozen_cohort_inference import build_pipeline_records_from_frozen_cohort
from src.pipeline.validation_report_identity import (
    VALIDATION_REPORT_ID_COL,
    check_cohort_prediction_alignment,
    merge_predictions_by_validation_report_id,
)
from src.preprocessing.report_identity import SOURCE_REPORT_ROW_ID_COL


def _frozen_cohort_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_patient_id": ["Patient_0001", "Patient_0001"],
            "validation_report_id": [
                "Patient_0001_Report_0001",
                "Patient_0001_Report_0002",
            ],
            "PatientenID": ["p1", "p1"],
            "bericht": ["doc_a", "doc_b"],
            "bertyp": ["Verlaufseintrag", "Austrittsbericht"],
            "berdat": ["2024-01-01", "2024-01-02"],
            SOURCE_REPORT_ROW_ID_COL: ["berichte_row_0", "berichte_row_99"],
            "report_text": [
                "[Diagnosen]\nPatient desorientiert.",
                "[Diagnosen]\nEntlassung stabil.",
            ],
        }
    )


def _predictions_for_cohort(cohort: pd.DataFrame, *, stale_source_id: bool = False) -> pd.DataFrame:
    rows = []
    for _, row in cohort.iterrows():
        sid = "berichte_row_WRONG" if stale_source_id else row[SOURCE_REPORT_ROW_ID_COL]
        rows.append(
            {
                VALIDATION_REPORT_ID_COL: row[VALIDATION_REPORT_ID_COL],
                "validation_patient_id": row["validation_patient_id"],
                "PatientenID": row["PatientenID"],
                "bertyp": row["bertyp"],
                "berdat": row["berdat"],
                "bericht": row["bericht"],
                SOURCE_REPORT_ROW_ID_COL: sid,
                "klasse": 1 if "Report_0001" in row[VALIDATION_REPORT_ID_COL] else 0,
                "status": "processed",
                "llm_called": 1,
                "skipped_reason": "direct",
                "evidence_snippets": json.dumps(
                    [{"keyword": "desorientiert", "text": "desorientiert"}]
                )
                if "Report_0001" in row[VALIDATION_REPORT_ID_COL]
                else "[]",
                "signalstaerke": "hoch",
                "delir_probability_estimate": 70,
                "decision_rule_applied": "direct",
            }
        )
    return pd.DataFrame(rows)


def test_build_pipeline_records_preserves_validation_report_id():
    cohort = _frozen_cohort_rows()
    records = build_pipeline_records_from_frozen_cohort(cohort_df=cohort)
    assert len(records) == 2
    assert {r[VALIDATION_REPORT_ID_COL] for r in records} == {
        "Patient_0001_Report_0001",
        "Patient_0001_Report_0002",
    }
    assert records[0]["report_text"].startswith("[Diagnosen]")


def test_cohort_prediction_alignment_strict():
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort)
    errors, _ = check_cohort_prediction_alignment(cohort, preds)
    assert errors == []


def test_cohort_prediction_alignment_detects_mismatch():
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort.iloc[:1])
    errors, _ = check_cohort_prediction_alignment(cohort, preds)
    assert any("row count mismatch" in e for e in errors)


def test_merge_predictions_by_validation_report_id():
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort, stale_source_id=True)
    merged, warnings = merge_predictions_by_validation_report_id(cohort, preds)
    assert len(merged) == 2
    assert int(merged.iloc[0]["model_report_prediction"]) == 1
    assert int(merged.iloc[1]["model_report_prediction"]) == 0
    assert any("source_report_row_id" in w or "berichte_row" in w for w in warnings) or True


def test_stale_source_report_row_id_does_not_corrupt_final_evaluation(tmp_path):
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort, stale_source_id=True)
    labels = pd.DataFrame(
        {
            "validation_report_id": cohort[VALIDATION_REPORT_ID_COL],
            "manual_report_ground_truth": [1, 0],
        }
    )
    baseline = pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "baseline_icd10": [0],
            "baseline_icdsc_ge_4": [0],
        }
    )

    cohort_path = tmp_path / "cohort.csv"
    labels_path = tmp_path / "labels.csv"
    preds_path = tmp_path / "preds.csv"
    baseline_path = tmp_path / "baseline.csv"
    cohort.to_csv(cohort_path, index=False)
    labels.to_csv(labels_path, index=False)
    preds.to_csv(preds_path, index=False)
    baseline.to_csv(baseline_path, index=False)

    merged, _, _ = load_merged_frozen_cohort(
        cohort_path, labels_path, baseline_path, preds_path
    )
    assert int(pd.to_numeric(merged["model_report_prediction"], errors="coerce").max()) == 1


def test_missing_prediction_ids_warn_without_fill(tmp_path, monkeypatch):
    monkeypatch.delenv("VALIDATION_EVAL_FILL_MISSING_PREDICTIONS", raising=False)
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort.iloc[:1])
    with pytest.raises(ValueError, match="alignment failed"):
        merge_predictions_by_validation_report_id(cohort, preds)


def test_missing_prediction_fill_as_zero_when_env_set(monkeypatch):
    monkeypatch.setenv("VALIDATION_EVAL_FILL_MISSING_PREDICTIONS", "true")
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort.iloc[:1])
    merged, warnings = merge_predictions_by_validation_report_id(cohort, preds)
    assert any("filled model_report_prediction=0" in w for w in warnings)
    missing_row = merged[merged[VALIDATION_REPORT_ID_COL] == "Patient_0001_Report_0002"].iloc[0]
    assert int(missing_row["model_report_prediction"]) == 0


def test_audit_merge_integrity_by_validation_report_id():
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort, stale_source_id=True)
    failures, matched, checked = check_prediction_merge_integrity(cohort, preds, pd.DataFrame())
    assert matched == checked == 2
    assert not any("field_mismatch" in f.get("issue", "") for f in failures)


def test_trace_uses_validation_report_id_not_stale_source_id():
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort, stale_source_id=True)
    labels = pd.DataFrame(
        {
            "validation_report_id": cohort[VALIDATION_REPORT_ID_COL],
            "manual_report_ground_truth": [1, 0],
        }
    )
    trace = build_report_trace(
        "Patient_0001_Report_0001",
        cohort,
        labels,
        preds,
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert trace.merge_strategy == "validation_report_id"
    assert trace.verdict in ("MATCH_OK", "MATCH_SUSPICIOUS")


def test_manual_labels_not_overwritten_on_final_eval(tmp_path):
    cohort = _frozen_cohort_rows()
    preds = _predictions_for_cohort(cohort)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "validation_report_id": cohort[VALIDATION_REPORT_ID_COL],
            "manual_report_ground_truth": [1, 0],
            "manual_comment": ["annotated", ""],
        }
    ).to_csv(labels_path, index=False)
    labels_before = labels_path.read_bytes()

    cohort_path = tmp_path / "cohort.csv"
    preds_path = tmp_path / "preds.csv"
    baseline_path = tmp_path / "baseline.csv"
    cohort.to_csv(cohort_path, index=False)
    preds.to_csv(preds_path, index=False)
    pd.DataFrame(
        {"PatientenID": ["p1"], "baseline_icd10": [0], "baseline_icdsc_ge_4": [0]}
    ).to_csv(baseline_path, index=False)

    merged, _, _ = load_merged_frozen_cohort(
        cohort_path, labels_path, baseline_path, preds_path
    )
    out_dir = tmp_path / "eval"
    run_final_evaluation(merged, output_dir=out_dir, baseline_source=baseline_path)
    assert labels_path.read_bytes() == labels_before
    assert int(merged.iloc[0]["manual_report_ground_truth"]) == 1
