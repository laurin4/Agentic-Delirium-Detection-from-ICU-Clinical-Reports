"""Raw Berichte spine must export every included report for selected patients."""

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.export_patient_validation_cohort import (
    build_patient_validation_cohort,
    build_patient_level_sampling_frame,
)
from src.analysis.validation_cohort_reports import (
    SOURCE_REPORT_ROW_ID_COL,
    assert_spine_row_count_preserved,
    build_complete_validation_reports_frame,
    load_raw_included_report_spine,
)
from src.preprocessing.berichte_filters import DOKUMENTATIONSBLATT_BERTYP


def _baseline(pid: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientenID": [pid],
            "max_icdsc": [0],
            "baseline_icd10": [0],
            "baseline_icdsc_ge_4": [0],
            "baseline_composite": [0],
        }
    )


def _matrix(pid: str, n_verlauf: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientenID": [pid],
            "baseline_icd10": [0],
            "baseline_icdsc_ge_4": [0],
            "model_patient_positive": [0],
            "n_verlaufseintrag": [n_verlauf],
            "n_verlegungsbericht": [0],
            "n_austrittsbericht": [0],
            "any_manual_review_candidate": [0],
        }
    )


def test_seven_raw_reports_two_predictions_exports_all_seven():
    pid = "p_seven"
    raw_rows = []
    for i in range(7):
        raw_rows.append(
            {
                "PatientenID": pid,
                "bername": f"report_{i}.txt",
                "bertyp": "Verlaufseintrag" if i < 5 else "Austrittsbericht",
                "berdat": f"2024-02-{i + 1:02d}",
                "diag": "clinical text",
            }
        )
    raw_rows.append(
        {
            "PatientenID": pid,
            "bericht": "doc.txt",
            "bertyp": DOKUMENTATIONSBLATT_BERTYP,
            "berdat": "2024-02-99",
        }
    )
    raw_df = pd.DataFrame(raw_rows)
    spine = load_raw_included_report_spine(Path("."), berichte_df=raw_df)
    assert len(spine) == 7

    from src.preprocessing.report_identity import PIPELINE_BERICHT_COL

    preds = pd.DataFrame(
        [
            {
                "PatientenID": pid,
                "bericht": spine.iloc[0][PIPELINE_BERICHT_COL],
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-02-01",
                "klasse": 1,
                "status": "processed",
                "llm_called": 1,
                "skipped_reason": "direct",
                "signalstaerke": "hoch",
                "delir_probability_estimate": 80,
                "manual_review_candidate": "False",
                "decision_rule_applied": "x",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 10,
                "llm_report_text_length": 5,
                "llm_text_reduction_method": "structured_evidence_extraction",
            },
            {
                "PatientenID": pid,
                "bericht": spine.iloc[3][PIPELINE_BERICHT_COL],
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-02-04",
                "klasse": 0,
                "status": "skipped",
                "llm_called": 0,
                "skipped_reason": "no_evidence_prefilter_skip",
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 0,
                "manual_review_candidate": "False",
                "decision_rule_applied": "no_evidence_prefilter_skip",
                "llm_skipped_by_prefilter": True,
                "llm_text_reduction_method": "no_evidence_prefilter_skip",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 10,
                "llm_report_text_length": 0,
            },
        ]
    )

    merged, stats = build_complete_validation_reports_frame(preds, [pid], berichte_df=spine)
    assert stats["raw_spine_selected_rows"] == 7
    assert stats["exported_cohort_rows"] == 7
    assert len(merged) == 7
    assert stats["prediction_matched_reports"] == 2
    assert int((merged["status"] == "missing_prediction").sum()) == 5

    cohort = build_patient_validation_cohort(
        preds,
        _baseline(pid),
        _matrix(pid, n_verlauf=5),
        [pid],
        berichte_reports=spine,
        raw_spine_for_assert=spine,
    )
    assert len(cohort) == 7
    assert SOURCE_REPORT_ROW_ID_COL in cohort.columns
    assert len(cohort[SOURCE_REPORT_ROW_ID_COL].unique()) == 7


def test_duplicate_verlaufseintrage_not_deduplicated():
    pid = "p_dup"
    raw_df = pd.DataFrame(
        [
            {
                "PatientenID": pid,
                "bericht": "same_name.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-03-01",
            },
            {
                "PatientenID": pid,
                "bericht": "same_name.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-03-02",
            },
            {
                "PatientenID": pid,
                "bericht": "other.txt",
                "bertyp": "Verlegungsbericht",
                "berdat": "2024-03-03",
            },
        ]
    )
    spine = load_raw_included_report_spine(Path("."), berichte_df=raw_df)
    assert len(spine) == 3

    preds = pd.DataFrame(
        [
            {
                "PatientenID": pid,
                "bericht": "same_name.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-03-01",
                "klasse": 1,
                "signalstaerke": "hoch",
                "delir_probability_estimate": 50,
                "manual_review_candidate": "False",
                "decision_rule_applied": "x",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 1,
                "llm_text_reduction_method": "structured_evidence_extraction",
            }
        ]
    )
    merged, stats = build_complete_validation_reports_frame(preds, [pid], berichte_df=spine)
    assert len(merged) == 3
    assert stats["raw_spine_selected_rows"] == stats["exported_cohort_rows"] == 3


def test_merge_does_not_reduce_row_count():
    pid = "p_keep"
    spine = load_raw_included_report_spine(
        Path("."),
        berichte_df=pd.DataFrame(
            [
                {
                    "PatientenID": pid,
                    "bericht": "a",
                    "bertyp": "Verlaufseintrag",
                    "berdat": "2024-01-01",
                },
                {
                    "PatientenID": pid,
                    "bericht": "b",
                    "bertyp": "Austrittsbericht",
                    "berdat": "2024-01-02",
                },
            ]
        ),
    )
    preds = pd.DataFrame(
        [
            {
                "PatientenID": pid,
                "bericht": "a",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "klasse": 0,
                "signalstaerke": "niedrig",
                "delir_probability_estimate": 0,
                "manual_review_candidate": "False",
                "decision_rule_applied": "x",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 0,
                "llm_text_reduction_method": "structured_evidence_extraction",
            },
            {
                "PatientenID": pid,
                "bericht": "a",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
                "klasse": 1,
                "signalstaerke": "hoch",
                "delir_probability_estimate": 90,
                "manual_review_candidate": "False",
                "decision_rule_applied": "y",
                "evidence_snippets": "[]",
                "delir_signale": "",
                "kontext": "",
                "begruendung": "",
                "original_report_text_length": 1,
                "llm_report_text_length": 1,
                "llm_text_reduction_method": "structured_evidence_extraction",
            },
        ]
    )
    merged, _ = build_complete_validation_reports_frame(preds, [pid], berichte_df=spine)
    assert len(merged) == len(spine)


def test_assert_spine_row_count_raises_on_mismatch():
    spine = pd.DataFrame(
        {
            SOURCE_REPORT_ROW_ID_COL: ["berichte_row_0", "berichte_row_1"],
            "PatientenID": ["p1", "p1"],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "berdat": ["2024-01-01", "2024-01-02"],
            "bericht": ["a", "b"],
        }
    )
    merged = spine.iloc[[0]].copy()
    with pytest.raises(ValueError, match="raw spine rows=2"):
        assert_spine_row_count_preserved(spine, merged)


def test_dokumentationsblatt_excluded_only():
    raw_df = pd.DataFrame(
        [
            {
                "PatientenID": "p1",
                "bericht": "v.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": "2024-01-01",
            },
            {
                "PatientenID": "p1",
                "bericht": "d.txt",
                "bertyp": DOKUMENTATIONSBLATT_BERTYP,
                "berdat": "2024-01-02",
            },
        ]
    )
    spine = load_raw_included_report_spine(Path("."), berichte_df=raw_df)
    assert len(spine) == 1
    assert spine.iloc[0]["bertyp"] == "Verlaufseintrag"


def test_sampling_frame_preserves_all_spine_rows_per_patient():
    pid = "p_multi"
    raw_df = pd.DataFrame(
        [
            {
                "PatientenID": pid,
                "bericht": f"r{i}.txt",
                "bertyp": "Verlaufseintrag",
                "berdat": f"2024-04-{i:02d}",
            }
            for i in range(1, 5)
        ]
    )
    matrix, stats = build_patient_level_sampling_frame(
        pd.DataFrame(), _baseline(pid), berichte_df=raw_df
    )
    assert stats["eligible_spine_patients"] == 1
    assert int(matrix.loc[matrix["PatientenID"] == pid, "n_verlaufseintrag"].iloc[0]) == 4
