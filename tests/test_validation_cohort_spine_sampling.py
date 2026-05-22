"""Patient validation sampling must use Berichte spine, not predictions-only."""

import pandas as pd

from src.analysis.export_patient_validation_cohort import (
    build_patient_level_sampling_frame,
    build_patient_validation_cohort,
    select_validation_patient_ids,
)
from src.preprocessing.berichte_filters import DOKUMENTATIONSBLATT_BERTYP


def _baseline(patient_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PatientenID": patient_ids,
            "max_icdsc": [0] * len(patient_ids),
            "baseline_icd10": [0] * len(patient_ids),
            "baseline_icdsc_ge_4": [0] * len(patient_ids),
            "baseline_composite": [0] * len(patient_ids),
        }
    )


def _berichte_spine(n_patients: int, reports_per_patient: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        pid = f"spine_p{i:03d}"
        for j in range(reports_per_patient):
            rows.append(
                {
                    "PatientenID": pid,
                    "bericht": f"{pid}_r{j}.txt",
                    "bertyp": "Verlaufseintrag",
                    "berdat": f"2024-01-{j + 1:02d}",
                }
            )
    rows.append(
        {
            "PatientenID": "spine_p000",
            "bericht": "doc_only.txt",
            "bertyp": DOKUMENTATIONSBLATT_BERTYP,
            "berdat": "2024-01-99",
        }
    )
    return pd.DataFrame(rows)


def _predictions_for_two_patients() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PatientenID": "spine_p000",
                "bericht": "spine_p000_r0.txt",
                "bertyp": "Verlaufseintrag",
                "klasse": 1,
                "signalstaerke": "hoch",
                "delir_probability_estimate": 80,
                "manual_review_candidate": "True",
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
                "PatientenID": "spine_p001",
                "bericht": "spine_p001_r0.txt",
                "bertyp": "Verlaufseintrag",
                "klasse": 0,
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


def test_spine_sampling_includes_patients_without_predictions():
    n = 25
    spine = _berichte_spine(n)
    preds = _predictions_for_two_patients()
    baseline = _baseline([f"spine_p{i:03d}" for i in range(n)])

    matrix, stats = build_patient_level_sampling_frame(
        preds, baseline, berichte_df=spine
    )
    assert stats["eligible_spine_patients"] == n
    assert stats["sampling_source"] == "berichte_spine"
    assert len(matrix) == n

    selected, _ = select_validation_patient_ids(matrix, target_n=10)
    assert len(selected) == 10
    without_pred = {f"spine_p{i:03d}" for i in range(2, n)}
    assert without_pred & set(selected)


def test_requested_n_patients_when_enough_eligible():
    n = 120
    spine = _berichte_spine(n, reports_per_patient=2)
    preds = pd.DataFrame()
    baseline = _baseline([f"spine_p{i:03d}" for i in range(n)])

    matrix, stats = build_patient_level_sampling_frame(
        preds, baseline, berichte_df=spine
    )
    assert stats["eligible_spine_patients"] == n

    selected, _ = select_validation_patient_ids(matrix, target_n=100)
    assert len(selected) == 100

    ctx = matrix[matrix["PatientenID"].isin(selected)]
    cohort = build_patient_validation_cohort(
        preds,
        baseline,
        ctx,
        selected,
        berichte_reports=spine[spine["PatientenID"].isin(selected)],
    )
    assert cohort["validation_patient_id"].nunique() == 100
    assert len(cohort) == 200


def test_missing_predictions_implicit_negative():
    spine = _berichte_spine(3)
    preds = _predictions_for_two_patients()
    matrix, _ = build_patient_level_sampling_frame(
        preds, _baseline(["spine_p000", "spine_p001", "spine_p002"]), berichte_df=spine
    )

    cohort = build_patient_validation_cohort(
        preds,
        _baseline(["spine_p000", "spine_p001", "spine_p002"]),
        matrix,
        ["spine_p002"],
        berichte_reports=spine[spine["PatientenID"] == "spine_p002"],
    )
    row = cohort.iloc[0]
    assert int(row["model_report_prediction"]) == 0
    assert row["status"] == "missing_prediction"
    assert int(row["llm_called"]) == 0
    assert row["skipped_reason"] == "missing_prediction_implicit_negative"
    assert str(row["manual_review_candidate"]).strip().lower() in ("false", "0", "")


def test_all_reports_per_selected_patient_included():
    spine = _berichte_spine(5, reports_per_patient=3)
    preds = pd.DataFrame()
    baseline = _baseline([f"spine_p{i:03d}" for i in range(5)])
    matrix, _ = build_patient_level_sampling_frame(preds, baseline, berichte_df=spine)

    cohort = build_patient_validation_cohort(
        preds,
        baseline,
        matrix[matrix["PatientenID"] == "spine_p003"],
        ["spine_p003"],
        berichte_reports=spine[spine["PatientenID"] == "spine_p003"],
    )
    assert len(cohort) == 3
    assert set(cohort["bericht"]) == {
        "spine_p003_r0.txt",
        "spine_p003_r1.txt",
        "spine_p003_r2.txt",
    }
