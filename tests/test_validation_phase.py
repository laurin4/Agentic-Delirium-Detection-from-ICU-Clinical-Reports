"""Validation-phase: Dokumentationsblatt, composite baseline, patient matrix, sampling."""

from pathlib import Path

import pandas as pd
import pytest

from src.agents.delirium_probability import delirium_probability_estimate
from src.analysis.export_manual_validation_sample import (
    TARGET_SAMPLE_SIZE,
    _assign_validation_category,
    build_validation_sample,
)
from src.analysis.patient_reporttype_matrix import (
    build_patient_reporttype_matrix,
    ensure_baseline_icdsc_ge_4_column,
)
from src.pipeline.prepare_structured_data import add_binary_baselines
from src.preprocessing.berichte_filters import (
    DOKUMENTATIONSBLATT_BERTYP,
    exclude_dokumentationsblatt,
    is_dokumentationsblatt,
)
from src.preprocessing.berichte_mapper import build_report_level_berichte_records


def test_is_dokumentationsblatt_exact():
    assert is_dokumentationsblatt("Dokumentationsblatt")
    assert not is_dokumentationsblatt("Verlaufseintrag")
    assert not is_dokumentationsblatt(" dokumentationsblatt ")


def test_exclude_dokumentationsblatt_keeps_raw_count():
    df = pd.DataFrame(
        {
            "PatientID": ["p1", "p1", "p2"],
            "bertyp": ["Verlaufseintrag", DOKUMENTATIONSBLATT_BERTYP, "Austrittsbericht"],
            "diag": ["a", "b", "c"],
        }
    )
    filtered, n = exclude_dokumentationsblatt(df)
    assert n == 1
    assert len(df) == 3
    assert len(filtered) == 2
    assert not filtered["bertyp"].map(is_dokumentationsblatt).any()


def test_build_report_level_excludes_dokumentationsblatt(tmp_path):
    path = tmp_path / "Berichte.csv"
    path.write_text(
        "PatientID;bertyp;diag\n"
        "p1;Verlaufseintrag;Delir\n"
        "p1;Dokumentationsblatt;noise\n"
        "p2;Austrittsbericht;ok\n",
        encoding="utf-8",
    )
    records, excluded = build_report_level_berichte_records(path)
    assert excluded == 1
    assert len(records) == 2
    assert {r["bertyp"] for r in records} == {"Verlaufseintrag", "Austrittsbericht"}


def test_baseline_composite_or_logic(monkeypatch):
    monkeypatch.setattr("src.pipeline.paths.BASELINE_COMPOSITE_MODE", "OR")
    df = pd.DataFrame(
        {
            "has_delir_icd10": [0, 1, 0, 1],
            "max_icdsc": [0, 2, 5, 3],
        }
    )
    out = add_binary_baselines(df)
    assert list(out["baseline_composite"]) == [0, 1, 1, 1]


def test_patient_reporttype_matrix_aggregation():
    preds = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1", "p1", "p2"],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag", "Dokumentationsblatt", "Austrittsbericht"],
            "klasse": [0, 1, 1, 0],
            "manual_review_candidate": ["False", "True", "True", "False"],
            "has_direct_delir_evidence": ["False", "True", "True", "False"],
            "has_indirect_delir_evidence": ["False", "False", "False", "False"],
        }
    )
    baseline = pd.DataFrame(
        {
            "PatientenID": ["p1", "p2"],
            "max_icdsc": [5, 0],
            "baseline_icd10": [0, 0],
            "baseline_icdsc_ge_4": [1, 0],
            "baseline_composite": [1, 0],
        }
    )
    m = build_patient_reporttype_matrix(preds, baseline)
    assert "baseline_icdsc_ge_4" in m.columns
    row1 = m.loc[m["PatientenID"] == "p1"].iloc[0]
    assert int(row1["baseline_icdsc_ge_4"]) == 1
    assert int(row1["ICDSC_max"]) == 5
    assert int(row1["Verlaufseintrag"]) == 1
    assert int(row1["n_verlaufseintrag"]) == 2
    assert int(row1["model_patient_positive"]) == 1
    assert int(row1["discrepancy_model_vs_baseline"]) == 0
    row2 = m.loc[m["PatientenID"] == "p2"].iloc[0]
    assert int(row2["Austrittsbericht"]) == 0
    assert int(row2["discrepancy_model_vs_baseline"]) == 0


def test_matrix_derives_baseline_icdsc_ge_4_from_icdsc_max():
    preds = pd.DataFrame(
        {
            "PatientenID": ["p1", "p2"],
            "bertyp": ["Verlaufseintrag", "Verlaufseintrag"],
            "klasse": [0, 0],
        }
    )
    baseline = pd.DataFrame(
        {
            "PatientenID": ["p1", "p2"],
            "max_icdsc": [5, 3],
            "baseline_icd10": [0, 0],
            "baseline_composite": [1, 0],
        }
    )
    m = build_patient_reporttype_matrix(preds, baseline)
    assert int(m.loc[m["PatientenID"] == "p1", "baseline_icdsc_ge_4"].iloc[0]) == 1
    assert int(m.loc[m["PatientenID"] == "p2", "baseline_icdsc_ge_4"].iloc[0]) == 0


def test_ensure_baseline_icdsc_ge_4_on_legacy_matrix_without_column():
    legacy = pd.DataFrame(
        {
            "PatientenID": ["p1", "p2"],
            "ICDSC_max": [6, 2],
            "ICD10": [0, 0],
            "model_patient_positive": [0, 0],
        }
    )
    out = ensure_baseline_icdsc_ge_4_column(legacy)
    assert list(out["baseline_icdsc_ge_4"]) == [1, 0]


def test_delirium_probability_estimate_ranges():
    assert delirium_probability_estimate("niedrig", 0) <= 25
    assert delirium_probability_estimate("hoch", 1) >= 75
    low = delirium_probability_estimate(
        "mittel",
        1,
        manual_review_candidate=True,
        decision_rule_applied="indirect_symptoms_positive_review_needed",
    )
    assert 0 <= low <= 100


def test_validation_sampling_mixed_categories():
    matrix = pd.DataFrame(
        {
            "PatientenID": [f"p{i}" for i in range(20)],
            "baseline_composite": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "model_patient_positive": [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
            "any_manual_review_candidate": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "any_direct_delir_evidence": [0] * 20,
            "any_indirect_delir_evidence": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    matrix["validation_sampling_category"] = matrix.apply(_assign_validation_category, axis=1)
    cats = set(matrix["validation_sampling_category"])
    assert "FP_composite" in cats or "FN_composite" in cats
    sample = build_validation_sample(matrix, pd.DataFrame(), target_size=12)
    assert len(sample) <= 12
    assert "validation_sampling_category" in sample.columns
    assert len(sample["validation_sampling_category"].unique()) >= 2


def test_clean_patient_id_value_no_float_artifact():
    from src.pipeline.schema_normalize import clean_patient_id_value

    assert clean_patient_id_value(12345) == "12345"
    assert clean_patient_id_value(12345.0) == "12345"
    assert clean_patient_id_value(" 99 ") == "99"
    assert clean_patient_id_value(float("nan")) == ""


def test_manual_validation_merge_int64_vs_object():
    from src.analysis.export_manual_validation_sample import build_validation_sample

    matrix = pd.DataFrame(
        {
            "PatientenID": [100, 200],
            "baseline_composite": [0, 1],
            "model_patient_positive": [1, 0],
            "any_manual_review_candidate": [0, 0],
            "any_direct_delir_evidence": [0, 0],
            "any_indirect_delir_evidence": [0, 0],
        }
    )
    matrix["PatientenID"] = matrix["PatientenID"].astype("int64")
    preds = pd.DataFrame(
        {
            "PatientenID": ["100", "200"],
            "evidence_snippets": ["[a]", "[b]"],
            "klasse": [1, 0],
        }
    )
    sample = build_validation_sample(matrix, preds, target_size=2)
    assert len(sample) == 2
    assert sample["PatientenID"].dtype == object
    assert set(sample["PatientenID"]) == {"100", "200"}
    assert sample["evidence_snippets"].notna().all()


def test_normalize_patient_id_column_merge_count_preserved():
    from src.pipeline.schema_normalize import normalize_patient_id_column

    left = pd.DataFrame({"PatientenID": [1, 2], "x": [10, 20]})
    left["PatientenID"] = left["PatientenID"].astype("int64")
    right = pd.DataFrame({"PatientenID": ["1", "2"], "y": [100, 200]})
    merged = normalize_patient_id_column(left).merge(
        normalize_patient_id_column(right), on="PatientenID", how="left"
    )
    assert len(merged) == 2
    assert merged["y"].tolist() == [100, 200]


def test_create_patient_matrix_module(tmp_path, monkeypatch):
    pred = tmp_path / "pred.csv"
    base = tmp_path / "base.csv"
    out = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "bertyp": ["Verlaufseintrag"],
            "klasse": [1],
            "manual_review_candidate": ["False"],
            "has_direct_delir_evidence": ["True"],
            "has_indirect_delir_evidence": ["False"],
            "evidence_snippets": ["[]"],
        }
    ).to_csv(pred, index=False)
    pd.DataFrame(
        {
            "PatientenID": ["p1"],
            "max_icdsc": [2],
            "baseline_icd10": [0],
            "baseline_composite": [0],
        }
    ).to_csv(base, index=False)

    import src.analysis.create_patient_reporttype_matrix as mod

    monkeypatch.setattr(mod, "DEFAULT_PREDICTIONS_PATH", pred)
    monkeypatch.setattr(mod, "STRUCTURED_BASELINE_PATH", base)
    monkeypatch.setattr(mod, "PATIENT_REPORTTYPE_MATRIX_PATH", out)
    monkeypatch.setattr(mod, "PATIENT_LEVEL_ANALYSIS_DIR", tmp_path)

    png = tmp_path / "matrix_preview.png"
    monkeypatch.setattr(mod, "PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG", png)
    monkeypatch.setenv("MATRIX_PREVIEW_PDF", "0")

    mod.main(predictions_path=pred, baseline_path=base, output_path=out)
    assert out.exists()
    assert png.exists()
    df = pd.read_csv(out)
    assert "discrepancy_model_vs_baseline" in df.columns
    assert int(df.loc[0, "discrepancy_model_vs_baseline"]) == 1
