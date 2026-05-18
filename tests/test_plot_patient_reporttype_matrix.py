"""Tests for patient-level report-type matrix preview plot."""

import os
from pathlib import Path

import pandas as pd

from src.analysis import plot_patient_reporttype_matrix as plot_mod
from src.analysis.patient_reporttype_matrix import build_patient_reporttype_matrix


def _tiny_matrix() -> pd.DataFrame:
    preds = pd.DataFrame(
        {
            "PatientenID": ["p1", "p1", "p2", "p3"],
            "bertyp": [
                "Verlaufseintrag",
                "Austrittsbericht",
                "Verlegungsbericht",
                "Verlaufseintrag",
            ],
            "klasse": [1, 0, 0, 1],
            "manual_review_candidate": ["False"] * 4,
            "has_direct_delir_evidence": ["True", "False", "False", "False"],
            "has_indirect_delir_evidence": ["False"] * 4,
        }
    )
    baseline = pd.DataFrame(
        {
            "PatientenID": ["p1", "p2", "p3"],
            "max_icdsc": [5, 2, 0],
            "baseline_icd10": [0, 1, 0],
            "baseline_composite": [1, 1, 0],
        }
    )
    m = build_patient_reporttype_matrix(preds, baseline)
    m.loc[m["PatientenID"] == "p1", "manual_ground_truth"] = "1"
    m.loc[m["PatientenID"] == "p2", "manual_ground_truth"] = "0"
    return m


def test_resolve_icdsc_from_icdsc_max():
    m = _tiny_matrix()
    s = plot_mod.resolve_icdsc_ge4_series(m)
    assert list(s) == [1, 0, 0]


def test_resolve_icdsc_prefers_baseline_column():
    m = _tiny_matrix()
    m["baseline_icdsc_ge_4"] = [0, 1, 0]
    s = plot_mod.resolve_icdsc_ge4_series(m)
    assert list(s) == [0, 1, 0]


def test_build_preview_respects_n(monkeypatch):
    monkeypatch.setenv("MATRIX_PREVIEW_N", "2")
    m = _tiny_matrix()
    states, ids, texts = plot_mod.build_preview_cell_data(m)
    assert len(ids) == 2
    assert states.shape[0] == 2
    assert states.shape[1] == 8
    assert texts[0][0] in ("0", "1")


def test_manual_empty_is_unlabeled():
    m = _tiny_matrix()
    states, _, texts = plot_mod.build_preview_cell_data(m, n_patients=3)
    # p3 has empty manual_ground_truth
    assert states[2, -1] == -1
    assert texts[2][-1] == ""


def test_plot_creates_png(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIX_PREVIEW_PDF", "0")
    png = tmp_path / "preview.png"
    pdf = tmp_path / "preview.pdf"
    out = plot_mod.plot_patient_reporttype_matrix_preview(
        _tiny_matrix(), png_path=png, pdf_path=pdf, n_patients=3
    )
    assert out == png
    assert png.exists()
    assert png.stat().st_size > 500
    assert not pdf.exists()


def test_main_from_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIX_PREVIEW_N", "2")
    monkeypatch.setenv("MATRIX_PREVIEW_PDF", "0")
    csv_path = tmp_path / "matrix.csv"
    png_path = tmp_path / "preview.png"
    _tiny_matrix().to_csv(csv_path, index=False)
    plot_mod.main(matrix_path=csv_path, png_path=png_path, pdf_path=tmp_path / "preview.pdf")
    assert png_path.exists()


def test_create_matrix_module_writes_preview(tmp_path, monkeypatch):
    pred = tmp_path / "pred.csv"
    base = tmp_path / "base.csv"
    out = tmp_path / "matrix.csv"
    png = tmp_path / "preview.png"
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
            "max_icdsc": [4],
            "baseline_icd10": [0],
            "baseline_composite": [1],
        }
    ).to_csv(base, index=False)

    import src.analysis.create_patient_reporttype_matrix as mod

    monkeypatch.setattr(mod, "DEFAULT_PREDICTIONS_PATH", pred)
    monkeypatch.setattr(mod, "STRUCTURED_BASELINE_PATH", base)
    monkeypatch.setattr(mod, "PATIENT_REPORTTYPE_MATRIX_PATH", out)
    monkeypatch.setattr(mod, "PATIENT_LEVEL_ANALYSIS_DIR", tmp_path)
    monkeypatch.setattr(mod, "PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG", png)
    monkeypatch.setenv("MATRIX_PREVIEW_PDF", "0")

    mod.main(predictions_path=pred, baseline_path=base, output_path=out)
    assert out.exists()
    assert png.exists()
