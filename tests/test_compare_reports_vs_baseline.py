import numpy as np
import pandas as pd

from src.pipeline.compare_reports_vs_baseline import run_compare


def _baseline_row(pid: str, **kwargs):
    base = {
        "PatientenID": pid,
        "has_delir_icd10": 0,
        "max_icdsc": 2,
        "baseline_icd10": 0,
        "baseline_icdsc_ge_1": 1,
        "baseline_icdsc_ge_2": 1,
        "baseline_icdsc_ge_3": 0,
        "baseline_icdsc_ge_4": 0,
        "baseline_icdsc_ge_5": 0,
        "baseline_icdsc_0": 0,
        "baseline_icdsc_1_to_3": 1,
        "baseline_icdsc_ge_4_grouped": 0,
        "baseline_composite": 0,
    }
    base.update(kwargs)
    return base


def test_compare_all_prediction_rows_evaluable(tmp_path):
    baseline = pd.DataFrame([_baseline_row("p1"), _baseline_row("p2")])
    pred = pd.DataFrame(
        {"PatientenID": ["p1", "p2"], "klasse": [1, 0], "extra_col": ["a", "b"]}
    )
    bpath = tmp_path / "baseline.csv"
    ppath = tmp_path / "pred.csv"
    out = tmp_path / "cmp.csv"
    excl = tmp_path / "excl.csv"
    baseline.to_csv(bpath, index=False)
    pred.to_csv(ppath, index=False)

    run_compare(baseline_path=bpath, predictions_path=ppath, output_path=out, excluded_path=excl)

    cmp_df = pd.read_csv(out)
    excl_df = pd.read_csv(excl)
    assert len(cmp_df) == 2
    assert "evidence_snippets" in cmp_df.columns
    assert len(excl_df) == 0
    assert set(cmp_df["PatientenID"].astype(str)) == {"p1", "p2"}
    assert cmp_df["agreement_report_vs_combined_baseline"].isna().all()


def test_compare_unmatched_predictions_excluded_predictions_file_unchanged(tmp_path):
    baseline = pd.DataFrame([_baseline_row("p1")])
    pred = pd.DataFrame({"PatientenID": ["p1", "orphan"], "klasse": [1, 0]})
    bpath = tmp_path / "baseline.csv"
    ppath = tmp_path / "pred.csv"
    out = tmp_path / "cmp.csv"
    excl = tmp_path / "excl.csv"
    baseline.to_csv(bpath, index=False)
    pred.to_csv(ppath, index=False)

    before = ppath.read_bytes()
    run_compare(baseline_path=bpath, predictions_path=ppath, output_path=out, excluded_path=excl)
    after = ppath.read_bytes()

    assert before == after
    cmp_df = pd.read_csv(out)
    excl_df = pd.read_csv(excl)
    assert len(cmp_df) == 1
    assert str(cmp_df.iloc[0]["PatientenID"]) == "p1"
    assert len(excl_df) == 1
    assert str(excl_df.iloc[0]["PatientenID"]) == "orphan"
    assert excl_df.iloc[0]["reason"] == "no_structured_baseline_row"
    assert "klasse" in excl_df.columns


def test_compare_incomplete_baseline_row_excluded(tmp_path):
    row = _baseline_row("p1")
    row["baseline_icd10"] = np.nan
    baseline = pd.DataFrame([row])
    pred = pd.DataFrame({"PatientenID": ["p1"], "klasse": [0]})
    bpath = tmp_path / "baseline.csv"
    ppath = tmp_path / "pred.csv"
    out = tmp_path / "cmp.csv"
    excl = tmp_path / "excl.csv"
    baseline.to_csv(bpath, index=False)
    pred.to_csv(ppath, index=False)

    run_compare(baseline_path=bpath, predictions_path=ppath, output_path=out, excluded_path=excl)

    cmp_df = pd.read_csv(out)
    excl_df = pd.read_csv(excl)
    assert len(cmp_df) == 0
    assert len(excl_df) == 1
    assert excl_df.iloc[0]["reason"] == "incomplete_baseline_columns"
