"""Manual review CSV export (TP/TN/FP/FN samples per primary baseline)."""

import pandas as pd

from src.analysis.run_error_review_export import run_manual_review_export


def _minimal_comparison_row(pid: str, klasse: int, b10: int, b4: int) -> dict:
    return {
        "PatientenID": pid,
        "bericht": f"{pid}.txt",
        "klasse": klasse,
        "klassifikation": "delir" if klasse == 1 else "kein_delir",
        "signalstaerke": "mittel",
        "anzahl_treffer": 1,
        "delir_signale": "",
        "evidence_snippets": "[]",
        "kontext": "",
        "begruendung": "",
        "has_delir_icd10": b10,
        "max_icdsc": 4 if b4 else 2,
        "baseline_icd10": b10,
        "baseline_icdsc_ge_1": 1,
        "baseline_icdsc_ge_2": 1,
        "baseline_icdsc_ge_3": int(b4),
        "baseline_icdsc_ge_4": b4,
        "baseline_icdsc_ge_5": 0,
        "baseline_icdsc_0": 0,
        "baseline_icdsc_1_to_3": 1 - b4,
        "baseline_icdsc_ge_4_grouped": b4,
        "llm_text_reduction_method": "keyword_windows",
        "original_report_text_length": 100,
        "llm_report_text_length": 100,
        "llm_skipped_by_prefilter": False,
    }


def test_manual_review_export_respects_five_per_category(tmp_path):
    rows = []
    # TP klasse=1 baseline_icd10=1 (10 rows -> export 5)
    for i in range(10):
        rows.append(_minimal_comparison_row(f"tp{i}", 1, 1, 0))
    for i in range(3):
        rows.append(_minimal_comparison_row(f"tn{i}", 0, 0, 0))
    for i in range(2):
        rows.append(_minimal_comparison_row(f"fp{i}", 1, 0, 0))
    rows.append(_minimal_comparison_row("fn0", 0, 1, 0))
    # No FN for baseline_icdsc_ge_4 in this toy set (all baseline_icdsc_ge_4=0)
    df = pd.DataFrame(rows)
    cmp_path = tmp_path / "cmp.csv"
    df.to_csv(cmp_path, index=False)
    out_dir = tmp_path / "manual_review"
    run_manual_review_export(cmp_path=cmp_path, out_dir=out_dir, max_per_category=5)

    icd10 = pd.read_csv(out_dir / "manual_review_cases_baseline_icd10.csv")
    assert len(icd10[icd10["error_type"] == "TP"]) == 5
    assert len(icd10[icd10["error_type"] == "TN"]) == 3
    assert len(icd10[icd10["error_type"] == "FP"]) == 2
    assert len(icd10[icd10["error_type"] == "FN"]) == 1

    summary = pd.read_csv(out_dir / "manual_review_summary.csv")
    assert len(summary) == 2
    icd10_sum = summary.loc[summary["baseline_name"] == "baseline_icd10"].iloc[0]
    assert int(icd10_sum["count_TP"]) == 10
    assert int(icd10_sum["exported_TP"]) == 5

    for col in ("manual_label_0_1_2", "manual_comment", "reviewer", "review_date"):
        assert col in icd10.columns

    report = (out_dir / "report.txt").read_text(encoding="utf-8")
    assert "False positives" in report
    assert "Manual review is required" in report


def test_manual_review_export_empty_category_no_failure(tmp_path):
    df = pd.DataFrame(
        [
            _minimal_comparison_row("a", 0, 0, 0),
            _minimal_comparison_row("b", 0, 0, 0),
        ]
    )
    cmp_path = tmp_path / "cmp.csv"
    df.to_csv(cmp_path, index=False)
    out_dir = tmp_path / "manual_review"
    run_manual_review_export(cmp_path=cmp_path, out_dir=out_dir)
    icd10 = pd.read_csv(out_dir / "manual_review_cases_baseline_icd10.csv")
    assert "error_type" in icd10.columns
    assert set(icd10["error_type"].unique()) <= {"TN"}
    summary = pd.read_csv(out_dir / "manual_review_summary.csv")
    row = summary.loc[summary["baseline_name"] == "baseline_icd10"].iloc[0]
    assert int(row["count_FP"]) == 0
    assert int(row["exported_FP"]) == 0
