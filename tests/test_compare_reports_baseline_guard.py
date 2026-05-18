"""Tests for baseline merge validation in compare_reports_vs_baseline."""

import pandas as pd
import pytest

from src.pipeline.compare_reports_vs_baseline import (
    REQUIRED_BASELINE_COLUMNS,
    _ensure_required_baseline_columns_exist,
    _split_evaluable_vs_excluded,
)


def _minimal_baseline_row(pid: str) -> dict:
    return {
        "PatientenID": pid,
        "has_delir_icd10": 0,
        "max_icdsc": 0.0,
        "baseline_icd10": 0,
        "baseline_icdsc_ge_1": 0,
        "baseline_icdsc_ge_2": 0,
        "baseline_icdsc_ge_3": 0,
        "baseline_icdsc_ge_4": 0,
        "baseline_icdsc_ge_5": 0,
        "baseline_icdsc_0": 1,
        "baseline_icdsc_1_to_3": 0,
        "baseline_icdsc_ge_4_grouped": 0,
        "baseline_composite": 0,
    }


def test_split_marks_unmatched_prediction_as_not_evaluable():
    merged = pd.DataFrame(
        [
            {**_minimal_baseline_row("p1"), "klasse": 0},
            {
                "PatientenID": "p_missing",
                "klasse": 0,
                **{c: float("nan") for c in REQUIRED_BASELINE_COLUMNS},
            },
        ]
    )
    baseline_ids = {"p1"}
    evaluable_mask, reason = _split_evaluable_vs_excluded(merged, baseline_ids)
    assert evaluable_mask.tolist() == [True, False]
    assert reason.tolist() == ["", "no_structured_baseline_row"]


def test_split_all_rows_evaluable_when_baseline_complete():
    merged = pd.DataFrame([{**_minimal_baseline_row("p1"), "klasse": 0}])
    baseline_ids = {"p1"}
    evaluable_mask, reason = _split_evaluable_vs_excluded(merged, baseline_ids)
    assert evaluable_mask.all()
    assert (reason == "").all()


def test_ensure_required_columns_raises_when_missing_in_merge_result():
    merged = pd.DataFrame([{"PatientenID": "p1", "klasse": 0}])
    with pytest.raises(ValueError) as excinfo:
        _ensure_required_baseline_columns_exist(merged)
    assert "missing required baseline columns" in str(excinfo.value)
