"""Tests for baseline merge validation in compare_reports_vs_baseline."""

import pandas as pd
import pytest

from src.pipeline.compare_reports_vs_baseline import (
    REQUIRED_BASELINE_COLUMNS,
    _raise_if_incomplete_baseline_merge,
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
    }


def test_raise_if_incomplete_merge_detects_unmatched_patient():
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
    with pytest.raises(ValueError) as excinfo:
        _raise_if_incomplete_baseline_merge(merged)
    msg = str(excinfo.value)
    assert "p_missing" in msg
    assert "without complete baseline" in msg


def test_raise_if_incomplete_merge_passes_when_complete():
    merged = pd.DataFrame([{**_minimal_baseline_row("p1"), "klasse": 0}])
    _raise_if_incomplete_baseline_merge(merged)


def test_raise_if_missing_required_column_in_merge_result():
    merged = pd.DataFrame([{"PatientenID": "p1", "klasse": 0}])
    with pytest.raises(ValueError) as excinfo:
        _raise_if_incomplete_baseline_merge(merged)
    assert "missing required baseline columns" in str(excinfo.value)
