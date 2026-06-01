"""Tests for prompt V1/V2 and versioned validation output paths."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.archive_current_v1_results import (
    PROTECTED_PATHS,
    archive_current_v1_results,
    copy_path,
)
from src.pipeline.paths import (
    FINAL_MANUAL_VALIDATION_EVAL_DIR,
    PROJECT_ROOT,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.pipeline.prompt_run_paths import (
    get_prompt_run_dir,
    get_versioned_final_eval_dir,
    get_versioned_predictions_path,
    is_versioned_validation_run,
)
from src.pipeline.prompt_selector import (
    load_interpretation_prompt,
    resolve_interpretation_prompt_path,
)

NEGATIVE_EXAMPLE = (
    "Neurologisch zeigte sich ein unauffälliger Verlauf ohne Auftreten "
    "fokal-neurologischer Defizite oder eines Delirs."
)

PROMPTS_DIR = PROJECT_ROOT / "prompts"
V1_PATH = PROMPTS_DIR / "delirium_case_classification_v1.txt"
V2_PATH = PROMPTS_DIR / "delirium_case_classification_v2.txt"


@pytest.fixture
def clean_prompt_run_env(monkeypatch):
    for key in ("DELIRIUM_PROMPT_VERSION", "VALIDATION_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    yield


def test_v1_prompt_exists():
    assert V1_PATH.is_file()
    assert V1_PATH.stat().st_size > 100


def test_v2_prompt_exists():
    assert V2_PATH.is_file()


def test_v2_contains_exact_negative_sentence():
    text = V2_PATH.read_text(encoding="utf-8")
    assert NEGATIVE_EXAMPLE in text


def test_prompt_selector_v1_v2():
    assert resolve_interpretation_prompt_path("v1").name == "delirium_case_classification_v1.txt"
    assert resolve_interpretation_prompt_path("v2").name == "delirium_case_classification_v2.txt"
    assert "Du bist ein klinisches Bewertungssystem" in load_interpretation_prompt("v1")


def test_versioned_predictions_path(clean_prompt_run_env, monkeypatch):
    from src.pipeline.prompt_run_paths import (
        resolve_cohort_predictions_output_path,
        resolve_validation_predictions_path,
    )

    assert resolve_cohort_predictions_output_path() == VALIDATION_COHORT_PREDICTIONS_PATH
    assert resolve_validation_predictions_path() == VALIDATION_COHORT_PREDICTIONS_PATH
    monkeypatch.setenv("DELIRIUM_PROMPT_VERSION", "v2")
    monkeypatch.setenv("VALIDATION_RUN_ID", "run_01")
    assert is_versioned_validation_run()
    expected = (
        get_prompt_run_dir("v2", "run_01")
        / "predictions"
        / "validation_cohort_predictions.csv"
    )
    assert get_versioned_predictions_path() == expected
    assert resolve_cohort_predictions_output_path() == expected
    assert resolve_validation_predictions_path() == expected


def test_versioned_final_eval_dir(clean_prompt_run_env, monkeypatch):
    assert get_versioned_final_eval_dir() == FINAL_MANUAL_VALIDATION_EVAL_DIR
    monkeypatch.setenv("DELIRIUM_PROMPT_VERSION", "v1")
    monkeypatch.setenv("VALIDATION_RUN_ID", "run_02")
    assert get_versioned_final_eval_dir() == get_prompt_run_dir("v1", "run_02") / "final_evaluation"


def test_default_legacy_paths_without_env(clean_prompt_run_env, monkeypatch):
    from src.pipeline import predictions_source as ps
    from src.pipeline.prompt_run_paths import resolve_cohort_predictions_output_path

    cohort = Path("/tmp/test_cohort_preds.csv")
    monkeypatch.setattr(ps, "VALIDATION_COHORT_PREDICTIONS_PATH", cohort)
    assert resolve_cohort_predictions_output_path() == VALIDATION_COHORT_PREDICTIONS_PATH
    assert ps.resolve_predictions_path(source=ps.PREDICTIONS_SOURCE_VALIDATION_COHORT) == cohort


def test_archive_copies_without_moving(tmp_path, monkeypatch):
    src_pred = tmp_path / "legacy" / "validation_cohort_predictions.csv"
    src_pred.parent.mkdir(parents=True)
    src_pred.write_text("a,b\n1,2\n", encoding="utf-8")
    src_eval = tmp_path / "legacy" / "final_evaluation"
    src_eval.mkdir()
    (src_eval / "final_metrics_summary.csv").write_text("method,x\nmodel,1\n", encoding="utf-8")

    dst_root = tmp_path / "prompt_runs" / "v1" / "run_01"
    targets = [
        (src_pred, dst_root / "predictions" / "validation_cohort_predictions.csv"),
        (src_eval, dst_root / "final_evaluation"),
    ]
    archive_current_v1_results(targets=targets)

    assert src_pred.exists()
    assert (dst_root / "predictions" / "validation_cohort_predictions.csv").exists()
    assert (dst_root / "final_evaluation" / "final_metrics_summary.csv").exists()


def test_archive_never_overwrites_frozen_labels():
    for protected in PROTECTED_PATHS:
        with pytest.raises(ValueError, match="protected"):
            copy_path(protected, protected.parent / "copy.csv")


def test_agent_interpretation_txt_unchanged_reference():
    legacy = PROMPTS_DIR / "agent_interpretation.txt"
    v1 = V1_PATH.read_text(encoding="utf-8")
    assert legacy.read_text(encoding="utf-8") == v1
