"""Smoke tests for run_pipeline helper functions.

These guard against regressions where module-level helpers reference
undefined names (which compileall does not catch, but a runtime call does).
"""

from src.pipeline.run_pipeline import (
    _assert_binary_klassen,
    _get_model_named_output_path,
    _get_output_path,
    _sanitize_provider_model_slug,
)


def test_get_output_path_returns_csv():
    path = _get_output_path()
    assert path.name == "agent1_agent2_agent3_results_prompt.csv"


def test_get_model_named_output_path_uses_provider_and_label():
    path = _get_model_named_output_path()
    name = path.name
    assert name.startswith("agent_results_")
    assert name.endswith(".csv")


def test_sanitize_provider_model_slug_strips_special_chars():
    assert _sanitize_provider_model_slug("ollama", "qwen2.5:7b") == "ollama_qwen2_5_7b"
    assert _sanitize_provider_model_slug("usz_api", "gemma4_26b_usz") == "usz_api_gemma4_26b_usz"


def test_assert_binary_klassen_accepts_zero_and_one():
    _assert_binary_klassen([{"klasse": 0}, {"klasse": "1"}])


def test_assert_binary_klassen_rejects_invalid():
    import pytest

    with pytest.raises(ValueError):
        _assert_binary_klassen([{"klasse": 2}])
    with pytest.raises(ValueError):
        _assert_binary_klassen([{"klasse": "abc"}])
