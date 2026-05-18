"""Smoke tests for run_pipeline helper functions.

These guard against regressions where module-level helpers reference
undefined names (which compileall does not catch, but a runtime call does).
"""

from src.pipeline.run_pipeline import (
    UNKNOWN_BERTYP,
    _assert_binary_klassen,
    _compact_line,
    accumulate_bertyp_stat,
    format_bertyp_summary_lines,
    resolve_bertyp,
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


def test_resolve_bertyp_unknown_when_missing():
    assert resolve_bertyp({}) == UNKNOWN_BERTYP
    assert resolve_bertyp({"bertyp": ""}) == UNKNOWN_BERTYP
    assert resolve_bertyp({"bertyp": "  Verlaufseintrag  "}) == "Verlaufseintrag"


def test_compact_line_includes_bertyp():
    ev = {
        "evidence_snippets": [],
        "original_report_text_length": 100,
        "llm_report_text_length": 0,
        "llm_text_reduction_method": "no_evidence_prefilter_skip",
    }
    line = _compact_line(
        1, 3, "p1", ev, bertyp="Austrittsbericht", status="skipped", klasse=0, signal="niedrig"
    )
    assert "bertyp=Austrittsbericht" in line
    assert "ID=p1" in line


def test_bertyp_summary_counts():
    from collections import defaultdict

    from src.pipeline.run_pipeline import _new_bertyp_stats

    stats = defaultdict(_new_bertyp_stats)
    accumulate_bertyp_stat(stats, "Verlaufseintrag", skipped=True, failed=False, klasse=0)
    accumulate_bertyp_stat(stats, "Verlaufseintrag", skipped=False, failed=False, klasse=1)
    accumulate_bertyp_stat(stats, "Austrittsbericht", skipped=False, failed=False, klasse=1)
    accumulate_bertyp_stat(stats, "Austrittsbericht", skipped=False, failed=True, klasse=0)

    lines = format_bertyp_summary_lines(dict(stats))
    text = "\n".join(lines)
    assert "Verlaufseintrag: total=2, sent_to_llm=1, skipped=1, positives=1" in text
    assert "Austrittsbericht: total=2, sent_to_llm=1, skipped=0, positives=1" in text
