"""Tests for LLM report text reduction (keyword windows + fallback)."""

import pytest

from src.preprocessing.report_text_llm_reduction import (
    METHOD_FALLBACK,
    METHOD_KEYWORD,
    TOTAL_CHAR_CAP_DEFAULT,
    reduce_report_text_for_llm,
)


def test_explicit_delir_in_diag_preserved_with_section_label():
    """Explicit delirium term in diagnoses stays in output with [Diagnosen] heading (keyword path)."""
    filler = "A" * 4000
    combined = f"[Diagnosen]\nPatient mit akutem Delir dokumentiert. {filler}\n\n[Jetziges Leiden]\nRuhe."
    r = reduce_report_text_for_llm(combined, window_chars=1000, total_cap=8000)
    assert r.llm_text_reduction_method == METHOD_KEYWORD
    assert r.reduced_text.startswith("[Diagnosen]")
    assert "Delir" in r.reduced_text or "delir" in r.reduced_text.lower()
    assert "[Diagnosen]" in r.reduced_text


def test_long_text_keyword_extracts_window():
    """Hit far from start yields a bounded window, not the full middle section."""
    prefix = "B" * 1500
    hit = "Patient zeigt Desorientierung."
    suffix = "C" * 1500
    jl = f"[Jetziges Leiden]\n{prefix}{hit}{suffix}"
    r = reduce_report_text_for_llm(jl, window_chars=200, total_cap=8000)
    assert r.llm_text_reduction_method == METHOD_KEYWORD
    assert "Desorientierung" in r.reduced_text
    assert len(r.reduced_text) < len(jl)
    assert "[Jetziges Leiden]" in r.reduced_text


def test_long_text_no_keyword_fallback_truncation():
    """No keyword: fallback uses per-field caps (plain blob = sequential sections)."""
    # No section headers, no delir keywords — only one long string in diag path
    noise = "Z" * 12000
    r = reduce_report_text_for_llm(noise, total_cap=8000)
    assert r.llm_text_reduction_method == METHOD_FALLBACK
    assert len(r.reduced_text) <= 8000
    assert "[Diagnosen]" in r.reduced_text


def test_reduced_text_respects_total_cap():
    """Keyword path joins blocks then truncates to total_cap."""
    parts = []
    for i in range(20):
        parts.append(f"[Diagnosen]\nAbschnitt {i} mit delir Bezug.\n" + "x" * 600)
    huge = "\n\n".join(parts)
    r = reduce_report_text_for_llm(huge, window_chars=500, total_cap=3000)
    assert len(r.reduced_text) <= 3000


def test_section_labels_preserved_in_fallback_with_markers():
    """Structured fallback keeps standard section headings."""
    txt = (
        "[Diagnosen]\n"
        + "Nur Routine.\n\n"
        + "[Jetziges Leiden]\n"
        + "Stabil.\n\n"
        + "[Prozedere]\n"
        + "Weiter.\n\n"
        + "[Epikrise]\n"
        + "Kurz."
    )
    r = reduce_report_text_for_llm(txt, total_cap=8000)
    assert r.llm_text_reduction_method == METHOD_FALLBACK
    for label in ("[Diagnosen]", "[Jetziges Leiden]", "[Prozedere]", "[Epikrise]"):
        assert label in r.reduced_text


def test_delir_keyword_hits_count_nonzero_when_keywords_present():
    t = "[Diagnosen]\nDelir und Vigilanzminderung."
    r = reduce_report_text_for_llm(t)
    assert r.delir_keyword_hits_count >= 2


@pytest.mark.parametrize("cap", [500, 8000])
def test_original_and_reduced_lengths_metadata(cap):
    raw = "word " * 2000
    r = reduce_report_text_for_llm(raw, total_cap=cap)
    assert r.original_report_text_length == len(raw.strip())
    assert r.llm_report_text_length == len(r.reduced_text)
    assert r.llm_report_text_length <= cap
