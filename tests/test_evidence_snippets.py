"""Evidence snippet helpers (interpretability export only)."""

from src.analysis.evidence_snippets import compute_evidence_snippets_cell, extract_evidence_snippets


def test_extract_evidence_includes_section_label():
    text = "[Diagnosen]\nAlter Mann mit Delir nach OP.\n\n[Epikrise]\nstabil"
    s = extract_evidence_snippets(text, max_snippet_len=200)
    assert "Delir" in s or "delir" in s.lower()
    assert "[Diagnosen]" in s or "Diagnosen" in s


def test_extract_evidence_truncates_long_window():
    long_body = "x" * 400 + " delir " + "y" * 400
    text = "[Prozedere]\n" + long_body
    s = extract_evidence_snippets(text, max_snippet_len=80)
    assert len(s) <= 82  # ellipsis
    assert "delir" in s.lower()


def test_extract_empty_returns_blank():
    assert extract_evidence_snippets("") == ""
    assert extract_evidence_snippets(None) == ""


def test_separator_joins_multiple_hits():
    text = "[Diagnosen]\ndelirium vorhanden.\n\n[Jetziges Leiden]\njemand verwirrt"
    s = extract_evidence_snippets(text, separator=";;", max_snippet_len=120, max_snippets=5)
    assert ";;" in s


def test_compute_evidence_snippets_cell_empty_is_brackets():
    assert compute_evidence_snippets_cell("", "", "") == "[]"


def test_compute_evidence_snippets_from_signals_without_report_text():
    s = compute_evidence_snippets_cell("", "Desorientierung im Raum", "")
    assert "Treffer" in s
    assert "Desorientierung" in s
