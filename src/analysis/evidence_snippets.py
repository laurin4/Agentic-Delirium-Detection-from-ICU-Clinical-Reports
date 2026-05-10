"""
Interpretability helpers: short text windows around delirium-related keyword hits.

Does not alter prediction logic; intended for downstream review CSVs only.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.preprocessing.delirium_hint_keywords import DELIRIUM_HINT_KEYWORDS

# Mirrors berichte_mapper._SECTION_FIELDS headings used in stitched report_text
_SECTION_MARKER_LABELS: Tuple[Tuple[str, str], ...] = (
    ("[Diagnosen]", "Diagnosen"),
    ("[Epikrise]", "Epikrise"),
    ("[Jetziges Leiden]", "Jetziges Leiden"),
    ("[Prozedere]", "Prozedere"),
)


def _section_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return non-overlapping (start, end, label) spans for predefined section headings."""
    if not text:
        return []
    hits: List[Tuple[int, str]] = []
    for marker, lab in _SECTION_MARKER_LABELS:
        start = 0
        ml = len(marker)
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            hits.append((idx, lab))
            start = idx + ml
    if not hits:
        return [(0, len(text), "")]
    hits.sort(key=lambda x: x[0])
    spans: List[Tuple[int, int, str]] = []
    if hits[0][0] > 0:
        spans.append((0, hits[0][0], ""))
    for i, (pos, lab) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        spans.append((pos, end, lab))
    return spans


def _section_label_for_index(text: str, index: int) -> str:
    if index < 0 or index >= len(text):
        return ""
    for start, end, lab in _section_spans(text):
        if start <= index < end:
            return lab if lab else ""
    return ""


def extract_evidence_snippets(
    report_text: Optional[str],
    max_snippet_len: int = 250,
    separator: str = " || ",
    max_snippets: int = 12,
) -> str:
    """
    Build short snippets around each distinct keyword hit in report_text.

    Preserves section labels as a prefix when the match falls inside a known block.
    """
    if not report_text or not str(report_text).strip():
        return ""
    src = str(report_text)
    low = src.lower()
    snippets: List[str] = []
    seen_ranges: List[Tuple[int, int]] = []

    for kw in DELIRIUM_HINT_KEYWORDS:
        needle = kw.lower()
        search_from = 0
        nk = len(needle)
        while True:
            i = low.find(needle, search_from)
            if i < 0:
                break
            overlap = False
            for a, b in seen_ranges:
                if not (i + nk <= a or i >= b):
                    overlap = True
                    break
            if not overlap:
                half = max(20, max_snippet_len // 2 - len(kw) // 2)
                a = max(0, i - half)
                b = min(len(src), i + len(kw) + half)
                raw = src[a:b].replace("\n", " ").strip()
                lab = _section_label_for_index(src, i)
                prefix = f"[{lab}] " if lab else ""
                snip = (prefix + raw).strip()
                if len(snip) > max_snippet_len:
                    snip = snip[: max_snippet_len - 1] + "…"
                snippets.append(snip)
                seen_ranges.append((i, i + max(nk, 1)))

            search_from = i + max(1, nk)
            if len(snippets) >= max_snippets:
                return separator.join(snippets)

    return separator.join(snippets)
