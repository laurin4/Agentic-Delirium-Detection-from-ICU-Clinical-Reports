"""
Reduce patient report text before LLM calls (Agent 1 / Agent 2) to avoid huge prompts.

Preserves clinically relevant snippets via keyword windows or structured fallback truncation.
Does not affect baselines or evaluation CSVs—only what is sent to the LLM in the pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from src.preprocessing.delirium_hint_keywords import DELIRIUM_HINT_KEYWORDS

LOGGER = logging.getLogger(__name__)

# Priority order for clinical sections (highest first)
FIELD_ORDER: Tuple[str, ...] = ("diag", "jetziges_leiden", "prozedere", "epikrise")

SECTION_HEADINGS: Dict[str, str] = {
    "diag": "[Diagnosen]",
    "jetziges_leiden": "[Jetziges Leiden]",
    "prozedere": "[Prozedere]",
    "epikrise": "[Epikrise]",
}

# Keywords (same master list as LLM prefilter; case-insensitive substring match)
DELIR_KEYWORDS: Tuple[str, ...] = DELIRIUM_HINT_KEYWORDS

# Search explicit delirium wording first within [Diagnosen] for ordering / emphasis
EXPLICIT_DIAG_TERMS: Tuple[str, ...] = ("delir", "delirium", "delirant", "delirös")

WINDOW_CHARS_DEFAULT = 1000
TOTAL_CHAR_CAP_DEFAULT = 8000

FALLBACK_DIAG = 3000
FALLBACK_JETZ = 2000
FALLBACK_PROZ = 1000
FALLBACK_EPIK = 1000

METHOD_KEYWORD = "keyword_windows"
METHOD_FALLBACK = "fallback_truncation"

_HEADER_RE = re.compile(
    r"\[(Diagnosen|Jetziges Leiden|Prozedere|Epikrise)\]",
)


@dataclass
class ReportTextReductionResult:
    reduced_text: str
    llm_text_reduction_method: str
    original_report_text_length: int
    llm_report_text_length: int
    delir_keyword_hits_count: int


def count_delir_keyword_hits(text: str) -> int:
    """Count total case-insensitive substring occurrences of any keyword in *text*."""
    if not text:
        return 0
    lower = text.lower()
    total = 0
    for kw in DELIR_KEYWORDS:
        k = kw.lower()
        start = 0
        while True:
            i = lower.find(k, start)
            if i < 0:
                break
            total += 1
            start = i + max(1, len(k))
    return total


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _keyword_intervals(
    field_text: str, keywords: Sequence[str], window_before: int, window_after: int
) -> List[Tuple[int, int]]:
    if not field_text:
        return []
    lower = field_text.lower()
    intervals: List[Tuple[int, int]] = []
    n = len(field_text)
    for kw in keywords:
        klow = kw.lower()
        start = 0
        while True:
            i = lower.find(klow, start)
            if i < 0:
                break
            a = max(0, i - window_before)
            b = min(n, i + len(kw) + window_after)
            intervals.append((a, b))
            start = i + max(1, len(kw))
    return _merge_intervals(intervals)


def _intervals_to_text(field_text: str, intervals: List[Tuple[int, int]]) -> str:
    if not intervals:
        return ""
    parts: List[str] = []
    for a, b in intervals:
        parts.append(field_text[a:b].strip())
    return "\n\n...\n\n".join(p for p in parts if p)


def parse_report_sections_with_markers(combined: str) -> Tuple[Dict[str, str], bool]:
    """
    Split combined report_text into section bodies.

    Returns (sections_dict, had_section_markers). If no [Section] markers exist in *combined*,
    the full string is placed in *diag* and had_section_markers is False.
    """
    empty = {k: "" for k in FIELD_ORDER}
    if not combined or not str(combined).strip():
        return dict(empty), False

    text = str(combined)
    positions: List[Tuple[int, int, str]] = []
    for m in _HEADER_RE.finditer(text):
        label = m.group(1)
        positions.append((m.start(), m.end(), label))

    if not positions:
        out = dict(empty)
        out["diag"] = text.strip()
        return out, False

    label_to_key: Dict[str, str] = {
        "Diagnosen": "diag",
        "Jetziges Leiden": "jetziges_leiden",
        "Prozedere": "prozedere",
        "Epikrise": "epikrise",
    }
    chunks: Dict[str, List[str]] = {k: [] for k in FIELD_ORDER}

    for i, (start, end, label) in enumerate(positions):
        content_start = end
        content_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[content_start:content_end].strip()
        key = label_to_key.get(label)
        if key:
            chunks[key].append(body)

    return {k: "\n\n".join(chunks[k]) for k in FIELD_ORDER}, True


def parse_report_sections(combined: str) -> Dict[str, str]:
    """Backward-compatible wrapper: sections only."""
    sections, _ = parse_report_sections_with_markers(combined)
    return sections


def _ordered_keywords_for_field(field_key: str) -> Tuple[str, ...]:
    if field_key != "diag":
        return DELIR_KEYWORDS
    seen = set()
    out: List[str] = []
    for t in EXPLICIT_DIAG_TERMS:
        if t not in seen:
            seen.add(t)
            out.append(t)
    for k in DELIR_KEYWORDS:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return tuple(out)


def reduce_report_text_for_llm(
    report_text: str,
    window_chars: int = WINDOW_CHARS_DEFAULT,
    total_cap: int = TOTAL_CHAR_CAP_DEFAULT,
) -> ReportTextReductionResult:
    """
    Produce shortened text for LLM agents and metadata for the predictions CSV.

    Strategy:
    1. Parse sections; search keywords per field in priority order; extract merged windows.
    2. If any keyword hit: build labeled blocks; cap epikrise contribution; cap total length.
    3. Else: fallback truncation per field caps.
    """
    original = report_text if report_text is None else str(report_text)
    stripped = original.strip()
    orig_len = len(stripped)
    hits_count = count_delir_keyword_hits(stripped)

    if not stripped:
        return ReportTextReductionResult(
            reduced_text="",
            llm_text_reduction_method=METHOD_FALLBACK,
            original_report_text_length=orig_len,
            llm_report_text_length=0,
            delir_keyword_hits_count=hits_count,
        )

    sections, had_markers = parse_report_sections_with_markers(stripped)

    # Keyword path: any keyword in any section?
    any_keyword = False
    for key in FIELD_ORDER:
        t = sections.get(key, "")
        if not t:
            continue
        kw_order = _ordered_keywords_for_field(key)
        if _keyword_intervals(t, kw_order, window_chars, window_chars):
            any_keyword = True
            break

    if any_keyword:
        blocks: List[str] = []
        for key in FIELD_ORDER:
            field_body = sections.get(key, "") or ""
            if not field_body:
                continue
            kw_order = _ordered_keywords_for_field(key)
            intervals = _keyword_intervals(field_body, kw_order, window_chars, window_chars)
            if not intervals:
                continue
            body_out = _intervals_to_text(field_body, intervals)
            if not body_out:
                continue
            heading = SECTION_HEADINGS[key]
            if key == "epikrise" and len(body_out) > FALLBACK_EPIK:
                body_out = body_out[:FALLBACK_EPIK]
            block = f"{heading}\n{body_out}"
            blocks.append(block)

        if not blocks:
            reduced, method = _build_fallback_text(sections, stripped, had_markers)
        else:
            # Ensure explicit delir in diag is at top when present
            diag = sections.get("diag") or ""
            explicit_in_diag = False
            if diag:
                dlow = diag.lower()
                explicit_in_diag = any(
                    et.lower() in dlow for et in EXPLICIT_DIAG_TERMS
                )
            joined = "\n\n".join(blocks)
            if explicit_in_diag and blocks:
                diag_blocks = [b for b in blocks if b.startswith(SECTION_HEADINGS["diag"])]
                other_blocks = [b for b in blocks if not b.startswith(SECTION_HEADINGS["diag"])]
                if diag_blocks:
                    reordered = "\n\n".join(diag_blocks + other_blocks)
                    joined = reordered
            reduced = joined[:total_cap] if len(joined) > total_cap else joined
            method = METHOD_KEYWORD
    else:
        reduced, method = _build_fallback_text(sections, stripped, had_markers)

    reduced = reduced.strip()
    if len(reduced) > total_cap:
        reduced = reduced[:total_cap]

    LOGGER.info(
        "LLM report text reduction: original_len=%d reduced_len=%d method=%s keyword_hits_in_source=%d",
        orig_len,
        len(reduced),
        method,
        hits_count,
    )

    return ReportTextReductionResult(
        reduced_text=reduced,
        llm_text_reduction_method=method,
        original_report_text_length=orig_len,
        llm_report_text_length=len(reduced),
        delir_keyword_hits_count=hits_count,
    )


def _build_fallback_text(
    sections: Dict[str, str], full_stripped: str, had_markers: bool
) -> Tuple[str, str]:
    """Returns (text, METHOD_FALLBACK)."""
    if not had_markers:
        body = fallback_from_plain_text(full_stripped)
        if body:
            return body, METHOD_FALLBACK
    body = _fallback_body(sections)
    if body:
        return body, METHOD_FALLBACK
    return fallback_from_plain_text(full_stripped), METHOD_FALLBACK


def _fallback_body(sections: Dict[str, str]) -> str:
    """Build fallback text: diag 3000, jetziges 2000, prozedere 1000, epikrise 1000 max."""
    parts: List[str] = []
    diag = sections.get("diag") or ""
    if diag:
        parts.append(f"{SECTION_HEADINGS['diag']}\n{diag[:FALLBACK_DIAG]}")

    jl = sections.get("jetziges_leiden") or ""
    if jl:
        parts.append(f"{SECTION_HEADINGS['jetziges_leiden']}\n{jl[:FALLBACK_JETZ]}")

    proz = sections.get("prozedere") or ""
    if proz:
        parts.append(f"{SECTION_HEADINGS['prozedere']}\n{proz[:FALLBACK_PROZ]}")

    ep = sections.get("epikrise") or ""
    if ep:
        parts.append(f"{SECTION_HEADINGS['epikrise']}\n{ep[:FALLBACK_EPIK]}")

    if parts:
        return "\n\n".join(parts)

    # no sections parsed: strip was non-empty but parse put nothing — use whole text as diag block
    return ""


def fallback_from_plain_text(whole: str) -> str:
    """When parse_report_sections put everything in diag, apply fallback caps sequentially."""
    s = (whole or "").strip()
    if not s:
        return ""
    # simulate sections: all in diag only for truncation chunks
    parts: List[str] = []
    parts.append(f"{SECTION_HEADINGS['diag']}\n{s[:FALLBACK_DIAG]}")
    rest = s[FALLBACK_DIAG:]
    if rest:
        parts.append(f"{SECTION_HEADINGS['jetziges_leiden']}\n{rest[:FALLBACK_JETZ]}")
    rest = rest[FALLBACK_JETZ:]
    if rest:
        parts.append(f"{SECTION_HEADINGS['prozedere']}\n{rest[:FALLBACK_PROZ]}")
    rest = rest[FALLBACK_PROZ:]
    if rest:
        parts.append(f"{SECTION_HEADINGS['epikrise']}\n{rest[:FALLBACK_EPIK]}")
    return "\n\n".join(parts)
