"""
Interpretability helpers: short text windows around delirium-related keyword hits.

Does not alter prediction logic; intended for downstream review CSVs only.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.preprocessing.delirium_hint_keywords import (
    DELIRIUM_HINT_KEYWORDS,
    haystack_contains_delirium_hint,
)

LOGGER = logging.getLogger(__name__)

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


def _truncate(s: str, max_len: int) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _snippets_from_delir_signale(delir_signale: str, max_each: int = 120, max_parts: int = 5) -> str:
    if not delir_signale or not str(delir_signale).strip():
        return ""
    parts = [p.strip() for p in str(delir_signale).split("|") if p.strip()]
    if not parts:
        return ""
    out: List[str] = []
    for p in parts[:max_parts]:
        out.append(_truncate(f"[Treffer] {p}", max_each))
    return " || ".join(out)


def _snippet_from_kontext(kontext: str, max_len: int = 250) -> str:
    if not kontext or not str(kontext).strip():
        return ""
    if not haystack_contains_delirium_hint(str(kontext)):
        return ""
    return _truncate(f"[Kontext] {kontext}", max_len)


def compute_evidence_snippets_cell(
    report_text: Optional[str],
    delir_signale: Optional[str] = None,
    kontext: Optional[str] = None,
    *,
    max_snippet_len: int = 250,
) -> str:
    """
    Single CSV cell: keyword windows from report text, else short signal/context fallbacks.

    Returns the literal ``[]`` when no interpretable evidence string was built.
    """
    from_text = extract_evidence_snippets(
        report_text or "",
        max_snippet_len=max_snippet_len,
        max_snippets=12,
    )
    if from_text:
        return from_text
    from_sig = _snippets_from_delir_signale(str(delir_signale or ""), max_each=min(120, max_snippet_len))
    if from_sig:
        return from_sig
    from_ctx = _snippet_from_kontext(str(kontext or ""), max_len=max_snippet_len)
    if from_ctx:
        return from_ctx
    return "[]"


def _patient_report_text_lookup() -> Dict[str, str]:
    """Best-effort map PatientenID -> report_text from Berichte.csv (paths.py)."""
    try:
        from src.preprocessing.berichte_mapper import build_patient_level_berichte_reports
    except ImportError as exc:  # pragma: no cover
        LOGGER.debug("Berichte mapper unavailable: %s", exc)
        return {}
    try:
        reports = build_patient_level_berichte_reports()
    except FileNotFoundError as exc:
        LOGGER.debug("Berichte CSV not available for evidence snippets: %s", exc)
        return {}
    if reports.empty:
        return {}
    out: Dict[str, str] = {}
    for _, row in reports.iterrows():
        pid = str(row.get("PatientenID", "") or "").strip()
        if not pid:
            continue
        txt = row.get("report_text")
        out[pid] = "" if txt is None or (isinstance(txt, float) and pd.isna(txt)) else str(txt)
    return out


def attach_evidence_snippets_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure column ``evidence_snippets`` exists; fill missing/empty from report text or signals.

    Preserves non-empty existing ``evidence_snippets`` values from upstream (e.g. pipeline CSV).
    """
    out = df.copy()
    if "evidence_snippets" not in out.columns:
        out["evidence_snippets"] = ""

    pid_text = _patient_report_text_lookup()

    def _cell(row: pd.Series) -> str:
        cur = row.get("evidence_snippets")
        if cur is not None and not (isinstance(cur, float) and pd.isna(cur)):
            s = str(cur).strip()
            if s and s.lower() != "nan" and s != "[]":
                return s
        pid = str(row.get("PatientenID", "") or "").strip()
        rt = row.get("report_text")
        if rt is None or (isinstance(rt, float) and pd.isna(rt)):
            text = ""
        else:
            text = str(rt)
        if not text.strip() and pid:
            text = pid_text.get(pid, "")
        ds = row.get("delir_signale")
        if ds is None or (isinstance(ds, float) and pd.isna(ds)):
            ds = ""
        kt = row.get("kontext")
        if kt is None or (isinstance(kt, float) and pd.isna(kt)):
            kt = ""
        return compute_evidence_snippets_cell(text, str(ds), str(kt))

    out["evidence_snippets"] = out.apply(_cell, axis=1)
    return out
