"""
Structured, rule-based delirium evidence extraction from full report text.

Produces bounded evidence for the LLM (not the full report) and metadata for CSV export.
Does not perform klasse classification — downstream agents + classify_delirium do that.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Keyword groups (longest phrases first within each list for safe matching) ---

DIRECT_DELIR: Tuple[str, ...] = (
    "hyperaktives delir",
    "hypoaktives delir",
    "delirium",
    "delirant",
    "delirös",
    "delir",
)

INDIRECT_SYMPTOM: Tuple[str, ...] = (
    "bewusstseinstrübung",
    "bewusstseinsstörung",
    "vigilanzminderung",
    "desorientierung",
    "verwirrtheit",
    "desorientiert",
    "soporös",
    "somnolent",
    "agitation",
    "agitiert",
    "unruhig",
    "vigilanz",
    "verwirrt",
)

NEGATION: Tuple[str, ...] = (
    "keine delirante symptomatik",
    "kein hinweis auf delir",
    "kein delirium",
    "delir ausgeschlossen",
    "nicht delirant",
    "ohne hinweis auf delir",
    "kein delir",
)

PROPHYLAXIS_OR_RISK: Tuple[str, ...] = (
    "delirprophylaxe",
    "delir-screening",
    "delirscreening",
    "delirrisiko",
    "risiko für delir",
    "risiko fuer delir",
    "delirprävention",
    "delirpraevention",
    "delir-prävention",
    "delir monitoring",
    "delir-monitoring",
)

SECTION_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("[Diagnosen]", "diag"),
    ("[Jetziges Leiden]", "jetziges_leiden"),
    ("[Epikrise]", "epikrise"),
    ("[Prozedere]", "prozedere"),
)

SECTION_PRIORITY: Dict[str, int] = {
    "diag": 1,
    "jetziges_leiden": 2,
    "epikrise": 3,
    "prozedere": 4,
    "unknown": 5,
}

SECTION_DISPLAY: Dict[str, str] = {
    "diag": "Diagnosen",
    "jetziges_leiden": "Jetziges Leiden",
    "epikrise": "Epikrise",
    "prozedere": "Prozedere",
    "unknown": "unknown",
}

EVIDENCE_TYPE_ORDER: Dict[str, int] = {
    "direct_delir": 1,
    "indirect_symptom": 2,
    "negation": 3,
    "prophylaxis_or_risk": 4,
}

METHOD_NO_EVIDENCE = "no_evidence_prefilter_skip"
METHOD_STRUCTURED = "structured_evidence_extraction"
METHOD_SHORT_REPORT_FULLTEXT = "short_report_no_evidence_fulltext"

SHORT_REPORT_BERTYPEN = ("Verlaufseintrag", "Verlegungsbericht", "Austrittsbericht")

LLM_INSTRUCTION_BLOCK = """Instruction:
Decide whether the evidence supports documented delirium.
Do not classify prophylaxis/risk/screening alone as delirium.
Do not classify negated/excluded delirium as delirium.
Indirect symptoms require clinical interpretation."""


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def send_short_reports_without_evidence_enabled() -> bool:
    """``SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM`` (default false)."""
    return _bool_env("SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM")


def short_report_char_threshold() -> int:
    """``SHORT_REPORT_CHAR_THRESHOLD`` (default 1000)."""
    raw = os.environ.get("SHORT_REPORT_CHAR_THRESHOLD", "1000").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1000


def should_send_short_report_without_evidence(
    report_text: str,
    bertyp: str,
    snippets: List[Dict[str, Any]],
    *,
    original_length: Optional[int] = None,
) -> bool:
    """
    Send full short report text to LLM when rule layer found no snippets but report is brief.

    Requires ``SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true`` and bertyp in
    Verlaufseintrag / Verlegungsbericht / Austrittsbericht.
    """
    from src.preprocessing.berichte_filters import normalize_bertyp

    if not send_short_reports_without_evidence_enabled():
        return False
    if llm_should_receive_evidence(snippets):
        return False
    bt = normalize_bertyp(bertyp)
    if bt not in SHORT_REPORT_BERTYPEN:
        return False
    length = original_length if original_length is not None else len(str(report_text or ""))
    return length <= short_report_char_threshold()


def apply_short_report_fulltext_to_evidence(
    evidence: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    """Override LLM input with capped full report for short-report fallback path."""
    max_llm = _max_llm_chars()
    body = str(report_text or "").strip()
    if len(body) > max_llm:
        body = body[: max_llm - 1] + "…"
    out = dict(evidence)
    out["llm_report_text"] = body
    out["llm_report_text_length"] = len(body)
    out["llm_text_reduction_method"] = METHOD_SHORT_REPORT_FULLTEXT
    return out


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(minimum, v)


def _window_sentences_default() -> int:
    raw = os.environ.get("EVIDENCE_WINDOW_SENTENCES", "").strip()
    if not raw:
        return 1
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _max_snippet_chars() -> int:
    return _int_env("EVIDENCE_MAX_SNIPPET_CHARS", 400, minimum=80)


def _max_snippets() -> int:
    return _int_env("EVIDENCE_MAX_SNIPPETS", 12, minimum=1)


def _max_llm_chars() -> int:
    return _int_env("EVIDENCE_MAX_LLM_CHARS", 8000, minimum=500)


def _flatten_keywords() -> List[Tuple[str, str]]:
    """(lowercase phrase, evidence_type) sorted by phrase length descending."""
    out: List[Tuple[str, str]] = []
    for phrase in DIRECT_DELIR:
        out.append((phrase.lower(), "direct_delir"))
    for phrase in INDIRECT_SYMPTOM:
        out.append((phrase.lower(), "indirect_symptom"))
    for phrase in NEGATION:
        out.append((phrase.lower(), "negation"))
    for phrase in PROPHYLAXIS_OR_RISK:
        out.append((phrase.lower(), "prophylaxis_or_risk"))
    out.sort(key=lambda x: len(x[0]), reverse=True)
    return out


def _section_for_index(section_ranges: List[Tuple[int, int, str]], idx: int) -> str:
    for start, end, sec in section_ranges:
        if start <= idx < end:
            return sec
    return "unknown"


def _build_section_ranges(text: str) -> List[Tuple[int, int, str]]:
    if not text:
        return [(0, 0, "unknown")]
    hits: List[Tuple[int, str]] = []
    for marker, lab in SECTION_MARKERS:
        pos = 0
        ml = len(marker)
        while True:
            i = text.find(marker, pos)
            if i < 0:
                break
            hits.append((i, lab))
            pos = i + ml
    if not hits:
        return [(0, len(text), "unknown")]
    hits.sort(key=lambda x: x[0])
    spans: List[Tuple[int, int, str]] = []
    if hits[0][0] > 0:
        spans.append((0, hits[0][0], "unknown"))
    for j, (pos, lab) in enumerate(hits):
        end = hits[j + 1][0] if j + 1 < len(hits) else len(text)
        spans.append((pos, end, lab))
    return spans


def _split_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Non-overlapping (start, end) spans for rough sentence units."""
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for p in parts:
        if not p:
            continue
        start = text.find(p, pos)
        if start < 0:
            start = pos
        end = start + len(p)
        spans.append((start, end))
        pos = end
    if not spans:
        return [(0, len(text))]
    return spans


def _sentence_idx_for_pos(sentence_spans: List[Tuple[int, int]], pos: int) -> int:
    for i, (a, b) in enumerate(sentence_spans):
        if a <= pos < b:
            return i
    return 0


def _window_text(
    text: str,
    sentence_spans: List[Tuple[int, int]],
    match_start: int,
    window_sentences: int,
    max_chars: int,
) -> str:
    si = _sentence_idx_for_pos(sentence_spans, match_start)
    lo = max(0, si - window_sentences)
    hi = min(len(sentence_spans) - 1, si + window_sentences)
    a = sentence_spans[lo][0]
    b = sentence_spans[hi][1]
    chunk = text[a:b].strip()
    chunk = re.sub(r"\s+", " ", chunk)
    if len(chunk) > max_chars:
        chunk = chunk[: max_chars - 1] + "…"
    return chunk


def _max_hits_per_keyword(etype: str) -> int:
    if etype == "prophylaxis_or_risk":
        return _int_env("EVIDENCE_MAX_HITS_PROPHYLAXIS", 2, minimum=1)
    return _int_env("EVIDENCE_MAX_HITS_PER_KEYWORD", 3, minimum=1)


def _cap_raw_matches(matches: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
    """Limit repeated identical keyword hits (e.g. many Delirprophylaxe lines)."""
    counts: Dict[Tuple[str, str], int] = {}
    out: List[Tuple[int, int, str, str]] = []
    for item in matches:
        phrase, etype = item[2], item[3]
        key = (etype, phrase.lower())
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > _max_hits_per_keyword(etype):
            continue
        out.append(item)
    return out


def _dedupe_snippets(snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for s in snippets:
        text_norm = re.sub(r"\s+", " ", (s.get("text") or "").strip().lower())[:200]
        key = (s.get("section"), s.get("evidence_type"), s.get("keyword"), text_norm)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _sort_snippets(snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(s: Dict[str, Any]) -> Tuple[int, int, str]:
        sec = str(s.get("section") or "unknown")
        sp = SECTION_PRIORITY.get(sec, 99)
        et = str(s.get("evidence_type") or "")
        eto = EVIDENCE_TYPE_ORDER.get(et, 99)
        return (sp, eto, str(s.get("text") or ""))

    return sorted(snippets, key=key)


def _assign_display_priorities(snippets: List[Dict[str, Any]]) -> None:
    for i, s in enumerate(snippets, start=1):
        s["priority"] = i


def _build_llm_report_text(snippets: List[Dict[str, Any]], max_total_chars: int) -> str:
    lines = ["Patient report evidence snippets:", ""]
    for s in snippets:
        sec = SECTION_DISPLAY.get(str(s.get("section")), str(s.get("section")))
        et = s.get("evidence_type")
        pr = s.get("priority", 0)
        lines.append(f"[{sec} | {et} | priority={pr}]")
        lines.append(str(s.get("text") or ""))
        lines.append("")
    lines.append(LLM_INSTRUCTION_BLOCK)
    body = "\n".join(lines)
    if len(body) <= max_total_chars:
        return body
    # Trim from bottom snippet blocks while keeping header + instruction
    keep = ["Patient report evidence snippets:", ""]
    tail = "\n\n" + LLM_INSTRUCTION_BLOCK
    budget = max_total_chars - len(tail) - len(keep[0]) - 2
    for s in snippets:
        sec = SECTION_DISPLAY.get(str(s.get("section")), str(s.get("section")))
        et = s.get("evidence_type")
        pr = s.get("priority", 0)
        block = f"[{sec} | {et} | priority={pr}]\n{str(s.get('text') or '')}\n"
        if sum(len(x) + 1 for x in keep) + len(block) > budget:
            break
        keep.append(block)
    return "\n".join(keep) + tail


def _flags_from_snippets(snippets: List[Dict[str, Any]]) -> Tuple[bool, bool, bool, bool]:
    types = {str(s.get("evidence_type")) for s in snippets}
    has_direct = "direct_delir" in types
    has_indirect = "indirect_symptom" in types
    has_neg = "negation" in types
    has_prophy = "prophylaxis_or_risk" in types
    has_prophy_only = has_prophy and not has_direct and not has_indirect
    return has_direct, has_indirect, has_neg, has_prophy_only


def llm_should_receive_evidence(snippets: List[Dict[str, Any]]) -> bool:
    """True if at least one snippet should be sent to the LLM (not negation-only)."""
    for s in snippets:
        et = str(s.get("evidence_type") or "")
        if et in ("direct_delir", "indirect_symptom", "prophylaxis_or_risk"):
            return True
    return False


def evidence_snippets_json_for_csv(snippets: List[Dict[str, Any]]) -> str:
    """Stable string for prediction / comparison CSV column."""
    if not snippets:
        return "[]"
    return json.dumps(snippets, ensure_ascii=False)


def extract_delirium_evidence(report_text: str) -> Dict[str, Any]:
    """
    Full-text rule scan → structured snippets + bounded ``llm_report_text`` for the LLM.

    Returns the dict shape described in project docs (see HANDOVER_SUMMARY.md).
    """
    src = str(report_text or "")
    original_len = len(src)
    max_snip = _max_snippets()
    max_snip_chars = _max_snippet_chars()
    win_sent = _window_sentences_default()
    max_llm = _max_llm_chars()

    if not src.strip():
        return {
            "original_report_text_length": original_len,
            "llm_report_text": "",
            "llm_report_text_length": 0,
            "llm_text_reduction_method": METHOD_NO_EVIDENCE,
            "evidence_snippets": [],
            "delir_keyword_hits_count": 0,
            "has_direct_delir_evidence": False,
            "has_indirect_delir_evidence": False,
            "has_negated_delir_evidence": False,
            "has_prophylaxis_or_risk_only": False,
        }

    low = src.lower()
    section_ranges = _build_section_ranges(src)
    keywords = _flatten_keywords()

    raw_matches: List[Tuple[int, int, str, str]] = []
    used: List[Tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        for u, v in used:
            if not (b <= u or a >= v):
                return True
        return False

    for phrase, etype in keywords:
        start = 0
        pl = len(phrase)
        while True:
            i = low.find(phrase, start)
            if i < 0:
                break
            end = i + pl
            if not overlaps(i, end):
                raw_matches.append((i, end, phrase, etype))
                used.append((i, end))
            start = i + max(1, pl)

    raw_matches.sort(key=lambda x: x[0])
    raw_matches = _cap_raw_matches(raw_matches)
    hit_count = len(raw_matches)

    global_sents = _split_sentence_spans(src)
    snippets: List[Dict[str, Any]] = []
    for start, end, phrase, etype in raw_matches:
        sec = _section_for_index(section_ranges, start)
        if not global_sents:
            tw = src[start:end]
            tw = re.sub(r"\s+", " ", tw).strip()
            if len(tw) > max_snip_chars:
                tw = tw[: max_snip_chars - 1] + "…"
        else:
            tw = _window_text(src, global_sents, start, win_sent, max_snip_chars)

        snippets.append(
            {
                "section": sec,
                "keyword": phrase,
                "evidence_type": etype,
                "priority": 0,
                "text": tw,
            }
        )

    snippets = _dedupe_snippets(snippets)
    snippets = _sort_snippets(snippets)[:max_snip]
    _assign_display_priorities(snippets)

    has_direct, has_indirect, has_neg, has_prophy_only = _flags_from_snippets(snippets)
    method = METHOD_STRUCTURED if llm_should_receive_evidence(snippets) else METHOD_NO_EVIDENCE
    llm_body = _build_llm_report_text(snippets, max_llm) if method == METHOD_STRUCTURED else ""

    return {
        "original_report_text_length": original_len,
        "llm_report_text": llm_body,
        "llm_report_text_length": len(llm_body),
        "llm_text_reduction_method": method,
        "evidence_snippets": snippets,
        "delir_keyword_hits_count": hit_count,
        "has_direct_delir_evidence": has_direct,
        "has_indirect_delir_evidence": has_indirect,
        "has_negated_delir_evidence": has_neg,
        "has_prophylaxis_or_risk_only": has_prophy_only,
    }
