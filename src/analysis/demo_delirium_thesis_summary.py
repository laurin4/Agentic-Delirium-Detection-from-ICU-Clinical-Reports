"""
Publication-quality case summaries for the bachelor thesis and presentation.

Reads anonymized v2 snapshots and renders concise, scientifically formatted
half-page summaries — not terminal walkthrough logs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.analysis.demo_delirium_snapshot import (
    normalize_demo_polarity,
    presentation_case_title,
)
from src.preprocessing.evidence_extraction import SECTION_DISPLAY

CASE_A_HEADING = "Case A — True Positive (Wahres Positiv)"
CASE_B_HEADING = "Case B — False Negative (Falsch Negativ)"

GUARDRAIL_LABELS: Dict[str, str] = {
    "direct_delir_positive": "Expliziter Delirnachweis in den Evidenz-Snippets",
    "delir_therapy_with_compatible_symptoms": "Delirtherapie mit kompatiblem Symptomkontext",
    "symptom_cluster_positive_review_needed": "Delir-kompatibles Symptomcluster (LLM positiv)",
    "symptom_cluster_with_alternative_review_needed": "Symptomcluster trotz alternativer Erklärung",
    "indirect_symptoms_positive_review_needed": "Indirekte Symptome mit LLM-positiver Bewertung",
    "isolated_indirect_not_positive": "Isolierte indirekte Symptome — nicht als Delir gewertet",
    "alternative_explanation_downgrade": "Alternative Erklärung ohne Delir-Cluster",
    "negated_delir_not_positive": "Negierter Delirhinweis",
    "prophylaxis_only_not_positive": "Nur Prophylaxe/Screening ohne Delirnachweis",
    "no_evidence_prefilter_skip": "Keine verwertbare Evidenz — LLM nicht aufgerufen",
    "llm_classification": "LLM-Signalstärke ohne expliziten Delirnachweis",
}

EVIDENCE_TYPE_PREFIX: Dict[str, str] = {
    "direct_delir": "",
    "indirect_symptom": "",
    "negation": "Negation: ",
    "prophylaxis_or_risk": "Prophylaxe/Risiko: ",
}

KEYWORD_LABELS: Dict[str, str] = {
    "delir": "Dokumentiertes Delir",
    "delirium": "Delirium",
    "delirtherapie": "Delirtherapie",
    "hypoaktives delir": "Hypoaktives Delir",
    "hyperaktives delir": "Hyperaktives Delir",
    "desorientierung": "Desorientierung",
    "verwirrtheit": "Verwirrtheit",
    "agitiert": "Agitation",
    "agitation": "Agitation",
    "unruhig": "Unruhe",
    "vigilanz": "Vigilanzminderung",
    "vigilanzschwankung": "Vigilanzschwankungen",
    "gcs": "GCS-Verschlechterung",
    "haloperidol": "Delirtherapie (Haloperidol)",
    "delirprophylaxe": "Delirprophylaxe",
    "kein delir": "Kein Delir (negiert)",
}


def _klasse_label(klasse: int) -> str:
    return "Delir" if int(klasse) == 1 else "Kein Delir"


def _guardrail_label(rule: str) -> str:
    key = str(rule or "").strip()
    return GUARDRAIL_LABELS.get(key, key.replace("_", " ").capitalize() if key else "—")


def _section_prose(section: str) -> str:
    return SECTION_DISPLAY.get(section, section.replace("_", " ").title())


def _sentences_from_report(report_text: str) -> List[str]:
    """Extract prose sentences; skip bare section headings."""
    sentences: List[str] = []
    for line in (report_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        for part in re.split(r"(?<=[.!?])\s+", stripped):
            part = part.strip()
            if len(part) >= 25:
                sentences.append(part)
    return sentences


def _score_sentence(sentence: str, keywords: Sequence[str]) -> int:
    sl = sentence.lower()
    score = 0
    for kw in keywords:
        k = str(kw or "").strip().lower()
        if k and k in sl:
            score += 4
    for term, pts in (
        ("delir", 5),
        ("desorient", 3),
        ("verwirr", 3),
        ("vigilanz", 3),
        ("agit", 2),
        ("unruh", 2),
        ("haloperidol", 3),
        ("cam-icu", 2),
    ):
        if term in sl:
            score += pts
    return score


def clinical_report_excerpt(
    snapshot: Dict[str, Any],
    *,
    min_sentences: int = 2,
    max_sentences: int = 4,
) -> List[str]:
    """Select the 2–4 most clinically relevant sentences from the report."""
    snippets = list((snapshot.get("extraction") or {}).get("evidence_snippets") or [])
    keywords = [str(s.get("keyword") or "") for s in snippets]
    sentences = _sentences_from_report(str(snapshot.get("report_text") or ""))
    if not sentences:
        return []

    scored = [( _score_sentence(s, keywords), s) for s in sentences]
    scored.sort(key=lambda x: (-x[0], sentences.index(x[1])))

    picked: List[str] = []
    seen: set[str] = set()
    for score, sent in scored:
        if score <= 0 and len(picked) >= min_sentences:
            break
        key = sent.lower()[:100]
        if key in seen:
            continue
        seen.add(key)
        picked.append(sent)
        if len(picked) >= max_sentences:
            break

    if len(picked) < min_sentences:
        for sent in sentences:
            key = sent.lower()[:100]
            if key in seen:
                continue
            seen.add(key)
            picked.append(sent)
            if len(picked) >= min_sentences:
                break
    return picked[:max_sentences]


def _evidence_phrase(snippet: Dict[str, Any]) -> str:
    et = str(snippet.get("evidence_type") or "")
    kw = str(snippet.get("keyword") or "").strip().lower()
    label = KEYWORD_LABELS.get(kw, kw.replace("_", " ").capitalize() if kw else "Klinischer Hinweis")
    prefix = EVIDENCE_TYPE_PREFIX.get(et, "")
    return f"{prefix}{label}".strip()


def rule_based_evidence_bullets(snapshot: Dict[str, Any]) -> List[str]:
    """Human-readable summary of rule-extracted evidence (deduplicated)."""
    snippets: List[Dict[str, Any]] = list((snapshot.get("extraction") or {}).get("evidence_snippets") or [])
    if not snippets:
        return ["Keine delir-relevanten Snippets extrahiert (Prefilter-Ebene)."]

    seen: set[str] = set()
    bullets: List[str] = []
    for snip in sorted(snippets, key=lambda s: int(s.get("priority") or 99)):
        phrase = _evidence_phrase(snip)
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        sec = _section_prose(str(snip.get("section") or ""))
        bullets.append(f"{phrase} ({sec})")
    return bullets


def _strip_llm_instructions(bundle: str) -> str:
    text = str(bundle or "").strip()
    if not text:
        return ""
    if "\n\nInstruction:" in text:
        text = text.split("\n\nInstruction:")[0].strip()
    if text.startswith("Patient report evidence snippets:"):
        text = text[len("Patient report evidence snippets:") :].strip()
    return text


def _dedupe_snippet_lines(bundle: str, max_blocks: int = 5) -> List[str]:
    """Unique snippet bodies for a compact evidence bundle display."""
    blocks: List[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"\n\n(?=\[)", bundle):
        chunk = chunk.strip()
        if not chunk:
            continue
        body = re.sub(r"^\[[^\]]+\]\s*", "", chunk.split("\n", 1)[-1] if "\n" in chunk else chunk)
        body = re.sub(r"\[[^\]]+\]\s*", "", body).strip()
        if len(body) < 12:
            continue
        key = body.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        header = chunk.split("\n", 1)[0].strip()
        blocks.append(f"{header}\n{body}" if "\n" not in chunk else chunk)
        if len(blocks) >= max_blocks:
            break
    return blocks


def _clinical_body_from_snippet(text: str) -> str:
    """Strip section labels from snippet text; keep clinical prose."""
    body = str(text or "")
    body = re.sub(r"\[[^\]]+\]\s*", "", body)
    return " ".join(body.split())


def evidence_bundle_for_llm(snapshot: Dict[str, Any]) -> str:
    """Condensed evidence forwarded to the LLM (no prompts or instructions)."""
    extraction = snapshot.get("extraction") or {}
    final = snapshot.get("final") or {}
    snippets: List[Dict[str, Any]] = list(extraction.get("evidence_snippets") or [])

    if final.get("llm_skipped_by_prefilter"):
        return "— (Kein Evidenz-Bündel: Prefilter — LLM nicht aufgerufen)"

    if snippets:
        lines: List[str] = []
        seen: set[str] = set()
        for snip in sorted(snippets, key=lambda s: int(s.get("priority") or 99)):
            body = _clinical_body_from_snippet(str(snip.get("text") or ""))
            if len(body) < 15:
                continue
            key = body.lower()[:100]
            if key in seen:
                continue
            seen.add(key)
            sec = _section_prose(str(snip.get("section") or ""))
            lines.append(f"• [{sec}] {body}")
            if len(lines) >= 5:
                break
        if lines:
            return "\n".join(lines)

    raw = str(
        snapshot.get("llm_input_text")
        or extraction.get("llm_report_text")
        or ""
    ).strip()
    bundle = _strip_llm_instructions(raw)
    if not bundle:
        return "—"
    blocks = _dedupe_snippet_lines(bundle, max_blocks=4)
    if blocks:
        cleaned = []
        for block in blocks:
            cleaned.append(_clinical_body_from_snippet(block))
        return "\n".join(f"• {line}" for line in cleaned if line)
    return bundle[:1600] + ("…" if len(bundle) > 1600 else "")


def llm_interpretation_bullets(snapshot: Dict[str, Any], *, max_bullets: int = 4) -> List[str]:
    """At most 3–4 bullets: reasoning, signal strength, conclusion."""
    final = snapshot.get("final") or {}
    agent2 = snapshot.get("agent2") or {}
    guard = snapshot.get("guardrails") or {}

    if final.get("llm_skipped_by_prefilter") or not agent2.get("ran"):
        reason = str(
            (snapshot.get("agent1") or {}).get("skip_reason")
            or final.get("decision_rule_applied")
            or "no_evidence_prefilter_skip"
        )
        return [
            "LLM-Interpretation entfiel: keine verwertbare Evidenz im Regel-Extraktionsschritt.",
            f"Guardrail: {_guardrail_label(reason)}.",
            f"Modellentscheid: {_klasse_label(int(final.get('klasse') or 0))}.",
        ][:max_bullets]

    parsed = agent2.get("parsed") or {}
    signal = str(parsed.get("signalstaerke") or guard.get("signalstaerke") or "niedrig")
    bullets: List[str] = []

    kontext = str(parsed.get("kontext") or guard.get("kontext") or "").strip()
    if kontext:
        bullets.append(kontext)

    begr = parsed.get("begruendung") or guard.get("begruendung") or []
    if isinstance(begr, list):
        for item in begr:
            text = str(item).strip()
            if text and text not in bullets:
                bullets.append(text)
    elif str(begr).strip():
        bullets.append(str(begr).strip())

    bullets.append(f"Signalstärke (LLM): {signal}.")

    if parsed.get("alternative_erklaerung") and len(bullets) < max_bullets:
        kw = parsed.get("alternative_erklaerung_keywords") or []
        if kw:
            bullets.append(f"Alternative Erklärung berücksichtigt: {', '.join(str(k) for k in kw)}.")
        else:
            bullets.append("Alternative klinische Erklärung als relevant eingestuft.")

    deduped: List[str] = []
    seen: set[str] = set()
    for b in bullets:
        key = b.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)
    return deduped[:max_bullets]


def final_decision_rows(snapshot: Dict[str, Any]) -> List[Tuple[str, str]]:
    final = snapshot.get("final") or {}
    guard = snapshot.get("guardrails") or {}
    ver = snapshot.get("verification") or {}
    klasse = int(final.get("klasse") or 0)
    gt = ver.get("manual_report_ground_truth")
    correct = ver.get("model_correct_vs_manual")

    if correct is True:
        verdict = "Korrekt"
    elif correct is False:
        verdict = "Inkorrekt"
    else:
        verdict = "—"

    rule = str(guard.get("decision_rule_applied") or final.get("decision_rule_applied") or "")
    return [
        ("Guardrail-Entscheid", _guardrail_label(rule)),
        ("Modellvorhersage", _klasse_label(klasse)),
        ("Manuelle Referenz", _klasse_label(int(gt)) if gt is not None else "—"),
        ("Bewertung", verdict),
    ]


def _case_heading(snapshot: Dict[str, Any]) -> str:
    pol = normalize_demo_polarity(str(snapshot.get("polarity") or ""))
    if pol == "false_negative":
        return CASE_B_HEADING
    ver = snapshot.get("verification") or {}
    final = snapshot.get("final") or {}
    if int(final.get("klasse") or 0) == 1 and ver.get("manual_report_ground_truth") == 1:
        return CASE_A_HEADING
    if int(final.get("klasse") or 0) == 0 and ver.get("manual_report_ground_truth") == 1:
        return CASE_B_HEADING
    return CASE_A_HEADING if pol == "positive" else CASE_B_HEADING


def render_thesis_case_summary_markdown(snapshot: Dict[str, Any]) -> str:
    """One case as thesis-ready Markdown (~half page)."""
    title = presentation_case_title(snapshot)
    heading = _case_heading(snapshot)
    case = snapshot.get("case") or {}
    bertyp = str(case.get("bertyp") or "").strip()

    lines: List[str] = [
        f"## {heading}",
        "",
        f"**{title}**" + (f" · {bertyp}" if bertyp else ""),
        "",
        "### 1. Klinischer Berichtsauszug",
        "",
    ]

    excerpts = clinical_report_excerpt(snapshot)
    for sent in excerpts:
        lines.append(f"> {sent}")
    if not excerpts:
        lines.append("> *(Kein Auszug verfügbar)*")

    lines.extend(["", "### 2. Regelbasierte Evidenzextraktion", ""])
    for item in rule_based_evidence_bullets(snapshot):
        lines.append(f"- {item}")

    lines.extend(["", "### 3. An das LLM übergebenes Evidenz-Bündel", ""])
    bundle = evidence_bundle_for_llm(snapshot)
    if bundle.startswith("—"):
        lines.append(bundle)
    else:
        for line in bundle.splitlines():
            lines.append(line)

    lines.extend(["", "### 4. LLM-Interpretation", ""])
    for item in llm_interpretation_bullets(snapshot):
        lines.append(f"- {item}")

    lines.extend(["", "### 5. Finale Entscheidung", "", "| Feld | Wert |", "|------|------|"])
    for label, value in final_decision_rows(snapshot):
        lines.append(f"| {label} | {value} |")
    lines.append("")
    return "\n".join(lines)


def render_thesis_case_summary_plain(snapshot: Dict[str, Any]) -> str:
    """Same content as Markdown, plain text for Word/LaTeX paste."""
    title = presentation_case_title(snapshot)
    heading = _case_heading(snapshot)
    case = snapshot.get("case") or {}
    bertyp = str(case.get("bertyp") or "").strip()

    parts: List[str] = [
        heading,
        "=" * len(heading),
        "",
        title + (f" · {bertyp}" if bertyp else ""),
        "",
        "1. Klinischer Berichtsauszug",
        "-" * 28,
        "",
    ]
    for sent in clinical_report_excerpt(snapshot):
        parts.append(f"  «{sent}»")
        parts.append("")
    parts.extend(["2. Regelbasierte Evidenzextraktion", "-" * 35, ""])
    for item in rule_based_evidence_bullets(snapshot):
        parts.append(f"  • {item}")
    parts.extend(["", "3. An das LLM übergebenes Evidenz-Bündel", "-" * 40, ""])
    bundle = evidence_bundle_for_llm(snapshot)
    for line in bundle.splitlines():
        parts.append(f"  {line}" if line.strip() else "")
    parts.extend(["", "4. LLM-Interpretation", "-" * 20, ""])
    for item in llm_interpretation_bullets(snapshot):
        parts.append(f"  • {item}")
    parts.extend(["", "5. Finale Entscheidung", "-" * 20, ""])
    for label, value in final_decision_rows(snapshot):
        parts.append(f"  {label:<22} {value}")
    parts.append("")
    return "\n".join(parts)


def render_combined_thesis_summaries_markdown(
    positive: Dict[str, Any],
    negative: Dict[str, Any],
) -> str:
    intro = (
        "# Pipeline-Fallbeispiele — Delir-Erkennung\n\n"
        "Zwei anonymisierte Validierungsfälle zur Illustration der mehrstufigen Pipeline "
        "(Regel-Extraktion → LLM-Interpretation → klinische Guardrails). "
        "Material für Ergebniskapitel und Präsentation.\n"
    )
    return "\n".join(
        [
            intro,
            render_thesis_case_summary_markdown(positive),
            "---\n",
            render_thesis_case_summary_markdown(negative),
        ]
    )


def render_combined_thesis_summaries_plain(
    positive: Dict[str, Any],
    negative: Dict[str, Any],
) -> str:
    sep = "\n" + ("═" * 72) + "\n\n"
    return (
        "Pipeline-Fallbeispiele — Delir-Erkennung\n"
        "═" * 72 + "\n\n"
        + render_thesis_case_summary_plain(positive)
        + sep
        + render_thesis_case_summary_plain(negative)
    )
