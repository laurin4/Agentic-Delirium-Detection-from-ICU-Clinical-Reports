"""
Hemorrhage-style plain-text walkthrough export for presentation slides.

One .txt per case with numbered STEPs — copy into PowerPoint / Figma as needed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.analysis.demo_delirium_snapshot import presentation_case_subtitle, presentation_case_title

SEP = "=" * 76
THIN = "-" * 76
RULE = "—" * 44


def _indent_block(text: object, *, prefix: str = "    ", limit: int = 4000) -> str:
    body = "" if text is None else str(text).strip()
    if len(body) > limit:
        body = body[:limit] + f"\n… [truncated · {len(str(text)):,} chars total]"
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in body.splitlines() or [""])


def _step(n: int, title: str) -> str:
    return f"\n{SEP}\n  STEP {n}  ·  {title}\n{SEP}\n"


def _explain(text: str) -> str:
    return f"{text.strip()}\n"


def _split_report_sections(text: str) -> List[tuple[str, str]]:
    sections: List[tuple[str, str]] = []
    heading = ""
    lines: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if heading or lines:
                body = "\n".join(lines).strip()
                if heading or body:
                    sections.append((heading or "Text", body))
            heading = stripped.strip("[]")
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if heading or body:
        sections.append((heading or "Text", body))
    if not sections and (text or "").strip():
        sections.append(("Text", text.strip()))
    return sections


def _format_keywords(snippets: List[Dict[str, Any]]) -> str:
    by_type: Dict[str, List[str]] = {}
    for s in snippets:
        et = str(s.get("evidence_type") or "unknown")
        kw = str(s.get("keyword") or "").strip()
        if kw and kw not in by_type.get(et, []):
            by_type.setdefault(et, []).append(kw)
    if not by_type:
        return "    (no keywords detected)\n"
    lines = ["  Detected keywords:\n"]
    for et in ("direct_delir", "indirect_symptom", "negation", "prophylaxis_or_risk"):
        if et in by_type:
            lines.append(f"    {et}: {', '.join(by_type[et])}")
    return "\n".join(lines) + "\n"


def _format_snippets(snippets: List[Dict[str, Any]]) -> str:
    if not snippets:
        return "    (no snippets)\n"
    lines = ["  Evidence snippets:\n"]
    for i, s in enumerate(snippets, start=1):
        sec = str(s.get("section") or "")
        et = str(s.get("evidence_type") or "")
        kw = str(s.get("keyword") or "")
        lines.append(f"    {i}. [{et.upper()} | {sec} | {kw!r}]")
        lines.append(_indent_block(s.get("text"), prefix="       "))
        lines.append("")
    return "\n".join(lines)


def _format_agent1_signals(interp: Dict[str, Any]) -> str:
    signals = interp.get("delir_signale") or {}
    if not isinstance(signals, dict) or not any(signals.values()):
        return "    (no structured Agent 1 signals stored)\n"
    lines = ["  Agent 1 signals (structured extraction):\n"]
    for key, vals in signals.items():
        if vals:
            joined = ", ".join(str(v) for v in vals)
            lines.append(f"    · {key}: {joined}")
    return "\n".join(lines) + "\n"


def render_walkthrough_txt(snapshot: Dict[str, Any]) -> str:
    """Render one anonymized case as a hemorrhage-style step-by-step .txt walkthrough."""
    case = snapshot.get("case") or {}
    extraction = snapshot.get("extraction") or {}
    interp = snapshot.get("interpretation") or {}
    final = snapshot.get("final") or {}
    report_text = str(snapshot.get("report_text") or "")
    snippets: List[Dict[str, Any]] = list(extraction.get("evidence_snippets") or [])
    llm_skipped = bool(final.get("llm_skipped_by_prefilter"))
    klasse = int(final.get("klasse") or 0)
    polarity = "POSITIVE · Delir" if klasse == 1 else "NEGATIVE · kein Delir"

    parts: List[str] = [
        SEP,
        f"  {presentation_case_title(snapshot)}",
        f"  {presentation_case_subtitle(snapshot)}",
        f"  {polarity}",
        SEP,
        "",
        "Pipeline blueprint (same stages as hemorrhage demo):",
        "  Clinical reports  →  Rule extraction  →  LLM input  →  Interpretation  →  Guardrails  →  klasse",
        "",
    ]

    # STEP 1 — original reports
    parts.append(_step(1, "Original clinical reports"))
    parts.append(_explain(
        "The pipeline receives completely unstructured German clinical documentation."
    ))
    for heading, body in _split_report_sections(report_text):
        parts.append(f"  [{heading}]  ({len(body):,} chars)\n")
        parts.append(_indent_block(body, limit=3500))
        parts.append("")

    # STEP 2 — rule extraction
    parts.append(_step(2, "Rule-based evidence extraction"))
    parts.append(_explain(
        "Deterministic keyword scan across all report sections.\n"
        "Only clinically relevant snippets are kept — not the full report."
    ))
    parts.append(_format_keywords(snippets))
    parts.append(_format_snippets(snippets))
    orig_len = extraction.get("original_report_text_length", len(report_text))
    llm_len = extraction.get("llm_report_text_length", 0)
    method = extraction.get("llm_text_reduction_method", "")
    parts.append(f"  Text reduction: {orig_len:,} chars → {llm_len:,} chars LLM bundle ({method})\n")

    # STEP 3 — LLM input
    parts.append(_step(3, "Evidence bundle presented to the LLM"))
    if llm_skipped:
        parts.append(_explain(
            "No actionable evidence — the LLM is skipped entirely.\n"
            "This is the efficiency layer: many reports never reach the model."
        ))
        parts.append("    (prefilter skip — nothing forwarded to Agent 1 / Agent 2)\n")
    else:
        parts.append(_explain(
            "Structured snippet bundle + clinical instruction block.\n"
            "This is what Agent 1 and Agent 2 receive instead of the full report."
        ))
        parts.append(_indent_block(extraction.get("llm_report_text") or "", limit=5000))

    # STEP 4 — Agent 1 (only if LLM ran)
    if not llm_skipped:
        parts.append(_step(4, "Agent 1 — structured signal extraction"))
        parts.append(_explain(
            "Agent 1 maps the evidence bundle to structured delirium signal categories."
        ))
        parts.append(_format_agent1_signals(interp))

    # STEP 5 — Agent 2 interpretation OR skip branch
    step_n = 5 if not llm_skipped else 4
    if llm_skipped:
        parts.append(f"\n{SEP}\n  LLM SKIPPED (prefilter)\n{SEP}\n")
        parts.append(_explain(
            "Reason: No hemorrhage-style second stage — here the binary question does not\n"
            "reach the LLM at all when the rule layer finds no actionable delirium evidence."
        ))
        reason = str(interp.get("skipped_reason") or final.get("decision_rule_applied") or "prefilter")
        parts.append(f"    skipped_reason: {reason}\n")
    else:
        parts.append(_step(5, "Agent 2 — interpretation  →  signal strength"))
        parts.append(_explain(
            "Agent 2 assigns signal strength and clinical context — klasse is derived later."
        ))
        parts.append(f"    signalstaerke:          {interp.get('signalstaerke', '')}\n")
        prob = interp.get("delir_probability_estimate", "")
        if str(prob).strip() not in ("", "nan", "None"):
            parts.append(f"    probability estimate:   {prob}\n")
        if interp.get("kontext"):
            parts.append(f"    kontext:\n{_indent_block(interp.get('kontext'))}")
        if interp.get("begruendung"):
            parts.append(f"    begruendung:\n{_indent_block(interp.get('begruendung'))}")

    # STEP 6 — guardrails
    guard_step = 6 if not llm_skipped else 5
    parts.append(_step(guard_step, "Clinical guardrails  →  final klasse"))
    parts.append(_explain(
        "Deterministic post-LLM rules: direct delir → positive;\n"
        "prophylaxis-only / negation-only / no evidence → negative."
    ))
    parts.append(f"    decision_rule_applied:   {final.get('decision_rule_applied', '')}\n")
    parts.append(f"    manual_review_candidate: {final.get('manual_review_candidate', False)}\n")
    parts.append(f"    klasse:                  {klasse}  ({'delir' if klasse == 1 else 'kein_delir'})\n")

    # STEP 7 — validation
    val_step = guard_step + 1
    parts.append(_step(val_step, "Validation label"))
    parts.append(f"    manual_report_ground_truth: {case.get('manual_report_ground_truth')}\n")
    correct = (snapshot.get("verification") or {}).get("model_correct_vs_manual")
    if correct is True:
        parts.append("    Model vs manual label:      CORRECT ✓\n")
    elif correct is False:
        parts.append("    Model vs manual label:      MISMATCH ✗\n")

    # Final box (hemorrhage-style)
    parts.append(f"\n{SEP}\n  Final structured output\n{SEP}\n")
    parts.append(f"  {RULE}\n")
    parts.append("  Final Classification\n\n")
    parts.append(f"  Delirium detected:     {'YES' if klasse == 1 else 'NO'}\n")
    parts.append(f"  klasse:                {klasse}\n")
    parts.append(f"  signalstaerke:         {interp.get('signalstaerke') or '—'}\n")
    parts.append(f"  decision_rule:         {final.get('decision_rule_applied') or '—'}\n")
    parts.append(f"  {RULE}\n")

    return "\n".join(parts)


def render_combined_walkthrough_txt(positive: Dict[str, Any], negative: Dict[str, Any]) -> str:
    return (
        "DELIRIUM PIPELINE DEMO — PRESENTATION WALKTHROUGH\n"
        f"{SEP}\n\n"
        "CASE A — TRUE POSITIVE\n\n"
        f"{render_walkthrough_txt(positive)}\n\n\n"
        f"{SEP}\n"
        "CASE B — TRUE NEGATIVE\n\n"
        f"{render_walkthrough_txt(negative)}\n"
    )
