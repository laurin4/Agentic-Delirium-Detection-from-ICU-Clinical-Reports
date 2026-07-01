"""
Delirium pipeline demo — thesis-ready case summaries and optional walkthrough exports.

Primary output: publication-quality TP + FN case summaries for thesis and presentation.

Usage:
    python -m src.analysis.demo_delirium_case --thesis
    python -m src.analysis.demo_delirium_case --snapshot-positive --snapshot-false-negative
    python -m src.analysis.demo_delirium_case --thesis

See docs/demo/DEMO_GUIDE.md.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.demo_delirium_snapshot import (
    ensure_default_snapshots,
    diagnose_preferred_fn_patients,
    generate_snapshot_from_validation,
    load_snapshot,
    normalize_demo_polarity,
    presentation_case_subtitle,
    presentation_case_title,
    presentation_polarity_banner,
    rank_validation_candidates,
    snippet_section_label,
)
from src.analysis.demo_delirium_trace import (
    SYSTEM_PROMPT_EXCERPT_CHARS,
    TEXT_BLOCK_CHARS,
    trace_is_v2,
)
from src.analysis.demo_delirium_walkthrough import (
    render_combined_walkthrough_txt,
    render_walkthrough_txt,
)
from src.analysis.demo_delirium_thesis_summary import (
    render_combined_thesis_summaries_markdown,
    render_combined_thesis_summaries_plain,
    render_thesis_case_summary_markdown,
    render_thesis_case_summary_plain,
)
from src.pipeline.paths import (
    DEMO_HTML_OUTPUT_DIR,
    DEMO_NEGATIVE_SNAPSHOT_PATH,
    DEMO_POSITIVE_SNAPSHOT_PATH,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)

SEP = "=" * 72
THIN = "-" * 72
RULE = "—" * 40

EVIDENCE_COLORS = {
    "direct_delir": "\033[1;32m",       # bold green
    "indirect_symptom": "\033[33m",     # yellow
    "negation": "\033[31m",             # red
    "prophylaxis_or_risk": "\033[2m",   # dim
}
RESET = "\033[0m"

EVIDENCE_BADGE = {
    "direct_delir": "DIRECT",
    "indirect_symptom": "INDIRECT",
    "negation": "NEGATION",
    "prophylaxis_or_risk": "PROPHYLAXIS",
}


def _use_color() -> bool:
    return sys.stdout.isatty()


def _etype_style(evidence_type: str) -> str:
    if not _use_color():
        return ""
    return EVIDENCE_COLORS.get(evidence_type, "")


def _pause(enabled: bool, *, last: bool = False) -> None:
    if not enabled:
        return
    label = "Press ENTER to finish…" if last else "Press ENTER to continue…"
    try:
        input(f"\n{label} ")
    except EOFError:
        pass


def _step(n: int, title: str) -> None:
    print(f"\n{SEP}")
    print(f"  STEP {n}  ·  {title}")
    print(SEP)


def _explain(text: str) -> None:
    print(f"\n{text}\n")


def _block(text: object, *, limit: Optional[int] = 1200, indent: str = "    ") -> None:
    body = "" if text is None else str(text)
    if limit and len(body) > limit:
        body = body[:limit] + f"\n… [truncated · {len(str(text)):,} chars total]"
    for line in body.splitlines() or [""]:
        print(f"{indent}{line}")


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


def _render_snippet(snippet: Dict[str, Any], index: int) -> None:
    et = str(snippet.get("evidence_type") or "unknown")
    badge = EVIDENCE_BADGE.get(et, et.upper())
    sec = snippet_section_label(snippet)
    kw = str(snippet.get("keyword") or "")
    style = _etype_style(et)
    reset = RESET if style else ""
    print(f"\n    {index}. [{badge}]  section={sec}  keyword={kw!r}")
    _block(str(snippet.get("text") or ""), limit=600, indent=f"    {style}")
    if style:
        print(reset, end="")


def _render_keywords(snippets: List[Dict[str, Any]]) -> None:
    by_type: Dict[str, List[str]] = {}
    for s in snippets:
        et = str(s.get("evidence_type") or "unknown")
        kw = str(s.get("keyword") or "").strip()
        if kw and kw not in by_type.get(et, []):
            by_type.setdefault(et, []).append(kw)
    if not by_type:
        print("    (no keywords detected)")
        return
    for et in ("direct_delir", "indirect_symptom", "negation", "prophylaxis_or_risk"):
        if et in by_type:
            style = _etype_style(et)
            reset = RESET if style else ""
            print(f"    {style}{et}{reset}: {', '.join(by_type[et])}")


def _render_interpretation(interp: Dict[str, Any], *, llm_skipped: bool) -> None:
    if llm_skipped:
        reason = str(interp.get("skipped_reason") or "no_evidence_prefilter_skip")
        print(f"    LLM not called — {reason}")
        return
    sig = str(interp.get("signalstaerke") or "")
    prob = interp.get("delir_probability_estimate", "")
    print(f"    signalstaerke:     {sig}")
    if str(prob).strip() not in ("", "nan"):
        print(f"    probability est.:  {prob}")
    kontext = str(interp.get("kontext") or "").strip()
    if kontext:
        print(f"    kontext:           {kontext}")
    begr = str(interp.get("begruendung") or "").strip()
    if begr:
        print(f"    begruendung:       {begr}")
    signals = interp.get("delir_signale") or {}
    if isinstance(signals, dict) and any(signals.values()):
        print("\n    Agent 1 signals:")
        for key, vals in signals.items():
            if vals:
                joined = ", ".join(str(v) for v in vals)
                print(f"      · {key}: {joined}")


def _excerpt(text: str, limit: Optional[int]) -> str:
    s = str(text or "")
    if limit and len(s) > limit:
        return s[:limit] + f"\n… [truncated · full prompt has {len(s):,} chars]"
    return s


def _collapse_evidence(user_prompt: str, input_text: str) -> str:
    if input_text and input_text in user_prompt:
        return user_prompt.replace(input_text, "‹[evidence bundle — shown in STEP 2]›")
    return user_prompt


def _render_agent_parsed(stage: Dict[str, Any], *, label: str) -> None:
    parsed = stage.get("parsed") or {}
    print("\n  [PARSED & VALIDATED]")
    if label == "Agent 1":
        for key in (
            "desorientierung",
            "delir_explizit",
            "hyperaktivitaet_agitation",
            "vigilanz",
            "delir_therapie",
            "delir_prophylaxe",
        ):
            vals = parsed.get(key) or []
            if vals:
                print(f"    {key}: {', '.join(str(v) for v in vals)}")
        if not any(parsed.get(k) for k in parsed):
            print("    (no signals)")
    else:
        print(f"    signalstaerke:  {parsed.get('signalstaerke', '')}")
        print(f"    kontext:        {parsed.get('kontext', '')}")
        begr = parsed.get("begruendung") or []
        if begr:
            print(f"    begruendung:    {' | '.join(str(b) for b in begr)}")
    note = stage.get("replay_note")
    if note:
        print(f"\n    Note: {note}")


def present_snapshot(snapshot: Dict[str, Any], *, pause: bool = True, full: bool = False) -> None:
    """Hemorrhage-style paced terminal walkthrough."""
    if not trace_is_v2(snapshot):
        print("[!] Snapshot format outdated — re-run --snapshot-positive / --snapshot-false-negative")
        return

    text_limit = None if full else TEXT_BLOCK_CHARS
    sys_limit = None if full else SYSTEM_PROMPT_EXCERPT_CHARS

    extraction = snapshot.get("extraction") or {}
    final = snapshot.get("final") or {}
    guard = snapshot.get("guardrails") or {}
    agent1 = snapshot.get("agent1") or {}
    agent2 = snapshot.get("agent2") or {}
    report_text = str(snapshot.get("report_text") or "")
    llm_input = str(snapshot.get("llm_input_text") or extraction.get("llm_report_text") or "")
    snippets: List[Dict[str, Any]] = list(extraction.get("evidence_snippets") or [])
    llm_skipped = bool(final.get("llm_skipped_by_prefilter"))
    klasse = int(final.get("klasse") or 0)
    polarity_banner = presentation_polarity_banner(snapshot)
    mode = snapshot.get("mode", "")

    print(f"\n{SEP}")
    print(f"  {presentation_case_title(snapshot)}")
    sub = presentation_case_subtitle(snapshot)
    if sub:
        print(f"  {sub}")
    print(f"  {polarity_banner}")
    if mode:
        print(f"  capture mode: {mode}")
    print(SEP)

    _step(1, "Original clinical reports")
    _explain("The pipeline receives completely unstructured German clinical documentation.")
    for heading, body in _split_report_sections(report_text):
        print(f"  [{heading}]  ({len(body):,} chars)")
        _block(body, limit=text_limit, indent="      ")
        print()
    _pause(pause)

    _step(2, "Rule-based evidence extraction")
    _explain(
        "Deterministic keyword scan — only clinically relevant snippets are forwarded.\n"
        "(Delirium-specific layer; hemorrhage sends full report text instead.)"
    )
    _render_keywords(snippets)
    print(f"\n  Snippets extracted: {len(snippets)}")
    for i, snip in enumerate(snippets[:6], start=1):
        _render_snippet(snip, i)
    orig_len = extraction.get("original_report_text_length", len(report_text))
    llm_len = extraction.get("llm_report_text_length", 0)
    print(
        f"\n  Reduction: {orig_len:,} chars → {llm_len:,} chars evidence bundle "
        f"({extraction.get('llm_text_reduction_method', '')})"
    )
    if llm_input and not llm_skipped:
        print("\n  [EVIDENCE BUNDLE — input to Agents 1 & 2]")
        _block(llm_input, limit=text_limit)
    _pause(pause)

    if llm_skipped or not agent1.get("ran"):
        print(f"\n{SEP}")
        print("  LLM SKIPPED (prefilter)")
        print(SEP)
        _explain(
            "No actionable evidence — Agents 1 and 2 are not called.\n"
            "Same idea as hemorrhage skipping Stage 2 when no hemorrhage is found."
        )
        print(f"    reason: {agent1.get('skip_reason') or final.get('decision_rule_applied')}")
        _pause(pause)
    else:
        _step(3, "Agent 1 prompt — structured signal extraction")
        _explain("Engineered rules + schema: map evidence snippets to delirium signal categories.")
        print("  [SYSTEM PROMPT — excerpt]")
        _block(_excerpt(agent1.get("system_prompt", ""), sys_limit))
        print("\n  [USER PROMPT]")
        _block(_collapse_evidence(agent1.get("user_prompt", ""), llm_input), limit=text_limit)
        _pause(pause)

        _step(4, "Agent 1 — real LLM response  →  validated JSON")
        print("  [RAW LLM RESPONSE]")
        _block(agent1.get("raw_response", ""))
        _render_agent_parsed(agent1, label="Agent 1")
        _pause(pause)

        _step(5, "Agent 2 prompt — interpretation / signal strength")
        _explain("Agent 2 reads the evidence bundle plus Agent 1 JSON; outputs signalstaerke.")
        print("  [SYSTEM PROMPT — excerpt]")
        _block(_excerpt(agent2.get("system_prompt", ""), sys_limit))
        print("\n  [USER PROMPT]")
        _block(_collapse_evidence(agent2.get("user_prompt", ""), llm_input), limit=text_limit)
        _pause(pause)

        _step(6, "Agent 2 — real LLM response  →  validated JSON")
        print("  [RAW LLM RESPONSE]")
        _block(agent2.get("raw_response", ""))
        _render_agent_parsed(agent2, label="Agent 2")
        _pause(pause)

    guard_step = 5 if llm_skipped else 7
    _step(guard_step, "Clinical guardrails  →  final klasse")
    _explain("Deterministic post-LLM rules enforce binary klasse and decision_rule_applied.")
    print(f"    decision_rule_applied:   {guard.get('decision_rule_applied', final.get('decision_rule_applied'))}")
    print(f"    manual_review_candidate: {guard.get('manual_review_candidate', final.get('manual_review_candidate'))}")
    print(f"    signalstaerke:           {guard.get('signalstaerke', final.get('signalstaerke'))}")
    print(f"    klasse:                  {klasse}  ({'delir' if klasse == 1 else 'kein_delir'})")
    _pause(pause)

    _step(guard_step + 1, "Validation label")
    ver = snapshot.get("verification") or {}
    print(f"    manual_report_ground_truth: {ver.get('manual_report_ground_truth')}")
    if ver.get("model_correct_vs_manual") is True:
        print("    Model vs manual label:      CORRECT ✓")
    elif ver.get("model_correct_vs_manual") is False:
        print("    Model vs manual label:      MISMATCH ✗")

    print(f"\n  {RULE}")
    print("  Final Classification\n")
    print(f"  Delirium detected:     {'YES' if klasse == 1 else 'NO'}")
    print(f"  klasse:                {klasse}")
    print(f"  signalstaerke:         {guard.get('signalstaerke', final.get('signalstaerke')) or '—'}")
    print(f"  decision_rule:         {guard.get('decision_rule_applied', final.get('decision_rule_applied')) or '—'}")
    print(f"  {RULE}")

    _explain("Pipeline summary:")
    for i, stage in enumerate(
        [
            "Clinical reports (unstructured German free-text)",
            "Rule-based evidence extraction (delirium-specific)",
            "Agent 1 — structured signals (LLM)",
            "Agent 2 — signal strength (LLM)",
            "Clinical guardrails → klasse 0/1",
            "Validation vs manual ground truth",
        ]
    ):
        print(f"      {stage}")
        if i < 5:
            print("                       ↓")
    _pause(pause, last=True)


# --------------------------------------------------------------------------- #
# HTML export (open in browser for slides)
# --------------------------------------------------------------------------- #
def _esc(text: object) -> str:
    return html.escape("" if text is None else str(text))


def _html_snippet_cards(snippets: List[Dict[str, Any]]) -> str:
    if not snippets:
        return '<p class="muted">No evidence snippets — prefilter skip.</p>'
    cards = []
    for s in snippets[:8]:
        et = str(s.get("evidence_type") or "unknown")
        cards.append(
            f'<div class="snippet {et}">'
            f'<div class="snippet-meta"><span class="badge {et}">{_esc(EVIDENCE_BADGE.get(et, et))}</span>'
            f' {_esc(snippet_section_label(s))} · keyword: <em>{_esc(s.get("keyword"))}</em></div>'
            f'<pre>{_esc(s.get("text"))}</pre></div>'
        )
    return "\n".join(cards)


def render_demo_html(positive: Dict[str, Any], negative: Dict[str, Any]) -> str:
    """Self-contained HTML with both cases for presentation."""

    def _agent_summary(snap: Dict[str, Any]) -> str:
        if not trace_is_v2(snap):
            return ""
        agent1 = snap.get("agent1") or {}
        agent2 = snap.get("agent2") or {}
        if not agent1.get("ran"):
            return '<p class="muted">Agents skipped (prefilter).</p>'
        a1 = agent1.get("parsed") or {}
        a2 = agent2.get("parsed") or {}
        return (
            f'<ul>'
            f'<li><strong>Agent 1:</strong> {len([k for k, v in (a1 or {}).items() if v])} signal categories</li>'
            f'<li><strong>Agent 2 signalstaerke:</strong> {_esc(a2.get("signalstaerke"))}</li>'
            f'<li><strong>capture:</strong> {_esc(snap.get("mode", "replay"))}</li>'
            f'</ul>'
        )

    def case_block(snap: Dict[str, Any]) -> str:
        extraction = snap.get("extraction") or {}
        guard = snap.get("guardrails") or {}
        agent2 = snap.get("agent2") or {}
        interp = snap.get("interpretation") or {}
        if trace_is_v2(snap):
            parsed2 = agent2.get("parsed") or {}
            signalstaerke = parsed2.get("signalstaerke") or guard.get("signalstaerke")
            kontext = parsed2.get("kontext") or guard.get("kontext")
            begruendung = parsed2.get("begruendung") or guard.get("begruendung")
        else:
            signalstaerke = interp.get("signalstaerke")
            kontext = interp.get("kontext")
            begruendung = interp.get("begruendung")
        final = snap.get("final") or {}
        snippets = list(extraction.get("evidence_snippets") or [])
        llm_skipped = bool(final.get("llm_skipped_by_prefilter"))
        klasse = int(final.get("klasse") or 0)
        llm_section = (
            '<p class="muted">LLM skipped — no actionable evidence.</p>'
            if llm_skipped
            else f'<pre class="llm-input">{_esc(extraction.get("llm_report_text"))}</pre>'
        )
        interp_section = (
            f'<p class="muted">Skipped ({_esc(interp.get("skipped_reason", "prefilter"))})</p>'
            if llm_skipped
            else (
                f'<ul>'
                f'<li><strong>signalstaerke:</strong> {_esc(signalstaerke)}</li>'
                f'<li><strong>kontext:</strong> {_esc(kontext)}</li>'
                f'<li><strong>begruendung:</strong> {_esc(begruendung)}</li>'
                f'</ul>'
                f'{_agent_summary(snap)}'
            )
        )
        verdict = "Delir (klasse=1)" if klasse == 1 else "Kein Delir (klasse=0)"
        verdict_class = "pos" if klasse == 1 else "neg"
        title = presentation_case_title(snap)
        subtitle = presentation_case_subtitle(snap)
        return f"""
<section class="case">
  <h2>{_esc(title)}</h2>
  <p class="case-id">{_esc(subtitle)}</p>
  <div class="pipeline">
    <div class="stage">
      <h3>1 · Original report</h3>
      <pre class="report">{_esc(snap.get("report_text"))}</pre>
    </div>
    <div class="arrow">→</div>
    <div class="stage">
      <h3>2 · Rule extraction</h3>
      {_html_snippet_cards(snippets)}
    </div>
    <div class="arrow">→</div>
    <div class="stage">
      <h3>3 · LLM input</h3>
      {llm_section}
    </div>
    <div class="arrow">→</div>
    <div class="stage">
      <h3>4 · Interpretation</h3>
      {interp_section}
    </div>
    <div class="arrow">→</div>
    <div class="stage final {verdict_class}">
      <h3>5 · Decision</h3>
      <p class="verdict">{_esc(verdict)}</p>
      <p><code>{_esc(final.get("decision_rule_applied"))}</code></p>
    </div>
  </div>
</section>"""

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8"/>
<title>Delirium Pipeline Demo</title>
<style>
  :root {{
    --direct: #1a7f37; --indirect: #9a6700; --negation: #cf222e; --prophy: #656d76;
    --bg: #f6f8fa; --card: #fff; --border: #d0d7de;
  }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 2rem; background: var(--bg); color: #1f2328; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .subtitle {{ color: #656d76; margin-bottom: 2rem; }}
  .legend {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; font-size: 0.9rem; }}
  .legend span {{ padding: 0.2rem 0.6rem; border-radius: 4px; color: #fff; }}
  .legend .direct_delir {{ background: var(--direct); }}
  .legend .indirect_symptom {{ background: var(--indirect); }}
  .legend .negation {{ background: var(--negation); }}
  .legend .prophylaxis_or_risk {{ background: var(--prophy); }}
  .case {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem; }}
  .case-id {{ color: #656d76; font-size: 0.9rem; }}
  .pipeline {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.5rem; margin-top: 1rem; }}
  .stage {{ flex: 1 1 200px; min-width: 180px; border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem; background: #fafbfc; }}
  .stage.final.pos {{ border-color: var(--direct); background: #dafbe1; }}
  .stage.final.neg {{ border-color: #0969da; background: #ddf4ff; }}
  .arrow {{ align-self: center; font-size: 1.5rem; color: #656d76; }}
  h3 {{ margin: 0 0 0.5rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em; color: #656d76; }}
  pre {{ white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; margin: 0; max-height: 220px; overflow: auto; }}
  .snippet {{ border-left: 4px solid #656d76; padding-left: 0.5rem; margin-bottom: 0.75rem; }}
  .snippet.direct_delir {{ border-color: var(--direct); }}
  .snippet.indirect_symptom {{ border-color: var(--indirect); }}
  .snippet.negation {{ border-color: var(--negation); }}
  .snippet.prophylaxis_or_risk {{ border-color: var(--prophy); }}
  .badge {{ font-size: 0.7rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 3px; color: #fff; }}
  .badge.direct_delir {{ background: var(--direct); }}
  .badge.indirect_symptom {{ background: var(--indirect); }}
  .badge.negation {{ background: var(--negation); }}
  .badge.prophylaxis_or_risk {{ background: var(--prophy); }}
  .snippet-meta {{ font-size: 0.75rem; margin-bottom: 0.25rem; }}
  .verdict {{ font-size: 1.1rem; font-weight: 700; margin: 0.25rem 0; }}
  .muted {{ color: #656d76; font-style: italic; }}
  @media (max-width: 900px) {{ .arrow {{ display: none; }} }}
</style>
</head>
<body>
<h1>Delirium Detection Pipeline</h1>
<p class="subtitle">Report → rule extraction → LLM interpretation → guardrails → klasse</p>
<div class="legend">
  <span class="direct_delir">direct_delir</span>
  <span class="indirect_symptom">indirect_symptom</span>
  <span class="negation">negation</span>
  <span class="prophylaxis_or_risk">prophylaxis_or_risk</span>
</div>
{case_block(positive)}
{case_block(negative)}
</body>
</html>"""


def export_demo_html(
    positive_path: Path = DEMO_POSITIVE_SNAPSHOT_PATH,
    negative_path: Path = DEMO_NEGATIVE_SNAPSHOT_PATH,
    output_path: Optional[Path] = None,
) -> Path:
    ensure_default_snapshots()
    pos = load_snapshot(positive_path)
    neg = load_snapshot(negative_path)
    out = output_path or (DEMO_HTML_OUTPUT_DIR / "delirium_pipeline_demo.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_demo_html(pos, neg), encoding="utf-8")
    return out


def _wrap_text(text: str, width: int = 42) -> str:
    words = str(text or "").split()
    if not words:
        return ""
    lines: List[str] = []
    line: List[str] = []
    for word in words:
        candidate = (" ".join(line + [word])).strip()
        if line and len(candidate) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def export_case_png(snapshot: Dict[str, Any], output_path: Path) -> Path:
    """Render one anonymized case as a 16:9 PNG for PowerPoint."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for PNG export. Install with: pip install matplotlib"
        ) from exc

    case = snapshot.get("case") or {}
    extraction = snapshot.get("extraction") or {}
    interp = snapshot.get("interpretation") or {}
    final = snapshot.get("final") or {}
    snippets = list(extraction.get("evidence_snippets") or [])[:4]
    llm_skipped = bool(final.get("llm_skipped_by_prefilter"))
    klasse = int(final.get("klasse") or 0)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = presentation_case_title(snapshot)
    subtitle = presentation_case_subtitle(snapshot)
    ax.text(0.03, 0.96, title, fontsize=18, fontweight="bold", va="top")
    ax.text(0.03, 0.92, subtitle, fontsize=11, color="#555555", va="top")

    stages = [
        ("1 · Report", _wrap_text(snapshot.get("report_text") or "", 38)[:420]),
        (
            "2 · Rule snippets",
            "\n".join(
                f"• [{s.get('evidence_type')}] {s.get('keyword')}"
                for s in snippets
            )
            or "— keine Snippets —",
        ),
        (
            "3 · LLM input",
            "— Prefilter skip —" if llm_skipped else _wrap_text(extraction.get("llm_report_text") or "", 38)[:320],
        ),
        (
            "4 · Interpretation",
            (
                f"signalstaerke: {interp.get('signalstaerke', '')}\n"
                f"{_wrap_text(interp.get('kontext') or '', 36)[:200]}"
            )
            if not llm_skipped
            else "LLM nicht aufgerufen",
        ),
        (
            "5 · Decision",
            f"klasse = {klasse}\n{final.get('decision_rule_applied', '')}",
        ),
    ]

    x0, w, h, gap = 0.03, 0.17, 0.62, 0.015
    y0 = 0.12
    for i, (label, body) in enumerate(stages):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch(
            (x, y0),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=1,
            edgecolor="#d0d7de",
            facecolor="#dafbe1" if i == 4 and klasse == 1 else ("#ddf4ff" if i == 4 else "#f6f8fa"),
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x + 0.01, y0 + h - 0.03, label, fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(x + 0.01, y0 + h - 0.07, body, fontsize=7.2, va="top", family="monospace", transform=ax.transAxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def export_demo_png(
    positive_path: Path = DEMO_POSITIVE_SNAPSHOT_PATH,
    negative_path: Path = DEMO_NEGATIVE_SNAPSHOT_PATH,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Export two slide-ready PNGs (one per case) for PowerPoint."""
    ensure_default_snapshots()
    out_dir = output_dir or DEMO_HTML_OUTPUT_DIR
    paths = [
        export_case_png(load_snapshot(positive_path), out_dir / "delirium_demo_fall_a.png"),
        export_case_png(load_snapshot(negative_path), out_dir / "delirium_demo_fall_b.png"),
    ]
    return paths


def export_demo_thesis(
    positive_path: Path = DEMO_POSITIVE_SNAPSHOT_PATH,
    negative_path: Path = DEMO_NEGATIVE_SNAPSHOT_PATH,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Export publication-quality thesis case summaries (Markdown + plain text)."""
    ensure_default_snapshots()
    out_dir = output_dir or DEMO_HTML_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = load_snapshot(positive_path)
    fn = load_snapshot(negative_path)
    paths = [
        out_dir / "thesis_case_a_true_positive.md",
        out_dir / "thesis_case_b_false_negative.md",
        out_dir / "thesis_pipeline_case_summaries.md",
        out_dir / "thesis_pipeline_case_summaries.txt",
    ]
    paths[0].write_text(render_thesis_case_summary_markdown(pos), encoding="utf-8")
    paths[1].write_text(render_thesis_case_summary_markdown(fn), encoding="utf-8")
    paths[2].write_text(render_combined_thesis_summaries_markdown(pos, fn), encoding="utf-8")
    paths[3].write_text(render_combined_thesis_summaries_plain(pos, fn), encoding="utf-8")
    return paths


def export_demo_txt(
    positive_path: Path = DEMO_POSITIVE_SNAPSHOT_PATH,
    negative_path: Path = DEMO_NEGATIVE_SNAPSHOT_PATH,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Export hemorrhage-style walkthrough .txt files (one per case + combined)."""
    ensure_default_snapshots()
    out_dir = output_dir or DEMO_HTML_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = load_snapshot(positive_path)
    neg = load_snapshot(negative_path)
    paths = [
        out_dir / "delirium_demo_fall_a_walkthrough.txt",
        out_dir / "delirium_demo_fall_b_walkthrough.txt",
        out_dir / "delirium_pipeline_demo_walkthrough.txt",
    ]
    paths[0].write_text(render_walkthrough_txt(pos), encoding="utf-8")
    paths[1].write_text(render_walkthrough_txt(neg), encoding="utf-8")
    paths[2].write_text(render_combined_walkthrough_txt(pos, neg), encoding="utf-8")
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _interactive_menu() -> None:
    print(f"\n{SEP}")
    print("  Delirium Pipeline Demo")
    print(SEP)
    print("  1  Export thesis case summaries (recommended)")
    print("  2  Positive case — terminal walkthrough")
    print("  3  False-negative case — terminal walkthrough")
    print("  4  Both cases — terminal walkthrough")
    print("  5  Export walkthrough .txt (legacy)")
    print("  6  Export HTML (browser preview)")
    print("  q  Quit")
    choice = input("\nChoice: ").strip().lower()
    if choice == "1":
        for path in export_demo_thesis():
            print(f"Wrote {path}")
    elif choice == "2":
        run_demo(positive=True)
    elif choice == "3":
        run_demo(negative=True)
    elif choice == "4":
        run_demo(both=True)
    elif choice == "5":
        for path in export_demo_txt():
            print(f"Wrote {path}")
    elif choice == "6":
        path = export_demo_html()
        print(f"Wrote {path}")
    else:
        print("Bye.")


def run_demo(
    *,
    positive: bool = False,
    negative: bool = False,
    both: bool = False,
    pause: bool = True,
    positive_path: Path = DEMO_POSITIVE_SNAPSHOT_PATH,
    negative_path: Path = DEMO_NEGATIVE_SNAPSHOT_PATH,
) -> None:
    ensure_default_snapshots()
    if both:
        positive = negative = True
    if not positive and not negative:
        positive = True
    if positive:
        present_snapshot(load_snapshot(positive_path), pause=pause)
    if negative:
        present_snapshot(load_snapshot(negative_path), pause=pause)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Delirium pipeline presentation demo.")
    parser.add_argument("--positive", action="store_true", help="Show positive (TP) case")
    parser.add_argument("--negative", action="store_true", help="Show false-negative (FN) case (legacy alias)")
    parser.add_argument("--false-negative", action="store_true", help="Show false-negative (FN) case")
    parser.add_argument("--both", action="store_true", help="Show both cases")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for ENTER between steps")
    parser.add_argument(
        "--thesis",
        action="store_true",
        help="Export publication-quality thesis case summaries (primary output)",
    )
    parser.add_argument(
        "--txt",
        action="store_true",
        help="Export legacy hemorrhage-style walkthrough .txt files",
    )
    parser.add_argument("--html", action="store_true", help="Export HTML preview to outputs/demo/")
    parser.add_argument(
        "--png",
        action="store_true",
        help="(Optional) Export PNG slides — prefer --txt for custom figures",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call LLM when building snapshots (server only; captures real raw JSON)",
    )
    parser.add_argument(
        "--snapshot-positive",
        action="store_true",
        help="Regenerate positive_case.json from validation data (or curated fallback)",
    )
    parser.add_argument(
        "--snapshot-false-negative",
        action="store_true",
        help="Regenerate FN case in negative_case.json (prefers Patient_0057 / Patient_0075)",
    )
    parser.add_argument(
        "--snapshot-negative",
        action="store_true",
        help="Alias for --snapshot-false-negative (legacy flag name)",
    )
    parser.add_argument("--validation-report-id", help="Force a specific validation_report_id")
    parser.add_argument(
        "--exclude-validation-report-id",
        action="append",
        default=[],
        metavar="ID",
        help="Skip report(s) when auto-picking (repeatable)",
    )
    parser.add_argument(
        "--list-positive-candidates",
        action="store_true",
        help="Print top auto-pick candidates for the TP case and exit",
    )
    parser.add_argument(
        "--list-false-negative-candidates",
        action="store_true",
        help="Print top auto-pick candidates for the FN case and exit",
    )
    parser.add_argument(
        "--diagnose-fn-patients",
        action="store_true",
        help="Explain Patient_0057 / Patient_0075 vs report-level FN requirements",
    )
    parser.add_argument(
        "--fn-patient",
        metavar="SUFFIX",
        help="When auto-picking FN, try this patient first (e.g. 0057 or 0075)",
    )
    parser.add_argument("--positive-snapshot", type=Path, default=DEMO_POSITIVE_SNAPSHOT_PATH)
    parser.add_argument("--negative-snapshot", type=Path, default=DEMO_NEGATIVE_SNAPSHOT_PATH)
    args = parser.parse_args(argv)
    exclude_ids = [x for x in args.exclude_validation_report_id if str(x).strip()]

    if args.list_positive_candidates:
        if not VALIDATION_COHORT_PREDICTIONS_PATH.exists():
            print(f"Missing predictions: {VALIDATION_COHORT_PREDICTIONS_PATH}")
            return 1
        preds = pd.read_csv(VALIDATION_COHORT_PREDICTIONS_PATH)
        labels = (
            pd.read_csv(FROZEN_MANUAL_REPORT_LABELS_PATH)
            if FROZEN_MANUAL_REPORT_LABELS_PATH.exists()
            else None
        )
        print("Top positive-case candidates (higher score = clearer slide):\n")
        for r in rank_validation_candidates(
            preds, labels, polarity="positive", exclude_ids=exclude_ids
        ):
            print(
                f"  {r['score']:3d}  {r['validation_report_id']}  "
                f"{r['bertyp']}  rule={r['decision_rule_applied']}  "
                f"snippets={r['snippet_count']}  len={r['report_length']}"
            )
        return 0

    if args.list_false_negative_candidates:
        if not VALIDATION_COHORT_PREDICTIONS_PATH.exists():
            print(f"Missing predictions: {VALIDATION_COHORT_PREDICTIONS_PATH}")
            return 1
        preds = pd.read_csv(VALIDATION_COHORT_PREDICTIONS_PATH)
        labels = (
            pd.read_csv(FROZEN_MANUAL_REPORT_LABELS_PATH)
            if FROZEN_MANUAL_REPORT_LABELS_PATH.exists()
            else None
        )
        print("Top false-negative candidates (higher score = clearer slide; prefers Patient_0057 / 0075):\n")
        for r in rank_validation_candidates(
            preds, labels, polarity="false_negative", exclude_ids=exclude_ids
        ):
            print(
                f"  {r['score']:3d}  {r['validation_report_id']}  "
                f"{r['bertyp']}  rule={r['decision_rule_applied']}  "
                f"snippets={r['snippet_count']}  len={r['report_length']}"
            )
        return 0

    if args.diagnose_fn_patients:
        if not VALIDATION_COHORT_PREDICTIONS_PATH.exists():
            print(f"Missing predictions: {VALIDATION_COHORT_PREDICTIONS_PATH}")
            return 1
        preds = pd.read_csv(VALIDATION_COHORT_PREDICTIONS_PATH)
        labels = (
            pd.read_csv(FROZEN_MANUAL_REPORT_LABELS_PATH)
            if FROZEN_MANUAL_REPORT_LABELS_PATH.exists()
            else None
        )
        print(
            "FN demo selection: report-level FN (model=0, manual=1) preferred; "
            "patient-level FN (model_patient_positive=0, derived_manual=1) also accepted.\n"
        )
        for block in diagnose_preferred_fn_patients(preds, labels):
            print(f"Patient suffix {block['patient_suffix']}:")
            print(f"  reports in cohort: {block['reports_found']}")
            print(
                f"  patient-level: FN={block['patient_level_fn']} "
                f"model_pos={block['model_patient_positive']} "
                f"derived_manual={block['derived_manual_patient_ground_truth']} "
                f"→ {block['patient_confusion_group'] or '?'}"
            )
            if block["pickable_fn_report_id"]:
                print(f"  → pickable FN report: {block['pickable_fn_report_id']}")
            else:
                print("  → not pickable as FN")
            if not block["all_reports"]:
                print("  (patient not found — check validation_report_id format)")
            for rep in block["all_reports"]:
                print(
                    f"    {rep['validation_report_id']}: "
                    f"model={rep['model_report_prediction']} manual_gt={rep['manual_report_ground_truth']} "
                    f"→ {rep['confusion']}  rule={rep['decision_rule_applied']}"
                )
            print()
        return 0

    snapshot_fn = args.snapshot_false_negative or args.snapshot_negative
    if args.snapshot_positive or snapshot_fn:
        if args.snapshot_positive:
            snap = generate_snapshot_from_validation(
                polarity="positive",
                out_path=args.positive_snapshot,
                validation_report_id=args.validation_report_id if args.snapshot_positive else None,
                exclude_validation_report_ids=exclude_ids or None,
                live=args.live,
            )
            print(f"Wrote positive snapshot → {args.positive_snapshot} ({presentation_case_title(snap)})")
            if args.live:
                print("  capture: live LLM ✓")
        if snapshot_fn:
            snap = generate_snapshot_from_validation(
                polarity="false_negative",
                out_path=args.negative_snapshot,
                validation_report_id=args.validation_report_id if snapshot_fn and not args.snapshot_positive else None,
                exclude_validation_report_ids=exclude_ids or None,
                preferred_fn_patient_suffix=args.fn_patient,
                live=args.live,
            )
            print(f"Wrote FN snapshot → {args.negative_snapshot} ({presentation_case_title(snap)})")
            if snap.get("source") == "curated_anonymized":
                print("  ⚠ curated fallback — NOT from validation cohort. Run --diagnose-fn-patients on server.")
            elif snap.get("case", {}).get("validation_report_id"):
                print(f"  validation_report_id: {snap['case'].get('validation_report_id')}")
            if args.live:
                print("  capture: live LLM ✓")
        return 0

    if args.thesis:
        paths = export_demo_thesis(args.positive_snapshot, args.negative_snapshot)
        for path in paths:
            print(f"Wrote {path}")
        return 0

    if args.txt:
        paths = export_demo_txt(args.positive_snapshot, args.negative_snapshot)
        for path in paths:
            print(f"Wrote {path}")
        return 0

    if args.html:
        path = export_demo_html(args.positive_snapshot, args.negative_snapshot)
        print(f"Wrote {path}")
        return 0

    if args.png:
        paths = export_demo_png(args.positive_snapshot, args.negative_snapshot)
        for path in paths:
            print(f"Wrote {path}")
        return 0

    if not (args.positive or args.negative or args.false_negative or args.both or args.thesis
            or args.txt or args.html or args.png or args.snapshot_positive or snapshot_fn
            or args.list_positive_candidates or args.list_false_negative_candidates):
        _interactive_menu()
        return 0

    if args.positive or args.negative or args.false_negative or args.both:
        run_demo(
            positive=args.positive or args.both,
            negative=args.negative or args.false_negative or args.both,
            both=args.both,
            pause=not args.no_pause,
            positive_path=args.positive_snapshot,
            negative_path=args.negative_snapshot,
        )
        return 0

    _interactive_menu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
