"""
Interactive delirium pipeline demo — presentation-ready walkthrough.

Replays frozen JSON snapshots instantly (no LLM, no Berichte.csv required).
Mirrors the hemorrhage demo pattern: original report → evidence → LLM → decision.

Usage:
    python -m src.analysis.demo_delirium_case                  # menu
    python -m src.analysis.demo_delirium_case --positive       # TP walkthrough
    python -m src.analysis.demo_delirium_case --negative       # TN walkthrough
    python -m src.analysis.demo_delirium_case --both           # both cases
    python -m src.analysis.demo_delirium_case --html           # export slide HTML
    python -m src.analysis.demo_delirium_case --snapshot-positive
    python -m src.analysis.demo_delirium_case --snapshot-negative

See docs/demo/DEMO_GUIDE.md.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analysis.demo_delirium_snapshot import (
    ensure_default_snapshots,
    generate_snapshot_from_validation,
    load_snapshot,
    presentation_case_subtitle,
    presentation_case_title,
    snippet_section_label,
)
from src.pipeline.paths import (
    DEMO_HTML_OUTPUT_DIR,
    DEMO_NEGATIVE_SNAPSHOT_PATH,
    DEMO_POSITIVE_SNAPSHOT_PATH,
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


def present_snapshot(snapshot: Dict[str, Any], *, pause: bool = True) -> None:
    """Render one case as a paced terminal walkthrough."""
    case = snapshot.get("case") or {}
    extraction = snapshot.get("extraction") or {}
    interp = snapshot.get("interpretation") or {}
    final = snapshot.get("final") or {}
    report_text = str(snapshot.get("report_text") or "")
    snippets: List[Dict[str, Any]] = list(extraction.get("evidence_snippets") or [])
    llm_skipped = bool(final.get("llm_skipped_by_prefilter"))
    klasse = int(final.get("klasse") or 0)
    polarity = "POSITIVE · Delir" if klasse == 1 else "NEGATIVE · kein Delir"
    title = presentation_case_title(snapshot)
    subtitle = presentation_case_subtitle(snapshot)

    print(f"\n{SEP}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"  {polarity}")
    print(SEP)

    _step(1, "Original clinical report")
    _explain(
        "Unstructured German ICU documentation — the full input to the pipeline."
    )
    for heading, body in _split_report_sections(report_text):
        print(f"  [{heading}]  ({len(body):,} chars)")
        _block(body, limit=900, indent="      ")
        print()
    _pause(pause)

    _step(2, "Rule-based evidence extraction")
    _explain(
        "Deterministic keyword scan across all report sections.\n"
        "Only clinically relevant snippets are kept — not the full report."
    )
    print("  Detected keywords:")
    _render_keywords(snippets)
    print(f"\n  Snippets extracted: {len(snippets)}")
    for i, snip in enumerate(snippets[:6], start=1):
        _render_snippet(snip, i)
    if len(snippets) > 6:
        print(f"\n    … and {len(snippets) - 6} more snippet(s)")
    method = str(extraction.get("llm_text_reduction_method") or "")
    orig_len = extraction.get("original_report_text_length", len(report_text))
    llm_len = extraction.get("llm_report_text_length", 0)
    print(f"\n  Reduction: {orig_len:,} chars → {llm_len:,} chars LLM bundle ({method})")
    _pause(pause)

    _step(3, "Evidence bundle sent to the LLM")
    if llm_skipped:
        _explain(
            "No actionable evidence → the LLM is skipped entirely.\n"
            "This is the efficiency layer: most reports never reach the model."
        )
        print("    (prefilter skip — nothing forwarded)")
    else:
        _explain("Structured snippet bundle + clinical instruction block.")
        _block(extraction.get("llm_report_text") or "", limit=1400)
    _pause(pause)

    _step(4, "LLM interpretation (Agent 2)")
    if llm_skipped:
        _explain("Skipped — see Step 3.")
    else:
        _explain("The model assigns signal strength and clinical context — not klasse yet.")
    _render_interpretation(interp, llm_skipped=llm_skipped)
    _pause(pause)

    _step(5, "Clinical guardrails → final decision")
    _explain(
        "Deterministic post-LLM rules: direct delir → positive;\n"
        "prophylaxis-only / negation-only / no evidence → negative."
    )
    print(f"    decision_rule_applied:  {final.get('decision_rule_applied', '')}")
    print(f"    manual_review_candidate: {final.get('manual_review_candidate', False)}")
    print(f"    klasse:                  {klasse}  ({'delir' if klasse == 1 else 'kein_delir'})")
    _pause(pause)

    _step(6, "Validation label")
    manual_gt = case.get("manual_report_ground_truth")
    correct = (snapshot.get("verification") or {}).get("model_correct_vs_manual")
    print(f"    manual_report_ground_truth: {manual_gt}")
    if correct is True:
        print("    Model vs manual label:      CORRECT ✓")
    elif correct is False:
        print("    Model vs manual label:      MISMATCH ✗")
    print(f"\n  {RULE}")
    print(f"  Final: klasse = {klasse}  ·  {polarity}")
    print(f"  {RULE}")
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
    def case_block(snap: Dict[str, Any]) -> str:
        extraction = snap.get("extraction") or {}
        interp = snap.get("interpretation") or {}
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
                f'<li><strong>signalstaerke:</strong> {_esc(interp.get("signalstaerke"))}</li>'
                f'<li><strong>kontext:</strong> {_esc(interp.get("kontext"))}</li>'
                f'<li><strong>begruendung:</strong> {_esc(interp.get("begruendung"))}</li>'
                f'</ul>'
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _interactive_menu() -> None:
    print(f"\n{SEP}")
    print("  Delirium Pipeline Demo")
    print(SEP)
    print("  1  Positive case (true positive)")
    print("  2  Negative case (true negative)")
    print("  3  Both cases")
    print("  4  Export HTML (browser preview)")
    print("  5  Export PNG slides (PowerPoint)")
    print("  q  Quit")
    choice = input("\nChoice: ").strip().lower()
    if choice == "1":
        run_demo(positive=True)
    elif choice == "2":
        run_demo(negative=True)
    elif choice == "3":
        run_demo(both=True)
    elif choice == "4":
        path = export_demo_html()
        print(f"Wrote {path}")
    elif choice == "5":
        for path in export_demo_png():
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
    parser.add_argument("--negative", action="store_true", help="Show negative (TN) case")
    parser.add_argument("--both", action="store_true", help="Show both cases")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for ENTER between steps")
    parser.add_argument("--html", action="store_true", help="Export HTML preview to outputs/demo/")
    parser.add_argument(
        "--png",
        action="store_true",
        help="Export PNG slides (recommended for PowerPoint) to outputs/demo/",
    )
    parser.add_argument(
        "--snapshot-positive",
        action="store_true",
        help="Regenerate positive_case.json from validation data (or curated fallback)",
    )
    parser.add_argument(
        "--snapshot-negative",
        action="store_true",
        help="Regenerate negative_case.json from validation data (or curated fallback)",
    )
    parser.add_argument("--validation-report-id", help="Force a specific validation_report_id")
    parser.add_argument("--positive-snapshot", type=Path, default=DEMO_POSITIVE_SNAPSHOT_PATH)
    parser.add_argument("--negative-snapshot", type=Path, default=DEMO_NEGATIVE_SNAPSHOT_PATH)
    args = parser.parse_args(argv)

    if args.snapshot_positive or args.snapshot_negative:
        if args.snapshot_positive:
            snap = generate_snapshot_from_validation(
                polarity="positive",
                out_path=args.positive_snapshot,
                validation_report_id=args.validation_report_id if args.snapshot_positive else None,
            )
            print(f"Wrote positive snapshot → {args.positive_snapshot} ({presentation_case_title(snap)})")
        if args.snapshot_negative:
            snap = generate_snapshot_from_validation(
                polarity="negative",
                out_path=args.negative_snapshot,
                validation_report_id=args.validation_report_id if args.snapshot_negative else None,
            )
            print(f"Wrote negative snapshot → {args.negative_snapshot} ({presentation_case_title(snap)})")
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

    if args.positive or args.negative or args.both:
        run_demo(
            positive=args.positive,
            negative=args.negative,
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
