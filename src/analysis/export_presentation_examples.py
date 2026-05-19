"""
Export presentation-ready examples: report excerpt → keywords → evidence → LLM → prediction.

Read-only on pipeline outputs; does not change prediction, baseline, or evaluation logic.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.analysis.cohort_counts import load_structured_baseline_rows
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    PREDICTIONS_DIR,
    PRESENTATION_EXAMPLES_CSV_PATH,
    PRESENTATION_EXAMPLES_DIR,
    PRESENTATION_EXAMPLES_MD_PATH,
    PRESENTATION_EXAMPLES_REPORT_PATH,
    REPORT_VS_BASELINE_PATH,
    STRUCTURED_BASELINE_PATH,
)
from src.pipeline.schema_normalize import normalize_patient_id_column
from src.preprocessing.evidence_extraction import (
    METHOD_NO_EVIDENCE,
    SECTION_DISPLAY,
    extract_delirium_evidence,
)
from src.preprocessing.berichte_filters import is_dokumentationsblatt, normalize_bertyp

LOGGER = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"

MAX_EXCERPT_CHARS = 800
MAX_SNIPPETS_PER_EXAMPLE = 3

PRESENTATION_CSV_COLUMNS: List[str] = [
    "example_id",
    "PatientenID",
    "bericht",
    "bertyp",
    "original_report_excerpt",
    "highlighted_keywords",
    "evidence_snippets",
    "evidence_types",
    "llm_input_excerpt",
    "model_prediction",
    "signalstaerke",
    "delir_probability_estimate",
    "decision_rule_applied",
    "manual_review_candidate",
    "baseline_icd10",
    "max_icdsc",
    "baseline_icdsc_ge_4",
    "baseline_composite",
    "interpretation_short",
    "slide_usage_suggestion",
]

HIGHLIGHT_BY_TYPE = {
    "direct_delir": ("==", "=="),
    "indirect_symptom": ("**", "**"),
    "negation": ("__", "__"),
    "prophylaxis_or_risk": ("__", "__"),
}


@dataclass(frozen=True)
class ExampleSpec:
    key: str
    title: str
    slide_usage: str
    matcher: Callable[[pd.Series], bool]


def _bool_cell(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in ("1", "true", "yes")


def _int_cell(value: object, default: int = 0) -> int:
    try:
        return int(pd.to_numeric(value, errors="coerce") or default)
    except (TypeError, ValueError):
        return default


def parse_evidence_snippets(raw: object) -> List[Dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return list(raw)
    text = str(raw).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _truncate(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _wrap_keyword(text: str, keyword: str, evidence_type: str) -> str:
    if not keyword or not text:
        return text
    left, right = HIGHLIGHT_BY_TYPE.get(evidence_type, ("", ""))
    if not left:
        return text
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        return f"{left}{match.group(0)}{right}"

    return pattern.sub(repl, text, count=1)


def highlight_text(text: str, snippets: List[Dict[str, Any]]) -> str:
    """Apply presentation markers for detected keywords (longest phrases first)."""
    out = text
    pairs: List[Tuple[str, str]] = []
    for s in snippets:
        kw = str(s.get("keyword") or "").strip()
        et = str(s.get("evidence_type") or "")
        if kw:
            pairs.append((kw, et))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    for kw, et in pairs:
        if kw.lower() in out.lower():
            out = _wrap_keyword(out, kw, et)
    return out


def _best_report_window(report_text: str, snippets: List[Dict[str, Any]], limit: int) -> str:
    if not report_text.strip():
        return ""
    if not snippets:
        return _truncate(report_text, limit)
    low = report_text.lower()
    anchor = 0
    for s in snippets:
        kw = str(s.get("keyword") or "").lower()
        if kw:
            pos = low.find(kw)
            if pos >= 0:
                anchor = pos
                break
    half = limit // 2
    start = max(0, anchor - half)
    end = min(len(report_text), start + limit)
    start = max(0, end - limit)
    chunk = report_text[start:end]
    if start > 0:
        chunk = "…" + chunk
    if end < len(report_text):
        chunk = chunk + "…"
    return chunk.strip()


def format_keywords_list(snippets: List[Dict[str, Any]]) -> str:
    by_type: Dict[str, List[str]] = {}
    for s in snippets:
        et = str(s.get("evidence_type") or "unknown")
        kw = str(s.get("keyword") or "").strip()
        if kw and kw not in by_type.get(et, []):
            by_type.setdefault(et, []).append(kw)
    lines: List[str] = []
    for et in ("direct_delir", "indirect_symptom", "negation", "prophylaxis_or_risk"):
        if et in by_type:
            lines.append(f"- {et}: {', '.join(by_type[et])}")
    return "\n".join(lines) if lines else "- (none)"


def format_snippets_markdown(snippets: List[Dict[str, Any]], limit: int = MAX_SNIPPETS_PER_EXAMPLE) -> str:
    lines: List[str] = []
    for s in snippets[:limit]:
        sec = SECTION_DISPLAY.get(str(s.get("section")), str(s.get("section")))
        et = s.get("evidence_type", "")
        body = highlight_text(str(s.get("text") or ""), [s])
        lines.append(f"- [{sec} | {et}] {body}")
    return "\n".join(lines) if lines else "- (none)"


def interpretation_short(row: pd.Series) -> str:
    parts: List[str] = []
    sig = str(row.get("signalstaerke") or "").strip()
    if sig:
        parts.append(f"Signalstärke: {sig}")
    prob = row.get("delir_probability_estimate")
    if prob is not None and str(prob).strip() not in ("", "nan"):
        parts.append(f"Probability estimate: {prob}")
    kontext = str(row.get("kontext") or "").strip()
    if kontext:
        parts.append(kontext[:300])
    begr = str(row.get("begruendung") or "").strip()
    if begr and begr not in kontext:
        parts.append(begr[:200])
    return " | ".join(parts) if parts else "No LLM interpretation (prefilter skip or failure)."


def load_report_text_lookup(berichte_path: Path = BERICHTE_INPUT_PATH) -> Dict[Tuple[str, str], str]:
    """Map (PatientenID, bericht) → stitched report_text; empty dict if Berichte missing."""
    if not berichte_path.exists():
        LOGGER.warning("Berichte.csv not found at %s; excerpts from evidence only.", berichte_path)
        return {}
    try:
        from src.preprocessing.berichte_mapper import build_report_level_berichte_records

        records, _ = build_report_level_berichte_records(berichte_path)
    except Exception as exc:
        LOGGER.warning("Could not load Berichte for excerpts: %s", exc)
        return {}
    lookup: Dict[Tuple[str, str], str] = {}
    for rec in records:
        pid = str(rec.get("PatientenID", "")).strip()
        bericht = str(rec.get("bericht", "")).strip()
        if pid and bericht:
            lookup[(pid, bericht)] = str(rec.get("report_text") or "")
    return lookup


def _prepare_predictions_frame(
    predictions: pd.DataFrame,
    baseline: Optional[pd.DataFrame],
    comparison: Optional[pd.DataFrame],
) -> pd.DataFrame:
    pred = normalize_patient_id_column(predictions.copy())
    if "bertyp" in pred.columns:
        pred["bertyp"] = pred["bertyp"].map(normalize_bertyp)
        pred = pred[~pred["bertyp"].map(is_dokumentationsblatt)].copy()

    if baseline is not None and not baseline.empty:
        base = normalize_patient_id_column(baseline).drop_duplicates("PatientenID", keep="first")
        cols = [c for c in ("baseline_icd10", "max_icdsc", "baseline_icdsc_ge_4", "baseline_composite") if c in base.columns]
        if cols:
            pred = pred.merge(base[["PatientenID", *cols]], on="PatientenID", how="left")

    if comparison is not None and not comparison.empty:
        comp = normalize_patient_id_column(comparison)
        merge_keys = ["PatientenID"]
        if "bericht" in comp.columns and "bericht" in pred.columns:
            merge_keys.append("bericht")
        comp_cols = [
            c
            for c in ("agreement_report_vs_baseline_composite", "prediction_binary")
            if c in comp.columns
        ]
        if comp_cols:
            pred = pred.merge(comp[merge_keys + comp_cols].drop_duplicates(merge_keys), on=merge_keys, how="left")

    return pred


def _example_specs() -> List[ExampleSpec]:
    def direct_pos(row: pd.Series) -> bool:
        return _bool_cell(row.get("has_direct_delir_evidence")) and _int_cell(row.get("klasse")) == 1

    def indirect_borderline(row: pd.Series) -> bool:
        if _bool_cell(row.get("has_direct_delir_evidence")):
            return False
        if not _bool_cell(row.get("has_indirect_delir_evidence")):
            return False
        return _int_cell(row.get("klasse")) == 1 or _bool_cell(row.get("manual_review_candidate"))

    def no_evidence(row: pd.Series) -> bool:
        if _bool_cell(row.get("llm_skipped_by_prefilter")):
            return True
        return str(row.get("llm_text_reduction_method") or "").strip() == METHOD_NO_EVIDENCE

    def discrepancy(row: pd.Series) -> bool:
        if "agreement_report_vs_baseline_composite" in row.index:
            val = row.get("agreement_report_vs_baseline_composite")
            if pd.notna(val):
                return not _bool_cell(val) if str(val).strip().lower() not in ("0", "1") else _int_cell(val) == 0
        bc = row.get("baseline_composite")
        if bc is None or (isinstance(bc, float) and pd.isna(bc)):
            return False
        return _int_cell(row.get("klasse")) != _int_cell(bc)

    def prophylaxis(row: pd.Series) -> bool:
        if _bool_cell(row.get("has_prophylaxis_or_risk_only")):
            return True
        return str(row.get("decision_rule_applied") or "") == "prophylaxis_only_not_positive"

    return [
        ExampleSpec(
            "clear_direct_delir",
            "Clear direct delir",
            "Methods / positive case: explicit delir documentation through full pipeline",
            direct_pos,
        ),
        ExampleSpec(
            "indirect_borderline",
            "Indirect / borderline positive",
            "Uncertainty slide: indirect symptoms, manual review or alternative-context downgrade",
            indirect_borderline,
        ),
        ExampleSpec(
            "no_evidence_skip",
            "No-evidence prefilter skip",
            "Efficiency slide: rule layer skips LLM when no actionable snippets",
            no_evidence,
        ),
        ExampleSpec(
            "baseline_discrepancy",
            "Discrepancy vs composite baseline",
            "Validation slide: model vs baseline_composite mismatch (patient-level baseline caveat)",
            discrepancy,
        ),
        ExampleSpec(
            "prophylaxis_risk",
            "Prophylaxis / risk only",
            "Guardrail slide: prophylaxis or screening without delirium diagnosis",
            prophylaxis,
        ),
    ]


def select_example_indices(df: pd.DataFrame) -> List[Tuple[int, ExampleSpec]]:
    """Pick up to one row per category; export all rows if fewer than one match total."""
    chosen: List[Tuple[int, ExampleSpec]] = []
    used: set[int] = set()
    for spec in _example_specs():
        for idx, row in df.iterrows():
            i = int(idx)
            if i in used:
                continue
            if spec.matcher(row):
                chosen.append((i, spec))
                used.add(i)
                break
    if not chosen and len(df) > 0:
        for idx in df.index[: min(5, len(df))]:
            i = int(idx)
            chosen.append((i, _example_specs()[0]))
    return chosen


def build_example_row(
    row: pd.Series,
    spec: ExampleSpec,
    example_num: int,
    report_lookup: Dict[Tuple[str, str], str],
) -> Dict[str, Any]:
    pid = str(row.get("PatientenID") or "")
    bericht = str(row.get("bericht") or "")
    snippets = parse_evidence_snippets(row.get("evidence_snippets"))
    report_text = report_lookup.get((pid, bericht), "")

    if report_text:
        ev = extract_delirium_evidence(report_text)
        llm_excerpt = _truncate(ev.get("llm_report_text") or "", MAX_EXCERPT_CHARS)
        if not snippets:
            snippets = ev.get("evidence_snippets") or []
    else:
        llm_excerpt = _truncate(
            "\n".join(
                f"[{SECTION_DISPLAY.get(str(s.get('section')), s.get('section'))} | {s.get('evidence_type')}] "
                f"{s.get('text', '')}"
                for s in snippets[:MAX_SNIPPETS_PER_EXAMPLE]
            ),
            MAX_EXCERPT_CHARS,
        )

    excerpt_raw = _best_report_window(report_text, snippets, MAX_EXCERPT_CHARS)
    if not excerpt_raw and snippets:
        excerpt_raw = " ".join(str(s.get("text") or "") for s in snippets[:3])
    excerpt_highlighted = highlight_text(excerpt_raw, snippets)

    kw_flat = []
    for s in snippets:
        et = str(s.get("evidence_type") or "")
        kw = str(s.get("keyword") or "")
        if kw:
            kw_flat.append(f"{et}:{kw}")

    display_snippets = snippets[:MAX_SNIPPETS_PER_EXAMPLE]
    evidence_types = " | ".join(sorted({str(s.get("evidence_type") or "") for s in display_snippets}))

    return {
        "example_id": f"example_{example_num:02d}_{spec.key}",
        "PatientenID": pid,
        "bericht": bericht,
        "bertyp": str(row.get("bertyp") or ""),
        "original_report_excerpt": excerpt_highlighted,
        "highlighted_keywords": "; ".join(kw_flat),
        "evidence_snippets": json.dumps(display_snippets, ensure_ascii=False),
        "evidence_types": evidence_types,
        "llm_input_excerpt": llm_excerpt,
        "model_prediction": _int_cell(row.get("klasse")),
        "signalstaerke": str(row.get("signalstaerke") or ""),
        "delir_probability_estimate": row.get("delir_probability_estimate", ""),
        "decision_rule_applied": str(row.get("decision_rule_applied") or ""),
        "manual_review_candidate": _bool_cell(row.get("manual_review_candidate")),
        "baseline_icd10": row.get("baseline_icd10", ""),
        "max_icdsc": row.get("max_icdsc", ""),
        "baseline_icdsc_ge_4": row.get("baseline_icdsc_ge_4", ""),
        "baseline_composite": row.get("baseline_composite", ""),
        "interpretation_short": interpretation_short(row),
        "slide_usage_suggestion": spec.slide_usage,
        "_spec_title": spec.title,
        "_spec_key": spec.key,
        "_snippets": display_snippets,
        "_keywords_md": format_keywords_list(snippets),
        "_snippets_md": format_snippets_markdown(snippets),
    }


def render_markdown_example(example: Dict[str, Any], example_num: int) -> str:
    title = example.get("_spec_title", "Example")
    lines = [
        f"## Example {example_num} — {title}",
        "",
        "### 1. Original report excerpt",
        example.get("original_report_excerpt") or "(not available)",
        "",
        "### 2. Rule-based keyword detection",
        example.get("_keywords_md", "- (none)"),
        "",
        "### 3. Evidence snippets sent to LLM",
        example.get("_snippets_md", "- (none)"),
        "",
        "### 4. LLM interpretation",
        example.get("interpretation_short", ""),
        "",
        "### 5. Final output",
        f"klasse = {example.get('model_prediction')}",
        f"baseline_composite = {example.get('baseline_composite', '')}",
        f"decision_rule_applied = {example.get('decision_rule_applied', '')}",
        f"manual_review_candidate = {example.get('manual_review_candidate')}",
        "",
        f"*Slide suggestion:* {example.get('slide_usage_suggestion', '')}",
        "",
    ]
    return "\n".join(lines)


def build_presentation_examples(
    predictions: pd.DataFrame,
    baseline: Optional[pd.DataFrame] = None,
    comparison: Optional[pd.DataFrame] = None,
    report_lookup: Optional[Dict[Tuple[str, str], str]] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """Return (csv_df, markdown_body, report_txt)."""
    df = _prepare_predictions_frame(predictions, baseline, comparison)
    lookup = report_lookup if report_lookup is not None else {}

    picks = select_example_indices(df)
    examples: List[Dict[str, Any]] = []
    md_blocks: List[str] = ["# Presentation examples — delirium detection pipeline", ""]
    report_lines = [
        "Presentation examples export report",
        "=" * 40,
        f"source_rows={len(df)}",
        f"selected_examples={len(picks)}",
        "",
        "Summary table",
        "-" * 40,
        "example_id | type | PatientenID | bertyp | prediction | baseline_composite | slide",
    ]

    for n, (idx, spec) in enumerate(picks, start=1):
        row = df.loc[idx]
        ex = build_example_row(row, spec, n, lookup)
        examples.append(ex)
        md_blocks.append(render_markdown_example(ex, n))
        md_blocks.append("---\n")
        report_lines.append(
            f"{ex['example_id']} | {spec.key} | {ex['PatientenID']} | {ex['bertyp']} | "
            f"{ex['model_prediction']} | {ex.get('baseline_composite', '')} | {ex['slide_usage_suggestion']}"
        )

    if not examples:
        md_blocks.append("_No prediction rows available for export._\n")
        report_lines.append("(no examples exported)")

    csv_df = pd.DataFrame(examples)
    if not csv_df.empty:
        drop_cols = [c for c in csv_df.columns if c.startswith("_")]
        csv_df = csv_df[[c for c in PRESENTATION_CSV_COLUMNS if c in csv_df.columns]]

    md_body = "\n".join(md_blocks)
    report_txt = "\n".join(report_lines) + "\n"
    return csv_df, md_body, report_txt


def main(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    comparison_path: Path = REPORT_VS_BASELINE_PATH,
    output_dir: Path = PRESENTATION_EXAMPLES_DIR,
    berichte_path: Path = BERICHTE_INPUT_PATH,
) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions missing: {predictions_path}. Run python -m src.pipeline.run_pipeline first."
        )

    preds = pd.read_csv(predictions_path)
    baseline: Optional[pd.DataFrame] = None
    if baseline_path.exists():
        baseline = load_structured_baseline_rows(baseline_path)
    else:
        LOGGER.warning("Baseline missing at %s; baseline columns left empty.", baseline_path)

    comparison: Optional[pd.DataFrame] = None
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
    else:
        LOGGER.info("Comparison file not found (optional): %s", comparison_path)

    lookup = load_report_text_lookup(berichte_path)
    csv_df, md_body, report_txt = build_presentation_examples(
        preds, baseline=baseline, comparison=comparison, report_lookup=lookup
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "presentation_examples.csv"
    md_path = output_dir / "presentation_examples.md"
    rep_path = output_dir / "presentation_examples_report.txt"

    csv_df.to_csv(csv_path, index=False)
    md_path.write_text(md_body, encoding="utf-8")
    rep_path.write_text(report_txt, encoding="utf-8")

    print(f"Wrote presentation examples CSV: {csv_path}")
    print(f"Wrote presentation examples MD: {md_path}")
    print(f"Wrote presentation examples report: {rep_path}")
    print(f"examples_exported={len(csv_df)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
