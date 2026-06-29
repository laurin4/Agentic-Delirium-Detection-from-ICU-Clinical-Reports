#!/usr/bin/env python3
"""
Summarize the rule-based evidence extraction stage for the frozen delirium
validation cohort (thesis Methods/Results).

READ-ONLY ANALYSIS. This script:
  * does NOT modify the production pipeline, prompts, or any stored data;
  * does NOT run any LLM inference;
  * only reads existing prediction-trace and frozen-cohort files.

------------------------------------------------------------------------------
WHICH INPUT FILES ARE USED
------------------------------------------------------------------------------
The rule-based extraction stage (src/preprocessing/evidence_extraction.py) is
deterministic and prompt-independent: V1 and V2 runs produce identical evidence
snippets for the same report text. Therefore ANY full-cohort prediction trace
that contains the extraction-metadata columns is a valid source.

Each prediction-trace row stores (written by src/pipeline/run_pipeline.py and
src/pipeline/cascade_report_inference.py):
  - validation_patient_id, validation_report_id
  - evidence_snippets            (JSON list of {section, keyword, evidence_type,
                                   priority, text})
  - original_report_text_length  (chars of the full report)
  - llm_report_text_length       (chars of the bounded evidence bundle sent to LLM)
  - llm_text_reduction_method    one of:
        "structured_evidence_extraction"     -> snippets sent to LLM (eligible)
        "short_report_no_evidence_fulltext"  -> short report full text sent (eligible)
        "no_evidence_prefilter_skip"         -> NOT sent to LLM (prefilter skip)
  - delir_keyword_hits_count, has_*_evidence flags, status, llm_called, ...

Prediction-trace sources are searched in this priority order (first existing
file with the most reports wins):
  1. outputs/analysis/manual_validation/prompt_runs/v1/run_01/predictions/
         validation_cohort_predictions.csv
  2. outputs/analysis/manual_validation/prompt_runs/v2/run_02/predictions/
         validation_cohort_predictions.csv
  3. outputs/analysis/manual_validation/cascade_v1_v2_v3/run_01/checkpoints/
         v1_inference.jsonl              (full V1 row stored under "full_row")
  4. outputs/predictions/validation_cohort_predictions.csv   (legacy)

Frozen-cohort denominators (for cross-checking patient/report counts) are read,
if present, from:
  - outputs/analysis/manual_validation/frozen_validation_cohort/
        manual_report_labels_frozen.csv        (one row per report)
  - outputs/analysis/manual_validation/frozen_validation_cohort/
        patient_validation_cohort_frozen.csv   (one row per patient)

Any statistic that cannot be computed from the available files is reported as
"not available" rather than guessed.

------------------------------------------------------------------------------
OUTPUTS
------------------------------------------------------------------------------
  results/rule_extraction_stats_validation.csv
  results/rule_extraction_stats_validation.md

Run:
  python -m scripts.analysis.rule_extraction_stats_validation
  # or
  python scripts/analysis/rule_extraction_stats_validation.py
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Repository-relative paths (no production-code imports; read-only) --------
# scripts/analysis/<this file>  ->  parents[2] == delirium_project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_VALIDATION_DIR = PROJECT_ROOT / "outputs" / "analysis" / "manual_validation"
PROMPT_RUNS_ROOT = MANUAL_VALIDATION_DIR / "prompt_runs"
CASCADE_DIR = MANUAL_VALIDATION_DIR / "cascade_v1_v2_v3"
FROZEN_DIR = MANUAL_VALIDATION_DIR / "frozen_validation_cohort"

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_CSV = RESULTS_DIR / "rule_extraction_stats_validation.csv"
OUT_MD = RESULTS_DIR / "rule_extraction_stats_validation.md"

# Prediction-trace candidates in priority order. (kind: "csv" | "jsonl")
TRACE_CANDIDATES: Tuple[Tuple[Path, str, str], ...] = (
    (PROMPT_RUNS_ROOT / "v1" / "run_01" / "predictions" / "validation_cohort_predictions.csv", "csv", "prompt_runs/v1/run_01"),
    (PROMPT_RUNS_ROOT / "v2" / "run_02" / "predictions" / "validation_cohort_predictions.csv", "csv", "prompt_runs/v2/run_02"),
    (CASCADE_DIR / "run_01" / "checkpoints" / "v1_inference.jsonl", "jsonl", "cascade_v1_v2_v3/run_01 V1 checkpoint"),
    (PROJECT_ROOT / "outputs" / "predictions" / "validation_cohort_predictions.csv", "csv", "outputs/predictions (legacy)"),
)

FROZEN_REPORT_LABELS = FROZEN_DIR / "manual_report_labels_frozen.csv"
FROZEN_PATIENT_COHORT = FROZEN_DIR / "patient_validation_cohort_frozen.csv"

# Method tokens (mirror src/preprocessing/evidence_extraction.py).
METHOD_NO_EVIDENCE = "no_evidence_prefilter_skip"
METHOD_STRUCTURED = "structured_evidence_extraction"
METHOD_SHORT_REPORT_FULLTEXT = "short_report_no_evidence_fulltext"

KNOWN_EVIDENCE_TYPES = ("direct_delir", "indirect_symptom", "negation", "prophylaxis_or_risk")

# Frozen cohort is expected to be 100 patients / 616 reports; used only to flag
# partial/stub traces, never to fabricate values.
EXPECTED_REPORTS = 616
NA = "not available"


# --- Loading ------------------------------------------------------------------
def _load_csv_trace(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_jsonl_trace(path: Path) -> pd.DataFrame:
    """Flatten cascade V1 checkpoint jsonl: prefer the stored 'full_row' dict."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            full = obj.get("full_row") if isinstance(obj, dict) else None
            row = dict(full) if isinstance(full, dict) else dict(obj)
            # Carry the top-level report id if the full_row lacks it.
            if "validation_report_id" not in row and isinstance(obj, dict):
                if obj.get("validation_report_id"):
                    row["validation_report_id"] = obj["validation_report_id"]
            rows.append(row)
    return pd.DataFrame(rows)


def select_trace() -> Optional[Tuple[pd.DataFrame, Path, str]]:
    """
    Return (dataframe, path, label) for the best available prediction trace, or
    None if no candidate exists. "Best" = existing candidate with the most rows,
    respecting priority order on ties.
    """
    best: Optional[Tuple[pd.DataFrame, Path, str]] = None
    best_rows = -1
    for path, kind, label in TRACE_CANDIDATES:
        if not path.exists():
            continue
        try:
            df = _load_csv_trace(path) if kind == "csv" else _load_jsonl_trace(path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"WARNING: could not read {path}: {exc}")
            continue
        if "evidence_snippets" not in df.columns:
            print(f"WARNING: {path} has no 'evidence_snippets' column; skipping.")
            continue
        if len(df) > best_rows:
            best = (df, path, label)
            best_rows = len(df)
    return best


def _parse_snippets(value: Any) -> List[Dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [s for s in value if isinstance(s, dict)]
    text = str(value).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    return [s for s in parsed if isinstance(s, dict)] if isinstance(parsed, list) else []


# --- Frozen cohort denominators ----------------------------------------------
def frozen_cohort_counts() -> Tuple[Optional[int], Optional[int], List[str]]:
    """Return (n_reports, n_patients, notes) from frozen cohort files if present."""
    notes: List[str] = []
    n_reports: Optional[int] = None
    n_patients: Optional[int] = None

    if FROZEN_REPORT_LABELS.exists():
        labels = pd.read_csv(FROZEN_REPORT_LABELS)
        n_reports = int(len(labels))
        for col in ("validation_patient_id", "PatientenID"):
            if col in labels.columns:
                n_patients = int(labels[col].astype(str).nunique())
                break
    else:
        notes.append(f"Frozen report labels not found ({FROZEN_REPORT_LABELS.name}).")

    if n_patients is None and FROZEN_PATIENT_COHORT.exists():
        cohort = pd.read_csv(FROZEN_PATIENT_COHORT)
        n_patients = int(len(cohort))
    elif not FROZEN_PATIENT_COHORT.exists():
        notes.append(f"Frozen patient cohort not found ({FROZEN_PATIENT_COHORT.name}).")

    return n_reports, n_patients, notes


# --- Core computation ---------------------------------------------------------
def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all rule-extraction statistics from a prediction-trace dataframe."""
    stats: Dict[str, Any] = {"notes": []}
    n_reports = int(len(df))
    stats["n_reports_trace"] = n_reports

    # Patients from the trace.
    pid_col = next((c for c in ("validation_patient_id", "PatientenID") if c in df.columns), None)
    stats["n_patients_trace"] = int(df[pid_col].astype(str).nunique()) if pid_col else None
    if pid_col is None:
        stats["notes"].append("No patient-id column in trace; patient count from trace unavailable.")

    # Per-report parsed snippet lists and counts.
    snippet_lists = [_parse_snippets(v) for v in df["evidence_snippets"].tolist()]
    snippet_counts = [len(s) for s in snippet_lists]

    reports_with_snippet = sum(1 for c in snippet_counts if c >= 1)
    reports_without_snippet = n_reports - reports_with_snippet
    stats["reports_with_snippet"] = reports_with_snippet
    stats["reports_without_snippet"] = reports_without_snippet

    # Prefilter skip / LLM eligibility from llm_text_reduction_method (if present).
    if "llm_text_reduction_method" in df.columns:
        methods = df["llm_text_reduction_method"].astype(str)
        n_skip = int((methods == METHOD_NO_EVIDENCE).sum())
        n_structured = int((methods == METHOD_STRUCTURED).sum())
        n_short_fulltext = int((methods == METHOD_SHORT_REPORT_FULLTEXT).sum())
        stats["reports_prefilter_skipped"] = n_skip
        stats["reports_sent_to_llm"] = n_structured + n_short_fulltext
        stats["reports_short_fulltext_fallback"] = n_short_fulltext
    else:
        stats["reports_prefilter_skipped"] = None
        stats["reports_sent_to_llm"] = None
        stats["reports_short_fulltext_fallback"] = None
        stats["notes"].append(
            "Column 'llm_text_reduction_method' missing; prefilter-skip / LLM-eligibility "
            "counts unavailable."
        )

    # Snippet totals and distribution.
    total_snippets = int(sum(snippet_counts))
    stats["total_snippets"] = total_snippets
    if snippet_counts:
        stats["snippets_mean"] = round(statistics.mean(snippet_counts), 4)
        stats["snippets_median"] = float(statistics.median(snippet_counts))
        stats["snippets_min"] = int(min(snippet_counts))
        stats["snippets_max"] = int(max(snippet_counts))
        # Distribution over reports that actually have >=1 snippet.
        nonzero = [c for c in snippet_counts if c >= 1]
        stats["snippets_mean_nonzero"] = round(statistics.mean(nonzero), 4) if nonzero else None
    else:
        for k in ("snippets_mean", "snippets_median", "snippets_min", "snippets_max", "snippets_mean_nonzero"):
            stats[k] = None

    # Evidence-type breakdown across all snippets.
    type_counts: Dict[str, int] = {}
    for snips in snippet_lists:
        for s in snips:
            et = str(s.get("evidence_type") or "unknown")
            type_counts[et] = type_counts.get(et, 0) + 1
    stats["evidence_type_counts"] = type_counts

    # Text-length reduction (optional).
    if "original_report_text_length" in df.columns and "llm_report_text_length" in df.columns:
        orig = pd.to_numeric(df["original_report_text_length"], errors="coerce")
        llm = pd.to_numeric(df["llm_report_text_length"], errors="coerce")
        orig_valid = orig.dropna()
        llm_valid = llm.dropna()
        stats["mean_original_len"] = round(float(orig_valid.mean()), 2) if len(orig_valid) else None
        stats["mean_llm_len"] = round(float(llm_valid.mean()), 2) if len(llm_valid) else None
        # Reduction computed on reports actually sent to the LLM (llm_len > 0),
        # so prefilter-skipped reports (llm_len == 0) do not distort the mean.
        sent_mask = llm > 0
        orig_sent = orig[sent_mask].dropna()
        llm_sent = llm[sent_mask].dropna()
        if len(orig_sent) and float(orig_sent.sum()) > 0:
            reduction = 1.0 - (float(llm_sent.sum()) / float(orig_sent.sum()))
            stats["pct_length_reduction_sent"] = round(100.0 * reduction, 2)
            stats["n_reports_for_reduction"] = int(sent_mask.sum())
        else:
            stats["pct_length_reduction_sent"] = None
            stats["n_reports_for_reduction"] = 0
    else:
        stats["mean_original_len"] = None
        stats["mean_llm_len"] = None
        stats["pct_length_reduction_sent"] = None
        stats["n_reports_for_reduction"] = None
        stats["notes"].append(
            "Length columns missing; text-reduction statistics unavailable."
        )

    # Optional: rule keyword hit count (pre-dedup/cap) if present.
    if "delir_keyword_hits_count" in df.columns:
        hits = pd.to_numeric(df["delir_keyword_hits_count"], errors="coerce").dropna()
        stats["total_keyword_hits_raw"] = int(hits.sum()) if len(hits) else None
    else:
        stats["total_keyword_hits_raw"] = None

    return stats


def _pct(part: Optional[int], whole: Optional[int]) -> Optional[float]:
    if part is None or whole is None or whole == 0:
        return None
    return round(100.0 * part / whole, 2)


def _fmt(value: Any) -> str:
    if value is None:
        return NA
    return str(value)


def _fmt_pct(value: Optional[float]) -> str:
    return NA if value is None else f"{value:.2f}%"


# --- Output rendering ---------------------------------------------------------
def build_rows(
    stats: Dict[str, Any],
    n_reports: int,
    n_patients_display: Optional[int],
    patients_source: str,
    reports_source: str,
) -> List[Dict[str, str]]:
    """Tidy metric/value/percent/note rows for the CSV."""
    rows: List[Dict[str, str]] = []

    def add(metric: str, value: Any, percent: Optional[float] = None, note: str = "") -> None:
        rows.append(
            {
                "metric": metric,
                "value": _fmt(value),
                "percent": _fmt_pct(percent) if percent is not None else "",
                "note": note,
            }
        )

    add("total_patients", n_patients_display, note=patients_source)
    add("total_reports", n_reports, note=reports_source)
    add(
        "reports_with_at_least_one_snippet",
        stats["reports_with_snippet"],
        _pct(stats["reports_with_snippet"], n_reports),
    )
    add(
        "reports_without_evidence_snippet",
        stats["reports_without_snippet"],
        _pct(stats["reports_without_snippet"], n_reports),
    )
    add(
        "reports_skipped_by_prefilter",
        stats["reports_prefilter_skipped"],
        _pct(stats["reports_prefilter_skipped"], n_reports),
        note="llm_text_reduction_method == no_evidence_prefilter_skip",
    )
    add(
        "reports_eligible_sent_to_llm",
        stats["reports_sent_to_llm"],
        _pct(stats["reports_sent_to_llm"], n_reports),
        note="structured_evidence_extraction + short_report_no_evidence_fulltext",
    )
    add(
        "reports_short_fulltext_fallback",
        stats["reports_short_fulltext_fallback"],
        _pct(stats["reports_short_fulltext_fallback"], n_reports),
        note="short reports without snippets sent as full text (if enabled)",
    )
    add("total_evidence_snippets", stats["total_snippets"])
    add("snippets_per_report_mean", stats["snippets_mean"], note="over all reports")
    add("snippets_per_report_median", stats["snippets_median"], note="over all reports")
    add("snippets_per_report_min", stats["snippets_min"], note="over all reports")
    add("snippets_per_report_max", stats["snippets_max"], note="over all reports")
    add(
        "snippets_per_report_mean_nonzero",
        stats["snippets_mean_nonzero"],
        note="over reports with >=1 snippet",
    )
    add("mean_original_report_length_chars", stats["mean_original_len"])
    add("mean_llm_evidence_bundle_length_chars", stats["mean_llm_len"])
    add(
        "pct_text_length_reduction",
        stats["pct_length_reduction_sent"],
        note=f"on {_fmt(stats['n_reports_for_reduction'])} reports sent to LLM (llm_len>0)",
    )
    add(
        "total_rule_keyword_hits_raw",
        stats["total_keyword_hits_raw"],
        note="pre-dedup/cap keyword matches (optional)",
    )

    # Evidence-type breakdown rows.
    type_counts: Dict[str, int] = stats["evidence_type_counts"]
    total_snip = stats["total_snippets"] or 0
    ordered_types = list(KNOWN_EVIDENCE_TYPES) + sorted(
        t for t in type_counts if t not in KNOWN_EVIDENCE_TYPES
    )
    for et in ordered_types:
        count = type_counts.get(et, 0)
        add(
            f"snippets_evidence_type__{et}",
            count,
            _pct(count, total_snip) if total_snip else None,
            note="share of all snippets",
        )
    return rows


def _evidence_type_table(stats: Dict[str, Any]) -> List[Tuple[str, int, Optional[float]]]:
    type_counts: Dict[str, int] = stats["evidence_type_counts"]
    total_snip = stats["total_snippets"] or 0
    ordered_types = list(KNOWN_EVIDENCE_TYPES) + sorted(
        t for t in type_counts if t not in KNOWN_EVIDENCE_TYPES
    )
    out: List[Tuple[str, int, Optional[float]]] = []
    for et in ordered_types:
        count = type_counts.get(et, 0)
        out.append((et, count, _pct(count, total_snip) if total_snip else None))
    return out


def build_markdown(
    stats: Dict[str, Any],
    n_reports: int,
    n_patients_display: Optional[int],
    patients_source: str,
    reports_source: str,
    trace_path: Path,
    trace_label: str,
    warnings: List[str],
) -> str:
    pct_with = _pct(stats["reports_with_snippet"], n_reports)
    pct_skip = _pct(stats["reports_prefilter_skipped"], n_reports)
    pct_sent = _pct(stats["reports_sent_to_llm"], n_reports)

    lines: List[str] = []
    lines.append("# Rule-based Evidence Extraction — Frozen Delirium Validation Cohort")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")
    lines.append(f"- **Prediction-trace source:** `{trace_path}` ({trace_label})")
    lines.append(f"- **Patient count source:** {patients_source}")
    lines.append(f"- **Report count source:** {reports_source}")
    lines.append("")

    if warnings:
        lines.append("> **Warnings**")
        for w in warnings:
            lines.append(f">")
            lines.append(f"> - {w}")
        lines.append("")

    # Summary table.
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value | % of reports |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Patients in validation cohort | {_fmt(n_patients_display)} | – |")
    lines.append(f"| Reports in validation cohort | {n_reports} | – |")
    lines.append(
        f"| Reports with ≥1 evidence snippet | {_fmt(stats['reports_with_snippet'])} | {_fmt_pct(pct_with)} |"
    )
    lines.append(
        f"| Reports without evidence snippet | {_fmt(stats['reports_without_snippet'])} | "
        f"{_fmt_pct(_pct(stats['reports_without_snippet'], n_reports))} |"
    )
    lines.append(
        f"| Reports skipped by prefilter (no LLM) | {_fmt(stats['reports_prefilter_skipped'])} | {_fmt_pct(pct_skip)} |"
    )
    lines.append(
        f"| Reports eligible / sent to LLM | {_fmt(stats['reports_sent_to_llm'])} | {_fmt_pct(pct_sent)} |"
    )
    lines.append(
        f"| — of which short-report full-text fallback | {_fmt(stats['reports_short_fulltext_fallback'])} | "
        f"{_fmt_pct(_pct(stats['reports_short_fulltext_fallback'], n_reports))} |"
    )
    lines.append(f"| Total evidence snippets extracted | {_fmt(stats['total_snippets'])} | – |")
    lines.append(f"| Snippets per report — mean | {_fmt(stats['snippets_mean'])} | – |")
    lines.append(f"| Snippets per report — median | {_fmt(stats['snippets_median'])} | – |")
    lines.append(f"| Snippets per report — min | {_fmt(stats['snippets_min'])} | – |")
    lines.append(f"| Snippets per report — max | {_fmt(stats['snippets_max'])} | – |")
    lines.append(
        f"| Snippets per report — mean (reports with ≥1) | {_fmt(stats['snippets_mean_nonzero'])} | – |"
    )
    lines.append(f"| Mean original report length (chars) | {_fmt(stats['mean_original_len'])} | – |")
    lines.append(f"| Mean LLM evidence-bundle length (chars) | {_fmt(stats['mean_llm_len'])} | – |")
    lines.append(
        f"| Text-length reduction before LLM | {_fmt_pct(stats['pct_length_reduction_sent'])} | – |"
    )
    lines.append("")

    # Evidence-type breakdown.
    lines.append("## Evidence-type breakdown")
    lines.append("")
    lines.append("| Evidence type | Snippets | % of all snippets |")
    lines.append("|---|---:|---:|")
    for et, count, pct in _evidence_type_table(stats):
        lines.append(f"| `{et}` | {count} | {_fmt_pct(pct)} |")
    lines.append("")

    # Thesis-ready interpretation.
    lines.append("## Interpretation (thesis-ready)")
    lines.append("")
    interp = _interpretation_paragraph(stats, n_reports, pct_with, pct_skip, pct_sent)
    lines.append(interp)
    lines.append("")
    lines.append(
        "_Definitions: a report is **eligible for LLM interpretation** when the rule layer "
        "produces at least one non-negation snippet (`direct_delir`, `indirect_symptom`, or "
        "`prophylaxis_or_risk`) or, where enabled, when a short report without snippets is "
        "forwarded as full text; reports with only negation evidence or no evidence are "
        "**skipped by the prefilter**. Counts are derived from the stored extraction metadata, "
        "not recomputed by re-running inference._"
    )
    return "\n".join(lines) + "\n"


def _interpretation_paragraph(
    stats: Dict[str, Any],
    n_reports: int,
    pct_with: Optional[float],
    pct_skip: Optional[float],
    pct_sent: Optional[float],
) -> str:
    rw = stats["reports_with_snippet"]
    skip = stats["reports_prefilter_skipped"]
    sent = stats["reports_sent_to_llm"]
    total_snip = stats["total_snippets"]

    type_counts: Dict[str, int] = stats["evidence_type_counts"]
    top = sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_str = ", ".join(f"{et} (n={c})" for et, c in top[:3]) if top else NA

    parts: List[str] = []
    parts.append(
        f"Of the {n_reports} reports in the frozen validation cohort, "
        f"{_fmt(rw)} ({_fmt_pct(pct_with)}) contained at least one rule-based "
        f"delirium-related evidence snippet."
    )
    if sent is not None and skip is not None:
        parts.append(
            f"After the negation/relevance prefilter, {_fmt(sent)} reports "
            f"({_fmt_pct(pct_sent)}) were eligible for LLM-based interpretation, "
            f"whereas {_fmt(skip)} reports ({_fmt_pct(pct_skip)}) were skipped before "
            f"any LLM call."
        )
    else:
        parts.append(
            "Prefilter-skip and LLM-eligibility counts are " + NA +
            " from the available files."
        )
    parts.append(
        f"In total, {_fmt(total_snip)} evidence snippets were extracted"
        + (f", most commonly from the categories {top_str}." if top else ".")
    )
    if stats["pct_length_reduction_sent"] is not None:
        parts.append(
            f"For reports forwarded to the LLM, the bounded evidence bundle reduced "
            f"the text length by approximately {stats['pct_length_reduction_sent']:.1f}% "
            f"relative to the original report text."
        )
    return " ".join(parts)


def write_csv(rows: List[Dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["metric", "value", "percent", "note"]).to_csv(OUT_CSV, index=False)


def write_md(text: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(text, encoding="utf-8")


def write_unavailable_outputs(searched: List[Path]) -> None:
    """No usable trace found: emit explicit 'not available' artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {"metric": "status", "value": NA, "percent": "", "note": "No prediction-trace file with extraction metadata was found."},
    ]
    for p in searched:
        rows.append({"metric": "searched_path", "value": str(p), "percent": "", "note": "missing or unreadable"})
    pd.DataFrame(rows, columns=["metric", "value", "percent", "note"]).to_csv(OUT_CSV, index=False)

    lines = [
        "# Rule-based Evidence Extraction — Frozen Delirium Validation Cohort",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Status: input data not available",
        "",
        "No prediction-trace file containing the rule-extraction metadata "
        "(`evidence_snippets`, `llm_text_reduction_method`, length columns) was found locally. "
        "These outputs live on the analysis server. Run this script there, or copy one of the "
        "following files into the corresponding path, then re-run:",
        "",
    ]
    for p in searched:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("_No statistics were computed. No values were guessed._")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Main ---------------------------------------------------------------------
def main() -> None:
    selected = select_trace()
    if selected is None:
        searched = [p for p, _, _ in TRACE_CANDIDATES]
        write_unavailable_outputs(searched)
        print("RULE-EXTRACTION STATS: input data NOT AVAILABLE locally.")
        print("Searched (none usable):")
        for p in searched:
            print(f"  - {p}")
        print(f"Wrote placeholder outputs:\n  {OUT_CSV}\n  {OUT_MD}")
        return

    df, trace_path, trace_label = selected
    stats = compute_stats(df)
    warnings: List[str] = list(stats["notes"])

    # Cohort denominators: prefer frozen cohort files, else fall back to trace.
    fz_reports, fz_patients, fz_notes = frozen_cohort_counts()
    warnings.extend(fz_notes)

    n_reports = stats["n_reports_trace"]
    reports_source = f"prediction trace ({trace_label})"
    if fz_reports is not None:
        reports_source = f"frozen cohort manual_report_labels_frozen.csv ({fz_reports} reports)"
        if fz_reports != n_reports:
            warnings.append(
                f"Trace has {n_reports} reports but frozen cohort has {fz_reports}; "
                f"statistics below are computed on the {n_reports} trace rows."
            )

    n_patients_display = fz_patients if fz_patients is not None else stats["n_patients_trace"]
    patients_source = (
        "frozen cohort file"
        if fz_patients is not None
        else f"prediction trace ({trace_label})"
    )

    # Flag partial / stub traces (e.g. a 1-row leftover) so numbers are not misread.
    if n_reports < EXPECTED_REPORTS and fz_reports is None:
        warnings.append(
            f"Trace contains only {n_reports} report(s) (< expected {EXPECTED_REPORTS}). "
            f"This appears to be a partial or stub trace. Run on the analysis server for the "
            f"full frozen cohort. All statistics below describe ONLY these {n_reports} report(s)."
        )

    rows = build_rows(stats, n_reports, n_patients_display, patients_source, reports_source)
    write_csv(rows)
    md = build_markdown(
        stats, n_reports, n_patients_display, patients_source, reports_source,
        trace_path, trace_label, warnings,
    )
    write_md(md)

    # Concise terminal summary.
    print("=" * 70)
    print("RULE-BASED EXTRACTION STATS — frozen delirium validation cohort")
    print("=" * 70)
    print(f"Trace source       : {trace_path} ({trace_label})")
    print(f"Patients           : {_fmt(n_patients_display)}  ({patients_source})")
    print(f"Reports            : {n_reports}  ({reports_source})")
    print(
        f"Reports w/ snippet : {_fmt(stats['reports_with_snippet'])} "
        f"({_fmt_pct(_pct(stats['reports_with_snippet'], n_reports))})"
    )
    print(
        f"Prefilter-skipped  : {_fmt(stats['reports_prefilter_skipped'])} "
        f"({_fmt_pct(_pct(stats['reports_prefilter_skipped'], n_reports))})"
    )
    print(
        f"Sent to LLM        : {_fmt(stats['reports_sent_to_llm'])} "
        f"({_fmt_pct(_pct(stats['reports_sent_to_llm'], n_reports))})"
    )
    print(f"Total snippets     : {_fmt(stats['total_snippets'])}")
    print(
        f"Snippets/report    : mean={_fmt(stats['snippets_mean'])} "
        f"median={_fmt(stats['snippets_median'])} "
        f"min={_fmt(stats['snippets_min'])} max={_fmt(stats['snippets_max'])}"
    )
    print(f"Text reduction     : {_fmt_pct(stats['pct_length_reduction_sent'])}")
    print("Evidence types     :")
    for et, count, pct in _evidence_type_table(stats):
        print(f"  - {et:<20} {count:>6}  {_fmt_pct(pct)}")
    if warnings:
        print("-" * 70)
        print("Warnings:")
        for w in warnings:
            print(f"  ! {w}")
    print("-" * 70)
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
