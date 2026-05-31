"""
Shared read-only trace logic for validation report ↔ prediction ↔ raw Berichte alignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.audit_validation_matching import (
    build_prediction_lookup,
    extract_evidence_search_strings,
    normalize_match_text,
    reconstruct_report_text_from_row,
    text_contains_phrase,
)
from src.analysis.export_presentation_examples import parse_evidence_snippets
from src.analysis.validation_cohort_reports import load_raw_included_report_spine
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.preprocessing.berichte_mapper import load_berichte_dataframe
from src.preprocessing.report_identity import (
    PIPELINE_BERICHT_COL,
    SOURCE_REPORT_ROW_ID_COL,
    assign_source_report_row_ids,
)

COHORT_TRACE_FIELDS: tuple[str, ...] = (
    "validation_patient_id",
    "validation_report_id",
    "PatientenID",
    SOURCE_REPORT_ROW_ID_COL,
    "bertyp",
    "berdat",
    "bericht",
    PIPELINE_BERICHT_COL,
    "status",
    "model_report_prediction",
    "evidence_snippets",
)

LABEL_TRACE_FIELDS: tuple[str, ...] = (
    "manual_report_ground_truth",
    "manual_comment",
)

PREDICTION_TRACE_FIELDS: tuple[str, ...] = (
    "PatientenID",
    SOURCE_REPORT_ROW_ID_COL,
    "bertyp",
    "berdat",
    "bericht",
    PIPELINE_BERICHT_COL,
    "klasse",
    "status",
    "llm_called",
    "skipped_reason",
    "evidence_snippets",
    "begruendung",
    "kontext",
    "delir_signale",
    "signalstaerke",
)

RAW_BERICHTE_TEXT_FIELDS: tuple[str, ...] = (
    "diag",
    "epikrise",
    "jetziges_leiden",
    "prozedere",
    "bericht",
    "text",
    "bername",
)


@dataclass
class ReportTrace:
    validation_report_id: str
    verdict: str = "MATCH_OK"
    issues: List[str] = field(default_factory=list)
    merge_strategy: str = ""
    merge_key: str = ""
    cohort_row: Optional[Dict[str, Any]] = None
    label_row: Optional[Dict[str, Any]] = None
    prediction_row: Optional[Dict[str, Any]] = None
    raw_berichte_row: Optional[Dict[str, Any]] = None
    raw_row_index: Optional[int] = None
    raw_report_text: str = ""
    evidence_checks: List[Dict[str, Any]] = field(default_factory=list)


def _series_to_dict(row: pd.Series, fields: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in fields:
        if f in row.index:
            val = row[f]
            if pd.isna(val):
                out[f] = ""
            else:
                out[f] = val
    return out


def _merge_key_tuple(row: pd.Series, merge_on: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(k, "")).strip() for k in merge_on)


def load_trace_inputs(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort missing: {cohort_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels missing: {labels_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions missing: {predictions_path}")

    cohort = pd.read_csv(cohort_path)
    labels = pd.read_csv(labels_path)
    preds = pd.read_csv(predictions_path)

    patient_ids = None
    if "PatientenID" in cohort.columns:
        patient_ids = sorted(cohort["PatientenID"].astype(str).unique())

    spine = load_raw_included_report_spine(berichte_path, patient_ids=patient_ids)

    if berichte_path.exists():
        raw_full = assign_source_report_row_ids(load_berichte_dataframe(berichte_path))
    else:
        raw_full = pd.DataFrame()

    return cohort, labels, preds, spine, raw_full


def lookup_raw_berichte_row(
    raw_full: pd.DataFrame,
    source_report_row_id: str,
    *,
    patienten_id: str = "",
    bertyp: str = "",
    berdat: str = "",
) -> Tuple[Optional[pd.Series], Optional[int]]:
    sid = str(source_report_row_id or "").strip()
    if sid.startswith("berichte_row_") and not raw_full.empty:
        try:
            idx = int(sid.rsplit("_", 1)[-1])
            if 0 <= idx < len(raw_full):
                return raw_full.iloc[idx], idx
        except ValueError:
            pass
        if SOURCE_REPORT_ROW_ID_COL in raw_full.columns:
            hit = raw_full[raw_full[SOURCE_REPORT_ROW_ID_COL].astype(str) == sid]
            if not hit.empty:
                return hit.iloc[0], int(hit.index[0]) if hit.index[0] is not None else None

    if raw_full.empty or not patienten_id:
        return None, None

    sub = raw_full[raw_full["PatientID"].astype(str).str.strip() == str(patienten_id).strip()]
    if bertyp and "bertyp" in sub.columns:
        sub = sub[sub["bertyp"].astype(str).str.strip() == str(bertyp).strip()]
    if berdat and "berdat" in sub.columns:
        sub = sub[sub["berdat"].astype(str).str.strip() == str(berdat).strip()]
    if len(sub) == 1:
        row = sub.iloc[0]
        idx = int(sub.index[0]) if sub.index[0] is not None else None
        return row, idx
    return None, None


def compare_field(name: str, left: Any, right: Any) -> Optional[str]:
    ls = str(left or "").strip()
    rs = str(right or "").strip()
    if ls and rs and ls != rs:
        return f"{name}_mismatch: cohort/pred '{ls}' vs '{rs}'"
    return None


def evaluate_evidence_against_raw(
    evidence_raw: object,
    raw_report_text: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    snippets = parse_evidence_snippets(evidence_raw)
    checks: List[Dict[str, Any]] = []
    issues: List[str] = []
    if not snippets:
        return checks, issues

    for search_str in extract_evidence_search_strings(snippets):
        found = text_contains_phrase(raw_report_text, search_str)
        checks.append(
            {
                "evidence_fragment": search_str,
                "found_in_raw_report": found,
            }
        )
        if not found:
            issues.append(f"evidence_not_in_raw_report: '{search_str}'")
    return checks, issues


def compute_trace_verdict(issues: List[str]) -> str:
    fail_markers = (
        "patientenid_mismatch",
        "source_report_row_id_mismatch",
        "evidence_not_in_raw_report",
        "no_prediction_row",
        "no_cohort_row",
    )
    lowered = [i.lower() for i in issues]
    if any(any(m in i for m in fail_markers) for i in lowered):
        return "MATCH_FAIL"
    if issues:
        return "MATCH_SUSPICIOUS"
    return "MATCH_OK"


def build_report_trace(
    validation_report_id: str,
    cohort: pd.DataFrame,
    labels: pd.DataFrame,
    preds: pd.DataFrame,
    spine: pd.DataFrame,
    raw_full: pd.DataFrame,
) -> ReportTrace:
    trace = ReportTrace(validation_report_id=validation_report_id)
    issues: List[str] = []

    if "validation_report_id" not in cohort.columns:
        issues.append("no_validation_report_id_column_in_cohort")
        trace.issues = issues
        trace.verdict = compute_trace_verdict(issues)
        return trace

    cohort_hits = cohort[cohort["validation_report_id"].astype(str) == validation_report_id]
    if cohort_hits.empty:
        issues.append(f"no_cohort_row: {validation_report_id}")
        trace.issues = issues
        trace.verdict = "MATCH_FAIL"
        return trace

    cohort_row = cohort_hits.iloc[0]
    trace.cohort_row = _series_to_dict(cohort_row, COHORT_TRACE_FIELDS)
    if PIPELINE_BERICHT_COL not in trace.cohort_row and "bericht" in cohort_row.index:
        trace.cohort_row[PIPELINE_BERICHT_COL] = cohort_row.get("bericht", "")

    if "validation_report_id" in labels.columns:
        lab_hits = labels[labels["validation_report_id"].astype(str) == validation_report_id]
        if not lab_hits.empty:
            trace.label_row = _series_to_dict(lab_hits.iloc[0], LABEL_TRACE_FIELDS)

    lookup, strategy, merge_on, _ = build_prediction_lookup(preds, spine)
    trace.merge_strategy = strategy
    if merge_on and all(k in cohort_row.index for k in merge_on):
        key = _merge_key_tuple(cohort_row, merge_on)
        trace.merge_key = "|".join(key)
        pred_row = lookup.get(key)
    else:
        pred_row = None
        issues.append("merge_keys_unavailable")

    status = str(cohort_row.get("status", "")).strip()
    if status == "missing_prediction":
        issues.append("cohort_status_missing_prediction")

    if pred_row is None and status != "missing_prediction":
        issues.append(f"no_prediction_row_for_merge_key: {trace.merge_key}")
    elif pred_row is not None:
        trace.prediction_row = _series_to_dict(pred_row, PREDICTION_TRACE_FIELDS)
        for field in ("PatientenID", SOURCE_REPORT_ROW_ID_COL, "bertyp", "berdat"):
            msg = compare_field(field, cohort_row.get(field), pred_row.get(field))
            if msg:
                issues.append(msg)

    sid = str(cohort_row.get(SOURCE_REPORT_ROW_ID_COL, "")).strip()
    raw_row, raw_idx = lookup_raw_berichte_row(
        raw_full,
        sid,
        patienten_id=str(cohort_row.get("PatientenID", "")),
        bertyp=str(cohort_row.get("bertyp", "")),
        berdat=str(cohort_row.get("berdat", "")),
    )
    trace.raw_row_index = raw_idx
    if raw_row is not None:
        raw_dict: Dict[str, Any] = {"raw_row_index": raw_idx}
        if "PatientID" in raw_row.index:
            raw_dict["PatientID"] = raw_row.get("PatientID", "")
        for f in ("bertyp", "berdat") + RAW_BERICHTE_TEXT_FIELDS:
            if f in raw_row.index:
                raw_dict[f] = raw_row.get(f, "")
        if SOURCE_REPORT_ROW_ID_COL in raw_row.index:
            raw_dict[SOURCE_REPORT_ROW_ID_COL] = raw_row.get(SOURCE_REPORT_ROW_ID_COL, "")
        trace.raw_berichte_row = raw_dict
        trace.raw_report_text = reconstruct_report_text_from_row(raw_row)
    else:
        issues.append("raw_berichte_row_not_found")

    if pred_row is not None and trace.raw_report_text:
        pred_sid = str(pred_row.get(SOURCE_REPORT_ROW_ID_COL, "")).strip()
        if sid and pred_sid and sid != pred_sid:
            issues.append(
                f"source_report_row_id_mismatch: cohort '{sid}' vs prediction '{pred_sid}'"
            )

    evidence_source = cohort_row.get("evidence_snippets", "")
    if pred_row is not None and not str(evidence_source or "").strip().strip("[]"):
        evidence_source = pred_row.get("evidence_snippets", evidence_source)

    ev_checks, ev_issues = evaluate_evidence_against_raw(
        evidence_source, trace.raw_report_text
    )
    trace.evidence_checks = ev_checks
    if status != "missing_prediction":
        issues.extend(ev_issues)

    trace.issues = issues
    trace.verdict = compute_trace_verdict(issues)
    return trace


def format_trace_report(trace: ReportTrace) -> str:
    lines = [
        "Validation report trace",
        "=" * 44,
        f"validation_report_id={trace.validation_report_id}",
        f"verdict={trace.verdict}",
        f"merge_strategy={trace.merge_strategy}",
        f"merge_key={trace.merge_key}",
        "",
    ]

    if trace.issues:
        lines.append("Issues")
        lines.append("-" * 44)
        for issue in trace.issues:
            lines.append(f"  - {issue}")
        lines.append("")

    lines.append("1. Frozen cohort row")
    lines.append("-" * 44)
    lines.extend(_format_section(trace.cohort_row))

    lines.append("2. Manual label row")
    lines.append("-" * 44)
    lines.extend(_format_section(trace.label_row or {"note": "(no label row)"}))

    lines.append("3. Matched prediction row")
    lines.append("-" * 44)
    lines.extend(_format_section(trace.prediction_row or {"note": "(no prediction row)"}))

    lines.append("4. Raw Berichte row")
    lines.append("-" * 44)
    lines.extend(_format_section(trace.raw_berichte_row or {"note": "(raw row not found)"}))

    lines.append("5. Text comparison")
    lines.append("-" * 44)
    raw_norm = normalize_match_text(trace.raw_report_text)
    lines.append(f"normalized_raw_report_text_length={len(raw_norm)}")
    lines.append(f"normalized_raw_report_text_preview={raw_norm[:500]}")
    lines.append("")
    if trace.evidence_checks:
        lines.append("evidence_fragment_checks:")
        for chk in trace.evidence_checks:
            flag = "YES" if chk.get("found_in_raw_report") else "NO"
            lines.append(f"  [{flag}] {chk.get('evidence_fragment')}")
    else:
        lines.append("evidence_fragment_checks: (none)")

    pred_ev = ""
    if trace.prediction_row:
        pred_ev = str(trace.prediction_row.get("evidence_snippets", ""))
    lines.append("")
    lines.append(f"prediction_evidence_snippets={pred_ev[:800]}")

    lines.extend(["", "6. Verdict", "-" * 44, trace.verdict, ""])
    return "\n".join(lines)


def _format_section(data: Optional[Dict[str, Any]]) -> List[str]:
    if not data:
        return ["  (empty)"]
    out: List[str] = []
    for key, val in data.items():
        s = str(val)
        if key == "evidence_snippets" and len(s) > 400:
            s = s[:400] + "..."
        out.append(f"  {key}: {s}")
    out.append("")
    return out


def trace_to_mismatch_record(trace: ReportTrace) -> Dict[str, Any]:
    missing_evidence = [
        c["evidence_fragment"]
        for c in trace.evidence_checks
        if not c.get("found_in_raw_report")
    ]
    return {
        "validation_report_id": trace.validation_report_id,
        "verdict": trace.verdict,
        "validation_patient_id": (trace.cohort_row or {}).get("validation_patient_id", ""),
        "PatientenID": (trace.cohort_row or {}).get("PatientenID", ""),
        "model_report_prediction": (trace.cohort_row or {}).get("model_report_prediction", ""),
        "status": (trace.cohort_row or {}).get("status", ""),
        "merge_strategy": trace.merge_strategy,
        "merge_key": trace.merge_key,
        "issues": " | ".join(trace.issues),
        "missing_evidence_fragments": " | ".join(missing_evidence),
    }
