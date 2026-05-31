"""
Strict audit: validation cohort / manual labels / predictions / raw Berichte alignment.

Read-only: does not modify predictions, manual labels, or frozen files.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.export_presentation_examples import parse_evidence_snippets
from src.analysis.validation_cohort_reports import (
    _prepare_predictions_for_merge,
    load_raw_included_report_spine,
)
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    MATCHING_AUDIT_DIR,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.preprocessing.berichte_mapper import _row_blocks
from src.pipeline.frozen_cohort_inference import (
    build_stable_report_text_index,
    resolve_frozen_cohort_report_text,
)
from src.pipeline.validation_report_identity import (
    VALIDATION_REPORT_ID_COL,
    check_cohort_prediction_alignment,
)
from src.preprocessing.report_identity import (
    PIPELINE_BERICHT_COL,
    SOURCE_REPORT_ROW_ID_COL,
    choose_prediction_merge_keys,
)

LOGGER = logging.getLogger(__name__)

HIGH_RISK_DELIR_PHRASES: tuple[str, ...] = (
    "hypoaktives delir",
    "hyperaktives delir",
    "delirant",
    "delirös",
    "deliroes",
    "delir",
)

TRIVIAL_EVIDENCE_KEYWORDS: frozenset[str] = frozenset(
    {
        "orientierung",
        "desorientierung",
        "vigilanz",
        "somnolent",
        "prophylaxe",
    }
)

_SECTION_FIELDS = ("diag", "epikrise", "jetziges_leiden", "prozedere")


@dataclass
class AuditResult:
    total_cohort_rows: int = 0
    total_prediction_rows: int = 0
    matched_prediction_rows: int = 0
    missing_predictions: int = 0
    evidence_rows_checked: int = 0
    evidence_not_found_count: int = 0
    evidence_trivial_not_found_count: int = 0
    high_risk_mismatch_count: int = 0
    patient_report_count_mismatches: int = 0
    duplicate_key_issues: int = 0
    patientenid_label_mismatches: int = 0
    prediction_integrity_failures: int = 0
    raw_text_mismatch_count: int = 0
    verdict: str = "PASS"
    duplicate_keys: List[Dict[str, Any]] = field(default_factory=list)
    patientenid_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    prediction_integrity: List[Dict[str, Any]] = field(default_factory=list)
    evidence_not_found: List[Dict[str, Any]] = field(default_factory=list)
    high_risk_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    patient_count_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    raw_text_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    sample_mismatch_cases: List[Dict[str, Any]] = field(default_factory=list)


def normalize_match_text(text: object) -> str:
    s = str(text or "").lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return re.sub(r"\s+", " ", s).strip()


def reconstruct_report_text_from_row(row: pd.Series) -> str:
    row_dict = {c: row.get(c, "") for c in _SECTION_FIELDS if c in row.index}
    if not row_dict:
        row_dict = {c: row.get(c, "") for c in _SECTION_FIELDS}
    text = _row_blocks(row_dict)
    return text or ""


def text_contains_phrase(haystack: str, needle: str) -> bool:
    h = normalize_match_text(haystack)
    n = normalize_match_text(needle)
    if not n or not h:
        return False
    if n in h:
        return True
    if len(n) > 20:
        core = n[: min(40, len(n))]
        return core in h
    return False


def extract_evidence_search_strings(snippets: List[Dict[str, Any]]) -> List[str]:
    strings: List[str] = []
    for snip in snippets:
        for key in ("keyword", "text"):
            val = str(snip.get(key) or "").strip()
            if len(val) >= 3:
                strings.append(val)
    seen: set[str] = set()
    out: List[str] = []
    for s in strings:
        norm = normalize_match_text(s)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(s)
    return out


def is_high_risk_delir_phrase(text: str) -> bool:
    low = normalize_match_text(text)
    return any(phrase in low for phrase in HIGH_RISK_DELIR_PHRASES)


def is_trivial_evidence_miss(keyword_or_text: str) -> bool:
    low = normalize_match_text(keyword_or_text)
    if is_high_risk_delir_phrase(low):
        return False
    if any(t in low for t in TRIVIAL_EVIDENCE_KEYWORDS):
        return True
    return len(low) < 8


def check_validation_report_id_uniqueness(
    cohort: pd.DataFrame, labels: pd.DataFrame
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for name, df in (("cohort", cohort), ("manual_labels", labels)):
        if "validation_report_id" not in df.columns:
            continue
        dup = df["validation_report_id"].astype(str).duplicated(keep=False)
        if dup.any():
            for rid in df.loc[dup, "validation_report_id"].astype(str).unique():
                issues.append(
                    {
                        "issue_type": "duplicate_validation_report_id",
                        "source": name,
                        "validation_report_id": rid,
                        "count": int(
                            (df["validation_report_id"].astype(str) == rid).sum()
                        ),
                    }
                )
    return issues


def check_patientenid_label_consistency(
    cohort: pd.DataFrame, labels: pd.DataFrame
) -> List[Dict[str, Any]]:
    if "validation_report_id" not in cohort.columns or "validation_report_id" not in labels.columns:
        return []
    lab = labels.drop_duplicates("validation_report_id", keep="first").set_index(
        "validation_report_id"
    )
    mismatches: List[Dict[str, Any]] = []
    for _, row in cohort.iterrows():
        rid = str(row.get("validation_report_id", ""))
        if rid not in lab.index:
            continue
        cohort_pid = str(row.get("PatientenID", "")).strip()
        label_pid = str(lab.loc[rid].get("PatientenID", "")).strip()
        if label_pid and cohort_pid and label_pid != cohort_pid:
            mismatches.append(
                {
                    "validation_report_id": rid,
                    "cohort_PatientenID": cohort_pid,
                    "labels_PatientenID": label_pid,
                }
            )
    return mismatches


def _merge_key_tuple(row: pd.Series, merge_on: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(k, "")).strip() for k in merge_on)


def build_prediction_lookup(
    preds: pd.DataFrame, spine: pd.DataFrame
) -> Tuple[Dict[Tuple[str, ...], pd.Series], str, List[str], List[Dict[str, Any]]]:
    merge_on, strategy = choose_prediction_merge_keys(spine, preds)
    preds_ready, _ = _prepare_predictions_for_merge(preds, spine, merge_on)
    lookup: Dict[Tuple[str, ...], pd.Series] = {}
    duplicates: List[Dict[str, Any]] = []
    for idx, row in preds_ready.iterrows():
        key = _merge_key_tuple(row, merge_on)
        if key in lookup:
            duplicates.append(
                {
                    "issue_type": "duplicate_prediction_merge_key",
                    "merge_strategy": strategy,
                    "merge_key": "|".join(key),
                }
            )
        else:
            lookup[key] = row
    return lookup, strategy, merge_on, duplicates


def check_duplicate_ambiguous_keys(
    cohort: pd.DataFrame,
    labels: pd.DataFrame,
    preds: pd.DataFrame,
    spine: pd.DataFrame,
) -> List[Dict[str, Any]]:
    issues = check_validation_report_id_uniqueness(cohort, labels)

    if SOURCE_REPORT_ROW_ID_COL in cohort.columns:
        dup = cohort[cohort[SOURCE_REPORT_ROW_ID_COL].astype(str).duplicated(keep=False)]
        for sid in dup[SOURCE_REPORT_ROW_ID_COL].astype(str).unique():
            if sid and sid.lower() not in ("nan", "none"):
                issues.append(
                    {
                        "issue_type": "duplicate_source_report_row_id_in_cohort",
                        "source_report_row_id": sid,
                        "count": int(
                            (cohort[SOURCE_REPORT_ROW_ID_COL].astype(str) == sid).sum()
                        ),
                    }
                )

    merge_on: List[str] = []
    if VALIDATION_REPORT_ID_COL in preds.columns:
        dup_p = preds[preds[VALIDATION_REPORT_ID_COL].astype(str).duplicated(keep=False)]
        for vid in dup_p[VALIDATION_REPORT_ID_COL].astype(str).unique():
            if vid and vid.lower() not in ("nan", "none"):
                issues.append(
                    {
                        "issue_type": "duplicate_validation_report_id_in_predictions",
                        "validation_report_id": vid,
                        "count": int(
                            (preds[VALIDATION_REPORT_ID_COL].astype(str) == vid).sum()
                        ),
                    }
                )
    else:
        _, _, merge_on, pred_dupes = build_prediction_lookup(preds, spine)
        issues.extend(pred_dupes)

    if merge_on and all(k in cohort.columns for k in merge_on):
        dup_c = cohort.duplicated(subset=list(merge_on), keep=False)
        if dup_c.any():
            for _, row in cohort.loc[dup_c, list(merge_on)].drop_duplicates().iterrows():
                issues.append(
                    {
                        "issue_type": "duplicate_cohort_merge_key",
                        "merge_key": "|".join(str(row[k]) for k in merge_on),
                    }
                )
    return issues


def check_prediction_merge_integrity(
    cohort: pd.DataFrame, preds: pd.DataFrame, spine: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], int, int]:
    if (
        VALIDATION_REPORT_ID_COL in cohort.columns
        and VALIDATION_REPORT_ID_COL in preds.columns
        and preds[VALIDATION_REPORT_ID_COL].astype(str).str.strip().ne("").any()
    ):
        return _check_prediction_merge_by_validation_report_id(cohort, preds)

    failures: List[Dict[str, Any]] = []
    lookup, strategy, merge_on, _ = build_prediction_lookup(preds, spine)
    if not merge_on or not all(k in cohort.columns for k in merge_on):
        return failures, 0, 0

    matched = 0
    checked = 0
    for _, row in cohort.iterrows():
        status = str(row.get("status", "")).strip()
        if status == "missing_prediction":
            continue
        checked += 1
        key = _merge_key_tuple(row, merge_on)
        pred_row = lookup.get(key)
        if pred_row is None:
            failures.append(
                {
                    "validation_report_id": row.get("validation_report_id", ""),
                    "issue": "no_prediction_row_for_merge_key",
                    "merge_strategy": strategy,
                    "merge_key": "|".join(key),
                }
            )
            continue
        matched += 1
        checks = [
            ("PatientenID", row.get("PatientenID"), pred_row.get("PatientenID")),
            ("bertyp", row.get("bertyp"), pred_row.get("bertyp")),
            ("berdat", row.get("berdat"), pred_row.get("berdat")),
        ]
        if SOURCE_REPORT_ROW_ID_COL in row.index and SOURCE_REPORT_ROW_ID_COL in pred_row.index:
            checks.append(
                (
                    SOURCE_REPORT_ROW_ID_COL,
                    row.get(SOURCE_REPORT_ROW_ID_COL),
                    pred_row.get(SOURCE_REPORT_ROW_ID_COL),
                )
            )
        cohort_ber = str(row.get(PIPELINE_BERICHT_COL, row.get("bericht", ""))).strip()
        pred_ber = str(pred_row.get("bericht", pred_row.get(PIPELINE_BERICHT_COL, ""))).strip()
        if cohort_ber or pred_ber:
            checks.append(("bericht/pipeline_bericht", cohort_ber, pred_ber))

        for field_name, left, right in checks:
            ls = str(left or "").strip()
            rs = str(right or "").strip()
            if ls and rs and ls != rs:
                failures.append(
                    {
                        "validation_report_id": row.get("validation_report_id", ""),
                        "issue": f"field_mismatch_{field_name}",
                        "cohort_value": ls,
                        "prediction_value": rs,
                        "merge_strategy": strategy,
                    }
                )
    return failures, matched, checked


def _check_prediction_merge_by_validation_report_id(
    cohort: pd.DataFrame,
    preds: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], int, int]:
    failures: List[Dict[str, Any]] = []
    alignment_errors, _ = check_cohort_prediction_alignment(cohort, preds)
    for err in alignment_errors:
        failures.append(
            {
                "validation_report_id": "",
                "issue": err,
                "merge_strategy": "validation_report_id",
            }
        )

    pred_lookup = {
        str(row[VALIDATION_REPORT_ID_COL]).strip(): row
        for _, row in preds.iterrows()
        if str(row.get(VALIDATION_REPORT_ID_COL, "")).strip()
    }

    matched = 0
    checked = 0
    for _, row in cohort.iterrows():
        status = str(row.get("status", "")).strip()
        if status == "missing_prediction":
            continue
        checked += 1
        vid = str(row.get(VALIDATION_REPORT_ID_COL, "")).strip()
        if not vid:
            failures.append(
                {
                    "validation_report_id": "",
                    "issue": "validation_report_id_missing_in_cohort",
                    "merge_strategy": "validation_report_id",
                }
            )
            continue
        pred_row = pred_lookup.get(vid)
        if pred_row is None:
            failures.append(
                {
                    "validation_report_id": vid,
                    "issue": "no_prediction_row_for_validation_report_id",
                    "merge_strategy": "validation_report_id",
                }
            )
            continue
        matched += 1
        for field in ("PatientenID", "bertyp", "berdat"):
            ls = str(row.get(field, "") or "").strip()
            rs = str(pred_row.get(field, "") or "").strip()
            if ls and rs and ls != rs:
                failures.append(
                    {
                        "validation_report_id": vid,
                        "issue": f"field_mismatch_{field}",
                        "cohort_value": ls,
                        "prediction_value": rs,
                        "merge_strategy": "validation_report_id",
                    }
                )
    return failures, matched, checked


def build_report_text_index(
    spine: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[Tuple[str, ...], str]]:
    by_source_id: Dict[str, str] = {}
    by_fallback: Dict[Tuple[str, ...], str] = {}
    for _, row in spine.iterrows():
        text = reconstruct_report_text_from_row(row)
        sid = str(row.get(SOURCE_REPORT_ROW_ID_COL, "")).strip()
        if sid and sid.lower() not in ("nan", "none"):
            by_source_id[sid] = text
        fb_key = (
            str(row.get("PatientenID", "")).strip(),
            str(row.get("bertyp", "")).strip(),
            str(row.get("berdat", "")).strip(),
        )
        if fb_key[0] and fb_key not in by_fallback:
            by_fallback[fb_key] = text
    return by_source_id, by_fallback


def resolve_raw_report_text(
    row: pd.Series,
    by_source_id: Dict[str, str],
    by_fallback: Dict[Tuple[str, ...], str],
) -> str:
    sid = str(row.get(SOURCE_REPORT_ROW_ID_COL, "")).strip()
    if sid and sid in by_source_id:
        return by_source_id[sid]
    fb_key = (
        str(row.get("PatientenID", "")).strip(),
        str(row.get("bertyp", "")).strip(),
        str(row.get("berdat", "")).strip(),
    )
    return by_fallback.get(fb_key, "")


def check_evidence_in_report(
    cohort: pd.DataFrame,
    by_source_id: Dict[str, str],
    by_fallback: Dict[Tuple[str, ...], str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int, int]:
    not_found: List[Dict[str, Any]] = []
    high_risk: List[Dict[str, Any]] = []
    checked = 0
    trivial_miss = 0
    text_index = build_stable_report_text_index()

    for _, row in cohort.iterrows():
        status = str(row.get("status", "")).strip()
        if status == "missing_prediction":
            continue
        raw_ev = row.get("evidence_snippets", "")
        snippets = parse_evidence_snippets(raw_ev)
        if not snippets:
            continue
        checked += 1
        report_text = resolve_frozen_cohort_report_text(row, text_index)
        if not report_text.strip():
            report_text = resolve_raw_report_text(row, by_source_id, by_fallback)
        if not report_text.strip():
            not_found.append(
                {
                    "validation_report_id": row.get("validation_report_id", ""),
                    "PatientenID": row.get("PatientenID", ""),
                    "issue": "raw_report_text_unavailable",
                    "evidence_snippets": str(raw_ev)[:200],
                }
            )
            continue

        for search_str in extract_evidence_search_strings(snippets):
            if text_contains_phrase(report_text, search_str):
                continue
            entry = {
                "validation_report_id": row.get("validation_report_id", ""),
                "PatientenID": row.get("PatientenID", ""),
                "source_report_row_id": row.get(SOURCE_REPORT_ROW_ID_COL, ""),
                "evidence_fragment": search_str,
                "status": status,
                "severity": "trivial" if is_trivial_evidence_miss(search_str) else "non_trivial",
            }
            not_found.append(entry)
            if is_trivial_evidence_miss(search_str):
                trivial_miss += 1
            if is_high_risk_delir_phrase(search_str):
                high_risk.append(
                    {
                        **entry,
                        "issue": "high_risk_delir_evidence_not_in_raw_report",
                    }
                )

    return not_found, high_risk, checked, len(not_found), trivial_miss


def check_patient_report_counts(
    cohort: pd.DataFrame, spine: pd.DataFrame
) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []
    if "PatientenID" not in cohort.columns:
        return mismatches
    cohort_counts = cohort.groupby("PatientenID").size()
    spine_counts = spine.groupby("PatientenID").size()
    for pid in sorted(set(cohort_counts.index.astype(str))):
        c_n = int(cohort_counts.get(pid, 0))
        s_n = int(spine_counts.get(pid, 0))
        if c_n != s_n:
            mismatches.append(
                {
                    "PatientenID": pid,
                    "cohort_report_rows": c_n,
                    "raw_berichte_included_rows": s_n,
                    "delta": c_n - s_n,
                }
            )
    return mismatches


def check_raw_text_excerpt_mismatch(
    cohort: pd.DataFrame,
    by_source_id: Dict[str, str],
    by_fallback: Dict[Tuple[str, ...], str],
) -> List[Dict[str, Any]]:
    """Flag when cohort excerpt columns strongly disagree with reconstructed raw text."""
    mismatches: List[Dict[str, Any]] = []
    excerpt_cols = [c for c in ("report_text", "llm_report_text", "bericht") if c in cohort.columns]
    if not excerpt_cols:
        return mismatches

    for _, row in cohort.iterrows():
        raw = resolve_raw_report_text(row, by_source_id, by_fallback)
        if not raw.strip():
            continue
        raw_norm = normalize_match_text(raw)
        for col in excerpt_cols:
            val = str(row.get(col, "") or "").strip()
            if not val or len(val) < 20:
                continue
            val_norm = normalize_match_text(val)
            if val_norm and val_norm not in raw_norm and raw_norm not in val_norm:
                prefix = val_norm[:40]
                if prefix and prefix not in raw_norm:
                    mismatches.append(
                        {
                            "validation_report_id": row.get("validation_report_id", ""),
                            "PatientenID": row.get("PatientenID", ""),
                            "column": col,
                            "issue": "cohort_excerpt_not_substring_of_raw_reconstruction",
                        }
                    )
                    break
    return mismatches


def compute_verdict(result: AuditResult) -> str:
    if result.high_risk_mismatch_count > 0:
        return "FAIL"
    non_trivial_evidence = (
        result.evidence_not_found_count - result.evidence_trivial_not_found_count
    )
    if non_trivial_evidence > 0:
        return "FAIL"
    if (
        result.patient_report_count_mismatches > 0
        or result.duplicate_key_issues > 0
        or result.patientenid_label_mismatches > 0
        or result.prediction_integrity_failures > 0
    ):
        return "WARNING"
    if result.evidence_trivial_not_found_count > 0 or result.raw_text_mismatch_count > 0:
        return "WARNING"
    return "PASS"


def format_audit_report(result: AuditResult) -> str:
    lines = [
        "Validation matching audit",
        "=" * 44,
        "",
        f"total_cohort_rows={result.total_cohort_rows}",
        f"total_prediction_rows={result.total_prediction_rows}",
        f"matched_prediction_rows={result.matched_prediction_rows}",
        f"missing_predictions={result.missing_predictions}",
        f"evidence_rows_checked={result.evidence_rows_checked}",
        f"evidence_not_found_count={result.evidence_not_found_count}",
        f"evidence_trivial_not_found_count={result.evidence_trivial_not_found_count}",
        f"high_risk_mismatch_count={result.high_risk_mismatch_count}",
        f"patient_report_count_mismatches={result.patient_report_count_mismatches}",
        f"duplicate_or_ambiguous_keys={result.duplicate_key_issues}",
        f"patientenid_label_mismatches={result.patientenid_label_mismatches}",
        f"prediction_integrity_failures={result.prediction_integrity_failures}",
        f"raw_text_mismatch_count={result.raw_text_mismatch_count}",
        "",
        f"VERDICT: {result.verdict}",
        "",
        "Interpretation",
        "-" * 44,
        "PASS: evidence aligns with raw reports; no high-risk delir leakage; counts match.",
        "WARNING: minor evidence parsing gaps or non-critical alignment issues.",
        "FAIL: delir-related evidence in predictions not found in the matched raw report.",
    ]
    return "\n".join(lines) + "\n"


def run_matching_audit(
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    output_dir: Path = MATCHING_AUDIT_DIR,
) -> AuditResult:
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort missing: {cohort_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Manual labels missing: {labels_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions missing: {predictions_path}")

    cohort = pd.read_csv(cohort_path)
    labels = pd.read_csv(labels_path)
    preds = pd.read_csv(predictions_path)

    patient_ids = sorted(cohort["PatientenID"].astype(str).unique()) if "PatientenID" in cohort.columns else None
    spine = load_raw_included_report_spine(berichte_path, patient_ids=patient_ids)
    by_source_id, by_fallback = build_report_text_index(spine)

    result = AuditResult(
        total_cohort_rows=len(cohort),
        total_prediction_rows=len(preds),
        missing_predictions=int((cohort.get("status", pd.Series(dtype=str)).astype(str) == "missing_prediction").sum())
        if "status" in cohort.columns
        else 0,
    )

    dup_issues = check_duplicate_ambiguous_keys(cohort, labels, preds, spine)
    result.duplicate_keys = dup_issues
    result.duplicate_key_issues = len(dup_issues)

    result.patientenid_mismatches = check_patientenid_label_consistency(cohort, labels)
    result.patientenid_label_mismatches = len(result.patientenid_mismatches)

    pred_failures, matched, _ = check_prediction_merge_integrity(cohort, preds, spine)
    result.prediction_integrity = pred_failures
    result.prediction_integrity_failures = len(pred_failures)
    result.matched_prediction_rows = matched

    ev_nf, high_risk, ev_checked, ev_nf_count, trivial = check_evidence_in_report(
        cohort, by_source_id, by_fallback
    )
    result.evidence_not_found = ev_nf
    result.high_risk_mismatches = high_risk
    result.evidence_rows_checked = ev_checked
    result.evidence_not_found_count = ev_nf_count
    result.evidence_trivial_not_found_count = trivial
    result.high_risk_mismatch_count = len(high_risk)

    result.patient_count_mismatches = check_patient_report_counts(cohort, spine)
    result.patient_report_count_mismatches = len(result.patient_count_mismatches)

    result.raw_text_mismatches = check_raw_text_excerpt_mismatch(
        cohort, by_source_id, by_fallback
    )
    result.raw_text_mismatch_count = len(result.raw_text_mismatches)

    samples: List[Dict[str, Any]] = []
    samples.extend(result.high_risk_mismatches[:10])
    samples.extend(
        [r for r in result.evidence_not_found if r.get("severity") == "non_trivial"][:10]
    )
    samples.extend(result.prediction_integrity[:5])
    result.sample_mismatch_cases = samples[:25]

    result.verdict = compute_verdict(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matching_audit_report.txt").write_text(
        format_audit_report(result), encoding="utf-8"
    )
    pd.DataFrame(result.evidence_not_found).to_csv(
        output_dir / "evidence_not_found_in_report.csv", index=False
    )
    pd.DataFrame(result.duplicate_keys).to_csv(
        output_dir / "duplicate_or_ambiguous_report_keys.csv", index=False
    )
    pd.DataFrame(result.patient_count_mismatches).to_csv(
        output_dir / "patient_report_count_mismatches.csv", index=False
    )
    pd.DataFrame(result.sample_mismatch_cases).to_csv(
        output_dir / "sample_mismatch_cases.csv", index=False
    )

    LOGGER.info("Matching audit verdict=%s output=%s", result.verdict, output_dir)
    return result


def main() -> None:
    result = run_matching_audit()
    print(format_audit_report(result))
    print(f"Wrote audit outputs to {MATCHING_AUDIT_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
