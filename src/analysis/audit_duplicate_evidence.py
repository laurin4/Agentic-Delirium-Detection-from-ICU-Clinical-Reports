"""
Read-only audit: duplicate evidence text across different patients.

Does not modify predictions, manual labels, frozen cohort, or evaluation exports.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.audit_validation_matching import (
    extract_evidence_search_strings,
    normalize_match_text,
    text_contains_phrase,
)
from src.analysis.export_presentation_examples import parse_evidence_snippets
from src.pipeline.frozen_cohort_inference import (
    build_stable_report_text_index,
    resolve_frozen_cohort_report_text,
)
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    FINAL_MANUAL_VALIDATION_EVAL_DIR,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.pipeline.validation_report_identity import VALIDATION_REPORT_ID_COL
from src.preprocessing.berichte_filters import normalize_bertyp

LOGGER = logging.getLogger(__name__)

MODEL_FP_PATH = FINAL_MANUAL_VALIDATION_EVAL_DIR / "model_FP.csv"
MODEL_TP_PATH = FINAL_MANUAL_VALIDATION_EVAL_DIR / "model_TP.csv"
DUPLICATE_EVIDENCE_AUDIT_CSV = FINAL_MANUAL_VALIDATION_EVAL_DIR / "duplicate_evidence_audit.csv"
DUPLICATE_EVIDENCE_AUDIT_REPORT = FINAL_MANUAL_VALIDATION_EVAL_DIR / "duplicate_evidence_audit_report.txt"

MIN_EVIDENCE_LEN = 25
TEMPLATE_MARKERS: tuple[str, ...] = (
    "nach extubation",
    "unauffaellige atemmechanik",
    "unauffällige atemmechanik",
    "atemmechanik",
    "verlauf unauffaellig",
    "ohne hinweis auf delir",
)

AUDIT_DETAIL_COLUMNS: tuple[str, ...] = (
    "duplicate_group_id",
    "verdict_group",
    "source_file",
    "validation_patient_id",
    "validation_report_id",
    "PatientenID",
    "bertyp",
    "berdat",
    "model_prediction",
    "evidence_text",
    "evidence_raw_excerpt",
    "in_frozen_cohort_text",
    "in_berichte_for_patient",
    "likely_template_text",
    "notes",
)


def _norm_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _primary_evidence_text(raw: object) -> str:
    """Best single phrase for duplicate grouping (snippet text preferred)."""
    snippets = parse_evidence_snippets(raw)
    if not snippets:
        text = _norm_id(raw)
        if len(text) >= MIN_EVIDENCE_LEN and not text.startswith("["):
            return text
        return ""

    for snip in snippets:
        body = _norm_id(snip.get("text"))
        if len(body) >= MIN_EVIDENCE_LEN:
            return body

    strings = extract_evidence_search_strings(snippets)
    if strings:
        return max(strings, key=len)

    return _norm_id(snippets[0].get("keyword", "")) if snippets else ""


def _evidence_group_key(text: str) -> str:
    norm = normalize_match_text(text)
    if len(norm) < MIN_EVIDENCE_LEN:
        return ""
    return norm[:200]


def _group_id_from_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"dup_{digest}"


def _is_template_like(text: str) -> bool:
    norm = normalize_match_text(text)
    return any(marker in norm for marker in TEMPLATE_MARKERS)


def _berichte_texts_for_patient(
    patienten_id: str,
    text_index: Dict[Tuple[str, str, str, str], str],
) -> List[str]:
    pid = _norm_id(patienten_id)
    return [text for key, text in text_index.items() if key[0] == pid and text.strip()]


def _phrase_in_any(haystacks: Sequence[str], phrase: str) -> bool:
    return any(text_contains_phrase(h, phrase) for h in haystacks if h)


def _collect_patient_level_rows(
    path: Path,
    source_name: str,
) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = row.get("representative_evidence", "")
        text = _primary_evidence_text(raw)
        if not text:
            continue
        rows.append(
            {
                "source_file": source_name,
                "validation_patient_id": _norm_id(row.get("validation_patient_id")),
                "validation_report_id": "",
                "PatientenID": _norm_id(row.get("PatientenID")),
                "bertyp": "",
                "berdat": "",
                "model_prediction": _norm_id(
                    row.get("model_patient_positive", row.get("model_report_prediction", ""))
                ),
                "evidence_text": text,
                "evidence_raw_excerpt": _norm_id(raw)[:300],
            }
        )
    return rows


def _collect_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = row.get("evidence_snippets", "")
        text = _primary_evidence_text(raw)
        if not text:
            continue
        klasse = pd.to_numeric(row.get("klasse"), errors="coerce")
        rows.append(
            {
                "source_file": "validation_cohort_predictions.csv",
                "validation_patient_id": _norm_id(row.get("validation_patient_id")),
                "validation_report_id": _norm_id(row.get(VALIDATION_REPORT_ID_COL)),
                "PatientenID": _norm_id(row.get("PatientenID")),
                "bertyp": normalize_bertyp(row.get("bertyp", "")),
                "berdat": _norm_id(row.get("berdat")),
                "model_prediction": "" if pd.isna(klasse) else str(int(klasse)),
                "evidence_text": text,
                "evidence_raw_excerpt": _norm_id(raw)[:300],
            }
        )
    return rows


def _enrich_with_source_checks(
    record: Dict[str, Any],
    cohort: pd.DataFrame,
    text_index: Dict[Tuple[str, str, str, str], str],
) -> Dict[str, Any]:
    phrase = record["evidence_text"]
    vid = record["validation_report_id"]
    pid = record["PatientenID"]

    cohort_text = ""
    if vid and not cohort.empty and VALIDATION_REPORT_ID_COL in cohort.columns:
        hits = cohort[cohort[VALIDATION_REPORT_ID_COL].astype(str) == vid]
        if not hits.empty:
            cohort_text = resolve_frozen_cohort_report_text(hits.iloc[0], text_index)

    berichte_texts = _berichte_texts_for_patient(pid, text_index) if pid else []

    in_cohort = _phrase_in_any([cohort_text], phrase) if cohort_text else False
    in_berichte = _phrase_in_any(berichte_texts, phrase) if berichte_texts else False

    notes: List[str] = []
    if not vid:
        notes.append("patient_level_export_no_report_id")
    if not cohort_text and vid:
        notes.append("cohort_report_text_empty_or_unresolved")
    if not berichte_texts and pid:
        notes.append("no_berichte_text_for_patient")

    out = dict(record)
    out["in_frozen_cohort_text"] = in_cohort
    out["in_berichte_for_patient"] = in_berichte
    out["likely_template_text"] = _is_template_like(phrase)
    out["notes"] = "; ".join(notes)
    return out


def _verdict_for_group(details: List[Dict[str, Any]]) -> str:
    patient_ids = {_norm_id(r["PatientenID"]) for r in details if _norm_id(r["PatientenID"])}
    if len(patient_ids) < 2:
        return "PASS"

    all_in_own_source = all(
        bool(r["in_frozen_cohort_text"]) or bool(r["in_berichte_for_patient"]) for r in details
    )
    any_missing = any(
        not bool(r["in_frozen_cohort_text"]) and not bool(r["in_berichte_for_patient"])
        for r in details
    )
    template_like = all(bool(r.get("likely_template_text")) for r in details)

    if any_missing:
        return "FAIL"
    if template_like or _is_template_like(details[0].get("evidence_text", "")):
        return "WARNING"
    if all_in_own_source:
        return "PASS"
    return "WARNING"


def run_duplicate_evidence_audit(
    fp_path: Path = MODEL_FP_PATH,
    tp_path: Path = MODEL_TP_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
) -> Tuple[pd.DataFrame, str]:
    cohort = pd.read_csv(cohort_path) if cohort_path.exists() else pd.DataFrame()
    text_index = build_stable_report_text_index(berichte_path)

    occurrences: List[Dict[str, Any]] = []
    occurrences.extend(_collect_patient_level_rows(fp_path, "model_FP.csv"))
    occurrences.extend(_collect_patient_level_rows(tp_path, "model_TP.csv"))
    occurrences.extend(_collect_prediction_rows(predictions_path))

    # Report-level rows from predictions with klasse==1 for FP/TP cross-check
    pos_pred_rows: List[Dict[str, Any]] = []
    if predictions_path.exists():
        pdf = pd.read_csv(predictions_path)
        if "klasse" in pdf.columns:
            pdf["_k"] = pd.to_numeric(pdf["klasse"], errors="coerce").fillna(0).astype(int)
            for _, row in pdf[pdf["_k"] == 1].iterrows():
                raw = row.get("evidence_snippets", "")
                text = _primary_evidence_text(raw)
                if not text:
                    continue
                pos_pred_rows.append(
                    {
                        "source_file": "validation_cohort_predictions.csv (klasse=1)",
                        "validation_patient_id": _norm_id(row.get("validation_patient_id")),
                        "validation_report_id": _norm_id(row.get(VALIDATION_REPORT_ID_COL)),
                        "PatientenID": _norm_id(row.get("PatientenID")),
                        "bertyp": normalize_bertyp(row.get("bertyp", "")),
                        "berdat": _norm_id(row.get("berdat")),
                        "model_prediction": "1",
                        "evidence_text": text,
                        "evidence_raw_excerpt": _norm_id(raw)[:300],
                    }
                )

    # Prefer report-level positive predictions for source checks; merge patient-level for discovery
    check_rows = pos_pred_rows if pos_pred_rows else occurrences

    enriched: List[Dict[str, Any]] = []
    for rec in occurrences:
        enriched.append(_enrich_with_source_checks(rec, cohort, text_index))

    # Re-enrich positive prediction rows for accurate per-report checks
    enriched_pos: List[Dict[str, Any]] = []
    for rec in pos_pred_rows:
        enriched_pos.append(_enrich_with_source_checks(rec, cohort, text_index))

    # Group duplicates on all occurrences (patient + report level)
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for rec in enriched:
        key = _evidence_group_key(rec["evidence_text"])
        if not key:
            continue
        by_key.setdefault(key, []).append(rec)

    detail_rows: List[Dict[str, Any]] = []
    group_verdicts: Dict[str, str] = {}

    for key, group in by_key.items():
        pids = {_norm_id(r["PatientenID"]) for r in group if _norm_id(r["PatientenID"])}
        if len(pids) < 2:
            continue

        gid = _group_id_from_key(key)
        # Use report-level enriched rows when available for verdict
        report_level = [
            r
            for r in enriched_pos
            if _evidence_group_key(r["evidence_text"]) == key
        ]
        verdict_src = report_level if len(report_level) >= 2 else group
        verdict = _verdict_for_group(verdict_src)
        group_verdicts[gid] = verdict

        for rec in group:
            detail_rows.append({**rec, "duplicate_group_id": gid, "verdict_group": verdict})

    audit_df = pd.DataFrame(detail_rows, columns=list(AUDIT_DETAIL_COLUMNS))
    report = _format_audit_report(audit_df, group_verdicts, occurrences, enriched_pos)
    return audit_df, report


def _format_audit_report(
    audit_df: pd.DataFrame,
    group_verdicts: Dict[str, str],
    all_occurrences: List[Dict[str, Any]],
    positive_preds: List[Dict[str, Any]],
) -> str:
    lines = [
        "Duplicate evidence audit (cross-patient)",
        "=" * 56,
        f"occurrences_scanned={len(all_occurrences)}",
        f"positive_prediction_rows={len(positive_preds)}",
        f"duplicate_groups={len(group_verdicts)}",
        "",
    ]

    if group_verdicts:
        fail_n = sum(1 for v in group_verdicts.values() if v == "FAIL")
        warn_n = sum(1 for v in group_verdicts.values() if v == "WARNING")
        pass_n = sum(1 for v in group_verdicts.values() if v == "PASS")
        affected_patients: set[str] = set()
        if not audit_df.empty and "PatientenID" in audit_df.columns:
            affected_patients = set(audit_df["PatientenID"].astype(str).map(_norm_id)) - {""}

        lines.extend(
            [
                "Verdict summary",
                "-" * 56,
                f"FAIL_groups={fail_n}",
                f"WARNING_groups={warn_n}",
                f"PASS_groups={pass_n}",
                f"affected_patients={len(affected_patients)}",
                "",
            ]
        )

        overall = "FAIL" if fail_n else ("WARNING" if warn_n else "PASS")
        lines.append(f"OVERALL_VERDICT: {overall}")
        lines.append("")
        lines.extend(
            [
                "Interpretation",
                "-" * 56,
                "- PASS: same text appears in each patient's own cohort/Berichte text (real duplicate clinical wording).",
                "- WARNING: duplicate text is present in sources but looks like shared template/boilerplate.",
                "- FAIL: same exported evidence text for a patient/report NOT found in that patient's source text",
                "  (suggests wrong merge, copy, or export aggregation bug).",
                "",
                "Final evaluation metrics impact",
                "-" * 56,
                "Patient-level metrics (TP/FP/FN) use model_patient_positive / klasse, not representative_evidence text.",
                "Duplicate representative_evidence strings in model_FP.csv affect qualitative FP review only,",
                "unless predictions were merged to the wrong validation_report_id.",
                "",
            ]
        )

        for gid, verdict in sorted(group_verdicts.items(), key=lambda x: x[0]):
            sub = audit_df[audit_df["duplicate_group_id"] == gid] if not audit_df.empty else audit_df
            lines.append(f"Group {gid} verdict={verdict}")
            if not sub.empty:
                preview = sub.iloc[0].get("evidence_text", "")
                lines.append(f"  text_preview={str(preview)[:120]}")
                for _, r in sub.drop_duplicates(
                    subset=["PatientenID", "validation_report_id", "source_file"]
                ).iterrows():
                    lines.append(
                        f"  - {r.get('source_file')} PatientenID={r.get('PatientenID')} "
                        f"vid={r.get('validation_patient_id')} rid={r.get('validation_report_id')} "
                        f"in_cohort={r.get('in_frozen_cohort_text')} in_berichte={r.get('in_berichte_for_patient')}"
                    )
            lines.append("")
    else:
        lines.append("No cross-patient duplicate evidence text detected (above minimum length).")
        lines.append("OVERALL_VERDICT: PASS")
        lines.append("")

    return "\n".join(lines)


def write_duplicate_evidence_audit(
    fp_path: Path = MODEL_FP_PATH,
    tp_path: Path = MODEL_TP_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    output_csv: Path = DUPLICATE_EVIDENCE_AUDIT_CSV,
    output_report: Path = DUPLICATE_EVIDENCE_AUDIT_REPORT,
) -> Tuple[pd.DataFrame, str]:
    audit_df, report = run_duplicate_evidence_audit(
        fp_path, tp_path, predictions_path, cohort_path, berichte_path
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_csv, index=False)
    output_report.write_text(report, encoding="utf-8")
    LOGGER.info(
        "Wrote duplicate evidence audit: %s (%d rows, %d groups)",
        output_csv,
        len(audit_df),
        audit_df["duplicate_group_id"].nunique() if not audit_df.empty else 0,
    )
    return audit_df, report


def main() -> None:
    _, report = write_duplicate_evidence_audit()
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
