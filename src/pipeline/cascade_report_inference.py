"""
Single-report inference for cascade stages (V1/V2) with explicit prompt version.

Reuses the same evidence → Agent1 → Agent2 → guardrails path as run_pipeline.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List

from src.agents.cascade_reviewer import interpret_cascade_reviewer
from src.agents.classification import classify_delirium
from src.agents.clinical_guardrails import apply_clinical_decision_guardrails
from src.agents.extraction import extract_passages
from src.agents.delirium_probability import delirium_probability_estimate
from src.agents.interpretation_llm import interpret_signals_llm
from src.pipeline.prompt_run_paths import normalize_prompt_version
from src.pipeline.run_pipeline import (
    SIGNAL_KEYS,
    _base_evidence_metadata,
    _guardrail_fields,
    _prediction_row_no_evidence,
    _prediction_row_pipeline_error,
    _processing_status_fields,
    _report_identity_fields,
    resolve_bertyp,
)
from src.pipeline.validation_report_identity import VALIDATION_REPORT_ID_COL
from src.preprocessing.evidence_extraction import (
    apply_short_report_fulltext_to_evidence,
    extract_delirium_evidence,
    llm_should_receive_evidence,
    should_send_short_report_without_evidence,
)

LOGGER = logging.getLogger(__name__)

STAGE2_MODE_V2 = "v2"
STAGE2_MODE_CASCADE_REVIEWER = "cascade_reviewer"
ALLOWED_STAGE2_MODES = frozenset({STAGE2_MODE_V2, STAGE2_MODE_CASCADE_REVIEWER})


def normalize_stage2_mode(mode: str) -> str:
    m = str(mode or STAGE2_MODE_V2).strip().lower()
    if m not in ALLOWED_STAGE2_MODES:
        allowed = ", ".join(sorted(ALLOWED_STAGE2_MODES))
        raise ValueError(f"Invalid stage2_mode='{mode}'. Allowed: {allowed}")
    return m


def _build_inference_row(
    *,
    report: dict,
    ev: Dict[str, Any],
    result: Dict[str, Any],
    interpretation: Dict[str, Any],
    guard: Dict[str, Any],
    classification: Dict[str, Any],
    patient_id: str,
    report_name: str,
    bertyp: str,
    prompt_version: str,
) -> Dict[str, Any]:
    final_klasse = int(guard["klasse"])
    final_signal = str(guard["signalstaerke"])
    final_kontext = str(guard.get("kontext") or interpretation.get("kontext", ""))
    final_begr = list(guard.get("begruendung") or [])
    klassifikation_begr_str = (
        " | ".join(str(x) for x in final_begr)
        if final_begr
        else " | ".join(classification.get("begruendung", []))
    )

    hits: List[str] = []
    for key in SIGNAL_KEYS:
        values = result.get(key, [])
        if isinstance(values, list):
            hits.extend(values)

    prob = delirium_probability_estimate(
        final_signal,
        final_klasse,
        manual_review_candidate=bool(guard.get("manual_review_candidate")),
        decision_rule_applied=str(guard.get("decision_rule_applied", "")),
        has_direct_delir_evidence=bool(ev.get("has_direct_delir_evidence")),
    )
    return {
        "PatientenID": patient_id,
        "bericht": report_name,
        "bertyp": bertyp,
        "prompt_version": prompt_version,
        **_report_identity_fields(report),
        **_base_evidence_metadata(ev),
        **_guardrail_fields(guard),
        "llm_skipped_by_prefilter": False,
        "anzahl_treffer": len(hits),
        "delir_signale": " | ".join(hits),
        "signalstaerke": final_signal,
        "delir_probability_estimate": prob,
        "kontext": final_kontext,
        "alternative_erklaerung": guard.get(
            "alternative_erklaerung", interpretation["alternative_erklaerung"]
        ),
        "alternative_erklaerung_keywords": " | ".join(
            interpretation.get("alternative_erklaerung_keywords", [])
        ),
        "begruendung": " | ".join(interpretation.get("begruendung", [])),
        "klasse": final_klasse,
        "klassifikation": guard["klassifikation"],
        "klassifikation_begruendung": klassifikation_begr_str,
        **_processing_status_fields(
            status="processed",
            llm_called=1,
            skipped_reason=str(guard.get("decision_rule_applied", "")),
        ),
    }


def infer_report_with_prompt_version(report: dict, prompt_version: str) -> Dict[str, Any]:
    """
    Run full report inference with a specific Agent 2 prompt version (v1 or v2).

    Returns a prediction row dict including validation_report_id when present on *report*.
    """
    version = normalize_prompt_version(prompt_version)
    full_report_text = str(report.get("report_text", "") or "")
    patient_id = str(report.get("PatientenID", "") or "").strip()
    report_name = str(report.get("bericht", "") or "").strip()
    bertyp = resolve_bertyp(report)

    ev = extract_delirium_evidence(full_report_text)
    snippets = ev.get("evidence_snippets") or []

    if not llm_should_receive_evidence(snippets):
        if should_send_short_report_without_evidence(
            full_report_text,
            bertyp,
            snippets,
            original_length=int(ev.get("original_report_text_length") or 0),
        ):
            ev = apply_short_report_fulltext_to_evidence(ev, full_report_text)
        else:
            row = _prediction_row_no_evidence(ev, patient_id, report_name, bertyp=bertyp)
            row["prompt_version"] = version
            row.update(_report_identity_fields(report))
            return row

    llm_text = ev["llm_report_text"]

    try:
        result = extract_passages(llm_text, patient_id=patient_id, report_name=report_name)
        interpretation = interpret_signals_llm(
            llm_text,
            result,
            patient_id=patient_id,
            report_name=report_name,
            prompt_version=version,
        )
        classification = classify_delirium(interpretation)
        guard = apply_clinical_decision_guardrails(
            interpretation,
            result,
            ev,
            llm_skipped=False,
        )
        return _build_inference_row(
            report=report,
            ev=ev,
            result=result,
            interpretation=interpretation,
            guard=guard,
            classification=classification,
            patient_id=patient_id,
            report_name=report_name,
            bertyp=bertyp,
            prompt_version=version,
        )

    except Exception as exc:
        LOGGER.exception(
            "Cascade inference failure patient=%s report=%s version=%s",
            patient_id,
            report.get(VALIDATION_REPORT_ID_COL, report_name),
            version,
        )
        err = f"{type(exc).__name__}: {exc}"
        row = _prediction_row_pipeline_error(ev, patient_id, report_name, err, bertyp=bertyp)
        row["prompt_version"] = version
        row.update(_report_identity_fields(report))
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(traceback.format_exc())
        return row


def infer_report_cascade_reviewer(report: dict) -> Dict[str, Any]:
    """
    Run cascade stage-2 reviewer on a V1-positive report (same evidence/guardrail path as V1/V2).
    """
    full_report_text = str(report.get("report_text", "") or "")
    patient_id = str(report.get("PatientenID", "") or "").strip()
    report_name = str(report.get("bericht", "") or "").strip()
    bertyp = resolve_bertyp(report)

    ev = extract_delirium_evidence(full_report_text)
    snippets = ev.get("evidence_snippets") or []

    if not llm_should_receive_evidence(snippets):
        if should_send_short_report_without_evidence(
            full_report_text,
            bertyp,
            snippets,
            original_length=int(ev.get("original_report_text_length") or 0),
        ):
            ev = apply_short_report_fulltext_to_evidence(ev, full_report_text)
        else:
            row = _prediction_row_no_evidence(ev, patient_id, report_name, bertyp=bertyp)
            row["prompt_version"] = STAGE2_MODE_CASCADE_REVIEWER
            row.update(_report_identity_fields(report))
            return row

    llm_text = ev["llm_report_text"]

    try:
        result = extract_passages(llm_text, patient_id=patient_id, report_name=report_name)
        interpretation = interpret_cascade_reviewer(
            llm_text,
            result,
            patient_id=patient_id,
            report_name=report_name,
        )
        classification = classify_delirium(interpretation)
        guard = apply_clinical_decision_guardrails(
            interpretation,
            result,
            ev,
            llm_skipped=False,
        )
        return _build_inference_row(
            report=report,
            ev=ev,
            result=result,
            interpretation=interpretation,
            guard=guard,
            classification=classification,
            patient_id=patient_id,
            report_name=report_name,
            bertyp=bertyp,
            prompt_version=STAGE2_MODE_CASCADE_REVIEWER,
        )

    except Exception as exc:
        LOGGER.exception(
            "Cascade reviewer inference failure patient=%s report=%s",
            patient_id,
            report.get(VALIDATION_REPORT_ID_COL, report_name),
        )
        err = f"{type(exc).__name__}: {exc}"
        row = _prediction_row_pipeline_error(ev, patient_id, report_name, err, bertyp=bertyp)
        row["prompt_version"] = STAGE2_MODE_CASCADE_REVIEWER
        row.update(_report_identity_fields(report))
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(traceback.format_exc())
        return row


def infer_report_stage2(report: dict, stage2_mode: str) -> Dict[str, Any]:
    """Dispatch stage-2 inference by mode (v2 classifier or cascade reviewer)."""
    mode = normalize_stage2_mode(stage2_mode)
    if mode == STAGE2_MODE_V2:
        return infer_report_with_prompt_version(report, "v2")
    return infer_report_cascade_reviewer(report)
