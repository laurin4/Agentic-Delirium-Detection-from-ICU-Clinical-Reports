"""
Trace why validation reports received decision_rule_applied=direct_delir_positive.

Read-only audit: compares stored prediction CSV fields, evidence snippets,
fresh rule-layer re-extraction, optional Agent 1 debug JSON, and guardrail logic.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.agents.clinical_guardrails import (
    _has_explicit_delir_signals,
    _qualifies_for_direct_delir_positive,
    _resolve_direct_evidence_flags,
    apply_clinical_decision_guardrails,
)
from src.analysis.export_presentation_examples import parse_evidence_snippets
from src.pipeline.frozen_cohort_inference import build_frozen_cohort_inference_records
from src.pipeline.paths import FROZEN_PATIENT_VALIDATION_COHORT_PATH
from src.pipeline.prompt_run_paths import resolve_validation_predictions_path
from src.pipeline.validation_report_identity import VALIDATION_REPORT_ID_COL
from src.preprocessing.evidence_extraction import extract_delirium_evidence

LOGGER = logging.getLogger(__name__)

SIGNAL_KEYS = (
    "desorientierung",
    "delir_explizit",
    "hyperaktivitaet_agitation",
    "vigilanz",
    "delir_therapie",
    "delir_prophylaxe",
)


def _bool_cell(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in ("1", "true", "yes")


def _snippets_by_type(snippets: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {
        "direct_delir": [],
        "negation": [],
        "indirect_symptom": [],
        "prophylaxis_or_risk": [],
        "other": [],
    }
    for s in snippets:
        et = str(s.get("evidence_type") or "other")
        out.setdefault(et, [])
        out[et].append(s)
    return out


def _load_agent1_from_debug(
    debug_dir: Path,
    patient_id: str,
    report_name: str,
) -> Optional[Dict[str, List[str]]]:
    """Best-effort load of Agent 1 parsed JSON from llm_debug files."""
    if not debug_dir.is_dir():
        return None
    needle_pid = str(patient_id or "").strip()
    needle_rep = str(report_name or "").strip().replace(".txt", "")
    candidates = sorted(debug_dir.glob("*Agent_1_Extraction*"), reverse=True)
    for path in candidates:
        name = path.name
        if needle_pid and needle_pid not in name:
            continue
        if needle_rep and needle_rep not in name and Path(report_name).stem not in name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parsed = data.get("parsed_output") or data.get("parsed") or data.get("result")
        if isinstance(parsed, dict) and "delir_explizit" in parsed:
            return {k: list(parsed.get(k) or []) for k in SIGNAL_KEYS}
    return None


def _explain_direct_delir_positive(
    *,
    has_direct_meta: bool,
    has_negated_meta: bool,
    has_explicit: bool,
    llm_skipped: bool,
    llm_method: str,
    agent1: Optional[Dict[str, List[str]]] = None,
) -> str:
    signals = agent1 or {k: [] for k in SIGNAL_KEYS}
    ev = {
        "has_direct_delir_evidence": has_direct_meta,
        "has_negated_delir_evidence": has_negated_meta,
    }
    has_rule_direct, has_agent1_explicit, has_agent1_allowed, has_direct_guard = (
        _resolve_direct_evidence_flags(signals, ev)
    )
    qualifies = _qualifies_for_direct_delir_positive(
        has_rule_direct,
        has_negated_meta,
        has_agent1_explicit,
        has_agent1_allowed,
    )

    if llm_skipped or llm_method == "no_evidence_prefilter_skip":
        return (
            "UNEXPECTED: direct_delir_positive should not apply when LLM was skipped "
            f"(llm_skipped={llm_skipped}, method={llm_method}). Check stale CSV or rerun mismatch."
        )

    if not qualifies:
        return (
            "UNEXPECTED: _qualifies_for_direct_delir_positive=False "
            f"(rule_direct={has_rule_direct}, negated={has_negated_meta}, "
            f"agent1_explicit={has_agent1_explicit}, agent1_allowed={has_agent1_allowed})."
        )

    parts: List[str] = []
    if has_rule_direct:
        parts.append("has_rule_direct=True (rule-layer direct_delir snippet)")
    if has_agent1_allowed:
        parts.append("has_agent1_explicit_allowed=True (delir_explizit without negation flag)")
    elif has_agent1_explicit and has_negated_meta:
        parts.append("Agent 1 delir_explizit ignored under has_negated_delir_evidence=True")

    if has_rule_direct and has_negated_meta and has_agent1_explicit:
        parts.append("mixed report: rule direct + negation corroborated by Agent 1 explicit")

    parts.append(f"_qualifies_for_direct_delir_positive={qualifies}")
    parts.append("Triggered at clinical_guardrails.py direct_delir_positive block")
    return " | ".join(parts)


def _cohort_report_text_index() -> Dict[str, str]:
    records = build_frozen_cohort_inference_records(
        cohort_path=FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    )
    return {
        str(r.get(VALIDATION_REPORT_ID_COL, "") or "").strip(): str(r.get("report_text") or "")
        for r in records
        if str(r.get(VALIDATION_REPORT_ID_COL, "") or "").strip()
    }


def audit_direct_delir_positive_reports(
    predictions_path: Path,
    *,
    patient_ids: Optional[Sequence[str]] = None,
    debug_dir: Path = Path("outputs/logs/llm_debug"),
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path)
    if "decision_rule_applied" not in pred.columns:
        raise ValueError(f"Missing decision_rule_applied in {predictions_path}")

    rows = pred[pred["decision_rule_applied"].astype(str) == "direct_delir_positive"].copy()
    if patient_ids:
        pid_set = {str(p).strip() for p in patient_ids}
        if "validation_patient_id" in rows.columns:
            rows = rows[rows["validation_patient_id"].astype(str).isin(pid_set)]

    report_text_by_id = _cohort_report_text_index()
    audit_rows: List[Dict[str, Any]] = []

    for _, row in rows.iterrows():
        vpid = str(row.get("validation_patient_id", "") or "")
        vrid = str(row.get(VALIDATION_REPORT_ID_COL, "") or "")
        pid = str(row.get("PatientenID", "") or "")
        report_name = str(row.get("bericht", "") or "")

        stored_snippets = parse_evidence_snippets(row.get("evidence_snippets"))
        stored_by_type = _snippets_by_type(stored_snippets)

        has_direct_stored = _bool_cell(row.get("has_direct_delir_evidence"))
        has_neg_stored = _bool_cell(row.get("has_negated_delir_evidence"))
        llm_method = str(row.get("llm_text_reduction_method", "") or "")
        llm_skipped = _bool_cell(row.get("llm_skipped_by_prefilter"))

        report_text = report_text_by_id.get(vrid, "")
        fresh = extract_delirium_evidence(report_text) if report_text else {}
        fresh_by_type = _snippets_by_type(fresh.get("evidence_snippets") or [])

        agent1 = _load_agent1_from_debug(debug_dir, pid, report_name) or {k: [] for k in SIGNAL_KEYS}
        delir_explizit = [str(x) for x in agent1.get("delir_explizit", []) if str(x).strip()]
        has_explicit = _has_explicit_delir_signals(agent1)
        _, _, has_agent1_allowed, has_direct_guard = _resolve_direct_evidence_flags(
            agent1,
            {
                "has_direct_delir_evidence": has_direct_stored,
                "has_negated_delir_evidence": has_neg_stored,
            },
        )

        ev_for_guard = {
            "has_direct_delir_evidence": has_direct_stored,
            "has_negated_delir_evidence": has_neg_stored,
            "has_indirect_delir_evidence": _bool_cell(row.get("has_indirect_delir_evidence")),
            "has_prophylaxis_or_risk_only": _bool_cell(row.get("has_prophylaxis_or_risk_only")),
            "llm_text_reduction_method": llm_method,
        }
        guard_replay = apply_clinical_decision_guardrails(
            {
                "signalstaerke": str(row.get("signalstaerke", "niedrig") or "niedrig"),
                "kontext": str(row.get("kontext", "") or ""),
                "alternative_erklaerung": _bool_cell(row.get("alternative_erklaerung")),
                "begruendung": [],
            },
            agent1,
            ev_for_guard,
            llm_skipped=llm_skipped,
        )

        trigger = _explain_direct_delir_positive(
            has_direct_meta=has_direct_stored,
            has_negated_meta=has_neg_stored,
            has_explicit=has_explicit,
            llm_skipped=llm_skipped,
            llm_method=llm_method,
            agent1=agent1,
        )

        pathway = "unknown"
        if has_direct_stored and has_neg_stored and has_explicit:
            pathway = "A+B_mixed_rule_direct_negation_corroborated_by_agent1"
        elif has_direct_stored and not has_neg_stored:
            pathway = "A_rule_direct_snippet"
        elif has_direct_stored and has_neg_stored:
            pathway = "unexpected_A_rule_direct_with_negation_without_agent1"
        elif has_agent1_allowed and not has_direct_stored:
            pathway = "B_agent1_explicit_allowed"
        elif has_explicit and has_neg_stored:
            pathway = "blocked_agent1_explicit_under_negation"

        audit_rows.append(
            {
                "validation_patient_id": vpid,
                VALIDATION_REPORT_ID_COL: vrid,
                "PatientenID": pid,
                "bericht": report_name,
                "klasse": row.get("klasse"),
                "llm_called": row.get("llm_called"),
                "llm_skipped_by_prefilter": llm_skipped,
                "llm_text_reduction_method": llm_method,
                "has_direct_delir_evidence_stored": has_direct_stored,
                "has_negated_delir_evidence_stored": has_neg_stored,
                "fresh_has_direct_delir_evidence": bool(fresh.get("has_direct_delir_evidence")),
                "fresh_has_negated_delir_evidence": bool(fresh.get("has_negated_delir_evidence")),
                "stored_vs_fresh_direct_mismatch": has_direct_stored != bool(
                    fresh.get("has_direct_delir_evidence")
                ),
                "agent1_delir_explizit": " | ".join(delir_explizit),
                "agent1_delir_explizit_found_in_debug": bool(delir_explizit),
                "_has_explicit_delir_signals": has_explicit,
                "has_agent1_explicit_allowed": has_agent1_allowed,
                "guardrail_has_direct": has_direct_guard,
                "stored_direct_snippets": json.dumps(
                    [
                        {"keyword": s.get("keyword"), "text": s.get("text")}
                        for s in stored_by_type.get("direct_delir", [])
                    ],
                    ensure_ascii=False,
                ),
                "stored_negation_snippets": json.dumps(
                    [
                        {"keyword": s.get("keyword"), "text": s.get("text")}
                        for s in stored_by_type.get("negation", [])
                    ],
                    ensure_ascii=False,
                ),
                "fresh_direct_snippets": json.dumps(
                    [
                        {"keyword": s.get("keyword"), "text": s.get("text")}
                        for s in fresh_by_type.get("direct_delir", [])
                    ],
                    ensure_ascii=False,
                ),
                "fresh_negation_snippets": json.dumps(
                    [
                        {"keyword": s.get("keyword"), "text": s.get("text")}
                        for s in fresh_by_type.get("negation", [])
                    ],
                    ensure_ascii=False,
                ),
                "inferred_pathway": pathway,
                "direct_delir_positive_trigger": trigger,
                "guard_replay_rule": guard_replay.get("decision_rule_applied"),
                "klassifikation_begruendung": str(row.get("klassifikation_begruendung", "") or "")[:500],
            }
        )

    out = pd.DataFrame(audit_rows)
    if output_csv and not out.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        LOGGER.info("Wrote %s (%d rows)", output_csv, len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit direct_delir_positive reports in validation predictions.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Defaults to resolve_validation_predictions_path()",
    )
    parser.add_argument(
        "--patient-id",
        action="append",
        default=[],
        help="Filter to validation_patient_id (repeatable), e.g. Patient_0042",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("outputs/logs/llm_debug"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path",
    )
    args = parser.parse_args()

    pred_path = args.predictions_path or resolve_validation_predictions_path()
    out_path = args.output_csv
    if out_path is None and pred_path.parent.name == "predictions":
        out_path = pred_path.parent.parent / "audits" / "direct_delir_positive_trace.csv"

    df = audit_direct_delir_positive_reports(
        pred_path,
        patient_ids=args.patient_id or None,
        debug_dir=args.debug_dir,
        output_csv=out_path,
    )

    if df.empty:
        print("No direct_delir_positive reports found for the given filter.")
        return

    cols = [
        "validation_patient_id",
        VALIDATION_REPORT_ID_COL,
        "has_direct_delir_evidence_stored",
        "has_negated_delir_evidence_stored",
        "agent1_delir_explizit",
        "_has_explicit_delir_signals",
        "inferred_pathway",
        "direct_delir_positive_trigger",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    if out_path:
        print(f"\nFull trace written to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
