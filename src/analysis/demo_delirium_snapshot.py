"""
Build self-contained JSON snapshots for the delirium pipeline presentation demo.

Snapshots hold everything needed to replay one report offline: original text,
rule-based evidence, LLM input bundle, interpretation, guardrails, and labels.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from src.analysis.export_presentation_examples import parse_evidence_snippets
from src.pipeline.frozen_cohort_inference import build_stable_report_text_index
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    DEMO_NEGATIVE_SNAPSHOT_PATH,
    DEMO_POSITIVE_SNAPSHOT_PATH,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.preprocessing.evidence_extraction import (
    METHOD_NO_EVIDENCE,
    SECTION_DISPLAY,
    extract_delirium_evidence,
)

LOGGER = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1

# Public-facing labels only — no real hospital or validation identifiers.
PRESENTATION_LABELS: Dict[str, Dict[str, str]] = {
    "positive": {
        "presentation_label": "Beispiel-Fall A (Delir positiv)",
        "presentation_report_label": "Beispiel-Bericht 1",
        "presentation_patient_label": "Beispiel-Fall A",
    },
    "negative": {
        "presentation_label": "Beispiel-Fall B (Delir negativ)",
        "presentation_report_label": "Beispiel-Bericht 2",
        "presentation_patient_label": "Beispiel-Fall B",
    },
}

# Scrub long numeric tokens that may be hospital patient IDs in free text.
_PATIENT_ID_IN_TEXT_RE = re.compile(r"\b\d{7,9}\b")
_VALIDATION_ID_IN_TEXT_RE = re.compile(r"Patient_\d{4,}(?:_Report_\d{4,})?", re.IGNORECASE)


def _scrub_identifiers_from_text(text: str, extra_tokens: Sequence[str]) -> str:
    """Remove known identifiers and common ID patterns from clinical free text."""
    out = str(text or "")
    for token in sorted({str(t).strip() for t in extra_tokens if str(t).strip()}, key=len, reverse=True):
        if len(token) < 4:
            continue
        out = re.sub(re.escape(token), "[anonymisiert]", out, flags=re.IGNORECASE)
    out = _VALIDATION_ID_IN_TEXT_RE.sub("[anonymisiert]", out)
    out = _PATIENT_ID_IN_TEXT_RE.sub("[Patient-ID]", out)
    return out


def anonymize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a presentation-safe copy: no PatientenID, no validation IDs, no report dates.

    Intended for thesis talks and public demos. Original identifiers are discarded.
    """
    out = copy.deepcopy(snapshot)
    polarity = str(out.get("polarity") or "positive")
    if polarity not in PRESENTATION_LABELS:
        polarity = "positive" if int((out.get("final") or {}).get("klasse") or 0) == 1 else "negative"
    labels = PRESENTATION_LABELS[polarity]

    case = dict(out.get("case") or {})
    scrub_tokens = [
        case.get("PatientenID"),
        case.get("validation_report_id"),
        case.get("validation_patient_id"),
        case.get("bericht"),
        case.get("berdat"),
    ]

    case["presentation_label"] = labels["presentation_label"]
    case["presentation_report_label"] = labels["presentation_report_label"]
    case["presentation_patient_label"] = labels["presentation_patient_label"]
    case.pop("PatientenID", None)
    case.pop("validation_report_id", None)
    case.pop("validation_patient_id", None)
    case["bericht"] = "[anonymisiert]"
    case["berdat"] = "—"
    out["case"] = case

    out["report_text"] = _scrub_identifiers_from_text(out.get("report_text") or "", scrub_tokens)

    extraction = dict(out.get("extraction") or {})
    snippets = []
    for snip in extraction.get("evidence_snippets") or []:
        s = dict(snip)
        s["text"] = _scrub_identifiers_from_text(s.get("text") or "", scrub_tokens)
        snippets.append(s)
    extraction["evidence_snippets"] = snippets
    if extraction.get("llm_report_text"):
        extraction["llm_report_text"] = _scrub_identifiers_from_text(
            extraction["llm_report_text"], scrub_tokens
        )
    out["extraction"] = extraction

    out["anonymized_for_presentation"] = True
    out.pop("selection", None)
    return out


def presentation_case_title(snapshot: Dict[str, Any]) -> str:
    """Header for terminal / HTML — never exposes real IDs."""
    case = snapshot.get("case") or {}
    return str(
        case.get("presentation_label")
        or case.get("presentation_report_label")
        or "Beispiel-Fall"
    )


def presentation_case_subtitle(snapshot: Dict[str, Any]) -> str:
    case = snapshot.get("case") or {}
    bertyp = str(case.get("bertyp") or "").strip()
    gt = case.get("manual_report_ground_truth")
    parts = [p for p in (bertyp, f"manuelles GT = {gt}" if gt is not None else "") if p]
    return " · ".join(parts)


# Anonymized demonstration reports (used when validation data is unavailable).
CURATED_POSITIVE_REPORT = """[Diagnosen]
Septischer Schock, akutes Nierenversagen, Delir.

[Epikrise]
Auf der Intensivstation dokumentiertes Delir mit Desorientierung zur Zeit. Delirtherapie mit Haloperidol, anschliessend Verbesserung der Vigilanz.

[Prozedere]
Delirmedikation ausschleichen, CAM-ICU Screening.
"""

CURATED_NEGATIVE_REPORT = """[Diagnosen]
Community-acquired Pneumonie, respiratorische Insuffizienz.

[Jetziges Leiden]
Zunehmende Dyspnoe bei bekannter COPD, ansonsten neurologisch ohne neu aufgetretene Auffälligkeiten.

[Epikrise]
Aufnahme auf IMC mit High-Flow-Sauerstoff und intravenöser Antibiotikatherapie. GCS stabil bei 15. Laborchemisch leichte Laktaterhöhung, kein septischer Schock.

[Prozedere]
Weiterführende pulmonale Therapie, schrittweise Entwöhnung von High-Flow.
"""

CURATED_POSITIVE_INTERPRETATION: Dict[str, Any] = {
    "delir_signale": {
        "desorientierung": ["Desorientierung zur Zeit"],
        "delir_explizit": ["Delir"],
        "hyperaktivitaet_agitation": [],
        "vigilanz": ["Vigilanz"],
        "delir_therapie": ["Haloperidol"],
        "delir_prophylaxe": [],
    },
    "signalstaerke": "hoch",
    "delir_probability_estimate": 88,
    "kontext": "Explizite Delir-Diagnose mit Desorientierung und Delirtherapie auf der Intensivstation.",
    "begruendung": "Delir in Diagnosen und Epikrise dokumentiert; Therapie und klinischer Verlauf stützen ein dokumentiertes Delir.",
    "alternative_erklaerung": False,
    "alternative_erklaerung_keywords": [],
}

CURATED_NEGATIVE_INTERPRETATION: Dict[str, Any] = {
    "delir_signale": {
        "desorientierung": [],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    },
    "signalstaerke": "niedrig",
    "delir_probability_estimate": 0,
    "kontext": "",
    "begruendung": "",
    "alternative_erklaerung": False,
    "alternative_erklaerung_keywords": [],
    "skipped_reason": "no_evidence_prefilter_skip",
}


def _bool_cell(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in ("1", "true", "yes")


def _int_cell(value: object, default: int = 0) -> int:
    try:
        n = pd.to_numeric(value, errors="coerce")
        if pd.isna(n):
            return default
        return int(n)
    except (TypeError, ValueError):
        return default


def _parse_delir_signale(raw: object) -> Dict[str, List[str]]:
    empty = {
        "desorientierung": [],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": [],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    }
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return empty
    if isinstance(raw, dict):
        out = dict(empty)
        for key in empty:
            val = raw.get(key, [])
            out[key] = [str(v) for v in val] if isinstance(val, list) else []
        return out
    text = str(raw).strip()
    if not text:
        return empty
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _parse_delir_signale(parsed)
    except json.JSONDecodeError:
        pass
    return empty


def _extraction_payload(report_text: str) -> Dict[str, Any]:
    ev = extract_delirium_evidence(report_text)
    snippets = ev.get("evidence_snippets") or []
    return {
        "original_report_text_length": int(ev.get("original_report_text_length", len(report_text))),
        "llm_report_text": ev.get("llm_report_text") or "",
        "llm_report_text_length": int(ev.get("llm_report_text_length", 0)),
        "llm_text_reduction_method": ev.get("llm_text_reduction_method") or "",
        "evidence_snippets": snippets,
        "delir_keyword_hits_count": int(ev.get("delir_keyword_hits_count", len(snippets))),
        "has_direct_delir_evidence": bool(ev.get("has_direct_delir_evidence")),
        "has_indirect_delir_evidence": bool(ev.get("has_indirect_delir_evidence")),
        "has_negated_delir_evidence": bool(ev.get("has_negated_delir_evidence")),
        "has_prophylaxis_or_risk_only": bool(ev.get("has_prophylaxis_or_risk_only")),
    }


def build_snapshot_from_row(
    row: pd.Series,
    *,
    report_text: str,
    manual_gt: Optional[int] = None,
    source: str = "validation_cohort",
) -> Dict[str, Any]:
    """Assemble a portable snapshot dict from one prediction row + report text."""
    snippets = parse_evidence_snippets(row.get("evidence_snippets"))
    extraction = _extraction_payload(report_text)
    if snippets:
        extraction["evidence_snippets"] = snippets
        extraction["llm_report_text"] = str(row.get("llm_report_text") or extraction["llm_report_text"])
    llm_called = _bool_cell(row.get("llm_called"))
    llm_skipped = _bool_cell(row.get("llm_skipped_by_prefilter"))
    klasse = _int_cell(row.get("klasse"))
    polarity = "positive" if klasse == 1 else "negative"
    prob = _int_cell(row.get("delir_probability_estimate"), default=-1)

    interpretation: Dict[str, Any] = {
        "delir_signale": _parse_delir_signale(row.get("delir_signale")),
        "signalstaerke": str(row.get("signalstaerke") or ""),
        "delir_probability_estimate": prob if prob >= 0 else "",
        "kontext": str(row.get("kontext") or ""),
        "begruendung": str(row.get("begruendung") or ""),
        "alternative_erklaerung": _bool_cell(row.get("alternative_erklaerung")),
        "alternative_erklaerung_keywords": str(row.get("alternative_erklaerung_keywords") or ""),
    }
    if llm_skipped:
        interpretation["skipped_reason"] = str(row.get("skipped_reason") or row.get("decision_rule_applied") or "")

    if manual_gt is None:
        manual_gt = _int_cell(row.get("manual_report_ground_truth"), default=-1)
        if manual_gt < 0:
            manual_gt = None

    return {
        "version": SNAPSHOT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "polarity": polarity,
        "case": {
            "validation_report_id": str(row.get("validation_report_id") or ""),
            "validation_patient_id": str(row.get("validation_patient_id") or ""),
            "PatientenID": str(row.get("PatientenID") or ""),
            "bertyp": str(row.get("bertyp") or ""),
            "berdat": str(row.get("berdat") or ""),
            "bericht": str(row.get("bericht") or ""),
            "manual_report_ground_truth": manual_gt,
        },
        "report_text": report_text,
        "extraction": extraction,
        "interpretation": interpretation,
        "final": {
            "klasse": klasse,
            "decision_rule_applied": str(row.get("decision_rule_applied") or ""),
            "llm_called": llm_called,
            "llm_skipped_by_prefilter": llm_skipped,
            "manual_review_candidate": _bool_cell(row.get("manual_review_candidate")),
            "status": str(row.get("status") or ""),
        },
        "verification": {
            "model_correct_vs_manual": (
                manual_gt is not None and klasse == manual_gt
            ),
        },
    }


def build_curated_snapshot(*, polarity: str) -> Dict[str, Any]:
    """Anonymized fallback snapshot when validation CSVs are not available."""
    if polarity == "positive":
        report_text = CURATED_POSITIVE_REPORT
        case_meta = {
            "bertyp": "Austrittsbericht",
            "manual_report_ground_truth": 1,
        }
        interpretation = dict(CURATED_POSITIVE_INTERPRETATION)
        final = {
            "klasse": 1,
            "decision_rule_applied": "direct_delir_positive",
            "llm_called": True,
            "llm_skipped_by_prefilter": False,
            "manual_review_candidate": False,
            "status": "success",
        }
    else:
        report_text = CURATED_NEGATIVE_REPORT
        case_meta = {
            "bertyp": "Verlaufseintrag",
            "manual_report_ground_truth": 0,
        }
        interpretation = dict(CURATED_NEGATIVE_INTERPRETATION)
        final = {
            "klasse": 0,
            "decision_rule_applied": "no_evidence_prefilter_skip",
            "llm_called": False,
            "llm_skipped_by_prefilter": True,
            "manual_review_candidate": False,
            "status": "skipped",
        }

    extraction = _extraction_payload(report_text)
    snap = {
        "version": SNAPSHOT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "curated_anonymized",
        "polarity": polarity,
        "case": case_meta,
        "report_text": report_text,
        "extraction": extraction,
        "interpretation": interpretation,
        "final": final,
        "verification": {"model_correct_vs_manual": True},
    }
    return anonymize_snapshot(snap)


def to_json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return to_json_safe(value.tolist())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def save_snapshot(snapshot: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = anonymize_snapshot(snapshot) if not snapshot.get("anonymized_for_presentation") else snapshot
    payload = to_json_safe(safe)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_snapshot(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_predictions_with_labels(
    predictions: pd.DataFrame,
    labels: Optional[pd.DataFrame],
) -> pd.DataFrame:
    pred = predictions.copy()
    if labels is None or labels.empty:
        return pred
    label_cols = [c for c in ("validation_report_id", "manual_report_ground_truth", "manual_comment") if c in labels.columns]
    if "validation_report_id" not in label_cols:
        return pred
    lab = labels[label_cols].drop_duplicates("validation_report_id")
    return pred.merge(lab, on="validation_report_id", how="left", suffixes=("", "_label"))


def _resolve_report_text(row: pd.Series, text_index: Dict[Tuple[str, str, str, str], str]) -> str:
    pid = str(row.get("PatientenID") or "").strip()
    bertyp = str(row.get("bertyp") or "").strip()
    berdat = str(row.get("berdat") or "").strip()
    for bericht in (
        str(row.get("bericht") or "").strip(),
        str(row.get("pipeline_bericht") or "").strip(),
    ):
        if bericht:
            key = (pid, bertyp, berdat, bericht)
            if key in text_index:
                return text_index[key]
    return ""


def _score_positive_row(row: pd.Series) -> int:
    """Prefer concise, clear direct-delir cases suitable for a thesis slide."""
    if _int_cell(row.get("manual_report_ground_truth"), -1) != 1:
        return -1
    if _int_cell(row.get("klasse")) != 1:
        return -1

    snippets = parse_evidence_snippets(row.get("evidence_snippets"))
    direct = [s for s in snippets if str(s.get("evidence_type")) == "direct_delir"]
    indirect = [s for s in snippets if str(s.get("evidence_type")) == "indirect_symptom"]

    score = 0
    if not direct:
        return -1
    if str(row.get("decision_rule_applied") or "") == "direct_delir_positive":
        score += 45
    elif str(row.get("decision_rule_applied") or "") == "llm_classification":
        score += 20
    else:
        score += 10

    if str(row.get("signalstaerke") or "").lower() == "hoch":
        score += 25
    if _bool_cell(row.get("llm_called")):
        score += 10
    if _bool_cell(row.get("manual_review_candidate")):
        score -= 50
    if _bool_cell(row.get("has_alternative_explanation")):
        score -= 35

    for s in direct:
        if str(s.get("section") or "") == "diag":
            score += 35
        kw = str(s.get("keyword") or "").lower()
        if kw in ("delir", "delirium", "delirant", "delirös"):
            score += 30
        elif "hypoaktiv" in kw or "hyperaktiv" in kw:
            score += 10

    score -= len(indirect) * 6
    score -= max(0, len(snippets) - 3) * 10

    orig_len = _int_cell(row.get("original_report_text_length"), 0)
    if 300 <= orig_len <= 1800:
        score += 20
    elif orig_len > 3500:
        score -= 30

    llm_len = _int_cell(row.get("llm_report_text_length"), 0)
    if 0 < llm_len <= 1000:
        score += 15
    elif llm_len > 2000:
        score -= 20

    if str(row.get("bertyp") or "") in ("Austrittsbericht", "Verlaufseintrag"):
        score += 5
    return score


def _score_negative_row(row: pd.Series) -> int:
    if _int_cell(row.get("manual_report_ground_truth"), -1) != 0:
        return -1
    if _int_cell(row.get("klasse")) != 0:
        return -1
    score = 0
    rule = str(row.get("decision_rule_applied") or "")
    if rule == "no_evidence_prefilter_skip":
        score += 100
    elif rule == "negated_delir_only_not_positive":
        score += 80
    elif rule == "prophylaxis_only_not_positive":
        score += 70
    elif _bool_cell(row.get("llm_skipped_by_prefilter")):
        score += 60
    if not _bool_cell(row.get("has_direct_delir_evidence")):
        score += 10
    if not _bool_cell(row.get("has_indirect_delir_evidence")):
        score += 10
    return score


def rank_validation_candidates(
    predictions: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    *,
    polarity: str,
    exclude_ids: Optional[Sequence[str]] = None,
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Return top scored TP/TN rows for manual inspection."""
    df = _merge_predictions_with_labels(predictions, labels)
    exclude = {str(x).strip() for x in (exclude_ids or []) if str(x).strip()}
    scorer = _score_positive_row if polarity == "positive" else _score_negative_row
    ranked: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        score = scorer(row)
        if score < 0:
            continue
        vid = str(row.get("validation_report_id") or "").strip()
        if not vid or vid in exclude:
            continue
        ranked.append(
            {
                "validation_report_id": vid,
                "score": score,
                "bertyp": str(row.get("bertyp") or ""),
                "decision_rule_applied": str(row.get("decision_rule_applied") or ""),
                "signalstaerke": str(row.get("signalstaerke") or ""),
                "snippet_count": len(parse_evidence_snippets(row.get("evidence_snippets"))),
                "report_length": _int_cell(row.get("original_report_text_length"), 0),
            }
        )
    ranked.sort(key=lambda x: (-x["score"], x["validation_report_id"]))
    return ranked[:top_n]


def autopick_validation_report_id(
    predictions: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    *,
    polarity: str,
    exclude_ids: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Pick the clearest verified TP or TN from merged validation predictions."""
    ranked = rank_validation_candidates(
        predictions, labels, polarity=polarity, exclude_ids=exclude_ids, top_n=1
    )
    return ranked[0]["validation_report_id"] if ranked else None


def generate_snapshot_from_validation(
    *,
    polarity: str,
    out_path: Path,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    validation_report_id: Optional[str] = None,
    exclude_validation_report_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Build a snapshot from frozen validation data.

    Falls back to curated anonymized demo cases when predictions are missing or stub-only.
    """
    if not predictions_path.exists() or predictions_path.stat().st_size < 200:
        LOGGER.warning("Predictions missing or stub-only; using curated %s case.", polarity)
        snap = build_curated_snapshot(polarity=polarity)
        save_snapshot(snap, out_path)
        return load_snapshot(out_path)

    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path) if labels_path.exists() else None
    merged = _merge_predictions_with_labels(predictions, labels)

    if validation_report_id:
        hits = merged[merged["validation_report_id"].astype(str) == validation_report_id]
        if hits.empty:
            raise ValueError(f"validation_report_id not found: {validation_report_id}")
        row = hits.iloc[0]
    else:
        picked_id = autopick_validation_report_id(
            predictions,
            labels,
            polarity=polarity,
            exclude_ids=exclude_validation_report_ids,
        )
        if not picked_id:
            LOGGER.warning("No suitable %s case in validation data; using curated demo.", polarity)
            snap = build_curated_snapshot(polarity=polarity)
            save_snapshot(snap, out_path)
            return snap
        row = merged[merged["validation_report_id"].astype(str) == picked_id].iloc[0]

    text_index = build_stable_report_text_index(berichte_path)
    report_text = _resolve_report_text(row, text_index)
    if not report_text.strip():
        snippets = parse_evidence_snippets(row.get("evidence_snippets"))
        if snippets:
            report_text = "\n\n".join(str(s.get("text") or "") for s in snippets)
    if not report_text.strip():
        LOGGER.warning("Report text not found; using curated %s case.", polarity)
        snap = build_curated_snapshot(polarity=polarity)
        save_snapshot(snap, out_path)
        return load_snapshot(out_path)

    manual_gt = _int_cell(row.get("manual_report_ground_truth"), -1)
    snap = build_snapshot_from_row(
        row,
        report_text=report_text,
        manual_gt=manual_gt if manual_gt >= 0 else None,
        source="validation_cohort",
    )
    ranked = rank_validation_candidates(
        predictions,
        labels,
        polarity=polarity,
        exclude_ids=exclude_validation_report_ids,
        top_n=20,
    )
    vid = str(row.get("validation_report_id") or "")
    snap["selection"] = {
        "validation_report_id_picked": vid,
        "rank": next((i + 1 for i, r in enumerate(ranked) if r["validation_report_id"] == vid), None),
        "score": next((r["score"] for r in ranked if r["validation_report_id"] == vid), None),
        "top_candidates": ranked[:5],
    }
    save_snapshot(snap, out_path)
    return load_snapshot(out_path)


def ensure_default_snapshots() -> Tuple[Path, Path]:
    """Create default demo snapshots under data/demo/ if missing."""
    pos_path = DEMO_POSITIVE_SNAPSHOT_PATH
    neg_path = DEMO_NEGATIVE_SNAPSHOT_PATH
    if not pos_path.exists():
        generate_snapshot_from_validation(polarity="positive", out_path=pos_path)
    if not neg_path.exists():
        generate_snapshot_from_validation(polarity="negative", out_path=neg_path)
    return pos_path, neg_path


def snippet_section_label(snippet: Dict[str, Any]) -> str:
    sec = str(snippet.get("section") or "unknown")
    return SECTION_DISPLAY.get(sec, sec)
