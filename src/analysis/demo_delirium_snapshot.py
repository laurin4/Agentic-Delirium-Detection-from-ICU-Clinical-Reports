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
Akutes Nierenversagen, septischer Schock, therapiebedürftiges hypoaktives Delir.

[Epikrise]
Auf der Intensivstation zeigte sich ein hypoaktives Delir mit ausgeprägter Desorientierung zur Zeit und Person. Reorientierungsmassnahmen und Delirtherapie mit niedrig dosiertem Haloperidol. Im weiteren Verlauf langsame Besserung der Vigilanz.

[Prozedere]
Delirmedikation schrittweise reduzieren, CAM-ICU Screening fortführen.
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
        "desorientierung": ["Desorientierung zur Zeit und Person"],
        "delir_explizit": ["hypoaktives Delir", "Delirtherapie"],
        "hyperaktivitaet_agitation": [],
        "vigilanz": ["Vigilanzminderung"],
        "delir_therapie": ["Haloperidol"],
        "delir_prophylaxe": [],
    },
    "signalstaerke": "hoch",
    "delir_probability_estimate": 92,
    "kontext": "Explizit dokumentiertes hypoaktives Delir mit Desorientierung und Delirtherapie auf der Intensivstation.",
    "begruendung": "Direkte Delir-Nennung in Diagnose und Epikrise; Therapie und klinischer Verlauf stützen ein dokumentiertes Delir.",
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
        "original_report_text_length": ev.get("original_report_text_length", len(report_text)),
        "llm_report_text": ev.get("llm_report_text") or "",
        "llm_report_text_length": ev.get("llm_report_text_length", 0),
        "llm_text_reduction_method": ev.get("llm_text_reduction_method") or "",
        "evidence_snippets": snippets,
        "delir_keyword_hits_count": ev.get("delir_keyword_hits_count", len(snippets)),
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

    interpretation: Dict[str, Any] = {
        "delir_signale": _parse_delir_signale(row.get("delir_signale")),
        "signalstaerke": str(row.get("signalstaerke") or ""),
        "delir_probability_estimate": row.get("delir_probability_estimate", ""),
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


def save_snapshot(snapshot: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = anonymize_snapshot(snapshot) if not snapshot.get("anonymized_for_presentation") else snapshot
    path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
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
    if _int_cell(row.get("manual_report_ground_truth"), -1) != 1:
        return -1
    if _int_cell(row.get("klasse")) != 1:
        return -1
    score = 0
    if _bool_cell(row.get("has_direct_delir_evidence")):
        score += 50
    if str(row.get("decision_rule_applied") or "") == "direct_delir_positive":
        score += 40
    if str(row.get("signalstaerke") or "").lower() == "hoch":
        score += 20
    if _bool_cell(row.get("llm_called")):
        score += 10
    snippets = parse_evidence_snippets(row.get("evidence_snippets"))
    score += min(len(snippets), 5) * 3
    if str(row.get("bertyp") or "") == "Austrittsbericht":
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


def autopick_validation_report_id(
    predictions: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    *,
    polarity: str,
) -> Optional[str]:
    """Pick the clearest verified TP or TN from merged validation predictions."""
    df = _merge_predictions_with_labels(predictions, labels)
    if df.empty or "validation_report_id" not in df.columns:
        return None
    scorer = _score_positive_row if polarity == "positive" else _score_negative_row
    best_id: Optional[str] = None
    best_score = -1
    for _, row in df.iterrows():
        score = scorer(row)
        if score > best_score:
            best_score = score
            best_id = str(row.get("validation_report_id") or "").strip() or None
    return best_id


def generate_snapshot_from_validation(
    *,
    polarity: str,
    out_path: Path,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    validation_report_id: Optional[str] = None,
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
        picked_id = autopick_validation_report_id(predictions, labels, polarity=polarity)
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
