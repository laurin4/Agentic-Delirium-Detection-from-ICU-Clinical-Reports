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

from src.analysis.build_manual_validation_progress import (
    _parse_report_gt,
    _patient_model_positive,
    assign_confusion_group,
)
from src.analysis.export_presentation_examples import parse_evidence_snippets
from src.analysis.manual_report_labels import merge_manual_report_labels
from src.pipeline.frozen_cohort_inference import build_stable_report_text_index
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    DEMO_NEGATIVE_SNAPSHOT_PATH,
    DEMO_POSITIVE_SNAPSHOT_PATH,
    FINAL_MANUAL_VALIDATION_EVAL_DIR,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.analysis.demo_delirium_trace import (
    SYSTEM_PROMPT_EXCERPT_CHARS,
    TEXT_BLOCK_CHARS,
    TRACE_VERSION,
    build_delirium_trace,
    parse_delir_signale,
    trace_is_v2,
)
from src.preprocessing.evidence_extraction import (
    SECTION_DISPLAY,
    extract_delirium_evidence,
)

LOGGER = logging.getLogger(__name__)

SNAPSHOT_VERSION = TRACE_VERSION

# Public-facing labels only — no real hospital or validation identifiers.
PRESENTATION_LABELS: Dict[str, Dict[str, str]] = {
    "positive": {
        "presentation_label": "Beispiel-Fall A (Delir positiv · TP)",
        "presentation_report_label": "Beispiel-Bericht 1",
        "presentation_patient_label": "Beispiel-Fall A",
    },
    "false_negative": {
        "presentation_label": "Beispiel-Fall B (Falsch negativ · FN)",
        "presentation_report_label": "Beispiel-Bericht 2",
        "presentation_patient_label": "Beispiel-Fall B",
    },
    # Legacy alias — second demo case is FN.
    "negative": {
        "presentation_label": "Beispiel-Fall B (Falsch negativ · FN)",
        "presentation_report_label": "Beispiel-Bericht 2",
        "presentation_patient_label": "Beispiel-Fall B",
    },
}

# Preferred FN patients for the thesis demo (hospital PatientenID from manual validation).
PREFERRED_FN_PATIENTEN_IDS: Tuple[str, ...] = ("308617", "308954")
# Legacy alias — same needles passed to patient_suffix_matches (PatientenID / validation IDs).
PREFERRED_FN_PATIENT_SUFFIXES: Tuple[str, ...] = PREFERRED_FN_PATIENTEN_IDS

DEMO_POLARITIES: Tuple[str, ...] = ("positive", "false_negative")

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


def normalize_demo_polarity(polarity: str) -> str:
    """Map legacy aliases to false_negative (second demo case is FN)."""
    p = str(polarity or "positive").strip().lower()
    if p in ("negative", "false_positive"):
        return "false_negative"
    return p


def patient_suffix_matches(row: pd.Series, suffix: str) -> bool:
    """Match hospital PatientenID (e.g. 308617) or validation_patient_id / report_id needles."""
    suf = str(suffix or "").strip()
    if not suf:
        return False
    bare = suf.lstrip("0") or suf
    padded = bare.zfill(4)
    needles = {
        suf,
        bare,
        padded,
        f"Patient_{suf}",
        f"Patient_{padded}",
        f"Patient_{bare}",
    }
    for col in ("PatientenID", "validation_patient_id", "validation_report_id"):
        val = str(row.get(col) or "").strip()
        if not val:
            continue
        if col == "PatientenID" and val == suf:
            return True
        for needle in needles:
            if needle and needle in val:
                return True
        if re.search(rf"Patient_0*{re.escape(bare)}\b", val, flags=re.IGNORECASE):
            return True
    return False


def is_preferred_fn_patient(row: pd.Series) -> bool:
    return any(patient_suffix_matches(row, suf) for suf in PREFERRED_FN_PATIENT_SUFFIXES)


def _report_model_klasse(row: pd.Series) -> int:
    """Report prediction — matches evaluation: model_report_prediction before klasse."""
    for col in ("model_report_prediction", "klasse"):
        val = _int_cell(row.get(col), -1)
        if val in (0, 1):
            return val
    return -1


def _manual_report_gt(row: pd.Series) -> int:
    return _int_cell(row.get("manual_report_ground_truth"), -1)


def _load_frozen_patient_manual_gt() -> Dict[str, int]:
    """validation_patient_id or PatientenID → derived_manual_patient_ground_truth (0/1)."""
    out: Dict[str, int] = {}
    for path in (
        FINAL_MANUAL_VALIDATION_EVAL_DIR / "patient_level_ground_truth.csv",
        FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    ):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        col = "derived_manual_patient_ground_truth"
        if col not in df.columns:
            continue
        for key_col in ("validation_patient_id", "PatientenID"):
            if key_col not in df.columns:
                continue
            for key, grp in df.groupby(key_col):
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                if len(vals):
                    out[str(key).strip()] = int(vals.max())
    return out


def _computed_patient_model_positive(grp: pd.DataFrame) -> Optional[int]:
    vals = [_report_model_klasse(row) for _, row in grp.iterrows()]
    valid = [v for v in vals if v in (0, 1)]
    if valid:
        return max(valid)
    return _patient_model_positive(grp)


def _computed_derived_manual_gt(
    grp: pd.DataFrame,
    frozen_patient_gt: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    gt_series = (
        grp["manual_report_ground_truth"]
        if "manual_report_ground_truth" in grp.columns
        else pd.Series(index=grp.index, dtype=object)
    )
    parsed = _parse_report_gt(gt_series)
    n_total = int(len(grp))
    n_labeled = int(parsed.notna().sum())
    vpid = (
        str(grp["validation_patient_id"].iloc[0]).strip()
        if "validation_patient_id" in grp.columns and len(grp)
        else ""
    )
    patienten_id = (
        str(grp["PatientenID"].iloc[0]).strip()
        if "PatientenID" in grp.columns and len(grp)
        else ""
    )
    frozen_val: Optional[int] = None
    if frozen_patient_gt:
        for key in (vpid, patienten_id):
            if key and key in frozen_patient_gt:
                frozen_val = frozen_patient_gt[key]
                break

    if (parsed == "1").any():
        return 1

    label_derived: Optional[int] = None
    if n_total > 0 and n_labeled == n_total:
        label_derived = 0

    if frozen_val is not None:
        if label_derived is None:
            return frozen_val
        return max(label_derived, frozen_val)

    return label_derived


def _is_patient_level_fn(
    grp: pd.DataFrame,
    frozen_patient_gt: Optional[Dict[str, int]] = None,
) -> bool:
    model_pos = _computed_patient_model_positive(grp)
    derived = _computed_derived_manual_gt(grp, frozen_patient_gt)
    return model_pos == 0 and derived == 1


def _patient_subset(merged: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return merged[merged.apply(lambda r: patient_suffix_matches(r, suffix), axis=1)]


def resolve_fn_manual_gt(
    merged: pd.DataFrame,
    row: pd.Series,
    *,
    frozen_patient_gt: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """Manual GT for FN verification: report label, else patient-level derived GT."""
    manual = _manual_report_gt(row)
    if manual == 1:
        return 1
    vpid = str(row.get("validation_patient_id") or "").strip()
    if vpid and "validation_patient_id" in merged.columns:
        grp = merged[merged["validation_patient_id"].astype(str) == vpid]
        if not grp.empty and _is_patient_level_fn(grp, frozen_patient_gt):
            return 1
    return manual if manual >= 0 else None


def presentation_polarity_banner(snapshot: Dict[str, Any]) -> str:
    """Short banner for terminal / walkthrough headers."""
    pol = normalize_demo_polarity(str(snapshot.get("polarity") or ""))
    klasse = int((snapshot.get("final") or {}).get("klasse") or 0)
    gt = (snapshot.get("verification") or {}).get("manual_report_ground_truth")
    if pol == "false_negative" or (klasse == 0 and gt == 1):
        return "FALSE NEGATIVE · Modell kein Delir, manuell Delir"
    if klasse == 1 and gt == 1:
        return "TRUE POSITIVE · Delir"
    if klasse == 0 and gt == 0:
        return "TRUE NEGATIVE · kein Delir"
    if klasse == 1 and gt == 0:
        return "FALSE POSITIVE · Modell Delir, manuell kein Delir"
    return "POSITIVE · Delir" if klasse == 1 else "NEGATIVE · kein Delir"


def anonymize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a presentation-safe copy: no PatientenID, no validation IDs, no report dates.

    Intended for thesis talks and public demos. Original identifiers are discarded.
    """
    out = copy.deepcopy(snapshot)
    polarity = normalize_demo_polarity(str(out.get("polarity") or "positive"))
    if polarity not in PRESENTATION_LABELS:
        gt = (out.get("verification") or {}).get("manual_report_ground_truth")
        klasse = int((out.get("final") or {}).get("klasse") or 0)
        polarity = "positive" if klasse == 1 and gt == 1 else "false_negative"
    labels = PRESENTATION_LABELS[polarity]
    out["polarity"] = polarity

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

CURATED_FN_REPORT = """[Diagnosen]
Sepsis, akutes Nierenversagen.

[Jetziges Leiden]
Seit Aufnahme zunehmende Verwirrtheit und Desorientierung zur Zeit, psychomotorische Verlangsamung.

[Epikrise]
Vigilanz fluktuierend, GCS 13-14. Klinisch Verdacht auf ZNS-Beteiligung bei Sepsis. Neurologisch keine fokalen Ausfälle.

[Verlauf]
Desorientierung und Vigilanzschwankungen über mehrere Tage, auch nach Optimierung der Nierenfunktion.
"""

CURATED_FN_INTERPRETATION: Dict[str, Any] = {
    "delir_signale": {
        "desorientierung": ["Desorientierung zur Zeit", "Verwirrtheit"],
        "delir_explizit": [],
        "hyperaktivitaet_agitation": [],
        "vigilanz": ["Vigilanz fluktuierend", "GCS 13-14"],
        "delir_therapie": [],
        "delir_prophylaxe": [],
    },
    "signalstaerke": "niedrig",
    "delir_probability_estimate": 28,
    "kontext": "Indirekte Vigilanz- und Orientierungsstörung bei Sepsis — Delir möglich, aber nicht explizit dokumentiert.",
    "begruendung": "Schwache indirekte Hinweise ohne explizite Delirdiagnose.",
    "alternative_erklaerung": False,
    "alternative_erklaerung_keywords": [],
}

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
    return parse_delir_signale(raw)


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
    klasse = _report_model_klasse(row)
    if klasse < 0:
        klasse = 0
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


def _curated_replay_row(polarity: str) -> pd.Series:
    pol = normalize_demo_polarity(polarity)
    if pol == "positive":
        interp = CURATED_POSITIVE_INTERPRETATION
        return pd.Series(
            {
                "delir_signale": json.dumps(interp["delir_signale"], ensure_ascii=False),
                "signalstaerke": interp["signalstaerke"],
                "kontext": interp["kontext"],
                "begruendung": interp["begruendung"],
                "alternative_erklaerung": interp["alternative_erklaerung"],
                "decision_rule_applied": "direct_delir_positive",
                "klasse": 1,
            }
        )
    if pol == "false_negative":
        interp = CURATED_FN_INTERPRETATION
        return pd.Series(
            {
                "delir_signale": json.dumps(interp["delir_signale"], ensure_ascii=False),
                "signalstaerke": interp["signalstaerke"],
                "kontext": interp["kontext"],
                "begruendung": interp["begruendung"],
                "alternative_erklaerung": interp["alternative_erklaerung"],
                "decision_rule_applied": "isolated_indirect_not_positive",
                "klasse": 0,
            }
        )
    return pd.Series(
        {
            "delir_signale": "{}",
            "signalstaerke": "niedrig",
            "kontext": "",
            "begruendung": "",
            "decision_rule_applied": "no_evidence_prefilter_skip",
            "klasse": 0,
        }
    )


def build_curated_snapshot(*, polarity: str) -> Dict[str, Any]:
    """Anonymized fallback snapshot when validation CSVs are not available."""
    pol = normalize_demo_polarity(polarity)
    if pol == "positive":
        report_text = CURATED_POSITIVE_REPORT
        bertyp = "Austrittsbericht"
        manual_gt = 1
    else:
        report_text = CURATED_FN_REPORT
        bertyp = "Verlaufseintrag"
        manual_gt = 1
    trace = build_delirium_trace(
        report_text=report_text,
        bertyp=bertyp,
        manual_gt=manual_gt,
        live=False,
        replay_row=_curated_replay_row(pol),
        source="curated_anonymized",
        polarity=pol,
        case_meta={"bertyp": bertyp, "manual_report_ground_truth": manual_gt},
    )
    return anonymize_snapshot(trace)


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
    if labels is not None and not labels.empty and "validation_report_id" in labels.columns:
        try:
            pred = merge_manual_report_labels(pred, labels, log_context="demo snapshot")
        except (ValueError, KeyError):
            label_cols = [
                c
                for c in ("validation_report_id", "manual_report_ground_truth", "manual_comment")
                if c in labels.columns
            ]
            if "validation_report_id" in label_cols:
                lab = labels[label_cols].drop_duplicates("validation_report_id")
                pred = pred.merge(lab, on="validation_report_id", how="left", suffixes=("", "_label"))
                if "manual_report_ground_truth_label" in pred.columns:
                    base = pred.get("manual_report_ground_truth")
                    if base is None:
                        pred["manual_report_ground_truth"] = pred["manual_report_ground_truth_label"]
                    else:
                        pred["manual_report_ground_truth"] = pd.to_numeric(
                            base, errors="coerce"
                        ).combine_first(
                            pd.to_numeric(pred["manual_report_ground_truth_label"], errors="coerce")
                        )
    return pred


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
    if _manual_report_gt(row) != 1:
        return -1
    if _report_model_klasse(row) != 1:
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


def _score_false_negative_row(row: pd.Series) -> int:
    """Prefer verified FN cases; boost configured FN patients (PatientenID 308617 / 308954)."""
    if _manual_report_gt(row) != 1:
        return -1
    if _report_model_klasse(row) != 0:
        return -1
    return _score_fn_illustration_row(row)


def _score_patient_fn_representative_row(row: pd.Series) -> int:
    """Best report to illustrate patient-level FN (model=0, patient manually delir+)."""
    if _report_model_klasse(row) != 0:
        return -1
    return _score_fn_illustration_row(row)


def _score_fn_illustration_row(row: pd.Series) -> int:
    score = 0
    if is_preferred_fn_patient(row):
        score += 1000
    if _manual_report_gt(row) == 1:
        score += 200
    if _bool_cell(row.get("llm_called")):
        score += 35
    if _bool_cell(row.get("llm_skipped_by_prefilter")):
        score += 30
    rule = str(row.get("decision_rule_applied") or "")
    if rule in ("isolated_indirect_not_positive", "alternative_explanation_downgrade"):
        score += 40
    if "indirect" in rule or "niedrig" in rule:
        score += 25
    if _bool_cell(row.get("has_indirect_delir_evidence")):
        score += 20
    if str(row.get("signalstaerke") or "").lower() == "niedrig":
        score += 15
    if _bool_cell(row.get("manual_review_candidate")):
        score += 10
    snippet_count = len(parse_evidence_snippets(row.get("evidence_snippets")))
    if 1 <= snippet_count <= 6:
        score += 10
    return score


def _score_negative_row(row: pd.Series) -> int:
    if _manual_report_gt(row) != 0:
        return -1
    if _report_model_klasse(row) != 0:
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
    """Return top scored TP/FN rows for manual inspection."""
    df = _merge_predictions_with_labels(predictions, labels)
    exclude = {str(x).strip() for x in (exclude_ids or []) if str(x).strip()}
    pol = normalize_demo_polarity(polarity)
    frozen_gt = _load_frozen_patient_manual_gt() if pol == "false_negative" else {}

    if pol == "false_negative":
        ranked = _rank_false_negative_candidates(df, exclude, frozen_gt)
    else:
        scorer = _score_positive_row
        ranked = []
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


def _rank_false_negative_candidates(
    df: pd.DataFrame,
    exclude: set[str],
    frozen_gt: Dict[str, int],
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    patient_has_strict_fn: set[str] = set()

    def _append_candidate(row: pd.Series, score: int, pick_mode: str) -> None:
        vid = str(row.get("validation_report_id") or "").strip()
        if not vid or vid in exclude:
            return
        ranked.append(
            {
                "validation_report_id": vid,
                "score": score,
                "bertyp": str(row.get("bertyp") or ""),
                "decision_rule_applied": str(row.get("decision_rule_applied") or ""),
                "signalstaerke": str(row.get("signalstaerke") or ""),
                "snippet_count": len(parse_evidence_snippets(row.get("evidence_snippets"))),
                "report_length": _int_cell(row.get("original_report_text_length"), 0),
                "pick_mode": pick_mode,
            }
        )

    for _, row in df.iterrows():
        score = _score_false_negative_row(row)
        if score >= 0:
            _append_candidate(row, score, "report_fn")
            vpid = str(row.get("validation_patient_id") or "").strip()
            if vpid:
                patient_has_strict_fn.add(vpid)

    if "validation_patient_id" not in df.columns:
        return ranked

    for vpid, grp in df.groupby("validation_patient_id"):
        vpid_s = str(vpid).strip()
        if not vpid_s or vpid_s in patient_has_strict_fn:
            continue
        if not _is_patient_level_fn(grp, frozen_gt):
            continue
        best_row: Optional[pd.Series] = None
        best_score = -1
        for _, row in grp.iterrows():
            score = _score_patient_fn_representative_row(row)
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is not None and best_score >= 0:
            _append_candidate(best_row, best_score, "patient_fn")
    return ranked


def _pick_fn_report_for_patient_suffix(
    merged: pd.DataFrame,
    suffix: str,
    exclude_ids: Optional[Sequence[str]] = None,
    *,
    frozen_gt: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """Best FN report for one patient: report-level FN first, else patient-level FN representative."""
    frozen_gt = frozen_gt if frozen_gt is not None else _load_frozen_patient_manual_gt()
    exclude = {str(x).strip() for x in (exclude_ids or []) if str(x).strip()}
    subset = _patient_subset(merged, suffix)
    if subset.empty:
        return None

    ranked: List[Tuple[int, str]] = []
    for _, row in subset.iterrows():
        vid = str(row.get("validation_report_id") or "").strip()
        if not vid or vid in exclude:
            continue
        score = _score_false_negative_row(row)
        if score >= 0:
            ranked.append((score, vid))
    if ranked:
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return ranked[0][1]

    if not _is_patient_level_fn(subset, frozen_gt):
        return None

    rep_ranked: List[Tuple[int, str]] = []
    for _, row in subset.iterrows():
        vid = str(row.get("validation_report_id") or "").strip()
        if not vid or vid in exclude:
            continue
        score = _score_patient_fn_representative_row(row)
        if score >= 0:
            rep_ranked.append((score, vid))
    if not rep_ranked:
        return None
    rep_ranked.sort(key=lambda x: (-x[0], x[1]))
    LOGGER.info(
        "Picked patient-level FN representative %s for suffix %s (no report-level FN on same row)",
        rep_ranked[0][1],
        suffix,
    )
    return rep_ranked[0][1]


def diagnose_preferred_fn_patients(
    predictions: pd.DataFrame,
    labels: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    """
    Explain preferred FN patients (PatientenID 308617 / 308954) for demo selection.

    Picks report-level FN (model=0, manual=1) when available; otherwise patient-level FN
    (model_patient_positive=0, derived_manual_patient_ground_truth=1) with best model=0 report.
    """
    merged = _merge_predictions_with_labels(predictions, labels)
    frozen_gt = _load_frozen_patient_manual_gt()
    out: List[Dict[str, Any]] = []
    for suffix in PREFERRED_FN_PATIENT_SUFFIXES:
        rows = _patient_subset(merged, suffix)
        model_pos = _computed_patient_model_positive(rows) if not rows.empty else None
        derived = _computed_derived_manual_gt(rows, frozen_gt) if not rows.empty else None
        patient_fn = _is_patient_level_fn(rows, frozen_gt) if not rows.empty else False
        entry: Dict[str, Any] = {
            "patient_suffix": suffix,
            "reports_found": int(len(rows)),
            "patient_level_fn": patient_fn,
            "model_patient_positive": model_pos,
            "derived_manual_patient_ground_truth": derived,
            "patient_confusion_group": assign_confusion_group(model_pos, derived),
            "report_level_fn_reports": [],
            "all_reports": [],
            "pickable_fn_report_id": _pick_fn_report_for_patient_suffix(
                merged, suffix, frozen_gt=frozen_gt
            ),
        }
        for _, row in rows.iterrows():
            model_k = _report_model_klasse(row)
            manual = _manual_report_gt(row)
            vid = str(row.get("validation_report_id") or "")
            if model_k == 0 and manual == 1:
                group = "FN"
            elif model_k == 1 and manual == 1:
                group = "TP"
            elif model_k == 1 and manual == 0:
                group = "FP"
            elif model_k == 0 and manual == 0:
                group = "TN"
            else:
                group = "?"
            rec = {
                "validation_report_id": vid,
                "model_report_prediction": model_k,
                "manual_report_ground_truth": manual,
                "confusion": group,
                "decision_rule_applied": str(row.get("decision_rule_applied") or ""),
            }
            entry["all_reports"].append(rec)
            if group == "FN":
                entry["report_level_fn_reports"].append(vid)
        out.append(entry)
    return out


def autopick_validation_report_id(
    predictions: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    *,
    polarity: str,
    exclude_ids: Optional[Sequence[str]] = None,
    preferred_fn_patient_suffix: Optional[str] = None,
) -> Optional[str]:
    """Pick the clearest verified TP or FN from merged validation predictions."""
    pol = normalize_demo_polarity(polarity)
    merged = _merge_predictions_with_labels(predictions, labels)
    frozen_gt = _load_frozen_patient_manual_gt() if pol == "false_negative" else {}

    if pol == "false_negative":
        suffixes: List[str] = []
        if preferred_fn_patient_suffix:
            suffixes.append(str(preferred_fn_patient_suffix).strip())
        suffixes.extend([s for s in PREFERRED_FN_PATIENT_SUFFIXES if s not in suffixes])
        for suffix in suffixes:
            if not suffix:
                continue
            picked = _pick_fn_report_for_patient_suffix(
                merged, suffix, exclude_ids, frozen_gt=frozen_gt
            )
            if picked:
                LOGGER.info("Picked FN report %s for patient suffix %s", picked, suffix)
                return picked
            subset = _patient_subset(merged, suffix)
            if not subset.empty:
                LOGGER.warning(
                    "Patient %s: %d report(s) in cohort but not pickable as FN "
                    "(patient_level_fn=%s, model_pos=%s, derived_manual=%s).",
                    suffix,
                    len(subset),
                    _is_patient_level_fn(subset, frozen_gt),
                    _computed_patient_model_positive(subset),
                    _computed_derived_manual_gt(subset, frozen_gt),
                )

    ranked = rank_validation_candidates(
        predictions, labels, polarity=polarity, exclude_ids=exclude_ids, top_n=1
    )
    return ranked[0]["validation_report_id"] if ranked else None


def _norm_id(value: object) -> str:
    """Normalize PatientenID / validation IDs for matching (308617.0 → 308617)."""
    s = str(value or "").strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def _load_model_fn_patients() -> pd.DataFrame:
    """Patients classified FN in final manual validation evaluation."""
    fn_path = FINAL_MANUAL_VALIDATION_EVAL_DIR / "model_FN.csv"
    if fn_path.exists():
        df = pd.read_csv(fn_path)
        if not df.empty:
            return df
    gt_path = FINAL_MANUAL_VALIDATION_EVAL_DIR / "patient_level_ground_truth.csv"
    if not gt_path.exists():
        return pd.DataFrame()
    gt = pd.read_csv(gt_path)
    mp = pd.to_numeric(gt.get("model_patient_positive"), errors="coerce")
    dm = pd.to_numeric(gt.get("derived_manual_patient_ground_truth"), errors="coerce")
    return gt.loc[(mp == 0) & (dm == 1)].copy()


def _match_row_by_patienten_id(df: pd.DataFrame, patienten_id: str) -> Optional[pd.Series]:
    if df.empty or "PatientenID" not in df.columns:
        return None
    target = _norm_id(patienten_id)
    for _, row in df.iterrows():
        if _norm_id(row.get("PatientenID")) == target:
            return row
    return None


def _reports_for_patienten_id(merged: pd.DataFrame, patienten_id: str) -> pd.DataFrame:
    if merged.empty:
        return merged
    target = _norm_id(patienten_id)
    if "PatientenID" in merged.columns:
        hits = merged[merged["PatientenID"].map(_norm_id) == target]
        if not hits.empty:
            return hits
    return pd.DataFrame()


def apply_patient_level_fn_verification(
    trace: Dict[str, Any],
    *,
    model_report_prediction: int,
    derived_manual_patient_gt: int = 1,
    model_patient_positive: int = 0,
) -> Dict[str, Any]:
    """
    Force patient-level FN labels into the snapshot verification block.

    Thesis Case B compares model report output vs patient-level manual reference (Delir).
    """
    klasse = int((trace.get("final") or {}).get("klasse") or model_report_prediction)
    trace["polarity"] = "false_negative"
    trace["verification"] = {
        "evaluation_level": "patient",
        "derived_manual_patient_ground_truth": int(derived_manual_patient_gt),
        "manual_report_ground_truth": int(derived_manual_patient_gt),
        "model_patient_positive": int(model_patient_positive),
        "model_report_prediction": int(model_report_prediction),
        "model_correct_vs_manual": False,
        "patient_confusion_group": "FN",
    }
    case = dict(trace.get("case") or {})
    case["manual_report_ground_truth"] = int(derived_manual_patient_gt)
    trace["case"] = case
    final = dict(trace.get("final") or {})
    final["klasse"] = klasse
    trace["final"] = final
    guard = dict(trace.get("guardrails") or {})
    guard["klasse"] = klasse
    trace["guardrails"] = guard
    return trace


def _pick_fn_report_row(
    merged: pd.DataFrame,
    *,
    patienten_id: str,
    fn_eval_row: pd.Series,
    validation_report_id: Optional[str] = None,
) -> pd.Series:
    if validation_report_id:
        hits = merged[merged["validation_report_id"].astype(str) == validation_report_id]
        if hits.empty:
            raise ValueError(f"validation_report_id not found: {validation_report_id}")
        return hits.iloc[0]

    subset = _reports_for_patienten_id(merged, patienten_id)
    if subset.empty and "validation_patient_id" in merged.columns:
        vpid = str(fn_eval_row.get("validation_patient_id") or "").strip()
        if vpid:
            subset = merged[merged["validation_patient_id"].astype(str) == vpid]

    if subset.empty:
        raise ValueError(
            f"No report rows in predictions for PatientenID {_norm_id(patienten_id)}. "
            "Re-run validation cohort predictions export."
        )

    ranked: List[Tuple[int, pd.Series]] = []
    for _, row in subset.iterrows():
        score = _score_patient_fn_representative_row(row)
        if score >= 0:
            ranked.append((score, row))
    if not ranked:
        raise ValueError(
            f"PatientenID {_norm_id(patienten_id)} has reports but none with model=0 for FN demo."
        )
    ranked.sort(key=lambda x: (-x[0], str(x[1].get("validation_report_id") or "")))
    return ranked[0][1]


def _fn_patienten_ids_to_try(requested: Optional[str]) -> List[str]:
    ids: List[str] = []
    if requested and str(requested).strip():
        ids.append(_norm_id(requested))
    for pid in PREFERRED_FN_PATIENTEN_IDS:
        if pid not in ids:
            ids.append(pid)
    return ids


def generate_fn_snapshot_from_eval(
    *,
    out_path: Path,
    patienten_id: Optional[str] = None,
    validation_report_id: Optional[str] = None,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    live: bool = False,
) -> Dict[str, Any]:
    """
    Build Case B from the final evaluation FN list — no heuristic autopick.

    Uses model_FN.csv (or patient_level_ground_truth FN rows), picks one model=0 report
    for the patient, and forces patient-level manual GT = Delir in verification.
    """
    if not predictions_path.exists() or predictions_path.stat().st_size < 200:
        LOGGER.warning("Predictions missing; using curated FN case (offline only).")
        snap = build_curated_snapshot(polarity="false_negative")
        save_snapshot(snap, out_path)
        return load_snapshot(out_path)

    fn_patients = _load_model_fn_patients()
    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path) if labels_path.exists() else None
    merged = _merge_predictions_with_labels(predictions, labels)

    last_error: Optional[str] = None
    for pid in _fn_patienten_ids_to_try(patienten_id):
        fn_row = _match_row_by_patienten_id(fn_patients, pid) if not fn_patients.empty else None
        if fn_row is None:
            subset = _reports_for_patienten_id(merged, pid)
            if subset.empty:
                last_error = f"PatientenID {pid} not in model_FN.csv and no prediction rows."
                continue
            if not _is_patient_level_fn(subset, _load_frozen_patient_manual_gt()):
                last_error = (
                    f"PatientenID {pid} in predictions but not patient-level FN "
                    f"(model_pos={_computed_patient_model_positive(subset)}, "
                    f"derived={_computed_derived_manual_gt(subset, _load_frozen_patient_manual_gt())})."
                )
                continue
            derived = _computed_derived_manual_gt(subset, _load_frozen_patient_manual_gt()) or 1
            model_pos = _computed_patient_model_positive(subset) or 0
        else:
            derived = int(pd.to_numeric(fn_row.get("derived_manual_patient_ground_truth"), errors="coerce") or 1)
            model_pos = int(pd.to_numeric(fn_row.get("model_patient_positive"), errors="coerce") or 0)
            if model_pos != 0 or derived != 1:
                last_error = f"PatientenID {pid} in eval table but not FN (model={model_pos}, manual={derived})."
                continue

        try:
            row = _pick_fn_report_row(
                merged,
                patienten_id=pid,
                fn_eval_row=fn_row if fn_row is not None else subset.iloc[0],
                validation_report_id=validation_report_id if patienten_id is None or _norm_id(patienten_id) == pid else None,
            )
        except ValueError as exc:
            last_error = str(exc)
            continue

        text_index = build_stable_report_text_index(berichte_path)
        report_text = _resolve_report_text(row, text_index)
        if not report_text.strip():
            snippets = parse_evidence_snippets(row.get("evidence_snippets"))
            if snippets:
                report_text = "\n\n".join(str(s.get("text") or "") for s in snippets)
        if not report_text.strip() and fn_row is not None:
            rep = str(fn_row.get("representative_evidence") or "").strip()
            if rep:
                report_text = rep
        if not report_text.strip():
            last_error = f"PatientenID {pid}: report text not found in Berichte.csv."
            continue

        model_report = _report_model_klasse(row)
        if model_report not in (0, 1):
            model_report = 0

        trace = build_delirium_trace(
            report_text=report_text,
            bertyp=str(row.get("bertyp") or ""),
            manual_gt=1,
            live=live,
            replay_row=row,
            source="validation_cohort",
            polarity="false_negative",
            case_meta={
                "validation_report_id": str(row.get("validation_report_id") or ""),
                "validation_patient_id": str(row.get("validation_patient_id") or ""),
                "PatientenID": str(row.get("PatientenID") or pid),
                "bertyp": str(row.get("bertyp") or ""),
                "berdat": str(row.get("berdat") or ""),
                "bericht": str(row.get("bericht") or ""),
                "manual_report_ground_truth": 1,
            },
        )
        trace = apply_patient_level_fn_verification(
            trace,
            model_report_prediction=model_report,
            derived_manual_patient_gt=derived,
            model_patient_positive=model_pos,
        )
        vid = str(row.get("validation_report_id") or "")
        trace["selection"] = {
            "fn_build_mode": "pinned_from_eval",
            "patienten_id": pid,
            "validation_report_id_picked": vid,
            "model_patient_positive": model_pos,
            "derived_manual_patient_ground_truth": derived,
            "model_report_prediction": model_report,
            "capture_mode": "live" if live else "replay_csv",
        }
        save_snapshot(trace, out_path)
        LOGGER.info("FN snapshot: PatientenID %s, report %s (patient-level FN, forced labels).", pid, vid)
        return load_snapshot(out_path)

    available = []
    if not fn_patients.empty and "PatientenID" in fn_patients.columns:
        available = sorted({_norm_id(x) for x in fn_patients["PatientenID"].dropna()})
    msg = (
        "Could not build FN snapshot from evaluation data.\n"
        f"  Tried PatientenID: {', '.join(_fn_patienten_ids_to_try(patienten_id))}\n"
        f"  Last error: {last_error or 'unknown'}\n"
        f"  FN patients in model_FN.csv: {', '.join(available[:20]) or '(file missing — run final manual validation eval)'}\n"
        "  Run: python3 -m src.analysis.demo_delirium_case --diagnose-fn-patients"
    )
    raise ValueError(msg)


def generate_snapshot_from_validation(
    *,
    polarity: str,
    out_path: Path,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    validation_report_id: Optional[str] = None,
    exclude_validation_report_ids: Optional[Sequence[str]] = None,
    preferred_fn_patient_suffix: Optional[str] = None,
    live: bool = False,
) -> Dict[str, Any]:
    """
    Build a snapshot from frozen validation data.

    Falls back to curated anonymized demo cases when predictions are missing or stub-only.
    """
    polarity = normalize_demo_polarity(polarity)
    if polarity == "false_negative":
        return generate_fn_snapshot_from_eval(
            out_path=out_path,
            patienten_id=preferred_fn_patient_suffix,
            validation_report_id=validation_report_id,
            predictions_path=predictions_path,
            labels_path=labels_path,
            berichte_path=berichte_path,
            live=live,
        )

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
            preferred_fn_patient_suffix=preferred_fn_patient_suffix,
        )
        if not picked_id:
            LOGGER.warning("No suitable %s case in validation data; using curated demo.", polarity)
            if polarity == "false_negative":
                LOGGER.warning(
                    "Tip: run --diagnose-fn-patients to inspect FN patients 308617 / 308954 "
                    "(patient-level FN is accepted when report-level FN is missing)."
                )
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

    frozen_gt = _load_frozen_patient_manual_gt()
    if polarity == "false_negative":
        manual_gt = resolve_fn_manual_gt(merged, row, frozen_patient_gt=frozen_gt)
    else:
        manual_gt = _manual_report_gt(row)
        manual_gt = manual_gt if manual_gt >= 0 else None
    trace = build_delirium_trace(
        report_text=report_text,
        bertyp=str(row.get("bertyp") or ""),
        manual_gt=manual_gt,
        live=live,
        replay_row=row,
        source="validation_cohort",
        polarity=polarity,
        case_meta={
            "validation_report_id": str(row.get("validation_report_id") or ""),
            "validation_patient_id": str(row.get("validation_patient_id") or ""),
            "PatientenID": str(row.get("PatientenID") or ""),
            "bertyp": str(row.get("bertyp") or ""),
            "berdat": str(row.get("berdat") or ""),
            "bericht": str(row.get("bericht") or ""),
            "manual_report_ground_truth": manual_gt,
        },
    )
    ranked = rank_validation_candidates(
        predictions,
        labels,
        polarity=polarity,
        exclude_ids=exclude_validation_report_ids,
        top_n=20,
    )
    vid = str(row.get("validation_report_id") or "")
    trace["selection"] = {
        "validation_report_id_picked": vid,
        "rank": next((i + 1 for i, r in enumerate(ranked) if r["validation_report_id"] == vid), None),
        "score": next((r["score"] for r in ranked if r["validation_report_id"] == vid), None),
        "top_candidates": ranked[:5],
        "capture_mode": "live" if live else "replay_csv",
    }
    save_snapshot(trace, out_path)
    return load_snapshot(out_path)


def ensure_default_snapshots() -> Tuple[Path, Path]:
    """Create default demo snapshots under data/demo/ if missing or outdated (v1)."""
    pos_path = DEMO_POSITIVE_SNAPSHOT_PATH
    neg_path = DEMO_NEGATIVE_SNAPSHOT_PATH
    for path, polarity in ((pos_path, "positive"), (neg_path, "false_negative")):
        needs_build = not path.exists()
        if not needs_build:
            try:
                snap = load_snapshot(path)
                needs_build = not trace_is_v2(snap)
                if polarity == "false_negative" and normalize_demo_polarity(str(snap.get("polarity"))) != "false_negative":
                    needs_build = True
            except (json.JSONDecodeError, OSError):
                needs_build = True
        if needs_build:
            generate_snapshot_from_validation(polarity=polarity, out_path=path)
    return pos_path, neg_path


def snippet_section_label(snippet: Dict[str, Any]) -> str:
    sec = str(snippet.get("section") or "unknown")
    return SECTION_DISPLAY.get(sec, sec)
