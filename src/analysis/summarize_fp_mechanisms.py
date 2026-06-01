"""
Qualitative false-positive mechanism summary for final manual validation.

Read-only: does not modify manual labels, predictions, or frozen cohort.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.analysis.export_presentation_examples import parse_evidence_snippets
from src.pipeline.paths import (
    FINAL_MANUAL_VALIDATION_EVAL_DIR,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)
from src.pipeline.validation_report_identity import VALIDATION_REPORT_ID_COL

LOGGER = logging.getLogger(__name__)

MODEL_FP_PATH = FINAL_MANUAL_VALIDATION_EVAL_DIR / "model_FP.csv"
FP_MECHANISM_SUMMARY_CSV = FINAL_MANUAL_VALIDATION_EVAL_DIR / "fp_mechanism_summary.csv"
FP_MECHANISM_REPORT_PATH = FINAL_MANUAL_VALIDATION_EVAL_DIR / "fp_mechanism_report.txt"

ERROR_CATEGORY_ORDER: tuple[str, ...] = (
    "short_report_fulltext_llm_positive",
    "direct_delir_mention_but_manual_negative",
    "sedation_or_extubation",
    "psychiatric_explanation",
    "neurologic_explanation",
    "isolated_disorientation",
    "isolated_agitation_or_unruhe",
    "vigilance_or_somnolence",
    "unclear",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "validation_patient_id",
    "PatientenID",
    "n_positive_model_reports",
    "positive_validation_report_ids",
    "decision_rule_applied_values",
    "signalstaerke_values",
    "llm_text_reduction_method_values",
    "llm_called_values",
    "evidence_snippets_empty_count",
    "evidence_keywords",
    "delir_signale",
    "representative_evidence_text",
    "suggested_error_category",
    "per_report_categories",
)

CATEGORY_PATTERNS: Dict[str, tuple[str, ...]] = {
    "short_report_fulltext_llm_positive": (
        "short_report_no_evidence_fulltext",
        "short_report_fulltext",
    ),
    "direct_delir_mention_but_manual_negative": (
        "direct_delir",
        "delir_explizit",
        "hyperaktives delir",
        "hypoaktives delir",
    ),
    "sedation_or_extubation": (
        "sedierung",
        "sediert",
        "narkose",
        "intubation",
        "extubation",
        "beatmung",
        "analges",
        "opioid",
        "midazolam",
        "propofol",
        "dexmedetomidin",
    ),
    "psychiatric_explanation": (
        "psychiatr",
        "depressiv",
        "angst",
        "halluzin",
        "wahn",
        "schizophren",
        "bipolar",
        "affektiv",
    ),
    "neurologic_explanation": (
        "schlaganfall",
        "epileps",
        "krampf",
        "hirnblutung",
        "subdural",
        "meningit",
        "neurolog",
        "parese",
        "aphas",
    ),
    "isolated_disorientation": (
        "desorient",
        "verwirr",
        "orientierungsstörung",
        "orientierungsstoerung",
    ),
    "isolated_agitation_or_unruhe": (
        "agitation",
        "agitiert",
        "unruh",
        "hyperaktiv",
        "psychomotor",
    ),
    "vigilance_or_somnolence": (
        "vigilanz",
        "somnol",
        "sopor",
        "bewusstseinstrübung",
        "bewusstseinsstoerung",
        "bewusstseinsstörung",
        "schläfrig",
        "schlafrig",
    ),
}


def _norm_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def _unique_pipe(values: Sequence[object]) -> str:
    seen: List[str] = []
    for v in values:
        s = _norm_str(v)
        if s and s not in seen:
            seen.append(s)
    return " | ".join(seen)


def _report_text_blob(row: pd.Series) -> str:
    parts: List[str] = []
    for col in (
        "evidence_snippets",
        "delir_signale",
        "kontext",
        "begruendung",
        "klassifikation_begruendung",
    ):
        if col in row.index:
            parts.append(_norm_str(row.get(col)))
    return " ".join(parts).lower()


def _collect_evidence_keywords(rows: pd.DataFrame) -> str:
    keywords: List[str] = []
    for _, row in rows.iterrows():
        for snip in parse_evidence_snippets(row.get("evidence_snippets")):
            kw = _norm_str(snip.get("keyword"))
            if kw and kw not in keywords:
                keywords.append(kw)
    return " | ".join(keywords)


def _representative_evidence_text(pos_rows: pd.DataFrame) -> str:
    """Prefer snippet text from highest-signal positive report."""
    if pos_rows.empty:
        return ""

    ordered = pos_rows.copy()
    if "signalstaerke" in ordered.columns:
        rank = {"hoch": 0, "mittel": 1, "niedrig": 2}
        ordered["_sig_rank"] = ordered["signalstaerke"].map(
            lambda x: rank.get(_norm_str(x).lower(), 9)
        )
        ordered = ordered.sort_values("_sig_rank", kind="mergesort")

    for _, row in ordered.iterrows():
        snippets = parse_evidence_snippets(row.get("evidence_snippets"))
        for snip in snippets:
            text = _norm_str(snip.get("text"))
            if text:
                return text[:500]
        signals = _norm_str(row.get("delir_signale"))
        if signals:
            return signals[:500]
        kontext = _norm_str(row.get("kontext"))
        if kontext:
            return kontext[:500]
    return ""


def _classify_report_row(row: pd.Series) -> str:
    blob = _report_text_blob(row)
    method = _norm_str(row.get("llm_text_reduction_method")).lower()
    rule = _norm_str(row.get("decision_rule_applied")).lower()
    snippets = parse_evidence_snippets(row.get("evidence_snippets"))

    if "short_report" in method or method == "short_report_no_evidence_fulltext":
        return "short_report_fulltext_llm_positive"

    if rule.startswith("direct_delir") or any(
        str(s.get("evidence_type", "")) == "direct_delir" for s in snippets
    ):
        return "direct_delir_mention_but_manual_negative"

    for category in ERROR_CATEGORY_ORDER:
        if category in ("short_report_fulltext_llm_positive", "direct_delir_mention_but_manual_negative", "unclear"):
            continue
        patterns = CATEGORY_PATTERNS.get(category, ())
        if any(p in blob or p in rule for p in patterns):
            return category

    if not snippets and method and "no_evidence" not in method:
        return "short_report_fulltext_llm_positive"

    return "unclear"


def _patient_category(report_categories: Sequence[str]) -> str:
    for cat in ERROR_CATEGORY_ORDER:
        if cat in report_categories:
            return cat
    return "unclear"


def _positive_prediction_rows(
    preds: pd.DataFrame,
    validation_patient_id: str,
    patienten_id: str,
    *,
    allowed_report_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    if preds.empty or "klasse" not in preds.columns:
        return preds.iloc[0:0].copy()

    work = preds.copy()
    work["_klasse"] = pd.to_numeric(work["klasse"], errors="coerce").fillna(0).astype(int)
    pos = work[work["_klasse"] == 1].copy()

    if validation_patient_id and "validation_patient_id" in pos.columns:
        pos = pos[pos["validation_patient_id"].astype(str) == validation_patient_id]
    elif patienten_id and "PatientenID" in pos.columns:
        pos = pos[pos["PatientenID"].astype(str) == patienten_id]

    if allowed_report_ids and VALIDATION_REPORT_ID_COL in pos.columns:
        pos = pos[pos[VALIDATION_REPORT_ID_COL].astype(str).map(_norm_str).isin(allowed_report_ids)]

    return pos.drop(columns=["_klasse"], errors="ignore")


def build_fp_mechanism_summary(
    fp_patients: pd.DataFrame,
    predictions: pd.DataFrame,
    cohort: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
  Build one summary row per FP patient from model_FP.csv and report-level predictions.
    """
    if fp_patients.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    cohort_ids_by_patient: Dict[str, set[str]] = {}
    if cohort is not None and not cohort.empty:
        if "validation_patient_id" in cohort.columns and VALIDATION_REPORT_ID_COL in cohort.columns:
            for vpid, grp in cohort.groupby("validation_patient_id"):
                cohort_ids_by_patient[str(vpid)] = set(
                    grp[VALIDATION_REPORT_ID_COL].astype(str).map(_norm_str)
                )
                cohort_ids_by_patient[str(vpid)].discard("")

    rows: List[Dict[str, Any]] = []
    for _, fp in fp_patients.iterrows():
        vpid = _norm_str(fp.get("validation_patient_id"))
        pid = _norm_str(fp.get("PatientenID"))

        allowed_ids = cohort_ids_by_patient.get(vpid)
        pos = _positive_prediction_rows(
            predictions,
            vpid,
            pid,
            allowed_report_ids=allowed_ids if allowed_ids else None,
        )

        report_categories = [_classify_report_row(pos.loc[idx]) for idx in pos.index]
        empty_snip_count = 0
        for _, r in pos.iterrows():
            if not parse_evidence_snippets(r.get("evidence_snippets")):
                empty_snip_count += 1

        pos_ids = (
            pos[VALIDATION_REPORT_ID_COL].astype(str).tolist()
            if VALIDATION_REPORT_ID_COL in pos.columns
            else []
        )

        rows.append(
            {
                "validation_patient_id": vpid,
                "PatientenID": pid,
                "n_positive_model_reports": int(len(pos)),
                "positive_validation_report_ids": " | ".join(
                    [_norm_str(x) for x in pos_ids if _norm_str(x)]
                ),
                "decision_rule_applied_values": _unique_pipe(
                    pos["decision_rule_applied"] if "decision_rule_applied" in pos.columns else []
                ),
                "signalstaerke_values": _unique_pipe(
                    pos["signalstaerke"] if "signalstaerke" in pos.columns else []
                ),
                "llm_text_reduction_method_values": _unique_pipe(
                    pos["llm_text_reduction_method"]
                    if "llm_text_reduction_method" in pos.columns
                    else []
                ),
                "llm_called_values": _unique_pipe(
                    pos["llm_called"] if "llm_called" in pos.columns else []
                ),
                "evidence_snippets_empty_count": empty_snip_count,
                "evidence_keywords": _collect_evidence_keywords(pos),
                "delir_signale": _unique_pipe(
                    pos["delir_signale"] if "delir_signale" in pos.columns else []
                ),
                "representative_evidence_text": _representative_evidence_text(pos),
                "suggested_error_category": _patient_category(report_categories),
                "per_report_categories": " | ".join(report_categories),
            }
        )

    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def format_fp_mechanism_report(summary: pd.DataFrame) -> str:
    lines = [
        "False positive mechanism summary (final manual validation)",
        "=" * 56,
        f"fp_patients={len(summary)}",
        "",
    ]
    if summary.empty:
        lines.append("No false positive patients in model_FP.csv.")
        return "\n".join(lines) + "\n"

    if "suggested_error_category" in summary.columns:
        counts = summary["suggested_error_category"].value_counts()
        lines.append("Category counts")
        lines.append("-" * 56)
        for cat, n in counts.items():
            lines.append(f"  {cat}: {n}")
        lines.append("")

    lines.append("Per-patient detail")
    lines.append("-" * 56)
    for _, row in summary.iterrows():
        lines.append(f"\n{row.get('validation_patient_id', '')} (PatientenID={row.get('PatientenID', '')})")
        lines.append(f"  category={row.get('suggested_error_category', '')}")
        lines.append(f"  n_positive_model_reports={row.get('n_positive_model_reports', 0)}")
        lines.append(f"  positive_reports={row.get('positive_validation_report_ids', '')}")
        lines.append(f"  decision_rules={row.get('decision_rule_applied_values', '')}")
        lines.append(f"  signalstaerke={row.get('signalstaerke_values', '')}")
        lines.append(f"  llm_methods={row.get('llm_text_reduction_method_values', '')}")
        lines.append(f"  empty_evidence_snippets_on_positives={row.get('evidence_snippets_empty_count', 0)}")
        lines.append(f"  keywords={row.get('evidence_keywords', '')}")
        if row.get("delir_signale"):
            lines.append(f"  delir_signale={row.get('delir_signale', '')}")
        if row.get("representative_evidence_text"):
            lines.append(f"  representative_evidence={row.get('representative_evidence_text', '')}")
        if row.get("per_report_categories"):
            lines.append(f"  per_report_categories={row.get('per_report_categories', '')}")

    lines.append("")
    return "\n".join(lines)


def run_fp_mechanism_summary(
    fp_path: Path = MODEL_FP_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    output_csv: Path = FP_MECHANISM_SUMMARY_CSV,
    output_report: Path = FP_MECHANISM_REPORT_PATH,
) -> Tuple[pd.DataFrame, str]:
    if not fp_path.exists():
        raise FileNotFoundError(f"model_FP.csv missing: {fp_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions missing: {predictions_path}")

    fp_patients = pd.read_csv(fp_path)
    preds = pd.read_csv(predictions_path)
    cohort = pd.read_csv(cohort_path) if cohort_path.exists() else None

    summary = build_fp_mechanism_summary(fp_patients, preds, cohort)
    report = format_fp_mechanism_report(summary)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    output_report.write_text(report, encoding="utf-8")
    LOGGER.info("Wrote FP mechanism summary: %s (%d patients)", output_csv, len(summary))
    return summary, report


def main() -> None:
    _, report = run_fp_mechanism_summary()
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
