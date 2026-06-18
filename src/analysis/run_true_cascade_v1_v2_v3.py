"""
Experimental V1→V2→V3 cascade pipeline for the frozen manual validation cohort.

Report flow:
  all reports → V1 (full inference)
  V1 negative → final negative
  V1 positive → V2 (full inference on original text)
  V2 positive → final positive
  V2 negative → V3 adjudicator → final positive/negative

Patient aggregation: max(report_predictions) on complete manual patients.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.agents.cascade_adjudicator_v3 import adjudicate_cascade_v3
from src.analysis.build_manual_validation_progress import assign_confusion_group
from src.analysis.final_manual_validation_evaluation import (
    MANUAL_GT_COL,
    attach_structured_baseline,
    compute_method_metrics,
    primary_evaluation_cohort,
)
from src.analysis.manual_report_labels import merge_manual_report_labels
from src.pipeline.cascade_report_inference import infer_report_with_prompt_version
from src.pipeline.frozen_cohort_inference import build_pipeline_records_from_frozen_cohort
from src.pipeline.paths import (
    CASCADE_V1_V2_V3_RUN_01_DIR,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    STRUCTURED_BASELINE_PATH,
)
from src.pipeline.prompt_run_paths import get_prompt_run_dir
from src.pipeline.validation_report_identity import (
    VALIDATION_PATIENT_ID_COL,
    VALIDATION_REPORT_ID_COL,
)

LOGGER = logging.getLogger(__name__)

V1_CHECKPOINT = "checkpoints/v1_inference.jsonl"
V2_CHECKPOINT = "checkpoints/v2_inference.jsonl"
V3_OUTPUTS = "v3_outputs.jsonl"

CASCADE_ERROR_COLUMNS: tuple[str, ...] = (
    "validation_patient_id",
    "PatientenID",
    MANUAL_GT_COL,
    "cascade_patient_positive",
    "v1_patient_positive",
    "v2_patient_positive",
    "baseline_icdsc_ge_4",
    "baseline_icd10",
    "baseline_composite_or",
    "baseline_composite_and",
    "n_reports_total",
    "n_positive_reports_manual",
)

COMPARISON_METHODS: tuple[tuple[str, str], ...] = (
    ("cascade_patient_positive", "cascade"),
    ("v1_patient_positive", "v1"),
    ("v2_patient_positive", "v2"),
    ("baseline_icdsc_ge_4", "icdsc"),
    ("baseline_icd10", "icd10"),
    ("baseline_composite_or", "composite_or"),
    ("baseline_composite_and", "composite_and"),
)


def _norm_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def load_jsonl_index(path: Path, key: str = VALIDATION_REPORT_ID_COL) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kid = _norm_id(row.get(key))
            if kid:
                out[kid] = row
    return out


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _binary_klasse(row: Dict[str, Any]) -> int:
    return int(row.get("klasse") or 0)


def _stage_row(stage: str, report_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        VALIDATION_REPORT_ID_COL: report_id,
        "stage": stage,
        "klasse": _binary_klasse(row),
        "signalstaerke": row.get("signalstaerke", ""),
        "kontext": row.get("kontext", ""),
        "begruendung": row.get("begruendung", ""),
        "decision_rule_applied": row.get("decision_rule_applied", ""),
        "evidence_snippets": row.get("evidence_snippets", ""),
        "delir_signale": row.get("delir_signale", ""),
        "status": row.get("status", ""),
        "prompt_version": row.get("prompt_version", ""),
        "full_row": row,
    }


def run_cascade_inference(
    records: Sequence[dict],
    output_dir: Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
    max_v3: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Execute cascade stages. Returns report rows, v3 queue metadata, and stage counts.
    """
    v1_path = output_dir / V1_CHECKPOINT
    v2_path = output_dir / V2_CHECKPOINT
    v3_path = output_dir / V3_OUTPUTS

    v1_done = load_jsonl_index(v1_path) if resume else {}
    v2_done = load_jsonl_index(v2_path) if resume else {}
    v3_done = load_jsonl_index(v3_path) if resume else {}

    report_rows: List[Dict[str, Any]] = []
    v3_queue_rows: List[Dict[str, Any]] = []
    counts = {
        "n_reports": len(records),
        "v1_negative_final": 0,
        "v2_confirmed_final": 0,
        "v3_adjudicated_final": 0,
        "v1_positive_to_v2": 0,
        "v3_queue": 0,
        "v3_calls_planned": 0,
        "v3_calls_made": 0,
    }
    v3_calls_made = 0

    for record in records:
        report_id = _norm_id(record.get(VALIDATION_REPORT_ID_COL))
        if not report_id:
            raise ValueError("Pipeline record missing validation_report_id")

        if report_id in v1_done:
            v1_row = v1_done[report_id].get("full_row") or v1_done[report_id]
        elif dry_run:
            v1_row = {"klasse": 0, "status": "dry_run"}
        else:
            v1_row = infer_report_with_prompt_version(record, "v1")
            append_jsonl(v1_path, _stage_row("v1", report_id, v1_row))
            v1_done[report_id] = _stage_row("v1", report_id, v1_row)

        v1_klasse = _binary_klasse(v1_row)
        cascade_stage = "v1_negative"
        cascade_klasse = 0
        v2_row: Optional[Dict[str, Any]] = None
        v3_row: Optional[Dict[str, Any]] = None

        if v1_klasse == 1:
            counts["v1_positive_to_v2"] += 1
            if report_id in v2_done:
                v2_row = v2_done[report_id].get("full_row") or v2_done[report_id]
            elif dry_run:
                v2_row = {"klasse": 0, "status": "dry_run"}
            else:
                v2_row = infer_report_with_prompt_version(record, "v2")
                append_jsonl(v2_path, _stage_row("v2", report_id, v2_row))
                v2_done[report_id] = _stage_row("v2", report_id, v2_row)

            v2_klasse = _binary_klasse(v2_row)
            if v2_klasse == 1:
                cascade_stage = "v2_confirmed"
                cascade_klasse = 1
                counts["v2_confirmed_final"] += 1
            else:
                counts["v3_queue"] += 1
                queue_meta = {
                    VALIDATION_REPORT_ID_COL: report_id,
                    VALIDATION_PATIENT_ID_COL: _norm_id(record.get(VALIDATION_PATIENT_ID_COL)),
                    "PatientenID": _norm_id(record.get("PatientenID")),
                    "bericht": _norm_id(record.get("bericht")),
                    "v1_klasse": v1_klasse,
                    "v2_klasse": v2_klasse,
                }
                v3_queue_rows.append(queue_meta)

                if report_id in v3_done:
                    v3_row = v3_done[report_id]
                    cascade_klasse = _binary_klasse(v3_row)
                    cascade_stage = "v3_adjudicated"
                    counts["v3_adjudicated_final"] += 1
                elif dry_run:
                    counts["v3_calls_planned"] += 1
                    cascade_stage = "v3_pending"
                    cascade_klasse = pd.NA  # type: ignore[assignment]
                else:
                    if max_v3 is not None and v3_calls_made >= max_v3:
                        cascade_stage = "v3_pending"
                        cascade_klasse = pd.NA  # type: ignore[assignment]
                    else:
                        counts["v3_calls_planned"] += 1
                        v3_result = adjudicate_cascade_v3(
                            str(record.get("report_text", "") or ""),
                            v1_output=v1_row,
                            v2_output=v2_row,
                            patient_id=_norm_id(record.get("PatientenID")),
                            report_name=_norm_id(record.get("bericht")),
                        )
                        v3_row = {
                            VALIDATION_REPORT_ID_COL: report_id,
                            **v3_result,
                        }
                        append_jsonl(v3_path, v3_row)
                        v3_done[report_id] = v3_row
                        v3_calls_made += 1
                        counts["v3_calls_made"] += 1
                        cascade_klasse = _binary_klasse(v3_row)
                        cascade_stage = "v3_adjudicated"
                        counts["v3_adjudicated_final"] += 1
        else:
            counts["v1_negative_final"] += 1

        report_rows.append(
            {
                VALIDATION_REPORT_ID_COL: report_id,
                VALIDATION_PATIENT_ID_COL: _norm_id(record.get(VALIDATION_PATIENT_ID_COL)),
                "PatientenID": _norm_id(record.get("PatientenID")),
                "bericht": _norm_id(record.get("bericht")),
                "bertyp": _norm_id(record.get("bertyp")),
                "v1_klasse": v1_klasse,
                "v2_klasse": _binary_klasse(v2_row) if v2_row is not None else pd.NA,
                "v3_klasse": _binary_klasse(v3_row) if v3_row is not None else pd.NA,
                "cascade_klasse": cascade_klasse,
                "cascade_stage": cascade_stage,
                "v1_signalstaerke": v1_row.get("signalstaerke", ""),
                "v2_signalstaerke": v2_row.get("signalstaerke", "") if v2_row else "",
                "v3_signalstaerke": v3_row.get("signalstaerke", "") if v3_row else "",
                "v1_decision_rule": v1_row.get("decision_rule_applied", ""),
                "v2_decision_rule": v2_row.get("decision_rule_applied", "") if v2_row else "",
            }
        )

    return report_rows, v3_queue_rows, counts


def aggregate_patient_predictions(
    report_df: pd.DataFrame,
    klasse_col: str,
    out_col: str,
) -> pd.DataFrame:
    work = report_df.copy()
    work["_pred"] = pd.to_numeric(work[klasse_col], errors="coerce")
    agg = (
        work.groupby(VALIDATION_PATIENT_ID_COL, sort=True)["_pred"]
        .max()
        .reset_index()
        .rename(columns={"_pred": out_col})
    )
    return agg


def load_archived_v2_predictions(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        LOGGER.warning(
            "Archived V2 predictions missing for comparison: %s. "
            "V2 comparison metrics will be omitted.",
            path,
        )
        return None
    return pd.read_csv(path)


def build_patient_evaluation_table(
    report_df: pd.DataFrame,
    *,
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    archived_v2_path: Path,
) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path)
    labels = pd.read_csv(labels_path)
    merged = merge_manual_report_labels(cohort, labels, log_context="cascade evaluation")
    merged = attach_structured_baseline(merged, baseline_path)

    cascade_cols = report_df[
        [
            VALIDATION_REPORT_ID_COL,
            "cascade_klasse",
            "v1_klasse",
        ]
    ].copy()
    merged = merged.merge(cascade_cols, on=VALIDATION_REPORT_ID_COL, how="left")
    merged["model_report_prediction"] = pd.to_numeric(merged["cascade_klasse"], errors="coerce")

    v2_preds = load_archived_v2_predictions(archived_v2_path)
    if v2_preds is not None:
        v2_small = v2_preds[[VALIDATION_REPORT_ID_COL, "klasse"]].rename(
            columns={"klasse": "v2_standalone_klasse"}
        )
        merged_v2 = merged.merge(v2_small, on=VALIDATION_REPORT_ID_COL, how="left")
    else:
        merged_v2 = merged.copy()
        merged_v2["v2_standalone_klasse"] = pd.NA

    patient_rows: List[Dict[str, Any]] = []
    for vpid, grp in merged_v2.groupby(VALIDATION_PATIENT_ID_COL, sort=True):
        n_total = int(len(grp))
        gt_col = grp["manual_report_ground_truth"] if "manual_report_ground_truth" in grp.columns else pd.Series(dtype=object)
        labeled = gt_col.notna() & (gt_col.astype(str).str.strip() != "")
        n_labeled = int(labeled.sum())
        n_missing = n_total - n_labeled
        is_complete = n_total > 0 and n_missing == 0
        n_pos = int((pd.to_numeric(gt_col[labeled], errors="coerce") == 1).sum()) if n_labeled else 0
        derived: Any = (1 if n_pos > 0 else 0) if is_complete else pd.NA

        cascade_pred = pd.to_numeric(grp["cascade_klasse"], errors="coerce")
        v1_pred = pd.to_numeric(grp["v1_klasse"], errors="coerce")
        v2_pred = pd.to_numeric(grp["v2_standalone_klasse"], errors="coerce")

        def _patient_max(series: pd.Series) -> Any:
            valid = series[series.isin([0, 1])]
            return int(valid.max()) if not valid.empty else pd.NA

        pid = ""
        if "PatientenID" in grp.columns:
            pvals = grp["PatientenID"].dropna()
            if len(pvals):
                pid = str(pvals.iloc[0])

        baseline = {}
        for col in (
            "baseline_icdsc_ge_4",
            "baseline_icd10",
            "baseline_composite_or",
            "baseline_composite_and",
        ):
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                baseline[col] = int(vals.iloc[0]) if len(vals) else pd.NA
            else:
                baseline[col] = pd.NA

        patient_rows.append(
            {
                VALIDATION_PATIENT_ID_COL: vpid,
                "PatientenID": pid,
                "n_reports_total": n_total,
                "n_reports_labeled": n_labeled,
                "n_reports_missing_label": n_missing,
                "is_patient_complete": int(is_complete),
                "n_positive_reports_manual": n_pos,
                MANUAL_GT_COL: derived,
                "cascade_patient_positive": _patient_max(cascade_pred),
                "v1_patient_positive": _patient_max(v1_pred),
                "v2_patient_positive": _patient_max(v2_pred),
                **baseline,
            }
        )

    return pd.DataFrame(patient_rows)


def evaluate_cascade_methods(patient_gt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    complete = primary_evaluation_cohort(patient_gt)
    manual = complete[MANUAL_GT_COL]
    metric_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []

    for pred_col, method_key in COMPARISON_METHODS:
        if pred_col not in complete.columns:
            LOGGER.warning("Skipping method %s: column %s missing", method_key, pred_col)
            continue
        m = compute_method_metrics(manual, complete[pred_col], method_name=method_key)
        metric_rows.append(m)
        confusion_rows.append(
            {"method": method_key, "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]}
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)


def export_cascade_error_slices(complete: pd.DataFrame, output_dir: Path) -> None:
    work = complete.copy()
    work["cascade_confusion_group"] = work.apply(
        lambda r: assign_confusion_group(
            r.get("cascade_patient_positive"),
            r.get(MANUAL_GT_COL),
        ),
        axis=1,
    )
    for label in ("TP", "FP", "TN", "FN"):
        subset = work[work["cascade_confusion_group"] == label]
        rows = [{col: subset.loc[idx].get(col, "") for col in CASCADE_ERROR_COLUMNS} for idx in subset.index]
        out = pd.DataFrame(rows, columns=list(CASCADE_ERROR_COLUMNS))
        out.to_csv(output_dir / f"cascade_{label}.csv", index=False)


def format_cascade_report(
    counts: Dict[str, int],
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    dry_run: bool,
    n_complete: int,
    n_total_patients: int,
) -> str:
    lines = [
        "True cascade V1→V2→V3 manual validation evaluation",
        "=" * 52,
        "",
        f"output_dir={output_dir}",
        f"dry_run={dry_run}",
        "",
        "Report-level cascade counts",
        "-" * 52,
        f"n_reports={counts.get('n_reports', 0)}",
        f"v1_negative_final={counts.get('v1_negative_final', 0)}",
        f"v1_positive_to_v2={counts.get('v1_positive_to_v2', 0)}",
        f"v2_confirmed_final={counts.get('v2_confirmed_final', 0)}",
        f"v3_queue={counts.get('v3_queue', 0)}",
        f"v3_adjudicated_final={counts.get('v3_adjudicated_final', 0)}",
        f"v3_calls_made={counts.get('v3_calls_made', 0)}",
        "",
        "Patient-level evaluation (complete manual patients only)",
        "-" * 52,
        f"n_patients_total={n_total_patients}",
        f"n_patients_complete={n_complete}",
        "",
    ]
    if not metrics.empty:
        lines.append("Metrics by method")
        lines.append("-" * 52)
        for _, row in metrics.iterrows():
            lines.append(
                f"{row['method']}: TP={int(row['tp'])} FP={int(row['fp'])} "
                f"TN={int(row['tn'])} FN={int(row['fn'])} "
                f"sens={row['sensitivity']:.4f} spec={row['specificity']:.4f} "
                f"PPV={row['ppv']:.4f} NPV={row['npv']:.4f} "
                f"F1={row['f1']:.4f} acc={row['accuracy']:.4f}"
            )
    return "\n".join(lines) + "\n"


def run_true_cascade(
    output_dir: Path = CASCADE_V1_V2_V3_RUN_01_DIR,
    *,
    dry_run: bool = False,
    resume: bool = False,
    max_v3: Optional[int] = None,
    archived_v2_path: Optional[Path] = None,
) -> str:
    if not FROZEN_PATIENT_VALIDATION_COHORT_PATH.exists():
        raise FileNotFoundError(
            f"Frozen validation cohort missing: {FROZEN_PATIENT_VALIDATION_COHORT_PATH}. "
            "Freeze or restore the cohort before running the cascade."
        )
    if not FROZEN_MANUAL_REPORT_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Frozen manual labels missing: {FROZEN_MANUAL_REPORT_LABELS_PATH}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_pipeline_records_from_frozen_cohort()
    report_rows, v3_queue_rows, counts = run_cascade_inference(
        records,
        output_dir,
        dry_run=dry_run,
        resume=resume,
        max_v3=max_v3,
    )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(output_dir / "cascade_report_predictions.csv", index=False)

    queue_df = pd.DataFrame(v3_queue_rows)
    queue_df.to_csv(output_dir / "v3_queue.csv", index=False)

    if dry_run or report_df["cascade_klasse"].isna().any():
        pending = int(report_df["cascade_klasse"].isna().sum())
        v2_path = archived_v2_path or (
            get_prompt_run_dir(version="v2", run_id="run_01")
            / "predictions"
            / "validation_cohort_predictions.csv"
        )
        report_text = format_cascade_report(
            counts,
            pd.DataFrame(),
            output_dir=output_dir,
            dry_run=dry_run,
            n_complete=0,
            n_total_patients=0,
        )
        report_text += f"\nPending cascade_klasse on {pending} reports (V3 not run or dry-run).\n"
        (output_dir / "report.txt").write_text(report_text, encoding="utf-8")
        return report_text

    v2_path = archived_v2_path or (
        get_prompt_run_dir(version="v2", run_id="run_01")
        / "predictions"
        / "validation_cohort_predictions.csv"
    )
    patient_gt = build_patient_evaluation_table(report_df, archived_v2_path=v2_path)
    patient_gt.to_csv(output_dir / "cascade_patient_predictions.csv", index=False)

    complete = primary_evaluation_cohort(patient_gt)
    metrics, confusion = evaluate_cascade_methods(patient_gt)
    metrics.to_csv(output_dir / "final_metrics_summary.csv", index=False)
    confusion.to_csv(output_dir / "confusion_counts.csv", index=False)
    export_cascade_error_slices(complete, output_dir)

    report_text = format_cascade_report(
        counts,
        metrics,
        output_dir=output_dir,
        dry_run=dry_run,
        n_complete=len(complete),
        n_total_patients=len(patient_gt),
    )
    (output_dir / "report.txt").write_text(report_text, encoding="utf-8")
    return report_text


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experimental V1→V2→V3 cascade on frozen validation cohort."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CASCADE_V1_V2_V3_RUN_01_DIR,
        help="Output directory (default: cascade_v1_v2_v3/run_01)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No LLM calls; write queue and planned counts only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/v1_inference.jsonl, v2_inference.jsonl, v3_outputs.jsonl.",
    )
    parser.add_argument(
        "--max-v3",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N new V3 adjudications in this run.",
    )
    parser.add_argument(
        "--archived-v2-path",
        type=Path,
        default=None,
        help="Read-only V2 standalone predictions for comparison (default: prompt_runs/v2/run_01).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    report = run_true_cascade(
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        max_v3=args.max_v3,
        archived_v2_path=args.archived_v2_path,
    )
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
