"""
Placeholder: compare repeated validation runs (run_01..run_03) within one prompt version.

Future: patient-level prediction stability across runs (same frozen cohort, same labels).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.pipeline.prompt_run_paths import PROMPT_RUNS_COMPARISON_DIR, PROMPT_RUNS_ROOT

LOGGER = logging.getLogger(__name__)

STABILITY_SUMMARY_CSV = PROMPT_RUNS_COMPARISON_DIR / "stability_summary.csv"
STABILITY_REPORT = PROMPT_RUNS_COMPARISON_DIR / "stability_report.txt"


def patient_predictions_path(version: str, run_id: str) -> Path:
    return (
        PROMPT_RUNS_ROOT
        / version
        / run_id
        / "final_evaluation"
        / "patient_level_ground_truth.csv"
    )


def analyze_prompt_run_stability(
    version: str = "v1",
    run_ids: Sequence[str] = ("run_01", "run_02", "run_03"),
) -> pd.DataFrame:
    """
    Compare model_patient_positive across runs for the same prompt version.

    Returns a per-patient summary (placeholder implementation).
    """
    frames: dict[str, pd.DataFrame] = {}
    for run_id in run_ids:
        path = patient_predictions_path(version, run_id)
        if not path.exists():
            LOGGER.warning("Missing patient GT for %s/%s: %s", version, run_id, path)
            continue
        df = pd.read_csv(path)
        if "validation_patient_id" not in df.columns:
            raise ValueError(f"validation_patient_id missing in {path}")
        col = "model_patient_positive"
        if col not in df.columns:
            raise ValueError(f"{col} missing in {path}")
        frames[run_id] = df[["validation_patient_id", "PatientenID", col]].rename(
            columns={col: f"model_pos_{run_id}"}
        )

    if not frames:
        return pd.DataFrame()

    merged = None
    for run_id, part in frames.items():
        merged = part if merged is None else merged.merge(
            part, on=["validation_patient_id", "PatientenID"], how="outer"
        )

    if merged is None:
        return pd.DataFrame()

    pred_cols = [c for c in merged.columns if c.startswith("model_pos_")]
    if len(pred_cols) >= 2:
        vals = merged[pred_cols].apply(pd.to_numeric, errors="coerce")
        merged["n_distinct_predictions"] = vals.nunique(axis=1)
        merged["stable_across_runs"] = merged["n_distinct_predictions"] <= 1
    return merged


def write_stability_summary(
    version: str = "v1",
    run_ids: Sequence[str] = ("run_01", "run_02", "run_03"),
    output_csv: Path = STABILITY_SUMMARY_CSV,
    output_report: Path = STABILITY_REPORT,
) -> tuple[pd.DataFrame, str]:
    summary = analyze_prompt_run_stability(version=version, run_ids=run_ids)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    unstable = 0
    if not summary.empty and "stable_across_runs" in summary.columns:
        unstable = int((~summary["stable_across_runs"].fillna(False)).sum())

    lines = [
        f"Prompt run stability ({version})",
        "=" * 44,
        f"runs={list(run_ids)}",
        f"patients={len(summary)}",
        f"unstable_patients={unstable}",
        "",
        "Placeholder: run after three repeated inference+evaluation cycles per version.",
        f"Output: {output_csv}",
    ]
    report = "\n".join(lines) + "\n"
    output_report.write_text(report, encoding="utf-8")
    return summary, report


def main() -> None:
    _, report = write_stability_summary()
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
