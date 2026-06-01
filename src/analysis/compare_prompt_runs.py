"""
Compare final manual validation metrics between prompt versions (V1 vs V2).

Default: v1/run_01 vs v2/run_01 under prompt_runs/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.pipeline.prompt_run_paths import PROMPT_RUNS_COMPARISON_DIR, PROMPT_RUNS_ROOT

LOGGER = logging.getLogger(__name__)

V1_METRICS_PATH = (
    PROMPT_RUNS_ROOT / "v1" / "run_01" / "final_evaluation" / "final_metrics_summary.csv"
)
V2_METRICS_PATH = (
    PROMPT_RUNS_ROOT / "v2" / "run_01" / "final_evaluation" / "final_metrics_summary.csv"
)
COMPARISON_CSV = PROMPT_RUNS_COMPARISON_DIR / "v1_vs_v2_metrics.csv"
COMPARISON_REPORT = PROMPT_RUNS_COMPARISON_DIR / "v1_vs_v2_report.txt"


def load_metrics(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics missing for {label}: {path}")
    df = pd.read_csv(path)
    df["prompt_run"] = label
    return df


def compare_metrics(
    v1_path: Path = V1_METRICS_PATH,
    v2_path: Path = V2_METRICS_PATH,
) -> pd.DataFrame:
    v1 = load_metrics(v1_path, "v1_run_01")
    v2 = load_metrics(v2_path, "v2_run_01")

    key_cols = [c for c in ("method", "reference", "metric") if c in v1.columns and c in v2.columns]
    if not key_cols:
        key_cols = [c for c in v1.columns if c in v2.columns and c != "prompt_run"]

    merged = v1.merge(
        v2,
        on=key_cols,
        how="outer",
        suffixes=("_v1", "_v2"),
    )

    value_cols_v1 = [c for c in merged.columns if c.endswith("_v1") and c != "prompt_run_v1"]
    for col_v1 in value_cols_v1:
        base = col_v1[: -len("_v1")]
        col_v2 = f"{base}_v2"
        if col_v2 in merged.columns:
            v1_num = pd.to_numeric(merged[col_v1], errors="coerce")
            v2_num = pd.to_numeric(merged[col_v2], errors="coerce")
            merged[f"{base}_delta_v2_minus_v1"] = v2_num - v1_num

    return merged


def format_comparison_report(comparison: pd.DataFrame) -> str:
    lines = [
        "V1 vs V2 prompt comparison (run_01)",
        "=" * 44,
        f"v1_metrics={V1_METRICS_PATH}",
        f"v2_metrics={V2_METRICS_PATH}",
        f"rows={len(comparison)}",
        "",
    ]
    if comparison.empty:
        lines.append("No metrics to compare.")
    else:
        preview_cols = [
            c
            for c in comparison.columns
            if any(x in c for x in ("method", "metric", "precision", "recall", "f1", "delta"))
        ]
        preview = comparison[preview_cols].head(20) if preview_cols else comparison.head(20)
        lines.append(preview.to_string(index=False))
    lines.extend(
        [
            "",
            "Note: V2 was developed after FP analysis on the same frozen cohort;",
            "same-cohort performance may be optimistic (not an independent hold-out).",
        ]
    )
    return "\n".join(lines) + "\n"


def write_prompt_run_comparison(
    v1_path: Path = V1_METRICS_PATH,
    v2_path: Path = V2_METRICS_PATH,
    output_csv: Path = COMPARISON_CSV,
    output_report: Path = COMPARISON_REPORT,
) -> tuple[pd.DataFrame, str]:
    comparison = compare_metrics(v1_path, v2_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)
    report = format_comparison_report(comparison)
    output_report.write_text(report, encoding="utf-8")
    LOGGER.info("Wrote comparison: %s", output_csv)
    return comparison, report


def main() -> None:
    _, report = write_prompt_run_comparison()
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
