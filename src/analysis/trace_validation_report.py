"""
Trace one validation_report_id through cohort, labels, predictions, and raw Berichte.

Read-only; does not modify frozen files, labels, or predictions.

Usage:
  python -m src.analysis.trace_validation_report --validation-report-id Patient_0001_Report_0001
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from src.analysis.validation_report_trace import (
    build_report_trace,
    format_trace_report,
    load_trace_inputs,
)
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    FROZEN_MANUAL_REPORT_LABELS_PATH,
    FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    MANUAL_VALIDATION_DIR,
    VALIDATION_COHORT_PREDICTIONS_PATH,
)

LOGGER = logging.getLogger(__name__)

TRACE_REPORTS_DIR = MANUAL_VALIDATION_DIR / "trace_reports"


def _safe_filename(validation_report_id: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", validation_report_id.strip())
    return safe or "unknown_report"


def trace_validation_report(
    validation_report_id: str,
    *,
    cohort_path: Path = FROZEN_PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = FROZEN_MANUAL_REPORT_LABELS_PATH,
    predictions_path: Path = VALIDATION_COHORT_PREDICTIONS_PATH,
    berichte_path: Path = BERICHTE_INPUT_PATH,
    output_dir: Path = TRACE_REPORTS_DIR,
) -> Path:
    cohort, labels, preds, spine, raw_full = load_trace_inputs(
        cohort_path, labels_path, predictions_path, berichte_path
    )
    trace = build_report_trace(
        validation_report_id, cohort, labels, preds, spine, raw_full
    )
    report = format_trace_report(trace)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_safe_filename(validation_report_id)}.txt"
    out_path.write_text(report, encoding="utf-8")
    LOGGER.info("Trace verdict=%s wrote %s", trace.verdict, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one validation report end-to-end.")
    parser.add_argument(
        "--validation-report-id",
        required=True,
        help="e.g. Patient_0001_Report_0001",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACE_REPORTS_DIR,
        help="Directory for trace text file",
    )
    args = parser.parse_args()

    out_path = trace_validation_report(
        args.validation_report_id,
        output_dir=args.output_dir,
    )
    print(out_path.read_text(encoding="utf-8"))
    print(f"Wrote trace to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
