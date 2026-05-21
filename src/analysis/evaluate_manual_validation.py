"""
Evaluate annotated patient validation cohort (report- and patient-level metrics).

Input (either):
  - Annotated patient_validation_cohort.csv, or
  - patient_validation_cohort.csv + manual_report_labels.csv (merged by validation_report_id)

Output: outputs/analysis/manual_validation/evaluation/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.analysis.manual_report_labels import load_cohort_for_manual_evaluation
from src.analysis.manual_validation_eval import evaluate_annotated_cohort
from src.pipeline.paths import (
    MANUAL_REPORT_LABELS_PATH,
    MANUAL_VALIDATION_EVAL_DIR,
    PATIENT_VALIDATION_COHORT_PATH,
)

LOGGER = logging.getLogger(__name__)


def main(
    cohort_path: Path = PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Optional[Path] = None,
    output_dir: Path = MANUAL_VALIDATION_EVAL_DIR,
) -> None:
    df = load_cohort_for_manual_evaluation(cohort_path, labels_path=labels_path)
    summary, report = evaluate_annotated_cohort(df, output_dir)
    print(report)
    if not summary.empty:
        print(f"Wrote metrics to {output_dir / 'tables' / 'metrics_summary.csv'}")
    elif labels_path is None and MANUAL_REPORT_LABELS_PATH.exists():
        print(
            "Hint: fill manual_report_ground_truth (0/1) in "
            f"{MANUAL_REPORT_LABELS_PATH} or in the full cohort CSV."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
