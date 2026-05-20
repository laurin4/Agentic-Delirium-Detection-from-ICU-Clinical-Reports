"""
Evaluate annotated patient validation cohort (report- and patient-level metrics).

Input: outputs/analysis/manual_validation/patient_validation_cohort.csv (after annotation)
Output: outputs/analysis/manual_validation/evaluation/
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.manual_validation_eval import evaluate_annotated_cohort
from src.pipeline.paths import (
    MANUAL_VALIDATION_EVAL_DIR,
    PATIENT_VALIDATION_COHORT_PATH,
)

LOGGER = logging.getLogger(__name__)


def main(
    cohort_path: Path = PATIENT_VALIDATION_COHORT_PATH,
    output_dir: Path = MANUAL_VALIDATION_EVAL_DIR,
) -> None:
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Annotated cohort missing: {cohort_path}. "
            "Run python -m src.analysis.export_patient_validation_cohort and annotate first."
        )
    df = pd.read_csv(cohort_path)
    summary, report = evaluate_annotated_cohort(df, output_dir)
    print(report)
    if not summary.empty:
        print(f"Wrote metrics to {output_dir / 'tables' / 'metrics_summary.csv'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
