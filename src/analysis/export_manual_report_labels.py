"""
Export slim manual annotation sheet from patient_validation_cohort.csv.

Output: outputs/analysis/manual_validation/manual_report_labels.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.manual_report_labels import build_manual_report_labels_sheet
from src.pipeline.paths import (
    MANUAL_REPORT_LABELS_PATH,
    MANUAL_VALIDATION_DIR,
    PATIENT_VALIDATION_COHORT_PATH,
)

LOGGER = logging.getLogger(__name__)


def main(
    cohort_path: Path = PATIENT_VALIDATION_COHORT_PATH,
    output_path: Path = MANUAL_REPORT_LABELS_PATH,
) -> None:
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort missing: {cohort_path}. "
            "Run python -m src.analysis.export_patient_validation_cohort first."
        )
    cohort = pd.read_csv(cohort_path)
    sheet = build_manual_report_labels_sheet(cohort)
    MANUAL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(output_path, index=False)
    print(f"Wrote manual report labels: {output_path}")
    print(f"report_rows={len(sheet)} unique_reports={sheet['validation_report_id'].nunique()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
