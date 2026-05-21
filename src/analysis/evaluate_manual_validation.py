"""
Evaluate annotated patient validation cohort (report- and patient-level metrics).

Input (either):
  - Frozen: frozen_validation_cohort/*_frozen.csv (preferred when present)
  - Annotated patient_validation_cohort.csv, or
  - patient_validation_cohort.csv + manual_report_labels.csv (merged by validation_report_id)

Output: outputs/analysis/manual_validation/evaluation/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.analysis.frozen_validation_cohort import resolve_validation_input_paths
from src.analysis.manual_report_labels import load_cohort_for_manual_evaluation
from src.analysis.manual_validation_eval import evaluate_annotated_cohort
from src.pipeline.paths import MANUAL_VALIDATION_EVAL_DIR

LOGGER = logging.getLogger(__name__)


def main(
    cohort_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    output_dir: Path = MANUAL_VALIDATION_EVAL_DIR,
) -> None:
    using_frozen = False
    if cohort_path is None:
        resolved_cohort, resolved_labels, using_frozen = resolve_validation_input_paths(
            None, labels_path
        )
    else:
        resolved_cohort = cohort_path
        resolved_labels = labels_path

    df = load_cohort_for_manual_evaluation(
        resolved_cohort,
        labels_path=resolved_labels,
        prefer_frozen=False,
    )
    summary, report = evaluate_annotated_cohort(df, output_dir)
    if using_frozen:
        print("Evaluation source: frozen validation cohort")
    print(report)
    if not summary.empty:
        print(f"Wrote metrics to {output_dir / 'tables' / 'metrics_summary.csv'}")
    elif resolved_labels is not None and resolved_labels.exists():
        print(
            "Hint: fill manual_report_ground_truth (0/1) in "
            f"{resolved_labels} (or annotate the full cohort CSV)."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
