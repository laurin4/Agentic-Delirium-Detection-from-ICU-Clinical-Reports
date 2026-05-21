"""
Freeze the manual validation cohort for fixed manual annotation / evaluation.

Copies patient_validation_cohort.csv and manual_report_labels.csv into
outputs/analysis/manual_validation/frozen_validation_cohort/ with checksum metadata.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from src.analysis.frozen_validation_cohort import (
    assert_can_write_frozen,
    build_frozen_metadata,
    frozen_cohort_paths,
    overwrite_frozen_validation_enabled,
)
from src.pipeline.paths import (
    FROZEN_VALIDATION_COHORT_DIR,
    MANUAL_REPORT_LABELS_PATH,
    PATIENT_VALIDATION_COHORT_PATH,
    PREDICTIONS_DIR,
    STRUCTURED_BASELINE_PATH,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"


def freeze_validation_cohort(
    cohort_path: Path = PATIENT_VALIDATION_COHORT_PATH,
    labels_path: Path = MANUAL_REPORT_LABELS_PATH,
    *,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    output_dir: Path = FROZEN_VALIDATION_COHORT_DIR,
    force_overwrite: bool = False,
) -> Path:
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort missing: {cohort_path}. "
            "Run python -m src.analysis.export_patient_validation_cohort first."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels missing: {labels_path}. "
            "Run python -m src.analysis.export_manual_report_labels first."
        )

    assert_can_write_frozen(force=force_overwrite or overwrite_frozen_validation_enabled())

    frozen_cohort, frozen_labels, metadata_path = frozen_cohort_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cohort_path, frozen_cohort)
    shutil.copy2(labels_path, frozen_labels)

    meta = build_frozen_metadata(
        cohort_path,
        labels_path,
        frozen_cohort_out=frozen_cohort,
        frozen_labels_out=frozen_labels,
        predictions_source=predictions_path,
        baseline_source=baseline_path,
    )
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    LOGGER.info(
        "Frozen validation cohort: patients=%s reports=%s",
        meta["n_unique_patients"],
        meta["n_report_rows"],
    )
    return metadata_path


def main() -> None:
    meta_path = freeze_validation_cohort()
    cohort_path, labels_path, _ = frozen_cohort_paths()
    print(f"Wrote frozen cohort: {cohort_path}")
    print(f"Wrote frozen labels: {labels_path}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
