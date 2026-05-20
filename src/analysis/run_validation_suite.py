"""
Run comparison, evaluation, and manual error-review export in one command.

Prerequisites (typical order):
  python -m src.pipeline.prepare_structured_data
  python -m src.pipeline.run_pipeline

This module does not run those steps automatically.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_module(mod: str) -> None:
    cmd = [sys.executable, "-m", mod]
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)


def main() -> None:
    _run_module("src.pipeline.compare_reports_vs_baseline")
    _run_module("src.pipeline.evaluate_predictions")
    _run_module("src.analysis.create_patient_reporttype_matrix")
    _run_module("src.analysis.plot_patient_reporttype_matrix")
    _run_module("src.analysis.export_patient_validation_cohort")
    # LEGACY: multiclass manual_label error review (optional)
    # _run_module("src.analysis.run_error_review_export")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
