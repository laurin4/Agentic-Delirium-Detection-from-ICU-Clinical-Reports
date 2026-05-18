"""
Build patient-level report-type matrix from report-level predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.patient_reporttype_matrix import build_patient_reporttype_matrix
from src.analysis.plot_patient_reporttype_matrix import plot_patient_reporttype_matrix_preview
from src.pipeline.paths import (
    PATIENT_LEVEL_ANALYSIS_DIR,
    PATIENT_REPORTTYPE_MATRIX_PATH,
    PATIENT_REPORTTYPE_MATRIX_PREVIEW_PDF,
    PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG,
    PREDICTIONS_DIR,
    STRUCTURED_BASELINE_PATH,
)
from src.analysis.cohort_counts import load_structured_baseline_rows
from src.pipeline.schema_normalize import normalize_patient_id_column

LOGGER = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_PATH = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"


def main(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    baseline_path: Path = STRUCTURED_BASELINE_PATH,
    output_path: Path = PATIENT_REPORTTYPE_MATRIX_PATH,
) -> None:
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions missing: {predictions_path}. Run python -m src.pipeline.run_pipeline first."
        )
    preds = normalize_patient_id_column(pd.read_csv(predictions_path))
    baseline = load_structured_baseline_rows(baseline_path)
    matrix = build_patient_reporttype_matrix(preds, baseline)

    PATIENT_LEVEL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, index=False)

    n_pat = len(matrix)
    n_disc = int(matrix["discrepancy_model_vs_baseline"].sum()) if "discrepancy_model_vs_baseline" in matrix.columns else 0
    preview_png = plot_patient_reporttype_matrix_preview(
        matrix,
        png_path=PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG,
        pdf_path=PATIENT_REPORTTYPE_MATRIX_PREVIEW_PDF,
    )

    print(f"Wrote patient report-type matrix: {output_path}")
    print(f"Wrote matrix preview plot: {preview_png}")
    print(f"patient_count={n_pat} discrepancy_model_vs_baseline={n_disc}")
    print("Dokumentationsblatt rows excluded from aggregation (not in predictions).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
