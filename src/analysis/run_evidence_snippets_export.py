"""
Attach interpretability column `evidence_snippets` to the report vs baseline merge.

Reads comparison CSV and patient-level report_text from Berichte (when available).
Does not change model predictions or merge logic used for scoring.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.evidence_snippets import extract_evidence_snippets
from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    EVIDENCE_SNIPPETS_DIR,
    EVIDENCE_SNIPPETS_TABLES_DIR,
    REPORT_VS_BASELINE_PATH,
)
from src.preprocessing.berichte_mapper import build_patient_level_berichte_reports

LOGGER = logging.getLogger(__name__)

OUTPUT_NAME = "comparison_with_evidence_snippets.csv"


def run_evidence_snippets_export(
    cmp_path: Path = REPORT_VS_BASELINE_PATH,
    out_tables: Path = EVIDENCE_SNIPPETS_TABLES_DIR,
) -> Path:
    out_tables.mkdir(parents=True, exist_ok=True)
    if not cmp_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {cmp_path}")

    df = pd.read_csv(cmp_path)
    df["PatientenID"] = df["PatientenID"].astype(str).str.strip()

    try:
        if not BERICHTE_INPUT_PATH.exists():
            raise FileNotFoundError(str(BERICHTE_INPUT_PATH))
        reports = build_patient_level_berichte_reports()
    except FileNotFoundError as exc:
        LOGGER.warning(
            "Berichte not available (%s). evidence_snippets will be empty.",
            exc,
        )
        reports = pd.DataFrame(columns=["PatientenID", "bericht", "report_text"])

    if not reports.empty:
        reports = reports.copy()
        reports["PatientenID"] = reports["PatientenID"].astype(str).str.strip()
        merged = df.merge(
            reports[["PatientenID", "report_text"]],
            on="PatientenID",
            how="left",
        )
    else:
        merged = df.copy()
        merged["report_text"] = ""

    merged["evidence_snippets"] = merged["report_text"].fillna("").map(
        lambda t: extract_evidence_snippets(t if isinstance(t, str) else str(t))
    )

    out_path = out_tables / OUTPUT_NAME
    merged.to_csv(out_path, index=False)
    LOGGER.info("Wrote %s rows → %s", len(merged), out_path)
    return out_path


def main() -> None:
    p = run_evidence_snippets_export()
    print(f"Evidence export: {p}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
