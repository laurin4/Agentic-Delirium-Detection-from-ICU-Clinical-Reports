"""
Run post-pipeline scientific analysis scripts in sequence.

Requires `outputs/comparisons/report_vs_baseline_comparison.csv` and (for field + evidence)
Berichte CSV at paths configured in src.pipeline.paths.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.analysis.run_error_review_export import main as error_review_main
from src.analysis.run_evidence_snippets_export import main as evidence_main
from src.analysis.run_field_signal_analysis import main as field_signal_main
from src.analysis.run_keyword_analysis import main as keyword_main
from src.pipeline.paths import (
    ERROR_REVIEW_TABLES_DIR,
    FIELD_SIGNAL_ANALYSIS_DIR,
    KEYWORD_ANALYSIS_DIR,
    EVIDENCE_SNIPPETS_TABLES_DIR,
    MANUAL_REVIEW_DIR,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    paths_to_print = [
        ("manual_review", MANUAL_REVIEW_DIR),
        ("keyword_analysis_root", KEYWORD_ANALYSIS_DIR),
        ("field_signal_root", FIELD_SIGNAL_ANALYSIS_DIR),
        ("evidence_tables", EVIDENCE_SNIPPETS_TABLES_DIR),
        ("legacy_error_review_tables", ERROR_REVIEW_TABLES_DIR),
    ]

    LOGGER.info("1/4 Manual review export (TP/TN/FP/FN samples)...")
    error_review_main()

    LOGGER.info("2/4 Keyword association analysis...")
    keyword_main()

    LOGGER.info("3/4 Field signal analysis...")
    field_signal_main()

    LOGGER.info("4/4 Evidence snippets export...")
    evidence_main()

    summary_path = MANUAL_REVIEW_DIR / "manual_review_summary.csv"
    if summary_path.exists():
        try:
            sdf = pd.read_csv(summary_path)
            if len(sdf) and "baseline_name" in sdf.columns:
                row = sdf.iloc[0]
                print("")
                print("Manual review export (first baseline row in summary):")
                print(
                    f"  baseline={row.get('baseline_name')!r}, "
                    f"evaluable n={int(row.get('total_evaluable_rows', 0))}, "
                    f"TP={int(row.get('count_TP', 0))}, TN={int(row.get('count_TN', 0))}, "
                    f"FP={int(row.get('count_FP', 0))}, FN={int(row.get('count_FN', 0))}"
                )
        except Exception as exc:
            LOGGER.warning("Could not print manual review summary teaser: %s", exc)

    print("")
    print("Full analysis suite — output directories:")
    for label, root in paths_to_print:
        print(f"  {label}: {root}")
    hint_path = FIELD_SIGNAL_ANALYSIS_DIR / "report.txt"
    if hint_path.exists():
        print("")
        print("Field signal summary excerpt (full file on disk):")
        try:
            lines = hint_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[:12]:
                print(f"  {line}")
            if len(lines) > 12:
                print("  ...")
        except OSError:
            LOGGER.warning("Could not read field signal report")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
