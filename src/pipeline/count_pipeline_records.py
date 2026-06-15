"""
Dry-run: how many reports full-corpus inference would process (no LLM).

Mirrors run_pipeline._get_report_records() for INPUT_MODE=berichte without cohort-only.
"""

from __future__ import annotations

import os
import sys

from src.pipeline.paths import BERICHTE_INPUT_PATH, MAX_REPORTS, parse_max_reports_env
from src.pipeline.validation_cohort_filter import validation_cohort_only_enabled
from src.preprocessing.berichte_mapper import build_report_level_berichte_records


def main() -> None:
    if validation_cohort_only_enabled():
        print(
            "ERROR: VALIDATION_COHORT_ONLY is set. Unset it for full-corpus count.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not BERICHTE_INPUT_PATH.exists():
        print(f"ERROR: Missing {BERICHTE_INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    records, excluded_db = build_report_level_berichte_records()
    total_before_cap = len(records)
    cap = MAX_REPORTS
    if cap is not None:
        records = records[:cap]

    pids = {str(r.get("PatientenID", "")) for r in records if r.get("PatientenID")}
    print("=== Full-corpus pipeline record count (dry run) ===")
    print(f"berichte_path={BERICHTE_INPUT_PATH.resolve()}")
    print(f"excluded_dokumentationsblatt={excluded_db}")
    print(f"records_with_text_blocks={total_before_cap}")
    print(f"MAX_REPORTS={cap!r} (from env at import; unset for full corpus)")
    print(f"records_after_max_reports_cap={len(records)}")
    print(f"unique_patients={len(pids)}")
    print(f"env_MAX_REPORTS_raw={os.environ.get('MAX_REPORTS', '')!r}")
    if cap is not None:
        print(
            "\nWARNING: MAX_REPORTS is active — full run would process only "
            f"{len(records)} reports, not the full corpus.",
            file=sys.stderr,
        )
        sys.exit(2)
    print("\nOK: no MAX_REPORTS cap; run_pipeline would process all listed records.")


if __name__ == "__main__":
    main()
