"""
Safe launcher for full-corpus delirium inference (not frozen validation cohort).

Forces full Berichte mode, preflight checks, backups, markers, flushed progress,
and optional smoke runs without touching the main predictions CSV.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models.model_config import LLM_MODEL_LABEL, LLM_PROVIDER
from src.pipeline.paths import BERICHTE_INPUT_PATH, FULL_PREDICTIONS_PATH, PREDICTIONS_DIR
from src.pipeline.prompt_selector import get_prompt_version_from_env
from src.pipeline.run_pipeline import (
    RUN_PIPELINE_MAX_REPORTS_OVERRIDE_ENV,
    RUN_PIPELINE_OUTPUT_PATH_ENV,
    RUN_PIPELINE_PROGRESS_FLUSH_ENV,
    RUN_PIPELINE_SKIP_MODEL_COPY_ENV,
    main as run_pipeline_main,
)
from src.pipeline.validation_cohort_filter import validation_cohort_only_enabled
from src.preprocessing.berichte_mapper import build_report_level_berichte_records

DEFAULT_CHECKPOINT_EVERY = 50
SMOKE_OUTPUT_PATH = PREDICTIONS_DIR / "full_corpus_smoke_5.csv"
SMOKE_N = 5

MARKER_RUNNING_SUFFIX = ".running"
MARKER_COMPLETED_SUFFIX = ".completed"
MARKER_FAILED_SUFFIX = ".failed"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _marker_path(output_csv: Path, suffix: str) -> Path:
    return output_csv.with_name(output_csv.name + suffix)


def _max_reports_env_active() -> bool:
    raw = os.environ.get("MAX_REPORTS", "").strip()
    if not raw:
        return False
    return raw.lower() != "all"


def _count_full_corpus_records() -> tuple[int, int]:
    records, excluded_db = build_report_level_berichte_records()
    return len(records), excluded_db


def _clear_cohort_only_env() -> None:
    os.environ.pop("VALIDATION_COHORT_ONLY", None)


def _backup_main_predictions(output_csv: Path) -> Optional[Path]:
    if not output_csv.exists():
        return None
    backup = output_csv.with_name(
        f"{output_csv.stem}.backup_{_utc_stamp()}{output_csv.suffix}"
    )
    shutil.copy2(output_csv, backup)
    return backup


def _write_marker(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _remove_marker(path: Path) -> None:
    if path.exists():
        path.unlink()


def _print_launch_plan(
    *,
    output_csv: Path,
    expected_records: int,
    excluded_db: int,
    checkpoint_every: int,
    smoke: bool,
    backup_path: Optional[Path],
) -> None:
    print("=== Full-corpus safe launcher ===")
    print(f"mode={'smoke' if smoke else 'full'}")
    print(f"berichte_path={BERICHTE_INPUT_PATH.resolve()}")
    print(f"expected_records={expected_records}")
    print(f"excluded_dokumentationsblatt={excluded_db}")
    print(f"output_csv={output_csv.resolve()}")
    print(f"prompt_version={get_prompt_version_from_env()}")
    print(f"llm_provider={LLM_PROVIDER}")
    print(f"llm_model_label={LLM_MODEL_LABEL}")
    print(f"checkpoint_every={checkpoint_every}")
    print(f"VALIDATION_COHORT_ONLY={os.environ.get('VALIDATION_COHORT_ONLY', '<unset>')}")
    print(f"MAX_REPORTS={os.environ.get('MAX_REPORTS', '<unset>')}")
    if backup_path:
        print(f"backup_created={backup_path.resolve()}")
    print(f"running_marker={_marker_path(output_csv, MARKER_RUNNING_SUFFIX).resolve()}")
    print("progress_flush=true (RUN_PIPELINE_PROGRESS_FLUSH)")
    print("")


def _validate_preflight(*, allow_max_reports: bool, smoke: bool) -> None:
    if validation_cohort_only_enabled():
        raise SystemExit(
            "Refusing to run: VALIDATION_COHORT_ONLY is enabled. "
            "Unset it for full-corpus inference."
        )

    if _max_reports_env_active() and not allow_max_reports and not smoke:
        raise SystemExit(
            "Refusing to run: MAX_REPORTS is set in the environment. "
            "Unset MAX_REPORTS or pass --allow-max-reports explicitly."
        )

    if not BERICHTE_INPUT_PATH.exists():
        raise SystemExit(f"Missing Berichte input: {BERICHTE_INPUT_PATH}")


def _configure_pipeline_env(
    *,
    output_csv: Path,
    checkpoint_every: int,
    record_limit: Optional[int],
    smoke: bool,
) -> dict[str, Optional[str]]:
    """Set launcher env overrides; returns previous values for restoration."""
    prev = {
        "VALIDATION_COHORT_ONLY": os.environ.pop("VALIDATION_COHORT_ONLY", None),
        RUN_PIPELINE_PROGRESS_FLUSH_ENV: os.environ.get(RUN_PIPELINE_PROGRESS_FLUSH_ENV),
        "PIPELINE_CHECKPOINT_EVERY": os.environ.get("PIPELINE_CHECKPOINT_EVERY"),
        RUN_PIPELINE_OUTPUT_PATH_ENV: os.environ.get(RUN_PIPELINE_OUTPUT_PATH_ENV),
        RUN_PIPELINE_MAX_REPORTS_OVERRIDE_ENV: os.environ.get(RUN_PIPELINE_MAX_REPORTS_OVERRIDE_ENV),
        RUN_PIPELINE_SKIP_MODEL_COPY_ENV: os.environ.get(RUN_PIPELINE_SKIP_MODEL_COPY_ENV),
    }
    os.environ[RUN_PIPELINE_PROGRESS_FLUSH_ENV] = "true"
    os.environ["PIPELINE_CHECKPOINT_EVERY"] = str(checkpoint_every)
    os.environ[RUN_PIPELINE_OUTPUT_PATH_ENV] = str(output_csv.resolve())

    if record_limit is not None:
        os.environ[RUN_PIPELINE_MAX_REPORTS_OVERRIDE_ENV] = str(record_limit)
    else:
        os.environ.pop(RUN_PIPELINE_MAX_REPORTS_OVERRIDE_ENV, None)

    if smoke:
        os.environ[RUN_PIPELINE_SKIP_MODEL_COPY_ENV] = "true"
    else:
        os.environ.pop(RUN_PIPELINE_SKIP_MODEL_COPY_ENV, None)
    return prev


def _restore_pipeline_env(prev: dict[str, Optional[str]]) -> None:
    for key, value in prev.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_safe(
    *,
    smoke: bool = False,
    allow_max_reports: bool = False,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    output_csv: Optional[Path] = None,
    record_limit: Optional[int] = None,
) -> int:
    _validate_preflight(allow_max_reports=allow_max_reports, smoke=smoke)

    if smoke:
        output_csv = SMOKE_OUTPUT_PATH
        record_limit = SMOKE_N
    else:
        output_csv = output_csv or FULL_PREDICTIONS_PATH
        record_limit = None

    expected_records, excluded_db = _count_full_corpus_records()
    if record_limit is not None:
        expected_records = min(expected_records, record_limit)

    backup_path: Optional[Path] = None
    if not smoke:
        backup_path = _backup_main_predictions(output_csv)

    running_marker = _marker_path(output_csv, MARKER_RUNNING_SUFFIX)
    completed_marker = _marker_path(output_csv, MARKER_COMPLETED_SUFFIX)
    failed_marker = _marker_path(output_csv, MARKER_FAILED_SUFFIX)

    _remove_marker(completed_marker)
    _remove_marker(failed_marker)

    _print_launch_plan(
        output_csv=output_csv,
        expected_records=expected_records,
        excluded_db=excluded_db,
        checkpoint_every=checkpoint_every,
        smoke=smoke,
        backup_path=backup_path,
    )

    prev_env = _configure_pipeline_env(
        output_csv=output_csv,
        checkpoint_every=checkpoint_every,
        record_limit=record_limit,
        smoke=smoke,
    )

    _write_marker(
        running_marker,
        [
            f"started_utc={_utc_stamp()}",
            f"output_csv={output_csv.resolve()}",
            f"expected_records={expected_records}",
            f"smoke={smoke}",
        ],
    )

    checkpoint_path = output_csv.with_name(f"{output_csv.stem}.checkpoint.csv")

    try:
        print("PIPELINE_LAUNCH: calling src.pipeline.run_pipeline.main()", flush=True)
        run_pipeline_main()
    except Exception as exc:
        _remove_marker(running_marker)
        _write_marker(
            failed_marker,
            [
                f"failed_utc={_utc_stamp()}",
                f"output_csv={output_csv.resolve()}",
                f"error={type(exc).__name__}: {exc}",
                f"checkpoint_kept={checkpoint_path.resolve() if checkpoint_path.exists() else 'none'}",
                "",
                traceback.format_exc(),
            ],
        )
        print(f"RUN_FAILED marker={failed_marker.resolve()}", flush=True)
        if checkpoint_path.exists():
            print(f"CHECKPOINT_KEPT path={checkpoint_path.resolve()}", flush=True)
        raise
    else:
        _remove_marker(running_marker)
        _write_marker(
            completed_marker,
            [
                f"completed_utc={_utc_stamp()}",
                f"output_csv={output_csv.resolve()}",
                f"rows_expected={expected_records}",
                f"smoke={smoke}",
            ],
        )
        print(f"RUN_COMPLETED marker={completed_marker.resolve()}", flush=True)
        print(f"OUTPUT_CSV={output_csv.resolve()}", flush=True)
        return 0
    finally:
        _restore_pipeline_env(prev_env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safe full-corpus delirium pipeline launcher (not validation cohort)."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Process exactly {SMOKE_N} reports; write {SMOKE_OUTPUT_PATH.name} only.",
    )
    parser.add_argument(
        "--max-reports",
        type=int,
        default=None,
        metavar="N",
        help=f"Smoke alias: only N={SMOKE_N} is supported (same as --smoke).",
    )
    parser.add_argument(
        "--allow-max-reports",
        action="store_true",
        help="Allow MAX_REPORTS env cap (not recommended for thesis full run).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=f"Write checkpoint CSV every N reports (default: {DEFAULT_CHECKPOINT_EVERY}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_reports is not None and args.smoke:
        parser.error("Use either --smoke or --max-reports 5, not both.")
    if args.max_reports is not None and args.max_reports != SMOKE_N:
        parser.error(f"Only --max-reports {SMOKE_N} is supported (use --smoke).")

    smoke = bool(args.smoke or args.max_reports == SMOKE_N)

    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be a positive integer.")

    # Ensure unbuffered stdout when launched directly.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    return run_safe(
        smoke=smoke,
        allow_max_reports=args.allow_max_reports,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as exc:
        raise
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
