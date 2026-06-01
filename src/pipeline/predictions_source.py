"""
Resolve which predictions CSV to use for baseline comparison and evaluation.

Environment:
  PREDICTIONS_SOURCE=full|validation_cohort  (default: full)

  full               -> outputs/predictions/agent1_agent2_agent3_results_prompt.csv
  validation_cohort  -> outputs/predictions/validation_cohort_predictions.csv
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from src.pipeline.paths import FULL_PREDICTIONS_PATH, VALIDATION_COHORT_PREDICTIONS_PATH
from src.pipeline.prompt_run_paths import (
    get_versioned_predictions_path,
    is_versioned_validation_run,
)

LOGGER = logging.getLogger(__name__)

PREDICTIONS_SOURCE_ENV = "PREDICTIONS_SOURCE"
PREDICTIONS_SOURCE_FULL = "full"
PREDICTIONS_SOURCE_VALIDATION_COHORT = "validation_cohort"
ALLOWED_PREDICTIONS_SOURCES = frozenset(
    {PREDICTIONS_SOURCE_FULL, PREDICTIONS_SOURCE_VALIDATION_COHORT}
)


def get_predictions_source() -> str:
    """Return normalized ``PREDICTIONS_SOURCE`` env value (default ``full``)."""
    raw = os.environ.get(PREDICTIONS_SOURCE_ENV, PREDICTIONS_SOURCE_FULL).strip().lower()
    if raw not in ALLOWED_PREDICTIONS_SOURCES:
        allowed = ", ".join(sorted(ALLOWED_PREDICTIONS_SOURCES))
        raise ValueError(
            f"Invalid {PREDICTIONS_SOURCE_ENV}='{raw}'. Allowed values: {allowed}"
        )
    return raw


def resolve_predictions_path(
    predictions_path: Optional[Path] = None,
    *,
    source: Optional[str] = None,
) -> Path:
    """
    Choose predictions CSV from explicit path, *source*, or ``PREDICTIONS_SOURCE`` env.
    """
    if predictions_path is not None:
        return Path(predictions_path)
    src = source if source is not None else get_predictions_source()
    if src == PREDICTIONS_SOURCE_VALIDATION_COHORT:
        if is_versioned_validation_run():
            return get_versioned_predictions_path()
        return VALIDATION_COHORT_PREDICTIONS_PATH
    return FULL_PREDICTIONS_PATH


def log_predictions_source(
    predictions_path: Path,
    *,
    source: Optional[str] = None,
    explicit_path: bool = False,
) -> None:
    """Log and print which prediction file is used."""
    if explicit_path:
        msg = f"Prediction path (explicit override): {predictions_path}"
        LOGGER.info("%s", msg)
        print(msg)
        return
    src = source if source is not None else get_predictions_source()
    msg = f"PREDICTIONS_SOURCE={src} predictions_path={predictions_path}"
    LOGGER.info("%s", msg)
    print(msg)
