"""
Configurable ``baseline_composite`` definition (OR vs AND).

Thesis / sensitive default: OR (ICDSC>=4 OR ICD10).
Temporary presentation mode: AND (ICDSC>=4 AND ICD10) for high-confidence coded delir.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.pipeline import paths as paths_module


BaselineCompositeMode = Literal["OR", "AND"]
VALID_BASELINE_COMPOSITE_MODES: tuple[str, ...] = ("OR", "AND")


def resolve_baseline_composite_mode(mode: str | None = None) -> BaselineCompositeMode:
    """Return normalized mode from argument or ``paths.BASELINE_COMPOSITE_MODE`` config."""
    if mode is not None:
        raw = mode.strip().upper()
    else:
        raw = str(paths_module.BASELINE_COMPOSITE_MODE).strip().upper()
    if raw not in VALID_BASELINE_COMPOSITE_MODES:
        raise ValueError(
            f"Invalid BASELINE_COMPOSITE_MODE={raw!r}. Allowed: {VALID_BASELINE_COMPOSITE_MODES}"
        )
    return raw  # type: ignore[return-value]


def compute_baseline_composite(
    baseline_icdsc_ge_4: pd.Series,
    baseline_icd10: pd.Series,
    *,
    mode: str | None = None,
) -> pd.Series:
    """Binary composite from ICDSC>=4 and ICD10 flags."""
    ge4 = pd.to_numeric(baseline_icdsc_ge_4, errors="coerce").fillna(0).astype(int).clip(0, 1)
    icd10 = pd.to_numeric(baseline_icd10, errors="coerce").fillna(0).astype(int).clip(0, 1)
    resolved = resolve_baseline_composite_mode(mode)
    if resolved == "AND":
        composite = (ge4 == 1) & (icd10 == 1)
    else:
        composite = (ge4 == 1) | (icd10 == 1)
    return composite.astype(int)


def format_baseline_composite_mode_banner() -> str:
    """Console banner for prepare_structured_data / evaluation."""
    mode = resolve_baseline_composite_mode()
    if mode == "AND":
        return (
            "[Baseline Composite Mode]\n"
            "AND (ICDSC >=4 AND ICD10) — high-confidence / secure delir cases (presentation)"
        )
    return "[Baseline Composite Mode]\nOR (ICDSC >=4 OR ICD10) — thesis / sensitive baseline"


def baseline_composite_short_label() -> str:
    mode = resolve_baseline_composite_mode()
    if mode == "AND":
        return "High-confidence delir baseline (AND)"
    return "Composite baseline (OR)"


def baseline_composite_definition_text() -> str:
    mode = resolve_baseline_composite_mode()
    if mode == "AND":
        return "baseline_composite = (baseline_icdsc_ge_4 == 1) AND (baseline_icd10 == 1)"
    return "baseline_composite = (baseline_icdsc_ge_4 == 1) OR (baseline_icd10 == 1)"


def baseline_composite_fp_interpretation_note() -> str:
    """Guidance when model-positive / baseline-negative (not strict false positives in AND mode)."""
    mode = resolve_baseline_composite_mode()
    if mode == "AND":
        return (
            "Model-positive / AND-baseline-negative cases may represent Delirkandidaten "
            "(possible uncoded or underdocumented delir), not automatic false positives."
        )
    return (
        "Model-positive / OR-baseline-negative cases may still warrant clinical review "
        "for underdocumentation."
    )
