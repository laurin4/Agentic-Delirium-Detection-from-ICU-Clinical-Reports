"""
Field-level delirium keyword hints in Berichte.csv vs structured baselines.

Requires external Berichte.csv and outputs/baseline/structured_baseline.csv
(from prepare_structured_data). No sklearn/scipy.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import (
    FIELD_DELIRIUM_ANALYSIS_DIR,
    FIELD_DELIRIUM_PLOTS_DIR,
    FIELD_DELIRIUM_TABLES_DIR,
    STRUCTURED_BASELINE_PATH,
)
from src.preprocessing.berichte_mapper import load_berichte_dataframe

LOGGER = logging.getLogger(__name__)

FIELD_COLUMNS = ["diag", "epikrise", "jetziges_leiden", "prozedere"]

# German / clinical substrings (case-insensitive). Includes ASCII fallbacks for umlauts.
DELIR_KEYWORDS: Tuple[str, ...] = (
    "delir",
    "delirium",
    "delirant",
    "delirös",
    "deliros",
    "hypoaktives delir",
    "hyperaktives delir",
    "verwirrt",
    "verwirrtheit",
    "desorientiert",
    "desorientierung",
    "agitation",
    "agitiert",
    "unruhig",
    "vigilanz",
    "vigilanzminderung",
    "somnolent",
    "soporös",
    "soporos",
    "bewusstseinsstörung",
    "bewusstseinsstorung",
    "bewusstseinstrübung",
    "bewusstseinstrubung",
)


def _mpl_config_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    cfg = root / "outputs" / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


def _text_has_delir_hint(text: object) -> bool:
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    low = s.lower()
    return any(kw.lower() in low for kw in DELIR_KEYWORDS)


def _odds_ratio_2x2(
    exposure: pd.Series, outcome: pd.Series
) -> Dict[str, object]:
    """exposure/outcome: 0/1. Contingency: a=hint+out+, b=hint+out-, c=hint-out+, d=hint-out-."""
    e = pd.to_numeric(exposure, errors="coerce").fillna(0).astype(int).clip(0, 1)
    o = pd.to_numeric(outcome, errors="coerce").fillna(0).astype(int).clip(0, 1)
    a = int(((e == 1) & (o == 1)).sum())
    b = int(((e == 1) & (o == 0)).sum())
    c = int(((e == 0) & (o == 1)).sum())
    d = int(((e == 0) & (o == 0)).sum())
    n = a + b + c + d
    haldane = min(a, b, c, d) == 0
    aa, bb, cc, dd = (a + 0.5, b + 0.5, c + 0.5, d + 0.5) if haldane else (float(a), float(b), float(c), float(d))
    denom = bb * cc
    or_val = (aa * dd) / denom if denom else float("nan")
    pr_hint_given_out_pos = a / (a + c) if (a + c) else float("nan")
    pr_out_given_hint_pos = a / (a + b) if (a + b) else float("nan")
    pr_hint = (a + b) / n if n else float("nan")
    pr_out = (a + c) / n if n else float("nan")
    return {
        "a_hint1_out1": a,
        "b_hint1_out0": b,
        "c_hint0_out1": c,
        "d_hint0_out0": d,
        "n": n,
        "haldane_applied": haldane,
        "odds_ratio": or_val,
        "rate_hint_given_out_positive": pr_hint_given_out_pos,
        "rate_outcome_positive_given_hint_positive": pr_out_given_hint_pos,
        "rate_hint_positive": pr_hint,
        "rate_outcome_positive": pr_out,
    }


def _plot_odds_ratios(rows: List[Dict[str, object]], out_path: Path) -> None:
    labels: List[str] = []
    values: List[float] = []
    for r in rows:
        orv = r.get("odds_ratio")
        if orv is None:
            continue
        try:
            fv = float(orv)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        labels.append(str(r.get("comparison", "")))
        values.append(fv)
    if not labels:
        LOGGER.warning("No finite odds ratios to plot.")
        return
    fig_h = max(3.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#2563eb")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Odds ratio (Haldane +0.5 if any cell is zero)")
    ax.set_title("Field delir-hint vs baseline outcomes")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir()))

    if not STRUCTURED_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Structured baseline missing: {STRUCTURED_BASELINE_PATH}. "
            "Run 'python -m src.pipeline.prepare_structured_data' first."
        )

    berichte = load_berichte_dataframe()
    if "PatientID" not in berichte.columns:
        raise ValueError(f"Berichte.csv must contain 'PatientID'. Found: {list(berichte.columns)}")

    for col in FIELD_COLUMNS:
        if col not in berichte.columns:
            LOGGER.warning("Berichte.csv missing field column '%s'. Filling empty.", col)
            berichte[col] = ""

    berichte["PatientID"] = berichte["PatientID"].astype(str).str.strip()

    for col in FIELD_COLUMNS:
        col_hint = f"_row_hint_{col}"
        berichte[col_hint] = berichte[col].apply(_text_has_delir_hint).astype(int)

    hint_cols = [f"_row_hint_{c}" for c in FIELD_COLUMNS]
    agg_map = {f"field_{c}_delir_hint": (f"_row_hint_{c}", "max") for c in FIELD_COLUMNS}
    patient_hints = berichte.groupby("PatientID", dropna=False, as_index=False).agg(**agg_map)
    patient_hints["any_field_delir_hint"] = patient_hints[[f"field_{c}_delir_hint" for c in FIELD_COLUMNS]].max(
        axis=1
    )

    patient_hints = patient_hints.rename(columns={"PatientID": "PatientenID"})
    patient_hints["PatientenID"] = patient_hints["PatientenID"].astype(str).str.strip()

    baseline = pd.read_csv(STRUCTURED_BASELINE_PATH)
    if "PatientenID" not in baseline.columns:
        raise ValueError("structured_baseline.csv must contain 'PatientenID'.")
    baseline["PatientenID"] = baseline["PatientenID"].astype(str).str.strip()

    need_cols = ["PatientenID", "baseline_icd10", "baseline_icdsc_ge_4"]
    missing = [c for c in need_cols if c not in baseline.columns]
    if missing:
        raise ValueError(
            "structured_baseline.csv is missing required columns for association analysis: "
            + ", ".join(missing)
            + ". Re-run prepare_structured_data with an up-to-date pipeline."
        )

    merged = patient_hints.merge(
        baseline[need_cols],
        on="PatientenID",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping PatientenID between Berichte.csv and structured_baseline.csv.")

    merged["baseline_icd10"] = pd.to_numeric(merged["baseline_icd10"], errors="coerce").fillna(0).astype(int)
    merged["baseline_icdsc_ge_4"] = (
        pd.to_numeric(merged["baseline_icdsc_ge_4"], errors="coerce").fillna(0).astype(int)
    )

    FIELD_DELIRIUM_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DELIRIUM_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DELIRIUM_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    merged.to_csv(FIELD_DELIRIUM_TABLES_DIR / "patient_field_delir_hints_with_baselines.csv", index=False)

    assoc_rows: List[Dict[str, object]] = []
    exposure_specs: List[Tuple[str, str]] = [
        ("field_diag_delir_hint", "diag"),
        ("field_epikrise_delir_hint", "epikrise"),
        ("field_jetziges_leiden_delir_hint", "jetziges_leiden"),
        ("field_prozedere_delir_hint", "prozedere"),
        ("any_field_delir_hint", "any_field"),
    ]
    outcomes: List[Tuple[str, str]] = [
        ("baseline_icd10", "icd10_delir"),
        ("baseline_icdsc_ge_4", "icdsc_ge_4"),
    ]

    for exp_col, exp_label in exposure_specs:
        for out_col, out_label in outcomes:
            stats = _odds_ratio_2x2(merged[exp_col], merged[out_col])
            assoc_rows.append(
                {
                    "field": exp_label,
                    "outcome": out_label,
                    "comparison": f"{exp_label}_vs_{out_label}",
                    **stats,
                }
            )

    assoc_df = pd.DataFrame(assoc_rows)
    assoc_df.to_csv(FIELD_DELIRIUM_TABLES_DIR / "association_odds_ratio.csv", index=False)

    contig_rows: List[Dict[str, object]] = []
    for r in assoc_rows:
        contig_rows.append(
            {
                "field": r["field"],
                "outcome": r["outcome"],
                "a_hint1_out1": r["a_hint1_out1"],
                "b_hint1_out0": r["b_hint1_out0"],
                "c_hint0_out1": r["c_hint0_out1"],
                "d_hint0_out0": r["d_hint0_out0"],
                "n": r["n"],
                "haldane_applied": r["haldane_applied"],
            }
        )
    pd.DataFrame(contig_rows).to_csv(FIELD_DELIRIUM_TABLES_DIR / "contingency_counts.csv", index=False)

    _plot_odds_ratios(assoc_rows, FIELD_DELIRIUM_PLOTS_DIR / "odds_ratio_by_field.png")

    report_path = FIELD_DELIRIUM_ANALYSIS_DIR / "report.txt"
    lines = [
        "Field-level delirium keyword analysis",
        "",
        f"n_patients_merged: {len(merged)}",
        f"tables: {FIELD_DELIRIUM_TABLES_DIR}",
        f"plots: {FIELD_DELIRIUM_PLOTS_DIR}",
        "",
        "Odds ratio (Haldane +0.5 to all cells if any count is zero).",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {FIELD_DELIRIUM_TABLES_DIR / 'patient_field_delir_hints_with_baselines.csv'}")
    print(f"Wrote: {FIELD_DELIRIUM_TABLES_DIR / 'association_odds_ratio.csv'}")
    print(f"Wrote: {FIELD_DELIRIUM_PLOTS_DIR / 'odds_ratio_by_field.png'}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
