"""
Field-level delirium signal analysis vs model prediction and structured baselines.

Merges patient-aggregated Berichte field flags with report_vs_baseline_comparison.csv.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import (
    BERICHTE_INPUT_PATH,
    FIELD_SIGNAL_ANALYSIS_DIR,
    FIELD_SIGNAL_PLOTS_DIR,
    FIELD_SIGNAL_TABLES_DIR,
    REPORT_VS_BASELINE_PATH,
)
from src.preprocessing.berichte_mapper import load_berichte_dataframe

LOGGER = logging.getLogger(__name__)

FIELD_COLUMNS = ("diag", "epikrise", "jetziges_leiden", "prozedere")

DELIR_FIELD_KEYWORDS: Tuple[str, ...] = (
    "delir",
    "delirium",
    "delirant",
    "delirös",
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
    "bewusstseinsstörung",
    "bewusstseinstrübung",
)


def _mpl_cfg() -> Path:
    root = Path(__file__).resolve().parents[2]
    cfg = root / "outputs" / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


def _text_has_delir_hint(text: object) -> bool:
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    low = s.lower()
    return any(kw.lower() in low for kw in DELIR_FIELD_KEYWORDS)


def _odds_ratio_2x2(exposure: pd.Series, outcome: pd.Series) -> Dict[str, object]:
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
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "n": n,
        "haldane_applied": haldane,
        "odds_ratio": or_val,
        "p_hint": (a + b) / n if n else float("nan"),
        "p_outcome": (a + c) / n if n else float("nan"),
    }


def _build_patient_field_hints() -> pd.DataFrame:
    if not BERICHTE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Berichte.csv required for field analysis: {BERICHTE_INPUT_PATH}. "
            "Deploy raw Berichte or run in an environment with data/raw."
        )
    berichte = load_berichte_dataframe(BERICHTE_INPUT_PATH)
    if "PatientID" not in berichte.columns:
        raise ValueError("Berichte.csv must contain PatientID.")
    for col in FIELD_COLUMNS:
        if col not in berichte.columns:
            LOGGER.warning("Berichte missing column %s; treating as empty.", col)
            berichte[col] = ""
    berichte["PatientID"] = berichte["PatientID"].astype(str).str.strip()
    for col in FIELD_COLUMNS:
        berichte[f"_hint_{col}"] = berichte[col].apply(_text_has_delir_hint).astype(int)
    agg_map = {f"field_{c}_delir_hint": (f"_hint_{c}", "max") for c in FIELD_COLUMNS}
    patient = berichte.groupby("PatientID", dropna=False, as_index=False).agg(**agg_map)
    patient = patient.rename(columns={"PatientID": "PatientenID"})
    patient["PatientenID"] = patient["PatientenID"].astype(str).str.strip()
    patient["any_field_delir_hint"] = patient[[f"field_{c}_delir_hint" for c in FIELD_COLUMNS]].max(axis=1)
    return patient


def run_field_signal_analysis(
    cmp_path: Path = REPORT_VS_BASELINE_PATH,
    out_dir: Path = FIELD_SIGNAL_ANALYSIS_DIR,
    tables_dir: Path = FIELD_SIGNAL_TABLES_DIR,
    plots_dir: Path = FIELD_SIGNAL_PLOTS_DIR,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg()))
    for d in (out_dir, tables_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not cmp_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {cmp_path}")

    hints = _build_patient_field_hints()
    cmp_df = pd.read_csv(cmp_path)
    cmp_df["PatientenID"] = cmp_df["PatientenID"].astype(str).str.strip()

    merge_cols = ["PatientenID"] + [
        c
        for c in hints.columns
        if c.startswith("field_") or c == "any_field_delir_hint"
    ]

    merged = cmp_df.merge(hints[merge_cols], on="PatientenID", how="inner")
    if merged.empty:
        raise ValueError("No overlapping PatientenID between comparison file and Berichte aggregation.")

    merged.to_csv(tables_dir / "comparison_with_field_hints.csv", index=False)

    prev_rows = []
    for c in FIELD_COLUMNS:
        hc = f"field_{c}_delir_hint"
        prev_rows.append(
            {
                "field": c,
                "prevalence_hint": float(merged[hc].mean()) if len(merged) else float("nan"),
                "overlap_pred_pos": float(((merged[hc] == 1) & (merged["klasse"] == 1)).mean())
                if "klasse" in merged.columns
                else float("nan"),
                "overlap_icd10": float(((merged[hc] == 1) & (merged["baseline_icd10"] == 1)).mean())
                if "baseline_icd10" in merged.columns
                else float("nan"),
                "overlap_icdsc_ge4": float(((merged[hc] == 1) & (merged["baseline_icdsc_ge_4"] == 1)).mean())
                if "baseline_icdsc_ge_4" in merged.columns
                else float("nan"),
            }
        )
    prev_df = pd.DataFrame(prev_rows)
    prev_df.to_csv(tables_dir / "field_prevalence_and_overlap.csv", index=False)

    outcomes: List[Tuple[str, str]] = []
    if "baseline_icd10" in merged.columns:
        outcomes.append(("baseline_icd10", "ICD10 delirium coded"))
    if "baseline_icdsc_ge_4" in merged.columns:
        outcomes.append(("baseline_icdsc_ge_4", "ICDSC≥4 baseline"))
    if "klasse" in merged.columns:
        outcomes.append(("klasse", "Model pred class 1"))

    or_rows = []
    for field in FIELD_COLUMNS:
        hc = f"field_{field}_delir_hint"
        for out_col, out_label in outcomes:
            oo = merged[out_col]
            stats = _odds_ratio_2x2(merged[hc], oo)
            or_rows.append(
                {
                    "field": field,
                    "outcome": out_col,
                    "outcome_label": out_label,
                    **stats,
                }
            )

    hc_any = merged["any_field_delir_hint"]
    for out_col, out_label in outcomes:
        stats = _odds_ratio_2x2(hc_any, merged[out_col])
        or_rows.append(
            {
                "field": "any_field",
                "outcome": out_col,
                "outcome_label": out_label,
                **stats,
            }
        )

    pd.DataFrame(or_rows).to_csv(tables_dir / "field_odds_ratios_vs_outcomes.csv", index=False)

    fields_lab = list(FIELD_COLUMNS) + ["any"]
    prev_vals = [float(merged[f"field_{f}_delir_hint"].mean()) for f in FIELD_COLUMNS] + [
        float(merged["any_field_delir_hint"].mean())
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(fields_lab, prev_vals, color="#2563eb")
    ax.set_ylabel("Prevalence (patient has hint in field)")
    ax.set_title(
        "Prevalence of delirium-related term hits by Berichte section\n(per patient)"
    )
    ax.set_ylim(0, max(0.08, max(prev_vals) * 1.15) if prev_vals else 1.0)
    for i, v in enumerate(prev_vals):
        ax.text(i, min(v + 0.01, ax.get_ylim()[1]), f"{v:.2%}", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "field_prevalence_bar.png", dpi=120)
    plt.close(fig)

    if outcomes:
        mats = []
        for out_col, _lbl in outcomes:
            row_vals = []
            for f in FIELD_COLUMNS:
                hc = f"field_{f}_delir_hint"
                row_vals.append(float(((merged[hc] == 1) & (merged[out_col] == 1)).sum()))
            mats.append(row_vals)
        mat = np.array(mats, dtype=float)
        fig2, ax2 = plt.subplots(figsize=(7.5, max(5, len(outcomes) * 1.8)))
        im = ax2.imshow(mat, cmap="Greens", aspect="auto")
        ax2.set_xticks(range(len(FIELD_COLUMNS)))
        ax2.set_xticklabels(FIELD_COLUMNS, rotation=22, ha="right", fontsize=9)
        ax2.set_yticks(range(len(outcomes)))
        ax2.set_yticklabels([o[1] for o in outcomes], fontsize=9)
        ax2.set_title("Joint counts: field hint ✓ ∩ outcome ✓")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax2.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=10)
        fig2.colorbar(im, ax=ax2)
        fig2.tight_layout()
        fig2.savefig(plots_dir / "field_outcome_overlap_heatmap.png", dpi=120)
        plt.close(fig2)

        def _stack_counts_for_outcome(out_col: str) -> Tuple[List[int], ...]:
            z00, z01, z10, z11 = [], [], [], []
            for f in FIELD_COLUMNS:
                hc = merged[f"field_{f}_delir_hint"]
                o = pd.to_numeric(merged[out_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
                z00.append(int(((hc == 0) & (o == 0)).sum()))
                z01.append(int(((hc == 0) & (o == 1)).sum()))
                z10.append(int(((hc == 1) & (o == 0)).sum()))
                z11.append(int(((hc == 1) & (o == 1)).sum()))
            return z00, z01, z10, z11

        stacked_specs: List[Tuple[str, str, str]] = []
        if "baseline_icd10" in merged.columns:
            stacked_specs.append(
                ("baseline_icd10", "ICD-10 delirium (structured)", "field_stacked_contingency_icd10.png")
            )
        if "baseline_icdsc_ge_4" in merged.columns:
            stacked_specs.append(
                ("baseline_icdsc_ge_4", "ICDSC≥4 (structured)", "field_stacked_contingency_icdsc_ge4.png")
            )

        for out_col, out_title, png_name in stacked_specs:
            z00, z01, z10, z11 = _stack_counts_for_outcome(out_col)
            fig_s, ax_s = plt.subplots(figsize=(9.0, 4.8))
            idx = np.arange(len(FIELD_COLUMNS))
            lab0 = "No keyword hit, outcome negative"
            lab1 = "No keyword hit, outcome positive"
            lab2 = "Keyword hit, outcome negative"
            lab3 = "Keyword hit, outcome positive"
            ax_s.bar(idx, z00, label=lab0, color="#e2e8f0")
            ax_s.bar(idx, z01, bottom=z00, label=lab1, color="#93c5fd")
            b2 = np.array(z00, dtype=float) + np.array(z01, dtype=float)
            ax_s.bar(idx, z10, bottom=b2, label=lab2, color="#cbd5e1")
            b3 = b2 + np.array(z10, dtype=float)
            ax_s.bar(idx, z11, bottom=b3, label=lab3, color="#059669")
            ax_s.set_xticks(idx)
            ax_s.set_xticklabels(FIELD_COLUMNS, rotation=22, ha="right", fontsize=9)
            ax_s.set_ylabel("Patient count")
            ax_s.set_title(
                f"Stacked contingencies by Berichte field\n(outcome: {out_title})"
            )
            ax_s.legend(loc="upper right", fontsize=7)
            fig_s.tight_layout()
            fig_s.savefig(plots_dir / png_name, dpi=120)
            plt.close(fig_s)

    or_plot = pd.DataFrame(or_rows)
    finite_icd10 = pd.DataFrame()
    if not or_plot.empty and "outcome" in or_plot.columns and "odds_ratio" in or_plot.columns:
        or_plot_icd10 = or_plot[
            or_plot["outcome"] == "baseline_icd10"
        ].dropna(subset=["odds_ratio"])
        finite_icd10 = or_plot_icd10[np.isfinite(or_plot_icd10["odds_ratio"])]
    if not finite_icd10.empty:
        fig3, ax3 = plt.subplots(figsize=(8.0, max(3.8, 0.35 * len(finite_icd10))))
        y = np.arange(len(finite_icd10))
        ax3.barh(y, finite_icd10["odds_ratio"].astype(float).values, color="#059669")
        ax3.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        ax3.set_yticks(y)
        ax3.set_yticklabels(finite_icd10["field"], fontsize=9)
        ax3.set_xlabel("Odds ratio (Haldane +0.5 if needed)")
        ax3.set_title(
            "Field keyword presence vs ICD-10 baseline\nOR>1 ⇒ higher ICD10 coding odds when hinted in section"
        )
        ax3.invert_yaxis()
        fig3.tight_layout()
        fig3.savefig(plots_dir / "field_odds_ratio_vs_icd10.png", dpi=120)
        plt.close(fig3)

    hm_or = (
        or_plot[or_plot["field"].isin(list(FIELD_COLUMNS))].copy()
        if not or_plot.empty and "field" in or_plot.columns
        else pd.DataFrame()
    )
    if not hm_or.empty and outcomes:
        short_names = {
            "baseline_icd10": "ICD-10",
            "baseline_icdsc_ge_4": "ICDSC≥4",
            "klasse": "Model klasse=1",
        }
        cols_order = [oc for oc, _ in outcomes if oc in hm_or["outcome"].values]
        piv = hm_or.pivot_table(index="field", columns="outcome", values="odds_ratio")
        want_cols = [c for c in cols_order if c in piv.columns]
        if not want_cols:
            mat_or = np.array([])
        else:
            piv = piv[want_cols].rename(columns=short_names)
            mat_or = piv.to_numpy(dtype=float)
        if want_cols and mat_or.size:
            fig_h, ax_h = plt.subplots(figsize=(max(6.5, piv.shape[1] * 2.2), max(4.2, piv.shape[0] * 1.0)))
            vmax = min(10.0, np.nanmax(mat_or)) if np.any(np.isfinite(mat_or)) else 1.0
            imh = ax_h.imshow(
                np.nan_to_num(mat_or, nan=0.0, posinf=vmax, neginf=0),
                aspect="auto",
                cmap="RdYlGn",
                vmin=0.5,
                vmax=max(vmax, 1.02),
            )
            ax_h.set_yticks(range(len(piv.index)))
            ax_h.set_yticklabels(list(piv.index), fontsize=9)
            ax_h.set_xticks(range(len(piv.columns)))
            ax_h.set_xticklabels(list(piv.columns), rotation=18, ha="right", fontsize=9)
            ax_h.set_title(
                "Odds ratios: delirium-keyword hit in section vs outcome\n"
                "(greenish = OR>1; grey cell = missing/undefined)"
            )
            for i in range(mat_or.shape[0]):
                for j in range(mat_or.shape[1]):
                    val = mat_or[i, j]
                    txt = f"{val:.2f}" if np.isfinite(val) else "—"
                    ax_h.text(j, i, txt, ha="center", va="center", fontsize=8)
            fig_h.colorbar(imh, ax=ax_h, fraction=0.046)
            fig_h.tight_layout()
            fig_h.savefig(plots_dir / "field_odds_ratio_heatmap.png", dpi=120)
            plt.close(fig_h)

    best_icd10 = (
        finite_icd10.sort_values("odds_ratio", ascending=False)["field"].head(5).tolist()
        if len(finite_icd10)
        else []
    )
    mx = prev_df.dropna(subset=["prevalence_hint"]).sort_values("prevalence_hint", ascending=False)
    strongest_field_name = mx.iloc[0]["field"] if len(mx) else ""

    ranked_icd_overlap = prev_df.dropna(subset=["overlap_icd10"]).sort_values(
        "overlap_icd10", ascending=False
    )
    top_icd_align = ranked_icd_overlap.iloc[0]["field"] if len(ranked_icd_overlap) else ""

    def _prev_for(field_name: str) -> float:
        sub = prev_df.loc[prev_df["field"] == field_name, "prevalence_hint"]
        return float(sub.iloc[0]) if len(sub) else float("nan")

    p_proz = _prev_for("prozedere")
    p_jl = _prev_for("jetziges_leiden")

    report_lines = [
        "Field signal analysis",
        f"n_comparison_rows_joined={len(merged)} (inner join PatientenID)",
        "",
        "Questions answered:",
        (
            f" - Delir-related terms documented in **prozedere**? prevalence={p_proz:.2%} of overlapping patients."
            if np.isfinite(p_proz)
            else " - Delir-related terms documented in **prozedere**? prevalence=n/a (missing field rows)."
        ),
        (
            f" - In **jetziges_leiden**? prevalence={p_jl:.2%}."
            if np.isfinite(p_jl)
            else " - In **jetziges_leiden**? prevalence=n/a."
        ),
        f" - Field with strongest hint prevalence: {strongest_field_name or '(n/a)'}",
        f" - Field with highest ICD10-aligned hint×positive overlap density (mean): {top_icd_align or '(inspect tables)'}",
        f" - Top OR vs ICD10 (field effects): {', '.join(best_icd10) if best_icd10 else '(see odds ratio table — small cells collapse OR)'}",
        "",
        "Notes: ICD/ICDSC baselines are imperfect references; overlap metrics depend on linkage quality.",
        f"tables: {tables_dir}",
        f"plots: {plots_dir}",
    ]
    (out_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Field signal tables: {tables_dir}")
    print(f"Report: {out_dir / 'report.txt'}")


def main() -> None:
    run_field_signal_analysis()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
