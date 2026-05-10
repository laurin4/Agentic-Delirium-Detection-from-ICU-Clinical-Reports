"""
Keyword frequency and association with predictions and structured baselines.

Reads: outputs/comparisons/report_vs_baseline_comparison.csv
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline.paths import (
    KEYWORD_ANALYSIS_DIR,
    KEYWORD_ANALYSIS_PLOTS_DIR,
    KEYWORD_ANALYSIS_TABLES_DIR,
    REPORT_VS_BASELINE_PATH,
)

LOGGER = logging.getLogger(__name__)

ANALYSIS_KEYWORDS: Tuple[str, ...] = (
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

TEXT_SOURCE_COLUMNS = (
    "delir_signale",
    "kontext",
    "begruendung",
    "klassifikation_begruendung",
    "alternative_erklaerung_keywords",
)


def _mpl_cfg() -> Path:
    root = Path(__file__).resolve().parents[2]
    cfg = root / "outputs" / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


def _row_text_blob(row: pd.Series) -> str:
    parts: List[str] = []
    for c in TEXT_SOURCE_COLUMNS:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            parts.append(s)
    return " ".join(parts).lower()


def count_keyword_occurrences(text_lower: str, kw: str) -> int:
    if not text_lower or not kw:
        return 0
    k = kw.lower()
    n = 0
    start = 0
    while True:
        i = text_lower.find(k, start)
        if i < 0:
            break
        n += 1
        start = i + max(1, len(k))
    return n


def row_has_keyword(text_lower: str, kw: str) -> bool:
    return kw.lower() in text_lower if text_lower else False


def _export_top_stratum_keywords(strat_rows: List[Dict[str, object]], tables_dir: Path) -> None:
    sdf = pd.DataFrame(strat_rows)
    if sdf.empty:
        return
    for strat in ["false_positive", "false_negative", "true_positive"]:
        sub = sdf[sdf["stratum_icd10"] == strat].copy()
        sub = sub.sort_values("fraction_with_keyword", ascending=False)
        sub.head(15).to_csv(tables_dir / f"top_keywords_{strat}.csv", index=False)


def run_keyword_analysis(
    cmp_path: Path = REPORT_VS_BASELINE_PATH,
    out_dir: Path = KEYWORD_ANALYSIS_DIR,
    tables_dir: Path = KEYWORD_ANALYSIS_TABLES_DIR,
    plots_dir: Path = KEYWORD_ANALYSIS_PLOTS_DIR,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg()))
    for d in (out_dir, tables_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not cmp_path.exists():
        raise FileNotFoundError(f"Missing {cmp_path}")

    df = pd.read_csv(cmp_path).copy()
    df["PatientenID"] = df["PatientenID"].astype(str).str.strip()

    klasse = pd.to_numeric(df.get("klasse"), errors="coerce").fillna(0).astype(int).clip(0, 1)
    baseline_icd10 = (
        pd.to_numeric(df["baseline_icd10"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        if "baseline_icd10" in df.columns
        else pd.Series(0, index=df.index)
    )
    baseline_ge4 = (
        pd.to_numeric(df["baseline_icdsc_ge_4"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        if "baseline_icdsc_ge_4" in df.columns
        else pd.Series(0, index=df.index)
    )

    blobs = df.apply(_row_text_blob, axis=1)

    overall_keyword_rows: List[Dict[str, object]] = []
    for kw in ANALYSIS_KEYWORDS:
        total_occ = sum(count_keyword_occurrences(b, kw) for b in blobs)
        rows_with_kw = int(blobs.apply(lambda x, k=kw: row_has_keyword(x, k)).sum())
        overall_keyword_rows.append(
            {"keyword": kw, "total_occurrences": total_occ, "n_rows_with_keyword": rows_with_kw}
        )
    overall_df = pd.DataFrame(overall_keyword_rows).sort_values("total_occurrences", ascending=False)
    overall_df.to_csv(tables_dir / "keyword_frequency_overall.csv", index=False)

    subsets = [
        ("pred_delir", klasse == 1),
        ("pred_non_delir", klasse == 0),
        ("icd10_positive", baseline_icd10 == 1),
        ("icdsc_ge4_positive", baseline_ge4 == 1),
    ]

    subset_long: List[Dict[str, object]] = []
    for subset_name, mask in subsets:
        sub_blobs = blobs[mask].reset_index(drop=True)
        nn = len(sub_blobs)
        for kw in ANALYSIS_KEYWORDS:
            occ = sum(count_keyword_occurrences(b, kw) for b in sub_blobs)
            nk = int(sub_blobs.apply(lambda x, k=kw: row_has_keyword(x, k)).sum())
            subset_long.append(
                {
                    "subset": subset_name,
                    "keyword": kw,
                    "n_rows_subset": nn,
                    "n_rows_keyword_present": nk,
                    "total_occurrences": occ,
                    "fraction_rows_with_keyword": (nk / nn) if nn else float("nan"),
                }
            )
    pd.DataFrame(subset_long).to_csv(tables_dir / "keyword_frequency_by_subset.csv", index=False)

    N = len(df)
    base_rate_cls1 = float(klasse.mean()) if N else float("nan")
    base_icd10 = float(baseline_icd10.mean()) if N else float("nan")
    base_ge4 = float(baseline_ge4.mean()) if N else float("nan")

    cond_rows: List[Dict[str, object]] = []
    for kw in ANALYSIS_KEYWORDS:
        m = blobs.apply(lambda x, k=kw: row_has_keyword(x, k))
        nk = int(m.sum())
        if nk == 0:
            cond_rows.append(
                {
                    "keyword": kw,
                    "n_rows_with_keyword": 0,
                    "p_klasse_1_given_keyword": float("nan"),
                    "p_baseline_icd10_given_keyword": float("nan"),
                    "p_baseline_icdsc_ge_4_given_keyword": float("nan"),
                    "p_klasse_1_population": base_rate_cls1,
                    "p_icd10_population": base_icd10,
                    "p_icdsc_ge4_population": base_ge4,
                }
            )
        else:
            cond_rows.append(
                {
                    "keyword": kw,
                    "n_rows_with_keyword": nk,
                    "p_klasse_1_given_keyword": float(klasse[m].mean()),
                    "p_baseline_icd10_given_keyword": float(baseline_icd10[m].mean()),
                    "p_baseline_icdsc_ge_4_given_keyword": float(baseline_ge4[m].mean()),
                    "p_klasse_1_population": base_rate_cls1,
                    "p_icd10_population": base_icd10,
                    "p_icdsc_ge4_population": base_ge4,
                }
            )
    pd.DataFrame(cond_rows).to_csv(tables_dir / "keyword_conditional_probabilities.csv", index=False)

    strat_rows: List[Dict[str, object]] = []
    if "baseline_icd10" in df.columns:
        y = baseline_icd10
        strata: Sequence[Tuple[str, pd.Series]] = [
            ("true_positive", ((klasse == 1) & (y == 1))),
            ("false_positive", ((klasse == 1) & (y == 0))),
            ("false_negative", ((klasse == 0) & (y == 1))),
            ("true_negative", ((klasse == 0) & (y == 0))),
        ]
        for strat_name, mask in strata:
            sub_blobs = blobs[mask]
            nc = len(sub_blobs)
            for kw in ANALYSIS_KEYWORDS:
                nk = sum(1 for b in sub_blobs if row_has_keyword(b, kw))
                strat_rows.append(
                    {
                        "stratum_icd10": strat_name,
                        "keyword": kw,
                        "n_rows": nc,
                        "n_rows_keyword_present": nk,
                        "fraction_with_keyword": (nk / nc) if nc else float("nan"),
                    }
                )
        pd.DataFrame(strat_rows).to_csv(
            tables_dir / "keyword_prominence_icd10_stratification.csv", index=False
        )
        _export_top_stratum_keywords(strat_rows, tables_dir)

    pivot = pd.DataFrame(subset_long).pivot(
        index="keyword", columns="subset", values="fraction_rows_with_keyword"
    )
    pivot = pivot.reindex([k for k in ANALYSIS_KEYWORDS]).fillna(0)

    fig, ax = plt.subplots(figsize=(9.5, max(6, 0.28 * len(pivot))))
    pivot.plot(kind="barh", ax=ax, fontsize=9)
    ax.set_xlabel("Fraction of rows containing keyword")
    ax.set_title(
        "Keyword presence rate by subset\nInference text combines model excerpts and rationale columns"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "keyword_fraction_by_subset.png", dpi=120)
    plt.close(fig)

    head_n = min(24, len(overall_df))
    head = overall_df.head(head_n)
    fig2, ax2 = plt.subplots(figsize=(10, max(5, 0.32 * len(head))))
    ax2.barh(np.arange(len(head)), head["total_occurrences"].values, color="#1d4ed8")
    ax2.set_yticks(np.arange(len(head)))
    ax2.set_yticklabels(head["keyword"], fontsize=9)
    ax2.set_xlabel("Total substring occurrences (all rows)")
    ax2.set_title("Keywords ranked by occurrence count")
    ax2.invert_yaxis()
    fig2.tight_layout()
    fig2.savefig(plots_dir / "keyword_occurrences_overall.png", dpi=120)
    plt.close(fig2)

    if pivot.shape[0] and pivot.shape[1]:
        hm = pivot.values.astype(float)
        fig3, ax3 = plt.subplots(figsize=(max(8.5, pivot.shape[1] * 2), max(8, 0.32 * pivot.shape[0])))
        im = ax3.imshow(hm, aspect="auto", cmap="Blues")
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(list(pivot.index), fontsize=7)
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(list(pivot.columns), rotation=28, ha="right", fontsize=8)
        ax3.set_title(
            "Keyword mention rate by clinical/model subset\n"
            "(fraction of rows in subset where inference text mentions term)"
        )
        ncells = hm.shape[0] * hm.shape[1]
        annotate = ncells <= 120
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                txt = f"{hm[i, j]:.2f}"
                lum = hm[i, j]
                clr = "#f8fafc" if lum > 0.45 else "#111827"
                if annotate:
                    ax3.text(j, i, txt, ha="center", va="center", fontsize=6, color=clr)
        fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.02)
        fig3.tight_layout()
        fig3.savefig(plots_dir / "keyword_subset_heatmap.png", dpi=120, bbox_inches="tight")
        plt.close(fig3)

    lines = [
        "Keyword analysis",
        f"n_rows={len(df)}",
        f"source={cmp_path}",
        f"Inference text concatenates (when columns exist): {', '.join(TEXT_SOURCE_COLUMNS)}.",
        "",
        "p_*_given_keyword = empirical prevalence among rows containing substring.",
        "ICD10 stratification for top-keyword exports uses baseline_icd10 as reference label.",
        "",
        f"tables: {tables_dir}",
        f"plots: {plots_dir}",
    ]
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Keyword tables: {tables_dir}")
    print(f"Report: {out_dir / 'report.txt'}")


def main() -> None:
    run_keyword_analysis()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
