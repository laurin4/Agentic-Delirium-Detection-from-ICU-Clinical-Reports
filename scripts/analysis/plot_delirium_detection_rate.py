#!/usr/bin/env python3
"""
Horizontal bar chart: proportion of manually validated delirium-positive patients
correctly detected by each method (thesis figure).

READ-ONLY / SELF-CONTAINED:
  * Uses fixed patient-level values from the thesis results (defined below).
  * Does NOT read or modify the production pipeline, prompts, or stored data.
  * matplotlib only (no seaborn).

Outputs:
  figures/delirium_detection_rate_by_method.pdf
  figures/delirium_detection_rate_by_method.png
  results/delirium_detection_rate_by_method.csv

Run:
  python scripts/analysis/plot_delirium_detection_rate.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt

# scripts/analysis/<this file> -> parents[2] == delirium_project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

PDF_PATH = FIGURES_DIR / "delirium_detection_rate_by_method.pdf"
PNG_PATH = FIGURES_DIR / "delirium_detection_rate_by_method.png"
CSV_PATH = RESULTS_DIR / "delirium_detection_rate_by_method.csv"

TOTAL_POSITIVE = 23

# (method, true positives, sensitivity %) — fixed thesis values.
METHODS: List[Tuple[str, int, float]] = [
    ("Composite AND", 4, 17.4),
    ("ICD-10", 5, 21.7),
    ("ICDSC", 17, 73.9),
    ("Composite OR", 18, 78.3),
    ("Reviewer Cascade", 20, 87.0),
    ("V2 Run 02", 21, 91.3),
]

X_LABEL = "Sensitivity (%)"


def write_csv(rows: List[Tuple[str, int, float]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["method,true_positives,total_positive,sensitivity_pct"]
    for name, tp, sens in rows:
        lines.append(f"{name},{tp},{TOTAL_POSITIVE},{sens}")
    CSV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(rows: List[Tuple[str, int, float]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    names = [r[0] for r in rows]
    tps = [r[1] for r in rows]
    sens = [r[2] for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(y, sens, color="#4C72B0", edgecolor="none", height=0.62, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel(X_LABEL, fontsize=12)
    ax.tick_params(axis="x", labelsize=10)

    ax.set_xlim(0, 100)

    # Light vertical grid only.
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color="0.85", zorder=0)
    ax.set_axisbelow(True)

    # Clean frame.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # End-of-bar labels (outside, whole-number percent): "TP/23 (XX%)".
    for bar, tp, s in zip(bars, tps, sens):
        label = f"{tp}/{TOTAL_POSITIVE} ({s:.0f}%)"
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=10,
            color="black",
            clip_on=False,  # allow labels past x=100 into the right margin
            zorder=4,
        )

    # Generous left/right margins so names and end labels are never cut off.
    fig.subplots_adjust(left=0.22, right=0.88)

    fig.savefig(PDF_PATH, facecolor="white", bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Order methods from lowest to highest sensitivity.
    rows = sorted(METHODS, key=lambda r: r[2])
    write_csv(rows)
    make_plot(rows)
    print("Wrote:")
    print(f"  {PDF_PATH}")
    print(f"  {PNG_PATH}")
    print(f"  {CSV_PATH}")


if __name__ == "__main__":
    main()
