"""
Supervisor-friendly heatmap preview of the patient-level report-type matrix.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

_mpl_config = Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.pipeline.paths import (
    PATIENT_LEVEL_ANALYSIS_DIR,
    PATIENT_REPORTTYPE_MATRIX_PATH,
    PATIENT_REPORTTYPE_MATRIX_PREVIEW_PDF,
    PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG,
)
from src.preprocessing.berichte_filters import REPORT_TYPES_FOR_MATRIX

LOGGER = logging.getLogger(__name__)

DEFAULT_PREVIEW_N = 7
TITLE = "Patient-level report-type matrix preview"
SUBTITLE = (
    "Dokumentationsblatt excluded; report-level predictions aggregated per patient"
)

# imshow codes: 0 = empty/manual NA, 1 = binary 0, 2 = binary 1
COLOR_EMPTY = "#d0d0d0"
COLOR_ZERO = "#fafafa"
COLOR_ONE = "#ef9a9a"

ICDSC_COLUMN_CANDIDATES = ("baseline_icdsc_ge_4", "ICDSC_ge_4")


def preview_patient_count() -> int:
    """Number of patients shown; override with env ``MATRIX_PREVIEW_N`` (default 7)."""
    raw = os.environ.get("MATRIX_PREVIEW_N", str(DEFAULT_PREVIEW_N)).strip()
    try:
        n = int(raw)
    except ValueError:
        LOGGER.warning("Invalid MATRIX_PREVIEW_N=%r; using %s", raw, DEFAULT_PREVIEW_N)
        n = DEFAULT_PREVIEW_N
    return max(1, n)


def save_preview_pdf() -> bool:
    """When false (env ``MATRIX_PREVIEW_PDF=0``), skip the PDF export."""
    flag = os.environ.get("MATRIX_PREVIEW_PDF", "1").strip().lower()
    return flag not in ("0", "false", "no")


def resolve_icdsc_ge4_series(matrix: pd.DataFrame) -> pd.Series:
    """Binary ICDSC≥4 column from matrix or derived from ``ICDSC_max``."""
    for col in ICDSC_COLUMN_CANDIDATES:
        if col in matrix.columns:
            return pd.to_numeric(matrix[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if "ICDSC_max" in matrix.columns:
        return (pd.to_numeric(matrix["ICDSC_max"], errors="coerce").fillna(0) >= 4).astype(int)
    return pd.Series(0, index=matrix.index, dtype=int)


def _binary_cell(value: object) -> Tuple[int, str]:
    v = int(pd.to_numeric(value, errors="coerce") or 0)
    v = 1 if v >= 1 else 0
    return v, str(v)


def _manual_cell(value: object) -> Tuple[int, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return -1, ""
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return -1, ""
    try:
        v = int(float(text))
    except ValueError:
        return -1, ""
    if v not in (0, 1):
        return -1, ""
    return v, str(v)


def _state_to_imshow(state: int) -> int:
    if state < 0:
        return 0
    if state == 0:
        return 1
    return 2


def preview_column_labels() -> List[str]:
    return [
        "ICDSC≥4",
        "ICD10",
        "Baseline\ncomposite",
        "Verlauf",
        "Verlegung",
        "Austritt",
        "Model\npatient+",
        "Manual\nGT",
    ]


def build_preview_cell_data(
    matrix: pd.DataFrame,
    *,
    n_patients: int | None = None,
) -> Tuple[np.ndarray, List[str], List[List[str]]]:
    """
    Build heatmap state grid and display strings for the first *n_patients* rows.

    Returns (states [n_rows, n_cols], patient_ids, text_grid).
    """
    if matrix.empty:
        raise ValueError("Matrix is empty; nothing to plot.")

    n = n_patients if n_patients is not None else preview_patient_count()
    subset = matrix.head(n).copy()
    patient_ids = [str(pid) for pid in subset["PatientenID"].tolist()]

    icdsc = resolve_icdsc_ge4_series(subset)
    columns_data: List[Tuple[pd.Series | None, bool]] = [
        (icdsc, False),
        (subset["ICD10"] if "ICD10" in subset.columns else pd.Series(0, index=subset.index), False),
        (
            subset["baseline_composite"]
            if "baseline_composite" in subset.columns
            else pd.Series(0, index=subset.index),
            False,
        ),
    ]
    for rt in REPORT_TYPES_FOR_MATRIX:
        col = subset[rt] if rt in subset.columns else pd.Series(0, index=subset.index)
        columns_data.append((col, False))
    columns_data.append(
        (
            subset["model_patient_positive"]
            if "model_patient_positive" in subset.columns
            else pd.Series(0, index=subset.index),
            False,
        )
    )
    if "manual_ground_truth" in subset.columns:
        columns_data.append((subset["manual_ground_truth"], True))
    else:
        columns_data.append((pd.Series([""] * len(subset), index=subset.index), True))

    n_cols = len(columns_data)
    states = np.zeros((len(subset), n_cols), dtype=int)
    texts: List[List[str]] = []

    for i in range(len(subset)):
        row_states: List[int] = []
        row_texts: List[str] = []
        for series, is_manual in columns_data:
            val = series.iloc[i] if series is not None else 0
            if is_manual:
                state, label = _manual_cell(val)
            else:
                state, label = _binary_cell(val)
            row_states.append(state)
            row_texts.append(label)
        states[i, :] = row_states
        texts.append(row_texts)

    return states, patient_ids, texts


def plot_patient_reporttype_matrix_preview(
    matrix: pd.DataFrame,
    *,
    png_path: Path = PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG,
    pdf_path: Path = PATIENT_REPORTTYPE_MATRIX_PREVIEW_PDF,
    n_patients: int | None = None,
) -> Path:
    """Render and save the matrix preview heatmap; returns path to PNG."""
    states, patient_ids, texts = build_preview_cell_data(matrix, n_patients=n_patients)
    col_labels = preview_column_labels()
    if len(col_labels) != states.shape[1]:
        col_labels = col_labels[: states.shape[1]]

    imshow_grid = np.vectorize(_state_to_imshow)(states)
    cmap = ListedColormap([COLOR_EMPTY, COLOR_ZERO, COLOR_ONE])

    n_rows, n_cols = imshow_grid.shape
    fig_w = max(10.0, 1.05 * n_cols + 3.0)
    fig_h = max(4.5, 0.55 * n_rows + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(imshow_grid, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(patient_ids, fontsize=9)
    ax.set_xlabel("Baseline / report type / model / manual", fontsize=10, labelpad=8)
    ax.set_ylabel("PatientenID", fontsize=10, labelpad=8)
    ax.tick_params(top=False, right=False)

    for i in range(n_rows):
        for j in range(n_cols):
            label = texts[i][j]
            state = int(states[i, j])
            if state < 0:
                continue
            color = "#424242" if state == 1 else "#616161"
            weight = "bold" if state == 1 else "normal"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=11,
                color=color,
                fontweight=weight,
            )

    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#bdbdbd", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(TITLE, fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.93, SUBTITLE, ha="center", va="top", fontsize=9.5, color="#424242")

    legend_handles = [
        Patch(facecolor=COLOR_ZERO, edgecolor="#9e9e9e", label="0 = no delir evidence"),
        Patch(facecolor=COLOR_ONE, edgecolor="#9e9e9e", label="1 = delir evidence / positive"),
        Patch(facecolor=COLOR_EMPTY, edgecolor="#9e9e9e", label="empty = not annotated"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.90])

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    if save_preview_pdf():
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main(
    matrix_path: Path = PATIENT_REPORTTYPE_MATRIX_PATH,
    png_path: Path = PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG,
    pdf_path: Path = PATIENT_REPORTTYPE_MATRIX_PREVIEW_PDF,
) -> None:
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Matrix CSV missing: {matrix_path}. "
            "Run python -m src.analysis.create_patient_reporttype_matrix first."
        )
    matrix = pd.read_csv(matrix_path)
    out = plot_patient_reporttype_matrix_preview(matrix, png_path=png_path, pdf_path=pdf_path)
    print(f"Wrote matrix preview plot: {out}")
    if save_preview_pdf():
        print(f"Wrote matrix preview PDF: {pdf_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
