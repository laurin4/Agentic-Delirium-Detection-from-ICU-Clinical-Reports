from pathlib import Path
import re
import pandas as pd

from src.pipeline.paths import PREDICTIONS_DIR, COMPARISONS_DIR, OUTPUTS_DIR


MATCHING_DIR = OUTPUTS_DIR / "analysis" / "matching"
MATCHING_TABLES_DIR = MATCHING_DIR / "tables"
MATCHING_REPORTS_DIR = MATCHING_DIR / "reports"

DELIR_PATTERNS = [
    r"\bdelir\b",
    r"\bdelirium\b",
    r"\bdelirant\b",
    r"\bdelirös\b",
    r"\bhypoaktives delir\b",
    r"\bhyperaktives delir\b",
    r"\bdesorient\w*\b",
    r"\bvigilanz\w*\b",
    r"\bagitiert\w*\b",
    r"\bverwirr\w*\b",
]


def _has_delir_text(text: str) -> bool:
    text = str(text or "").lower()
    return any(re.search(pattern, text) for pattern in DELIR_PATTERNS)


def main():
    MATCHING_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MATCHING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"
    cmp_path = COMPARISONS_DIR / "report_vs_baseline_comparison.csv"

    pred = pd.read_csv(pred_path).copy()
    cmp = pd.read_csv(cmp_path).copy()

    pred["PatientenID"] = pred["PatientenID"].astype(str).str.strip()
    cmp["PatientenID"] = cmp["PatientenID"].astype(str).str.strip()

    matched_ids = set(cmp["PatientenID"].dropna().astype(str).str.strip())
    pred["matched_to_baseline"] = pred["PatientenID"].isin(matched_ids)

    text_col = "kontext"
    if "bericht" in pred.columns and "kontext" not in pred.columns:
        text_col = "bericht"

    pred["has_delir_text_signal"] = False
    if "delir_signale" in pred.columns:
        pred["has_delir_text_signal"] = pred["delir_signale"].fillna("").astype(str).str.strip().ne("")

    if text_col in pred.columns:
        pred["has_delir_text_keyword"] = pred[text_col].apply(_has_delir_text)
    else:
        pred["has_delir_text_keyword"] = False

    pred["has_any_delir_hint"] = pred["has_delir_text_signal"] | pred["has_delir_text_keyword"]

    unmatched = pred[~pred["matched_to_baseline"]].copy()
    unmatched_delir = unmatched[unmatched["has_any_delir_hint"]].copy()

    summary = pd.DataFrame(
        [
            ("n_predictions_total", len(pred)),
            ("n_matched_to_baseline", int(pred["matched_to_baseline"].sum())),
            ("n_unmatched", int((~pred["matched_to_baseline"]).sum())),
            ("n_unmatched_with_delir_hint", len(unmatched_delir)),
            ("n_matched_with_delir_hint", int(pred.loc[pred["matched_to_baseline"], "has_any_delir_hint"].sum())),
        ],
        columns=["metric", "count"],
    )

    pred.to_csv(MATCHING_TABLES_DIR / "prediction_matching_status.csv", index=False)
    unmatched.to_csv(MATCHING_TABLES_DIR / "unmatched_predictions.csv", index=False)
    unmatched_delir.to_csv(MATCHING_TABLES_DIR / "unmatched_predictions_with_delir_hint.csv", index=False)
    summary.to_csv(MATCHING_TABLES_DIR / "matching_summary.csv", index=False)

    with open(MATCHING_REPORTS_DIR / "matching_report.txt", "w", encoding="utf-8") as f:
        f.write("Matching analysis between predictions and baseline\n")
        f.write("=" * 60 + "\n\n")
        for _, row in summary.iterrows():
            f.write(f"{row['metric']}: {row['count']}\n")

    print(f"Matching tables: {MATCHING_TABLES_DIR}")
    print(f"Matching report: {MATCHING_REPORTS_DIR / 'matching_report.txt'}")


if __name__ == "__main__":
    main()