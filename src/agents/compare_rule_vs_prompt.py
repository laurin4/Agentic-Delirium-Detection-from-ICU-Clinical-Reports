

from pathlib import Path
import csv
from src.pipeline.paths import PREDICTIONS_DIR, COMPARISONS_DIR


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_by_report(rows):
    return {row["bericht"]: row for row in rows}


def normalize(value: str) -> str:
    return (value or "").strip()


def main():
    rule_path = PREDICTIONS_DIR / "agent1_agent2_agent3_results_rule.csv"
    prompt_path = PREDICTIONS_DIR / "agent1_agent2_agent3_results_prompt.csv"
    compare_path = COMPARISONS_DIR / "compare_rule_vs_prompt.csv"

    if not rule_path.exists():
        print(f"Fehlt: {rule_path}")
        return
    if not prompt_path.exists():
        print(f"Fehlt: {prompt_path}")
        return

    rule_rows = load_csv(rule_path)
    prompt_rows = load_csv(prompt_path)

    rule_index = index_by_report(rule_rows)
    prompt_index = index_by_report(prompt_rows)

    all_reports = sorted(set(rule_index) | set(prompt_index))
    comparison_rows = []

    print("\n=== Vergleich: rule vs prompt ===\n")

    for report in all_reports:
        rule_row = rule_index.get(report, {})
        prompt_row = prompt_index.get(report, {})

        rule_klasse = normalize(rule_row.get("klasse", ""))
        prompt_klasse = normalize(prompt_row.get("klasse", ""))
        rule_klassifikation = normalize(rule_row.get("klassifikation", ""))
        prompt_klassifikation = normalize(prompt_row.get("klassifikation", ""))
        rule_signalstaerke = normalize(rule_row.get("signalstaerke", ""))
        prompt_signalstaerke = normalize(prompt_row.get("signalstaerke", ""))
        rule_signale = normalize(rule_row.get("delir_signale", ""))
        prompt_signale = normalize(prompt_row.get("delir_signale", ""))

        same_class = rule_klasse == prompt_klasse and rule_klassifikation == prompt_klassifikation
        same_signal_strength = rule_signalstaerke == prompt_signalstaerke
        same_signals = rule_signale == prompt_signale

        status = "gleich" if (same_class and same_signal_strength and same_signals) else "unterschied"

        print(f"[{report}] {status}")
        print(f"  rule   -> klasse={rule_klasse}, klassifikation={rule_klassifikation}, signalstaerke={rule_signalstaerke}")
        print(f"  prompt -> klasse={prompt_klasse}, klassifikation={prompt_klassifikation}, signalstaerke={prompt_signalstaerke}")
        if not same_signals:
            print(f"  rule_signale:   {rule_signale}")
            print(f"  prompt_signale: {prompt_signale}")
        print()

        comparison_rows.append(
            {
                "bericht": report,
                "status": status,
                "rule_klasse": rule_klasse,
                "prompt_klasse": prompt_klasse,
                "rule_klassifikation": rule_klassifikation,
                "prompt_klassifikation": prompt_klassifikation,
                "rule_signalstaerke": rule_signalstaerke,
                "prompt_signalstaerke": prompt_signalstaerke,
                "rule_delir_signale": rule_signale,
                "prompt_delir_signale": prompt_signale,
            }
        )

    compare_path.parent.mkdir(parents=True, exist_ok=True)
    with open(compare_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bericht",
                "status",
                "rule_klasse",
                "prompt_klasse",
                "rule_klassifikation",
                "prompt_klassifikation",
                "rule_signalstaerke",
                "prompt_signalstaerke",
                "rule_delir_signale",
                "prompt_delir_signale",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Vergleich gespeichert in: {compare_path}")


if __name__ == "__main__":
    main()