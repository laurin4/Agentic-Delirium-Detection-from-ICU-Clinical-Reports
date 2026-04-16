# Delirium Detection Pipeline

Pipeline zur Extraktion und Klassifikation von Delir aus Diagnose-Einträgen (Text) und Vergleich mit strukturierten Daten (ICD10, ICDSC).

---

## Daten (Standard: `Data/Raw/` CSV)

Lege die drei Dateien **exakt mit diesen Namen** unter `Data/Raw/` ab:

| Datei | Pfad |
|--------|------|
| `Diagnosenliste.csv` | `Data/Raw/` |
| `ICD.csv` | `Data/Raw/` |
| `ICDSC.csv` | `Data/Raw/` |

Die aktiven Pfade und der Modus (`real` vs. `synthetic`) stehen zentral in `src/pipeline/paths.py` (`DATA_MODE`, `ICD10_PATH`, `ICDSC_PATH`, `DIAGNOSIS_INPUT_PATH`).

Unterstützte Formate pro Datei: **`.csv`** sowie optional **`.xlsx` / `.xls`** (über `src/pipeline/tabular_io.read_tabular`).

---

## Projektstruktur (Auszug)

```
Data/
  Raw/                # ICD.csv, ICDSC.csv, Diagnosenliste.csv
data/                 # optional synthetic/test data
outputs/
  baseline/
  predictions/
  comparisons/
  evaluation/
  validation/
src/
  agents/
  pipeline/
  preprocessing/
  validation/
```

---

## Pipeline (Ubuntu / lokal identisch)

```bash
python -m src.pipeline.prepare_structured_data
python -m src.pipeline.run_pipeline
python -m src.pipeline.compare_reports_vs_baseline
python -m src.pipeline.evaluate_predictions
python -m src.validation.validate_inputs
python -m src.analysis.run_exploration
python -m src.analysis.run_analysis
```

- **Diagnose-Input:** `INPUT_MODE = "diagnosis"` in `src/pipeline/run_pipeline.py` (Standard); Quelle = `DIAGNOSIS_INPUT_PATH` aus `paths.py`.
- **Optional:** `INPUT_MODE = "txt"` für `data/anonymized/generische Arztberichte/*.txt`.

### Outputs (Auszug)

- `outputs/baseline/structured_baseline.csv`
- `outputs/predictions/agent1_agent2_agent3_results_*.csv`
- `outputs/comparisons/report_vs_baseline_comparison.csv`
- `outputs/evaluation/` — Multiclass-Metriken, CSVs, `plots/`; binäre Hilfsmetriken als `*_binary_secondary*`
- `outputs/validation/` — `validation_results.csv`, `validation_summary.txt`
- `outputs/analysis/`
  - `exploration/` (raw-input EDA: top diagnosis terms, ICD frequencies, missingness, temporal patterns, patient activity)
  - `tables/` (input quality, distributions, patient-level deep tables, confusion/error tables)
  - `plots/` (class distribution, signal composition, confusion heatmap, error-vs-hits, report-length histogram)
  - `reports/analysis_summary.txt`

---

## Klassifikation (Agent 3)

| Klasse | Bedeutung |
|--------|-----------|
| 0 | kein Delir |
| 1 | mögliches Delir |
| 2 | dokumentiertes Delir |

---

## Baseline (3 Klassen, Referenz aus ICD10 + ICDSC)

- Klasse 2: gültiger F05-Delircode (ohne F05.1) **und** `ICDSC_DelirFlag == 1`
- Klasse 1: nicht 2, aber F05* (ohne F05.1) **oder** Flag **oder** `max(ICDSC_Value) >= 4`
- Klasse 0: sonst

Details: `PROJECT_STATUS.md`.

---

## Deployment (Ubuntu / Server)

1. Projektordner kopieren, virtuelle Umgebung mit `requirements.txt` installieren.
2. Die drei CSV-Dateien nach `Data/Raw/` legen (`ICD.csv`, `ICDSC.csv`, `Diagnosenliste.csv`).
3. Befehlskette wie oben ausführen.

**Hinweise:** Sensible Daten nicht ins Repo legen (`data/` ist in `.gitignore`). Unter Linux ggf. Locale/Fontconfig-Meldungen von Matplotlib — Evaluation setzt `MPLCONFIGDIR` unter `outputs/.mplconfig`.

### Optional: Docker (Ubuntu)

```bash
docker build -f docker/Dockerfile -t delirium-pipeline .
docker run --rm -it \
  -v "$(pwd)/Data/Raw:/app/Data/Raw" \
  -v "$(pwd)/outputs:/app/outputs" \
  delirium-pipeline \
  python -m src.pipeline.prepare_structured_data
```

Danach analog im Container ausführen:
- `python -m src.pipeline.run_pipeline`
- `python -m src.pipeline.compare_reports_vs_baseline`
- `python -m src.pipeline.evaluate_predictions`
- `python -m src.validation.validate_inputs`
- `python -m src.analysis.run_exploration`
- `python -m src.analysis.run_analysis`

---

## Synthetische Testdaten (optional)

```bash
python scripts/generate_synthetic_data.py
```

In `paths.py` `DATA_MODE = "synthetic"` setzen, um die generierten CSVs zu nutzen (Standard bleibt `real`).
