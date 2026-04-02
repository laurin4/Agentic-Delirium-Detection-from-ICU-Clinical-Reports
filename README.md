

# Delirium Detection Pipeline

Pipeline zur Extraktion und Klassifikation von Delir aus klinischen Berichten und Vergleich mit strukturierten Daten (ICD10, ICDSC).

---

## Projektstruktur

```
data/
  structured/
    icd10.csv
    icdsc.csv
  anonymized/
    generische Arztberichte/

outputs/
  baseline/
  predictions/
  comparisons/
  evaluation/

src/
  agents/
  pipeline/
```

---

## Pipeline Schritte

### 1. Strukturierte Daten vorbereiten
```bash
python -m src.pipeline.prepare_structured_data
```

→ erzeugt:
```
outputs/baseline/structured_baseline.csv
```

---

### 2. Reports verarbeiten (Agent 1–3)
```bash
python -m src.pipeline.run_pipeline
```

→ erzeugt:
```
outputs/predictions/agent1_agent2_agent3_results_*.csv
```

---

### 3. Vergleich mit Baseline
```bash
python -m src.pipeline.compare_reports_vs_baseline
```

→ erzeugt:
```
outputs/comparisons/report_vs_baseline_comparison.csv
```

---

### 4. Evaluation
```bash
python -m src.pipeline.evaluate_predictions
```

→ erzeugt:
```
outputs/evaluation/
  evaluation_summary.csv
  confusion_matrix_combined_baseline.csv
  false_positives.csv
  false_negatives.csv
```

---

## Klassifikation

| Klasse | Bedeutung |
|------|----------|
| 0 | kein Delir |
| 1 | mögliches Delir |
| 2 | dokumentiertes Delir |

---

## Baseline Definition

Delir = 1 wenn:
- ICD10 enthält **F05**
- UND ICDSC_DelirFlag = 1

sonst:
- Delir = 0

---

## Evaluation

Vergleich basiert auf:

- Prediction: `klasse == 2`
- Baseline: `baseline_delir_reference == 1`

Metriken:
- Accuracy
- Precision
- Recall
- F1

---

## Wichtige Annahmen

- PatientenID ist im Berichtstext enthalten
- CSV-Strukturen:
  - ICD10: `PatientenID, Code, IsHauptDiagn`
  - ICDSC: `PatientenID, ICDSC_Time, ICDSC_Value, ICDSC_DelirFlag`

---

## Deployment (Ubuntu / Server)

Vorbereitung:
- nur Pfade in `src/pipeline/paths.py` anpassen
- Daten in `data/` ablegen

Dann Pipeline identisch ausführen.
