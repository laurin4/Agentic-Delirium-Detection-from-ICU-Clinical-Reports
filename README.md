# Delirium detection from anonymized ICU reports

Binary delirium detection from German clinical report text, compared against multiple structured baselines derived from ICD-10 and ICDSC data.

## Purpose

- **Model output (binary):** `klasse = 0` → no delirium (`no_delir`), `klasse = 1` → delirium (`delir`).
- **Signal strength** (interpretation): `niedrig` | `mittel` | `hoch` → classification maps **`mittel` and `hoch` → klasse 1**, **`niedrig` → klasse 0**.
- **Baselines:** Several binary baseline columns per patient for evaluation (see below).
- **Primary LLM backend:** USZ local HTTP API (Gemma-class model). **Ollama** remains available as an optional comparison backend only.

Sensitive data are **not** stored in this repository; paths point to local CSVs you provide.

---

## Primary input: `data/raw/Berichte.csv`

Configured via `src/pipeline/paths.py` (`DATA_MODE`, `BERICHTE_INPUT_PATH`). Default production layout uses **`data/raw/Berichte.csv`** (semicolon-separated).

Expected columns include:

| Column | Role |
|--------|------|
| `PatientID` | Patient identifier (joined to baseline `PatientenID`) |
| `berdat` | Report date — **used only for sorting** rows per patient |
| `bertyp` | Optional metadata |
| `bername` | **Excluded** from model text |
| `diag`, `epikrise`, `jetziges_leiden`, `prozedere` | Combined into **`report_text`** per patient (section blocks), sorted by `berdat` |

**`Diagnosenliste.csv` is removed** from the active pipeline. Clinical text lives in **`Berichte.csv`** (`diag`, `epikrise`, `jetziges_leiden`, `prozedere` → section blocks in `report_text`).

Legacy fallback (`INPUT_MODE` in `run_pipeline.py`): `diagnosis` (synthetic only) or `txt` — production uses **`berichte`** only.

---

## Structured baselines (`outputs/baseline/structured_baseline.csv`)

Produced by `prepare_structured_data` from **`data/raw/ICD.csv`** and **`data/raw/ICDSC.csv`** only (semicolon-separated).

| File | Columns |
|------|---------|
| `ICD.csv` | `PatientID`, `icd_hd`, `icd_code` |
| `ICDSC.csv` | `PatientID`, `ICDSC_Max` (patient-level maximum) |

`PatientID` is normalized internally to `PatientenID` in the baseline artifact.

**Binary baseline columns** (all included in primary evaluation):

- `baseline_icdsc_ge_1` … `baseline_icdsc_ge_5`
- `baseline_icdsc_0`
- `baseline_icdsc_1_to_3`
- `baseline_icdsc_ge_4_grouped`
- `baseline_icd10`

**ICD-10 delirium definition:** main diagnosis **`icd_hd == 1`** and codes **`F05.0`**, **`F05.8`**, **`F05.9`** only. **`F05.1`** (alcohol-related delirium) and other F05 subcodes are excluded from the baseline cohort.

**LLM prompts** are German (`prompts/agent_*.txt`); JSON keys remain unchanged for parsers.

**Short-report fallback** (optional): `SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true`, `SHORT_REPORT_CHAR_THRESHOLD=1000` — sends capped full text for short `Verlaufseintrag` / `Verlegungsbericht` / `Austrittsbericht` when the rule layer finds no snippets.

**Primary baseline `baseline_composite`** — set in `src/pipeline/paths.py`:

```python
BASELINE_COMPOSITE_MODE = "AND"  # or "OR" for thesis default
```

| Mode | Definition | Use |
|------|------------|-----|
| `OR` | ICDSC≥4 **or** ICD10 | Thesis / sensitive baseline |
| `AND` | ICDSC≥4 **and** ICD10 | Presentation: «sichere Delirfälle» (high-confidence coded delir) |

Model-positive / AND-baseline-negative cases are **Delirkandidaten** (possible uncoded or underdocumented delir), not strict false positives.

**ICDSC:** `max_icdsc` = `ICDSC_Max`; binary columns derived from thresholds (e.g. `baseline_icdsc_ge_4` ⇔ `ICDSC_Max >= 4`).

**Legacy:** `baseline_reference_class` (0/1/2) may still be written for backward compatibility; it is **not** the primary evaluation target. Use binary baselines and `evaluate_predictions`.

### Manual validation (PRIMARY thesis evaluation)

| Unit | Role |
|------|------|
| Report | One prediction per report (`klasse` → `model_report_prediction`) |
| Patient cohort | 100 unique patients; export **all** Verlauf / Verlegung / Austritt reports each |
| Manual GT | `manual_report_ground_truth` (0/1) per report |
| Derived patient GT | `derived_manual_patient_ground_truth` = max(report GT) — do not annotate manually |
| ICDSC / ICD10 in cohort | Reference signals only (exploratory comparison in `evaluate_manual_validation`) |

**Full thesis run (copy-paste):** see **[HANDOVER_SUMMARY.md — FINAL THESIS WORKFLOW (FULL RUN)](HANDOVER_SUMMARY.md#final-thesis-workflow-full-run)**. Summary: `unset MAX_REPORTS` → full `prepare_structured_data` + `run_pipeline` + `run_validation_suite` → export cohort + slim labels → `freeze_validation_cohort` → annotate `manual_report_labels_frozen.csv` → `evaluate_manual_validation`.

---

## LLM providers

### Primary: USZ API (default)

No local Ollama is required for the default run.

```bash
export LLM_PROVIDER=usz_api
export USZ_LLM_URL=http://localhost:8100/generate
export LLM_MODEL_LABEL=gemma4_26b_usz
```

If `LLM_MODEL_LABEL` is unset with `usz_api`, the code defaults to **`gemma4_26b_usz`**.  
`OLLAMA_MODEL` is **not** required when using USZ.

### Optional comparison: Ollama

```bash
export LLM_PROVIDER=ollama
export OLLAMA_URL=http://127.0.0.1:11500
export OLLAMA_MODEL=qwen2.5:7b
```

### Shared generation settings (both backends)

```bash
export LLM_TEMPERATURE=0
export LLM_TOP_P=1
export LLM_MAX_TOKENS=1000
export LLM_TIMEOUT=120
export LLM_LONG_INPUT_WARNING_CHARS=12000
```

USZ additionally:

```bash
export LLM_DISABLE_THINK=false
```

Ollama context window:

```bash
export OLLAMA_NUM_CTX=8192
```

Ollama maps `LLM_MAX_TOKENS` → `num_predict`, and uses `LLM_TEMPERATURE`, `LLM_TOP_P`, `OLLAMA_NUM_CTX` in the chat `options`.

**Outputs:**

- Always: `outputs/predictions/agent1_agent2_agent3_results_prompt.csv` (downstream steps read this file).
- Copy: `outputs/predictions/agent_results_<provider>_<model_label>.csv`  
  Examples: `agent_results_usz_api_gemma4_26b_usz.csv`, `agent_results_ollama_qwen2_5_7b.csv`.

**Limit run size (pilot / dev only):** by default the **full** evaluatable corpus is processed. Cap with `MAX_REPORTS`:

```bash
MAX_REPORTS=60 python -m src.pipeline.run_pipeline   # first 60 reports (stable loader order)
MAX_REPORTS=all python -m src.pipeline.run_pipeline   # explicit full corpus (same as unset)
```

If `MAX_REPORTS` is unset, there is **no** report cap (`paths.DEFAULT_MAX_REPORTS = None`).

**Optional SQLite append log** (CSV remains the canonical artifact):

```bash
ENABLE_SQLITE_LOGGING=true python -m src.pipeline.run_pipeline
```

Writes rows to `outputs/logs/prediction_run.sqlite` (see `src/pipeline/sqlite_logging.py`).

**Pre-LLM evidence extraction** (`src/preprocessing/evidence_extraction.py`): the stitched `report_text` is scanned with rule-based keyword groups (direct delirium wording, indirect symptoms, negations, prophylaxis/screening). Only bounded, section-labeled snippets are sent to the LLM as `llm_report_text`. If nothing actionable is found (e.g. negation-only), the LLM is skipped and `klasse=0`. Tune caps with:

```bash
export EVIDENCE_MAX_SNIPPETS=12
export EVIDENCE_MAX_LLM_CHARS=8000
export EVIDENCE_WINDOW_SENTENCES=1
export EVIDENCE_MAX_SNIPPET_CHARS=400
export DEBUG_LLM_OUTPUT=false   # true = verbose per-agent dumps to stdout
```

---

## Command order (recommended)

For the **final thesis validation path** (full dataset, frozen cohort, patient-level primary metrics), use **[HANDOVER_SUMMARY.md — FINAL THESIS WORKFLOW (FULL RUN)](HANDOVER_SUMMARY.md#final-thesis-workflow-full-run)** instead of the exploratory list below.

From the project root (`delirium_project/`):

```bash
python -m src.pipeline.prepare_structured_data
python -m src.analysis.run_data_coverage_analysis   # Berichte vs baseline; Dokumentationsblatt excluded from counts
python -m src.analysis.run_icd_icdsc_overlap_analysis
python -m src.pipeline.run_pipeline              # report-level predictions (excludes bertyp=Dokumentationsblatt)
python -m src.pipeline.compare_reports_vs_baseline
python -m src.pipeline.evaluate_predictions      # primary baseline: baseline_composite (see BASELINE_COMPOSITE_MODE)
python -m src.analysis.create_patient_reporttype_matrix
python -m src.analysis.export_patient_validation_cohort   # PRIMARY: 100 patients, all reports each
python -m src.analysis.export_manual_report_labels         # slim annotation CSV
python -m src.analysis.freeze_validation_cohort          # fixed cohort (after full inference)
# Annotate manual_report_labels.csv (or frozen copy), then:
python -m src.analysis.evaluate_manual_validation        # uses frozen_validation_cohort/ when present
python -m src.analysis.export_presentation_examples     # slide-ready pipeline walkthrough examples
python -m src.analysis.run_field_delirium_analysis
```

After `compare_reports_vs_baseline`, you can generate **interpretability and error-review exports** (science / review tooling; read-only on model logic):

```bash
python -m src.analysis.run_error_review_export
python -m src.analysis.run_keyword_analysis
python -m src.analysis.run_field_signal_analysis
python -m src.analysis.run_evidence_snippets_export
python -m src.analysis.run_full_analysis_suite   # runs manual review export + keyword + field signal + evidence export
```

**One-shot validation helpers** (assumes structured baseline + predictions already exist where applicable):

```bash
python -m src.analysis.run_validation_suite
```

Outputs land under `outputs/analysis/manual_review/` (TP/TN/FP/FN samples for primary baselines), `keyword_analysis/`, `field_signal_analysis/`, `analysis/evidence/tables/`, and legacy `outputs/analysis/error_review/` (unused by the new manual review export).

Optional: `python -m src.validation.validate_inputs`, `python -m src.analysis.run_exploration`, `python -m src.analysis.run_analysis`, `python -m src.analysis.run_false_negative_review`.

---

## Outputs (overview)

| Area | Location |
|------|-----------|
| Structured baseline | `outputs/baseline/structured_baseline.csv` |
| Predictions (canonical) | `outputs/predictions/agent1_agent2_agent3_results_prompt.csv` |
| Predictions (tagged copy) | `outputs/predictions/agent_results_<provider>_<model_label>.csv` |
| Report vs baseline merge | `outputs/comparisons/report_vs_baseline_comparison.csv` (evaluable rows only) |
| Excluded predictions (no / incomplete baseline) | `outputs/comparisons/report_vs_baseline_excluded_missing_baseline.csv` |
| Binary baseline evaluation | `outputs/evaluation/binary_baselines/` (tables, plots, `report.txt`) |
| Data coverage | `outputs/analysis/data_coverage/` |
| ICD vs ICDSC overlap | `outputs/analysis/icd_icdsc_overlap/` |
| Field keyword / OR analysis | `outputs/analysis/field_delirium/` |
| Manual review (TP/TN/FP/FN samples, primary baselines) | `outputs/analysis/manual_review/` |
| Legacy error-review directory | `outputs/analysis/error_review/` |
| Optional SQLite prediction log | `outputs/logs/prediction_run.sqlite` |
| Keyword / term stratification | `outputs/analysis/keyword_analysis/` |
| Field signal vs model / baselines | `outputs/analysis/field_signal_analysis/` |
| Evidence snippets (interpretability CSV) | `outputs/analysis/evidence/tables/` |
| LLM debug dumps | `outputs/logs/llm_debug/` |

---

## USZ API smoke test

```bash
python scripts/test_usz_llm_api.py
```

---

## Installation

Python **3.9+** recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Further reading

- **`RUNBOOK.md`** — server setup, troubleshooting, sanity checks after a run.
- **`GIT_SETUP.md`** — how to sync code between Mac and Ubuntu without committing patient data.
- **`PROJECT_STATUS.md`** — brief pointer; detailed narrative lives in this README.

Before any commit, run the safety check:

```bash
python scripts/check_no_sensitive_files.py
```

---

## Synthetic data (optional)

```bash
python scripts/generate_synthetic_data.py
```

Set `DATA_MODE = "synthetic"` in `src/pipeline/paths.py` to use generated CSVs under `data/structured/raw/` (see `paths.py`).
