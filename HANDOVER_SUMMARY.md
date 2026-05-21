# Handover Summary

## Project Purpose
- Detect ICU delirium from clinical diagnosis text using a 3-agent pipeline.
- Compare model predictions against an operational structured baseline (ICD10 + ICDSC).
- Provide reproducible validation, evaluation, and exploratory analysis outputs.

## Core Architecture
- **Pre-LLM layer**: structured rule-based evidence extraction (`src/preprocessing/evidence_extraction.py`) — full `report_text` is scanned; only bounded, section-tagged snippets are assembled for the LLM.
- **Agent 1**: extraction (`src/agents/extraction.py`) — JSON signal buckets from the **evidence bundle** (not the full report).
- **Agent 2**: interpretation (rule/prompt; default prompt) (`src/agents/interpretation.py`, `src/agents/interpretation_llm.py`) — assigns **signalstaerke** (`niedrig` / `mittel` / `hoch`).
- **Agent 3**: classification (`src/agents/classification.py`) — **binary** `klasse` 0/1 from signal strength (`mittel`/`hoch` → 1, `niedrig` → 0).
- **Prediction unit**: one row per **report** in `Berichte.csv` (report-level). Patient-level validation uses `patient_reporttype_matrix.csv`.
- **Excluded from processing** (raw CSV unchanged): `bertyp == Dokumentationsblatt` — logged as `excluded_dokumentationsblatt_count`.

## Evidence extraction (scientific / scalability)
- **Binary output only**: `klasse` ∈ {0, 1}. There is **no** multiclass prediction head.
- **Signal strength** remains `niedrig` | `mittel` | `hoch` (interpretation only); mapping to `klasse` is unchanged.
- The **entire** stitched `report_text` is scanned with deterministic keyword groups:
  - **direct_delir**, **indirect_symptom**, **negation**, **prophylaxis_or_risk** (see `evidence_extraction.py`).
- **Negated** delirium phrases are **not** treated as positive evidence; **prophylaxis / screening / risk-only** mentions are **not** auto-positive for delirium (the LLM is instructed; final class still flows through signal strength).
- If **no** snippet qualifies for LLM review, the LLM is normally **skipped** (`no_evidence_prefilter_skip`, `klasse=0`). Optional **short-report fallback** (see env below) sends capped full text for brief `Verlaufseintrag` / `Verlegungsbericht` / `Austrittsbericht` without keyword hits.
- If actionable snippets exist, `llm_text_reduction_method=structured_evidence_extraction` and the LLM receives **`llm_report_text`**: labeled snippets — **not** the full chart.
- **LLM prompts** (`prompts/agent_extraction.txt`, `prompts/agent_interpretation.txt`) are **German**; JSON field names (`signalstaerke`, `alternative_erklaerung`, …) stay English for parsers.
- **Transparency**: describe this two-stage design (rules → LLM) in thesis/defense materials; CSV stores structured `evidence_snippets` (JSON list) plus boolean flags for audit.
- **Clinical guardrails** (`src/agents/clinical_guardrails.py`, after Agent 2): hard-excludes **no evidence**, **prophylaxis/conditional-only**, **negated delirium**, and **isolated weak indirect symptoms** (single agitation/vigilance/GCS-only). **Direct delir** and **delirium-compatible symptom clusters** (≥2 indirect dimensions or therapy+symptoms) may stay `klasse=1`; clusters with `alternative_erklaerung` are flagged for review. **Isolated indirect + dominant alternative** → `klasse=0`, `alternative_explanation_downgrade`.

### Environment (evidence + logging)
| Variable | Default | Role |
|----------|---------|------|
| `EVIDENCE_MAX_SNIPPETS` | 12 | Max structured snippets per patient. |
| `EVIDENCE_MAX_LLM_CHARS` | 8000 | Cap on assembled LLM evidence bundle size. |
| `EVIDENCE_WINDOW_SENTENCES` | 1 | Sentences before/after the hit sentence in each window. |
| `EVIDENCE_MAX_SNIPPET_CHARS` | 400 | Max characters per snippet `text` field. |
| `DEBUG_LLM_OUTPUT` | false | If true, print verbose per-agent debug (full previews, raw LLM). |
| `LLM_TEMPERATURE` | (provider default) | Recommended **0** for reproducible extraction/interpretation. |
| `LLM_TOP_P` | (provider default) | Recommended **1** with `LLM_TEMPERATURE=0`. |
| `SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM` | false | If true, short reports without snippet hits still go to LLM (full capped text). |
| `SHORT_REPORT_CHAR_THRESHOLD` | 1000 | Max `original_report_text_length` for short-report fallback. |

## Pipeline stages
1. Prepare structured baseline (`src/pipeline/prepare_structured_data.py`)
2. Run text pipeline (`src/pipeline/run_pipeline.py`)
3. Compare predictions vs baseline (`src/pipeline/compare_reports_vs_baseline.py`)
4. Evaluate metrics/plots (`src/pipeline/evaluate_predictions.py`)
5. Validate input consistency (`src/validation/validate_inputs.py`)
6. Advanced exploration (`src/analysis/run_exploration.py`)
7. In-depth analysis (`src/analysis/run_analysis.py`)

## Data Sources (Current Default)
Centralized in `src/pipeline/paths.py`.

- `DATA_MODE = "real"` (default)
- **Final production raw inputs** (semicolon-separated under `data/raw/`):
  - `Berichte.csv` — primary text (`PatientID`, clinical fields → `report_text`)
  - `ICD.csv` — `PatientID; icd_hd; icd_code`
  - `ICDSC.csv` — `PatientID; ICDSC_Max` (patient-level maximum score)
- **No** `Diagnosenliste.csv` in the active pipeline (`LEGACY_DIAGNOSIS_INPUT_PATH` only for documentation).

Optional synthetic mode (`DATA_MODE = "synthetic"`):
  - `data/structured/raw/synthetic_icd10.csv`, `synthetic_icdsc.csv`, `synthetic_berichte.csv`
  - Legacy: `data/anonymized/beispiele/synthetic_diagnoses.csv` (INPUT_MODE=diagnosis only)

## Important Logic
- **Baseline construction** (`prepare_structured_data`): ICD + ICDSC only → `structured_baseline.csv`.
- **ICD-10 delirium** (`has_delir_icd10` / `baseline_icd10`): main diagnosis `icd_hd == 1` and codes **F05.0, F05.8, F05.9** only. **F05.1** (alcohol-related / Entzugsdelir) and other F05 subcodes are **excluded** from the intended cohort. Implementation: exact allowlist in `is_valid_delir_icd10_code()` — no `startswith("F05")` / no `INCLUDE_ALL_F05_PRESENTATION_MODE`.
- **ICDSC** (`max_icdsc`): from `ICDSC_Max`; thresholds `baseline_icdsc_ge_*`, `baseline_icdsc_0`, `baseline_icdsc_1_to_3`, `baseline_icdsc_ge_4_grouped`.
- **Primary validation baseline** `baseline_composite` — configurable in `src/pipeline/paths.py` as `BASELINE_COMPOSITE_MODE`:
  - **`OR`** (thesis): `(baseline_icdsc_ge_4 == 1) OR (baseline_icd10 == 1)` — broader/sensitive.
  - **`AND`** (temporary presentation): `(baseline_icdsc_ge_4 == 1) AND (baseline_icd10 == 1)` — high-confidence / «sichere Delirfälle».
  - Model-positive / AND-baseline-negative → interpret as **Delirkandidaten** (possible uncoded delir), not automatic false positives.
- **Legacy** multiclass `baseline_reference_class` may still be written; primary evaluation uses binary baselines including `baseline_composite`.
- **Deprecated:** `Diagnosenliste.csv` / `diagnosis_mapper` — not used in production. `Berichte.csv` columns `diag`, `epikrise`, `jetziges_leiden`, `prozedere` map to report sections `[Diagnosen]`, `[Epikrise]`, etc.
- **Exploration** (`run_exploration.py`): Berichte + ICD + ICDSC + structured baseline + predictions; no crash when legacy diagnosis path is absent.

## Single Source of Path Truth
- `src/pipeline/paths.py` is the central config.
- Do not hardcode paths elsewhere.
- New analysis/exploration output dirs are also defined there.

## Most Important Commands

### Full one-command run
```bash
./scripts/run_all.sh
```

### Preflight (recommended before sensitive runs)
```bash
./scripts/preflight_check.sh
```
### Pre-prompt (pilot / dev only — not thesis validation)

```bash
export MAX_REPORTS=60   # caps REPORT rows — do NOT use for final validation
export DEBUG_LLM_OUTPUT=false
export ENABLE_SQLITE_LOGGING=true
```

For the thesis validation path, use **[FINAL THESIS WORKFLOW (FULL RUN)](#final-thesis-workflow-full-run)** instead.

### Manual step-by-step (reference / partial runs)
```bash
python3 -m src.pipeline.prepare_structured_data
python3 -m src.pipeline.run_pipeline
python3 -m src.pipeline.compare_reports_vs_baseline
python3 -m src.pipeline.evaluate_predictions
python3 -m src.validation.validate_inputs
python3 -m src.analysis.run_exploration
python3 -m src.analysis.run_analysis
python3 -m src.analysis.run_validation_suite
python3 -m src.analysis.create_patient_reporttype_matrix
python3 -m src.analysis.export_patient_validation_cohort   # PRIMARY manual validation export
python3 -m src.analysis.export_manual_report_labels        # slim sheet for annotation
python3 -m src.analysis.freeze_validation_cohort         # lock 100-patient cohort (after full inference)
# Annotate manual_report_labels.csv (or frozen copy); then:
python3 -m src.analysis.evaluate_manual_validation         # prefers frozen_validation_cohort/ if present
python3 -m src.analysis.export_presentation_examples
```

## Validation architecture (PRIMARY)

| Unit | Definition |
|------|------------|
| Prediction | 1 report → 1 `model_report_prediction` (from `klasse`) |
| Validation cohort | 100 unique patients (`PATIENT_VALIDATION_N`); **all** evaluatable reports per patient (Berichte spine + predictions; includes prefilter-skipped rows) |
| Manual annotation | `manual_report_ground_truth` (0/1) per report |
| Patient manual GT | **Derived:** `derived_manual_patient_ground_truth` = max(report GT) per patient |
| ICDSC / ICD10 | **Reference signals only** — not absolute truth |

Legacy (deprecated): `export_manual_validation_sample`, `run_error_review_export` (`manual_label_0_1_2`).

---

## FINAL THESIS WORKFLOW (FULL RUN)

Copy-paste sequence for the **production thesis run**: full inference → validation cohort → freeze → manual annotation → evaluation.

Run all commands from the project root (`delirium_project/`). Ensure `data/raw/Berichte.csv`, `ICD.csv`, and `ICDSC.csv` are present and the USZ LLM service is reachable (see environment notes above).

### 1. Full dataset inference

For final thesis evaluation, the **full** report corpus must be processed **before** exporting the manual validation cohort.

**Why:** Validation is patient-level. Every selected patient must have a **complete report trajectory** (processed, prefilter-skipped, and guardrail-negative rows). The cohort export joins `Berichte.csv` with predictions; predictions must cover the full run first.

**Critical:** Do **not** set a numeric `MAX_REPORTS` for thesis validation. `MAX_REPORTS` limits **report rows**, not patients — a partial run under-represents patients and biases validation. Default in `paths.py` is already **no cap**; only `unset` if your shell still exports a limit from an earlier pilot.

```bash
cd delirium_project

unset MAX_REPORTS
# Optional explicit full run: export MAX_REPORTS=all

export DEBUG_LLM_OUTPUT=false
export ENABLE_SQLITE_LOGGING=true
export SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true
export SHORT_REPORT_CHAR_THRESHOLD=1000

# LLM (USZ) — adjust if your server differs
export LLM_PROVIDER=usz_api
export LLM_TEMPERATURE=0
export LLM_TOP_P=1

python3 -m src.pipeline.prepare_structured_data
python3 -m src.pipeline.run_pipeline
python3 -m src.analysis.run_validation_suite
```

| Step | Role |
|------|------|
| `prepare_structured_data` | Builds `structured_baseline.csv` (ICDSC + ICD10; F05.0/F05.8/F05.9 main diagnosis only). |
| `run_pipeline` | Report-level inference for **all** included Berichte rows (Dokumentationsblatt excluded). |
| `run_validation_suite` | Compare vs baseline, evaluation plots, patient matrix, **and** `export_patient_validation_cohort` (mutable copy). |

Optional after inference: `python3 -m src.validation.validate_inputs`

**Primary predictions file:** `outputs/predictions/agent1_agent2_agent3_results_prompt.csv`

### 2. Export validation cohort

Run **after** full inference. Re-export if predictions or baseline change.

```bash
export PATIENT_VALIDATION_N=100

python3 -m src.analysis.export_patient_validation_cohort
python3 -m src.analysis.export_manual_report_labels
```

| Output | Purpose |
|--------|---------|
| `outputs/analysis/manual_validation/patient_validation_cohort.csv` | Wide cohort (all reports for 100 patients; stable IDs). |
| `outputs/analysis/manual_validation/manual_report_labels.csv` | **Slim sheet for annotation** (copy before editing if not freezing immediately). |
| `outputs/analysis/manual_validation/patient_validation_cohort_report.txt` | Counts + processing summary (`status`, skipped vs LLM, etc.). |

**Included report types:** Verlaufseintrag, Verlegungsbericht, Austrittsbericht.

**Skipped reports are included** — prefilter skip (`no_evidence_prefilter_skip`) is still a model decision (`klasse=0`, `status=skipped`). Guardrail negatives after LLM are included too.

**Pilot (30 patients):** `export PATIENT_VALIDATION_N=30` with the same commands (only for dry runs, not thesis freeze).

### 3. Freeze validation dataset

Locks the cohort for manual review. **Do not re-freeze** after annotation starts unless you intentionally reset the study (requires `OVERWRITE_FROZEN_VALIDATION=true`).

```bash
python3 -m src.analysis.freeze_validation_cohort
```

**Fixed thesis dataset directory:**

`outputs/analysis/manual_validation/frozen_validation_cohort/`

| File | Role |
|------|------|
| `patient_validation_cohort_frozen.csv` | Full frozen cohort (reference). |
| `manual_report_labels_frozen.csv` | **Annotate this file** (or copy labels back into this path). |
| `frozen_cohort_metadata.json` | Timestamp, patient/report counts, checksums, source paths. |

### 4. Manual validation

**Primary annotation file:**

`outputs/analysis/manual_validation/frozen_validation_cohort/manual_report_labels_frozen.csv`

Annotate only:

- **`manual_report_ground_truth`** — required  
- `manual_comment` — optional  

**Allowed values:** `0` or `1`

| Value | Meaning |
|-------|---------|
| `1` | Clinically plausible delir documented in **this** report |
| `0` | No delir documented in **this** report |

**Do not** fill a manual patient-level label. Patient-level ground truth is derived automatically:

`derived_manual_patient_ground_truth = max(manual_report_ground_truth)` per `validation_patient_id`

Use columns `status`, `llm_called`, `skipped_reason` in the frozen full cohort for context (prefilter vs LLM-processed).

### 5. Final evaluation

```bash
python3 -m src.analysis.evaluate_manual_validation
```

Uses **frozen** cohort + frozen labels when `frozen_validation_cohort/` exists.

**Outputs:** `outputs/analysis/manual_validation/evaluation/` (metrics tables, confusion plots, `evaluation_report.txt`)

| Level | Comparison | Role |
|-------|------------|------|
| Report | `model_report_prediction` vs `manual_report_ground_truth` | Per-report agreement |
| **Patient (PRIMARY)** | `model_patient_positive` vs `derived_manual_patient_ground_truth` | **Main thesis evaluation** |
| Exploratory | ICDSC / ICD10 vs derived patient GT | Reference signals only — not absolute truth |

### 6. Important warnings

- **`MAX_REPORTS` limits reports, not patients** — never use it for the thesis validation run.
- **Never regenerate** the validation cohort after manual annotation has started (use the frozen copy).
- **Never mix** predictions from a new `run_pipeline` with an old frozen cohort — re-freeze only with a deliberate reset (`export OVERWRITE_FROZEN_VALIDATION=true`).
- **Always evaluate** using frozen files when they exist (`evaluate_manual_validation` prefers them automatically).
- **Skipped / prefilter-negative reports belong in the cohort** — excluding them inflates performance.
- **`Dokumentationsblatt` stays excluded** from processing and validation (raw CSV unchanged).
- **ICDSC / ICD10** in the cohort are reference signals, not manual ground truth.

---

## Validation outputs
- `outputs/analysis/patient_level/patient_reporttype_matrix.csv` — exploratory patient matrix.
- `outputs/analysis/manual_validation/patient_validation_cohort.csv` — **PRIMARY** export (`Patient_0001`, `Patient_0001_Report_0001`, …).
- `outputs/analysis/manual_validation/manual_report_labels.csv` — slim annotation sheet (merge by `validation_report_id` at evaluation).
- `outputs/analysis/manual_validation/frozen_validation_cohort/` — **fixed** cohort for thesis validation (`*_frozen.csv`, `frozen_cohort_metadata.json`). Re-freeze only with `OVERWRITE_FROZEN_VALIDATION=true`.
- `outputs/analysis/manual_validation/patient_validation_cohort_report.txt` — cohort summary + methodology.
- `outputs/analysis/manual_validation/evaluation/` — metrics after annotation (`evaluate_manual_validation`).
- Exploratory: `delir_probability_estimate` (0–100) in predictions CSV; not used for final `klasse`.
- `outputs/analysis/presentation_examples/` — CSV + Markdown examples for slides (excerpt → keywords → evidence → LLM → prediction).

## Output Structure
- `outputs/baseline/` → structured baseline tables
- `outputs/predictions/` → Agent 1/2/3 outputs
- `outputs/comparisons/` → merged prediction vs baseline
- `outputs/evaluation/` → multiclass metrics, confusion matrices, error exports, plots
- `outputs/validation/` → validation checks (`validation_results.csv`, `validation_summary.txt`)
- `outputs/analysis/`
  - `exploration/` → raw-data EDA tables/plots/report
  - `tables/`, `plots/`, `reports/` → in-depth analytical views

## Docker / Ubuntu Notes
- Dockerfile exists at `docker/Dockerfile`.
- Build:
```bash
docker build -f docker/Dockerfile -t delirium-pipeline .
```
- Example run:
```bash
docker run --rm -it \
  -v "$(pwd)/Data/Raw:/app/Data/Raw" \
  -v "$(pwd)/outputs:/app/outputs" \
  delirium-pipeline \
  python -m src.pipeline.prepare_structured_data
```

## Sensitive Data Safety
- `data/` and `Data/` are gitignored.
- Keep real ICU data only local/server-side.
- Avoid committing raw input files.

## Current Known Practical State
- Code and tests are passing in the current local setup.
- Placeholder `Data/Raw` files exist for dry-run wiring; replace with real files on Ubuntu.
- If ICD files are empty, pipeline runs but baseline/evaluation quality is limited (expected).

## Quick Troubleshooting
- No reports processed (`Anzahl Berichte: 0`):
  - Check `data/raw/Berichte.csv` exists and has rows with `PatientID`.
- Baseline empty or wrong joins:
  - Re-run `python -m src.pipeline.prepare_structured_data`.
  - Check `data/raw/ICD.csv` (`PatientID; icd_hd; icd_code`) and `data/raw/ICDSC.csv` (`PatientID; ICDSC_Max`).
- Format issues:
  - Reader supports `.csv`, `.xlsx`, `.xls` via `src/pipeline/tabular_io.py`.
