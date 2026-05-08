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

Fallback inputs (`INPUT_MODE` in `run_pipeline.py`): diagnosis list or TXT bundles — see code comments; production assumes **Berichte**.

---

## Structured baselines (`outputs/baseline/structured_baseline.csv`)

Produced by `prepare_structured_data` from **`data/raw/ICD.csv`** and **`data/raw/ICDSC.csv`** (paths from `paths.py`).

**Binary baseline columns** (all included in primary evaluation):

- `baseline_icdsc_ge_1` … `baseline_icdsc_ge_5`
- `baseline_icdsc_0`
- `baseline_icdsc_1_to_3`
- `baseline_icdsc_ge_4_grouped`
- `baseline_icd10`

**ICD-10 delirium definition:** codes **`F05.0`**, **`F05.8`**, **`F05.9`** only — **`F05.1` excluded**.

**Legacy:** `baseline_reference_class` (0/1/2) may still be written for backward compatibility; it is **not** the primary evaluation target. Use binary baselines and `evaluate_predictions`.

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
export LLM_TEMPERATURE=0.1
export LLM_TOP_P=0.9
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

---

## Command order (recommended)

From the project root (`delirium_project/`):

```bash
python -m src.pipeline.prepare_structured_data
python -m src.analysis.run_data_coverage_analysis
python -m src.pipeline.run_pipeline
python -m src.pipeline.compare_reports_vs_baseline
python -m src.pipeline.evaluate_predictions
python -m src.analysis.run_field_delirium_analysis
```

Optional: `python -m src.validation.validate_inputs`, `python -m src.analysis.run_exploration`, `python -m src.analysis.run_analysis`, `python -m src.analysis.run_false_negative_review`.

---

## Outputs (overview)

| Area | Location |
|------|-----------|
| Structured baseline | `outputs/baseline/structured_baseline.csv` |
| Predictions (canonical) | `outputs/predictions/agent1_agent2_agent3_results_prompt.csv` |
| Predictions (tagged copy) | `outputs/predictions/agent_results_<provider>_<model_label>.csv` |
| Report vs baseline merge | `outputs/comparisons/report_vs_baseline_comparison.csv` |
| Binary baseline evaluation | `outputs/evaluation/binary_baselines/` (tables, plots, `report.txt`) |
| Data coverage | `outputs/analysis/data_coverage/` |
| Field keyword / OR analysis | `outputs/analysis/field_delirium/` |
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
