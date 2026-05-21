# Runbook — Delirium pipeline (production-oriented)

Assume project root: `delirium_project/`.

## Server setup

1. Python 3.9+ and virtualenv (recommended).
2. `pip install -r requirements.txt`
3. Place inputs under **`data/raw/`** (semicolon-separated; see `src/pipeline/paths.py`):
   - `Berichte.csv` — `PatientID` + clinical text fields
   - `ICD.csv` — `PatientID; icd_hd; icd_code`
   - `ICDSC.csv` — `PatientID; ICDSC_Max`
   - No `Diagnosenliste.csv` in the active pipeline (legacy only).
4. Start the **USZ LLM** HTTP service so `USZ_LLM_URL` is reachable (default `http://localhost:8100/generate`).

## Environment — primary (USZ)

```bash
export LLM_PROVIDER=usz_api
export USZ_LLM_URL=http://localhost:8100/generate
export LLM_MODEL_LABEL=gemma4_26b_usz
export LLM_TEMPERATURE=0
export LLM_TOP_P=1
export LLM_MAX_TOKENS=1000
export LLM_TIMEOUT=120
export LLM_LONG_INPUT_WARNING_CHARS=12000
```

## Evidence extraction + console verbosity

Rule-based extraction (`src/preprocessing/evidence_extraction.py`) runs **before** Agent 1. Optional tuning:

```bash
export EVIDENCE_MAX_SNIPPETS=12
export EVIDENCE_MAX_LLM_CHARS=8000
export EVIDENCE_WINDOW_SENTENCES=1
export EVIDENCE_MAX_SNIPPET_CHARS=400
```

Compact per-patient logging is default. For deep debugging:

```bash
export DEBUG_LLM_OUTPUT=true
```

## USZ API smoke test

```bash
python scripts/test_usz_llm_api.py
```

Expect HTTP 200 and JSON body with a `response` field. If the service is down, fix networking or URL before `run_pipeline`.

## Primary production run (USZ)

```bash
python -m src.pipeline.prepare_structured_data
python -m src.analysis.run_data_coverage_analysis
python -m src.pipeline.run_pipeline
python -m src.pipeline.compare_reports_vs_baseline
python -m src.pipeline.evaluate_predictions
python -m src.analysis.run_field_delirium_analysis
```

## Optional Ollama comparison run

```bash
export LLM_PROVIDER=ollama
export OLLAMA_URL=http://127.0.0.1:11500
export OLLAMA_MODEL=qwen2.5:7b
export OLLAMA_NUM_CTX=8192
python -m src.pipeline.run_pipeline
```

Downstream steps still read **`outputs/predictions/agent1_agent2_agent3_results_prompt.csv`** — re-run `compare_reports_vs_baseline` and `evaluate_predictions` after swapping providers.

## Sanity checks after a run

| Check | What to do |
|-------|------------|
| **klasse distribution** | Count values in `agent1_agent2_agent3_results_prompt.csv` (`klasse` column). |
| **signalstaerke** | Tabulate `signalstaerke` in the same file. |
| **Baseline positives** | In `report_vs_baseline_comparison.csv` or `structured_baseline.csv`, sum `baseline_icd10`, `baseline_icdsc_ge_4`, etc. |
| **Outputs** | Confirm canonical CSV + `agent_results_<provider>_<model_label>.csv` exist under `outputs/predictions/`. |
| **Binary evaluation** | Open `outputs/evaluation/binary_baselines/tables/binary_baseline_summary.csv`. |

## Troubleshooting

### `Berichte.csv` missing

`run_pipeline` with `INPUT_MODE=berichte` raises **`FileNotFoundError`** with the expected path. Either add the file under `data/raw/` or temporarily set `INPUT_MODE='diagnosis'` in `run_pipeline.py` (fallback — not primary production).

### USZ API unavailable

LLM calls fail; agents catch errors and may return empty extraction / failed JSON parsing (see `outputs/logs/llm_debug/`). Restore the API or switch to Ollama for debugging.

### All predictions `klasse=0`

Check `signalstaerke` and raw LLM JSON in debug logs; verify the USZ model returns valid JSON for agents 1–2. Coverage analysis (`run_data_coverage_analysis`) helps confirm text fields are non-empty.

### JSON parsing failures

Inspect `outputs/logs/llm_debug/*.json`. Prompts are unchanged by design; backend must return parseable JSON (or adjust prompts in a future change).

### Long input warning

Log line: `Long LLM input detected ...`. No truncation is applied. If outputs degrade, consider shorter reports or future chunking (not implemented yet).

## Manual validation (PRIMARY thesis evaluation)

**Canonical copy-paste workflow:** [HANDOVER_SUMMARY.md — FINAL THESIS WORKFLOW (FULL RUN)](HANDOVER_SUMMARY.md#final-thesis-workflow-full-run)

### Quick reference

1. **Full inference** — `unset MAX_REPORTS` (never cap reports for thesis validation). Then:

```bash
export DEBUG_LLM_OUTPUT=false
export ENABLE_SQLITE_LOGGING=true
export SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true
export SHORT_REPORT_CHAR_THRESHOLD=1000

python -m src.pipeline.prepare_structured_data
python -m src.pipeline.run_pipeline
python -m src.analysis.run_validation_suite
```

2. **Export cohort** (after full inference):

```bash
export PATIENT_VALIDATION_N=100
python -m src.analysis.export_patient_validation_cohort
python -m src.analysis.export_manual_report_labels
```

3. **Freeze** (fixed thesis dataset — do not regenerate after annotation starts):

```bash
python -m src.analysis.freeze_validation_cohort
```

Annotate: `outputs/analysis/manual_validation/frozen_validation_cohort/manual_report_labels_frozen.csv` — column `manual_report_ground_truth` only (`0` / `1`). Skipped prefilter reports are included.

4. **Evaluate** (prefers frozen files):

```bash
python -m src.analysis.evaluate_manual_validation
```

Outputs: `outputs/analysis/manual_validation/evaluation/`. **Patient-level** metrics (`model_patient_positive` vs `derived_manual_patient_ground_truth`) are the primary thesis evaluation.

**Warnings:** `MAX_REPORTS` limits reports not patients; never mix a new pipeline run with an old frozen cohort; Dokumentationsblatt excluded; ICDSC/ICD10 are reference only.

Legacy: `export_manual_validation_sample`, `run_error_review_export` (multiclass `manual_label_0_1_2`).

## Compile / tests (CI-style)

```bash
python -m compileall src scripts
python -m pytest tests -q
```
