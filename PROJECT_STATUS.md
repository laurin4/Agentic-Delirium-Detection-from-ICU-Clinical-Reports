# PROJECT STATUS

**Authoritative project description:** see **`README.md`**.

This file is kept short so documentation does not drift from the code.

## Snapshot

- **Primary report input:** `data/raw/Berichte.csv` (see `paths.py`, `INPUT_MODE` in `run_pipeline.py`).
- **Primary LLM:** USZ HTTP API (`LLM_PROVIDER` defaults to `usz_api` in `model_config.py`).
- **Optional comparison LLM:** Ollama (`LLM_PROVIDER=ollama`).
- **Model output:** binary `klasse` ∈ {0, 1}; interpretation signal strength `niedrig` | `mittel` | `hoch`.
- **Evaluation:** binary baselines under `outputs/evaluation/binary_baselines/` (`evaluate_predictions`).
- **Legacy:** `baseline_reference_class` (multiclass) may still exist in `structured_baseline.csv`; it is **not** the primary evaluation target.

For commands, environment variables, and baseline definitions, use **`README.md`** and **`RUNBOOK.md`**.
