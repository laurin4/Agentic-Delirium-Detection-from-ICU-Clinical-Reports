#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/Ba_venv/bin/python"}"

echo "=== Full Delirium Pipeline Run ==="
echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not executable at $PYTHON_BIN"
  echo "Hint: create venv or set PYTHON_BIN explicitly."
  exit 1
fi

run_step() {
  local label="$1"
  shift
  echo "[RUN] $label"
  "$PYTHON_BIN" -m "$@"
  echo
}

run_step "Prepare structured baseline" src.pipeline.prepare_structured_data
run_step "Run Agent pipeline (predictions)" src.pipeline.run_pipeline
run_step "Compare predictions vs baseline" src.pipeline.compare_reports_vs_baseline
run_step "Evaluate metrics and plots" src.pipeline.evaluate_predictions
run_step "Validate data consistency" src.validation.validate_inputs
run_step "Advanced data exploration (raw inputs)" src.analysis.run_exploration
run_step "In-depth analysis and visualizations" src.analysis.run_analysis

echo "Full run completed successfully."
echo "Outputs:"
echo "  - outputs/predictions/"
echo "  - outputs/comparisons/"
echo "  - outputs/evaluation/"
echo "  - outputs/validation/"
echo "  - outputs/analysis/exploration/"
echo "  - outputs/analysis/"
