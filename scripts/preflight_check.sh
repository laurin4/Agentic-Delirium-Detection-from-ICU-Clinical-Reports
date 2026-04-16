#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/Ba_venv/bin/python"}"

DATA_DIR="$ROOT_DIR/Data/Raw"
DIAG_FILE="$DATA_DIR/Diagnosenliste.csv"
ICD_FILE="$DATA_DIR/ICD.csv"
ICDSC_FILE="$DATA_DIR/ICDSC.csv"

echo "=== Delirium Pipeline Preflight ==="
echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not executable at $PYTHON_BIN"
  echo "Hint: create/activate venv or set PYTHON_BIN explicitly."
  exit 1
fi

for file in "$DIAG_FILE" "$ICD_FILE" "$ICDSC_FILE"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: Missing required input file: $file"
    exit 1
  fi
done

echo "[1/6] Quick input preview"
"$PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path

files = [
    Path("$DIAG_FILE"),
    Path("$ICD_FILE"),
    Path("$ICDSC_FILE"),
]
for path in files:
    df = pd.read_csv(path)
    print(f"{path.name}: rows={len(df)}, cols={list(df.columns)}")
PY
echo

echo "[2/6] prepare_structured_data"
"$PYTHON_BIN" -m src.pipeline.prepare_structured_data
echo

echo "[3/6] run_pipeline"
"$PYTHON_BIN" -m src.pipeline.run_pipeline
echo

echo "[4/6] compare_reports_vs_baseline"
"$PYTHON_BIN" -m src.pipeline.compare_reports_vs_baseline
echo

echo "[5/6] evaluate_predictions"
"$PYTHON_BIN" -m src.pipeline.evaluate_predictions
echo

echo "[6/6] validate_inputs"
"$PYTHON_BIN" -m src.validation.validate_inputs
echo

echo "Preflight completed successfully."
echo "Check outputs under:"
echo "  - outputs/predictions/"
echo "  - outputs/comparisons/"
echo "  - outputs/evaluation/"
echo "  - outputs/validation/"
