#!/usr/bin/env bash
set -euo pipefail

# Build Linux-compatible wheels for offline Ubuntu installs.
# Default target matches x86_64 manylinux + Python 3.12.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${ROOT_DIR}/requirements.txt"
OUT_DIR="${ROOT_DIR}/wheelhouse_linux"

PYTHON_VERSION="${PYTHON_VERSION:-312}"
PLATFORM_TAG="${PLATFORM_TAG:-manylinux2014_x86_64}"
IMPLEMENTATION="${IMPLEMENTATION:-cp}"
ABI_TAG="${ABI_TAG:-cp${PYTHON_VERSION}}"

mkdir -p "${OUT_DIR}"

python -m pip download \
  --requirement "${REQ_FILE}" \
  --dest "${OUT_DIR}" \
  --only-binary=:all: \
  --platform "${PLATFORM_TAG}" \
  --implementation "${IMPLEMENTATION}" \
  --python-version "${PYTHON_VERSION}" \
  --abi "${ABI_TAG}"

echo "Linux wheelhouse refreshed at: ${OUT_DIR}"
