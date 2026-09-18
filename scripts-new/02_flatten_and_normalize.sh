#!/usr/bin/env bash
# Merge all per-guard classified JSONL files and normalize the nested
# classifier object into one analysis-ready schema.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${PROJECT_ROOT}/results"

python "${PROJECT_ROOT}/src-new/02_flatten_and_normalize.py" \
  --inputs "${PROJECT_ROOT}/data/classified/**/*.jsonl" \
  --output "${PROJECT_ROOT}/results/all_guard_predictions.jsonl"
