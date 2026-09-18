#!/usr/bin/env bash
# Compute k-aware guard detection, evasion, coverage, and subgroup results
# with paper-quality bootstrap repetitions.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

python -u "${PROJECT_ROOT}/src-new/03_guard_analysis.py" \
  --input "${PROJECT_ROOT}/results/all_guard_predictions.jsonl" \
  --outdir "${PROJECT_ROOT}/results/guard_analysis_final" \
  --bootstrap 1000 \
  --seed 42
