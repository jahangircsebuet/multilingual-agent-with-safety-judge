#!/usr/bin/env bash
# Summarize sample_id coverage and admission overlap across the five guards.
# Produces the admitted/common-subset counts cited in the paper
# (e.g. 3,190 AprielGuard-admitted; 2,524/1,237/102/18 common with
# XGuard/CREST/GuardReasoner/WildGuard).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

python -u "${PROJECT_ROOT}/src-new/05_guard_sample_id_overlap.py" \
  --input-dir "${PROJECT_ROOT}/data/sampledid_added_classified" \
  --output-file "${PROJECT_ROOT}/results/guard_sample_id_overlap_summary.json"
