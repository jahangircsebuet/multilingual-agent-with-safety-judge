#!/usr/bin/env bash
# For each guard, keep only the long-context instances whose sample_id was
# admitted (classifier.label == "safe") by that guard. Writes into a shared
# data/filtered_long_context/ directory, one file per
# (context type, length, position, guard).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

for guard in aprielguard crest guardreasoner wildguard xguard; do
  python "${PROJECT_ROOT}/src-new/07_filter_long_context_data.py" \
    --classified-dir "${PROJECT_ROOT}/data/sampledid_added_classified" \
    --long-context-dir "${PROJECT_ROOT}/data/long_context" \
    --model "${guard}" \
    --k 1
done
