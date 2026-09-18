#!/usr/bin/env bash
# Consolidate the final agent-run population used by the analysis step:
#   1. Copy the AprielGuard agent-run output into
#      data/agent_run_for_analysis/aprielguard/ (the reference population).
#   2. For each other guard, reuse the AprielGuard agent inference for
#      sample_ids jointly admitted by AprielGuard and that guard, instead of
#      re-running the downstream models on identical inputs.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_SCRIPT="${PROJECT_ROOT}/src-new/09_create_guard_agent_files_from_apriel.py"
ANALYSIS_DIR="${PROJECT_ROOT}/data/agent_run_for_analysis"

mkdir -p "${ANALYSIS_DIR}/aprielguard"
find "${PROJECT_ROOT}/data/agent_run_with_long_context" -maxdepth 1 -name 'longctx_conversation_*_aprielguard_k=1_*.jsonl' \
  -exec cp {} "${ANALYSIS_DIR}/aprielguard/" \;

for guard in crest guardreasoner wildguard xguard; do
  python -u "${PYTHON_SCRIPT}" \
    --input-dir "${ANALYSIS_DIR}" \
    --output-dir "${ANALYSIS_DIR}" \
    --classified-root "${PROJECT_ROOT}/data/sampledid_added_classified" \
    --guard-model "${guard}"
done
