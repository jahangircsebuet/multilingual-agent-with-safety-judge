#!/usr/bin/env bash
# Final long-context analysis: paired 8K<->32K and position comparisons,
# agent comparisons, transition matrices, category/language/resource-tier
# breakdowns, Wilson CIs, paired-bootstrap CIs, McNemar tests. Produces every
# figNN_*/table under results/agent_run_result_analysis/, the source of the
# paper's figures and tables.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

python -u "${PROJECT_ROOT}/src-new/10_agent_run_result_analysis.py" \
  --input-dir "${PROJECT_ROOT}/data/agent_run_for_analysis" \
  --output-dir "${PROJECT_ROOT}/results/agent_run_result_analysis" \
  --reference-guard aprielguard \
  --expected-cases 3190 \
  --expected-cases-by-guard \
    crest=1237 \
    guardreasoner=102 \
    wildguard=18 \
    xguard=2524 \
  --bootstrap 2000 \
  --risk-languages-per-group 20 \
  --seed 42
