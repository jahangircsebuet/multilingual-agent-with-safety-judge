#!/usr/bin/env bash
# Runs the full pipeline end to end, in execution order. Each step is also
# runnable on its own (see the numbered scripts in this directory).
#
# Requires a CUDA GPU for steps 01, 08, 09 (guard classification and
# downstream agent inference). Set PROJECT_ROOT to override the repo root
# auto-detected below.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/01_classify_prompts_with_guards.sh"
"${SCRIPT_DIR}/02_flatten_and_normalize.sh"
"${SCRIPT_DIR}/03_guard_analysis.sh"
"${SCRIPT_DIR}/04_add_sample_id_to_classified.sh"
"${SCRIPT_DIR}/05_guard_sample_id_overlap.sh"
"${SCRIPT_DIR}/06_generate_long_context_instances.sh"
"${SCRIPT_DIR}/07_filter_long_context_data.sh"
"${SCRIPT_DIR}/08_run_agent_long_context.sh"
"${SCRIPT_DIR}/09_create_guard_agent_files_from_apriel.sh"
"${SCRIPT_DIR}/10_agent_run_result_analysis.sh"
