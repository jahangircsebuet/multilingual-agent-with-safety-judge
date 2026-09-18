#!/usr/bin/env bash
# Generate the 12 long-context variants (2 context types x 2 lengths x 3
# positions) per multilingual root prompt. Only the 6 "conversation"-type
# variants are used downstream; the 6 "document"-type variants are generated
# but not evaluated in this paper.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

python "${PROJECT_ROOT}/src-new/06_generate_long_context_instances.py" \
  --input "${PROJECT_ROOT}/data/stratified/stratified_topk_k1_with_prompts.jsonl" \
  --output-dir "${PROJECT_ROOT}/data/long_context" \
  --prompt-field prompt \
  --root-id-field root_id \
  --language-field language \
  --reference-tokenizer Qwen/Qwen2.5-14B-Instruct
