#!/usr/bin/env bash
# Run both downstream agents (Llama-3.1-8B-Instruct, Qwen2.5-14B-Instruct)
# over every AprielGuard-filtered long-context file (the primary reference
# population, 3,190 admitted prompts x 2 lengths x 3 positions). Writes
# data/agent_run_with_long_context/.
#
# --max-input-tokens 34000 covers both the 8K and 32K conditions in one
# pass. On memory-constrained GPUs, lower --batch-size and/or add
# --load-in-4bit --bnb-4bit-quant-type nf4 --bnb-4bit-compute-dtype bfloat16
# --bnb-4bit-use-double-quant (this is how the original experiment was run,
# split across several smaller invocations to fit available GPU memory).
# --resume makes reruns safe to interrupt.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_SCRIPT="${PROJECT_ROOT}/src-new/08_run_agent_long_context.py"
INPUT_DIR="${PROJECT_ROOT}/data/filtered_long_context"
OUTPUT_DIR="${PROJECT_ROOT}/data/agent_run_with_long_context"

python -u "${PYTHON_SCRIPT}" \
  --input-dir "${INPUT_DIR}" \
  --classifier-model aprielguard \
  --output-dir "${OUTPUT_DIR}" \
  --model Qwen/Qwen2.5-14B-Instruct \
  --k 1 \
  --prompt-condition realistic_tool \
  --batch-size 2 \
  --max-input-tokens 34000 \
  --max-new-tokens 48 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --resume \
  --seed 42

python -u "${PYTHON_SCRIPT}" \
  --input-dir "${INPUT_DIR}" \
  --classifier-model aprielguard \
  --output-dir "${OUTPUT_DIR}" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --k 1 \
  --prompt-condition realistic_tool \
  --batch-size 2 \
  --max-input-tokens 34000 \
  --max-new-tokens 48 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --resume \
  --seed 42
