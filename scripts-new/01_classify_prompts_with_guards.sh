#!/usr/bin/env bash
# Classify the multilingual prompt pool with each of the five runtime guards
# used in the paper (aprielguard, crest, guardreasoner, wildguard, xguard),
# k=1. Writes data/classified/<guard>/<guard>_dataset_k=1.jsonl.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_SCRIPT="${PROJECT_ROOT}/src-new/01_classify_prompts_with_guards.py"

run_classifier () {
  local model_name="$1"
  local classifier="$2"
  local k="$3"
  shift 3

  local input_file="${PROJECT_ROOT}/data/stratified/stratified_topk_k${k}_with_prompts.jsonl"
  local output_dir="${PROJECT_ROOT}/data/classified/${model_name}"
  local output_file="${output_dir}/${model_name}_dataset_k=${k}.jsonl"

  mkdir -p "${output_dir}"

  echo "============================================================"
  echo "Running model=${model_name}, classifier=${classifier}, k=${k}"
  echo "Input : ${input_file}"
  echo "Output: ${output_file}"
  echo "============================================================"

  python "${PYTHON_SCRIPT}" \
    --input "${input_file}" \
    --output "${output_file}" \
    --classifier "${classifier}" \
    "$@"
}

run_classifier guardreasoner guardreasoner 1 \
  --device auto

run_classifier xguard xguard 1 \
  --device auto \
  --max_new_tokens 512

run_classifier aprielguard aprielguard 1 \
  --device auto \
  --torch_dtype auto \
  --max_new_tokens 512

run_classifier wildguard wildguard 1 \
  --device auto \
  --torch_dtype auto \
  --max_new_tokens 512

run_classifier crest crest 1 \
  --device auto \
  --torch_dtype auto
