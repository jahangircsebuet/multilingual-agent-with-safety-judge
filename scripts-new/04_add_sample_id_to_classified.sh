#!/usr/bin/env bash
# Add the shared sample_id (root_id + language + SHA256(root_id||language||prompt))
# to every JSONL file under data/classified/, preserving directory structure.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

python "${PROJECT_ROOT}/src-new/04_add_sample_id_to_classified.py" \
  --input-dir "${PROJECT_ROOT}/data/classified" \
  --output-dir "${PROJECT_ROOT}/data/sampledid_added_classified"
