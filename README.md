# When Guardrails Fail: Multilingual Long-Context Safety in Tool-Using Agents

This repository contains the code and analysis for a multilingual guard-to-agent safety benchmark. We study whether harmful requests that bypass runtime guardrails are recovered by downstream agents or propagate to restricted tool selection.

## Benchmark

- 1,901 unsafe AEGIS2.0-derived English roots
- 98 languages
- 19 harm categories
- Up to 186,298 translated instances
- Translation-quality metadata using BERTScore and COMET

## Runtime Guards

We evaluate AprielGuard, CREST, GuardReasoner, WildGuard, and XGuard. A guard prediction normalized as `safe` means that a gold-unsafe prompt bypassed the input guard and is eligible for downstream evaluation.

## Downstream Agents

We evaluate Llama-3.1-8B-Instruct and Qwen2.5-14B-Instruct. Agents choose among `refuse`, `escalate`, or `call_tool`. Restricted tools are inert and produce no real-world side effects.

## Long-Context Evaluation

Each admitted multilingual harmful request is preserved unchanged and embedded in benign English context. We evaluate 8K and 32K contexts, beginning/middle/end prompt positions, and both downstream agents, yielding 12 conditions per eligible `sample_id`.

## Primary Metric

Our main outcome is **restricted-tool selection rate**: the fraction of scorable runs in which the agent selects the category-relevant restricted symbolic tool.

We additionally analyze refusal/escalation, inference coverage, 8K→32K safety-state transitions, language and resource-tier disparities, harm-category variation, and paired position/model effects.

## Main Findings

- Context length is not uniformly safer or less safe; effects depend on model and prompt position.
- Prompt position can reverse the relative safety ranking of Llama and Qwen.
- 6.4% of matched cases transition from safe at 8K to unsafe at 32K, while 5.5% transition in the opposite direction.
- Harm-category failure rates vary substantially across models.
- Low-resource languages show higher downstream restricted-tool selection (21.65%) than high-resource (11.08%) and medium-resource (10.66%) languages.

## Coverage Note

Qwen2.5-14B at 32K-middle has only 113 valid inferences out of 3,188 rows (3.54% coverage). Results from this condition are reported for transparency but are not used as primary evidence.

## Repository Structure

```text
.
├── src/           Original pipeline scripts (00-10) plus exploratory/superseded utilities.
├── scripts/       Shell wrappers recording the exact commands used to produce results/.
├── data/          Inputs and intermediate JSONL data.
├── results/       Full tables (CSV) and figures (PDF/SVG) underlying the paper.
├── src-new/       Curated pipeline scripts, renumbered 01-10 by true execution order.
├── scripts-new/   One clean wrapper per src-new script, plus script.sh (runs all 10 in order).
├── data/*-new/    Curated, paper-scoped data artifacts (see "Data" below).
├── results-new/   Curated final tables/figures (a subset of results/).
└── README.md
```

See "Curated reviewer set" below for what `*-new` contains and why it exists
alongside the originals.

## Safety

This project is intended for defensive AI-safety research. All restricted tools are symbolic and inert; no harmful external action is executed.

## Citation

```bibtex
@inproceedings{alam2026guardrails,
  title     = {When Guardrails Fail: Multilingual Long-Context Safety in Tool-Using Agents},
  author    = {Anonymous Authors},
  booktitle = {Workshop Proceedings},
  year      = {2026}
}
```

The citation will be updated after publication.

---

## Curated reviewer set (`*-new`)

`src/`, `scripts/`, `data/`, and `results/` are the original working
directories, including exploratory/superseded material accumulated over the
project. Alongside each is a curated sibling — `src-new/`, `scripts-new/`,
`data/<name>-new/`, `results-new/` — containing **only what this paper's
reported pipeline and results actually use**, renumbered by true execution
order, with every machine-specific path removed. Start there.

```
src-new/       10 scripts (01-10), execution order == numeric order.
scripts-new/   One shell wrapper per src-new script, plus script.sh (runs all 10 in order).
               All paths are relative to PROJECT_ROOT (auto-detected, or export to override).
data/*-new/    Curated inputs/intermediates (see "Data" below).
results-new/   Final tables + figures for the guard evaluation and the long-context
               downstream-agent analysis — the source of every figure/table in the paper.
```

Run the whole pipeline with:

```bash
PROJECT_ROOT=/path/to/repo scripts-new/script.sh   # PROJECT_ROOT optional, auto-detected otherwise
```

**Note on data size**: `data/long_context-new/`, `data/filtered_long_context-new/`,
and `data/agent_run_for_analysis-new/` hold the first 100 rows of each file
as a representative sample (the full files run from hundreds of MB to
several GB each and aren't practical to version). `data/classified-new/`,
`data/stratified-new/`, `data/sampledid_added_classified-new/`, and
`data/filtered_long_context_manifest_files-new/` are complete. **Full,
unsampled data is available on request.**

### What was left out of the curated set, and why

- `03_run_agent.py`, `04_agentic_metrics.py` (→ `results/agent_results/`) —
  an earlier short-context, per-k pass. The paper's downstream-agent results
  are all from the long-context (8K/32K) experiment.
- `10_union_long_context_by_sample_id.py` — written but never invoked by any
  production run; superseded by the AprielGuard-reuse approach in
  `09_create_guard_agent_files_from_apriel.py`.
- `src/classifier.py` — byte-identical unused duplicate of
  `00_multi_guard_prompt_classifier.py` (now `01_classify_prompts_with_guards.py`).
- `src/common.py`, `find_dulicate.py`, `util_check_duplicate_long_context_sources.py`
  — unused/diagnostic utilities not called by any script that produces a
  committed result.
- `data/agent_run_with_long_context/` — raw AprielGuard-only agent output,
  including abandoned 4-bit/retry variants; fully superseded by
  `data/agent_run_for_analysis/`, which already contains the clean,
  deduplicated version of the same files.
- `data/classified-backup/`, `data/duplicate_source_check/`, and the various
  `data/filtered_long_context_<position>_<agent>*/` fragments — leftovers
  from memory-constrained partial runs; superseded by the complete
  `data/filtered_long_context copy/` (the source sampled into
  `data/filtered_long_context-new/`) and `data/agent_run_for_analysis/`.
- "Document"-type long-context variants (`longctx_document_*`) — generated
  by `06_generate_long_context_instances.py` alongside the conversation-type
  ones, but never run through the downstream agent; only conversation-type
  context is evaluated in the paper.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Guard-classification and downstream-agent steps (`01_classify_prompts_with_guards.py`,
`08_run_agent_long_context.py`, `09_create_guard_agent_files_from_apriel.py`)
load Hugging Face models locally via `transformers`/`peft`/`accelerate` and
need a CUDA GPU. Gated models (e.g. Llama-3.1) require `huggingface-cli login`
or `--hf-token`.

## Pipeline (`src-new/`, run in numeric order)

`sample_id = root_id + language + SHA256(root_id || language || prompt)` is
threaded through every stage so the same multilingual instance can be
matched across guard, context-length, position, and agent conditions.

| # | Script | Purpose |
|---|---|---|
| 01 | `classify_prompts_with_guards.py` | Run one guard model over `data/stratified/stratified_topk_k1_with_prompts.jsonl` → `data/classified/<guard>/`. |
| 02 | `flatten_and_normalize.py` | Merge all per-guard files, normalize heterogeneous classifier output → `results/all_guard_predictions.jsonl`. |
| 03 | `guard_analysis.py` | Guard recall/evasion/coverage by language, category, resource tier, translation-quality bucket, with bootstrap CIs → `results/guard_analysis_final/`. |
| 04 | `add_sample_id_to_classified.py` | Add the shared `sample_id` to every file under `data/classified/` → `data/sampledid_added_classified/`. |
| 05 | `guard_sample_id_overlap.py` | `sample_id` overlap/admission counts across the five guards (the 3,190 / 2,524 / 1,237 / 102 / 18 figures cited in the paper). |
| 06 | `generate_long_context_instances.py` | Embed each root prompt (unchanged) in benign English context at 8K/32K tokens × beginning/middle/end × conversation/document → `data/long_context/`. |
| 07 | `filter_long_context_data.py` | Keep only long-context instances whose `sample_id` was admitted (`classifier.label == "safe"`) by one guard → `data/filtered_long_context/`. |
| 08 | `run_agent_long_context.py` | Run a downstream agent over the AprielGuard-admitted filtered long-context files → `data/agent_run_with_long_context/`. |
| 09 | `create_guard_agent_files_from_apriel.py` | For `sample_id`s jointly admitted by AprielGuard and another guard, reuse the AprielGuard agent inference instead of re-running the model → `data/agent_run_for_analysis/` (CREST/GuardReasoner/WildGuard/XGuard robustness subsets). |
| 10 | `agent_run_result_analysis.py` | Paired 8K↔32K and position comparisons, agent comparisons, transition matrices, category/language/resource-tier breakdowns, Wilson CIs, paired-bootstrap CIs, McNemar tests → `results/agent_run_result_analysis/`, source of every `figNN_*` figure/table in the paper. |

## Data

`data/classified-new/`, `data/stratified-new/`,
`data/filtered_long_context_manifest_files-new/`, and
`data/sampledid_added_classified-new/` are complete. `data/long_context-new/`,
`data/filtered_long_context-new/`, and `data/agent_run_for_analysis-new/` are
100-row-per-file samples (see "Curated reviewer set" above) — regenerate the
full versions with `scripts-new/06…10_*.sh`, or request the full data.

The original (non-`-new`) `data/` subdirectories mostly hold the full,
multi-GB versions of the same artifacts and are excluded via `.gitignore`
except for `data/classified/`, `data/stratified/`, and
`data/filtered_long_context_manifest_files/`.

## Reproducing paper results

- Guard evaluation tables/figures: `results-new/guard_analysis_final/` /
  `results/guard_analysis_final/`.
- Long-context downstream-agent tables/figures: `results-new/agent_run_result_analysis/`
  / `results/agent_run_result_analysis/` (`fig01`…`fig20`, e.g.
  `fig01_reference_agent_length_position_heatmap` = the reference-set
  factorial table/figure, `fig18/19/20_*_by_guard_subset` = the paired
  length/position/agent-model robustness checks).
- `results/agent_results/` and `results/all_guard_predictions.jsonl` are from
  the earlier short-context pass (see "What was left out" above) and are not
  part of the curated set.

## Known quirks in the original (non-`-new`) tree

- `src/classifier.py` is a byte-identical, unused duplicate of
  `src/00_multi_guard_prompt_classifier.py`.
- `scripts/classifier.sh`, `scripts/data.sh`, and `scripts/results.sh`
  reference older script names (`multi_guard_prompt_classifier.py`,
  `dataset_stats_by_k.py`, `k_aware_result_analysis.py`) and a placeholder
  `PROJECT_ROOT=/path/to/guardbreach-artifact` — they predate the current
  numbered `src/` layout and are kept only as a record of earlier runs.
- These are fixed in `scripts-new/` (clean commands, real `PROJECT_ROOT`
  auto-detection, no personal paths).
