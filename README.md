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
├── data/
├── src/
├── results/
├── figures/
└── README.md
```

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
