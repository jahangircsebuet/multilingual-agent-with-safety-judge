#!/usr/bin/env python3
"""
10_agent_run_result_analysis.py

Publication-oriented analysis for long-context downstream-agent results stored
under guard-specific subdirectories.

Expected input layout
---------------------
<INPUT_DIR>/
    aprielguard/
        longctx_conversation_8k_beginning_aprielguard_k=1_llama31_8b.jsonl
        ... 12 files ...
    crest/
        ... 12 files ...
    guardreasoner/
        ... 12 files ...
    wildguard/
        ... 12 files ...
    xguard/
        ... 12 files ...

Important interpretation
------------------------
The target-guard directories (crest, guardreasoner, wildguard, xguard) may
contain downstream inference REUSED from AprielGuard for sample_ids jointly
admitted by AprielGuard and that target guard. Therefore:

* guard_subset identifies the population represented by a subdirectory;
* rows from different guard_subset values are NOT assumed to be independent;
* differences between guard subsets are descriptive population-composition
  differences, not causal effects of changing the input guard;
* the main agent/context analyses use --reference-guard (default aprielguard)
  to avoid repeatedly weighting the same reused inference;
* guard-subset tables/figures are provided as robustness/sensitivity analyses.

Primary outcome
---------------
restricted_tool_selection among scorable rows:
    successful inference + recognized decision.
Inference and parse failures are reported separately and are never counted safe.

Global uniqueness key
---------------------
One row is identified by:
    guard_subset
    + sample_id
    + context_length_name
    + unsafe_prompt_position
    + agent_model

The guard dimension MUST be part of the key now that multiple subdirectories
are analyzed. Otherwise the copied target-guard rows would be incorrectly
collapsed with AprielGuard rows.

Outputs
-------
tables/  : 30+ publication/diagnostic CSV files
figures/ : 15+ PDF and SVG figures
analysis_manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_DECISIONS = {"refuse", "escalate", "call_tool"}
LENGTH_ORDER = ["8k", "32k"]
POSITION_ORDER = ["beginning", "middle", "end"]
AGENT_ORDER = ["Llama-3.1-8B", "Qwen2.5-14B"]

KEY_FIELDS = [
    "guard_subset",
    "sample_id",
    "context_length_name",
    "unsafe_prompt_position",
    "agent_model",
]

FILE_RE = re.compile(
    r"^longctx_"
    r"(conversation|document)_"
    r"(8k|32k)_"
    r"(beginning|middle|end)_"
    r"(.+?)_"
    r"k=(\d+)_"
    r"(.+)\.jsonl$"
)

COLORS = {
    "navy": "#334E68",
    "ochre": "#C49A6C",
    "slate": "#6B7C85",
    "sage": "#7A9E7E",
    "brick": "#B36A5E",
    "purple": "#8B7E9F",
    "gray": "#A7B0B5",
    "light_gray": "#E0E0E0",
    "dark": "#263238",
}
PALETTE = [
    COLORS["navy"], COLORS["ochre"], COLORS["sage"],
    COLORS["brick"], COLORS["purple"], COLORS["slate"],
]
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "muted_risk",
    ["#F7F8F8", "#D8E0E4", "#9AAAB2", "#B78B77", "#8C4F45"],
)


# =============================================================================
# STYLE
# =============================================================================

def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.edgecolor": "#B8C0C4",
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C0C4")
    ax.spines["bottom"].set_color("#B8C0C4")
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COLORS["light_gray"], linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# HELPERS
# =============================================================================

def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "none", "null", "nan"}


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def to_float(value: Any) -> float:
    try:
        if is_blank(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_pct(n: float, d: float) -> float:
    return math.nan if d == 0 else 100.0 * n / d


def canonical_agent(agent_model: str, agent_tag: str) -> str:
    text = f"{agent_model} {agent_tag}".lower()
    if "qwen" in text:
        return "Qwen2.5-14B"
    if "llama" in text:
        return "Llama-3.1-8B"
    return agent_model or agent_tag or "unknown"


def ordered_condition(length: str, position: str) -> str:
    return f"{str(length).upper()}-{str(position).title()}"


def guard_display(value: str) -> str:
    mapping = {
        "aprielguard": "AprielGuard",
        "crest": "CREST",
        "guardreasoner": "GuardReasoner",
        "wildguard": "WildGuard",
        "xguard": "XGuard",
    }
    return mapping.get(str(value).lower(), str(value))


def parse_expected_cases_by_guard(values: list[str] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --expected-cases-by-guard entry: {item!r}. Use guard=count, e.g. crest=1237"
            )
        guard, count = item.split("=", 1)
        guard = guard.strip().lower()
        count_int = int(count.strip())
        if count_int < 0:
            raise ValueError("Expected case counts must be >= 0")
        result[guard] = count_int
    return result


# =============================================================================
# FILE METADATA / KEY
# =============================================================================

def parse_result_filename(path: Path) -> dict[str, Any] | None:
    m = FILE_RE.match(path.name)
    if not m:
        return None
    return {
        "file_context_type": m.group(1),
        "file_context_length": m.group(2),
        "file_prompt_position": m.group(3),
        "file_guard": m.group(4).lower(),
        "file_k": int(m.group(5)),
        "file_agent_tag": m.group(6),
    }


def infer_guard_subset(path: Path, input_dir: Path, file_guard: str) -> str:
    rel = path.relative_to(input_dir)
    # Preferred structure: input_dir/<guard>/file.jsonl
    if len(rel.parts) >= 2:
        return str(rel.parts[0]).strip().lower()
    # Backward compatibility for files directly in input_dir.
    return str(file_guard).strip().lower()


def make_composite_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("guard_subset", "")).strip().lower(),
        str(record.get("sample_id", "")).strip(),
        str(record.get("context_length_name", "")).strip().lower(),
        str(record.get("unsafe_prompt_position", "")).strip().lower(),
        str(record.get("agent_model", "")).strip(),
    )


def composite_key_text(key: tuple[str, ...]) -> str:
    return "||".join(key)


def composite_key_complete(key: tuple[str, ...]) -> bool:
    return all(bool(str(x).strip()) for x in key)


# =============================================================================
# STATISTICS
# =============================================================================

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return 100 * max(0.0, center - half), 100 * min(1.0, center + half)


def mcnemar_exact_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    if binomtest is None:
        return math.nan
    return float(binomtest(min(n01, n10), n=n, p=0.5, alternative="two-sided").pvalue)


def bootstrap_paired_delta(
    a: np.ndarray,
    b: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(a) == 0:
        return math.nan, math.nan, math.nan
    d = b.astype(float) - a.astype(float)
    point = 100.0 * float(d.mean())
    if iterations <= 0 or len(d) < 2:
        return point, math.nan, math.nan
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    left = iterations
    while left > 0:
        m = min(200, left)
        idx = rng.integers(0, len(d), size=(m, len(d)))
        chunks.append(d[idx].mean(axis=1))
        left -= m
    vals = 100.0 * np.concatenate(chunks)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =============================================================================
# ROW NORMALIZATION
# =============================================================================

def compact_result(
    row: dict[str, Any],
    source_relative_path: str,
    file_info: dict[str, Any],
    guard_subset: str,
    line_number: int,
    file_index: int,
    extra_metadata_fields: list[str],
) -> dict[str, Any]:
    inference_error = "" if is_blank(row.get("inference_error")) else str(row.get("inference_error"))
    valid_inference = inference_error == ""
    decision = str(row.get("agent_decision", "unknown")).strip().lower() or "unknown"
    decision_valid = valid_inference and decision in VALID_DECISIONS
    parse_error = "" if is_blank(row.get("decision_parse_error")) else str(row.get("decision_parse_error"))
    parse_failure = valid_inference and (not decision_valid or parse_error != "")

    tool_valid = boolish(row.get("tool_name_valid"))
    restricted_tool = boolish(row.get("restricted_tool_selection"))
    unsafe_attempt = boolish(row.get("unsafe_action_attempt"))
    safe_recovery = decision_valid and decision in {"refuse", "escalate"}
    invalid_tool_call = valid_inference and decision == "call_tool" and not tool_valid

    batch_seconds = to_float(row.get("batch_seconds"))
    batch_size = to_float(row.get("batch_size_actual"))
    seconds_per_case = (
        batch_seconds / batch_size
        if not math.isnan(batch_seconds) and not math.isnan(batch_size) and batch_size > 0
        else math.nan
    )

    agent_tag = str(row.get("agent_tag", file_info["file_agent_tag"])).strip()
    agent_model = str(row.get("agent_model", agent_tag)).strip()
    agent_name = canonical_agent(agent_model, agent_tag)

    context_type = str(row.get("context_type", file_info["file_context_type"])).strip().lower()
    context_length = str(row.get("context_length_name", file_info["file_context_length"])).strip().lower()
    prompt_position = str(row.get("unsafe_prompt_position", file_info["file_prompt_position"])).strip().lower()
    input_guard = str(row.get("input_guard", file_info["file_guard"])).strip().lower()

    quantization = str(row.get("agent_quantization", "none"))
    quant_type = str(row.get("bnb_4bit_quant_type", ""))
    compute_dtype = str(row.get("bnb_4bit_compute_dtype", ""))
    double_quant = boolish(row.get("bnb_4bit_use_double_quant", False))
    agent_config = " | ".join(
        [x for x in [agent_name, quantization, quant_type, compute_dtype, "double_quant" if double_quant else ""] if x]
    )

    result: dict[str, Any] = {
        "_source_file": source_relative_path,
        "_line_number": line_number,
        "_file_index": file_index,
        "guard_subset": guard_subset,
        "guard_subset_display": guard_display(guard_subset),
        "file_guard": file_info["file_guard"],
        "input_guard": input_guard,
        "guard_metadata_matches_path": (
            guard_subset == file_info["file_guard"] and input_guard == file_info["file_guard"]
        ),
        "agent_inference_reused": boolish(row.get("agent_inference_reused", False)),
        "agent_inference_reused_from_guard": str(row.get("agent_inference_reused_from_guard", "")),
        "agent_inference_reused_for_guard": str(row.get("agent_inference_reused_for_guard", "")),
        "agent_inference_source_file": str(row.get("agent_inference_source_file", "")),
        "long_context_id": str(row.get("long_context_id", "")).strip(),
        "sample_id": str(row.get("sample_id", "")).strip(),
        "root_id": str(row.get("root_id", "")).strip(),
        "language": str(row.get("language", "unknown")),
        "category": str(row.get("category", "unknown")),
        "tier": str(row.get("tier", "unknown")),
        "quality_bucket": str(row.get("quality_bucket", "unknown")),
        "f1": to_float(row.get("f1")),
        "comet": to_float(row.get("comet")),
        "combined_score": to_float(row.get("combined_score")),
        "selection_score": to_float(row.get("selection_score")),
        "prompt_len": to_float(row.get("prompt_len")),
        "context_type": context_type,
        "context_length_name": context_length,
        "unsafe_prompt_position": prompt_position,
        "unsafe_prompt_position_fraction": to_float(row.get("unsafe_prompt_position_fraction")),
        "context_actual_tokens": to_float(row.get("context_actual_tokens")),
        "rendered_agent_input_tokens": to_float(row.get("rendered_agent_input_tokens")),
        "agent_model": agent_model,
        "agent_tag": agent_tag,
        "agent_name": agent_name,
        "agent_config": agent_config,
        "agent_quantization": quantization,
        "prompt_condition": str(row.get("prompt_condition", "realistic_tool")),
        "input_guard_allows": boolish(row.get("input_guard_allows", True)),
        "agent_called": boolish(row.get("agent_called", True)),
        "agent_decision": decision,
        "valid_inference": valid_inference,
        "decision_valid": decision_valid,
        "parse_failure": parse_failure,
        "safe_recovery": safe_recovery,
        "tool_name_valid": tool_valid,
        "invalid_tool_call": invalid_tool_call,
        "restricted_tool_selection": restricted_tool,
        "unsafe_action_attempt": unsafe_attempt,
        "attempted_tool": "" if is_blank(row.get("attempted_tool")) else str(row.get("attempted_tool")),
        "offered_inert_tool": "" if is_blank(row.get("offered_inert_tool")) else str(row.get("offered_inert_tool")),
        "decision_parse_source": str(row.get("decision_parse_source", "")),
        "decision_parse_error": parse_error,
        "reason_code": "" if is_blank(row.get("reason_code")) else str(row.get("reason_code")),
        "inference_error": inference_error,
        "batch_seconds": batch_seconds,
        "batch_size_actual": batch_size,
        "approx_seconds_per_case": seconds_per_case,
        "k_subset": int(row.get("k_subset", file_info["file_k"])),
    }

    for field in extra_metadata_fields:
        result[field] = row.get(field, "unknown")

    key = make_composite_key(result)
    result["analysis_key"] = composite_key_text(key)
    result["condition"] = ordered_condition(context_length, prompt_position)
    return result


def record_priority(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(record["valid_inference"]),
        int(record["decision_valid"]),
        int(not record["parse_failure"]),
        int(record["_file_index"]),
        int(record["_line_number"]),
    )


# =============================================================================
# RECURSIVE LOADER
# =============================================================================

def load_all_results(
    input_dir: Path,
    expected_reference_cases: int | None,
    expected_cases_by_guard: dict[str, int],
    reference_guard: str,
    extra_metadata_fields: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(input_dir.rglob("longctx_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No longctx_*.jsonl files found recursively under {input_dir}")

    chosen_global: dict[tuple[str, ...], dict[str, Any]] = {}
    occurrences: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    missing_key_rows: list[dict[str, Any]] = []
    raw_valid_json_rows = 0

    for file_index, path in enumerate(files):
        file_info = parse_result_filename(path)
        if file_info is None:
            print(f"[SKIP] unrecognized filename: {path.relative_to(input_dir)}")
            continue

        guard_subset = infer_guard_subset(path, input_dir, file_info["file_guard"])
        relative_path = str(path.relative_to(input_dir))
        print(f"[READ] {relative_path}")

        raw_nonblank = 0
        bad = 0
        missing_key = 0
        file_keys: Counter[tuple[str, ...]] = Counter()
        file_valid_inference = 0
        file_scorable = 0
        file_sample_ids: set[str] = set()
        reuse_rows = 0
        metadata_mismatch_rows = 0

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                raw_nonblank += 1

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    bad += 1
                    malformed.append({
                        "source_file": relative_path,
                        "guard_subset": guard_subset,
                        "line_number": line_no,
                        "error": str(exc),
                    })
                    continue

                if not isinstance(obj, dict):
                    bad += 1
                    malformed.append({
                        "source_file": relative_path,
                        "guard_subset": guard_subset,
                        "line_number": line_no,
                        "error": "not_json_object",
                    })
                    continue

                raw_valid_json_rows += 1
                rec = compact_result(
                    obj,
                    relative_path,
                    file_info,
                    guard_subset,
                    line_no,
                    file_index,
                    extra_metadata_fields,
                )

                key = make_composite_key(rec)
                if not composite_key_complete(key):
                    missing_key += 1
                    missing_key_rows.append({
                        "source_file": relative_path,
                        "guard_subset": guard_subset,
                        "line_number": line_no,
                        **{field: rec.get(field, "") for field in KEY_FIELDS},
                    })
                    continue

                file_keys[key] += 1
                occurrences[key].append(rec)
                if rec["valid_inference"]:
                    file_valid_inference += 1
                if rec["decision_valid"]:
                    file_scorable += 1
                if rec["sample_id"]:
                    file_sample_ids.add(rec["sample_id"])
                if rec["agent_inference_reused"]:
                    reuse_rows += 1
                if not rec["guard_metadata_matches_path"]:
                    metadata_mismatch_rows += 1

                if key not in chosen_global or record_priority(rec) >= record_priority(chosen_global[key]):
                    chosen_global[key] = rec

        unique_file_keys = len(file_keys)
        duplicates_within_file = sum(max(0, c - 1) for c in file_keys.values())

        expected = expected_cases_by_guard.get(guard_subset)
        if expected is None and guard_subset == reference_guard:
            expected = expected_reference_cases
        completion = safe_pct(unique_file_keys, expected) if expected else math.nan

        inventory.append({
            "source_file": relative_path,
            "guard_subset": guard_subset,
            "guard_subset_display": guard_display(guard_subset),
            "file_guard": file_info["file_guard"],
            "context_type": file_info["file_context_type"],
            "context_length": file_info["file_context_length"],
            "prompt_position": file_info["file_prompt_position"],
            "agent_tag": file_info["file_agent_tag"],
            "raw_nonblank_rows": raw_nonblank,
            "malformed_rows": bad,
            "missing_key_rows": missing_key,
            "unique_composite_keys_in_file": unique_file_keys,
            "duplicate_key_rows_within_file": duplicates_within_file,
            "unique_sample_ids_in_file": len(file_sample_ids),
            "valid_inference_rows_raw": file_valid_inference,
            "scorable_rows_raw": file_scorable,
            "agent_inference_reused_rows": reuse_rows,
            "guard_metadata_mismatch_rows": metadata_mismatch_rows,
            "expected_cases": expected if expected is not None else math.nan,
            "completion_pct": completion,
        })

    if not chosen_global:
        raise RuntimeError("No usable rows with complete composite keys were found.")

    duplicate_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    diagnostic_fields = [
        "long_context_id",
        "context_type",
        "agent_config",
        "agent_tag",
        "agent_quantization",
        "prompt_condition",
        "input_guard",
        "agent_inference_reused",
        "agent_inference_reused_from_guard",
    ]

    for key, rows in occurrences.items():
        if len(rows) <= 1:
            continue
        files_for_key = sorted({r["_source_file"] for r in rows})
        duplicate_rows.append({
            "analysis_key": composite_key_text(key),
            "guard_subset": key[0],
            "sample_id": key[1],
            "context_length_name": key[2],
            "unsafe_prompt_position": key[3],
            "agent_model": key[4],
            "occurrences": len(rows),
            "source_file_count": len(files_for_key),
            "source_files": " || ".join(files_for_key),
        })

        distinct = {field: {str(r.get(field, "")) for r in rows} for field in diagnostic_fields}
        conflicting = [field for field, values in distinct.items() if len(values) > 1]
        # Outcome conflicts matter even if provenance metadata agrees.
        outcome_fields = ["agent_decision", "restricted_tool_selection", "reason_code", "inference_error"]
        outcome_distinct = {field: {str(r.get(field, "")) for r in rows} for field in outcome_fields}
        outcome_conflicts = [field for field, values in outcome_distinct.items() if len(values) > 1]

        if conflicting or outcome_conflicts:
            row_out = {
                "analysis_key": composite_key_text(key),
                "guard_subset": key[0],
                "sample_id": key[1],
                "context_length_name": key[2],
                "unsafe_prompt_position": key[3],
                "agent_model": key[4],
                "occurrences": len(rows),
                "conflicting_metadata_fields": ";".join(conflicting),
                "conflicting_outcome_fields": ";".join(outcome_conflicts),
                "source_files": " || ".join(files_for_key),
            }
            for field in diagnostic_fields:
                row_out[f"values_{field}"] = " || ".join(sorted(distinct[field]))
            for field in outcome_fields:
                row_out[f"values_{field}"] = " || ".join(sorted(outcome_distinct[field]))
            collision_rows.append(row_out)

    data = pd.DataFrame(list(chosen_global.values()))
    if "_file_index" in data.columns:
        data = data.drop(columns=["_file_index"])

    print()
    print(f"Raw valid JSON rows: {raw_valid_json_rows:,}")
    print(f"Unique rows after guard-aware GLOBAL dedup: {len(data):,}")
    print(f"Duplicate occurrences removed: {sum(len(v) - 1 for v in occurrences.values()):,}")
    print(f"Keys with metadata/outcome collisions: {len(collision_rows):,}")
    print()

    malformed_df = pd.DataFrame(
        malformed + [dict(row, error="missing_composite_key") for row in missing_key_rows]
    )
    return (
        data,
        pd.DataFrame(inventory),
        pd.DataFrame(duplicate_rows),
        pd.DataFrame(collision_rows),
        malformed_df,
    )


# =============================================================================
# METRICS / TABLES
# =============================================================================

def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    valid = int(frame["valid_inference"].sum()) if n else 0
    scorable = frame[frame["decision_valid"]] if n else frame
    ns = len(scorable)
    k_unsafe = int(scorable["restricted_tool_selection"].sum()) if ns else 0
    ci_lo, ci_hi = wilson_ci(k_unsafe, ns)

    n_refuse = int((scorable["agent_decision"] == "refuse").sum()) if ns else 0
    n_escalate = int((scorable["agent_decision"] == "escalate").sum()) if ns else 0
    n_call = int((scorable["agent_decision"] == "call_tool").sum()) if ns else 0
    n_safe = int(scorable["safe_recovery"].sum()) if ns else 0

    return {
        "n_rows": n,
        "n_unique_analysis_key": int(frame["analysis_key"].nunique()) if n else 0,
        "n_unique_sample_id": int(frame.loc[frame["sample_id"] != "", "sample_id"].nunique()) if n else 0,
        "n_valid_inference": valid,
        "inference_success_pct": safe_pct(valid, n),
        "n_inference_failure": n - valid,
        "inference_failure_pct": safe_pct(n - valid, n),
        "n_scorable": ns,
        "scorable_pct": safe_pct(ns, n),
        "n_parse_failure": int(frame["parse_failure"].sum()) if n else 0,
        "parse_failure_pct_valid_inference": safe_pct(int(frame["parse_failure"].sum()), valid) if n else math.nan,
        "n_refuse": n_refuse,
        "refuse_pct_scorable": safe_pct(n_refuse, ns),
        "n_escalate": n_escalate,
        "escalate_pct_scorable": safe_pct(n_escalate, ns),
        "n_call_tool": n_call,
        "call_tool_pct_scorable": safe_pct(n_call, ns),
        "n_safe_recovery": n_safe,
        "safe_recovery_pct_scorable": safe_pct(n_safe, ns),
        "n_restricted_tool_selection": k_unsafe,
        "restricted_tool_selection_pct_scorable": safe_pct(k_unsafe, ns),
        "restricted_tool_selection_ci95_low": ci_lo,
        "restricted_tool_selection_ci95_high": ci_hi,
        "n_invalid_tool_call": int(frame["invalid_tool_call"].sum()) if n else 0,
        "mean_rendered_input_tokens": frame["rendered_agent_input_tokens"].mean() if n else math.nan,
        "median_rendered_input_tokens": frame["rendered_agent_input_tokens"].median() if n else math.nan,
        "mean_seconds_per_case": frame["approx_seconds_per_case"].mean() if n else math.nan,
        "median_seconds_per_case": frame["approx_seconds_per_case"].median() if n else math.nan,
        "p95_seconds_per_case": frame["approx_seconds_per_case"].quantile(0.95) if n else math.nan,
        "n_reused_inference_rows": int(frame["agent_inference_reused"].sum()) if n else 0,
    }


def grouped_summary(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame([metrics(frame)])
    out = []
    grouper: str | list[str] = cols[0] if len(cols) == 1 else cols
    for keys, group in frame.groupby(grouper, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(cols, keys))
        row.update(metrics(group))
        out.append(row)
    return pd.DataFrame(out)


def decision_composition_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = frame[frame["decision_valid"]].copy()
    if d.empty:
        return pd.DataFrame()
    cols = group_cols + ["agent_decision"]
    tab = d.groupby(cols, dropna=False).size().rename("n").reset_index()
    totals = tab.groupby(group_cols, dropna=False)["n"].transform("sum")
    tab["pct"] = 100.0 * tab["n"] / totals
    return tab


def language_risk_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for language, group in frame.groupby("language", dropna=False, sort=True):
        total = len(group)
        valid = int(group["valid_inference"].sum())
        scorable = group[group["decision_valid"]]
        n = len(scorable)
        unsafe = int(scorable["restricted_tool_selection"].sum())
        risk = safe_pct(unsafe, n)
        lo, hi = wilson_ci(unsafe, n)
        rows.append({
            "language": language,
            "n_total_rows": total,
            "n_valid_inference": valid,
            "inference_success_pct": safe_pct(valid, total),
            "n_scorable": n,
            "scorable_pct": safe_pct(n, total),
            "n_restricted_tool_selection": unsafe,
            "restricted_tool_selection_pct": risk,
            "ci95_low": lo,
            "ci95_high": hi,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    valid = result[result["restricted_tool_selection_pct"].notna()].copy()
    valid = valid.sort_values(
        ["restricted_tool_selection_pct", "n_scorable", "language"],
        ascending=[False, False, True],
    )
    rank_map = {language: rank for rank, language in enumerate(valid["language"], start=1)}
    result["risk_rank"] = result["language"].map(rank_map)
    return result.sort_values(["risk_rank", "language"], na_position="last").reset_index(drop=True)


def select_language_risk_strata(language_risk: pd.DataFrame, n_each: int) -> pd.DataFrame:
    if language_risk.empty:
        return pd.DataFrame()
    data = language_risk[language_risk["restricted_tool_selection_pct"].notna()].copy()
    if data.empty:
        return pd.DataFrame()
    n_each = min(n_each, len(data) // 3)
    if n_each < 1:
        return pd.DataFrame()

    highest = data.sort_values(
        ["restricted_tool_selection_pct", "n_scorable", "language"],
        ascending=[False, False, True],
    ).head(n_each).copy()
    highest["risk_group"] = "Highest-risk"

    remaining = data[~data["language"].isin(highest["language"])].copy()
    lowest = remaining.sort_values(
        ["restricted_tool_selection_pct", "n_scorable", "language"],
        ascending=[True, False, True],
    ).head(n_each).copy()
    lowest["risk_group"] = "Lowest-risk"

    remaining = remaining[~remaining["language"].isin(lowest["language"])].copy()
    median_risk = float(data["restricted_tool_selection_pct"].median())
    remaining["distance_from_median"] = (
        remaining["restricted_tool_selection_pct"] - median_risk
    ).abs()
    medium = remaining.sort_values(
        ["distance_from_median", "n_scorable", "language"],
        ascending=[True, False, True],
    ).head(n_each).copy()
    medium["risk_group"] = "Medium-risk"

    result = pd.concat([highest, medium, lowest], ignore_index=True)
    result["global_median_risk_pct"] = median_risk
    return result


def paired_comparison(
    frame: pd.DataFrame,
    condition_col: str,
    pairs: list[tuple[str, str]],
    group_cols: list[str],
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    frame = frame[frame["sample_id"] != ""].copy()
    grouper: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols

    for group_index, (keys, group) in enumerate(frame.groupby(grouper, dropna=False, sort=True)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys))

        for pair_index, (ca, cb) in enumerate(pairs):
            a = (
                group[group[condition_col] == ca]
                .drop_duplicates("sample_id", keep="last")
                .set_index("sample_id")
            )
            b = (
                group[group[condition_col] == cb]
                .drop_duplicates("sample_id", keep="last")
                .set_index("sample_id")
            )
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            a = a.loc[common]
            b = b.loc[common]
            mask = a["decision_valid"].astype(bool) & b["decision_valid"].astype(bool)
            if int(mask.sum()) == 0:
                continue
            ya = a.loc[mask, "restricted_tool_selection"].astype(bool).to_numpy()
            yb = b.loc[mask, "restricted_tool_selection"].astype(bool).to_numpy()
            n01 = int(((~ya) & yb).sum())
            n10 = int((ya & (~yb)).sum())
            delta, lo, hi = bootstrap_paired_delta(
                ya, yb, bootstrap, seed + group_index * 100 + pair_index
            )
            risk_a = float(ya.mean())
            risk_b = float(yb.mean())
            rr = risk_b / risk_a if risk_a > 0 else (math.inf if risk_b > 0 else math.nan)
            row = dict(base)
            row.update({
                "comparison_variable": condition_col,
                "condition_a": ca,
                "condition_b": cb,
                "n_overlap": len(common),
                "n_scorable_pairs": int(mask.sum()),
                "unsafe_pct_a": 100.0 * risk_a,
                "unsafe_pct_b": 100.0 * risk_b,
                "delta_percentage_points_b_minus_a": delta,
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "risk_ratio_b_over_a": rr,
                "a_safe_b_unsafe": n01,
                "a_unsafe_b_safe": n10,
                "mcnemar_exact_p": mcnemar_exact_p(n01, n10),
            })
            out.append(row)
    return pd.DataFrame(out)


def length_transition_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    group_cols = ["guard_subset", "agent_model", "unsafe_prompt_position"]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        a = (
            group[group["context_length_name"] == "8k"]
            .drop_duplicates("sample_id", keep="last")
            .set_index("sample_id")
        )
        b = (
            group[group["context_length_name"] == "32k"]
            .drop_duplicates("sample_id", keep="last")
            .set_index("sample_id")
        )
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        a = a.loc[common]
        b = b.loc[common]
        mask = a["decision_valid"].astype(bool) & b["decision_valid"].astype(bool)
        a = a.loc[mask]
        b = b.loc[mask]
        if len(a) == 0:
            continue
        ua = a["restricted_tool_selection"].astype(bool)
        ub = b["restricted_tool_selection"].astype(bool)
        base = dict(zip(group_cols, keys))
        for from_lab, from_val in [("safe", False), ("unsafe", True)]:
            for to_lab, to_val in [("safe", False), ("unsafe", True)]:
                n = int(((ua == from_val) & (ub == to_val)).sum())
                out.append({
                    **base,
                    "from_8k": from_lab,
                    "to_32k": to_lab,
                    "n": n,
                    "pct_of_pairs": safe_pct(n, len(a)),
                    "n_pairs": len(a),
                })
    return pd.DataFrame(out)


def position_bin_table(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame[frame["decision_valid"] & frame["unsafe_prompt_position_fraction"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    bins = np.linspace(0, 1, 11)
    d["position_bin"] = pd.cut(d["unsafe_prompt_position_fraction"], bins=bins, include_lowest=True)
    rows = []
    for keys, group in d.groupby(
        ["agent_name", "context_length_name", "position_bin"],
        observed=True,
        dropna=False,
    ):
        k = int(group["restricted_tool_selection"].sum())
        n = len(group)
        lo, hi = wilson_ci(k, n)
        rows.append({
            "agent_name": keys[0],
            "context_length_name": keys[1],
            "position_bin": str(keys[2]),
            "n": n,
            "unsafe_pct": safe_pct(k, n),
            "ci95_low": lo,
            "ci95_high": hi,
            "mean_position_fraction": group["unsafe_prompt_position_fraction"].mean(),
        })
    return pd.DataFrame(rows)


def latency_table(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame[np.isfinite(frame["approx_seconds_per_case"])].copy()
    if d.empty:
        return pd.DataFrame()
    return (
        d.groupby(
            ["guard_subset", "agent_name", "context_length_name", "unsafe_prompt_position"],
            dropna=False,
        )["approx_seconds_per_case"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            std="std",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p95=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )


# =============================================================================
# PLOT HELPERS
# =============================================================================

def risk_summary_for_plot(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    d = frame[frame["decision_valid"]]
    if d.empty:
        return pd.DataFrame()
    grouper: str | list[str] = groups[0] if len(groups) == 1 else groups
    for keys, group in d.groupby(grouper, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        k = int(group["restricted_tool_selection"].sum())
        n = len(group)
        lo, hi = wilson_ci(k, n)
        row = dict(zip(groups, keys))
        row.update({"n": n, "unsafe_pct": safe_pct(k, n), "ci_low": lo, "ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    fmt: str = ".1f",
    cbar_label: str = "Restricted-tool selection (%)",
) -> None:
    arr = matrix.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    im = ax.imshow(arr, aspect="auto", cmap=HEATMAP_CMAP, vmin=0, vmax=max(vmax, 1e-9))
    ax.set_xticks(np.arange(matrix.shape[1]), labels=matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = 0.55 * vmax
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(
                    j, i, format(value, fmt), ha="center", va="center", fontsize=7.2,
                    color="white" if value >= threshold else COLORS["dark"],
                )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)


def forest_plot(
    table: pd.DataFrame,
    labels: pd.Series,
    title: str,
    xlabel: str,
    output: Path,
) -> None:
    if table.empty:
        return
    d = table.copy()
    d["_label"] = labels.astype(str)
    d = d.sort_values("delta_percentage_points_b_minus_a")
    y = np.arange(len(d))
    point = d["delta_percentage_points_b_minus_a"].to_numpy(dtype=float)
    lo = d["delta_ci95_low"].to_numpy(dtype=float)
    hi = d["delta_ci95_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, max(4.0, 0.30 * len(d) + 1.6)))
    valid = np.isfinite(point) & np.isfinite(lo) & np.isfinite(hi)
    if valid.any():
        ax.errorbar(
            point[valid], y[valid],
            xerr=[point[valid] - lo[valid], hi[valid] - point[valid]],
            fmt="o", capsize=3, color=COLORS["navy"], ecolor=COLORS["slate"],
        )
    ax.axvline(0, color=COLORS["gray"], linewidth=1)
    ax.set_yticks(y, d["_label"])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    clean_axes(ax, "x")
    save_figure(fig, output)


# =============================================================================
# FIGURES
# =============================================================================

def fig01_reference_agent_length_position(reference: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(reference, ["agent_name", "context_length_name", "unsafe_prompt_position"])
    if summary.empty:
        return
    summary["condition"] = summary.apply(
        lambda r: ordered_condition(r["context_length_name"], r["unsafe_prompt_position"]), axis=1
    )
    desired = [ordered_condition(l, p) for l in LENGTH_ORDER for p in POSITION_ORDER]
    matrix = summary.pivot(index="agent_name", columns="condition", values="unsafe_pct")
    matrix = matrix.reindex(index=[a for a in AGENT_ORDER if a in matrix.index])
    matrix = matrix.reindex(columns=[x for x in desired if x in matrix.columns])
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    draw_heatmap(ax, matrix)
    ax.set_title("Reference-set safety failure by agent, context length, and prompt position", loc="left", fontweight="bold")
    ax.set_xlabel("Context condition")
    ax.set_ylabel("Downstream agent")
    save_figure(fig, figdir / "fig01_reference_agent_length_position_heatmap")


def fig02_guard_agent_heatmap(data: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(data, ["guard_subset", "agent_name"])
    if summary.empty:
        return
    matrix = summary.pivot(index="guard_subset", columns="agent_name", values="unsafe_pct")
    matrix.index = [guard_display(x) for x in matrix.index]
    fig, ax = plt.subplots(figsize=(6.8, max(3.7, 0.55 * len(matrix) + 1.7)))
    draw_heatmap(ax, matrix)
    ax.set_title("Downstream risk on each guard-admitted subset", loc="left", fontweight="bold")
    ax.set_xlabel("Downstream agent")
    ax.set_ylabel("Guard subset")
    save_figure(fig, figdir / "fig02_guard_subset_agent_heatmap")


def fig03_guard_length(data: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(data, ["guard_subset", "context_length_name"])
    if summary.empty:
        return
    guards = list(dict.fromkeys(summary["guard_subset"]))
    x = np.arange(len(guards))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, length in enumerate(LENGTH_ORDER):
        sub = summary[summary["context_length_name"] == length].set_index("guard_subset").reindex(guards)
        y = sub["unsafe_pct"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            x + (idx - 0.5) * 0.12, y, yerr=[y - lo, hi - y],
            fmt="o", capsize=2.5, label=length.upper(), color=PALETTE[idx],
        )
    ax.set_xticks(x, [guard_display(g) for g in guards], rotation=20, ha="right")
    ax.set_ylabel("Restricted-tool selection (%)")
    ax.set_xlabel("Guard-admitted subset")
    ax.set_title("8K versus 32K risk across guard-admitted subsets", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    clean_axes(ax, "y")
    save_figure(fig, figdir / "fig03_guard_subset_context_length_ci")


def fig04_guard_position(data: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(data, ["guard_subset", "unsafe_prompt_position"])
    if summary.empty:
        return
    guards = list(dict.fromkeys(summary["guard_subset"]))
    x = np.arange(len(guards))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    offsets = np.linspace(-0.18, 0.18, len(POSITION_ORDER))
    for idx, position in enumerate(POSITION_ORDER):
        sub = summary[summary["unsafe_prompt_position"] == position].set_index("guard_subset").reindex(guards)
        y = sub["unsafe_pct"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            x + offsets[idx], y, yerr=[y - lo, hi - y],
            fmt="o", capsize=2.2, label=position.title(), color=PALETTE[idx],
        )
    ax.set_xticks(x, [guard_display(g) for g in guards], rotation=20, ha="right")
    ax.set_ylabel("Restricted-tool selection (%)")
    ax.set_xlabel("Guard-admitted subset")
    ax.set_title("Prompt-position risk across guard-admitted subsets", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    clean_axes(ax, "y")
    save_figure(fig, figdir / "fig04_guard_subset_prompt_position_ci")


def fig05_guard_condition_heatmap(data: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(data, ["guard_subset", "context_length_name", "unsafe_prompt_position"])
    if summary.empty:
        return
    summary["condition"] = summary.apply(
        lambda r: ordered_condition(r["context_length_name"], r["unsafe_prompt_position"]), axis=1
    )
    desired = [ordered_condition(l, p) for l in LENGTH_ORDER for p in POSITION_ORDER]
    matrix = summary.pivot(index="guard_subset", columns="condition", values="unsafe_pct")
    matrix = matrix.reindex(columns=[x for x in desired if x in matrix.columns])
    matrix.index = [guard_display(x) for x in matrix.index]
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.52 * len(matrix) + 1.7)))
    draw_heatmap(ax, matrix)
    ax.set_title("Context-length × position risk by guard-admitted subset", loc="left", fontweight="bold")
    ax.set_xlabel("Context condition")
    ax.set_ylabel("Guard subset")
    save_figure(fig, figdir / "fig05_guard_subset_length_position_heatmap")


def fig06_reference_decision_composition(reference: pd.DataFrame, figdir: Path) -> None:
    d = reference[reference["decision_valid"]].copy()
    if d.empty:
        return
    d["group"] = d["agent_name"] + " | " + d["condition"]
    tab = pd.crosstab(d["group"], d["agent_decision"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(8.8, max(4.0, 0.30 * len(tab) + 1.4)))
    left = np.zeros(len(tab))
    for decision, color in [
        ("refuse", COLORS["navy"]),
        ("escalate", COLORS["sage"]),
        ("call_tool", COLORS["brick"]),
    ]:
        vals = tab[decision].to_numpy() if decision in tab.columns else np.zeros(len(tab))
        ax.barh(np.arange(len(tab)), vals, left=left, label=decision.replace("_", " ").title(), color=color)
        left += vals
    ax.set_yticks(np.arange(len(tab)), tab.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Decision composition among scorable outputs (%)")
    ax.set_title("Reference-set decision composition", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    clean_axes(ax, "x")
    save_figure(fig, figdir / "fig06_reference_decision_composition")


def fig07_language_risk(language_risk: pd.DataFrame, figdir: Path, n_each: int) -> None:
    selected = select_language_risk_strata(language_risk, n_each)
    if selected.empty:
        return
    group_order = ["Highest-risk", "Medium-risk", "Lowest-risk"]
    group_colors = {
        "Highest-risk": COLORS["brick"],
        "Medium-risk": COLORS["ochre"],
        "Lowest-risk": COLORS["sage"],
    }
    rows = []
    for group_name in group_order:
        sub = selected[selected["risk_group"] == group_name].sort_values(
            "restricted_tool_selection_pct", ascending=False
        )
        for _, row in sub.iterrows():
            rows.append((group_name, row))
        rows.append(("separator", None))
    if rows and rows[-1][0] == "separator":
        rows.pop()

    y, labels, risks, lo, hi, colors = [], [], [], [], [], []
    pos = 0
    for group_name, row in rows:
        if group_name == "separator":
            pos += 1.5
            continue
        y.append(pos)
        labels.append(str(row["language"]))
        risks.append(float(row["restricted_tool_selection_pct"]))
        lo.append(float(row["ci95_low"]))
        hi.append(float(row["ci95_high"]))
        colors.append(group_colors[group_name])
        pos += 1

    y_arr = np.asarray(y, dtype=float)
    risk_arr = np.asarray(risks, dtype=float)
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 15.5))
    ax.barh(y_arr, risk_arr, color=colors, alpha=0.85, height=0.68)
    ax.errorbar(risk_arr, y_arr, xerr=[risk_arr - lo_arr, hi_arr - risk_arr], fmt="none", ecolor=COLORS["dark"], capsize=1.5)
    ax.set_yticks(y_arr, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Restricted-tool selection among scorable runs (%)")
    ax.set_title("Reference-set cross-lingual downstream risk", loc="left", fontweight="bold")
    clean_axes(ax, "x")
    save_figure(fig, figdir / "fig07_reference_language_risk_high_medium_low")


def fig08_resource_tier(reference: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(reference, ["tier"])
    if summary.empty:
        return
    summary = summary.sort_values("tier")
    x = np.arange(len(summary))
    risk = summary["unsafe_pct"].to_numpy(dtype=float)
    lo = summary["ci_low"].to_numpy(dtype=float)
    hi = summary["ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar(x, risk, color=COLORS["navy"], alpha=0.85, width=0.62)
    ax.errorbar(x, risk, yerr=[risk - lo, hi - risk], fmt="none", ecolor=COLORS["dark"], capsize=3)
    ax.set_xticks(x, summary["tier"], rotation=15, ha="right")
    ax.set_xlabel("Language-resource tier")
    ax.set_ylabel("Restricted-tool selection (%)")
    ax.set_title("Reference-set risk by language-resource tier", loc="left", fontweight="bold")
    clean_axes(ax, "y")
    save_figure(fig, figdir / "fig08_reference_resource_tier_aggregate")


def fig09_tier_agent(reference: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(reference, ["agent_name", "tier"])
    if summary.empty:
        return
    tiers = list(dict.fromkeys(summary["tier"]))
    agents = list(dict.fromkeys(summary["agent_name"]))
    x = np.arange(len(tiers))
    offsets = np.linspace(-0.12, 0.12, len(agents))
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    for idx, agent in enumerate(agents):
        sub = summary[summary["agent_name"] == agent].set_index("tier").reindex(tiers)
        y = sub["unsafe_pct"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        ax.errorbar(x + offsets[idx], y, yerr=[y - lo, hi - y], fmt="o", capsize=2.5, label=agent, color=PALETTE[idx])
    ax.set_xticks(x, tiers, rotation=15, ha="right")
    ax.set_xlabel("Language-resource tier")
    ax.set_ylabel("Restricted-tool selection (%)")
    ax.set_title("Reference-set resource-tier risk by agent", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    clean_axes(ax, "y")
    save_figure(fig, figdir / "fig09_reference_resource_tier_agent_ci")


def fig10_category_agent(reference: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(reference, ["category", "agent_name"])
    if summary.empty:
        return
    support = reference[reference["decision_valid"]].groupby("category").size().sort_values(ascending=False)
    categories = support.head(20).index
    matrix = (
        summary[summary["category"].isin(categories)]
        .pivot(index="category", columns="agent_name", values="unsafe_pct")
        .reindex(categories)
    )
    fig, ax = plt.subplots(figsize=(7.2, max(5.2, 0.34 * len(matrix) + 1.5)))
    draw_heatmap(ax, matrix)
    ax.set_title("Reference-set safety failure by harm category and agent", loc="left", fontweight="bold")
    ax.set_xlabel("Downstream agent")
    ax.set_ylabel("Harm category")
    save_figure(fig, figdir / "fig10_reference_category_agent_heatmap")


def fig11_category_guard(data: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(data, ["category", "guard_subset"])
    if summary.empty:
        return
    support = data[data["decision_valid"]].groupby("category").size().sort_values(ascending=False)
    categories = support.head(20).index
    matrix = (
        summary[summary["category"].isin(categories)]
        .pivot(index="category", columns="guard_subset", values="unsafe_pct")
        .reindex(categories)
    )
    matrix.columns = [guard_display(x) for x in matrix.columns]
    fig, ax = plt.subplots(figsize=(8.0, max(5.2, 0.34 * len(matrix) + 1.5)))
    draw_heatmap(ax, matrix)
    ax.set_title("Harm-category risk across guard-admitted subsets", loc="left", fontweight="bold")
    ax.set_xlabel("Guard subset")
    ax.set_ylabel("Harm category")
    save_figure(fig, figdir / "fig11_category_guard_subset_heatmap")


def fig12_translation_quality(reference: pd.DataFrame, figdir: Path) -> None:
    summary = risk_summary_for_plot(reference, ["agent_name", "quality_bucket"])
    if summary.empty:
        return
    buckets = list(dict.fromkeys(summary["quality_bucket"]))
    agents = list(dict.fromkeys(summary["agent_name"]))
    x = np.arange(len(buckets))
    offsets = np.linspace(-0.12, 0.12, len(agents))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for idx, agent in enumerate(agents):
        sub = summary[summary["agent_name"] == agent].set_index("quality_bucket").reindex(buckets)
        y = sub["unsafe_pct"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        ax.errorbar(x + offsets[idx], y, yerr=[y - lo, hi - y], fmt="o", capsize=2.5, label=agent, color=PALETTE[idx])
    ax.set_xticks(x, buckets, rotation=20, ha="right")
    ax.set_xlabel("Translation-quality bucket")
    ax.set_ylabel("Restricted-tool selection (%)")
    ax.set_title("Reference-set safety risk versus translation quality", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    clean_axes(ax, "y")
    save_figure(fig, figdir / "fig12_reference_translation_quality_ci")


def fig13_transition_reference(transitions: pd.DataFrame, reference_guard: str, figdir: Path) -> None:
    d = transitions[transitions["guard_subset"] == reference_guard].copy()
    if d.empty:
        return
    agg = d.groupby(["from_8k", "to_32k"])["n"].sum().reset_index()
    order = ["safe", "unsafe"]
    matrix = agg.pivot(index="from_8k", columns="to_32k", values="n").reindex(index=order, columns=order).fillna(0)
    total = matrix.to_numpy().sum()
    pct = 100.0 * matrix / total if total > 0 else matrix
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    draw_heatmap(ax, pct, fmt=".1f", cbar_label="Share of matched pairs (%)")
    ax.set_title("Reference-set safety-state transition: 8K → 32K", loc="left", fontweight="bold")
    ax.set_xlabel("32K outcome")
    ax.set_ylabel("8K outcome")
    save_figure(fig, figdir / "fig13_reference_length_transition_matrix")


def fig14_position_fraction(position_bins: pd.DataFrame, figdir: Path) -> None:
    if position_bins.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for idx, ((agent, length), group) in enumerate(position_bins.groupby(["agent_name", "context_length_name"])):
        group = group.sort_values("mean_position_fraction")
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(group["mean_position_fraction"], group["unsafe_pct"], marker="o", markersize=3.8, linewidth=1.4, label=f"{agent} | {length.upper()}", color=color)
        ax.fill_between(group["mean_position_fraction"], group["ci95_low"], group["ci95_high"], alpha=0.08, color=color)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized target-prompt position")
    ax.set_ylabel("Restricted-tool selection (%)")
    ax.set_title("Reference-set risk as the harmful prompt moves through context", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    clean_axes(ax, "both")
    save_figure(fig, figdir / "fig14_reference_position_fraction_curve")


def fig15_latency(data: pd.DataFrame, figdir: Path) -> None:
    summary = latency_table(data)
    if summary.empty:
        return
    d = summary.groupby(["guard_subset", "context_length_name"])["median"].mean().reset_index()
    guards = list(dict.fromkeys(d["guard_subset"]))
    x = np.arange(len(guards))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    for idx, length in enumerate(LENGTH_ORDER):
        sub = d[d["context_length_name"] == length].set_index("guard_subset").reindex(guards)
        ax.bar(x + (idx - 0.5) * width, sub["median"], width=width, label=length.upper(), color=PALETTE[idx])
    ax.set_xticks(x, [guard_display(g) for g in guards], rotation=20, ha="right")
    ax.set_ylabel("Median approximate seconds/case")
    ax.set_xlabel("Guard subset")
    ax.set_title("Long-context inference cost by subset and length", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    clean_axes(ax, "y")
    save_figure(fig, figdir / "fig15_latency_by_guard_subset_length")


def fig16_failure(data: pd.DataFrame, figdir: Path) -> None:
    rows = []
    for keys, group in data.groupby(["guard_subset", "agent_name", "context_length_name"], sort=True):
        rows.append({
            "label": f"{guard_display(keys[0])} | {keys[1]} | {keys[2].upper()}",
            "inference_failure": 100.0 * (~group["valid_inference"]).mean(),
            "parse_failure": 100.0 * group["parse_failure"].mean(),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return
    y = np.arange(len(result))
    fig, ax = plt.subplots(figsize=(9.0, max(4.2, 0.28 * len(result) + 1.5)))
    ax.barh(y, result["inference_failure"], color=COLORS["slate"], label="Inference failure")
    ax.barh(y, result["parse_failure"], left=result["inference_failure"], color=COLORS["ochre"], label="Parse failure")
    ax.set_yticks(y, result["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Failure rate over rows (%)")
    ax.set_title("Inference and parsing failure diagnostics", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    clean_axes(ax, "x")
    save_figure(fig, figdir / "fig16_failure_diagnostics_by_guard_subset")


def fig17_subset_coverage(data: pd.DataFrame, figdir: Path) -> None:
    counts = data.groupby("guard_subset")["sample_id"].nunique().sort_values(ascending=False)
    if counts.empty:
        return
    labels = [guard_display(x) for x in counts.index]
    y = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.barh(y, counts.values, color=COLORS["navy"], alpha=0.85)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Unique sample_ids represented")
    ax.set_title("Analysis population size by guard-admitted subset", loc="left", fontweight="bold")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f" {int(value):,}", va="center", fontsize=8)
    clean_axes(ax, "x")
    save_figure(fig, figdir / "fig17_guard_subset_sample_coverage")


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[WRITE] {path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze long-context agent results recursively across guard-specific subdirectories"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reference-guard",
        default="aprielguard",
        help="Guard subset used for non-duplicated main agent/context/language analysis. Default: aprielguard",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=None,
        help="Expected sample count for the reference guard only, e.g. 3190",
    )
    parser.add_argument(
        "--expected-cases-by-guard",
        nargs="*",
        default=[],
        help="Optional per-subset expected counts: crest=1237 guardreasoner=102 wildguard=18 xguard=2524",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--risk-languages-per-group", type=int, default=20)
    parser.add_argument("--extra-metadata-fields", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_style()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    reference_guard = args.reference_guard.strip().lower()
    expected_map = parse_expected_cases_by_guard(args.expected_cases_by_guard)

    data, inventory, duplicate_keys, key_collisions, malformed = load_all_results(
        input_dir,
        args.expected_cases,
        expected_map,
        reference_guard,
        args.extra_metadata_fields,
    )

    guards = sorted(data["guard_subset"].dropna().astype(str).unique())
    if reference_guard not in guards:
        raise ValueError(
            f"Reference guard {reference_guard!r} not found. Available guard subsets: {guards}"
        )
    reference = data[data["guard_subset"] == reference_guard].copy()

    print(f"Loaded {len(data):,} guard-aware deduplicated rows.")
    print(f"Guard subsets: {guards}")
    print(f"Reference guard rows: {len(reference):,}")
    print(f"Reference unique sample_ids: {reference['sample_id'].nunique():,}")
    print(f"Reference languages: {reference['language'].nunique():,}")
    print(f"Agents: {sorted(data['agent_name'].unique())}")
    print(f"Context lengths: {sorted(data['context_length_name'].unique())}")
    print(f"Prompt positions: {sorted(data['unsafe_prompt_position'].unique())}")
    print()

    # -------------------------------------------------------------------------
    # Tables (30+)
    # -------------------------------------------------------------------------
    tables: dict[str, pd.DataFrame] = {}
    tables["00_file_inventory"] = inventory
    tables["01_reference_overall_summary"] = grouped_summary(reference, [])
    tables["02_guard_subset_coverage"] = grouped_summary(data, ["guard_subset"])
    tables["03_guard_subset_reuse_provenance"] = grouped_summary(data, ["guard_subset", "agent_inference_reused"])
    tables["04_by_guard_subset_agent"] = grouped_summary(data, ["guard_subset", "agent_name"])
    tables["05_reference_by_agent"] = grouped_summary(reference, ["agent_name"])
    tables["06_reference_by_context_length"] = grouped_summary(reference, ["context_length_name"])
    tables["07_reference_by_prompt_position"] = grouped_summary(reference, ["unsafe_prompt_position"])
    tables["08_reference_by_length_position"] = grouped_summary(reference, ["context_length_name", "unsafe_prompt_position"])
    tables["09_reference_by_agent_length"] = grouped_summary(reference, ["agent_name", "context_length_name"])
    tables["10_reference_by_agent_position"] = grouped_summary(reference, ["agent_name", "unsafe_prompt_position"])
    tables["11_reference_by_agent_length_position"] = grouped_summary(reference, ["agent_name", "context_length_name", "unsafe_prompt_position"])
    tables["12_by_guard_subset_length"] = grouped_summary(data, ["guard_subset", "context_length_name"])
    tables["13_by_guard_subset_position"] = grouped_summary(data, ["guard_subset", "unsafe_prompt_position"])
    tables["14_by_guard_subset_agent_length"] = grouped_summary(data, ["guard_subset", "agent_name", "context_length_name"])
    tables["15_by_guard_subset_agent_position"] = grouped_summary(data, ["guard_subset", "agent_name", "unsafe_prompt_position"])
    tables["16_by_guard_subset_length_position"] = grouped_summary(data, ["guard_subset", "context_length_name", "unsafe_prompt_position"])
    tables["17_by_guard_subset_agent_length_position"] = grouped_summary(data, ["guard_subset", "agent_name", "context_length_name", "unsafe_prompt_position"])
    tables["18_reference_decision_composition"] = decision_composition_table(reference, ["agent_name", "context_length_name", "unsafe_prompt_position"])
    tables["19_by_guard_subset_decision_composition"] = decision_composition_table(data, ["guard_subset", "agent_name", "context_length_name", "unsafe_prompt_position"])

    # Category / language / tier / quality: primary publication analyses use reference set.
    tables["20_reference_by_category"] = grouped_summary(reference, ["category"])
    tables["21_reference_by_category_agent"] = grouped_summary(reference, ["category", "agent_name"])
    tables["22_reference_by_category_length"] = grouped_summary(reference, ["category", "context_length_name"])
    tables["23_reference_by_category_position"] = grouped_summary(reference, ["category", "unsafe_prompt_position"])
    tables["24_guard_subset_by_category"] = grouped_summary(data, ["guard_subset", "category"])
    tables["25_reference_by_language"] = grouped_summary(reference, ["language"])
    tables["26_reference_by_language_agent"] = grouped_summary(reference, ["language", "agent_name"])
    tables["27_reference_by_language_length"] = grouped_summary(reference, ["language", "context_length_name"])
    tables["28_reference_by_language_position"] = grouped_summary(reference, ["language", "unsafe_prompt_position"])

    language_risk = language_risk_table(reference)
    language_strata = select_language_risk_strata(language_risk, args.risk_languages_per_group)
    tables["29_reference_language_risk_all_languages"] = language_risk
    tables["30_reference_language_risk_figure_selection"] = language_strata

    tables["31_reference_resource_tier_aggregate"] = grouped_summary(reference, ["tier"])
    tables["32_reference_by_tier_agent"] = grouped_summary(reference, ["tier", "agent_name"])
    tables["33_guard_subset_by_tier"] = grouped_summary(data, ["guard_subset", "tier"])
    tables["34_reference_by_quality_bucket"] = grouped_summary(reference, ["quality_bucket"])
    tables["35_reference_by_quality_agent"] = grouped_summary(reference, ["quality_bucket", "agent_name"])

    # Paired comparisons are performed INSIDE each guard subset.
    length_pairs = paired_comparison(
        data,
        "context_length_name",
        [("8k", "32k")],
        ["guard_subset", "agent_model", "unsafe_prompt_position"],
        args.bootstrap,
        args.seed,
    )
    position_pairs = paired_comparison(
        data,
        "unsafe_prompt_position",
        [("beginning", "middle"), ("beginning", "end"), ("middle", "end")],
        ["guard_subset", "agent_model", "context_length_name"],
        args.bootstrap,
        args.seed,
    )
    agent_models = sorted(data["agent_model"].dropna().unique())
    agent_pairs = paired_comparison(
        data,
        "agent_model",
        list(combinations(agent_models, 2)),
        ["guard_subset", "context_length_name", "unsafe_prompt_position"],
        args.bootstrap,
        args.seed,
    ) if len(agent_models) >= 2 else pd.DataFrame()

    tables["36_paired_length_8k_vs_32k_by_guard_subset"] = length_pairs
    tables["37_paired_prompt_position_by_guard_subset"] = position_pairs
    tables["38_paired_agent_model_by_guard_subset"] = agent_pairs

    transitions = length_transition_table(data)
    tables["39_length_transition_8k_to_32k_by_guard_subset"] = transitions
    tables["40_latency_summary_by_guard_subset"] = latency_table(data)
    tables["41_inference_failures"] = data.loc[
        ~data["valid_inference"],
        [
            "_source_file", "guard_subset", "analysis_key", "sample_id", "language", "category", "tier",
            "agent_model", "context_length_name", "unsafe_prompt_position", "rendered_agent_input_tokens", "inference_error",
        ],
    ]
    tables["42_duplicate_composite_keys"] = duplicate_keys
    tables["43_composite_key_collisions"] = key_collisions
    tables["44_malformed_or_missing_key_rows"] = malformed

    pos_bins = position_bin_table(reference)
    tables["45_reference_prompt_position_bins"] = pos_bins

    # Optional metadata.
    extra_index = 50
    for field in args.extra_metadata_fields:
        if field not in data.columns:
            continue
        tables[f"{extra_index}_{field}_reference"] = grouped_summary(reference, [field])
        extra_index += 1
        tables[f"{extra_index}_{field}_guard_subset"] = grouped_summary(data, ["guard_subset", field])
        extra_index += 1

    for name, table in tables.items():
        save_csv(table, table_dir / f"{name}.csv")

    compact_columns = [
        "_source_file", "_line_number", "analysis_key", "guard_subset", "guard_subset_display",
        "file_guard", "input_guard", "guard_metadata_matches_path", "agent_inference_reused",
        "agent_inference_reused_from_guard", "agent_inference_reused_for_guard",
        "long_context_id", "sample_id", "root_id", "language", "category", "tier", "quality_bucket",
        "f1", "comet", "combined_score", "context_type", "context_length_name", "unsafe_prompt_position",
        "unsafe_prompt_position_fraction", "context_actual_tokens", "rendered_agent_input_tokens",
        "agent_model", "agent_tag", "agent_name", "agent_config", "agent_quantization", "prompt_condition",
        "valid_inference", "decision_valid", "agent_decision", "safe_recovery", "restricted_tool_selection",
        "unsafe_action_attempt", "tool_name_valid", "invalid_tool_call", "parse_failure",
        "decision_parse_source", "decision_parse_error", "reason_code", "inference_error", "approx_seconds_per_case",
    ]
    for field in args.extra_metadata_fields:
        if field in data.columns:
            compact_columns.append(field)
    save_csv(data[compact_columns], table_dir / "46_analysis_rows_compact.csv")

    # -------------------------------------------------------------------------
    # Figures (17 base + 3 paired forests = up to 20)
    # -------------------------------------------------------------------------
    fig01_reference_agent_length_position(reference, fig_dir)
    fig02_guard_agent_heatmap(data, fig_dir)
    fig03_guard_length(data, fig_dir)
    fig04_guard_position(data, fig_dir)
    fig05_guard_condition_heatmap(data, fig_dir)
    fig06_reference_decision_composition(reference, fig_dir)
    fig07_language_risk(language_risk, fig_dir, args.risk_languages_per_group)
    fig08_resource_tier(reference, fig_dir)
    fig09_tier_agent(reference, fig_dir)
    fig10_category_agent(reference, fig_dir)
    fig11_category_guard(data, fig_dir)
    fig12_translation_quality(reference, fig_dir)
    fig13_transition_reference(transitions, reference_guard, fig_dir)
    fig14_position_fraction(pos_bins, fig_dir)
    fig15_latency(data, fig_dir)
    fig16_failure(data, fig_dir)
    fig17_subset_coverage(data, fig_dir)

    if not length_pairs.empty:
        labels = (
            length_pairs["guard_subset"].map(guard_display)
            + " | "
            + length_pairs["agent_model"].astype(str).map(lambda x: canonical_agent(x, ""))
            + " | "
            + length_pairs["unsafe_prompt_position"].astype(str)
        )
        forest_plot(
            length_pairs,
            labels,
            "Paired effect of increasing context from 8K to 32K",
            "Δ restricted-tool selection (32K − 8K), percentage points",
            fig_dir / "fig18_paired_length_effect_by_guard_subset",
        )

    if not position_pairs.empty:
        labels = (
            position_pairs["guard_subset"].map(guard_display)
            + " | "
            + position_pairs["agent_model"].astype(str).map(lambda x: canonical_agent(x, ""))
            + " | "
            + position_pairs["context_length_name"].str.upper()
            + " | "
            + position_pairs["condition_a"]
            + "→"
            + position_pairs["condition_b"]
        )
        forest_plot(
            position_pairs,
            labels,
            "Paired effect of harmful-prompt position",
            "Δ restricted-tool selection, percentage points",
            fig_dir / "fig19_paired_position_effect_by_guard_subset",
        )

    if not agent_pairs.empty:
        labels = (
            agent_pairs["guard_subset"].map(guard_display)
            + " | "
            + agent_pairs["context_length_name"].str.upper()
            + " | "
            + agent_pairs["unsafe_prompt_position"].astype(str)
        )
        forest_plot(
            agent_pairs,
            labels,
            "Paired downstream-agent comparison",
            "Δ restricted-tool selection (model B − model A), percentage points",
            fig_dir / "fig20_paired_agent_model_effect_by_guard_subset",
        )

    manifest = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "recursive_input_scan": True,
        "guard_subsets": guards,
        "reference_guard": reference_guard,
        "files_analyzed": int(len(inventory)),
        "deduplicated_rows": int(len(data)),
        "reference_guard_rows": int(len(reference)),
        "reference_guard_unique_sample_ids": int(reference["sample_id"].nunique()),
        "analysis_key_fields": KEY_FIELDS,
        "paired_identifier": "sample_id",
        "primary_outcome": "restricted_tool_selection",
        "behavioral_denominator": "decision_valid == True",
        "bootstrap_iterations": args.bootstrap,
        "risk_languages_per_group": args.risk_languages_per_group,
        "expected_reference_cases": args.expected_cases,
        "expected_cases_by_guard": expected_map,
        "guard_subset_interpretation": (
            "Target-guard directories may contain AprielGuard inference reused on pairwise common admitted sample_ids. "
            "Guard-subset comparisons are descriptive population/subset analyses, not causal effects of the guard."
        ),
        "notes": [
            "Files are read recursively from guard-specific subdirectories.",
            "The guard subset is part of the global uniqueness key.",
            "Main language/category/tier/quality plots use the reference guard to avoid repeated weighting of copied inference.",
            "Paired length, position, and agent comparisons are computed separately within each guard subset.",
            "Inference failures are never counted as safe outcomes.",
        ],
    }
    manifest_path = output_dir / "analysis_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Input recursively scanned : {input_dir}")
    print(f"Guard subsets             : {', '.join(guards)}")
    print(f"Reference guard           : {reference_guard}")
    print(f"Reference sample_ids      : {reference['sample_id'].nunique():,}")
    print(f"All deduplicated rows     : {len(data):,}")
    print(f"Tables                    : {table_dir}")
    print(f"Figures                   : {fig_dir}")
    print(f"Manifest                  : {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
