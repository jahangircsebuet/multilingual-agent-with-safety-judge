#!/usr/bin/env python3
"""
09_create_guard_agent_files_from_apriel.py

Create target-guard downstream-agent result files by reusing completed
AprielGuard downstream-agent inference for sample_ids admitted by BOTH:

    AprielGuard
    AND
    the selected target guard.

Supported target guards:
    crest
    guardreasoner
    wildguard
    xguard

The script creates exactly 12 conversation-context files:

    2 context lengths:
        8k
        32k

    3 positions:
        beginning
        middle
        end

    2 agents:
        llama31_8b
        qwen25_14b

Important
---------
Classification files are read STRICTLY.

Agent-run source files are read TOLERANTLY because interrupted/resumed
generation can leave a malformed JSONL line. Malformed agent-run rows are:

    - skipped
    - NEVER treated as safe
    - NEVER reconstructed/fabricated
    - recorded in the output manifest

All valid source fields are preserved.

The copied row is marked with:

    input_guard = target guard
    agent_inference_reused = True
    agent_inference_reused_from_guard = aprielguard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


# =============================================================================
# CONSTANTS
# =============================================================================

SOURCE_GUARD = "aprielguard"

SUPPORTED_TARGET_GUARDS = {
    "crest",
    "guardreasoner",
    "wildguard",
    "xguard",
}

EXPECTED_COMMON_COUNTS = {
    "crest": 1237,
    "guardreasoner": 102,
    "wildguard": 18,
    "xguard": 2524,
}


# =============================================================================
# CLASSIFIED FILES
#
# Resolved relative to --classified-root (default: <repo_root>/data/
# sampledid_added_classified), so no machine-specific path is baked in here.
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLASSIFIED_ROOT = REPO_ROOT / "data" / "sampledid_added_classified"

CLASSIFIED_RELATIVE_FILES = {
    "aprielguard": Path("aprielguard/aprielguard_dataset_k=1.jsonl"),
    "crest": Path("crest/crest_dataset_k=1.jsonl"),
    "guardreasoner": Path("guardreasoner/guardreasoner_dataset_k=1.jsonl"),
    "wildguard": Path("wildguard/wildguard_dataset_k=1.jsonl"),
    "xguard": Path("xguard/xguard_dataset_k=1.jsonl"),
}


def build_classified_files(classified_root: Path) -> dict[str, Path]:
    return {
        guard: classified_root / relative_path
        for guard, relative_path in CLASSIFIED_RELATIVE_FILES.items()
    }


# =============================================================================
# EXPERIMENT CONDITIONS
# =============================================================================

CONTEXT_LENGTHS = [
    "8k",
    "32k",
]

PROMPT_POSITIONS = [
    "beginning",
    "middle",
    "end",
]

AGENT_TAGS = [
    "llama31_8b",
    "qwen25_14b",
]

VALID_DECISIONS = {
    "refuse",
    "escalate",
    "call_tool",
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def is_blank(value: Any) -> bool:

    if value is None:
        return True

    return str(value).strip().lower() in {
        "",
        "none",
        "null",
        "nan",
    }


# =============================================================================
# STRICT JSONL READER
#
# Used for classification files because these should be clean.
# =============================================================================

def read_jsonl_strict(
    path: Path,
) -> Iterable[tuple[int, dict[str, Any]]]:

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON\n"
                    f"File: {path}\n"
                    f"Line: {line_number}\n"
                    f"Error: {exc}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):

                raise ValueError(
                    f"Expected JSON object\n"
                    f"File: {path}\n"
                    f"Line: {line_number}"
                )

            yield (
                line_number,
                row,
            )


# =============================================================================
# TOLERANT AGENT-RUN JSONL READER
#
# Corrupted rows are skipped and logged rather than crashing the entire job.
# =============================================================================

def read_agent_jsonl_tolerant(
    path: Path,
    malformed_records: list[dict[str, Any]],
) -> Iterable[tuple[int, dict[str, Any]]]:

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            raw_line = line

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                malformed_records.append(
                    {
                        "source_file":
                            str(path),

                        "line_number":
                            line_number,

                        "error":
                            str(exc),

                        "line_character_count":
                            len(raw_line),

                        # Helpful debugging preview without writing the
                        # entire huge 32K-context row into the manifest.
                        "line_start_preview":
                            raw_line[:300],

                        "line_end_preview":
                            raw_line[-300:],
                    }
                )

                print(
                    f"[WARNING] Skipping malformed JSON row | "
                    f"{path.name} | "
                    f"line {line_number} | "
                    f"{exc}"
                )

                continue

            if not isinstance(
                row,
                dict,
            ):

                malformed_records.append(
                    {
                        "source_file":
                            str(path),

                        "line_number":
                            line_number,

                        "error":
                            "JSON value is not an object",

                        "line_character_count":
                            len(raw_line),
                    }
                )

                print(
                    f"[WARNING] Skipping non-object row | "
                    f"{path.name} | "
                    f"line {line_number}"
                )

                continue

            yield (
                line_number,
                row,
            )


# =============================================================================
# LOAD IDS PERMITTED BY ONE GUARD
# =============================================================================

def load_permitted_sample_ids(
    path: Path,
    permitted_label: str = "safe",
) -> set[str]:

    permitted_ids: set[str] = set()

    for line_number, row in read_jsonl_strict(
        path
    ):

        sample_id = str(
            row.get(
                "sample_id",
                "",
            )
        ).strip()

        if not sample_id:

            raise ValueError(
                f"Missing sample_id\n"
                f"File: {path}\n"
                f"Line: {line_number}"
            )

        classifier = row.get(
            "classifier"
        )

        if not isinstance(
            classifier,
            dict,
        ):

            raise ValueError(
                f"Missing/invalid classifier\n"
                f"File: {path}\n"
                f"Line: {line_number}"
            )

        classifier_label = str(
            classifier.get(
                "label",
                "",
            )
        ).strip().lower()

        # For unsafe benchmark prompts:
        #
        # classifier.label == safe
        #
        # means the harmful prompt passed the guard.
        if classifier_label == permitted_label:

            permitted_ids.add(
                sample_id
            )

    return permitted_ids


# =============================================================================
# FILENAMES
# =============================================================================

def source_filename(
    context_length: str,
    position: str,
    agent_tag: str,
) -> str:

    return (
        f"longctx_conversation_"
        f"{context_length}_"
        f"{position}_"
        f"{SOURCE_GUARD}_"
        f"k=1_"
        f"{agent_tag}.jsonl"
    )


def target_filename(
    context_length: str,
    position: str,
    target_guard: str,
    agent_tag: str,
) -> str:

    return (
        f"longctx_conversation_"
        f"{context_length}_"
        f"{position}_"
        f"{target_guard}_"
        f"k=1_"
        f"{agent_tag}.jsonl"
    )


# =============================================================================
# FIND APRIELGUARD SOURCE FILE
# =============================================================================

def find_source_file(
    input_dir: Path,
    filename: str,
) -> Path:

    candidates = [

        # Current structure:
        input_dir
        / filename,

        # Also support:
        # agent_run_for_analysis/aprielguard/file.jsonl
        input_dir
        / SOURCE_GUARD
        / filename,
    ]

    existing = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(existing) == 1:

        return existing[0]

    if len(existing) > 1:

        raise RuntimeError(
            "Source file exists in multiple locations:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    raise FileNotFoundError(
        "Source AprielGuard file not found.\n"
        "Checked:\n"
        + "\n".join(
            f"  {path}"
            for path in candidates
        )
    )


# =============================================================================
# DUPLICATE-ROW PRIORITY
# =============================================================================

def row_priority(
    row: dict[str, Any],
    line_number: int,
) -> tuple[int, int, int, int]:
    """
    Prefer:

        1. successful inference
        2. recognized decision
        3. no parse error
        4. later row

    This is useful when a resumed run appended a successful retry after a
    previously failed row.
    """

    successful_inference = int(
        is_blank(
            row.get(
                "inference_error"
            )
        )
    )

    decision = str(
        row.get(
            "agent_decision",
            "",
        )
    ).strip().lower()

    valid_decision = int(
        decision
        in VALID_DECISIONS
    )

    no_parse_error = int(
        is_blank(
            row.get(
                "decision_parse_error"
            )
        )
    )

    return (
        successful_inference,
        valid_decision,
        no_parse_error,
        line_number,
    )


# =============================================================================
# COPY ONE EXPERIMENT FILE
# =============================================================================

def create_target_file(
    *,
    source_file: Path,
    target_file: Path,
    target_guard: str,
    common_sample_ids: set[str],
) -> dict[str, Any]:

    malformed_records: list[
        dict[str, Any]
    ] = []

    chosen: dict[
        str,
        tuple[
            tuple[int, int, int, int],
            dict[str, Any],
        ],
    ] = {}

    source_nonblank_valid_rows = 0
    source_matching_occurrences = 0

    # =========================================================================
    # READ SOURCE TOLERANTLY
    # =========================================================================

    for (
        line_number,
        row,
    ) in read_agent_jsonl_tolerant(
        source_file,
        malformed_records,
    ):

        source_nonblank_valid_rows += 1

        sample_id = str(
            row.get(
                "sample_id",
                "",
            )
        ).strip()

        if not sample_id:

            # Valid JSON but unusable for matching.
            continue

        if sample_id not in common_sample_ids:

            continue

        source_matching_occurrences += 1

        priority = row_priority(
            row,
            line_number,
        )

        if (
            sample_id not in chosen

            or priority
            >= chosen[
                sample_id
            ][0]
        ):

            chosen[
                sample_id
            ] = (
                priority,
                row,
            )

    # =========================================================================
    # COVERAGE
    # =========================================================================

    found_ids = set(
        chosen.keys()
    )

    missing_ids = (
        common_sample_ids
        - found_ids
    )

    duplicate_matching_occurrences = (
        source_matching_occurrences
        - len(
            found_ids
        )
    )

    # =========================================================================
    # ALWAYS CREATE TARGET FILE
    # =========================================================================

    target_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_written = 0

    with target_file.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for sample_id in sorted(
            chosen.keys()
        ):

            original_row = chosen[
                sample_id
            ][1]

            # Preserve every source field.
            row = dict(
                original_row
            )

            original_input_guard = str(
                row.get(
                    "input_guard",
                    SOURCE_GUARD,
                )
            ).strip()

            # -----------------------------------------------------------------
            # Target-guard interpretation.
            # -----------------------------------------------------------------

            row[
                "original_agent_run_input_guard"
            ] = original_input_guard

            row[
                "input_guard"
            ] = target_guard

            row[
                "input_guard_allows"
            ] = True

            # -----------------------------------------------------------------
            # Explicit provenance.
            # -----------------------------------------------------------------

            row[
                "agent_inference_reused"
            ] = True

            row[
                "agent_inference_reused_from_guard"
            ] = SOURCE_GUARD

            row[
                "agent_inference_reused_for_guard"
            ] = target_guard

            row[
                "agent_inference_source_file"
            ] = source_file.name

            row[
                "pairwise_common_with_aprielguard"
            ] = True

            # -----------------------------------------------------------------
            # Leave these untouched:
            #
            # sample_id
            # long_context_id
            # root_id
            # language
            # prompt
            # agent_prompt
            # context_length_name
            # unsafe_prompt_position
            # agent_model
            # agent_decision
            # restricted_tool_selection
            # raw_agent_output
            # translation metadata
            # etc.
            # -----------------------------------------------------------------

            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            rows_written += 1

    return {

        "source_file":
            str(
                source_file
            ),

        "target_file":
            str(
                target_file
            ),

        "valid_source_rows":
            source_nonblank_valid_rows,

        "malformed_source_rows":
            len(
                malformed_records
            ),

        "malformed_source_row_details":
            malformed_records,

        "common_sample_ids_expected":
            len(
                common_sample_ids
            ),

        "common_matching_occurrences_raw":
            source_matching_occurrences,

        "duplicate_matching_occurrences_removed":
            duplicate_matching_occurrences,

        "common_sample_ids_found":
            len(
                found_ids
            ),

        "rows_written":
            rows_written,

        "missing_common_sample_ids_count":
            len(
                missing_ids
            ),

        "missing_common_sample_ids":
            sorted(
                missing_ids
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Reuse AprielGuard downstream-agent results "
            "for common sample_ids admitted by another guard."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing AprielGuard agent-run files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output root. A target-guard subdirectory is created."
        ),
    )

    parser.add_argument(
        "--guard-model",
        required=True,
        choices=sorted(
            SUPPORTED_TARGET_GUARDS
        ),
    )

    parser.add_argument(
        "--classified-root",
        default=str(DEFAULT_CLASSIFIED_ROOT),
        help=(
            "Directory containing sample_id-augmented guard files "
            "(see 04_add_sample_id_to_classified.py). "
            f"Default: {DEFAULT_CLASSIFIED_ROOT}"
        ),
    )

    parser.add_argument(
        "--permitted-label",
        default="safe",
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    target_guard = (
        args.guard_model
        .strip()
        .lower()
    )

    permitted_label = (
        args.permitted_label
        .strip()
        .lower()
    )

    CLASSIFIED_FILES = build_classified_files(
        Path(args.classified_root)
    )

    # =========================================================================
    # CLASSIFICATION FILES
    # =========================================================================

    apriel_classified = (
        CLASSIFIED_FILES[
            SOURCE_GUARD
        ]
    )

    target_classified = (
        CLASSIFIED_FILES[
            target_guard
        ]
    )

    for path in (
        apriel_classified,
        target_classified,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Classification file not found:\n"
                f"{path}"
            )

    # =========================================================================
    # COMPUTE COMMON PERMITTED IDS
    # =========================================================================

    apriel_ids = load_permitted_sample_ids(
        apriel_classified,
        permitted_label,
    )

    target_ids = load_permitted_sample_ids(
        target_classified,
        permitted_label,
    )

    common_ids = (
        apriel_ids
        & target_ids
    )

    print()
    print("=" * 96)
    print("COMMON GUARD-PERMITTED SAMPLE IDS")
    print("=" * 96)

    print(
        f"Source guard                : "
        f"{SOURCE_GUARD}"
    )

    print(
        f"Target guard                : "
        f"{target_guard}"
    )

    print(
        f"AprielGuard permitted       : "
        f"{len(apriel_ids):,}"
    )

    print(
        f"{target_guard} permitted"
        f"{' ' * max(1, 13-len(target_guard))}: "
        f"{len(target_ids):,}"
    )

    print(
        f"Common permitted sample_ids : "
        f"{len(common_ids):,}"
    )

    # =========================================================================
    # KNOWN COUNT SANITY CHECK
    # =========================================================================

    expected = EXPECTED_COMMON_COUNTS.get(
        target_guard
    )

    if (
        expected is not None
        and len(
            common_ids
        )
        != expected
    ):

        print()
        print(
            "[WARNING] Pairwise overlap differs from "
            "the previously observed value."
        )

        print(
            f"Previous value : {expected:,}"
        )

        print(
            f"Current value  : {len(common_ids):,}"
        )

    # =========================================================================
    # OUTPUT DIRECTORY
    # =========================================================================

    target_dir = (
        output_dir
        / target_guard
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # CREATE THE 12 FILES
    # =========================================================================

    manifest_files = []

    condition_index = 0

    for context_length in CONTEXT_LENGTHS:

        for position in PROMPT_POSITIONS:

            for agent_tag in AGENT_TAGS:

                condition_index += 1

                source_name = source_filename(
                    context_length,
                    position,
                    agent_tag,
                )

                target_name = target_filename(
                    context_length,
                    position,
                    target_guard,
                    agent_tag,
                )

                source_file = find_source_file(
                    input_dir,
                    source_name,
                )

                target_file = (
                    target_dir
                    / target_name
                )

                print()
                print(
                    f"[{condition_index:02d}/12] "
                    f"{source_file.name}"
                )

                result = create_target_file(
                    source_file=source_file,
                    target_file=target_file,
                    target_guard=target_guard,
                    common_sample_ids=common_ids,
                )

                manifest_files.append(
                    result
                )

                print(
                    f"    common IDs expected : "
                    f"{result['common_sample_ids_expected']:,}"
                )

                print(
                    f"    common IDs found    : "
                    f"{result['common_sample_ids_found']:,}"
                )

                print(
                    f"    rows written        : "
                    f"{result['rows_written']:,}"
                )

                print(
                    f"    malformed rows      : "
                    f"{result['malformed_source_rows']:,}"
                )

                print(
                    f"    missing common IDs  : "
                    f"{result['missing_common_sample_ids_count']:,}"
                )

                print(
                    f"    target              : "
                    f"{target_file.name}"
                )

    # =========================================================================
    # MANIFEST
    # =========================================================================

    manifest = {

        "source_guard":
            SOURCE_GUARD,

        "target_guard":
            target_guard,

        "permission_definition":
            (
                f'classifier.label == "{permitted_label}"'
            ),

        "aprielguard_permitted_sample_ids":
            len(
                apriel_ids
            ),

        "target_guard_permitted_sample_ids":
            len(
                target_ids
            ),

        "common_permitted_sample_ids":
            len(
                common_ids
            ),

        "common_sample_ids":
            sorted(
                common_ids
            ),

        "reuse_policy":
            (
                "Downstream agent inference is reused only for "
                "sample_ids admitted by both AprielGuard and target guard."
            ),

        "files":
            manifest_files,
    }

    manifest_file = (
        target_dir
        / (
            f"{SOURCE_GUARD}_"
            f"{target_guard}_"
            f"common_agent_run_manifest.json"
        )
    )

    with manifest_file.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    files_created = sum(
        1
        for result
        in manifest_files
        if Path(
            result[
                "target_file"
            ]
        ).exists()
    )

    files_complete = sum(
        1
        for result
        in manifest_files
        if (
            result[
                "rows_written"
            ]
            ==
            len(
                common_ids
            )
        )
    )

    total_malformed = sum(
        result[
            "malformed_source_rows"
        ]
        for result
        in manifest_files
    )

    print()
    print("=" * 96)
    print("COMPLETE")
    print("=" * 96)

    print(
        f"Target guard              : "
        f"{target_guard}"
    )

    print(
        f"Common sample_ids         : "
        f"{len(common_ids):,}"
    )

    print(
        f"Files created             : "
        f"{files_created}/12"
    )

    print(
        f"Fully covered files       : "
        f"{files_complete}/12"
    )

    print(
        f"Malformed source rows     : "
        f"{total_malformed:,}"
    )

    print(
        f"Output directory          : "
        f"{target_dir}"
    )

    print(
        f"Manifest                  : "
        f"{manifest_file}"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()