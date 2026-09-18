#!/usr/bin/env python3
"""
07_filter_long_context_data.py

Filter long-context data using the predictions of ONE selected guard model.

IMPORTANT
=========
The permission decision is based ONLY on:

    row["classifier"]["label"]

NOT on:

    row["label"]

The top-level `label` is the ground-truth dataset label and is typically
"unsafe" for these benchmark prompts.

Guard interpretation
--------------------
classifier.label == "safe"
    -> harmful prompt was PERMITTED by the guard
    -> include this sample_id

classifier.label == "unsafe"
    -> harmful prompt was BLOCKED by the guard
    -> exclude this sample_id


Workflow
--------
1. Select one guard directory with --model.
2. Find its k-specific evaluated JSONL file.
3. Read classifier.label for every row.
4. Collect sample_id only when classifier.label == "safe".
5. Filter all 12 long-context files by those sample_id values.
6. Write COMPLETE rows from the long-context files.

Example
-------
python 07_filter_long_context_data.py \
    --classified-dir /path/to/sampledid_added_classified \
    --long-context-dir /path/to/long_context \
    --model llamaguard_permissive \
    --k 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


# ============================================================
# JSONL READER
# ============================================================

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Read a JSONL file one row at a time."""

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON\n"
                    f"File: {path}\n"
                    f"Line: {line_number}\n"
                    f"Error: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object\n"
                    f"File: {path}\n"
                    f"Line: {line_number}"
                )

            yield row


# ============================================================
# FIND THE SELECTED GUARD FILE
# ============================================================

def find_guard_file(
    classified_dir: Path,
    model_name: str,
    k: int,
) -> Path:
    """
    Find the requested k-specific JSONL file inside one guard directory.

    Example:

    sampledid_added_classified/
    └── llamaguard_permissive/
        ├── llamaguard_permissive_dataset_k=1.jsonl
        └── llamaguard_permissive_dataset_k=2.jsonl
    """

    model_dir = classified_dir / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist:\n{model_dir}"
        )

    if not model_dir.is_dir():
        raise NotADirectoryError(
            f"Expected model directory:\n{model_dir}"
        )

    candidates = sorted(
        model_dir.rglob(f"*k={k}*.jsonl")
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No k={k} JSONL file found for model "
            f"'{model_name}' in:\n{model_dir}"
        )

    if len(candidates) > 1:

        files = "\n".join(
            f"  - {path}"
            for path in candidates
        )

        raise RuntimeError(
            f"Multiple k={k} JSONL files found for "
            f"'{model_name}'. Expected exactly one:\n"
            f"{files}"
        )

    return candidates[0]


# ============================================================
# EXTRACT GUARD PREDICTION
# ============================================================

def get_classifier_label(
    row: Dict[str, Any],
    file_path: Path,
    row_number: int,
) -> str:
    """
    Return the label produced by the guard model.

    IMPORTANT:
    This function intentionally reads:

        row["classifier"]["label"]

    It NEVER uses the top-level row["label"].
    """

    if "classifier" not in row:
        raise KeyError(
            f"Missing 'classifier' field\n"
            f"File: {file_path}\n"
            f"Row: {row_number}"
        )

    classifier = row["classifier"]

    if not isinstance(classifier, dict):
        raise TypeError(
            f"'classifier' must be an object/dictionary\n"
            f"File: {file_path}\n"
            f"Row: {row_number}"
        )

    if "label" not in classifier:
        raise KeyError(
            f"Missing 'classifier.label'\n"
            f"File: {file_path}\n"
            f"Row: {row_number}"
        )

    label = str(
        classifier["label"]
    ).strip().lower()

    if not label:
        raise ValueError(
            f"Empty classifier.label\n"
            f"File: {file_path}\n"
            f"Row: {row_number}"
        )

    return label


# ============================================================
# LOAD PERMITTED SAMPLE IDS
# ============================================================

def load_permitted_sample_ids(
    guard_file: Path,
    permitted_label: str = "safe",
) -> Dict[str, Any]:
    """
    Collect sample_ids whose GUARD prediction is "safe".

    Example source row:

        "label": "unsafe",             <-- gold/reference label

        "classifier": {
            "classifier_name": "...",
            "label": "safe"            <-- guard prediction
        }

    This row IS INCLUDED because classifier.label == "safe".
    """

    permitted_ids: Set[str] = set()

    total_rows = 0
    permitted_rows = 0
    blocked_rows = 0

    observed_classifier_labels: Dict[str, int] = {}

    permitted_label = (
        permitted_label.strip().lower()
    )

    for row_number, row in enumerate(
        read_jsonl(guard_file),
        start=1,
    ):

        total_rows += 1

        # ----------------------------------------------------
        # sample_id
        # ----------------------------------------------------

        if "sample_id" not in row:
            raise KeyError(
                f"Missing sample_id\n"
                f"File: {guard_file}\n"
                f"Row: {row_number}"
            )

        sample_id = str(
            row["sample_id"]
        ).strip()

        if not sample_id:
            raise ValueError(
                f"Empty sample_id\n"
                f"File: {guard_file}\n"
                f"Row: {row_number}"
            )

        # ----------------------------------------------------
        # GUARD prediction
        #
        # THIS IS THE IMPORTANT LINE.
        # ----------------------------------------------------

        classifier_label = get_classifier_label(
            row=row,
            file_path=guard_file,
            row_number=row_number,
        )

        observed_classifier_labels[
            classifier_label
        ] = (
            observed_classifier_labels.get(
                classifier_label,
                0,
            )
            + 1
        )

        # ----------------------------------------------------
        # PERMITTED?
        #
        # safe = harmful request passed the guard
        # ----------------------------------------------------

        if classifier_label == permitted_label:

            permitted_ids.add(sample_id)
            permitted_rows += 1

        else:

            blocked_rows += 1

    return {
        "sample_ids": permitted_ids,
        "total_rows": total_rows,
        "permitted_rows": permitted_rows,
        "blocked_rows": blocked_rows,
        "unique_permitted_ids": len(permitted_ids),
        "observed_classifier_labels":
            observed_classifier_labels,
    }


# ============================================================
# FIND LONG CONTEXT FILES
# ============================================================

def find_long_context_files(
    long_context_dir: Path,
) -> List[Path]:
    """Find all 12 longctx_*.jsonl source files."""

    files = sorted(
        long_context_dir.glob(
            "longctx_*.jsonl"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No longctx_*.jsonl files found in:\n"
            f"{long_context_dir}"
        )

    return files


# ============================================================
# OUTPUT FILE NAME
# ============================================================

def make_output_filename(
    source_file: Path,
    model_name: str,
    k: int,
) -> str:
    """
    Example:

    longctx_conversation_8k_beginning.jsonl

    ->

    longctx_conversation_8k_beginning_
    llamaguard_permissive_k=1.jsonl
    """

    return (
        f"{source_file.stem}_"
        f"{model_name}_"
        f"k={k}.jsonl"
    )


# ============================================================
# FILTER ONE LONG CONTEXT FILE
# ============================================================

def filter_long_context_file(
    source_file: Path,
    output_file: Path,
    permitted_sample_ids: Set[str],
) -> Dict[str, Any]:
    """
    Filter a long-context file using permitted sample_id values.

    Output records come ENTIRELY from the long-context file.
    """

    total_rows = 0
    filtered_rows = 0

    matched_ids: Set[str] = set()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for row_number, row in enumerate(
            read_jsonl(source_file),
            start=1,
        ):

            total_rows += 1

            if "sample_id" not in row:
                raise KeyError(
                    f"Missing sample_id\n"
                    f"File: {source_file}\n"
                    f"Row: {row_number}"
                )

            sample_id = str(
                row["sample_id"]
            ).strip()

            # ------------------------------------------------
            # Keep only prompts permitted by the guard.
            # ------------------------------------------------

            if sample_id in permitted_sample_ids:

                fout.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                filtered_rows += 1
                matched_ids.add(sample_id)

    missing_ids = (
        permitted_sample_ids
        - matched_ids
    )

    return {
        "source_rows": total_rows,
        "filtered_rows": filtered_rows,
        "matched_ids": len(matched_ids),
        "missing_ids": len(missing_ids),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(args: argparse.Namespace) -> None:

    classified_dir = Path(
        args.classified_dir
    )

    long_context_dir = Path(
        args.long_context_dir
    )

    # Default output:
    #
    # data/
    # ├── long_context/
    # └── filtered_long_context/
    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

    else:

        output_dir = (
            long_context_dir.parent
            / "filtered_long_context"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Find selected guard file
    # ========================================================

    guard_file = find_guard_file(
        classified_dir=classified_dir,
        model_name=args.model,
        k=args.k,
    )

    # ========================================================
    # Determine permitted sample_ids
    #
    # ONLY classifier.label is checked here.
    # ========================================================

    guard_results = (
        load_permitted_sample_ids(
            guard_file=guard_file,
            permitted_label=args.permitted_label,
        )
    )

    permitted_ids = (
        guard_results["sample_ids"]
    )

    # ========================================================
    # Find 12 long-context conditions
    # ========================================================

    long_context_files = (
        find_long_context_files(
            long_context_dir
        )
    )

    # ========================================================
    # Print guard statistics
    # ========================================================

    print("=" * 80)
    print("GUARD FILTERING CONFIGURATION")
    print("=" * 80)

    print(
        f"Guard/model              : {args.model}"
    )

    print(
        f"k                        : {args.k}"
    )

    print(
        f"Guard source file        : {guard_file}"
    )

    print(
        f"Permission field         : classifier.label"
    )

    print(
        f"Permitted label          : {args.permitted_label}"
    )

    print(
        f"Guard evaluated rows     : "
        f"{guard_results['total_rows']:,}"
    )

    print(
        f"Permitted rows           : "
        f"{guard_results['permitted_rows']:,}"
    )

    print(
        f"Blocked/other rows       : "
        f"{guard_results['blocked_rows']:,}"
    )

    print(
        f"Unique permitted IDs     : "
        f"{guard_results['unique_permitted_ids']:,}"
    )

    print(
        "Classifier labels seen  : "
        f"{guard_results['observed_classifier_labels']}"
    )

    print(
        f"Long-context files       : "
        f"{len(long_context_files)}"
    )

    print(
        f"Output directory         : "
        f"{output_dir}"
    )

    print("=" * 80)

    # ========================================================
    # Filter all 12 files
    # ========================================================

    manifest_outputs = []

    for index, source_file in enumerate(
        long_context_files,
        start=1,
    ):

        output_filename = (
            make_output_filename(
                source_file=source_file,
                model_name=args.model,
                k=args.k,
            )
        )

        output_file = (
            output_dir
            / output_filename
        )

        result = filter_long_context_file(
            source_file=source_file,
            output_file=output_file,
            permitted_sample_ids=permitted_ids,
        )

        print(
            f"[{index:02d}/{len(long_context_files):02d}] "
            f"{source_file.name}"
        )

        print(
            f"    source rows   : "
            f"{result['source_rows']:,}"
        )

        print(
            f"    filtered rows : "
            f"{result['filtered_rows']:,}"
        )

        print(
            f"    missing IDs   : "
            f"{result['missing_ids']:,}"
        )

        print(
            f"    output        : "
            f"{output_file.name}"
        )

        manifest_outputs.append(
            {
                "source_file":
                    str(source_file),

                "output_file":
                    str(output_file),

                "source_rows":
                    result["source_rows"],

                "filtered_rows":
                    result["filtered_rows"],

                "missing_permitted_ids":
                    result["missing_ids"],
            }
        )

    # ========================================================
    # WRITE MANIFEST
    # ========================================================

    manifest = {
        "model":
            args.model,

        "k":
            args.k,

        "guard_file":
            str(guard_file),

        "permission_field":
            "classifier.label",

        "permitted_label":
            args.permitted_label,

        "guard_total_rows":
            guard_results["total_rows"],

        "permitted_rows":
            guard_results["permitted_rows"],

        "unique_permitted_ids":
            guard_results[
                "unique_permitted_ids"
            ],

        "observed_classifier_labels":
            guard_results[
                "observed_classifier_labels"
            ],

        "long_context_file_count":
            len(long_context_files),

        "outputs":
            manifest_outputs,
    }

    manifest_file = (
        output_dir
        / (
            f"filtered_long_context_"
            f"{args.model}_"
            f"k={args.k}_manifest.json"
        )
    )

    with manifest_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)

    print(
        f"Guard                     : {args.model}"
    )

    print(
        f"Permission evaluated from : classifier.label"
    )

    print(
        f"Permitted label           : {args.permitted_label}"
    )

    print(
        f"Files generated           : "
        f"{len(long_context_files)}"
    )

    print(
        f"Output directory          : "
        f"{output_dir}"
    )

    print(
        f"Manifest                  : "
        f"{manifest_file}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Filter long-context instances using "
            "classifier.label from one guard model."
        )
    )

    parser.add_argument(
        "--classified-dir",
        required=True,
        help=(
            "Root directory containing guard-evaluated "
            "files with sample_id."
        ),
    )

    parser.add_argument(
        "--long-context-dir",
        required=True,
        help=(
            "Directory containing the 12 "
            "long-context JSONL files."
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Guard directory name, e.g. "
            "llamaguard_permissive, nemotron, "
            "qwen3guard, wildguard."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Subset k value. Default: 1.",
    )

    parser.add_argument(
        "--permitted-label",
        default="safe",
        help=(
            "Value of classifier.label meaning "
            "the guard permitted the prompt. "
            "Default: safe."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. "
            "Default: sibling directory "
            "filtered_long_context."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    run(args)