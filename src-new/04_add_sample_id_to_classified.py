#!/usr/bin/env python3
"""
04_add_sample_id_to_classified.py

Recursively add `sample_id` to every JSONL file inside a classified
results directory while preserving the complete directory structure.

The generated sample_id uses exactly the same definition used by the
long-context generation pipeline:

    sample_id = root_id + language + SHA256(root_id || language || prompt)

More precisely:

    key = f"{root_id}||{language}||{prompt}"
    digest = SHA256(key)[:16]

Example input structure
-----------------------
data/classified/
├── aprielguard/
│   ├── aprielguard_dataset_k=1.jsonl
│   └── aprielguard_dataset_k=2.jsonl
├── crest/
│   ├── crest_dataset_k=1.jsonl
│   └── crest_dataset_k=2.jsonl
├── nemotron/
│   └── ...
└── wildguard/
    └── ...

Example output structure
------------------------
data/sampledid_added_classified/
├── aprielguard/
│   ├── aprielguard_dataset_k=1.jsonl
│   └── aprielguard_dataset_k=2.jsonl
├── crest/
│   ├── crest_dataset_k=1.jsonl
│   └── crest_dataset_k=2.jsonl
├── nemotron/
│   └── ...
└── wildguard/
    └── ...

Usage
-----
python add_sample_id_to_classified_directory.py \
    --input-dir data/classified \
    --output-dir data/sampledid_added_classified
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any


# ============================================================
# SAMPLE ID GENERATION
# ============================================================

def make_sample_id(
    root_id: str,
    language: str,
    prompt: str,
) -> str:
    """
    Create the same deterministic sample_id used in the
    long-context generation script.

    Identity:
        root_id + language + prompt

    Parameters
    ----------
    root_id:
        Original root prompt identifier.

    language:
        Language of the multilingual prompt.

    prompt:
        Multilingual prompt text.

    Returns
    -------
    str
        Deterministic sample identifier.
    """

    # Apply exactly the same normalization used in the
    # long-context generation code.
    root_id = str(root_id).strip()
    language = str(language).strip()
    prompt = str(prompt).strip()

    key = (
        f"{root_id}||"
        f"{language}||"
        f"{prompt}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()[:16]

    # Keep the language readable inside the ID.
    safe_language = (
        language
        .replace(" ", "_")
        .replace("/", "_")
    )

    return (
        f"{root_id}__"
        f"{safe_language}__"
        f"{digest}"
    )


# ============================================================
# PROCESS ONE JSONL FILE
# ============================================================

def process_jsonl_file(
    input_path: Path,
    output_path: Path,
) -> Dict[str, int]:
    """
    Add sample_id to every row of one JSONL file.

    All existing fields are preserved unchanged.

    Parameters
    ----------
    input_path:
        Source JSONL file.

    output_path:
        Destination JSONL file.

    Returns
    -------
    dict
        Processing statistics.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_processed = 0
    existing_sample_ids = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as fin, output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for line_number, line in enumerate(
            fin,
            start=1,
        ):

            line = line.strip()

            # Ignore blank lines.
            if not line:
                continue

            # ------------------------------------------------
            # Parse JSON row.
            # ------------------------------------------------

            try:
                row: Dict[str, Any] = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in file:\n"
                    f"  {input_path}\n"
                    f"Line: {line_number}\n"
                    f"Error: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in:\n"
                    f"  {input_path}\n"
                    f"Line: {line_number}"
                )

            # ------------------------------------------------
            # Check required fields.
            # ------------------------------------------------

            required_fields = [
                "root_id",
                "language",
                "prompt",
            ]

            missing_fields = [
                field
                for field in required_fields
                if field not in row
            ]

            if missing_fields:
                raise KeyError(
                    f"Missing required field(s) "
                    f"{missing_fields}\n"
                    f"File: {input_path}\n"
                    f"Line: {line_number}"
                )

            # ------------------------------------------------
            # Generate sample_id.
            # ------------------------------------------------

            generated_sample_id = make_sample_id(
                root_id=row["root_id"],
                language=row["language"],
                prompt=row["prompt"],
            )

            # If sample_id already exists, we still recompute it
            # using the canonical function. This guarantees that
            # all files follow the exact same definition.
            if "sample_id" in row:
                existing_sample_ids += 1

            row["sample_id"] = generated_sample_id

            # ------------------------------------------------
            # Write complete row.
            # ------------------------------------------------

            fout.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            rows_processed += 1

    return {
        "rows_processed": rows_processed,
        "existing_sample_ids": existing_sample_ids,
    }


# ============================================================
# PROCESS COMPLETE DIRECTORY
# ============================================================

def process_directory(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """
    Recursively find every .jsonl file in input_dir,
    add sample_id, and recreate the same relative path
    under output_dir.
    """

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory:\n"
            f"{input_dir}"
        )

    # Resolve paths so we can protect against accidentally
    # writing into the same directory.
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()

    if input_resolved == output_resolved:
        raise ValueError(
            "Input and output directories must be different."
        )

    # --------------------------------------------------------
    # Find every JSONL file recursively.
    # --------------------------------------------------------

    jsonl_files = sorted(
        input_dir.rglob("*.jsonl")
    )

    if not jsonl_files:
        print(
            f"[WARNING] No .jsonl files found under "
            f"{input_dir}"
        )
        return

    print("=" * 72)
    print("ADDING sample_id TO CLASSIFIED DATA")
    print("=" * 72)

    print(
        f"Input directory : {input_dir}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"JSONL files found: {len(jsonl_files):,}"
    )

    print("=" * 72)

    total_rows = 0
    total_existing_ids = 0
    processed_files = 0

    # --------------------------------------------------------
    # Process each file.
    # --------------------------------------------------------

    for file_index, input_path in enumerate(
        jsonl_files,
        start=1,
    ):

        # Example:
        #
        # input_path:
        #   data/classified/aprielguard/file.jsonl
        #
        # relative_path:
        #   aprielguard/file.jsonl
        #
        # output_path:
        #   data/sampledid_added_classified/
        #   aprielguard/file.jsonl

        relative_path = input_path.relative_to(
            input_dir
        )

        output_path = (
            output_dir
            / relative_path
        )

        print()
        print(
            f"[{file_index}/{len(jsonl_files)}] "
            f"{relative_path}"
        )

        stats = process_jsonl_file(
            input_path=input_path,
            output_path=output_path,
        )

        rows_processed = stats[
            "rows_processed"
        ]

        existing_ids = stats[
            "existing_sample_ids"
        ]

        total_rows += rows_processed
        total_existing_ids += existing_ids
        processed_files += 1

        print(
            f"    Rows processed : "
            f"{rows_processed:,}"
        )

        print(
            f"    Output         : "
            f"{output_path}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("PROCESSING COMPLETE")
    print("=" * 72)

    print(
        f"Files processed          : "
        f"{processed_files:,}"
    )

    print(
        f"Rows processed           : "
        f"{total_rows:,}"
    )

    print(
        f"Existing IDs overwritten : "
        f"{total_existing_ids:,}"
    )

    print(
        f"Output directory         : "
        f"{output_dir}"
    )

    print("=" * 72)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Recursively add sample_id to every JSONL "
            "file in a classified results directory "
            "while preserving the complete folder structure."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Root classified-data directory. "
            "Example: data/classified"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Destination root directory. "
            "Example: data/sampledid_added_classified"
        ),
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    process_directory(
        input_dir=Path(
            args.input_dir
        ),
        output_dir=Path(
            args.output_dir
        ),
    )