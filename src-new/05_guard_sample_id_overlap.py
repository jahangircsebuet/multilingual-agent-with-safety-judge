#!/usr/bin/env python3
"""
05_guard_sample_id_overlap.py

Summarize sample_id coverage and overlap across five guardrail
classification JSONL files.

Two overlap analyses are produced:

1. ALL CLASSIFIED SAMPLE IDS
   - unique sample_id count per guard
   - pairwise overlaps
   - union
   - intersection across all five guards

2. GUARD-PERMITTED SAMPLE IDS
   - only rows where:
         row["classifier"]["label"] == "safe"
   - unique admitted sample_id count per guard
   - pairwise overlaps
   - union
   - intersection across all five guards
   - number admitted by exactly 1, 2, 3, 4, or 5 guards
   - exact guard combinations
   - actual common sample_id values

IMPORTANT
---------
The guard prediction is:

    row["classifier"]["label"]

NOT:

    row["label"]

For the unsafe benchmark data:

    classifier.label == "safe"

means the harmful prompt was PERMITTED by the guard.

Output
------
A single JSON summary file.
"""

from __future__ import annotations

import argparse
import json

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


# =============================================================================
# INPUT FILES
#
# Paths are resolved relative to --input-dir (default: <repo_root>/data/
# sampledid_added_classified), so no machine-specific path is baked in here.
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "sampledid_added_classified"

GUARD_RELATIVE_FILES = {
    "aprielguard": Path("aprielguard/aprielguard_dataset_k=1.jsonl"),
    "crest": Path("crest/crest_dataset_k=1.jsonl"),
    "guardreasoner": Path("guardreasoner/guardreasoner_dataset_k=1.jsonl"),
    "wildguard": Path("wildguard/wildguard_dataset_k=1.jsonl"),
    "xguard": Path("xguard/xguard_dataset_k=1.jsonl"),
}


def build_guard_files(input_dir: Path) -> dict[str, Path]:
    return {
        guard: input_dir / relative_path
        for guard, relative_path in GUARD_RELATIVE_FILES.items()
    }


# =============================================================================
# LOAD ONE CLASSIFIED FILE
# =============================================================================

def load_guard_file(
    path: Path,
    permitted_label: str = "safe",
) -> dict[str, Any]:

    all_sample_ids: list[str] = []

    permitted_ids: set[str] = set()

    classifier_label_counts = Counter()

    total_rows = 0
    malformed_rows = 0
    missing_sample_id_rows = 0
    missing_classifier_rows = 0

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

            total_rows += 1

            # -----------------------------------------------------------------
            # Parse row.
            # -----------------------------------------------------------------

            try:
                row = json.loads(
                    line
                )

            except json.JSONDecodeError:
                malformed_rows += 1
                continue

            if not isinstance(
                row,
                dict,
            ):
                malformed_rows += 1
                continue

            # -----------------------------------------------------------------
            # sample_id
            # -----------------------------------------------------------------

            sample_id = str(
                row.get(
                    "sample_id",
                    "",
                )
            ).strip()

            if not sample_id:
                missing_sample_id_rows += 1
                continue

            all_sample_ids.append(
                sample_id
            )

            # -----------------------------------------------------------------
            # Guard prediction.
            # -----------------------------------------------------------------

            classifier = row.get(
                "classifier"
            )

            if not isinstance(
                classifier,
                dict,
            ):

                missing_classifier_rows += 1
                continue

            classifier_label = str(
                classifier.get(
                    "label",
                    "",
                )
            ).strip().lower()

            if not classifier_label:
                classifier_label = "missing"

            classifier_label_counts[
                classifier_label
            ] += 1

            # -----------------------------------------------------------------
            # For unsafe benchmark prompts:
            #
            # safe = guard permitted the harmful request.
            # -----------------------------------------------------------------

            if classifier_label == permitted_label:

                permitted_ids.add(
                    sample_id
                )

    unique_all_ids = set(
        all_sample_ids
    )

    return {

        "total_rows":
            total_rows,

        "all_ids":
            unique_all_ids,

        "permitted_ids":
            permitted_ids,

        "unique_sample_id_count":
            len(
                unique_all_ids
            ),

        "duplicate_sample_id_rows":
            (
                len(
                    all_sample_ids
                )
                -
                len(
                    unique_all_ids
                )
            ),

        "permitted_unique_sample_id_count":
            len(
                permitted_ids
            ),

        "classifier_label_counts":
            dict(
                sorted(
                    classifier_label_counts.items()
                )
            ),

        "malformed_rows":
            malformed_rows,

        "missing_sample_id_rows":
            missing_sample_id_rows,

        "missing_classifier_rows":
            missing_classifier_rows,
    }


# =============================================================================
# INTERSECTION ACROSS ALL GUARDS
# =============================================================================

def intersection_all(
    id_sets: dict[str, set[str]],
) -> set[str]:

    sets = list(
        id_sets.values()
    )

    if not sets:
        return set()

    common = set(
        sets[0]
    )

    for current_set in sets[1:]:

        common &= (
            current_set
        )

    return common


# =============================================================================
# PAIRWISE OVERLAP
# =============================================================================

def pairwise_overlap(
    id_sets: dict[str, set[str]],
) -> list[dict[str, Any]]:

    results = []

    guards = list(
        id_sets.keys()
    )

    for guard_a, guard_b in combinations(
        guards,
        2,
    ):

        ids_a = id_sets[
            guard_a
        ]

        ids_b = id_sets[
            guard_b
        ]

        intersection = (
            ids_a
            & ids_b
        )

        union = (
            ids_a
            | ids_b
        )

        results.append(
            {
                "guard_a":
                    guard_a,

                "guard_b":
                    guard_b,

                "guard_a_unique_count":
                    len(
                        ids_a
                    ),

                "guard_b_unique_count":
                    len(
                        ids_b
                    ),

                "intersection_count":
                    len(
                        intersection
                    ),

                "union_count":
                    len(
                        union
                    ),

                "jaccard":
                    (
                        len(
                            intersection
                        )
                        /
                        len(
                            union
                        )

                        if union

                        else None
                    ),

                "pct_of_guard_a_shared":
                    (
                        100.0
                        * len(
                            intersection
                        )
                        /
                        len(
                            ids_a
                        )

                        if ids_a

                        else None
                    ),

                "pct_of_guard_b_shared":
                    (
                        100.0
                        * len(
                            intersection
                        )
                        /
                        len(
                            ids_b
                        )

                        if ids_b

                        else None
                    ),

                "intersection_sample_ids":
                    sorted(
                        intersection
                    ),
            }
        )

    return results


# =============================================================================
# GUARD MEMBERSHIP PATTERN
# =============================================================================

def membership_patterns(
    id_sets: dict[str, set[str]],
) -> dict[str, Any]:

    # sample_id -> list of guards
    sample_to_guards = defaultdict(
        list
    )

    for guard, sample_ids in id_sets.items():

        for sample_id in sample_ids:

            sample_to_guards[
                sample_id
            ].append(
                guard
            )

    breadth_counts = Counter()

    exact_combination_counts = Counter()

    exact_combination_ids = defaultdict(
        list
    )

    for sample_id, guards in sample_to_guards.items():

        guards = sorted(
            guards
        )

        guard_count = len(
            guards
        )

        combination = "+".join(
            guards
        )

        breadth_counts[
            guard_count
        ] += 1

        exact_combination_counts[
            combination
        ] += 1

        exact_combination_ids[
            combination
        ].append(
            sample_id
        )

    exact_combinations = []

    for (
        combination,
        count,
    ) in sorted(
        exact_combination_counts.items(),
        key=lambda x: (
            -len(
                x[0].split(
                    "+"
                )
            ),
            -x[1],
            x[0],
        ),
    ):

        guards = combination.split(
            "+"
        )

        exact_combinations.append(
            {
                "guards":
                    guards,

                "guard_count":
                    len(
                        guards
                    ),

                "sample_count":
                    count,

                "sample_ids":
                    sorted(
                        exact_combination_ids[
                            combination
                        ]
                    ),
            }
        )

    return {

        "sample_count_by_number_of_guards": {

            str(
                guard_count
            ):
                count

            for (
                guard_count,
                count,
            ) in sorted(
                breadth_counts.items()
            )
        },

        "exact_guard_combinations":
            exact_combinations,
    }


# =============================================================================
# COMPLETE OVERLAP SUMMARY
# =============================================================================

def summarize_sets(
    id_sets: dict[str, set[str]],
) -> dict[str, Any]:

    # -------------------------------------------------------------------------
    # Union.
    # -------------------------------------------------------------------------

    union_ids: set[str] = set()

    for sample_ids in id_sets.values():

        union_ids |= (
            sample_ids
        )

    # -------------------------------------------------------------------------
    # Intersection.
    # -------------------------------------------------------------------------

    common_ids = intersection_all(
        id_sets
    )

    # -------------------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------------------

    return {

        "per_guard_unique_sample_id_counts": {

            guard:
                len(
                    sample_ids
                )

            for (
                guard,
                sample_ids,
            ) in id_sets.items()
        },

        "union_count":
            len(
                union_ids
            ),

        "union_sample_ids":
            sorted(
                union_ids
            ),

        "intersection_all_guards_count":
            len(
                common_ids
            ),

        "intersection_all_guards_sample_ids":
            sorted(
                common_ids
            ),

        "pairwise_overlap":
            pairwise_overlap(
                id_sets
            ),

        "membership_patterns":
            membership_patterns(
                id_sets
            ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Summarize sample_id overlap across "
            "guardrail classification files."
        )
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help=(
            "Single JSON output summary file."
        ),
    )

    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=(
            "Directory containing sample_id-augmented guard files "
            "(see 04_add_sample_id_to_classified.py). "
            f"Default: {DEFAULT_INPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--permitted-label",
        default="safe",
        help=(
            "classifier.label meaning the harmful prompt "
            "was permitted by the guard. Default: safe."
        ),
    )

    args = parser.parse_args()

    GUARD_FILES = build_guard_files(
        Path(args.input_dir)
    )

    permitted_label = (
        args.permitted_label
        .strip()
        .lower()
    )

    output_file = Path(
        args.output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # Validate files.
    # =========================================================================

    for guard, path in GUARD_FILES.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing input file for {guard}:\n"
                f"{path}"
            )

    # =========================================================================
    # Read classified data.
    # =========================================================================

    guard_results = {}

    print(
        "=" * 90
    )

    print(
        "GUARDRAIL SAMPLE_ID OVERLAP ANALYSIS"
    )

    print(
        "=" * 90
    )

    for guard, path in GUARD_FILES.items():

        result = load_guard_file(
            path=path,
            permitted_label=permitted_label,
        )

        guard_results[
            guard
        ] = result

        print()

        print(
            f"{guard.upper()}"
        )

        print(
            f"  total rows                 : "
            f"{result['total_rows']:,}"
        )

        print(
            f"  unique sample_id           : "
            f"{result['unique_sample_id_count']:,}"
        )

        print(
            f"  duplicate sample_id rows   : "
            f"{result['duplicate_sample_id_rows']:,}"
        )

        print(
            f"  permitted unique sample_id : "
            f"{result['permitted_unique_sample_id_count']:,}"
        )

        print(
            f"  classifier labels          : "
            f"{result['classifier_label_counts']}"
        )

    # =========================================================================
    # All classified IDs.
    # =========================================================================

    all_id_sets = {

        guard:
            result[
                "all_ids"
            ]

        for guard, result
        in guard_results.items()
    }

    # =========================================================================
    # Guard-permitted IDs.
    # =========================================================================

    permitted_id_sets = {

        guard:
            result[
                "permitted_ids"
            ]

        for guard, result
        in guard_results.items()
    }

    # =========================================================================
    # Compute overlap.
    # =========================================================================

    all_classified_summary = summarize_sets(
        all_id_sets
    )

    permitted_summary = summarize_sets(
        permitted_id_sets
    )

    # =========================================================================
    # Per-file summary.
    # =========================================================================

    file_summary = {}

    for guard, result in guard_results.items():

        file_summary[
            guard
        ] = {

            "file":
                str(
                    GUARD_FILES[
                        guard
                    ]
                ),

            "total_rows":
                result[
                    "total_rows"
                ],

            "unique_sample_id_count":
                result[
                    "unique_sample_id_count"
                ],

            "duplicate_sample_id_rows":
                result[
                    "duplicate_sample_id_rows"
                ],

            "classifier_label_counts":
                result[
                    "classifier_label_counts"
                ],

            "permitted_unique_sample_id_count":
                result[
                    "permitted_unique_sample_id_count"
                ],

            "malformed_rows":
                result[
                    "malformed_rows"
                ],

            "missing_sample_id_rows":
                result[
                    "missing_sample_id_rows"
                ],

            "missing_classifier_rows":
                result[
                    "missing_classifier_rows"
                ],
        }

    # =========================================================================
    # Common sample IDs admitted by ALL five guards.
    # =========================================================================

    common_permitted_ids = (
        permitted_summary[
            "intersection_all_guards_sample_ids"
        ]
    )

    # =========================================================================
    # Build ONE output object.
    # =========================================================================

    summary = {

        "description":
            (
                "sample_id overlap summary across "
                "five guardrail classification models"
            ),

        "guards":
            list(
                GUARD_FILES.keys()
            ),

        "guard_count":
            len(
                GUARD_FILES
            ),

        "permitted_definition":
            (
                f'classifier.label == "{permitted_label}"'
            ),

        "file_summary":
            file_summary,

        # ---------------------------------------------------------------------
        # All rows in classification files.
        # ---------------------------------------------------------------------

        "all_classified_sample_ids":
            all_classified_summary,

        # ---------------------------------------------------------------------
        # Harmful requests permitted by each guard.
        # ---------------------------------------------------------------------

        "permitted_sample_ids":
            permitted_summary,

        # ---------------------------------------------------------------------
        # Convenience field:
        # strict common downstream-analysis population.
        # ---------------------------------------------------------------------

        "common_permitted_by_all_five_guards": {

            "count":
                len(
                    common_permitted_ids
                ),

            "sample_ids":
                common_permitted_ids,
        },
    }

    # =========================================================================
    # Write SINGLE JSON file.
    # =========================================================================

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================================
    # Console report.
    # =========================================================================

    print()

    print(
        "=" * 90
    )

    print(
        "ALL CLASSIFIED SAMPLE IDS"
    )

    print(
        "=" * 90
    )

    for (
        guard,
        count,
    ) in all_classified_summary[
        "per_guard_unique_sample_id_counts"
    ].items():

        print(
            f"{guard:<20}: "
            f"{count:>8,}"
        )

    print()

    print(
        f"Union across all guards       : "
        f"{all_classified_summary['union_count']:,}"
    )

    print(
        f"Common across all five guards : "
        f"{all_classified_summary['intersection_all_guards_count']:,}"
    )

    # =========================================================================
    # Permitted sample IDs.
    # =========================================================================

    print()

    print(
        "=" * 90
    )

    print(
        "PERMITTED HARMFUL SAMPLE IDS"
    )

    print(
        f'Definition: classifier.label == "{permitted_label}"'
    )

    print(
        "=" * 90
    )

    for (
        guard,
        count,
    ) in permitted_summary[
        "per_guard_unique_sample_id_counts"
    ].items():

        print(
            f"{guard:<20}: "
            f"{count:>8,}"
        )

    print()

    print(
        f"Union of permitted IDs        : "
        f"{permitted_summary['union_count']:,}"
    )

    print(
        f"Common permitted by all five  : "
        f"{permitted_summary['intersection_all_guards_count']:,}"
    )

    print()

    print(
        "Number of harmful samples admitted by N guards:"
    )

    for (
        number_of_guards,
        count,
    ) in permitted_summary[
        "membership_patterns"
    ][
        "sample_count_by_number_of_guards"
    ].items():

        print(
            f"  admitted by exactly "
            f"{number_of_guards} guard(s): "
            f"{count:,}"
        )

    # =========================================================================
    # Pairwise overlaps.
    # =========================================================================

    print()

    print(
        "PAIRWISE PERMITTED OVERLAP"
    )

    print(
        "-" * 90
    )

    for row in permitted_summary[
        "pairwise_overlap"
    ]:

        print(
            f"{row['guard_a']:<15} "
            f"<-> "
            f"{row['guard_b']:<15} "
            f": {row['intersection_count']:>6,}"
        )

    print()

    print(
        "=" * 90
    )

    print(
        f"Output written to:\n{output_file}"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":

    main()