#!/usr/bin/env python3
"""Prepare the modeling table without modifying the source CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = Path("data/processed/japanese_sensory_modeling.csv")
SPLIT_SEED = 20261007


def assign_grouped_folds(target: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Series:
    """Assign provisional 3-fold IDs while keeping candidate families together."""
    group_sizes = target.groupby("family_candidate_key", dropna=False).size().reset_index(name="n")
    rng = np.random.default_rng(seed)
    group_sizes["tie_break"] = rng.random(len(group_sizes))
    group_sizes = group_sizes.sort_values(["n", "tie_break"], ascending=[False, True])

    fold_loads = {1: 0, 2: 0, 3: 0}
    assignment: dict[str, int] = {}
    for row in group_sizes.itertuples(index=False):
        fold = min(fold_loads, key=lambda value: (fold_loads[value], value))
        assignment[row.family_candidate_key] = fold
        fold_loads[fold] += int(row.n)

    return target["family_candidate_key"].map(assignment).astype("Int64")


def prepare(input_path: Path, output_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path)

    required = {
        "word_id",
        "is_ideophone",
        "rating_review_status",
        "rating_issue_codes",
        "family_candidate_key",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = data["rating_review_status"].eq("no_local_issue_detected")
    data["use_in_primary_analysis"] = clean.astype(int)
    data["primary_exclusion_reason"] = np.where(
        clean,
        "",
        data["rating_issue_codes"].fillna("pending_data_review"),
    )
    data["analysis_role"] = np.select(
        [~clean, data["is_ideophone"].eq(1)],
        ["excluded_pending_review", "ideophone_target"],
        default="ordinary_source",
    )

    data["outer_fold_provisional"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    target_mask = clean & data["is_ideophone"].eq(1)
    data.loc[target_mask, "outer_fold_provisional"] = assign_grouped_folds(data.loc[target_mask])

    data["split_review_status"] = np.select(
        [~clean, target_mask],
        ["excluded", "pending_family_confirmation"],
        default="not_applicable",
    )

    if int(clean.sum()) != 504:
        raise ValueError(f"Expected 504 primary-analysis words, found {int(clean.sum())}")
    counts = data.loc[clean].groupby("is_ideophone").size().to_dict()
    if counts != {0: 396, 1: 108}:
        raise ValueError(f"Unexpected primary-analysis category counts: {counts}")

    fold_counts = data.loc[target_mask, "outer_fold_provisional"].value_counts().sort_index()
    if fold_counts.sum() != 108 or len(fold_counts) != 3:
        raise ValueError(f"Unexpected fold counts: {fold_counts.to_dict()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Saved {len(data)} rows to {output_path}")
    print(f"Primary analysis: {int(clean.sum())} rows")
    print(f"Fold counts: {fold_counts.to_dict()}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()

