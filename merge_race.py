"""Merge race column from turning_interval_frequency.csv into final_cohort.csv.

Produces final_cohort_with_race.csv with exhaustive validation to ensure
the join is exact and no data is corrupted.
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_COHORT_PATH = os.path.join(SCRIPT_DIR, "data", "final_cohort.csv")
TURNING_INTERVAL_PATH = os.path.join(SCRIPT_DIR, "data", "turning_interval_frequency.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "final_cohort_with_race.csv")

FLOAT_TOL = 1e-9


def validate_inputs(fc, tif):
    """Pre-merge validation."""
    print("=" * 60)
    print("PRE-MERGE VALIDATION")
    print("=" * 60)

    assert "stay_id" in fc.columns, "final_cohort missing stay_id"
    assert "stay_id" in tif.columns, "turning_interval missing stay_id"
    assert "race" in tif.columns, "turning_interval missing race column"
    assert "race" not in fc.columns, "final_cohort already has race column"
    print("[PASS] Required columns exist")

    fc_ids = set(fc["stay_id"].unique())
    tif_ids = set(tif["stay_id"].unique())
    matched = fc_ids & tif_ids
    only_fc = fc_ids - tif_ids
    only_tif = tif_ids - fc_ids

    print(f"\n  final_cohort unique stay_ids:    {len(fc_ids):,}")
    print(f"  turning_interval unique stay_ids: {len(tif_ids):,}")
    print(f"  Matched:                          {len(matched):,}")
    print(f"  Only in final_cohort:             {len(only_fc):,}")
    print(f"  Only in turning_interval:         {len(only_tif):,}")
    print(f"  Coverage: {len(matched)/len(fc_ids)*100:.1f}% of final_cohort stay_ids")

    # Verify care interval values are identical for matched stay_ids
    print("\n  Verifying average_item_interval == turning_interval_frequency ...")
    fc_dedup = fc.drop_duplicates("stay_id")[["stay_id", "average_item_interval"]]
    tif_subset = tif[["stay_id", "turning_interval_frequency"]]
    check = fc_dedup.merge(tif_subset, on="stay_id", how="inner")
    diff = (check["average_item_interval"] - check["turning_interval_frequency"]).abs()
    n_mismatch = (diff > FLOAT_TOL).sum()

    if n_mismatch == 0:
        print(f"[PASS] All {len(check):,} matched stay_ids have identical interval values")
    else:
        print(f"[FAIL] {n_mismatch} stay_ids have mismatched interval values!")
        print(check[diff > FLOAT_TOL].head(10))
        sys.exit(1)

    # Verify race has no nulls in source
    race_nulls = tif["race"].isna().sum()
    print(f"\n  Race nulls in turning_interval: {race_nulls}")
    assert race_nulls == 0, "turning_interval has null race values"
    print("[PASS] Race column is 100% complete in source")

    # Verify turning_interval has exactly 1 row per stay_id
    tif_dupes = tif.groupby("stay_id").size()
    n_dupes = (tif_dupes > 1).sum()
    assert n_dupes == 0, f"turning_interval has {n_dupes} duplicate stay_ids"
    print("[PASS] turning_interval has exactly 1 row per stay_id")

    return matched, only_fc


def merge_race(fc, tif):
    """Left-join race from turning_interval onto final_cohort."""
    race_lookup = tif[["stay_id", "race"]].copy()
    merged = fc.merge(race_lookup, on="stay_id", how="left")
    return merged


def validate_output(merged, fc, matched_ids, unmatched_fc_ids):
    """Post-merge validation."""
    print("\n" + "=" * 60)
    print("POST-MERGE VALIDATION")
    print("=" * 60)

    # Row count unchanged
    assert len(merged) == len(fc), (
        f"Row count changed! {len(fc):,} -> {len(merged):,}"
    )
    print(f"[PASS] Row count preserved: {len(merged):,}")

    # Column count = original + 1
    assert len(merged.columns) == len(fc.columns) + 1, (
        f"Expected {len(fc.columns) + 1} cols, got {len(merged.columns)}"
    )
    print(f"[PASS] Column count: {len(merged.columns)} (original {len(fc.columns)} + race)")

    # All original columns unchanged
    for col in fc.columns:
        if fc[col].dtype in [np.float64, np.float32]:
            match = np.allclose(
                fc[col].fillna(-999).values,
                merged[col].fillna(-999).values,
                atol=FLOAT_TOL,
            )
        else:
            match = (fc[col].fillna("__NULL__") == merged[col].fillna("__NULL__")).all()
        assert match, f"Column {col} was modified during merge!"
    print("[PASS] All original columns unchanged")

    # Row order preserved
    assert (fc["stay_id"].values == merged["stay_id"].values).all(), "Row order changed!"
    print("[PASS] Row order preserved")

    # Race populated for matched stay_ids
    matched_mask = merged["stay_id"].isin(matched_ids)
    race_null_matched = merged.loc[matched_mask, "race"].isna().sum()
    assert race_null_matched == 0, f"{race_null_matched} matched rows have null race"
    print(f"[PASS] Race populated for all {matched_mask.sum():,} matched rows")

    # Race null only for unmatched stay_ids
    unmatched_mask = merged["stay_id"].isin(unmatched_fc_ids)
    n_unmatched_rows = unmatched_mask.sum()
    race_null_total = merged["race"].isna().sum()
    print(f"\n  Rows with null race: {race_null_total:,} (from {len(unmatched_fc_ids)} unmatched stay_ids)")
    assert race_null_total == n_unmatched_rows, (
        f"Unexpected null race count: {race_null_total} vs {n_unmatched_rows} unmatched rows"
    )
    print(f"[PASS] Null race count matches unmatched stay_id rows exactly")

    # Race distribution
    print(f"\n  Race distribution in merged dataset:")
    dist = merged["race"].value_counts(dropna=False)
    for race, count in dist.items():
        label = race if pd.notna(race) else "<NULL>"
        print(f"    {label:10s}: {count:6,} ({count/len(merged)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)


def main():
    print(f"Loading final_cohort:      {FINAL_COHORT_PATH}")
    fc = pd.read_csv(FINAL_COHORT_PATH)
    print(f"  -> {len(fc):,} rows, {len(fc.columns)} cols")

    print(f"Loading turning_interval:  {TURNING_INTERVAL_PATH}")
    tif = pd.read_csv(TURNING_INTERVAL_PATH)
    print(f"  -> {len(tif):,} rows, {len(tif.columns)} cols\n")

    matched_ids, unmatched_fc_ids = validate_inputs(fc, tif)

    print("\nMerging race column ...")
    merged = merge_race(fc, tif)

    validate_output(merged, fc, matched_ids, unmatched_fc_ids)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {merged.shape[0]:,} rows x {merged.shape[1]} cols")


if __name__ == "__main__":
    main()
