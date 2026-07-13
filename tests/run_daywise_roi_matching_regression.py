from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "core", _REPO_ROOT / "matching"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing
from run_daywise_roi_matching import run_daywise_roi_matching


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_expected(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_pair_schedule(output_dir: Path, expected: dict[str, object]) -> None:
    pairwise = pd.read_csv(output_dir / "pairwise_summary.csv")
    actual_pairs = [(str(row.day_a), str(row.day_b)) for row in pairwise.itertuples(index=False)]
    if [list(pair) for pair in actual_pairs] != expected.get("pair_schedule", []):
        raise AssertionError("Pair schedule does not match the expected regression baseline.")


def _compare_counts(output_dir: Path, expected: dict[str, object]) -> None:
    pairwise = pd.read_csv(output_dir / "pairwise_summary.csv")
    tracks_high = pd.read_csv(output_dir / "tracks_high.csv")
    tracks_balanced = pd.read_csv(output_dir / "tracks_balanced.csv")
    track_length = pd.read_csv(output_dir / "track_length_summary.csv")
    cycle_high = pd.read_csv(output_dir / "cycle_consistency_high.csv")
    cycle_balanced = pd.read_csv(output_dir / "cycle_consistency_balanced.csv")

    expected_counts = expected.get("expected_counts", {})
    if expected_counts:
        adjacent = pairwise.loc[pairwise["pair_gap"].astype(int) == 1].copy()
        for pair_name, expected_count in expected_counts.get("high_adjacent_edges", {}).items():
            day_a, day_b = pair_name.split("->")
            observed = int(adjacent.loc[(adjacent["day_a"].astype(str) == day_a) & (adjacent["day_b"].astype(str) == day_b), "n_high"].iloc[0])
            if observed != int(expected_count):
                raise AssertionError(f"High count mismatch for {pair_name}: {observed} != {expected_count}")
        for pair_name, expected_count in expected_counts.get("balanced_adjacent_edges", {}).items():
            day_a, day_b = pair_name.split("->")
            observed = int(adjacent.loc[(adjacent["day_a"].astype(str) == day_a) & (adjacent["day_b"].astype(str) == day_b), "n_balanced"].iloc[0])
            if observed != int(expected_count):
                raise AssertionError(f"Balanced count mismatch for {pair_name}: {observed} != {expected_count}")

        complete_high = tracks_high.loc[(tracks_high["n_days_present"].astype(int) == 7) & (tracks_high["missing_internal_days"].astype(int) == 0)]
        complete_balanced = tracks_balanced.loc[(tracks_balanced["n_days_present"].astype(int) == 7) & (tracks_balanced["missing_internal_days"].astype(int) == 0)]
        one_gap_high = tracks_high.loc[(tracks_high["n_days_present"].astype(int) == 6) & (tracks_high["missing_internal_days"].astype(int) == 1)]
        one_gap_balanced = tracks_balanced.loc[(tracks_balanced["n_days_present"].astype(int) == 6) & (tracks_balanced["missing_internal_days"].astype(int) == 1)]
        if len(complete_high) != int(expected_counts.get("high_complete_7_7", -1)):
            raise AssertionError("High 7/7 count mismatch.")
        if len(complete_balanced) != int(expected_counts.get("balanced_complete_7_7", -1)):
            raise AssertionError("Balanced 7/7 count mismatch.")
        if len(one_gap_high) != int(expected_counts.get("high_one_gap_6_7", -1)):
            raise AssertionError("High one-gap count mismatch.")
        if len(one_gap_balanced) != int(expected_counts.get("balanced_one_gap_6_7", -1)):
            raise AssertionError("Balanced one-gap count mismatch.")

    expected_cycles = expected.get("expected_cycle_counts", {})
    for policy, cycle_table in (("high", cycle_high), ("balanced", cycle_balanced)):
        for day_a, day_b, day_c, n_composed, n_comparable, n_agree, agreement in expected_cycles.get(policy, []):
            row = cycle_table.loc[
                (cycle_table["day_a"].astype(str) == day_a)
                & (cycle_table["day_b"].astype(str) == day_b)
                & (cycle_table["day_c"].astype(str) == day_c)
            ].iloc[0]
            if int(row["n_composed"]) != int(n_composed) or int(row["n_comparable"]) != int(n_comparable) or int(row["n_agree"]) != int(n_agree):
                raise AssertionError(f"Cycle counts mismatch for {policy} {day_a}->{day_b}->{day_c}")
            if abs(float(row["agreement"]) - float(agreement)) > 1e-6:
                raise AssertionError(f"Cycle agreement mismatch for {policy} {day_a}->{day_b}->{day_c}")

    for file_name, expected_hash in expected.get("expected_hashes", {}).items():
        path = output_dir / file_name.replace("(1)", "")
        if not path.exists():
            raise AssertionError(f"Missing regression output: {path}")
        observed_hash = _sha256_file(path)
        if observed_hash != expected_hash:
            raise AssertionError(f"Hash mismatch for {file_name}: {observed_hash} != {expected_hash}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", default=str(_REPO_ROOT / "tests" / "regression" / "roi_matching_7day_expected.json"))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    expected = _load_expected(Path(args.baseline_json))
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(expected.get("baseline_directory", "tests/data_private/roi_matching_7day"))
    if not dataset_dir.exists():
        print(f"Skipping regression: dataset directory not found at {dataset_dir}")
        return 0
    manifest_path = Path(args.manifest) if args.manifest else dataset_dir / str(expected.get("manifest_name", "manifest.csv"))
    if not manifest_path.exists():
        print(f"Skipping regression: manifest not found at {manifest_path}")
        return 0
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "regression_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_daywise_roi_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        spacing=VoxelSpacing(),
        params=AffineOverlapParams(),
        save_candidates=False,
        overwrite=True,
    )

    _compare_pair_schedule(output_dir, expected)
    _compare_counts(output_dir, expected)
    print("Regression comparison completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
