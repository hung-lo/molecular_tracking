"""CLI and orchestration for conservative post-hoc endpoint stitching."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matching", _REPO_ROOT / "plotting", _REPO_ROOT / "core"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from endpoint_evaluator import _read_csv, _read_json, _session_table, load_matcher_spacing
from endpoint_stitcher import (
    ALGORITHM_VERSION, StitcherConfig, assign_stitches,
    benchmark_threshold_sweep, build_stitch_candidates, build_stitched_tracks,
    build_synthetic_stitch_benchmark, edge_table, eligible_source_endpoint_ids,
    select_review_samples, summarize_stitch_benchmark,
)
from endpoint_stitch_review_plots import plot_stitch_review_panels


REQUIRED_MATCH_FILES = (
    "session_manifest_resolved.csv", "roi_features.csv", "pairwise_transforms.csv",
    "tracks_{policy}.csv", "track_edges_{policy}.csv",
)
REQUIRED_EVALUATION_FILES = (
    "endpoint_events.csv", "endpoint_candidates.csv", "endpoint_classification.csv",
    "synthetic_gap_benchmark.csv",
)


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _prepare_output(output_dir: Path, overwrite: bool, policy: str) -> pd.DataFrame:
    previous = _read_csv(output_dir / "manual_stitch_review_manifest.csv") if output_dir.exists() else pd.DataFrame()
    known = {
        "stitch_candidates.csv", "stitch_proposals.csv", "stitch_assignments.csv",
        "track_stitch_edges.csv", "track_uid_stitch_map.csv",
        "stitch_summary.json", "stitch_run_log.json", "synthetic_stitch_benchmark.csv",
        "synthetic_stitch_threshold_sweep.csv", "manual_stitch_review_manifest.csv",
    }
    known.update(f"tracks_{value}_stitched.csv" for value in ("graph", "balanced", "high"))
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in known:
            path = output_dir / name
            if path.is_file():
                path.unlink()
        panel_dir = output_dir / "review_panels"
        if panel_dir.is_dir():
            for path in panel_dir.glob("*.png"):
                path.unlink()
    return previous


def _manual_manifest(proposals: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "stitch_edge_id", "source_track_uid", "source_session_index", "source_label",
        "target_track_uid", "target_session_index", "target_label", "session_gap",
        "candidate_tier", "assignment_status", "review_sample_reason", "review_reasons",
        "manual_class", "manual_confidence", "reviewer_notes", "manual_label_transfer_warning",
    ]
    selected = proposals.loc[proposals.get("candidate_tier", pd.Series(index=proposals.index, dtype=str)).ne("reject")].copy()
    for column in ("manual_class", "manual_confidence", "reviewer_notes"):
        selected[column] = ""
    output = selected.reindex(columns=columns)
    output["review_sample_reason"] = output["review_sample_reason"].fillna("")
    output["manual_label_transfer_warning"] = ""
    identity_columns = ["source_track_uid", "source_session_index", "source_label", "target_track_uid", "target_session_index", "target_label"]
    if not previous.empty and "stitch_edge_id" in previous:
        old = previous.drop_duplicates("stitch_edge_id").set_index("stitch_edge_id")
        for index, edge_id in output["stitch_edge_id"].items():
            if edge_id not in old.index:
                continue
            matches = all(column in old.columns and str(output.at[index, column]) == str(old.at[edge_id, column]) for column in identity_columns)
            if not matches:
                output.at[index, "manual_label_transfer_warning"] = "identity_mismatch_not_restored"
                continue
            for column in ("manual_class", "manual_confidence", "reviewer_notes"):
                value = old.at[edge_id, column] if column in old else ""
                if pd.notna(value) and str(value).strip():
                    output.at[index, column] = value
    return output


def run_endpoint_stitching(
    match_dir: str | Path,
    evaluation_dir: str | Path,
    output_dir: str | Path,
    *,
    policy: str = "graph",
    max_gap_sessions: int = 3,
    search_radius_um: float = 15.0,
    profile: str = "conservative_v1",
    benchmark_replicates: int = 20,
    benchmark_cases_per_gap: int = 50,
    random_seed: int = 0,
    max_review_panels: int | None = 100,
    overwrite: bool = False,
    proposal_only: bool = False,
    write_stitched_tracks: bool = False,
    state_csv: str | Path | None = None,
    extraction_dir: str | Path | None = None,
    min_benchmark_precision: float = 0.999,
    max_negative_fpr: float = 0.001,
    force_write_despite_benchmark_failure: bool = False,
) -> dict[str, Any]:
    match_dir, evaluation_dir, output_dir = map(lambda value: Path(value).resolve(), (match_dir, evaluation_dir, output_dir))
    if profile != "conservative_v1":
        raise ValueError("profile must be conservative_v1")
    if policy not in {"graph", "balanced", "high"}:
        raise ValueError("policy must be graph, balanced, or high")
    if not 1 <= max_gap_sessions <= 3 or search_radius_um <= 0 or benchmark_replicates < 0:
        raise ValueError("max_gap_sessions must be 1..3, search radius positive, and benchmark replicates nonnegative")
    if benchmark_cases_per_gap < 0:
        raise ValueError("benchmark cases per gap must be nonnegative")
    if not 0 <= min_benchmark_precision <= 1 or not 0 <= max_negative_fpr <= 1:
        raise ValueError("benchmark precision/FPR guardrails must be between 0 and 1")
    if match_dir in {evaluation_dir, output_dir} or evaluation_dir == output_dir or output_dir.is_relative_to(match_dir) or output_dir.is_relative_to(evaluation_dir):
        raise ValueError("match, evaluation, and output directories must be separate")
    config = StitcherConfig(max_gap_sessions=max_gap_sessions, search_radius_um=search_radius_um, benchmark_cases_per_gap=benchmark_cases_per_gap)
    match_paths = [match_dir / name.format(policy=policy) for name in REQUIRED_MATCH_FILES]
    evaluation_paths = [evaluation_dir / name for name in REQUIRED_EVALUATION_FILES]
    missing = [str(path) for path in match_paths + evaluation_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing stitcher inputs: " + ", ".join(missing))
    if state_csv is not None and not Path(state_csv).is_file():
        raise FileNotFoundError(f"state CSV was not found: {state_csv}")
    if extraction_dir is not None and not Path(extraction_dir).is_dir():
        raise FileNotFoundError(f"extraction directory was not found: {extraction_dir}")
    previous_manual = _prepare_output(output_dir, overwrite, policy)
    total_start = time.perf_counter()
    canonical_paths = sorted(path for path in match_dir.iterdir() if path.is_file())
    canonical_before = _hashes(canonical_paths)
    sessions = _session_table(_read_csv(match_dir / "session_manifest_resolved.csv"))
    features = _read_csv(match_dir / "roi_features.csv")
    transforms = _read_csv(match_dir / "pairwise_transforms.csv")
    tracks = _read_csv(match_dir / f"tracks_{policy}.csv")
    endpoints = _read_csv(evaluation_dir / "endpoint_events.csv")
    evaluator_candidates = _read_csv(evaluation_dir / "endpoint_candidates.csv")
    classifications = _read_csv(evaluation_dir / "endpoint_classification.csv")
    spacing, spacing_source = load_matcher_spacing(match_dir)
    load_inputs_sec = time.perf_counter() - total_start
    source_endpoint_ids = eligible_source_endpoint_ids(endpoints, tracks, sessions, policy=policy)

    stage_start = time.perf_counter()
    candidates = build_stitch_candidates(
        endpoints, evaluator_candidates, classifications, tracks, features, sessions, transforms,
        policy=policy, spacing=spacing, config=config,
    )
    build_candidates_sec = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    assignments = assign_stitches(candidates, config=config)
    global_assignment_sec = time.perf_counter() - stage_start
    proposals = assignments.loc[assignments.get("candidate_tier", pd.Series(index=assignments.index, dtype=str)).ne("reject")].copy()
    stage_start = time.perf_counter()
    benchmark = build_synthetic_stitch_benchmark(
        tracks, features, sessions, transforms, policy=policy, spacing=spacing, config=config,
        replicates=benchmark_replicates, cases_per_gap=config.benchmark_cases_per_gap, random_seed=random_seed,
    )
    benchmark_sec = time.perf_counter() - stage_start
    benchmark_metrics = summarize_stitch_benchmark(
        benchmark, min_precision=min_benchmark_precision, max_negative_fpr=max_negative_fpr,
        min_positive_cases=config.min_benchmark_positive_cases,
        min_negative_controls=config.min_benchmark_negative_controls,
        min_cases_per_gap=config.min_benchmark_cases_per_gap,
    )
    sweep = benchmark_threshold_sweep(benchmark, config=config)
    stage_start = time.perf_counter()
    stitched, uid_map, invariants = build_stitched_tracks(tracks, assignments, sessions, policy=policy)
    build_stitched_view_sec = time.perf_counter() - stage_start
    allowed_to_write = bool(benchmark_metrics["guardrail_passed"] or force_write_despite_benchmark_failure)
    wrote_stitched = bool(write_stitched_tracks and not proposal_only and allowed_to_write)
    review = select_review_samples(assignments, max_panels=max_review_panels, random_seed=random_seed, config=config)

    candidates.to_csv(output_dir / "stitch_candidates.csv", index=False)
    proposals.to_csv(output_dir / "stitch_proposals.csv", index=False)
    assignments.to_csv(output_dir / "stitch_assignments.csv", index=False)
    edge_table(assignments).to_csv(output_dir / "track_stitch_edges.csv", index=False)
    benchmark.to_csv(output_dir / "synthetic_stitch_benchmark.csv", index=False)
    sweep.to_csv(output_dir / "synthetic_stitch_threshold_sweep.csv", index=False)
    manual = _manual_manifest(assignments, previous_manual)
    if not review.empty and "stitch_edge_id" in review:
        sampled_reasons = review.set_index("stitch_edge_id")["review_sample_reason"]
        manual["review_sample_reason"] = manual["stitch_edge_id"].map(sampled_reasons).fillna("")
    manual.to_csv(output_dir / "manual_stitch_review_manifest.csv", index=False)
    if wrote_stitched:
        uid_map.to_csv(output_dir / "track_uid_stitch_map.csv", index=False)
        stitched.to_csv(output_dir / f"tracks_{policy}_stitched.csv", index=False)
    stage_start = time.perf_counter()
    plot_stitch_review_panels(
        review, output_dir / "review_panels", max_panels=max_review_panels,
        sessions=sessions, features=features, transforms=transforms, spacing=spacing,
        candidate_rows=assignments,
    )
    review_plotting_sec = time.perf_counter() - stage_start

    accepted_mask = assignments["accepted_by_global_assignment"].astype(bool) if "accepted_by_global_assignment" in assignments else pd.Series(False, index=assignments.index)
    accepted = assignments.loc[accepted_mask]
    summary = {
        "n_canonical_tracks": int(len(tracks)),
        "n_unresolved_source_endpoints": int(candidates["endpoint_id"].nunique()) if not candidates.empty else 0,
        "n_unresolved_source_endpoints_total": len(source_endpoint_ids),
        "n_source_endpoints_with_candidate_starts": int(candidates["endpoint_id"].nunique()) if not candidates.empty else 0,
        "n_source_endpoints_without_candidate_starts": max(0, len(source_endpoint_ids) - (int(candidates["endpoint_id"].nunique()) if not candidates.empty else 0)),
        "n_candidate_track_start_edges": int(len(candidates)),
        "n_auto_eligible_edges": int(candidates.get("auto_eligible", pd.Series(dtype=bool)).sum()),
        "n_review_edges": int(candidates.get("review_eligible", pd.Series(dtype=bool)).sum()),
        "n_rejected_edges": int((candidates.get("candidate_tier", pd.Series(dtype=str)) == "reject").sum()),
        "n_global_accepted_stitches": int(len(accepted)),
        "n_global_collision_rejections": int((assignments.get("assignment_status", pd.Series(dtype=str)) == "global_collision_rejection").sum()),
        "accepted_by_gap": {str(int(key)): int(value) for key, value in accepted.get("session_gap", pd.Series(dtype=int)).value_counts().sort_index().items()},
        "accepted_by_elapsed_day_gap": {str(key): int(value) for key, value in accepted.get("elapsed_day_gap", pd.Series(dtype=float)).value_counts().sort_index().items()},
        "accepted_target_singletons": int((accepted.get("target_n_days_present", pd.Series(dtype=int)) == 1).sum()),
        "n_stitched_components": int((stitched.get("n_member_tracks", pd.Series(dtype=int)) > 1).sum()),
        "n_tracks_after_stitching": int(len(stitched)),
        **invariants,
        "synthetic_benchmark_passed": bool(benchmark_metrics["guardrail_passed"]),
        "synthetic_precision_by_gap": {key: value["precision"] for key, value in benchmark_metrics["by_gap"].items()},
        "synthetic_recall_by_gap": {key: value["recall"] for key, value in benchmark_metrics["by_gap"].items()},
        "synthetic_negative_fpr": benchmark_metrics["negative_fpr"],
        "benchmark_metrics": benchmark_metrics,
        "review_sample_count": int(len(review)),
        "stitched_tracks_requested": bool(write_stitched_tracks),
        "stitched_tracks_written": wrote_stitched,
        "forced_write_despite_benchmark_failure": bool(force_write_despite_benchmark_failure and wrote_stitched and not benchmark_metrics["guardrail_passed"]),
        "stitched_write_blocked_by_benchmark": bool(write_stitched_tracks and not wrote_stitched and not proposal_only),
        "canonical_matcher_modified": False,
        "canonical_matcher_outputs_overwritten": False,
        "state_features_used_for_assignment": False,
        "masks_interpolated": False,
    }
    (output_dir / "stitch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True, default=_json_value), encoding="utf-8")
    evaluator_log = _read_json(evaluation_dir / "evaluation_run_log.json")
    git_commit = ""
    try:
        import subprocess
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    total_sec = time.perf_counter() - total_start
    run_log = {
        "algorithm_version": ALGORITHM_VERSION, "profile": profile, "policy": policy,
        "config": asdict(config), "benchmark_replicates": benchmark_replicates,
        "benchmark_cases_per_gap": benchmark_cases_per_gap,
        "random_seed": random_seed, "spacing_um": asdict(spacing), "spacing_source": spacing_source,
        "match_dir": str(match_dir), "evaluation_dir": str(evaluation_dir), "output_dir": str(output_dir),
        "optional_state_provided_review_only": state_csv is not None,
        "optional_extraction_provided_review_only": extraction_dir is not None,
        "canonical_input_sha256": canonical_before,
        "evaluator_input_sha256": _hashes(evaluation_paths),
        "repo_git_commit": git_commit,
        "evaluator_algorithm_version": evaluator_log.get("evaluator_version"),
        "evaluator_repo_git_commit": evaluator_log.get("evaluator_repo_git_commit"),
        "stage_durations_seconds": {
            "load_inputs_sec": load_inputs_sec, "build_candidates_sec": build_candidates_sec,
            "global_assignment_sec": global_assignment_sec, "benchmark_sec": benchmark_sec,
            "build_stitched_view_sec": build_stitched_view_sec, "review_plotting_sec": review_plotting_sec,
            "total_sec": total_sec,
        },
        "benchmark_case_counts": {
            "by_type": {str(key): int(value) for key, value in benchmark.get("case_type", pd.Series(dtype=str)).value_counts().items()},
            "by_gap": {str(int(key)): int(value) for key, value in benchmark.get("session_gap", pd.Series(dtype=int)).value_counts().sort_index().items()},
        },
        "canonical_matcher_outputs_unchanged": canonical_before == _hashes(canonical_paths),
        "state_features_used_for_assignment": False, "eclipse_recomputed": False,
        "masks_interpolated": False, "png_only": True,
    }
    (output_dir / "stitch_run_log.json").write_text(json.dumps(run_log, indent=2, sort_keys=True), encoding="utf-8")
    if not run_log["canonical_matcher_outputs_unchanged"]:
        raise AssertionError("canonical matcher input changed during stitching")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=("graph", "balanced", "high"), default="graph")
    parser.add_argument("--max-gap-sessions", type=int, default=3)
    parser.add_argument("--search-radius-um", type=float, default=15.0)
    parser.add_argument("--profile", default="conservative_v1")
    parser.add_argument("--benchmark-replicates", type=int, default=20)
    parser.add_argument("--benchmark-cases-per-gap", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-review-panels", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--proposal-only", action="store_true")
    parser.add_argument("--write-stitched-tracks", action="store_true")
    parser.add_argument("--state-csv")
    parser.add_argument("--extraction-dir")
    parser.add_argument("--min-benchmark-precision", type=float, default=.999)
    parser.add_argument("--max-negative-fpr", type=float, default=.001)
    parser.add_argument("--force-write-despite-benchmark-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_endpoint_stitching(**vars(args))
    print(f"Stitcher completed: {args.output_dir}")
    print(f"Accepted stitches: {summary['n_global_accepted_stitches']} | benchmark passed: {summary['synthetic_benchmark_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
