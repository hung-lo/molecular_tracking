"""Command-line runner for experimental daywise spatial-graph ROI matching."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _import_dir in (_REPO_ROOT / "core", _REPO_ROOT / "matching"):
    _import_dir_str = str(_import_dir)
    if _import_dir_str not in sys.path:
        sys.path.append(_import_dir_str)

from affine_overlap_matcher import AffineOverlapParams, PairMatchResult, RestrictedTransform, VoxelSpacing
from daywise_roi_matcher_qc_plots import DaywiseQCPlotConfig, generate_matching_qc
from match_policy_registry import DEFAULT_ANALYSIS_POLICIES, SUPPORTED_MATCH_POLICIES, resolve_requested_policies
from roi_track_graph import (
    build_cycle_consistency_tables,
    build_track_length_summary_table,
    build_tracks_from_pair_tables,
    summarize_track_cycle_metadata,
)
from run_daywise_roi_matching import (
    PAIRWISE_CANDIDATE_COLUMNS,
    PAIRWISE_MATCH_COLUMNS,
    PAIRWISE_SUMMARY_COLUMNS,
    PAIRWISE_TRANSFORM_COLUMNS,
    run_daywise_roi_matching,
)
from spatial_graph_matcher import (
    GRAPH_MATCHER_ALGORITHM_VERSION,
    GRAPH_MATCHER_IMPLEMENTATION_VERSION,
    GraphPairMatchResult,
    SpatialGraphParams,
    refine_pair_with_spatial_graph,
)
from session_manifest import load_session_manifest

GRAPH_PAIRWISE_SUMMARY_COLUMNS = PAIRWISE_SUMMARY_COLUMNS + [
    "match_policy",
    "n_graph",
    "n_graph_anchors",
    "n_graph_changed",
    "graph_match_policy",
]
GRAPH_PAIRWISE_MATCH_COLUMNS = PAIRWISE_MATCH_COLUMNS + [
    "base_score",
    "graph_rule",
    "graph_status",
    "graph_support_count",
    "graph_support_fraction",
    "graph_residual_median_um",
    "graph_residual_mean_um",
    "graph_residual_p90_um",
    "graph_inlier_fraction",
    "graph_score",
    "refined_score",
    "is_graph_anchor",
    "assignment_source",
]
GRAPH_CHANGES_COLUMNS = ["label_a", "label_b", "in_balanced", "in_graph", "changed"]

GRAPH_RUNNER_ALGORITHM_VERSION = "daywise_graph_runner_v1"


def format_duration_seconds(duration_seconds: float) -> str:
    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def _affine_git_commit_from_log(payload: dict[str, object]) -> str | None:
    """Resolve affine-stage provenance without relabeling a legacy graph log."""

    if "affine_matcher_git_commit" in payload:
        value = payload.get("affine_matcher_git_commit")
        return str(value) if value else None

    graph_stage_markers = (
        "graph_matcher_algorithm_version",
        "graph_runner_version",
        "graph_params",
        "graph_row_counts",
        "graph_output_paths",
    )
    if any(marker in payload for marker in graph_stage_markers):
        return None

    value = payload.get("git_commit")
    return str(value) if value else None


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _restricted_transform_from_row(row: pd.Series) -> RestrictedTransform:
    return RestrictedTransform(
        z_intercept=float(row.get("z_intercept", 0.0)),
        z_scale=float(row.get("z_scale", 1.0)),
        y_intercept=float(row.get("y_intercept", 0.0)),
        y_from_y=float(row.get("y_from_y", 1.0)),
        y_from_x=float(row.get("y_from_x", 0.0)),
        x_intercept=float(row.get("x_intercept", 0.0)),
        x_from_y=float(row.get("x_from_y", 0.0)),
        x_from_x=float(row.get("x_from_x", 1.0)),
        method=str(row.get("method", "translation_only")),
        fallback_reason=row.get("fallback_reason", None),
        n_seed=int(row.get("n_seed", 0)),
        n_inlier=int(row.get("n_inlier", 0)),
        residual_median_um=row.get("residual_median_um", None),
        residual_p95_um=row.get("residual_p95_um", None),
    )


def _pair_result_from_outputs(
    *,
    day_a: str,
    day_b: str,
    pair_candidates: pd.DataFrame,
    pair_high: pd.DataFrame,
    pair_balanced: pd.DataFrame,
    pair_summary: pd.Series,
    pair_transform: pd.Series,
) -> PairMatchResult:
    candidates = pair_candidates.copy().reset_index(drop=True)
    high_matches = pair_high.copy().reset_index(drop=True)
    balanced_matches = pair_balanced.copy().reset_index(drop=True)
    summary = pair_summary.to_dict()
    summary.update({"day_a": day_a, "day_b": day_b})
    transform = _restricted_transform_from_row(pair_transform)
    return PairMatchResult(candidates=candidates, high_matches=high_matches, balanced_matches=balanced_matches, summary=summary, transform=transform)


def _build_features_by_session(roi_features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    features_by_session: dict[str, pd.DataFrame] = {}
    for session_id, subset in roi_features.groupby("session_id", sort=False):
        frame = subset.copy()
        if "label" not in frame.columns:
            raise ValueError("roi_features is missing the label column.")
        features_by_session[str(session_id)] = frame.set_index("label", drop=False)
    return features_by_session


def _pair_table_groups(table: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    if table.empty:
        return {}
    day_a_values = table["day_a"].astype(str).to_numpy()
    day_b_values = table["day_b"].astype(str).to_numpy()
    positions_by_pair: dict[tuple[str, str], list[int]] = {}
    for position, key in enumerate(zip(day_a_values, day_b_values)):
        positions_by_pair.setdefault((str(key[0]), str(key[1])), []).append(position)
    return {
        key: table.iloc[positions].copy().reset_index(drop=True)
        for key, positions in positions_by_pair.items()
    }


def _graph_pair_task(
    task: tuple[str, str, PairMatchResult, pd.DataFrame, pd.DataFrame, VoxelSpacing, SpatialGraphParams, int],
) -> tuple[GraphPairMatchResult, float]:
    """Refine one independent session pair and retain deterministic task order."""

    day_a, day_b, baseline_result, features_a, features_b, spacing, graph_params, pair_gap = task
    pair_start_seconds = time.perf_counter()
    result = refine_pair_with_spatial_graph(
        session_a=day_a,
        session_b=day_b,
        baseline_result=baseline_result,
        features_a=features_a,
        features_b=features_b,
        spacing=spacing,
        params=graph_params,
        pair_gap=pair_gap,
    )
    return result, float(time.perf_counter() - pair_start_seconds)


def _export_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_daywise_graph_matching(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    spacing: VoxelSpacing | None = None,
    params: AffineOverlapParams | None = None,
    graph_params: SpatialGraphParams | None = None,
    max_pair_gap: int = 2,
    pair_workers: int = 1,
    overwrite: bool = False,
    resume: bool = False,
    skip_qc: bool = False,
    qc_output_dir: str | Path | None = None,
    qc_image_format: str = "png",
    qc_dpi: int = 150,
    qc_max_examples: int = 20,
    qc_max_total_examples: int = 100,
    qc_random_seed: int = 0,
    require_qc_success: bool = False,
    stage_callback: Callable[[str, float], None] | None = None,
) -> Path:
    workflow_start_seconds = time.perf_counter()
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    spacing = spacing or VoxelSpacing()
    params = params or AffineOverlapParams()
    graph_params = graph_params or SpatialGraphParams()
    if int(pair_workers) < 1:
        raise ValueError("--pair-workers must be at least 1.")
    pair_workers = int(pair_workers)

    affine_stage_start = time.perf_counter()
    baseline_output_dir = run_daywise_roi_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        spacing=spacing,
        params=params,
        max_pair_gap=max_pair_gap,
        pair_workers=pair_workers,
        save_candidates=True,
        overwrite=overwrite,
        resume=resume,
        skip_qc=True,
        qc_output_dir=qc_output_dir,
        qc_image_format=qc_image_format,
        qc_dpi=qc_dpi,
        qc_max_examples=qc_max_examples,
        qc_max_total_examples=qc_max_total_examples,
        qc_random_seed=qc_random_seed,
        require_qc_success=require_qc_success,
    )
    affine_run_log_payload = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
    affine_matcher_git_commit = _affine_git_commit_from_log(affine_run_log_payload)
    if stage_callback is not None:
        stage_callback("daywise_affine_roi_matching", time.perf_counter() - affine_stage_start)

    run_start_seconds = time.perf_counter()
    stage_start_seconds = time.perf_counter()
    manifest_records = load_session_manifest(manifest_path)
    manifest_records = [record for record in manifest_records if record.required or True]
    ordered_sessions = [record.session_id for record in manifest_records]

    resolved_manifest = _load_csv(output_dir / "session_manifest_resolved.csv")
    if resolved_manifest.empty:
        raise FileNotFoundError(f"Missing resolved manifest from baseline run: {output_dir / 'session_manifest_resolved.csv'}")
    roi_features = _load_csv(output_dir / "roi_features.csv")
    pairwise_summary = _load_csv(output_dir / "pairwise_summary.csv")
    pairwise_transforms = _load_csv(output_dir / "pairwise_transforms.csv")
    pairwise_candidates = _load_csv(output_dir / "pairwise_candidates.csv")
    pairwise_high = _load_csv(output_dir / "pairwise_matches_high.csv")
    pairwise_balanced = _load_csv(output_dir / "pairwise_matches_balanced.csv")

    if pairwise_candidates.empty:
        raise FileNotFoundError("Graph runner requires pairwise_candidates.csv from the baseline run.")

    if "session_id" in roi_features.columns:
        roi_features["session_id"] = roi_features["session_id"].astype(str)
    features_by_session = _build_features_by_session(roi_features)
    pairwise_candidates_by_pair = _pair_table_groups(pairwise_candidates)
    pairwise_high_by_pair = _pair_table_groups(pairwise_high)
    pairwise_balanced_by_pair = _pair_table_groups(pairwise_balanced)
    pairwise_summary_by_pair = _pair_table_groups(pairwise_summary)
    graph_input_loading_seconds = time.perf_counter() - stage_start_seconds

    graph_tasks = []
    for row in pairwise_transforms.itertuples(index=False):
        day_a = str(row.day_a)
        day_b = str(row.day_b)
        pair_candidates_table = pairwise_candidates_by_pair.get((day_a, day_b), pairwise_candidates.iloc[0:0].copy())
        pair_high_table = pairwise_high_by_pair.get((day_a, day_b), pairwise_high.iloc[0:0].copy())
        pair_balanced_table = pairwise_balanced_by_pair.get((day_a, day_b), pairwise_balanced.iloc[0:0].copy())
        pair_summary_rows = pairwise_summary_by_pair.get((day_a, day_b))
        if pair_summary_rows is None or pair_summary_rows.empty:
            raise ValueError(f"Missing pairwise summary row for {day_a} -> {day_b}.")
        baseline_result = _pair_result_from_outputs(
            day_a=day_a,
            day_b=day_b,
            pair_candidates=pair_candidates_table,
            pair_high=pair_high_table,
            pair_balanced=pair_balanced_table,
            pair_summary=pair_summary_rows.iloc[0],
            pair_transform=pd.Series(row._asdict()),
        )
        graph_tasks.append((
            day_a,
            day_b,
            baseline_result,
            features_by_session[day_a],
            features_by_session[day_b],
            spacing,
            graph_params,
            int(pair_summary_rows.iloc[0].get("pair_gap", 0)) if "pair_gap" in pair_summary_rows.columns else 0,
        ))

    stage_start_seconds = time.perf_counter()
    if pair_workers == 1:
        completed_graph_pairs = map(_graph_pair_task, graph_tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=pair_workers)
        completed_graph_pairs = executor.map(_graph_pair_task, graph_tasks)
    graph_results: list[GraphPairMatchResult] = []
    graph_pair_timings = []
    try:
        for graph_result, pair_elapsed_seconds in completed_graph_pairs:
            graph_results.append(graph_result)
            graph_pair_timings.append({
                "day_a": str(graph_result.summary["day_a"]),
                "day_b": str(graph_result.summary["day_b"]),
                "pair_gap": int(graph_result.summary.get("pair_gap", 0)),
                "elapsed_sec": float(pair_elapsed_seconds),
                **(graph_result.timings_seconds or {}),
            })
    finally:
        if executor is not None:
            executor.shutdown()
    graph_pair_processing_seconds = time.perf_counter() - stage_start_seconds

    stage_start_seconds = time.perf_counter()
    graph_pair_tables = {
        (result.summary["day_a"], result.summary["day_b"]): result.graph_matches
        for result in graph_results
    }
    graph_tracks, graph_edges = build_tracks_from_pair_tables(
        day_names=ordered_sessions,
        features_by_session=features_by_session,
        pair_tables=graph_pair_tables,
        match_policy="graph",
    )
    graph_track_building_seconds = time.perf_counter() - stage_start_seconds

    stage_start_seconds = time.perf_counter()
    graph_cycle_summary, graph_cycle_edge_checks = build_cycle_consistency_tables(
        day_names=ordered_sessions,
        pair_tables=graph_pair_tables,
        tracks_table=graph_tracks,
        match_policy="graph",
    )
    graph_tracks = summarize_track_cycle_metadata(graph_tracks, graph_cycle_edge_checks)
    graph_length_summary = build_track_length_summary_table(graph_tracks)
    graph_cycle_consistency_seconds = time.perf_counter() - stage_start_seconds

    stage_start_seconds = time.perf_counter()
    graph_pairwise_matches = pd.concat([result.graph_matches for result in graph_results], ignore_index=True) if graph_results else pd.DataFrame(columns=GRAPH_PAIRWISE_MATCH_COLUMNS)
    graph_pairwise_summary = pd.DataFrame([result.summary for result in graph_results], columns=GRAPH_PAIRWISE_SUMMARY_COLUMNS)
    if not graph_pairwise_summary.empty:
        graph_pairwise_summary = graph_pairwise_summary.sort_values(["day_a", "day_b"]).reset_index(drop=True)
    graph_changes = pd.concat([result.changes for result in graph_results], ignore_index=True) if graph_results else pd.DataFrame(columns=GRAPH_CHANGES_COLUMNS)

    graph_pairwise_matches.to_csv(output_dir / "pairwise_matches_graph.csv", index=False)
    graph_pairwise_summary.to_csv(output_dir / "pairwise_summary_graph.csv", index=False)
    graph_tracks.to_csv(output_dir / "tracks_graph.csv", index=False)
    graph_cycle_summary.to_csv(output_dir / "cycle_consistency_graph.csv", index=False)
    graph_cycle_edge_checks.to_csv(output_dir / "cycle_edge_checks_graph.csv", index=False)
    graph_edges.to_csv(output_dir / "track_edges_graph.csv", index=False)
    graph_length_summary.to_csv(output_dir / "track_length_summary_graph.csv", index=False)
    graph_changes.to_csv(output_dir / "graph_match_changes.csv", index=False)
    graph_serialization_seconds = time.perf_counter() - stage_start_seconds

    run_log_payload = affine_run_log_payload
    warnings: list[str] = []
    qc_output_path = Path(qc_output_dir).resolve() if qc_output_dir is not None else output_dir / "qc"
    graph_runner_git_commit = _git_commit()
    graph_total_wall_seconds = float(time.perf_counter() - run_start_seconds)
    run_log_payload.update(
        {
            # Backward-compatible top-level provenance identifies the final graph stage.
            "git_commit": graph_runner_git_commit,
            "git_commit_role": "graph_runner",
            "affine_matcher_git_commit": affine_matcher_git_commit,
            "graph_runner_git_commit": graph_runner_git_commit,
            "graph_matcher_algorithm_version": GRAPH_MATCHER_ALGORITHM_VERSION,
            "graph_matcher_implementation_version": GRAPH_MATCHER_IMPLEMENTATION_VERSION,
            "graph_runner_version": GRAPH_RUNNER_ALGORITHM_VERSION,
            "graph_params": asdict(graph_params),
            "pair_workers": pair_workers,
            "policies": ["high", "balanced", "graph"],
            "supported_match_policies": list(SUPPORTED_MATCH_POLICIES),
            "default_analysis_policies": list(DEFAULT_ANALYSIS_POLICIES),
            "graph_row_counts": {
                "pairwise_matches_graph": int(len(graph_pairwise_matches)),
                "pairwise_summary_graph": int(len(graph_pairwise_summary)),
                "tracks_graph": int(len(graph_tracks)),
                "cycle_consistency_graph": int(len(graph_cycle_summary)),
                "cycle_edge_checks_graph": int(len(graph_cycle_edge_checks)),
                "track_edges_graph": int(len(graph_edges)),
                "track_length_summary_graph": int(len(graph_length_summary)),
            },
            "graph_output_paths": {
                "pairwise_matches_graph": str(output_dir / "pairwise_matches_graph.csv"),
                "pairwise_summary_graph": str(output_dir / "pairwise_summary_graph.csv"),
                "tracks_graph": str(output_dir / "tracks_graph.csv"),
                "cycle_consistency_graph": str(output_dir / "cycle_consistency_graph.csv"),
                "cycle_edge_checks_graph": str(output_dir / "cycle_edge_checks_graph.csv"),
                "track_edges_graph": str(output_dir / "track_edges_graph.csv"),
                "track_length_summary_graph": str(output_dir / "track_length_summary_graph.csv"),
                "graph_match_changes": str(output_dir / "graph_match_changes.csv"),
            },
        }
    )
    run_log_payload["matching_status"] = "completed"
    run_log_payload["qc_status"] = "not_requested" if skip_qc else "pending"
    run_log_payload["qc_output_dir"] = str(qc_output_path)
    run_log_payload["qc_error_type"] = None
    run_log_payload["qc_error"] = None
    run_log_payload["run_finished_utc"] = datetime.now(timezone.utc).isoformat()
    run_log_payload["warnings"] = warnings
    runtime_profile = run_log_payload.setdefault("runtime_profile", {})
    runtime_profile["graph_stage_durations_seconds"] = {
        "input_and_transform_loading": float(graph_input_loading_seconds),
        "graph_support_anchor_and_pairwise_assignment": float(graph_pair_processing_seconds),
        "track_graph_construction": float(graph_track_building_seconds),
        "pair_refinement_total": float(graph_pair_processing_seconds),
        "graph_track_building_total": float(graph_track_building_seconds),
        "graph_cycle_consistency_total": float(graph_cycle_consistency_seconds),
        "pairwise_and_track_output_serialization": float(graph_serialization_seconds),
        "graph_runner_total": graph_total_wall_seconds,
    }
    runtime_profile["graph_pair_timings_seconds"] = graph_pair_timings
    runtime_profile["graph_total_wall_seconds"] = graph_total_wall_seconds
    runtime_profile["graph_compute_wall_seconds"] = graph_total_wall_seconds
    runtime_profile["matching_qc_wall_seconds"] = 0.0 if skip_qc else None
    runtime_profile["matcher_total_wall_seconds"] = float(time.perf_counter() - workflow_start_seconds)
    _export_json(output_dir / "run_log.json", run_log_payload)

    if stage_callback is not None:
        stage_callback("graph_roi_matching", graph_total_wall_seconds)
    print(f"[{format_duration_seconds(graph_total_wall_seconds)}] Graph ROI matching completed", flush=True)

    if not skip_qc:
        qc_dir = qc_output_path
        qc_start_seconds = time.perf_counter()
        try:
            generate_matching_qc(
                DaywiseQCPlotConfig(
                    match_dir=output_dir,
                    output_dir=qc_dir,
                    sample_limit=int(qc_max_examples),
                    review_seed=int(qc_random_seed),
                    include_skip_pairs=True,
                    image_format=str(qc_image_format),
                    dpi=int(qc_dpi),
                    max_examples_per_category=int(qc_max_examples),
                    max_total_examples=int(qc_max_total_examples),
                    generate_visual_examples=True,
                    random_seed=int(qc_random_seed),
                )
            )
        except Exception as exc:
            run_log_payload["qc_status"] = "failed"
            run_log_payload["qc_error_type"] = type(exc).__name__
            run_log_payload["qc_error"] = str(exc)
            warnings.append("qc_failed")
            if require_qc_success:
                raise
        else:
            run_log_payload["qc_status"] = "completed"
            run_log_payload["qc_output_dir"] = str(qc_dir)
        finally:
            qc_duration_seconds = float(time.perf_counter() - qc_start_seconds)
            runtime_profile["matching_qc_wall_seconds"] = qc_duration_seconds
            runtime_profile["matcher_total_wall_seconds"] = float(time.perf_counter() - workflow_start_seconds)
            _export_json(output_dir / "run_log.json", run_log_payload)
            if stage_callback is not None:
                stage_callback("matching_qc", qc_duration_seconds)
            status = "completed" if run_log_payload["qc_status"] == "completed" else "failed"
            print(f"[{format_duration_seconds(qc_duration_seconds)}] Matching QC {status}", flush=True)
    return baseline_output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Resolved session manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for graph matcher outputs.")
    parser.add_argument("--xy-um-per-px", type=float, default=710.0 / 1024.0, help="XY pixel spacing in micrometers.")
    parser.add_argument("--z-um-per-plane", type=float, default=5.0, help="Z spacing in micrometers.")
    parser.add_argument("--max-pair-gap", type=int, default=2, help="Maximum allowed session gap for pairwise matching.")
    parser.add_argument("--pair-workers", type=int, default=1, help="Independent session-pair worker processes (default: 1).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory.")
    parser.add_argument("--resume", action="store_true", help="Reuse a prior exact-matching output directory when possible.")
    parser.add_argument("--skip-qc", action="store_true", help="Skip automatic QC generation after graph matching.")
    parser.add_argument("--qc-output-dir", default=None, help="Directory for automatic QC outputs.")
    parser.add_argument("--qc-image-format", default="png", help="Image format for QC figures.")
    parser.add_argument("--qc-dpi", type=int, default=150, help="DPI for QC figures.")
    parser.add_argument("--qc-max-examples", type=int, default=20, help="Maximum examples per QC category.")
    parser.add_argument("--qc-max-total-examples", type=int, default=100, help="Maximum total QC review examples.")
    parser.add_argument("--qc-random-seed", type=int, default=0, help="Random seed for QC sampling.")
    parser.add_argument("--require-qc-success", action="store_true", help="Fail the run if QC generation fails.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    spacing = VoxelSpacing(z_um=args.z_um_per_plane, y_um=args.xy_um_per_px, x_um=args.xy_um_per_px)
    output_dir = run_daywise_graph_matching(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        spacing=spacing,
        params=AffineOverlapParams(),
        graph_params=SpatialGraphParams(),
        max_pair_gap=args.max_pair_gap,
        pair_workers=args.pair_workers,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        skip_qc=bool(args.skip_qc),
        qc_output_dir=args.qc_output_dir,
        qc_image_format=str(args.qc_image_format),
        qc_dpi=int(args.qc_dpi),
        qc_max_examples=int(args.qc_max_examples),
        qc_max_total_examples=int(args.qc_max_total_examples),
        qc_random_seed=int(args.qc_random_seed),
        require_qc_success=bool(args.require_qc_success),
    )
    print(f"output_dir={output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
