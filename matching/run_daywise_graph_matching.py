"""Command-line runner for experimental daywise spatial-graph ROI matching."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

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
from run_daywise_roi_matching import run_daywise_roi_matching
from spatial_graph_matcher import GRAPH_MATCHER_ALGORITHM_VERSION, GraphPairMatchResult, SpatialGraphParams, refine_pair_with_spatial_graph
from session_manifest import load_session_manifest

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


def _pair_table_group(table: pd.DataFrame, day_a: str, day_b: str) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    mask = (table["day_a"].astype(str) == str(day_a)) & (table["day_b"].astype(str) == str(day_b))
    return table.loc[mask].copy().reset_index(drop=True)


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
) -> Path:
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    spacing = spacing or VoxelSpacing()
    params = params or AffineOverlapParams()
    graph_params = graph_params or SpatialGraphParams()

    baseline_output_dir = run_daywise_roi_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        spacing=spacing,
        params=params,
        max_pair_gap=max_pair_gap,
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

    run_start_seconds = time.perf_counter()
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

    pair_results: dict[tuple[str, str], PairMatchResult] = {}
    graph_results: list[GraphPairMatchResult] = []
    for row in pairwise_transforms.itertuples(index=False):
        day_a = str(row.day_a)
        day_b = str(row.day_b)
        pair_candidates_table = _pair_table_group(pairwise_candidates, day_a, day_b)
        pair_high_table = _pair_table_group(pairwise_high, day_a, day_b)
        pair_balanced_table = _pair_table_group(pairwise_balanced, day_a, day_b)
        pair_summary_row = pairwise_summary.loc[(pairwise_summary["day_a"].astype(str) == day_a) & (pairwise_summary["day_b"].astype(str) == day_b)]
        if pair_summary_row.empty:
            raise ValueError(f"Missing pairwise summary row for {day_a} -> {day_b}.")
        baseline_result = _pair_result_from_outputs(
            day_a=day_a,
            day_b=day_b,
            pair_candidates=pair_candidates_table,
            pair_high=pair_high_table,
            pair_balanced=pair_balanced_table,
            pair_summary=pair_summary_row.iloc[0],
            pair_transform=pd.Series(row._asdict()),
        )
        graph_result = refine_pair_with_spatial_graph(
            session_a=day_a,
            session_b=day_b,
            baseline_result=baseline_result,
            features_a=features_by_session[day_a],
            features_b=features_by_session[day_b],
            spacing=spacing,
            params=graph_params,
            pair_gap=int(pair_summary_row.iloc[0].get("pair_gap", 0)) if "pair_gap" in pair_summary_row.columns else None,
        )
        pair_results[(day_a, day_b)] = baseline_result
        graph_results.append(graph_result)

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
    graph_cycle_summary, graph_cycle_edge_checks = build_cycle_consistency_tables(
        day_names=ordered_sessions,
        pair_tables=graph_pair_tables,
        tracks_table=graph_tracks,
        match_policy="graph",
    )
    graph_tracks = summarize_track_cycle_metadata(graph_tracks, graph_cycle_edge_checks)
    graph_length_summary = build_track_length_summary_table(graph_tracks)

    graph_pairwise_matches = pd.concat([result.graph_matches for result in graph_results], ignore_index=True) if graph_results else pd.DataFrame()
    graph_pairwise_summary = pd.DataFrame([result.summary for result in graph_results]).sort_values(["day_a", "day_b"]).reset_index(drop=True)
    graph_changes = pd.concat([result.changes for result in graph_results], ignore_index=True) if graph_results else pd.DataFrame()

    graph_pairwise_matches.to_csv(output_dir / "pairwise_matches_graph.csv", index=False)
    graph_pairwise_summary.to_csv(output_dir / "pairwise_summary_graph.csv", index=False)
    graph_tracks.to_csv(output_dir / "tracks_graph.csv", index=False)
    graph_cycle_summary.to_csv(output_dir / "cycle_consistency_graph.csv", index=False)
    graph_cycle_edge_checks.to_csv(output_dir / "cycle_edge_checks_graph.csv", index=False)
    graph_edges.to_csv(output_dir / "track_edges_graph.csv", index=False)
    graph_length_summary.to_csv(output_dir / "track_length_summary_graph.csv", index=False)
    graph_changes.to_csv(output_dir / "graph_match_changes.csv", index=False)

    run_log = _load_csv(output_dir / "run_log.json")
    run_log_payload = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
    run_log_payload.update(
        {
            "graph_matcher_algorithm_version": GRAPH_MATCHER_ALGORITHM_VERSION,
            "graph_runner_version": GRAPH_RUNNER_ALGORITHM_VERSION,
            "graph_params": asdict(graph_params),
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
    run_log_payload["run_finished_utc"] = datetime.now(timezone.utc).isoformat()
    _export_json(output_dir / "run_log.json", run_log_payload)

    if not skip_qc:
        qc_dir = Path(qc_output_dir).resolve() if qc_output_dir is not None else output_dir / "qc"
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

    total_duration_seconds = time.perf_counter() - run_start_seconds
    print(f"[{format_duration_seconds(total_duration_seconds)}] Graph ROI matching completed", flush=True)
    return baseline_output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Resolved session manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for graph matcher outputs.")
    parser.add_argument("--xy-um-per-px", type=float, default=710.0 / 1024.0, help="XY pixel spacing in micrometers.")
    parser.add_argument("--z-um-per-plane", type=float, default=5.0, help="Z spacing in micrometers.")
    parser.add_argument("--max-pair-gap", type=int, default=2, help="Maximum allowed session gap for pairwise matching.")
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
