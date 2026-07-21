"""Run the complete daywise ROI matching, extraction, and QC workflow.

Recommended repository location
-------------------------------
    core/run_daywise_master_pipeline.py

This runner performs the following steps:

1. Run the existing affine-overlap matcher and spatial-graph refinement together.
2. Compare the graph matches with the affine ``balanced`` matches.
3. Keep the graph result as the final assignment, while annotating tracks as:
      - ``consensus``: every accepted graph edge is also present in balanced.
      - ``graph_only``: at least one accepted graph edge differs from balanced.
      - ``unmatched``: a singleton ROI with no accepted longitudinal edge.
4. Run matched ROI intensity extraction using the annotated graph tracks.
5. Add agreement-source columns to all compatible extraction tables.
6. Run the repository's existing matched-output quick plots.
7. Add a wrapped red-vs-green linear-fit plot with at most seven day panels per row.
8. Write one top-level run manifest and stable, predictable output paths.

The graph result is used for disagreements because it is the selected fallback
policy. The affine ``balanced`` result is used only to determine agreement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
for _import_dir in (
    _REPO_ROOT / "core",
    _REPO_ROOT / "matching",
    _REPO_ROOT / "plotting",
):
    _import_dir_string = str(_import_dir)
    if _import_dir_string not in sys.path:
        sys.path.append(_import_dir_string)

from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing
from analysis_paths import get_dataset_analysis_dir, resolve_dataset_dir
from run_daywise_graph_matching import run_daywise_graph_matching
from run_daywise_green_red_linear_fit_summary import compute_regression_ci_band
from roi_log_ratio_analysis import summarize_daily_green_red_linear_fits
from run_daywise_matched_roi_pipeline import (
    DaywiseMatchedPipelineConfig,
    run_daywise_matched_roi_pipeline,
)
from run_weekly_matched_output_quick_plots import build_quick_plots
from session_manifest import load_session_manifest


MASTER_RUNNER_VERSION = "daywise_master_graph_affine_consensus_v1"
AGREEMENT_COLUMNS = [
    "track_match_source",
    "n_accepted_graph_edges",
    "n_consensus_edges",
    "n_graph_only_edges",
    "consensus_edge_fraction",
    "has_graph_only_edge",
]
EDGE_KEY_COLUMNS = ["day_a", "day_b", "label_a", "label_b"]


@dataclass(frozen=True)
class MasterPipelineConfig:
    dataset: str
    manifest: str
    output_root: str | None = None
    run_name: str | None = None
    xy_um_per_px: float = 710.0 / 1024.0
    z_um_per_plane: float = 5.0
    max_pair_gap: int = 2
    green_dark: float = 319.0
    red_dark: float = 534.0
    epsilon: float = 1.0
    top_n: int = 30
    plot_columns: int = 7
    overwrite: bool = False
    resume: bool = False
    skip_matching_qc: bool = False
    skip_quick_plots: bool = False
    require_qc_success: bool = False
    qc_dpi: int = 150
    qc_max_examples: int = 20
    qc_max_total_examples: int = 100
    qc_random_seed: int = 0
    segmentation_qc_mode: str = "all_required"
    min_segmentation_pass_fraction: float = 1.0
    min_volume_um3: float | None = None
    max_volume_um3: float | None = None
    min_bbox_depth_planes: float | None = None
    max_bbox_depth_planes: float | None = None
    exclude_xy_edge: bool = False
    exclude_z_edge: bool = False
    max_volume_ratio_from_track_median: float | None = None


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _log(start_seconds: float, message: str) -> None:
    print(f"[{_format_duration(time.perf_counter() - start_seconds)}] {message}", flush=True)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "daywise_run"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
    )


def _edge_key_frame(frame: pd.DataFrame) -> pd.MultiIndex:
    missing = [column for column in EDGE_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Pairwise match table is missing columns: {missing}")
    normalized = frame.loc[:, EDGE_KEY_COLUMNS].copy()
    normalized["day_a"] = normalized["day_a"].astype(str)
    normalized["day_b"] = normalized["day_b"].astype(str)
    normalized["label_a"] = pd.to_numeric(normalized["label_a"], errors="raise").astype(int)
    normalized["label_b"] = pd.to_numeric(normalized["label_b"], errors="raise").astype(int)
    return pd.MultiIndex.from_frame(normalized)


def _manifest_metadata(manifest_path: Path) -> dict[str, Any]:
    records = load_session_manifest(manifest_path)
    if not records:
        raise ValueError(f"No sessions were found in manifest: {manifest_path}")
    required_records = [record for record in records if record.required]
    date_records = required_records or records
    first_date = min(record.acquisition_date for record in date_records)
    last_date = max(record.acquisition_date for record in date_records)
    return {
        "records": records,
        "n_sessions": len(records),
        "n_required_sessions": len(required_records),
        "first_date": first_date,
        "last_date": last_date,
        "start_date": first_date.strftime("%Y%m%d"),
        "session_ids": [str(record.session_id) for record in records],
    }


def _default_run_name(dataset_dir: Path, manifest_meta: dict[str, Any]) -> str:
    first_date = manifest_meta["first_date"].strftime("%Y%m%d")
    last_date = manifest_meta["last_date"].strftime("%Y%m%d")
    n_sessions = int(manifest_meta["n_sessions"])
    return _safe_name(
        f"{dataset_dir.name}_{first_date}_to_{last_date}_{n_sessions}s_graph_affine_balanced"
    )


def _prepare_run_directory(config: MasterPipelineConfig) -> tuple[Path, Path, Path, Path]:
    dataset_dir = resolve_dataset_dir(config.dataset)
    analysis_dir = get_dataset_analysis_dir(config.dataset)
    output_root = (
        Path(config.output_root).expanduser().resolve()
        if config.output_root
        else analysis_dir / "daywise_master_runs"
    )
    manifest_meta = _manifest_metadata(Path(config.manifest).expanduser().resolve())
    run_name = _safe_name(config.run_name or _default_run_name(dataset_dir, manifest_meta))
    run_dir = output_root / run_name

    if run_dir.exists() and config.overwrite:
        shutil.rmtree(run_dir)
    elif run_dir.exists() and not config.resume:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use --resume or --overwrite."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    match_dir = run_dir / "matching"
    extraction_dir = run_dir / "extraction"
    plots_dir = run_dir / "plots"
    return run_dir, match_dir, extraction_dir, plots_dir


def _node_to_track_lookup(tracks: pd.DataFrame) -> dict[tuple[str, int], tuple[str, int]]:
    session_columns = [
        column
        for column in tracks.columns
        if column.endswith("_roi") and column not in {"roi", "roi_id"}
    ]
    lookup: dict[tuple[str, int], tuple[str, int]] = {}
    for _, row in tracks.iterrows():
        track_uid = str(row["track_uid"])
        cluster_id = int(row["cluster_id"])
        for column in session_columns:
            value = row.get(column, pd.NA)
            if pd.isna(value):
                continue
            session_id = column[: -len("_roi")]
            lookup[(session_id, int(value))] = (track_uid, cluster_id)
    return lookup


def annotate_graph_affine_agreement(match_dir: Path) -> dict[str, Any]:
    """Annotate accepted graph tracks by agreement with balanced affine matching."""

    balanced_path = match_dir / "pairwise_matches_balanced.csv"
    graph_path = match_dir / "pairwise_matches_graph.csv"
    graph_edges_path = match_dir / "track_edges_graph.csv"
    graph_tracks_path = match_dir / "tracks_graph.csv"

    for required_path in (
        balanced_path,
        graph_path,
        graph_edges_path,
        graph_tracks_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required matching output was not found: {required_path}")

    balanced = _read_csv(balanced_path)
    graph = _read_csv(graph_path)
    graph_edges = _read_csv(graph_edges_path)
    graph_tracks = _read_csv(graph_tracks_path)
    if graph_tracks.empty:
        raise ValueError("tracks_graph.csv is empty; no graph tracks can be annotated.")

    balanced_keys = set(_edge_key_frame(balanced).tolist()) if not balanced.empty else set()

    graph_keys = _edge_key_frame(graph)
    graph = graph.copy()
    graph["agreement_status"] = [
        "consensus" if key in balanced_keys else "graph_only" for key in graph_keys.tolist()
    ]
    graph["agrees_with_affine_balanced"] = graph["agreement_status"].eq("consensus")
    graph.to_csv(match_dir / "pairwise_matches_graph_agreement.csv", index=False)

    accepted_edges = graph_edges.loc[_bool_series(graph_edges["accepted_for_track"])].copy()
    if accepted_edges.empty:
        accepted_edges["agreement_status"] = pd.Series(dtype=str)
        accepted_edges["track_uid"] = pd.Series(dtype=str)
        accepted_edges["cluster_id"] = pd.Series(dtype="Int64")
    else:
        accepted_keys = _edge_key_frame(accepted_edges)
        accepted_edges["agreement_status"] = [
            "consensus" if key in balanced_keys else "graph_only"
            for key in accepted_keys.tolist()
        ]
        accepted_edges["agrees_with_affine_balanced"] = accepted_edges[
            "agreement_status"
        ].eq("consensus")

        node_lookup = _node_to_track_lookup(graph_tracks)
        track_uids: list[str] = []
        cluster_ids: list[int] = []
        for row in accepted_edges.itertuples(index=False):
            node_a = (str(row.day_a), int(row.label_a))
            node_b = (str(row.day_b), int(row.label_b))
            owner_a = node_lookup.get(node_a)
            owner_b = node_lookup.get(node_b)
            if owner_a is None or owner_b is None:
                raise ValueError(f"Accepted graph edge has an unmapped node: {node_a} -> {node_b}")
            if owner_a != owner_b:
                raise ValueError(
                    "Accepted graph edge endpoints map to different tracks: "
                    f"{node_a} ({owner_a}) -> {node_b} ({owner_b})"
                )
            track_uids.append(owner_a[0])
            cluster_ids.append(owner_a[1])
        accepted_edges["track_uid"] = track_uids
        accepted_edges["cluster_id"] = cluster_ids

    accepted_edges.to_csv(match_dir / "accepted_graph_edges_with_agreement.csv", index=False)

    if accepted_edges.empty:
        edge_summary = pd.DataFrame(
            columns=[
                "track_uid",
                "n_accepted_graph_edges",
                "n_consensus_edges",
                "n_graph_only_edges",
            ]
        )
    else:
        edge_summary = (
            accepted_edges.assign(
                is_consensus=accepted_edges["agreement_status"].eq("consensus").astype(int),
                is_graph_only=accepted_edges["agreement_status"].eq("graph_only").astype(int),
            )
            .groupby("track_uid", as_index=False)
            .agg(
                n_accepted_graph_edges=("agreement_status", "size"),
                n_consensus_edges=("is_consensus", "sum"),
                n_graph_only_edges=("is_graph_only", "sum"),
            )
        )

    annotated_tracks = graph_tracks.drop(columns=AGREEMENT_COLUMNS, errors="ignore").merge(
        edge_summary,
        on="track_uid",
        how="left",
        validate="one_to_one",
    )
    for column in (
        "n_accepted_graph_edges",
        "n_consensus_edges",
        "n_graph_only_edges",
    ):
        annotated_tracks[column] = annotated_tracks[column].fillna(0).astype(int)
    annotated_tracks["consensus_edge_fraction"] = np.where(
        annotated_tracks["n_accepted_graph_edges"] > 0,
        annotated_tracks["n_consensus_edges"]
        / annotated_tracks["n_accepted_graph_edges"],
        np.nan,
    )
    annotated_tracks["has_graph_only_edge"] = annotated_tracks["n_graph_only_edges"] > 0
    annotated_tracks["track_match_source"] = np.select(
        [
            annotated_tracks["n_accepted_graph_edges"].eq(0),
            annotated_tracks["n_graph_only_edges"].gt(0),
        ],
        ["unmatched", "graph_only"],
        default="consensus",
    )
    annotated_tracks["cluster_id"] = pd.to_numeric(
        annotated_tracks["cluster_id"], errors="raise"
    ).astype(int)
    for session_column in [
        column for column in annotated_tracks.columns if column.endswith("_roi")
    ]:
        annotated_tracks[session_column] = pd.to_numeric(
            annotated_tracks[session_column], errors="coerce"
        ).astype("Int64")

    raw_backup = match_dir / "tracks_graph_base.csv"
    if not raw_backup.exists():
        graph_tracks.to_csv(raw_backup, index=False)
    annotated_tracks.to_csv(graph_tracks_path, index=False)
    annotated_tracks.to_csv(match_dir / "tracks_graph_agreement.csv", index=False)

    track_source_summary = (
        annotated_tracks.groupby("track_match_source", dropna=False)
        .size()
        .rename("n_tracks")
        .reset_index()
        .sort_values("track_match_source")
    )
    track_source_summary.to_csv(match_dir / "graph_affine_agreement_track_summary.csv", index=False)

    summary = {
        "affine_comparison_policy": "balanced",
        "final_assignment_policy": "graph",
        "n_balanced_pairwise_matches": int(len(balanced)),
        "n_graph_pairwise_matches": int(len(graph)),
        "n_accepted_graph_edges": int(len(accepted_edges)),
        "n_consensus_accepted_edges": int(
            accepted_edges["agreement_status"].eq("consensus").sum()
        )
        if not accepted_edges.empty
        else 0,
        "n_graph_only_accepted_edges": int(
            accepted_edges["agreement_status"].eq("graph_only").sum()
        )
        if not accepted_edges.empty
        else 0,
        "track_source_counts": {
            str(row.track_match_source): int(row.n_tracks)
            for row in track_source_summary.itertuples(index=False)
        },
        "outputs": {
            "annotated_tracks": str(graph_tracks_path),
            "base_tracks_backup": str(raw_backup),
            "pairwise_agreement": str(match_dir / "pairwise_matches_graph_agreement.csv"),
            "accepted_edge_agreement": str(match_dir / "accepted_graph_edges_with_agreement.csv"),
            "track_summary": str(match_dir / "graph_affine_agreement_track_summary.csv"),
        },
    }
    _write_json(match_dir / "graph_affine_agreement_summary.json", summary)

    run_log_path = match_dir / "run_log.json"
    if run_log_path.exists():
        run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
        run_log["graph_affine_agreement"] = summary
        _write_json(run_log_path, run_log)

    return summary


def _replace_path_strings(value: Any, old_root: str, new_root: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_path_strings(item, old_root, new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_path_strings(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    return value


def _relocate_extraction_output(source_dir: Path, destination_dir: Path) -> Path:
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    if source_dir == destination_dir:
        return destination_dir
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.move(str(source_dir), str(destination_dir))

    run_log_path = destination_dir / "run_log.json"
    if run_log_path.exists():
        payload = json.loads(run_log_path.read_text(encoding="utf-8"))
        payload = _replace_path_strings(payload, str(source_dir), str(destination_dir))
        _write_json(run_log_path, payload)
    return destination_dir


def _agreement_metadata(tracks_path: Path) -> pd.DataFrame:
    tracks = _read_csv(tracks_path)
    columns = ["cluster_id", "track_uid", *AGREEMENT_COLUMNS]
    missing = [column for column in columns if column not in tracks.columns]
    if missing:
        raise ValueError(f"Annotated graph tracks are missing columns: {missing}")
    return tracks.loc[:, columns].drop_duplicates("track_uid").reset_index(drop=True)


def _merge_agreement_into_table(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return table
    clean = table.drop(columns=AGREEMENT_COLUMNS, errors="ignore")
    agreement_only = metadata.loc[:, ["track_uid", *AGREEMENT_COLUMNS]]
    if "track_uid" in clean.columns:
        return clean.merge(agreement_only, on="track_uid", how="left", validate="many_to_one")
    if "cluster_id" in clean.columns:
        by_cluster = metadata.loc[:, ["cluster_id", *AGREEMENT_COLUMNS]].drop_duplicates(
            "cluster_id"
        )
        return clean.merge(by_cluster, on="cluster_id", how="left", validate="many_to_one")
    if "roi_id" in clean.columns:
        by_roi = metadata.loc[:, ["cluster_id", *AGREEMENT_COLUMNS]].rename(
            columns={"cluster_id": "roi_id"}
        )
        return clean.merge(by_roi, on="roi_id", how="left", validate="many_to_one")
    return table


def annotate_extraction_outputs(extraction_dir: Path, tracks_path: Path) -> dict[str, Any]:
    metadata = _agreement_metadata(tracks_path)
    annotated_files: list[str] = []
    for csv_path in sorted(extraction_dir.glob("*.csv")):
        table = _read_csv(csv_path)
        if table.empty:
            continue
        annotated = _merge_agreement_into_table(table, metadata)
        if annotated is table or not any(
            column in annotated.columns for column in AGREEMENT_COLUMNS
        ):
            continue
        annotated.to_csv(csv_path, index=False)
        annotated_files.append(csv_path.name)

    metadata.to_csv(extraction_dir / "graph_affine_agreement_track_metadata.csv", index=False)
    consensus_tracks = metadata.loc[metadata["track_match_source"].eq("consensus")]
    graph_only_tracks = metadata.loc[metadata["track_match_source"].eq("graph_only")]
    consensus_tracks.to_csv(extraction_dir / "final_tracks_consensus.csv", index=False)
    graph_only_tracks.to_csv(extraction_dir / "final_tracks_graph_only.csv", index=False)

    metrics_path = extraction_dir / "matched_roi_log_ratio_metrics_complete.csv"
    if metrics_path.exists():
        metrics = _read_csv(metrics_path)
        if "track_match_source" in metrics.columns:
            metrics.loc[metrics["track_match_source"].eq("consensus")].to_csv(
                extraction_dir / "matched_roi_log_ratio_metrics_consensus.csv",
                index=False,
            )
            metrics.loc[metrics["track_match_source"].eq("graph_only")].to_csv(
                extraction_dir / "matched_roi_log_ratio_metrics_graph_only.csv",
                index=False,
            )

    summary = {
        "annotated_csv_files": annotated_files,
        "n_consensus_tracks": int(len(consensus_tracks)),
        "n_graph_only_tracks": int(len(graph_only_tracks)),
        "n_unmatched_tracks": int(
            metadata["track_match_source"].eq("unmatched").sum()
        ),
    }
    _write_json(extraction_dir / "graph_affine_agreement_annotation_log.json", summary)
    return summary


def _filter_green_artifacts(day_table: pd.DataFrame, green_artifact_threshold: float) -> tuple[pd.DataFrame, int]:
    if day_table.empty:
        return day_table.copy(), 0
    cleaned = day_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["red", "green"]).copy()
    if cleaned.empty:
        return cleaned, 0
    artifact_mask = cleaned["green"].gt(float(green_artifact_threshold))
    excluded_count = int(artifact_mask.sum())
    if excluded_count == 0:
        return cleaned.reset_index(drop=True), 0
    return cleaned.loc[~artifact_mask].reset_index(drop=True), excluded_count


def _actual_date_label(day_table: pd.DataFrame, day_value: int, start_date: str) -> str:
    if "acquisition_date" in day_table.columns:
        values = day_table["acquisition_date"].dropna()
        if not values.empty:
            parsed = pd.to_datetime(values.iloc[0], errors="coerce")
            if pd.notna(parsed):
                return parsed.strftime("%Y%m%d")
    reference_date = pd.to_datetime(start_date, format="%Y%m%d")
    return (reference_date + pd.to_timedelta(int(day_value), unit="D")).strftime("%Y%m%d")


def plot_wrapped_daywise_linear_relationships(
    metrics_path: Path,
    fit_summary_path: Path,
    output_path: Path,
    *,
    start_date: str,
    max_columns: int = 7,
    green_artifact_threshold: float = 1500.0,
) -> None:
    """Plot one red-vs-green panel per session, wrapping after max_columns."""

    metrics = _read_csv(metrics_path)
    if "match_policy" in metrics.columns:
        metrics = metrics.loc[metrics["match_policy"].astype(str).eq("graph")].copy()
    if metrics.empty:
        raise ValueError("No graph-policy metrics were available for wrapped linear-fit plotting.")

    fit_summary = summarize_daily_green_red_linear_fits(metrics)
    fit_summary["day"] = pd.to_numeric(fit_summary["day"], errors="raise").astype(int)
    day_values = sorted(fit_summary["day"].unique().tolist())
    n_columns = min(max(1, int(max_columns)), len(day_values))
    n_rows = int(math.ceil(len(day_values) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.6 * n_columns, 3.65 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    flat_axes = axes.ravel()

    panels: list[tuple[int, pd.DataFrame, int]] = []
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    excluded_total = 0
    for day_value in day_values:
        columns = ["red", "green"]
        if "track_match_source" in metrics.columns:
            columns.append("track_match_source")
        if "acquisition_date" in metrics.columns:
            columns.append("acquisition_date")
        day_table_raw = (
            metrics.loc[metrics["day"].eq(day_value), columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["red", "green"])
            .reset_index(drop=True)
        )
        day_table, excluded_count = _filter_green_artifacts(day_table_raw, green_artifact_threshold)
        if day_table.empty and not day_table_raw.empty:
            day_table = day_table_raw.copy()
            excluded_count = 0
        excluded_total += excluded_count
        if not day_table.empty:
            x_min = min(x_min, float(day_table["red"].min()))
            x_max = max(x_max, float(day_table["red"].max()))
            y_min = min(y_min, float(day_table["green"].min()))
            y_max = max(y_max, float(day_table["green"].max()))
        panels.append((int(day_value), day_table, excluded_count))

    for axis, (day_value, day_table, excluded_count) in zip(flat_axes, panels, strict=False):
        x_values = day_table["red"].to_numpy(dtype=float)
        y_values = day_table["green"].to_numpy(dtype=float)
        if len(day_table) >= 2:
            x_grid, y_hat, y_low, y_high = compute_regression_ci_band(x_values, y_values)
        else:
            x_grid = np.asarray([], dtype=float)
            y_hat = np.asarray([], dtype=float)
            y_low = np.asarray([], dtype=float)
            y_high = np.asarray([], dtype=float)

        fit_row = fit_summary.loc[fit_summary["day"].eq(day_value)].iloc[0]
        if "track_match_source" in day_table.columns:
            consensus = day_table["track_match_source"].eq("consensus")
            graph_only = day_table["track_match_source"].eq("graph_only")
            other = ~(consensus | graph_only)
            axis.scatter(
                day_table.loc[consensus, "red"],
                day_table.loc[consensus, "green"],
                s=10,
                alpha=0.22,
                color="#1f3b4d",
                edgecolors="none",
                rasterized=True,
                label="Consensus",
            )
            axis.scatter(
                day_table.loc[graph_only, "red"],
                day_table.loc[graph_only, "green"],
                s=18,
                alpha=0.70,
                facecolors="none",
                edgecolors="#d97706",
                linewidths=0.7,
                rasterized=True,
                label="Graph only",
            )
            if other.any():
                axis.scatter(
                    day_table.loc[other, "red"],
                    day_table.loc[other, "green"],
                    s=9,
                    alpha=0.18,
                    color="0.5",
                    edgecolors="none",
                    rasterized=True,
                )
        else:
            axis.scatter(
                x_values,
                y_values,
                s=10,
                alpha=0.2,
                color="#1f3b4d",
                edgecolors="none",
                rasterized=True,
            )

        if len(x_grid) > 0 and np.all(np.isfinite(y_hat)):
            if np.all(np.isfinite(y_low)) and np.all(np.isfinite(y_high)):
                axis.fill_between(
                    x_grid,
                    y_low,
                    y_high,
                    color="#8ecae6",
                    alpha=0.3,
                    linewidth=0,
                )
            axis.plot(x_grid, y_hat, color="#d62828", linewidth=1.8)

        date_label = _actual_date_label(day_table, day_value, start_date)
        consensus_n = int(
            day_table.get("track_match_source", pd.Series(dtype=str)).eq("consensus").sum()
        )
        graph_only_n = int(
            day_table.get("track_match_source", pd.Series(dtype=str)).eq("graph_only").sum()
        )
        source_text = (
            f"\nconsensus={consensus_n}, graph-only={graph_only_n}"
            if "track_match_source" in day_table.columns
            else ""
        )
        artifact_text = (
            f"\nexcluded green>{green_artifact_threshold:g}: {excluded_count}"
            if excluded_count
            else ""
        )
        axis.set_title(f"Session {day_value} | {date_label}", fontsize=10)
        axis.set_xlabel("Corrected red", fontsize=9)
        axis.set_ylabel("Corrected green", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.text(
            0.04,
            0.96,
            (
                f"slope={fit_row['slope']:.3f}\n"
                f"R²={fit_row['r_squared']:.3f}\n"
                f"n={int(fit_row['n_rois'])}{source_text}{artifact_text}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.88},
        )

    for axis in flat_axes[len(panels) :]:
        axis.set_visible(False)

    if np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max):
        x_pad = max((x_max - x_min) * 0.05, 1.0)
        y_pad = max((y_max - y_min) * 0.05, 1.0)
        for axis in flat_axes[: len(panels)]:
            axis.set_xlim(x_min - x_pad, x_max + x_pad)
            axis.set_ylim(y_min - y_pad, y_max + y_pad)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper right", frameon=False)
    figure.suptitle(
        "Daywise corrected red-green relationships: graph final assignments",
        fontsize=14,
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Green values above {green_artifact_threshold:g} were excluded from the fit; "
            f"total excluded rows = {excluded_total}."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.01, 0.03, 0.99, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_master_summary(
    run_dir: Path,
    config: MasterPipelineConfig,
    agreement_summary: dict[str, Any],
    extraction_annotation: dict[str, Any],
    match_dir: Path,
    extraction_dir: Path,
    plots_dir: Path,
) -> None:
    lines = [
        "# Daywise master ROI run",
        "",
        f"- Runner version: `{MASTER_RUNNER_VERSION}`",
        f"- Dataset: `{config.dataset}`",
        f"- Manifest: `{config.manifest}`",
        f"- Final assignment policy: `graph`",
        f"- Agreement comparison policy: `balanced` affine-overlap",
        "",
        "## Agreement rule",
        "",
        "- A track is `consensus` when every accepted graph edge is also present in the balanced affine result.",
        "- A track is `graph_only` when at least one accepted graph edge is absent from the balanced affine result.",
        "- Graph-only tracks remain in the final extracted dataset and are explicitly marked.",
        "- Singleton ROIs are marked `unmatched` and normally disappear during complete-session filtering.",
        "",
        "## Counts",
        "",
        f"- Consensus accepted edges: `{agreement_summary['n_consensus_accepted_edges']}`",
        f"- Graph-only accepted edges: `{agreement_summary['n_graph_only_accepted_edges']}`",
        f"- Consensus tracks: `{extraction_annotation['n_consensus_tracks']}`",
        f"- Graph-only tracks: `{extraction_annotation['n_graph_only_tracks']}`",
        "",
        "## Output locations",
        "",
        f"- Matching and matching QC: `{match_dir}`",
        f"- Intensity extraction and tables: `{extraction_dir}`",
        f"- Quick plots: `{plots_dir}`",
        "",
        "The wrapped linear-fit plot is `plots/graph/daywise_green_red_linear_fit_scatters_wrapped_7cols.png` by default.",
    ]
    (run_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_master_pipeline(config: MasterPipelineConfig) -> Path:
    start_seconds = time.perf_counter()
    manifest_path = Path(config.manifest).expanduser().resolve()
    dataset_dir = resolve_dataset_dir(config.dataset)
    manifest_meta = _manifest_metadata(manifest_path)
    run_dir, match_dir, extraction_dir, plots_dir = _prepare_run_directory(config)

    _log(start_seconds, f"Master run directory: {run_dir}")
    _log(start_seconds, "Running affine-overlap baseline plus graph refinement")
    spacing = VoxelSpacing(
        z_um=float(config.z_um_per_plane),
        y_um=float(config.xy_um_per_px),
        x_um=float(config.xy_um_per_px),
    )
    run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=match_dir,
        spacing=spacing,
        params=AffineOverlapParams(),
        max_pair_gap=int(config.max_pair_gap),
        overwrite=bool(config.overwrite),
        resume=bool(config.resume),
        skip_qc=bool(config.skip_matching_qc),
        qc_output_dir=match_dir / "qc",
        qc_dpi=int(config.qc_dpi),
        qc_max_examples=int(config.qc_max_examples),
        qc_max_total_examples=int(config.qc_max_total_examples),
        qc_random_seed=int(config.qc_random_seed),
        require_qc_success=bool(config.require_qc_success),
    )

    _log(start_seconds, "Annotating graph tracks by agreement with balanced affine matches")
    agreement_summary = annotate_graph_affine_agreement(match_dir)

    required_extraction_files = [
        extraction_dir / "matched_roi_log_ratio_metrics_complete.csv",
        extraction_dir / "matched_daywise_green_red_linear_fit_summary.csv",
    ]
    if config.resume and all(path.exists() for path in required_extraction_files):
        _log(start_seconds, f"Reusing existing extraction output: {extraction_dir}")
    else:
        _log(start_seconds, "Extracting graph-policy matched ROI intensities")
        temporary_extraction_dir = run_daywise_matched_roi_pipeline(
            DaywiseMatchedPipelineConfig(
                dataset=str(dataset_dir),
                manifest=str(manifest_path),
                match_dir=str(match_dir),
                policies=("graph",),
                green_dark=float(config.green_dark),
                red_dark=float(config.red_dark),
                epsilon=float(config.epsilon),
                segmentation_qc_mode=str(config.segmentation_qc_mode),
                min_segmentation_pass_fraction=float(
                    config.min_segmentation_pass_fraction
                ),
                min_volume_um3=config.min_volume_um3,
                max_volume_um3=config.max_volume_um3,
                min_bbox_depth_planes=config.min_bbox_depth_planes,
                max_bbox_depth_planes=config.max_bbox_depth_planes,
                exclude_xy_edge=bool(config.exclude_xy_edge),
                exclude_z_edge=bool(config.exclude_z_edge),
                max_volume_ratio_from_track_median=(
                    config.max_volume_ratio_from_track_median
                ),
            )
        )
        extraction_dir = _relocate_extraction_output(
            Path(temporary_extraction_dir), extraction_dir
        )

    _log(start_seconds, "Propagating consensus and graph-only labels into extraction tables")
    extraction_annotation = annotate_extraction_outputs(
        extraction_dir, match_dir / "tracks_graph.csv"
    )

    graph_plot_dir = plots_dir / "graph"
    if not config.skip_quick_plots:
        _log(start_seconds, "Rendering existing matched-output quick plots")
        build_quick_plots(
            analysis_dir=extraction_dir,
            start_date=str(manifest_meta["start_date"]),
            top_n=int(config.top_n),
            policy="graph",
            output_dir=plots_dir,
        )

    _log(start_seconds, "Rendering wrapped daywise red-green linear-fit panels")
    wrapped_plot_path = (
        graph_plot_dir
        / f"daywise_green_red_linear_fit_scatters_wrapped_{int(config.plot_columns)}cols.png"
    )
    plot_wrapped_daywise_linear_relationships(
        metrics_path=extraction_dir / "matched_roi_log_ratio_metrics_complete.csv",
        fit_summary_path=(
            extraction_dir / "matched_daywise_green_red_linear_fit_summary.csv"
        ),
        output_path=wrapped_plot_path,
        start_date=str(manifest_meta["start_date"]),
        max_columns=int(config.plot_columns),
    )

    finished_utc = datetime.now(timezone.utc).isoformat()
    total_seconds = time.perf_counter() - start_seconds
    run_manifest = {
        "master_runner_version": MASTER_RUNNER_VERSION,
        "run_started_local": datetime.now().astimezone().isoformat(),
        "run_finished_utc": finished_utc,
        "duration_seconds": float(total_seconds),
        "duration_hms": _format_duration(total_seconds),
        "config": asdict(config),
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "manifest": {
            "n_sessions": int(manifest_meta["n_sessions"]),
            "n_required_sessions": int(manifest_meta["n_required_sessions"]),
            "first_date": manifest_meta["first_date"].isoformat(),
            "last_date": manifest_meta["last_date"].isoformat(),
            "session_ids": manifest_meta["session_ids"],
        },
        "agreement": agreement_summary,
        "extraction_annotation": extraction_annotation,
        "outputs": {
            "run_dir": str(run_dir),
            "matching_dir": str(match_dir),
            "matching_qc_dir": str(match_dir / "qc"),
            "extraction_dir": str(extraction_dir),
            "plots_dir": str(plots_dir),
            "wrapped_linear_fit_plot": str(wrapped_plot_path),
        },
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)
    _write_master_summary(
        run_dir=run_dir,
        config=config,
        agreement_summary=agreement_summary,
        extraction_annotation=extraction_annotation,
        match_dir=match_dir,
        extraction_dir=extraction_dir,
        plots_dir=plots_dir,
    )

    _log(start_seconds, f"Master pipeline completed in {_format_duration(total_seconds)}")
    print(f"output_dir={run_dir}", flush=True)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset alias or dataset directory.")
    parser.add_argument("--manifest", required=True, help="Daywise session manifest CSV.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Default: <dataset>/analysis/daywise_master_runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable run folder name. A dataset/date/session name is generated by default.",
    )
    parser.add_argument("--xy-um-per-px", type=float, default=710.0 / 1024.0)
    parser.add_argument("--z-um-per-plane", type=float, default=5.0)
    parser.add_argument("--max-pair-gap", type=int, default=2)
    parser.add_argument("--green-dark", type=float, default=319.0)
    parser.add_argument("--red-dark", type=float, default=534.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--plot-columns",
        type=int,
        default=7,
        help="Maximum number of day panels per row in the wrapped linear-fit plot.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-matching-qc", action="store_true")
    parser.add_argument("--skip-quick-plots", action="store_true")
    parser.add_argument("--require-qc-success", action="store_true")
    parser.add_argument("--qc-dpi", type=int, default=150)
    parser.add_argument("--qc-max-examples", type=int, default=20)
    parser.add_argument("--qc-max-total-examples", type=int, default=100)
    parser.add_argument("--qc-random-seed", type=int, default=0)
    parser.add_argument(
        "--segmentation-qc-mode",
        default="all_required",
        choices=["all_required", "fraction"],
    )
    parser.add_argument("--min-segmentation-pass-fraction", type=float, default=1.0)
    parser.add_argument("--min-volume-um3", type=float, default=None)
    parser.add_argument("--max-volume-um3", type=float, default=None)
    parser.add_argument("--min-bbox-depth-planes", type=float, default=None)
    parser.add_argument("--max-bbox-depth-planes", type=float, default=None)
    parser.add_argument("--exclude-xy-edge", action="store_true")
    parser.add_argument("--exclude-z-edge", action="store_true")
    parser.add_argument("--max-volume-ratio-from-track-median", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.overwrite and args.resume:
        raise ValueError("Choose only one of --overwrite or --resume.")
    if int(args.plot_columns) < 1:
        raise ValueError("--plot-columns must be at least 1.")
    config = MasterPipelineConfig(
        dataset=args.dataset,
        manifest=args.manifest,
        output_root=args.output_root,
        run_name=args.run_name,
        xy_um_per_px=args.xy_um_per_px,
        z_um_per_plane=args.z_um_per_plane,
        max_pair_gap=args.max_pair_gap,
        green_dark=args.green_dark,
        red_dark=args.red_dark,
        epsilon=args.epsilon,
        top_n=args.top_n,
        plot_columns=args.plot_columns,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        skip_matching_qc=bool(args.skip_matching_qc),
        skip_quick_plots=bool(args.skip_quick_plots),
        require_qc_success=bool(args.require_qc_success),
        qc_dpi=args.qc_dpi,
        qc_max_examples=args.qc_max_examples,
        qc_max_total_examples=args.qc_max_total_examples,
        qc_random_seed=args.qc_random_seed,
        segmentation_qc_mode=args.segmentation_qc_mode,
        min_segmentation_pass_fraction=args.min_segmentation_pass_fraction,
        min_volume_um3=args.min_volume_um3,
        max_volume_um3=args.max_volume_um3,
        min_bbox_depth_planes=args.min_bbox_depth_planes,
        max_bbox_depth_planes=args.max_bbox_depth_planes,
        exclude_xy_edge=bool(args.exclude_xy_edge),
        exclude_z_edge=bool(args.exclude_z_edge),
        max_volume_ratio_from_track_median=args.max_volume_ratio_from_track_median,
    )
    return run_master_pipeline(config)


if __name__ == "__main__":
    main()
