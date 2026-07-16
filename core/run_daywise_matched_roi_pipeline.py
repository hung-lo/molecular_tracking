"""Run daywise matched ROI extraction from daily masks and registered images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _import_dir in (_REPO_ROOT / "core", _REPO_ROOT / "matching"):
    _import_dir_str = str(_import_dir)
    if _import_dir_str not in sys.path:
        sys.path.append(_import_dir_str)

from analysis_paths import get_dataset_analysis_dir, resolve_dataset_dir
from roi_log_ratio_analysis import (
    apply_channel_dark_correction,
    compute_green_red_fit_residuals,
    compute_log_ratio_metrics,
    extract_roi_mean_intensities,
    filter_complete_rois,
    summarize_daily_green_red_linear_fits,
    summarize_residual_sign_changes,
    wide_table_from_long_table,
)
from session_manifest import SessionRecord, load_session_manifest, validate_manifest_for_intensity
from match_policy_registry import DEFAULT_ANALYSIS_POLICIES, SUPPORTED_MATCH_POLICIES, resolve_requested_policies

ANALYSIS_VERSION = "0.2.0"


@dataclass(frozen=True)
class SegmentationQCConfig:
    mode: str = "all_required"
    min_volume_um3: float | None = None
    max_volume_um3: float | None = None
    min_bbox_depth_planes: float | None = None
    max_bbox_depth_planes: float | None = None
    exclude_xy_edge: bool = False
    exclude_z_edge: bool = False
    max_volume_ratio_from_track_median: float | None = None
    min_segmentation_pass_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"all_required", "fraction"}:
            raise ValueError("mode must be either 'all_required' or 'fraction'.")
        if not 0.0 < float(self.min_segmentation_pass_fraction) <= 1.0:
            raise ValueError("min_segmentation_pass_fraction must be within (0, 1].")
        if self.max_volume_ratio_from_track_median is not None and float(self.max_volume_ratio_from_track_median) <= 0:
            raise ValueError("max_volume_ratio_from_track_median must be positive when provided.")


@dataclass(frozen=True)
class DaywiseMatchedPipelineConfig:
    dataset: str
    manifest: str
    match_dir: str
    policies: tuple[str, ...] = ("high", "balanced")
    green_dark: float = 319.0
    red_dark: float = 534.0
    epsilon: float = 1.0
    segmentation_qc_mode: str = "all_required"
    min_segmentation_pass_fraction: float = 1.0
    min_volume_um3: float | None = None
    max_volume_um3: float | None = None
    min_bbox_depth_planes: float | None = None
    max_bbox_depth_planes: float | None = None
    exclude_xy_edge: bool = False
    exclude_z_edge: bool = False
    max_volume_ratio_from_track_median: float | None = None


def format_duration_seconds(duration_seconds: float) -> str:
    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_stage_start(stage_label: str, pipeline_start_seconds: float) -> float:
    elapsed = time.perf_counter() - pipeline_start_seconds
    print(f"[{format_duration_seconds(elapsed)}] {stage_label}...", flush=True)
    return time.perf_counter()


def log_stage_end(
    stage_label: str,
    pipeline_start_seconds: float,
    stage_start_seconds: float,
    stage_durations_seconds: dict[str, float],
    stage_key: str,
    detail: str | None = None,
) -> None:
    stage_duration = time.perf_counter() - stage_start_seconds
    stage_durations_seconds[stage_key] = float(stage_duration)
    elapsed = time.perf_counter() - pipeline_start_seconds
    detail_suffix = "" if not detail else f" | {detail}"
    print(
        f"[{format_duration_seconds(elapsed)}] Finished {stage_label} "
        f"({stage_duration:.1f}s){detail_suffix}",
        flush=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _package_versions() -> dict[str, str | None]:
    return {
        "numpy": _safe_package_version("numpy"),
        "pandas": _safe_package_version("pandas"),
        "scipy": _safe_package_version("scipy"),
        "scikit-image": _safe_package_version("scikit-image"),
        "tifffile": _safe_package_version("tifffile"),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _manifest_dataframe(records: list[SessionRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "session_index": int(record.session_index),
                "session_id": str(record.session_id),
                "acquisition_date": record.acquisition_date.isoformat(),
                "mask_path": str(record.mask_path),
                "red_image_path": str(record.red_image_path) if record.red_image_path is not None else "",
                "green_image_path": str(record.green_image_path) if record.green_image_path is not None else "",
                "required": bool(record.required),
            }
        )
    return pd.DataFrame(rows)


def _normalize_manifest_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    output["session_index"] = pd.to_numeric(output["session_index"], errors="raise").astype(int)
    output["session_id"] = output["session_id"].astype(str)
    output["acquisition_date"] = output["acquisition_date"].astype(str)
    for column in ["mask_path", "red_image_path", "green_image_path"]:
        if column in output.columns:
            output[column] = output[column].fillna("").astype(str)
    if "required" in output.columns:
        output["required"] = output["required"].map(lambda value: str(value).strip().lower() == "true")
    return output.sort_values("session_index").reset_index(drop=True)


def _elapsed_days_lookup(records: list[SessionRecord]) -> dict[str, int]:
    required_dates = [record.acquisition_date for record in records if record.required]
    if not required_dates:
        return {str(record.session_id): 0 for record in records}
    start_date = min(required_dates)
    return {str(record.session_id): int((record.acquisition_date - start_date).days) for record in records}


def _load_mask_stack(path: Path) -> np.ndarray:
    try:
        return np.asarray(tifffile.memmap(path))
    except Exception:
        return tifffile.imread(path)


def _load_image_stack(path: Path) -> np.ndarray:
    return tifffile.imread(path)


def _load_match_dir(match_dir: Path) -> dict[str, Path]:
    required = {
        "session_manifest_resolved": match_dir / "session_manifest_resolved.csv",
        "roi_features": match_dir / "roi_features.csv",
        "run_log": match_dir / "run_log.json",
    }
    optional = {
        "tracks_high": match_dir / "tracks_high.csv",
        "tracks_balanced": match_dir / "tracks_balanced.csv",
        "tracks_graph": match_dir / "tracks_graph.csv",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing matcher output file {name!r}: {path}")
    for name, path in optional.items():
        if path.exists():
            required[name] = path
    return required


def _policy_observation_lookup(
    match_table: pd.DataFrame,
    records: list[SessionRecord],
    policy: str,
) -> pd.DataFrame:
    elapsed_days_by_session = _elapsed_days_lookup(records)
    rows: list[dict[str, object]] = []
    for _, track_row in match_table.iterrows():
        cluster_id = int(track_row["cluster_id"])
        track_uid = str(track_row["track_uid"])
        for record in records:
            label_value = track_row.get(f"{record.session_id}_roi", pd.NA)
            if pd.isna(label_value):
                continue
            rows.append(
                {
                    "match_policy": policy,
                    "roi_id": cluster_id,
                    "track_uid": track_uid,
                    "cluster_id": cluster_id,
                    "session_index": int(record.session_index),
                    "day": int(record.session_index),
                    "session_id": str(record.session_id),
                    "acquisition_date": record.acquisition_date.isoformat(),
                    "elapsed_days": int(elapsed_days_by_session[str(record.session_id)]),
                    "mask_label": int(label_value),
                    "required": bool(record.required),
                }
            )
    return pd.DataFrame(rows)


def _extract_policy_raw_table(
    *,
    policy: str,
    tracks_table: pd.DataFrame,
    records: list[SessionRecord],
    green_dark: float,
    red_dark: float,
) -> pd.DataFrame:
    observation_lookup = _policy_observation_lookup(tracks_table, records, policy)
    if observation_lookup.empty:
        return pd.DataFrame(
            columns=[
                "match_policy",
                "roi_id",
                "track_uid",
                "cluster_id",
                "session_index",
                "day",
                "session_id",
                "acquisition_date",
                "elapsed_days",
                "mask_label",
                "required",
                "channel",
                "mean_intensity",
                "image",
                "image_path",
                "mask_path",
                "zero_hit_pass",
                "dark_value",
            ]
        )

    rows: list[pd.DataFrame] = []
    for record in records:
        mask_stack = _load_mask_stack(record.mask_path)
        session_observations = observation_lookup.loc[observation_lookup["session_id"] == record.session_id].copy()
        if session_observations.empty:
            continue
        for channel, dark_value in (("red", red_dark), ("green", green_dark)):
            image_path = record.red_image_path if channel == "red" else record.green_image_path
            assert image_path is not None
            image_stack = _load_image_stack(image_path)
            extracted_rows = extract_roi_mean_intensities(
                image_stack=image_stack,
                mask_stack=mask_stack,
                exclude_zero_pixels=True,
            )
            extracted_table = pd.DataFrame(extracted_rows)
            if extracted_table.empty:
                extracted_table = pd.DataFrame(columns=["roi_id", "mean_intensity"])
            extracted_table = extracted_table.rename(columns={"roi_id": "mask_label"})
            merged = session_observations.merge(
                extracted_table,
                on="mask_label",
                how="left",
                validate="many_to_one",
            )
            merged["channel"] = channel
            merged["mean_intensity"] = merged["mean_intensity"].astype(float)
            merged["zero_hit_pass"] = merged["mean_intensity"].notna()
            merged["image"] = image_path.name
            merged["image_path"] = str(image_path)
            merged["mask_path"] = str(record.mask_path)
            merged["dark_value"] = float(dark_value)
            rows.append(merged)

    raw_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not raw_table.empty:
        raw_table = raw_table.sort_values(
            ["match_policy", "roi_id", "day", "channel"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    return raw_table


def _track_lookup(tracks_table: pd.DataFrame, policy: str, records: list[SessionRecord]) -> pd.DataFrame:
    elapsed_days_by_session = _elapsed_days_lookup(records)
    rows: list[dict[str, object]] = []
    for _, track_row in tracks_table.iterrows():
        cluster_id = int(track_row["cluster_id"])
        track_uid = str(track_row["track_uid"])
        for record in records:
            label_value = track_row.get(f"{record.session_id}_roi", pd.NA)
            if pd.isna(label_value):
                continue
            rows.append(
                {
                    "match_policy": policy,
                    "roi_id": cluster_id,
                    "track_uid": track_uid,
                    "cluster_id": cluster_id,
                    "session_index": int(record.session_index),
                    "day": int(record.session_index),
                    "session_id": str(record.session_id),
                    "acquisition_date": record.acquisition_date.isoformat(),
                    "elapsed_days": int(elapsed_days_by_session[str(record.session_id)]),
                    "mask_label": int(label_value),
                    "required": bool(record.required),
                }
            )
    return pd.DataFrame(rows)


def _build_roi_day_table(corrected_table: pd.DataFrame, track_lookup: pd.DataFrame) -> pd.DataFrame:
    wide_table = wide_table_from_long_table(
        corrected_table,
        intensity_column="mean_intensity_corrected",
    )
    metadata = track_lookup.loc[
        :, [
            "roi_id",
            "day",
            "track_uid",
            "cluster_id",
            "match_policy",
            "session_index",
            "session_id",
            "acquisition_date",
            "elapsed_days",
            "required",
        ]
    ].drop_duplicates()
    roi_day_table = wide_table.merge(
        metadata,
        on=["roi_id", "day"],
        how="left",
        validate="one_to_one",
    )
    return roi_day_table.sort_values(["match_policy", "roi_id", "day"]).reset_index(drop=True)


def _apply_geometry_qc(geometry_long: pd.DataFrame, qc_config: SegmentationQCConfig) -> pd.DataFrame:
    table = geometry_long.copy()
    if table.empty:
        table["segmentation_qc_status"] = pd.Series(dtype=str)
        table["track_median_volume_um3"] = pd.Series(dtype=float)
        table["volume_ratio_to_track_median"] = pd.Series(dtype=float)
        table["volume_fold_change_to_track_median"] = pd.Series(dtype=float)
        table["geometry_qc_pass"] = pd.Series(dtype="boolean")
        for column in [
            "geometry_min_volume_pass",
            "geometry_max_volume_pass",
            "geometry_min_bbox_depth_pass",
            "geometry_max_bbox_depth_pass",
            "geometry_xy_edge_pass",
            "geometry_z_edge_pass",
            "geometry_volume_ratio_pass",
        ]:
            table[column] = pd.Series(dtype="boolean")
        return table

    table["track_median_volume_um3"] = table.groupby("roi_id")["volume_um3"].transform("median")
    table["volume_ratio_to_track_median"] = table["volume_um3"] / table["track_median_volume_um3"]
    table["volume_fold_change_to_track_median"] = np.maximum(
        table["volume_ratio_to_track_median"],
        1.0 / table["volume_ratio_to_track_median"],
    )
    configured = any(
        value is not None
        for value in [
            qc_config.min_volume_um3,
            qc_config.max_volume_um3,
            qc_config.min_bbox_depth_planes,
            qc_config.max_bbox_depth_planes,
            qc_config.max_volume_ratio_from_track_median,
        ]
    ) or qc_config.exclude_xy_edge or qc_config.exclude_z_edge
    table["segmentation_qc_status"] = "configured" if configured else "not_configured"
    if not configured:
        table["geometry_qc_pass"] = pd.NA
        for column in [
            "geometry_min_volume_pass",
            "geometry_max_volume_pass",
            "geometry_min_bbox_depth_pass",
            "geometry_max_bbox_depth_pass",
            "geometry_xy_edge_pass",
            "geometry_z_edge_pass",
            "geometry_volume_ratio_pass",
        ]:
            table[column] = pd.NA
        return table

    table["geometry_min_volume_pass"] = True
    table["geometry_max_volume_pass"] = True
    table["geometry_min_bbox_depth_pass"] = True
    table["geometry_max_bbox_depth_pass"] = True
    table["geometry_xy_edge_pass"] = True
    table["geometry_z_edge_pass"] = True
    table["geometry_volume_ratio_pass"] = True

    if qc_config.min_volume_um3 is not None:
        table["geometry_min_volume_pass"] = table["volume_um3"] >= float(qc_config.min_volume_um3)
    if qc_config.max_volume_um3 is not None:
        table["geometry_max_volume_pass"] = table["volume_um3"] <= float(qc_config.max_volume_um3)
    if qc_config.min_bbox_depth_planes is not None:
        table["geometry_min_bbox_depth_pass"] = table["bbox_depth_planes"] >= float(qc_config.min_bbox_depth_planes)
    if qc_config.max_bbox_depth_planes is not None:
        table["geometry_max_bbox_depth_pass"] = table["bbox_depth_planes"] <= float(qc_config.max_bbox_depth_planes)
    if qc_config.exclude_xy_edge:
        table["geometry_xy_edge_pass"] = ~table["touches_xy_edge"].astype(bool)
    if qc_config.exclude_z_edge:
        table["geometry_z_edge_pass"] = ~table["touches_z_edge"].astype(bool)
    if qc_config.max_volume_ratio_from_track_median is not None:
        threshold = float(qc_config.max_volume_ratio_from_track_median)
        table["geometry_volume_ratio_pass"] = table["volume_fold_change_to_track_median"] <= threshold

    geometry_columns = [
        "geometry_min_volume_pass",
        "geometry_max_volume_pass",
        "geometry_min_bbox_depth_pass",
        "geometry_max_bbox_depth_pass",
        "geometry_xy_edge_pass",
        "geometry_z_edge_pass",
        "geometry_volume_ratio_pass",
    ]
    table["geometry_qc_pass"] = table[geometry_columns].all(axis=1)
    return table


def _track_geometry_summary(geometry_long: pd.DataFrame, qc_config: SegmentationQCConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for roi_id, group in geometry_long.groupby("roi_id", sort=True):
        required_group = group.loc[group["required"].astype(bool)].copy()
        n_required_sessions = int(required_group["session_id"].nunique())
        if not required_group.empty and "geometry_qc_pass" in required_group:
            n_geometry_qc_pass = int(required_group["geometry_qc_pass"].fillna(False).astype(bool).sum())
        else:
            n_geometry_qc_pass = 0
        pass_fraction = float(n_geometry_qc_pass / n_required_sessions) if n_required_sessions > 0 else np.nan
        if qc_config.mode == "fraction":
            segmentation_qc_pass_all_required_days = bool(pass_fraction >= float(qc_config.min_segmentation_pass_fraction))
        else:
            segmentation_qc_pass_all_required_days = bool(n_required_sessions > 0 and n_geometry_qc_pass == n_required_sessions)
        if group["segmentation_qc_status"].iloc[0] == "not_configured":
            segmentation_qc_pass_all_required_days = pd.NA
        n_edge_sessions = int((group["touches_xy_edge"].astype(bool) | group["touches_z_edge"].astype(bool)).sum())
        volumes = pd.to_numeric(group["volume_um3"], errors="coerce").dropna()
        max_volume_fold_change = float(volumes.max() / volumes.min()) if len(volumes) >= 2 and volumes.min() > 0 else np.nan
        rows.append(
            {
                "roi_id": int(roi_id),
                "segmentation_qc_status": str(group["segmentation_qc_status"].iloc[0]),
                "n_required_sessions": n_required_sessions,
                "n_geometry_qc_pass": n_geometry_qc_pass,
                "segmentation_qc_pass_fraction": pass_fraction,
                "segmentation_qc_pass_all_required_days": segmentation_qc_pass_all_required_days,
                "n_edge_sessions": n_edge_sessions,
                "max_volume_fold_change": max_volume_fold_change,
                "track_median_volume_um3": float(volumes.median()) if len(volumes) else np.nan,
                "min_volume_um3": float(volumes.min()) if len(volumes) else np.nan,
                "max_volume_um3": float(volumes.max()) if len(volumes) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("roi_id").reset_index(drop=True)
    return summary


def _compute_centroid_steps(geometry_long: pd.DataFrame) -> pd.DataFrame:
    table = geometry_long.sort_values(["roi_id", "session_index"]).copy()
    table["centroid_step_um_from_previous_observed_session"] = np.nan
    for _, group in table.groupby("roi_id", sort=False):
        previous = None
        for index in group.index:
            current = np.asarray(
                [
                    float(table.at[index, "centroid_z_um"]),
                    float(table.at[index, "centroid_y_um"]),
                    float(table.at[index, "centroid_x_um"]),
                ],
                dtype=float,
            )
            if previous is not None:
                table.at[index, "centroid_step_um_from_previous_observed_session"] = float(np.linalg.norm(current - previous))
            previous = current
    return table


def _roi_presence_summary(roi_day_table: pd.DataFrame, required_days: list[int]) -> pd.DataFrame:
    if roi_day_table.empty:
        return pd.DataFrame(
            columns=[
                "roi_id",
                "n_required_sessions_present",
                "n_required_sessions_missing",
                "missing_required_internal_days",
                "all_required_sessions_present",
                "exactly_one_required_session_missing",
            ]
        )

    valid_rows = roi_day_table.dropna(subset=["red", "green"]).copy()
    required_days = [int(day) for day in required_days]
    total_required = int(len(required_days))
    day_positions = {day: position for position, day in enumerate(required_days)}
    rows: list[dict[str, object]] = []
    for roi_id, group in valid_rows.groupby("roi_id", sort=True):
        present_required_days = sorted({int(day) for day in group.loc[group["required"].astype(bool), "day"] if int(day) in day_positions})
        present_set = set(present_required_days)
        n_present = len(present_set)
        n_missing = total_required - n_present
        missing_internal = 0
        if n_present > 0:
            positions = [day_positions[day] for day in present_set]
            first_pos = min(positions)
            last_pos = max(positions)
            missing_internal = sum(1 for day in required_days[first_pos : last_pos + 1] if day not in present_set)
        rows.append(
            {
                "roi_id": int(roi_id),
                "n_required_sessions_present": n_present,
                "n_required_sessions_missing": n_missing,
                "missing_required_internal_days": int(missing_internal),
                "all_required_sessions_present": bool(n_missing == 0),
                "exactly_one_required_session_missing": bool(n_missing == 1),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("roi_id").reset_index(drop=True)
    return summary


def _merge_reasons(existing: Any, additions: list[str]) -> str:
    reasons: list[str] = []
    if existing is not None and not pd.isna(existing):
        for reason in str(existing).split(","):
            reason = reason.strip()
            if reason:
                reasons.append(reason)
    for reason in additions:
        if reason and reason not in reasons:
            reasons.append(reason)
    return ",".join(reasons)


def _policy_analysis(
    *,
    policy: str,
    tracks_table: pd.DataFrame,
    records: list[SessionRecord],
    roi_features: pd.DataFrame,
    green_dark: float,
    red_dark: float,
    epsilon: float,
    required_days: list[int],
    qc_config: SegmentationQCConfig,
) -> dict[str, Any]:
    tracks_table = tracks_table.copy()
    tracks_table["roi_id"] = pd.to_numeric(tracks_table["cluster_id"], errors="raise").astype(int)
    track_lookup = _track_lookup(tracks_table, policy, records)
    raw_table = _extract_policy_raw_table(
        policy=policy,
        tracks_table=tracks_table,
        records=records,
        green_dark=green_dark,
        red_dark=red_dark,
    )
    empty_result = {
        "raw_table": raw_table,
        "corrected_table": raw_table,
        "roi_day_table": pd.DataFrame(),
        "complete_table": pd.DataFrame(),
        "metrics_table": pd.DataFrame(),
        "fit_summary": pd.DataFrame(),
        "residual_table": pd.DataFrame(),
        "residual_summary": pd.DataFrame(),
        "geometry_long": pd.DataFrame(),
        "track_summary": pd.DataFrame(),
        "matched_tracks": pd.DataFrame(),
        "filter_counts": pd.DataFrame(),
        "primary_matching": pd.DataFrame(),
        "primary_full_qc": pd.DataFrame(),
        "sensitivity_complete": pd.DataFrame(),
        "one_gap": pd.DataFrame(),
        "review_flagged": pd.DataFrame(),
        "complete_track_set": pd.DataFrame(),
        "one_gap_track_set": pd.DataFrame(),
    }
    if raw_table.empty:
        return empty_result

    corrected_table = apply_channel_dark_correction(
        intensity_table=raw_table,
        green_dark=green_dark,
        red_dark=red_dark,
        intensity_column="mean_intensity",
        corrected_column="mean_intensity_corrected",
        clip_floor=None,
    )

    roi_day_table = _build_roi_day_table(corrected_table, track_lookup)
    complete_table = filter_complete_rois(roi_day_table, required_days=required_days)

    metrics_table = pd.DataFrame()
    fit_summary = pd.DataFrame()
    residual_table = pd.DataFrame()
    residual_summary = pd.DataFrame()
    if not complete_table.empty:
        metrics_table = compute_log_ratio_metrics(complete_table, epsilon=epsilon)
        fit_summary = summarize_daily_green_red_linear_fits(metrics_table)
        fit_summary.insert(0, "match_policy", policy)
        residual_table = compute_green_red_fit_residuals(
            metrics_table,
            fit_summary=fit_summary.drop(columns=["match_policy"]),
        )
        residual_summary = summarize_residual_sign_changes(residual_table)
        residual_summary = residual_summary.merge(
            track_lookup.loc[:, ["roi_id", "track_uid", "cluster_id", "match_policy"]].drop_duplicates("roi_id"),
            on="roi_id",
            how="left",
            validate="one_to_one",
        )
        residual_table = residual_table.merge(
            residual_summary.loc[:, ["roi_id", "track_uid", "cluster_id", "match_policy"]],
            on="roi_id",
            how="left",
            validate="many_to_one",
        )

    geometry_long = track_lookup.merge(
        roi_features,
        left_on=["session_id", "mask_label"],
        right_on=["session_id", "label"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_feature"),
    )
    geometry_long = _compute_centroid_steps(geometry_long)
    geometry_long = _apply_geometry_qc(geometry_long, qc_config)
    track_summary = _track_geometry_summary(geometry_long, qc_config)
    matched_tracks = tracks_table.copy().merge(track_summary, on="roi_id", how="left", validate="one_to_one")
    presence_summary = _roi_presence_summary(roi_day_table, required_days)
    matched_tracks = matched_tracks.merge(presence_summary, on="roi_id", how="left", validate="one_to_one")

    matched_tracks["n_required_sessions_present"] = matched_tracks["n_required_sessions_present"].fillna(0).astype(int)
    matched_tracks["n_required_sessions_missing"] = matched_tracks["n_required_sessions_missing"].fillna(len(required_days)).astype(int)
    matched_tracks["missing_required_internal_days"] = matched_tracks["missing_required_internal_days"].fillna(0).astype(int)
    matched_tracks["all_required_sessions_present"] = matched_tracks["all_required_sessions_present"].fillna(False).astype(bool)
    matched_tracks["exactly_one_required_session_missing"] = matched_tracks["exactly_one_required_session_missing"].fillna(False).astype(bool)
    matched_tracks["segmentation_failure"] = (
        matched_tracks["segmentation_qc_status"].astype(str) == "configured"
    ) & (~matched_tracks["segmentation_qc_pass_all_required_days"].fillna(False).astype(bool))
    matched_tracks["edge_heavy"] = matched_tracks["n_edge_sessions"].fillna(0).astype(int) > 0
    matched_tracks["review_required"] = matched_tracks["review_required"].fillna(False).astype(bool)
    matched_tracks["review_required"] = matched_tracks["review_required"] | matched_tracks["segmentation_failure"] | matched_tracks["edge_heavy"]
    matched_tracks["review_reasons"] = matched_tracks.apply(
        lambda row: _merge_reasons(
            row.get("review_reasons", ""),
            [
                "segmentation_failure" if bool(row.get("segmentation_failure", False)) else "",
                "edge_heavy" if bool(row.get("edge_heavy", False)) else "",
            ],
        ),
        axis=1,
    )

    total_required = len(required_days)
    complete_required = matched_tracks.loc[
        (matched_tracks["n_required_sessions_present"] >= total_required)
        & (matched_tracks["missing_required_internal_days"] == 0)
    ].copy()
    zero_hit_complete = complete_required.copy()
    cycle_qc = zero_hit_complete.loc[~zero_hit_complete["has_cycle_conflict"].fillna(False).astype(bool)].copy()
    if qc_config.mode == "fraction" or geometry_long["segmentation_qc_status"].iloc[0] != "not_configured":
        segmentation_qc = cycle_qc.loc[cycle_qc["segmentation_qc_pass_all_required_days"].fillna(False).astype(bool)].copy()
    else:
        segmentation_qc = cycle_qc.iloc[0:0].copy()
    one_gap_candidates = matched_tracks.loc[
        (matched_tracks["n_required_sessions_present"] == max(total_required - 1, 0))
        & (matched_tracks["missing_required_internal_days"] == 1)
        & (matched_tracks["n_days_present"].astype(int) >= 2)
    ].copy()
    one_gap_track_set = one_gap_candidates.loc[~one_gap_candidates["has_cycle_conflict"].fillna(False).astype(bool)].copy()

    primary_matching = cycle_qc.loc[cycle_qc["match_policy"].eq("high")].copy() if policy == "high" else cycle_qc.iloc[0:0].copy()
    sensitivity_complete = cycle_qc.loc[cycle_qc["match_policy"].eq("balanced")].copy() if policy == "balanced" else cycle_qc.iloc[0:0].copy()
    if policy == "high":
        primary_full_qc = primary_matching.loc[primary_matching["roi_id"].isin(segmentation_qc["roi_id"])].copy()
    else:
        primary_full_qc = primary_matching.iloc[0:0].copy()

    if policy == "high":
        primary_final = primary_full_qc if not primary_full_qc.empty else segmentation_qc.loc[segmentation_qc["match_policy"].eq("high")].copy()
    else:
        primary_final = segmentation_qc.copy()

    review_flagged = matched_tracks.loc[
        matched_tracks["review_required"].fillna(False).astype(bool)
        | matched_tracks["segmentation_failure"].fillna(False).astype(bool)
        | matched_tracks["edge_heavy"].fillna(False).astype(bool)
    ].copy()

    filter_rows = []
    filter_rows.append({"match_policy": policy, "step_order": 0, "step": "all_tracks", "count": int(len(matched_tracks))})
    filter_rows.append({"match_policy": policy, "step_order": 1, "step": "at_least_two_sessions", "count": int(len(matched_tracks.loc[matched_tracks["n_days_present"].astype(int) >= 2]))})
    filter_rows.append({"match_policy": policy, "step_order": 2, "step": "complete_required_sessions", "count": int(len(complete_required))})
    filter_rows.append({"match_policy": policy, "step_order": 3, "step": "zero_hit_complete", "count": int(len(zero_hit_complete))})
    filter_rows.append({"match_policy": policy, "step_order": 4, "step": "cycle_qc", "count": int(len(cycle_qc))})
    filter_rows.append({"match_policy": policy, "step_order": 5, "step": "segmentation_qc", "count": int(len(segmentation_qc))})
    filter_rows.append({"match_policy": policy, "step_order": 6, "step": "primary_final", "count": int(len(primary_final))})
    filter_rows.append({"match_policy": policy, "step_order": 7, "step": "one_internal_gap", "count": int(len(one_gap_track_set))})
    filter_counts = pd.DataFrame(filter_rows)
    if not filter_counts.empty:
        starting_count = float(filter_counts.loc[filter_counts["step_order"] == 0, "count"].iloc[0])
        filter_counts["pct_of_start"] = np.where(starting_count > 0, 100.0 * filter_counts["count"].astype(float) / starting_count, 0.0)
        previous_counts = filter_counts["count"].shift(1, fill_value=starting_count).astype(float)
        filter_counts["pct_of_previous"] = np.where(previous_counts > 0, 100.0 * filter_counts["count"].astype(float) / previous_counts, 0.0)
    else:
        filter_counts["pct_of_start"] = []
        filter_counts["pct_of_previous"] = []

    return {
        "raw_table": raw_table,
        "corrected_table": corrected_table,
        "roi_day_table": roi_day_table,
        "complete_table": complete_table,
        "metrics_table": metrics_table,
        "fit_summary": fit_summary,
        "residual_table": residual_table,
        "residual_summary": residual_summary,
        "geometry_long": geometry_long,
        "track_summary": track_summary,
        "matched_tracks": matched_tracks,
        "filter_counts": filter_counts,
        "primary_matching": primary_matching,
        "primary_full_qc": primary_full_qc,
        "sensitivity_complete": sensitivity_complete,
        "one_gap": one_gap_track_set,
        "review_flagged": review_flagged,
        "complete_track_set": cycle_qc,
        "one_gap_track_set": one_gap_track_set,
    }


def _write_summary_markdown(
    output_dir: Path,
    config: SegmentationQCConfig,
    filter_counts: pd.DataFrame,
    fit_summary: pd.DataFrame,
    manifest_records: list[SessionRecord],
    primary_matching: pd.DataFrame,
    primary_full_qc: pd.DataFrame,
    sensitivity_complete: pd.DataFrame,
    one_gap_high: pd.DataFrame,
    one_gap_balanced: pd.DataFrame,
) -> None:
    summary_lines = [
        "# Daywise Matched ROI Pipeline",
        "",
        "Goal: measure matched ROI clusters across daily Cellpose-SAM masks using",
        "daywise registered image stacks, then apply the same zero-hit and",
        "complete-day filtering used in the registered ROI workflow.",
        "",
        "Key analysis choices:",
        f"- Segmentation QC mode: `{config.mode}`",
        f"- Dark values: green `{319.0}`, red `{534.0}`",
        "",
        "Sessions:",
    ]
    for record in manifest_records:
        summary_lines.append(
            f"- {record.session_id}: day {record.session_index} ({record.acquisition_date.isoformat()})"
        )
    summary_lines.extend(["", "Filter summary:"])
    for _, row in filter_counts.sort_values(["match_policy", "step_order"]).iterrows():
        summary_lines.append(
            f"- {row['match_policy']} / {row['step']}: `{int(row['count'])}` "
            f"({float(row['pct_of_start']):.1f}% of start)"
        )
    summary_lines.extend(
        [
            "",
            "Primary / sensitivity counts:",
            f"- high complete matching: `{len(primary_matching)}`",
            f"- high complete full QC: `{len(primary_full_qc)}`",
            f"- balanced complete: `{len(sensitivity_complete)}`",
            f"- high one-gap sensitivity: `{len(one_gap_high)}`",
            f"- balanced one-gap sensitivity: `{len(one_gap_balanced)}`",
        ]
    )
    if not fit_summary.empty:
        summary_lines.extend(["", "Green-vs-red fit summary:"])
        for _, row in fit_summary.sort_values(["match_policy", "day"]).iterrows():
            summary_lines.append(
                f"- {row['match_policy']} day {int(row['day'])}: slope `{row['slope']:.3f}`, "
                f"intercept `{row['intercept']:.1f}`, R² `{row['r_squared']:.3f}`"
            )
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")


def run_daywise_matched_roi_pipeline(config: DaywiseMatchedPipelineConfig) -> Path:
    pipeline_start_seconds = time.perf_counter()
    stage_durations_seconds: dict[str, float] = {}
    run_started_utc = datetime.now(timezone.utc).isoformat()

    resolve_dataset_dir(config.dataset)
    match_dir = Path(config.match_dir).resolve()
    if not match_dir.exists():
        raise FileNotFoundError(f"match_dir was not found: {match_dir}")

    manifest_records = load_session_manifest(config.manifest)
    validate_manifest_for_intensity(manifest_records)
    required_manifest_records = [record for record in manifest_records if record.required]
    required_days = [int(record.session_index) for record in required_manifest_records]
    match_paths = _load_match_dir(match_dir)

    expected_manifest = _normalize_manifest_dataframe(_manifest_dataframe(manifest_records))
    match_manifest = _normalize_manifest_dataframe(pd.read_csv(match_paths["session_manifest_resolved"]))
    expected_columns = [
        "session_index",
        "session_id",
        "acquisition_date",
        "mask_path",
        "red_image_path",
        "green_image_path",
        "required",
    ]
    if list(match_manifest.columns) != expected_columns or not match_manifest.equals(expected_manifest):
        raise ValueError("The resolved manifest in match_dir does not match the provided manifest.")

    output_root = get_dataset_analysis_dir(config.dataset)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"daywise_matched_roi_pipeline_{match_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"[{format_duration_seconds(0.0)}] Pipeline start | dataset={config.dataset} | "
        f"manifest={config.manifest} | match_dir={match_dir.name}",
        flush=True,
    )
    print(f"[{format_duration_seconds(0.0)}] Output directory: {output_dir}", flush=True)

    resolved_manifest = _manifest_dataframe(manifest_records)
    resolved_manifest.to_csv(output_dir / "session_manifest_resolved.csv", index=False)

    roi_features = pd.read_csv(match_paths["roi_features"])
    if "session_id" in roi_features.columns:
        roi_features["session_id"] = roi_features["session_id"].astype(str)
    requested_policies = resolve_requested_policies(config.policies, match_dir)
    policy_inputs: dict[str, pd.DataFrame] = {}
    for policy in requested_policies:
        tracks_path = match_paths.get(f"tracks_{policy}")
        if tracks_path is None:
            raise FileNotFoundError(f"Requested policy {policy!r} is not available in {match_dir}.")
        policy_inputs[policy] = pd.read_csv(tracks_path)

    qc_config = SegmentationQCConfig(
        mode=config.segmentation_qc_mode,
        min_volume_um3=config.min_volume_um3,
        max_volume_um3=config.max_volume_um3,
        min_bbox_depth_planes=config.min_bbox_depth_planes,
        max_bbox_depth_planes=config.max_bbox_depth_planes,
        exclude_xy_edge=config.exclude_xy_edge,
        exclude_z_edge=config.exclude_z_edge,
        max_volume_ratio_from_track_median=config.max_volume_ratio_from_track_median,
        min_segmentation_pass_fraction=config.min_segmentation_pass_fraction,
    )

    policy_results: dict[str, dict[str, Any]] = {}
    for policy in requested_policies:
        stage_start_seconds = log_stage_start(
            stage_label=f"Extract matched intensities for {policy}",
            pipeline_start_seconds=pipeline_start_seconds,
        )
        policy_result = _policy_analysis(
            policy=policy,
            tracks_table=policy_inputs[policy],
            records=manifest_records,
            roi_features=roi_features,
            green_dark=config.green_dark,
            red_dark=config.red_dark,
            epsilon=config.epsilon,
            required_days=required_days,
            qc_config=qc_config,
        )
        policy_results[policy] = policy_result
        log_stage_end(
            stage_label=f"Extract matched intensities for {policy}",
            pipeline_start_seconds=pipeline_start_seconds,
            stage_start_seconds=stage_start_seconds,
            stage_durations_seconds=stage_durations_seconds,
            stage_key=f"{policy}_analysis",
            detail=f"{len(policy_result['complete_table'])} complete ROI/day rows",
        )

    def _concat(policy_key: str) -> pd.DataFrame:
        tables = [policy_results[policy][policy_key] for policy in requested_policies if policy in policy_results]
        if not tables:
            return pd.DataFrame()
        return pd.concat(tables, ignore_index=True)

    raw_table = _concat("raw_table")
    corrected_table = _concat("corrected_table")
    roi_day_table = _concat("roi_day_table")
    complete_table = _concat("complete_table")
    metrics_table = _concat("metrics_table")
    fit_summary = _concat("fit_summary")
    residual_table = _concat("residual_table")
    residual_summary = _concat("residual_summary")
    geometry_long = _concat("geometry_long")
    matched_tracks = _concat("matched_tracks")
    filter_counts = _concat("filter_counts")
    primary_matching = _concat("primary_matching")
    primary_full_qc = _concat("primary_full_qc")
    sensitivity_complete = _concat("sensitivity_complete")
    one_gap_high = policy_results.get("high", {}).get("one_gap", pd.DataFrame())
    one_gap_balanced = policy_results.get("balanced", {}).get("one_gap", pd.DataFrame())
    review_flagged = _concat("review_flagged")

    def _sort_if_present(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if table.empty:
            return table
        existing_columns = [column for column in columns if column in table.columns]
        if not existing_columns:
            return table.reset_index(drop=True)
        return table.sort_values(existing_columns).reset_index(drop=True)

    raw_table = _sort_if_present(raw_table, ["match_policy", "roi_id", "day", "channel"])
    corrected_table = _sort_if_present(corrected_table, ["match_policy", "roi_id", "day", "channel"])
    roi_day_table = _sort_if_present(roi_day_table, ["match_policy", "roi_id", "day"])
    complete_table = _sort_if_present(complete_table, ["match_policy", "roi_id", "day"])
    metrics_table = _sort_if_present(metrics_table, ["match_policy", "roi_id", "day", "channel"])
    fit_summary = _sort_if_present(fit_summary, ["match_policy", "day"])
    residual_table = _sort_if_present(residual_table, ["match_policy", "roi_id", "day", "channel"])
    residual_summary = _sort_if_present(residual_summary, ["match_policy", "roi_id"])
    geometry_long = _sort_if_present(geometry_long, ["match_policy", "track_uid", "session_index"])
    matched_tracks = _sort_if_present(matched_tracks, ["match_policy", "cluster_id"])
    filter_counts = _sort_if_present(filter_counts, ["match_policy", "step_order"])
    primary_matching = _sort_if_present(primary_matching, ["match_policy", "cluster_id"])
    primary_full_qc = _sort_if_present(primary_full_qc, ["match_policy", "cluster_id"])
    sensitivity_complete = _sort_if_present(sensitivity_complete, ["match_policy", "cluster_id"])
    one_gap_high = _sort_if_present(one_gap_high, ["match_policy", "cluster_id"])
    one_gap_balanced = _sort_if_present(one_gap_balanced, ["match_policy", "cluster_id"])
    review_flagged = _sort_if_present(review_flagged, ["match_policy", "cluster_id"])

    raw_table.to_csv(output_dir / "matched_roi_intensity_results_raw.csv", index=False)
    corrected_table.to_csv(output_dir / "matched_roi_intensity_results_dark_corrected.csv", index=False)
    geometry_long.to_csv(output_dir / "matched_roi_geometry_qc_long.csv", index=False)
    matched_tracks.to_csv(output_dir / "matched_track_qc_summary.csv", index=False)
    roi_day_table.to_csv(output_dir / "matched_roi_day_table_complete.csv", index=False)
    metrics_table.to_csv(output_dir / "matched_roi_log_ratio_metrics_complete.csv", index=False)
    fit_summary.to_csv(output_dir / "matched_daywise_green_red_linear_fit_summary.csv", index=False)
    residual_table.to_csv(output_dir / "matched_roi_metrics_with_green_red_fit_residuals.csv", index=False)
    residual_summary.to_csv(output_dir / "matched_roi_residual_sign_change_summary.csv", index=False)
    primary_matching.to_csv(output_dir / "primary_high_complete_matching.csv", index=False)
    primary_full_qc.to_csv(output_dir / "primary_high_complete_full_qc.csv", index=False)
    sensitivity_complete.to_csv(output_dir / "sensitivity_balanced_complete.csv", index=False)
    one_gap_high.to_csv(output_dir / "sensitivity_high_one_internal_gap.csv", index=False)
    one_gap_balanced.to_csv(output_dir / "sensitivity_balanced_one_internal_gap.csv", index=False)
    review_flagged.to_csv(output_dir / "review_flagged_tracks.csv", index=False)
    filter_counts.to_csv(output_dir / "filter_step_counts_with_percentages.csv", index=False)

    warnings: list[str] = []
    if qc_config.mode == "all_required" and all(
        value is None
        for value in [
            qc_config.min_volume_um3,
            qc_config.max_volume_um3,
            qc_config.min_bbox_depth_planes,
            qc_config.max_bbox_depth_planes,
            qc_config.max_volume_ratio_from_track_median,
        ]
    ) and not qc_config.exclude_xy_edge and not qc_config.exclude_z_edge:
        warnings.append("segmentation_qc_not_configured")

    output_paths = {
        "session_manifest_resolved": str(output_dir / "session_manifest_resolved.csv"),
        "matched_roi_intensity_results_raw": str(output_dir / "matched_roi_intensity_results_raw.csv"),
        "matched_roi_intensity_results_dark_corrected": str(output_dir / "matched_roi_intensity_results_dark_corrected.csv"),
        "matched_roi_geometry_qc_long": str(output_dir / "matched_roi_geometry_qc_long.csv"),
        "matched_track_qc_summary": str(output_dir / "matched_track_qc_summary.csv"),
        "matched_roi_day_table_complete": str(output_dir / "matched_roi_day_table_complete.csv"),
        "matched_roi_log_ratio_metrics_complete": str(output_dir / "matched_roi_log_ratio_metrics_complete.csv"),
        "matched_daywise_green_red_linear_fit_summary": str(output_dir / "matched_daywise_green_red_linear_fit_summary.csv"),
        "matched_roi_metrics_with_green_red_fit_residuals": str(output_dir / "matched_roi_metrics_with_green_red_fit_residuals.csv"),
        "matched_roi_residual_sign_change_summary": str(output_dir / "matched_roi_residual_sign_change_summary.csv"),
        "primary_high_complete_matching": str(output_dir / "primary_high_complete_matching.csv"),
        "primary_high_complete_full_qc": str(output_dir / "primary_high_complete_full_qc.csv"),
        "sensitivity_balanced_complete": str(output_dir / "sensitivity_balanced_complete.csv"),
        "sensitivity_high_one_internal_gap": str(output_dir / "sensitivity_high_one_internal_gap.csv"),
        "sensitivity_balanced_one_internal_gap": str(output_dir / "sensitivity_balanced_one_internal_gap.csv"),
        "review_flagged_tracks": str(output_dir / "review_flagged_tracks.csv"),
        "filter_step_counts_with_percentages": str(output_dir / "filter_step_counts_with_percentages.csv"),
    }
    if qc_config.mode != "all_required" or any(
        value is not None
        for value in [
            qc_config.min_volume_um3,
            qc_config.max_volume_um3,
            qc_config.min_bbox_depth_planes,
            qc_config.max_bbox_depth_planes,
            qc_config.max_volume_ratio_from_track_median,
        ]
    ) or qc_config.exclude_xy_edge or qc_config.exclude_z_edge:
        output_paths["primary_high_complete_full_qc_status"] = "configured"
    else:
        output_paths["primary_high_complete_full_qc_status"] = "not_configured"

    total_duration_seconds = time.perf_counter() - pipeline_start_seconds
    run_finished_utc = datetime.now(timezone.utc).isoformat()
    run_log_payload = {
        "analysis_version": ANALYSIS_VERSION,
        "run_started_utc": run_started_utc,
        "run_finished_utc": run_finished_utc,
        "config": asdict(config),
        "segmentation_qc_config": asdict(qc_config),
        "manifest_path": str(config.manifest),
        "match_dir": str(match_dir),
        "manifest_sha256": _sha256_file(Path(config.manifest).resolve()),
        "match_run_log_sha256": _sha256_file(match_paths["run_log"]),
        "package_versions": _package_versions(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "warnings": warnings,
        "stage_durations_seconds": stage_durations_seconds,
        "total_duration_seconds": float(total_duration_seconds),
        "total_duration_hms": format_duration_seconds(total_duration_seconds),
        "row_counts": {
            "matched_roi_intensity_results_raw": int(len(raw_table)),
            "matched_roi_intensity_results_dark_corrected": int(len(corrected_table)),
            "matched_roi_geometry_qc_long": int(len(geometry_long)),
            "matched_track_qc_summary": int(len(matched_tracks)),
            "matched_roi_day_table_complete": int(len(roi_day_table)),
            "matched_roi_log_ratio_metrics_complete": int(len(metrics_table)),
            "matched_daywise_green_red_linear_fit_summary": int(len(fit_summary)),
            "matched_roi_metrics_with_green_red_fit_residuals": int(len(residual_table)),
            "matched_roi_residual_sign_change_summary": int(len(residual_summary)),
            "primary_high_complete_matching": int(len(primary_matching)),
            "primary_high_complete_full_qc": int(len(primary_full_qc)),
            "sensitivity_balanced_complete": int(len(sensitivity_complete)),
            "sensitivity_high_one_internal_gap": int(len(one_gap_high)),
            "sensitivity_balanced_one_internal_gap": int(len(one_gap_balanced)),
            "review_flagged_tracks": int(len(review_flagged)),
            "filter_step_counts_with_percentages": int(len(filter_counts)),
        },
        "output_paths": output_paths,
    }
    (output_dir / "run_log.json").write_text(json.dumps(run_log_payload, indent=2, sort_keys=True), encoding="utf-8")

    _write_summary_markdown(
        output_dir=output_dir,
        config=qc_config,
        filter_counts=filter_counts,
        fit_summary=fit_summary,
        manifest_records=manifest_records,
        primary_matching=primary_matching,
        primary_full_qc=primary_full_qc,
        sensitivity_complete=sensitivity_complete,
        one_gap_high=one_gap_high,
        one_gap_balanced=one_gap_balanced,
    )

    print(f"[{format_duration_seconds(total_duration_seconds)}] Pipeline completed", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset alias or explicit dataset directory path.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV used for matching and extraction.")
    parser.add_argument("--match-dir", required=True, help="Output directory from matching/run_daywise_roi_matching.py.")
    parser.add_argument("--policies", nargs="+", default=["high", "balanced"], help="Policies to analyze.")
    parser.add_argument("--green-dark", type=float, default=319.0, help="Green-channel dark offset.")
    parser.add_argument("--red-dark", type=float, default=534.0, help="Red-channel dark offset.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Positive offset used for log-ratio metrics.")
    parser.add_argument("--segmentation-qc-mode", default="all_required", choices=["all_required", "fraction"])
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
    config = DaywiseMatchedPipelineConfig(
        dataset=args.dataset,
        manifest=args.manifest,
        match_dir=args.match_dir,
        policies=tuple(args.policies),
        green_dark=args.green_dark,
        red_dark=args.red_dark,
        epsilon=args.epsilon,
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
    return run_daywise_matched_roi_pipeline(config)


if __name__ == "__main__":
    main()
