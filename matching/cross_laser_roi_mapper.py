"""Isolated same-session 920-to-1050 ROI identity mapping.

This module reuses affine_overlap_matcher instead of creating another matcher.
It maps moving 920 labels into the fixed 1050 identity space, but it does not
assign longitudinal track IDs or extract fluorescence values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
from pathlib import Path
import re

import numpy as np
import pandas as pd

from affine_overlap_matcher import (
    MATCHER_ALGORITHM_VERSION,
    AffineOverlapParams,
    PairMatchResult,
    RestrictedTransform,
    VoxelSpacing,
    extract_roi_features,
    match_pair,
)


CROSS_LASER_ALGORITHM_VERSION = "cross_laser_affine_overlap_v1"
CROSS_LASER_RELATIONSHIP_TYPE = "cross_laser_same_session"
# match_pair only places pair_gap in its returned summary. Passing None is
# numerically neutral and avoids applying a fictitious day gap.
PAIR_GAP_HANDLING = "pair_gap=None; provenance-only in affine_overlap_matcher"


@dataclass(frozen=True)
class CrossLaserSource:
    """Provenance for one native segmentation source."""

    name: str
    laser_nm: int
    channel: str
    mask_path: str | None = None


@dataclass
class CrossLaserSourceResult:
    """Full matcher evidence and complete coverage for one source relation."""

    fixed_source: CrossLaserSource
    source: CrossLaserSource
    fixed_features: pd.DataFrame
    moving_features: pd.DataFrame
    candidates: pd.DataFrame
    high_matches: pd.DataFrame
    balanced_matches: pd.DataFrame
    fixed_coverage: pd.DataFrame
    moving_coverage: pd.DataFrame
    summary: dict[str, object]
    transform: RestrictedTransform
    fixed_shape_zyx: tuple[int, int, int]
    moving_shape_zyx: tuple[int, int, int]


def _sha256_if_file(path: str | None) -> str | None:
    """Hash an existing source file without modifying it."""

    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_prefix(source: CrossLaserSource) -> str:
    """Return stable, compact coverage-column prefixes."""

    lowered = source.name.lower()
    if "green" in lowered:
        return "green"
    if "red" in lowered:
        return "red"
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def _validate_cross_laser_masks(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate native label stacks before matching.

    extract_roi_features performs the shared matcher validation for 3D,
    integer, nonnegative, nonempty masks with a zero background. Shape is an
    explicit cross-laser contract: v1 never crops, pads, or resamples labels.
    """

    fixed = np.asarray(fixed_mask)
    moving = np.asarray(moving_mask)
    if fixed.shape != moving.shape:
        raise ValueError(
            "Cross-laser fixed and moving masks must share the same ZYX shape; "
            f"got {fixed.shape} and {moving.shape}."
        )
    return fixed, moving


def inverse_restricted_transform(
    transform: RestrictedTransform,
    coordinates_zyx: np.ndarray,
) -> np.ndarray:
    """Map fixed-space points back into moving-space coordinates."""

    coordinates = np.asarray(coordinates_zyx, dtype=float)
    scalar_input = coordinates.ndim == 1
    if scalar_input:
        coordinates = coordinates[np.newaxis, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates_zyx must have shape (..., 3).")
    if not np.isfinite(float(transform.z_scale)) or float(transform.z_scale) == 0:
        raise ValueError("Restricted transform is not invertible because z_scale is zero.")

    xy_matrix = np.asarray(
        [
            [transform.y_from_y, transform.y_from_x],
            [transform.x_from_y, transform.x_from_x],
        ],
        dtype=float,
    )
    if not np.isfinite(xy_matrix).all() or np.isclose(np.linalg.det(xy_matrix), 0.0):
        raise ValueError("Restricted transform is not invertible because its XY block is singular.")

    z = (coordinates[:, 0] - float(transform.z_intercept)) / float(transform.z_scale)
    xy_fixed = coordinates[:, 1:] - np.asarray(
        [transform.y_intercept, transform.x_intercept], dtype=float
    )
    xy = np.linalg.solve(xy_matrix, xy_fixed.T).T
    inverted = np.column_stack([z, xy])
    return inverted[0] if scalar_input else inverted


def _feature_values(features: pd.DataFrame, labels: np.ndarray, columns: list[str]) -> np.ndarray:
    """Return feature values in label order while preserving original IDs."""

    return features.reindex(labels)[columns].to_numpy(dtype=float)


def _annotate_pair_table(
    table: pd.DataFrame,
    *,
    mouse_id: str,
    session_id: str,
    acquisition_date: str,
    fixed_source: CrossLaserSource,
    moving_source: CrossLaserSource,
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    transform: RestrictedTransform,
    spacing: VoxelSpacing,
) -> pd.DataFrame:
    """Add explicit source provenance and raw/aligned coordinate diagnostics."""

    output = table.copy()
    provenance = {
        "mouse_id": mouse_id,
        "session_id": session_id,
        "acquisition_date": acquisition_date,
        "relationship_type": CROSS_LASER_RELATIONSHIP_TYPE,
        "fixed_laser_nm": int(fixed_source.laser_nm),
        "moving_laser_nm": int(moving_source.laser_nm),
        "fixed_source": fixed_source.name,
        "moving_source": moving_source.name,
        "fixed_mask_path": fixed_source.mask_path or "",
        "fixed_mask_sha256": _sha256_if_file(fixed_source.mask_path) or "",
        "moving_mask_path": moving_source.mask_path or "",
        "moving_mask_sha256": _sha256_if_file(moving_source.mask_path) or "",
        "transform_direction": f"{moving_source.name} -> {fixed_source.name}",
        "transform_method": transform.method,
        "transform_fallback_reason": transform.fallback_reason or "",
    }
    for column, value in reversed(tuple(provenance.items())):
        output.insert(0, column, value)

    diagnostic_columns = [
        "fixed_label",
        "moving_label",
        "centroid_1050_z",
        "centroid_1050_y",
        "centroid_1050_x",
        "centroid_920_z",
        "centroid_920_y",
        "centroid_920_x",
        "centroid_920_aligned_z",
        "centroid_920_aligned_y",
        "centroid_920_aligned_x",
        "raw_delta_z_planes",
        "raw_delta_y_px",
        "raw_delta_x_px",
        "raw_delta_z_um",
        "raw_delta_y_um",
        "raw_delta_x_um",
        "aligned_residual_z_um",
        "aligned_residual_y_um",
        "aligned_residual_x_um",
        "aligned_residual_distance_um",
    ]
    if output.empty:
        for column in diagnostic_columns:
            output[column] = pd.Series(dtype=float)
        return output

    labels_fixed = output["label_a"].to_numpy(dtype=int)
    labels_moving = output["label_b"].to_numpy(dtype=int)
    fixed_coordinates = _feature_values(
        fixed_features, labels_fixed, ["centroid_z", "centroid_y", "centroid_x"]
    )
    moving_coordinates = _feature_values(
        moving_features, labels_moving, ["centroid_z", "centroid_y", "centroid_x"]
    )
    aligned_coordinates = transform.apply(moving_coordinates)
    raw_delta = fixed_coordinates - moving_coordinates
    aligned_delta = fixed_coordinates - aligned_coordinates
    spacing_zyx = spacing.as_zyx_array()
    raw_delta_um = raw_delta * spacing_zyx
    aligned_delta_um = aligned_delta * spacing_zyx

    output["fixed_label"] = labels_fixed
    output["moving_label"] = labels_moving
    if fixed_source.laser_nm == 1050:
        output["label_1050"] = labels_fixed
        output[f"label_920_{fixed_source.channel if moving_source.channel == fixed_source.channel else moving_source.channel}"] = labels_moving
        output["label_920"] = labels_moving
    elif fixed_source.laser_nm == moving_source.laser_nm == 920:
        output[f"label_920_{fixed_source.channel}"] = labels_fixed
        output[f"label_920_{moving_source.channel}"] = labels_moving
    for index, axis in enumerate(("z", "y", "x")):
        output[f"centroid_1050_{axis}"] = fixed_coordinates[:, index]
        output[f"centroid_920_{axis}"] = moving_coordinates[:, index]
        output[f"centroid_920_aligned_{axis}"] = aligned_coordinates[:, index]
        unit = "planes" if axis == "z" else "px"
        output[f"raw_delta_{axis}_{unit}"] = raw_delta[:, index]
        output[f"raw_delta_{axis}_um"] = raw_delta_um[:, index]
        output[f"aligned_residual_{axis}_um"] = aligned_delta_um[:, index]
    output["aligned_residual_distance_um"] = np.linalg.norm(aligned_delta_um, axis=1)
    return output


def _transform_bbox_corners_to_moving(
    feature: pd.Series,
    transform: RestrictedTransform,
) -> np.ndarray:
    """Map all fixed ROI bounding-box corners into moving coordinates."""

    starts = [
        float(feature["bbox_z0"]),
        float(feature["bbox_y0"]),
        float(feature["bbox_x0"]),
    ]
    # Bboxes are half-open. The closest occupied coordinate to their far edge
    # is one voxel below the exclusive stop.
    stops = [
        float(feature["bbox_z1"]) - 1.0,
        float(feature["bbox_y1"]) - 1.0,
        float(feature["bbox_x1"]) - 1.0,
    ]
    corners = np.asarray(list(product(*zip(starts, stops))), dtype=float)
    return inverse_restricted_transform(transform, corners)


def classify_common_volume(
    fixed_features: pd.DataFrame,
    transform: RestrictedTransform,
    moving_shape_zyx: tuple[int, int, int],
) -> pd.DataFrame:
    """Classify every fixed ROI's transformed observability in moving support."""

    shape = np.asarray(moving_shape_zyx, dtype=float)
    rows: list[dict[str, object]] = []
    for feature in fixed_features.itertuples(index=False):
        series = pd.Series(feature._asdict())
        corners = _transform_bbox_corners_to_moving(series, transform)
        inside = ((corners >= 0.0) & (corners <= (shape - 1.0))).all(axis=1)
        if bool(inside.all()):
            status = "inside_common_volume"
        elif bool(np.any(corners.max(axis=0) < 0.0) or np.any(corners.min(axis=0) > (shape - 1.0))):
            status = "outside_common_volume"
        else:
            status = "partially_inside_common_volume"
        rows.append(
            {
                "label_1050": int(series["label"]),
                "common_volume_status": status,
                "cross_laser_edge_clipped": status == "partially_inside_common_volume",
            }
        )
    return pd.DataFrame(rows)


def _best_evidence(table: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Select the deterministic best candidate for each fixed or moving label."""

    if table.empty:
        return pd.DataFrame(columns=[label_column])
    return (
        table.sort_values(
            ["score", "dice", "distance_um", "label_a", "label_b"],
            ascending=[False, False, True, True, True],
        )
        .drop_duplicates(label_column, keep="first")
        .reset_index(drop=True)
    )


def _evidence_status(
    *, high_label: object, balanced_label: object, best_label: object
) -> str:
    if pd.notna(high_label):
        return "high"
    if pd.notna(balanced_label):
        return "balanced_only"
    if pd.notna(best_label):
        return "candidate_but_rejected"
    return "no_candidate"


def build_fixed_coverage(result: CrossLaserSourceResult) -> pd.DataFrame:
    """Return one complete fixed-side row for every fixed label."""

    prefix = _source_prefix(result.source)
    fixed = result.fixed_features.reset_index(drop=True).rename(
        columns={
            "label": "label_1050",
            "area_voxels": "area_1050_voxels",
            "volume_um3": "volume_1050_um3",
            "centroid_z": "centroid_1050_z",
            "centroid_y": "centroid_1050_y",
            "centroid_x": "centroid_1050_x",
            "centroid_z_um": "centroid_1050_z_um",
            "centroid_y_um": "centroid_1050_y_um",
            "centroid_x_um": "centroid_1050_x_um",
            "touches_z_edge": "touches_1050_z_edge",
            "touches_xy_edge": "touches_1050_xy_edge",
        }
    )
    fixed["mouse_id"] = result.summary["mouse_id"]
    fixed["session_id"] = result.summary["session_id"]
    fixed["acquisition_date"] = result.summary["acquisition_date"]
    fixed = fixed.loc[:, ["mouse_id", "session_id", "acquisition_date"] + [column for column in fixed if column not in {"mouse_id", "session_id", "acquisition_date"}]]
    observability = classify_common_volume(
        result.fixed_features, result.transform, result.moving_shape_zyx
    )
    fixed = fixed.merge(observability, on="label_1050", how="left", validate="one_to_one")

    candidates = _best_evidence(result.candidates, "label_1050")
    high = _best_evidence(result.high_matches, "label_1050")
    balanced = _best_evidence(result.balanced_matches, "label_1050")
    candidate_fields = ["label_1050", "label_920", "score", "dice", "distance_um"]
    fixed = fixed.merge(
        candidates[candidate_fields].rename(
            columns={
                "label_920": f"{prefix}_best_candidate_label",
                "score": f"{prefix}_best_candidate_score",
                "dice": f"{prefix}_best_candidate_dice",
                "distance_um": f"{prefix}_best_candidate_distance_um",
            }
        ),
        on="label_1050",
        how="left",
    )
    fixed = fixed.merge(
        high[["label_1050", "label_920"]].rename(
            columns={"label_920": f"{prefix}_high_label_920"}
        ),
        on="label_1050",
        how="left",
    )
    fixed = fixed.merge(
        balanced[["label_1050", "label_920"]].rename(
            columns={"label_920": f"{prefix}_balanced_label_920"}
        ),
        on="label_1050",
        how="left",
    )
    fixed[f"{prefix}_status"] = [
        _evidence_status(
            high_label=row[f"{prefix}_high_label_920"],
            balanced_label=row[f"{prefix}_balanced_label_920"],
            best_label=row[f"{prefix}_best_candidate_label"],
        )
        for _, row in fixed.iterrows()
    ]
    return fixed.sort_values("label_1050").reset_index(drop=True)


def build_moving_coverage(result: CrossLaserSourceResult) -> pd.DataFrame:
    """Return one complete moving-side row for every native moving label."""

    moving = result.moving_features.reset_index(drop=True).rename(
        columns={
            "label": "label_920",
            "area_voxels": "area_920_voxels",
            "volume_um3": "volume_920_um3",
            "centroid_z": "centroid_920_z",
            "centroid_y": "centroid_920_y",
            "centroid_x": "centroid_920_x",
            "centroid_z_um": "centroid_920_z_um",
            "centroid_y_um": "centroid_920_y_um",
            "centroid_x_um": "centroid_920_x_um",
            "touches_z_edge": "touches_920_z_edge",
            "touches_xy_edge": "touches_920_xy_edge",
        }
    )
    moving["mouse_id"] = result.summary["mouse_id"]
    moving["session_id"] = result.summary["session_id"]
    moving["acquisition_date"] = result.summary["acquisition_date"]
    moving["source"] = result.source.name
    moving = moving.loc[:, ["mouse_id", "session_id", "acquisition_date", "source"] + [column for column in moving if column not in {"mouse_id", "session_id", "acquisition_date", "source"}]]
    candidates = _best_evidence(result.candidates, "label_920")
    high = _best_evidence(result.high_matches, "label_920")
    balanced = _best_evidence(result.balanced_matches, "label_920")
    moving = moving.merge(
        candidates[["label_920", "label_1050", "score", "dice", "distance_um"]].rename(
            columns={
                "label_1050": "best_candidate_label_1050",
                "score": "best_candidate_score",
                "dice": "best_candidate_dice",
                "distance_um": "best_candidate_distance_um",
            }
        ),
        on="label_920",
        how="left",
    )
    moving = moving.merge(
        high[["label_920", "label_1050"]].rename(
            columns={"label_1050": "high_label_1050"}
        ),
        on="label_920",
        how="left",
    )
    moving = moving.merge(
        balanced[["label_920", "label_1050"]].rename(
            columns={"label_1050": "balanced_label_1050"}
        ),
        on="label_920",
        how="left",
    )
    moving["mapping_status"] = [
        _evidence_status(
            high_label=row["high_label_1050"],
            balanced_label=row["balanced_label_1050"],
            best_label=row["best_candidate_label_1050"],
        )
        for _, row in moving.iterrows()
    ]
    return moving.sort_values("label_920").reset_index(drop=True)


def map_cross_laser_source(
    *,
    mouse_id: str,
    session_id: str,
    acquisition_date: str,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    fixed_source: CrossLaserSource,
    moving_source: CrossLaserSource,
    spacing: VoxelSpacing,
    affine_params: AffineOverlapParams | None = None,
) -> CrossLaserSourceResult:
    """Map one moving segmentation into a fixed canonical label space."""

    fixed, moving = _validate_cross_laser_masks(fixed_mask, moving_mask)
    fixed_features = extract_roi_features(fixed, session_id=session_id, spacing=spacing)
    moving_features = extract_roi_features(moving, session_id=session_id, spacing=spacing)
    pair_result: PairMatchResult = match_pair(
        session_a=session_id,
        session_b=session_id,
        mask_a=fixed,
        mask_b=moving,
        params=affine_params,
        spacing=spacing,
        features_a=fixed_features,
        features_b=moving_features,
        pair_gap=None,
    )
    annotate_args = {
        "mouse_id": mouse_id,
        "session_id": session_id,
        "acquisition_date": acquisition_date,
        "fixed_source": fixed_source,
        "moving_source": moving_source,
        "fixed_features": fixed_features,
        "moving_features": moving_features,
        "transform": pair_result.transform,
        "spacing": spacing,
    }
    candidates = _annotate_pair_table(pair_result.candidates, **annotate_args)
    high_matches = _annotate_pair_table(pair_result.high_matches, **annotate_args)
    balanced_matches = _annotate_pair_table(pair_result.balanced_matches, **annotate_args)
    summary = dict(pair_result.summary)
    summary.update(
        {
            "cross_laser_algorithm_version": CROSS_LASER_ALGORITHM_VERSION,
            "affine_matcher_algorithm_version": MATCHER_ALGORITHM_VERSION,
            "relationship_type": CROSS_LASER_RELATIONSHIP_TYPE,
            "pair_gap_handling": PAIR_GAP_HANDLING,
            "mouse_id": mouse_id,
            "session_id": session_id,
            "acquisition_date": acquisition_date,
            "fixed_source": fixed_source.name,
            "moving_source": moving_source.name,
            "fixed_laser_nm": int(fixed_source.laser_nm),
            "moving_laser_nm": int(moving_source.laser_nm),
        }
    )
    result = CrossLaserSourceResult(
        fixed_source=fixed_source,
        source=moving_source,
        fixed_features=fixed_features,
        moving_features=moving_features,
        candidates=candidates,
        high_matches=high_matches,
        balanced_matches=balanced_matches,
        fixed_coverage=pd.DataFrame(),
        moving_coverage=pd.DataFrame(),
        summary=summary,
        transform=pair_result.transform,
        fixed_shape_zyx=tuple(int(value) for value in fixed.shape),
        moving_shape_zyx=tuple(int(value) for value in moving.shape),
    )
    # Only 1050-fixed relations need 1050 coverage tables. The 920-red to
    # 920-green consistency relation is consumed as pair evidence only.
    if fixed_source.laser_nm != 1050:
        return result
    result.fixed_coverage = build_fixed_coverage(result)
    result.moving_coverage = build_moving_coverage(result)
    observable = result.fixed_coverage["common_volume_status"].ne("outside_common_volume")
    result.summary.update(
        {
            "n_fixed_observable": int(observable.sum()),
            "n_high_all_fixed": int(len(high_matches)),
            "n_high_observable_fixed": int(
                result.fixed_coverage.loc[
                    observable, f"{_source_prefix(moving_source)}_high_label_920"
                ].notna().sum()
            ),
        }
    )
    return result


def resolve_identity_evidence(
    primary_fixed_coverage: pd.DataFrame,
    *,
    secondary_fixed_coverage: pd.DataFrame | None = None,
    green_red_high_matches: pd.DataFrame | None = None,
    secondary_status: str = "not_evaluated",
) -> pd.DataFrame:
    """Resolve primary/secondary same-session evidence at each fixed label."""

    required = {"label_1050", "green_status", "green_high_label_920"}
    missing = required.difference(primary_fixed_coverage.columns)
    if missing:
        raise ValueError(f"primary_fixed_coverage is missing required columns: {sorted(missing)}")
    resolution = primary_fixed_coverage.copy()
    resolution["primary_green_status"] = resolution["green_status"]
    resolution["primary_green_label_920"] = resolution["green_high_label_920"]
    resolution["primary_green_confidence"] = resolution["primary_green_status"].where(
        resolution["primary_green_status"].eq("high"), ""
    )
    resolution["secondary_red_status"] = secondary_status
    resolution["secondary_red_label_920"] = np.nan
    resolution["secondary_red_confidence"] = ""
    if secondary_fixed_coverage is not None:
        secondary = secondary_fixed_coverage[
            ["label_1050", "red_status", "red_high_label_920"]
        ].rename(
            columns={
                "red_status": "secondary_red_status",
                "red_high_label_920": "secondary_red_label_920",
            }
        )
        resolution = resolution.drop(
            columns=["secondary_red_status", "secondary_red_label_920"]
        ).merge(secondary, on="label_1050", how="left", validate="one_to_one")
        resolution["secondary_red_status"] = resolution["secondary_red_status"].fillna(
            "no_candidate"
        )
        resolution["secondary_red_confidence"] = resolution[
            "secondary_red_status"
        ].where(resolution["secondary_red_status"].eq("high"), "")

    green_to_fixed = {
        int(row.primary_green_label_920): int(row.label_1050)
        for row in resolution.itertuples(index=False)
        if row.primary_green_status == "high" and pd.notna(row.primary_green_label_920)
    }
    red_to_green: dict[int, int] = {}
    if green_red_high_matches is not None and not green_red_high_matches.empty:
        for row in green_red_high_matches.itertuples(index=False):
            red_label = row.label_920_red if hasattr(row, "label_920_red") else row.label_920
            green_label = row.label_920_green if hasattr(row, "label_920_green") else row.label_1050
            red_to_green[int(red_label)] = int(green_label)

    conflict_values: list[bool] = []
    statuses: list[str] = []
    sources: list[str] = []
    labels: list[float] = []
    recommended: list[bool] = []
    provisional: list[bool] = []
    review: list[bool] = []
    for row in resolution.itertuples(index=False):
        fixed_label = int(row.label_1050)
        if getattr(row, "common_volume_status", "") == "outside_common_volume":
            conflict_values.append(False)
            statuses.append("outside_common_volume")
            sources.append("")
            labels.append(np.nan)
            recommended.append(False)
            provisional.append(False)
            review.append(False)
            continue
        primary_high = (
            row.primary_green_status == "high"
            and pd.notna(row.primary_green_label_920)
        )
        secondary_high = (
            row.secondary_red_status == "high"
            and pd.notna(row.secondary_red_label_920)
        )
        conflict = False
        if secondary_high:
            green_label = red_to_green.get(int(row.secondary_red_label_920))
            conflict = (
                green_label is not None
                and green_to_fixed.get(green_label) not in {None, fixed_label}
            )
        conflict_values.append(conflict)
        if primary_high:
            statuses.append("primary_high")
            sources.append("920_green_primary")
            labels.append(float(row.primary_green_label_920))
            recommended.append(True)
            provisional.append(False)
            review.append(False)
        elif conflict:
            statuses.append("cross_source_conflict")
            sources.append("")
            labels.append(np.nan)
            recommended.append(False)
            provisional.append(False)
            review.append(True)
        elif secondary_high:
            statuses.append("secondary_high_rescue_candidate")
            sources.append("920_red_secondary")
            labels.append(float(row.secondary_red_label_920))
            recommended.append(False)
            provisional.append(True)
            review.append(True)
        elif row.primary_green_status == "balanced_only":
            statuses.append("primary_balanced_only")
            sources.append("")
            labels.append(np.nan)
            recommended.append(False)
            provisional.append(False)
            review.append(True)
        elif row.primary_green_status == "candidate_but_rejected":
            statuses.append("candidate_but_rejected")
            sources.append("")
            labels.append(np.nan)
            recommended.append(False)
            provisional.append(False)
            review.append(True)
        else:
            statuses.append("no_candidate")
            sources.append("")
            labels.append(np.nan)
            recommended.append(False)
            provisional.append(False)
            review.append(False)
    resolution["cross_source_conflict"] = conflict_values
    resolution["resolved_status"] = statuses
    resolution["resolved_920_source"] = sources
    resolution["resolved_label_920"] = labels
    resolution["recommended_for_identity"] = recommended
    resolution["provisional_identity"] = provisional
    resolution["review_required"] = review
    return resolution.sort_values("label_1050").reset_index(drop=True)


def accepted_pairs_by_source(result: CrossLaserSourceResult) -> pd.DataFrame:
    """Return high and balanced accepted assignments without collapsing policy."""

    accepted = pd.concat(
        [result.high_matches, result.balanced_matches], ignore_index=True, sort=False
    )
    if accepted.empty:
        return accepted
    accepted["confidence_tier"] = np.where(
        accepted["assignment_policy"].astype(str).eq("high"),
        "high",
        "balanced",
    )
    return accepted.sort_values(
        ["session_id", "moving_source", "assignment_policy", "label_1050", "label_920"]
    ).reset_index(drop=True)


def transform_record(result: CrossLaserSourceResult, spacing: VoxelSpacing) -> dict[str, object]:
    """Serialize one explicit transform relation and its coarse shift."""

    transform = result.transform
    summary = result.summary
    return {
        "mouse_id": summary["mouse_id"],
        "session_id": summary["session_id"],
        "acquisition_date": summary["acquisition_date"],
        "relationship_type": CROSS_LASER_RELATIONSHIP_TYPE,
        "fixed_source": result.fixed_source.name,
        "moving_source": result.source.name,
        "transform_direction": f"{result.source.name} -> {result.fixed_source.name}",
        "z_intercept": transform.z_intercept,
        "z_scale": transform.z_scale,
        "y_intercept": transform.y_intercept,
        "y_from_y": transform.y_from_y,
        "y_from_x": transform.y_from_x,
        "x_intercept": transform.x_intercept,
        "x_from_y": transform.x_from_y,
        "x_from_x": transform.x_from_x,
        "method": transform.method,
        "fallback_reason": transform.fallback_reason,
        "n_seed": transform.n_seed,
        "n_inlier": transform.n_inlier,
        "residual_median_um": transform.residual_median_um,
        "residual_p95_um": transform.residual_p95_um,
        "shift_z_planes": summary["shift_z"],
        "shift_y_px": summary["shift_y"],
        "shift_x_px": summary["shift_x"],
        "shift_z_um": float(summary["shift_z"]) * spacing.z_um,
        "shift_y_um": float(summary["shift_y"]) * spacing.y_um,
        "shift_x_um": float(summary["shift_x"]) * spacing.x_um,
    }


def relabel_primary_high_mask(
    moving_mask: np.ndarray,
    high_matches: pd.DataFrame,
) -> np.ndarray:
    """Return a native 920 mask carrying only primary-high fixed label values."""

    source = np.asarray(moving_mask)
    if source.ndim != 3 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("moving_mask must be a 3D integer label mask.")
    mapping = {
        int(row.label_920): int(row.label_1050)
        for row in high_matches.itertuples(index=False)
    }
    maximum = max(mapping.values(), default=0)
    dtype = np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32
    relabelled = np.zeros(source.shape, dtype=dtype)
    for moving_label, fixed_label in mapping.items():
        relabelled[source == moving_label] = fixed_label
    return relabelled
