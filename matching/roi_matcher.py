"""Match 3D ROI label masks across weekly imaging sessions.

This module implements a precision-first ROI matching workflow for labeled
``(z, y, x)`` mask stacks that have already been co-registered into roughly the
same XY space. The intended use case is multi-week ROI tracking where small
residual XY shifts remain after image registration.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import tifffile

DEFAULT_XY_UM_PER_PX = 710.0 / 1024.0
DEFAULT_Z_UM_PER_PLANE = 5.0
DEFAULT_TRACK_LENGTH_THRESHOLDS = (4, 5, 6, 7)


def format_duration_seconds(duration_seconds: float) -> str:
    """Format elapsed wall-clock time as ``HH:MM:SS``."""

    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _serialize_output_paths(output_paths: dict[str, object]) -> dict[str, object]:
    """Convert output paths into JSON-serializable strings."""

    serialized: dict[str, object] = {}
    for key, value in output_paths.items():
        if isinstance(value, (str, Path)):
            serialized[key] = str(value)
        elif isinstance(value, list):
            serialized[key] = [str(item) for item in value]
        else:
            serialized[key] = value
    return serialized


@dataclass(frozen=True)
class MatchParams:
    """Store matcher thresholds and scoring parameters.

    Parameters
    ----------
    patch_radius : int
        Radius in pixels for square center-plane ROI patches. The extracted patch
        shape is ``(2 * patch_radius + 1, 2 * patch_radius + 1)``.
    patch_size : int
        Spatial helper scale in pixels. The current implementation uses it only
        when choosing a coarse spatial-hash cell size.
    overlap : int
        Reserved overlap parameter in pixels kept for interface stability.
    max_dist_xy : float
        Maximum residual XY centroid distance in pixels after applying the
        estimated pairwise translation.
    max_dist_z : float
        Maximum centroid distance along the z axis in planes.
    min_score : float
        Minimum pairwise match score on the unit interval ``[0, 1]``.
    min_gap_support : float
        Minimum confidence required for a one-week gap bridge.
    edge_margin : int
        Distance-to-edge threshold in pixels for classifying a ROI as truncated.
    edge_relax_xy : float
        Additional allowed XY displacement in pixels for edge ROIs.
    edge_score_bonus : float
        Small additive score bonus for edge ROIs.
    translation_max_shift : int
        Maximum absolute XY shift in pixels searched by phase correlation.
    use_translation : bool
        Whether to estimate and apply a global pairwise XY translation.
    low_confidence_threshold : float
        Threshold below which a track is flagged for review.
    """

    patch_radius: int = 12
    patch_size: int = 64
    overlap: int = 16
    max_dist_xy: float = 15.0
    max_dist_z: float = 5.0
    min_score: float = 0.45
    min_gap_support: float = 0.80
    edge_margin: int = 20
    edge_relax_xy: float = 3.0
    edge_score_bonus: float = 0.02
    translation_max_shift: int = 32
    use_translation: bool = True
    low_confidence_threshold: float = 0.55


@dataclass(frozen=True)
class ROIRecord:
    """Describe one ROI extracted from a labeled 3D mask stack.

    Parameters
    ----------
    day : str
        Session identifier associated with the ROI.
    label : int
        Positive ROI label value from the mask stack.
    centroid_z : float
        ROI centroid along the z axis in planes.
    centroid_y : float
        ROI centroid along the y axis in pixels.
    centroid_x : float
        ROI centroid along the x axis in pixels.
    center_plane : int
        Z plane index with the largest within-plane ROI area.
    area_center : int
        Number of ROI pixels in ``center_plane``.
    area_total : int
        Total number of ROI voxels across the full ``(z, y, x)`` stack.
    patch : numpy.ndarray
        Float32 center-plane binary patch with shape
        ``(2 * patch_radius + 1, 2 * patch_radius + 1)``.
    radial_profile : numpy.ndarray
        One-dimensional normalized radial occupancy profile of ``patch``.
    center_coords : numpy.ndarray
        ``(n_pixels, 2)`` array of relative center-plane ``(y, x)`` coordinates
        in pixels.
    dist_to_edge : float
        Minimum centroid distance to any XY image boundary in pixels.
    is_edge : bool
        Whether ``dist_to_edge < edge_margin``.
    bbox_zyx : tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
        Inclusive ROI bounding box ``((z_min, z_max), (y_min, y_max),
        (x_min, x_max))`` in voxel indices.
    """

    day: str
    label: int
    centroid_z: float
    centroid_y: float
    centroid_x: float
    center_plane: int
    area_center: int
    area_total: int
    patch: np.ndarray
    radial_profile: np.ndarray
    center_coords: np.ndarray
    dist_to_edge: float
    is_edge: bool
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


# Helper functions


def _extract_patch(binary_plane: np.ndarray, cy: float, cx: float, radius: int) -> np.ndarray:
    """Extract a zero-padded center-plane patch around one ROI centroid.

    Parameters
    ----------
    binary_plane : numpy.ndarray
        Boolean or integer image with shape ``(y, x)`` containing one ROI mask.
    cy : float
        ROI centroid y coordinate in pixels.
    cx : float
        ROI centroid x coordinate in pixels.
    radius : int
        Patch radius in pixels.

    Returns
    -------
    numpy.ndarray
        Float32 patch with shape ``(2 * radius + 1, 2 * radius + 1)``.
    """

    cy_i = int(round(cy))
    cx_i = int(round(cx))
    y0 = cy_i - radius
    y1 = cy_i + radius + 1
    x0 = cx_i - radius
    x1 = cx_i + radius + 1
    height, width = binary_plane.shape
    patch = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    yy0 = max(y0, 0)
    yy1 = min(y1, height)
    xx0 = max(x0, 0)
    xx1 = min(x1, width)
    py0 = yy0 - y0
    px0 = xx0 - x0
    patch[py0:py0 + (yy1 - yy0), px0:px0 + (xx1 - xx0)] = binary_plane[yy0:yy1, xx0:xx1].astype(np.float32)
    return patch


def _radial_profile(binary_patch: np.ndarray, bins: int = 8) -> np.ndarray:
    """Measure radial occupancy of a binary patch around its center.

    Parameters
    ----------
    binary_patch : numpy.ndarray
        Float or boolean image with shape ``(y, x)``.
    bins : int
        Number of radial bins.

    Returns
    -------
    numpy.ndarray
        Float32 vector with shape ``(bins,)`` normalized to sum to 1 when the
        patch contains any foreground pixels.
    """

    cy = (binary_patch.shape[0] - 1) / 2.0
    cx = (binary_patch.shape[1] - 1) / 2.0
    yy, xx = np.indices(binary_patch.shape)
    radius_image = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0.0, float(radius_image.max()) + 1e-6, bins + 1)
    profile = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        in_bin = (radius_image >= edges[index]) & (radius_image < edges[index + 1])
        if np.any(in_bin):
            profile[index] = float(binary_patch[in_bin].mean())
    total = float(profile.sum())
    if total > 0.0:
        profile /= total
    return profile


def _coords_iou(coords_a: np.ndarray, coords_b: np.ndarray, radius: int = 20) -> float:
    """Compute IoU between two relative center-plane ROI coordinate sets.

    Parameters
    ----------
    coords_a : numpy.ndarray
        Float array with shape ``(n_a, 2)`` containing relative ``(y, x)``
        center-plane coordinates in pixels.
    coords_b : numpy.ndarray
        Float array with shape ``(n_b, 2)`` containing relative ``(y, x)``
        center-plane coordinates in pixels.
    radius : int
        Half-width of the rasterized comparison canvas in pixels.

    Returns
    -------
    float
        Intersection-over-union on the unit interval ``[0, 1]``.
    """

    size = 2 * radius + 1
    canvas_a = np.zeros((size, size), dtype=bool)
    canvas_b = np.zeros((size, size), dtype=bool)
    for coords, canvas in ((coords_a, canvas_a), (coords_b, canvas_b)):
        ys = np.round(coords[:, 0] + radius).astype(int)
        xs = np.round(coords[:, 1] + radius).astype(int)
        valid = (ys >= 0) & (ys < size) & (xs >= 0) & (xs < size)
        canvas[ys[valid], xs[valid]] = True
    intersection = int(np.logical_and(canvas_a, canvas_b).sum())
    union = int(np.logical_or(canvas_a, canvas_b).sum())
    if union == 0:
        return 0.0
    return float(intersection / union)


def _phase_correlation_shift(reference_image: np.ndarray, moving_image: np.ndarray, max_shift: int) -> tuple[float, float]:
    """Estimate the XY shift that aligns one 2D image to another.

    Parameters
    ----------
    reference_image : numpy.ndarray
        Float-like image with shape ``(y, x)``.
    moving_image : numpy.ndarray
        Float-like image with shape ``(y, x)`` that should be shifted into the
        reference frame.
    max_shift : int
        Maximum absolute shift in pixels searched around the correlation peak.

    Returns
    -------
    tuple[float, float]
        Estimated translation ``(shift_y, shift_x)`` in pixels that should be
        added to coordinates from ``moving_image`` to align them with
        ``reference_image``.
    """

    reference = reference_image.astype(np.float32, copy=False)
    moving = moving_image.astype(np.float32, copy=False)
    reference = reference - float(reference.mean())
    moving = moving - float(moving.mean())
    correlation = fftconvolve(reference, moving[::-1, ::-1], mode="same")
    center_y, center_x = np.array(correlation.shape) // 2
    y0 = max(0, center_y - max_shift)
    y1 = min(correlation.shape[0], center_y + max_shift + 1)
    x0 = max(0, center_x - max_shift)
    x1 = min(correlation.shape[1], center_x + max_shift + 1)
    sub_image = correlation[y0:y1, x0:x1]
    peak_y_local, peak_x_local = np.unravel_index(np.argmax(sub_image), sub_image.shape)
    peak_y = y0 + int(peak_y_local)
    peak_x = x0 + int(peak_x_local)
    return float(peak_y - center_y), float(peak_x - center_x)


def _build_spatial_hash(records: list[ROIRecord], cell_size: float) -> dict[tuple[int, int], list[int]]:
    """Index ROI centroids on a coarse XY grid for fast neighborhood queries.

    Parameters
    ----------
    records : list[ROIRecord]
        ROI records whose centroid coordinates are expressed in pixels.
    cell_size : float
        Spatial hash cell width in pixels.

    Returns
    -------
    dict[tuple[int, int], list[int]]
        Mapping from integer cell coordinates to record indices.
    """

    cells: dict[tuple[int, int], list[int]] = {}
    for record_index, record in enumerate(records):
        cell_key = (int(record.centroid_y // cell_size), int(record.centroid_x // cell_size))
        cells.setdefault(cell_key, []).append(record_index)
    return cells


def _query_spatial_hash(
    cells: dict[tuple[int, int], list[int]],
    y_coord: float,
    x_coord: float,
    radius: float,
    cell_size: float,
) -> list[int]:
    """Return candidate ROI indices near one XY query location.

    Parameters
    ----------
    cells : dict[tuple[int, int], list[int]]
        Spatial hash returned by :func:`_build_spatial_hash`.
    y_coord : float
        Query y coordinate in pixels.
    x_coord : float
        Query x coordinate in pixels.
    radius : float
        Search radius in pixels.
    cell_size : float
        Spatial hash cell width in pixels.

    Returns
    -------
    list[int]
        Candidate record indices that fall inside nearby hash cells.
    """

    center_y = int(y_coord // cell_size)
    center_x = int(x_coord // cell_size)
    span = int(math.ceil(radius / cell_size))
    indices: list[int] = []
    for grid_y in range(center_y - span, center_y + span + 1):
        for grid_x in range(center_x - span, center_x + span + 1):
            indices.extend(cells.get((grid_y, grid_x), []))
    return indices


def _compute_bbox_overlap_fraction(record_a: ROIRecord, record_b: ROIRecord) -> float:
    """Measure XY overlap between two ROI bounding boxes.

    Parameters
    ----------
    record_a : ROIRecord
        First ROI with inclusive bounding-box coordinates in pixels.
    record_b : ROIRecord
        Second ROI with inclusive bounding-box coordinates in pixels.

    Returns
    -------
    float
        Bounding-box IoU in XY after ignoring z extent.
    """

    ay0, ay1 = record_a.bbox_zyx[1]
    ax0, ax1 = record_a.bbox_zyx[2]
    by0, by1 = record_b.bbox_zyx[1]
    bx0, bx1 = record_b.bbox_zyx[2]
    intersection_h = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    intersection_w = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    intersection = intersection_h * intersection_w
    area_a = (ay1 - ay0 + 1) * (ax1 - ax0 + 1)
    area_b = (by1 - by0 + 1) * (bx1 - bx0 + 1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _edge_distance(mask_shape_yx: tuple[int, int], cy: float, cx: float) -> float:
    """Compute the centroid distance to the nearest XY image boundary.

    Parameters
    ----------
    mask_shape_yx : tuple[int, int]
        Image shape ``(height, width)`` in pixels.
    cy : float
        Centroid y coordinate in pixels.
    cx : float
        Centroid x coordinate in pixels.

    Returns
    -------
    float
        Minimum centroid-to-edge distance in pixels.
    """

    height, width = mask_shape_yx
    return float(min(cy, cx, (height - 1) - cy, (width - 1) - cx))


def _evaluate_shift_support(
    records_a: list[ROIRecord],
    records_b: list[ROIRecord],
    shift_yx: tuple[float, float],
    max_dist_xy: float,
    max_dist_z: float,
) -> tuple[int, float]:
    """Measure how well a candidate global shift is supported by ROI centroids.

    Parameters
    ----------
    records_a : list[ROIRecord]
        Reference-session ROIs.
    records_b : list[ROIRecord]
        Moving-session ROIs whose centroids are shifted by ``shift_yx``.
    shift_yx : tuple[float, float]
        Translation ``(shift_y, shift_x)`` in pixels.
    max_dist_xy : float
        Maximum allowed XY residual in pixels.
    max_dist_z : float
        Maximum allowed z residual in planes.

    Returns
    -------
    tuple[int, float]
        Number of supported nearest-neighbor centroid pairs and the median XY
        residual distance in pixels among supported pairs.
    """

    if not records_a or not records_b:
        return 0, math.inf

    shift_y, shift_x = shift_yx
    residuals: list[float] = []
    for record_a in records_a:
        best_residual = math.inf
        for record_b in records_b:
            dz = abs(record_a.centroid_z - record_b.centroid_z)
            if dz > max_dist_z:
                continue
            residual_y = record_a.centroid_y - (record_b.centroid_y + shift_y)
            residual_x = record_a.centroid_x - (record_b.centroid_x + shift_x)
            dxy = math.hypot(residual_y, residual_x)
            if dxy <= max_dist_xy and dxy < best_residual:
                best_residual = dxy
        if math.isfinite(best_residual):
            residuals.append(float(best_residual))
    if not residuals:
        return 0, math.inf
    return len(residuals), float(np.median(residuals))


def _select_pair_shift(
    records_a: list[ROIRecord],
    records_b: list[ROIRecord],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    params: MatchParams,
) -> tuple[float, float]:
    """Choose a conservative pairwise XY translation estimate.

    Parameters
    ----------
    records_a : list[ROIRecord]
        Reference-session ROIs.
    records_b : list[ROIRecord]
        Moving-session ROIs.
    mask_a : numpy.ndarray
        Reference mask stack with shape ``(z, y, x)``.
    mask_b : numpy.ndarray
        Moving mask stack with shape ``(z, y, x)``.
    params : MatchParams
        Matcher thresholds and translation settings.

    Returns
    -------
    tuple[float, float]
        Translation ``(shift_y, shift_x)`` in pixels to apply to ``records_b``.
    """

    if not params.use_translation:
        return 0.0, 0.0
    phase_shift = estimate_pair_shift(mask_a, mask_b, max_shift=params.translation_max_shift)
    candidate_shifts = [(0.0, 0.0), phase_shift]
    best_shift = (0.0, 0.0)
    best_score = -math.inf
    for shift_yx in candidate_shifts:
        support_count, median_residual = _evaluate_shift_support(
            records_a=records_a,
            records_b=records_b,
            shift_yx=shift_yx,
            max_dist_xy=params.max_dist_xy,
            max_dist_z=params.max_dist_z,
        )
        residual_term = float(median_residual if math.isfinite(median_residual) else 1e6)
        shift_norm = math.hypot(shift_yx[0], shift_yx[1])
        support_score = float(support_count) - 0.05 * residual_term - 0.02 * shift_norm
        if support_score > best_score:
            best_score = support_score
            best_shift = shift_yx
    return best_shift


def _track_node_map(tracks: list[dict[str, object]], day_names: list[str]) -> dict[tuple[str, int], int]:
    """Map each occupied day/label node to its current track index.

    Parameters
    ----------
    tracks : list[dict[str, object]]
        Track records using ``"{day}_roi"`` keys and scalar metadata values.
    day_names : list[str]
        Ordered session names.

    Returns
    -------
    dict[tuple[str, int], int]
        Mapping from ``(day_name, label)`` to track index.
    """

    node_to_track: dict[tuple[str, int], int] = {}
    for track_index, track in enumerate(tracks):
        for day_name in day_names:
            value = track.get(f"{day_name}_roi", pd.NA)
            if pd.notna(value):
                node_to_track[(day_name, int(value))] = track_index
    return node_to_track


def _occupied_day_indices(track: dict[str, object], day_names: list[str]) -> list[int]:
    """List occupied day indices for one track.

    Parameters
    ----------
    track : dict[str, object]
        Track record containing ``"{day}_roi"`` keys.
    day_names : list[str]
        Ordered session names.

    Returns
    -------
    list[int]
        Sorted day indices whose ROI entries are present.
    """

    occupied = []
    for day_index, day_name in enumerate(day_names):
        if pd.notna(track.get(f"{day_name}_roi", pd.NA)):
            occupied.append(day_index)
    return occupied


def _summarize_track(track: dict[str, object], day_names: list[str]) -> None:
    """Refresh count and confidence summary fields for one mutable track.

    Parameters
    ----------
    track : dict[str, object]
        Mutable track record containing ROI assignments and confidence lists.
    day_names : list[str]
        Ordered session names.

    Returns
    -------
    None
        The input track dictionary is updated in place.
    """

    occupied_indices = _occupied_day_indices(track, day_names)
    track["n_days_present"] = int(len(occupied_indices))
    if occupied_indices:
        first_index = occupied_indices[0]
        last_index = occupied_indices[-1]
        missing_internal = 0
        for day_index in range(first_index, last_index + 1):
            if pd.isna(track.get(f"{day_names[day_index]}_roi", pd.NA)):
                missing_internal += 1
        track["missing_intermediate_days"] = int(missing_internal)
    else:
        track["missing_intermediate_days"] = 0
    confidences = list(track.get("_confidences", []))
    if confidences:
        track["mean_confidence"] = float(np.mean(confidences))
        track["min_confidence"] = float(np.min(confidences))
    else:
        track["mean_confidence"] = 0.0
        track["min_confidence"] = 0.0


def _finalize_tracks(tracks: list[dict[str, object]], day_names: list[str]) -> pd.DataFrame:
    """Convert mutable track dictionaries into a sorted output table.

    Parameters
    ----------
    tracks : list[dict[str, object]]
        Mutable track records produced during track assembly.
    day_names : list[str]
        Ordered session names.

    Returns
    -------
    pandas.DataFrame
        Table with one row per track and one ROI column per day.
    """

    rows: list[dict[str, object]] = []
    for track in tracks:
        row = {f"{day_name}_roi": track.get(f"{day_name}_roi", pd.NA) for day_name in day_names}
        row["n_days_present"] = int(track["n_days_present"])
        row["mean_confidence"] = float(track["mean_confidence"])
        row["min_confidence"] = float(track["min_confidence"])
        row["used_gap_bridge"] = bool(track.get("used_gap_bridge", False))
        row["missing_intermediate_days"] = int(track["missing_intermediate_days"])
        rows.append(row)
    if rows:
        table = pd.DataFrame(rows)
    else:
        table = pd.DataFrame(
            columns=[
                *[f"{day_name}_roi" for day_name in day_names],
                "n_days_present",
                "mean_confidence",
                "min_confidence",
                "used_gap_bridge",
                "missing_intermediate_days",
            ]
        )
    table = table.sort_values(["n_days_present", "mean_confidence"], ascending=[False, False]).reset_index(drop=True)
    table.insert(0, "cluster_id", np.arange(1, len(table) + 1))
    return table


# User-facing computational functions


def extract_roi_records(
    mask_zyx: np.ndarray,
    day_name: str,
    patch_radius: int = 12,
    edge_margin: int = 20,
) -> list[ROIRecord]:
    """Extract per-ROI geometry and shape features from one labeled 3D mask.

    Parameters
    ----------
    mask_zyx : numpy.ndarray
        Integer label stack with shape ``(z, y, x)``. Zero denotes background
        and positive integers denote ROI labels.
    day_name : str
        Session identifier attached to every extracted ROI.
    patch_radius : int, default=12
        Radius in pixels used for center-plane patch extraction.
    edge_margin : int, default=20
        Edge classification threshold in pixels.

    Returns
    -------
    list[ROIRecord]
        ROI records sorted by ascending label value.
    """

    labels = np.unique(mask_zyx)
    labels = labels[labels > 0]
    height, width = mask_zyx.shape[-2:]
    records: list[ROIRecord] = []
    for label in labels:
        z_coords, y_coords, x_coords = np.where(mask_zyx == label)
        if len(z_coords) == 0:
            continue
        centroid_z = float(z_coords.mean())
        centroid_y = float(y_coords.mean())
        centroid_x = float(x_coords.mean())
        z_min = int(z_coords.min())
        z_max = int(z_coords.max())
        y_min = int(y_coords.min())
        y_max = int(y_coords.max())
        x_min = int(x_coords.min())
        x_max = int(x_coords.max())
        plane_counts = [int(np.count_nonzero(mask_zyx[z_index] == label)) for z_index in range(z_min, z_max + 1)]
        center_plane = int(z_min + int(np.argmax(plane_counts)))
        center_mask = mask_zyx[center_plane] == label
        plane_y, plane_x = np.where(center_mask)
        relative_coords = np.stack([plane_y - centroid_y, plane_x - centroid_x], axis=1).astype(np.float32)
        patch = _extract_patch(center_mask, centroid_y, centroid_x, radius=patch_radius)
        dist_to_edge = _edge_distance((height, width), centroid_y, centroid_x)
        records.append(
            ROIRecord(
                day=day_name,
                label=int(label),
                centroid_z=centroid_z,
                centroid_y=centroid_y,
                centroid_x=centroid_x,
                center_plane=center_plane,
                area_center=int(center_mask.sum()),
                area_total=int(len(z_coords)),
                patch=patch,
                radial_profile=_radial_profile(patch),
                center_coords=relative_coords,
                dist_to_edge=dist_to_edge,
                is_edge=bool(dist_to_edge < edge_margin),
                bbox_zyx=((z_min, z_max), (y_min, y_max), (x_min, x_max)),
            )
        )
    return records


def estimate_pair_shift(mask_a: np.ndarray, mask_b: np.ndarray, max_shift: int = 32) -> tuple[float, float]:
    """Estimate the global XY shift that aligns ``mask_b`` to ``mask_a``.

    Parameters
    ----------
    mask_a : numpy.ndarray
        Reference label stack with shape ``(z, y, x)``.
    mask_b : numpy.ndarray
        Moving label stack with shape ``(z, y, x)``.
    max_shift : int, default=32
        Maximum searched absolute shift in pixels.

    Returns
    -------
    tuple[float, float]
        Translation ``(shift_y, shift_x)`` in pixels that should be added to
        ``mask_b`` XY coordinates to align them with ``mask_a``.
    """

    projection_a = (mask_a > 0).max(axis=0).astype(np.float32)
    projection_b = (mask_b > 0).max(axis=0).astype(np.float32)
    return _phase_correlation_shift(projection_a, projection_b, max_shift=max_shift)


def score_candidate_pair(
    record_a: ROIRecord,
    record_b: ROIRecord,
    shift_yx: tuple[float, float],
    params: MatchParams,
) -> tuple[float, dict[str, float]]:
    """Score one candidate ROI pairing after XY shift correction.

    Parameters
    ----------
    record_a : ROIRecord
        ROI from the reference session.
    record_b : ROIRecord
        ROI from the moving session whose centroid should be shifted by
        ``shift_yx`` before comparison.
    shift_yx : tuple[float, float]
        Translation ``(shift_y, shift_x)`` in pixels applied to ``record_b``.
    params : MatchParams
        Matcher thresholds and scoring settings.

    Returns
    -------
    tuple[float, dict[str, float]]
        Pair score on ``[0, 1]`` and a component dictionary containing the
        measured geometry and shape terms.
    """

    shift_y, shift_x = shift_yx
    residual_y = record_a.centroid_y - (record_b.centroid_y + shift_y)
    residual_x = record_a.centroid_x - (record_b.centroid_x + shift_x)
    dxy = float(math.hypot(residual_y, residual_x))
    dz = float(abs(record_a.centroid_z - record_b.centroid_z))
    local_max_xy = float(params.max_dist_xy + (params.edge_relax_xy if (record_a.is_edge or record_b.is_edge) else 0.0))
    if dxy > local_max_xy or dz > params.max_dist_z:
        return 0.0, {}

    area_ratio = float(min(record_a.area_center, record_b.area_center) / max(record_a.area_center, record_b.area_center))
    patch_a = record_a.patch.ravel()
    patch_b = record_b.patch.ravel()
    denom = float(np.linalg.norm(patch_a) * np.linalg.norm(patch_b))
    patch_cos = float(np.dot(patch_a, patch_b) / denom) if denom > 0.0 else 0.0
    radial_l1 = float(np.abs(record_a.radial_profile - record_b.radial_profile).sum())
    iou = _coords_iou(record_a.center_coords, record_b.center_coords)
    bbox_iou = _compute_bbox_overlap_fraction(record_a, record_b)
    spatial_term = math.exp(-((dxy / max(1.0, 0.45 * local_max_xy)) ** 2))
    z_term = math.exp(-((dz / max(1.0, 0.55 * params.max_dist_z)) ** 2))
    radial_term = math.exp(-2.0 * radial_l1)
    score = (
        0.28 * spatial_term
        + 0.12 * z_term
        + 0.14 * patch_cos
        + 0.24 * iou
        + 0.08 * area_ratio
        + 0.04 * radial_term
        + 0.10 * bbox_iou
    )
    if record_a.is_edge or record_b.is_edge:
        score += params.edge_score_bonus
    components = {
        "dxy": dxy,
        "dz": dz,
        "area_ratio": area_ratio,
        "patch_cos": patch_cos,
        "radial_l1": radial_l1,
        "iou": iou,
        "bbox_iou": bbox_iou,
        "shift_y": float(shift_y),
        "shift_x": float(shift_x),
        "edge_case": float(record_a.is_edge or record_b.is_edge),
    }
    return float(np.clip(score, 0.0, 1.0)), components


def generate_candidate_pairs(
    records_a: list[ROIRecord],
    records_b: list[ROIRecord],
    shift_yx: tuple[float, float],
    params: MatchParams,
) -> pd.DataFrame:
    """Generate scored candidate matches between two ROI sets.

    Parameters
    ----------
    records_a : list[ROIRecord]
        ROI records from the reference session.
    records_b : list[ROIRecord]
        ROI records from the moving session.
    shift_yx : tuple[float, float]
        Translation ``(shift_y, shift_x)`` in pixels added to ``records_b``
        centroids before geometric gating.
    params : MatchParams
        Matcher thresholds and scoring settings.

    Returns
    -------
    pandas.DataFrame
        Candidate table with one row per surviving ROI pair and columns that
        include ROI indices, labels, score, and score components.
    """

    if not records_a or not records_b:
        return pd.DataFrame(
            columns=[
                "idx_a",
                "label_a",
                "idx_b",
                "label_b",
                "score",
                "dxy",
                "dz",
                "area_ratio",
                "patch_cos",
                "radial_l1",
                "iou",
                "bbox_iou",
                "shift_y",
                "shift_x",
                "edge_case",
            ]
        )

    cell_size = float(max(params.max_dist_xy, params.patch_size - params.overlap, 8))
    cells_b = _build_spatial_hash(records_b, cell_size=cell_size)
    rows: list[dict[str, float | int]] = []
    for index_a, record_a in enumerate(records_a):
        target_y = record_a.centroid_y - shift_yx[0]
        target_x = record_a.centroid_x - shift_yx[1]
        search_radius = float(params.max_dist_xy + (params.edge_relax_xy if record_a.is_edge else 0.0))
        nearby_indices = _query_spatial_hash(cells_b, target_y, target_x, radius=search_radius, cell_size=cell_size)
        if not nearby_indices:
            continue
        for index_b in sorted(set(nearby_indices)):
            record_b = records_b[index_b]
            score, components = score_candidate_pair(record_a, record_b, shift_yx=shift_yx, params=params)
            if score < params.min_score:
                continue
            row = {
                "idx_a": int(index_a),
                "label_a": int(record_a.label),
                "idx_b": int(index_b),
                "label_b": int(record_b.label),
                "score": float(score),
            }
            row.update(components)
            rows.append(row)
    return pd.DataFrame(rows)


def solve_pairwise_assignment(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Select one-to-one matches from a candidate table using a greedy policy.

    Parameters
    ----------
    candidate_table : pandas.DataFrame
        Candidate match table containing at least ``idx_a``, ``idx_b``,
        ``label_a``, ``label_b``, and ``score`` columns.

    Returns
    -------
    pandas.DataFrame
        One-to-one accepted matches with an added ``confidence`` column. This
        precision-first solver leaves ambiguous leftovers unmatched instead of
        forcing a full assignment.
    """

    if candidate_table is None or len(candidate_table) == 0:
        return pd.DataFrame(columns=["idx_a", "label_a", "idx_b", "label_b", "score", "confidence"])

    table = candidate_table.copy()
    best_by_a = table.groupby("idx_a")["score"].apply(lambda values: sorted(values.tolist(), reverse=True))
    best_by_b = table.groupby("idx_b")["score"].apply(lambda values: sorted(values.tolist(), reverse=True))
    confidences = []
    for _, row in table.iterrows():
        a_scores = best_by_a[int(row["idx_a"])]
        b_scores = best_by_b[int(row["idx_b"])]
        margin_a = float(row["score"] - (a_scores[1] if len(a_scores) > 1 else 0.0))
        margin_b = float(row["score"] - (b_scores[1] if len(b_scores) > 1 else 0.0))
        confidence = float(np.clip(0.70 * row["score"] + 0.15 * margin_a + 0.15 * margin_b, 0.0, 1.0))
        confidences.append(confidence)
    table["confidence"] = confidences
    table = table.sort_values(["score", "confidence"], ascending=[False, False]).reset_index(drop=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    accepted_rows = []
    for _, row in table.iterrows():
        index_a = int(row["idx_a"])
        index_b = int(row["idx_b"])
        if index_a in used_a or index_b in used_b:
            continue
        best_for_a = float(best_by_a[index_a][0])
        best_for_b = float(best_by_b[index_b][0])
        if float(row["score"]) + 1e-9 < best_for_a:
            continue
        if float(row["score"]) + 1e-9 < best_for_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        accepted_rows.append(row.to_dict())
    if not accepted_rows:
        return table.iloc[0:0].copy()
    return pd.DataFrame(accepted_rows)


def build_tracks_from_pairwise_matches(
    day_names: list[str],
    pair_tables: dict[tuple[str, str], pd.DataFrame],
    min_gap_support: float = 0.80,
) -> pd.DataFrame:
    """Assemble multi-week ROI tracks from pairwise match tables.

    Parameters
    ----------
    day_names : list[str]
        Ordered session identifiers.
    pair_tables : dict[tuple[str, str], pandas.DataFrame]
        Mapping from ordered day pairs to one-to-one pairwise match tables.
        Adjacent pairs define the backbone. One-week skip pairs may be used to
        bridge exactly one missing intermediate day when support is strong.
    min_gap_support : float, default=0.80
        Minimum confidence required for a gap bridge.

    Returns
    -------
    pandas.DataFrame
        Track table with one row per putative cell identity across days.
    """

    next_map: dict[tuple[str, int], tuple[str, int, float]] = {}
    prev_nodes: set[tuple[str, int]] = set()
    all_nodes: set[tuple[str, int]] = set()

    for day_index in range(len(day_names) - 1):
        day_a = day_names[day_index]
        day_b = day_names[day_index + 1]
        pair_table = pair_tables.get((day_a, day_b))
        if pair_table is None or len(pair_table) == 0:
            continue
        for _, row in pair_table.iterrows():
            node_a = (day_a, int(row["label_a"]))
            node_b = (day_b, int(row["label_b"]))
            confidence = float(row["confidence"] if "confidence" in row else row["score"])
            next_map[node_a] = (day_b, int(row["label_b"]), confidence)
            prev_nodes.add(node_b)
            all_nodes.add(node_a)
            all_nodes.add(node_b)

    for (day_a, day_b), pair_table in pair_tables.items():
        if pair_table is None or len(pair_table) == 0:
            continue
        all_nodes.update((day_a, int(value)) for value in pair_table["label_a"].astype(int).tolist())
        all_nodes.update((day_b, int(value)) for value in pair_table["label_b"].astype(int).tolist())

    starts = [node for node in sorted(all_nodes) if node not in prev_nodes]
    seen: set[tuple[str, int]] = set()
    tracks: list[dict[str, object]] = []
    for start_node in starts:
        if start_node in seen:
            continue
        track: dict[str, object] = {"_confidences": [], "used_gap_bridge": False}
        current = start_node
        while current not in seen:
            seen.add(current)
            current_day, current_label = current
            track[f"{current_day}_roi"] = int(current_label)
            next_node = next_map.get(current)
            if next_node is None:
                break
            track["_confidences"].append(float(next_node[2]))
            current = (next_node[0], next_node[1])
        _summarize_track(track, day_names)
        tracks.append(track)

    for node in sorted(all_nodes):
        if node in seen:
            continue
        day_name, label = node
        track = {f"{day_name}_roi": int(label), "_confidences": [], "used_gap_bridge": False}
        _summarize_track(track, day_names)
        tracks.append(track)

    gap_rows = []
    for (day_a, day_b), pair_table in pair_tables.items():
        index_a = day_names.index(day_a)
        index_b = day_names.index(day_b)
        if index_b - index_a != 2:
            continue
        if pair_table is None or len(pair_table) == 0:
            continue
        for _, row in pair_table.iterrows():
            confidence = float(row["confidence"] if "confidence" in row else row["score"])
            if confidence < min_gap_support:
                continue
            gap_rows.append(
                {
                    "day_a": day_a,
                    "label_a": int(row["label_a"]),
                    "day_b": day_b,
                    "label_b": int(row["label_b"]),
                    "confidence": confidence,
                }
            )
    gap_rows.sort(key=lambda row: row["confidence"], reverse=True)

    for gap_row in gap_rows:
        node_to_track = _track_node_map(tracks, day_names)
        source_node = (str(gap_row["day_a"]), int(gap_row["label_a"]))
        target_node = (str(gap_row["day_b"]), int(gap_row["label_b"]))
        if source_node not in node_to_track or target_node not in node_to_track:
            continue
        source_track_index = node_to_track[source_node]
        target_track_index = node_to_track[target_node]
        if source_track_index == target_track_index:
            continue

        source_track = tracks[source_track_index]
        target_track = tracks[target_track_index]
        source_day_index = day_names.index(source_node[0])
        target_day_index = day_names.index(target_node[0])
        if target_day_index - source_day_index != 2:
            continue
        intermediate_day = day_names[source_day_index + 1]
        if pd.notna(source_track.get(f"{intermediate_day}_roi", pd.NA)):
            continue
        if pd.notna(target_track.get(f"{intermediate_day}_roi", pd.NA)):
            continue

        conflict_found = False
        for day_name in day_names:
            source_value = source_track.get(f"{day_name}_roi", pd.NA)
            target_value = target_track.get(f"{day_name}_roi", pd.NA)
            if pd.notna(source_value) and pd.notna(target_value) and int(source_value) != int(target_value):
                conflict_found = True
                break
        if conflict_found:
            continue

        merged_track: dict[str, object] = {
            "_confidences": list(source_track.get("_confidences", [])) + list(target_track.get("_confidences", [])),
            "used_gap_bridge": True,
        }
        for day_name in day_names:
            source_value = source_track.get(f"{day_name}_roi", pd.NA)
            target_value = target_track.get(f"{day_name}_roi", pd.NA)
            if pd.notna(source_value):
                merged_track[f"{day_name}_roi"] = int(source_value)
            elif pd.notna(target_value):
                merged_track[f"{day_name}_roi"] = int(target_value)
        merged_track["_confidences"].append(float(gap_row["confidence"]))
        _summarize_track(merged_track, day_names)

        lower_index = min(source_track_index, target_track_index)
        upper_index = max(source_track_index, target_track_index)
        tracks[lower_index] = merged_track
        del tracks[upper_index]

    return _finalize_tracks(tracks, day_names)


def add_qc_flags(tracks_table: pd.DataFrame, low_confidence_threshold: float = 0.55) -> pd.DataFrame:
    """Attach simple QC flags to a track table.

    Parameters
    ----------
    tracks_table : pandas.DataFrame
        Track table with confidence summaries and optional edge metadata.
    low_confidence_threshold : float, default=0.55
        Threshold for ``low_confidence`` review flags.

    Returns
    -------
    pandas.DataFrame
        Copy of ``tracks_table`` with boolean QC columns added.
    """

    table = tracks_table.copy()
    if "min_confidence" not in table.columns:
        table["min_confidence"] = 0.0
    if "used_gap_bridge" not in table.columns:
        table["used_gap_bridge"] = False
    if "any_edge_roi" not in table.columns:
        table["any_edge_roi"] = False
    if "missing_intermediate_days" not in table.columns:
        table["missing_intermediate_days"] = 0

    table["low_confidence"] = table["min_confidence"].astype(float) < float(low_confidence_threshold)
    table["gap_bridge_qc"] = table["used_gap_bridge"].fillna(False).astype(bool)
    table["edge_qc"] = table["any_edge_roi"].fillna(False).astype(bool)
    table["fragmented_track_qc"] = pd.to_numeric(table["missing_intermediate_days"], errors="coerce").fillna(0).astype(int) > 0
    table["needs_review"] = (
        table["low_confidence"].astype(bool)
        | table["gap_bridge_qc"].astype(bool)
        | table["edge_qc"].astype(bool)
        | table["fragmented_track_qc"].astype(bool)
    )
    return table




def _emit_log(log_fn: Callable[[str], None] | None, message: str) -> None:
    """Send one progress message to an optional logging callback."""

    if log_fn is not None:
        log_fn(message)


def match_roi_masks(
    mask_stacks: list[np.ndarray],
    day_names: list[str],
    params: MatchParams | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Run end-to-end ROI matching on multiple labeled mask stacks.

    Parameters
    ----------
    mask_stacks : list[numpy.ndarray]
        List of integer label stacks, each with shape ``(z, y, x)`` in a common
        registered space.
    day_names : list[str]
        Session names aligned one-to-one with ``mask_stacks``.
    params : MatchParams or None, default=None
        Matcher parameters. When ``None``, :class:`MatchParams` defaults are
        used.
    log_fn : collections.abc.Callable[[str], None] or None, default=None
        Optional callback used for lightweight console-style progress logging.

    Returns
    -------
    tuple[pandas.DataFrame, dict[tuple[str, str], pandas.DataFrame], pandas.DataFrame]
        ``(tracks_table, pair_tables, qc_table)`` where ``pair_tables`` contains
        adjacent and one-gap pairwise match tables.
    """

    if params is None:
        params = MatchParams()
    if len(mask_stacks) != len(day_names):
        raise ValueError("mask_stacks and day_names must have the same length")
    if len(mask_stacks) == 0:
        raise ValueError("mask_stacks must not be empty")

    expected_shape = tuple(mask_stacks[0].shape)
    for mask_stack in mask_stacks[1:]:
        if tuple(mask_stack.shape) != expected_shape:
            raise ValueError("All mask stacks must share the same (z, y, x) shape")

    _emit_log(
        log_fn,
        f"[roi_matcher] Starting match run for {len(day_names)} sessions with shared shape {expected_shape}.",
    )
    all_records: dict[str, list[ROIRecord]] = {}
    for day_name, mask_stack in zip(day_names, mask_stacks):
        _emit_log(log_fn, f"[roi_matcher] Extracting ROI records for {day_name}...")
        records = extract_roi_records(
            mask_zyx=mask_stack,
            day_name=day_name,
            patch_radius=params.patch_radius,
            edge_margin=params.edge_margin,
        )
        all_records[day_name] = records
        _emit_log(log_fn, f"[roi_matcher]   {day_name}: extracted {len(records)} ROIs")

    pair_tables: dict[tuple[str, str], pd.DataFrame] = {}
    max_pair_span = 2
    total_pairs = sum(
        min(len(day_names), index_a + max_pair_span + 1) - (index_a + 1)
        for index_a in range(len(day_names))
    )
    pair_counter = 0
    for index_a, day_a in enumerate(day_names):
        for index_b in range(index_a + 1, min(len(day_names), index_a + max_pair_span + 1)):
            day_b = day_names[index_b]
            pair_counter += 1
            _emit_log(log_fn, f"[roi_matcher] Pair {pair_counter}/{total_pairs}: {day_a} vs {day_b}")
            shift_yx = _select_pair_shift(
                records_a=all_records[day_a],
                records_b=all_records[day_b],
                mask_a=mask_stacks[index_a],
                mask_b=mask_stacks[index_b],
                params=params,
            )
            candidate_table = generate_candidate_pairs(
                records_a=all_records[day_a],
                records_b=all_records[day_b],
                shift_yx=shift_yx,
                params=params,
            )
            pair_metadata = {
                "week_a": str(day_a),
                "week_b": str(day_b),
                "shift_y": float(shift_yx[0]),
                "shift_x": float(shift_yx[1]),
                "candidate_count": int(len(candidate_table)),
            }
            assigned_table = solve_pairwise_assignment(candidate_table)
            if len(assigned_table) > 0:
                assigned_table = assigned_table.sort_values(["score", "confidence"], ascending=[False, False]).reset_index(drop=True)
            assigned_table.attrs.update(pair_metadata)
            assigned_table.attrs["accepted_match_count"] = int(len(assigned_table))
            pair_tables[(day_a, day_b)] = assigned_table
            _emit_log(
                log_fn,
                f"[roi_matcher]   shift=(dy={shift_yx[0]:.2f}, dx={shift_yx[1]:.2f}) | "
                f"candidates={len(candidate_table)} | accepted={len(assigned_table)}",
            )

    _emit_log(log_fn, "[roi_matcher] Building multi-week tracks from pairwise matches...")
    tracks_table = build_tracks_from_pairwise_matches(
        day_names=day_names,
        pair_tables=pair_tables,
        min_gap_support=params.min_gap_support,
    )

    records_by_node = {
        (day_name, record.label): record
        for day_name, records in all_records.items()
        for record in records
    }
    tracked_nodes = {
        (day_name, int(row[f"{day_name}_roi"]))
        for _, row in tracks_table.iterrows()
        for day_name in day_names
        if pd.notna(row[f"{day_name}_roi"])
    }
    singleton_rows: list[dict[str, object]] = []
    for day_name, records in all_records.items():
        for record in records:
            if (day_name, record.label) in tracked_nodes:
                continue
            row = {f"{name}_roi": pd.NA for name in day_names}
            row[f"{day_name}_roi"] = int(record.label)
            row["n_days_present"] = 1
            row["mean_confidence"] = 0.0
            row["min_confidence"] = 0.0
            row["used_gap_bridge"] = False
            row["missing_intermediate_days"] = 0
            singleton_rows.append(row)
    if singleton_rows:
        tracks_table = pd.concat([tracks_table, pd.DataFrame(singleton_rows)], ignore_index=True)
    _emit_log(log_fn, f"[roi_matcher] Added {len(singleton_rows)} singleton tracks.")

    edge_flags = []
    edge_distances = []
    for _, row in tracks_table.iterrows():
        row_edge_flags = []
        row_edge_distances = []
        for day_name in day_names:
            label_value = row.get(f"{day_name}_roi", pd.NA)
            if pd.isna(label_value):
                continue
            record = records_by_node[(day_name, int(label_value))]
            row_edge_flags.append(bool(record.is_edge))
            row_edge_distances.append(float(record.dist_to_edge))
        edge_flags.append(any(row_edge_flags))
        edge_distances.append(float(min(row_edge_distances)) if row_edge_distances else math.nan)
    tracks_table["any_edge_roi"] = edge_flags
    tracks_table["min_dist_to_edge"] = edge_distances
    tracks_table = tracks_table.sort_values(["n_days_present", "mean_confidence"], ascending=[False, False]).reset_index(drop=True)
    tracks_table["cluster_id"] = np.arange(1, len(tracks_table) + 1)

    qc_table = add_qc_flags(tracks_table, low_confidence_threshold=params.low_confidence_threshold)
    n_review = int(qc_table["needs_review"].sum()) if "needs_review" in qc_table.columns else 0
    _emit_log(
        log_fn,
        f"[roi_matcher] Finished matching: {len(tracks_table)} total tracks, {n_review} flagged for review.",
    )
    return tracks_table, pair_tables, qc_table


def _quantile_or_nan(values: pd.Series, quantile: float) -> float:
    """Return one quantile from a numeric series or ``nan`` when empty."""

    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return math.nan
    return float(numeric_values.quantile(quantile))


def build_pairwise_match_diagnostics_table(
    pair_tables: dict[tuple[str, str], pd.DataFrame],
    xy_um_per_px: float = DEFAULT_XY_UM_PER_PX,
    z_um_per_plane: float = DEFAULT_Z_UM_PER_PLANE,
) -> pd.DataFrame:
    """Build one long accepted-match diagnostic table across all session pairs."""

    columns = [
        "week_a",
        "week_b",
        "roi_a",
        "roi_b",
        "distance_xy_px",
        "distance_xy_um",
        "distance_z_planes",
        "distance_z_um",
        "match_score",
        "confidence",
    ]
    rows: list[dict[str, float | int | str]] = []
    for (day_a, day_b), pair_table in pair_tables.items():
        if pair_table is None or len(pair_table) == 0:
            continue
        for _, row in pair_table.iterrows():
            distance_xy_px = float(row["dxy"]) if "dxy" in row else math.nan
            distance_z_planes = float(row["dz"]) if "dz" in row else math.nan
            rows.append(
                {
                    "week_a": str(day_a),
                    "week_b": str(day_b),
                    "roi_a": int(row["label_a"]),
                    "roi_b": int(row["label_b"]),
                    "distance_xy_px": distance_xy_px,
                    "distance_xy_um": distance_xy_px * float(xy_um_per_px) if math.isfinite(distance_xy_px) else math.nan,
                    "distance_z_planes": distance_z_planes,
                    "distance_z_um": distance_z_planes * float(z_um_per_plane) if math.isfinite(distance_z_planes) else math.nan,
                    "match_score": float(row["score"]),
                    "confidence": float(row["confidence"]) if "confidence" in row else math.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_pairwise_match_summary_table(
    pair_tables: dict[tuple[str, str], pd.DataFrame],
    xy_um_per_px: float = DEFAULT_XY_UM_PER_PX,
    z_um_per_plane: float = DEFAULT_Z_UM_PER_PLANE,
) -> pd.DataFrame:
    """Summarize candidate and accepted-match diagnostics for each session pair."""

    rows: list[dict[str, float | int | str]] = []
    for (day_a, day_b), pair_table in pair_tables.items():
        pair_table = pair_table if pair_table is not None else pd.DataFrame()
        dxy_values = pair_table["dxy"] if "dxy" in pair_table.columns else pd.Series(dtype=float)
        dz_values = pair_table["dz"] if "dz" in pair_table.columns else pd.Series(dtype=float)
        score_values = pair_table["score"] if "score" in pair_table.columns else pd.Series(dtype=float)
        confidence_values = pair_table["confidence"] if "confidence" in pair_table.columns else pd.Series(dtype=float)

        median_xy_px = _quantile_or_nan(dxy_values, 0.5)
        max_xy_px = float(pd.to_numeric(dxy_values, errors="coerce").max()) if not dxy_values.empty else math.nan
        median_z_planes = _quantile_or_nan(dz_values, 0.5)
        max_z_planes = float(pd.to_numeric(dz_values, errors="coerce").max()) if not dz_values.empty else math.nan
        rows.append(
            {
                "week_a": str(day_a),
                "week_b": str(day_b),
                "candidate_pairs": int(pair_table.attrs.get("candidate_count", len(pair_table))),
                "accepted_reciprocal_matches": int(pair_table.attrs.get("accepted_match_count", len(pair_table))),
                "shift_y_px": float(pair_table.attrs.get("shift_y", math.nan)),
                "shift_x_px": float(pair_table.attrs.get("shift_x", math.nan)),
                "median_accepted_xy_px": median_xy_px,
                "median_accepted_xy_um": median_xy_px * float(xy_um_per_px) if math.isfinite(median_xy_px) else math.nan,
                "max_accepted_xy_px": max_xy_px,
                "max_accepted_xy_um": max_xy_px * float(xy_um_per_px) if math.isfinite(max_xy_px) else math.nan,
                "median_accepted_z_planes": median_z_planes,
                "median_accepted_z_um": median_z_planes * float(z_um_per_plane) if math.isfinite(median_z_planes) else math.nan,
                "max_accepted_z_planes": max_z_planes,
                "max_accepted_z_um": max_z_planes * float(z_um_per_plane) if math.isfinite(max_z_planes) else math.nan,
                "score_p10": _quantile_or_nan(score_values, 0.10),
                "score_median": _quantile_or_nan(score_values, 0.50),
                "score_p90": _quantile_or_nan(score_values, 0.90),
                "confidence_p10": _quantile_or_nan(confidence_values, 0.10),
                "confidence_median": _quantile_or_nan(confidence_values, 0.50),
                "confidence_p90": _quantile_or_nan(confidence_values, 0.90),
            }
        )
    return pd.DataFrame(rows)


def build_track_length_summary_table(
    tracks_table: pd.DataFrame,
    thresholds: tuple[int, ...] = DEFAULT_TRACK_LENGTH_THRESHOLDS,
) -> pd.DataFrame:
    """Count how many tracks last at least the requested number of sessions."""

    n_days_present = pd.to_numeric(tracks_table.get("n_days_present", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    rows = [
        {
            "minimum_weeks_present": int(threshold),
            "n_tracks": int((n_days_present >= int(threshold)).sum()),
        }
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def export_match_tables(
    output_prefix: Path | str,
    tracks_table: pd.DataFrame,
    pair_tables: dict[tuple[str, str], pd.DataFrame],
    qc_table: pd.DataFrame,
    *,
    xy_um_per_px: float = DEFAULT_XY_UM_PER_PX,
    z_um_per_plane: float = DEFAULT_Z_UM_PER_PLANE,
) -> dict[str, object]:
    """Write track, pairwise, and QC tables to CSV files.

    Parameters
    ----------
    output_prefix : pathlib.Path or str
        Output path prefix. The function writes ``<prefix>.csv`` for tracks,
        ``<prefix>_qc.csv`` for QC, and one ``<prefix>_<day_a>_vs_<day_b>.csv``
        file per pair table.
    tracks_table : pandas.DataFrame
        Multi-week track table.
    pair_tables : dict[tuple[str, str], pandas.DataFrame]
        Pairwise match tables keyed by ordered day pairs.
    qc_table : pandas.DataFrame
        QC table aligned row-for-row with ``tracks_table``.

    Returns
    -------
    dict[str, object]
        Dictionary containing output paths under ``tracks``, ``qc``,
        ``pair_tables``, ``pair_diagnostics``, ``pair_summary``, and
        ``track_length_summary`` keys.
    """

    prefix_path = Path(output_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_path = prefix_path.with_suffix(".csv")
    qc_path = prefix_path.with_name(f"{prefix_path.name}_qc.csv")
    tracks_table.to_csv(tracks_path, index=False)
    qc_table.to_csv(qc_path, index=False)
    pair_paths = []
    for (day_a, day_b), pair_table in pair_tables.items():
        pair_path = prefix_path.with_name(f"{prefix_path.name}_{day_a}_vs_{day_b}.csv")
        pair_table.to_csv(pair_path, index=False)
        pair_paths.append(pair_path)

    pair_diagnostics_table = build_pairwise_match_diagnostics_table(
        pair_tables=pair_tables,
        xy_um_per_px=xy_um_per_px,
        z_um_per_plane=z_um_per_plane,
    )
    pair_diagnostics_path = prefix_path.with_name(f"{prefix_path.name}_pairwise_match_diagnostics.csv")
    pair_diagnostics_table.to_csv(pair_diagnostics_path, index=False)

    pair_summary_table = build_pairwise_match_summary_table(
        pair_tables=pair_tables,
        xy_um_per_px=xy_um_per_px,
        z_um_per_plane=z_um_per_plane,
    )
    pair_summary_path = prefix_path.with_name(f"{prefix_path.name}_pairwise_match_summary.csv")
    pair_summary_table.to_csv(pair_summary_path, index=False)

    track_length_summary_table = build_track_length_summary_table(tracks_table)
    track_length_summary_path = prefix_path.with_name(f"{prefix_path.name}_track_length_summary.csv")
    track_length_summary_table.to_csv(track_length_summary_path, index=False)

    return {
        "tracks": tracks_path,
        "qc": qc_path,
        "pair_tables": pair_paths,
        "pair_diagnostics": pair_diagnostics_path,
        "pair_summary": pair_summary_path,
        "track_length_summary": track_length_summary_path,
    }


def write_run_log(
    output_prefix: Path,
    day_names: list[str],
    mask_paths: list[str],
    params: MatchParams,
    output_paths: dict[str, object],
    total_duration_seconds: float,
) -> Path:
    """Write a JSON run log for one matcher CLI execution."""

    run_log_path = output_prefix.with_name(f"{output_prefix.name}_run_log.json")
    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "day_names": day_names,
        "mask_paths": mask_paths,
        "params": asdict(params),
        "output_paths": _serialize_output_paths(output_paths),
        "total_duration_seconds": float(total_duration_seconds),
        "total_duration_hms": format_duration_seconds(total_duration_seconds),
    }
    run_log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_log_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ROI matcher script.

    Parameters
    ----------
    argv : list[str] or None, default=None
        Optional command-line token list. When ``None``, arguments are read from
        ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """

    parser = argparse.ArgumentParser(description="Match weekly ROI masks in registered space.")
    parser.add_argument("--masks", nargs="+", required=True, help="Paths to labeled 3D ROI mask TIFF stacks.")
    parser.add_argument("--days", nargs="+", required=False, help="Optional day names aligned to --masks.")
    parser.add_argument("--output-prefix", required=True, help="Output CSV prefix for tracks and QC tables.")
    parser.add_argument("--patch-radius", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--max-dist-xy", type=float, default=15.0)
    parser.add_argument("--max-dist-z", type=float, default=5.0)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--min-gap-support", type=float, default=0.80)
    parser.add_argument("--edge-margin", type=int, default=20)
    parser.add_argument("--edge-relax-xy", type=float, default=3.0)
    parser.add_argument("--edge-score-bonus", type=float, default=0.02)
    parser.add_argument("--translation-max-shift", type=int, default=32)
    parser.add_argument("--disable-translation", action="store_true")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging during matching.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the ROI matcher CLI and write CSV outputs.

    Parameters
    ----------
    argv : list[str] or None, default=None
        Optional command-line token list. When ``None``, arguments are read from
        ``sys.argv``.

    Returns
    -------
    None
        Results are written to CSV files on disk.
    """

    run_start_seconds = time.perf_counter()
    args = parse_args(argv)
    day_names = args.days if args.days else [f"day{index + 1}" for index in range(len(args.masks))]
    if len(day_names) != len(args.masks):
        raise ValueError("The number of --days entries must match the number of --masks entries.")
    output_prefix = Path(args.output_prefix)
    params = MatchParams(
        patch_radius=args.patch_radius,
        patch_size=args.patch_size,
        overlap=args.overlap,
        max_dist_xy=args.max_dist_xy,
        max_dist_z=args.max_dist_z,
        min_score=args.min_score,
        min_gap_support=args.min_gap_support,
        edge_margin=args.edge_margin,
        edge_relax_xy=args.edge_relax_xy,
        edge_score_bonus=args.edge_score_bonus,
        translation_max_shift=args.translation_max_shift,
        use_translation=(not args.disable_translation),
        low_confidence_threshold=args.low_confidence_threshold,
    )
    log_fn = None if args.quiet else print
    _emit_log(log_fn, f"[roi_matcher] Starting ROI matching for {len(day_names)} sessions")
    _emit_log(log_fn, f"[roi_matcher] Loading {len(args.masks)} mask stacks...")
    mask_stacks = []
    for day_name, mask_path in zip(day_names, args.masks):
        _emit_log(log_fn, f"[roi_matcher]   reading {day_name}: {mask_path}")
        mask_stacks.append(tifffile.imread(Path(mask_path)))
    _emit_log(log_fn, "[roi_matcher] Running pairwise and gap-aware matching")
    tracks_table, pair_tables, qc_table = match_roi_masks(
        mask_stacks=mask_stacks,
        day_names=day_names,
        params=params,
        log_fn=log_fn,
    )
    _emit_log(log_fn, f"[roi_matcher] Matching produced {len(tracks_table)} tracks")
    _emit_log(log_fn, f"[roi_matcher] Writing outputs with prefix {output_prefix}")
    output_paths = export_match_tables(
        output_prefix=output_prefix,
        tracks_table=tracks_table,
        pair_tables=pair_tables,
        qc_table=qc_table,
    )
    total_duration_seconds = time.perf_counter() - run_start_seconds
    run_log_path = write_run_log(
        output_prefix=output_prefix,
        day_names=day_names,
        mask_paths=[str(Path(mask_path)) for mask_path in args.masks],
        params=params,
        output_paths=output_paths,
        total_duration_seconds=total_duration_seconds,
    )
    print(tracks_table.head(20).to_string())
    print(f"saved tracks to {output_paths['tracks']}")
    print(f"saved qc to {output_paths['qc']}")
    print(f"saved pair diagnostics to {output_paths['pair_diagnostics']}")
    print(f"saved pair summary to {output_paths['pair_summary']}")
    print(f"saved track-length summary to {output_paths['track_length_summary']}")
    print(f"saved run log to {run_log_path}")
    print(f"[roi_matcher] Total duration: {format_duration_seconds(total_duration_seconds)}")
    print(f"total_duration={format_duration_seconds(total_duration_seconds)}")


if __name__ == "__main__":
    main()
