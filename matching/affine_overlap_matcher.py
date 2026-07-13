"""Affine-overlap ROI matching for daywise Cellpose-SAM masks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from skimage.registration import phase_cross_correlation


MATCHER_ALGORITHM_VERSION = "affine_overlap_v1"


def _as_float(value: Any) -> float:
    """Return a float value from a numeric-like object."""

    return float(np.asarray(value, dtype=float))


def _as_int(value: Any) -> int:
    """Return an int value from a numeric-like object."""

    return int(np.asarray(value).item())


def _require_positive_finite(name: str, value: float) -> float:
    """Validate that a numeric parameter is finite and strictly positive."""

    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


def _require_nonnegative_finite(name: str, value: float) -> float:
    """Validate that a numeric parameter is finite and nonnegative."""

    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _extract_field(row: Mapping[str, Any] | pd.Series, field: str) -> Any:
    """Return one field from a mapping-like object or pandas row."""

    if isinstance(row, pd.Series):
        return row[field]
    return row[field]


def _validate_mask_array(mask_zyx: np.ndarray) -> np.ndarray:
    """Validate one labeled mask stack for matching."""

    mask = np.asarray(mask_zyx)
    if mask.ndim != 3:
        raise ValueError("Mask arrays must be 3D (z, y, x).")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Mask arrays must use an integer dtype.")
    if np.any(mask < 0):
        raise ValueError("Mask labels must be nonnegative.")
    if not np.any(mask == 0):
        raise ValueError("Mask arrays must include background label 0.")
    if not np.any(mask > 0):
        raise ValueError("All-background masks are not allowed.")
    return mask


@dataclass(frozen=True)
class VoxelSpacing:
    """Physical spacing in ``(z, y, x)`` order."""

    z_um: float = 5.0
    y_um: float = 710.0 / 1024.0
    x_um: float = 710.0 / 1024.0

    def __post_init__(self) -> None:
        _require_positive_finite("z_um", self.z_um)
        _require_positive_finite("y_um", self.y_um)
        _require_positive_finite("x_um", self.x_um)

    def as_zyx_array(self) -> np.ndarray:
        """Return spacing as a float array in ``(z, y, x)`` order."""

        return np.asarray([self.z_um, self.y_um, self.x_um], dtype=float)


@dataclass(frozen=True)
class AffineOverlapParams:
    """Parameter bundle for the affine-overlap matcher."""

    occupancy_sigma_xy: float = 2.0
    phase_upsample_factor: int = 10
    seed_min_dice: float = 0.50
    min_affine_seeds: int = 50
    affine_trim_iterations: int = 4
    affine_trim_quantile: float = 0.95
    affine_max_residual_um: float = 8.0
    centroid_candidate_max_distance_um: float = 10.0
    overlap_candidate_min_dice: float = 0.10
    overlap_only_ambiguity: float = 1.25
    spatial_score_scale_um: float = 4.0
    weight_spatial: float = 0.40
    weight_dice: float = 0.32
    weight_area_ratio: float = 0.15
    weight_ambiguity: float = 0.13

    def __post_init__(self) -> None:
        _require_positive_finite("occupancy_sigma_xy", self.occupancy_sigma_xy)
        if int(self.phase_upsample_factor) < 1:
            raise ValueError("phase_upsample_factor must be at least 1.")
        if not 0.0 <= float(self.seed_min_dice) <= 1.0:
            raise ValueError("seed_min_dice must be within [0, 1].")
        if int(self.min_affine_seeds) < 1:
            raise ValueError("min_affine_seeds must be positive.")
        if int(self.affine_trim_iterations) < 1:
            raise ValueError("affine_trim_iterations must be positive.")
        if not 0.0 < float(self.affine_trim_quantile) < 1.0:
            raise ValueError("affine_trim_quantile must be within (0, 1).")
        _require_positive_finite("affine_max_residual_um", self.affine_max_residual_um)
        _require_positive_finite("centroid_candidate_max_distance_um", self.centroid_candidate_max_distance_um)
        if not 0.0 <= float(self.overlap_candidate_min_dice) <= 1.0:
            raise ValueError("overlap_candidate_min_dice must be within [0, 1].")
        _require_positive_finite("overlap_only_ambiguity", self.overlap_only_ambiguity)
        _require_positive_finite("spatial_score_scale_um", self.spatial_score_scale_um)
        for name, weight in (
            ("weight_spatial", self.weight_spatial),
            ("weight_dice", self.weight_dice),
            ("weight_area_ratio", self.weight_area_ratio),
            ("weight_ambiguity", self.weight_ambiguity),
        ):
            _require_nonnegative_finite(name, weight)
        weight_sum = float(self.weight_spatial + self.weight_dice + self.weight_area_ratio + self.weight_ambiguity)
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Score weights must sum to approximately 1.0.")


@dataclass(frozen=True)
class ROIRecord:
    """Convenience structure for a single ROI feature row."""

    session_id: str
    label: int
    centroid_z: float
    centroid_y: float
    centroid_x: float
    area_voxels: float
    bbox_z0: int
    bbox_y0: int
    bbox_x0: int
    bbox_z1: int
    bbox_y1: int
    bbox_x1: int


@dataclass(frozen=True)
class RestrictedTransform:
    """Restricted B-to-A affine transform with no Z/XY cross-coupling."""

    z_intercept: float
    z_scale: float
    y_intercept: float
    y_from_y: float
    y_from_x: float
    x_intercept: float
    x_from_y: float
    x_from_x: float
    method: str
    fallback_reason: str | None
    n_seed: int
    n_inlier: int
    residual_median_um: float | None
    residual_p95_um: float | None

    def apply(self, coordinates_zyx: np.ndarray) -> np.ndarray:
        """Apply the transform to ``(z, y, x)`` coordinates."""

        coordinates = np.asarray(coordinates_zyx, dtype=float)
        scalar_input = coordinates.ndim == 1
        if scalar_input:
            coordinates = coordinates[np.newaxis, :]
        if coordinates.shape[-1] != 3:
            raise ValueError("coordinates_zyx must have shape (..., 3).")
        z = coordinates[:, 0]
        y = coordinates[:, 1]
        x = coordinates[:, 2]
        transformed = np.column_stack(
            [
                self.z_intercept + self.z_scale * z,
                self.y_intercept + self.y_from_y * y + self.y_from_x * x,
                self.x_intercept + self.x_from_y * y + self.x_from_x * x,
            ]
        )
        if scalar_input:
            return transformed[0]
        return transformed


@dataclass
class PairMatchResult:
    """Collected outputs for one matched pair of sessions."""

    candidates: pd.DataFrame
    high_matches: pd.DataFrame
    balanced_matches: pd.DataFrame
    summary: dict[str, object]
    transform: RestrictedTransform


def extract_roi_features(
    mask_zyx: np.ndarray,
    session_id: str,
    spacing: VoxelSpacing,
) -> pd.DataFrame:
    """Extract geometry features for all labeled ROIs in one mask stack."""

    mask = _validate_mask_array(mask_zyx)
    properties = regionprops_table(mask, properties=("label", "area", "centroid", "bbox"))
    labels = np.asarray(properties["label"], dtype=int)
    if labels.size == 0:
        raise ValueError(f"Session {session_id}: no positive ROI labels were found.")

    z0 = np.asarray(properties["bbox-0"], dtype=int)
    y0 = np.asarray(properties["bbox-1"], dtype=int)
    x0 = np.asarray(properties["bbox-2"], dtype=int)
    z1 = np.asarray(properties["bbox-3"], dtype=int)
    y1 = np.asarray(properties["bbox-4"], dtype=int)
    x1 = np.asarray(properties["bbox-5"], dtype=int)
    area_voxels = np.asarray(properties["area"], dtype=float)
    centroid_z = np.asarray(properties["centroid-0"], dtype=float)
    centroid_y = np.asarray(properties["centroid-1"], dtype=float)
    centroid_x = np.asarray(properties["centroid-2"], dtype=float)
    spacing_zyx = spacing.as_zyx_array()

    records = pd.DataFrame(
        {
            "session_id": session_id,
            "label": labels,
            "area_voxels": area_voxels,
            "volume_um3": area_voxels * float(np.prod(spacing_zyx)),
            "centroid_z": centroid_z,
            "centroid_y": centroid_y,
            "centroid_x": centroid_x,
            "centroid_z_um": centroid_z * spacing_zyx[0],
            "centroid_y_um": centroid_y * spacing_zyx[1],
            "centroid_x_um": centroid_x * spacing_zyx[2],
            "bbox_z0": z0,
            "bbox_y0": y0,
            "bbox_x0": x0,
            "bbox_z1": z1,
            "bbox_y1": y1,
            "bbox_x1": x1,
            "bbox_depth_planes": z1 - z0,
            "bbox_height_px": y1 - y0,
            "bbox_width_px": x1 - x0,
            "touches_z_edge": (z0 == 0) | (z1 == mask.shape[0]),
            "touches_xy_edge": (y0 == 0) | (y1 == mask.shape[1]) | (x0 == 0) | (x1 == mask.shape[2]),
        }
    )
    records = records.sort_values("label").reset_index(drop=True)
    records = records.set_index("label", drop=False)
    return records


def _binary_occupancy_projection(mask: np.ndarray) -> np.ndarray:
    """Return a smoothed occupancy projection across z."""

    projection = (mask > 0).sum(axis=0).astype(np.float32)
    return ndimage.gaussian_filter(projection, sigma=2.0)


def _z_occupancy_profile(mask: np.ndarray) -> np.ndarray:
    """Return a centered occupancy profile across z planes."""

    return (mask > 0).sum(axis=(1, 2)).astype(np.float64)


def estimate_global_shift(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    params: AffineOverlapParams,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate coarse ``(z, y, x)`` shift from occupancy structure."""

    mask_a = _validate_mask_array(mask_a)
    mask_b = _validate_mask_array(mask_b)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Mask pairs must share the same shape.")

    summary: dict[str, object] = {
        "xy_fallback": False,
        "z_fallback": False,
        "xy_fallback_reason": None,
        "z_fallback_reason": None,
        "phase_upsample_factor": int(params.phase_upsample_factor),
        "occupancy_sigma_xy": float(params.occupancy_sigma_xy),
    }

    projection_a = _binary_occupancy_projection(mask_a)
    projection_b = _binary_occupancy_projection(mask_b)
    if not np.isfinite(projection_a).all() or not np.isfinite(projection_b).all():
        shift_yx = np.zeros(2, dtype=float)
        summary["xy_fallback"] = True
        summary["xy_fallback_reason"] = "nonfinite_projection"
    elif np.allclose(projection_a, projection_a.flat[0]) or np.allclose(projection_b, projection_b.flat[0]):
        shift_yx = np.zeros(2, dtype=float)
        summary["xy_fallback"] = True
        summary["xy_fallback_reason"] = "constant_projection"
    else:
        try:
            shift_yx, _, _ = phase_cross_correlation(
                projection_a,
                projection_b,
                upsample_factor=int(params.phase_upsample_factor),
                normalization=None,
            )
            shift_yx = np.asarray(shift_yx, dtype=float)
            if not np.isfinite(shift_yx).all():
                raise FloatingPointError("nonfinite phase correlation shift")
        except Exception as exc:  # pragma: no cover - defensive fallback
            shift_yx = np.zeros(2, dtype=float)
            summary["xy_fallback"] = True
            summary["xy_fallback_reason"] = type(exc).__name__

    profile_a = _z_occupancy_profile(mask_a)
    profile_b = _z_occupancy_profile(mask_b)
    centered_a = profile_a - profile_a.mean()
    centered_b = profile_b - profile_b.mean()
    if (
        not np.isfinite(centered_a).all()
        or not np.isfinite(centered_b).all()
        or np.allclose(centered_a, 0.0)
        or np.allclose(centered_b, 0.0)
    ):
        shift_z = 0.0
        summary["z_fallback"] = True
        summary["z_fallback_reason"] = "constant_profile"
    else:
        correlation = fftconvolve(centered_a, centered_b[::-1], mode="full")
        if correlation.size == 0 or not np.isfinite(correlation).any():
            shift_z = 0.0
            summary["z_fallback"] = True
            summary["z_fallback_reason"] = "nonfinite_correlation"
        else:
            shift_z = float(np.argmax(correlation) - (len(centered_b) - 1))

    shift = np.asarray([shift_z, float(shift_yx[0]), float(shift_yx[1])], dtype=float)
    summary["shift_z"] = float(shift[0])
    summary["shift_y"] = float(shift[1])
    summary["shift_x"] = float(shift[2])
    return shift, summary


def _coarse_shift_slices(shape: tuple[int, int, int], shift_zyx: np.ndarray) -> tuple[tuple[slice, slice, slice] | None, tuple[slice, slice, slice] | None]:
    """Return aligned cropped slices for an integer-rounded shift."""

    rounded = np.rint(np.asarray(shift_zyx, dtype=float)).astype(int)
    slices_a: list[slice] = []
    slices_b: list[slice] = []
    for dimension, shift in zip(shape, rounded, strict=True):
        if abs(shift) >= dimension:
            return None, None
        if shift >= 0:
            slices_a.append(slice(shift, dimension))
            slices_b.append(slice(0, dimension - shift))
        else:
            slices_a.append(slice(0, dimension + shift))
            slices_b.append(slice(-shift, dimension))
    return tuple(slices_a), tuple(slices_b)


def build_sparse_overlap_table(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    shift_zyx: np.ndarray,
    area_by_label_a: pd.Series,
    area_by_label_b: pd.Series,
) -> pd.DataFrame:
    """Build a sparse label-pair overlap table after integer-rounded shift."""

    mask_a = _validate_mask_array(mask_a)
    mask_b = _validate_mask_array(mask_b)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Mask pairs must share the same shape.")

    slices_a, slices_b = _coarse_shift_slices(mask_a.shape, shift_zyx)
    columns = ["label_a", "label_b", "intersection_voxels", "dice", "iou"]
    if slices_a is None or slices_b is None:
        return pd.DataFrame(columns=columns)

    aligned_a = mask_a[slices_a]
    aligned_b = mask_b[slices_b]
    positive = (aligned_a > 0) & (aligned_b > 0)
    if not np.any(positive):
        return pd.DataFrame(columns=columns)

    labels_a = aligned_a[positive].astype(np.int64, copy=False)
    labels_b = aligned_b[positive].astype(np.int64, copy=False)
    pair_stack = np.ascontiguousarray(np.column_stack([labels_a, labels_b]))
    unique_pairs, counts = np.unique(pair_stack, axis=0, return_counts=True)
    pair_labels_a = unique_pairs[:, 0].astype(int)
    pair_labels_b = unique_pairs[:, 1].astype(int)

    area_lookup_a = area_by_label_a.astype(float).to_dict()
    area_lookup_b = area_by_label_b.astype(float).to_dict()
    area_a = np.asarray([float(area_lookup_a[int(label)]) for label in pair_labels_a], dtype=float)
    area_b = np.asarray([float(area_lookup_b[int(label)]) for label in pair_labels_b], dtype=float)
    intersection = counts.astype(float)
    dice = 2.0 * intersection / (area_a + area_b)
    iou = intersection / (area_a + area_b - intersection)

    table = pd.DataFrame(
        {
            "label_a": pair_labels_a,
            "label_b": pair_labels_b,
            "intersection_voxels": counts.astype(int),
            "dice": dice,
            "iou": iou,
        }
    )
    return table.sort_values(["label_a", "label_b"]).reset_index(drop=True)


def select_mutual_overlap_pairs(overlap_table: pd.DataFrame) -> pd.DataFrame:
    """Select mutual-best overlap pairs with deterministic tie-breaking."""

    if overlap_table.empty:
        return pd.DataFrame(columns=overlap_table.columns)

    table = overlap_table.copy()
    table = table.sort_values(["label_a", "dice", "label_b"], ascending=[True, False, True])
    best_a = table.drop_duplicates("label_a", keep="first")
    table_b = overlap_table.copy().sort_values(["label_b", "dice", "label_a"], ascending=[True, False, True])
    best_b = table_b.drop_duplicates("label_b", keep="first")

    best_b_keys = {(int(row.label_a), int(row.label_b)) for row in best_b.itertuples(index=False)}
    keep = [
        (int(row.label_a), int(row.label_b)) in best_b_keys
        for row in best_a.itertuples(index=False)
    ]
    return best_a.loc[keep].sort_values(["label_a", "label_b"]).reset_index(drop=True)


def _fit_linear_system(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Solve a least-squares system and return coefficients plus rank."""

    coef, *_rest = np.linalg.lstsq(X, y, rcond=None)
    rank = int(np.linalg.matrix_rank(X))
    return coef, rank


def fit_restricted_transform(
    features_a: pd.DataFrame,
    features_b: pd.DataFrame,
    seeds: pd.DataFrame,
    global_shift_zyx: np.ndarray,
    spacing: VoxelSpacing,
    params: AffineOverlapParams,
) -> RestrictedTransform:
    """Fit the restricted B-to-A transform used by the prototype."""

    global_shift_zyx = np.asarray(global_shift_zyx, dtype=float)
    fallback = lambda reason, n_seed=0, n_inlier=0: RestrictedTransform(
        z_intercept=float(global_shift_zyx[0]),
        z_scale=1.0,
        y_intercept=float(global_shift_zyx[1]),
        y_from_y=1.0,
        y_from_x=0.0,
        x_intercept=float(global_shift_zyx[2]),
        x_from_y=0.0,
        x_from_x=1.0,
        method="translation_only",
        fallback_reason=reason,
        n_seed=int(n_seed),
        n_inlier=int(n_inlier),
        residual_median_um=None,
        residual_p95_um=None,
    )

    if len(seeds) < int(params.min_affine_seeds):
        return fallback("insufficient_seeds", n_seed=len(seeds))

    if "label" not in features_a.columns or "label" not in features_b.columns:
        return fallback("missing_label_column", n_seed=len(seeds))

    seed_labels_a = seeds["label_a"].astype(int).to_numpy()
    seed_labels_b = seeds["label_b"].astype(int).to_numpy()
    try:
        coords_a = features_a.loc[seed_labels_a, ["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
        coords_b = features_b.loc[seed_labels_b, ["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    except KeyError:
        return fallback("seed_label_missing", n_seed=len(seeds))

    keep = np.ones(len(coords_a), dtype=bool)
    z_intercept = float(global_shift_zyx[0])
    z_scale = 1.0
    y_intercept = float(global_shift_zyx[1])
    y_from_y = 1.0
    y_from_x = 0.0
    x_intercept = float(global_shift_zyx[2])
    x_from_y = 0.0
    x_from_x = 1.0
    residuals_um: np.ndarray | None = None

    for _iteration in range(int(params.affine_trim_iterations)):
        zX = np.column_stack([np.ones(keep.sum(), dtype=float), coords_b[keep, 0]])
        xyX = np.column_stack([np.ones(keep.sum(), dtype=float), coords_b[keep, 1], coords_b[keep, 2]])
        z_coef, z_rank = _fit_linear_system(zX, coords_a[keep, 0])
        y_coef, y_rank = _fit_linear_system(xyX, coords_a[keep, 1])
        x_coef, x_rank = _fit_linear_system(xyX, coords_a[keep, 2])
        if z_rank < 2 or y_rank < 3 or x_rank < 3:
            return fallback("rank_deficient", n_seed=len(seeds), n_inlier=int(keep.sum()))
        if not (np.isfinite(z_coef).all() and np.isfinite(y_coef).all() and np.isfinite(x_coef).all()):
            return fallback("nonfinite_coefficients", n_seed=len(seeds), n_inlier=int(keep.sum()))

        z_intercept, z_scale = map(float, z_coef)
        y_intercept, y_from_y, y_from_x = map(float, y_coef)
        x_intercept, x_from_y, x_from_x = map(float, x_coef)

        pred = np.column_stack(
            [
                z_intercept + z_scale * coords_b[:, 0],
                y_intercept + y_from_y * coords_b[:, 1] + y_from_x * coords_b[:, 2],
                x_intercept + x_from_y * coords_b[:, 1] + x_from_x * coords_b[:, 2],
            ]
        )
        residuals_um = np.linalg.norm((coords_a - pred) * spacing.as_zyx_array(), axis=1)
        current_inlier_residuals = residuals_um[keep]
        if current_inlier_residuals.size == 0:
            return fallback("no_inliers_after_fit", n_seed=len(seeds), n_inlier=0)
        cutoff = min(float(params.affine_max_residual_um), float(np.quantile(current_inlier_residuals, params.affine_trim_quantile)))
        new_keep = residuals_um <= cutoff
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    if keep.sum() < 3:
        return fallback("insufficient_final_inliers", n_seed=len(seeds), n_inlier=int(keep.sum()))

    zX = np.column_stack([np.ones(keep.sum(), dtype=float), coords_b[keep, 0]])
    xyX = np.column_stack([np.ones(keep.sum(), dtype=float), coords_b[keep, 1], coords_b[keep, 2]])
    z_coef, z_rank = _fit_linear_system(zX, coords_a[keep, 0])
    y_coef, y_rank = _fit_linear_system(xyX, coords_a[keep, 1])
    x_coef, x_rank = _fit_linear_system(xyX, coords_a[keep, 2])
    if z_rank < 2 or y_rank < 3 or x_rank < 3:
        return fallback("rank_deficient_final", n_seed=len(seeds), n_inlier=int(keep.sum()))
    if not (np.isfinite(z_coef).all() and np.isfinite(y_coef).all() and np.isfinite(x_coef).all()):
        return fallback("nonfinite_final_coefficients", n_seed=len(seeds), n_inlier=int(keep.sum()))

    z_intercept, z_scale = map(float, z_coef)
    y_intercept, y_from_y, y_from_x = map(float, y_coef)
    x_intercept, x_from_y, x_from_x = map(float, x_coef)
    pred = np.column_stack(
        [
            z_intercept + z_scale * coords_b[:, 0],
            y_intercept + y_from_y * coords_b[:, 1] + y_from_x * coords_b[:, 2],
            x_intercept + x_from_y * coords_b[:, 1] + x_from_x * coords_b[:, 2],
        ]
    )
    residuals_um = np.linalg.norm((coords_a - pred) * spacing.as_zyx_array(), axis=1)
    residual_median_um = float(np.median(residuals_um[keep]))
    residual_p95_um = float(np.quantile(residuals_um[keep], params.affine_trim_quantile))
    return RestrictedTransform(
        z_intercept=float(z_intercept),
        z_scale=float(z_scale),
        y_intercept=float(y_intercept),
        y_from_y=float(y_from_y),
        y_from_x=float(y_from_x),
        x_intercept=float(x_intercept),
        x_from_y=float(x_from_y),
        x_from_x=float(x_from_x),
        method="restricted_affine",
        fallback_reason=None,
        n_seed=int(len(seeds)),
        n_inlier=int(keep.sum()),
        residual_median_um=residual_median_um,
        residual_p95_um=residual_p95_um,
    )


def _nearest_neighbor_data(tree: cKDTree, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Query a KD-tree with a safe ``k`` and normalize the output shapes."""

    if len(points) == 0:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=int)
    query_k = min(k, len(points))
    distances, indices = tree.query(points, k=query_k)
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if query_k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]
    return distances, indices


def _ambiguity_from_neighbors(
    nearest_distance: float,
    second_distance: float | None,
    fallback_ambiguity: float,
) -> float:
    """Compute the deterministic nearest-neighbor ambiguity ratio."""

    if second_distance is None or not np.isfinite(second_distance) or second_distance <= 0:
        return float(fallback_ambiguity)
    return float(nearest_distance / second_distance)


def _candidate_source_label(in_centroid: bool, in_overlap: bool) -> str:
    """Return the candidate source classification string."""

    if in_centroid and in_overlap:
        return "both"
    if in_centroid:
        return "mutual_centroid"
    return "mutual_overlap"


def _candidate_score(
    distance_um: float,
    ambiguity: float,
    dice: float,
    area_ratio: float,
    params: AffineOverlapParams,
) -> tuple[float, float, float]:
    """Compute the prototype score components and final score."""

    spatial_term = math.exp(-((distance_um / params.spatial_score_scale_um) ** 2))
    ambiguity_term = max(0.0, 1.0 - min(ambiguity, 1.0))
    score = (
        params.weight_spatial * spatial_term
        + params.weight_dice * dice
        + params.weight_area_ratio * area_ratio
        + params.weight_ambiguity * ambiguity_term
    )
    return float(spatial_term), float(ambiguity_term), float(score)


def _candidate_rules(dice: float, distance_um: float, area_ratio: float, ambiguity: float) -> tuple[bool, bool]:
    """Return the exact high and balanced inclusion rules."""

    high = ((dice >= 0.40) and (distance_um <= 8.0) and (area_ratio >= 0.30)) or (
        (distance_um <= 3.5) and (area_ratio >= 0.55) and (ambiguity <= 0.70)
    )
    balanced = high or (
        (dice >= 0.20) and (distance_um <= 7.0) and (area_ratio >= 0.30)
    ) or (
        (distance_um <= 5.0) and (area_ratio >= 0.40) and (ambiguity <= 0.85)
    )
    return high, balanced


def generate_candidate_pairs(
    features_a: pd.DataFrame,
    features_b: pd.DataFrame,
    transform: RestrictedTransform,
    global_shift_zyx: np.ndarray,
    overlap_table: pd.DataFrame,
    params: AffineOverlapParams,
    spacing: VoxelSpacing,
) -> pd.DataFrame:
    """Generate scored pairwise candidates from centroid and overlap evidence."""

    if len(features_a) == 0 or len(features_b) == 0:
        return pd.DataFrame(
            columns=[
                "idx_a",
                "idx_b",
                "label_a",
                "label_b",
                "candidate_source",
                "distance_um",
                "ambiguity",
                "dice",
                "iou",
                "area_ratio",
                "spatial_term",
                "ambiguity_term",
                "score",
                "high_rule",
                "balanced_rule",
            ]
        )

    spacing_array = spacing.as_zyx_array()
    coords_a = features_a[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    coords_b_raw = features_b[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    coords_b = transform.apply(coords_b_raw)
    physical_a = coords_a * spacing_array
    physical_b = coords_b * spacing_array

    tree_a = cKDTree(physical_a)
    tree_b = cKDTree(physical_b)
    distances_a, indices_a = _nearest_neighbor_data(tree_b, physical_a, k=2)
    distances_b, indices_b = _nearest_neighbor_data(tree_a, physical_b, k=2)

    labels_a = features_a.index.to_numpy(dtype=int)
    labels_b = features_b.index.to_numpy(dtype=int)
    label_to_index_a = {int(label): index for index, label in enumerate(labels_a)}
    label_to_index_b = {int(label): index for index, label in enumerate(labels_b)}
    overlap_lookup = {
        (int(row.label_a), int(row.label_b)): (float(row.dice), float(row.iou))
        for row in overlap_table.itertuples(index=False)
    }

    mutual_centroid_pairs: dict[tuple[int, int], dict[str, Any]] = {}
    fallback_ambiguity = float(params.overlap_only_ambiguity)
    for index_a in range(len(labels_a)):
        if len(labels_b) == 0:
            break
        nearest_index_b = int(indices_a[index_a, 0])
        nearest_distance_um = float(distances_a[index_a, 0])
        if nearest_distance_um > float(params.centroid_candidate_max_distance_um):
            continue
        if len(labels_a) == 0 or len(labels_b) == 0:
            continue
        if int(indices_b[nearest_index_b, 0]) != index_a:
            continue
        second_distance_a = float(distances_a[index_a, 1]) if distances_a.shape[1] > 1 else None
        second_distance_b = float(distances_b[nearest_index_b, 1]) if distances_b.shape[1] > 1 else None
        ambiguity = max(
            _ambiguity_from_neighbors(nearest_distance_um, second_distance_a, fallback_ambiguity),
            _ambiguity_from_neighbors(float(distances_b[nearest_index_b, 0]), second_distance_b, fallback_ambiguity),
        )
        label_a = int(labels_a[index_a])
        label_b = int(labels_b[nearest_index_b])
        mutual_centroid_pairs[(label_a, label_b)] = {
            "idx_a": index_a,
            "idx_b": nearest_index_b,
            "distance_um": nearest_distance_um,
            "ambiguity": ambiguity,
        }

    candidate_keys = set(mutual_centroid_pairs.keys())
    for row in overlap_table.itertuples(index=False):
        if float(row.dice) < float(params.overlap_candidate_min_dice):
            continue
        candidate_keys.add((int(row.label_a), int(row.label_b)))

    rows: list[dict[str, Any]] = []
    for label_a, label_b in sorted(candidate_keys):
        index_a = label_to_index_a[label_a]
        index_b = label_to_index_b[label_b]
        in_centroid = (label_a, label_b) in mutual_centroid_pairs
        in_overlap = (label_a, label_b) in overlap_lookup
        candidate_source = _candidate_source_label(in_centroid, in_overlap)
        if in_centroid:
            distance_um = float(mutual_centroid_pairs[(label_a, label_b)]["distance_um"])
            ambiguity = float(mutual_centroid_pairs[(label_a, label_b)]["ambiguity"])
        else:
            distance_um = float(np.linalg.norm((physical_a[index_a] - physical_b[index_b])))
            ambiguity = float(params.overlap_only_ambiguity)
        dice, iou = overlap_lookup.get((label_a, label_b), (0.0, 0.0))
        area_a = float(features_a.at[label_a, "area_voxels"])
        area_b = float(features_b.at[label_b, "area_voxels"])
        area_ratio = float(min(area_a, area_b) / max(area_a, area_b))
        spatial_term, ambiguity_term, score = _candidate_score(distance_um, ambiguity, dice, area_ratio, params)
        high_rule, balanced_rule = _candidate_rules(dice, distance_um, area_ratio, ambiguity)
        rows.append(
            {
                "idx_a": int(index_a),
                "idx_b": int(index_b),
                "label_a": int(label_a),
                "label_b": int(label_b),
                "candidate_source": candidate_source,
                "distance_um": distance_um,
                "ambiguity": float(ambiguity),
                "dice": float(dice),
                "iou": float(iou),
                "area_ratio": area_ratio,
                "spatial_term": spatial_term,
                "ambiguity_term": ambiguity_term,
                "score": score,
                "high_rule": bool(high_rule),
                "balanced_rule": bool(balanced_rule),
            }
        )

    columns = [
        "idx_a",
        "idx_b",
        "label_a",
        "label_b",
        "candidate_source",
        "distance_um",
        "ambiguity",
        "dice",
        "iou",
        "area_ratio",
        "spatial_term",
        "ambiguity_term",
        "score",
        "high_rule",
        "balanced_rule",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows)
    table = table.sort_values(["score", "dice", "distance_um", "label_a", "label_b"], ascending=[False, False, True, True, True]).reset_index(drop=True)
    return table.loc[:, columns]


def greedy_one_to_one(candidate_table: pd.DataFrame, rule_column: str) -> pd.DataFrame:
    """Select one-to-one matches with deterministic greedy assignment."""

    if candidate_table is None or len(candidate_table) == 0:
        columns = list(candidate_table.columns) if candidate_table is not None else []
        return pd.DataFrame(columns=columns)

    if rule_column not in candidate_table.columns:
        raise ValueError(f"candidate_table is missing rule column {rule_column!r}.")

    table = candidate_table.loc[candidate_table[rule_column].astype(bool)].copy()
    if table.empty:
        return table.iloc[0:0].copy()

    table = table.sort_values(
        ["score", "dice", "distance_um", "label_a", "label_b"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    accepted_rows: list[dict[str, Any]] = []
    for row in table.itertuples(index=False):
        label_a = int(row.label_a)
        label_b = int(row.label_b)
        if label_a in used_a or label_b in used_b:
            continue
        used_a.add(label_a)
        used_b.add(label_b)
        accepted_rows.append(row._asdict())

    if not accepted_rows:
        return table.iloc[0:0].copy()
    accepted = pd.DataFrame(accepted_rows)
    accepted["assignment_policy"] = str(rule_column).removesuffix("_rule")
    return accepted.reset_index(drop=True)


def match_pair(
    session_a: str,
    session_b: str,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    params: AffineOverlapParams | None = None,
    spacing: VoxelSpacing | None = None,
    features_a: pd.DataFrame | None = None,
    features_b: pd.DataFrame | None = None,
    pair_gap: int | None = None,
) -> PairMatchResult:
    """Match two sessions and return the full pairwise result bundle."""

    params = params or AffineOverlapParams()
    spacing = spacing or VoxelSpacing()
    mask_a = _validate_mask_array(mask_a)
    mask_b = _validate_mask_array(mask_b)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Mask pairs must share the same shape.")

    if features_a is None:
        features_a = extract_roi_features(mask_a, session_id=session_a, spacing=spacing)
    if features_b is None:
        features_b = extract_roi_features(mask_b, session_id=session_b, spacing=spacing)

    shift_zyx, shift_summary = estimate_global_shift(mask_a, mask_b, params=params)
    area_a = features_a["area_voxels"]
    area_b = features_b["area_voxels"]
    overlap_table = build_sparse_overlap_table(mask_a, mask_b, shift_zyx, area_a, area_b)
    mutual_overlap = select_mutual_overlap_pairs(overlap_table)
    seeds = mutual_overlap.loc[mutual_overlap["dice"] >= float(params.seed_min_dice)].copy()
    transform = fit_restricted_transform(features_a, features_b, seeds, shift_zyx, spacing, params)

    candidates = generate_candidate_pairs(
        features_a=features_a,
        features_b=features_b,
        transform=transform,
        global_shift_zyx=shift_zyx,
        overlap_table=overlap_table,
        params=params,
        spacing=spacing,
    )
    high_matches = greedy_one_to_one(candidates, "high_rule")
    balanced_matches = greedy_one_to_one(candidates, "balanced_rule")

    summary = {
        "day_a": session_a,
        "day_b": session_b,
        "pair_gap": int(pair_gap) if pair_gap is not None else None,
        "shift_z": float(shift_zyx[0]),
        "shift_y": float(shift_zyx[1]),
        "shift_x": float(shift_zyx[2]),
        "n_a": int(len(features_a)),
        "n_b": int(len(features_b)),
        "n_overlap_pairs": int(len(overlap_table)),
        "n_mutual_overlap": int(len(mutual_overlap)),
        "n_seed": int(len(seeds)),
        "n_transform_fit": int(transform.n_inlier),
        "n_high": int(len(high_matches)),
        "n_balanced": int(len(balanced_matches)),
        "high_frac_smaller": float(len(high_matches) / min(len(features_a), len(features_b))),
        "balanced_frac_smaller": float(len(balanced_matches) / min(len(features_a), len(features_b))),
        "refinement_method": "affine" if transform.method == "restricted_affine" else "translation_only",
        "transform_method": transform.method,
        "transform_fallback_reason": transform.fallback_reason,
        "transform_residual_median_um": transform.residual_median_um,
        "transform_residual_p95_um": transform.residual_p95_um,
        "xy_fallback": bool(shift_summary.get("xy_fallback", False)),
        "z_fallback": bool(shift_summary.get("z_fallback", False)),
        "xy_fallback_reason": shift_summary.get("xy_fallback_reason"),
        "z_fallback_reason": shift_summary.get("z_fallback_reason"),
    }
    return PairMatchResult(
        candidates=candidates,
        high_matches=high_matches,
        balanced_matches=balanced_matches,
        summary=summary,
        transform=transform,
    )


def candidate_edge_keys(candidate_table: pd.DataFrame) -> set[tuple[int, int]]:
    """Return a set of candidate label pairs."""

    if candidate_table is None or candidate_table.empty:
        return set()
    return {(int(row.label_a), int(row.label_b)) for row in candidate_table.itertuples(index=False)}


def accepted_edge_keys(edge_table: pd.DataFrame) -> set[tuple[int, int]]:
    """Return a set of accepted edge label pairs."""

    if edge_table is None or edge_table.empty:
        return set()
    return {(int(row.label_a), int(row.label_b)) for row in edge_table.itertuples(index=False)}

