"""Opt-in image registration experiments; production mask-only matching is unchanged.

Transforms map native B coordinates into A. The smooth field is an inverse
displacement in the A frame, sampled on ``flow_step`` voxel spacing. Intensities
are used only to estimate geometry, never to change the original measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage.registration import optical_flow_tvl1, phase_cross_correlation

from affine_overlap_matcher import (
    AffineOverlapParams, VoxelSpacing, build_sparse_overlap_table,
    generate_candidate_pairs, greedy_one_to_one,
)


@dataclass
class ImageTransform:
    matrix: np.ndarray
    offset: np.ndarray
    flow: np.ndarray | None = None
    flow_step: np.ndarray | None = None

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float)
        self.offset = np.asarray(self.offset, dtype=float)
        if (self.matrix.shape != (3, 3) or self.offset.shape != (3,)
                or not np.isfinite(self.matrix).all() or not np.isfinite(self.offset).all()
                or np.linalg.det(self.matrix) <= 0):
            raise ValueError("Expected a finite, orientation-preserving 3D affine transform.")
        if self.flow is not None:
            self.flow = np.asarray(self.flow, dtype=np.float32)
            self.flow_step = np.asarray(self.flow_step, dtype=float)
            if (self.flow.ndim != 4 or self.flow.shape[0] != 3
                    or min(self.flow.shape[1:]) < 2 or not np.isfinite(self.flow).all()
                    or self.flow_step.shape != (3,) or not np.isfinite(self.flow_step).all()
                    or np.any(self.flow_step <= 0)):
                raise ValueError("Invalid displacement field or voxel spacing.")

    def displacement(self, points):
        points = np.asarray(points, dtype=float)
        if self.flow is None:
            return np.zeros_like(points)
        return np.column_stack([
            ndi.map_coordinates(v, (points / self.flow_step).T, order=1,
                                mode="nearest", prefilter=False)
            for v in self.flow
        ]) * self.flow_step

    def inverse(self, points):
        """A-to-native-B sampling coordinates; accepts an N x 3 array."""
        points = np.asarray(points, dtype=float)
        return (points + self.displacement(points) - self.offset) @ np.linalg.inv(self.matrix).T

    def apply(self, points):
        """B-to-A centroid coordinates; invert the smooth field by fixed point."""
        affine = np.asarray(points, dtype=float) @ self.matrix.T + self.offset
        result = affine.copy()
        for _ in range(20 if self.flow is not None else 0):
            updated = affine - self.displacement(result)
            if np.max(np.abs(updated - result), initial=0) < 1e-5:
                result = updated
                break
            result = updated
        return result


def restricted_transform(row):
    return ImageTransform(
        [[row.z_scale, 0, 0], [0, row.y_from_y, row.y_from_x],
         [0, row.x_from_y, row.x_from_x]],
        [row.z_intercept, row.y_intercept, row.x_intercept],
    )


def warp_volume(volume, transform, shape, *, step=(1, 1, 1), order=1):
    """Resample B on the A grid. ``step`` describes both input/output grids."""
    step = np.asarray(step, dtype=float)
    if transform.flow is None:
        matrix = transform.matrix * step[None, :] / step[:, None]
        inverse = np.linalg.inv(matrix)
        return ndi.affine_transform(volume, inverse, -inverse @ (transform.offset / step),
                                    output_shape=shape, order=order, prefilter=False)
    # Plane chunks bound memory for 41 x 768 x 1536 native stacks.
    result = np.empty(shape, dtype=volume.dtype)
    yx = np.indices(shape[1:], dtype=np.float32).reshape(2, -1).T
    for z in range(shape[0]):
        points = np.column_stack([np.full(len(yx), z), yx]) * step
        coords = transform.inverse(points) / step
        result[z] = ndi.map_coordinates(volume, coords.T, order=order,
                                        prefilter=False).reshape(shape[1:])
    return result


def correlation(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-12 else float("nan")


def prepare_images(red_a, red_b, spacing):
    """Anti-alias to ~2.8 um XY and subtract a 11 um XY background."""
    if (red_a.shape != red_b.shape or red_a.ndim != 3 or min(red_a.shape) < 9
            or not np.isfinite(red_a).all() or not np.isfinite(red_b).all()):
        raise ValueError("Red volumes must have matching, finite 3D shapes (all axes >=9).")
    spacing = spacing.as_zyx_array()
    step = np.maximum(1, np.rint([1, 2.8 / spacing[1], 2.8 / spacing[2]])).astype(int)
    images = []
    for red in (red_a, red_b):
        small = ndi.uniform_filter(red.astype(np.float32), size=tuple(step))[
            ::step[0], ::step[1], ::step[2]]
        if min(small.shape) < 9:
            raise ValueError("Downsampled red volumes must have at least 9 samples on each axis.")
        highpass = small - ndi.gaussian_filter(small, [0, 11 / (spacing[1]*step[1]),
                                                       11 / (spacing[2]*step[2])])
        if np.std(highpass) < 1e-6:
            raise ValueError("Red volume has insufficient structural contrast.")
        images.append(highpass)
    return images[0], images[1], step


def fit_image_affine(a, b, initial, step):
    """Fit a full affine to red anatomy; no ROI correspondence enters the fit."""
    warped = warp_volume(b, initial, a.shape, step=step)
    shape = np.array(a.shape); center = (shape - 1) / 2
    samples = []; shifts = []
    for y in range(0, shape[1]-31, 48):
        for x in range(0, shape[2]-31, 64):
            sl = np.s_[3:-3, y:min(y+48, shape[1]), x:min(x+64, shape[2])]
            aa, bb = a[sl], warped[sl]
            if min(np.std(aa), np.std(bb)) < 1e-6:
                continue
            shift, _, _ = phase_cross_correlation(aa, bb, normalization=None, upsample_factor=2)
            if np.max(np.abs(shift)) < 12:
                samples.append([1, (y+(aa.shape[1]-1)/2-center[1])/center[1],
                                (x+(aa.shape[2]-1)/2-center[2])/center[2]])
                shifts.append(-shift)
    parameters = np.zeros((3, 4))
    if len(samples) >= 6:
        parameters[:, [0, 2, 3]] = np.linalg.lstsq(samples, shifts, rcond=None)[0].T
    rng = np.random.default_rng(42)
    points = np.column_stack([rng.uniform(3, n-4, 70000) for n in shape])
    design = np.column_stack([np.ones(len(points)), (points-center)/center])
    target = ndi.map_coordinates(a, points.T, order=1, prefilter=False)

    def objective(p):
        coords = points + design @ p.reshape(3, 4).T
        return -correlation(target, ndi.map_coordinates(warped, coords.T, order=1, prefilter=False))

    result = minimize(objective, parameters.ravel(), method="L-BFGS-B",
                      bounds=[(-15, 15)]*12,
                      options={"maxiter":150, "ftol":1e-9, "gtol":1e-5, "eps":.005, "maxls":30})
    p = result.x.reshape(3, 4)
    inverse = np.eye(3) + p[:, 1:] / center
    correction = np.linalg.inv(inverse)
    offset = -correction @ (p[:, 0] - p[:, 1:] @ np.ones(3))
    correction = correction * step[:, None] / step[None, :]
    transform = ImageTransform(correction @ initial.matrix,
                               correction @ initial.offset + offset * step)
    return transform, {"optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                       "training_ncc": -float(result.fun), "evaluations": int(result.nfev)}


def fit_smooth_field(a, b, affine, step, spacing, *, smoothing_um=15.0):
    """Smooth the geometric displacement field in physical units, not the data."""
    if not np.isfinite(smoothing_um) or smoothing_um < 5:
        raise ValueError("smoothing_um must be finite and >=5 um.")
    extra = np.maximum(1, np.rint(5.5 / (step * spacing.as_zyx_array()))).astype(int)
    slices = tuple(slice(None, None, int(x)) for x in extra)
    aa = ndi.uniform_filter(a, size=tuple(extra))[slices]
    bb = ndi.uniform_filter(warp_volume(b, affine, a.shape, step=step), size=tuple(extra))[slices]
    scale = lambda v: np.clip(v / max(np.percentile(np.abs(v), 99.5), 1e-6), -1, 2).astype(np.float32)
    flow = optical_flow_tvl1(scale(aa), scale(bb), attachment=8, tightness=.3,
                            num_warp=7, num_iter=15, prefilter=True)
    flow_step = step * extra
    sigma = smoothing_um / (flow_step * spacing.as_zyx_array())
    flow = np.array([ndi.gaussian_filter(v, sigma) for v in flow])
    return replace(affine, flow=flow, flow_step=flow_step)


def image_quality(a, b, transform, step, spacing):
    """Fixed, held-out pixels (different RNG from fitting) plus deformation QC."""
    rng = np.random.default_rng(917)
    q = np.column_stack([rng.uniform(3, n-4, 30000) for n in a.shape])
    qb = transform.inverse(q*step)/step
    inside = np.all((qb >= 0) & (qb <= np.array(b.shape)-1), axis=1)
    reference = ndi.map_coordinates(a, q[inside].T, order=1, prefilter=False)
    moving = ndi.map_coordinates(b, qb[inside].T, order=1, prefilter=False)
    physical = transform.matrix * spacing.as_zyx_array()[:, None] / spacing.as_zyx_array()[None, :]
    singular = np.linalg.svd(physical, compute_uv=False)
    quality = {"heldout_ncc": correlation(reference, moving), "sample_overlap": float(inside.mean()),
               "affine_det": float(np.linalg.det(transform.matrix)),
               "affine_singular_min": float(singular.min()), "affine_singular_max": float(singular.max()),
               "jacobian_min": 1., "jacobian_p01": 1., "jacobian_p99": 1., "displacement_p99_um": 0.}
    if transform.flow is not None:
        # det(I + grad(u)) is the inverse local warp determinant, excluding affine.
        jac = np.empty((*transform.flow.shape[1:], 3, 3), dtype=np.float32)
        for i, field in enumerate(transform.flow):
            for j, derivative in enumerate(np.gradient(field)):
                jac[..., i, j] = derivative + (i == j)
        determinants = np.linalg.det(jac)
        lengths = np.linalg.norm(transform.flow * (transform.flow_step * spacing.as_zyx_array())[:,None,None,None], axis=0)
        quality.update(jacobian_min=float(determinants.min()), jacobian_p01=float(np.quantile(determinants,.01)),
                       jacobian_p99=float(np.quantile(determinants,.99)), displacement_p99_um=float(np.quantile(lengths,.99)))
    return quality


def match_transformed_masks(mask_a, mask_b, features_a, features_b, transform, spacing):
    """Reuse exact production gates and assignment; recompute overlap on A grid."""
    warped = warp_volume(mask_b, transform, mask_a.shape, order=0)
    labels, counts = np.unique(warped, return_counts=True)
    warped_areas = pd.Series(counts, index=labels)
    overlap = build_sparse_overlap_table(mask_a, warped, np.zeros(3),
                                         features_a.area_voxels, warped_areas)
    candidates = generate_candidate_pairs(features_a, features_b, transform, np.zeros(3),
                                         overlap, AffineOverlapParams(), spacing)
    matches = greedy_one_to_one(candidates, "balanced_rule")
    points = features_b[["centroid_z", "centroid_y", "centroid_x"]].to_numpy()
    error = np.linalg.norm((transform.inverse(transform.apply(points))-points)*spacing.as_zyx_array(), axis=1)
    return candidates, matches, float(error.max(initial=0))


def correspondence_changes(baseline, candidate):
    original = set(zip(baseline.label_a, baseline.label_b))
    new = set(zip(candidate.label_a, candidate.label_b))
    # Strong existing matches are evidence, not ground truth. Audit conflicts;
    # never silently lock/union old and new assignments to inflate retention.
    anchors = baseline[(baseline.dice >= .65) & (baseline.distance_um <= 3)
                       & (baseline.area_ratio >= .55) & (baseline.ambiguity <= .7)]
    anchor_pairs = set(zip(anchors.label_a, anchors.label_b))
    ca = dict(new); cb = {b:a for a,b in new}
    conflicts = sum((a in ca and ca[a] != b) or (b in cb and cb[b] != a) for a,b in anchor_pairs)
    return {"retained":len(original & new), "added":len(new-original), "lost":len(original-new),
            "n_anchors":len(anchor_pairs), "anchors_retained":len(anchor_pairs & new),
            "anchor_conflicts":conflicts}


def guard_reasons(row, baseline):
    """Conservative experimental fallback, not a guarantee of identity accuracy."""
    reasons = []
    if not np.isfinite(row["heldout_ncc"]) or row["heldout_ncc"] < baseline["heldout_ncc"]-.005:
        reasons.append("image_similarity_decreased")
    if row["sample_overlap"] < baseline["sample_overlap"]-.02:
        reasons.append("overlap_decreased")
    if not (.8 <= row["affine_singular_min"] <= row["affine_singular_max"] <= 1.25):
        reasons.append("affine_distortion")
    if row["jacobian_min"] <= 0 or row["jacobian_p01"] < .5 or row["jacobian_p99"] > 2:
        reasons.append("local_distortion")
    if row["displacement_p99_um"] > 25 or row["inverse_error_max_um"] > .25:
        reasons.append("displacement_or_inverse_error")
    if row["n_matches"] < baseline["n_matches"]:
        reasons.append("match_count_decreased")
    if row["anchor_conflicts"] or row["anchors_retained"] < .99*row["n_anchors"]:
        reasons.append("confident_identity_changed_or_lost")
    return reasons
