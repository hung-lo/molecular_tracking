from __future__ import annotations

import numpy as np
import pytest

from affine_overlap_matcher import (
    RestrictedTransform,
    VoxelSpacing,
    build_sparse_overlap_table,
    extract_roi_features,
)
from roi_matcher import MatchParams, match_roi_masks
from run_registered_roi_pipeline import compute_fixed_crop_bounds


def test_crop_bounds_support_independent_rectangular_dimensions() -> None:
    bounds = compute_fixed_crop_bounds(
        image_shape_yx=(512, 768), y_center=256, x_center=384,
        crop_height_px=384, crop_width_px=640,
    )
    y0, y1, x0, x1 = bounds
    assert (y1 - y0, x1 - x0) == (384, 640)
    assert (y0, y1) == (64, 448)
    assert (x0, x1) == (64, 704)


def test_crop_bounds_reject_oversized_rectangular_request() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        compute_fixed_crop_bounds(
            image_shape_yx=(512, 768), y_center=256, x_center=384,
            crop_height_px=513, crop_width_px=640,
        )


def test_roi_matcher_handles_rectangular_volume() -> None:
    shape = (7, 93, 157)
    mask_a = np.zeros(shape, dtype=np.int32)
    mask_b = np.zeros(shape, dtype=np.int32)
    mask_a[2:6, 20:40, 70:100] = 1
    mask_b[2:6, 20:40, 70:100] = 1

    tracks, pair_tables, _qc = match_roi_masks(
        [mask_a, mask_b], ["a", "b"],
        params=MatchParams(use_translation=False, patch_radius=4, edge_margin=5),
    )
    assert tracks.loc[0, "a_roi"] == 1
    assert tracks.loc[0, "b_roi"] == 1
    assert pair_tables[("a", "b")].iloc[0][["label_a", "label_b"]].tolist() == [1, 1]


def test_affine_overlap_preserves_rectangular_xy_axes() -> None:
    shape = (6, 80, 140)
    mask_a = np.zeros(shape, dtype=np.int32)
    mask_b = np.zeros(shape, dtype=np.int32)
    mask_a[1:5, 11:29, 83:121] = 1
    mask_b[1:5, 11:29, 83:121] = 1

    features = extract_roi_features(mask_a, "a", VoxelSpacing())
    overlap = build_sparse_overlap_table(
        mask_a, mask_b, np.zeros(3), features["area_voxels"], features["area_voxels"]
    )
    assert overlap.iloc[0][["label_a", "label_b", "dice"]].tolist() == [1, 1, 1.0]

    transform = RestrictedTransform(
        z_intercept=0.0, z_scale=1.0, y_intercept=3.0, y_from_y=1.0,
        y_from_x=0.0, x_intercept=-5.0, x_from_y=0.0, x_from_x=1.0,
        method="test", fallback_reason=None, n_seed=0, n_inlier=0,
        residual_median_um=None, residual_p95_um=None,
    )
    assert np.allclose(transform.apply(np.array([2.0, 20.0, 100.0])), [2.0, 23.0, 95.0])
