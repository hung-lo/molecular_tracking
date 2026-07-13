from __future__ import annotations

import numpy as np
import pandas as pd

from affine_overlap_matcher import (
    AffineOverlapParams,
    VoxelSpacing,
    build_sparse_overlap_table,
    extract_roi_features,
    fit_restricted_transform,
    generate_candidate_pairs,
    greedy_one_to_one,
    match_pair,
    select_mutual_overlap_pairs,
)


def _make_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["a", "a"],
            "label": [1, 2],
            "area_voxels": [1.0, 1.0],
            "volume_um3": [1.0, 1.0],
            "centroid_z": [0.0, 1.0],
            "centroid_y": [0.0, 1.0],
            "centroid_x": [0.0, 1.0],
            "centroid_z_um": [0.0, 1.0],
            "centroid_y_um": [0.0, 1.0],
            "centroid_x_um": [0.0, 1.0],
        }
    ).set_index("label", drop=False)


def test_extract_roi_features_reports_exclusive_bboxes_and_noncontiguous_labels() -> None:
    mask = np.zeros((2, 3, 4), dtype=np.uint16)
    mask[0, 0, 0] = 5
    mask[1, 2, 3] = 9

    features = extract_roi_features(mask, session_id="day0", spacing=VoxelSpacing())

    assert features.index.tolist() == [5, 9]
    assert features.loc[5, "bbox_z1"] == 1
    assert features.loc[5, "bbox_y1"] == 1
    assert features.loc[5, "bbox_x1"] == 1
    assert features.loc[9, "bbox_z0"] == 1
    assert features.loc[9, "bbox_y0"] == 2
    assert features.loc[9, "bbox_x0"] == 3


def test_sparse_overlap_and_mutual_pairs_detects_identical_labels() -> None:
    mask_a = np.zeros((2, 2, 2), dtype=np.uint16)
    mask_b = np.zeros((2, 2, 2), dtype=np.uint16)
    mask_a[0, 0, 0] = 1
    mask_a[1, 1, 1] = 2
    mask_b[0, 0, 0] = 1
    mask_b[1, 1, 1] = 2

    features_a = extract_roi_features(mask_a, session_id="a", spacing=VoxelSpacing())
    features_b = extract_roi_features(mask_b, session_id="b", spacing=VoxelSpacing())
    overlap = build_sparse_overlap_table(mask_a, mask_b, np.zeros(3), features_a["area_voxels"], features_b["area_voxels"])
    mutual = select_mutual_overlap_pairs(overlap)

    assert overlap["dice"].tolist() == [1.0, 1.0]
    assert overlap["iou"].tolist() == [1.0, 1.0]
    assert mutual[["label_a", "label_b"]].to_dict(orient="records") == [{"label_a": 1, "label_b": 1}, {"label_a": 2, "label_b": 2}]


def test_fit_restricted_transform_falls_back_with_too_few_seeds() -> None:
    features_a = _make_features()
    features_b = _make_features().rename(columns={"session_id": "session_id_b"})
    seeds = pd.DataFrame({"label_a": [1], "label_b": [1]})

    transform = fit_restricted_transform(
        features_a=features_a,
        features_b=features_a,
        seeds=seeds,
        global_shift_zyx=np.array([1.0, 2.0, 3.0]),
        spacing=VoxelSpacing(),
        params=AffineOverlapParams(min_affine_seeds=50),
    )

    assert transform.method == "translation_only"
    assert transform.fallback_reason == "insufficient_seeds"
    assert transform.apply(np.array([0.0, 0.0, 0.0])).shape == (3,)


def test_generate_candidate_pairs_and_greedy_assignment_use_thresholds() -> None:
    features_a = _make_features()
    features_b = _make_features().rename(columns={"session_id": "session_id_b"})
    overlap = pd.DataFrame(
        {
            "label_a": [1, 2],
            "label_b": [1, 2],
            "intersection_voxels": [1, 1],
            "dice": [1.0, 1.0],
            "iou": [1.0, 1.0],
        }
    )
    transform = fit_restricted_transform(
        features_a=features_a,
        features_b=features_a,
        seeds=pd.DataFrame({"label_a": [1], "label_b": [1]}),
        global_shift_zyx=np.zeros(3),
        spacing=VoxelSpacing(),
        params=AffineOverlapParams(min_affine_seeds=50),
    )

    candidates = generate_candidate_pairs(
        features_a=features_a,
        features_b=features_a,
        transform=transform,
        global_shift_zyx=np.zeros(3),
        overlap_table=overlap,
        params=AffineOverlapParams(min_affine_seeds=50),
        spacing=VoxelSpacing(),
    )
    high = greedy_one_to_one(candidates, "high_rule")
    balanced = greedy_one_to_one(candidates, "balanced_rule")

    assert set(candidates["candidate_source"]) == {"both"}
    assert len(high) >= 1
    assert len(balanced) >= 1


def test_match_pair_small_masks_returns_summary() -> None:
    mask_a = np.zeros((2, 2, 2), dtype=np.uint16)
    mask_b = np.zeros((2, 2, 2), dtype=np.uint16)
    mask_a[0, 0, 0] = 1
    mask_a[0, 1, 1] = 2
    mask_b[0, 0, 0] = 1
    mask_b[0, 1, 1] = 2

    result = match_pair("a", "b", mask_a, mask_b)

    assert result.summary["n_high"] >= 1
    assert result.summary["n_balanced"] >= 1
    assert result.summary["transform_method"] == "translation_only"
