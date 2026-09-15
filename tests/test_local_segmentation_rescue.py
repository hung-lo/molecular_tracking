from __future__ import annotations

import numpy as np
import pandas as pd

from postprocessing.local_segmentation_rescue import (
    TransformGraph,
    compute_crop_bounds,
    dice_iou,
    rank_candidates,
    segment_local_threshold,
)


def test_crop_bounds_xyz_to_zyx_and_clipping_are_deterministic():
    bounds = compute_crop_bounds((1.2, 5.0, 2.0), (10, 20, 30), (5, 8, 10))
    assert bounds.start_zyx == (0, 1, 0)
    assert bounds.stop_zyx == (5, 9, 10)
    assert bounds.shape_zyx == (5, 8, 10)
    assert bounds.edge_clipped
    assert bounds.as_dict()["crop_shape_zyx"] == [5, 8, 10]


def test_dice_iou_and_threshold_segmenter_are_deterministic():
    a = np.zeros((3, 4, 5), bool)
    b = np.zeros_like(a)
    a[1, 1:3, 1:3] = True
    b[1, 2:4, 1:3] = True
    assert dice_iou(a, b) == (0.5, 1 / 3)
    image = np.full((5, 12, 12), 10.0, dtype=float)
    image[2, 4:7, 4:7] = 100
    image[2, 4:7, 8:11] = 90
    first = segment_local_threshold(image, threshold_percentile=80, min_voxels=3)
    second = segment_local_threshold(image, threshold_percentile=80, min_voxels=3)
    assert first[2] == second[2]
    assert first[1].tolist() == second[1].tolist()
    assert len(first[0]) == 2


def test_candidate_ranking_ignores_biological_state_columns():
    candidates = [
        {"candidate_id": 2, "centroid_xyz": [5, 5, 5], "volume_um3": 100, "touches_crop_edge": False, "eclipse_z": -10},
        {"candidate_id": 1, "centroid_xyz": [6, 5, 5], "volume_um3": 100, "touches_crop_edge": False, "eclipse_z": 10},
    ]
    ranked = rank_candidates(candidates, [5, 5, 5], (1, 1, 1))
    assert [item["candidate_id"] for item in ranked] == [2, 1]
    candidates[0]["eclipse_z"] = 1000
    assert rank_candidates(candidates, [5, 5, 5], (1, 1, 1))[0]["candidate_id"] == 2


def test_transform_graph_direct_and_composed_provenance():
    transforms = pd.DataFrame([
        {"day_a": "s0", "day_b": "s1", "z_intercept": 1, "z_scale": 1, "y_intercept": 2, "y_from_y": 1, "y_from_x": 0, "x_intercept": 3, "x_from_y": 0, "x_from_x": 1},
        {"day_a": "s1", "day_b": "s2", "z_intercept": 1, "z_scale": 1, "y_intercept": 2, "y_from_y": 1, "y_from_x": 0, "x_intercept": 3, "x_from_y": 0, "x_from_x": 1},
    ])
    graph = TransformGraph(transforms)
    direct = graph.project("s0", "s1", [10, 20, 30])
    composed = graph.project("s0", "s2", [10, 20, 30])
    assert direct is not None and direct[1] == "direct"
    assert composed is not None and composed[1] == "composed"
    assert np.allclose(direct[0], [9, 18, 27])
    assert np.allclose(composed[0], [8, 16, 24])
