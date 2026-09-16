from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from postprocessing.local_segmentation_rescue import (
    RunContext,
    TransformGraph,
    BackendUnavailable,
    _preflight_backend,
    _predict_target,
    _find_real_cases,
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


def test_hidden_target_volume_cannot_change_candidate_ranking():
    candidates = [
        {"candidate_id": 1, "centroid_xyz": [5, 5, 5], "volume_um3": 10, "touches_crop_edge": False},
        {"candidate_id": 2, "centroid_xyz": [6, 5, 5], "volume_um3": 1000, "touches_crop_edge": False},
    ]
    first = [row["candidate_id"] for row in rank_candidates(candidates, [5.2, 5, 5], (1, 1, 1), 10)]
    second = [row["candidate_id"] for row in rank_candidates(candidates, [5.2, 5, 5], (1, 1, 1), 10000)]
    assert first == second


def test_prediction_context_is_explicit_and_one_sided_ignores_future():
    sessions = pd.DataFrame([{"session_id": f"s{i}", "session_index": i} for i in range(3)])
    features = pd.DataFrame([
        {"session_id": "s0", "label": 1, "centroid_z": 0, "centroid_y": 0, "centroid_x": 0},
        {"session_id": "s2", "label": 1, "centroid_z": 0, "centroid_y": 0, "centroid_x": 100},
    ])
    transforms = pd.DataFrame([
        {"day_a": "s0", "day_b": "s1"},
        {"day_a": "s1", "day_b": "s2"},
    ])
    context = RunContext(__import__("pathlib").Path("/tmp/run"), __import__("pathlib").Path("/tmp/matching"), features, pd.DataFrame(), sessions, transforms, (1, 1, 1), "test", {}, "", "", "", {})
    lookup = {(str(row.session_id), int(row.label)): row for _, row in features.iterrows()}
    graph = TransformGraph(transforms)
    observations = {0: ("s0", 1), 2: ("s2", 1)}
    one = _predict_target(context, observations, 1, lookup, graph, benchmark_context="endpoint_one_sided")
    two = _predict_target(context, observations, 1, lookup, graph, benchmark_context="internal_gap_two_sided")
    assert one["predicted_xyz"] == [0.0, 0.0, 0.0]
    assert two["predicted_xyz"] == [50.0, 0.0, 0.0]


def test_real_evaluator_schema_derives_immediate_next_target(tmp_path: Path):
    sessions = pd.DataFrame([{"session_id": f"s{i}", "session_index": i} for i in range(3)])
    context = RunContext(Path("/tmp/run"), Path("/tmp/matching"), pd.DataFrame(), pd.DataFrame(), sessions, pd.DataFrame(), (1, 1, 1), "test", {}, "", "", "", {})
    artifact = tmp_path / "endpoint_classification.csv"
    pd.DataFrame([{"endpoint_id": "e1", "track_uid": "t1", "end_session_index": 1, "end_session_id": "s1", "end_label": 7, "classification": "no_mask_near_prediction"}]).to_csv(artifact, index=False)
    cases = _find_real_cases(context, None, 0, artifact)
    assert len(cases) == 1
    assert cases[0]["source_session_index"] == 1
    assert cases[0]["target_session_index"] == 2
    assert cases[0]["target_session"] == "s2"
    assert cases[0]["classification_source_sha256"]


def test_cellpose_preflight_fails_cleanly_when_backend_is_missing():
    try:
        _preflight_backend("cellpose_sam", device="cuda")
    except BackendUnavailable:
        return
    # A configured Cellpose environment is also valid; the test only asserts
    # that preflight does not silently return the threshold backend.
    assert _preflight_backend("cellpose_sam", device="cpu")["pretrained_model"] == "cpsam_v2"


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
