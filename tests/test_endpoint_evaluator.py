from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from affine_overlap_matcher import RestrictedTransform, VoxelSpacing
from endpoint_evaluator import (
    SyntheticTrustConfig,
    apply_transform_b_to_a,
    build_manual_review_manifest,
    build_runtime_summary,
    build_state_dropout_stratification,
    build_synthetic_gap_benchmark,
    classify_endpoint_events,
    compose_transforms,
    detect_endpoint_events,
    invert_restricted_transform,
    project_centroid_forward,
    search_endpoint_candidates,
)


def _transform(**updates: float) -> RestrictedTransform:
    values = {
        "z_intercept": 1.0, "z_scale": 2.0, "y_intercept": 3.0, "y_from_y": 2.0,
        "y_from_x": 1.0, "x_intercept": -1.0, "x_from_y": 1.0, "x_from_x": 3.0,
        "method": "test", "fallback_reason": None, "n_seed": 0, "n_inlier": 0,
        "residual_median_um": None, "residual_p95_um": None,
    }
    values.update(updates)
    return RestrictedTransform(**values)


def _sessions(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "session_index": range(n),
        "session_id": [f"s{i}" for i in range(n)],
        "acquisition_date": [f"2026-01-{i + 1:02d}" for i in range(n)],
        "mask_path": [""] * n,
    })


def _identity_transforms(n: int = 4, include_gap2: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n - 1):
        rows.append({
            "day_a": f"s{i}", "day_b": f"s{i+1}", "pair_gap": 1,
            "z_intercept": 0.0, "z_scale": 1.0,
            "y_intercept": 0.0, "y_from_y": 1.0, "y_from_x": 0.0,
            "x_intercept": 0.0, "x_from_y": 0.0, "x_from_x": 1.0,
            "method": "restricted_affine", "fallback_reason": "",
        })
    if include_gap2:
        for i in range(n - 2):
            rows.append({
                "day_a": f"s{i}", "day_b": f"s{i+2}", "pair_gap": 2,
                "z_intercept": 0.0, "z_scale": 1.0,
                "y_intercept": 0.0, "y_from_y": 1.0, "y_from_x": 0.0,
                "x_intercept": 0.0, "x_from_y": 0.0, "x_from_x": 1.0,
                "method": "restricted_affine", "fallback_reason": "",
            })
    return pd.DataFrame(rows)


def _features(rows: list[tuple[int, int, float, float, bool]]) -> pd.DataFrame:
    output = []
    for session_index, label, y, x, edge in rows:
        output.append({
            "session_index": session_index,
            "session_id": f"s{session_index}",
            "label": label,
            "centroid_z": 1.0, "centroid_y": y, "centroid_x": x,
            "centroid_z_um": 1.0, "centroid_y_um": y, "centroid_x_um": x,
            "volume_um3": 100.0, "touches_z_edge": False, "touches_xy_edge": edge,
        })
    return pd.DataFrame(output)


def test_inverse_transform_round_trip_and_forward_projection() -> None:
    point_b = np.array([2.0, 4.0, 5.0])
    transform = _transform()
    point_a = apply_transform_b_to_a(point_b, transform)
    recovered = project_centroid_forward(point_a, transform)
    assert np.allclose(recovered, point_b)
    assert np.allclose(apply_transform_b_to_a(point_a, invert_restricted_transform(transform)), point_b)


def test_composed_transform_matches_sequential_application() -> None:
    first = _transform(z_intercept=1.0, z_scale=1.0, y_intercept=2.0, y_from_y=1.0, y_from_x=0.0, x_intercept=3.0, x_from_y=0.0, x_from_x=1.0)
    second = _transform(z_intercept=4.0, z_scale=1.0, y_intercept=5.0, y_from_y=1.0, y_from_x=0.0, x_intercept=6.0, x_from_y=0.0, x_from_x=1.0)
    point = np.array([1.0, 2.0, 3.0])
    assert np.allclose(apply_transform_b_to_a(point, compose_transforms(first, second)), apply_transform_b_to_a(apply_transform_b_to_a(point, second), first))


def test_near_singular_transform_is_rejected() -> None:
    singular = _transform(y_from_y=1.0, y_from_x=2.0, x_from_y=2.0, x_from_x=4.0)
    try:
        invert_restricted_transform(singular)
    except ValueError as exc:
        assert "singular" in str(exc)
    else:
        raise AssertionError("near-singular transform was accepted")


def test_endpoint_detection_excludes_right_censored_final_session_and_finds_gap_return() -> None:
    sessions = _sessions(4)
    features = _features([(0, 1, 10, 10, False), (2, 2, 10, 10, False), (3, 3, 10, 10, False)])
    tracks = pd.DataFrame([{
        "track_uid": "trackA", "cluster_id": 1,
        "s0_roi": 1, "s1_roi": pd.NA, "s2_roi": 2, "s3_roi": 3,
        "match_policy": "graph",
    }])
    endpoints = detect_endpoint_events(tracks, features, sessions, lookahead=3)
    assert endpoints["end_session_index"].tolist() == [0]
    row = endpoints.iloc[0]
    assert bool(row["same_track_returns"])
    assert int(row["same_track_return_gap"]) == 2
    assert int(row["same_track_return_session_index"]) == 2


def test_candidate_search_ranks_owners_joins_graph_evidence_and_composes_t3() -> None:
    sessions = _sessions(4)
    spacing = VoxelSpacing(z_um=1.0, y_um=1.0, x_um=1.0)
    features = _features([
        (0, 1, 10.0, 10.0, False),
        (1, 10, 10.5, 10.0, False), (1, 11, 13.0, 10.0, False),
        (2, 20, 10.4, 10.0, False),
        (3, 30, 10.2, 10.0, False),
    ])
    tracks = pd.DataFrame([
        {"track_uid": "source", "cluster_id": 1, "s0_roi": 1, "s1_roi": pd.NA, "s2_roi": pd.NA, "s3_roi": pd.NA},
        {"track_uid": "new1", "cluster_id": 2, "s0_roi": pd.NA, "s1_roi": 10, "s2_roi": pd.NA, "s3_roi": pd.NA},
        {"track_uid": "other", "cluster_id": 3, "s0_roi": pd.NA, "s1_roi": 11, "s2_roi": 20, "s3_roi": 30},
    ])
    endpoints = pd.DataFrame([{
        "endpoint_id": "e1", "track_uid": "source", "end_session_index": 0,
        "end_session_id": "s0", "end_label": 1, "same_track_returns": False,
        "touches_z_edge": False, "touches_xy_edge": False,
    }])
    base = pd.DataFrame([{
        "day_a": "s0", "day_b": "s1", "label_a": 1, "label_b": 10,
        "dice": 0.61, "iou": 0.44, "ambiguity": 0.2, "score": 0.72,
        "candidate_source": "both",
    }])
    graph = pd.DataFrame([{
        "day_a": "s0", "day_b": "s1", "label_a": 1, "label_b": 10,
        "graph_status": "accepted_graph", "graph_support_count": 4,
        "graph_support_fraction": 0.8, "graph_residual_median_um": 0.9,
        "graph_inlier_fraction": 0.75, "refined_score": 0.91,
        "assignment_source": "graph",
    }])
    candidates, status = search_endpoint_candidates(
        endpoints, tracks, features, sessions, _identity_transforms(4, include_gap2=False),
        lookahead=3, search_radius_um=5.0, spacing=spacing,
        candidates=base, graph_matches=graph,
    )
    s1 = candidates.loc[candidates["target_session_id"].eq("s1")].sort_values("target_rank_by_distance")
    assert s1["target_label"].tolist() == [10, 11]
    assert int(s1.iloc[0]["target_rank_by_distance"]) == 1
    assert np.isclose(float(s1.iloc[0]["nearest_second_nearest_margin_um"]), 2.5)
    assert s1.iloc[0]["target_track_uid"] == "new1"
    assert bool(s1.iloc[0]["target_track_starts_here"])
    assert bool(s1.iloc[0]["target_is_singleton"])
    assert bool(s1.iloc[0]["existing_candidate_found"])
    assert bool(s1.iloc[0]["existing_graph_match"])
    assert s1.iloc[0]["graph_status"] == "accepted_graph"
    assert np.isclose(float(s1.iloc[0]["refined_score"]), 0.91)
    s3 = candidates.loc[candidates["target_session_id"].eq("s3")].iloc[0]
    assert s3["transform_source"] == "composed_adjacent"
    assert status["e1"] == ["ok", "ok", "ok"]


def test_classification_distinguishes_new_singleton_edge_and_no_mask() -> None:
    endpoints = pd.DataFrame([
        {"endpoint_id": "same", "track_uid": "a", "end_session_index": 0, "end_session_id": "s0", "end_label": 1, "same_track_returns": True},
        {"endpoint_id": "single", "track_uid": "b", "end_session_index": 0, "end_session_id": "s0", "end_label": 2, "same_track_returns": False},
        {"endpoint_id": "edge", "track_uid": "c", "end_session_index": 0, "end_session_id": "s0", "end_label": 3, "same_track_returns": False, "touches_xy_edge": True},
        {"endpoint_id": "none", "track_uid": "d", "end_session_index": 0, "end_session_id": "s0", "end_label": 4, "same_track_returns": False},
    ])
    candidates = pd.DataFrame([{
        "endpoint_id": "single", "target_session_index": 1, "target_label": 10,
        "target_track_starts_here": True, "target_is_singleton": True,
        "transform_reliable": True,
    }])
    result = classify_endpoint_events(endpoints, candidates, {
        "same": ["ok"], "single": ["ok"], "edge": ["ok"], "none": ["no_mask_near_prediction"],
    }).set_index("endpoint_id")
    assert result.loc["same", "classification"] == "same_track_gap_recovered"
    assert result.loc["single", "classification"] == "nearby_singleton_candidate"
    assert result.loc["edge", "classification"] == "edge_or_out_of_fov"
    assert result.loc["none", "classification"] == "no_mask_near_prediction"


def test_synthetic_benchmark_keeps_failed_search_and_rejects_preexisting_gap() -> None:
    sessions = _sessions(4)
    spacing = VoxelSpacing(z_um=1.0, y_um=1.0, x_um=1.0)
    # Continuous track drifts 10 um by s2; radius 2 means the known target is outside.
    features = _features([
        (0, 1, 0.0, 0.0, False), (1, 2, 0.0, 0.0, False), (2, 3, 10.0, 0.0, False), (3, 4, 10.0, 0.0, False),
        (0, 11, 20.0, 0.0, False), (2, 13, 20.0, 0.0, False), (3, 14, 20.0, 0.0, False),
    ])
    tracks = pd.DataFrame([
        {"track_uid": "continuous", "s0_roi": 1, "s1_roi": 2, "s2_roi": 3, "s3_roi": 4, "min_score": .9, "min_dice": .8},
        {"track_uid": "already_gapped", "s0_roi": 11, "s1_roi": pd.NA, "s2_roi": 13, "s3_roi": 14, "min_score": .9, "min_dice": .8},
    ])
    trust = SyntheticTrustConfig(require_consensus=False, require_interior=True)
    bench1 = build_synthetic_gap_benchmark(
        tracks, pd.DataFrame(), features, sessions, _identity_transforms(4),
        search_radius_um=2.0, spacing=spacing, random_seed=7, trust_config=trust,
    )
    bench2 = build_synthetic_gap_benchmark(
        tracks, pd.DataFrame(), features, sessions, _identity_transforms(4),
        search_radius_um=2.0, spacing=spacing, random_seed=7, trust_config=trust,
    )
    pd.testing.assert_frame_equal(bench1, bench2)
    row = bench1.loc[(bench1["track_uid"] == "continuous") & (bench1["source_session_index"] == 0) & (bench1["session_gap"] == 2)].iloc[0]
    assert not bool(row["true_target_in_search_radius"])
    assert not bool(row["top1_correct"])
    assert int(row["true_target_rank"]) >= 1
    # The already-gapped track may produce other continuous intervals, but never s0->s2.
    assert not ((bench1["track_uid"] == "already_gapped") & (bench1["source_session_index"] == 0) & (bench1["target_session_index"] == 2)).any()


def test_state_dropout_excludes_final_session_from_denominator() -> None:
    sessions = _sessions(3)
    features = _features([(0, 1, 0, 0, False), (1, 2, 0, 0, False), (2, 3, 0, 0, False)])
    tracks = pd.DataFrame([{"track_uid": "t", "s0_roi": 1, "s1_roi": 2, "s2_roi": 3, "match_policy": "graph"}])
    endpoints = pd.DataFrame([{"endpoint_id": "e", "track_uid": "t", "end_session_index": 1}])
    classifications = pd.DataFrame([{"endpoint_id": "e", "classification": "no_mask_near_prediction"}])
    state = pd.DataFrame([
        {"track_uid": "t", "session_index": 0, "eclipse_core_state": "Middle", "red": 1},
        {"track_uid": "t", "session_index": 1, "eclipse_core_state": "Low", "red": 1},
        {"track_uid": "t", "session_index": 2, "eclipse_core_state": "Low", "red": 1},
    ])
    summary = build_state_dropout_stratification(
        state, endpoints, classifications, tracks, features, sessions,
        policy="graph", spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1), local_density_radius_um=5,
    )
    overall = summary.loc[summary["stratum_type"].eq("overall")].set_index("state")
    assert int(overall.loc["low", "n_observations"]) == 1
    assert float(overall.loc["low", "dropout_rate"]) == 1.0
    assert int(overall.loc["middle", "n_observations"]) == 1
    assert float(overall.loc["middle", "dropout_rate"]) == 0.0


def test_manual_state_controls_are_same_session_and_covariate_matched() -> None:
    endpoints = pd.DataFrame([
        {"endpoint_id": "low", "track_uid": "l", "end_session_index": 1, "end_session_id": "s1", "end_label": 1, "eclipse_core_state": "Low", "red": 100, "volume_um3": 100, "centroid_z_um": 5, "centroid_y_um": 10, "centroid_x_um": 10, "local_density": .1, "n_days_present": 8},
        {"endpoint_id": "middle_close", "track_uid": "m1", "end_session_index": 1, "end_session_id": "s1", "end_label": 2, "eclipse_core_state": "Middle", "red": 105, "volume_um3": 105, "centroid_z_um": 5, "centroid_y_um": 11, "centroid_x_um": 10, "local_density": .11, "n_days_present": 8},
        {"endpoint_id": "middle_far", "track_uid": "m2", "end_session_index": 1, "end_session_id": "s1", "end_label": 3, "eclipse_core_state": "Middle", "red": 900, "volume_um3": 900, "centroid_z_um": 30, "centroid_y_um": 90, "centroid_x_um": 90, "local_density": 2.0, "n_days_present": 2},
        {"endpoint_id": "middle_other_session", "track_uid": "m3", "end_session_index": 2, "end_session_id": "s2", "end_label": 4, "eclipse_core_state": "Middle", "red": 100, "volume_um3": 100, "centroid_z_um": 5, "centroid_y_um": 10, "centroid_x_um": 10, "local_density": .1, "n_days_present": 8},
    ])
    features = _features([(1, 1, 10, 10, False), (1, 2, 11, 10, False), (1, 3, 90, 90, False), (2, 4, 10, 10, False)])
    manifest = build_manual_review_manifest(endpoints, features, max_review_panels=10, random_seed=0)
    controls = manifest.loc[manifest["sampling_stratum"].eq("middle_control")]
    assert len(controls) == 1
    assert controls.iloc[0]["endpoint_id"] == "middle_close"
    assert controls.iloc[0]["matched_low_endpoint_id"] == "low"
    assert "local_density" in controls.iloc[0]["state_match_covariates"]


def test_runtime_summary_does_not_mislabel_copied_graph_elapsed_time(tmp_path: Path) -> None:
    pd.DataFrame([{"day_a": "s0", "day_b": "s1", "pair_gap": 1, "elapsed_sec": 5.0, "n_a": 10, "n_b": 11, "transform_method": "affine"}]).to_csv(tmp_path / "pairwise_summary.csv", index=False)
    pd.DataFrame([{"day_a": "s0", "day_b": "s1", "elapsed_sec": 5.0, "n_graph": 7, "n_graph_anchors": 3, "n_graph_changed": 1}]).to_csv(tmp_path / "pairwise_summary_graph.csv", index=False)
    pd.DataFrame([{"day_a": "s0", "day_b": "s1", "label_a": 1, "label_b": 2}]).to_csv(tmp_path / "pairwise_candidates.csv", index=False)
    runtime = build_runtime_summary(tmp_path, "graph")
    assert np.isnan(float(runtime.iloc[0]["graph_stage_seconds"]))
    assert int(runtime.iloc[0]["n_graph"]) == 7
    assert int(runtime.iloc[0]["candidate_count"]) == 1
