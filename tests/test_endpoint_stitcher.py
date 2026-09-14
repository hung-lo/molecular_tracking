from __future__ import annotations

from pathlib import Path

import pandas as pd

from affine_overlap_matcher import VoxelSpacing
from endpoint_evaluator import search_endpoint_candidates
from endpoint_stitcher import (
    StitcherConfig, assign_stitches, benchmark_threshold_sweep, build_stitch_candidates,
    build_stitched_tracks, build_synthetic_stitch_benchmark, select_review_samples,
    stable_stitch_edge_id, summarize_stitch_benchmark, tier_stitch_candidates,
)


def stitch_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = pd.DataFrame({
        "session_index": range(6), "session_id": [f"s{i}" for i in range(6)],
        "acquisition_date": [f"2026-01-{i + 1:02d}" for i in range(6)],
    })
    transforms = pd.DataFrame([
        {
            "day_a": f"s{i}", "day_b": f"s{i + gap}", "pair_gap": gap,
            "z_intercept": 0, "z_scale": 1, "y_intercept": 0, "y_from_y": 1,
            "y_from_x": 0, "x_intercept": 0, "x_from_y": 0, "x_from_x": 1,
            "method": "restricted_affine", "fallback_reason": "",
            "residual_median_um": .1, "residual_p95_um": .2,
        }
        for gap in (1, 2) for i in range(6 - gap)
    ])
    rows = []
    tracks = []
    for track_index, y in enumerate((0.0, 10.0, 20.0, 30.0), start=1):
        track = {
            "track_uid": f"t{track_index}", "cluster_id": track_index,
            "has_cycle_conflict": False, "edge_heavy": False,
            "first_session_index": 0, "last_session_index": 5,
            "n_days_present": 6, "missing_internal_days": 0,
        }
        for session_index in range(6):
            label = track_index * 100 + session_index
            track[f"s{session_index}_roi"] = label
            rows.append({
                "session_index": session_index, "session_id": f"s{session_index}", "label": label,
                "centroid_z": 1.0, "centroid_y": y, "centroid_x": 5.0,
                "volume_um3": 100.0, "touches_z_edge": False, "touches_xy_edge": False,
            })
        tracks.append(track)
    return sessions, pd.DataFrame(rows), transforms, pd.DataFrame(tracks)


def _fragment_fixture(gap: int = 1):
    sessions, features, transforms, tracks = stitch_fixture()
    source, target = tracks.iloc[0].copy(), tracks.iloc[0].copy()
    source["track_uid"], target["track_uid"] = "source", "target"
    source_position, target_position = 1, 1 + gap
    for position in range(6):
        if position > source_position:
            source[f"s{position}_roi"] = pd.NA
        if position < target_position:
            target[f"s{position}_roi"] = pd.NA
    source["first_session_index"], source["last_session_index"], source["n_days_present"] = 0, source_position, 2
    target["first_session_index"], target["last_session_index"], target["n_days_present"] = target_position, 5, 6 - target_position
    fragmented = pd.concat([pd.DataFrame([source, target]), tracks.iloc[1:]], ignore_index=True)
    endpoint = pd.DataFrame([{
        "endpoint_id": "e1", "track_uid": "source", "end_session_index": source_position,
        "end_session_id": f"s{source_position}", "end_label": int(source[f"s{source_position}_roi"]),
        "same_track_returns": False, "touches_z_edge": False, "touches_xy_edge": False,
    }])
    broad, _ = search_endpoint_candidates(
        endpoint, fragmented, features, sessions, transforms, lookahead=3,
        search_radius_um=15, spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
    )
    classification = pd.DataFrame([{"endpoint_id": "e1", "classification": "nearby_new_track_candidate"}])
    return sessions, features, transforms, fragmented, endpoint, broad, classification


def test_candidate_evidence_is_state_blind_and_auto_accepts_stable_continuation() -> None:
    args = _fragment_fixture(1)
    sessions, features, transforms, tracks, endpoint, broad, classification = args
    result = build_stitch_candidates(
        endpoint, broad, classification, tracks, features, sessions, transforms,
        spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
    )
    true = result.loc[result["target_track_uid"].eq("target")].iloc[0]
    assert true["candidate_tier"] == "auto_accept_candidate"
    assert bool(true["reciprocal_rank1"])
    assert int(true["anchor_support_count"]) == 3
    assert float(true["anchor_residual_median_um"]) == 0
    assert true["transform_component_methods"] == "restricted_affine"
    assert float(true["transform_component_residual_median_um_max"]) == .1
    assert float(true["transform_component_residual_p95_um_max"]) == .2
    gap2_args = _fragment_fixture(2)
    gap2 = build_stitch_candidates(
        gap2_args[4], gap2_args[5], gap2_args[6], gap2_args[3], gap2_args[1], gap2_args[0], gap2_args[2],
        spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
    )
    gap2_true = gap2.loc[gap2["target_track_uid"].eq("target")].iloc[0]
    assert gap2_true["transform_source"] == "direct_stored"
    assert float(gap2_true["direct_vs_composed_projection_delta_um"]) == 0
    gap3_args = _fragment_fixture(3)
    gap3 = build_stitch_candidates(
        gap3_args[4], gap3_args[5], gap3_args[6], gap3_args[3], gap3_args[1], gap3_args[0], gap3_args[2],
        spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
    )
    gap3_true = gap3.loc[gap3["target_track_uid"].eq("target")].iloc[0]
    assert gap3_true["transform_source"] == "composed_adjacent"
    assert gap3_true["transform_component_methods"].count("restricted_affine") == 3
    assert float(gap3_true["transform_component_residual_p95_um_max"]) == .2
    changed = endpoint.assign(eclipse_z=999, eclipse_core_state="Low", green=-100, red=1e9)
    changed_result = build_stitch_candidates(
        changed, broad.assign(eclipse_z=-999, green=1e12).sample(frac=1, random_state=7), classification,
        tracks.sample(frac=1, random_state=8), features.sample(frac=1, random_state=9), sessions,
        transforms.sample(frac=1, random_state=10), spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
    )
    pd.testing.assert_frame_equal(result, changed_result)


def test_singletons_nonreciprocal_and_insufficient_anchors_are_never_auto() -> None:
    base = pd.DataFrame([{
        "source_track_uid": "a", "target_track_uid": "b", "source_session_index": 0,
        "target_session_index": 1, "session_gap": 1, "target_track_starts_here": True,
        "transform_reliable": True, "projected_distance_um": 1.0,
        "source_n_days_present": 1, "target_n_days_present": 1,
        "forward_rank_track_starts": 1, "reverse_rank_source_endpoints": 2,
        "forward_second_margin_um": 5, "anchor_support_count": 2,
        "source_history_n": 1, "target_future_n": 0,
    }])
    result = tier_stitch_candidates(base).iloc[0]
    assert result["candidate_tier"] == "manual_review"
    assert "singleton_source" in result["review_reasons"]
    assert "singleton_target" in result["review_reasons"]
    assert "reverse_not_rank1" in result["review_reasons"]


def _safe_edges() -> pd.DataFrame:
    return pd.DataFrame([
        {"stitch_edge_id": "e1", "source_track_uid": "a", "target_track_uid": "x", "auto_eligible": True, "projected_distance_um": 1.0, "anchor_residual_median_um": 1.0, "source_history_distance_median_um": 1.0, "target_future_distance_median_um": 1.0, "forward_second_margin_um": 5, "reverse_second_margin_um": 5, "session_gap": 1},
        {"stitch_edge_id": "e2", "source_track_uid": "b", "target_track_uid": "x", "auto_eligible": True, "projected_distance_um": 2.0, "anchor_residual_median_um": 1.0, "source_history_distance_median_um": 1.0, "target_future_distance_median_um": 1.0, "forward_second_margin_um": 5, "reverse_second_margin_um": 5, "session_gap": 1},
        {"stitch_edge_id": "e3", "source_track_uid": "b", "target_track_uid": "y", "auto_eligible": False, "projected_distance_um": .1, "session_gap": 1},
    ])


def test_global_assignment_is_one_to_one_uses_no_link_and_is_row_order_invariant() -> None:
    first = assign_stitches(_safe_edges()).sort_values("stitch_edge_id").reset_index(drop=True)
    second = assign_stitches(_safe_edges().sample(frac=1, random_state=4)).sort_values("stitch_edge_id").reset_index(drop=True)
    assert first.loc[first["accepted_by_global_assignment"], "stitch_edge_id"].tolist() == ["e1"]
    assert first.loc[first["stitch_edge_id"].eq("e3"), "assignment_status"].iloc[0] == "failed_gate"
    pd.testing.assert_series_equal(first["accepted_by_global_assignment"], second["accepted_by_global_assignment"])


def test_transitive_merge_conserves_nodes_and_preserves_gap_missingness() -> None:
    sessions = pd.DataFrame({"session_index": range(5), "session_id": [f"s{i}" for i in range(5)]})
    tracks = pd.DataFrame([
        {"track_uid": "early", "s0_roi": 1, "s1_roi": 2, "s2_roi": pd.NA, "s3_roi": pd.NA, "s4_roi": pd.NA},
        {"track_uid": "late", "s0_roi": pd.NA, "s1_roi": pd.NA, "s2_roi": pd.NA, "s3_roi": 3, "s4_roi": 4},
    ])
    assignments = pd.DataFrame([{
        "source_track_uid": "early", "target_track_uid": "late", "accepted_by_global_assignment": True,
        "session_gap": 2, "elapsed_day_gap": 2,
    }])
    stitched, mapping, invariants = build_stitched_tracks(tracks, assignments, sessions)
    assert len(stitched) == 1 and stitched.iloc[0]["track_uid"] == "early"
    assert pd.isna(stitched.iloc[0]["s2_roi"])
    assert invariants["observation_conservation_passed"]
    assert len(mapping) == 2


def test_pseudo_fragment_benchmark_has_all_gaps_and_explicit_negatives() -> None:
    sessions, features, transforms, tracks = stitch_fixture()
    benchmark = build_synthetic_stitch_benchmark(
        tracks, features, sessions, transforms, spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
        replicates=2, random_seed=0,
    )
    assert {1, 2, 3}.issubset(set(benchmark.loc[benchmark["case_type"].eq("positive"), "session_gap"]))
    assert {"no_successor", "target_only"}.issubset(set(benchmark["case_type"]))
    metrics = summarize_stitch_benchmark(benchmark)
    assert metrics["n_positive_truth"] == 12
    assert metrics["n_negative_controls"] == 16
    assert metrics["negative_fpr"] >= 0
    assert {"no_successor_no_mask", "no_successor_nonstart_mask", "target_only"}.issubset(set(benchmark["negative_subtype"]))
    assert benchmark["replicate"].nunique() == 1
    assert metrics["n_unique_positive_truth_tracks"] == 4
    assert metrics["n_unique_positive_truth_splits"] == 12
    assert benchmark.loc[benchmark["case_type"].eq("positive"), "canonical_truth_split_id"].is_unique
    assert benchmark["collision"].any()
    assert (benchmark.loc[benchmark["collision"], "n_auto_edges_in_component"] >= 2).all()
    collision_rows = benchmark.loc[benchmark["case_id"].isin(["positive_r000_g1_0003", "collision_competitor_r000_g1"])]
    collision_positive = collision_rows.loc[collision_rows["case_type"].eq("positive")].iloc[0]
    assert bool(collision_positive["accepted"])
    assert collision_positive["assigned_target_track_uid"] == collision_positive["true_target_track_uid"]
    assert int(collision_rows["accepted"].sum()) == 1
    assert not metrics["benchmark_sample_sufficient"]
    assert not metrics["guardrail_passed"]


def test_target_feature_edge_is_rejected_without_track_edge_flag() -> None:
    sessions, features, transforms, tracks, endpoints, broad, classifications = _fragment_fixture(1)
    target_label = int(tracks.loc[tracks["track_uid"].eq("target"), "s2_roi"].iloc[0])
    features.loc[features["session_id"].eq("s2") & features["label"].eq(target_label), "touches_xy_edge"] = True
    result = build_stitch_candidates(
        endpoints, broad, classifications, tracks.drop(columns=["edge_heavy"]), features,
        sessions, transforms, spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
    )
    target = result.loc[result["target_track_uid"].eq("target")].iloc[0]
    assert bool(target["target_touches_xy_edge"])
    assert target["candidate_tier"] == "reject"
    assert "edge_or_out_of_fov" in target["rejection_reasons"]


def test_missing_source_or_target_feature_row_fails_closed() -> None:
    sessions, features, transforms, tracks, endpoints, broad, classifications = _fragment_fixture(1)
    target_label = int(tracks.loc[tracks["track_uid"].eq("target"), "s2_roi"].iloc[0])
    for missing_key in (("s1", int(endpoints.iloc[0]["end_label"])), ("s2", target_label)):
        reduced = features.loc[~(features["session_id"].eq(missing_key[0]) & features["label"].eq(missing_key[1]))]
        result = build_stitch_candidates(
            endpoints, broad, classifications, tracks, reduced, sessions, transforms,
            spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
        )
        row = result.loc[result["target_track_uid"].eq("target")].iloc[0]
        field = "source_feature_row_missing" if missing_key[0] == "s1" else "target_feature_row_missing"
        assert bool(row[field])
        assert row["candidate_tier"] == "reject"
        assert "missing_feature_row" in row["rejection_reasons"]


def test_target_only_control_uses_first_observed_future_when_next_session_missing() -> None:
    sessions, features, transforms, tracks = stitch_fixture()
    tracks["s3_roi"] = tracks["s3_roi"].astype("Int64")
    tracks.loc[:, "s3_roi"] = pd.NA
    benchmark = build_synthetic_stitch_benchmark(
        tracks, features, sessions, transforms, spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
        replicates=1, cases_per_gap=1, random_seed=0,
    )
    assert not benchmark.loc[benchmark["case_type"].eq("target_only")].empty


def test_stable_edge_ids_ignore_unrelated_rows() -> None:
    edge = {
        "source_track_uid": "source", "source_session_index": 1, "source_label": 12,
        "target_track_uid": "target", "target_session_index": 3, "target_label": 34,
    }
    assert stable_stitch_edge_id(edge) == stable_stitch_edge_id({**edge, "projected_distance_um": 999})
    assert stable_stitch_edge_id(edge) != stable_stitch_edge_id({**edge, "target_track_uid": "other"})


def test_benchmark_metrics_separate_positive_errors_from_negative_fpr() -> None:
    benchmark = pd.DataFrame([
        {"case_type": "positive", "session_gap": 1, "accepted": True, "correct_assignment": True, "false_positive": False},
        {"case_type": "positive", "session_gap": 1, "accepted": True, "correct_assignment": False, "false_positive": True},
        {"case_type": "no_successor", "session_gap": 1, "accepted": False, "correct_assignment": False, "false_positive": False},
    ])
    metrics = summarize_stitch_benchmark(benchmark)
    assert metrics["wrong_target_count"] == 1
    assert metrics["negative_false_positive_count"] == 0
    assert metrics["negative_fpr"] == 0
    assert metrics["precision"] == .5


def test_perfect_sufficient_benchmark_can_pass_statistical_guard() -> None:
    positives = pd.DataFrame({
        "case_type": "positive", "session_gap": ([1] * 334) + ([2] * 333) + ([3] * 333),
        "accepted": True, "correct_assignment": True, "false_positive": False,
    })
    negatives = pd.DataFrame({
        "case_type": "no_successor", "session_gap": ([1] * 1000) + ([2] * 1000) + ([3] * 1000),
        "accepted": False, "correct_assignment": False, "false_positive": False,
    })
    metrics = summarize_stitch_benchmark(pd.concat([positives, negatives], ignore_index=True))
    assert metrics["benchmark_sample_sufficient"]
    assert metrics["precision_wilson_lower_onesided_95"] >= .995
    assert metrics["negative_fpr_wilson_upper_onesided_95"] <= .001
    assert metrics["guardrail_passed"]


def test_recycled_truth_splits_fail_sample_guard() -> None:
    positives = pd.DataFrame({
        "case_type": "positive", "session_gap": ([1] * 334) + ([2] * 333) + ([3] * 333),
        "accepted": True, "correct_assignment": True, "false_positive": False,
        "canonical_truth_track_uid": "same_track", "canonical_truth_split_id": "same_split",
    })
    negatives = pd.DataFrame({
        "case_type": "no_successor", "session_gap": ([1] * 1000) + ([2] * 1000) + ([3] * 1000),
        "accepted": False, "correct_assignment": False, "false_positive": False,
    })
    metrics = summarize_stitch_benchmark(pd.concat([positives, negatives], ignore_index=True))
    assert metrics["n_positive_truth"] == 1000
    assert metrics["n_unique_positive_truth_splits"] == 1
    assert not metrics["benchmark_sample_sufficient"]


def test_threshold_sweep_replays_assignment_and_changes_global_winner() -> None:
    rows = []
    for target_uid, distance, anchor, history, future in (("x", 4.5, 0, 0, 0), ("y", 4.0, 4, 5, 5)):
        rows.append({
            "stitch_edge_id": f"edge_{target_uid}", "source_track_uid": "source", "target_track_uid": target_uid,
            "source_session_index": 1, "target_session_index": 3, "source_label": 11, "target_label": 22,
            "session_gap": 2, "target_track_starts_here": True, "transform_reliable": True,
            "projected_distance_um": distance, "source_n_days_present": 2, "target_n_days_present": 2,
            "forward_rank_track_starts": 1, "forward_rank_all_masks": 1, "reverse_rank_source_endpoints": 1,
            "forward_second_margin_um": 5, "reverse_second_margin_um": 5, "anchor_support_count": 3,
            "anchor_residual_median_um": anchor, "anchor_inlier_fraction": 1.0,
            "source_history_n": 2, "source_history_distance_median_um": history,
            "target_future_n": 1, "target_future_distance_median_um": future,
        })
    candidates = pd.DataFrame(rows)
    benchmark = pd.DataFrame([{"case_type": "positive", "session_gap": 2}])
    benchmark.attrs["threshold_replay_batches"] = [{
        "batch_id": "batch", "candidates": candidates,
        "cases": [{"case_id": "positive", "case_type": "positive", "session_gap": 2, "source_uid": "source", "target_uid": "x"}],
    }]
    sweep = benchmark_threshold_sweep(benchmark)
    baseline = sweep.loc[(sweep["distance_scale"] == 1.0) & (sweep["min_forward_margin_um"] == 3.0)].iloc[0]
    narrow = sweep.loc[(sweep["distance_scale"] == .8) & (sweep["min_forward_margin_um"] == 3.0)].iloc[0]
    assert baseline["assigned_positive_target_track_uids"] == "x"
    assert narrow["assigned_positive_target_track_uids"] == "y"


def test_review_sampling_only_claims_spatial_distribution_with_xyz_bins() -> None:
    assignments = pd.DataFrame([
        {"stitch_edge_id": "a", "candidate_tier": "auto_accept_candidate", "accepted_by_global_assignment": True, "assignment_status": "accepted", "session_gap": 1, "review_reasons": "", "volume_ratio": 1.0, "predicted_z_um": 0, "predicted_y_um": 0, "predicted_x_um": 0},
        {"stitch_edge_id": "b", "candidate_tier": "auto_accept_candidate", "accepted_by_global_assignment": True, "assignment_status": "accepted", "session_gap": 1, "review_reasons": "", "volume_ratio": 1.0, "predicted_z_um": 100, "predicted_y_um": 100, "predicted_x_um": 100},
    ])
    selected = select_review_samples(assignments, max_panels=10)
    assert "spatially_distributed_accepted" in set(selected["review_sample_reason"])


def test_review_sampling_is_deterministic_and_records_reason() -> None:
    assignments = pd.DataFrame([
        {"stitch_edge_id": "a", "candidate_tier": "auto_accept_candidate", "accepted_by_global_assignment": True, "assignment_status": "accepted", "session_gap": 1, "review_reasons": "", "volume_ratio": .2},
        {"stitch_edge_id": "b", "candidate_tier": "auto_accept_candidate", "accepted_by_global_assignment": False, "assignment_status": "global_collision_rejection", "session_gap": 2, "review_reasons": "", "volume_ratio": 1.0},
    ])
    first = select_review_samples(assignments, max_panels=2, random_seed=4)
    second = select_review_samples(assignments, max_panels=2, random_seed=4)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["review_sample_reason"]) == {"auto_accept_gap_1", "global_collision_rejection"}
