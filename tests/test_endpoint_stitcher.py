from __future__ import annotations

from pathlib import Path

import pandas as pd

from affine_overlap_matcher import VoxelSpacing
from endpoint_evaluator import search_endpoint_candidates
from endpoint_stitcher import (
    StitcherConfig, assign_stitches, build_stitch_candidates, build_stitched_tracks,
    build_synthetic_stitch_benchmark, summarize_stitch_benchmark, tier_stitch_candidates,
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
    assert metrics["negative_fpr"] == 0
