from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from affine_overlap_matcher import greedy_one_to_one
from pairwise_shadow_diagnostics import diagnose_candidate_table, trace_missing_candidate_gates


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label_a": [1, 1, 2, 2],
            "label_b": [20, 21, 20, 21],
            "score": [0.90, 0.80, 0.85, 0.70],
            "dice": [0.80, 0.30, 0.40, 0.30],
            "iou": [0.67, 0.18, 0.25, 0.18],
            "distance_um": [1.0, 6.0, 3.0, 6.0],
            "ambiguity": [0.2, 0.9, 0.8, 1.0],
            "area_ratio": [0.9, 0.5, 0.9, 0.5],
            "high_rule": [True, False, True, False],
            "balanced_rule": [True, True, True, True],
        }
    )


def test_candidate_ranks_and_margins_follow_production_order_deterministically() -> None:
    candidates = _candidates()
    first = diagnose_candidate_table(candidates)
    second = diagnose_candidate_table(candidates)

    assert_frame_equal(first, second)
    by_edge = first.set_index(["label_a", "label_b"])
    assert by_edge.loc[(1, 20), "candidate_rank"] == 1
    assert by_edge.loc[(1, 21), "candidate_rank"] == 2
    assert by_edge.loc[(2, 20), "candidate_rank"] == 1
    assert by_edge.loc[(2, 20), "reciprocal_candidate_rank"] == 2
    assert by_edge.loc[(1, 20), "reciprocal_candidate_rank"] == 1
    assert abs(by_edge.loc[(1, 20), "score_margin_to_source_second_best"] - 0.10) < 1e-12
    assert by_edge.loc[(1, 21), "score_margin_to_source_second_best"] == 0.0
    assert by_edge.loc[(1, 20), "candidate_count_for_source"] == 2
    assert by_edge.loc[(1, 20), "candidate_count_for_target"] == 2


def test_selected_competitor_and_one_to_one_rejection_are_attributed() -> None:
    candidates = _candidates()
    selected = greedy_one_to_one(candidates, "balanced_rule")
    diagnosed = diagnose_candidate_table(candidates, selected)
    by_edge = diagnosed.set_index(["label_a", "label_b"])

    assert set(map(tuple, selected[["label_a", "label_b"]].to_numpy())) == {(1, 20), (2, 21)}
    target_loss = by_edge.loc[(2, 20)]
    assert bool(target_loss["one_to_one_rejected"])
    assert target_loss["one_to_one_rejection_reason"] == "same_source_and_target_conflicts"
    assert bool(target_loss["same_source_competition"])
    assert bool(target_loss["different_source_same_target_competition"])
    assert (target_loss["selected_competing_source"], target_loss["selected_competing_target"]) == (1, 20)
    source_loss = by_edge.loc[(1, 21)]
    assert bool(source_loss["one_to_one_rejected"])
    assert source_loss["one_to_one_rejection_reason"] == "same_source_and_target_conflicts"
    assert bool(source_loss["same_source_competition"])
    assert bool(source_loss["different_source_same_target_competition"])
    assert (source_loss["selected_winner_target"], source_loss["selected_winner_score"]) == (20, 0.90)


def test_rule_failure_reasons_match_current_production_booleans() -> None:
    candidate = _candidates().iloc[[1]].copy()
    high_only = candidate.iloc[0].copy()
    high_only["dice"] = 0.25
    high_only["distance_um"] = 6.0
    high_only["area_ratio"] = 0.5
    high_only["ambiguity"] = 0.9
    high_only["high_rule"] = False
    high_only["balanced_rule"] = True

    low = candidate.iloc[0].copy()
    low["dice"] = 0.15
    low["distance_um"] = 9.0
    low["area_ratio"] = 0.25
    low["ambiguity"] = 0.9
    low["high_rule"] = False
    low["balanced_rule"] = False
    table = pd.DataFrame([high_only, low])
    diagnosed = diagnose_candidate_table(table)

    assert diagnosed["high_rule_recomputed"].tolist() == [False, False]
    assert diagnosed["balanced_rule_recomputed"].tolist() == [True, False]
    assert diagnosed["high_rule_only_failure"].tolist() == [True, False]
    assert "high_rule_only_failure" in diagnosed.loc[0, "rule_failure_reason_string"]
    assert "balanced_rule_failed" in diagnosed.loc[1, "rule_failure_reason_string"]
    assert not diagnosed["production_score_gate_present"].any()
    assert not diagnosed["production_reciprocal_rank_gate_present"].any()
    assert not diagnosed["production_margin_gate_present"].any()


def test_missing_candidate_trace_reports_the_first_actual_failed_route_gate() -> None:
    far = trace_missing_candidate_gates(
        source_feature_present=True,
        target_feature_present=True,
        transform_available=True,
        source_nearest_distance_um=11.0,
        exact_pair_distance_um=12.0,
        mutual_nearest_pair=False,
        intersection_voxels=0,
        dice=0.0,
    )
    assert far["first_failed_gate"] == "centroid_candidate_max_distance_um"
    assert far["overlap_route_failure"] == "no_mask_intersection"
    assert not far["candidate_would_be_generated"]

    near_nonmutual = trace_missing_candidate_gates(
        source_feature_present=True,
        target_feature_present=True,
        transform_available=True,
        source_nearest_distance_um=4.0,
        exact_pair_distance_um=5.0,
        mutual_nearest_pair=False,
        intersection_voxels=2,
        dice=0.05,
    )
    assert near_nonmutual["first_failed_gate"] == "mutual_nearest_centroid_pair"
    assert near_nonmutual["overlap_route_failure"] == "overlap_candidate_min_dice"
    assert not near_nonmutual["candidate_would_be_generated"]

    centroid_rescue = trace_missing_candidate_gates(
        source_feature_present=True,
        target_feature_present=True,
        transform_available=True,
        source_nearest_distance_um=4.0,
        exact_pair_distance_um=5.0,
        mutual_nearest_pair=True,
        intersection_voxels=0,
        dice=0.0,
    )
    assert centroid_rescue["candidate_would_be_generated"]


def test_shadow_diagnostic_is_read_only_and_preserves_production_assignment() -> None:
    candidates = _candidates()
    candidates_before = candidates.copy(deep=True)
    production_before = greedy_one_to_one(candidates, "balanced_rule")
    production_before_copy = production_before.copy(deep=True)

    diagnosed = diagnose_candidate_table(candidates, production_before)

    assert_frame_equal(candidates, candidates_before)
    assert_frame_equal(production_before, production_before_copy)
    replayed = greedy_one_to_one(candidates, "balanced_rule")
    assert_frame_equal(replayed, production_before)
    assert diagnosed["one_to_one_selected"].sum() == len(production_before)
