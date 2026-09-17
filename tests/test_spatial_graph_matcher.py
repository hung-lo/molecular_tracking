from __future__ import annotations

import math

import numpy as np
import pandas as pd

from affine_overlap_matcher import PairMatchResult, RestrictedTransform, VoxelSpacing
from spatial_graph_matcher import (
    GRAPH_CANDIDATE_COLUMNS,
    SpatialGraphParams,
    _anchor_query_radius,
    _local_support_stats,
    _prepare_anchor_geometry,
    add_graph_consistency_scores,
    aligned_physical_coordinates,
    compare_balanced_and_graph_matches,
    graph_one_to_one_assignment,
    refine_pair_with_spatial_graph,
    select_graph_anchors,
    summarize_graph_pair,
)


def _reference_local_support_stats(
    anchor_coords_a: np.ndarray,
    anchor_coords_b: np.ndarray,
    cand_a: np.ndarray,
    cand_b: np.ndarray,
    params: SpatialGraphParams,
) -> dict[str, object]:
    """Test-only literal reference for the pre-KD-tree local scoring path."""

    if len(anchor_coords_a) == 0:
        return {
            "graph_support_count": 0,
            "graph_support_fraction": 0.0,
            "graph_residual_median_um": np.nan,
            "graph_residual_mean_um": np.nan,
            "graph_residual_p90_um": np.nan,
            "graph_inlier_fraction": np.nan,
            "graph_score": np.nan,
            "graph_status": "insufficient_support",
        }

    dist_a = np.linalg.norm(anchor_coords_a - cand_a, axis=1)
    dist_b = np.linalg.norm(anchor_coords_b - cand_b, axis=1)
    local_mask = (dist_a <= float(params.radius_um)) & (dist_b <= float(params.radius_um))
    local_indices = np.flatnonzero(local_mask)
    if local_indices.size == 0:
        return {
            "graph_support_count": 0,
            "graph_support_fraction": 0.0,
            "graph_residual_median_um": np.nan,
            "graph_residual_mean_um": np.nan,
            "graph_residual_p90_um": np.nan,
            "graph_inlier_fraction": np.nan,
            "graph_score": np.nan,
            "graph_status": "insufficient_support",
        }

    local_a_order = local_indices[np.argsort(dist_a[local_indices])[: min(int(params.k_neighbors), len(local_indices))]]
    local_b_order = local_indices[np.argsort(dist_b[local_indices])[: min(int(params.k_neighbors), len(local_indices))]]
    support_indices = np.intersect1d(local_a_order, local_b_order, assume_unique=False)
    support_count = int(len(support_indices))
    if support_count < int(params.min_anchor_support):
        return {
            "graph_support_count": support_count,
            "graph_support_fraction": float(support_count / max(len(local_indices), 1)),
            "graph_residual_median_um": np.nan,
            "graph_residual_mean_um": np.nan,
            "graph_residual_p90_um": np.nan,
            "graph_inlier_fraction": np.nan,
            "graph_score": np.nan,
            "graph_status": "insufficient_support",
        }

    residuals = []
    for anchor_index in support_indices:
        v_a = anchor_coords_a[anchor_index] - cand_a
        v_b = anchor_coords_b[anchor_index] - cand_b
        residuals.append(float(np.linalg.norm(v_a - v_b)))
    residuals_array = np.asarray(residuals, dtype=float)
    graph_residual_median_um = float(np.median(residuals_array))
    graph_residual_mean_um = float(np.mean(residuals_array))
    graph_residual_p90_um = float(np.quantile(residuals_array, 0.90))
    graph_inlier_fraction = float(np.mean(residuals_array <= float(params.inlier_residual_um)))
    support_term = min(1.0, support_count / max(int(params.min_anchor_support), 1))
    residual_term = math.exp(-((graph_residual_median_um / float(params.residual_scale_um)) ** 2))
    inlier_term = 0.5 + 0.5 * graph_inlier_fraction
    graph_score = float(support_term * residual_term * inlier_term)
    return {
        "graph_support_count": support_count,
        "graph_support_fraction": float(support_count / max(len(local_indices), 1)),
        "graph_residual_median_um": graph_residual_median_um,
        "graph_residual_mean_um": graph_residual_mean_um,
        "graph_residual_p90_um": graph_residual_p90_um,
        "graph_inlier_fraction": graph_inlier_fraction,
        "graph_score": graph_score,
        "graph_status": "scored",
    }


def _optimized_local_support_stats(
    anchor_coords_a: np.ndarray,
    anchor_coords_b: np.ndarray,
    cand_a: np.ndarray,
    cand_b: np.ndarray,
    params: SpatialGraphParams,
) -> dict[str, object]:
    coords_a = {100: cand_a}
    coords_b = {200: cand_b}
    labels = np.arange(len(anchor_coords_a), dtype=int)
    coords_a.update({int(label): coordinate for label, coordinate in zip(labels, anchor_coords_a)})
    coords_b.update({int(label): coordinate for label, coordinate in zip(labels, anchor_coords_b)})
    prepared = _prepare_anchor_geometry(labels, labels, coords_a, coords_b)
    rough_indices = prepared.tree_a.query_ball_point(
        np.asarray([cand_a]),
        r=_anchor_query_radius(float(params.radius_um)),
        p=2.0,
        eps=0.0,
        workers=1,
    )[0]
    return _local_support_stats(
        label_a=100,
        label_b=200,
        prepared_geometry=prepared,
        rough_anchor_indices=np.asarray(rough_indices, dtype=int),
        coords_a_by_label=coords_a,
        coords_b_by_label=coords_b,
        params=params,
    )


def _assert_stats_equal(actual: dict[str, object], expected: dict[str, object]) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        left = actual[key]
        right = expected[key]
        if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
            if np.isnan(left) and np.isnan(right):
                continue
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12)
        else:
            assert left == right


def test_randomized_kdtree_local_support_matches_reference() -> None:
    rng = np.random.default_rng(20260917)
    radii = [1.0, 5.0, 45.0, 100.0]
    for case in range(100):
        n_anchors = int(rng.integers(1, 80))
        anchor_coords_a = rng.uniform(-160.0, 160.0, size=(n_anchors, 3))
        anchor_coords_b = anchor_coords_a + rng.normal(0.0, 3.0 + case % 4, size=(n_anchors, 3))
        cand_a = rng.uniform(-160.0, 160.0, size=3)
        cand_b = cand_a + rng.normal(0.0, 4.0, size=3)
        params = SpatialGraphParams(
            k_neighbors=int(rng.integers(1, min(n_anchors, 10) + 1)),
            radius_um=radii[case % len(radii)],
            min_anchor_support=int(rng.integers(1, min(n_anchors, 8) + 1)),
        )
        expected = _reference_local_support_stats(anchor_coords_a, anchor_coords_b, cand_a, cand_b, params)
        actual = _optimized_local_support_stats(anchor_coords_a, anchor_coords_b, cand_a, cand_b, params)
        _assert_stats_equal(actual, expected)


def test_pathological_radius_and_tie_cases_match_reference() -> None:
    radius = 5.0
    params = SpatialGraphParams(k_neighbors=3, min_anchor_support=1, radius_um=radius)
    cases = [
        (np.asarray([[radius, 0, 0.0]]), np.asarray([[radius, 0, 0.0]]), np.zeros(3), np.zeros(3)),
        (np.asarray([[np.nextafter(radius, 0.0), 0, 0.0]]), np.asarray([[np.nextafter(radius, 0.0), 0, 0.0]]), np.zeros(3), np.zeros(3)),
        (np.asarray([[np.nextafter(radius, np.inf), 0, 0.0]]), np.asarray([[np.nextafter(radius, np.inf), 0, 0.0]]), np.zeros(3), np.zeros(3)),
        (np.asarray([[3, 4, 0.0], [-3, -4, 0.0], [4, 3, 0.0], [-4, -3, 0.0]]), np.asarray([[3, 4, 0.0], [-3, -4, 0.0], [4, 3, 0.0], [-4, -3, 0.0]]), np.zeros(3), np.zeros(3)),
        (np.asarray([[0, 0, 0.0], [1, 0, 0.0], [0, 1, 0.0]]), np.asarray([[0, 0, 0.0], [1, 0, 0.0], [0, 1, 0.0]]), np.zeros(3), np.zeros(3)),
        (np.asarray([[1, 0, 0.0], [2, 0, 0.0]]), np.asarray([[20, 0, 0.0], [1, 0, 0.0]]), np.zeros(3), np.zeros(3)),
        (np.asarray([[1, 0, 0.0], [2, 0, 0.0]]), np.asarray([[1, 0, 0.0], [20, 0, 0.0]]), np.zeros(3), np.zeros(3)),
        (np.asarray([[20, 0, 0.0]]), np.asarray([[20, 0, 0.0]]), np.zeros(3), np.zeros(3)),
        (np.asarray([[1, 0, 0.0], [2, 0, 0.0]]), np.asarray([[2, 0, 0.0], [1, 0, 0.0]]), np.zeros(3), np.zeros(3)),
    ]
    for anchor_coords_a, anchor_coords_b, cand_a, cand_b in cases:
        case_params = params
        if len(anchor_coords_a) == 2 and np.array_equal(anchor_coords_a, [[1, 0, 0], [2, 0, 0]]):
            case_params = SpatialGraphParams(k_neighbors=1, min_anchor_support=2, radius_um=radius)
        expected = _reference_local_support_stats(anchor_coords_a, anchor_coords_b, cand_a, cand_b, case_params)
        actual = _optimized_local_support_stats(anchor_coords_a, anchor_coords_b, cand_a, cand_b, case_params)
        _assert_stats_equal(actual, expected)


def _candidate_table() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray], SpatialGraphParams]:
    coords_a = {
        1: np.asarray([0.0, 0.0, 0.0]),
        2: np.asarray([10.0, 0.0, 0.0]),
        3: np.asarray([0.0, 10.0, 0.0]),
        4: np.asarray([10.0, 10.0, 0.0]),
        11: np.asarray([1.0, 1.0, 0.0]),
        12: np.asarray([30.0, 30.0, 0.0]),
        99: np.asarray([0.0, 0.0, 0.0]),
    }
    coords_b = {
        101: np.asarray([0.0, 0.0, 0.0]),
        102: np.asarray([10.0, 0.0, 0.0]),
        103: np.asarray([0.0, 10.0, 0.0]),
        104: np.asarray([10.0, 10.0, 0.0]),
        111: np.asarray([1.0, 1.0, 0.0]),
        112: np.asarray([30.0, 30.0, 0.0]),
        199: np.asarray([0.0, 0.0, 0.0]),
    }
    candidates = pd.DataFrame(
        [
            {"label_a": 1, "label_b": 101, "score": 0.95, "dice": 0.9, "distance_um": 1.0, "balanced_rule": True},
            {"label_a": 11, "label_b": 111, "score": 0.80, "dice": 0.7, "distance_um": 2.0, "balanced_rule": True},
            {"label_a": 12, "label_b": 112, "score": 0.70, "dice": 0.6, "distance_um": 3.0, "balanced_rule": True},
            {"label_a": 99, "label_b": 199, "score": 0.10, "dice": 0.1, "distance_um": 30.0, "balanced_rule": False},
        ]
    )
    anchors = pd.DataFrame([{"label_a": 1, "label_b": 101, "anchor_rank": 0}])
    return candidates, anchors, coords_a, coords_b, SpatialGraphParams()


def _reference_add_graph_consistency_scores(
    candidates: pd.DataFrame,
    anchors: pd.DataFrame,
    coords_a_by_label: dict[int, np.ndarray],
    coords_b_by_label: dict[int, np.ndarray],
    params: SpatialGraphParams,
) -> pd.DataFrame:
    """Test-only old candidate-table scorer using the reference local function."""

    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=GRAPH_CANDIDATE_COLUMNS)
    table = candidates.copy()
    table["base_score"] = pd.to_numeric(table.get("score", np.nan), errors="coerce")
    table["graph_rule"] = table.get("balanced_rule", False).astype(bool)
    table["graph_status"] = "not_balanced"
    table["match_policy"] = "graph"
    table["graph_support_count"] = 0
    table["graph_support_fraction"] = 0.0
    table["graph_residual_median_um"] = np.nan
    table["graph_residual_mean_um"] = np.nan
    table["graph_residual_p90_um"] = np.nan
    table["graph_inlier_fraction"] = np.nan
    table["graph_score"] = np.nan
    table["refined_score"] = table["base_score"]
    table["is_graph_anchor"] = False
    if anchors is None or anchors.empty:
        table.loc[table["graph_rule"], "graph_status"] = "no_anchor_fallback"
        return table
    anchor_pairs = {(int(row.label_a), int(row.label_b)) for row in anchors.itertuples(index=False)}
    anchor_a = anchors["label_a"].astype(int).to_numpy()
    anchor_b = anchors["label_b"].astype(int).to_numpy()
    anchor_coords_a = np.vstack([coords_a_by_label[int(label)] for label in anchor_a])
    anchor_coords_b = np.vstack([coords_b_by_label[int(label)] for label in anchor_b])
    for index, row in table.loc[table["graph_rule"]].iterrows():
        label_a = int(row["label_a"])
        label_b = int(row["label_b"])
        if (label_a, label_b) in anchor_pairs:
            table.at[index, "is_graph_anchor"] = True
            table.at[index, "graph_status"] = "anchor"
            table.at[index, "graph_score"] = 1.0
            table.at[index, "refined_score"] = float(row["base_score"])
            continue
        stats = _reference_local_support_stats(
            anchor_coords_a,
            anchor_coords_b,
            coords_a_by_label[label_a],
            coords_b_by_label[label_b],
            params,
        )
        for key, value in stats.items():
            table.at[index, key] = value
        if table.at[index, "graph_status"] == "scored":
            table.at[index, "refined_score"] = float((1.0 - float(params.graph_weight)) * float(row["base_score"]) + float(params.graph_weight) * float(stats["graph_score"]))
        else:
            table.at[index, "refined_score"] = float(row["base_score"])
    return table


def test_whole_candidate_table_matches_reference() -> None:
    candidates, anchors, coords_a, coords_b, params = _candidate_table()
    expected = _reference_add_graph_consistency_scores(candidates, anchors, coords_a, coords_b, params)
    actual = add_graph_consistency_scores(candidates, anchors, coords_a, coords_b, params)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    expected_matches = graph_one_to_one_assignment(expected, anchors, params)
    actual_matches = graph_one_to_one_assignment(actual, anchors, params)
    pd.testing.assert_frame_equal(actual_matches, expected_matches, check_exact=True)
    assert list(zip(actual_matches["label_a"], actual_matches["label_b"])) == [(1, 101), (11, 111), (12, 112)]


def test_no_anchor_fallback_matches_reference() -> None:
    candidates, anchors, coords_a, coords_b, params = _candidate_table()
    empty_anchors = anchors.iloc[0:0].copy()
    expected = _reference_add_graph_consistency_scores(candidates, empty_anchors, coords_a, coords_b, params)
    actual = add_graph_consistency_scores(candidates, empty_anchors, coords_a, coords_b, params)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.loc[actual["graph_rule"], "graph_status"].tolist() == ["no_anchor_fallback"] * 3


def test_end_to_end_pair_matches_reference_scoring() -> None:
    candidates, _unused_anchors, coords_a, coords_b, params = _candidate_table()
    features_a = pd.DataFrame(
        [{"label": label, "centroid_z": coordinate[0], "centroid_y": coordinate[1], "centroid_x": coordinate[2]} for label, coordinate in coords_a.items()]
    ).set_index("label", drop=False)
    features_b = pd.DataFrame(
        [{"label": label, "centroid_z": coordinate[0], "centroid_y": coordinate[1], "centroid_x": coordinate[2]} for label, coordinate in coords_b.items()]
    ).set_index("label", drop=False)
    high_matches = pd.DataFrame(
        [{"label_a": 1, "label_b": 101, "score": 0.95, "dice": 0.9, "distance_um": 1.0, "area_ratio": 1.0, "ambiguity": 0.1, "candidate_source": "both"}]
    )
    balanced_matches = candidates.loc[candidates["balanced_rule"]].copy()
    baseline = PairMatchResult(
        candidates=candidates,
        high_matches=high_matches,
        balanced_matches=balanced_matches,
        summary={"n_a": len(features_a), "n_b": len(features_b), "pair_gap": 1},
        transform=RestrictedTransform(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, "identity", None, 1, 1, 0.0, 0.0),
    )
    spacing = VoxelSpacing(z_um=1.0, y_um=1.0, x_um=1.0)
    actual = refine_pair_with_spatial_graph(
        session_a="a",
        session_b="b",
        baseline_result=baseline,
        features_a=features_a,
        features_b=features_b,
        spacing=spacing,
        params=params,
        pair_gap=1,
    )

    coords_a_aligned, coords_b_aligned = aligned_physical_coordinates(features_a, features_b, baseline.transform, spacing)
    expected_anchors = select_graph_anchors(high_matches, params)
    expected_candidates = _reference_add_graph_consistency_scores(candidates, expected_anchors, coords_a_aligned, coords_b_aligned, params)
    expected_matches = graph_one_to_one_assignment(expected_candidates, expected_anchors, params)
    expected_changes = compare_balanced_and_graph_matches(balanced_matches, expected_matches)
    expected_summary = summarize_graph_pair(
        session_a="a",
        session_b="b",
        pair_gap=1,
        baseline_result=baseline,
        anchors=expected_anchors,
        graph_matches=expected_matches,
        changes=expected_changes,
    )
    pd.testing.assert_frame_equal(actual.anchors, expected_anchors, check_exact=True)
    pd.testing.assert_frame_equal(actual.candidates, expected_candidates, check_exact=True)
    pd.testing.assert_frame_equal(actual.graph_matches, expected_matches, check_exact=True)
    pd.testing.assert_frame_equal(actual.changes, expected_changes, check_exact=True)
    assert actual.summary == expected_summary
