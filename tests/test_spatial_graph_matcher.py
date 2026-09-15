from __future__ import annotations

import math

import numpy as np

from spatial_graph_matcher import (
    SpatialGraphParams,
    _anchor_neighborhood_cache,
    _local_support_stats,
)


def _brute_support(anchor_a: np.ndarray, anchor_b: np.ndarray, cand_a: np.ndarray, cand_b: np.ndarray, params: SpatialGraphParams) -> dict[str, object]:
    dist_a = np.linalg.norm(anchor_a - cand_a, axis=1)
    dist_b = np.linalg.norm(anchor_b - cand_b, axis=1)
    local_indices = np.flatnonzero((dist_a <= params.radius_um) & (dist_b <= params.radius_um))
    local_a_order = local_indices[np.argsort(dist_a[local_indices])[: min(params.k_neighbors, len(local_indices))]]
    local_b_order = local_indices[np.argsort(dist_b[local_indices])[: min(params.k_neighbors, len(local_indices))]]
    support_indices = np.intersect1d(local_a_order, local_b_order, assume_unique=False)
    support_count = len(support_indices)
    residuals = np.asarray([
        np.linalg.norm((anchor_a[index] - cand_a) - (anchor_b[index] - cand_b))
        for index in support_indices
    ])
    median = float(np.median(residuals))
    mean = float(np.mean(residuals))
    p90 = float(np.quantile(residuals, 0.9))
    inlier = float(np.mean(residuals <= params.inlier_residual_um))
    score = math.exp(-((median / params.residual_scale_um) ** 2)) * (0.5 + 0.5 * inlier)
    return {
        "graph_support_count": support_count,
        "graph_support_fraction": support_count / len(local_indices),
        "graph_residual_median_um": median,
        "graph_residual_mean_um": mean,
        "graph_residual_p90_um": p90,
        "graph_inlier_fraction": inlier,
        "graph_score": score,
        "graph_status": "scored",
    }


def test_kdtree_neighborhood_cache_preserves_brute_force_support_exactly() -> None:
    rng = np.random.default_rng(7)
    anchor_a = rng.normal(size=(50, 3)) * 20.0
    anchor_b = anchor_a + rng.normal(size=(50, 3))
    cand_a = np.zeros(3)
    cand_b = np.zeros(3)
    anchor_a[0] = [45.0, 0.0, 0.0]
    anchor_b[0] = [45.0, 0.0, 0.0]
    coords_a = {100: cand_a}
    coords_b = {200: cand_b}
    params = SpatialGraphParams()

    neighborhoods_a = _anchor_neighborhood_cache(np.asarray([100]), coords_a, anchor_a, params.radius_um)
    neighborhoods_b = _anchor_neighborhood_cache(np.asarray([200]), coords_b, anchor_b, params.radius_um)
    actual = _local_support_stats(
        label_a=100,
        label_b=200,
        anchor_coords_a=anchor_a,
        anchor_coords_b=anchor_b,
        neighborhood_a=neighborhoods_a[100],
        neighborhood_b=neighborhoods_b[200],
        coords_a_by_label=coords_a,
        coords_b_by_label=coords_b,
        params=params,
    )
    expected = _brute_support(anchor_a, anchor_b, cand_a, cand_b, params)
    assert actual == expected
