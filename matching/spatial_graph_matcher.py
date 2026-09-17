"""Experimental local spatial-graph refinement for daywise ROI matching."""

from __future__ import annotations

from dataclasses import dataclass
import time
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from affine_overlap_matcher import PairMatchResult, VoxelSpacing, RestrictedTransform

GRAPH_MATCHER_ALGORITHM_VERSION = "local_spatial_graph_v1"
GRAPH_MATCHER_IMPLEMENTATION_VERSION = "kdtree_prefilter_v1"
GRAPH_CANDIDATE_COLUMNS = [
    "idx_a",
    "idx_b",
    "label_a",
    "label_b",
    "candidate_source",
    "distance_um",
    "ambiguity",
    "dice",
    "iou",
    "area_ratio",
    "spatial_term",
    "ambiguity_term",
    "score",
    "high_rule",
    "balanced_rule",
    "base_score",
    "graph_rule",
    "graph_status",
    "match_policy",
    "graph_support_count",
    "graph_support_fraction",
    "graph_residual_median_um",
    "graph_residual_mean_um",
    "graph_residual_p90_um",
    "graph_inlier_fraction",
    "graph_score",
    "refined_score",
    "is_graph_anchor",
]
GRAPH_MATCH_COLUMNS = GRAPH_CANDIDATE_COLUMNS + ["assignment_policy", "assignment_source"]


@dataclass(frozen=True)
class SpatialGraphParams:
    k_neighbors: int = 6
    radius_um: float = 45.0
    min_anchor_support: int = 3
    anchor_min_dice: float = 0.40
    anchor_max_distance_um: float = 5.0
    anchor_min_area_ratio: float = 0.40
    anchor_max_ambiguity: float = 0.65
    anchor_require_overlap_evidence: bool = True
    residual_scale_um: float = 4.0
    inlier_residual_um: float = 6.0
    graph_weight: float = 0.20
    lock_anchors: bool = True
    allow_new_candidates: bool = False
    reject_strong_conflicts: bool = False
    max_iterations: int = 1

    def __post_init__(self) -> None:
        if int(self.k_neighbors) < 1:
            raise ValueError("k_neighbors must be at least 1.")
        if float(self.radius_um) <= 0:
            raise ValueError("radius_um must be positive.")
        if int(self.min_anchor_support) < 1:
            raise ValueError("min_anchor_support must be at least 1.")
        if float(self.residual_scale_um) <= 0:
            raise ValueError("residual_scale_um must be positive.")
        if float(self.inlier_residual_um) <= 0:
            raise ValueError("inlier_residual_um must be positive.")
        if not 0.0 <= float(self.graph_weight) <= 1.0:
            raise ValueError("graph_weight must be within [0, 1].")
        if int(self.max_iterations) != 1:
            raise ValueError("max_iterations must be 1 for v1.")
        if self.allow_new_candidates:
            raise ValueError("allow_new_candidates must be False for v1.")


@dataclass
class GraphPairMatchResult:
    candidates: pd.DataFrame
    anchors: pd.DataFrame
    graph_matches: pd.DataFrame
    changes: pd.DataFrame
    summary: dict[str, object]
    timings_seconds: dict[str, float] | None = None


@dataclass(frozen=True)
class _PreparedAnchorGeometry:
    anchor_coords_a: np.ndarray
    anchor_coords_b: np.ndarray
    tree_a: cKDTree


def _as_int(value: Any) -> int:
    return int(np.asarray(value).item())


def _as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=float))


def _candidate_key_table(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame(columns=["label_a", "label_b"])
    keys = table[["label_a", "label_b"]].copy()
    keys["label_a"] = pd.to_numeric(keys["label_a"], errors="raise").astype(int)
    keys["label_b"] = pd.to_numeric(keys["label_b"], errors="raise").astype(int)
    return keys.drop_duplicates().reset_index(drop=True)


def aligned_physical_coordinates(
    features_a: pd.DataFrame,
    features_b: pd.DataFrame,
    transform: RestrictedTransform,
    spacing: VoxelSpacing,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    spacing_array = spacing.as_zyx_array()
    coords_a = features_a[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float) * spacing_array
    coords_b = transform.apply(features_b[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)) * spacing_array
    coords_a_by_label = {int(label): coords_a[index] for index, label in enumerate(features_a.index.to_numpy(dtype=int))}
    coords_b_by_label = {int(label): coords_b[index] for index, label in enumerate(features_b.index.to_numpy(dtype=int))}
    return coords_a_by_label, coords_b_by_label


def build_spatial_index(labels: np.ndarray, coordinates_um: np.ndarray) -> tuple[cKDTree, dict[int, int]]:
    labels = np.asarray(labels, dtype=int)
    coordinates_um = np.asarray(coordinates_um, dtype=float)
    if len(labels) != len(coordinates_um):
        raise ValueError("labels and coordinates_um must have the same length.")
    return cKDTree(coordinates_um), {int(label): index for index, label in enumerate(labels)}


def select_graph_anchors(high_matches: pd.DataFrame, params: SpatialGraphParams) -> pd.DataFrame:
    if high_matches is None or high_matches.empty:
        columns = list(high_matches.columns) if high_matches is not None else []
        return pd.DataFrame(columns=columns)

    table = high_matches.copy()
    required_columns = {"label_a", "label_b", "dice", "distance_um", "area_ratio", "ambiguity"}
    missing = required_columns.difference(table.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"high_matches is missing required columns: {missing_str}")

    anchor = (
        (pd.to_numeric(table["dice"], errors="coerce") >= float(params.anchor_min_dice))
        & (pd.to_numeric(table["distance_um"], errors="coerce") <= float(params.anchor_max_distance_um))
        & (pd.to_numeric(table["area_ratio"], errors="coerce") >= float(params.anchor_min_area_ratio))
        & (pd.to_numeric(table["ambiguity"], errors="coerce") <= float(params.anchor_max_ambiguity))
    )
    if params.anchor_require_overlap_evidence and "candidate_source" in table.columns:
        anchor &= table["candidate_source"].astype(str).isin({"both", "mutual_overlap"})
    anchors = table.loc[anchor].copy()
    if anchors.empty:
        return anchors.reset_index(drop=True)
    anchors = anchors.sort_values(["score", "dice", "distance_um", "label_a", "label_b"], ascending=[False, False, True, True, True]).reset_index(drop=True)
    anchors["anchor_rank"] = np.arange(len(anchors), dtype=int)
    anchors["is_graph_anchor"] = True
    return anchors


def _local_support_stats(
    *,
    label_a: int,
    label_b: int,
    prepared_geometry: _PreparedAnchorGeometry,
    rough_anchor_indices: np.ndarray,
    coords_a_by_label: dict[int, np.ndarray],
    coords_b_by_label: dict[int, np.ndarray],
    params: SpatialGraphParams,
) -> dict[str, object]:
    anchor_coords_a = prepared_geometry.anchor_coords_a
    anchor_coords_b = prepared_geometry.anchor_coords_b
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

    cand_a = coords_a_by_label[int(label_a)]
    cand_b = coords_b_by_label[int(label_b)]

    rough_anchor_indices = np.asarray(rough_anchor_indices, dtype=int)
    rough_anchor_indices.sort()
    if rough_anchor_indices.size == 0:
        local_indices = np.asarray([], dtype=int)
    else:
        rough_distances_a = np.linalg.norm(anchor_coords_a[rough_anchor_indices] - cand_a, axis=1)
        rough_distances_b = np.linalg.norm(anchor_coords_b[rough_anchor_indices] - cand_b, axis=1)
        local_mask = (rough_distances_a <= float(params.radius_um)) & (rough_distances_b <= float(params.radius_um))
        local_indices = rough_anchor_indices[local_mask]
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

    local_distances_a = rough_distances_a[local_mask]
    local_distances_b = rough_distances_b[local_mask]
    local_a_order = local_indices[np.argsort(local_distances_a)[: min(int(params.k_neighbors), len(local_indices))]]
    local_b_order = local_indices[np.argsort(local_distances_b)[: min(int(params.k_neighbors), len(local_indices))]]
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


def _prepare_anchor_geometry(
    anchor_labels_a: np.ndarray,
    anchor_labels_b: np.ndarray,
    coords_a_by_label: dict[int, np.ndarray],
    coords_b_by_label: dict[int, np.ndarray],
) -> _PreparedAnchorGeometry:
    anchor_coords_a = np.vstack([coords_a_by_label[int(label)] for label in anchor_labels_a])
    anchor_coords_b = np.vstack([coords_b_by_label[int(label)] for label in anchor_labels_b])
    return _PreparedAnchorGeometry(
        anchor_coords_a=anchor_coords_a,
        anchor_coords_b=anchor_coords_b,
        tree_a=cKDTree(anchor_coords_a),
    )


def _anchor_query_radius(radius_um: float) -> float:
    return float(radius_um) + max(1e-12, abs(float(radius_um)) * 1e-12)


def add_graph_consistency_scores(
    candidates: pd.DataFrame,
    anchors: pd.DataFrame,
    coords_a_by_label: dict[int, np.ndarray],
    coords_b_by_label: dict[int, np.ndarray],
    params: SpatialGraphParams,
) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=GRAPH_CANDIDATE_COLUMNS)

    table = candidates.copy()
    n_rows = len(table)
    label_a_values = table["label_a"].to_numpy(dtype=int)
    label_b_values = table["label_b"].to_numpy(dtype=int)
    score_source = table["score"] if "score" in table.columns else pd.Series(np.nan, index=table.index)
    base_score_values = pd.to_numeric(score_source, errors="coerce").to_numpy(dtype=float)
    graph_rule_source = table["balanced_rule"] if "balanced_rule" in table.columns else pd.Series(False, index=table.index)
    graph_rule_values = graph_rule_source.astype(bool).to_numpy()
    graph_positions = np.flatnonzero(graph_rule_values)

    graph_status = np.full(n_rows, "not_balanced", dtype=object)
    graph_support_count = np.zeros(n_rows, dtype=int)
    graph_support_fraction = np.zeros(n_rows, dtype=float)
    graph_residual_median_um = np.full(n_rows, np.nan, dtype=float)
    graph_residual_mean_um = np.full(n_rows, np.nan, dtype=float)
    graph_residual_p90_um = np.full(n_rows, np.nan, dtype=float)
    graph_inlier_fraction = np.full(n_rows, np.nan, dtype=float)
    graph_score = np.full(n_rows, np.nan, dtype=float)
    refined_score = base_score_values.copy()
    is_graph_anchor = np.zeros(n_rows, dtype=bool)

    if anchors is not None and not anchors.empty:
        anchor_pairs = {(int(row.label_a), int(row.label_b)) for row in anchors.itertuples(index=False)}
        anchor_a = anchors["label_a"].astype(int).to_numpy()
        anchor_b = anchors["label_b"].astype(int).to_numpy()
        prepared_geometry = _prepare_anchor_geometry(anchor_a, anchor_b, coords_a_by_label, coords_b_by_label)
        non_anchor_graph_positions = np.asarray(
            [
                position
                for position in graph_positions
                if (int(label_a_values[position]), int(label_b_values[position])) not in anchor_pairs
            ],
            dtype=int,
        )
        rough_neighbors: list[np.ndarray] = []
        if non_anchor_graph_positions.size:
            candidate_coords_a = np.vstack([coords_a_by_label[int(label_a_values[position])] for position in non_anchor_graph_positions])
            rough_neighbors = [
                np.sort(np.asarray(indices, dtype=int))
                for indices in prepared_geometry.tree_a.query_ball_point(
                    candidate_coords_a,
                    r=_anchor_query_radius(float(params.radius_um)),
                    p=2.0,
                    eps=0.0,
                    workers=1,
                )
            ]

        rough_neighbor_position = 0
        for position in graph_positions:
            label_a = int(label_a_values[position])
            label_b = int(label_b_values[position])
            if (label_a, label_b) in anchor_pairs:
                is_graph_anchor[position] = True
                graph_status[position] = "anchor"
                graph_score[position] = 1.0
                refined_score[position] = base_score_values[position]
                continue
            stats = _local_support_stats(
                label_a=label_a,
                label_b=label_b,
                prepared_geometry=prepared_geometry,
                rough_anchor_indices=rough_neighbors[rough_neighbor_position],
                coords_a_by_label=coords_a_by_label,
                coords_b_by_label=coords_b_by_label,
                params=params,
            )
            rough_neighbor_position += 1
            graph_status[position] = stats["graph_status"]
            graph_support_count[position] = int(stats["graph_support_count"])
            graph_support_fraction[position] = float(stats["graph_support_fraction"])
            graph_residual_median_um[position] = float(stats["graph_residual_median_um"])
            graph_residual_mean_um[position] = float(stats["graph_residual_mean_um"])
            graph_residual_p90_um[position] = float(stats["graph_residual_p90_um"])
            graph_inlier_fraction[position] = float(stats["graph_inlier_fraction"])
            graph_score[position] = float(stats["graph_score"])
            if graph_status[position] == "scored":
                refined_score[position] = float(
                    (1.0 - float(params.graph_weight)) * base_score_values[position]
                    + float(params.graph_weight) * float(stats["graph_score"])
                )
    else:
        graph_status[graph_rule_values] = "no_anchor_fallback"

    table["base_score"] = base_score_values
    table["graph_rule"] = graph_rule_values
    table["graph_status"] = graph_status
    table["match_policy"] = "graph"
    table["graph_support_count"] = graph_support_count
    table["graph_support_fraction"] = graph_support_fraction
    table["graph_residual_median_um"] = graph_residual_median_um
    table["graph_residual_mean_um"] = graph_residual_mean_um
    table["graph_residual_p90_um"] = graph_residual_p90_um
    table["graph_inlier_fraction"] = graph_inlier_fraction
    table["graph_score"] = graph_score
    table["refined_score"] = refined_score
    table["is_graph_anchor"] = is_graph_anchor
    return table


def graph_one_to_one_assignment(graph_candidates: pd.DataFrame, anchors: pd.DataFrame, params: SpatialGraphParams) -> pd.DataFrame:
    if graph_candidates is None or graph_candidates.empty:
        return pd.DataFrame(columns=GRAPH_MATCH_COLUMNS)

    table = graph_candidates.copy()
    graph_rule = table.get("graph_rule", table.get("balanced_rule", False)).astype(bool)
    candidates = table.loc[graph_rule].copy()
    if candidates.empty:
        return pd.DataFrame(columns=GRAPH_MATCH_COLUMNS)

    if anchors is None or anchors.empty:
        balanced_mask = candidates["balanced_rule"].astype(bool) if "balanced_rule" in candidates.columns else pd.Series(True, index=candidates.index)
        fallback = candidates.loc[balanced_mask].copy()
        if fallback.empty:
            return pd.DataFrame(columns=GRAPH_MATCH_COLUMNS)
        fallback["assignment_policy"] = "graph"
        fallback["assignment_source"] = "baseline_fallback"
        fallback["match_policy"] = "graph"
        for column, default in [
            ("base_score", fallback.get("score", np.nan)),
            ("graph_rule", fallback.get("balanced_rule", False)),
            ("graph_status", "baseline_fallback"),
            ("graph_support_count", 0),
            ("graph_support_fraction", 0.0),
            ("graph_residual_median_um", np.nan),
            ("graph_residual_mean_um", np.nan),
            ("graph_residual_p90_um", np.nan),
            ("graph_inlier_fraction", np.nan),
            ("graph_score", np.nan),
            ("refined_score", fallback.get("score", np.nan)),
            ("is_graph_anchor", False),
        ]:
            if column not in fallback.columns:
                fallback[column] = default
        return fallback.reset_index(drop=True)

    anchor_pairs = {(int(row.label_a), int(row.label_b)) for row in anchors.itertuples(index=False)}
    locked = candidates.loc[candidates.apply(lambda row: (int(row["label_a"]), int(row["label_b"])) in anchor_pairs, axis=1)].copy()
    locked = locked.sort_values(["anchor_rank", "label_a", "label_b"], ascending=[True, True, True]) if "anchor_rank" in locked.columns else locked.sort_values(["label_a", "label_b"])

    accepted_rows: list[dict[str, object]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for row in locked.itertuples(index=False):
        label_a = int(row.label_a)
        label_b = int(row.label_b)
        if label_a in used_a or label_b in used_b:
            continue
        used_a.add(label_a)
        used_b.add(label_b)
        data = row._asdict()
        data["assignment_policy"] = "graph"
        data["assignment_source"] = "locked_anchor"
        data["match_policy"] = "graph"
        accepted_rows.append(data)

    remaining = candidates.loc[~candidates["label_a"].isin(used_a) & ~candidates["label_b"].isin(used_b)].copy()
    remaining = remaining.sort_values(
        ["refined_score", "graph_support_count", "graph_residual_median_um", "dice", "distance_um", "label_a", "label_b"],
        ascending=[False, False, True, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    for row in remaining.itertuples(index=False):
        label_a = int(row.label_a)
        label_b = int(row.label_b)
        if label_a in used_a or label_b in used_b:
            continue
        used_a.add(label_a)
        used_b.add(label_b)
        data = row._asdict()
        data["assignment_policy"] = "graph"
        data["assignment_source"] = "graph_refined"
        data["match_policy"] = "graph"
        accepted_rows.append(data)

    if not accepted_rows:
        return pd.DataFrame(columns=GRAPH_MATCH_COLUMNS)

    accepted = pd.DataFrame(accepted_rows)
    for column in [
        "base_score",
        "graph_rule",
        "graph_status",
        "graph_support_count",
        "graph_support_fraction",
        "graph_residual_median_um",
        "graph_residual_mean_um",
        "graph_residual_p90_um",
        "graph_inlier_fraction",
        "graph_score",
        "refined_score",
        "is_graph_anchor",
        "assignment_policy",
        "assignment_source",
        "match_policy",
    ]:
        if column not in accepted.columns:
            if column in {"graph_status", "assignment_policy", "assignment_source", "match_policy"}:
                accepted[column] = "graph" if column in {"assignment_policy", "match_policy"} else "graph_refined"
            elif column == "graph_rule":
                accepted[column] = accepted.get("balanced_rule", False)
            elif column == "base_score" or column == "refined_score":
                accepted[column] = accepted.get("score", np.nan)
            elif column == "graph_support_count":
                accepted[column] = 0
            elif column == "graph_support_fraction":
                accepted[column] = 0.0
            elif column == "is_graph_anchor":
                accepted[column] = False
            else:
                accepted[column] = np.nan
    return accepted.reset_index(drop=True)


def compare_balanced_and_graph_matches(balanced_matches: pd.DataFrame, graph_matches: pd.DataFrame) -> pd.DataFrame:
    balanced = _candidate_key_table(balanced_matches)
    graph = _candidate_key_table(graph_matches)
    balanced["in_balanced"] = True
    graph["in_graph"] = True
    merged = balanced.merge(graph, on=["label_a", "label_b"], how="outer")
    merged["in_balanced"] = merged["in_balanced"].fillna(False)
    merged["in_graph"] = merged["in_graph"].fillna(False)
    merged["changed"] = merged["in_balanced"] != merged["in_graph"]
    return merged.sort_values(["label_a", "label_b"]).reset_index(drop=True)


def summarize_graph_pair(
    *,
    session_a: str,
    session_b: str,
    pair_gap: int | None,
    baseline_result: PairMatchResult,
    anchors: pd.DataFrame,
    graph_matches: pd.DataFrame,
    changes: pd.DataFrame,
) -> dict[str, object]:
    summary = dict(baseline_result.summary)
    summary.update(
        {
            "day_a": session_a,
            "day_b": session_b,
            "pair_gap": int(pair_gap) if pair_gap is not None else None,
            "match_policy": "graph",
            "n_graph": int(len(graph_matches)),
            "n_graph_anchors": int(len(anchors)),
            "n_graph_changed": int(changes["changed"].sum()) if not changes.empty else 0,
            "graph_match_policy": "graph",
        }
    )
    return summary


def refine_pair_with_spatial_graph(
    *,
    session_a: str,
    session_b: str,
    baseline_result: PairMatchResult,
    features_a: pd.DataFrame,
    features_b: pd.DataFrame,
    spacing: VoxelSpacing,
    params: SpatialGraphParams | None = None,
    pair_gap: int | None = None,
) -> GraphPairMatchResult:
    params = params or SpatialGraphParams()
    transform = baseline_result.transform
    candidates = baseline_result.candidates.copy()
    high_matches = baseline_result.high_matches.copy()
    balanced_matches = baseline_result.balanced_matches.copy()

    stage_start = time.perf_counter()
    coords_a_by_label, coords_b_by_label = aligned_physical_coordinates(features_a, features_b, transform, spacing)
    coordinate_setup_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    anchors = select_graph_anchors(high_matches, params)
    anchor_selection_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    graph_candidates = add_graph_consistency_scores(candidates, anchors, coords_a_by_label, coords_b_by_label, params)
    graph_support_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    if anchors.empty:
        graph_matches = balanced_matches.copy()
        if graph_matches.empty:
            graph_matches = pd.DataFrame(columns=GRAPH_MATCH_COLUMNS)
        else:
            graph_matches = graph_matches.copy()
            graph_matches["assignment_policy"] = "graph"
            graph_matches["assignment_source"] = "baseline_fallback"
            graph_matches["match_policy"] = "graph"
            graph_matches["base_score"] = graph_matches.get("score", np.nan)
            graph_matches["graph_rule"] = graph_matches.get("balanced_rule", False)
            graph_matches["graph_status"] = "baseline_fallback"
            graph_matches["graph_support_count"] = 0
            graph_matches["graph_support_fraction"] = 0.0
            graph_matches["graph_residual_median_um"] = np.nan
            graph_matches["graph_residual_mean_um"] = np.nan
            graph_matches["graph_residual_p90_um"] = np.nan
            graph_matches["graph_inlier_fraction"] = np.nan
            graph_matches["graph_score"] = np.nan
            graph_matches["refined_score"] = graph_matches.get("score", np.nan)
            graph_matches["is_graph_anchor"] = False
    else:
        graph_matches = graph_one_to_one_assignment(graph_candidates, anchors, params)
    assignment_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    changes = compare_balanced_and_graph_matches(balanced_matches, graph_matches)
    summary = summarize_graph_pair(
        session_a=session_a,
        session_b=session_b,
        pair_gap=pair_gap,
        baseline_result=baseline_result,
        anchors=anchors,
        graph_matches=graph_matches,
        changes=changes,
    )
    comparison_and_summary_seconds = time.perf_counter() - stage_start
    return GraphPairMatchResult(
        candidates=graph_candidates,
        anchors=anchors,
        graph_matches=graph_matches,
        changes=changes,
        summary=summary,
        timings_seconds={
            "coordinate_setup": float(coordinate_setup_seconds),
            "anchor_selection": float(anchor_selection_seconds),
            "graph_support": float(graph_support_seconds),
            "pairwise_assignment": float(assignment_seconds),
            "comparison_and_summary": float(comparison_and_summary_seconds),
        },
    )
