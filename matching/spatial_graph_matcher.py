"""Experimental local spatial-graph refinement for daywise ROI matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from affine_overlap_matcher import PairMatchResult, VoxelSpacing, RestrictedTransform

GRAPH_MATCHER_ALGORITHM_VERSION = "local_spatial_graph_v1"


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
    anchor_table: pd.DataFrame,
    coords_a_by_label: dict[int, np.ndarray],
    coords_b_by_label: dict[int, np.ndarray],
    params: SpatialGraphParams,
) -> dict[str, object]:
    if anchor_table.empty:
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

    anchor_a = anchor_table["label_a"].astype(int).to_numpy()
    anchor_b = anchor_table["label_b"].astype(int).to_numpy()
    anchor_coords_a = np.vstack([coords_a_by_label[int(label)] for label in anchor_a])
    anchor_coords_b = np.vstack([coords_b_by_label[int(label)] for label in anchor_b])
    cand_a = coords_a_by_label[int(label_a)]
    cand_b = coords_b_by_label[int(label_b)]

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


def add_graph_consistency_scores(
    candidates: pd.DataFrame,
    anchors: pd.DataFrame,
    coords_a_by_label: dict[int, np.ndarray],
    coords_b_by_label: dict[int, np.ndarray],
    params: SpatialGraphParams,
) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=list(candidates.columns) if candidates is not None else [])

    table = candidates.copy()
    table["base_score"] = pd.to_numeric(table.get("score", np.nan), errors="coerce")
    table["graph_rule"] = table.get("balanced_rule", False).astype(bool)
    table["graph_status"] = "not_balanced"
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
    for index, row in table.loc[table["graph_rule"]].iterrows():
        label_a = int(row["label_a"])
        label_b = int(row["label_b"])
        if (label_a, label_b) in anchor_pairs:
            table.at[index, "is_graph_anchor"] = True
            table.at[index, "graph_status"] = "anchor"
            table.at[index, "graph_score"] = 1.0
            table.at[index, "refined_score"] = float(row["base_score"])
            continue
        stats = _local_support_stats(
            label_a=label_a,
            label_b=label_b,
            anchor_table=anchors,
            coords_a_by_label=coords_a_by_label,
            coords_b_by_label=coords_b_by_label,
            params=params,
        )
        for key, value in stats.items():
            table.at[index, key] = value
        if table.at[index, "graph_status"] == "scored":
            table.at[index, "refined_score"] = float((1.0 - float(params.graph_weight)) * float(row["base_score"]) + float(params.graph_weight) * float(stats["graph_score"]))
        else:
            table.at[index, "refined_score"] = float(row["base_score"])
    return table


def graph_one_to_one_assignment(graph_candidates: pd.DataFrame, anchors: pd.DataFrame, params: SpatialGraphParams) -> pd.DataFrame:
    if graph_candidates is None or graph_candidates.empty:
        columns = list(graph_candidates.columns) if graph_candidates is not None else []
        return pd.DataFrame(columns=columns)

    table = graph_candidates.copy()
    graph_rule = table.get("graph_rule", table.get("balanced_rule", False)).astype(bool)
    candidates = table.loc[graph_rule].copy()
    if candidates.empty:
        return candidates

    if anchors is None or anchors.empty:
        balanced_mask = candidates["balanced_rule"].astype(bool) if "balanced_rule" in candidates.columns else pd.Series(True, index=candidates.index)
        fallback = candidates.loc[balanced_mask].copy()
        fallback["assignment_policy"] = "graph"
        fallback["assignment_source"] = "baseline_fallback"
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
        accepted_rows.append(data)

    if not accepted_rows:
        balanced_mask = candidates["balanced_rule"].astype(bool) if "balanced_rule" in candidates.columns else pd.Series(True, index=candidates.index)
        fallback = candidates.loc[balanced_mask].copy()
        fallback["assignment_policy"] = "graph"
        fallback["assignment_source"] = "baseline_fallback"
        return fallback.reset_index(drop=True)

    accepted = pd.DataFrame(accepted_rows)
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

    coords_a_by_label, coords_b_by_label = aligned_physical_coordinates(features_a, features_b, transform, spacing)
    anchors = select_graph_anchors(high_matches, params)
    graph_candidates = add_graph_consistency_scores(candidates, anchors, coords_a_by_label, coords_b_by_label, params)
    if anchors.empty:
        graph_matches = balanced_matches.copy()
        if not graph_matches.empty:
            graph_matches = graph_matches.copy()
            graph_matches["assignment_policy"] = "graph"
            graph_matches["assignment_source"] = "baseline_fallback"
    else:
        graph_matches = graph_one_to_one_assignment(graph_candidates, anchors, params)

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
    return GraphPairMatchResult(
        candidates=graph_candidates,
        anchors=anchors,
        graph_matches=graph_matches,
        changes=changes,
        summary=summary,
    )
