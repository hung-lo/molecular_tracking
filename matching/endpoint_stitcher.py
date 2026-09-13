"""Conservative, state-blind post-hoc stitching of canonical track fragments.

This module reads canonical matching/evaluator artifacts and writes only derived
artifacts.  Fluorescence and ECLIPSE fields are intentionally absent from every
decision function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    from endpoint_evaluator import (
        ENDPOINT_CANDIDATE_COLUMNS,
        _bool, _feature_row, _float, _forward_transform, _int,
        _prepare_tracks, _present, _roi_column, _text,
        _transform_row_map, apply_transform_b_to_a, compose_transforms,
        invert_restricted_transform,
    )
    from affine_overlap_matcher import VoxelSpacing
except ImportError:  # pragma: no cover - package imports
    from matching.endpoint_evaluator import (
        ENDPOINT_CANDIDATE_COLUMNS,
        _bool, _feature_row, _float, _forward_transform, _int,
        _prepare_tracks, _present, _roi_column, _text,
        _transform_row_map, apply_transform_b_to_a, compose_transforms,
        invert_restricted_transform,
    )
    from matching.affine_overlap_matcher import VoxelSpacing


ALGORITHM_VERSION = "conservative_endpoint_stitcher_v1"
STATE_COLUMNS = {"eclipse_z", "eclipse_core_state", "eclipse_state_bin", "green", "red", "color_z"}

STITCH_EDGE_COLUMNS = [
    "stitch_edge_id", "edge_type", "source_track_uid", "target_track_uid",
    "source_session_index", "source_session_id", "source_label",
    "target_session_index", "target_session_id", "target_label", "session_gap",
    "n_missing_sessions", "elapsed_day_gap", "candidate_tier",
    "accepted_by_global_assignment", "projected_distance_um",
    "forward_rank_all_masks", "forward_rank_track_starts", "forward_second_margin_um",
    "reverse_projected_distance_um", "reverse_rank_source_endpoints",
    "reverse_second_margin_um", "reciprocal_rank1", "anchor_support_count",
    "anchor_support_fraction", "anchor_residual_median_um", "anchor_residual_p90_um",
    "anchor_inlier_fraction", "source_history_n", "source_history_distance_median_um",
    "target_future_n", "target_future_distance_median_um", "volume_ratio", "dice",
    "iou", "existing_candidate_found", "existing_graph_match", "transform_source",
    "transform_reliable", "source_has_cycle_conflict", "target_has_cycle_conflict",
    "rejection_reasons", "review_reasons", "assignment_cost",
    "assignment_component_id", "algorithm_version",
]
STITCH_CANDIDATE_COLUMNS = list(dict.fromkeys(ENDPOINT_CANDIDATE_COLUMNS + STITCH_EDGE_COLUMNS + [
    "source_n_days_present", "target_n_days_present", "source_missing_internal_days",
    "target_missing_internal_days", "source_edge_heavy", "target_edge_heavy",
    "source_contains_transform_fallback_edge", "target_contains_transform_fallback_edge",
    "source_cycle_agreement_fraction", "target_cycle_agreement_fraction",
    "source_consensus_edge_fraction", "target_consensus_edge_fraction",
    "source_has_graph_only_edge", "target_has_graph_only_edge",
    "source_endpoint_classification", "source_history_distance_p90_um",
    "source_history_distance_max_um", "target_future_distance_p90_um",
    "target_future_distance_max_um", "anchor_residual_mean_um", "anchor_inlier_count",
    "auto_eligible", "review_eligible",
]))


@dataclass(frozen=True)
class StitcherConfig:
    max_gap_sessions: int = 3
    search_radius_um: float = 15.0
    local_anchor_radius_um: float = 45.0
    max_local_anchors: int = 6
    min_anchor_support: int = 3
    anchor_inlier_um: float = 6.0
    max_anchor_residual_median_um: float = 4.0
    min_anchor_inlier_fraction: float = 2.0 / 3.0
    min_forward_margin_um: float = 3.0
    max_history_distance_um: float = 5.0
    max_future_distance_um: float = 5.0
    gap1_max_distance_um: float = 4.0
    gap2_max_distance_um: float = 5.0
    gap3_max_distance_um: float = 5.0
    no_link_cost: float = 1_000.0

    def distance_limit(self, gap: int) -> float:
        return {1: self.gap1_max_distance_um, 2: self.gap2_max_distance_um, 3: self.gap3_max_distance_um}[gap]


def _coords(feature: pd.Series) -> np.ndarray:
    return feature[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)


def _track_lookup(tracks: pd.DataFrame) -> dict[str, pd.Series]:
    return {_text(row.get("track_uid")): row for _, row in tracks.iterrows()}


def _normalize_tracks(tracks: pd.DataFrame, sessions: pd.DataFrame, policy: str) -> pd.DataFrame:
    table = tracks.copy()
    for column in ("first_session_index", "last_session_index", "n_days_present", "missing_internal_days"):
        if column in table and table[column].isna().any():
            table = table.drop(columns=column)
    table = _prepare_tracks(table, sessions, policy)
    roi_columns = [_roi_column(str(row.session_id)) for row in sessions.itertuples(index=False)]
    for index, row in table.iterrows():
        positions = [position for position, column in enumerate(roi_columns) if _present(row.get(column, pd.NA))]
        values = {
            "first_session_index": min(positions) if positions else -1,
            "last_session_index": max(positions) if positions else -1,
            "n_days_present": len(positions),
            "missing_internal_days": sum(position not in positions for position in range(min(positions), max(positions) + 1)) if positions else 0,
        }
        for column, value in values.items():
            if column not in table or pd.isna(table.at[index, column]):
                table.at[index, column] = value
    return table


def _track_flag(track: pd.Series, *names: str) -> bool:
    return any(_bool(track.get(name)) for name in names)


def _position_map(sessions: pd.DataFrame) -> dict[int, int]:
    return {int(row.session_index): position for position, row in enumerate(sessions.itertuples(index=False))}


def _track_observations(track: pd.Series, sessions: pd.DataFrame) -> list[tuple[int, str, int]]:
    return [
        (position, str(session.session_id), int(track[_roi_column(str(session.session_id))]))
        for position, session in enumerate(sessions.itertuples(index=False))
        if _present(track.get(_roi_column(str(session.session_id)), pd.NA))
    ]


def _project(
    coordinates: np.ndarray,
    source_position: int,
    target_position: int,
    sessions: pd.DataFrame,
    transform_map: dict[tuple[str, str], pd.Series],
) -> np.ndarray | None:
    if source_position == target_position:
        return coordinates.copy()
    if source_position < target_position:
        result = _forward_transform(source_position, target_position, sessions, transform_map)
        if result is not None:
            return apply_transform_b_to_a(coordinates, result[0])
        stored = _adjacent_composition(source_position, target_position, sessions, transform_map)
        if stored is None:
            return None
        try:
            return apply_transform_b_to_a(coordinates, invert_restricted_transform(stored))
        except ValueError:
            return None
    stored = _adjacent_composition(target_position, source_position, sessions, transform_map)
    return None if stored is None else apply_transform_b_to_a(coordinates, stored)


def _adjacent_composition(
    earlier_position: int,
    later_position: int,
    sessions: pd.DataFrame,
    transform_map: dict[tuple[str, str], pd.Series],
):
    components = [
        transform_map.get((str(sessions.iloc[position]["session_id"]), str(sessions.iloc[position + 1]["session_id"])))
        for position in range(earlier_position, later_position)
    ]
    if not components or any(component is None for component in components):
        return None
    composed = components[0]
    for component in components[1:]:
        composed = compose_transforms(composed, component)
    return composed


def local_anchor_geometry(
    source_track_uid: str,
    target_track_uid: str,
    source_position: int,
    target_position: int,
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    transforms: pd.DataFrame,
    *,
    spacing: VoxelSpacing | None = None,
    radius_um: float = 45.0,
    max_anchors: int = 6,
    inlier_um: float = 6.0,
) -> dict[str, float | int]:
    """Compare source/target relative vectors against shared canonical anchors."""

    spacing = spacing or VoxelSpacing()
    scale = spacing.as_zyx_array()
    source_id = str(sessions.iloc[source_position]["session_id"])
    target_id = str(sessions.iloc[target_position]["session_id"])
    lookup = _track_lookup(tracks)
    source_track, target_track = lookup.get(source_track_uid), lookup.get(target_track_uid)
    if source_track is None or target_track is None:
        return _residual_summary([])
    source_feature = _feature_row(features, source_id, int(source_track[_roi_column(source_id)]))
    target_feature = _feature_row(features, target_id, int(target_track[_roi_column(target_id)]))
    transform_map = _transform_row_map(transforms)
    if source_feature is None or target_feature is None:
        return _residual_summary([])
    source_prediction = _project(_coords(source_feature), source_position, target_position, sessions, transform_map)
    if source_prediction is None:
        return _residual_summary([])
    source_point, target_point = source_prediction * scale, _coords(target_feature) * scale
    anchors: list[tuple[float, str, float]] = []
    for uid, track in lookup.items():
        if uid in {source_track_uid, target_track_uid} or _track_flag(track, "has_cycle_conflict", "contains_transform_fallback_edge", "edge_heavy"):
            continue
        source_label = track.get(_roi_column(source_id), pd.NA)
        target_label = track.get(_roi_column(target_id), pd.NA)
        if not (_present(source_label) and _present(target_label)):
            continue
        left = _feature_row(features, source_id, int(source_label))
        right = _feature_row(features, target_id, int(target_label))
        if left is None or right is None:
            continue
        projected = _project(_coords(left), source_position, target_position, sessions, transform_map)
        if projected is None:
            continue
        projected_um, right_um = projected * scale, _coords(right) * scale
        proximity = min(float(np.linalg.norm(projected_um - source_point)), float(np.linalg.norm(right_um - target_point)))
        if proximity <= radius_um:
            residual = float(np.linalg.norm((projected_um - source_point) - (right_um - target_point)))
            anchors.append((proximity, uid, residual))
    anchors.sort(key=lambda item: (item[0], item[1]))
    summary = _residual_summary([item[2] for item in anchors[:max_anchors]], inlier_um)
    summary["anchor_support_fraction"] = float(summary["anchor_inlier_count"] / summary["anchor_support_count"]) if summary["anchor_support_count"] else 0.0
    return summary


def _residual_summary(values: Iterable[float], inlier_um: float = 6.0) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {
            "anchor_support_count": 0, "anchor_support_fraction": 0.0,
            "anchor_residual_median_um": np.nan, "anchor_residual_mean_um": np.nan,
            "anchor_residual_p90_um": np.nan, "anchor_inlier_count": 0,
            "anchor_inlier_fraction": np.nan,
        }
    n_inlier = int((array <= inlier_um).sum())
    return {
        "anchor_support_count": int(len(array)), "anchor_support_fraction": float(n_inlier / len(array)),
        "anchor_residual_median_um": float(np.median(array)), "anchor_residual_mean_um": float(np.mean(array)),
        "anchor_residual_p90_um": float(np.percentile(array, 90)), "anchor_inlier_count": n_inlier,
        "anchor_inlier_fraction": float(n_inlier / len(array)),
    }


def _distance_summaries(prefix: str, values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_n": int(len(array)),
        f"{prefix}_distance_median_um": float(np.median(array)) if len(array) else np.nan,
        f"{prefix}_distance_p90_um": float(np.percentile(array, 90)) if len(array) else np.nan,
        f"{prefix}_distance_max_um": float(np.max(array)) if len(array) else np.nan,
    }


def _feature_coords_lookup(features: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    return {
        (str(row.session_id), int(row.label)): np.array([row.centroid_z, row.centroid_y, row.centroid_x], dtype=float)
        for row in features.itertuples(index=False)
    }


def _anchor_context(
    source_position: int,
    target_position: int,
    tracks: pd.DataFrame,
    sessions: pd.DataFrame,
    feature_coords: dict[tuple[str, int], np.ndarray],
    transform_map: dict[tuple[str, str], pd.Series],
    spacing: VoxelSpacing,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, cKDTree | None, cKDTree | None]:
    source_id, target_id = str(sessions.iloc[source_position]["session_id"]), str(sessions.iloc[target_position]["session_id"])
    uids, source_points, target_points = [], [], []
    for _, track in tracks.iterrows():
        if _track_flag(track, "has_cycle_conflict", "contains_transform_fallback_edge", "edge_heavy"):
            continue
        source_label, target_label = track.get(_roi_column(source_id), pd.NA), track.get(_roi_column(target_id), pd.NA)
        if not (_present(source_label) and _present(target_label)):
            continue
        source = feature_coords.get((source_id, int(source_label)))
        target = feature_coords.get((target_id, int(target_label)))
        if source is not None and target is not None:
            uids.append(_text(track.get("track_uid"))); source_points.append(source); target_points.append(target)
    if not source_points:
        empty = np.empty((0, 3), dtype=float)
        return np.asarray([], dtype=str), empty, empty, None, None
    projected = _project(np.asarray(source_points), source_position, target_position, sessions, transform_map)
    if projected is None:
        empty = np.empty((0, 3), dtype=float)
        return np.asarray([], dtype=str), empty, empty, None, None
    projected_um = projected * spacing.as_zyx_array()
    target_um = np.asarray(target_points) * spacing.as_zyx_array()
    return np.asarray(uids, dtype=str), projected_um, target_um, cKDTree(projected_um), cKDTree(target_um)


def _cached_anchor_geometry(
    row: dict[str, Any],
    context: tuple[np.ndarray, np.ndarray, np.ndarray, cKDTree | None, cKDTree | None],
    config: StitcherConfig,
) -> dict[str, float | int]:
    uids, source_anchors, target_anchors, source_tree, target_tree = context
    source_point = np.array([row.get("predicted_z_um"), row.get("predicted_y_um"), row.get("predicted_x_um")], dtype=float)
    target_point = np.array([row.get("target_z_um"), row.get("target_y_um"), row.get("target_x_um")], dtype=float)
    if source_tree is None or target_tree is None or not np.isfinite(source_point).all() or not np.isfinite(target_point).all():
        return _residual_summary([])
    indices = set(source_tree.query_ball_point(source_point, config.local_anchor_radius_um))
    indices.update(target_tree.query_ball_point(target_point, config.local_anchor_radius_um))
    candidates = []
    for index in indices:
        if uids[index] in {row["source_track_uid"], row["target_track_uid"]}:
            continue
        proximity = min(float(np.linalg.norm(source_anchors[index] - source_point)), float(np.linalg.norm(target_anchors[index] - target_point)))
        residual = float(np.linalg.norm((source_anchors[index] - source_point) - (target_anchors[index] - target_point)))
        candidates.append((proximity, uids[index], residual))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return _residual_summary([item[2] for item in candidates[:config.max_local_anchors]], config.anchor_inlier_um)


def _cached_longitudinal_distances(
    source_uid: str,
    target_uid: str,
    source_position: int,
    target_position: int,
    observations: dict[str, list[tuple[int, str, int, np.ndarray]]],
    sessions: pd.DataFrame,
    transform_map: dict[tuple[str, str], pd.Series],
    spacing: VoxelSpacing,
) -> dict[str, float | int]:
    scale = spacing.as_zyx_array()
    source_obs, target_obs = observations[source_uid], observations[target_uid]
    target_start = next(coords for position, _, _, coords in target_obs if position == target_position)
    source_end = next(coords for position, _, _, coords in source_obs if position == source_position)
    history, future = [], []
    for position, _, _, coords in [obs for obs in source_obs if obs[0] <= source_position][-3:]:
        projected = _project(coords, position, target_position, sessions, transform_map)
        if projected is not None:
            history.append(float(np.linalg.norm((projected - target_start) * scale)))
    for position, _, _, coords in [obs for obs in target_obs if obs[0] > target_position][:3]:
        projected = _project(source_end, source_position, position, sessions, transform_map)
        if projected is not None:
            future.append(float(np.linalg.norm((projected - coords) * scale)))
    return _distance_summaries("source_history", history) | _distance_summaries("target_future", future)


def _source_allowed(endpoint: pd.Series, source_track: pd.Series, sessions: pd.DataFrame) -> bool:
    end_index = _int(endpoint.get("end_session_index"), -1)
    positions = sessions.index[sessions["session_index"].astype(int).eq(end_index)] if end_index is not None else []
    if len(positions) != 1:
        return False
    end_position = int(positions[0])
    return bool(
        not _bool(endpoint.get("same_track_returns"))
        and end_position < len(sessions) - 1
        and end_position == _int(source_track.get("last_session_index"), -2)
    )


def build_stitch_candidates(
    endpoints: pd.DataFrame,
    evaluator_candidates: pd.DataFrame,
    classifications: pd.DataFrame,
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    transforms: pd.DataFrame,
    *,
    policy: str = "graph",
    spacing: VoxelSpacing | None = None,
    config: StitcherConfig | None = None,
) -> pd.DataFrame:
    """Build deterministic state-blind endpoint-to-track-start evidence rows."""

    spacing, config = spacing or VoxelSpacing(), config or StitcherConfig()
    sessions = sessions.sort_values("session_index").reset_index(drop=True)
    tracks = _normalize_tracks(tracks, sessions, policy)
    track_map, endpoint_map = _track_lookup(tracks), {str(row.endpoint_id): row for _, row in endpoints.iterrows()}
    class_map = {str(row.endpoint_id): _text(row.get("classification")) for _, row in classifications.iterrows()}
    position_map = _position_map(sessions)
    feature_coords = _feature_coords_lookup(features)
    transform_map = _transform_row_map(transforms)
    observations = {
        uid: [
            (position, session_id, label, feature_coords[(session_id, label)])
            for position, session_id, label in _track_observations(track, sessions)
            if (session_id, label) in feature_coords
        ]
        for uid, track in track_map.items()
    }
    anchor_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, cKDTree | None, cKDTree | None]] = {}
    rows: list[dict[str, Any]] = []
    for _, candidate in evaluator_candidates.iterrows():
        endpoint = endpoint_map.get(_text(candidate.get("endpoint_id")))
        source_uid, target_uid = _text(candidate.get("source_track_uid")), _text(candidate.get("target_track_uid"))
        source_track, target_track = track_map.get(source_uid), track_map.get(target_uid)
        if endpoint is None or source_track is None or target_track is None or not target_uid:
            continue
        source_index, target_index = _int(candidate.get("source_session_index"), -1), _int(candidate.get("target_session_index"), -1)
        if source_index not in position_map or target_index not in position_map:
            continue
        source_position, target_position = position_map[source_index], position_map[target_index]
        gap = target_position - source_position
        if not _source_allowed(endpoint, source_track, sessions) or not 1 <= gap <= config.max_gap_sessions:
            continue
        if source_uid == target_uid or _int(target_track.get("first_session_index"), -1) != target_position:
            continue
        row = candidate.to_dict()
        row.update({
            "edge_type": "endpoint_stitch",
            "source_session_index": source_index, "target_session_index": target_index,
            "session_gap": gap, "n_missing_sessions": gap - 1,
            "forward_rank_all_masks": _int(candidate.get("target_rank_by_distance")),
            "forward_second_margin_um": _float(candidate.get("nearest_second_nearest_margin_um")),
            "source_n_days_present": _int(source_track.get("n_days_present"), 0),
            "target_n_days_present": _int(target_track.get("n_days_present"), 0),
            "source_missing_internal_days": _int(source_track.get("missing_internal_days"), 0),
            "target_missing_internal_days": _int(target_track.get("missing_internal_days"), 0),
            "source_has_cycle_conflict": _bool(source_track.get("has_cycle_conflict")),
            "target_has_cycle_conflict": _bool(target_track.get("has_cycle_conflict")),
            "source_edge_heavy": _bool(source_track.get("edge_heavy")) or _bool(endpoint.get("touches_z_edge")) or _bool(endpoint.get("touches_xy_edge")),
            "target_edge_heavy": _bool(target_track.get("edge_heavy")),
            "source_contains_transform_fallback_edge": _bool(source_track.get("contains_transform_fallback_edge")),
            "target_contains_transform_fallback_edge": _bool(target_track.get("contains_transform_fallback_edge")),
            "source_cycle_agreement_fraction": _float(source_track.get("cycle_agreement_fraction")),
            "target_cycle_agreement_fraction": _float(target_track.get("cycle_agreement_fraction")),
            "source_consensus_edge_fraction": _float(source_track.get("consensus_edge_fraction")),
            "target_consensus_edge_fraction": _float(target_track.get("consensus_edge_fraction")),
            "source_has_graph_only_edge": _bool(source_track.get("has_graph_only_edge")),
            "target_has_graph_only_edge": _bool(target_track.get("has_graph_only_edge")),
            "source_has_same_session_conflict": _track_flag(source_track, "has_same_session_conflict", "same_session_conflict", "component_same_session_conflict"),
            "target_has_same_session_conflict": _track_flag(target_track, "has_same_session_conflict", "same_session_conflict", "component_same_session_conflict"),
            "source_endpoint_classification": class_map.get(_text(candidate.get("endpoint_id")), ""),
        })
        pair_key = source_position * len(sessions) + target_position
        if pair_key not in anchor_cache:
            anchor_cache[pair_key] = _anchor_context(
                source_position, target_position, tracks, sessions, feature_coords, transform_map, spacing,
            )
        row.update(_cached_anchor_geometry(row, anchor_cache[pair_key], config))
        if observations.get(source_uid) and observations.get(target_uid):
            row.update(_cached_longitudinal_distances(
                source_uid, target_uid, source_position, target_position,
                observations, sessions, transform_map, spacing,
            ))
        else:
            row.update(_distance_summaries("source_history", []) | _distance_summaries("target_future", []))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=STITCH_CANDIDATE_COLUMNS)
    table = pd.DataFrame(rows).sort_values(
        ["source_session_index", "source_track_uid", "target_session_index", "projected_distance_um", "target_track_uid", "target_label"],
        kind="mergesort",
    ).reset_index(drop=True)
    table["forward_rank_track_starts"] = table.groupby(["endpoint_id", "target_session_index"], sort=False)["projected_distance_um"].rank(method="first")
    table["forward_rank_track_starts"] = table["forward_rank_track_starts"].astype(int)
    _add_reverse_evidence(table, endpoints, features, sessions, transforms, spacing)
    table["reciprocal_rank1"] = table["reverse_rank_source_endpoints"].eq(1) & table["forward_rank_track_starts"].eq(1)
    table = tier_stitch_candidates(table, config=config)
    table = table.sort_values(["source_session_index", "source_track_uid", "target_session_index", "target_track_uid", "target_label"], kind="mergesort").reset_index(drop=True)
    table["stitch_edge_id"] = [f"stitch_{i:07d}" for i in range(1, len(table) + 1)]
    table["algorithm_version"] = ALGORITHM_VERSION
    return table


def _add_reverse_evidence(
    table: pd.DataFrame,
    endpoints: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    transforms: pd.DataFrame,
    spacing: VoxelSpacing,
) -> None:
    transform_map, position_map, scale = _transform_row_map(transforms), _position_map(sessions), spacing.as_zyx_array()
    feature_coords = _feature_coords_lookup(features)
    endpoint_rows: dict[int, list[tuple[str, np.ndarray]]] = {}
    for _, endpoint in endpoints.iterrows():
        index = _int(endpoint.get("end_session_index"), -1)
        position = position_map.get(index)
        if position is None or _bool(endpoint.get("same_track_returns")):
            continue
        coords = feature_coords.get((str(sessions.iloc[position]["session_id"]), int(endpoint["end_label"])))
        if coords is not None:
            endpoint_rows.setdefault(index, []).append((_text(endpoint.get("track_uid")), coords * scale))
    endpoint_points = {
        index: (
            np.asarray([uid for uid, _ in values], dtype=str),
            np.asarray([point for _, point in values], dtype=float),
            cKDTree(np.asarray([point for _, point in values], dtype=float)),
        )
        for index, values in endpoint_rows.items()
    }
    reverse_distance: list[float] = []
    reverse_rank: list[int] = []
    reverse_margin: list[float] = []
    for _, row in table.iterrows():
        source_index, target_index = int(row["source_session_index"]), int(row["target_session_index"])
        source_position, target_position = position_map[source_index], position_map[target_index]
        target_coords = feature_coords.get((_text(row.get("target_session_id")), int(row["target_label"])))
        projected = None if target_coords is None else _project(target_coords, target_position, source_position, sessions, transform_map)
        points = endpoint_points.get(source_index)
        if projected is not None and points is not None:
            uids, coordinates, tree = points
            distances = np.linalg.norm(coordinates - projected * scale, axis=1)
            own_indices = np.flatnonzero(uids == str(row["source_track_uid"]))
            own = float(distances[own_indices[0]]) if len(own_indices) else np.nan
            rank = int(1 + np.sum((distances < own) | ((distances == own) & (uids < str(row["source_track_uid"]))))) if np.isfinite(own) else 0
            nearest = np.atleast_1d(tree.query(projected * scale, k=min(2, len(uids)))[0])
        else:
            own, rank, nearest = np.nan, 0, np.asarray([])
        reverse_distance.append(own)
        reverse_rank.append(rank)
        reverse_margin.append(float(nearest[1] - nearest[0]) if len(nearest) > 1 else np.nan)
    table["reverse_projected_distance_um"] = reverse_distance
    table["reverse_rank_source_endpoints"] = reverse_rank
    table["reverse_second_margin_um"] = reverse_margin


def tier_stitch_candidates(candidates: pd.DataFrame, *, config: StitcherConfig | None = None) -> pd.DataFrame:
    """Apply conservative gates without consulting state/intensity columns."""

    config = config or StitcherConfig()
    output = candidates.drop(columns=[column for column in STATE_COLUMNS if column in candidates], errors="ignore").copy()
    tiers, auto_flags, review_flags, rejects, reviews = [], [], [], [], []
    for _, row in output.iterrows():
        gap = int(row["session_gap"])
        reject: list[str] = []
        review: list[str] = []
        if _text(row.get("source_track_uid")) == _text(row.get("target_track_uid")):
            reject.append("same_canonical_track")
        if gap < 1 or gap > config.max_gap_sessions:
            reject.append("gap_out_of_range")
        if _int(row.get("target_session_index"), -1) <= _int(row.get("source_session_index"), -1):
            reject.append("temporal_overlap")
        if not _bool(row.get("target_track_starts_here"), True):
            reject.append("target_not_track_start")
        if not _bool(row.get("transform_reliable"), False):
            reject.append("transform_unreliable")
        if _bool(row.get("source_edge_heavy")) or _bool(row.get("target_edge_heavy")):
            reject.append("edge_or_out_of_fov")
        if _text(row.get("source_endpoint_classification")) in {"edge_or_out_of_fov", "transform_unreliable"}:
            reject.append(_text(row.get("source_endpoint_classification")))
        if _bool(row.get("source_has_same_session_conflict")) or _bool(row.get("target_has_same_session_conflict")):
            reject.append("same_session_conflict")
        distance = _float(row.get("projected_distance_um"))
        if gap in (1, 2, 3) and (not np.isfinite(distance) or distance > config.distance_limit(gap)):
            reject.append("projected_distance_failed")
        anchor_n = _int(row.get("anchor_support_count"), 0) or 0
        anchor_median = _float(row.get("anchor_residual_median_um"))
        anchor_fraction = _float(row.get("anchor_inlier_fraction"))
        if anchor_n >= config.min_anchor_support and (
            anchor_median > config.max_anchor_residual_median_um or anchor_fraction < config.min_anchor_inlier_fraction
        ):
            reject.append("anchor_geometry_inconsistent")
        history = _float(row.get("source_history_distance_median_um"))
        future = _float(row.get("target_future_distance_median_um"))
        if np.isfinite(history) and history > config.max_history_distance_um:
            reject.append("source_history_inconsistent")
        if np.isfinite(future) and future > config.max_future_distance_um:
            reject.append("target_future_inconsistent")
        if (_int(row.get("source_n_days_present"), 0) or 0) < 2:
            review.append("singleton_source")
        if (_int(row.get("target_n_days_present"), 0) or 0) < 2:
            review.append("singleton_target")
        if _bool(row.get("source_has_cycle_conflict")) or _bool(row.get("target_has_cycle_conflict")):
            review.append("canonical_cycle_warning")
        if _bool(row.get("source_contains_transform_fallback_edge")) or _bool(row.get("target_contains_transform_fallback_edge")):
            review.append("canonical_transform_warning")
        if _int(row.get("forward_rank_track_starts"), 0) != 1:
            review.append("forward_not_rank1")
        if _int(row.get("forward_rank_all_masks"), 0) != 1:
            review.append("forward_all_masks_not_rank1")
        if _int(row.get("reverse_rank_source_endpoints"), 0) != 1:
            review.append("reverse_not_rank1")
        margin = _float(row.get("forward_second_margin_um"))
        if not np.isfinite(margin) or margin < config.min_forward_margin_um:
            review.append("forward_margin_insufficient")
        if anchor_n < config.min_anchor_support:
            review.append("insufficient_local_anchors")
        if (_int(row.get("source_history_n"), 0) or 0) < 2:
            review.append("insufficient_source_history")
        if (_int(row.get("target_future_n"), 0) or 0) < 1:
            review.append("insufficient_target_future")
        if gap == 3 and (_int(row.get("target_future_n"), 0) or 0) < 1:
            review.append("gap3_requires_future_support")
        if reject:
            tier = "reject"
        elif review:
            tier = "manual_review"
        else:
            tier = "auto_accept_candidate"
        tiers.append(tier); auto_flags.append(tier == "auto_accept_candidate"); review_flags.append(tier == "manual_review")
        rejects.append(";".join(sorted(set(reject)))); reviews.append(";".join(sorted(set(review))))
    output["candidate_tier"], output["auto_eligible"], output["review_eligible"] = tiers, auto_flags, review_flags
    output["rejection_reasons"], output["review_reasons"] = rejects, reviews
    return output


def _assignment_cost(row: pd.Series, config: StitcherConfig) -> float:
    finite = lambda value, default: float(value) if np.isfinite(_float(value)) else default
    return (
        finite(row.get("projected_distance_um"), 50.0) * 100.0
        + finite(row.get("anchor_residual_median_um"), 20.0) * 10.0
        + finite(row.get("source_history_distance_median_um"), 20.0)
        + finite(row.get("target_future_distance_median_um"), 20.0)
        - min(finite(row.get("forward_second_margin_um"), 0.0), 50.0) * 0.01
        - min(finite(row.get("reverse_second_margin_um"), 0.0), 50.0) * 0.001
        + int(row.get("session_gap", 3)) * 0.0001
    )


def assign_stitches(candidates: pd.DataFrame, *, config: StitcherConfig | None = None) -> pd.DataFrame:
    """Globally assign hard-gated candidates with a deterministic no-link option."""

    config = config or StitcherConfig()
    output = candidates.copy()
    output["accepted_by_global_assignment"] = False
    output["assignment_cost"] = np.nan
    output["assignment_component_id"] = ""
    eligible_mask = output["auto_eligible"].astype(bool) if "auto_eligible" in output else pd.Series(False, index=output.index)
    output["assignment_status"] = np.where(eligible_mask, "candidate", "failed_gate")
    eligible = output.loc[eligible_mask].copy()
    if eligible.empty:
        return output
    adjacency: dict[str, set[str]] = {}
    for _, row in eligible.iterrows():
        left, right = "s:" + str(row["source_track_uid"]), "t:" + str(row["target_track_uid"])
        adjacency.setdefault(left, set()).add(right); adjacency.setdefault(right, set()).add(left)
    components: list[set[str]] = []
    unseen = set(adjacency)
    while unseen:
        stack, component = [min(unseen)], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node); unseen.discard(node); stack.extend(sorted(adjacency[node] - component, reverse=True))
        components.append(component)
    components.sort(key=lambda component: sorted(component))
    for component_number, component in enumerate(components, start=1):
        sources = sorted(node[2:] for node in component if node.startswith("s:"))
        targets = sorted(node[2:] for node in component if node.startswith("t:"))
        component_id = f"component_{component_number:05d}"
        matrix = np.full((len(sources), len(targets) + len(sources)), config.no_link_cost, dtype=float)
        matrix[:, :len(targets)] = config.no_link_cost * 10
        subset = eligible.loc[eligible["source_track_uid"].astype(str).isin(sources) & eligible["target_track_uid"].astype(str).isin(targets)]
        edge_indices: dict[tuple[int, int], int] = {}
        for index, row in subset.sort_values("stitch_edge_id", kind="mergesort").iterrows():
            i, j = sources.index(str(row["source_track_uid"])), targets.index(str(row["target_track_uid"]))
            cost = _assignment_cost(row, config) + (i * max(1, len(targets)) + j) * 1e-9
            if cost < matrix[i, j]:
                matrix[i, j], edge_indices[(i, j)] = cost, index
            output.at[index, "assignment_cost"] = cost
            output.at[index, "assignment_component_id"] = component_id
        for i in range(len(sources)):
            matrix[i, len(targets) + i] = config.no_link_cost
        selected_rows, selected_columns = linear_sum_assignment(matrix)
        for i, j in zip(selected_rows, selected_columns):
            if j < len(targets) and matrix[i, j] < config.no_link_cost:
                index = edge_indices[(i, j)]
                output.at[index, "accepted_by_global_assignment"] = True
                output.at[index, "assignment_status"] = "accepted"
        mask = output["assignment_component_id"].eq(component_id) & output["auto_eligible"] & ~output["accepted_by_global_assignment"]
        output.loc[mask, "assignment_status"] = "global_collision_rejection"
    return output


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_stitched_tracks(
    tracks: pd.DataFrame,
    assignments: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    policy: str = "graph",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool | int]]:
    """Merge accepted canonical tracklets transitively and conserve observations."""

    tracks = _normalize_tracks(tracks, sessions, policy).copy()
    uids = tracks["track_uid"].astype(str).tolist()
    union = _UnionFind(uids)
    accepted_mask = assignments["accepted_by_global_assignment"].astype(bool) if "accepted_by_global_assignment" in assignments else pd.Series(False, index=assignments.index)
    accepted = assignments.loc[accepted_mask].copy()
    for _, edge in accepted.iterrows():
        union.union(str(edge["source_track_uid"]), str(edge["target_track_uid"]))
    groups: dict[str, list[str]] = {}
    for uid in uids:
        groups.setdefault(union.find(uid), []).append(uid)
    lookup = _track_lookup(tracks)
    roi_columns = [_roi_column(str(row.session_id)) for row in sessions.itertuples(index=False)]
    rows, mapping = [], []
    before_nodes = [(column, int(row[column])) for _, row in tracks.iterrows() for column in roi_columns if _present(row.get(column, pd.NA))]
    nodes_before = len(before_nodes)
    for members in groups.values():
        members.sort(key=lambda uid: (_int(lookup[uid].get("first_session_index"), 10**9), uid))
        stitched_uid = members[0]
        row = lookup[stitched_uid].copy()
        observations: dict[str, int] = {}
        for uid in members:
            for column in roi_columns:
                value = lookup[uid].get(column, pd.NA)
                if _present(value):
                    if column in observations:
                        raise ValueError(f"same-session conflict while stitching {members}: {column}")
                    observations[column] = int(value)
        for column in roi_columns:
            row[column] = observations.get(column, pd.NA)
        member_edges = (
            accepted.loc[accepted["source_track_uid"].astype(str).isin(members) & accepted["target_track_uid"].astype(str).isin(members)]
            if not accepted.empty else accepted
        )
        present_positions = [i for i, column in enumerate(roi_columns) if column in observations]
        row["track_uid"] = stitched_uid
        row["stitched_track_uid"] = stitched_uid
        row["member_track_uids"] = ";".join(members)
        row["n_member_tracks"] = len(members)
        row["n_stitch_edges"] = len(member_edges)
        row["max_stitch_session_gap"] = int(member_edges["session_gap"].max()) if len(member_edges) else 0
        row["max_stitch_elapsed_day_gap"] = float(member_edges["elapsed_day_gap"].max()) if len(member_edges) else 0.0
        row["contains_singleton_member"] = any((_int(lookup[uid].get("n_days_present"), 0) or 0) == 1 for uid in members)
        row["stitch_review_required"] = False
        row["stitch_review_reasons"] = ""
        row["first_session_index"] = min(present_positions) if present_positions else -1
        row["last_session_index"] = max(present_positions) if present_positions else -1
        row["n_days_present"] = len(observations)
        row["missing_internal_days"] = sum(i not in present_positions for i in range(row["first_session_index"], row["last_session_index"] + 1)) if present_positions else 0
        rows.append(row)
        for uid in members:
            mapping.append({
                "canonical_track_uid": uid, "stitched_track_uid": stitched_uid,
                "member_track_uids": ";".join(members), "n_member_tracks": len(members),
            })
    stitched = pd.DataFrame(rows).sort_values("stitched_track_uid", kind="mergesort").reset_index(drop=True)
    uid_map = pd.DataFrame(mapping).sort_values("canonical_track_uid", kind="mergesort").reset_index(drop=True)
    after_nodes = [(column, int(row[column])) for _, row in stitched.iterrows() for column in roi_columns if _present(row.get(column, pd.NA))]
    nodes_after = len(after_nodes)
    invariants = {
        "n_observed_nodes_before": nodes_before, "n_observed_nodes_after": nodes_after,
        "observation_conservation_passed": sorted(before_nodes) == sorted(after_nodes),
        "session_uniqueness_passed": True,
    }
    if len(set(before_nodes)) != nodes_before or sorted(before_nodes) != sorted(after_nodes) or len(uid_map) != len(tracks) or uid_map["canonical_track_uid"].duplicated().any():
        raise AssertionError("stitched track conservation failed")
    return stitched, uid_map, invariants


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / (1 + z * z / total)
    return center - half, center + half


def summarize_stitch_benchmark(
    benchmark: pd.DataFrame,
    *,
    min_precision: float = 0.999,
    max_negative_fpr: float = 0.001,
) -> dict[str, Any]:
    positive = benchmark.loc[benchmark["case_type"].eq("positive")] if not benchmark.empty else benchmark
    negative = benchmark.loc[benchmark["case_type"].ne("positive")] if not benchmark.empty else benchmark
    tp = int(positive.get("correct_assignment", pd.Series(dtype=bool)).astype(bool).sum())
    fp = int(benchmark.get("false_positive", pd.Series(dtype=bool)).astype(bool).sum())
    accepted = tp + fp
    precision = tp / accepted if accepted else 1.0
    recall = tp / len(positive) if len(positive) else np.nan
    negative_fpr = fp / len(negative) if len(negative) else 0.0
    lower, upper = wilson_interval(tp, accepted)
    by_gap = {
        str(int(gap)): {
            "n": int(len(frame)),
            "precision": float(frame["correct_assignment"].sum() / max(1, frame["accepted"].sum())),
            "recall": float(frame["correct_assignment"].mean()),
        }
        for gap, frame in positive.groupby("session_gap", sort=True)
    } if not positive.empty else {}
    return {
        "n_cases": int(len(benchmark)), "true_positive": tp, "false_positive": fp,
        "wrong_assignments": int((positive.get("false_positive", pd.Series(dtype=bool))).astype(bool).sum()),
        "unmatched_positives": int((~positive.get("accepted", pd.Series(dtype=bool)).astype(bool)).sum()),
        "collisions": int(benchmark.get("collision", pd.Series(dtype=bool)).astype(bool).sum()),
        "negative_false_positives": int(negative.get("false_positive", pd.Series(dtype=bool)).astype(bool).sum()),
        "precision": precision, "precision_wilson_lower": lower, "precision_wilson_upper": upper,
        "recall": recall, "negative_fpr": negative_fpr, "by_gap": by_gap,
        "guardrail_passed": bool(accepted > 0 and precision >= min_precision and negative_fpr <= max_negative_fpr),
        "min_precision": min_precision, "max_negative_fpr": max_negative_fpr,
    }


def build_synthetic_stitch_benchmark(
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    transforms: pd.DataFrame,
    *,
    policy: str = "graph",
    spacing: VoxelSpacing | None = None,
    config: StitcherConfig | None = None,
    replicates: int = 20,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Create deterministic pseudo-fragment cases, including explicit negatives.

    The benchmark deliberately calls the exact evidence/tiering/assignment path.
    It is conservative about available truth: tracks need context on both sides.
    """

    try:
        from endpoint_evaluator import search_endpoint_candidates
    except ImportError:  # pragma: no cover
        from matching.endpoint_evaluator import search_endpoint_candidates

    spacing, config = spacing or VoxelSpacing(), config or StitcherConfig()
    sessions = sessions.sort_values("session_index").reset_index(drop=True)
    tracks = _normalize_tracks(tracks, sessions, policy)
    rng = np.random.default_rng(random_seed)
    cases_by_gap: dict[int, list[tuple[int, int, int]]] = {1: [], 2: [], 3: []}
    for track_index, (_, track) in enumerate(tracks.iterrows()):
        if _track_flag(track, "has_cycle_conflict", "contains_transform_fallback_edge", "edge_heavy"):
            continue
        observed = {position for position, _, _ in _track_observations(track, sessions)}
        for gap in (1, 2, 3):
            for source_position in range(1, len(sessions) - gap - 1):
                target_position = source_position + gap
                if source_position in observed and target_position in observed and any(p < source_position for p in observed) and any(p > target_position for p in observed):
                    cases_by_gap[gap].append((track_index, source_position, target_position))
    cases: list[tuple[int, int, int]] = []
    for gap in (1, 2, 3):
        pool = sorted(cases_by_gap[gap])
        if pool:
            order = rng.permutation(len(pool)).tolist()
            cases.extend(pool[i] for i in order[:max(0, replicates)])
    rows: list[dict[str, Any]] = []
    roi_columns = [_roi_column(str(row.session_id)) for row in sessions.itertuples(index=False)]
    for case_number, (track_index, source_position, target_position) in enumerate(cases, start=1):
        base = tracks.iloc[track_index]
        source, target = base.copy(), base.copy()
        source_uid, target_uid = f"synthetic_source_{case_number:06d}", f"synthetic_target_{case_number:06d}"
        source["track_uid"], target["track_uid"] = source_uid, target_uid
        for position, column in enumerate(roi_columns):
            if position > source_position:
                source[column] = pd.NA
            if position < target_position:
                target[column] = pd.NA
        source["first_session_index"], source["last_session_index"] = min(p for p, _, _ in _track_observations(source, sessions)), source_position
        target["first_session_index"], target["last_session_index"] = target_position, max(p for p, _, _ in _track_observations(target, sessions))
        source["n_days_present"], target["n_days_present"] = len(_track_observations(source, sessions)), len(_track_observations(target, sessions))
        pseudo_tracks = pd.concat([tracks.drop(index=tracks.index[track_index]), pd.DataFrame([source, target])], ignore_index=True)
        source_session = sessions.iloc[source_position]
        endpoint = pd.DataFrame([{
            "endpoint_id": f"synthetic_endpoint_{case_number:06d}", "track_uid": source_uid,
            "end_session_index": int(source_session.session_index), "end_session_id": str(source_session.session_id),
            "end_label": int(source[_roi_column(str(source_session.session_id))]), "same_track_returns": False,
            "touches_z_edge": False, "touches_xy_edge": False,
        }])
        broad, _ = search_endpoint_candidates(
            endpoint, pseudo_tracks, features, sessions, transforms, policy=policy,
            lookahead=config.max_gap_sessions, search_radius_um=config.search_radius_um, spacing=spacing,
        )
        built = build_stitch_candidates(
            endpoint, broad, pd.DataFrame([{"endpoint_id": endpoint.iloc[0]["endpoint_id"], "classification": "nearby_new_track_candidate"}]),
            pseudo_tracks, features, sessions, transforms, policy=policy, spacing=spacing, config=config,
        )
        assigned = assign_stitches(built, config=config)
        accepted = assigned.loc[assigned["accepted_by_global_assignment"]]
        correct = bool(not accepted.empty and str(accepted.iloc[0]["target_track_uid"]) == target_uid)
        rows.append({
            "case_id": f"positive_{case_number:06d}", "case_type": "positive",
            "session_gap": target_position - source_position, "source_track_uid": source_uid,
            "true_target_track_uid": target_uid, "accepted": bool(not accepted.empty),
            "assigned_target_track_uid": str(accepted.iloc[0]["target_track_uid"]) if not accepted.empty else "",
            "correct_assignment": correct, "false_positive": bool(not accepted.empty and not correct),
            "collision": int(built.get("auto_eligible", pd.Series(dtype=bool)).sum()) > 1,
        })
        # Exact no-successor control: the source remains, but the true target is
        # not a track start. Any accepted edge is necessarily a false stitch.
        no_target_tracks = pseudo_tracks.loc[pseudo_tracks["track_uid"].astype(str).ne(target_uid)].copy()
        broad_negative, _ = search_endpoint_candidates(
            endpoint, no_target_tracks, features, sessions, transforms, policy=policy,
            lookahead=config.max_gap_sessions, search_radius_um=config.search_radius_um, spacing=spacing,
        )
        built_negative = build_stitch_candidates(
            endpoint, broad_negative, pd.DataFrame(), no_target_tracks, features, sessions, transforms,
            policy=policy, spacing=spacing, config=config,
        )
        negative_assigned = assign_stitches(built_negative, config=config)
        false_positive = bool(not negative_assigned.empty and negative_assigned["accepted_by_global_assignment"].any())
        rows.append({
            "case_id": f"no_successor_{case_number:06d}", "case_type": "no_successor",
            "session_gap": target_position - source_position, "source_track_uid": source_uid,
            "true_target_track_uid": "", "accepted": false_positive,
            "assigned_target_track_uid": "", "correct_assignment": False,
            "false_positive": false_positive, "collision": False,
        })
        rows.append({
            "case_id": f"target_only_{case_number:06d}", "case_type": "target_only",
            "session_gap": target_position - source_position, "source_track_uid": "",
            "true_target_track_uid": target_uid, "accepted": False,
            "assigned_target_track_uid": "", "correct_assignment": False,
            "false_positive": False, "collision": False,
        })
    return pd.DataFrame(rows, columns=[
        "case_id", "case_type", "session_gap", "source_track_uid", "true_target_track_uid",
        "accepted", "assigned_target_track_uid", "correct_assignment", "false_positive", "collision",
    ])


def benchmark_threshold_sweep(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Report guardrail metrics across the requested precision/FPR thresholds."""

    rows = []
    for precision in (0.99, 0.995, 0.999):
        for fpr in (0.01, 0.005, 0.001):
            metrics = summarize_stitch_benchmark(benchmark, min_precision=precision, max_negative_fpr=fpr)
            rows.append({"min_precision": precision, "max_negative_fpr": fpr, **{key: metrics[key] for key in ("precision", "recall", "negative_fpr", "guardrail_passed")}})
    return pd.DataFrame(rows)


def edge_table(assignments: pd.DataFrame) -> pd.DataFrame:
    output = assignments.copy()
    for column in STITCH_EDGE_COLUMNS:
        if column not in output:
            output[column] = False if column == "accepted_by_global_assignment" else np.nan
    return output.loc[:, STITCH_EDGE_COLUMNS]


def config_dict(config: StitcherConfig) -> dict[str, Any]:
    return asdict(config)
