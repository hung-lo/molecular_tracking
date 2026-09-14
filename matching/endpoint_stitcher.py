"""Conservative, state-blind post-hoc stitching of canonical track fragments.

This module reads canonical matching/evaluator artifacts and writes only derived
artifacts.  Fluorescence and ECLIPSE fields are intentionally absent from every
decision function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
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


ALGORITHM_VERSION = "conservative_endpoint_stitcher_v2"
STATE_COLUMNS = {"eclipse_z", "eclipse_core_state", "eclipse_state_bin", "green", "red", "color_z"}

STITCH_EDGE_COLUMNS = [
    "stitch_edge_id", "edge_type", "source_track_uid", "target_track_uid",
    "source_session_index", "source_session_id", "source_label",
    "target_session_index", "target_session_id", "target_label", "session_gap",
    "n_missing_sessions", "elapsed_day_gap", "candidate_tier",
    "accepted_by_global_assignment", "assignment_status", "projected_distance_um",
    "forward_rank_all_masks", "forward_rank_track_starts", "forward_second_margin_um",
    "reverse_projected_distance_um", "reverse_rank_source_endpoints",
    "reverse_second_margin_um", "reciprocal_rank1", "anchor_support_count",
    "anchor_support_fraction", "anchor_residual_median_um", "anchor_residual_p90_um",
    "anchor_inlier_fraction", "source_history_n", "source_history_distance_median_um",
    "target_future_n", "target_future_distance_median_um", "volume_ratio", "dice",
    "iou", "existing_candidate_found", "existing_graph_match", "transform_source",
    "transform_component_methods", "transform_component_fallback_reasons",
    "transform_component_residual_median_um_max", "transform_component_residual_p95_um_max",
    "direct_vs_composed_projection_delta_um", "source_touches_z_edge",
    "source_touches_xy_edge", "target_touches_z_edge", "target_touches_xy_edge",
    "source_feature_row_missing", "target_feature_row_missing",
    "source_feature_edge_evidence_missing", "target_feature_edge_evidence_missing",
    "source_edge_heavy", "target_edge_heavy",
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

MIN_BENCHMARK_POSITIVE_CASES = 1000
MIN_BENCHMARK_NEGATIVE_CONTROLS = 2000
MIN_BENCHMARK_CASES_PER_GAP = 100
WILSON_ONE_SIDED_95 = 1.6448536269514722


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
    benchmark_cases_per_gap: int = 50
    min_benchmark_positive_cases: int = MIN_BENCHMARK_POSITIVE_CASES
    min_benchmark_negative_controls: int = MIN_BENCHMARK_NEGATIVE_CONTROLS
    min_benchmark_cases_per_gap: int = MIN_BENCHMARK_CASES_PER_GAP

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


def stable_stitch_edge_id(row: pd.Series | dict[str, Any]) -> str:
    """Return an ID tied to the canonical endpoint/start identity, not row order."""

    key = {
        "source_track_uid": _text(row.get("source_track_uid")),
        "source_session_index": _int(row.get("source_session_index")),
        "source_label": _int(row.get("source_label")),
        "target_track_uid": _text(row.get("target_track_uid")),
        "target_session_index": _int(row.get("target_session_index")),
        "target_label": _int(row.get("target_label")),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"stitch_{digest}"


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


def _feature_edge_lookup(features: pd.DataFrame) -> dict[tuple[str, int], tuple[bool, bool]]:
    if not {"touches_z_edge", "touches_xy_edge"}.issubset(features.columns):
        return {}
    return {
        (str(row.session_id), int(row.label)): (
            _bool(getattr(row, "touches_z_edge", True)),
            _bool(getattr(row, "touches_xy_edge", True)),
        )
        for row in features.itertuples(index=False)
    }


def _transform_provenance(
    source_position: int,
    target_position: int,
    source_coords: np.ndarray | None,
    sessions: pd.DataFrame,
    transform_map: dict[tuple[str, str], pd.Series],
    spacing: VoxelSpacing,
) -> dict[str, Any]:
    projected = _forward_transform(source_position, target_position, sessions, transform_map)
    if projected is None:
        return {
            "transform_source": "", "transform_reliable": False,
            "transform_component_methods": "", "transform_component_fallback_reasons": "",
            "transform_component_residual_median_um_max": np.nan,
            "transform_component_residual_p95_um_max": np.nan,
            "direct_vs_composed_projection_delta_um": np.nan,
        }
    _, transform_source, reliable, qc = projected
    delta = np.nan
    if target_position - source_position == 2 and source_coords is not None:
        source_id = str(sessions.iloc[source_position]["session_id"])
        middle_id = str(sessions.iloc[source_position + 1]["session_id"])
        target_id = str(sessions.iloc[target_position]["session_id"])
        direct = transform_map.get((source_id, target_id))
        first = transform_map.get((source_id, middle_id))
        second = transform_map.get((middle_id, target_id))
        if direct is not None and first is not None and second is not None:
            try:
                direct_prediction = apply_transform_b_to_a(source_coords, invert_restricted_transform(direct))
                composed = compose_transforms(first, second)
                composed_prediction = apply_transform_b_to_a(source_coords, invert_restricted_transform(composed))
                delta = float(np.linalg.norm((direct_prediction - composed_prediction) * spacing.as_zyx_array()))
            except ValueError:
                pass
    return {
        "transform_source": transform_source,
        "transform_reliable": bool(reliable),
        "transform_component_methods": _text(qc.get("methods")),
        "transform_component_fallback_reasons": _text(qc.get("fallback_reasons")),
        "transform_component_residual_median_um_max": _float(qc.get("residual_median_um_max")),
        "transform_component_residual_p95_um_max": _float(qc.get("residual_p95_um_max")),
        "direct_vs_composed_projection_delta_um": delta,
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
    source_column, target_column = _roi_column(source_id), _roi_column(target_id)
    if source_column not in tracks or target_column not in tracks:
        empty = np.empty((0, 3), dtype=float)
        return np.asarray([], dtype=str), empty, empty, None, None
    valid = tracks[source_column].map(_present) & tracks[target_column].map(_present)
    for column in ("has_cycle_conflict", "contains_transform_fallback_edge", "edge_heavy"):
        if column in tracks:
            valid &= ~tracks[column].map(_bool)
    uids, source_points, target_points = [], [], []
    for index in tracks.index[valid]:
        source_label, target_label = int(tracks.at[index, source_column]), int(tracks.at[index, target_column])
        source = feature_coords.get((source_id, source_label))
        target = feature_coords.get((target_id, target_label))
        if source is not None and target is not None:
            uids.append(_text(tracks.at[index, "track_uid"])); source_points.append(source); target_points.append(target)
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
    target_starts = [coords for position, _, _, coords in target_obs if position == target_position]
    source_ends = [coords for position, _, _, coords in source_obs if position == source_position]
    if not target_starts or not source_ends:
        return _distance_summaries("source_history", []) | _distance_summaries("target_future", [])
    target_start, source_end = target_starts[0], source_ends[0]
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
    feature_edges = _feature_edge_lookup(features)
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
        source_feature_key = (str(candidate.get("source_session_id")), _int(candidate.get("source_label"), -1))
        target_feature_key = (str(candidate.get("target_session_id")), _int(candidate.get("target_label"), -1))
        source_feature_missing = source_feature_key not in feature_coords
        target_feature_missing = target_feature_key not in feature_coords
        source_edge_evidence_missing = source_feature_key not in feature_edges
        target_edge_evidence_missing = target_feature_key not in feature_edges
        source_touches_z, source_touches_xy = feature_edges.get(source_feature_key, (True, True))
        target_touches_z, target_touches_xy = feature_edges.get(target_feature_key, (True, True))
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
            "source_touches_z_edge": source_touches_z, "source_touches_xy_edge": source_touches_xy,
            "target_touches_z_edge": target_touches_z, "target_touches_xy_edge": target_touches_xy,
            "source_feature_row_missing": source_feature_missing,
            "target_feature_row_missing": target_feature_missing,
            "source_feature_edge_evidence_missing": source_edge_evidence_missing,
            "target_feature_edge_evidence_missing": target_edge_evidence_missing,
            "source_edge_heavy": source_feature_missing or source_touches_z or source_touches_xy,
            "target_edge_heavy": target_feature_missing or target_touches_z or target_touches_xy,
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
        row.update(_transform_provenance(
            source_position, target_position, feature_coords.get(source_feature_key),
            sessions, transform_map, spacing,
        ))
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
    table = table.sort_values(["source_session_index", "source_track_uid", "target_session_index", "target_track_uid", "target_label"], kind="mergesort").drop_duplicates(
        ["source_track_uid", "source_session_index", "source_label", "target_track_uid", "target_session_index", "target_label"], keep="first"
    ).reset_index(drop=True)
    table["stitch_edge_id"] = table.apply(stable_stitch_edge_id, axis=1)
    table["algorithm_version"] = ALGORITHM_VERSION
    return table


def eligible_source_endpoint_ids(
    endpoints: pd.DataFrame,
    tracks: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    policy: str = "graph",
) -> list[str]:
    """Count unresolved endpoint sources even when no target start is nearby."""

    lookup = _track_lookup(_normalize_tracks(tracks, sessions, policy))
    return [
        _text(endpoint.get("endpoint_id"))
        for _, endpoint in endpoints.iterrows()
        if _source_allowed(endpoint, lookup.get(_text(endpoint.get("track_uid")), pd.Series(dtype=object)), sessions)
    ]


def select_review_samples(
    assignments: pd.DataFrame,
    *,
    max_panels: int | None = 100,
    random_seed: int = 0,
    config: StitcherConfig | None = None,
) -> pd.DataFrame:
    """Select deterministic, category-aware review rows from post-assignment data."""

    if assignments.empty or max_panels == 0:
        return assignments.head(0).assign(review_sample_reason=pd.Series(dtype=str))
    table = assignments.loc[assignments.get("candidate_tier", pd.Series(index=assignments.index, dtype=str)).ne("reject")].copy()
    if table.empty:
        return table.assign(review_sample_reason=pd.Series(dtype=str))
    table = table.sort_values("stitch_edge_id", kind="mergesort")
    config = config or StitcherConfig()
    selected: list[int] = []
    reasons: dict[int, str] = {}

    def take(mask: pd.Series, reason: str, limit: int | None = None) -> None:
        count = 0
        for index in table.index[mask]:
            if index in reasons:
                continue
            reasons[index] = reason; selected.append(index); count += 1
            if limit is not None and count >= limit:
                break

    def take_spatial(mask: pd.Series, reason: str, limit: int) -> None:
        columns = ["predicted_z_um", "predicted_y_um", "predicted_x_um"]
        if not all(column in table for column in columns):
            return
        spatial = table.loc[mask, columns + ["stitch_edge_id"]].copy()
        points = spatial[columns].apply(pd.to_numeric, errors="coerce")
        spatial = spatial.loc[points.notna().all(axis=1)].copy()
        if spatial.empty:
            return
        bins = np.floor(points.loc[spatial.index].to_numpy(dtype=float) / 25.0).astype(int)
        spatial["_spatial_bin"] = [tuple(value) for value in bins]
        chosen_bins: set[tuple[int, int, int]] = set()
        for index, row in spatial.sort_values(["_spatial_bin", "stitch_edge_id"], kind="mergesort").iterrows():
            if row["_spatial_bin"] in chosen_bins or index in reasons:
                continue
            chosen_bins.add(row["_spatial_bin"])
            reasons[index] = reason; selected.append(index)
            if len(chosen_bins) >= limit:
                return

    for gap in (1, 2, 3):
        take(table.get("accepted_by_global_assignment", pd.Series(False, index=table.index)) & table.get("session_gap", pd.Series(-1, index=table.index)).eq(gap), f"auto_accept_gap_{gap}", 1)
    take(table.get("assignment_status", pd.Series(index=table.index, dtype=str)).eq("global_collision_rejection"), "global_collision_rejection", 3)
    for reason in ("singleton", "insufficient_local_anchors", "insufficient_target_future", "insufficient_source_history", "forward_margin_insufficient"):
        take(table.get("review_reasons", pd.Series(index=table.index, dtype=str)).astype(str).str.contains(reason, regex=False, na=False), reason, 2)
    distances = pd.to_numeric(table.get("projected_distance_um", pd.Series(index=table.index, dtype=float)), errors="coerce")
    limits = table.get("session_gap", pd.Series(index=table.index, dtype=int)).map({gap: config.distance_limit(gap) for gap in (1, 2, 3)})
    margins = pd.to_numeric(table.get("forward_second_margin_um", pd.Series(index=table.index, dtype=float)), errors="coerce")
    take((distances.ge(limits * .8) | margins.between(config.min_forward_margin_um, config.min_forward_margin_um + 1.5)).fillna(False), "near_threshold_distance_or_margin", 3)
    take(pd.to_numeric(table.get("volume_ratio", pd.Series(index=table.index, dtype=float)), errors="coerce").le(.3), "poor_volume_ratio_state_blind_geometry", 2)
    take_spatial(table.get("accepted_by_global_assignment", pd.Series(False, index=table.index)), "spatially_distributed_accepted", 5)
    limit = len(table) if max_panels is None else max(0, int(max_panels))
    if len(selected) < limit:
        remaining = table.index[~table.index.isin(selected)].to_numpy()
        order = np.random.default_rng(random_seed).permutation(len(remaining))
        for position in order:
            index = int(remaining[position]); reasons[index] = "seeded_remainder"; selected.append(index)
            if len(selected) >= limit:
                break
    selected = selected[:limit]
    output = table.loc[selected].copy()
    output["review_sample_reason"] = [reasons[index] for index in selected]
    return output.sort_values("stitch_edge_id", kind="mergesort").reset_index(drop=True)


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
        if _bool(row.get("source_feature_row_missing")) or _bool(row.get("target_feature_row_missing")):
            reject.append("missing_feature_row")
        if _bool(row.get("source_feature_edge_evidence_missing")) or _bool(row.get("target_feature_edge_evidence_missing")):
            reject.append("missing_feature_edge_evidence")
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
    before_nodes = [
        (column, int(value))
        for column in roi_columns if column in tracks
        for value in tracks[column].tolist() if _present(value)
    ]
    nodes_before = len(before_nodes)
    for members in groups.values():
        members.sort(key=lambda uid: (_int(lookup[uid].get("first_session_index"), 10**9), uid))
        stitched_uid = members[0]
        row = lookup[stitched_uid].to_dict()
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
    after_nodes = [
        (column, int(value))
        for column in roi_columns if column in stitched
        for value in stitched[column].tolist() if _present(value)
    ]
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
    min_positive_cases: int = MIN_BENCHMARK_POSITIVE_CASES,
    min_negative_controls: int = MIN_BENCHMARK_NEGATIVE_CONTROLS,
    min_cases_per_gap: int = MIN_BENCHMARK_CASES_PER_GAP,
) -> dict[str, Any]:
    benchmark = benchmark if benchmark is not None else pd.DataFrame()
    positive = benchmark.loc[benchmark.get("case_type", pd.Series(index=benchmark.index, dtype=str)).eq("positive")]
    negative = benchmark.loc[benchmark.get("case_type", pd.Series(index=benchmark.index, dtype=str)).ne("positive")]
    accepted_column = benchmark.get("accepted", pd.Series(False, index=benchmark.index)).astype(bool)
    correct_column = benchmark.get("correct_assignment", pd.Series(False, index=benchmark.index)).astype(bool)
    false_column = benchmark.get("false_positive", pd.Series(False, index=benchmark.index)).astype(bool)
    tp = int(correct_column.loc[positive.index].sum())
    wrong_target = int(false_column.loc[positive.index].sum())
    negative_false_links = int(false_column.loc[negative.index].sum())
    accepted = int(accepted_column.sum())
    precision = tp / accepted if accepted else np.nan
    recall = tp / len(positive) if len(positive) else np.nan
    negative_fpr = negative_false_links / len(negative) if len(negative) else np.nan
    lower, _ = wilson_interval(tp, accepted, z=WILSON_ONE_SIDED_95)
    _, negative_upper = wilson_interval(negative_false_links, len(negative), z=WILSON_ONE_SIDED_95)
    by_gap: dict[str, dict[str, Any]] = {}
    for gap, frame in benchmark.groupby("session_gap", sort=True):
        gap_positive = frame.loc[frame["case_type"].eq("positive")]
        gap_negative = frame.loc[frame["case_type"].ne("positive")]
        gap_truth_splits = gap_positive.get("canonical_truth_split_id", pd.Series(index=gap_positive.index, dtype=str)).replace("", np.nan).dropna()
        if gap_truth_splits.empty:
            gap_truth_splits = pd.Series(gap_positive.index.astype(str), index=gap_positive.index)
        gap_tp = int(correct_column.loc[gap_positive.index].sum())
        gap_wrong = int(false_column.loc[gap_positive.index].sum())
        gap_negative_fp = int(false_column.loc[gap_negative.index].sum())
        gap_accepted = int(accepted_column.loc[frame.index].sum())
        by_gap[str(int(gap))] = {
            "n": int(len(frame)), "n_positive_truth": int(len(gap_positive)),
            "n_unique_positive_truth_tracks": int(gap_positive.get("canonical_truth_track_uid", pd.Series(index=gap_positive.index, dtype=str)).replace("", np.nan).dropna().nunique()) if not gap_positive.empty else 0,
            "n_unique_positive_truth_splits": int(gap_truth_splits.nunique()),
            "n_negative_controls": int(len(gap_negative)), "accepted": gap_accepted,
            "true_positive": gap_tp, "wrong_target": gap_wrong,
            "negative_false_links": gap_negative_fp,
            "precision": gap_tp / gap_accepted if gap_accepted else np.nan,
            "recall": gap_tp / len(gap_positive) if len(gap_positive) else np.nan,
            "negative_fpr": gap_negative_fp / len(gap_negative) if len(gap_negative) else np.nan,
        }
    truth_tracks = positive.get("canonical_truth_track_uid", pd.Series(index=positive.index, dtype=str)).replace("", np.nan).dropna()
    truth_splits = positive.get("canonical_truth_split_id", pd.Series(index=positive.index, dtype=str)).replace("", np.nan).dropna()
    if truth_splits.empty:
        truth_splits = pd.Series(positive.index.astype(str), index=positive.index)
    unique_splits_by_gap = {
        key: int(value.get("n_unique_positive_truth_splits", 0))
        for key, value in by_gap.items()
    }
    sample_sufficient = bool(
        len(positive) >= min_positive_cases
        and truth_splits.nunique() >= min_positive_cases
        and len(negative) >= min_negative_controls
        and all(
            key in by_gap
            and int(by_gap[key]["n_positive_truth"]) >= min_cases_per_gap
            and unique_splits_by_gap.get(key, 0) >= min_cases_per_gap
            and np.isfinite(float(by_gap[key]["precision"]))
            for key in ("1", "2", "3")
        )
    )
    gap_guardrails_passed = bool(all(
        key in by_gap
        and np.isfinite(float(by_gap[key]["precision"]))
        and float(by_gap[key]["precision"]) >= min_precision
        and np.isfinite(float(by_gap[key]["negative_fpr"]))
        and float(by_gap[key]["negative_fpr"]) <= max_negative_fpr
        for key in ("1", "2", "3")
    ))
    empirical_pass = bool(
        np.isfinite(precision) and precision >= min_precision
        and np.isfinite(negative_fpr) and negative_fpr <= max_negative_fpr
    )
    interval_pass = bool(
        np.isfinite(lower) and lower >= 0.995
        and np.isfinite(negative_upper) and negative_upper <= max_negative_fpr
    )
    return {
        "n_cases": int(len(benchmark)), "n_positive_truth": int(len(positive)),
        "n_negative_controls": int(len(negative)), "n_accepted": accepted,
        "n_unique_positive_truth_tracks": int(truth_tracks.nunique()),
        "n_unique_positive_truth_splits": int(truth_splits.nunique()),
        "n_unique_positive_truth_splits_by_gap": unique_splits_by_gap,
        "true_positive": tp, "wrong_target_count": wrong_target,
        "false_positive": wrong_target + negative_false_links,
        "negative_false_positive_count": negative_false_links,
        "wrong_assignments": wrong_target,
        "unmatched_positives": int((~accepted_column.loc[positive.index]).sum()),
        "collisions": int(benchmark.get("collision", pd.Series(False, index=benchmark.index)).astype(bool).sum()),
        "precision": precision, "precision_wilson_lower_onesided_95": lower,
        "precision_wilson_lower": lower,
        "negative_fpr": negative_fpr, "negative_fpr_wilson_upper_onesided_95": negative_upper,
        "recall": recall, "by_gap": by_gap,
        "benchmark_sample_sufficient": sample_sufficient,
        "gap_guardrails_passed": gap_guardrails_passed,
        "empirical_guardrails_passed": empirical_pass, "interval_guardrails_passed": interval_pass,
        "guardrail_passed": bool(sample_sufficient and gap_guardrails_passed and empirical_pass and interval_pass),
        "min_precision": min_precision, "max_negative_fpr": max_negative_fpr,
        "min_positive_cases": min_positive_cases, "min_negative_controls": min_negative_controls,
        "min_cases_per_gap": min_cases_per_gap,
    }


def _build_isolated_synthetic_benchmark(
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


def _trusted_split_pool(
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    transforms: pd.DataFrame,
    config: StitcherConfig,
) -> dict[int, list[tuple[int, int, int]]]:
    feature_edges, transform_map = _feature_edge_lookup(features), _transform_row_map(transforms)
    output: dict[int, list[tuple[int, int, int]]] = {1: [], 2: [], 3: []}
    for track_index, (_, track) in enumerate(tracks.iterrows()):
        if _track_flag(track, "has_cycle_conflict", "contains_transform_fallback_edge", "edge_heavy"):
            continue
        observations = {position: (session_id, label) for position, session_id, label in _track_observations(track, sessions)}
        observed_positions = sorted(observations)
        for gap in (1, 2, 3):
            for source_position in range(1, len(sessions) - gap - 1):
                target_position = source_position + gap
                if not observed_positions or any(position not in observations for position in range(observed_positions[0], source_position + 1)):
                    continue
                if not all(position in observations for position in range(source_position, target_position + 1)):
                    continue
                if not any(position < source_position for position in observations) or not any(position > target_position for position in observations):
                    continue
                source_id, source_label = observations[source_position]
                target_id, target_label = observations[target_position]
                if feature_edges.get((source_id, source_label), (True, True)) != (False, False) or feature_edges.get((target_id, target_label), (True, True)) != (False, False):
                    continue
                transform = _forward_transform(source_position, target_position, sessions, transform_map)
                if transform is None or not transform[2]:
                    continue
                output[gap].append((track_index, source_position, target_position))
    return output


def _split_row(track: pd.Series, sessions: pd.DataFrame, uid: str, first: int, last: int) -> pd.Series:
    row = track.copy()
    row["track_uid"] = uid
    roi_columns = [_roi_column(str(session.session_id)) for session in sessions.itertuples(index=False)]
    for position, column in enumerate(roi_columns):
        if position < first or position > last:
            row[column] = pd.NA
    row["first_session_index"], row["last_session_index"] = first, last
    row["n_days_present"] = last - first + 1
    row["missing_internal_days"] = 0
    return row


def _copy_feature_row(features: pd.DataFrame, session_id: str, label: int, new_label: int, *, y_offset: float = 0.0) -> dict[str, Any] | None:
    feature = _feature_row(features, session_id, label)
    if feature is None:
        return None
    row = feature.to_dict()
    row["label"] = new_label
    row["centroid_y"] = float(row["centroid_y"]) + y_offset
    if "centroid_y_um" in row:
        row["centroid_y_um"] = float(row["centroid_y_um"]) + y_offset
    return row


def _clone_track_with_unique_labels(
    track: pd.Series,
    sessions: pd.DataFrame,
    features: pd.DataFrame,
    uid: str,
    label_offset: int,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Clone one track and its feature rows for a synthetic control."""

    clone = track.copy()
    clone["track_uid"] = uid
    copied: list[dict[str, Any]] = []
    for session in sessions.itertuples(index=False):
        session_id = str(session.session_id)
        column = _roi_column(session_id)
        value = clone.get(column, pd.NA)
        if not _present(value):
            continue
        old_label = int(value)
        new_label = old_label + label_offset
        feature = _copy_feature_row(features, session_id, old_label, new_label)
        if feature is None:
            continue
        clone[column] = new_label
        copied.append(feature)
    return clone, copied


def _benchmark_evidence(assigned: pd.DataFrame, source_uid: str, target_uid: str) -> dict[str, Any]:
    if assigned.empty:
        return {}
    subset = assigned.loc[assigned["source_track_uid"].astype(str).eq(source_uid)]
    if target_uid:
        exact = subset.loc[subset["target_track_uid"].astype(str).eq(target_uid)]
        if not exact.empty:
            return exact.sort_values("projected_distance_um", kind="mergesort").iloc[0].to_dict()
    if not subset.empty:
        return subset.sort_values(["candidate_tier", "projected_distance_um", "target_track_uid"], kind="mergesort").iloc[0].to_dict()
    return {}


def _benchmark_case_rows(assigned: pd.DataFrame, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted = assigned.loc[assigned.get("accepted_by_global_assignment", pd.Series(False, index=assigned.index)).astype(bool)] if not assigned.empty else pd.DataFrame()
    auto = assigned.loc[assigned.get("auto_eligible", pd.Series(False, index=assigned.index)).astype(bool)] if not assigned.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for case in cases:
        source_uid, target_uid = case.get("source_uid", ""), case.get("target_uid", "")
        source_accepted = accepted.loc[accepted["source_track_uid"].astype(str).eq(source_uid)] if source_uid and not accepted.empty else pd.DataFrame()
        target_accepted = accepted.loc[accepted["target_track_uid"].astype(str).eq(target_uid)] if target_uid and not accepted.empty else pd.DataFrame()
        assigned_target = str(source_accepted.iloc[0]["target_track_uid"]) if not source_accepted.empty else ""
        is_positive = case["case_type"] == "positive"
        correct = bool(is_positive and assigned_target == target_uid)
        false_positive = bool(
            (is_positive and not source_accepted.empty and not correct)
            or (not is_positive and (not source_accepted.empty if source_uid else not target_accepted.empty))
        )
        evidence = _benchmark_evidence(assigned, source_uid, target_uid)
        component = _text(evidence.get("assignment_component_id"))
        component_edges = auto.loc[auto["assignment_component_id"].astype(str).eq(component)] if component and not auto.empty else pd.DataFrame()
        competing = component_edges.loc[
            component_edges["source_track_uid"].astype(str).eq(source_uid)
            | component_edges["target_track_uid"].astype(str).eq(target_uid)
        ] if not component_edges.empty else pd.DataFrame()
        collision = len(competing) > 1
        accepted_case = bool(not source_accepted.empty if source_uid else not target_accepted.empty)
        rows.append({
            "case_id": case["case_id"], "case_type": case["case_type"], "negative_subtype": case.get("negative_subtype", ""),
            "replicate": case.get("replicate", 0), "session_gap": case["session_gap"], "source_track_uid": source_uid,
            "true_target_track_uid": target_uid, "accepted": accepted_case,
            "assigned_target_track_uid": assigned_target, "correct_assignment": correct,
            "false_positive": false_positive, "collision": collision,
            "collision_resolved_correctly": bool(collision and (correct or (not is_positive and not false_positive))),
            "assignment_component_id": component, "n_auto_edges_in_component": int(len(component_edges)),
            "projected_distance_um": _float(evidence.get("projected_distance_um")),
            "forward_margin_um": _float(evidence.get("forward_second_margin_um")),
            "anchor_residual_median_um": _float(evidence.get("anchor_residual_median_um")),
            "anchor_inlier_fraction": _float(evidence.get("anchor_inlier_fraction")),
            "source_history_distance_median_um": _float(evidence.get("source_history_distance_median_um")),
            "target_future_distance_median_um": _float(evidence.get("target_future_distance_median_um")),
            "forward_rank": _int(evidence.get("forward_rank_all_masks")), "reverse_rank": _int(evidence.get("reverse_rank_source_endpoints")),
            "canonical_truth_track_uid": case.get("canonical_truth_track_uid", ""),
            "canonical_truth_split_id": case.get("canonical_truth_split_id", ""),
        })
    return rows


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
    cases_per_gap: int | None = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Run batched pseudo-fragment assignments with real negative controls."""

    try:
        from endpoint_evaluator import search_endpoint_candidates
    except ImportError:  # pragma: no cover
        from matching.endpoint_evaluator import search_endpoint_candidates
    spacing, config = spacing or VoxelSpacing(), config or StitcherConfig()
    sessions = sessions.sort_values("session_index").reset_index(drop=True)
    tracks = _normalize_tracks(tracks, sessions, policy)
    cases_per_gap = config.benchmark_cases_per_gap if cases_per_gap is None else int(cases_per_gap)
    if replicates < 0 or cases_per_gap < 0:
        raise ValueError("benchmark replicates and cases per gap must be nonnegative")
    pools = _trusted_split_pool(tracks, features, sessions, transforms, config)
    rng = np.random.default_rng(random_seed)
    used_truth_units: set[tuple[int, int]] = set()
    replay_batches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    roi_columns = [_roi_column(str(session.session_id)) for session in sessions.itertuples(index=False)]
    for replicate in range(int(replicates)):
        for gap in (1, 2, 3):
            pool = pools[gap]
            if not pool or cases_per_gap == 0:
                continue
            order = rng.permutation(len(pool)).tolist()
            chosen: list[tuple[int, int, int]] = []
            selected_indices: set[int] = set()
            for index in order:
                item = pool[index]
                truth_unit = (item[0], gap)
                if item[0] in selected_indices or truth_unit in used_truth_units:
                    continue
                selected_indices.add(item[0])
                used_truth_units.add(truth_unit)
                chosen.append(item)
                if len(chosen) >= cases_per_gap:
                    break
            if not chosen:
                continue
            remaining = [item for item in pool if item[0] not in selected_indices]
            base_tracks = tracks.drop(index=tracks.index[list(selected_indices)]).copy()
            pseudo_rows: list[pd.Series] = []
            endpoint_rows: list[dict[str, Any]] = []
            feature_table = features.copy()
            positive_cases: list[dict[str, Any]] = []
            negative_cases: list[dict[str, Any]] = []
            target_only_cases: list[dict[str, Any]] = []
            for ordinal, (track_index, source_position, target_position) in enumerate(chosen, start=1):
                base = tracks.iloc[track_index]
                source_uid = f"synthetic_r{replicate:03d}_g{gap}_source_{ordinal:04d}"
                target_uid = f"synthetic_r{replicate:03d}_g{gap}_target_{ordinal:04d}"
                source = _split_row(base, sessions, source_uid, 0, source_position)
                target = _split_row(base, sessions, target_uid, target_position, len(sessions) - 1)
                pseudo_rows.extend([source, target])
                source_session = sessions.iloc[source_position]
                endpoint_rows.append({
                    "endpoint_id": f"synthetic_r{replicate:03d}_g{gap}_endpoint_{ordinal:04d}",
                    "track_uid": source_uid, "end_session_index": int(source_session.session_index),
                    "end_session_id": str(source_session.session_id),
                    "end_label": int(source[_roi_column(str(source_session.session_id))]),
                    "same_track_returns": False, "touches_z_edge": False, "touches_xy_edge": False,
                })
                positive_cases.append({
                    "case_id": f"positive_r{replicate:03d}_g{gap}_{ordinal:04d}",
                    "source_uid": source_uid, "target_uid": target_uid, "session_gap": gap,
                    "track_index": track_index, "source_position": source_position, "target_position": target_position,
                    "case_type": "positive", "canonical_truth_track_uid": _text(base.get("track_uid")),
                    "canonical_truth_split_id": f"{base.get('track_uid')}|g{gap}|s{source_position}|t{target_position}",
                })

                # Use an unselected truth track for a true no-successor control
                # when available, so its real target feature can be removed
                # without changing a positive in this batch.
                subtype = "no_successor_no_mask" if ordinal % 2 else "no_successor_nonstart_mask"
                if subtype == "no_successor_no_mask" or ordinal > len(remaining):
                    control_base = tracks.iloc[remaining[ordinal - 1][0]] if ordinal <= len(remaining) else base
                    control, copied_features = _clone_track_with_unique_labels(
                        control_base, sessions, features, f"unused_control_{replicate}_{gap}_{ordinal}",
                        3_000_000 + replicate * 100_000 + gap * 10_000 + ordinal * 100,
                    )
                    feature_table = pd.concat([feature_table, pd.DataFrame(copied_features)], ignore_index=True)
                    if subtype == "no_successor_nonstart_mask":
                        pseudo_rows.append(control.copy())
                else:
                    control = tracks.iloc[remaining[ordinal - 1][0]]
                control_source_uid = f"synthetic_r{replicate:03d}_g{gap}_no_successor_{ordinal:04d}"
                control_source = _split_row(control, sessions, control_source_uid, 0, source_position)
                pseudo_rows.append(control_source)
                control_source_label = int(control_source[_roi_column(str(source_session.session_id))])
                endpoint_rows.append({
                    "endpoint_id": f"synthetic_r{replicate:03d}_g{gap}_no_successor_endpoint_{ordinal:04d}",
                    "track_uid": control_source_uid, "end_session_index": int(source_session.session_index),
                    "end_session_id": str(source_session.session_id), "end_label": control_source_label,
                    "same_track_returns": False, "touches_z_edge": False, "touches_xy_edge": False,
                })
                control_observations = {position: (session_id, label) for position, session_id, label in _track_observations(control, sessions)}
                control_target_id, control_target_label = control_observations[target_position]
                if subtype == "no_successor_no_mask":
                    drop = (str(control_target_id), int(control_target_label))
                    feature_table = feature_table.loc[~(feature_table["session_id"].astype(str).eq(drop[0]) & feature_table["label"].astype(int).eq(drop[1]))]
                negative_cases.append({"case_id": f"{subtype}_r{replicate:03d}_g{gap}_{ordinal:04d}", "source_uid": control_source_uid, "target_uid": "", "session_gap": gap, "case_type": "no_successor", "negative_subtype": subtype})

            # Force one end-to-end source collision in every non-empty batch.
            # The duplicate endpoint has the same geometry, so the real LAP,
            # rather than an isolated case shortcut, must resolve it.
            collision_candidates = [case for case in positive_cases if case["target_position"] >= 3]
            collision_case = min(collision_candidates or positive_cases, key=lambda case: (case["target_position"], case["case_id"]))
            collision_base = tracks.iloc[collision_case["track_index"]]
            collision_source_uid = f"synthetic_r{replicate:03d}_g{gap}_collision_source"
            collision_source_position = collision_case["target_position"] - 2
            if collision_source_position < 1 or collision_source_position == collision_case["source_position"]:
                collision_source_position = collision_case["target_position"] - 1
            collision_source = _split_row(collision_base, sessions, collision_source_uid, 0, collision_source_position)
            pseudo_rows.append(collision_source)
            collision_session = sessions.iloc[collision_source_position]
            endpoint_rows.append({
                "endpoint_id": f"synthetic_r{replicate:03d}_g{gap}_collision_endpoint",
                "track_uid": collision_source_uid, "end_session_index": int(collision_session.session_index),
                "end_session_id": str(collision_session.session_id),
                "end_label": int(collision_source[_roi_column(str(collision_session.session_id))]),
                "same_track_returns": False, "touches_z_edge": False, "touches_xy_edge": False,
            })
            negative_cases.append({
                "case_id": f"collision_competitor_r{replicate:03d}_g{gap}", "source_uid": collision_source_uid,
                "target_uid": collision_case["target_uid"], "session_gap": collision_case["target_position"] - collision_source_position,
                "case_type": "collision_competitor", "negative_subtype": "collision_competitor",
            })

            # A target-only decoy is a real target start in this same LAP but
            # has no matching source endpoint.
            base = tracks.iloc[chosen[0][0]]
            decoy_uid = f"synthetic_r{replicate:03d}_g{gap}_target_only"
            decoy = base.copy(); decoy["track_uid"] = decoy_uid
            for column in roi_columns:
                decoy[column] = pd.NA
            decoy_position = chosen[0][2]
            base_observations = {position: (session_id, label) for position, session_id, label in _track_observations(base, sessions)}
            future_positions = [position for position in base_observations if position > decoy_position]
            if not future_positions:
                continue
            future_position = min(future_positions)
            decoy_label = -int(2_000_000 + replicate * 10_000 + gap)
            decoy[roi_columns[decoy_position]] = decoy_label
            decoy[roi_columns[future_position]] = decoy_label - 1
            decoy["first_session_index"], decoy["last_session_index"] = decoy_position, future_position
            decoy["n_days_present"], decoy["missing_internal_days"] = 2, future_position - decoy_position - 1
            pseudo_rows.append(decoy)
            for position, label in ((decoy_position, decoy_label), (future_position, decoy_label - 1)):
                source_id, source_label = base_observations[position]
                copied = _copy_feature_row(features, source_id, source_label, label, y_offset=7.0)
                if copied is not None:
                    feature_table = pd.concat([feature_table, pd.DataFrame([copied])], ignore_index=True)
            target_only_cases.append({"case_id": f"target_only_r{replicate:03d}_g{gap}", "source_uid": "", "target_uid": decoy_uid, "session_gap": gap, "case_type": "target_only", "negative_subtype": "target_only"})
            pseudo_tracks = pd.concat([base_tracks, pd.DataFrame(pseudo_rows)], ignore_index=True)
            endpoints = pd.DataFrame(endpoint_rows)
            broad, _ = search_endpoint_candidates(
                endpoints, pseudo_tracks, feature_table, sessions, transforms, policy=policy,
                lookahead=config.max_gap_sessions, search_radius_um=config.search_radius_um, spacing=spacing,
            )
            classes = pd.DataFrame({"endpoint_id": endpoints["endpoint_id"], "classification": "nearby_new_track_candidate"})
            candidates = build_stitch_candidates(
                endpoints, broad, classes, pseudo_tracks, feature_table, sessions, transforms,
                policy=policy, spacing=spacing, config=config,
            )
            assigned = assign_stitches(candidates, config=config)
            cases = positive_cases + negative_cases + target_only_cases
            for case in cases:
                case["replicate"] = replicate
            rows.extend(_benchmark_case_rows(assigned, cases))
            replay_batches.append({"batch_id": f"synthetic_r{replicate:03d}_g{gap}", "candidates": candidates.copy(), "cases": cases})
    output = pd.DataFrame(rows, columns=[
        "case_id", "case_type", "negative_subtype", "replicate", "session_gap", "source_track_uid", "true_target_track_uid",
        "accepted", "assigned_target_track_uid", "correct_assignment", "false_positive", "collision", "collision_resolved_correctly",
        "assignment_component_id", "n_auto_edges_in_component", "projected_distance_um", "forward_margin_um",
        "anchor_residual_median_um", "anchor_inlier_fraction", "source_history_distance_median_um",
        "target_future_distance_median_um", "forward_rank", "reverse_rank", "canonical_truth_track_uid", "canonical_truth_split_id",
    ])
    output.attrs["threshold_replay_batches"] = replay_batches
    return output


def benchmark_threshold_sweep(benchmark: pd.DataFrame, *, config: StitcherConfig | None = None) -> pd.DataFrame:
    """Replay each saved synthetic batch through tiering and global assignment."""

    config = config or StitcherConfig()
    batches = benchmark.attrs.get("threshold_replay_batches") if benchmark is not None else None
    if batches is None:
        raise ValueError("benchmark_threshold_sweep requires replay batches from build_synthetic_stitch_benchmark")
    rows: list[dict[str, Any]] = []
    for distance_scale in (0.8, 1.0, 1.2):
        for margin in (1.5, 3.0, 4.5):
            profile = replace(
                config,
                gap1_max_distance_um=config.gap1_max_distance_um * distance_scale,
                gap2_max_distance_um=config.gap2_max_distance_um * distance_scale,
                gap3_max_distance_um=config.gap3_max_distance_um * distance_scale,
                min_forward_margin_um=margin,
            )
            replayed: list[dict[str, Any]] = []
            n_auto_edges = 0
            for batch in batches:
                tiered = tier_stitch_candidates(batch["candidates"], config=profile)
                assigned = assign_stitches(tiered, config=profile)
                n_auto_edges += int(assigned.get("auto_eligible", pd.Series(False, index=assigned.index)).astype(bool).sum())
                replayed.extend(_benchmark_case_rows(assigned, batch["cases"]))
            replayed_frame = pd.DataFrame(replayed)
            case_types = replayed_frame.get("case_type", pd.Series(dtype=str))
            positives = int(case_types.eq("positive").sum())
            accepted_count = int(replayed_frame.get("accepted", pd.Series(dtype=bool)).astype(bool).sum())
            true_positive = int(replayed_frame.get("correct_assignment", pd.Series(dtype=bool)).astype(bool).sum())
            negative = replayed_frame.loc[case_types.ne("positive")] if not replayed_frame.empty else replayed_frame
            negative_fp = int(negative.get("false_positive", pd.Series(dtype=bool)).astype(bool).sum())
            negative_count = int(len(negative))
            positive_targets = replayed_frame.loc[case_types.eq("positive"), "assigned_target_track_uid"] if not replayed_frame.empty else pd.Series(dtype=str)
            rows.append({
                "distance_scale": distance_scale, "min_forward_margin_um": margin,
                "assignment_replayed": True, "n_positive_truth": positives, "n_cases": len(replayed_frame),
                "n_auto_eligible_edges": n_auto_edges, "n_selected_cases": n_auto_edges,
                "accepted": accepted_count, "true_positive": true_positive,
                "recall": float(true_positive / positives) if positives else np.nan,
                "precision": float(true_positive / accepted_count) if accepted_count else np.nan,
                "assigned_positive_target_track_uids": ";".join(sorted(set(positive_targets.astype(str)) - {""})),
                "negative_false_positive_count": int(negative_fp),
                "negative_fpr": float(negative_fp / negative_count) if negative_count else np.nan,
            })
    return pd.DataFrame(rows)


def edge_table(assignments: pd.DataFrame) -> pd.DataFrame:
    output = assignments.copy()
    for column in STITCH_EDGE_COLUMNS:
        if column not in output:
            output[column] = False if column == "accepted_by_global_assignment" else np.nan
    return output.loc[:, STITCH_EDGE_COLUMNS]


def config_dict(config: StitcherConfig) -> dict[str, Any]:
    return asdict(config)
