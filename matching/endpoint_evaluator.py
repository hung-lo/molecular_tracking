"""Read-only diagnostics for daywise matcher endpoints.

The matcher stores transforms from later-session coordinates (B) into the
earlier session (A).  This module inverts those transforms only for
evaluation; it never writes to a matching directory.
"""

from __future__ import annotations

from datetime import date
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree

try:  # Works both as a package import and when the CLI is run as a script.
    from affine_overlap_matcher import RestrictedTransform, VoxelSpacing
except ImportError:  # pragma: no cover - exercised by package imports
    from matching.affine_overlap_matcher import RestrictedTransform, VoxelSpacing


ENDPOINT_COLUMNS = [
    "endpoint_id", "track_uid", "cluster_id", "end_session_index", "end_session_id",
    "end_acquisition_date", "end_label", "first_session_index", "last_session_index",
    "n_days_present", "missing_internal_days", "n_adjacent_edges", "n_gap_edges",
    "used_gap_bridge", "min_score", "mean_score", "min_dice", "max_distance_um",
    "max_ambiguity", "track_match_source", "cycle_agreement_fraction",
    "has_cycle_conflict", "cycle_unchecked", "eclipse_z", "eclipse_core_state",
    "green", "red", "centroid_z_um", "centroid_y_um", "centroid_x_um", "volume_um3",
    "touches_z_edge", "touches_xy_edge", "same_track_returns", "same_track_return_gap",
    "same_track_return_session_index",
]

ENDPOINT_CANDIDATE_COLUMNS = [
    "endpoint_id", "source_track_uid", "source_session_index", "source_session_id",
    "source_label", "target_session_index", "target_session_id", "target_label",
    "target_track_uid", "session_gap", "elapsed_day_gap", "predicted_z_um",
    "predicted_y_um", "predicted_x_um", "target_z_um", "target_y_um", "target_x_um",
    "projected_distance_um", "source_volume_um3", "target_volume_um3", "volume_ratio",
    "target_track_starts_here", "target_is_singleton", "target_track_match_source",
    "existing_candidate_found", "existing_high_match", "existing_balanced_match",
    "existing_graph_match", "dice", "iou", "ambiguity", "base_score", "refined_score",
    "candidate_source", "graph_status", "graph_support_count", "graph_support_fraction",
    "graph_residual_median_um", "graph_inlier_fraction", "target_rank_by_distance",
    "nearest_second_nearest_margin_um", "transform_source", "transform_reliable",
]

CLASSIFICATION_COLUMNS = [
    "endpoint_id", "track_uid", "end_session_index", "end_session_id", "end_label",
    "classification", "candidate_count", "same_track_returns", "manual_class",
    "manual_target_track_uid", "manual_target_session_index", "manual_target_label",
    "manual_confidence", "reviewer_notes",
]

SYNTHETIC_COLUMNS = [
    "case_id", "track_uid", "source_session_index", "target_session_index",
    "true_target_in_search_radius", "true_target_rank", "top1_correct", "candidate_count",
    "nearest_second_margin_um", "projected_distance_um", "session_gap", "elapsed_day_gap",
    "source_volume", "local_density", "edge_status", "session_pair",
]

RUNTIME_COLUMNS = [
    "session_pair", "day_a", "day_b", "pair_gap", "elapsed_sec", "n_a", "n_b",
    "candidate_count", "transform_method", "transform_fallback_reason", "graph_stage_seconds",
]


def _value(source: Mapping[str, Any] | pd.Series | object, key: str, default: Any = None) -> Any:
    if isinstance(source, pd.Series):
        return source.get(key, default)
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    try:
        return bool(value)
    except Exception:
        return default


def _text(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return str(value)


def _transform_value(transform: Mapping[str, Any] | pd.Series | object, name: str, default: float) -> float:
    return _float(_value(transform, name, default), default)


def _transform_matrix(transform: Mapping[str, Any] | pd.Series | object) -> tuple[np.ndarray, np.ndarray]:
    intercept = np.array([
        _transform_value(transform, "y_intercept", 0.0),
        _transform_value(transform, "x_intercept", 0.0),
    ], dtype=float)
    matrix = np.array([
        [_transform_value(transform, "y_from_y", 1.0), _transform_value(transform, "y_from_x", 0.0)],
        [_transform_value(transform, "x_from_y", 0.0), _transform_value(transform, "x_from_x", 1.0)],
    ], dtype=float)
    return intercept, matrix


def _new_transform(
    *, z_intercept: float, z_scale: float, intercept: np.ndarray, matrix: np.ndarray,
    method: str, fallback_reason: str | None = None, source: object | None = None,
) -> RestrictedTransform:
    return RestrictedTransform(
        z_intercept=float(z_intercept), z_scale=float(z_scale),
        y_intercept=float(intercept[0]), y_from_y=float(matrix[0, 0]), y_from_x=float(matrix[0, 1]),
        x_intercept=float(intercept[1]), x_from_y=float(matrix[1, 0]), x_from_x=float(matrix[1, 1]),
        method=method, fallback_reason=fallback_reason,
        n_seed=_int(_value(source, "n_seed", 0), 0) or 0,
        n_inlier=_int(_value(source, "n_inlier", 0), 0) or 0,
        residual_median_um=_value(source, "residual_median_um", None),
        residual_p95_um=_value(source, "residual_p95_um", None),
    )


def apply_transform_b_to_a(
    coordinates_b: np.ndarray | list[float],
    transform: Mapping[str, Any] | pd.Series | object,
) -> np.ndarray:
    """Apply a stored B→A restricted transform to ``(..., z, y, x)`` points."""

    coordinates = np.asarray(coordinates_b, dtype=float)
    scalar = coordinates.ndim == 1
    if scalar:
        coordinates = coordinates[None, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (..., 3)")
    intercept, matrix = _transform_matrix(transform)
    output = coordinates.copy()
    output[:, 0] = _transform_value(transform, "z_intercept", 0.0) + _transform_value(transform, "z_scale", 1.0) * coordinates[:, 0]
    output[:, 1:3] = intercept + coordinates[:, 1:3] @ matrix.T
    return output[0] if scalar else output


def invert_restricted_transform(
    transform: Mapping[str, Any] | pd.Series | object,
    *, singular_tolerance: float = 1e-10,
) -> RestrictedTransform:
    """Return the A→B inverse of one stored B→A transform."""

    intercept, matrix = _transform_matrix(transform)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) <= float(singular_tolerance):
        raise ValueError("restricted XY transform is singular or near-singular")
    z_scale = _transform_value(transform, "z_scale", 1.0)
    if not np.isfinite(z_scale) or abs(z_scale) <= float(singular_tolerance):
        raise ValueError("restricted Z transform is singular or near-singular")
    inverse_matrix = np.linalg.inv(matrix)
    return _new_transform(
        z_intercept=-_transform_value(transform, "z_intercept", 0.0) / z_scale,
        z_scale=1.0 / z_scale,
        intercept=-inverse_matrix @ intercept,
        matrix=inverse_matrix,
        method="inverse",
        source=transform,
    )


def compose_transforms(
    b_to_a: Mapping[str, Any] | pd.Series | object,
    c_to_b: Mapping[str, Any] | pd.Series | object,
) -> RestrictedTransform:
    """Compose ``C→B`` after ``B→A`` and return the resulting C→A map."""

    intercept_a, matrix_a = _transform_matrix(b_to_a)
    intercept_b, matrix_b = _transform_matrix(c_to_b)
    z_scale_a = _transform_value(b_to_a, "z_scale", 1.0)
    z_scale_b = _transform_value(c_to_b, "z_scale", 1.0)
    return _new_transform(
        z_intercept=_transform_value(b_to_a, "z_intercept", 0.0) + z_scale_a * _transform_value(c_to_b, "z_intercept", 0.0),
        z_scale=z_scale_a * z_scale_b,
        intercept=intercept_a + matrix_a @ intercept_b,
        matrix=matrix_a @ matrix_b,
        method="composed_adjacent",
        fallback_reason=_first_fallback(b_to_a, c_to_b),
        source=b_to_a,
    )


def _first_fallback(*transforms: object) -> str | None:
    for transform in transforms:
        value = _value(transform, "fallback_reason", None)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value)
    return None


def project_centroid_forward(
    centroid_b: np.ndarray | list[float],
    transform_b_to_a: Mapping[str, Any] | pd.Series | object,
) -> np.ndarray:
    """Project a centroid from B into A using the repository convention."""

    return apply_transform_b_to_a(centroid_b, transform_b_to_a)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def discover_evaluator_inputs(match_dir: str | Path, policy: str = "graph") -> dict[str, Any]:
    """Discover required/optional artifacts without modifying the match run."""

    root = Path(match_dir).resolve()
    policy = str(policy).lower()
    required_names = ["session_manifest_resolved.csv", "roi_features.csv", "pairwise_transforms.csv", f"tracks_{policy}.csv"]
    optional_names = [
        "pairwise_candidates.csv", f"pairwise_matches_{policy}.csv", f"track_edges_{policy}.csv",
        f"pairwise_summary_{policy}.csv", "pairwise_summary.csv", "run_log.json",
    ]
    paths = {name: root / name for name in required_names + optional_names}
    return {
        "match_dir": str(root),
        "policy": policy,
        "required": {name: str(paths[name]) for name in required_names if paths[name].is_file()},
        "missing_required": [name for name in required_names if not paths[name].is_file()],
        "optional": {name: str(paths[name]) for name in optional_names if paths[name].is_file()},
        "missing_optional": [name for name in optional_names if not paths[name].is_file()],
    }


def _session_table(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"session_index", "session_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"session manifest is missing columns: {', '.join(missing)}")
    table = manifest.copy()
    table["session_index"] = table["session_index"].astype(int)
    table["session_id"] = table["session_id"].astype(str)
    return table.sort_values("session_index").reset_index(drop=True)


def _session_date(row: pd.Series) -> str:
    value = row.get("acquisition_date", "")
    return "" if pd.isna(value) else str(value)


def _roi_column(session_id: str) -> str:
    return f"{session_id}_roi"


def _present(value: Any) -> bool:
    return not (value is None or pd.isna(value))


def _prepare_tracks(tracks: pd.DataFrame, sessions: pd.DataFrame, policy: str) -> pd.DataFrame:
    if tracks.empty:
        return tracks.copy()
    table = tracks.copy()
    if "track_uid" not in table.columns:
        first = sessions.iloc[0]["session_id"]
        table["track_uid"] = table.get("cluster_id", pd.Series(range(len(table)))).map(lambda x: f"{first}:{int(x)}")
    if "match_policy" not in table.columns:
        table["match_policy"] = policy
    roi_columns = [_roi_column(str(row.session_id)) for row in sessions.itertuples()]
    present_matrix = pd.DataFrame({column: table[column].notna() if column in table else False for column in roi_columns}, index=table.index)
    if "n_days_present" not in table.columns:
        table["n_days_present"] = present_matrix.sum(axis=1).astype(int)
    if "first_session_index" not in table.columns:
        table["first_session_index"] = present_matrix.apply(lambda row: int(np.flatnonzero(row.to_numpy())[0]) if row.any() else -1, axis=1)
    if "last_session_index" not in table.columns:
        table["last_session_index"] = present_matrix.apply(lambda row: int(np.flatnonzero(row.to_numpy())[-1]) if row.any() else -1, axis=1)
    if "missing_internal_days" not in table.columns:
        table["missing_internal_days"] = [
            int(sum(not bool(present_matrix.iloc[index, column]) for column in range(int(first), int(last) + 1))) if int(first) >= 0 else 0
            for index, (first, last) in enumerate(zip(table["first_session_index"], table["last_session_index"]))
        ]
    for column in ("n_adjacent_edges", "n_gap_edges", "min_score", "mean_score", "min_dice", "max_distance_um", "max_ambiguity", "cycle_agreement_fraction"):
        if column not in table.columns:
            table[column] = np.nan
    for column in ("used_gap_bridge", "has_cycle_conflict", "cycle_unchecked"):
        if column not in table.columns:
            table[column] = False if column != "cycle_unchecked" else True
    return table


def _feature_lookup(features: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    if features.empty:
        return {}
    if "session_id" not in features.columns or "label" not in features.columns:
        raise ValueError("roi_features.csv must contain session_id and label columns")
    return {(str(row.session_id), int(row.label)): row for row in features.itertuples(index=False)}


def _feature_row(features: pd.DataFrame, session_id: str, label: int) -> pd.Series | None:
    if features.empty:
        return None
    subset = features.loc[(features["session_id"].astype(str) == str(session_id)) & (features["label"].astype(int) == int(label))]
    return None if subset.empty else subset.iloc[0]


def _owner_map(tracks: pd.DataFrame, sessions: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    output: dict[tuple[str, int], pd.Series] = {}
    for _, track in tracks.iterrows():
        for row in sessions.itertuples():
            value = track.get(_roi_column(str(row.session_id)), pd.NA)
            if _present(value):
                output[(str(row.session_id), int(value))] = track
    return output


def _state_merge(endpoint: dict[str, Any], state: pd.DataFrame) -> None:
    if state.empty or "track_uid" not in state.columns or "session_index" not in state.columns:
        return
    subset = state.loc[(state["track_uid"].astype(str) == str(endpoint["track_uid"])) & (state["session_index"].astype(int) == int(endpoint["end_session_index"]))]
    if subset.empty:
        return
    for column in ("eclipse_z", "eclipse_core_state", "green", "red"):
        if column in subset.columns:
            endpoint[column] = subset.iloc[0][column]


def detect_endpoint_events(
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    policy: str = "graph",
    lookahead: int = 3,
    state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect observations followed by an absent next-session ROI."""

    if lookahead < 1:
        raise ValueError("lookahead must be >= 1")
    tracks = _prepare_tracks(tracks, sessions, policy)
    state = state if state is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for track_index, (_, track) in enumerate(tracks.iterrows(), start=1):
        for session_position, session in enumerate(sessions.itertuples(index=False)):
            session_id = str(session.session_id)
            label_value = track.get(_roi_column(session_id), pd.NA)
            if not _present(label_value):
                continue
            next_present = False
            if session_position + 1 < len(sessions):
                next_value = track.get(_roi_column(str(sessions.iloc[session_position + 1]["session_id"])), pd.NA)
                next_present = _present(next_value)
            if next_present:
                continue
            label = int(label_value)
            feature = _feature_row(features, session_id, label)
            endpoint_id = f"endpoint_{track_index:06d}_{session_position:04d}_{label}"
            row: dict[str, Any] = {
                "endpoint_id": endpoint_id,
                "track_uid": _text(track.get("track_uid")),
                "cluster_id": track.get("cluster_id", pd.NA),
                "end_session_index": int(session.session_index),
                "end_session_id": session_id,
                "end_acquisition_date": _session_date(pd.Series(session._asdict())),
                "end_label": label,
                "first_session_index": _int(track.get("first_session_index"), -1),
                "last_session_index": _int(track.get("last_session_index"), -1),
                "n_days_present": _int(track.get("n_days_present"), 0),
                "missing_internal_days": _int(track.get("missing_internal_days"), 0),
                "n_adjacent_edges": _int(track.get("n_adjacent_edges")),
                "n_gap_edges": _int(track.get("n_gap_edges")),
                "used_gap_bridge": _bool(track.get("used_gap_bridge")),
                "min_score": _float(track.get("min_score")),
                "mean_score": _float(track.get("mean_score")),
                "min_dice": _float(track.get("min_dice")),
                "max_distance_um": _float(track.get("max_distance_um")),
                "max_ambiguity": _float(track.get("max_ambiguity")),
                "track_match_source": _text(track.get("match_policy", policy), policy),
                "cycle_agreement_fraction": _float(track.get("cycle_agreement_fraction")),
                "has_cycle_conflict": _bool(track.get("has_cycle_conflict")),
                "cycle_unchecked": _bool(track.get("cycle_unchecked"), True),
                "eclipse_z": np.nan, "eclipse_core_state": "", "green": np.nan, "red": np.nan,
                "centroid_z_um": np.nan, "centroid_y_um": np.nan, "centroid_x_um": np.nan,
                "volume_um3": np.nan, "touches_z_edge": False, "touches_xy_edge": False,
            }
            if feature is not None:
                for column in ("centroid_z_um", "centroid_y_um", "centroid_x_um", "volume_um3", "touches_z_edge", "touches_xy_edge"):
                    if column in feature.index:
                        row[column] = feature[column]
            returns: list[tuple[int, int]] = []
            for future_position in range(session_position + 1, min(len(sessions), session_position + lookahead + 1)):
                future_id = str(sessions.iloc[future_position]["session_id"])
                value = track.get(_roi_column(future_id), pd.NA)
                if _present(value):
                    returns.append((future_position, int(value)))
            row["same_track_returns"] = bool(returns)
            row["same_track_return_gap"] = int(returns[0][0] - session_position) if returns else np.nan
            row["same_track_return_session_index"] = int(sessions.iloc[returns[0][0]]["session_index"]) if returns else np.nan
            _state_merge(row, state)
            rows.append(row)
    return pd.DataFrame(rows, columns=ENDPOINT_COLUMNS)


def _transform_row_map(transforms: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    if transforms.empty:
        return {}
    required = {"day_a", "day_b"}
    if not required.issubset(transforms.columns):
        raise ValueError("pairwise_transforms.csv must contain day_a and day_b columns")
    return {(str(row.day_a), str(row.day_b)): row for _, row in transforms.iterrows()}


def _forward_transform(
    source_position: int,
    target_position: int,
    sessions: pd.DataFrame,
    transform_map: dict[tuple[str, str], pd.Series],
) -> tuple[RestrictedTransform, str, bool] | None:
    gap = target_position - source_position
    if gap < 1:
        return None
    source_id = str(sessions.iloc[source_position]["session_id"])
    target_id = str(sessions.iloc[target_position]["session_id"])
    if gap <= 2:
        stored = transform_map.get((source_id, target_id))
        if stored is None:
            return None
        try:
            inverse = invert_restricted_transform(stored)
        except ValueError:
            return None
        return inverse, "direct_stored", _first_fallback(stored) is None
    if gap != 3:
        return None
    components: list[pd.Series] = []
    for position in range(source_position, target_position):
        pair = (str(sessions.iloc[position]["session_id"]), str(sessions.iloc[position + 1]["session_id"]))
        stored = transform_map.get(pair)
        if stored is None:
            return None
        components.append(stored)
    composed = compose_transforms(components[0], components[1])
    composed = compose_transforms(composed, components[2])
    try:
        inverse = invert_restricted_transform(composed)
    except ValueError:
        return None
    return inverse, "composed_adjacent", _first_fallback(*components) is None


def _evidence_maps(
    candidates: pd.DataFrame,
    high: pd.DataFrame,
    balanced: pd.DataFrame,
    graph: pd.DataFrame,
) -> tuple[dict[tuple[str, str, int, int], pd.Series], set[tuple[str, str, int, int]], set[tuple[str, str, int, int]], set[tuple[str, str, int, int]]]:
    candidate_map: dict[tuple[str, str, int, int], pd.Series] = {}
    for _, row in candidates.iterrows():
        key = (_text(row.get("day_a")), _text(row.get("day_b")), _int(row.get("label_a"), -1) or -1, _int(row.get("label_b"), -1) or -1)
        candidate_map[key] = row

    def keys(table: pd.DataFrame) -> set[tuple[str, str, int, int]]:
        if table.empty:
            return set()
        return {
            (_text(row.get("day_a")), _text(row.get("day_b")), _int(row.get("label_a"), -1) or -1, _int(row.get("label_b"), -1) or -1)
            for _, row in table.iterrows()
        }
    return candidate_map, keys(high), keys(balanced), keys(graph)


def _date_gap(sessions: pd.DataFrame, source_position: int, target_position: int) -> float:
    try:
        left = date.fromisoformat(str(sessions.iloc[source_position]["acquisition_date"])[:10])
        right = date.fromisoformat(str(sessions.iloc[target_position]["acquisition_date"])[:10])
        return float((right - left).days)
    except (KeyError, TypeError, ValueError):
        return float(target_position - source_position)


def _target_feature_table(features: pd.DataFrame, session_id: str) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    subset = features.loc[features["session_id"].astype(str).eq(str(session_id))].copy()
    if subset.empty:
        return subset
    for column in ("centroid_z", "centroid_y", "centroid_x"):
        if column not in subset.columns:
            raise ValueError(f"roi_features.csv is missing {column}")
    return subset.sort_values("label").reset_index(drop=True)


def _search_candidates_for_pair(
    *,
    endpoint_id: str,
    source_track_uid: str,
    source_position: int,
    source_label: int,
    target_position: int,
    source_feature: pd.Series,
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    transform_map: dict[tuple[str, str], pd.Series],
    candidate_map: dict[tuple[str, str, int, int], pd.Series],
    high_keys: set[tuple[str, str, int, int]],
    balanced_keys: set[tuple[str, str, int, int]],
    graph_keys: set[tuple[str, str, int, int]],
    owner_map: dict[tuple[str, int], pd.Series],
    spacing: VoxelSpacing,
    search_radius_um: float,
) -> tuple[list[dict[str, Any]], str]:
    projected = _forward_transform(source_position, target_position, sessions, transform_map)
    if projected is None:
        source_id = str(sessions.iloc[source_position]["session_id"])
        target_id = str(sessions.iloc[target_position]["session_id"])
        if (source_id, target_id) in transform_map or target_position - source_position == 3:
            return [], "transform_unreliable"
        return [], "insufficient_input"
    forward, transform_source, reliable = projected
    source_coords = np.array([
        _float(source_feature.get("centroid_z")), _float(source_feature.get("centroid_y")), _float(source_feature.get("centroid_x")),
    ])
    if not np.isfinite(source_coords).all():
        return [], "insufficient_input"
    predicted = apply_transform_b_to_a(source_coords, forward)
    target_id = str(sessions.iloc[target_position]["session_id"])
    out_of_fov = False
    mask_path = _value(sessions.iloc[target_position], "mask_path", "")
    mask_path_text = "" if mask_path is None or pd.isna(mask_path) else str(mask_path)
    if mask_path_text and Path(mask_path_text).is_file():
        try:
            shape = tuple(tifffile.memmap(Path(mask_path_text)).shape)
            out_of_fov = any(value < 0 or value >= limit for value, limit in zip(predicted, shape, strict=True))
        except Exception:
            out_of_fov = False
    target_features = _target_feature_table(features, target_id)
    if target_features.empty:
        return [], "no_mask_near_prediction"
    target_coords = target_features[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    target_physical = target_coords * spacing.as_zyx_array()
    predicted_physical = predicted * spacing.as_zyx_array()
    tree = cKDTree(target_physical)
    nearby = [int(index) for index in tree.query_ball_point(predicted_physical, r=float(search_radius_um))]
    distances = np.linalg.norm(target_physical - predicted_physical, axis=1)
    nearby.sort(key=lambda index: (float(distances[index]), int(target_features.iloc[index]["label"])))
    nearest_distances, _ = tree.query(predicted_physical, k=min(2, len(target_features)))
    nearest_distances = np.atleast_1d(np.asarray(nearest_distances, dtype=float))
    second_margin = float(nearest_distances[1] - nearest_distances[0]) if len(nearest_distances) > 1 else np.nan
    source_id = str(sessions.iloc[source_position]["session_id"])
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(nearby, start=1):
        target = target_features.iloc[index]
        target_label = int(target["label"])
        key = (source_id, target_id, int(source_label), target_label)
        evidence = candidate_map.get(key)
        owner = owner_map.get((target_id, target_label))
        target_track_uid = _text(owner.get("track_uid")) if owner is not None else ""
        distance = float(distances[index])
        source_volume = _float(source_feature.get("volume_um3"))
        target_volume = _float(target.get("volume_um3"))
        rows.append({
            "endpoint_id": endpoint_id,
            "source_track_uid": source_track_uid,
            "source_session_index": int(sessions.iloc[source_position]["session_index"]),
            "source_session_id": source_id,
            "source_label": int(source_label),
            "target_session_index": int(sessions.iloc[target_position]["session_index"]),
            "target_session_id": target_id,
            "target_label": target_label,
            "target_track_uid": target_track_uid,
            "session_gap": int(target_position - source_position),
            "elapsed_day_gap": _date_gap(sessions, source_position, target_position),
            "predicted_z_um": float(predicted_physical[0]), "predicted_y_um": float(predicted_physical[1]), "predicted_x_um": float(predicted_physical[2]),
            "target_z_um": float(target["centroid_z"] * spacing.z_um), "target_y_um": float(target["centroid_y"] * spacing.y_um), "target_x_um": float(target["centroid_x"] * spacing.x_um),
            "projected_distance_um": distance,
            "source_volume_um3": source_volume, "target_volume_um3": target_volume,
            "volume_ratio": min(source_volume, target_volume) / max(source_volume, target_volume) if np.isfinite(source_volume) and np.isfinite(target_volume) and max(source_volume, target_volume) > 0 else np.nan,
            "target_track_starts_here": bool(owner is not None and _int(owner.get("first_session_index"), -1) == int(target_position)),
            "target_is_singleton": bool(owner is not None and _int(owner.get("n_days_present"), 0) == 1),
            "target_track_match_source": _text(owner.get("match_policy")) if owner is not None else "",
            "existing_candidate_found": evidence is not None,
            "existing_high_match": key in high_keys,
            "existing_balanced_match": key in balanced_keys,
            "existing_graph_match": key in graph_keys,
            "dice": _float(evidence.get("dice")) if evidence is not None else np.nan,
            "iou": _float(evidence.get("iou")) if evidence is not None else np.nan,
            "ambiguity": _float(evidence.get("ambiguity")) if evidence is not None else np.nan,
            "base_score": _float(evidence.get("base_score", evidence.get("score"))) if evidence is not None else np.nan,
            "refined_score": _float(evidence.get("refined_score", evidence.get("score"))) if evidence is not None else np.nan,
            "candidate_source": _text(evidence.get("candidate_source")) if evidence is not None else "evaluator_kdtree",
            "graph_status": _text(evidence.get("graph_status")) if evidence is not None else "",
            "graph_support_count": _float(evidence.get("graph_support_count")) if evidence is not None else np.nan,
            "graph_support_fraction": _float(evidence.get("graph_support_fraction")) if evidence is not None else np.nan,
            "graph_residual_median_um": _float(evidence.get("graph_residual_median_um")) if evidence is not None else np.nan,
            "graph_inlier_fraction": _float(evidence.get("graph_inlier_fraction")) if evidence is not None else np.nan,
            "target_rank_by_distance": int(rank),
            "nearest_second_nearest_margin_um": second_margin,
            "transform_source": transform_source,
            "transform_reliable": bool(reliable),
        })
    if not reliable:
        return rows, "transform_unreliable"
    return rows, "edge_or_out_of_fov" if out_of_fov else "ok"


def search_endpoint_candidates(
    endpoints: pd.DataFrame,
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    transforms: pd.DataFrame,
    *,
    policy: str = "graph",
    lookahead: int = 3,
    search_radius_um: float = 15.0,
    spacing: VoxelSpacing | None = None,
    candidates: pd.DataFrame | None = None,
    high_matches: pd.DataFrame | None = None,
    balanced_matches: pd.DataFrame | None = None,
    graph_matches: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Search future ROI centroids without assigning any track links."""

    if search_radius_um <= 0:
        raise ValueError("search_radius_um must be positive")
    spacing = spacing or VoxelSpacing()
    tracks = _prepare_tracks(tracks, sessions, policy)
    transform_map = _transform_row_map(transforms)
    candidate_map, high_keys, balanced_keys, graph_keys = _evidence_maps(
        candidates if candidates is not None else pd.DataFrame(),
        high_matches if high_matches is not None else pd.DataFrame(),
        balanced_matches if balanced_matches is not None else pd.DataFrame(),
        graph_matches if graph_matches is not None else pd.DataFrame(),
    )
    owner_map = _owner_map(tracks, sessions)
    rows: list[dict[str, Any]] = []
    status: dict[str, list[str]] = {}
    for _, endpoint in endpoints.iterrows():
        endpoint_id = _text(endpoint.get("endpoint_id"))
        source_position = int(sessions.index[sessions["session_index"].astype(int).eq(int(endpoint["end_session_index"]))][0])
        source_id = str(sessions.iloc[source_position]["session_id"])
        source_feature = _feature_row(features, source_id, int(endpoint["end_label"]))
        if source_feature is None:
            status.setdefault(endpoint_id, []).append("insufficient_input")
            continue
        endpoint_status: list[str] = []
        for target_position in range(source_position + 1, min(len(sessions), source_position + lookahead + 1)):
            pair_rows, pair_status = _search_candidates_for_pair(
                endpoint_id=endpoint_id, source_track_uid=_text(endpoint.get("track_uid")),
                source_position=source_position, source_label=int(endpoint["end_label"]), target_position=target_position,
                source_feature=source_feature, tracks=tracks, features=features, sessions=sessions,
                transform_map=transform_map, candidate_map=candidate_map, high_keys=high_keys,
                balanced_keys=balanced_keys, graph_keys=graph_keys, owner_map=owner_map,
                spacing=spacing, search_radius_um=search_radius_um,
            )
            rows.extend(pair_rows)
            endpoint_status.append(pair_status)
        status[endpoint_id] = endpoint_status
    table = pd.DataFrame(rows, columns=ENDPOINT_CANDIDATE_COLUMNS)
    if not table.empty:
        table = table.sort_values(["endpoint_id", "target_session_index", "target_rank_by_distance", "target_label"]).reset_index(drop=True)
    return table, status


def classify_endpoint_events(
    endpoints: pd.DataFrame,
    candidates: pd.DataFrame,
    status: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Assign deterministic triage labels; manual labels remain blank."""

    status = status or {}
    rows: list[dict[str, Any]] = []
    for _, endpoint in endpoints.iterrows():
        endpoint_id = _text(endpoint.get("endpoint_id"))
        subset = candidates.loc[candidates["endpoint_id"].astype(str).eq(endpoint_id)] if not candidates.empty else pd.DataFrame()
        statuses = status.get(endpoint_id, [])
        if _bool(endpoint.get("same_track_returns")):
            classification = "same_track_gap_recovered"
        elif any(value == "transform_unreliable" for value in statuses) or (not subset.empty and not subset["transform_reliable"].all()):
            classification = "transform_unreliable"
        elif any(value == "edge_or_out_of_fov" for value in statuses) or _bool(endpoint.get("touches_z_edge")) or _bool(endpoint.get("touches_xy_edge")):
            classification = "edge_or_out_of_fov"
        elif not subset.empty:
            unique_targets = subset.drop_duplicates(["target_session_index", "target_label"])
            if len(unique_targets) > 1:
                classification = "multiple_nearby_candidates"
            elif _bool(unique_targets.iloc[0].get("target_is_singleton")):
                classification = "nearby_singleton_candidate"
            else:
                classification = "nearby_new_track_candidate"
        elif not statuses or all(value == "insufficient_input" for value in statuses):
            classification = "insufficient_input"
        elif any(value == "no_mask_near_prediction" for value in statuses):
            classification = "no_mask_near_prediction"
        else:
            classification = "no_mask_near_prediction"
        rows.append({
            "endpoint_id": endpoint_id, "track_uid": _text(endpoint.get("track_uid")),
            "end_session_index": _int(endpoint.get("end_session_index")), "end_session_id": _text(endpoint.get("end_session_id")),
            "end_label": _int(endpoint.get("end_label")), "classification": classification,
            "candidate_count": int(len(subset)), "same_track_returns": _bool(endpoint.get("same_track_returns")),
            "manual_class": "", "manual_target_track_uid": "", "manual_target_session_index": "",
            "manual_target_label": "", "manual_confidence": "", "reviewer_notes": "",
        })
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


def _synthetic_trusted(track: pd.Series, gap: int) -> bool:
    if _int(track.get("n_days_present"), 0) < gap + 1:
        return False
    if _bool(track.get("has_cycle_conflict")) or _bool(track.get("contains_transform_fallback_edge")):
        return False
    for field, threshold, direction in (("min_score", 0.35, "min"), ("min_dice", 0.05, "min"), ("max_distance_um", 20.0, "max")):
        value = _float(track.get(field))
        if np.isfinite(value) and ((direction == "min" and value < threshold) or (direction == "max" and value > threshold)):
            return False
    return True


def build_synthetic_gap_benchmark(
    tracks: pd.DataFrame,
    endpoints: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    transforms: pd.DataFrame,
    *,
    policy: str = "graph",
    search_radius_um: float = 15.0,
    spacing: VoxelSpacing | None = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Benchmark one- and two-missing-session projections on trusted tracks."""

    del endpoints  # Endpoint observations are not ground truth for this benchmark.
    spacing = spacing or VoxelSpacing()
    tracks = _prepare_tracks(tracks, sessions, policy)
    transform_map = _transform_row_map(transforms)
    owner_map = _owner_map(tracks, sessions)
    rows: list[dict[str, Any]] = []
    for track_index, (_, track) in enumerate(tracks.iterrows()):
        for gap in (2, 3):
            if not _synthetic_trusted(track, gap):
                continue
            for source_position in range(0, len(sessions) - gap):
                source_id = str(sessions.iloc[source_position]["session_id"])
                target_id = str(sessions.iloc[source_position + gap]["session_id"])
                source_value = track.get(_roi_column(source_id), pd.NA)
                target_value = track.get(_roi_column(target_id), pd.NA)
                if not (_present(source_value) and _present(target_value)):
                    continue
                source_feature = _feature_row(features, source_id, int(source_value))
                target_feature = _feature_row(features, target_id, int(target_value))
                if source_feature is None or target_feature is None:
                    continue
                pair_rows, pair_status = _search_candidates_for_pair(
                    endpoint_id=f"synthetic_{track_index}_{source_position}_{gap}",
                    source_track_uid=_text(track.get("track_uid")), source_position=source_position,
                    source_label=int(source_value), target_position=source_position + gap,
                    source_feature=source_feature, tracks=tracks, features=features, sessions=sessions,
                    transform_map=transform_map, candidate_map={}, high_keys=set(), balanced_keys=set(), graph_keys=set(),
                    owner_map=owner_map, spacing=spacing, search_radius_um=search_radius_um,
                )
                if not pair_rows or pair_status != "ok":
                    continue
                true_label = int(target_value)
                ranked = sorted(pair_rows, key=lambda row: (float(row["projected_distance_um"]), int(row["target_label"])))
                true = next((row for row in ranked if int(row["target_label"]) == true_label), None)
                if true is None:
                    projected_distance = np.nan
                    true_rank = np.nan
                    in_radius = False
                else:
                    projected_distance = float(true["projected_distance_um"])
                    true_rank = int(ranked.index(true) + 1)
                    in_radius = True
                source_edge = _bool(source_feature.get("touches_z_edge")) or _bool(source_feature.get("touches_xy_edge"))
                target_edge = _bool(target_feature.get("touches_z_edge")) or _bool(target_feature.get("touches_xy_edge"))
                rows.append({
                    "case_id": f"gap_{gap}_{track_index}_{source_position}", "track_uid": _text(track.get("track_uid")),
                    "source_session_index": int(sessions.iloc[source_position]["session_index"]),
                    "target_session_index": int(sessions.iloc[source_position + gap]["session_index"]),
                    "true_target_in_search_radius": bool(in_radius), "true_target_rank": true_rank,
                    "top1_correct": bool(in_radius and true_rank == 1), "candidate_count": len(ranked),
                    "nearest_second_margin_um": ranked[0].get("nearest_second_nearest_margin_um", np.nan),
                    "projected_distance_um": projected_distance, "session_gap": gap,
                    "elapsed_day_gap": _date_gap(sessions, source_position, source_position + gap),
                    "source_volume": _float(source_feature.get("volume_um3")),
                    "local_density": float(len(ranked) / ((4.0 / 3.0) * np.pi * search_radius_um ** 3)),
                    "edge_status": "edge" if source_edge or target_edge else "interior",
                    "session_pair": f"{source_id}->{target_id}",
                })
    table = pd.DataFrame(rows, columns=SYNTHETIC_COLUMNS)
    if not table.empty:
        # Seed is part of the contract even when the deterministic input order needs no sampling.
        table = table.iloc[np.random.default_rng(int(random_seed)).permutation(len(table))].sort_values(["session_gap", "source_session_index", "track_uid"]).reset_index(drop=True)
    return table


def _spatial_stratum(endpoint: pd.Series, features: pd.DataFrame, session_id: str) -> str:
    feature = _feature_row(features, session_id, int(endpoint["end_label"]))
    if feature is None:
        return "unknown"
    y = _float(feature.get("centroid_y")); x = _float(feature.get("centroid_x"))
    all_features = features.loc[features["session_id"].astype(str).eq(session_id)] if not features.empty else pd.DataFrame()
    if all_features.empty:
        return "unknown"
    y0, y1 = _float(all_features["centroid_y"].min(), 0.0), _float(all_features["centroid_y"].max(), 1.0)
    x0, x1 = _float(all_features["centroid_x"].min(), 0.0), _float(all_features["centroid_x"].max(), 1.0)
    gy = int(np.clip(np.floor(8 * (y - y0) / max(y1 - y0, 1e-9)), 0, 7))
    gx = int(np.clip(np.floor(8 * (x - x0) / max(x1 - x0, 1e-9)), 0, 7))
    return f"grid_{gy}x{gx}"


def build_manual_review_manifest(
    endpoints: pd.DataFrame,
    features: pd.DataFrame,
    *,
    max_review_panels: int | None = 100,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Select a deterministic spatial sample and, when present, Low-state rows."""

    if endpoints.empty:
        return pd.DataFrame(columns=["endpoint_id", "sample_type", "sampling_stratum", "spatial_stratum", "state_stratum"])
    rng = np.random.default_rng(int(random_seed))
    work = endpoints.copy()
    work["spatial_stratum"] = [_spatial_stratum(row, features, str(row["end_session_id"])) for _, row in work.iterrows()]
    state_column = "eclipse_core_state"
    general_indices: list[int] = []
    for _, group in work.groupby("spatial_stratum", sort=True):
        values = group.index.to_numpy()
        general_indices.append(int(values[rng.integers(len(values))]))
    selected: list[dict[str, Any]] = []
    for index in general_indices:
        row = work.loc[index].to_dict()
        row.update({"sample_type": "general_spatial", "sampling_stratum": "general", "state_stratum": ""})
        selected.append(row)
    if state_column in work.columns and work[state_column].astype(str).str.strip().ne("").any():
        low = work.loc[work[state_column].astype(str).str.lower().eq("low")]
        middle = work.loc[work[state_column].astype(str).str.lower().eq("middle")]
        controls = middle.groupby("end_session_index", sort=True).head(1)
        for _, row in pd.concat([low, controls]).drop_duplicates("endpoint_id").iterrows():
            item = row.to_dict()
            state = "low" if str(row[state_column]).lower() == "low" else "middle_control"
            item.update({"sample_type": "state_dependent", "sampling_stratum": state, "state_stratum": state})
            selected.append(item)
    manifest = pd.DataFrame(selected).drop_duplicates("endpoint_id", keep="first")
    if max_review_panels is not None and max_review_panels >= 0 and len(manifest) > int(max_review_panels):
        order = rng.permutation(len(manifest))[: int(max_review_panels)]
        manifest = manifest.iloc[order].sort_values(["sample_type", "endpoint_id"]).reset_index(drop=True)
    if manifest.empty:
        return pd.DataFrame(columns=["endpoint_id", "sample_type", "sampling_stratum", "spatial_stratum", "state_stratum"])
    return manifest.reset_index(drop=True)


def build_dropout_summary(
    endpoints: pd.DataFrame,
    classifications: pd.DataFrame,
    sessions: pd.DataFrame,
    tracks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_counts: dict[int, int] = {int(value): 0 for value in sessions["session_index"]}
    if tracks is None:
        for _, endpoint in endpoints.iterrows():
            all_counts[int(endpoint["end_session_index"])] = all_counts.get(int(endpoint["end_session_index"]), 0) + 1
    else:
        for _, track in tracks.iterrows():
            for _, session in sessions.iterrows():
                if _present(track.get(_roi_column(str(session["session_id"])), pd.NA)):
                    index = int(session["session_index"])
                    all_counts[index] = all_counts.get(index, 0) + 1
    grouped = classifications.groupby(["end_session_index", "end_session_id"], dropna=False) if not classifications.empty else []
    grouped_map = {(int(index), str(session)): frame for (index, session), frame in grouped}
    for _, session in sessions.iterrows():
        index = int(session["session_index"]); session_id = str(session["session_id"])
        frame = grouped_map.get((index, session_id), pd.DataFrame())
        rows.append({
            "session_index": index, "session_id": session_id, "n_endpoint_observations": int(len(frame)),
            "n_endpoints": int(len(frame)), "endpoint_rate": float(len(frame) / all_counts[index]) if all_counts[index] else np.nan,
            "same_track_gap_recovered": int((frame["classification"] == "same_track_gap_recovered").sum()) if not frame.empty else 0,
            "nearby_new_track_candidate": int((frame["classification"] == "nearby_new_track_candidate").sum()) if not frame.empty else 0,
            "nearby_singleton_candidate": int((frame["classification"] == "nearby_singleton_candidate").sum()) if not frame.empty else 0,
            "multiple_nearby_candidates": int((frame["classification"] == "multiple_nearby_candidates").sum()) if not frame.empty else 0,
            "manual_labels_available": False,
        })
    return pd.DataFrame(rows)


def build_runtime_summary(match_dir: str | Path, policy: str = "graph") -> pd.DataFrame:
    root = Path(match_dir)
    summary = _read_csv(root / "pairwise_summary.csv")
    graph_summary = _read_csv(root / f"pairwise_summary_{policy}.csv")
    candidates = _read_csv(root / "pairwise_candidates.csv")
    if summary.empty:
        return pd.DataFrame(columns=RUNTIME_COLUMNS)
    candidate_counts = candidates.groupby(["day_a", "day_b"]).size().to_dict() if not candidates.empty and {"day_a", "day_b"}.issubset(candidates.columns) else {}
    graph_seconds = graph_summary.set_index(["day_a", "day_b"]).get("elapsed_sec", pd.Series(dtype=float)).to_dict() if not graph_summary.empty and {"day_a", "day_b"}.issubset(graph_summary.columns) else {}
    rows = []
    for _, row in summary.iterrows():
        key = (_text(row.get("day_a")), _text(row.get("day_b")))
        rows.append({
            "session_pair": f"{key[0]}->{key[1]}", "day_a": key[0], "day_b": key[1],
            "pair_gap": _int(row.get("pair_gap")), "elapsed_sec": _float(row.get("elapsed_sec")),
            "n_a": _int(row.get("n_a")), "n_b": _int(row.get("n_b")),
            "candidate_count": int(candidate_counts.get(key, 0)),
            "transform_method": _text(row.get("transform_method")),
            "transform_fallback_reason": _text(row.get("transform_fallback_reason")),
            "graph_stage_seconds": _float(graph_seconds.get(key)),
        })
    return pd.DataFrame(rows, columns=RUNTIME_COLUMNS).sort_values(["pair_gap", "day_a", "day_b"]).reset_index(drop=True)


def _package_versions() -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "tifffile"):
        try:
            output[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            output[package] = None
    return output


def _write_csv(path: Path, table: pd.DataFrame, columns: list[str]) -> None:
    output = table.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    output.loc[:, columns].to_csv(path, index=False)


def _merge_extraction_enrichment(endpoints: pd.DataFrame, extraction_dir: str | Path | None) -> pd.DataFrame:
    """Copy stable-key QC fields when an extraction directory is supplied."""

    if extraction_dir is None or endpoints.empty:
        return endpoints
    root = Path(extraction_dir).resolve()
    output = endpoints.copy()
    names = (
        "matched_track_qc_summary.csv", "matched_roi_geometry_qc_long.csv",
        "matched_roi_log_ratio_metrics_all_observed.csv", "matched_roi_trajectory_observations_eligible.csv",
        "graph_affine_agreement_track_metadata.csv",
    )
    for name in names:
        table = _read_csv(root / name)
        if table.empty or not {"track_uid", "session_index"}.issubset(table.columns):
            continue
        source = table.drop_duplicates(["track_uid", "session_index"]).copy()
        lookup = source.set_index([source["track_uid"].astype(str), source["session_index"].astype(int)])
        for column in ("green", "red", "eclipse_z", "eclipse_core_state"):
            if column not in source.columns:
                continue
            if column not in output.columns:
                output[column] = np.nan
            for index, row in output.iterrows():
                key = (_text(row.get("track_uid")), _int(row.get("end_session_index"), -1))
                if key not in lookup.index:
                    continue
                value = lookup.loc[key, column]
                current = output.at[index, column]
                if pd.notna(value) and (pd.isna(current) or str(current).strip() == ""):
                    output.at[index, column] = value
    return output


def _state_endpoint_metrics(endpoints: pd.DataFrame, state: pd.DataFrame) -> dict[str, Any]:
    if state.empty or endpoints.empty or "eclipse_core_state" not in endpoints.columns:
        return {"available": False}
    states = endpoints["eclipse_core_state"].astype(str).str.strip().str.lower()
    if not states.isin({"low", "middle"}).any():
        return {"available": False}
    state_column = "eclipse_core_state" if "eclipse_core_state" in state.columns else "eclipse_state_bin" if "eclipse_state_bin" in state.columns else None
    if state_column is None:
        return {"available": False}
    denominator_states = state[state_column].astype(str).str.strip().str.lower().value_counts()
    endpoint_counts = states.value_counts()
    rates = {name: float(endpoint_counts.get(name, 0) / denominator_states.get(name, 0)) for name in ("low", "middle") if denominator_states.get(name, 0)}
    low_rate = rates.get("low", np.nan); middle_rate = rates.get("middle", np.nan)
    return {
        "available": True,
        "P(dropout at t+1 | Low)": low_rate,
        "P(dropout at t+1 | Middle)": middle_rate,
        "risk_ratio_low_to_middle": float(low_rate / middle_rate) if np.isfinite(low_rate) and np.isfinite(middle_rate) and middle_rate else np.nan,
        "risk_difference_low_minus_middle": float(low_rate - middle_rate) if np.isfinite(low_rate) and np.isfinite(middle_rate) else np.nan,
        "denominator_note": "Rates use endpoint observations carrying the supplied state; manual segmentation labels are not inferred.",
    }


def evaluate_daywise_tracking(
    match_dir: str | Path,
    output_dir: str | Path,
    *,
    policy: str = "graph",
    lookahead: int = 3,
    search_radius_um: float = 15.0,
    crop_radius_um: float = 45.0,
    z_radius: int = 1,
    extraction_dir: str | Path | None = None,
    state_csv: str | Path | None = None,
    max_review_panels: int | None = 100,
    random_seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run all read-only evaluator stages and write tables/PNG review figures."""

    match_dir = Path(match_dir).resolve(); output_dir = Path(output_dir).resolve()
    if match_dir == output_dir:
        raise ValueError("output_dir must be separate from match_dir")
    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_evaluator_inputs(match_dir, policy)
    if discovered["missing_required"]:
        raise FileNotFoundError("Missing evaluator inputs: " + ", ".join(discovered["missing_required"]))

    manifest = _session_table(_read_csv(match_dir / "session_manifest_resolved.csv"))
    features = _read_csv(match_dir / "roi_features.csv")
    transforms = _read_csv(match_dir / "pairwise_transforms.csv")
    tracks = _read_csv(match_dir / f"tracks_{policy}.csv")
    pair_candidates = _read_csv(match_dir / "pairwise_candidates.csv")
    high = _read_csv(match_dir / "pairwise_matches_high.csv")
    balanced = _read_csv(match_dir / "pairwise_matches_balanced.csv")
    graph = _read_csv(match_dir / "pairwise_matches_graph.csv")
    if state_csv:
        state_path = Path(state_csv).resolve()
        if not state_path.is_file():
            raise FileNotFoundError(f"State CSV was not found: {state_path}")
        state = _read_csv(state_path)
        if not {"track_uid", "session_index"}.issubset(state.columns):
            raise ValueError("state CSV must contain track_uid and session_index columns")
    else:
        state = pd.DataFrame()
    endpoints = detect_endpoint_events(tracks, features, manifest, policy=policy, lookahead=lookahead, state=state)
    endpoints = _merge_extraction_enrichment(endpoints, extraction_dir)
    endpoint_candidates, candidate_status = search_endpoint_candidates(
        endpoints, tracks, features, manifest, transforms, policy=policy, lookahead=lookahead,
        search_radius_um=search_radius_um, candidates=pair_candidates, high_matches=high,
        balanced_matches=balanced, graph_matches=graph,
    )
    classifications = classify_endpoint_events(endpoints, endpoint_candidates, candidate_status)
    previous_classifications = _read_csv(output_dir / "endpoint_classification.csv")
    if not previous_classifications.empty and "endpoint_id" in previous_classifications.columns:
        manual_columns = [column for column in CLASSIFICATION_COLUMNS if column.startswith("manual_") or column == "reviewer_notes"]
        old = previous_classifications.set_index(previous_classifications["endpoint_id"].astype(str))
        for index, endpoint_id in classifications["endpoint_id"].astype(str).items():
            if endpoint_id not in old.index:
                continue
            for column in manual_columns:
                value = old.loc[endpoint_id].get(column, "")
                if pd.notna(value) and str(value).strip():
                    classifications.at[index, column] = value
    synthetic = build_synthetic_gap_benchmark(tracks, endpoints, features, manifest, transforms, policy=policy, search_radius_um=search_radius_um, random_seed=random_seed)
    review_manifest = build_manual_review_manifest(endpoints, features, max_review_panels=max_review_panels, random_seed=random_seed)
    dropout = build_dropout_summary(endpoints, classifications, manifest, tracks)
    runtime = build_runtime_summary(match_dir, policy)

    _write_csv(output_dir / "endpoint_events.csv", endpoints, ENDPOINT_COLUMNS)
    _write_csv(output_dir / "endpoint_candidates.csv", endpoint_candidates, ENDPOINT_CANDIDATE_COLUMNS)
    _write_csv(output_dir / "endpoint_classification.csv", classifications, CLASSIFICATION_COLUMNS)
    _write_csv(output_dir / "synthetic_gap_benchmark.csv", synthetic, SYNTHETIC_COLUMNS)
    _write_csv(output_dir / "manual_review_manifest.csv", review_manifest, list(review_manifest.columns) if not review_manifest.empty else ["endpoint_id", "sample_type", "sampling_stratum", "spatial_stratum", "state_stratum"])
    dropout.to_csv(output_dir / "dropout_summary.csv", index=False)
    runtime.to_csv(output_dir / "matching_runtime_summary.csv", index=False)

    try:
        from endpoint_review_plots import plot_endpoint_contact_sheet, plot_runtime_summaries
    except ImportError:  # pragma: no cover - package import path
        from plotting.endpoint_review_plots import plot_endpoint_contact_sheet, plot_runtime_summaries
    plot_dir = output_dir / "review_panels"; plot_dir.mkdir(exist_ok=True)
    if not review_manifest.empty:
        plot_endpoint_contact_sheet(
            review_manifest, manifest, features, endpoint_candidates, transforms,
            output_dir=plot_dir, max_panels=max_review_panels, crop_radius_um=crop_radius_um, z_radius=z_radius,
        )
    plot_runtime_summaries(runtime, output_dir)

    classification_counts = classifications["classification"].value_counts().sort_index().to_dict() if not classifications.empty else {}
    summary = {
        "policy": policy, "lookahead": int(lookahead), "search_radius_um": float(search_radius_um),
        "n_sessions": int(len(manifest)), "n_tracks": int(len(tracks)), "n_endpoints": int(len(endpoints)),
        "n_endpoint_candidates": int(len(endpoint_candidates)), "classification_counts": {str(k): int(v) for k, v in classification_counts.items()},
        "same_track_gap_recoveries": int((classifications["classification"] == "same_track_gap_recovered").sum()) if not classifications.empty else 0,
        "synthetic_gap_cases": int(len(synthetic)),
        "synthetic_top1_recovery": float(synthetic["top1_correct"].mean()) if not synthetic.empty else np.nan,
        "manual_ground_truth_metrics_available": bool(classifications["manual_class"].astype(str).str.strip().ne("").any()) if not classifications.empty else False,
        "manual_class_counts": {str(key): int(value) for key, value in classifications.loc[classifications["manual_class"].astype(str).str.strip().ne(""), "manual_class"].value_counts().sort_index().to_dict().items()} if not classifications.empty else {},
        "state_dependent": _state_endpoint_metrics(endpoints, state),
        "runtime_bottleneck": runtime.iloc[int(runtime["elapsed_sec"].fillna(-np.inf).argmax())][["session_pair", "elapsed_sec", "candidate_count"]].to_dict() if not runtime.empty and runtime["elapsed_sec"].notna().any() else None,
        "state_data_available": bool(not state.empty),
        "optional_inputs": {"extraction_dir": str(Path(extraction_dir).resolve()) if extraction_dir else None, "state_csv": str(Path(state_csv).resolve()) if state_csv else None},
        "canonical_matcher_modified": False,
        "canonical_matcher_outputs_written": False,
        "input_discovery": discovered,
    }
    (output_dir / "matcher_evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    run_log = {
        "evaluator_version": "daywise_endpoint_evaluator_v1", "policy": policy, "lookahead": int(lookahead),
        "search_radius_um": float(search_radius_um), "random_seed": int(random_seed), "match_dir": str(match_dir),
        "output_dir": str(output_dir), "input_files_found": discovered["required"] | discovered["optional"],
        "input_files_missing": discovered["missing_required"] + discovered["missing_optional"],
        "optional_state_used": bool(not state.empty), "optional_extraction_dir": str(extraction_dir) if extraction_dir else None,
        "python_package_versions": _package_versions(), "git_commit": _git_commit(Path(__file__).resolve().parent.parent),
        "row_counts": {"endpoint_events": len(endpoints), "endpoint_candidates": len(endpoint_candidates), "endpoint_classification": len(classifications), "synthetic_gap_benchmark": len(synthetic), "manual_review_manifest": len(review_manifest), "matching_runtime_summary": len(runtime)},
        "canonical_matcher_outputs_unchanged": True,
        "eclipse_recomputed": False,
        "png_only": True,
    }
    (output_dir / "evaluation_run_log.json").write_text(json.dumps(run_log, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    return summary
