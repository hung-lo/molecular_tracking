"""Read-only diagnostics for daywise matcher endpoints.

The matcher stores transforms from later-session coordinates (B) into the
earlier session (A).  This module inverts those transforms only for
evaluation; it never writes to a matching directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
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
    "touches_z_edge", "touches_xy_edge", "local_density", "same_track_returns",
    "same_track_return_gap", "same_track_return_session_index", "preceding_day_a",
    "preceding_label_a", "preceding_pair_gap", "preceding_score", "preceding_dice",
    "preceding_distance_um", "preceding_ambiguity", "preceding_candidate_source",
    "preceding_assignment_source", "preceding_graph_status",
    "preceding_graph_support_fraction", "preceding_graph_residual_median_um",
    "preceding_transform_method", "preceding_transform_fallback_reason",
    "geometry_qc_pass", "segmentation_qc_status", "segmentation_failure",
    "edge_heavy", "review_required", "review_reasons", "consensus_edge_fraction",
    "has_graph_only_edge",
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
    "distance_um", "area_ratio", "spatial_term", "ambiguity_term", "score", "high_rule",
    "balanced_rule", "candidate_source", "graph_rule", "graph_status", "graph_support_count",
    "graph_support_fraction", "graph_residual_median_um", "graph_residual_mean_um",
    "graph_residual_p90_um", "graph_inlier_fraction", "graph_score", "is_graph_anchor",
    "assignment_source", "target_rank_by_distance", "nearest_second_nearest_margin_um",
    "transform_source", "transform_reliable", "transform_component_methods",
    "transform_component_fallback_reasons", "transform_component_residual_median_um_max",
    "transform_component_residual_p95_um_max", "direct_vs_composed_projection_delta_um",
]

CLASSIFICATION_COLUMNS = [
    "endpoint_id", "track_uid", "end_session_index", "end_session_id", "end_label",
    "classification", "candidate_count", "stitch_candidate_count", "same_track_returns", "manual_class",
    "manual_target_track_uid", "manual_target_session_index", "manual_target_label",
    "manual_confidence", "reviewer_notes",
]

SYNTHETIC_COLUMNS = [
    "case_id", "track_uid", "source_session_index", "target_session_index",
    "true_target_in_search_radius", "true_target_rank", "top1_correct", "candidate_count",
    "nearest_second_margin_um", "projected_distance_um", "session_gap", "elapsed_day_gap",
    "source_volume", "local_density", "edge_status", "session_pair", "benchmark_status",
]

RUNTIME_COLUMNS = [
    "session_pair", "day_a", "day_b", "pair_gap", "elapsed_sec", "n_a", "n_b",
    "candidate_count", "transform_method", "transform_fallback_reason", "n_graph",
    "n_graph_anchors", "n_graph_changed", "graph_stage_seconds",
]


@dataclass(frozen=True)
class SyntheticTrustConfig:
    """Conservative, configurable silver-standard filters for pseudo-gap tests."""

    min_score: float = 0.35
    min_dice: float = 0.10
    max_distance_um: float = 5.0
    max_ambiguity: float = 0.85
    require_consensus: bool = True
    require_no_cycle_conflict: bool = True
    require_no_transform_fallback: bool = True
    require_interior: bool = True


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
    centroid_a: np.ndarray | list[float],
    transform_b_to_a: Mapping[str, Any] | pd.Series | object,
) -> np.ndarray:
    """Project an earlier-session A centroid forward into later-session B.

    Canonical matcher transforms are stored B→A, so endpoint projection must
    explicitly invert that transform.  Keeping this operation named and tested
    avoids accidentally applying the stored transform in the wrong direction.
    """

    return apply_transform_b_to_a(centroid_a, invert_restricted_transform(transform_b_to_a))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
        "pairwise_candidates.csv", "pairwise_matches_high.csv", "pairwise_matches_balanced.csv",
        "pairwise_matches_graph.csv", f"track_edges_{policy}.csv", f"pairwise_summary_{policy}.csv",
        "pairwise_summary.csv", f"cycle_consistency_{policy}.csv",
        f"cycle_edge_checks_{policy}.csv", "graph_match_changes.csv",
        "pairwise_matches_graph_agreement.csv", "accepted_graph_edges_with_agreement.csv",
        "tracks_graph_agreement.csv", "graph_affine_agreement_track_summary.csv", "run_log.json",
    ]
    optional_names = list(dict.fromkeys(optional_names))
    paths = {name: root / name for name in required_names + optional_names}
    return {
        "match_dir": str(root),
        "policy": policy,
        "required": {name: str(paths[name]) for name in required_names if paths[name].is_file()},
        "missing_required": [name for name in required_names if not paths[name].is_file()],
        "optional": {name: str(paths[name]) for name in optional_names if paths[name].is_file()},
        "missing_optional": [name for name in optional_names if not paths[name].is_file()],
    }


def load_matcher_spacing(match_dir: str | Path) -> tuple[VoxelSpacing, str]:
    """Load exact matcher spacing from provenance, falling back to repo defaults."""

    payload = _read_json(Path(match_dir) / "run_log.json")
    spacing_payload = payload.get("spacing", {}) if isinstance(payload, dict) else {}
    if isinstance(spacing_payload, Mapping):
        try:
            spacing = VoxelSpacing(
                z_um=float(spacing_payload["z_um"]),
                y_um=float(spacing_payload["y_um"]),
                x_um=float(spacing_payload["x_um"]),
            )
        except (KeyError, TypeError, ValueError):
            pass
        else:
            return spacing, "run_log.json"
    return VoxelSpacing(), "repository_default"


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
    # Cache each wide track row once; the old track×session iterrows loop
    # repeatedly materialized the full row and made large endpoint batches
    # needlessly expensive.
    track_rows = {index: row for index, row in tracks.iterrows()}
    for row in sessions.itertuples():
        session_id = str(row.session_id)
        column = _roi_column(session_id)
        if column not in tracks:
            continue
        for index, value in tracks[column].items():
            if _present(value):
                output[(session_id, int(value))] = track_rows[index]
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
    if "eclipse_core_state" not in subset.columns and "eclipse_state_bin" in subset.columns:
        endpoint["eclipse_core_state"] = subset.iloc[0]["eclipse_state_bin"]


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
            # The last acquired session is right-censored: there is no t+1 on
            # which to define next-session loss, so it is not an endpoint event.
            if session_position + 1 >= len(sessions):
                continue
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
                "track_match_source": _text(track.get("track_match_source", track.get("match_policy", policy)), policy),
                "cycle_agreement_fraction": _float(track.get("cycle_agreement_fraction")),
                "has_cycle_conflict": _bool(track.get("has_cycle_conflict")),
                "cycle_unchecked": _bool(track.get("cycle_unchecked"), True),
                "eclipse_z": np.nan, "eclipse_core_state": "", "green": np.nan, "red": np.nan,
                "centroid_z_um": np.nan, "centroid_y_um": np.nan, "centroid_x_um": np.nan,
                "volume_um3": np.nan, "touches_z_edge": False, "touches_xy_edge": False,
                "local_density": np.nan,
                "preceding_day_a": "", "preceding_label_a": np.nan, "preceding_pair_gap": np.nan,
                "preceding_score": np.nan, "preceding_dice": np.nan,
                "preceding_distance_um": np.nan, "preceding_ambiguity": np.nan,
                "preceding_candidate_source": "", "preceding_assignment_source": "",
                "preceding_graph_status": "", "preceding_graph_support_fraction": np.nan,
                "preceding_graph_residual_median_um": np.nan,
                "preceding_transform_method": "", "preceding_transform_fallback_reason": "",
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


def add_endpoint_local_density(
    endpoints: pd.DataFrame,
    features: pd.DataFrame,
    *,
    spacing: VoxelSpacing,
    radius_um: float,
) -> pd.DataFrame:
    """Add a same-session neighborhood density around each endpoint centroid."""

    if endpoints.empty:
        return endpoints.copy()
    output = endpoints.copy()
    densities: list[float] = []
    for _, endpoint in output.iterrows():
        session_id = _text(endpoint.get("end_session_id"))
        session_features = _target_feature_table(features, session_id)
        if session_features.empty:
            densities.append(np.nan)
            continue
        coords = session_features[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
        physical = coords * spacing.as_zyx_array()
        source = np.array([
            _float(endpoint.get("centroid_z_um")),
            _float(endpoint.get("centroid_y_um")),
            _float(endpoint.get("centroid_x_um")),
        ])
        if not np.isfinite(source).all():
            feature = _feature_row(features, session_id, int(endpoint["end_label"]))
            if feature is None:
                densities.append(np.nan)
                continue
            source = feature[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float) * spacing.as_zyx_array()
        tree = cKDTree(physical)
        count = max(0, len(tree.query_ball_point(source, r=float(radius_um))) - 1)
        volume = (4.0 / 3.0) * np.pi * float(radius_um) ** 3
        densities.append(float(count / volume) if volume > 0 else np.nan)
    output["local_density"] = densities
    return output


def enrich_endpoint_preceding_evidence(
    endpoints: pd.DataFrame,
    track_edges: pd.DataFrame,
    policy_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the accepted incoming edge and graph evidence to each endpoint."""

    if endpoints.empty:
        return endpoints.copy()
    output = endpoints.copy()
    if track_edges.empty or not {"day_b", "label_b"}.issubset(track_edges.columns):
        return output
    edges = track_edges.copy()
    if "accepted_for_track" in edges.columns:
        edges = edges.loc[edges["accepted_for_track"].map(_bool)]
    match_lookup: dict[tuple[str, str, int, int], pd.Series] = {}
    if not policy_matches.empty and {"day_a", "day_b", "label_a", "label_b"}.issubset(policy_matches.columns):
        for _, match in policy_matches.iterrows():
            key = (_text(match.get("day_a")), _text(match.get("day_b")), int(match["label_a"]), int(match["label_b"]))
            match_lookup[key] = match
    fields = {
        "score": "preceding_score", "dice": "preceding_dice", "distance_um": "preceding_distance_um",
        "ambiguity": "preceding_ambiguity", "candidate_source": "preceding_candidate_source",
        "assignment_source": "preceding_assignment_source", "graph_status": "preceding_graph_status",
        "graph_support_fraction": "preceding_graph_support_fraction",
        "graph_residual_median_um": "preceding_graph_residual_median_um",
        "transform_method": "preceding_transform_method",
        "transform_fallback_reason": "preceding_transform_fallback_reason",
    }
    for index, endpoint in output.iterrows():
        subset = edges.loc[
            edges["day_b"].astype(str).eq(_text(endpoint.get("end_session_id")))
            & pd.to_numeric(edges["label_b"], errors="coerce").eq(int(endpoint["end_label"]))
        ].copy()
        if subset.empty:
            continue
        if "pair_gap" in subset.columns:
            subset = subset.sort_values(["pair_gap", "day_a"], ascending=[True, False])
        edge = subset.iloc[0]
        output.at[index, "preceding_day_a"] = _text(edge.get("day_a"))
        output.at[index, "preceding_label_a"] = _int(edge.get("label_a"))
        output.at[index, "preceding_pair_gap"] = _int(edge.get("pair_gap"))
        key = (_text(edge.get("day_a")), _text(edge.get("day_b")), _int(edge.get("label_a"), -1) or -1, _int(edge.get("label_b"), -1) or -1)
        match = match_lookup.get(key)
        for source_field, target_field in fields.items():
            value = _value(match, source_field, None) if match is not None else None
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = edge.get(source_field, np.nan if source_field not in {"candidate_source", "assignment_source", "graph_status", "transform_method", "transform_fallback_reason"} else "")
            output.at[index, target_field] = value
    return output


def _transform_row_map(transforms: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    if transforms.empty:
        return {}
    required = {"day_a", "day_b"}
    if not required.issubset(transforms.columns):
        raise ValueError("pairwise_transforms.csv must contain day_a and day_b columns")
    return {(str(row.day_a), str(row.day_b)): row for _, row in transforms.iterrows()}


def _component_transform_qc(components: list[pd.Series]) -> dict[str, Any]:
    methods = [_text(component.get("method")) for component in components]
    fallbacks = [_text(component.get("fallback_reason")) for component in components if _text(component.get("fallback_reason")).strip()]
    median_values = [_float(component.get("residual_median_um")) for component in components]
    p95_values = [_float(component.get("residual_p95_um")) for component in components]
    finite_median = [value for value in median_values if np.isfinite(value)]
    finite_p95 = [value for value in p95_values if np.isfinite(value)]
    return {
        "methods": ";".join(methods),
        "fallback_reasons": ";".join(fallbacks),
        "residual_median_um_max": max(finite_median) if finite_median else np.nan,
        "residual_p95_um_max": max(finite_p95) if finite_p95 else np.nan,
        "reliable": not fallbacks,
    }


def _forward_transform(
    source_position: int,
    target_position: int,
    sessions: pd.DataFrame,
    transform_map: dict[tuple[str, str], pd.Series],
) -> tuple[RestrictedTransform, str, bool, dict[str, Any]] | None:
    gap = target_position - source_position
    if gap < 1:
        return None
    source_id = str(sessions.iloc[source_position]["session_id"])
    target_id = str(sessions.iloc[target_position]["session_id"])
    if gap <= 2:
        stored = transform_map.get((source_id, target_id))
        if stored is not None:
            try:
                inverse = invert_restricted_transform(stored)
            except ValueError:
                return None
            qc = _component_transform_qc([stored])
            return inverse, "direct_stored", bool(qc["reliable"]), qc
        if gap == 1:
            return None
        components: list[pd.Series] = []
        for position in range(source_position, target_position):
            pair = (str(sessions.iloc[position]["session_id"]), str(sessions.iloc[position + 1]["session_id"]))
            component = transform_map.get(pair)
            if component is None:
                return None
            components.append(component)
        composed = compose_transforms(components[0], components[1])
        try:
            inverse = invert_restricted_transform(composed)
        except ValueError:
            return None
        qc = _component_transform_qc(components)
        return inverse, "composed_adjacent_fallback", bool(qc["reliable"]), qc
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
    qc = _component_transform_qc(components)
    return inverse, "composed_adjacent", bool(qc["reliable"]), qc


def _evidence_maps(
    candidates: pd.DataFrame,
    high: pd.DataFrame,
    balanced: pd.DataFrame,
    graph: pd.DataFrame,
) -> tuple[dict[tuple[str, str, int, int], pd.Series], set[tuple[str, str, int, int]], set[tuple[str, str, int, int]], set[tuple[str, str, int, int]]]:
    def row_key(row: pd.Series) -> tuple[str, str, int, int]:
        return (_text(row.get("day_a")), _text(row.get("day_b")), _int(row.get("label_a"), -1) or -1, _int(row.get("label_b"), -1) or -1)

    def keys(table: pd.DataFrame) -> set[tuple[str, str, int, int]]:
        if table.empty:
            return set()
        return {row_key(row) for _, row in table.iterrows()}

    combined: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for table in (candidates, high, balanced, graph):
        if table.empty:
            continue
        for _, row in table.iterrows():
            key = row_key(row)
            target = combined.setdefault(key, {})
            for column, value in row.items():
                if pd.isna(value):
                    continue
                # Later tables are more specific: accepted graph rows carry
                # graph/refined evidence that baseline candidate rows do not.
                target[column] = value
    candidate_map = {key: pd.Series(values) for key, values in combined.items()}
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


def _build_target_indices(
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    spacing: VoxelSpacing,
) -> dict[str, tuple[pd.DataFrame, np.ndarray, cKDTree | None]]:
    """Build each future-session centroid table/KD-tree once per evaluator run."""

    output: dict[str, tuple[pd.DataFrame, np.ndarray, cKDTree | None]] = {}
    scale = spacing.as_zyx_array()
    for session_id in sessions["session_id"].astype(str):
        table = _target_feature_table(features, session_id)
        if table.empty:
            output[session_id] = (table, np.empty((0, 3), dtype=float), None)
            continue
        physical = table[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float) * scale
        output[session_id] = (table, physical, cKDTree(physical))
    return output


def _direct_vs_composed_delta_um(
    source_position: int,
    target_position: int,
    source_coords: np.ndarray,
    sessions: pd.DataFrame,
    transform_map: dict[tuple[str, str], pd.Series],
    spacing: VoxelSpacing,
) -> float:
    """Compare direct t→t+2 projection against adjacent composition when possible."""

    if target_position - source_position != 2:
        return np.nan
    source_id = str(sessions.iloc[source_position]["session_id"])
    target_id = str(sessions.iloc[target_position]["session_id"])
    direct = transform_map.get((source_id, target_id))
    first = transform_map.get((source_id, str(sessions.iloc[source_position + 1]["session_id"])))
    second = transform_map.get((str(sessions.iloc[source_position + 1]["session_id"]), target_id))
    if direct is None or first is None or second is None:
        return np.nan
    try:
        direct_prediction = project_centroid_forward(source_coords, direct)
        composed = compose_transforms(first, second)
        composed_prediction = apply_transform_b_to_a(source_coords, invert_restricted_transform(composed))
    except ValueError:
        return np.nan
    return float(np.linalg.norm((direct_prediction - composed_prediction) * spacing.as_zyx_array()))


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
    target_indices: dict[str, tuple[pd.DataFrame, np.ndarray, cKDTree | None]],
    spacing: VoxelSpacing,
    search_radius_um: float,
    mask_shape_cache: dict[str, tuple[int, ...] | None] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    projected = _forward_transform(source_position, target_position, sessions, transform_map)
    if projected is None:
        source_id = str(sessions.iloc[source_position]["session_id"])
        target_id = str(sessions.iloc[target_position]["session_id"])
        if (source_id, target_id) in transform_map or target_position - source_position == 3:
            return [], "transform_unreliable"
        return [], "insufficient_input"
    forward, transform_source, reliable, transform_qc = projected
    source_coords = np.array([
        _float(source_feature.get("centroid_z")), _float(source_feature.get("centroid_y")), _float(source_feature.get("centroid_x")),
    ])
    if not np.isfinite(source_coords).all():
        return [], "insufficient_input"
    predicted = apply_transform_b_to_a(source_coords, forward)
    target_id = str(sessions.iloc[target_position]["session_id"])
    out_of_fov = False
    shape: tuple[int, ...] | None = None
    if mask_shape_cache is not None and target_id in mask_shape_cache:
        shape = mask_shape_cache[target_id]
    else:
        mask_path = _value(sessions.iloc[target_position], "mask_path", "")
        mask_path_text = "" if mask_path is None or pd.isna(mask_path) else str(mask_path)
        if mask_path_text and Path(mask_path_text).is_file():
            try:
                shape = tuple(int(value) for value in tifffile.memmap(Path(mask_path_text)).shape)
            except Exception:
                shape = None
        if mask_shape_cache is not None:
            mask_shape_cache[target_id] = shape
    if shape is not None:
        out_of_fov = any(value < 0 or value >= limit for value, limit in zip(predicted, shape, strict=True))
    target_features, target_physical, tree = target_indices.get(
        target_id,
        (pd.DataFrame(), np.empty((0, 3), dtype=float), None),
    )
    if target_features.empty:
        return [], "no_mask_near_prediction"
    predicted_physical = predicted * spacing.as_zyx_array()
    if tree is None:
        return [], "no_mask_near_prediction"
    nearby = [int(index) for index in tree.query_ball_point(predicted_physical, r=float(search_radius_um))]
    distances = np.linalg.norm(target_physical - predicted_physical, axis=1)
    nearby.sort(key=lambda index: (float(distances[index]), int(target_features.iloc[index]["label"])))
    nearest_distances, _ = tree.query(predicted_physical, k=min(2, len(target_features)))
    nearest_distances = np.atleast_1d(np.asarray(nearest_distances, dtype=float))
    second_margin = float(nearest_distances[1] - nearest_distances[0]) if len(nearest_distances) > 1 else np.nan
    direct_vs_composed_delta = _direct_vs_composed_delta_um(
        source_position, target_position, source_coords, sessions, transform_map, spacing
    )
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
            "target_track_match_source": _text(owner.get("track_match_source", owner.get("match_policy", ""))) if owner is not None else "",
            "existing_candidate_found": evidence is not None,
            "existing_high_match": key in high_keys,
            "existing_balanced_match": key in balanced_keys,
            "existing_graph_match": key in graph_keys,
            "dice": _float(evidence.get("dice")) if evidence is not None else np.nan,
            "iou": _float(evidence.get("iou")) if evidence is not None else np.nan,
            "ambiguity": _float(evidence.get("ambiguity")) if evidence is not None else np.nan,
            "base_score": _float(evidence.get("base_score", evidence.get("score"))) if evidence is not None else np.nan,
            "refined_score": _float(evidence.get("refined_score", evidence.get("score"))) if evidence is not None else np.nan,
            "distance_um": _float(evidence.get("distance_um")) if evidence is not None else np.nan,
            "area_ratio": _float(evidence.get("area_ratio")) if evidence is not None else np.nan,
            "spatial_term": _float(evidence.get("spatial_term")) if evidence is not None else np.nan,
            "ambiguity_term": _float(evidence.get("ambiguity_term")) if evidence is not None else np.nan,
            "score": _float(evidence.get("score")) if evidence is not None else np.nan,
            "high_rule": _bool(evidence.get("high_rule")) if evidence is not None else False,
            "balanced_rule": _bool(evidence.get("balanced_rule")) if evidence is not None else False,
            "candidate_source": _text(evidence.get("candidate_source")) if evidence is not None else "evaluator_kdtree",
            "graph_rule": _bool(evidence.get("graph_rule")) if evidence is not None else False,
            "graph_status": _text(evidence.get("graph_status")) if evidence is not None else "",
            "graph_support_count": _float(evidence.get("graph_support_count")) if evidence is not None else np.nan,
            "graph_support_fraction": _float(evidence.get("graph_support_fraction")) if evidence is not None else np.nan,
            "graph_residual_median_um": _float(evidence.get("graph_residual_median_um")) if evidence is not None else np.nan,
            "graph_residual_mean_um": _float(evidence.get("graph_residual_mean_um")) if evidence is not None else np.nan,
            "graph_residual_p90_um": _float(evidence.get("graph_residual_p90_um")) if evidence is not None else np.nan,
            "graph_inlier_fraction": _float(evidence.get("graph_inlier_fraction")) if evidence is not None else np.nan,
            "graph_score": _float(evidence.get("graph_score")) if evidence is not None else np.nan,
            "is_graph_anchor": _bool(evidence.get("is_graph_anchor")) if evidence is not None else False,
            "assignment_source": _text(evidence.get("assignment_source")) if evidence is not None else "",
            "target_rank_by_distance": int(rank),
            "nearest_second_nearest_margin_um": second_margin,
            "transform_source": transform_source,
            "transform_reliable": bool(reliable),
            "transform_component_methods": _text(transform_qc.get("methods")),
            "transform_component_fallback_reasons": _text(transform_qc.get("fallback_reasons")),
            "transform_component_residual_median_um_max": _float(transform_qc.get("residual_median_um_max")),
            "transform_component_residual_p95_um_max": _float(transform_qc.get("residual_p95_um_max")),
            "direct_vs_composed_projection_delta_um": direct_vs_composed_delta,
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
    target_indices = _build_target_indices(features, sessions, spacing)
    rows: list[dict[str, Any]] = []
    status: dict[str, list[str]] = {}
    mask_shape_cache: dict[str, tuple[int, ...] | None] = {}
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
                target_indices=target_indices, spacing=spacing, search_radius_um=search_radius_um,
                mask_shape_cache=mask_shape_cache,
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
        reliable_subset = subset.loc[subset["transform_reliable"].map(_bool)] if not subset.empty and "transform_reliable" in subset.columns else subset
        stitch_subset = reliable_subset.loc[reliable_subset["target_track_starts_here"].map(_bool)] if not reliable_subset.empty and "target_track_starts_here" in reliable_subset.columns else reliable_subset
        if _bool(endpoint.get("same_track_returns")):
            classification = "same_track_gap_recovered"
        elif _bool(endpoint.get("touches_z_edge")) or _bool(endpoint.get("touches_xy_edge")) or (statuses and all(value == "edge_or_out_of_fov" for value in statuses)):
            classification = "edge_or_out_of_fov"
        elif not stitch_subset.empty:
            unique_targets = stitch_subset.drop_duplicates(["target_session_index", "target_label"])
            if len(unique_targets) > 1:
                classification = "multiple_nearby_candidates"
            elif _bool(unique_targets.iloc[0].get("target_is_singleton")):
                classification = "nearby_singleton_candidate"
            else:
                classification = "nearby_new_track_candidate"
        elif not reliable_subset.empty:
            # Nearby masks already owned by tracks that began earlier are not
            # stitch targets.  They still make the local field ambiguous.
            classification = "multiple_nearby_candidates"
        elif any(value == "transform_unreliable" for value in statuses) or (not subset.empty and not subset["transform_reliable"].all()):
            classification = "transform_unreliable"
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
            "candidate_count": int(len(subset)), "stitch_candidate_count": int(len(stitch_subset)),
            "same_track_returns": _bool(endpoint.get("same_track_returns")),
            "manual_class": "", "manual_target_track_uid": "", "manual_target_session_index": "",
            "manual_target_label": "", "manual_confidence": "", "reviewer_notes": "",
        })
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


def _synthetic_trusted(track: pd.Series, gap: int, config: SyntheticTrustConfig) -> bool:
    if _int(track.get("n_days_present"), 0) < gap + 1:
        return False
    if config.require_no_cycle_conflict and _bool(track.get("has_cycle_conflict")):
        return False
    if config.require_no_transform_fallback and _bool(track.get("contains_transform_fallback_edge")):
        return False
    if config.require_consensus and "track_match_source" in track.index:
        source = _text(track.get("track_match_source")).strip().lower()
        if source and source != "consensus":
            return False
    for field, threshold, direction in (
        ("min_score", config.min_score, "min"),
        ("min_dice", config.min_dice, "min"),
        ("max_distance_um", config.max_distance_um, "max"),
        ("max_ambiguity", config.max_ambiguity, "max"),
    ):
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
    trust_config: SyntheticTrustConfig | None = None,
) -> pd.DataFrame:
    """Benchmark one- and two-missing-session projections on trusted tracks."""

    del endpoints  # Endpoint observations are not ground truth for this benchmark.
    spacing = spacing or VoxelSpacing()
    trust_config = trust_config or SyntheticTrustConfig()
    tracks = _prepare_tracks(tracks, sessions, policy)
    transform_map = _transform_row_map(transforms)
    owner_map = _owner_map(tracks, sessions)
    target_indices = _build_target_indices(features, sessions, spacing)
    feature_lookup: dict[tuple[str, int], pd.Series] = {
        (str(row["session_id"]), int(row["label"])): row
        for _, row in features.iterrows()
    }
    target_labels: dict[str, np.ndarray] = {
        session_id: table["label"].to_numpy(dtype=np.int64, copy=True)
        for session_id, (table, _, _) in target_indices.items()
        if not table.empty
    }
    mask_shape_cache: dict[str, tuple[int, ...] | None] = {}
    rows: list[dict[str, Any]] = []
    for track_index, (_, track) in enumerate(tracks.iterrows()):
        for gap in (2, 3):
            if not _synthetic_trusted(track, gap, trust_config):
                continue
            for source_position in range(0, len(sessions) - gap):
                source_id = str(sessions.iloc[source_position]["session_id"])
                target_id = str(sessions.iloc[source_position + gap]["session_id"])
                # Silver-standard pseudo-gaps must begin as genuinely
                # continuous identities; otherwise we would benchmark on a
                # real missing observation rather than an artificial gap.
                interval_values = [
                    track.get(_roi_column(str(sessions.iloc[position]["session_id"])), pd.NA)
                    for position in range(source_position, source_position + gap + 1)
                ]
                if not all(_present(value) for value in interval_values):
                    continue
                source_value = track.get(_roi_column(source_id), pd.NA)
                target_value = track.get(_roi_column(target_id), pd.NA)
                if not (_present(source_value) and _present(target_value)):
                    continue
                source_feature = feature_lookup.get((source_id, int(source_value)))
                target_feature = feature_lookup.get((target_id, int(target_value)))
                if source_feature is None or target_feature is None:
                    continue
                source_edge = _bool(source_feature.get("touches_z_edge")) or _bool(source_feature.get("touches_xy_edge"))
                target_edge = _bool(target_feature.get("touches_z_edge")) or _bool(target_feature.get("touches_xy_edge"))
                if trust_config.require_interior and (source_edge or target_edge):
                    continue
                pair_rows, pair_status = _search_candidates_for_pair(
                    endpoint_id=f"synthetic_{track_index}_{source_position}_{gap}",
                    source_track_uid=_text(track.get("track_uid")), source_position=source_position,
                    source_label=int(source_value), target_position=source_position + gap,
                    source_feature=source_feature, tracks=tracks, features=features, sessions=sessions,
                    transform_map=transform_map, candidate_map={}, high_keys=set(), balanced_keys=set(), graph_keys=set(),
                    owner_map=owner_map, target_indices=target_indices, spacing=spacing, search_radius_um=search_radius_um,
                    mask_shape_cache=mask_shape_cache,
                )
                true_label = int(target_value)
                ranked = sorted(pair_rows, key=lambda row: (float(row["projected_distance_um"]), int(row["target_label"])))
                true = next((row for row in ranked if int(row["target_label"]) == true_label), None)
                projected_distance = float(true["projected_distance_um"]) if true is not None else np.nan
                in_radius = true is not None
                true_rank: float | int = np.nan
                nearest_second_margin = np.nan
                projected = _forward_transform(source_position, source_position + gap, sessions, transform_map)
                if projected is not None:
                    forward, _, _, _ = projected
                    source_coords = source_feature[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
                    prediction = apply_transform_b_to_a(source_coords, forward) * spacing.as_zyx_array()
                    target_table, target_physical, target_tree = target_indices[target_id]
                    all_distances = np.linalg.norm(target_physical - prediction, axis=1)
                    if len(all_distances):
                        labels = target_labels.get(target_id)
                        if labels is None:
                            labels = target_table["label"].to_numpy(dtype=np.int64, copy=True)
                            target_labels[target_id] = labels
                        matching_indices = np.flatnonzero(labels == true_label)
                        if len(matching_indices):
                            true_index = int(matching_indices[0])
                            true_distance = float(all_distances[true_index])
                            # Exact rank under the historical ordering
                            # (distance, label), without sorting every ROI for
                            # every pseudo-gap case. Labels are unique within a
                            # session, so ties are resolved by label value.
                            closer = int(np.count_nonzero(all_distances < true_distance))
                            tied_before = int(np.count_nonzero((all_distances == true_distance) & (labels < true_label)))
                            true_rank = closer + tied_before + 1
                            projected_distance = true_distance
                            in_radius = projected_distance <= float(search_radius_um)
                        if target_tree is not None:
                            nearest, _ = target_tree.query(prediction, k=min(2, len(target_table)))
                            nearest = np.atleast_1d(np.asarray(nearest, dtype=float))
                            if len(nearest) > 1:
                                nearest_second_margin = float(nearest[1] - nearest[0])
                rows.append({
                    "case_id": f"gap_{gap}_{track_index}_{source_position}", "track_uid": _text(track.get("track_uid")),
                    "source_session_index": int(sessions.iloc[source_position]["session_index"]),
                    "target_session_index": int(sessions.iloc[source_position + gap]["session_index"]),
                    "true_target_in_search_radius": bool(in_radius), "true_target_rank": true_rank,
                    "top1_correct": bool(in_radius and true_rank == 1), "candidate_count": len(ranked),
                    "nearest_second_margin_um": nearest_second_margin,
                    "projected_distance_um": projected_distance, "session_gap": gap,
                    "elapsed_day_gap": _date_gap(sessions, source_position, source_position + gap),
                    "source_volume": _float(source_feature.get("volume_um3")),
                    "local_density": float(len(ranked) / ((4.0 / 3.0) * np.pi * search_radius_um ** 3)),
                    "edge_status": "edge" if source_edge or target_edge else "interior",
                    "session_pair": f"{source_id}->{target_id}",
                    "benchmark_status": pair_status,
                })
    table = pd.DataFrame(rows, columns=SYNTHETIC_COLUMNS)
    if not table.empty:
        # Deterministic order; seed is retained in the run log for future
        # optional sampling without changing present benchmark semantics.
        _ = np.random.default_rng(int(random_seed))
        table = table.sort_values(["session_gap", "source_session_index", "track_uid", "case_id"]).reset_index(drop=True)
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
    """Select spatial review rows plus Low endpoints with matched Middle controls."""

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
    selected_state: list[dict[str, Any]] = []
    if state_column in work.columns and work[state_column].astype(str).str.strip().ne("").any():
        low = work.loc[work[state_column].astype(str).str.strip().str.lower().eq("low")].copy()
        middle = work.loc[work[state_column].astype(str).str.strip().str.lower().eq("middle")].copy()
        used_middle: set[str] = set()
        covariates = ["red", "volume_um3", "centroid_z_um", "centroid_y_um", "centroid_x_um", "local_density", "n_days_present"]
        for _, low_row in low.sort_values(["end_session_index", "endpoint_id"]).iterrows():
            low_item = low_row.to_dict()
            low_item.update({
                "sample_type": "state_dependent", "sampling_stratum": "low", "state_stratum": "low",
                "matched_low_endpoint_id": _text(low_row.get("endpoint_id")), "state_match_distance": 0.0,
                "state_match_covariates": ";".join(covariates),
            })
            selected_state.append(low_item)
            pool = middle.loc[
                middle["end_session_index"].astype(int).eq(int(low_row["end_session_index"]))
                & ~middle["endpoint_id"].astype(str).isin(used_middle)
            ].copy()
            if pool.empty:
                continue
            combined = pd.concat([pd.DataFrame([low_row]), pool], ignore_index=True)
            scales: dict[str, float] = {}
            for column in covariates:
                if column not in combined.columns:
                    continue
                numeric = pd.to_numeric(combined[column], errors="coerce")
                scale = float(numeric.std(ddof=0)) if numeric.notna().sum() > 1 else np.nan
                if not np.isfinite(scale) or scale <= 0:
                    median = float(numeric.median()) if numeric.notna().any() else np.nan
                    mad = float((numeric - median).abs().median()) if numeric.notna().any() else np.nan
                    scale = mad * 1.4826 if np.isfinite(mad) and mad > 0 else 1.0
                scales[column] = scale

            def match_distance(row: pd.Series) -> float:
                terms: list[float] = []
                for column, scale in scales.items():
                    left = _float(low_row.get(column)); right = _float(row.get(column))
                    if np.isfinite(left) and np.isfinite(right):
                        terms.append(((right - left) / scale) ** 2)
                return float(np.sqrt(np.mean(terms))) if terms else np.inf

            pool["_state_match_distance"] = pool.apply(match_distance, axis=1)
            control = pool.sort_values(["_state_match_distance", "endpoint_id"]).iloc[0]
            used_middle.add(_text(control.get("endpoint_id")))
            control_item = control.drop(labels=["_state_match_distance"], errors="ignore").to_dict()
            control_item.update({
                "sample_type": "state_dependent", "sampling_stratum": "middle_control",
                "state_stratum": "middle_control", "matched_low_endpoint_id": _text(low_row.get("endpoint_id")),
                "state_match_distance": float(control["_state_match_distance"]),
                "state_match_covariates": ";".join(covariates),
            })
            selected_state.append(control_item)

    selected_general: list[dict[str, Any]] = []
    for index in general_indices:
        row = work.loc[index].to_dict()
        row.update({
            "sample_type": "general_spatial", "sampling_stratum": "general", "state_stratum": "",
            "matched_low_endpoint_id": "", "state_match_distance": np.nan, "state_match_covariates": "",
        })
        selected_general.append(row)

    # State-dependent review is prioritized so Low endpoints are genuinely
    # oversampled; the spatial sample fills remaining capacity.
    manifest = pd.DataFrame(selected_state + selected_general).drop_duplicates("endpoint_id", keep="first")
    if max_review_panels is not None and max_review_panels >= 0 and len(manifest) > int(max_review_panels):
        state_rows = manifest.loc[manifest["sample_type"].eq("state_dependent")].copy()
        general_rows = manifest.loc[~manifest["sample_type"].eq("state_dependent")].copy()
        state_rows = state_rows.sort_values(["end_session_index", "matched_low_endpoint_id", "sampling_stratum", "endpoint_id"])
        remaining = max(0, int(max_review_panels) - len(state_rows))
        if len(state_rows) >= int(max_review_panels):
            manifest = state_rows.head(int(max_review_panels)).copy()
        else:
            general_order = rng.permutation(len(general_rows))[:remaining] if len(general_rows) else []
            manifest = pd.concat([state_rows, general_rows.iloc[general_order]], ignore_index=True)
    if manifest.empty:
        return pd.DataFrame(columns=[
            "endpoint_id", "sample_type", "sampling_stratum", "spatial_stratum", "state_stratum",
            "matched_low_endpoint_id", "state_match_distance", "state_match_covariates",
        ])
    return manifest.sort_values(["sample_type", "end_session_index", "endpoint_id"]).reset_index(drop=True)


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
        manual = frame["manual_class"].astype(str).str.strip() if not frame.empty and "manual_class" in frame.columns else pd.Series(dtype=str)
        rows.append({
            "session_index": index, "session_id": session_id, "n_endpoint_observations": int(len(frame)),
            "n_endpoints": int(len(frame)), "endpoint_rate": float(len(frame) / all_counts[index]) if all_counts[index] else np.nan,
            "same_track_gap_recovered": int((frame["classification"] == "same_track_gap_recovered").sum()) if not frame.empty else 0,
            "nearby_new_track_candidate": int((frame["classification"] == "nearby_new_track_candidate").sum()) if not frame.empty else 0,
            "nearby_singleton_candidate": int((frame["classification"] == "nearby_singleton_candidate").sum()) if not frame.empty else 0,
            "multiple_nearby_candidates": int((frame["classification"] == "multiple_nearby_candidates").sum()) if not frame.empty else 0,
            "manual_labels_available": bool(manual.ne("").any()),
            "manual_segmentation_dropout": int((manual == "segmentation_dropout").sum()),
            "manual_matcher_fragment": int((manual == "matcher_fragment").sum()),
            "manual_true_disappearance": int((manual == "true_disappearance").sum()),
            "manual_edge_or_fov_loss": int((manual == "edge_or_fov_loss").sum()),
            "manual_ambiguous": int((manual == "ambiguous").sum()),
        })
    return pd.DataFrame(rows)


def _safe_quintile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series("missing", index=series.index, dtype=object)
    valid = numeric.notna()
    if valid.sum() < 2:
        return output
    try:
        ranked = numeric.loc[valid].rank(method="first")
        bins = pd.qcut(ranked, q=min(5, int(valid.sum())), labels=False, duplicates="drop")
    except ValueError:
        return output
    output.loc[valid] = [f"Q{int(value) + 1}" for value in bins]
    return output


def build_state_dropout_stratification(
    state: pd.DataFrame,
    endpoints: pd.DataFrame,
    classifications: pd.DataFrame,
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    policy: str,
    spacing: VoxelSpacing,
    local_density_radius_um: float,
) -> pd.DataFrame:
    """Calculate Low/Middle endpoint risk overall and across practical strata."""

    if state.empty or not {"track_uid", "session_index"}.issubset(state.columns):
        return pd.DataFrame()
    state_column = "eclipse_core_state" if "eclipse_core_state" in state.columns else "eclipse_state_bin" if "eclipse_state_bin" in state.columns else None
    if state_column is None:
        return pd.DataFrame()
    base = state.copy()
    base["track_uid"] = base["track_uid"].astype(str)
    base["session_index"] = pd.to_numeric(base["session_index"], errors="coerce")
    base = base.loc[base["session_index"].notna()].copy()
    base["session_index"] = base["session_index"].astype(int)
    if sessions.empty:
        return pd.DataFrame()
    valid_indices = set(sessions["session_index"].astype(int).tolist())
    last_index = int(sessions["session_index"].max())
    base = base.loc[base["session_index"].isin(valid_indices) & base["session_index"].lt(last_index)].copy()
    base["state"] = base[state_column].astype(str).str.strip().str.lower()
    base = base.loc[base["state"].isin({"low", "middle"})].drop_duplicates(["track_uid", "session_index"])
    if base.empty:
        return pd.DataFrame()

    endpoint_lookup = {
        (_text(row.get("track_uid")), int(row["end_session_index"])): row
        for _, row in endpoints.iterrows()
    }
    classification_lookup = {
        _text(row.get("endpoint_id")): _text(row.get("classification"))
        for _, row in classifications.iterrows()
    }
    prepared_tracks = _prepare_tracks(tracks, sessions, policy)
    track_lookup = {str(row["track_uid"]): row for _, row in prepared_tracks.iterrows()}
    session_lookup = {int(row["session_index"]): row for _, row in sessions.iterrows()}
    target_indices = _build_target_indices(features, sessions, spacing)
    density_volume = (4.0 / 3.0) * np.pi * float(local_density_radius_um) ** 3

    records: list[dict[str, Any]] = []
    for _, observation in base.iterrows():
        track_uid = str(observation["track_uid"]); session_index = int(observation["session_index"])
        endpoint = endpoint_lookup.get((track_uid, session_index))
        track = track_lookup.get(track_uid)
        session = session_lookup.get(session_index)
        session_id = _text(session.get("session_id")) if session is not None else ""
        label = _int(track.get(_roi_column(session_id))) if track is not None and session_id else None
        feature = _feature_row(features, session_id, label) if label is not None else None
        local_density = np.nan
        if feature is not None and session_id in target_indices:
            table, physical, tree = target_indices[session_id]
            if tree is not None:
                source = feature[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float) * spacing.as_zyx_array()
                count = max(0, len(tree.query_ball_point(source, r=float(local_density_radius_um))) - 1)
                local_density = float(count / density_volume) if density_volume > 0 else np.nan
        records.append({
            "track_uid": track_uid, "session_index": session_index, "session_id": session_id,
            "state": str(observation["state"]), "is_endpoint": endpoint is not None,
            "classification": classification_lookup.get(_text(endpoint.get("endpoint_id")) if endpoint is not None else "", ""),
            "red": _float(observation.get("red")),
            "volume_um3": _float(feature.get("volume_um3")) if feature is not None else np.nan,
            "edge_status": "edge" if feature is not None and (_bool(feature.get("touches_z_edge")) or _bool(feature.get("touches_xy_edge"))) else "interior",
            "local_density": local_density,
            "track_source": _text(track.get("track_match_source", track.get("match_policy", policy)), policy) if track is not None else policy,
        })
    observations = pd.DataFrame(records)
    observations["red_quintile"] = _safe_quintile(observations["red"])
    observations["volume_quintile"] = _safe_quintile(observations["volume_um3"])
    observations["local_density_quintile"] = _safe_quintile(observations["local_density"])

    strata: list[tuple[str, pd.Series]] = [
        ("overall", pd.Series("all", index=observations.index)),
        ("session", observations["session_id"].astype(str)),
        ("red_quintile", observations["red_quintile"]),
        ("volume_quintile", observations["volume_quintile"]),
        ("edge_status", observations["edge_status"]),
        ("local_density_quintile", observations["local_density_quintile"]),
        ("track_source", observations["track_source"].astype(str)),
    ]
    rows: list[dict[str, Any]] = []
    for stratum_type, values in strata:
        work = observations.assign(_stratum=values.astype(str))
        for (stratum_value, state_name), group in work.groupby(["_stratum", "state"], sort=True):
            endpoint_group = group.loc[group["is_endpoint"]]
            rows.append({
                "summary_type": "state_dropout", "stratum_type": stratum_type,
                "stratum_value": str(stratum_value), "state": state_name,
                "n_observations": int(len(group)), "n_endpoints": int(group["is_endpoint"].sum()),
                "dropout_rate": float(group["is_endpoint"].mean()) if len(group) else np.nan,
                "same_track_gap_recovered": int((endpoint_group["classification"] == "same_track_gap_recovered").sum()),
                "nearby_new_track_candidate": int((endpoint_group["classification"] == "nearby_new_track_candidate").sum()),
                "nearby_singleton_candidate": int((endpoint_group["classification"] == "nearby_singleton_candidate").sum()),
                "multiple_nearby_candidates": int((endpoint_group["classification"] == "multiple_nearby_candidates").sum()),
            })
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    for (stratum_type, stratum_value), indices in output.groupby(["stratum_type", "stratum_value"]).groups.items():
        frame = output.loc[list(indices)]
        low = frame.loc[frame["state"].eq("low"), "dropout_rate"]
        middle = frame.loc[frame["state"].eq("middle"), "dropout_rate"]
        if low.empty or middle.empty:
            risk_ratio = np.nan; risk_difference = np.nan
        else:
            low_rate = float(low.iloc[0]); middle_rate = float(middle.iloc[0])
            risk_ratio = low_rate / middle_rate if middle_rate > 0 else np.nan
            risk_difference = low_rate - middle_rate
        output.loc[list(indices), "risk_ratio_low_to_middle"] = risk_ratio
        output.loc[list(indices), "risk_difference_low_minus_middle"] = risk_difference
    return output.sort_values(["stratum_type", "stratum_value", "state"]).reset_index(drop=True)


def build_runtime_summary(match_dir: str | Path, policy: str = "graph") -> pd.DataFrame:
    root = Path(match_dir)
    summary = _read_csv(root / "pairwise_summary.csv")
    graph_summary = _read_csv(root / f"pairwise_summary_{policy}.csv")
    candidates = _read_csv(root / "pairwise_candidates.csv")
    if summary.empty:
        return pd.DataFrame(columns=RUNTIME_COLUMNS)
    candidate_counts = candidates.groupby(["day_a", "day_b"]).size().to_dict() if not candidates.empty and {"day_a", "day_b"}.issubset(candidates.columns) else {}
    graph_lookup = graph_summary.set_index(["day_a", "day_b"]) if not graph_summary.empty and {"day_a", "day_b"}.issubset(graph_summary.columns) else pd.DataFrame()
    rows = []
    for _, row in summary.iterrows():
        key = (_text(row.get("day_a")), _text(row.get("day_b")))
        graph_row = graph_lookup.loc[key] if not graph_lookup.empty and key in graph_lookup.index else None
        # pairwise_summary_graph.csv currently copies the affine summary's
        # elapsed_sec; it is not graph-stage wall time.  Only use a dedicated
        # graph timing field when a future matcher version emits one.
        graph_stage_seconds = np.nan
        if graph_row is not None:
            for field in ("graph_elapsed_sec", "graph_stage_seconds", "graph_elapsed_seconds"):
                value = _float(_value(graph_row, field, np.nan))
                if np.isfinite(value):
                    graph_stage_seconds = value
                    break
        rows.append({
            "session_pair": f"{key[0]}->{key[1]}", "day_a": key[0], "day_b": key[1],
            "pair_gap": _int(row.get("pair_gap")), "elapsed_sec": _float(row.get("elapsed_sec")),
            "n_a": _int(row.get("n_a")), "n_b": _int(row.get("n_b")),
            "candidate_count": int(candidate_counts.get(key, 0)),
            "transform_method": _text(row.get("transform_method")),
            "transform_fallback_reason": _text(row.get("transform_fallback_reason")),
            "n_graph": _int(_value(graph_row, "n_graph", None)),
            "n_graph_anchors": _int(_value(graph_row, "n_graph_anchors", None)),
            "n_graph_changed": _int(_value(graph_row, "n_graph_changed", None)),
            "graph_stage_seconds": graph_stage_seconds,
        })
    return pd.DataFrame(rows, columns=RUNTIME_COLUMNS).sort_values(["pair_gap", "day_a", "day_b"]).reset_index(drop=True)


def load_master_stage_timings(match_dir: str | Path) -> dict[str, float]:
    """Load master-pipeline stage timings when the matching directory has them."""

    root = Path(match_dir).resolve()
    candidates = [root / "run_manifest.json", root.parent / "run_manifest.json", root.parent.parent / "run_manifest.json"]
    for path in candidates:
        payload = _read_json(path)
        timings = payload.get("stage_durations_seconds", {}) if payload else {}
        if isinstance(timings, Mapping):
            output: dict[str, float] = {}
            for key, value in timings.items():
                number = _float(value)
                if np.isfinite(number):
                    output[str(key)] = float(number)
            if output:
                return output
    return {}


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
    for column, default in (
        ("geometry_qc_pass", np.nan), ("segmentation_qc_status", ""),
        ("segmentation_failure", np.nan), ("edge_heavy", np.nan),
        ("review_required", np.nan), ("review_reasons", ""),
        ("consensus_edge_fraction", np.nan), ("has_graph_only_edge", np.nan),
    ):
        if column not in output.columns:
            output[column] = default

    def assign_enrichment_value(index: object, column: str, value: object) -> None:
        nonlocal output
        if pd.isna(value):
            return
        if column not in output.columns:
            output[column] = pd.Series([pd.NA] * len(output), index=output.index, dtype=object)
        dtype = output[column].dtype
        if pd.api.types.is_numeric_dtype(dtype) and isinstance(
            value, (str, np.str_, bool, np.bool_)
        ):
            # Fixed-schema endpoint tables can infer all-missing optional QC
            # columns as float64. Real extraction tables may later provide
            # strings (for example ``not_configured``) or booleans. Pandas 3
            # rejects those assignments instead of silently widening the dtype,
            # so widen only the affected evaluator column before assignment.
            output[column] = output[column].astype(object)
        output.at[index, column] = value

    def fill_session_table(name: str, columns: tuple[str, ...]) -> None:
        nonlocal output
        table = _read_csv(root / name)
        if table.empty or not {"track_uid", "session_index"}.issubset(table.columns):
            return
        source = table.copy()
        source["track_uid"] = source["track_uid"].astype(str)
        source["session_index"] = pd.to_numeric(source["session_index"], errors="coerce")
        source = source.loc[source["session_index"].notna()].copy()
        source["session_index"] = source["session_index"].astype(int)
        source = source.drop_duplicates(["track_uid", "session_index"])
        lookup = {(str(row["track_uid"]), int(row["session_index"])): row for _, row in source.iterrows()}
        for index, endpoint in output.iterrows():
            row = lookup.get((_text(endpoint.get("track_uid")), int(endpoint["end_session_index"])))
            if row is None:
                continue
            for column in columns:
                if column not in row.index:
                    continue
                current = output.at[index, column] if column in output.columns else np.nan
                missing = pd.isna(current) or (isinstance(current, str) and not current.strip())
                if missing:
                    assign_enrichment_value(index, column, row[column])

    def fill_track_table(name: str, columns: tuple[str, ...]) -> None:
        nonlocal output
        table = _read_csv(root / name)
        if table.empty or "track_uid" not in table.columns:
            return
        source = table.copy(); source["track_uid"] = source["track_uid"].astype(str)
        source = source.drop_duplicates("track_uid")
        lookup = {str(row["track_uid"]): row for _, row in source.iterrows()}
        for index, endpoint in output.iterrows():
            row = lookup.get(_text(endpoint.get("track_uid")))
            if row is None:
                continue
            for column in columns:
                if column not in row.index:
                    continue
                current = output.at[index, column] if column in output.columns else np.nan
                missing = pd.isna(current) or (isinstance(current, str) and not current.strip())
                if missing:
                    assign_enrichment_value(index, column, row[column])

    fill_session_table(
        "matched_roi_geometry_qc_long.csv",
        ("geometry_qc_pass", "segmentation_qc_status", "green", "red"),
    )
    fill_session_table(
        "matched_roi_log_ratio_metrics_all_observed.csv",
        ("green", "red", "eclipse_z", "eclipse_core_state"),
    )
    fill_session_table(
        "matched_roi_trajectory_observations_eligible.csv",
        ("green", "red", "eclipse_z", "eclipse_core_state", "geometry_qc_pass"),
    )
    fill_track_table(
        "matched_track_qc_summary.csv",
        ("segmentation_failure", "edge_heavy", "review_required", "review_reasons", "track_match_source", "consensus_edge_fraction", "has_graph_only_edge"),
    )
    fill_track_table(
        "graph_affine_agreement_track_metadata.csv",
        ("track_match_source", "consensus_edge_fraction", "has_graph_only_edge"),
    )
    # Be permissive with optional enrichment tables that happen to carry
    # observation-level keys/metrics, while retaining track-level handling
    # for the repository's canonical schemas.
    fill_session_table(
        "matched_track_qc_summary.csv",
        ("green", "red", "eclipse_z", "eclipse_core_state"),
    )
    fill_session_table(
        "graph_affine_agreement_track_metadata.csv",
        ("green", "red", "eclipse_z", "eclipse_core_state"),
    )
    return output


def _state_endpoint_metrics(endpoints: pd.DataFrame, state: pd.DataFrame, sessions: pd.DataFrame) -> dict[str, Any]:
    if state.empty or endpoints.empty or "eclipse_core_state" not in endpoints.columns:
        return {"available": False}
    states = endpoints["eclipse_core_state"].astype(str).str.strip().str.lower()
    if not states.isin({"low", "middle"}).any():
        return {"available": False}
    state_column = "eclipse_core_state" if "eclipse_core_state" in state.columns else "eclipse_state_bin" if "eclipse_state_bin" in state.columns else None
    if state_column is None:
        return {"available": False}
    eligible_state = state.copy()
    eligible_state["session_index"] = pd.to_numeric(eligible_state["session_index"], errors="coerce")
    if not sessions.empty:
        last_index = int(sessions["session_index"].max())
        eligible_state = eligible_state.loc[eligible_state["session_index"].notna() & eligible_state["session_index"].astype(int).lt(last_index)]
    eligible_state = eligible_state.drop_duplicates(["track_uid", "session_index"])
    denominator_states = eligible_state[state_column].astype(str).str.strip().str.lower().value_counts()
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


def _prepare_output_directory(output_dir: Path, *, overwrite: bool) -> pd.DataFrame:
    """Prepare evaluator outputs while preserving any prior manual labels."""

    previous = _read_csv(output_dir / "endpoint_classification.csv") if output_dir.exists() else pd.DataFrame()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        generated_files = {
            "endpoint_events.csv", "endpoint_candidates.csv", "endpoint_classification.csv",
            "synthetic_gap_benchmark.csv", "manual_review_manifest.csv", "dropout_summary.csv",
            "matching_runtime_summary.csv", "matcher_evaluation_summary.json", "evaluation_run_log.json",
            "matching_runtime_by_pair.png", "matching_runtime_vs_candidate_count.png", "matching_runtime_by_gap.png",
        }
        for name in generated_files:
            path = output_dir / name
            if path.is_file():
                path.unlink()
        review_dir = output_dir / "review_panels"
        if review_dir.exists():
            shutil.rmtree(review_dir)
    return previous


def _restore_manual_columns(classifications: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    if classifications.empty or previous.empty or "endpoint_id" not in previous.columns:
        return classifications
    output = classifications.copy()
    manual_columns = [column for column in CLASSIFICATION_COLUMNS if column.startswith("manual_") or column == "reviewer_notes"]
    old = previous.copy(); old["endpoint_id"] = old["endpoint_id"].astype(str)
    old = old.drop_duplicates("endpoint_id").set_index("endpoint_id")
    for index, endpoint_id in output["endpoint_id"].astype(str).items():
        if endpoint_id not in old.index:
            continue
        for column in manual_columns:
            if column not in old.columns:
                continue
            value = old.at[endpoint_id, column]
            if pd.notna(value) and str(value).strip():
                output.at[index, column] = value
    return output


def _tracking_metrics(
    tracks: pd.DataFrame,
    sessions: pd.DataFrame,
    endpoints: pd.DataFrame,
    classifications: pd.DataFrame,
) -> dict[str, Any]:
    evaluable = 0
    for _, track in tracks.iterrows():
        for position in range(max(0, len(sessions) - 1)):
            session_id = str(sessions.iloc[position]["session_id"])
            if _present(track.get(_roi_column(session_id), pd.NA)):
                evaluable += 1
    n_endpoints = int(len(endpoints))
    same_gap = int(endpoints["same_track_returns"].map(_bool).sum()) if not endpoints.empty else 0
    class_counts = classifications["classification"].value_counts().to_dict() if not classifications.empty else {}
    return {
        "n_evaluable_track_observations": int(evaluable),
        "n_next_session_losses": n_endpoints,
        "next_session_same_track_retention": float((evaluable - n_endpoints) / evaluable) if evaluable else np.nan,
        "next_session_loss_rate": float(n_endpoints / evaluable) if evaluable else np.nan,
        "existing_gap_recovery_count": same_gap,
        "existing_gap_recovery_rate_among_losses": float(same_gap / n_endpoints) if n_endpoints else np.nan,
        "nearby_new_track_candidate_rate_among_losses": float(class_counts.get("nearby_new_track_candidate", 0) / n_endpoints) if n_endpoints else np.nan,
        "nearby_singleton_candidate_rate_among_losses": float(class_counts.get("nearby_singleton_candidate", 0) / n_endpoints) if n_endpoints else np.nan,
        "multiple_nearby_candidates_rate_among_losses": float(class_counts.get("multiple_nearby_candidates", 0) / n_endpoints) if n_endpoints else np.nan,
        "no_mask_near_prediction_rate_among_losses": float(class_counts.get("no_mask_near_prediction", 0) / n_endpoints) if n_endpoints else np.nan,
    }


def _synthetic_metrics(synthetic: pd.DataFrame) -> dict[str, Any]:
    if synthetic.empty:
        return {"n_cases": 0, "top1_recovery": np.nan, "true_target_in_radius_rate": np.nan, "by_session_gap": {}}
    by_gap: dict[str, Any] = {}
    for gap, frame in synthetic.groupby("session_gap", sort=True):
        by_gap[str(int(gap))] = {
            "n_cases": int(len(frame)),
            "true_target_in_radius_rate": float(frame["true_target_in_search_radius"].astype(bool).mean()),
            "top1_recovery": float(frame["top1_correct"].astype(bool).mean()),
            "median_projected_distance_um": float(pd.to_numeric(frame["projected_distance_um"], errors="coerce").median()),
        }
    return {
        "n_cases": int(len(synthetic)),
        "top1_recovery": float(synthetic["top1_correct"].astype(bool).mean()),
        "true_target_in_radius_rate": float(synthetic["true_target_in_search_radius"].astype(bool).mean()),
        "by_session_gap": by_gap,
    }


def _manual_metrics(classifications: pd.DataFrame) -> dict[str, Any]:
    if classifications.empty or "manual_class" not in classifications.columns:
        return {"available": False}
    manual = classifications["manual_class"].astype(str).str.strip()
    labeled = classifications.loc[manual.ne("")].copy()
    if labeled.empty:
        return {"available": False}
    counts = labeled["manual_class"].astype(str).value_counts().sort_index().to_dict()
    denominator = int((labeled["manual_class"].astype(str) != "ignore").sum())
    metrics: dict[str, Any] = {"available": True, "n_labeled": int(len(labeled)), "n_nonignored": denominator, "class_counts": {str(k): int(v) for k, v in counts.items()}}
    for label, key in (
        ("segmentation_dropout", "segmentation_dropout_rate"),
        ("matcher_fragment", "matcher_fragment_rate"),
        ("true_disappearance", "true_disappearance_rate"),
        ("edge_or_fov_loss", "edge_or_fov_loss_rate"),
        ("ambiguous", "ambiguous_rate"),
    ):
        metrics[key] = float((labeled["manual_class"].astype(str) == label).sum() / denominator) if denominator else np.nan
    metrics["id_switch_rate"] = None
    metrics["id_switch_note"] = "Not definable from endpoint manual labels alone without a separate wrong-link adjudication set."
    return metrics


def _runtime_metrics(runtime: pd.DataFrame, stage_timings: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {"stage_durations_seconds": {str(k): float(v) for k, v in stage_timings.items()}}
    if runtime.empty:
        result.update({"slowest_pair": None, "elapsed_candidate_correlation": np.nan, "median_elapsed_by_gap": {}, "likely_bottlenecks": []})
        return result
    elapsed = pd.to_numeric(runtime["elapsed_sec"], errors="coerce")
    if elapsed.notna().any():
        slow_index = elapsed.idxmax()
        row = runtime.loc[slow_index]
        result["slowest_pair"] = {
            "session_pair": _text(row.get("session_pair")), "pair_gap": _int(row.get("pair_gap")),
            "elapsed_sec": _float(row.get("elapsed_sec")), "candidate_count": _int(row.get("candidate_count")),
        }
    else:
        result["slowest_pair"] = None
    valid = elapsed.notna() & pd.to_numeric(runtime["candidate_count"], errors="coerce").notna()
    if valid.sum() >= 2:
        result["elapsed_candidate_correlation"] = float(np.corrcoef(elapsed.loc[valid], pd.to_numeric(runtime.loc[valid, "candidate_count"]))[0, 1])
    else:
        result["elapsed_candidate_correlation"] = np.nan
    result["median_elapsed_by_gap"] = {
        str(int(gap)): float(pd.to_numeric(frame["elapsed_sec"], errors="coerce").median())
        for gap, frame in runtime.groupby("pair_gap", dropna=True, sort=True)
    }
    bottlenecks: list[str] = []
    corr = _float(result.get("elapsed_candidate_correlation"))
    if np.isfinite(corr) and corr >= 0.5:
        bottlenecks.append("pair_runtime_scales_with_candidate_count")
    affine_total = _float(stage_timings.get("daywise_affine_roi_matching"))
    graph_total = _float(stage_timings.get("graph_roi_matching"))
    if np.isfinite(graph_total) and np.isfinite(affine_total) and graph_total > affine_total:
        bottlenecks.append("graph_stage_total_exceeds_affine_stage_total")
    if result.get("slowest_pair") is not None:
        bottlenecks.append("inspect_slowest_session_pair")
    result["likely_bottlenecks"] = bottlenecks
    return result


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
    synthetic_min_score: float = 0.35,
    synthetic_min_dice: float = 0.10,
    synthetic_max_distance_um: float = 5.0,
    synthetic_max_ambiguity: float = 0.85,
    synthetic_require_consensus: bool = True,
    synthetic_require_no_cycle_conflict: bool = True,
    synthetic_require_no_transform_fallback: bool = True,
    synthetic_require_interior: bool = True,
) -> dict[str, Any]:
    """Run all read-only evaluator stages and write tables/PNG review figures."""

    match_dir = Path(match_dir).resolve()
    output_dir = Path(output_dir).resolve()
    policy = str(policy).strip().lower()
    if policy not in {"graph", "balanced", "high"}:
        raise ValueError("policy must be one of: graph, balanced, high")
    if match_dir == output_dir:
        raise ValueError("output_dir must be separate from match_dir")
    if lookahead < 1 or lookahead > 3:
        raise ValueError("lookahead must be between 1 and 3 for this evaluator")
    if search_radius_um <= 0 or crop_radius_um <= 0 or z_radius < 0:
        raise ValueError("search/crop radii must be positive and z_radius must be nonnegative")

    previous_classifications = _prepare_output_directory(output_dir, overwrite=overwrite)
    discovered = discover_evaluator_inputs(match_dir, policy)
    if discovered["missing_required"]:
        raise FileNotFoundError("Missing evaluator inputs: " + ", ".join(discovered["missing_required"]))

    manifest = _session_table(_read_csv(match_dir / "session_manifest_resolved.csv"))
    features = _read_csv(match_dir / "roi_features.csv")
    transforms = _read_csv(match_dir / "pairwise_transforms.csv")
    tracks = _read_csv(match_dir / f"tracks_{policy}.csv")
    track_edges = _read_csv(match_dir / f"track_edges_{policy}.csv")
    pair_candidates = _read_csv(match_dir / "pairwise_candidates.csv")
    high = _read_csv(match_dir / "pairwise_matches_high.csv")
    balanced = _read_csv(match_dir / "pairwise_matches_balanced.csv")
    graph = _read_csv(match_dir / "pairwise_matches_graph.csv")
    policy_matches = {"high": high, "balanced": balanced, "graph": graph}[policy]
    spacing, spacing_source = load_matcher_spacing(match_dir)

    if state_csv:
        state_path = Path(state_csv).resolve()
        if not state_path.is_file():
            raise FileNotFoundError(f"State CSV was not found: {state_path}")
        state = _read_csv(state_path)
        if not {"track_uid", "session_index"}.issubset(state.columns):
            raise ValueError("state CSV must contain track_uid and session_index columns")
    else:
        state = pd.DataFrame()

    endpoints = detect_endpoint_events(
        tracks, features, manifest, policy=policy, lookahead=lookahead, state=state
    )
    endpoints = add_endpoint_local_density(
        endpoints, features, spacing=spacing, radius_um=search_radius_um
    )
    endpoints = enrich_endpoint_preceding_evidence(endpoints, track_edges, policy_matches)
    endpoints = _merge_extraction_enrichment(endpoints, extraction_dir)

    endpoint_candidates, candidate_status = search_endpoint_candidates(
        endpoints,
        tracks,
        features,
        manifest,
        transforms,
        policy=policy,
        lookahead=lookahead,
        search_radius_um=search_radius_um,
        spacing=spacing,
        candidates=pair_candidates,
        high_matches=high,
        balanced_matches=balanced,
        graph_matches=graph,
    )
    classifications = classify_endpoint_events(endpoints, endpoint_candidates, candidate_status)
    classifications = _restore_manual_columns(classifications, previous_classifications)

    trust_config = SyntheticTrustConfig(
        min_score=float(synthetic_min_score),
        min_dice=float(synthetic_min_dice),
        max_distance_um=float(synthetic_max_distance_um),
        max_ambiguity=float(synthetic_max_ambiguity),
        require_consensus=bool(synthetic_require_consensus),
        require_no_cycle_conflict=bool(synthetic_require_no_cycle_conflict),
        require_no_transform_fallback=bool(synthetic_require_no_transform_fallback),
        require_interior=bool(synthetic_require_interior),
    )
    synthetic = build_synthetic_gap_benchmark(
        tracks,
        endpoints,
        features,
        manifest,
        transforms,
        policy=policy,
        search_radius_um=search_radius_um,
        spacing=spacing,
        random_seed=random_seed,
        trust_config=trust_config,
    )

    review_manifest = build_manual_review_manifest(
        endpoints, features, max_review_panels=max_review_panels, random_seed=random_seed
    )
    if not review_manifest.empty and not classifications.empty:
        review_manifest = review_manifest.merge(
            classifications[[
                "endpoint_id", "classification", "candidate_count", "stitch_candidate_count",
                "manual_class", "manual_target_track_uid", "manual_target_session_index",
                "manual_target_label", "manual_confidence", "reviewer_notes",
            ]],
            on="endpoint_id",
            how="left",
            validate="one_to_one",
        )

    tracking_dropout = build_dropout_summary(endpoints, classifications, manifest, tracks)
    if not tracking_dropout.empty:
        tracking_dropout.insert(0, "summary_type", "tracking_session")
        tracking_dropout.insert(1, "stratum_type", "session")
        tracking_dropout.insert(2, "stratum_value", tracking_dropout["session_id"].astype(str))
    state_dropout = build_state_dropout_stratification(
        state,
        endpoints,
        classifications,
        tracks,
        features,
        manifest,
        policy=policy,
        spacing=spacing,
        local_density_radius_um=search_radius_um,
    )
    dropout = pd.concat([tracking_dropout, state_dropout], ignore_index=True, sort=False)

    runtime = build_runtime_summary(match_dir, policy)
    stage_timings = load_master_stage_timings(match_dir)

    _write_csv(output_dir / "endpoint_events.csv", endpoints, ENDPOINT_COLUMNS)
    _write_csv(output_dir / "endpoint_candidates.csv", endpoint_candidates, ENDPOINT_CANDIDATE_COLUMNS)
    _write_csv(output_dir / "endpoint_classification.csv", classifications, CLASSIFICATION_COLUMNS)
    _write_csv(output_dir / "synthetic_gap_benchmark.csv", synthetic, SYNTHETIC_COLUMNS)
    review_columns = list(review_manifest.columns) if not review_manifest.empty else [
        "endpoint_id", "sample_type", "sampling_stratum", "spatial_stratum", "state_stratum",
        "matched_low_endpoint_id", "state_match_distance", "state_match_covariates",
    ]
    _write_csv(output_dir / "manual_review_manifest.csv", review_manifest, review_columns)
    dropout.to_csv(output_dir / "dropout_summary.csv", index=False)
    runtime.to_csv(output_dir / "matching_runtime_summary.csv", index=False)

    try:
        from endpoint_review_plots import plot_endpoint_contact_sheet, plot_runtime_summaries
    except ImportError:  # pragma: no cover - package import path
        from plotting.endpoint_review_plots import plot_endpoint_contact_sheet, plot_runtime_summaries
    plot_dir = output_dir / "review_panels"
    plot_dir.mkdir(exist_ok=True)
    if not review_manifest.empty:
        plot_endpoint_contact_sheet(
            review_manifest,
            manifest,
            features,
            endpoint_candidates,
            transforms,
            output_dir=plot_dir,
            max_panels=max_review_panels,
            crop_radius_um=crop_radius_um,
            z_radius=z_radius,
            spacing=spacing,
        )
    else:
        # Still emit the required deterministic contact-sheet placeholder.
        plot_endpoint_contact_sheet(
            review_manifest,
            manifest,
            features,
            endpoint_candidates,
            transforms,
            output_dir=plot_dir,
            max_panels=max_review_panels,
            crop_radius_um=crop_radius_um,
            z_radius=z_radius,
            spacing=spacing,
        )
    plot_runtime_summaries(runtime, output_dir)

    classification_counts = classifications["classification"].value_counts().sort_index().to_dict() if not classifications.empty else {}
    tracking_metrics = _tracking_metrics(tracks, manifest, endpoints, classifications)
    synthetic_metrics = _synthetic_metrics(synthetic)
    manual_metrics = _manual_metrics(classifications)
    state_metrics = _state_endpoint_metrics(endpoints, state, manifest)
    runtime_metrics = _runtime_metrics(runtime, stage_timings)

    summary = {
        "policy": policy,
        "lookahead": int(lookahead),
        "search_radius_um": float(search_radius_um),
        "spacing_um": asdict(spacing),
        "spacing_source": spacing_source,
        "n_sessions": int(len(manifest)),
        "n_tracks": int(len(tracks)),
        "n_endpoints": int(len(endpoints)),
        "n_endpoint_candidates": int(len(endpoint_candidates)),
        "classification_counts": {str(k): int(v) for k, v in classification_counts.items()},
        "tracking": tracking_metrics,
        "synthetic_gap_benchmark": synthetic_metrics,
        "synthetic_trust_config": asdict(trust_config),
        "manual_ground_truth": manual_metrics,
        "state_dependent": state_metrics,
        "runtime": runtime_metrics,
        # Backward-compatible headline fields for quick inspection.
        "same_track_gap_recoveries": int(tracking_metrics["existing_gap_recovery_count"]),
        "synthetic_gap_cases": int(synthetic_metrics["n_cases"]),
        "synthetic_top1_recovery": synthetic_metrics["top1_recovery"],
        "manual_ground_truth_metrics_available": bool(manual_metrics.get("available", False)),
        "manual_class_counts": manual_metrics.get("class_counts", {}),
        "runtime_bottleneck": runtime_metrics.get("slowest_pair"),
        "state_data_available": bool(not state.empty),
        "optional_inputs": {
            "extraction_dir": str(Path(extraction_dir).resolve()) if extraction_dir else None,
            "state_csv": str(Path(state_csv).resolve()) if state_csv else None,
        },
        "canonical_matcher_modified": False,
        "canonical_matcher_outputs_written": False,
        "input_discovery": discovered,
    }
    (output_dir / "matcher_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
    )

    matcher_run_log = _read_json(match_dir / "run_log.json")
    run_log = {
        "evaluator_version": "daywise_endpoint_evaluator_v2",
        "policy": policy,
        "lookahead": int(lookahead),
        "search_radius_um": float(search_radius_um),
        "crop_radius_um": float(crop_radius_um),
        "z_radius": int(z_radius),
        "random_seed": int(random_seed),
        "synthetic_trust_config": asdict(trust_config),
        "spacing_um": asdict(spacing),
        "spacing_source": spacing_source,
        "match_dir": str(match_dir),
        "output_dir": str(output_dir),
        "input_files_found": discovered["required"] | discovered["optional"],
        "input_files_missing": discovered["missing_required"] + discovered["missing_optional"],
        "optional_state_used": bool(not state.empty),
        "optional_extraction_dir": str(Path(extraction_dir).resolve()) if extraction_dir else None,
        "python_package_versions": _package_versions(),
        "evaluator_repo_git_commit": _git_commit(Path(__file__).resolve().parent.parent),
        "source_matcher_git_commit": matcher_run_log.get("git_commit"),
        "source_matcher_algorithm_version": matcher_run_log.get("algorithm_version"),
        "source_graph_matcher_algorithm_version": matcher_run_log.get("graph_matcher_algorithm_version"),
        "master_stage_durations_seconds": stage_timings,
        "row_counts": {
            "endpoint_events": len(endpoints),
            "endpoint_candidates": len(endpoint_candidates),
            "endpoint_classification": len(classifications),
            "synthetic_gap_benchmark": len(synthetic),
            "manual_review_manifest": len(review_manifest),
            "dropout_summary": len(dropout),
            "matching_runtime_summary": len(runtime),
        },
        "omissions": {
            "state_analysis_skipped": bool(state.empty),
            "extraction_enrichment_skipped": extraction_dir is None,
            "per_pair_graph_timing_unavailable": bool(runtime.empty or runtime["graph_stage_seconds"].isna().all()),
        },
        "canonical_matcher_outputs_unchanged": True,
        "canonical_matching_rerun": False,
        "eclipse_recomputed": False,
        "png_only": True,
    }
    (output_dir / "evaluation_run_log.json").write_text(
        json.dumps(run_log, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
    )
    return summary
