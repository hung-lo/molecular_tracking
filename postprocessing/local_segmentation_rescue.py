"""Validation-only local segmentation rescue evaluator.

Phase D1 deliberately has no production write path.  It predicts a missing
target position from spatial/transform evidence, segments only a bounded raw
image crop, and writes proposal/benchmark tables outside canonical outputs.
The CPU connected-component segmenter is a deterministic fallback used when
the production Cellpose/SAM packages are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile


DEFAULT_SPACING_ZYX = (5.0, 710.0 / 1024.0, 710.0 / 1024.0)
DEFAULT_CROP_SHAPE_ZYX = (25, 96, 96)
SEGMENTER_NAME = "threshold_connected_components_v1_cpu_fallback"
OUTPUT_REPORT = "PHASE_D1_LOCAL_SEGMENTATION_RESCUE_REPORT_20260915.md"


@dataclass(frozen=True)
class CropBounds:
    """A bounded crop in ZYX array coordinates."""

    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]
    requested_shape_zyx: tuple[int, int, int]
    volume_shape_zyx: tuple[int, int, int]
    edge_clipped: bool

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(stop - start for start, stop in zip(self.start_zyx, self.stop_zyx))

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(start, stop) for start, stop in zip(self.start_zyx, self.stop_zyx))  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        return {
            "crop_start_zyx": list(self.start_zyx),
            "crop_stop_zyx": list(self.stop_zyx),
            "crop_shape_zyx": list(self.shape_zyx),
            "requested_crop_shape_zyx": list(self.requested_shape_zyx),
            "edge_clipped": bool(self.edge_clipped),
        }


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    matching_dir: Path
    features: pd.DataFrame
    tracks: pd.DataFrame
    sessions: pd.DataFrame
    transforms: pd.DataFrame
    spacing_zyx: tuple[float, float, float]
    spacing_source: str
    run_log: dict[str, Any]
    matching_sha256: str
    manifest_sha256: str
    git_commit: str
    image_hashes: dict[str, dict[str, str]]


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _present(value: Any) -> bool:
    return value is not None and not pd.isna(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def compute_crop_bounds(
    center_xyz: Iterable[float],
    volume_shape_zyx: Iterable[int],
    crop_shape_zyx: Iterable[int] = DEFAULT_CROP_SHAPE_ZYX,
) -> CropBounds:
    """Return deterministic clipped bounds; input centers are XYZ, arrays ZYX."""

    center = np.asarray(tuple(center_xyz), dtype=float)
    volume = tuple(int(v) for v in volume_shape_zyx)
    requested = tuple(int(v) for v in crop_shape_zyx)
    if center.shape != (3,):
        raise ValueError("center_xyz must contain exactly three values")
    if len(volume) != 3 or len(requested) != 3 or any(v <= 0 for v in requested):
        raise ValueError("volume_shape_zyx and crop_shape_zyx must be positive 3-vectors")
    if any(v <= 0 for v in volume):
        raise ValueError("volume_shape_zyx must be positive")
    starts: list[int] = []
    stops: list[int] = []
    clipped = False
    for axis, (coord, size, limit) in enumerate(zip(center[::-1], requested, volume)):
        if size >= limit:
            starts.append(0)
            stops.append(limit)
            clipped |= size > limit
            continue
        start = int(round(coord)) - size // 2
        start = max(0, min(start, limit - size))
        stop = start + size
        clipped |= int(round(coord)) - size // 2 != start
        starts.append(start)
        stops.append(stop)
    return CropBounds(tuple(starts), tuple(stops), requested, volume, clipped)


def dice_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    """Return Dice and IoU for two boolean masks."""

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} != {b.shape}")
    a_count = int(a.sum())
    b_count = int(b.sum())
    intersection = int(np.logical_and(a, b).sum())
    union = a_count + b_count - intersection
    if a_count == b_count == 0:
        return 1.0, 1.0
    return (2.0 * intersection / (a_count + b_count) if a_count + b_count else 0.0,
            intersection / union if union else 0.0)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    coords = np.argwhere(mask)
    if not len(coords):
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(int(v) for v in (*lo, *hi))


def _apply_stored_b_to_a(point_xyz: np.ndarray, row: Mapping[str, Any] | pd.Series) -> np.ndarray:
    """Apply the canonical stored B→A restricted transform."""

    p = np.asarray(point_xyz, dtype=float)
    matrix = np.array([
        [_finite(row.get("y_from_y"), 1.0), _finite(row.get("y_from_x"), 0.0)],
        [_finite(row.get("x_from_y"), 0.0), _finite(row.get("x_from_x"), 1.0)],
    ])
    intercept = np.array([_finite(row.get("y_intercept"), 0.0), _finite(row.get("x_intercept"), 0.0)])
    out = p.copy()
    out[0] = _finite(row.get("z_intercept"), 0.0) + _finite(row.get("z_scale"), 1.0) * p[0]
    out[1:] = intercept + matrix @ p[1:]
    return out


def _apply_forward(point_xyz: np.ndarray, row: Mapping[str, Any] | pd.Series) -> np.ndarray:
    """Map A→B by inverting the stored B→A transform."""

    z_scale = _finite(row.get("z_scale"), 1.0)
    matrix = np.array([
        [_finite(row.get("y_from_y"), 1.0), _finite(row.get("y_from_x"), 0.0)],
        [_finite(row.get("x_from_y"), 0.0), _finite(row.get("x_from_x"), 1.0)],
    ])
    if abs(z_scale) <= 1e-10 or abs(float(np.linalg.det(matrix))) <= 1e-10:
        raise ValueError("singular transform")
    p = np.asarray(point_xyz, dtype=float)
    out = p.copy()
    out[0] = (p[0] - _finite(row.get("z_intercept"), 0.0)) / z_scale
    intercept = np.array([_finite(row.get("y_intercept"), 0.0), _finite(row.get("x_intercept"), 0.0)])
    out[1:] = np.linalg.solve(matrix, p[1:] - intercept)
    return out


def _transform_matrix(row: Mapping[str, Any] | pd.Series, forward: bool) -> np.ndarray:
    """Homogeneous XYZ transform for composing transform evidence."""

    basis = np.eye(4, dtype=float)
    if forward:
        basis[:3, 3] = _apply_forward(np.zeros(3), row)
        basis[0, 0] = 1.0 / _finite(row.get("z_scale"), 1.0)
        matrix = np.array([
            [_finite(row.get("y_from_y"), 1.0), _finite(row.get("y_from_x"), 0.0)],
            [_finite(row.get("x_from_y"), 0.0), _finite(row.get("x_from_x"), 1.0)],
        ])
        basis[1:3, 1:3] = np.linalg.inv(matrix)
    else:
        basis[:3, 3] = _apply_stored_b_to_a(np.zeros(3), row)
        basis[0, 0] = _finite(row.get("z_scale"), 1.0)
        basis[1:3, 1:3] = np.array([
            [_finite(row.get("y_from_y"), 1.0), _finite(row.get("y_from_x"), 0.0)],
            [_finite(row.get("x_from_y"), 0.0), _finite(row.get("x_from_x"), 1.0)],
        ])
    return basis


class TransformGraph:
    """Small deterministic transform graph; canonical files are read only."""

    def __init__(self, transforms: pd.DataFrame):
        self._edges: dict[str, list[tuple[str, np.ndarray, str]]] = {}
        for _, row in transforms.iterrows():
            a, b = str(row.get("day_a")), str(row.get("day_b"))
            if not a or not b or a == "nan" or b == "nan":
                continue
            try:
                self._edges.setdefault(a, []).append((b, _transform_matrix(row, True), "direct"))
                self._edges.setdefault(b, []).append((a, _transform_matrix(row, False), "direct"))
            except (ValueError, np.linalg.LinAlgError):
                continue
        for key in self._edges:
            self._edges[key].sort(key=lambda item: item[0])

    def project(self, source_session: str, target_session: str, point_xyz: Iterable[float]) -> tuple[np.ndarray, str, list[str]] | None:
        source_session, target_session = str(source_session), str(target_session)
        point = np.asarray(tuple(point_xyz), dtype=float)
        if source_session == target_session:
            return point, "direct", [source_session]
        queue: list[tuple[str, np.ndarray, list[str], list[str]]] = [(source_session, np.eye(4), [source_session], [])]
        visited = {source_session}
        while queue:
            current, composed, path, methods = queue.pop(0)
            for nxt, edge, method in self._edges.get(current, []):
                if nxt in visited:
                    continue
                new_composed = edge @ composed
                new_path = path + [nxt]
                new_methods = methods + [method]
                if nxt == target_session:
                    homogeneous = np.r_[point, 1.0]
                    projected = (new_composed @ homogeneous)[:3]
                    return projected, ("direct" if len(new_methods) == 1 else "composed"), new_path
                visited.add(nxt)
                queue.append((nxt, new_composed, new_path, new_methods))
        return None


def rank_candidates(
    candidates: list[dict[str, Any]],
    predicted_xyz: Iterable[float],
    spacing_zyx: Iterable[float] = DEFAULT_SPACING_ZYX,
    expected_volume_um3: float | None = None,
) -> list[dict[str, Any]]:
    """Rank only by geometry; no intensity or state columns are consulted."""

    predicted = np.asarray(tuple(predicted_xyz), dtype=float)
    # Candidate/prediction coordinates are XYZ; the configured spacing is ZYX.
    spacing = np.asarray(tuple(spacing_zyx), dtype=float)[::-1]
    for candidate in candidates:
        distance = float(np.linalg.norm((np.asarray(candidate["centroid_xyz"]) - predicted) * spacing))
        candidate["distance_from_prediction_um"] = distance
        volume = _finite(candidate.get("volume_um3"))
        if expected_volume_um3 and expected_volume_um3 > 0 and np.isfinite(volume):
            candidate["volume_ratio_to_expected"] = volume / expected_volume_um3
            size_penalty = abs(float(np.log(max(volume / expected_volume_um3, 1e-12))))
        else:
            candidate["volume_ratio_to_expected"] = np.nan
            size_penalty = 0.0
        candidate["geometric_rank_score"] = distance + 0.5 * size_penalty + (2.0 if candidate.get("touches_crop_edge") else 0.0)
    return sorted(candidates, key=lambda item: (float(item["geometric_rank_score"]), int(item["candidate_id"])))


def segment_local_threshold(
    image_crop_zyx: np.ndarray,
    spacing_zyx: Iterable[float] = DEFAULT_SPACING_ZYX,
    *,
    threshold_percentile: float = 85.0,
    min_voxels: int = 20,
    connectivity: int = 1,
    crop_start_zyx: Iterable[int] = (0, 0, 0),
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    """Segment a crop with deterministic thresholded 3D connected components."""

    image = np.asarray(image_crop_zyx)
    if image.ndim != 3:
        raise ValueError("image_crop_zyx must be 3D ZYX")
    finite = image[np.isfinite(image)]
    if not len(finite):
        return [], np.zeros(image.shape, dtype=np.int32), np.nan
    if not 0 < threshold_percentile < 100:
        raise ValueError("threshold_percentile must be in (0, 100)")
    threshold = float(np.percentile(finite, threshold_percentile))
    # Avoid treating a flat background as foreground when a high percentile is
    # still equal to the minimum (common in sparse synthetic crops).
    finite_min = float(np.min(finite))
    if threshold <= finite_min:
        threshold = float(np.nextafter(finite_min, np.inf))
    binary = np.isfinite(image) & (image >= threshold)
    structure = ndimage.generate_binary_structure(3, int(connectivity))
    labels, count = ndimage.label(binary, structure=structure)
    spacing = np.asarray(tuple(spacing_zyx), dtype=float)
    start = np.asarray(tuple(crop_start_zyx), dtype=int)
    candidates: list[dict[str, Any]] = []
    compact = np.zeros(labels.shape, dtype=np.int32)
    next_id = 0
    for label in range(1, int(count) + 1):
        coords = np.argwhere(labels == label)
        if len(coords) < int(min_voxels):
            continue
        next_id += 1
        compact[labels == label] = next_id
        centroid_zyx = coords.mean(axis=0) + start
        bounds = _bbox(labels == label)
        assert bounds is not None
        local_lo = np.asarray(bounds[:3])
        local_hi = np.asarray(bounds[3:])
        global_bbox = tuple(int(v) for v in (*local_lo + start, *local_hi + start))
        touches_crop = bool(np.any(local_lo == 0) or np.any(local_hi == np.asarray(image.shape)))
        candidates.append({
            "candidate_id": next_id,
            "local_label": next_id,
            "centroid_zyx": centroid_zyx.tolist(),
            "centroid_xyz": centroid_zyx[::-1].tolist(),
            "volume_voxels": int(len(coords)),
            "volume_um3": float(len(coords) * np.prod(spacing)),
            "bbox_zyx": global_bbox,
            "bbox_depth": int(global_bbox[3] - global_bbox[0]),
            "touches_crop_edge": touches_crop,
            "touches_volume_edge": False,
            "geometry_source": SEGMENTER_NAME,
        })
    return candidates, compact, threshold


def _load_tif(path: str | Path) -> np.ndarray:
    return tifffile.memmap(Path(path), mode="r")


def _load_run_context(run_dir: str | Path) -> RunContext:
    root = Path(run_dir).expanduser().resolve()
    matching = root / "matching"
    if not matching.is_dir():
        raise FileNotFoundError(f"Missing matching directory: {matching}")
    manifest_path = root / "selected_session_manifest.csv"
    if not manifest_path.is_file():
        manifest_path = matching / "session_manifest_resolved.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing selected session manifest below {root}")
    features_path = matching / "roi_features.csv"
    tracks_path = matching / "tracks_graph.csv"
    transforms_path = matching / "pairwise_transforms.csv"
    for path in (features_path, tracks_path, transforms_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical input: {path}")
    sessions = pd.read_csv(manifest_path, low_memory=False)
    sessions["session_id"] = sessions["session_id"].astype(str)
    sessions["session_index"] = pd.to_numeric(sessions["session_index"], errors="raise").astype(int)
    sessions = sessions.sort_values("session_index").reset_index(drop=True)
    features = pd.read_csv(features_path, low_memory=False)
    features["session_id"] = features["session_id"].astype(str)
    features["label"] = pd.to_numeric(features["label"], errors="raise").astype(int)
    tracks = pd.read_csv(tracks_path, low_memory=False)
    transforms = pd.read_csv(transforms_path, low_memory=False)
    run_log_path = matching / "run_log.json"
    run_log = json.loads(run_log_path.read_text(encoding="utf-8")) if run_log_path.is_file() else {}
    spacing_payload = run_log.get("spacing", {}) if isinstance(run_log, dict) else {}
    try:
        spacing = tuple(float(spacing_payload[key]) for key in ("z_um", "y_um", "x_um"))
        spacing_source = "matching/run_log.json"
    except (KeyError, TypeError, ValueError):
        spacing = DEFAULT_SPACING_ZYX
        spacing_source = "repository_default"
    matching_hash = hashlib.sha256()
    for path in (features_path, tracks_path, transforms_path):
        matching_hash.update(_sha256(path).encode())
    image_hashes: dict[str, dict[str, str]] = {}
    for item in run_log.get("input_hashes", []) if isinstance(run_log, dict) else []:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id", ""))
        if session_id:
            image_hashes[session_id] = {key: str(item.get(key, "")) for key in ("green_sha256", "red_sha256", "mask_sha256", "green_image_path", "red_image_path", "mask_path")}
    return RunContext(
        root, matching, features, tracks, sessions, transforms, spacing, spacing_source, run_log,
        matching_hash.hexdigest(), _sha256(manifest_path), str(run_log.get("git_commit", _git_commit())), image_hashes,
    )


def _feature_lookup(context: RunContext) -> dict[tuple[str, int], pd.Series]:
    return {(str(row.session_id), int(row.label)): row for _, row in context.features.iterrows()}


def _track_observations(context: RunContext, track: pd.Series) -> dict[int, tuple[str, int]]:
    output: dict[int, tuple[str, int]] = {}
    for session in context.sessions.itertuples(index=False):
        column = f"{session.session_id}_roi"
        value = track.get(column, pd.NA)
        if _present(value):
            output[int(session.session_index)] = (str(session.session_id), int(value))
    return output


def _predict_target(
    context: RunContext,
    observations: dict[int, tuple[str, int]],
    target_index: int,
    lookup: dict[tuple[str, int], pd.Series],
    graph: TransformGraph,
) -> dict[str, Any]:
    target_id = str(context.sessions.iloc[target_index].session_id)
    neighbors = [index for index in observations if index != target_index]
    before = [index for index in neighbors if index < target_index]
    after = [index for index in neighbors if index > target_index]
    before_index = max(before) if before else None
    after_index = min(after) if after else None
    projected: list[np.ndarray] = []
    methods: list[str] = []
    paths: list[list[str]] = []
    for index in (before_index, after_index):
        if index is None:
            continue
        source_id, label = observations[index]
        feature = lookup.get((source_id, label))
        if feature is None:
            continue
        point = feature[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(float)
        result = graph.project(source_id, target_id, point)
        if result is not None and np.all(np.isfinite(result[0])):
            projected.append(result[0][::-1])
            methods.append(result[1])
            paths.append(result[2])
    if projected:
        predicted = np.mean(np.vstack(projected), axis=0)
        method = "direct" if all(item == "direct" for item in methods) else "composed"
        return {"predicted_xyz": predicted.tolist(), "transform_method": method, "transform_paths": paths, "transform_fallback_reason": ""}
    available = [index for index in (before_index, after_index) if index is not None and (index in observations)]
    if len(available) == 2:
        points = []
        for index in available:
            session_id, label = observations[index]
            feature = lookup.get((session_id, label))
            if feature is not None:
                points.append(feature[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(float)[::-1])
        if points:
            return {"predicted_xyz": np.mean(np.vstack(points), axis=0).tolist(), "transform_method": "fallback", "transform_paths": [], "transform_fallback_reason": "transform_unavailable"}
    return {"predicted_xyz": [np.nan, np.nan, np.nan], "transform_method": "unavailable", "transform_paths": [], "transform_fallback_reason": "no_neighbor_observation"}


def _eligible_synthetic_cases(context: RunContext) -> list[dict[str, Any]]:
    lookup = _feature_lookup(context)
    cases: list[dict[str, Any]] = []
    for _, track in context.tracks.iterrows():
        observations = _track_observations(context, track)
        if len(observations) < 3:
            continue
        track_uid = str(track.get("track_uid", track.get("cluster_id", "")))
        for target_index in range(1, len(context.sessions) - 1):
            if target_index not in observations or target_index - 1 not in observations or target_index + 1 not in observations:
                continue
            target_session, target_label = observations[target_index]
            truth = lookup.get((target_session, target_label))
            if truth is None or bool(truth.get("touches_z_edge", False)) or bool(truth.get("touches_xy_edge", False)):
                continue
            source_session, source_label = observations[target_index - 1]
            cases.append({
                "mouse": context.run_dir.parts[-5] if len(context.run_dir.parts) >= 5 else "unknown",
                "run": context.run_dir.name,
                "track_id": track_uid,
                "track_uid": track_uid,
                "source_session_index": target_index - 1,
                "target_session_index": target_index,
                "source_session": source_session,
                "target_session": target_session,
                "source_roi_id": source_label,
                "target_truth_roi_id": target_label,
                "source_label": source_label,
                "target_truth_label": target_label,
                "expected_volume_um3": _finite(truth.get("volume_um3")),
            })
    return cases


def _sample_cases(cases: list[dict[str, Any]], sample_size: int | None, seed: int) -> list[dict[str, Any]]:
    cases = sorted(cases, key=lambda row: (int(row["target_session_index"]), str(row["track_id"]), int(row["source_roi_id"])))
    if not sample_size or sample_size >= len(cases):
        return cases
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(cases), size=int(sample_size), replace=False))
    return [cases[int(index)] for index in indices]


def _find_real_cases(context: RunContext, max_cases: int | None, seed: int) -> list[dict[str, Any]]:
    """Use evaluator no-mask rows when present, otherwise derive internal gaps."""

    candidates: list[dict[str, Any]] = []
    for path in sorted(context.run_dir.rglob("*.csv")):
        if path.name in {"tracks_graph.csv", "roi_features.csv", "pairwise_transforms.csv"}:
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if "no_mask_near_prediction" not in path.name and not ({"track_uid", "session_index"} <= set(header)):
            continue
        try:
            table = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        text = table.agg(lambda row: " ".join(map(str, row.tolist())), axis=1)
        selected = table.loc[text.str.contains("no_mask_near_prediction", case=False, na=False)]
        if not selected.empty:
            for _, row in selected.iterrows():
                candidates.append({
                    "mouse": "unknown", "run": context.run_dir.name, "track_id": str(row.get("track_uid", "")), "track_uid": str(row.get("track_uid", "")),
                    "source_session_index": int(_finite(row.get("session_index", row.get("end_session_index", 0)), 0)),
                    "target_session_index": int(_finite(row.get("target_session_index", row.get("session_index", 0)), 0)),
                    "source_session": str(row.get("session_id", row.get("end_session_id", ""))), "target_session": str(row.get("target_session_id", "")),
                    "source_roi_id": row.get("roi_id", row.get("end_label", "")), "target_truth_roi_id": "", "source_label": row.get("roi_id", row.get("end_label", "")), "target_truth_label": "", "expected_volume_um3": np.nan,
                })
            break
    if not candidates:
        for _, track in context.tracks.iterrows():
            observations = _track_observations(context, track)
            track_uid = str(track.get("track_uid", track.get("cluster_id", "")))
            for target_index in range(1, len(context.sessions) - 1):
                if target_index in observations or target_index - 1 not in observations or target_index + 1 not in observations:
                    continue
                source_session, source_label = observations[target_index - 1]
                candidates.append({
                    "mouse": "unknown", "run": context.run_dir.name, "track_id": track_uid, "track_uid": track_uid,
                    "source_session_index": target_index - 1, "target_session_index": target_index,
                    "source_session": source_session, "target_session": str(context.sessions.iloc[target_index].session_id),
                    "source_roi_id": source_label, "target_truth_roi_id": "", "source_label": source_label, "target_truth_label": "", "expected_volume_um3": np.nan,
                })
    return _sample_cases(candidates, max_cases, seed)


def _session_row(context: RunContext, session_id: str) -> pd.Series:
    rows = context.sessions.loc[context.sessions["session_id"].astype(str).eq(str(session_id))]
    if rows.empty:
        raise KeyError(f"Unknown session: {session_id}")
    return rows.iloc[0]


def _path_value(row: pd.Series, key: str) -> Path | None:
    value = row.get(key)
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_file() else None


def _case_provenance(context: RunContext, target_session: str) -> dict[str, Any]:
    hashes = context.image_hashes.get(str(target_session), {})
    return {
        "git_commit": context.git_commit,
        "canonical_run_path": str(context.run_dir),
        "canonical_matching_sha256": context.matching_sha256,
        "selected_session_manifest_sha256": context.manifest_sha256,
        "target_input_image_hash": hashes.get("red_sha256", ""),
        "target_mask_hash": hashes.get("mask_sha256", ""),
        "segmentation_model": SEGMENTER_NAME,
        "segmentation_parameters": {"threshold_percentile": 85.0, "min_voxels": 20, "connectivity": 1},
        "voxel_spacing_zyx_um": list(context.spacing_zyx),
        "proposal_only": True,
    }


def _run_case(
    context: RunContext,
    case: dict[str, Any],
    *,
    synthetic: bool,
    crop_shape_zyx: tuple[int, int, int],
    threshold_percentile: float,
    min_voxels: int,
    lookup: dict[tuple[str, int], pd.Series],
    graph: TransformGraph,
    image_cache: dict[str, np.ndarray],
    mask_cache: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    target_index = int(case["target_session_index"])
    target_session = str(case.get("target_session", ""))
    if not target_session or target_session == "nan":
        target_session = str(context.sessions.iloc[target_index].session_id)
    target_row = _session_row(context, target_session)
    source_session = str(case.get("source_session", ""))
    if not source_session or source_session == "nan":
        source_session = str(context.sessions.iloc[max(0, int(case["source_session_index"]))].session_id)
    source_label = int(case["source_label"])
    source_feature = lookup.get((source_session, source_label))
    observations = {int(case["source_session_index"]): (source_session, source_label)}
    if synthetic:
        track = context.tracks.loc[context.tracks["track_uid"].astype(str).eq(str(case["track_uid"]))]
        if not track.empty:
            observations = _track_observations(context, track.iloc[0])
    prediction = _predict_target(context, observations, target_index, lookup, graph)
    image_path = _path_value(target_row, "red_image_path") or _path_value(target_row, "green_image_path")
    status = "candidate_generated"
    if image_path is None:
        status = "missing_raw_image"
        base = {**case, **prediction, "status": status, "failure_reason": status, "n_candidates": 0, **_case_provenance(context, target_session)}
        return base, [], None, {}
    if not np.all(np.isfinite(prediction["predicted_xyz"])):
        status = "transform_unavailable"
        base = {**case, **prediction, "status": status, "failure_reason": status, "n_candidates": 0, "input_image_path": str(image_path), **_case_provenance(context, target_session)}
        return base, [], None, {}
    image = image_cache.setdefault(str(image_path), _load_tif(image_path))
    bounds = compute_crop_bounds(prediction["predicted_xyz"], image.shape, crop_shape_zyx)
    crop = np.asarray(image[bounds.slices])
    candidates, labels, threshold = segment_local_threshold(crop, context.spacing_zyx, threshold_percentile=threshold_percentile, min_voxels=min_voxels, crop_start_zyx=bounds.start_zyx)
    expected = _finite(case.get("expected_volume_um3"))
    ranked = rank_candidates(candidates, prediction["predicted_xyz"], context.spacing_zyx, expected if np.isfinite(expected) else None)
    if not ranked:
        status = "no_segmentation_object"
    elif len(ranked) > 1:
        status = "multiple_candidates"
    selected = ranked[0] if ranked else None
    for candidate in ranked:
        candidate.update({"mouse": case.get("mouse", "unknown"), "run": context.run_dir.name, "track_id": case["track_id"], "track_uid": case["track_uid"], "source_session": source_session, "target_session": target_session, "source_roi_id": source_label, "target_truth_roi_id": case.get("target_truth_roi_id", ""), "predicted_x": prediction["predicted_xyz"][0], "predicted_y": prediction["predicted_xyz"][1], "predicted_z": prediction["predicted_xyz"][2], "transform_method": prediction["transform_method"], "transform_paths": json.dumps(prediction["transform_paths"]), "transform_fallback_reason": prediction["transform_fallback_reason"], "crop_start_zyx": list(bounds.start_zyx), "crop_stop_zyx": list(bounds.stop_zyx), "threshold": threshold, **_case_provenance(context, target_session)})
    truth_metrics: dict[str, Any] = {}
    bias: dict[str, Any] = {}
    truth_mask = None
    if synthetic:
        truth_label = int(case["target_truth_label"])
        mask_path = _path_value(target_row, "mask_path")
        if mask_path is not None:
            truth_stack = mask_cache.setdefault(str(mask_path), _load_tif(mask_path))
            truth_mask = np.asarray(truth_stack[bounds.slices]) == truth_label
            truth_rank = []
            for candidate in ranked:
                candidate_mask = labels == int(candidate["local_label"])
                dice, iou = dice_iou(candidate_mask, truth_mask)
                candidate["truth_dice"] = dice
                candidate["truth_iou"] = iou
                truth_rank.append((dice, iou, int(candidate["candidate_id"])))
            best_truth = max(truth_rank, default=(0.0, 0.0, 0))
            selected_mask = labels == int(selected["local_label"]) if selected else np.zeros(crop.shape, bool)
            selected_dice, selected_iou = dice_iou(selected_mask, truth_mask)
            truth_volume = int(truth_mask.sum())
            selected_volume = int(selected_mask.sum())
            spacing_xyz = np.asarray(context.spacing_zyx)[::-1]
            wrong_candidates = [item for item in ranked if float(item.get("truth_dice", 0.0)) <= 0.5]
            nearest_wrong = min((float(item["distance_from_prediction_um"]) for item in wrong_candidates), default=np.nan)
            truth_feature_xyz = lookup[(target_session, truth_label)][["centroid_z", "centroid_y", "centroid_x"]].to_numpy(float)[::-1]
            truth_bbox = _bbox(truth_mask)
            selected_bbox = _bbox(selected_mask)
            bbox_overlap = np.nan
            if truth_bbox is not None and selected_bbox is not None:
                intersection = np.maximum(0, np.minimum(truth_bbox[3:], selected_bbox[3:]) - np.maximum(truth_bbox[:3], selected_bbox[:3]))
                intersection_volume = int(np.prod(intersection))
                truth_bbox_volume = int(np.prod(np.asarray(truth_bbox[3:]) - np.asarray(truth_bbox[:3])))
                selected_bbox_volume = int(np.prod(np.asarray(selected_bbox[3:]) - np.asarray(selected_bbox[:3])))
                bbox_union = truth_bbox_volume + selected_bbox_volume - intersection_volume
                bbox_overlap = intersection_volume / bbox_union if bbox_union else 0.0
            truth_metrics = {"rescue_found": bool(selected), "selected_candidate_exists": bool(selected), "selected_candidate_is_correct": bool(selected and selected_dice > 0.5), "wrong_neighbor": bool(selected and selected_dice <= 0.5), "prediction_error_um": float(np.linalg.norm((np.asarray(selected["centroid_xyz"]) - np.asarray(prediction["predicted_xyz"])) * spacing_xyz)) if selected else np.nan, "centroid_error_um": float(np.linalg.norm((np.asarray(selected["centroid_xyz"]) - truth_feature_xyz) * spacing_xyz)) if selected else np.nan, "selected_centroid_error_to_truth_um": float(np.linalg.norm((np.asarray(selected["centroid_xyz"]) - truth_feature_xyz) * spacing_xyz)) if selected else np.nan, "dice_3d": selected_dice, "iou_3d": selected_iou, "volume_ratio_rescue_to_truth": selected_volume / truth_volume if truth_volume else np.nan, "absolute_volume_error_voxels": abs(selected_volume - truth_volume), "bbox_overlap": bbox_overlap, "truth_rank_among_candidates": (1 + sorted((item[0] for item in truth_rank), reverse=True).index(selected_dice)) if selected else np.nan, "best_candidate_dice": best_truth[0], "nearest_wrong_candidate_distance_um": nearest_wrong, "number_of_candidates": len(ranked), "edge_limited": bool(bounds.edge_clipped or (selected and selected.get("touches_crop_edge"))), "truth_volume_voxels": truth_volume}
            green_path = _path_value(target_row, "green_image_path")
            red_path = _path_value(target_row, "red_image_path")
            if green_path is not None and red_path is not None and selected:
                green = image_cache.setdefault(str(green_path), _load_tif(green_path))
                red = image_cache.setdefault(str(red_path), _load_tif(red_path))
                rescued = selected_mask
                truth_green = float(np.asarray(green[bounds.slices])[truth_mask].mean()) if truth_mask.any() else np.nan
                rescue_green = float(np.asarray(green[bounds.slices])[rescued].mean()) if rescued.any() else np.nan
                truth_red = float(np.asarray(red[bounds.slices])[truth_mask].mean()) if truth_mask.any() else np.nan
                rescue_red = float(np.asarray(red[bounds.slices])[rescued].mean()) if rescued.any() else np.nan
                truth_ratio = truth_green / truth_red if np.isfinite(truth_green) and np.isfinite(truth_red) and truth_red else np.nan
                rescue_ratio = rescue_green / rescue_red if np.isfinite(rescue_green) and np.isfinite(rescue_red) and rescue_red else np.nan
                bias = {"green_original": truth_green, "green_rescued": rescue_green, "green_absolute_difference": abs(rescue_green - truth_green), "green_relative_difference": (rescue_green - truth_green) / truth_green if truth_green else np.nan, "red_original": truth_red, "red_rescued": rescue_red, "red_absolute_difference": abs(rescue_red - truth_red), "red_relative_difference": (rescue_red - truth_red) / truth_red if truth_red else np.nan, "ratio_original": truth_ratio, "ratio_rescued": rescue_ratio, "ratio_absolute_difference": abs(rescue_ratio - truth_ratio), "ratio_relative_difference": (rescue_ratio - truth_ratio) / truth_ratio if truth_ratio else np.nan, "eclipse_original": np.nan, "eclipse_rescued": np.nan}
    base = {**case, **prediction, "status": status, "failure_reason": "" if status in {"candidate_generated", "multiple_candidates"} else status, "n_candidates": len(ranked), "selected_candidate_id": selected["candidate_id"] if selected else np.nan, "crop_center_xyz": prediction["predicted_xyz"], "crop_start_zyx": list(bounds.start_zyx), "crop_stop_zyx": list(bounds.stop_zyx), "crop_shape_zyx": list(bounds.shape_zyx), "crop_physical_size_um": (np.asarray(bounds.shape_zyx) * context.spacing_zyx).tolist(), "edge_clipping": bounds.edge_clipped, "input_image_path": str(image_path), "input_image_hash": context.image_hashes.get(target_session, {}).get("red_sha256", ""), "threshold": threshold, "voxel_spacing_zyx_um": list(context.spacing_zyx), **truth_metrics, **bias, **_case_provenance(context, target_session)}
    return base, ranked, selected, {"truth_mask": truth_mask, "candidate_labels": labels, "crop": crop, "bounds": bounds, "image_path": image_path}


def _write_pngs(output_dir: Path, records: list[dict[str, Any]], artifacts: list[dict[str, Any]], *, synthetic: bool) -> tuple[Path, Path]:
    """Write a small deterministic set of review panels and required plots."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_dir = output_dir / "review_panels"
    plot_dir = output_dir / "summary_plots"
    panel_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(artifacts[:12]):
        record, selected, arrays = item
        if not arrays:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        raw = np.asarray(arrays["crop"])
        axes[0].imshow(np.max(raw, axis=0), cmap="gray")
        axes[0].set_title("target raw/local")
        truth = arrays.get("truth_mask")
        if truth is not None:
            axes[1].imshow(np.max(raw, axis=0), cmap="gray")
            axes[1].contour(np.max(truth, axis=0), levels=[0.5], colors="lime")
            axes[1].set_title("hidden truth")
        else:
            axes[1].axis("off")
        axes[2].imshow(np.max(raw, axis=0), cmap="gray")
        if selected:
            labels = arrays["candidate_labels"] == int(selected["local_label"])
            axes[2].contour(np.max(labels, axis=0), levels=[0.5], colors="cyan")
        axes[2].set_title("rescued candidate")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"{record.get('track_id', '')} {record.get('target_session', '')} {record.get('status', '')}")
        fig.tight_layout()
        fig.savefig(panel_dir / f"case_{index:04d}.png", dpi=120)
        plt.close(fig)
    metrics = pd.DataFrame(records)
    plot_names = {"centroid_error_distribution": "centroid_error_um", "dice_distribution": "dice_3d", "iou_distribution": "iou_3d", "volume_ratio_distribution": "volume_ratio_rescue_to_truth", "candidate_count_distribution": "n_candidates"}
    for name, column in plot_names.items():
        fig, axis = plt.subplots(figsize=(5, 3))
        values = pd.to_numeric(metrics.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        if len(values):
            axis.hist(values, bins=min(30, max(5, len(values) // 10)), color="#2878b5")
            axis.set_xlabel(column)
        else:
            axis.text(0.5, 0.5, "no finite values", ha="center", va="center")
        axis.set_title(name.replace("_", " "))
        fig.tight_layout(); fig.savefig(plot_dir / f"{name}.png", dpi=120); plt.close(fig)
    # These are descriptive only; state variables never enter ranking.
    fig, axis = plt.subplots(figsize=(7, 3))
    if not metrics.empty and "target_session" in metrics and "n_candidates" in metrics:
        grouped = metrics.assign(candidate_generated=pd.to_numeric(metrics["n_candidates"], errors="coerce").gt(0)).groupby("target_session", sort=False)["candidate_generated"].mean()
        axis.bar(np.arange(len(grouped)), grouped.to_numpy(), color="#4c9f70"); axis.set_xticks(np.arange(len(grouped))); axis.set_xticklabels(grouped.index, rotation=70, ha="right", fontsize=7); axis.set_ylim(0, 1)
    else:
        axis.text(0.5, 0.5, "no cases", ha="center", va="center")
    axis.set_title("rescue rate by session"); fig.tight_layout(); fig.savefig(plot_dir / "rescue_rate_by_session.png", dpi=120); plt.close(fig)
    for name, xcol, ycol in (("rescue_rate_by_local_crowding", "n_candidates", "dice_3d"), ("rescue_quality_vs_prediction_error", "prediction_error_um", "dice_3d"), ("rescue_quality_vs_depth", "truth_volume_voxels", "dice_3d"), ("measurement_bias", "green_absolute_difference", "red_absolute_difference")):
        fig, axis = plt.subplots(figsize=(5, 3))
        x = pd.to_numeric(metrics.get(xcol, pd.Series(dtype=float)), errors="coerce")
        y = pd.to_numeric(metrics.get(ycol, pd.Series(dtype=float)), errors="coerce")
        valid = x.notna() & y.notna()
        if valid.any(): axis.scatter(x[valid], y[valid], s=8, alpha=0.45)
        else: axis.text(0.5, 0.5, "no finite values", ha="center", va="center")
        axis.set_xlabel(xcol); axis.set_ylabel(ycol); axis.set_title(name.replace("_", " ")); fig.tight_layout(); fig.savefig(plot_dir / f"{name}.png", dpi=120); plt.close(fig)
    return panel_dir, plot_dir


def _summary(records: list[dict[str, Any]], candidates: list[dict[str, Any]], mode: str, context: RunContext, seed: int) -> dict[str, Any]:
    table = pd.DataFrame(records)
    summary: dict[str, Any] = {"mode": mode, "mouse": context.run_dir.parts[-5] if len(context.run_dir.parts) >= 5 else "unknown", "run": context.run_dir.name, "n_eligible": len(records), "n_attempted": len(records), "n_with_candidate": int(table.get("n_candidates", pd.Series(dtype=float)).gt(0).sum()) if not table.empty else 0, "n_multiple_candidates": int(table.get("status", pd.Series(dtype=str)).eq("multiple_candidates").sum()) if not table.empty else 0, "n_no_candidate": int(table.get("status", pd.Series(dtype=str)).eq("no_segmentation_object").sum()) if not table.empty else 0, "status_counts": table.get("status", pd.Series(dtype=str)).value_counts().to_dict() if not table.empty else {}, "random_seed": seed, "proposal_only": True, "segmentation_model": SEGMENTER_NAME, "canonical_outputs_modified": False, "state_variables_used_for_identity": False}
    if mode == "synthetic_benchmark":
        for column in ("selected_candidate_is_correct", "centroid_error_um", "dice_3d", "iou_3d", "volume_ratio_rescue_to_truth", "wrong_neighbor"):
            values = table[column] if column in table else pd.Series(dtype=float)
            numeric = pd.to_numeric(values, errors="coerce")
            if column == "selected_candidate_is_correct": summary["correct_cell_rate"] = float(values.fillna(False).mean()) if len(values) else np.nan
            elif column == "centroid_error_um": summary["median_centroid_error_um"] = float(numeric.median()) if numeric.notna().any() else np.nan; summary["p90_centroid_error_um"] = float(numeric.quantile(0.9)) if numeric.notna().any() else np.nan
            elif column == "dice_3d": summary["median_dice"] = float(numeric.median()) if numeric.notna().any() else np.nan
            elif column == "iou_3d": summary["median_iou"] = float(numeric.median()) if numeric.notna().any() else np.nan
            elif column == "volume_ratio_rescue_to_truth": summary["median_volume_ratio"] = float(numeric.median()) if numeric.notna().any() else np.nan
            elif column == "wrong_neighbor": summary["wrong_neighbor_rate"] = float(values.fillna(False).mean()) if len(values) else np.nan
    return summary


def _write_report(output_dir: Path, context: RunContext, mode: str, summary: dict[str, Any], focused_note: str = "") -> Path:
    report = output_dir / OUTPUT_REPORT
    lines = [f"# Phase D1 Local Segmentation Rescue Report (2026-09-15)", "", "## STARTING/ENDING COMMIT", f"- Canonical run provenance commit: `{context.git_commit}`", f"- Evaluator repository commit: `{_git_commit()}`", f"- Mode: `{mode}`", "", "## IMPLEMENTATION", f"- Module: `postprocessing/local_segmentation_rescue.py`", f"- Segmenter: `{SEGMENTER_NAME}`", f"- Spacing (ZYX um): `{context.spacing_zyx}` ({context.spacing_source})", f"- Crop: `{DEFAULT_CROP_SHAPE_ZYX}` voxels, configurable via CLI", "- Transform evidence: direct, composed, fallback, or unavailable; prediction and ranking are state-free.", "- Truth leakage: target label/mask is loaded only after candidate segmentation for synthetic evaluation.", "", "## BENCHMARK/PROPOSALS", f"- Eligible/attempted: `{summary.get('n_eligible', 0)}` / `{summary.get('n_attempted', 0)}`", f"- Candidate generated: `{summary.get('n_with_candidate', 0)}`; multiple: `{summary.get('n_multiple_candidates', 0)}`; no candidate: `{summary.get('n_no_candidate', 0)}`", f"- Status counts: `{summary.get('status_counts', {})}`"]
    if mode == "synthetic_benchmark":
        lines += [f"- Correct-cell rate: `{summary.get('correct_cell_rate', np.nan)}`", f"- Median/P90 centroid error (um): `{summary.get('median_centroid_error_um', np.nan)}` / `{summary.get('p90_centroid_error_um', np.nan)}`", f"- Median Dice/IoU: `{summary.get('median_dice', np.nan)}` / `{summary.get('median_iou', np.nan)}`", f"- Median volume ratio: `{summary.get('median_volume_ratio', np.nan)}`"]
    lines += ["", "## MEASUREMENT BIAS", "- Green/Red/ratio are evaluation-only; ECLIPSE fields remain unavailable unless supplied by an existing extraction table.", f"- Measurement-bias table: `{summary.get('output_paths', {}).get('measurement_bias', '')}`", "", "## REVIEW ARTIFACTS", f"- Panels: `{summary.get('review_panel_dir', '')}`", f"- Summary plots: `{summary.get('summary_plot_dir', '')}`", "", "## HARD CONSTRAINTS", "- Canonical masks/tracks/track IDs changed: **NO**", "- Matching thresholds or ECLIPSE calculation changed: **NO**", "- ECLIPSE/Green/Red/state used for identity: **NO**", "- Rescued masks or measurements written to primary extraction: **NO**", "", "## CONCLUSION", "- Production acceptance threshold: **not defined in Phase D1**", "- Scientific promise: **UNCERTAIN pending benchmark distributions and review**", "- Next step: inspect review PNGs and benchmark distributions before any production gate.", "", focused_note]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def evaluate(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "synthetic_benchmark",
    sample_size: int | None = 1000,
    seed: int = 20260915,
    crop_shape_zyx: Iterable[int] = DEFAULT_CROP_SHAPE_ZYX,
    threshold_percentile: float = 85.0,
    min_voxels: int = 20,
) -> dict[str, Any]:
    """Run a proposal-only benchmark and persist all Phase D1 artifacts."""

    if mode not in {"synthetic_benchmark", "real_proposals"}:
        raise ValueError("mode must be synthetic_benchmark or real_proposals")
    context = _load_run_context(run_dir)
    root = Path(output_dir).expanduser().resolve()
    forbidden = {"matching", "extraction", "segmentation", "cellposeSAM_masks"}
    if root == context.run_dir or any(part in forbidden for part in root.relative_to(context.run_dir).parts) if root.is_relative_to(context.run_dir) else False:
        raise ValueError("output_dir must be a validation directory, not a canonical matching/extraction/segmentation path")
    root.mkdir(parents=True, exist_ok=True)
    crop_shape = tuple(int(v) for v in crop_shape_zyx)
    if len(crop_shape) != 3:
        raise ValueError("crop_shape_zyx must contain three values")
    lookup = _feature_lookup(context)
    graph = TransformGraph(context.transforms)
    cases = _eligible_synthetic_cases(context) if mode == "synthetic_benchmark" else _find_real_cases(context, sample_size, seed)
    cases = _sample_cases(cases, sample_size, seed) if mode == "synthetic_benchmark" else cases
    records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    image_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[str, np.ndarray] = {}
    for case in cases:
        case = {**case, "random_seed": int(seed)}
        record, candidates, selected, arrays = _run_case(context, case, synthetic=mode == "synthetic_benchmark", crop_shape_zyx=crop_shape, threshold_percentile=threshold_percentile, min_voxels=min_voxels, lookup=lookup, graph=graph, image_cache=image_cache, mask_cache=mask_cache)
        records.append(record)
        candidate_records.extend(candidates)
        # Keep only a small review sample in memory; the full benchmark stays
        # in compact CSV rows rather than retaining every 3-D crop.
        if arrays and len(artifacts) < 12:
            artifacts.append((record, selected, arrays))
    cases_path = root / ("synthetic_hide_rescue_cases.csv" if mode == "synthetic_benchmark" else "local_rescue_real_cases.csv")
    candidates_path = root / ("synthetic_hide_rescue_candidates.csv" if mode == "synthetic_benchmark" else "local_rescue_real_candidates.csv")
    metrics_path = root / "synthetic_hide_rescue_metrics.csv"
    summary_path = root / ("synthetic_hide_rescue_summary.json" if mode == "synthetic_benchmark" else "local_rescue_real_summary.json")
    pd.DataFrame(records).to_csv(cases_path, index=False)
    pd.DataFrame(candidate_records).to_csv(candidates_path, index=False)
    if mode == "synthetic_benchmark":
        pd.DataFrame(records).to_csv(metrics_path, index=False)
        bias_columns = [column for column in pd.DataFrame(records).columns if any(token in column for token in ("green_", "red_", "ratio_", "eclipse_"))]
        pd.DataFrame(records, columns=["track_uid", "target_session", *bias_columns]).to_csv(root / "synthetic_hide_rescue_measurement_bias.csv", index=False)
    summary = _summary(records, candidate_records, mode, context, seed)
    summary.update({"output_dir": str(root), "output_paths": {"cases": str(cases_path), "candidates": str(candidates_path), "metrics": str(metrics_path) if mode == "synthetic_benchmark" else None, "measurement_bias": str(root / "synthetic_hide_rescue_measurement_bias.csv") if mode == "synthetic_benchmark" else None, "summary": str(summary_path)}, "crop_shape_zyx": list(crop_shape), "threshold_percentile": threshold_percentile, "min_voxels": min_voxels})
    summary_path.write_text(json.dumps(summary, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n", encoding="utf-8")
    panel_dir, plot_dir = _write_pngs(root, records, artifacts, synthetic=mode == "synthetic_benchmark")
    summary["review_panel_dir"] = str(panel_dir)
    summary["summary_plot_dir"] = str(plot_dir)
    summary_path.write_text(json.dumps(summary, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n", encoding="utf-8")
    _write_report(root, context, mode, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("synthetic_benchmark", "real_proposals"), default="synthetic_benchmark")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--crop-shape-zyx", type=int, nargs=3, default=DEFAULT_CROP_SHAPE_ZYX)
    parser.add_argument("--threshold-percentile", type=float, default=85.0)
    parser.add_argument("--min-voxels", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate(args.run_dir, args.output_dir, mode=args.mode, sample_size=args.sample_size, seed=args.seed, crop_shape_zyx=args.crop_shape_zyx, threshold_percentile=args.threshold_percentile, min_voxels=args.min_voxels)
    print(json.dumps(summary, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
