#!/usr/bin/env python3
"""Read-only, registration-based Fucci tracking-dropout audit.

The audit compares graph observations with independent mask overlap in the
registered image frame. It reads production derivatives and unfiltered QC
masks, and writes only to the explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "core", ROOT / "matching", ROOT / "postprocessing"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from project_config import load_project_config, is_relative_to  # noqa: E402
from endpoint_evaluator import invert_restricted_transform  # noqa: E402
from image_registration import load_image_transform, restricted_transform, warp_volume  # noqa: E402
from fucci_color_state import fit_linear_modal_backbone  # noqa: E402


SCHEMA_VERSION = "fucci_tracking_dropout_audit_v1"
Z_SCALE = 0.16202241397736572
PRIMARY_SOURCE_OVERLAP = 0.15
PRIMARY_DICE = 0.10
STRICT_SOURCE_OVERLAP = 0.35
STRICT_DICE = 0.25
MIN_SIZE_VOXELS = 100
PRIMARY_OUTCOMES = [
    "TRACKED_OK",
    "TARGET_PRESENT_GRAPH_IDENTITY_FAILURE",
    "MIN_SIZE_FILTER_DROPOUT",
    "FRAGMENTATION_OR_MERGE_AMBIGUITY",
    "NO_MASK_BUT_RED_SIGNAL_REMAINS",
    "REPORTER_OBJECT_DISAPPEARANCE",
    "NOT_ASSESSABLE",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _date_session_map(features: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for session_id in features["session_id"].astype(str).unique():
        tail = session_id.rsplit("_", 1)[-1]
        if len(tail) == 8 and tail.isdigit():
            result[tail] = session_id
    return result


def discover_run(derivatives_root: Path, mouse_id: str, laser_nm: int,
                 start: str, next_session: str, third: str) -> Path:
    """Find an existing run with direct transforms for the requested pairs."""
    runs_root = derivatives_root / mouse_id / "longitudinal" / str(laser_nm) / "runs"
    wanted = {(start, next_session), (start, third)}
    candidates: list[tuple[int, Path]] = []
    for transform_path in runs_root.glob("*/matching/pairwise_transforms.csv"):
        try:
            table = _read_csv(transform_path)
        except Exception:
            continue
        pairs = set(zip(table.get("day_a", pd.Series(dtype=str)).astype(str).str[-8:],
                        table.get("day_b", pd.Series(dtype=str)).astype(str).str[-8:]))
        if wanted.issubset(pairs):
            # Prefer the smallest matching run, reducing irrelevant graph context.
            candidates.append((len(table), transform_path.parents[1]))
    if not candidates:
        raise FileNotFoundError(
            f"No saved matching run contains direct transforms for {start}->{next_session} "
            f"and {start}->{third} under {runs_root}"
        )
    return sorted(candidates, key=lambda item: (item[0], str(item[1])))[0][1]


def _voxels_by_label(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return voxel counts and z/y/x centroids without a full-volume coordinate grid."""
    max_label = int(np.max(mask))
    counts = np.zeros(max_label + 1, dtype=np.int64)
    sums_z = np.zeros(max_label + 1, dtype=np.float64)
    sums_y = np.zeros(max_label + 1, dtype=np.float64)
    sums_x = np.zeros(max_label + 1, dtype=np.float64)
    height, width = mask.shape[1:]
    y_weights = np.repeat(np.arange(height, dtype=np.float64), width)
    x_weights = np.tile(np.arange(width, dtype=np.float64), height)
    for z, plane in enumerate(mask):
        flat = plane.ravel()
        counts += np.bincount(flat, minlength=max_label + 1)
        sums_z += z * np.bincount(flat, minlength=max_label + 1)
        sums_y += np.bincount(flat, weights=y_weights, minlength=max_label + 1)
        sums_x += np.bincount(flat, weights=x_weights, minlength=max_label + 1)
    centroids = np.full((max_label + 1, 3), np.nan, dtype=float)
    valid = counts > 0
    centroids[valid, 0] = sums_z[valid] / counts[valid]
    centroids[valid, 1] = sums_y[valid] / counts[valid]
    centroids[valid, 2] = sums_x[valid] / counts[valid]
    return counts, centroids, np.asarray([max_label], dtype=np.int64)


def _label_boxes(mask: np.ndarray) -> list[tuple[slice, slice, slice] | None]:
    boxes = ndi.find_objects(mask)
    return list(boxes)


def _source_coordinates(mask: np.ndarray, label: int,
                        boxes: list[tuple[slice, slice, slice] | None]) -> np.ndarray:
    box = boxes[label - 1] if 0 < label <= len(boxes) else None
    if box is None:
        return np.empty((0, 3), dtype=np.int32)
    local = np.argwhere(mask[box] == label)
    if not len(local):
        return np.empty((0, 3), dtype=np.int32)
    return local + np.asarray([axis.start for axis in box], dtype=np.int32)


def propagate_source_roi(coords: np.ndarray, source_to_target: Any,
                         target_shape: tuple[int, int, int]) -> tuple[np.ndarray, float]:
    """Forward-map source voxel centers with the saved restricted transform."""
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.int32), 0.0
    transformed = np.rint(source_to_target.apply(coords.astype(float))).astype(np.int64)
    # Count transformed unique positions before clipping to quantify FoV loss.
    unique = np.unique(transformed, axis=0)
    inside = np.ones(len(unique), dtype=bool)
    for axis, size in enumerate(target_shape):
        inside &= (unique[:, axis] >= 0) & (unique[:, axis] < size)
    kept = unique[inside].astype(np.int32)
    coverage = float(len(kept) / len(unique)) if len(unique) else 0.0
    return kept, coverage


def overlap_candidates(propagated_coords: np.ndarray, target_labels: np.ndarray,
                       counts: np.ndarray, centroids: np.ndarray,
                       spacing_zyx: tuple[float, float, float],
                       boxes: list[tuple[slice, slice, slice] | None] | None = None,
                       *, source_overlap_min: float = PRIMARY_SOURCE_OVERLAP,
                       dice_min: float = PRIMARY_DICE) -> list[dict[str, Any]]:
    """Measure all nonzero target components intersecting the propagated mask."""
    if len(propagated_coords) == 0:
        return []
    values = target_labels[tuple(propagated_coords.T)]
    labels, intersections = np.unique(values[values > 0], return_counts=True)
    rows: list[dict[str, Any]] = []
    source_n = int(len(propagated_coords))
    source_centroid = propagated_coords.mean(axis=0)
    spacing = np.asarray(spacing_zyx, dtype=float)
    for label, intersection in zip(labels.astype(int), intersections.astype(int)):
        target_n = int(counts[label]) if label < len(counts) else 0
        if target_n <= 0:
            continue
        source_fraction = float(intersection / source_n)
        target_fraction = float(intersection / target_n)
        dice = float(2 * intersection / (source_n + target_n))
        centroid = centroids[label]
        displacement = (centroid - source_centroid) * spacing
        box = boxes[label - 1] if boxes is not None and 0 < label <= len(boxes) else None
        touches_z = bool(box is not None and (box[0].start == 0 or box[0].stop == target_labels.shape[0]))
        touches_xy = bool(box is not None and (box[1].start == 0 or box[1].stop == target_labels.shape[1]
                                                or box[2].start == 0 or box[2].stop == target_labels.shape[2]))
        rows.append({
            "label": int(label), "target_voxels": target_n,
            "intersection_voxels": int(intersection),
            "source_overlap_fraction": source_fraction,
            "target_overlap_fraction": target_fraction,
            "dice": dice,
            "centroid_displacement_pixels": float(np.linalg.norm(centroid - source_centroid)),
            "centroid_displacement_um": float(np.linalg.norm(displacement)),
            "touches_z_boundary": touches_z, "touches_xy_boundary": touches_xy,
            "plausible": bool(source_fraction >= source_overlap_min or
                               (dice >= dice_min and target_fraction >= 0.10)),
            "strict_plausible": bool(source_fraction >= STRICT_SOURCE_OVERLAP or
                                      (dice >= STRICT_DICE and target_fraction >= 0.15)),
        })
    return sorted(rows, key=lambda row: (row["source_overlap_fraction"], row["dice"]), reverse=True)


def _boundary(coords: np.ndarray, shape: tuple[int, int, int], *,
              z_margin: int = 2, xy_margin: int = 3) -> bool:
    if len(coords) == 0:
        return True
    return bool(
        coords[:, 0].min() <= z_margin or coords[:, 0].max() >= shape[0] - z_margin - 1
        or coords[:, 1].min() <= xy_margin or coords[:, 1].max() >= shape[1] - xy_margin - 1
        or coords[:, 2].min() <= xy_margin or coords[:, 2].max() >= shape[2] - xy_margin - 1
    )


def _local_ncc(source_red: np.ndarray, target_registered_red: np.ndarray,
               source_coords: np.ndarray, source_shape: tuple[int, int, int]) -> float:
    """High-pass local NCC outside the source ROI, used as an observability flag."""
    if not len(source_coords):
        return float("nan")
    center = np.round(source_coords.mean(axis=0)).astype(int)
    z0, z1 = max(0, center[0] - 3), min(source_shape[0], center[0] + 4)
    y0, y1 = max(0, center[1] - 16), min(source_shape[1], center[1] + 17)
    x0, x1 = max(0, center[2] - 16), min(source_shape[2], center[2] + 17)
    src = source_red[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
    tgt = target_registered_red[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
    if src.size < 200 or tgt.shape != src.shape:
        return float("nan")
    src = src - ndi.gaussian_filter(src, sigma=(0.8, 3.0, 3.0))
    tgt = tgt - ndi.gaussian_filter(tgt, sigma=(0.8, 3.0, 3.0))
    local = np.zeros(src.shape, dtype=bool)
    local_coords = source_coords - np.asarray([z0, y0, x0])
    valid = np.all((local_coords >= 0) & (local_coords < np.asarray(src.shape)), axis=1)
    local[tuple(local_coords[valid].T)] = True
    # Exclude the source object and a 2-pixel lateral buffer from landmark scoring.
    local = ndi.binary_dilation(local, structure=np.ones((1, 5, 5), dtype=bool))
    keep = ~local
    a, b = src[keep].astype(float), tgt[keep].astype(float)
    if len(a) < 100:
        return float("nan")
    a -= a.mean(); b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 1e-10 else float("nan")


def _local_measurement(image: np.ndarray, coords: np.ndarray,
                       occupied: np.ndarray) -> dict[str, float]:
    """Measure fixed-mask and expanded signal plus a nearby non-ROI shell."""
    blank = {
        "mean": np.nan, "median": np.nan, "integrated": np.nan, "q90": np.nan,
        "background_mean": np.nan, "background_median": np.nan,
        "background_mad": np.nan, "net_q90": np.nan, "net_integrated": np.nan,
        "expanded_net_q90": np.nan,
    }
    if len(coords) == 0:
        return blank
    # Keep a compact local box; the 2D ring is approximately 1.4-2.8 um wide.
    lo = np.maximum(coords.min(axis=0) - np.asarray([0, 5, 5]), 0)
    hi = np.minimum(coords.max(axis=0) + np.asarray([1, 6, 6]), image.shape)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    local_shape = tuple(int(b - a) for a, b in zip(lo, hi))
    mask = np.zeros(local_shape, dtype=bool)
    local_coords = coords - lo
    inside = np.all((local_coords >= 0) & (local_coords < np.asarray(local_shape)), axis=1)
    mask[tuple(local_coords[inside].T)] = True
    # XY dilation only: one z-plane step is 5 um and is not a thin isotropic shell.
    structure = np.ones((1, 5, 5), dtype=bool)
    expanded = ndi.binary_dilation(mask, structure=structure)
    occupied_local = occupied[slices] > 0
    shell = expanded & ~mask & ~occupied_local
    values = image[slices][mask].astype(float)
    expanded_values = image[slices][expanded].astype(float)
    background = image[slices][shell].astype(float)
    if not len(values):
        return blank
    if len(background):
        bg_median = float(np.median(background))
        bg_mean = float(np.mean(background))
        bg_mad = float(1.4826 * np.median(np.abs(background - bg_median)))
    else:
        bg_median = bg_mean = bg_mad = np.nan
    q90 = float(np.quantile(values, 0.90))
    expanded_q90 = float(np.quantile(expanded_values, 0.90)) if len(expanded_values) else q90
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "integrated": float(np.sum(values)), "q90": q90,
        "background_mean": bg_mean, "background_median": bg_median,
        "background_mad": bg_mad,
        "net_q90": float(q90 - bg_median) if np.isfinite(bg_median) else np.nan,
        "net_integrated": float(np.sum(values - bg_median)) if np.isfinite(bg_median) else np.nan,
        "expanded_net_q90": float(expanded_q90 - bg_median) if np.isfinite(bg_median) else np.nan,
    }


def _candidate_fields(candidates: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    plausible = [candidate for candidate in candidates if candidate["plausible"]]
    best = plausible[0] if plausible else (candidates[0] if candidates else {})
    result: dict[str, Any] = {
        f"{prefix}_candidate_count": len(candidates),
        f"{prefix}_plausible_count": len(plausible),
        f"{prefix}_best_label": best.get("label", np.nan),
        f"{prefix}_best_target_voxels": best.get("target_voxels", np.nan),
        f"{prefix}_best_intersection_voxels": best.get("intersection_voxels", np.nan),
        f"{prefix}_best_source_overlap": best.get("source_overlap_fraction", np.nan),
        f"{prefix}_best_target_overlap": best.get("target_overlap_fraction", np.nan),
        f"{prefix}_best_dice": best.get("dice", np.nan),
        f"{prefix}_best_displacement_px": best.get("centroid_displacement_pixels", np.nan),
        f"{prefix}_best_displacement_um": best.get("centroid_displacement_um", np.nan),
        f"{prefix}_best_touches_z_boundary": bool(best.get("touches_z_boundary", False)) if best else False,
        f"{prefix}_best_touches_xy_boundary": bool(best.get("touches_xy_boundary", False)) if best else False,
        f"{prefix}_best_strict_plausible": bool(best.get("strict_plausible", False)) if best else False,
        f"{prefix}_candidate_labels": ";".join(str(c["label"]) for c in plausible),
        f"{prefix}_candidate_boundary_flags_json": json.dumps([
            {"label": c["label"], "touches_z_boundary": c["touches_z_boundary"],
             "touches_xy_boundary": c["touches_xy_boundary"]} for c in candidates[:8]
        ], sort_keys=True),
        f"{prefix}_joint_source_overlap_fraction": float(sum(
            c["source_overlap_fraction"] for c in plausible if c["source_overlap_fraction"] >= 0.10
        )),
        f"{prefix}_multiple_components_jointly_explain": bool(
            sum(c["source_overlap_fraction"] >= 0.10 for c in plausible) >= 2
            and sum(c["source_overlap_fraction"] for c in plausible if c["source_overlap_fraction"] >= 0.10) >= 0.40
        ),
        f"{prefix}_candidate_metrics_json": json.dumps(candidates[:8], sort_keys=True),
    }
    return result


def _ambiguous(candidates: list[dict[str, Any]], propagated_n: int) -> tuple[bool, str]:
    plausible = [c for c in candidates if c["plausible"]]
    fragments = [c for c in plausible if c["source_overlap_fraction"] >= 0.10]
    joint = sum(c["source_overlap_fraction"] for c in fragments)
    if len(fragments) >= 2 and joint >= 0.40:
        return True, "multiple_components_jointly_cover_source"
    if plausible:
        top = plausible[0]
        if (top["source_overlap_fraction"] >= 0.35
                and top["target_overlap_fraction"] < 0.12
                and top["target_voxels"] >= max(3 * propagated_n, 300)):
            return True, "single_target_component_may_be_merge"
    return False, ""


def classify_primary_outcome(*, assessable: bool, canonical_candidates: list[dict[str, Any]],
                             unfiltered_candidates: list[dict[str, Any]],
                             graph_target_labels: set[int],
                             fragmentation_or_merge: bool,
                             red_signal_remains: bool) -> str:
    """Mutually exclusive precedence for one source ROI and one target date."""
    canonical = [c for c in canonical_candidates if c["plausible"]]
    unfiltered = [c for c in unfiltered_candidates if c["plausible"]]
    # A graph edge counts as normal only when independent overlap identifies one
    # unambiguous canonical candidate and the assigned label is that candidate.
    if not assessable:
        return "NOT_ASSESSABLE"
    if (len(canonical) == 1 and not fragmentation_or_merge
            and int(canonical[0]["label"]) in graph_target_labels):
        return "TRACKED_OK"
    if fragmentation_or_merge:
        return "FRAGMENTATION_OR_MERGE_AMBIGUITY"
    if canonical:
        return "TARGET_PRESENT_GRAPH_IDENTITY_FAILURE"
    if unfiltered and all(int(c["target_voxels"]) < MIN_SIZE_VOXELS for c in unfiltered):
        return "MIN_SIZE_FILTER_DROPOUT"
    if unfiltered:
        # An >=100 component in QC but no spatially corresponding canonical label
        # cannot be attributed to min_size; retain it as correspondence ambiguity.
        return "FRAGMENTATION_OR_MERGE_AMBIGUITY"
    if red_signal_remains:
        return "NO_MASK_BUT_RED_SIGNAL_REMAINS"
    return "REPORTER_OBJECT_DISAPPEARANCE"


def _primary_assessable(*, geometry_assessable: bool, image_assessable: bool,
                        canonical_candidates: list[dict[str, Any]],
                        unfiltered_candidates: list[dict[str, Any]]) -> bool:
    """Allow strong mask correspondence with sound geometry when local NCC is weak.

    NCC is needed for image-only/no-mask calls. A plausible target mask gives
    direct spatial evidence, but weak-NCC cases use the stricter overlap rule and
    still require adequate transform coverage and no source/target truncation.
    """
    mask_evidence = any(c.get("strict_plausible", False) for c in canonical_candidates)
    mask_evidence |= any(c.get("strict_plausible", False) for c in unfiltered_candidates)
    return bool(geometry_assessable and (image_assessable or mask_evidence))


def _assessment_candidates(candidates: list[dict[str, Any]], *, image_assessable: bool) -> list[dict[str, Any]]:
    """Select primary candidates; weak-NCC calls require strict mask overlap."""
    return [dict(candidate, plausible=bool(candidate.get("plausible", False)
                                           and (image_assessable or candidate.get("strict_plausible", False))))
            for candidate in candidates]


def _fit_historical_state(native: pd.DataFrame, source_session_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = native.loc[native.session_id.astype(str).eq(source_session_id)].copy()
    required = {"mask_label", "red_raw", "red", "green", "ratio_qc_pass"}
    if not required.issubset(source.columns):
        raise ValueError(f"Native ROI table missing historical state columns: {sorted(required - set(source.columns))}")
    raw_red = pd.to_numeric(source.red_raw, errors="coerce")
    cutoff = float(raw_red.quantile(0.50))
    source["source_red_bright"] = raw_red.ge(cutoff)
    corrected_green = pd.to_numeric(source.green, errors="coerce")
    corrected_red = pd.to_numeric(source.red, errors="coerce")
    good = (source.source_red_bright & corrected_green.le(5000)
            & np.isfinite(corrected_green) & corrected_green.gt(0)
            & np.isfinite(corrected_red) & corrected_red.gt(0))
    fit = source.loc[good].copy()
    if len(fit) < 50:
        raise ValueError("Historical modal state requires at least 50 valid top-half ROIs")
    modal_fit = fit_linear_modal_backbone(
        pd.to_numeric(fit.red).to_numpy(float),
        pd.to_numeric(fit.green).to_numpy(float),
    )
    source["predicted_green_historical"] = modal_fit.predict(corrected_red.to_numpy(float))
    source["historical_z"] = np.nan
    valid = (source.source_red_bright & corrected_green.le(5000)
             & np.isfinite(corrected_green) & corrected_green.gt(0)
             & np.isfinite(corrected_red) & corrected_red.gt(0)
             & np.isfinite(source.predicted_green_historical)
             & source.predicted_green_historical.gt(0))
    source.loc[valid, "historical_z"] = (
        np.log2(corrected_green[valid] / source.loc[valid, "predicted_green_historical"]) / Z_SCALE
    )
    source["historical_state"] = "OTHER_OR_INVALID"
    source.loc[~source.source_red_bright, "historical_state"] = "NOT_RED_BRIGHT"
    source.loc[valid & source.historical_z.lt(-3), "historical_state"] = "Low"
    source.loc[valid & source.historical_z.between(-1, 1, inclusive="both"), "historical_state"] = "Middle"
    source.loc[valid & source.historical_z.ge(-3) & source.historical_z.lt(-1), "historical_state"] = "LOW_TRANSITION"
    source.loc[valid & source.historical_z.gt(1) & source.historical_z.le(3), "historical_state"] = "HIGH_TRANSITION"
    source.loc[valid & source.historical_z.gt(3), "historical_state"] = "High"
    state_invalid = source.source_red_bright & corrected_green.gt(5000)
    source.loc[state_invalid, "historical_state"] = "GREEN_ARTIFACT_EXCLUDED"
    fit_record = {
        "session_id": source_session_id,
        "red_bright_raw_red_cutoff": cutoff,
        "n_native_rois": int(len(source)),
        "n_red_bright": int(source.source_red_bright.sum()),
        "n_fit_top_half_after_green_cutoff": int(len(fit)),
        "corrected_green_artifact_cutoff": 5000,
        "fit_method": "repository Gaussian-kernel modal regression (OLS initialized)",
        **modal_fit.to_dict(),
        "z_scale": Z_SCALE,
        "state_convention": "Low: Z < -3; core Middle: -1 <= Z <= 1; fit is top half raw Red after corrected Green >5000 exclusion",
    }
    return source, fit_record


def _signal_remains(row: dict[str, Any], threshold: float) -> bool:
    net = max(float(row.get("next_red_net_q90", np.nan)),
              float(row.get("next_red_expanded_net_q90", np.nan)))
    source = float(row.get("source_red_net_q90", np.nan))
    background_mad = float(row.get("next_red_background_mad", np.nan))
    if not np.isfinite(net) or not np.isfinite(source) or source <= 0:
        return False
    noise_floor = 3 * background_mad if np.isfinite(background_mad) else 0.0
    return bool(net > max(noise_floor, 0.0) and net / source >= threshold)


def _find_run_paths(run_dir: Path, dates: tuple[str, str, str]) -> dict[str, Path]:
    matching = run_dir / "matching"
    extraction = run_dir / "extraction"
    required = {
        "transforms": matching / "pairwise_transforms.csv",
        "tracks": matching / "tracks_graph.csv",
        "edges": matching / "track_edges_graph.csv",
        "features": matching / "roi_features.csv",
        "native": extraction / "matched_session_population_roi_metrics.csv",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Saved {name} input not found in run {run_dir}: {path}")
    registration_qc = matching / "pairwise_registration_qc.csv"
    if registration_qc.is_file():
        required["registration_qc"] = registration_qc
    return required


def _track_lookup(tracks: pd.DataFrame, source_session_id: str,
                  source_label: int) -> dict[str, Any]:
    column = f"{source_session_id}_roi"
    if column not in tracks:
        return {"track_uid": "", "next_label": np.nan, "third_label": np.nan}
    rows = tracks.loc[pd.to_numeric(tracks[column], errors="coerce").eq(source_label)]
    if rows.empty:
        return {"track_uid": "", "next_label": np.nan, "third_label": np.nan}
    # A graph partition should assign a source node to exactly one track. If it
    # does not, preserve the ambiguity rather than picking a convenient row.
    if len(rows) > 1:
        return {"track_uid": "AMBIGUOUS_SOURCE_TRACK", "next_label": np.nan,
                "third_label": np.nan}
    row = rows.iloc[0]
    next_col = f"{row.get('_audit_next_session_id', '')}_roi"
    third_col = f"{row.get('_audit_third_session_id', '')}_roi"
    return {"track_uid": str(row.get("track_uid", "")),
            "next_label": row.get(next_col, np.nan),
            "third_label": row.get(third_col, np.nan),
            "track_row": row}


def _direct_graph_maps(edges: pd.DataFrame, source_id: str, next_id: str,
                       third_id: str) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Build source->target observations only from accepted graph edges.

    The primary Oct 5 identity decision never requires Oct 2. It uses the direct
    Oct 5->6 accepted edge, and Oct 5->7 gap edges plus accepted Oct 6->7 edges.
    """
    accepted = edges.loc[edges.accepted_for_track.fillna(False).astype(bool)].copy()
    def pair_map(left: str, right: str) -> dict[int, set[int]]:
        subset = accepted.loc[accepted.day_a.astype(str).eq(left) & accepted.day_b.astype(str).eq(right)]
        result: dict[int, set[int]] = {}
        for row in subset.itertuples(index=False):
            result.setdefault(int(row.label_a), set()).add(int(row.label_b))
        return result
    next_map = pair_map(source_id, next_id)
    direct_third = pair_map(source_id, third_id)
    next_to_third = pair_map(next_id, third_id)
    third_map: dict[int, set[int]] = {label: set(labels) for label, labels in direct_third.items()}
    for source_label, next_labels in next_map.items():
        for label in next_labels:
            third_map.setdefault(source_label, set()).update(next_to_third.get(label, set()))
    return next_map, third_map


def _load_transform(transforms: pd.DataFrame, source_session_id: str,
                    target_session_id: str,
                    registration_qc: pd.DataFrame | None = None) -> tuple[pd.Series, Any, Any]:
    rows = transforms.loc[transforms.day_a.astype(str).eq(source_session_id)
                          & transforms.day_b.astype(str).eq(target_session_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one stored transform {source_session_id}->{target_session_id}; found {len(rows)}")
    stored = rows.iloc[0].copy()
    stage = str(stored.get("method", "legacy"))
    qc_rows = pd.DataFrame() if registration_qc is None else registration_qc.loc[
        registration_qc.day_a.astype(str).eq(source_session_id)
        & registration_qc.day_b.astype(str).eq(target_session_id)
    ]
    if len(qc_rows) > 1:
        raise ValueError(f"Expected at most one registration QC row for {source_session_id}->{target_session_id}")
    if len(qc_rows) == 1:
        qc = qc_rows.iloc[0]
        selected_stage = qc.get("selected_registration_stage")
        if pd.notna(selected_stage) and str(selected_stage).strip():
            stage = str(selected_stage)
        for field in ("requested_registration_mode", "selected_registration_stage", "smoothing_um",
                      "fallback_used", "fallback_reason", "transform_path", "local_qc_pass", "affine_qc_pass"):
            if field in qc.index:
                stored[f"registration_{field}"] = qc[field]
    if stage in {"image_affine_local", "image_affine"}:
        transform_path = str(stored.get("registration_transform_path", "")).strip()
        if not transform_path or not Path(transform_path).is_file():
            raise ValueError(f"selected {stage} transform artifact is missing for {source_session_id}->{target_session_id}")
        image_transform = load_image_transform(transform_path)
        if image_transform.method != stage:
            raise ValueError(f"selected stage {stage} does not match saved transform method {image_transform.method}")
        if (stage == "image_affine_local") != (image_transform.flow is not None):
            raise ValueError(f"selected stage {stage} has an incompatible saved displacement field")

        class _ForwardImageTransform:
            def __init__(self, image):
                self.image = image

            def apply(self, points):
                return self.image.inverse(points)

        return stored, _ForwardImageTransform(image_transform), image_transform
    source_to_target_row = invert_restricted_transform(stored)
    source_to_target = restricted_transform(source_to_target_row)
    target_to_source = restricted_transform(stored)
    return stored, source_to_target, target_to_source


def _image_and_mask_metrics(image: np.ndarray, coords: np.ndarray,
                            occupied: np.ndarray, prefix: str) -> dict[str, float]:
    channels: dict[str, float] = {}
    for metric, value in _local_measurement(image, coords, occupied).items():
        channels[f"{prefix}_{metric}"] = value
    return channels


def _edge_flags(coords: np.ndarray, shape: tuple[int, int, int]) -> tuple[bool, bool]:
    if not len(coords):
        return True, True
    return (_boundary(coords, shape, z_margin=2, xy_margin=3),
            bool(coords[:, 0].min() <= 1 or coords[:, 0].max() >= shape[0] - 2))


def _roi_neighbor_density(features: pd.DataFrame, source_session_id: str,
                          spacing: tuple[float, float, float]) -> dict[int, float]:
    group = features.loc[features.session_id.astype(str).eq(source_session_id)].copy()
    coords = group[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(float).copy()
    coords *= np.asarray(spacing)
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=30.0)
    return {int(label): float(max(0, len(indexes) - 1) / (4 / 3 * np.pi * 30.0**3))
            for label, indexes in zip(group.label.astype(int), neighbors)}


def _calibrate_signal_threshold(rows: pd.DataFrame) -> tuple[float, float, int]:
    controls = rows.loc[
        rows.historical_state.eq("Middle")
        & rows.next_canonical_plausible
        & rows.next_graph_observed
        & rows.next_image_assessable
    ].copy()
    numerator = pd.to_numeric(controls.next_red_net_q90, errors="coerce")
    denominator = pd.to_numeric(controls.source_red_net_q90, errors="coerce")
    ratios = (numerator / denominator).replace([np.inf, -np.inf], np.nan).dropna()
    ratios = ratios.loc[ratios.ge(0)]
    if len(ratios) >= 20:
        return max(0.20, float(ratios.quantile(0.05))), max(0.10, float(ratios.quantile(0.10))), len(ratios)
    # A conservative minimum retention rule is used when too few stable controls
    # exist to estimate a tail threshold reliably.
    return 0.20, 0.15, len(ratios)


def _match_controls(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = rows.loc[rows.historical_state.isin(["Low", "Middle"]) & rows.next_assessable].copy()
    if cohort.empty:
        return pd.DataFrame(), pd.DataFrame()
    low = cohort.loc[cohort.historical_state.eq("Low")].copy()
    middle = cohort.loc[cohort.historical_state.eq("Middle")].copy()
    columns = ["log_source_voxels", "log_corrected_red", "source_centroid_z_um", "source_local_density_30um"]
    for frame in (low, middle):
        frame["log_source_voxels"] = np.log1p(pd.to_numeric(frame.source_voxel_count, errors="coerce"))
        frame["log_corrected_red"] = np.log1p(pd.to_numeric(frame.source_red_corrected, errors="coerce").clip(lower=0))
    if low.empty or middle.empty:
        return pd.DataFrame(), pd.DataFrame()
    joined = pd.concat([low[columns], middle[columns]], ignore_index=True)
    scale = joined.std(ddof=0).replace(0, 1).fillna(1)
    center = joined.median().fillna(0)
    x_low = ((low[columns].fillna(center) - center) / scale).to_numpy(float)
    x_middle = ((middle[columns].fillna(center) - center) / scale).to_numpy(float)
    cost = np.sqrt(((x_low[:, None, :] - x_middle[None, :, :]) ** 2).mean(axis=2))
    boundary_mismatch = (low.source_boundary_flag.astype(bool).to_numpy()[:, None]
                         != middle.source_boundary_flag.astype(bool).to_numpy()[None, :])
    cost += boundary_mismatch * 2.0
    low_idx, middle_idx = linear_sum_assignment(cost)
    match_rows: list[dict[str, Any]] = []
    for i, j in zip(low_idx, middle_idx):
        left, right = low.iloc[i], middle.iloc[j]
        match_rows.append({
            "low_source_label": int(left.source_label), "middle_source_label": int(right.source_label),
            "distance": float(cost[i, j]), "low_primary_outcome": left.next_primary_outcome,
            "middle_primary_outcome": right.next_primary_outcome,
            "low_track_id": left.source_track_id, "middle_track_id": right.source_track_id,
            "low_source_voxels": int(left.source_voxel_count),
            "middle_source_voxels": int(right.source_voxel_count),
            "low_red_corrected": float(left.source_red_corrected),
            "middle_red_corrected": float(right.source_red_corrected),
            "low_depth_um": float(left.source_centroid_z_um),
            "middle_depth_um": float(right.source_centroid_z_um),
        })
    pairs = pd.DataFrame(match_rows)
    summary_rows: list[dict[str, Any]] = []
    metrics = {
        "graph_not_observed": lambda frame: ~frame.next_graph_observed.astype(bool),
        "canonical_target_detected": lambda frame: frame.next_canonical_plausible.astype(bool),
        "unfiltered_target_detected": lambda frame: frame.next_unfiltered_plausible.astype(bool),
        "min_size_dropout": lambda frame: frame.next_primary_outcome.eq("MIN_SIZE_FILTER_DROPOUT"),
        "graph_identity_failure": lambda frame: frame.next_primary_outcome.eq("TARGET_PRESENT_GRAPH_IDENTITY_FAILURE"),
        "segmentation_failure_red_remains": lambda frame: frame.next_primary_outcome.eq("NO_MASK_BUT_RED_SIGNAL_REMAINS"),
        "reporter_object_disappearance": lambda frame: frame.next_primary_outcome.eq("REPORTER_OBJECT_DISAPPEARANCE"),
        "fragmentation_merge": lambda frame: frame.next_primary_outcome.eq("FRAGMENTATION_OR_MERGE_AMBIGUITY"),
        "not_assessable": lambda frame: frame.next_primary_outcome.eq("NOT_ASSESSABLE"),
    }
    low_pairs = low.set_index("source_label")
    middle_pairs = middle.set_index("source_label")
    for metric, predicate in metrics.items():
        lv = np.asarray([bool(predicate(low_pairs.loc[int(label):int(label)]).iloc[0])
                         for label in pairs.low_source_label], dtype=bool)
        mv = np.asarray([bool(predicate(middle_pairs.loc[int(label):int(label)]).iloc[0])
                         for label in pairs.middle_source_label], dtype=bool)
        summary_rows.append({
            "metric": metric, "matched_pairs": int(len(pairs)),
            "low_numerator": int(lv.sum()), "low_denominator": int(len(lv)),
            "low_fraction": float(lv.mean()) if len(lv) else np.nan,
            "middle_numerator": int(mv.sum()), "middle_denominator": int(len(mv)),
            "middle_fraction": float(mv.mean()) if len(mv) else np.nan,
            "paired_difference_low_minus_middle": float((lv.astype(float) - mv.astype(float)).mean()) if len(lv) else np.nan,
        })
    return pd.DataFrame(summary_rows), pairs


def _fit_ridge_logistic(x: np.ndarray, y: np.ndarray, penalty: float = 1.0) -> tuple[np.ndarray, bool]:
    x = np.column_stack([np.ones(len(x)), x])
    def loss(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(x @ beta, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        value = float(np.sum(np.logaddexp(0, eta) - y * eta) + penalty * np.sum(beta[1:] ** 2) / 2)
        gradient = x.T @ (p - y)
        gradient[1:] += penalty * beta[1:]
        return value, gradient
    result = minimize(lambda beta: loss(beta), np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return result.x, bool(result.success)


def _adjusted_models(rows: pd.DataFrame) -> pd.DataFrame:
    base = rows.loc[rows.historical_state.isin(["Low", "Middle"])].copy()
    features = ["log_volume", "log_red", "depth", "boundary", "local_ncc"]
    base["log_volume"] = np.log1p(pd.to_numeric(base.source_voxel_count, errors="coerce"))
    base["log_red"] = np.log1p(pd.to_numeric(base.source_red_corrected, errors="coerce").clip(lower=0))
    base["depth"] = pd.to_numeric(base.source_centroid_z_um, errors="coerce")
    base["boundary"] = base.source_boundary_flag.astype(float)
    base["local_ncc"] = pd.to_numeric(base.next_local_registration_ncc, errors="coerce")
    base = base.dropna(subset=features)
    if len(base) < 20 or base.historical_state.nunique() < 2:
        return pd.DataFrame([{"model": "unavailable", "reason": "fewer than 20 complete Low/Middle rows", "n": int(len(base))}])
    outcomes = {
        "graph_dropout": ((~base.next_graph_observed.astype(bool)).astype(float).to_numpy(), base),
        "any_primary_nontracked": ((~base.next_primary_outcome.eq("TRACKED_OK")).astype(float).to_numpy(), base),
        "min_size_dropout": (base.next_primary_outcome.eq("MIN_SIZE_FILTER_DROPOUT").astype(float).to_numpy(), base),
        "segmentation_failure_red_remains": (base.next_primary_outcome.eq("NO_MASK_BUT_RED_SIGNAL_REMAINS").astype(float).to_numpy(), base),
        "reporter_object_disappearance": (base.next_primary_outcome.eq("REPORTER_OBJECT_DISAPPEARANCE").astype(float).to_numpy(), base),
    }
    result_rows: list[dict[str, Any]] = []
    for name, (all_y, all_use) in outcomes.items():
        # Graph observation is known even when local geometry is not assessable,
        # so adjust that outcome across the full Low/Middle source cohort. The
        # image/mask mechanism models are conditional on assessable geometry.
        use = all_use if name == "graph_dropout" else all_use.loc[all_use.next_assessable].copy()
        y = all_y if name == "graph_dropout" else all_y[all_use.next_assessable.to_numpy()]
        if len(use) < 20 or use.historical_state.nunique() < 2:
            continue
        center = use[features].median()
        scale = use[features].std(ddof=0).replace(0, 1)
        x_cov = ((use[features] - center) / scale).to_numpy(float)
        low = use.historical_state.eq("Low").astype(float).to_numpy()
        x = np.column_stack([low, x_cov])
        beta, success = _fit_ridge_logistic(x, y)
        beta_low = float(beta[1])
        center_x_mid = np.zeros(x.shape[1]); center_x_low = center_x_mid.copy(); center_x_low[0] = 1.0
        def sigmoid(v: float) -> float:
            return float(1 / (1 + math.exp(-max(-35.0, min(35.0, v)))))
        p_middle = sigmoid(float(beta[0] + center_x_mid @ beta[1:]))
        p_low = sigmoid(float(beta[0] + center_x_low @ beta[1:]))
        result_rows.append({
            "model": name, "n": int(len(y)), "events": int(y.sum()),
            "sample_scope": "all Low/Middle sources" if name == "graph_dropout" else "assessable Low/Middle sources",
            "low_log_odds_coefficient_ridge_1": beta_low,
            "low_adjusted_odds_ratio": float(math.exp(np.clip(beta_low, -20, 20))),
            "predicted_probability_middle_at_median_covariates": p_middle,
            "predicted_probability_low_at_median_covariates": p_low,
            "adjusted_probability_difference_low_minus_middle": p_low - p_middle,
            "converged": success,
            "covariates": "Low + log(volume) + log(corrected Red) + depth + source boundary + local Red NCC; standardized covariates; ridge penalty=1",
        })
    return pd.DataFrame(result_rows)


def _rate_summary(rows: pd.DataFrame) -> pd.DataFrame:
    metrics: dict[str, Any] = {
        "graph_observed_next": lambda g: g.next_graph_observed.astype(bool),
        "canonical_mask_target_detected": lambda g: g.next_canonical_plausible.astype(bool),
        "unfiltered_mask_target_detected": lambda g: g.next_unfiltered_plausible.astype(bool),
        "min_size_dropout": lambda g: g.next_primary_outcome.eq("MIN_SIZE_FILTER_DROPOUT"),
        "graph_identity_failure": lambda g: g.next_primary_outcome.eq("TARGET_PRESENT_GRAPH_IDENTITY_FAILURE"),
        "no_mask_red_signal_remains": lambda g: g.next_primary_outcome.eq("NO_MASK_BUT_RED_SIGNAL_REMAINS"),
        "reporter_object_disappearance": lambda g: g.next_primary_outcome.eq("REPORTER_OBJECT_DISAPPEARANCE"),
        "fragmentation_or_merge": lambda g: g.next_primary_outcome.eq("FRAGMENTATION_OR_MERGE_AMBIGUITY"),
        "not_assessable": lambda g: g.next_primary_outcome.eq("NOT_ASSESSABLE"),
    }
    result: list[dict[str, Any]] = []
    for state, group in rows.loc[rows.historical_state.isin(["Low", "Middle"])].groupby("historical_state"):
        graph_lost = ~group.next_graph_observed.astype(bool)
        n_graph_lost = int(graph_lost.sum())
        for metric, function in metrics.items():
            values = function(group).astype(bool)
            result.append({"state": state, "metric": metric, "numerator": int(values.sum()),
                           "denominator": int(len(values)), "fraction": float(values.mean()) if len(values) else np.nan,
                           "assessable_denominator": int(group.next_assessable.astype(bool).sum()),
                           "graph_unobserved_denominator": n_graph_lost,
                           "fraction_of_graph_unobserved": float(values.loc[graph_lost].mean()) if n_graph_lost else np.nan})
        for outcome in PRIMARY_OUTCOMES:
            values = group.next_primary_outcome.eq(outcome)
            canonical_n = int(group.next_canonical_plausible.astype(bool).sum())
            result.append({"state": state, "metric": f"primary_outcome:{outcome}",
                           "numerator": int(values.sum()), "denominator": int(len(group)),
                           "fraction": float(values.mean()) if len(group) else np.nan,
                           "assessable_denominator": int(group.next_assessable.astype(bool).sum()),
                           "graph_unobserved_denominator": n_graph_lost,
                           "fraction_of_graph_unobserved": (float(values.loc[graph_lost].sum() / n_graph_lost)
                                                            if n_graph_lost else np.nan),
                           "plausible_canonical_denominator": canonical_n,
                           "fraction_conditional_on_canonical": (float(values.sum() / canonical_n)
                                                                  if outcome == "TARGET_PRESENT_GRAPH_IDENTITY_FAILURE" and canonical_n else np.nan)})
    return pd.DataFrame(result).sort_values(["state", "metric"]).reset_index(drop=True)


def _starting_covariate_summary(rows: pd.DataFrame) -> pd.DataFrame:
    use = rows.loc[rows.historical_state.isin(["Low", "Middle"])].copy()
    measures = {
        "source_volume_voxels": "source_voxel_count",
        "source_volume_um3": "source_volume_um3",
        "corrected_red_mean": "source_red_corrected",
        "integrated_red_proxy_raw_sum": "source_red_integrated",
        "corrected_green_mean": "source_green_corrected",
        "depth_um": "source_centroid_z_um",
        "local_density_30um": "source_local_density_30um",
        "local_registration_ncc_oct6": "next_local_registration_ncc",
    }
    records = []
    for metric, column in measures.items():
        for state in ("Low", "Middle"):
            values = pd.to_numeric(use.loc[use.historical_state.eq(state), column], errors="coerce").dropna()
            records.append({"metric": metric, "state": state, "n": int(len(values)),
                            "median": float(values.median()) if len(values) else np.nan,
                            "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                            "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                            "mean": float(values.mean()) if len(values) else np.nan})
    for state in ("Low", "Middle"):
        group = use.loc[use.historical_state.eq(state)]
        records.append({"metric": "source_boundary_fraction", "state": state, "n": int(len(group)),
                        "median": float(group.source_boundary_flag.astype(bool).mean()),
                        "q25": np.nan, "q75": np.nan,
                        "mean": float(group.source_boundary_flag.astype(bool).mean())})
    return pd.DataFrame(records)


def _manual_review_shortlist(rows: pd.DataFrame, per_state_outcome: int = 3) -> pd.DataFrame:
    outcomes = [
        "REPORTER_OBJECT_DISAPPEARANCE", "MIN_SIZE_FILTER_DROPOUT",
        "TARGET_PRESENT_GRAPH_IDENTITY_FAILURE", "NO_MASK_BUT_RED_SIGNAL_REMAINS",
        "FRAGMENTATION_OR_MERGE_AMBIGUITY",
    ]
    selected = []
    for state in ("Low", "Middle"):
        for outcome in outcomes:
            group = rows.loc[rows.historical_state.eq(state)
                             & rows.next_primary_outcome.eq(outcome)
                             & rows.next_assessable].copy()
            group["review_priority"] = outcomes.index(outcome) + (0 if state == "Low" else 10)
            group = group.sort_values(["next_local_registration_ncc", "source_red_corrected"],
                                      ascending=[False, False], na_position="last")
            if not group.empty:
                selected.append(group.head(per_state_outcome))
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True).drop_duplicates("source_label", keep="first").sort_values(
        ["review_priority", "next_local_registration_ncc"], ascending=[True, False]
    ).reset_index(drop=True)


def _compact_case_table(frame: pd.DataFrame) -> pd.DataFrame:
    bulky = [column for column in frame.columns
             if column.endswith("_candidate_metrics_json") or column.endswith("_candidate_boundary_flags_json")]
    return frame.drop(columns=bulky, errors="ignore")


def _plot_outputs(rows: pd.DataFrame, matched: pd.DataFrame, models: pd.DataFrame, figures: Path,
                  case_pairs: pd.DataFrame, arrays: dict[str, np.ndarray],
                  date_ids: tuple[str, str, str], source_features: pd.DataFrame,
                  target_candidates: dict[tuple[int, str], tuple[list[dict[str, Any]], list[dict[str, Any]]]],
                  outlier_labels: set[int]) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    states = ["Low", "Middle"]
    order = PRIMARY_OUTCOMES
    fig, ax = plt.subplots(figsize=(12, 5.5))
    matrix = np.zeros((len(order), len(states)))
    for j, state in enumerate(states):
        group = rows.loc[rows.historical_state.eq(state)]
        denom = max(1, len(group))
        matrix[:, j] = [group.next_primary_outcome.eq(outcome).sum() / denom for outcome in order]
    bottom = np.zeros(2)
    colors = ["#27864b", "#c84b40", "#e0aa32", "#8b6dba", "#4287b8", "#7a3030", "#888888"]
    for i, outcome in enumerate(order):
        ax.bar(states, matrix[i], bottom=bottom, label=outcome.replace("_", " "), color=colors[i])
        bottom += matrix[i]
    ax.set_ylabel("Fraction of Red-bright source ROIs")
    ax.set_title(f"{date_ids[1]} primary outcome by {date_ids[0]} Fucci state")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout(); path = figures / "outcome_fractions_low_vs_middle.png"; fig.savefig(path, dpi=160); plt.close(fig); created.append(str(path))

    fig, (ax, model_ax) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    if not matched.empty:
        pivot = matched.set_index("metric")
        show = ["graph_not_observed", "canonical_target_detected", "unfiltered_target_detected",
                "min_size_dropout", "graph_identity_failure", "segmentation_failure_red_remains",
                "reporter_object_disappearance", "fragmentation_merge"]
        vals = [pivot.loc[x, "paired_difference_low_minus_middle"] if x in pivot.index else np.nan for x in show]
        ax.bar(np.arange(len(show)), vals, color=["#ba4141" if v > 0 else "#3d719e" for v in vals])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(show)), [x.replace("_", "\n") for x in show], rotation=0, fontsize=8)
        ax.set_ylabel("Matched paired rate difference (Low − Middle)")
        ax.set_title("Low vs Middle after 1:1 observability matching")
    else:
        ax.text(0.5, 0.5, "No complete Low/Middle matched set", ha="center", va="center")
        ax.axis("off")
    if "low_adjusted_odds_ratio" in models:
        show_models = models.loc[models.model.isin(["graph_dropout", "any_primary_nontracked",
                                                     "min_size_dropout", "segmentation_failure_red_remains",
                                                     "reporter_object_disappearance"])].copy()
        x_positions = np.arange(len(show_models))
        middle_p = show_models.predicted_probability_middle_at_median_covariates.to_numpy(float)
        low_p = show_models.predicted_probability_low_at_median_covariates.to_numpy(float)
        width = 0.36
        model_ax.bar(x_positions - width / 2, middle_p, width, label="Middle", color="#3d719e")
        model_ax.bar(x_positions + width / 2, low_p, width, label="Low", color="#ba4141")
        model_ax.set_xticks(x_positions, [str(x).replace("_", "\n") for x in show_models.model], fontsize=7)
        model_ax.set_ylabel("Predicted probability at median covariates")
        model_ax.set_title("Adjusted logistic sensitivity")
        model_ax.legend(fontsize=8)
    else:
        model_ax.text(0.5, 0.5, "Adjusted model unavailable", ha="center", va="center")
        model_ax.axis("off")
    fig.tight_layout(); path = figures / "matched_outcome_rate_differences.png"; fig.savefig(path, dpi=160); plt.close(fig); created.append(str(path))

    plot_rows = rows.loc[rows.historical_state.isin(states)].copy()
    fig, ax = plt.subplots(figsize=(13, 6))
    groups, labels = [], []
    for outcome in order:
        for state in states:
            values = pd.to_numeric(plot_rows.loc[plot_rows.historical_state.eq(state)
                                                  & plot_rows.next_primary_outcome.eq(outcome), "source_voxel_count"], errors="coerce").dropna()
            if len(values):
                groups.append(values); labels.append(f"{state}\n{outcome.replace('_', ' ')}")
    if groups:
        ax.boxplot(groups, showfliers=False, showmeans=True)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel(f"{date_ids[0]} source volume (voxels, log scale)")
    ax.set_title("Source ROI size by primary outcome and state")
    fig.tight_layout(); path = figures / "source_volume_by_outcome_state.png"; fig.savefig(path, dpi=160); plt.close(fig); created.append(str(path))

    dropout = rows.loc[rows.next_primary_outcome.isin(["MIN_SIZE_FILTER_DROPOUT", "FRAGMENTATION_OR_MERGE_AMBIGUITY"])]
    sizes = pd.to_numeric(dropout.next_unfiltered_best_target_voxels, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(sizes):
        ax.hist(sizes, bins=np.arange(0, max(110, int(sizes.max()) + 100), 25), color="#ba8f38", alpha=0.85)
    ax.axvline(MIN_SIZE_VOXELS, color="#a32323", linestyle="--", label="canonical min_size=100")
    ax.set_xlabel("Best plausible unfiltered target size (voxels)"); ax.set_ylabel("Source ROI count")
    ax.set_title("Unfiltered target sizes among graph dropouts"); ax.legend()
    fig.tight_layout(); path = figures / "unfiltered_target_size_dropouts.png"; fig.savefig(path, dpi=160); plt.close(fig); created.append(str(path))

    fig, ax = plt.subplots(figsize=(13, 6))
    signal_groups, signal_labels = [], []
    for outcome in order:
        values = pd.to_numeric(plot_rows.loc[plot_rows.next_primary_outcome.eq(outcome), "next_red_retention_ratio"], errors="coerce").dropna()
        if len(values):
            signal_groups.append(values.clip(-1, 3)); signal_labels.append(outcome.replace("_", "\n"))
    if signal_groups:
        ax.boxplot(signal_groups, showfliers=False)
        ax.set_xticks(np.arange(1, len(signal_labels) + 1), signal_labels, fontsize=7)
    ax.axhline(1.0, color="#333333", linestyle=":")
    ax.set_ylabel(f"Fixed-region Red net-q90 retention vs {date_ids[0]}")
    ax.set_title("Fixed propagated-region Red signal change")
    fig.tight_layout(); path = figures / "fixed_region_red_signal_by_outcome.png"; fig.savefig(path, dpi=160); plt.close(fig); created.append(str(path))

    dropouts = rows.loc[rows.next_primary_outcome.ne("TRACKED_OK")]
    fates = dropouts.oct7_fate.value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([str(x).replace("_", "\n") for x in fates.index], fates.values, color="#51799d")
    ax.set_ylabel(f"{date_ids[1]} dropout count"); ax.set_title(f"{date_ids[2]} fate of {date_ids[1]} graph dropouts")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); path = figures / "oct7_fate_of_oct6_dropouts.png"; fig.savefig(path, dpi=160); plt.close(fig); created.append(str(path))

    # Stratified visual validation, fixed per-channel limits shared across dates.
    # This is a review aid only; no image pixels are used by the classifier here.
    rng = np.random.default_rng(20261012)
    sample_rows: list[pd.Series] = []
    # Stable compact sampling per category, in outcome/state order.
    for outcome in [x for x in order if x != "NOT_ASSESSABLE"]:
        for state in states:
            group = rows.loc[rows.next_primary_outcome.eq(outcome) & rows.historical_state.eq(state)].sort_values("source_label")
            if len(group):
                take = min(2, len(group))
                if take < len(group):
                    selected = rng.choice(group.index.to_numpy(), take, replace=False)
                    group = group.loc[sorted(selected)]
                sample_rows.extend([row for _, row in group.iterrows()])
    if sample_rows:
        limit_rows = 3 * len(sample_rows)
        fig, axes = plt.subplots(len(sample_rows), 3, figsize=(13, max(4, 2.3 * len(sample_rows))), squeeze=False)
        # Percentile estimates are computed from deterministic strided pixel samples.
        image_arrays = [arrays[k] for k in ("source_red", "next_red", "third_red")]
        green_arrays = [arrays[k] for k in ("source_green", "next_green", "third_green")]
        red_samples = np.concatenate([a.ravel()[:: max(1, a.size // 200000)] for a in image_arrays])
        green_samples = np.concatenate([a.ravel()[:: max(1, a.size // 200000)] for a in green_arrays])
        red_limits = np.quantile(red_samples, [0.02, 0.995]); green_limits = np.quantile(green_samples, [0.02, 0.995])
        mask_arrays = [arrays[k] for k in ("source_mask", "next_mask", "third_mask")]
        unfiltered_arrays = [arrays[k] for k in ("source_unfiltered", "next_unfiltered", "third_unfiltered")]
        for row_index, item in enumerate(sample_rows):
            label = int(item.source_label)
            title = f"{item.historical_state} / {item.next_primary_outcome} / ROI {label}"
            for col, (date_id, session_id) in enumerate(zip(
                date_ids, (item["_source_session_id"], item["_next_session_id"], item["_third_session_id"])
            )):
                ax = axes[row_index, col]
                point = np.asarray([item.source_centroid_z, item.source_centroid_y, item.source_centroid_x], dtype=float)
                if col:
                    point = np.asarray(item[f"{('next' if col == 1 else 'third')}_transformed_centroid_zyx"], dtype=float)
                center = np.rint(point).astype(int)
                image_red = image_arrays[col]; image_green = green_arrays[col]
                z0, z1 = max(0, center[0] - 2), min(image_red.shape[0], center[0] + 3)
                y0, y1 = max(0, center[1] - 18), min(image_red.shape[1], center[1] + 19)
                x0, x1 = max(0, center[2] - 18), min(image_red.shape[2], center[2] + 19)
                r = np.max(image_red[z0:z1, y0:y1, x0:x1], axis=0)
                g = np.max(image_green[z0:z1, y0:y1, x0:x1], axis=0)
                rgb = np.zeros((*r.shape, 3), dtype=float)
                rgb[..., 0] = np.clip((r - red_limits[0]) / max(1e-9, red_limits[1] - red_limits[0]), 0, 1)
                rgb[..., 1] = np.clip((g - green_limits[0]) / max(1e-9, green_limits[1] - green_limits[0]), 0, 1)
                ax.imshow(rgb, origin="upper")
                if col == 0:
                    source_coords = _source_coordinates(mask_arrays[0], label, _label_boxes(mask_arrays[0]))
                    local = source_coords[(source_coords[:, 0] >= z0) & (source_coords[:, 0] < z1)]
                    if len(local):
                        layer = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=bool)
                        inside = ((local[:, 1] >= y0) & (local[:, 1] < y1) & (local[:, 2] >= x0) & (local[:, 2] < x1))
                        q = local[inside] - np.asarray([z0, y0, x0]); layer[tuple(q.T)] = True
                        ax.contour(np.max(layer, axis=0), levels=[0.5], colors=["yellow"], linewidths=0.8)
                else:
                    day_key = date_ids[col]
                    canonical, unfiltered = target_candidates.get((label, day_key), ([], []))
                    best_canonical = next((c for c in canonical if c["plausible"]), canonical[0] if canonical else None)
                    best_unfiltered = next((c for c in unfiltered if c["plausible"]), unfiltered[0] if unfiltered else None)
                    for candidate, volume, color in ((best_canonical, mask_arrays[col], "cyan"),
                                                     (best_unfiltered, unfiltered_arrays[col], "magenta")):
                        if candidate is None:
                            continue
                        target_label = int(candidate["label"])
                        crop = volume[z0:z1, y0:y1, x0:x1] == target_label
                        if crop.any():
                            ax.contour(np.max(crop, axis=0), levels=[0.5], colors=[color], linewidths=0.8)
                    # Also show the propagated source outline in yellow. Recompute
                    # only for the small review sample to keep the CSV compact.
                    prefix = "next" if col == 1 else "third"
                    source_coords = _source_coordinates(arrays["source_mask"], label, arrays["_source_boxes"])
                    coords, _coverage = propagate_source_roi(
                        source_coords, arrays[f"_{prefix}_source_to_target"], volume.shape
                    )
                    if len(coords):
                        local = coords[(coords[:, 0] >= z0) & (coords[:, 0] < z1)]
                        inside = ((local[:, 1] >= y0) & (local[:, 1] < y1) & (local[:, 2] >= x0) & (local[:, 2] < x1))
                        layer = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=bool)
                        if inside.any():
                            q = local[inside] - np.asarray([z0, y0, x0]); layer[tuple(q.T)] = True
                            ax.contour(np.max(layer, axis=0), levels=[0.5], colors=["yellow"], linewidths=0.8)
                ax.set_title(f"{date_id}\n{title if col == 0 else ''}", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{rows.mouse_id.iloc[0]} registered ROIs: red/green; yellow=propagated source, cyan=canonical, magenta=unfiltered", fontsize=10, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=0.7)
        path = figures / "representative_contact_sheets.png"; fig.savefig(path, dpi=135); plt.close(fig); created.append(str(path))
    return created


def _serialize_candidates(value: list[dict[str, Any]]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def run_audit(*, project_config: str | Path, mouse_id: str, start_session: str,
              next_session: str, third_session: str, unfiltered_mask_root: str | Path,
              output_dir: str | Path, laser_nm: int = 1050, run_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_project_config(project_config)
    output = Path(output_dir).expanduser().resolve()
    if is_relative_to(output, config.paths.derivatives_root) or is_relative_to(output, config.paths.raw_root):
        raise ValueError("Audit output must be outside raw_root and production derivatives_root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Audit output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"
    dates = (str(start_session), str(next_session), str(third_session))
    run = Path(run_dir).resolve() if run_dir else discover_run(config.paths.derivatives_root, mouse_id, laser_nm, *dates)
    input_paths = _find_run_paths(run, dates)
    transforms = _read_csv(input_paths["transforms"])
    registration_qc = _read_csv(input_paths["registration_qc"]) if "registration_qc" in input_paths else pd.DataFrame()
    tracks = _read_csv(input_paths["tracks"])
    graph_edges = _read_csv(input_paths["edges"])
    features = _read_csv(input_paths["features"])
    native = _read_csv(input_paths["native"])
    session_map = _date_session_map(features)
    if any(date not in session_map for date in dates):
        raise ValueError(f"Requested dates are not present in selected run features: {dates}; available={sorted(session_map)}")
    source_id, next_id, third_id = (session_map[date] for date in dates)
    tracks = tracks.copy()
    tracks["_audit_next_session_id"] = next_id
    tracks["_audit_third_session_id"] = third_id
    source_path = config.paths.derivatives_root / mouse_id / "sessions" / dates[0] / str(laser_nm)
    next_path = config.paths.derivatives_root / mouse_id / "sessions" / dates[1] / str(laser_nm)
    third_path = config.paths.derivatives_root / mouse_id / "sessions" / dates[2] / str(laser_nm)
    qc_root = Path(unfiltered_mask_root).expanduser().resolve()
    mask_paths = {
        "source": source_path / "segmentation" / "mask.tif",
        "next": next_path / "segmentation" / "mask.tif",
        "third": third_path / "segmentation" / "mask.tif",
        "source_unfiltered": qc_root / mouse_id / dates[0] / str(laser_nm) / "mask_unfiltered.tif",
        "next_unfiltered": qc_root / mouse_id / dates[1] / str(laser_nm) / "mask_unfiltered.tif",
        "third_unfiltered": qc_root / mouse_id / dates[2] / str(laser_nm) / "mask_unfiltered.tif",
    }
    image_paths = {
        "source_red": source_path / "preprocessing" / "red.tif",
        "source_green": source_path / "preprocessing" / "green.tif",
        "next_red": next_path / "preprocessing" / "red.tif",
        "next_green": next_path / "preprocessing" / "green.tif",
        "third_red": third_path / "preprocessing" / "red.tif",
        "third_green": third_path / "preprocessing" / "green.tif",
    }
    missing = [str(path) for path in [*mask_paths.values(), *image_paths.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required read-only audit inputs are missing:\n" + "\n".join(missing))

    # Read fixed inputs. Nothing in raw_root/derivatives_root is opened for writing.
    arrays: dict[str, np.ndarray] = {}
    for key, path in mask_paths.items():
        arrays["source_mask" if key == "source" else "next_mask" if key == "next" else
               "third_mask" if key == "third" else key] = tifffile.imread(path)
    for key, path in image_paths.items():
        arrays[key] = tifffile.imread(path)
    expected_shape = arrays["source_mask"].shape
    if len(expected_shape) != 3 or any(arrays[key].shape != expected_shape for key in arrays):
        bad = {key: array.shape for key, array in arrays.items() if array.shape != expected_shape}
        if bad:
            raise ValueError(f"Image and mask arrays must share the source 3D shape in this run: {bad}")
    # Current primary source masks are canonical and all dates in the audit are expected to align in shape.
    shapes = {key: tuple(arrays[key].shape) for key in arrays}
    spacing = (5.0, 0.693, 0.693)
    voxel_volume = float(np.prod(spacing))

    source_native, state_fit = _fit_historical_state(native, source_id)
    source_features = features.loc[features.session_id.astype(str).eq(source_id)].copy()
    if source_features.label.duplicated().any():
        raise ValueError("Source geometry features contain duplicate labels")
    source_features = source_features.set_index(source_features.label.astype(int), drop=False)
    for key, session_id in (("next", next_id), ("third", third_id)):
        count_key = f"{key}_mask"
        counts, centers, _ = _voxels_by_label(arrays[count_key])
        arrays[f"_{key}_counts"] = counts
        arrays[f"_{key}_centroids"] = centers
        arrays[f"_{key}_boxes"] = _label_boxes(arrays[count_key])
        count_u, center_u, _ = _voxels_by_label(arrays[f"{key}_unfiltered"])
        arrays[f"_{key}_unfiltered_counts"] = count_u
        arrays[f"_{key}_unfiltered_centroids"] = center_u
        arrays[f"_{key}_unfiltered_boxes"] = _label_boxes(arrays[f"{key}_unfiltered"])
        arrays[f"_{key}_occupied"] = np.maximum(arrays[f"{key}_mask"], arrays[f"{key}_unfiltered"])

    source_boxes = _label_boxes(arrays["source_mask"])
    arrays["_source_boxes"] = source_boxes
    red_key_to_features: dict[int, dict[str, Any]] = {}
    for day_key in ("next", "third"):
        stored, source_to_target, target_to_source = _load_transform(
            transforms, source_id, next_id if day_key == "next" else third_id, registration_qc
        )
        arrays[f"_{day_key}_stored_transform"] = stored
        arrays[f"_{day_key}_source_to_target"] = source_to_target
        arrays[f"_{day_key}_target_to_source"] = target_to_source
        arrays[f"_{day_key}_registered_red"] = warp_volume(
            arrays[f"{day_key}_red"], target_to_source, arrays["source_mask"].shape, order=1
        ).astype(np.float32, copy=False)

    spacing_rows = []
    source_unfiltered_counts, source_unfiltered_centroids, _ = _voxels_by_label(arrays["source_unfiltered"])
    arrays["_source_unfiltered_counts"] = source_unfiltered_counts
    arrays["_source_unfiltered_centroids"] = source_unfiltered_centroids
    arrays["_source_unfiltered_boxes"] = _label_boxes(arrays["source_unfiltered"])
    for date, key in zip(dates, ("source", "next", "third")):
        from cellpose import utils as cellpose_utils
        unfiltered_counts = arrays[f"_{key}_unfiltered_counts"]
        source_canonical_count = int(np.unique(arrays[f"{key}_mask"])[1:].size)
        unfiltered_count = int(np.count_nonzero(unfiltered_counts[1:]))
        qc_min100 = cellpose_utils.fill_holes_and_remove_small_masks(
            arrays[f"{key}_unfiltered"].copy(), min_size=100)
        qc_count = int(np.unique(qc_min100)[1:].size)
        del qc_min100
        spacing_rows.append({"session": date, "canonical_component_count": source_canonical_count,
                             "unfiltered_component_count": unfiltered_count,
                             "qc_min_size_100_component_count": qc_count,
                             "unfiltered_mask_path": str(mask_paths.get(f"{key}_unfiltered", "")),
                             "qc_min_size_100_path": "derived in memory from unfiltered mask"})

    native_index = source_native.set_index(source_native.mask_label.astype(int), drop=False)
    source_density = _roi_neighbor_density(features, source_id, spacing)
    graph_next_map, graph_third_map = _direct_graph_maps(graph_edges, source_id, next_id, third_id)
    source_shape = arrays["source_mask"].shape
    feature_by_date = {
        next_id: features.loc[features.session_id.astype(str).eq(next_id)],
        third_id: features.loc[features.session_id.astype(str).eq(third_id)],
    }
    records: list[dict[str, Any]] = []
    by_target: dict[tuple[int, str], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    source_labels = sorted(int(value) for value in source_native.mask_label.tolist())
    print(f"Loaded {mouse_id} source/target arrays; auditing {len(source_labels):,} {dates[0]} source ROIs", flush=True)
    for source_index, source_label in enumerate(source_labels):
        if source_index % 500 == 0:
            print(f"  ROI {source_index:,}/{len(source_labels):,}", flush=True)
        if source_label not in source_features.index or source_label not in native_index.index:
            continue
        native_row = native_index.loc[source_label]
        geometry = source_features.loc[source_label]
        coords = _source_coordinates(arrays["source_mask"], source_label, source_boxes)
        if len(coords) == 0:
            continue
        track = _track_lookup(tracks, source_id, source_label)
        graph_next = graph_next_map.get(source_label, set())
        graph_third = graph_third_map.get(source_label, set())
        source_measure_red = _local_measurement(arrays["source_red"], coords, arrays["source_mask"])
        source_measure_green = _local_measurement(arrays["source_green"], coords, arrays["source_mask"])
        source_boundary = bool(geometry.get("touches_xy_edge", False) or geometry.get("touches_z_edge", False)
                               or _boundary(coords, source_shape))
        row: dict[str, Any] = {
            "mouse_id": mouse_id, "source_session": dates[0], "next_session": dates[1],
            "third_session": dates[2], "source_session_id": source_id,
            "source_label": int(source_label), "source_track_id": track.get("track_uid", ""),
            "historical_state": str(native_row.historical_state),
            "source_red_bright": bool(native_row.source_red_bright),
            "source_historical_z": float(native_row.historical_z) if pd.notna(native_row.historical_z) else np.nan,
            "source_red_raw": float(native_row.red_raw),
            "source_red_corrected": float(native_row.red),
            "source_green_corrected": float(native_row.green),
            "source_predicted_green": float(native_row.predicted_green_historical) if pd.notna(native_row.predicted_green_historical) else np.nan,
            "source_voxel_count": int(geometry.area_voxels),
            "source_volume_um3": float(geometry.area_voxels) * voxel_volume,
            "source_centroid_z": float(geometry.centroid_z),
            "source_centroid_y": float(geometry.centroid_y),
            "source_centroid_x": float(geometry.centroid_x),
            "source_centroid_z_um": float(geometry.centroid_z_um),
            "source_centroid_y_um": float(geometry.centroid_y_um),
            "source_centroid_x_um": float(geometry.centroid_x_um),
            "source_boundary_flag": source_boundary,
            "source_touches_z_boundary": bool(geometry.get("touches_z_edge", False)),
            "source_touches_xy_boundary": bool(geometry.get("touches_xy_edge", False)),
            "source_local_density_30um": source_density.get(source_label, np.nan),
            "source_red_mean": source_measure_red["mean"],
            "source_red_median": source_measure_red["median"],
            "source_red_integrated": source_measure_red["integrated"],
            "source_red_q90": source_measure_red["q90"],
            "source_red_background_median": source_measure_red["background_median"],
            "source_red_background_mad": source_measure_red["background_mad"],
            "source_red_net_q90": source_measure_red["net_q90"],
            "source_green_mean": source_measure_green["mean"],
            "source_green_median": source_measure_green["median"],
            "source_green_integrated": source_measure_green["integrated"],
            "source_green_q90": source_measure_green["q90"],
            "graph_observed_next": bool(graph_next), "graph_observed_third": bool(graph_third),
            "graph_next_target_label": next(iter(graph_next)) if len(graph_next) == 1 else np.nan,
            "graph_third_target_label": next(iter(graph_third)) if len(graph_third) == 1 else np.nan,
            "graph_next_target_labels_json": json.dumps(sorted(graph_next)),
            "graph_third_target_labels_json": json.dumps(sorted(graph_third)),
            "oct6_graph_gap_oct7_observed": bool(not graph_next and graph_third),
        }
        for day_key, target_session_id in (("next", next_id), ("third", third_id)):
            prop, coverage = propagate_source_roi(coords, arrays[f"_{day_key}_source_to_target"],
                                                  arrays[f"{day_key}_mask"].shape)
            canonical = overlap_candidates(prop, arrays[f"{day_key}_mask"], arrays[f"_{day_key}_counts"],
                                           arrays[f"_{day_key}_centroids"], spacing,
                                           arrays[f"_{day_key}_boxes"])
            unfiltered = overlap_candidates(prop, arrays[f"{day_key}_unfiltered"], arrays[f"_{day_key}_unfiltered_counts"],
                                            arrays[f"_{day_key}_unfiltered_centroids"], spacing,
                                            arrays[f"_{day_key}_unfiltered_boxes"])
            by_target[(source_label, dates[1] if day_key == "next" else dates[2])] = (canonical, unfiltered)
            row.update(_candidate_fields(canonical, f"{day_key}_canonical"))
            row.update(_candidate_fields(unfiltered, f"{day_key}_unfiltered"))
            source_to_target = arrays[f"_{day_key}_source_to_target"]
            projected_centroid = source_to_target.apply(np.asarray([geometry.centroid_z,
                                                                     geometry.centroid_y,
                                                                     geometry.centroid_x]))
            target_boundary = _boundary(prop, arrays[f"{day_key}_mask"].shape)
            local_ncc = _local_ncc(arrays["source_red"], arrays[f"_{day_key}_registered_red"], coords, source_shape)
            assess_reasons = []
            if source_boundary:
                assess_reasons.append("source_boundary")
            if coverage < 0.90:
                assess_reasons.append("transformed_roi_coverage_below_90pct")
            if target_boundary:
                assess_reasons.append("propagated_roi_near_target_boundary")
            if not np.isfinite(local_ncc) or local_ncc < 0.05:
                assess_reasons.append("local_red_ncc_below_0.05")
            geometry_assessable = bool(coverage >= 0.90 and len(prop) > 0
                                       and not source_boundary and not target_boundary)
            image_assessable = bool(geometry_assessable and np.isfinite(local_ncc) and local_ncc >= 0.05)
            occupied = arrays[f"_{day_key}_occupied"]
            red_metrics = _image_and_mask_metrics(arrays[f"{day_key}_red"], prop, occupied, f"{day_key}_red")
            green_metrics = _image_and_mask_metrics(arrays[f"{day_key}_green"], prop, occupied, f"{day_key}_green")
            row.update(red_metrics); row.update(green_metrics)
            prefix = "next" if day_key == "next" else "third"
            row[f"{prefix}_transform_method"] = str(arrays[f"_{day_key}_stored_transform"].get("method", ""))
            row[f"{prefix}_transform_fallback_reason"] = str(arrays[f"_{day_key}_stored_transform"].get("fallback_reason", ""))
            row[f"{prefix}_requested_registration_mode"] = str(arrays[f"_{day_key}_stored_transform"].get("registration_requested_registration_mode", ""))
            row[f"{prefix}_selected_registration_stage"] = str(arrays[f"_{day_key}_stored_transform"].get("registration_selected_registration_stage", row[f"{prefix}_transform_method"]))
            row[f"{prefix}_smoothing_um"] = arrays[f"_{day_key}_stored_transform"].get("registration_smoothing_um", np.nan)
            row[f"{prefix}_transform_fallback_used"] = bool(arrays[f"_{day_key}_stored_transform"].get("registration_fallback_used", False))
            row[f"{prefix}_transform_qc_pass"] = bool(
                arrays[f"_{day_key}_stored_transform"].get(
                    "registration_local_qc_pass" if row[f"{prefix}_selected_registration_stage"] == "image_affine_local" else "registration_affine_qc_pass",
                    row[f"{prefix}_selected_registration_stage"] == "legacy",
                )
            )
            row[f"{prefix}_transform_residual_median_um"] = float(arrays[f"_{day_key}_stored_transform"].get("residual_median_um", np.nan))
            row[f"{prefix}_transform_residual_p95_um"] = float(arrays[f"_{day_key}_stored_transform"].get("residual_p95_um", np.nan))
            row[f"{prefix}_transform_coverage_fraction"] = coverage
            row[f"{prefix}_propagated_voxel_count"] = int(len(prop))
            row[f"{prefix}_target_boundary_flag"] = target_boundary
            row[f"{prefix}_local_registration_ncc"] = local_ncc
            row[f"{prefix}_geometry_assessable"] = geometry_assessable
            row[f"{prefix}_image_assessable"] = image_assessable
            row[f"{prefix}_transformed_centroid_zyx"] = projected_centroid.tolist()
            assessment_canonical = _assessment_candidates(canonical, image_assessable=image_assessable)
            assessment_unfiltered = _assessment_candidates(unfiltered, image_assessable=image_assessable)
            assessable = _primary_assessable(
                geometry_assessable=geometry_assessable,
                image_assessable=image_assessable,
                canonical_candidates=canonical,
                unfiltered_candidates=unfiltered,
            )
            row[f"{prefix}_assessable"] = assessable
            if assessable and not image_assessable:
                reason_text = "mask_correspondence_assessable_despite_weak_local_ncc"
            else:
                reason_text = ";".join(assess_reasons) if assess_reasons else "assessable"
            row[f"{prefix}_assessability_reason"] = reason_text
            canonical_plausible = [c for c in assessment_canonical if c["plausible"]]
            unfiltered_plausible = [c for c in assessment_unfiltered if c["plausible"]]
            ambiguous, reason = _ambiguous(canonical_plausible if canonical_plausible else unfiltered_plausible,
                                           len(prop))
            row[f"{prefix}_canonical_plausible"] = bool(canonical_plausible)
            row[f"{prefix}_unfiltered_plausible"] = bool(unfiltered_plausible)
            row[f"{prefix}_ambiguity_flag"] = ambiguous
            row[f"{prefix}_ambiguity_reason"] = reason
            graph_labels = graph_next if day_key == "next" else graph_third
            # Signals are calibrated from stable core-Middle controls after all rows
            # are constructed. Store a provisional flag and overwrite below.
            row[f"{prefix}_red_signal_remains"] = False
            row[f"{prefix}_primary_outcome"] = ""
            if day_key == "next":
                row["next_graph_observed"] = bool(graph_next)
                row["next_graph_target_label"] = next(iter(graph_next)) if len(graph_next) == 1 else np.nan
            else:
                row["third_graph_observed"] = bool(graph_third)
                row["third_graph_target_label"] = next(iter(graph_third)) if len(graph_third) == 1 else np.nan
        row["next_red_retention_ratio"] = (
            max(float(row.get("next_red_net_q90", np.nan)), float(row.get("next_red_expanded_net_q90", np.nan)))
            / float(row["source_red_net_q90"])
            if np.isfinite(row.get("source_red_net_q90", np.nan)) and row["source_red_net_q90"] > 0 else np.nan
        )
        row["third_red_retention_ratio"] = (
            max(float(row.get("third_red_net_q90", np.nan)), float(row.get("third_red_expanded_net_q90", np.nan)))
            / float(row["source_red_net_q90"])
            if np.isfinite(row.get("source_red_net_q90", np.nan)) and row["source_red_net_q90"] > 0 else np.nan
        )
        records.append(row)

    audit = pd.DataFrame(records)
    if audit.empty:
        raise ValueError("No Oct 5 source ROIs could be joined across native intensity and geometry tables")
    print(f"Finished registered ROI measurements ({len(audit):,} rows); calibrating signal and outcome tables", flush=True)
    # Record convenient path fields for montage generator without emitting the
    # full volumes into the output table.
    audit["_source_session_id"] = source_id
    audit["_next_session_id"] = next_id
    audit["_third_session_id"] = third_id
    signal_cutoff, signal_sensitivity_cutoff, n_controls = _calibrate_signal_threshold(audit)
    for row_index, row in audit.iterrows():
        for prefix in ("next", "third"):
            remain = _signal_remains({
                "next_red_net_q90": row[f"{prefix}_red_net_q90"],
                "next_red_expanded_net_q90": row[f"{prefix}_red_expanded_net_q90"],
                "source_red_net_q90": row["source_red_net_q90"],
                "next_red_background_mad": row[f"{prefix}_red_background_mad"],
            }, signal_cutoff)
            audit.loc[row_index, f"{prefix}_red_signal_remains"] = remain
            candidates, unfiltered = by_target[(int(row.source_label), dates[1] if prefix == "next" else dates[2])]
            candidates = _assessment_candidates(candidates, image_assessable=bool(row[f"{prefix}_image_assessable"]))
            unfiltered = _assessment_candidates(unfiltered, image_assessable=bool(row[f"{prefix}_image_assessable"]))
            graph_labels = set(map(int, json.loads(row.graph_next_target_labels_json if prefix == "next"
                                                   else row.graph_third_target_labels_json)))
            outcome = classify_primary_outcome(
                assessable=bool(row[f"{prefix}_assessable"]),
                canonical_candidates=candidates,
                unfiltered_candidates=unfiltered,
                graph_target_labels=graph_labels,
                fragmentation_or_merge=bool(row[f"{prefix}_ambiguity_flag"]),
                red_signal_remains=remain,
            )
            audit.loc[row_index, f"{prefix}_primary_outcome"] = outcome
    # Primary Oct 6 aliases make the CSV and downstream summaries easier to use.
    audit["next_primary_outcome"] = audit.next_primary_outcome.astype(str)
    audit["next_assessable"] = audit.next_assessable.astype(bool)
    audit["next_red_signal_remains"] = audit.next_red_signal_remains.astype(bool)
    audit["next_canonical_plausible"] = audit.next_canonical_plausible.astype(bool)
    audit["next_unfiltered_plausible"] = audit.next_unfiltered_plausible.astype(bool)
    audit["next_local_registration_ncc"] = pd.to_numeric(audit.next_local_registration_ncc, errors="coerce")
    strict_outcomes: list[str] = []
    for _, row in audit.iterrows():
        canonical, unfiltered = by_target[(int(row.source_label), dates[1])]
        canonical = _assessment_candidates(canonical, image_assessable=bool(row.next_image_assessable))
        unfiltered = _assessment_candidates(unfiltered, image_assessable=bool(row.next_image_assessable))
        strict_canonical = [dict(candidate, plausible=bool(candidate["strict_plausible"])) for candidate in canonical]
        strict_unfiltered = [dict(candidate, plausible=bool(candidate["strict_plausible"])) for candidate in unfiltered]
        strict_ambiguous, _ = _ambiguous(strict_canonical if any(c["plausible"] for c in strict_canonical)
                                         else strict_unfiltered, int(row.next_propagated_voxel_count))
        strict_assessable = _primary_assessable(
            geometry_assessable=bool(row.next_geometry_assessable),
            image_assessable=bool(row.next_image_assessable),
            canonical_candidates=strict_canonical,
            unfiltered_candidates=strict_unfiltered,
        )
        graph_labels = set(map(int, json.loads(row.graph_next_target_labels_json)))
        strict_outcomes.append(classify_primary_outcome(
            assessable=strict_assessable, canonical_candidates=strict_canonical,
            unfiltered_candidates=strict_unfiltered, graph_target_labels=graph_labels,
            fragmentation_or_merge=strict_ambiguous, red_signal_remains=bool(row.next_red_signal_remains),
        ))
    audit["next_primary_outcome_strict_overlap"] = strict_outcomes
    # Derive Oct 7 fate among Oct 6 non-tracked source objects.
    fates = []
    for _, row in audit.iterrows():
        if row.next_primary_outcome == "TRACKED_OK":
            fates.append("NOT_APPLICABLE_OCT6_TRACKED")
        elif not bool(row.third_assessable):
            fates.append("NOT_ASSESSABLE")
        elif bool(row.graph_observed_third):
            fates.append("SAME_GRAPH_TRACK_REAPPEARS")
        elif bool(row.third_canonical_plausible):
            fates.append("PLAUSIBLE_OBJECT_REAPPEARS_GRAPH_IDENTITY_DIFFERS")
        elif bool(row.third_unfiltered_plausible) and int(row.third_unfiltered_best_target_voxels) < MIN_SIZE_VOXELS:
            fates.append("SUBTHRESHOLD_OBJECT_PERSISTS_OR_RETURNS")
        elif bool(row.third_red_signal_remains):
            fates.append("NO_MASK_BUT_RED_SIGNAL_REMAINS")
        else:
            fates.append("NO_CORRESPONDING_OBJECT_OR_SIGNAL")
    audit["oct7_fate"] = fates

    # Sensitivity classification with a less conservative signal-retention cutoff.
    sensitivity_outcomes = []
    for _, row in audit.iterrows():
        candidates, unfiltered = by_target[(int(row.source_label), dates[1])]
        candidates = _assessment_candidates(candidates, image_assessable=bool(row.next_image_assessable))
        unfiltered = _assessment_candidates(unfiltered, image_assessable=bool(row.next_image_assessable))
        graph_labels = set(map(int, json.loads(row.graph_next_target_labels_json)))
        remain = _signal_remains({
            "next_red_net_q90": row.next_red_net_q90,
            "next_red_expanded_net_q90": row.next_red_expanded_net_q90,
            "source_red_net_q90": row.source_red_net_q90,
            "next_red_background_mad": row.next_red_background_mad,
        }, signal_sensitivity_cutoff)
        sensitivity_outcomes.append(classify_primary_outcome(
            assessable=bool(row.next_assessable), canonical_candidates=candidates,
            unfiltered_candidates=unfiltered, graph_target_labels=graph_labels,
            fragmentation_or_merge=bool(row.next_ambiguity_flag), red_signal_remains=remain,
        ))
    audit["next_primary_outcome_signal_sensitivity"] = sensitivity_outcomes

    # Count validation: compare native and provided QC masks, without regenerating.
    validation = pd.DataFrame(spacing_rows)
    expected_counts = {"20261005": (5139, 4538), "20261006": (4926, None),
                       "20261007": (4813, None)} if mouse_id == "Fucci-Dead_3" else {}
    for index, row in validation.iterrows():
        expected_unfiltered, expected_canonical = expected_counts.get(row.session, (None, None))
        validation.loc[index, "expected_unfiltered_count"] = expected_unfiltered
        validation.loc[index, "expected_canonical_count_or_approx"] = (expected_canonical if expected_canonical is not None else row.qc_min_size_100_component_count)
        validation.loc[index, "unfiltered_count_delta"] = (row.unfiltered_component_count - expected_unfiltered) if expected_unfiltered is not None and pd.notna(row.unfiltered_component_count) else np.nan
        validation.loc[index, "qc_canonical_count_delta"] = (row.qc_min_size_100_component_count - expected_canonical) if expected_canonical is not None and pd.notna(row.qc_min_size_100_component_count) else np.nan
    validation.to_csv(output / "canonical_unfiltered_consistency.csv", index=False)

    matched_summary, matched_pairs = _match_controls(audit)
    print(f"Matched {len(matched_pairs):,} Low/Middle pairs; writing reports and figures", flush=True)
    models = _adjusted_models(audit)
    summary = _rate_summary(audit)
    covariates = _starting_covariate_summary(audit)
    strict_table = audit.copy()
    strict_table["next_primary_outcome"] = strict_table["next_primary_outcome_strict_overlap"]
    strict_summary = _rate_summary(strict_table)
    strict_summary["overlap_rule"] = "strict: source overlap >=0.35 or Dice >=0.25 and target overlap >=0.15"
    # Add core Middle/Lows with count/fraction for all explicit outcomes.
    audit.to_csv(output / "roi_outcome_audit.csv", index=False)
    summary.to_csv(output / "outcome_summary_by_state.csv", index=False)
    covariates.to_csv(output / "starting_covariate_summary.csv", index=False)
    strict_summary.to_csv(output / "overlap_sensitivity_summary.csv", index=False)
    matched_summary.to_csv(output / "matched_control_summary.csv", index=False)
    matched_pairs.to_csv(output / "matched_control_pairs.csv", index=False)
    models.to_csv(output / "adjusted_model_summary.csv", index=False)
    min_size_cases = audit.loc[audit.next_primary_outcome.eq("MIN_SIZE_FILTER_DROPOUT")].copy()
    _compact_case_table(min_size_cases).to_csv(output / "min_size_dropout_cases.csv", index=False)
    identity_cases = audit.loc[audit.next_primary_outcome.eq("TARGET_PRESENT_GRAPH_IDENTITY_FAILURE")].copy()
    _compact_case_table(identity_cases).to_csv(output / "graph_identity_failure_cases.csv", index=False)
    reporter_cases = audit.loc[audit.next_primary_outcome.eq("REPORTER_OBJECT_DISAPPEARANCE")].copy()
    _compact_case_table(reporter_cases).to_csv(output / "reporter_disappearance_candidates.csv", index=False)
    review_shortlist = _manual_review_shortlist(audit)
    review_shortlist.to_csv(output / "manual_review_shortlist.csv", index=False)
    # More inclusive residual-fluorescence candidates are useful for reviewer selection.
    _compact_case_table(audit.loc[audit.next_primary_outcome.eq("NO_MASK_BUT_RED_SIGNAL_REMAINS")]).to_csv(
        output / "segmentation_failure_red_signal_cases.csv", index=False)
    _compact_case_table(audit.loc[audit.oct7_fate.isin(["SAME_GRAPH_TRACK_REAPPEARS", "PLAUSIBLE_OBJECT_REAPPEARS_GRAPH_IDENTITY_DIFFERS",
                                   "SUBTHRESHOLD_OBJECT_PERSISTS_OR_RETURNS", "NO_MASK_BUT_RED_SIGNAL_REMAINS",
                                   "NO_CORRESPONDING_OBJECT_OR_SIGNAL"])]).to_csv(
        output / "oct7_fates_of_oct6_dropouts.csv", index=False)
    models.to_csv(output / "adjusted_model_summary.csv", index=False)

    # Detect the known pattern by values and spatial correspondence, not a label ID.
    near_proof = audit.loc[
        audit.source_voxel_count.between(90, 115)
        & audit.next_unfiltered_plausible
        & pd.to_numeric(audit.next_unfiltered_best_target_voxels, errors="coerce").between(80, 99)
        & (audit.third_canonical_plausible | audit.third_unfiltered_plausible)
        & (pd.to_numeric(audit.third_canonical_best_target_voxels, errors="coerce").fillna(0).ge(150)
           | pd.to_numeric(audit.third_unfiltered_best_target_voxels, errors="coerce").fillna(0).ge(150))
    ].copy()
    near_proof.to_csv(output / "known_min_size_rebound_validation.csv", index=False)

    # Generate graph and QC summary plots plus representative contact sheet.
    plot_paths = _plot_outputs(
        audit, matched_summary, models, figures_dir, matched_pairs, arrays, dates,
        source_features.reset_index(drop=True), by_target, set(),
    )
    write_paths = sorted(str(path.resolve()) for path in output.rglob("*") if path.is_file())
    input_hashes = {str(path): _sha256(path) for path in [*input_paths.values(), *mask_paths.values(), *image_paths.values()]}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True, check=False).stdout.strip(),
        "audit_code_sha256": _sha256(Path(__file__)),
        "mouse_id": mouse_id, "laser_nm": int(laser_nm), "sessions": list(dates),
        "project_config": str(Path(project_config).resolve()), "run_dir": str(run),
        "input_sha256": input_hashes, "state_fit": state_fit,
        "signal_remaining_threshold_fraction_of_source_net_q90": signal_cutoff,
        "signal_remaining_sensitivity_threshold": signal_sensitivity_cutoff,
        "n_stable_middle_signal_controls": n_controls,
        "primary_overlap_rule": {"source_fraction_min": PRIMARY_SOURCE_OVERLAP,
                                 "dice_min": PRIMARY_DICE, "strict_source_fraction_min": STRICT_SOURCE_OVERLAP,
                                 "strict_dice_min": STRICT_DICE},
        "weak_ncc_mask_correspondence_rule": "Strict overlap rule required when local Red NCC <0.05; image-only/no-mask calls are not classified below this NCC threshold.",
        "primary_outcome_precedence": [
            "NOT_ASSESSABLE if geometry QC fails, or if no plausible mask correspondence exists and image NCC is insufficient",
            "TRACKED_OK if a single unambiguous canonical target overlaps and graph selects it",
            "FRAGMENTATION_OR_MERGE_AMBIGUITY for multiple jointly plausible components or merge suspicion",
            "TARGET_PRESENT_GRAPH_IDENTITY_FAILURE for a single canonical target without matching graph identity",
            "MIN_SIZE_FILTER_DROPOUT for only plausible unfiltered targets below 100 voxels",
            "NO_MASK_BUT_RED_SIGNAL_REMAINS if no plausible mask and calibrated fixed-region Red signal persists",
            "REPORTER_OBJECT_DISAPPEARANCE if no plausible mask or calibrated fixed-region Red signal persists",
        ],
        "spacing_zyx_um": list(spacing), "voxel_volume_um3": voxel_volume,
        "unfiltered_masks": {date: str(mask_paths[f"{key}_unfiltered"])
                             for date, key in ((dates[1], "next"), (dates[2], "third"))},
        "transform_convention": "Saved pairwise transforms map later session B coordinates into earlier session A; the audit uses the repository inverse utility to propagate A voxel centers forward into B.",
        "limitations": [
            "Fixed-reference image measures are Red/Green intensity stacks, not whole-cell ground truth.",
            "Local NCC is a contextual observability check and is not a deformable registration or cell identity test.",
            "Source state is recomputed from native Oct 5 ROI values using the supplied historical threshold convention because no saved assignment table exists for these sessions.",
            "The audit is one mouse and does not support animal-level inferential claims.",
        ],
        "outputs": write_paths,
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report = _write_report(output, audit, summary, matched_summary, matched_pairs, models,
                           validation, state_fit, signal_cutoff, signal_sensitivity_cutoff,
                           n_controls, near_proof, plot_paths, run, review_shortlist, covariates,
                           mouse_id=mouse_id, dates=dates)
    # Rewrite manifest after adding the report.
    manifest["outputs"] = sorted(str(path.resolve()) for path in output.rglob("*") if path.is_file())
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_dir": str(output), "report": str(report), "n_source_rows": int(len(audit)),
            "n_low": int(audit.historical_state.eq("Low").sum()),
            "n_middle": int(audit.historical_state.eq("Middle").sum()),
            "n_min_size": int(len(min_size_cases)), "n_identity_failure": int(len(identity_cases)),
            "n_reporter_disappearance": int(len(reporter_cases)), "n_known_rebound": int(len(near_proof)),
            "run_dir": str(run)}


def _write_report(output: Path, audit: pd.DataFrame, summary: pd.DataFrame,
                  matched: pd.DataFrame, pairs: pd.DataFrame, models: pd.DataFrame,
                  validation: pd.DataFrame, state_fit: dict[str, Any],
                  signal_threshold: float, sensitivity_threshold: float,
                  n_signal_controls: int, proof: pd.DataFrame,
                  plot_paths: list[str], run_dir: Path,
                  review_shortlist: pd.DataFrame,
                  covariates: pd.DataFrame,
                  mouse_id: str = "Fucci-Dead_3",
                  dates: tuple[str, str, str] = ("20261005", "20261006", "20261007")) -> Path:
    def count(state: str, outcome: str) -> tuple[int, int, float]:
        group = audit.loc[audit.historical_state.eq(state)]
        n = int(group.next_primary_outcome.eq(outcome).sum())
        den = int(len(group))
        return n, den, n / den if den else float("nan")
    if "modal_slope" in state_fit:
        state_assignment_text = (
            "Source state was assigned from corrected ROI intensities using "
            "the repository's deterministic Gaussian-kernel modal regression, initialized by OLS, with the established 0.6 robust-scale bandwidth. "
            f"The raw-Red top-half cutoff was `{state_fit['red_bright_raw_red_cutoff']:.3f}`; **{state_fit['n_fit_top_half_after_green_cutoff']:,}** fit ROIs remained after excluding corrected Green >5,000 and requiring finite positive corrected Red/Green. "
            f"The modal slope was `{state_fit['modal_slope']:.8f}` and intercept `{state_fit['modal_intercept']:.5f}` (robust residual scale `{state_fit['robust_initial_scale_green_units']:.3f}`, bandwidth `{state_fit['modal_bandwidth_green_units']:.3f}`). "
            f"Z is `log2(corrected Green / predicted Green) / {state_fit['z_scale']:.15g}`. Low is Z < −3, core Middle is −1 ≤ Z ≤ 1, High is Z > 3; transition/other values remain separate."
        )
    else:
        state_assignment_text = (
            f"The saved run contains native ROI intensities but no state-assignment table for these dates. Oct 5 state was therefore reconstructed from the native ROI table: raw Red top-half cutoff `{state_fit['red_bright_raw_red_cutoff']:.3f}`, OLS corrected Green on corrected Red fit using **{state_fit['n_fit_top_half_after_green_cutoff']:,}** ROIs after excluding corrected Green >5,000, slope `{state_fit['ols_slope_corrected_green_on_corrected_red']:.6g}`, intercept `{state_fit['ols_intercept']:.6g}`, and the specified Z divisor `{state_fit['z_scale']:.15g}`. Low is Z < −3 and core Middle is −1 ≤ Z ≤ 1. This state is not a claim about cell-cycle phase or pG1 identity."
        )
    sections = [
        "# Dead_3 Fucci tracking-dropout root-cause audit",
        "",
        f"- Primary comparison: `{audit.source_session.iloc[0]} -> {audit.next_session.iloc[0]}`; endpoint: `{audit.third_session.iloc[0]}`.",
        f"- Existing matching run: `{run_dir}`.",
        f"- Source rows: **{len(audit):,}** Oct 5 canonical ROIs; source Red-bright: **{int(audit.source_red_bright.sum()):,}**; Low: **{int(audit.historical_state.eq('Low').sum()):,}**; core Middle: **{int(audit.historical_state.eq('Middle').sum()):,}**.",
        "- One mouse only. Percentages describe ROIs within Fucci-Dead_3 and are not animal-level statistical inference.",
        "",
        "## State assignment and correspondence rules",
        "",
        state_assignment_text,
        "",
        "Source mask voxel centers were propagated with the saved Oct 5→Oct 6 and Oct 5→Oct 7 affine transforms using the repository's transform inversion and resampling conventions. Candidate matching is based on 3D overlap; a target is plausible when source overlap is ≥15%, or Dice ≥0.10 with target overlap ≥10%. A stricter sensitivity rule uses ≥35% source overlap, or Dice ≥0.25 with target overlap ≥15%. Oct 2 was not used as an identity anchor: graph links are read from accepted direct Oct 5→6 and Oct 5→7/Oct 6→7 edges.",
        "",
        f"For no-mask calls, Red signal was measured in the fixed propagated source mask and a 2D lateral expansion, with a nearby shell that excludes target segmentation. Residual signal threshold was calibrated at the 5th percentile of stable Middle target/source net-q90 retention (floor 0.20), yielding **{signal_threshold:.3f}** from **{n_signal_controls}** stable Middle controls; sensitivity uses **{sensitivity_threshold:.3f}**. When local high-pass Red NCC is ≥0.05, mask correspondence uses the primary 3D overlap rule; when NCC is <0.05, mask-based classification requires the stricter overlap rule (source overlap ≥35%, or Dice ≥0.25 with target overlap ≥15%). Either case requires at least 90% transformed ROI coverage and no source/target volume-boundary truncation. Weak-NCC no-mask calls are not classified.",
        "",
        "Outcome precedence: not assessable QC; unambiguous canonical overlap plus matching graph target; fragmentation/merge ambiguity; canonical target with graph mismatch; only plausible unfiltered <100 target; no plausible mask with calibrated Red remaining; no plausible mask or calibrated Red remaining. This leaves the categories mutually exclusive.",
        "",
        "## Oct 6 primary outcomes",
        "",
        "Counts below use all source ROIs in the named state; all percentages include the exact state denominator.",
        "",
        "| Outcome | Low / state denominator | Core Middle / state denominator | Low / graph-unobserved denominator | Middle / graph-unobserved denominator |",
        "|---|---:|---:|---:|---:|",
    ]
    for outcome in PRIMARY_OUTCOMES:
        n_l, d_l, p_l = count("Low", outcome); n_m, d_m, p_m = count("Middle", outcome)
        gl = audit.loc[audit.historical_state.eq("Low") & ~audit.next_graph_observed.astype(bool)]
        gm = audit.loc[audit.historical_state.eq("Middle") & ~audit.next_graph_observed.astype(bool)]
        gl_n, gm_n = int(len(gl)), int(len(gm))
        gl_count, gm_count = int(gl.next_primary_outcome.eq(outcome).sum()), int(gm.next_primary_outcome.eq(outcome).sum())
        gl_pct = gl_count / gl_n if gl_n else float("nan")
        gm_pct = gm_count / gm_n if gm_n else float("nan")
        sections.append(f"| `{outcome}` | {n_l}/{d_l} ({p_l:.1%}) | {n_m}/{d_m} ({p_m:.1%}) | {gl_count}/{gl_n} ({gl_pct:.1%}) | {gm_count}/{gm_n} ({gm_pct:.1%}) |")
    sections += ["", "### Key mechanisms among graph-unobserved ROIs", ""]
    for state in ("Low", "Middle"):
        g = audit.loc[audit.historical_state.eq(state)]
        graph_lost = g.loc[~g.next_graph_observed.astype(bool)]
        lost = int(len(graph_lost))
        if lost:
            section_counts = {name: int(graph_lost.next_primary_outcome.eq(name).sum()) for name in PRIMARY_OUTCOMES}
            sections.append(f"- **{state}:** {lost}/{len(g)} lacked an Oct 6 graph observation. Within those {lost} losses: {section_counts['MIN_SIZE_FILTER_DROPOUT']} were min-size cases, {section_counts['TARGET_PRESENT_GRAPH_IDENTITY_FAILURE']} had a plausible canonical target with graph identity failure, {section_counts['NO_MASK_BUT_RED_SIGNAL_REMAINS']} had no plausible mask but retained Red signal, {section_counts['REPORTER_OBJECT_DISAPPEARANCE']} were conservative reporter-object disappearance candidates, {section_counts['FRAGMENTATION_OR_MERGE_AMBIGUITY']} were ambiguous, and {section_counts['NOT_ASSESSABLE']} were not assessable.")
        else:
            sections.append(f"- **{state}:** no graph-observation losses in the {len(g)} ROI denominator.")
    low_canonical = int(audit.loc[audit.historical_state.eq("Low"), "next_canonical_plausible"].sum())
    middle_canonical = int(audit.loc[audit.historical_state.eq("Middle"), "next_canonical_plausible"].sum())
    low_identity = int(audit.loc[(audit.historical_state.eq("Low")) & audit.next_primary_outcome.eq("TARGET_PRESENT_GRAPH_IDENTITY_FAILURE")].shape[0])
    middle_identity = int(audit.loc[(audit.historical_state.eq("Middle")) & audit.next_primary_outcome.eq("TARGET_PRESENT_GRAPH_IDENTITY_FAILURE")].shape[0])
    sections += ["", f"Graph identity failure conditional on a plausible canonical target was **{low_identity}/{low_canonical} ({low_identity/low_canonical:.1%}) in Low** and **{middle_identity}/{middle_canonical} ({middle_identity/middle_canonical:.1%}) in core Middle**. The unfiltered target is treated as a min-size case only when it is spatially plausible, below 100 voxels, and no canonical target is plausible.", ""]
    assessable_rows = audit.loc[audit.historical_state.isin(["Low", "Middle"]) & audit.next_primary_outcome.eq("NOT_ASSESSABLE")]
    assess_counts = assessable_rows.groupby(["historical_state", "next_assessability_reason"]).size().to_dict()
    sections.append("Not-assessable reasons overlap when flags coexist. Combined reason counts among state-classified ROIs are: " + "; ".join(f"{state} `{reason}` {int(n)}" for (state, reason), n in sorted(assess_counts.items())) + ".")
    sections.append("")

    sections += ["## Starting observability differences", "",
                 "The table reports medians and interquartile ranges for the Oct 5 Low and core-Middle cohorts. These are descriptive ROI summaries from this one mouse.",
                 "", "| Covariate | Low median (Q1–Q3) | Core Middle median (Q1–Q3) |", "|---|---:|---:|"]
    covariate_labels = {
        "source_volume_voxels": "Source size (voxels)",
        "source_volume_um3": "Source volume (µm³)",
        "corrected_red_mean": "Corrected Red mean",
        "integrated_red_proxy_raw_sum": "Integrated Red proxy (raw sum)",
        "corrected_green_mean": "Corrected Green mean",
        "depth_um": "Z depth (µm)",
        "source_boundary_fraction": "Source boundary fraction",
        "local_density_30um": "Local density within 30 µm",
        "local_registration_ncc_oct6": "Oct 6 local Red NCC",
    }
    for metric, label in covariate_labels.items():
        left = covariates.loc[(covariates.metric.eq(metric)) & covariates.state.eq("Low")].iloc[0]
        right = covariates.loc[(covariates.metric.eq(metric)) & covariates.state.eq("Middle")].iloc[0]
        def fmt_cov(row: pd.Series) -> str:
            if metric == "source_boundary_fraction":
                return f"{row['mean']:.1%}"
            return f"{row['median']:.3g} ({row['q25']:.3g}–{row['q75']:.3g})"
        sections.append(f"| {label} | {fmt_cov(left)} | {fmt_cov(right)} |")
    sections.append("")

    sections += ["## Covariate adjustment", ""]
    if not pairs.empty:
        sections.append(f"The 1:1 matched comparison paired **{len(pairs)} Low ROIs** to unique core-Middle controls using standardized log starting voxel count, log corrected Red, Z depth, and local density; source boundary status had an exact-match penalty. Matching is within this single mouse and remains diagnostic.")
        sections.append("")
        sections.append("| Metric | Low matched rate | Middle matched rate | Paired difference | N pairs |")
        sections.append("|---|---:|---:|---:|---:|")
        for _, row in matched.iterrows():
            sections.append(f"| {row.metric} | {row.low_numerator}/{row.low_denominator} ({row.low_fraction:.1%}) | {row.middle_numerator}/{row.middle_denominator} ({row.middle_fraction:.1%}) | {row.paired_difference_low_minus_middle:+.1%} | {row.matched_pairs} |")
    else:
        sections.append("No complete Low/Middle matched set was available after observability QC.")
    sections.append("")
    sections.append("A ridge logistic sensitivity model adjusts for Low state, log source volume, log corrected Red brightness, depth, source boundary status, and local Red NCC. The graph-observation model uses all Low/Middle sources; outcome-mechanism models use assessable rows only. Coefficients are diagnostics, not animal-level p-values:")
    sections.append("")
    if "low_adjusted_odds_ratio" in models:
        sections.append("| Model outcome | N | Events | Low adjusted OR | Predicted Middle | Predicted Low | Adjusted probability difference |")
        sections.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in models.iterrows():
            sections.append(f"| {row.model} | {int(row.n)} | {int(row.events)} | {row.low_adjusted_odds_ratio:.3g} | {row.predicted_probability_middle_at_median_covariates:.1%} | {row.predicted_probability_low_at_median_covariates:.1%} | {row.adjusted_probability_difference_low_minus_middle:+.1%} |")
    else:
        sections.append("The model did not have enough complete Low and Middle cases to fit.")
    sections += ["", "## Fixed-region signal, registration, and Oct 7 fate", ""]
    analysis_states = audit.historical_state.isin(["Low", "Middle"])
    dropouts = audit.loc[analysis_states & ~audit.next_graph_observed.astype(bool)]
    fate_counts = dropouts.oct7_fate.value_counts()
    if len(dropouts):
        sections.append(f"Among **{len(dropouts)}** Low/core-Middle source ROIs without an Oct 6 graph observation, Oct 7 fate counts are: " + "; ".join(f"`{name}` {int(value)}/{len(dropouts)}" for name, value in fate_counts.items()) + ".")
        for state in ("Low", "Middle"):
            group = dropouts.loc[dropouts.historical_state.eq(state)]
            counts = group.oct7_fate.value_counts()
            sections.append(f"- **{state} Oct 7 fate** ({len(group)} Oct 6 dropouts): " + "; ".join(f"`{name}` {int(value)}/{len(group)}" for name, value in counts.items()) + ".")
    else:
        sections.append("There were no Oct 6 graph observation gaps to follow to Oct 7.")
    gap = audit.loc[analysis_states & audit.oct6_graph_gap_oct7_observed.astype(bool)]
    gap_by_state = audit.loc[analysis_states & audit.oct6_graph_gap_oct7_observed.astype(bool)].groupby("historical_state").size().to_dict()
    sections.append(f"One-session graph gaps in the Low/Middle cohort (no Oct 6 source-track observation, same source track present Oct 7): **{len(gap)}**; Low {int(gap_by_state.get('Low', 0))}/{int((audit.historical_state.eq('Low')).sum())}, Middle {int(gap_by_state.get('Middle', 0))}/{int((audit.historical_state.eq('Middle')).sum())}. A graph reappearance alone is not biological recovery; its Oct 6 mask and fixed-region signal classifications remain the relevant evidence.")
    sections.append("")
    segmentation_count = int((analysis_states & audit.next_primary_outcome.eq("NO_MASK_BUT_RED_SIGNAL_REMAINS")).sum())
    disappearance_count = int((analysis_states & audit.next_primary_outcome.eq("REPORTER_OBJECT_DISAPPEARANCE")).sum())
    sections.append(f"Within the Low/Middle cohort, the classification found **{segmentation_count}** segmentation-failure/Red-retained cases and **{disappearance_count}** conservative reporter-object disappearance candidates; both require local image coverage and NCC QC. The reporter-object category does not establish cell death.")
    sections.append("")
    sections.append("## Validation checks")
    sections.append("")
    sections.append("| Session | Canonical components | Unfiltered components | QC min_size=100 components | Expected unfiltered |")
    sections.append("|---|---:|---:|---:|---:|")
    for _, row in validation.iterrows():
        canonical = "n/a" if pd.isna(row.canonical_component_count) else int(row.canonical_component_count)
        unfiltered = "n/a" if pd.isna(row.unfiltered_component_count) else int(row.unfiltered_component_count)
        qc_count = "missing" if pd.isna(row.qc_min_size_100_component_count) else int(row.qc_min_size_100_component_count)
        expected = "n/a" if pd.isna(row.expected_unfiltered_count) else int(row.expected_unfiltered_count)
        sections.append(f"| {row.session} | {canonical} | {unfiltered} | {qc_count} | {expected} |")
    sections.append("")
    if len(proof) and mouse_id == "Fucci-Dead_3":
        sections.append("The known approximately 101→93→large rebound pattern was automatically recovered from overlap and size criteria (no label ID was hard-coded):")
        sections.append("")
        for _, row in proof.iterrows():
            t7 = row.third_canonical_best_target_voxels if pd.notna(row.third_canonical_best_target_voxels) else row.third_unfiltered_best_target_voxels
            sections.append(f"- Source label {int(row.source_label)}: {int(row.source_voxel_count)} Oct 5 voxels → {int(row.next_unfiltered_best_target_voxels)} plausible Oct 6 unfiltered voxels → {int(t7)} Oct 7 target voxels; state `{row.historical_state}`, Oct 6 outcome `{row.next_primary_outcome}`.")
            if row.next_primary_outcome == "NOT_ASSESSABLE":
                sections.append(f"  The pattern is automatically detected, but is excluded from the primary min-size count because Oct 6 is not assessable (`{row.next_assessability_reason}`; coverage {row.next_transform_coverage_fraction:.1%}, target-boundary flag {bool(row.next_target_boundary_flag)}, local NCC {row.next_local_registration_ncc:.3f}). It remains an observed rebound candidate for manual review.")
            elif row.next_primary_outcome == "MIN_SIZE_FILTER_DROPOUT":
                sections.append("  This rebound is included in the primary min-size dropout count based on the registered sub-100-voxel Oct 6 component meeting the stricter overlap rule; its low local NCC does not block mask-based correspondence because geometry and coverage pass.")
    elif mouse_id == "Fucci-Dead_3":
        sections.append("The automatic search did not find a row meeting source 90–115, Oct 6 unfiltered 80–99, and Oct 7 plausible target ≥150 voxel criteria. The proof case needs review against the supplied candidate table or a different unfiltered-mask build.")
    else:
        sections.append("The pre-specified Dead_3 subthreshold-rebound proof case is not an acceptance criterion for this mouse; all local min-size calls use registered overlap and are listed in `min_size_dropout_cases.csv`.")
    sections += ["", "The last two columns give each outcome's composition among all graph-unobserved ROIs in that state; these denominators directly answer the root-cause fraction question. Graph identity failure conditional on a plausible canonical target is reported in `outcome_summary_by_state.csv` using its explicit canonical-target denominator. Strict-overlap sensitivity results are in `overlap_sensitivity_summary.csv`, and the signal-retention sensitivity classification is in `roi_outcome_audit.csv`. Exact source/target counts are in `canonical_unfiltered_consistency.csv`.", "", "## Manual review and validation candidates", ""]
    review = review_shortlist.copy()
    if len(review):
        if "modal_slope" in state_fit and mouse_id == "Fucci-Dead_3" and (output / "figures/representative_contact_sheets_original_OLS_state.png").exists():
            sections.append("The explicit stratified shortlist in `manual_review_shortlist.csv` was regenerated with the corrected modal state labels. The carried-over contact sheet is retained as `figures/representative_contact_sheets_original_OLS_state.png`; its old state captions were not recomputed or used for this corrected shortlist. Do not select solely from graph absence.")
        else:
            sections.append("The modal-state stratified shortlist is in `manual_review_shortlist.csv` and the matching contact sheet. For ex-vivo validation start with assessable Low reporter-object candidates with high local NCC and no plausible canonical or unfiltered target; also review subthreshold cases and matched Middle comparators. Do not select solely from graph absence.")
        sections.append("")
        sections.append("| State | Source label | Outcome | Source voxels | Z | Corrected Red | Local NCC | Target Red retention |")
        sections.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for _, row in review.head(12).iterrows():
            sections.append(f"| {row.historical_state} | {int(row.source_label)} | {row.next_primary_outcome} | {int(row.source_voxel_count)} | {row.source_historical_z:.2f} | {row.source_red_corrected:.1f} | {row.next_local_registration_ncc:.3f} | {row.next_red_retention_ratio:.2f} |")
    else:
        sections.append("No assessable dropout cases met the manual review shortlist categories.")
    sections += ["", "## Interpretation and production recommendations", "",
        "- Keep canonical `min_size=100` unchanged unless segmentation-quality validation independently supports a change.",
        "- Preserve unfiltered/sub-100 components as shadow tracking/QC candidates. The audit only promotes a min-size explanation when registered overlap supports it.",
        "- Consider allowing identity continuity through a subthreshold candidate only with image evidence and a confidence/QC flag; do not add that object to the canonical ROI population.",
        "- Emit state-dependent observability flags only if they are descriptive and do not redefine cell identity or biological fate.",
        "- Pair graph disappearance with fixed-region Red/Green measurements. A tracking gap alone is neither recovery nor death.",
        "- This audit cannot establish cell death, degeneration, or pG1 identity. The existing masks are Red-channel Cellpose objects, not validated whole-cell or nuclear boundaries.",
        "",
        "No production pipeline files or canonical masks, transforms, graph outputs, manifests, or segmentation products were modified.",
        "",
        "## Figures",
        "",
    ]
    for path in plot_paths:
        sections.append(f"- `{Path(path).relative_to(output)}`")
    sections += ["", "## Output and reproducibility", "",
        f"All audit outputs are under `{output}`. Run metadata and SHA-256 input hashes are in `audit_manifest.json`; all source data were opened read-only. The audit command and test results are recorded in the enclosing Codex response.",
        ""]
    path = output / "AUDIT_REPORT.md"
    report_text = "\n".join(sections)
    report_text = report_text.replace("Fucci-Dead_3", mouse_id).replace("Dead_3", mouse_id)
    report_text = report_text.replace("Oct 5", dates[0]).replace("Oct 6", dates[1]).replace("Oct 7", dates[2])
    report_text = report_text.replace(
        "Oct 2 was not used as an identity anchor:",
        "Earlier sessions were not used as identity anchors:",
    )
    path.write_text(report_text, encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", required=True)
    parser.add_argument("--mouse-id", required=True)
    parser.add_argument("--start-session", required=True)
    parser.add_argument("--next-session", required=True)
    parser.add_argument("--third-session", required=True)
    parser.add_argument("--unfiltered-mask-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--laser-nm", type=int, default=1050)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(
        project_config=args.project_config, mouse_id=args.mouse_id,
        start_session=args.start_session, next_session=args.next_session,
        third_session=args.third_session, unfiltered_mask_root=args.unfiltered_mask_root,
        output_dir=args.output_dir, laser_nm=args.laser_nm, run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
