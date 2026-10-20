"""Add observed Low-entry and native/fixed-cohort composition summaries to an audit.

The backward cohort maps later canonical ROI voxels to the earlier image with
the saved registration transform. It is a measurement cohort, not identity
matching or proof that a neuron existed before it became detectable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from audit_state_dependent_tracking_dropout import _fit_historical_state, _load_transform


def _median(values) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def _slope(red, green) -> float:
    x, y = np.asarray(red, dtype=float), np.asarray(green, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 3 or np.std(x[valid]) <= 0:
        return float("nan")
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def low_entry_fate(audit: pd.DataFrame, native: pd.DataFrame,
                   features: pd.DataFrame, mouse: str, dates: list[str],
                   session_ids: dict[str, str]) -> pd.DataFrame:
    next_id = session_ids[dates[1]]
    next_state, _ = _fit_historical_state(native, next_id)
    next_state = next_state.drop_duplicates("mask_label").set_index("mask_label")
    next_features = features.loc[features.session_id.astype(str).eq(next_id)]
    next_sizes = next_features.set_index("label").area_voxels.to_dict()
    rows = []
    for row in audit.itertuples(index=False):
        if str(row.next_primary_outcome) != "TRACKED_OK":
            continue
        label = pd.to_numeric(getattr(row, "graph_next_target_label"), errors="coerce")
        if pd.isna(label) or int(label) not in next_state.index:
            continue
        target = next_state.loc[int(label)]
        source_state = str(row.historical_state)
        target_state = str(target.historical_state)
        if target_state == "Low" and source_state != "Low":
            entry = "Middle_to_Low" if source_state == "Middle" else "nonLow_to_Low"
        elif source_state == "Middle" and target_state == "Middle":
            entry = "Middle_to_Middle_control"
        else:
            continue
        rows.append({
            "mouse_id": mouse, "source_session": dates[0], "entry_session": dates[1],
            "fate_session": dates[2], "source_label": int(row.source_label),
            "entry_label": int(label), "entry_type": entry,
            "source_state": source_state, "entry_state": target_state,
            "source_z": row.source_historical_z, "entry_z": target.historical_z,
            "source_green_corrected": row.source_green_corrected,
            "entry_green_corrected": target.green,
            "source_red_corrected": row.source_red_corrected,
            "entry_red_corrected": target.red,
            "source_green_red_ratio": row.source_green_corrected / row.source_red_corrected
            if row.source_red_corrected > 0 else np.nan,
            "entry_green_red_ratio": target.green / target.red if target.red > 0 else np.nan,
            "source_voxels": row.source_voxel_count,
            "entry_voxels": next_sizes.get(int(label), np.nan),
            "entry_fragmentation_flag": row.next_ambiguity_flag,
            "entry_graph_observed": row.next_graph_observed,
            "third_graph_observed": row.third_graph_observed,
            "third_primary_outcome": row.third_primary_outcome,
            "third_red_retention_ratio": row.third_red_retention_ratio,
            "third_canonical_plausible": row.third_canonical_plausible,
            "third_unfiltered_plausible": row.third_unfiltered_plausible,
        })
    return pd.DataFrame(rows)


def _backward_cohort(source_mask: np.ndarray, source_red: np.ndarray,
                     source_green: np.ndarray, target_mask: np.ndarray,
                     target_to_source, target_native: pd.DataFrame) -> dict:
    boxes = ndi.find_objects(target_mask)
    measures = []
    for row in target_native.itertuples(index=False):
        label = int(row.mask_label)
        if label < 1 or label > len(boxes) or boxes[label - 1] is None:
            continue
        box = boxes[label - 1]
        coords = np.argwhere(target_mask[box] == label)
        coords += np.array([axis.start for axis in box], dtype=int)
        transformed = np.rint(target_to_source.apply(coords.astype(float))).astype(int)
        inside = np.ones(len(transformed), dtype=bool)
        for axis, size in enumerate(source_mask.shape):
            inside &= (transformed[:, axis] >= 0) & (transformed[:, axis] < size)
        if inside.mean() < 0.90:
            continue
        mapped = np.unique(transformed[inside], axis=0)
        values = tuple(mapped.T)
        measures.append({
            "earlier_red_mean": float(np.mean(source_red[values])),
            "earlier_green_mean": float(np.mean(source_green[values])),
            "later_red_native": float(row.red_raw),
            "later_green_native": float(row.green_raw),
            "earlier_mask_overlap_fraction": float(np.mean(source_mask[values] != 0)),
        })
    frame = pd.DataFrame(measures)
    if frame.empty:
        return {"n": 0}
    return {
        "n": int(len(frame)),
        "red_median": _median(frame.earlier_red_mean),
        "green_median": _median(frame.earlier_green_mean),
        "ratio_median": _median(frame.earlier_green_mean / frame.earlier_red_mean.replace(0, np.nan)),
        "slope_green_on_red": _slope(frame.earlier_red_mean, frame.earlier_green_mean),
        "earlier_mask_detected_fraction": float((frame.earlier_mask_overlap_fraction >= 0.15).mean()),
        "earlier_mask_missing_count": int((frame.earlier_mask_overlap_fraction < 0.15).sum()),
        "later_red_native_median": _median(frame.later_red_native),
        "later_green_native_median": _median(frame.later_green_native),
    }


def population_composition(audit: pd.DataFrame, native: pd.DataFrame,
                           transforms: pd.DataFrame, mouse: str, dates: list[str],
                           derivatives_root: Path,
                           session_ids: dict[str, str],
                           registration_qc: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    source_id = session_ids[dates[0]]
    source_dir = derivatives_root / mouse / "sessions" / dates[0] / "1050"
    source_mask = tifffile.imread(source_dir / "segmentation" / "mask.tif")
    source_red = tifffile.imread(source_dir / "preprocessing" / "red.tif")
    source_green = tifffile.imread(source_dir / "preprocessing" / "green.tif")
    for date in dates:
        session = native.loc[native.session_id.astype(str).eq(session_ids[date])]
        rows.append({
            "mouse_id": mouse, "source_session": dates[0], "measurement_session": date,
            "cohort": "native_per_session", "n": int(len(session)),
            "red_median": _median(session.red_raw),
            "green_median": _median(session.green_raw),
            "ratio_median": _median(session.green_raw / session.red_raw.replace(0, np.nan)),
            "slope_green_on_red": _slope(session.red_raw, session.green_raw),
            "canonical_detected_fraction": 1.0,
            "canonical_missing_count": 0,
        })
    rows.append({
        "mouse_id": mouse, "source_session": dates[0], "measurement_session": dates[0],
        "cohort": "forward_fixed_source", "n": int(len(audit)),
        "red_median": _median(audit.source_red_mean),
        "green_median": _median(audit.source_green_mean),
        "ratio_median": _median(audit.source_green_mean / audit.source_red_mean.replace(0, np.nan)),
        "slope_green_on_red": _slope(audit.source_red_mean, audit.source_green_mean),
        "canonical_detected_fraction": 1.0, "canonical_missing_count": 0,
    })
    for date, prefix in ((dates[1], "next"), (dates[2], "third")):
        good = audit.loc[(pd.to_numeric(audit[f"{prefix}_transform_coverage_fraction"], errors="coerce") >= 0.90)
                         & ~audit[f"{prefix}_target_boundary_flag"].astype(bool)].copy()
        detected = good[f"{prefix}_canonical_plausible"].astype(bool)
        rows.append({
            "mouse_id": mouse, "source_session": dates[0], "measurement_session": date,
            "cohort": "forward_fixed_source", "n": int(len(good)),
            "red_median": _median(good[f"{prefix}_red_mean"]),
            "green_median": _median(good[f"{prefix}_green_mean"]),
            "ratio_median": _median(good[f"{prefix}_green_mean"] / good[f"{prefix}_red_mean"].replace(0, np.nan)),
            "slope_green_on_red": _slope(good[f"{prefix}_red_mean"], good[f"{prefix}_green_mean"]),
            "canonical_detected_fraction": float(detected.mean()) if len(good) else np.nan,
            "canonical_missing_count": int((~detected).sum()),
        })
        target_id = session_ids[date]
        _, _, target_to_source = _load_transform(transforms, source_id, target_id, registration_qc)
        target_dir = derivatives_root / mouse / "sessions" / date / "1050"
        target_mask = tifffile.imread(target_dir / "segmentation" / "mask.tif")
        target_native = native.loc[native.session_id.astype(str).eq(target_id)]
        backward = _backward_cohort(source_mask, source_red, source_green,
                                    target_mask, target_to_source, target_native)
        rows.append({
            "mouse_id": mouse, "source_session": dates[0], "measurement_session": date,
            "cohort": "backward_later_detected_at_source", "n": backward["n"],
            "red_median": backward.get("red_median", np.nan),
            "green_median": backward.get("green_median", np.nan),
            "ratio_median": backward.get("ratio_median", np.nan),
            "slope_green_on_red": backward.get("slope_green_on_red", np.nan),
            "canonical_detected_fraction": backward.get("earlier_mask_detected_fraction", np.nan),
            "canonical_missing_count": backward.get("earlier_mask_missing_count", np.nan),
            "later_red_native_median": backward.get("later_red_native_median", np.nan),
            "later_green_native_median": backward.get("later_green_native_median", np.nan),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--derivatives-root", type=Path,
                        default=Path("/mnt/d/_data/_newAAV_2026/molecular_tracking_derivatives"))
    args = parser.parse_args()
    output = args.audit_dir
    manifest = json.loads((output / "audit_manifest.json").read_text())
    mouse, dates = manifest["mouse_id"], manifest["sessions"]
    run = Path(manifest["run_dir"])
    audit = pd.read_csv(output / "roi_outcome_audit.csv", low_memory=False)
    native = pd.read_csv(run / "extraction" / "matched_session_population_roi_metrics.csv", low_memory=False)
    features = pd.read_csv(run / "matching" / "roi_features.csv", low_memory=False)
    transforms = pd.read_csv(run / "matching" / "pairwise_transforms.csv", low_memory=False)
    registration_path = run / "matching" / "pairwise_registration_qc.csv"
    registration_qc = pd.read_csv(registration_path, low_memory=False) if registration_path.is_file() else None
    session_ids = {str(value)[-8:]: str(value) for value in native.session_id.astype(str).unique()}
    if any(date not in session_ids for date in dates):
        raise ValueError(f"Missing native session IDs for {dates}")
    entries = low_entry_fate(audit, native, features, mouse, dates, session_ids)
    entries.to_csv(output / "low_entry_fate.csv", index=False)
    composition = population_composition(audit, native, transforms, mouse, dates,
                                         args.derivatives_root, session_ids, registration_qc)
    composition.to_csv(output / "population_composition_summary.csv", index=False)
    print(f"{mouse}: {len(entries)} entry/control rows, {len(composition)} composition rows", flush=True)


if __name__ == "__main__":
    main()
