"""Measure saved image-affine fixed regions for full-series entry events.

Only existing TIFF images, canonical masks, and saved transforms are read.
Unknown correspondence remains unknown when no direct saved transform exists.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from audit_state_dependent_tracking_dropout import (
    _label_boxes, _load_transform, _local_measurement, _source_coordinates,
    propagate_source_roi,
)

ROOT = Path("/mnt/d/_data/_newAAV_2026/_qc/fucci_cross_mouse_final_audit_20261013/event_centered")
DERIVATIVES = Path("/mnt/d/_data/_newAAV_2026/molecular_tracking_derivatives")


def enrich(frame: pd.DataFrame, runs: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    frames = []
    stats = {"measured": 0, "missing_transform": 0, "registration_qc_failed": 0,
             "no_followup": 0,
             "poor_coverage": 0}
    for mouse, mouse_rows in frame.groupby("mouse_id", sort=False):
        matching_dir = Path(runs[mouse]) / "matching"
        transforms = pd.read_csv(matching_dir / "pairwise_transforms.csv")
        registration = pd.read_csv(matching_dir / "pairwise_registration_qc.csv")
        mouse_rows = mouse_rows.copy()
        for ahead in (1, 2):
            prefix = f"plus{ahead}"
            for (entry_date, target_date), group in mouse_rows.groupby(
                    ["entry_session", f"{prefix}_session"], dropna=False, sort=False):
                if pd.isna(target_date) or str(target_date) == "" or pd.isna(entry_date):
                    stats["no_followup"] += len(group)
                    continue
                entry_date, target_date = str(int(entry_date)), str(int(target_date))
                if (pd.Timestamp(target_date)-pd.Timestamp(entry_date)).days > 7:
                    stats["no_followup"] += len(group)
                    continue
                source_id = f"WT_{mouse}_{entry_date}"
                target_id = f"WT_{mouse}_{target_date}"
                qc = registration.loc[registration.day_a.astype(str).eq(source_id)
                                      & registration.day_b.astype(str).eq(target_id)]
                stage = str(qc.selected_registration_stage.iloc[0]) if len(qc) == 1 else "MISSING_QC"
                mouse_rows.loc[group.index, f"{prefix}_fixed_registration_stage"] = stage
                qc_pass = False
                if len(qc) == 1:
                    qc_field = {"image_affine_local": "local_qc_pass", "image_affine": "affine_qc_pass"}.get(stage)
                    value = qc.iloc[0].get(qc_field, False) if qc_field else (stage == "legacy")
                    qc_pass = False if pd.isna(value) else (str(value).strip().lower() in {"true", "1", "yes"})
                if stage not in {"image_affine_local", "image_affine", "legacy"} or not qc_pass:
                    mouse_rows.loc[group.index, f"{prefix}_fixed_region_status"] = "NO_VALID_SELECTED_REGISTRATION"
                    stats["registration_qc_failed"] += len(group)
                    continue
                try:
                    _, transform, _ = _load_transform(transforms, source_id, target_id, registration)
                except ValueError:
                    mouse_rows.loc[group.index, f"{prefix}_fixed_region_status"] = "NO_SAVED_DIRECT_TRANSFORM"
                    stats["missing_transform"] += len(group)
                    continue
                base = DERIVATIVES / mouse / "sessions"
                source_mask = tifffile.imread(base / entry_date / "1050/segmentation/mask.tif")
                target_mask = tifffile.imread(base / target_date / "1050/segmentation/mask.tif")
                target_red = tifffile.imread(base / target_date / "1050/preprocessing/red.tif")
                target_green = tifffile.imread(base / target_date / "1050/preprocessing/green.tif")
                boxes = _label_boxes(source_mask)
                for idx, row in group.iterrows():
                    coords = _source_coordinates(source_mask, int(row.entry_label), boxes)
                    mapped, coverage = propagate_source_roi(coords, transform, target_mask.shape)
                    mouse_rows.at[idx, f"{prefix}_fixed_region_coverage"] = coverage
                    mouse_rows.at[idx, f"{prefix}_fixed_region_voxels"] = len(mapped)
                    if coverage < .8 or len(mapped) < 20:
                        mouse_rows.at[idx, f"{prefix}_fixed_region_status"] = "INSUFFICIENT_COVERAGE"
                        stats["poor_coverage"] += 1
                        continue
                    for channel, image in (("red", target_red), ("green", target_green)):
                        measure = _local_measurement(image, mapped, target_mask)
                        for metric in ("mean", "median", "q90", "net_q90"):
                            mouse_rows.at[idx, f"{prefix}_{channel}_fixed_{metric}"] = measure[metric]
                    values = target_mask[tuple(mapped.T)]
                    mouse_rows.at[idx, f"{prefix}_fixed_canonical_overlap_voxels"] = int(np.count_nonzero(values))
                    mouse_rows.at[idx, f"{prefix}_fixed_region_status"] = "MEASURED_SELECTED_TRANSFORM"
                    stats["measured"] += 1
                del source_mask, target_mask, target_red, target_green
        frames.append(mouse_rows)
        print(f"{mouse}: fixed-region enrichment complete", flush=True)
    enriched = pd.concat(frames).sort_index()
    return enriched, stats


def plot() -> None:
    events = pd.read_csv(ROOT / "low_entry_events.csv", low_memory=False)
    controls = pd.read_csv(ROOT / "matched_middle_controls.csv", low_memory=False)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, channel in zip(axes, ("red", "green")):
        for frame, label, color in ((events.loc[events.entry_type.str.contains("Low")],
                                     "Low entry", "#b84746"),
                                    (controls.loc[controls.matched_to_entry_type.str.contains("Low")],
                                     "matched Middle", "#3e7290")):
            by_mouse = []
            for mouse, group in frame.groupby("mouse_id"):
                values = [pd.to_numeric(group[f"plus{ahead}_{channel}_fixed_mean"],
                                        errors="coerce").median() for ahead in (1, 2)]
                by_mouse.append(values)
                ax.plot([1, 2], values, color=color, alpha=.18, lw=.7)
            if by_mouse:
                median = np.nanmedian(np.asarray(by_mouse, dtype=float), axis=0)
                ax.plot([1, 2], median, marker="o", color=color, lw=2.2,
                        label=f"{label}: median of mice")
        ax.set_ylabel(f"Fixed region {channel.title()} mean")
        ax.legend()
    axes[-1].set_xticks([1, 2], ["first follow-up", "second follow-up"])
    fig.suptitle("Saved image-affine fixed regions; each thin line is one mouse")
    fig.tight_layout()
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "fixed_region_followup_signal.png", dpi=160)
    plt.close(fig)


def main() -> None:
    provenance = json.loads((ROOT / "provenance.json").read_text())
    names = ("low_entry_events.csv", "matched_middle_controls.csv")
    input_frames = []
    for name in names:
        data = pd.read_csv(ROOT / name, low_memory=False)
        data["__source_name"] = name
        data["__source_row"] = np.arange(len(data))
        input_frames.append(data)
    enriched, combined_stats = enrich(pd.concat(input_frames, ignore_index=True),
                                      provenance["graph_runs"])
    stats = {"combined": combined_stats}
    for name in names:
        subset = enriched.loc[enriched.__source_name.eq(name)].sort_values("__source_row")
        stats[name] = {
            "measured": sum(int(subset[f"plus{ahead}_fixed_region_status"].eq(
                "MEASURED_SAVED_AFFINE").sum()) for ahead in (1, 2)),
            "n_event_or_control_rows": len(subset),
        }
        subset.drop(columns=["__source_name", "__source_row"]).to_csv(ROOT / name, index=False)
    plot()
    stats["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stats["source_run_dirs"] = provenance["graph_runs"]
    (ROOT / "fixed_region_measurement_summary.json").write_text(
        json.dumps(stats, indent=2) + "\n")
    print(stats, flush=True)


if __name__ == "__main__":
    main()
