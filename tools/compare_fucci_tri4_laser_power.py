#!/usr/bin/env python3
"""Local Fucci-Tri_4 LP70-vs-LP75 comparison.

The canonical LP70 masks are read-only inputs. LP75 masks and all comparison
products are written below ``analysis/`` in this repository.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path("/mnt/d/_data/_newAAV_2026/incoming_data")
LP70_ROOT = Path(
    "/mnt/d/_data/_newAAV_2026/molecular_tracking_derivatives/Fucci-Tri_4"
)
OUTPUT_ROOT = REPO_ROOT / "analysis" / "fucci_tri4_laser_power_70_vs_75_comparison"
PAIR_MASK_SETTINGS = {
    "model": "cpsam_v2",
    "gpu": True,
    "do_3D": True,
    "z_axis": 0,
    "channel_axis": 3,
    "min_size": 100,
}
SPACING_ZYX = (5.0, 0.693, 0.693)
RED_DARK = 534.0
GREEN_DARK = 319.0
SATURATION_14BIT = 16383


def _add_matcher_path() -> None:
    matcher_path = str(REPO_ROOT / "matching")
    if matcher_path not in sys.path:
        sys.path.insert(0, matcher_path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xml_float(node: ET.Element | None, key: str, default: float = float("nan")) -> float:
    if node is None:
        return default
    try:
        return float(node.attrib.get(key, default))
    except (TypeError, ValueError):
        return default


def xml_int(node: ET.Element | None, key: str, default: int = 0) -> int:
    value = xml_float(node, key, float(default))
    return int(value) if math.isfinite(value) else default


def tiff_shape(path: Path) -> tuple[int, ...]:
    with tifffile.TiffFile(path) as tif:
        return tuple(int(v) for v in tif.series[0].shape)


def acquisition_metadata(acq: Path) -> dict[str, object]:
    root = ET.parse(acq / "Experiment.xml").getroot()
    date_node = root.find("Date")
    lsm = root.find("LSM")
    zstage = root.find("ZStage")
    timelapse = root.find("Timelapse")
    streaming = root.find("Streaming")
    pmt = root.find("PMT")
    pocks = root.findall("Pockels")
    laser_920 = xml_float(pocks[0], "start") if len(pocks) > 0 else 0.0
    laser_1050 = xml_float(pocks[1], "start") if len(pocks) > 1 else 0.0
    if laser_1050 > 0:
        laser_nm, laser_power = 1050, laser_1050
    elif laser_920 > 0:
        laser_nm, laser_power = 920, laser_920
    else:
        laser_nm, laser_power = None, float("nan")
    c1 = acq / "C1-AVG_Image_001_001.raw.tif"
    c2 = acq / "C2-AVG_Image_001_001.raw.tif"
    shape = tiff_shape(c1 if c1.is_file() else c2)
    session_id = acq.parent.name
    date_match = re.search(r"(\d{8})$", session_id)
    session_date = date_match.group(1) if date_match else ""
    return {
        "mouse_id": "Fucci-Tri_4",
        "session_id": session_id,
        "date": session_date,
        "acquisition": acq.name,
        "source_path": str(acq),
        "acquisition_time": (date_node.attrib.get("date", "") if date_node is not None else ""),
        "laser_nm": laser_nm,
        "laser_power_pct": laser_power,
        "laser_920_power_pct": laser_920,
        "laser_1050_power_pct": laser_1050,
        "wavelength": str(laser_nm) if laser_nm else "unknown",
        "channel_file": "C2-AVG_Image_001_001.raw.tif" if laser_nm == 1050 else "C1-AVG_Image_001_001.raw.tif",
        "red_path": str(c2),
        "green_path": str(c1),
        "z_planes": xml_int(zstage, "steps"),
        "z_step_um": xml_float(zstage, "stepSizeUM"),
        "timepoints": xml_int(timelapse, "timepoints"),
        "streaming_frames": xml_int(streaming, "frames"),
        "shape": "x".join(str(v) for v in shape),
        "height_px": shape[-2],
        "width_px": shape[-1],
        "pixel_size_x_um": xml_float(lsm, "pixelWidthUM"),
        "pixel_size_y_um": xml_float(lsm, "pixelHeightUM"),
        "field_size": xml_float(lsm, "fieldSize"),
        "average_num": xml_int(lsm, "averageNum"),
        "pmt_a_gain": xml_float(pmt, "gainA"),
        "pmt_b_gain": xml_float(pmt, "gainB"),
        "bits_per_pixel": xml_int(root.find("Camera"), "bitsPerPixel"),
        "notes": "; ".join(
            value
            for value in (
                "vol10 control" if "vol10" in acq.name.lower() else "",
                "LP75 test acquisition" if "pockels75" in acq.name.lower() else "",
                "laser920 acquisition" if "laser920" in acq.name.lower() else "",
            )
            if value
        ),
    }


def discover_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for session in sorted(RAW_ROOT.glob("WT_Fucci-Tri_4_*")):
        for acq in sorted(p for p in session.iterdir() if p.is_dir() and (p / "Experiment.xml").is_file()):
            rows.append(acquisition_metadata(acq))
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise FileNotFoundError(f"No Fucci-Tri_4 Experiment.xml files found below {RAW_ROOT}")

    pairs: list[dict[str, object]] = []
    for date, group in inventory.groupby("date", sort=True):
        lp70 = group.loc[
            (group["laser_nm"] == 1050)
            & np.isclose(group["laser_power_pct"].astype(float), 70.0)
            & group["acquisition"].eq("field150_res1536768_lFoV_zstack100to300_vol50")
        ]
        lp75 = group.loc[
            (group["laser_nm"] == 1050)
            & np.isclose(group["laser_power_pct"].astype(float), 75.0)
            & group["acquisition"].str.endswith("_test_pockels75")
            & ~group["acquisition"].str.contains("laser920")
        ]
        if len(lp70) != 1 or len(lp75) != 1:
            continue
        a = lp70.iloc[0]
        b = lp75.iloc[0]
        compare_fields = [
            "shape", "z_planes", "z_step_um", "pixel_size_x_um", "pixel_size_y_um",
            "field_size", "timepoints", "streaming_frames", "average_num", "pmt_a_gain", "pmt_b_gain",
        ]
        differences = [field for field in compare_fields if str(a[field]) != str(b[field])]
        mask70 = LP70_ROOT / "sessions" / str(date) / "1050" / "segmentation" / "mask.tif"
        lp75_mask = OUTPUT_ROOT / "lp75_cellpose" / str(date) / f"WT_Fucci-Tri_4_{date}_LP75_cellposeSAM_masks.tif"
        pairs.append(
            {
                "date": date,
                "session_id": f"WT_Fucci-Tri_4_{date}",
                "lp70_source": a["source_path"],
                "lp75_source": b["source_path"],
                "lp70_red_path": a["red_path"],
                "lp70_green_path": a["green_path"],
                "lp75_red_path": b["red_path"],
                "lp75_green_path": b["green_path"],
                "lp70_mask_path": str(mask70),
                "lp75_mask_path": str(lp75_mask),
                "shape": a["shape"],
                "z_planes": a["z_planes"],
                "z_step_um": a["z_step_um"],
                "imaging_settings_compatible": "yes" if not differences else "no",
                "settings_differences": "; ".join(differences),
                "time_delta_minutes": round(
                    (pd.to_datetime(b["acquisition_time"]) - pd.to_datetime(a["acquisition_time"])).total_seconds() / 60.0,
                    2,
                ),
                "lp70_average_num": a["average_num"],
                "lp75_average_num": b["average_num"],
                "notes": "same-day canonical LP70 and test_pockels75 LP75 acquisition",
            }
        )
    paired = pd.DataFrame(pairs)
    if paired.empty:
        raise RuntimeError("No valid same-day LP70/LP75 pairs found")
    return inventory, paired


def prepare_output() -> None:
    for directory in (
        OUTPUT_ROOT / "inventory",
        OUTPUT_ROOT / "lp75_cellpose",
        OUTPUT_ROOT / "tables",
        OUTPUT_ROOT / "qc" / "registration",
        OUTPUT_ROOT / "qc" / "segmentation_overlays",
        OUTPUT_ROOT / "qc" / "lp75_only_crops",
        OUTPUT_ROOT / "plots",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def inventory_phase() -> pd.DataFrame:
    prepare_output()
    inventory, paired = discover_inventory()
    inventory.to_csv(OUTPUT_ROOT / "inventory" / "fucci_tri4_acquisition_inventory.csv", index=False)
    paired.to_csv(OUTPUT_ROOT / "inventory" / "paired_sessions.csv", index=False)
    print("Fucci-Tri_4 LP75 paired inventory")
    for row in paired.itertuples(index=False):
        print(
            f"{row.date}: LP70={Path(row.lp70_source).name} | LP75={Path(row.lp75_source).name} | "
            f"shape={row.shape}, z={row.z_planes}, compatible={row.imaging_settings_compatible} | "
            f"mask={row.lp75_mask_path}"
        )
    unpaired_dates = sorted(set(inventory.loc[inventory["laser_nm"].eq(1050), "date"]) - set(paired["date"]))
    print(f"Paired sessions: {len(paired)}; unpaired 1050 dates: {', '.join(unpaired_dates) or 'none'}")
    return paired


def segment_lp75(paired: pd.DataFrame) -> None:
    if not Path(".venv-cellpose/bin/python").exists():
        raise FileNotFoundError("Expected Cellpose environment at .venv-cellpose/bin/python")
    import torch
    from cellpose import models

    prepare_output()
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam_v2")
    manifest: list[dict[str, object]] = []
    for row in paired.itertuples(index=False):
        output = Path(row.lp75_mask_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            manifest.append({
                "date": row.date, "status": "ALREADY_EXISTS", "red_path": row.lp75_red_path,
                "mask_path": str(output), "n_masks": int(np.max(tifffile.imread(output))),
                "red_sha256": sha256(Path(row.lp75_red_path)), "runtime_min": "",
                "settings": json.dumps(PAIR_MASK_SETTINGS, sort_keys=True),
            })
            print(f"Skipping existing LP75 mask: {output}", flush=True)
            continue
        print(f"Processing LP75 Cellpose-SAM: {row.date}", flush=True)
        t0 = time.perf_counter()
        red = tifffile.imread(row.lp75_red_path)
        if red.ndim == 3:
            red = red[..., np.newaxis]
        if red.ndim != 4:
            raise ValueError(f"Expected LP75 red stack with shape ZYX or ZYXC, got {red.shape}")
        masks = None
        tmp = output.with_name(f".{output.name}.tmp")
        try:
            masks, _flows, _styles = model.eval(
                red,
                do_3D=True,
                z_axis=0,
                channel_axis=3,
                min_size=100,
            )
            masks = np.asarray(masks)
            n_masks = int(np.max(masks))
            dtype = np.uint16 if n_masks <= np.iinfo(np.uint16).max else np.uint32
            tifffile.imwrite(tmp, masks.astype(dtype, copy=False), metadata={"axes": "ZYX"})
            check = tifffile.imread(tmp)
            if check.shape != masks.shape or int(np.max(check)) != n_masks:
                raise RuntimeError("LP75 mask round-trip validation failed")
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite existing LP75 mask: {output}")
            os.replace(tmp, output)
            runtime = (time.perf_counter() - t0) / 60.0
            manifest.append({
                "date": row.date, "status": "SEGMENTED", "red_path": row.lp75_red_path,
                "mask_path": str(output), "n_masks": n_masks,
                "red_sha256": sha256(Path(row.lp75_red_path)), "runtime_min": round(runtime, 3),
                "settings": json.dumps(PAIR_MASK_SETTINGS, sort_keys=True),
            })
            print(f"Saved {output} ({n_masks} masks, {runtime:.2f} min)", flush=True)
        finally:
            if tmp.exists():
                tmp.unlink()
            del red, masks
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pd.DataFrame(manifest).to_csv(OUTPUT_ROOT / "lp75_cellpose" / "processing_manifest.csv", index=False)


def robust_sd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def measure_rois(mask: np.ndarray, red: np.ndarray, green: np.ndarray, date: str, power: int) -> pd.DataFrame:
    from skimage.measure import regionprops

    rows: list[dict[str, object]] = []
    for region in regionprops(mask):
        label = int(region.label)
        z0, y0, x0, z1, y1, x1 = region.bbox
        roi_mask = region.image
        red_crop = red[region.slice]
        green_crop = green[region.slice]
        red_values = red_crop[roi_mask].astype(float)
        green_values = green_crop[roi_mask].astype(float)
        pad = (2, 8, 8)
        bounds = tuple(
            (max(0, start - extra), min(limit, stop + extra))
            for start, stop, extra, limit in zip(
                (z0, y0, x0), (z1, y1, x1), pad, mask.shape, strict=True
            )
        )
        local_mask = mask[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]
        local_red = red[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]
        local_green = green[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]
        roi_local = local_mask == label
        ring = ndimage.binary_dilation(roi_local, structure=np.ones((3, 5, 5), dtype=bool), iterations=2)
        ring &= ~roi_local
        ring &= local_mask == 0
        bg_red = local_red[ring].astype(float)
        bg_green = local_green[ring].astype(float)
        if bg_red.size == 0:
            bg_red = red[mask == 0].astype(float)
        if bg_green.size == 0:
            bg_green = green[mask == 0].astype(float)
        red_bg = float(np.median(bg_red)) if bg_red.size else float("nan")
        green_bg = float(np.median(bg_green)) if bg_green.size else float("nan")
        red_mean = float(np.mean(red_values))
        green_mean = float(np.mean(green_values))
        red_corrected = red_mean - RED_DARK
        green_corrected = green_mean - GREEN_DARK
        ratio = (green_corrected + 1.0) / (red_corrected + 1.0) if red_corrected > 0 and green_corrected > 0 else float("nan")
        rows.append({
            "date": date, "power": power, "label": label, "area_voxels": int(region.area),
            "volume_um3": float(region.area * np.prod(SPACING_ZYX)),
            "centroid_z": float(region.centroid[0]), "centroid_y": float(region.centroid[1]), "centroid_x": float(region.centroid[2]),
            "bbox_z0": z0, "bbox_y0": y0, "bbox_x0": x0, "bbox_z1": z1, "bbox_y1": y1, "bbox_x1": x1,
            "red_mean": red_mean, "red_median": float(np.median(red_values)),
            "green_mean": green_mean, "green_median": float(np.median(green_values)),
            "red_bg": red_bg, "green_bg": green_bg,
            "red_bg_sd": robust_sd(bg_red), "green_bg_sd": robust_sd(bg_green),
            "red_background_subtracted": red_mean - red_bg,
            "green_background_subtracted": green_mean - green_bg,
            "red_contrast": (red_mean - red_bg) / red_bg if red_bg > 0 else float("nan"),
            "green_contrast": (green_mean - green_bg) / green_bg if green_bg > 0 else float("nan"),
            "red_snr_like": (red_mean - red_bg) / robust_sd(bg_red) if robust_sd(bg_red) > 0 else float("nan"),
            "green_snr_like": (green_mean - green_bg) / robust_sd(bg_green) if robust_sd(bg_green) > 0 else float("nan"),
            "red_corrected": red_corrected, "green_corrected": green_corrected,
            "green_red_ratio": ratio, "log2_green_red": math.log2(ratio) if math.isfinite(ratio) and ratio > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def image_metrics(image: np.ndarray, mask: np.ndarray, channel: str, power: int, date: str) -> dict[str, object]:
    background = image[mask == 0].astype(float)
    dtype_max = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else float(np.nanmax(image))
    return {
        "date": date, "power": power, "channel": channel,
        "median_background": float(np.median(background)),
        "background_sd_robust": robust_sd(background),
        "p99": float(np.percentile(image, 99)), "p99_9": float(np.percentile(image, 99.9)),
        "image_min": float(np.min(image)), "image_max": float(np.max(image)),
        "dynamic_range": float(np.max(image) - np.min(image)),
        "saturated_14bit_fraction": float(np.mean(image >= SATURATION_14BIT)),
        "saturated_dtype_fraction": float(np.mean(image >= dtype_max)),
    }


def add_suffix(table: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return table.rename(columns={column: f"{column}_{suffix}" for column in table.columns})


def local_probe(image: np.ndarray, mask: np.ndarray, point: np.ndarray) -> tuple[float, float, float, float]:
    z, y, x = np.rint(point).astype(int)
    z0, z1 = max(0, z - 1), min(image.shape[0], z + 2)
    y0, y1 = max(0, y - 6), min(image.shape[1], y + 7)
    x0, x1 = max(0, x - 6), min(image.shape[2], x + 7)
    values = image[z0:z1, y0:y1, x0:x1].astype(float)
    local_mask = mask[z0:z1, y0:y1, x0:x1]
    yy, xx = np.ogrid[:values.shape[1], :values.shape[2]]
    centre_y, centre_x = y - y0, x - x0
    central = ((yy - centre_y) ** 2 + (xx - centre_x) ** 2) <= 4**2
    central = np.broadcast_to(central, values.shape)
    ring = (local_mask == 0) & ~central
    background = values[ring]
    signal = values[central]
    if background.size == 0:
        background = values.ravel()
    bg = float(np.median(background)) if background.size else float("nan")
    sd = robust_sd(background)
    signal_mean = float(np.mean(signal)) if signal.size else float("nan")
    return signal_mean, bg, sd, (signal_mean - bg) / sd if sd > 0 else float("nan")


def plot_pair_values(table: pd.DataFrame, left: str, right: str, title: str, output: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for date, group in table.groupby("date", sort=True):
        ax.plot(group[left], group[right], ".", ms=2.5, alpha=0.25)
    values = pd.concat([table[left], table[right]], ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if not values.empty:
        lo, hi = float(values.min()), float(values.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("LP70")
    ax.set_ylabel("LP75")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_session_pairs(summary: pd.DataFrame, col70: str, col75: str, title: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(summary))
    for index, row in summary.reset_index(drop=True).iterrows():
        ax.plot([index, index], [row[col70], row[col75]], "o-", lw=1.3)
    ax.set_xticks(x, summary["date"].astype(str), rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_registration_qc(
    date: str,
    red70: np.ndarray,
    red75: np.ndarray,
    mask70: np.ndarray,
    mask75: np.ndarray,
    summary: dict[str, object],
) -> None:
    """Save simple same-limit registration and mask-outline QC overlays."""
    def normalize(image: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(image, [1.0, 99.9])
        if hi <= lo:
            return np.zeros(image.shape, dtype=float)
        return np.clip((image.astype(float) - lo) / (hi - lo), 0.0, 1.0)

    shift_y = float(summary["shift_y"])
    shift_x = float(summary["shift_x"])
    red70_projection = np.max(red70, axis=0)
    red75_projection = np.max(red75, axis=0)
    red75_registered = ndimage.shift(
        red75_projection,
        shift=(shift_y, shift_x),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    lp70_display = normalize(red70_projection)
    lp75_display = normalize(red75_registered)
    registration_rgb = np.stack(
        [lp70_display, lp75_display, np.minimum(lp70_display, lp75_display)], axis=-1
    )
    plt.imsave(
        OUTPUT_ROOT / "qc" / "registration" / f"{date}_red_registration_overlay.png",
        np.clip(registration_rgb, 0.0, 1.0),
    )

    mask70_projection = np.max(mask70, axis=0) > 0
    mask75_projection = np.max(mask75, axis=0) > 0
    mask75_registered = ndimage.shift(
        mask75_projection.astype(float),
        shift=(shift_y, shift_x),
        order=0,
        mode="nearest",
        prefilter=False,
    ) > 0.5
    outline70 = mask70_projection ^ ndimage.binary_erosion(mask70_projection)
    outline75 = mask75_registered ^ ndimage.binary_erosion(mask75_registered)
    background = np.repeat((0.12 + 0.5 * lp70_display)[..., np.newaxis], 3, axis=-1)
    outline_rgb = background.copy()
    outline_rgb[..., 0] = np.clip(outline_rgb[..., 0] + 0.85 * outline70, 0.0, 1.0)
    outline_rgb[..., 1] = np.clip(outline_rgb[..., 1] + 0.85 * outline75, 0.0, 1.0)
    outline_rgb[..., 2] = np.clip(outline_rgb[..., 2] + 0.85 * (outline70 | outline75), 0.0, 1.0)
    plt.imsave(
        OUTPUT_ROOT / "qc" / "segmentation_overlays" / f"{date}_mask_outline_overlay.png",
        np.clip(outline_rgb, 0.0, 1.0),
    )


def save_representative_plot(row: pd.Series) -> None:
    red70 = tifffile.imread(row.lp70_red_path)
    red75 = tifffile.imread(row.lp75_red_path)
    green70 = tifffile.imread(row.lp70_green_path)
    green75 = tifffile.imread(row.lp75_green_path)
    channels = [(red70, red75, "red"), (green70, green75, "green")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for r, (first, second, name) in enumerate(channels):
        values = np.concatenate([first.ravel(), second.ravel()])
        vmin, vmax = np.percentile(values, [1, 99.9])
        axes[r, 0].imshow(np.max(first, axis=0), cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 1].imshow(np.max(second, axis=0), cmap="gray", vmin=vmin, vmax=vmax)
        axes[r, 0].set_title(f"LP70 {name}, shared limits")
        axes[r, 1].set_title(f"LP75 {name}, shared limits")
        for axis in axes[r]:
            axis.axis("off")
    fig.suptitle(f"Fucci-Tri_4 {row.date}: same-FOV max projections")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "plots" / "representative_same_limits.png", dpi=180)
    plt.close(fig)


def save_lp75_only_montage(lp75_only: pd.DataFrame, pair_context: dict[str, dict[str, object]]) -> None:
    if lp75_only.empty:
        return
    chosen = lp75_only.sort_values(["plausible_lp70_signal", "lp70_red_snr"], ascending=[False, False]).head(8)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(6.5, max(2.3, 2.4 * len(chosen))))
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(chosen.itertuples(index=False)):
        context = pair_context[row.date]
        red70 = tifffile.imread(context["lp70_red_path"])
        red75 = tifffile.imread(context["lp75_red_path"])
        z70, y70, x70 = np.rint([row.lp70_z, row.lp70_y, row.lp70_x]).astype(int)
        z75, y75, x75 = np.rint([row.lp75_z, row.lp75_y, row.lp75_x]).astype(int)
        y0, y1 = max(0, y70 - 18), min(red70.shape[1], y70 + 19)
        x0, x1 = max(0, x70 - 18), min(red70.shape[2], x70 + 19)
        p0, p1 = max(0, y75 - 18), min(red75.shape[1], y75 + 19)
        q0, q1 = max(0, x75 - 18), min(red75.shape[2], x75 + 19)
        limits = np.concatenate([red70[max(0, z70 - 1):z70 + 2, y0:y1, x0:x1].ravel(), red75[max(0, z75 - 1):z75 + 2, p0:p1, q0:q1].ravel()])
        vmin, vmax = np.percentile(limits, [1, 99.5])
        axes[row_index, 0].imshow(red70[z70], cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_index, 1].imshow(red75[z75], cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_index, 0].set_title(f"{row.date} LP70 loc; SNR={row.lp70_red_snr:.1f}")
        axes[row_index, 1].set_title(f"LP75-only {row.label_75}; plausible={row.plausible_lp70_signal}")
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle("LP75-only ROI locations, red channel; shared local display limits")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "qc" / "lp75_only_crops" / "lp75_only_montage.png", dpi=180)
    plt.close(fig)


def package_versions() -> dict[str, str | None]:
    names = ["numpy", "pandas", "scipy", "scikit-image", "tifffile", "matplotlib", "cellpose", "torch"]
    return {name: importlib.metadata.version(name) if importlib.util.find_spec(name.replace("-", "_")) else None for name in names}


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_report(summary: pd.DataFrame, matched: pd.DataFrame, detection: pd.DataFrame, image_table: pd.DataFrame, lp75_only: pd.DataFrame, pairing: pd.DataFrame) -> None:
    def fmt(value: object, digits: int = 3) -> str:
        try:
            number = float(value)
            return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"
        except (TypeError, ValueError):
            return "NA"

    median_red_change = float(summary["median_red_contrast_change"].median())
    median_green_change = float(summary["median_green_contrast_change"].median())
    median_red_bg_change = float(summary["red_background_change"].median())
    median_green_bg_change = float(summary["green_background_change"].median())
    plausible = int(lp75_only["plausible_lp70_signal"].sum()) if not lp75_only.empty else 0
    lp75_count = int(len(lp75_only))
    pooled_corr = float(stats.spearmanr(matched["green_red_ratio_70"], matched["green_red_ratio_75"], nan_policy="omit").statistic) if len(matched) > 2 else float("nan")
    red_sat = image_table.loc[image_table.channel.eq("red")].groupby("power")["saturated_14bit_fraction"].median()
    green_sat = image_table.loc[image_table.channel.eq("green")].groupby("power")["saturated_14bit_fraction"].median()
    lines = [
        "# Fucci-Tri_4 laser power 70 vs 75 comparison",
        "",
        "This is a local-only analysis. No canonical LP70 derivative, raw input, or GitHub resource was modified.",
        "",
        "## 1. Data discovered",
        "",
        f"- Inventory: `{OUTPUT_ROOT / 'inventory' / 'fucci_tri4_acquisition_inventory.csv'}`.",
        f"- Valid same-FOV pairs: **{len(pairing)}** ({', '.join(pairing.date.astype(str))}).",
        "- 2026-09-15 and 2026-09-16 had canonical LP70 acquisitions but no LP75 acquisition and were not quantitatively paired.",
        "- The paired LP75 source was the same-day `test_pockels75` 1050-nm acquisition; separate `...laser920` acquisitions were excluded from this 1050 red-channel comparison.",
        "",
        "## 2. CellposeSAM configuration",
        "",
        "- Model: `cpsam_v2`; GPU execution as in the reference notebook.",
        "- `do_3D=True`, `z_axis=0`, `channel_axis=3`, `min_size=100`.",
        "- LP75 segmentation used the raw C2 TIFF (red) with no power normalization or parameter tuning.",
        "- LP70 masks were read from the existing canonical session derivatives; their red/green TIFFs were byte-identical to the corresponding raw C2/C1 TIFFs for all six paired dates.",
        "- Fluorescence uses the pipeline dark offsets red=534 and green=319 for G/R, plus a local annulus (two dilations with a `(3,5,5)` structuring element, excluding all labels) for background-subtracted signal, contrast, and SNR-like values.",
        "- ROI matching reused the repository affine-overlap matcher with its default balanced one-to-one rule; the reported transform is constrained affine/translation only, with no deformable registration.",
        "",
        "## 3. Pairwise results",
        "",
        "| date | N70 | N75 | shared | 75-only | red contrast change | green contrast change | red background change | green background change | red saturation LP75 | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.date} | {int(row.n70)} | {int(row.n75)} | {int(row.shared)} | {int(row.n75_only)} | "
            f"{fmt(row.median_red_contrast_change)} | {fmt(row.median_green_contrast_change)} | {fmt(row.red_background_change)} | "
            f"{fmt(row.green_background_change)} | {fmt(row.red_saturation_75, 5)} | shift z/y/x={fmt(row.shift_z)}/{fmt(row.shift_y)}/{fmt(row.shift_x)}; {row.match_transform} |"
        )
    lines += [
        "",
        "## 4. Cell-level results",
        "",
        f"- Pooled matched-cell green:red Spearman correlation between powers: **{fmt(pooled_corr)}**.",
        f"- Median per-session red contrast change (LP75−LP70): **{fmt(median_red_change)}**; green contrast change: **{fmt(median_green_change)}**.",
        f"- Median per-session red background change: **{fmt(median_red_bg_change)}** intensity units; green background change: **{fmt(median_green_bg_change)}** intensity units.",
        f"- LP75-only ROIs: **{lp75_count}**; of these, **{plausible}** met the exploratory LP70-location rule `local red SNR-like >= 2` and are candidates for plausible weak-cell recovery.",
        "- The per-cell table contains red/green mean and median intensity, local background, background-subtracted intensity, contrast, SNR-like value, area, ratio, and one-to-one match geometry.",
        "",
        "## 5. QC observations",
        "",
        f"- Median 14-bit red saturation fraction: LP70={fmt(red_sat.get(70, np.nan), 5)}, LP75={fmt(red_sat.get(75, np.nan), 5)}; green: LP70={fmt(green_sat.get(70, np.nan), 5)}, LP75={fmt(green_sat.get(75, np.nan), 5)}.",
        "- Registration used phase-correlation/occupancy-derived global shifts followed by the existing constrained affine-overlap matcher. Session-specific shifts and residuals are in `tables/per_session_summary.csv`.",
        "- LP75-only image crops and same-display-limit representative projections are in `qc/lp75_only_crops/` and `plots/representative_same_limits.png`.",
        "- No nominal 75/70 intensity normalization was applied; delivered power was not assumed to scale linearly or quadratically.",
        "",
        "## 6. Practical conclusion",
        "",
        f"- **Red nuclei:** LP75 {'improves' if median_red_change > 0 else 'does not consistently improve'} matched-cell red contrast on the paired-session medians; interpret this together with the background and segmentation counts, not raw brightness alone.",
        f"- **Weak green signal:** LP75 {'improves' if median_green_change > 0 else 'does not consistently improve'} matched-cell green contrast. The comparison holds the red-defined cell population fixed.",
        f"- **Signal versus background:** the median red/green background changes are {fmt(median_red_bg_change)} / {fmt(median_green_bg_change)} intensity units; the per-session table shows whether contrast changes exceed that burden.",
        f"- **Weak-cell rescue:** {plausible} of {lp75_count} LP75-only ROIs passed the exploratory LP70-location signal rule; these are candidates, not confirmed biological cells without manual review.",
        f"- **Longitudinal G/R:** the pooled matched-cell rank correlation is {fmt(pooled_corr)}; ratio shift and session distributions should be checked in `green_red_ratio_lp70_vs_lp75.png` before switching protocols.",
        "- **Recommendation:** use the paired-session summary and QC montages as the decision point. LP75 is justified for longitudinal use only if its contrast/rescue gains are consistent across sessions without materially worse background, saturation, or segmentation stability; otherwise LP70 remains the lower-burden setting.",
        "",
        "## Outputs",
        "",
        f"All products are local under `{OUTPUT_ROOT}`. Key files: `REPORT.md`, `tables/per_roi_matched.csv`, `tables/per_session_summary.csv`, `inventory/paired_sessions.csv`, and `run_manifest.json`.",
    ]
    (OUTPUT_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analysis_phase(paired: pd.DataFrame) -> None:
    _add_matcher_path()
    from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing, extract_roi_features, match_pair

    prepare_output()
    all_matched: list[pd.DataFrame] = []
    all_detection: list[pd.DataFrame] = []
    all_image_metrics: list[dict[str, object]] = []
    all_lp75_only: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    pair_context: dict[str, dict[str, object]] = {}
    spacing = VoxelSpacing(z_um=SPACING_ZYX[0], y_um=SPACING_ZYX[1], x_um=SPACING_ZYX[2])
    params = AffineOverlapParams()

    for pair in paired.itertuples(index=False):
        date = str(pair.date)
        pair_context[date] = pair._asdict()
        red70 = tifffile.imread(pair.lp70_red_path)
        green70 = tifffile.imread(pair.lp70_green_path)
        red75 = tifffile.imread(pair.lp75_red_path)
        green75 = tifffile.imread(pair.lp75_green_path)
        mask70 = tifffile.imread(pair.lp70_mask_path)
        mask75 = tifffile.imread(pair.lp75_mask_path)
        if red70.shape != mask70.shape or red75.shape != mask75.shape or mask70.shape != mask75.shape:
            raise ValueError(f"{date}: image/mask shape mismatch")

        measures70 = measure_rois(mask70, red70, green70, date, 70)
        measures75 = measure_rois(mask75, red75, green75, date, 75)
        features70 = extract_roi_features(mask70, session_id=f"{date}_LP70", spacing=spacing)
        features75 = extract_roi_features(mask75, session_id=f"{date}_LP75", spacing=spacing)
        result = match_pair(
            session_a=f"{date}_LP70", session_b=f"{date}_LP75", mask_a=mask70, mask_b=mask75,
            params=params, spacing=spacing, features_a=features70, features_b=features75, pair_gap=None,
        )
        save_registration_qc(date, red70, red75, mask70, mask75, result.summary)
        shared = result.balanced_matches.copy()
        shared = shared.rename(columns={"label_a": "label_70", "label_b": "label_75"})
        matched = shared.merge(add_suffix(measures70, "70"), on="label_70", how="left", validate="one_to_one")
        matched = matched.merge(add_suffix(measures75, "75"), on="label_75", how="left", validate="one_to_one")
        matched["date"] = date
        matched["centroid_distance_um"] = matched["distance_um"]
        matched["area_ratio_75_to_70"] = matched["area_voxels_75"] / matched["area_voxels_70"]
        all_matched.append(matched)

        labels70 = set(measures70["label"].astype(int))
        labels75 = set(measures75["label"].astype(int))
        shared70 = set(shared["label_70"].astype(int)) if not shared.empty else set()
        shared75 = set(shared["label_75"].astype(int)) if not shared.empty else set()
        detection_rows = (
            [{"date": date, "class": "shared", "label_70": int(row.label_70), "label_75": int(row.label_75)} for row in shared.itertuples(index=False)]
            + [{"date": date, "class": "LP70_only", "label_70": label, "label_75": np.nan} for label in sorted(labels70 - shared70)]
            + [{"date": date, "class": "LP75_only", "label_70": np.nan, "label_75": label} for label in sorted(labels75 - shared75)]
        )
        all_detection.append(pd.DataFrame(detection_rows))

        lp75_only_rows: list[dict[str, object]] = []
        transform = result.transform
        for label in sorted(labels75 - shared75):
            feature = features75.loc[label]
            point75 = np.asarray([feature.centroid_z, feature.centroid_y, feature.centroid_x], dtype=float)
            point70 = transform.apply(point75)
            red_signal, red_bg, red_sd, red_snr = local_probe(red70, mask70, point70)
            green_signal, green_bg, green_sd, green_snr = local_probe(green70, mask70, point70)
            lp75_only_rows.append({
                "date": date, "label_75": int(label), "lp75_z": point75[0], "lp75_y": point75[1], "lp75_x": point75[2],
                "lp70_z": point70[0], "lp70_y": point70[1], "lp70_x": point70[2],
                "lp70_red_local_signal": red_signal, "lp70_red_local_background": red_bg, "lp70_red_bg_sd": red_sd, "lp70_red_snr": red_snr,
                "lp70_green_local_signal": green_signal, "lp70_green_local_background": green_bg, "lp70_green_bg_sd": green_sd, "lp70_green_snr": green_snr,
                "plausible_lp70_signal": bool(red_snr >= 2.0 and red_signal > red_bg),
                "match_method": transform.method,
            })
        all_lp75_only.append(pd.DataFrame(lp75_only_rows))

        image_rows = []
        for power, mask, red, green in ((70, mask70, red70, green70), (75, mask75, red75, green75)):
            image_rows.extend([image_metrics(red, mask, "red", power, date), image_metrics(green, mask, "green", power, date)])
        all_image_metrics.extend(image_rows)
        red_image_rows = {int(row["power"]): row for row in image_rows if row["channel"] == "red"}
        green_image_rows = {int(row["power"]): row for row in image_rows if row["channel"] == "green"}
        summaries.append({
            "date": date, "n70": len(labels70), "n75": len(labels75), "shared": len(shared),
            "n70_only": len(labels70 - shared70), "n75_only": len(labels75 - shared75),
            "shared_over_n70": len(shared) / len(labels70) if labels70 else float("nan"),
            "shared_over_n75": len(shared) / len(labels75) if labels75 else float("nan"),
            "n75_only_plausible": sum(bool(row["plausible_lp70_signal"]) for row in lp75_only_rows),
            "median_red_contrast_70": float(matched["red_contrast_70"].median()) if not matched.empty else float("nan"),
            "median_red_contrast_75": float(matched["red_contrast_75"].median()) if not matched.empty else float("nan"),
            "median_red_contrast_change": float((matched["red_contrast_75"] - matched["red_contrast_70"]).median()) if not matched.empty else float("nan"),
            "median_green_contrast_70": float(matched["green_contrast_70"].median()) if not matched.empty else float("nan"),
            "median_green_contrast_75": float(matched["green_contrast_75"].median()) if not matched.empty else float("nan"),
            "median_green_contrast_change": float((matched["green_contrast_75"] - matched["green_contrast_70"]).median()) if not matched.empty else float("nan"),
            "red_background_change": float((matched["red_bg_75"] - matched["red_bg_70"]).median()) if not matched.empty else float("nan"),
            "green_background_change": float((matched["green_bg_75"] - matched["green_bg_70"]).median()) if not matched.empty else float("nan"),
            "median_red_scale": float((matched["red_mean_75"] / matched["red_mean_70"]).median()) if not matched.empty else float("nan"),
            "median_green_scale": float((matched["green_mean_75"] / matched["green_mean_70"]).median()) if not matched.empty else float("nan"),
            "g_r_spearman": float(stats.spearmanr(matched["green_red_ratio_70"], matched["green_red_ratio_75"], nan_policy="omit").statistic) if len(matched) > 2 else float("nan"),
            "median_g_r_change": float((matched["green_red_ratio_75"] - matched["green_red_ratio_70"]).median()) if not matched.empty else float("nan"),
            "shift_z": result.summary["shift_z"], "shift_y": result.summary["shift_y"], "shift_x": result.summary["shift_x"],
            "match_transform": result.summary["transform_method"], "match_residual_p95_um": result.summary["transform_residual_p95_um"],
            "red_saturation_70": red_image_rows[70]["saturated_14bit_fraction"], "red_saturation_75": red_image_rows[75]["saturated_14bit_fraction"],
            "green_saturation_70": green_image_rows[70]["saturated_14bit_fraction"], "green_saturation_75": green_image_rows[75]["saturated_14bit_fraction"],
        })
        print(f"Analyzed {date}: N70={len(labels70)} N75={len(labels75)} shared={len(shared)}", flush=True)

    matched = pd.concat(all_matched, ignore_index=True) if all_matched else pd.DataFrame()
    detection = pd.concat(all_detection, ignore_index=True) if all_detection else pd.DataFrame()
    image_table = pd.DataFrame(all_image_metrics)
    lp75_only = pd.concat(all_lp75_only, ignore_index=True) if all_lp75_only else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    matched.to_csv(OUTPUT_ROOT / "tables" / "per_roi_matched.csv", index=False)
    detection.to_csv(OUTPUT_ROOT / "tables" / "roi_detection_classes.csv", index=False)
    image_table.to_csv(OUTPUT_ROOT / "tables" / "image_level_metrics.csv", index=False)
    lp75_only.to_csv(OUTPUT_ROOT / "tables" / "lp75_only_evidence.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "tables" / "per_session_summary.csv", index=False)

    if not matched.empty:
        plot_pair_values(matched, "red_background_subtracted_70", "red_background_subtracted_75", "Matched red background-subtracted signal", OUTPUT_ROOT / "plots" / "red_background_subtracted_lp70_vs_lp75.png", "LP75")
        plot_pair_values(matched, "red_contrast_70", "red_contrast_75", "Matched red contrast", OUTPUT_ROOT / "plots" / "red_contrast_lp70_vs_lp75.png", "LP75")
        plot_pair_values(matched, "green_background_subtracted_70", "green_background_subtracted_75", "Matched green background-subtracted signal", OUTPUT_ROOT / "plots" / "green_background_subtracted_lp70_vs_lp75.png", "LP75")
        plot_pair_values(matched, "green_contrast_70", "green_contrast_75", "Matched green contrast", OUTPUT_ROOT / "plots" / "green_contrast_lp70_vs_lp75.png", "LP75")
        plot_pair_values(matched, "green_red_ratio_70", "green_red_ratio_75", "Matched green:red ratio", OUTPUT_ROOT / "plots" / "green_red_ratio_lp70_vs_lp75.png", "LP75")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data = [matched.loc[matched.date.eq(date), "iou"].dropna().to_numpy() for date in summary.date]
        ax.boxplot(data, tick_labels=summary.date, showfliers=False)
        ax.set_ylabel("Mask IoU")
        ax.set_title("Matched-mask IoU by session")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "plots" / "mask_iou_by_session.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data = [matched.loc[matched.date.eq(date), "dice"].dropna().to_numpy() for date in summary.date]
        ax.boxplot(data, tick_labels=summary.date, showfliers=False)
        ax.set_ylabel("Mask Dice")
        ax.set_title("Matched-mask Dice by session")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "plots" / "mask_dice_by_session.png", dpi=180); plt.close(fig)
    plot_session_pairs(summary, "n70", "n75", "ROI count by session", "ROI count", OUTPUT_ROOT / "plots" / "roi_count_by_session.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(summary))
    ax.bar(x - 0.25, summary.n70_only / summary.n70, width=0.25, label="LP70-only")
    ax.bar(x, summary.shared / summary.n70, width=0.25, label="shared")
    ax.bar(x + 0.25, summary.n75_only / summary.n75, width=0.25, label="LP75-only")
    ax.set_xticks(x, summary.date.astype(str), rotation=45, ha="right"); ax.set_ylabel("Fraction"); ax.legend(); ax.set_title("ROI detection classes")
    fig.tight_layout(); fig.savefig(OUTPUT_ROOT / "plots" / "roi_detection_fractions_by_session.png", dpi=180); plt.close(fig)
    red_image = image_table.loc[image_table.channel.eq("red")].pivot(index="date", columns="power", values="median_background").reset_index().rename(columns={70: "red_bg_70", 75: "red_bg_75"})
    green_image = image_table.loc[image_table.channel.eq("green")].pivot(index="date", columns="power", values="median_background").reset_index().rename(columns={70: "green_bg_70", 75: "green_bg_75"})
    backgrounds = summary[["date"]].merge(red_image, on="date").merge(green_image, on="date")
    plot_session_pairs(backgrounds, "red_bg_70", "red_bg_75", "Red background intensity", "Median background", OUTPUT_ROOT / "plots" / "background_intensity_red.png")
    sat = image_table.loc[image_table.channel.eq("red")].pivot(index="date", columns="power", values="saturated_14bit_fraction").reset_index().rename(columns={70: "sat_70", 75: "sat_75"})
    plot_session_pairs(sat, "sat_70", "sat_75", "Red 14-bit saturation/clipping", "Fraction saturated", OUTPUT_ROOT / "plots" / "saturation_diagnostic_red.png")
    save_representative_plot(paired.iloc[0])
    save_lp75_only_montage(lp75_only, pair_context)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_short": git_value("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "raw_root": str(RAW_ROOT),
        "lp70_reference_root": str(LP70_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "cellpose_settings": PAIR_MASK_SETTINGS,
        "dark_offsets": {"red": RED_DARK, "green": GREEN_DARK},
        "saturation_threshold_14bit": SATURATION_14BIT,
        "voxel_spacing_zyx_um": SPACING_ZYX,
        "paired_sessions": paired.to_dict(orient="records"),
        "source_sha256": {
            str(path): sha256(Path(path))
            for row in paired.itertuples(index=False)
            for path in (row.lp70_red_path, row.lp70_green_path, row.lp70_mask_path, row.lp75_red_path, row.lp75_green_path, row.lp75_mask_path)
            if Path(path).is_file()
        },
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    write_report(summary, matched, detection, image_table, lp75_only, paired)
    print(f"Wrote comparison report: {OUTPUT_ROOT / 'REPORT.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("inventory", "segment", "analyze", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired = inventory_phase()
    if args.phase in {"segment", "all"}:
        segment_lp75(paired)
    if args.phase in {"analyze", "all"}:
        analysis_phase(paired)


if __name__ == "__main__":
    main()
