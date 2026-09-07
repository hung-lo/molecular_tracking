"""Session-native ROI extraction and green-vs-red normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from roi_log_ratio_analysis import (
    apply_channel_dark_correction,
    compute_log_ratio_metrics,
    extract_roi_mean_intensities,
    summarize_daily_green_red_linear_fits,
    wide_table_from_long_table,
)


def extract_session_population(records, *, green_dark: float, red_dark: float, epsilon: float) -> pd.DataFrame:
    required = [record for record in records if record.required]
    start_date = min(record.acquisition_date for record in required)
    rows = []
    for record in required:
        mask = tifffile.imread(record.mask_path)
        labels = pd.DataFrame({"mask_label": np.unique(mask[mask > 0]).astype(int)})
        for channel, path, dark in (
            ("red", record.red_image_path, red_dark),
            ("green", record.green_image_path, green_dark),
        ):
            extracted = pd.DataFrame(extract_roi_mean_intensities(tifffile.imread(path), mask, exclude_zero_pixels=True))
            extracted = extracted.rename(columns={"roi_id": "mask_label"})
            table = labels.merge(extracted, on="mask_label", how="left", validate="one_to_one")
            table["channel"] = channel
            table["roi_id"] = table["mask_label"]
            table["image"] = path.name
            table["zero_hit_pass"] = table["mean_intensity"].notna()
            table["dark_value"] = float(dark)
            table["session_index"] = int(record.session_index)
            table["day"] = int(record.session_index)
            table["session_id"] = str(record.session_id)
            table["acquisition_date"] = record.acquisition_date.isoformat()
            table["elapsed_days"] = int((record.acquisition_date - start_date).days)
            rows.append(table)
    long = pd.concat(rows, ignore_index=True)
    corrected = apply_channel_dark_correction(
        long, green_dark=green_dark, red_dark=red_dark,
        intensity_column="mean_intensity", corrected_column="mean_intensity_corrected",
        clip_floor=None,
    )
    wide = wide_table_from_long_table(corrected, intensity_column="mean_intensity_corrected")
    metadata = corrected.pivot_table(
        index=["day", "mask_label"], columns="channel",
        values=["mean_intensity", "dark_value", "zero_hit_pass"], aggfunc="first",
    )
    metadata.columns = [f"{channel}_{name if name != 'mean_intensity' else 'raw'}" for name, channel in metadata.columns]
    metadata = metadata.reset_index()
    session_meta = corrected[["day", "session_index", "session_id", "acquisition_date", "elapsed_days"]].drop_duplicates()
    wide = wide.rename(columns={"roi_id": "mask_label"}).merge(metadata, on=["day", "mask_label"]).merge(session_meta, on="day")
    wide["roi_id"] = wide["mask_label"]
    wide["session_roi_label"] = wide["mask_label"]
    metrics = compute_log_ratio_metrics(wide, epsilon=epsilon)
    return metrics.drop(columns=[column for column in metrics if column.startswith("day0_") or "first_observed" in column or column.startswith("delta_")])


def fit_session_population(population: pd.DataFrame) -> pd.DataFrame:
    fits = summarize_daily_green_red_linear_fits(population)
    diagnostics = []
    for day, group in population.groupby("day", sort=True):
        valid = group.loc[group["ratio_qc_pass"].eq(True)]
        fit = fits.loc[fits["day"].eq(day)].iloc[0]
        residuals = valid["green"] - (fit["intercept"] + fit["slope"] * valid["red"])
        diagnostics.append({
            "day": int(day), "n_rois_total": int(len(group)),
            "n_rois_signal_valid": int(len(valid)),
            "signal_valid_fraction": float(len(valid) / len(group)) if len(group) else np.nan,
            "residual_mean_fit_population": float(residuals.mean()),
            "residual_std_fit_population": float(residuals.std(ddof=1)),
        })
    fits = fits.merge(pd.DataFrame(diagnostics), on="day", validate="one_to_one")
    fits["normalization_population"] = "all_valid_session_rois"
    return fits


def summarize_signal_qc(population: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in population.groupby("day", sort=True):
        row = group.iloc[0]
        n = len(group)
        rows.append({
            "session_id": row["session_id"], "session_index": int(row["session_index"]),
            "acquisition_date": row["acquisition_date"], "elapsed_days": int(row["elapsed_days"]),
            "n_native_rois": n,
            "n_red_zero_hit_pass": int(group["red_zero_hit_pass"].sum()),
            "n_green_zero_hit_pass": int(group["green_zero_hit_pass"].sum()),
            "n_red_signal_valid": int(group["red_signal_qc_pass"].sum()),
            "n_green_signal_valid": int(group["green_signal_qc_pass"].sum()),
            "n_ratio_valid": int(group["ratio_qc_pass"].sum()),
            "red_signal_valid_fraction": float(group["red_signal_qc_pass"].mean()),
            "green_signal_valid_fraction": float(group["green_signal_qc_pass"].mean()),
            "ratio_valid_fraction": float(group["ratio_qc_pass"].mean()),
            "n_fit_population": int(group["ratio_qc_pass"].sum()),
        })
    return pd.DataFrame(rows)
