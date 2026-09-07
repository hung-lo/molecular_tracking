"""Session-native ROI extraction and green-vs-red normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from roi_log_ratio_analysis import compute_log_ratio_metrics, extract_roi_mean_intensities, summarize_daily_green_red_linear_fits


def extract_session_population(records, *, green_dark: float, red_dark: float, epsilon: float) -> pd.DataFrame:
    required = [record for record in records if record.required]
    start_date = min(record.acquisition_date for record in required)
    rows = []
    for record in required:
        mask = tifffile.imread(record.mask_path)
        labels = pd.DataFrame({"mask_label": np.unique(mask[mask > 0]).astype(int)})
        labels["session_index"] = int(record.session_index)
        labels["day"] = int(record.session_index)
        labels["session_id"] = str(record.session_id)
        labels["acquisition_date"] = record.acquisition_date.isoformat()
        labels["elapsed_days"] = int((record.acquisition_date - start_date).days)
        for channel, path, dark in (
            ("red", record.red_image_path, red_dark),
            ("green", record.green_image_path, green_dark),
        ):
            extracted = pd.DataFrame(extract_roi_mean_intensities(tifffile.imread(path), mask, exclude_zero_pixels=True))
            if not extracted.empty:
                extracted = extracted.rename(columns={"roi_id": "mask_label", "mean_intensity": f"{channel}_raw"})
                labels = labels.merge(extracted[["mask_label", f"{channel}_raw"]], on="mask_label", how="left", validate="one_to_one")
            else:
                labels[f"{channel}_raw"] = np.nan
            labels[f"{channel}_dark_value"] = float(dark)
            labels[f"{channel}_zero_hit_pass"] = labels[f"{channel}_raw"].notna()
            labels[channel] = labels[f"{channel}_raw"] - float(dark)
        rows.append(labels)
    wide = pd.concat(rows, ignore_index=True)
    wide["session_roi_label"] = wide["mask_label"]
    wide["__session_native_roi_key"] = wide["session_id"].astype(str) + "::" + wide["mask_label"].astype(str)
    wide["roi_id"] = wide["__session_native_roi_key"]
    metrics = compute_log_ratio_metrics(wide, epsilon=epsilon)
    return metrics.drop(columns=["roi_id", "__session_native_roi_key"] + [column for column in metrics if column.startswith("day0_") or "first_observed" in column or column.startswith("delta_")])


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
    metadata = population[["day", "session_index", "session_id", "acquisition_date", "elapsed_days"]].drop_duplicates("day")
    fits = fits.merge(metadata, on="day", how="left", validate="one_to_one")
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
