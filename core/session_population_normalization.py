"""Session-native ROI extraction and green-vs-red normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from roi_log_ratio_analysis import (
    compute_log_ratio_metrics,
    extract_roi_mean_intensities,
    summarize_daily_green_red_linear_fits_tolerant,
)


FIT_POPULATION_RULE = (
    "signal-valid Green values with modified_z <= 10 and no conservative upper-tail "
    "outlier above Q3 + 30*IQR; if MAD is zero use that IQR rule alone"
)


def flag_fit_population_artifacts(population: pd.DataFrame) -> pd.DataFrame:
    """Flag conservative, session-local extreme Green values for fitting."""

    output = population.copy()
    output["fit_population_artifact"] = False
    output["fit_population_include"] = output["ratio_qc_pass"].eq(True)
    output["artifact_reason"] = ""
    for _, indexes in output.groupby("day", sort=False).groups.items():
        group = output.loc[indexes]
        valid = group.loc[group["ratio_qc_pass"].eq(True), "green"].astype(float)
        if len(valid) < 3:
            continue
        median = float(valid.median())
        mad = float(np.median(np.abs(valid.to_numpy() - median)))
        q1, q3 = valid.quantile([0.25, 0.75]).tolist()
        iqr = float(q3 - q1)
        upper_iqr = float(q3 + 30.0 * iqr) if iqr > 0 else median
        if mad > 0:
            modified_z = 0.6745 * (valid - median) / mad
            artifact = (modified_z > 10.0) & (valid > upper_iqr)
        else:
            artifact = valid > upper_iqr
        artifact_indexes = valid.index[artifact]
        output.loc[artifact_indexes, "fit_population_artifact"] = True
        output.loc[artifact_indexes, "fit_population_include"] = False
        output.loc[artifact_indexes, "artifact_reason"] = "green_robust_upper_outlier"
    return output


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
    metrics = metrics.drop(columns=["roi_id", "__session_native_roi_key"] + [column for column in metrics if column.startswith("day0_") or "first_observed" in column or column.startswith("delta_")])
    return flag_fit_population_artifacts(metrics)


def fit_session_population(population: pd.DataFrame, records=None) -> pd.DataFrame:
    if "fit_population_include" not in population.columns:
        population = flag_fit_population_artifacts(population)
    fit_population = population.loc[population["fit_population_include"].eq(True)].copy()
    metadata_columns = ["day", "session_index", "session_id", "acquisition_date", "elapsed_days"]
    metadata = population.reindex(columns=metadata_columns).drop_duplicates("day")
    if records is not None:
        required_records = [record for record in records if record.required]
        start_date = min((record.acquisition_date for record in required_records), default=None)
        record_metadata = pd.DataFrame([
            {
                "day": int(record.session_index),
                "session_index": int(record.session_index),
                "session_id": str(record.session_id),
                "acquisition_date": record.acquisition_date.isoformat(),
                "elapsed_days": int((record.acquisition_date - start_date).days) if start_date else np.nan,
            }
            for record in required_records
        ], columns=metadata_columns)
        metadata = record_metadata
    expected_days = metadata["day"].dropna().astype(int).tolist()
    fits = summarize_daily_green_red_linear_fits_tolerant(
        fit_population,
        expected_days=expected_days,
    )
    diagnostics = []
    for day in expected_days:
        group = population.loc[population["day"].eq(day)]
        valid = group.loc[group["fit_population_include"].eq(True)]
        fit = fits.loc[fits["day"].eq(day)].iloc[0]
        if fit["fit_status"] == "valid":
            residuals = valid["green"] - (fit["intercept"] + fit["slope"] * valid["red"])
            residual_mean = float(residuals.mean())
            residual_std = float(residuals.std(ddof=1))
        else:
            residual_mean = np.nan
            residual_std = np.nan
        n_signal_valid = int(group["ratio_qc_pass"].eq(True).sum()) if "ratio_qc_pass" in group else 0
        diagnostics.append({
            "day": int(day), "n_rois_total": int(len(group)),
            "n_rois_signal_valid": n_signal_valid,
            "signal_valid_fraction": float(n_signal_valid / len(group)) if len(group) else np.nan,
            "n_fit_included": int(len(valid)),
            "n_fit_excluded": int(n_signal_valid - len(valid)),
            "fit_population_fraction": float(len(valid) / n_signal_valid) if n_signal_valid else np.nan,
            "residual_mean_fit_population": residual_mean,
            "residual_std_fit_population": residual_std,
        })
    diagnostic_columns = [
        "day", "n_rois_total", "n_rois_signal_valid", "signal_valid_fraction",
        "n_fit_included", "n_fit_excluded", "fit_population_fraction",
        "residual_mean_fit_population", "residual_std_fit_population",
    ]
    fits = fits.merge(pd.DataFrame(diagnostics, columns=diagnostic_columns), on="day", validate="one_to_one")
    fits = fits.merge(metadata, on="day", how="left", validate="one_to_one")
    fits["normalization_population"] = "fit_clean_signal_valid_session_rois"
    fits["fit_population_rule"] = FIT_POPULATION_RULE
    return fits


def summarize_signal_qc(population: pd.DataFrame, records=None) -> pd.DataFrame:
    rows = []
    if records is not None:
        grouped = {
            int(day): group
            for day, group in population.groupby("day", sort=True)
        }
        sessions = []
        required_records = [record for record in records if record.required]
        start_date = min((record.acquisition_date for record in required_records), default=None)
        for record in required_records:
            day = int(record.session_index)
            group = grouped.get(day, population.iloc[0:0])
            sessions.append((record, group, {
                "session_id": str(record.session_id),
                "session_index": day,
                "acquisition_date": record.acquisition_date.isoformat(),
                "elapsed_days": int((record.acquisition_date - start_date).days) if start_date else 0,
            }))
    else:
        sessions = []
        for _, group in population.groupby("day", sort=True):
            row = group.iloc[0]
            sessions.append((None, group, row))
    for _, group, row in sessions:
        n = len(group)
        count = lambda column: int(group[column].fillna(False).sum()) if column in group else 0
        rows.append({
            "session_id": row["session_id"], "session_index": int(row["session_index"]),
            "acquisition_date": row["acquisition_date"], "elapsed_days": int(row["elapsed_days"]),
            "n_native_rois": n,
            "n_red_zero_hit_pass": count("red_zero_hit_pass"),
            "n_green_zero_hit_pass": count("green_zero_hit_pass"),
            "n_red_signal_valid": count("red_signal_qc_pass"),
            "n_green_signal_valid": count("green_signal_qc_pass"),
            "n_ratio_valid": count("ratio_qc_pass"),
            "red_signal_valid_fraction": float(count("red_signal_qc_pass") / n) if n else np.nan,
            "green_signal_valid_fraction": float(count("green_signal_qc_pass") / n) if n else np.nan,
            "ratio_valid_fraction": float(count("ratio_qc_pass") / n) if n else np.nan,
            "n_fit_population": count("fit_population_include"),
            "n_fit_excluded_artifact": count("fit_population_artifact"),
        })
    return pd.DataFrame(rows)
