from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def extract_day_from_image_name(image_name: str, start_date: str = "20260511") -> int:
    """Convert an image filename into a day index.

    Parameters
    ----------
    image_name : str
        Filename with an 8-digit date prefix such as ``20260511_R.tif``.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format that defines day 0.

    Returns
    -------
    int
        Integer day offset relative to ``start_date`` in units of days.
    """

    date_key = image_name.split("_")[0]
    image_date = pd.to_datetime(date_key, format="%Y%m%d")
    reference_date = pd.to_datetime(start_date, format="%Y%m%d")
    return int((image_date - reference_date).days)


def wide_table_from_long_table(
    intensity_table: pd.DataFrame,
    intensity_column: str = "mean_intensity_corrected",
    start_date: str = "20260511",
) -> pd.DataFrame:
    """Pivot a long ROI intensity table into one row per ROI and day.

    Parameters
    ----------
    intensity_table : pandas.DataFrame
        Table with columns ``roi_id``, ``image``, ``channel``, and an intensity
        column. Intensities are arbitrary fluorescence units per ROI.
    intensity_column : str, default="mean_intensity_corrected"
        Column name that contains the per-ROI intensity value to analyze.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format for computing day indices when the
        table does not already include a ``day`` column.

    Returns
    -------
    pandas.DataFrame
        Wide table with columns ``roi_id`` (int), ``day`` (int), ``red`` (float),
        and ``green`` (float). Each row represents one ROI on one imaging day.
    """

    required_columns = {"roi_id", "image", "channel", intensity_column}
    missing_columns = required_columns.difference(intensity_table.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    table = intensity_table.copy()
    if "day" not in table.columns:
        table["day"] = table["image"].map(
            lambda image_name: extract_day_from_image_name(
                str(image_name), start_date=start_date
            )
        )

    wide_table = (
        table.pivot_table(
            index=["roi_id", "day"],
            columns="channel",
            values=intensity_column,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
        .sort_values(["roi_id", "day"])
        .reset_index(drop=True)
    )

    return wide_table


def filter_complete_rois(
    roi_day_table: pd.DataFrame,
    required_days: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Keep only ROIs that are present on every required day.

    Parameters
    ----------
    roi_day_table : pandas.DataFrame
        Wide ROI/day table with columns ``roi_id``, ``day``, ``red``, and
        ``green``. Intensities are arbitrary fluorescence units.
    required_days : sequence of int or None, default=None
        Day indices that must be present for a ROI to pass filtering. When
        ``None``, the unique days present in the table are used.

    Returns
    -------
    pandas.DataFrame
        Subset of ``roi_day_table`` containing only ROIs with non-missing red and
        green intensities on every required day. If a ROI contains extra valid
        days beyond ``required_days``, those rows are retained rather than
        dropped.
    """

    required_columns = {"roi_id", "day", "red", "green"}
    missing_columns = required_columns.difference(roi_day_table.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    valid_rows = roi_day_table.dropna(subset=["red", "green"]).copy()
    if required_days is None:
        required_days = sorted(int(day) for day in valid_rows["day"].unique())
    required_day_set = {int(day) for day in required_days}

    days_by_roi = (
        valid_rows.groupby("roi_id")["day"]
        .agg(lambda day_values: {int(day) for day in day_values})
    )
    complete_roi_ids = [
        int(roi_id)
        for roi_id, day_set in days_by_roi.items()
        if required_day_set.issubset(day_set)
    ]

    filtered = valid_rows[valid_rows["roi_id"].isin(complete_roi_ids)].copy()
    filtered = filtered.sort_values(["roi_id", "day"]).reset_index(drop=True)
    return filtered


def compute_log_ratio_metrics(
    roi_day_table: pd.DataFrame,
    epsilon: float = 1.0,
) -> pd.DataFrame:
    """Compute brightness and day-0-normalized green/red metrics.

    Parameters
    ----------
    roi_day_table : pandas.DataFrame
        Wide ROI/day table with columns ``roi_id``, ``day``, ``red``, and
        ``green``. Intensities are arbitrary fluorescence units.
    epsilon : float, default=1.0
        Small positive offset added to both channels before taking the log ratio.
        The offset is in the same intensity units as the input table.

    Returns
    -------
    pandas.DataFrame
        Copy of the input table with added columns:
        ``brightness``, ``green_fraction``, ``log2_green_over_red``,
        ``day0_log2_green_over_red``, and ``delta_log2_green_over_red``.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    metrics = roi_day_table.copy().sort_values(["roi_id", "day"]).reset_index(drop=True)
    metrics["brightness"] = metrics["red"] + metrics["green"]
    metrics["green_fraction"] = np.where(
        metrics["brightness"] > 0,
        metrics["green"] / metrics["brightness"],
        np.nan,
    )
    metrics["log2_green_over_red"] = np.log2(
        (metrics["green"] + epsilon) / (metrics["red"] + epsilon)
    )
    metrics["day0_log2_green_over_red"] = metrics.groupby("roi_id")[
        "log2_green_over_red"
    ].transform("first")
    metrics["delta_log2_green_over_red"] = (
        metrics["log2_green_over_red"] - metrics["day0_log2_green_over_red"]
    )
    return metrics


def extract_roi_mean_intensities(
    image_stack: np.ndarray,
    mask_stack: np.ndarray,
    exclude_zero_pixels: bool = True,
) -> list[dict[str, float | int]]:
    """Extract mean ROI intensities from a 3D image using an integer label mask.

    Parameters
    ----------
    image_stack : numpy.ndarray
        Fluorescence image stack with shape ``(z, y, x)`` in arbitrary
        fluorescence units.
    mask_stack : numpy.ndarray
        Integer ROI label image with the same shape ``(z, y, x)``. Background
        pixels must be labeled as zero.
    exclude_zero_pixels : bool, default=True
        When ``True``, exclude any ROI that contains one or more zero-valued
        image pixels, matching the current zero-exclusion rule for edge-clipped
        registered images.

    Returns
    -------
    list of dict
        One dictionary per retained ROI with keys ``roi_id`` and
        ``mean_intensity``.
    """

    if image_stack.shape != mask_stack.shape:
        raise ValueError("image_stack and mask_stack must have the same shape.")

    from scipy import ndimage

    roi_ids = np.unique(mask_stack)
    roi_ids = roi_ids[roi_ids > 0]

    if exclude_zero_pixels:
        zero_hit_roi_ids = np.unique(mask_stack[np.asarray(image_stack) == 0])
        zero_hit_roi_ids = zero_hit_roi_ids[zero_hit_roi_ids > 0]
        roi_ids = roi_ids[~np.isin(roi_ids, zero_hit_roi_ids)]

    if len(roi_ids) == 0:
        return []

    mean_values = np.atleast_1d(
        ndimage.mean(
            np.asarray(image_stack, dtype=float),
            labels=mask_stack,
            index=roi_ids,
        )
    )

    rows: list[dict[str, float | int]] = []
    for roi_id, mean_value in zip(roi_ids, mean_values, strict=True):
        rows.append(
            {
                "roi_id": int(roi_id),
                "mean_intensity": float(mean_value),
            }
        )
    return rows


def extract_registered_dataset_roi_intensity_table(
    image_dir: str | Path,
    mask_path: str | Path,
    start_date: str,
    channels: tuple[str, ...] = ("red", "green"),
    exclude_zero_pixels: bool = True,
    day0_mode: str = "raw",
) -> pd.DataFrame:
    """Extract a long ROI-intensity table from registered per-day image stacks.

    Parameters
    ----------
    image_dir : str or pathlib.Path
        Directory containing the registered per-day TIFF stacks. The expected
        image shape is ``(z, y, x)`` in arbitrary fluorescence units.
    mask_path : str or pathlib.Path
        Integer ROI label TIFF with shape ``(z, y, x)`` that matches the
        registered image geometry. Background pixels must be labeled as zero.
    start_date : str
        Reference date in ``YYYYMMDD`` format that defines day 0 for the
        current dataset.
    channels : tuple[str, ...], default=("red", "green")
        Channel names to extract. Supported values are ``"red"`` and
        ``"green"``.
    exclude_zero_pixels : bool, default=True
        Whether to drop any ROI that touches a zero-valued pixel in a given
        image stack, matching the current registered-image QC rule.
    day0_mode : {"raw", "syn"}, default="raw"
        Which day-0 image variant to select when both raw and registered files
        exist.

    Returns
    -------
    pandas.DataFrame
        Long-format ROI table with columns ``roi_id`` (int), ``mean_intensity``
        (float, arbitrary fluorescence units), ``image`` (str), ``channel``
        (str), and ``day`` (int). Each row represents one ROI in one image.
    """

    import tifffile

    image_dir = Path(image_dir)
    mask_path = Path(mask_path)
    mask_stack = tifffile.imread(mask_path)
    registered_lookup = build_registered_image_lookup(
        image_dir=image_dir,
        channels=channels,
        start_date=start_date,
        day0_mode=day0_mode,
    )

    rows: list[dict[str, float | int | str]] = []
    for (day, channel), image_path in sorted(registered_lookup.items()):
        image_stack = tifffile.imread(image_path)
        extracted_rows = extract_roi_mean_intensities(
            image_stack=image_stack,
            mask_stack=mask_stack,
            exclude_zero_pixels=exclude_zero_pixels,
        )
        for row in extracted_rows:
            rows.append(
                {
                    "roi_id": int(row["roi_id"]),
                    "mean_intensity": float(row["mean_intensity"]),
                    "image": image_path.name,
                    "channel": str(channel),
                    "day": int(day),
                }
            )

    return pd.DataFrame(rows).sort_values(["roi_id", "day", "channel"]).reset_index(drop=True)


def add_day0_normalized_column(
    roi_metrics: pd.DataFrame,
    source_column: str,
    baseline_column_name: str | None = None,
    normalized_column_name: str | None = None,
) -> pd.DataFrame:
    """Add per-ROI day-0 baseline and day-0-normalized versions of a metric.

    Parameters
    ----------
    roi_metrics : pandas.DataFrame
        ROI/day table that includes ``roi_id`` and the requested source column.
    source_column : str
        Column to normalize relative to each ROI's day-0 value.
    baseline_column_name : str or None, default=None
        Optional name for the added day-0 baseline column. When omitted, the
        name ``<source_column>_day0`` is used.
    normalized_column_name : str or None, default=None
        Optional name for the normalized column. When omitted, the name
        ``<source_column>_normalized_to_day0`` is used.

    Returns
    -------
    pandas.DataFrame
        Copy of ``roi_metrics`` with added baseline and normalized columns. When
        the day-0 baseline is zero, the normalized value is reported as ``NaN``.
    """

    required_columns = {"roi_id", source_column}
    missing_columns = required_columns.difference(roi_metrics.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    baseline_column_name = baseline_column_name or f"{source_column}_day0"
    normalized_column_name = normalized_column_name or f"{source_column}_normalized_to_day0"

    output = roi_metrics.copy().sort_values(["roi_id", "day"]).reset_index(drop=True)
    output[baseline_column_name] = output.groupby("roi_id")[source_column].transform("first")
    output[normalized_column_name] = np.where(
        output[baseline_column_name] > 0,
        output[source_column] / output[baseline_column_name],
        np.nan,
    )
    return output


def apply_channel_dark_correction(
    intensity_table: pd.DataFrame,
    green_dark: float,
    red_dark: float,
    intensity_column: str = "mean_intensity",
    corrected_column: str = "mean_intensity_corrected",
    clip_floor: float | None = None,
) -> pd.DataFrame:
    """Subtract fixed per-channel dark values from a ROI intensity table.

    Parameters
    ----------
    intensity_table : pandas.DataFrame
        ROI intensity table with at least ``channel`` and ``intensity_column``.
        The supported channel labels are ``"green"`` and ``"red"``.
    green_dark : float
        Dark offset to subtract from the green channel in arbitrary fluorescence
        units.
    red_dark : float
        Dark offset to subtract from the red channel in arbitrary fluorescence
        units.
    intensity_column : str, default="mean_intensity"
        Column containing the raw per-ROI mean intensity values.
    corrected_column : str, default="mean_intensity_corrected"
        Name of the output corrected intensity column.
    clip_floor : float or None, default=None
        Optional lower bound applied after subtraction. Use ``None`` to preserve
        negative corrected values.

    Returns
    -------
    pandas.DataFrame
        Copy of ``intensity_table`` with added dark constants and corrected
        intensity values.
    """

    required_columns = {"channel", intensity_column}
    missing_columns = required_columns.difference(intensity_table.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    output = intensity_table.copy()
    channel_offsets = {"green": float(green_dark), "red": float(red_dark)}
    unknown_channels = set(output["channel"].dropna().unique()).difference(channel_offsets)
    if unknown_channels:
        unknown_str = ", ".join(sorted(str(channel_name) for channel_name in unknown_channels))
        raise ValueError(f"Unsupported channel labels: {unknown_str}")

    output["dark_value"] = output["channel"].map(channel_offsets).astype(float)
    output[corrected_column] = output[intensity_column].astype(float) - output["dark_value"]
    if clip_floor is not None:
        output[corrected_column] = output[corrected_column].clip(lower=float(clip_floor))
    return output


def summarize_daily_green_red_linear_fits(
    roi_metrics: pd.DataFrame,
    red_column: str = "red",
    green_column: str = "green",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit day-specific linear models of corrected green intensity versus red.

    Parameters
    ----------
    roi_metrics : pandas.DataFrame
        ROI/day table with at least ``day``, ``red_column``, and
        ``green_column``. Both intensity columns must be in the same corrected
        fluorescence units.
    red_column : str, default="red"
        Column used as the predictor ``x`` in the per-day linear model
        ``green = intercept + slope * red``.
    green_column : str, default="green"
        Column used as the response ``y`` in the per-day linear model
        ``green = intercept + slope * red``.
    alpha : float, default=0.05
        Two-sided type-I error rate for the reported confidence intervals.

    Returns
    -------
    pandas.DataFrame
        One row per day with columns:
        ``day`` (int), ``n_rois`` (int), ``slope`` (float), ``intercept``
        (float), ``r_value`` (float), ``r_squared`` (float), ``p_value``
        (float), ``slope_stderr`` (float), ``intercept_stderr`` (float),
        ``slope_ci_low`` (float), ``slope_ci_high`` (float),
        ``intercept_ci_low`` (float), and ``intercept_ci_high`` (float). All
        confidence-interval bounds use the same fluorescence units as the
        fitted parameter.
    """

    from scipy import stats

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    required_columns = {"day", red_column, green_column}
    missing_columns = required_columns.difference(roi_metrics.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    summary_rows: list[dict[str, float | int]] = []
    for day, day_table in roi_metrics.groupby("day", sort=True):
        valid_rows = day_table[[red_column, green_column]].replace([np.inf, -np.inf], np.nan)
        valid_rows = valid_rows.dropna().reset_index(drop=True)
        if len(valid_rows) < 2:
            raise ValueError(
                f"Day {day} has fewer than 2 valid ROIs after dropping invalid rows."
            )

        x_values = valid_rows[red_column].to_numpy(dtype=float)
        y_values = valid_rows[green_column].to_numpy(dtype=float)
        fit = stats.linregress(x_values, y_values)

        degrees_of_freedom = len(valid_rows) - 2
        if degrees_of_freedom > 0:
            t_critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=degrees_of_freedom))
            slope_ci_low = float(fit.slope - t_critical * fit.stderr)
            slope_ci_high = float(fit.slope + t_critical * fit.stderr)
            intercept_ci_low = float(
                fit.intercept - t_critical * fit.intercept_stderr
            )
            intercept_ci_high = float(
                fit.intercept + t_critical * fit.intercept_stderr
            )
        else:
            slope_ci_low = np.nan
            slope_ci_high = np.nan
            intercept_ci_low = np.nan
            intercept_ci_high = np.nan

        summary_rows.append(
            {
                "day": int(day),
                "n_rois": int(len(valid_rows)),
                "slope": float(fit.slope),
                "intercept": float(fit.intercept),
                "r_value": float(fit.rvalue),
                "r_squared": float(fit.rvalue**2),
                "p_value": float(fit.pvalue),
                "slope_stderr": float(fit.stderr),
                "intercept_stderr": float(fit.intercept_stderr),
                "slope_ci_low": slope_ci_low,
                "slope_ci_high": slope_ci_high,
                "intercept_ci_low": intercept_ci_low,
                "intercept_ci_high": intercept_ci_high,
            }
        )

    return pd.DataFrame(summary_rows).sort_values("day").reset_index(drop=True)


def compute_green_red_fit_residuals(
    roi_metrics: pd.DataFrame,
    fit_summary: pd.DataFrame | None = None,
    red_column: str = "red",
    green_column: str = "green",
) -> pd.DataFrame:
    """Compute day-specific signed deviations from the fitted green-vs-red line.

    Parameters
    ----------
    roi_metrics : pandas.DataFrame
        ROI/day table with at least ``roi_id``, ``day``, ``red_column``, and
        ``green_column``. Both intensity columns must be in the same corrected
        fluorescence units.
    fit_summary : pandas.DataFrame or None, default=None
        Optional table with one row per day and columns ``day``, ``slope``, and
        ``intercept``. When omitted, the day-wise fits are recomputed from
        ``roi_metrics`` using :func:`summarize_daily_green_red_linear_fits`.
    red_column : str, default="red"
        Predictor column used in the day-wise linear model.
    green_column : str, default="green"
        Response column used in the day-wise linear model.

    Returns
    -------
    pandas.DataFrame
        Copy of ``roi_metrics`` with added columns:
        ``slope`` and ``intercept`` from the day-specific fit,
        ``predicted_green_from_fit`` (same fluorescence units as ``green``),
        ``green_fit_residual`` (signed vertical deviation in corrected green
        units), ``green_fit_signed_distance`` (signed perpendicular distance in
        corrected fluorescence units), ``day0_green_fit_residual``,
        ``delta_green_fit_residual``, ``day0_green_fit_signed_distance``, and
        ``delta_green_fit_signed_distance``.
    """

    required_columns = {"roi_id", "day", red_column, green_column}
    missing_columns = required_columns.difference(roi_metrics.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    if fit_summary is None:
        fit_summary = summarize_daily_green_red_linear_fits(
            roi_metrics=roi_metrics,
            red_column=red_column,
            green_column=green_column,
        )

    fit_required_columns = {"day", "slope", "intercept"}
    missing_fit_columns = fit_required_columns.difference(fit_summary.columns)
    if missing_fit_columns:
        missing_str = ", ".join(sorted(missing_fit_columns))
        raise ValueError(f"Missing required fit-summary columns: {missing_str}")

    output = roi_metrics.copy().sort_values(["roi_id", "day"]).reset_index(drop=True)
    fit_table = fit_summary.loc[:, ["day", "slope", "intercept"]].copy()
    if fit_table["day"].duplicated().any():
        raise ValueError("fit_summary must contain at most one row per day.")

    output = output.merge(fit_table, on="day", how="left", validate="many_to_one")
    if output["slope"].isna().any() or output["intercept"].isna().any():
        raise ValueError("fit_summary did not provide slope and intercept for every day.")

    output["predicted_green_from_fit"] = (
        output["intercept"].astype(float)
        + output["slope"].astype(float) * output[red_column].astype(float)
    )
    output["green_fit_residual"] = (
        output[green_column].astype(float) - output["predicted_green_from_fit"]
    )
    output["green_fit_signed_distance"] = output["green_fit_residual"] / np.sqrt(
        1.0 + output["slope"].astype(float) ** 2
    )
    output["day0_green_fit_residual"] = output.groupby("roi_id")[
        "green_fit_residual"
    ].transform("first")
    output["delta_green_fit_residual"] = (
        output["green_fit_residual"] - output["day0_green_fit_residual"]
    )
    output["day0_green_fit_signed_distance"] = output.groupby("roi_id")[
        "green_fit_signed_distance"
    ].transform("first")
    output["delta_green_fit_signed_distance"] = (
        output["green_fit_signed_distance"] - output["day0_green_fit_signed_distance"]
    )
    return output


def summarize_residual_sign_changes(
    residual_table: pd.DataFrame,
    residual_column: str = "green_fit_residual",
) -> pd.DataFrame:
    """Summarize residual excursions and sign changes for each ROI.

    Parameters
    ----------
    residual_table : pandas.DataFrame
        ROI/day table that includes ``roi_id``, ``day``, and
        ``residual_column``. Residual values are signed deviations in corrected
        fluorescence units.
    residual_column : str, default="green_fit_residual"
        Column containing the signed residual trajectory to summarize.

    Returns
    -------
    pandas.DataFrame
        One row per ROI with columns ``roi_id``, the minimum, maximum, range,
        and final residual value in corrected fluorescence units, plus the
        integer column ``green_fit_residual_sign_change_count``.
    """

    required_columns = {"roi_id", "day", residual_column}
    missing_columns = required_columns.difference(residual_table.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    summary_rows: list[dict[str, float | int]] = []
    for roi_id, roi_table in residual_table.groupby("roi_id", sort=True):
        roi_table = roi_table.sort_values("day").reset_index(drop=True)
        residual_values = roi_table[residual_column].to_numpy(dtype=float)
        residual_sign = np.sign(residual_values)
        nonzero_sign = residual_sign.copy()
        for index in range(1, len(nonzero_sign)):
            if nonzero_sign[index] == 0:
                nonzero_sign[index] = nonzero_sign[index - 1]
        sign_change_count = int(np.sum(np.diff(nonzero_sign) != 0))
        summary_rows.append(
            {
                "roi_id": int(roi_id),
                "min_green_fit_residual": float(np.min(residual_values)),
                "max_green_fit_residual": float(np.max(residual_values)),
                "green_fit_residual_range": float(
                    np.max(residual_values) - np.min(residual_values)
                ),
                "last_green_fit_residual": float(residual_values[-1]),
                "green_fit_residual_sign_change_count": sign_change_count,
            }
        )

    return pd.DataFrame(summary_rows).sort_values("roi_id").reset_index(drop=True)


def select_ranked_roi_days(
    roi_day_table: pd.DataFrame,
    ranking_table: pd.DataFrame,
    top_n: int,
    ranking_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Attach a ROI ranking table and keep only the requested top-N ROIs.

    Parameters
    ----------
    roi_day_table : pandas.DataFrame
        ROI/day table with at least ``roi_id`` and ``day``. Each row represents
        one ROI on one day.
    ranking_table : pandas.DataFrame
        One-row-per-ROI ranking table that includes ``roi_id`` and the columns
        listed in ``ranking_columns``.
    top_n : int
        Number of ranked ROIs to keep. Must be positive.
    ranking_columns : sequence of str
        Ranking-table columns to merge into the ROI/day table. Typical examples
        include ``selection_rank`` and the ranking metric used to select the
        ROIs.

    Returns
    -------
    pandas.DataFrame
        ROI/day subset that includes only the first ``top_n`` ranked ROIs, with
        the requested ranking columns merged in and rows sorted by
        ``selection_rank`` then ``day`` when available.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    ranking_columns = list(ranking_columns)
    required_roi_columns = {"roi_id", "day"}
    missing_roi_columns = required_roi_columns.difference(roi_day_table.columns)
    if missing_roi_columns:
        missing_str = ", ".join(sorted(missing_roi_columns))
        raise ValueError(f"Missing required roi_day_table columns: {missing_str}")

    required_ranking_columns = {"roi_id"}.union(ranking_columns)
    missing_ranking_columns = required_ranking_columns.difference(ranking_table.columns)
    if missing_ranking_columns:
        missing_str = ", ".join(sorted(missing_ranking_columns))
        raise ValueError(f"Missing required ranking_table columns: {missing_str}")

    top_ranking = ranking_table.sort_values("selection_rank").head(int(top_n)).copy()
    selected = roi_day_table[roi_day_table["roi_id"].isin(top_ranking["roi_id"])].copy()
    selected = selected.merge(
        top_ranking.loc[:, ["roi_id", *ranking_columns]],
        on="roi_id",
        how="left",
        validate="many_to_one",
    )
    sort_columns = [column for column in ["selection_rank", "day"] if column in selected.columns]
    if sort_columns:
        selected = selected.sort_values(sort_columns).reset_index(drop=True)
    else:
        selected = selected.reset_index(drop=True)
    return selected


def summarize_roi_metrics(roi_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize longitudinal fluorescence behavior for each ROI.

    Parameters
    ----------
    roi_metrics : pandas.DataFrame
        ROI/day table generated by :func:`compute_log_ratio_metrics`.

    Returns
    -------
    pandas.DataFrame
        One row per ROI with stability and fluctuation metrics in arbitrary
        fluorescence units and log2-ratio units.
    """

    required_columns = {
        "roi_id",
        "day",
        "red",
        "green",
        "brightness",
        "log2_green_over_red",
        "delta_log2_green_over_red",
    }
    missing_columns = required_columns.difference(roi_metrics.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    summaries: list[dict[str, float | int]] = []
    for roi_id, roi_table in roi_metrics.groupby("roi_id", sort=True):
        roi_table = roi_table.sort_values("day").reset_index(drop=True)
        delta_values = roi_table["delta_log2_green_over_red"].to_numpy()
        day_values = roi_table["day"].to_numpy()
        min_delta_index = int(np.argmin(delta_values))

        summaries.append(
            {
                "roi_id": int(roi_id),
                "n_days": int(len(roi_table)),
                "day0_red": float(roi_table["red"].iloc[0]),
                "day0_green": float(roi_table["green"].iloc[0]),
                "day0_brightness": float(roi_table["brightness"].iloc[0]),
                "day0_log2_green_over_red": float(
                    roi_table["log2_green_over_red"].iloc[0]
                ),
                "red_cv": _coefficient_of_variation(roi_table["red"].to_numpy()),
                "green_cv": _coefficient_of_variation(roi_table["green"].to_numpy()),
                "brightness_cv": _coefficient_of_variation(
                    roi_table["brightness"].to_numpy()
                ),
                "min_delta_log2_green_over_red": float(delta_values[min_delta_index]),
                "day_of_min_delta": int(day_values[min_delta_index]),
                "max_delta_log2_green_over_red": float(np.max(delta_values)),
                "delta_log2_range": float(
                    np.max(delta_values) - np.min(delta_values)
                ),
                "day_last_delta_log2_green_over_red": float(delta_values[-1]),
            }
        )

    return pd.DataFrame(summaries).sort_values("roi_id").reset_index(drop=True)


def classify_roi_log_ratio_trajectories(
    roi_metrics: pd.DataFrame,
    delta_column: str = "delta_log2_green_over_red",
    stable_abs_threshold: float = 0.25,
    directional_change_threshold: float = 0.35,
    oscillation_abs_threshold: float = 0.25,
    return_abs_threshold: float = 0.15,
) -> pd.DataFrame:
    """Classify ROI log-ratio trajectories into stable, directional, or mixed groups.

    Parameters
    ----------
    roi_metrics : pandas.DataFrame
        ROI/day table with columns ``roi_id``, ``day``, and ``delta_column``.
        The delta values are log2-ratio changes relative to day 0 and are
        dimensionless.
    delta_column : str, default="delta_log2_green_over_red"
        Column containing the day-0-normalized log-ratio trajectory to classify.
    stable_abs_threshold : float, default=0.25
        Maximum absolute delta log2-ratio allowed for the ``stable`` category.
        This threshold is dimensionless.
    directional_change_threshold : float, default=0.35
        Minimum signed excursion required to call a ROI ``mostly_up`` or
        ``mostly_down``. This threshold is dimensionless.
    oscillation_abs_threshold : float, default=0.25
        Minimum positive and negative excursion required before a trajectory can
        be labeled ``oscillatory``. This threshold is dimensionless.
    return_abs_threshold : float, default=0.15
        Minimum final signed offset from day 0 required to support a directional
        ``mostly_up`` or ``mostly_down`` call. This threshold is dimensionless.

    Returns
    -------
    pandas.DataFrame
        One row per ROI with the assigned ``trajectory_category`` plus summary
        metrics including peak positive and negative excursion, final delta, and
        significant sign-change count.
    """

    required_columns = {"roi_id", "day", delta_column}
    missing_columns = required_columns.difference(roi_metrics.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required trajectory columns: {missing_str}")
    if stable_abs_threshold < 0:
        raise ValueError("stable_abs_threshold must be non-negative.")
    if directional_change_threshold < 0:
        raise ValueError("directional_change_threshold must be non-negative.")
    if oscillation_abs_threshold < 0:
        raise ValueError("oscillation_abs_threshold must be non-negative.")
    if return_abs_threshold < 0:
        raise ValueError("return_abs_threshold must be non-negative.")

    summaries: list[dict[str, float | int | str]] = []
    for roi_id, roi_table in roi_metrics.groupby("roi_id", sort=True):
        roi_table = roi_table.sort_values("day").reset_index(drop=True)
        delta_values = roi_table[delta_column].astype(float).to_numpy()
        max_abs_delta = float(np.max(np.abs(delta_values)))
        min_delta = float(np.min(delta_values))
        max_delta = float(np.max(delta_values))
        final_delta = float(delta_values[-1])
        sign_change_count = _count_thresholded_sign_changes(
            delta_values=delta_values,
            threshold=float(oscillation_abs_threshold),
        )

        if max_abs_delta < stable_abs_threshold:
            category = "stable"
        elif (
            min_delta <= -oscillation_abs_threshold
            and max_delta >= oscillation_abs_threshold
            and sign_change_count >= 1
        ):
            category = "oscillatory"
        elif (
            min_delta <= -directional_change_threshold
            and final_delta <= -return_abs_threshold
            and abs(min_delta) >= max_delta
        ):
            category = "mostly_down"
        elif (
            max_delta >= directional_change_threshold
            and final_delta >= return_abs_threshold
            and max_delta >= abs(min_delta)
        ):
            category = "mostly_up"
        else:
            category = "stable"

        summaries.append(
            {
                "roi_id": int(roi_id),
                "trajectory_category": category,
                "max_abs_delta_log2_green_over_red": max_abs_delta,
                "min_delta_log2_green_over_red": min_delta,
                "max_delta_log2_green_over_red": max_delta,
                "final_delta_log2_green_over_red": final_delta,
                "significant_sign_change_count": int(sign_change_count),
                "stable_abs_threshold": float(stable_abs_threshold),
                "directional_change_threshold": float(directional_change_threshold),
                "oscillation_abs_threshold": float(oscillation_abs_threshold),
                "return_abs_threshold": float(return_abs_threshold),
            }
        )

    return pd.DataFrame(summaries).sort_values("roi_id").reset_index(drop=True)


def attach_roi_size_metrics(
    roi_summary: pd.DataFrame,
    roi_size_table: pd.DataFrame,
) -> pd.DataFrame:
    """Attach per-ROI size metrics to a summary table.

    Parameters
    ----------
    roi_summary : pandas.DataFrame
        ROI summary table with a ``roi_id`` column.
    roi_size_table : pandas.DataFrame
        Table with ``roi_id`` plus one or more ROI size columns, such as
        ``proj_area_px`` or ``eq_diam_um``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``roi_summary`` with the non-``roi_id`` columns from
        ``roi_size_table`` merged in.
    """

    if "roi_id" not in roi_summary.columns or "roi_id" not in roi_size_table.columns:
        raise ValueError("Both tables must include a roi_id column.")

    size_columns = ["roi_id"] + [
        column_name for column_name in roi_size_table.columns if column_name != "roi_id"
    ]
    return roi_summary.merge(
        roi_size_table[size_columns],
        on="roi_id",
        how="left",
    )


def estimate_size_filter_bounds(
    roi_size_table: pd.DataFrame,
    area_column: str = "proj_area_px",
    lower_quantile: float = 0.05,
    upper_iqr_multiplier: float = 1.5,
) -> dict[str, float]:
    """Estimate conservative lower and upper ROI size bounds.

    Parameters
    ----------
    roi_size_table : pandas.DataFrame
        Table with one row per ROI and a size column such as ``proj_area_px``.
    area_column : str, default="proj_area_px"
        Column used to define the size inlier range.
    lower_quantile : float, default=0.05
        Quantile used for the lower size threshold.
    upper_iqr_multiplier : float, default=1.5
        Tukey-style multiplier used to define the upper threshold as
        ``q3 + multiplier * IQR``.

    Returns
    -------
    dict[str, float]
        Dictionary containing ``lower_bound``, ``upper_bound``, ``q1``, ``q3``,
        and ``iqr`` for the requested size column.
    """

    if area_column not in roi_size_table.columns:
        raise ValueError(f"Missing required size column: {area_column}")
    if not 0.0 <= lower_quantile <= 1.0:
        raise ValueError("lower_quantile must be between 0 and 1.")
    if upper_iqr_multiplier < 0:
        raise ValueError("upper_iqr_multiplier must be non-negative.")

    area_values = roi_size_table[area_column].astype(float)
    q1 = float(area_values.quantile(0.25))
    q3 = float(area_values.quantile(0.75))
    iqr = q3 - q1
    lower_bound = float(area_values.quantile(lower_quantile))
    upper_bound = float(q3 + upper_iqr_multiplier * iqr)

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
    }


def flag_shape_qc_rois(
    roi_size_table: pd.DataFrame,
    min_circularity: float = 0.65,
    min_solidity: float = 0.80,
    max_axis_ratio: float = 2.0,
) -> pd.DataFrame:
    """Flag ROIs that pass a simple projected-shape quality screen.

    Parameters
    ----------
    roi_size_table : pandas.DataFrame
        One row per ROI with columns ``roi_id``, ``circularity``, ``solidity``,
        and ``axis_ratio``. These metrics are dimensionless and are computed on
        the XY projected ROI mask.
    min_circularity : float, default=0.65
        Minimum allowed projected circularity, where
        ``4 * pi * area / perimeter^2`` is dimensionless and larger values are
        more compact.
    min_solidity : float, default=0.80
        Minimum allowed projected solidity, defined as ``area / convex_area`` in
        the XY projection. This metric is dimensionless.
    max_axis_ratio : float, default=2.0
        Maximum allowed projected elongation, defined as
        ``major_axis_length / minor_axis_length``. This metric is dimensionless.

    Returns
    -------
    pandas.DataFrame
        Copy of ``roi_size_table`` with added boolean column ``shape_qc_pass``
        plus threshold metadata columns ``shape_qc_min_circularity``,
        ``shape_qc_min_solidity``, and ``shape_qc_max_axis_ratio``.
    """

    required_columns = {"roi_id", "circularity", "solidity", "axis_ratio"}
    missing_columns = required_columns.difference(roi_size_table.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required shape columns: {missing_str}")
    if min_circularity < 0:
        raise ValueError("min_circularity must be non-negative.")
    if min_solidity < 0:
        raise ValueError("min_solidity must be non-negative.")
    if max_axis_ratio <= 0:
        raise ValueError("max_axis_ratio must be positive.")

    flagged = roi_size_table.copy()
    flagged["shape_qc_pass"] = (
        (flagged["circularity"] >= float(min_circularity))
        & (flagged["solidity"] >= float(min_solidity))
        & (flagged["axis_ratio"] <= float(max_axis_ratio))
    )
    flagged["shape_qc_min_circularity"] = float(min_circularity)
    flagged["shape_qc_min_solidity"] = float(min_solidity)
    flagged["shape_qc_max_axis_ratio"] = float(max_axis_ratio)
    return flagged


def _build_changing_roi_candidates(
    roi_summary: pd.DataFrame,
    red_cv_max: float,
    min_day0_brightness_quantile: float,
    direction: str,
    min_proj_area_px: float | None,
    max_proj_area_px: float | None,
) -> tuple[pd.DataFrame, float, str]:
    """Build the candidate ROI pool for directional selection."""

    if not 0.0 <= min_day0_brightness_quantile <= 1.0:
        raise ValueError("min_day0_brightness_quantile must be between 0 and 1.")

    required_columns = {
        "roi_id",
        "day0_brightness",
        "day0_green",
        "red_cv",
        "min_delta_log2_green_over_red",
        "max_delta_log2_green_over_red",
        "delta_log2_range",
    }
    if min_proj_area_px is not None or max_proj_area_px is not None:
        required_columns.add("proj_area_px")
    missing_columns = required_columns.difference(roi_summary.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    brightness_threshold = float(
        roi_summary["day0_brightness"].quantile(min_day0_brightness_quantile)
    )
    candidates = roi_summary[
        (roi_summary["red_cv"] <= red_cv_max)
        & (roi_summary["day0_brightness"] >= brightness_threshold)
        & (roi_summary["day0_green"] > 0)
    ].copy()
    if min_proj_area_px is not None:
        candidates = candidates[candidates["proj_area_px"] >= min_proj_area_px]
    if max_proj_area_px is not None:
        candidates = candidates[candidates["proj_area_px"] <= max_proj_area_px]

    if direction == "decreasing":
        selection_metric_column = "min_delta_log2_green_over_red"
    else:
        selection_metric_column = "max_delta_log2_green_over_red"
    return candidates, brightness_threshold, selection_metric_column


def select_top_changing_rois(
    roi_summary: pd.DataFrame,
    max_rois: int = 15,
    red_cv_max: float = 0.1,
    min_day0_brightness_quantile: float = 0.25,
    direction: str = "decreasing",
    min_proj_area_px: float | None = None,
    max_proj_area_px: float | None = None,
    random_sample: bool = False,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Select ROIs with the strongest green loss or gain among red-stable cells.

    Parameters
    ----------
    roi_summary : pandas.DataFrame
        ROI summary table generated by :func:`summarize_roi_metrics`.
    max_rois : int, default=15
        Maximum number of ROIs to return.
    red_cv_max : float, default=0.1
        Maximum allowed coefficient of variation for the red channel.
    min_day0_brightness_quantile : float, default=0.25
        Quantile threshold applied to day-0 brightness before ranking. This is
        dimensionless and must lie in the interval [0, 1].
    direction : {"decreasing", "increasing"}, default="decreasing"
        Whether to rank ROIs by the strongest negative or positive
        day-0-normalized log-ratio excursion.
    min_proj_area_px : float or None, default=None
        Optional lower bound on projected ROI area in pixels.
    max_proj_area_px : float or None, default=None
        Optional upper bound on projected ROI area in pixels.
    random_sample : bool, default=False
        When ``True``, sample up to ``max_rois`` candidates uniformly at random
        instead of taking the strongest ranked ROIs.
    random_seed : int, default=0
        Seed for the random sampler used when ``random_sample`` is enabled.

    Returns
    -------
    pandas.DataFrame
        Ranked subset of ``roi_summary`` with added ``selection_rank`` and
        threshold metadata columns.
    """

    if max_rois <= 0:
        raise ValueError("max_rois must be positive.")
    if direction not in {"decreasing", "increasing"}:
        raise ValueError("direction must be either 'decreasing' or 'increasing'.")

    candidates, brightness_threshold, selection_metric_column = _build_changing_roi_candidates(
        roi_summary=roi_summary,
        red_cv_max=red_cv_max,
        min_day0_brightness_quantile=min_day0_brightness_quantile,
        direction=direction,
        min_proj_area_px=min_proj_area_px,
        max_proj_area_px=max_proj_area_px,
    )

    if candidates.empty:
        ranked = candidates.copy()
    elif random_sample:
        rng = np.random.default_rng(int(random_seed))
        sample_size = min(int(max_rois), len(candidates))
        sampled_positions = rng.choice(len(candidates), size=sample_size, replace=False)
        ranked = candidates.iloc[sampled_positions].copy().reset_index(drop=True)
        ranked = ranked.iloc[rng.permutation(len(ranked))].reset_index(drop=True)
    else:
        if direction == "decreasing":
            ranked = candidates.sort_values(
                [
                    "min_delta_log2_green_over_red",
                    "delta_log2_range",
                    "day0_brightness",
                ],
                ascending=[True, False, False],
            ).head(max_rois)
        else:
            ranked = candidates.sort_values(
                [
                    "max_delta_log2_green_over_red",
                    "delta_log2_range",
                    "day0_brightness",
                ],
                ascending=[False, False, False],
            ).head(max_rois)

    ranked = ranked.reset_index(drop=True)
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["brightness_threshold"] = brightness_threshold
    ranked["selection_direction"] = direction
    ranked["selection_metric_column"] = (
        "random_sample" if random_sample else selection_metric_column
    )
    ranked["selection_random_seed"] = np.nan if not random_sample else int(random_seed)
    ranked["min_proj_area_px_threshold"] = (
        np.nan if min_proj_area_px is None else float(min_proj_area_px)
    )
    ranked["max_proj_area_px_threshold"] = (
        np.nan if max_proj_area_px is None else float(max_proj_area_px)
    )
    return ranked


def compute_roi_size_table(
    mask_stack: np.ndarray,
    xy_um_per_px: float | None = None,
    z_um_per_plane: float | None = None,
) -> pd.DataFrame:
    """Measure ROI footprint and span metrics from a 3D label mask.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer ROI label image with shape ``(z, y, x)``.
    xy_um_per_px : float or None, default=None
        XY pixel size in micrometers per pixel. When provided, projected area and
        equivalent diameter are also reported in micrometers.
    z_um_per_plane : float or None, default=None
        Z step size in micrometers per plane. When provided, the ROI z-span is
        also reported in micrometers.

    Returns
    -------
    pandas.DataFrame
        One row per ROI with projected area, projected perimeter, circularity,
        solidity, eccentricity, axis ratio, equivalent diameter, voxel count,
        and x/y/z span metrics.
    """

    if mask_stack.ndim != 3:
        raise ValueError("mask_stack must have shape (z, y, x).")

    from skimage import measure

    rows: list[dict[str, float | int]] = []
    for region in measure.regionprops(mask_stack):
        z_start, y_start, x_start, z_stop, y_stop, x_stop = region.bbox
        roi_crop = mask_stack[z_start:z_stop, y_start:y_stop, x_start:x_stop] == region.label
        proj_mask = np.any(roi_crop, axis=0).astype(np.uint8)
        proj_area_px = int(proj_mask.sum())
        eq_diam_px = float(np.sqrt(4.0 * proj_area_px / np.pi))
        proj_region = measure.regionprops(proj_mask)[0]
        proj_perimeter_px = float(measure.perimeter_crofton(proj_mask))
        circularity = (
            float(4.0 * np.pi * proj_area_px / (proj_perimeter_px**2))
            if proj_perimeter_px > 0
            else float("nan")
        )
        axis_major_length = float(
            proj_region.axis_major_length
            if hasattr(proj_region, "axis_major_length")
            else proj_region.major_axis_length
        )
        axis_minor_length = float(
            proj_region.axis_minor_length
            if hasattr(proj_region, "axis_minor_length")
            else proj_region.minor_axis_length
        )
        axis_ratio = (
            float(axis_major_length / axis_minor_length)
            if axis_minor_length > 0
            else float("nan")
        )

        row = {
            "roi_id": int(region.label),
            "voxel_count": int(region.area),
            "proj_area_px": proj_area_px,
            "proj_perimeter_px": proj_perimeter_px,
            "circularity": circularity,
            "solidity": float(proj_region.solidity),
            "eccentricity": float(proj_region.eccentricity),
            "axis_ratio": axis_ratio,
            "eq_diam_px": eq_diam_px,
            "z_span_px": int(z_stop - z_start),
            "y_span_px": int(y_stop - y_start),
            "x_span_px": int(x_stop - x_start),
        }
        if xy_um_per_px is not None:
            row["proj_area_um2"] = float(proj_area_px * (xy_um_per_px**2))
            row["eq_diam_um"] = float(eq_diam_px * xy_um_per_px)
        if z_um_per_plane is not None:
            row["z_span_um"] = float((z_stop - z_start) * z_um_per_plane)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("roi_id").reset_index(drop=True)


def prepare_stack_for_display(
    image_stack: np.ndarray,
    dark_value: float = 0.0,
    clip_floor: float = 0.0,
) -> np.ndarray:
    """Apply optional display-time dark subtraction to a 3D image stack.

    Parameters
    ----------
    image_stack : numpy.ndarray
        Fluorescence image stack with shape ``(z, y, x)``.
    dark_value : float, default=0.0
        Constant offset to subtract from every pixel in the stack for display.
    clip_floor : float, default=0.0
        Lower bound applied after subtraction for display purposes only.

    Returns
    -------
    numpy.ndarray
        Float array with the same shape as ``image_stack`` after dark
        subtraction and lower clipping.
    """

    display_stack = np.asarray(image_stack, dtype=float) - float(dark_value)
    if clip_floor is not None:
        display_stack = np.clip(display_stack, a_min=float(clip_floor), a_max=None)
    return display_stack


def compute_roi_crop_bounds(
    mask_stack: np.ndarray,
    roi_id: int,
    pad_xy: int = 20,
    min_crop_size: int = 48,
    z_pad: int = 0,
) -> dict[str, int]:
    """Compute a centered 3D crop around one ROI.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer ROI label image with shape ``(z, y, x)``.
    roi_id : int
        Positive ROI label to extract.
    pad_xy : int, default=20
        Extra padding, in pixels, added in both x and y around the ROI.
    min_crop_size : int, default=48
        Minimum crop width and height in pixels.
    z_pad : int, default=0
        Extra padding, in slices, added before max-projection.

    Returns
    -------
    dict[str, int]
        Dictionary with inclusive-start, exclusive-stop crop bounds named
        ``z_start``, ``z_stop``, ``y_start``, ``y_stop``, ``x_start``, and
        ``x_stop``.
    """

    if mask_stack.ndim != 3:
        raise ValueError("mask_stack must have shape (z, y, x).")

    roi_mask = mask_stack == roi_id
    if not np.any(roi_mask):
        raise ValueError(f"ROI {roi_id} was not found in the mask stack.")

    z_coords, y_coords, x_coords = np.nonzero(roi_mask)
    z_start = max(0, int(z_coords.min()) - z_pad)
    z_stop = min(mask_stack.shape[0], int(z_coords.max()) + 1 + z_pad)

    y_min = int(y_coords.min())
    y_max = int(y_coords.max())
    x_min = int(x_coords.min())
    x_max = int(x_coords.max())

    y_start = max(0, y_min - int(pad_xy))
    y_stop = min(mask_stack.shape[1], y_max + 1 + int(pad_xy))
    x_start = max(0, x_min - int(pad_xy))
    x_stop = min(mask_stack.shape[2], x_max + 1 + int(pad_xy))

    y_start, y_stop = _expand_bounds_to_min_size(
        start=y_start,
        stop=y_stop,
        limit=mask_stack.shape[1],
        min_size=min_crop_size,
    )
    x_start, x_stop = _expand_bounds_to_min_size(
        start=x_start,
        stop=x_stop,
        limit=mask_stack.shape[2],
        min_size=min_crop_size,
    )

    return {
        "z_start": z_start,
        "z_stop": z_stop,
        "y_start": y_start,
        "y_stop": y_stop,
        "x_start": x_start,
        "x_stop": x_stop,
    }


def project_roi_stack_view(
    image_stack: np.ndarray,
    mask_stack: np.ndarray,
    roi_id: int,
    pad_xy: int = 20,
    min_crop_size: int = 48,
    z_pad: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Create a max-projection crop and ROI mask projection for display.

    Parameters
    ----------
    image_stack : numpy.ndarray
        Fluorescence image stack with shape ``(z, y, x)`` in arbitrary
        fluorescence units.
    mask_stack : numpy.ndarray
        Integer ROI label image with the same shape convention ``(z, y, x)``.
    roi_id : int
        Positive ROI label to extract.
    pad_xy : int, default=20
        Extra padding, in pixels, added in both x and y around the ROI.
    min_crop_size : int, default=48
        Minimum crop width and height in pixels.
    z_pad : int, default=0
        Extra padding, in slices, added before max-projection.

    Returns
    -------
    tuple
        ``(image_projection, roi_projection, bounds)`` where the two projection
        arrays have shape ``(y, x)`` and ``bounds`` is the dictionary returned by
        :func:`compute_roi_crop_bounds`.
    """

    if image_stack.shape != mask_stack.shape:
        raise ValueError("image_stack and mask_stack must have the same shape.")

    bounds = compute_roi_crop_bounds(
        mask_stack=mask_stack,
        roi_id=roi_id,
        pad_xy=pad_xy,
        min_crop_size=min_crop_size,
        z_pad=z_pad,
    )

    image_projection, roi_projection = project_roi_view_from_bounds(
        image_stack=image_stack,
        mask_stack=mask_stack,
        roi_id=roi_id,
        bounds=bounds,
    )
    return image_projection, roi_projection, bounds


def _parse_dated_channel_tiff_name(
    image_name: str,
) -> tuple[str, str, bool] | None:
    """Parse a dated TIFF filename into ``(date_key, channel, is_syn)``.

    Parameters
    ----------
    image_name : str
        TIFF filename such as ``20260511_R.tif`` or
        ``20260511_G_crop_256_SyN.tif``.

    Returns
    -------
    tuple[str, str, bool] or None
        Parsed ``(YYYYMMDD, channel, is_syn)`` triple, or ``None`` when the
        filename does not describe a day/channel image stack.
    """

    path = Path(image_name)
    if path.suffix.lower() != ".tif":
        return None

    parts = path.stem.split("_")
    if len(parts) < 2 or len(parts[0]) != 8 or not parts[0].isdigit():
        return None

    channel_token = parts[1]
    if channel_token not in {"R", "G"}:
        return None

    trailing_tokens = parts[2:]
    trailing_lower = [token.lower() for token in trailing_tokens]
    if "syn" in trailing_lower and trailing_lower[-1] != "syn":
        return None
    if any("mask" in token for token in trailing_lower) or "roi" in trailing_lower:
        return None

    channel = "red" if channel_token == "R" else "green"
    is_syn = bool(trailing_lower) and trailing_lower[-1] == "syn"
    return parts[0], channel, is_syn


def build_registered_image_lookup(
    image_dir: str | Path,
    channels: tuple[str, ...] = ("red", "green"),
    start_date: str = "20260511",
    day0_mode: str = "raw",
) -> dict[tuple[int, str], Path]:
    """Map each day and channel to the corresponding registered TIFF file.

    Parameters
    ----------
    image_dir : str or pathlib.Path
        Directory that contains day-0 raw images plus registered ``*_SyN.tif``
        images, optionally with extra tokens between the channel and ``SyN``
        such as ``20260511_R_crop_256_SyN.tif``.
    channels : tuple[str, ...], default=("red", "green")
        Channel names to include. Supported names are ``"red"`` and ``"green"``.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format that defines day 0 for this
        dataset.
    day0_mode : {"raw", "syn"}, default="raw"
        Which day-0 image variant to select. The default preserves the
        original pipeline behavior of using the raw day-0 image, while
        ``"syn"`` selects the registered day-0 image when available.

    Returns
    -------
    dict[tuple[int, str], pathlib.Path]
        Mapping from ``(day, channel)`` to the selected TIFF path.
    """

    image_dir = Path(image_dir)
    if day0_mode not in {"raw", "syn"}:
        raise ValueError("day0_mode must be 'raw' or 'syn'.")

    lookup: dict[tuple[int, str], Path] = {}
    for channel in channels:
        if channel not in {"red", "green"}:
            raise ValueError(f"Unsupported channel: {channel}")

    for path in sorted(image_dir.glob("*.tif")):
        parsed = _parse_dated_channel_tiff_name(path.name)
        if parsed is None:
            continue

        _date_key, channel, is_syn = parsed
        if channel not in channels:
            continue

        day = extract_day_from_image_name(path.name, start_date=start_date)
        if day < 0:
            continue
        if day == 0:
            if day0_mode == "raw" and not is_syn:
                lookup[(day, channel)] = path
            elif day0_mode == "syn" and is_syn:
                lookup[(day, channel)] = path
        elif is_syn:
            lookup[(day, channel)] = path

    return lookup


def build_raw_image_lookup(
    image_dir: str | Path,
    channels: tuple[str, ...] = ("red", "green"),
    start_date: str = "20260511",
) -> dict[tuple[int, str], Path]:
    """Map each day and channel to the corresponding raw TIFF file.

    Parameters
    ----------
    image_dir : str or pathlib.Path
        Directory that contains raw day-specific TIFF files such as
        ``20260512_R.tif`` and ``20260512_G.tif``.
    channels : tuple[str, ...], default=("red", "green")
        Channel names to include. Supported names are ``"red"`` and ``"green"``.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format that defines day 0 for this
        dataset.

    Returns
    -------
    dict[tuple[int, str], pathlib.Path]
        Mapping from ``(day, channel)`` to the selected raw TIFF path.
    """

    image_dir = Path(image_dir)
    for channel in channels:
        if channel not in {"red", "green"}:
            raise ValueError(f"Unsupported channel: {channel}")

    lookup: dict[tuple[int, str], Path] = {}
    for path in sorted(image_dir.glob("*.tif")):
        parsed = _parse_dated_channel_tiff_name(path.name)
        if parsed is None:
            continue

        _date_key, channel, is_syn = parsed
        if channel not in channels or is_syn:
            continue

        day = extract_day_from_image_name(path.name, start_date=start_date)
        if day < 0:
            continue
        lookup[(day, channel)] = path

    return lookup


def build_inverse_warped_mask_lookup(
    image_dir: str | Path,
    day0_mask_name: str = "mean_image_merge_cp_masks_SAM.tif",
    inverse_mask_suffix: str = "_ROI_mask_SyN_inversed.tif",
    preferred_channel: str | None = None,
    start_date: str = "20260511",
) -> dict[int, Path]:
    """Map each day to the ROI mask defined in that day's raw image space.

    Parameters
    ----------
    image_dir : str or pathlib.Path
        Directory that contains the day-0 ROI mask plus one inverse-warped ROI
        mask per moving day.
    day0_mask_name : str, default="mean_image_merge_cp_masks_SAM.tif"
        Filename of the day-0 ROI label image. This mask is assumed to already
        be in the raw day-0 image space.
    inverse_mask_suffix : str, default="_ROI_mask_SyN_inversed.tif"
        Filename suffix for the inverse-warped ROI masks that live in the raw
        moving-image space for later days.
    preferred_channel : {"red", "green"} or None, default=None
        Optional channel preference used when more than one inverse-warped mask
        exists for the same day. When ``None``, the lookup requires at most one
        matching inverse mask per day.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format that defines day 0 for this
        dataset.

    Returns
    -------
    dict[int, pathlib.Path]
        Mapping from day index to the ROI mask TIFF for that day, where the
        value for day 0 is ``day0_mask_name`` and later days point to the
        selected inverse-warped masks.
    """

    image_dir = Path(image_dir)
    lookup: dict[int, Path] = {0: image_dir / day0_mask_name}
    if not lookup[0].exists():
        raise FileNotFoundError(f"Day-0 ROI mask was not found: {lookup[0]}")

    if preferred_channel is not None and preferred_channel not in {"red", "green"}:
        raise ValueError("preferred_channel must be 'red', 'green', or None.")

    candidate_paths = sorted(image_dir.glob(f"*{inverse_mask_suffix}"))
    candidates_by_day: dict[int, list[Path]] = {}
    for path in candidate_paths:
        day = extract_day_from_image_name(path.name, start_date=start_date)
        candidates_by_day.setdefault(day, []).append(path)

    channel_tag = None
    if preferred_channel is not None:
        channel_tag = "_R_" if preferred_channel == "red" else "_G_"

    for day, candidates in sorted(candidates_by_day.items()):
        if len(candidates) == 1:
            lookup[day] = candidates[0]
            continue

        if channel_tag is not None:
            preferred_candidates = [path for path in candidates if channel_tag in path.name]
            if len(preferred_candidates) == 1:
                lookup[day] = preferred_candidates[0]
                continue

        candidate_str = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Ambiguous inverse ROI masks for day {day}: {candidate_str}. "
            "Provide a preferred channel or disambiguate the filenames."
        )

    return lookup


def select_roi_neighbor_z_slices(
    mask_stack: np.ndarray,
    roi_id: int,
) -> dict[str, int]:
    """Select centroid-centered z-1, z, and z+1 slices for one ROI.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer ROI label image with shape ``(z, y, x)``.
    roi_id : int
        Positive ROI label to locate in ``mask_stack``.

    Returns
    -------
    dict[str, int]
        Dictionary with slice indices named ``z_minus1``, ``z_center``, and
        ``z_plus1``. The returned indices are clipped to the valid stack bounds
        and therefore may repeat at the top or bottom of the stack.
    """

    if mask_stack.ndim != 3:
        raise ValueError("mask_stack must have shape (z, y, x).")

    roi_mask = mask_stack == roi_id
    if not np.any(roi_mask):
        raise ValueError(f"ROI {roi_id} was not found in the mask stack.")

    z_coords = np.nonzero(roi_mask)[0]
    z_center = int(np.round(np.mean(z_coords)))
    z_center = int(np.clip(z_center, 0, mask_stack.shape[0] - 1))

    return {
        "z_minus1": max(0, z_center - 1),
        "z_center": z_center,
        "z_plus1": min(mask_stack.shape[0] - 1, z_center + 1),
    }


def _coefficient_of_variation(values: np.ndarray) -> float:
    """Return the population coefficient of variation for a 1D numeric array."""

    values = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values))
    if np.isclose(mean_value, 0.0):
        return float("nan")
    return float(np.std(values, ddof=0) / mean_value)


def _count_thresholded_sign_changes(
    delta_values: np.ndarray,
    threshold: float,
) -> int:
    """Count sign changes after removing small-amplitude delta values.

    Parameters
    ----------
    delta_values : numpy.ndarray
        One-dimensional array of day-0-normalized log-ratio deltas. These values
        are dimensionless.
    threshold : float
        Minimum absolute delta required for a point to contribute a sign. This
        threshold is dimensionless.

    Returns
    -------
    int
        Number of sign changes among the retained non-zero-sign trajectory
        points.
    """

    delta_values = np.asarray(delta_values, dtype=float)
    significant_values = delta_values[np.abs(delta_values) >= float(threshold)]
    if len(significant_values) == 0:
        return 0

    signs = np.sign(significant_values).astype(int)
    compressed_signs = [int(signs[0])]
    for sign in signs[1:]:
        if sign != compressed_signs[-1]:
            compressed_signs.append(int(sign))
    return max(0, len(compressed_signs) - 1)


def project_roi_view_from_bounds(
    image_stack: np.ndarray,
    mask_stack: np.ndarray,
    roi_id: int,
    bounds: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project a ROI view using precomputed crop bounds.

    Parameters
    ----------
    image_stack : numpy.ndarray
        Fluorescence image stack with shape ``(z, y, x)`` in arbitrary
        fluorescence units.
    mask_stack : numpy.ndarray
        Integer ROI label image with the same shape convention ``(z, y, x)``.
    roi_id : int
        Positive ROI label to extract.
    bounds : dict[str, int]
        Crop bounds produced by :func:`compute_roi_crop_bounds`.

    Returns
    -------
    tuple
        ``(image_projection, roi_projection)`` where both arrays have shape
        ``(y, x)`` and the ROI projection is boolean.
    """

    if image_stack.shape != mask_stack.shape:
        raise ValueError("image_stack and mask_stack must have the same shape.")

    z_slice = slice(bounds["z_start"], bounds["z_stop"])
    y_slice = slice(bounds["y_start"], bounds["y_stop"])
    x_slice = slice(bounds["x_start"], bounds["x_stop"])

    image_crop = image_stack[z_slice, y_slice, x_slice]
    roi_crop = mask_stack[z_slice, y_slice, x_slice] == roi_id

    image_projection = np.max(image_crop, axis=0)
    roi_projection = np.max(roi_crop, axis=0).astype(bool)
    return image_projection, roi_projection


def _expand_bounds_to_min_size(
    start: int,
    stop: int,
    limit: int,
    min_size: int,
) -> tuple[int, int]:
    """Expand an integer interval to a minimum size without crossing limits."""

    current_size = stop - start
    if current_size >= min_size:
        return start, stop

    missing = min_size - current_size
    lower_extra = missing // 2
    upper_extra = missing - lower_extra

    start = max(0, start - lower_extra)
    stop = min(limit, stop + upper_extra)

    if stop - start < min_size:
        if start == 0:
            stop = min(limit, min_size)
        elif stop == limit:
            start = max(0, limit - min_size)

    return start, stop
