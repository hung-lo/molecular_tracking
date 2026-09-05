"""Numerical QC for isolated same-session cross-laser ROI mapping."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_long_axis(image_shape_yx: tuple[int, int]) -> str:
    """Return the longer native FoV axis with deterministic tie handling."""

    height, width = (int(image_shape_yx[0]), int(image_shape_yx[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_shape_yx must contain positive dimensions")
    return "x" if width >= height else "y"


def _with_long_axis_position(
    table: pd.DataFrame,
    *,
    image_shape_yx: tuple[int, int],
) -> pd.DataFrame:
    """Add a normalized fixed-space long-axis position."""

    output = table.copy()
    axis = detect_long_axis(image_shape_yx)
    coordinate = "centroid_1050_x" if axis == "x" else "centroid_1050_y"
    denominator = max(int(image_shape_yx[1] if axis == "x" else image_shape_yx[0]) - 1, 1)
    output["long_axis"] = axis
    output["long_axis_position_normalized"] = pd.to_numeric(
        output[coordinate], errors="coerce"
    ).clip(0, denominator) / denominator
    return output


def high_confidence_long_axis_statistics(
    accepted_pairs: pd.DataFrame,
    *,
    image_shape_yx: tuple[int, int],
    bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return high-only points and medians from exactly that same population."""

    if bins < 1:
        raise ValueError("bins must be positive")
    high = accepted_pairs.copy()
    if "assignment_policy" in high:
        high = high.loc[high["assignment_policy"].astype(str).eq("high")].copy()
    if high.empty:
        return high, pd.DataFrame(
            columns=[
                "bin",
                "bin_center",
                "n_high",
                "median_raw_delta_z_planes",
                "median_abs_raw_delta_z_planes",
                "median_aligned_residual_distance_um",
            ]
        )
    high = _with_long_axis_position(high, image_shape_yx=image_shape_yx)
    high["bin"] = np.minimum(
        (high["long_axis_position_normalized"].clip(0, 0.999999) * bins).astype(int),
        bins - 1,
    )
    medians = (
        high.groupby("bin", sort=True)
        .agg(
            bin_center=("long_axis_position_normalized", "median"),
            n_high=("label_1050", "count"),
            median_raw_delta_z_planes=("raw_delta_z_planes", "median"),
            median_abs_raw_delta_z_planes=("raw_delta_z_planes", lambda values: np.median(np.abs(values))),
            median_aligned_residual_distance_um=("aligned_residual_distance_um", "median"),
        )
        .reset_index()
    )
    return high, medians


def fixed_coverage_by_long_axis(
    fixed_coverage: pd.DataFrame,
    *,
    image_shape_yx: tuple[int, int],
    bins: int = 5,
) -> pd.DataFrame:
    """Calculate fixed high coverage from all observable fixed labels."""

    if bins < 1:
        raise ValueError("bins must be positive")
    required = {"label_1050", "common_volume_status", "centroid_1050_y", "centroid_1050_x"}
    missing = required.difference(fixed_coverage.columns)
    if missing:
        raise ValueError(f"fixed_coverage is missing required columns: {sorted(missing)}")
    coverage = fixed_coverage.loc[
        fixed_coverage["common_volume_status"].astype(str).ne("outside_common_volume")
    ].copy()
    if coverage.empty:
        return pd.DataFrame(
            columns=["bin", "bin_center", "n_observable_1050", "n_high", "high_fraction"]
        )
    high_column = next(
        (column for column in ("green_high_label_920", "red_high_label_920", "primary_green_label_920") if column in coverage),
        None,
    )
    if high_column not in coverage:
        raise ValueError("fixed_coverage does not include a primary green high-match column")
    coverage = _with_long_axis_position(coverage, image_shape_yx=image_shape_yx)
    coverage["bin"] = np.minimum(
        (coverage["long_axis_position_normalized"].clip(0, 0.999999) * bins).astype(int),
        bins - 1,
    )
    output = (
        coverage.groupby("bin", sort=True)
        .agg(
            bin_center=("long_axis_position_normalized", "median"),
            n_observable_1050=("label_1050", "count"),
            n_high=(high_column, lambda values: int(values.notna().sum())),
        )
        .reset_index()
    )
    output["high_fraction"] = output["n_high"] / output["n_observable_1050"]
    return output


def select_cross_laser_examples(
    table: pd.DataFrame,
    *,
    limit: int = 12,
    status_column: str = "resolved_status",
    image_shape_yx: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Select deterministic spatially distributed rows for cross-laser review."""

    if limit < 1 or table.empty:
        return table.iloc[0:0].copy()
    output = table.copy()
    if image_shape_yx is not None:
        output = _with_long_axis_position(output, image_shape_yx=image_shape_yx)
    elif "long_axis_position_normalized" not in output:
        if {"centroid_1050_x", "centroid_1050_y"}.issubset(output.columns):
            maximum_x = max(float(pd.to_numeric(output["centroid_1050_x"]).max()), 1.0)
            maximum_y = max(float(pd.to_numeric(output["centroid_1050_y"]).max()), 1.0)
            axis = "x" if maximum_x >= maximum_y else "y"
            coordinate = "centroid_1050_x" if axis == "x" else "centroid_1050_y"
            output["long_axis_position_normalized"] = pd.to_numeric(
                output[coordinate], errors="coerce"
            ) / max(float(pd.to_numeric(output[coordinate], errors="coerce").max()), 1.0)
        else:
            output["long_axis_position_normalized"] = 0.0
    output["_selection_row"] = np.arange(len(output))
    output["_bin"] = np.minimum(
        (output["long_axis_position_normalized"].fillna(0).clip(0, 0.999999) * 5).astype(int),
        4,
    )
    score_column = "green_best_candidate_score" if "green_best_candidate_score" in output else None
    sort_columns = ["_bin"]
    ascending = [True]
    if score_column:
        sort_columns.append(score_column)
        ascending.append(False)
    sort_columns.extend(["label_1050", "_selection_row"])
    ascending.extend([True, True])
    output = output.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    selected: list[pd.Series] = []
    for _, group in output.groupby("_bin", sort=True):
        selected.append(group.iloc[0])
        if len(selected) >= limit:
            break
    selected_ids = {int(row["_selection_row"]) for row in selected}
    for _, row in output.iterrows():
        if int(row["_selection_row"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(int(row["_selection_row"]))
        if len(selected) >= limit:
            break
    result = pd.DataFrame(selected).drop(columns=["_selection_row", "_bin"], errors="ignore")
    if status_column in result:
        return result.sort_values([status_column, "label_1050"], kind="mergesort").reset_index(drop=True)
    return result.sort_values("label_1050", kind="mergesort").reset_index(drop=True)


def _save_figure(path: Path, figure: plt.Figure) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_cross_laser_qc(
    *,
    output_dir: str | Path,
    fixed_coverage: pd.DataFrame,
    accepted_pairs: pd.DataFrame,
    image_shape_yx: tuple[int, int],
    identity_resolution: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write numerical cross-laser QC without affecting mapping outputs."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    high, medians = high_confidence_long_axis_statistics(
        accepted_pairs, image_shape_yx=image_shape_yx
    )
    coverage = fixed_coverage_by_long_axis(
        fixed_coverage, image_shape_yx=image_shape_yx
    )
    medians_path = root / "high_residual_long_axis_medians.csv"
    coverage_path = root / "fixed_coverage_by_long_axis.csv"
    medians.to_csv(medians_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    outputs["high_residual_long_axis_medians"] = medians_path
    outputs["fixed_coverage_by_long_axis"] = coverage_path

    if not high.empty:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for source, group in high.groupby("moving_source", sort=True):
            axes[0].hist(group["raw_delta_z_planes"], bins=20, alpha=0.55, label=source)
            axes[1].hist(group["aligned_residual_z_um"], bins=20, alpha=0.55, label=source)
        axes[0].set_title("Raw delta z")
        axes[0].set_xlabel("1050 minus 920 (planes)")
        axes[1].set_title("Post-transform z residual")
        axes[1].set_xlabel("1050 minus aligned 920 (um)")
        axes[1].legend(frameon=False)
        outputs["axial_displacement"] = _save_figure(root / "axial_displacement.png", figure)

        figure, axis = plt.subplots(figsize=(6, 4))
        for source, group in high.groupby("moving_source", sort=True):
            axis.hist(
                group["aligned_residual_distance_um"],
                bins=20,
                alpha=0.55,
                label=source,
            )
        axis.set_title("Aligned residual distance")
        axis.set_xlabel("um")
        axis.legend(frameon=False)
        outputs["residual_distance"] = _save_figure(root / "residual_distance.png", figure)

        figure, axis = plt.subplots(figsize=(6, 5))
        axis.quiver(
            high["centroid_1050_x"],
            high["centroid_1050_y"],
            high["aligned_residual_x_um"],
            high["aligned_residual_y_um"],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
        )
        axis.set_title("High-match aligned residual vectors")
        axis.set_xlabel("1050 x (px)")
        axis.set_ylabel("1050 y (px)")
        axis.invert_yaxis()
        outputs["residual_vectors"] = _save_figure(root / "residual_vectors.png", figure)

        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].scatter(high["long_axis_position_normalized"], high["raw_delta_z_planes"], s=10)
        axes[0].plot(medians["bin_center"], medians["median_raw_delta_z_planes"], "o-")
        axes[0].set_title("Raw delta z vs long axis")
        axes[1].scatter(
            high["long_axis_position_normalized"],
            np.abs(high["raw_delta_z_planes"]),
            s=10,
        )
        axes[1].plot(medians["bin_center"], medians["median_abs_raw_delta_z_planes"], "o-")
        axes[1].set_title("Absolute raw delta z")
        axes[2].scatter(
            high["long_axis_position_normalized"],
            high["aligned_residual_distance_um"],
            s=10,
        )
        axes[2].plot(
            medians["bin_center"],
            medians["median_aligned_residual_distance_um"],
            "o-",
        )
        axes[2].set_title("Aligned residual distance")
        for axis in axes:
            axis.set_xlabel("Normalized long-axis position")
        outputs["long_axis_residuals"] = _save_figure(root / "long_axis_residuals.png", figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(coverage["bin"].astype(str), coverage["high_fraction"])
    axis.set_ylim(0, 1)
    axis.set_title("Primary high fraction among observable 1050 ROIs")
    axis.set_xlabel("Long-axis bin")
    axis.set_ylabel("High fraction")
    outputs["fixed_coverage"] = _save_figure(root / "fixed_coverage.png", figure)

    if identity_resolution is not None and not identity_resolution.empty:
        primary = set(identity_resolution.loc[identity_resolution["primary_green_status"].eq("high"), "label_1050"])
        secondary = set(identity_resolution.loc[identity_resolution["secondary_red_status"].eq("high"), "label_1050"])
        categories = ["green_high_only", "red_high_only", "both_high", "cross_source_conflict", "red_rescue_candidate"]
        counts = {
            "green_high_only": len(primary - secondary),
            "red_high_only": len(secondary - primary),
            "both_high": len(primary & secondary),
            "cross_source_conflict": int(identity_resolution["cross_source_conflict"].sum()),
            "red_rescue_candidate": int(identity_resolution["resolved_status"].eq("secondary_high_rescue_candidate").sum()),
        }
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(categories, [int(counts[category]) for category in categories])
        axis.set_title("Green/red source comparison")
        axis.tick_params(axis="x", labelrotation=20)
        outputs["source_comparison"] = _save_figure(root / "source_comparison.png", figure)
    return outputs


def _crop_plane(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    label: int,
    radius: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a native centered raw plane and matching label mask crop."""

    labels = np.argwhere(mask == int(label))
    if labels.size == 0:
        raise ValueError(f"Label {label} is absent from its native mask.")
    z, y, x = np.rint(labels.mean(axis=0)).astype(int)
    if image.ndim != 3 or image.shape != mask.shape:
        raise ValueError("Raw image and label mask must be same-shape ZYX stacks.")
    y0, y1 = max(y - radius, 0), min(y + radius + 1, image.shape[1])
    x0, x1 = max(x - radius, 0), min(x + radius + 1, image.shape[2])
    return image[z, y0:y1, x0:x1], mask[z, y0:y1, x0:x1] == int(label)


def render_cross_laser_contact_sheet(
    *,
    output_path: str | Path,
    fixed_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_image: np.ndarray,
    moving_mask: np.ndarray,
    pairs: pd.DataFrame,
    title: str,
    limit: int = 8,
) -> Path:
    """Render native raw-image examples for deterministic source-level pairs.

    Each side is centered in its own native coordinate space. The renderer is
    deliberately visual-only; no image intensity is used in correspondence.
    """

    required = {"label_1050", "label_920"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Example pairs are missing required columns: {sorted(missing)}")
    selected = pairs.iloc[:limit].copy()
    if selected.empty:
        raise ValueError("No cross-laser examples are available for this contact sheet.")
    figure, axes = plt.subplots(len(selected), 2, figsize=(7, 3 * len(selected)), squeeze=False)
    for row_index, row in enumerate(selected.itertuples(index=False)):
        fixed_plane, fixed_outline = _crop_plane(
            fixed_image, fixed_mask, label=int(row.label_1050)
        )
        moving_plane, moving_outline = _crop_plane(
            moving_image, moving_mask, label=int(row.label_920)
        )
        for axis, plane, outline, source in (
            (axes[row_index, 0], fixed_plane, fixed_outline, "1050 red"),
            (axes[row_index, 1], moving_plane, moving_outline, "920 native"),
        ):
            axis.imshow(plane, cmap="gray")
            axis.contour(outline, levels=[0.5], colors=["cyan"], linewidths=0.8)
            axis.set_title(source)
            axis.set_axis_off()
        score = getattr(row, "score", np.nan)
        dice = getattr(row, "dice", np.nan)
        distance = getattr(row, "distance_um", np.nan)
        raw_dz = getattr(row, "raw_delta_z_planes", np.nan)
        residual = getattr(row, "aligned_residual_distance_um", np.nan)
        figure.text(
            0.5,
            (len(selected) - row_index - 0.97) / len(selected),
            (
                f"1050={int(row.label_1050)} 920={int(row.label_920)} "
                f"score={score:.3g} dice={dice:.3g} dist={distance:.3g} um "
                f"raw dz={raw_dz:+.2f} residual={residual:.3g} um"
            ),
            ha="center",
            fontsize=8,
        )
    figure.suptitle(title, y=0.995)
    return _save_figure(Path(output_path), figure)
