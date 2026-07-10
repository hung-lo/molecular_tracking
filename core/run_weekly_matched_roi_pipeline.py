"""Run week-matched ROI extraction from weekly average masks and registered day stacks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
import argparse
import json
import time

import pandas as pd
import tifffile

from analysis_paths import get_dataset_analysis_dir, resolve_dataset_dir
from roi_log_ratio_analysis import (
    apply_channel_dark_correction,
    build_registered_image_lookup,
    compute_green_red_fit_residuals,
    compute_log_ratio_metrics,
    extract_roi_mean_intensities,
    filter_complete_rois,
    summarize_daily_green_red_linear_fits,
    summarize_residual_sign_changes,
    wide_table_from_long_table,
)
from run_registered_roi_pipeline import infer_start_date_from_dataset_dir


ANALYSIS_VERSION = "0.1.0"


@dataclass(frozen=True)
class WeeklyMatchedPipelineConfig:
    """Store parameters for the weekly matched ROI extraction workflow."""

    dataset: str
    match_csv: str
    start_date: str | None = None
    week_mask_template: str = "{week_name}_average_cp_masks.tif"
    green_dark: float = 319.0
    red_dark: float = 534.0
    epsilon: float = 1.0


def format_duration_seconds(duration_seconds: float) -> str:
    """Format an elapsed wall-clock duration as ``HH:MM:SS``."""

    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_stage_start(stage_label: str, pipeline_start_seconds: float) -> float:
    """Print a stage-start message and return the stage timer."""

    elapsed = time.perf_counter() - pipeline_start_seconds
    print(f"[{format_duration_seconds(elapsed)}] {stage_label}...", flush=True)
    return time.perf_counter()


def log_stage_end(
    stage_label: str,
    pipeline_start_seconds: float,
    stage_start_seconds: float,
    stage_durations_seconds: dict[str, float],
    stage_key: str,
    detail: str | None = None,
) -> None:
    """Print a stage-end message and persist the duration."""

    stage_duration = time.perf_counter() - stage_start_seconds
    stage_durations_seconds[stage_key] = float(stage_duration)
    elapsed = time.perf_counter() - pipeline_start_seconds
    detail_suffix = "" if not detail else f" | {detail}"
    print(
        f"[{format_duration_seconds(elapsed)}] Finished {stage_label} "
        f"({stage_duration:.1f}s){detail_suffix}",
        flush=True,
    )


def extract_ordered_week_names(match_table: pd.DataFrame) -> list[str]:
    """Return sorted matcher week names such as ``["week1", "week2"]``."""

    week_names = []
    for column in match_table.columns:
        if not column.startswith("week") or not column.endswith("_roi"):
            continue
        week_label = column[:-4]
        suffix = week_label.removeprefix("week")
        if suffix.isdigit():
            week_names.append(week_label)

    if not week_names:
        raise ValueError("match_csv must include week ROI columns such as 'week1_roi'.")

    return sorted(week_names, key=lambda value: int(value.removeprefix("week")))


def build_calendar_week_groups(
    registered_lookup: dict[tuple[int, str], Path],
) -> list[dict[str, object]]:
    """Group registered imaging days into chronological Monday-based weeks."""

    day_to_date: dict[int, date] = {}
    for (day, _channel), path in sorted(registered_lookup.items()):
        if day in day_to_date:
            continue
        day_to_date[int(day)] = datetime.strptime(path.name[:8], "%Y%m%d").date()

    grouped_days: dict[date, list[int]] = defaultdict(list)
    for day, imaging_date in sorted(day_to_date.items()):
        week_start = imaging_date.fromordinal(imaging_date.toordinal() - imaging_date.weekday())
        grouped_days[week_start].append(int(day))

    groups: list[dict[str, object]] = []
    for week_start in sorted(grouped_days):
        days = sorted(grouped_days[week_start])
        groups.append(
            {
                "week_start": week_start,
                "days": days,
                "date_keys": [day_to_date[day].strftime("%Y%m%d") for day in days],
            }
        )
    return groups


def validate_registered_lookup(
    registered_lookup: dict[tuple[int, str], Path],
    channels: tuple[str, ...] = ("red", "green"),
) -> list[int]:
    """Ensure every detected day has all required channel images."""

    required_days = sorted({int(day) for day, _channel in registered_lookup})
    missing_entries = [
        f"day {day} {channel}"
        for day in required_days
        for channel in channels
        if (day, channel) not in registered_lookup
    ]
    if missing_entries:
        missing_str = ", ".join(missing_entries)
        raise FileNotFoundError(f"Missing registered TIFFs for: {missing_str}")
    return required_days


def resolve_week_mask_path(
    dataset_dir: str | Path,
    week_name: str,
    week_mask_template: str = "{week_name}_average_cp_masks.tif",
) -> Path:
    """Resolve one week-average mask path, with singular/plural fallback."""

    dataset_dir = Path(dataset_dir)
    candidate_names = [week_mask_template.format(week_name=week_name)]
    candidate_names.extend(
        [
            f"{week_name}_average_cp_masks.tif",
            f"{week_name}_average_cp_mask.tif",
        ]
    )

    seen_names: set[str] = set()
    for candidate_name in candidate_names:
        if candidate_name in seen_names:
            continue
        seen_names.add(candidate_name)
        candidate_path = dataset_dir / candidate_name
        if candidate_path.exists():
            return candidate_path

    candidate_str = ", ".join(candidate_names)
    raise FileNotFoundError(f"No week mask found for {week_name}. Tried: {candidate_str}")


def extract_weekly_matched_roi_intensity_table(
    dataset_dir: str | Path,
    match_table: pd.DataFrame,
    start_date: str,
    week_mask_template: str = "{week_name}_average_cp_masks.tif",
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Measure matched cluster intensities from weekly masks and daywise SyN stacks."""

    dataset_dir = Path(dataset_dir)
    registered_lookup = build_registered_image_lookup(
        image_dir=dataset_dir,
        start_date=start_date,
        day0_mode="syn",
    )
    required_days = validate_registered_lookup(registered_lookup)

    week_names = extract_ordered_week_names(match_table)
    calendar_groups = build_calendar_week_groups(registered_lookup)
    if len(calendar_groups) != len(week_names):
        raise ValueError(
            "Matcher weeks do not match detected calendar weeks: "
            f"{len(week_names)} matcher weeks vs {len(calendar_groups)} detected groups."
        )

    week_assignment_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for week_name, group in zip(week_names, calendar_groups, strict=True):
        mask_path = resolve_week_mask_path(
            dataset_dir=dataset_dir,
            week_name=week_name,
            week_mask_template=week_mask_template,
        )
        mask_stack = tifffile.imread(mask_path)

        week_column = f"{week_name}_roi"
        cluster_lookup = match_table.loc[
            match_table[week_column].notna(),
            ["cluster_id", week_column],
        ].copy()
        cluster_lookup["cluster_id"] = pd.to_numeric(cluster_lookup["cluster_id"], errors="raise").astype(int)
        cluster_lookup["week_roi_id"] = pd.to_numeric(cluster_lookup[week_column], errors="raise").astype(int)
        cluster_lookup = cluster_lookup.loc[:, ["cluster_id", "week_roi_id"]]
        if cluster_lookup["week_roi_id"].duplicated().any():
            duplicated_ids = sorted(cluster_lookup.loc[cluster_lookup["week_roi_id"].duplicated(), "week_roi_id"].unique().tolist())
            raise ValueError(f"Duplicate {week_name} ROI ids in match_csv: {duplicated_ids}")

        for day, date_key in zip(group["days"], group["date_keys"], strict=True):
            week_assignment_rows.append(
                {
                    "week_name": week_name,
                    "week_start": group["week_start"].isoformat(),
                    "day": int(day),
                    "date": str(date_key),
                    "mask_name": mask_path.name,
                }
            )
            for channel in ("red", "green"):
                image_path = registered_lookup[(int(day), channel)]
                image_stack = tifffile.imread(image_path)
                extracted_rows = extract_roi_mean_intensities(
                    image_stack=image_stack,
                    mask_stack=mask_stack,
                    exclude_zero_pixels=True,
                )
                if not extracted_rows:
                    continue

                extracted_table = pd.DataFrame(extracted_rows).rename(columns={"roi_id": "week_roi_id"})
                matched_table = cluster_lookup.merge(
                    extracted_table,
                    on="week_roi_id",
                    how="inner",
                    validate="one_to_one",
                )
                if matched_table.empty:
                    continue

                matched_table["image"] = image_path.name
                matched_table["channel"] = channel
                matched_table["day"] = int(day)
                matched_table["date"] = str(date_key)
                matched_table["week_name"] = week_name
                matched_table["mask_name"] = mask_path.name
                raw_rows.extend(
                    matched_table[
                        [
                            "cluster_id",
                            "week_name",
                            "week_roi_id",
                            "mean_intensity",
                            "image",
                            "channel",
                            "day",
                            "date",
                            "mask_name",
                        ]
                    ].to_dict(orient="records")
                )

    raw_table = pd.DataFrame(raw_rows)
    if not raw_table.empty:
        raw_table = raw_table.sort_values(["cluster_id", "day", "channel"]).reset_index(drop=True)

    week_assignment_table = pd.DataFrame(week_assignment_rows).sort_values(["week_name", "day"]).reset_index(drop=True)
    return raw_table, week_assignment_table, required_days


def prepare_internal_roi_table(cluster_table: pd.DataFrame) -> pd.DataFrame:
    """Rename ``cluster_id`` to ``roi_id`` for reuse of existing helpers."""

    output = cluster_table.copy()
    output["cluster_id"] = pd.to_numeric(output["cluster_id"], errors="raise").astype(int)
    output = output.rename(columns={"cluster_id": "roi_id"})
    return output


def restore_cluster_id_column(table: pd.DataFrame) -> pd.DataFrame:
    """Rename the internal ``roi_id`` column back to ``cluster_id``."""

    output = table.copy()
    if "roi_id" in output.columns:
        output = output.rename(columns={"roi_id": "cluster_id"})
    return output


def write_summary_markdown(
    output_dir: Path,
    config: WeeklyMatchedPipelineConfig,
    filter_counts: pd.DataFrame,
    fit_summary: pd.DataFrame,
    week_assignment_table: pd.DataFrame,
) -> None:
    """Write a short markdown summary for one weekly matched ROI run."""

    week_lines = [
        f"- {row['week_name']}: day {int(row['day'])} ({row['date']}) using `{row['mask_name']}`"
        for _, row in week_assignment_table.iterrows()
    ]
    summary_lines = [
        "# Weekly Matched ROI Pipeline",
        "",
        "Goal: measure matched ROI clusters across weekly average masks using",
        "daywise registered (`*_SyN`) stacks, then apply the same zero-hit and",
        "complete-day filtering used in the registered ROI workflow.",
        "",
        "Key analysis choices:",
        f"- Dataset: `{config.dataset}`",
        f"- Matcher CSV: `{config.match_csv}`",
        f"- Start date (day 0): `{config.start_date}`",
        f"- Week mask template: `{config.week_mask_template}`",
        f"- Dark values: green `{config.green_dark}`, red `{config.red_dark}`",
        "",
        "Week assignments:",
        *week_lines,
        "",
        "Filter summary:",
    ]
    for _, row in filter_counts.iterrows():
        summary_lines.append(
            f"- {row['step']}: `{int(row['count'])}` clusters "
            f"({row['pct_of_start']:.1f}% of matcher clusters)"
        )
    if not fit_summary.empty:
        summary_lines.extend(["", "Green-vs-red fit summary:"])
        for _, row in fit_summary.iterrows():
            summary_lines.append(
                f"- Day {int(row['day'])}: slope `{row['slope']:.3f}`, "
                f"intercept `{row['intercept']:.1f}`, R² `{row['r_squared']:.3f}`"
            )
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")


def write_run_log(
    output_dir: Path,
    config: WeeklyMatchedPipelineConfig,
    filter_counts: pd.DataFrame,
    week_assignment_table: pd.DataFrame,
    stage_durations_seconds: dict[str, float],
) -> None:
    """Write a JSON run log for reproducibility."""

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "run_timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "filter_counts": filter_counts.to_dict(orient="records"),
        "week_assignments": week_assignment_table.to_dict(orient="records"),
        "stage_durations_seconds": stage_durations_seconds,
    }
    (output_dir / "run_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_weekly_matched_roi_pipeline(config: WeeklyMatchedPipelineConfig) -> Path:
    """Run the weekly matched ROI extraction pipeline."""

    pipeline_start_seconds = time.perf_counter()
    stage_durations_seconds: dict[str, float] = {}
    dataset_dir = resolve_dataset_dir(config.dataset)
    match_csv_path = Path(config.match_csv).resolve()
    if not match_csv_path.exists():
        raise FileNotFoundError(f"match_csv was not found: {match_csv_path}")

    effective_start_date = config.start_date or infer_start_date_from_dataset_dir(dataset_dir)
    config = WeeklyMatchedPipelineConfig(**{**asdict(config), "start_date": effective_start_date})

    analysis_root = get_dataset_analysis_dir(config.dataset)
    analysis_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = analysis_root / f"weekly_matched_roi_pipeline_{match_csv_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"[{format_duration_seconds(0.0)}] Pipeline start | dataset={config.dataset} | "
        f"day0={config.start_date} | match_csv={match_csv_path.name}",
        flush=True,
    )
    print(f"[{format_duration_seconds(0.0)}] Output directory: {output_dir}", flush=True)

    stage_start_seconds = log_stage_start(
        stage_label="Load matcher CSV and extract matched ROI intensities",
        pipeline_start_seconds=pipeline_start_seconds,
    )
    match_table = pd.read_csv(match_csv_path)
    raw_table, week_assignment_table, required_days = extract_weekly_matched_roi_intensity_table(
        dataset_dir=dataset_dir,
        match_table=match_table,
        start_date=config.start_date,
        week_mask_template=config.week_mask_template,
    )
    raw_table.to_csv(output_dir / "weekly_matched_roi_intensity_results_raw.csv", index=False)
    week_assignment_table.to_csv(output_dir / "week_assignments.csv", index=False)
    log_stage_end(
        stage_label="Load matcher CSV and extract matched ROI intensities",
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        stage_key="extract_weekly_matched_intensities",
        detail=f"{raw_table['cluster_id'].nunique() if not raw_table.empty else 0} clusters observed after zero filtering",
    )

    stage_start_seconds = log_stage_start(
        stage_label="Dark correction and complete-day cluster table",
        pipeline_start_seconds=pipeline_start_seconds,
    )
    working_raw_table = prepare_internal_roi_table(raw_table)
    corrected_table = apply_channel_dark_correction(
        intensity_table=working_raw_table,
        green_dark=config.green_dark,
        red_dark=config.red_dark,
        intensity_column="mean_intensity",
        corrected_column="mean_intensity_corrected",
        clip_floor=None,
    )
    roi_day_table = wide_table_from_long_table(
        corrected_table,
        intensity_column="mean_intensity_corrected",
        start_date=config.start_date,
    )
    complete_table = filter_complete_rois(roi_day_table, required_days=required_days)
    corrected_export = restore_cluster_id_column(corrected_table)
    complete_export = restore_cluster_id_column(complete_table)
    corrected_export.to_csv(output_dir / "weekly_matched_roi_intensity_results_dark_corrected.csv", index=False)
    complete_export.to_csv(output_dir / "weekly_matched_roi_day_table_complete.csv", index=False)
    log_stage_end(
        stage_label="Dark correction and complete-day cluster table",
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        stage_key="dark_correction_and_complete_days",
        detail=f"{complete_export['cluster_id'].nunique() if not complete_export.empty else 0} clusters present on all detected days",
    )

    fit_summary = pd.DataFrame()
    residual_export = pd.DataFrame()
    metrics_export = pd.DataFrame()
    residual_summary_export = pd.DataFrame()
    if not complete_table.empty:
        stage_start_seconds = log_stage_start(
            stage_label="Log-ratio metrics and green-vs-red residuals",
            pipeline_start_seconds=pipeline_start_seconds,
        )
        metrics_table = compute_log_ratio_metrics(complete_table, epsilon=config.epsilon)
        fit_summary = summarize_daily_green_red_linear_fits(metrics_table)
        residual_table = compute_green_red_fit_residuals(metrics_table, fit_summary=fit_summary)
        residual_summary = summarize_residual_sign_changes(residual_table)
        metrics_with_residuals = residual_table.merge(
            residual_summary,
            on="roi_id",
            how="left",
            validate="many_to_one",
        )
        metrics_export = restore_cluster_id_column(metrics_table)
        residual_export = restore_cluster_id_column(metrics_with_residuals)
        residual_summary_export = restore_cluster_id_column(residual_summary)
        metrics_export.to_csv(output_dir / "weekly_matched_roi_log_ratio_metrics_complete.csv", index=False)
        fit_summary.to_csv(output_dir / "weekly_matched_daywise_green_red_linear_fit_summary.csv", index=False)
        residual_export.to_csv(output_dir / "weekly_matched_roi_metrics_with_green_red_fit_residuals.csv", index=False)
        residual_summary_export.to_csv(output_dir / "weekly_matched_roi_residual_sign_change_summary.csv", index=False)
        log_stage_end(
            stage_label="Log-ratio metrics and green-vs-red residuals",
            pipeline_start_seconds=pipeline_start_seconds,
            stage_start_seconds=stage_start_seconds,
            stage_durations_seconds=stage_durations_seconds,
            stage_key="metrics_and_residuals",
            detail=f"{len(fit_summary)} imaging days summarized",
        )

    filter_counts = pd.DataFrame(
        [
            {"step": "matcher_clusters", "count": int(match_table["cluster_id"].nunique())},
            {"step": "observed_after_zero_filter", "count": int(raw_table["cluster_id"].nunique()) if not raw_table.empty else 0},
            {"step": "complete_days", "count": int(complete_export["cluster_id"].nunique()) if not complete_export.empty else 0},
        ]
    )
    starting_count = float(filter_counts["count"].iloc[0]) if len(filter_counts) else 0.0
    if starting_count > 0:
        filter_counts["pct_of_start"] = 100.0 * filter_counts["count"].astype(float) / starting_count
        filter_counts["pct_of_previous"] = (
            100.0
            * filter_counts["count"].astype(float)
            / filter_counts["count"].shift(1, fill_value=starting_count).astype(float)
        )
    else:
        filter_counts["pct_of_start"] = 0.0
        filter_counts["pct_of_previous"] = 0.0
    filter_counts.to_csv(output_dir / "filter_step_counts_with_percentages.csv", index=False)

    write_summary_markdown(
        output_dir=output_dir,
        config=config,
        filter_counts=filter_counts,
        fit_summary=fit_summary,
        week_assignment_table=week_assignment_table,
    )
    write_run_log(
        output_dir=output_dir,
        config=config,
        filter_counts=filter_counts,
        week_assignment_table=week_assignment_table,
        stage_durations_seconds=stage_durations_seconds,
    )

    total_duration_seconds = time.perf_counter() - pipeline_start_seconds
    print(f"[{format_duration_seconds(total_duration_seconds)}] Pipeline completed", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"n_matcher_clusters={int(filter_counts.loc[0, 'count'])}", flush=True)
    print(f"n_complete_day_clusters={int(filter_counts.loc[2, 'count'])}", flush=True)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the weekly matched ROI pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset alias or explicit dataset directory path.",
    )
    parser.add_argument(
        "--match-csv",
        required=True,
        help="CSV exported from the ROI matcher with columns like week1_roi, week2_roi, ...",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional reference date in YYYYMMDD format. If omitted, the earliest raw TIFF date is used.",
    )
    parser.add_argument(
        "--week-mask-template",
        default="{week_name}_average_cp_masks.tif",
        help="Filename template for weekly average masks inside the dataset directory.",
    )
    parser.add_argument("--green-dark", type=float, default=319.0, help="Green-channel dark offset.")
    parser.add_argument("--red-dark", type=float, default=534.0, help="Red-channel dark offset.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Positive offset used for log-ratio metrics.")
    return parser.parse_args(argv)


def main() -> None:
    """Run the weekly matched ROI pipeline from the command line."""

    args = parse_args()
    config = WeeklyMatchedPipelineConfig(
        dataset=args.dataset,
        match_csv=args.match_csv,
        start_date=args.start_date,
        week_mask_template=args.week_mask_template,
        green_dark=args.green_dark,
        red_dark=args.red_dark,
        epsilon=args.epsilon,
    )
    run_weekly_matched_roi_pipeline(config)


if __name__ == "__main__":
    main()
