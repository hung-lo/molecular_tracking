"""Render quick QC plots from a completed matched ROI analysis output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import sys
import time

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _import_dir in (
    _REPO_ROOT / "core",
    _REPO_ROOT / "plotting",
    _REPO_ROOT / "matching",
):
    _import_dir_str = str(_import_dir)
    if _import_dir_str not in sys.path:
        sys.path.append(_import_dir_str)


from roi_log_ratio_analysis import (
    select_ranked_roi_days,
    select_top_changing_rois,
    summarize_roi_metrics,
)
from run_registered_roi_pipeline import plot_population_summary, plot_ranked_heatmap
from run_daywise_green_red_fit_residuals import (
    plot_directional_roi_residuals_vs_day,
    plot_directional_roi_trajectories_scatter,
)
from run_daywise_green_red_linear_fit_summary import (
    plot_daywise_scatter_summary,
    plot_fit_parameter_summary,
)


def format_duration_seconds(duration_seconds: float) -> str:
    """Format elapsed wall-clock time as ``HH:MM:SS``."""

    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_message(run_start_seconds: float, message: str) -> None:
    """Print one elapsed-time progress message."""

    elapsed_seconds = time.perf_counter() - run_start_seconds
    print(f"[{format_duration_seconds(elapsed_seconds)}] {message}", flush=True)


def write_run_log(
    output_dir: Path,
    analysis_dir: Path,
    start_date: str,
    top_n: int,
    policy: str,
    total_duration_seconds: float,
) -> None:
    """Write a compact text run log for the quick-plot export."""

    log_lines = [
        f"run_timestamp={datetime.now().isoformat()}",
        f"analysis_dir={analysis_dir}",
        f"start_date={start_date}",
        f"top_n={int(top_n)}",
        f"policy={policy}",
        f"total_duration_seconds={float(total_duration_seconds):.3f}",
        f"total_duration_hms={format_duration_seconds(total_duration_seconds)}",
    ]
    (output_dir / "run_log.txt").write_text("\n".join(log_lines), encoding='utf-8')


def _load_metrics_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if "roi_id" not in table.columns and "cluster_id" in table.columns:
        table = table.rename(columns={"cluster_id": "roi_id"})
    if "roi_id" not in table.columns:
        raise ValueError(f"Missing roi_id/cluster_id column in {path}")
    return table


def _load_fit_summary(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required_columns = {"day", "slope", "intercept", "r_squared", "n_rois"}
    missing = required_columns.difference(table.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required fit-summary columns in {path}: {missing_str}")
    return table


def _first_existing_path(analysis_dir: Path, *names: str) -> Path:
    for name in names:
        candidate = analysis_dir / name
        if candidate.exists():
            return candidate
    expected = ", ".join(names)
    raise FileNotFoundError(f"None of the expected input tables were found in {analysis_dir}: {expected}")


def _filter_table_by_policy(table: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Keep a single match policy when policy-specific data is available."""

    policy_columns = [column for column in table.columns if column == "match_policy" or column.startswith("match_policy_")]
    if table.empty or policy == "all" or not policy_columns:
        return table
    policy_mask = pd.Series(True, index=table.index)
    for column in policy_columns:
        policy_mask &= table[column].astype(str) == policy
    filtered = table.loc[policy_mask].copy()
    if filtered.empty:
        available = ", ".join(sorted({str(value) for column in policy_columns for value in table[column].dropna().astype(str).unique()}))
        raise ValueError(
            f"Requested match policy {policy!r} was not found; available policies: {available or 'none'}"
        )
    return filtered.reset_index(drop=True)


def _resolve_output_dir(analysis_dir: Path, output_dir: str | Path | None, policy: str) -> Path:
    """Resolve a policy-specific output directory for quick plots."""

    base_output_dir = Path(output_dir).resolve() if output_dir else (analysis_dir / "quick_plots")
    return base_output_dir / policy


def build_quick_plots(
    analysis_dir: str | Path,
    start_date: str = "20260511",
    top_n: int = 30,
    policy: str = "high",
    output_dir: str | Path | None = None,
) -> Path:
    """Render a small set of no-rerun QC plots from saved matched output tables."""

    run_start_seconds = time.perf_counter()
    analysis_dir = Path(analysis_dir)
    log_message(run_start_seconds, f"Starting matched ROI quick plots | analysis_dir={analysis_dir}")
    metrics_path = _first_existing_path(
        analysis_dir,
        "weekly_matched_roi_log_ratio_metrics_complete.csv",
        "matched_roi_log_ratio_metrics_complete.csv",
    )
    residuals_path = _first_existing_path(
        analysis_dir,
        "weekly_matched_roi_metrics_with_green_red_fit_residuals.csv",
        "matched_roi_metrics_with_green_red_fit_residuals.csv",
    )
    fit_summary_path = _first_existing_path(
        analysis_dir,
        "weekly_matched_daywise_green_red_linear_fit_summary.csv",
        "matched_daywise_green_red_linear_fit_summary.csv",
    )

    log_message(run_start_seconds, f"Loading saved matched tables from {analysis_dir}")
    metrics_table = _filter_table_by_policy(_load_metrics_table(metrics_path), policy)
    residuals_table = _filter_table_by_policy(_load_metrics_table(residuals_path), policy)
    fit_summary = _filter_table_by_policy(_load_fit_summary(fit_summary_path), policy)
    unique_days = int(fit_summary["day"].nunique()) if not fit_summary.empty and "day" in fit_summary.columns else 0
    log_message(
        run_start_seconds,
        f"Policy filter: {policy} | metric_rows={len(metrics_table)} | fit_rows={len(fit_summary)} | unique_days={unique_days}",
    )

    output_dir = _resolve_output_dir(analysis_dir, output_dir, policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_message(run_start_seconds, f"Output directory: {output_dir}")
    log_message(run_start_seconds, "Rendering population and day-wise fit summary plots")

    plot_population_summary(
        roi_metrics=metrics_table,
        output_path=output_dir / "population_longitudinal_summary.png",
        start_date=start_date,
        include_traces=False,
    )
    plot_population_summary(
        roi_metrics=metrics_table,
        output_path=output_dir / "population_longitudinal_summary_with_traces.png",
        start_date=start_date,
        include_traces=True,
    )
    plot_daywise_scatter_summary(
        roi_metrics=metrics_table,
        fit_summary=fit_summary,
        output_path=output_dir / "daywise_green_red_linear_fit_scatters.png",
        start_date=start_date,
    )
    plot_fit_parameter_summary(
        fit_summary=fit_summary,
        output_path=output_dir / "daywise_green_red_linear_fit_parameters.png",
        start_date=start_date,
    )

    roi_summary = summarize_roi_metrics(metrics_table)
    ranking_columns = [
        "selection_rank",
        "min_delta_log2_green_over_red",
        "max_delta_log2_green_over_red",
        "delta_log2_range",
        "red_cv",
        "day0_brightness",
    ]

    for direction_label in ("decreasing", "increasing"):
        top_rois = select_top_changing_rois(
            roi_summary=roi_summary,
            max_rois=top_n,
            direction=direction_label,
        )
        if top_rois.empty:
            log_message(run_start_seconds, f"No ranked {direction_label} clusters found; skipping directional plots")
            continue

        log_message(
            run_start_seconds,
            f"Rendering {direction_label} directional plots for {len(top_rois)} ranked clusters",
        )
        ranked_roi_days = select_ranked_roi_days(
            roi_day_table=residuals_table,
            ranking_table=top_rois,
            top_n=min(int(top_n), len(top_rois)),
            ranking_columns=ranking_columns,
        )
        ranked_roi_days.to_csv(
            output_dir / f"top{len(top_rois)}_{direction_label}_roi_day_metrics.csv",
            index=False,
        )
        top_rois.to_csv(
            output_dir / f"top{len(top_rois)}_{direction_label}_clusters.csv",
            index=False,
        )

        plot_ranked_heatmap(
            ranked_roi_days=ranked_roi_days,
            value_column="delta_log2_green_over_red",
            output_path=output_dir / f"top{len(top_rois)}_{direction_label}_delta_log2_heatmap.png",
            start_date=start_date,
            title=f"Top {len(top_rois)} {direction_label} matched clusters: delta log2(G/R)",
            colorbar_label="Delta log2(G/R)",
            cmap="coolwarm",
            center_zero=True,
        )
        plot_directional_roi_residuals_vs_day(
            ranked_roi_table=ranked_roi_days,
            output_path=output_dir / f"top{len(top_rois)}_{direction_label}_green_fit_residual_vs_day.png",
            title_prefix=f"Top {len(top_rois)}",
            direction_label=direction_label,
            start_date=start_date,
        )
        plot_directional_roi_trajectories_scatter(
            ranked_roi_table=ranked_roi_days,
            fit_summary=fit_summary,
            output_path=output_dir / f"top{len(top_rois)}_{direction_label}_green_red_trajectory_scatter_average_fit.png",
            title_prefix=f"Top {len(top_rois)}",
            direction_label=direction_label,
            start_date=start_date,
        )

    summary_lines = [
        "Quick plots generated from saved matched ROI outputs.",
        f"Analysis directory: {analysis_dir}",
        f"Start date: {start_date}",
        f"Top-N directional plots: {top_n}",
        f"Policy filter: {policy}",
        "",
        "Core plots:",
        "- population_longitudinal_summary.png",
        "- population_longitudinal_summary_with_traces.png",
        "- daywise_green_red_linear_fit_scatters.png",
        "- daywise_green_red_linear_fit_parameters.png",
        "",
        "Directional plots are written when ranked clusters are available.",
    ]
    (output_dir / "README.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    total_duration_seconds = time.perf_counter() - run_start_seconds
    write_run_log(
        output_dir=output_dir,
        analysis_dir=analysis_dir,
        start_date=start_date,
        top_n=top_n,
        policy=policy,
        total_duration_seconds=total_duration_seconds,
    )
    print(f"[{format_duration_seconds(total_duration_seconds)}] Completed matched ROI quick plots", flush=True)
    print(f"total_duration={format_duration_seconds(total_duration_seconds)}")
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the quick-plot renderer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, help="Completed matched ROI analysis output directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional root directory for the quick-plot exports; policy-specific subfolders are created beneath it.",
    )
    parser.add_argument("--start-date", default="20260511", help="Reference date in YYYYMMDD format.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top increasing/decreasing clusters to plot.")
    parser.add_argument(
        "--policy",
        default="high",
        choices=["high", "balanced", "all"],
        help="When match_policy is present, choose which policy to plot. Use 'all' to keep both.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Render quick QC plots from an existing analysis output directory."""

    args = parse_args()
    output_dir = build_quick_plots(
        analysis_dir=args.analysis_dir,
        start_date=args.start_date,
        top_n=args.top_n,
        policy=args.policy,
        output_dir=args.output_dir,
    )
    print(f"quick_plot_dir={output_dir}")


if __name__ == "__main__":
    main()
