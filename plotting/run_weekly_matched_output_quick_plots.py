"""Render quick QC plots from a completed matched ROI analysis output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import json
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


from project_cli import add_project_selector, resolve_analysis_selection
from project_config import validate_output_path, is_relative_to
from roi_log_ratio_analysis import (
    has_usable_day0_baseline,
    select_ranked_roi_days,
    select_top_changing_rois,
    summarize_roi_metrics,
    summarize_window_metrics_for_ranking,
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


def _load_native_fit_population(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"day", "session_id", "session_roi_label", "red", "green"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing native fit-population columns in {path}: {sorted(missing)}")
    if {"red_signal_qc_pass", "green_signal_qc_pass"}.issubset(table.columns):
        table = table.loc[table["red_signal_qc_pass"].eq(True) & table["green_signal_qc_pass"].eq(True)].copy()
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


def _first_nonempty_metrics_path(analysis_dir: Path, *names: str) -> Path:
    existing = [analysis_dir / name for name in names if (analysis_dir / name).is_file()]
    if not existing:
        raise FileNotFoundError(f"None of the expected input tables were found in {analysis_dir}: {', '.join(names)}")
    for candidate in existing:
        try:
            if not pd.read_csv(candidate, nrows=1).empty:
                return candidate
        except pd.errors.EmptyDataError:
            continue
    return existing[0]


def _read_window_selection(analysis_dir: Path) -> tuple[dict[str, object] | None, str]:
    path = analysis_dir / "longest_supported_window.json"
    if not path.is_file():
        return None, "longest_supported_window_metadata_missing"
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "longest_supported_window_metadata_invalid"
    if selection.get("status") != "valid" or not selection.get("window_session_indices"):
        return selection, "no_supported_window"
    return selection, ""


def _window_residual_table(
    window_metrics: pd.DataFrame,
    residuals_table: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Attach native-session fit residuals by track and source-session keys."""

    output = window_metrics.copy()
    residual_columns = [
        column for column in (
            "fit_status", "fit_reason", "green_fit_residual", "green_fit_signed_distance",
            "predicted_green_from_fit",
        )
        if column in residuals_table.columns
    ]
    for column in residual_columns:
        output = output.drop(columns=column, errors="ignore")
    if not residual_columns:
        output["green_fit_residual"] = float("nan")
        return output, "residual_columns_unavailable"
    required_keys = ["roi_id", "session_index", "session_id"]
    if not set(required_keys).issubset(output.columns) or not set(required_keys).issubset(residuals_table.columns):
        output["green_fit_residual"] = float("nan")
        return output, "source_session_join_keys_unavailable"
    keys = [*required_keys]
    if "day" in output and "day" in residuals_table:
        keys.append("day")
    if "match_policy" in output and "match_policy" in residuals_table:
        keys.append("match_policy")
    left = output.copy()
    right = residuals_table.loc[:, [*keys, *residual_columns]].copy()
    for frame in (left, right):
        frame["roi_id"] = pd.to_numeric(frame["roi_id"], errors="coerce")
        frame["session_index"] = pd.to_numeric(frame["session_index"], errors="coerce")
        frame["session_id"] = frame["session_id"].astype(str)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        output["green_fit_residual"] = float("nan")
        return output, "source_session_join_not_unique"
    try:
        output = left.merge(right, on=keys, how="left", validate="one_to_one")
    except (pd.errors.MergeError, ValueError):
        output = left.copy()
        output["green_fit_residual"] = float("nan")
        return output, "source_session_join_failed"
    if "green_fit_residual" not in output:
        output["green_fit_residual"] = float("nan")
    return output, "joined_by_roi_and_source_session"


def _ranking_provenance(
    *,
    analysis_dir: Path,
    population: str,
    metric_column: str,
    input_metrics_path: Path,
    residual_input_path: Path,
    selection: dict[str, object] | None,
    metrics: pd.DataFrame,
    residual_join_status: str,
) -> dict[str, object]:
    manifest_path = analysis_dir / "session_manifest_resolved.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    window_indices = [int(value) for value in (selection or {}).get("window_session_indices", [])]
    window_ids = [str(value) for value in (selection or {}).get("window_session_ids", [])]
    window_dates = [str(value) for value in (selection or {}).get("window_acquisition_dates", [])]
    if population == "longest_supported_window" and window_indices:
        baseline_index = window_indices[0]
        baseline_id = window_ids[0] if window_ids else ""
        baseline_date = window_dates[0] if window_dates else ""
    elif population == "longest_supported_window":
        baseline_index = None
        baseline_id = ""
        baseline_date = ""
    else:
        baseline_index = 0
        baseline_id = ""
        baseline_date = ""
        if not manifest.empty and "session_index" in manifest:
            baseline_rows = manifest.loc[pd.to_numeric(manifest["session_index"], errors="coerce").eq(0)]
            if not baseline_rows.empty:
                baseline_id = str(baseline_rows.iloc[0].get("session_id", ""))
                baseline_date = str(baseline_rows.iloc[0].get("acquisition_date", ""))
        if not baseline_id and "session_id" in metrics and "day" in metrics:
            baseline_rows = metrics.loc[metrics["day"].eq(0)]
            if not baseline_rows.empty:
                baseline_id = str(baseline_rows.iloc[0]["session_id"])
                baseline_date = str(baseline_rows.iloc[0].get("acquisition_date", ""))
    first_manifest = manifest.sort_values("session_index").iloc[0] if not manifest.empty and "session_index" in manifest else None
    first_absent_note = ""
    if population == "longest_supported_window" and first_manifest is not None:
        first_index = int(first_manifest["session_index"])
        if first_index not in window_indices:
            first_absent_note = (
                f"Source session {first_index} ({first_manifest.get('session_id', '')}, "
                f"{first_manifest.get('acquisition_date', '')}) remains in the manifest but has no rows in the selected-window cohort."
            )
    return {
        "ranking_population": population,
        "ranking_baseline_session_index": baseline_index,
        "ranking_baseline_session_id": baseline_id,
        "ranking_baseline_acquisition_date": baseline_date,
        "window_status": (selection or {}).get("status", "unavailable"),
        "window_start_session_index": window_indices[0] if window_indices else None,
        "window_end_session_index": window_indices[-1] if window_indices else None,
        "window_start_session_id": window_ids[0] if window_ids else "",
        "window_end_session_id": window_ids[-1] if window_ids else "",
        "window_start_acquisition_date": window_dates[0] if window_dates else "",
        "window_end_acquisition_date": window_dates[-1] if window_dates else "",
        "window_n_source_sessions": len(window_indices),
        "window_n_candidate_tracks": int((selection or {}).get("n_complete_tracks", 0) or 0),
        "window_session_indices": window_indices,
        "window_session_ids": window_ids,
        "ranking_input_metrics_table": str(input_metrics_path.name),
        "ranking_metric_column": metric_column,
        "residual_input_table": residual_input_path.name,
        "residual_join_status": residual_join_status,
        "ranking_baseline_note": first_absent_note,
    }


def _annotate_rank_table(table: pd.DataFrame, provenance: dict[str, object]) -> pd.DataFrame:
    output = table.copy()
    for column, value in provenance.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        output[column] = value
    return output


def _population_title(analysis_dir: Path) -> str:
    run_log = analysis_dir / "run_log.json"
    if run_log.is_file():
        try:
            status = json.loads(run_log.read_text(encoding="utf-8")).get("segmentation_qc_status")
        except (OSError, json.JSONDecodeError):
            status = None
        if status == "not_configured_bypassed":
            return "ROI population summaries (segmentation QC bypassed)"
        if status:
            return "ROI population summaries (segmentation QC applied)"
    return "ROI population summaries"


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


def _resolve_output_dir(analysis_dir: Path, output_dir: str | Path | None, policy: str, project_context=None) -> Path:
    """Resolve a policy-specific output directory for quick plots."""

    base_output_dir = Path(output_dir).resolve() if output_dir else (analysis_dir / "quick_plots")
    if project_context is not None:
        assert project_context.project_config is not None
        base_output_dir = validate_output_path(base_output_dir, project_context.project_config)
        if not is_relative_to(base_output_dir, project_context.analysis_dir):
            raise ValueError("Project quick-plot output must stay under the selected analysis run directory")
    return base_output_dir / policy


def build_quick_plots(
    analysis_dir: str | Path,
    start_date: str | None = None,
    top_n: int = 30,
    policy: str = "high",
    output_dir: str | Path | None = None,
    project_context=None,
) -> Path:
    """Render a small set of no-rerun QC plots from saved matched output tables."""

    run_start_seconds = time.perf_counter()
    analysis_dir = Path(analysis_dir)
    if start_date is None:
        metadata_candidates = [analysis_dir / "run_manifest.json", analysis_dir.parent / "run_manifest.json", analysis_dir / "run_log.json"]
        for metadata_path in metadata_candidates:
            if not metadata_path.is_file():
                continue
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            first_date = payload.get("manifest", {}).get("first_date") or payload.get("start_date")
            if first_date:
                start_date = str(first_date).replace("-", "")
                break
        if start_date is None:
            resolved_manifest = analysis_dir / "session_manifest_resolved.csv"
            if resolved_manifest.is_file():
                manifest_frame = pd.read_csv(resolved_manifest)
                if "acquisition_date" in manifest_frame and not manifest_frame.empty:
                    start_date = str(manifest_frame["acquisition_date"].min()).replace("-", "")
        if start_date is None:
            raise ValueError("No reference-date metadata was found; pass --start-date for this legacy output.")
    log_message(run_start_seconds, f"Starting matched ROI quick plots | analysis_dir={analysis_dir}")
    metrics_path = _first_nonempty_metrics_path(
        analysis_dir,
        "matched_roi_metrics_primary_final.csv",
        "weekly_matched_roi_log_ratio_metrics_complete.csv",
        "matched_roi_log_ratio_metrics_complete.csv",
        "matched_roi_log_ratio_metrics_all_observed.csv",
    )
    fit_population_path = analysis_dir / "matched_session_population_roi_metrics.csv"
    if not fit_population_path.is_file():
        fit_population_path = metrics_path
    residuals_path = _first_nonempty_metrics_path(
        analysis_dir,
        "matched_roi_metrics_primary_final_with_green_red_fit_residuals.csv",
        "weekly_matched_roi_metrics_with_green_red_fit_residuals.csv",
        "matched_roi_metrics_with_green_red_fit_residuals.csv",
        "matched_roi_metrics_with_session_normalized_residuals_all_observed.csv",
    )
    fit_summary_path = _first_existing_path(
        analysis_dir,
        "weekly_matched_daywise_green_red_linear_fit_summary.csv",
        "matched_daywise_green_red_linear_fit_summary.csv",
    )

    log_message(run_start_seconds, f"Loading saved matched tables from {analysis_dir}")
    metrics_table = _filter_table_by_policy(_load_metrics_table(metrics_path), policy)
    fit_population = (
        _load_native_fit_population(fit_population_path)
        if fit_population_path.name == "matched_session_population_roi_metrics.csv"
        else _filter_table_by_policy(_load_metrics_table(fit_population_path), policy)
    )
    residuals_table = _filter_table_by_policy(_load_metrics_table(residuals_path), policy)
    fit_summary = _filter_table_by_policy(_load_fit_summary(fit_summary_path), policy)
    unique_days = int(fit_summary["day"].nunique()) if not fit_summary.empty and "day" in fit_summary.columns else 0
    log_message(
        run_start_seconds,
        f"Policy filter: {policy} | metric_rows={len(metrics_table)} | fit_rows={len(fit_summary)} | unique_days={unique_days}",
    )

    output_dir = _resolve_output_dir(analysis_dir, output_dir, policy, project_context)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_message(run_start_seconds, f"Output directory: {output_dir}")
    log_message(run_start_seconds, "Rendering population and day-wise fit summary plots")

    plot_population_summary(
        roi_metrics=metrics_table,
        output_path=output_dir / "population_longitudinal_summary.png",
        start_date=start_date,
        include_traces=False,
        population_title=_population_title(analysis_dir),
    )
    plot_population_summary(
        roi_metrics=metrics_table,
        output_path=output_dir / "population_longitudinal_summary_with_traces.png",
        start_date=start_date,
        include_traces=True,
        population_title=_population_title(analysis_dir),
    )
    plot_daywise_scatter_summary(
        roi_metrics=fit_population,
        fit_summary=fit_summary,
        output_path=output_dir / "daywise_green_red_linear_fit_scatters.png",
        start_date=start_date,
    )
    plot_fit_parameter_summary(
        fit_summary=fit_summary,
        output_path=output_dir / "daywise_green_red_linear_fit_parameters.png",
        roi_metrics=fit_population,
        start_date=start_date,
    )

    roi_summary = summarize_roi_metrics(metrics_table) if not metrics_table.empty else pd.DataFrame()
    ranking_population = "original_day0"
    ranking_metrics_table = metrics_table
    ranking_residual_table = residuals_table
    ranking_input_path = metrics_path
    ranking_residual_path = residuals_path
    ranking_metric_column = "delta_log2_green_over_red"
    ranking_summary = roi_summary
    window_selection, window_problem = _read_window_selection(analysis_dir)
    residual_join_status = "original_population_residual_table"
    top_level_reason = ""
    if (
        {"roi_id", "session_index", "session_id"}.issubset(metrics_table.columns)
        and {"roi_id", "session_index", "session_id"}.issubset(residuals_table.columns)
    ):
        ranking_residual_table, residual_join_status = _window_residual_table(
            metrics_table, residuals_table,
        )

    if not has_usable_day0_baseline(metrics_table):
        ranking_population = "longest_supported_window"
        ranking_metric_column = "delta_log2_green_over_red_from_window_baseline"
        ranking_input_path = analysis_dir / "matched_roi_metrics_longest_supported_window.csv"
        all_observed_residual_path = analysis_dir / "matched_roi_metrics_with_session_normalized_residuals_all_observed.csv"
        ranking_residual_path = all_observed_residual_path if all_observed_residual_path.is_file() else residuals_path
        ranking_summary = pd.DataFrame()
        ranking_metrics_table = pd.DataFrame()
        ranking_residual_table = pd.DataFrame()
        if window_problem:
            top_level_reason = window_problem
        elif not ranking_input_path.is_file():
            top_level_reason = "selected_window_metrics_table_missing"
        else:
            try:
                ranking_metrics_table = _filter_table_by_policy(
                    _load_metrics_table(ranking_input_path), policy,
                )
                if ranking_metrics_table.empty:
                    top_level_reason = "selected_window_has_no_rows_for_requested_policy"
                else:
                    ranking_summary = summarize_window_metrics_for_ranking(
                        ranking_metrics_table,
                        window_session_indices=window_selection["window_session_indices"],
                    )
                    if ranking_summary.empty:
                        top_level_reason = "selected_window_has_no_eligible_ranked_tracks"
                    if ranking_residual_path.is_file():
                        residual_source = _filter_table_by_policy(
                            _load_metrics_table(ranking_residual_path), policy,
                        )
                        ranking_residual_table, residual_join_status = _window_residual_table(
                            ranking_metrics_table, residual_source,
                        )
                    else:
                        ranking_residual_table = ranking_metrics_table.copy()
                        ranking_residual_table["green_fit_residual"] = float("nan")
                        residual_join_status = "native_residual_table_missing"
            except (ValueError, KeyError, pd.errors.EmptyDataError) as error:
                ranking_metrics_table = pd.DataFrame()
                ranking_residual_table = pd.DataFrame()
                ranking_summary = pd.DataFrame()
                top_level_reason = f"selected_window_inputs_invalid:{type(error).__name__}"

    provenance = _ranking_provenance(
        analysis_dir=analysis_dir,
        population=ranking_population,
        metric_column=ranking_metric_column,
        input_metrics_path=ranking_input_path,
        residual_input_path=ranking_residual_path,
        selection=window_selection,
        metrics=ranking_metrics_table,
        residual_join_status=residual_join_status,
    )
    provenance["ranking_n_candidate_tracks"] = int(len(ranking_summary))
    provenance["ranking_original_day0_baseline_available"] = bool(
        has_usable_day0_baseline(metrics_table)
    )
    provenance["ranking_fallback_reason"] = (
        "original_day0_baseline_unavailable" if ranking_population == "longest_supported_window" else ""
    )
    ranking_columns = [
        "selection_rank",
        "min_delta_log2_green_over_red",
        "max_delta_log2_green_over_red",
        "delta_log2_range",
        "red_cv",
        "day0_brightness",
        "ranking_baseline_red",
        "ranking_baseline_green",
        "ranking_baseline_brightness",
    ]
    selected_counts: dict[str, int] = {}
    direction_reasons: dict[str, str] = {}
    source_label = (
        f"{ranking_population}; baseline session "
        f"{provenance['ranking_baseline_session_index']} "
        f"({provenance['ranking_baseline_session_id']}, "
        f"{provenance['ranking_baseline_acquisition_date']})"
    )
    for direction_label in ("decreasing", "increasing"):
        if ranking_summary.empty:
            top_rois = pd.DataFrame(columns=["selection_rank", "roi_id", "selection_mode"])
            direction_reasons[direction_label] = top_level_reason or "no_eligible_tracks_for_ranking"
        else:
            top_rois = select_top_changing_rois(
                roi_summary=ranking_summary,
                max_rois=top_n,
                direction=direction_label,
                ranking_mode="final",
            )
            if top_rois.empty:
                direction_reasons[direction_label] = (
                    "no_window_cohort_tracks_met_directional_change_and_ranking_qc"
                    if ranking_population == "longest_supported_window"
                    else "no_original_day0_tracks_met_directional_change_and_ranking_qc"
                )
        if top_rois.empty:
            log_message(run_start_seconds, f"No ranked {direction_label} clusters found; skipping directional plots")
            empty_ranked = _annotate_rank_table(top_rois, provenance)
            empty_ranked.to_csv(output_dir / f"top0_{direction_label}_clusters.csv", index=False)
            selected_counts[direction_label] = 0
            continue

        for baseline_name in ("red", "green", "brightness"):
            column = f"ranking_baseline_{baseline_name}"
            if column not in top_rois and f"day0_{baseline_name}" in top_rois:
                top_rois[column] = top_rois[f"day0_{baseline_name}"]
        top_rois = _annotate_rank_table(top_rois, provenance)
        log_message(
            run_start_seconds,
            f"Rendering {direction_label} directional plots for {len(top_rois)} ranked clusters",
        )
        ranked_roi_days = select_ranked_roi_days(
            roi_day_table=ranking_residual_table,
            ranking_table=top_rois,
            top_n=min(int(top_n), len(top_rois)),
            ranking_columns=ranking_columns,
        )
        if ranked_roi_days.empty:
            direction_reasons[direction_label] = "ranked_rois_have_no_source_session_observations"
            pd.DataFrame(columns=["selection_rank", "roi_id", "selection_mode"]).to_csv(
                output_dir / f"top0_{direction_label}_clusters.csv", index=False,
            )
            selected_counts[direction_label] = 0
            continue
        ranked_roi_days = _annotate_rank_table(ranked_roi_days, provenance)
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
            value_column=ranking_metric_column,
            output_path=output_dir / f"top{len(top_rois)}_{direction_label}_delta_log2_heatmap.png",
            start_date=start_date,
            title=(
                f"Top {len(top_rois)} {direction_label} matched clusters: delta log2(G/R)\n"
                f"Ranking baseline: session {provenance['ranking_baseline_session_index']} "
                f"({provenance['ranking_baseline_session_id']}, {provenance['ranking_baseline_acquisition_date']})"
            ),
            colorbar_label="Delta log2(G/R)",
            cmap="coolwarm",
            center_zero=True,
        )
        plot_directional_roi_residuals_vs_day(
            ranked_roi_table=ranked_roi_days,
            output_path=output_dir / f"top{len(top_rois)}_{direction_label}_green_fit_residual_vs_day.png",
            title_prefix=(
                f"Top {len(top_rois)}; ranking relative to {source_label}; "
                "residuals remain native-session fit residuals"
            ),
            direction_label=direction_label,
            start_date=start_date,
        )
        plot_directional_roi_trajectories_scatter(
            ranked_roi_table=ranked_roi_days,
            fit_summary=fit_summary,
            output_path=output_dir / f"top{len(top_rois)}_{direction_label}_green_red_trajectory_scatter_average_fit.png",
            title_prefix=f"Top {len(top_rois)}; ranking relative to {source_label}",
            direction_label=direction_label,
            start_date=start_date,
        )
        selected_counts[direction_label] = int(len(top_rois))

    ranking_status = {
        "status": "evaluated" if any(selected_counts.values()) else "no_eligible_data",
        "reason": "" if any(selected_counts.values()) else (top_level_reason or "no_directional_candidates_met_ranking_criteria"),
        "directions": {},
        **provenance,
        "ranking_input_metrics_path": str(ranking_input_path),
        "residual_input_path": str(ranking_residual_path),
        "ranking_delta_semantics": (
            "Window-relative to the first actual source session in the selected complete cohort."
            if ranking_population == "longest_supported_window"
            else "Original source Day-0-relative log2(G/R)."
        ),
        "residual_semantics": "Signed residuals remain observed green minus that source session's native fitted green; they are not shifted to the ranking baseline.",
    }
    for direction_label in ("decreasing", "increasing"):
        selected_count = int(selected_counts.get(direction_label, 0))
        ranking_status["directions"][direction_label] = {
            "status": "evaluated" if selected_count else "no_eligible_data",
            "n_ranked_rois": selected_count,
            "reason": direction_reasons.get(direction_label, "") if not selected_count else "",
        }
    (output_dir / "ranked_roi_status.json").write_text(
        json.dumps(ranking_status, indent=2, sort_keys=True), encoding="utf-8",
    )

    summary_lines = [
        "Quick plots generated from saved matched ROI outputs.",
        f"Analysis directory: {analysis_dir}",
        f"Start date: {start_date}",
        f"Top-N directional plots: {top_n}",
        f"Policy filter: {policy}",
        f"Ranking population: {ranking_population}",
        f"Ranking baseline: session {provenance['ranking_baseline_session_index']} "
        f"({provenance['ranking_baseline_session_id']}, {provenance['ranking_baseline_acquisition_date']})",
        f"Ranking metric: {ranking_metric_column}",
        f"Ranking input table: {ranking_input_path.name}",
        "Residual plots retain native-session fit residual definitions; no ranking-baseline subtraction is applied.",
        str(provenance["ranking_baseline_note"]),
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
    add_project_selector(parser)
    parser.add_argument("--analysis-dir", required=True, help="Completed matched ROI analysis output directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional root directory for the quick-plot exports; policy-specific subfolders are created beneath it.",
    )
    parser.add_argument("--start-date", default=None, help="Reference date in YYYYMMDD format.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top increasing/decreasing clusters to plot.")
    parser.add_argument(
        "--policy",
        default="high",
        choices=["high", "balanced", "graph", "all"],
        help="When match_policy is present, choose which policy to plot. Use 'all' to keep both.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Render quick QC plots from an existing analysis output directory."""

    args = parse_args()
    _context, analysis_dir = resolve_analysis_selection(analysis_dir=args.analysis_dir,dataset=args.dataset,project_config=args.project_config,mouse_id=args.mouse_id,laser_nm=args.laser_nm)
    output_dir = build_quick_plots(
        analysis_dir=analysis_dir,
        start_date=args.start_date,
        top_n=args.top_n,
        policy=args.policy,
        output_dir=args.output_dir,
        project_context=_context,
    )
    print(f"quick_plot_dir={output_dir}")


if __name__ == "__main__":
    main()
