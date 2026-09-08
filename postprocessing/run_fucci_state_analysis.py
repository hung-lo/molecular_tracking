#!/usr/bin/env python3
"""Build color-Z trajectories, PCA, and hysteretic state-entry summaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, _ROOT / "core", _ROOT / "postprocessing", _ROOT / "plotting"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fucci_pca import compute_color_z_pca, pca_output_tables
from fucci_run_io import file_sha256, resolve_fucci_master_run
from fucci_state_events import build_event_aligned_observations, detect_middle_entry_events, summarize_event_aligned_observations
from plotting.fucci_color_state_plots import plot_event_summary, plot_pca, plot_pca_loadings
from trajectory_eligibility import TrajectoryEligibilityConfig, build_trajectory_eligibility, build_trajectory_matrices


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _reset_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Expected directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _window(summary: pd.DataFrame, axis: str, window_days: float) -> pd.DataFrame:
    if summary.empty:
        return summary
    if axis == "session_index":
        return summary.loc[summary["relative_session_index"].abs() <= window_days]
    return summary.loc[summary["relative_elapsed_days"].abs() <= window_days]


def run_state_analysis(
    color_state_dir: str | Path,
    *,
    event_min_sessions: int = 8,
    event_axis: str = "elapsed_days",
    event_window_days: float = 7,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate all downstream color-Z products under an existing color-state run."""

    color_dir = Path(color_state_dir).expanduser().resolve()
    if not color_dir.is_dir():
        raise FileNotFoundError(f"Color-state directory was not found: {color_dir}")
    if event_axis not in {"elapsed_days", "session_index"}:
        raise ValueError("event_axis must be elapsed_days or session_index")
    log_path = color_dir / "run_manifest.json"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing color-state run manifest: {log_path}")
    log = json.loads(log_path.read_text(encoding="utf-8"))
    source_dir = Path(log["source_master_run_dir"]).expanduser().resolve()
    source = resolve_fucci_master_run(source_dir, require_matched=True)
    scored_path = color_dir / "normalization/matched_roi_color_state_all_observed.csv"
    fit_path = color_dir / "normalization/color_state_session_fits.csv"
    if not scored_path.is_file() or not fit_path.is_file():
        raise FileNotFoundError("Color-state normalization outputs are incomplete")
    scored = pd.read_csv(scored_path)
    scored["ratio_qc_pass"] = _bool_series(scored["ratio_qc_pass"])
    if "color_state_qc_pass" in scored:
        scored["color_state_qc_pass"] = _bool_series(scored["color_state_qc_pass"])
    if "geometry_qc_pass" in scored:
        if not scored["geometry_qc_pass"].isna().all():
            scored["geometry_qc_pass"] = _bool_series(scored["geometry_qc_pass"])
    fits = pd.read_csv(fit_path)
    fits["session_id"] = fits["session_id"].astype(str)
    session_ids = fits.sort_values("session_index")["session_id"].tolist()
    tracks = pd.read_csv(source.track_summary_path)  # type: ignore[arg-type]
    geometry = pd.read_csv(source.geometry_path)  # type: ignore[arg-type]
    if "geometry_qc_pass" not in scored:
        keys = [key for key in ("match_policy", "roi_id", "session_index") if key in scored and key in geometry]
        if len(keys) >= 2:
            scored = scored.merge(geometry[keys + ["geometry_qc_pass"]].drop_duplicates(keys), on=keys, how="left", validate="many_to_one")
            if not scored["geometry_qc_pass"].isna().all():
                scored["geometry_qc_pass"] = _bool_series(scored["geometry_qc_pass"])

    trajectory_dir = color_dir / "trajectory"
    pca_dir = color_dir / "pca"
    events_dir = color_dir / "events"
    if not overwrite and any(path.exists() for path in (trajectory_dir, pca_dir, events_dir)):
        raise FileExistsError("State-analysis output already exists; use --overwrite")
    for path in (trajectory_dir, pca_dir, events_dir):
        _reset_dir(path)

    source_trajectory = source.extraction_run_log.get("trajectory_eligibility", {})
    eligibility_config = TrajectoryEligibilityConfig(
        min_sessions=int(source_trajectory.get("min_sessions", 2)),
        min_session_fraction=source_trajectory.get("min_session_fraction"),
        max_internal_missing_sessions=source_trajectory.get("max_internal_missing_sessions"),
        require_first_session=bool(source_trajectory.get("require_first_session", False)),
        require_last_session=bool(source_trajectory.get("require_last_session", False)),
    )
    eligibility, eligible = build_trajectory_eligibility(scored, tracks, session_ids, eligibility_config, feature_column="color_z")
    matrices = build_trajectory_matrices(eligible, session_ids, feature_column="color_z")
    matrix_names = {
        "raw": "color_z_trajectory_matrix.csv",
        "mask": "color_z_trajectory_observation_mask.csv",
        "missingness": "color_z_trajectory_missingness_by_session.csv",
        "centered": "color_z_trajectory_matrix_centered_with_nan.csv",
        "complete_raw": "color_z_pca_complete_case_raw.csv",
        "complete_centered": "color_z_pca_complete_case_centered.csv",
        "complete_centering": "color_z_pca_complete_case_centering_summary.csv",
    }
    eligibility.to_csv(trajectory_dir / "color_state_trajectory_eligibility.csv", index=False)
    eligible.to_csv(trajectory_dir / "color_state_trajectory_observations_eligible.csv", index=False)
    for key, filename in matrix_names.items():
        matrices[key].to_csv(trajectory_dir / filename, index=False)

    complete_raw = matrices["complete_raw"]
    pca_result = compute_color_z_pca(complete_raw, session_ids)
    pca_tables = pca_output_tables(pca_result, fits)
    for key, filename in (("scores", "color_z_pca_scores.csv"), ("loadings", "color_z_pca_loadings.csv"), ("explained_variance", "color_z_pca_explained_variance.csv"), ("centering_summary", "color_z_pca_complete_case_centering_summary.csv")):
        pca_tables[key].to_csv(pca_dir / filename, index=False)
    (pca_dir / "color_z_pca_run_log.json").write_text(json.dumps({
        "schema_version": "fucci_color_z_pca_v1",
        "reference_json_sha256": log.get("reference_json_sha256"),
        "reference_robust_sd_log2": log.get("reference_robust_sd_log2"),
        "feature_column": "color_z",
        "session_ids": session_ids,
        "centering": "column_mean_complete_case_population",
        "scaling": "none",
        "sign_convention": "largest_absolute_loading_positive",
        "n_complete_cases": int(len(complete_raw)),
        "source_color_state_run_manifest_sha256": file_sha256(log_path),
    }, indent=2, sort_keys=True), encoding="utf-8")
    plot_pca(pca_tables["scores"], color_dir / "plots/pca_pc1_pc2.png")
    plot_pca_loadings(pca_tables["loadings"], color_dir / "plots/pca_loadings_vs_elapsed_days.png")

    events = detect_middle_entry_events(scored, min_usable_sessions=event_min_sessions)
    aligned = build_event_aligned_observations(scored, events)
    events.to_csv(events_dir / "color_state_entry_events.csv", index=False)
    aligned.to_csv(events_dir / "color_state_event_aligned_observations.csv", index=False)
    event_summaries: dict[str, pd.DataFrame] = {}
    for event_type, filename, title in (
        ("middle_to_low", "middle_to_low_event_summary.csv", "Middle-to-low color-Z entries"),
        ("middle_to_high", "middle_to_high_event_summary.csv", "Middle-to-high color-Z entries"),
    ):
        subset = events.loc[events["event_type"].eq(event_type)]
        subset_aligned = aligned.loc[aligned["event_id"].isin(subset["event_id"])]
        summary = summarize_event_aligned_observations(subset_aligned, first_events_only=True, events=subset)
        event_summaries[event_type] = summary
        summary.to_csv(events_dir / filename, index=False)
        plot_event_summary(_window(summary, event_axis, event_window_days), color_dir / f"plots/{event_type}_event_triggered_color_z.png", title=title, axis_label=event_axis)

    analysis_log = {
        "schema_version": "fucci_state_analysis_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_color_state_run_manifest_sha256": file_sha256(log_path),
        "reference_json_sha256": log.get("reference_json_sha256"),
        "reference_robust_sd_log2": log.get("reference_robust_sd_log2"),
        "source_master_run_dir": str(source.run_dir),
        "feature_column": "color_z",
        "trajectory_eligibility": {"min_sessions": eligibility_config.min_sessions, "missing_values": "preserved_as_nan", "geometry_qc_merged": True},
        "pca": {"centering": "complete_case_column_mean", "scaling": "none", "sign_convention": "largest_absolute_loading_positive"},
        "events": {"min_usable_sessions": event_min_sessions, "axis": event_axis, "window_days_for_plot": event_window_days, "all_observations_retained_in_event_table": True},
        "row_counts": {"trajectory_eligible": int(len(eligible)), "complete_cases": int(len(complete_raw)), "entry_events": int(len(events)), "event_aligned_observations": int(len(aligned))},
    }
    (events_dir / "run_log.json").write_text(json.dumps(analysis_log, indent=2, sort_keys=True), encoding="utf-8")
    (color_dir / "state_analysis_run_log.json").write_text(json.dumps(analysis_log, indent=2, sort_keys=True), encoding="utf-8")
    return analysis_log


def parse_args(argv: list[str] | None = None) -> Any:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-state-dir", required=True)
    parser.add_argument("--event-min-sessions", type=int, default=8)
    parser.add_argument("--event-axis", choices=("elapsed_days", "session_index"), default="elapsed_days")
    parser.add_argument("--event-window-days", type=float, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_state_analysis(args.color_state_dir, event_min_sessions=args.event_min_sessions, event_axis=args.event_axis, event_window_days=args.event_window_days, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
