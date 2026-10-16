"""Render top/bottom final-session directional ROIs in native raw space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for _directory in (_ROOT / "core", _ROOT / "plotting"):
    if str(_directory) not in sys.path:
        sys.path.append(str(_directory))

from roi_log_ratio_analysis import (
    decide_ranking_population,
    has_usable_day0_baseline,
    select_top_changing_rois,
)
from run_matched_roi_quick_view import _tracks_from_raw_table, render_ranked_roi_batch


METRIC_NAMES = (
    "matched_roi_metrics_primary_final.csv",
    "weekly_matched_roi_log_ratio_metrics_complete.csv",
    "matched_roi_log_ratio_metrics_complete.csv",
    "matched_roi_log_ratio_metrics_all_observed.csv",
)
RAW_NAMES = (
    "matched_roi_intensity_results_raw.csv",
    "weekly_matched_roi_intensity_results_raw.csv",
)


def _first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of {names} was found in {root}")


def _first_nonempty_metrics(root: Path, names: tuple[str, ...]) -> Path:
    existing = [root / name for name in names if (root / name).is_file()]
    if not existing:
        raise FileNotFoundError(f"None of {names} was found in {root}")
    for candidate in existing:
        try:
            if not pd.read_csv(candidate, nrows=1).empty:
                return candidate
        except pd.errors.EmptyDataError:
            continue
    return existing[0]


def _resolve_run_inputs(run_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve current master-run inputs, with support for legacy flat runs."""
    extraction = run_dir / "extraction"
    if extraction.is_dir():
        return (
            _first_nonempty_metrics(extraction, METRIC_NAMES),
            _first_existing(extraction, RAW_NAMES),
            _first_existing(extraction, ("session_manifest_resolved.csv",)) if (extraction / "session_manifest_resolved.csv").is_file() else _first_existing(run_dir / "matching", ("session_manifest_resolved.csv",)),
        )
    return (
        _first_nonempty_metrics(run_dir, METRIC_NAMES),
        _first_existing(run_dir, RAW_NAMES),
        _first_existing(run_dir, ("session_manifest_resolved.csv",)),
    )


def _resolve_feature_table(run_dir: Path) -> Path | None:
    extraction = run_dir / "extraction"
    for candidate in (
        extraction / "roi_features.csv",
        run_dir / "matching" / "roi_features.csv",
        run_dir / "roi_features.csv",
    ):
        if candidate.is_file():
            return candidate
    return None


def _filter_policy(table: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "all" or "match_policy" not in table.columns:
        return table
    filtered = table.loc[table["match_policy"].astype(str).eq(policy)].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for match policy {policy!r}")
    return filtered


def build_ranked_roi_views(
    run_dir: str | Path,
    *,
    policy: str = "graph",
    top_n: int = 30,
    directions: tuple[str, ...] = ("increasing", "decreasing"),
    render_z_radius: int = 0,
    z_radius: int | None = None,
    output_dir: str | Path | None = None,
    ranking_population: str = "auto",
) -> Path:
    """Rank current metrics and render each selected track with the existing viewer."""
    root = Path(run_dir).resolve()
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if z_radius is not None:
        render_z_radius = z_radius
    if render_z_radius < 0:
        raise ValueError("render_z_radius must be >= 0")
    metrics_path, raw_path, manifest_path = _resolve_run_inputs(root)
    analysis_root = metrics_path.parent
    metrics = _filter_policy(pd.read_csv(metrics_path), policy)
    raw = _filter_policy(pd.read_csv(raw_path), policy)
    manifest = pd.read_csv(manifest_path)
    feature_path = _resolve_feature_table(root)
    features = pd.read_csv(feature_path) if feature_path is not None else None
    output_root = Path(output_dir).resolve() if output_dir else root / "plots" / policy / "single_roi_raw_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    ranking_input_path = metrics_path
    render_manifest = manifest
    selection: dict[str, object] = {}
    selection_path = analysis_root / "longest_supported_window.json"
    window_metrics_path = analysis_root / "matched_roi_metrics_longest_supported_window.csv"
    window_tracks_path = analysis_root / "matched_roi_tracks_longest_supported_window.csv"
    window_problem = "longest_supported_window_metadata_missing"
    if selection_path.is_file():
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            window_problem = "no_supported_window" if selection.get("status") != "valid" else ""
        except (OSError, json.JSONDecodeError):
            selection = {}
            window_problem = "longest_supported_window_metadata_invalid"
    saved_window_metrics = None
    saved_window_tracks = None
    if selection.get("status") == "valid":
        try:
            if window_metrics_path.is_file():
                saved_window_metrics = _filter_policy(pd.read_csv(window_metrics_path), policy)
            else:
                window_problem = "selected_window_metrics_table_missing"
            if window_tracks_path.is_file():
                saved_window_tracks = _filter_policy(pd.read_csv(window_tracks_path), policy)
            else:
                window_problem = "selected_window_tracks_table_missing"
        except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as error:
            saved_window_metrics = None
            saved_window_tracks = None
            window_problem = f"selected_window_inputs_invalid:{type(error).__name__}"

    ranking_decision = decide_ranking_population(
        metrics,
        window_selection=selection or None,
        window_metrics=saved_window_metrics,
        window_tracks=saved_window_tracks,
        window_unavailable_reason=window_problem or None,
        requested_population=ranking_population,
    )
    ranking_population = ranking_decision.population
    ranking_metric_column = (
        "delta_log2_green_over_red_from_window_baseline"
        if ranking_population == "longest_supported_window"
        else "delta_log2_green_over_red"
    )
    ranking_input_path = window_metrics_path if ranking_population == "longest_supported_window" else metrics_path
    summary = (
        ranking_decision.window_summary
        if ranking_population == "longest_supported_window"
        else ranking_decision.original_summary
    )
    tracks = _tracks_from_raw_table(raw, policy) if not raw.empty and policy != "all" else pd.DataFrame()
    fallback_reason = (
        ranking_decision.window_status_reason
        if (
            summary.empty
            and ranking_decision.original_day0_signal_valid_count == 0
            and ranking_decision.window_status_reason == "no_supported_window"
        )
        else ranking_decision.decision_reason
    )
    if ranking_population == "longest_supported_window" and saved_window_tracks is not None:
        tracks = saved_window_tracks.copy()
        selected_indices = {int(value) for value in selection.get("window_session_indices", [])}
        if "session_index" not in manifest:
            fallback_reason = "source_manifest_missing_session_indices"
        else:
            manifest_indices = pd.to_numeric(manifest["session_index"], errors="coerce")
            render_manifest = manifest.loc[manifest_indices.isin(selected_indices)].copy()
            if set(pd.to_numeric(render_manifest["session_index"], errors="coerce").dropna().astype(int)) != selected_indices:
                fallback_reason = "selected_window_sessions_missing_from_source_manifest"
        if not tracks.empty and "track_uid" in tracks:
            tracks = tracks.loc[tracks["track_uid"].astype(str).isin(set(summary.get("track_uid", pd.Series(dtype=str)).astype(str)))].copy()

    window_indices = [int(value) for value in selection.get("window_session_indices", [])]
    window_ids = [str(value) for value in selection.get("window_session_ids", [])]
    window_dates = [str(value) for value in selection.get("window_acquisition_dates", [])]
    if ranking_population == "longest_supported_window" and window_indices:
        baseline_index = window_indices[0]
        baseline_id = window_ids[0] if window_ids else ""
        baseline_date = window_dates[0] if window_dates else ""
    elif ranking_population == "longest_supported_window":
        baseline_index, baseline_id, baseline_date = None, "", ""
    else:
        baseline_index, baseline_id, baseline_date = 0, "", ""
        if not manifest.empty and "session_index" in manifest:
            baseline_rows = manifest.loc[pd.to_numeric(manifest["session_index"], errors="coerce").eq(0)]
            if not baseline_rows.empty:
                baseline_id = str(baseline_rows.iloc[0].get("session_id", ""))
                baseline_date = str(baseline_rows.iloc[0].get("acquisition_date", ""))
    first_note = ""
    if ranking_population == "longest_supported_window" and not manifest.empty and window_indices:
        first_row = manifest.sort_values("session_index").iloc[0]
        if int(first_row["session_index"]) not in window_indices:
            first_note = (
                f"Source session {int(first_row['session_index'])} ({first_row.get('session_id', '')}, "
                f"{first_row.get('acquisition_date', '')}) remains in the manifest but is absent from the selected-window cohort."
            )
    provenance = {
        "ranking_population": ranking_population,
        "ranking_baseline_session_index": baseline_index,
        "ranking_baseline_session_id": baseline_id,
        "ranking_baseline_acquisition_date": baseline_date,
        "ranking_baseline_source_session_index": baseline_index,
        "ranking_baseline_source_session_id": baseline_id,
        "ranking_baseline_source_acquisition_date": baseline_date,
        "window_status": selection.get("status", "unavailable"),
        "window_start_session_index": window_indices[0] if window_indices else None,
        "window_end_session_index": window_indices[-1] if window_indices else None,
        "window_start_session_id": window_ids[0] if window_ids else "",
        "window_end_session_id": window_ids[-1] if window_ids else "",
        "window_start_acquisition_date": window_dates[0] if window_dates else "",
        "window_end_acquisition_date": window_dates[-1] if window_dates else "",
        "window_n_source_sessions": len(window_indices),
        "window_n_candidate_tracks": int(selection.get("n_complete_tracks", 0) or 0),
        "window_session_indices": window_indices,
        "window_session_ids": window_ids,
        "ranking_input_metrics_table": ranking_input_path.name,
        "ranking_metric_column": ranking_metric_column,
        "ranking_delta_semantics": (
            "Window-relative to the first actual source session in the selected complete cohort."
            if ranking_population == "longest_supported_window"
            else "Original source Day-0-relative log2(G/R)."
        ),
        "ranking_n_candidate_tracks": int(len(summary)),
        "ranking_original_day0_baseline_available": bool(has_usable_day0_baseline(metrics)),
        "ranking_decision_reason": ranking_decision.decision_reason,
        "ranking_original_day0_signal_valid_count": ranking_decision.original_day0_signal_valid_count,
        "ranking_original_day0_rankable_track_count": ranking_decision.original_day0_rankable_track_count,
        "ranking_window_rankable_track_count": ranking_decision.window_rankable_track_count,
        "ranking_selected_rankable_track_count": ranking_decision.selected_rankable_track_count,
        "ranking_low_n_warning": ranking_decision.low_n_warning,
        "ranking_low_n_threshold": ranking_decision.low_n_threshold,
        "ranking_window_status_reason": ranking_decision.window_status_reason,
        "ranking_fallback_reason": ranking_decision.decision_reason if ranking_population == "longest_supported_window" else "",
        "ranking_baseline_note": first_note,
        "ranking_status_reason": fallback_reason,
    }

    if summary.empty or (ranking_population == "longest_supported_window" and tracks.empty):
        reason = fallback_reason or (
            "no_valid_day0_baseline_for_day0_change_ranking"
            if ranking_population == "original_day0"
            else "selected_window_has_no_eligible_ranked_tracks"
        )
        empty_index = pd.DataFrame(columns=[
            "selection_direction", "selection_rank", "roi_id", "cluster_id", "track_uid",
            "selection_mode", "selection_metric_column", "selection_value",
            "ranking_baseline_red", "ranking_baseline_green", "ranking_baseline_brightness",
            "png_path", "metadata_csv_path",
        ])
        for key, value in provenance.items():
            empty_index[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        empty_index.to_csv(output_root / "ranked_roi_batch_index.csv", index=False)
        profile = {"status": "no_eligible_data", "reason": reason, **provenance}
        (output_root / "ranked_roi_render_profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8",
        )
        (output_root / "README.txt").write_text(
            f"Ranked raw-space views were not generated.\nRanking population: {ranking_population}\n"
            f"Reason: {reason}\nRanking baseline: session {baseline_index} ({baseline_id}, {baseline_date})\n"
            f"Ranking decision: {ranking_decision.decision_reason}\n"
            f"Source session indices spanned: {window_indices}\n"
            f"Ranking input: {ranking_input_path.name}; metric: {ranking_metric_column}\n"
            f"Delta semantics: {provenance['ranking_delta_semantics']}\n"
            f"Original Day-0 signal-valid tracks: {ranking_decision.original_day0_signal_valid_count}\n"
            f"Original Day-0 rankable tracks: {ranking_decision.original_day0_rankable_track_count}\n"
            f"Window rankable tracks: {ranking_decision.window_rankable_track_count}\n"
            f"Selected rankable tracks: {ranking_decision.selected_rankable_track_count}\n"
            f"Low-N warning (<{ranking_decision.low_n_threshold}): {ranking_decision.low_n_warning}\n"
            f"{first_note}\n", encoding="utf-8",
        )
        return output_root

    index_rows = []
    render_specs = []
    for direction in directions:
        if direction not in {"increasing", "decreasing"}:
            raise ValueError(f"Unsupported direction: {direction!r}")
        ranked = select_top_changing_rois(summary, max_rois=top_n, direction=direction, ranking_mode="final")
        direction_dir = output_root / f"top{top_n}_{direction}"
        direction_dir.mkdir(parents=True, exist_ok=True)
        for _, selected in ranked.iterrows():
            roi_id = selected.get("roi_id", selected.get("cluster_id", ""))
            cluster_id = selected.get("cluster_id", roi_id)
            rank = int(selected["selection_rank"])
            stem = f"{'increase' if direction == 'increasing' else 'decrease'}_rank{rank:02d}_roi_{roi_id}"
            png_path = direction_dir / f"{stem}.png"
            track_uid = selected.get("track_uid", "")
            if pd.isna(track_uid) or not str(track_uid):
                matches = tracks.loc[tracks["cluster_id"].astype(str).eq(str(cluster_id))]
                if len(matches) != 1:
                    raise ValueError(f"Could not resolve a unique track for ranked ROI {cluster_id!r}")
                track_uid = matches.iloc[0]["track_uid"]
            if ranking_population == "longest_supported_window":
                track_matches = tracks.loc[tracks["track_uid"].astype(str).eq(str(track_uid))]
                if len(track_matches) != 1:
                    raise ValueError(f"Selected window does not contain a unique track {track_uid!r}")
            render_specs.append((str(track_uid), png_path))
            metadata_path = png_path.with_name(png_path.stem + "_metadata.csv")
            index_rows.append({
                "selection_direction": direction, "selection_rank": rank,
                "roi_id": roi_id, "cluster_id": cluster_id,
                "track_uid": str(track_uid),
                "selection_mode": selected["selection_mode"],
                "selection_metric_column": selected["selection_metric_column"],
                "selection_value": selected[selected["selection_metric_column"]],
                "ranking_baseline_red": selected.get("ranking_baseline_red", selected.get("day0_red", float("nan"))),
                "ranking_baseline_green": selected.get("ranking_baseline_green", selected.get("day0_green", float("nan"))),
                "ranking_baseline_brightness": selected.get("ranking_baseline_brightness", selected.get("day0_brightness", float("nan"))),
                "png_path": str(png_path), "metadata_csv_path": str(metadata_path),
                **provenance,
            })
    if not render_specs:
        reason = fallback_reason or (
            "no_valid_day0_baseline_for_day0_change_ranking"
            if ranking_population == "original_day0" and "day0_brightness" in summary and summary["day0_brightness"].isna().all()
            else "no_rois_met_ranking_criteria"
        )
        empty_columns = [
            "selection_direction", "selection_rank", "roi_id", "cluster_id", "track_uid",
            "selection_mode", "selection_metric_column", "selection_value",
            "ranking_baseline_red", "ranking_baseline_green", "ranking_baseline_brightness",
            "png_path", "metadata_csv_path",
            *provenance.keys(),
        ]
        pd.DataFrame(index_rows, columns=empty_columns).to_csv(output_root / "ranked_roi_batch_index.csv", index=False)
        (output_root / "ranked_roi_render_profile.json").write_text(
            json.dumps({"status": "no_eligible_data", "reason": reason, **provenance}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_root / "README.txt").write_text(
            f"Ranked raw-space views were not generated.\nRanking population: {ranking_population}\n"
            f"Ranking baseline: session {baseline_index} ({baseline_id}, {baseline_date})\n"
            f"Ranking decision: {ranking_decision.decision_reason}\n"
            f"Source session indices spanned: {window_indices}\n"
            f"Ranking input: {ranking_input_path.name}; metric: {ranking_metric_column}\n"
            f"Delta semantics: {provenance['ranking_delta_semantics']}\n"
            f"Original Day-0 rankable tracks: {ranking_decision.original_day0_rankable_track_count}\n"
            f"Selected rankable tracks: {ranking_decision.selected_rankable_track_count}\n{first_note}\n",
            encoding="utf-8",
        )
        return output_root
    if ranking_population == "longest_supported_window":
        selected_indices = set(window_indices)
        manifest_indices = pd.to_numeric(render_manifest["session_index"], errors="coerce")
        render_manifest = render_manifest.loc[manifest_indices.isin(selected_indices)].copy()
    profile = render_ranked_roi_batch(
        specs=render_specs,
        tracks_table=tracks,
        session_table=render_manifest,
        feature_table=features,
        z_radius=render_z_radius,
    )
    pd.DataFrame(index_rows).to_csv(output_root / "ranked_roi_batch_index.csv", index=False)
    profile.update(provenance)
    (output_root / "ranked_roi_render_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "README.txt").write_text(
        "Ranked raw-space views rendered from saved extraction outputs.\n"
        f"Ranking population: {ranking_population}\n"
        f"Ranking baseline: session {baseline_index} ({baseline_id}, {baseline_date})\n"
        f"Source session indices spanned: {window_indices}\n"
        f"Ranking metric: {ranking_metric_column}\n"
        f"Ranking input: {ranking_input_path.name}\n"
        f"Delta semantics: {provenance['ranking_delta_semantics']}\n"
        f"Ranking decision: {ranking_decision.decision_reason}\n"
        f"Original Day-0 signal-valid tracks: {ranking_decision.original_day0_signal_valid_count}\n"
        f"Original Day-0 rankable tracks: {ranking_decision.original_day0_rankable_track_count}\n"
        f"Window rankable tracks: {ranking_decision.window_rankable_track_count}\n"
        f"Selected rankable tracks: {ranking_decision.selected_rankable_track_count}\n"
        f"Low-N warning (<{ranking_decision.low_n_threshold}): {ranking_decision.low_n_warning}\n{first_note}\n",
        encoding="utf-8",
    )
    return output_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--policy", choices=["high", "balanced", "graph"], default="graph")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--directions", default="increasing,decreasing")
    parser.add_argument("--render-z-radius", "--z-radius", dest="render_z_radius", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--ranking-population",
        choices=["auto", "original_day0", "longest_supported_window"],
        default="auto",
        help="Choose the ranking display population; auto preserves Day 0 when at least two tracks are rankable.",
    )
    args = parser.parse_args(argv)
    output = build_ranked_roi_views(
        args.run_dir, policy=args.policy, top_n=args.top_n,
        directions=tuple(part.strip() for part in args.directions.split(",") if part.strip()),
        render_z_radius=args.render_z_radius, output_dir=args.output_dir,
        ranking_population=args.ranking_population,
    )
    print(f"ranked_roi_output_dir={output}")


if __name__ == "__main__":
    main()
