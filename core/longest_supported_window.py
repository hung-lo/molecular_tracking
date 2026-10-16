"""Select an additive, data-supported contiguous matched-track window."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_longest_supported_window(
    metrics: pd.DataFrame,
    native_population: pd.DataFrame,
    records,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Rank contiguous source-session spans and return the winning cohort.

    Cohort membership requires one measured, dual-channel signal-valid,
    native-fit-eligible observation for the track on every session in a span.
    A candidate also needs at least two tracks and a nonconstant red predictor
    on every day. No observations are rematched or imputed.
    """
    required_records = [record for record in records if record.required]
    columns = [
        "window_start_session_index", "window_end_session_index", "window_length",
        "window_start_session_id", "window_end_session_id", "n_complete_tracks",
        "status", "reason", "day_fit_statuses",
    ]
    if not required_records:
        raise ValueError("At least one required manifest session is needed to select a window.")
    needed_metrics = {"track_uid", "session_id", "session_index", "session_roi_label", "red", "green", "ratio_qc_pass", "log2_green_over_red"}
    missing_metrics = needed_metrics.difference(metrics.columns)
    if missing_metrics:
        raise ValueError(f"Missing required metric columns: {', '.join(sorted(missing_metrics))}")
    needed_native = {"session_id", "session_roi_label", "fit_population_include"}
    missing_native = needed_native.difference(native_population.columns)
    if missing_native:
        raise ValueError(f"Missing native fit-population columns: {', '.join(sorted(missing_native))}")
    if native_population.duplicated(["session_id", "session_roi_label"]).any():
        raise ValueError("Native population has duplicate session/ROI labels.")

    observed = metrics.copy()
    if observed.duplicated(["track_uid", "session_index"]).any():
        raise ValueError("Matched metrics contain duplicate track/session observations.")
    observed["session_id"] = observed["session_id"].astype(str)
    native = native_population[["session_id", "session_roi_label", "fit_population_include"]].copy()
    native["session_id"] = native["session_id"].astype(str)
    observed = observed.merge(
        native,
        on=["session_id", "session_roi_label"],
        how="left",
        validate="many_to_one",
    )
    observed["window_observation_eligible"] = (
        observed["ratio_qc_pass"].eq(True)
        & observed["fit_population_include"].eq(True)
        & np.isfinite(pd.to_numeric(observed["red"], errors="coerce"))
        & np.isfinite(pd.to_numeric(observed["green"], errors="coerce"))
    )
    day_order = [int(record.session_index) for record in required_records]
    record_by_day = {int(record.session_index): record for record in required_records}
    ranking_rows: list[dict[str, object]] = []
    accepted: list[tuple[int, int, int, set[str], list[int]]] = []
    for start_pos in range(len(day_order)):
        for end_pos in range(start_pos + 2, len(day_order) + 1):
            window_days = day_order[start_pos:end_pos]
            candidate = observed.loc[
                observed["session_index"].isin(window_days)
                & observed["window_observation_eligible"]
            ]
            counts = candidate.groupby("track_uid")["session_index"].nunique()
            complete_ids = set(counts.index[counts.eq(len(window_days))].astype(str))
            reasons: list[str] = []
            day_fit_statuses: list[dict[str, object]] = []
            if len(complete_ids) < 2:
                reasons.append("fewer_than_two_complete_fit_eligible_tracks")
            for day in window_days:
                day_values = candidate.loc[
                    candidate["track_uid"].astype(str).isin(complete_ids)
                    & candidate["session_index"].eq(day)
                ]
                day_reason = ""
                if len(day_values) < 2:
                    day_reason = "fewer_than_two_complete_fit_eligible_tracks"
                elif np.ptp(day_values["red"].astype(float).to_numpy()) == 0:
                    day_reason = "constant_red_predictor"
                day_fit_statuses.append({"session_index": int(day), "status": "valid" if not day_reason else "not_estimable", "reason": day_reason})
                if day_reason and day_reason not in reasons:
                    reasons.append(day_reason)
            status = "valid" if not reasons else "rejected"
            start_record, end_record = record_by_day[window_days[0]], record_by_day[window_days[-1]]
            ranking_rows.append({
                "window_start_session_index": int(window_days[0]),
                "window_end_session_index": int(window_days[-1]),
                "window_length": len(window_days),
                "window_start_session_id": str(start_record.session_id),
                "window_end_session_id": str(end_record.session_id),
                "n_complete_tracks": len(complete_ids),
                "status": status,
                "reason": ";".join(reasons),
                "day_fit_statuses": day_fit_statuses,
            })
            if status == "valid":
                accepted.append((len(window_days), len(complete_ids), start_pos, complete_ids, window_days))

    ranking = pd.DataFrame(ranking_rows, columns=columns)
    ranking = ranking.sort_values(
        ["window_length", "n_complete_tracks", "window_start_session_index"],
        ascending=[False, False, True],
    ).reset_index(drop=True) if not ranking.empty else ranking
    if not accepted:
        result = {
            "status": "no_supported_window",
            "selection_rule": "Longest contiguous source-order span with at least two complete tracks; each track must have genuine dual-channel signal-valid, native-fit-eligible observations on every session; every day's red predictor must be nonconstant. Tie-break by more complete tracks, then earlier source session.",
            "window_session_indices": [], "window_session_ids": [], "window_acquisition_dates": [],
            "window_length": 0, "n_complete_tracks": 0,
            "rejected_candidates": ranking.to_dict("records"),
        }
        return result, ranking, observed.iloc[0:0].copy()

    accepted.sort(key=lambda item: (-item[0], -item[1], item[2]))
    length, n_tracks, _, complete_ids, window_days = accepted[0]
    selected = observed.loc[
        observed["session_index"].isin(window_days)
        & observed["track_uid"].astype(str).isin(complete_ids)
    ].copy()
    baseline_day = window_days[0]
    baseline = selected.loc[
        selected["session_index"].eq(baseline_day),
        ["track_uid", "session_id", "log2_green_over_red"],
    ].rename(columns={
        "session_id": "window_baseline_session_id",
        "log2_green_over_red": "window_baseline_log2_green_over_red",
    })
    selected = selected.merge(baseline, on="track_uid", how="left", validate="many_to_one")
    selected["delta_log2_green_over_red_from_window_baseline"] = (
        selected["log2_green_over_red"] - selected["window_baseline_log2_green_over_red"]
    )
    record_lookup = {int(record.session_index): record for record in required_records}
    selected_ranking = ranking.loc[
        ranking["window_start_session_index"].eq(int(window_days[0]))
        & ranking["window_end_session_index"].eq(int(window_days[-1]))
        & ranking["status"].eq("valid")
    ].iloc[0]
    result = {
        "status": "valid",
        "selection_rule": "Longest contiguous source-order span with at least two complete tracks; each track must have genuine dual-channel signal-valid, native-fit-eligible observations on every session; every day's red predictor must be nonconstant. Tie-break by more complete tracks, then earlier source session.",
        "window_session_indices": [int(day) for day in window_days],
        "window_session_ids": [str(record_lookup[day].session_id) for day in window_days],
        "window_acquisition_dates": [record_lookup[day].acquisition_date.isoformat() for day in window_days],
        "window_length": int(length),
        "n_complete_tracks": int(n_tracks),
        "rejected_candidates": ranking.loc[ranking["status"].eq("rejected")].to_dict("records"),
        "selected_day_fit_statuses": selected_ranking["day_fit_statuses"],
    }
    return result, ranking, selected
