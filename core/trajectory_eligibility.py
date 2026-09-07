"""Additive trajectory eligibility and PCA-preparation tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectoryEligibilityConfig:
    min_sessions: int = 2
    min_session_fraction: float | None = None
    max_internal_missing_sessions: int | None = None
    require_first_session: bool = False
    require_last_session: bool = False

    def __post_init__(self) -> None:
        if self.min_sessions < 1:
            raise ValueError("trajectory_min_sessions must be at least 1")
        if self.min_session_fraction is not None and not 0 < self.min_session_fraction <= 1:
            raise ValueError("trajectory_min_session_fraction must be within (0, 1]")
        if self.max_internal_missing_sessions is not None and self.max_internal_missing_sessions < 0:
            raise ValueError("trajectory_max_internal_missing_sessions must be nonnegative")


def build_trajectory_eligibility(
    observations: pd.DataFrame,
    tracks: pd.DataFrame,
    session_ids: list[str],
    config: TrajectoryEligibilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    usable_observations = observations.loc[
        observations["ratio_qc_pass"].eq(True)
        & observations["green_fit_signed_distance"].notna()
    ].copy()
    if "geometry_qc_pass" in usable_observations and not usable_observations["geometry_qc_pass"].isna().all():
        usable_observations = usable_observations.loc[usable_observations["geometry_qc_pass"].eq(True)]
    for _, track in tracks.iterrows():
        track_uid = str(track["track_uid"])
        observed = observations.loc[observations["track_uid"].astype(str).eq(track_uid)]
        usable = usable_observations.loc[usable_observations["track_uid"].astype(str).eq(track_uid)]
        usable_indices = sorted(usable["session_index"].astype(int).unique())
        present_indices = sorted(observed["session_index"].astype(int).unique())
        n_required = len(session_ids)
        n_usable = len(usable_indices)
        internal_missing = 0
        max_run = 0
        if usable_indices:
            span = set(range(usable_indices[0], usable_indices[-1] + 1))
            internal_missing = len(span.difference(usable_indices))
            run = 1
            max_run = 1
            for previous, current in zip(usable_indices, usable_indices[1:]):
                run = run + 1 if current == previous + 1 else 1
                max_run = max(max_run, run)
        fraction = n_usable / n_required if n_required else 0.0
        reasons = []
        if n_usable < config.min_sessions:
            reasons.append("below_min_sessions")
        if config.min_session_fraction is not None and fraction < config.min_session_fraction:
            reasons.append("below_min_fraction")
        if config.max_internal_missing_sessions is not None and internal_missing > config.max_internal_missing_sessions:
            reasons.append("too_many_internal_missing_sessions")
        if config.require_first_session and 0 not in usable_indices:
            reasons.append("missing_first_session")
        if config.require_last_session and n_required - 1 not in usable_indices:
            reasons.append("missing_last_session")
        has_cycle_conflict = pd.notna(track.get("has_cycle_conflict")) and bool(track.get("has_cycle_conflict"))
        if has_cycle_conflict:
            reasons.append("cycle_conflict")
        first = usable.sort_values("session_index").iloc[0] if n_usable else None
        last = usable.sort_values("session_index").iloc[-1] if n_usable else None
        rows.append({
            "match_policy": track.get("match_policy"), "roi_id": track.get("roi_id"),
            "track_uid": track_uid, "cluster_id": track.get("cluster_id"),
            "n_required_sessions": n_required, "n_track_labels_present": len(present_indices),
            "n_signal_valid_sessions": int(observed["ratio_qc_pass"].eq(True).sum()),
            "n_usable_trajectory_sessions": n_usable,
            "n_track_present_but_signal_invalid": len(present_indices) - int(observed["ratio_qc_pass"].eq(True).sum()),
            "usable_session_fraction": fraction,
            "first_usable_session_index": first["session_index"] if first is not None else np.nan,
            "last_usable_session_index": last["session_index"] if last is not None else np.nan,
            "first_usable_session_id": first["session_id"] if first is not None else None,
            "last_usable_session_id": last["session_id"] if last is not None else None,
            "first_usable_elapsed_days": first["elapsed_days"] if first is not None else np.nan,
            "last_usable_elapsed_days": last["elapsed_days"] if last is not None else np.nan,
            "usable_elapsed_day_span": (last["elapsed_days"] - first["elapsed_days"]) if first is not None else np.nan,
            "n_internal_missing_sessions": internal_missing, "max_consecutive_usable_sessions": max_run,
            "has_first_required_session": 0 in usable_indices,
            "has_last_required_session": n_required - 1 in usable_indices,
            "has_cycle_conflict": has_cycle_conflict,
            "used_gap_bridge": pd.notna(track.get("used_gap_bridge")) and bool(track.get("used_gap_bridge")),
            "review_required": pd.notna(track.get("review_required")) and bool(track.get("review_required")),
            "review_reasons": track.get("review_reasons", ""),
            "trajectory_min_sessions_threshold": config.min_sessions,
            "trajectory_min_fraction_threshold": config.min_session_fraction,
            "trajectory_max_internal_missing_threshold": config.max_internal_missing_sessions,
            "trajectory_require_first_session": config.require_first_session,
            "trajectory_require_last_session": config.require_last_session,
            "trajectory_eligible": not reasons,
            "trajectory_ineligibility_reasons": ",".join(reasons),
        })
    eligibility = pd.DataFrame(rows)
    eligible_ids = set(eligibility.loc[eligibility["trajectory_eligible"], "track_uid"].astype(str))
    eligible_observations = usable_observations.loc[
        usable_observations["track_uid"].astype(str).isin(eligible_ids)
    ].copy()
    return eligibility, eligible_observations


def build_trajectory_matrices(
    observations: pd.DataFrame, session_ids: list[str]
) -> dict[str, pd.DataFrame]:
    index_columns = [column for column in ["match_policy", "track_uid", "roi_id"] if column in observations]
    raw = observations.pivot_table(
        index=index_columns, columns="session_id", values="green_fit_signed_distance", aggfunc="first"
    ).reindex(columns=session_ids)
    mask = raw.notna().astype(int)
    centered = raw - raw.mean(axis=0)
    center_summary = pd.DataFrame({
        "session_id": session_ids, "n_observed": raw.count().to_numpy(),
        "mean_before_centering": raw.mean().to_numpy(), "std_before_centering": raw.std().to_numpy(),
        "mean_after_centering": centered.mean().to_numpy(),
    })
    complete_raw = raw.dropna()
    complete_centered = complete_raw - complete_raw.mean(axis=0)
    complete_summary = pd.DataFrame({
        "session_id": session_ids, "n_observed": complete_raw.count().to_numpy(),
        "mean_before_centering": complete_raw.mean().to_numpy(),
        "mean_after_centering": complete_centered.mean().to_numpy(),
    })
    missingness = pd.DataFrame({
        "session_id": session_ids, "n_eligible_tracks": len(raw),
        "n_observed": raw.count().to_numpy(), "n_missing": raw.isna().sum().to_numpy(),
        "observed_fraction": raw.notna().mean().to_numpy(),
    })
    return {
        "raw": raw.reset_index(), "mask": mask.reset_index(), "missingness": missingness,
        "centered": centered.reset_index(), "centering": center_summary,
        "complete_raw": complete_raw.reset_index(),
        "complete_centered": complete_centered.reset_index(),
        "complete_centering": complete_summary,
    }
