"""Hysteretic middle-to-extreme state entry events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fucci_color_state import color_core_state, color_state_bin


def _group_columns(observations: pd.DataFrame) -> list[str]:
    columns = [column for column in ("match_policy", "track_uid", "roi_id") if column in observations]
    if "track_uid" not in columns:
        raise ValueError("Event analysis requires track_uid")
    return columns


def _z_column(observations: pd.DataFrame) -> str:
    if "eclipse_z" in observations:
        return "eclipse_z"
    if "color_z" in observations:
        return "color_z"
    raise ValueError("Event analysis requires eclipse_z")


def _usable_count(group: pd.DataFrame) -> int:
    if "color_state_qc_pass" in group:
        valid = group["color_state_qc_pass"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        valid = pd.to_numeric(group[_z_column(group)], errors="coerce").notna()
    return int(group.loc[valid, "session_id"].astype(str).nunique()) if "session_id" in group else int(valid.sum())


def detect_middle_entry_events(
    observations: pd.DataFrame,
    *,
    min_usable_sessions: int = 8,
) -> pd.DataFrame:
    """Return every middle-to-low/high entry, ignoring transition observations."""

    if min_usable_sessions < 1:
        raise ValueError("min_usable_sessions must be at least 1")
    required = {"track_uid", "roi_id", "session_index", "elapsed_days"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Event observations are missing: {', '.join(sorted(missing))}")
    group_columns = _group_columns(observations)
    z_column = _z_column(observations)
    rows: list[dict[str, object]] = []
    for group_key, group in observations.groupby(group_columns, sort=False, dropna=False):
        group = group.sort_values("session_index", kind="stable").reset_index(drop=True)
        usable_count = _usable_count(group)
        if usable_count < min_usable_sessions:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(group_columns, group_key, strict=True))
        previous_state: str | None = None
        ranks = {"middle_to_low": 0, "middle_to_high": 0}
        for _, observation in group.iterrows():
            value = pd.to_numeric(pd.Series([observation[z_column]]), errors="coerce").iloc[0]
            state = color_core_state(float(value)) if pd.notna(value) else None
            if state is None:
                continue
            event_type = None
            if previous_state == "middle" and state == "low":
                event_type = "middle_to_low"
            elif previous_state == "middle" and state == "high":
                event_type = "middle_to_high"
            if event_type is not None:
                ranks[event_type] += 1
                event_id = "|".join([str(identity.get("match_policy", "")), str(identity["track_uid"]), str(identity["roi_id"]), event_type, str(ranks[event_type])])
                rows.append({
                    **identity,
                    "event_id": event_id,
                    "event_type": event_type,
                    "event_rank_within_roi_direction": ranks[event_type],
                    "onset_session_index": int(observation["session_index"]),
                    "onset_session_id": observation.get("session_id"),
                    "onset_acquisition_date": observation.get("acquisition_date"),
                    "onset_elapsed_days": float(observation["elapsed_days"]),
                    "onset_color_z": float(value),
                    "onset_eclipse_z": float(value),
                    "previous_core_state": previous_state,
                    "previous_core_session_index": _previous_core_index(group, observation.name, previous_state),
                    "previous_core_elapsed_days": _previous_core_elapsed(group, observation.name, previous_state),
                    "n_usable_sessions_for_track": usable_count,
                })
            previous_state = state
    columns = [
        "event_id", "match_policy", "track_uid", "roi_id", "event_type", "event_rank_within_roi_direction",
        "onset_session_index", "onset_session_id", "onset_acquisition_date", "onset_elapsed_days", "onset_color_z", "onset_eclipse_z",
        "previous_core_state", "previous_core_session_index", "previous_core_elapsed_days", "n_usable_sessions_for_track",
    ]
    return pd.DataFrame(rows, columns=[column for column in columns if column in rows[0]] if rows else columns)


def _previous_core_index(group: pd.DataFrame, current_index: int, state: str | None) -> int | float:
    if state is None:
        return np.nan
    for index in range(current_index - 1, -1, -1):
        z_column = _z_column(group)
        value = pd.to_numeric(pd.Series([group.loc[index, z_column]]), errors="coerce").iloc[0]
        if pd.notna(value) and color_core_state(float(value)) is not None:
            return int(group.loc[index, "session_index"])
    return np.nan


def _previous_core_elapsed(group: pd.DataFrame, current_index: int, state: str | None) -> float:
    if state is None:
        return np.nan
    for index in range(current_index - 1, -1, -1):
        z_column = _z_column(group)
        value = pd.to_numeric(pd.Series([group.loc[index, z_column]]), errors="coerce").iloc[0]
        if pd.notna(value) and color_core_state(float(value)) is not None:
            return float(group.loc[index, "elapsed_days"])
    return np.nan


def build_event_aligned_observations(
    observations: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Keep all source observations and add session/day coordinates around events."""

    columns = ["event_id", "track_uid", "roi_id", "event_type", "source_session_index", "source_session_id", "source_elapsed_days", "relative_session_index", "relative_elapsed_days", "color_z", "eclipse_z", "color_state_bin", "eclipse_state_bin"]
    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame(columns=columns)
    group_columns = [column for column in ("match_policy", "track_uid", "roi_id") if column in observations and column in events]
    for _, event in events.iterrows():
        mask = pd.Series(True, index=observations.index)
        for column in group_columns:
            mask &= observations[column].astype(str).eq(str(event[column]))
        source = observations.loc[mask].sort_values("session_index")
        for _, item in source.iterrows():
            z_column = _z_column(source)
            value = pd.to_numeric(pd.Series([item.get(z_column)]), errors="coerce").iloc[0]
            rows.append({
                "event_id": event["event_id"],
                "match_policy": event.get("match_policy"),
                "track_uid": event["track_uid"],
                "roi_id": event["roi_id"],
                "event_type": event["event_type"],
                "source_session_index": int(item["session_index"]),
                "source_session_id": item.get("session_id"),
                "source_elapsed_days": float(item["elapsed_days"]),
                "relative_session_index": int(item["session_index"]) - int(event["onset_session_index"]),
                "relative_elapsed_days": float(item["elapsed_days"]) - float(event["onset_elapsed_days"]),
                "color_z": value,
                "eclipse_z": value,
                "color_state_bin": item.get("color_state_bin") or (color_state_bin(float(value)) if pd.notna(value) else None),
                "eclipse_state_bin": item.get("eclipse_state_bin") or (color_state_bin(float(value)) if pd.notna(value) else None),
            })
    return pd.DataFrame(rows, columns=columns)


def first_events_per_roi_direction(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events.loc[events["event_rank_within_roi_direction"].eq(1)].copy()


def summarize_event_aligned_observations(
    aligned: pd.DataFrame,
    *,
    first_events_only: bool = True,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize event-triggered color-Z values with event and ROI counts."""

    if aligned.empty:
        return pd.DataFrame(columns=["relative_session_index", "relative_elapsed_days", "mean_color_z", "sem_color_z", "median_color_z", "mean_eclipse_z", "sem_eclipse_z", "median_eclipse_z", "n_events_contributing", "n_unique_rois_contributing"])
    source = aligned
    if first_events_only and events is not None:
        first_ids = set(first_events_per_roi_direction(events)["event_id"])
        source = source.loc[source["event_id"].isin(first_ids)]
    z_column = _z_column(source)
    source = source.loc[pd.to_numeric(source[z_column], errors="coerce").notna()].copy()
    rows = []
    for (session_offset, elapsed_offset), group in source.groupby(["relative_session_index", "relative_elapsed_days"], sort=True):
        values = pd.to_numeric(group[z_column], errors="coerce").to_numpy(dtype=float)
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        median = float(np.median(values))
        rows.append({
            "relative_session_index": int(session_offset),
            "relative_elapsed_days": float(elapsed_offset),
            "mean_color_z": mean,
            "sem_color_z": sem,
            "median_color_z": median,
            "mean_eclipse_z": mean,
            "sem_eclipse_z": sem,
            "median_eclipse_z": median,
            "n_events_contributing": int(group["event_id"].nunique()),
            "n_unique_rois_contributing": int(group[["track_uid", "roi_id"]].drop_duplicates().shape[0]),
        })
    return pd.DataFrame(rows)
