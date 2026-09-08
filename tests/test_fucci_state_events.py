import pandas as pd

from fucci_state_events import (
    build_event_aligned_observations,
    detect_middle_entry_events,
    summarize_event_aligned_observations,
)


def _observations(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "match_policy": "graph", "track_uid": "track1", "roi_id": 1,
        "session_index": range(len(values)), "session_id": [f"s{i}" for i in range(len(values))],
        "elapsed_days": [0, 2, 5, 9, 12, 20, 24, 28, 31][:len(values)], "color_z": values,
        "color_state_qc_pass": True,
    })


def test_transition_zones_create_hysteretic_entries_and_retain_repeats() -> None:
    observations = _observations([0, -1.5, -2.5, -0.5, -1.5, -2.2, 0, 1.5, 2.5])
    events = detect_middle_entry_events(observations, min_usable_sessions=8)
    assert events["event_type"].tolist() == ["middle_to_low", "middle_to_low", "middle_to_high"]
    assert events["event_rank_within_roi_direction"].tolist() == [1, 2, 1]
    assert events["onset_session_index"].tolist() == [2, 5, 8]
    aligned = build_event_aligned_observations(observations, events)
    summary = summarize_event_aligned_observations(aligned, events=events)
    assert summary["n_events_contributing"].max() == 2
    assert aligned.loc[aligned["event_id"].eq(events.iloc[0]["event_id"]), "relative_elapsed_days"].iloc[0] == -5


def test_low_to_high_is_not_a_middle_entry() -> None:
    events = detect_middle_entry_events(_observations([-2.5, 2.5] + [0] * 6), min_usable_sessions=8)
    assert events.empty
