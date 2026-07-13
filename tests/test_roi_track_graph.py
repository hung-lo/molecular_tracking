from __future__ import annotations

import pandas as pd

from roi_track_graph import build_cycle_consistency_tables, build_tracks_from_pair_tables, summarize_track_cycle_metadata


def _features(labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"label": labels}, index=labels)


def _edge_table(label_a: int, label_b: int, pair_gap: int = 1, score: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label_a": label_a,
                "label_b": label_b,
                "score": score,
                "dice": 1.0,
                "distance_um": 0.0,
                "ambiguity": 0.5,
                "high_rule": True,
                "balanced_rule": True,
                "candidate_source": "both",
                "assignment_policy": "high",
                "transform_method": "restricted_affine",
                "transform_fallback_reason": None,
                "pair_gap": pair_gap,
                "match_policy": "high",
            }
        ]
    )


def test_build_tracks_preserves_singletons_and_deterministic_ids() -> None:
    day_names = ["d0", "d1", "d2"]
    features_by_session = {
        "d0": _features([1, 2]),
        "d1": _features([11, 22]),
        "d2": _features([111, 222]),
    }
    pair_tables = {
        ("d0", "d1"): _edge_table(1, 11),
        ("d1", "d2"): _edge_table(11, 111),
    }

    tracks, edges = build_tracks_from_pair_tables(day_names, features_by_session, pair_tables, match_policy="high")

    assert tracks["track_uid"].iloc[0] == "d0:1"
    assert tracks["n_days_present"].max() == 3
    assert int((tracks["n_days_present"] == 1).sum()) == 3
    assert tracks["d0_roi"].dtype.name == "Int64"
    assert set(edges["accepted_for_track"].tolist()) == {True}


def test_cycle_consistency_metadata_flags_conflict() -> None:
    day_names = ["d0", "d1", "d2"]
    features_by_session = {
        "d0": _features([1]),
        "d1": _features([11]),
        "d2": _features([111, 222]),
    }
    pair_tables = {
        ("d0", "d1"): _edge_table(1, 11),
        ("d1", "d2"): _edge_table(11, 111),
        ("d0", "d2"): _edge_table(1, 222),
    }

    tracks, _ = build_tracks_from_pair_tables(day_names, features_by_session, pair_tables, match_policy="high")
    cycle_summary, cycle_checks = build_cycle_consistency_tables(day_names, pair_tables, tracks, match_policy="high")
    tracks = summarize_track_cycle_metadata(tracks, cycle_checks)

    assert cycle_summary["n_comparable"].iloc[0] == 1
    assert cycle_summary["n_agree"].iloc[0] == 0
    assert bool(tracks.loc[tracks["track_uid"] == "d0:1", "has_cycle_conflict"].iloc[0])
    assert bool(tracks.loc[tracks["track_uid"] == "d0:1", "review_required"].iloc[0])
