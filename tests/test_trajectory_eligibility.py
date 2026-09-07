import numpy as np
import pandas as pd

from trajectory_eligibility import TrajectoryEligibilityConfig, build_trajectory_eligibility, build_trajectory_matrices


def test_eligibility_and_centering_preserve_missingness() -> None:
    rows = []
    for track, indices in (("a", range(3)), ("b", (0, 2)), ("c", (0,))):
        for index in indices:
            rows.append({"match_policy": "graph", "track_uid": track, "roi_id": track, "session_index": index, "session_id": f"s{index}", "elapsed_days": index * 2, "ratio_qc_pass": True, "green_fit_signed_distance": float(index + (track == "b"))})
    observations = pd.DataFrame(rows)
    tracks = pd.DataFrame({"match_policy": "graph", "track_uid": ["a", "b", "c"], "roi_id": ["a", "b", "c"], "cluster_id": [1, 2, 3], "has_cycle_conflict": False})
    eligibility, eligible = build_trajectory_eligibility(observations, tracks, ["s0", "s1", "s2"], TrajectoryEligibilityConfig(min_sessions=2))
    assert eligibility.set_index("track_uid")["trajectory_eligible"].to_dict() == {"a": True, "b": True, "c": False}
    assert eligibility.set_index("track_uid").loc["b", "n_internal_missing_sessions"] == 1
    matrices = build_trajectory_matrices(eligible, ["s0", "s1", "s2"])
    centered = matrices["centered"].set_index("track_uid")[["s0", "s1", "s2"]]
    assert np.allclose(centered.mean(), 0)
    assert np.array_equal(matrices["mask"][["s0", "s1", "s2"]].to_numpy(), matrices["raw"][["s0", "s1", "s2"]].notna().astype(int).to_numpy())
    assert matrices["complete_centered"][["s0", "s1", "s2"]].isna().sum().sum() == 0


def test_signal_invalid_is_not_tracking_missingness() -> None:
    observations = pd.DataFrame([
        {"track_uid": "d", "roi_id": 4, "session_index": i, "session_id": f"s{i}", "elapsed_days": i, "ratio_qc_pass": i != 1, "green_fit_signed_distance": float(i) if i != 1 else np.nan}
        for i in range(3)
    ])
    tracks = pd.DataFrame({"track_uid": ["d"], "roi_id": [4], "cluster_id": [4], "has_cycle_conflict": [False]})
    eligibility, _ = build_trajectory_eligibility(observations, tracks, ["s0", "s1", "s2"], TrajectoryEligibilityConfig(min_sessions=2))
    row = eligibility.iloc[0]
    assert row["n_track_labels_present"] == 3
    assert row["n_signal_valid_sessions"] == 2
    assert row["n_usable_trajectory_sessions"] == 2
    assert row["n_track_present_but_signal_invalid"] == 1
