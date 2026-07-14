from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_weekly_matched_output_quick_plots import _filter_table_by_policy, _resolve_output_dir


def test_filter_table_by_policy_filters_suffixed_policy_columns() -> None:
    table = pd.DataFrame(
        {
            "roi_id": [1, 2, 3, 4],
            "day": [0, 1, 0, 1],
            "match_policy_x": ["high", "high", "balanced", "balanced"],
            "match_policy_y": ["high", "high", "balanced", "balanced"],
        }
    )

    filtered = _filter_table_by_policy(table, "high")

    assert filtered["roi_id"].tolist() == [1, 2]
    assert set(filtered["match_policy_x"].astype(str)) == {"high"}
    assert set(filtered["match_policy_y"].astype(str)) == {"high"}


def test_filter_table_by_policy_is_noop_without_policy_column() -> None:
    table = pd.DataFrame({"roi_id": [1, 2], "day": [0, 1]})

    filtered = _filter_table_by_policy(table, "high")

    pd.testing.assert_frame_equal(filtered, table)


def test_resolve_output_dir_separates_policies() -> None:
    analysis_dir = Path('/tmp/analysis')

    default_high = _resolve_output_dir(analysis_dir, None, 'high')
    custom_balanced = _resolve_output_dir(analysis_dir, Path('/tmp/custom_quick_plots'), 'balanced')

    assert default_high == analysis_dir / 'quick_plots' / 'high'
    assert custom_balanced == Path('/tmp/custom_quick_plots') / 'balanced'

