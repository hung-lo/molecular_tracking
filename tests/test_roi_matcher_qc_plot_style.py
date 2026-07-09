import numpy as np
import pandas as pd

from roi_matcher_qc_plots import (
    build_single_plane_display_layers,
    format_cluster_header,
    format_panel_title,
)


def test_build_single_plane_display_layers_separates_matched_and_neighbor_masks() -> None:
    neighbor_a = np.zeros((6, 6), dtype=bool)
    neighbor_b = np.zeros((6, 6), dtype=bool)
    matched = np.zeros((6, 6), dtype=bool)
    neighbor_a[0:2, 0:2] = True
    neighbor_b[4:6, 4:6] = True
    matched[2:5, 2:4] = True

    display = build_single_plane_display_layers(
        matched_mask=matched,
        neighbor_masks={10: neighbor_a, 11: neighbor_b},
    )

    assert display.shape == (6, 6)
    assert set(np.unique(display)) == {0, 1, 2}
    assert np.all(display[neighbor_a] == 1)
    assert np.all(display[neighbor_b] == 1)
    assert np.all(display[matched] == 2)


def test_build_single_plane_display_layers_without_match_keeps_neighbors_only() -> None:
    neighbor = np.zeros((5, 5), dtype=bool)
    neighbor[1:4, 1:4] = True

    display = build_single_plane_display_layers(
        matched_mask=None,
        neighbor_masks={20: neighbor},
    )

    assert set(np.unique(display)) == {0, 1}
    assert np.all(display[neighbor] == 1)


def test_format_panel_title_reports_week_roi_and_z() -> None:
    title = format_panel_title(day_name="week2", matched_label=445, z_index=29)

    assert title == "week2\nROI 445 | z=29"


def test_format_panel_title_marks_missing_roi() -> None:
    title = format_panel_title(day_name="week3", matched_label=None, z_index=28)

    assert title == "week3\nmissing | z=28"


def test_format_cluster_header_reports_cluster_days_and_confidence() -> None:
    track_row = pd.Series(
        {
            "cluster_id": 1,
            "n_days_present": 4,
            "mean_confidence": 0.8237353649,
        }
    )

    header = format_cluster_header(track_row)

    assert header == "Cluster 1 (n_days=4, confidence=0.824)"
