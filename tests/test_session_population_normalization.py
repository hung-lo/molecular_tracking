from datetime import date

import numpy as np
import pandas as pd
import tifffile

from session_manifest import SessionRecord
from session_population_normalization import extract_session_population, summarize_signal_qc


def _write_session(tmp_path, session_id, red, green):
    mask = np.array([[[1, 2]]], dtype=np.uint16)
    mask_path = tmp_path / f"{session_id}_mask.tif"
    red_path = tmp_path / f"{session_id}_red.tif"
    green_path = tmp_path / f"{session_id}_green.tif"
    tifffile.imwrite(mask_path, mask)
    tifffile.imwrite(red_path, np.array([[[red[0], red[1]]]], dtype=np.uint16))
    tifffile.imwrite(green_path, np.array([[[green[0], green[1]]]], dtype=np.uint16))
    return SessionRecord(0, session_id, date(2026, 1, 1), mask_path, red_path, green_path)


def test_native_population_keeps_both_channel_zero_hit_and_has_no_longitudinal_id(tmp_path):
    first = _write_session(tmp_path, "s0", (10, 0), (20, 0))
    second = _write_session(tmp_path, "s1", (10, 30), (20, 0))
    second = SessionRecord(1, "s1", date(2026, 1, 2), second.mask_path, second.red_image_path, second.green_image_path)
    population = extract_session_population([first, second], green_dark=0, red_dark=0, epsilon=1)

    assert len(population) == 4
    assert "roi_id" not in population.columns
    assert set(zip(population.session_id, population.session_roi_label)) == {("s0", 1), ("s0", 2), ("s1", 1), ("s1", 2)}
    zero_hit = population.loc[(population.session_id == "s0") & (population.session_roi_label == 2)].iloc[0]
    assert pd.isna(zero_hit.red) and pd.isna(zero_hit.green)
    assert not bool(zero_hit.red_zero_hit_pass) and not bool(zero_hit.green_zero_hit_pass)
    assert not bool(zero_hit.ratio_qc_pass)
    qc = summarize_signal_qc(population).set_index("session_id")
    assert qc.loc["s0", "n_native_rois"] == 2
    assert qc.loc["s0", "n_ratio_valid"] == 1
