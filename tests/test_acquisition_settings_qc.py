from acquisition_settings_qc import acquisition_settings_qc


def test_selected_laser_acquisition_changes_warn() -> None:
    rows = [
        {"session_id": "s0", "laser_nm": 1050, "pmt_a_gain": 1, "pmt_b_gain": 2, "pockels_1050_start_pct": 3, "pockels_1050_stop_pct": 4, "pockels_920_start_pct": 9, "pockels_920_stop_pct": 9, "average_num": 2},
        {"session_id": "s1", "laser_nm": 1050, "pmt_a_gain": 5, "pmt_b_gain": 2, "pockels_1050_start_pct": 3, "pockels_1050_stop_pct": 4, "pockels_920_start_pct": 1, "pockels_920_stop_pct": 1, "average_num": 2},
    ]
    table, qc = acquisition_settings_qc(rows, ["s0", "s1"], 1050)
    assert table["session_id"].tolist() == ["s0", "s1"]
    assert qc["status"] == "warning"
    assert qc["changed_required_fields"] == ["pmt_a_gain"]
