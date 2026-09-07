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


def test_acquisition_qc_constant_and_nonselected_laser_change() -> None:
    rows = [{"session_id": f"s{i}", "pmt_a_gain": 1, "pmt_b_gain": 2, "pockels_1050_start_pct": 3, "pockels_1050_stop_pct": 4, "pockels_920_start_pct": i, "pockels_920_stop_pct": i, "average_num": 2} for i in range(2)]
    _, constant = acquisition_settings_qc(rows, ["s0", "s1"], 1050)
    assert constant["status"] == "pass"
    rows[1]["pockels_1050_stop_pct"] = 8
    _, changed = acquisition_settings_qc(rows, ["s0", "s1"], 1050)
    assert changed["status"] == "warning"
    assert "pockels_1050_stop_pct" in changed["changed_required_fields"]
    assert "pockels_920_start_pct" not in changed["changed_required_fields"]


def test_acquisition_qc_only_uses_selected_subset() -> None:
    rows = [{"session_id": "s0", "pmt_a_gain": 1}, {"session_id": "s1", "pmt_a_gain": 1}, {"session_id": "s2", "pmt_a_gain": 9}]
    _, qc = acquisition_settings_qc(rows, ["s0", "s1"], 1050)
    assert qc["status"] == "pass" and qc["n_sessions"] == 2


def test_software_change_warns_but_is_not_strict_numeric_failure() -> None:
    rows = [{"session_id": "s0", "pmt_a_gain": 1, "pmt_b_gain": 2, "pockels_1050_start_pct": 3, "pockels_1050_stop_pct": 4, "average_num": 2, "software_version": "a"}, {"session_id": "s1", "pmt_a_gain": 1, "pmt_b_gain": 2, "pockels_1050_start_pct": 3, "pockels_1050_stop_pct": 4, "average_num": 2, "software_version": "b"}]
    _, qc = acquisition_settings_qc(rows, ["s0", "s1"], 1050)
    assert qc["status"] == "warning"
    assert qc["changed_required_fields"] == []
