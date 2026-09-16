from acquisition_settings_qc import acquisition_settings_qc, acquisition_settings_qc_table, validate_acquisition_row


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


def _configured_row(**overrides):
    row = {
        "mouse_id": "Fucci-Tri_3", "session_id": "s0", "acquisition_date": "2026-08-20", "acquisition_id": "a0",
        "pmt_a_gain": 10, "pmt_b_gain": 10, "pockels_920_start_pct": 60, "pockels_920_stop_pct": 60,
        "pockels_1050_start_pct": 0, "pockels_1050_stop_pct": 0,
    }
    row.update(overrides)
    return row


def test_configured_session_passes_and_is_analysis_eligible():
    result = validate_acquisition_row(_configured_row())
    assert result["settings_qc_pass"] is True and result["analysis_eligible"] is True


def test_wrong_pmt_reports_named_mismatch():
    result = validate_acquisition_row(_configured_row(pmt_b_gain=8))
    assert result["settings_qc_pass"] is False and "PMT_B=8 expected 10" in result["settings_qc_reason"]


def test_wrong_laser_and_multiple_problems_are_all_reported():
    result = validate_acquisition_row(_configured_row(pmt_a_gain=9, pockels_920_stop_pct=50))
    assert "PMT_A=9 expected 10" in result["settings_qc_reason"]
    assert "920=50 expected 60" in result["settings_qc_reason"]


def test_missing_fields_fail_closed_and_unknown_mouse_is_not_guessed():
    missing = validate_acquisition_row({"mouse_id": "Fucci-Dead_1"})
    unknown = validate_acquisition_row({"mouse_id": "unknown"})
    assert missing["settings_qc_pass"] is False and missing["analysis_eligible"] is False
    assert unknown["settings_qc_pass"] is False and unknown["analysis_eligible"] is False
    assert "no acquisition QC configuration" in unknown["settings_qc_reason"]


def test_selected_laser_must_have_an_active_expected_power():
    result = validate_acquisition_row(_configured_row(laser_nm=1050, pockels_920_start_pct=0, pockels_920_stop_pct=0))
    assert result["settings_qc_pass"] is False and "1050=0 expected 60" in result["settings_qc_reason"]


def test_qc_table_has_required_gate_columns():
    table, summary = acquisition_settings_qc_table([_configured_row()])
    assert {"settings_qc_pass", "settings_qc_reason", "analysis_eligible"}.issubset(table.columns)
    assert summary["status"] == "PASS"


def test_non_fucci_rows_are_not_reported_as_fucci_failures():
    table, summary = acquisition_settings_qc_table([{"mouse_id": "mouse_1", "session_id": "s0", "analysis_included": True, "pmt_a_gain": 999}])
    assert bool(table.iloc[0]["settings_qc_pass"]) is True
    assert bool(table.iloc[0]["analysis_eligible"]) is True
    assert summary["status"] == "PASS"


def test_non_configured_hard_failure_remains_failed_in_qc_table():
    table, summary = acquisition_settings_qc_table([{
        "mouse_id": "mouse_1", "session_id": "s0", "role": "missing_xml",
        "analysis_included": False, "settings_qc_pass": False,
        "analysis_eligible": False, "settings_qc_reason": "missing Experiment.xml",
    }])
    assert bool(table.iloc[0]["settings_qc_pass"]) is False
    assert bool(table.iloc[0]["analysis_eligible"]) is False
    assert "missing Experiment.xml" in table.iloc[0]["settings_qc_reason"]
    assert summary["status"] == "FAIL"
