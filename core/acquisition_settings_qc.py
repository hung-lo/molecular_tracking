"""ThorImage acquisition-settings QC and selected-session consistency checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


REPORT_COLUMNS = [
    "session_index", "session_id", "acquisition_date", "acquisition_id", "laser_nm",
    "pmt_a_gain", "pmt_b_gain", "pockels_920_start_pct", "pockels_920_stop_pct",
    "pockels_1050_start_pct", "pockels_1050_stop_pct", "average_num", "software_version",
    "pixel_size_x_um", "pixel_size_y_um", "z_step_um",
]

# One authoritative source for the four configured Fucci mice.
EXPECTED_ACQUISITION_SETTINGS: dict[str, dict[str, float]] = {
    "Fucci-Tri_1": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 50, "laser_1050_power": 50},
    "Fucci-Tri_3": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 60, "laser_1050_power": 60},
    "Fucci-Dead_1": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 70, "laser_1050_power": 70},
    "Fucci-Dead_2": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 70, "laser_1050_power": 70},
}

QC_COLUMNS = [
    "mouse_id", "session_id", "acquisition_date", "acquisition_id", "source_path", "laser_nm",
    "pmt_a_gain", "pmt_b_gain", "pockels_920_start_pct", "pockels_920_stop_pct",
    "pockels_1050_start_pct", "pockels_1050_stop_pct", "expected_pmt_gain_a", "expected_pmt_gain_b",
    "expected_laser_920_power", "expected_laser_1050_power", "settings_qc_pass", "settings_qc_reason",
    "analysis_eligible",
]


def validate_acquisition_row(row: Mapping[str, Any], *, tolerance: float = 1e-6) -> dict[str, Any]:
    """Validate one catalog row against configured settings, returning all mismatches."""

    mouse_id = str(row.get("mouse_id", ""))
    expected = EXPECTED_ACQUISITION_SETTINGS.get(mouse_id)
    if expected is None:
        return {"settings_qc_pass": False, "analysis_eligible": False, "settings_qc_reason": "no acquisition QC configuration for mouse", "settings_qc_mismatches": ["unknown_mouse"]}
    mismatches: list[str] = []

    def check(name: str, actual: Any, expected_value: float, label: str) -> None:
        try:
            value = float(actual)
        except (TypeError, ValueError):
            mismatches.append(f"{label}={actual} expected {expected_value:g}")
            return
        if not math.isclose(value, expected_value, rel_tol=tolerance, abs_tol=tolerance):
            mismatches.append(f"{label}={value:g} expected {expected_value:g}")

    check("pmt_a_gain", row.get("pmt_a_gain"), expected["pmt_gain_a"], "PMT_A")
    check("pmt_b_gain", row.get("pmt_b_gain"), expected["pmt_gain_b"], "PMT_B")
    # A Pockels pair is checked when that laser is active in this acquisition;
    # inactive mapped lasers are valid for the paired single-laser acquisition.
    for wavelength in (920, 1050):
        start = row.get(f"pockels_{wavelength}_start_pct")
        stop = row.get(f"pockels_{wavelength}_stop_pct")
        try:
            selected_laser = int(float(row.get("laser_nm"))) == wavelength
        except (TypeError, ValueError):
            selected_laser = False
        try:
            active = abs(float(start)) > tolerance or abs(float(stop)) > tolerance
        except (TypeError, ValueError):
            active = selected_laser
        active = active or selected_laser
        if active:
            expected_power = expected[f"laser_{wavelength}_power"]
            check(f"pockels_{wavelength}_start_pct", start, expected_power, str(wavelength))
            check(f"pockels_{wavelength}_stop_pct", stop, expected_power, str(wavelength))
    passed = not mismatches
    return {
        "settings_qc_pass": passed,
        "analysis_eligible": passed,
        "settings_qc_reason": "PASS" if passed else "; ".join(mismatches),
        "settings_qc_mismatches": mismatches,
    }


def acquisition_settings_qc_table(rows: list[dict[str, Any]], *, tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a deterministic per-acquisition QC table and summary."""

    records: list[dict[str, Any]] = []
    for row in rows:
        mouse_id = str(row.get("mouse_id", ""))
        expected = EXPECTED_ACQUISITION_SETTINGS.get(mouse_id, {})
        if expected:
            result = validate_acquisition_row(row, tolerance=tolerance)
        else:
            result = {"settings_qc_pass": True, "analysis_eligible": bool(row.get("analysis_included", True)), "settings_qc_reason": "not a configured Fucci workflow"}
        record = {key: row.get(key) for key in QC_COLUMNS}
        record.update({
            "expected_pmt_gain_a": expected.get("pmt_gain_a"), "expected_pmt_gain_b": expected.get("pmt_gain_b"),
            "expected_laser_920_power": expected.get("laser_920_power"), "expected_laser_1050_power": expected.get("laser_1050_power"),
            **{key: result[key] for key in ("settings_qc_pass", "settings_qc_reason", "analysis_eligible")},
        })
        records.append(record)
    table = pd.DataFrame(records, columns=QC_COLUMNS)
    failed = table.loc[~table["settings_qc_pass"].fillna(False)] if not table.empty else table
    return table, {
        "status": "PASS" if failed.empty else "FAIL",
        "n_sessions": int(len(table)), "n_pass": int(table["settings_qc_pass"].fillna(False).sum()) if not table.empty else 0,
        "n_fail": int(len(failed)), "numeric_tolerance": tolerance,
        "failed_sessions": [{"session_id": str(row.session_id), "reason": str(row.settings_qc_reason)} for row in failed.itertuples(index=False)],
    }


def write_acquisition_settings_qc_artifacts(rows: list[dict[str, Any]], output_dir: str | Path, *, tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write the required CSV and PNG QC artifacts without touching raw data."""

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    table, summary = acquisition_settings_qc_table(rows, tolerance=tolerance)
    table.to_csv(output / "acquisition_settings_qc.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 0.45 * max(len(table), 1)), 7), sharex=True)
    labels = table["session_id"].astype(str).tolist() if not table.empty else []
    x = list(range(len(labels)))
    for axis, actual_a, actual_b, expected_a, expected_b, title in (
        (axes[0], "pmt_a_gain", "pmt_b_gain", "expected_pmt_gain_a", "expected_pmt_gain_b", "PMT gains"),
        (axes[1], "pockels_920_start_pct", "pockels_920_stop_pct", "expected_laser_920_power", "expected_laser_920_power", "920-nm Pockels power"),
        (axes[2], "pockels_1050_start_pct", "pockels_1050_stop_pct", "expected_laser_1050_power", "expected_laser_1050_power", "1050-nm Pockels power"),
    ):
        if labels:
            actual_a_values = pd.to_numeric(table[actual_a], errors="coerce")
            actual_b_values = pd.to_numeric(table[actual_b], errors="coerce")
            axis.plot(x, actual_a_values, "o", label="start/A")
            axis.plot(x, actual_b_values, "x", label="stop/B")
            expected_values = pd.to_numeric(table[expected_a], errors="coerce")
            axis.plot(x, expected_values, "--", color="black", label="expected")
            failed_x = [i for i, passed in enumerate(table["settings_qc_pass"].fillna(False)) if not passed]
            if failed_x:
                for failed_index in failed_x:
                    axis.axvline(failed_index, color="red", alpha=0.18, linewidth=3)
                if expected_values.notna().any():
                    bad_a = [i for i in failed_x if pd.notna(actual_a_values.iloc[i]) and pd.notna(expected_values.iloc[i]) and not math.isclose(float(actual_a_values.iloc[i]), float(expected_values.iloc[i]), rel_tol=tolerance, abs_tol=tolerance)]
                    bad_b = [i for i in failed_x if pd.notna(actual_b_values.iloc[i]) and pd.notna(expected_values.iloc[i]) and not math.isclose(float(actual_b_values.iloc[i]), float(expected_values.iloc[i]), rel_tol=tolerance, abs_tol=tolerance)]
                    if bad_a:
                        axis.plot(bad_a, actual_a_values.iloc[bad_a], "X", color="red", markersize=10, label="QC FAIL")
                    if bad_b:
                        axis.plot(bad_b, actual_b_values.iloc[bad_b], "X", color="red", markersize=10)
        axis.set_title(title); axis.grid(alpha=0.25)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(output / "acquisition_settings_qc.png", dpi=150); plt.close(fig)
    return table, summary


def acquisition_settings_qc(rows: list[dict], session_ids: list[str], laser_nm: int, *, tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict]:
    selected = {str(session_id): index for index, session_id in enumerate(session_ids)}
    frame = pd.DataFrame([row for row in rows if str(row.get("session_id")) in selected]).copy()
    if frame.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS), {"status": "unavailable_legacy", "changed_required_fields": [], "informational_changes": []}
    frame["session_index"] = frame["session_id"].astype(str).map(selected)
    for column in REPORT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[REPORT_COLUMNS].sort_values("session_index").reset_index(drop=True)
    required = ["pmt_a_gain", "pmt_b_gain", f"pockels_{laser_nm}_start_pct", f"pockels_{laser_nm}_stop_pct", "average_num"]
    changed = []
    for column in required:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
        if len(values) > 1 and not all(math.isclose(values[0], value, rel_tol=tolerance, abs_tol=tolerance) for value in values[1:]):
            changed.append(column)
    informational = []
    versions = frame["software_version"].dropna().astype(str).unique()
    if len(versions) > 1:
        informational.append("software_version")
    return frame, {
        "status": "warning" if changed or informational else "pass",
        "laser_nm": int(laser_nm), "numeric_tolerance": tolerance,
        "changed_required_fields": changed, "informational_changes": informational,
        "n_sessions": len(frame),
    }
