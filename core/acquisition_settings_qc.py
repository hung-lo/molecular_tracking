"""Selected-session acquisition setting provenance and consistency checks."""

from __future__ import annotations

import math

import pandas as pd


REPORT_COLUMNS = [
    "session_index", "session_id", "acquisition_date", "acquisition_id", "laser_nm",
    "pmt_a_gain", "pmt_b_gain", "pockels_920_start_pct", "pockels_920_stop_pct",
    "pockels_1050_start_pct", "pockels_1050_stop_pct", "average_num", "software_version",
    "pixel_size_x_um", "pixel_size_y_um", "z_step_um",
]


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
        "status": "warning" if changed else "pass",
        "laser_nm": int(laser_nm), "numeric_tolerance": tolerance,
        "changed_required_fields": changed, "informational_changes": informational,
        "n_sessions": len(frame),
    }
