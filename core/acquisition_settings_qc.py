"""ThorImage acquisition-settings QC and selected-session consistency checks."""

from __future__ import annotations

import math
from pathlib import Path
import re
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
    "expected_laser_920_power", "expected_laser_1050_power", "settings_qc_pass", "settings_qc_status", "settings_qc_reason",
    "analysis_eligible",
]

_VOL10_RE = re.compile(r"(?<![A-Za-z0-9])vol10(?![A-Za-z0-9])", re.IGNORECASE)


def _selected_laser(value: Any, wavelength: int) -> bool:
    try:
        return int(float(value)) == wavelength
    except (TypeError, ValueError, OverflowError):
        return False


def _laser_is_active(start: Any, stop: Any, *, selected: bool, tolerance: float) -> bool:
    try:
        active = abs(float(start)) > tolerance or abs(float(stop)) > tolerance
    except (TypeError, ValueError, OverflowError):
        active = selected
    return active or selected


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
        selected_laser = _selected_laser(row.get("laser_nm"), wavelength)
        if _laser_is_active(start, stop, selected=selected_laser, tolerance=tolerance):
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
    n_vol10_excluded = 0
    for row in rows:
        if bool(row.get("is_vol10_control")) or _VOL10_RE.search(str(row.get("acquisition_id", ""))):
            n_vol10_excluded += 1
            continue
        if not row.get("analysis_included", True) and row.get("role") in {"alignment_only", "auxiliary_or_test", "noncanonical"}:
            continue
        mouse_id = str(row.get("mouse_id", ""))
        expected = EXPECTED_ACQUISITION_SETTINGS.get(mouse_id, {})
        if row.get("role") in {"missing_xml", "malformed_xml"} and row.get("settings_qc_pass") is False:
            result = {
                "settings_qc_pass": False,
                "analysis_eligible": False,
                "settings_qc_status": "fail",
                "settings_qc_reason": str(row.get("settings_qc_reason") or "acquisition metadata unavailable"),
            }
        elif expected:
            result = validate_acquisition_row(row, tolerance=tolerance)
            result["settings_qc_status"] = "pass" if result["settings_qc_pass"] else "fail"
        elif row.get("settings_qc_status") == "not_configured":
            result = {
                "settings_qc_pass": None,
                "analysis_eligible": False,
                "settings_qc_status": "not_configured",
                "settings_qc_reason": str(row.get("settings_qc_reason") or "no acquisition QC configuration for Fucci mouse"),
            }
        elif row.get("settings_qc_pass") is False:
            result = {
                "settings_qc_pass": False,
                "analysis_eligible": False,
                "settings_qc_status": "fail",
                "settings_qc_reason": str(row.get("settings_qc_reason") or "acquisition metadata unavailable"),
            }
        else:
            result = {"settings_qc_pass": True, "analysis_eligible": bool(row.get("analysis_included", True)), "settings_qc_status": "pass", "settings_qc_reason": "not a configured Fucci workflow"}
        record = {key: row.get(key) for key in QC_COLUMNS}
        record.update({
            "expected_pmt_gain_a": expected.get("pmt_gain_a"), "expected_pmt_gain_b": expected.get("pmt_gain_b"),
            "expected_laser_920_power": expected.get("laser_920_power"), "expected_laser_1050_power": expected.get("laser_1050_power"),
            **{key: result[key] for key in ("settings_qc_pass", "settings_qc_status", "settings_qc_reason", "analysis_eligible")},
        })
        records.append(record)
    table = pd.DataFrame(records, columns=QC_COLUMNS)
    failed = table.loc[table["settings_qc_status"].eq("fail")] if not table.empty else table
    return table, {
        "status": "PASS" if failed.empty else "FAIL",
        "n_sessions": int(len(table)), "n_pass": int(table["settings_qc_status"].eq("pass").sum()) if not table.empty else 0,
        "n_fail": int(len(failed)), "n_vol10_excluded": n_vol10_excluded, "n_not_configured": int(table["settings_qc_status"].eq("not_configured").sum()) if not table.empty else 0,
        "numeric_tolerance": tolerance,
        "failed_sessions": [{"session_id": str(row.session_id), "acquisition_id": str(row.acquisition_id), "reason": str(row.settings_qc_reason)} for row in failed.itertuples(index=False)],
    }


def write_acquisition_settings_qc_artifacts(rows: list[dict[str, Any]], output_dir: str | Path, *, tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write the required CSV and PNG QC artifacts without touching raw data."""

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    table, summary = acquisition_settings_qc_table(rows, tolerance=tolerance)
    table.to_csv(output / "acquisition_settings_qc.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(max(11, 0.55 * max(len(table), 1)), 11), sharex=True)
    labels = []
    if not table.empty:
        base_labels = [
            str(row.acquisition_date if pd.notna(row.acquisition_date) and row.acquisition_date else row.session_id)
            for row in table.itertuples(index=False)
        ]
        counts = pd.Series(base_labels).value_counts()
        labels = [
            f"{base}\\n{row.acquisition_id}" if counts[base] > 1 else base
            for base, row in zip(base_labels, table.itertuples(index=False), strict=True)
        ]
    x = list(range(len(labels)))
    panels = (
        ("pmt_a_gain", None, "expected_pmt_gain_a", None, "PMT A gain", "gain", None),
        ("pmt_b_gain", None, "expected_pmt_gain_b", None, "PMT B gain", "gain", None),
        ("pockels_920_start_pct", "pockels_920_stop_pct", "expected_laser_920_power", "expected_laser_920_power", "920-nm Pockels power", "%", 920),
        ("pockels_1050_start_pct", "pockels_1050_stop_pct", "expected_laser_1050_power", "expected_laser_1050_power", "1050-nm Pockels power", "%", 1050),
    )
    statuses = table["settings_qc_status"].astype(str) if not table.empty else pd.Series(dtype=str)
    for axis, (actual_a, actual_b, expected_a, expected_b, title, ylabel, wavelength) in zip(axes, panels, strict=True):
        active = pd.Series(True, index=table.index)
        if labels and wavelength is not None:
            active = pd.Series(
                [
                    _laser_is_active(
                        table.iloc[i][actual_a], table.iloc[i][actual_b],
                        selected=_selected_laser(table.iloc[i]["laser_nm"], wavelength),
                        tolerance=tolerance,
                    )
                    for i in range(len(table))
                ],
                index=table.index,
            )
        if labels:
            series = [(actual_a, expected_a, "start/A" if actual_b else "actual")]
            if actual_b:
                series.append((actual_b, expected_b, "stop/B"))
            plotted_values = []
            for actual_name, expected_name, point_label in series:
                actual_values = pd.to_numeric(table[actual_name], errors="coerce").where(active)
                expected_values = pd.to_numeric(table[expected_name], errors="coerce").where(active)
                plotted_values.extend([actual_values, expected_values])
                pass_indices = [i for i, value in enumerate(statuses.eq("pass") & actual_values.notna()) if value]
                fail_indices = [i for i, value in enumerate(statuses.eq("fail") & actual_values.notna()) if value]
                if pass_indices:
                    axis.plot(pass_indices, actual_values.iloc[pass_indices], "o", color="tab:green", label="actual / pass")
                if fail_indices:
                    axis.plot(fail_indices, actual_values.iloc[fail_indices], "o", color="tab:orange", alpha=0.7, label="actual / failed session")
                if expected_values.notna().any() and (expected_name != expected_a or point_label in {"actual", "start/A"}):
                    axis.plot(x, expected_values, "--", color="black", label="expected" if point_label in {"actual", "start/A"} else None)
                for i in range(len(table)):
                    if statuses.iloc[i] != "fail" or not active.iloc[i] or pd.isna(actual_values.iloc[i]) or pd.isna(expected_values.iloc[i]):
                        continue
                    if math.isclose(float(actual_values.iloc[i]), float(expected_values.iloc[i]), rel_tol=tolerance, abs_tol=tolerance):
                        continue
                    axis.plot([i], [actual_values.iloc[i]], "X", color="red", markersize=10, label="QC mismatch")
                    marker_label = "PMT A" if actual_name == "pmt_a_gain" else "PMT B" if actual_name == "pmt_b_gain" else f"{wavelength} {point_label.split('/')[0]}"
                    axis.annotate(f"{marker_label}={float(actual_values.iloc[i]):g}", (i, actual_values.iloc[i]), xytext=(4, 5), textcoords="offset points", fontsize=7, color="red")
            for index in [i for i, value in enumerate(statuses.eq("fail")) if value]:
                axis.axvspan(index - 0.45, index + 0.45, color="red", alpha=0.08)
            values = pd.concat(plotted_values).dropna()
            if not values.empty:
                low, high = float(values.min()), float(values.max())
                pad = max((high - low) * 0.15, abs((high + low) / 2) * 0.05, 0.5)
                axis.set_ylim(low - pad, high + pad)
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    mouse_ids = sorted({str(value) for value in table["mouse_id"].dropna()}) if not table.empty else []
    expected_text = "; ".join(
        f"{mouse}: PMT=10, 920={EXPECTED_ACQUISITION_SETTINGS[mouse]['laser_920_power']:g}, 1050={EXPECTED_ACQUISITION_SETTINGS[mouse]['laser_1050_power']:g}"
        for mouse in mouse_ids if mouse in EXPECTED_ACQUISITION_SETTINGS
    )
    n_pass = int(table["settings_qc_status"].eq("pass").sum()) if not table.empty else 0
    n_fail = int(table["settings_qc_status"].eq("fail").sum()) if not table.empty else 0
    fig.suptitle(
        f"Acquisition settings QC — {', '.join(mouse_ids) or 'no analysis acquisitions'}\n"
        f"{expected_text or 'No configured Fucci settings'} | n analysis acquisitions={len(table)}, pass={n_pass}, fail={n_fail}",
        fontsize=12,
    )
    handles, labels_legend = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label and label not in labels_legend:
                handles.append(handle); labels_legend.append(label)
    if handles:
        fig.legend(handles, labels_legend, loc="upper right", bbox_to_anchor=(0.99, 0.985), fontsize=8)
    fig.subplots_adjust(top=0.86, bottom=0.2, hspace=0.48, right=0.88)
    fig.savefig(output / "acquisition_settings_qc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
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
