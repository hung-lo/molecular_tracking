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

# One authoritative source for every Fucci mouse enabled for longitudinal analysis.
EXPECTED_ACQUISITION_SETTINGS: dict[str, dict[str, float]] = {
    "Fucci-Tri_1": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 50, "laser_1050_power": 50},
    "Fucci-Tri_3": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 60, "laser_1050_power": 60},
    "Fucci-Tri_4": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 70, "laser_1050_power": 70},
    "Fucci-Dead_1": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 70, "laser_1050_power": 70},
    "Fucci-Dead_2": {"pmt_gain_a": 10, "pmt_gain_b": 10, "laser_920_power": 70, "laser_1050_power": 70},
}

QC_COLUMNS = [
    "mouse_id", "pipeline_enabled", "pipeline_exclusion_reason", "session_id", "acquisition_date", "acquisition_id", "source_path", "role", "analysis_included", "is_vol10_control", "laser_nm",
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
    mismatches = list(dict.fromkeys(mismatches))
    passed = not mismatches
    return {
        "settings_qc_pass": passed,
        "analysis_eligible": passed,
        "settings_qc_reason": "PASS" if passed else "; ".join(mismatches),
        "settings_qc_mismatches": mismatches,
    }


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return str(value).strip().lower() == "true"


def acquisition_settings_qc_table(rows: list[dict[str, Any]], *, tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a complete acquisition audit table and analysis-specific summary."""

    records: list[dict[str, Any]] = []
    n_vol10_excluded = 0
    for row in rows:
        mouse_id = str(row.get("mouse_id", ""))
        enabled = _as_bool(row.get("pipeline_enabled"), default=True)
        included = _as_bool(row.get("analysis_included"), default=True)
        is_vol10 = _as_bool(row.get("is_vol10_control")) or bool(_VOL10_RE.search(str(row.get("acquisition_id", ""))))
        expected = EXPECTED_ACQUISITION_SETTINGS.get(mouse_id, {})
        if is_vol10:
            n_vol10_excluded += 1
        if not enabled:
            result = {
                "settings_qc_pass": None,
                "analysis_eligible": False,
                "settings_qc_status": "not_applicable_pipeline_excluded",
                "settings_qc_reason": f"pipeline excluded: {row.get('pipeline_exclusion_reason') or 'no reason supplied'}",
            }
        elif is_vol10:
            result = {
                "settings_qc_pass": None,
                "analysis_eligible": False,
                "settings_qc_status": "not_applicable_vol10",
                "settings_qc_reason": "excluded: _vol10 acquisition is not used for analysis",
            }
        elif row.get("role") in {"missing_xml", "malformed_xml"} and row.get("settings_qc_pass") is False:
            result = {
                "settings_qc_pass": False,
                "analysis_eligible": False,
                "settings_qc_status": "fail",
                "settings_qc_reason": str(row.get("settings_qc_reason") or "acquisition metadata unavailable"),
            }
        elif not included:
            result = {
                "settings_qc_pass": None,
                "analysis_eligible": False,
                "settings_qc_status": "not_applicable",
                "settings_qc_reason": str(row.get("settings_qc_reason") or f"excluded: {row.get('role') or 'non-analysis'} acquisition"),
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
            result = {
                "settings_qc_pass": True,
                "analysis_eligible": included,
                "settings_qc_status": "pass",
                "settings_qc_reason": "not a configured Fucci workflow",
            }
        record = {key: row.get(key) for key in QC_COLUMNS}
        record.update({
            "pipeline_enabled": enabled,
            "pipeline_exclusion_reason": str(row.get("pipeline_exclusion_reason") or ""),
            "analysis_included": included,
            "is_vol10_control": is_vol10,
            "expected_pmt_gain_a": expected.get("pmt_gain_a"),
            "expected_pmt_gain_b": expected.get("pmt_gain_b"),
            "expected_laser_920_power": expected.get("laser_920_power"),
            "expected_laser_1050_power": expected.get("laser_1050_power"),
            **{key: result[key] for key in ("settings_qc_pass", "settings_qc_status", "settings_qc_reason", "analysis_eligible")},
        })
        records.append(record)
    table = pd.DataFrame(records, columns=QC_COLUMNS)
    failed = table.loc[table["settings_qc_status"].eq("fail")] if not table.empty else table
    analysis = table.loc[table["pipeline_enabled"] & table["analysis_included"]] if not table.empty else table
    analysis_failed = analysis.loc[analysis["settings_qc_status"].eq("fail")] if not analysis.empty else analysis
    pipeline_excluded = []
    if not table.empty:
        pipeline_excluded = [
            {"mouse_id": str(row.mouse_id), "reason": str(row.pipeline_exclusion_reason)}
            for row in table.loc[~table["pipeline_enabled"], ["mouse_id", "pipeline_exclusion_reason"]].drop_duplicates().itertuples(index=False)
        ]
    return table, {
        "status": "PASS" if failed.empty else "FAIL",
        "n_sessions": int(len(table)),
        "n_pass": int(table["settings_qc_status"].eq("pass").sum()) if not table.empty else 0,
        "n_fail": int(len(failed)),
        "n_vol10_excluded": n_vol10_excluded,
        "n_not_configured": int(table["settings_qc_status"].eq("not_configured").sum()) if not table.empty else 0,
        "n_analysis_acquisitions": int(len(analysis)),
        "n_analysis_pass": int(analysis["settings_qc_status"].eq("pass").sum()) if not analysis.empty else 0,
        "n_analysis_fail": int(len(analysis_failed)),
        "n_other_catalog_acquisitions": int(len(table) - len(analysis)),
        "pipeline_excluded_mice": pipeline_excluded,
        "numeric_tolerance": tolerance,
        "failed_sessions": [{"session_id": str(row.session_id), "acquisition_id": str(row.acquisition_id), "reason": str(row.settings_qc_reason)} for row in failed.itertuples(index=False)],
    }


def _plot_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unique_plot_values(values: list[float], tolerance: float) -> list[float]:
    return [value for index, value in enumerate(values) if not any(math.isclose(value, prior, rel_tol=tolerance, abs_tol=tolerance) for prior in values[:index])]


def _session_plot_rows(table: pd.DataFrame, *, tolerance: float = 1e-6) -> list[dict[str, Any]]:
    """Collapse canonical acquisition rows into one plotting record per session."""
    rows = table.loc[table["pipeline_enabled"] & table["analysis_included"] & table["mouse_id"].isin(EXPECTED_ACQUISITION_SETTINGS)]
    summaries = []
    for (mouse_id, acquisition_date, session_id), frame in rows.groupby(["mouse_id", "acquisition_date", "session_id"], sort=True):
        expected = EXPECTED_ACQUISITION_SETTINGS[str(mouse_id)]
        summary: dict[str, Any] = {"mouse_id": str(mouse_id), "acquisition_date": str(acquisition_date), "session_id": str(session_id), "issues": []}
        for column, label, expected_key in (("pmt_a_gain", "PMT A", "pmt_gain_a"), ("pmt_b_gain", "PMT B", "pmt_gain_b")):
            values = _unique_plot_values([value for value in (_plot_number(item) for item in frame[column]) if value is not None], tolerance)
            summary[column] = values
            if len(values) > 1:
                summary["issues"].append(f"{label} inconsistent: {'/'.join(f'{value:g}' for value in values)}")
            elif values and not math.isclose(values[0], expected[expected_key], rel_tol=tolerance, abs_tol=tolerance):
                summary["issues"].append(f"{label}={values[0]:g} != {expected[expected_key]:g}")
        for wavelength in (920, 1050):
            selected = frame.loc[pd.to_numeric(frame["laser_nm"], errors="coerce").eq(wavelength)]
            values: list[float] = []
            disagreements: list[tuple[float, float]] = []
            for item in selected.itertuples(index=False):
                start = _plot_number(getattr(item, f"pockels_{wavelength}_start_pct")); stop = _plot_number(getattr(item, f"pockels_{wavelength}_stop_pct"))
                if start is not None: values.append(start)
                if stop is not None and (start is None or not math.isclose(start, stop, rel_tol=tolerance, abs_tol=tolerance)): values.append(stop)
                if start is not None and stop is not None and not math.isclose(start, stop, rel_tol=tolerance, abs_tol=tolerance): disagreements.append((start, stop))
            values = _unique_plot_values(values, tolerance); summary[f"laser_{wavelength}"] = values
            expected_power = expected[f"laser_{wavelength}_power"]
            if disagreements:
                summary["issues"].append(f"{wavelength} start={disagreements[0][0]:g} stop={disagreements[0][1]:g}")
            elif values and not all(math.isclose(value, expected_power, rel_tol=tolerance, abs_tol=tolerance) for value in values):
                summary["issues"].append(f"{wavelength}={'/'.join(f'{value:g}' for value in values)} != {expected_power:g}")
        summaries.append(summary)
    return summaries


def write_acquisition_settings_qc_artifacts(rows: list[dict[str, Any]], output_dir: str | Path, *, tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write the full audit CSV and a compact, analysis-only PNG."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table, summary = acquisition_settings_qc_table(rows, tolerance=tolerance)
    table.to_csv(output / "acquisition_settings_qc.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = _session_plot_rows(table, tolerance=tolerance)
    configured_mice = [mouse for mouse in EXPECTED_ACQUISITION_SETTINGS if mouse in {row["mouse_id"] for row in summaries}]
    if not configured_mice:
        configured_mice = sorted({row["mouse_id"] for row in summaries})
    n_rows = max(len(configured_mice), 1)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15.5, max(3.0, 2.35 * n_rows)), squeeze=False)
    fig.suptitle("Acquisition settings QC — analysis-relevant canonical acquisitions", fontsize=14, fontweight="bold")

    for row_index, mouse in enumerate(configured_mice or ["no analysis acquisitions"]):
        pmt_axis, laser_axis = axes[row_index]
        expected = EXPECTED_ACQUISITION_SETTINGS.get(mouse, {})
        frame = [row for row in summaries if row["mouse_id"] == mouse]
        labels = [row["acquisition_date"][5:] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["acquisition_date"]) else row["session_id"] for row in frame]
        failed_x = [index for index, row in enumerate(frame) if row["issues"]]
        n_fail = len(failed_x)
        n_pass = len(frame) - n_fail
        expected_pmt = expected.get("pmt_gain_a")
        expected_power = expected.get("laser_1050_power")
        pmt_axis.text(-0.17, 0.5, mouse, transform=pmt_axis.transAxes, ha="right", va="center", fontsize=11, fontweight="bold")
        pmt_axis.set_title(
            f"expected PMT {expected.get('pmt_gain_a', '—')}/{expected.get('pmt_gain_b', '—')}, "
            f"power {expected_power if expected_power is not None else '—'} | "
            f"{len(frame)} sessions | {n_pass} PASS / {n_fail} FAIL",
            loc="left", fontsize=9,
        )

        pmt_values: list[float] = [float(expected_pmt)] if expected_pmt is not None else []
        laser_values: list[float] = [float(expected_power)] if expected_power is not None else []
        for column, label, marker, color, offset, expected_key, axis in (("pmt_a_gain", "PMT A", "o", "tab:blue", -0.07, "pmt_gain_a", pmt_axis), ("pmt_b_gain", "PMT B", "s", "tab:orange", 0.07, "pmt_gain_b", pmt_axis), ("laser_920", "920", "o", "tab:blue", -0.07, "laser_920_power", laser_axis), ("laser_1050", "1050", "s", "tab:orange", 0.07, "laser_1050_power", laser_axis)):
            for x, row in enumerate(frame):
                for value in row[column]:
                    axis.scatter([x + offset], [value], marker=marker, color=color, s=28, label=label)
                    if not math.isclose(value, expected[expected_key], rel_tol=tolerance, abs_tol=tolerance): axis.scatter([x + offset], [value], marker="x", color="red", s=70, linewidths=1.8, label="QC mismatch")
                    (pmt_values if axis is pmt_axis else laser_values).append(value)

        for axis, reference, values, ylabel in (
            (pmt_axis, expected_pmt, pmt_values, "gain"),
            (laser_axis, expected_power, laser_values, "%"),
        ):
            if reference is not None:
                axis.axhline(float(reference), color="black", linestyle="--", linewidth=1, label="expected")
            if values:
                low, high = min(values), max(values)
                pad = max((high - low) * 0.2, 0.5)
                axis.set_ylim(low - pad, high + pad)
            for x in failed_x:
                axis.axvspan(x - 0.4, x + 0.4, color="red", alpha=0.08)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.set_xlim(-0.6, max(len(frame) - 0.4, 0.6))
            tick_step = 1 if len(frame) <= 15 else 2 if len(frame) <= 30 else max(3, (len(frame) + 14) // 15)
            tick_positions = sorted(set(range(0, len(frame), tick_step)) | set(failed_x))
            axis.set_xticks(tick_positions)
            axis.set_xticklabels([labels[x] for x in tick_positions], rotation=0, fontsize=8)

        for x, row in enumerate(frame):
            if row["issues"]:
                laser_axis.annotate(
                    "\n".join(row["issues"])[:80],
                    (x, laser_axis.get_ylim()[1]), xytext=(0, -3), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color="red",
                )

    handles, legend_labels = [], []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(handles), fontsize=8)
    fig.text(
        0.5, 0.012,
        f"Analysis acquisitions: {summary['n_analysis_pass']} PASS / {summary['n_analysis_fail']} FAIL; "
        f"other catalog acquisitions retained in CSV: {summary['n_other_catalog_acquisitions']}.",
        ha="center", fontsize=8,
    )
    fig.subplots_adjust(left=0.2, right=0.98, top=0.87, bottom=0.07, hspace=0.75, wspace=0.26)
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
