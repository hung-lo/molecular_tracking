"""Build a frozen, equal-mouse Fucci-Dead color-state reference."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from fucci_color_state import MODAL_BANDWIDTH_SCALE, MODAL_METHOD, STATE_THRESHOLDS, fit_session_modal_fits, robust_sd_mad
from fucci_run_io import file_sha256, resolve_fucci_master_run
from plotting.fucci_color_state_plots import plot_dead_reference_residuals, plot_dead_reference_robust_sd


REFERENCE_SCHEMA_VERSION = "fucci_dead_color_reference_v1"


def aggregate_equal_mouse_reference(sessions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Reduce session spreads to one median per mouse, then average mice equally."""

    required = {"mouse_id", "residual_robust_sd_log2"}
    missing = required.difference(sessions.columns)
    if missing:
        raise ValueError(f"Reference session table is missing: {', '.join(sorted(missing))}")
    mice = (
        sessions.groupby("mouse_id", sort=True)["residual_robust_sd_log2"]
        .agg(
            n_sessions="count",
            median_session_robust_sd_log2="median",
            mean_session_robust_sd_log2="mean",
            min_session_robust_sd_log2="min",
            max_session_robust_sd_log2="max",
        )
        .reset_index()
    )
    reference = float(mice["median_session_robust_sd_log2"].mean())
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Dead reference robust SD is invalid or zero")
    return mice, reference


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Reference output already exists: {output_dir}; use --overwrite")
        if not output_dir.is_dir():
            raise ValueError(f"Reference output is not a directory: {output_dir}")
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_reference_output(output_dir: Path, inputs: list[Any]) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    home = Path.home()
    if output_dir == Path("/") or output_dir == home or home.is_relative_to(output_dir):
        raise ValueError(f"Refusing dangerous reference output directory: {output_dir}")
    if output_dir == repo_root or repo_root.is_relative_to(output_dir):
        raise ValueError(f"Refusing repository or ancestor as reference output: {output_dir}")
    for item in inputs:
        if output_dir.is_relative_to(item.run_dir) or item.run_dir.is_relative_to(output_dir):
            raise ValueError(f"Reference output cannot overlap input master run: {output_dir}")


def build_dead_reference(
    reference_run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    min_fit_rois: int = 50,
    modal_bandwidth_scale: float = MODAL_BANDWIDTH_SCALE,
    allow_single_reference_mouse: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit every control session and write one immutable reference bundle."""

    if not reference_run_dirs:
        raise ValueError("At least one --reference-run-dir value is required")
    if len(reference_run_dirs) < 2 and not allow_single_reference_mouse:
        raise ValueError("At least two --reference-run-dir values are required unless allow_single_reference_mouse is enabled")
    inputs = [resolve_fucci_master_run(path, require_matched=False) for path in reference_run_dirs]
    mouse_ids = {item.mouse_id for item in inputs}
    if len(mouse_ids) < 2 and not allow_single_reference_mouse:
        raise ValueError("Reference runs must contain at least two control mice")
    output = Path(output_dir).expanduser().resolve()
    _validate_reference_output(output, inputs)

    session_tables: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    for item in inputs:
        population = pd.read_csv(item.session_population_path)
        fits = fit_session_modal_fits(
            population,
            min_fit_rois=min_fit_rois,
            modal_bandwidth_scale=modal_bandwidth_scale,
        )
        fits["mouse_id"] = item.mouse_id
        fits["run_dir"] = str(item.run_dir)
        fits["run_manifest_sha256"] = file_sha256(item.run_manifest_path)
        fits["extraction_run_log_sha256"] = file_sha256(item.extraction_run_log_path)
        fits["session_population_sha256"] = file_sha256(item.session_population_path)
        session_tables.append(fits)
        run_records.append({
            "run_dir": str(item.run_dir),
            "mouse_id": item.mouse_id,
            "laser_nm": item.laser_nm,
            "session_ids": list(item.session_ids),
            "input_hashes": {
                "run_manifest.json": file_sha256(item.run_manifest_path),
                "extraction/run_log.json": file_sha256(item.extraction_run_log_path),
                f"extraction/{item.session_population_path.name}": file_sha256(item.session_population_path),
            },
        })
    sessions = pd.concat(session_tables, ignore_index=True)
    mice, reference_sd = aggregate_equal_mouse_reference(sessions)

    centered_residuals: list[np.ndarray] = []
    for item in inputs:
        population = pd.read_csv(item.session_population_path)
        fit_table = sessions.loc[sessions["run_dir"].eq(str(item.run_dir))]
        for session_id, group in population.groupby(population["session_id"].astype(str), sort=False):
            fit = fit_table.loc[fit_table["session_id"].astype(str).eq(session_id)].iloc[0]
            valid = group.loc[group["ratio_qc_pass"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
            red = pd.to_numeric(valid["red"], errors="coerce").to_numpy(dtype=float)
            green = pd.to_numeric(valid["green"], errors="coerce").to_numpy(dtype=float)
            predicted = float(fit["modal_intercept"]) + float(fit["modal_slope"]) * red
            keep = np.isfinite(predicted) & (predicted > 0) & np.isfinite(green) & (green > 0)
            residuals = np.log2(green[keep] / predicted[keep])
            centered_residuals.append(residuals - np.median(residuals))
    pooled_centered_sd = robust_sd_mad(np.concatenate(centered_residuals)) if centered_residuals else np.nan

    sessions = sessions.sort_values(["mouse_id", "session_index"]).reset_index(drop=True)
    _prepare_output(output, overwrite)
    sessions.to_csv(output / "fucci_dead_reference_sessions.csv", index=False)
    mice.to_csv(output / "fucci_dead_reference_mice.csv", index=False)
    plot_dead_reference_robust_sd(sessions, output / "plots/dead_reference_robust_sd_by_session.png")
    plot_dead_reference_residuals(sessions, output / "plots/dead_reference_residual_distributions.png")

    reference = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit": {
            "method": MODAL_METHOD,
            "bandwidth_scale": float(modal_bandwidth_scale),
            "min_fit_rois": int(min_fit_rois),
            "predictor": "red",
            "response": "green",
            "fit_population": "all_ratio_valid_native_session_rois",
            "intercept": True,
            "optimizer": "Nelder-Mead",
        },
        "residual": {"formula": "log2(green / predicted_green)", "epsilon": 0},
        "spread": {
            "estimator": "1.4826 * MAD",
            "per_mouse_aggregation": "median_of_session_robust_sd",
            "across_mouse_aggregation": "arithmetic_mean_equal_mouse_weight",
            "reference_robust_sd_log2": reference_sd,
            "median_all_session_robust_sd_log2": float(sessions["residual_robust_sd_log2"].median()),
            "mean_all_session_robust_sd_log2": float(sessions["residual_robust_sd_log2"].mean()),
            "pooled_session_centered_residual_robust_sd_log2": float(pooled_centered_sd),
            "n_reference_mice": int(len(mice)),
            "n_reference_sessions": int(len(sessions)),
        },
        "state_thresholds_z": STATE_THRESHOLDS,
        "reference_runs": run_records,
        "input_hashes": {record["run_dir"]: record["input_hashes"] for record in run_records},
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "package_versions": _package_versions(),
    }
    (output / "fucci_dead_color_reference.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = [
        "# Fucci-Dead Color Reference",
        "",
        f"- Schema: `{REFERENCE_SCHEMA_VERSION}`",
        f"- Reference robust SD: `{reference_sd:.8g}` log2 units",
        f"- Reference mice: `{', '.join(sorted(mouse_ids))}`",
        f"- Sessions: `{len(sessions)}`",
        "- Fit population: all ratio-valid native session ROIs.",
        "- Target scores must use this frozen denominator without target re-scaling.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return reference
