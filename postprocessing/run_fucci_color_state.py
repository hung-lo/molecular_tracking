#!/usr/bin/env python3
"""Apply a frozen Fucci-Dead reference to one explicit master run."""

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

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, _ROOT / "core", _ROOT / "matching", _ROOT / "plotting", _ROOT / "postprocessing"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fucci_color_state import MODAL_BANDWIDTH_SCALE, fit_session_modal_fits, score_color_state_table
from fucci_run_io import file_sha256, is_filesystem_root, resolve_fucci_master_run
from plotting.fucci_color_state_plots import plot_color_z_distribution, plot_modal_fit_sd_zones


POSTPROCESS_SCHEMA_VERSION = "fucci_color_state_postprocess_v1"


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _read_reference(path: str | Path) -> tuple[dict[str, Any], float]:
    reference_path = Path(path).expanduser().resolve()
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read reference JSON: {reference_path}") from exc
    if reference.get("schema_version") != "fucci_dead_color_reference_v1":
        raise ValueError("Unsupported Fucci-Dead reference schema")
    try:
        value = float(reference["spread"]["reference_robust_sd_log2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Reference is missing spread.reference_robust_sd_log2") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Reference robust SD must be positive and finite")
    return reference, value


def _select_policy(observations: pd.DataFrame, policy: str | None) -> pd.DataFrame:
    if "match_policy" not in observations:
        raise ValueError("Matched observations must contain match_policy")
    available = sorted(observations["match_policy"].dropna().astype(str).unique())
    if policy is None:
        if len(available) != 1:
            raise ValueError(f"Multiple match policies are present; pass --policy from {available}")
        policy = available[0]
    if policy not in available:
        raise ValueError(f"Requested policy {policy!r} is not available; choose from {available}")
    selected = observations.loc[observations["match_policy"].astype(str).eq(policy)].copy()
    duplicate_keys = selected.duplicated(["match_policy", "track_uid", "session_id"], keep=False)
    if duplicate_keys.any():
        raise ValueError("Matched observations contain duplicate (match_policy, track_uid, session_id) rows")
    return selected


def _merge_geometry(observations: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in ("match_policy", "roi_id", "session_index") if key in observations and key in geometry]
    if len(keys) < 2:
        return observations
    flags = geometry[keys + ["geometry_qc_pass"]].drop_duplicates(keys)
    return observations.merge(flags, on=keys, how="left", validate="many_to_one")


def _occupancy(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = ["strong_low", "low_transition", "middle", "high_transition", "strong_high"]
    state_column = "eclipse_state_bin" if "eclipse_state_bin" in scored else "color_state_bin"
    qc_column = "eclipse_state_qc_pass" if "eclipse_state_qc_pass" in scored else "color_state_qc_pass"
    reason_column = "eclipse_state_qc_reason" if "eclipse_state_qc_reason" in scored else "color_state_qc_reason"
    occupancy_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for session_id, group in scored.groupby("session_id", sort=False):
        valid = group.loc[group[qc_column].eq(True)]
        total = len(valid)
        source = group.iloc[0]
        row: dict[str, Any] = {
            "session_id": str(session_id),
            "session_index": source.get("session_index", np.nan),
            "acquisition_date": source.get("acquisition_date"),
            "elapsed_days": source.get("elapsed_days", np.nan),
            "n_valid_observations": int(total),
        }
        for state in bins:
            count = int(valid[state_column].eq(state).sum())
            row[f"n_{state}"] = count
            row[f"pct_{state}"] = float(100 * count / total) if total else np.nan
        occupancy_rows.append(row)
        qc_rows.append({
            "session_id": str(session_id),
            "session_index": source.get("session_index", np.nan),
            "acquisition_date": source.get("acquisition_date"),
            "elapsed_days": source.get("elapsed_days", np.nan),
            "n_total_observations": int(len(group)),
            "n_valid_observations": int(total),
            "n_invalid_observations": int(len(group) - total),
            "invalid_reason_counts": json.dumps(group.loc[~group[qc_column], reason_column].value_counts().to_dict(), sort_keys=True),
        })
    return pd.DataFrame(occupancy_rows), pd.DataFrame(qc_rows)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Color-state output already exists: {path}; use --overwrite")
        if not path.is_dir():
            raise ValueError(f"Color-state output is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def run_color_state(
    run_dir: str | Path,
    reference: str | Path,
    *,
    policy: str | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build normalization and scoring outputs for one master run."""

    source = resolve_fucci_master_run(run_dir, require_matched=True)
    reference_path = Path(reference).expanduser().resolve()
    reference_json, reference_sd = _read_reference(reference_path)
    fit_settings = reference_json.get("fit", {})
    min_fit_rois = int(fit_settings.get("min_fit_rois", 50))
    bandwidth_scale = float(fit_settings.get("bandwidth_scale", MODAL_BANDWIDTH_SCALE))
    native = pd.read_csv(source.session_population_path)
    observations = pd.read_csv(source.matched_observations_path)  # type: ignore[arg-type]
    selected = _select_policy(observations, policy)
    geometry = pd.read_csv(source.geometry_path)  # type: ignore[arg-type]
    selected = _merge_geometry(selected, geometry)
    session_fits = fit_session_modal_fits(
        native,
        min_fit_rois=min_fit_rois,
        modal_bandwidth_scale=bandwidth_scale,
    )
    native_scored = score_color_state_table(native, session_fits, reference_sd)
    scored = score_color_state_table(selected, session_fits, reference_sd)
    occupancy, native_qc = _occupancy(native_scored)
    matched_only_occupancy, matched_qc = _occupancy(scored)
    output = Path(output_dir).expanduser().resolve() if output_dir else source.run_dir / "postprocess" / "fucci_color_state"
    repo_root = Path(__file__).resolve().parent.parent
    home = Path.home()
    if is_filesystem_root(output) or output == home or home.is_relative_to(output):
        raise ValueError(f"Refusing dangerous color-state output directory: {output}")
    if output == repo_root or repo_root.is_relative_to(output):
        raise ValueError(f"Refusing repository or ancestor as color-state output: {output}")
    protected = (source.run_dir, source.extraction_dir, source.run_dir / "postprocess")
    if any(output == path or path.is_relative_to(output) for path in protected):
        raise ValueError("Color-state output cannot equal or contain a protected master-run ancestor")
    _prepare_output(output, overwrite)
    normalization = output / "normalization"
    normalization.mkdir()
    plots = output / "plots"
    plots.mkdir()
    session_fits.to_csv(normalization / "color_state_session_fits.csv", index=False)
    native_scored.to_csv(normalization / "session_population_eclipse_state.csv", index=False)
    scored.to_csv(normalization / "matched_roi_color_state_all_observed.csv", index=False)
    occupancy.to_csv(normalization / "eclipse_state_occupancy_by_session.csv", index=False)
    matched_only_occupancy.to_csv(normalization / "matched_only_eclipse_state_occupancy_by_session.csv", index=False)
    native_qc.to_csv(normalization / "eclipse_state_qc_summary_by_session.csv", index=False)
    matched_qc.to_csv(normalization / "matched_only_eclipse_state_qc_summary_by_session.csv", index=False)
    first_fit = session_fits.iloc[0]
    first_id = str(first_fit["session_id"])
    first_native = native.loc[native["session_id"].astype(str).eq(first_id)]
    plot_modal_fit_sd_zones(first_native, first_fit, reference_sd, plots / "green_red_modal_fit_sd_zones_example.png")
    plot_color_z_distribution(native_scored, plots / "eclipse_z_distribution.png")

    input_hashes = {
        "run_manifest.json": file_sha256(source.run_manifest_path),
        "extraction/run_log.json": file_sha256(source.extraction_run_log_path),
        f"extraction/{source.session_population_path.name}": file_sha256(source.session_population_path),
        f"extraction/{source.matched_observations_path.name}": file_sha256(source.matched_observations_path),
        f"extraction/{source.geometry_path.name}": file_sha256(source.geometry_path),
        f"extraction/{source.track_summary_path.name}": file_sha256(source.track_summary_path),
    }
    log = {
        "postprocess_schema_version": POSTPROCESS_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "package_versions": _package_versions(),
        "mouse_id": source.mouse_id,
        "laser_nm": source.laser_nm,
        "source_master_run_dir": str(source.run_dir),
        "source_master_run_manifest_sha256": input_hashes["run_manifest.json"],
        "source_extraction_run_log_sha256": input_hashes["extraction/run_log.json"],
        "source_input_table_sha256": input_hashes,
        "reference_json_path": str(reference_path),
        "reference_json_sha256": file_sha256(reference_path),
        "reference_schema_version": reference_json["schema_version"],
        "reference_robust_sd_log2": reference_sd,
        "fit": {"method": "linear_gaussian_kernel_modal", "min_fit_rois": min_fit_rois, "bandwidth_scale": bandwidth_scale, "predictor": "red", "response": "green", "optimizer": "Nelder-Mead"},
        "state_thresholds_z": reference_json.get("state_thresholds_z"),
        "policy": str(selected["match_policy"].iloc[0]) if len(selected) else policy,
        "occupancy_basis": "all_valid_native_session_rois",
        "output_paths": {"session_fits": str(normalization / "color_state_session_fits.csv"), "native_state": str(normalization / "session_population_eclipse_state.csv"), "matched_observations": str(normalization / "matched_roi_color_state_all_observed.csv"), "observations": str(normalization / "matched_roi_color_state_all_observed.csv"), "occupancy": str(normalization / "eclipse_state_occupancy_by_session.csv"), "matched_only_occupancy": str(normalization / "matched_only_eclipse_state_occupancy_by_session.csv")},
        "row_counts": {"native": int(len(native)), "native_valid_scored_observations": int(native_scored["eclipse_state_qc_pass"].sum()), "matched_observations": int(len(scored)), "matched_valid_scored_observations": int(scored["eclipse_state_qc_pass"].sum()), "sessions": int(len(session_fits))},
        "warnings": [],
    }
    (normalization / "run_log.json").write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    summary = [
        "# Fucci Color-State Postprocessing",
        "",
        f"- Source master run: `{source.run_dir}`",
        f"- Policy: `{log['policy']}`",
        f"- Dead reference robust SD: `{reference_sd:.8g}` log2 units",
        f"- Valid native observations: `{int(native_scored['eclipse_state_qc_pass'].sum())} / {len(native_scored)}`",
        "- Session backbones use all ratio-valid native ROIs; no target re-scaling was applied.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return log


def parse_args(argv: list[str] | None = None) -> Any:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_color_state(args.run_dir, args.reference, policy=args.policy, output_dir=args.output_dir, overwrite=args.overwrite)
    print(f"output_dir={Path(result['output_paths']['observations']).parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
