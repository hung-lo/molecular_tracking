#!/usr/bin/env python3
"""Build isolated same-session 920-to-1050 ROI identity maps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "core", ROOT / "matching", ROOT / "plotting"):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)

from affine_overlap_matcher import (
    MATCHER_ALGORITHM_VERSION,
    AffineOverlapParams,
    VoxelSpacing,
)
from cross_laser_roi_mapper import (
    CROSS_LASER_ALGORITHM_VERSION,
    PAIR_GAP_HANDLING,
    CrossLaserSource,
    accepted_pairs_by_source,
    map_cross_laser_source,
    relabel_primary_high_mask,
    resolve_identity_evidence,
    transform_record,
)
from cross_laser_roi_qc import (
    generate_cross_laser_qc,
    render_cross_laser_contact_sheet,
    select_cross_laser_examples,
)
from project_cli import file_sha256
from project_config import ProjectConfig, load_project_config, validate_output_path


@dataclass(frozen=True)
class SessionPair:
    """One canonical mouse/date pair eligible for cross-laser mapping."""

    session_id: str
    acquisition_date: str
    row_1050: dict[str, str]
    row_920: dict[str, str]


def _atomic_csv(path: Path, table: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def _csv_row_count(path: Path) -> int:
    try:
        return int(len(pd.read_csv(path)))
    except pd.errors.EmptyDataError:
        return 0


def _atomic_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def _safe_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _read_catalog(config: ProjectConfig) -> tuple[Path, list[dict[str, str]], dict[str, Any]]:
    path = config.paths.derivatives_root / "_catalog" / "acquisitions.generated.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Acquisition catalog was not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report_path = path.with_name("validation_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    return path.resolve(), rows, report


def _included_rows(
    rows: list[dict[str, str]], mouse_id: str, laser_nm: int
) -> dict[tuple[str, str], dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("mouse_id") != mouse_id:
            continue
        if str(row.get("analysis_included", "")).lower() != "true":
            continue
        if int(row.get("laser_nm") or -1) != int(laser_nm):
            continue
        key = (str(row["session_id"]), str(row["acquisition_date"]))
        grouped.setdefault(key, []).append(row)
    duplicates = {key: values for key, values in grouped.items() if len(values) != 1}
    if duplicates:
        raise ValueError(
            f"Canonical catalog contains duplicate included {laser_nm}-nm acquisitions: "
            f"{sorted(duplicates)}"
        )
    return {key: values[0] for key, values in grouped.items()}


def _request_matches(pair: SessionPair, request: str) -> bool:
    normalized = request.strip()
    return normalized in {
        pair.session_id,
        pair.acquisition_date,
        pair.acquisition_date.replace("-", ""),
    }


def discover_canonical_session_pairs(
    config: ProjectConfig,
    *,
    mouse_id: str,
    requested_sessions: tuple[str, ...] = (),
) -> tuple[list[SessionPair], list[dict[str, str]], Path, dict[str, Any]]:
    """Resolve only catalog-approved 1050/920 pairs; never glob raw folders."""

    catalog, rows, report = _read_catalog(config)
    rows_1050 = _included_rows(rows, mouse_id, config.rig.primary_laser_nm)
    rows_920 = _included_rows(rows, mouse_id, config.rig.optional_laser_nm)
    keys = sorted(set(rows_1050).union(rows_920), key=lambda key: (key[1], key[0]))
    pairs: list[SessionPair] = []
    skipped: list[dict[str, str]] = []
    for key in keys:
        row_1050 = rows_1050.get(key)
        row_920 = rows_920.get(key)
        session_id, acquisition_date = key
        if row_1050 is None:
            skipped.append(
                {
                    "session_id": session_id,
                    "acquisition_date": acquisition_date,
                    "status": "skipped",
                    "reason": "missing_canonical_1050_acquisition",
                }
            )
            continue
        if row_920 is None:
            skipped.append(
                {
                    "session_id": session_id,
                    "acquisition_date": acquisition_date,
                    "status": "skipped",
                    "reason": "missing_canonical_920_acquisition",
                }
            )
            continue
        pair = SessionPair(session_id, acquisition_date, row_1050, row_920)
        base_1050 = _session_laser_dir(config, mouse_id, pair, 1050)
        base_920 = _session_laser_dir(config, mouse_id, pair, 920)
        missing = [
            str(path)
            for path in (
                base_1050 / "segmentation" / "mask.tif",
                base_920 / "segmentation" / "mask.tif",
            )
            if not path.is_file()
        ]
        if missing:
            skipped.append(
                {
                    "session_id": session_id,
                    "acquisition_date": acquisition_date,
                    "status": "skipped",
                    "reason": "missing_required_mask:" + ";".join(missing),
                }
            )
            continue
        pairs.append(pair)

    if requested_sessions:
        requested = tuple(requested_sessions)
        selected = [
            pair for pair in pairs if any(_request_matches(pair, item) for item in requested)
        ]
        missing_requests = [
            item
            for item in requested
            if not any(_request_matches(pair, item) for pair in selected)
        ]
        if missing_requests:
            known_skips = [
                skip
                for skip in skipped
                if any(
                    item in {
                        skip["session_id"],
                        skip["acquisition_date"],
                        skip["acquisition_date"].replace("-", ""),
                    }
                    for item in missing_requests
                )
            ]
            raise ValueError(
                "Requested session(s) do not have canonical 1050/920 acquisitions and "
                f"required masks: {missing_requests}; details={known_skips}"
            )
        pairs = selected
    if not pairs:
        raise ValueError(
            f"No canonical paired 920/1050 sessions with masks are available for {mouse_id}."
        )
    return pairs, skipped, catalog, report


def _session_laser_dir(
    config: ProjectConfig,
    mouse_id: str,
    pair: SessionPair,
    laser_nm: int,
) -> Path:
    return (
        config.paths.derivatives_root
        / mouse_id
        / "sessions"
        / pair.acquisition_date.replace("-", "")
        / str(laser_nm)
    )


def _spacing_from_pair(pair: SessionPair) -> VoxelSpacing:
    """Use catalog-derived spacing and reject geometry disagreement."""

    fields = ("pixel_size_x_um", "pixel_size_y_um", "z_step_um")
    values_1050 = [float(pair.row_1050[field]) for field in fields]
    values_920 = [float(pair.row_920[field]) for field in fields]
    if not np.allclose(values_1050, values_920, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Cross-laser catalog spacing differs for {pair.session_id}: "
            f"1050={values_1050}, 920={values_920}"
        )
    return VoxelSpacing(
        z_um=values_1050[2],
        y_um=values_1050[1],
        x_um=values_1050[0],
    )


def _load_mask(path: Path) -> np.ndarray:
    try:
        return np.asarray(tifffile.memmap(path))
    except Exception:
        return tifffile.imread(path)


def _feature_output(
    features: pd.DataFrame,
    *,
    mouse_id: str,
    pair: SessionPair,
    source: CrossLaserSource,
) -> pd.DataFrame:
    table = features.reset_index(drop=True).copy()
    table["mouse_id"] = mouse_id
    table["session_id"] = pair.session_id
    table["acquisition_date"] = pair.acquisition_date
    table["source"] = source.name
    table["laser_nm"] = source.laser_nm
    table["channel"] = source.channel
    leading = ["mouse_id", "session_id", "acquisition_date", "source", "laser_nm", "channel"]
    return table.loc[:, leading + [column for column in table if column not in leading]]


def _ensure_red_columns(
    fixed_coverage: pd.DataFrame,
    *,
    secondary_status: str,
) -> pd.DataFrame:
    output = fixed_coverage.copy()
    for column in (
        "red_best_candidate_label",
        "red_best_candidate_score",
        "red_best_candidate_dice",
        "red_best_candidate_distance_um",
        "red_high_label_920",
        "red_balanced_label_920",
    ):
        if column not in output:
            output[column] = np.nan
    output["red_status"] = secondary_status
    return output


def _merge_secondary_evidence(
    primary_coverage: pd.DataFrame,
    secondary_coverage: pd.DataFrame | None,
    *,
    secondary_status: str,
) -> pd.DataFrame:
    if secondary_coverage is None:
        return _ensure_red_columns(primary_coverage, secondary_status=secondary_status)
    columns = [
        "label_1050",
        "red_best_candidate_label",
        "red_best_candidate_score",
        "red_best_candidate_dice",
        "red_best_candidate_distance_um",
        "red_high_label_920",
        "red_balanced_label_920",
        "red_status",
    ]
    output = primary_coverage.merge(
        secondary_coverage[columns], on="label_1050", how="left", validate="one_to_one"
    )
    output["red_status"] = output["red_status"].fillna(secondary_status)
    return output


def _session_summary(
    *,
    primary,
    resolution: pd.DataFrame,
    secondary=None,
) -> dict[str, object]:
    fixed = primary.fixed_coverage
    observable = fixed["common_volume_status"].ne("outside_common_volume")
    row: dict[str, object] = {
        "mouse_id": primary.summary["mouse_id"],
        "session_id": primary.summary["session_id"],
        "acquisition_date": primary.summary["acquisition_date"],
        "n_1050_rois": int(len(primary.fixed_features)),
        "n_920_rois": int(len(primary.moving_features)),
        "n_1050_observable": int(observable.sum()),
        "n_high": int(len(primary.high_matches)),
        "n_balanced": int(len(primary.balanced_matches)),
        "high_fraction_all_1050": float(len(primary.high_matches) / len(primary.fixed_features)),
        "high_fraction_observable_1050": float(
            primary.fixed_coverage.loc[observable, "green_high_label_920"].notna().sum()
            / max(int(observable.sum()), 1)
        ),
        "high_fraction_920": float(len(primary.high_matches) / len(primary.moving_features)),
        "balanced_fraction_all_1050": float(
            len(primary.balanced_matches) / len(primary.fixed_features)
        ),
        "balanced_fraction_observable_1050": float(
            primary.fixed_coverage.loc[observable, "green_balanced_label_920"].notna().sum()
            / max(int(observable.sum()), 1)
        ),
        "balanced_fraction_920": float(
            len(primary.balanced_matches) / len(primary.moving_features)
        ),
        "shift_z_planes": primary.summary["shift_z"],
        "shift_y_px": primary.summary["shift_y"],
        "shift_x_px": primary.summary["shift_x"],
        "transform_n_seed": primary.transform.n_seed,
        "transform_n_inlier": primary.transform.n_inlier,
        "transform_residual_median_um": primary.transform.residual_median_um,
        "transform_residual_p95_um": primary.transform.residual_p95_um,
        "median_raw_dz_planes": primary.high_matches["raw_delta_z_planes"].median(),
        "median_aligned_residual_z_um": primary.high_matches[
            "aligned_residual_z_um"
        ].median(),
        "median_aligned_residual_distance_um": primary.high_matches[
            "aligned_residual_distance_um"
        ].median(),
        "n_high_both_sources_same_1050": 0,
        "n_primary_high_only": int(len(primary.high_matches)),
        "n_secondary_high_only": 0,
        "n_source_conflicts": int(resolution["cross_source_conflict"].sum()),
        "n_secondary_rescue_candidates": int(
            resolution["resolved_status"].eq("secondary_high_rescue_candidate").sum()
        ),
    }
    if secondary is not None:
        primary_labels = set(
            primary.high_matches["label_1050"].astype(int).tolist()
        )
        secondary_labels = set(
            secondary.high_matches["label_1050"].astype(int).tolist()
        )
        row["n_high_both_sources_same_1050"] = int(len(primary_labels & secondary_labels))
        row["n_primary_high_only"] = int(len(primary_labels - secondary_labels))
        row["n_secondary_high_only"] = int(len(secondary_labels - primary_labels))
        row["n_920_red_rois"] = int(len(secondary.moving_features))
    return row


def _near_miss_representatives(primary, labels: set[int]) -> pd.DataFrame:
    """Keep one useful balanced or rejected candidate per fixed 1050 ROI."""
    if not labels:
        return primary.candidates.iloc[:0].copy()
    order = ["score", "dice", "distance_um", "label_920"]
    balanced = primary.balanced_matches.loc[primary.balanced_matches["label_1050"].astype(int).isin(labels)]
    balanced = balanced.sort_values(order, ascending=[False, False, True, True]).drop_duplicates("label_1050")
    used = set(balanced["label_1050"].astype(int))
    rejected = primary.candidates.loc[primary.candidates["label_1050"].astype(int).isin(labels - used)]
    rejected = rejected.sort_values(order, ascending=[False, False, True, True]).drop_duplicates("label_1050")
    return pd.concat([balanced, rejected], ignore_index=True, sort=False)


def _default_run_name(pairs: list[SessionPair]) -> str:
    dates = [pair.acquisition_date.replace("-", "") for pair in pairs]
    if len(dates) == 1:
        scope = dates[0]
    else:
        scope = f"{dates[0]}_to_{dates[-1]}_{len(dates)}sessions"
    return f"cross_laser_920_to_1050_{scope}"


def _prepare_run_dir(
    config: ProjectConfig,
    *,
    mouse_id: str,
    run_name: str,
    output_root: str | None,
    overwrite: bool,
) -> Path:
    default_root = (
        config.paths.derivatives_root / mouse_id / "cross_laser" / "920_to_1050" / "runs"
    )
    root = validate_output_path(output_root or default_root, config)
    run_dir = validate_output_path(root / run_name, config)
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Cross-laser run already exists: {run_dir}; choose --run-name or --overwrite."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_cross_laser_roi_map(
    *,
    project_config: str | Path,
    mouse_id: str,
    session_ids: tuple[str, ...] = (),
    use_920_red_secondary: bool = False,
    require_920_red_secondary: bool = False,
    run_name: str | None = None,
    output_root: str | None = None,
    overwrite: bool = False,
    require_qc_success: bool = False,
) -> Path:
    """Run isolated canonical same-session 920-to-1050 mapping."""

    started = datetime.now(timezone.utc)
    config = load_project_config(project_config)
    pairs, skipped, catalog_path, report = discover_canonical_session_pairs(
        config, mouse_id=mouse_id, requested_sessions=session_ids
    )
    run_dir = _prepare_run_dir(
        config,
        mouse_id=mouse_id,
        run_name=run_name or _default_run_name(pairs),
        output_root=output_root,
        overwrite=overwrite,
    )
    affine_params = AffineOverlapParams()
    fixed_source_name = "1050_red"
    primary_source_name = "920_green_primary"
    feature_1050: list[pd.DataFrame] = []
    feature_green: list[pd.DataFrame] = []
    feature_red: list[pd.DataFrame] = []
    green_candidates: list[pd.DataFrame] = []
    green_high: list[pd.DataFrame] = []
    green_balanced: list[pd.DataFrame] = []
    red_candidates: list[pd.DataFrame] = []
    red_high: list[pd.DataFrame] = []
    red_balanced: list[pd.DataFrame] = []
    consistency_candidates: list[pd.DataFrame] = []
    consistency_high: list[pd.DataFrame] = []
    consistency_balanced: list[pd.DataFrame] = []
    accepted_rows: list[pd.DataFrame] = []
    fixed_coverages: list[pd.DataFrame] = []
    moving_green_coverages: list[pd.DataFrame] = []
    moving_red_coverages: list[pd.DataFrame] = []
    resolutions: list[pd.DataFrame] = []
    transforms: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    resolved_sessions: list[dict[str, object]] = []
    source_hashes: list[dict[str, str]] = []
    session_inputs: list[dict[str, object]] = []
    qc_outputs: dict[str, list[str]] = {}
    qc_errors: list[str] = []

    for pair in pairs:
        spacing = _spacing_from_pair(pair)
        dir_1050 = _session_laser_dir(config, mouse_id, pair, 1050)
        dir_920 = _session_laser_dir(config, mouse_id, pair, 920)
        fixed_mask_path = dir_1050 / "segmentation" / "mask.tif"
        green_mask_path = dir_920 / "segmentation" / "mask.tif"
        red_mask_path = dir_920 / "segmentation" / "mask_red.tif"
        fixed_mask = _load_mask(fixed_mask_path)
        green_mask = _load_mask(green_mask_path)
        initial_hashes = {
            str(fixed_mask_path): file_sha256(fixed_mask_path),
            str(green_mask_path): file_sha256(green_mask_path),
        }
        fixed_source = CrossLaserSource(
            fixed_source_name, 1050, "red", str(fixed_mask_path)
        )
        green_source = CrossLaserSource(
            primary_source_name, 920, "green", str(green_mask_path)
        )
        session_inputs.append(
            {
                "session_id": pair.session_id,
                "acquisition_date": pair.acquisition_date,
                "fixed_mask_path": str(fixed_mask_path),
                "fixed_mask_sha256": initial_hashes[str(fixed_mask_path)],
                "moving_green_mask_path": str(green_mask_path),
                "moving_green_mask_sha256": initial_hashes[str(green_mask_path)],
                "fixed_qc_image_path": str(dir_1050 / "preprocessing" / "red.tif"),
                "moving_green_qc_image_path": str(dir_920 / "preprocessing" / "green.tif"),
                "spacing_um": {"z": spacing.z_um, "y": spacing.y_um, "x": spacing.x_um},
            }
        )
        primary = map_cross_laser_source(
            mouse_id=mouse_id,
            session_id=pair.session_id,
            acquisition_date=pair.acquisition_date,
            fixed_mask=fixed_mask,
            moving_mask=green_mask,
            fixed_source=fixed_source,
            moving_source=green_source,
            spacing=spacing,
            affine_params=affine_params,
        )
        secondary = None
        consistency = None
        secondary_status = "not_evaluated"
        if use_920_red_secondary:
            if red_mask_path.is_file():
                secondary_status = "evaluated"
                initial_hashes[str(red_mask_path)] = file_sha256(red_mask_path)
                session_inputs[-1]["moving_red_mask_path"] = str(red_mask_path)
                session_inputs[-1]["moving_red_mask_sha256"] = initial_hashes[str(red_mask_path)]
                session_inputs[-1]["moving_red_qc_image_path"] = str(dir_920 / "preprocessing" / "red.tif")
                red_mask = _load_mask(red_mask_path)
                red_source = CrossLaserSource(
                    "920_red_secondary", 920, "red", str(red_mask_path)
                )
                secondary = map_cross_laser_source(
                    mouse_id=mouse_id,
                    session_id=pair.session_id,
                    acquisition_date=pair.acquisition_date,
                    fixed_mask=fixed_mask,
                    moving_mask=red_mask,
                    fixed_source=fixed_source,
                    moving_source=red_source,
                    spacing=spacing,
                    affine_params=affine_params,
                )
                consistency = map_cross_laser_source(
                    mouse_id=mouse_id,
                    session_id=pair.session_id,
                    acquisition_date=pair.acquisition_date,
                    fixed_mask=green_mask,
                    moving_mask=red_mask,
                    fixed_source=green_source,
                    moving_source=red_source,
                    spacing=spacing,
                    affine_params=affine_params,
                )
            else:
                secondary_status = "missing"
                if require_920_red_secondary:
                    raise FileNotFoundError(
                        f"Required 920-red secondary mask is missing: {red_mask_path}"
                    )
        merged_coverage = _merge_secondary_evidence(
            primary.fixed_coverage,
            secondary.fixed_coverage if secondary is not None else None,
            secondary_status=secondary_status,
        )
        resolution = resolve_identity_evidence(
            merged_coverage,
            secondary_fixed_coverage=secondary.fixed_coverage if secondary else None,
            green_red_high_matches=consistency.high_matches if consistency else None,
            secondary_status=secondary_status,
        )
        # Keep all primary coverage columns and append only resolution fields.
        resolution_fields = [
            "label_1050",
            "primary_green_status",
            "primary_green_label_920",
            "primary_green_confidence",
            "secondary_red_status",
            "secondary_red_label_920",
            "secondary_red_confidence",
            "cross_source_conflict",
            "resolved_status",
            "resolved_920_source",
            "resolved_label_920",
            "recommended_for_identity",
            "provisional_identity",
            "review_required",
        ]
        merged_coverage = merged_coverage.merge(
            resolution[resolution_fields], on="label_1050", how="left", validate="one_to_one"
        )
        feature_1050.append(
            _feature_output(primary.fixed_features, mouse_id=mouse_id, pair=pair, source=fixed_source)
        )
        feature_green.append(
            _feature_output(primary.moving_features, mouse_id=mouse_id, pair=pair, source=green_source)
        )
        green_candidates.append(primary.candidates)
        green_high.append(primary.high_matches)
        green_balanced.append(primary.balanced_matches)
        accepted_rows.append(accepted_pairs_by_source(primary))
        moving_green_coverages.append(primary.moving_coverage)
        transforms.append(transform_record(primary, spacing))
        if secondary is not None:
            feature_red.append(
                _feature_output(secondary.moving_features, mouse_id=mouse_id, pair=pair, source=secondary.source)
            )
            red_candidates.append(secondary.candidates)
            red_high.append(secondary.high_matches)
            red_balanced.append(secondary.balanced_matches)
            accepted_rows.append(accepted_pairs_by_source(secondary))
            moving_red_coverages.append(secondary.moving_coverage)
            transforms.append(transform_record(secondary, spacing))
        if consistency is not None:
            consistency_candidates.append(consistency.candidates)
            consistency_high.append(consistency.high_matches)
            consistency_balanced.append(consistency.balanced_matches)
            transforms.append(transform_record(consistency, spacing))
        fixed_coverages.append(merged_coverage)
        resolutions.append(resolution)
        summaries.append(_session_summary(primary=primary, resolution=resolution, secondary=secondary))
        resolved_sessions.append(
            {
                "mouse_id": mouse_id,
                "session_id": pair.session_id,
                "acquisition_date": pair.acquisition_date,
                "status": "processed",
                "secondary_red_status": secondary_status,
                "fixed_mask_path": str(fixed_mask_path),
                "moving_green_mask_path": str(green_mask_path),
                "moving_red_mask_path": str(red_mask_path) if red_mask_path.is_file() else "",
            }
        )
        relabelled = relabel_primary_high_mask(green_mask, primary.high_matches)
        relabel_path = (
            run_dir
            / "relabelled_masks"
            / f"{pair.session_id}_920_green_native_as_1050_high.tif"
        )
        relabel_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(relabel_path, relabelled)
        fixed_labels = set(np.unique(fixed_mask).astype(int))
        output_labels = set(np.unique(relabelled).astype(int))
        if not output_labels.issubset(fixed_labels):
            raise RuntimeError("Relabelled mask includes labels absent from the 1050 mask.")
        if relabelled.shape != green_mask.shape:
            raise RuntimeError("Relabelled mask shape differs from native 920-green mask.")
        for path, before_hash in initial_hashes.items():
            after_hash = file_sha256(path)
            if after_hash != before_hash:
                raise RuntimeError(f"Source mask was modified during cross-laser mapping: {path}")
            source_hashes.append({"path": path, "sha256": before_hash})
        try:
            qcs = generate_cross_laser_qc(
                output_dir=run_dir / "qc" / pair.session_id,
                fixed_coverage=merged_coverage,
                accepted_pairs=accepted_pairs_by_source(primary),
                image_shape_yx=(fixed_mask.shape[1], fixed_mask.shape[2]),
                identity_resolution=resolution,
            )
            if secondary is not None:
                qcs.update({f"920_red_{name}": path for name, path in generate_cross_laser_qc(
                    output_dir=run_dir / "qc" / pair.session_id / "920_red",
                    fixed_coverage=secondary.fixed_coverage,
                    accepted_pairs=accepted_pairs_by_source(secondary),
                    image_shape_yx=(fixed_mask.shape[1], fixed_mask.shape[2]),
                ).items()})
            image_shape_yx = (fixed_mask.shape[1], fixed_mask.shape[2])
            examples = {
                "primary_green_high": select_cross_laser_examples(
                    resolution.loc[resolution["resolved_status"].eq("primary_high")], image_shape_yx=image_shape_yx
                ),
                "primary_green_near_miss": select_cross_laser_examples(
                    resolution.loc[
                        resolution["primary_green_status"].isin(
                            ["balanced_only", "candidate_but_rejected"]
                        )
                    ], image_shape_yx=image_shape_yx
                ),
                "secondary_red_rescue_candidate": select_cross_laser_examples(
                    resolution.loc[resolution["resolved_status"].eq("secondary_high_rescue_candidate")], image_shape_yx=image_shape_yx
                ),
                "cross_source_conflict": select_cross_laser_examples(
                    resolution.loc[resolution["resolved_status"].eq("cross_source_conflict")], image_shape_yx=image_shape_yx
                ),
            }
            for name, table in examples.items():
                _atomic_csv(run_dir / "qc" / pair.session_id / f"{name}_examples.csv", table)
            fixed_image_path = dir_1050 / "preprocessing" / "red.tif"
            green_image_path = dir_920 / "preprocessing" / "green.tif"
            if not fixed_image_path.is_file() or not green_image_path.is_file():
                raise FileNotFoundError(
                    f"Raw QC images are missing: {fixed_image_path}; {green_image_path}"
                )
            fixed_image = _load_mask(fixed_image_path)
            green_image = _load_mask(green_image_path)
            primary_example_groups = {
                "primary_green_high": primary.high_matches,
                "primary_green_near_miss": _near_miss_representatives(
                    primary,
                    set(resolution.loc[resolution["primary_green_status"].isin(["balanced_only", "candidate_but_rejected"]), "label_1050"].astype(int)),
                ),
            }
            for name, table in primary_example_groups.items():
                if not table.empty:
                    selected = select_cross_laser_examples(table, image_shape_yx=image_shape_yx)
                    path = render_cross_laser_contact_sheet(
                        output_path=run_dir / "qc" / pair.session_id / f"{name}.png",
                        fixed_image=fixed_image,
                        fixed_mask=fixed_mask,
                        moving_image=green_image,
                        moving_mask=green_mask,
                        pairs=selected,
                        title=f"{pair.session_id}: {name}",
                    )
                    qcs[name] = path
            if secondary is not None:
                red_image_path = dir_920 / "preprocessing" / "red.tif"
                if not red_image_path.is_file():
                    raise FileNotFoundError(f"Raw 920-red QC image is missing: {red_image_path}")
                red_image = _load_mask(red_image_path)
                rescue_labels = set(
                    resolution.loc[
                        resolution["resolved_status"].eq("secondary_high_rescue_candidate"),
                        "label_1050",
                    ].astype(int)
                )
                conflict_labels = set(
                    resolution.loc[resolution["cross_source_conflict"], "label_1050"].astype(int)
                )
                for name, labels in {
                    "secondary_red_rescue_candidate": rescue_labels,
                    "cross_source_conflict": conflict_labels,
                }.items():
                    table = secondary.high_matches.loc[
                        secondary.high_matches["label_1050"].astype(int).isin(labels)
                    ]
                    if not table.empty:
                        selected = select_cross_laser_examples(table, image_shape_yx=image_shape_yx)
                        path = render_cross_laser_contact_sheet(
                            output_path=run_dir / "qc" / pair.session_id / f"{name}.png",
                            fixed_image=fixed_image,
                            fixed_mask=fixed_mask,
                            moving_image=red_image,
                            moving_mask=red_mask,
                            pairs=selected,
                            title=f"{pair.session_id}: {name}",
                        )
                        qcs[name] = path
            qc_outputs[pair.session_id] = [str(path) for path in qcs.values()]
        except Exception as exc:
            message = f"{pair.session_id}: {type(exc).__name__}: {exc}"
            qc_errors.append(message)
            if require_qc_success:
                raise

    for skipped_row in skipped:
        resolved_sessions.append({"mouse_id": mouse_id, **skipped_row, "secondary_red_status": ""})
    combined = lambda tables: pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()
    outputs = {
        "resolved_sessions": _atomic_csv(run_dir / "resolved_sessions.csv", pd.DataFrame(resolved_sessions)),
        "roi_features_1050": _atomic_csv(run_dir / "roi_features_1050.csv", combined(feature_1050)),
        "roi_features_920_green": _atomic_csv(run_dir / "roi_features_920_green.csv", combined(feature_green)),
        "candidates_920_green_to_1050": _atomic_csv(run_dir / "candidates_920_green_to_1050.csv", combined(green_candidates)),
        "matches_920_green_to_1050_high": _atomic_csv(run_dir / "matches_920_green_to_1050_high.csv", combined(green_high)),
        "matches_920_green_to_1050_balanced": _atomic_csv(run_dir / "matches_920_green_to_1050_balanced.csv", combined(green_balanced)),
        "transforms": _atomic_csv(run_dir / "transforms.csv", pd.DataFrame(transforms)),
        "roi_map_by_source": _atomic_csv(run_dir / "roi_map_by_source.csv", combined(accepted_rows)),
        "fixed_roi_coverage": _atomic_csv(run_dir / "fixed_roi_coverage.csv", combined(fixed_coverages)),
        "moving_roi_coverage_green": _atomic_csv(run_dir / "moving_roi_coverage_green.csv", combined(moving_green_coverages)),
        "identity_resolution": _atomic_csv(run_dir / "identity_resolution.csv", combined(resolutions)),
        "session_summary": _atomic_csv(run_dir / "session_summary.csv", pd.DataFrame(summaries)),
    }
    if use_920_red_secondary:
        outputs.update(
            {
                "roi_features_920_red": _atomic_csv(run_dir / "roi_features_920_red.csv", combined(feature_red)),
                "candidates_920_red_to_1050": _atomic_csv(run_dir / "candidates_920_red_to_1050.csv", combined(red_candidates)),
                "matches_920_red_to_1050_high": _atomic_csv(run_dir / "matches_920_red_to_1050_high.csv", combined(red_high)),
                "matches_920_red_to_1050_balanced": _atomic_csv(run_dir / "matches_920_red_to_1050_balanced.csv", combined(red_balanced)),
                "candidates_920_red_to_green": _atomic_csv(run_dir / "candidates_920_red_to_green.csv", combined(consistency_candidates)),
                "matches_920_red_to_green_high": _atomic_csv(run_dir / "matches_920_red_to_green_high.csv", combined(consistency_high)),
                "matches_920_red_to_green_balanced": _atomic_csv(run_dir / "matches_920_red_to_green_balanced.csv", combined(consistency_balanced)),
                "moving_roi_coverage_red": _atomic_csv(run_dir / "moving_roi_coverage_red.csv", combined(moving_red_coverages)),
            }
        )
    finished = datetime.now(timezone.utc)
    manifest = {
        "cross_laser_algorithm_version": CROSS_LASER_ALGORITHM_VERSION,
        "affine_matcher_algorithm_version": MATCHER_ALGORITHM_VERSION,
        "git_commit": _git_commit(),
        "mouse_id": mouse_id,
        "processed_sessions": [
            {"session_id": pair.session_id, "acquisition_date": pair.acquisition_date}
            for pair in pairs
        ],
        "skipped_sessions": skipped,
        "fixed_source": {"name": fixed_source_name, "laser_nm": 1050, "channel": "red"},
        "primary_moving_source": {
            "name": primary_source_name,
            "laser_nm": 920,
            "channel": "green",
        },
        "secondary_red_enabled": bool(use_920_red_secondary),
        "project_config_path": str(Path(project_config).resolve()),
        "catalog_path": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "catalog_version": report.get("catalog_version"),
        "mask_hashes": source_hashes,
        "session_inputs": session_inputs,
        "affine_overlap_params": asdict(affine_params),
        "pair_gap_handling": PAIR_GAP_HANDLING,
        "start_utc": started.isoformat(),
        "end_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "package_versions": {
            "python": platform.python_version(),
            "numpy": _safe_package_version("numpy"),
            "pandas": _safe_package_version("pandas"),
            "scipy": _safe_package_version("scipy"),
            "scikit-image": _safe_package_version("scikit-image"),
            "tifffile": _safe_package_version("tifffile"),
        },
        "qc_status": "completed" if not qc_errors else "completed_with_warnings",
        "qc_errors": qc_errors,
        "qc_outputs": qc_outputs,
        "outputs": {
            name: {"path": str(path), "rows": _csv_row_count(path)}
            for name, path in outputs.items()
        },
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    summary = [
        "# Cross-laser 920 to 1050 ROI mapping",
        "",
        f"- Mouse: {mouse_id}",
        f"- Algorithm: {CROSS_LASER_ALGORITHM_VERSION}",
        f"- Fixed canonical source: {fixed_source_name}",
        f"- Primary moving source: {primary_source_name}",
        f"- Secondary 920-red enabled: {use_920_red_secondary}",
        f"- Pair-gap handling: {PAIR_GAP_HANDLING}",
        f"- Processed sessions: {len(pairs)}",
        f"- Skipped sessions: {len(skipped)}",
        f"- QC status: {manifest['qc_status']}",
        "",
        "Primary green high matches are the only recommended identities in v1.",
        "Secondary red high matches are provisional review candidates only.",
        "",
        "## Output locations",
        "",
    ]
    summary.extend(f"- {name}: {path}" for name, path in outputs.items())
    (run_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", required=True)
    parser.add_argument("--mouse-id", required=True)
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--use-920-red-secondary", action="store_true")
    parser.add_argument("--require-920-red-secondary", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-qc-success", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.require_920_red_secondary and not args.use_920_red_secondary:
        raise ValueError("--require-920-red-secondary requires --use-920-red-secondary")
    run_dir = run_cross_laser_roi_map(
        project_config=args.project_config,
        mouse_id=args.mouse_id,
        session_ids=tuple(args.session_id),
        use_920_red_secondary=bool(args.use_920_red_secondary),
        require_920_red_secondary=bool(args.require_920_red_secondary),
        run_name=args.run_name,
        output_root=args.output_root,
        overwrite=bool(args.overwrite),
        require_qc_success=bool(args.require_qc_success),
    )
    print(f"output_dir={run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
