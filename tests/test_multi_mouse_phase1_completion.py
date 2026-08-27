from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import pytest

import run_920_two_day_cp3nuclei_analysis as run_920
import run_weekly_matched_roi_pipeline as run_weekly
from dataset_catalog import _classification, build_manifest_plan
from notebook_project_setup import resolve_notebook_project_paths
from project_cli import ready_manifest_path, resolve_processed_dataset, resolve_selection
from project_config import load_project_config
from thorimage_xml import parse_experiment_xml

FIX = Path(__file__).parent / "fixtures" / "thorimage"


def make_project(
    tmp_path: Path,
    *,
    primary_laser_nm: int = 1111,
    optional_laser_nm: int = 888,
    pockels_1_laser_nm: int = 888,
    pockels_2_laser_nm: int = 1111,
):
    raw = tmp_path / "raw"
    derivatives = tmp_path / "derivatives"
    raw.mkdir()
    derivatives.mkdir()
    mice = tmp_path / "mice.csv"
    mice.write_text(
        "mouse_id,experimental_group,cohort,viral_constructs,raw_mouse_folder,reference_session_or_folder\n"
        "m,g,c,v,folder,\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "project.toml"
    config_path.write_text(
        (
            f'[paths]\n'
            f'raw_root="{raw}"\n'
            f'derivatives_root="{derivatives}"\n'
            f'mice_csv="{mice}"\n'
            f'[rig]\n'
            f'primary_laser_nm={primary_laser_nm}\n'
            f'optional_laser_nm={optional_laser_nm}\n'
            f'pockels_1_laser_nm={pockels_1_laser_nm}\n'
            f'pockels_2_laser_nm={pockels_2_laser_nm}\n'
            f'chan_a_signal="green"\n'
            f'chan_b_signal="red"\n'
            f'[canonical_volume]\n'
            f'imaging_planes=41\n'
            f'flyback_planes=1\n'
            f'z_step_um=5.0\n'
            f'volumes=50\n'
        ),
        encoding="utf-8",
    )
    return config_path, load_project_config(config_path)


def row(session="session_20260820", acquisition="canonical", laser=1111):
    return {
        "mouse_id": "m",
        "session_id": session,
        "acquisition_date": "2026-08-20",
        "acquisition_id": acquisition,
        "source_path": f"/raw/{session}/{acquisition}",
        "analysis_included": True,
        "laser_nm": laser,
    }


def _write_catalog(config, rows):
    catalog_dir = config.paths.derivatives_root / "_catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog = catalog_dir / "acquisitions.generated.csv"
    fieldnames = list(rows[0])
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return catalog


def _make_920_project(tmp_path: Path):
    config_path, config = make_project(
        tmp_path,
        primary_laser_nm=920,
        optional_laser_nm=1050,
        pockels_1_laser_nm=920,
        pockels_2_laser_nm=1050,
    )
    _write_catalog(
        config,
        [
            {
                "mouse_id": "m",
                "session_id": "session_20260820",
                "acquisition_date": "2026-08-20",
                "acquisition_id": "canonical_920",
                "source_path": "/raw/session_20260820/canonical_920",
                "analysis_included": True,
                "laser_nm": 920,
                "pixel_size_x_um": "0.73",
                "pixel_size_y_um": "0.73",
                "z_step_um": "4.5",
            }
        ],
    )
    return config_path, config


def test_legacy_selection_ignores_forced_wrapper_laser(tmp_path):
    legacy = tmp_path / "legacy920"
    legacy.mkdir()
    context = resolve_selection(dataset=legacy, laser_nm=920)
    assert context.mode == "legacy"
    assert context.dataset_dir == legacy.resolve()
    assert context.laser_nm == 920


def test_project_selection_validates_mouse(tmp_path):
    config_path, _ = make_project(tmp_path)
    with pytest.raises(ValueError, match="Unknown mouse_id"):
        resolve_selection(project_config=config_path, mouse_id="wrong")


def test_config_driven_wavelength_and_vol10_metadata(tmp_path):
    _, config = make_project(tmp_path)
    meta = parse_experiment_xml(FIX / "rectangular_920.xml")
    role, included, laser, warnings = _classification("filed_vol10_laser888", meta, config)
    assert (role, included, laser) == ("alignment_only", False, 888)
    assert not warnings


def test_duplicate_acquisition_rejected_before_manifest_write(tmp_path):
    _, config = make_project(tmp_path)
    with pytest.raises(ValueError, match="Duplicate included acquisitions"):
        build_manifest_plan(config, [row(acquisition="a"), row(acquisition="b")], "m", 1111)
    assert not (config.paths.derivatives_root / "m").exists()


def test_stale_ready_manifest_rejected_when_required_file_disappears(tmp_path):
    config_path, config = make_project(tmp_path)
    catalog_dir = config.paths.derivatives_root / "_catalog"
    catalog_dir.mkdir()
    catalog = catalog_dir / "acquisitions.generated.csv"
    full = row()
    full.update({"pixel_size_x_um": "1", "pixel_size_y_um": "1", "z_step_um": "5", "experimental_group": "g", "cohort": "c"})
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(full))
        writer.writeheader()
        writer.writerow(full)
    base = config.paths.derivatives_root / "m" / "sessions" / "20260820" / "1111"
    files = [base / "segmentation" / "mask.tif", base / "preprocessing" / "red.tif", base / "preprocessing" / "green.tif"]
    for file_path in files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"x")
    manifest, ready = build_manifest_plan(config, [full], "m", 1111, source_catalog=catalog)
    assert ready
    context = resolve_selection(project_config=config_path, mouse_id="m")
    assert ready_manifest_path(context) == manifest
    files[0].unlink()
    with pytest.raises(FileNotFoundError, match="Stale ready manifest"):
        ready_manifest_path(context)


def test_notebook_project_paths_keep_outputs_under_derivatives(tmp_path):
    config_path, config = _make_920_project(tmp_path)
    paths = resolve_notebook_project_paths(config_path, "m", 920)

    expected_dataset_dir = config.paths.derivatives_root / "m" / "longitudinal" / "920"
    assert paths.preprocessing_dir == expected_dataset_dir / "registered"
    assert paths.registration_dir == expected_dataset_dir / "registered"
    assert paths.segmentation_dir == expected_dataset_dir / "registered"
    assert paths.registered_product_dir == expected_dataset_dir / "registered"
    assert paths.weekly_registered_product_dir == expected_dataset_dir / "weekly_registered"
    assert paths.sessions_dir == config.paths.derivatives_root / "m" / "sessions"

    for path in (
        paths.preprocessing_dir,
        paths.registration_dir,
        paths.segmentation_dir,
        paths.registered_product_dir,
        paths.weekly_registered_product_dir,
        paths.sessions_dir,
    ):
        assert not path.is_relative_to(config.paths.raw_root)
        assert path.is_relative_to(config.paths.derivatives_root)


def test_weekly_registered_product_resolves_exactly(tmp_path):
    config_path, _ = _make_920_project(tmp_path)
    context = resolve_selection(project_config=config_path, mouse_id="m", laser_nm=920)
    weekly_product = context.dataset_dir / "weekly_registered"
    weekly_product.mkdir(parents=True)
    (weekly_product / "week1_average_cp_masks.tif").write_bytes(b"synthetic")

    assert resolve_processed_dataset(context, product_name="weekly_registered") == weekly_product.resolve()


def test_920_wrapper_project_mode_uses_project_run_root(tmp_path, monkeypatch):
    config_path, config = _make_920_project(tmp_path)
    registered_dir = config.paths.derivatives_root / "m" / "longitudinal" / "920" / "registered"
    registered_dir.mkdir(parents=True)
    captured = {}

    def fake_run_registered_roi_pipeline(config):
        captured["config"] = config
        return tmp_path / "fake-output"

    monkeypatch.setattr(run_920, "run_registered_roi_pipeline", fake_run_registered_roi_pipeline)
    monkeypatch.setattr(
        run_920,
        "parse_args",
        lambda argv=None: argparse.Namespace(
            dataset=None,
            project_config=config_path,
            mouse_id="m",
            laser_nm=None,
            start_date="20260511",
            mask_name="mask.tif",
            green_dark=1.0,
            red_dark=2.0,
            xy_um_per_px=3.0,
            z_um_per_plane=4.0,
            max_top_rois=5,
            inverse_mask_suffix="_ROI_mask_SyN_inversed.tif",
            inverse_mask_channel="auto",
            raw_space_half_window_z=5,
            enable_raw_space_validation=False,
            skip_raw_space_validation=False,
        ),
    )

    run_920.main()

    captured_config = captured["config"]
    assert captured_config.dataset == str(registered_dir.resolve())
    assert captured_config.output_root == str(config.paths.derivatives_root / "m" / "longitudinal" / "920" / "runs")
    assert captured_config.xy_um_per_px == 0.73
    assert captured_config.z_um_per_plane == 4.5


def test_920_wrapper_legacy_dataset_keeps_output_root_none(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy920"
    legacy.mkdir()
    captured = {}

    def fake_run_registered_roi_pipeline(config):
        captured["config"] = config
        return legacy / "fake-output"

    monkeypatch.setattr(run_920, "run_registered_roi_pipeline", fake_run_registered_roi_pipeline)
    monkeypatch.setattr(
        run_920,
        "parse_args",
        lambda argv=None: argparse.Namespace(
            dataset=legacy,
            project_config=None,
            mouse_id=None,
            laser_nm=920,
            start_date="20260511",
            mask_name="mask.tif",
            green_dark=1.0,
            red_dark=2.0,
            xy_um_per_px=3.0,
            z_um_per_plane=4.0,
            max_top_rois=5,
            inverse_mask_suffix="_ROI_mask_SyN_inversed.tif",
            inverse_mask_channel="auto",
            raw_space_half_window_z=5,
            enable_raw_space_validation=False,
            skip_raw_space_validation=False,
        ),
    )

    run_920.main()

    captured_config = captured["config"]
    assert captured_config.dataset == str(legacy.resolve())
    assert captured_config.output_root is None


def test_weekly_runner_project_mode_uses_weekly_registered_and_runs_root(tmp_path, monkeypatch):
    config_path, config = _make_920_project(tmp_path)
    weekly_registered = config.paths.derivatives_root / "m" / "longitudinal" / "920" / "weekly_registered"
    weekly_registered.mkdir(parents=True)
    match_csv = tmp_path / "matched_tracks.csv"
    match_csv.write_text("cluster_id,week1_roi\n1,1\n", encoding="utf-8")
    captured = {}

    def fake_run_weekly_matched_roi_pipeline(config):
        captured["config"] = config
        return tmp_path / "fake-output"

    monkeypatch.setattr(run_weekly, "run_weekly_matched_roi_pipeline", fake_run_weekly_matched_roi_pipeline)
    monkeypatch.setattr(
        run_weekly,
        "parse_args",
        lambda argv=None: argparse.Namespace(
            dataset=None,
            project_config=config_path,
            mouse_id="m",
            laser_nm=920,
            match_csv=match_csv,
            start_date=None,
            week_mask_template="{week_name}_average_cp_masks.tif",
            green_dark=1.0,
            red_dark=2.0,
            epsilon=1.0,
        ),
    )

    run_weekly.main()

    captured_config = captured["config"]
    assert captured_config.dataset == str(weekly_registered.resolve())
    assert captured_config.output_root == str(config.paths.derivatives_root / "m" / "longitudinal" / "920" / "runs")


def test_weekly_notebook_uses_registered_source_and_weekly_registered_output():
    notebook = json.loads((Path(__file__).resolve().parent.parent / "notebooks" / "weeklyRegister_20260531.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")

    assert "registered_input_dir = project_paths.registered_product_dir.as_posix()" in code
    assert "build_expected_weekly_product_filenames" in code
    assert "build_weekly_product_metadata" in code
    assert "stable_registered_stem" in code
    assert "weekly_stage_dir = prepare_weekly_product_workspace" in code
    assert "REFRESH_WEEKLY_PRODUCT = False" in code
    assert "replace_existing=REFRESH_WEEKLY_PRODUCT" in code
    assert "cropping_enabled = DO_CROP_IMAGE" in code
    assert "publish_staged_weekly_product" in code
    assert "project_paths.preprocessing_dir.as_posix()" not in code
