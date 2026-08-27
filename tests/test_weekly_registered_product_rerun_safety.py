from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import tifffile

import run_weekly_matched_roi_pipeline as run_weekly
from project_config import load_project_config
from roi_log_ratio_analysis import build_registered_image_lookup
from weekly_registered_product import (
    prepare_weekly_product_workspace,
    publish_staged_weekly_product,
)


def _write_stack(path: Path, values: list[int], shape: tuple[int, int, int] = (1, 1, 3)) -> None:
    stack = np.asarray(values, dtype=np.uint16).reshape(shape)
    tifffile.imwrite(path, stack)


def _make_project(tmp_path: Path):
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
            f'primary_laser_nm=920\n'
            f'optional_laser_nm=1050\n'
            f'pockels_1_laser_nm=920\n'
            f'pockels_2_laser_nm=1050\n'
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


def _write_match_csv(path: Path) -> None:
    path.write_text("cluster_id,week1_roi\n1,1\n", encoding="utf-8")


def _write_project_weekly_product(
    product_dir: Path,
    *,
    duplicate_crop_variants: bool = False,
    incomplete: bool = False,
    inconsistent: bool = False,
) -> None:
    product_dir.mkdir(parents=True, exist_ok=True)
    if duplicate_crop_variants:
        _write_stack(product_dir / "20260511_R_crop_64x64_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260511_R_crop_512x512_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260511_G_crop_64x64_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260511_G_crop_512x512_SyN.tif", [1, 2, 3])
    elif incomplete:
        _write_stack(product_dir / "20260511_R_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260511_G_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260512_R_SyN.tif", [1, 2, 3])
    elif inconsistent:
        _write_stack(product_dir / "20260511_R_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260511_G_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260512_R_SyN.tif", [1, 2, 3, 4], shape=(1, 1, 4))
        _write_stack(product_dir / "20260512_G_SyN.tif", [1, 2, 3, 4], shape=(1, 1, 4))
    else:
        _write_stack(product_dir / "20260511_R_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260511_G_SyN.tif", [1, 2, 3])
        _write_stack(product_dir / "20260512_R_SyN.tif", [4, 5, 6])
        _write_stack(product_dir / "20260512_G_SyN.tif", [4, 5, 6])
        tifffile.imwrite(product_dir / "week1_average_cp_masks.tif", np.asarray([[[1, 2, 3]]], dtype=np.uint16))
        (product_dir / "weekly_product_metadata.json").write_text(
            '{"crop_shape": [1, 3], "published_files": ["20260511_R_SyN.tif"]}',
            encoding="utf-8",
        )


def test_duplicate_selected_registered_candidates_raise_and_name_both_files(tmp_path: Path) -> None:
    product_dir = tmp_path / "weekly_registered"
    _write_project_weekly_product(product_dir, duplicate_crop_variants=True)

    with pytest.raises(ValueError) as excinfo:
        build_registered_image_lookup(product_dir, start_date="20260511", day0_mode="syn")

    message = str(excinfo.value)
    assert "Duplicate registered image candidates" in message
    assert "crop_64x64_SyN.tif" in message
    assert "crop_512x512_SyN.tif" in message


def test_raw_plus_syn_day0_variants_still_work_for_both_modes(tmp_path: Path) -> None:
    product_dir = tmp_path / "weekly_registered"
    product_dir.mkdir()
    _write_stack(product_dir / "20260511_R.tif", [1, 2, 3])
    _write_stack(product_dir / "20260511_G.tif", [4, 5, 6])
    _write_stack(product_dir / "20260511_R_SyN.tif", [7, 8, 9])
    _write_stack(product_dir / "20260511_G_SyN.tif", [10, 11, 12])
    _write_stack(product_dir / "20260512_R_SyN.tif", [13, 14, 15])
    _write_stack(product_dir / "20260512_G_SyN.tif", [16, 17, 18])

    raw_lookup = build_registered_image_lookup(product_dir, start_date="20260511", day0_mode="raw")
    syn_lookup = build_registered_image_lookup(product_dir, start_date="20260511", day0_mode="syn")

    assert raw_lookup[(0, "red")].name == "20260511_R.tif"
    assert raw_lookup[(0, "green")].name == "20260511_G.tif"
    assert syn_lookup[(0, "red")].name == "20260511_R_SyN.tif"
    assert syn_lookup[(0, "green")].name == "20260511_G_SyN.tif"
    assert raw_lookup[(1, "red")].name == "20260512_R_SyN.tif"
    assert syn_lookup[(1, "green")].name == "20260512_G_SyN.tif"


def test_clean_single_crop_weekly_product_validates_successfully(tmp_path: Path) -> None:
    _config_path, _config = _make_project(tmp_path)
    product_dir = tmp_path / "weekly_registered"
    _write_project_weekly_product(product_dir)
    match_csv = tmp_path / "matches.csv"
    _write_match_csv(match_csv)

    match_table = run_weekly.validate_weekly_registered_product(product_dir, match_csv, start_date="20260511")
    assert match_table["cluster_id"].tolist() == [1]


def test_refresh_required_when_published_product_exists(tmp_path: Path) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    weekly_output_dir.mkdir()
    (weekly_output_dir / "20260511_R_SyN.tif").write_text("old", encoding="utf-8")
    (weekly_output_dir / "20260511_G_SyN.tif").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="REFRESH_WEEKLY_PRODUCT=True"):
        prepare_weekly_product_workspace(weekly_output_dir, refresh=False)

    assert (weekly_output_dir / "20260511_R_SyN.tif").exists()
    assert (weekly_output_dir / "20260511_G_SyN.tif").exists()


def test_refresh_removes_only_allowlisted_weekly_files(tmp_path: Path) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    registered_dir = tmp_path / "registered"
    raw_dir = tmp_path / "raw"
    weekly_output_dir.mkdir()
    registered_dir.mkdir()
    raw_dir.mkdir()
    (weekly_output_dir / "20260511_R_SyN.tif").write_text("old", encoding="utf-8")
    (weekly_output_dir / "20260511_G_SyN.tif").write_text("old", encoding="utf-8")
    (weekly_output_dir / "sentinel.txt").write_text("keep", encoding="utf-8")
    (weekly_output_dir / "weekly_product_metadata.json").write_text("{}", encoding="utf-8")
    (registered_dir / "source.tif").write_text("registered", encoding="utf-8")
    (raw_dir / "source.tif").write_text("raw", encoding="utf-8")

    staging_dir = prepare_weekly_product_workspace(weekly_output_dir, refresh=True)

    assert staging_dir.is_dir()
    assert not (weekly_output_dir / "20260511_R_SyN.tif").exists()
    assert not (weekly_output_dir / "20260511_G_SyN.tif").exists()
    assert (weekly_output_dir / "sentinel.txt").exists()
    assert (registered_dir / "source.tif").exists()
    assert (raw_dir / "source.tif").exists()


def test_failed_staged_publish_leaves_previous_product_intact(tmp_path: Path) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    weekly_output_dir.mkdir()
    (weekly_output_dir / "20260511_R_SyN.tif").write_text("published", encoding="utf-8")
    (weekly_output_dir / "20260511_G_SyN.tif").write_text("published", encoding="utf-8")
    staging_dir = weekly_output_dir / ".staging" / "run_failed"
    staging_dir.mkdir(parents=True)
    (staging_dir / "20260511_R_SyN.tif").write_text("staged", encoding="utf-8")

    with pytest.raises(ValueError, match="expected published file set"):
        publish_staged_weekly_product(
            staging_dir,
            weekly_output_dir,
            expected_filenames=["20260511_R_SyN.tif", "20260511_G_SyN.tif", "week1_average_cp_masks.tif"],
            metadata={"crop_shape": [1, 3]},
        )

    assert (weekly_output_dir / "20260511_R_SyN.tif").read_text(encoding="utf-8") == "published"
    assert (weekly_output_dir / "20260511_G_SyN.tif").read_text(encoding="utf-8") == "published"


def test_project_weekly_runner_rejects_invalid_prepared_products_before_output_dir(tmp_path: Path, monkeypatch) -> None:
    config_path, _config = _make_project(tmp_path)
    match_csv = tmp_path / "matches.csv"
    _write_match_csv(match_csv)
    weekly_product = tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "weekly_registered"
    weekly_product.mkdir(parents=True)
    _write_project_weekly_product(weekly_product, duplicate_crop_variants=True)

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
            green_dark=319.0,
            red_dark=534.0,
            epsilon=1.0,
        ),
    )

    with pytest.raises(ValueError, match="Duplicate registered image candidates"):
        run_weekly.main()

    assert not (tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "runs").exists()


def test_project_weekly_runner_rejects_incomplete_prepared_products_before_output_dir(tmp_path: Path, monkeypatch) -> None:
    config_path, _config = _make_project(tmp_path)
    match_csv = tmp_path / "matches.csv"
    _write_match_csv(match_csv)
    weekly_product = tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "weekly_registered"
    weekly_product.mkdir(parents=True)
    _write_project_weekly_product(weekly_product, incomplete=True)

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
            green_dark=319.0,
            red_dark=534.0,
            epsilon=1.0,
        ),
    )

    with pytest.raises(FileNotFoundError, match="Missing registered TIFFs"):
        run_weekly.main()

    assert not (tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "runs").exists()


def test_project_weekly_runner_rejects_dimension_inconsistent_prepared_products_before_output_dir(tmp_path: Path, monkeypatch) -> None:
    config_path, _config = _make_project(tmp_path)
    match_csv = tmp_path / "matches.csv"
    _write_match_csv(match_csv)
    weekly_product = tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "weekly_registered"
    weekly_product.mkdir(parents=True)
    _write_project_weekly_product(weekly_product, inconsistent=True)

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
            green_dark=319.0,
            red_dark=534.0,
            epsilon=1.0,
        ),
    )

    with pytest.raises(ValueError, match="inconsistent TIFF dimensions"):
        run_weekly.main()

    assert not (tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "runs").exists()
