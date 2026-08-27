from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import tifffile

import run_weekly_matched_roi_pipeline as run_weekly
import weekly_registered_product as weekly_product
from project_config import load_project_config
from roi_log_ratio_analysis import build_registered_image_lookup
from weekly_registered_product import (
    build_expected_weekly_product_filenames,
    build_weekly_product_metadata,
    derive_reference_week_name_from_medoid_image,
    prepare_weekly_product_workspace,
    publish_staged_weekly_product,
    stable_registered_stem,
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


def _build_week1_stage(
    staging_dir: Path,
    *,
    day0_date: str,
    day1_date: str,
    crop_label: str | None,
    values_offset: int = 0,
    include_crossweek_outputs: bool = True,
) -> tuple[dict[str, list[str]], set[str]]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_crop_{crop_label}" if crop_label else ""

    day0_week2 = (datetime.strptime(day0_date, "%Y%m%d") + timedelta(days=7)).strftime("%Y%m%d")
    day1_week2 = (datetime.strptime(day1_date, "%Y%m%d") + timedelta(days=7)).strftime("%Y%m%d")

    week1_source_file_names = [
        f"{day0_date}_R{suffix}.tif",
        f"{day0_date}_G{suffix}.tif",
        f"{day1_date}_R{suffix}.tif",
        f"{day1_date}_G{suffix}.tif",
    ]
    week2_source_file_names = [
        f"{day0_week2}_R{suffix}.tif",
        f"{day0_week2}_G{suffix}.tif",
        f"{day1_week2}_R{suffix}.tif",
        f"{day1_week2}_G{suffix}.tif",
    ]
    source_file_names = week1_source_file_names + week2_source_file_names
    stage_file_names = [f"{stable_registered_stem(name)}_SyN.tif" for name in source_file_names]
    for index, file_name in enumerate(stage_file_names):
        _write_stack(
            staging_dir / file_name,
            [values_offset + index * 3 + 1, values_offset + index * 3 + 2, values_offset + index * 3 + 3],
        )

    _write_stack(staging_dir / "week1_average.tif", [values_offset + 100, values_offset + 101, values_offset + 102])
    _write_stack(
        staging_dir / "week1_average_cp_masks.tif",
        [values_offset + 200, values_offset + 201, values_offset + 202],
    )
    if include_crossweek_outputs:
        _write_stack(
            staging_dir / "week1_average_SyN.tif",
            [values_offset + 300, values_offset + 301, values_offset + 302],
        )
        _write_stack(
            staging_dir / "week1_average_cp_masks_SyN.tif",
            [values_offset + 400, values_offset + 401, values_offset + 402],
        )

    _write_stack(staging_dir / "week2_average.tif", [values_offset + 500, values_offset + 501, values_offset + 502])
    _write_stack(
        staging_dir / "week2_average_cp_masks.tif",
        [values_offset + 600, values_offset + 601, values_offset + 602],
    )

    week_dict = {
        "week1": [str(staging_dir / file_name) for file_name in week1_source_file_names],
        "week2": [str(staging_dir / file_name) for file_name in week2_source_file_names],
    }
    expected_names = build_expected_weekly_product_filenames(week_dict, ["week1", "week2"], reference_week_name="week2")
    return week_dict, expected_names


def _build_three_week_stage(
    staging_dir: Path,
    *,
    crop_label: str | None,
    reference_week_name: str = "week1",
    values_offset: int = 0,
) -> tuple[dict[str, list[str]], set[str]]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_crop_{crop_label}" if crop_label else ""
    week_specs = {
        "week1": ("20260511", "20260512"),
        "week2": ("20260518", "20260519"),
        "week3": ("20260525", "20260526"),
    }
    week_dict: dict[str, list[str]] = {}
    for week_index, (week_name, (day0_date, day1_date)) in enumerate(week_specs.items(), start=1):
        source_file_names = [
            f"{day0_date}_R{suffix}.tif",
            f"{day0_date}_G{suffix}.tif",
            f"{day1_date}_R{suffix}.tif",
            f"{day1_date}_G{suffix}.tif",
        ]
        week_dict[week_name] = [str(staging_dir / file_name) for file_name in source_file_names]
        stage_file_names = [f"{stable_registered_stem(name)}_SyN.tif" for name in source_file_names]
        for index, file_name in enumerate(stage_file_names):
            _write_stack(
                staging_dir / file_name,
                [
                    values_offset + week_index * 100 + index * 3 + 1,
                    values_offset + week_index * 100 + index * 3 + 2,
                    values_offset + week_index * 100 + index * 3 + 3,
                ],
            )
        _write_stack(
            staging_dir / f"{week_name}_average.tif",
            [values_offset + week_index * 1000 + 1, values_offset + week_index * 1000 + 2, values_offset + week_index * 1000 + 3],
        )
        _write_stack(
            staging_dir / f"{week_name}_average_cp_masks.tif",
            [values_offset + week_index * 1000 + 11, values_offset + week_index * 1000 + 12, values_offset + week_index * 1000 + 13],
        )
        if week_name != reference_week_name:
            _write_stack(
                staging_dir / f"{week_name}_average_SyN.tif",
                [values_offset + week_index * 1000 + 21, values_offset + week_index * 1000 + 22, values_offset + week_index * 1000 + 23],
            )
            _write_stack(
                staging_dir / f"{week_name}_average_cp_masks_SyN.tif",
                [values_offset + week_index * 1000 + 31, values_offset + week_index * 1000 + 32, values_offset + week_index * 1000 + 33],
            )
    expected_names = build_expected_weekly_product_filenames(
        week_dict,
        ["week1", "week2", "week3"],
        reference_week_name=reference_week_name,
    )
    return week_dict, expected_names


def _publish_weekly_product(
    weekly_output_dir: Path,
    *,
    registered_input_dir: Path,
    day0_date: str,
    day1_date: str,
    crop_label: str | None,
    cropping_enabled: bool,
    values_offset: int = 0,
    prepare_refresh: bool = False,
    replace_existing: bool = False,
) -> Path:
    staging_dir = prepare_weekly_product_workspace(weekly_output_dir, refresh=prepare_refresh)
    week_dict, expected_names = _build_week1_stage(
        staging_dir,
        day0_date=day0_date,
        day1_date=day1_date,
        crop_label=crop_label,
        values_offset=values_offset,
    )
    metadata = build_weekly_product_metadata(
        crop_shape=None if not cropping_enabled else [1, 3],
        cropping_enabled=cropping_enabled,
        crop_label=crop_label,
        refresh_requested=replace_existing,
        registered_input_dir=registered_input_dir.as_posix(),
        source_registered_files=[
            Path(path).name
            for paths in week_dict.values()
            for path in paths
        ],
        spacing_zyx=(5.0, 0.69, 0.69),
        staging_dir=staging_dir,
        weekly_output_dir=weekly_output_dir,
        week_names=["week1", "week2"],
        reference_week_name="week2",
        published_at="2026-08-27T00:00:00",
    )
    return publish_staged_weekly_product(
        staging_dir,
        weekly_output_dir,
        expected_filenames=sorted(expected_names),
        metadata_filename="weekly_product_metadata.json",
        replace_existing=replace_existing,
        metadata=metadata,
    )


def _snapshot_published_product(weekly_output_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in weekly_product.list_published_weekly_product_files(weekly_output_dir)
    }


def _write_siblings(tmp_path: Path) -> dict[str, Path]:
    registered_dir = tmp_path / "registered"
    raw_root = tmp_path / "raw"
    other_mouse = tmp_path / "derivatives" / "other_mouse" / "longitudinal" / "920" / "weekly_registered"
    registered_dir.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    other_mouse.mkdir(parents=True)
    (registered_dir / "sentinel.txt").write_text("registered", encoding="utf-8")
    (raw_root / "sentinel.txt").write_text("raw", encoding="utf-8")
    (other_mouse / "sentinel.txt").write_text("other", encoding="utf-8")
    return {"registered": registered_dir, "raw": raw_root, "other_mouse": other_mouse}


def test_stable_registered_stem_and_expected_filenames_handle_cropped_and_uncropped_names(tmp_path: Path) -> None:
    assert stable_registered_stem("20260511_R.tif") == "20260511_R"
    assert stable_registered_stem("20260511_R_crop_256x128.tif") == "20260511_R"
    assert stable_registered_stem("20260511_G_crop_256x128_SyN.tif") == "20260511_G"

    week_dict = {
        "week1": [
            str(tmp_path / "20260511_R_crop_256x128.tif"),
            str(tmp_path / "20260511_G_crop_256x128.tif"),
            str(tmp_path / "20260512_R_crop_256x128.tif"),
            str(tmp_path / "20260512_G_crop_256x128.tif"),
        ],
        "week2": [
            str(tmp_path / "20260518_R_crop_256x128.tif"),
            str(tmp_path / "20260518_G_crop_256x128.tif"),
            str(tmp_path / "20260519_R_crop_256x128.tif"),
            str(tmp_path / "20260519_G_crop_256x128.tif"),
        ],
        "week3": [
            str(tmp_path / "20260525_R_crop_256x128.tif"),
            str(tmp_path / "20260525_G_crop_256x128.tif"),
            str(tmp_path / "20260526_R_crop_256x128.tif"),
            str(tmp_path / "20260526_G_crop_256x128.tif"),
        ],
    }
    expected = build_expected_weekly_product_filenames(week_dict, ["week1", "week2", "week3"], reference_week_name="week1")
    assert {
        "20260511_R_SyN.tif",
        "20260511_G_SyN.tif",
        "20260512_R_SyN.tif",
        "20260512_G_SyN.tif",
        "week1_average.tif",
        "week1_average_cp_masks.tif",
        "week2_average.tif",
        "week2_average_cp_masks.tif",
        "week2_average_SyN.tif",
        "week2_average_cp_masks_SyN.tif",
        "week3_average.tif",
        "week3_average_cp_masks.tif",
        "week3_average_SyN.tif",
        "week3_average_cp_masks_SyN.tif",
    }.issubset(expected)
    assert "week1_average_SyN.tif" not in expected
    assert "week1_average_cp_masks_SyN.tif" not in expected

    raw_week_dict = {"week1": [str(tmp_path / "20260511_R.tif"), str(tmp_path / "20260511_G.tif")]}
    raw_expected = build_expected_weekly_product_filenames(raw_week_dict, ["week1"])
    assert "20260511_R_SyN.tif" in raw_expected
    assert "20260511_G_SyN.tif" in raw_expected

    with pytest.raises(ValueError, match="reference_week_name is required"):
        build_expected_weekly_product_filenames(week_dict, ["week1", "week2", "week3"])


def test_reference_week_name_helper_requires_exact_week_average_name() -> None:
    assert derive_reference_week_name_from_medoid_image("week3_average.tif") == "week3"

    for bad_name in (
        "week3_average_SyN.tif",
        "week3_average_cp_masks.tif",
        "week3.tif",
        "medoid_average.tif",
        "weekX_average.tif",
    ):
        with pytest.raises(ValueError, match="Medoid image"):
            derive_reference_week_name_from_medoid_image(bad_name)

    week_dict = {
        "week1": [str(Path("week1_R.tif")), str(Path("week1_G.tif"))],
        "week2": [str(Path("week2_R.tif")), str(Path("week2_G.tif"))],
        "week3": [str(Path("week3_R.tif")), str(Path("week3_G.tif"))],
    }
    with pytest.raises(ValueError, match="reference_week_name is required"):
        build_expected_weekly_product_filenames(week_dict, ["week1", "week2", "week3"])


def test_uncropped_published_names_are_recognized_and_validate_without_crop_shape_mismatch(tmp_path: Path) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    metadata_path = _publish_weekly_product(
        weekly_output_dir,
        registered_input_dir=tmp_path / "registered",
        day0_date="20260511",
        day1_date="20260512",
        crop_label=None,
        cropping_enabled=False,
        prepare_refresh=False,
        replace_existing=False,
    )

    lookup = build_registered_image_lookup(weekly_output_dir, start_date="20260511", day0_mode="syn")
    assert lookup[(0, "red")].name == "20260511_R_SyN.tif"
    assert lookup[(0, "green")].name == "20260511_G_SyN.tif"
    assert lookup[(1, "red")].name == "20260512_R_SyN.tif"
    assert lookup[(1, "green")].name == "20260512_G_SyN.tif"

    match_csv = tmp_path / "matches.csv"
    _write_match_csv(match_csv)
    match_table = run_weekly.validate_weekly_registered_product(
        weekly_output_dir,
        match_csv,
        start_date="20260511",
        require_metadata=True,
    )
    assert match_table["cluster_id"].tolist() == [1]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["crop_shape"] is None
    assert metadata["cropping_enabled"] is False
    expected_published_files = [
        "20260511_G_SyN.tif",
        "20260511_R_SyN.tif",
        "20260512_G_SyN.tif",
        "20260512_R_SyN.tif",
        "20260518_G_SyN.tif",
        "20260518_R_SyN.tif",
        "20260519_G_SyN.tif",
        "20260519_R_SyN.tif",
        "week1_average.tif",
        "week1_average_SyN.tif",
        "week1_average_cp_masks.tif",
        "week1_average_cp_masks_SyN.tif",
        "week2_average.tif",
        "week2_average_cp_masks.tif",
    ]
    validated = weekly_product.validate_published_weekly_product(weekly_output_dir, require_metadata=True)
    assert validated is not None
    assert validated["published_files"] == expected_published_files
    assert sorted(validated["published_sha256"]) == expected_published_files
    assert metadata["published_files"] == expected_published_files
    assert sorted(metadata["published_sha256"]) == expected_published_files
    assert metadata["published_sha256"]["20260511_R_SyN.tif"]


def test_non_last_reference_week_publishes_and_validates(tmp_path: Path) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    staging_dir = prepare_weekly_product_workspace(weekly_output_dir, refresh=False)
    week_dict, expected_names = _build_three_week_stage(
        staging_dir,
        crop_label="256x128",
        reference_week_name="week1",
        values_offset=700,
    )
    registered_input_dir = tmp_path / "registered"
    registered_input_dir.mkdir()
    metadata = build_weekly_product_metadata(
        crop_shape=[1, 3],
        cropping_enabled=True,
        crop_label="256x128",
        refresh_requested=False,
        registered_input_dir=registered_input_dir.as_posix(),
        source_registered_files=[Path(path).name for paths in week_dict.values() for path in paths],
        spacing_zyx=(5.0, 0.69, 0.69),
        staging_dir=staging_dir,
        weekly_output_dir=weekly_output_dir,
        week_names=["week1", "week2", "week3"],
        reference_week_name="week1",
        published_at="2026-08-27T00:00:00",
    )

    metadata_path = publish_staged_weekly_product(
        staging_dir,
        weekly_output_dir,
        expected_filenames=sorted(expected_names),
        metadata_filename="weekly_product_metadata.json",
        replace_existing=False,
        metadata=metadata,
    )

    validated = weekly_product.validate_published_weekly_product(weekly_output_dir, require_metadata=True)
    assert validated is not None
    metadata_on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata_on_disk["reference_week_name"] == "week1"
    assert metadata_on_disk["published_files"] == validated["published_files"]
    assert "week1_average_SyN.tif" not in metadata_on_disk["published_files"]
    assert "week1_average_cp_masks_SyN.tif" not in metadata_on_disk["published_files"]
    assert "week2_average_SyN.tif" in metadata_on_disk["published_files"]
    assert "week3_average_SyN.tif" in metadata_on_disk["published_files"]


def test_refresh_required_when_published_product_exists(tmp_path: Path) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    registered_dir = tmp_path / "registered"
    raw_root = tmp_path / "raw"
    weekly_output_dir.mkdir()
    registered_dir.mkdir()
    raw_root.mkdir()
    before = _publish_weekly_product(
        weekly_output_dir,
        registered_input_dir=registered_dir,
        day0_date="20260511",
        day1_date="20260512",
        crop_label=None,
        cropping_enabled=False,
        prepare_refresh=False,
        replace_existing=False,
    )
    snapshot = _snapshot_published_product(weekly_output_dir)

    with pytest.raises(FileExistsError, match="REFRESH_WEEKLY_PRODUCT=True"):
        prepare_weekly_product_workspace(weekly_output_dir, refresh=False)

    assert _snapshot_published_product(weekly_output_dir) == snapshot
    assert before.read_text(encoding="utf-8") == (weekly_output_dir / "weekly_product_metadata.json").read_text(encoding="utf-8")


def test_crossweek_registration_failure_leaves_previous_product_unchanged(tmp_path: Path) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    siblings = _write_siblings(tmp_path)
    _publish_weekly_product(
        weekly_output_dir,
        registered_input_dir=siblings["registered"],
        day0_date="20260511",
        day1_date="20260512",
        crop_label=None,
        cropping_enabled=False,
        prepare_refresh=False,
        replace_existing=False,
    )
    before = _snapshot_published_product(weekly_output_dir)

    staging_dir = prepare_weekly_product_workspace(weekly_output_dir, refresh=True)
    _build_week1_stage(
        staging_dir,
        day0_date="20260518",
        day1_date="20260519",
        crop_label="256x128",
        values_offset=100,
        include_crossweek_outputs=False,
    )

    with pytest.raises(RuntimeError, match="simulated cross-week failure"):
        _write_stack(staging_dir / "week1_average_SyN.tif", [700, 701, 702])
        _write_stack(staging_dir / "week1_average_cp_masks_SyN.tif", [800, 801, 802])
        raise RuntimeError("simulated cross-week failure")

    assert _snapshot_published_product(weekly_output_dir) == before
    assert (siblings["registered"] / "sentinel.txt").read_text(encoding="utf-8") == "registered"
    assert (siblings["raw"] / "sentinel.txt").read_text(encoding="utf-8") == "raw"
    assert (siblings["other_mouse"] / "sentinel.txt").read_text(encoding="utf-8") == "other"
    assert staging_dir.exists()


def test_refresh_publish_rolls_back_exactly_when_mid_commit_fails(tmp_path: Path, monkeypatch) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    siblings = _write_siblings(tmp_path)
    _publish_weekly_product(
        weekly_output_dir,
        registered_input_dir=siblings["registered"],
        day0_date="20260511",
        day1_date="20260512",
        crop_label=None,
        cropping_enabled=False,
        prepare_refresh=False,
        replace_existing=False,
    )
    before = _snapshot_published_product(weekly_output_dir)

    staging_dir = prepare_weekly_product_workspace(weekly_output_dir, refresh=True)
    week_dict, expected_names = _build_week1_stage(
        staging_dir,
        day0_date="20260518",
        day1_date="20260519",
        crop_label="256x128",
        values_offset=200,
    )
    metadata = build_weekly_product_metadata(
        crop_shape=[1, 3],
        cropping_enabled=True,
        crop_label="256x128",
        refresh_requested=True,
        registered_input_dir=siblings["registered"].as_posix(),
        source_registered_files=[
            Path(path).name
            for paths in week_dict.values()
            for path in paths
        ],
        spacing_zyx=(5.0, 0.69, 0.69),
        staging_dir=staging_dir,
        weekly_output_dir=weekly_output_dir,
        week_names=["week1", "week2"],
        reference_week_name="week2",
        published_at="2026-08-27T00:00:00",
    )

    real_replace = weekly_product.os.replace
    staged_move_count = {"count": 0}

    def flaky_replace(src, dst):
        src_path = Path(src)
        if src_path.parent == staging_dir:
            staged_move_count["count"] += 1
            if staged_move_count["count"] == 2:
                raise RuntimeError("simulated mid-commit failure")
        return real_replace(src, dst)

    monkeypatch.setattr(weekly_product.os, "replace", flaky_replace)

    with pytest.raises(RuntimeError, match="simulated mid-commit failure"):
        publish_staged_weekly_product(
            staging_dir,
            weekly_output_dir,
            expected_filenames=sorted(expected_names),
            metadata_filename="weekly_product_metadata.json",
            replace_existing=True,
            metadata=metadata,
        )

    assert _snapshot_published_product(weekly_output_dir) == before
    assert (siblings["registered"] / "sentinel.txt").read_text(encoding="utf-8") == "registered"
    assert (siblings["raw"] / "sentinel.txt").read_text(encoding="utf-8") == "raw"
    assert (siblings["other_mouse"] / "sentinel.txt").read_text(encoding="utf-8") == "other"


def test_successful_refresh_replaces_old_product_and_keeps_siblings(tmp_path: Path, monkeypatch) -> None:
    weekly_output_dir = tmp_path / "weekly_registered"
    siblings = _write_siblings(tmp_path)
    _publish_weekly_product(
        weekly_output_dir,
        registered_input_dir=siblings["registered"],
        day0_date="20260511",
        day1_date="20260512",
        crop_label=None,
        cropping_enabled=False,
        prepare_refresh=False,
        replace_existing=False,
    )
    old_snapshot = _snapshot_published_product(weekly_output_dir)

    staging_dir = prepare_weekly_product_workspace(weekly_output_dir, refresh=True)
    week_dict, expected_names = _build_week1_stage(
        staging_dir,
        day0_date="20260518",
        day1_date="20260519",
        crop_label="256x128",
        values_offset=300,
    )
    metadata = build_weekly_product_metadata(
        crop_shape=[1, 3],
        cropping_enabled=True,
        crop_label="256x128",
        refresh_requested=True,
        registered_input_dir=siblings["registered"].as_posix(),
        source_registered_files=[
            Path(path).name
            for paths in week_dict.values()
            for path in paths
        ],
        spacing_zyx=(5.0, 0.69, 0.69),
        staging_dir=staging_dir,
        weekly_output_dir=weekly_output_dir,
        week_names=["week1", "week2"],
        reference_week_name="week2",
        published_at="2026-08-27T00:00:00",
    )

    replace_log: list[tuple[str, str]] = []
    real_replace = weekly_product.os.replace

    def recording_replace(src, dst):
        replace_log.append((Path(src).name, Path(dst).name))
        return real_replace(src, dst)

    monkeypatch.setattr(weekly_product.os, "replace", recording_replace)

    metadata_path = publish_staged_weekly_product(
        staging_dir,
        weekly_output_dir,
        expected_filenames=sorted(expected_names),
        metadata_filename="weekly_product_metadata.json",
        replace_existing=True,
        metadata=metadata,
    )

    published_snapshot = _snapshot_published_product(weekly_output_dir)
    assert published_snapshot != old_snapshot
    assert old_snapshot["20260511_R_SyN.tif"] != published_snapshot["20260518_R_SyN.tif"]
    assert not staging_dir.exists()
    assert replace_log[-1][1] == metadata_path.name

    validated_metadata = weekly_product.validate_published_weekly_product(weekly_output_dir, require_metadata=True)
    assert validated_metadata is not None

    metadata_on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
    content_names = metadata_on_disk["published_files"]
    assert content_names == [
        "20260518_G_SyN.tif",
        "20260518_R_SyN.tif",
        "20260519_G_SyN.tif",
        "20260519_R_SyN.tif",
        "20260525_G_SyN.tif",
        "20260525_R_SyN.tif",
        "20260526_G_SyN.tif",
        "20260526_R_SyN.tif",
        "week1_average.tif",
        "week1_average_SyN.tif",
        "week1_average_cp_masks.tif",
        "week1_average_cp_masks_SyN.tif",
        "week2_average.tif",
        "week2_average_cp_masks.tif",
    ]
    assert metadata_on_disk["published_files"] == validated_metadata["published_files"]
    assert sorted(metadata_on_disk["published_sha256"]) == content_names
    assert sorted(validated_metadata["published_sha256"]) == content_names
    for name in content_names:
        assert metadata_on_disk["published_sha256"][name]
        assert weekly_output_dir.joinpath(name).exists()
    assert (siblings["registered"] / "sentinel.txt").read_text(encoding="utf-8") == "registered"
    assert (siblings["raw"] / "sentinel.txt").read_text(encoding="utf-8") == "raw"
    assert (siblings["other_mouse"] / "sentinel.txt").read_text(encoding="utf-8") == "other"


@pytest.mark.parametrize(
    "duplicate_crop_variants,incomplete,inconsistent,expected_exception,expected_message",
    [
        (True, False, False, ValueError, "Duplicate registered image candidates"),
        (False, True, False, FileNotFoundError, "Missing registered TIFFs"),
        (False, False, True, ValueError, "inconsistent TIFF dimensions"),
    ],
)
def test_project_weekly_runner_rejects_invalid_prepared_products_before_output_dir(
    tmp_path: Path,
    monkeypatch,
    duplicate_crop_variants: bool,
    incomplete: bool,
    inconsistent: bool,
    expected_exception,
    expected_message: str,
) -> None:
    config_path, _config = _make_project(tmp_path)
    match_csv = tmp_path / "matches.csv"
    _write_match_csv(match_csv)
    weekly_product_dir = tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "weekly_registered"
    weekly_product_dir.mkdir(parents=True)

    if duplicate_crop_variants:
        _write_stack(weekly_product_dir / "20260511_R_crop_64x64_SyN.tif", [1, 2, 3])
        _write_stack(weekly_product_dir / "20260511_R_crop_512x512_SyN.tif", [1, 2, 3])
        _write_stack(weekly_product_dir / "20260511_G_crop_64x64_SyN.tif", [1, 2, 3])
        _write_stack(weekly_product_dir / "20260511_G_crop_512x512_SyN.tif", [1, 2, 3])
    elif incomplete:
        _write_stack(weekly_product_dir / "20260511_R_SyN.tif", [1, 2, 3])
        _write_stack(weekly_product_dir / "20260511_G_SyN.tif", [1, 2, 3])
        _write_stack(weekly_product_dir / "20260512_R_SyN.tif", [1, 2, 3])
    elif inconsistent:
        _write_stack(weekly_product_dir / "20260511_R_SyN.tif", [1, 2, 3])
        _write_stack(weekly_product_dir / "20260511_G_SyN.tif", [1, 2, 3])
        _write_stack(weekly_product_dir / "20260512_R_SyN.tif", [1, 2, 3, 4], shape=(1, 1, 4))
        _write_stack(weekly_product_dir / "20260512_G_SyN.tif", [1, 2, 3, 4], shape=(1, 1, 4))

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

    with pytest.raises(expected_exception, match=expected_message):
        run_weekly.main()

    assert not (tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "runs").exists()


def test_project_mode_requires_weekly_product_metadata(tmp_path: Path, monkeypatch) -> None:
    config_path, _config = _make_project(tmp_path)
    match_csv = tmp_path / "matches.csv"
    _write_match_csv(match_csv)
    weekly_product_dir = tmp_path / "derivatives" / "m" / "longitudinal" / "920" / "weekly_registered"
    weekly_product_dir.mkdir(parents=True)
    _write_stack(weekly_product_dir / "20260511_R_SyN.tif", [1, 2, 3])
    _write_stack(weekly_product_dir / "20260511_G_SyN.tif", [4, 5, 6])
    _write_stack(weekly_product_dir / "20260512_R_SyN.tif", [7, 8, 9])
    _write_stack(weekly_product_dir / "20260512_G_SyN.tif", [10, 11, 12])
    tifffile.imwrite(weekly_product_dir / "week1_average_cp_masks.tif", np.asarray([[[1, 2, 3]]], dtype=np.uint16))

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

    with pytest.raises(FileNotFoundError, match="Weekly product metadata was not found"):
        run_weekly.main()
