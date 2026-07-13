from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import tifffile

from session_manifest import load_session_manifest, validate_manifest_for_intensity, validate_manifest_for_matching


def _write_stack(path: Path) -> None:
    tifffile.imwrite(path, [[0]], dtype="uint16")


def test_load_session_manifest_sorts_by_session_index_and_resolves_paths(tmp_path: Path) -> None:
    mask1 = tmp_path / "masks" / "day1_mask.tif"
    mask2 = tmp_path / "masks" / "day0_mask.tif"
    mask1.parent.mkdir(parents=True, exist_ok=True)
    mask2.parent.mkdir(parents=True, exist_ok=True)
    _write_stack(mask1)
    _write_stack(mask2)

    manifest = pd.DataFrame(
        [
            {
                "session_index": 1,
                "session_id": "day1",
                "acquisition_date": "2026-05-12",
                "mask_path": "masks/day1_mask.tif",
                "red_image_path": "",
                "green_image_path": "",
                "required": True,
            },
            {
                "session_index": 0,
                "session_id": "day0",
                "acquisition_date": "2026-05-11",
                "mask_path": "masks/day0_mask.tif",
                "red_image_path": "",
                "green_image_path": "",
                "required": False,
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    records = load_session_manifest(manifest_path)

    assert [record.session_index for record in records] == [0, 1]
    assert records[0].mask_path == mask2.resolve()
    assert records[1].mask_path == mask1.resolve()
    validate_manifest_for_matching(records)


def test_load_session_manifest_rejects_invalid_boolean(tmp_path: Path) -> None:
    _write_stack(tmp_path / "mask.tif")
    manifest = pd.DataFrame(
        [
            {
                "session_index": 0,
                "session_id": "day0",
                "acquisition_date": "2026-05-11",
                "mask_path": "mask.tif",
                "red_image_path": "",
                "green_image_path": "",
                "required": "maybe",
            },
            {
                "session_index": 1,
                "session_id": "day1",
                "acquisition_date": "2026-05-12",
                "mask_path": "mask.tif",
                "red_image_path": "",
                "green_image_path": "",
                "required": "true",
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="required"):
        load_session_manifest(manifest_path)


def test_validate_manifest_for_intensity_requires_both_channels(tmp_path: Path) -> None:
    _write_stack(tmp_path / "mask0.tif")
    _write_stack(tmp_path / "mask1.tif")
    red = tmp_path / "red.tif"
    green = tmp_path / "green.tif"
    _write_stack(red)
    _write_stack(green)

    manifest = pd.DataFrame(
        [
            {
                "session_index": 0,
                "session_id": "day0",
                "acquisition_date": "2026-05-11",
                "mask_path": "mask0.tif",
                "red_image_path": "red.tif",
                "green_image_path": "green.tif",
                "required": True,
            },
            {
                "session_index": 1,
                "session_id": "day1",
                "acquisition_date": "2026-05-12",
                "mask_path": "mask1.tif",
                "red_image_path": "",
                "green_image_path": "",
                "required": True,
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    records = load_session_manifest(manifest_path)
    validate_manifest_for_matching(records)
    with pytest.raises(ValueError, match="image"):
        validate_manifest_for_intensity(records)
