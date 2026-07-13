from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tifffile

from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing
from daywise_roi_matcher_qc_plots import DaywiseQCPlotConfig, generate_daywise_qc_plots
from run_daywise_roi_matching import run_daywise_roi_matching


def _write_stack(path: Path, data: np.ndarray) -> None:
    tifffile.imwrite(path, data.astype(np.uint16))


def _build_dataset(tmp_path: Path) -> tuple[Path, Path]:
    mask = np.zeros((2, 3, 3), dtype=np.uint16)
    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 2
    mask[1, 2, 2] = 3
    for day in ["20260511", "20260512"]:
        _write_stack(tmp_path / f"{day}_mask.tif", mask)

    manifest = pd.DataFrame(
        [
            {
                "session_index": 0,
                "session_id": "20260511",
                "acquisition_date": "2026-05-11",
                "mask_path": str(tmp_path / "20260511_mask.tif"),
                "red_image_path": "",
                "green_image_path": "",
                "required": True,
            },
            {
                "session_index": 1,
                "session_id": "20260512",
                "acquisition_date": "2026-05-12",
                "mask_path": str(tmp_path / "20260512_mask.tif"),
                "red_image_path": "",
                "green_image_path": "",
                "required": True,
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    match_dir = tmp_path / "match_out"
    run_daywise_roi_matching(
        manifest_path=manifest_path,
        output_dir=match_dir,
        spacing=VoxelSpacing(),
        params=AffineOverlapParams(),
        overwrite=True,
    )
    return manifest_path, match_dir


def test_generate_daywise_qc_plots_writes_review_sample(tmp_path: Path) -> None:
    _, match_dir = _build_dataset(tmp_path)
    output_dir = generate_daywise_qc_plots(
        DaywiseQCPlotConfig(
            match_dir=str(match_dir),
            output_dir=str(tmp_path / "qc_plots"),
            sample_limit=3,
            review_seed=11,
        )
    )

    review_sample = pd.read_csv(output_dir / "manual_review_sample.csv")
    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))

    assert not review_sample.empty
    assert (output_dir / "cycle_agreement.png").exists()
    assert run_log["review_sample_rows"] == len(review_sample)
    assert run_log["saved_plots"]
