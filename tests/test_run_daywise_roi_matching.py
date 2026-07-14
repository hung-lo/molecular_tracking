from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tifffile

from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing
from run_daywise_roi_matching import run_daywise_roi_matching


def _write_stack(path: Path, data: np.ndarray) -> None:
    tifffile.imwrite(path, data.astype(np.uint16))


def _build_dataset(tmp_path: Path) -> Path:
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
    return manifest_path


def test_run_daywise_roi_matching_exports_required_tables(tmp_path: Path) -> None:
    manifest_path = _build_dataset(tmp_path)
    output_dir = run_daywise_roi_matching(
        manifest_path=manifest_path,
        output_dir=tmp_path / "match_out",
        spacing=VoxelSpacing(),
        params=AffineOverlapParams(),
        save_candidates=True,
        overwrite=True,
    )

    assert (output_dir / "session_manifest_resolved.csv").exists()
    assert (output_dir / "pairwise_summary.csv").exists()
    assert (output_dir / "tracks_high.csv").exists()
    assert (output_dir / "tracks_balanced.csv").exists()
    assert (output_dir / "pairwise_candidates.csv").exists()
    assert (output_dir / "qc").exists()
    assert (output_dir / "qc" / "qc_report.md").exists()

    pairwise_summary = pd.read_csv(output_dir / "pairwise_summary.csv")
    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))

    assert pairwise_summary["elapsed_sec"].iloc[0] > 0
    assert run_log["matching_status"] == "completed"
    assert run_log["qc_status"] == "completed"
    assert run_log["row_counts"]["tracks_high"] > 0
    assert run_log["row_counts"]["tracks_balanced"] > 0
    assert run_log["output_paths"]["tracks_high"].endswith("tracks_high.csv")
    assert run_log["qc_output_dir"].endswith("qc")
