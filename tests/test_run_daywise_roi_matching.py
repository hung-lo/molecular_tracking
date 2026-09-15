from __future__ import annotations

from pathlib import Path
import json

import pytest
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
    for day in ["20260511", "20260512", "20260513"]:
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
                "session_index": 2,
                "session_id": "20260513",
                "acquisition_date": "2026-05-13",
                "mask_path": str(tmp_path / "20260513_mask.tif"),
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

@pytest.mark.parametrize("max_pair_gap", [0, -1, 3, 1.5])
def test_run_daywise_roi_matching_rejects_invalid_pair_gap_values(tmp_path: Path, max_pair_gap: float) -> None:
    manifest_path = _build_dataset(tmp_path)

    with pytest.raises(ValueError):
        run_daywise_roi_matching(
            manifest_path=manifest_path,
            output_dir=tmp_path / f"match_out_{max_pair_gap}",
            spacing=VoxelSpacing(),
            params=AffineOverlapParams(),
            max_pair_gap=max_pair_gap,
            overwrite=True,
        )


def test_pair_workers_preserve_exact_scientific_outputs(tmp_path: Path) -> None:
    manifest_path = _build_dataset(tmp_path)
    outputs = []
    for workers in (1, 2):
        outputs.append(run_daywise_roi_matching(
            manifest_path=manifest_path,
            output_dir=tmp_path / f"match_workers_{workers}",
            spacing=VoxelSpacing(),
            params=AffineOverlapParams(),
            pair_workers=workers,
            save_candidates=True,
            overwrite=True,
            skip_qc=True,
        ))

    scientific_csvs = [
        "roi_features.csv", "pairwise_transforms.csv", "pairwise_matches_high.csv",
        "pairwise_matches_balanced.csv", "pairwise_candidates.csv", "tracks_high.csv",
        "tracks_balanced.csv", "cycle_consistency_high.csv", "cycle_consistency_balanced.csv",
        "cycle_edge_checks_high.csv", "cycle_edge_checks_balanced.csv", "track_edges_high.csv",
        "track_edges_balanced.csv", "track_length_summary.csv", "session_manifest_resolved.csv",
    ]
    for filename in scientific_csvs:
        pd.testing.assert_frame_equal(pd.read_csv(outputs[0] / filename), pd.read_csv(outputs[1] / filename))
    left_summary = pd.read_csv(outputs[0] / "pairwise_summary.csv").drop(columns="elapsed_sec")
    right_summary = pd.read_csv(outputs[1] / "pairwise_summary.csv").drop(columns="elapsed_sec")
    pd.testing.assert_frame_equal(left_summary, right_summary)

    run_log = json.loads((outputs[1] / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["pair_workers"] == 2
    assert len(run_log["runtime_profile"]["pair_timings_seconds"]) == 3
