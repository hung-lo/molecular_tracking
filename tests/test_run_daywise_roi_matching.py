from __future__ import annotations

from pathlib import Path
import json

import pytest
import numpy as np
import pandas as pd
import tifffile

import run_daywise_roi_matching as runner
from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing
from image_registration import ImageTransform
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


def test_image_registration_mode_exports_qc_and_preserves_native_inputs(tmp_path: Path, monkeypatch) -> None:
    shape = (9, 36, 36)
    mask = np.zeros(shape, dtype=np.uint16)
    mask[1:3, 4:7, 4:7] = 1
    mask[4:6, 16:19, 16:19] = 2
    mask[6:8, 28:31, 28:31] = 3
    red = np.random.default_rng(7).normal(size=shape).astype(np.float32)
    manifest_rows = []
    for index, day in enumerate(("20260511", "20260512")):
        mask_path = tmp_path / f"{day}_mask.tif"
        red_path = tmp_path / f"{day}_red.tif"
        _write_stack(mask_path, mask)
        tifffile.imwrite(red_path, red)
        manifest_rows.append({
            "session_index": index,
            "session_id": day,
            "acquisition_date": f"2026-05-{11 + index:02d}",
            "mask_path": str(mask_path),
            "red_image_path": str(red_path),
            "green_image_path": "",
            "required": True,
        })
    manifest_path = tmp_path / "image_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    before_mask = {path.name: path.read_bytes() for path in tmp_path.glob("*_mask.tif")}
    before_red = {path.name: path.read_bytes() for path in tmp_path.glob("*_red.tif")}

    def fake_fit(_a, _b, initial, _step):
        return ImageTransform(np.eye(3), np.zeros(3), method="image_affine"), {"optimizer_success": True}

    def fake_local(_a, _b, affine, _step, _spacing, *, smoothing_um):
        assert smoothing_um == 15.0
        return ImageTransform(affine.matrix, affine.offset, method="image_affine_local")

    def fake_quality(_a, _b, transform, _step, _spacing):
        return {
                "heldout_ncc": 1.01 if transform.method == "image_affine_local" else 1.0,
            "sample_overlap": 1.0,
            "affine_det": 1.0,
            "affine_singular_min": 1.0,
            "affine_singular_max": 1.0,
            "jacobian_min": 1.0,
            "jacobian_p01": 1.0,
            "jacobian_p99": 1.0,
            "displacement_p50_um": 0.0,
            "displacement_p95_um": 0.0,
            "displacement_p99_um": 0.0,
        }

    monkeypatch.setattr(runner, "fit_image_affine", fake_fit)
    monkeypatch.setattr(runner, "fit_smooth_field", fake_local)
    monkeypatch.setattr(runner, "image_quality", fake_quality)
    output_dir = run_daywise_roi_matching(
        manifest_path=manifest_path,
        output_dir=tmp_path / "image_out",
        registration_mode="image_affine_local",
        registration_smoothing_um=15.0,
        save_candidates=True,
        overwrite=True,
        skip_qc=True,
    )

    qc = pd.read_csv(output_dir / "pairwise_registration_qc.csv")
    assert qc.loc[0, "selected_registration_stage"] == "image_affine_local"
    assert qc.loc[0, "fallback_used"] is False or not bool(qc.loc[0, "fallback_used"])
    assert Path(qc.loc[0, "transform_path"]).exists()
    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["registration_mode"] == "image_affine_local"
    assert run_log["registration_smoothing_um"] == 15.0
    assert before_mask == {path.name: path.read_bytes() for path in tmp_path.glob("*_mask.tif")}
    assert before_red == {path.name: path.read_bytes() for path in tmp_path.glob("*_red.tif")}

    with pytest.raises(FileExistsError):
        run_daywise_roi_matching(
            manifest_path=manifest_path,
            output_dir=output_dir,
            registration_mode="legacy",
            resume=True,
            skip_qc=True,
        )
