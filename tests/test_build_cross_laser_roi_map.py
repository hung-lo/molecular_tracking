from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_cross_laser_roi_map import (
    _session_summary,
    discover_canonical_session_pairs,
    run_cross_laser_roi_map,
)
from project_config import load_project_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    derivatives = tmp_path / "derivatives"
    raw.mkdir()
    mice = tmp_path / "mice.csv"
    mice.write_text(
        "mouse_id,experimental_group,cohort,raw_mouse_folder,reference_session_or_folder\n"
        "mouse_1,group,cohort,folder,\n",
        encoding="utf-8",
    )
    config = tmp_path / "project.toml"
    config.write_text(
        f"""[paths]
raw_root="{raw}"
derivatives_root="{derivatives}"
mice_csv="{mice}"
[rig]
primary_laser_nm=1050
optional_laser_nm=920
pockels_1_laser_nm=920
pockels_2_laser_nm=1050
chan_a_signal="green"
chan_b_signal="red"
[canonical_volume]
imaging_planes=12
flyback_planes=1
z_step_um=2.5
volumes=1
""",
        encoding="utf-8",
    )
    return config, raw, derivatives


def _catalog_row(
    *,
    session_id: str,
    acquisition_date: str,
    acquisition_id: str,
    laser_nm: int,
) -> dict[str, str]:
    return {
        "mouse_id": "mouse_1",
        "session_id": session_id,
        "acquisition_date": acquisition_date,
        "acquisition_id": acquisition_id,
        "source_path": f"/raw/{acquisition_id}",
        "analysis_included": "true",
        "laser_nm": str(laser_nm),
        "pixel_size_x_um": "0.7",
        "pixel_size_y_um": "0.7",
        "z_step_um": "2.5",
    }


def _write_catalog(derivatives: Path, rows: list[dict[str, str]]) -> None:
    catalog_dir = derivatives / "_catalog"
    catalog_dir.mkdir(parents=True)
    with (catalog_dir / "acquisitions.generated.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (catalog_dir / "validation_report.json").write_text(
        '{"catalog_version":"thorimage_catalog_v1","errors":[]}',
        encoding="utf-8",
    )


def _write_mask(path: Path, *, label_offset: int = 0) -> None:
    mask = np.zeros((12, 36, 36), dtype=np.uint16)
    mask[3:5, 4:9, 4:9] = 11 + label_offset
    mask[5:7, 16:21, 16:21] = 22 + label_offset
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, mask)


def _session_mask(derivatives: Path, date: str, laser_nm: int) -> Path:
    return (
        derivatives
        / "mouse_1"
        / "sessions"
        / date.replace("-", "")
        / str(laser_nm)
        / "segmentation"
        / "mask.tif"
    )


def test_one_session_cli_run_isolated_and_preserves_source_masks(tmp_path: Path) -> None:
    config_path, _, derivatives = _project(tmp_path)
    rows = [
        _catalog_row(
            session_id="session_20260819",
            acquisition_date="2026-08-19",
            acquisition_id="acq_1050",
            laser_nm=1050,
        ),
        _catalog_row(
            session_id="session_20260819",
            acquisition_date="2026-08-19",
            acquisition_id="acq_920",
            laser_nm=920,
        ),
        _catalog_row(
            session_id="session_20260820",
            acquisition_date="2026-08-20",
            acquisition_id="acq_1050_only",
            laser_nm=1050,
        ),
    ]
    _write_catalog(derivatives, rows)
    fixed_path = _session_mask(derivatives, "2026-08-19", 1050)
    moving_path = _session_mask(derivatives, "2026-08-19", 920)
    _write_mask(fixed_path)
    _write_mask(moving_path, label_offset=100)
    _write_mask(moving_path.with_name("mask_red.tif"), label_offset=200)
    fixed_before = _sha256(fixed_path)
    moving_before = _sha256(moving_path)

    run_dir = run_cross_laser_roi_map(
        project_config=config_path,
        mouse_id="mouse_1",
        session_ids=("20260819",),
        run_name="one-session",
        use_920_red_secondary=True,
    )

    assert run_dir == derivatives / "mouse_1" / "cross_laser" / "920_to_1050" / "runs" / "one-session"
    assert _sha256(fixed_path) == fixed_before
    assert _sha256(moving_path) == moving_before
    assert (run_dir / "fixed_roi_coverage.csv").is_file()
    assert (run_dir / "roi_map_by_source.csv").is_file()
    assert (run_dir / "relabelled_masks" / "session_20260819_920_green_native_as_1050_high.tif").is_file()
    manifest = pd.read_json(run_dir / "run_manifest.json", typ="series")
    assert manifest["secondary_red_enabled"]
    assert manifest["qc_status"] in {"completed", "completed_with_warnings"}
    resolution = pd.read_csv(run_dir / "identity_resolution.csv")
    assert len(resolution) == 2
    assert set(resolution["resolved_status"]) == {"primary_high"}
    assert set(resolution["secondary_red_status"]) == {"high"}
    relabelled = tifffile.imread(
        run_dir / "relabelled_masks" / "session_20260819_920_green_native_as_1050_high.tif"
    )
    assert relabelled.shape == tifffile.imread(moving_path).shape
    assert set(np.unique(relabelled)).issubset(set(np.unique(tifffile.imread(fixed_path))))
    consistency = pd.read_csv(run_dir / "matches_920_red_to_green_high.csv")
    assert {"fixed_label", "moving_label", "label_920_green", "label_920_red"}.issubset(consistency.columns)
    assert "label_1050" not in consistency.columns


def test_session_summary_counts_green_red_set_membership() -> None:
    def source(labels: list[int]) -> SimpleNamespace:
        pairs = pd.DataFrame({"label_1050": labels, "raw_delta_z_planes": [0.0] * len(labels), "aligned_residual_z_um": [0.0] * len(labels), "aligned_residual_distance_um": [0.0] * len(labels)})
        coverage = pd.DataFrame({"common_volume_status": ["inside_common_volume"] * 4, "green_high_label_920": [1, 2, 3, np.nan], "green_balanced_label_920": [1, 2, 3, np.nan]})
        return SimpleNamespace(summary={"mouse_id": "m", "session_id": "s", "acquisition_date": "2026-01-01", "shift_z": 0, "shift_y": 0, "shift_x": 0}, fixed_features=pd.DataFrame({"label": [1, 2, 3, 4]}), moving_features=pd.DataFrame({"label": [1, 2, 3, 4]}), high_matches=pairs, balanced_matches=pairs, fixed_coverage=coverage, transform=SimpleNamespace(n_seed=0, n_inlier=0, residual_median_um=0.0, residual_p95_um=0.0))
    resolution = pd.DataFrame({"resolved_status": ["primary_high"] * 4, "cross_source_conflict": [False] * 4})
    summary = _session_summary(primary=source([1, 2, 3]), secondary=source([2, 3, 4]), resolution=resolution)
    assert (summary["n_high_both_sources_same_1050"], summary["n_primary_high_only"], summary["n_secondary_high_only"]) == (2, 1, 1)


def test_unpaired_session_is_skipped_by_default_and_requested_session_fails(tmp_path: Path) -> None:
    config_path, _, derivatives = _project(tmp_path)
    rows = [
        _catalog_row(
            session_id="session_20260819",
            acquisition_date="2026-08-19",
            acquisition_id="acq_1050",
            laser_nm=1050,
        ),
        _catalog_row(
            session_id="session_20260820",
            acquisition_date="2026-08-20",
            acquisition_id="acq_920",
            laser_nm=920,
        ),
    ]
    _write_catalog(derivatives, rows)
    config = load_project_config(config_path)

    with pytest.raises(ValueError, match="No canonical paired"):
        discover_canonical_session_pairs(config, mouse_id="mouse_1")
    with pytest.raises(ValueError, match="Requested session"):
        discover_canonical_session_pairs(
            config, mouse_id="mouse_1", requested_sessions=("20260819",)
        )
