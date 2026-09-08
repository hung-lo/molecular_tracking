import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd
import pytest

from fucci_run_io import is_filesystem_root, resolve_fucci_master_run


def test_filesystem_root_detection_is_platform_independent() -> None:
    assert is_filesystem_root(PurePosixPath("/"))
    assert is_filesystem_root(PureWindowsPath("C:/"))
    assert is_filesystem_root(PureWindowsPath("D:/"))
    assert is_filesystem_root(PureWindowsPath("F:/"))
    assert not is_filesystem_root(PureWindowsPath("C:"))
    assert not is_filesystem_root(PurePosixPath("/tmp/output"))
    assert not is_filesystem_root(PureWindowsPath("C:/data"))
    assert not is_filesystem_root(PureWindowsPath("D:/analysis/output"))


def _run(tmp_path: Path, *, version: str = "0.3.1", population: str = "all_valid_session_rois", with_manifest: bool = True) -> Path:
    root = tmp_path / "copied_run"
    extraction = root / "extraction"
    extraction.mkdir(parents=True)
    if with_manifest:
        (root / "run_manifest.json").write_text(json.dumps({"project": {"mouse_id": "Dead_1", "laser_nm": 1050}, "manifest": {"session_ids": ["s0"]}}))
    (extraction / "run_log.json").write_text(json.dumps({"analysis_version": version, "normalization": {"population": population}}))
    pd.DataFrame({"session_id": ["s0"], "session_index": [0], "acquisition_date": ["2026-01-01"], "elapsed_days": [0], "red": [10.0], "green": [12.0], "ratio_qc_pass": [True]}).to_csv(extraction / "matched_session_population_roi_metrics.csv", index=False)
    return root


def test_current_run_resolves_structurally_and_stale_paths_do_not_matter(tmp_path: Path) -> None:
    root = _run(tmp_path)
    result = resolve_fucci_master_run(root, require_matched=False)
    assert result.mouse_id == "Dead_1"
    assert result.session_ids == ("s0",)


def test_old_or_wrong_runs_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not current"):
        resolve_fucci_master_run(_run(tmp_path / "old", version="0.3.0"), require_matched=False)
    with pytest.raises(ValueError, match="not current"):
        resolve_fucci_master_run(_run(tmp_path / "wrong", population="complete_tracks"), require_matched=False)
    with pytest.raises(FileNotFoundError, match="manifest"):
        resolve_fucci_master_run(_run(tmp_path / "missing", with_manifest=False), require_matched=False)
