from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import pytest

import run_daywise_master_pipeline as master
from run_daywise_master_pipeline import (
    MasterPipelineConfig,
    SessionSelection,
    _parse_session_selection,
    _select_session_records,
    _session_selection_provenance,
    _verify_resume_session_selection,
    _write_selected_session_manifest,
    parse_args,
)
from session_manifest import SessionRecord, load_session_manifest


def test_run_daywise_master_pipeline_parse_args_and_defaults() -> None:
    args = parse_args(
        [
            "--dataset",
            "/tmp/dataset",
            "--manifest",
            "/tmp/daywise_session_manifest.csv",
        ]
    )

    assert args.dataset == "/tmp/dataset"
    assert args.manifest == "/tmp/daywise_session_manifest.csv"
    assert args.plot_columns == 7
    assert args.top_n == 30
    assert args.segmentation_qc_mode == "all_required"
    assert args.overwrite is False
    assert args.resume is False
    assert args.sessions is None
    assert args.skip_ranked_roi_views is False
    assert args.ranked_roi_z_radius == 3


def test_run_daywise_master_pipeline_config_defaults() -> None:
    config = MasterPipelineConfig(dataset="/tmp/dataset", manifest="/tmp/manifest.csv")

    assert config.dataset == "/tmp/dataset"
    assert config.manifest == "/tmp/manifest.csv"
    assert config.plot_columns == 7
    assert config.top_n == 30
    assert config.segmentation_qc_mode == "all_required"
    assert config.skip_ranked_roi_views is False
    assert config.ranked_roi_z_radius == 3


def _build_records(tmp_path: Path, count: int = 6) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    for index in range(count):
        session_dir = tmp_path / f"session_{index}"
        session_dir.mkdir(parents=True)
        mask_path = session_dir / "mask.tif"
        red_path = session_dir / "red.tif"
        green_path = session_dir / "green.tif"
        for path in (mask_path, red_path, green_path):
            path.write_bytes(b"test")
        records.append(
            SessionRecord(
                session_index=index,
                session_id=f"session_{index}",
                acquisition_date=date(2026, 5, 11) + timedelta(days=index),
                mask_path=mask_path.resolve(),
                red_image_path=red_path.resolve(),
                green_image_path=green_path.resolve(),
                required=True,
            )
        )
    return records


def test_run_daywise_master_pipeline_parser_accepts_ranked_roi_options() -> None:
    args = parse_args(
        [
            "--dataset",
            "/tmp/dataset",
            "--manifest",
            "/tmp/manifest.csv",
            "--skip-ranked-roi-views",
            "--ranked-roi-z-radius",
            "0",
        ]
    )
    assert args.skip_ranked_roi_views is True
    assert args.ranked_roi_z_radius == 0


def test_run_daywise_master_pipeline_rejects_negative_ranked_roi_z_radius() -> None:
    with pytest.raises(ValueError, match="ranked_roi_z_radius"):
        master.run_master_pipeline(
            MasterPipelineConfig(
                dataset="/tmp/dataset",
                manifest="/tmp/manifest.csv",
                ranked_roi_z_radius=-1,
            )
        )


def test_run_daywise_master_pipeline_parser_accepts_sessions() -> None:
    args = parse_args(["--dataset", "/tmp/dataset", "--manifest", "/tmp/manifest.csv", "--sessions", "first:5"])
    assert args.sessions == "first:5"


def test_parse_session_selection() -> None:
    assert _parse_session_selection(None) == SessionSelection("all", None, None)
    assert _parse_session_selection(" first:5 ") == SessionSelection("first", 5, "first:5")
    assert _parse_session_selection("last:5") == SessionSelection("last", 5, "last:5")

    for value in ("first:0", "last:1", "first:-5", "5", "middle:5", "first:abc", "first:5:2"):
        with pytest.raises(ValueError, match="--sessions"):
            _parse_session_selection(value)


def test_select_session_records_preserves_source_order_and_rejects_oversized(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    assert [record.session_index for record in _select_session_records(records, _parse_session_selection("first:3"))] == [0, 1, 2]
    assert [record.session_index for record in _select_session_records(records, _parse_session_selection("last:3"))] == [3, 4, 5]
    with pytest.raises(ValueError, match="contains only 6 sessions"):
        _select_session_records(records, _parse_session_selection("first:7"))


def test_subset_default_run_names_reflect_selected_scope(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    source_meta = master._records_metadata(records)
    first_selection = _parse_session_selection("first:3")
    last_selection = _parse_session_selection("last:3")

    first_name = master._default_run_name(
        Path("dataset"),
        master._records_metadata(_select_session_records(records, first_selection)),
        first_selection,
    )
    last_name = master._default_run_name(
        Path("dataset"),
        master._records_metadata(_select_session_records(records, last_selection)),
        last_selection,
    )
    all_name = master._default_run_name(Path("dataset"), source_meta, _parse_session_selection(None))

    assert "20260511_to_20260513_3s_first3" in first_name
    assert "20260514_to_20260516_3s_last3" in last_name
    assert first_name != last_name
    assert all_name == "dataset_20260511_to_20260516_6s_graph_affine_balanced"


def test_selected_manifest_reindexes_and_preserves_resolved_paths(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    records = _build_records(source_dir)
    source_manifest = source_dir / "manifest.csv"
    source_manifest.write_text(
        "session_index,session_id,acquisition_date,mask_path,red_image_path,green_image_path,required\n"
        + "\n".join(
            f"{record.session_index},{record.session_id},{record.acquisition_date.isoformat()},"
            f"{record.mask_path.relative_to(source_dir)},"
            f"{record.red_image_path.relative_to(source_dir)},"
            f"{record.green_image_path.relative_to(source_dir)},true"
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    source_records = load_session_manifest(source_manifest)
    selected = _select_session_records(source_records, _parse_session_selection("last:3"))
    selected_manifest = tmp_path / "run" / "selected_session_manifest.csv"

    _write_selected_session_manifest(selected_manifest, selected)
    loaded = load_session_manifest(selected_manifest)

    assert [record.session_index for record in loaded] == [0, 1, 2]
    assert [record.session_id for record in loaded] == ["session_3", "session_4", "session_5"]
    assert [record.mask_path for record in loaded] == [record.mask_path for record in selected]
    assert [record.red_image_path for record in loaded] == [record.red_image_path for record in selected]
    assert [record.green_image_path for record in loaded] == [record.green_image_path for record in selected]
    rows = selected_manifest.read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",")[1] == "source_session_index"
    assert [row.split(",")[1] for row in rows[1:]] == ["3", "4", "5"]
    assert all(row.rsplit(",", 1)[-1] == "true" for row in rows[1:])


def test_resume_rejects_different_effective_selection(tmp_path: Path) -> None:
    records = _build_records(tmp_path)
    first = _select_session_records(records, _parse_session_selection("first:3"))
    last = _select_session_records(records, _parse_session_selection("last:3"))
    run_dir = tmp_path / "run"
    effective_manifest = run_dir / "selected_session_manifest.csv"
    _write_selected_session_manifest(effective_manifest, first)
    first_selection = _session_selection_provenance(
        _parse_session_selection("first:3"), records, first, master.file_sha256(effective_manifest)
    )
    (run_dir / "session_selection.json").write_text(json.dumps(first_selection), encoding="utf-8")
    last_manifest_bytes = master._selected_manifest_csv_bytes(last)
    last_selection = _session_selection_provenance(
        _parse_session_selection("last:3"), records, last, master._sha256_bytes(last_manifest_bytes)
    )

    with pytest.raises(ValueError, match="differs"):
        _verify_resume_session_selection(
            run_dir,
            expected_selection=last_selection,
            effective_manifest_path=effective_manifest,
            expected_manifest_sha256=last_selection["effective_manifest_sha256"],
        )


def test_master_pipeline_passes_same_effective_manifest_to_matching_and_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _build_records(tmp_path / "source")
    source_manifest = tmp_path / "source_manifest.csv"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        "session_index,session_id,acquisition_date,mask_path,red_image_path,green_image_path,required\n"
        + "\n".join(
            f"{record.session_index},{record.session_id},{record.acquisition_date.isoformat()},"
            f"{record.mask_path},{record.red_image_path},{record.green_image_path},true"
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_root = tmp_path / "runs"
    seen: dict[str, object] = {}

    def fake_matching(**kwargs: object) -> Path:
        seen["matching_manifest"] = kwargs["manifest_path"]
        return Path(kwargs["output_dir"])

    def fake_extraction(config: object) -> Path:
        seen["extraction_manifest"] = getattr(config, "manifest")
        return tmp_path / "temporary_extraction"

    monkeypatch.setattr(master, "run_daywise_graph_matching", fake_matching)
    monkeypatch.setattr(
        master,
        "annotate_graph_affine_agreement",
        lambda _path: {"n_consensus_accepted_edges": 0, "n_graph_only_accepted_edges": 0},
    )
    monkeypatch.setattr(master, "run_daywise_matched_roi_pipeline", fake_extraction)
    monkeypatch.setattr(
        master,
        "_relocate_extraction_output",
        lambda _source, target: target,
    )
    monkeypatch.setattr(
        master,
        "annotate_extraction_outputs",
        lambda _path, _tracks: {"n_consensus_tracks": 0, "n_graph_only_tracks": 0},
    )
    monkeypatch.setattr(master, "plot_wrapped_daywise_linear_relationships", lambda **_kwargs: None)

    run_dir = master.run_master_pipeline(
        MasterPipelineConfig(
            dataset=str(dataset_dir),
            manifest=str(source_manifest),
            output_root=str(output_root),
            run_name="subset-test",
            sessions="last:3",
            skip_quick_plots=True,
            skip_ranked_roi_views=True,
        )
    )

    expected_manifest = run_dir / "selected_session_manifest.csv"
    assert Path(seen["matching_manifest"]) == expected_manifest
    assert Path(str(seen["extraction_manifest"])) == expected_manifest
    assert (run_dir / "session_selection.json").is_file()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["session_selection"]["selected_source_session_indices"] == [3, 4, 5]
    assert run_manifest["manifest_path"] == str(expected_manifest.resolve())


def _run_mocked_master_for_ranked_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    skip_ranked_roi_views: bool,
    skip_quick_plots: bool = True,
) -> tuple[Path, dict[str, object]]:
    records = _build_records(tmp_path / "source")
    source_manifest = tmp_path / "source_manifest.csv"
    source_manifest.write_text(
        "session_index,session_id,acquisition_date,mask_path,red_image_path,green_image_path,required\n"
        + "\n".join(
            f"{record.session_index},{record.session_id},{record.acquisition_date.isoformat()},"
            f"{record.mask_path},{record.red_image_path},{record.green_image_path},true"
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_root = tmp_path / "runs"
    seen: dict[str, object] = {}

    def fake_matching(**kwargs: object) -> Path:
        return Path(kwargs["output_dir"])

    def fake_extraction(_config: object) -> Path:
        return tmp_path / "temporary_extraction"

    def fake_ranked_views(**kwargs: object) -> Path:
        seen["ranked_kwargs"] = kwargs
        return (
            Path(str(kwargs["run_dir"]))
            / "plots"
            / "graph"
            / "single_roi_raw_validation"
        )

    monkeypatch.setattr(master, "run_daywise_graph_matching", fake_matching)
    monkeypatch.setattr(
        master,
        "annotate_graph_affine_agreement",
        lambda _path: {"n_consensus_accepted_edges": 0, "n_graph_only_accepted_edges": 0},
    )
    monkeypatch.setattr(master, "run_daywise_matched_roi_pipeline", fake_extraction)
    monkeypatch.setattr(master, "_relocate_extraction_output", lambda _source, target: target)
    monkeypatch.setattr(
        master,
        "annotate_extraction_outputs",
        lambda _path, _tracks: {"n_consensus_tracks": 0, "n_graph_only_tracks": 0},
    )
    monkeypatch.setattr(master, "plot_wrapped_daywise_linear_relationships", lambda **_kwargs: None)
    monkeypatch.setattr(master, "build_ranked_roi_views", fake_ranked_views)

    config = MasterPipelineConfig(
        dataset=str(dataset_dir),
        manifest=str(source_manifest),
        output_root=str(output_root),
        run_name="ranked-test",
        top_n=4,
        ranked_roi_z_radius=2,
        skip_quick_plots=skip_quick_plots,
        skip_ranked_roi_views=skip_ranked_roi_views,
    )
    run_dir = master.run_master_pipeline(config)
    return run_dir, seen


def test_master_pipeline_integrates_ranked_roi_views_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, seen = _run_mocked_master_for_ranked_views(
        tmp_path, monkeypatch, skip_ranked_roi_views=False, skip_quick_plots=True
    )

    assert seen["ranked_kwargs"] == {
        "run_dir": run_dir,
        "policy": "graph",
        "top_n": 4,
        "directions": ("increasing", "decreasing"),
        "z_radius": 2,
    }
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    expected_dir = run_dir / "plots" / "graph" / "single_roi_raw_validation"
    assert manifest["outputs"]["ranked_roi_views_dir"] == str(expected_dir)
    assert manifest["outputs"]["ranked_roi_batch_index"] == str(
        expected_dir / "ranked_roi_batch_index.csv"
    )
    summary = (run_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert str(expected_dir) in summary


def test_master_pipeline_skips_ranked_roi_views_and_records_null_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, seen = _run_mocked_master_for_ranked_views(
        tmp_path, monkeypatch, skip_ranked_roi_views=True
    )

    assert "ranked_kwargs" not in seen
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["ranked_roi_views_dir"] is None
    assert manifest["outputs"]["ranked_roi_batch_index"] is None
    summary = (run_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Ranked individual ROI views:" in summary
    assert "skipped" in summary
