from __future__ import annotations

from run_daywise_master_pipeline import MasterPipelineConfig, parse_args


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


def test_run_daywise_master_pipeline_config_defaults() -> None:
    config = MasterPipelineConfig(dataset="/tmp/dataset", manifest="/tmp/manifest.csv")

    assert config.dataset == "/tmp/dataset"
    assert config.manifest == "/tmp/manifest.csv"
    assert config.plot_columns == 7
    assert config.top_n == 30
    assert config.segmentation_qc_mode == "all_required"
