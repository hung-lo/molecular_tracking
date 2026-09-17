from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import pandas as pd
from pathlib import Path

import postprocessing.local_segmentation_rescue as rescue
from postprocessing.local_segmentation_rescue import (
    RunContext,
    TransformGraph,
    BackendUnavailable,
    _preflight_backend,
    _predict_target,
    _find_real_cases,
    _eligible_synthetic_cases_with_audit,
    _sample_cases,
    _select_review_artifacts,
    _overlap_decomposition,
    compute_crop_bounds,
    dice_iou,
    rank_candidates,
    segment_local_threshold,
)


def test_crop_bounds_xyz_to_zyx_and_clipping_are_deterministic():
    bounds = compute_crop_bounds((1.2, 5.0, 2.0), (10, 20, 30), (5, 8, 10))
    assert bounds.start_zyx == (0, 1, 0)
    assert bounds.stop_zyx == (5, 9, 10)
    assert bounds.shape_zyx == (5, 8, 10)
    assert bounds.edge_clipped
    assert bounds.as_dict()["crop_shape_zyx"] == [5, 8, 10]


def test_dice_iou_and_threshold_segmenter_are_deterministic():
    a = np.zeros((3, 4, 5), bool)
    b = np.zeros_like(a)
    a[1, 1:3, 1:3] = True
    b[1, 2:4, 1:3] = True
    assert dice_iou(a, b) == (0.5, 1 / 3)
    image = np.full((5, 12, 12), 10.0, dtype=float)
    image[2, 4:7, 4:7] = 100
    image[2, 4:7, 8:11] = 90
    first = segment_local_threshold(image, threshold_percentile=80, min_voxels=3)
    second = segment_local_threshold(image, threshold_percentile=80, min_voxels=3)
    assert first[2] == second[2]
    assert first[1].tolist() == second[1].tolist()
    assert len(first[0]) == 2


def test_overlap_decomposition_separates_identity_from_contamination():
    result = _overlap_decomposition([(90, 7), (5, 9), (5, 11)], 100, 7, 90)
    assert result["truth_is_top1"] is True
    assert result["identity_ranking_category"] == "dominant_truth_overlap"
    assert result["top1_overlap_fraction_of_candidate"] == 0.9
    assert result["top2_overlap_fraction_of_candidate"] == 0.05
    assert result["n_canonical_labels_overlap_ge_5pct_candidate"] == 3
    assert result["segmentation_contamination_category"] == "minor_neighbor_contamination"


def test_overlap_decomposition_identity_categories_and_flags_are_independent():
    truth = _overlap_decomposition([(90, 7), (10, 9)], 100, 7, 90)
    wrong_with_truth = _overlap_decomposition([(70, 9), (20, 7)], 100, 7, 20)
    wrong_without_truth = _overlap_decomposition([(70, 9)], 100, 7, 0)
    no_canonical = _overlap_decomposition([], 100, 7, 0)
    no_candidate = _overlap_decomposition([], 0, 7, 0, selected=False)

    assert truth["identity_ranking_category"] == "dominant_truth_overlap"
    assert truth["truth_overlap_present"] and not truth["no_truth_overlap"]
    assert wrong_with_truth["identity_ranking_category"] == "dominant_wrong_label"
    assert wrong_with_truth["truth_overlap_present"]
    assert wrong_without_truth["identity_ranking_category"] == "dominant_wrong_label"
    assert wrong_without_truth["no_truth_overlap"]
    assert no_canonical["identity_ranking_category"] == "no_canonical_overlap"
    assert no_canonical["no_truth_overlap"]
    assert no_candidate["identity_ranking_category"] == "no_candidate"
    assert no_candidate["segmentation_contamination_category"] == "no_candidate"


def test_canonical_overlap_serialization_reproduces_top1_top2_deterministically():
    result = _overlap_decomposition([(12, 6988), (176, 6412), (2, 7001)], 200, 6988, 12)
    assert result["canonical_overlap_voxels_json"] == "[[6412,176],[6988,12],[7001,2]]"
    serialized = json.loads(result["canonical_overlap_voxels_json"])
    assert serialized[0] == [6412, 176] and serialized[1] == [6988, 12]
    assert result["top1_canonical_label"] == serialized[0][0]
    assert result["top1_overlap_voxels"] == serialized[0][1]
    assert result["top2_canonical_label"] == serialized[1][0]
    assert result["top2_overlap_voxels"] == serialized[1][1]


def test_candidate_ranking_ignores_biological_state_columns():
    candidates = [
        {"candidate_id": 2, "centroid_xyz": [5, 5, 5], "volume_um3": 100, "touches_crop_edge": False, "eclipse_z": -10},
        {"candidate_id": 1, "centroid_xyz": [6, 5, 5], "volume_um3": 100, "touches_crop_edge": False, "eclipse_z": 10},
    ]
    ranked = rank_candidates(candidates, [5, 5, 5], (1, 1, 1))
    assert [item["candidate_id"] for item in ranked] == [2, 1]
    candidates[0]["eclipse_z"] = 1000
    assert rank_candidates(candidates, [5, 5, 5], (1, 1, 1))[0]["candidate_id"] == 2


def test_hidden_target_volume_cannot_change_candidate_ranking():
    candidates = [
        {"candidate_id": 1, "centroid_xyz": [5, 5, 5], "volume_um3": 10, "touches_crop_edge": False},
        {"candidate_id": 2, "centroid_xyz": [6, 5, 5], "volume_um3": 1000, "touches_crop_edge": False},
    ]
    first = [row["candidate_id"] for row in rank_candidates(candidates, [5.2, 5, 5], (1, 1, 1), 10)]
    second = [row["candidate_id"] for row in rank_candidates(candidates, [5.2, 5, 5], (1, 1, 1), 10000)]
    assert first == second


def test_prediction_context_is_explicit_and_one_sided_ignores_future():
    sessions = pd.DataFrame([{"session_id": f"s{i}", "session_index": i} for i in range(3)])
    features = pd.DataFrame([
        {"session_id": "s0", "label": 1, "centroid_z": 0, "centroid_y": 0, "centroid_x": 0},
        {"session_id": "s2", "label": 1, "centroid_z": 0, "centroid_y": 0, "centroid_x": 100},
    ])
    transforms = pd.DataFrame([
        {"day_a": "s0", "day_b": "s1"},
        {"day_a": "s1", "day_b": "s2"},
    ])
    context = RunContext(__import__("pathlib").Path("/tmp/run"), __import__("pathlib").Path("/tmp/matching"), features, pd.DataFrame(), sessions, transforms, (1, 1, 1), "test", {}, "", "", "", {})
    lookup = {(str(row.session_id), int(row.label)): row for _, row in features.iterrows()}
    graph = TransformGraph(transforms)
    observations = {0: ("s0", 1), 2: ("s2", 1)}
    one = _predict_target(context, observations, 1, lookup, graph, benchmark_context="endpoint_one_sided")
    two = _predict_target(context, observations, 1, lookup, graph, benchmark_context="internal_gap_two_sided")
    assert one["predicted_xyz"] == [0.0, 0.0, 0.0]
    assert two["predicted_xyz"] == [50.0, 0.0, 0.0]


def test_real_evaluator_schema_derives_immediate_next_target(tmp_path: Path):
    sessions = pd.DataFrame([{"session_id": f"s{i}", "session_index": i} for i in range(3)])
    context = RunContext(Path("/tmp/run"), Path("/tmp/matching"), pd.DataFrame(), pd.DataFrame(), sessions, pd.DataFrame(), (1, 1, 1), "test", {}, "", "", "", {})
    artifact = tmp_path / "endpoint_classification.csv"
    pd.DataFrame([{"endpoint_id": "e1", "track_uid": "t1", "end_session_index": 1, "end_session_id": "s1", "end_label": 7, "classification": "no_mask_near_prediction"}]).to_csv(artifact, index=False)
    cases = _find_real_cases(context, None, 0, artifact)
    assert len(cases) == 1
    assert cases[0]["source_session_index"] == 1
    assert cases[0]["target_session_index"] == 2
    assert cases[0]["target_session"] == "s2"
    assert cases[0]["classification_source_sha256"]


def test_real_evaluator_falls_through_nan_source_alias_to_end_label(tmp_path: Path):
    sessions = pd.DataFrame([{"session_id": f"s{i}", "session_index": i} for i in range(3)])
    context = RunContext(Path("/tmp/run"), Path("/tmp/matching"), pd.DataFrame(), pd.DataFrame(), sessions, pd.DataFrame(), (1, 1, 1), "test", {}, "", "", "", {})
    artifact = tmp_path / "endpoint_classification.csv"
    pd.DataFrame([{"endpoint_id": "e1", "track_uid": "t1", "end_session_index": 1, "end_session_id": "s1", "source_label": np.nan, "roi_id": np.nan, "end_label": 7, "classification": "no_mask_near_prediction"}]).to_csv(artifact, index=False)
    cases = _find_real_cases(context, None, 0, artifact)
    assert len(cases) == 1 and cases[0]["source_label"] == 7


def _synthetic_context(tracks: pd.DataFrame) -> RunContext:
    sessions = pd.DataFrame([{"session_id": f"s{i}", "session_index": i} for i in range(3)])
    features = pd.DataFrame([
        {"session_id": f"s{i}", "label": 1, "centroid_z": 0, "centroid_y": 0, "centroid_x": i, "touches_z_edge": False, "touches_xy_edge": False}
        for i in range(3)
    ])
    return RunContext(Path("/tmp/run"), Path("/tmp/matching"), features, tracks, sessions, pd.DataFrame(), (1, 1, 1), "test", {}, "", "", "", {})


def test_synthetic_trust_missing_evidence_fails_closed():
    context = _synthetic_context(pd.DataFrame([{"track_uid": "t1", "s0_roi": 1, "s1_roi": 1, "s2_roi": 1}]))
    audit: dict[str, int] = {}
    cases, excluded = _eligible_synthetic_cases_with_audit(context, audit)
    assert cases == [] and excluded == 1
    assert audit["missing_evidence"] == 1


def test_synthetic_trust_missing_target_edge_evidence_fails_closed():
    context = _synthetic_context(pd.DataFrame([{"track_uid": "t1", "s0_roi": 1, "s1_roi": 1, "s2_roi": 1}]))
    features = context.features.drop(columns=["touches_z_edge", "touches_xy_edge"])
    context = RunContext(context.run_dir, context.matching_dir, features, context.tracks, context.sessions, context.transforms, context.spacing_zyx, context.spacing_source, context.run_log, context.matching_sha256, context.manifest_sha256, context.git_commit, context.image_hashes)
    audit: dict[str, int] = {}
    cases, excluded = _eligible_synthetic_cases_with_audit(context, audit)
    assert cases == [] and excluded == 1 and audit["missing_evidence"] == 1


def test_synthetic_trust_records_verified_evidence_without_hard_coded_claims():
    context = _synthetic_context(pd.DataFrame([{
        "track_uid": "t1", "s0_roi": 1, "s1_roi": 1, "s2_roi": 1,
        "track_match_source": "consensus", "has_cycle_conflict": False, "cycle_unchecked": False,
        "contains_transform_fallback_edge": False, "n_adjacent_edges": 2,
        "max_distance_um": 4.0, "max_ambiguity": 0.2,
    }]))
    cases, excluded = _eligible_synthetic_cases_with_audit(context)
    assert excluded == 0 and len(cases) == 1
    assert cases[0]["synthetic_trust_status"] == "trusted"
    assert "consensus" in cases[0]["synthetic_trust_reasons"]
    assert "track_match_source" in cases[0]["synthetic_trust_fields_verified"]
    assert cases[0]["synthetic_trust_evidence_source"] == "track_summary"


def test_synthetic_trust_can_derive_evidence_from_canonical_edges():
    context = _synthetic_context(pd.DataFrame([{
        "track_uid": "t1", "s0_roi": 1, "s1_roi": 1, "s2_roi": 1,
        "_edges": [{"candidate_source": "both", "pair_gap": 1, "distance_um": 4.0, "ambiguity": 0.2, "transform_fallback_reason": ""}, {"candidate_source": "both", "pair_gap": 1, "distance_um": 3.0, "ambiguity": 0.1, "transform_fallback_reason": ""}],
    }]))
    context = RunContext(context.run_dir, context.matching_dir, context.features, context.tracks, context.sessions, context.transforms, context.spacing_zyx, context.spacing_source, context.run_log, context.matching_sha256, context.manifest_sha256, context.git_commit, context.image_hashes, cycle_edge_checks=pd.DataFrame([{"track_uid": "t1", "cycle_agrees": True}]))
    cases, excluded = _eligible_synthetic_cases_with_audit(context)
    assert excluded == 0 and cases[0]["synthetic_trust_evidence_source"] == "cycle_edge_checks+track_edges"
    assert "candidate_source" in cases[0]["synthetic_trust_fields_verified"]


def test_review_selection_includes_late_categories():
    artifacts = []
    for index in range(100):
        category = "correct_identity_good_mask" if index < 99 else "wrong_neighbor_identity"
        artifacts.append(({"identity_category": category, "dice_3d": 0.5, "centroid_error_um": 1.0}, None, {}))
    selected = _select_review_artifacts(artifacts, synthetic=True)
    assert any(item[0]["identity_category"] == "wrong_neighbor_identity" for item in selected)


def test_review_selection_samples_contamination_field_independently_of_identity():
    artifact = ({
        "identity_category": "correct_identity_good_mask",
        "identity_ranking_category": "dominant_truth_overlap",
        "segmentation_contamination_category": "substantial_neighbor_contamination",
        "dice_3d": 0.9,
        "centroid_error_um": 1.0,
    }, None, {})
    selected = _select_review_artifacts([artifact], synthetic=True)
    assert selected == [artifact]
    assert selected[0][0]["segmentation_contamination_category"] == "substantial_neighbor_contamination"


def test_summary_exposes_independent_identity_counts():
    context = _synthetic_context(pd.DataFrame())
    records = [
        {"selected_candidate_is_correct": True, "identity_ranking_category": "dominant_truth_overlap", "truth_overlap_present": True, "centroid_error_um": 1.0, "dice_3d": 0.8, "iou_3d": 0.7, "volume_ratio_rescue_to_truth": 1.0, "wrong_neighbor": False, "top1_overlap_fraction_of_candidate": 0.9, "top2_overlap_fraction_of_candidate": 0.05, "truth_overlap_fraction_of_candidate": 0.9},
        {"selected_candidate_is_correct": False, "identity_ranking_category": "dominant_wrong_label", "truth_overlap_present": False, "centroid_error_um": 2.0, "dice_3d": 0.2, "iou_3d": 0.1, "volume_ratio_rescue_to_truth": 0.8, "wrong_neighbor": True, "top1_overlap_fraction_of_candidate": 0.8, "top2_overlap_fraction_of_candidate": 0.1, "truth_overlap_fraction_of_candidate": 0.0},
        {"selected_candidate_is_correct": False, "identity_ranking_category": "no_candidate", "truth_overlap_present": False, "centroid_error_um": np.nan, "dice_3d": np.nan, "iou_3d": np.nan, "volume_ratio_rescue_to_truth": np.nan, "wrong_neighbor": False, "top1_overlap_fraction_of_candidate": np.nan, "top2_overlap_fraction_of_candidate": np.nan, "truth_overlap_fraction_of_candidate": np.nan},
    ]
    summary = rescue._summary(records, [], "synthetic_benchmark", context, 1)
    assert summary["truth_is_top1_count"] == 1 and summary["truth_is_top1_rate"] == 1 / 3
    assert summary["not_truth_top1_count"] == 2 and summary["not_truth_top1_rate"] == 2 / 3
    assert summary["dominant_wrong_label_count"] == 1 and summary["no_canonical_overlap_count"] == 0
    assert summary["no_truth_overlap_count"] == 2 and summary["no_truth_overlap_rate"] == 2 / 3


def test_case_sampling_is_deterministic():
    cases = [{"target_session_index": i % 3, "track_id": f"t{i}", "source_roi_id": i} for i in range(20)]
    assert _sample_cases(cases, 5, 42) == _sample_cases(cases, 5, 42)


def test_evaluate_review_reservoir_sees_categories_after_case_64(tmp_path: Path, monkeypatch):
    context = _synthetic_context(pd.DataFrame())
    cases = [{"target_session_index": 1, "track_id": f"t{i}", "source_roi_id": i, "source_label": 1, "target_session": "s1"} for i in range(100)]
    monkeypatch.setattr(rescue, "_load_run_context", lambda path: context)
    monkeypatch.setattr(rescue, "_eligible_synthetic_cases_with_audit", lambda context, audit=None: (cases, 0))

    def fake_run_case(_context, case, **kwargs):
        category = "correct_identity_good_mask" if case["source_roi_id"] < 99 else "wrong_neighbor_identity"
        return ({**case, "identity_category": category, "status": "candidate_generated", "n_candidates": 1, "dice_3d": 0.5, "centroid_error_um": 1.0}, [], None, {"crop": np.zeros((1, 2, 2)), "candidate_labels": np.zeros((1, 2, 2), dtype=int)})

    monkeypatch.setattr(rescue, "_run_case", fake_run_case)
    summary = rescue.evaluate(context.run_dir, tmp_path / "output", mode="synthetic_benchmark", sample_size=None, backend=rescue.THRESHOLD_BACKEND)
    assert summary["review_selection_full_population"] is True
    assert summary["review_selection_category_counts"]["wrong_neighbor_identity"] == 1


def test_cellpose_cache_is_keyed_by_device_and_version(monkeypatch):
    calls: list[bool] = []

    class FakeModel:
        def __init__(self, *, gpu: bool, pretrained_model: str):
            calls.append(gpu)

    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")
    fake_models.CellposeModel = FakeModel
    fake_cellpose.models = fake_models
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "fake-torch"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(rescue.importlib_metadata, "version", lambda name: "fake-cellpose")
    rescue._CELLPOSE_MODELS.clear()
    cpu = _preflight_backend("cellpose_sam", device="cpu")
    cuda = _preflight_backend("cellpose_sam", device="cuda")
    _preflight_backend("cellpose_sam", device="cpu")
    assert calls == [False, True]
    assert cpu["resolved_device"] == "cpu" and cuda["resolved_device"] == "cuda"
    assert cpu["cellpose_version"] == "fake-cellpose"
    assert cpu["model_cache_key"] == ["cpsam_v2", "cpu"]


def _minimal_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    matching = run / "matching"
    matching.mkdir(parents=True)
    pd.DataFrame([{"session_index": 0, "session_id": "s0"}, {"session_index": 1, "session_id": "s1"}]).to_csv(run / "selected_session_manifest.csv", index=False)
    pd.DataFrame(columns=["session_id", "label", "centroid_z", "centroid_y", "centroid_x"]).to_csv(matching / "roi_features.csv", index=False)
    pd.DataFrame(columns=["track_uid"]).to_csv(matching / "tracks_graph.csv", index=False)
    pd.DataFrame(columns=["day_a", "day_b"]).to_csv(matching / "pairwise_transforms.csv", index=False)
    return run


def test_missing_evaluator_artifact_fails_closed_and_preserves_canonical_inputs(tmp_path: Path):
    run = _minimal_run(tmp_path)
    canonical = [run / "selected_session_manifest.csv", run / "matching" / "roi_features.csv", run / "matching" / "tracks_graph.csv", run / "matching" / "pairwise_transforms.csv"]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in canonical}
    summary = rescue.evaluate(run, tmp_path / "output", mode="real_proposals", backend=rescue.THRESHOLD_BACKEND, endpoint_classification=tmp_path / "missing.csv")
    assert summary["status"] == "evaluator_artifact_unavailable"
    assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in canonical} == before
    assert rescue.main(["--run-dir", str(run), "--mode", "real_proposals", "--backend", rescue.THRESHOLD_BACKEND, "--output-dir", str(tmp_path / "cli-output"), "--endpoint-classification", str(tmp_path / "missing.csv")]) == 2


def test_zero_valid_evaluator_rows_fails_closed(tmp_path: Path):
    run = _minimal_run(tmp_path)
    artifact = tmp_path / "endpoint_classification.csv"
    pd.DataFrame([{"classification": "no_mask_near_prediction", "end_session_index": 99, "end_label": 7}]).to_csv(artifact, index=False)
    summary = rescue.evaluate(run, tmp_path / "output", mode="real_proposals", backend=rescue.THRESHOLD_BACKEND, endpoint_classification=artifact)
    assert summary["evaluator_artifact_available"] is True
    assert summary["evaluator_no_mask_near_prediction_rows"] == 1
    assert summary["rows_parsed_successfully"] == 0
    assert summary["status"] == "no_valid_evaluator_rows"


def test_cellpose_preflight_fails_cleanly_when_backend_is_missing():
    try:
        _preflight_backend("cellpose_sam", device="cuda")
    except BackendUnavailable:
        return
    # A configured Cellpose environment is also valid; the test only asserts
    # that preflight does not silently return the threshold backend.
    assert _preflight_backend("cellpose_sam", device="cpu")["pretrained_model"] == "cpsam_v2"


def test_transform_graph_direct_and_composed_provenance():
    transforms = pd.DataFrame([
        {"day_a": "s0", "day_b": "s1", "z_intercept": 1, "z_scale": 1, "y_intercept": 2, "y_from_y": 1, "y_from_x": 0, "x_intercept": 3, "x_from_y": 0, "x_from_x": 1},
        {"day_a": "s1", "day_b": "s2", "z_intercept": 1, "z_scale": 1, "y_intercept": 2, "y_from_y": 1, "y_from_x": 0, "x_intercept": 3, "x_from_y": 0, "x_from_x": 1},
    ])
    graph = TransformGraph(transforms)
    direct = graph.project("s0", "s1", [10, 20, 30])
    composed = graph.project("s0", "s2", [10, 20, 30])
    assert direct is not None and direct[1] == "direct"
    assert composed is not None and composed[1] == "composed"
    assert np.allclose(direct[0], [9, 18, 27])
    assert np.allclose(composed[0], [8, 16, 24])
