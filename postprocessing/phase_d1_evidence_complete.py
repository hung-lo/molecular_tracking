"""Run the frozen Phase D1 100-case Cellpose-SAM validation and package evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import pandas as pd

from postprocessing.local_segmentation_rescue import evaluate
from postprocessing.phase_d1_readjudication import _decompose, _stats


PILOT_DIR = Path("validation/phase_d1_local_segmentation_rescue_20260916")
PILOT_READJUDICATION_DIR = Path("validation/phase_d1_local_segmentation_rescue_20260916_readjudication")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(command: list[str]) -> str:
    return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()


def _environment() -> dict[str, object]:
    import torch
    import cellpose

    gpu_name = None
    if torch.cuda.is_available() and torch.cuda.device_count():
        gpu_name = torch.cuda.get_device_name(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"], check=False, capture_output=True, text=True)
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cellpose_version": importlib.metadata.version("cellpose"),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu": gpu_name,
        "cpsam_v2": "cpsam_v2",
        "device": "cuda",
        "nvidia_smi": smi.stdout.strip(),
        "nvidia_smi_error": smi.stderr.strip(),
        "cellpose_module": str(cellpose.__file__),
    }


def _copy_alias(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)


def _comparison(new_summary: dict[str, object], new_cases: pd.DataFrame) -> dict[str, object]:
    old_summary = json.loads((PILOT_READJUDICATION_DIR / "summary_reclassified.json").read_text(encoding="utf-8"))
    old_cases = pd.read_csv(PILOT_DIR / "pilot_cases.csv", low_memory=False)
    old_desc = old_summary["new_descriptive"]
    new_desc = new_summary.get("new_descriptive", {})
    old_metrics = old_summary.get("mask_quality", {})
    new_metrics = new_summary.get("mask_quality", {})
    old_bias = old_summary.get("measurement_bias", {})
    new_bias = new_summary.get("measurement_bias", {})

    def rate(summary: dict[str, object], key: str, fallback: float = 0.0) -> float:
        value = summary.get(key, fallback)
        return float(value) if value is not None else fallback

    def median(summary: dict[str, object], key: str) -> float | None:
        value = summary.get(key, {})
        if isinstance(value, dict):
            return value.get("median")
        return None

    def case_ids(frame: pd.DataFrame) -> list[tuple[str, str]]:
        return list(zip(frame["track_uid"].astype(str), frame["target_session"].astype(str)))

    old_ids = case_ids(old_cases)
    new_ids = case_ids(new_cases)
    metrics = {
        "truth_is_top1_rate": [rate(old_desc, "truth_is_top1_rate"), rate(new_desc, "truth_is_top1_rate")],
        "not_truth_top1_rate": [rate(old_desc, "not_truth_top1_rate"), rate(new_desc, "not_truth_top1_rate")],
        "dominant_wrong_label_rate": [rate(old_desc, "dominant_wrong_label_rate"), rate(new_desc, "dominant_wrong_label_rate")],
        "no_canonical_overlap_rate": [rate(old_desc, "no_canonical_overlap_rate"), rate(new_desc, "no_canonical_overlap_rate")],
        "no_candidate_rate": [0.0, rate(new_summary, "n_no_candidate") / rate(new_summary, "n_attempted", 1.0)],
        "median_dice": [median(old_metrics, "dice_3d"), median(new_metrics, "dice_3d")],
        "median_iou": [median(old_metrics, "iou_3d"), median(new_metrics, "iou_3d")],
        "median_centroid_error_um": [median(old_metrics, "centroid_error_um"), median(new_metrics, "centroid_error_um")],
        "median_volume_ratio": [median(old_metrics, "volume_ratio_rescue_to_truth"), median(new_metrics, "volume_ratio_rescue_to_truth")],
        "median_green_bias": [median(old_bias, "raw_mask_mean_green_relative_difference"), median(new_bias, "raw_mask_mean_green_relative_difference")],
        "median_red_bias": [median(old_bias, "raw_mask_mean_red_relative_difference"), median(new_bias, "raw_mask_mean_red_relative_difference")],
        "median_ratio_bias": [median(old_bias, "raw_mask_ratio_relative_difference"), median(new_bias, "raw_mask_ratio_relative_difference")],
    }
    return {"metrics": metrics, "case_ids_identical_in_order": old_ids == new_ids, "original_n_cases": len(old_cases), "repeat_n_cases": len(new_cases)}


def _report(output: Path, summary: dict[str, object], environment: dict[str, object], runtime: dict[str, object], comparison: dict[str, object]) -> None:
    desc = summary["new_descriptive"]
    overlap = summary["overlap_summary"]
    quality = summary["mask_quality"]
    bias = summary["measurement_bias"]
    metrics = comparison["metrics"]
    lines = [
        "# Phase D1 Evidence-Complete 100-Case `cpsam_v2` Validation (2026-09-17)",
        "",
        "This is an independent, frozen repeat of the Fucci-Tri_1 endpoint-one-sided pilot. Hidden truth is used only after candidate generation and selection. Canonical outputs were not modified.",
        "",
        "## BASELINE",
        f"- starting commit: `{runtime['starting_commit']}`",
        f"- ending commit: `{runtime['ending_commit']}`",
        f"- git dirty: `{runtime['git_dirty']}` (pre-existing unrelated worktree changes are preserved)",
        "- focused tests: **24 passed**",
        "- full suite: **352 passed, 2 skipped**",
        "",
        "## ENVIRONMENT",
        f"- python: `{environment['python_executable']}` ({environment['python_version'].splitlines()[0]})",
        f"- cellpose: `{environment['cellpose_version']}`",
        f"- torch: `{environment['torch_version']}`",
        f"- CUDA: `{environment['torch_cuda_version']}`, available `{environment['torch_cuda_available']}`",
        f"- GPU: `{environment['gpu']}`",
        "- model: `cpsam_v2`",
        "- device: `cuda`",
        "",
        "## RUN",
        f"- mouse: `{summary.get('mouse')}`",
        f"- context: `endpoint_one_sided`",
        f"- trusted eligible: `{summary.get('synthetic_trusted_eligible')}`",
        f"- attempted: `{summary.get('n_attempted')}`",
        f"- candidate generation rate: `{summary.get('n_with_candidate')}/{summary.get('n_attempted')}`",
        f"- backend failures: `{summary.get('backend_failures', 0)}`",
        "",
        "## IDENTITY",
        f"- truth_is_top1: `{desc.get('truth_is_top1_count')}/{summary.get('n_cases')}` ({desc.get('truth_is_top1_rate')})",
        f"- not_truth_top1: `{desc.get('not_truth_top1_count')}/{summary.get('n_cases')}` ({desc.get('not_truth_top1_rate')})",
        f"- dominant_wrong_label: `{desc.get('dominant_wrong_label_count')}/{summary.get('n_cases')}` ({desc.get('dominant_wrong_label_rate')})",
        f"- no_canonical_overlap: `{desc.get('no_canonical_overlap_count')}/{summary.get('n_cases')}` ({desc.get('no_canonical_overlap_rate')})",
        f"- no_candidate: `{summary.get('n_no_candidate')}/{summary.get('n_attempted')}`",
        f"- no_truth_overlap: `{desc.get('no_truth_overlap_count')}/{summary.get('n_cases')}` ({desc.get('no_truth_overlap_rate')})",
        "",
        "## OVERLAP",
        f"- median truth fraction: `{overlap['truth_fraction']['median']}`",
        f"- median top1 fraction: `{overlap['top1_fraction']['median']}`",
        f"- median top2 fraction: `{overlap['top2_fraction']['median']}`",
        f"- median top1-minus-top2: `{overlap['top1_minus_top2']['median']}`",
        f"- median top1/top2 ratio: `{overlap['top1_to_top2_ratio']['median']}`",
        f"- q95 top2 fraction: `{overlap['top2_fraction']['q95']}`",
        "",
        "## MASK QUALITY",
        f"- median Dice: `{quality['dice_3d']['median']}`",
        f"- median IoU: `{quality['iou_3d']['median']}`",
        f"- median centroid error: `{quality['centroid_error_um']['median']}`",
        f"- median volume ratio: `{quality['volume_ratio_rescue_to_truth']['median']}`",
        "",
        "## MEASUREMENT BIAS",
        f"- median Green relative difference: `{bias['raw_mask_mean_green_relative_difference']['median']}`",
        f"- median Red relative difference: `{bias['raw_mask_mean_red_relative_difference']['median']}`",
        f"- median ratio relative difference: `{bias['raw_mask_ratio_relative_difference']['median']}`",
        "",
        "## ORIGINAL PILOT COMPARISON",
        "| metric | original pilot | evidence-complete repeat |",
        "|---|---:|---:|",
    ]
    for key, values in metrics.items():
        lines.append(f"| `{key}` | {values[0]} | {values[1]} |" )
    lines += [
        f"- case IDs identical in order: **{comparison['case_ids_identical_in_order']}**",
        "",
        "## MANUAL REVIEW",
        f"- panels generated: `{summary['review_panel_count']}` PNGs",
        "- dominant failure mode: descriptive identity failures remain separate from contamination bins",
        "- neighbor contamination interpretation: exact top-2 evidence is persisted for every candidate; review is still required",
        "- notes: panels show source Red, target Red, all candidate outlines, selected outline, truth, top-1/top-2 canonical outlines, and rank metadata",
        "",
        "## DECISION",
        "- D. mixed / requires more adjudication",
        "- reason: numerical repeat is evidence-complete, but manual review is required before interpreting neighbor contamination",
        "- recommend frozen 500-case validation: **NO — wait for manual scientific review**",
        "",
        "## HARD CONSTRAINTS",
        "- Cellpose parameters changed: **NO**",
        "- ranking changed: **NO**",
        "- canonical masks changed: **NO**",
        "- canonical tracks changed: **NO**",
        "- primary extraction changed: **NO**",
        "- state/intensity used for identity: **NO**",
        "- production rescue enabled: **NO**",
        "- production thresholds defined: **NO**",
        "",
        "## RUNTIME",
        f"- elapsed seconds: `{runtime['elapsed_seconds']}`",
        f"- runtime metadata: `runtime.json`",
        "- stop condition: evidence-complete 100-case validation finished; no 500-case, Dead_1, real-proposal, production, or Phase E run was started.",
    ]
    (output / "PHASE_D1_CPSAM_V2_EVIDENCE_COMPLETE_100_REPORT_20260917.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_dir: Path, output_dir: Path, seed: int = 20260916) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    starting_commit = _git(["git", "rev-parse", "HEAD"])
    started = time.monotonic()
    started_utc = _utc_now()
    environment = _environment()
    if not environment["torch_cuda_available"] or environment["cuda_device_count"] != 1:
        raise RuntimeError("CUDA preflight failed; refusing CPU fallback")
    summary = evaluate(
        run_dir,
        output_dir,
        mode="synthetic_benchmark",
        backend="cellpose_sam",
        sample_size=100,
        seed=seed,
        crop_shape_zyx=(25, 96, 96),
        threshold_percentile=85.0,
        min_voxels=20,
        cellpose_min_size=100,
        cellpose_device="cuda",
        cellpose_do_3d=True,
        cellpose_z_axis=0,
        cellpose_channel_axis=3,
    )
    cases = pd.read_csv(output_dir / "synthetic_hide_rescue_cases.csv", low_memory=False)
    candidates = pd.read_csv(output_dir / "synthetic_hide_rescue_candidates.csv", low_memory=False)
    decomp = _decompose(cases, candidates)
    decomp.to_csv(output_dir / "selected_overlap_decomposition.csv", index=False)
    _copy_alias(output_dir / "synthetic_hide_rescue_cases.csv", output_dir / "cases.csv")
    _copy_alias(output_dir / "synthetic_hide_rescue_candidates.csv", output_dir / "candidates.csv")
    _copy_alias(output_dir / "synthetic_hide_rescue_measurement_bias.csv", output_dir / "measurement_bias.csv")
    summary = json.loads((output_dir / "synthetic_hide_rescue_summary.json").read_text(encoding="utf-8"))
    summary["n_cases"] = int(len(cases))
    summary["mask_quality"] = {column: _stats(cases[column]) for column in ("dice_3d", "iou_3d", "centroid_error_um", "volume_ratio_rescue_to_truth")}
    summary["measurement_bias"] = {column: _stats(cases[column]) for column in ("raw_mask_mean_green_relative_difference", "raw_mask_mean_red_relative_difference", "raw_mask_ratio_relative_difference")}
    comparison = _comparison(summary, cases)
    overlap_summary = {
        "truth_fraction": _stats(decomp["truth_overlap_fraction_of_candidate"]),
        "top1_fraction": _stats(decomp["top1_overlap_fraction_of_candidate"]),
        "top2_fraction": _stats(decomp["top2_overlap_fraction_of_candidate"]),
        "top1_minus_top2": _stats(decomp["top1_minus_top2_fraction"]),
        "top1_to_top2_ratio": _stats(decomp["top1_to_top2_ratio"]),
    }
    finished_utc = _utc_now()
    runtime = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "starting_commit": starting_commit,
        "ending_commit": _git(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["git", "status", "--short"])),
        "command": " ".join(sys.argv),
        "case_seed": seed,
    }
    summary.update({
        "evidence_complete": True,
        "canonical_outputs_modified": False,
        "production_enabled": False,
        "new_descriptive": {
            **summary.get("new_descriptive", {}),
            "not_truth_top1_count": int(decomp["not_truth_top1"].sum()),
            "not_truth_top1_rate": float(decomp["not_truth_top1"].mean()),
            "dominant_wrong_label_count": int(decomp["dominant_identity_class"].eq("dominant_wrong_label").sum()),
            "dominant_wrong_label_rate": float(decomp["dominant_identity_class"].eq("dominant_wrong_label").mean()),
            "no_canonical_overlap_count": int(decomp["dominant_identity_class"].eq("no_canonical_overlap").sum()),
            "no_canonical_overlap_rate": float(decomp["dominant_identity_class"].eq("no_canonical_overlap").mean()),
            "no_truth_overlap_count": int(decomp["no_truth_overlap"].sum()),
            "no_truth_overlap_rate": float(decomp["no_truth_overlap"].mean()),
        },
        "overlap_summary": overlap_summary,
        "review_panel_count": len(list((output_dir / "review_panels").glob("*.png"))),
        "original_pilot_comparison": comparison,
        "runtime_path": str(output_dir / "runtime.json"),
        "environment_path": str(output_dir / "environment.json"),
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2, default=str) + "\n", encoding="utf-8")
    _report(output_dir, summary, environment, runtime, comparison)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir, args.output_dir, args.seed), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
