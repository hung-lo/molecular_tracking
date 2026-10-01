"""Evaluate opt-in image registration against saved pairwise matches; never rerun production.

Input CSV columns: matching_dir, session_a, session_b. Output must be a new folder.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "matching"))
from affine_overlap_matcher import VoxelSpacing
from image_registration import (
    restricted_transform, prepare_images, fit_image_affine, fit_smooth_field,
    image_quality, match_transformed_masks, correspondence_changes, guard_reasons,
)
from roi_track_graph import build_cycle_consistency_tables


def _pair(table, a, b):
    return table[(table.day_a == a) & (table.day_b == b)].copy()


def evaluate_pair(job):
    started = time.monotonic()
    matching = Path(job["matching_dir"]).resolve()
    a, b = job["session_a"], job["session_b"]
    out = Path(job["output"])
    manifest = pd.read_csv(matching / "session_manifest_resolved.csv").set_index("session_id")
    features = pd.read_csv(matching / "roi_features.csv")
    fa = features[features.session_id == a].set_index("label", drop=False)
    fb = features[features.session_id == b].set_index("label", drop=False)
    if fa.empty or fb.empty:
        raise ValueError(f"Missing features for {a} / {b}")
    paths = {f"{name}_{side}":Path(manifest.loc[session, column]).resolve()
             for side, session in [("a",a),("b",b)]
             for name,column in [("red","red_image_path"),("mask","mask_path")]}
    source_run = matching.parent
    source_sessions = {p.parents[2] for p in paths.values()}
    for source in [source_run, *source_sessions]:
        if out.is_relative_to(source):
            raise ValueError("Diagnostic output cannot be inside a source run or session.")
    # Confirm feature voxel units rather than assume one rig's nominal spacing.
    spacing_values = []
    for axis in "zyx":
        estimates = []
        for f in (fa, fb):
            nonzero = f[f[f"centroid_{axis}"] > 0]
            estimates.extend((nonzero[f"centroid_{axis}_um"] / nonzero[f"centroid_{axis}"]).tolist())
        if not estimates or not np.allclose(estimates, estimates[0], rtol=1e-5):
            raise ValueError("Feature voxel spacings are missing or inconsistent.")
        spacing_values.append(float(np.median(estimates)))
    spacing = VoxelSpacing(*spacing_values)
    tr = _pair(pd.read_csv(matching / "pairwise_transforms.csv"), a, b)
    if len(tr) != 1:
        raise ValueError("Expected one saved pair transform.")
    baseline_transform = restricted_transform(tr.iloc[0])
    base_file = matching / "pairwise_matches_graph.csv"
    if not base_file.exists():
        base_file = matching / "pairwise_matches_balanced.csv"
    base_matches = _pair(pd.read_csv(base_file), a, b)
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"matching_dir":str(matching), "session_a":a, "session_b":b,
                  "spacing":asdict(spacing), "smoothing_um":job["smoothing_um"],
                  "baseline":str(base_file),
                  "inputs":{k:{"path":str(p), "size":p.stat().st_size,
                               "mtime_ns":p.stat().st_mtime_ns} for k,p in paths.items()}}
    (out / "inputs.json").write_text(json.dumps(provenance, indent=2))
    masks = [tifffile.imread(paths["mask_"+side]) for side in "ab"]
    reds = [tifffile.imread(paths["red_"+side]) for side in "ab"]
    if any(v.shape != masks[0].shape for v in masks+reds):
        raise ValueError("Image/mask grids differ.")
    for mask, f in zip(masks, [fa, fb]):
        labels, counts = np.unique(mask, return_counts=True)
        areas = pd.Series(counts[labels>0], index=labels[labels>0])
        if set(areas.index) != set(f.index) or not np.allclose(areas.reindex(f.index), f.area_voxels):
            raise ValueError("Saved features no longer match source segmentation.")
    image_a, image_b, step = prepare_images(*reds, spacing)
    del reds
    rows = []; matches_by_stage = {"baseline":base_matches}; models = {"baseline":baseline_transform}
    baseline = {"stage":"baseline", "n_matches":len(base_matches),
                "inverse_error_max_um":0., **image_quality(image_a,image_b,baseline_transform,step,spacing),
                **correspondence_changes(base_matches,base_matches)}
    baseline.update(sym_pct=200*len(base_matches)/(len(fa)+len(fb)), n_a=len(fa), n_b=len(fb),
                    fallback_reasons="", eligible=True)
    rows.append(baseline)
    base_matches.to_csv(out / "baseline_matches.csv", index=False)
    fit_info = {}
    for stage in ["recomputed_overlap", "image_affine", "smooth_local"]:
        if stage == "recomputed_overlap":
            transform = baseline_transform
        elif stage == "image_affine":
            transform, fit_info = fit_image_affine(image_a,image_b,baseline_transform,step)
        else:
            transform = fit_smooth_field(image_a,image_b,models["image_affine"],step,spacing,
                                         smoothing_um=job["smoothing_um"])
        candidates, matches, inverse_error = match_transformed_masks(*masks,fa,fb,transform,spacing)
        quality = image_quality(image_a,image_b,transform,step,spacing)
        changes = correspondence_changes(base_matches,matches)
        row = {"stage":stage,"n_matches":len(matches),"n_candidates":len(candidates),
               "n_pass":int(candidates.balanced_rule.sum()),
               "sym_pct":200*len(matches)/(len(fa)+len(fb)),"n_a":len(fa),"n_b":len(fb),
               "inverse_error_max_um":inverse_error, **quality, **changes}
        reasons = guard_reasons(row,baseline)
        if stage != "recomputed_overlap" and not fit_info["optimizer_success"]:
            reasons.append("affine_optimizer_not_converged")
        previous = baseline if stage != "smooth_local" else rows[-1]
        if stage == "smooth_local" and quality["heldout_ncc"] < previous["heldout_ncc"]+.005:
            reasons.append("local_field_has_no_meaningful_image_gain")
        row.update(eligible=not reasons, fallback_reasons=";".join(reasons))
        rows.append(row); models[stage]=transform; matches_by_stage[stage]=matches
        matches.to_csv(out / f"{stage}_matches.csv", index=False)
        candidates.to_csv(out / f"{stage}_candidates.csv", index=False)
        arrays={"matrix":transform.matrix,"offset":transform.offset}
        if transform.flow is not None:
            arrays.update(flow=transform.flow,flow_step=transform.flow_step)
        np.savez_compressed(out / f"{stage}_transform.npz", **arrays)
        print(f"{a} -> {b}: {stage} {row['sym_pct']:.2f}% (guard: {row['fallback_reasons'] or 'pass'})", flush=True)
    # Prefer the most refined eligible stage. Never union incompatible assignments.
    selected = next(row["stage"] for row in reversed(rows) if row["eligible"])
    chosen = matches_by_stage[selected].copy()
    chosen["registration_stage"] = selected
    chosen.to_csv(out / "guarded_matches.csv", index=False)
    for row in rows:
        row.update(session_a=a,session_b=b,matching_dir=str(matching),selected=selected,
                   is_selected=row["stage"] == selected)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "registration_fit.json").write_text(json.dumps(fit_info, indent=2))
    (out / "completion.json").write_text(json.dumps({"elapsed_seconds":time.monotonic()-started,
                                                    "selected":selected}, indent=2))
    return rows


def audit_cycles(output):
    """Check experimental A-B-C paths against saved high-confidence A-C pairs.

    The A-C reference is unchanged production evidence, not identity ground truth.
    Reuse the existing cycle audit rather than construct experimental tracks.
    """
    output = Path(output)
    pairs = pd.read_csv(output / "pairs.csv").reset_index(names="pair_index")
    summaries = []; details = []
    for directory, group in pairs.groupby("matching_dir", sort=False):
        reference = pd.read_csv(Path(directory) / "pairwise_matches_high.csv")
        for ab in group.itertuples():
            for bc in group[group.session_a == ab.session_b].itertuples():
                days = [ab.session_a, ab.session_b, bc.session_b]
                ac = _pair(reference, days[0], days[2])
                if ac.empty:
                    continue
                for stage in ["baseline", "recomputed_overlap", "image_affine", "smooth_local", "guarded"]:
                    tables = {(days[0],days[1]):pd.read_csv(output/f"pair_{ab.pair_index:03d}"/f"{stage}_matches.csv"),
                              (days[1],days[2]):pd.read_csv(output/f"pair_{bc.pair_index:03d}"/f"{stage}_matches.csv"),
                              (days[0],days[2]):ac}
                    summary, detail = build_cycle_consistency_tables(
                        days, tables, pd.DataFrame(columns=["track_uid"]), stage)
                    summary["matching_dir"] = directory
                    summary["reference"] = "saved_production_high_A_to_C"
                    summaries.append(summary); details.append(detail)
    if summaries:
        pd.concat(summaries,ignore_index=True).to_csv(output/"cycle_audit_summary.csv",index=False)
        pd.concat(details,ignore_index=True).to_csv(output/"cycle_audit_details.csv",index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing-um", type=float, default=15.)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1 or not np.isfinite(args.smoothing_um) or args.smoothing_um < 5:
        parser.error("workers must be positive and smoothing-um must be >=5.")
    pairs = pd.read_csv(args.pairs, dtype=str)
    if pairs.empty or not {"matching_dir","session_a","session_b"}.issubset(pairs.columns):
        parser.error("pairs CSV needs matching_dir, session_a, session_b and at least one row.")
    args.output.mkdir(parents=True, exist_ok=False)
    pairs.to_csv(args.output / "pairs.csv", index=False)
    code_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__),ROOT/'matching/image_registration.py',ROOT/'matching/affine_overlap_matcher.py']}
    (args.output / "settings.json").write_text(json.dumps({"smoothing_um":args.smoothing_um,
        "workers":args.workers,"code_sha256":code_hashes,"pairwise_only":True,
        "warning":"No identity ground truth. Guarded retention/counts do not prove absence of harm."},indent=2))
    jobs = [{**row,"output":str((args.output/f"pair_{i:03d}").resolve()),
             "smoothing_um":args.smoothing_um} for i,row in enumerate(pairs.to_dict("records"))]
    metrics=[]
    if args.workers == 1:
        for job in jobs:
            metrics.extend(evaluate_pair(job))
            pd.DataFrame(metrics).to_csv(args.output / "summary.csv",index=False)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(evaluate_pair,jobs):
                metrics.extend(result)
                pd.DataFrame(metrics).to_csv(args.output / "summary.csv",index=False)
    audit_cycles(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
