"""Evaluate opt-in image registration against saved pairwise matches; never rerun production.

Input CSV columns: matching_dir, session_a, session_b. Output must be a new folder.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
    image_quality, match_transformed_masks, correspondence_changes, identity_conflicts,
    classify_identity_conflicts, identity_guard_reasons, guard_reasons,
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
    provenance = {"mouse":job.get("mouse", ""), "matching_dir":str(matching),
                  "session_a":a, "session_b":b,
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
    baseline = {"mouse":job.get("mouse", ""), "stage":"baseline", "smoothing_um":job["smoothing_um"],
                "n_matches":len(base_matches),
                "inverse_error_max_um":0., **image_quality(image_a,image_b,baseline_transform,step,spacing),
                **correspondence_changes(base_matches,base_matches)}
    baseline.update(sym_pct=200*len(base_matches)/(len(fa)+len(fb)), n_a=len(fa), n_b=len(fb),
                    direct_matches=len(base_matches), mapping_overlap_pct=200*len(base_matches)/(len(fa)+len(fb)),
                    image_correlation=baseline["heldout_ncc"], round_trip_error_max_um=0.,
                    delta_image_correlation_vs_affine=np.nan, cycle_consistency_pct=np.nan,
                    comparable_cycle_paths=0, identity_guard_reasons="", fallback_reasons="", eligible=True)
    rows.append(baseline)
    base_matches.to_csv(out / "baseline_matches.csv", index=False)
    identity_conflicts(base_matches, base_matches).to_csv(out / "baseline_identity_conflicts.csv", index=False)
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
        conflicts = identity_conflicts(base_matches, matches)
        row = {"stage":stage,"n_matches":len(matches),"n_candidates":len(candidates),
               "mouse":job.get("mouse", ""), "smoothing_um":job["smoothing_um"],
               "n_pass":int(candidates.balanced_rule.sum()),
               "sym_pct":200*len(matches)/(len(fa)+len(fb)),"n_a":len(fa),"n_b":len(fb),
               "inverse_error_max_um":inverse_error, **quality, **changes}
        row.update(direct_matches=len(matches), mapping_overlap_pct=row["sym_pct"],
                   image_correlation=row["heldout_ncc"], round_trip_error_max_um=inverse_error,
                   delta_image_correlation_vs_affine=np.nan,
                   cycle_consistency_pct=np.nan, comparable_cycle_paths=0,
                   identity_guard_reasons="")
        reasons = guard_reasons(row,baseline)
        if stage != "recomputed_overlap" and not fit_info["optimizer_success"]:
            reasons.append("affine_optimizer_not_converged")
        previous = baseline if stage != "smooth_local" else rows[-1]
        if stage == "smooth_local" and quality["heldout_ncc"] < previous["heldout_ncc"]+.005:
            reasons.append("local_field_has_no_meaningful_image_gain")
        if stage == "image_affine":
            affine_ncc = row["heldout_ncc"]
        elif stage == "smooth_local":
            affine_ncc = next(item["heldout_ncc"] for item in rows if item["stage"] == "image_affine")
            row["delta_image_correlation_vs_affine"] = row["heldout_ncc"] - affine_ncc
        row.update(eligible=not reasons, fallback_reasons=";".join(reasons))
        rows.append(row); models[stage]=transform; matches_by_stage[stage]=matches
        matches.to_csv(out / f"{stage}_matches.csv", index=False)
        conflicts.to_csv(out / f"{stage}_identity_conflicts.csv", index=False)
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
                   guard_pass=bool(row["eligible"]), guard_failure_reason=row["fallback_reasons"],
                   is_selected=row["stage"] == selected)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "registration_fit.json").write_text(json.dumps(fit_info, indent=2))
    (out / "completion.json").write_text(json.dumps({"elapsed_seconds":time.monotonic()-started,
                                                    "selected":selected}, indent=2))
    return rows


STAGES = ["baseline", "recomputed_overlap", "image_affine", "smooth_local"]


def _pair_map(pairs):
    return {(str(row.matching_dir), str(row.session_a), str(row.session_b)): int(row.pair_index)
            for row in pairs.itertuples()}


def _independent_cycle_contexts(pairs):
    """Return only independently evaluated A-B/B-C/A-C contexts."""
    lookup = _pair_map(pairs)
    contexts = {}
    for ab in pairs.itertuples():
        for bc in pairs.itertuples():
            if str(bc.matching_dir) != str(ab.matching_dir) or str(bc.session_a) != str(ab.session_b):
                continue
            ac_index = lookup.get((str(ab.matching_dir), str(ab.session_a), str(bc.session_b)))
            if ac_index is None:
                continue
            contexts[int(ab.pair_index)] = (int(bc.pair_index), ac_index, "ab")
            contexts[int(bc.pair_index)] = (int(ab.pair_index), ac_index, "bc")
            break
    return contexts


def _cycle_metrics(ab, bc, ac):
    map_ab = {int(row.label_a): int(row.label_b) for row in ab.itertuples(index=False)}
    map_bc = {int(row.label_a): int(row.label_b) for row in bc.itertuples(index=False)}
    map_ac = {int(row.label_a): int(row.label_b) for row in ac.itertuples(index=False)}
    composed = {a: map_bc[b] for a, b in map_ab.items() if b in map_bc}
    comparable = {a: c for a, c in composed.items() if a in map_ac}
    agree = sum(map_ac[a] == c for a, c in comparable.items())
    return 100 * agree / len(comparable) if comparable else np.nan, len(comparable)


def _annotate_conflicts(conflicts, row):
    conflicts = conflicts.copy()
    fields = {
        "mouse": row.get("mouse", ""), "session_A": row["session_a"], "session_B": row["session_b"],
        "ROI_A": conflicts.get("roi_a", pd.Series(dtype="Int64")),
        "old_ROI_B": conflicts.get("old_roi_b", pd.Series(dtype="Int64")),
        "new_ROI_B": conflicts.get("new_roi_b", pd.Series(dtype="Int64")),
        "old_distance": conflicts.get("old_distance_um", pd.Series(dtype=float)),
        "new_distance": conflicts.get("new_distance_um", pd.Series(dtype=float)),
        "old_Dice": conflicts.get("old_dice", pd.Series(dtype=float)),
        "new_Dice": conflicts.get("new_dice", pd.Series(dtype=float)),
        "old_ambiguity": conflicts.get("old_ambiguity", pd.Series(dtype=float)),
        "new_ambiguity": conflicts.get("new_ambiguity", pd.Series(dtype=float)),
        "old_local_image_support": conflicts.get("old_local_image_support", pd.Series(dtype=float)),
        "new_local_image_support": conflicts.get("new_local_image_support", pd.Series(dtype=float)),
    }
    for name, value in fields.items():
        conflicts[name] = value
    return conflicts


def finalize_guarded(output):
    """Apply identity-audit evidence after all requested pair evaluations exist."""
    output = Path(output)
    pairs = pd.read_csv(output / "pairs.csv").reset_index(names="pair_index")
    contexts = _independent_cycle_contexts(pairs)
    all_rows = []
    for pair in pairs.itertuples():
        pair_dir = output / f"pair_{pair.pair_index:03d}"
        metrics = pd.read_csv(pair_dir / "metrics.csv")
        for column in ("identity_guard_reasons", "fallback_reasons", "guard_failure_reason"):
            if column in metrics:
                metrics[column] = metrics[column].fillna("").astype(object)
        context = contexts.get(int(pair.pair_index))
        for stage in STAGES[1:]:
            mask = metrics.stage == stage
            if not mask.any():
                continue
            conflicts = pd.read_csv(pair_dir / f"{stage}_identity_conflicts.csv")
            cycle_pct = np.nan; comparable = 0
            if context is not None:
                bridge_index, direct_index, role = context
                bridge_dir = output / f"pair_{bridge_index:03d}"
                direct_dir = output / f"pair_{direct_index:03d}"
                bridge = pd.read_csv(bridge_dir / f"{stage}_matches.csv")
                direct = pd.read_csv(direct_dir / f"{stage}_matches.csv")
                if role == "ab":
                    conflicts = classify_identity_conflicts(conflicts, bridge, direct, pair_role="ab")
                    ab = pd.read_csv(pair_dir / f"{stage}_matches.csv")
                    cycle_pct, comparable = _cycle_metrics(ab, bridge, direct)
                else:
                    conflicts = classify_identity_conflicts(conflicts, bridge, direct, pair_role="bc")
                    ab = pd.read_csv(bridge_dir / f"{stage}_matches.csv")
                    bc = pd.read_csv(pair_dir / f"{stage}_matches.csv")
                    cycle_pct, comparable = _cycle_metrics(ab, bc, direct)
            conflicts = _annotate_conflicts(conflicts, metrics.loc[mask].iloc[0])
            conflicts.to_csv(pair_dir / f"{stage}_identity_conflicts.csv", index=False)
            row = metrics.loc[mask].iloc[0].to_dict()
            identity_reasons = identity_guard_reasons(row, conflicts)
            existing = [part for part in str(row.get("fallback_reasons", "")).split(";") if part]
            reasons = list(dict.fromkeys(existing + identity_reasons))
            metrics.loc[mask, "cycle_consistency_pct"] = cycle_pct
            metrics.loc[mask, "comparable_cycle_paths"] = comparable
            metrics.loc[mask, "identity_guard_reasons"] = ";".join(identity_reasons)
            metrics.loc[mask, "fallback_reasons"] = ";".join(reasons)
            metrics.loc[mask, "eligible"] = not reasons
        selected = next(row.stage for row in metrics.iloc[::-1].itertuples() if bool(row.eligible))
        selected_matches = pd.read_csv(pair_dir / f"{selected}_matches.csv").copy()
        selected_matches["registration_stage"] = selected
        selected_matches.to_csv(pair_dir / "guarded_matches.csv", index=False)
        selected_conflicts = pd.read_csv(pair_dir / f"{selected}_identity_conflicts.csv") if selected != "baseline" else identity_conflicts(selected_matches, selected_matches)
        selected_conflicts.to_csv(pair_dir / "guarded_identity_conflicts.csv", index=False)
        metrics["selected"] = selected
        metrics["is_selected"] = metrics.stage == selected
        metrics["guard_pass"] = metrics.eligible.astype(bool)
        metrics["guard_failure_reason"] = metrics.fallback_reasons
        metrics.to_csv(pair_dir / "metrics.csv", index=False)
        (pair_dir / "completion.json").write_text(json.dumps({"selected":selected}, indent=2))
        all_rows.append(metrics)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(output / "summary.csv", index=False)


def audit_cycles(output):
    """Report cycles, preferring independently evaluated A-C pairs when present."""
    output = Path(output)
    pairs = pd.read_csv(output / "pairs.csv").reset_index(names="pair_index")
    lookup = _pair_map(pairs)
    summaries = []; details = []
    for directory, group in pairs.groupby("matching_dir", sort=False):
        reference = pd.read_csv(Path(directory) / "pairwise_matches_high.csv")
        for ab in group.itertuples():
            for bc in group[group.session_a == ab.session_b].itertuples():
                days = [ab.session_a, ab.session_b, bc.session_b]
                ac_index = lookup.get((str(directory), str(days[0]), str(days[2])))
                if ac_index is not None:
                    ac_dir = output / f"pair_{ac_index:03d}"
                    source = "independently_evaluated_A_to_C"
                else:
                    ac_dir = None
                    source = "saved_production_high_A_to_C"
                for stage in STAGES + ["guarded"]:
                    ac = pd.read_csv(ac_dir / f"{stage}_matches.csv") if ac_dir else _pair(reference, days[0], days[2])
                    if ac.empty:
                        continue
                    tables = {(days[0],days[1]):pd.read_csv(output/f"pair_{ab.pair_index:03d}"/f"{stage}_matches.csv"),
                              (days[1],days[2]):pd.read_csv(output/f"pair_{bc.pair_index:03d}"/f"{stage}_matches.csv"),
                              (days[0],days[2]):ac}
                    summary, detail = build_cycle_consistency_tables(
                        days, tables, pd.DataFrame(columns=["track_uid"]), stage)
                    summary["matching_dir"] = directory
                    summary["reference"] = source
                    summaries.append(summary); details.append(detail)
    if summaries:
        pd.concat(summaries,ignore_index=True).to_csv(output/"cycle_audit_summary.csv",index=False)
        pd.concat(details,ignore_index=True).to_csv(output/"cycle_audit_details.csv",index=False)


def cross_smoothing_identity(output, scales):
    """Compare actual smooth-local assignments across a completed scale grid."""
    output = Path(output)
    pairs = pd.read_csv(output / "pairs.csv").reset_index(names="pair_index")
    summary_rows = []; detail_rows = []
    scale_pairs = [(a, b) for index, a in enumerate(scales) for b in scales[index + 1:]]
    for pair in pairs.itertuples():
        maps = {}
        for scale in scales:
            matches = pd.read_csv(output/f"smoothing_{scale:g}um"/f"pair_{pair.pair_index:03d}"/"smooth_local_matches.csv")
            maps[scale] = {int(row.label_a):int(row.label_b) for row in matches.itertuples(index=False)}
        union = set().union(*(set(mapping) for mapping in maps.values()))
        common = set.intersection(*(set(mapping) for mapping in maps.values()))
        details = []
        for roi_a in sorted(union):
            values = {scale:maps[scale].get(roi_a, pd.NA) for scale in scales}
            present = [int(value) for value in values.values() if pd.notna(value)]
            counts = Counter(present)
            modal_roi_b, modal_count = counts.most_common(1)[0] if counts else (pd.NA, 0)
            details.append({
                "mouse":getattr(pair, "mouse", ""), "session_a":pair.session_a, "session_b":pair.session_b,
                "roi_a":roi_a, **{f"roi_b_{scale:g}um":value for scale,value in values.items()},
                "n_scales_present":len(present), "modal_roi_b":modal_roi_b, "modal_count":modal_count,
                "identical_all_scales":len(present) == len(scales) and len(counts) == 1,
                "identical_in_at_least_3_scales":modal_count >= 3,
                "roi_b_changes_with_smoothing":len(counts) > 1,
                "single_smoothing_only":len(present) == 1,
            })
        detail = pd.DataFrame(details)
        detail_rows.append(detail)
        row = {
            "mouse":getattr(pair, "mouse", ""), "session_a":pair.session_a, "session_b":pair.session_b,
            **{f"n_matches_{scale:g}um":len(maps[scale]) for scale in scales},
            "n_roi_a_union":len(union), "n_roi_a_all_scales":len(common),
            "n_exact_assignments_all_scales":int(detail.identical_all_scales.sum()),
            "fraction_identical_all_scales":float(detail.identical_all_scales.sum()/max(len(common),1)),
            "fraction_identical_in_at_least_3_scales":float(detail.identical_in_at_least_3_scales.mean()),
            "fraction_roi_b_changes_with_smoothing":float(detail.roi_b_changes_with_smoothing.mean()),
            "fraction_single_smoothing_only":float(detail.single_smoothing_only.mean()),
        }
        for left, right in scale_pairs:
            both = set(maps[left]) & set(maps[right])
            exact = sum(maps[left][roi] == maps[right][roi] for roi in both)
            row[f"n_common_{left:g}_vs_{right:g}um"] = len(both)
            row[f"assignment_agreement_{left:g}_vs_{right:g}um"] = exact/max(len(both),1)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output/"cross_smoothing_identity_summary.csv",index=False)
    pd.concat(detail_rows,ignore_index=True).to_csv(output/"cross_smoothing_identity_details.csv",index=False)


def _run_once(pairs, output, smoothing_um, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pairs.to_csv(output / "pairs.csv", index=False)
    code_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__),ROOT/'matching/image_registration.py',ROOT/'matching/affine_overlap_matcher.py']}
    (output / "settings.json").write_text(json.dumps({"smoothing_um":smoothing_um,
        "workers":workers,"code_sha256":code_hashes,"pairwise_only":True,
        "warning":"No identity ground truth. Guarded retention/counts do not prove absence of harm."},indent=2))
    jobs = [{**row,"output":str((output/f"pair_{i:03d}").resolve()),
             "smoothing_um":smoothing_um} for i,row in enumerate(pairs.to_dict("records"))]
    metrics=[]
    if workers == 1:
        for job in jobs:
            metrics.extend(evaluate_pair(job))
            pd.DataFrame(metrics).to_csv(output / "summary.csv",index=False)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(evaluate_pair,jobs):
                metrics.extend(result)
                pd.DataFrame(metrics).to_csv(output / "summary.csv",index=False)
    finalize_guarded(output)
    audit_cycles(output)
    return pd.read_csv(output / "summary.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing-um", type=float, default=15.)
    parser.add_argument("--smoothing-grid", type=float, nargs="+", default=None,
                        help="Run independent local registrations at each listed smoothing scale.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    scales = args.smoothing_grid or [args.smoothing_um]
    if (args.workers < 1 or any(not np.isfinite(scale) or scale < 5 for scale in scales)
            or len(set(scales)) != len(scales)):
        parser.error("workers must be positive, smoothing scales must be finite and >=5, and unique.")
    pairs = pd.read_csv(args.pairs, dtype=str)
    if pairs.empty or not {"matching_dir","session_a","session_b"}.issubset(pairs.columns):
        parser.error("pairs CSV needs matching_dir, session_a, session_b and at least one row.")
    if len(scales) == 1:
        _run_once(pairs, args.output, scales[0], args.workers)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    pairs.to_csv(args.output / "pairs.csv", index=False)
    summaries = []
    for scale in scales:
        scale_dir = args.output / f"smoothing_{scale:g}um"
        summary = _run_once(pairs, scale_dir, scale, args.workers)
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(args.output / "robustness_summary.csv", index=False)
    cross_smoothing_identity(args.output, scales)
    (args.output / "settings.json").write_text(json.dumps({
        "smoothing_grid_um": scales, "workers": args.workers,
        "pairwise_only": True, "summary": "robustness_summary.csv",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
