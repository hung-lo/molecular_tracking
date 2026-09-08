#!/usr/bin/env python3
"""Compare color-state postprocessing runs without pooling their statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, _ROOT / "postprocessing", _ROOT / "plotting"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from plotting.fucci_color_state_plots import plot_comparison_distributions, plot_comparison_modal_fits, plot_comparison_state_percentages


def _label(color_dir: Path, log: dict[str, Any], index: int, labels: list[str] | None) -> str:
    if labels and index < len(labels):
        return labels[index]
    source = Path(log.get("source_master_run_dir", color_dir)).name
    if log.get("mouse_id"):
        return str(log["mouse_id"])
    if isinstance(log.get("project"), dict) and log["project"].get("mouse_id"):
        return str(log["project"]["mouse_id"])
    return source


def _parse_overrides(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--session-index values must be LABEL=INDEX")
        label, index = value.split("=", 1)
        result[label] = int(index)
    return result


def compare_color_state_runs(
    color_state_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    labels: list[str] | None = None,
    session_overrides: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not color_state_dirs:
        raise ValueError("At least one --color-state-dir is required")
    overrides = _parse_overrides(session_overrides or [])
    runs: list[dict[str, Any]] = []
    for index, value in enumerate(color_state_dirs):
        directory = Path(value).expanduser().resolve()
        log = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        label = _label(directory, log, index, labels)
        scored = pd.read_csv(directory / "normalization/matched_roi_color_state_all_observed.csv")
        occupancy = pd.read_csv(directory / "normalization/eclipse_state_occupancy_by_session.csv")
        fits = pd.read_csv(directory / "normalization/color_state_session_fits.csv")
        occupancy = occupancy.merge(fits[["session_id", "session_index"]], on="session_id", how="left", validate="one_to_one")
        median_low = float(occupancy["pct_strong_low"].median())
        if label in overrides:
            chosen = occupancy.loc[occupancy["session_index"].eq(overrides[label])]
            if chosen.empty:
                raise ValueError(f"No session index {overrides[label]} exists for {label}")
            chosen = chosen.iloc[0]
        else:
            chosen = occupancy.iloc[(occupancy["pct_strong_low"] - median_low).abs().argmin()]
        fit = fits.loc[fits["session_id"].astype(str).eq(str(chosen["session_id"]))].iloc[0]
        native_path = Path(log["source_master_run_dir"]) / "extraction/matched_session_population_roi_metrics.csv"
        native = pd.read_csv(native_path)
        runs.append({"label": label, "scored": scored, "occupancy": occupancy, "chosen": chosen, "fit": fit, "native": native})

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Comparison output already exists: {output}; use --overwrite")
        if not output.is_dir():
            raise ValueError(f"Comparison output is not a directory: {output}")
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True)
    distribution_tables = [(run["label"], run["scored"]) for run in runs]
    plot_comparison_distributions(distribution_tables, output / "comparison_color_z_distributions.png")
    plot_comparison_modal_fits([(run["label"], run["native"].loc[run["native"]["session_id"].astype(str).eq(str(run["chosen"]["session_id"]))], run["fit"]) for run in runs], output / "comparison_green_red_sd_zones.png")

    occupancy_rows = []
    for run in runs:
        selected = run["occupancy"].loc[run["occupancy"]["session_id"].astype(str).eq(str(run["chosen"]["session_id"]))].iloc[0]
        for state in ("strong_low", "low_transition", "middle", "high_transition", "strong_high"):
            occupancy_rows.append({"label": run["label"], "session_id": selected["session_id"], "session_index": selected["session_index"], "color_state_bin": state, "count": selected[f"n_{state}"], "percentage": selected[f"pct_{state}"]})
    occupancy_table = pd.DataFrame(occupancy_rows)
    occupancy_table.to_csv(output / "comparison_session_state_occupancy.csv", index=False)
    plot_comparison_state_percentages(occupancy_table, output / "comparison_state_bin_percentages.png")
    chosen_sessions = {run["label"]: {"session_id": str(run["chosen"]["session_id"]), "session_index": int(run["chosen"]["session_index"]), "strong_low_percentage": float(run["chosen"]["pct_strong_low"])} for run in runs}
    result = {"schema_version": "fucci_color_state_comparison_v1", "representative_sessions": chosen_sessions}
    (output / "run_log.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-state-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--session-index", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    compare_color_state_runs(args.color_state_dir, args.output_dir, labels=args.label, session_overrides=args.session_index, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
