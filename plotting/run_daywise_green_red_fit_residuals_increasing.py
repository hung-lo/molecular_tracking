"""Run the day-wise green-vs-red residual analysis for increasing ROIs."""

from __future__ import annotations

import argparse

from run_daywise_green_red_fit_residuals import run_directional_residual_analysis
from project_cli import add_project_selector, resolve_exact_analysis_dir, resolve_selection


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the increasing-residual script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the dataset alias or path under ``dataset``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    add_project_selector(parser)
    parser.add_argument("--analysis-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the residual-based companion analysis for the top 30 increasing ROIs.

    The ranked ROI list comes from the current mean-merge SAM size+shape
    filtered branch. Outputs are written to a dated run directory under the
    same branch so the increasing and decreasing residual analyses remain
    directly comparable.
    """

    args = parse_args()
    context=resolve_selection(dataset=args.dataset,project_config=args.project_config,mouse_id=args.mouse_id,laser_nm=args.laser_nm)
    analysis_dir=resolve_exact_analysis_dir(context,args.analysis_dir) if context.mode == "project" or args.analysis_dir else None
    run_directional_residual_analysis(
        dataset=args.dataset,
        direction_label="increasing",
        output_dir_prefix="daywise_green_red_fit_residuals_increasing",
        analysis_dir=analysis_dir,
    )


if __name__ == "__main__":
    main()
