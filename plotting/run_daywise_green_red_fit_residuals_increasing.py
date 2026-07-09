"""Run the day-wise green-vs-red residual analysis for increasing ROIs."""

from __future__ import annotations

import argparse

from run_daywise_green_red_fit_residuals import run_directional_residual_analysis


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the increasing-residual script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the dataset alias or path under ``dataset``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="1050",
        help="Dataset alias (e.g. 1050 or 920) or an explicit dataset directory path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the residual-based companion analysis for the top 30 increasing ROIs.

    The ranked ROI list comes from the current mean-merge SAM size+shape
    filtered branch. Outputs are written to a dated run directory under the
    same branch so the increasing and decreasing residual analyses remain
    directly comparable.
    """

    args = parse_args()
    run_directional_residual_analysis(
        dataset=args.dataset,
        direction_label="increasing",
        output_dir_prefix="daywise_green_red_fit_residuals_increasing",
    )


if __name__ == "__main__":
    main()
