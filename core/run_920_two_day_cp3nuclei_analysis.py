"""Compatibility wrapper for the 920 nm registered-space ROI pipeline.

This script keeps the old entrypoint name for convenience, but it now delegates
all work to the shared registered-space pipeline. Despite the filename, it is no
longer limited to two imaging days.
"""

from __future__ import annotations

import argparse

from run_registered_roi_pipeline import RegisteredPipelineConfig, run_registered_roi_pipeline


DEFAULT_920_MASK_NAME = "mean_image_G_SyN_cp_masks_cpSAM.tif"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the 920 nm compatibility wrapper.

    Parameters
    ----------
    argv : list[str] or None, default=None
        Optional command-line argument list. When ``None``, arguments are read
        from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments for the 920 nm registered-space pipeline.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="920",
        help="Dataset alias or explicit dataset directory path. Defaults to the current 920_data dataset.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional reference date in YYYYMMDD format that defines day 0. If omitted, the earliest raw TIFF date is used.",
    )
    parser.add_argument(
        "--mask-name",
        default=DEFAULT_920_MASK_NAME,
        help="ROI mask filename inside the dataset directory.",
    )
    parser.add_argument(
        "--green-dark",
        type=float,
        default=319.0,
        help="Green-channel dark offset in arbitrary fluorescence units.",
    )
    parser.add_argument(
        "--red-dark",
        type=float,
        default=534.0,
        help="Red-channel dark offset in arbitrary fluorescence units.",
    )
    parser.add_argument(
        "--xy-um-per-px",
        type=float,
        default=0.693,
        help="XY resolution in micrometers per pixel.",
    )
    parser.add_argument(
        "--z-um-per-plane",
        type=float,
        default=5.0,
        help="Z step size in micrometers per plane.",
    )
    parser.add_argument(
        "--max-top-rois",
        type=int,
        default=30,
        help="Maximum number of ranked increasing or decreasing ROIs to export.",
    )
    parser.add_argument(
        "--inverse-mask-suffix",
        default="_ROI_mask_SyN_inversed.tif",
        help="Filename suffix used to find inverse-warped ROI masks in raw image space.",
    )
    parser.add_argument(
        "--inverse-mask-channel",
        choices=["auto", "red", "green"],
        default="auto",
        help="Preferred inverse-mask channel. Use auto to default to the 920 dataset rule.",
    )
    parser.add_argument(
        "--raw-space-half-window-z",
        type=int,
        default=5,
        help="Number of z planes to include above and below the ROI centroid for raw-space validation.",
    )
    raw_space_group = parser.add_mutually_exclusive_group()
    raw_space_group.add_argument(
        "--enable-raw-space-validation",
        action="store_true",
        help="Enable optional raw-space inverse-mask validation. This is off by default because it is often the slowest stage.",
    )
    raw_space_group.add_argument(
        "--skip-raw-space-validation",
        action="store_true",
        help="Compatibility flag that keeps raw-space inverse-mask validation disabled.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the shared registered-space ROI pipeline with 920 nm defaults."""

    args = parse_args()
    inverse_mask_channel = None if args.inverse_mask_channel == "auto" else args.inverse_mask_channel
    skip_raw_space_validation = True
    if args.enable_raw_space_validation:
        skip_raw_space_validation = False
    if args.skip_raw_space_validation:
        skip_raw_space_validation = True
    config = RegisteredPipelineConfig(
        dataset=args.dataset,
        start_date=args.start_date,
        mask_name=args.mask_name,
        green_dark=args.green_dark,
        red_dark=args.red_dark,
        xy_um_per_px=args.xy_um_per_px,
        z_um_per_plane=args.z_um_per_plane,
        max_top_rois=args.max_top_rois,
        inverse_mask_suffix=args.inverse_mask_suffix,
        inverse_mask_channel=inverse_mask_channel,
        raw_space_half_window_z=args.raw_space_half_window_z,
        skip_raw_space_validation=skip_raw_space_validation,
    )
    run_registered_roi_pipeline(config)


if __name__ == "__main__":
    main()
