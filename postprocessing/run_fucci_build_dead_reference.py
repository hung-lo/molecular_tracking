#!/usr/bin/env python3
"""Build a Fucci-Dead color-state reference from explicit master runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, _ROOT / "postprocessing", _ROOT / "plotting"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fucci_reference import build_dead_reference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-fit-rois", type=int, default=50)
    parser.add_argument("--modal-bandwidth-scale", type=float, default=0.6)
    parser.add_argument("--allow-single-reference-mouse", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_dead_reference(
        args.reference_run_dir,
        args.output_dir,
        min_fit_rois=args.min_fit_rois,
        modal_bandwidth_scale=args.modal_bandwidth_scale,
        allow_single_reference_mouse=args.allow_single_reference_mouse,
        overwrite=args.overwrite,
    )
    print(f"reference_dir={Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
