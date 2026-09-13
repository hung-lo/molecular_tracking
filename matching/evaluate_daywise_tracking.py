"""CLI for the read-only daywise matcher endpoint evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matching", _REPO_ROOT / "plotting", _REPO_ROOT / "core"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from endpoint_evaluator import evaluate_daywise_tracking


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", required=True, help="Existing canonical matching output directory.")
    parser.add_argument("--output-dir", required=True, help="Separate evaluator output directory.")
    parser.add_argument("--policy", choices=("graph", "balanced", "high"), default="graph")
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--search-radius-um", type=float, default=15.0)
    parser.add_argument("--crop-radius-um", type=float, default=45.0)
    parser.add_argument("--z-radius", type=int, default=1)
    parser.add_argument("--extraction-dir", default=None)
    parser.add_argument("--state-csv", default=None)
    parser.add_argument("--max-review-panels", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate_daywise_tracking(
        args.match_dir, args.output_dir, policy=args.policy, lookahead=args.lookahead,
        search_radius_um=args.search_radius_um, crop_radius_um=args.crop_radius_um,
        z_radius=args.z_radius, extraction_dir=args.extraction_dir, state_csv=args.state_csv,
        max_review_panels=args.max_review_panels, random_seed=args.random_seed, overwrite=args.overwrite,
    )
    print(f"Evaluator completed: {args.output_dir}")
    print(f"Endpoints: {summary['n_endpoints']} | candidates: {summary['n_endpoint_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
