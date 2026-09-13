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
    parser.add_argument("--synthetic-min-score", type=float, default=0.35)
    parser.add_argument("--synthetic-min-dice", type=float, default=0.10)
    parser.add_argument("--synthetic-max-distance-um", type=float, default=5.0)
    parser.add_argument("--synthetic-max-ambiguity", type=float, default=0.85)
    parser.add_argument(
        "--synthetic-require-consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require graph tracks annotated as affine-balanced consensus when that metadata exists.",
    )
    parser.add_argument(
        "--synthetic-require-no-cycle-conflict",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--synthetic-require-no-transform-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--synthetic-require-interior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate_daywise_tracking(
        args.match_dir, args.output_dir, policy=args.policy, lookahead=args.lookahead,
        search_radius_um=args.search_radius_um, crop_radius_um=args.crop_radius_um,
        z_radius=args.z_radius, extraction_dir=args.extraction_dir, state_csv=args.state_csv,
        max_review_panels=args.max_review_panels, random_seed=args.random_seed, overwrite=args.overwrite,
        synthetic_min_score=args.synthetic_min_score, synthetic_min_dice=args.synthetic_min_dice,
        synthetic_max_distance_um=args.synthetic_max_distance_um,
        synthetic_max_ambiguity=args.synthetic_max_ambiguity,
        synthetic_require_consensus=args.synthetic_require_consensus,
        synthetic_require_no_cycle_conflict=args.synthetic_require_no_cycle_conflict,
        synthetic_require_no_transform_fallback=args.synthetic_require_no_transform_fallback,
        synthetic_require_interior=args.synthetic_require_interior,
    )
    print(f"Evaluator completed: {args.output_dir}")
    print(f"Endpoints: {summary['n_endpoints']} | candidates: {summary['n_endpoint_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
