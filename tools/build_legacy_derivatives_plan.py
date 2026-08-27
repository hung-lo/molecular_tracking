#!/usr/bin/env python3
"""Build the read-only phase2a legacy derivatives inventory and migration plan."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from legacy_derivatives_plan import build_legacy_derivatives_audit
from project_config import load_project_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-config", required=True)
    parser.add_argument("--acquisition-catalog", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_project_config(args.project_config)
    inventory_rows, plan_rows, audit_dir = build_legacy_derivatives_audit(
        config,
        acquisition_catalog_path=args.acquisition_catalog,
        output_dir=args.output_dir,
        write_outputs=not args.dry_run,
    )
    print(f"inventory_rows={len(inventory_rows)} plan_rows={len(plan_rows)} audit_dir={audit_dir}")
    if args.dry_run:
        print("dry-run: no files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
