#!/usr/bin/env python3
"""Scan raw ThorImage data and write a deterministic derivatives catalog."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"core"))
from dataset_catalog import discover_catalog, write_catalog
from project_config import load_project_config

def parse_args(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-config",required=True)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--strict",action="store_true")
    return parser.parse_args(argv)

def main(argv=None):
    args=parse_args(argv); config=load_project_config(args.project_config)
    if not config.paths.raw_root.is_dir(): raise FileNotFoundError(f"raw_root is unreadable: {config.paths.raw_root}")
    rows,report=discover_catalog(config)
    for mouse,summary in report["summary"].items():
        print(mouse, " ".join(f"{key}={value}" for key,value in summary.items()))
    if args.dry_run:
        print("dry-run: no files written")
    elif args.strict and report["errors"]:
        print(f"strict validation failed: {len(report['errors'])} error(s); no catalog written")
        return 1
    else:
        output=write_catalog(config,rows,report); print(f"validation_report={output/'validation_report.json'}")
    return 1 if args.strict and report["errors"] else 0

if __name__=="__main__": raise SystemExit(main())
