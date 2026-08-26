#!/usr/bin/env python3
"""Create a ready daywise manifest, or a preparation plan when inputs are absent."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"core"))
from dataset_catalog import build_manifest_plan
from project_config import load_project_config

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--project-config",required=True); parser.add_argument("--mouse-id",required=True); parser.add_argument("--laser-nm",type=int,default=None); parser.add_argument("--catalog",default=None)
    args=parser.parse_args(argv); config=load_project_config(args.project_config)
    catalog=Path(args.catalog).resolve() if args.catalog else config.paths.derivatives_root/"_catalog"/"acquisitions.generated.csv"
    if not catalog.is_file(): raise FileNotFoundError(f"Acquisition catalog was not found: {catalog}")
    with catalog.open(encoding="utf-8",newline="") as h: rows=list(csv.DictReader(h))
    for row in rows:
        row["analysis_included"]=str(row["analysis_included"]).lower()=="true"; row["laser_nm"]=int(row["laser_nm"]) if row["laser_nm"] else None
    report_path=config.paths.derivatives_root/"_catalog"/"validation_report.json"
    report=json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    path,ready=build_manifest_plan(config,rows,args.mouse_id,args.laser_nm,source_catalog=catalog,validation_report=report); print(f"{'manifest' if ready else 'plan'}={path}")
    return 0
if __name__=="__main__": raise SystemExit(main())
