#!/usr/bin/env python3
"""Run the verified Phase 2B legacy derivatives copy migration."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))

from legacy_derivatives_migrate import APPROVAL_TOKEN, run_phase2b_migration, verify_phase2b_bundle
from project_config import load_project_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-config', required=True)
    parser.add_argument('--phase2a-plan', required=True)
    parser.add_argument('--approval-token', default=None)
    parser.add_argument('--execute-copy', action='store_true')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--repo-root', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_project_config(args.project_config)
    if args.verify_only and args.execute_copy:
        raise ValueError('--verify-only cannot be combined with --execute-copy')
    if args.verify_only:
        report = verify_phase2b_bundle(
            config,
            phase2a_plan_path=args.phase2a_plan,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            write_outputs=True,
        )
        print(f"repo_commit={report['repository']['commit']}")
        print(f"plan_sha256={report['phase2a_plan_sha256']}")
        print(f"approved_rows={report['approved_rows']} deferred_rows={report['deferred_rows']}")
        print(f"journal_rows={report['journal_rows']} result_rows={report['result_rows']}")
        print(f"verification_report_path={report.get('verification_report_path', '')}")
        print(f"raw_tree_unchanged={report['raw_tree_unchanged']} legacy_tree_unchanged={report['legacy_tree_unchanged']}")
        print(f"source_files_retained={report['source_files_retained']}")
        return 0
    summary = run_phase2b_migration(
        config,
        phase2a_plan_path=args.phase2a_plan,
        approval_token=args.approval_token or APPROVAL_TOKEN,
        execute_copy=args.execute_copy and not args.dry_run,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        write_outputs=True,
    )
    print(f'repo_commit={summary.repo_commit}')
    print(f'plan_sha256={summary.plan_sha256}')
    print(f'approved_rows={summary.approved_rows} deferred_rows={summary.deferred_rows}')
    print(f'copied_verified={summary.copied_verified} already_present_verified={summary.already_present_verified}')
    print(f'report_path={summary.report_path}')
    if summary.journal_path is not None:
        print(f'journal_path={summary.journal_path}')
    print(f'verified_copy_executed={summary.verified_copy_executed}')
    print(f'move_or_delete_executed={summary.move_or_delete_executed}')
    print(f'source_files_retained={summary.source_files_retained}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
