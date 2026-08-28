from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from legacy_derivatives_migrate import (
    APPROVAL_TOKEN,
    COPY_CHUNK_SIZE,
    FrozenPhase2ABaseline,
    Phase2BMigrationError,
    _copy_verified,
    build_tree_inventory,
    compare_tree_rows,
    verify_phase2b_bundle,
)
from project_config import CanonicalVolumeConfig, ProjectConfig, ProjectPaths, RigConfig


@pytest.fixture
def phase2b_config(tmp_path: Path) -> ProjectConfig:
    raw_root = tmp_path / 'raw'
    derivatives_root = tmp_path / 'derivatives'
    legacy_root = tmp_path / 'legacy'
    raw_root.mkdir()
    derivatives_root.mkdir()
    legacy_root.mkdir()
    mice_csv = tmp_path / 'mice.csv'
    mice_csv.write_text('mouse_id' + chr(10), encoding='utf-8')
    return ProjectConfig(
        source_path=tmp_path / 'project.toml',
        paths=ProjectPaths(
            raw_root=raw_root,
            derivatives_root=derivatives_root,
            mice_csv=mice_csv,
            legacy_fucci_tri_root=legacy_root,
        ),
        rig=RigConfig(1050, 920, 920, 1050, 'green', 'red'),
        canonical_volume=CanonicalVolumeConfig(41, 1, 5.0, 50),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _make_row(source_path: Path, target_path: Path) -> dict[str, str]:
    stat_result = source_path.stat()
    return {
        'source_path': source_path.as_posix(),
        'proposed_target': target_path.as_posix(),
        'inferred_mouse_id': 'mouse_a',
        'inferred_laser_nm': '1050',
        'product_class': 'session',
        'target_scope': 'session',
        'inferred_session_date': '2026-08-28',
        'date_token_role': 'session_date',
        'catalog_session_match': 'True',
        'inference_status': 'resolved',
        'action': 'review_required',
        'collision_status': 'clear',
        'source_size_bytes': str(stat_result.st_size),
        'source_mtime_utc': '2026-08-28T00:00:00+00:00',
        'reason_evidence': 'unit-test',
        'phase2a_plan_sha256': 'plan-sha',
        'phase2a_plan_row_count': '1',
        'phase2a_repository_commit': 'phase2a-commit',
        'phase2b_executor_commit': 'executor-commit',
        'phase2b_disposition': 'copy_approved',
        'phase2b_approval_token': APPROVAL_TOKEN,
        'phase2b_approved_utc': '2026-08-28T00:00:00+00:00',
    }


def _source_snapshot(source_path: Path) -> dict[str, tuple[int, int]]:
    stat_result = source_path.stat()
    return {source_path.as_posix(): (stat_result.st_size, stat_result.st_mtime_ns)}


def _write_plan_bundle(base_dir: Path, source_path: Path, target_path: Path) -> Path:
    plan_dir = base_dir / 'phase2a_audit'
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / 'legacy_derivatives_migration_plan.csv'
    row = _make_row(source_path, target_path)
    with plan_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    report = {
        'repository': {'commit': 'phase2a-commit'},
        'legacy_derivatives_inventory': {
            'row_count': 1,
            'migration_plan_row_count': 1,
            'collision_status_counts': {'clear': 1, 'not_applicable': 0},
            'inference_status_counts': {'resolved': 1, 'unmapped': 0},
            'inventory_path': '',
            'migration_plan_path': '',
        },
        'catalog': {
            'files': {'acquisitions.generated.csv': {'path': '', 'sha256': 'acq-sha'}},
            'parseable_experiment_xml': 0,
        },
        'live_inventory': {
            'observed_summary': {
                'mouse_a': {
                    'alignment_only': 0,
                    'auxiliary_or_test': 0,
                    'canonical_1050': 1,
                    'canonical_920': 0,
                    'noncanonical': 0,
                    'sessions': 1,
                }
            }
        },
        'manifest_plans': {'plans': [{'mouse_id': 'mouse_a', 'laser_nm': 1050, 'row_count': 1}]},
    }
    (plan_dir / 'phase2a_report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return plan_path


def _prepare_completed_bundle(
    config: ProjectConfig,
    *,
    source_path: Path,
    target_path: Path,
    plan_path: Path,
    phase2a_commit: str = 'phase2a-commit',
    result_status: str = 'copied_verified',
) -> Path:
    output_dir = config.paths.derivatives_root / '_catalog' / 'phase2b_migration'
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _make_row(source_path, target_path)
    source_sha = _sha256(source_path)
    result_row = {
        'sequence': '1',
        'source_path': source_path.as_posix(),
        'target_path': target_path.as_posix(),
        'status': result_status,
        'source_sha256': source_sha,
        'target_sha256': source_sha,
        'source_size_bytes': str(source_path.stat().st_size),
        'target_size_bytes': str(source_path.stat().st_size),
        'source_mtime_ns': str(source_path.stat().st_mtime_ns),
        'temp_path': '',
        'completed_utc': '2026-08-28T00:00:00+00:00',
        'journal_path': (output_dir / 'phase2b_copy_journal.jsonl').as_posix(),
        'reason': 'unit-test',
    }
    journal_row = {
        **result_row,
        'plan_sha256': _sha256(plan_path),
        'phase2a_repository_commit': phase2a_commit,
        'phase2b_executor_commit': 'executor-commit',
        'source_plan_row_count': 1,
        'journaled_utc': '2026-08-28T00:00:00+00:00',
    }
    with (output_dir / 'phase2b_approved_copy_plan.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    with (output_dir / 'phase2b_deferred_unmapped.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
    with (output_dir / 'phase2b_preflight.json').open('w', encoding='utf-8') as handle:
        json.dump({'status': 'ready'}, handle)
    with (output_dir / 'phase2b_copy_journal.jsonl').open('w', encoding='utf-8', newline=chr(10)) as handle:
        handle.write(json.dumps(journal_row, sort_keys=True) + chr(10))
    with (output_dir / 'phase2b_copy_results.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_row.keys()))
        writer.writeheader()
        writer.writerow(result_row)
    tree_rows = build_tree_inventory(config.paths.raw_root)
    with (output_dir / 'raw_tree_before.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['relative_path', 'object_type', 'size_bytes', 'mtime_utc'])
        writer.writeheader()
        writer.writerows(tree_rows)
    with (output_dir / 'raw_tree_after.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['relative_path', 'object_type', 'size_bytes', 'mtime_utc'])
        writer.writeheader()
        writer.writerows(tree_rows)
    legacy_rows = build_tree_inventory(config.paths.legacy_fucci_tri_root)
    with (output_dir / 'legacy_tree_before.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['relative_path', 'object_type', 'size_bytes', 'mtime_utc'])
        writer.writeheader()
        writer.writerows(legacy_rows)
    with (output_dir / 'legacy_tree_after.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['relative_path', 'object_type', 'size_bytes', 'mtime_utc'])
        writer.writeheader()
        writer.writerows(legacy_rows)
    comparison = {
        'added_paths': [],
        'removed_paths': [],
        'changed_size_paths': [],
        'changed_modification_time_paths': [],
        'raw_tree_unchanged': True,
        'legacy_tree_unchanged': True,
    }
    (output_dir / 'raw_tree_comparison.json').write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding='utf-8')
    (output_dir / 'legacy_tree_comparison.json').write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding='utf-8')
    (output_dir / 'phase2b_report.json').write_text(
        json.dumps(
            {
                'report_version': 'phase2b_verified_copy_v1',
                'repository': {'commit': phase2a_commit, 'status': ''},
                'raw_immutability': {'raw_tree_unchanged': True, 'legacy_tree_unchanged': True},
                'legacy_immutability': {'raw_tree_unchanged': True, 'legacy_tree_unchanged': True},
                'copy_execution': {'approved_rows_accounted': 1, 'deferred_rows_accounted': 0},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding='utf-8',
    )
    return output_dir


def test_copy_verified_uses_verified_temp_sibling_and_cleans_temp(phase2b_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = phase2b_config.paths.legacy_fucci_tri_root / 'source.bin'
    source_path.write_bytes(b'hello world')
    target_path = phase2b_config.paths.derivatives_root / 'nested' / 'final.bin'
    row = _make_row(source_path, target_path)
    monkeypatch.setattr('legacy_derivatives_migrate._promote_temp_file', lambda temp_path, target_path: temp_path.rename(target_path))
    result = _copy_verified(
        row,
        config=phase2b_config,
        legacy_root=phase2b_config.paths.legacy_fucci_tri_root,
        source_snapshot=_source_snapshot(source_path),
        journal_records={},
        journal_path=phase2b_config.paths.derivatives_root / 'journal.jsonl',
        sequence=1,
    )
    assert result['status'] == 'copied_verified'
    assert result['temp_path'].endswith('.phase2b.tmp')
    assert target_path.read_bytes() == source_path.read_bytes()
    assert not Path(result['temp_path']).exists()
    assert not any(target_path.parent.glob('*.phase2b.tmp'))


def test_copy_verified_recovers_existing_verified_temp_sibling_without_recopy(phase2b_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = phase2b_config.paths.legacy_fucci_tri_root / 'source.bin'
    source_path.write_bytes(b'recover me')
    target_path = phase2b_config.paths.derivatives_root / 'nested' / 'final.bin'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.parent / f'.{target_path.name}.resume.phase2b.tmp'
    temp_path.write_bytes(source_path.read_bytes())
    xb_calls: list[Path] = []
    real_open = Path.open

    def open_spy(self: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get('mode', 'r')
        if mode == 'xb':
            xb_calls.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_spy, raising=False)
    monkeypatch.setattr('legacy_derivatives_migrate._promote_temp_file', lambda temp_path, target_path: temp_path.rename(target_path))
    result = _copy_verified(
        _make_row(source_path, target_path),
        config=phase2b_config,
        legacy_root=phase2b_config.paths.legacy_fucci_tri_root,
        source_snapshot=_source_snapshot(source_path),
        journal_records={},
        journal_path=phase2b_config.paths.derivatives_root / 'journal.jsonl',
        sequence=1,
    )
    assert result['status'] == 'copied_verified'
    assert xb_calls == []
    assert target_path.read_bytes() == source_path.read_bytes()
    assert not temp_path.exists()


def test_copy_verified_mid_stream_failure_cleans_only_current_temp(phase2b_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = phase2b_config.paths.legacy_fucci_tri_root / 'source.bin'
    source_path.write_bytes(b'abcdefghijklmno')
    target_path = phase2b_config.paths.derivatives_root / 'nested' / 'final.bin'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    keep_path = target_path.parent / 'keep.me'
    keep_path.write_text('keep', encoding='utf-8')
    monkeypatch.setattr('legacy_derivatives_migrate.COPY_CHUNK_SIZE', 4)
    real_open = Path.open

    class FailingReader:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self._reads = 0

        def read(self, size=-1):
            self._reads += 1
            if self._reads > 1:
                raise RuntimeError('simulated mid-stream failure')
            return self._wrapped.read(size)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self._wrapped.close()
            return False

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def open_spy(self: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get('mode', 'r')
        handle = real_open(self, *args, **kwargs)
        if self == source_path and mode == 'rb':
            return FailingReader(handle)
        return handle

    monkeypatch.setattr(Path, 'open', open_spy, raising=False)
    with pytest.raises(RuntimeError, match='simulated mid-stream failure'):
        _copy_verified(
            _make_row(source_path, target_path),
            config=phase2b_config,
            legacy_root=phase2b_config.paths.legacy_fucci_tri_root,
            source_snapshot=_source_snapshot(source_path),
            journal_records={},
            journal_path=phase2b_config.paths.derivatives_root / 'journal.jsonl',
            sequence=1,
        )
    assert not target_path.exists()
    assert keep_path.exists()
    assert not any(target_path.parent.glob('*.phase2b.tmp'))


def test_copy_verified_recovery_after_temp_verification_before_promotion(phase2b_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = phase2b_config.paths.legacy_fucci_tri_root / 'source.bin'
    source_path.write_bytes(b'resume from temp')
    target_path = phase2b_config.paths.derivatives_root / 'nested' / 'final.bin'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.parent / f'.{target_path.name}.resume.phase2b.tmp'
    temp_path.write_bytes(source_path.read_bytes())
    xb_calls: list[Path] = []
    real_open = Path.open

    def open_spy(self: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get('mode', 'r')
        if mode == 'xb':
            xb_calls.append(self)
        return real_open(self, *args, **kwargs)

    def promote_and_race(temp_path_arg: Path, target_path_arg: Path) -> None:
        target_path_arg.write_bytes(source_path.read_bytes())
        raise FileExistsError(target_path_arg)

    monkeypatch.setattr(Path, 'open', open_spy, raising=False)
    monkeypatch.setattr('legacy_derivatives_migrate._promote_temp_file', promote_and_race)
    result = _copy_verified(
        _make_row(source_path, target_path),
        config=phase2b_config,
        legacy_root=phase2b_config.paths.legacy_fucci_tri_root,
        source_snapshot=_source_snapshot(source_path),
        journal_records={},
        journal_path=phase2b_config.paths.derivatives_root / 'journal.jsonl',
        sequence=1,
    )
    assert result['status'] == 'already_present_verified'
    assert xb_calls == []
    assert target_path.read_bytes() == source_path.read_bytes()
    assert temp_path.exists()


def test_copy_verified_rejects_existing_mismatched_target_without_modification(phase2b_config: ProjectConfig) -> None:
    source_path = phase2b_config.paths.legacy_fucci_tri_root / 'source.bin'
    source_path.write_bytes(b'source')
    target_path = phase2b_config.paths.derivatives_root / 'nested' / 'final.bin'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    original_bytes = b'different target'
    target_path.write_bytes(original_bytes)
    with pytest.raises(Phase2BMigrationError, match='Destination conflict'):
        _copy_verified(
            _make_row(source_path, target_path),
            config=phase2b_config,
            legacy_root=phase2b_config.paths.legacy_fucci_tri_root,
            source_snapshot=_source_snapshot(source_path),
            journal_records={},
            journal_path=phase2b_config.paths.derivatives_root / 'journal.jsonl',
            sequence=1,
        )
    assert target_path.read_bytes() == original_bytes


def test_copy_verified_existing_identical_target_returns_already_present_verified_without_writes(phase2b_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = phase2b_config.paths.legacy_fucci_tri_root / 'source.bin'
    source_path.write_bytes(b'identical')
    target_path = phase2b_config.paths.derivatives_root / 'nested' / 'final.bin'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(source_path.read_bytes())
    promotion_calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal promotion_calls
        promotion_calls += 1
        raise AssertionError('promotion should not be attempted for identical targets')

    monkeypatch.setattr('legacy_derivatives_migrate._promote_temp_file', fail_if_called)
    before_mtime = target_path.stat().st_mtime_ns
    before_listing = sorted(path.name for path in target_path.parent.iterdir())
    result = _copy_verified(
        _make_row(source_path, target_path),
        config=phase2b_config,
        legacy_root=phase2b_config.paths.legacy_fucci_tri_root,
        source_snapshot=_source_snapshot(source_path),
        journal_records={},
        journal_path=phase2b_config.paths.derivatives_root / 'journal.jsonl',
        sequence=1,
    )
    after_listing = sorted(path.name for path in target_path.parent.iterdir())
    assert result['status'] == 'already_present_verified'
    assert promotion_calls == 0
    assert target_path.stat().st_mtime_ns == before_mtime
    assert before_listing == after_listing


def test_copy_verified_promotion_failure_fails_closed(phase2b_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = phase2b_config.paths.legacy_fucci_tri_root / 'source.bin'
    source_path.write_bytes(b'fail closed')
    target_path = phase2b_config.paths.derivatives_root / 'nested' / 'final.bin'
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def fail_promotion(*args, **kwargs):
        raise Phase2BMigrationError('unsupported no-clobber promotion')

    monkeypatch.setattr('legacy_derivatives_migrate._promote_temp_file', fail_promotion)
    with pytest.raises(Phase2BMigrationError, match='unsupported no-clobber promotion'):
        _copy_verified(
            _make_row(source_path, target_path),
            config=phase2b_config,
            legacy_root=phase2b_config.paths.legacy_fucci_tri_root,
            source_snapshot=_source_snapshot(source_path),
            journal_records={},
            journal_path=phase2b_config.paths.derivatives_root / 'journal.jsonl',
            sequence=1,
        )
    assert not target_path.exists()
    assert not any(target_path.parent.glob('*.phase2b.tmp'))


def test_compare_tree_rows_uses_semantic_legacy_field() -> None:
    rows = [{'relative_path': '.', 'object_type': 'directory', 'size_bytes': '', 'mtime_utc': '2026-08-28T00:00:00+00:00'}]
    comparison = compare_tree_rows(rows, rows, unchanged_key='legacy_tree_unchanged')
    assert comparison['raw_tree_unchanged'] is True
    assert comparison['legacy_tree_unchanged'] is True


def test_verify_only_does_not_write(phase2b_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = phase2b_config.paths.legacy_fucci_tri_root / 'source.bin'
    source_path.write_bytes(b'verify only')
    target_path = phase2b_config.paths.derivatives_root / 'nested' / 'target.bin'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(source_path.read_bytes())
    plan_path = _write_plan_bundle(phase2b_config.paths.derivatives_root, source_path, target_path)
    output_dir = _prepare_completed_bundle(phase2b_config, source_path=source_path, target_path=target_path, plan_path=plan_path)
    baseline = FrozenPhase2ABaseline(
        repository_commit='phase2a-commit',
        inventory_row_count=1,
        migration_plan_row_count=1,
        resolved_rows_with_targets=1,
        deferred_unmapped_rows=0,
        resolved_source_bytes=source_path.stat().st_size,
        deferred_source_bytes=0,
        inventory_sha256='inventory-sha',
        migration_plan_sha256=_sha256(plan_path),
        acquisition_catalog_sha256='acq-sha',
        raw_tree_sha256=_sha256(output_dir / 'raw_tree_before.csv'),
        catalog_parseable_experiment_xml=0,
        catalog_observed_summary={
            'mouse_a': {
                'alignment_only': 0,
                'auxiliary_or_test': 0,
                'canonical_1050': 1,
                'canonical_920': 0,
                'noncanonical': 0,
                'sessions': 1,
            }
        },
        manifest_plan_rows={('mouse_a', 1050): 1},
    )
    before_listing = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.iterdir())
    monkeypatch.setattr('legacy_derivatives_migrate._git_status', lambda *_args, **_kwargs: '')
    monkeypatch.setattr('legacy_derivatives_migrate._git_commit', lambda *_args, **_kwargs: 'repo-commit')
    report = verify_phase2b_bundle(
        phase2b_config,
        phase2a_plan_path=plan_path,
        output_dir=output_dir,
        frozen=baseline,
        repo_root=Path.cwd(),
        write_outputs=False,
    )
    after_listing = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.iterdir())
    assert before_listing == after_listing
    assert not (output_dir / 'phase2b_verify_only_report.json').exists()
    assert report['verification_only'] is True
    assert report['raw_tree_unchanged'] is True
    assert report['legacy_tree_unchanged'] is True
    assert report['source_files_retained'] is True
    assert report['source_bytes_accounted'] == source_path.stat().st_size
