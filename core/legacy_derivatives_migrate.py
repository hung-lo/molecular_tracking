from __future__ import annotations

import csv
import errno
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legacy_derivatives_plan import build_legacy_derivatives_audit
from project_config import ProjectConfig, is_relative_to, validate_output_path

APPROVAL_TOKEN = "PHASE2B_COPY_APPROVED"
COPY_CHUNK_SIZE = 8 * 1024 * 1024
MIN_FREE_SPACE_BYTES = 20 * 1024 * 1024 * 1024

EXPECTED_PHASE2A_COMMIT = "dfa0756bcecd0b394727a7fca27a4e5077cac67a"
EXPECTED_PHASE2A_INVENTORY_ROWS = 2929
EXPECTED_PHASE2A_PLAN_ROWS = 2929
EXPECTED_PHASE2A_RESOLVED_ROWS = 2857
EXPECTED_PHASE2A_DEFERRED_ROWS = 72
EXPECTED_PHASE2A_RESOLVED_SOURCE_BYTES = 16543364137
EXPECTED_PHASE2A_DEFERRED_SOURCE_BYTES = 26907277
EXPECTED_PHASE2A_INVENTORY_SHA256 = "9c262b602aa606e472440e49e549b6b4eedb5c85367ccbda72ed360c872ec598"
EXPECTED_PHASE2A_PLAN_SHA256 = "e8a95e36dcd36ce72b3a719ed64aeeae63732f61bb8a03e2b2a45aed133813e6"
EXPECTED_ACQUISITION_CATALOG_SHA256 = "1732f603769b40eb46a22c9d7128af60653242068a5ac45e853b4912361fb650"
EXPECTED_RAW_TREE_SHA256 = "71ec367143c6a65fe2f601dd0bb8a28076e22432ac68bfd998419adc3bf3940a"

EXPECTED_CATALOG_PARSEABLE_EXPERIMENT_XML = 95
EXPECTED_CATALOG_OBSERVED_SUMMARY = {
    "Fucci-Dead_1": {"alignment_only": 5, "auxiliary_or_test": 0, "canonical_1050": 2, "canonical_920": 2, "noncanonical": 0, "sessions": 2},
    "Fucci-Dead_2": {"alignment_only": 4, "auxiliary_or_test": 0, "canonical_1050": 2, "canonical_920": 2, "noncanonical": 0, "sessions": 2},
    "Fucci-Tri_1": {"alignment_only": 22, "auxiliary_or_test": 5, "canonical_1050": 25, "canonical_920": 16, "noncanonical": 3, "sessions": 25},
    "Fucci-Tri_2": {"alignment_only": 3, "auxiliary_or_test": 0, "canonical_1050": 2, "canonical_920": 2, "noncanonical": 0, "sessions": 2},
}
EXPECTED_MANIFEST_PLAN_ROWS = {
    ("Fucci-Dead_1", 1050): 2,
    ("Fucci-Dead_1", 920): 2,
    ("Fucci-Dead_2", 1050): 2,
    ("Fucci-Dead_2", 920): 2,
    ("Fucci-Tri_1", 1050): 25,
    ("Fucci-Tri_1", 920): 16,
    ("Fucci-Tri_2", 1050): 2,
    ("Fucci-Tri_2", 920): 2,
}

TREE_FIELDNAMES = ["relative_path", "object_type", "size_bytes", "mtime_utc"]
APPROVED_PLAN_FIELDNAMES = [
    "source_path",
    "proposed_target",
    "inferred_mouse_id",
    "inferred_laser_nm",
    "product_class",
    "target_scope",
    "inferred_session_date",
    "date_token_role",
    "catalog_session_match",
    "inference_status",
    "action",
    "collision_status",
    "source_size_bytes",
    "source_mtime_utc",
    "reason_evidence",
    "phase2a_plan_sha256",
    "phase2a_plan_row_count",
    "phase2a_repository_commit",
    "phase2b_executor_commit",
    "phase2b_disposition",
    "phase2b_approval_token",
    "phase2b_approved_utc",
]
DEFERRED_PLAN_FIELDNAMES = APPROVED_PLAN_FIELDNAMES
COPY_RESULT_FIELDNAMES = [
    "sequence",
    "source_path",
    "target_path",
    "status",
    "source_sha256",
    "target_sha256",
    "source_size_bytes",
    "target_size_bytes",
    "source_mtime_ns",
    "temp_path",
    "completed_utc",
    "journal_path",
    "reason",
]

AT_FDCWD = -100
RENAME_NOREPLACE = 1
PROMOTION_STRATEGY = "exclusive_direct"


class Phase2BMigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenPhase2ABaseline:
    repository_commit: str = EXPECTED_PHASE2A_COMMIT
    inventory_row_count: int = EXPECTED_PHASE2A_INVENTORY_ROWS
    migration_plan_row_count: int = EXPECTED_PHASE2A_PLAN_ROWS
    resolved_rows_with_targets: int = EXPECTED_PHASE2A_RESOLVED_ROWS
    deferred_unmapped_rows: int = EXPECTED_PHASE2A_DEFERRED_ROWS
    resolved_source_bytes: int = EXPECTED_PHASE2A_RESOLVED_SOURCE_BYTES
    deferred_source_bytes: int = EXPECTED_PHASE2A_DEFERRED_SOURCE_BYTES
    inventory_sha256: str = EXPECTED_PHASE2A_INVENTORY_SHA256
    migration_plan_sha256: str = EXPECTED_PHASE2A_PLAN_SHA256
    acquisition_catalog_sha256: str = EXPECTED_ACQUISITION_CATALOG_SHA256
    raw_tree_sha256: str = EXPECTED_RAW_TREE_SHA256
    catalog_parseable_experiment_xml: int = EXPECTED_CATALOG_PARSEABLE_EXPERIMENT_XML
    catalog_observed_summary: dict[str, dict[str, int]] | None = None
    manifest_plan_rows: dict[tuple[str, int], int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalog_observed_summary", self.catalog_observed_summary or EXPECTED_CATALOG_OBSERVED_SUMMARY)
        object.__setattr__(self, "manifest_plan_rows", self.manifest_plan_rows or EXPECTED_MANIFEST_PLAN_ROWS)


@dataclass(frozen=True)
class Phase2BPaths:
    output_dir: Path
    approved_plan_csv: Path
    deferred_unmapped_csv: Path
    preflight_json: Path
    journal_jsonl: Path
    results_csv: Path
    report_json: Path
    legacy_tree_before_csv: Path
    legacy_tree_after_csv: Path
    raw_tree_before_csv: Path
    raw_tree_after_csv: Path
    legacy_tree_comparison_json: Path
    raw_tree_comparison_json: Path


@dataclass(frozen=True)
class Phase2BRunSummary:
    repo_commit: str
    plan_sha256: str
    approved_rows: int
    deferred_rows: int
    approved_source_bytes: int
    deferred_source_bytes: int
    copied_verified: int
    already_present_verified: int
    source_bytes_accounted: int
    disk_free_before_bytes: int
    disk_free_after_bytes: int | None
    preflight_path: Path
    report_path: Path
    journal_path: Path | None
    results_path: Path | None
    approved_plan_path: Path
    deferred_plan_path: Path
    legacy_tree_before_path: Path
    legacy_tree_after_path: Path
    raw_tree_before_path: Path
    raw_tree_after_path: Path
    legacy_tree_unchanged: bool
    raw_tree_unchanged: bool
    verified_copy_executed: bool
    move_or_delete_executed: bool
    source_files_retained: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_token(token: str | None) -> str:
    return token.strip() if isinstance(token, str) else ''


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_SIZE), b""):
            total += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), total


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _tree_row(path: Path, root: Path) -> dict[str, Any]:
    rel = "." if path == root else path.relative_to(root).as_posix()
    if path.is_symlink():
        stat_result = path.lstat()
        object_type = "symlink"
        size_bytes = ""
    elif path.is_dir():
        stat_result = path.stat()
        object_type = "directory"
        size_bytes = ""
    elif path.is_file():
        stat_result = path.stat()
        object_type = "file"
        size_bytes = str(stat_result.st_size)
    else:
        stat_result = path.lstat()
        object_type = "other"
        size_bytes = ""
    return {
        "relative_path": rel,
        "object_type": object_type,
        "size_bytes": size_bytes,
        "mtime_utc": datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat(),
    }


def build_tree_inventory(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Tree root was not found: {root_path}")
    rows = [_tree_row(root_path, root_path)]
    for entry in sorted(root_path.rglob("*"), key=lambda item: item.relative_to(root_path).as_posix()):
        rows.append(_tree_row(entry, root_path))
    return rows


def write_tree_inventory(root: str | Path, csv_path: str | Path) -> tuple[list[dict[str, Any]], str]:
    rows = build_tree_inventory(root)
    csv_path = Path(csv_path)
    _write_csv_atomic(csv_path, rows, TREE_FIELDNAMES)
    return rows, _sha256_file(csv_path)


def compare_tree_rows(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> dict[str, Any]:
    before = {row["relative_path"]: row for row in before_rows}
    after = {row["relative_path"]: row for row in after_rows}
    before_paths = set(before)
    after_paths = set(after)
    common = before_paths & after_paths
    changed_size_paths = sorted(path for path in common if before[path]["size_bytes"] != after[path]["size_bytes"])
    changed_modification_time_paths = sorted(path for path in common if before[path]["mtime_utc"] != after[path]["mtime_utc"])
    return {
        "added_paths": sorted(after_paths - before_paths),
        "removed_paths": sorted(before_paths - after_paths),
        "changed_size_paths": changed_size_paths,
        "changed_modification_time_paths": changed_modification_time_paths,
        "raw_tree_unchanged": not (after_paths - before_paths or before_paths - after_paths or changed_size_paths or changed_modification_time_paths),
    }


def _load_phase2a_report(phase2a_plan_path: Path) -> dict[str, Any]:
    report_path = phase2a_plan_path.parent / "phase2a_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Phase 2A report was not found: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_status(repo_root: Path) -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=repo_root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _validate_phase2a_report(report: dict[str, Any], frozen: FrozenPhase2ABaseline) -> None:
    if report["repository"]["commit"] != frozen.repository_commit:
        raise Phase2BMigrationError("Phase 2A repository commit mismatch")
    legacy = report["legacy_derivatives_inventory"]
    if legacy["row_count"] != frozen.inventory_row_count:
        raise Phase2BMigrationError("Phase 2A inventory row-count mismatch")
    if legacy["migration_plan_row_count"] != frozen.migration_plan_row_count:
        raise Phase2BMigrationError("Phase 2A migration-plan row-count mismatch")
    if legacy["collision_status_counts"] != {"clear": frozen.resolved_rows_with_targets, "not_applicable": frozen.deferred_unmapped_rows}:
        raise Phase2BMigrationError("Phase 2A collision-status counts mismatch")
    if legacy["inference_status_counts"] != {"resolved": frozen.resolved_rows_with_targets, "unmapped": frozen.deferred_unmapped_rows}:
        raise Phase2BMigrationError("Phase 2A inference-status counts mismatch")
    if legacy["inventory_path"] and _sha256_file(Path(legacy["inventory_path"])) != frozen.inventory_sha256:
        raise Phase2BMigrationError("Phase 2A inventory hash mismatch")
    if legacy["migration_plan_path"] and _sha256_file(Path(legacy["migration_plan_path"])) != frozen.migration_plan_sha256:
        raise Phase2BMigrationError("Phase 2A migration-plan hash mismatch")
    if report["catalog"]["files"]["acquisitions.generated.csv"]["sha256"] != frozen.acquisition_catalog_sha256:
        raise Phase2BMigrationError("Phase 2A acquisition catalog hash mismatch")
    if report["catalog"]["parseable_experiment_xml"] != frozen.catalog_parseable_experiment_xml:
        raise Phase2BMigrationError("Phase 2A parseable XML count mismatch")
    if report["live_inventory"]["observed_summary"] != frozen.catalog_observed_summary:
        raise Phase2BMigrationError("Phase 2A live inventory summary mismatch")
    manifests = {(item["mouse_id"], item["laser_nm"]): item for item in report["manifest_plans"]["plans"]}
    if set(manifests) != set(frozen.manifest_plan_rows):
        raise Phase2BMigrationError("Phase 2A manifest-plan coverage mismatch")
    for key, expected_rows in frozen.manifest_plan_rows.items():
        if manifests[key]["row_count"] != expected_rows:
            raise Phase2BMigrationError("Phase 2A manifest-plan row-count mismatch")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_source_path(source_path: Path, legacy_root: Path) -> None:
    resolved = source_path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        raise Phase2BMigrationError(f"Source must be a regular, non-symlink file: {resolved}")
    if not is_relative_to(resolved, legacy_root.resolve()):
        raise Phase2BMigrationError(f"Source escapes legacy root: {resolved}")


def _validate_target_path(target_path: Path, config: ProjectConfig, legacy_root: Path) -> None:
    resolved = target_path.expanduser().resolve()
    derivatives = config.paths.derivatives_root.resolve()
    raw_root = config.paths.raw_root.resolve()
    legacy_root = legacy_root.resolve()
    if not is_relative_to(resolved, derivatives):
        raise Phase2BMigrationError(f"Target must resolve beneath derivatives_root: {resolved}")
    if is_relative_to(resolved, raw_root):
        raise Phase2BMigrationError(f"Target must not resolve inside raw_root: {resolved}")
    if is_relative_to(resolved, legacy_root):
        raise Phase2BMigrationError(f"Target must not resolve inside legacy_fucci_tri_root: {resolved}")


def _ensure_parent_chain(parent: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_parent = parent.resolve()
    if not is_relative_to(resolved_parent, resolved_root):
        raise Phase2BMigrationError(f"Target escapes derivatives root: {resolved_parent}")
    current = resolved_root
    for part in resolved_parent.relative_to(resolved_root).parts:
        current = current / part
        if current.exists():
            if current.is_symlink():
                raise Phase2BMigrationError(f"Symlinked target parent rejected: {current}")
            if not current.is_dir():
                raise Phase2BMigrationError(f"Non-directory target parent rejected: {current}")
        else:
            current.mkdir()


def _plan_snapshot(path: Path) -> tuple[int, int, str]:
    stat_result = path.stat()
    return stat_result.st_size, stat_result.st_mtime_ns, datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).isoformat()


def _promote_temp_file(temp_path: Path, target_path: Path) -> None:
    if not hasattr(os, "rename") and not hasattr(os, "replace"):
        raise Phase2BMigrationError("No rename support available")
    if os.name != "posix":
        raise Phase2BMigrationError("Phase 2B executor requires a POSIX filesystem for atomic no-clobber promotion")
    renameat2 = getattr(os, "renameat2", None)
    if renameat2 is not None:
        try:
            renameat2(temp_path, target_path, src_dir_fd=None, dst_dir_fd=None, flags=RENAME_NOREPLACE)  # type: ignore[call-arg]
            return
        except TypeError:
            # Older Python builds may expose renameat2 but not this call signature; fall through to ctypes.
            pass
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    func = getattr(libc, "renameat2", None)
    if func is None:
        raise Phase2BMigrationError("renameat2(RENAME_NOREPLACE) is unavailable on this platform")
    func.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    func.restype = ctypes.c_int
    src_b = os.fsencode(temp_path)
    dst_b = os.fsencode(target_path)
    result = func(AT_FDCWD, src_b, AT_FDCWD, dst_b, RENAME_NOREPLACE)
    if result != 0:
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(target_path)
        raise OSError(err, os.strerror(err), str(target_path))


def _journal_records(journal_path: Path, expected_plan_sha256: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not journal_path.exists():
        return records
    with journal_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise Phase2BMigrationError(f"Malformed journal record on line {line_no}: {exc}") from exc
            if record.get("plan_sha256") != expected_plan_sha256:
                raise Phase2BMigrationError("Journal plan hash does not match the approved plan")
            target = record.get("target_path")
            if not target:
                raise Phase2BMigrationError(f"Malformed journal record on line {line_no}: missing target_path")
            records[str(target)] = record
    return records


def _append_journal(journal_path: Path, record: dict[str, Any]) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _phase2b_paths(output_dir: Path) -> Phase2BPaths:
    return Phase2BPaths(
        output_dir=output_dir,
        approved_plan_csv=output_dir / "phase2b_approved_copy_plan.csv",
        deferred_unmapped_csv=output_dir / "phase2b_deferred_unmapped.csv",
        preflight_json=output_dir / "phase2b_preflight.json",
        journal_jsonl=output_dir / "phase2b_copy_journal.jsonl",
        results_csv=output_dir / "phase2b_copy_results.csv",
        report_json=output_dir / "phase2b_report.json",
        legacy_tree_before_csv=output_dir / "legacy_tree_before.csv",
        legacy_tree_after_csv=output_dir / "legacy_tree_after.csv",
        raw_tree_before_csv=output_dir / "raw_tree_before.csv",
        raw_tree_after_csv=output_dir / "raw_tree_after.csv",
        legacy_tree_comparison_json=output_dir / "legacy_tree_comparison.json",
        raw_tree_comparison_json=output_dir / "raw_tree_comparison.json",
    )


def _split_plan_rows(rows: list[dict[str, str]], frozen: FrozenPhase2ABaseline, plan_sha256: str, repo_commit: str, approval_token: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approved: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    approved_bytes = 0
    deferred_bytes = 0
    approved_utc = _utc_now()
    for row in rows:
        source_size = int(row["source_size_bytes"])
        common = dict(row)
        common.update(
            {
                "phase2a_plan_sha256": plan_sha256,
                "phase2a_plan_row_count": str(frozen.migration_plan_row_count),
                "phase2a_repository_commit": frozen.repository_commit,
                "phase2b_executor_commit": repo_commit,
                "phase2b_approval_token": approval_token if row.get("proposed_target") else "",
                "phase2b_approved_utc": approved_utc,
            }
        )
        if row.get("proposed_target"):
            if row["inference_status"] != "resolved" or row["collision_status"] != "clear" or row["action"] != "review_required":
                raise Phase2BMigrationError(f"Unexpected resolvable row in Phase 2A plan: {row['source_path']}")
            common["phase2b_disposition"] = "copy_approved"
            approved.append(common)
            approved_bytes += source_size
        else:
            if row["inference_status"] != "unmapped" or row["collision_status"] != "not_applicable":
                raise Phase2BMigrationError(f"Unexpected deferred row in Phase 2A plan: {row['source_path']}")
            common["phase2b_disposition"] = "deferred_unmapped"
            deferred.append(common)
            deferred_bytes += source_size
    if len(approved) != frozen.resolved_rows_with_targets:
        raise Phase2BMigrationError(f"Expected {frozen.resolved_rows_with_targets} approved rows, got {len(approved)}")
    if len(deferred) != frozen.deferred_unmapped_rows:
        raise Phase2BMigrationError(f"Expected {frozen.deferred_unmapped_rows} deferred rows, got {len(deferred)}")
    if approved_bytes != frozen.resolved_source_bytes:
        raise Phase2BMigrationError(f"Approved-source byte count mismatch: expected {frozen.resolved_source_bytes}, got {approved_bytes}")
    if deferred_bytes != frozen.deferred_source_bytes:
        raise Phase2BMigrationError(f"Deferred-source byte count mismatch: expected {frozen.deferred_source_bytes}, got {deferred_bytes}")
    return approved, deferred


def _copy_direct_verified(row: dict[str, Any], *, config: ProjectConfig, legacy_root: Path, source_snapshot: dict[str, tuple[int, int]], journal_records: dict[str, dict[str, Any]], journal_path: Path, sequence: int, source_sha: str, source_size: int, source_mtime_ns: int) -> dict[str, Any]:
    source_path = Path(row["source_path"]).expanduser().resolve()
    target_path = Path(row["proposed_target"]).expanduser().resolve()
    journal = journal_records.get(target_path.as_posix())
    if journal is not None:
        if journal.get("plan_sha256") != row["phase2a_plan_sha256"]:
            raise Phase2BMigrationError(f"Journal plan hash mismatch for {target_path}")
        if not target_path.exists():
            raise Phase2BMigrationError(f"Journaled target is missing on resume: {target_path}")
        target_sha, target_size = _sha256_stream(target_path)
        if target_sha != journal.get("target_sha256") or target_size != int(journal.get("target_size_bytes", -1)):
            raise Phase2BMigrationError(f"Journaled target no longer matches on resume: {target_path}")
        return {
            "sequence": sequence,
            "source_path": source_path.as_posix(),
            "target_path": target_path.as_posix(),
            "status": "already_present_verified",
            "source_sha256": source_sha,
            "target_sha256": target_sha,
            "source_size_bytes": source_size,
            "target_size_bytes": target_size,
            "source_mtime_ns": source_mtime_ns,
            "temp_path": "",
            "completed_utc": _utc_now(),
            "journal_path": journal_path.as_posix(),
            "reason": "journaled target verified on resume",
        }

    if target_path.exists():
        target_sha, target_size = _sha256_stream(target_path)
        if target_sha == source_sha and target_size == source_size:
            return {
                "sequence": sequence,
                "source_path": source_path.as_posix(),
                "target_path": target_path.as_posix(),
                "status": "already_present_verified",
                "source_sha256": source_sha,
                "target_sha256": target_sha,
                "source_size_bytes": source_size,
                "target_size_bytes": target_size,
                "source_mtime_ns": source_mtime_ns,
                "temp_path": "",
                "completed_utc": _utc_now(),
                "journal_path": journal_path.as_posix(),
                "reason": "identical target already present",
            }
        raise Phase2BMigrationError(f"Destination conflict for {target_path}")

    _ensure_parent_chain(target_path.parent, config.paths.derivatives_root)
    created_target = False
    try:
        with source_path.open("rb") as source_handle, target_path.open("xb") as target_handle:
            created_target = True
            for chunk in iter(lambda: source_handle.read(COPY_CHUNK_SIZE), b""):
                target_handle.write(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.chmod(target_path, os.stat(source_path).st_mode & 0o777)
        os.utime(target_path, ns=(source_path.stat().st_atime_ns, source_path.stat().st_mtime_ns))
        final_sha, final_size = _sha256_stream(target_path)
        if final_sha != source_sha or final_size != source_size:
            raise Phase2BMigrationError(f"Final target verification failed for {target_path}")
        dir_fd = os.open(str(target_path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return {
            "sequence": sequence,
            "source_path": source_path.as_posix(),
            "target_path": target_path.as_posix(),
            "status": "copied_verified",
            "source_sha256": source_sha,
            "target_sha256": final_sha,
            "source_size_bytes": source_size,
            "target_size_bytes": final_size,
            "source_mtime_ns": source_mtime_ns,
            "temp_path": "",
            "completed_utc": _utc_now(),
            "journal_path": journal_path.as_posix(),
            "reason": "copied via direct exclusive create",
        }
    except FileExistsError:
        target_sha, target_size = _sha256_stream(target_path)
        if target_sha == source_sha and target_size == source_size:
            return {
                "sequence": sequence,
                "source_path": source_path.as_posix(),
                "target_path": target_path.as_posix(),
                "status": "already_present_verified",
                "source_sha256": source_sha,
                "target_sha256": target_sha,
                "source_size_bytes": source_size,
                "target_size_bytes": target_size,
                "source_mtime_ns": source_mtime_ns,
                "temp_path": "",
                "completed_utc": _utc_now(),
                "journal_path": journal_path.as_posix(),
                "reason": "target appeared during exclusive create but matched source",
            }
        raise Phase2BMigrationError(f"Destination conflict during direct copy for {target_path}")
    except Exception:
        if created_target and target_path.exists():
            try:
                target_path.unlink()
            except FileNotFoundError:
                pass
        raise


def _copy_verified(row: dict[str, Any], *, config: ProjectConfig, legacy_root: Path, source_snapshot: dict[str, tuple[int, int]], journal_records: dict[str, dict[str, Any]], journal_path: Path, sequence: int) -> dict[str, Any]:
    source_path = Path(row["source_path"]).expanduser().resolve()
    target_path = Path(row["proposed_target"]).expanduser().resolve()
    _validate_source_path(source_path, legacy_root)
    _validate_target_path(target_path, config, legacy_root)
    source_size, source_mtime_ns, _source_mtime_utc = _plan_snapshot(source_path)
    snapshot_size, snapshot_mtime_ns = source_snapshot[source_path.as_posix()]
    if source_size != snapshot_size or source_mtime_ns != snapshot_mtime_ns:
        raise Phase2BMigrationError(f"Source changed or became invalid during copy: {source_path}")

    source_sha, _ = _sha256_stream(source_path)
    journal = journal_records.get(target_path.as_posix())
    if journal is not None:
        if journal.get("plan_sha256") != row["phase2a_plan_sha256"]:
            raise Phase2BMigrationError(f"Journal plan hash mismatch for {target_path}")
        if not target_path.exists():
            raise Phase2BMigrationError(f"Journaled target is missing on resume: {target_path}")
        target_sha, target_size = _sha256_stream(target_path)
        if target_sha != journal.get("target_sha256") or target_size != int(journal.get("target_size_bytes", -1)):
            raise Phase2BMigrationError(f"Journaled target no longer matches on resume: {target_path}")
        return {
            "sequence": sequence,
            "source_path": source_path.as_posix(),
            "target_path": target_path.as_posix(),
            "status": "already_present_verified",
            "source_sha256": source_sha,
            "target_sha256": target_sha,
            "source_size_bytes": source_size,
            "target_size_bytes": target_size,
            "source_mtime_ns": source_mtime_ns,
            "temp_path": "",
            "completed_utc": _utc_now(),
            "journal_path": journal_path.as_posix(),
            "reason": "journaled target verified on resume",
        }

    if target_path.exists():
        target_sha, target_size = _sha256_stream(target_path)
        if target_sha == source_sha and target_size == source_size:
            return {
                "sequence": sequence,
                "source_path": source_path.as_posix(),
                "target_path": target_path.as_posix(),
                "status": "already_present_verified",
                "source_sha256": source_sha,
                "target_sha256": target_sha,
                "source_size_bytes": source_size,
                "target_size_bytes": target_size,
                "source_mtime_ns": source_mtime_ns,
                "temp_path": "",
                "completed_utc": _utc_now(),
                "journal_path": journal_path.as_posix(),
                "reason": "identical target already present",
            }
        raise Phase2BMigrationError(f"Destination conflict for {target_path}")

    if PROMOTION_STRATEGY == "exclusive_direct":
        return _copy_direct_verified(
            row,
            config=config,
            legacy_root=legacy_root,
            source_snapshot=source_snapshot,
            journal_records=journal_records,
            journal_path=journal_path,
            sequence=sequence,
            source_sha=source_sha,
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
        )

    _ensure_parent_chain(target_path.parent, config.paths.derivatives_root)
    temp_path = target_path.parent / f".{target_path.name}.{os.getpid()}.{uuid.uuid4().hex}.phase2b.tmp"
    try:
        with source_path.open("rb") as source_handle, temp_path.open("wb") as temp_handle:
            for chunk in iter(lambda: source_handle.read(COPY_CHUNK_SIZE), b""):
                temp_handle.write(chunk)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        os.chmod(temp_path, os.stat(source_path).st_mode & 0o777)
        os.utime(temp_path, ns=(source_path.stat().st_atime_ns, source_path.stat().st_mtime_ns))
        temp_sha, temp_size = _sha256_stream(temp_path)
        if temp_sha != source_sha or temp_size != source_size:
            raise Phase2BMigrationError(f"Temporary copy verification failed for {source_path}")
        try:
            _promote_temp_file(temp_path, target_path)
        except FileExistsError:
            target_sha, target_size = _sha256_stream(target_path)
            if target_sha == source_sha and target_size == source_size:
                return {
                    "sequence": sequence,
                    "source_path": source_path.as_posix(),
                    "target_path": target_path.as_posix(),
                    "status": "already_present_verified",
                    "source_sha256": source_sha,
                    "target_sha256": target_sha,
                    "source_size_bytes": source_size,
                    "target_size_bytes": target_size,
                    "source_mtime_ns": source_mtime_ns,
                    "temp_path": temp_path.as_posix(),
                    "completed_utc": _utc_now(),
                    "journal_path": journal_path.as_posix(),
                    "reason": "target appeared during promotion but matched source",
                }
            raise Phase2BMigrationError(f"Destination conflict during promotion for {target_path}")
        dir_fd = os.open(str(target_path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        final_sha, final_size = _sha256_stream(target_path)
        if final_sha != source_sha or final_size != source_size:
            raise Phase2BMigrationError(f"Final target verification failed for {target_path}")
        return {
            "sequence": sequence,
            "source_path": source_path.as_posix(),
            "target_path": target_path.as_posix(),
            "status": "copied_verified",
            "source_sha256": source_sha,
            "target_sha256": final_sha,
            "source_size_bytes": source_size,
            "target_size_bytes": final_size,
            "source_mtime_ns": source_mtime_ns,
            "temp_path": temp_path.as_posix(),
            "completed_utc": _utc_now(),
            "journal_path": journal_path.as_posix(),
            "reason": "copied via temp sibling and atomic no-clobber promotion",
        }
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _progress_message(completed: int, total: int, bytes_done: int, started: float) -> str:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = bytes_done / elapsed
    eta_seconds = (total - completed) / (completed / elapsed) if completed else 0.0
    return f"{completed}/{total} files | {bytes_done / (1024 ** 3):.2f} GiB copied | {rate / (1024 ** 3):.2f} GiB/s | ETA {eta_seconds / 60:.1f} min"


def _phase2b_report(
    *,
    repo_commit: str,
    repo_status: str,
    plan_path: Path,
    plan_sha256: str,
    approved_rows: list[dict[str, Any]],
    deferred_rows: list[dict[str, Any]],
    scope_summary: Any,
    preflight_path: Path,
    raw_before_path: Path,
    raw_after_path: Path,
    legacy_before_path: Path,
    legacy_after_path: Path,
    raw_before_rows: list[dict[str, Any]],
    raw_after_rows: list[dict[str, Any]],
    legacy_before_rows: list[dict[str, Any]],
    legacy_after_rows: list[dict[str, Any]],
    copy_results: list[dict[str, Any]],
    copied_verified: int,
    already_present_verified: int,
    disk_free_before: int,
    disk_free_after: int | None,
    verified_copy_executed: bool,
    move_or_delete_executed: bool,
    source_files_retained: bool,
    approved_plan_path: Path,
    deferred_plan_path: Path,
    journal_path: Path | None,
    results_path: Path | None,
    phase2a_report: dict[str, Any],
) -> dict[str, Any]:
    approved_target_scope_counts = dict(Counter(row["target_scope"] for row in approved_rows))
    approved_inference_counts = dict(Counter(row["inference_status"] for row in approved_rows))
    approved_action_counts = dict(Counter(row["action"] for row in approved_rows))
    approved_collision_counts = dict(Counter(row["collision_status"] for row in approved_rows))
    deferred_target_scope_counts = dict(Counter(row["target_scope"] for row in deferred_rows))
    copy_status_counts = dict(Counter(row["status"] for row in copy_results))
    raw_diff = compare_tree_rows(raw_before_rows, raw_after_rows)
    legacy_diff = compare_tree_rows(legacy_before_rows, legacy_after_rows)
    return {
        "report_version": "phase2b_verified_copy_v1",
        "generated_utc": _utc_now(),
        "repository": {
            "commit": repo_commit,
            "path": Path(__file__).resolve().parents[1].as_posix(),
            "status": repo_status,
            "baseline_phase2a_commit": EXPECTED_PHASE2A_COMMIT,
        },
        "phase": "2B",
        "supersession_note": "The earlier 2,920-row Phase 2A plan is obsolete; this Phase 2B run uses the corrected 2,929-row scope that includes the previously omitted styled QC tree.",
        "baseline_delta": {
            "baseline_row_count": 2920,
            "corrected_row_count": 2929,
            "delta_row_count": 9,
            "added_root": "roi_matcher_qc_examples_syn_20260615_01_styled",
            "added_root_row_count": 9,
            "note": "The row-count increase is a scope correction, not a source-data change.",
        },
        "phase2a_baseline": {
            "plan_path": plan_path.as_posix(),
            "plan_sha256": plan_sha256,
            "inventory_row_count": EXPECTED_PHASE2A_INVENTORY_ROWS,
            "migration_plan_row_count": EXPECTED_PHASE2A_PLAN_ROWS,
            "resolved_rows_with_targets": EXPECTED_PHASE2A_RESOLVED_ROWS,
            "deferred_unmapped_rows": EXPECTED_PHASE2A_DEFERRED_ROWS,
            "resolved_source_bytes": EXPECTED_PHASE2A_RESOLVED_SOURCE_BYTES,
            "deferred_source_bytes": EXPECTED_PHASE2A_DEFERRED_SOURCE_BYTES,
            "inventory_sha256": EXPECTED_PHASE2A_INVENTORY_SHA256,
            "migration_plan_sha256": EXPECTED_PHASE2A_PLAN_SHA256,
            "acquisition_catalog_sha256": EXPECTED_ACQUISITION_CATALOG_SHA256,
            "raw_tree_sha256": EXPECTED_RAW_TREE_SHA256,
            "included_roots": list(scope_summary.included_roots),
            "ignored_top_level_entries": [f"{name}:{kind}" for name, kind in scope_summary.ignored_top_level_entries],
            "approved_target_scope_counts": approved_target_scope_counts,
            "approved_inference_status_counts": approved_inference_counts,
            "approved_action_counts": approved_action_counts,
            "approved_collision_status_counts": approved_collision_counts,
            "deferred_target_scope_counts": deferred_target_scope_counts,
        },
        "approved_plan": {
            "path": approved_plan_path.as_posix(),
            "sha256": _sha256_file(approved_plan_path),
            "size_bytes": approved_plan_path.stat().st_size,
            "row_count": len(approved_rows),
            "source_bytes": sum(int(row["source_size_bytes"]) for row in approved_rows),
        },
        "deferred_plan": {
            "path": deferred_plan_path.as_posix(),
            "sha256": _sha256_file(deferred_plan_path),
            "size_bytes": deferred_plan_path.stat().st_size,
            "row_count": len(deferred_rows),
            "source_bytes": sum(int(row["source_size_bytes"]) for row in deferred_rows),
        },
        "catalog": phase2a_report["catalog"],
        "live_inventory": phase2a_report["live_inventory"],
        "manifest_plans": phase2a_report["manifest_plans"],
        "preflight_path": preflight_path.as_posix(),
        "raw_immutability": {
            "before_inventory": raw_before_path.as_posix(),
            "after_inventory": raw_after_path.as_posix(),
            "before_sha256": _sha256_file(raw_before_path),
            "after_sha256": _sha256_file(raw_after_path),
            **raw_diff,
        },
        "legacy_immutability": {
            "before_inventory": legacy_before_path.as_posix(),
            "after_inventory": legacy_after_path.as_posix(),
            "before_sha256": _sha256_file(legacy_before_path),
            "after_sha256": _sha256_file(legacy_after_path),
            "added_paths": legacy_diff["added_paths"],
            "removed_paths": legacy_diff["removed_paths"],
            "changed_size_paths": legacy_diff["changed_size_paths"],
            "changed_modification_time_paths": legacy_diff["changed_modification_time_paths"],
            "legacy_tree_unchanged": legacy_diff["raw_tree_unchanged"],
        },
        "copy_execution": {
            "verified_copy_executed": verified_copy_executed,
            "move_or_delete_executed": move_or_delete_executed,
            "source_files_retained": source_files_retained,
            "approved_rows_accounted": len(approved_rows),
            "deferred_rows_accounted": len(deferred_rows),
            "approved_rows_copied_verified": copied_verified,
            "approved_rows_already_present_verified": already_present_verified,
            "copy_status_counts": copy_status_counts,
            "source_bytes_accounted": sum(int(row["source_size_bytes"]) for row in approved_rows),
            "deferred_source_bytes": sum(int(row["source_size_bytes"]) for row in deferred_rows),
            "journal_path": journal_path.as_posix() if journal_path is not None else "",
            "results_path": results_path.as_posix() if results_path is not None else "",
        },
        "filesystem": {
            "disk_free_before_bytes": disk_free_before,
            "disk_free_after_bytes": disk_free_after,
        },
        "acceptance": {
            "all_approved_rows_completed": copied_verified + already_present_verified == len(approved_rows),
            "zero_destination_conflicts": copy_status_counts.get("destination_conflict", 0) == 0,
            "zero_overwrites": not move_or_delete_executed,
            "zero_source_changes": legacy_diff["raw_tree_unchanged"] and raw_diff["raw_tree_unchanged"],
            "zero_unmapped_copied": len(deferred_rows) == EXPECTED_PHASE2A_DEFERRED_ROWS,
        },
    }


def run_phase2b_migration(
    config: ProjectConfig,
    *,
    phase2a_plan_path: str | Path,
    approval_token: str | None = None,
    execute_copy: bool = False,
    output_dir: str | Path | None = None,
    frozen: FrozenPhase2ABaseline = FrozenPhase2ABaseline(),
    repo_root: str | Path | None = None,
    write_outputs: bool = True,
) -> Phase2BRunSummary:
    if execute_copy and _normalize_token(approval_token) != APPROVAL_TOKEN:
        raise Phase2BMigrationError("Missing or invalid approval token for execute-copy")

    plan_path = Path(phase2a_plan_path).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Phase 2A migration-plan CSV was not found: {plan_path}")
    phase2a_report = _load_phase2a_report(plan_path)
    _validate_phase2a_report(phase2a_report, frozen)
    plan_sha256 = _sha256_file(plan_path)
    if plan_sha256 != frozen.migration_plan_sha256:
        raise Phase2BMigrationError(f"Phase 2A plan hash mismatch: expected {frozen.migration_plan_sha256}, got {plan_sha256}")
    plan_rows = _read_rows(plan_path)
    if len(plan_rows) != frozen.migration_plan_row_count:
        raise Phase2BMigrationError(f"Phase 2A plan row-count mismatch: expected {frozen.migration_plan_row_count}, got {len(plan_rows)}")

    repo_root_path = Path(repo_root).expanduser().resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    repo_commit = _git_commit(repo_root_path)
    repo_status = _git_status(repo_root_path)
    if repo_status:
        raise Phase2BMigrationError(f"Repository must be clean before Phase 2B execution: {repo_status}")

    acq_catalog_path = Path(phase2a_report["catalog"]["files"]["acquisitions.generated.csv"]["path"])
    scope_inventory, scope_plan, _, scope_summary = build_legacy_derivatives_audit(
        config,
        acquisition_catalog_path=acq_catalog_path,
        output_dir=config.paths.derivatives_root / "_catalog" / "phase2b_migration",
        write_outputs=False,
        return_summary=True,
    )
    if len(scope_inventory) != frozen.inventory_row_count or len(scope_plan) != frozen.migration_plan_row_count:
        raise Phase2BMigrationError("Phase 2A allowlisted scope no longer matches the frozen inventory counts")

    approved_rows, deferred_rows = _split_plan_rows(plan_rows, frozen, plan_sha256, repo_commit, _normalize_token(approval_token))

    legacy_root = Path(config.paths.legacy_fucci_tri_root).expanduser().resolve()
    if not legacy_root.is_dir():
        raise FileNotFoundError(f"legacy_fucci_tri_root is unreadable: {legacy_root}")

    target_paths = [Path(row["proposed_target"]).expanduser().resolve() for row in approved_rows]
    if len(target_paths) != len(set(target_paths)):
        raise Phase2BMigrationError("Approved targets are not unique")
    for target_path in target_paths:
        _validate_target_path(target_path, config, legacy_root)

    output_dir_path = validate_output_path(output_dir if output_dir is not None else (config.paths.derivatives_root / "_catalog" / "phase2b_migration"), config)

    free_before = shutil.disk_usage(config.paths.derivatives_root).free
    approved_source_bytes = sum(int(row["source_size_bytes"]) for row in approved_rows)
    largest_approved = max((int(row["source_size_bytes"]) for row in approved_rows), default=0)
    required_free = max(MIN_FREE_SPACE_BYTES, approved_source_bytes + largest_approved + 1024 * 1024 * 1024)
    if free_before < required_free:
        raise Phase2BMigrationError(f"Insufficient disk space: free={free_before}, required={required_free}")

    source_snapshot: dict[str, tuple[int, int]] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    for row in approved_rows:
        source_path = Path(row["source_path"]).expanduser().resolve()
        _validate_source_path(source_path, legacy_root)
        size_bytes, mtime_ns, mtime_utc = _plan_snapshot(source_path)
        source_snapshot[source_path.as_posix()] = (size_bytes, mtime_ns)
        source_meta[source_path.as_posix()] = {"size_bytes": size_bytes, "mtime_ns": mtime_ns, "mtime_utc": mtime_utc}

    raw_before_rows, raw_before_sha = write_tree_inventory(config.paths.raw_root, output_dir_path / "raw_tree_before.csv")
    if raw_before_sha != frozen.raw_tree_sha256:
        raise Phase2BMigrationError(f"Raw-tree fingerprint mismatch: expected {frozen.raw_tree_sha256}, got {raw_before_sha}")
    legacy_before_rows, _ = write_tree_inventory(legacy_root, output_dir_path / "legacy_tree_before.csv")

    approved_plan_path = output_dir_path / "phase2b_approved_copy_plan.csv"
    deferred_plan_path = output_dir_path / "phase2b_deferred_unmapped.csv"
    preflight_path = output_dir_path / "phase2b_preflight.json"
    journal_path = output_dir_path / "phase2b_copy_journal.jsonl"
    results_path = output_dir_path / "phase2b_copy_results.csv"
    report_path = output_dir_path / "phase2b_report.json"
    raw_before_path = output_dir_path / "raw_tree_before.csv"
    raw_after_path = output_dir_path / "raw_tree_after.csv"
    legacy_before_path = output_dir_path / "legacy_tree_before.csv"
    legacy_after_path = output_dir_path / "legacy_tree_after.csv"
    raw_comparison_path = output_dir_path / "raw_tree_comparison.json"
    legacy_comparison_path = output_dir_path / "legacy_tree_comparison.json"

    if write_outputs:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        _write_csv_atomic(approved_plan_path, approved_rows, APPROVED_PLAN_FIELDNAMES)
        _write_csv_atomic(deferred_plan_path, deferred_rows, DEFERRED_PLAN_FIELDNAMES)

    preflight = {
        "generated_utc": _utc_now(),
        "repository": {"commit": repo_commit, "status": repo_status, "path": repo_root_path.as_posix()},
        "phase2a_baseline": {
            "plan_path": plan_path.as_posix(),
            "plan_sha256": plan_sha256,
            "inventory_row_count": frozen.inventory_row_count,
            "migration_plan_row_count": frozen.migration_plan_row_count,
            "resolved_rows_with_targets": frozen.resolved_rows_with_targets,
            "deferred_unmapped_rows": frozen.deferred_unmapped_rows,
            "resolved_source_bytes": frozen.resolved_source_bytes,
            "deferred_source_bytes": frozen.deferred_source_bytes,
            "inventory_sha256": frozen.inventory_sha256,
            "migration_plan_sha256": frozen.migration_plan_sha256,
            "acquisition_catalog_sha256": frozen.acquisition_catalog_sha256,
            "raw_tree_sha256": frozen.raw_tree_sha256,
        },
        "catalog": phase2a_report["catalog"],
        "live_inventory": phase2a_report["live_inventory"],
        "manifest_plans": phase2a_report["manifest_plans"],
        "phase2a_scope_summary": {
            "included_roots": list(scope_summary.included_roots),
            "ignored_top_level_entries": [f"{name}:{kind}" for name, kind in scope_summary.ignored_top_level_entries],
        },
        "filesystem": {
            "raw_root": config.paths.raw_root.as_posix(),
            "derivatives_root": config.paths.derivatives_root.as_posix(),
            "legacy_root": legacy_root.as_posix(),
            "free_before_bytes": free_before,
            "required_free_bytes": required_free,
            "sufficient_free_space": free_before >= required_free,
            "approved_target_count": len(target_paths),
        },
        "sources": {row["source_path"]: source_meta[row["source_path"]] for row in approved_rows},
        "checks": {
            "plan_hash_matches": plan_sha256 == frozen.migration_plan_sha256,
            "approved_rows": len(approved_rows) == frozen.resolved_rows_with_targets,
            "deferred_rows": len(deferred_rows) == frozen.deferred_unmapped_rows,
            "raw_tree_hash_matches": raw_before_sha == frozen.raw_tree_sha256,
            "unique_targets": len(target_paths) == len(set(target_paths)),
            "repo_clean": repo_status == "",
        },
        "status": "passed",
    }
    if write_outputs:
        _write_json_atomic(preflight_path, preflight)

    if not execute_copy:
        raw_after_rows = list(raw_before_rows)
        legacy_after_rows = list(legacy_before_rows)
        if write_outputs:
            _write_csv_atomic(raw_after_path, raw_after_rows, TREE_FIELDNAMES)
            _write_csv_atomic(legacy_after_path, legacy_after_rows, TREE_FIELDNAMES)
            _write_json_atomic(raw_comparison_path, compare_tree_rows(raw_before_rows, raw_after_rows))
            _write_json_atomic(legacy_comparison_path, compare_tree_rows(legacy_before_rows, legacy_after_rows))
            report = _phase2b_report(
                repo_commit=repo_commit,
                repo_status=repo_status,
                plan_path=plan_path,
                plan_sha256=plan_sha256,
                approved_rows=approved_rows,
                deferred_rows=deferred_rows,
                scope_summary=scope_summary,
                preflight_path=preflight_path,
                raw_before_path=raw_before_path,
                raw_after_path=raw_after_path,
                legacy_before_path=legacy_before_path,
                legacy_after_path=legacy_after_path,
                raw_before_rows=raw_before_rows,
                raw_after_rows=raw_after_rows,
                legacy_before_rows=legacy_before_rows,
                legacy_after_rows=legacy_after_rows,
                copy_results=[],
                copied_verified=0,
                already_present_verified=0,
                disk_free_before=free_before,
                disk_free_after=None,
                verified_copy_executed=False,
                move_or_delete_executed=False,
                source_files_retained=True,
                approved_plan_path=approved_plan_path,
                deferred_plan_path=deferred_plan_path,
                journal_path=None,
                results_path=None,
                phase2a_report=phase2a_report,
            )
            _write_json_atomic(report_path, report)
        return Phase2BRunSummary(
            repo_commit=repo_commit,
            plan_sha256=plan_sha256,
            approved_rows=len(approved_rows),
            deferred_rows=len(deferred_rows),
            approved_source_bytes=approved_source_bytes,
            deferred_source_bytes=sum(int(row["source_size_bytes"]) for row in deferred_rows),
            copied_verified=0,
            already_present_verified=0,
            source_bytes_accounted=0,
            disk_free_before_bytes=free_before,
            disk_free_after_bytes=None,
            preflight_path=preflight_path,
            report_path=report_path,
            journal_path=None,
            results_path=None,
            approved_plan_path=approved_plan_path,
            deferred_plan_path=deferred_plan_path,
            legacy_tree_before_path=legacy_before_path,
            legacy_tree_after_path=legacy_after_path,
            raw_tree_before_path=raw_before_path,
            raw_tree_after_path=raw_after_path,
            legacy_tree_unchanged=True,
            raw_tree_unchanged=True,
            verified_copy_executed=False,
            move_or_delete_executed=False,
            source_files_retained=True,
        )

    if not journal_path.exists() and any(path.exists() for path in target_paths):
        raise Phase2BMigrationError("Targets already exist before first execution")

    journal_records = _journal_records(journal_path, plan_sha256)
    copy_results: list[dict[str, Any]] = []
    copied_verified = 0
    already_present_verified = 0
    source_bytes_accounted = 0
    started = time.monotonic()
    for sequence, row in enumerate(approved_rows, start=1):
        result = _copy_verified(
            row,
            config=config,
            legacy_root=legacy_root,
            source_snapshot=source_snapshot,
            journal_records=journal_records,
            journal_path=journal_path,
            sequence=sequence,
        )
        copy_results.append(result)
        source_bytes_accounted += int(result["source_size_bytes"])
        if result["status"] == "copied_verified":
            copied_verified += 1
        elif result["status"] == "already_present_verified":
            already_present_verified += 1
        else:
            raise Phase2BMigrationError(f"Unexpected copy result status: {result['status']}")
        journal_record = {
            **result,
            "plan_sha256": plan_sha256,
            "phase2a_repository_commit": frozen.repository_commit,
            "phase2b_executor_commit": repo_commit,
            "source_plan_row_count": frozen.migration_plan_row_count,
            "journaled_utc": _utc_now(),
        }
        _append_journal(journal_path, journal_record)
        journal_records[result["target_path"]] = journal_record
        if sequence == 1 or sequence == len(approved_rows) or sequence % 100 == 0:
            print(_progress_message(sequence, len(approved_rows), source_bytes_accounted, started))

    raw_after_rows, _ = write_tree_inventory(config.paths.raw_root, raw_after_path)
    legacy_after_rows, _ = write_tree_inventory(legacy_root, legacy_after_path)
    if write_outputs:
        _write_json_atomic(raw_comparison_path, compare_tree_rows(raw_before_rows, raw_after_rows))
        _write_json_atomic(legacy_comparison_path, compare_tree_rows(legacy_before_rows, legacy_after_rows))
        _write_csv_atomic(results_path, copy_results, COPY_RESULT_FIELDNAMES)
    raw_after_sha = _sha256_file(raw_after_path)
    if raw_after_sha != raw_before_sha:
        raise Phase2BMigrationError("Raw-tree fingerprint changed during Phase 2B")
    disk_free_after = shutil.disk_usage(config.paths.derivatives_root).free
    if write_outputs:
        report = _phase2b_report(
            repo_commit=repo_commit,
            repo_status=repo_status,
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            approved_rows=approved_rows,
            deferred_rows=deferred_rows,
            scope_summary=scope_summary,
            preflight_path=preflight_path,
            raw_before_path=raw_before_path,
            raw_after_path=raw_after_path,
            legacy_before_path=legacy_before_path,
            legacy_after_path=legacy_after_path,
            raw_before_rows=raw_before_rows,
            raw_after_rows=raw_after_rows,
            legacy_before_rows=legacy_before_rows,
            legacy_after_rows=legacy_after_rows,
            copy_results=copy_results,
            copied_verified=copied_verified,
            already_present_verified=already_present_verified,
            disk_free_before=free_before,
            disk_free_after=disk_free_after,
            verified_copy_executed=True,
            move_or_delete_executed=False,
            source_files_retained=True,
            approved_plan_path=approved_plan_path,
            deferred_plan_path=deferred_plan_path,
            journal_path=journal_path,
            results_path=results_path,
            phase2a_report=phase2a_report,
        )
        _write_json_atomic(report_path, report)
    return Phase2BRunSummary(
        repo_commit=repo_commit,
        plan_sha256=plan_sha256,
        approved_rows=len(approved_rows),
        deferred_rows=len(deferred_rows),
        approved_source_bytes=approved_source_bytes,
        deferred_source_bytes=sum(int(row["source_size_bytes"]) for row in deferred_rows),
        copied_verified=copied_verified,
        already_present_verified=already_present_verified,
        source_bytes_accounted=source_bytes_accounted,
        disk_free_before_bytes=free_before,
        disk_free_after_bytes=disk_free_after,
        preflight_path=preflight_path,
        report_path=report_path,
        journal_path=journal_path,
        results_path=results_path,
        approved_plan_path=approved_plan_path,
        deferred_plan_path=deferred_plan_path,
        legacy_tree_before_path=legacy_before_path,
        legacy_tree_after_path=legacy_after_path,
        raw_tree_before_path=raw_before_path,
        raw_tree_after_path=raw_after_path,
        legacy_tree_unchanged=compare_tree_rows(legacy_before_rows, legacy_after_rows)["raw_tree_unchanged"],
        raw_tree_unchanged=compare_tree_rows(raw_before_rows, raw_after_rows)["raw_tree_unchanged"],
        verified_copy_executed=True,
        move_or_delete_executed=False,
        source_files_retained=True,
    )
