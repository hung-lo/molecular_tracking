"""Helpers for safe publication of the weekly_registered compatibility product."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from typing import Iterable

PUBLISHED_WEEKLY_PRODUCT_GLOBS = (
    "*_SyN.tif",
    "*_average.tif",
    "*_average_cp_masks.tif",
)
PUBLISHED_WEEKLY_PRODUCT_FILES = (
    "weekly_product_metadata.json",
    "crop_metadata.json",
)
STAGING_ROOT_NAME = ".staging"


def _resolve_dir(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def list_published_weekly_product_files(weekly_output_dir: str | Path) -> list[Path]:
    """Return top-level published weekly product files, excluding staging."""

    weekly_output_dir = _resolve_dir(weekly_output_dir)
    files: list[Path] = []
    for pattern in PUBLISHED_WEEKLY_PRODUCT_GLOBS:
        files.extend(path for path in weekly_output_dir.glob(pattern) if path.is_file())
    for file_name in PUBLISHED_WEEKLY_PRODUCT_FILES:
        path = weekly_output_dir / file_name
        if path.is_file():
            files.append(path)
    return sorted({path.resolve() for path in files})


def has_published_weekly_product(weekly_output_dir: str | Path) -> bool:
    """Return ``True`` when top-level compatibility files are already present."""

    return bool(list_published_weekly_product_files(weekly_output_dir))


def clear_published_weekly_product(weekly_output_dir: str | Path) -> list[Path]:
    """Delete only the allowlisted top-level weekly product files and staging tree."""

    weekly_output_dir = _resolve_dir(weekly_output_dir)
    removed: list[Path] = []
    if not weekly_output_dir.exists():
        return removed

    for path in list_published_weekly_product_files(weekly_output_dir):
        if path.is_file():
            path.unlink()
            removed.append(path)

    staging_root = weekly_output_dir / STAGING_ROOT_NAME
    if staging_root.is_dir():
        shutil.rmtree(staging_root)
        removed.append(staging_root)

    return removed


def prepare_weekly_product_workspace(
    weekly_output_dir: str | Path,
    *,
    refresh: bool = False,
) -> Path:
    """Prepare a clean staging directory for the weekly compatibility product."""

    weekly_output_dir = _resolve_dir(weekly_output_dir)
    weekly_output_dir.mkdir(parents=True, exist_ok=True)

    existing = list_published_weekly_product_files(weekly_output_dir)
    if existing and not refresh:
        existing_str = ", ".join(path.name for path in existing)
        raise FileExistsError(
            "weekly_registered already contains published compatibility files: "
            f"{existing_str}. Set REFRESH_WEEKLY_PRODUCT=True to replace them."
        )

    if refresh:
        clear_published_weekly_product(weekly_output_dir)

    staging_root = weekly_output_dir / STAGING_ROOT_NAME
    staging_dir = staging_root / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_staged_weekly_product(
    staging_dir: str | Path,
    weekly_output_dir: str | Path,
    *,
    expected_filenames: Iterable[str] | None = None,
    metadata: dict[str, object] | None = None,
    metadata_filename: str = "weekly_product_metadata.json",
) -> Path:
    """Publish staged weekly product files atomically one-by-one.

    The staged files are moved into the top-level compatibility directory using
    ``os.replace``. The metadata file is written only after the final file set is
    in place.
    """

    staging_dir = _resolve_dir(staging_dir)
    weekly_output_dir = _resolve_dir(weekly_output_dir)
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Weekly staging directory was not found: {staging_dir}")
    if not weekly_output_dir.is_dir():
        raise FileNotFoundError(f"Weekly output directory was not found: {weekly_output_dir}")

    staged_files = sorted(path for path in staging_dir.iterdir() if path.is_file())
    if not staged_files:
        raise FileNotFoundError(f"Weekly staging directory is empty: {staging_dir}")

    if expected_filenames is not None:
        expected_names = {str(name) for name in expected_filenames}
        staged_names = {path.name for path in staged_files}
        missing = sorted(expected_names - staged_names)
        extra = sorted(staged_names - expected_names)
        if missing or extra:
            detail_parts = []
            if missing:
                detail_parts.append(f"missing={missing}")
            if extra:
                detail_parts.append(f"extra={extra}")
            detail = ", ".join(detail_parts)
            raise ValueError(
                "Weekly staged product does not match the expected published file set: "
                f"{detail}"
            )

    published_files: list[Path] = []
    for staged_path in staged_files:
        final_path = weekly_output_dir / staged_path.name
        os.replace(staged_path, final_path)
        published_files.append(final_path)

    published_manifest = {
        **(metadata or {}),
        "published_files": [path.name for path in sorted(published_files, key=lambda path: path.name)],
        "published_sha256": {
            path.name: sha256(path)
            for path in sorted(published_files, key=lambda path: path.name)
        },
        "published_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = weekly_output_dir / metadata_filename
    metadata_tmp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=weekly_output_dir,
        delete=False,
    )
    try:
        with metadata_tmp as handle:
            json.dump(published_manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(Path(metadata_tmp.name), metadata_path)
    except Exception:
        if Path(metadata_tmp.name).exists():
            Path(metadata_tmp.name).unlink(missing_ok=True)
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    return metadata_path
