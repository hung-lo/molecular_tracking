"""Helpers for safe publication of the weekly_registered compatibility product."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import tifffile

PUBLISHED_WEEKLY_PRODUCT_GLOBS = (
    "*_SyN.tif",
    "*_average.tif",
    "*_average_SyN.tif",
    "*_average_cp_masks.tif",
    "*_average_cp_masks_SyN.tif",
)
PUBLISHED_WEEKLY_PRODUCT_FILES = (
    "weekly_product_metadata.json",
    "crop_metadata.json",
)
STAGING_ROOT_NAME = ".staging"
METADATA_FILENAME = "weekly_product_metadata.json"


def _resolve_dir(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def stable_registered_stem(image_path: str | Path) -> str:
    """Return the stable stem used for published registered TIFF names."""

    return Path(image_path).stem.split("_crop", 1)[0]


def derive_reference_week_name_from_medoid_image(medoid_image: str | Path) -> str:
    """Derive the weekly reference name from a medoid average TIFF filename."""

    medoid_name = Path(medoid_image).name
    if not medoid_name.endswith(".tif"):
        raise ValueError(f"Medoid image must be a TIFF file: {medoid_name}")

    stem = Path(medoid_image).stem
    if not stem.endswith("_average"):
        raise ValueError(f"Medoid image must end with '_average.tif': {medoid_name}")

    week_name = stem.removesuffix("_average")
    if not week_name.startswith("week") or not week_name[4:].isdigit():
        raise ValueError(f"Medoid image must be named like 'weekN_average.tif': {medoid_name}")

    return week_name


def _resolve_reference_week_name(
    week_names: Sequence[str],
    reference_week_name: str | None,
) -> str:
    if not week_names:
        raise ValueError("week_names must contain at least one week.")

    if reference_week_name is None:
        if len(week_names) == 1:
            resolved_reference = week_names[0]
        else:
            raise ValueError(
                "reference_week_name is required when more than one week is present."
            )
    else:
        resolved_reference = reference_week_name

    if week_names.count(resolved_reference) != 1:
        raise ValueError(
            f"reference_week_name must occur exactly once in week_names: {resolved_reference}"
        )
    return resolved_reference


def build_expected_weekly_product_filenames(
    week_dict: Mapping[str, Sequence[str]],
    week_names: Sequence[str],
    *,
    reference_week_name: str | None = None,
) -> set[str]:
    """Derive the expected published file names from staged weekly inputs."""

    resolved_reference_week_name = _resolve_reference_week_name(week_names, reference_week_name)
    expected_names: set[str] = set()
    for week_name in week_names:
        if week_name not in week_dict:
            raise ValueError(f"week_dict is missing expected week key: {week_name}")
        week_list = [path for path in week_dict[week_name] if "_R" in Path(path).name]
        for image_path in week_list:
            red_name = f"{stable_registered_stem(image_path)}_SyN.tif"
            green_source = Path(image_path).with_name(Path(image_path).name.replace("_R", "_G", 1))
            green_name = f"{stable_registered_stem(green_source)}_SyN.tif"
            expected_names.add(red_name)
            expected_names.add(green_name)
        expected_names.add(f"{week_name}_average.tif")
        expected_names.add(f"{week_name}_average_cp_masks.tif")
        if week_name != resolved_reference_week_name:
            expected_names.add(f"{week_name}_average_SyN.tif")
            expected_names.add(f"{week_name}_average_cp_masks_SyN.tif")
    return expected_names


def build_weekly_product_metadata(
    *,
    crop_shape: Sequence[int] | None,
    cropping_enabled: bool,
    crop_label: str | None,
    refresh_requested: bool,
    registered_input_dir: str,
    source_registered_files: Sequence[str],
    spacing_zyx: Sequence[float],
    staging_dir: str | Path,
    weekly_output_dir: str | Path,
    week_names: Sequence[str],
    reference_week_name: str | None = None,
    published_at: str,
) -> dict[str, object]:
    """Build the notebook metadata payload for a published weekly product."""

    resolved_reference_week_name = _resolve_reference_week_name(week_names, reference_week_name)
    return {
        "crop_shape": None if crop_shape is None else [int(value) for value in crop_shape],
        "crop_label": crop_label,
        "cropping_enabled": bool(cropping_enabled),
        "refresh_requested": bool(refresh_requested),
        "registered_input_dir": registered_input_dir,
        "source_registered_files": list(source_registered_files),
        "spacing_zyx": [float(value) for value in spacing_zyx],
        "staging_dir": Path(staging_dir).as_posix(),
        "weekly_output_dir": Path(weekly_output_dir).as_posix(),
        "week_names": list(week_names),
        "reference_week_name": resolved_reference_week_name,
        "published_at": published_at,
    }


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


def _published_content_files(weekly_output_dir: str | Path) -> list[Path]:
    return [
        path
        for path in list_published_weekly_product_files(weekly_output_dir)
        if path.name not in PUBLISHED_WEEKLY_PRODUCT_FILES
    ]


def has_published_weekly_product(weekly_output_dir: str | Path) -> bool:
    """Return ``True`` when top-level compatibility files are already present."""

    return bool(list_published_weekly_product_files(weekly_output_dir))


def clear_published_weekly_product(weekly_output_dir: str | Path) -> list[Path]:
    """Delete only the allowlisted top-level weekly product files.

    Staging directories are intentionally left untouched so a concurrent or
    recoverable transaction is not destroyed accidentally.
    """

    weekly_output_dir = _resolve_dir(weekly_output_dir)
    removed: list[Path] = []
    if not weekly_output_dir.exists():
        return removed

    for path in list_published_weekly_product_files(weekly_output_dir):
        if path.is_file():
            path.unlink()
            removed.append(path)

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


def read_tiff_shape(path: str | Path) -> tuple[int, ...]:
    """Read a TIFF shape from metadata without loading the full image data."""

    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"TIFF file has no series: {path}")
        return tuple(int(value) for value in tif.series[0].shape)


def validate_staged_weekly_product(
    staging_dir: str | Path,
    *,
    expected_filenames: Iterable[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> tuple[list[Path], tuple[int, ...]]:
    """Validate a staged weekly product before any top-level files are replaced."""

    staging_dir = _resolve_dir(staging_dir)
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Weekly staging directory was not found: {staging_dir}")

    staged_files = sorted(path for path in staging_dir.iterdir() if path.is_file())
    if not staged_files:
        raise FileNotFoundError(f"Weekly staging directory is empty: {staging_dir}")

    if expected_filenames is not None:
        expected_names = sorted(str(name) for name in expected_filenames)
        staged_names = sorted(path.name for path in staged_files)
        if staged_names != expected_names:
            missing = sorted(set(expected_names) - set(staged_names))
            extra = sorted(set(staged_names) - set(expected_names))
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

    tif_shape_to_paths: dict[tuple[int, ...], list[Path]] = {}
    for path in staged_files:
        if path.suffix.lower() != ".tif":
            continue
        shape = read_tiff_shape(path)
        tif_shape_to_paths.setdefault(shape, []).append(path)
    if len(tif_shape_to_paths) != 1:
        detail = "; ".join(
            f"shape {shape}: {', '.join(path.name for path in paths)}"
            for shape, paths in sorted(tif_shape_to_paths.items(), key=lambda item: item[0])
        )
        raise ValueError(f"Prepared weekly product contains inconsistent TIFF dimensions: {detail}")

    selected_shape = next(iter(tif_shape_to_paths))
    if metadata is not None:
        cropping_enabled = bool(metadata.get("cropping_enabled", False))
        crop_shape = metadata.get("crop_shape")
        if cropping_enabled:
            if crop_shape is None:
                raise ValueError("Weekly product metadata marked cropping_enabled=True but crop_shape is missing.")
            crop_shape_tuple = tuple(int(value) for value in crop_shape)
            if tuple(int(value) for value in selected_shape[-2:]) != crop_shape_tuple:
                raise ValueError(
                    f"Prepared weekly product crop_shape {crop_shape_tuple} does not match TIFF dimensions {selected_shape}"
                )
        elif crop_shape is not None:
            raise ValueError("Weekly product metadata must set crop_shape to null when cropping is disabled.")

    return staged_files, selected_shape


def validate_published_weekly_product(
    weekly_output_dir: str | Path,
    *,
    metadata_filename: str = METADATA_FILENAME,
    require_metadata: bool = False,
) -> dict[str, object] | None:
    """Validate a published weekly product against the metadata manifest."""

    weekly_output_dir = _resolve_dir(weekly_output_dir)
    content_files = _published_content_files(weekly_output_dir)
    metadata_path = weekly_output_dir / metadata_filename

    if not metadata_path.is_file():
        if require_metadata:
            raise FileNotFoundError(f"Weekly product metadata was not found: {metadata_path}")
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    published_files = metadata.get("published_files")
    if not isinstance(published_files, list):
        raise ValueError("Weekly product metadata must include a list named 'published_files'.")

    actual_names = sorted(path.name for path in content_files)
    expected_names = sorted(str(name) for name in published_files)
    if actual_names != expected_names:
        raise ValueError(
            "Weekly product metadata does not match the current allowlisted files: "
            f"expected={expected_names}, actual={actual_names}"
        )

    published_sha256 = metadata.get("published_sha256")
    if not isinstance(published_sha256, dict):
        raise ValueError("Weekly product metadata must include a mapping named 'published_sha256'.")

    expected_hash_names = sorted(str(name) for name in published_sha256)
    if expected_hash_names != expected_names:
        raise ValueError(
            "Weekly product metadata hashes do not match the published file list: "
            f"expected={expected_names}, hash_keys={expected_hash_names}"
        )

    for path in content_files:
        actual_hash = sha256(path)
        expected_hash = str(published_sha256.get(path.name))
        if actual_hash != expected_hash:
            raise ValueError(
                f"Weekly product hash mismatch for {path.name}: expected {expected_hash}, got {actual_hash}"
            )

    return metadata


def _backup_current_weekly_product(
    weekly_output_dir: Path,
    backup_dir: Path,
    *,
    metadata_filename: str,
) -> list[Path]:
    backup_dir.mkdir(parents=True, exist_ok=False)
    moved: list[Path] = []
    for path in list_published_weekly_product_files(weekly_output_dir):
        target = backup_dir / path.name
        os.replace(path, target)
        moved.append(target)
    metadata_path = weekly_output_dir / metadata_filename
    if metadata_path.is_file() and metadata_path not in moved:
        target = backup_dir / metadata_path.name
        os.replace(metadata_path, target)
        moved.append(target)
    return moved


def _restore_weekly_product_from_backup(
    weekly_output_dir: Path,
    backup_dir: Path,
) -> None:
    for path in sorted(backup_dir.iterdir(), key=lambda value: value.name):
        if path.is_file():
            os.replace(path, weekly_output_dir / path.name)


def _remove_allowlisted_weekly_product_files(weekly_output_dir: Path, metadata_filename: str) -> None:
    for path in list_published_weekly_product_files(weekly_output_dir):
        if path.is_file():
            path.unlink()
    metadata_path = weekly_output_dir / metadata_filename
    if metadata_path.is_file():
        metadata_path.unlink()


def publish_staged_weekly_product(
    staging_dir: str | Path,
    weekly_output_dir: str | Path,
    *,
    expected_filenames: Iterable[str] | None = None,
    metadata: Mapping[str, object] | None = None,
    metadata_filename: str = METADATA_FILENAME,
    replace_existing: bool = False,
    require_metadata: bool = False,
) -> Path:
    """Publish staged weekly product files transactionally.

    Staged files are validated first. If an existing published product is being
    replaced, the current allowlisted files are backed up, the staged product is
    published, and the metadata file is written last. Any failure rolls the top
    level back to the exact previous published state.
    """

    staging_dir = _resolve_dir(staging_dir)
    weekly_output_dir = _resolve_dir(weekly_output_dir)
    if not weekly_output_dir.is_dir():
        raise FileNotFoundError(f"Weekly output directory was not found: {weekly_output_dir}")

    staged_files, _selected_shape = validate_staged_weekly_product(
        staging_dir,
        expected_filenames=expected_filenames,
        metadata=metadata,
    )

    existing_files = list_published_weekly_product_files(weekly_output_dir)
    if existing_files and not replace_existing:
        existing_str = ", ".join(path.name for path in existing_files)
        raise FileExistsError(
            "weekly_registered already contains published compatibility files: "
            f"{existing_str}. Set REFRESH_WEEKLY_PRODUCT=True to replace them."
        )

    transaction_root = None
    backup_dir = None
    if existing_files:
        transaction_root = weekly_output_dir / STAGING_ROOT_NAME / f"txn_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        backup_dir = transaction_root / "backup"
        try:
            _backup_current_weekly_product(
                weekly_output_dir,
                backup_dir,
                metadata_filename=metadata_filename,
            )
        except Exception:
            if backup_dir is not None and backup_dir.is_dir():
                _restore_weekly_product_from_backup(weekly_output_dir, backup_dir)
            raise

    committed = False
    try:
        for staged_path in staged_files:
            os.replace(staged_path, weekly_output_dir / staged_path.name)

        content_files = _published_content_files(weekly_output_dir)
        content_manifest = {
            **(metadata or {}),
            "published_files": [path.name for path in sorted(content_files, key=lambda path: path.name)],
            "published_sha256": {
                path.name: sha256(path)
                for path in sorted(content_files, key=lambda path: path.name)
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
        tmp_path = Path(metadata_tmp.name)
        try:
            with metadata_tmp as handle:
                json.dump(content_manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, metadata_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        validate_published_weekly_product(
            weekly_output_dir,
            metadata_filename=metadata_filename,
            require_metadata=require_metadata or metadata is not None,
        )
        committed = True
        return metadata_path
    except Exception:
        _remove_allowlisted_weekly_product_files(weekly_output_dir, metadata_filename)
        if backup_dir is not None and backup_dir.is_dir():
            _restore_weekly_product_from_backup(weekly_output_dir, backup_dir)
        raise
    finally:
        if committed:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if transaction_root is not None and transaction_root.exists():
                shutil.rmtree(transaction_root)
