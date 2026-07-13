"""Parse and validate session manifests for daywise ROI matching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import csv
from pathlib import Path


_REQUIRED_COLUMNS = {
    "session_index",
    "session_id",
    "acquisition_date",
    "mask_path",
    "red_image_path",
    "green_image_path",
    "required",
}


@dataclass(frozen=True)
class SessionRecord:
    """Describe one session in a longitudinal imaging manifest."""

    session_index: int
    session_id: str
    acquisition_date: date
    mask_path: Path
    red_image_path: Path | None = None
    green_image_path: Path | None = None
    required: bool = True


def _parse_strict_bool(value: str | None, *, field_name: str, row_number: int) -> bool:
    """Parse a manifest boolean value without permissive heuristics."""

    if value is None:
        raise ValueError(f"Row {row_number}: missing required boolean field {field_name!r}.")
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(
        f"Row {row_number}: ambiguous boolean value {value!r} for {field_name!r}; "
        "use only true/false."
    )


def _resolve_manifest_path(raw_path: str | None, manifest_dir: Path, *, field_name: str, row_number: int, allow_blank: bool = False) -> Path | None:
    """Resolve one manifest path relative to the manifest directory."""

    if raw_path is None:
        if allow_blank:
            return None
        raise ValueError(f"Row {row_number}: missing required path for {field_name!r}.")

    text = str(raw_path).strip()
    if text == "":
        if allow_blank:
            return None
        raise ValueError(f"Row {row_number}: empty required path for {field_name!r}.")

    path = Path(text)
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _parse_required_field(raw_value: str | None, *, row_number: int) -> bool:
    """Parse the required column with no implicit coercion."""

    if raw_value is None:
        raise ValueError(f"Row {row_number}: missing required column 'required'.")
    return _parse_strict_bool(raw_value, field_name="required", row_number=row_number)


def load_session_manifest(path: str | Path) -> list[SessionRecord]:
    """Load a CSV manifest and return sorted session records."""

    manifest_path = Path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file was not found: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest path is not a file: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest file is empty.")
        fieldnames = {field.strip() for field in reader.fieldnames if field is not None}
        missing_columns = sorted(_REQUIRED_COLUMNS.difference(fieldnames))
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {', '.join(missing_columns)}")

        records: list[SessionRecord] = []
        manifest_dir = manifest_path.parent
        for row_number, row in enumerate(reader, start=2):
            if row is None:
                continue

            def get(name: str) -> str | None:
                return row.get(name)

            session_index_raw = get("session_index")
            session_id_raw = get("session_id")
            acquisition_date_raw = get("acquisition_date")
            mask_path_raw = get("mask_path")
            red_image_path_raw = get("red_image_path")
            green_image_path_raw = get("green_image_path")
            required_raw = get("required")

            if session_index_raw is None or str(session_index_raw).strip() == "":
                raise ValueError(f"Row {row_number}: missing session_index.")
            try:
                session_index = int(str(session_index_raw).strip())
            except ValueError as exc:
                raise ValueError(f"Row {row_number}: invalid session_index {session_index_raw!r}.") from exc

            session_id = str(session_id_raw).strip() if session_id_raw is not None else ""
            if session_id == "":
                raise ValueError(f"Row {row_number}: empty session_id.")

            if acquisition_date_raw is None or str(acquisition_date_raw).strip() == "":
                raise ValueError(f"Row {row_number}: empty acquisition_date.")
            try:
                acquisition_dt = date.fromisoformat(str(acquisition_date_raw).strip())
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_number}: invalid acquisition_date {acquisition_date_raw!r}; "
                    "use ISO format YYYY-MM-DD."
                ) from exc

            mask_path = _resolve_manifest_path(
                mask_path_raw,
                manifest_dir,
                field_name="mask_path",
                row_number=row_number,
                allow_blank=False,
            )
            assert mask_path is not None
            red_image_path = _resolve_manifest_path(
                red_image_path_raw,
                manifest_dir,
                field_name="red_image_path",
                row_number=row_number,
                allow_blank=True,
            )
            green_image_path = _resolve_manifest_path(
                green_image_path_raw,
                manifest_dir,
                field_name="green_image_path",
                row_number=row_number,
                allow_blank=True,
            )
            required = _parse_required_field(required_raw, row_number=row_number)

            records.append(
                SessionRecord(
                    session_index=session_index,
                    session_id=session_id,
                    acquisition_date=acquisition_dt,
                    mask_path=mask_path,
                    red_image_path=red_image_path,
                    green_image_path=green_image_path,
                    required=required,
                )
            )

    records = sorted(records, key=lambda record: record.session_index)
    validate_manifest_for_matching(records)
    return records


def validate_manifest_for_matching(records: list[SessionRecord]) -> None:
    """Validate the manifest contract needed for matching-only workflows."""

    if len(records) < 2:
        raise ValueError("At least two sessions are required for matching.")

    session_indices = [record.session_index for record in records]
    expected_indices = list(range(len(records)))
    if session_indices != expected_indices:
        raise ValueError(
            "session_index values must be contiguous from 0..N-1 after sorting by session_index."
        )

    session_ids = [record.session_id for record in records]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("session_id values must be unique.")

    acquisition_dates = [record.acquisition_date for record in records]
    if any(left > right for left, right in zip(acquisition_dates, acquisition_dates[1:])):
        raise ValueError("acquisition_date values must be nondecreasing.")

    for record in records:
        if record.mask_path.name == "":
            raise ValueError(f"Session {record.session_id}: empty mask_path.")
        if not record.mask_path.exists():
            raise FileNotFoundError(f"Mask file was not found: {record.mask_path}")
        if not record.mask_path.is_file():
            raise FileNotFoundError(f"Mask path is not a file: {record.mask_path}")


def validate_manifest_for_intensity(records: list[SessionRecord]) -> None:
    """Validate the manifest contract needed for intensity extraction."""

    validate_manifest_for_matching(records)

    for record in records:
        if record.red_image_path is None or record.green_image_path is None:
            raise ValueError(
                f"Session {record.session_id}: red_image_path and green_image_path are required for intensity extraction."
            )
        if record.red_image_path == record.green_image_path:
            raise ValueError(
                f"Session {record.session_id}: red and green image paths must be distinct."
            )
        for path in (record.red_image_path, record.green_image_path):
            if not path.exists():
                raise FileNotFoundError(f"Image file was not found: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Image path is not a file: {path}")

