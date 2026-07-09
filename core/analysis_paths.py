"""Shared path helpers for multi-dataset fluorescence ROI analyses.

This module centralizes the directory layout for the current project so the
analysis scripts can switch between dataset folders such as ``1050_data`` and
``920_data`` without hardcoding one absolute path per script.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ALIASES: dict[str, str] = {
    "1050": "1050_data",
    "920": "920_data",
}


def resolve_dataset_dir(
    dataset: str | Path | None = "1050",
) -> Path:
    """Resolve a dataset label or path to a concrete dataset directory.

    Parameters
    ----------
    dataset : str, pathlib.Path, or None, default="1050"
        Dataset identifier. Supported short aliases are ``"1050"`` and
        ``"920"``. A custom absolute or relative path may also be supplied. If
        ``None``, the default dataset alias ``"1050"`` is used.

    Returns
    -------
    pathlib.Path
        Existing dataset directory. The returned path is absolute.
    """

    if dataset is None:
        dataset = "1050"

    if isinstance(dataset, Path):
        dataset_dir = dataset
    else:
        dataset_name = str(dataset)
        dataset_dir = PROJECT_ROOT / DATASET_ALIASES.get(dataset_name, dataset_name)

    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory was not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")
    return dataset_dir


def get_dataset_analysis_dir(
    dataset: str | Path | None = "1050",
) -> Path:
    """Return the canonical analysis-output directory for one dataset.

    Parameters
    ----------
    dataset : str, pathlib.Path, or None, default="1050"
        Dataset identifier accepted by :func:`resolve_dataset_dir`.

    Returns
    -------
    pathlib.Path
        Absolute path to ``<dataset_dir>/analysis``. The directory is not
        created automatically by this helper.
    """

    return resolve_dataset_dir(dataset) / "analysis"


def get_shape_qc_analysis_dir(
    dataset: str | Path | None = "1050",
) -> Path:
    """Return the current mean-merge SAM shape-QC analysis directory.

    Parameters
    ----------
    dataset : str, pathlib.Path, or None, default="1050"
        Dataset identifier accepted by :func:`resolve_dataset_dir`.

    Returns
    -------
    pathlib.Path
        Absolute path to the current shape-QC analysis folder:
        ``<dataset_dir>/analysis/roi_log_ratio_outputs_dark_median_corrected_meanMergeCPSAM_ROIs/shape_qc_filter``.
    """

    return (
        get_dataset_analysis_dir(dataset)
        / "roi_log_ratio_outputs_dark_median_corrected_meanMergeCPSAM_ROIs"
        / "shape_qc_filter"
    )
