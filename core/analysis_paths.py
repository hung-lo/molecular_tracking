"""Explicit legacy path helpers; project paths live in :mod:`project_cli`."""
from __future__ import annotations
from pathlib import Path

DATASET_ALIASES={"1050":"1050_data","920":"920_data"}

def resolve_dataset_dir(dataset: str|Path|None=None, *, legacy_root: str|Path|None=None) -> Path:
    """Resolve an explicit directory, or an alias only with an explicit legacy root."""
    if dataset is None: raise ValueError("An explicit dataset path is required; no global dataset is selected.")
    if isinstance(dataset,Path): path=dataset
    else:
        name=str(dataset)
        if name in DATASET_ALIASES:
            if legacy_root is None: raise ValueError("Legacy aliases require an explicitly configured legacy_root.")
            path=Path(legacy_root)/DATASET_ALIASES[name]
        else: path=Path(name)
    path=path.expanduser().resolve()
    if not path.exists(): raise FileNotFoundError(f"Dataset directory was not found: {path}")
    if not path.is_dir(): raise NotADirectoryError(f"Dataset path is not a directory: {path}")
    return path

def get_dataset_analysis_dir(dataset: str|Path|None=None, *, legacy_root: str|Path|None=None) -> Path:
    return resolve_dataset_dir(dataset,legacy_root=legacy_root)/"analysis"

def get_shape_qc_analysis_dir(dataset: str|Path|None=None, *, legacy_root: str|Path|None=None) -> Path:
    return get_dataset_analysis_dir(dataset,legacy_root=legacy_root)/"roi_log_ratio_outputs_dark_median_corrected_meanMergeCPSAM_ROIs"/"shape_qc_filter"
