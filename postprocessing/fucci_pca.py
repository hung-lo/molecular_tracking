"""NumPy-only complete-case PCA for color-Z trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PCAResult:
    identifiers: pd.DataFrame
    session_ids: tuple[str, ...]
    centered_matrix: np.ndarray
    scores: np.ndarray
    loadings: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    column_means: np.ndarray


def compute_color_z_pca(
    matrix: pd.DataFrame | np.ndarray,
    session_ids: list[str] | tuple[str, ...] | None = None,
) -> PCAResult:
    """Compute centered SVD PCA and orient each component deterministically."""

    if isinstance(matrix, pd.DataFrame):
        identifiers = matrix[[column for column in ("match_policy", "track_uid", "roi_id") if column in matrix]].reset_index(drop=True)
        if session_ids is None:
            session_ids = tuple(str(column) for column in matrix.columns if column not in identifiers.columns)
        values = matrix.loc[:, list(session_ids)].to_numpy(dtype=float)
    else:
        identifiers = pd.DataFrame()
        if session_ids is None:
            session_ids = tuple(str(index) for index in range(np.asarray(matrix).shape[1]))
        values = np.asarray(matrix, dtype=float)
    session_ids = tuple(str(value) for value in session_ids)
    if values.ndim != 2 or values.shape[1] != len(session_ids):
        raise ValueError("PCA matrix shape does not match session_ids")
    if values.shape[0] == 0:
        raise ValueError("PCA requires at least one complete-case row")
    if not np.isfinite(values).all():
        raise ValueError("PCA requires complete-case values without NaN")
    means = values.mean(axis=0)
    centered = values - means
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    for component in range(vt.shape[0]):
        pivot = int(np.argmax(np.abs(vt[component])))
        if vt[component, pivot] < 0:
            vt[component] *= -1
            u[:, component] *= -1
    denominator = max(values.shape[0] - 1, 1)
    explained = singular_values**2 / denominator
    total = float(explained.sum())
    ratios = explained / total if total > 0 else np.zeros_like(explained)
    return PCAResult(
        identifiers=identifiers,
        session_ids=session_ids,
        centered_matrix=centered,
        scores=u * singular_values,
        loadings=vt,
        singular_values=singular_values,
        explained_variance=explained,
        explained_variance_ratio=ratios,
        column_means=means,
    )


def pca_output_tables(
    result: PCAResult,
    session_metadata: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Convert a PCA result into the canonical score, loading, and variance tables."""

    scores = result.identifiers.copy()
    for index in range(result.scores.shape[1]):
        scores[f"PC{index + 1}"] = result.scores[:, index]
    metadata = session_metadata.copy() if session_metadata is not None else pd.DataFrame({"session_id": result.session_ids})
    metadata["session_id"] = metadata["session_id"].astype(str)
    loading_rows: list[dict[str, Any]] = []
    for component, values in enumerate(result.loadings, start=1):
        for session_id, loading in zip(result.session_ids, values, strict=True):
            row: dict[str, Any] = {"component": f"PC{component}", "session_id": session_id, "loading": float(loading)}
            match = metadata.loc[metadata["session_id"].eq(session_id)]
            if not match.empty:
                source = match.iloc[0]
                for key in ("session_index", "acquisition_date", "elapsed_days"):
                    if key in source:
                        row[key] = source[key]
            loading_rows.append(row)
    variance = pd.DataFrame({
        "component": [f"PC{index + 1}" for index in range(len(result.singular_values))],
        "singular_value": result.singular_values,
        "explained_variance": result.explained_variance,
        "explained_variance_ratio": result.explained_variance_ratio,
    })
    variance["cumulative_explained_variance_ratio"] = variance["explained_variance_ratio"].cumsum()
    centering = pd.DataFrame({"session_id": result.session_ids, "n_complete_cases": len(result.identifiers), "mean_before_centering": result.column_means, "mean_after_centering": result.centered_matrix.mean(axis=0)})
    return {
        "scores": scores,
        "loadings": pd.DataFrame(loading_rows),
        "explained_variance": variance,
        "centering_summary": centering,
    }


compute_pca = compute_color_z_pca
