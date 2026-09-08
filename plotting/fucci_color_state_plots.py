"""Small deterministic plots for Fucci color-state products."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(figure: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_dead_reference_robust_sd(sessions: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    if not sessions.empty:
        for mouse_id, group in sessions.groupby("mouse_id", sort=False):
            axis.plot(group["session_index"], group["residual_robust_sd_log2"], "o-", label=str(mouse_id))
        axis.legend()
    axis.set(xlabel="Session index", ylabel="Robust SD of log2 residual", title="Fucci-Dead reference spread")
    _save(figure, path)


def plot_dead_reference_residuals(sessions: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    if not sessions.empty:
        for mouse_id, group in sessions.groupby("mouse_id", sort=False):
            values = group["residual_median_log2"].to_numpy(dtype=float)
            axis.plot(group["session_index"], values, "o-", label=str(mouse_id))
        axis.legend()
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Session index", ylabel="Residual median", title="Fucci-Dead modal residual diagnostics")
    _save(figure, path)


def plot_modal_fit_sd_zones(
    native: pd.DataFrame,
    fit: pd.Series,
    reference_sd: float,
    path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    valid = native.loc[native["ratio_qc_pass"].astype(str).str.lower().isin({"true", "1", "yes", "y"})].copy()
    red = pd.to_numeric(valid["red"], errors="coerce")
    green = pd.to_numeric(valid["green"], errors="coerce")
    axis.scatter(red, green, s=8, alpha=0.25, color="0.25")
    x = np.linspace(float(red.min()), float(red.max()), 200) if len(red) else np.array([0, 1])
    predicted = float(fit["modal_intercept"]) + float(fit["modal_slope"]) * x
    axis.plot(x, predicted, color="#d62828", label="modal backbone")
    for threshold, style in ((-2, "--"), (-1, ":"), (1, ":"), (2, "--")):
        axis.plot(x, predicted * 2 ** (threshold * reference_sd), color="#457b9d", linestyle=style, linewidth=0.8)
    axis.set(xlabel="Corrected red", ylabel="Corrected green", title="Modal fit and Dead-referenced Z zones")
    axis.legend()
    _save(figure, path)


def plot_color_z_distribution(scored: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    values = pd.to_numeric(scored.get("eclipse_z", scored.get("color_z", pd.Series(dtype=float))), errors="coerce").dropna()
    axis.hist(values, bins=40, color="#457b9d", alpha=0.85)
    for threshold in (-2, -1, 1, 2):
        axis.axvline(threshold, color="#d62828", linewidth=0.8)
    axis.set(xlabel="ECLIPSE Z-score (Z_E)", ylabel="Observations", title="Fucci ECLIPSE state distribution")
    _save(figure, path)


def plot_pca(scores: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    if {"PC1", "PC2"}.issubset(scores.columns):
        axis.scatter(scores["PC1"], scores["PC2"], s=16, alpha=0.8)
    axis.set(xlabel="PC1", ylabel="PC2", title="Color-Z PCA")
    _save(figure, path)


def plot_pca_loadings(loadings: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    if not loadings.empty:
        for component, group in loadings.groupby("component", sort=True):
            axis.plot(group["elapsed_days"], group["loading"], "o-", label=str(component))
        axis.legend()
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Elapsed days", ylabel="Loading", title="Color-Z PCA loadings")
    _save(figure, path)


def plot_event_summary(summary: pd.DataFrame, path: str | Path, *, title: str, axis_label: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    if not summary.empty:
        x = summary["relative_elapsed_days"] if axis_label == "elapsed_days" else summary["relative_session_index"]
        mean_column = "mean_eclipse_z" if "mean_eclipse_z" in summary else "mean_color_z"
        sem_column = "sem_eclipse_z" if "sem_eclipse_z" in summary else "sem_color_z"
        mean = summary[mean_column].to_numpy(dtype=float)
        sem = summary[sem_column].to_numpy(dtype=float)
        axis.plot(x, mean, color="#1d3557")
        axis.fill_between(x, mean - sem, mean + sem, color="#457b9d", alpha=0.25)
    axis.axvline(0, color="#d62828", linewidth=0.8)
    axis.set(xlabel=f"Relative {axis_label}", ylabel="ECLIPSE Z-score (Z_E)", title=title)
    _save(figure, path)


def plot_comparison_distributions(tables: list[tuple[str, pd.DataFrame]], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    for label, table in tables:
        values = pd.to_numeric(table.get("eclipse_z", table.get("color_z", pd.Series(dtype=float))), errors="coerce").dropna()
        if len(values):
            axis.hist(values, bins=40, density=True, histtype="step", linewidth=1.5, label=label)
    axis.set(xlabel="ECLIPSE Z-score (Z_E)", ylabel="Density", title="ECLIPSE Z-score distributions")
    axis.legend()
    _save(figure, path)


def plot_comparison_state_percentages(table: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    if not table.empty:
        pivot = table.pivot(index="label", columns="color_state_bin", values="percentage").fillna(0)
        pivot.plot.bar(stacked=True, ax=axis)
    axis.set(ylabel="Percentage", title="Color-state occupancy")
    _save(figure, path)


def plot_comparison_modal_fits(tables: list[tuple[str, pd.DataFrame, pd.Series]], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for label, native, fit in tables:
        valid = native.loc[native["ratio_qc_pass"].astype(str).str.lower().isin({"true", "1", "yes"})]
        red = pd.to_numeric(valid["red"], errors="coerce")
        green = pd.to_numeric(valid["green"], errors="coerce")
        axis.scatter(red, green, s=7, alpha=0.15, label=label)
        if len(red):
            x = np.linspace(float(red.min()), float(red.max()), 100)
            axis.plot(x, float(fit["modal_intercept"]) + float(fit["modal_slope"]) * x, linewidth=1.5)
    axis.set(xlabel="Corrected red", ylabel="Corrected green", title="Representative modal backbones")
    axis.legend()
    _save(figure, path)
