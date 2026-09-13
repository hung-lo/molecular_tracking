"""State-blind PNG review plots for endpoint stitch proposals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from endpoint_review_plots import _coordinate_in_session, _crop, _load_array
    from affine_overlap_matcher import VoxelSpacing
except ImportError:  # pragma: no cover
    from plotting.endpoint_review_plots import _coordinate_in_session, _crop, _load_array
    from matching.affine_overlap_matcher import VoxelSpacing


def _text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _first_array(*values: Any) -> np.ndarray | None:
    for value in values:
        array = _load_array(value)
        if array is not None:
            return array
    return None


def _panel(proposal: pd.Series, path: Path) -> None:
    fig, (ax, info) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    source = np.array([proposal.get("predicted_y_um", np.nan), proposal.get("predicted_x_um", np.nan)], dtype=float)
    target = np.array([proposal.get("target_y_um", np.nan), proposal.get("target_x_um", np.nan)], dtype=float)
    if np.isfinite(source).all():
        ax.scatter(source[1], source[0], marker="x", s=90, linewidth=2, color="#1f77b4", label="projected endpoint")
    if np.isfinite(target).all():
        ax.scatter(target[1], target[0], marker="o", s=80, facecolors="none", linewidth=2, color="#d62728", label="track start")
    ax.set(title="State-blind endpoint geometry", xlabel="x (µm)", ylabel="y (µm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.2)
    if ax.collections:
        ax.legend(loc="best", fontsize=8)
    fields = [
        ("source", proposal.get("source_track_uid")), ("target", proposal.get("target_track_uid")),
        ("sessions", f"{_text(proposal.get('source_session_id'))} → {_text(proposal.get('target_session_id'))}"),
        ("gap", proposal.get("session_gap")), ("tier", proposal.get("candidate_tier")),
        ("distance µm", proposal.get("projected_distance_um")),
        ("forward/reverse rank", f"{_text(proposal.get('forward_rank_track_starts'))}/{_text(proposal.get('reverse_rank_source_endpoints'))}"),
        ("anchor n/median µm", f"{_text(proposal.get('anchor_support_count'))}/{_text(proposal.get('anchor_residual_median_um'))}"),
        ("history/future n", f"{_text(proposal.get('source_history_n'))}/{_text(proposal.get('target_future_n'))}"),
        ("review", proposal.get("review_reasons")), ("reject", proposal.get("rejection_reasons")),
    ]
    info.axis("off")
    info.text(0, 1, "\n".join(f"{name}: {_text(value)}" for name, value in fields), va="top", fontsize=9, wrap=True)
    fig.suptitle(_text(proposal.get("stitch_edge_id")), fontsize=11)
    fig.subplots_adjust(left=.08, right=.98, bottom=.12, top=.88, wspace=.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _image_panel(
    proposal: pd.Series,
    sessions: pd.DataFrame,
    features: pd.DataFrame,
    transforms: pd.DataFrame,
    path: Path,
    spacing: VoxelSpacing,
    crop_radius_um: float,
) -> bool:
    sessions = sessions.sort_values("session_index").reset_index(drop=True)
    source_matches = sessions.index[sessions["session_index"].astype(int).eq(int(proposal["source_session_index"]))]
    target_matches = sessions.index[sessions["session_index"].astype(int).eq(int(proposal["target_session_index"]))]
    if len(source_matches) == 0 or len(target_matches) == 0:
        return False
    source_position, target_position = int(source_matches[0]), int(target_matches[0])
    source_id, source_label = _text(proposal.get("source_session_id")), int(proposal["source_label"])
    source_features = features.loc[features["session_id"].astype(str).eq(source_id) & features["label"].astype(int).eq(source_label)]
    if source_features.empty:
        return False
    source_coordinate = source_features.iloc[0][["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    positions = list(range(max(0, source_position - 2), min(len(sessions), target_position + 3)))
    arrays = [_first_array(session.get("red_image_path", session.get("red")), session.get("green_image_path", session.get("green")), session.get("mask_path")) for _, session in sessions.iloc[positions].iterrows()]
    if not any(array is not None for array in arrays):
        return False
    radius = (max(1, int(round(crop_radius_um / spacing.y_um))), max(1, int(round(crop_radius_um / spacing.x_um))))
    fig, axes = plt.subplots(3, len(positions), figsize=(max(10, 2.35 * len(positions)), 7.3), squeeze=False)
    for column, position in enumerate(positions):
        session = sessions.iloc[position]
        coordinate = _coordinate_in_session(source_position, position, source_coordinate, sessions, transforms)
        if coordinate is None:
            coordinate = source_coordinate
        z = int(round(float(coordinate[0])))
        red = _crop(_load_array(session.get("red_image_path", session.get("red"))), coordinate, radius, z, 1)
        green = _crop(_load_array(session.get("green_image_path", session.get("green"))), coordinate, radius, z, 1)
        mask = _crop(_load_array(session.get("mask_path")), coordinate, radius, z, 1)
        for row, (image, cmap) in enumerate(((red, "Reds"), (green, "Greens"))):
            vmax = float(np.percentile(image, 99)) if np.any(image) else 1.0
            axes[row, column].imshow(image, cmap=cmap, vmin=0, vmax=max(vmax, 1e-9))
        axes[2, column].imshow(red, cmap="gray")
        labels = []
        if position == source_position:
            labels.append((int(proposal["source_label"]), "cyan"))
        if position == target_position:
            labels.append((int(proposal["target_label"]), "magenta"))
        for label, color in labels:
            if np.any(mask == label):
                axes[2, column].contour(mask == label, levels=[.5], colors=color, linewidths=1)
        for row in range(3):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
        axes[0, column].set_title(_text(session.get("session_id")), fontsize=9)
    axes[0, 0].set_ylabel("Red/raw"); axes[1, 0].set_ylabel("Green"); axes[2, 0].set_ylabel("Mask overlay")
    footer = (
        f"{_text(proposal.get('source_track_uid'))} → {_text(proposal.get('target_track_uid'))} | "
        f"gap={_text(proposal.get('session_gap'))} d={_text(proposal.get('projected_distance_um'))} µm | "
        f"rank={_text(proposal.get('forward_rank_track_starts'))}/{_text(proposal.get('reverse_rank_source_endpoints'))} | "
        f"anchors={_text(proposal.get('anchor_support_count'))}, residual={_text(proposal.get('anchor_residual_median_um'))} µm | "
        f"tier={_text(proposal.get('candidate_tier'))}"
    )
    fig.suptitle(_text(proposal.get("stitch_edge_id")), fontsize=11)
    fig.text(.02, .02, footer, fontsize=8, wrap=True)
    fig.subplots_adjust(left=.05, right=.995, bottom=.09, top=.91, wspace=.04, hspace=.08)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_stitch_review_panels(
    proposals: pd.DataFrame,
    output_dir: str | Path,
    *,
    max_panels: int | None = 100,
    sessions: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    transforms: pd.DataFrame | None = None,
    spacing: VoxelSpacing | None = None,
    crop_radius_um: float = 45.0,
) -> list[Path]:
    """Write deterministic individual PNGs and a readable contact sheet."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = proposals.loc[proposals.get("candidate_tier", pd.Series(index=proposals.index, dtype=str)).ne("reject")].copy()
    if not selected.empty:
        selected = selected.sort_values(["candidate_tier", "session_gap", "stitch_edge_id"], kind="mergesort")
    if max_panels is not None:
        selected = selected.head(max(0, int(max_panels)))
    paths: list[Path] = []
    for _, proposal in selected.iterrows():
        path = output / f"{proposal['stitch_edge_id']}.png"
        plotted = False
        if sessions is not None and features is not None and transforms is not None:
            plotted = _image_panel(proposal, sessions, features, transforms, path, spacing or VoxelSpacing(), crop_radius_um)
        if not plotted:
            _panel(proposal, path)
        paths.append(path)
    _contact_sheet(selected, output / "stitch_contact_sheet.png")
    return paths


def _contact_sheet(proposals: pd.DataFrame, path: Path) -> None:
    rows = max(1, len(proposals))
    fig, axes = plt.subplots(rows, 1, figsize=(12, max(2.5, 1.15 * rows)), squeeze=False)
    if proposals.empty:
        axes[0, 0].text(.5, .5, "No stitch proposals selected", ha="center", va="center")
        axes[0, 0].axis("off")
    else:
        for ax, (_, row) in zip(axes[:, 0], proposals.iterrows()):
            ax.axis("off")
            ax.text(0, .65, f"{row['stitch_edge_id']}  {_text(row.get('source_track_uid'))} → {_text(row.get('target_track_uid'))}", fontsize=9, weight="bold")
            ax.text(0, .2, f"gap={_text(row.get('session_gap'))}  d={_text(row.get('projected_distance_um'))} µm  tier={_text(row.get('candidate_tier'))}  reasons={_text(row.get('review_reasons'))}", fontsize=8)
    fig.subplots_adjust(left=.03, right=.99, bottom=.03, top=.98, hspace=.15)
    fig.savefig(path, dpi=140)
    plt.close(fig)
