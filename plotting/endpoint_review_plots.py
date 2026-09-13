"""PNG-only review figures for the daywise endpoint evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

try:
    from endpoint_evaluator import compose_transforms, invert_restricted_transform, apply_transform_b_to_a
except ImportError:  # pragma: no cover - package import path
    from matching.endpoint_evaluator import compose_transforms, invert_restricted_transform, apply_transform_b_to_a
try:
    from affine_overlap_matcher import VoxelSpacing
except ImportError:  # pragma: no cover
    from matching.affine_overlap_matcher import VoxelSpacing


def _text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _load_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if value is None or _text(value).strip() == "":
        return None
    path = Path(_text(value))
    if not path.is_file():
        return None
    return np.asarray(tifffile.imread(path))


def _manifest_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    table = manifest.copy()
    table["session_index"] = table["session_index"].astype(int)
    table["session_id"] = table["session_id"].astype(str)
    return table.sort_values("session_index").reset_index(drop=True)


def _feature(features: pd.DataFrame, session_id: str, label: int) -> pd.Series | None:
    if features.empty or not {"session_id", "label"}.issubset(features.columns):
        return None
    rows = features.loc[features["session_id"].astype(str).eq(str(session_id)) & features["label"].astype(int).eq(int(label))]
    return None if rows.empty else rows.iloc[0]


def _transform_map(transforms: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    if transforms.empty:
        return {}
    return {(str(row.day_a), str(row.day_b)): row for _, row in transforms.iterrows()}


def _coordinate_in_session(
    source_position: int,
    target_position: int,
    coordinate: np.ndarray,
    sessions: pd.DataFrame,
    transforms: pd.DataFrame,
) -> np.ndarray | None:
    if source_position == target_position:
        return np.asarray(coordinate, dtype=float)
    lookup = _transform_map(transforms)
    if target_position < source_position:
        stored = lookup.get((str(sessions.iloc[target_position].session_id), str(sessions.iloc[source_position].session_id)))
        if stored is None:
            return None
        return apply_transform_b_to_a(coordinate, stored)
    components: list[pd.Series] = []
    for position in range(source_position, target_position):
        stored = lookup.get((str(sessions.iloc[position].session_id), str(sessions.iloc[position + 1].session_id)))
        if stored is None:
            return None
        components.append(stored)
    composed = components[0]
    for component in components[1:]:
        composed = compose_transforms(composed, component)
    try:
        return apply_transform_b_to_a(coordinate, invert_restricted_transform(composed))
    except ValueError:
        return None


def _crop(
    array: np.ndarray | None,
    center: np.ndarray | None,
    radius_px: tuple[int, int],
    z_index: int = 0,
    z_radius: int = 0,
) -> np.ndarray:
    height = 2 * int(radius_px[0]) + 1
    width = 2 * int(radius_px[1]) + 1
    if array is None:
        return np.zeros((height, width), dtype=float)
    if array.ndim == 3:
        z = int(np.clip(round(float(z_index)), 0, array.shape[0] - 1))
        z0 = max(0, z - int(z_radius)); z1 = min(array.shape[0], z + int(z_radius) + 1)
        plane = np.max(array[z0:z1], axis=0)
    elif array.ndim == 2:
        plane = array
    else:
        raise ValueError("review images and masks must be 2D or 3D")
    if center is None:
        return np.zeros((height, width), dtype=plane.dtype)
    cy, cx = int(round(float(center[1]))), int(round(float(center[2])))
    y0, x0 = cy - radius_px[0], cx - radius_px[1]
    output = np.zeros((height, width), dtype=plane.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + height, plane.shape[0]), min(x0 + width, plane.shape[1])
    if sy1 > sy0 and sx1 > sx0:
        output[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = plane[sy0:sy1, sx0:sx1]
    return output


def _save(fig: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    if output_path is not None:
        path = Path(output_path)
        if path.suffix.lower() != ".png":
            raise ValueError("endpoint review figures must be PNG files")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_endpoint_review_panel(
    endpoint: Mapping[str, Any] | pd.Series,
    sessions: pd.DataFrame,
    features: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    transforms: pd.DataFrame | None = None,
    *,
    output_path: str | Path | None = None,
    crop_radius_um: float = 45.0,
    z_radius: int = 1,
    spacing: VoxelSpacing | None = None,
) -> plt.Figure:
    """Render the t-1…t+3 raw/overlay endpoint panel."""

    spacing = spacing or VoxelSpacing()
    if crop_radius_um <= 0 or z_radius < 0:
        raise ValueError("crop_radius_um must be positive and z_radius must be nonnegative")
    sessions = _manifest_rows(sessions)
    transforms = transforms if transforms is not None else pd.DataFrame()
    candidates = candidates if candidates is not None else pd.DataFrame()
    source_index = int(endpoint["end_session_index"])
    source_position = int(sessions.index[sessions["session_index"].eq(source_index)][0])
    source_id = str(endpoint["end_session_id"])
    source_feature = _feature(features, source_id, int(endpoint["end_label"]))
    if source_feature is None:
        source_coordinate = np.array([
            float(endpoint.get("centroid_z_um", 0.0)) / spacing.z_um,
            float(endpoint.get("centroid_y_um", 0.0)) / spacing.y_um,
            float(endpoint.get("centroid_x_um", 0.0)) / spacing.x_um,
        ])
    else:
        source_coordinate = source_feature[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
    radius_px = (max(1, int(round(crop_radius_um / spacing.y_um))), max(1, int(round(crop_radius_um / spacing.x_um))))
    relative_positions = list(range(-1, 4))
    fig, axes = plt.subplots(3, 5, figsize=(15, 8), squeeze=False, constrained_layout=True)
    candidate_subset = candidates.loc[candidates["endpoint_id"].astype(str).eq(_text(endpoint.get("endpoint_id")))] if not candidates.empty and "endpoint_id" in candidates else pd.DataFrame()
    cache: dict[str, dict[str, np.ndarray | None]] = {}

    def session_array(session: pd.Series, key: str) -> np.ndarray | None:
        session_id = str(session.session_id)
        cache.setdefault(session_id, {})
        if key not in cache[session_id]:
            cache[session_id][key] = _load_array(session.get(f"{key}_image_path", session.get(key)))
        return cache[session_id][key]

    for column, relative in enumerate(relative_positions):
        position = source_position + relative
        if not 0 <= position < len(sessions):
            for row in range(3):
                axes[row, column].axis("off")
            continue
        session = sessions.iloc[position]
        session_id = str(session.session_id)
        coordinate = _coordinate_in_session(source_position, position, source_coordinate, sessions, transforms)
        if coordinate is None:
            coordinate = source_coordinate
        z_index = int(round(float(coordinate[0])))
        red = _crop(session_array(session, "red"), coordinate, radius_px, z_index, z_radius)
        green = _crop(session_array(session, "green"), coordinate, radius_px, z_index, z_radius)
        mask = _load_array(session.get("mask_path"))
        mask_crop = _crop(mask, coordinate, radius_px, z_index, z_radius)
        for axis, image, cmap in ((axes[0, column], red, "Reds"), (axes[1, column], green, "Greens")):
            vmax = float(np.percentile(image, 99)) if np.any(image) else 1.0
            axis.imshow(image, cmap=cmap, vmin=0, vmax=max(vmax, 1e-9))
            axis.set_xticks([]); axis.set_yticks([])
        overlay = np.asarray(red, dtype=float)
        axes[2, column].imshow(overlay, cmap="gray")
        current_label = int(endpoint["end_label"]) if position == source_position else None
        if current_label is not None and np.any(mask_crop == current_label):
            axes[2, column].contour(mask_crop == current_label, levels=[0.5], colors="cyan", linewidths=0.9)
        elif np.any(mask_crop > 0):
            axes[2, column].contour(mask_crop > 0, levels=[0.5], colors="cyan", linewidths=0.7)
        axes[2, column].axhline(radius_px[0], color="yellow", linewidth=0.5)
        axes[2, column].axvline(radius_px[1], color="yellow", linewidth=0.5)
        current = candidate_subset.loc[candidate_subset["target_session_index"].astype(int).eq(int(session.session_index))] if not candidate_subset.empty else pd.DataFrame()
        for _, candidate in current.iterrows():
            target_feature = _feature(features, session_id, int(candidate["target_label"]))
            if target_feature is None:
                continue
            target_coordinate = target_feature[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
            local = target_coordinate - coordinate + np.array([0.0, radius_px[0], radius_px[1]])
            target_mask = mask_crop == int(candidate["target_label"])
            if np.any(target_mask):
                axes[2, column].contour(target_mask, levels=[0.5], colors="magenta", linewidths=0.7)
            axes[2, column].plot(local[2], local[1], "x", color="magenta", markersize=6)
            axes[2, column].text(local[2] + 2, local[1], f"#{int(candidate['target_rank_by_distance'])}", color="magenta", fontsize=6)
        acquisition = _text(session.get("acquisition_date"))[:10]
        axes[0, column].set_title(f"{relative:+d} {session_id}\n{acquisition}", fontsize=8)
    axes[0, 0].set_ylabel("Red raw")
    axes[1, 0].set_ylabel("Green raw")
    axes[2, 0].set_ylabel("Red + masks")
    fig.suptitle(f"{_text(endpoint.get('endpoint_id'))} | track {_text(endpoint.get('track_uid'))}", fontsize=11)
    return _save(fig, output_path)


def plot_endpoint_candidate_overlay(
    endpoint: Mapping[str, Any] | pd.Series,
    candidates: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot projected endpoint candidates in physical XY coordinates."""

    endpoint_id = _text(endpoint.get("endpoint_id"))
    table = candidates.loc[candidates["endpoint_id"].astype(str).eq(endpoint_id)] if not candidates.empty else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    if not table.empty:
        axis.scatter(table["target_x_um"], table["target_y_um"], c=table["session_gap"], cmap="viridis", s=35)
        for _, row in table.iterrows():
            axis.text(row["target_x_um"], row["target_y_um"], str(int(row["target_rank_by_distance"])), fontsize=8)
        axis.scatter(table["predicted_x_um"].iloc[0], table["predicted_y_um"].iloc[0], marker="+", c="red", s=100)
    axis.set_xlabel("X (µm)"); axis.set_ylabel("Y (µm)"); axis.set_title(f"Endpoint candidates: {endpoint_id}")
    return _save(fig, output_path)


def plot_endpoint_contact_sheet(
    endpoints: pd.DataFrame,
    sessions: pd.DataFrame,
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    transforms: pd.DataFrame,
    *,
    output_dir: str | Path,
    max_panels: int | None = 100,
    crop_radius_um: float = 45.0,
    z_radius: int = 1,
    spacing: VoxelSpacing | None = None,
) -> list[Path]:
    """Write one PNG panel per selected endpoint and a PNG contact sheet."""

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    selected = endpoints.head(int(max_panels)) if max_panels is not None else endpoints
    paths: list[Path] = []
    thumbnails: list[np.ndarray] = []
    for _, endpoint in selected.iterrows():
        path = output_dir / f"{_text(endpoint.get('endpoint_id'))}.png"
        plot_endpoint_review_panel(endpoint, sessions, features, candidates, transforms, output_path=path, crop_radius_um=crop_radius_um, z_radius=z_radius, spacing=spacing)
        paths.append(path)
        thumbnails.append(plt.imread(path))
    sheet_path = output_dir / "endpoint_contact_sheet.png"
    if thumbnails:
        ncols = min(4, len(thumbnails)); nrows = int(np.ceil(len(thumbnails) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for index, image in enumerate(thumbnails):
            axes[index // ncols, index % ncols].imshow(image); axes[index // ncols, index % ncols].axis("off")
        for index in range(len(thumbnails), nrows * ncols):
            axes[index // ncols, index % ncols].axis("off")
        _save(fig, sheet_path)
    else:
        fig, axis = plt.subplots(figsize=(4, 3)); axis.text(0.5, 0.5, "No endpoints selected", ha="center", va="center"); axis.axis("off"); _save(fig, sheet_path)
    paths.append(sheet_path)
    return paths


def plot_track_presence_timeline(
    track: Mapping[str, Any] | pd.Series,
    sessions: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot observed/missing track presence by session."""

    sessions = _manifest_rows(sessions)
    present = [pd.notna(track.get(f"{session.session_id}_roi", np.nan)) for _, session in sessions.iterrows()]
    fig, axis = plt.subplots(figsize=(max(4, len(sessions) * 0.7), 2.5), constrained_layout=True)
    axis.scatter(sessions["session_index"], np.asarray(present, dtype=int), c=np.asarray(present, dtype=int), cmap="RdYlGn", vmin=0, vmax=1, s=60)
    axis.set_yticks([0, 1], ["missing", "present"]); axis.set_xlabel("Session index"); axis.set_title(f"Track presence: {_text(track.get('track_uid'))}")
    return _save(fig, output_path)


def plot_runtime_summaries(runtime: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Write the three required runtime PNG summaries."""

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / "matching_runtime_by_pair.png", output_dir / "matching_runtime_vs_candidate_count.png", output_dir / "matching_runtime_by_gap.png"]
    if runtime.empty:
        for path in outputs:
            fig, axis = plt.subplots(figsize=(5, 3)); axis.text(0.5, 0.5, "No runtime data", ha="center", va="center"); axis.axis("off"); _save(fig, path)
        return outputs
    fig, axis = plt.subplots(figsize=(8, 3), constrained_layout=True)
    axis.bar(np.arange(len(runtime)), runtime["elapsed_sec"].fillna(0).to_numpy(dtype=float)); axis.set_xticks(np.arange(len(runtime)), runtime["session_pair"].astype(str), rotation=60, ha="right", fontsize=7); axis.set_ylabel("Seconds"); axis.set_title("Matching runtime by pair"); _save(fig, outputs[0])
    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True); axis.scatter(runtime["candidate_count"], runtime["elapsed_sec"]); axis.set_xlabel("Candidate count"); axis.set_ylabel("Seconds"); axis.set_title("Runtime vs candidate count"); _save(fig, outputs[1])
    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True); runtime.boxplot(column="elapsed_sec", by="pair_gap", ax=axis); fig.suptitle(""); axis.set_title("Runtime by session gap"); axis.set_xlabel("Pair gap"); axis.set_ylabel("Seconds"); _save(fig, outputs[2])
    return outputs
