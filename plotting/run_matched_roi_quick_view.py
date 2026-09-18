"""Raw-space QC viewer for one matched longitudinal ROI."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from longitudinal_session import resolve_longitudinal_session_metadata

def extract_centered_crop_with_padding(image_2d, *, center_y, center_x, height, width, fill_value=0):
    if height <= 0 or width <= 0: raise ValueError("crop dimensions must be positive")
    out=np.full((height,width), fill_value, dtype=image_2d.dtype)
    cy, cx = height//2, width//2
    y0=int(round(center_y))-cy; x0=int(round(center_x))-cx
    sy0=max(0,y0); sx0=max(0,x0); sy1=min(image_2d.shape[0],y0+height); sx1=min(image_2d.shape[1],x0+width)
    if sy1>sy0 and sx1>sx0: out[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=image_2d[sy0:sy1,sx0:sx1]
    return out

def _track(tracks, cluster_id, track_uid):
    if cluster_id is not None:
        col='cluster_id' if 'cluster_id' in tracks else 'roi_id'
        rows=tracks[tracks[col].astype(str)==str(cluster_id)]
    else: rows=tracks[tracks['track_uid'].astype(str)==str(track_uid)]
    if rows.empty: raise ValueError('Selected longitudinal track was not found')
    if len(rows)>1: raise ValueError('Selected longitudinal track is not unique')
    return rows.iloc[0]

def select_render_z_indices(roi_z, n_z, render_z_radius=0):
    if render_z_radius < 0: raise ValueError('render_z_radius must be >= 0')
    if n_z <= 0: raise ValueError('n_z must be positive')
    roi_z = int(roi_z)
    if not 0 <= roi_z < n_z: raise ValueError('roi_z must be within the stack')
    return tuple(range(max(0, roi_z - render_z_radius), min(n_z - 1, roi_z + render_z_radius) + 1))

def plot_matched_roi_raw_slices(*, cluster_id=None, track_uid=None, tracks_table, session_table, output_path=None, crop_padding_px=20, min_crop_size_px=48, z_radius=0, render_z_radius=None, lut_low_percentile=2.0, lut_high_percentile=98.0):
    if (cluster_id is None)==(track_uid is None): raise ValueError('Provide exactly one of cluster_id or track_uid')
    if render_z_radius is not None: z_radius = render_z_radius
    if z_radius<0: raise ValueError('render_z_radius must be >= 0')
    if not (0.0 <= lut_low_percentile < lut_high_percentile <= 100.0): raise ValueError('LUT percentiles must satisfy 0 <= low < high <= 100')
    track=_track(tracks_table,cluster_id,track_uid); sessions=resolve_longitudinal_session_metadata(session_table)
    loaded=[]; max_h=max_w=0
    for _,s in sessions.iterrows():
        mask=tifffile.imread(s.mask_path)
        label=track.get(f"{s.session_id}_roi", np.nan)
        if pd.isna(label): loaded.append((s,None,None,None,None)); continue
        coords=np.where(mask==int(label))
        if len(coords[0])==0: loaded.append((s,None,None,None,None)); continue
        z0=int(round(coords[0].mean())); yc=int(round(coords[1].mean())); xc=int(round(coords[2].mean()))
        h=int(coords[1].max()-coords[1].min()+1+2*crop_padding_px); w=int(coords[2].max()-coords[2].min()+1+2*crop_padding_px); max_h=max(max_h,h); max_w=max(max_w,w)
        loaded.append((s,mask,None,None,(int(label),z0,yc,xc)))
    height=max(min_crop_size_px,max_h); width=max(min_crop_size_px,max_w); offsets=tuple(range(z_radius,-z_radius-1,-1)); nrows=2*len(offsets); fig,axes=plt.subplots(nrows,max(1,len(loaded)),figsize=(2.2*max(1,len(loaded)),1.8*nrows),squeeze=False,constrained_layout=True)
    red_cmap=LinearSegmentedColormap.from_list('raw_red',['black','#FF00FF']); green_cmap=LinearSegmentedColormap.from_list('raw_green',['black','#00FF00'])
    reds=[]; greens=[]; prepared=[]
    for s,mask,red,green,info in loaded:
        day=[]
        if info:
            label,z0,yc,xc=info
            valid_z_indices = set(select_render_z_indices(z0, mask.shape[0], z_radius))
            for off in offsets:
                requested_z = z0 + off
                if requested_z not in valid_z_indices:
                    rc=np.zeros((height,width),dtype=float); gc=np.zeros((height,width),dtype=float); mc=np.zeros((height,width),dtype=np.uint8)
                else:
                    z=requested_z; red_plane=tifffile.imread(s.red_image_path,key=z); green_plane=tifffile.imread(s.green_image_path,key=z); rc=extract_centered_crop_with_padding(red_plane,center_y=yc,center_x=xc,height=height,width=width); gc=extract_centered_crop_with_padding(green_plane,center_y=yc,center_x=xc,height=height,width=width); mc=extract_centered_crop_with_padding((mask[z]==label).astype(np.uint8),center_y=yc,center_x=xc,height=height,width=width)
                if requested_z in valid_z_indices:
                    reds.append(rc.ravel()); greens.append(gc.ravel())
                day.append((rc,gc,mc,off,requested_z not in valid_z_indices))
        prepared.append(day)
    rv=float(np.percentile(np.concatenate(reds),lut_high_percentile)) if reds else 0.0; rlo=float(np.percentile(np.concatenate(reds),lut_low_percentile)) if reds else 0.0
    gv=float(np.percentile(np.concatenate(greens),lut_high_percentile)) if greens else 0.0; glo=float(np.percentile(np.concatenate(greens),lut_low_percentile)) if greens else 0.0
    if rv <= rlo: rv = rlo + 1.0
    if gv <= glo: gv = glo + 1.0
    red_norm=Normalize(vmin=rlo,vmax=rv); green_norm=Normalize(vmin=glo,vmax=gv)
    metadata_rows=[]
    for col,(s,_,_,_,info) in enumerate(loaded):
        for i,off in enumerate(offsets):
            if i < len(prepared[col]):
                rc,gc,mc,_,out_of_stack=prepared[col][i];
                requested_z = (info[1] + off) if info else np.nan
                metadata_rows.append({"match_policy": track.get("match_policy", ""), "roi_id": track.get("roi_id", cluster_id if cluster_id is not None else ""), "cluster_id": track.get("cluster_id", ""), "track_uid": track.get("track_uid", ""), "session_index": s.session_index, "session_id": s.session_id, "elapsed_days": s.elapsed_days, "acquisition_date": s.acquisition_date.strftime("%Y-%m-%d"), "session_roi_label": info[0] if info else np.nan, "z_offset": off, "z_center_abs": info[1] if info else np.nan, "requested_z_abs": requested_z, "displayed_z_abs": requested_z if not out_of_stack else np.nan, "z_out_of_bounds": out_of_stack, "y_center": info[2] if info else np.nan, "x_center": info[3] if info else np.nan, "crop_height": height, "crop_width": width, "mask_path": s.mask_path, "red_image_path": s.red_image_path, "green_image_path": s.green_image_path, "matched": True, "lut_low_percentile": lut_low_percentile, "lut_high_percentile": lut_high_percentile, "red_vmin": rlo, "red_vmax": rv, "green_vmin": glo, "green_vmax": gv})
                axes[i,col].imshow(rc,cmap=red_cmap,norm=red_norm); axes[i+len(offsets),col].imshow(gc,cmap=green_cmap,norm=green_norm)
                if out_of_stack:
                    axes[i,col].set_visible(False); axes[i+len(offsets),col].set_visible(False)
                else:
                    for ax in (axes[i,col],axes[i+len(offsets),col]):
                        if mc.any(): ax.contour(mc,levels=[.5],colors='white',linewidths=.6)
            else:
                axes[i,col].text(.5,.5,'not matched',ha='center',va='center'); axes[i+len(offsets),col].text(.5,.5,'not matched',ha='center',va='center')
                metadata_rows.append({"match_policy": track.get("match_policy", ""), "roi_id": track.get("roi_id", cluster_id if cluster_id is not None else ""), "cluster_id": track.get("cluster_id", ""), "track_uid": track.get("track_uid", ""), "session_index": s.session_index, "session_id": s.session_id, "elapsed_days": s.elapsed_days, "acquisition_date": s.acquisition_date.strftime("%Y-%m-%d"), "session_roi_label": np.nan, "z_offset": off, "z_center_abs": np.nan, "requested_z_abs": np.nan, "displayed_z_abs": np.nan, "z_out_of_bounds": np.nan, "y_center": np.nan, "x_center": np.nan, "crop_height": height, "crop_width": width, "mask_path": s.mask_path, "red_image_path": s.red_image_path, "green_image_path": s.green_image_path, "matched": False, "lut_low_percentile": lut_low_percentile, "lut_high_percentile": lut_high_percentile, "red_vmin": rlo, "red_vmax": rv, "green_vmin": glo, "green_vmax": gv})
            if col == 0:
                axes[i,col].set_ylabel(f'Red\nz={off:+d}'); axes[i+len(offsets),col].set_ylabel(f'Green\nz={off:+d}')
        label=f"Day {int(s.elapsed_days)}\n{s.acquisition_date.strftime('%Y-%m-%d')}"; axes[0,col].set_title(label)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    red_axes=axes[:len(offsets),:].ravel().tolist(); green_axes=axes[len(offsets):,:].ravel().tolist()
    fig.colorbar(ScalarMappable(norm=red_norm,cmap=red_cmap), ax=red_axes, fraction=.02, pad=.02, label="Red intensity")
    fig.colorbar(ScalarMappable(norm=green_norm,cmap=green_cmap), ax=green_axes, fraction=.02, pad=.02, label="Green intensity")
    if output_path is not None:
        fig.savefig(output_path,dpi=180,bbox_inches="tight",pad_inches=0.12); plt.close(fig)
        pd.DataFrame(metadata_rows).to_csv(Path(output_path).with_name(Path(output_path).stem + "_metadata.csv"), index=False)
    return fig


def _batch_feature_geometry(features: pd.DataFrame | None, session_id: str, label: int, padding: int):
    if features is None or features.empty or not {"session_id", "label"}.issubset(features.columns):
        return None
    rows = features.loc[
        features["session_id"].astype(str).eq(str(session_id))
        & pd.to_numeric(features["label"], errors="coerce").eq(int(label))
    ]
    required = {"centroid_z", "centroid_y", "centroid_x", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1"}
    if rows.empty or not required.issubset(rows.columns):
        return None
    row = rows.iloc[0]
    return (
        int(round(float(row.centroid_z))),
        int(round(float(row.centroid_y))),
        int(round(float(row.centroid_x))),
        int(row.bbox_y1 - row.bbox_y0 + 2 * padding),
        int(row.bbox_x1 - row.bbox_x0 + 2 * padding),
    )


def _prepare_batch_track(
    track: pd.Series,
    sessions: pd.DataFrame,
    volumes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    features: pd.DataFrame | None,
    *,
    crop_padding_px: int,
    min_crop_size_px: int,
    z_radius: int,
    profile: dict[str, object],
):
    loaded = []
    max_h = max_w = 0
    for _, session in sessions.iterrows():
        session_id = str(session.session_id)
        label = track.get(f"{session_id}_roi", np.nan)
        if pd.isna(label):
            loaded.append((session, None, None, None, None))
            continue
        label = int(label)
        mask, red, green = volumes[session_id]
        geometry = _batch_feature_geometry(features, session_id, label, crop_padding_px)
        if geometry is None:
            coords = np.where(mask == label)
            if len(coords[0]) == 0:
                loaded.append((session, None, None, None, None))
                continue
            geometry = (
                int(round(float(coords[0].mean()))),
                int(round(float(coords[1].mean()))),
                int(round(float(coords[2].mean()))),
                int(coords[1].max() - coords[1].min() + 1 + 2 * crop_padding_px),
                int(coords[2].max() - coords[2].min() + 1 + 2 * crop_padding_px),
            )
            profile["geometry_fallbacks"] = int(profile.get("geometry_fallbacks", 0)) + 1
        z0, yc, xc, height, width = geometry
        max_h = max(max_h, height)
        max_w = max(max_w, width)
        loaded.append((session, mask, red, green, (label, z0, yc, xc)))

    height = max(min_crop_size_px, max_h)
    width = max(min_crop_size_px, max_w)
    offsets = tuple(range(z_radius, -z_radius - 1, -1))
    prepared = []
    reds = []
    greens = []
    for session, mask, red, green, info in loaded:
        day = []
        if info is not None:
            label, z0, yc, xc = info
            valid_z_indices = set(select_render_z_indices(z0, mask.shape[0], z_radius))
            for offset in offsets:
                requested_z = z0 + offset
                if requested_z not in valid_z_indices:
                    rc = np.zeros((height, width), dtype=float)
                    gc = np.zeros((height, width), dtype=float)
                    mc = np.zeros((height, width), dtype=np.uint8)
                else:
                    rc = extract_centered_crop_with_padding(red[requested_z], center_y=yc, center_x=xc, height=height, width=width)
                    gc = extract_centered_crop_with_padding(green[requested_z], center_y=yc, center_x=xc, height=height, width=width)
                    mc = extract_centered_crop_with_padding((mask[requested_z] == label).astype(np.uint8), center_y=yc, center_x=xc, height=height, width=width)
                    reds.append(rc.ravel())
                    greens.append(gc.ravel())
                day.append((rc, gc, mc, offset, requested_z not in valid_z_indices))
        prepared.append(day)
    return loaded, prepared, reds, greens, height, width, offsets


def _render_prepared_track(track, loaded, prepared, reds, greens, height, width, offsets, output_path, *, lut_low_percentile, lut_high_percentile):
    rv = float(np.percentile(np.concatenate(reds), lut_high_percentile)) if reds else 0.0
    rlo = float(np.percentile(np.concatenate(reds), lut_low_percentile)) if reds else 0.0
    gv = float(np.percentile(np.concatenate(greens), lut_high_percentile)) if greens else 0.0
    glo = float(np.percentile(np.concatenate(greens), lut_low_percentile)) if greens else 0.0
    if rv <= rlo: rv = rlo + 1.0
    if gv <= glo: gv = glo + 1.0
    red_norm = Normalize(vmin=rlo, vmax=rv)
    green_norm = Normalize(vmin=glo, vmax=gv)
    nrows = 2 * len(offsets)
    fig, axes = plt.subplots(nrows, max(1, len(loaded)), figsize=(2.2 * max(1, len(loaded)), 1.8 * nrows), squeeze=False, constrained_layout=True)
    red_cmap = LinearSegmentedColormap.from_list("raw_red", ["black", "#FF00FF"])
    green_cmap = LinearSegmentedColormap.from_list("raw_green", ["black", "#00FF00"])
    metadata_rows = []
    for col, (session, _, _, _, info) in enumerate(loaded):
        for i, offset in enumerate(offsets):
            if i < len(prepared[col]):
                rc, gc, mc, _, out_of_stack = prepared[col][i]
                requested_z = (info[1] + offset) if info else np.nan
                metadata_rows.append({"match_policy": track.get("match_policy", ""), "roi_id": track.get("roi_id", ""), "cluster_id": track.get("cluster_id", ""), "track_uid": track.get("track_uid", ""), "session_index": session.session_index, "session_id": session.session_id, "elapsed_days": session.elapsed_days, "acquisition_date": session.acquisition_date.strftime("%Y-%m-%d"), "session_roi_label": info[0] if info else np.nan, "z_offset": offset, "z_center_abs": info[1] if info else np.nan, "requested_z_abs": requested_z, "displayed_z_abs": requested_z if not out_of_stack else np.nan, "z_out_of_bounds": out_of_stack, "y_center": info[2] if info else np.nan, "x_center": info[3] if info else np.nan, "crop_height": rc.shape[0], "crop_width": rc.shape[1], "mask_path": session.mask_path, "red_image_path": session.red_image_path, "green_image_path": session.green_image_path, "matched": True, "lut_low_percentile": lut_low_percentile, "lut_high_percentile": lut_high_percentile, "red_vmin": rlo, "red_vmax": rv, "green_vmin": glo, "green_vmax": gv})
                axes[i, col].imshow(rc, cmap=red_cmap, norm=red_norm)
                axes[i + len(offsets), col].imshow(gc, cmap=green_cmap, norm=green_norm)
                if out_of_stack:
                    axes[i, col].set_visible(False)
                    axes[i + len(offsets), col].set_visible(False)
                else:
                    for ax in (axes[i, col], axes[i + len(offsets), col]):
                        if mc.any():
                            ax.contour(mc, levels=[.5], colors="white", linewidths=.6)
            else:
                axes[i, col].text(.5, .5, "not matched", ha="center", va="center")
                axes[i + len(offsets), col].text(.5, .5, "not matched", ha="center", va="center")
                metadata_rows.append({"match_policy": track.get("match_policy", ""), "roi_id": track.get("roi_id", ""), "cluster_id": track.get("cluster_id", ""), "track_uid": track.get("track_uid", ""), "session_index": session.session_index, "session_id": session.session_id, "elapsed_days": session.elapsed_days, "acquisition_date": session.acquisition_date.strftime("%Y-%m-%d"), "session_roi_label": np.nan, "z_offset": offset, "z_center_abs": np.nan, "requested_z_abs": np.nan, "displayed_z_abs": np.nan, "z_out_of_bounds": np.nan, "y_center": np.nan, "x_center": np.nan, "crop_height": height, "crop_width": width, "mask_path": session.mask_path, "red_image_path": session.red_image_path, "green_image_path": session.green_image_path, "matched": False, "lut_low_percentile": lut_low_percentile, "lut_high_percentile": lut_high_percentile, "red_vmin": rlo, "red_vmax": rv, "green_vmin": glo, "green_vmax": gv})
            if col == 0:
                axes[i, col].set_ylabel(f"Red\nz={offset:+d}")
                axes[i + len(offsets), col].set_ylabel(f"Green\nz={offset:+d}")
        axes[0, col].set_title(f"Day {int(session.elapsed_days)}\n{session.acquisition_date.strftime('%Y-%m-%d')}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    red_axes = axes[:len(offsets), :].ravel().tolist()
    green_axes = axes[len(offsets):, :].ravel().tolist()
    fig.colorbar(ScalarMappable(norm=red_norm, cmap=red_cmap), ax=red_axes, fraction=.02, pad=.02, label="Red intensity")
    fig.colorbar(ScalarMappable(norm=green_norm, cmap=green_cmap), ax=green_axes, fraction=.02, pad=.02, label="Green intensity")
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        pd.DataFrame(metadata_rows).to_csv(Path(output_path).with_name(Path(output_path).stem + "_metadata.csv"), index=False)


def _feature_batch_ready(specs, tracks_by_uid, sessions, features, padding):
    if features is None:
        return False
    for track_uid, _ in specs:
        track = tracks_by_uid[str(track_uid)]
        for _, session in sessions.iterrows():
            label = track.get(f"{session.session_id}_roi", np.nan)
            if not pd.isna(label) and _batch_feature_geometry(features, str(session.session_id), int(label), padding) is None:
                return False
    return True


def _render_ranked_feature_batch(*, specs, tracks_by_uid, sessions, features, z_radius, crop_padding_px, min_crop_size_px, lut_low_percentile, lut_high_percentile):
    started = time.perf_counter()
    unique_uids = list(dict.fromkeys(str(track_uid) for track_uid, _ in specs))
    states = {
        uid: {"track": tracks_by_uid[uid], "loaded": [], "prepared": [], "reds": [], "greens": [], "height": 0, "width": 0}
        for uid in unique_uids
    }
    for uid in unique_uids:
        state = states[uid]
        for _, session in sessions.iterrows():
            label = state["track"].get(f"{session.session_id}_roi", np.nan)
            if pd.isna(label):
                state["loaded"].append((session, None, None, None, None))
                state["prepared"].append([])
                continue
            geometry = _batch_feature_geometry(features, str(session.session_id), int(label), crop_padding_px)
            z0, yc, xc, height, width = geometry
            state["height"] = max(state["height"], height)
            state["width"] = max(state["width"], width)
            state["loaded"].append((session, None, None, None, (int(label), z0, yc, xc)))
            state["prepared"].append(None)
    for state in states.values():
        state["height"] = max(min_crop_size_px, state["height"])
        state["width"] = max(min_crop_size_px, state["width"])

    offsets = tuple(range(z_radius, -z_radius - 1, -1))
    load_started = time.perf_counter()
    for session_index, (_, session) in enumerate(sessions.iterrows()):
        session_id = str(session.session_id)
        mask = tifffile.imread(session.mask_path)
        red = tifffile.imread(session.red_image_path)
        green = tifffile.imread(session.green_image_path)
        for state in states.values():
            info = state["loaded"][session_index][4]
            day = []
            if info is not None:
                label, z0, yc, xc = info
                valid_z_indices = set(select_render_z_indices(z0, mask.shape[0], z_radius))
                for offset in offsets:
                    requested_z = z0 + offset
                    if requested_z not in valid_z_indices:
                        rc = np.zeros((state["height"], state["width"]), dtype=float)
                        gc = np.zeros((state["height"], state["width"]), dtype=float)
                        mc = np.zeros((state["height"], state["width"]), dtype=np.uint8)
                    else:
                        rc = extract_centered_crop_with_padding(red[requested_z], center_y=yc, center_x=xc, height=state["height"], width=state["width"])
                        gc = extract_centered_crop_with_padding(green[requested_z], center_y=yc, center_x=xc, height=state["height"], width=state["width"])
                        mc = extract_centered_crop_with_padding((mask[requested_z] == label).astype(np.uint8), center_y=yc, center_x=xc, height=state["height"], width=state["width"])
                        state["reds"].append(rc.ravel())
                        state["greens"].append(gc.ravel())
                    day.append((rc, gc, mc, offset, requested_z not in valid_z_indices))
            state["prepared"][session_index] = day
        del mask, red, green
    prepare_seconds = time.perf_counter() - load_started
    render_started = time.perf_counter()
    for track_uid, output_path in specs:
        state = states[str(track_uid)]
        _render_prepared_track(state["track"], state["loaded"], state["prepared"], state["reds"], state["greens"], state["height"], state["width"], offsets, output_path, lut_low_percentile=lut_low_percentile, lut_high_percentile=lut_high_percentile)
    return {
        "n_selected_outputs": len(specs),
        "n_unique_tracks_prepared": len(unique_uids),
        "n_sessions": len(sessions),
        "reads": int(len(sessions) * 3),
        "fallbacks": 0,
        "full_stack_reads": int(len(sessions) * 3),
        "geometry_fallbacks": 0,
        "prepare_seconds": prepare_seconds,
        "render_seconds": time.perf_counter() - render_started,
        "total_seconds": time.perf_counter() - started,
    }


def render_ranked_roi_batch(*, specs, tracks_table, session_table, feature_table=None, z_radius=0, crop_padding_px=20, min_crop_size_px=48, lut_low_percentile=2.0, lut_high_percentile=98.0):
    """Render selected tracks while retaining only one raw volume per session."""
    if z_radius < 0:
        raise ValueError("render_z_radius must be >= 0")
    if not specs:
        return {"n_selected_outputs": 0, "n_unique_tracks_prepared": 0, "n_sessions": 0, "reads": 0, "fallbacks": 0, "full_stack_reads": 0, "geometry_fallbacks": 0, "prepare_seconds": 0.0, "render_seconds": 0.0, "total_seconds": 0.0}
    started = time.perf_counter()
    sessions = resolve_longitudinal_session_metadata(session_table)
    profile: dict[str, object] = {"n_selected_outputs": len(specs), "n_unique_tracks_prepared": len({str(spec[0]) for spec in specs}), "n_sessions": len(sessions), "reads": 0, "fallbacks": 0, "full_stack_reads": 0, "geometry_fallbacks": 0}
    tracks_by_uid = {str(row.track_uid): row for _, row in tracks_table.iterrows()}
    if _feature_batch_ready(specs, tracks_by_uid, sessions, feature_table, crop_padding_px):
        return _render_ranked_feature_batch(
            specs=specs,
            tracks_by_uid=tracks_by_uid,
            sessions=sessions,
            features=feature_table,
            z_radius=z_radius,
            crop_padding_px=crop_padding_px,
            min_crop_size_px=min_crop_size_px,
            lut_low_percentile=lut_low_percentile,
            lut_high_percentile=lut_high_percentile,
        )
    volumes = {}
    load_started = time.perf_counter()
    for _, session in sessions.iterrows():
        session_id = str(session.session_id)
        volumes[session_id] = (
            tifffile.imread(session.mask_path),
            tifffile.imread(session.red_image_path),
            tifffile.imread(session.green_image_path),
        )
    profile["full_stack_reads"] = int(len(sessions) * 3)
    profile["reads"] = int(len(sessions) * 3)
    profile["prepare_seconds"] = time.perf_counter() - load_started
    prepared_by_uid = {}
    for track_uid, _ in specs:
        key = str(track_uid)
        if key not in prepared_by_uid:
            track = tracks_by_uid[key]
            prepared_by_uid[key] = _prepare_batch_track(track, sessions, volumes, feature_table, crop_padding_px=crop_padding_px, min_crop_size_px=min_crop_size_px, z_radius=z_radius, profile=profile)
    render_started = time.perf_counter()
    for track_uid, output_path in specs:
        track = tracks_by_uid[str(track_uid)]
        loaded, prepared, reds, greens, height, width, offsets = prepared_by_uid[str(track_uid)]
        _render_prepared_track(track, loaded, prepared, reds, greens, height, width, offsets, output_path, lut_low_percentile=lut_low_percentile, lut_high_percentile=lut_high_percentile)
    profile["render_seconds"] = time.perf_counter() - render_started
    profile["total_seconds"] = time.perf_counter() - started
    profile["fallbacks"] = int(profile.get("geometry_fallbacks", 0))
    return profile

def _tracks_from_raw_table(raw_table, policy):
    selected = raw_table.loc[raw_table["match_policy"].eq(policy)].copy()
    if selected.empty: raise ValueError(f"No rows found for policy {policy!r}")
    key = "cluster_id" if "cluster_id" in selected.columns else "roi_id"
    check_cols = [key, "roi_id", "track_uid", "session_id", "session_roi_label"]
    counts = selected.groupby([key, "track_uid", "session_id"])["session_roi_label"].nunique(dropna=True)
    if (counts > 1).any(): raise ValueError("Conflicting session_roi_label values across channel rows")
    rows = selected[check_cols].drop_duplicates()
    tracks = rows.groupby([key, "roi_id", "track_uid"], as_index=False).first()
    for _, row in tracks.iterrows():
        values = rows[(rows[key] == row[key]) & (rows["track_uid"] == row["track_uid"])]
        for _, value in values.iterrows(): tracks.loc[tracks.index[tracks[key] == row[key]][0], f"{value.session_id}_roi"] = value.session_roi_label
    tracks["match_policy"] = policy
    return tracks

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--analysis-dir',required=True)
    g=ap.add_mutually_exclusive_group(required=True); g.add_argument('--cluster-id'); g.add_argument('--track-uid')
    ap.add_argument('--policy',choices=['high','balanced','graph'],default='high'); ap.add_argument('--output'); ap.add_argument('--render-z-radius','--z-radius',dest='render_z_radius',type=int,default=0); ap.add_argument('--lut-low-percentile',type=float,default=2.0); ap.add_argument('--lut-high-percentile',type=float,default=98.0)
    a=ap.parse_args(argv); root=Path(a.analysis_dir); manifest=pd.read_csv(root/'session_manifest_resolved.csv'); raw=pd.read_csv(root/'matched_roi_intensity_results_raw.csv'); tracks=_tracks_from_raw_table(raw,a.policy)
    out=Path(a.output or root/f"raw_roi_{a.policy}_{a.cluster_id or a.track_uid}.png"); plot_matched_roi_raw_slices(cluster_id=a.cluster_id,track_uid=a.track_uid,tracks_table=tracks,session_table=manifest,output_path=out,render_z_radius=a.render_z_radius,lut_low_percentile=a.lut_low_percentile,lut_high_percentile=a.lut_high_percentile)
if __name__=='__main__': main()
