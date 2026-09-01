"""Shared session timing resolution for longitudinal plots and viewers."""
from __future__ import annotations
from datetime import datetime
import pandas as pd

def resolve_plot_day_axis(table: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    """Return one row per recording session with real elapsed days and labels."""
    if table.empty:
        return pd.DataFrame(columns=["source_day", "plot_day", "date_label"])
    source = "day" if "day" in table.columns else "session_index" if "session_index" in table.columns else None
    if source is None:
        raise ValueError("table must contain day or session_index")
    work = table.copy()
    work["source_day"] = work[source]
    for col in ("elapsed_days", "acquisition_date"):
        if col in work.columns:
            vals = work.groupby("source_day", dropna=False)[col].nunique(dropna=True)
            if (vals > 1).any():
                raise ValueError(f"Conflicting {col} metadata for a source day")
    if "elapsed_days" in work.columns:
        work["plot_day"] = pd.to_numeric(work["elapsed_days"], errors="raise")
    elif "acquisition_date" in work.columns:
        dates = pd.to_datetime(work["acquisition_date"], errors="raise")
        work["plot_day"] = (dates - dates.min()).dt.days
    else:
        work["plot_day"] = pd.to_numeric(work["source_day"], errors="raise")
    if "acquisition_date" in work.columns:
        dates = pd.to_datetime(work["acquisition_date"], errors="raise")
    else:
        if start_date is None:
            raise ValueError("start_date is required when acquisition_date is unavailable")
        dates = pd.to_datetime(start_date, format="%Y%m%d") + pd.to_timedelta(work["source_day"], unit="D")
    work["date_label"] = dates.dt.strftime("%Y-%m-%d")
    return (work[["source_day", "plot_day", "date_label"]]
            .drop_duplicates()
            .sort_values(["plot_day", "source_day"])
            .reset_index(drop=True))

def resolve_longitudinal_session_metadata(session_manifest: pd.DataFrame) -> pd.DataFrame:
    """Normalize manifest timing and raw paths for longitudinal consumers."""
    required = {"session_index", "session_id", "acquisition_date", "mask_path", "red_image_path", "green_image_path"}
    missing = required - set(session_manifest.columns)
    if missing: raise ValueError(f"Missing session metadata columns: {', '.join(sorted(missing))}")
    out = session_manifest.copy()
    out["acquisition_date"] = pd.to_datetime(out["acquisition_date"], errors="raise")
    out = out.sort_values("session_index").reset_index(drop=True)
    if "elapsed_days" not in out.columns:
        out["elapsed_days"] = (out["acquisition_date"] - out["acquisition_date"].min()).dt.days
    else:
        out["elapsed_days"] = pd.to_numeric(out["elapsed_days"], errors="raise")
    return out[["session_index", "session_id", "acquisition_date", "elapsed_days", "mask_path", "red_image_path", "green_image_path"]]
