"""Strict, path-aware parser for ThorImageLS Experiment.xml metadata."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

class ThorImageParseError(ValueError):
    pass

@dataclass(frozen=True)
class PockelsValue:
    start: float
    stop: float

@dataclass(frozen=True)
class ThorImageMetadata:
    xml_path: Path
    experiment_date: str
    unix_time: int | None
    software_version: str
    experiment_status: str
    objective: str
    pixel_x: int
    pixel_y: int
    width_um: float
    height_um: float
    pixel_width_um: float
    pixel_height_um: float
    average_mode: int | None
    average_num: int
    z_steps: int
    z_step_um: float
    flyback_frames: int
    streaming_frames: int
    z_fast_enable: int
    raw_data: int
    timepoints: int
    pockels: tuple[PockelsValue, ...]
    pmt_a_enable: int
    pmt_a_gain: float
    pmt_b_enable: int
    pmt_b_gain: float
    original_acquisition_path: str | None

def _node(root: ET.Element, name: str, path: Path) -> ET.Element:
    node = root.find(f".//{name}")
    if node is None:
        raise ThorImageParseError(f"{path}: missing required <{name}> element")
    return node

def _attr(node: ET.Element, names: tuple[str, ...], cast, path: Path, *, required=True, default=None):
    for name in names:
        if name in node.attrib and str(node.attrib[name]).strip() != "":
            try:
                return cast(node.attrib[name])
            except (TypeError, ValueError) as exc:
                raise ThorImageParseError(f"{path}: invalid {node.tag}.{name}={node.attrib[name]!r}") from exc
    if required:
        raise ThorImageParseError(f"{path}: missing {node.tag}.{'/'.join(names)} attribute")
    return default

def _date_text(node: ET.Element, path: Path) -> tuple[str, int | None]:
    raw = _attr(node, ("date", "value"), str, path)
    parsed = None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(raw, fmt)
            break
        except ValueError:
            pass
    if parsed is None:
        raise ThorImageParseError(f"{path}: invalid Date value {raw!r}")
    unix_raw = node.attrib.get("uTime") or node.attrib.get("unixTime")
    try:
        unix_time = int(float(unix_raw)) if unix_raw not in (None, "") else None
    except ValueError as exc:
        raise ThorImageParseError(f"{path}: invalid Date Unix time {unix_raw!r}") from exc
    return parsed.date().isoformat(), unix_time

def parse_experiment_xml(path: str | Path) -> ThorImageMetadata:
    xml_path = Path(path).resolve()
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ThorImageParseError(f"{xml_path}: unable to parse XML: {exc}") from exc
    date_node, lsm, zstage = _node(root,"Date",xml_path), _node(root,"LSM",xml_path), _node(root,"ZStage",xml_path)
    streaming, timelapse = _node(root,"Streaming",xml_path), _node(root,"Timelapse",xml_path)
    experiment_date, unix_time = _date_text(date_node, xml_path)
    pixel_x, pixel_y = _attr(lsm,("pixelX",),int,xml_path), _attr(lsm,("pixelY",),int,xml_path)
    width_um, height_um = _attr(lsm,("widthUM",),float,xml_path), _attr(lsm,("heightUM",),float,xml_path)
    px_w = _attr(lsm,("pixelWidthUM",),float,xml_path)
    px_h = _attr(lsm,("pixelHeightUM",),float,xml_path)
    if abs(px_w - width_um / pixel_x) > max(1e-4, px_w * .01) or abs(px_h - height_um / pixel_y) > max(1e-4, px_h * .01):
        raise ThorImageParseError(f"{xml_path}: pixelWidthUM/pixelHeightUM disagree with physical dimensions")
    pockels = tuple(PockelsValue(_attr(n,("start",),float,xml_path), _attr(n,("stop",),float,xml_path)) for n in root.findall(".//Pockels"))
    pmt = root.find(".//PMT")
    if pmt is None:
        raise ThorImageParseError(f"{xml_path}: missing required <PMT> element")
    name = root.find(".//Name")
    software = root.find(".//Software")
    objective = root.find(".//Objective") if root.find(".//Objective") is not None else root.find(".//Magnification")
    status_node = root.find(".//ExperimentStatus")
    return ThorImageMetadata(
        xml_path, experiment_date, unix_time,
        _attr(software,("version",),str,xml_path,required=False,default="") if software is not None else "",
        _attr(status_node,("value",),str,xml_path,required=False,default=_attr(root,("status",),str,xml_path,required=False,default="")) if status_node is not None else _attr(root,("status",),str,xml_path,required=False,default=""),
        _attr(objective,("name","objectiveName"),str,xml_path,required=False,default="") if objective is not None else "",
        pixel_x,pixel_y,width_um,height_um,px_w,px_h,
        _attr(lsm,("averageMode",),int,xml_path,required=False,default=None), _attr(lsm,("averageNum",),int,xml_path),
        _attr(zstage,("steps",),int,xml_path), _attr(zstage,("stepSizeUM",),float,xml_path),
        _attr(streaming,("flybackFrames",),int,xml_path), _attr(streaming,("frames",),int,xml_path),
        _attr(streaming,("zFastEnable",),int,xml_path), _attr(streaming,("rawData",),int,xml_path),
        _attr(timelapse,("timepoints",),int,xml_path), pockels,
        _attr(pmt,("enableA", "channelAEnable"),int,xml_path), _attr(pmt,("gainA", "channelAGain"),float,xml_path),
        _attr(pmt,("enableB", "channelBEnable"),int,xml_path), _attr(pmt,("gainB", "channelBGain"),float,xml_path),
        _attr(name,("path",),str,xml_path,required=False,default=None) if name is not None else None,
    )
