# -*- coding: utf-8 -*-
"""I/O helpers for P/V850 offset-correspondence audit v1_b."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .t3_p_v_offset_correspondence_settings import PVOffsetCorrespondenceSettings


def ensure_output_dirs(settings: PVOffsetCorrespondenceSettings) -> Dict[str, Path]:
    dirs = {
        "root": settings.output_dir,
        "tables": settings.tables_dir,
        "figures": settings.figures_dir,
        "summary": settings.summary_dir,
        "logs": settings.logs_dir,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def load_smoothed_fields(settings: PVOffsetCorrespondenceSettings) -> Tuple[Dict[str, np.ndarray], Path]:
    pre = settings.project_root / Path(settings.foundation_preprocess_rel)
    candidates = [pre / "smoothed_fields.npz", pre / "fields_smoothed.npz", pre / "smooth5_fields.npz"]
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError("Cannot find smoothed_fields npz. Tried:\n" + "\n".join(map(str, candidates)))
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}, path


def _expanded_aliases(aliases: Iterable[str]) -> Tuple[str, ...]:
    base = []
    alias_map = {
        "precip": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "precipitation": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "p": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "pr": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "rain": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "v850": ["v850_smoothed", "v_smoothed", "meridional_wind_smoothed"],
        "v": ["v850_smoothed", "v_smoothed"],
        "lat": ["lats", "latitude"],
        "lon": ["lons", "longitude"],
    }
    for a in aliases:
        a = str(a)
        base.extend([a, f"{a}_smoothed", f"smoothed_{a}"])
        base.extend(alias_map.get(a.lower(), []))
    seen = set()
    out = []
    for a in base:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return tuple(out)


def get_field_key(data: Dict[str, np.ndarray], aliases: Iterable[str]) -> str:
    lower = {k.lower(): k for k in data.keys()}
    for a in _expanded_aliases(aliases):
        if a in data:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    raise KeyError(f"Cannot find field key among aliases {list(aliases)}. Available: {list(data.keys())}")


def get_lat_lon_years(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    lat_key = get_field_key(data, ["lat", "lats", "latitude"])
    lon_key = get_field_key(data, ["lon", "lons", "longitude"])
    years = None
    for ykey in ["years", "year", "yrs"]:
        if ykey in data:
            years = np.asarray(data[ykey]).astype(int)
            break
    return np.asarray(data[lat_key], dtype=float), np.asarray(data[lon_key], dtype=float), years


def resolve_day_mapping(field: np.ndarray, data: Dict[str, np.ndarray]) -> Tuple[str, Dict[int, int]]:
    n_day = int(field.shape[1])
    for key in ["days", "day", "doys", "dayofseason"]:
        if key in data:
            vals = np.asarray(data[key]).astype(int).tolist()
            return f"coordinate:{key}", {int(v): i for i, v in enumerate(vals)}
    if n_day == 183:
        return "shape183_day_minus_1", {d: d - 1 for d in range(1, 184)}
    if n_day >= 184:
        return "shape_ge184_direct_day_index_warning", {d: d for d in range(1, min(n_day, 184))}
    raise ValueError(f"Cannot resolve day mapping for field with n_day={n_day}.")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
