# -*- coding: utf-8 -*-
"""I/O helpers for T3 V->P transition-chain report v1_b."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .t3_v_to_p_transition_chain_settings import TransitionChainReportSettings


def ensure_output_dirs(settings: TransitionChainReportSettings) -> Dict[str, Path]:
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


def normalize_year_day_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    colmap = {}
    if "year" not in out.columns:
        for c in ["years", "Year", "YEAR", "yr"]:
            if c in out.columns:
                colmap[c] = "year"
                break
    if "day" not in out.columns:
        for c in ["day_index", "doy", "DOY", "Day", "dayofseason"]:
            if c in out.columns:
                colmap[c] = "day"
                break
    if colmap:
        out = out.rename(columns=colmap)
    if "year" not in out.columns or "day" not in out.columns:
        raise ValueError("Index CSV must contain year/day columns or recognizable aliases.")
    out["year"] = out["year"].astype(int)
    out["day"] = out["day"].astype(int)
    return out


def load_index_values(settings: TransitionChainReportSettings) -> Tuple[pd.DataFrame, Path]:
    idx_dir = settings.project_root / Path(settings.foundation_indices_rel)
    candidates = [
        idx_dir / "index_values_smoothed.csv",
        idx_dir / "indices_smoothed.csv",
        idx_dir / "index_values.csv",
        idx_dir / "smoothed_indices.csv",
        idx_dir / "index_anomalies.csv",
    ]
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError("Cannot find smooth5 index CSV. Tried:\n" + "\n".join(map(str, candidates)))
    return normalize_year_day_columns(pd.read_csv(path)), path


def load_smoothed_fields(settings: TransitionChainReportSettings) -> Tuple[Dict[str, np.ndarray], Path]:
    pre = settings.project_root / Path(settings.foundation_preprocess_rel)
    candidates = [pre / "smoothed_fields.npz", pre / "fields_smoothed.npz", pre / "smooth5_fields.npz"]
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError("Cannot find smoothed_fields npz. Tried:\n" + "\n".join(map(str, candidates)))
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}, path


def get_field_key(data: Dict[str, np.ndarray], aliases: Iterable[str]) -> str:
    lower = {k.lower(): k for k in data.keys()}
    expanded: List[str] = []
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
        expanded.extend([a, f"{a}_smoothed", f"smoothed_{a}"])
        expanded.extend(alias_map.get(str(a).lower(), []))
    seen = set()
    for a in expanded:
        if a in seen:
            continue
        seen.add(a)
        if a in data:
            return a
        if str(a).lower() in lower:
            return lower[str(a).lower()]
    raise KeyError(f"Cannot find field key among aliases {list(aliases)}. Available: {list(data.keys())}")


def get_lat_lon_years(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    lat_key = get_field_key(data, ["lat", "lats", "latitude"])
    lon_key = get_field_key(data, ["lon", "lons", "longitude"])
    years = None
    for ykey in ["years", "year", "yrs"]:
        if ykey in data:
            years = np.asarray(data[ykey]).astype(int)
            break
    return np.asarray(data[lat_key]), np.asarray(data[lon_key]), years


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


def build_year_index(field_years: Optional[np.ndarray], index_years: Iterable[int], n_field_years: int) -> Tuple[str, Dict[int, int]]:
    idx_years_sorted = sorted({int(y) for y in index_years})
    if field_years is not None:
        return "npz_years", {int(y): i for i, y in enumerate(np.asarray(field_years).astype(int).tolist())}
    if len(idx_years_sorted) == n_field_years:
        return "assumed_sorted_index_years", {int(y): i for i, y in enumerate(idx_years_sorted)}
    raise ValueError(
        "Field npz has no years coordinate and index-year count does not match field year dimension. "
        f"n_index_years={len(idx_years_sorted)}, n_field_years={n_field_years}."
    )


def load_previous_region_response(settings: TransitionChainReportSettings) -> pd.DataFrame:
    path = settings.previous_tables_dir / "region_response_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing previous region_response_summary.csv: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
