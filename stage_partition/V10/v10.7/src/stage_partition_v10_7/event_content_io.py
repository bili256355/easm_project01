from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .h_profile_builder import HProfile, build_h_profile, build_h_state_matrix
from .io_utils import load_smoothed_fields
from .utils import safe_read_csv


@dataclass
class HContentState:
    profile: HProfile
    raw_matrix: np.ndarray  # day x feature, seasonal/year mean raw H profile
    state_matrix: np.ndarray  # valid_day x feature, standardized state matrix
    valid_day_index: np.ndarray
    feature_table: pd.DataFrame
    state_source: str
    profile_status: str


@dataclass
class SpatialFieldData:
    field: np.ndarray  # year x day x lat x lon OR day x lat x lon converted to year=1
    lat: np.ndarray
    lon: np.ndarray
    years: np.ndarray | None
    field_key: str
    lat_key: str
    lon_key: str
    year_key: str | None
    has_year_dim: bool
    status: str


def load_h_content_state(project_root: Path) -> tuple[HContentState, dict[str, Any]]:
    """Reconstruct H profile/state using the same V10.7_a builder.

    The V10.7_a h_state_feature_table is metadata rather than day x feature values in the current package,
    so V10.7_c always reconstructs the raw/state matrices from smoothed_fields.npz. This is recorded in run_meta.
    """
    settings = Settings().with_project_root(project_root)
    smoothed_path = settings.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    profile = build_h_profile(smoothed, settings.h_profile)
    state_pack = build_h_state_matrix(profile, settings.state)
    meta = {
        "state_reconstruction_source": "V10.7_a H profile builder + foundation smoothed_fields.npz",
        "smoothed_fields_path": str(smoothed_path),
        "h_profile_field_key": settings.h_profile.field_key,
        "h_profile_lat_range": list(settings.h_profile.lat_range),
        "h_profile_lon_range": list(settings.h_profile.lon_range),
        "h_profile_shape_year_day_feature": list(profile.raw_cube.shape),
        "state_matrix_shape": list(np.asarray(state_pack["state_matrix"]).shape),
        "valid_day_count": int(len(state_pack["valid_day_index"])),
    }
    return HContentState(
        profile=profile,
        raw_matrix=np.asarray(state_pack["raw_matrix"], dtype=float),
        state_matrix=np.asarray(state_pack["state_matrix"], dtype=float),
        valid_day_index=np.asarray(state_pack["valid_day_index"], dtype=int),
        feature_table=state_pack["feature_table"].copy(),
        state_source=str(smoothed_path),
        profile_status="reconstructed_from_smoothed_fields",
    ), meta


def _first_existing_key(d: dict[str, np.ndarray], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in d:
            return k
    lower = {k.lower(): k for k in d}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def _normalize_field_shape(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, bool, str]:
    arr = np.asarray(field, dtype=float)
    nlat = len(lat)
    nlon = len(lon)
    if arr.ndim == 4:
        # Prefer year x day x lat x lon, but infer lat/lon axes if possible.
        shape = arr.shape
        lat_axes = [i for i, n in enumerate(shape) if n == nlat]
        lon_axes = [i for i, n in enumerate(shape) if n == nlon]
        if 2 in lat_axes and 3 in lon_axes:
            return arr, True, "assumed_year_day_lat_lon"
        if lat_axes and lon_axes:
            lat_axis = lat_axes[0]
            lon_axis = lon_axes[0]
            remaining = [i for i in range(4) if i not in (lat_axis, lon_axis)]
            # Assume remaining axes are year/day, choose order as they appear.
            trans = remaining + [lat_axis, lon_axis]
            return np.transpose(arr, trans), True, f"transposed_axes_{trans}_to_year_day_lat_lon"
        raise ValueError(f"Cannot infer lat/lon axes for 4D field shape={shape}, nlat={nlat}, nlon={nlon}")
    if arr.ndim == 3:
        shape = arr.shape
        lat_axes = [i for i, n in enumerate(shape) if n == nlat]
        lon_axes = [i for i, n in enumerate(shape) if n == nlon]
        if 1 in lat_axes and 2 in lon_axes:
            return arr[None, ...], False, "assumed_day_lat_lon_added_dummy_year"
        if lat_axes and lon_axes:
            lat_axis = lat_axes[0]
            lon_axis = lon_axes[0]
            day_axis = [i for i in range(3) if i not in (lat_axis, lon_axis)][0]
            trans = [day_axis, lat_axis, lon_axis]
            return np.transpose(arr, trans)[None, ...], False, f"transposed_axes_{trans}_to_day_lat_lon_added_dummy_year"
        raise ValueError(f"Cannot infer lat/lon axes for 3D field shape={shape}, nlat={nlat}, nlon={nlon}")
    raise ValueError(f"Expected 3D or 4D H spatial field; got shape {arr.shape}")


def load_spatial_field(project_root: Path, possible_field_keys: tuple[str, ...], possible_lat_keys: tuple[str, ...], possible_lon_keys: tuple[str, ...], possible_year_keys: tuple[str, ...]) -> tuple[SpatialFieldData | None, dict[str, Any]]:
    settings = Settings().with_project_root(project_root)
    smoothed_path = settings.foundation.smoothed_fields_path()
    meta: dict[str, Any] = {"smoothed_fields_path": str(smoothed_path)}
    if not smoothed_path.exists():
        meta["spatial_status"] = "skipped_missing_smoothed_fields"
        return None, meta
    smoothed = load_smoothed_fields(smoothed_path)
    field_key = _first_existing_key(smoothed, possible_field_keys)
    lat_key = _first_existing_key(smoothed, possible_lat_keys)
    lon_key = _first_existing_key(smoothed, possible_lon_keys)
    year_key = _first_existing_key(smoothed, possible_year_keys)
    meta.update({
        "available_keys": sorted(smoothed.keys()),
        "field_key_detected": field_key,
        "lat_key_detected": lat_key,
        "lon_key_detected": lon_key,
        "year_key_detected": year_key,
    })
    if field_key is None or lat_key is None or lon_key is None:
        meta["spatial_status"] = "skipped_missing_field_or_latlon_key"
        return None, meta
    lat = np.asarray(smoothed[lat_key], dtype=float)
    lon = np.asarray(smoothed[lon_key], dtype=float)
    try:
        field4, has_year_dim, shape_status = _normalize_field_shape(np.asarray(smoothed[field_key]), lat, lon)
    except Exception as exc:
        meta["spatial_status"] = f"skipped_shape_error: {exc}"
        return None, meta
    years = None
    if year_key is not None:
        try:
            years = np.asarray(smoothed[year_key])
            if len(years) != field4.shape[0]:
                years = None
        except Exception:
            years = None
    if years is None and has_year_dim:
        years = np.arange(field4.shape[0], dtype=int)
    meta.update({
        "spatial_status": "loaded",
        "field_shape_year_day_lat_lon": list(field4.shape),
        "field_shape_status": shape_status,
        "has_year_dim": bool(has_year_dim),
        "n_years": int(field4.shape[0]),
        "n_days": int(field4.shape[1]),
    })
    return SpatialFieldData(
        field=field4,
        lat=lat,
        lon=lon,
        years=years,
        field_key=field_key,
        lat_key=lat_key,
        lon_key=lon_key,
        year_key=year_key,
        has_year_dim=bool(has_year_dim),
        status="loaded",
    ), meta


def load_optional_v10_7_b_outputs(project_root: Path) -> dict[str, pd.DataFrame | None]:
    base = project_root / "stage_partition" / "V10" / "v10.7" / "outputs" / "h_w045_scale_diagnostic_v10_7_b" / "tables"
    return {
        "ridge_summary": safe_read_csv(base / "h_w045_ridge_family_summary_v10_7_b.csv"),
        "target_response": safe_read_csv(base / "h_w045_target_day_scale_response_v10_7_b.csv"),
        "energy_map": safe_read_csv(base / "h_w045_scale_energy_map_v10_7_b.csv"),
    }
