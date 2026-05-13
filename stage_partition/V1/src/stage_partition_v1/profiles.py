
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings


FIELD_MAP = {
    'P': 'precip',
    'V': 'v850',
    'H': 'z500',
    'Je': 'u200',
    'Jw': 'u200',
}


def _mask_between(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.isfinite(arr) & (arr >= lower) & (arr <= upper)


def _safe_nanmean(arr: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(arr)
    count = np.sum(valid, axis=axis)
    summed = np.nansum(arr, axis=axis)
    out = np.full(summed.shape, np.nan, dtype=np.float64)
    np.divide(summed, count, out=out, where=count > 0)
    return out


def _prepare_interp_inputs(src_lats: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    valid = np.isfinite(values) & np.isfinite(src_lats)
    if valid.sum() < 2:
        return None, None
    src = np.asarray(src_lats[valid], dtype=np.float64)
    vals = np.asarray(values[valid], dtype=np.float64)
    order = np.argsort(src)
    src_sorted = src[order]
    vals_sorted = vals[order]
    if np.unique(src_sorted).size < 2:
        return None, None
    return src_sorted, vals_sorted


def _interp_profile_to_grid(profile: np.ndarray, src_lats: np.ndarray, dst_lats: np.ndarray) -> np.ndarray:
    out = np.full((profile.shape[0], profile.shape[1], dst_lats.size), np.nan, dtype=np.float64)
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            row = profile[i, j, :]
            src_sorted, vals_sorted = _prepare_interp_inputs(src_lats, row)
            if src_sorted is None:
                continue
            out[i, j, :] = np.interp(dst_lats, src_sorted, vals_sorted, left=np.nan, right=np.nan)
    return out


def _interp_daily_climatology(profile: np.ndarray, src_lats: np.ndarray, dst_lats: np.ndarray) -> np.ndarray:
    out = np.full((profile.shape[0], dst_lats.size), np.nan, dtype=np.float64)
    for d in range(profile.shape[0]):
        row = profile[d, :]
        src_sorted, vals_sorted = _prepare_interp_inputs(src_lats, row)
        if src_sorted is None:
            continue
        out[d, :] = np.interp(dst_lats, src_sorted, vals_sorted, left=np.nan, right=np.nan)
    return out


def _extract_lonmean_profile(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: Tuple[float, float], lon_range: Tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lat_mask = _mask_between(lat, *lat_range)
    lon_mask = _mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f'纬度范围 {lat_range} 未命中。')
    if not np.any(lon_mask):
        raise ValueError(f'经度范围 {lon_range} 未命中。')
    sub = np.asarray(field[:, :, lat_mask, :][:, :, :, lon_mask], dtype=np.float64)
    profile = _safe_nanmean(sub, axis=3)
    return profile, np.asarray(lat[lat_mask], dtype=np.float64)


def build_all_profiles(smoothed_bundle: Dict[str, np.ndarray], clim_bundle: Dict[str, np.ndarray], settings: StagePartitionV1Settings) -> Dict[str, Dict[str, np.ndarray]]:
    lat = np.asarray(smoothed_bundle['lat'], dtype=np.float64)
    lon = np.asarray(smoothed_bundle['lon'], dtype=np.float64)
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for obj, spec in settings.profiles.profile_specs.items():
        lon_range = tuple(spec['lon_range'])
        grid_start, grid_end = spec['lat_grid']
        dst_lats = np.arange(grid_start, grid_end + 0.1, settings.profiles.lat_step_deg, dtype=np.float64)
        field_key = FIELD_MAP[obj]
        smoothed_field = np.asarray(smoothed_bundle[f'{field_key}_smoothed'], dtype=np.float64)
        clim_field = np.asarray(clim_bundle[f'{field_key}_clim'], dtype=np.float64)
        smoothed_prof_raw, src_lats = _extract_lonmean_profile(smoothed_field, lat, lon, (grid_start, grid_end), lon_range)
        clim_prof_raw, _ = _extract_lonmean_profile(clim_field[np.newaxis, :, :, :], lat, lon, (grid_start, grid_end), lon_range)
        smoothed_prof = _interp_profile_to_grid(smoothed_prof_raw, src_lats, dst_lats)
        clim_interp = _interp_daily_climatology(clim_prof_raw[0], src_lats, dst_lats)
        climatology_cube = np.repeat(clim_interp[np.newaxis, :, :], smoothed_prof.shape[0], axis=0)
        anomaly_prof = smoothed_prof - climatology_cube
        result[obj] = {
            'smoothed': smoothed_prof,
            'climatology': clim_interp,
            'anomaly': anomaly_prof,
            'lat_grid': dst_lats,
        }
    return result


def summarize_profile_validity(profile_dict: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for obj, payload in profile_dict.items():
        smoothed = np.asarray(payload['smoothed'], dtype=np.float64)
        for idx, lat in enumerate(payload['lat_grid']):
            slab = smoothed[:, :, idx]
            rows.append({
                'object_name': obj,
                'lat_value': float(lat),
                'nan_fraction': float(np.isnan(slab).sum() / slab.size),
                'finite_fraction': float(np.isfinite(slab).sum() / slab.size),
                'all_nan_any_day': bool(np.any(np.all(~np.isfinite(slab), axis=0))),
            })
    return pd.DataFrame(rows)
