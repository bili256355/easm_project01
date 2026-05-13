from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import ProfileGridConfig
from .safe_stats import safe_nanmean


@dataclass
class ObjectProfile:
    name: str
    raw_cube: np.ndarray  # [year, day, lat_feature]
    lat_grid: np.ndarray
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]
    empty_slice_mask: np.ndarray  # [year, day, lat_feature], True where lon-mean slice was fully missing


def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(lower, upper), max(lower, upper)
    return (arr >= lo) & (arr <= hi)


def _ascending_pair(lat_vals: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(lat_vals)
    return lat_vals[order], vals[..., order]


def _interp_profile_to_grid(profile: np.ndarray, src_lats: np.ndarray, dst_lats: np.ndarray) -> np.ndarray:
    out = np.full((profile.shape[0], profile.shape[1], dst_lats.size), np.nan, dtype=float)
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            row = profile[i, j, :]
            valid = np.isfinite(row) & np.isfinite(src_lats)
            if valid.sum() < 2:
                continue
            src = src_lats[valid]
            vals = row[valid]
            src, vals = _ascending_pair(src, vals)
            out[i, j, :] = np.interp(dst_lats, src, vals, left=np.nan, right=np.nan)
    return out


def _build_profile_from_field(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                              lon_range: tuple[float, float], lat_range: tuple[float, float], lat_step_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_mask = _mask_between(lat, *lat_range)
    lon_mask = _mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f'No latitude points in requested range {lat_range}')
    if not np.any(lon_mask):
        raise ValueError(f'No longitude points in requested range {lon_range}')
    src_lats = lat[lat_mask]
    subset = field[:, :, lat_mask, :][:, :, :, lon_mask]
    prof, valid_count = safe_nanmean(subset, axis=-1, return_valid_count=True)  # [year, day, lat]
    empty_slice_mask = valid_count == 0
    lo, hi = min(lat_range), max(lat_range)
    dst_lats = np.arange(lo, hi + 1e-9, lat_step_deg)
    prof_interp = _interp_profile_to_grid(prof, src_lats, dst_lats)
    empty_interp = _interp_profile_to_grid(empty_slice_mask.astype(float), src_lats, dst_lats)
    empty_interp = np.where(np.isfinite(empty_interp), empty_interp >= 0.5, True)
    return prof_interp, dst_lats, empty_interp


def build_profiles(smoothed: dict, cfg: ProfileGridConfig) -> dict[str, ObjectProfile]:
    lat = smoothed['lat']
    lon = smoothed['lon']
    profs: dict[str, ObjectProfile] = {}
    specs = {
        'P': ('precip_smoothed', cfg.p_lon_range, cfg.p_lat_range),
        'V': ('v850_smoothed', cfg.v_lon_range, cfg.v_lat_range),
        'H': ('z500_smoothed', cfg.h_lon_range, cfg.h_lat_range),
        'Je': ('u200_smoothed', cfg.je_lon_range, cfg.je_lat_range),
        'Jw': ('u200_smoothed', cfg.jw_lon_range, cfg.jw_lat_range),
    }
    for name, (field_key, lon_range, lat_range) in specs.items():
        cube, lat_grid, empty_mask = _build_profile_from_field(smoothed[field_key], lat, lon, lon_range, lat_range, cfg.lat_step_deg)
        profs[name] = ObjectProfile(name=name, raw_cube=cube, lat_grid=lat_grid, lon_range=lon_range, lat_range=lat_range, empty_slice_mask=empty_mask)
    return profs


def summarize_profile_validity(profiles: dict[str, ObjectProfile]) -> pd.DataFrame:
    rows = []
    for name, obj in profiles.items():
        cube = obj.raw_cube
        for idx, latv in enumerate(obj.lat_grid):
            col = cube[:, :, idx]
            finite = np.isfinite(col)
            rows.append({
                'object_name': name,
                'lat_feature_index': idx,
                'lat_value': float(latv),
                'nan_fraction': float((~finite).mean()),
                'finite_fraction': float(finite.mean()),
                'all_nan_any_day': bool(np.any(np.all(~finite, axis=0))),
                'all_nan_all_days': bool(np.all(~finite)),
            })
    return pd.DataFrame(rows)


def summarize_profile_empty_slices(profiles: dict[str, ObjectProfile]) -> pd.DataFrame:
    rows = []
    for name, obj in profiles.items():
        idx = np.argwhere(obj.empty_slice_mask)
        for year_idx, day_idx, lat_idx in idx:
            rows.append({
                'object_name': name,
                'year_index': int(year_idx),
                'day_index': int(day_idx),
                'lat_feature_index': int(lat_idx),
                'lat_value': float(obj.lat_grid[lat_idx]),
                'reason': 'all_nan_over_lon',
            })
    return pd.DataFrame(rows, columns=['object_name','year_index','day_index','lat_feature_index','lat_value','reason'])
