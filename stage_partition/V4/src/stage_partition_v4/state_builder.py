from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import ProfileGridConfig, StateBuilderConfig
from .safe_stats import safe_nanmean, safe_daily_energy, build_all_nan_mask

@dataclass
class PointLayerProfile:
    name: str
    raw_cube: np.ndarray
    lat_grid: np.ndarray
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]
    empty_slice_mask: np.ndarray

def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(lower, upper), max(lower, upper)
    return (arr >= lo) & (arr <= hi)

def _ascending_pair(lat_vals: np.ndarray, vals: np.ndarray):
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
            src = src_lats[valid]; vals = row[valid]
            src, vals = _ascending_pair(src, vals)
            out[i, j, :] = np.interp(dst_lats, src, vals, left=np.nan, right=np.nan)
    return out

def _build_profile_from_field(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lon_range: tuple[float, float], lat_range: tuple[float, float], lat_step_deg: float):
    lat_mask = _mask_between(lat, *lat_range); lon_mask = _mask_between(lon, *lon_range)
    if not np.any(lat_mask): raise ValueError(f'No latitude points in requested range {lat_range}')
    if not np.any(lon_mask): raise ValueError(f'No longitude points in requested range {lon_range}')
    src_lats = lat[lat_mask]
    subset = field[:, :, lat_mask, :][:, :, :, lon_mask]
    prof, valid_count = safe_nanmean(subset, axis=-1, return_valid_count=True)
    empty_slice_mask = valid_count == 0
    lo, hi = min(lat_range), max(lat_range)
    dst_lats = np.arange(lo, hi + 1e-9, lat_step_deg)
    prof_interp = _interp_profile_to_grid(prof, src_lats, dst_lats)
    empty_interp = _interp_profile_to_grid(empty_slice_mask.astype(float), src_lats, dst_lats)
    empty_interp = np.where(np.isfinite(empty_interp), empty_interp >= 0.5, True)
    return prof_interp, dst_lats, empty_interp

def build_profiles(smoothed: dict, cfg: ProfileGridConfig) -> dict[str, PointLayerProfile]:
    lat = smoothed['lat']; lon = smoothed['lon']; profs = {}
    specs = {'P': ('precip_smoothed', cfg.p_lon_range, cfg.p_lat_range), 'V': ('v850_smoothed', cfg.v_lon_range, cfg.v_lat_range), 'H': ('z500_smoothed', cfg.h_lon_range, cfg.h_lat_range), 'Je': ('u200_smoothed', cfg.je_lon_range, cfg.je_lat_range), 'Jw': ('u200_smoothed', cfg.jw_lon_range, cfg.jw_lat_range)}
    for name, (field_key, lon_range, lat_range) in specs.items():
        cube, lat_grid, empty_mask = _build_profile_from_field(smoothed[field_key], lat, lon, lon_range, lat_range, cfg.lat_step_deg)
        profs[name] = PointLayerProfile(name=name, raw_cube=cube, lat_grid=lat_grid, lon_range=lon_range, lat_range=lat_range, empty_slice_mask=empty_mask)
    return profs

def summarize_profile_validity(profiles: dict[str, PointLayerProfile]) -> pd.DataFrame:
    rows = []
    for name, obj in profiles.items():
        cube = obj.raw_cube
        for idx, latv in enumerate(obj.lat_grid):
            col = cube[:, :, idx]; finite = np.isfinite(col)
            rows.append({'object_name': name, 'lat_feature_index': idx, 'lat_value': float(latv), 'nan_fraction': float((~finite).mean()), 'finite_fraction': float(finite.mean()), 'all_nan_any_day': bool(np.any(np.all(~finite, axis=0))), 'all_nan_all_days': bool(np.all(~finite))})
    return pd.DataFrame(rows)

def summarize_profile_empty_slices(profiles: dict[str, PointLayerProfile]) -> pd.DataFrame:
    rows = []
    for name, obj in profiles.items():
        idx = np.argwhere(obj.empty_slice_mask)
        for year_idx, day_idx, lat_idx in idx:
            rows.append({'object_name': name, 'year_index': int(year_idx), 'day_index': int(day_idx), 'lat_feature_index': int(lat_idx), 'lat_value': float(obj.lat_grid[lat_idx]), 'reason': 'all_nan_over_lon'})
    return pd.DataFrame(rows, columns=['object_name','year_index','day_index','lat_feature_index','lat_value','reason'])

def _zscore_along_day(x: np.ndarray):
    mean, _ = safe_nanmean(x, axis=0, return_valid_count=True)
    centered = x - mean[None, :]
    sq = np.square(centered)
    var, _ = safe_nanmean(sq, axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    z = centered / std[None, :]
    return z, mean, std

def _apply_equal_block_contribution(x: np.ndarray, block_slices: dict[str, slice]):
    y = x.copy(); rows = []
    for name, slc in block_slices.items():
        width = slc.stop - slc.start; factor = 1.0 / np.sqrt(width)
        before = float(np.sqrt(np.nanmean(np.square(y[:, slc])))); y[:, slc] *= factor; after = float(np.sqrt(np.nanmean(np.square(y[:, slc]))))
        rows.append({'object_name': name, 'block_size': int(width), 'applied_weight': float(factor), 'weighted_rms_before': before, 'weighted_rms_after': after})
    return y, pd.DataFrame(rows)

def _build_state_from_seasonal_blocks(seasonal_blocks: list[np.ndarray], feature_rows: list[dict], cfg: StateBuilderConfig) -> dict:
    raw = np.concatenate(seasonal_blocks, axis=1)
    z, feat_mean, feat_std = _zscore_along_day(raw); scaled = z.copy()
    block_slices = {}
    for item in feature_rows:
        if item['feature_in_object'] == 0:
            block_slices[item['object_name']] = slice(item['feature_index'], item['feature_index'] + item['object_width'])
    weights = pd.DataFrame([])
    if cfg.block_equal_contribution:
        scaled, weights = _apply_equal_block_contribution(scaled, block_slices)
    valid_day_mask = np.all(np.isfinite(scaled), axis=1) if cfg.trim_invalid_days else np.ones(raw.shape[0], dtype=bool)
    valid_day_index = np.where(valid_day_mask)[0].astype(int)
    raw_rows = []; std_rows = []; empty_rows = []
    for name, slc in block_slices.items():
        raw_rows.append({'object_name': name, **safe_daily_energy(raw[:, slc])}); std_rows.append({'object_name': name, **safe_daily_energy(z[:, slc])})
        all_nan_mask = build_all_nan_mask(raw[:, slc], axis=0); valid_days_per_feature = np.sum(np.isfinite(raw[:, slc]), axis=0)
        for j in range(slc.stop - slc.start):
            empty_rows.append({'object_name': name, 'feature_index': int(slc.start + j), 'all_nan_flag': bool(all_nan_mask[j]), 'n_valid_days': int(valid_days_per_feature[j]), 'n_missing_days': int(raw.shape[0] - valid_days_per_feature[j])})
    meta = {'n_days': int(raw.shape[0]), 'n_features': int(raw.shape[1]), 'n_valid_days': int(valid_day_index.size), 'n_invalid_days': int(raw.shape[0] - valid_day_index.size), 'valid_day_index': valid_day_index.tolist(), 'invalid_day_index': np.where(~valid_day_mask)[0].astype(int).tolist(), 'block_slices': {k: [v.start, v.stop] for k, v in block_slices.items()}, 'state_expression_name': 'raw_smoothed_zscore_block_equal'}
    scale_rows = []
    for item in feature_rows:
        idx = item['feature_index']
        scale_rows.append({'feature_index': idx, 'object_name': item['object_name'], 'lat_value': item['lat_value'], 'raw_mean_day': float(feat_mean[idx]), 'raw_std_day': float(feat_std[idx]), 'z_mean_day': float(np.nanmean(z[:, idx])) if np.isfinite(z[:, idx]).any() else np.nan, 'z_std_day': float(np.nanstd(z[:, idx])) if np.isfinite(z[:, idx]).any() else np.nan})
    return {'raw_matrix': raw, 'state_matrix': scaled, 'valid_day_mask': valid_day_mask, 'valid_day_index': valid_day_index, 'state_vector_meta': meta, 'state_feature_scale_before_after': pd.DataFrame(scale_rows), 'state_block_energy_before_after': {'raw': pd.DataFrame(raw_rows), 'standardized': pd.DataFrame(std_rows), 'weights': weights}, 'state_empty_feature_audit': pd.DataFrame(empty_rows)}

def _build_seasonal(profiles: dict[str, PointLayerProfile], year_indices=None):
    seasonal_blocks = []; feature_rows = []; start = 0
    for name in ['P','V','H','Je','Jw']:
        cube = profiles[name].raw_cube if year_indices is None else profiles[name].raw_cube[np.asarray(year_indices, dtype=int), :, :]
        seasonal, _ = safe_nanmean(cube, axis=0, return_valid_count=True)
        seasonal_blocks.append(seasonal); width = seasonal.shape[1]
        for j, latv in enumerate(profiles[name].lat_grid):
            feature_rows.append({'feature_index': start + j, 'object_name': name, 'object_width': width, 'feature_in_object': j, 'lat_value': float(latv)})
        start += width
    return seasonal_blocks, feature_rows

def build_state_matrix(profiles: dict[str, PointLayerProfile], cfg: StateBuilderConfig) -> dict:
    seasonal_blocks, feature_rows = _build_seasonal(profiles, None)
    out = _build_state_from_seasonal_blocks(seasonal_blocks, feature_rows, cfg)
    out['feature_table'] = pd.DataFrame(feature_rows)[['feature_index','object_name','lat_value']].copy(); return out

def build_year_state_matrix(profiles: dict[str, PointLayerProfile], year_idx: int, cfg: StateBuilderConfig) -> dict:
    seasonal_blocks=[]; feature_rows=[]; start=0
    for name in ['P','V','H','Je','Jw']:
        seasonal = profiles[name].raw_cube[year_idx, :, :].astype(float); seasonal_blocks.append(seasonal); width = seasonal.shape[1]
        for j, latv in enumerate(profiles[name].lat_grid):
            feature_rows.append({'feature_index': start + j, 'object_name': name, 'object_width': width, 'feature_in_object': j, 'lat_value': float(latv)})
        start += width
    out = _build_state_from_seasonal_blocks(seasonal_blocks, feature_rows, cfg)
    out['feature_table'] = pd.DataFrame(feature_rows)[['feature_index','object_name','lat_value']].copy(); return out

def build_resampled_state_matrix(profiles: dict[str, PointLayerProfile], sampled_year_indices: np.ndarray, cfg: StateBuilderConfig) -> dict:
    seasonal_blocks, feature_rows = _build_seasonal(profiles, np.asarray(sampled_year_indices, dtype=int))
    out = _build_state_from_seasonal_blocks(seasonal_blocks, feature_rows, cfg)
    out['feature_table'] = pd.DataFrame(feature_rows)[['feature_index','object_name','lat_value']].copy(); out['sampled_year_indices'] = np.asarray(sampled_year_indices, dtype=int); return out
