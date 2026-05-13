from __future__ import annotations

import numpy as np
import pandas as pd

from .profiles import ObjectProfile
from .config import StateVectorConfig
from .safe_stats import safe_nanmean, safe_daily_energy, build_all_nan_mask


def _zscore_along_day(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, mean_count = safe_nanmean(x, axis=0, return_valid_count=True)
    centered = x - mean[None, :]
    sq = np.square(centered)
    var, var_count = safe_nanmean(sq, axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    z = centered / std[None, :]
    return z, mean, std


def _apply_equal_block_contribution(x: np.ndarray, block_slices: dict[str, slice]) -> tuple[np.ndarray, pd.DataFrame]:
    y = x.copy()
    rows = []
    for name, slc in block_slices.items():
        width = slc.stop - slc.start
        factor = 1.0 / np.sqrt(width)
        before = float(np.sqrt(np.nanmean(np.square(y[:, slc]))))
        y[:, slc] = y[:, slc] * factor
        after = float(np.sqrt(np.nanmean(np.square(y[:, slc]))))
        rows.append({
            'object_name': name,
            'block_size': int(width),
            'applied_weight': float(factor),
            'weighted_rms_before': before,
            'weighted_rms_after': after,
        })
    return y, pd.DataFrame(rows)


def build_state_matrix(profiles: dict[str, ObjectProfile], cfg: StateVectorConfig) -> dict:
    feature_arrays = []
    feature_rows = []
    start = 0
    block_slices: dict[str, slice] = {}
    raw_energy_rows = []
    for name in ['P', 'V', 'H', 'Je', 'Jw']:
        cube = profiles[name].raw_cube
        seasonal, seasonal_valid_count = safe_nanmean(cube, axis=0, return_valid_count=True)  # [day, lat_feature]
        width = seasonal.shape[1]
        feature_arrays.append(seasonal)
        block_slices[name] = slice(start, start + width)
        for j, latv in enumerate(profiles[name].lat_grid):
            feature_rows.append({'feature_index': start + j, 'object_name': name, 'lat_value': float(latv)})
        raw_energy = safe_daily_energy(seasonal)
        raw_energy_rows.append({
            'object_name': name,
            'rms': raw_energy['rms'],
            'daily_energy_mean': raw_energy['daily_energy_mean'],
            'p95_abs': raw_energy['p95_abs'],
        })
        start += width
    raw = np.concatenate(feature_arrays, axis=1)
    z, feat_mean, feat_std = _zscore_along_day(raw)
    scaled = z.copy()
    block_weight_df = pd.DataFrame([])
    if cfg.block_equal_contribution:
        scaled, block_weight_df = _apply_equal_block_contribution(scaled, block_slices)
    std_energy_rows = []
    empty_feature_rows = []
    for name, slc in block_slices.items():
        block = z[:, slc]
        stats = safe_daily_energy(block)
        std_energy_rows.append({
            'object_name': name,
            'rms': stats['rms'],
            'daily_energy_mean': stats['daily_energy_mean'],
            'p95_abs': stats['p95_abs'],
        })
        seasonal_block = raw[:, slc]
        all_nan_mask = build_all_nan_mask(seasonal_block, axis=0)
        valid_days_per_feature = np.sum(np.isfinite(seasonal_block), axis=0)
        for j in range(seasonal_block.shape[1]):
            empty_feature_rows.append({
                'object_name': name,
                'feature_index': int(slc.start + j),
                'all_nan_flag': bool(all_nan_mask[j]),
                'n_valid_days': int(valid_days_per_feature[j]),
                'n_missing_days': int(seasonal_block.shape[0] - valid_days_per_feature[j]),
            })
    valid_day_mask = np.all(np.isfinite(scaled), axis=1)
    valid_day_index = np.where(valid_day_mask)[0].astype(int)
    meta = {
        'n_days': int(raw.shape[0]),
        'n_features': int(raw.shape[1]),
        'n_valid_days': int(valid_day_index.size),
        'n_invalid_days': int(raw.shape[0] - valid_day_index.size),
        'valid_day_index': valid_day_index.tolist(),
        'invalid_day_index': np.where(~valid_day_mask)[0].astype(int).tolist(),
        'block_slices': {k: [v.start, v.stop] for k, v in block_slices.items()},
    }
    scale_rows = []
    for i in range(raw.shape[1]):
        scale_rows.append({
            'feature_index': i,
            'object_name': feature_rows[i]['object_name'],
            'lat_value': feature_rows[i]['lat_value'],
            'raw_mean_day': float(feat_mean[i]),
            'raw_std_day': float(feat_std[i]),
            'z_mean_day': float(np.nanmean(z[:, i])),
            'z_std_day': float(np.nanstd(z[:, i])),
        })
    return {
        'raw_matrix': raw,
        'state_matrix': scaled,
        'feature_table': pd.DataFrame(feature_rows),
        'state_vector_meta': meta,
        'state_feature_scale_before_after': pd.DataFrame(scale_rows),
        'state_block_energy_before_after': {
            'raw': pd.DataFrame(raw_energy_rows),
            'standardized': pd.DataFrame(std_energy_rows),
            'weights': block_weight_df,
        },
        'state_empty_feature_audit': pd.DataFrame(empty_feature_rows),
        'block_slices': block_slices,
        'valid_day_mask': valid_day_mask,
        'valid_day_index': valid_day_index,
    }
