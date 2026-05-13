from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV2Settings
from .safe_stats import safe_nanmean
from .backend_ruptures_window import run_ruptures_window
from .window_native_ruptures import build_primary_windows_from_ruptures


def _drop_invalid_days(state_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_day_mask = np.all(np.isfinite(state_matrix), axis=1)
    valid_day_index = np.where(valid_day_mask)[0].astype(int)
    return state_matrix[valid_day_mask, :], valid_day_index


def parameter_path_stability_ruptures(state_matrix: np.ndarray, settings: StagePartitionV2Settings) -> pd.DataFrame:
    state_valid, day_index = _drop_invalid_days(state_matrix)
    rw_rows = []
    for width in settings.support.ruptures_width_grid:
        for pen in settings.support.ruptures_pen_grid:
            cfg = settings.ruptures_window
            cfg_local = type(cfg)(**{**cfg.__dict__, 'width': int(width), 'selection_mode': 'pen', 'pen': float(pen)})
            out = run_ruptures_window(state_valid, cfg_local, day_index=day_index)
            wins, _, _, _ = build_primary_windows_from_ruptures(out['points'], out['profile'], settings.edge, settings.ruptures_window_construction)
            for _, row in wins.iterrows():
                rw_rows.append({
                    'width': width,
                    'pen': pen,
                    'window_id': row['window_id'],
                    'start_day': row['start_day'],
                    'end_day': row['end_day'],
                    'center_day': row['center_day'],
                    'main_peak_day': row['main_peak_day'],
                    'n_support_points': row['n_support_points'],
                })
    return pd.DataFrame(rw_rows)


def bootstrap_stability_ruptures(profile_dict: dict, settings: StagePartitionV2Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(settings.support.random_seed)
    n_years = next(iter(profile_dict.values())).raw_cube.shape[0]
    years_idx = np.arange(n_years)
    rw_rows = []
    invalid_rows = []
    sample_rows = []
    min_required = max(2 * int(settings.ruptures_window.width), 3)
    for rep in range(settings.support.bootstrap_reps):
        sample_idx = rng.choice(years_idx, size=n_years, replace=True)
        blocks = []
        block_order = ['P', 'V', 'H', 'Je', 'Jw']
        per_object_invalid_days = {}
        for name in block_order:
            seasonal, valid_count = safe_nanmean(profile_dict[name].raw_cube[sample_idx, :, :], axis=0, return_valid_count=True)
            blocks.append(seasonal)
            all_nan_mask = valid_count == 0
            per_object_invalid_days[name] = int(np.sum(np.any(all_nan_mask, axis=1)))
            invalid_rows.append({
                'backend': 'ruptures_window',
                'rep': rep,
                'object_name': name,
                'n_invalid_day_lat_cells': int(np.sum(all_nan_mask)),
                'n_valid_day_lat_cells': int(np.sum(valid_count > 0)),
                'n_invalid_days': int(np.sum(np.any(all_nan_mask, axis=1))),
                'skip_reason': '' if np.sum(valid_count > 0) > 0 else 'all_nan_object',
            })
        raw = np.concatenate(blocks, axis=1)
        mean, mean_count = safe_nanmean(raw, axis=0, return_valid_count=True)
        centered = raw - mean[None, :]
        var = safe_nanmean(np.square(centered), axis=0)
        std = np.sqrt(var)
        std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
        state = centered / std[None, :]
        widths = [b.shape[1] for b in blocks]
        s = 0
        for width in widths:
            state[:, s:s + width] *= 1.0 / np.sqrt(width)
            s += width
        valid_day_mask = np.all(np.isfinite(state), axis=1)
        valid_day_index = np.where(valid_day_mask)[0].astype(int)
        sample_rows.append({
            'backend': 'ruptures_window',
            'rep': rep,
            'n_total_days': int(state.shape[0]),
            'n_valid_days': int(valid_day_index.size),
            'n_invalid_days': int(state.shape[0] - valid_day_index.size),
            'invalid_fraction': float((state.shape[0] - valid_day_index.size) / state.shape[0]),
            'sample_valid_flag': bool(valid_day_index.size >= min_required),
            'skip_reason': '' if valid_day_index.size >= min_required else 'insufficient_valid_days',
        })
        if valid_day_index.size < min_required:
            continue
        state_valid = state[valid_day_mask, :]
        rw_out = run_ruptures_window(state_valid, settings.ruptures_window, day_index=valid_day_index)
        rw_wins, _, _, _ = build_primary_windows_from_ruptures(rw_out['points'], rw_out['profile'], settings.edge, settings.ruptures_window_construction)
        for _, row in rw_wins.iterrows():
            rw_rows.append({
                'rep': rep,
                'center_day': row['center_day'],
                'start_day': row['start_day'],
                'end_day': row['end_day'],
                'main_peak_day': row['main_peak_day'],
                'n_support_points': row['n_support_points'],
            })
    invalid_df = pd.DataFrame(invalid_rows)
    sample_df = pd.DataFrame(sample_rows)
    summary_df = pd.DataFrame(columns=['backend','n_total_samples','n_valid_samples','n_invalid_samples','invalid_fraction'])
    if not sample_df.empty:
        n_total = len(sample_df)
        n_valid = int(sample_df['sample_valid_flag'].sum())
        n_invalid = n_total - n_valid
        summary_df = pd.DataFrame([{
            'backend': 'ruptures_window',
            'n_total_samples': n_total,
            'n_valid_samples': n_valid,
            'n_invalid_samples': n_invalid,
            'invalid_fraction': float(n_invalid / n_total) if n_total > 0 else np.nan,
        }])
    return pd.DataFrame(rw_rows), invalid_df, summary_df


def permutation_significance_ruptures(state_matrix: np.ndarray, settings: StagePartitionV2Settings) -> pd.DataFrame:
    state_valid, day_index = _drop_invalid_days(state_matrix)
    rng = np.random.default_rng(settings.support.random_seed)
    reps = settings.support.permutation_reps
    rw_out = run_ruptures_window(state_valid, settings.ruptures_window, day_index=day_index)
    rw_obs = float(np.nanmax(rw_out['profile'].to_numpy(dtype=float))) if not rw_out['profile'].empty else np.nan
    rw_null = []
    for _ in range(reps):
        perm = rng.permutation(state_valid.shape[0])
        Xp = state_valid[perm, :]
        perm_days = day_index[perm]
        rw_perm = run_ruptures_window(Xp, settings.ruptures_window, day_index=perm_days)
        rw_null.append(float(np.nanmax(rw_perm['profile'].to_numpy(dtype=float))) if not rw_perm['profile'].empty else np.nan)
    rw_null_arr = np.asarray(rw_null, dtype=float)
    rw_p = float((np.sum(rw_null_arr >= rw_obs) + 1) / (np.sum(np.isfinite(rw_null_arr)) + 1)) if np.isfinite(rw_obs) else np.nan
    return pd.DataFrame([{
        'backend': 'ruptures_window',
        'observed_max_score': rw_obs,
        'empirical_pvalue': rw_p,
    }])
