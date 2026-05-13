from __future__ import annotations

import numpy as np
import pandas as pd


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    med = float(np.nanmedian(values))
    return float(np.nanmedian(np.abs(values - med)))


def select_best_matched_peak(
    peaks_df: pd.DataFrame,
    target_day: int,
    max_center_offset_days: int = 2,
    prominence_min: float = 0.0,
    select_mode: str = 'closest_then_prominence_then_score',
):
    if peaks_df is None or peaks_df.empty:
        return None
    sub = peaks_df.copy()
    sub = sub[pd.to_numeric(sub['peak_day'], errors='coerce').notna()].copy()
    if sub.empty:
        return None
    sub['peak_day'] = sub['peak_day'].astype(int)
    if 'peak_prominence' not in sub.columns:
        sub['peak_prominence'] = np.nan
    if 'peak_rank' not in sub.columns:
        sub['peak_rank'] = np.arange(1, len(sub) + 1)
    sub = sub[sub['peak_day'].between(int(target_day) - int(max_center_offset_days), int(target_day) + int(max_center_offset_days))].copy()
    if sub.empty:
        return None
    sub = sub[sub['peak_prominence'].fillna(0.0) >= float(prominence_min)].copy()
    if sub.empty:
        return None
    sub['center_offset_days'] = sub['peak_day'].astype(int) - int(target_day)
    sub['center_offset_abs'] = sub['center_offset_days'].abs()
    if select_mode == 'closest_then_prominence_then_score':
        sub = sub.sort_values(['center_offset_abs', 'peak_prominence', 'peak_score'], ascending=[True, False, False])
    else:
        sub = sub.sort_values(['peak_score', 'peak_prominence', 'center_offset_abs'], ascending=[False, False, True])
    return sub.iloc[0].to_dict()


def compute_local_matched_statistic(
    profile: pd.Series,
    matched_peak_day: int,
    matched_peak_score: float,
    target_day: int,
    background_window_days: int = 7,
    exclude_core_days: int = 1,
):
    if profile is None or profile.empty or not np.isfinite(matched_peak_score):
        return float('nan'), float('nan'), float('nan')
    s = profile.sort_index()
    idx = s.index.to_numpy(dtype=int)
    vals = s.to_numpy(dtype=float)
    left = int(target_day) - int(background_window_days)
    right = int(target_day) + int(background_window_days)
    bg_mask = (idx >= left) & (idx <= right)
    core_left = int(matched_peak_day) - int(exclude_core_days)
    core_right = int(matched_peak_day) + int(exclude_core_days)
    core_mask = (idx >= core_left) & (idx <= core_right)
    bg = vals[bg_mask & (~core_mask)]
    bg = bg[np.isfinite(bg)]
    if bg.size < 3:
        return float('nan'), float('nan'), float('nan')
    med = float(np.nanmedian(bg))
    mad = _mad(bg)
    if not np.isfinite(mad) or mad <= 0:
        return float('nan'), med, mad
    stat = float((float(matched_peak_score) - med) / (mad + 1e-8))
    return stat, med, mad


def build_matched_stat_record(
    *,
    replicate_type: str,
    replicate_id: int,
    point_id: str,
    point_day: int,
    matched_peak_exists: bool,
    matched_peak_day,
    center_offset_days,
    matched_peak_score,
    matched_peak_prominence,
    matched_peak_rank,
    matched_stat,
    local_background_median,
    local_background_mad,
    match_quality_flag: str,
    local_window_days: int,
):
    return {
        'replicate_type': replicate_type,
        'replicate_id': replicate_id,
        'point_id': point_id,
        'point_day': int(point_day),
        'matched_peak_exists': bool(matched_peak_exists),
        'matched_peak_day': matched_peak_day,
        'center_offset_days': center_offset_days,
        'matched_peak_score': matched_peak_score,
        'matched_peak_prominence': matched_peak_prominence,
        'matched_peak_rank': matched_peak_rank,
        'matched_stat': matched_stat,
        'local_background_median': local_background_median,
        'local_background_mad': local_background_mad,
        'match_quality_flag': match_quality_flag,
        'local_window_start': int(point_day) - int(local_window_days),
        'local_window_end': int(point_day) + int(local_window_days),
    }
