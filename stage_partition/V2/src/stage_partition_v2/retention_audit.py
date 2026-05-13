from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from .config import EdgeConfig


def _peak_table(series: pd.Series, cfg: EdgeConfig) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=['peak_day', 'peak_value', 'peak_rank', 'peak_prominence'])
    s = series.sort_index().astype(float).fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=cfg.min_distance_days)
    if peaks.size == 0:
        return pd.DataFrame(columns=['peak_day', 'peak_value', 'peak_rank', 'peak_prominence'])
    prominences = peak_prominences(values, peaks)[0]
    df = pd.DataFrame({
        'peak_day': s.index.to_numpy(dtype=int)[peaks],
        'peak_value': values[peaks],
        'peak_prominence': prominences.astype(float),
    }).sort_values('peak_value', ascending=False).reset_index(drop=True)
    df['peak_rank'] = np.arange(1, len(df) + 1)
    return df


def _nearest_match(day: int, centers: np.ndarray, tol: int) -> bool:
    if centers.size == 0:
        return False
    return bool(np.min(np.abs(centers - int(day))) <= tol)


def build_retention_audit(series: pd.Series, primary_windows: pd.DataFrame, edge_windows: pd.DataFrame,
                          backend: str, cfg: EdgeConfig) -> pd.DataFrame:
    peaks = _peak_table(series, cfg)
    if peaks.empty:
        return pd.DataFrame(columns=[
            'backend', 'peak_day', 'peak_value', 'peak_prominence', 'peak_rank',
            'candidate_kind', 'retained_as', 'drop_stage', 'drop_reason'
        ])
    primary_centers = primary_windows['center_day'].to_numpy(dtype=int) if not primary_windows.empty else np.asarray([], dtype=int)
    edge_centers = edge_windows['center_day'].to_numpy(dtype=int) if not edge_windows.empty else np.asarray([], dtype=int)
    rows = []
    for _, row in peaks.iterrows():
        peak_day = int(row['peak_day'])
        retained_as = 'dropped'
        drop_stage = 'backend_native_rule'
        drop_reason = 'not_promoted_to_primary_or_edge'
        if _nearest_match(peak_day, primary_centers, cfg.nearest_peak_search_radius):
            retained_as = 'primary'
            drop_stage = None
            drop_reason = None
        elif _nearest_match(peak_day, edge_centers, cfg.nearest_peak_search_radius):
            retained_as = 'edge'
            drop_stage = 'edge_selection'
            drop_reason = None
        rows.append({
            'backend': backend,
            'peak_day': peak_day,
            'peak_value': float(row['peak_value']),
            'peak_prominence': float(row['peak_prominence']),
            'peak_rank': int(row['peak_rank']),
            'candidate_kind': 'raw_local_peak',
            'retained_as': retained_as,
            'drop_stage': drop_stage,
            'drop_reason': drop_reason,
        })
    return pd.DataFrame(rows)
