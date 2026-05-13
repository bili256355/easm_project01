from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, peak_prominences

from .config import EdgeConfig


def _nearest_peak(peaks: np.ndarray, target: int, radius: int) -> int | None:
    if peaks.size == 0:
        return None
    diffs = np.abs(peaks - target)
    k = int(np.argmin(diffs))
    if diffs[k] <= radius:
        return int(peaks[k])
    return None


def _interval_from_peak(series: pd.Series, peak_idx: int, rel_height: float) -> tuple[int, int, float, float]:
    values = series.fillna(0.0).to_numpy(dtype=float)
    peaks = np.array([peak_idx], dtype=int)
    prominences = peak_prominences(values, peaks)[0]
    widths, _, left_ips, right_ips = peak_widths(values, peaks, rel_height=rel_height)
    start = int(max(0, np.floor(left_ips[0])))
    end = int(min(len(values) - 1, np.ceil(right_ips[0])))
    return start, end, float(prominences[0]), float(widths[0])


def windows_from_movingwindow(points: pd.Series, scores: pd.Series, edge_cfg: EdgeConfig) -> pd.DataFrame:
    s = scores.sort_index().fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=edge_cfg.min_distance_days)
    rows = []
    for i, point in enumerate(points.astype(int).tolist(), start=1):
        peak = _nearest_peak(peaks, int(point), edge_cfg.nearest_peak_search_radius)
        if peak is None:
            peak = int(point)
        start, end, prom, width = _interval_from_peak(s, peak, edge_cfg.width_rel_height)
        rows.append({
            'window_id': f'MW{i:02d}',
            'backend': 'movingwindow',
            'window_type': 'primary',
            'start_day': int(s.index[start]),
            'end_day': int(s.index[end]),
            'center_day': int(s.index[peak]),
            'native_support': float(s.iloc[peak]),
            'peak_prominence': prom,
            'peak_width': width,
        })
    return pd.DataFrame(rows)


def windows_from_ruptures(points: pd.Series, profile: pd.Series, edge_cfg: EdgeConfig) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame(columns=['window_id','backend','window_type','start_day','end_day','center_day','native_support','peak_prominence','peak_width'])
    s = profile.sort_index().fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=edge_cfg.min_distance_days)
    rows = []
    for i, point in enumerate(points.astype(int).tolist(), start=1):
        peak = _nearest_peak(peaks, int(point), edge_cfg.nearest_peak_search_radius)
        if peak is None:
            # map breakpoint to nearest index in profile
            peak = int(np.argmin(np.abs(s.index.to_numpy(dtype=int) - int(point))))
        start, end, prom, width = _interval_from_peak(s, peak, edge_cfg.width_rel_height)
        rows.append({
            'window_id': f'RW{i:02d}',
            'backend': 'ruptures_window',
            'window_type': 'primary',
            'start_day': int(s.index[start]),
            'end_day': int(s.index[end]),
            'center_day': int(s.index[peak]),
            'native_support': float(s.iloc[peak]),
            'peak_prominence': prom,
            'peak_width': width,
        })
    return pd.DataFrame(rows)
