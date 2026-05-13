from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .config import EdgeConfig


def _build_edges(series: pd.Series, primary_centers: list[int], backend_name: str, prefix: str, cfg: EdgeConfig) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=['window_id','backend','window_type','start_day','end_day','center_day','native_support','peak_prominence','peak_width'])
    s = series.sort_index().fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=cfg.min_distance_days)
    if peaks.size == 0:
        return pd.DataFrame(columns=['window_id','backend','window_type','start_day','end_day','center_day','native_support','peak_prominence','peak_width'])
    prominences = peak_prominences(values, peaks)[0]
    prom_threshold = float(np.quantile(prominences, cfg.prominence_quantile)) if prominences.size else np.inf
    widths, _, left_ips, right_ips = peak_widths(values, peaks, rel_height=cfg.width_rel_height)
    rows = []
    keep_count = 0
    primary_centers = np.asarray(primary_centers, dtype=int)
    for peak, prom, width, l, r in zip(peaks, prominences, widths, left_ips, right_ips):
        center_day = int(s.index[peak])
        if prom < prom_threshold:
            continue
        if primary_centers.size and np.min(np.abs(primary_centers - center_day)) <= cfg.min_distance_days:
            continue
        keep_count += 1
        rows.append({
            'window_id': f'{prefix}{keep_count:02d}',
            'backend': backend_name,
            'window_type': 'edge',
            'start_day': int(s.index[max(0, int(np.floor(l)))]),
            'end_day': int(s.index[min(len(s)-1, int(np.ceil(r)))]),
            'center_day': center_day,
            'native_support': float(s.iloc[peak]),
            'peak_prominence': float(prom),
            'peak_width': float(width),
        })
    return pd.DataFrame(rows)


def edge_from_movingwindow(scores: pd.Series, primary_windows: pd.DataFrame, cfg: EdgeConfig) -> pd.DataFrame:
    centers = primary_windows['center_day'].tolist() if not primary_windows.empty else []
    return _build_edges(scores, centers, 'movingwindow', 'MWE', cfg)


def edge_from_ruptures(profile: pd.Series, primary_windows: pd.DataFrame, cfg: EdgeConfig) -> pd.DataFrame:
    centers = primary_windows['center_day'].tolist() if not primary_windows.empty else []
    return _build_edges(profile, centers, 'ruptures_window', 'RWE', cfg)
