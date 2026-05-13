from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .config import StagePartitionV2Settings
from .backend_movingwindow import movingwindow_api_mode, run_movingwindow
from .translate_windows import windows_from_movingwindow
from .edge_from_backend import edge_from_movingwindow


def _peak_core(scores: pd.Series, min_distance_days: int):
    s = scores.sort_index().astype(float).fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=min_distance_days)
    if peaks.size == 0:
        return s, np.asarray([], dtype=int), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    prominences = peak_prominences(values, peaks)[0]
    widths, _, left_ips, right_ips = peak_widths(values, peaks, rel_height=0.5)
    return s, peaks, prominences.astype(float), widths.astype(float), left_ips.astype(float), right_ips.astype(float)


def movingwindow_score_shape_audit(scores: pd.Series, cfg) -> pd.DataFrame:
    s, peaks, prominences, widths, left_ips, right_ips = _peak_core(scores, cfg.min_distance_days)
    rows = []
    values = s.to_numpy(dtype=float)
    for peak, prom, width, l, r in zip(peaks, prominences, widths, left_ips, right_ips):
        left_i = max(0, int(np.floor(l)))
        right_i = min(len(values) - 1, int(np.ceil(r)))
        peak_val = float(values[peak])
        left_slope = float(peak_val - values[left_i]) if left_i < peak else 0.0
        right_slope = float(peak_val - values[right_i]) if right_i > peak else 0.0
        plateau = int(np.sum(np.abs(values[left_i:right_i + 1] - peak_val) < 1e-8))
        rows.append({
            'peak_day': int(s.index[peak]),
            'score_peak': peak_val,
            'prominence': float(prom),
            'peak_width_days': float(width),
            'left_slope': left_slope,
            'right_slope': right_slope,
            'plateau_length': plateau,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values('score_peak', ascending=False).reset_index(drop=True)
        out['local_rank'] = np.arange(1, len(out) + 1)
    return out


def movingwindow_threshold_audit(scores: pd.Series, primary_windows: pd.DataFrame, edge_windows: pd.DataFrame,
                                 settings: StagePartitionV2Settings) -> pd.DataFrame:
    shape = movingwindow_score_shape_audit(scores, settings.edge)
    if shape.empty:
        return pd.DataFrame(columns=['peak_day', 'score_peak', 'threshold_used', 'distance_to_threshold', 'passes_threshold', 'blocked_by_detection_interval', 'native_accept_flag', 'api_mode'])
    api_mode = movingwindow_api_mode()
    primary_centers = primary_windows['center_day'].to_numpy(dtype=int) if not primary_windows.empty else np.asarray([], dtype=int)
    edge_centers = edge_windows['center_day'].to_numpy(dtype=int) if not edge_windows.empty else np.asarray([], dtype=int)
    threshold_used = None
    if api_mode == 'threshold_api' and settings.movingwindow.threshold_scale is not None:
        threshold_used = float(settings.movingwindow.threshold_scale)
    rows = []
    for _, row in shape.iterrows():
        day = int(row['peak_day'])
        native_accept = bool(day in primary_centers)
        rows.append({
            'peak_day': day,
            'score_peak': float(row['score_peak']),
            'threshold_used': threshold_used,
            'distance_to_threshold': (float(row['score_peak']) - threshold_used) if threshold_used is not None else np.nan,
            'passes_threshold': (float(row['score_peak']) >= threshold_used) if threshold_used is not None else np.nan,
            'blocked_by_detection_interval': False if native_accept else np.nan,
            'native_accept_flag': native_accept,
            'edge_flag': bool(day in edge_centers),
            'api_mode': api_mode,
        })
    return pd.DataFrame(rows)


def movingwindow_peak_rank_audit(scores: pd.Series, primary_windows: pd.DataFrame, edge_windows: pd.DataFrame,
                                 settings: StagePartitionV2Settings) -> pd.DataFrame:
    shape = movingwindow_score_shape_audit(scores, settings.edge)
    if shape.empty:
        return pd.DataFrame(columns=['peak_day', 'score_peak', 'global_rank', 'prominence_rank', 'width_rank', 'retained_as_primary', 'retained_as_edge'])
    out = shape.copy()
    out['global_rank'] = out['local_rank']
    out['prominence_rank'] = out['prominence'].rank(ascending=False, method='min').astype(int)
    out['width_rank'] = out['peak_width_days'].rank(ascending=False, method='min').astype(int)
    primary_centers = set(primary_windows['center_day'].astype(int).tolist()) if not primary_windows.empty else set()
    edge_centers = set(edge_windows['center_day'].astype(int).tolist()) if not edge_windows.empty else set()
    out['retained_as_primary'] = out['peak_day'].astype(int).isin(primary_centers)
    out['retained_as_edge'] = out['peak_day'].astype(int).isin(edge_centers)
    return out[['peak_day', 'score_peak', 'global_rank', 'prominence_rank', 'width_rank', 'retained_as_primary', 'retained_as_edge']]


def movingwindow_parameter_sensitivity(state_valid: np.ndarray, day_index: np.ndarray,
                                       settings: StagePartitionV2Settings) -> pd.DataFrame:
    rows = []
    api_mode = movingwindow_api_mode()
    if api_mode == 'threshold_api':
        for bw in settings.support.movingwindow_bandwidth_grid:
            for ts in settings.support.movingwindow_threshold_scale_grid:
                cfg = type(settings.movingwindow)(**{**settings.movingwindow.__dict__, 'bandwidth': int(bw), 'threshold_scale': ts})
                out = run_movingwindow(state_valid, cfg, day_index=day_index)
                primary = windows_from_movingwindow(out['points'], out['scores'], settings.edge)
                edge = edge_from_movingwindow(out['scores'], primary, settings.edge)
                for _, row in primary.iterrows():
                    rows.append({'api_mode': api_mode, 'bandwidth': bw, 'threshold_scale': ts, 'window_type': 'primary', 'center_day': int(row['center_day']), 'start_day': int(row['start_day']), 'end_day': int(row['end_day'])})
                for _, row in edge.iterrows():
                    rows.append({'api_mode': api_mode, 'bandwidth': bw, 'threshold_scale': ts, 'window_type': 'edge', 'center_day': int(row['center_day']), 'start_day': int(row['start_day']), 'end_day': int(row['end_day'])})
    else:
        for bw in settings.support.movingwindow_bandwidth_grid:
            cfg = type(settings.movingwindow)(**{**settings.movingwindow.__dict__, 'bandwidth': int(bw)})
            out = run_movingwindow(state_valid, cfg, day_index=day_index)
            primary = windows_from_movingwindow(out['points'], out['scores'], settings.edge)
            edge = edge_from_movingwindow(out['scores'], primary, settings.edge)
            for _, row in primary.iterrows():
                rows.append({'api_mode': api_mode, 'bandwidth': bw, 'threshold_scale': None, 'window_type': 'primary', 'center_day': int(row['center_day']), 'start_day': int(row['start_day']), 'end_day': int(row['end_day'])})
            for _, row in edge.iterrows():
                rows.append({'api_mode': api_mode, 'bandwidth': bw, 'threshold_scale': None, 'window_type': 'edge', 'center_day': int(row['center_day']), 'start_day': int(row['start_day']), 'end_day': int(row['end_day'])})
    return pd.DataFrame(rows)


def target_mid_may_window_audit(mw_scores: pd.Series, rw_profile: pd.Series,
                                mw_retention: pd.DataFrame, rw_retention: pd.DataFrame,
                                target_day: int = 45, radius: int = 10) -> pd.DataFrame:
    def nearest(series: pd.Series):
        if series.empty:
            return None, np.nan
        idx = np.asarray(series.index, dtype=int)
        k = int(np.argmin(np.abs(idx - target_day)))
        return int(idx[k]), float(series.iloc[k])

    mw_day, mw_val = nearest(mw_scores)
    rw_day, rw_val = nearest(rw_profile)

    def retain_info(df: pd.DataFrame):
        if df.empty:
            return None, None
        sub = df.loc[np.abs(df['peak_day'].astype(int) - target_day) <= radius]
        if sub.empty:
            return None, None
        sub = sub.sort_values('peak_rank')
        return str(sub.iloc[0]['retained_as']), int(sub.iloc[0]['peak_day'])

    mw_ret, mw_peak = retain_info(mw_retention)
    rw_ret, rw_peak = retain_info(rw_retention)

    return pd.DataFrame([{
        'target_day': target_day,
        'radius': radius,
        'movingwindow_nearest_day': mw_day,
        'movingwindow_score_at_nearest': mw_val,
        'movingwindow_retention_result': mw_ret,
        'movingwindow_retention_peak_day': mw_peak,
        'ruptures_nearest_day': rw_day,
        'ruptures_profile_at_nearest': rw_val,
        'ruptures_retention_result': rw_ret,
        'ruptures_retention_peak_day': rw_peak,
    }])
