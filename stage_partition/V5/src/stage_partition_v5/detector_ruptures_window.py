from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from .config import RupturesWindowConfig
from .timeline import day_index_to_month_day

def _import_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except Exception as e:
        raise ImportError('V4 requires the ruptures package in the user environment. Install ruptures before running stage_partition/V4.') from e

def _map_breakpoints_to_days(points_local: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None: return points_local.astype(int)
    arr = np.asarray(day_index, dtype=int)
    return pd.Series([int(arr[max(0, min(len(arr)-1, int(p)-1))]) for p in points_local.astype(int).tolist()], name='changepoint', dtype=int)

def _map_profile_index(profile: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None or profile.empty:
        return profile
    arr = np.asarray(day_index, dtype=int)
    mapped_idx = []
    for i in profile.index.to_numpy(dtype=int):
        idx = max(0, min(len(arr) - 1, int(i)))
        mapped_idx.append(int(arr[idx]))
    out = profile.copy()
    out.index = np.asarray(mapped_idx, dtype=int)
    return out

def extract_ranked_local_peaks(profile: pd.Series, min_distance_days: int, prominence_min: float = 0.0) -> pd.DataFrame:
    cols = ['peak_id','peak_day','peak_score','peak_prominence','peak_rank','source_type']
    if profile.empty: return pd.DataFrame(columns=cols)
    s = profile.sort_index().astype(float).fillna(0.0); values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=max(1, int(min_distance_days)))
    if peaks.size == 0: return pd.DataFrame(columns=cols)
    prominences, _, _ = peak_prominences(values, peaks)
    rows = [{'peak_id': f'LP{i:03d}', 'peak_day': int(s.index[int(p)]), 'peak_score': float(values[int(p)]), 'peak_prominence': float(prom), 'source_type': 'local_peak'} for i,(p,prom) in enumerate(zip(peaks,prominences), start=1) if float(prom) >= float(prominence_min)]
    df = pd.DataFrame(rows)
    if df.empty: return pd.DataFrame(columns=cols)
    df = df.sort_values(['peak_score','peak_prominence','peak_day'], ascending=[False,False,True]).reset_index(drop=True); df['peak_rank']=np.arange(1,len(df)+1,dtype=int); return df[cols]

def _nearest_peak_day(local_peaks_df: pd.DataFrame, target_day: int, radius: int) -> int:
    if local_peaks_df is None or local_peaks_df.empty:
        return int(target_day)
    sub = local_peaks_df.copy()
    sub['abs_offset'] = (sub['peak_day'].astype(int) - int(target_day)).abs()
    sub = sub.sort_values(['abs_offset', 'peak_score', 'peak_day'], ascending=[True, False, True]).reset_index(drop=True)
    if sub.empty:
        return int(target_day)
    best = sub.iloc[0]
    if int(best['abs_offset']) <= int(radius):
        return int(best['peak_day'])
    return int(target_day)

def build_primary_points_table(points: pd.Series, profile: pd.Series, *, local_peaks_df: pd.DataFrame | None = None, nearest_peak_search_radius_days: int = 10) -> pd.DataFrame:
    cols = ['point_id','point_day','month_day','peak_score','source_type','raw_point_day','matched_peak_day']
    if points is None or len(points) == 0:
        return pd.DataFrame(columns=cols)
    rows = []
    for i, raw_point_day in enumerate(sorted(int(x) for x in pd.Series(points).astype(int).tolist()), start=1):
        point_day = _nearest_peak_day(local_peaks_df, raw_point_day, nearest_peak_search_radius_days) if local_peaks_df is not None else int(raw_point_day)
        rows.append({
            'point_id': f'RP{i:03d}',
            'point_day': int(point_day),
            'month_day': day_index_to_month_day(int(point_day)),
            'peak_score': float(profile.get(int(point_day), np.nan)) if profile is not None and not profile.empty else np.nan,
            'source_type': 'formal_primary',
            'raw_point_day': int(raw_point_day),
            'matched_peak_day': int(point_day),
        })
    return pd.DataFrame(rows, columns=cols)

def run_ruptures_window(state_matrix: np.ndarray, cfg: RupturesWindowConfig, day_index: np.ndarray | None = None) -> dict:
    rpt = _import_ruptures(); signal = np.asarray(state_matrix, dtype=float)
    algo = rpt.Window(width=cfg.width, model=cfg.model, min_size=cfg.min_size, jump=cfg.jump).fit(signal)
    if cfg.selection_mode == 'fixed_n_bkps':
        if cfg.fixed_n_bkps is None: raise ValueError('fixed_n_bkps must be set when selection_mode=fixed_n_bkps')
        bkps = algo.predict(n_bkps=cfg.fixed_n_bkps)
    elif cfg.selection_mode == 'pen': bkps = algo.predict(pen=cfg.pen)
    elif cfg.selection_mode == 'epsilon': bkps = algo.predict(epsilon=cfg.epsilon)
    else: raise ValueError(f'Unsupported selection_mode={cfg.selection_mode}')
    points_local = pd.Series([int(x) for x in bkps[:-1]], name='changepoint', dtype=int)
    score = getattr(algo, 'score', None)
    if score is None:
        profile_raw = pd.Series(dtype=float, name='profile')
    else:
        arr = np.asarray(score, dtype=float).ravel()
        width_half = int(algo.width // 2)
        idx = np.arange(width_half, width_half + len(arr), dtype=int)
        profile_raw = pd.Series(arr, index=idx, name='profile')
    profile = _map_profile_index(profile_raw, day_index)
    points = _map_breakpoints_to_days(points_local, day_index)
    return {'points': points, 'profile': profile, 'points_local': points_local, 'profile_local': profile_raw}

def run_point_detector(state_matrix: np.ndarray, valid_day_index: np.ndarray, cfg: RupturesWindowConfig, local_peak_distance_days: int | None = None) -> dict:
    out = run_ruptures_window(state_matrix, cfg, day_index=valid_day_index)
    peak_distance = int(local_peak_distance_days if local_peak_distance_days is not None else cfg.local_peak_min_distance_days)
    local_peaks_df = extract_ranked_local_peaks(out['profile'], min_distance_days=peak_distance, prominence_min=0.0)
    primary_points_df = build_primary_points_table(
        out['points'],
        out['profile'],
        local_peaks_df=local_peaks_df,
        nearest_peak_search_radius_days=cfg.nearest_peak_search_radius_days,
    )
    return {**out, 'primary_points_df': primary_points_df, 'local_peaks_df': local_peaks_df}
