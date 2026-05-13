from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from .config import RupturesWindowConfig


def _import_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except Exception as e:
        raise ImportError(
            'V3 requires the ruptures package in the user environment. Install ruptures before running stage_partition/V3.'
        ) from e


def _map_breakpoints_to_days(points_local: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None:
        return points_local.astype(int)
    arr = np.asarray(day_index, dtype=int)
    mapped = []
    for p in points_local.astype(int).tolist():
        idx = max(0, min(len(arr) - 1, p - 1))
        mapped.append(int(arr[idx]))
    return pd.Series(mapped, name='changepoint', dtype=int)


def _map_profile_index(profile: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None or profile.empty:
        return profile
    arr = np.asarray(day_index, dtype=int)
    mapped_idx = []
    for i in profile.index.to_numpy(dtype=int):
        idx = max(0, min(len(arr) - 1, i))
        mapped_idx.append(int(arr[idx]))
    out = profile.copy()
    out.index = np.asarray(mapped_idx, dtype=int)
    return out


def extract_ranked_local_peaks(profile: pd.Series, min_distance_days: int, prominence_min: float = 0.0) -> pd.DataFrame:
    cols = [
        'peak_id', 'peak_day', 'peak_score', 'peak_prominence', 'peak_rank',
        'left_support_day', 'right_support_day',
    ]
    if profile.empty:
        return pd.DataFrame(columns=cols)
    s = profile.sort_index().astype(float).fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=max(1, int(min_distance_days)))
    if peaks.size == 0:
        return pd.DataFrame(columns=cols)
    prominences, left_bases, right_bases = peak_prominences(values, peaks)
    rows = []
    for i, (p, prom, lb, rb) in enumerate(zip(peaks, prominences, left_bases, right_bases), start=1):
        if float(prom) < float(prominence_min):
            continue
        rows.append({
            'peak_id': f'LP{i:03d}',
            'peak_day': int(s.index[int(p)]),
            'peak_score': float(values[int(p)]),
            'peak_prominence': float(prom),
            'left_support_day': int(s.index[int(lb)]),
            'right_support_day': int(s.index[int(rb)]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.sort_values(['peak_score', 'peak_prominence', 'peak_day'], ascending=[False, False, True]).reset_index(drop=True)
    df['peak_rank'] = np.arange(1, len(df) + 1, dtype=int)
    return df[cols]


def run_ruptures_window(state_matrix: np.ndarray, cfg: RupturesWindowConfig, day_index: np.ndarray | None = None) -> dict:
    rpt = _import_ruptures()
    algo = rpt.Window(width=cfg.width, model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
    algo.fit(state_matrix)
    selection_mode = cfg.selection_mode
    if selection_mode == 'fixed_n_bkps':
        if cfg.fixed_n_bkps is None:
            raise ValueError('selection_mode=fixed_n_bkps requires fixed_n_bkps.')
        bkps = algo.predict(n_bkps=int(cfg.fixed_n_bkps))
    elif selection_mode == 'pen':
        if cfg.pen is None:
            raise ValueError('selection_mode=pen requires pen.')
        bkps = algo.predict(pen=float(cfg.pen))
    elif selection_mode == 'epsilon':
        if cfg.epsilon is None:
            raise ValueError('selection_mode=epsilon requires epsilon.')
        bkps = algo.predict(epsilon=float(cfg.epsilon))
    else:
        raise ValueError(f'Unknown ruptures selection_mode: {selection_mode}')

    points_local = pd.Series([int(x) for x in bkps[:-1]], name='changepoint', dtype=int)
    points = _map_breakpoints_to_days(points_local, day_index)
    score = getattr(algo, 'score', None)
    if score is None:
        profile = pd.Series(dtype=float, name='profile')
    else:
        arr = np.asarray(score, dtype=float).ravel()
        width_half = int(algo.width // 2)
        idx = np.arange(width_half, width_half + len(arr))
        profile = pd.Series(arr, index=idx, name='profile')
        profile = _map_profile_index(profile, day_index)
    return {
        'backend_name': 'ruptures_window',
        'detector': algo,
        'points': points,
        'points_local': points_local,
        'profile': profile,
        'config': asdict(cfg),
    }
