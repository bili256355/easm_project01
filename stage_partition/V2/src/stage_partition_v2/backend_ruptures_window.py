from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd

from .config import RupturesWindowConfig


def _import_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except Exception as e:
        raise ImportError(
            'ruptures.Window backend requires ruptures in the user environment. '
            'Install the dependencies listed in requirements_stage_partition_v2.txt.'
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


def run_ruptures_window(state_matrix: np.ndarray, cfg: RupturesWindowConfig, day_index: np.ndarray | None = None) -> dict:
    rpt = _import_ruptures()
    algo = rpt.Window(width=cfg.width, model=cfg.model, min_size=cfg.min_size, jump=cfg.jump)
    algo.fit(state_matrix)
    selection_mode = cfg.selection_mode
    if selection_mode == 'fixed_n_bkps':
        if cfg.fixed_n_bkps is None:
            raise ValueError('Ruptures selection_mode=fixed_n_bkps requires fixed_n_bkps to be set.')
        bkps = algo.predict(n_bkps=int(cfg.fixed_n_bkps))
    elif selection_mode == 'pen':
        if cfg.pen is None:
            raise ValueError('Ruptures selection_mode=pen requires pen to be set.')
        bkps = algo.predict(pen=float(cfg.pen))
    elif selection_mode == 'epsilon':
        if cfg.epsilon is None:
            raise ValueError('Ruptures selection_mode=epsilon requires epsilon to be set.')
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
