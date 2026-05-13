from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings
from .utils import moving_average_1d, nanmean_no_warn, nanstd_no_warn


def _local_shift_for_matrix(matrix: np.ndarray, half_window: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    n_day = matrix.shape[0]
    score = np.full(n_day, np.nan, dtype=np.float64)
    scale = np.full(n_day, np.nan, dtype=np.float64)
    for t in range(half_window, n_day - half_window):
        left = matrix[t - half_window:t, :]
        right = matrix[t + 1:t + half_window + 1, :]
        if left.size == 0 or right.size == 0:
            continue
        left_mean = nanmean_no_warn(left, axis=0)
        right_mean = nanmean_no_warn(right, axis=0)
        pooled = np.vstack([left, right])
        local_sd = nanstd_no_warn(pooled, axis=0)
        local_scale = float(np.sqrt(np.nanmean(local_sd ** 2))) if np.isfinite(local_sd).any() else np.nan
        diff_norm = float(np.linalg.norm(np.nan_to_num(right_mean - left_mean, nan=0.0)))
        score[t] = diff_norm / (local_scale + eps) if np.isfinite(local_scale) else np.nan
        scale[t] = local_scale
    return score, scale


def build_general_score_curve(state_mean: np.ndarray, settings: StagePartitionV1Settings) -> pd.DataFrame:
    raw, local_scale = _local_shift_for_matrix(np.asarray(state_mean, dtype=np.float64), settings.score.half_window, settings.score.local_scale_eps)
    smooth = moving_average_1d(raw, settings.score.smooth_window)
    return pd.DataFrame({
        'day_index': np.arange(state_mean.shape[0], dtype=int),
        'score_raw': raw,
        'score_smooth': smooth,
        'local_scale': local_scale,
    })


def build_yearwise_score_cube(state_cube: np.ndarray, settings: StagePartitionV1Settings) -> np.ndarray:
    score_cube = np.full((state_cube.shape[0], state_cube.shape[1]), np.nan, dtype=np.float64)
    for y in range(state_cube.shape[0]):
        raw, _scale = _local_shift_for_matrix(state_cube[y], settings.score.half_window, settings.score.local_scale_eps)
        score_cube[y] = moving_average_1d(raw, settings.score.smooth_window)
    return score_cube
