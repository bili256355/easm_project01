from __future__ import annotations

import numpy as np
import pandas as pd

from .detector_ruptures_window import extract_ranked_local_peaks


def generate_yearwise_circular_shift_replicate(rng: np.random.Generator, n_years: int, n_days: int) -> np.ndarray:
    return rng.integers(0, n_days, size=n_years, endpoint=False)


def extract_null_global_peak_score(profile: pd.Series, min_distance_days: int, prominence_min: float) -> float:
    peaks_df = extract_ranked_local_peaks(profile, min_distance_days=min_distance_days, prominence_min=prominence_min)
    if peaks_df.empty:
        return 0.0
    return float(peaks_df.iloc[0]['peak_score'])


def extract_null_local_peak_for_point(profile: pd.Series, point_day: int, local_window_days: int, min_distance_days: int, prominence_min: float) -> tuple[float, int | None, bool]:
    peaks_df = extract_ranked_local_peaks(profile, min_distance_days=min_distance_days, prominence_min=prominence_min)
    if peaks_df.empty:
        return 0.0, None, True
    mask = peaks_df['peak_day'].astype(int).between(int(point_day) - int(local_window_days), int(point_day) + int(local_window_days))
    local = peaks_df.loc[mask].copy()
    if local.empty:
        return 0.0, None, True
    local = local.sort_values(['peak_score', 'peak_prominence', 'peak_day'], ascending=[False, False, True]).reset_index(drop=True)
    row = local.iloc[0]
    return float(row['peak_score']), int(row['peak_day']), False


def summarize_null_statistics(observed_peak_score: float, null_global_scores: np.ndarray, null_local_scores: np.ndarray) -> dict:
    null_global_scores = np.asarray(null_global_scores, dtype=float)
    null_local_scores = np.asarray(null_local_scores, dtype=float)
    finite_global = null_global_scores[np.isfinite(null_global_scores)]
    finite_local = null_local_scores[np.isfinite(null_local_scores)]
    gmed = float(np.nanmedian(finite_global)) if finite_global.size else np.nan
    gp95 = float(np.nanquantile(finite_global, 0.95)) if finite_global.size else np.nan
    lmed = float(np.nanmedian(finite_local)) if finite_local.size else np.nan
    lp95 = float(np.nanquantile(finite_local, 0.95)) if finite_local.size else np.nan
    ratio_g = float(observed_peak_score / gmed) if np.isfinite(observed_peak_score) and np.isfinite(gmed) and gmed > 0 else np.nan
    ratio_l = float(observed_peak_score / lmed) if np.isfinite(observed_peak_score) and np.isfinite(lmed) and lmed > 0 else np.nan
    scale_warn = bool((np.isfinite(ratio_l) and ratio_l < 0.4) or (np.isfinite(ratio_g) and ratio_g < 0.15))
    return {
        'null_global_peak_median': gmed,
        'null_global_peak_p95': gp95,
        'null_local_peak_median': lmed,
        'null_local_peak_p95': lp95,
        'score_scale_ratio_global': ratio_g,
        'score_scale_ratio_local': ratio_l,
        'scale_warning_flag': scale_warn,
        'local_peak_exists_fraction': float(np.mean(finite_local > 0)) if finite_local.size else np.nan,
    }
