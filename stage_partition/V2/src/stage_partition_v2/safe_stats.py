from __future__ import annotations

import numpy as np


def safe_nanmean(arr: np.ndarray, axis, return_valid_count: bool = False):
    """Mean ignoring NaN without triggering RuntimeWarning on all-NaN slices.

    All-NaN slices return np.nan and can optionally return the finite-count tensor.
    """
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    count = np.sum(finite, axis=axis)
    total = np.nansum(arr, axis=axis)
    mean = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
    if return_valid_count:
        return mean, count
    return mean


def safe_daily_energy(arr: np.ndarray) -> dict[str, float]:
    """Compute simple energy summaries on valid values only, without warnings."""
    arr = np.asarray(arr, dtype=float)
    sq = np.square(arr)
    day_mean = safe_nanmean(sq, axis=1)
    finite_day = np.isfinite(day_mean)
    if np.any(finite_day):
        daily_energy_mean = float(np.mean(day_mean[finite_day]))
    else:
        daily_energy_mean = np.nan
    finite_sq = np.isfinite(sq)
    if np.any(finite_sq):
        rms = float(np.sqrt(np.mean(sq[finite_sq])))
    else:
        rms = np.nan
    abs_vals = np.abs(arr[np.isfinite(arr)])
    p95_abs = float(np.quantile(abs_vals, 0.95)) if abs_vals.size > 0 else np.nan
    return {
        'rms': rms,
        'daily_energy_mean': daily_energy_mean,
        'p95_abs': p95_abs,
    }


def build_all_nan_mask(arr: np.ndarray, axis) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return np.sum(np.isfinite(arr), axis=axis) == 0
