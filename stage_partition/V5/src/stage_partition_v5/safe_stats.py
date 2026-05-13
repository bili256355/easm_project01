from __future__ import annotations
import numpy as np

def safe_nanmean(x: np.ndarray, axis=None, return_valid_count: bool = False):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    count = np.sum(valid, axis=axis)
    total = np.nansum(np.where(valid, x, np.nan), axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = total / np.where(count == 0, np.nan, count)
    return (mean, count) if return_valid_count else mean

def build_all_nan_mask(x: np.ndarray, axis=0) -> np.ndarray:
    return ~np.any(np.isfinite(x), axis=axis)

def safe_daily_energy(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return {'rms': np.nan, 'daily_energy_mean': np.nan, 'p95_abs': np.nan}
    rms = float(np.sqrt(np.nanmean(np.square(x))))
    sq = np.square(x)
    daily_energy = safe_nanmean(sq, axis=1) if x.ndim >= 2 else sq
    mean = float(np.nanmean(daily_energy)) if np.isfinite(daily_energy).any() else np.nan
    p95 = float(np.nanquantile(np.abs(x[finite]), 0.95)) if np.any(finite) else np.nan
    return {'rms': rms, 'daily_energy_mean': mean, 'p95_abs': p95}
