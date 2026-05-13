\
from __future__ import annotations

import numpy as np
import pandas as pd

from .index_metadata import VARIABLE_ORDER


def anomaly_reconstruction_check(values: pd.DataFrame, clim: pd.DataFrame, anom: pd.DataFrame, tolerance: float = 1e-8) -> pd.DataFrame:
    clim_by_day = clim.set_index("day")
    value_idx = values.set_index(["year", "day"])
    anom_idx = anom.set_index(["year", "day"])
    rows = []
    for name in VARIABLE_ORDER:
        day_values = value_idx.index.get_level_values("day")
        reconstructed = value_idx[name].to_numpy(dtype=float) - clim_by_day.loc[day_values, name].to_numpy(dtype=float)
        actual = anom_idx[name].reindex(value_idx.index).to_numpy(dtype=float)
        diff = actual - reconstructed
        rows.append({
            "index_name": name,
            "max_abs_error": float(np.nanmax(np.abs(diff))) if np.any(np.isfinite(diff)) else np.nan,
            "mean_abs_error": float(np.nanmean(np.abs(diff))) if np.any(np.isfinite(diff)) else np.nan,
            "n_error_above_tolerance": int(np.nansum(np.abs(diff) > tolerance)),
            "status": "pass" if int(np.nansum(np.abs(diff) > tolerance)) == 0 else "warning",
        })
    return pd.DataFrame(rows)


def anomaly_daily_mean_check(anom: pd.DataFrame, tolerance: float = 1e-8) -> pd.DataFrame:
    rows = []
    for name in VARIABLE_ORDER:
        daily_mean = anom.groupby("day")[name].mean()
        vals = daily_mean.to_numpy(dtype=float)
        max_abs = float(np.nanmax(np.abs(vals))) if np.any(np.isfinite(vals)) else np.nan
        rows.append({
            "index_name": name,
            "max_abs_daily_mean": max_abs,
            "mean_abs_daily_mean": float(np.nanmean(np.abs(vals))) if np.any(np.isfinite(vals)) else np.nan,
            "rms_daily_mean": float(np.sqrt(np.nanmean(vals * vals))) if np.any(np.isfinite(vals)) else np.nan,
            "status": "pass" if np.isfinite(max_abs) and max_abs <= tolerance else "warning",
        })
    return pd.DataFrame(rows)
