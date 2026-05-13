\
from __future__ import annotations

import numpy as np
import pandas as pd

from .index_metadata import VARIABLE_ORDER


def build_yearwise_shape_audit(index_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in VARIABLE_ORDER:
        pivot = index_values.pivot(index="year", columns="day", values=name).sort_index(axis=0).sort_index(axis=1)
        arr = pivot.to_numpy(dtype=float)
        years = pivot.index.to_numpy(dtype=int)

        mu_day = np.nanmean(arr, axis=0)
        dev = arr - mu_day[None, :]
        dx = np.diff(arr, axis=1)
        curvature = arr[:, 2:] - 2.0 * arr[:, 1:-1] + arr[:, :-2]

        offset_mean = np.nanmean(dev, axis=1)
        offset_abs_mean = np.nanmean(np.abs(dev), axis=1)
        roughness_mean_abs = np.nanmean(np.abs(dx), axis=1)
        roughness_max_abs = np.nanmax(np.abs(dx), axis=1)
        curvature_mean_abs = np.nanmean(np.abs(curvature), axis=1)

        q_offset = np.nanquantile(offset_abs_mean, 0.90)
        q_rough = np.nanquantile(roughness_mean_abs, 0.90)
        q_curv = np.nanquantile(curvature_mean_abs, 0.90)

        for i, year in enumerate(years):
            rows.append({
                "index_name": name,
                "year": int(year),
                "offset_mean": float(offset_mean[i]) if np.isfinite(offset_mean[i]) else np.nan,
                "offset_abs_mean": float(offset_abs_mean[i]) if np.isfinite(offset_abs_mean[i]) else np.nan,
                "roughness_mean_abs": float(roughness_mean_abs[i]) if np.isfinite(roughness_mean_abs[i]) else np.nan,
                "roughness_max_abs": float(roughness_max_abs[i]) if np.isfinite(roughness_max_abs[i]) else np.nan,
                "curvature_mean_abs": float(curvature_mean_abs[i]) if np.isfinite(curvature_mean_abs[i]) else np.nan,
                "flag_large_offset": bool(np.isfinite(offset_abs_mean[i]) and offset_abs_mean[i] > q_offset),
                "flag_large_roughness": bool(np.isfinite(roughness_mean_abs[i]) and roughness_mean_abs[i] > q_rough),
                "flag_large_curvature": bool(np.isfinite(curvature_mean_abs[i]) and curvature_mean_abs[i] > q_curv),
            })
    return pd.DataFrame(rows)
