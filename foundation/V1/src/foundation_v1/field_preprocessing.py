from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _rolling_sum_and_count(field: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(field)
    safe_values = np.where(finite_mask, field, 0.0)

    cumulative_sum = np.cumsum(safe_values, axis=1, dtype=np.float64)
    cumulative_count = np.cumsum(finite_mask.astype(np.int32), axis=1, dtype=np.int32)

    zero_sum = np.zeros((field.shape[0], 1, field.shape[2], field.shape[3]), dtype=np.float64)
    zero_count = np.zeros((field.shape[0], 1, field.shape[2], field.shape[3]), dtype=np.int32)

    cumulative_sum = np.concatenate([zero_sum, cumulative_sum], axis=1)
    cumulative_count = np.concatenate([zero_count, cumulative_count], axis=1)

    window_sum = cumulative_sum[:, window:, :, :] - cumulative_sum[:, :-window, :, :]
    window_count = cumulative_count[:, window:, :, :] - cumulative_count[:, :-window, :, :]
    return window_sum, window_count


def smooth_field(field: np.ndarray, window: int = 9) -> np.ndarray:
    if field.ndim != 4:
        raise ValueError(f"待平滑字段必须为四维，当前 shape={field.shape}")
    if window % 2 == 0:
        raise ValueError("平滑窗口必须为奇数。")

    field = np.asarray(field, dtype=np.float64)
    n_year, n_day, n_lat, n_lon = field.shape
    if window > n_day:
        raise ValueError(f"平滑窗口 {window} 大于 day 维长度 {n_day}。")

    half = window // 2
    out = np.full((n_year, n_day, n_lat, n_lon), np.nan, dtype=np.float64)
    window_sum, window_count = _rolling_sum_and_count(field, window=window)
    window_mean = np.divide(
        window_sum,
        window_count,
        out=np.full_like(window_sum, np.nan, dtype=np.float64),
        where=window_count > 0,
    )
    out[:, half : n_day - half, :, :] = window_mean
    return out


def smooth_all_fields(fields: Dict[str, np.ndarray], window: int = 9) -> Dict[str, np.ndarray]:
    return {name: smooth_field(arr, window=window) for name, arr in fields.items()}


def compute_daily_climatology(smoothed_field: np.ndarray) -> np.ndarray:
    if smoothed_field.ndim != 4:
        raise ValueError(f"smoothed_field 必须为四维，当前 shape={smoothed_field.shape}")
    valid_count = np.sum(np.isfinite(smoothed_field), axis=0)
    summed = np.nansum(smoothed_field, axis=0)
    climatology = np.full(summed.shape, np.nan, dtype=np.float64)
    np.divide(summed, valid_count, out=climatology, where=valid_count > 0)
    return climatology


def compute_all_daily_climatology(smoothed_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: compute_daily_climatology(arr) for name, arr in smoothed_fields.items()}


def compute_anomaly(smoothed_field: np.ndarray, daily_climatology: np.ndarray) -> np.ndarray:
    if smoothed_field.ndim != 4:
        raise ValueError(f"smoothed_field 必须为四维，当前 shape={smoothed_field.shape}")
    if daily_climatology.ndim != 3:
        raise ValueError(f"daily_climatology 必须为三维，当前 shape={daily_climatology.shape}")
    return smoothed_field - daily_climatology[None, :, :, :]


def compute_all_anomalies(
    smoothed_fields: Dict[str, np.ndarray],
    daily_climatology: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        name: compute_anomaly(smoothed_fields[name], daily_climatology[name])
        for name in smoothed_fields
    }


def _nan_fraction(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.isnan(arr).sum() / arr.size)


def build_nan_report(
    raw_fields: Dict[str, np.ndarray],
    smoothed_fields: Dict[str, np.ndarray],
    climatology_fields: Dict[str, np.ndarray],
    anomaly_fields: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for field_name in raw_fields:
        rows.append(
            {
                "field": field_name,
                "raw_nan_fraction": _nan_fraction(np.asarray(raw_fields[field_name], dtype=np.float64)),
                "smoothed_nan_fraction": _nan_fraction(np.asarray(smoothed_fields[field_name], dtype=np.float64)),
                "clim_nan_fraction": _nan_fraction(np.asarray(climatology_fields[field_name], dtype=np.float64)),
                "anom_nan_fraction": _nan_fraction(np.asarray(anomaly_fields[field_name], dtype=np.float64)),
            }
        )
    return pd.DataFrame(rows)


def build_field_stats(field_groups: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for stage_name, fields in field_groups.items():
        for field_name, arr in fields.items():
            values = np.asarray(arr, dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                rows.append(
                    {
                        "field": field_name,
                        "stage": stage_name,
                        "min": np.nan,
                        "max": np.nan,
                        "mean": np.nan,
                        "std": np.nan,
                        "finite_count": 0,
                    }
                )
                continue
            rows.append(
                {
                    "field": field_name,
                    "stage": stage_name,
                    "min": float(finite_values.min()),
                    "max": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "std": float(finite_values.std()),
                    "finite_count": int(finite_values.size),
                }
            )
    return pd.DataFrame(rows)
