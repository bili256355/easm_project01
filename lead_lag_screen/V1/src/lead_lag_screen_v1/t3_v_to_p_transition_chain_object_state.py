# -*- coding: utf-8 -*-
"""Object-state and object-change computations for transition-chain report."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .t3_v_to_p_transition_chain_settings import RegionSpec, TransitionChainReportSettings


def region_mask(lat: np.ndarray, lon: np.ndarray, spec: RegionSpec) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Expected 1D lat/lon arrays.")
    return (lat >= spec.lat_min) & (lat <= spec.lat_max), (lon >= spec.lon_min) & (lon <= spec.lon_max)


def coslat_weights(lat: np.ndarray, lon: np.ndarray, lat_mask: np.ndarray, lon_mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(np.asarray(lat)[lat_mask]))[:, None]
    weights = np.where(np.isfinite(weights), weights, 0.0)
    return weights * np.ones((1, int(lon_mask.sum())))


def weighted_region_mean_2d(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: RegionSpec) -> float:
    lat_mask, lon_mask = region_mask(lat, lon, spec)
    if not lat_mask.any() or not lon_mask.any():
        return float("nan")
    sub = np.asarray(arr)[np.ix_(lat_mask, lon_mask)]
    weights = coslat_weights(lat, lon, lat_mask, lon_mask)
    valid = np.isfinite(sub)
    denom = np.nansum(weights * valid)
    if denom <= 0:
        return float("nan")
    return float(np.nansum(sub * weights * valid) / denom)


def weighted_region_sum_2d(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: RegionSpec) -> float:
    lat_mask, lon_mask = region_mask(lat, lon, spec)
    if not lat_mask.any() or not lon_mask.any():
        return float("nan")
    sub = np.asarray(arr)[np.ix_(lat_mask, lon_mask)]
    weights = coslat_weights(lat, lon, lat_mask, lon_mask)
    valid = np.isfinite(sub)
    if not valid.any():
        return float("nan")
    return float(np.nansum(sub * weights * valid))


def fraction_condition_2d(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: RegionSpec, condition: str) -> float:
    lat_mask, lon_mask = region_mask(lat, lon, spec)
    if not lat_mask.any() or not lon_mask.any():
        return float("nan")
    sub = np.asarray(arr)[np.ix_(lat_mask, lon_mask)]
    valid = np.isfinite(sub)
    if not valid.any():
        return float("nan")
    if condition == "positive":
        return float(np.mean(sub[valid] > 0))
    if condition == "negative":
        return float(np.mean(sub[valid] < 0))
    raise ValueError(condition)


def window_mean_map(field: np.ndarray, window: Tuple[int, int], day_to_field_i: Dict[int, int]) -> np.ndarray:
    start, end = window
    day_indices = [day_to_field_i[d] for d in range(start, end + 1) if d in day_to_field_i]
    if not day_indices:
        raise ValueError(f"No mapped days for window {window}")
    return np.nanmean(field[:, day_indices, :, :], axis=(0, 1))


def compute_object_mean_maps(
    precip: np.ndarray,
    v850: np.ndarray,
    settings: TransitionChainReportSettings,
    day_to_field_i: Dict[int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    p_maps = {}
    v_maps = {}
    for w in settings.window_order:
        p_maps[w] = window_mean_map(precip, settings.windows[w], day_to_field_i)
        v_maps[w] = window_mean_map(v850, settings.windows[w], day_to_field_i)
    return p_maps, v_maps


def build_object_state_region_summary(
    p_maps: Dict[str, np.ndarray],
    v_maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    settings: TransitionChainReportSettings,
) -> pd.DataFrame:
    rows: List[dict] = []
    for w in settings.window_order:
        for region, spec in settings.regions.items():
            p = p_maps[w]
            v = v_maps[w]
            lat_mask, lon_mask = region_mask(lat, lon, spec)
            rows.append({
                "window": w,
                "region": region,
                "P_mean": weighted_region_mean_2d(p, lat, lon, spec),
                "P_integrated": weighted_region_sum_2d(p, lat, lon, spec),
                "V850_mean": weighted_region_mean_2d(v, lat, lon, spec),
                "V850_abs_mean": weighted_region_mean_2d(np.abs(v), lat, lon, spec),
                "V850_positive_fraction": fraction_condition_2d(v, lat, lon, spec, "positive"),
                "V850_negative_fraction": fraction_condition_2d(v, lat, lon, spec, "negative"),
                "n_valid_grid": int(lat_mask.sum() * lon_mask.sum()),
            })
    return pd.DataFrame(rows)


def _delta_pct(target: float, reference: float) -> float:
    if not np.isfinite(target) or not np.isfinite(reference) or abs(reference) <= 1.0e-12:
        return float("nan")
    return float((target - reference) / abs(reference) * 100.0)


def build_object_change_region_delta(state_df: pd.DataFrame, settings: TransitionChainReportSettings) -> pd.DataFrame:
    rows: List[dict] = []
    key = state_df.set_index(["window", "region"])
    comparisons = dict(settings.comparisons)
    # Special comparison: T3_full minus max(T3_early,T3_late), per metric.
    for comparison, pair in comparisons.items():
        target, reference = pair
        for region in settings.regions.keys():
            t = key.loc[(target, region)]
            r = key.loc[(reference, region)]
            rows.append({
                "comparison": comparison,
                "target_window": target,
                "reference_window": reference,
                "region": region,
                "P_delta": float(t["P_mean"] - r["P_mean"]),
                "P_delta_percent": _delta_pct(float(t["P_mean"]), float(r["P_mean"])),
                "P_integrated_delta": float(t["P_integrated"] - r["P_integrated"]),
                "V850_delta": float(t["V850_mean"] - r["V850_mean"]),
                "V850_delta_percent": _delta_pct(float(t["V850_mean"]), float(r["V850_mean"])),
                "V850_abs_delta": float(t["V850_abs_mean"] - r["V850_abs_mean"]),
            })
    # T3_full minus max subwindow (uses the larger of early/late for each metric)
    for region in settings.regions.keys():
        full = key.loc[("T3_full", region)]
        early = key.loc[("T3_early", region)]
        late = key.loc[("T3_late", region)]
        p_ref = max(float(early["P_mean"]), float(late["P_mean"]))
        v_ref = max(float(early["V850_mean"]), float(late["V850_mean"]))
        vabs_ref = max(float(early["V850_abs_mean"]), float(late["V850_abs_mean"]))
        rows.append({
            "comparison": "T3_full_minus_max_subwindow",
            "target_window": "T3_full",
            "reference_window": "max(T3_early,T3_late)",
            "region": region,
            "P_delta": float(full["P_mean"] - p_ref),
            "P_delta_percent": _delta_pct(float(full["P_mean"]), p_ref),
            "P_integrated_delta": float("nan"),
            "V850_delta": float(full["V850_mean"] - v_ref),
            "V850_delta_percent": _delta_pct(float(full["V850_mean"]), v_ref),
            "V850_abs_delta": float(full["V850_abs_mean"] - vabs_ref),
        })
    return pd.DataFrame(rows)


def build_object_delta_maps(
    p_maps: Dict[str, np.ndarray],
    v_maps: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    p_delta = {
        "T3_full_minus_S3": p_maps["T3_full"] - p_maps["S3"],
        "T3_late_minus_T3_early": p_maps["T3_late"] - p_maps["T3_early"],
        "S4_minus_T3_full": p_maps["S4"] - p_maps["T3_full"],
    }
    v_delta = {
        "T3_full_minus_S3": v_maps["T3_full"] - v_maps["S3"],
        "T3_late_minus_T3_early": v_maps["T3_late"] - v_maps["T3_early"],
        "S4_minus_T3_full": v_maps["S4"] - v_maps["T3_full"],
    }
    return p_delta, v_delta
