# -*- coding: utf-8 -*-
"""Region and weighted-summary helpers for field-explanation audit."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from .t3_v_to_p_field_explanation_settings import RegionSpec


def region_mask(lat: np.ndarray, lon: np.ndarray, spec: RegionSpec) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("This audit currently expects 1D lat/lon coordinates.")
    lat_mask = (lat >= spec.lat_min) & (lat <= spec.lat_max)
    lon_mask = (lon >= spec.lon_min) & (lon <= spec.lon_max)
    return lat_mask, lon_mask


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
    return float(np.nansum(sub * weights) / denom)


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


def positive_area_fraction(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: RegionSpec) -> float:
    lat_mask, lon_mask = region_mask(lat, lon, spec)
    if not lat_mask.any() or not lon_mask.any():
        return float("nan")
    sub = np.asarray(arr)[np.ix_(lat_mask, lon_mask)]
    valid = np.isfinite(sub)
    if not valid.any():
        return float("nan")
    return float(np.mean(sub[valid] > 0.0))


def sign_consistent_area_fraction(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: RegionSpec) -> float:
    lat_mask, lon_mask = region_mask(lat, lon, spec)
    if not lat_mask.any() or not lon_mask.any():
        return float("nan")
    sub = np.asarray(arr)[np.ix_(lat_mask, lon_mask)]
    valid = np.isfinite(sub)
    if not valid.any():
        return float("nan")
    region_mean = np.nanmean(sub[valid])
    if not np.isfinite(region_mean) or abs(region_mean) <= 0:
        return float("nan")
    return float(np.mean(np.sign(sub[valid]) == np.sign(region_mean)))


def weighted_region_series(field_samples: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: RegionSpec) -> np.ndarray:
    """Return a cos(lat)-weighted regional mean time series from samples x lat x lon."""
    lat_mask, lon_mask = region_mask(lat, lon, spec)
    if not lat_mask.any() or not lon_mask.any():
        return np.full((field_samples.shape[0],), np.nan)
    sub = np.asarray(field_samples)[:, :, :][:, np.ix_(lat_mask, lon_mask)[0], :]
    # np.ix_ with three dims is awkward; use direct slicing for clarity.
    sub = np.asarray(field_samples)[:, lat_mask, :][:, :, lon_mask]
    weights = coslat_weights(lat, lon, lat_mask, lon_mask)
    valid = np.isfinite(sub)
    denom = np.nansum(weights[None, :, :] * valid, axis=(1, 2))
    num = np.nansum(sub * weights[None, :, :] * valid, axis=(1, 2))
    out = np.full((field_samples.shape[0],), np.nan, dtype=float)
    ok = denom > 0
    out[ok] = num[ok] / denom[ok]
    return out


def region_specs_as_dict(regions: Dict[str, RegionSpec]) -> Dict[str, Dict[str, float]]:
    return {name: asdict(spec) for name, spec in regions.items()}
