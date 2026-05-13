\
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .index_metadata import INDEX_METADATA, REGIONS, VARIABLE_ORDER


def mask_between(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr) & (arr >= lower) & (arr <= upper)


def subset_field(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]):
    lat_mask = mask_between(lat, *lat_range)
    lon_mask = mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"Latitude range {lat_range} does not hit any grid points.")
    if not np.any(lon_mask):
        raise ValueError(f"Longitude range {lon_range} does not hit any grid points.")
    return field[:, :, lat_mask, :][:, :, :, lon_mask], lat[lat_mask], lon[lon_mask]


def _sample_indices(index_values: pd.DataFrame, years: np.ndarray, index_name: str, high_q: float, low_q: float):
    data = index_values[["year", "day", index_name]].copy()
    data = data[np.isfinite(data[index_name].to_numpy(dtype=float))]
    high_thr = float(data[index_name].quantile(high_q))
    low_thr = float(data[index_name].quantile(low_q))
    high = data[data[index_name] >= high_thr]
    low = data[data[index_name] <= low_thr]

    year_to_i = {int(y): i for i, y in enumerate(years.astype(int))}

    def rows_to_indices(df: pd.DataFrame):
        yi = []
        di = []
        for _, row in df.iterrows():
            y = int(row["year"])
            d = int(row["day"]) - 1
            if y in year_to_i and d >= 0:
                yi.append(year_to_i[y])
                di.append(d)
        return np.asarray(yi, dtype=int), np.asarray(di, dtype=int)

    return high_thr, low_thr, rows_to_indices(high), rows_to_indices(low), len(data)


def _take_samples(field: np.ndarray, sample_idx: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    yi, di = sample_idx
    if yi.size == 0:
        return np.empty((0,) + field.shape[2:], dtype=float)
    return field[yi, di, :, :]


def build_physical_composites(index_values: pd.DataFrame, fields: Dict[str, np.ndarray], high_q: float = 0.80, low_q: float = 0.20):
    lat = fields["lat"]
    lon = fields["lon"]
    years = fields["years"].astype(int)

    composites: Dict[str, Dict[str, object]] = {}
    sample_rows = []

    for name in VARIABLE_ORDER:
        meta = INDEX_METADATA[name]
        family = meta["family"]
        region = REGIONS[family]
        field_name = meta["field"]
        field = fields[field_name]

        high_thr, low_thr, high_idx, low_idx, n_total = _sample_indices(index_values, years, name, high_q, low_q)

        subfield, sublat, sublon = subset_field(field, lat, lon, region["lat_range"], region["lon_range"])
        high_samples = _take_samples(subfield, high_idx)
        low_samples = _take_samples(subfield, low_idx)

        high_comp = np.nanmean(high_samples, axis=0) if high_samples.shape[0] else np.full(subfield.shape[2:], np.nan)
        low_comp = np.nanmean(low_samples, axis=0) if low_samples.shape[0] else np.full(subfield.shape[2:], np.nan)
        diff = high_comp - low_comp

        high_profile = np.nanmean(high_comp, axis=1)
        low_profile = np.nanmean(low_comp, axis=1)
        diff_profile = high_profile - low_profile

        composites[name] = {
            "meta": meta,
            "field_name": field_name,
            "family": family,
            "lat": sublat,
            "lon": sublon,
            "high": high_comp,
            "low": low_comp,
            "diff": diff,
            "high_profile": high_profile,
            "low_profile": low_profile,
            "diff_profile": diff_profile,
            "high_threshold": high_thr,
            "low_threshold": low_thr,
            "n_total_samples": int(n_total),
            "n_high_samples": int(high_idx[0].size),
            "n_low_samples": int(low_idx[0].size),
        }

        sample_rows.append({
            "index_name": name,
            "family": family,
            "field_name": field_name,
            "n_total_samples": int(n_total),
            "n_high_samples": int(high_idx[0].size),
            "n_low_samples": int(low_idx[0].size),
            "high_threshold": high_thr,
            "low_threshold": low_thr,
        })

    sample_info = pd.DataFrame(sample_rows)
    return composites, sample_info


def build_physical_summary(composites: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for name, obj in composites.items():
        diff = np.asarray(obj["diff"], dtype=float)
        finite = diff[np.isfinite(diff)]
        rows.append({
            "index_name": name,
            "family": obj["family"],
            "field_name": obj["field_name"],
            "main_plot_type": obj["meta"]["physical_check_type"],
            "high_threshold": obj["high_threshold"],
            "low_threshold": obj["low_threshold"],
            "n_high_samples": obj["n_high_samples"],
            "n_low_samples": obj["n_low_samples"],
            "expected_meaning": obj["meta"]["expected_meaning"],
            "contrast_strength_std_diff": float(np.nanstd(finite)) if finite.size else np.nan,
            "contrast_strength_max_abs_diff": float(np.nanmax(np.abs(finite))) if finite.size else np.nan,
            "direction_consistency": "to_review",
            "structure_clarity": "to_review",
            "contrast_strength": "to_review",
            "overall_grade": "to_review",
            "short_comment": "Review physical composite figures.",
        })
    return pd.DataFrame(rows)
