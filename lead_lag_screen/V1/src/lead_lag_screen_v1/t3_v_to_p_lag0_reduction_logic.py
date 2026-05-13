from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import pandas as pd

V_INDICES = ["V_strength", "V_pos_centroid_lat", "V_NS_diff"]
P_INDICES = [
    "P_main_band_share",
    "P_south_band_share_18_24",
    "P_main_minus_south",
    "P_spread_lat",
    "P_north_band_share_35_45",
    "P_north_minus_main_35_45",
    "P_total_centroid_lat_10_50",
]

FIELD_BY_FAMILY = {"P": "precip", "V": "v850", "H": "z500", "Je": "u200", "Jw": "u200"}


def safe_float(v, default=np.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    x = x[mask]
    y = y[mask]
    xs = x.std(ddof=1)
    ys = y.std(ddof=1)
    if xs <= 0 or ys <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def lagged_arrays(index_df: pd.DataFrame, source: str, target: str, target_days: list[int], lag: int) -> tuple[np.ndarray, np.ndarray]:
    wide = index_df[["year", "day", source, target]].copy()
    src_map = {(int(r.year), int(r.day)): float(getattr(r, source)) for r in wide.itertuples(index=False) if pd.notna(getattr(r, source))}
    tgt_map = {(int(r.year), int(r.day)): float(getattr(r, target)) for r in wide.itertuples(index=False) if pd.notna(getattr(r, target))}
    years = sorted(wide["year"].dropna().astype(int).unique())
    xs: list[float] = []
    ys: list[float] = []
    for year in years:
        for t_day in target_days:
            s_day = int(t_day) - int(lag)
            s_val = src_map.get((year, s_day))
            t_val = tgt_map.get((year, int(t_day)))
            if s_val is None or t_val is None:
                continue
            if math.isfinite(s_val) and math.isfinite(t_val):
                xs.append(s_val)
                ys.append(t_val)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def lag_profile_from_indices(index_df: pd.DataFrame, source: str, target: str, day_range: tuple[int, int], max_lag: int = 5) -> pd.DataFrame:
    days = list(range(int(day_range[0]), int(day_range[1]) + 1))
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        x, y = lagged_arrays(index_df, source, target, days, lag)
        r = pearson_r(x, y)
        rows.append({
            "source_variable": source,
            "target_variable": target,
            "lag": lag,
            "n_pairs": int(np.isfinite(x).sum()),
            "signed_r": r,
            "abs_r": abs(r) if math.isfinite(r) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_lag_profile(profile: pd.DataFrame) -> dict[str, object]:
    if profile.empty:
        return {}
    pos = profile[profile["lag"] > 0].sort_values("abs_r", ascending=False).head(1)
    neg = profile[profile["lag"] < 0].sort_values("abs_r", ascending=False).head(1)
    zero = profile[profile["lag"] == 0].head(1)
    rec = {
        "positive_peak_lag": int(pos["lag"].iloc[0]) if not pos.empty and pd.notna(pos["lag"].iloc[0]) else np.nan,
        "positive_peak_signed_r": float(pos["signed_r"].iloc[0]) if not pos.empty and pd.notna(pos["signed_r"].iloc[0]) else np.nan,
        "positive_peak_abs_r": float(pos["abs_r"].iloc[0]) if not pos.empty and pd.notna(pos["abs_r"].iloc[0]) else np.nan,
        "lag0_signed_r": float(zero["signed_r"].iloc[0]) if not zero.empty and pd.notna(zero["signed_r"].iloc[0]) else np.nan,
        "lag0_abs_r": float(zero["abs_r"].iloc[0]) if not zero.empty and pd.notna(zero["abs_r"].iloc[0]) else np.nan,
        "negative_peak_lag": int(neg["lag"].iloc[0]) if not neg.empty and pd.notna(neg["lag"].iloc[0]) else np.nan,
        "negative_peak_signed_r": float(neg["signed_r"].iloc[0]) if not neg.empty and pd.notna(neg["signed_r"].iloc[0]) else np.nan,
        "negative_peak_abs_r": float(neg["abs_r"].iloc[0]) if not neg.empty and pd.notna(neg["abs_r"].iloc[0]) else np.nan,
    }
    rec["profile_type"] = classify_profile(rec)
    return rec


def classify_profile(row: dict[str, object] | pd.Series) -> str:
    t_pos = safe_float(row.get("positive_peak_abs_r"))
    t0 = safe_float(row.get("lag0_abs_r"))
    tneg = safe_float(row.get("negative_peak_abs_r"))
    vals = [v for v in [t_pos, t0, tneg] if math.isfinite(v)]
    if not vals or max(vals) < 0.20:
        return "weak_all_lags"
    mx = max(vals)
    near = 0.03
    if math.isfinite(t0) and abs(t0 - mx) <= near and math.isfinite(t_pos) and abs(t_pos - t0) <= near:
        return "flat_lag0_positive_close"
    if math.isfinite(t0) and t0 == mx:
        return "lag0_peak"
    if math.isfinite(t_pos) and t_pos == mx and (not math.isfinite(tneg) or t_pos - tneg > near) and (not math.isfinite(t0) or t_pos - t0 > near):
        return "positive_peak_clear"
    if math.isfinite(tneg) and tneg == mx:
        return "reverse_peak"
    return "multi_peak_or_close"


def latlon_mask(lat: np.ndarray, lon: np.ndarray, extent: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = extent
    lat_mask = np.isfinite(lat) & (lat >= lat_min) & (lat <= lat_max)
    lon_mask = np.isfinite(lon) & (lon >= lon_min) & (lon <= lon_max)
    if not np.any(lat_mask):
        raise ValueError(f"No lat grid points in range {lat_min}-{lat_max}")
    if not np.any(lon_mask):
        raise ValueError(f"No lon grid points in range {lon_min}-{lon_max}")
    return lat_mask, lon_mask


def subset_map(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, extent: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_mask, lon_mask = latlon_mask(lat, lon, extent)
    return field[:, :, lat_mask, :][:, :, :, lon_mask], lat[lat_mask], lon[lon_mask]


def sample_positions(index_df: pd.DataFrame, years: np.ndarray, index_name: str, day_range: tuple[int, int], high_q: float, low_q: float) -> tuple[float, float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], int]:
    start, end = day_range
    df = index_df[(index_df["day"] >= start) & (index_df["day"] <= end)][["year", "day", index_name]].copy()
    df = df[np.isfinite(df[index_name].to_numpy(dtype=float))]
    if df.empty:
        return np.nan, np.nan, (np.array([], dtype=int), np.array([], dtype=int)), (np.array([], dtype=int), np.array([], dtype=int)), 0
    high_thr = float(df[index_name].quantile(high_q))
    low_thr = float(df[index_name].quantile(low_q))
    high = df[df[index_name] >= high_thr]
    low = df[df[index_name] <= low_thr]
    year_to_i = {int(y): i for i, y in enumerate(years.astype(int))}

    def rows_to_idx(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        yi: list[int] = []
        di: list[int] = []
        for r in rows.itertuples(index=False):
            y = int(r.year)
            d0 = int(r.day) - 1
            if y in year_to_i and d0 >= 0:
                yi.append(year_to_i[y])
                di.append(d0)
        return np.asarray(yi, dtype=int), np.asarray(di, dtype=int)
    return high_thr, low_thr, rows_to_idx(high), rows_to_idx(low), int(len(df))


def composite_for_index(index_df: pd.DataFrame, fields: dict[str, np.ndarray], index_name: str, field_key: str, day_range: tuple[int, int], high_q: float, low_q: float, extent: tuple[float, float, float, float]) -> dict[str, object]:
    field_full = fields[field_key]
    lat = fields["lat"]
    lon = fields["lon"]
    years = fields["years"].astype(int)
    field, sublat, sublon = subset_map(field_full, lat, lon, extent)
    high_thr, low_thr, high_idx, low_idx, n_total = sample_positions(index_df, years, index_name, day_range, high_q, low_q)

    def take(idx: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        yi, di = idx
        if yi.size == 0:
            return np.empty((0,) + field.shape[2:], dtype=float)
        valid = (yi >= 0) & (yi < field.shape[0]) & (di >= 0) & (di < field.shape[1])
        return field[yi[valid], di[valid], :, :]

    high_samples = take(high_idx)
    low_samples = take(low_idx)
    high_comp = np.nanmean(high_samples, axis=0) if high_samples.shape[0] else np.full(field.shape[2:], np.nan)
    low_comp = np.nanmean(low_samples, axis=0) if low_samples.shape[0] else np.full(field.shape[2:], np.nan)
    diff = high_comp - low_comp
    finite = diff[np.isfinite(diff)]
    zonal_profile = np.nanmean(diff, axis=1)
    meridional_profile = np.nanmean(diff, axis=0)
    return {
        "index_name": index_name,
        "field_key": field_key,
        "lat": sublat,
        "lon": sublon,
        "high": high_comp,
        "low": low_comp,
        "diff": diff,
        "n_total_samples": n_total,
        "n_high_samples": int(high_samples.shape[0]),
        "n_low_samples": int(low_samples.shape[0]),
        "high_threshold": high_thr,
        "low_threshold": low_thr,
        "diff_mean": float(np.nanmean(finite)) if finite.size else np.nan,
        "diff_std": float(np.nanstd(finite)) if finite.size else np.nan,
        "diff_max_abs": float(np.nanmax(np.abs(finite))) if finite.size else np.nan,
        "zonal_profile_max_abs": float(np.nanmax(np.abs(zonal_profile))) if np.isfinite(zonal_profile).any() else np.nan,
        "meridional_profile_max_abs": float(np.nanmax(np.abs(meridional_profile))) if np.isfinite(meridional_profile).any() else np.nan,
    }


def finite_corr_map(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    return pearson_r(av, bv)
