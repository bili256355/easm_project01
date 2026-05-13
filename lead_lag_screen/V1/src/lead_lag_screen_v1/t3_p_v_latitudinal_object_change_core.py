# -*- coding: utf-8 -*-
"""Core computations for P/V850 latitudinal object-change audit v1_a."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .t3_p_v_latitudinal_object_change_settings import (
    BoxSpec,
    LonSectorSpec,
    PVLatitudinalObjectChangeSettings,
)


def finite_nanmean(arr: np.ndarray, axis=None) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=axis)




def safe_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate y over x with compatibility across NumPy versions."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.size == 0:
        return float("nan")
    if y_arr.size == 1 or x_arr.size == 1:
        return float(y_arr[0])
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_arr, x_arr))
    if hasattr(np, "trapz"):
        return float(np.trapz(y_arr, x_arr))
    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    dx = np.diff(x_sorted)
    return float(np.nansum(0.5 * (y_sorted[:-1] + y_sorted[1:]) * dx))


def mask_box(lat: np.ndarray, lon: np.ndarray, box: BoxSpec) -> np.ndarray:
    lat_mask = (lat >= box.lat_min) & (lat <= box.lat_max)
    lon_mask = (lon >= box.lon_min) & (lon <= box.lon_max)
    return lat_mask[:, None] & lon_mask[None, :]


def mask_sector(lon: np.ndarray, sector: LonSectorSpec) -> np.ndarray:
    return (lon >= sector.lon_min) & (lon <= sector.lon_max)


def sorted_lat_profile(lat: np.ndarray, profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(lat, dtype=float))
    return np.asarray(lat, dtype=float)[order], np.asarray(profile, dtype=float)[order]


def maybe_smooth_profile(profile: np.ndarray, n: int) -> np.ndarray:
    prof = np.asarray(profile, dtype=float)
    if n <= 1 or prof.size < n:
        return prof.copy()
    if n % 2 == 0:
        n += 1
    pad = n // 2
    padded = np.pad(prof, (pad, pad), mode="edge")
    kernel = np.ones(n, dtype=float) / float(n)
    return np.convolve(padded, kernel, mode="valid")


def window_day_indices(day_map: Dict[int, int], start: int, end: int) -> List[int]:
    out = []
    missing = []
    for d in range(int(start), int(end) + 1):
        if d in day_map:
            out.append(day_map[d])
        else:
            missing.append(d)
    if missing:
        raise KeyError(f"Missing day(s) in field day coordinate: {missing}")
    return out


def compute_window_mean_maps(
    field: np.ndarray,
    windows: Dict[str, Tuple[int, int]],
    day_map: Dict[int, int],
) -> Dict[str, np.ndarray]:
    # field dims: year, day, lat, lon
    out: Dict[str, np.ndarray] = {}
    for w, (start, end) in windows.items():
        days = window_day_indices(day_map, start, end)
        out[w] = finite_nanmean(field[:, days, :, :], axis=(0, 1))
    return out


def summarize_regions(
    p_maps: Dict[str, np.ndarray],
    v_maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    settings: PVLatitudinalObjectChangeSettings,
) -> pd.DataFrame:
    rows = []
    for window in settings.window_order:
        p_map = p_maps[window]
        v_map = v_maps[window]
        for region_name, box in settings.regions.items():
            m = mask_box(lat, lon, box)
            p_vals = p_map[m]
            v_vals = v_map[m]
            rows.append({
                "window": window,
                "region": region_name,
                "lat_min": box.lat_min,
                "lat_max": box.lat_max,
                "lon_min": box.lon_min,
                "lon_max": box.lon_max,
                "P_mean": float(np.nanmean(p_vals)),
                "P_integrated": float(np.nansum(p_vals)),
                "V850_mean": float(np.nanmean(v_vals)),
                "V850_abs_mean": float(np.nanmean(np.abs(v_vals))),
                "V850_positive_mean": float(np.nanmean(np.maximum(v_vals, 0.0))),
                "V850_negative_mean": float(np.nanmean(np.minimum(v_vals, 0.0))),
                "V850_positive_fraction": float(np.nanmean(v_vals > 0.0)),
                "V850_negative_fraction": float(np.nanmean(v_vals < 0.0)),
                "n_valid_grid": int(np.sum(np.isfinite(p_vals) & np.isfinite(v_vals))),
            })
    return pd.DataFrame(rows)


def direction(delta: float, eps: float) -> str:
    if delta > eps:
        return "increase"
    if delta < -eps:
        return "decrease"
    return "near_zero"


def build_object_change_region_delta(
    state_df: pd.DataFrame,
    settings: PVLatitudinalObjectChangeSettings,
) -> pd.DataFrame:
    rows = []
    idx = {(r.window, r.region): r for r in state_df.itertuples(index=False)}
    for comp, (target, ref) in settings.comparisons.items():
        for region in settings.regions.keys():
            a = idx[(target, region)]
            b = idx[(ref, region)]
            p_delta = float(a.P_mean - b.P_mean)
            v_delta = float(a.V850_mean - b.V850_mean)
            v_abs_delta = float(a.V850_abs_mean - b.V850_abs_mean)
            rows.append({
                "comparison": comp,
                "target_window": target,
                "reference_window": ref,
                "region": region,
                "P_delta": p_delta,
                "P_delta_percent": float(p_delta / b.P_mean * 100.0) if abs(b.P_mean) > 1e-12 else np.nan,
                "P_direction": direction(p_delta, settings.p_delta_epsilon),
                "V850_delta": v_delta,
                "V850_delta_percent": float(v_delta / b.V850_mean * 100.0) if abs(b.V850_mean) > 1e-12 else np.nan,
                "V850_direction": direction(v_delta, settings.v_delta_epsilon),
                "V850_abs_delta": v_abs_delta,
                "V850_abs_delta_percent": float(v_abs_delta / b.V850_abs_mean * 100.0) if abs(b.V850_abs_mean) > 1e-12 else np.nan,
            })
    return pd.DataFrame(rows)


def compute_lat_profiles(
    maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    sectors: Dict[str, LonSectorSpec],
    window_order: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for window in window_order:
        mp = maps[window]
        for sector_name, sec in sectors.items():
            lm = mask_sector(lon, sec)
            if not np.any(lm):
                continue
            prof = finite_nanmean(mp[:, lm], axis=1)
            lat_sorted, prof_sorted = sorted_lat_profile(lat, prof)
            for la, val in zip(lat_sorted, prof_sorted):
                rows.append({
                    "window": window,
                    "sector": sector_name,
                    "lat": float(la),
                    "value": float(val),
                })
    return pd.DataFrame(rows)


def build_profile_delta(profile_df: pd.DataFrame, settings: PVLatitudinalObjectChangeSettings, value_name: str) -> pd.DataFrame:
    rows = []
    # profile_df columns: window, sector, lat, value
    for comp, (target, ref) in settings.comparisons.items():
        tdf = profile_df[profile_df["window"] == target]
        rdf = profile_df[profile_df["window"] == ref]
        merged = tdf.merge(rdf, on=["sector", "lat"], suffixes=("_target", "_reference"))
        for r in merged.itertuples(index=False):
            rows.append({
                "comparison": comp,
                "target_window": target,
                "reference_window": ref,
                "sector": r.sector,
                "lat": float(r.lat),
                f"{value_name}_target": float(r.value_target),
                f"{value_name}_reference": float(r.value_reference),
                f"{value_name}_delta": float(r.value_target - r.value_reference),
            })
    return pd.DataFrame(rows)


def generate_lat_bands(lat: np.ndarray, step: float) -> List[Tuple[str, float, float]]:
    finite = np.asarray(lat, dtype=float)[np.isfinite(lat)]
    mn = float(np.floor(np.nanmin(finite) / step) * step)
    mx = float(np.ceil(np.nanmax(finite) / step) * step)
    bands: List[Tuple[str, float, float]] = []
    x = mn
    while x < mx - 1e-9:
        lo = x
        hi = min(x + step, mx)
        # Include only bands overlapping data range.
        if hi >= np.nanmin(finite) - 1e-9 and lo <= np.nanmax(finite) + 1e-9:
            bands.append((f"{lo:g}_{hi:g}N", lo, hi))
        x += step
    return bands


def summarize_latbands(
    p_maps: Dict[str, np.ndarray],
    v_maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    settings: PVLatitudinalObjectChangeSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bands = generate_lat_bands(lat, settings.latband_step_deg)
    p_rows = []
    v_rows = []
    for window in settings.window_order:
        p_map = p_maps[window]
        v_map = v_maps[window]
        for sector_name, sec in settings.lon_sectors.items():
            lon_mask = mask_sector(lon, sec)
            if not np.any(lon_mask):
                continue
            for band_name, lo, hi in bands:
                lat_mask = (lat >= lo) & (lat < hi if hi < max(lat.max(), lat.min()) else lat <= hi)
                m = lat_mask[:, None] & lon_mask[None, :]
                if not np.any(m):
                    continue
                p_vals = p_map[m]
                v_vals = v_map[m]
                p_rows.append({
                    "window": window,
                    "sector": sector_name,
                    "lat_band": band_name,
                    "lat_min": lo,
                    "lat_max": hi,
                    "P_mean": float(np.nanmean(p_vals)),
                    "P_integrated": float(np.nansum(p_vals)),
                    "P_share_within_sector_window": np.nan,  # filled below
                    "n_valid_grid": int(np.sum(np.isfinite(p_vals))),
                })
                v_rows.append({
                    "window": window,
                    "sector": sector_name,
                    "lat_band": band_name,
                    "lat_min": lo,
                    "lat_max": hi,
                    "V850_mean": float(np.nanmean(v_vals)),
                    "V850_abs_mean": float(np.nanmean(np.abs(v_vals))),
                    "V850_positive_mean": float(np.nanmean(np.maximum(v_vals, 0.0))),
                    "V850_positive_integrated": float(np.nansum(np.maximum(v_vals, 0.0))),
                    "n_valid_grid": int(np.sum(np.isfinite(v_vals))),
                })
    pdf = pd.DataFrame(p_rows)
    if not pdf.empty:
        totals = pdf.groupby(["window", "sector"], dropna=False)["P_integrated"].transform("sum")
        pdf["P_share_within_sector_window"] = np.where(np.abs(totals) > 1e-12, pdf["P_integrated"] / totals, np.nan)
    return pdf, pd.DataFrame(v_rows)


def _local_maxima(y: np.ndarray) -> List[int]:
    y = np.asarray(y, dtype=float)
    out = []
    for i in range(1, len(y) - 1):
        if not np.isfinite(y[i]):
            continue
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and (y[i] > y[i - 1] or y[i] > y[i + 1]):
            out.append(i)
    if len(y) >= 2:
        if np.isfinite(y[0]) and y[0] > y[1]:
            out.insert(0, 0)
        if np.isfinite(y[-1]) and y[-1] > y[-2]:
            out.append(len(y) - 1)
    return out


def detect_precip_bands_for_profile(
    lat: np.ndarray,
    profile: np.ndarray,
    settings: PVLatitudinalObjectChangeSettings,
) -> List[Dict[str, float]]:
    lat, prof_raw = sorted_lat_profile(lat, profile)
    prof = maybe_smooth_profile(prof_raw, settings.profile_smooth_points)
    if prof.size == 0 or not np.any(np.isfinite(prof)):
        return []
    max_val = float(np.nanmax(prof))
    if max_val <= 0 or not np.isfinite(max_val):
        return []
    min_peak = max(settings.p_peak_relative_threshold * max_val, settings.p_peak_absolute_threshold)
    candidates = [i for i in _local_maxima(prof) if prof[i] >= min_peak]
    # Enforce minimum peak distance by taking stronger peaks first.
    candidates = sorted(candidates, key=lambda i: prof[i], reverse=True)
    kept: List[int] = []
    for i in candidates:
        if all(abs(lat[i] - lat[j]) >= settings.p_min_peak_distance_deg for j in kept):
            kept.append(i)
    kept = sorted(kept, key=lambda i: lat[i])
    bands: List[Dict[str, float]] = []
    for i in kept:
        peak = float(prof[i])
        cutoff = settings.p_edge_fraction_of_peak * peak
        left = i
        while left > 0 and prof[left] >= cutoff:
            left -= 1
        right = i
        while right < len(prof) - 1 and prof[right] >= cutoff:
            right += 1
        south_edge = float(lat[left])
        north_edge = float(lat[right])
        width = north_edge - south_edge
        if width < settings.p_min_band_width_deg:
            # Widen to nearest gridpoint span if possible, but keep flag via narrow_width.
            pass
        segment = slice(left, right + 1)
        vals = prof_raw[segment]
        lats = lat[segment]
        integ = safe_trapezoid(vals, lats)
        total = safe_trapezoid(np.maximum(prof_raw, 0.0), lat)
        denom = float(np.nansum(vals))
        centroid = float(np.nansum(vals * lats) / denom) if abs(denom) > 1e-12 else np.nan
        bands.append({
            "peak_lat": float(lat[i]),
            "peak_value": peak,
            "south_edge": south_edge,
            "north_edge": north_edge,
            "band_width": float(width),
            "band_integrated_amount": integ,
            "band_share": float(integ / total) if abs(total) > 1e-12 else np.nan,
            "band_centroid_lat": centroid,
            "narrow_width_flag": bool(width < settings.p_min_band_width_deg),
        })
    return bands


def build_precip_multiband_summary(
    p_profile_df: pd.DataFrame,
    settings: PVLatitudinalObjectChangeSettings,
) -> pd.DataFrame:
    rows = []
    for (window, sector), g in p_profile_df.groupby(["window", "sector"], dropna=False):
        lat = g["lat"].to_numpy(dtype=float)
        prof = g["value"].to_numpy(dtype=float)
        bands = detect_precip_bands_for_profile(lat, prof, settings)
        n = len(bands)
        for rank, b in enumerate(bands, start=1):
            rows.append({
                "window": window,
                "sector": sector,
                "n_precip_bands": n,
                "multiband": bool(n >= 2),
                "band_rank": rank,
                **b,
            })
        if n == 0:
            rows.append({
                "window": window,
                "sector": sector,
                "n_precip_bands": 0,
                "multiband": False,
                "band_rank": np.nan,
                "peak_lat": np.nan,
                "peak_value": np.nan,
                "south_edge": np.nan,
                "north_edge": np.nan,
                "band_width": np.nan,
                "band_integrated_amount": np.nan,
                "band_share": np.nan,
                "band_centroid_lat": np.nan,
                "narrow_width_flag": False,
            })
    return pd.DataFrame(rows)


def build_precip_band_transition_links(
    band_df: pd.DataFrame,
    settings: PVLatitudinalObjectChangeSettings,
) -> pd.DataFrame:
    rows = []
    adjacent = list(zip(settings.window_order[:-1], settings.window_order[1:]))
    max_match_distance = 8.0
    for sector in settings.lon_sectors.keys():
        for from_w, to_w in adjacent:
            a = band_df[(band_df["window"] == from_w) & (band_df["sector"] == sector) & band_df["peak_lat"].notna()].copy()
            b = band_df[(band_df["window"] == to_w) & (band_df["sector"] == sector) & band_df["peak_lat"].notna()].copy()
            used_to = set()
            for ai, ar in a.iterrows():
                if b.empty:
                    rows.append({
                        "from_window": from_w,
                        "to_window": to_w,
                        "sector": sector,
                        "from_band_rank": ar.get("band_rank"),
                        "to_band_rank": np.nan,
                        "link_type": "disappear",
                        "peak_lat_shift": np.nan,
                        "peak_value_change": np.nan,
                        "band_share_change": np.nan,
                        "confidence": "low_no_target_band",
                    })
                    continue
                distances = (b["peak_lat"] - ar["peak_lat"]).abs()
                for idx in used_to:
                    if idx in distances.index:
                        distances.loc[idx] = np.inf
                best_idx = distances.idxmin()
                best_dist = float(distances.loc[best_idx])
                if np.isfinite(best_dist) and best_dist <= max_match_distance:
                    br = b.loc[best_idx]
                    used_to.add(best_idx)
                    # Rough overlap check.
                    overlap = min(float(ar["north_edge"]), float(br["north_edge"])) - max(float(ar["south_edge"]), float(br["south_edge"]))
                    link_type = "continue" if overlap >= -1e-9 or best_dist <= settings.p_min_peak_distance_deg else "uncertain"
                    rows.append({
                        "from_window": from_w,
                        "to_window": to_w,
                        "sector": sector,
                        "from_band_rank": ar.get("band_rank"),
                        "to_band_rank": br.get("band_rank"),
                        "link_type": link_type,
                        "peak_lat_shift": float(br["peak_lat"] - ar["peak_lat"]),
                        "peak_value_change": float(br["peak_value"] - ar["peak_value"]),
                        "band_share_change": float(br["band_share"] - ar["band_share"]) if pd.notna(br["band_share"]) and pd.notna(ar["band_share"]) else np.nan,
                        "confidence": "medium_peak_distance_and_overlap" if link_type == "continue" else "low_no_overlap",
                    })
                else:
                    rows.append({
                        "from_window": from_w,
                        "to_window": to_w,
                        "sector": sector,
                        "from_band_rank": ar.get("band_rank"),
                        "to_band_rank": np.nan,
                        "link_type": "disappear",
                        "peak_lat_shift": np.nan,
                        "peak_value_change": np.nan,
                        "band_share_change": np.nan,
                        "confidence": "low_no_nearby_target_band",
                    })
            for bi, br in b.iterrows():
                if bi not in used_to:
                    rows.append({
                        "from_window": from_w,
                        "to_window": to_w,
                        "sector": sector,
                        "from_band_rank": np.nan,
                        "to_band_rank": br.get("band_rank"),
                        "link_type": "emerge",
                        "peak_lat_shift": np.nan,
                        "peak_value_change": np.nan,
                        "band_share_change": np.nan,
                        "confidence": "low_no_source_band",
                    })
    return pd.DataFrame(rows)


def v850_feature_summary(
    v_profile_df: pd.DataFrame,
    settings: PVLatitudinalObjectChangeSettings,
) -> pd.DataFrame:
    rows = []
    for (window, sector), g in v_profile_df.groupby(["window", "sector"], dropna=False):
        lat = g["lat"].to_numpy(dtype=float)
        raw = g["value"].to_numpy(dtype=float)
        lat, raw = sorted_lat_profile(lat, raw)
        pos = np.maximum(raw, 0.0)
        absv = np.abs(raw)

        def peak_lat(vals: np.ndarray) -> float:
            if vals.size == 0 or not np.any(np.isfinite(vals)):
                return np.nan
            return float(lat[int(np.nanargmax(vals))])

        def centroid(vals: np.ndarray) -> float:
            denom = float(np.nansum(vals))
            return float(np.nansum(vals * lat) / denom) if abs(denom) > 1e-12 else np.nan

        pos_positive = pos > 0
        north_edge = float(np.nanmax(lat[pos_positive])) if np.any(pos_positive) else np.nan
        rows.append({
            "window": window,
            "sector": sector,
            "V850_mean_peak_lat": peak_lat(raw),
            "V850_positive_peak_lat": peak_lat(pos),
            "V850_positive_centroid_lat": centroid(pos),
            "V850_positive_north_edge": north_edge,
            "V850_positive_integrated_strength": safe_trapezoid(pos, lat),
            "V850_abs_peak_lat": peak_lat(absv),
            "V850_abs_centroid_lat": centroid(absv),
        })
    return pd.DataFrame(rows)


def precip_feature_summary(band_df: pd.DataFrame, p_profile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, sector), g in p_profile_df.groupby(["window", "sector"], dropna=False):
        lat = g["lat"].to_numpy(dtype=float)
        prof = g["value"].to_numpy(dtype=float)
        lat, prof = sorted_lat_profile(lat, prof)
        denom = float(np.nansum(prof))
        centroid = float(np.nansum(prof * lat) / denom) if abs(denom) > 1e-12 else np.nan
        b = band_df[(band_df["window"] == window) & (band_df["sector"] == sector) & band_df["peak_lat"].notna()]
        if b.empty:
            primary_peak = np.nan
            primary_val = np.nan
            n_bands = 0
            multiband = False
        else:
            top = b.sort_values("peak_value", ascending=False).iloc[0]
            primary_peak = float(top["peak_lat"])
            primary_val = float(top["peak_value"])
            n_bands = int(b["n_precip_bands"].iloc[0])
            multiband = bool(n_bands >= 2)
        rows.append({
            "window": window,
            "sector": sector,
            "P_n_bands": n_bands,
            "P_multiband": multiband,
            "P_primary_peak_lat": primary_peak,
            "P_primary_peak_value": primary_val,
            "P_total_centroid_lat": centroid,
        })
    return pd.DataFrame(rows)


def build_p_v_latitudinal_feature_summary(precip_feat: pd.DataFrame, v_feat: pd.DataFrame) -> pd.DataFrame:
    return precip_feat.merge(v_feat, on=["window", "sector"], how="outer")
