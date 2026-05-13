# -*- coding: utf-8 -*-
"""Core computations for P/V850 offset-correspondence audit v1_b."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .t3_p_v_offset_correspondence_settings import LonSectorSpec, PVOffsetCorrespondenceSettings


def finite_nanmean(arr: np.ndarray, axis=None) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=axis)


def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    dx = np.diff(x)
    return float(np.nansum((y[:-1] + y[1:]) * 0.5 * dx))


def sorted_lat_profile(lat: np.ndarray, profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(lat, dtype=float))
    return np.asarray(lat, dtype=float)[order], np.asarray(profile, dtype=float)[order]


def mask_sector(lon: np.ndarray, sector: LonSectorSpec) -> np.ndarray:
    return (lon >= sector.lon_min) & (lon <= sector.lon_max)


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
    out: Dict[str, np.ndarray] = {}
    for w, (start, end) in windows.items():
        days = window_day_indices(day_map, start, end)
        out[w] = finite_nanmean(field[:, days, :, :], axis=(0, 1))
    return out


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
                rows.append({"window": window, "sector": sector_name, "lat": float(la), "value": float(val)})
    return pd.DataFrame(rows)


def build_profile_delta(profile_df: pd.DataFrame, settings: PVOffsetCorrespondenceSettings, value_name: str) -> pd.DataFrame:
    rows = []
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


def _local_maxima(y: np.ndarray) -> List[int]:
    y = np.asarray(y, dtype=float)
    out: List[int] = []
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


def _local_minima(y: np.ndarray) -> List[int]:
    return _local_maxima(-np.asarray(y, dtype=float))


def _enforce_peak_distance(candidates: List[int], lat: np.ndarray, values: np.ndarray, min_distance: float) -> List[int]:
    candidates = sorted(candidates, key=lambda i: values[i], reverse=True)
    kept: List[int] = []
    for i in candidates:
        if all(abs(lat[i] - lat[j]) >= min_distance for j in kept):
            kept.append(i)
    return sorted(kept, key=lambda i: lat[i])


def _band_edges_fraction(lat: np.ndarray, prof: np.ndarray, peak_idx: int, peak_value: float, edge_fraction: float) -> Tuple[int, int]:
    cutoff = edge_fraction * peak_value
    left = peak_idx
    while left > 0 and prof[left] >= cutoff:
        left -= 1
    right = peak_idx
    while right < len(prof) - 1 and prof[right] >= cutoff:
        right += 1
    return left, right


def _band_edges_sign(lat: np.ndarray, delta: np.ndarray, peak_idx: int, sign: int) -> Tuple[int, int]:
    # Extend until the sign changes or the profile edge is reached.
    left = peak_idx
    while left > 0 and np.isfinite(delta[left - 1]) and sign * delta[left - 1] > 0:
        left -= 1
    right = peak_idx
    while right < len(delta) - 1 and np.isfinite(delta[right + 1]) and sign * delta[right + 1] > 0:
        right += 1
    return left, right


def detect_p_clim_bands_for_profile(lat: np.ndarray, profile: np.ndarray, settings: PVOffsetCorrespondenceSettings) -> List[Dict[str, float]]:
    lat, prof_raw = sorted_lat_profile(lat, profile)
    prof = maybe_smooth_profile(prof_raw, settings.profile_smooth_points)
    if prof.size == 0 or not np.any(np.isfinite(prof)):
        return []
    max_val = float(np.nanmax(prof))
    if max_val <= 0 or not np.isfinite(max_val):
        return []
    min_peak = max(settings.p_peak_relative_threshold * max_val, settings.p_peak_absolute_threshold)
    cand = [i for i in _local_maxima(prof) if prof[i] >= min_peak]
    kept = _enforce_peak_distance(cand, lat, prof, settings.min_peak_distance_deg)
    total = trapz_compat(np.maximum(prof_raw, 0.0), lat)
    out: List[Dict[str, float]] = []
    for i in kept:
        peak = float(prof[i])
        left, right = _band_edges_fraction(lat, prof, i, peak, settings.edge_fraction_of_peak)
        vals = prof_raw[left:right + 1]
        lats = lat[left:right + 1]
        integ = trapz_compat(np.maximum(vals, 0.0), lats)
        denom = float(np.nansum(np.maximum(vals, 0.0)))
        centroid = float(np.nansum(np.maximum(vals, 0.0) * lats) / denom) if abs(denom) > 1e-12 else np.nan
        out.append({
            "p_clim_peak_lat": float(lat[i]),
            "p_clim_peak_value": peak,
            "p_clim_band_south_edge": float(lat[left]),
            "p_clim_band_north_edge": float(lat[right]),
            "p_clim_band_width": float(lat[right] - lat[left]),
            "p_clim_band_integrated_amount": float(integ),
            "p_clim_band_share": float(integ / total) if abs(total) > 1e-12 else np.nan,
            "p_clim_band_centroid_lat": centroid,
            "narrow_width_flag": bool(float(lat[right] - lat[left]) < settings.min_band_width_deg),
        })
    return out


def build_p_clim_band_summary(p_profile_df: pd.DataFrame, settings: PVOffsetCorrespondenceSettings) -> pd.DataFrame:
    rows = []
    for (window, sector), g in p_profile_df.groupby(["window", "sector"], dropna=False):
        bands = detect_p_clim_bands_for_profile(g["lat"].to_numpy(float), g["value"].to_numpy(float), settings)
        n = len(bands)
        for rank, b in enumerate(bands, start=1):
            rows.append({"window": window, "sector": sector, "n_p_clim_bands": n, "p_clim_multiband": bool(n >= 2), "p_clim_band_id": rank, **b})
        if n == 0:
            rows.append({
                "window": window, "sector": sector, "n_p_clim_bands": 0, "p_clim_multiband": False, "p_clim_band_id": np.nan,
                "p_clim_peak_lat": np.nan, "p_clim_peak_value": np.nan,
                "p_clim_band_south_edge": np.nan, "p_clim_band_north_edge": np.nan,
                "p_clim_band_width": np.nan, "p_clim_band_integrated_amount": np.nan,
                "p_clim_band_share": np.nan, "p_clim_band_centroid_lat": np.nan,
                "narrow_width_flag": False,
            })
    return pd.DataFrame(rows)


def detect_p_change_bands_for_delta(lat: np.ndarray, delta_profile: np.ndarray, settings: PVOffsetCorrespondenceSettings, change_type: str) -> List[Dict[str, float]]:
    lat, raw = sorted_lat_profile(lat, delta_profile)
    sm = maybe_smooth_profile(raw, settings.profile_smooth_points)
    if sm.size == 0 or not np.any(np.isfinite(sm)):
        return []
    sign = 1 if change_type == "positive" else -1
    vals = sign * sm
    max_val = float(np.nanmax(vals))
    if max_val <= 0 or not np.isfinite(max_val):
        return []
    min_peak = max(settings.p_delta_peak_relative_threshold * max_val, settings.p_delta_peak_absolute_threshold)
    cand = [i for i in _local_maxima(vals) if vals[i] >= min_peak]
    kept = _enforce_peak_distance(cand, lat, vals, settings.min_peak_distance_deg)
    out: List[Dict[str, float]] = []
    for i in kept:
        left, right = _band_edges_sign(lat, raw, i, sign)
        lats = lat[left:right + 1]
        band_vals = raw[left:right + 1]
        signed_vals = sign * band_vals
        integ = sign * trapz_compat(band_vals, lats)
        denom = float(np.nansum(np.maximum(signed_vals, 0.0)))
        centroid = float(np.nansum(np.maximum(signed_vals, 0.0) * lats) / denom) if abs(denom) > 1e-12 else np.nan
        out.append({
            "p_change_peak_lat": float(lat[i]),
            "p_change_peak_value": float(raw[i]),
            "p_change_band_south_edge": float(lat[left]),
            "p_change_band_north_edge": float(lat[right]),
            "p_change_band_width": float(lat[right] - lat[left]),
            "p_change_band_integrated_amount": float(integ),
            "p_change_band_centroid_lat": centroid,
        })
    return out


def build_p_change_band_summary(p_delta_df: pd.DataFrame, settings: PVOffsetCorrespondenceSettings) -> pd.DataFrame:
    rows = []
    for (comparison, sector), g in p_delta_df.groupby(["comparison", "sector"], dropna=False):
        lat = g["lat"].to_numpy(float)
        col = "P_delta"
        # build_profile_delta names P_delta when value_name='P'; support old variants too
        if col not in g.columns:
            col = [c for c in g.columns if c.endswith("_delta")][0]
        profile = g[col].to_numpy(float)
        for change_type in ["positive", "negative"]:
            bands = detect_p_change_bands_for_delta(lat, profile, settings, change_type)
            n = len(bands)
            for rank, b in enumerate(bands, start=1):
                rows.append({"comparison": comparison, "sector": sector, "p_change_type": change_type, "n_p_change_bands_of_type": n, "p_change_band_id": rank, **b})
            if n == 0:
                rows.append({
                    "comparison": comparison, "sector": sector, "p_change_type": change_type, "n_p_change_bands_of_type": 0,
                    "p_change_band_id": np.nan, "p_change_peak_lat": np.nan, "p_change_peak_value": np.nan,
                    "p_change_band_south_edge": np.nan, "p_change_band_north_edge": np.nan,
                    "p_change_band_width": np.nan, "p_change_band_integrated_amount": np.nan,
                    "p_change_band_centroid_lat": np.nan,
                })
    return pd.DataFrame(rows)


def v_clim_structure_summary(v_profile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, sector), g in v_profile_df.groupby(["window", "sector"], dropna=False):
        lat, raw = sorted_lat_profile(g["lat"].to_numpy(float), g["value"].to_numpy(float))
        pos = np.maximum(raw, 0.0)
        mask = pos > 0
        if np.any(mask):
            pos_peak_idx = int(np.nanargmax(pos))
            denom = float(np.nansum(pos))
            centroid = float(np.nansum(pos * lat) / denom) if abs(denom) > 1e-12 else np.nan
            south_edge = float(np.nanmin(lat[mask]))
            north_edge = float(np.nanmax(lat[mask]))
            band_width = north_edge - south_edge
            integrated = trapz_compat(pos, lat)
            peak_lat = float(lat[pos_peak_idx])
            peak_value = float(pos[pos_peak_idx])
        else:
            peak_lat = centroid = south_edge = north_edge = band_width = integrated = peak_value = np.nan
        rows.append({
            "window": window,
            "sector": sector,
            "v_positive_peak_lat": peak_lat,
            "v_positive_peak_value": peak_value,
            "v_positive_centroid_lat": centroid,
            "v_positive_south_edge": south_edge,
            "v_positive_north_edge": north_edge,
            "v_positive_integrated_strength": float(integrated),
            "v_positive_band_width": float(band_width) if np.isfinite(band_width) else np.nan,
        })
    return pd.DataFrame(rows)


def _strongest_peak(lat: np.ndarray, vals: np.ndarray, kind: str, rel: float, abs_thr: float, min_dist: float) -> Tuple[float, float]:
    lat, vals = sorted_lat_profile(lat, vals)
    work = vals.copy() if kind == "peak" else -vals.copy()
    if not np.any(np.isfinite(work)):
        return np.nan, np.nan
    max_val = float(np.nanmax(work))
    if max_val <= 0 or not np.isfinite(max_val):
        return np.nan, np.nan
    min_peak = max(rel * max_val, abs_thr)
    cand = [i for i in _local_maxima(work) if work[i] >= min_peak]
    if not cand:
        i = int(np.nanargmax(work))
        if work[i] < min_peak:
            return np.nan, np.nan
        cand = [i]
    kept = _enforce_peak_distance(cand, lat, work, min_dist)
    i = max(kept, key=lambda j: work[j])
    return float(lat[i]), float(vals[i])


def v_change_structure_summary(v_delta_df: pd.DataFrame, v_clim_df: pd.DataFrame, settings: PVOffsetCorrespondenceSettings) -> pd.DataFrame:
    rows = []
    clim_idx = {(r.window, r.sector): r for r in v_clim_df.itertuples(index=False)}
    for (comparison, sector), g in v_delta_df.groupby(["comparison", "sector"], dropna=False):
        target, ref = settings.comparisons[str(comparison)]
        lat = g["lat"].to_numpy(float)
        col = "V850_delta" if "V850_delta" in g.columns else [c for c in g.columns if c.endswith("_delta")][0]
        delta = g[col].to_numpy(float)
        lat_s, delta_s = sorted_lat_profile(lat, delta)
        grad = np.gradient(delta_s, lat_s)
        peak_lat, peak_val = _strongest_peak(lat_s, delta_s, "peak", settings.v_change_relative_threshold, settings.v_change_absolute_threshold, settings.min_peak_distance_deg)
        trough_lat, trough_val = _strongest_peak(lat_s, delta_s, "trough", settings.v_change_relative_threshold, settings.v_change_absolute_threshold, settings.min_peak_distance_deg)
        grad_peak_lat, grad_peak_val = _strongest_peak(lat_s, np.abs(grad), "peak", settings.gradient_change_relative_threshold, settings.gradient_change_absolute_threshold, settings.min_peak_distance_deg)
        tr = clim_idx.get((target, sector), None)
        rr = clim_idx.get((ref, sector), None)
        if tr is not None and rr is not None:
            edge_shift = float(tr.v_positive_north_edge - rr.v_positive_north_edge)
            centroid_shift = float(tr.v_positive_centroid_lat - rr.v_positive_centroid_lat)
        else:
            edge_shift = centroid_shift = np.nan
        rows.append({
            "comparison": comparison,
            "target_window": target,
            "reference_window": ref,
            "sector": sector,
            "v_change_peak_lat": peak_lat,
            "v_change_peak_value": peak_val,
            "v_change_trough_lat": trough_lat,
            "v_change_trough_value": trough_val,
            "v_gradient_change_peak_lat": grad_peak_lat,
            "v_gradient_change_peak_value": grad_peak_val,
            "v_positive_north_edge_shift": edge_shift,
            "v_positive_centroid_shift": centroid_shift,
        })
    return pd.DataFrame(rows)


def interp_at(lat: np.ndarray, values: np.ndarray, target_lat: float) -> float:
    lat, values = sorted_lat_profile(lat, values)
    if not np.isfinite(target_lat) or target_lat < np.nanmin(lat) or target_lat > np.nanmax(lat):
        return np.nan
    return float(np.interp(target_lat, lat, values))


def build_p_clim_band_to_v_clim_structure(
    p_clim_df: pd.DataFrame,
    v_profile_df: pd.DataFrame,
    v_clim_df: pd.DataFrame,
    settings: PVOffsetCorrespondenceSettings,
) -> pd.DataFrame:
    rows = []
    vprof = {(w, s): g for (w, s), g in v_profile_df.groupby(["window", "sector"], dropna=False)}
    vclim = {(r.window, r.sector): r for r in v_clim_df.itertuples(index=False)}
    for r in p_clim_df.itertuples(index=False):
        if not np.isfinite(getattr(r, "p_clim_peak_lat", np.nan)):
            continue
        key = (r.window, r.sector)
        if key not in vprof or key not in vclim:
            continue
        vg = vprof[key]
        lat = vg["lat"].to_numpy(float)
        v = vg["value"].to_numpy(float)
        vc = vclim[key]
        peak_lat = float(r.p_clim_peak_lat)
        row = {
            "window": r.window,
            "sector": r.sector,
            "p_clim_band_id": r.p_clim_band_id,
            "p_clim_peak_lat": peak_lat,
            "p_clim_peak_value": r.p_clim_peak_value,
            "p_clim_band_share": r.p_clim_band_share,
            "v_positive_peak_lat": vc.v_positive_peak_lat,
            "v_positive_centroid_lat": vc.v_positive_centroid_lat,
            "v_positive_north_edge": vc.v_positive_north_edge,
            "v_positive_south_edge": vc.v_positive_south_edge,
            "distance_p_to_v_peak": peak_lat - float(vc.v_positive_peak_lat) if np.isfinite(vc.v_positive_peak_lat) else np.nan,
            "distance_p_to_v_centroid": peak_lat - float(vc.v_positive_centroid_lat) if np.isfinite(vc.v_positive_centroid_lat) else np.nan,
            "distance_p_to_v_north_edge": peak_lat - float(vc.v_positive_north_edge) if np.isfinite(vc.v_positive_north_edge) else np.nan,
            "distance_p_to_v_south_edge": peak_lat - float(vc.v_positive_south_edge) if np.isfinite(vc.v_positive_south_edge) else np.nan,
        }
        for off in settings.offset_degrees:
            name = "same_lat" if off == 0 else (f"south_{abs(int(off))}" if off < 0 else f"north_{int(off)}")
            row[f"v_{name}"] = interp_at(lat, v, peak_lat + off)
        rows.append(row)
    return pd.DataFrame(rows)


def build_p_change_peak_to_v_change_structure(
    p_change_df: pd.DataFrame,
    v_delta_df: pd.DataFrame,
    v_change_df: pd.DataFrame,
    settings: PVOffsetCorrespondenceSettings,
) -> pd.DataFrame:
    rows = []
    vdelta = {(c, s): g for (c, s), g in v_delta_df.groupby(["comparison", "sector"], dropna=False)}
    vchange = {(r.comparison, r.sector): r for r in v_change_df.itertuples(index=False)}
    for r in p_change_df.itertuples(index=False):
        peak_lat = getattr(r, "p_change_peak_lat", np.nan)
        if not np.isfinite(peak_lat):
            continue
        key = (r.comparison, r.sector)
        if key not in vdelta or key not in vchange:
            continue
        vg = vdelta[key]
        lat = vg["lat"].to_numpy(float)
        col = "V850_delta" if "V850_delta" in vg.columns else [c for c in vg.columns if c.endswith("_delta")][0]
        vdel = vg[col].to_numpy(float)
        lat_s, vdel_s = sorted_lat_profile(lat, vdel)
        grad = np.gradient(vdel_s, lat_s)
        vc = vchange[key]
        row = {
            "comparison": r.comparison,
            "sector": r.sector,
            "p_change_type": r.p_change_type,
            "p_change_band_id": r.p_change_band_id,
            "p_change_peak_lat": peak_lat,
            "p_change_peak_value": r.p_change_peak_value,
            "p_change_band_integrated_amount": r.p_change_band_integrated_amount,
            "v_change_peak_lat": vc.v_change_peak_lat,
            "v_change_peak_value": vc.v_change_peak_value,
            "v_change_trough_lat": vc.v_change_trough_lat,
            "v_change_trough_value": vc.v_change_trough_value,
            "v_gradient_change_peak_lat": vc.v_gradient_change_peak_lat,
            "v_gradient_change_peak_value": vc.v_gradient_change_peak_value,
            "v_positive_north_edge_shift": vc.v_positive_north_edge_shift,
            "v_positive_centroid_shift": vc.v_positive_centroid_shift,
            "distance_p_change_to_v_change_peak": peak_lat - float(vc.v_change_peak_lat) if np.isfinite(vc.v_change_peak_lat) else np.nan,
            "distance_p_change_to_v_change_trough": peak_lat - float(vc.v_change_trough_lat) if np.isfinite(vc.v_change_trough_lat) else np.nan,
            "distance_p_change_to_v_gradient_peak": peak_lat - float(vc.v_gradient_change_peak_lat) if np.isfinite(vc.v_gradient_change_peak_lat) else np.nan,
        }
        # Approximate shifted edge as edge shift direction only; distance to actual target edge not meaningful in delta-only table.
        for off in settings.offset_degrees:
            name = "same_lat" if off == 0 else (f"south_{abs(int(off))}" if off < 0 else f"north_{int(off)}")
            row[f"v_{name}_delta"] = interp_at(lat_s, vdel_s, peak_lat + off)
            row[f"dV_{name}_delta"] = interp_at(lat_s, grad, peak_lat + off)
        rows.append(row)
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
        bands.append((f"{lo:g}_{hi:g}N", lo, hi))
        x += step
    return bands


def build_p_highlat_v_north_edge_correspondence(
    p_profile_df: pd.DataFrame,
    v_clim_df: pd.DataFrame,
    settings: PVOffsetCorrespondenceSettings,
) -> pd.DataFrame:
    rows = []
    # Use lat bands directly from profiles, focusing on >=35N.
    all_lat = p_profile_df["lat"].to_numpy(float)
    bands = [(name, lo, hi) for name, lo, hi in generate_lat_bands(all_lat, settings.latband_step_deg) if hi > 35.0]
    vclim = {(r.window, r.sector): r for r in v_clim_df.itertuples(index=False)}
    for comp, (target, ref) in settings.comparisons.items():
        for sector in settings.lon_sectors.keys():
            t = p_profile_df[(p_profile_df["window"] == target) & (p_profile_df["sector"] == sector)]
            r = p_profile_df[(p_profile_df["window"] == ref) & (p_profile_df["sector"] == sector)]
            if t.empty or r.empty:
                continue
            m = t.merge(r, on=["sector", "lat"], suffixes=("_target", "_reference"))
            row = {"comparison": comp, "sector": sector}
            for name, lo, hi in bands:
                sub = m[(m["lat"] >= lo) & (m["lat"] < hi if hi < np.nanmax(all_lat) else m["lat"] <= hi)]
                if sub.empty:
                    row[f"P_{name}_delta"] = np.nan
                else:
                    row[f"P_{name}_delta"] = float(np.nanmean(sub["value_target"] - sub["value_reference"]))
            vt = vclim.get((target, sector), None)
            vr = vclim.get((ref, sector), None)
            if vt is not None and vr is not None:
                row["V_positive_north_edge_shift"] = float(vt.v_positive_north_edge - vr.v_positive_north_edge)
                row["V_positive_centroid_shift"] = float(vt.v_positive_centroid_lat - vr.v_positive_centroid_lat)
            else:
                row["V_positive_north_edge_shift"] = np.nan
                row["V_positive_centroid_shift"] = np.nan
            high_cols = [k for k in row if k.startswith("P_") and k.endswith("_delta")]
            high_vals = [row[k] for k in high_cols if np.isfinite(row[k])]
            edge = row["V_positive_north_edge_shift"]
            if np.isfinite(edge) and edge > 0 and high_vals and sum(v > 0 for v in high_vals) >= max(1, len(high_vals)//2):
                ctype = "v_north_edge_extends_with_highlat_p_increase"
            elif np.isfinite(edge) and edge < 0 and high_vals and sum(v < 0 for v in high_vals) >= max(1, len(high_vals)//2):
                ctype = "v_north_edge_retracts_with_highlat_p_decline"
            else:
                ctype = "mixed_or_not_aligned"
            row["correspondence_type"] = ctype
            rows.append(row)
    return pd.DataFrame(rows)


def build_diagnosis_table(
    clim_corr: pd.DataFrame,
    change_corr: pd.DataFrame,
    highlat_corr: pd.DataFrame,
    settings: PVOffsetCorrespondenceSettings,
) -> pd.DataFrame:
    rows = []

    def add(diagnosis_id, support_level, comparison_or_window, sector, primary, counter, allowed, forbidden):
        rows.append({
            "diagnosis_id": diagnosis_id,
            "support_level": support_level,
            "comparison_or_window": comparison_or_window,
            "sector": sector,
            "primary_evidence": primary,
            "counter_evidence": counter,
            "allowed_statement": allowed,
            "forbidden_statement": forbidden,
        })

    # Same-region correspondence insufficiency is a design-level warning, always retained.
    add(
        "same_region_correspondence_insufficient",
        "methodological_warning",
        "all",
        "all",
        "P and V850 are compared through pre-registered offset/edge/gradient structures, not only same-region same-sign values.",
        "None; this is a design constraint.",
        "Same-region P and V850 deltas are insufficient as a standalone correspondence claim.",
        "Do not infer no correspondence solely from opposite-signed same-region P and V850 changes.",
    )

    if not highlat_corr.empty:
        for r in highlat_corr.itertuples(index=False):
            ctype = r.correspondence_type
            if ctype == "v_north_edge_extends_with_highlat_p_increase":
                add(
                    "v_northward_extension_with_p_highlat_increase",
                    "supported_in_this_sector_comparison",
                    r.comparison,
                    r.sector,
                    f"V_positive_north_edge_shift={r.V_positive_north_edge_shift:.6g}; high-lat P deltas mostly positive.",
                    "Does not prove causality or lagged influence.",
                    "V850 positive north-edge extension is object-layer consistent with high-latitude P increase for this sector/comparison.",
                    "Do not state that V850 caused high-latitude P increase.",
                )
            elif ctype == "v_north_edge_retracts_with_highlat_p_decline":
                add(
                    "v_retreat_with_p_highlat_decline",
                    "supported_in_this_sector_comparison",
                    r.comparison,
                    r.sector,
                    f"V_positive_north_edge_shift={r.V_positive_north_edge_shift:.6g}; high-lat P deltas mostly negative.",
                    "Does not prove causality or lagged influence.",
                    "V850 positive north-edge retreat is object-layer consistent with high-latitude P decline for this sector/comparison.",
                    "Do not state that V850 caused high-latitude P retreat.",
                )

    # Offset candidate counts.
    if not change_corr.empty:
        near_edge = change_corr[change_corr["distance_p_change_to_v_gradient_peak"].abs() <= 5.0]
        add(
            "p_change_peak_to_v_change_structure_offset",
            "case_count_only",
            "all_comparisons",
            "all_sectors",
            f"n_change_peaks_within_5deg_of_v_gradient_change_peak={len(near_edge)}",
            "Distance-only cases are not mechanism proof.",
            "P change peaks can be inspected against V850 change/gradient structures using the dedicated table.",
            "Do not treat the nearest V structure as a post-hoc causal match.",
        )

    if not clim_corr.empty:
        add(
            "p_clim_band_to_v_clim_structure_offset",
            "available_for_review",
            "all_windows",
            "all_sectors",
            f"n_p_clim_band_to_v_structure_rows={len(clim_corr)}",
            "Climatological alignment does not imply change correspondence.",
            "P climatological bands can be compared with V850 climatological peak/centroid/edge positions.",
            "Do not use climatological P bands as P change peaks.",
        )

    return pd.DataFrame(rows)
