"""V7-z-je-audit-c: Je physical variance audit for W45.

This module is intentionally narrow. It does not modify the V7-z main
pipeline. It tests whether the Je early shape-pattern peak near day33 can be
physically described as a low spatial-variance / flattened-structure episode
in the Je region.

Main idea:
- Rebuild Je raw 2° profile and raw 2D regional fields from smoothed u200.
- Compute spatial variance / std after removing the spatial mean.
- Compare day30-34 low-variance window with neighboring and W45 windows.
- Bootstrap over years to determine whether the variance dip is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import os
import math

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class JePhysicalVarianceAuditConfig:
    version: str = "v7_z_c"
    output_tag: str = "W45_Je_physical_variance_audit_v7_z_c"
    field_aliases: Tuple[str, ...] = (
        "u200", "U200", "u200_smoothed", "U200_smoothed", "u", "U", "uwnd200", "ua200"
    )
    lat_aliases: Tuple[str, ...] = ("lat", "latitude", "lats")
    lon_aliases: Tuple[str, ...] = ("lon", "longitude", "lons")
    year_aliases: Tuple[str, ...] = ("years", "year")
    je_lon_min: float = 120.0
    je_lon_max: float = 150.0
    je_lat_min: float = 25.0
    je_lat_max: float = 45.0
    lat_step: float = 2.0
    full_start_day: int = 0
    full_end_day: int = 70
    early_pre_start: int = 20
    early_pre_end: int = 29
    low_variance_start: int = 30
    low_variance_end: int = 34
    early_core_start: int = 26
    early_core_end: int = 33
    early_post_start: int = 35
    early_post_end: int = 39
    w45_start: int = 40
    w45_end: int = 48
    late_main_start: int = 40
    late_main_end: int = 52
    n_bootstrap: int = 1000
    random_seed: int = 42
    low_quantile: float = 0.10
    min_window_start: int = 20
    min_window_end: int = 40
    eps: float = 1e-12
    save_by_year_metrics: bool = False
    skip_figures: bool = False


def _log(msg: str) -> None:
    print(msg, flush=True)


def _first_existing_key(keys: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    keyset = set(keys)
    for a in aliases:
        if a in keyset:
            return a
    lower_map = {k.lower(): k for k in keys}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def _default_smoothed_path(v7_root: Path) -> Path:
    project_root = v7_root.parents[1] if len(v7_root.parents) >= 2 else v7_root
    return project_root / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _load_npz(v7_root: Path, cfg: JePhysicalVarianceAuditConfig) -> Tuple[Dict[str, np.ndarray], Path, pd.DataFrame]:
    path = Path(os.environ.get("V7Z_SMOOTHED_FIELDS", str(_default_smoothed_path(v7_root))))
    audit_rows = []
    if not path.exists():
        audit_rows.append({"input": "smoothed_fields", "path": str(path), "exists": False, "status": "missing"})
        return {}, path, pd.DataFrame(audit_rows)
    with np.load(path, allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files}
    audit_rows.append({"input": "smoothed_fields", "path": str(path), "exists": True, "status": "loaded", "keys": ";".join(data.keys())})
    return data, path, pd.DataFrame(audit_rows)


def _resolve_required_arrays(data: Dict[str, np.ndarray], cfg: JePhysicalVarianceAuditConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    rows = []
    u_key = _first_existing_key(data.keys(), cfg.field_aliases)
    lat_key = _first_existing_key(data.keys(), cfg.lat_aliases)
    lon_key = _first_existing_key(data.keys(), cfg.lon_aliases)
    year_key = _first_existing_key(data.keys(), cfg.year_aliases)
    for name, key in [("u200", u_key), ("lat", lat_key), ("lon", lon_key), ("years", year_key)]:
        rows.append({"required": name, "resolved_key": key, "status": "ok" if key is not None or name == "years" else "missing"})
    if u_key is None or lat_key is None or lon_key is None:
        raise KeyError("Could not resolve u200/lat/lon keys. See input audit output.")
    arr = np.asarray(data[u_key], dtype=float)
    lat = np.asarray(data[lat_key], dtype=float).ravel()
    lon = np.asarray(data[lon_key], dtype=float).ravel()
    years = np.asarray(data[year_key]).ravel() if year_key is not None else None
    return arr, lat, lon, years, pd.DataFrame(rows)


def _standardize_field_dims(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, years: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return field as (year, day, lat, lon)."""
    shape = arr.shape
    if arr.ndim == 4:
        lat_axes = [i for i, s in enumerate(shape) if s == len(lat)]
        lon_axes = [i for i, s in enumerate(shape) if s == len(lon)]
        if not lat_axes or not lon_axes:
            raise ValueError(f"Could not infer lat/lon axes from array shape {shape}.")
        lat_axis = lat_axes[0]
        lon_axis = lon_axes[0]
        remaining = [i for i in range(4) if i not in (lat_axis, lon_axis)]
        year_axis = None
        if years is not None:
            cand = [i for i in remaining if shape[i] == len(years)]
            year_axis = cand[0] if cand else None
        if year_axis is None:
            # Assume the smaller remaining dimension is year if plausible; otherwise first.
            rem_sizes = [(shape[i], i) for i in remaining]
            rem_sizes_sorted = sorted(rem_sizes)
            year_axis = rem_sizes_sorted[0][1]
        day_axis = [i for i in remaining if i != year_axis][0]
        out = np.moveaxis(arr, [year_axis, day_axis, lat_axis, lon_axis], [0, 1, 2, 3])
        return out, {"year_axis": year_axis, "day_axis": day_axis, "lat_axis": lat_axis, "lon_axis": lon_axis}
    if arr.ndim == 3:
        lat_axes = [i for i, s in enumerate(shape) if s == len(lat)]
        lon_axes = [i for i, s in enumerate(shape) if s == len(lon)]
        if not lat_axes or not lon_axes:
            raise ValueError(f"Could not infer lat/lon axes from 3D array shape {shape}.")
        lat_axis = lat_axes[0]
        lon_axis = lon_axes[0]
        day_axis = [i for i in range(3) if i not in (lat_axis, lon_axis)][0]
        out3 = np.moveaxis(arr, [day_axis, lat_axis, lon_axis], [0, 1, 2])
        out = out3[None, ...]
        return out, {"year_axis": -1, "day_axis": day_axis, "lat_axis": lat_axis, "lon_axis": lon_axis}
    raise ValueError(f"Unsupported u200 array shape: {shape}")


def _lon_mask(lon: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    lon_mod = np.mod(lon, 360.0)
    lo = lon_min % 360.0
    hi = lon_max % 360.0
    if lo <= hi:
        return (lon_mod >= lo) & (lon_mod <= hi)
    return (lon_mod >= lo) | (lon_mod <= hi)


def _select_region(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: JePhysicalVarianceAuditConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_mask = (lat >= min(cfg.je_lat_min, cfg.je_lat_max)) & (lat <= max(cfg.je_lat_min, cfg.je_lat_max))
    lon_mask = _lon_mask(lon, cfg.je_lon_min, cfg.je_lon_max)
    if lat_mask.sum() < 2 or lon_mask.sum() < 2:
        raise ValueError("Je region selection produced too few lat/lon points.")
    lat_sel = lat[lat_mask]
    lon_sel = lon[lon_mask]
    region = field[:, :, lat_mask, :][:, :, :, lon_mask]
    # Sort latitude ascending for consistent output.
    order = np.argsort(lat_sel)
    return region[:, :, order, :], lat_sel[order], lon_sel


def _target_profile_lats(cfg: JePhysicalVarianceAuditConfig) -> np.ndarray:
    return np.arange(cfg.je_lat_min, cfg.je_lat_max + 0.1 * cfg.lat_step, cfg.lat_step, dtype=float)


def _build_profiles(region: np.ndarray, lat_sel: np.ndarray, cfg: JePhysicalVarianceAuditConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Build lon-mean profile and interpolate to 2-degree target lats."""
    lon_mean = np.nanmean(region, axis=3)  # year, day, lat
    target = _target_profile_lats(cfg)
    profiles = np.empty((lon_mean.shape[0], lon_mean.shape[1], len(target)), dtype=float)
    for y in range(lon_mean.shape[0]):
        for d in range(lon_mean.shape[1]):
            profiles[y, d, :] = np.interp(target, lat_sel, lon_mean[y, d, :])
    return profiles, target


def _lat_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() == 0:
        w = np.ones_like(lat, dtype=float)
    return w / w.sum()


def _weighted_mean(x: np.ndarray, w: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.nansum(x * np.expand_dims(w, tuple(range(x.ndim - 1))) if False else x * w, axis=axis)


def _profile_metrics(prof: np.ndarray, target_lat: np.ndarray) -> pd.DataFrame:
    """Metrics from climatological profile: prof shape (day, lat).

    Hotfix note: smoothed fields may have boundary days where the whole profile
    is NaN. Those days must be carried as invalid rows rather than causing
    np.nanargmax / np.nanmax to crash.
    """
    base_w = _lat_weights(target_lat)
    rows = []
    valid_flags = []
    for d in range(prof.shape[0]):
        x = prof[d, :].astype(float)
        finite = np.isfinite(x)
        if finite.sum() < 2:
            rows.append({
                "day": d,
                "profile_valid": False,
                "profile_n_finite": int(finite.sum()),
                "profile_mean": np.nan,
                "profile_spatial_variance": np.nan,
                "profile_spatial_std": np.nan,
                "profile_amplitude": np.nan,
                "profile_weighted_l2_norm": np.nan,
                "axis_lat": np.nan,
                "centroid_lat": np.nan,
                "spread": np.nan,
                "width_proxy": np.nan,
                "NS_contrast": np.nan,
                "skewness": np.nan,
            })
            valid_flags.append(False)
            continue

        w = np.where(finite, base_w, 0.0)
        wsum = float(np.nansum(w))
        if wsum <= 0:
            w = np.where(finite, 1.0, 0.0)
            wsum = float(np.nansum(w))
        w = w / max(wsum, 1e-12)

        mean = float(np.nansum(w * x))
        anom = np.where(finite, x - mean, np.nan)
        var = float(np.nansum(w * anom ** 2))
        std = math.sqrt(max(var, 0.0))
        amp = float(np.nanmax(x[finite]) - np.nanmin(x[finite]))
        l2 = float(math.sqrt(np.nansum(w * x ** 2)))
        imax = int(np.nanargmax(np.where(finite, x, np.nan)))
        axis_lat = float(target_lat[imax])

        pos = np.where(finite, np.maximum(x, 0.0), np.nan)
        pos_weight = float(np.nansum(w * pos))
        if pos_weight > 1e-12:
            centroid = float(np.nansum(w * pos * target_lat) / pos_weight)
            spread = float(math.sqrt(max(np.nansum(w * pos * (target_lat - centroid) ** 2) / pos_weight, 0.0)))
        else:
            absx = np.where(finite, np.abs(x), np.nan)
            abs_weight = max(float(np.nansum(w * absx)), 1e-12)
            centroid = float(np.nansum(w * absx * target_lat) / abs_weight)
            spread = float(math.sqrt(max(np.nansum(w * absx * (target_lat - centroid) ** 2) / abs_weight, 0.0)))

        south = (target_lat >= 25) & (target_lat < 35) & finite
        north = (target_lat >= 35) & (target_lat <= 45) & finite
        ns = float(np.nanmean(x[north]) - np.nanmean(x[south])) if north.any() and south.any() else np.nan
        if np.isfinite(std) and std > 1e-12:
            z = (x[finite] - np.nanmean(x[finite])) / (np.nanstd(x[finite]) + 1e-12)
            skew = float(np.nanmean(z ** 3))
        else:
            skew = np.nan
        rows.append({
            "day": d,
            "profile_valid": True,
            "profile_n_finite": int(finite.sum()),
            "profile_mean": mean,
            "profile_spatial_variance": var,
            "profile_spatial_std": std,
            "profile_amplitude": amp,
            "profile_weighted_l2_norm": l2,
            "axis_lat": axis_lat,
            "centroid_lat": centroid,
            "spread": spread,
            "width_proxy": spread,
            "NS_contrast": ns,
            "skewness": skew,
        })
        valid_flags.append(True)
    df = pd.DataFrame(rows)

    # Daily speed based on profile difference. If either adjacent day is all-NaN,
    # preserve NaN rather than pretending the speed is zero.
    speeds = [np.nan]
    for d in range(1, prof.shape[0]):
        x0 = prof[d - 1, :].astype(float)
        x1 = prof[d, :].astype(float)
        finite = np.isfinite(x0) & np.isfinite(x1)
        if finite.sum() < 2:
            speeds.append(np.nan)
            continue
        w = np.where(finite, base_w, 0.0)
        w = w / max(float(np.nansum(w)), 1e-12)
        diff = x1 - x0
        speeds.append(float(math.sqrt(np.nansum(w * diff ** 2))))
    df["raw_daily_speed"] = speeds
    return df


def _region_2d_metrics(region_clim: np.ndarray, lat_sel: np.ndarray) -> pd.DataFrame:
    """Metrics from climatological 2D region: region_clim shape (day, lat, lon).

    All-NaN boundary days are emitted as invalid rows instead of crashing in
    nanmax/nanmin.
    """
    lat_w = np.cos(np.deg2rad(lat_sel))
    lat_w = np.where(np.isfinite(lat_w) & (lat_w > 0), lat_w, 0.0)
    base_weights = lat_w[:, None] * np.ones((len(lat_sel), region_clim.shape[2]))
    rows = []
    for d in range(region_clim.shape[0]):
        x = region_clim[d, :, :].astype(float)
        finite = np.isfinite(x)
        if finite.sum() < 2:
            rows.append({
                "day": d,
                "region2d_valid": False,
                "region2d_n_finite": int(finite.sum()),
                "region2d_mean": np.nan,
                "region2d_spatial_variance": np.nan,
                "region2d_spatial_std": np.nan,
                "region2d_amplitude": np.nan,
                "region2d_weighted_l2_norm": np.nan,
            })
            continue
        weights = np.where(finite, base_weights, 0.0)
        weights = weights / max(float(np.nansum(weights)), 1e-12)
        mean = float(np.nansum(weights * x))
        anom = np.where(finite, x - mean, np.nan)
        var = float(np.nansum(weights * anom ** 2))
        std = math.sqrt(max(var, 0.0))
        rows.append({
            "day": d,
            "region2d_valid": True,
            "region2d_n_finite": int(finite.sum()),
            "region2d_mean": mean,
            "region2d_spatial_variance": var,
            "region2d_spatial_std": std,
            "region2d_amplitude": float(np.nanmax(x[finite]) - np.nanmin(x[finite])),
            "region2d_weighted_l2_norm": float(math.sqrt(np.nansum(weights * x ** 2))),
        })
    return pd.DataFrame(rows)


def _add_rank_flags(df: pd.DataFrame, cfg: JePhysicalVarianceAuditConfig) -> pd.DataFrame:
    out = df.copy()
    mask = (out["day"] >= cfg.full_start_day) & (out["day"] <= cfg.full_end_day)
    for col in ["profile_spatial_std", "profile_spatial_variance", "region2d_spatial_std", "region2d_spatial_variance"]:
        if col not in out.columns:
            continue
        vals = out.loc[mask, col].astype(float)
        valid_vals = vals[np.isfinite(vals)]
        out[col + "_qrank"] = np.nan
        out[col + "_low_q10_flag"] = False
        if len(valid_vals) == 0:
            continue
        ranks = vals.rank(pct=True, method="average")
        out.loc[mask, col + "_qrank"] = ranks.values
        q = float(valid_vals.quantile(cfg.low_quantile))
        out.loc[mask, col + "_low_q10_flag"] = out.loc[mask, col] <= q
    return out


def _window_mean(df: pd.DataFrame, col: str, start: int, end: int) -> float:
    m = (df["day"] >= start) & (df["day"] <= end)
    return float(df.loc[m, col].mean())


def _window_summary(metrics: pd.DataFrame, cfg: JePhysicalVarianceAuditConfig) -> pd.DataFrame:
    windows = {
        "early_pre_day20_29": (cfg.early_pre_start, cfg.early_pre_end),
        "early_core_day26_33": (cfg.early_core_start, cfg.early_core_end),
        "low_variance_day30_34": (cfg.low_variance_start, cfg.low_variance_end),
        "early_post_day35_39": (cfg.early_post_start, cfg.early_post_end),
        "W45_day40_48": (cfg.w45_start, cfg.w45_end),
        "late_main_day40_52": (cfg.late_main_start, cfg.late_main_end),
    }
    rows = []
    cols = [
        "profile_mean", "profile_spatial_variance", "profile_spatial_std", "profile_amplitude", "profile_weighted_l2_norm",
        "axis_lat", "centroid_lat", "spread", "NS_contrast", "region2d_mean", "region2d_spatial_variance",
        "region2d_spatial_std", "region2d_amplitude"
    ]
    for name, (s, e) in windows.items():
        row = {"window": name, "start_day": s, "end_day": e}
        for c in cols:
            if c in metrics.columns:
                row[c + "_mean"] = _window_mean(metrics, c, s, e)
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_window_diffs(profiles: np.ndarray, region: np.ndarray, target_lat: np.ndarray, lat_sel: np.ndarray, cfg: JePhysicalVarianceAuditConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_seed)
    n_year = profiles.shape[0]
    rows = []
    min_rows = []
    comparisons = [
        ("low_vs_pre", cfg.low_variance_start, cfg.low_variance_end, cfg.early_pre_start, cfg.early_pre_end),
        ("low_vs_post", cfg.low_variance_start, cfg.low_variance_end, cfg.early_post_start, cfg.early_post_end),
        ("low_vs_W45", cfg.low_variance_start, cfg.low_variance_end, cfg.w45_start, cfg.w45_end),
        ("early_core_vs_late", cfg.early_core_start, cfg.early_core_end, cfg.late_main_start, cfg.late_main_end),
    ]
    n_boot = cfg.n_bootstrap
    for b in range(n_boot):
        idx = rng.integers(0, n_year, size=n_year)
        prof_clim = np.nanmean(profiles[idx, :, :], axis=0)
        reg_clim = np.nanmean(region[idx, :, :, :], axis=0)
        met = _profile_metrics(prof_clim, target_lat).merge(_region_2d_metrics(reg_clim, lat_sel), on="day", how="left")
        # Min day in the audit range. Boundary NaN days or all-invalid bootstrap
        # samples must not crash idxmin or be interpreted as valid minima.
        rmask = (met["day"] >= cfg.min_window_start) & (met["day"] <= cfg.min_window_end)
        sub = met.loc[rmask]

        p_valid = sub[np.isfinite(sub["profile_spatial_std"].astype(float))]
        if len(p_valid) > 0:
            p_min_day = int(p_valid.loc[p_valid["profile_spatial_std"].idxmin(), "day"])
            p_min_in = bool(cfg.low_variance_start <= p_min_day <= cfg.low_variance_end)
        else:
            p_min_day = np.nan
            p_min_in = False

        r_valid = sub[np.isfinite(sub["region2d_spatial_std"].astype(float))]
        if len(r_valid) > 0:
            r_min_day = int(r_valid.loc[r_valid["region2d_spatial_std"].idxmin(), "day"])
            r_min_in = bool(cfg.low_variance_start <= r_min_day <= cfg.low_variance_end)
        else:
            r_min_day = np.nan
            r_min_in = False

        min_rows.append({
            "bootstrap_id": b,
            "profile_std_min_day_day20_40": p_min_day,
            "region2d_std_min_day_day20_40": r_min_day,
            "profile_min_in_day30_34": p_min_in,
            "region2d_min_in_day30_34": r_min_in,
        })
        for comp, a_s, a_e, b_s, b_e in comparisons:
            for col in ["profile_spatial_std", "profile_spatial_variance", "region2d_spatial_std", "region2d_spatial_variance", "profile_amplitude", "region2d_amplitude", "profile_mean", "region2d_mean"]:
                a_val = _window_mean(met, col, a_s, a_e)
                b_val = _window_mean(met, col, b_s, b_e)
                # delta is first window minus second window. Negative means low/early smaller than comparator.
                rows.append({
                    "bootstrap_id": b,
                    "comparison": comp,
                    "metric": col,
                    "first_window_start": a_s,
                    "first_window_end": a_e,
                    "second_window_start": b_s,
                    "second_window_end": b_e,
                    "first_minus_second": a_val - b_val,
                })
    samples = pd.DataFrame(rows)
    mins = pd.DataFrame(min_rows)
    return samples, mins


def _summarize_bootstrap(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comp, metric), g in samples.groupby(["comparison", "metric"]):
        vals = g["first_minus_second"].astype(float).values
        rows.append({
            "comparison": comp,
            "metric": metric,
            "median": float(np.nanmedian(vals)),
            "q025": float(np.nanquantile(vals, 0.025)),
            "q05": float(np.nanquantile(vals, 0.05)),
            "q95": float(np.nanquantile(vals, 0.95)),
            "q975": float(np.nanquantile(vals, 0.975)),
            "P_first_less_than_second": float(np.nanmean(vals < 0)),
            "P_first_greater_than_second": float(np.nanmean(vals > 0)),
            "decision": "first_lower_supported" if np.nanquantile(vals, 0.975) < 0 else ("first_higher_supported" if np.nanquantile(vals, 0.025) > 0 else ("first_lower_tendency" if np.nanmean(vals < 0) >= 0.80 else ("first_higher_tendency" if np.nanmean(vals > 0) >= 0.80 else "unresolved")))
        })
    return pd.DataFrame(rows)


def _summarize_minima(mins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, col in [("profile_spatial_std", "profile_std_min_day_day20_40"), ("region2d_spatial_std", "region2d_std_min_day_day20_40")]:
        vals = mins[col].astype(float).values
        flag_col = "profile_min_in_day30_34" if kind == "profile_spatial_std" else "region2d_min_in_day30_34"
        rows.append({
            "metric": kind,
            "min_day_median": float(np.nanmedian(vals)),
            "min_day_q025": float(np.nanquantile(vals, 0.025)),
            "min_day_q975": float(np.nanquantile(vals, 0.975)),
            "P_min_in_day30_34": float(mins[flag_col].mean()),
            "decision": "minimum_day30_34_supported" if float(mins[flag_col].mean()) >= 0.80 else ("minimum_day30_34_tendency" if float(mins[flag_col].mean()) >= 0.50 else "minimum_not_specific_to_day30_34")
        })
    return pd.DataFrame(rows)


def _decision(metrics: pd.DataFrame, bs_summary: pd.DataFrame, min_summary: pd.DataFrame, cfg: JePhysicalVarianceAuditConfig) -> pd.DataFrame:
    rows = []
    # observed low norm/variance flags around day30-34
    low_mask = (metrics["day"] >= cfg.low_variance_start) & (metrics["day"] <= cfg.low_variance_end)
    profile_low_flags = int(metrics.loc[low_mask, "profile_spatial_std_low_q10_flag"].sum()) if "profile_spatial_std_low_q10_flag" in metrics else 0
    region_low_flags = int(metrics.loc[low_mask, "region2d_spatial_std_low_q10_flag"].sum()) if "region2d_spatial_std_low_q10_flag" in metrics else 0
    rows.append({"check_item": "observed_profile_low_variance_days_day30_34", "result": profile_low_flags, "decision": "low_variance_present" if profile_low_flags >= 2 else "not_strong", "evidence": "number of day30-34 days below day0-70 q10 profile spatial std", "risk": "low profile variance can amplify shape-normalized pattern changes"})
    rows.append({"check_item": "observed_region2d_low_variance_days_day30_34", "result": region_low_flags, "decision": "low_variance_present" if region_low_flags >= 2 else "not_strong", "evidence": "number of day30-34 days below day0-70 q10 2D spatial std", "risk": "2D variance test checks whether profile low norm reflects broader regional flattening"})
    # bootstrap low vs pre/post/W45
    for comp in ["low_vs_pre", "low_vs_post", "low_vs_W45"]:
        for metric in ["profile_spatial_std", "region2d_spatial_std"]:
            sub = bs_summary[(bs_summary["comparison"] == comp) & (bs_summary["metric"] == metric)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({"check_item": f"bootstrap_{comp}_{metric}", "result": r["median"], "decision": r["decision"], "evidence": f"P_first_less_than_second={r['P_first_less_than_second']:.3f}; q025={r['q025']:.3g}; q975={r['q975']:.3g}", "risk": "supports low-variance physical interpretation only if first_lower_supported/tendency"})
    for _, r in min_summary.iterrows():
        rows.append({"check_item": f"bootstrap_minimum_{r['metric']}", "result": r["P_min_in_day30_34"], "decision": r["decision"], "evidence": f"min day median={r['min_day_median']}; q025={r['min_day_q025']}; q975={r['min_day_q975']}", "risk": "tests whether day30-34 is specifically a minimum period"})
    # final type
    profile_low_supported = any((bs_summary["comparison"].isin(["low_vs_pre", "low_vs_post"])) & (bs_summary["metric"] == "profile_spatial_std") & (bs_summary["decision"].isin(["first_lower_supported", "first_lower_tendency"])))
    region_low_supported = any((bs_summary["comparison"].isin(["low_vs_pre", "low_vs_post"])) & (bs_summary["metric"] == "region2d_spatial_std") & (bs_summary["decision"].isin(["first_lower_supported", "first_lower_tendency"])))
    p_min = float(min_summary.loc[min_summary["metric"] == "profile_spatial_std", "P_min_in_day30_34"].iloc[0]) if not min_summary.empty else np.nan
    r_min = float(min_summary.loc[min_summary["metric"] == "region2d_spatial_std", "P_min_in_day30_34"].iloc[0]) if not min_summary.empty else np.nan
    if profile_low_supported and region_low_supported and (p_min >= 0.5 or r_min >= 0.5):
        final_type = "profile_and_2D_variance_dip_supported"
    elif profile_low_supported:
        final_type = "profile_variance_dip_supported"
    elif profile_low_flags >= 2:
        final_type = "observed_profile_variance_dip_only"
    else:
        final_type = "variance_dip_unresolved"
    rows.append({"check_item": "final_physical_variance_status", "result": final_type, "decision": final_type, "evidence": f"profile_low_supported={profile_low_supported}; region_low_supported={region_low_supported}; P_profile_min_day30_34={p_min:.3f}; P_region_min_day30_34={r_min:.3f}", "risk": "final status describes physical variance evidence, not causal mechanism"})
    return pd.DataFrame(rows)


def _plot_outputs(out_dir: Path, metrics: pd.DataFrame, cfg: JePhysicalVarianceAuditConfig) -> None:
    if cfg.skip_figures or plt is None:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Variance/std plot
    fig, ax = plt.subplots(figsize=(10, 5))
    m = (metrics["day"] >= cfg.full_start_day) & (metrics["day"] <= cfg.full_end_day)
    ax.plot(metrics.loc[m, "day"], metrics.loc[m, "profile_spatial_std"], label="profile spatial std")
    if "region2d_spatial_std" in metrics:
        ax.plot(metrics.loc[m, "day"], metrics.loc[m, "region2d_spatial_std"], label="2D regional spatial std")
    ax.axvspan(cfg.low_variance_start, cfg.low_variance_end, alpha=0.2, label="day30-34")
    ax.axvspan(cfg.w45_start, cfg.w45_end, alpha=0.15, label="W45")
    ax.set_xlabel("day")
    ax.set_ylabel("spatial std")
    ax.set_title("Je spatial variance / low-norm physical audit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "Je_spatial_variance_timeseries_v7_z_c.png", dpi=180)
    plt.close(fig)
    # Mean vs variance plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics.loc[m, "day"], metrics.loc[m, "profile_mean"], label="profile mean")
    ax2 = ax.twinx()
    ax2.plot(metrics.loc[m, "day"], metrics.loc[m, "profile_spatial_std"], linestyle="--", label="profile spatial std")
    ax.axvspan(cfg.low_variance_start, cfg.low_variance_end, alpha=0.2)
    ax.axvspan(cfg.w45_start, cfg.w45_end, alpha=0.15)
    ax.set_xlabel("day")
    ax.set_ylabel("profile mean")
    ax2.set_ylabel("profile spatial std")
    ax.set_title("Je mean vs spatial-structure variance")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "Je_mean_vs_variance_v7_z_c.png", dpi=180)
    plt.close(fig)


def _write_summary(out_dir: Path, decision: pd.DataFrame, window_summary: pd.DataFrame, bs_summary: pd.DataFrame, min_summary: pd.DataFrame, cfg: JePhysicalVarianceAuditConfig) -> None:
    final_row = decision[decision["check_item"] == "final_physical_variance_status"]
    final_status = str(final_row["decision"].iloc[0]) if not final_row.empty else "unknown"
    lines = []
    lines.append("# Je Physical Variance Audit for W45 (V7-z-c)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Test whether the Je day30-34 low shape-norm episode can be physically described as a decrease in spatial variance / flattened profile structure, rather than only as a numerical normalization artifact.")
    lines.append("")
    lines.append("## Final status")
    lines.append(f"- `{final_status}`")
    lines.append("")
    lines.append("## Key interpretation")
    if "supported" in final_status:
        lines.append("- The day30-34 period has physical support as a low spatial-variance / flattened-structure episode in Je. This supports the interpretation that the shape-pattern peak is amplified by a real low-variance background, not produced from nothing.")
    elif "observed" in final_status:
        lines.append("- The observed climatological curve shows a low-variance episode, but bootstrap support is insufficient or incomplete. Treat as a physical tendency, not a hard conclusion.")
    else:
        lines.append("- The variance-dip interpretation is unresolved in this audit. Do not use it as physical evidence without further testing.")
    lines.append("")
    lines.append("## Window means")
    for _, r in window_summary.iterrows():
        pstd = r.get("profile_spatial_std_mean", np.nan)
        rstd = r.get("region2d_spatial_std_mean", np.nan)
        lines.append(f"- {r['window']} ({int(r['start_day'])}-{int(r['end_day'])}): profile_std={pstd:.4g}, region2d_std={rstd:.4g}")
    lines.append("")
    lines.append("## Bootstrap highlights")
    for _, r in bs_summary.iterrows():
        if r["comparison"] in ["low_vs_pre", "low_vs_post", "low_vs_W45"] and r["metric"] in ["profile_spatial_std", "region2d_spatial_std"]:
            lines.append(f"- {r['comparison']} / {r['metric']}: median delta={r['median']:.4g}, q025={r['q025']:.4g}, q975={r['q975']:.4g}, P(first<second)={r['P_first_less_than_second']:.3f}, decision={r['decision']}")
    lines.append("")
    lines.append("## Minimum-day bootstrap")
    for _, r in min_summary.iterrows():
        lines.append(f"- {r['metric']}: P(min in day30-34)={r['P_min_in_day30_34']:.3f}, median min day={r['min_day_median']}, decision={r['decision']}")
    lines.append("")
    lines.append("## Allowed statement")
    lines.append("- Je day30-34 may be described as a low spatial-variance / flattened-profile episode only if the final status is supported or observed-profile-only; otherwise keep it as an unresolved hypothesis.")
    lines.append("")
    lines.append("## Forbidden statement")
    lines.append("- Do not claim this proves Je causally drives W45 transitions. Do not claim shape-pattern day33 is a confirmed main physical transition equal to the robust raw/profile day46 peak.")
    (out_dir / "Je_physical_variance_audit_summary_v7_z_c.md").write_text("\n".join(lines), encoding="utf-8")


def run_W45_Je_physical_variance_audit_v7_z_c(v7_root: Path | str) -> None:
    v7_root = Path(v7_root)
    cfg = JePhysicalVarianceAuditConfig()
    if os.environ.get("V7Z_JE_AUDIT_C_DEBUG_N_BOOTSTRAP"):
        cfg.n_bootstrap = int(os.environ["V7Z_JE_AUDIT_C_DEBUG_N_BOOTSTRAP"])
    if os.environ.get("V7Z_JE_AUDIT_C_SKIP_FIGURES"):
        cfg.skip_figures = os.environ["V7Z_JE_AUDIT_C_SKIP_FIGURES"].strip() not in ("0", "false", "False", "")
    if os.environ.get("V7Z_JE_AUDIT_C_SAVE_BY_YEAR"):
        cfg.save_by_year_metrics = os.environ["V7Z_JE_AUDIT_C_SAVE_BY_YEAR"].strip() not in ("0", "false", "False", "")

    out_dir = v7_root / "outputs" / cfg.output_tag
    log_dir = v7_root / "logs" / cfg.output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    _log("[1/7] Load smoothed fields and resolve u200/lat/lon")
    data, smoothed_path, input_audit = _load_npz(v7_root, cfg)
    try:
        arr, lat, lon, years, key_audit = _resolve_required_arrays(data, cfg)
        input_audit = pd.concat([input_audit, key_audit], ignore_index=True)
    except Exception as exc:
        input_audit["error"] = str(exc)
        input_audit.to_csv(out_dir / "Je_physical_variance_input_audit_v7_z_c.csv", index=False)
        raise
    input_audit.to_csv(out_dir / "Je_physical_variance_input_audit_v7_z_c.csv", index=False)

    _log("[2/7] Standardize dimensions and build Je region/profile")
    field, axis_info = _standardize_field_dims(arr, lat, lon, years)
    region, lat_sel, lon_sel = _select_region(field, lat, lon, cfg)
    profiles, target_lat = _build_profiles(region, lat_sel, cfg)
    n_year, n_day = profiles.shape[0], profiles.shape[1]
    if years is None:
        years = np.arange(n_year)
    profile_clim = np.nanmean(profiles, axis=0)
    region_clim = np.nanmean(region, axis=0)

    # Save profile climatology.
    prof_rows = []
    for d in range(profile_clim.shape[0]):
        for j, la in enumerate(target_lat):
            prof_rows.append({"day": d, "lat": float(la), "Je_raw_profile_value": float(profile_clim[d, j])})
    pd.DataFrame(prof_rows).to_csv(out_dir / "Je_raw_profile_climatology_v7_z_c.csv", index=False)

    if cfg.save_by_year_metrics:
        rows = []
        for yi, yr in enumerate(years):
            for d in range(n_day):
                for j, la in enumerate(target_lat):
                    rows.append({"year": int(yr) if np.issubdtype(np.asarray(years).dtype, np.integer) else str(yr), "day": d, "lat": float(la), "Je_raw_profile_value": float(profiles[yi, d, j])})
        pd.DataFrame(rows).to_csv(out_dir / "Je_raw_profile_by_year_v7_z_c.csv", index=False)

    _log("[3/7] Compute profile and 2D physical variance metrics")
    metrics = _profile_metrics(profile_clim, target_lat).merge(_region_2d_metrics(region_clim, lat_sel), on="day", how="left")
    metrics = _add_rank_flags(metrics, cfg)
    metrics.to_csv(out_dir / "Je_physical_variance_timeseries_v7_z_c.csv", index=False)
    window_summary = _window_summary(metrics, cfg)
    window_summary.to_csv(out_dir / "Je_physical_variance_window_summary_v7_z_c.csv", index=False)

    _log(f"[4/7] Run paired year bootstrap for variance-window contrasts (n={cfg.n_bootstrap})")
    samples, mins = _bootstrap_window_diffs(profiles, region, target_lat, lat_sel, cfg)
    # sample file can be large but still manageable; write summary by default and samples for audit.
    samples.to_csv(out_dir / "Je_physical_variance_bootstrap_samples_v7_z_c.csv", index=False)
    mins.to_csv(out_dir / "Je_physical_variance_minimum_bootstrap_samples_v7_z_c.csv", index=False)
    bs_summary = _summarize_bootstrap(samples)
    min_summary = _summarize_minima(mins)
    bs_summary.to_csv(out_dir / "Je_physical_variance_bootstrap_summary_v7_z_c.csv", index=False)
    min_summary.to_csv(out_dir / "Je_physical_variance_minimum_bootstrap_summary_v7_z_c.csv", index=False)

    _log("[5/7] Apply physical-variance decision rules")
    decision = _decision(metrics, bs_summary, min_summary, cfg)
    decision.to_csv(out_dir / "Je_physical_variance_decision_v7_z_c.csv", index=False)

    _log("[6/7] Write figures and markdown summary")
    _plot_outputs(out_dir, metrics, cfg)
    _write_summary(out_dir, decision, window_summary, bs_summary, min_summary, cfg)

    _log("[7/7] Write run metadata")
    run_meta = {
        "version": cfg.version,
        "output_tag": cfg.output_tag,
        "smoothed_fields": str(smoothed_path),
        "axis_info": axis_info,
        "n_year": int(n_year),
        "n_day": int(n_day),
        "target_lat": [float(x) for x in target_lat],
        "je_region": {"lon_min": cfg.je_lon_min, "lon_max": cfg.je_lon_max, "lat_min": cfg.je_lat_min, "lat_max": cfg.je_lat_max},
        "config": asdict(cfg),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    final_status = decision.loc[decision["check_item"] == "final_physical_variance_status", "decision"].iloc[0]
    summary = {"final_physical_variance_status": final_status, "output_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"Done. Output: {out_dir}")
