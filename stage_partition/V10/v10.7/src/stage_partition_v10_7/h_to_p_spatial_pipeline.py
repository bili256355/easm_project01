from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import shutil
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs  # type: ignore
    HAS_CARTOPY = True
except Exception:
    ccrs = None
    HAS_CARTOPY = False


# =============================================================================
# V10.7_l: H_E2 structure -> M_P rainband spatial verification
# =============================================================================
# Method boundary:
# - This is NOT a causal test.
# - This is NOT a full E2->M multivariate mapping test.
# - It verifies the strongest V10.7_k H-source structural candidate:
#   E2 H morphology transition (west extent / zonal width / north edge)
#   against M-stage precipitation rainband structure and spatial composites.
# - It does NOT control away P/V/Je/Jw as covariates.
# =============================================================================


@dataclass
class Settings:
    project_root: Path
    n_perm: int = 1000
    n_boot: int = 500
    group_frac: float = 0.30
    progress: bool = False
    smoothed_fields_path_override: Path | None = None
    version: str = "v10.7_l"
    output_tag: str = "h_e2_to_m_p_spatial_verification_v10_7_l"
    random_seed: int = 20260514

    e2_pre: tuple[int, int] = (27, 31)
    e2_post: tuple[int, int] = (34, 38)
    m_pre: tuple[int, int] = (40, 43)
    m_post: tuple[int, int] = (45, 48)
    m_full: tuple[int, int] = (40, 48)

    h_lat_range: tuple[float, float] = (15.0, 35.0)
    h_lon_range: tuple[float, float] = (110.0, 140.0)
    p_lat_range: tuple[float, float] = (15.0, 35.0)
    p_lon_range: tuple[float, float] = (110.0, 140.0)

    h_metrics_main: tuple[str, ...] = (
        "H_west_extent_lon_transition",
        "H_zonal_width_transition",
        "H_north_edge_lat_transition",
    )
    h_metrics_aux: tuple[str, ...] = (
        "H_strength_transition",
        "H_centroid_lat_transition",
    )
    p_metrics_main: tuple[str, ...] = (
        "P_centroid_lat_transition",
        "P_main_band_share_transition",
        "P_south_band_share_18_24_transition",
        "P_main_minus_south_transition",
        "P_spread_lat_transition",
    )
    modes: tuple[str, ...] = ("anomaly", "local_background_removed", "raw")
    primary_modes: tuple[str, ...] = ("anomaly", "local_background_removed")

    def smoothed_fields_path(self) -> Path:
        if self.smoothed_fields_path_override is not None:
            return self.smoothed_fields_path_override
        return self.project_root / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"

    def output_root(self) -> Path:
        return self.project_root / "stage_partition" / "V10" / "v10.7" / "outputs" / self.output_tag

    def to_dict(self) -> dict[str, Any]:
        def conv(x: Any) -> Any:
            if isinstance(x, Path):
                return str(x)
            if isinstance(x, tuple):
                return [conv(v) for v in x]
            if isinstance(x, list):
                return [conv(v) for v in x]
            if isinstance(x, dict):
                return {str(k): conv(v) for k, v in x.items()}
            return x
        return conv(asdict(self))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(settings: Settings, msg: str) -> None:
    if settings.progress:
        print(f"[V10.7_l] {msg}", flush=True)


def clean_output_root(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for sub in ("tables", "figures", "run_meta"):
        (path / sub).mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_nanmean(a: np.ndarray, axis=None, keepdims: bool = False):
    arr = np.asarray(a, dtype=float)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis, keepdims=keepdims)
    total = np.nansum(arr, axis=axis, keepdims=keepdims)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return np.where(count > 0, mean, np.nan)


def first_key(data: dict[str, np.ndarray], candidates: tuple[str, ...]) -> str | None:
    lower = {k.lower(): k for k in data.keys()}
    for c in candidates:
        if c in data:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_npz(settings: Settings) -> dict[str, Any]:
    path = settings.smoothed_fields_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields file: {path}")
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    lat_key = first_key(data, ("lat", "latitude", "lats"))
    lon_key = first_key(data, ("lon", "longitude", "lons"))
    year_key = first_key(data, ("year", "years"))
    day_key = first_key(data, ("day", "days", "day_index"))
    if lat_key is None or lon_key is None:
        raise KeyError(f"Could not detect lat/lon keys. Available={sorted(data.keys())}")
    return {
        "path": path,
        "data": data,
        "lat": np.asarray(data[lat_key], dtype=float),
        "lon": np.asarray(data[lon_key], dtype=float),
        "lat_key": lat_key,
        "lon_key": lon_key,
        "year_key": year_key,
        "day_key": day_key,
    }


def normalize_field_dims(field: np.ndarray, data: dict[str, np.ndarray], year_key: str | None, day_key: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=float)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected year x day x lat x lon or day x lat x lon, got {field.shape}")
    if year_key is not None and len(np.asarray(data[year_key]).ravel()) == arr.shape[0]:
        years = np.asarray(data[year_key]).ravel()
    else:
        years = np.arange(arr.shape[0], dtype=int)
    if day_key is not None and len(np.asarray(data[day_key]).ravel()) == arr.shape[1]:
        days = np.asarray(data[day_key]).ravel()
    else:
        days = np.arange(arr.shape[1], dtype=int)
    return arr, years, days


def subset_domain(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]):
    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
    lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError(f"No grid points in lat={lat_range}, lon={lon_range}")
    sub = field[:, :, lat_mask, :][:, :, :, lon_mask]
    sub_lat = lat[lat_mask]
    sub_lon = lon[lon_mask]
    order_lat = np.argsort(sub_lat)
    order_lon = np.argsort(sub_lon)
    sub = sub[:, :, order_lat, :][:, :, :, order_lon]
    return sub, sub_lat[order_lat], sub_lon[order_lon]


def day_mask(days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    lo, hi = win
    return (days >= lo) & (days <= hi)


def window_field_mean(field: np.ndarray, days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    m = day_mask(days, win)
    if not np.any(m):
        return np.full((field.shape[0], field.shape[2], field.shape[3]), np.nan)
    return safe_nanmean(field[:, m], axis=1)


def transition_field(field: np.ndarray, days: np.ndarray, pre: tuple[int, int], post: tuple[int, int]) -> np.ndarray:
    return window_field_mean(field, days, post) - window_field_mean(field, days, pre)


def daily_anomaly(field: np.ndarray) -> np.ndarray:
    clim = safe_nanmean(field, axis=0, keepdims=True)
    return field - clim


def local_background_removed_field(field: np.ndarray, days: np.ndarray, event_win: tuple[int, int], bg_win: tuple[int, int]) -> np.ndarray:
    # Removes a local background mean for event states. For transitions, use raw post-pre minus linear background expected change.
    event = day_mask(days, event_win)
    bg = day_mask(days, bg_win) & (~event)
    if not np.any(bg):
        return field * np.nan
    bg_mean = safe_nanmean(field[:, bg], axis=1, keepdims=True)
    return field - bg_mean


def local_background_removed_transition_field(field: np.ndarray, days: np.ndarray, pre: tuple[int, int], post: tuple[int, int], bg_win: tuple[int, int]) -> np.ndarray:
    pre_m = day_mask(days, pre)
    post_m = day_mask(days, post)
    event = pre_m | post_m
    bg = day_mask(days, bg_win) & (~event)
    obs = transition_field(field, days, pre, post)
    if np.sum(bg) < 3:
        return obs * np.nan
    pre_mid = float(np.nanmean(days[pre_m]))
    post_mid = float(np.nanmean(days[post_m]))
    # Fit linear background separately for each year/grid point with vectorized least squares.
    x = days[bg].astype(float)
    x0 = x - x.mean()
    denom = np.sum(x0 ** 2)
    z = field[:, bg, :, :]
    zmean = safe_nanmean(z, axis=1, keepdims=True)
    numer = np.nansum((z - zmean) * x0.reshape(1, -1, 1, 1), axis=1)
    slope = numer / denom if denom > 0 else np.full_like(obs, np.nan)
    expected = slope * (post_mid - pre_mid)
    return obs - expected


def _field_rms_daily(sub: np.ndarray) -> np.ndarray:
    return np.sqrt(safe_nanmean(sub ** 2, axis=(2, 3)))


def _lat_weighted_metrics_daily(sub: np.ndarray, lat: np.ndarray, positive_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    prof = safe_nanmean(sub, axis=3)
    weights = np.maximum(prof, 0.0) if positive_only else np.abs(prof)
    total = np.nansum(weights, axis=2)
    latv = lat.reshape(1, 1, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cent = np.nansum(weights * latv, axis=2) / total
        spread = np.sqrt(np.nansum(weights * (latv - cent[:, :, None]) ** 2, axis=2) / total)
    cent = np.where(total > 1e-12, cent, np.nan)
    spread = np.where(total > 1e-12, spread, np.nan)
    return cent, spread


def _band_share_daily(sub: np.ndarray, lat: np.ndarray, band: tuple[float, float], positive_only: bool = True) -> np.ndarray:
    arr = np.maximum(sub, 0.0) if positive_only else np.abs(sub)
    total = np.nansum(arr, axis=(2, 3))
    lo, hi = sorted(band)
    mask = (lat >= lo) & (lat <= hi)
    band_sum = np.nansum(arr[:, :, mask, :], axis=(2, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = band_sum / total
    return np.where(total > 1e-12, share, np.nan)


def _h_extent_metrics_daily(sub: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_y, n_d = sub.shape[:2]
    west = np.full((n_y, n_d), np.nan)
    width = np.full((n_y, n_d), np.nan)
    north_edge = np.full((n_y, n_d), np.nan)
    south_edge = np.full((n_y, n_d), np.nan)
    for y in range(n_y):
        for d in range(n_d):
            f = sub[y, d]
            if not np.isfinite(f).any():
                continue
            thr = np.nanpercentile(f, 80)
            mask = f >= thr
            if not np.any(mask):
                continue
            lon_active = lon[np.any(mask, axis=0)]
            lat_active = lat[np.any(mask, axis=1)]
            if lon_active.size:
                west[y, d] = float(np.nanmin(lon_active))
                width[y, d] = float(np.nanmax(lon_active) - np.nanmin(lon_active))
            if lat_active.size:
                south_edge[y, d] = float(np.nanmin(lat_active))
                north_edge[y, d] = float(np.nanmax(lat_active))
    return west, width, north_edge, south_edge


def window_mean_metric(metric: np.ndarray, days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    m = day_mask(days, win)
    if not np.any(m):
        return np.full(metric.shape[0], np.nan)
    return safe_nanmean(metric[:, m], axis=1)


def window_transition_metric(metric: np.ndarray, days: np.ndarray, pre: tuple[int, int], post: tuple[int, int]) -> np.ndarray:
    return window_mean_metric(metric, days, post) - window_mean_metric(metric, days, pre)


def build_h_daily_metrics(h_field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    metrics: dict[str, np.ndarray] = {}
    metrics["H_strength"] = _field_rms_daily(h_field)
    cent, _ = _lat_weighted_metrics_daily(h_field, lat, positive_only=False)
    metrics["H_centroid_lat"] = cent
    west, width, north_edge, south_edge = _h_extent_metrics_daily(h_field, lat, lon)
    metrics["H_west_extent_lon"] = west
    metrics["H_zonal_width"] = width
    metrics["H_north_edge_lat"] = north_edge
    metrics["H_south_edge_lat"] = south_edge
    return metrics


def build_p_daily_metrics(p_field: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    metrics: dict[str, np.ndarray] = {}
    metrics["P_total_strength"] = _field_rms_daily(p_field)
    cent, spread = _lat_weighted_metrics_daily(p_field, lat, positive_only=True)
    metrics["P_centroid_lat"] = cent
    metrics["P_spread_lat"] = spread
    main = _band_share_daily(p_field, lat, (24.0, 35.0), positive_only=True)
    south = _band_share_daily(p_field, lat, (18.0, 24.0), positive_only=True)
    metrics["P_main_band_share"] = main
    metrics["P_south_band_share_18_24"] = south
    metrics["P_main_minus_south"] = main - south
    return metrics


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xs = pd.Series(x).rank().to_numpy(dtype=float)
    ys = pd.Series(y).rank().to_numpy(dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if np.sum(ok) < 4:
        return np.nan
    return float(np.corrcoef(xs[ok], ys[ok])[0, 1])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 4:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def permutation_p_high_low(metric_by_year: np.ndarray, high_idx: np.ndarray, low_idx: np.ndarray, rng: np.random.Generator, n_perm: int) -> tuple[float, float, float]:
    obs = float(np.nanmean(metric_by_year[high_idx]) - np.nanmean(metric_by_year[low_idx]))
    n_h = int(np.sum(high_idx))
    n_l = int(np.sum(low_idx))
    ok = np.isfinite(metric_by_year)
    vals = metric_by_year[ok]
    if vals.size < n_h + n_l or n_h == 0 or n_l == 0:
        return obs, np.nan, np.nan
    diffs = np.full(n_perm, np.nan)
    for i in range(n_perm):
        perm = rng.permutation(vals.size)
        h = vals[perm[:n_h]]
        l = vals[perm[n_h:n_h+n_l]]
        diffs[i] = np.nanmean(h) - np.nanmean(l)
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1.0) / (np.sum(np.isfinite(diffs)) + 1.0)
    return obs, float(np.nanpercentile(np.abs(diffs), 90)), float(p)


def bootstrap_ci_high_low(metric_by_year: np.ndarray, high_idx: np.ndarray, low_idx: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    hvals = metric_by_year[high_idx]
    lvals = metric_by_year[low_idx]
    hvals = hvals[np.isfinite(hvals)]
    lvals = lvals[np.isfinite(lvals)]
    if hvals.size < 2 or lvals.size < 2:
        return np.nan, np.nan
    diffs = []
    for _ in range(n_boot):
        h = rng.choice(hvals, size=hvals.size, replace=True)
        l = rng.choice(lvals, size=lvals.size, replace=True)
        diffs.append(float(np.nanmean(h) - np.nanmean(l)))
    return float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def composite_field_stats(field_by_year: np.ndarray, high_idx: np.ndarray, low_idx: np.ndarray, rng: np.random.Generator, n_perm: int) -> dict[str, float]:
    high = safe_nanmean(field_by_year[high_idx], axis=0)
    low = safe_nanmean(field_by_year[low_idx], axis=0)
    diff = high - low
    obs_norm = float(np.sqrt(np.nanmean(diff ** 2)))
    obs_abs = float(np.nanmean(np.abs(diff)))
    pos_frac = float(np.nanmean(diff > 0))
    neg_frac = float(np.nanmean(diff < 0))
    n_h = int(np.sum(high_idx)); n_l = int(np.sum(low_idx))
    valid_year = np.array([np.isfinite(field_by_year[i]).any() for i in range(field_by_year.shape[0])])
    vals = field_by_year[valid_year]
    if vals.shape[0] < n_h + n_l or n_h == 0 or n_l == 0:
        p = np.nan
        null90 = np.nan
    else:
        norms = []
        for _ in range(n_perm):
            perm = rng.permutation(vals.shape[0])
            ph = safe_nanmean(vals[perm[:n_h]], axis=0)
            pl = safe_nanmean(vals[perm[n_h:n_h+n_l]], axis=0)
            norms.append(float(np.sqrt(np.nanmean((ph - pl) ** 2))))
        norms = np.asarray(norms)
        p = float((np.sum(norms >= obs_norm) + 1.0) / (np.sum(np.isfinite(norms)) + 1.0))
        null90 = float(np.nanpercentile(norms, 90))
    return {
        "diff_norm": obs_norm,
        "diff_abs_mean": obs_abs,
        "positive_area_fraction": pos_frac,
        "negative_area_fraction": neg_frac,
        "permutation_null_p90_norm": null90,
        "permutation_p": p,
    }


def make_groups(values: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    n = int(np.sum(ok))
    k = max(2, int(math.floor(n * frac)))
    order = np.argsort(x[ok])
    idx_ok = np.where(ok)[0]
    low = np.zeros_like(ok, dtype=bool)
    high = np.zeros_like(ok, dtype=bool)
    low[idx_ok[order[:k]]] = True
    high[idx_ok[order[-k:]]] = True
    mid = ok & (~low) & (~high)
    return high, low, mid


def build_mode_fields(raw_h: np.ndarray, raw_p: np.ndarray, days: np.ndarray, settings: Settings) -> dict[str, dict[str, np.ndarray]]:
    fields: dict[str, dict[str, np.ndarray]] = {}
    fields["raw"] = {"H": raw_h, "P": raw_p}
    fields["anomaly"] = {"H": daily_anomaly(raw_h), "P": daily_anomaly(raw_p)}
    # For daily metrics, local background state removal around E2/M. For transition fields, use dedicated helper later.
    fields["local_background_removed"] = {
        "H": local_background_removed_field(raw_h, days, settings.e2_pre[0:1] + settings.e2_post[1:2] if False else (27, 38), (18, 48)),
        "P": local_background_removed_field(raw_p, days, settings.m_full, (30, 60)),
    }
    return fields


def prepare_metrics(settings: Settings, npz: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = npz["data"]
    h_key = first_key(data, ("z500_smoothed", "z500", "H", "hgt500", "geopotential_500", "zg500"))
    p_key = first_key(data, ("precip_smoothed", "precip", "P", "pr", "rain", "tp"))
    if h_key is None or p_key is None:
        raise KeyError(f"Missing H/P fields. H={h_key}, P={p_key}, available={sorted(data.keys())}")
    h_raw, years, days = normalize_field_dims(data[h_key], data, npz["year_key"], npz["day_key"])
    p_raw, years_p, days_p = normalize_field_dims(data[p_key], data, npz["year_key"], npz["day_key"])
    if len(years) != len(years_p) or len(days) != len(days_p):
        raise ValueError("H and P dimensions do not match")
    h_sub, h_lat, h_lon = subset_domain(h_raw, npz["lat"], npz["lon"], settings.h_lat_range, settings.h_lon_range)
    p_sub, p_lat, p_lon = subset_domain(p_raw, npz["lat"], npz["lon"], settings.p_lat_range, settings.p_lon_range)

    audit_rows = [
        {"object": "H", "source_field": h_key, "lat_range": settings.h_lat_range, "lon_range": settings.h_lon_range, "loaded": True, "notes": "H structure from z500 domain"},
        {"object": "P", "source_field": p_key, "lat_range": settings.p_lat_range, "lon_range": settings.p_lon_range, "loaded": True, "notes": "P rainband structure from precip domain"},
    ]
    audit_df = pd.DataFrame(audit_rows)

    # Mode fields.
    fields_by_mode: dict[str, dict[str, np.ndarray]] = {
        "raw": {"H": h_sub, "P": p_sub},
        "anomaly": {"H": daily_anomaly(h_sub), "P": daily_anomaly(p_sub)},
        "local_background_removed": {
            "H": local_background_removed_field(h_sub, days, (27, 38), (18, 48)),
            "P": local_background_removed_field(p_sub, days, settings.m_full, (30, 60)),
        },
    }

    rows = []
    p_spatial: dict[str, dict[str, np.ndarray]] = {}
    profiles: dict[str, dict[str, np.ndarray]] = {}

    for mode, obj_fields in fields_by_mode.items():
        h_metrics_daily = build_h_daily_metrics(obj_fields["H"], h_lat, h_lon)
        p_metrics_daily = build_p_daily_metrics(obj_fields["P"], p_lat)
        # Local-background transition metrics use field-level linear detrended transition for P maps; daily metrics already state residuals.
        p_full_field = window_field_mean(obj_fields["P"], days, settings.m_full)
        if mode == "local_background_removed":
            p_transition_field = local_background_removed_transition_field(p_sub, days, settings.m_pre, settings.m_post, (30, 60))
        else:
            p_transition_field = transition_field(obj_fields["P"], days, settings.m_pre, settings.m_post)
        p_spatial[mode] = {"P_M_full_field": p_full_field, "P_M_transition_field": p_transition_field}
        profiles[mode] = {
            "P_M_full_profile": safe_nanmean(p_full_field, axis=2),
            "P_M_transition_profile": safe_nanmean(p_transition_field, axis=2),
        }
        # H transitions.
        for base_name, daily in h_metrics_daily.items():
            vals = window_transition_metric(daily, days, settings.e2_pre, settings.e2_post)
            rows.extend({"year": y, "mode": mode, "object": "H", "metric": f"{base_name}_transition", "value": v} for y, v in zip(years, vals))
        # P transitions and full states.
        for base_name, daily in p_metrics_daily.items():
            vals_t = window_transition_metric(daily, days, settings.m_pre, settings.m_post)
            vals_f = window_mean_metric(daily, days, settings.m_full)
            rows.extend({"year": y, "mode": mode, "object": "P", "metric": f"{base_name}_transition", "value": v} for y, v in zip(years, vals_t))
            rows.extend({"year": y, "mode": mode, "object": "P", "metric": f"{base_name}_M_full", "value": v} for y, v in zip(years, vals_f))
    metric_df = pd.DataFrame(rows)
    context = {"years": years, "days": days, "p_lat": p_lat, "p_lon": p_lon, "p_spatial": p_spatial, "profiles": profiles, "audit_df": audit_df}
    return metric_df, context


def pivot_metric(metric_df: pd.DataFrame, mode: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    sub = metric_df[(metric_df["mode"] == mode) & (metric_df["metric"] == metric)].sort_values("year")
    return sub["year"].to_numpy(), sub["value"].to_numpy(dtype=float)


def plot_map_triplet(lat: np.ndarray, lon: np.ndarray, high: np.ndarray, low: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    diff = high - low
    vals = np.concatenate([high.ravel(), low.ravel(), diff.ravel()])
    vmax = np.nanpercentile(np.abs(vals), 98) if np.isfinite(vals).any() else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig = plt.figure(figsize=(13, 4))
    arrays = [(high, "High H"), (low, "Low H"), (diff, "High - Low")]
    for i, (arr, subtitle) in enumerate(arrays, 1):
        if HAS_CARTOPY:
            ax = fig.add_subplot(1, 3, i, projection=ccrs.PlateCarree())
            mesh = ax.pcolormesh(lon, lat, arr, transform=ccrs.PlateCarree(), shading="auto", vmin=-vmax, vmax=vmax)
            # Avoid Natural Earth download during unattended runs; keep Cartopy projection without coastlines.
            try:
                pass
            except Exception:
                pass
        else:
            ax = fig.add_subplot(1, 3, i)
            mesh = ax.pcolormesh(lon, lat, arr, shading="auto", vmin=-vmax, vmax=vmax)
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_title(subtitle, fontsize=9)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_profile(lat: np.ndarray, high: np.ndarray, low: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(high, lat, label="High H")
    ax.plot(low, lat, label="Low H")
    ax.plot(high - low, lat, label="High-Low", linestyle="--")
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("P profile value")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scatter(x: np.ndarray, y: np.ndarray, years: np.ndarray, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y)
    for xi, yi, yr in zip(x, y, years):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(str(yr), (xi, yi), fontsize=6, alpha=0.7)
    r = spearman_corr(x, y)
    ax.set_title(f"{title}\nSpearman r={r:.3f}" if np.isfinite(r) else title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_analysis(settings: Settings, metric_df: pd.DataFrame, context: dict[str, Any], out: Path) -> dict[str, Any]:
    rng = np.random.default_rng(settings.random_seed)
    years = context["years"]
    p_lat = context["p_lat"]
    p_lon = context["p_lon"]
    group_rows = []
    p_metric_rows = []
    spatial_rows = []
    influence_rows = []
    route_rows = []
    direction_rows = []

    direction_rows.extend([
        {"metric": "H_west_extent_lon_transition", "definition": "post_E2 westernmost active lon - pre_E2 westernmost active lon", "larger_value_means": "western active edge is farther east / less westward extent, if lon increases eastward", "smaller_value_means": "western active edge is farther west / stronger westward extension", "transition_positive_means": "west edge shifts eastward or westward extent weakens", "transition_negative_means": "west edge shifts westward or westward extent strengthens", "physical_interpretation_allowed": "Only after confirming longitude convention and active-area threshold behavior."},
        {"metric": "H_zonal_width_transition", "definition": "post_E2 zonal active width - pre_E2 zonal active width", "larger_value_means": "wider active H domain", "smaller_value_means": "narrower active H domain", "transition_positive_means": "zonal expansion", "transition_negative_means": "zonal contraction", "physical_interpretation_allowed": "Candidate morphology indicator, not direct WPSH index."},
        {"metric": "H_north_edge_lat_transition", "definition": "post_E2 north active edge - pre_E2 north active edge", "larger_value_means": "active north edge farther north", "smaller_value_means": "active north edge farther south", "transition_positive_means": "north edge shifts north", "transition_negative_means": "north edge shifts south", "physical_interpretation_allowed": "Candidate edge-position indicator."},
    ])

    # Keep first implementation focused on the three V10.7_k-supported main H morphology metrics.
    h_metrics = list(settings.h_metrics_main)
    for mode in settings.primary_modes:
        for h_metric in h_metrics:
            _log(settings, f"stage 3/7 analyze mode={mode} h_metric={h_metric}")
            yrs, h_vals = pivot_metric(metric_df, mode, h_metric)
            if len(yrs) == 0:
                continue
            high_idx, low_idx, mid_idx = make_groups(h_vals, settings.group_frac)
            for yi, hv, grp in zip(yrs, h_vals, np.where(high_idx, "high", np.where(low_idx, "low", "middle"))):
                if np.isfinite(hv):
                    group_rows.append({"mode": mode, "h_metric": h_metric, "group_fraction": settings.group_frac, "year": yi, "group": grp, "h_metric_value": hv, "h_metric_z": (hv - np.nanmean(h_vals)) / (np.nanstd(h_vals) if np.nanstd(h_vals) > 1e-12 else np.nan)})
            # P metrics.
            for p_metric in settings.p_metrics_main:
                _, p_vals = pivot_metric(metric_df, mode, p_metric)
                if p_vals.size != h_vals.size:
                    continue
                obs, null90, pp = permutation_p_high_low(p_vals, high_idx, low_idx, rng, settings.n_perm)
                ci_lo, ci_hi = bootstrap_ci_high_low(p_vals, high_idx, low_idx, rng, settings.n_boot)
                r = spearman_corr(h_vals, p_vals)
                pr = pearson_corr(h_vals, p_vals)
                p_metric_rows.append({
                    "mode": mode, "h_metric": h_metric, "p_metric": p_metric,
                    "spearman_r": r, "pearson_r": pr,
                    "high_mean": float(np.nanmean(p_vals[high_idx])), "low_mean": float(np.nanmean(p_vals[low_idx])),
                    "high_minus_low": obs,
                    "permutation_null_p90_absdiff": null90,
                    "permutation_p": pp,
                    "bootstrap_ci_low": ci_lo, "bootstrap_ci_high": ci_hi,
                    "direction_consistent_with_v10_7_k": "manual_check_required",
                })
                # Influence / LOO.
                for i, yr in enumerate(yrs):
                    mask = np.ones_like(h_vals, dtype=bool)
                    mask[i] = False
                    influence_rows.append({
                        "mode": mode, "h_metric": h_metric, "p_metric": p_metric, "year": yr,
                        "h_value": h_vals[i], "p_value": p_vals[i],
                        "is_high_h": bool(high_idx[i]), "is_low_h": bool(low_idx[i]),
                        "leave_one_year_out_spearman": spearman_corr(h_vals[mask], p_vals[mask]),
                    })
            # Spatial fields and profiles.
            for field_name in ("P_M_full_field", "P_M_transition_field"):
                field = context["p_spatial"][mode][field_name]
                high_field = safe_nanmean(field[high_idx], axis=0)
                low_field = safe_nanmean(field[low_idx], axis=0)
                stats = composite_field_stats(field, high_idx, low_idx, rng, settings.n_perm)
                spatial_rows.append({"mode": mode, "h_metric": h_metric, "field_mode": field_name, **stats})
                short = field_name.replace("P_M_", "")
                plot_map_triplet(p_lat, p_lon, high_field, low_field, f"{mode} {h_metric} {short}", out / "figures" / f"p_spatial_{mode}_{h_metric}_{short}_v10_7_l.png")
                profile_name = "P_M_full_profile" if field_name == "P_M_full_field" else "P_M_transition_profile"
                prof = context["profiles"][mode][profile_name]
                high_prof = safe_nanmean(prof[high_idx], axis=0)
                low_prof = safe_nanmean(prof[low_idx], axis=0)
                plot_profile(p_lat, high_prof, low_prof, f"{mode} {h_metric} {short}", out / "figures" / f"p_profile_{mode}_{h_metric}_{short}_v10_7_l.png")
            # Scatter for most central P metrics.
            for p_metric in ("P_centroid_lat_transition", "P_main_band_share_transition", "P_main_minus_south_transition"):
                _, p_vals = pivot_metric(metric_df, mode, p_metric)
                if p_vals.size == h_vals.size:
                    plot_scatter(h_vals, p_vals, yrs, f"{mode}: {h_metric} vs {p_metric}", h_metric, p_metric, out / "figures" / f"scatter_{mode}_{h_metric}_to_{p_metric}_v10_7_l.png")

    group_df = pd.DataFrame(group_rows)
    p_metric_df = pd.DataFrame(p_metric_rows)
    spatial_df = pd.DataFrame(spatial_rows)
    influence_df = pd.DataFrame(influence_rows)
    direction_df = pd.DataFrame(direction_rows)

    # Route decision: conservative and only for main H metrics.
    primary = p_metric_df[(p_metric_df["mode"].isin(settings.primary_modes)) & (p_metric_df["h_metric"].isin(settings.h_metrics_main))]
    strong = primary[(primary["permutation_p"] <= 0.10) & (primary["bootstrap_ci_low"] * primary["bootstrap_ci_high"] > 0)]
    hp_strong = strong[strong["p_metric"].isin(("P_centroid_lat_transition", "P_main_band_share_transition", "P_south_band_share_18_24_transition", "P_main_minus_south_transition"))]
    if len(hp_strong) >= 4:
        status = "spatial_metric_support_for_H_to_P_structure_mapping_candidate"
        implication = "Proceed to physical sign interpretation and targeted spatial-profile discussion; not causal proof."
    elif len(hp_strong) > 0:
        status = "partial_metric_support_requires_manual_spatial_review"
        implication = "Keep as candidate; inspect maps/profiles and influence before interpretation."
    else:
        status = "not_supported_at_metric_composite_level"
        implication = "Do not promote H->P structural mapping based on V10.7_l."
    route_rows.append({
        "decision_item": "H_E2_structure_to_M_P_rainband_spatial_verification",
        "status": status,
        "evidence": f"primary significant P rainband metric rows={len(hp_strong)}; total candidate rows={len(strong)}",
        "route_implication": implication,
    })
    route_rows.append({
        "decision_item": "sign_direction_interpretation",
        "status": "manual_direction_audit_required",
        "evidence": "H_west_extent_lon sign depends on longitude convention: smaller west_extent means farther west active edge.",
        "route_implication": "Do not write physical sign explanation before checking direction audit table and maps.",
    })
    route_df = pd.DataFrame(route_rows)

    write_dataframe(group_df, out / "tables" / "h_e2_group_years_v10_7_l.csv")
    write_dataframe(p_metric_df, out / "tables" / "h_group_p_metric_composite_summary_v10_7_l.csv")
    write_dataframe(spatial_df, out / "tables" / "h_group_p_spatial_composite_summary_v10_7_l.csv")
    write_dataframe(influence_df, out / "tables" / "h_to_p_influence_by_year_v10_7_l.csv")
    write_dataframe(direction_df, out / "tables" / "h_metric_direction_audit_v10_7_l.csv")
    write_dataframe(route_df, out / "tables" / "h_e2_to_m_p_spatial_route_decision_v10_7_l.csv")
    return {"route_status": status, "candidate_rows": int(len(hp_strong))}


def write_summary(settings: Settings, out: Path, analysis_meta: dict[str, Any]) -> Path:
    path = out / "summary_h_e2_to_m_p_spatial_verification_v10_7_l.md"
    text = f"""# V10.7_l H_E2 structure → M_P rainband spatial verification

## Task
This audit verifies the narrow V10.7_k candidate that E2/W33 H morphology transitions correspond to M/W45 precipitation rainband structure changes.

## Method boundary
- This is not causal inference.
- This is not a full E2→M multivariate mapping audit.
- It does not control away W45 component objects.
- It tests high-H vs low-H year composites, lat profiles, metric differences, permutation tests, bootstrap CIs, and year influence.

## Settings
- n_perm = {settings.n_perm}
- n_boot = {settings.n_boot}
- group_frac = {settings.group_frac}
- H metrics = {list(settings.h_metrics_main)}
- P metrics = {list(settings.p_metrics_main)}

## Route status
{analysis_meta.get('route_status')}

Candidate P rainband metric rows: {analysis_meta.get('candidate_rows')}

## Key files
- tables/h_e2_group_years_v10_7_l.csv
- tables/h_group_p_metric_composite_summary_v10_7_l.csv
- tables/h_group_p_spatial_composite_summary_v10_7_l.csv
- tables/h_to_p_influence_by_year_v10_7_l.csv
- tables/h_metric_direction_audit_v10_7_l.csv
- tables/h_e2_to_m_p_spatial_route_decision_v10_7_l.csv

## Forbidden interpretations
- Do not state that H causes P.
- Do not state that H controls W45.
- Do not interpret the sign of H_west_extent_lon before checking the direction audit.
- Do not generalize this H→P verification to full W33→W45 mapping.
"""
    path.write_text(text, encoding="utf-8")
    return path


def run_h_e2_to_m_p_spatial_verification_v10_7_l(
    project_root: Path,
    n_perm: int = 1000,
    n_boot: int = 500,
    group_frac: float = 0.30,
    progress: bool = False,
    smoothed_fields_path: Path | None = None,
) -> dict[str, Any]:
    settings = Settings(project_root=Path(project_root), n_perm=n_perm, n_boot=n_boot, group_frac=group_frac, progress=progress, smoothed_fields_path_override=smoothed_fields_path)
    out = settings.output_root()
    clean_output_root(out)
    _log(settings, "stage 1/7 load smoothed fields")
    npz = load_npz(settings)
    _log(settings, "stage 2/7 build H/P structure metrics and fields")
    metric_df, context = prepare_metrics(settings, npz)
    write_dataframe(context["audit_df"], out / "tables" / "h_to_p_input_audit_v10_7_l.csv")
    write_dataframe(metric_df, out / "tables" / "h_p_structure_metrics_by_year_v10_7_l.csv")
    _log(settings, "stage 3/7 run high-low composite and permutation audits")
    analysis_meta = run_analysis(settings, metric_df, context, out)
    _log(settings, "stage 4/7 save composite arrays")
    arrays_path = out / "run_meta" / "h_to_p_composite_arrays_v10_7_l.npz"
    np.savez_compressed(arrays_path, lat=context["p_lat"], lon=context["p_lon"], years=context["years"], **{f"{m}_{k}": v for m, dd in context["p_spatial"].items() for k, v in dd.items()})
    _log(settings, "stage 5/7 write run meta")
    run_meta = {
        "version": settings.version,
        "task": "H_E2 structure to M_P rainband spatial verification",
        "started_or_completed_utc": now_utc(),
        "settings": settings.to_dict(),
        "smoothed_fields_path": str(npz["path"]),
        "has_cartopy": HAS_CARTOPY,
        "n_years": int(len(context["years"])),
        "output_root": str(out),
        "analysis_meta": analysis_meta,
        "method_boundary": [
            "not causal inference",
            "not full E2-to-M mapping",
            "not control-regression",
            "H-source/P-target narrow verification",
        ],
    }
    write_json(run_meta, out / "run_meta" / "run_meta_v10_7_l.json")
    _log(settings, "stage 6/7 write summary")
    summary_path = write_summary(settings, out, analysis_meta)
    _log(settings, "stage 7/7 done")
    return {"output_root": str(out), "summary_path": str(summary_path), **run_meta}
