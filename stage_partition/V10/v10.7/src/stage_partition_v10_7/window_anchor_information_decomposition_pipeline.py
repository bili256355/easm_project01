from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import math
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# V10.7_m: window-anchor information decomposition
# =============================================================================
# Questions answered:
#   Q2: Which information form is effective: transition/state/mean/slope/background?
#   Q1/Q4: Is E2/W33 a specific useful source-window anchor for H_zonal_width -> M_P?
#   Q3: Did scalarized transition representation lose useful signed structural-component information?
#   Q5-min: Is the result source/target metric specific within H -> P?
#
# Method boundary:
#   - NOT causal inference.
#   - NOT full W33 -> W45 mapping.
#   - NOT full P/V/H/Je/Jw object-network audit.
#   - NOT proof that transition windows represent most object information.
#   - Does NOT control away P/V/Je/Jw.
#   - Decomposes and stress-tests V10.7_l interpretation.
# =============================================================================


@dataclass
class Settings:
    project_root: Path
    n_perm: int = 1000
    n_boot: int = 500
    group_frac: float = 0.30
    n_random_windows: int = 1000
    random_seed: int = 20260515
    progress: bool = False
    smoothed_fields_path_override: Path | None = None

    version: str = "v10.7_m"
    output_tag: str = "window_anchor_information_decomposition_v10_7_m"

    e2_pre: tuple[int, int] = (27, 31)
    e2_post: tuple[int, int] = (34, 38)
    m_pre: tuple[int, int] = (40, 43)
    m_post: tuple[int, int] = (45, 48)

    source_anchor_len: int = 12
    source_anchor_internal_pre_len: int = 5
    source_anchor_internal_gap_len: int = 2
    source_anchor_internal_post_len: int = 5
    source_anchor_min_start: int = 0
    source_anchor_max_end: int = 39

    h_lat_range: tuple[float, float] = (15.0, 35.0)
    h_lon_range: tuple[float, float] = (110.0, 140.0)
    p_lat_range: tuple[float, float] = (15.0, 35.0)
    p_lon_range: tuple[float, float] = (110.0, 140.0)

    modes: tuple[str, ...] = ("anomaly", "local_background_removed", "raw")
    primary_modes: tuple[str, ...] = ("anomaly", "local_background_removed")

    h_component_metrics: tuple[str, ...] = (
        "H_strength",
        "H_centroid_lat",
        "H_west_extent_lon",
        "H_zonal_width",
        "H_north_edge_lat",
        "H_south_edge_lat",
    )
    p_target_metrics: tuple[str, ...] = (
        "P_total_strength",
        "P_centroid_lat",
        "P_spread_lat",
        "P_main_band_share",
        "P_south_band_share_18_24",
        "P_main_minus_south",
    )
    p_rainband_metrics: tuple[str, ...] = (
        "P_main_band_share_transition",
        "P_south_band_share_18_24_transition",
        "P_main_minus_south_transition",
    )
    p_v10_7_l_primary_metrics: tuple[str, ...] = (
        "P_centroid_lat_transition",
        "P_main_band_share_transition",
        "P_south_band_share_18_24_transition",
        "P_main_minus_south_transition",
    )

    def smoothed_fields_path(self) -> Path:
        if self.smoothed_fields_path_override is not None:
            return self.smoothed_fields_path_override
        return (
            self.project_root
            / "foundation"
            / "V1"
            / "outputs"
            / "baseline_a"
            / "preprocess"
            / "smoothed_fields.npz"
        )

    def output_root(self) -> Path:
        return (
            self.project_root
            / "stage_partition"
            / "V10"
            / "v10.7"
            / "outputs"
            / self.output_tag
        )

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
        print(f"[V10.7_m] {msg}", flush=True)


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


def first_key(data: dict[str, Any], candidates: Iterable[str]) -> str | None:
    lower = {str(k).lower(): k for k in data.keys()}
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


def normalize_field_dims(
    field: np.ndarray,
    data: dict[str, Any],
    year_key: str | None,
    day_key: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def subset_domain(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def daily_anomaly(field: np.ndarray) -> np.ndarray:
    clim = safe_nanmean(field, axis=0, keepdims=True)
    return field - clim


def local_background_removed_field(
    field: np.ndarray,
    days: np.ndarray,
    event_win: tuple[int, int],
    bg_win: tuple[int, int],
) -> np.ndarray:
    event = day_mask(days, event_win)
    bg = day_mask(days, bg_win) & (~event)
    if not np.any(bg):
        return field * np.nan
    bg_mean = safe_nanmean(field[:, bg], axis=1, keepdims=True)
    return field - bg_mean


def window_field_mean(field: np.ndarray, days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    m = day_mask(days, win)
    if not np.any(m):
        return np.full((field.shape[0], field.shape[2], field.shape[3]), np.nan)
    return safe_nanmean(field[:, m], axis=1)


def transition_field(field: np.ndarray, days: np.ndarray, pre: tuple[int, int], post: tuple[int, int]) -> np.ndarray:
    return window_field_mean(field, days, post) - window_field_mean(field, days, pre)


def _field_rms_daily(sub: np.ndarray) -> np.ndarray:
    return np.sqrt(safe_nanmean(sub ** 2, axis=(2, 3)))


def _lat_weighted_metrics_daily(
    sub: np.ndarray,
    lat: np.ndarray,
    positive_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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


def window_mean_metric(metric: np.ndarray, days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    m = day_mask(days, win)
    if not np.any(m):
        return np.full(metric.shape[0], np.nan)
    return safe_nanmean(metric[:, m], axis=1)


def window_transition_metric(metric: np.ndarray, days: np.ndarray, pre: tuple[int, int], post: tuple[int, int]) -> np.ndarray:
    return window_mean_metric(metric, days, post) - window_mean_metric(metric, days, pre)


def window_slope_metric(metric: np.ndarray, days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    m = day_mask(days, win)
    x = days[m].astype(float)
    if x.size < 3:
        return np.full(metric.shape[0], np.nan)
    x0 = x - x.mean()
    denom = float(np.sum(x0 ** 2))
    if denom <= 0:
        return np.full(metric.shape[0], np.nan)
    y = metric[:, m]
    ymean = safe_nanmean(y, axis=1, keepdims=True)
    return np.nansum((y - ymean) * x0.reshape(1, -1), axis=1) / denom


def anchor_to_pre_post(anchor_start: int, settings: Settings) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pre_len = settings.source_anchor_internal_pre_len
    gap_len = settings.source_anchor_internal_gap_len
    post_len = settings.source_anchor_internal_post_len
    pre = (anchor_start, anchor_start + pre_len - 1)
    post_start = anchor_start + pre_len + gap_len
    post = (post_start, post_start + post_len - 1)
    full = (anchor_start, post[1])
    return pre, post, full


def build_anchor_definitions(settings: Settings) -> pd.DataFrame:
    rows = []
    named = {
        "E1_W18": 12,
        "E2_W33": 27,
        "E2_shift_left": 22,
        "E2_shift_right_safe": 28,
    }
    for name, start in named.items():
        pre, post, full = anchor_to_pre_post(start, settings)
        if full[1] <= settings.source_anchor_max_end:
            rows.append({"anchor_name": name, "anchor_type": "named", "anchor_start": start, "anchor_end": full[1], "pre_start": pre[0], "pre_end": pre[1], "post_start": post[0], "post_end": post[1]})
    for start in range(settings.source_anchor_min_start, settings.source_anchor_max_end - settings.source_anchor_len + 2):
        pre, post, full = anchor_to_pre_post(start, settings)
        if full[1] <= settings.source_anchor_max_end:
            rows.append({"anchor_name": f"slide_{start:03d}_{full[1]:03d}", "anchor_type": "sliding", "anchor_start": start, "anchor_end": full[1], "pre_start": pre[0], "pre_end": pre[1], "post_start": post[0], "post_end": post[1]})
    return pd.DataFrame(rows).drop_duplicates(subset=["anchor_name"])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 4:
        return np.nan
    xs = pd.Series(x[ok]).rank().to_numpy(dtype=float)
    ys = pd.Series(y[ok]).rank().to_numpy(dtype=float)
    return float(np.corrcoef(xs, ys)[0, 1])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 4:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def make_groups(values: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    n = int(np.sum(ok))
    if n < 6:
        z = np.zeros_like(ok, dtype=bool)
        return z, z.copy(), ok
    k = max(2, int(math.floor(n * frac)))
    k = min(k, n // 2)
    idx_ok = np.where(ok)[0]
    order = np.argsort(x[ok])
    low = np.zeros_like(ok, dtype=bool)
    high = np.zeros_like(ok, dtype=bool)
    low[idx_ok[order[:k]]] = True
    high[idx_ok[order[-k:]]] = True
    mid = ok & (~low) & (~high)
    return high, low, mid


def permutation_p_high_low(
    target_values: np.ndarray,
    high_idx: np.ndarray,
    low_idx: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> tuple[float, float, float]:
    target_values = np.asarray(target_values, dtype=float)
    obs = float(np.nanmean(target_values[high_idx]) - np.nanmean(target_values[low_idx]))
    ok = np.isfinite(target_values)
    vals = target_values[ok]
    n_h = int(np.sum(high_idx))
    n_l = int(np.sum(low_idx))
    if vals.size < n_h + n_l or n_h == 0 or n_l == 0 or n_perm <= 0:
        return obs, np.nan, np.nan
    rand = rng.random((n_perm, vals.size))
    perms = np.argsort(rand, axis=1)
    h = vals[perms[:, :n_h]]
    l = vals[perms[:, n_h:n_h + n_l]]
    diffs = np.nanmean(h, axis=1) - np.nanmean(l, axis=1)
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1.0) / (np.sum(np.isfinite(diffs)) + 1.0)
    return obs, float(np.nanpercentile(np.abs(diffs), 90)), float(p)


def bootstrap_ci_high_low(
    target_values: np.ndarray,
    high_idx: np.ndarray,
    low_idx: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    hvals = np.asarray(target_values[high_idx], dtype=float)
    lvals = np.asarray(target_values[low_idx], dtype=float)
    hvals = hvals[np.isfinite(hvals)]
    lvals = lvals[np.isfinite(lvals)]
    if hvals.size < 2 or lvals.size < 2 or n_boot <= 0:
        return np.nan, np.nan
    h_idx = rng.integers(0, hvals.size, size=(n_boot, hvals.size))
    l_idx = rng.integers(0, lvals.size, size=(n_boot, lvals.size))
    diffs = np.nanmean(hvals[h_idx], axis=1) - np.nanmean(lvals[l_idx], axis=1)
    return float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def relation_score(row: dict[str, Any]) -> float:
    p = row.get("permutation_p", np.nan)
    ci_lo = row.get("bootstrap_ci_low", np.nan)
    ci_hi = row.get("bootstrap_ci_high", np.nan)
    r = row.get("spearman_r", np.nan)
    effect = row.get("high_minus_low", np.nan)
    sig = 0.0
    if np.isfinite(p) and p <= 0.10:
        sig += 1.0
    if np.isfinite(ci_lo) and np.isfinite(ci_hi) and ((ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)):
        sig += 1.0
    return float(sig + (abs(r) if np.isfinite(r) else 0.0) + 0.05 * (abs(effect) if np.isfinite(effect) else 0.0))


def evaluate_relation(
    source_values: np.ndarray,
    target_values: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
    n_boot: int,
    group_frac: float,
    formal: bool = True,
) -> dict[str, Any]:
    source_values = np.asarray(source_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    ok = np.isfinite(source_values) & np.isfinite(target_values)
    out: dict[str, Any] = {
        "n_valid_years": int(np.sum(ok)),
        "spearman_r": spearman_corr(source_values, target_values),
        "pearson_r": pearson_corr(source_values, target_values),
    }
    high, low, _ = make_groups(source_values, group_frac)
    high &= ok
    low &= ok
    out["n_high"] = int(np.sum(high))
    out["n_low"] = int(np.sum(low))
    out["high_mean_target"] = float(np.nanmean(target_values[high])) if np.any(high) else np.nan
    out["low_mean_target"] = float(np.nanmean(target_values[low])) if np.any(low) else np.nan
    if formal:
        obs, null90, p = permutation_p_high_low(target_values, high, low, rng, n_perm)
        ci_lo, ci_hi = bootstrap_ci_high_low(target_values, high, low, rng, n_boot)
        out.update({
            "high_minus_low": obs,
            "permutation_null_p90_absdiff": null90,
            "permutation_p": p,
            "bootstrap_ci_low": ci_lo,
            "bootstrap_ci_high": ci_hi,
            "evaluation_mode": "formal_perm_boot",
        })
    else:
        obs = float(np.nanmean(target_values[high]) - np.nanmean(target_values[low])) if np.any(high) and np.any(low) else np.nan
        out.update({
            "high_minus_low": obs,
            "permutation_null_p90_absdiff": np.nan,
            "permutation_p": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "evaluation_mode": "rank_only_no_perm_boot",
        })
    out["primary_score"] = relation_score(out)
    out["support_flag"] = bool(
        (np.isfinite(out.get("permutation_p", np.nan)) and out["permutation_p"] <= 0.10)
        and np.isfinite(out.get("bootstrap_ci_low", np.nan))
        and np.isfinite(out.get("bootstrap_ci_high", np.nan))
        and ((out["bootstrap_ci_low"] > 0 and out["bootstrap_ci_high"] > 0) or (out["bootstrap_ci_low"] < 0 and out["bootstrap_ci_high"] < 0))
    )
    return out


def zscore_by_year(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return x * np.nan
    return (x - mu) / sd


def prepare_mode_fields(settings: Settings, h_sub: np.ndarray, p_sub: np.ndarray, days: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    fields = {
        "raw": {"H": h_sub, "P": p_sub},
        "anomaly": {"H": daily_anomaly(h_sub), "P": daily_anomaly(p_sub)},
        "local_background_removed": {
            "H": local_background_removed_field(h_sub, days, (27, 38), (18, 48)),
            "P": local_background_removed_field(p_sub, days, (40, 48), (30, 60)),
        },
    }
    return {m: fields[m] for m in settings.modes}


def prepare_inputs(settings: Settings) -> dict[str, Any]:
    npz = load_npz(settings)
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
    fields_by_mode = prepare_mode_fields(settings, h_sub, p_sub, days)
    metrics_by_mode: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for mode, fields in fields_by_mode.items():
        metrics_by_mode[mode] = {
            "H": build_h_daily_metrics(fields["H"], h_lat, h_lon),
            "P": build_p_daily_metrics(fields["P"], p_lat),
        }
    audit_df = pd.DataFrame([
        {"object": "H", "source_field": h_key, "lat_range": settings.h_lat_range, "lon_range": settings.h_lon_range, "loaded": True, "notes": "H structure from z500-like field."},
        {"object": "P", "source_field": p_key, "lat_range": settings.p_lat_range, "lon_range": settings.p_lon_range, "loaded": True, "notes": "P rainband structure from precip-like field."},
    ])
    return {
        "npz_path": str(npz["path"]),
        "years": years,
        "days": days,
        "h_lat": h_lat,
        "h_lon": h_lon,
        "p_lat": p_lat,
        "p_lon": p_lon,
        "fields_by_mode": fields_by_mode,
        "metrics_by_mode": metrics_by_mode,
        "input_audit": audit_df,
    }


def make_source_feature_table(settings: Settings, inputs: dict[str, Any], anchors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = inputs["years"]
    days = inputs["days"]
    for mode, objects in inputs["metrics_by_mode"].items():
        h_metrics = objects["H"]
        for _, a in anchors.iterrows():
            pre = (int(a.pre_start), int(a.pre_end))
            post = (int(a.post_start), int(a.post_end))
            full = (int(a.anchor_start), int(a.anchor_end))
            for metric_name in settings.h_component_metrics:
                daily = h_metrics[metric_name]
                values = {
                    "pre_state": window_mean_metric(daily, days, pre),
                    "post_state": window_mean_metric(daily, days, post),
                    "window_mean": window_mean_metric(daily, days, full),
                    "signed_transition": window_transition_metric(daily, days, pre, post),
                    "abs_transition": np.abs(window_transition_metric(daily, days, pre, post)),
                    "slope": window_slope_metric(daily, days, full),
                }
                for form, arr in values.items():
                    for yr, val in zip(years, arr):
                        rows.append({
                            "year": yr,
                            "mode": mode,
                            "anchor_name": a.anchor_name,
                            "anchor_type": a.anchor_type,
                            "anchor_start": int(a.anchor_start),
                            "anchor_end": int(a.anchor_end),
                            "pre_start": int(a.pre_start),
                            "pre_end": int(a.pre_end),
                            "post_start": int(a.post_start),
                            "post_end": int(a.post_end),
                            "source_metric": metric_name,
                            "information_form": form,
                            "source_value": val,
                        })
    return pd.DataFrame(rows)


def make_target_feature_table(settings: Settings, inputs: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = inputs["years"]
    days = inputs["days"]
    for mode, objects in inputs["metrics_by_mode"].items():
        p_metrics = objects["P"]
        for metric_name in settings.p_target_metrics:
            arr = window_transition_metric(p_metrics[metric_name], days, settings.m_pre, settings.m_post)
            for yr, val in zip(years, arr):
                rows.append({
                    "year": yr,
                    "mode": mode,
                    "target_metric": f"{metric_name}_transition",
                    "target_family": classify_p_metric(f"{metric_name}_transition"),
                    "target_value": val,
                })
    return pd.DataFrame(rows)


def classify_p_metric(metric: str) -> str:
    if "total_strength" in metric:
        return "P_strength"
    if "centroid" in metric:
        return "P_position"
    if "spread" in metric:
        return "P_spread"
    if "main_band_share" in metric or "south_band_share" in metric or "main_minus_south" in metric:
        return "P_rainband_structure"
    return "P_other"


def source_target_relation_table(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    settings: Settings,
    rng: np.random.Generator,
    filter_query: str | None = None,
    formal: bool = True,
) -> pd.DataFrame:
    src = source_df.copy()
    if filter_query:
        src = src.query(filter_query).copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["mode", "anchor_name", "anchor_type", "source_metric", "information_form"]
    for key, g in src.groupby(group_cols, dropna=False):
        mode, anchor_name, anchor_type, source_metric, information_form = key
        s = g[["year", "source_value", "anchor_start", "anchor_end"]].drop_duplicates("year").sort_values("year")
        tsub = target_df[target_df["mode"] == mode]
        for target_metric, tg in tsub.groupby("target_metric"):
            merged = s.merge(tg[["year", "target_metric", "target_family", "target_value"]], on="year", how="inner")
            rel = evaluate_relation(
                merged["source_value"].to_numpy(dtype=float),
                merged["target_value"].to_numpy(dtype=float),
                rng,
                settings.n_perm,
                settings.n_boot,
                settings.group_frac,
                formal=formal,
            )
            rows.append({
                "mode": mode,
                "anchor_name": anchor_name,
                "anchor_type": anchor_type,
                "anchor_start": int(s["anchor_start"].iloc[0]),
                "anchor_end": int(s["anchor_end"].iloc[0]),
                "source_metric": source_metric,
                "information_form": information_form,
                "target_metric": target_metric,
                "target_family": merged["target_family"].iloc[0] if not merged.empty else classify_p_metric(target_metric),
                **rel,
            })
    return pd.DataFrame(rows)


def summarize_h_zonal_width_information_form(rel: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    sub = rel[
        (rel["anchor_name"] == "E2_W33")
        & (rel["source_metric"] == "H_zonal_width")
        & (rel["target_metric"].isin(settings.p_v10_7_l_primary_metrics))
        & (rel["mode"].isin(settings.primary_modes))
    ].copy()
    rows = []
    for (mode, form), g in sub.groupby(["mode", "information_form"]):
        rows.append({
            "mode": mode,
            "information_form": form,
            "mean_primary_score": float(np.nanmean(g["primary_score"])),
            "n_support": int(np.nansum(g["support_flag"].astype(int))),
            "mean_abs_spearman": float(np.nanmean(np.abs(g["spearman_r"]))),
            "targets": ";".join(sorted(g["target_metric"].unique())),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_by_mode"] = out.groupby("mode")["mean_primary_score"].rank(ascending=False, method="min")
    decisions = []
    for mode, g in out.groupby("mode"):
        top = g.sort_values("mean_primary_score", ascending=False).iloc[0]
        form = str(top["information_form"])
        if top["n_support"] <= 0:
            decision = "unclear_no_supported_information_form"
        elif form == "signed_transition":
            decision = "transition_dominant"
        elif form == "post_state":
            decision = "post_state_dominant"
        elif form == "pre_state":
            decision = "pre_state_background_dominant"
        elif form == "window_mean":
            decision = "mean_state_dominant"
        elif form == "slope":
            decision = "slope_dominant"
        elif form == "abs_transition":
            decision = "unsigned_transition_magnitude_dominant"
        else:
            decision = "unclear_mixed"
        decisions.append({"mode": mode, "route_decision": decision, "top_information_form": form})
    return out.merge(pd.DataFrame(decisions), on="mode", how="left")


def anchor_specificity_summary(rel: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = rel[
        (rel["source_metric"] == "H_zonal_width")
        & (rel["target_metric"].isin(settings.p_rainband_metrics))
        & (rel["mode"].isin(settings.primary_modes))
    ].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    anchor_scores = sub.groupby(["mode", "anchor_name", "anchor_type", "anchor_start", "anchor_end", "information_form"], as_index=False).agg(
        mean_score=("primary_score", "mean"),
        mean_abs_spearman=("spearman_r", lambda x: float(np.nanmean(np.abs(x)))),
        n_targets=("target_metric", "nunique"),
    )
    rows = []
    for (mode, form), g in anchor_scores.groupby(["mode", "information_form"]):
        e2 = g[g["anchor_name"] == "E2_W33"]
        if e2.empty:
            continue
        e2_score = float(e2["mean_score"].iloc[0])
        sliding = g[g["anchor_type"] == "sliding"]
        named = g[g["anchor_type"] == "named"]
        pct = float(np.nanmean(sliding["mean_score"] <= e2_score) * 100.0) if not sliding.empty else np.nan
        better_than_named = int(np.sum(e2_score > named[named["anchor_name"] != "E2_W33"]["mean_score"].to_numpy(dtype=float)))
        if np.isfinite(pct) and pct >= 95 and better_than_named >= 2:
            decision = "E2_anchor_supported"
        elif np.isfinite(pct) and pct >= 90:
            decision = "E2_anchor_partial"
        elif np.isfinite(pct) and pct < 75:
            decision = "not_E2_specific"
        else:
            decision = "broad_or_unclear_preseason_background"
        rows.append({
            "mode": mode,
            "information_form": form,
            "E2_mean_score": e2_score,
            "E2_percentile_among_sliding_windows": pct,
            "E2_better_than_n_named_controls": better_than_named,
            "n_named_controls": int(max(0, named["anchor_name"].nunique() - 1)),
            "anchor_decision": decision,
        })
    random_rows = []
    if settings.n_random_windows > 0:
        sliding_all = anchor_scores[anchor_scores["anchor_type"] == "sliding"]
        for (mode, form), g in sliding_all.groupby(["mode", "information_form"]):
            vals = g["mean_score"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            sampled = rng.choice(vals, size=settings.n_random_windows, replace=True)
            e2_row = anchor_scores[(anchor_scores["mode"] == mode) & (anchor_scores["information_form"] == form) & (anchor_scores["anchor_name"] == "E2_W33")]
            e2_score = float(e2_row["mean_score"].iloc[0]) if not e2_row.empty else np.nan
            random_rows.append({
                "mode": mode,
                "information_form": form,
                "n_random_windows": settings.n_random_windows,
                "random_score_p50": float(np.nanpercentile(sampled, 50)),
                "random_score_p90": float(np.nanpercentile(sampled, 90)),
                "random_score_p95": float(np.nanpercentile(sampled, 95)),
                "E2_mean_score": e2_score,
                "E2_exceeds_random_p95": bool(np.isfinite(e2_score) and e2_score > np.nanpercentile(sampled, 95)),
            })
    return anchor_scores.merge(pd.DataFrame(rows), on=["mode", "information_form"], how="left"), pd.DataFrame(random_rows)


def build_representation_features(settings: Settings, inputs: dict[str, Any]) -> pd.DataFrame:
    years = inputs["years"]
    days = inputs["days"]
    rows = []
    for mode, objects in inputs["metrics_by_mode"].items():
        h_metrics = objects["H"]
        component_transitions = {}
        for metric in settings.h_component_metrics:
            component_transitions[metric] = window_transition_metric(h_metrics[metric], days, settings.e2_pre, settings.e2_post)
        z_components = np.column_stack([zscore_by_year(v) for v in component_transitions.values()])
        component_norm = np.sqrt(np.nanmean(z_components ** 2, axis=1))
        abs_component_mean = np.nanmean(np.abs(z_components), axis=1)
        # Field-level scalar transition RMS. Signless by construction.
        h_field = inputs["fields_by_mode"][mode]["H"]
        h_trans_field = transition_field(h_field, days, settings.e2_pre, settings.e2_post)
        h_field_rms = np.sqrt(safe_nanmean(h_trans_field ** 2, axis=(1, 2)))
        scalar_map = {
            "H_object_transition_rms": h_field_rms,
            "H_component_transition_norm": component_norm,
            "H_component_abs_transition_mean": abs_component_mean,
        }
        for name, arr in scalar_map.items():
            for yr, val in zip(years, arr):
                rows.append({
                    "year": yr,
                    "mode": mode,
                    "anchor_name": "E2_W33",
                    "anchor_type": "named",
                    "anchor_start": settings.e2_pre[0],
                    "anchor_end": settings.e2_post[1],
                    "source_metric": name,
                    "representation_type": "scalarized_transition",
                    "information_form": "scalar_transition",
                    "source_value": val,
                })
        for metric, arr in component_transitions.items():
            for yr, val in zip(years, arr):
                rows.append({
                    "year": yr,
                    "mode": mode,
                    "anchor_name": "E2_W33",
                    "anchor_type": "named",
                    "anchor_start": settings.e2_pre[0],
                    "anchor_end": settings.e2_post[1],
                    "source_metric": f"{metric}_signed_transition",
                    "representation_type": "signed_component_transition",
                    "information_form": "signed_transition",
                    "source_value": val,
                })
    return pd.DataFrame(rows)


def representation_comparison(rep_source_df: pd.DataFrame, target_df: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel = source_target_relation_table(rep_source_df, target_df, settings, rng, formal=True)
    rel = rel.merge(rep_source_df[["source_metric", "representation_type"]].drop_duplicates(), on="source_metric", how="left")
    sub = rel[
        (rel["target_metric"].isin(settings.p_rainband_metrics))
        & (rel["mode"].isin(settings.primary_modes))
    ].copy()
    summary = sub.groupby(["mode", "representation_type"], as_index=False).agg(
        mean_primary_score=("primary_score", "mean"),
        n_support=("support_flag", "sum"),
        mean_abs_spearman=("spearman_r", lambda x: float(np.nanmean(np.abs(x)))),
    )
    rows = []
    for mode, g in summary.groupby("mode"):
        scalar = g[g["representation_type"] == "scalarized_transition"]
        signed = g[g["representation_type"] == "signed_component_transition"]
        if scalar.empty or signed.empty:
            decision = "insufficient_representation_comparison"
        else:
            scalar_score = float(scalar["mean_primary_score"].iloc[0])
            signed_score = float(signed["mean_primary_score"].iloc[0])
            if signed_score > scalar_score and signed["n_support"].iloc[0] > scalar["n_support"].iloc[0]:
                decision = "scalarization_loss_supported"
            elif scalar_score >= signed_score and scalar["n_support"].iloc[0] > 0:
                decision = "scalarization_not_main_issue_or_object_scope_issue"
            elif signed_score <= 0 and scalar_score <= 0:
                decision = "no_representation_support"
            else:
                decision = "mixed_representation_support"
        rows.append({"mode": mode, "representation_route_decision": decision})
    return rel, summary.merge(pd.DataFrame(rows), on="mode", how="left")


def metric_specificity(rel: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = rel[
        (rel["anchor_name"] == "E2_W33")
        & (rel["mode"].isin(settings.primary_modes))
        & (rel["information_form"].isin(["signed_transition", "post_state", "window_mean", "pre_state", "slope"]))
    ].copy()
    source_summary = sub[sub["target_metric"].isin(settings.p_rainband_metrics)].groupby(["mode", "source_metric", "information_form"], as_index=False).agg(
        mean_score_to_p_rainband=("primary_score", "mean"),
        n_support_to_p_rainband=("support_flag", "sum"),
        mean_abs_spearman_to_p_rainband=("spearman_r", lambda x: float(np.nanmean(np.abs(x)))),
    )
    target_summary = sub[sub["source_metric"] == "H_zonal_width"].groupby(["mode", "target_family", "information_form"], as_index=False).agg(
        mean_score_from_h_zonal_width=("primary_score", "mean"),
        n_support_from_h_zonal_width=("support_flag", "sum"),
        mean_abs_spearman_from_h_zonal_width=("spearman_r", lambda x: float(np.nanmean(np.abs(x)))),
    )
    decisions = []
    for mode in settings.primary_modes:
        src = source_summary[source_summary["mode"] == mode]
        tgt = target_summary[target_summary["mode"] == mode]
        top_src = src.sort_values("mean_score_to_p_rainband", ascending=False).head(1)
        top_tgt = tgt.sort_values("mean_score_from_h_zonal_width", ascending=False).head(1)
        src_name = str(top_src["source_metric"].iloc[0]) if not top_src.empty else "NA"
        tgt_name = str(top_tgt["target_family"].iloc[0]) if not top_tgt.empty else "NA"
        if src_name == "H_zonal_width" and tgt_name == "P_rainband_structure":
            decision = "specific_H_zonal_width_to_P_rainband"
        elif tgt_name != "P_rainband_structure":
            decision = "target_not_rainband_specific"
        elif src_name != "H_zonal_width":
            decision = "source_not_zonal_width_specific"
        else:
            decision = "general_or_unclear_specificity"
        decisions.append({"mode": mode, "top_source_metric": src_name, "top_target_family": tgt_name, "metric_specificity_decision": decision})
    return source_summary, target_summary, pd.DataFrame(decisions)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, hue: str | None = None, rotate: bool = True) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(df)), 4.5))
    if hue is None:
        labels = df[x].astype(str).tolist()
        vals = df[y].to_numpy(dtype=float)
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right")
    else:
        pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_md(out: Path, summaries: dict[str, Any]) -> None:
    lines = []
    lines.append("# V10.7_m window-anchor information decomposition summary\n")
    lines.append("This file is a diagnostic summary, not a paper conclusion. It separates direct outputs, derived judgments, allowed interpretations, forbidden interpretations, and next-step implications.\n")
    for section, obj in summaries.items():
        lines.append(f"\n## {section}\n")
        if isinstance(obj, pd.DataFrame):
            if obj.empty:
                lines.append("No rows produced.\n")
            else:
                lines.append(obj.head(20).to_markdown(index=False))
                lines.append("\n")
        else:
            lines.append(str(obj))
            lines.append("\n")
    lines.append("\n## Global forbidden interpretations\n")
    lines.append("- Do not state H causes P.\n")
    lines.append("- Do not state H controls W45.\n")
    lines.append("- Do not generalize to full W33-to-W45 mapping.\n")
    lines.append("- Do not say the strength class failed; V10.7_i is a scalarized-transition limitation.\n")
    lines.append("- Do not say transition windows represent most object information.\n")
    (out / "summary_window_anchor_information_decomposition_v10_7_m.md").write_text("\n".join(lines), encoding="utf-8")


def run_window_anchor_information_decomposition_v10_7_m(settings: Settings) -> dict[str, Any]:
    out = settings.output_root()
    clean_output_root(out)
    rng = np.random.default_rng(settings.random_seed)

    _log(settings, "stage 1/8 load fields and build daily metrics")
    inputs = prepare_inputs(settings)
    write_dataframe(inputs["input_audit"], out / "tables" / "input_audit_v10_7_m.csv")

    _log(settings, "stage 2/8 build source anchors and yearly features")
    anchors = build_anchor_definitions(settings)
    write_dataframe(anchors, out / "tables" / "source_anchor_definitions_v10_7_m.csv")
    source_features = make_source_feature_table(settings, inputs, anchors)
    target_features = make_target_feature_table(settings, inputs)
    write_dataframe(source_features, out / "tables" / "yearly_source_features_v10_7_m.csv")
    write_dataframe(target_features, out / "tables" / "yearly_target_features_v10_7_m.csv")

    _log(settings, "stage 3/8 Q2 information-form decomposition for fixed/named anchors")
    named_source = source_features[source_features["anchor_type"] == "named"].copy()
    q2_rel = source_target_relation_table(named_source, target_features, settings, rng, formal=True)
    write_dataframe(q2_rel, out / "tables" / "information_form_decomposition_v10_7_m.csv")
    q2_route = summarize_h_zonal_width_information_form(q2_rel, settings)
    write_dataframe(q2_route, out / "tables" / "h_zonal_width_information_form_route_decision_v10_7_m.csv")

    _log(settings, "stage 4/8 Q1/Q4 anchor specificity over sliding windows")
    # Sliding anchor specificity is rank-based by default to avoid turning random-window screening into the main significance layer.
    sliding_source = source_features[source_features["source_metric"] == "H_zonal_width"].copy()
    anchor_rel = source_target_relation_table(sliding_source, target_features, settings, rng, formal=False)
    write_dataframe(anchor_rel, out / "tables" / "anchor_specificity_all_windows_v10_7_m.csv")
    anchor_summary, random_summary = anchor_specificity_summary(anchor_rel, settings, rng)
    write_dataframe(anchor_summary, out / "tables" / "e2_anchor_rank_summary_v10_7_m.csv")
    write_dataframe(random_summary, out / "tables" / "random_window_null_summary_v10_7_m.csv")

    _log(settings, "stage 5/8 Q3 scalarized-vs-signed representation comparison")
    rep_features = build_representation_features(settings, inputs)
    write_dataframe(rep_features, out / "tables" / "yearly_representation_source_features_v10_7_m.csv")
    rep_rel, rep_summary = representation_comparison(rep_features, target_features, settings, rng)
    write_dataframe(rep_rel, out / "tables" / "representation_comparison_v10_7_m.csv")
    write_dataframe(rep_summary, out / "tables" / "scalar_vs_signed_component_route_decision_v10_7_m.csv")

    _log(settings, "stage 6/8 Q5-min source/target metric specificity")
    source_spec, target_spec, metric_decision = metric_specificity(q2_rel, settings)
    write_dataframe(source_spec, out / "tables" / "source_metric_specificity_v10_7_m.csv")
    write_dataframe(target_spec, out / "tables" / "target_metric_specificity_v10_7_m.csv")
    write_dataframe(metric_decision, out / "tables" / "metric_specificity_route_decision_v10_7_m.csv")

    _log(settings, "stage 7/8 figures")
    if not q2_route.empty:
        for mode in q2_route["mode"].unique():
            sub = q2_route[q2_route["mode"] == mode].sort_values("mean_primary_score", ascending=False)
            plot_bar(sub, "information_form", "mean_primary_score", f"Q2 H_zonal_width information forms ({mode})", out / "figures" / f"information_form_rank_h_zonal_width_{mode}_v10_7_m.png")
    if not anchor_summary.empty:
        for mode in anchor_summary["mode"].unique():
            sub = anchor_summary[anchor_summary["mode"] == mode]
            plot_bar(sub, "information_form", "E2_percentile_among_sliding_windows", f"Q1/Q4 E2 percentile among sliding windows ({mode})", out / "figures" / f"e2_anchor_percentile_{mode}_v10_7_m.png")
    if not rep_summary.empty:
        for mode in rep_summary["mode"].unique():
            sub = rep_summary[rep_summary["mode"] == mode]
            plot_bar(sub, "representation_type", "mean_primary_score", f"Q3 scalarized vs signed component ({mode})", out / "figures" / f"scalar_vs_signed_component_{mode}_v10_7_m.png")
    if not target_spec.empty:
        for mode in target_spec["mode"].unique():
            sub = target_spec[target_spec["mode"] == mode]
            plot_bar(sub, "target_family", "mean_score_from_h_zonal_width", f"Q5 target-family specificity ({mode})", out / "figures" / f"target_family_specificity_{mode}_v10_7_m.png", hue="information_form")

    _log(settings, "stage 8/8 run_meta and summary")
    run_meta = {
        "created_utc": now_utc(),
        "settings": settings.to_dict(),
        "input_npz_path": inputs["npz_path"],
        "questions_answered": {
            "Q2": "information form decomposition: transition/state/mean/slope/background proxy",
            "Q1_Q4": "E2/W33 anchor specificity against E1/shifted/sliding/random source windows",
            "Q3": "scalarized transition representation vs signed structural-component transition representation",
            "Q5_minimal": "source/target metric specificity within H -> P narrow channel",
        },
        "method_boundary": [
            "not causal inference",
            "not full W33-to-W45 mapping",
            "not full object-network audit",
            "not proof that transition windows represent all object information",
            "does not control away P/V/Je/Jw",
            "does not replace V10.7_l; decomposes and stress-tests its interpretation",
            "random/sliding anchor specificity uses rank-only screening by default; formal permutation/bootstrap remains on fixed/named relation tables and representation comparison",
        ],
        "forbidden_interpretations": [
            "Do not state H causes P",
            "Do not state H controls W45",
            "Do not generalize to full E2->M or W33->W45 mapping",
            "Do not say strength class failed; V10.7_i only constrains scalarized transition mapping",
            "Do not say transition windows represent most object information unless future full-series/memory audit supports it",
        ],
        "outputs": {
            "tables": sorted([p.name for p in (out / "tables").glob("*.csv")]),
            "figures": sorted([p.name for p in (out / "figures").glob("*.png")]),
        },
    }
    write_json(run_meta, out / "run_meta" / "run_meta_v10_7_m.json")
    write_summary_md(out, {
        "Q2 direct output: information form route decision": q2_route,
        "Q1/Q4 direct output: E2 anchor rank summary": anchor_summary,
        "Q3 direct output: scalarized vs signed route decision": rep_summary,
        "Q5-min direct output: metric specificity decision": metric_decision,
    })
    _log(settings, f"done: {out}")
    return run_meta


__all__ = ["Settings", "run_window_anchor_information_decomposition_v10_7_m"]
