"""
V7-z-clean: W45 multi-object pre-post state-growth clean mainline.

Purpose
-------
A clean W45 mainline that extracts the stabilized implementation and validation logic:
1. Object-window detection uses only raw/profile object-state input.
2. Pattern is retained inside pre-post extraction via R_diff / S_pattern, not as the main
   object-window detector input.
3. Evidence is summarized by evidence families with a co-transition veto; weak repeated
   tendencies cannot be stacked into hard lead claims.

This module is intentionally self-contained and does not modify V7-z/v/w/x/y outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


VERSION = "v7_z_clean"
OUTPUT_TAG = "W45_multi_object_prepost_clean_mainline_v7_z_clean"


@dataclass(frozen=True)
class ObjectSpec:
    object_name: str
    field_role: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lat_step: float = 2.0


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    pre_start: int
    pre_end: int
    post_start: int
    post_end: int
    role: str


@dataclass
class CleanConfig:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    w45_start: int = 40
    w45_end: int = 48
    anchor_day: int = 45
    detection_start: int = 0
    detection_end: int = 70
    curve_start: int = 0
    curve_end: int = 74
    compare_start: int = 35
    compare_end: int = 53
    early_start: int = 30
    early_end: int = 39
    core_start: int = 40
    core_end: int = 45
    late_start: int = 46
    late_end: int = 53
    detector_width: int = 20
    detector_min_size: int = 2
    peak_min_distance: int = 3
    max_peaks_per_object: int = 5
    band_max_half_width: int = 10
    band_min_half_width: int = 2
    band_score_ratio: float = 0.50
    band_floor_quantile: float = 0.35
    bootstrap_n: int = 1000
    random_seed: int = 42
    peak_match_days: int = 5
    strict_peak_match_days: int = 2
    near_peak_match_days: int = 8
    selected_window_overlap_min: float = 0.25
    low_dynamic_range_eps: float = 1e-10
    norm_eps: float = 1e-12
    save_bootstrap_metric_samples: bool = True
    save_by_year_profiles: bool = False
    skip_figures: bool = False

    @staticmethod
    def from_env() -> "CleanConfig":
        cfg = CleanConfig()
        debug_n = os.environ.get("V7Z_CLEAN_DEBUG_N_BOOTSTRAP") or os.environ.get("V7Z_DEBUG_N_BOOTSTRAP")
        if debug_n:
            cfg.bootstrap_n = int(debug_n)
        if os.environ.get("V7Z_CLEAN_SKIP_FIGURES") == "1" or os.environ.get("V7Z_SKIP_FIGURES") == "1":
            cfg.skip_figures = True
        if os.environ.get("V7Z_CLEAN_SAVE_BY_YEAR_PROFILES") == "1":
            cfg.save_by_year_profiles = True
        if os.environ.get("V7Z_CLEAN_NO_BOOTSTRAP_SAMPLES") == "1":
            cfg.save_bootstrap_metric_samples = False
        return cfg


OBJECT_SPECS: List[ObjectSpec] = [
    ObjectSpec("P", "precip", 105, 125, 15, 39),
    ObjectSpec("V", "v850", 105, 125, 10, 30),
    ObjectSpec("H", "z500", 110, 140, 15, 35),
    ObjectSpec("Je", "u200", 120, 150, 25, 45),
    ObjectSpec("Jw", "u200", 80, 110, 25, 45),
]

BASELINES: List[BaselineConfig] = [
    BaselineConfig("C0_full_stage", 0, 39, 49, 74, "main_full_stage"),
    BaselineConfig("C1_buffered_stage", 0, 34, 54, 69, "buffered_sensitivity"),
    BaselineConfig("C2_immediate_pre", 25, 34, 54, 69, "immediate_pre_sensitivity"),
]

FIELD_ALIASES = {
    "precip": [
        "precip_smoothed", "precipitation_smoothed", "pr_smoothed", "P_smoothed",
        "precip", "precipitation", "pr", "P", "tp", "rain", "rainfall",
    ],
    "v850": ["v850_smoothed", "v850", "v_smoothed", "v", "V", "v850_anom"],
    "z500": ["z500_smoothed", "z500", "hgt500_smoothed", "hgt500", "H", "z"],
    "u200": ["u200_smoothed", "u200", "u_smoothed", "u", "U200", "u200_anom"],
    "lat": ["lat", "latitude", "lats", "nav_lat"],
    "lon": ["lon", "longitude", "lons", "nav_lon"],
    "years": ["years", "year", "yrs"],
}


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _nanmean(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(a, axis=axis)


def _safe_quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanquantile(arr, q))


def _prob_positive(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr > 0))


def _prob_negative(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr < 0))


def _weighted_mean(x: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def _weighted_norm(x: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    return float(math.sqrt(np.sum(w[m] * x[m] ** 2) / np.sum(w[m])))


def _weighted_distance(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), weights)


def _weighted_corr(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0)
    if np.sum(m) < 3:
        return float("nan")
    aa = a[m]
    bb = b[m]
    ww = w[m]
    am = np.sum(aa * ww) / np.sum(ww)
    bm = np.sum(bb * ww) / np.sum(ww)
    ac = aa - am
    bc = bb - bm
    denom = math.sqrt(np.sum(ww * ac ** 2) * np.sum(ww * bc ** 2))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.sum(ww * ac * bc) / denom)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> Tuple[float, float]:
    if not all(np.isfinite([a0, a1, b0, b1])):
        return 0.0, 0.0
    left = max(a0, b0)
    right = min(a1, b1)
    ov = max(0.0, right - left + 1.0)
    la = max(0.0, a1 - a0 + 1.0)
    lb = max(0.0, b1 - b0 + 1.0)
    denom = min(la, lb) if min(la, lb) > 0 else float("nan")
    frac = ov / denom if np.isfinite(denom) and denom > 0 else 0.0
    return float(ov), float(frac)


# -----------------------------------------------------------------------------
# Input loading and profile construction
# -----------------------------------------------------------------------------

def _find_key(npz: np.lib.npyio.NpzFile, aliases: Sequence[str]) -> Optional[str]:
    keys_lower = {k.lower(): k for k in npz.files}
    for a in aliases:
        if a in npz.files:
            return a
        if a.lower() in keys_lower:
            return keys_lower[a.lower()]
    return None


def _load_npz_fields(path: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"smoothed fields not found: {path}")
    npz = np.load(path, allow_pickle=True)
    audit_rows = []
    out: Dict[str, np.ndarray] = {}
    for role, aliases in FIELD_ALIASES.items():
        key = _find_key(npz, aliases)
        audit_rows.append({"role": role, "resolved_key": key, "status": "found" if key else "missing"})
        if key:
            out[role] = np.asarray(npz[key])
    missing = [r["role"] for r in audit_rows if r["status"] == "missing" and r["role"] in ["lat", "lon", "u200", "v850", "z500", "precip"]]
    audit = pd.DataFrame(audit_rows)
    if missing:
        raise KeyError(f"missing required fields in {path}: {missing}; available keys={npz.files}")
    return out, audit


def _as_year_day_lat_lon(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, years: Optional[np.ndarray]) -> np.ndarray:
    """Return array shaped [year, day, lat, lon]."""
    arr = np.asarray(arr, dtype=float)
    nlat, nlon = len(lat), len(lon)
    if arr.ndim == 4:
        # Common order [year, day, lat, lon]
        if arr.shape[2] == nlat and arr.shape[3] == nlon:
            return arr
        # [year, day, lon, lat]
        if arr.shape[2] == nlon and arr.shape[3] == nlat:
            return np.transpose(arr, (0, 1, 3, 2))
        # [day, year, lat, lon]
        if arr.shape[2] == nlat and arr.shape[3] == nlon and years is not None and arr.shape[1] == len(years):
            return np.transpose(arr, (1, 0, 2, 3))
    if arr.ndim == 3:
        # [time, lat, lon] -> [year, day, lat, lon]
        if arr.shape[1] == nlat and arr.shape[2] == nlon:
            nt = arr.shape[0]
            if years is not None and len(years) > 0 and nt % len(years) == 0:
                nd = nt // len(years)
                return arr.reshape(len(years), nd, nlat, nlon)
            # fallback: assume one year
            return arr.reshape(1, nt, nlat, nlon)
        # [time, lon, lat]
        if arr.shape[1] == nlon and arr.shape[2] == nlat:
            arr2 = np.transpose(arr, (0, 2, 1))
            nt = arr2.shape[0]
            if years is not None and len(years) > 0 and nt % len(years) == 0:
                nd = nt // len(years)
                return arr2.reshape(len(years), nd, nlat, nlon)
            return arr2.reshape(1, nt, nlat, nlon)
    raise ValueError(f"Cannot infer field dimensions for array shape {arr.shape}, lat={nlat}, lon={nlon}")


def _target_lats(spec: ObjectSpec) -> np.ndarray:
    lo = min(spec.lat_min, spec.lat_max)
    hi = max(spec.lat_min, spec.lat_max)
    n = int(round((hi - lo) / spec.lat_step))
    vals = lo + np.arange(n + 1) * spec.lat_step
    vals[-1] = hi
    return vals.astype(float)


def _build_object_profile(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: ObjectSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build object profile [year, day, target_lat]."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    arr = np.asarray(field, dtype=float)
    lo_lat, hi_lat = min(spec.lat_min, spec.lat_max), max(spec.lat_min, spec.lat_max)
    lo_lon, hi_lon = min(spec.lon_min, spec.lon_max), max(spec.lon_min, spec.lon_max)
    lat_mask = (lat >= lo_lat) & (lat <= hi_lat)
    lon_mask = (lon >= lo_lon) & (lon <= hi_lon)
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points in {lo_lat}-{hi_lat} for {spec.object_name}")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points in {lo_lon}-{hi_lon} for {spec.object_name}")
    region = arr[:, :, lat_mask, :][:, :, :, lon_mask]
    # average over lon; later interpolate over lat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        prof_native = np.nanmean(region, axis=-1)  # y,d,lat_native
    lat_native = lat[lat_mask]
    order = np.argsort(lat_native)
    lat_sorted = lat_native[order]
    prof_sorted = prof_native[:, :, order]
    target = _target_lats(spec)
    y, d, _ = prof_sorted.shape
    out = np.full((y, d, len(target)), np.nan, dtype=float)
    for iy in range(y):
        for iday in range(d):
            x = prof_sorted[iy, iday, :]
            m = np.isfinite(x) & np.isfinite(lat_sorted)
            if np.sum(m) >= 2:
                out[iy, iday, :] = np.interp(target, lat_sorted[m], x[m], left=np.nan, right=np.nan)
    weights = np.cos(np.deg2rad(target))
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return out, target, weights


# -----------------------------------------------------------------------------
# Detector
# -----------------------------------------------------------------------------

def _window_scores(X: np.ndarray, weights: np.ndarray, cfg: CleanConfig) -> pd.DataFrame:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    half = cfg.detector_width // 2
    rows = []
    for day in range(cfg.detection_start, min(cfg.detection_end, n - 1) + 1):
        l0, l1 = day - half, day - 1
        r0, r1 = day, day + half - 1
        if l0 < 0 or r1 >= n:
            score = np.nan
            valid = False
        else:
            left = X[l0:l1 + 1, :]
            right = X[r0:r1 + 1, :]
            if left.shape[0] < cfg.detector_min_size or right.shape[0] < cfg.detector_min_size:
                score = np.nan
                valid = False
            else:
                lm = _nanmean(left, axis=0)
                rm = _nanmean(right, axis=0)
                score = _weighted_distance(rm, lm, weights)
                valid = np.isfinite(score)
        rows.append({"day": day, "detector_score": score, "score_valid": valid})
    return pd.DataFrame(rows)


def _is_far_enough(day: int, selected: List[int], min_dist: int) -> bool:
    return all(abs(day - s) >= min_dist for s in selected)


def _extract_peaks(score_df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    df = score_df.copy()
    s = df["detector_score"].to_numpy(dtype=float)
    days = df["day"].to_numpy(dtype=int)
    finite = np.isfinite(s)
    if not np.any(finite):
        return pd.DataFrame(columns=["peak_id", "peak_day", "peak_score", "peak_prominence", "peak_rank"])
    candidates = []
    for i in range(1, len(s) - 1):
        if not np.isfinite(s[i]):
            continue
        if (not np.isfinite(s[i - 1]) or s[i] >= s[i - 1]) and (not np.isfinite(s[i + 1]) or s[i] >= s[i + 1]):
            local_base = np.nanmedian([x for x in [s[i - 1], s[i + 1]] if np.isfinite(x)])
            prom = s[i] - local_base if np.isfinite(local_base) else s[i]
            candidates.append((days[i], s[i], prom))
    # If no local maxima due to monotonic scores, use global max.
    if not candidates:
        imax = int(np.nanargmax(s))
        candidates.append((days[imax], s[imax], s[imax]))
    candidates = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)
    selected_days: List[int] = []
    rows = []
    for d, score, prom in candidates:
        if not _is_far_enough(int(d), selected_days, cfg.peak_min_distance):
            continue
        selected_days.append(int(d))
        rank = len(rows) + 1
        rows.append({
            "peak_id": f"CP{rank:03d}",
            "peak_day": int(d),
            "peak_score": float(score),
            "peak_prominence": float(prom),
            "peak_rank": rank,
        })
        if len(rows) >= cfg.max_peaks_per_object:
            break
    return pd.DataFrame(rows)


def _band_for_peak(score_df: pd.DataFrame, peak_day: int, peak_score: float, cfg: CleanConfig) -> Tuple[int, int, str, str]:
    df = score_df.set_index("day")
    finite_scores = df["detector_score"].to_numpy(dtype=float)
    floor = _safe_quantile(finite_scores, cfg.band_floor_quantile)
    threshold = max(cfg.band_score_ratio * peak_score, floor if np.isfinite(floor) else -np.inf)
    start = peak_day
    end = peak_day
    left_reason = "max_half_width"
    right_reason = "max_half_width"
    for dd in range(1, cfg.band_max_half_width + 1):
        day = peak_day - dd
        if day not in df.index:
            left_reason = "missing_day"
            break
        sc = float(df.loc[day, "detector_score"])
        if not np.isfinite(sc) or sc < threshold:
            left_reason = "score_below_threshold"
            break
        start = day
    for dd in range(1, cfg.band_max_half_width + 1):
        day = peak_day + dd
        if day not in df.index:
            right_reason = "missing_day"
            break
        sc = float(df.loc[day, "detector_score"])
        if not np.isfinite(sc) or sc < threshold:
            right_reason = "score_below_threshold"
            break
        end = day
    start = min(start, peak_day - cfg.band_min_half_width)
    end = max(end, peak_day + cfg.band_min_half_width)
    start = max(cfg.detection_start, start)
    end = min(cfg.detection_end, end)
    return int(start), int(end), left_reason, right_reason


def _run_detector_for_profile(X: np.ndarray, weights: np.ndarray, cfg: CleanConfig, object_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scores = _window_scores(X, weights, cfg)
    peaks = _extract_peaks(scores, cfg)
    rows = []
    for _, p in peaks.iterrows():
        b0, b1, lr, rr = _band_for_peak(scores, int(p["peak_day"]), float(p["peak_score"]), cfg)
        ov, ov_frac = _interval_overlap(b0, b1, cfg.w45_start, cfg.w45_end)
        rows.append({
            "object": object_name,
            "candidate_id": p["peak_id"],
            "peak_day": int(p["peak_day"]),
            "band_start_day": b0,
            "band_end_day": b1,
            "peak_score": float(p["peak_score"]),
            "peak_prominence": float(p["peak_prominence"]),
            "peak_rank": int(p["peak_rank"]),
            "overlap_days_with_W45": ov,
            "overlap_fraction_with_W45": ov_frac,
            "left_stop_reason": lr,
            "right_stop_reason": rr,
        })
    return scores.assign(object=object_name), pd.DataFrame(rows)


def _select_w45_relevant_candidate(cand: pd.DataFrame, cfg: CleanConfig) -> Optional[pd.Series]:
    if cand.empty:
        return None
    c = cand.copy()
    c["support_tier"] = c.get("bootstrap_support", pd.Series([np.nan] * len(c))).apply(
        lambda x: 3 if x >= 0.95 else (2 if x >= 0.80 else (1 if x >= 0.50 else 0))
    )
    # W45-front/main relevance: overlap W45 or peak in broad W45-front interval.
    c["w45_overlap_flag"] = c["overlap_fraction_with_W45"].fillna(0) > 0
    c["front_relevance_flag"] = c["peak_day"].between(25, 56)
    c["distance_to_anchor"] = (c["peak_day"] - cfg.anchor_day).abs()
    c["selection_score"] = (
        c["support_tier"] * 1000
        + c["w45_overlap_flag"].astype(int) * 250
        + c["front_relevance_flag"].astype(int) * 100
        - c["distance_to_anchor"]
        + c["peak_score"].rank(pct=True).fillna(0) * 10
    )
    row = c.sort_values(["selection_score", "peak_score"], ascending=False).iloc[0]
    return row


def _select_boot_candidate(cand: pd.DataFrame, cfg: CleanConfig) -> float:
    # Observed-free selection for bootstrap: W45/front relevance plus score.
    if cand.empty:
        return float("nan")
    c = cand.copy()
    c["w45_overlap_flag"] = c["overlap_fraction_with_W45"].fillna(0) > 0
    c["front_relevance_flag"] = c["peak_day"].between(25, 56)
    c["distance_to_anchor"] = (c["peak_day"] - cfg.anchor_day).abs()
    c["selection_score"] = (
        c["w45_overlap_flag"].astype(int) * 250
        + c["front_relevance_flag"].astype(int) * 100
        - c["distance_to_anchor"]
        + c["peak_score"].rank(pct=True).fillna(0) * 10
    )
    row = c.sort_values(["selection_score", "peak_score"], ascending=False).iloc[0]
    return float(row["peak_day"])


# -----------------------------------------------------------------------------
# State/growth curves
# -----------------------------------------------------------------------------

def _day_slice(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    n = arr.shape[0]
    s = max(0, start)
    e = min(n - 1, end)
    if s > e:
        return arr[0:0]
    return arr[s:e + 1]


def _compute_state_growth_for_object(object_name: str, X: np.ndarray, weights: np.ndarray, baselines: Sequence[BaselineConfig], cfg: CleanConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """X is climatological day x feature."""
    rows = []
    branch_rows = []
    for b in baselines:
        pre = _nanmean(_day_slice(X, b.pre_start, b.pre_end), axis=0)
        post = _nanmean(_day_slice(X, b.post_start, b.post_end), axis=0)
        pre_valid = np.any(np.isfinite(pre))
        post_valid = np.any(np.isfinite(post))
        pre_rdiff_vals = []
        post_rdiff_vals = []
        # First pass R_diff to normalize S_pattern.
        rdiff_by_day: Dict[int, float] = {}
        for day in range(cfg.curve_start, min(cfg.curve_end, X.shape[0] - 1) + 1):
            x = X[day]
            rpre = _weighted_corr(x, pre, weights) if pre_valid else np.nan
            rpost = _weighted_corr(x, post, weights) if post_valid else np.nan
            rdiff = rpost - rpre if np.isfinite(rpre) and np.isfinite(rpost) else np.nan
            rdiff_by_day[day] = rdiff
            if b.pre_start <= day <= b.pre_end:
                pre_rdiff_vals.append(rdiff)
            if b.post_start <= day <= b.post_end:
                post_rdiff_vals.append(rdiff)
        pre_mu = _nanmean(np.asarray(pre_rdiff_vals, dtype=float))
        post_mu = _nanmean(np.asarray(post_rdiff_vals, dtype=float))
        dyn = post_mu - pre_mu if np.isfinite(pre_mu) and np.isfinite(post_mu) else np.nan
        pattern_valid = np.isfinite(dyn) and abs(dyn) > cfg.low_dynamic_range_eps
        prev_sdist = np.nan
        prev_spatt = np.nan
        for day in range(cfg.curve_start, min(cfg.curve_end, X.shape[0] - 1) + 1):
            x = X[day]
            dpre = _weighted_distance(x, pre, weights) if pre_valid else np.nan
            dpost = _weighted_distance(x, post, weights) if post_valid else np.nan
            denom = dpre + dpost if np.isfinite(dpre) and np.isfinite(dpost) else np.nan
            sdist = dpre / denom if np.isfinite(denom) and denom > 0 else np.nan
            rpre = _weighted_corr(x, pre, weights) if pre_valid else np.nan
            rpost = _weighted_corr(x, post, weights) if post_valid else np.nan
            rdiff = rdiff_by_day.get(day, np.nan)
            spatt = (rdiff - pre_mu) / dyn if pattern_valid and np.isfinite(rdiff) else np.nan
            vdist = sdist - prev_sdist if np.isfinite(sdist) and np.isfinite(prev_sdist) else np.nan
            vpatt = spatt - prev_spatt if np.isfinite(spatt) and np.isfinite(prev_spatt) else np.nan
            rows.append({
                "object": object_name,
                "baseline_config": b.name,
                "baseline_role": b.role,
                "day": day,
                "D_pre": dpre,
                "D_post": dpost,
                "S_dist": sdist,
                "R_pre": rpre,
                "R_post": rpost,
                "R_diff": rdiff,
                "S_pattern": spatt,
                "V_dist": vdist,
                "V_pattern": vpatt,
                "pattern_dynamic_range": dyn,
                "pattern_branch_valid": bool(pattern_valid),
            })
            prev_sdist = sdist
            prev_spatt = spatt
        branch_rows.append({
            "object": object_name,
            "baseline_config": b.name,
            "pre_start": b.pre_start,
            "pre_end": b.pre_end,
            "post_start": b.post_start,
            "post_end": b.post_end,
            "pattern_dynamic_range": dyn,
            "pattern_branch_valid": bool(pattern_valid),
            "Rdiff_pre_mean": pre_mu,
            "Rdiff_post_mean": post_mu,
        })
    state = pd.DataFrame(rows)
    growth = state[["object", "baseline_config", "day", "V_dist", "V_pattern"]].copy()
    return state, growth, pd.DataFrame(branch_rows)


def _positive_growth_center(curve: pd.DataFrame, branch: str, cfg: CleanConfig) -> float:
    vcol = "V_dist" if branch == "dist" else "V_pattern"
    d = curve[(curve["day"] >= cfg.early_start) & (curve["day"] <= cfg.late_end)].copy()
    vals = d[vcol].to_numpy(dtype=float)
    days = d["day"].to_numpy(dtype=float)
    pos = np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)
    if np.nansum(pos) <= 0:
        return float("nan")
    return float(np.nansum(days * pos) / np.nansum(pos))


def _segment_mean(df: pd.DataFrame, col: str, start: int, end: int) -> float:
    sub = df[(df["day"] >= start) & (df["day"] <= end)][col].to_numpy(dtype=float)
    return float(_nanmean(sub))


def _pairwise_curve_metrics(state: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    rows = []
    objects = sorted(state["object"].unique())
    baselines = sorted(state["baseline_config"].unique())
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            for base in baselines:
                da = state[(state["object"] == a) & (state["baseline_config"] == base)]
                db = state[(state["object"] == b) & (state["baseline_config"] == base)]
                if da.empty or db.empty:
                    continue
                ma = da.set_index("day")
                mb = db.set_index("day")
                days = sorted(set(ma.index).intersection(set(mb.index)))
                days = [d for d in days if cfg.compare_start <= d <= cfg.compare_end]
                if not days:
                    continue
                for branch, scol, vcol in [("dist", "S_dist", "V_dist"), ("pattern", "S_pattern", "V_pattern")]:
                    va = ma.loc[days, scol].to_numpy(dtype=float)
                    vb = mb.loc[days, scol].to_numpy(dtype=float)
                    adv = va - vb
                    adv_mean = float(_nanmean(adv))
                    ahead_frac = float(np.nanmean(adv > 0)) if np.any(np.isfinite(adv)) else np.nan
                    ca = _positive_growth_center(da, branch, cfg)
                    cb = _positive_growth_center(db, branch, cfg)
                    gdelta = cb - ca if np.isfinite(ca) and np.isfinite(cb) else np.nan
                    rows.append({
                        "object_A": a,
                        "object_B": b,
                        "baseline_config": base,
                        "metric_family": f"state_{branch}",
                        "metric_name": "mean_state_advantage_A_minus_B",
                        "delta_definition": "A_minus_B; positive means A state progress is higher/ahead",
                        "observed": adv_mean,
                        "P_A_direction_observed": bool(adv_mean > 0) if np.isfinite(adv_mean) else False,
                        "P_B_direction_observed": bool(adv_mean < 0) if np.isfinite(adv_mean) else False,
                        "ahead_fraction_A_gt_B": ahead_frac,
                    })
                    rows.append({
                        "object_A": a,
                        "object_B": b,
                        "baseline_config": base,
                        "metric_family": f"growth_{branch}",
                        "metric_name": "growth_center_delta_B_minus_A",
                        "delta_definition": "B_center_minus_A_center; positive means A growth center earlier",
                        "observed": gdelta,
                        "P_A_direction_observed": bool(gdelta > 0) if np.isfinite(gdelta) else False,
                        "P_B_direction_observed": bool(gdelta < 0) if np.isfinite(gdelta) else False,
                        "ahead_fraction_A_gt_B": np.nan,
                    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Physical/profile feature support
# -----------------------------------------------------------------------------

def _profile_features_for_object(object_name: str, X: np.ndarray, target_lat: np.ndarray, weights: np.ndarray, cfg: CleanConfig) -> pd.DataFrame:
    rows = []
    for day in range(cfg.curve_start, min(cfg.curve_end, X.shape[0] - 1) + 1):
        x = X[day]
        valid = np.isfinite(x)
        if not np.any(valid):
            rows.append({"object": object_name, "day": day, "profile_valid": False})
            continue
        wmean = _weighted_mean(x, weights)
        centered = x - wmean
        std = _weighted_norm(centered, weights)
        amp = float(np.nanmax(x) - np.nanmin(x)) if np.any(valid) else np.nan
        try:
            imax = int(np.nanargmax(x))
            axis_lat = float(target_lat[imax])
        except Exception:
            axis_lat = np.nan
        pos = np.where(np.isfinite(x) & (x > 0), x, 0.0)
        if np.nansum(pos) > 0:
            centroid = float(np.nansum(target_lat * pos) / np.nansum(pos))
            spread = float(math.sqrt(np.nansum(pos * (target_lat - centroid) ** 2) / np.nansum(pos)))
        else:
            ab = np.where(np.isfinite(x), np.abs(x), 0.0)
            centroid = float(np.nansum(target_lat * ab) / np.nansum(ab)) if np.nansum(ab) > 0 else np.nan
            spread = float(math.sqrt(np.nansum(ab * (target_lat - centroid) ** 2) / np.nansum(ab))) if np.nansum(ab) > 0 else np.nan
        mid = (np.nanmin(target_lat) + np.nanmax(target_lat)) / 2.0
        north = x[target_lat >= mid]
        south = x[target_lat < mid]
        ns = float(_nanmean(north) - _nanmean(south)) if north.size and south.size else np.nan
        rows.append({
            "object": object_name,
            "day": day,
            "profile_valid": True,
            "profile_mean": wmean,
            "profile_spatial_std": std,
            "profile_amplitude": amp,
            "axis_lat": axis_lat,
            "centroid_lat": centroid,
            "spread_lat": spread,
            "NS_contrast": ns,
        })
    df = pd.DataFrame(rows)
    # daily speed as raw profile movement.
    speeds = []
    for day in df["day"]:
        day = int(day)
        if day <= cfg.curve_start or day >= X.shape[0]:
            speeds.append(np.nan)
            continue
        speeds.append(_weighted_distance(X[day], X[day - 1], weights))
    df["raw_profile_daily_speed"] = speeds
    return df


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

def _bootstrap_profiles(profiles_by_object: Dict[str, np.ndarray], sample_idx: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for obj, prof in profiles_by_object.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[obj] = np.nanmean(prof[sample_idx, :, :], axis=0)
    return out


def _summarize_samples(values: Sequence[float], positive_label: str, negative_label: str) -> Dict[str, object]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "median": np.nan, "q05": np.nan, "q95": np.nan, "q025": np.nan, "q975": np.nan,
            "P_positive": np.nan, "P_negative": np.nan, "decision": "invalid_no_samples",
        }
    q025 = float(np.quantile(finite, 0.025))
    q975 = float(np.quantile(finite, 0.975))
    ppos = float(np.mean(finite > 0))
    pneg = float(np.mean(finite < 0))
    if q025 > 0:
        dec = f"{positive_label}_supported"
    elif q975 < 0:
        dec = f"{negative_label}_supported"
    elif ppos >= 0.80:
        dec = f"{positive_label}_tendency"
    elif pneg >= 0.80:
        dec = f"{negative_label}_tendency"
    else:
        dec = "unresolved"
    return {
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "q025": q025,
        "q975": q975,
        "P_positive": ppos,
        "P_negative": pneg,
        "decision": dec,
    }


# -----------------------------------------------------------------------------
# Evidence families and gate
# -----------------------------------------------------------------------------

def _family_from_decisions(decisions: List[str], a_token: str, b_token: str) -> str:
    if not decisions:
        return "invalid"
    a_sup = sum(d == f"{a_token}_supported" for d in decisions)
    b_sup = sum(d == f"{b_token}_supported" for d in decisions)
    a_ten = sum(d == f"{a_token}_tendency" for d in decisions)
    b_ten = sum(d == f"{b_token}_tendency" for d in decisions)
    if a_sup > 0 and b_sup == 0:
        return "A_supported"
    if b_sup > 0 and a_sup == 0:
        return "B_supported"
    if a_sup > 0 and b_sup > 0:
        return "conflict"
    if a_ten > 0 and b_ten == 0:
        return "A_tendency"
    if b_ten > 0 and a_ten == 0:
        return "B_tendency"
    if a_ten > 0 and b_ten > 0:
        return "mixed_tendency"
    return "unresolved"


def _window_support_class(support: float) -> str:
    if not np.isfinite(support):
        return "unknown"
    if support >= 0.95:
        return "accepted_window_95"
    if support >= 0.80:
        return "candidate_window_80"
    if support >= 0.50:
        return "weak_window_50"
    return "unstable_window"


def _classify_window_family(delta_summary: pd.Series, overlap_frac: float) -> str:
    dec = str(delta_summary.get("decision", "unresolved"))
    if dec == "A_earlier_supported":
        return "A_supported"
    if dec == "B_earlier_supported":
        return "B_supported"
    if dec == "A_earlier_tendency":
        return "A_tendency"
    if dec == "B_earlier_tendency":
        return "B_tendency"
    if overlap_frac >= 0.25:
        return "co_transition"
    return "unresolved"


def _decide_pair(row: pd.Series) -> Tuple[str, str, str, bool, str, str]:
    """Return final_class, evidence_level, sensitivity_status, gate_pass, allowed, forbidden."""
    win = row.get("raw_profile_window_family", "unresolved")
    state_dist = row.get("state_dist_family", "unresolved")
    state_pat = row.get("state_pattern_family", "unresolved")
    grow_dist = row.get("growth_dist_family", "unresolved")
    grow_pat = row.get("growth_pattern_family", "unresolved")
    curve_fams = [state_dist, state_pat, grow_dist, grow_pat]
    a_sup = sum(x == "A_supported" for x in curve_fams)
    b_sup = sum(x == "B_supported" for x in curve_fams)
    a_ten = sum(x == "A_tendency" for x in curve_fams)
    b_ten = sum(x == "B_tendency" for x in curve_fams)
    conflict = any(x in ["conflict", "mixed_tendency"] for x in curve_fams)
    sensitivity = "clean"
    if conflict:
        sensitivity = "branch_or_baseline_conflict"
    if win == "co_transition":
        if a_sup + a_ten > b_sup + b_ten and (a_sup + a_ten) > 0:
            return (
                "co_transition_with_A_curve_tendency",
                "Level2_curve_tendency_only" if a_sup == 0 else "Level3_curve_supported_with_cotransition",
                sensitivity,
                True,
                "A and B co-transition in raw/profile object-window; A shows curve-level ahead tendency.",
                "Do not write A leads B as a hard object-window timing claim.",
            )
        if b_sup + b_ten > a_sup + a_ten and (b_sup + b_ten) > 0:
            return (
                "co_transition_with_B_curve_tendency",
                "Level2_curve_tendency_only" if b_sup == 0 else "Level3_curve_supported_with_cotransition",
                sensitivity,
                True,
                "A and B co-transition in raw/profile object-window; B shows curve-level ahead tendency.",
                "Do not write B leads A as a hard object-window timing claim.",
            )
        return ("co_transition", "Level3_window_overlap", sensitivity, True, "A and B co-transition in raw/profile object-window.", "Do not force a lead relation.")
    if win in ["A_supported", "A_tendency"]:
        if b_sup > 0 and a_sup == 0:
            return ("branch_split", "Level2_conflicting_families", "branch_split", False, "A raw/profile timing and B curve support conflict.", "Do not choose a single winner.")
        if a_sup > 0:
            return ("A_leads_B_candidate", "Level4_window_and_curve_supported", sensitivity, True, "A is earlier in raw/profile window and supported by curve metrics.", "Do not infer causality.")
        if a_ten > 0:
            return ("A_layer_specific_lead_or_front", "Level3_window_with_curve_tendency", sensitivity, True, "A is earlier/front in raw/profile window with curve tendency.", "Do not overstate as all-layer lead.")
        return ("A_window_only_downgraded", "Level2_window_only", sensitivity, False, "A raw/profile window is earlier but curve support is insufficient.", "Do not promote to final lead claim.")
    if win in ["B_supported", "B_tendency"]:
        if a_sup > 0 and b_sup == 0:
            return ("branch_split", "Level2_conflicting_families", "branch_split", False, "B raw/profile timing and A curve support conflict.", "Do not choose a single winner.")
        if b_sup > 0:
            return ("B_leads_A_candidate", "Level4_window_and_curve_supported", sensitivity, True, "B is earlier in raw/profile window and supported by curve metrics.", "Do not infer causality.")
        if b_ten > 0:
            return ("B_layer_specific_lead_or_front", "Level3_window_with_curve_tendency", sensitivity, True, "B is earlier/front in raw/profile window with curve tendency.", "Do not overstate as all-layer lead.")
        return ("B_window_only_downgraded", "Level2_window_only", sensitivity, False, "B raw/profile window is earlier but curve support is insufficient.", "Do not promote to final lead claim.")
    # No window support: curve-only signals are downgraded.
    if a_sup + a_ten > b_sup + b_ten and (a_sup + a_ten) > 0:
        return ("A_curve_tendency_only", "Level1_or_2_curve_only", sensitivity, False, "A shows curve-level tendency only.", "Do not promote to lead without raw/profile window support.")
    if b_sup + b_ten > a_sup + a_ten and (b_sup + b_ten) > 0:
        return ("B_curve_tendency_only", "Level1_or_2_curve_only", sensitivity, False, "B shows curve-level tendency only.", "Do not promote to lead without raw/profile window support.")
    return ("unresolved", "Level0_unresolved", sensitivity, False, "Current evidence does not support a timing claim.", "Do not interpret as synchrony unless co-transition is supported.")


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def _write_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _maybe_plot_curves(state: pd.DataFrame, out_dir: Path, cfg: CleanConfig) -> None:
    if cfg.skip_figures or plt is None:
        return
    fig_dir = out_dir / "figures"
    _ensure_dir(fig_dir)
    for branch in ["S_dist", "S_pattern"]:
        for base in state["baseline_config"].unique():
            fig, ax = plt.subplots(figsize=(10, 5))
            for obj in state["object"].unique():
                d = state[(state["object"] == obj) & (state["baseline_config"] == base)]
                ax.plot(d["day"], d[branch], label=obj, lw=1.5)
            ax.axvspan(cfg.w45_start, cfg.w45_end, alpha=0.15, label="W45")
            ax.set_title(f"{branch} ({base})")
            ax.set_xlabel("day")
            ax.set_ylabel(branch)
            ax.legend(ncol=3, fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"W45_clean_{branch}_{base}_v7_z_clean.png", dpi=180)
            plt.close(fig)


def _write_summary(path: Path, cfg: CleanConfig, selected: pd.DataFrame, family: pd.DataFrame, final_df: pd.DataFrame, downgraded_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# W45 multi-object pre-post clean mainline (V7-z-clean)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("This clean mainline keeps raw/profile as the only object-window detection input, and keeps pattern inside pre-post extraction via R_diff / S_pattern.")
    lines.append("")
    lines.append("## Object-window selected candidates")
    if selected.empty:
        lines.append("No selected raw/profile candidates were generated.")
    else:
        for _, r in selected.sort_values("object").iterrows():
            lines.append(f"- {r['object']}: peak day {r['selected_peak_day']}, window day {r['selected_band_start_day']}–{r['selected_band_end_day']}, support={r.get('bootstrap_support', np.nan):.3f}, class={r.get('support_class', 'unknown')}")
    lines.append("")
    lines.append("## Final claims")
    if final_df.empty:
        lines.append("No final claims passed the hardened evidence gate.")
    else:
        for _, r in final_df.iterrows():
            lines.append(f"- {r['object_A']}-{r['object_B']}: {r['final_structure_class']} ({r['evidence_level']})")
            lines.append(f"  - Allowed: {r['allowed_statement']}")
            lines.append(f"  - Forbidden: {r['forbidden_statement']}")
    lines.append("")
    lines.append("## Downgraded signals")
    if downgraded_df.empty:
        lines.append("No downgraded signals.")
    else:
        for _, r in downgraded_df.iterrows():
            lines.append(f"- {r['object_A']}-{r['object_B']}: {r['final_structure_class']} ({r['why_not_final']})")
    lines.append("")
    lines.append("## Method notes")
    lines.append("- Shape-normalized pattern detector is not used as a main object-window input in this clean mainline.")
    lines.append("- R_diff / S_pattern remains a pre-post pattern-similarity extraction branch.")
    lines.append("- Co-transition veto prevents multiple weak curve tendencies from becoming hard lead claims.")
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_W45_multi_object_prepost_clean_mainline_v7_z_clean(v7_root: Path | str) -> None:
    cfg = CleanConfig.from_env()
    v7_root = Path(v7_root)
    project_root = v7_root.parents[1]
    smoothed_path = Path(os.environ.get(
        "V7Z_CLEAN_SMOOTHED_FIELDS",
        str(project_root / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"),
    ))
    out_dir = Path(os.environ.get(
        "V7Z_CLEAN_OUTPUT_DIR",
        str(v7_root / "outputs" / OUTPUT_TAG),
    ))
    log_dir = v7_root / "logs" / OUTPUT_TAG
    _ensure_dir(out_dir)
    _ensure_dir(log_dir)

    run_meta = {
        "version": cfg.version,
        "output_tag": cfg.output_tag,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "v7_root": str(v7_root),
        "project_root": str(project_root),
        "smoothed_fields": str(smoothed_path),
        "config": asdict(cfg),
    }
    _write_json(out_dir / "W45_clean_run_config_v7_z_clean.json", run_meta)

    _log("[1/9] Load input fields")
    fields, key_audit = _load_npz_fields(smoothed_path)
    lat = np.asarray(fields["lat"], dtype=float)
    lon = np.asarray(fields["lon"], dtype=float)
    years = np.asarray(fields.get("years", [])) if "years" in fields else None
    _safe_to_csv(key_audit, out_dir / "W45_clean_input_key_audit_v7_z_clean.csv")

    _log("[2/9] Build object profiles")
    profiles_by_object: Dict[str, np.ndarray] = {}
    clim_by_object: Dict[str, np.ndarray] = {}
    lat_by_object: Dict[str, np.ndarray] = {}
    weights_by_object: Dict[str, np.ndarray] = {}
    obj_rows = []
    for spec in OBJECT_SPECS:
        arr0 = fields[spec.field_role]
        arr = _as_year_day_lat_lon(arr0, lat, lon, years)
        prof, target_lat, weights = _build_object_profile(arr, lat, lon, spec)
        profiles_by_object[spec.object_name] = prof
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            clim = np.nanmean(prof, axis=0)
        clim_by_object[spec.object_name] = clim
        lat_by_object[spec.object_name] = target_lat
        weights_by_object[spec.object_name] = weights
        obj_rows.append({
            "object": spec.object_name,
            "field_role": spec.field_role,
            "lon_min": spec.lon_min,
            "lon_max": spec.lon_max,
            "lat_min": spec.lat_min,
            "lat_max": spec.lat_max,
            "lat_step": spec.lat_step,
            "n_year": prof.shape[0],
            "n_day": prof.shape[1],
            "n_lat_feature": prof.shape[2],
            "valid_fraction": float(np.mean(np.isfinite(prof))),
        })
    object_registry = pd.DataFrame(obj_rows)
    _safe_to_csv(object_registry, out_dir / "W45_clean_object_registry_v7_z_clean.csv")
    _safe_to_csv(pd.DataFrame([asdict(b) for b in BASELINES]), out_dir / "W45_clean_baseline_config_table_v7_z_clean.csv")

    _log("[3/9] Run observed raw/profile object-window detectors")
    score_frames = []
    cand_frames = []
    for obj, X in clim_by_object.items():
        scores, cand = _run_detector_for_profile(X, weights_by_object[obj], cfg, obj)
        score_frames.append(scores)
        cand_frames.append(cand)
    score_df = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    cand_df = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    _safe_to_csv(score_df, out_dir / "W45_clean_raw_profile_detector_scores_v7_z_clean.csv")

    _log("[4/9] Bootstrap raw/profile object-window candidates")
    rng = np.random.default_rng(cfg.random_seed)
    n_year = next(iter(profiles_by_object.values())).shape[0]
    # candidate support by observed candidates
    support_records = []
    selected_peak_samples: Dict[str, List[float]] = {obj: [] for obj in profiles_by_object}
    candidate_match_samples: Dict[Tuple[str, str], List[float]] = {}
    for obj in profiles_by_object:
        for cid in cand_df.loc[cand_df["object"] == obj, "candidate_id"].tolist():
            candidate_match_samples[(obj, cid)] = []
    for ib in range(cfg.bootstrap_n):
        if ib % max(1, cfg.bootstrap_n // 10) == 0:
            _log(f"      bootstrap {ib}/{cfg.bootstrap_n}")
        sample = rng.integers(0, n_year, size=n_year)
        boot_clim = _bootstrap_profiles(profiles_by_object, sample)
        for obj, Xb in boot_clim.items():
            _, cb = _run_detector_for_profile(Xb, weights_by_object[obj], cfg, obj)
            sel_day = _select_boot_candidate(cb, cfg)
            selected_peak_samples[obj].append(sel_day)
            obs_c = cand_df[cand_df["object"] == obj]
            boot_days = cb["peak_day"].to_numpy(dtype=float) if not cb.empty else np.asarray([], dtype=float)
            for _, oc in obs_c.iterrows():
                od = float(oc["peak_day"])
                if boot_days.size == 0 or not np.isfinite(od):
                    candidate_match_samples[(obj, oc["candidate_id"])].append(np.nan)
                else:
                    idx = int(np.nanargmin(np.abs(boot_days - od)))
                    bd = boot_days[idx]
                    candidate_match_samples[(obj, oc["candidate_id"])].append(bd if abs(bd - od) <= cfg.near_peak_match_days else np.nan)
    # Summarize candidate support
    cand_df = cand_df.copy()
    boot_rows = []
    for idx, row in cand_df.iterrows():
        obj, cid, od = row["object"], row["candidate_id"], float(row["peak_day"])
        samples = np.asarray(candidate_match_samples.get((obj, cid), []), dtype=float)
        finite = samples[np.isfinite(samples)]
        support = float(np.mean(np.isfinite(samples) & (np.abs(samples - od) <= cfg.peak_match_days))) if samples.size else np.nan
        strict = float(np.mean(np.isfinite(samples) & (np.abs(samples - od) <= cfg.strict_peak_match_days))) if samples.size else np.nan
        near = float(np.mean(np.isfinite(samples) & (np.abs(samples - od) <= cfg.near_peak_match_days))) if samples.size else np.nan
        cand_df.loc[idx, "bootstrap_support"] = support
        cand_df.loc[idx, "strict_support"] = strict
        cand_df.loc[idx, "near_support"] = near
        cand_df.loc[idx, "support_class"] = _window_support_class(support)
        boot_rows.append({
            "object": obj,
            "candidate_id": cid,
            "peak_day": od,
            "bootstrap_support": support,
            "strict_support": strict,
            "near_support": near,
            "return_day_median": float(np.nanmedian(finite)) if finite.size else np.nan,
            "return_day_q025": _safe_quantile(finite, 0.025),
            "return_day_q975": _safe_quantile(finite, 0.975),
            "support_class": _window_support_class(support),
        })
    _safe_to_csv(cand_df, out_dir / "W45_clean_object_profile_window_registry_v7_z_clean.csv")
    boot_summary = pd.DataFrame(boot_rows)
    _safe_to_csv(boot_summary, out_dir / "W45_clean_profile_window_bootstrap_summary_v7_z_clean.csv")

    _log("[5/9] Select W45-relevant main raw/profile windows")
    selected_rows = []
    for obj in sorted(profiles_by_object.keys()):
        row = _select_w45_relevant_candidate(cand_df[cand_df["object"] == obj], cfg)
        if row is None:
            selected_rows.append({"object": obj, "selection_status": "no_candidate"})
        else:
            selected_rows.append({
                "object": obj,
                "selection_status": "selected",
                "selected_candidate_id": row["candidate_id"],
                "selected_peak_day": int(row["peak_day"]),
                "selected_band_start_day": int(row["band_start_day"]),
                "selected_band_end_day": int(row["band_end_day"]),
                "peak_score": float(row["peak_score"]),
                "bootstrap_support": float(row.get("bootstrap_support", np.nan)),
                "support_class": row.get("support_class", "unknown"),
                "overlap_fraction_with_W45": float(row.get("overlap_fraction_with_W45", np.nan)),
                "selection_reason": "prefer accepted/candidate, W45-overlap/front relevance, support, and score; avoids weak CP001-only timing",
            })
    selected_df = pd.DataFrame(selected_rows)
    _safe_to_csv(selected_df, out_dir / "W45_clean_main_window_selection_v7_z_clean.csv")

    _log("[6/9] Compute pre-post state/growth curves and physical feature support")
    state_frames = []
    growth_frames = []
    branch_frames = []
    feature_frames = []
    for obj, X in clim_by_object.items():
        st, gr, br = _compute_state_growth_for_object(obj, X, weights_by_object[obj], BASELINES, cfg)
        state_frames.append(st)
        growth_frames.append(gr)
        branch_frames.append(br)
        feature_frames.append(_profile_features_for_object(obj, X, lat_by_object[obj], weights_by_object[obj], cfg))
    state_df = pd.concat(state_frames, ignore_index=True)
    growth_df = pd.concat(growth_frames, ignore_index=True)
    branch_df = pd.concat(branch_frames, ignore_index=True)
    features_df = pd.concat(feature_frames, ignore_index=True)
    _safe_to_csv(state_df, out_dir / "W45_clean_state_progress_curves_v7_z_clean.csv")
    _safe_to_csv(growth_df, out_dir / "W45_clean_growth_speed_curves_v7_z_clean.csv")
    _safe_to_csv(branch_df, out_dir / "W45_clean_pattern_branch_validity_v7_z_clean.csv")
    _safe_to_csv(features_df, out_dir / "W45_clean_object_physical_feature_timeseries_v7_z_clean.csv")
    pair_obs = _pairwise_curve_metrics(state_df, cfg)
    _safe_to_csv(pair_obs, out_dir / "W45_clean_pairwise_observed_curve_metrics_v7_z_clean.csv")

    _log("[7/9] Bootstrap pre-post pairwise state/growth metrics")
    pair_sample_records = []
    selected_delta_records = []
    # Reuse independent RNG seed for reproducibility in metric bootstrap.
    rng = np.random.default_rng(cfg.random_seed + 101)
    pair_keys = [(a, b) for i, a in enumerate(sorted(profiles_by_object.keys())) for b in sorted(profiles_by_object.keys())[i + 1:]]
    for ib in range(cfg.bootstrap_n):
        if ib % max(1, cfg.bootstrap_n // 10) == 0:
            _log(f"      metric bootstrap {ib}/{cfg.bootstrap_n}")
        sample = rng.integers(0, n_year, size=n_year)
        boot_clim = _bootstrap_profiles(profiles_by_object, sample)
        # selected peak day per object
        boot_selected = {}
        boot_selected_band = {}
        for obj, Xb in boot_clim.items():
            _, cb = _run_detector_for_profile(Xb, weights_by_object[obj], cfg, obj)
            sd = _select_boot_candidate(cb, cfg)
            boot_selected[obj] = sd
            if not cb.empty and np.isfinite(sd):
                # closest candidate row to selected day
                k = int(np.nanargmin(np.abs(cb["peak_day"].to_numpy(dtype=float) - sd)))
                rr = cb.iloc[k]
                boot_selected_band[obj] = (float(rr["band_start_day"]), float(rr["band_end_day"]))
            else:
                boot_selected_band[obj] = (np.nan, np.nan)
        # State/growth bootstrap curves
        st_list = []
        for obj, Xb in boot_clim.items():
            st, _, _ = _compute_state_growth_for_object(obj, Xb, weights_by_object[obj], BASELINES, cfg)
            st_list.append(st)
        st_b = pd.concat(st_list, ignore_index=True)
        pair_b = _pairwise_curve_metrics(st_b, cfg)
        pair_b["bootstrap_id"] = ib
        pair_sample_records.append(pair_b)
        for a, b in pair_keys:
            da = boot_selected.get(a, np.nan)
            db = boot_selected.get(b, np.nan)
            selected_delta_records.append({
                "bootstrap_id": ib,
                "object_A": a,
                "object_B": b,
                "A_selected_peak_day": da,
                "B_selected_peak_day": db,
                "selected_peak_delta_B_minus_A": db - da if np.isfinite(da) and np.isfinite(db) else np.nan,
            })
    pair_samples = pd.concat(pair_sample_records, ignore_index=True) if pair_sample_records else pd.DataFrame()
    selected_delta_samples = pd.DataFrame(selected_delta_records)
    if cfg.save_bootstrap_metric_samples:
        _safe_to_csv(pair_samples, out_dir / "W45_clean_pairwise_bootstrap_metric_samples_v7_z_clean.csv")
        _safe_to_csv(selected_delta_samples, out_dir / "W45_clean_selected_peak_delta_bootstrap_samples_v7_z_clean.csv")

    # Summaries for curve metrics
    summary_rows = []
    if not pair_obs.empty:
        for _, obs in pair_obs.iterrows():
            mask = (
                (pair_samples["object_A"] == obs["object_A"]) &
                (pair_samples["object_B"] == obs["object_B"]) &
                (pair_samples["baseline_config"] == obs["baseline_config"]) &
                (pair_samples["metric_family"] == obs["metric_family"]) &
                (pair_samples["metric_name"] == obs["metric_name"])
            ) if not pair_samples.empty else pd.Series([], dtype=bool)
            vals = pair_samples.loc[mask, "observed"].to_numpy(dtype=float) if not pair_samples.empty else np.asarray([], dtype=float)
            s = _summarize_samples(vals, "A", "B")
            row = obs.to_dict()
            row.update({
                "bootstrap_median": s["median"],
                "bootstrap_q05": s["q05"],
                "bootstrap_q95": s["q95"],
                "bootstrap_q025": s["q025"],
                "bootstrap_q975": s["q975"],
                "P_A_direction": s["P_positive"],
                "P_B_direction": s["P_negative"],
                "decision": s["decision"],
            })
            summary_rows.append(row)
    pair_summary = pd.DataFrame(summary_rows)
    _safe_to_csv(pair_summary, out_dir / "W45_clean_pairwise_stat_summary_v7_z_clean.csv")

    # Summarize selected peak deltas
    selected_delta_summary_rows = []
    for a, b in pair_keys:
        vals = selected_delta_samples[(selected_delta_samples["object_A"] == a) & (selected_delta_samples["object_B"] == b)]["selected_peak_delta_B_minus_A"].to_numpy(dtype=float)
        s = _summarize_samples(vals, "A_earlier", "B_earlier")
        # observed selected delta and overlap
        sa = selected_df[selected_df["object"] == a].iloc[0]
        sb = selected_df[selected_df["object"] == b].iloc[0]
        if sa.get("selection_status") == "selected" and sb.get("selection_status") == "selected":
            odelta = float(sb["selected_peak_day"] - sa["selected_peak_day"])
            _, ov_frac = _interval_overlap(float(sa["selected_band_start_day"]), float(sa["selected_band_end_day"]), float(sb["selected_band_start_day"]), float(sb["selected_band_end_day"]))
        else:
            odelta = np.nan
            ov_frac = np.nan
        selected_delta_summary_rows.append({
            "object_A": a,
            "object_B": b,
            "delta_definition": "B_selected_peak_day_minus_A_selected_peak_day; positive means A selected raw/profile window earlier",
            "delta_observed": odelta,
            "delta_median": s["median"],
            "delta_q025": s["q025"],
            "delta_q975": s["q975"],
            "P_A_earlier": s["P_positive"],
            "P_B_earlier": s["P_negative"],
            "decision": s["decision"],
            "observed_selected_window_overlap_fraction": ov_frac,
        })
    selected_delta_summary = pd.DataFrame(selected_delta_summary_rows)
    _safe_to_csv(selected_delta_summary, out_dir / "W45_clean_selected_peak_delta_v7_z_clean.csv")

    _log("[8/9] Build evidence families and apply hardened gate")
    family_rows = []
    final_rows = []
    downgraded_rows = []
    for a, b in pair_keys:
        ds = selected_delta_summary[(selected_delta_summary["object_A"] == a) & (selected_delta_summary["object_B"] == b)].iloc[0]
        winfam = _classify_window_family(ds, float(ds.get("observed_selected_window_overlap_fraction", 0)))
        fam_map = {"raw_profile_window_family": winfam}
        for fam_name in ["state_dist", "state_pattern", "growth_dist", "growth_pattern"]:
            metric_family = fam_name.replace("_dist", "_dist").replace("_pattern", "_pattern")
            if fam_name == "state_dist":
                mf = "state_dist"
            elif fam_name == "state_pattern":
                mf = "state_pattern"
            elif fam_name == "growth_dist":
                mf = "growth_dist"
            else:
                mf = "growth_pattern"
            sub = pair_summary[(pair_summary["object_A"] == a) & (pair_summary["object_B"] == b) & (pair_summary["metric_family"] == mf)]
            fam_map[f"{fam_name}_family"] = _family_from_decisions(sub["decision"].dropna().astype(str).tolist(), "A", "B")
        row = {"object_A": a, "object_B": b, **fam_map}
        final_class, evid, sens, pass_gate, allowed, forbidden = _decide_pair(pd.Series(row))
        row.update({
            "final_structure_class": final_class,
            "evidence_level": evid,
            "sensitivity_status": sens,
            "gate_pass": pass_gate,
            "allowed_statement": allowed,
            "forbidden_statement": forbidden,
        })
        family_rows.append(row)
        if pass_gate:
            final_rows.append({
                "claim_id": f"{a}_{b}_{final_class}",
                "object_A": a,
                "object_B": b,
                "final_structure_class": final_class,
                "evidence_level": evid,
                "sensitivity_status": sens,
                "supporting_evidence_families": json.dumps(fam_map, ensure_ascii=False),
                "allowed_statement": allowed,
                "forbidden_statement": forbidden,
            })
        else:
            downgraded_rows.append({
                "signal_id": f"{a}_{b}_{final_class}",
                "object_A": a,
                "object_B": b,
                "final_structure_class": final_class,
                "evidence_level": evid,
                "why_not_final": forbidden,
                "available_evidence_families": json.dumps(fam_map, ensure_ascii=False),
                "possible_interpretation": allowed,
            })
    family_df = pd.DataFrame(family_rows)
    final_df = pd.DataFrame(final_rows)
    downgraded_df = pd.DataFrame(downgraded_rows)
    _safe_to_csv(family_df, out_dir / "W45_clean_evidence_family_summary_v7_z_clean.csv")
    _safe_to_csv(family_df, out_dir / "W45_clean_timing_structure_classification_v7_z_clean.csv")
    gate_df = family_df[["object_A", "object_B", "final_structure_class", "evidence_level", "sensitivity_status", "gate_pass", "allowed_statement", "forbidden_statement"]].copy()
    _safe_to_csv(gate_df, out_dir / "W45_clean_evidence_gate_table_v7_z_clean.csv")
    _safe_to_csv(final_df, out_dir / "W45_clean_final_claim_registry_v7_z_clean.csv")
    _safe_to_csv(downgraded_df, out_dir / "W45_clean_downgraded_signal_registry_v7_z_clean.csv")

    _log("[9/9] Write summary and figures")
    _maybe_plot_curves(state_df, out_dir, cfg)
    _write_summary(out_dir / "W45_clean_mainline_summary_v7_z_clean.md", cfg, selected_df, family_df, final_df, downgraded_df)
    run_meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_meta["n_final_claims"] = int(len(final_df))
    run_meta["n_downgraded_signals"] = int(len(downgraded_df))
    _write_json(out_dir / "run_meta.json", run_meta)
    _write_json(out_dir / "summary.json", {
        "version": cfg.version,
        "output_tag": cfg.output_tag,
        "n_objects": len(OBJECT_SPECS),
        "n_bootstrap": cfg.bootstrap_n,
        "n_final_claims": int(len(final_df)),
        "n_downgraded_signals": int(len(downgraded_df)),
        "core_method": "raw/profile object-window + pre-post S_dist/R_diff-S_pattern + hardened evidence-family gate",
        "shape_pattern_detector_status": "excluded_from_main_object_window; use only as external sensitivity probe",
    })
    _log(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    # Allow direct module execution during debugging.
    root = Path(__file__).resolve().parents[2]
    run_W45_multi_object_prepost_clean_mainline_v7_z_clean(root)
