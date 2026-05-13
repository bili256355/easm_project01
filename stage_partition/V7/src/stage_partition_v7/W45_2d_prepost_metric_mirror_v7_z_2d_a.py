"""
V7-z-2d-a: W45 2D-field pre-post metric mirror.

Purpose
-------
A minimal 2D-field mirror of the cleaned profile-based W45 pre-post mainline.
It does NOT run 2D change-point detection and it does NOT rewrite final claims.
It only computes the 2D analogues of the profile mainline pre-post metrics:

    S_dist_2d, R_pre_2d, R_post_2d, R_diff_2d, S_pattern_2d,
    V_dist_2d, V_pattern_2d,

then compares these field-based metrics with the existing profile-based clean
mainline metrics when those outputs are available.

Interpretation boundary
-----------------------
The clean mainline remains profile-based. This 2D mirror is an audit / comparison
layer: it asks whether the profile-based pre-post signals have corresponding
2D-field pre-post signals. It does not generate a replacement final claim registry.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import time
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


VERSION = "v7_z_2d_a"
OUTPUT_TAG = "W45_2d_prepost_metric_mirror_v7_z_2d_a"


@dataclass(frozen=True)
class ObjectSpec:
    object_name: str
    field_role: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    pre_start: int
    pre_end: int
    post_start: int
    post_end: int
    role: str


@dataclass
class Mirror2DConfig:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    w45_start: int = 40
    w45_end: int = 48
    anchor_day: int = 45
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
    bootstrap_n: int = 1000
    random_seed: int = 42
    low_dynamic_range_eps: float = 1e-10
    norm_eps: float = 1e-12
    save_bootstrap_metric_samples: bool = False
    skip_figures: bool = False

    @staticmethod
    def from_env() -> "Mirror2DConfig":
        cfg = Mirror2DConfig()
        debug_n = os.environ.get("V7Z_2D_A_DEBUG_N_BOOTSTRAP") or os.environ.get("V7Z_DEBUG_N_BOOTSTRAP")
        if debug_n:
            cfg.bootstrap_n = int(debug_n)
        if os.environ.get("V7Z_2D_A_SKIP_FIGURES") == "1" or os.environ.get("V7Z_SKIP_FIGURES") == "1":
            cfg.skip_figures = True
        if os.environ.get("V7Z_2D_A_SAVE_BOOTSTRAP_SAMPLES") == "1":
            cfg.save_bootstrap_metric_samples = True
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
# Utilities
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _nanmean(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


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


def _weighted_norm_flat(x: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    return float(math.sqrt(np.sum(w[m] * x[m] ** 2) / np.sum(w[m])))


def _weighted_distance_flat(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_norm_flat(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), weights)


def _weighted_corr_flat(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0)
    if np.sum(m) < 3:
        return float("nan")
    am = np.sum(a[m] * w[m]) / np.sum(w[m])
    bm = np.sum(b[m] * w[m]) / np.sum(w[m])
    aa = a[m] - am
    bb = b[m] - bm
    cov = np.sum(w[m] * aa * bb) / np.sum(w[m])
    va = np.sum(w[m] * aa * aa) / np.sum(w[m])
    vb = np.sum(w[m] * bb * bb) / np.sum(w[m])
    if va <= 0 or vb <= 0:
        return float("nan")
    return float(cov / math.sqrt(va * vb))


def _growth_center(days: np.ndarray, values: np.ndarray) -> float:
    days = np.asarray(days, dtype=float)
    v = np.asarray(values, dtype=float)
    pos = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    if np.nansum(pos) <= 0:
        return float("nan")
    return float(np.nansum(days * pos) / np.nansum(pos))


def _growth_area(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.nansum(np.where(np.isfinite(v) & (v > 0), v, 0.0)))


def _segment_mask(days: np.ndarray, start: int, end: int) -> np.ndarray:
    d = np.asarray(days, dtype=int)
    return (d >= start) & (d <= end)


def _segment_mean(days: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    m = _segment_mask(days, start, end)
    if not np.any(m):
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=float)[m]))


def _segment_growth_share(days: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    v = np.asarray(values, dtype=float)
    pos = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    total = float(np.nansum(pos))
    if total <= 0:
        return float("nan")
    m = _segment_mask(days, start, end)
    return float(np.nansum(pos[m]) / total)


# -----------------------------------------------------------------------------
# Loading and field extraction
# -----------------------------------------------------------------------------

def _find_key(npz: np.lib.npyio.NpzFile, aliases: Sequence[str]) -> Optional[str]:
    lower = {k.lower(): k for k in npz.files}
    for a in aliases:
        if a in npz.files:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _load_npz_fields(path: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"smoothed fields not found: {path}")
    npz = np.load(path, allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    rows = []
    for role, aliases in FIELD_ALIASES.items():
        k = _find_key(npz, aliases)
        rows.append({"role": role, "resolved_key": k, "status": "found" if k else "missing"})
        if k:
            out[role] = np.asarray(npz[k])
    required = ["lat", "lon", "precip", "v850", "z500", "u200"]
    missing = [r for r in required if r not in out]
    audit = pd.DataFrame(rows)
    if missing:
        raise KeyError(f"Missing required fields {missing}; available keys={npz.files}")
    return out, audit


def _as_year_day_lat_lon(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, years: Optional[np.ndarray]) -> np.ndarray:
    """Return field array shaped [year, day, lat, lon]."""
    arr = np.asarray(arr, dtype=float)
    nlat, nlon = len(lat), len(lon)
    if arr.ndim == 4:
        if arr.shape[2] == nlat and arr.shape[3] == nlon:
            return arr
        if arr.shape[2] == nlon and arr.shape[3] == nlat:
            return np.transpose(arr, (0, 1, 3, 2))
        if years is not None and arr.shape[1] == len(years) and arr.shape[2] == nlat and arr.shape[3] == nlon:
            return np.transpose(arr, (1, 0, 2, 3))
    if arr.ndim == 3:
        if arr.shape[1] == nlat and arr.shape[2] == nlon:
            nt = arr.shape[0]
            if years is not None and len(years) > 0 and nt % len(years) == 0:
                return arr.reshape(len(years), nt // len(years), nlat, nlon)
            return arr.reshape(1, nt, nlat, nlon)
        if arr.shape[1] == nlon and arr.shape[2] == nlat:
            arr2 = np.transpose(arr, (0, 2, 1))
            nt = arr2.shape[0]
            if years is not None and len(years) > 0 and nt % len(years) == 0:
                return arr2.reshape(len(years), nt // len(years), nlat, nlon)
            return arr2.reshape(1, nt, nlat, nlon)
    raise ValueError(f"Cannot infer [year,day,lat,lon] dimensions for {arr.shape}")


def _extract_region_2d(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: ObjectSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return [year, day, lat_region, lon_region], lat_region, lon_region, 2D weights."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lo_lat, hi_lat = min(spec.lat_min, spec.lat_max), max(spec.lat_min, spec.lat_max)
    lo_lon, hi_lon = min(spec.lon_min, spec.lon_max), max(spec.lon_min, spec.lon_max)
    lat_mask = (lat >= lo_lat) & (lat <= hi_lat)
    lon_mask = (lon >= lo_lon) & (lon <= hi_lon)
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points for {spec.object_name}: {lo_lat}-{hi_lat}")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points for {spec.object_name}: {lo_lon}-{hi_lon}")
    region = np.asarray(field, dtype=float)[:, :, lat_mask, :][:, :, :, lon_mask]
    lat_sel = lat[lat_mask]
    lon_sel = lon[lon_mask]
    # Sort latitude ascending for deterministic output; data follows the same order.
    order = np.argsort(lat_sel)
    lat_sel = lat_sel[order]
    region = region[:, :, order, :]
    w_lat = np.cos(np.deg2rad(lat_sel))
    w_lat = np.where(np.isfinite(w_lat) & (w_lat > 0), w_lat, 1.0)
    weights = np.repeat(w_lat[:, None], len(lon_sel), axis=1)
    return region, lat_sel, lon_sel, weights


# -----------------------------------------------------------------------------
# 2D pre-post metrics
# -----------------------------------------------------------------------------

def _compute_2d_curves(region_by_year: np.ndarray, weights_2d: np.ndarray, baseline: BaselineConfig, cfg: Mirror2DConfig) -> pd.DataFrame:
    """Compute observed 2D pre-post curves from year/day region field."""
    clim = _nanmean(region_by_year, axis=0)  # day, lat, lon
    ndays = clim.shape[0]
    day_end = min(cfg.curve_end, ndays - 1)
    days = np.arange(cfg.curve_start, day_end + 1, dtype=int)
    pre_days = np.arange(baseline.pre_start, min(baseline.pre_end, ndays - 1) + 1)
    post_days = np.arange(baseline.post_start, min(baseline.post_end, ndays - 1) + 1)
    if len(pre_days) == 0 or len(post_days) == 0:
        raise ValueError(f"Invalid baseline days for ndays={ndays}: {baseline}")
    F_pre = _nanmean(clim[pre_days], axis=0)
    F_post = _nanmean(clim[post_days], axis=0)
    # Dynamic range of R_diff for S_pattern normalization.
    pre_rdiff = []
    post_rdiff = []
    for d in pre_days:
        rp = _weighted_corr_flat(clim[d], F_pre, weights_2d)
        rpost = _weighted_corr_flat(clim[d], F_post, weights_2d)
        pre_rdiff.append(rpost - rp if np.isfinite(rp) and np.isfinite(rpost) else np.nan)
    for d in post_days:
        rp = _weighted_corr_flat(clim[d], F_pre, weights_2d)
        rpost = _weighted_corr_flat(clim[d], F_post, weights_2d)
        post_rdiff.append(rpost - rp if np.isfinite(rp) and np.isfinite(rpost) else np.nan)
    pre_mean = float(np.nanmean(pre_rdiff)) if np.any(np.isfinite(pre_rdiff)) else float("nan")
    post_mean = float(np.nanmean(post_rdiff)) if np.any(np.isfinite(post_rdiff)) else float("nan")
    dynamic = post_mean - pre_mean if np.isfinite(pre_mean) and np.isfinite(post_mean) else float("nan")
    branch_valid = bool(np.isfinite(dynamic) and abs(dynamic) > cfg.low_dynamic_range_eps)
    rows = []
    prev_sdist = np.nan
    prev_spat = np.nan
    for d in days:
        F = clim[d]
        D_pre = _weighted_distance_flat(F, F_pre, weights_2d)
        D_post = _weighted_distance_flat(F, F_post, weights_2d)
        denom = D_pre + D_post if np.isfinite(D_pre) and np.isfinite(D_post) else np.nan
        S_dist = float(D_pre / denom) if np.isfinite(denom) and denom > cfg.norm_eps else np.nan
        R_pre = _weighted_corr_flat(F, F_pre, weights_2d)
        R_post = _weighted_corr_flat(F, F_post, weights_2d)
        R_diff = R_post - R_pre if np.isfinite(R_pre) and np.isfinite(R_post) else np.nan
        S_pattern = (R_diff - pre_mean) / dynamic if branch_valid and np.isfinite(R_diff) else np.nan
        V_dist = S_dist - prev_sdist if np.isfinite(S_dist) and np.isfinite(prev_sdist) else np.nan
        V_pattern = S_pattern - prev_spat if np.isfinite(S_pattern) and np.isfinite(prev_spat) else np.nan
        rows.append({
            "baseline_config": baseline.name,
            "baseline_role": baseline.role,
            "day": int(d),
            "D_pre_2d": D_pre,
            "D_post_2d": D_post,
            "S_dist_2d": S_dist,
            "R_pre_2d": R_pre,
            "R_post_2d": R_post,
            "R_diff_2d": R_diff,
            "S_pattern_2d": S_pattern,
            "V_dist_2d": V_dist,
            "V_pattern_2d": V_pattern,
            "pattern_dynamic_range_2d": dynamic,
            "pattern_branch_valid_2d": branch_valid,
            "basis": "2d_field",
        })
        prev_sdist = S_dist
        prev_spat = S_pattern
    return pd.DataFrame(rows)


def _single_object_metrics_from_curves(curves: pd.DataFrame, cfg: Mirror2DConfig) -> Dict[str, float]:
    days = curves["day"].to_numpy(dtype=int)
    out: Dict[str, float] = {}
    for branch, state_col, growth_col in [
        ("dist", "S_dist_2d", "V_dist_2d"),
        ("pattern", "S_pattern_2d", "V_pattern_2d"),
    ]:
        s = curves[state_col].to_numpy(dtype=float)
        g = curves[growth_col].to_numpy(dtype=float)
        out[f"{branch}_mean_compare"] = _segment_mean(days, s, cfg.compare_start, cfg.compare_end)
        out[f"{branch}_mean_early"] = _segment_mean(days, s, cfg.early_start, cfg.early_end)
        out[f"{branch}_mean_core"] = _segment_mean(days, s, cfg.core_start, cfg.core_end)
        out[f"{branch}_mean_late"] = _segment_mean(days, s, cfg.late_start, cfg.late_end)
        out[f"{branch}_growth_center"] = _growth_center(days, g)
        out[f"{branch}_positive_growth_area"] = _growth_area(g)
        out[f"{branch}_early_growth_share"] = _segment_growth_share(days, g, cfg.early_start, cfg.early_end)
        out[f"{branch}_core_growth_share"] = _segment_growth_share(days, g, cfg.core_start, cfg.core_end)
        out[f"{branch}_late_growth_share"] = _segment_growth_share(days, g, cfg.late_start, cfg.late_end)
    return out


def _pairwise_metrics(metric_by_obj: Dict[str, Dict[str, float]], objects: Sequence[str]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            ma = metric_by_obj[a]
            mb = metric_by_obj[b]
            for branch in ["dist", "pattern"]:
                for segment in ["compare", "early", "core", "late"]:
                    key = f"{branch}_mean_{segment}"
                    delta = ma.get(key, np.nan) - mb.get(key, np.nan)
                    rows.append({
                        "object_A": a,
                        "object_B": b,
                        "metric_family": f"2d_{branch}_state_{segment}",
                        "delta_definition": f"{key}_A_minus_B; positive means A higher/ahead",
                        "delta_observed": delta,
                    })
                key = f"{branch}_growth_center"
                delta = mb.get(key, np.nan) - ma.get(key, np.nan)
                rows.append({
                    "object_A": a,
                    "object_B": b,
                    "metric_family": f"2d_{branch}_growth_center",
                    "delta_definition": f"{key}_B_minus_A; positive means A earlier growth",
                    "delta_observed": delta,
                })
                for seg in ["early", "core", "late"]:
                    key = f"{branch}_{seg}_growth_share"
                    delta = ma.get(key, np.nan) - mb.get(key, np.nan)
                    rows.append({
                        "object_A": a,
                        "object_B": b,
                        "metric_family": f"2d_{branch}_{seg}_growth_share",
                        "delta_definition": f"{key}_A_minus_B; positive means A larger share",
                        "delta_observed": delta,
                    })
    return pd.DataFrame(rows)


def _summarize_bootstrap(values: Sequence[float], observed: float, positive_label: str = "positive_supported") -> Dict[str, object]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    ppos = _prob_positive(arr)
    pneg = _prob_negative(arr)
    q025 = _safe_quantile(arr, 0.025)
    q975 = _safe_quantile(arr, 0.975)
    if np.isfinite(q025) and q025 > 0:
        decision = positive_label
    elif np.isfinite(q975) and q975 < 0:
        decision = "negative_supported"
    elif np.isfinite(ppos) and ppos >= 0.80:
        decision = "positive_tendency"
    elif np.isfinite(pneg) and pneg >= 0.80:
        decision = "negative_tendency"
    else:
        decision = "unresolved"
    return {
        "observed": observed,
        "bootstrap_median": _safe_quantile(arr, 0.5),
        "q025": q025,
        "q975": q975,
        "P_positive": ppos,
        "P_negative": pneg,
        "decision": decision,
        "n_valid_bootstrap": int(arr.size),
    }


# -----------------------------------------------------------------------------
# Clean-profile comparison
# -----------------------------------------------------------------------------

def _load_clean_profile_curves(v7_root: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Path, str]:
    env_dir = os.environ.get("V7Z_CLEAN_RESULT_DIR")
    if env_dir:
        clean_dir = Path(env_dir)
    else:
        clean_dir = v7_root / "outputs" / "W45_multi_object_prepost_clean_mainline_v7_z_clean"
    state_path = clean_dir / "W45_clean_state_progress_curves_v7_z_clean.csv"
    growth_path = clean_dir / "W45_clean_growth_speed_curves_v7_z_clean.csv"
    if state_path.exists():
        state = pd.read_csv(state_path)
    else:
        return None, None, clean_dir, "missing_clean_state_progress_curves"
    if growth_path.exists():
        growth = pd.read_csv(growth_path)
    else:
        growth = None
    return state, growth, clean_dir, "loaded" if state is not None else "missing"


def _profile_metric_summary(profile_curves: pd.DataFrame, cfg: Mirror2DConfig) -> pd.DataFrame:
    rows = []
    for (obj, base), g in profile_curves.groupby(["object", "baseline_config"]):
        days = g["day"].to_numpy(dtype=int)
        for branch, state_col, growth_col in [("dist", "S_dist", "V_dist"), ("pattern", "S_pattern", "V_pattern")]:
            if state_col not in g.columns:
                continue
            s = g[state_col].to_numpy(dtype=float)
            v = g[growth_col].to_numpy(dtype=float) if growth_col in g.columns else np.full_like(s, np.nan)
            rows.extend([
                {"object": obj, "baseline_config": base, "basis": "lat_profile", "metric_family": f"{branch}_mean_compare", "value": _segment_mean(days, s, cfg.compare_start, cfg.compare_end)},
                {"object": obj, "baseline_config": base, "basis": "lat_profile", "metric_family": f"{branch}_mean_early", "value": _segment_mean(days, s, cfg.early_start, cfg.early_end)},
                {"object": obj, "baseline_config": base, "basis": "lat_profile", "metric_family": f"{branch}_mean_core", "value": _segment_mean(days, s, cfg.core_start, cfg.core_end)},
                {"object": obj, "baseline_config": base, "basis": "lat_profile", "metric_family": f"{branch}_mean_late", "value": _segment_mean(days, s, cfg.late_start, cfg.late_end)},
                {"object": obj, "baseline_config": base, "basis": "lat_profile", "metric_family": f"{branch}_growth_center", "value": _growth_center(days, v)},
                {"object": obj, "baseline_config": base, "basis": "lat_profile", "metric_family": f"{branch}_early_growth_share", "value": _segment_growth_share(days, v, cfg.early_start, cfg.early_end)},
                {"object": obj, "baseline_config": base, "basis": "lat_profile", "metric_family": f"{branch}_core_growth_share", "value": _segment_growth_share(days, v, cfg.core_start, cfg.core_end)},
                {"object": obj, "baseline_config": base, "basis": "lat_profile", "metric_family": f"{branch}_late_growth_share", "value": _segment_growth_share(days, v, cfg.late_start, cfg.late_end)},
            ])
    return pd.DataFrame(rows)


def _field2d_metric_summary_from_curves(curves: pd.DataFrame, cfg: Mirror2DConfig) -> pd.DataFrame:
    rows = []
    for (obj, base), g in curves.groupby(["object", "baseline_config"]):
        vals = _single_object_metrics_from_curves(g, cfg)
        for k, v in vals.items():
            rows.append({"object": obj, "baseline_config": base, "basis": "2d_field", "metric_family": k, "value": v})
    return pd.DataFrame(rows)


def _agreement_class(profile_value: float, field_value: float, metric_family: str) -> str:
    if not np.isfinite(profile_value) or not np.isfinite(field_value):
        return "unresolved"
    # For bounded state means/shares, use absolute thresholds. For centers, use days.
    if "growth_center" in metric_family:
        diff = field_value - profile_value
        if abs(diff) <= 2.0:
            return "consistent"
        if abs(diff) <= 5.0:
            return "similar_tendency"
        return "profile_2d_offset"
    diff = field_value - profile_value
    if abs(diff) <= 0.10:
        return "consistent"
    if abs(diff) <= 0.25:
        return "similar_tendency"
    # If both on same side of 0.5 for state-like means, call partial.
    if (profile_value - 0.5) * (field_value - 0.5) > 0:
        return "same_side_different_magnitude"
    return "opposite_or_cross_threshold"


def _build_profile_vs_2d_comparison(profile_curves: Optional[pd.DataFrame], curves2d: pd.DataFrame, cfg: Mirror2DConfig) -> pd.DataFrame:
    field_summary = _field2d_metric_summary_from_curves(curves2d, cfg)
    if profile_curves is None or profile_curves.empty:
        out = field_summary.copy()
        out = out.rename(columns={"value": "field2d_value"})
        out["profile_value"] = np.nan
        out["difference_2d_minus_profile"] = np.nan
        out["agreement_class"] = "profile_missing"
        out["interpretation"] = "Clean profile curves were not found; only 2D metrics are available."
        return out[["object", "baseline_config", "metric_family", "profile_value", "field2d_value", "difference_2d_minus_profile", "agreement_class", "interpretation"]]
    prof_summary = _profile_metric_summary(profile_curves, cfg)
    merged = prof_summary.merge(
        field_summary,
        on=["object", "baseline_config", "metric_family"],
        how="outer",
        suffixes=("_profile", "_2d"),
    )
    rows = []
    for _, r in merged.iterrows():
        pv = r.get("value_profile", np.nan)
        fv = r.get("value_2d", np.nan)
        cls = _agreement_class(pv, fv, str(r.get("metric_family")))
        diff = fv - pv if np.isfinite(pv) and np.isfinite(fv) else np.nan
        if cls == "consistent":
            interp = "2D-field metric is consistent with the profile-based metric."
        elif cls == "similar_tendency":
            interp = "2D-field metric has a similar tendency but differs moderately in magnitude/timing."
        elif cls == "same_side_different_magnitude":
            interp = "Profile and 2D metrics are on the same side of the reference level, but magnitudes differ."
        elif cls == "opposite_or_cross_threshold":
            interp = "Profile and 2D metrics may differ in interpretation; inspect this object/baseline."
        elif cls == "profile_2d_offset":
            interp = "Growth center differs by more than the small-offset threshold."
        else:
            interp = "Comparison unresolved due to missing or invalid values."
        rows.append({
            "object": r.get("object"),
            "baseline_config": r.get("baseline_config"),
            "metric_family": r.get("metric_family"),
            "profile_value": pv,
            "field2d_value": fv,
            "difference_2d_minus_profile": diff,
            "agreement_class": cls,
            "interpretation": interp,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

def _compute_all_curves_for_year_indices(regions: Dict[str, Tuple[np.ndarray, np.ndarray]], year_idx: np.ndarray, cfg: Mirror2DConfig) -> pd.DataFrame:
    rows = []
    for spec in OBJECT_SPECS:
        region, weights = regions[spec.object_name]
        boot_region = region[year_idx]
        for base in BASELINES:
            df = _compute_2d_curves(boot_region, weights, base, cfg)
            df.insert(0, "object", spec.object_name)
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _observed_pairwise_summary(curves2d: pd.DataFrame, cfg: Mirror2DConfig) -> pd.DataFrame:
    rows = []
    for base, gbase in curves2d.groupby("baseline_config"):
        metric_by_obj = {}
        objects = []
        for obj, g in gbase.groupby("object"):
            objects.append(obj)
            metric_by_obj[obj] = _single_object_metrics_from_curves(g, cfg)
        pair = _pairwise_metrics(metric_by_obj, sorted(objects))
        pair.insert(0, "baseline_config", base)
        rows.append(pair)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _bootstrap_summaries(regions: Dict[str, Tuple[np.ndarray, np.ndarray]], observed_curves: pd.DataFrame, cfg: Mirror2DConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    rng = np.random.default_rng(cfg.random_seed)
    # Determine number of years from first object.
    first_region = next(iter(regions.values()))[0]
    ny = first_region.shape[0]
    observed_single_rows = []
    for (obj, base), g in observed_curves.groupby(["object", "baseline_config"]):
        vals = _single_object_metrics_from_curves(g, cfg)
        for metric, value in vals.items():
            observed_single_rows.append({"object": obj, "baseline_config": base, "metric": metric, "observed": value})
    obs_single = pd.DataFrame(observed_single_rows)
    obs_pair = _observed_pairwise_summary(observed_curves, cfg)

    single_samples: Dict[Tuple[str, str, str], List[float]] = {(r.object, r.baseline_config, r.metric): [] for r in obs_single.itertuples(index=False)}
    pair_samples: Dict[Tuple[str, str, str, str], List[float]] = {}
    for r in obs_pair.itertuples(index=False):
        pair_samples[(r.object_A, r.object_B, r.baseline_config, r.metric_family)] = []
    sample_rows = [] if cfg.save_bootstrap_metric_samples else None
    for ib in range(cfg.bootstrap_n):
        idx = rng.integers(0, ny, size=ny)
        boot_curves = _compute_all_curves_for_year_indices(regions, idx, cfg)
        for (obj, base), g in boot_curves.groupby(["object", "baseline_config"]):
            vals = _single_object_metrics_from_curves(g, cfg)
            for metric, value in vals.items():
                key = (obj, base, metric)
                if key in single_samples:
                    single_samples[key].append(value)
                    if sample_rows is not None:
                        sample_rows.append({"bootstrap_id": ib, "object": obj, "baseline_config": base, "metric": metric, "value": value})
        pair_boot = _observed_pairwise_summary(boot_curves, cfg)
        for r in pair_boot.itertuples(index=False):
            key = (r.object_A, r.object_B, r.baseline_config, r.metric_family)
            if key in pair_samples:
                pair_samples[key].append(r.delta_observed)
    single_summary_rows = []
    for r in obs_single.itertuples(index=False):
        s = _summarize_bootstrap(single_samples.get((r.object, r.baseline_config, r.metric), []), r.observed)
        s.update({"object": r.object, "baseline_config": r.baseline_config, "metric": r.metric})
        single_summary_rows.append(s)
    single_summary = pd.DataFrame(single_summary_rows)

    pair_summary_rows = []
    for r in obs_pair.itertuples(index=False):
        vals = pair_samples.get((r.object_A, r.object_B, r.baseline_config, r.metric_family), [])
        s = _summarize_bootstrap(vals, r.delta_observed)
        s.update({
            "object_A": r.object_A,
            "object_B": r.object_B,
            "baseline_config": r.baseline_config,
            "metric_family": r.metric_family,
            "delta_definition": r.delta_definition,
            "delta_observed": r.delta_observed,
        })
        pair_summary_rows.append(s)
    pair_summary = pd.DataFrame(pair_summary_rows)
    sample_df = pd.DataFrame(sample_rows) if sample_rows is not None else None
    return single_summary, pair_summary, sample_df


# -----------------------------------------------------------------------------
# Output / summary / plots
# -----------------------------------------------------------------------------

def _write_summary(paths: Dict[str, Path], cfg: Mirror2DConfig, input_audit: pd.DataFrame, clean_status: str, comparison: pd.DataFrame, single_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# W45 2D-field pre-post metric mirror (V7-z-2d-a)\n")
    lines.append("## Purpose\n")
    lines.append("This is a minimal 2D-field mirror of the profile-based W45 clean mainline. It computes 2D analogues of S_dist, R_diff/S_pattern, and growth metrics. It does not run 2D change-point detection and does not overwrite clean-mainline final claims.\n")
    lines.append("## Method boundary\n")
    lines.append("- Timing anchor / clean mainline: profile-based.\n")
    lines.append("- This audit basis: full object-specific 2D regional fields.\n")
    lines.append("- Outputs here are comparison and support diagnostics, not final timing claims.\n")
    lines.append("## Input status\n")
    lines.append(f"- Clean profile curves status: `{clean_status}`.\n")
    if not input_audit.empty:
        missing = input_audit[input_audit["status"] != "found"]
        if not missing.empty:
            lines.append("- Missing optional/required keys were recorded in the input audit table.\n")
    lines.append("## Main output files\n")
    for name in [
        "W45_2d_state_progress_curves_v7_z_2d_a.csv",
        "W45_2d_growth_speed_curves_v7_z_2d_a.csv",
        "W45_2d_single_object_metric_summary_v7_z_2d_a.csv",
        "W45_2d_pairwise_metric_summary_v7_z_2d_a.csv",
        "W45_profile_vs_2d_prepost_metric_comparison_v7_z_2d_a.csv",
    ]:
        lines.append(f"- `{name}`\n")
    lines.append("## Profile-vs-2D agreement overview\n")
    if not comparison.empty and "agreement_class" in comparison.columns:
        counts = comparison["agreement_class"].value_counts(dropna=False).to_dict()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}\n")
    lines.append("## Notes for interpretation\n")
    lines.append("- `S_pattern_2d` is a weighted 2D field correlation-based pre/post similarity progress, not a lat-profile correlation.\n")
    lines.append("- Disagreement between profile and 2D metrics should be treated as an audit signal, not as an automatic refutation of the profile clean mainline.\n")
    lines.append("- No 2D object-window or final claim is generated in this version.\n")
    (paths["out"] / "W45_2d_prepost_metric_mirror_summary_v7_z_2d_a.md").write_text("".join(lines), encoding="utf-8")


def _plot_basic_overview(curves2d: pd.DataFrame, paths: Dict[str, Path], cfg: Mirror2DConfig) -> None:
    if cfg.skip_figures or plt is None:
        return
    fig_dir = paths["out"] / "figures"
    _ensure_dir(fig_dir)
    for obj, gobj in curves2d.groupby("object"):
        for metric in ["S_dist_2d", "S_pattern_2d"]:
            fig = plt.figure(figsize=(9, 4))
            ax = fig.add_subplot(111)
            for base, g in gobj.groupby("baseline_config"):
                ax.plot(g["day"], g[metric], label=base)
            ax.axvspan(cfg.w45_start, cfg.w45_end, alpha=0.15)
            ax.set_title(f"{obj} {metric}")
            ax.set_xlabel("day")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{obj}_{metric}_v7_z_2d_a.png", dpi=160)
            plt.close(fig)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_W45_2d_prepost_metric_mirror_v7_z_2d_a(v7_root: Path | str) -> None:
    v7_root = Path(v7_root)
    cfg = Mirror2DConfig.from_env()
    out_dir = v7_root / "outputs" / OUTPUT_TAG
    log_dir = v7_root / "logs" / OUTPUT_TAG
    _ensure_dir(out_dir)
    _ensure_dir(log_dir)
    paths = {"out": out_dir, "log": log_dir}
    t0 = time.time()
    _log("[1/7] Load input fields")
    smoothed_path = Path(os.environ.get("V7Z_SMOOTHED_FIELDS", str(v7_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz")))
    if not smoothed_path.exists():
        # V7 root is usually D:/easm_project01/stage_partition/V7, so parents[1] may be stage_partition.
        alt = v7_root.parent.parent / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"
        if alt.exists():
            smoothed_path = alt
    fields, input_audit = _load_npz_fields(smoothed_path)
    lat = np.asarray(fields["lat"], dtype=float)
    lon = np.asarray(fields["lon"], dtype=float)
    years = np.asarray(fields["years"]) if "years" in fields else None
    input_audit.to_csv(out_dir / "W45_2d_input_key_audit_v7_z_2d_a.csv", index=False)

    _log("[2/7] Extract 2D object regions")
    regions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    object_rows = []
    for spec in OBJECT_SPECS:
        arr = _as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        region, lat_sel, lon_sel, weights = _extract_region_2d(arr, lat, lon, spec)
        regions[spec.object_name] = (region, weights)
        object_rows.append({
            **asdict(spec),
            "n_years": region.shape[0],
            "n_days": region.shape[1],
            "n_lat": region.shape[2],
            "n_lon": region.shape[3],
            "basis": "2d_field",
        })
    pd.DataFrame(object_rows).to_csv(out_dir / "W45_2d_object_registry_v7_z_2d_a.csv", index=False)
    pd.DataFrame([asdict(b) for b in BASELINES]).to_csv(out_dir / "W45_2d_baseline_config_table_v7_z_2d_a.csv", index=False)

    _log("[3/7] Compute observed 2D pre-post curves")
    curve_rows = []
    for spec in OBJECT_SPECS:
        region, weights = regions[spec.object_name]
        for base in BASELINES:
            df = _compute_2d_curves(region, weights, base, cfg)
            df.insert(0, "object", spec.object_name)
            curve_rows.append(df)
    curves2d = pd.concat(curve_rows, ignore_index=True)
    curves2d.to_csv(out_dir / "W45_2d_state_progress_curves_v7_z_2d_a.csv", index=False)
    curves2d[["object", "baseline_config", "day", "V_dist_2d", "V_pattern_2d", "basis"]].to_csv(out_dir / "W45_2d_growth_speed_curves_v7_z_2d_a.csv", index=False)

    _log("[4/7] Load clean profile curves and build profile-vs-2D comparison")
    profile_curves, _profile_growth, clean_dir, clean_status = _load_clean_profile_curves(v7_root)
    comparison = _build_profile_vs_2d_comparison(profile_curves, curves2d, cfg)
    comparison.to_csv(out_dir / "W45_profile_vs_2d_prepost_metric_comparison_v7_z_2d_a.csv", index=False)

    _log("[5/7] Run paired year-bootstrap for 2D metrics")
    single_summary, pair_summary, sample_df = _bootstrap_summaries(regions, curves2d, cfg)
    single_summary.to_csv(out_dir / "W45_2d_single_object_metric_summary_v7_z_2d_a.csv", index=False)
    pair_summary.to_csv(out_dir / "W45_2d_pairwise_metric_summary_v7_z_2d_a.csv", index=False)
    if sample_df is not None:
        sample_df.to_csv(out_dir / "W45_2d_bootstrap_metric_samples_v7_z_2d_a.csv", index=False)

    _log("[6/7] Write run metadata and summary")
    run_meta = {
        "version": cfg.version,
        "output_tag": cfg.output_tag,
        "smoothed_fields": str(smoothed_path),
        "clean_profile_result_dir": str(clean_dir),
        "clean_profile_status": clean_status,
        "basis": "2d_field",
        "does_2d_change_point_detection": False,
        "does_final_claim_rewrite": False,
        "n_bootstrap": cfg.bootstrap_n,
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json = {
        "n_curve_rows": int(len(curves2d)),
        "n_single_metric_rows": int(len(single_summary)),
        "n_pairwise_metric_rows": int(len(pair_summary)),
        "profile_vs_2d_agreement_counts": comparison["agreement_class"].value_counts(dropna=False).to_dict() if "agreement_class" in comparison.columns else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(paths, cfg, input_audit, clean_status, comparison, single_summary, pair_summary)
    _plot_basic_overview(curves2d, paths, cfg)

    _log("[7/7] Done")
    _log(f"Outputs: {out_dir}")



# -----------------------------------------------------------------------------
# Hotfix 01 performance overrides
# -----------------------------------------------------------------------------
# The first v7_z_2d_a implementation was scientifically usable but slow because
# the bootstrap path repeatedly built pandas DataFrames and called scalar
# weighted-distance/correlation functions inside nested Python loops.  The
# overrides below keep the same output schema and numerical definitions, but
# vectorize the per-day 2D distance/correlation calculations and summarize
# bootstrap metrics directly from numpy arrays.  This is intentionally a runtime
# performance hotfix only; it does not change the scientific definitions.


def _flatten_region_and_weights(region_by_year: np.ndarray, weights_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    region = np.asarray(region_by_year, dtype=float)
    if region.ndim != 4:
        raise ValueError(f"Expected [year,day,lat,lon] region, got {region.shape}")
    flat = region.reshape(region.shape[0], region.shape[1], -1)
    w = np.asarray(weights_2d, dtype=float).reshape(-1)
    return flat, w


def _weighted_distance_rows(X: np.ndarray, ref: np.ndarray, weights: np.ndarray, norm_eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(X) & np.isfinite(ref)[None, :] & np.isfinite(w)[None, :] & (w[None, :] > 0)
    diff = np.where(valid, X - ref[None, :], 0.0)
    denom = np.sum(np.where(valid, w[None, :], 0.0), axis=1)
    num = np.sum(np.where(valid, w[None, :] * diff * diff, 0.0), axis=1)
    out = np.full(X.shape[0], np.nan, dtype=float)
    m = denom > norm_eps
    out[m] = np.sqrt(num[m] / denom[m])
    return out


def _weighted_corr_rows(X: np.ndarray, ref: np.ndarray, weights: np.ndarray, norm_eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    ref = np.asarray(ref, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(X) & np.isfinite(ref)[None, :] & np.isfinite(w)[None, :] & (w[None, :] > 0)
    ww = np.where(valid, w[None, :], 0.0)
    denom = np.sum(ww, axis=1)
    out = np.full(X.shape[0], np.nan, dtype=float)
    m = denom > norm_eps
    if not np.any(m):
        return out
    X0 = np.where(valid, X, 0.0)
    R0 = np.where(valid, ref[None, :], 0.0)
    xmean = np.zeros(X.shape[0], dtype=float)
    rmean = np.zeros(X.shape[0], dtype=float)
    xmean[m] = np.sum(ww[m] * X0[m], axis=1) / denom[m]
    rmean[m] = np.sum(ww[m] * R0[m], axis=1) / denom[m]
    xc = np.where(valid, X - xmean[:, None], 0.0)
    rc = np.where(valid, ref[None, :] - rmean[:, None], 0.0)
    cov = np.sum(ww * xc * rc, axis=1) / np.where(denom > 0, denom, np.nan)
    vx = np.sum(ww * xc * xc, axis=1) / np.where(denom > 0, denom, np.nan)
    vr = np.sum(ww * rc * rc, axis=1) / np.where(denom > 0, denom, np.nan)
    ok = m & (vx > norm_eps) & (vr > norm_eps)
    out[ok] = cov[ok] / np.sqrt(vx[ok] * vr[ok])
    return out


def _curve_arrays_from_clim_flat(clim_flat: np.ndarray, weights_flat: np.ndarray, baseline: BaselineConfig, cfg: Mirror2DConfig) -> Dict[str, np.ndarray | float | bool]:
    clim = np.asarray(clim_flat, dtype=float)
    ndays = clim.shape[0]
    day_end = min(cfg.curve_end, ndays - 1)
    days = np.arange(cfg.curve_start, day_end + 1, dtype=int)
    pre_days = np.arange(baseline.pre_start, min(baseline.pre_end, ndays - 1) + 1, dtype=int)
    post_days = np.arange(baseline.post_start, min(baseline.post_end, ndays - 1) + 1, dtype=int)
    if len(pre_days) == 0 or len(post_days) == 0:
        raise ValueError(f"Invalid baseline days for ndays={ndays}: {baseline}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        F_pre = np.nanmean(clim[pre_days], axis=0)
        F_post = np.nanmean(clim[post_days], axis=0)
    X_pre = clim[pre_days]
    X_post = clim[post_days]
    pre_rdiff = _weighted_corr_rows(X_pre, F_post, weights_flat, cfg.norm_eps) - _weighted_corr_rows(X_pre, F_pre, weights_flat, cfg.norm_eps)
    post_rdiff = _weighted_corr_rows(X_post, F_post, weights_flat, cfg.norm_eps) - _weighted_corr_rows(X_post, F_pre, weights_flat, cfg.norm_eps)
    pre_mean = float(np.nanmean(pre_rdiff)) if np.any(np.isfinite(pre_rdiff)) else float("nan")
    post_mean = float(np.nanmean(post_rdiff)) if np.any(np.isfinite(post_rdiff)) else float("nan")
    dynamic = post_mean - pre_mean if np.isfinite(pre_mean) and np.isfinite(post_mean) else float("nan")
    branch_valid = bool(np.isfinite(dynamic) and abs(dynamic) > cfg.low_dynamic_range_eps)
    X = clim[days]
    D_pre = _weighted_distance_rows(X, F_pre, weights_flat, cfg.norm_eps)
    D_post = _weighted_distance_rows(X, F_post, weights_flat, cfg.norm_eps)
    denom = D_pre + D_post
    S_dist = np.full_like(D_pre, np.nan, dtype=float)
    ok = np.isfinite(denom) & (denom > cfg.norm_eps)
    S_dist[ok] = D_pre[ok] / denom[ok]
    R_pre = _weighted_corr_rows(X, F_pre, weights_flat, cfg.norm_eps)
    R_post = _weighted_corr_rows(X, F_post, weights_flat, cfg.norm_eps)
    R_diff = R_post - R_pre
    S_pattern = np.full_like(R_diff, np.nan, dtype=float)
    if branch_valid:
        S_pattern[np.isfinite(R_diff)] = (R_diff[np.isfinite(R_diff)] - pre_mean) / dynamic
    V_dist = np.full_like(S_dist, np.nan, dtype=float)
    V_pattern = np.full_like(S_pattern, np.nan, dtype=float)
    if len(days) > 1:
        md = np.isfinite(S_dist[1:]) & np.isfinite(S_dist[:-1])
        mp = np.isfinite(S_pattern[1:]) & np.isfinite(S_pattern[:-1])
        V_dist[1:][md] = S_dist[1:][md] - S_dist[:-1][md]
        V_pattern[1:][mp] = S_pattern[1:][mp] - S_pattern[:-1][mp]
    return {
        "days": days,
        "D_pre": D_pre,
        "D_post": D_post,
        "S_dist": S_dist,
        "R_pre": R_pre,
        "R_post": R_post,
        "R_diff": R_diff,
        "S_pattern": S_pattern,
        "V_dist": V_dist,
        "V_pattern": V_pattern,
        "dynamic": float(dynamic),
        "branch_valid": bool(branch_valid),
    }


def _metrics_from_curve_arrays(arr: Dict[str, np.ndarray | float | bool], cfg: Mirror2DConfig) -> Dict[str, float]:
    days = np.asarray(arr["days"], dtype=int)
    out: Dict[str, float] = {}
    for branch, state_key, growth_key in [
        ("dist", "S_dist", "V_dist"),
        ("pattern", "S_pattern", "V_pattern"),
    ]:
        s = np.asarray(arr[state_key], dtype=float)
        g = np.asarray(arr[growth_key], dtype=float)
        out[f"{branch}_mean_compare"] = _segment_mean(days, s, cfg.compare_start, cfg.compare_end)
        out[f"{branch}_mean_early"] = _segment_mean(days, s, cfg.early_start, cfg.early_end)
        out[f"{branch}_mean_core"] = _segment_mean(days, s, cfg.core_start, cfg.core_end)
        out[f"{branch}_mean_late"] = _segment_mean(days, s, cfg.late_start, cfg.late_end)
        out[f"{branch}_growth_center"] = _growth_center(days, g)
        out[f"{branch}_positive_growth_area"] = _growth_area(g)
        out[f"{branch}_early_growth_share"] = _segment_growth_share(days, g, cfg.early_start, cfg.early_end)
        out[f"{branch}_core_growth_share"] = _segment_growth_share(days, g, cfg.core_start, cfg.core_end)
        out[f"{branch}_late_growth_share"] = _segment_growth_share(days, g, cfg.late_start, cfg.late_end)
    return out


def _compute_2d_curves(region_by_year: np.ndarray, weights_2d: np.ndarray, baseline: BaselineConfig, cfg: Mirror2DConfig) -> pd.DataFrame:  # type: ignore[override]
    """Vectorized observed 2D pre-post curves from year/day region field."""
    flat, w = _flatten_region_and_weights(region_by_year, weights_2d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clim = np.nanmean(flat, axis=0)
    arr = _curve_arrays_from_clim_flat(clim, w, baseline, cfg)
    rows = []
    for i, d in enumerate(np.asarray(arr["days"], dtype=int)):
        rows.append({
            "baseline_config": baseline.name,
            "baseline_role": baseline.role,
            "day": int(d),
            "D_pre_2d": float(np.asarray(arr["D_pre"])[i]),
            "D_post_2d": float(np.asarray(arr["D_post"])[i]),
            "S_dist_2d": float(np.asarray(arr["S_dist"])[i]),
            "R_pre_2d": float(np.asarray(arr["R_pre"])[i]),
            "R_post_2d": float(np.asarray(arr["R_post"])[i]),
            "R_diff_2d": float(np.asarray(arr["R_diff"])[i]),
            "S_pattern_2d": float(np.asarray(arr["S_pattern"])[i]),
            "V_dist_2d": float(np.asarray(arr["V_dist"])[i]),
            "V_pattern_2d": float(np.asarray(arr["V_pattern"])[i]),
            "pattern_dynamic_range_2d": float(arr["dynamic"]),
            "pattern_branch_valid_2d": bool(arr["branch_valid"]),
            "basis": "2d_field",
        })
    return pd.DataFrame(rows)


def _pair_rows_from_metric_by_obj(metric_by_obj: Dict[str, Dict[str, float]], objects: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            ma = metric_by_obj[a]
            mb = metric_by_obj[b]
            for branch in ["dist", "pattern"]:
                for segment in ["compare", "early", "core", "late"]:
                    key = f"{branch}_mean_{segment}"
                    rows.append({
                        "object_A": a,
                        "object_B": b,
                        "metric_family": f"2d_{branch}_state_{segment}",
                        "delta_definition": f"{key}_A_minus_B; positive means A higher/ahead",
                        "delta_observed": ma.get(key, np.nan) - mb.get(key, np.nan),
                    })
                key = f"{branch}_growth_center"
                rows.append({
                    "object_A": a,
                    "object_B": b,
                    "metric_family": f"2d_{branch}_growth_center",
                    "delta_definition": f"{key}_B_minus_A; positive means A earlier growth",
                    "delta_observed": mb.get(key, np.nan) - ma.get(key, np.nan),
                })
                for seg in ["early", "core", "late"]:
                    key = f"{branch}_{seg}_growth_share"
                    rows.append({
                        "object_A": a,
                        "object_B": b,
                        "metric_family": f"2d_{branch}_{seg}_growth_share",
                        "delta_definition": f"{key}_A_minus_B; positive means A larger share",
                        "delta_observed": ma.get(key, np.nan) - mb.get(key, np.nan),
                    })
    return rows


def _bootstrap_summaries(regions: Dict[str, Tuple[np.ndarray, np.ndarray]], observed_curves: pd.DataFrame, cfg: Mirror2DConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:  # type: ignore[override]
    """Fast paired year-bootstrap for 2D metrics.

    This override avoids constructing bootstrap DataFrames.  It computes numpy
    metric dictionaries directly from vectorized curve arrays, then appends only
    scalar metric samples.  It also flattens each object region once outside the
    bootstrap loop.
    """
    rng = np.random.default_rng(cfg.random_seed)
    flat_regions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        obj: _flatten_region_and_weights(region, weights)
        for obj, (region, weights) in regions.items()
    }
    first_region = next(iter(flat_regions.values()))[0]
    ny = first_region.shape[0]
    objects = sorted(flat_regions.keys())

    observed_single_rows = []
    for (obj, base), g in observed_curves.groupby(["object", "baseline_config"]):
        vals = _single_object_metrics_from_curves(g, cfg)
        for metric, value in vals.items():
            observed_single_rows.append({"object": obj, "baseline_config": base, "metric": metric, "observed": value})
    obs_single = pd.DataFrame(observed_single_rows)
    obs_pair = _observed_pairwise_summary(observed_curves, cfg)

    single_samples: Dict[Tuple[str, str, str], List[float]] = {(r.object, r.baseline_config, r.metric): [] for r in obs_single.itertuples(index=False)}
    pair_samples: Dict[Tuple[str, str, str, str], List[float]] = {}
    for r in obs_pair.itertuples(index=False):
        pair_samples[(r.object_A, r.object_B, r.baseline_config, r.metric_family)] = []
    sample_rows = [] if cfg.save_bootstrap_metric_samples else None

    log_every_env = os.environ.get("V7Z_2D_A_BOOTSTRAP_LOG_EVERY")
    if log_every_env:
        log_every = max(1, int(log_every_env))
    else:
        log_every = max(1, min(100, cfg.bootstrap_n // 10 if cfg.bootstrap_n >= 10 else 1))

    for ib in range(cfg.bootstrap_n):
        idx = rng.integers(0, ny, size=ny)
        metric_by_base: Dict[str, Dict[str, Dict[str, float]]] = {b.name: {} for b in BASELINES}
        for obj in objects:
            region_flat, weights_flat = flat_regions[obj]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                clim = np.nanmean(region_flat[idx], axis=0)
            for base in BASELINES:
                arr = _curve_arrays_from_clim_flat(clim, weights_flat, base, cfg)
                vals = _metrics_from_curve_arrays(arr, cfg)
                metric_by_base[base.name][obj] = vals
                for metric, value in vals.items():
                    key = (obj, base.name, metric)
                    if key in single_samples:
                        single_samples[key].append(value)
                        if sample_rows is not None:
                            sample_rows.append({"bootstrap_id": ib, "object": obj, "baseline_config": base.name, "metric": metric, "value": value})
        for base in BASELINES:
            for r in _pair_rows_from_metric_by_obj(metric_by_base[base.name], objects):
                key = (r["object_A"], r["object_B"], base.name, r["metric_family"])
                if key in pair_samples:
                    pair_samples[key].append(float(r["delta_observed"]))
        if (ib + 1) % log_every == 0 or ib == cfg.bootstrap_n - 1:
            _log(f"    vectorized bootstrap {ib + 1}/{cfg.bootstrap_n}")

    single_summary_rows = []
    for r in obs_single.itertuples(index=False):
        s = _summarize_bootstrap(single_samples.get((r.object, r.baseline_config, r.metric), []), r.observed)
        s.update({"object": r.object, "baseline_config": r.baseline_config, "metric": r.metric})
        single_summary_rows.append(s)
    single_summary = pd.DataFrame(single_summary_rows)

    pair_summary_rows = []
    for r in obs_pair.itertuples(index=False):
        vals = pair_samples.get((r.object_A, r.object_B, r.baseline_config, r.metric_family), [])
        s = _summarize_bootstrap(vals, r.delta_observed)
        s.update({
            "object_A": r.object_A,
            "object_B": r.object_B,
            "baseline_config": r.baseline_config,
            "metric_family": r.metric_family,
            "delta_definition": r.delta_definition,
            "delta_observed": r.delta_observed,
        })
        pair_summary_rows.append(s)
    pair_summary = pd.DataFrame(pair_summary_rows)
    sample_df = pd.DataFrame(sample_rows) if sample_rows is not None else None
    return single_summary, pair_summary, sample_df


__all__ = ["run_W45_2d_prepost_metric_mirror_v7_z_2d_a"]
