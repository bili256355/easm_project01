from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, peak_prominences
except Exception as exc:  # pragma: no cover
    find_peaks = None
    peak_prominences = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

VERSION = "v7_z"
OUTPUT_TAG = "W45_multi_object_prepost_stat_validation_v7_z"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
SYSTEM_W45 = (40, 48)
OBJECT_DETECTION_RANGE = (0, 70)
CURVE_RANGE = (0, 74)
MAIN_COMPARISON_RANGE = (35, 53)
SEGMENTS = {
    "early": (30, 39),
    "core": (40, 45),
    "late": (46, 53),
}
OBJECT_ORDER = ("P", "V", "H", "Je", "Jw")
EPS = 1.0e-12


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    field_aliases: tuple[str, ...]
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]


OBJECT_SPECS: dict[str, ObjectSpec] = {
    "P": ObjectSpec(
        "P",
        (
            "precip_smoothed",
            "precipitation_smoothed",
            "p_smoothed",
            "P_smoothed",
            "pr_smoothed",
            "precip",
            "precipitation",
            "P",
        ),
        (105.0, 125.0),
        (15.0, 39.0),
    ),
    "V": ObjectSpec(
        "V",
        ("v850_smoothed", "V_smoothed", "v_smoothed", "v850", "V"),
        (105.0, 125.0),
        (10.0, 30.0),
    ),
    "H": ObjectSpec("H", ("z500_smoothed", "H_smoothed", "z500", "H"), (110.0, 140.0), (15.0, 35.0)),
    "Je": ObjectSpec("Je", ("u200_smoothed", "u200", "Je_smoothed", "Je"), (120.0, 150.0), (25.0, 45.0)),
    "Jw": ObjectSpec("Jw", ("u200_smoothed", "u200", "Jw_smoothed", "Jw"), (80.0, 110.0), (25.0, 45.0)),
}


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    pre: tuple[int, int]
    post: tuple[int, int]
    role: str
    notes: str


BASELINE_CONFIGS: tuple[BaselineConfig, ...] = (
    BaselineConfig("C0_full_stage", (0, 39), (49, 74), "main_full_stage", "W45 accepted window outside full pre/post stage"),
    BaselineConfig("C1_buffered_stage", (0, 34), (54, 69), "buffered_sensitivity", "Remove 5-day buffers around W45 and next-window front edge"),
    BaselineConfig("C2_immediate_pre", (25, 34), (54, 69), "immediate_pre_diagnostic", "Near pre-state diagnostic, not equal to full-stage baseline"),
)


@dataclass(frozen=True)
class ProfileConfig:
    lat_step_deg: float = 2.0


@dataclass(frozen=True)
class DetectorConfig:
    width: int = 20
    model: str = "l2"
    min_size: int = 2
    jump: int = 1
    selection_mode: str = "pen"
    pen: float = 4.0
    fixed_n_bkps: Optional[int] = None
    epsilon: Optional[float] = None
    local_peak_min_distance_days: int = 3
    nearest_peak_search_radius_days: int = 10


@dataclass(frozen=True)
class BandConfig:
    min_band_half_width_days: int = 2
    max_band_half_width_days: int = 10
    peak_floor_quantile: float = 0.35
    prominence_ratio_threshold: float = 0.50
    merge_gap_days: int = 1
    significant_peak_threshold: float = 0.95


@dataclass(frozen=True)
class BootstrapConfig:
    n_bootstrap: int = 1000
    random_seed: int = 42
    strict_match_max_abs_offset_days: int = 2
    match_max_abs_offset_days: int = 5
    near_match_max_abs_offset_days: int = 8
    progress: bool = True


@dataclass(frozen=True)
class V7ZConfig:
    project_root: str = r"D:\easm_project01"
    smoothed_fields_relpath: str = r"foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz"
    output_tag: str = OUTPUT_TAG
    objects: tuple[str, ...] = OBJECT_ORDER
    profile: ProfileConfig = ProfileConfig()
    detector: DetectorConfig = DetectorConfig()
    band: BandConfig = BandConfig()
    bootstrap: BootstrapConfig = BootstrapConfig()
    write_figures: bool = True


@dataclass
class Paths:
    v7_root: Path
    project_root: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path


@dataclass
class ObjectProfile:
    name: str
    field_key: str
    raw_cube: np.ndarray  # years x days x lat_feature
    lat_grid: np.ndarray
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]


# -----------------------------------------------------------------------------
# Basic IO/utilities
# -----------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict[str, Any], path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _safe_nanmean(a: np.ndarray, axis=None, return_valid_count: bool = False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(a, axis=axis)
    if return_valid_count:
        count = np.sum(np.isfinite(a), axis=axis)
        return mean, count
    return mean


def _resolve_paths(v7_root: Optional[Path], cfg: V7ZConfig) -> Paths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = Path(cfg.project_root)
    if not project_root.exists():
        project_root = v7_root.parents[1]
    output_dir = v7_root / "outputs" / cfg.output_tag
    log_dir = v7_root / "logs" / cfg.output_tag
    figure_dir = output_dir / "figures"
    for p in (output_dir, log_dir, figure_dir):
        _ensure_dir(p)
    return Paths(v7_root=v7_root, project_root=project_root, output_dir=output_dir, log_dir=log_dir, figure_dir=figure_dir)


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _resolve_field_key(files: dict[str, Any], spec: ObjectSpec) -> tuple[str | None, list[str]]:
    for alias in spec.field_aliases:
        if alias in files:
            return alias, []
    # relaxed lower-case search
    lower_map = {str(k).lower(): k for k in files.keys()}
    for alias in spec.field_aliases:
        if alias.lower() in lower_map:
            return str(lower_map[alias.lower()]), []
    return None, list(spec.field_aliases)


def _day_to_month_day(day: int) -> str:
    month_lengths = [(4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]
    d = int(day)
    for month, length in month_lengths:
        if d < length:
            return f"{month:02d}-{d + 1:02d}"
        d -= length
    return f"day{int(day)}"


def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(float(lower), float(upper)), max(float(lower), float(upper))
    return (arr >= lo) & (arr <= hi)


def _lat_weights(lat_grid: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(np.asarray(lat_grid, dtype=float)))
    w[~np.isfinite(w)] = np.nan
    return w


def _weighted_mean_1d(x: np.ndarray, w: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(w)
    if good.sum() == 0:
        return np.nan
    return float(np.nansum(x[good] * w[good]) / np.nansum(w[good]))


def _weighted_norm_1d(x: np.ndarray, w: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(w)
    if good.sum() == 0:
        return np.nan
    return float(np.sqrt(np.nansum(w[good] * x[good] ** 2) / np.nansum(w[good])))


def _weighted_distance(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return _weighted_norm_1d(diff, w)


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
    if good.sum() < 2:
        return np.nan
    wg = w[good]
    ag = a[good]
    bg = b[good]
    ma = np.nansum(wg * ag) / np.nansum(wg)
    mb = np.nansum(wg * bg) / np.nansum(wg)
    da = ag - ma
    db = bg - mb
    va = np.nansum(wg * da * da)
    vb = np.nansum(wg * db * db)
    if va <= EPS or vb <= EPS:
        return np.nan
    return float(np.nansum(wg * da * db) / np.sqrt(va * vb))


def _interp_profile_to_grid(profile: np.ndarray, src_lats: np.ndarray, dst_lats: np.ndarray) -> np.ndarray:
    out = np.full((profile.shape[0], profile.shape[1], dst_lats.size), np.nan, dtype=float)
    order = np.argsort(src_lats)
    src_sorted = np.asarray(src_lats)[order]
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            vals = profile[i, j, :][order]
            valid = np.isfinite(vals) & np.isfinite(src_sorted)
            if valid.sum() < 2:
                continue
            out[i, j, :] = np.interp(dst_lats, src_sorted[valid], vals[valid], left=np.nan, right=np.nan)
    return out


def _build_profile_from_field(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: ObjectSpec, lat_step_deg: float) -> ObjectProfile:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_mask = _mask_between(lat, *spec.lat_range)
    lon_mask = _mask_between(lon, *spec.lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points for {spec.name} in {spec.lat_range}")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points for {spec.name} in {spec.lon_range}")
    subset = np.asarray(field, dtype=float)[:, :, lat_mask, :][:, :, :, lon_mask]
    prof = _safe_nanmean(subset, axis=-1)
    src_lats = lat[lat_mask]
    lo, hi = min(spec.lat_range), max(spec.lat_range)
    dst_lats = np.arange(lo, hi + 1.0e-9, lat_step_deg)
    if dst_lats.size < 2:
        raise ValueError(f"Too few destination latitudes for {spec.name}: {dst_lats}")
    prof2 = _interp_profile_to_grid(prof, src_lats, dst_lats)
    return ObjectProfile(spec.name, "", prof2, dst_lats, spec.lon_range, spec.lat_range)


def _zscore_features(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd < EPS)] = 1.0
    return (x - mu) / sd


def _shape_normalize_cube(cube: np.ndarray, lat_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cube = np.asarray(cube, dtype=float)
    w = _lat_weights(lat_grid)
    sqrt_w = np.sqrt(w / np.nanmean(w))
    out = np.full_like(cube, np.nan, dtype=float)
    norms = np.full(cube.shape[:2], np.nan, dtype=float)
    valid = np.zeros(cube.shape[:2], dtype=bool)
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            row = cube[i, j, :]
            good = np.isfinite(row) & np.isfinite(w)
            if good.sum() < 2:
                continue
            mu = _weighted_mean_1d(row, w)
            centered = row - mu
            norm = _weighted_norm_1d(centered, w)
            norms[i, j] = norm
            if not np.isfinite(norm) or norm < EPS:
                continue
            out[i, j, :] = (centered / norm) * sqrt_w
            valid[i, j] = True
    return out, norms, valid


def _clim_matrix(profile: ObjectProfile, sampled_year_indices: Optional[np.ndarray] = None) -> np.ndarray:
    cube = profile.raw_cube if sampled_year_indices is None else profile.raw_cube[np.asarray(sampled_year_indices, dtype=int), :, :]
    return _safe_nanmean(cube, axis=0)


def _raw_state_matrix(profile: ObjectProfile, sampled_year_indices: Optional[np.ndarray] = None) -> np.ndarray:
    return _zscore_features(_clim_matrix(profile, sampled_year_indices))


def _shape_state_matrix(profile: ObjectProfile, sampled_year_indices: Optional[np.ndarray] = None) -> np.ndarray:
    cube = profile.raw_cube if sampled_year_indices is None else profile.raw_cube[np.asarray(sampled_year_indices, dtype=int), :, :]
    pat, _, _ = _shape_normalize_cube(cube, profile.lat_grid)
    return _safe_nanmean(pat, axis=0)


def _finite_day_subset(matrix: np.ndarray, day_range: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = map(int, day_range)
    days = np.arange(matrix.shape[0], dtype=int)
    mask = (days >= lo) & (days <= hi)
    sub_days = days[mask]
    sub = np.asarray(matrix, dtype=float)[sub_days, :]
    finite = np.all(np.isfinite(sub), axis=1)
    return sub[finite, :], sub_days[finite]


# -----------------------------------------------------------------------------
# Detector utilities
# -----------------------------------------------------------------------------


def _import_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except Exception as exc:  # pragma: no cover
        raise ImportError("V7-z requires the ruptures package in the user's environment.") from exc


def _map_profile_index(profile: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None or profile.empty:
        return profile
    arr = np.asarray(day_index, dtype=int)
    mapped = []
    for i in profile.index.to_numpy(dtype=int):
        idx = max(0, min(len(arr) - 1, int(i)))
        mapped.append(int(arr[idx]))
    out = profile.copy()
    out.index = np.asarray(mapped, dtype=int)
    return out


def _map_breakpoints_to_days(points_local: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None:
        return points_local.astype(int)
    arr = np.asarray(day_index, dtype=int)
    vals = [int(arr[max(0, min(len(arr) - 1, int(p) - 1))]) for p in points_local.astype(int).tolist()]
    return pd.Series(vals, name="changepoint", dtype=int)


def _run_ruptures_window(state_matrix: np.ndarray, cfg: DetectorConfig, day_index: np.ndarray | None = None) -> dict[str, Any]:
    rpt = _import_ruptures()
    signal = np.asarray(state_matrix, dtype=float)
    if signal.shape[0] < max(2 * int(cfg.width), 3):
        return {"profile": pd.Series(dtype=float, name="profile"), "points": pd.Series(dtype=int), "status": "skipped_insufficient_valid_days"}
    algo = rpt.Window(width=int(cfg.width), model=cfg.model, min_size=int(cfg.min_size), jump=int(cfg.jump)).fit(signal)
    if cfg.selection_mode == "pen":
        bkps = algo.predict(pen=float(cfg.pen))
    elif cfg.selection_mode == "fixed_n_bkps":
        bkps = algo.predict(n_bkps=int(cfg.fixed_n_bkps or 1))
    elif cfg.selection_mode == "epsilon":
        bkps = algo.predict(epsilon=float(cfg.epsilon or 1.0))
    else:
        raise ValueError(f"Unsupported selection_mode={cfg.selection_mode}")
    points_local = pd.Series([int(x) for x in bkps[:-1]], name="changepoint", dtype=int)
    score = getattr(algo, "score", None)
    if score is None:
        profile_raw = pd.Series(dtype=float, name="profile")
    else:
        arr = np.asarray(score, dtype=float).ravel()
        width_half = int(algo.width // 2)
        idx = np.arange(width_half, width_half + len(arr), dtype=int)
        profile_raw = pd.Series(arr, index=idx, name="profile")
    return {
        "profile": _map_profile_index(profile_raw, day_index),
        "profile_local": profile_raw,
        "points": _map_breakpoints_to_days(points_local, day_index),
        "status": "success",
    }


def _extract_local_peaks(profile: pd.Series, min_distance_days: int) -> pd.DataFrame:
    cols = ["peak_id", "peak_day", "peak_score", "peak_prominence", "peak_rank"]
    if _SCIPY_IMPORT_ERROR is not None:
        raise ImportError("V7-z requires scipy.signal for peak extraction.") from _SCIPY_IMPORT_ERROR
    if profile is None or profile.empty:
        return pd.DataFrame(columns=cols)
    s = profile.sort_index().astype(float).fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=max(1, int(min_distance_days)))
    if peaks.size == 0:
        return pd.DataFrame(columns=cols)
    prominences, _, _ = peak_prominences(values, peaks)
    rows = []
    for p, prom in zip(peaks, prominences):
        rows.append({
            "peak_id": "LP000",  # replaced after ranking
            "peak_day": int(s.index[int(p)]),
            "peak_score": float(values[int(p)]),
            "peak_prominence": float(prom),
        })
    df = pd.DataFrame(rows).sort_values(["peak_score", "peak_prominence", "peak_day"], ascending=[False, False, True]).reset_index(drop=True)
    df["peak_rank"] = np.arange(1, len(df) + 1, dtype=int)
    df["peak_id"] = [f"CP{i:03d}" for i in range(1, len(df) + 1)]
    return df[cols]


def _build_band(profile: pd.Series, peak_day: int, cfg: BandConfig) -> dict[str, Any]:
    if profile is None or profile.empty or int(peak_day) not in profile.index:
        return {"band_start_day": int(peak_day), "band_end_day": int(peak_day), "support_floor": np.nan, "left_stop_reason": "missing_profile", "right_stop_reason": "missing_profile"}
    s = profile.sort_index().astype(float)
    peak_score = float(s.loc[int(peak_day)])
    finite = s[np.isfinite(s)]
    if finite.empty:
        floor = np.nan
    else:
        floor = float(max(np.nanquantile(finite.to_numpy(), cfg.peak_floor_quantile), peak_score * cfg.prominence_ratio_threshold))
    min_lo = int(peak_day) - int(cfg.min_band_half_width_days)
    min_hi = int(peak_day) + int(cfg.min_band_half_width_days)
    lo = int(peak_day)
    while lo - 1 in s.index and (int(peak_day) - (lo - 1)) <= int(cfg.max_band_half_width_days):
        if lo - 1 <= min_lo or float(s.loc[lo - 1]) >= floor:
            lo -= 1
        else:
            break
    hi = int(peak_day)
    while hi + 1 in s.index and ((hi + 1) - int(peak_day)) <= int(cfg.max_band_half_width_days):
        if hi + 1 >= min_hi or float(s.loc[hi + 1]) >= floor:
            hi += 1
        else:
            break
    return {"band_start_day": int(lo), "band_end_day": int(hi), "support_floor": floor, "left_stop_reason": "floor_or_width", "right_stop_reason": "floor_or_width"}


def _run_detector_for_matrix(matrix: np.ndarray, cfg: V7ZConfig) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    sub, days = _finite_day_subset(matrix, OBJECT_DETECTION_RANGE)
    result = _run_ruptures_window(sub, cfg.detector, day_index=days)
    profile = result["profile"]
    peaks = _extract_local_peaks(profile, cfg.detector.local_peak_min_distance_days)
    band_rows = []
    for _, row in peaks.iterrows():
        band = _build_band(profile, int(row["peak_day"]), cfg.band)
        band_rows.append({**row.to_dict(), **band})
    bands = pd.DataFrame(band_rows)
    return profile, peaks, bands


def _match_peak(profile: pd.Series, observed_day: int, radius: int, min_distance: int) -> dict[str, Any]:
    peaks = _extract_local_peaks(profile, min_distance)
    if peaks.empty:
        return {"matched": False, "matched_day": np.nan, "offset": np.nan, "match_status": "no_peak"}
    sub = peaks.copy()
    sub["abs_offset"] = (sub["peak_day"].astype(int) - int(observed_day)).abs()
    sub = sub[sub["abs_offset"] <= int(radius)]
    if sub.empty:
        return {"matched": False, "matched_day": np.nan, "offset": np.nan, "match_status": "no_match"}
    sub = sub.sort_values(["abs_offset", "peak_score"], ascending=[True, False]).reset_index(drop=True)
    best = sub.iloc[0]
    return {"matched": True, "matched_day": int(best["peak_day"]), "offset": int(best["peak_day"]) - int(observed_day), "match_status": "matched"}


# -----------------------------------------------------------------------------
# State/growth curves and metrics
# -----------------------------------------------------------------------------


def _days_from_range(day_range: tuple[int, int], n_days: int) -> np.ndarray:
    lo, hi = map(int, day_range)
    lo = max(0, lo)
    hi = min(n_days - 1, hi)
    return np.arange(lo, hi + 1, dtype=int)


def _compute_state_curves_for_object(profile: ObjectProfile, sampled_year_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
    mat = _clim_matrix(profile, sampled_year_indices)
    n_days = mat.shape[0]
    w = _lat_weights(profile.lat_grid)
    rows = []
    for bc in BASELINE_CONFIGS:
        pre_days = _days_from_range(bc.pre, n_days)
        post_days = _days_from_range(bc.post, n_days)
        pre = _safe_nanmean(mat[pre_days, :], axis=0)
        post = _safe_nanmean(mat[post_days, :], axis=0)
        rdiff_all = []
        for d in range(n_days):
            rpre = _weighted_corr(mat[d, :], pre, w)
            rpost = _weighted_corr(mat[d, :], post, w)
            rdiff_all.append(rpost - rpre if np.isfinite(rpre) and np.isfinite(rpost) else np.nan)
        rdiff_all = np.asarray(rdiff_all, dtype=float)
        r0 = float(np.nanmean(rdiff_all[pre_days])) if pre_days.size else np.nan
        r1 = float(np.nanmean(rdiff_all[post_days])) if post_days.size else np.nan
        dyn = r1 - r0
        for d in _days_from_range(CURVE_RANGE, n_days):
            dpre = _weighted_distance(mat[d, :], pre, w)
            dpost = _weighted_distance(mat[d, :], post, w)
            sdist = dpre / (dpre + dpost) if np.isfinite(dpre) and np.isfinite(dpost) and (dpre + dpost) > EPS else np.nan
            spattern = (rdiff_all[d] - r0) / dyn if np.isfinite(rdiff_all[d]) and np.isfinite(dyn) and abs(dyn) > EPS else np.nan
            rows.append({
                "object": profile.name,
                "baseline_config": bc.name,
                "day": int(d),
                "S_dist": float(sdist) if np.isfinite(sdist) else np.nan,
                "S_pattern": float(spattern) if np.isfinite(spattern) else np.nan,
                "R_diff": float(rdiff_all[d]) if np.isfinite(rdiff_all[d]) else np.nan,
                "S_pattern_dynamic_range": float(dyn) if np.isfinite(dyn) else np.nan,
                "branch_validity_flag": "valid" if np.isfinite(dyn) and abs(dyn) > 1.0e-8 else "low_dynamic_range",
            })
    return pd.DataFrame(rows)


def _compute_growth_curves(state_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (obj, bc), sub in state_df.groupby(["object", "baseline_config"], sort=False):
        sub = sub.sort_values("day")
        for branch, scol in (("distance", "S_dist"), ("pattern", "S_pattern")):
            vals = sub[scol].to_numpy(dtype=float)
            days = sub["day"].to_numpy(dtype=int)
            v = np.full_like(vals, np.nan, dtype=float)
            v[1:] = vals[1:] - vals[:-1]
            # centered rolling mean 3
            vs = pd.Series(v).rolling(3, center=True, min_periods=1).mean().to_numpy(dtype=float)
            for d, s, vv, vss in zip(days, vals, v, vs):
                rows.append({
                    "object": obj,
                    "baseline_config": bc,
                    "branch": branch,
                    "day": int(d),
                    "S": float(s) if np.isfinite(s) else np.nan,
                    "V": float(vv) if np.isfinite(vv) else np.nan,
                    "V_smooth3": float(vss) if np.isfinite(vss) else np.nan,
                })
    return pd.DataFrame(rows)


def _range_mask(days: np.ndarray, day_range: tuple[int, int]) -> np.ndarray:
    return (days >= int(day_range[0])) & (days <= int(day_range[1]))


def _growth_metrics_for_series(days: np.ndarray, v: np.ndarray) -> dict[str, float]:
    mask = _range_mask(days, MAIN_COMPARISON_RANGE)
    dd = days[mask]
    vv = v[mask]
    pos = np.where(np.isfinite(vv) & (vv > 0), vv, 0.0)
    area = float(np.nansum(pos))
    center = float(np.nansum(dd * pos) / area) if area > EPS else np.nan
    out: dict[str, float] = {"positive_growth_area": area, "growth_center": center}
    for seg, rng in SEGMENTS.items():
        smask = _range_mask(days, rng)
        sarea = float(np.nansum(np.where(np.isfinite(v[smask]) & (v[smask] > 0), v[smask], 0.0)))
        out[f"{seg}_growth_share"] = sarea / area if area > EPS else np.nan
    out["negative_growth_area"] = float(np.nansum(np.where(np.isfinite(vv) & (vv < 0), -vv, 0.0)))
    return out


def _state_metric(state_df: pd.DataFrame, obj: str, bc: str, branch: str) -> dict[str, float]:
    col = "S_dist" if branch == "distance" else "S_pattern"
    sub = state_df[(state_df["object"] == obj) & (state_df["baseline_config"] == bc)].sort_values("day")
    days = sub["day"].to_numpy(dtype=int)
    vals = sub[col].to_numpy(dtype=float)
    mask = _range_mask(days, MAIN_COMPARISON_RANGE)
    out = {"mean_state": float(np.nanmean(vals[mask])) if np.any(mask) else np.nan}
    for seg, rng in SEGMENTS.items():
        sm = _range_mask(days, rng)
        out[f"{seg}_mean_state"] = float(np.nanmean(vals[sm])) if np.any(sm) else np.nan
    return out


def _growth_metric(growth_df: pd.DataFrame, obj: str, bc: str, branch: str) -> dict[str, float]:
    sub = growth_df[(growth_df["object"] == obj) & (growth_df["baseline_config"] == bc) & (growth_df["branch"] == branch)].sort_values("day")
    return _growth_metrics_for_series(sub["day"].to_numpy(dtype=int), sub["V_smooth3"].to_numpy(dtype=float))


def _object_state_growth_metrics(state_df: pd.DataFrame, growth_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for obj in OBJECT_ORDER:
        for bc in [b.name for b in BASELINE_CONFIGS]:
            for branch in ("distance", "pattern"):
                sm = _state_metric(state_df, obj, bc, branch)
                for k, v in sm.items():
                    rows.append({"object": obj, "baseline_config": bc, "metric_family": f"state_{branch}", "metric_name": k, "value": v})
                gm = _growth_metric(growth_df, obj, bc, branch)
                for k, v in gm.items():
                    rows.append({"object": obj, "baseline_config": bc, "metric_family": f"growth_{branch}", "metric_name": k, "value": v})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Classification helpers
# -----------------------------------------------------------------------------


def _quantiles(vals: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": np.nan, "q05": np.nan, "q95": np.nan, "q025": np.nan, "q975": np.nan, "n_valid": 0}
    return {
        "median": float(np.nanmedian(arr)),
        "q05": float(np.nanquantile(arr, 0.05)),
        "q95": float(np.nanquantile(arr, 0.95)),
        "q025": float(np.nanquantile(arr, 0.025)),
        "q975": float(np.nanquantile(arr, 0.975)),
        "n_valid": int(arr.size),
    }


def _support_class(frac: float) -> str:
    if not np.isfinite(frac):
        return "unstable_window"
    if frac >= 0.95:
        return "accepted_window"
    if frac >= 0.80:
        return "candidate_window"
    if frac >= 0.50:
        return "weak_window"
    return "unstable_window"


def _direction_decision(delta_vals: list[float], positive_label: str, negative_label: str, tendency_threshold: float = 0.80) -> dict[str, Any]:
    q = _quantiles(delta_vals)
    arr = np.asarray(delta_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    p_pos = float(np.mean(arr > 0)) if arr.size else np.nan
    p_neg = float(np.mean(arr < 0)) if arr.size else np.nan
    if arr.size == 0:
        decision = "invalid"
    elif np.isfinite(q["q025"]) and q["q025"] > 0:
        decision = f"{positive_label}_supported"
    elif np.isfinite(q["q975"]) and q["q975"] < 0:
        decision = f"{negative_label}_supported"
    elif np.isfinite(p_pos) and p_pos >= tendency_threshold:
        decision = f"{positive_label}_tendency"
    elif np.isfinite(p_neg) and p_neg >= tendency_threshold:
        decision = f"{negative_label}_tendency"
    else:
        decision = "unresolved"
    return {**q, "P_positive": p_pos, "P_negative": p_neg, "decision": decision}


def _relation_to_w45(start: float, end: float, peak_day: float) -> str:
    if not np.isfinite(peak_day):
        return "unknown"
    s0, s1 = SYSTEM_W45
    if end < s0:
        return "pre_W45"
    if start > s1:
        return "post_W45"
    if start >= s0 and end <= s1:
        return "within_W45"
    return "overlap_W45"


def _window_overlap_days(a0: float, a1: float, b0: float, b1: float) -> int:
    if not all(np.isfinite([a0, a1, b0, b1])):
        return 0
    lo = max(int(round(a0)), int(round(b0)))
    hi = min(int(round(a1)), int(round(b1)))
    return max(0, hi - lo + 1)


# -----------------------------------------------------------------------------
# Observed and bootstrap runners
# -----------------------------------------------------------------------------


def _build_profiles(files: dict[str, Any], cfg: V7ZConfig, outdir: Path) -> tuple[dict[str, ObjectProfile], pd.DataFrame]:
    lat = np.asarray(files.get("lat"), dtype=float)
    lon = np.asarray(files.get("lon"), dtype=float)
    rows = []
    profiles: dict[str, ObjectProfile] = {}
    for obj in cfg.objects:
        spec = OBJECT_SPECS[obj]
        key, aliases = _resolve_field_key(files, spec)
        rows.append({"object": obj, "resolved_key": key or "", "candidate_aliases": ";".join(aliases), "status": "found" if key else "missing"})
        if key is None:
            continue
        field = np.asarray(files[key], dtype=float)
        prof = _build_profile_from_field(field, lat, lon, spec, cfg.profile.lat_step_deg)
        prof.field_key = key
        profiles[obj] = prof
    audit = pd.DataFrame(rows)
    _write_csv(audit, outdir / "W45_multi_object_input_key_audit_v7_z.csv")
    missing = audit[audit["status"] != "found"]
    if not missing.empty:
        raise KeyError(f"V7-z missing required fields for objects: {missing['object'].tolist()}. See input_key_audit output.")
    return profiles, audit


def _detector_observed(profiles: dict[str, ObjectProfile], cfg: V7ZConfig) -> dict[str, Any]:
    score_rows, peak_rows, band_rows, window_rows = [], [], [], []
    observed_candidates: dict[tuple[str, str, str], int] = {}
    for obj, prof in profiles.items():
        for dtype, mat in (("raw_profile", _raw_state_matrix(prof)), ("shape_pattern", _shape_state_matrix(prof))):
            profile, peaks, bands = _run_detector_for_matrix(mat, cfg)
            for d, v in profile.items():
                score_rows.append({"object": obj, "detector_type": dtype, "day": int(d), "detector_score": float(v)})
            if bands.empty:
                continue
            for _, r in bands.iterrows():
                cid = str(r["peak_id"])
                peak_day = int(r["peak_day"])
                observed_candidates[(obj, dtype, cid)] = peak_day
                row = {
                    "object": obj,
                    "detector_type": dtype,
                    "candidate_id": cid,
                    "peak_day": peak_day,
                    "peak_score": float(r["peak_score"]),
                    "peak_prominence": float(r["peak_prominence"]),
                    "peak_rank": int(r["peak_rank"]),
                    "window_start": int(r["band_start_day"]),
                    "window_end": int(r["band_end_day"]),
                    "relation_to_W45": _relation_to_w45(r["band_start_day"], r["band_end_day"], peak_day),
                    "overlap_days_with_W45": _window_overlap_days(r["band_start_day"], r["band_end_day"], *SYSTEM_W45),
                }
                peak_rows.append(row)
                band_rows.append(row)
                window_rows.append({**row, "window_id": f"{obj}_{dtype}_{cid}", "center_day": 0.5 * (int(r["band_start_day"]) + int(r["band_end_day"]))})
    return {
        "score_df": pd.DataFrame(score_rows),
        "peaks_df": pd.DataFrame(peak_rows),
        "bands_df": pd.DataFrame(band_rows),
        "windows_df": pd.DataFrame(window_rows),
        "observed_candidates": observed_candidates,
    }


def _bootstrap_iteration(
    b: int,
    sampled_idx: np.ndarray,
    profiles: dict[str, ObjectProfile],
    cfg: V7ZConfig,
    observed_candidates: dict[tuple[str, str, str], int],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    # Detector return days.
    match_rows: list[dict[str, Any]] = []
    for obj, prof in profiles.items():
        matrices = {
            "raw_profile": _raw_state_matrix(prof, sampled_idx),
            "shape_pattern": _shape_state_matrix(prof, sampled_idx),
        }
        for dtype, mat in matrices.items():
            sub, days = _finite_day_subset(mat, OBJECT_DETECTION_RANGE)
            det = _run_ruptures_window(sub, cfg.detector, day_index=days)
            profile = det["profile"]
            for (cobj, cdtype, cid), obs_day in observed_candidates.items():
                if cobj != obj or cdtype != dtype:
                    continue
                m = _match_peak(profile, obs_day, cfg.bootstrap.match_max_abs_offset_days, cfg.detector.local_peak_min_distance_days)
                strict = bool(m["matched"] and abs(float(m["offset"])) <= cfg.bootstrap.strict_match_max_abs_offset_days)
                near = bool(m["matched"] and abs(float(m["offset"])) <= cfg.bootstrap.near_match_max_abs_offset_days)
                match_rows.append({
                    "bootstrap_id": int(b),
                    "object": obj,
                    "detector_type": dtype,
                    "candidate_id": cid,
                    "observed_peak_day": int(obs_day),
                    "matched_return_day": m["matched_day"],
                    "match_offset": m["offset"],
                    "match_status": m["match_status"],
                    "matched": bool(m["matched"]),
                    "strict_matched": strict,
                    "near_matched": near,
                })
    # Curves for this bootstrap.
    state_parts = []
    for obj, prof in profiles.items():
        state_parts.append(_compute_state_curves_for_object(prof, sampled_idx))
    state_df = pd.concat(state_parts, ignore_index=True)
    growth_df = _compute_growth_curves(state_df)
    obj_metrics = _object_state_growth_metrics(state_df, growth_df)
    return match_rows, state_df, growth_df


def _summarize_window_bootstrap(match_df: pd.DataFrame, observed_windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if match_df.empty:
        return pd.DataFrame()
    for (obj, dtype, cid), sub in match_df.groupby(["object", "detector_type", "candidate_id"], sort=False):
        n = len(sub)
        matched = sub[sub["matched"] == True]
        days = matched["matched_return_day"].astype(float).to_numpy() if not matched.empty else np.array([], dtype=float)
        q = _quantiles(days)
        frac = len(matched) / n if n else np.nan
        strict = float(sub["strict_matched"].mean()) if n else np.nan
        near = float(sub["near_matched"].mean()) if n else np.nan
        obs = observed_windows[(observed_windows["object"] == obj) & (observed_windows["detector_type"] == dtype) & (observed_windows["candidate_id"] == cid)]
        obs_day = float(obs.iloc[0]["peak_day"]) if not obs.empty else np.nan
        rows.append({
            "object": obj,
            "detector_type": dtype,
            "candidate_id": cid,
            "observed_peak_day": obs_day,
            "bootstrap_match_fraction": frac,
            "strict_match_fraction": strict,
            "near_match_fraction": near,
            "return_day_median": q["median"],
            "return_day_q05": q["q05"],
            "return_day_q95": q["q95"],
            "return_day_q025": q["q025"],
            "return_day_q975": q["q975"],
            "n_valid_matches": q["n_valid"],
            "support_class": _support_class(frac),
        })
    return pd.DataFrame(rows)


def _observed_state_growth(profiles: dict[str, ObjectProfile]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_df = pd.concat([_compute_state_curves_for_object(p) for p in profiles.values()], ignore_index=True)
    growth_df = _compute_growth_curves(state_df)
    obj_metrics = _object_state_growth_metrics(state_df, growth_df)
    return state_df, growth_df, obj_metrics


def _summarize_single_metrics(observed_obj_metrics: pd.DataFrame, bootstrap_obj_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["object", "baseline_config", "metric_family", "metric_name"]
    for key, obs_sub in observed_obj_metrics.groupby(keys, sort=False):
        obj, bc, fam, name = key
        obs_val = float(obs_sub["value"].iloc[0])
        boot = bootstrap_obj_metrics
        for k, v in zip(keys, key):
            boot = boot[boot[k] == v]
        vals = boot["value"].astype(float).to_numpy() if not boot.empty else np.array([], dtype=float)
        q = _quantiles(vals)
        p_pos = float(np.mean(vals > 0)) if vals.size else np.nan
        rows.append({
            "object": obj,
            "baseline_config": bc,
            "metric_family": fam,
            "metric_name": name,
            "observed": obs_val,
            "median": q["median"],
            "q05": q["q05"],
            "q95": q["q95"],
            "q025": q["q025"],
            "q975": q["q975"],
            "P_positive": p_pos,
            "n_valid": q["n_valid"],
        })
    return pd.DataFrame(rows)


def _pairwise_metrics_from_obj_metrics(obj_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(OBJECT_ORDER, 2):
        for bc in [x.name for x in BASELINE_CONFIGS]:
            # state advantage: A - B. growth delta: B center - A center.
            for branch in ("distance", "pattern"):
                a_state = obj_metrics[(obj_metrics["object"] == a) & (obj_metrics["baseline_config"] == bc) & (obj_metrics["metric_family"] == f"state_{branch}") & (obj_metrics["metric_name"] == "mean_state")]
                b_state = obj_metrics[(obj_metrics["object"] == b) & (obj_metrics["baseline_config"] == bc) & (obj_metrics["metric_family"] == f"state_{branch}") & (obj_metrics["metric_name"] == "mean_state")]
                if not a_state.empty and not b_state.empty:
                    delta = float(a_state["value"].iloc[0] - b_state["value"].iloc[0])
                    rows.append({"object_A": a, "object_B": b, "baseline_config": bc, "metric_family": f"state_{branch}", "metric_name": "mean_state_advantage_A_minus_B", "delta_definition": "A_minus_B_positive_means_A_higher_state", "delta": delta})
                a_g = obj_metrics[(obj_metrics["object"] == a) & (obj_metrics["baseline_config"] == bc) & (obj_metrics["metric_family"] == f"growth_{branch}") & (obj_metrics["metric_name"] == "growth_center")]
                b_g = obj_metrics[(obj_metrics["object"] == b) & (obj_metrics["baseline_config"] == bc) & (obj_metrics["metric_family"] == f"growth_{branch}") & (obj_metrics["metric_name"] == "growth_center")]
                if not a_g.empty and not b_g.empty:
                    delta = float(b_g["value"].iloc[0] - a_g["value"].iloc[0])
                    rows.append({"object_A": a, "object_B": b, "baseline_config": bc, "metric_family": f"growth_{branch}", "metric_name": "growth_center_delta_B_minus_A", "delta_definition": "B_minus_A_positive_means_A_earlier_growth", "delta": delta})
    return pd.DataFrame(rows)


def _pairwise_peak_delta(match_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dtype in ("raw_profile", "shape_pattern"):
        for a, b in combinations(OBJECT_ORDER, 2):
            a_sub = match_df[(match_df["object"] == a) & (match_df["detector_type"] == dtype) & (match_df["candidate_id"] == "CP001")][["bootstrap_id", "matched_return_day", "matched"]].rename(columns={"matched_return_day": "a_day", "matched": "a_matched"})
            b_sub = match_df[(match_df["object"] == b) & (match_df["detector_type"] == dtype) & (match_df["candidate_id"] == "CP001")][["bootstrap_id", "matched_return_day", "matched"]].rename(columns={"matched_return_day": "b_day", "matched": "b_matched"})
            if a_sub.empty or b_sub.empty:
                continue
            m = a_sub.merge(b_sub, on="bootstrap_id", how="inner")
            m = m[(m["a_matched"] == True) & (m["b_matched"] == True)]
            if m.empty:
                continue
            vals = m["b_day"].astype(float).to_numpy() - m["a_day"].astype(float).to_numpy()
            q = _direction_decision(vals, "A_earlier", "B_earlier")
            rows.append({
                "object_A": a,
                "object_B": b,
                "detector_type": dtype,
                "metric_family": f"{dtype}_peak_timing",
                "metric_name": "CP001_peak_delta_B_minus_A",
                "delta_definition": "B_peak_day_minus_A_peak_day_positive_means_A_earlier",
                "delta_median": q["median"],
                "delta_q05": q["q05"],
                "delta_q95": q["q95"],
                "delta_q025": q["q025"],
                "delta_q975": q["q975"],
                "P_A_earlier_or_higher": q["P_positive"],
                "P_B_earlier_or_higher": q["P_negative"],
                "decision": q["decision"],
                "n_valid": q["n_valid"],
            })
    return pd.DataFrame(rows)


def _summarize_pairwise(observed_pair: pd.DataFrame, bootstrap_pairs: pd.DataFrame, peak_delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not observed_pair.empty and not bootstrap_pairs.empty:
        keys = ["object_A", "object_B", "baseline_config", "metric_family", "metric_name", "delta_definition"]
        for key, obs_sub in observed_pair.groupby(keys, sort=False):
            obs = float(obs_sub["delta"].iloc[0])
            boot = bootstrap_pairs
            for k, v in zip(keys, key):
                boot = boot[boot[k] == v]
            vals = boot["delta"].astype(float).to_numpy() if not boot.empty else np.array([], dtype=float)
            # Positive means A higher/earlier for our deltas.
            q = _direction_decision(vals, "A", "B")
            rows.append({
                "object_A": key[0],
                "object_B": key[1],
                "baseline_config": key[2],
                "metric_family": key[3],
                "metric_name": key[4],
                "delta_definition": key[5],
                "delta_observed": obs,
                "delta_median": q["median"],
                "delta_q05": q["q05"],
                "delta_q95": q["q95"],
                "delta_q025": q["q025"],
                "delta_q975": q["q975"],
                "P_A_earlier_or_higher": q["P_positive"],
                "P_B_earlier_or_higher": q["P_negative"],
                "decision": q["decision"],
                "n_valid": q["n_valid"],
            })
    if peak_delta_df is not None and not peak_delta_df.empty:
        rows.extend(peak_delta_df.to_dict("records"))
    return pd.DataFrame(rows)


def _baseline_sensitivity(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pairwise_df.empty:
        return pd.DataFrame()
    sub = pairwise_df[pairwise_df["baseline_config"].notna()] if "baseline_config" in pairwise_df.columns else pd.DataFrame()
    keys = ["object_A", "object_B", "metric_family", "metric_name"]
    if sub.empty:
        return pd.DataFrame()
    for key, g in sub.groupby(keys, sort=False):
        dec = {str(r["baseline_config"]): str(r["decision"]) for _, r in g.iterrows()}
        signs = []
        for d in dec.values():
            if d.startswith("A_") or d.startswith("A_supported") or d.startswith("A_tendency"):
                signs.append("A")
            elif d.startswith("B_") or d.startswith("B_supported") or d.startswith("B_tendency"):
                signs.append("B")
            else:
                signs.append("U")
        non_u = [s for s in signs if s != "U"]
        if len(non_u) == 0:
            cls = "baseline_unresolved"
        elif len(set(non_u)) > 1:
            cls = "baseline_sensitive"
        elif dec.get("C0_full_stage", "").startswith(non_u[0]) and dec.get("C1_buffered_stage", "").startswith(non_u[0]):
            cls = "stable_or_stable_tendency"
        elif dec.get("C2_immediate_pre", "").startswith(non_u[0]) and not (dec.get("C0_full_stage", "").startswith(non_u[0]) and dec.get("C1_buffered_stage", "").startswith(non_u[0])):
            cls = "immediate_pre_sensitive"
        else:
            cls = "partial_baseline_support"
        rows.append({"object_A": key[0], "object_B": key[1], "metric_family": key[2], "metric_name": key[3], "baseline_sensitivity": cls, **{f"{k}_decision": v for k, v in dec.items()}})
    return pd.DataFrame(rows)


def _branch_sensitivity(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(OBJECT_ORDER, 2):
        for layer in ("state", "growth"):
            sub = pairwise_df[(pairwise_df["object_A"] == a) & (pairwise_df["object_B"] == b) & (pairwise_df["metric_family"].astype(str).str.startswith(layer))]
            decisions = sub["decision"].astype(str).tolist() if not sub.empty else []
            has_a = any(d.startswith("A") for d in decisions)
            has_b = any(d.startswith("B") for d in decisions)
            if has_a and has_b:
                cls = "branch_conflict"
            elif has_a:
                cls = "A_direction_in_at_least_one_branch"
            elif has_b:
                cls = "B_direction_in_at_least_one_branch"
            else:
                cls = "branch_unresolved"
            rows.append({"object_A": a, "object_B": b, "layer": layer, "branch_sensitivity": cls, "n_metrics": len(decisions)})
    return pd.DataFrame(rows)


def _detector_sensitivity(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(OBJECT_ORDER, 2):
        sub = pairwise_df[(pairwise_df["object_A"] == a) & (pairwise_df["object_B"] == b) & (pairwise_df["metric_family"].astype(str).str.contains("peak_timing", na=False))]
        raw_dec = sub[sub["detector_type"] == "raw_profile"]["decision"].astype(str).tolist() if "detector_type" in sub.columns else []
        shp_dec = sub[sub["detector_type"] == "shape_pattern"]["decision"].astype(str).tolist() if "detector_type" in sub.columns else []
        def sign(dec):
            if any(d.startswith("A") for d in dec):
                return "A"
            if any(d.startswith("B") for d in dec):
                return "B"
            return "U"
        rs, ss = sign(raw_dec), sign(shp_dec)
        if rs == ss and rs != "U":
            cls = "detector_consistent"
        elif rs != "U" and ss != "U" and rs != ss:
            cls = "detector_conflict"
        elif rs != "U":
            cls = "raw_profile_only"
        elif ss != "U":
            cls = "shape_pattern_only"
        else:
            cls = "detector_unresolved"
        rows.append({"object_A": a, "object_B": b, "raw_profile_relation": rs, "shape_pattern_relation": ss, "detector_sensitivity": cls})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Hotfix 01: stricter W45-relevant evidence gate
# -----------------------------------------------------------------------------

def _support_rank(cls: str) -> int:
    cls = str(cls)
    if cls.startswith("accepted"):
        return 3
    if cls.startswith("candidate"):
        return 2
    if cls.startswith("weak"):
        return 1
    return 0


def _w45_relevance_score(row: pd.Series) -> tuple:
    """Rank candidates for pairwise timing.

    Hotfix rule: do not mechanically use CP001. Prefer accepted/candidate windows
    that overlap W45 or its immediate front/back diagnostic zone. Weak early peaks
    cannot automatically become the main timing candidate when a stronger W45-relevant
    candidate exists.
    """
    support = _support_rank(row.get("support_class", ""))
    overlap = float(row.get("overlap_days_with_W45", 0) or 0)
    peak = float(row.get("peak_day", np.nan))
    # Main W45 context includes W45 and its immediate front/back range.
    if np.isfinite(peak):
        distance_to_anchor = abs(peak - ANCHOR_DAY)
        in_context = 1 if 25 <= peak <= 55 else 0
        front_context = 1 if 30 <= peak <= 45 else 0
    else:
        distance_to_anchor = 999
        in_context = 0
        front_context = 0
    # Prefer support, W45/context relevance, then high score/prominence, then closeness.
    return (
        support,
        1 if overlap > 0 else 0,
        in_context,
        front_context,
        float(row.get("bootstrap_match_fraction", np.nan) or 0),
        float(row.get("peak_score", np.nan) or 0),
        -distance_to_anchor,
    )


def _select_w45_relevant_main_windows(windows_df: pd.DataFrame) -> pd.DataFrame:
    if windows_df is None or windows_df.empty:
        return pd.DataFrame()
    rows = []
    for (obj, dtype), sub in windows_df.groupby(["object", "detector_type"], sort=False):
        cand = sub.copy()
        # Ensure needed columns exist even if run output is older.
        if "support_class" not in cand.columns:
            cand["support_class"] = cand.get("bootstrap_match_fraction", np.nan).apply(_support_class)
        cand["_rank_tuple"] = cand.apply(_w45_relevance_score, axis=1)
        cand = cand.sort_values("_rank_tuple", ascending=False)
        sel = cand.iloc[0].copy()
        excluded = []
        for _, r in cand.iloc[1:].iterrows():
            excluded.append(f"{r.get('candidate_id','')}:day{r.get('peak_day','')}/{r.get('support_class','')}")
        role = str(sel.get("relation_to_W45", "unknown"))
        peak = float(sel.get("peak_day", np.nan))
        if np.isfinite(peak) and peak < SYSTEM_W45[0] and role == "overlap_W45":
            role = "W45_front_overlap"
        rows.append({
            "object": obj,
            "detector_type": dtype,
            "selected_candidate_id": sel.get("candidate_id", ""),
            "selected_peak_day": sel.get("peak_day", np.nan),
            "selected_window_start": sel.get("window_start", np.nan),
            "selected_window_end": sel.get("window_end", np.nan),
            "selected_support_class": sel.get("support_class", ""),
            "selected_bootstrap_support": sel.get("bootstrap_match_fraction", np.nan),
            "selected_relation_to_W45": role,
            "selection_reason": "hotfix_01_rank_by_support_W45_overlap_context_score_not_CP001",
            "excluded_candidates": ";".join(excluded),
        })
    return pd.DataFrame(rows)


def _pairwise_selected_peak_delta(match_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if match_df is None or match_df.empty or selected_df is None or selected_df.empty:
        return pd.DataFrame()
    for dtype in ("raw_profile", "shape_pattern"):
        for a, b in combinations(OBJECT_ORDER, 2):
            a_sel = selected_df[(selected_df["object"] == a) & (selected_df["detector_type"] == dtype)]
            b_sel = selected_df[(selected_df["object"] == b) & (selected_df["detector_type"] == dtype)]
            if a_sel.empty or b_sel.empty:
                continue
            a_cid = str(a_sel.iloc[0]["selected_candidate_id"])
            b_cid = str(b_sel.iloc[0]["selected_candidate_id"])
            a_sub = match_df[(match_df["object"] == a) & (match_df["detector_type"] == dtype) & (match_df["candidate_id"].astype(str) == a_cid)][["bootstrap_id", "matched_return_day", "matched"]].rename(columns={"matched_return_day": "a_day", "matched": "a_matched"})
            b_sub = match_df[(match_df["object"] == b) & (match_df["detector_type"] == dtype) & (match_df["candidate_id"].astype(str) == b_cid)][["bootstrap_id", "matched_return_day", "matched"]].rename(columns={"matched_return_day": "b_day", "matched": "b_matched"})
            if a_sub.empty or b_sub.empty:
                continue
            m = a_sub.merge(b_sub, on="bootstrap_id", how="inner")
            m = m[(m["a_matched"] == True) & (m["b_matched"] == True)]
            if m.empty:
                continue
            vals = m["b_day"].astype(float).to_numpy() - m["a_day"].astype(float).to_numpy()
            q = _direction_decision(vals, "A_earlier", "B_earlier")
            rows.append({
                "object_A": a,
                "object_B": b,
                "detector_type": dtype,
                "metric_family": f"{dtype}_selected_peak_timing",
                "metric_name": "selected_peak_delta_B_minus_A",
                "delta_definition": "B_selected_peak_day_minus_A_selected_peak_day_positive_means_A_earlier",
                "A_selected_candidate_id": a_cid,
                "B_selected_candidate_id": b_cid,
                "A_selected_peak_day": a_sel.iloc[0]["selected_peak_day"],
                "B_selected_peak_day": b_sel.iloc[0]["selected_peak_day"],
                "delta_median": q["median"],
                "delta_q05": q["q05"],
                "delta_q95": q["q95"],
                "delta_q025": q["q025"],
                "delta_q975": q["q975"],
                "P_A_earlier_or_higher": q["P_positive"],
                "P_B_earlier_or_higher": q["P_negative"],
                "decision": q["decision"],
                "n_valid": q["n_valid"],
            })
    return pd.DataFrame(rows)


def _family_decision_from_metric_rows(rows: pd.DataFrame) -> str:
    if rows is None or rows.empty:
        return "unresolved"
    decs = rows["decision"].astype(str).tolist()
    a_sup = sum(d.startswith("A") and "supported" in d for d in decs)
    b_sup = sum(d.startswith("B") and "supported" in d for d in decs)
    a_ten = sum(d.startswith("A") and "tendency" in d for d in decs)
    b_ten = sum(d.startswith("B") and "tendency" in d for d in decs)
    if (a_sup or a_ten) and (b_sup or b_ten):
        return "conflict"
    if a_sup:
        return "A_supported"
    if b_sup:
        return "B_supported"
    if a_ten:
        return "A_tendency"
    if b_ten:
        return "B_tendency"
    return "unresolved"


def _selected_window_overlap_relation(selected_df: pd.DataFrame, a: str, b: str, dtype: str) -> tuple[str, int]:
    a_row = selected_df[(selected_df["object"] == a) & (selected_df["detector_type"] == dtype)]
    b_row = selected_df[(selected_df["object"] == b) & (selected_df["detector_type"] == dtype)]
    if a_row.empty or b_row.empty:
        return "unresolved", 0
    ar, br = a_row.iloc[0], b_row.iloc[0]
    ov = _window_overlap_days(ar["selected_window_start"], ar["selected_window_end"], br["selected_window_start"], br["selected_window_end"])
    if ov >= 5:
        return "co_transition", ov
    return "not_overlapping", ov


def _build_evidence_family_summary(pairwise_df: pd.DataFrame, selected_delta: pd.DataFrame, selected_windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Use selected peak timing rather than CP001 timing for window families.
    combined = pairwise_df.copy()
    if selected_delta is not None and not selected_delta.empty:
        combined = pd.concat([combined, selected_delta], ignore_index=True, sort=False)
    for a, b in combinations(OBJECT_ORDER, 2):
        for dtype in ("raw_profile", "shape_pattern"):
            fam_name = f"{dtype}_window_family"
            metric_sub = selected_delta[(selected_delta["object_A"] == a) & (selected_delta["object_B"] == b) & (selected_delta["detector_type"] == dtype)] if selected_delta is not None and not selected_delta.empty else pd.DataFrame()
            fam = _family_decision_from_metric_rows(metric_sub)
            overlap_relation, overlap_days = _selected_window_overlap_relation(selected_windows, a, b, dtype)
            # Overlap veto: unresolved selected timing + strong window overlap = co-transition family.
            if overlap_relation == "co_transition" and fam == "unresolved":
                fam = "co_transition"
            rows.append({"object_A": a, "object_B": b, "evidence_family": fam_name, "family_decision": fam, "window_overlap_days": overlap_days})
        for fam in ("state_distance", "state_pattern", "growth_distance", "growth_pattern"):
            sub = pairwise_df[(pairwise_df["object_A"] == a) & (pairwise_df["object_B"] == b) & (pairwise_df["metric_family"] == fam)]
            rows.append({"object_A": a, "object_B": b, "evidence_family": f"{fam}_family", "family_decision": _family_decision_from_metric_rows(sub), "window_overlap_days": np.nan})
    return pd.DataFrame(rows)


def _hardened_structure_from_families(fam: pd.DataFrame) -> tuple[str, str, str, str]:
    d = {r["evidence_family"]: r["family_decision"] for _, r in fam.iterrows()}
    raw = d.get("raw_profile_window_family", "unresolved")
    shp = d.get("shape_pattern_window_family", "unresolved")
    curve_fams = [d.get(k, "unresolved") for k in ("state_distance_family", "state_pattern_family", "growth_distance_family", "growth_pattern_family")]
    a_curve = sum(x in ("A_supported", "A_tendency") for x in curve_fams)
    b_curve = sum(x in ("B_supported", "B_tendency") for x in curve_fams)
    a_curve_supported = any(x == "A_supported" for x in curve_fams)
    b_curve_supported = any(x == "B_supported" for x in curve_fams)
    curve_conflict = a_curve > 0 and b_curve > 0
    detector_conflict = (raw.startswith("A") and shp.startswith("B")) or (raw.startswith("B") and shp.startswith("A"))
    cotrans_both = raw == "co_transition" and shp == "co_transition"
    # Layer-specific states.
    if detector_conflict:
        return "detector_split", "detector_conflict", "branch_or_detector_specific", "detector families disagree; do not collapse to single order"
    if cotrans_both:
        if a_curve and not b_curve:
            return "co_transition_with_A_curve_tendency", "window_cotransition_curve_tendency", "Level2_curve_tendency_only", "window overlap veto prevents A_leads_B"
        if b_curve and not a_curve:
            return "co_transition_with_B_curve_tendency", "window_cotransition_curve_tendency", "Level2_curve_tendency_only", "window overlap veto prevents B_leads_A"
        return "co_transition", "window_cotransition", "Level3_window_overlap", "selected windows overlap"
    # One detector family has direction while the other is co-transition/unresolved.
    for side, lab in (("A", "A"), ("B", "B")):
        window_dir = raw.startswith(side) or shp.startswith(side)
        other_detector_cotrans = (raw == "co_transition" or shp == "co_transition")
        if window_dir and other_detector_cotrans:
            if side == "A" and a_curve:
                return "A_layer_specific_lead_or_front", "layer_specific_with_curve_support", "Level3_single_family_supported", "one detector or layer supports A, other co-transitions/unresolved"
            if side == "B" and b_curve:
                return "B_layer_specific_lead_or_front", "layer_specific_with_curve_support", "Level3_single_family_supported", "one detector or layer supports B, other co-transitions/unresolved"
            return f"{side}_detector_specific_timing", "detector_specific_only", "Level2_single_detector", "detector-specific signal without curve support"
    # Stronger lead needs at least one window direction and curve support in same direction.
    if (raw.startswith("A") or shp.startswith("A")) and (a_curve_supported or a_curve >= 2) and not curve_conflict:
        return "A_leads_B_candidate", "window_and_curve_A", "Level4_window_and_curve_supported", "selected window plus curve families favor A"
    if (raw.startswith("B") or shp.startswith("B")) and (b_curve_supported or b_curve >= 2) and not curve_conflict:
        return "B_leads_A_candidate", "window_and_curve_B", "Level4_window_and_curve_supported", "selected window plus curve families favor B"
    if curve_conflict:
        return "branch_split", "curve_branch_conflict", "Level1_conflict", "curve families disagree"
    if a_curve and not b_curve:
        return "A_curve_tendency_only", "curve_only", "Level2_curve_tendency_only", "no selected-window timing support"
    if b_curve and not a_curve:
        return "B_curve_tendency_only", "curve_only", "Level2_curve_tendency_only", "no selected-window timing support"
    return "unresolved", "unresolved", "Level0_unresolved", "no sufficient family-level evidence"


def _classify_structures_hardened(family_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if family_df is None or family_df.empty:
        return pd.DataFrame()
    for (a, b), fam in family_df.groupby(["object_A", "object_B"], sort=False):
        cls, sens, level, note = _hardened_structure_from_families(fam)
        rows.append({
            "object_A": a,
            "object_B": b,
            "final_structure_class": cls,
            "hardened_sensitivity_status": sens,
            "hardened_evidence_level": level,
            "hotfix_note": note,
            "allowed_statement": f"For {a}-{b}, hotfix_01 classification is {cls}; use family table before interpretation.",
            "forbidden_statement": "Do not infer causality/pathway; do not upgrade co-transition or curve-only tendency into a lead claim.",
        })
    return pd.DataFrame(rows)


def _evidence_gates_hardened(class_df: pd.DataFrame, family_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gates, finals, downgraded = [], [], []
    final_allowed_prefixes = {
        "A_leads_B_candidate", "B_leads_A_candidate", "A_layer_specific_lead_or_front", "B_layer_specific_lead_or_front",
        "co_transition", "co_transition_with_A_curve_tendency", "co_transition_with_B_curve_tendency", "detector_split", "branch_split",
    }
    for _, r in class_df.iterrows():
        a, b, cls = r["object_A"], r["object_B"], r["final_structure_class"]
        fam = family_df[(family_df["object_A"] == a) & (family_df["object_B"] == b)]
        fam_text = ";".join([f"{rr['evidence_family']}={rr['family_decision']}" for _, rr in fam.iterrows()])
        pass_gate = cls in final_allowed_prefixes and not cls.endswith("curve_tendency_only")
        # Co-transition with curve tendency is allowed as a low-level final only if both window families co-transition.
        if cls.startswith("co_transition_with_"):
            pass_gate = True
        reason = "passed_hardened_gate" if pass_gate else "downgraded_by_hardened_gate"
        claim_id = f"{a}_{b}_{cls}"
        gates.append({
            "claim_id": claim_id,
            "object_A": a,
            "object_B": b,
            "claim_type": cls,
            "required_evidence": "hotfix_01 family-level gate with co-transition veto and selected-window timing",
            "available_evidence": fam_text,
            "gate_pass": bool(pass_gate),
            "failure_reason": "" if pass_gate else reason,
        })
        row_common = {
            "object_A": a,
            "object_B": b,
            "final_structure_class": cls,
            "evidence_level": r["hardened_evidence_level"],
            "supporting_metrics": fam_text,
            "sensitivity_status": r["hardened_sensitivity_status"],
            "allowed_statement": r["allowed_statement"],
            "forbidden_statement": r["forbidden_statement"],
        }
        if pass_gate:
            finals.append({"claim_id": claim_id, "claim": f"{a}-{b}: {cls}", **row_common})
        else:
            downgraded.append({
                "signal_id": claim_id,
                "object": "",
                "object_A": a,
                "object_B": b,
                "signal_description": f"{a}-{b}: {cls}",
                "observed_evidence": fam_text,
                "why_not_final": reason,
                "possible_interpretation": "retain as downgraded signal; do not use as final claim",
                "next_test_needed": "inspect family-level evidence and selected main-window table",
            })
    return pd.DataFrame(gates), pd.DataFrame(finals), pd.DataFrame(downgraded)


def _classify_structures(pairwise_df: pd.DataFrame, base_sens: pd.DataFrame, branch_sens: pd.DataFrame, det_sens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(OBJECT_ORDER, 2):
        sub = pairwise_df[(pairwise_df["object_A"] == a) & (pairwise_df["object_B"] == b)]
        decs = sub["decision"].astype(str).tolist() if not sub.empty else []
        a_count = sum(d.startswith("A") for d in decs)
        b_count = sum(d.startswith("B") for d in decs)
        # Basic sensitivity checks.
        bs = base_sens[(base_sens["object_A"] == a) & (base_sens["object_B"] == b)]
        baseline_sensitive = bool((bs.get("baseline_sensitivity", pd.Series(dtype=str)).astype(str) == "baseline_sensitive").any()) if not bs.empty else False
        br = branch_sens[(branch_sens["object_A"] == a) & (branch_sens["object_B"] == b)]
        branch_conflict = bool((br.get("branch_sensitivity", pd.Series(dtype=str)).astype(str) == "branch_conflict").any()) if not br.empty else False
        ds = det_sens[(det_sens["object_A"] == a) & (det_sens["object_B"] == b)]
        detector_conflict = bool((ds.get("detector_sensitivity", pd.Series(dtype=str)).astype(str) == "detector_conflict").any()) if not ds.empty else False
        if baseline_sensitive:
            cls = "baseline_sensitive"
        elif detector_conflict:
            cls = "detector_split"
        elif branch_conflict and a_count and b_count:
            cls = "branch_split"
        elif a_count >= 3 and b_count == 0:
            cls = "A_leads_B"
        elif b_count >= 3 and a_count == 0:
            cls = "B_leads_A"
        elif a_count >= 2 and b_count >= 1:
            cls = "A_early_B_catchup_or_branch_split"
        elif b_count >= 2 and a_count >= 1:
            cls = "B_early_A_catchup_or_branch_split"
        elif a_count == 0 and b_count == 0:
            cls = "unresolved"
        else:
            cls = "weak_directional_tendency"
        allowed = f"For {a}-{b}, current V7-z classification is {cls}; consult metric-level rows before interpretation."
        forbidden = "Do not infer causality/pathway; do not collapse branch-sensitive results into a single order."
        rows.append({
            "object_A": a,
            "object_B": b,
            "n_A_direction_metrics": a_count,
            "n_B_direction_metrics": b_count,
            "baseline_sensitivity": "baseline_sensitive" if baseline_sensitive else "not_flagged",
            "branch_sensitivity": "branch_conflict" if branch_conflict else "not_flagged",
            "detector_sensitivity": "detector_conflict" if detector_conflict else "not_flagged",
            "final_structure_class": cls,
            "allowed_statement": allowed,
            "forbidden_statement": forbidden,
        })
    return pd.DataFrame(rows)


def _evidence_gates(class_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gates, finals, downgraded = [], [], []
    for _, r in class_df.iterrows():
        a, b, cls = r["object_A"], r["object_B"], r["final_structure_class"]
        sub = pairwise_df[(pairwise_df["object_A"] == a) & (pairwise_df["object_B"] == b)]
        supported = sub[sub["decision"].astype(str).str.contains("supported", na=False)]
        tendency = sub[sub["decision"].astype(str).str.contains("tendency", na=False)]
        pass_gate = False
        level = "observed_or_unresolved"
        if len(supported) >= 2 and cls not in {"baseline_sensitive", "detector_split", "branch_split", "unresolved", "invalid"}:
            pass_gate = True
            level = "bootstrap_supported_multi_metric"
        elif len(tendency) + len(supported) >= 2 and cls not in {"baseline_sensitive", "detector_split", "unresolved", "invalid"}:
            pass_gate = True
            level = "bootstrap_tendency_multi_metric"
        reason = "passed" if pass_gate else f"not_enough_supported_metrics_or_sensitive_class; supported={len(supported)}, tendency={len(tendency)}, class={cls}"
        claim_id = f"{a}_{b}_{cls}"
        gates.append({
            "claim_id": claim_id,
            "object_A": a,
            "object_B": b,
            "claim_type": cls,
            "required_evidence": "at least two supported/tendency metrics and no unresolved sensitivity class",
            "available_evidence": f"supported={len(supported)}; tendency={len(tendency)}",
            "gate_pass": bool(pass_gate),
            "failure_reason": reason if not pass_gate else "",
        })
        if pass_gate:
            finals.append({
                "claim_id": claim_id,
                "claim": f"{a}-{b}: {cls}",
                "object_A": a,
                "object_B": b,
                "final_structure_class": cls,
                "evidence_level": level,
                "supporting_metrics": ";".join(supported["metric_family"].astype(str).head(8).tolist() + tendency["metric_family"].astype(str).head(8).tolist()),
                "sensitivity_status": f"baseline={r['baseline_sensitivity']}; branch={r['branch_sensitivity']}; detector={r['detector_sensitivity']}",
                "allowed_statement": r["allowed_statement"],
                "forbidden_statement": r["forbidden_statement"],
            })
        else:
            downgraded.append({
                "signal_id": claim_id,
                "object": "",
                "object_A": a,
                "object_B": b,
                "signal_description": f"{a}-{b}: {cls}",
                "observed_evidence": f"supported={len(supported)}; tendency={len(tendency)}",
                "why_not_final": reason,
                "possible_interpretation": "retain as downgraded signal; do not use as final claim",
                "next_test_needed": "inspect metric rows and sensitivity tables",
            })
    return pd.DataFrame(gates), pd.DataFrame(finals), pd.DataFrame(downgraded)


# -----------------------------------------------------------------------------
# Figure and summary helpers
# -----------------------------------------------------------------------------


def _write_summary(paths: Paths, class_df: pd.DataFrame, final_df: pd.DataFrame, downgraded_df: pd.DataFrame, cfg: V7ZConfig) -> None:
    lines = []
    lines.append("# V7-z W45 multi-object pre-post statistical validation")
    lines.append("")
    lines.append(f"Run time: {_now_iso()}")
    lines.append(f"Window: {WINDOW_ID}, accepted_window=day{SYSTEM_W45[0]}–{SYSTEM_W45[1]}, anchor_day={ANCHOR_DAY}")
    lines.append("")
    lines.append("## Scope")
    lines.append("Fixed W45; objects P/V/H/Je/Jw; profile-object and shape-pattern detectors; C0/C1/C2 pre-post curves; paired year bootstrap; evidence gate.")
    lines.append("")
    lines.append("## Final claims")
    if final_df.empty:
        lines.append("No final claims passed the current evidence gate. See downgraded signal registry.")
    else:
        for _, r in final_df.iterrows():
            lines.append(f"- **{r['claim']}** — {r['evidence_level']}; {r['sensitivity_status']}")
    lines.append("")
    lines.append("## Timing structure classifications")
    if class_df.empty:
        lines.append("No pairwise classifications available.")
    else:
        for _, r in class_df.iterrows():
            obj_a = r.get("object_A", "")
            obj_b = r.get("object_B", "")
            cls = r.get("final_structure_class", "")
            # Hotfix 02: summary must support both the legacy row-count classification
            # table and the hardened family-level classification table introduced in
            # hotfix_01. The hardened table intentionally no longer contains
            # n_A_direction_metrics / n_B_direction_metrics, so use available fields
            # rather than indexing columns directly.
            if "n_A_direction_metrics" in r.index or "n_B_direction_metrics" in r.index:
                a_metrics = r.get("n_A_direction_metrics", "")
                b_metrics = r.get("n_B_direction_metrics", "")
                detail = f" (A metrics={a_metrics}, B metrics={b_metrics})"
            else:
                level = r.get("hardened_evidence_level", r.get("evidence_level", ""))
                sens = r.get("hardened_sensitivity_status", r.get("sensitivity_status", ""))
                detail_parts = [str(x) for x in (level, sens) if str(x)]
                detail = f" ({'; '.join(detail_parts)})" if detail_parts else ""
            lines.append(f"- {obj_a}-{obj_b}: {cls}{detail}")
    lines.append("")
    lines.append("## Downgraded signals")
    lines.append(f"Number of downgraded signals: {len(downgraded_df)}")
    lines.append("")
    lines.append("## Forbidden interpretations")
    lines.append("- Do not infer causality/pathway from V7-z timing structure.")
    lines.append("- Do not collapse branch-split or baseline-sensitive results into a single winner.")
    lines.append("- Do not treat candidate_window support <0.95 as accepted_window.")
    _write_text("\n".join(lines), paths.output_dir / "W45_multi_object_summary_v7_z.md")


def _maybe_write_figures(paths: Paths, score_df: pd.DataFrame, state_df: pd.DataFrame, growth_df: pd.DataFrame, class_df: pd.DataFrame, cfg: V7ZConfig) -> None:
    if not cfg.write_figures:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    # Detector score figures by detector type.
    for dtype, fname in (("raw_profile", "W45_multi_object_profile_window_scores_v7_z.png"), ("shape_pattern", "W45_multi_object_shape_pattern_window_scores_v7_z.png")):
        sub = score_df[score_df["detector_type"] == dtype]
        if sub.empty:
            continue
        plt.figure(figsize=(10, 5))
        for obj, g in sub.groupby("object", sort=False):
            plt.plot(g["day"], g["detector_score"], label=obj)
        plt.axvspan(SYSTEM_W45[0], SYSTEM_W45[1], alpha=0.15)
        plt.title(f"{dtype} detector scores")
        plt.xlabel("day")
        plt.ylabel("score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.figure_dir / fname, dpi=180)
        plt.close()
    # State progress quick figure C0.
    for branch, col, fname in (("distance", "S_dist", "W45_multi_object_state_progress_curves_v7_z.png"), ("pattern", "S_pattern", "W45_multi_object_state_pattern_curves_v7_z.png")):
        sub = state_df[state_df["baseline_config"] == "C0_full_stage"]
        if sub.empty:
            continue
        plt.figure(figsize=(10, 5))
        for obj, g in sub.groupby("object", sort=False):
            plt.plot(g["day"], g[col], label=obj)
        plt.axvspan(SYSTEM_W45[0], SYSTEM_W45[1], alpha=0.15)
        plt.title(f"C0 {branch} state progress")
        plt.xlabel("day")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.figure_dir / fname, dpi=180)
        plt.close()


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def run_W45_multi_object_prepost_stat_validation_v7_z(v7_root: Optional[Path] = None) -> dict[str, Any]:
    debug_n = os.environ.get("V7Z_DEBUG_N_BOOTSTRAP")
    skip_fig = os.environ.get("V7Z_SKIP_FIGURES", "0") == "1"
    cfg = V7ZConfig(
        bootstrap=BootstrapConfig(n_bootstrap=int(debug_n) if debug_n else 1000),
        write_figures=not skip_fig,
    )
    paths = _resolve_paths(v7_root, cfg)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{_now_iso()}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("[1/9] Load input fields")
    env_npz = os.environ.get("V7Z_SMOOTHED_FIELDS_NPZ")
    smoothed_path = Path(env_npz) if env_npz else paths.project_root / cfg.smoothed_fields_relpath
    files = _load_npz(smoothed_path)
    years = np.asarray(files.get("years"), dtype=int)
    if years.size == 0:
        raise KeyError("smoothed fields npz must contain years")

    log("[2/9] Build object profiles")
    profiles, input_audit = _build_profiles(files, cfg, paths.output_dir)
    obj_reg = pd.DataFrame([
        {"object": obj, "field_key": p.field_key, "lat_min": p.lat_range[0], "lat_max": p.lat_range[1], "lon_min": p.lon_range[0], "lon_max": p.lon_range[1], "n_years": p.raw_cube.shape[0], "n_days": p.raw_cube.shape[1], "n_lat_features": p.raw_cube.shape[2]}
        for obj, p in profiles.items()
    ])
    _write_csv(obj_reg, paths.output_dir / "W45_multi_object_object_registry_v7_z.csv")
    baseline_df = pd.DataFrame([{ "baseline_config": b.name, "pre_start": b.pre[0], "pre_end": b.pre[1], "post_start": b.post[0], "post_end": b.post[1], "role": b.role, "notes": b.notes } for b in BASELINE_CONFIGS])
    _write_csv(baseline_df, paths.output_dir / "W45_multi_object_baseline_config_table_v7_z.csv")

    log("[3/9] Run observed profile and shape-pattern detectors")
    observed = _detector_observed(profiles, cfg)
    score_df = observed["score_df"]
    windows_df = observed["windows_df"]
    _write_csv(score_df, paths.output_dir / "W45_multi_object_detector_score_profile_v7_z.csv")
    raw_win = windows_df[windows_df["detector_type"] == "raw_profile"].copy() if not windows_df.empty else pd.DataFrame()
    shp_win = windows_df[windows_df["detector_type"] == "shape_pattern"].copy() if not windows_df.empty else pd.DataFrame()
    _write_csv(raw_win, paths.output_dir / "W45_multi_object_profile_window_registry_v7_z.csv")
    _write_csv(shp_win, paths.output_dir / "W45_multi_object_shape_pattern_window_registry_v7_z.csv")

    log("[4/9] Compute observed pre-post state/growth curves")
    state_df, growth_df, observed_obj_metrics = _observed_state_growth(profiles)
    _write_csv(state_df, paths.output_dir / "W45_multi_object_state_progress_curves_v7_z.csv")
    _write_csv(growth_df, paths.output_dir / "W45_multi_object_growth_speed_curves_v7_z.csv")
    observed_pair_metrics = _pairwise_metrics_from_obj_metrics(observed_obj_metrics)

    log("[5/9] Run paired bootstrap")
    rng = np.random.default_rng(cfg.bootstrap.random_seed)
    all_match_rows: list[dict[str, Any]] = []
    all_obj_metric_frames: list[pd.DataFrame] = []
    all_pair_metric_frames: list[pd.DataFrame] = []
    n_years = years.size
    for b in range(cfg.bootstrap.n_bootstrap):
        if cfg.bootstrap.progress and (b == 0 or (b + 1) % max(1, cfg.bootstrap.n_bootstrap // 10) == 0):
            log(f"  bootstrap {b + 1}/{cfg.bootstrap.n_bootstrap}")
        sampled_idx = rng.integers(0, n_years, size=n_years)
        match_rows, b_state, b_growth = _bootstrap_iteration(b, sampled_idx, profiles, cfg, observed["observed_candidates"])
        all_match_rows.extend(match_rows)
        b_obj_metrics = _object_state_growth_metrics(b_state, b_growth)
        b_obj_metrics["bootstrap_id"] = b
        all_obj_metric_frames.append(b_obj_metrics)
        b_pair = _pairwise_metrics_from_obj_metrics(b_obj_metrics)
        b_pair["bootstrap_id"] = b
        all_pair_metric_frames.append(b_pair)

    match_df = pd.DataFrame(all_match_rows)
    bootstrap_obj_metrics = pd.concat(all_obj_metric_frames, ignore_index=True) if all_obj_metric_frames else pd.DataFrame()
    bootstrap_pair_metrics = pd.concat(all_pair_metric_frames, ignore_index=True) if all_pair_metric_frames else pd.DataFrame()

    log("[6/9] Summarize bootstrap and pairwise statistics")
    window_boot = _summarize_window_bootstrap(match_df, windows_df)
    _write_csv(window_boot, paths.output_dir / "W45_multi_object_window_bootstrap_summary_v7_z.csv")
    _write_csv(match_df, paths.output_dir / "W45_multi_object_peak_return_day_distribution_v7_z.csv")
    # Merge window support back to registries.
    if not raw_win.empty:
        raw_out = raw_win.merge(window_boot, on=["object", "detector_type", "candidate_id"], how="left")
    else:
        raw_out = raw_win
    if not shp_win.empty:
        shp_out = shp_win.merge(window_boot, on=["object", "detector_type", "candidate_id"], how="left")
    else:
        shp_out = shp_win
    _write_csv(raw_out, paths.output_dir / "W45_multi_object_profile_window_registry_v7_z.csv")
    _write_csv(shp_out, paths.output_dir / "W45_multi_object_shape_pattern_window_registry_v7_z.csv")

    single_summary = _summarize_single_metrics(observed_obj_metrics, bootstrap_obj_metrics)
    _write_csv(single_summary, paths.output_dir / "W45_multi_object_single_object_stat_summary_v7_z.csv")

    # Hotfix 01: select W45-relevant main windows before pairwise peak timing.
    selected_windows = _select_w45_relevant_main_windows(windows_df.merge(window_boot, on=["object", "detector_type", "candidate_id"], how="left") if not window_boot.empty else windows_df)
    _write_csv(selected_windows, paths.output_dir / "W45_multi_object_main_window_selection_v7_z_hotfix_01.csv")
    selected_peak_delta = _pairwise_selected_peak_delta(match_df, selected_windows)
    _write_csv(selected_peak_delta, paths.output_dir / "W45_multi_object_selected_peak_delta_v7_z_hotfix_01.csv")

    # Keep legacy CP001 timing in pairwise metric table for audit, but classify using selected-window families.
    legacy_peak_delta = _pairwise_peak_delta(match_df)
    pairwise_summary = _summarize_pairwise(observed_pair_metrics, bootstrap_pair_metrics, pd.concat([legacy_peak_delta, selected_peak_delta], ignore_index=True, sort=False) if not selected_peak_delta.empty else legacy_peak_delta)
    _write_csv(pairwise_summary, paths.output_dir / "W45_multi_object_pairwise_stat_summary_v7_z.csv")

    log("[7/9] Classify sensitivity and timing structures")
    baseline_sens = _baseline_sensitivity(pairwise_summary)
    branch_sens = _branch_sensitivity(pairwise_summary)
    detector_sens = _detector_sensitivity(pairwise_summary)
    # Legacy classification is retained for audit, but final claims use hardened family-level classification.
    legacy_class_df = _classify_structures(pairwise_summary, baseline_sens, branch_sens, detector_sens)
    family_df = _build_evidence_family_summary(pairwise_summary, selected_peak_delta, selected_windows)
    class_df = _classify_structures_hardened(family_df)
    _write_csv(baseline_sens, paths.output_dir / "W45_multi_object_baseline_sensitivity_summary_v7_z.csv")
    _write_csv(branch_sens, paths.output_dir / "W45_multi_object_branch_sensitivity_summary_v7_z.csv")
    _write_csv(detector_sens, paths.output_dir / "W45_multi_object_detector_sensitivity_summary_v7_z.csv")
    _write_csv(legacy_class_df, paths.output_dir / "W45_multi_object_timing_structure_classification_legacy_v7_z_hotfix_01.csv")
    _write_csv(family_df, paths.output_dir / "W45_multi_object_evidence_family_summary_v7_z_hotfix_01.csv")
    _write_csv(class_df, paths.output_dir / "W45_multi_object_timing_structure_classification_v7_z.csv")

    log("[8/9] Apply hardened evidence gates")
    gate_df, final_df, downgraded_df = _evidence_gates_hardened(class_df, family_df)
    _write_csv(gate_df, paths.output_dir / "W45_multi_object_evidence_gate_table_v7_z.csv")
    _write_csv(final_df, paths.output_dir / "W45_multi_object_final_claim_registry_v7_z.csv")
    _write_csv(downgraded_df, paths.output_dir / "W45_multi_object_downgraded_signal_registry_v7_z.csv")

    log("[9/9] Write outputs and summary")
    config_json = {
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "system_W45": SYSTEM_W45,
        "objects": cfg.objects,
        "object_detection_range": OBJECT_DETECTION_RANGE,
        "curve_range": CURVE_RANGE,
        "main_comparison_range": MAIN_COMPARISON_RANGE,
        "segments": SEGMENTS,
        "baseline_configs": [asdict(b) for b in BASELINE_CONFIGS],
        "detector": asdict(cfg.detector),
        "band": asdict(cfg.band),
        "bootstrap": asdict(cfg.bootstrap),
        "smoothed_fields_path": str(smoothed_path),
    }
    _write_json(config_json, paths.output_dir / "W45_multi_object_run_config_v7_z.json")
    _write_json({**config_json, "run_time": _now_iso(), "n_final_claims": len(final_df), "n_downgraded_signals": len(downgraded_df)}, paths.output_dir / "run_meta.json")
    _write_summary(paths, class_df, final_df, downgraded_df, cfg)
    _maybe_write_figures(paths, score_df, state_df, growth_df, class_df, cfg)
    _write_text("\n".join(log_lines), paths.log_dir / "run.log")

    return {
        "output_dir": str(paths.output_dir),
        "n_final_claims": int(len(final_df)),
        "n_downgraded_signals": int(len(downgraded_df)),
        "n_pairwise_rows": int(len(pairwise_summary)),
    }


if __name__ == "__main__":
    run_W45_multi_object_prepost_stat_validation_v7_z()
