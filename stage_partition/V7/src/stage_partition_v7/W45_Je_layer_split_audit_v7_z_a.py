from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

VERSION = "v7_z_je_audit_a"
OUTPUT_TAG = "W45_Je_layer_split_audit_v7_z_a"
SYSTEM_W45 = (40, 48)
AUDIT_RANGE = (0, 70)
JE_RAW_REFERENCE_DAY = 46
JE_SHAPE_REFERENCE_DAY = 33
EPS = 1.0e-12

# Je object definition, kept consistent with V6/V7-z.
JE_LON_RANGE = (120.0, 150.0)
JE_LAT_RANGE = (25.0, 45.0)
LAT_STEP_DEG = 2.0
U200_ALIASES = ("u200_smoothed", "u200", "Je_smoothed", "Je")

DAY33_BEFORE = (27, 32)
DAY33_AFTER = (34, 39)
DAY46_BEFORE = (40, 45)
DAY46_AFTER = (47, 52)


@dataclass(frozen=True)
class JeAuditConfig:
    project_root: str = r"D:\easm_project01"
    smoothed_fields_relpath: str = r"foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz"
    v7z_result_relpath: str = r"stage_partition\V7\outputs\W45_multi_object_prepost_stat_validation_v7_z"
    output_tag: str = OUTPUT_TAG
    write_figures: bool = True
    save_by_year: bool = False


@dataclass
class Paths:
    v7_root: Path
    project_root: Path
    smoothed_fields_path: Path
    v7z_result_dir: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path


# -----------------------------------------------------------------------------
# Basic utilities
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


def _safe_nanmean(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _resolve_paths(v7_root: Optional[Path], cfg: JeAuditConfig) -> Paths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = Path(os.environ.get("V7Z_PROJECT_ROOT", cfg.project_root))
    if not project_root.exists():
        # typical layout: D:/easm_project01/stage_partition/V7 -> project root is parents[1]
        project_root = v7_root.parents[1]

    smoothed_env = os.environ.get("V7Z_SMOOTHED_FIELDS")
    smoothed_fields_path = Path(smoothed_env) if smoothed_env else project_root / cfg.smoothed_fields_relpath

    result_env = os.environ.get("V7Z_RESULT_DIR")
    v7z_result_dir = Path(result_env) if result_env else project_root / cfg.v7z_result_relpath

    output_dir = v7_root / "outputs" / cfg.output_tag
    log_dir = v7_root / "logs" / cfg.output_tag
    figure_dir = output_dir / "figures"
    for p in (output_dir, log_dir, figure_dir):
        _ensure_dir(p)
    return Paths(v7_root, project_root, smoothed_fields_path, v7z_result_dir, output_dir, log_dir, figure_dir)


def _load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find smoothed fields npz: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _resolve_key(files: dict[str, Any], aliases: tuple[str, ...]) -> tuple[str | None, str]:
    for key in aliases:
        if key in files:
            return key, "exact"
    lower = {str(k).lower(): str(k) for k in files.keys()}
    for key in aliases:
        if key.lower() in lower:
            return lower[key.lower()], "case_insensitive"
    return None, "missing"


def _mask_between(arr: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = min(bounds), max(bounds)
    arr = np.asarray(arr, dtype=float)
    return (arr >= lo) & (arr <= hi)


def _lat_weights(lat_grid: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(np.asarray(lat_grid, dtype=float)))
    w[~np.isfinite(w)] = np.nan
    return w


def _weighted_mean_1d(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(x) & np.isfinite(w)
    if good.sum() == 0 or np.nansum(w[good]) <= 0:
        return np.nan
    return float(np.nansum(x[good] * w[good]) / np.nansum(w[good]))


def _weighted_norm_1d(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(x) & np.isfinite(w)
    if good.sum() == 0 or np.nansum(w[good]) <= 0:
        return np.nan
    return float(np.sqrt(np.nansum(w[good] * x[good] ** 2) / np.nansum(w[good])))


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


def _weighted_distance(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return _weighted_norm_1d(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), w)


def _interp_profile_to_grid(profile: np.ndarray, src_lats: np.ndarray, dst_lats: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    out = np.full((profile.shape[0], profile.shape[1], dst_lats.size), np.nan, dtype=float)
    order = np.argsort(src_lats)
    src_sorted = np.asarray(src_lats, dtype=float)[order]
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            vals = profile[i, j, :][order]
            valid = np.isfinite(vals) & np.isfinite(src_sorted)
            if valid.sum() < 2:
                continue
            out[i, j, :] = np.interp(dst_lats, src_sorted[valid], vals[valid], left=np.nan, right=np.nan)
    return out


def _day_to_month_day(day: int) -> str:
    d = int(day)
    for month, length in [(4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]:
        if d < length:
            return f"{month:02d}-{d + 1:02d}"
        d -= length
    return f"day{int(day)}"


# -----------------------------------------------------------------------------
# Profile construction and normalization
# -----------------------------------------------------------------------------


def _standardize_field_dims(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return field as years x days x lat x lon. Abort if ambiguous."""
    arr = np.asarray(field, dtype=float)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        # days x lat x lon; add a synthetic year axis for debug/single-clim files.
        if arr.shape[-2] == len(lat) and arr.shape[-1] == len(lon):
            return arr[None, :, :, :]
    raise ValueError(f"Expected field with dims years x days x lat x lon; got shape {arr.shape}")


def _build_je_profile(files: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str, pd.DataFrame]:
    lat_key, lat_status = _resolve_key(files, ("lat", "latitude", "lats"))
    lon_key, lon_status = _resolve_key(files, ("lon", "longitude", "lons"))
    u_key, u_status = _resolve_key(files, U200_ALIASES)
    audit = pd.DataFrame(
        [
            {"required": "lat", "resolved_key": lat_key, "status": lat_status},
            {"required": "lon", "resolved_key": lon_key, "status": lon_status},
            {"required": "u200", "resolved_key": u_key, "status": u_status, "aliases": ";".join(U200_ALIASES)},
        ]
    )
    if lat_key is None or lon_key is None or u_key is None:
        raise KeyError("Missing one of required keys: lat/lon/u200. See input_key_audit output.")
    lat = np.asarray(files[lat_key], dtype=float)
    lon = np.asarray(files[lon_key], dtype=float)
    field = _standardize_field_dims(files[u_key], lat, lon)
    lat_mask = _mask_between(lat, JE_LAT_RANGE)
    lon_mask = _mask_between(lon, JE_LON_RANGE)
    if not np.any(lat_mask):
        raise ValueError(f"No lat points in Je range {JE_LAT_RANGE}")
    if not np.any(lon_mask):
        raise ValueError(f"No lon points in Je range {JE_LON_RANGE}")
    subset = field[:, :, lat_mask, :][:, :, :, lon_mask]
    prof = _safe_nanmean(subset, axis=-1)  # years x days x selected lat
    src_lats = lat[lat_mask]
    dst_lats = np.arange(min(JE_LAT_RANGE), max(JE_LAT_RANGE) + 1.0e-9, LAT_STEP_DEG)
    prof2 = _interp_profile_to_grid(prof, src_lats, dst_lats)
    return prof2, dst_lats, str(u_key), audit


def _shape_normalize_cube(cube: np.ndarray, lat_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cube = np.asarray(cube, dtype=float)
    w = _lat_weights(lat_grid)
    sqrt_w = np.sqrt(w / np.nanmean(w))
    out = np.full_like(cube, np.nan, dtype=float)
    means = np.full(cube.shape[:2], np.nan, dtype=float)
    norms = np.full(cube.shape[:2], np.nan, dtype=float)
    stds = np.full(cube.shape[:2], np.nan, dtype=float)
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
            means[i, j] = mu
            norms[i, j] = norm
            stds[i, j] = np.nanstd(row[good])
            if not np.isfinite(norm) or norm < EPS:
                continue
            out[i, j, :] = (centered / norm) * sqrt_w
            valid[i, j] = True
    return out, means, norms, stds, valid


def _climatology(cube: np.ndarray) -> np.ndarray:
    return _safe_nanmean(cube, axis=0)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def _profile_to_rows(mat: np.ndarray, lat_grid: np.ndarray, value_name: str) -> pd.DataFrame:
    rows = []
    for day in range(mat.shape[0]):
        if day < AUDIT_RANGE[0] or day > AUDIT_RANGE[1]:
            continue
        for lat, val in zip(lat_grid, mat[day, :]):
            rows.append({"day": int(day), "date_mmdd": _day_to_month_day(day), "lat": float(lat), value_name: float(val) if np.isfinite(val) else np.nan})
    return pd.DataFrame(rows)


def _feature_timeseries(raw_clim: np.ndarray, lat_grid: np.ndarray) -> pd.DataFrame:
    w = _lat_weights(lat_grid)
    rows = []
    for day in range(raw_clim.shape[0]):
        if day < AUDIT_RANGE[0] or day > AUDIT_RANGE[1]:
            continue
        row = raw_clim[day, :]
        good = np.isfinite(row) & np.isfinite(w)
        if good.sum() < 2:
            rows.append({"day": day, "feature_valid": False})
            continue
        rg = row[good]
        lg = lat_grid[good]
        wg = w[good]
        mean = _weighted_mean_1d(row, w)
        l2 = _weighted_norm_1d(row, w)
        centered_norm = _weighted_norm_1d(row - mean, w)
        max_idx = int(np.nanargmax(rg))
        min_idx = int(np.nanargmin(rg))
        amplitude = float(np.nanmax(rg) - np.nanmin(rg))
        pos = np.maximum(rg, 0.0)
        if np.nansum(pos) > EPS:
            centroid = float(np.nansum(lg * pos * wg) / np.nansum(pos * wg))
            spread = float(np.sqrt(np.nansum(wg * pos * (lg - centroid) ** 2) / np.nansum(wg * pos)))
        else:
            absw = np.abs(rg)
            centroid = float(np.nansum(lg * absw * wg) / np.nansum(absw * wg)) if np.nansum(absw * wg) > EPS else np.nan
            spread = float(np.sqrt(np.nansum(wg * absw * (lg - centroid) ** 2) / np.nansum(wg * absw))) if np.nansum(absw * wg) > EPS else np.nan
        north = (lg >= 35.0) & (lg <= 45.0)
        south = (lg >= 25.0) & (lg < 35.0)
        north_mean = float(np.nanmean(rg[north])) if np.any(north) else np.nan
        south_mean = float(np.nanmean(rg[south])) if np.any(south) else np.nan
        sd = np.nanstd(rg)
        skew = float(np.nanmean(((rg - np.nanmean(rg)) / sd) ** 3)) if np.isfinite(sd) and sd > EPS else np.nan
        rows.append(
            {
                "day": int(day),
                "date_mmdd": _day_to_month_day(day),
                "feature_valid": True,
                "strength_mean": mean,
                "strength_max": float(np.nanmax(rg)),
                "strength_min": float(np.nanmin(rg)),
                "amplitude": amplitude,
                "weighted_l2_norm": l2,
                "centered_l2_norm": centered_norm,
                "axis_lat": float(lg[max_idx]),
                "min_lat": float(lg[min_idx]),
                "centroid_lat": centroid,
                "spread": spread,
                "width_proxy": spread,
                "north_mean_35_45": north_mean,
                "south_mean_25_35": south_mean,
                "NS_contrast": north_mean - south_mean if np.isfinite(north_mean) and np.isfinite(south_mean) else np.nan,
                "skewness": skew,
            }
        )
    return pd.DataFrame(rows)


def _norm_audit(raw_cube: np.ndarray, shape_norms: np.ndarray, means: np.ndarray, stds: np.ndarray, lat_grid: np.ndarray) -> pd.DataFrame:
    raw_clim = _climatology(raw_cube)
    w = _lat_weights(lat_grid)
    rows = []
    clim_norms = []
    for day in range(raw_clim.shape[0]):
        if day < AUDIT_RANGE[0] or day > AUDIT_RANGE[1]:
            continue
        row = raw_clim[day, :]
        mu = _weighted_mean_1d(row, w)
        centered = row - mu
        clim_norm = _weighted_norm_1d(centered, w)
        clim_norms.append(clim_norm)
    finite_norms = np.array([x for x in clim_norms if np.isfinite(x)], dtype=float)
    q10 = float(np.nanquantile(finite_norms, 0.10)) if finite_norms.size else np.nan
    q50 = float(np.nanquantile(finite_norms, 0.50)) if finite_norms.size else np.nan
    q90 = float(np.nanquantile(finite_norms, 0.90)) if finite_norms.size else np.nan
    for day in range(raw_clim.shape[0]):
        if day < AUDIT_RANGE[0] or day > AUDIT_RANGE[1]:
            continue
        row = raw_clim[day, :]
        mu = _weighted_mean_1d(row, w)
        centered = row - mu
        clim_norm = _weighted_norm_1d(centered, w)
        qrank = float(np.mean(finite_norms <= clim_norm)) if np.isfinite(clim_norm) and finite_norms.size else np.nan
        year_norm_vals = shape_norms[:, day] if day < shape_norms.shape[1] else np.array([])
        rows.append(
            {
                "day": int(day),
                "date_mmdd": _day_to_month_day(day),
                "raw_weighted_mean_clim": mu,
                "raw_weighted_std_clim": float(np.nanstd(row)),
                "raw_weighted_l2_norm_clim": _weighted_norm_1d(row, w),
                "shape_norm_denominator_clim": clim_norm,
                "shape_norm_denominator_year_median": float(np.nanmedian(year_norm_vals)) if year_norm_vals.size else np.nan,
                "shape_norm_qrank_clim": qrank,
                "shape_norm_q10_day0_70": q10,
                "shape_norm_q50_day0_70": q50,
                "shape_norm_q90_day0_70": q90,
                "shape_valid_flag": bool(np.isfinite(clim_norm) and clim_norm > EPS),
                "is_low_norm_warning": bool(np.isfinite(clim_norm) and np.isfinite(q10) and clim_norm <= q10),
                "is_day33_neighbor": bool(30 <= day <= 36),
                "is_day46_neighbor": bool(43 <= day <= 49),
            }
        )
    return pd.DataFrame(rows)


def _mean_days(mat: np.ndarray, day_window: tuple[int, int]) -> np.ndarray:
    lo, hi = day_window
    lo = max(int(lo), 0)
    hi = min(int(hi), mat.shape[0] - 1)
    return _safe_nanmean(mat[lo : hi + 1, :], axis=0)


def _before_after_tables(raw_clim: np.ndarray, shape_clim: np.ndarray, lat_grid: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = [
        ("day33_shape_peak", DAY33_BEFORE, DAY33_AFTER),
        ("day46_raw_peak", DAY46_BEFORE, DAY46_AFTER),
    ]
    raw_rows = []
    shape_rows = []
    metric_rows = []
    validity_rows = []
    w = _lat_weights(lat_grid)
    for event, before_win, after_win in events:
        for label, win in (("before", before_win), ("after", after_win)):
            validity_rows.append(
                {
                    "event": event,
                    "phase": label,
                    "start_day": win[0],
                    "end_day": win[1],
                    "n_days_requested": int(win[1] - win[0] + 1),
                    "within_raw_range": bool(win[0] >= 0 and win[1] < raw_clim.shape[0]),
                }
            )
        rb = _mean_days(raw_clim, before_win)
        ra = _mean_days(raw_clim, after_win)
        sb = _mean_days(shape_clim, before_win)
        sa = _mean_days(shape_clim, after_win)
        for phase, arr in (("before", rb), ("after", ra), ("after_minus_before", ra - rb)):
            for lat, val in zip(lat_grid, arr):
                raw_rows.append({"event": event, "phase": phase, "lat": float(lat), "profile_value": float(val) if np.isfinite(val) else np.nan})
        for phase, arr in (("before", sb), ("after", sa), ("after_minus_before", sa - sb)):
            for lat, val in zip(lat_grid, arr):
                shape_rows.append({"event": event, "phase": phase, "lat": float(lat), "profile_value": float(val) if np.isfinite(val) else np.nan})
        raw_l2 = _weighted_distance(ra, rb, w)
        shape_l2 = _weighted_distance(sa, sb, w)
        raw_mean_change = _weighted_mean_1d(ra, w) - _weighted_mean_1d(rb, w)
        raw_amp_change = (np.nanmax(ra) - np.nanmin(ra)) - (np.nanmax(rb) - np.nanmin(rb))
        raw_axis_change = float(lat_grid[int(np.nanargmax(ra))] - lat_grid[int(np.nanargmax(rb))]) if np.any(np.isfinite(ra)) and np.any(np.isfinite(rb)) else np.nan
        shape_axis_change = float(lat_grid[int(np.nanargmax(sa))] - lat_grid[int(np.nanargmax(sb))]) if np.any(np.isfinite(sa)) and np.any(np.isfinite(sb)) else np.nan
        corr_shape = _weighted_corr(sa, sb, w)
        # Soft dominant-change label. Ratios are descriptive, not inferential.
        if not np.isfinite(raw_l2) or not np.isfinite(shape_l2):
            dom = "weak_or_invalid"
        elif shape_l2 > raw_l2 * 1.15 and abs(raw_amp_change) < np.nanstd(raw_clim) * 0.20:
            dom = "shape_dominated"
        elif abs(raw_amp_change) > np.nanstd(raw_clim) * 0.35 or raw_l2 > shape_l2 * 1.25:
            dom = "amplitude_or_raw_state_dominated"
        elif abs(raw_axis_change) >= 2.0 or abs(shape_axis_change) >= 2.0:
            dom = "axis_shift_dominated"
        else:
            dom = "mixed"
        metric_rows.append(
            {
                "event": event,
                "before_start": before_win[0],
                "before_end": before_win[1],
                "after_start": after_win[0],
                "after_end": after_win[1],
                "raw_l2_change": raw_l2,
                "raw_mean_change": raw_mean_change,
                "raw_amplitude_change": raw_amp_change,
                "raw_axis_lat_change": raw_axis_change,
                "shape_l2_change": shape_l2,
                "shape_corr_before_after": corr_shape,
                "shape_axis_lat_change": shape_axis_change,
                "dominant_change_type": dom,
            }
        )
    return pd.DataFrame(raw_rows), pd.DataFrame(shape_rows), pd.DataFrame(metric_rows), pd.DataFrame(validity_rows)


# -----------------------------------------------------------------------------
# V7-z result loading and detector-score/peak summaries
# -----------------------------------------------------------------------------


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_v7z_outputs(result_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "profile_registry": "W45_multi_object_profile_window_registry_v7_z.csv",
        "shape_registry": "W45_multi_object_shape_pattern_window_registry_v7_z.csv",
        "score_profile": "W45_multi_object_detector_score_profile_v7_z.csv",
        "return_days": "W45_multi_object_peak_return_day_distribution_v7_z.csv",
        "main_selection": "W45_multi_object_main_window_selection_v7_z_hotfix_01.csv",
    }
    return {name: _safe_read_csv(result_dir / fname) for name, fname in files.items()}


def _je_peak_reference_tables(v7z: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for key, detector_type in (("profile_registry", "raw_profile"), ("shape_registry", "shape_pattern")):
        df = v7z.get(key, pd.DataFrame())
        if df.empty:
            continue
        sub = df[(df.get("object") == "Je") & (df.get("detector_type") == detector_type)].copy()
        for _, r in sub.iterrows():
            rows.append({**r.to_dict(), "source_table": key})
    peak_df = pd.DataFrame(rows)

    selection = v7z.get("main_selection", pd.DataFrame())
    if not selection.empty:
        sel = selection[selection.get("object") == "Je"].copy()
    else:
        sel = pd.DataFrame()
    return peak_df, sel


def _detector_score_comparison(v7z: dict[str, pd.DataFrame]) -> pd.DataFrame:
    score = v7z.get("score_profile", pd.DataFrame())
    if score.empty:
        return pd.DataFrame(columns=["day", "raw_profile_detector_score", "shape_pattern_detector_score", "raw_peak_flag", "shape_peak_flag", "system_W45_flag"])
    sub = score[score.get("object") == "Je"].copy()
    if sub.empty:
        return pd.DataFrame(columns=["day", "raw_profile_detector_score", "shape_pattern_detector_score", "raw_peak_flag", "shape_peak_flag", "system_W45_flag"])
    piv = sub.pivot_table(index="day", columns="detector_type", values="detector_score", aggfunc="first").reset_index()
    piv = piv.rename(columns={"raw_profile": "raw_profile_detector_score", "shape_pattern": "shape_pattern_detector_score"})
    if "raw_profile_detector_score" not in piv.columns:
        piv["raw_profile_detector_score"] = np.nan
    if "shape_pattern_detector_score" not in piv.columns:
        piv["shape_pattern_detector_score"] = np.nan
    piv["raw_peak_flag"] = piv["day"].astype(int).eq(JE_RAW_REFERENCE_DAY)
    piv["shape_peak_flag"] = piv["day"].astype(int).eq(JE_SHAPE_REFERENCE_DAY)
    piv["system_W45_flag"] = piv["day"].between(SYSTEM_W45[0], SYSTEM_W45[1])
    return piv.sort_values("day")


def _bootstrap_peak_separation(v7z: dict[str, pd.DataFrame], selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dist = v7z.get("return_days", pd.DataFrame())
    if dist.empty:
        empty = pd.DataFrame()
        return empty, pd.DataFrame([{"status": "unavailable_from_existing_outputs", "reason": "return_day_distribution_missing"}])

    def _selected_candidate(detector_type: str, fallback_day: int) -> Optional[str]:
        if selection is not None and not selection.empty:
            sub = selection[(selection.get("object") == "Je") & (selection.get("detector_type") == detector_type)]
            if not sub.empty and "selected_candidate_id" in sub.columns:
                return str(sub.iloc[0]["selected_candidate_id"])
        sub = dist[(dist.get("object") == "Je") & (dist.get("detector_type") == detector_type) & (dist.get("observed_peak_day") == fallback_day)]
        if not sub.empty:
            return str(sub.iloc[0]["candidate_id"])
        sub = dist[(dist.get("object") == "Je") & (dist.get("detector_type") == detector_type)]
        return str(sub.iloc[0]["candidate_id"]) if not sub.empty else None

    raw_cand = _selected_candidate("raw_profile", JE_RAW_REFERENCE_DAY)
    shp_cand = _selected_candidate("shape_pattern", JE_SHAPE_REFERENCE_DAY)
    if raw_cand is None or shp_cand is None:
        return pd.DataFrame(), pd.DataFrame([{"status": "unavailable_from_existing_outputs", "reason": "selected_candidate_missing", "raw_candidate": raw_cand, "shape_candidate": shp_cand}])

    raw = dist[(dist.get("object") == "Je") & (dist.get("detector_type") == "raw_profile") & (dist.get("candidate_id") == raw_cand)][["bootstrap_id", "matched_return_day", "matched"]].copy()
    shp = dist[(dist.get("object") == "Je") & (dist.get("detector_type") == "shape_pattern") & (dist.get("candidate_id") == shp_cand)][["bootstrap_id", "matched_return_day", "matched"]].copy()
    raw = raw.rename(columns={"matched_return_day": "raw_profile_return_day", "matched": "raw_matched"})
    shp = shp.rename(columns={"matched_return_day": "shape_pattern_return_day", "matched": "shape_matched"})
    merged = pd.merge(raw, shp, on="bootstrap_id", how="inner")
    merged["delta_raw_minus_shape"] = merged["raw_profile_return_day"] - merged["shape_pattern_return_day"]
    merged["both_matched"] = merged["raw_matched"].astype(bool) & merged["shape_matched"].astype(bool)
    valid = merged[merged["both_matched"] & np.isfinite(merged["delta_raw_minus_shape"])]
    if valid.empty:
        summary = pd.DataFrame([{"status": "unavailable_from_existing_outputs", "reason": "no_joint_matched_bootstrap", "raw_candidate": raw_cand, "shape_candidate": shp_cand}])
    else:
        delta = valid["delta_raw_minus_shape"].astype(float).to_numpy()
        summary = pd.DataFrame(
            [
                {
                    "status": "available",
                    "raw_candidate": raw_cand,
                    "shape_candidate": shp_cand,
                    "n_joint_matches": int(valid.shape[0]),
                    "raw_median": float(np.nanmedian(valid["raw_profile_return_day"])),
                    "raw_q025": float(np.nanquantile(valid["raw_profile_return_day"], 0.025)),
                    "raw_q975": float(np.nanquantile(valid["raw_profile_return_day"], 0.975)),
                    "shape_median": float(np.nanmedian(valid["shape_pattern_return_day"])),
                    "shape_q025": float(np.nanquantile(valid["shape_pattern_return_day"], 0.025)),
                    "shape_q975": float(np.nanquantile(valid["shape_pattern_return_day"], 0.975)),
                    "delta_median": float(np.nanmedian(delta)),
                    "delta_q025": float(np.nanquantile(delta, 0.025)),
                    "delta_q975": float(np.nanquantile(delta, 0.975)),
                    "P_raw_later_than_shape": float(np.nanmean(delta > 0)),
                    "decision": "raw_later_than_shape_supported" if np.nanquantile(delta, 0.025) > 0 else ("raw_later_than_shape_tendency" if np.nanmean(delta > 0) >= 0.80 else "unresolved"),
                }
            ]
        )
    return merged, summary


# -----------------------------------------------------------------------------
# Decision and summary
# -----------------------------------------------------------------------------


def _decision_table(peak_df: pd.DataFrame, selection_df: pd.DataFrame, norm_df: pd.DataFrame, diff_metrics: pd.DataFrame, sep_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    shape_peak = peak_df[(peak_df.get("detector_type") == "shape_pattern") & (peak_df.get("peak_day") == JE_SHAPE_REFERENCE_DAY)] if not peak_df.empty else pd.DataFrame()
    raw_peak = peak_df[(peak_df.get("detector_type") == "raw_profile") & (peak_df.get("peak_day") == JE_RAW_REFERENCE_DAY)] if not peak_df.empty else pd.DataFrame()
    shape_support = float(shape_peak.iloc[0].get("bootstrap_match_fraction", np.nan)) if not shape_peak.empty else np.nan
    raw_support = float(raw_peak.iloc[0].get("bootstrap_match_fraction", np.nan)) if not raw_peak.empty else np.nan
    norm_33 = norm_df[norm_df["day"].between(30, 36)]
    norm_low = bool(norm_33["is_low_norm_warning"].any()) if not norm_33.empty and "is_low_norm_warning" in norm_33 else True
    sep_decision = str(sep_summary.iloc[0].get("decision", "unavailable")) if not sep_summary.empty else "unavailable"
    d33 = diff_metrics[diff_metrics["event"] == "day33_shape_peak"]
    d46 = diff_metrics[diff_metrics["event"] == "day46_raw_peak"]
    d33_type = str(d33.iloc[0].get("dominant_change_type", "unavailable")) if not d33.empty else "unavailable"
    d46_type = str(d46.iloc[0].get("dominant_change_type", "unavailable")) if not d46.empty else "unavailable"

    rows.append(
        {
            "check_item": "shape_peak_reliability_day33",
            "result": f"support={shape_support:.3f}" if np.isfinite(shape_support) else "support_unavailable",
            "decision": "shape_peak_reliable" if np.isfinite(shape_support) and shape_support >= 0.95 and not norm_low else ("shape_peak_normalization_sensitive" if norm_low else "shape_peak_unresolved"),
            "evidence": "shape-pattern registry + norm audit",
            "risk": "low_norm_warning_near_day33" if norm_low else "none_detected_from_norm_audit",
        }
    )
    rows.append(
        {
            "check_item": "raw_profile_peak_reliability_day46",
            "result": f"support={raw_support:.3f}" if np.isfinite(raw_support) else "support_unavailable",
            "decision": "raw_peak_reliable" if np.isfinite(raw_support) and raw_support >= 0.95 else "raw_peak_unresolved",
            "evidence": "raw/profile registry",
            "risk": "none_detected" if np.isfinite(raw_support) and raw_support >= 0.95 else "support_below_accepted_or_unavailable",
        }
    )
    rows.append(
        {
            "check_item": "raw_vs_shape_peak_separation",
            "result": sep_decision,
            "decision": sep_decision,
            "evidence": "V7-z bootstrap return-day distribution" if sep_decision != "unavailable" else "return-day distribution unavailable or unmatched",
            "risk": "uses_existing_bootstrap_matches_not_recomputed_in_audit",
        }
    )
    rows.append(
        {
            "check_item": "day33_change_type",
            "result": d33_type,
            "decision": d33_type,
            "evidence": "before-after raw/shape profile metrics around day33",
            "risk": "soft_label_thresholds_descriptive_not_inferential",
        }
    )
    rows.append(
        {
            "check_item": "day46_change_type",
            "result": d46_type,
            "decision": d46_type,
            "evidence": "before-after raw/shape profile metrics around day46",
            "risk": "soft_label_thresholds_descriptive_not_inferential",
        }
    )

    if (np.isfinite(shape_support) and shape_support >= 0.95 and not norm_low and np.isfinite(raw_support) and raw_support >= 0.95 and sep_decision in {"raw_later_than_shape_supported", "raw_later_than_shape_tendency"}):
        final_type = "pattern_preconditioning_profile_late"
    elif norm_low:
        final_type = "shape_normalization_sensitive"
    elif sep_decision == "unresolved":
        final_type = "observed_split_but_bootstrap_separation_unresolved"
    else:
        final_type = "split_audit_incomplete_or_unresolved"
    rows.append(
        {
            "check_item": "final_Je_type",
            "result": final_type,
            "decision": final_type,
            "evidence": "combined norm + registry + before-after + bootstrap separation audit",
            "risk": "not_a_causality_or_pathway_test",
        }
    )
    return pd.DataFrame(rows)


def _write_summary(paths: Paths, decision_df: pd.DataFrame, peak_df: pd.DataFrame, norm_df: pd.DataFrame, diff_df: pd.DataFrame, sep_df: pd.DataFrame) -> None:
    def _get_decision(item: str) -> str:
        sub = decision_df[decision_df["check_item"] == item]
        return str(sub.iloc[0]["decision"]) if not sub.empty else "unavailable"

    final_type = _get_decision("final_Je_type")
    shape_reliability = _get_decision("shape_peak_reliability_day33")
    raw_reliability = _get_decision("raw_profile_peak_reliability_day46")
    separation = _get_decision("raw_vs_shape_peak_separation")
    lines = []
    lines.append("# Je Layer-Split Audit for W45")
    lines.append("")
    lines.append(f"Version: `{VERSION}`")
    lines.append(f"Generated: `{_now_iso()}`")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("Audit whether Je's early shape-pattern peak around day33 and late raw/profile peak around day46 represent a credible layer split or a normalization/detector artifact.")
    lines.append("")
    lines.append("## 2. Main decision")
    lines.append(f"- Final Je type: `{final_type}`")
    lines.append(f"- Shape peak reliability: `{shape_reliability}`")
    lines.append(f"- Raw/profile peak reliability: `{raw_reliability}`")
    lines.append(f"- Raw-vs-shape peak separation: `{separation}`")
    lines.append("")
    lines.append("## 3. Peak references from V7-z")
    if peak_df.empty:
        lines.append("- V7-z registry peaks were not available.")
    else:
        je_rows = peak_df.sort_values(["detector_type", "peak_rank" if "peak_rank" in peak_df.columns else "peak_day"])
        for _, r in je_rows.iterrows():
            lines.append(
                f"- {r.get('detector_type')}: {r.get('candidate_id')} peak day {r.get('peak_day')}, "
                f"window {r.get('window_start')}-{r.get('window_end')}, support {r.get('bootstrap_match_fraction')}, class {r.get('support_class')}"
            )
    lines.append("")
    lines.append("## 4. Normalization audit")
    day33 = norm_df[norm_df["day"].between(30, 36)] if not norm_df.empty else pd.DataFrame()
    if day33.empty:
        lines.append("- Normalization audit unavailable.")
    else:
        low_days = day33[day33["is_low_norm_warning"] == True]["day"].tolist() if "is_low_norm_warning" in day33 else []
        min_rank = day33["shape_norm_qrank_clim"].min() if "shape_norm_qrank_clim" in day33 else np.nan
        lines.append(f"- Day30–36 low-norm warning days: {low_days}")
        lines.append(f"- Minimum day30–36 norm quantile rank: {min_rank:.3f}" if np.isfinite(min_rank) else "- Norm quantile rank unavailable.")
    lines.append("")
    lines.append("## 5. Before/after profile metrics")
    if diff_df.empty:
        lines.append("- Before/after metrics unavailable.")
    else:
        for _, r in diff_df.iterrows():
            lines.append(
                f"- {r['event']}: dominant={r['dominant_change_type']}, raw_l2={r['raw_l2_change']:.4g}, "
                f"shape_l2={r['shape_l2_change']:.4g}, raw_amp_change={r['raw_amplitude_change']:.4g}, "
                f"raw_axis_change={r['raw_axis_lat_change']:.3g}, shape_axis_change={r['shape_axis_lat_change']:.3g}"
            )
    lines.append("")
    lines.append("## 6. Bootstrap separation")
    if sep_df.empty:
        lines.append("- Bootstrap separation summary unavailable.")
    else:
        r = sep_df.iloc[0]
        lines.append("- " + "; ".join([f"{c}={r[c]}" for c in sep_df.columns if c in {"status", "delta_median", "delta_q025", "delta_q975", "P_raw_later_than_shape", "decision"}]))
    lines.append("")
    lines.append("## 7. Allowed statement")
    if final_type == "pattern_preconditioning_profile_late":
        lines.append("Je can be described as a layer-split object: shape-pattern adjustment appears before the raw/profile main adjustment.")
    elif final_type == "shape_normalization_sensitive":
        lines.append("Je's early shape-pattern signal should be treated as normalization-sensitive and should not be upgraded to a stable pattern-preconditioning result.")
    else:
        lines.append("Je shows an observed raw/shape split, but the audit did not fully support upgrading it to a stable layer-split conclusion.")
    lines.append("")
    lines.append("## 8. Forbidden statement")
    lines.append("- Do not write that Je overall leads all objects.")
    lines.append("- Do not write that Je drives P/V/H/Jw.")
    lines.append("- Do not collapse shape-pattern timing and raw/profile timing into one transition day.")
    _write_text("\n".join(lines) + "\n", paths.output_dir / "Je_layer_split_audit_summary_v7_z_a.md")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def _maybe_plot(paths: Paths, raw_clim: np.ndarray, shape_clim: np.ndarray, lat_grid: np.ndarray, norm_df: pd.DataFrame, score_df: pd.DataFrame, feature_df: pd.DataFrame, raw_ba: pd.DataFrame, shape_ba: pd.DataFrame, cfg: JeAuditConfig) -> None:
    if not cfg.write_figures:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _ensure_dir(paths.figure_dir)

    # Detector scores
    if not score_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(score_df["day"], score_df["raw_profile_detector_score"], label="raw/profile score")
        ax.plot(score_df["day"], score_df["shape_pattern_detector_score"], label="shape-pattern score")
        ax.axvspan(SYSTEM_W45[0], SYSTEM_W45[1], alpha=0.15, label="system W45")
        ax.axvline(JE_SHAPE_REFERENCE_DAY, linestyle="--", label="shape day33")
        ax.axvline(JE_RAW_REFERENCE_DAY, linestyle=":", label="raw day46")
        ax.set_xlabel("day")
        ax.set_ylabel("detector score")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "Je_raw_vs_shape_detector_scores_v7_z_a.png", dpi=160)
        plt.close(fig)

    # Norm audit
    if not norm_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(norm_df["day"], norm_df["shape_norm_denominator_clim"], label="shape norm denominator")
        for q, lab in (("shape_norm_q10_day0_70", "q10"), ("shape_norm_q50_day0_70", "q50"), ("shape_norm_q90_day0_70", "q90")):
            if q in norm_df and norm_df[q].notna().any():
                ax.axhline(norm_df[q].dropna().iloc[0], linestyle="--", label=lab)
        ax.axvline(JE_SHAPE_REFERENCE_DAY, linestyle="--")
        ax.axvline(JE_RAW_REFERENCE_DAY, linestyle=":")
        ax.set_xlabel("day")
        ax.set_ylabel("norm")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "Je_shape_norm_audit_v7_z_a.png", dpi=160)
        plt.close(fig)

    # Heatmaps
    days = np.arange(raw_clim.shape[0])
    mask = (days >= AUDIT_RANGE[0]) & (days <= AUDIT_RANGE[1])
    for mat, fname, title in ((raw_clim, "Je_raw_profile_heatmap_v7_z_a.png", "Je raw profile"), (shape_clim, "Je_shape_profile_heatmap_v7_z_a.png", "Je shape-normalized profile")):
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(mat[mask, :].T, aspect="auto", origin="lower", extent=[AUDIT_RANGE[0], AUDIT_RANGE[1], float(lat_grid[0]), float(lat_grid[-1])])
        ax.axvspan(SYSTEM_W45[0], SYSTEM_W45[1], alpha=0.15)
        ax.axvline(JE_SHAPE_REFERENCE_DAY, linestyle="--")
        ax.axvline(JE_RAW_REFERENCE_DAY, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("day")
        ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax, label="value")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / fname, dpi=160)
        plt.close(fig)

    # Feature timeseries
    if not feature_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in ["amplitude", "axis_lat", "centroid_lat", "spread", "NS_contrast"]:
            if col in feature_df:
                vals = feature_df[col].astype(float)
                sd = vals.std(skipna=True)
                mu = vals.mean(skipna=True)
                if np.isfinite(sd) and sd > EPS:
                    ax.plot(feature_df["day"], (vals - mu) / sd, label=f"{col} (z)")
        ax.axvspan(SYSTEM_W45[0], SYSTEM_W45[1], alpha=0.15)
        ax.axvline(JE_SHAPE_REFERENCE_DAY, linestyle="--")
        ax.axvline(JE_RAW_REFERENCE_DAY, linestyle=":")
        ax.set_xlabel("day")
        ax.set_ylabel("standardized value")
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "Je_profile_feature_timeseries_v7_z_a.png", dpi=160)
        plt.close(fig)

    # Before-after profiles
    for ba_df, fname, title in ((raw_ba, "Je_before_after_raw_profiles_v7_z_a.png", "Je raw before/after"), (shape_ba, "Je_before_after_shape_profiles_v7_z_a.png", "Je shape before/after")):
        if ba_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for event in ba_df["event"].unique():
            for phase in ("before", "after"):
                sub = ba_df[(ba_df["event"] == event) & (ba_df["phase"] == phase)]
                ax.plot(sub["profile_value"], sub["lat"], label=f"{event} {phase}")
        ax.set_title(title)
        ax.set_xlabel("profile value")
        ax.set_ylabel("latitude")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(paths.figure_dir / fname, dpi=160)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def run_W45_Je_layer_split_audit_v7_z_a(v7_root: Optional[Path] = None) -> None:
    cfg = JeAuditConfig(
        write_figures=os.environ.get("V7Z_JE_AUDIT_SKIP_FIGURES", "0") not in {"1", "true", "TRUE", "yes", "YES"},
        save_by_year=os.environ.get("V7Z_JE_AUDIT_SAVE_BY_YEAR", "0") in {"1", "true", "TRUE", "yes", "YES"},
    )
    paths = _resolve_paths(v7_root, cfg)
    log_path = paths.log_dir / "run.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    log("[1/8] Load smoothed fields and build Je profile")
    files = _load_npz(paths.smoothed_fields_path)
    try:
        raw_cube, lat_grid, u_key, key_audit = _build_je_profile(files)
    except Exception:
        # Write key audit if possible, then re-raise.
        key, status = _resolve_key(files, U200_ALIASES)
        key_audit = pd.DataFrame([{"required": "u200", "resolved_key": key, "status": status, "aliases": ";".join(U200_ALIASES)}])
        _write_csv(key_audit, paths.output_dir / "Je_input_key_audit_v7_z_a.csv")
        raise
    _write_csv(key_audit, paths.output_dir / "Je_input_key_audit_v7_z_a.csv")
    raw_clim = _climatology(raw_cube)
    _write_csv(_profile_to_rows(raw_clim, lat_grid, "raw_profile_value"), paths.output_dir / "Je_raw_profile_climatology_v7_z_a.csv")
    if cfg.save_by_year:
        rows = []
        for yi in range(raw_cube.shape[0]):
            for day in range(max(0, AUDIT_RANGE[0]), min(raw_cube.shape[1] - 1, AUDIT_RANGE[1]) + 1):
                for lat, val in zip(lat_grid, raw_cube[yi, day, :]):
                    rows.append({"year_index": int(yi), "day": int(day), "lat": float(lat), "raw_profile_value": float(val) if np.isfinite(val) else np.nan})
        _write_csv(pd.DataFrame(rows), paths.output_dir / "Je_raw_profile_by_year_v7_z_a.csv")

    log("[2/8] Build shape-normalized Je profile and norm audit")
    shape_cube, means, norms, stds, valid = _shape_normalize_cube(raw_cube, lat_grid)
    shape_clim = _climatology(shape_cube)
    _write_csv(_profile_to_rows(shape_clim, lat_grid, "shape_profile_value"), paths.output_dir / "Je_shape_profile_climatology_v7_z_a.csv")
    norm_df = _norm_audit(raw_cube, norms, means, stds, lat_grid)
    _write_csv(norm_df, paths.output_dir / "Je_shape_normalization_norm_audit_v7_z_a.csv")

    log("[3/8] Compute Je profile feature time series")
    feature_df = _feature_timeseries(raw_clim, lat_grid)
    _write_csv(feature_df, paths.output_dir / "Je_profile_feature_timeseries_v7_z_a.csv")

    log("[4/8] Compute day33/day46 before-after profile diagnostics")
    raw_ba, shape_ba, diff_metrics, validity_df = _before_after_tables(raw_clim, shape_clim, lat_grid)
    _write_csv(raw_ba, paths.output_dir / "Je_before_after_raw_profiles_v7_z_a.csv")
    _write_csv(shape_ba, paths.output_dir / "Je_before_after_shape_profiles_v7_z_a.csv")
    _write_csv(diff_metrics, paths.output_dir / "Je_before_after_difference_metrics_v7_z_a.csv")
    _write_csv(validity_df, paths.output_dir / "Je_window_validity_audit_v7_z_a.csv")

    log("[5/8] Load V7-z detector and bootstrap outputs")
    v7z = _load_v7z_outputs(paths.v7z_result_dir)
    peak_df, selection_df = _je_peak_reference_tables(v7z)
    score_df = _detector_score_comparison(v7z)
    sep_detail, sep_summary = _bootstrap_peak_separation(v7z, selection_df)
    _write_csv(peak_df, paths.output_dir / "Je_v7z_peak_reference_table_v7_z_a.csv")
    _write_csv(selection_df, paths.output_dir / "Je_v7z_main_window_selection_v7_z_a.csv")
    _write_csv(score_df, paths.output_dir / "Je_raw_vs_shape_detector_scores_v7_z_a.csv")
    _write_csv(sep_detail, paths.output_dir / "Je_bootstrap_peak_separation_v7_z_a.csv")
    _write_csv(sep_summary, paths.output_dir / "Je_bootstrap_peak_separation_summary_v7_z_a.csv")

    log("[6/8] Classify Je layer split")
    decision_df = _decision_table(peak_df, selection_df, norm_df, diff_metrics, sep_summary)
    _write_csv(decision_df, paths.output_dir / "Je_layer_split_audit_decision_v7_z_a.csv")

    log("[7/8] Write figures and markdown summary")
    _maybe_plot(paths, raw_clim, shape_clim, lat_grid, norm_df, score_df, feature_df, raw_ba, shape_ba, cfg)
    _write_summary(paths, decision_df, peak_df, norm_df, diff_metrics, sep_summary)

    log("[8/8] Write run metadata")
    _write_json(
        {
            "version": VERSION,
            "output_tag": OUTPUT_TAG,
            "created_at": _now_iso(),
            "project_root": str(paths.project_root),
            "v7_root": str(paths.v7_root),
            "smoothed_fields_path": str(paths.smoothed_fields_path),
            "v7z_result_dir": str(paths.v7z_result_dir),
            "u200_key": u_key,
            "Je_lon_range": JE_LON_RANGE,
            "Je_lat_range": JE_LAT_RANGE,
            "lat_step_deg": LAT_STEP_DEG,
            "system_W45": SYSTEM_W45,
            "audit_range": AUDIT_RANGE,
            "day33_before": DAY33_BEFORE,
            "day33_after": DAY33_AFTER,
            "day46_before": DAY46_BEFORE,
            "day46_after": DAY46_AFTER,
            "write_figures": cfg.write_figures,
            "save_by_year": cfg.save_by_year,
            "outputs": {
                "decision": "Je_layer_split_audit_decision_v7_z_a.csv",
                "summary": "Je_layer_split_audit_summary_v7_z_a.md",
            },
        },
        paths.output_dir / "run_meta.json",
    )
    log(f"Done. Outputs written to: {paths.output_dir}")


__all__ = ["run_W45_Je_layer_split_audit_v7_z_a"]
