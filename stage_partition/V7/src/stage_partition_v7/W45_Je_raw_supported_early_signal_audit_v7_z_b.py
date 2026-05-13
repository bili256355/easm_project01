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

VERSION = "v7_z_je_audit_b"
OUTPUT_TAG = "W45_Je_raw_supported_early_signal_audit_v7_z_b"
EPS = 1.0e-12

JE_LON_RANGE = (120.0, 150.0)
JE_LAT_RANGE = (25.0, 45.0)
LAT_STEP_DEG = 2.0
U200_ALIASES = ("u200_smoothed", "u200", "Je_smoothed", "Je")

FULL_RANGE = (0, 70)
EARLY_CANDIDATE_RANGE = (20, 36)
EARLY_CORE_RANGE = (26, 33)
LATE_MAIN_RANGE = (40, 52)
SYSTEM_W45 = (40, 48)

EVENT_WINDOWS = {
    "day26_raw_early_candidate": {"center_day": 26, "before": (20, 25), "after": (27, 32)},
    "day33_shape_sensitive_candidate": {"center_day": 33, "before": (27, 32), "after": (34, 39)},
    "day46_raw_main_peak": {"center_day": 46, "before": (40, 45), "after": (47, 52)},
}


@dataclass(frozen=True)
class JeRawSupportAuditConfig:
    project_root: str = r"D:\easm_project01"
    smoothed_fields_relpath: str = r"foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz"
    v7z_result_relpath: str = r"stage_partition\V7\outputs\W45_multi_object_prepost_stat_validation_v7_z"
    je_audit_a_relpath: str = r"stage_partition\V7\outputs\W45_Je_layer_split_audit_v7_z_a"
    output_tag: str = OUTPUT_TAG
    n_bootstrap: int = 300
    random_seed: int = 42
    detector_half_width: int = 5
    local_peak_min_distance_days: int = 3
    write_figures: bool = True
    save_bootstrap_samples: bool = True


@dataclass
class Paths:
    v7_root: Path
    project_root: Path
    smoothed_fields_path: Path
    v7z_result_dir: Path
    je_audit_a_dir: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path


# -----------------------------------------------------------------------------
# IO and small utilities
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


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _safe_nanmean(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _resolve_paths(v7_root: Optional[Path], cfg: JeRawSupportAuditConfig) -> Paths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()

    project_root = Path(os.environ.get("V7Z_PROJECT_ROOT", cfg.project_root))
    if not project_root.exists():
        project_root = v7_root.parents[1]

    smoothed_path = Path(os.environ.get("V7Z_SMOOTHED_FIELDS", "")) if os.environ.get("V7Z_SMOOTHED_FIELDS") else project_root / cfg.smoothed_fields_relpath
    v7z_result_dir = Path(os.environ.get("V7Z_RESULT_DIR", "")) if os.environ.get("V7Z_RESULT_DIR") else project_root / cfg.v7z_result_relpath
    je_audit_a_dir = Path(os.environ.get("V7Z_JE_AUDIT_A_DIR", "")) if os.environ.get("V7Z_JE_AUDIT_A_DIR") else project_root / cfg.je_audit_a_relpath

    output_dir = v7_root / "outputs" / cfg.output_tag
    log_dir = v7_root / "logs" / cfg.output_tag
    figure_dir = output_dir / "figures"
    for p in (output_dir, log_dir, figure_dir):
        _ensure_dir(p)
    return Paths(v7_root, project_root, smoothed_path, v7z_result_dir, je_audit_a_dir, output_dir, log_dir, figure_dir)


def _day_to_month_day(day: int) -> str:
    d = int(day)
    for month, length in [(4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]:
        if d < length:
            return f"{month:02d}-{d + 1:02d}"
        d -= length
    return f"day{int(day)}"


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


def _weighted_distance(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return _weighted_norm_1d(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), w)


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


# -----------------------------------------------------------------------------
# Profile construction
# -----------------------------------------------------------------------------


def _standardize_field_dims(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3 and arr.shape[-2] == len(lat) and arr.shape[-1] == len(lon):
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
        raise KeyError("Missing lat/lon/u200 keys. See Je_raw_supported_input_audit_v7_z_b.csv.")
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
    profile = _safe_nanmean(subset, axis=-1)  # years x days x selected lat
    src_lats = lat[lat_mask]
    dst_lats = np.arange(min(JE_LAT_RANGE), max(JE_LAT_RANGE) + 1e-9, LAT_STEP_DEG)
    profile2 = _interp_profile_to_grid(profile, src_lats, dst_lats)
    return profile2, dst_lats, str(u_key), audit


def _climatology(cube: np.ndarray) -> np.ndarray:
    return _safe_nanmean(cube, axis=0)


# -----------------------------------------------------------------------------
# Feature metrics and before-after metrics
# -----------------------------------------------------------------------------


def _profile_features_from_matrix(mat: np.ndarray, lat_grid: np.ndarray) -> pd.DataFrame:
    w = _lat_weights(lat_grid)
    rows = []
    for day in range(mat.shape[0]):
        if day < FULL_RANGE[0] or day > FULL_RANGE[1]:
            continue
        row = mat[day, :]
        good = np.isfinite(row) & np.isfinite(w)
        if good.sum() < 2:
            rows.append({"day": day, "feature_valid": False})
            continue
        rg = row[good]
        lg = lat_grid[good]
        wg = w[good]
        mean = _weighted_mean_1d(row, w)
        l2 = _weighted_norm_1d(row, w)
        max_idx = int(np.nanargmax(rg))
        min_idx = int(np.nanargmin(rg))
        amp = float(np.nanmax(rg) - np.nanmin(rg))
        pos = np.maximum(rg, 0.0)
        if np.nansum(pos * wg) > EPS:
            centroid = float(np.nansum(lg * pos * wg) / np.nansum(pos * wg))
            spread = float(np.sqrt(np.nansum(wg * pos * (lg - centroid) ** 2) / np.nansum(wg * pos)))
        else:
            ab = np.abs(rg)
            centroid = float(np.nansum(lg * ab * wg) / np.nansum(ab * wg)) if np.nansum(ab * wg) > EPS else np.nan
            spread = float(np.sqrt(np.nansum(wg * ab * (lg - centroid) ** 2) / np.nansum(wg * ab))) if np.nansum(ab * wg) > EPS else np.nan
        north = (lg >= 35.0) & (lg <= 45.0)
        south = (lg >= 25.0) & (lg < 35.0)
        nmean = float(np.nanmean(rg[north])) if np.any(north) else np.nan
        smean = float(np.nanmean(rg[south])) if np.any(south) else np.nan
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
                "amplitude": amp,
                "weighted_l2_norm": l2,
                "axis_lat": float(lg[max_idx]),
                "min_lat": float(lg[min_idx]),
                "centroid_lat": centroid,
                "spread": spread,
                "width_proxy": spread,
                "north_mean_35_45": nmean,
                "south_mean_25_35": smean,
                "NS_contrast": nmean - smean if np.isfinite(nmean) and np.isfinite(smean) else np.nan,
                "skewness": skew,
            }
        )
    df = pd.DataFrame(rows)
    # Raw daily profile speed is a direct profile metric rather than a scalar feature derivative.
    speeds = []
    for day in df["day"].astype(int):
        if day + 1 < mat.shape[0]:
            speeds.append(_weighted_distance(mat[day + 1, :], mat[day, :], w))
        else:
            speeds.append(np.nan)
    df["raw_daily_speed"] = speeds
    return df


def _mean_days(mat: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    lo, hi = win
    lo = max(0, int(lo))
    hi = min(mat.shape[0] - 1, int(hi))
    return _safe_nanmean(mat[lo : hi + 1, :], axis=0)


def _features_from_profile(row: np.ndarray, lat_grid: np.ndarray) -> dict[str, float]:
    mat = np.asarray(row, dtype=float)[None, :]
    df = _profile_features_from_matrix(mat, lat_grid)
    if df.empty:
        return {}
    return {k: df.iloc[0][k] for k in df.columns if k not in ("day", "date_mmdd", "feature_valid", "raw_daily_speed")}


def _before_after_metrics_for_clim(clim: np.ndarray, lat_grid: np.ndarray) -> pd.DataFrame:
    w = _lat_weights(lat_grid)
    rows = []
    for event, spec in EVENT_WINDOWS.items():
        before = _mean_days(clim, spec["before"])
        after = _mean_days(clim, spec["after"])
        f_before = _features_from_profile(before, lat_grid)
        f_after = _features_from_profile(after, lat_grid)
        row = {
            "event": event,
            "center_day": spec["center_day"],
            "before_start": spec["before"][0],
            "before_end": spec["before"][1],
            "after_start": spec["after"][0],
            "after_end": spec["after"][1],
            "raw_l2_change": _weighted_distance(after, before, w),
            "profile_corr_before_after": _weighted_corr(after, before, w),
        }
        for key in ("strength_mean", "strength_max", "amplitude", "weighted_l2_norm", "axis_lat", "centroid_lat", "spread", "width_proxy", "NS_contrast", "skewness"):
            b = f_before.get(key, np.nan)
            a = f_after.get(key, np.nan)
            row[f"{key}_before"] = b
            row[f"{key}_after"] = a
            row[f"{key}_change"] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
            if key in ("axis_lat", "centroid_lat", "spread", "width_proxy", "NS_contrast"):
                row[f"{key}_abs_change"] = abs(row[f"{key}_change"]) if np.isfinite(row[f"{key}_change"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _before_after_profile_rows(clim: np.ndarray, lat_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for event, spec in EVENT_WINDOWS.items():
        before = _mean_days(clim, spec["before"])
        after = _mean_days(clim, spec["after"])
        for phase, arr in (("before", before), ("after", after), ("after_minus_before", after - before)):
            for lat, val in zip(lat_grid, arr):
                rows.append({"event": event, "phase": phase, "lat": float(lat), "profile_value": float(val) if np.isfinite(val) else np.nan})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Feature-peak audit and raw detector sensitivity
# -----------------------------------------------------------------------------


def _local_before_after_score_matrix(mat: np.ndarray, lat_grid: np.ndarray, scope: tuple[int, int], half_width: int = 5) -> pd.DataFrame:
    w = _lat_weights(lat_grid)
    lo, hi = scope
    rows = []
    for day in range(lo, hi + 1):
        b0, b1 = day - half_width, day - 1
        a0, a1 = day + 1, day + half_width
        if b0 < 0 or a1 >= mat.shape[0]:
            score = np.nan
        else:
            before = _mean_days(mat, (b0, b1))
            after = _mean_days(mat, (a0, a1))
            score = _weighted_distance(after, before, w)
        rows.append({"day": int(day), "detector_score": score, "half_width": int(half_width)})
    return pd.DataFrame(rows)


def _find_local_peaks(score_df: pd.DataFrame, min_distance: int = 3, top_n: int = 6) -> pd.DataFrame:
    if score_df.empty or "detector_score" not in score_df.columns:
        return pd.DataFrame()
    df = score_df.dropna(subset=["detector_score"]).copy()
    if df.empty:
        return pd.DataFrame()
    days = df["day"].to_numpy(dtype=int)
    scores = df["detector_score"].to_numpy(dtype=float)
    candidates = []
    for i, (d, s) in enumerate(zip(days, scores)):
        left = scores[i - 1] if i > 0 else -np.inf
        right = scores[i + 1] if i + 1 < len(scores) else -np.inf
        if s >= left and s >= right:
            candidates.append((int(d), float(s)))
    # Greedy distance pruning by score.
    candidates = sorted(candidates, key=lambda x: (-x[1], x[0]))
    chosen: list[tuple[int, float]] = []
    for d, s in candidates:
        if all(abs(d - cd) >= min_distance for cd, _ in chosen):
            chosen.append((d, s))
        if len(chosen) >= top_n:
            break
    chosen = sorted(chosen, key=lambda x: x[0])
    if not chosen:
        return pd.DataFrame()
    all_scores = scores[np.isfinite(scores)]
    rows = []
    for rank, (d, s) in enumerate(sorted(chosen, key=lambda x: -x[1]), start=1):
        prom = s - float(np.nanmedian(all_scores)) if all_scores.size else np.nan
        rows.append(
            {
                "candidate_id": f"CP{rank:03d}",
                "peak_day": int(d),
                "peak_score": float(s),
                "peak_prominence_vs_median": float(prom),
                "peak_rank": int(rank),
                "is_in_early_candidate_range": bool(EARLY_CANDIDATE_RANGE[0] <= d <= EARLY_CANDIDATE_RANGE[1]),
                "is_in_early_core_range": bool(EARLY_CORE_RANGE[0] <= d <= EARLY_CORE_RANGE[1]),
                "is_in_late_main_range": bool(LATE_MAIN_RANGE[0] <= d <= LATE_MAIN_RANGE[1]),
                "relation_to_day33": "near_day33" if abs(d - 33) <= 4 else "other",
                "relation_to_day46": "near_day46" if abs(d - 46) <= 4 else "other",
            }
        )
    return pd.DataFrame(rows)


def _detector_sensitivity(clim: np.ndarray, lat_grid: np.ndarray, cfg: JeRawSupportAuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = {
        "full_day0_70": FULL_RANGE,
        "early_only_day0_40": (0, 40),
        "late_excluded_day0_39": (0, 39),
    }
    score_rows = []
    peak_rows = []
    for name, scope in scopes.items():
        scores = _local_before_after_score_matrix(clim, lat_grid, scope, cfg.detector_half_width)
        scores["detector_scope"] = name
        peaks = _find_local_peaks(scores, cfg.local_peak_min_distance_days)
        if not peaks.empty:
            peaks["detector_scope"] = name
            # Soft observed-only support labels, because this is not the V7-z bootstrap detector.
            peaks["support_class_observed"] = np.where(peaks["peak_rank"] == 1, "observed_main", "observed_secondary")
            peak_rows.append(peaks)
        score_rows.append(scores)
    score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    peak_df = pd.concat(peak_rows, ignore_index=True) if peak_rows else pd.DataFrame()
    return score_df, peak_df


def _feature_peak_audit(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "strength_mean",
        "strength_max",
        "amplitude",
        "weighted_l2_norm",
        "axis_lat",
        "centroid_lat",
        "spread",
        "width_proxy",
        "NS_contrast",
        "skewness",
        "raw_daily_speed",
    ]
    rows = []
    fdf = feature_df.set_index("day")
    for feat in feature_cols:
        if feat not in fdf.columns:
            continue
        vals = fdf[feat].astype(float)
        speed = vals.diff().abs()
        early = speed.loc[(speed.index >= EARLY_CANDIDATE_RANGE[0]) & (speed.index <= EARLY_CANDIDATE_RANGE[1])]
        late = speed.loc[(speed.index >= LATE_MAIN_RANGE[0]) & (speed.index <= LATE_MAIN_RANGE[1])]
        full = speed.loc[(speed.index >= FULL_RANGE[0]) & (speed.index <= FULL_RANGE[1])]
        early_day = int(early.idxmax()) if early.notna().any() else np.nan
        late_day = int(late.idxmax()) if late.notna().any() else np.nan
        full_day = int(full.idxmax()) if full.notna().any() else np.nan
        early_score = float(early.max()) if early.notna().any() else np.nan
        late_score = float(late.max()) if late.notna().any() else np.nan
        ratio = early_score / late_score if np.isfinite(early_score) and np.isfinite(late_score) and abs(late_score) > EPS else np.nan
        if np.isfinite(ratio) and ratio >= 0.75:
            interp = "early_change_comparable_to_late"
        elif np.isfinite(ratio) and ratio >= 0.35:
            interp = "early_change_present_but_weaker_than_late"
        else:
            interp = "late_dominated_or_no_early_feature_peak"
        rows.append(
            {
                "feature_name": feat,
                "full_range_peak_day": full_day,
                "early_range_peak_day": early_day,
                "late_range_peak_day": late_day,
                "early_range_max_score": early_score,
                "late_range_max_score": late_score,
                "early_to_late_score_ratio": ratio,
                "is_early_peak_in_core_day26_33": bool(np.isfinite(early_day) and EARLY_CORE_RANGE[0] <= early_day <= EARLY_CORE_RANGE[1]),
                "interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


def _cumulative_change(clim: np.ndarray, lat_grid: np.ndarray) -> pd.DataFrame:
    w = _lat_weights(lat_grid)
    ref = _mean_days(clim, (20, 25))
    ref_feat = _features_from_profile(ref, lat_grid)
    rows = []
    for day in range(FULL_RANGE[0], FULL_RANGE[1] + 1):
        row = clim[day, :]
        feat = _features_from_profile(row, lat_grid)
        rows.append(
            {
                "day": int(day),
                "date_mmdd": _day_to_month_day(day),
                "cum_raw_change_from_day20_25": _weighted_distance(row, ref, w),
                "cum_axis_shift_abs": abs(feat.get("axis_lat", np.nan) - ref_feat.get("axis_lat", np.nan)) if np.isfinite(feat.get("axis_lat", np.nan)) and np.isfinite(ref_feat.get("axis_lat", np.nan)) else np.nan,
                "cum_centroid_shift_abs": abs(feat.get("centroid_lat", np.nan) - ref_feat.get("centroid_lat", np.nan)) if np.isfinite(feat.get("centroid_lat", np.nan)) and np.isfinite(ref_feat.get("centroid_lat", np.nan)) else np.nan,
                "cum_NS_contrast_change": feat.get("NS_contrast", np.nan) - ref_feat.get("NS_contrast", np.nan) if np.isfinite(feat.get("NS_contrast", np.nan)) and np.isfinite(ref_feat.get("NS_contrast", np.nan)) else np.nan,
                "cum_amplitude_change": feat.get("amplitude", np.nan) - ref_feat.get("amplitude", np.nan) if np.isfinite(feat.get("amplitude", np.nan)) and np.isfinite(ref_feat.get("amplitude", np.nan)) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"median": np.nan, "q05": np.nan, "q95": np.nan, "q025": np.nan, "q975": np.nan}
    return {
        "median": float(np.nanmedian(vals)),
        "q05": float(np.nanquantile(vals, 0.05)),
        "q95": float(np.nanquantile(vals, 0.95)),
        "q025": float(np.nanquantile(vals, 0.025)),
        "q975": float(np.nanquantile(vals, 0.975)),
    }


def _run_before_after_bootstrap(raw_cube: np.ndarray, lat_grid: np.ndarray, cfg: JeRawSupportAuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_years = raw_cube.shape[0]
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for b in range(cfg.n_bootstrap):
        idx = rng.integers(0, n_years, size=n_years)
        clim = _climatology(raw_cube[idx, :, :])
        metrics = _before_after_metrics_for_clim(clim, lat_grid)
        for _, r in metrics.iterrows():
            rows.append({"bootstrap_id": b, **r.to_dict()})
    samples = pd.DataFrame(rows)
    summary_rows = []
    if not samples.empty:
        metric_cols = [
            "raw_l2_change",
            "amplitude_change",
            "axis_lat_abs_change",
            "centroid_lat_abs_change",
            "width_proxy_abs_change",
            "spread_abs_change",
            "NS_contrast_change",
            "profile_corr_before_after",
        ]
        # Some columns may not exist if feature names changed; restrict to available columns.
        metric_cols = [c for c in metric_cols if c in samples.columns]
        for event, sub in samples.groupby("event"):
            for metric in metric_cols:
                vals = sub[metric].to_numpy(dtype=float)
                q = _quantile_summary(vals)
                pos = float(np.nanmean(vals > 0)) if np.isfinite(vals).any() else np.nan
                # raw_l2 and abs metrics are directionless and expected > 0 if any change exists.
                if metric in ("raw_l2_change", "axis_lat_abs_change", "centroid_lat_abs_change", "width_proxy_abs_change", "spread_abs_change"):
                    decision = "supported_positive_change" if np.isfinite(q["q025"]) and q["q025"] > 0 else "unresolved"
                elif metric == "profile_corr_before_after":
                    decision = "profile_changed" if np.isfinite(q["q975"]) and q["q975"] < 0.98 else "high_correlation_or_unresolved"
                else:
                    decision = "positive_tendency" if np.isfinite(pos) and pos >= 0.80 else ("negative_tendency" if np.isfinite(pos) and pos <= 0.20 else "unresolved")
                summary_rows.append(
                    {
                        "event": event,
                        "metric": metric,
                        "observed": np.nan,
                        **q,
                        "P_positive": pos,
                        "decision": decision,
                    }
                )
    return samples, pd.DataFrame(summary_rows)


def _run_detector_bootstrap(raw_cube: np.ndarray, lat_grid: np.ndarray, cfg: JeRawSupportAuditConfig) -> pd.DataFrame:
    n_years = raw_cube.shape[0]
    rng = np.random.default_rng(cfg.random_seed + 17)
    rows = []
    scopes = {
        "full_day0_70": FULL_RANGE,
        "early_only_day0_40": (0, 40),
        "late_excluded_day0_39": (0, 39),
    }
    for b in range(cfg.n_bootstrap):
        idx = rng.integers(0, n_years, size=n_years)
        clim = _climatology(raw_cube[idx, :, :])
        for scope_name, scope in scopes.items():
            scores = _local_before_after_score_matrix(clim, lat_grid, scope, cfg.detector_half_width)
            peaks = _find_local_peaks(scores, cfg.local_peak_min_distance_days, top_n=3)
            if peaks.empty:
                rows.append({"bootstrap_id": b, "detector_scope": scope_name, "main_peak_day": np.nan, "has_early_peak_day26_33": False, "has_day33_neighbor_peak": False, "has_late_peak_day40_52": False})
                continue
            rows.append(
                {
                    "bootstrap_id": b,
                    "detector_scope": scope_name,
                    "main_peak_day": int(peaks.sort_values("peak_rank").iloc[0]["peak_day"]),
                    "has_early_peak_day26_33": bool(peaks["peak_day"].between(26, 33).any()),
                    "has_day33_neighbor_peak": bool(peaks["peak_day"].between(29, 36).any()),
                    "has_late_peak_day40_52": bool(peaks["peak_day"].between(40, 52).any()),
                }
            )
    if not rows:
        return pd.DataFrame()
    boot = pd.DataFrame(rows)
    summary_rows = []
    for scope_name, sub in boot.groupby("detector_scope"):
        vals = sub["main_peak_day"].to_numpy(dtype=float)
        q = _quantile_summary(vals)
        summary_rows.append(
            {
                "detector_scope": scope_name,
                **q,
                "P_main_peak_in_early26_33": float(np.nanmean((vals >= 26) & (vals <= 33))) if np.isfinite(vals).any() else np.nan,
                "P_has_any_early_peak_day26_33": float(sub["has_early_peak_day26_33"].mean()),
                "P_has_day33_neighbor_peak": float(sub["has_day33_neighbor_peak"].mean()),
                "P_has_late_peak_day40_52": float(sub["has_late_peak_day40_52"].mean()),
                "decision": "early_raw_peak_reproduced" if float(sub["has_early_peak_day26_33"].mean()) >= 0.50 else "early_raw_peak_not_stably_reproduced",
            }
        )
    return pd.DataFrame(summary_rows)


# -----------------------------------------------------------------------------
# Input audits, decisions and summaries
# -----------------------------------------------------------------------------


def _input_audit(paths: Paths) -> pd.DataFrame:
    checks = [
        ("smoothed_fields", paths.smoothed_fields_path),
        ("v7z_result_dir", paths.v7z_result_dir),
        ("je_audit_a_dir", paths.je_audit_a_dir),
        ("je_audit_a_decision", paths.je_audit_a_dir / "Je_layer_split_audit_decision_v7_z_a.csv"),
        ("je_audit_a_norm", paths.je_audit_a_dir / "Je_shape_normalization_norm_audit_v7_z_a.csv"),
    ]
    return pd.DataFrame([{"input_name": name, "path": str(path), "exists": path.exists(), "status": "ok" if path.exists() else "missing"} for name, path in checks])


def _load_previous_audit_a(paths: Paths) -> dict[str, pd.DataFrame]:
    return {
        "decision": _safe_read_csv(paths.je_audit_a_dir / "Je_layer_split_audit_decision_v7_z_a.csv"),
        "norm": _safe_read_csv(paths.je_audit_a_dir / "Je_shape_normalization_norm_audit_v7_z_a.csv"),
        "diff": _safe_read_csv(paths.je_audit_a_dir / "Je_before_after_difference_metrics_v7_z_a.csv"),
    }


def _peak_vs_ramp_audit(cum_df: pd.DataFrame, detector_peaks: pd.DataFrame, feature_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    early_cum = cum_df[cum_df["day"].between(26, 33)]
    late_cum = cum_df[cum_df["day"].between(40, 52)]
    early_gain = float(early_cum["cum_raw_change_from_day20_25"].max() - early_cum["cum_raw_change_from_day20_25"].min()) if not early_cum.empty else np.nan
    late_gain = float(late_cum["cum_raw_change_from_day20_25"].max() - late_cum["cum_raw_change_from_day20_25"].min()) if not late_cum.empty else np.nan
    early_peak_present = False
    if not detector_peaks.empty:
        early_peak_present = bool(detector_peaks["peak_day"].between(26, 33).any())
    n_feature_early = int(feature_audit.get("is_early_peak_in_core_day26_33", pd.Series(dtype=bool)).sum()) if not feature_audit.empty else 0
    if early_peak_present and n_feature_early >= 2:
        classification = "weak_peak_like"
    elif np.isfinite(early_gain) and np.isfinite(late_gain) and early_gain > 0 and (early_gain / (late_gain + EPS)) >= 0.25:
        classification = "ramp_or_precursor_like"
    elif np.isfinite(late_gain) and late_gain > 0:
        classification = "late_peak_dominated"
    else:
        classification = "unresolved"
    rows.append(
        {
            "early_cumulative_gain_day26_33": early_gain,
            "late_cumulative_gain_day40_52": late_gain,
            "early_to_late_cumulative_gain_ratio": early_gain / late_gain if np.isfinite(early_gain) and np.isfinite(late_gain) and abs(late_gain) > EPS else np.nan,
            "early_detector_peak_present_day26_33": early_peak_present,
            "n_raw_features_with_peak_in_day26_33": n_feature_early,
            "classification": classification,
        }
    )
    return pd.DataFrame(rows)


def _final_decision(
    ba_summary: pd.DataFrame,
    detector_boot: pd.DataFrame,
    feature_audit: pd.DataFrame,
    peak_vs_ramp: pd.DataFrame,
    previous_audit: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    def _event_metric_decision(event: str, metric: str) -> str:
        if ba_summary.empty:
            return "unavailable"
        sub = ba_summary[(ba_summary["event"] == event) & (ba_summary["metric"] == metric)]
        if sub.empty:
            return "missing"
        return str(sub.iloc[0].get("decision", "missing"))

    day33_raw_change = _event_metric_decision("day33_shape_sensitive_candidate", "raw_l2_change")
    day26_raw_change = _event_metric_decision("day26_raw_early_candidate", "raw_l2_change")
    day46_raw_change = _event_metric_decision("day46_raw_main_peak", "raw_l2_change")
    early_raw_supported = day33_raw_change == "supported_positive_change" or day26_raw_change == "supported_positive_change"
    late_supported = day46_raw_change == "supported_positive_change"

    n_feature_early = int(feature_audit.get("is_early_peak_in_core_day26_33", pd.Series(dtype=bool)).sum()) if not feature_audit.empty else 0
    detector_early_support = False
    if not detector_boot.empty:
        early_scope = detector_boot[detector_boot["detector_scope"].isin(["early_only_day0_40", "late_excluded_day0_39"])]
        if not early_scope.empty:
            detector_early_support = bool((early_scope["P_has_any_early_peak_day26_33"] >= 0.50).any())
    ramp_class = peak_vs_ramp.iloc[0].get("classification", "unresolved") if not peak_vs_ramp.empty else "unresolved"

    norm_sensitive_previous = False
    prev_dec = previous_audit.get("decision", pd.DataFrame())
    if not prev_dec.empty and "decision" in prev_dec.columns:
        norm_sensitive_previous = bool(prev_dec["decision"].astype(str).str.contains("normalization_sensitive", case=False, na=False).any())

    if early_raw_supported and detector_early_support and n_feature_early >= 2 and ramp_class == "weak_peak_like":
        final_status = "raw_supported_weak_early_peak"
    elif early_raw_supported and ramp_class in ("ramp_or_precursor_like", "weak_peak_like") and n_feature_early >= 1:
        final_status = "raw_supported_early_ramp"
    elif early_raw_supported and late_supported and ramp_class == "late_peak_dominated":
        final_status = "late_peak_precursor_only"
    elif norm_sensitive_previous and not early_raw_supported:
        final_status = "normalization_sensitive_only"
    else:
        final_status = "unresolved"

    rows.extend(
        [
            {
                "check_item": "day26_or_day33_raw_before_after_support",
                "result": "supported" if early_raw_supported else "not_supported",
                "decision": "early_raw_change_supported" if early_raw_supported else "early_raw_change_not_supported",
                "evidence": f"day26 raw_l2={day26_raw_change}; day33 raw_l2={day33_raw_change}",
                "risk": "raw_l2 is directionless; compare with feature/detector checks",
            },
            {
                "check_item": "day46_raw_main_support",
                "result": "supported" if late_supported else "not_supported",
                "decision": "late_raw_main_supported" if late_supported else "late_raw_main_not_supported",
                "evidence": f"day46 raw_l2={day46_raw_change}",
                "risk": "none_for_raw_l2_directionless_change",
            },
            {
                "check_item": "raw_feature_early_support",
                "result": str(n_feature_early),
                "decision": "feature_early_support_present" if n_feature_early >= 1 else "feature_early_support_absent",
                "evidence": "number of direct raw features with peak in day26-33",
                "risk": "feature peaks are descriptive unless bootstrap-supported separately",
            },
            {
                "check_item": "early_window_detector_support",
                "result": str(detector_early_support),
                "decision": "early_detector_reproduced" if detector_early_support else "early_detector_not_stable",
                "evidence": "early-only / late-excluded raw detector bootstrap",
                "risk": "detector is local raw contrast, not the full V7-z detector",
            },
            {
                "check_item": "peak_vs_ramp_structure",
                "result": ramp_class,
                "decision": ramp_class,
                "evidence": "cumulative raw change + detector/feature peak audit",
                "risk": "classification is diagnostic, not a physical mechanism",
            },
            {
                "check_item": "previous_normalization_risk",
                "result": str(norm_sensitive_previous),
                "decision": "previous_audit_flagged_normalization_sensitivity" if norm_sensitive_previous else "previous_audit_no_normalization_warning_loaded",
                "evidence": "V7-z-je-audit-a decision table",
                "risk": "only available if audit-a outputs are present",
            },
            {
                "check_item": "final_Je_day26_33_status",
                "result": final_status,
                "decision": final_status,
                "evidence": "combined raw before-after + feature + detector + ramp + audit-a sensitivity",
                "risk": "does not alter V7-z final claims automatically",
            },
        ]
    )
    return pd.DataFrame(rows)


def _write_summary(paths: Paths, decision: pd.DataFrame, ba_summary: pd.DataFrame, feature_audit: pd.DataFrame, detector_boot: pd.DataFrame, peak_vs_ramp: pd.DataFrame) -> None:
    final = "unavailable"
    if not decision.empty:
        sub = decision[decision["check_item"] == "final_Je_day26_33_status"]
        if not sub.empty:
            final = str(sub.iloc[0]["decision"])
    lines = []
    lines.append("# Je raw-supported early signal audit v7_z_b")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Test whether Je day26–33 is supported by raw-profile evidence, or only by the normalization-sensitive shape-pattern detector.")
    lines.append("")
    lines.append("## Final decision")
    lines.append(f"- `Je_day26_33_status`: `{final}`")
    lines.append("")
    lines.append("## Key checks")
    for _, r in decision.iterrows():
        lines.append(f"- {r.get('check_item')}: `{r.get('decision')}` — {r.get('evidence')}")
    lines.append("")
    lines.append("## Before-after bootstrap summary")
    if ba_summary.empty:
        lines.append("- Not available.")
    else:
        for _, r in ba_summary.iterrows():
            if r.get("metric") == "raw_l2_change":
                lines.append(f"- {r.get('event')} raw_l2_change: median={r.get('median')}, q025={r.get('q025')}, q975={r.get('q975')}, decision={r.get('decision')}")
    lines.append("")
    lines.append("## Feature peak audit")
    if feature_audit.empty:
        lines.append("- Not available.")
    else:
        early_feats = feature_audit[feature_audit.get("is_early_peak_in_core_day26_33", False) == True]
        lines.append(f"- Raw features peaking in day26–33: {len(early_feats)}")
        for _, r in early_feats.iterrows():
            lines.append(f"  - {r.get('feature_name')}: early peak day {r.get('early_range_peak_day')}, early/late ratio={r.get('early_to_late_score_ratio')}")
    lines.append("")
    lines.append("## Detector bootstrap")
    if detector_boot.empty:
        lines.append("- Not available.")
    else:
        for _, r in detector_boot.iterrows():
            lines.append(f"- {r.get('detector_scope')}: P_has_any_early_peak_day26_33={r.get('P_has_any_early_peak_day26_33')}, decision={r.get('decision')}")
    lines.append("")
    lines.append("## Allowed statement")
    if final == "raw_supported_weak_early_peak":
        lines.append("Je has a raw-profile-supported weak early structural peak around day26–33, although it remains weaker than the day46 raw-profile main adjustment.")
    elif final == "raw_supported_early_ramp":
        lines.append("Je shows a raw-profile-supported early ramp around day26–33, but not a clean independent early peak.")
    elif final == "late_peak_precursor_only":
        lines.append("Je day26–33 is best treated as a weak precursor/ramp toward the robust day46 raw-profile main adjustment.")
    elif final == "normalization_sensitive_only":
        lines.append("Je day33 should remain a normalization-sensitive shape-pattern signal, without enough raw-profile support to upgrade it.")
    else:
        lines.append("Je day26–33 remains unresolved under the raw-support audit.")
    lines.append("")
    lines.append("## Forbidden statement")
    lines.append("- Do not write that Je day33 is a confirmed early physical transition unless raw-supported weak peak criteria are met.")
    lines.append("- Do not use this audit to infer Je causal influence on other objects.")
    _write_text("\n".join(lines) + "\n", paths.output_dir / "Je_raw_supported_early_signal_summary_v7_z_b.md")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def _maybe_plot(paths: Paths, feature_df: pd.DataFrame, ba_profiles: pd.DataFrame, detector_scores: pd.DataFrame, cumulative_df: pd.DataFrame, cfg: JeRawSupportAuditConfig) -> None:
    if not cfg.write_figures:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    # Feature timeseries.
    if not feature_df.empty:
        for cols, name in [
            (["strength_mean", "amplitude", "weighted_l2_norm"], "Je_raw_feature_timeseries_strength_v7_z_b.png"),
            (["axis_lat", "centroid_lat", "spread", "NS_contrast"], "Je_raw_feature_timeseries_position_shape_v7_z_b.png"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 4))
            for col in cols:
                if col in feature_df.columns:
                    ax.plot(feature_df["day"], feature_df[col], label=col)
            ax.axvspan(40, 48, alpha=0.15, label="W45")
            ax.axvspan(26, 33, alpha=0.10, label="early core")
            ax.legend(fontsize=8)
            ax.set_xlabel("day")
            ax.set_title(name.replace("_", " "))
            fig.tight_layout()
            fig.savefig(paths.figure_dir / name, dpi=160)
            plt.close(fig)
    # Before-after profiles.
    if not ba_profiles.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for event in ba_profiles["event"].unique():
            for phase in ("before", "after"):
                sub = ba_profiles[(ba_profiles["event"] == event) & (ba_profiles["phase"] == phase)]
                ax.plot(sub["profile_value"], sub["lat"], label=f"{event}:{phase}")
        ax.legend(fontsize=7)
        ax.set_xlabel("Je raw profile")
        ax.set_ylabel("lat")
        ax.set_title("Je before-after raw profiles")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "Je_raw_before_after_profiles_v7_z_b.png", dpi=160)
        plt.close(fig)
    # Detector scores.
    if not detector_scores.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        for scope, sub in detector_scores.groupby("detector_scope"):
            ax.plot(sub["day"], sub["detector_score"], label=scope)
        ax.axvspan(40, 48, alpha=0.15, label="W45")
        ax.axvspan(26, 33, alpha=0.10, label="early core")
        ax.axvline(33, linestyle="--", linewidth=1)
        ax.axvline(46, linestyle="--", linewidth=1)
        ax.legend(fontsize=7)
        ax.set_title("Je raw detector sensitivity scores")
        ax.set_xlabel("day")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "Je_early_vs_late_raw_detector_scores_v7_z_b.png", dpi=160)
        plt.close(fig)
    # Cumulative change.
    if not cumulative_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cumulative_df["day"], cumulative_df["cum_raw_change_from_day20_25"], label="cum raw change")
        ax.axvspan(40, 48, alpha=0.15, label="W45")
        ax.axvspan(26, 33, alpha=0.10, label="early core")
        ax.legend(fontsize=8)
        ax.set_xlabel("day")
        ax.set_title("Je raw cumulative change from day20-25")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "Je_raw_cumulative_change_v7_z_b.png", dpi=160)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def run_W45_Je_raw_supported_early_signal_audit_v7_z_b(v7_root: Optional[Path] = None) -> dict[str, Any]:
    cfg = JeRawSupportAuditConfig()
    if os.environ.get("V7Z_JE_AUDIT_B_DEBUG_N_BOOTSTRAP"):
        cfg = JeRawSupportAuditConfig(
            n_bootstrap=int(os.environ["V7Z_JE_AUDIT_B_DEBUG_N_BOOTSTRAP"]),
            write_figures=os.environ.get("V7Z_JE_AUDIT_B_SKIP_FIGURES", "0") not in ("1", "true", "True"),
        )
    else:
        cfg = JeRawSupportAuditConfig(write_figures=os.environ.get("V7Z_JE_AUDIT_B_SKIP_FIGURES", "0") not in ("1", "true", "True"))

    paths = _resolve_paths(v7_root, cfg)
    log_path = paths.log_dir / "run.log"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] {msg}\n")

    _write_csv(_input_audit(paths), paths.output_dir / "Je_raw_supported_input_audit_v7_z_b.csv")
    _write_json({"version": VERSION, "output_tag": OUTPUT_TAG, "config": asdict(cfg), "start_time": _now_iso()}, paths.output_dir / "run_meta.json")

    log("[1/9] Load smoothed fields and build Je raw profile")
    files = _load_npz(paths.smoothed_fields_path)
    raw_cube, lat_grid, u_key, input_key_audit = _build_je_profile(files)
    _write_csv(input_key_audit, paths.output_dir / "Je_raw_supported_input_key_audit_v7_z_b.csv")
    raw_clim = _climatology(raw_cube)

    profile_rows = []
    for day in range(FULL_RANGE[0], FULL_RANGE[1] + 1):
        for lat, val in zip(lat_grid, raw_clim[day, :]):
            profile_rows.append({"day": int(day), "date_mmdd": _day_to_month_day(day), "lat": float(lat), "raw_profile_value": float(val) if np.isfinite(val) else np.nan})
    _write_csv(pd.DataFrame(profile_rows), paths.output_dir / "Je_raw_profile_climatology_v7_z_b.csv")

    log("[2/9] Compute Je raw feature timeseries")
    feature_df = _profile_features_from_matrix(raw_clim, lat_grid)
    _write_csv(feature_df, paths.output_dir / "Je_raw_feature_timeseries_v7_z_b.csv")

    log("[3/9] Run feature peak audit")
    feature_audit = _feature_peak_audit(feature_df)
    _write_csv(feature_audit, paths.output_dir / "Je_raw_feature_peak_audit_v7_z_b.csv")

    log("[4/9] Compute raw before-after observed metrics")
    ba_profiles = _before_after_profile_rows(raw_clim, lat_grid)
    ba_observed = _before_after_metrics_for_clim(raw_clim, lat_grid)
    _write_csv(ba_profiles, paths.output_dir / "Je_raw_before_after_profiles_v7_z_b.csv")
    _write_csv(ba_observed, paths.output_dir / "Je_raw_before_after_observed_metrics_v7_z_b.csv")

    log(f"[5/9] Run raw before-after bootstrap (n={cfg.n_bootstrap})")
    ba_samples, ba_summary = _run_before_after_bootstrap(raw_cube, lat_grid, cfg)
    if cfg.save_bootstrap_samples:
        _write_csv(ba_samples, paths.output_dir / "Je_raw_before_after_bootstrap_samples_v7_z_b.csv")
    # attach observed values where available
    if not ba_summary.empty and not ba_observed.empty:
        obs_map = ba_observed.set_index("event").to_dict(orient="index")
        obs_vals = []
        for _, r in ba_summary.iterrows():
            obs_vals.append(obs_map.get(r["event"], {}).get(r["metric"], np.nan))
        ba_summary["observed"] = obs_vals
    _write_csv(ba_summary, paths.output_dir / "Je_raw_before_after_bootstrap_summary_v7_z_b.csv")

    log("[6/9] Run early-window raw detector sensitivity")
    detector_scores, detector_peaks = _detector_sensitivity(raw_clim, lat_grid, cfg)
    _write_csv(detector_scores, paths.output_dir / "Je_early_window_raw_detector_scores_v7_z_b.csv")
    _write_csv(detector_peaks, paths.output_dir / "Je_early_window_raw_detector_sensitivity_v7_z_b.csv")

    log(f"[7/9] Run light detector bootstrap (n={cfg.n_bootstrap})")
    detector_boot = _run_detector_bootstrap(raw_cube, lat_grid, cfg)
    _write_csv(detector_boot, paths.output_dir / "Je_early_window_raw_detector_bootstrap_summary_v7_z_b.csv")

    log("[8/9] Compute cumulative/ramp audit and final decision")
    cumulative_df = _cumulative_change(raw_clim, lat_grid)
    _write_csv(cumulative_df, paths.output_dir / "Je_raw_cumulative_change_v7_z_b.csv")
    peak_vs_ramp = _peak_vs_ramp_audit(cumulative_df, detector_peaks, feature_audit)
    _write_csv(peak_vs_ramp, paths.output_dir / "Je_early_peak_vs_ramp_audit_v7_z_b.csv")

    prev = _load_previous_audit_a(paths)
    decision = _final_decision(ba_summary, detector_boot, feature_audit, peak_vs_ramp, prev)
    _write_csv(decision, paths.output_dir / "Je_day33_raw_support_decision_v7_z_b.csv")

    log("[9/9] Write figures and markdown summary")
    _maybe_plot(paths, feature_df, ba_profiles, detector_scores, cumulative_df, cfg)
    _write_summary(paths, decision, ba_summary, feature_audit, detector_boot, peak_vs_ramp)

    final_status = "unknown"
    sub = decision[decision["check_item"] == "final_Je_day26_33_status"]
    if not sub.empty:
        final_status = str(sub.iloc[0]["decision"])
    summary = {
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "u200_key": u_key,
        "n_years": int(raw_cube.shape[0]),
        "n_days": int(raw_cube.shape[1]),
        "n_bootstrap": int(cfg.n_bootstrap),
        "final_Je_day26_33_status": final_status,
        "output_dir": str(paths.output_dir),
        "end_time": _now_iso(),
    }
    _write_json(summary, paths.output_dir / "summary.json")
    log(f"Done. final_Je_day26_33_status={final_status}. Output: {paths.output_dir}")
    return summary


if __name__ == "__main__":
    run_W45_Je_raw_supported_early_signal_audit_v7_z_b()
