from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from stage_partition_v6.io import load_smoothed_fields
from stage_partition_v6.state_builder import build_profiles

from .config import StagePartitionV7Settings

FIELD = "H"
FIELD_KEY = "z500_smoothed"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
OUTPUT_TAG = "w45_H_latbin_profile_progress_v7_i"


@dataclass
class W45HLatBinPaths:
    v7_root: Path
    project_root: Path
    v7e_output_dir: Path
    v7f_output_dir: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _resolve_paths(v7_root: Optional[Path]) -> W45HLatBinPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = v7_root.parents[1]
    return W45HLatBinPaths(
        v7_root=v7_root,
        project_root=project_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7f_output_dir=v7_root / "outputs" / "w45_H_feature_progress_v7_f",
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _configure_settings(paths: W45HLatBinPaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    return settings


def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(float(lower), float(upper)), max(float(lower), float(upper))
    return (arr >= lo) & (arr <= hi)


def _safe_nanmean(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _finite_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if int(good.sum()) == 0:
        return np.nan
    return float(_safe_nanmean(x[good]))


def _load_w45_window(paths: W45HLatBinPaths, n_days: int) -> dict:
    win_path = paths.v7e_output_dir / "accepted_windows_used_v7_e.csv"
    if win_path.exists():
        df = pd.read_csv(win_path)
        sub = pd.DataFrame()
        if "window_id" in df.columns:
            sub = df[df["window_id"].astype(str) == WINDOW_ID]
        if sub.empty and "anchor_day" in df.columns:
            sub = df[pd.to_numeric(df["anchor_day"], errors="coerce") == ANCHOR_DAY]
        if not sub.empty:
            r = sub.iloc[0].to_dict()
            return {
                "window_id": str(r.get("window_id", WINDOW_ID)),
                "anchor_day": int(r.get("anchor_day", ANCHOR_DAY)),
                "accepted_window_start": int(r.get("accepted_window_start", 40)),
                "accepted_window_end": int(r.get("accepted_window_end", 48)),
                "analysis_window_start": int(r.get("analysis_window_start", 30)),
                "analysis_window_end": int(r.get("analysis_window_end", 60)),
                "pre_period_start": int(r.get("pre_period_start", 30)),
                "pre_period_end": int(r.get("pre_period_end", 37)),
                "post_period_start": int(r.get("post_period_start", 53)),
                "post_period_end": int(r.get("post_period_end", 60)),
                "source": "field_transition_progress_timing_v7_e/accepted_windows_used_v7_e.csv",
            }
    return {
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "accepted_window_start": 40,
        "accepted_window_end": 48,
        "analysis_window_start": max(0, ANCHOR_DAY - 15),
        "analysis_window_end": min(n_days - 1, ANCHOR_DAY + 15),
        "pre_period_start": max(0, ANCHOR_DAY - 15),
        "pre_period_end": max(0, ANCHOR_DAY - 8),
        "post_period_start": min(n_days - 1, ANCHOR_DAY + 8),
        "post_period_end": min(n_days - 1, ANCHOR_DAY + 15),
        "source": "fallback_v7e_documented_anchor_pm15",
    }


def _count_crossings(vals: np.ndarray, threshold: float) -> int:
    vals = np.asarray(vals, dtype=float)
    good = np.isfinite(vals)
    if int(good.sum()) < 2:
        return 0
    above = vals[good] >= float(threshold)
    return int(np.sum(above[1:] != above[:-1]))


def _first_stable_crossing(days: np.ndarray, vals: np.ndarray, threshold: float, stable_days: int) -> float:
    days = np.asarray(days, dtype=int)
    vals = np.asarray(vals, dtype=float)
    stable_days = max(1, int(stable_days))
    if len(vals) < stable_days:
        return np.nan
    for i in range(0, len(vals) - stable_days + 1):
        win = vals[i : i + stable_days]
        if np.all(np.isfinite(win)) and np.all(win >= float(threshold)):
            return float(days[i])
    return np.nan


def _separation_label(ratio: float, settings: StagePartitionV7Settings) -> str:
    cfg = settings.progress_timing
    if not np.isfinite(ratio):
        return "unavailable"
    if ratio >= float(cfg.separation_clear_ratio):
        return "clear_separation"
    if ratio >= float(cfg.separation_moderate_ratio):
        return "moderate_separation"
    if ratio >= float(cfg.separation_weak_ratio):
        return "weak_separation"
    return "no_clear_separation"


def _build_current_interp_h_profile(smoothed: dict, settings: StagePartitionV7Settings) -> tuple[np.ndarray, np.ndarray]:
    """Return current V6/V7 H profile: lon-mean then np.interp to 2-degree lat points."""
    profs = build_profiles(smoothed, settings.profile)
    obj = profs[FIELD]
    return np.asarray(obj.raw_cube, dtype=float), np.asarray(obj.lat_grid, dtype=float)


def _build_latbin_h_profile(smoothed: dict, settings: StagePartitionV7Settings) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Build H profile by true latitude-bin averaging after lon-mean.

    This is the central V7-i implementation change. Current V6 build_profiles does:
      lon mean -> interpolation to 2-degree latitude grid.
    Here we do:
      lon mean -> latitude bin mean over raw latitudes for each 2-degree feature center.

    It intentionally keeps the same target feature centers as the current 2-degree grid,
    but each feature now has a recorded raw-lat support interval and raw-lat count.
    """
    cfg = settings.profile
    lat = np.asarray(smoothed["lat"], dtype=float)
    lon = np.asarray(smoothed["lon"], dtype=float)
    field = np.asarray(smoothed[FIELD_KEY], dtype=float)
    lat_range = cfg.h_lat_range
    lon_range = cfg.h_lon_range
    step = float(cfg.lat_step_deg)
    lo, hi = min(lat_range), max(lat_range)

    lat_mask = _mask_between(lat, lo, hi)
    lon_mask = _mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"No H latitude points in range {lat_range}")
    if not np.any(lon_mask):
        raise ValueError(f"No H longitude points in range {lon_range}")

    raw_lats = lat[lat_mask]
    # years x days x raw_lats x lon -> years x days x raw_lats
    subset = field[:, :, lat_mask, :][:, :, :, lon_mask]
    lon_mean = _safe_nanmean(subset, axis=-1)

    centers = np.arange(lo, hi + 1e-9, step, dtype=float)
    out = np.full((field.shape[0], field.shape[1], centers.size), np.nan, dtype=float)
    rows: list[dict] = []
    for j, center in enumerate(centers):
        left = max(lo, float(center - step / 2.0))
        right = min(hi, float(center + step / 2.0))
        if j == centers.size - 1:
            mask = (raw_lats >= left) & (raw_lats <= right)
        else:
            mask = (raw_lats >= left) & (raw_lats < right)
        if not np.any(mask):
            # Fallback: nearest raw latitude, clearly marked. This should be rare.
            k = int(np.nanargmin(np.abs(raw_lats - center)))
            mask = np.zeros_like(raw_lats, dtype=bool)
            mask[k] = True
            method = "nearest_raw_lat_fallback"
        else:
            method = "lat_bin_mean_after_lon_mean"
        vals = lon_mean[:, :, mask]
        out[:, :, j] = _safe_nanmean(vals, axis=-1)
        rows.append({
            "field": FIELD,
            "feature_index": int(j),
            "feature_lat_center": float(center),
            "lat_bin_left": float(left),
            "lat_bin_right": float(right),
            "lat_bin_rule": "left_closed_right_open_except_last_bin",
            "n_raw_lat_points": int(mask.sum()),
            "raw_lat_min": float(np.nanmin(raw_lats[mask])) if np.any(mask) else np.nan,
            "raw_lat_max": float(np.nanmax(raw_lats[mask])) if np.any(mask) else np.nan,
            "source_raw_lat_values": ";".join(f"{x:.6g}" for x in raw_lats[mask]),
            "aggregation_method": method,
            "lon_aggregation_before_lat_bin": "nanmean_over_H_lon_range",
            "lat_aggregation": "nanmean_over_raw_latitudes_in_bin",
            "area_weighted": False,
            "note": "This feature is a true latitude-bin mean, not np.interp at the feature center.",
        })
    return out, centers, pd.DataFrame(rows), raw_lats


def _seasonal_mean(cube: np.ndarray, year_indices: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(cube, dtype=float)
    if year_indices is not None:
        arr = arr[np.asarray(year_indices, dtype=int), :, :]
    return _safe_nanmean(arr, axis=0)


def _progress_for_unit(
    seasonal: np.ndarray,
    feature_indices: list[int],
    unit_meta: dict,
    window: dict,
    settings: StagePartitionV7Settings,
) -> tuple[dict, pd.DataFrame]:
    cfg = settings.progress_timing
    mat = np.asarray(seasonal[:, feature_indices], dtype=float)
    astart = int(window["analysis_window_start"])
    aend = int(window["analysis_window_end"])
    pre_start = int(window["pre_period_start"])
    pre_end = int(window["pre_period_end"])
    post_start = int(window["post_period_start"])
    post_end = int(window["post_period_end"])
    anchor = int(window["anchor_day"])
    base = {
        "window_id": str(window["window_id"]),
        "anchor_day": anchor,
        "field": FIELD,
        **unit_meta,
        "accepted_window_start": int(window["accepted_window_start"]),
        "accepted_window_end": int(window["accepted_window_end"]),
        "analysis_window_start": astart,
        "analysis_window_end": aend,
        "pre_period_start": pre_start,
        "pre_period_end": pre_end,
        "post_period_start": post_start,
        "post_period_end": post_end,
    }
    pre = mat[pre_start : pre_end + 1, :]
    post = mat[post_start : post_end + 1, :]
    pre_good_days = int(np.sum(np.any(np.isfinite(pre), axis=1)))
    post_good_days = int(np.sum(np.any(np.isfinite(post), axis=1)))
    if pre_good_days < int(cfg.min_period_days) or post_good_days < int(cfg.min_period_days):
        row = {**base, "pre_post_distance": np.nan, "within_pre_variability": np.nan, "within_post_variability": np.nan, "separation_ratio": np.nan, "pre_post_separation_label": "period_too_short", "onset_day": np.nan, "midpoint_day": np.nan, "finish_day": np.nan, "duration": np.nan, "progress_monotonicity_corr": np.nan, "n_crossings_025": np.nan, "n_crossings_050": np.nan, "n_crossings_075": np.nan, "progress_quality_label": "period_too_short", "near_left_boundary": False, "near_right_boundary": False}
        return row, pd.DataFrame()

    pre_proto = _safe_nanmean(pre, axis=0)
    post_proto = _safe_nanmean(post, axis=0)
    vec = post_proto - pre_proto
    norm2 = float(np.nansum(np.square(vec)))
    pre_post_distance = float(np.sqrt(norm2)) if np.isfinite(norm2) else np.nan
    pre_d = np.sqrt(np.nansum(np.square(pre - pre_proto[None, :]), axis=1))
    post_d = np.sqrt(np.nansum(np.square(post - post_proto[None, :]), axis=1))
    pre_var = _finite_mean(pre_d)
    post_var = _finite_mean(post_d)
    within = _finite_mean(np.asarray([pre_var, post_var], dtype=float))
    separation_ratio = float(pre_post_distance / (within + 1e-12)) if np.isfinite(within) else np.nan
    sep_label = _separation_label(separation_ratio, settings)

    days: list[int] = []
    vals: list[float] = []
    curve_rows: list[dict] = []
    if np.isfinite(norm2) and norm2 > float(cfg.min_transition_norm):
        for day in range(astart, aend + 1):
            x = mat[day, :]
            if not np.any(np.isfinite(x)):
                continue
            raw = float(np.nansum((x - pre_proto) * vec) / (norm2 + 1e-12))
            progress = float(np.clip(raw, float(cfg.progress_clip_min), float(cfg.progress_clip_max)))
            days.append(int(day))
            vals.append(progress)
            curve_rows.append({**base, "day": int(day), "relative_to_anchor": int(day - anchor), "raw_progress": raw, "progress": progress})

    days_arr = np.asarray(days, dtype=int)
    vals_arr = np.asarray(vals, dtype=float)
    if len(vals_arr) == 0:
        onset = mid = finish = duration = corr = np.nan
        c25 = c50 = c75 = np.nan
        quality = "no_clear_prepost_separation" if sep_label == "no_clear_separation" else "progress_unavailable"
    else:
        onset = _first_stable_crossing(days_arr, vals_arr, float(cfg.threshold_onset), int(cfg.stable_crossing_days))
        mid = _first_stable_crossing(days_arr, vals_arr, float(cfg.threshold_midpoint), int(cfg.stable_crossing_days))
        finish = _first_stable_crossing(days_arr, vals_arr, float(cfg.threshold_finish), int(cfg.stable_crossing_days))
        duration = float(finish - onset + 1) if np.isfinite(onset) and np.isfinite(finish) else np.nan
        good = np.isfinite(vals_arr)
        corr = np.nan
        if int(good.sum()) >= 3 and np.nanstd(vals_arr[good]) > 1e-12:
            corr = float(np.corrcoef(days_arr[good].astype(float), vals_arr[good])[0, 1])
        c25 = _count_crossings(vals_arr, float(cfg.threshold_onset))
        c50 = _count_crossings(vals_arr, float(cfg.threshold_midpoint))
        c75 = _count_crossings(vals_arr, float(cfg.threshold_finish))
        near_left = any(np.isfinite(z) and z <= astart + int(cfg.boundary_margin_days) for z in [onset, mid])
        near_right = any(np.isfinite(z) and z >= aend - int(cfg.boundary_margin_days) for z in [mid, finish])
        excess_crossings = max(int(c25) - 1, 0) + max(int(c50) - 1, 0) + max(int(c75) - 1, 0)
        nonmono = (np.isfinite(corr) and corr < float(cfg.monotonic_corr_threshold)) or excess_crossings >= 1
        if sep_label == "no_clear_separation":
            quality = "no_clear_prepost_separation"
        elif not np.isfinite(mid):
            quality = "partial_progress"
        elif near_left or near_right:
            quality = "boundary_limited_progress"
        elif nonmono:
            quality = "nonmonotonic_progress"
        elif sep_label in {"clear_separation", "moderate_separation"} and np.isfinite(onset) and np.isfinite(finish):
            quality = "monotonic_clear_progress"
        else:
            quality = "monotonic_broad_progress"

    near_left_boundary = bool(any(np.isfinite(z) and z <= astart + int(cfg.boundary_margin_days) for z in [onset, mid]))
    near_right_boundary = bool(any(np.isfinite(z) and z >= aend - int(cfg.boundary_margin_days) for z in [mid, finish]))
    row = {
        **base,
        "pre_post_distance": pre_post_distance,
        "within_pre_variability": pre_var,
        "within_post_variability": post_var,
        "separation_ratio": separation_ratio,
        "pre_post_separation_label": sep_label,
        "onset_day": onset,
        "midpoint_day": mid,
        "finish_day": finish,
        "duration": duration,
        "progress_monotonicity_corr": corr,
        "n_crossings_025": c25,
        "n_crossings_050": c50,
        "n_crossings_075": c75,
        "progress_quality_label": quality,
        "near_left_boundary": near_left_boundary,
        "near_right_boundary": near_right_boundary,
    }
    return row, pd.DataFrame(curve_rows)


def _load_or_make_bootstrap_indices(n_years: int, paths: W45HLatBinPaths, settings: StagePartitionV7Settings) -> list[np.ndarray]:
    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    src = paths.v7e_output_dir / "bootstrap_resample_year_indices_v7_e.csv"
    out: list[np.ndarray] = []
    if src.exists():
        df = pd.read_csv(src)
        if {"bootstrap_id", "sampled_year_indices"}.issubset(df.columns):
            for _, r in df.sort_values("bootstrap_id").iterrows():
                vals = [int(x) for x in str(r["sampled_year_indices"]).split(";") if str(x).strip()]
                if vals:
                    out.append(np.asarray(vals, dtype=int))
            if len(out) >= n_boot:
                return out[:n_boot]
    rng = np.random.default_rng(int(settings.bootstrap.random_seed))
    return [rng.integers(0, n_years, size=n_years, dtype=int) for _ in range(n_boot)]


def _summarize_samples(samples: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if samples.empty:
        return pd.DataFrame()
    for keys, sub in samples.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: val for col, val in zip(group_cols, keys)}
        mids = pd.to_numeric(sub["midpoint_day"], errors="coerce").dropna().to_numpy(dtype=float)
        onsets = pd.to_numeric(sub["onset_day"], errors="coerce").dropna().to_numpy(dtype=float)
        finishes = pd.to_numeric(sub["finish_day"], errors="coerce").dropna().to_numpy(dtype=float)
        durations = pd.to_numeric(sub["duration"], errors="coerce").dropna().to_numpy(dtype=float)
        qual_counts = sub["progress_quality_label"].astype(str).value_counts().to_dict() if "progress_quality_label" in sub.columns else {}
        sep_counts = sub["pre_post_separation_label"].astype(str).value_counts().to_dict() if "pre_post_separation_label" in sub.columns else {}
        row = {**base, "n_samples": int(len(sub)), "n_valid_midpoints": int(mids.size), "valid_midpoint_fraction": float(mids.size / len(sub)) if len(sub) else np.nan}
        if mids.size:
            row.update({
                "onset_median": float(np.nanmedian(onsets)) if onsets.size else np.nan,
                "onset_q05": float(np.nanpercentile(onsets, 5)) if onsets.size else np.nan,
                "onset_q95": float(np.nanpercentile(onsets, 95)) if onsets.size else np.nan,
                "midpoint_median": float(np.nanmedian(mids)),
                "midpoint_q05": float(np.nanpercentile(mids, 5)),
                "midpoint_q95": float(np.nanpercentile(mids, 95)),
                "midpoint_q25": float(np.nanpercentile(mids, 25)),
                "midpoint_q75": float(np.nanpercentile(mids, 75)),
                "midpoint_iqr": float(np.nanpercentile(mids, 75) - np.nanpercentile(mids, 25)),
                "midpoint_q90_width": float(np.nanpercentile(mids, 95) - np.nanpercentile(mids, 5)),
                "finish_median": float(np.nanmedian(finishes)) if finishes.size else np.nan,
                "finish_q05": float(np.nanpercentile(finishes, 5)) if finishes.size else np.nan,
                "finish_q95": float(np.nanpercentile(finishes, 95)) if finishes.size else np.nan,
                "duration_median": float(np.nanmedian(durations)) if durations.size else np.nan,
                "duration_q25": float(np.nanpercentile(durations, 25)) if durations.size else np.nan,
                "duration_q75": float(np.nanpercentile(durations, 75)) if durations.size else np.nan,
            })
        else:
            row.update({"onset_median": np.nan, "midpoint_median": np.nan, "midpoint_q05": np.nan, "midpoint_q95": np.nan, "midpoint_iqr": np.nan, "midpoint_q90_width": np.nan})
        row["dominant_progress_quality_label"] = max(qual_counts, key=qual_counts.get) if qual_counts else "none"
        row["dominant_prepost_separation_label"] = max(sep_counts, key=sep_counts.get) if sep_counts else "none"
        rows.append(row)
    return pd.DataFrame(rows)


def _assign_bin_timing_classes(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    mids = pd.to_numeric(out.get("midpoint_median"), errors="coerce")
    vals = mids[np.isfinite(mids)].to_numpy(dtype=float)
    if vals.size == 0:
        out["timing_class"] = "unstable_latbin_feature"
        return out
    q33 = float(np.nanpercentile(vals, 33.333))
    q66 = float(np.nanpercentile(vals, 66.667))
    labels = []
    for _, r in out.iterrows():
        qlabel = str(r.get("dominant_progress_quality_label", ""))
        sep = str(r.get("dominant_prepost_separation_label", ""))
        med = pd.to_numeric(pd.Series([r.get("midpoint_median")]), errors="coerce").iloc[0]
        valid_frac = float(r.get("valid_midpoint_fraction", 0.0)) if pd.notna(r.get("valid_midpoint_fraction", np.nan)) else 0.0
        if not np.isfinite(med) or valid_frac < 0.5 or sep in {"no_clear_separation", "unavailable"} or "nonmonotonic" in qlabel:
            labels.append("unstable_latbin_feature")
        elif med <= q33:
            labels.append("early_latbin_candidate")
        elif med >= q66:
            labels.append("late_latbin_candidate")
        else:
            labels.append("middle_latbin_candidate")
    out["timing_class"] = labels
    out["timing_class_note"] = "diagnostic tertile class based on lat-bin midpoint_median within W45 H lat-bin features; not a statistical confirmation label"
    return out


def _load_v7e_whole_h(paths: W45HLatBinPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv", required=False)
    summ = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_summary_v7_e.csv", required=False)
    if not obs.empty and {"window_id", "field"}.issubset(obs.columns):
        obs = obs[(obs["window_id"].astype(str) == WINDOW_ID) & (obs["field"].astype(str) == FIELD)].copy()
    if not summ.empty and {"window_id", "field"}.issubset(summ.columns):
        summ = summ[(summ["window_id"].astype(str) == WINDOW_ID) & (summ["field"].astype(str) == FIELD)].copy()
    return obs, summ


def _load_v7f_single_feature(paths: W45HLatBinPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = _read_csv(paths.v7f_output_dir / "w45_H_feature_progress_observed_v7_f.csv", required=False)
    summ = _read_csv(paths.v7f_output_dir / "w45_H_feature_progress_bootstrap_summary_v7_f.csv", required=False)
    return obs, summ


def _profile_construction_comparison(current_cube: np.ndarray, current_lats: np.ndarray, latbin_cube: np.ndarray, latbin_lats: np.ndarray, bin_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if current_cube.shape[2] != latbin_cube.shape[2]:
        # Still emit available bin map and basic dimensions.
        for _, r in bin_map.iterrows():
            rows.append({**r.to_dict(), "current_interp_available": False, "interp_minus_latbin_mean_abs": np.nan, "interp_latbin_corr_over_all_year_day": np.nan})
        return pd.DataFrame(rows)
    for j, center in enumerate(latbin_lats):
        cur = current_cube[:, :, j].reshape(-1)
        binned = latbin_cube[:, :, j].reshape(-1)
        valid = np.isfinite(cur) & np.isfinite(binned)
        diff = cur[valid] - binned[valid]
        corr = np.nan
        if int(valid.sum()) >= 3 and np.nanstd(cur[valid]) > 1e-12 and np.nanstd(binned[valid]) > 1e-12:
            corr = float(np.corrcoef(cur[valid], binned[valid])[0, 1])
        br = bin_map.iloc[j].to_dict()
        rows.append({
            **br,
            "current_interp_lat_center": float(current_lats[j]) if j < len(current_lats) else np.nan,
            "current_interp_available": True,
            "n_valid_year_day_pairs": int(valid.sum()),
            "interp_minus_latbin_mean": float(np.nanmean(diff)) if diff.size else np.nan,
            "interp_minus_latbin_mean_abs": float(np.nanmean(np.abs(diff))) if diff.size else np.nan,
            "interp_minus_latbin_std": float(np.nanstd(diff)) if diff.size else np.nan,
            "interp_latbin_corr_over_all_year_day": corr,
            "construction_difference_note": "current profile is np.interp at center after lon-mean; latbin profile is raw-lat bin mean after lon-mean",
        })
    return pd.DataFrame(rows)


def _build_comparison_table(paths: W45HLatBinPaths, latbin_summary: pd.DataFrame, bin_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    whole_obs, whole_summ = _load_v7e_whole_h(paths)
    if not whole_summ.empty:
        r = whole_summ.iloc[0]
        rows.append({
            "unit_type": "whole_field_H_current_interp_profile",
            "unit_label": "whole_field_H",
            "n_units": 1,
            "median_q90_width": r.get("midpoint_q90_width", r.get("midpoint_q95", np.nan) - r.get("midpoint_q05", np.nan) if "midpoint_q95" in r and "midpoint_q05" in r else np.nan),
            "n_unstable_units": np.nan,
            "dominant_progress_quality_label": whole_obs.iloc[0].get("progress_quality_label", "unavailable") if not whole_obs.empty else "unavailable",
            "interpretation": "V7-e whole-field result, based on current interpolated 2-degree profile construction.",
        })
    feat_obs, feat_summ = _load_v7f_single_feature(paths)
    if not feat_summ.empty:
        width_col = "midpoint_q90_width" if "midpoint_q90_width" in feat_summ.columns else None
        qwidth = pd.to_numeric(feat_summ[width_col], errors="coerce") if width_col else pd.Series(dtype=float)
        unstable = feat_summ["timing_class"].astype(str).str.contains("unstable", na=False).sum() if "timing_class" in feat_summ.columns else np.nan
        rows.append({
            "unit_type": "single_2deg_interp_feature_v7_f",
            "unit_label": "single_interp_feature",
            "n_units": int(len(feat_summ)),
            "median_q90_width": float(np.nanmedian(qwidth)) if len(qwidth) else np.nan,
            "n_unstable_units": int(unstable) if pd.notna(unstable) else np.nan,
            "dominant_progress_quality_label": feat_summ.get("dominant_progress_quality_label", pd.Series(dtype=str)).astype(str).value_counts().idxmax() if "dominant_progress_quality_label" in feat_summ.columns and not feat_summ.empty else "unavailable",
            "interpretation": "V7-f single-feature result using current 2-degree interpolated lat points; diagnostic only.",
        })
    if not latbin_summary.empty:
        qwidth = pd.to_numeric(latbin_summary["midpoint_q90_width"], errors="coerce")
        unstable = latbin_summary["timing_class"].astype(str).str.contains("unstable", na=False).sum() if "timing_class" in latbin_summary.columns else np.nan
        rows.append({
            "unit_type": "single_2deg_latbin_mean_feature_v7_i",
            "unit_label": "single_latbin_mean_feature",
            "n_units": int(len(latbin_summary)),
            "median_q90_width": float(np.nanmedian(qwidth)) if len(qwidth) else np.nan,
            "n_unstable_units": int(unstable) if pd.notna(unstable) else np.nan,
            "dominant_progress_quality_label": latbin_summary["dominant_progress_quality_label"].astype(str).value_counts().idxmax() if "dominant_progress_quality_label" in latbin_summary.columns and not latbin_summary.empty else "unavailable",
            "interpretation": "V7-i single-feature result using true raw-lat bin mean after lon mean; tests whether 2-degree feature construction had real lat-direction denoising.",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # A conservative diagnostic flag; not a scientific result by itself.
        latbin = out[out["unit_type"] == "single_2deg_latbin_mean_feature_v7_i"]
        interp = out[out["unit_type"] == "single_2deg_interp_feature_v7_f"]
        if not latbin.empty and not interp.empty:
            lw = pd.to_numeric(latbin["median_q90_width"], errors="coerce").iloc[0]
            iw = pd.to_numeric(interp["median_q90_width"], errors="coerce").iloc[0]
            lu = pd.to_numeric(latbin["n_unstable_units"], errors="coerce").iloc[0]
            iu = pd.to_numeric(interp["n_unstable_units"], errors="coerce").iloc[0]
            if np.isfinite(lw) and np.isfinite(iw) and (lw < iw or (pd.notna(lu) and pd.notna(iu) and lu < iu)):
                verdict = "latbin_mean_improves_some_stability"
            elif np.isfinite(lw) and np.isfinite(iw) and lw >= iw and (pd.isna(lu) or pd.isna(iu) or lu >= iu):
                verdict = "latbin_mean_does_not_improve_stability"
            else:
                verdict = "latbin_mean_effect_unclear"
        else:
            verdict = "comparison_to_v7f_unavailable"
        out["latbin_vs_interp_verdict"] = verdict
    return out


def _build_upstream_implication(comparison: pd.DataFrame, construction_cmp: pd.DataFrame) -> pd.DataFrame:
    verdict = "comparison_unavailable"
    if not comparison.empty and "latbin_vs_interp_verdict" in comparison.columns:
        verdict = str(comparison["latbin_vs_interp_verdict"].dropna().iloc[0]) if comparison["latbin_vs_interp_verdict"].notna().any() else verdict
    rows = []
    rows.append({
        "diagnostic_item": "current_2deg_profile_is_interpolation_not_lat_mean",
        "status": "confirmed_by_V6_state_builder_code",
        "evidence": "V6 build_profiles uses lon nanmean then np.interp to dst_lats; V7-i builds an explicit lat-bin mean alternative.",
        "upstream_implication": "Previous V7-f/g/h feature-space regional diagnostics were based on interpolated 2-degree latitude points, not true 2-degree latitude-bin averages.",
        "recommended_followup": "Use V7-i results to decide whether lat-bin mean profile should replace interpolation before any further W45-H region/timing audit.",
    })
    rows.append({
        "diagnostic_item": "latbin_mean_improves_stability",
        "status": "yes" if verdict == "latbin_mean_improves_some_stability" else "no_or_unclear",
        "evidence": verdict,
        "upstream_implication": "If yes, instability was partly caused by missing latitude-bin denoising at the 2-degree construction step.",
        "recommended_followup": "If improvement is substantial, rerun W45-H progress and then reassess whether broader region aggregation is still needed.",
    })
    rows.append({
        "diagnostic_item": "latbin_mean_is_not_enough",
        "status": "yes" if verdict == "latbin_mean_does_not_improve_stability" else "unknown_or_no",
        "evidence": verdict,
        "upstream_implication": "If yes, W45-H instability is not mainly fixed by changing interpolation to latitude-bin mean; inspect progress definition/window/field behavior instead.",
        "recommended_followup": "Do not continue mechanical aggregation unless lat-bin construction first gives a usable baseline.",
    })
    if not construction_cmp.empty and "interp_minus_latbin_mean_abs" in construction_cmp.columns:
        med_abs = pd.to_numeric(construction_cmp["interp_minus_latbin_mean_abs"], errors="coerce").median()
    else:
        med_abs = np.nan
    rows.append({
        "diagnostic_item": "interp_vs_latbin_numeric_difference",
        "status": "reported",
        "evidence": f"median_abs_difference={med_abs}" if np.isfinite(med_abs) else "unavailable",
        "upstream_implication": "Large differences imply that profile-construction choice can materially affect progress timing.",
        "recommended_followup": "Inspect w45_H_profile_construction_comparison_v7_i.csv before interpreting any lat-bin progress result.",
    })
    return pd.DataFrame(rows)


def _plot_heatmap(curves: pd.DataFrame, paths: W45HLatBinPaths) -> None:
    if curves.empty:
        return
    try:
        import matplotlib.pyplot as plt
        piv = curves.pivot_table(index="feature_lat_center", columns="day", values="progress", aggfunc="first")
        piv = piv.sort_index(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", origin="lower", extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
        ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_xlabel("day index")
        ax.set_ylabel("H lat-bin feature center")
        ax.set_title("W45 H lat-bin-mean feature progress")
        fig.colorbar(im, ax=ax, label="progress")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_latbin_progress_heatmap_v7_i.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        _ensure_dir(paths.log_dir)
        (paths.log_dir / "plot_heatmap_error.txt").write_text(str(exc), encoding="utf-8")


def _plot_midpoint(summary: pd.DataFrame, paths: W45HLatBinPaths) -> None:
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
        df = summary.sort_values("feature_lat_center")
        x = pd.to_numeric(df["feature_lat_center"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["midpoint_median"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(df["midpoint_q05"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df["midpoint_q95"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(y - lo, 0), np.maximum(hi - y, 0)])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-")
        ax.axhline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_xlabel("H lat-bin feature center")
        ax.set_ylabel("bootstrap median midpoint day")
        ax.set_title("W45 H lat-bin midpoint by latitude")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_latbin_midpoint_by_lat_v7_i.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        _ensure_dir(paths.log_dir)
        (paths.log_dir / "plot_midpoint_error.txt").write_text(str(exc), encoding="utf-8")


def _write_summary_md(paths: W45HLatBinPaths, construction_cmp: pd.DataFrame, observed: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, upstream: pd.DataFrame, input_audit: dict) -> None:
    lines: list[str] = []
    lines.append("# W45 H lat-bin profile progress audit v7_i")
    lines.append("")
    lines.append("## Purpose")
    lines.append("This audit fixes the upstream implementation question: current 2-degree H features are produced by lon-mean plus latitude interpolation, not by true 2-degree latitude-bin averaging. V7-i tests a lat-bin-mean alternative before any further W45-H region aggregation.")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Window: W002 / anchor day 45")
    lines.append("- Field: H / z500 only")
    lines.append("- Main implementation change: current np.interp 2-degree latitude feature -> true raw-lat bin mean after lon mean")
    lines.append("- No causal/pathway interpretation; no statistical-threshold change.")
    lines.append("")
    if not comparison.empty:
        lines.append("## Unit-level comparison")
        for _, r in comparison.iterrows():
            lines.append(f"- {r.get('unit_type')}: n_units={r.get('n_units')}, median_q90_width={r.get('median_q90_width')}, n_unstable_units={r.get('n_unstable_units')}, dominant_quality={r.get('dominant_progress_quality_label')}")
        verdict = comparison["latbin_vs_interp_verdict"].dropna().iloc[0] if "latbin_vs_interp_verdict" in comparison.columns and comparison["latbin_vs_interp_verdict"].notna().any() else "unavailable"
        lines.append(f"- latbin_vs_interp_verdict: {verdict}")
        lines.append("")
    if not upstream.empty:
        lines.append("## Upstream implications")
        for _, r in upstream.iterrows():
            lines.append(f"- {r.get('diagnostic_item')}: {r.get('status')} — {r.get('upstream_implication')}")
        lines.append("")
    if not summary.empty and "timing_class" in summary.columns:
        lines.append("## Lat-bin timing classes")
        counts = summary["timing_class"].astype(str).value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("## Input audit")
    for k, v in input_audit.get("checks", {}).items():
        lines.append(f"- {k}: {v}")
    (paths.output_dir / "w45_H_latbin_profile_progress_summary_v7_i.md").write_text("\n".join(lines), encoding="utf-8")


def _input_audit(paths: W45HLatBinPaths, settings: StagePartitionV7Settings, window: dict) -> dict:
    return {
        "created_at": _now_iso(),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "output_tag": OUTPUT_TAG,
        "input_paths": {
            "smoothed_fields": str(settings.foundation.smoothed_fields_path()),
            "v7e_output_dir": str(paths.v7e_output_dir),
            "v7f_output_dir": str(paths.v7f_output_dir),
        },
        "window": window,
        "checks": {
            "smoothed_fields_exists": settings.foundation.smoothed_fields_path().exists(),
            "v7e_output_dir_exists": paths.v7e_output_dir.exists(),
            "v7f_output_dir_exists": paths.v7f_output_dir.exists(),
            "window_source": window.get("source", "unknown"),
            "implementation_change": "latbin_mean_after_lon_mean_replaces_np_interp_lat_sampling_for_H_profile",
        },
    }


def run_w45_H_latbin_profile_progress_v7_i(v7_root: Path | None = None) -> None:
    paths = _resolve_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)
    settings = _configure_settings(paths)

    meta = {
        "status": "running",
        "created_at": _now_iso(),
        "output_tag": OUTPUT_TAG,
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "does_not_modify_v7e": True,
        "does_not_use_region_aggregation": True,
        "purpose": "test whether replacing current 2-degree latitude interpolation with true 2-degree latitude-bin mean improves W45-H progress stability",
    }
    try:
        smoothed = load_smoothed_fields(settings.foundation.smoothed_fields_path())
        current_cube, current_lats = _build_current_interp_h_profile(smoothed, settings)
        latbin_cube, latbin_lats, bin_map, raw_lats = _build_latbin_h_profile(smoothed, settings)
        n_years, n_days, n_feat = latbin_cube.shape
        window = _load_w45_window(paths, n_days)
        input_audit = _input_audit(paths, settings, window)
        input_audit["raw_lat_range_summary"] = {
            "n_raw_lat_in_H_range": int(len(raw_lats)),
            "raw_lat_min": float(np.nanmin(raw_lats)),
            "raw_lat_max": float(np.nanmax(raw_lats)),
            "latbin_feature_count": int(n_feat),
        }
        _write_json(input_audit, paths.output_dir / "input_audit_v7_i.json")

        construction_cmp = _profile_construction_comparison(current_cube, current_lats, latbin_cube, latbin_lats, bin_map)
        _write_csv(bin_map, paths.output_dir / "w45_H_latbin_feature_provenance_v7_i.csv")
        _write_csv(construction_cmp, paths.output_dir / "w45_H_profile_construction_comparison_v7_i.csv")

        observed_seasonal = _seasonal_mean(latbin_cube)
        observed_rows: list[dict] = []
        curve_frames: list[pd.DataFrame] = []
        for j, center in enumerate(latbin_lats):
            br = bin_map.iloc[j]
            unit_meta = {
                "profile_construction": "latbin_mean_after_lon_mean",
                "feature_index": int(j),
                "feature_lat_center": float(center),
                "lat_bin_left": float(br["lat_bin_left"]),
                "lat_bin_right": float(br["lat_bin_right"]),
                "n_raw_lat_points": int(br["n_raw_lat_points"]),
            }
            row, curves = _progress_for_unit(observed_seasonal, [j], unit_meta, window, settings)
            observed_rows.append(row)
            if not curves.empty:
                curve_frames.append(curves)
        observed = pd.DataFrame(observed_rows)
        curves_long = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        _write_csv(observed, paths.output_dir / "w45_H_latbin_progress_observed_v7_i.csv")
        _write_csv(curves_long, paths.output_dir / "w45_H_latbin_progress_curves_long_v7_i.csv")

        boot_indices = _load_or_make_bootstrap_indices(n_years, paths, settings)
        boot_rows: list[dict] = []
        for b, idx in enumerate(boot_indices):
            seas = _seasonal_mean(latbin_cube, idx)
            for j, center in enumerate(latbin_lats):
                br = bin_map.iloc[j]
                unit_meta = {
                    "bootstrap_id": int(b),
                    "profile_construction": "latbin_mean_after_lon_mean",
                    "feature_index": int(j),
                    "feature_lat_center": float(center),
                    "lat_bin_left": float(br["lat_bin_left"]),
                    "lat_bin_right": float(br["lat_bin_right"]),
                    "n_raw_lat_points": int(br["n_raw_lat_points"]),
                }
                row, _ = _progress_for_unit(seas, [j], unit_meta, window, settings)
                row["sampled_year_indices"] = ";".join(str(int(x)) for x in np.asarray(idx, dtype=int).tolist())
                boot_rows.append(row)
        boot = pd.DataFrame(boot_rows)
        _write_csv(boot, paths.output_dir / "w45_H_latbin_progress_bootstrap_samples_v7_i.csv")

        summary = _summarize_samples(boot, ["feature_index", "feature_lat_center", "lat_bin_left", "lat_bin_right", "n_raw_lat_points"])
        summary = _assign_bin_timing_classes(summary)
        _write_csv(summary, paths.output_dir / "w45_H_latbin_progress_bootstrap_summary_v7_i.csv")
        _write_csv(summary[[c for c in ["feature_index", "feature_lat_center", "lat_bin_left", "lat_bin_right", "midpoint_median", "midpoint_q05", "midpoint_q95", "midpoint_q90_width", "dominant_progress_quality_label", "timing_class", "timing_class_note"] if c in summary.columns]], paths.output_dir / "w45_H_latbin_timing_spread_v7_i.csv")

        comparison = _build_comparison_table(paths, summary, bin_map)
        _write_csv(comparison, paths.output_dir / "w45_H_interp_vs_latbin_progress_comparison_v7_i.csv")
        upstream = _build_upstream_implication(comparison, construction_cmp)
        _write_csv(upstream, paths.output_dir / "w45_H_latbin_upstream_implication_v7_i.csv")

        _plot_heatmap(curves_long, paths)
        _plot_midpoint(summary, paths)
        _write_summary_md(paths, construction_cmp, observed, summary, comparison, upstream, input_audit)

        meta.update({
            "status": "success",
            "finished_at": _now_iso(),
            "n_years": int(n_years),
            "n_days": int(n_days),
            "n_latbin_features": int(n_feat),
            "n_bootstrap": int(len(boot_indices)),
            "outputs": {
                "feature_provenance": str(paths.output_dir / "w45_H_latbin_feature_provenance_v7_i.csv"),
                "construction_comparison": str(paths.output_dir / "w45_H_profile_construction_comparison_v7_i.csv"),
                "observed": str(paths.output_dir / "w45_H_latbin_progress_observed_v7_i.csv"),
                "bootstrap_summary": str(paths.output_dir / "w45_H_latbin_progress_bootstrap_summary_v7_i.csv"),
                "interp_vs_latbin_comparison": str(paths.output_dir / "w45_H_interp_vs_latbin_progress_comparison_v7_i.csv"),
                "upstream_implication": str(paths.output_dir / "w45_H_latbin_upstream_implication_v7_i.csv"),
            },
            "notes": [
                "This run directly audits the upstream implementation issue: current 2-degree profile construction is interpolation, not true latitude-bin averaging.",
                "V7-i replaces only the H profile construction for this W45 diagnostic with lat-bin mean after lon mean.",
                "This run does not change V7-e, V7-e1, V7-e2, V7-f, V7-g, or V7-h outputs.",
                "Do not interpret lat-bin progress as causal/pathway evidence.",
            ],
        })
    except Exception as exc:
        meta.update({"status": "failed", "failed_at": _now_iso(), "error": repr(exc)})
        _write_json(meta, paths.output_dir / "run_meta.json")
        raise
    _write_json(meta, paths.output_dir / "run_meta.json")
    _write_json(meta, paths.log_dir / "run_meta.json")
    log_text = "\n".join([
        "# W45 H lat-bin profile progress v7_i",
        "",
        "This run tests the implementation-level correction from latitude interpolation to latitude-bin mean.",
        "It is not a new field-order claim and does not modify V7-e outputs.",
        "",
        f"Status: {meta['status']}",
    ])
    (paths.log_dir / "w45_H_latbin_profile_progress_v7_i.md").write_text(log_text, encoding="utf-8")


__all__ = ["run_w45_H_latbin_profile_progress_v7_i"]
