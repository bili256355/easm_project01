from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from stage_partition_v6.io import load_smoothed_fields

from .config import StagePartitionV7Settings

OUTPUT_TAG = "w45_H_Jw_raw025_process_audit_v7_s"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELDS = ("H", "Jw")
FIELD_SPECS = {
    "H": {"field_key": "z500_smoothed", "domain_attr_lon": "h_lon_range", "domain_attr_lat": "h_lat_range"},
    "Jw": {"field_key": "u200_smoothed", "domain_attr_lon": "jw_lon_range", "domain_attr_lat": "jw_lat_range"},
}
THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
EPS = 1e-12


@dataclass
class V7SPaths:
    v7_root: Path
    project_root: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path
    v7e_output_dir: Path
    v7p_output_dir: Path
    v7q_output_dir: Path
    v7r_output_dir: Path


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


def _q(values: np.ndarray | list[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanquantile(arr, q)) if arr.size else np.nan


def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(float(lower), float(upper)), max(float(lower), float(upper))
    return (arr >= lo) & (arr <= hi)


def _resolve_paths(v7_root: Optional[Path]) -> V7SPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = v7_root.parents[1]
    return V7SPaths(
        v7_root=v7_root,
        project_root=project_root,
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7p_output_dir=v7_root / "outputs" / "w45_process_relation_rebuild_v7_p",
        v7q_output_dir=v7_root / "outputs" / "w45_feature_process_resolution_v7_q",
        v7r_output_dir=v7_root / "outputs" / "w45_H_Jw_feature_relation_exploration_v7_r",
    )


def _configure_settings(paths: V7SPaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    # Optional debug override for development only. The default remains the project
    # bootstrap setting, normally 1000. Use this only for local smoke/debug runs:
    #   set V7S_DEBUG_N_BOOTSTRAP=20
    env_debug = os.environ.get("V7S_DEBUG_N_BOOTSTRAP", "").strip()
    if env_debug:
        settings.bootstrap.debug_n_bootstrap = int(env_debug)
    return settings


def _progress_log(paths: V7SPaths, message: str) -> None:
    _ensure_dir(paths.log_dir)
    log_path = paths.log_dir / "run_progress_v7_s.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_now_iso()}] {message}\n")
    print(f"[V7-s] {message}", flush=True)


def _load_w45_window(paths: V7SPaths, n_days: int) -> dict[str, Any]:
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
        "source": "fallback_anchor_pm15",
    }


def _axis_index(lat_or_lon: np.ndarray, value_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(lat_or_lon, dtype=float)
    mask = _mask_between(values, *value_range)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No coordinate values in requested range {value_range}")
    coord_unsorted = values[idx]
    order = np.argsort(coord_unsorted)
    idx_sorted = idx[order]
    coord_sorted = coord_unsorted[order]
    return idx_sorted, coord_sorted, order


def _prepare_raw_field(smoothed: dict[str, Any], field: str, settings: StagePartitionV7Settings) -> dict[str, Any]:
    spec = FIELD_SPECS[field]
    field_key = str(spec["field_key"])
    raw = np.asarray(smoothed[field_key], dtype=float)
    if raw.ndim != 4:
        raise ValueError(f"Expected {field_key} shape years x days x lat x lon; got {raw.shape}")
    lat = np.asarray(smoothed["lat"], dtype=float)
    lon = np.asarray(smoothed["lon"], dtype=float)
    lat_range = tuple(getattr(settings.profile, str(spec["domain_attr_lat"])))
    lon_range = tuple(getattr(settings.profile, str(spec["domain_attr_lon"])))
    lat_idx, lats, _ = _axis_index(lat, lat_range)
    lon_idx, lons, _ = _axis_index(lon, lon_range)
    subset = raw[:, :, lat_idx, :][:, :, :, lon_idx]
    weights = np.cos(np.deg2rad(lats))[:, None] * np.ones((len(lats), len(lons)), dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return {
        "field": field,
        "field_key": field_key,
        "data": np.asarray(subset, dtype=float),
        "lat": np.asarray(lats, dtype=float),
        "lon": np.asarray(lons, dtype=float),
        "lat_indices_original": np.asarray(lat_idx, dtype=int),
        "lon_indices_original": np.asarray(lon_idx, dtype=int),
        "lat_range_requested": lat_range,
        "lon_range_requested": lon_range,
        "weights": weights,
        "n_years": int(subset.shape[0]),
        "n_days": int(subset.shape[1]),
    }


def _weighted_sum(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(v) & np.isfinite(w)
    if not good.any():
        return np.nan
    return float(np.nansum(v[good] * w[good]))


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(v) & np.isfinite(w)
    if not good.any():
        return np.nan
    den = float(np.nansum(w[good]))
    if den <= EPS:
        return np.nan
    return float(np.sqrt(max(np.nansum((v[good] ** 2) * w[good]) / den, 0.0)))


def _clip_progress(x: float, settings: StagePartitionV7Settings) -> float:
    if not np.isfinite(x):
        return np.nan
    cfg = settings.progress_timing
    return float(np.clip(x, float(cfg.progress_clip_min), float(cfg.progress_clip_max)))


def _days_from_window(window: dict[str, Any]) -> np.ndarray:
    return np.arange(int(window["analysis_window_start"]), int(window["analysis_window_end"]) + 1, dtype=int)


def _period_days(window: dict[str, Any], prefix: str) -> np.ndarray:
    return np.arange(int(window[f"{prefix}_period_start"]), int(window[f"{prefix}_period_end"]) + 1, dtype=int)


def _region_progress_from_avg_state(
    avg_state: np.ndarray,
    weights_2d: np.ndarray,
    mask: np.ndarray,
    window: dict[str, Any],
    settings: StagePartitionV7Settings,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(avg_state, dtype=float)
    days = _days_from_window(window)
    pre_days = _period_days(window, "pre")
    post_days = _period_days(window, "post")
    if np.any(days < 0) or np.any(days >= state.shape[0]):
        raise ValueError(f"Analysis days {days[0]}-{days[-1]} outside state length {state.shape[0]}")
    if np.any(pre_days < 0) or np.any(post_days >= state.shape[0]):
        raise ValueError("Pre/post days outside state length")
    m = np.asarray(mask, dtype=bool)
    if m.shape != state.shape[1:]:
        raise ValueError(f"Mask shape {m.shape} does not match state spatial shape {state.shape[1:]}")
    if not m.any():
        empty = pd.DataFrame()
        return empty, {}, np.empty(0), np.empty(0), np.empty(0)
    w = np.asarray(weights_2d, dtype=float)
    # Compact the spatial domain immediately. This avoids recomputing full-field
    # pre/post arrays for every diagnostic region and keeps the 1000-bootstrap
    # run tractable on raw025 grids.
    flat_mask = m.ravel()
    state_region = state.reshape(state.shape[0], -1)[:, flat_mask]
    w_region = w.ravel()[flat_mask]
    pre = _safe_nanmean(state_region[pre_days, :], axis=0)
    post = _safe_nanmean(state_region[post_days, :], axis=0)
    tv = post - pre
    denom = _weighted_sum(tv * tv, w_region)
    raw_vals: list[float] = []
    clipped_vals: list[float] = []
    dist_pre: list[float] = []
    dist_post: list[float] = []
    for d in days:
        cur = state_region[d, :]
        if not np.isfinite(denom) or abs(denom) < float(settings.progress_timing.min_transition_norm):
            raw = np.nan
        else:
            raw = _weighted_sum((cur - pre) * tv, w_region) / denom
        raw_vals.append(float(raw) if np.isfinite(raw) else np.nan)
        clipped_vals.append(_clip_progress(raw, settings))
        dist_pre.append(_weighted_rms(cur - pre, w_region))
        dist_post.append(_weighted_rms(cur - post, w_region))
    raw_arr = np.asarray(raw_vals, dtype=float)
    clip_arr = np.asarray(clipped_vals, dtype=float)
    daily_delta = np.r_[np.nan, np.diff(raw_arr)]
    base = pd.DataFrame({
        "day": days.astype(int),
        "progress_raw": raw_arr,
        "progress_clipped_0_1": clip_arr,
        "distance_to_pre": np.asarray(dist_pre, dtype=float),
        "distance_to_post": np.asarray(dist_post, dtype=float),
        "daily_delta_progress": daily_delta,
    })
    pre_var = float(np.nanmean([_weighted_rms(state_region[d, :] - pre, w_region) for d in pre_days]))
    post_var = float(np.nanmean([_weighted_rms(state_region[d, :] - post, w_region) for d in post_days]))
    transition_norm = _weighted_rms(tv, w_region)
    sep_ratio = float(transition_norm / (pre_var + post_var)) if np.isfinite(transition_norm) and np.isfinite(pre_var + post_var) and (pre_var + post_var) > EPS else np.nan
    meta = {
        "n_cells": int(m.sum()),
        "transition_norm": transition_norm,
        "within_pre_variability": pre_var,
        "within_post_variability": post_var,
        "separation_ratio": sep_ratio,
        "pre_post_separation_label": _separation_label(sep_ratio, settings),
    }
    return base, meta, pre, post, tv


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


def _first_stable_crossing(days: np.ndarray, vals: np.ndarray, threshold: float, stable_days: int) -> float:
    days = np.asarray(days, dtype=int)
    vals = np.asarray(vals, dtype=float)
    stable_days = max(1, int(stable_days))
    if vals.size < stable_days:
        return np.nan
    for i in range(0, vals.size - stable_days + 1):
        win = vals[i : i + stable_days]
        if np.all(np.isfinite(win)) and np.all(win >= float(threshold)):
            return float(days[i])
    return np.nan


def _count_threshold_crossings(vals: np.ndarray, threshold: float) -> int:
    vals = np.asarray(vals, dtype=float)
    good = np.isfinite(vals)
    if good.sum() < 2:
        return 0
    above = vals[good] >= threshold
    return int(np.sum(above[1:] != above[:-1]))


def _markers_from_curve(curves: pd.DataFrame, window: dict[str, Any], settings: StagePartitionV7Settings) -> dict[str, Any]:
    days = curves["day"].to_numpy(dtype=int)
    vals = curves["progress_clipped_0_1"].to_numpy(dtype=float)
    raw = curves["progress_raw"].to_numpy(dtype=float)
    pre_mask = (days >= int(window["pre_period_start"])) & (days <= int(window["pre_period_end"]))
    out: dict[str, Any] = {}
    if pre_mask.any():
        pre_vals = vals[pre_mask]
        out["departure90"] = _first_stable_crossing(days, vals, _q(pre_vals, 0.90), settings.progress_timing.stable_crossing_days)
        out["departure95"] = _first_stable_crossing(days, vals, _q(pre_vals, 0.95), settings.progress_timing.stable_crossing_days)
    else:
        out["departure90"] = np.nan
        out["departure95"] = np.nan
    for thr in THRESHOLDS:
        key = f"t{int(round(thr * 100)):02d}" if thr < 1.0 else f"t{int(round(thr * 100))}"
        out[key] = _first_stable_crossing(days, vals, thr, settings.progress_timing.stable_crossing_days)
    delta = np.r_[np.nan, np.diff(raw)]
    if np.isfinite(delta).any():
        out["peak_raw"] = float(days[int(np.nanargmax(delta))])
        if len(delta) >= 3:
            smooth = pd.Series(delta).rolling(3, center=True, min_periods=1).mean().to_numpy(dtype=float)
            out["peak_smooth3"] = float(days[int(np.nanargmax(smooth))])
        else:
            out["peak_smooth3"] = out["peak_raw"]
    else:
        out["peak_raw"] = np.nan
        out["peak_smooth3"] = np.nan
    t25 = out.get("t25", np.nan)
    t50 = out.get("t50", np.nan)
    t75 = out.get("t75", np.nan)
    out["duration_25_75"] = float(t75 - t25) if np.isfinite(t25) and np.isfinite(t75) else np.nan
    out["tail_50_75"] = float(t75 - t50) if np.isfinite(t50) and np.isfinite(t75) else np.nan
    out["early_span_25_50"] = float(t50 - t25) if np.isfinite(t25) and np.isfinite(t50) else np.nan
    out["n_crossings_025"] = _count_threshold_crossings(vals, 0.25)
    out["n_crossings_050"] = _count_threshold_crossings(vals, 0.50)
    out["n_crossings_075"] = _count_threshold_crossings(vals, 0.75)
    valid = np.isfinite(vals)
    if valid.sum() >= 3 and np.nanstd(vals[valid]) > EPS:
        out["progress_monotonicity_corr"] = float(np.corrcoef(days[valid], vals[valid])[0, 1])
    else:
        out["progress_monotonicity_corr"] = np.nan
    return out


def _detect_retreat(curves: pd.DataFrame, anchor_day: int, post_start: int) -> dict[str, Any]:
    df = curves.sort_values("day").copy()
    days = df["day"].to_numpy(dtype=int)
    vals = df["progress_raw"].to_numpy(dtype=float)
    good = np.isfinite(vals)
    if good.sum() < 3:
        return {"retreat_detected": False, "retreat_start_day": np.nan, "retreat_end_day": np.nan, "retreat_drop": np.nan, "retreat_label": "insufficient_curve"}
    # Start from the highest H progress before or near anchor; this preserves the user's observed day43-ish maximum.
    eligible_high = good & (days <= int(anchor_day))
    if not eligible_high.any():
        eligible_high = good & (days <= int(anchor_day) + 2)
    if not eligible_high.any():
        return {"retreat_detected": False, "retreat_start_day": np.nan, "retreat_end_day": np.nan, "retreat_drop": np.nan, "retreat_label": "no_pre_anchor_high"}
    high_pos = np.where(eligible_high)[0][int(np.nanargmax(vals[eligible_high]))]
    eligible_low = good & (np.arange(vals.size) > high_pos) & (days <= int(post_start))
    if not eligible_low.any():
        return {"retreat_detected": False, "retreat_start_day": float(days[high_pos]), "retreat_end_day": np.nan, "retreat_drop": np.nan, "retreat_label": "no_post_high_low_candidate"}
    low_candidates = np.where(eligible_low)[0]
    low_pos = low_candidates[int(np.nanargmin(vals[low_candidates]))]
    drop = float(vals[low_pos] - vals[high_pos])
    span = int(days[low_pos] - days[high_pos])
    detected = bool(drop < 0 and span >= 1)
    if detected and span >= 2:
        label = "multi_day_retreat_candidate"
    elif detected:
        label = "one_day_retreat_candidate"
    else:
        label = "no_retreat"
    return {
        "retreat_detected": detected,
        "retreat_start_day": float(days[high_pos]),
        "retreat_end_day": float(days[low_pos]),
        "retreat_start_progress": float(vals[high_pos]),
        "retreat_end_progress": float(vals[low_pos]),
        "retreat_drop": drop,
        "retreat_span_days": span,
        "retreat_label": label,
    }


def _detect_crossings(h_curves: pd.DataFrame, jw_curves: pd.DataFrame, window: dict[str, Any]) -> pd.DataFrame:
    h = h_curves[["day", "progress_raw", "daily_delta_progress"]].rename(columns={"progress_raw": "H_progress", "daily_delta_progress": "H_daily_delta"})
    j = jw_curves[["day", "progress_raw", "daily_delta_progress"]].rename(columns={"progress_raw": "Jw_progress", "daily_delta_progress": "Jw_daily_delta"})
    df = h.merge(j, on="day", how="inner").sort_values("day")
    if df.empty:
        return pd.DataFrame()
    df["H_minus_Jw"] = df["H_progress"] - df["Jw_progress"]
    rows: list[dict[str, Any]] = []
    arr = df.to_dict("records")
    for i in range(1, len(arr)):
        prev = arr[i - 1]
        cur = arr[i]
        d0 = prev["H_minus_Jw"]
        d1 = cur["H_minus_Jw"]
        if not np.isfinite(d0) or not np.isfinite(d1):
            continue
        if d0 > 0 and d1 <= 0:
            if cur["H_daily_delta"] <= 0 and cur["Jw_daily_delta"] > 0:
                interp = "H_retreat_with_Jw_overtake"
            elif cur["day"] >= int(window["post_period_start"]):
                interp = "late_convergence_crossing"
            else:
                interp = "H_to_Jw_progress_crossing"
            rows.append({
                "crossing_type": "H_to_Jw_progress_crossing",
                "first_crossing_day": int(cur["day"]),
                "previous_day": int(prev["day"]),
                "H_progress_before": prev["H_progress"],
                "Jw_progress_before": prev["Jw_progress"],
                "H_progress_after": cur["H_progress"],
                "Jw_progress_after": cur["Jw_progress"],
                "H_daily_change_at_crossing": cur["H_daily_delta"],
                "Jw_daily_change_at_crossing": cur["Jw_daily_delta"],
                "H_minus_Jw_before": d0,
                "H_minus_Jw_after": d1,
                "interpretation_label": interp,
            })
        elif d0 < 0 and d1 >= 0:
            rows.append({
                "crossing_type": "Jw_to_H_progress_crossing",
                "first_crossing_day": int(cur["day"]),
                "previous_day": int(prev["day"]),
                "H_progress_before": prev["H_progress"],
                "Jw_progress_before": prev["Jw_progress"],
                "H_progress_after": cur["H_progress"],
                "Jw_progress_after": cur["Jw_progress"],
                "H_daily_change_at_crossing": cur["H_daily_delta"],
                "Jw_daily_change_at_crossing": cur["Jw_daily_delta"],
                "H_minus_Jw_before": d0,
                "H_minus_Jw_after": d1,
                "interpretation_label": "Jw_to_H_progress_crossing",
            })
    return pd.DataFrame(rows)


def _grid_progress(avg_state: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    tv = post - pre
    out = np.full_like(avg_state, np.nan, dtype=float)
    good = np.isfinite(tv) & (np.abs(tv) > EPS)
    if good.any():
        out[:, good] = (avg_state[:, good] - pre[good][None, :]) / tv[good][None, :]
    return out


def _day_index(days: np.ndarray, day: int) -> int:
    where = np.where(days == int(day))[0]
    if where.size == 0:
        # nearest day fallback remains explicit in output via actual_start/end_day.
        return int(np.argmin(np.abs(days - int(day))))
    return int(where[0])


def _amplitude_label(abs_amp: float, q50: float, q75: float) -> str:
    if not np.isfinite(abs_amp):
        return "invalid_amplitude"
    if abs_amp <= EPS:
        return "zero_or_tiny_amplitude"
    if np.isfinite(q75) and abs_amp >= q75:
        return "high_transition_amplitude"
    if np.isfinite(q50) and abs_amp >= q50:
        return "moderate_transition_amplitude"
    return "low_transition_amplitude"


def _contribution_label(value: float, q25: float, q75: float, phenomenon: str) -> str:
    if not np.isfinite(value):
        return "invalid_contribution"
    if value > 0 and np.isfinite(q75) and value >= q75:
        return f"{phenomenon}_positive_core"
    if value < 0 and np.isfinite(q25) and value <= q25:
        return f"{phenomenon}_negative_core"
    if value > 0:
        return f"{phenomenon}_positive"
    if value < 0:
        return f"{phenomenon}_negative"
    return f"{phenomenon}_near_zero"


def _contribution_map(
    field_data: dict[str, Any],
    avg_state: np.ndarray,
    window: dict[str, Any],
    start_day: int,
    end_day: int,
    phenomenon: str,
    settings: StagePartitionV7Settings,
) -> pd.DataFrame:
    days = np.arange(avg_state.shape[0], dtype=int)
    pre = _safe_nanmean(avg_state[_period_days(window, "pre")], axis=0)
    post = _safe_nanmean(avg_state[_period_days(window, "post")], axis=0)
    tv = post - pre
    progress = _grid_progress(avg_state, pre, post)
    si = _day_index(days, start_day)
    ei = _day_index(days, end_day)
    contrib = progress[ei] - progress[si]
    lat = np.asarray(field_data["lat"], dtype=float)
    lon = np.asarray(field_data["lon"], dtype=float)
    abs_amp = np.abs(tv)
    valid_amp = abs_amp[np.isfinite(abs_amp) & (abs_amp > EPS)]
    q50 = _q(valid_amp, 0.50)
    q75 = _q(valid_amp, 0.75)
    valid_contrib = contrib[np.isfinite(contrib)]
    cq25 = _q(valid_contrib, 0.25)
    cq75 = _q(valid_contrib, 0.75)
    rows: list[dict[str, Any]] = []
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            c = float(contrib[i, j]) if np.isfinite(contrib[i, j]) else np.nan
            amp = float(abs_amp[i, j]) if np.isfinite(abs_amp[i, j]) else np.nan
            rows.append({
                "field": field_data["field"],
                "phenomenon": phenomenon,
                "interval_start_day_requested": int(start_day),
                "interval_end_day_requested": int(end_day),
                "actual_start_day": int(days[si]),
                "actual_end_day": int(days[ei]),
                "lat": float(la),
                "lon": float(lo),
                "grid_i": int(i),
                "grid_j": int(j),
                "transition_amplitude": float(tv[i, j]) if np.isfinite(tv[i, j]) else np.nan,
                "abs_transition_amplitude": amp,
                "amplitude_label": _amplitude_label(amp, q50, q75),
                "progress_start": float(progress[si, i, j]) if np.isfinite(progress[si, i, j]) else np.nan,
                "progress_end": float(progress[ei, i, j]) if np.isfinite(progress[ei, i, j]) else np.nan,
                "contribution": c,
                "contribution_label": _contribution_label(c, cq25, cq75, phenomenon),
                "interpretation_boundary": "raw-grid contribution; requires spatial coherence and bootstrap before physical-region interpretation",
            })
    return pd.DataFrame(rows)


def _make_mask(shape: tuple[int, int], indices: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for i, j in indices:
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            mask[i, j] = True
    return mask


def _region_definitions(
    field_data: dict[str, Any],
    contribution_tables: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    field = str(field_data["field"])
    lats = np.asarray(field_data["lat"], dtype=float)
    lons = np.asarray(field_data["lon"], dtype=float)
    shape = (len(lats), len(lons))
    regions: list[dict[str, Any]] = []
    # whole field
    regions.append({
        "field": field,
        "region_scheme": "whole_field",
        "region_id": f"{field}_whole",
        "region_label": "whole_field",
        "mask": np.ones(shape, dtype=bool),
        "bootstrap_eligible": True,
        "region_construction_note": "all raw grid cells in object domain",
    })
    # raw latitude bands, each latitude row across all longitudes.
    for i, la in enumerate(lats):
        mask = np.zeros(shape, dtype=bool)
        mask[i, :] = True
        regions.append({
            "field": field,
            "region_scheme": "raw_latband",
            "region_id": f"{field}_lat_{i:03d}",
            "region_label": f"lat_{la:.3f}",
            "mask": mask,
            "bootstrap_eligible": False,
            "region_construction_note": "one raw latitude row averaged over object longitude domain; observed curves only by default to avoid excessive bootstrap cost",
        })
    # equal-count three latitude regions.
    labels = ["R1_low", "R2_mid", "R3_high"]
    for ridx, inds in enumerate(np.array_split(np.arange(len(lats)), 3)):
        mask = np.zeros(shape, dtype=bool)
        mask[inds, :] = True
        regions.append({
            "field": field,
            "region_scheme": "equal_count_three_region",
            "region_id": f"{field}_three_{ridx}",
            "region_label": labels[ridx],
            "mask": mask,
            "bootstrap_eligible": True,
            "region_construction_note": "three equal-count raw latitude chunks after sorting latitude ascending",
        })
    # contribution-defined regions: deliberately post-processing diagnostic masks.
    for phenomenon, tbl in contribution_tables.items():
        if tbl.empty or "field" not in tbl or field not in set(tbl["field"].astype(str)):
            continue
        sub = tbl[tbl["field"].astype(str) == field].copy()
        if sub.empty:
            continue
        for suffix, selector, note in [
            ("positive_core", sub["contribution"] > 0, "positive contribution cells"),
            ("negative_core", sub["contribution"] < 0, "negative contribution cells"),
        ]:
            cand = sub.loc[selector & np.isfinite(sub["contribution"])].copy()
            if cand.empty:
                continue
            if "positive" in suffix:
                cutoff = _q(cand["contribution"].to_numpy(dtype=float), 0.75)
                core = cand[cand["contribution"] >= cutoff]
            else:
                cutoff = _q(cand["contribution"].to_numpy(dtype=float), 0.25)
                core = cand[cand["contribution"] <= cutoff]
            if core.empty:
                continue
            inds = [(int(r.grid_i), int(r.grid_j)) for r in core.itertuples()]
            mask = _make_mask(shape, inds)
            if not mask.any():
                continue
            regions.append({
                "field": field,
                "region_scheme": "contribution_region",
                "region_id": f"{field}_{phenomenon}_{suffix}",
                "region_label": f"{phenomenon}_{suffix}",
                "mask": mask,
                "bootstrap_eligible": True,
                "region_construction_note": f"post-processing diagnostic mask from {phenomenon}: {note}; not a physical region until independently justified",
            })
    return regions


def _region_row_from_mask(field_data: dict[str, Any], reg: dict[str, Any]) -> dict[str, Any]:
    mask = np.asarray(reg["mask"], dtype=bool)
    lats = np.asarray(field_data["lat"], dtype=float)
    lons = np.asarray(field_data["lon"], dtype=float)
    lat_idx, lon_idx = np.where(mask)
    lat_vals = lats[lat_idx] if lat_idx.size else np.asarray([], dtype=float)
    lon_vals = lons[lon_idx] if lon_idx.size else np.asarray([], dtype=float)
    return {
        "field": reg["field"],
        "region_scheme": reg["region_scheme"],
        "region_id": reg["region_id"],
        "region_label": reg["region_label"],
        "lat_min": float(np.nanmin(lat_vals)) if lat_vals.size else np.nan,
        "lat_max": float(np.nanmax(lat_vals)) if lat_vals.size else np.nan,
        "lon_min": float(np.nanmin(lon_vals)) if lon_vals.size else np.nan,
        "lon_max": float(np.nanmax(lon_vals)) if lon_vals.size else np.nan,
        "n_grid_cells": int(mask.sum()),
        "bootstrap_eligible": bool(reg.get("bootstrap_eligible", True)),
        "region_construction_note": str(reg.get("region_construction_note", "")),
    }


def _phase_label(day: int, window: dict[str, Any], h_retreat: Optional[dict[str, Any]] = None) -> str:
    if day < int(window["pre_period_start"]):
        return "pre_window"
    if int(window["pre_period_start"]) <= day <= int(window["pre_period_end"]):
        return "pre_period"
    if day < int(window["anchor_day"]):
        return "early_progress"
    if abs(day - int(window["anchor_day"])) <= 1:
        return "anchor_near"
    if h_retreat and h_retreat.get("retreat_detected") and np.isfinite(h_retreat.get("retreat_start_day", np.nan)) and np.isfinite(h_retreat.get("retreat_end_day", np.nan)):
        if int(h_retreat["retreat_start_day"]) <= day <= int(h_retreat["retreat_end_day"]):
            return "retreat_or_crossing"
    if day < int(window["post_period_start"]):
        return "late_recovery_or_catchup"
    if int(window["post_period_start"]) <= day <= int(window["post_period_end"]):
        return "post_period"
    return "post_window"


def _make_observed_outputs(
    raw_fields: dict[str, dict[str, Any]],
    window: dict[str, Any],
    settings: StagePartitionV7Settings,
    paths: V7SPaths,
) -> dict[str, Any]:
    avg_states: dict[str, np.ndarray] = {f: _safe_nanmean(raw_fields[f]["data"], axis=0) for f in FIELDS}
    whole_curves: dict[str, pd.DataFrame] = {}
    whole_markers: dict[str, Any] = {}
    prepost: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    whole_region_rows: list[dict[str, Any]] = []
    for field in FIELDS:
        _progress_log(paths, f"observed whole-field progress: {field}")
        fd = raw_fields[field]
        mask = np.ones(fd["weights"].shape, dtype=bool)
        curves, meta, pre, post, tv = _region_progress_from_avg_state(avg_states[field], fd["weights"], mask, window, settings)
        curves.insert(0, "field", field)
        curves["phase_label"] = [_phase_label(int(d), window) for d in curves["day"]]
        whole_curves[field] = curves
        markers = _markers_from_curve(curves, window, settings)
        whole_markers[field] = {**markers, **meta}
        prepost[field] = (pre, post, tv)
        whole_region_rows.append({
            "field": field,
            "region_scheme": "whole_field",
            "region_id": f"{field}_whole",
            "region_label": "whole_field",
            **meta,
            **markers,
        })
    h_retreat = _detect_retreat(whole_curves["H"], int(window["anchor_day"]), int(window["post_period_start"]))
    for field in FIELDS:
        whole_curves[field]["phase_label"] = [_phase_label(int(d), window, h_retreat) for d in whole_curves[field]["day"]]
    crossing_df = _detect_crossings(whole_curves["H"], whole_curves["Jw"], window)
    whole_df = pd.concat([whole_curves[f] for f in FIELDS], ignore_index=True)
    _write_csv(whole_df, paths.output_dir / "w45_H_Jw_raw025_wholefield_progress_curves_v7_s.csv")
    _write_csv(crossing_df, paths.output_dir / "w45_H_Jw_raw025_crossing_events_v7_s.csv")
    # Contribution intervals. H uses detected retreat if available; otherwise the documented day43-48 audit window.
    retreat_start = int(h_retreat["retreat_start_day"]) if h_retreat.get("retreat_detected") and np.isfinite(h_retreat.get("retreat_start_day", np.nan)) else 43
    retreat_end = int(h_retreat["retreat_end_day"]) if h_retreat.get("retreat_detected") and np.isfinite(h_retreat.get("retreat_end_day", np.nan)) else 48
    intervals = {
        "H_early_frontloaded": ("H", 35, retreat_start),
        "H_retreat": ("H", retreat_start, retreat_end),
        "H_late_recovery": ("H", retreat_end, 56),
        "Jw_catchup": ("Jw", max(43, retreat_start), 53),
    }
    contribution_tables: dict[str, pd.DataFrame] = {}
    for phenomenon, (field, start, end) in intervals.items():
        _progress_log(paths, f"contribution map: {phenomenon} day{start}-{end}")
        df = _contribution_map(raw_fields[field], avg_states[field], window, start, end, phenomenon, settings)
        contribution_tables[phenomenon] = df
        out_name = {
            "H_early_frontloaded": "w45_H_raw025_early_contribution_map_v7_s.csv",
            "H_retreat": "w45_H_raw025_retreat_contribution_map_v7_s.csv",
            "H_late_recovery": "w45_H_raw025_late_recovery_contribution_map_v7_s.csv",
            "Jw_catchup": "w45_Jw_raw025_catchup_contribution_map_v7_s.csv",
        }[phenomenon]
        _write_csv(df, paths.output_dir / out_name)
    all_contrib = pd.concat(list(contribution_tables.values()), ignore_index=True) if contribution_tables else pd.DataFrame()
    _write_csv(all_contrib, paths.output_dir / "w45_H_Jw_raw025_contribution_maps_all_v7_s.csv")
    # Region definitions and region curves.
    region_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    marker_rows: list[dict[str, Any]] = []
    region_objs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for field in FIELDS:
        for reg in _region_definitions(raw_fields[field], contribution_tables):
            region_objs.append((raw_fields[field], reg))
            rr = _region_row_from_mask(raw_fields[field], reg)
            region_rows.append(rr)
            curves, meta, _, _, _ = _region_progress_from_avg_state(avg_states[field], raw_fields[field]["weights"], reg["mask"], window, settings)
            if curves.empty:
                continue
            curves.insert(0, "field", field)
            curves.insert(1, "region_scheme", reg["region_scheme"])
            curves.insert(2, "region_id", reg["region_id"])
            curves.insert(3, "region_label", reg["region_label"])
            curves["phase_label"] = [_phase_label(int(d), window, h_retreat) for d in curves["day"]]
            curve_frames.append(curves)
            markers = _markers_from_curve(curves, window, settings)
            marker_rows.append({**rr, **meta, **markers})
    region_def = pd.DataFrame(region_rows)
    region_curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    region_markers = pd.DataFrame(marker_rows)
    _write_csv(region_def, paths.output_dir / "w45_H_Jw_raw025_region_definitions_v7_s.csv")
    _write_csv(region_curves, paths.output_dir / "w45_H_Jw_raw025_region_progress_curves_v7_s.csv")
    _write_csv(region_markers, paths.output_dir / "w45_H_Jw_raw025_region_timing_markers_v7_s.csv")
    return {
        "avg_states": avg_states,
        "whole_curves": whole_curves,
        "whole_markers": whole_markers,
        "whole_region_rows": whole_region_rows,
        "h_retreat": h_retreat,
        "crossing_df": crossing_df,
        "contribution_tables": contribution_tables,
        "region_objs": region_objs,
        "region_def": region_def,
        "region_curves": region_curves,
        "region_markers": region_markers,
    }


def _load_or_make_bootstrap_indices(n_years: int, paths: V7SPaths, settings: StagePartitionV7Settings) -> list[np.ndarray]:
    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    src = paths.v7e_output_dir / "bootstrap_resample_year_indices_v7_e.csv"
    out: list[np.ndarray] = []
    if src.exists():
        df = pd.read_csv(src)
        if {"bootstrap_id", "sampled_year_indices"}.issubset(df.columns):
            for _, r in df.sort_values("bootstrap_id").iterrows():
                vals = [int(x) for x in str(r["sampled_year_indices"]).split(";") if str(x).strip()]
                if vals:
                    arr = np.asarray(vals, dtype=int)
                    if np.all((arr >= 0) & (arr < n_years)):
                        out.append(arr)
            if len(out) >= n_boot:
                return out[:n_boot]
    rng = np.random.default_rng(int(settings.bootstrap.random_seed))
    return [rng.integers(0, n_years, size=n_years, dtype=int) for _ in range(n_boot)]


def _bootstrap_region_summaries(
    raw_fields: dict[str, dict[str, Any]],
    observed: dict[str, Any],
    window: dict[str, Any],
    settings: StagePartitionV7Settings,
    paths: V7SPaths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_years = min(int(raw_fields[f]["data"].shape[0]) for f in FIELDS)
    resamples = _load_or_make_bootstrap_indices(n_years, paths, settings)
    sample_rows: list[dict[str, Any]] = []
    progress_every = max(1, int(settings.bootstrap.progress_every))
    # Region objects were built from observed contribution masks. This is intentional: bootstrap tests stability of those diagnostic masks.
    for bid, idx in enumerate(resamples):
        if bid % progress_every == 0:
            _progress_log(paths, f"bootstrap region markers {bid}/{len(resamples)}")
        avg_cache = {field: _safe_nanmean(raw_fields[field]["data"][idx, :, :, :], axis=0) for field in FIELDS}
        for fd, reg in observed["region_objs"]:
            if not bool(reg.get("bootstrap_eligible", True)):
                continue
            field = fd["field"]
            try:
                curves, meta, _, _, _ = _region_progress_from_avg_state(avg_cache[field], fd["weights"], reg["mask"], window, settings)
            except Exception as exc:  # keep samples auditable rather than hiding a bad region.
                sample_rows.append({
                    "bootstrap_id": int(bid),
                    "field": field,
                    "region_scheme": reg["region_scheme"],
                    "region_id": reg["region_id"],
                    "region_label": reg["region_label"],
                    "sample_status": "failed",
                    "failure_reason": str(exc),
                })
                continue
            markers = _markers_from_curve(curves, window, settings) if not curves.empty else {}
            sample_rows.append({
                "bootstrap_id": int(bid),
                "field": field,
                "region_scheme": reg["region_scheme"],
                "region_id": reg["region_id"],
                "region_label": reg["region_label"],
                "sample_status": "ok",
                **meta,
                **markers,
            })
    samples = pd.DataFrame(sample_rows)
    _write_csv(samples, paths.output_dir / "w45_H_Jw_raw025_region_bootstrap_samples_v7_s.csv")
    if samples.empty:
        return samples, pd.DataFrame()
    marker_cols = ["departure90", "departure95", "t10", "t25", "t50", "t75", "peak_raw", "peak_smooth3", "duration_25_75", "tail_50_75", "early_span_25_50", "progress_monotonicity_corr", "separation_ratio"]
    group_cols = ["field", "region_scheme", "region_id", "region_label"]
    rows: list[dict[str, Any]] = []
    ok_samples = samples[samples.get("sample_status", "ok") == "ok"].copy()
    for keys, sub in ok_samples.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n_samples"] = int(len(sub))
        for col in marker_cols:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            row[f"{col}_valid_fraction"] = float(np.isfinite(vals).sum() / len(vals)) if len(vals) else np.nan
            row[f"{col}_median"] = _q(vals, 0.50)
            row[f"{col}_q05"] = _q(vals, 0.05)
            row[f"{col}_q95"] = _q(vals, 0.95)
            row[f"{col}_q025"] = _q(vals, 0.025)
            row[f"{col}_q975"] = _q(vals, 0.975)
            q05, q95 = row[f"{col}_q05"], row[f"{col}_q95"]
            q025, q975 = row[f"{col}_q025"], row[f"{col}_q975"]
            row[f"{col}_q90_width"] = float(q95 - q05) if np.isfinite(q05) and np.isfinite(q95) else np.nan
            row[f"{col}_q95_width"] = float(q975 - q025) if np.isfinite(q025) and np.isfinite(q975) else np.nan
        if "pre_post_separation_label" in sub.columns:
            row["dominant_pre_post_separation_label"] = str(sub["pre_post_separation_label"].mode().iloc[0]) if not sub["pre_post_separation_label"].mode().empty else "unknown"
        rows.append(row)
    summary = pd.DataFrame(rows)
    _write_csv(summary, paths.output_dir / "w45_H_Jw_raw025_region_bootstrap_summary_v7_s.csv")
    return samples, summary


def _phenomenon_decision(
    phenomenon: str,
    observed: dict[str, Any],
    bootstrap_summary: pd.DataFrame,
) -> dict[str, Any]:
    h_curves = observed["whole_curves"].get("H", pd.DataFrame())
    j_curves = observed["whole_curves"].get("Jw", pd.DataFrame())
    h_retreat = observed["h_retreat"]
    crossing = observed["crossing_df"]
    raw_status = "unresolved"
    evidence = "not_evaluated"
    if phenomenon == "H early-frontloaded":
        if not h_curves.empty:
            sub = h_curves[(h_curves["day"] >= 35) & (h_curves["day"] <= 43)]
            rise = float(sub["progress_raw"].iloc[-1] - sub["progress_raw"].iloc[0]) if len(sub) >= 2 else np.nan
            raw_status = "yes" if np.isfinite(rise) and rise > 0 else "no_or_unresolved"
            evidence = f"H day35-43 progress change={rise:.4g}" if np.isfinite(rise) else "H early interval unavailable"
    elif phenomenon == "H anchor retreat":
        raw_status = "yes" if bool(h_retreat.get("retreat_detected")) else "no_or_unresolved"
        evidence = f"{h_retreat.get('retreat_label')} drop={h_retreat.get('retreat_drop')} start={h_retreat.get('retreat_start_day')} end={h_retreat.get('retreat_end_day')}"
    elif phenomenon == "Jw mid_late_catchup":
        if not j_curves.empty:
            sub = j_curves[(j_curves["day"] >= 43) & (j_curves["day"] <= 53)]
            rise = float(sub["progress_raw"].iloc[-1] - sub["progress_raw"].iloc[0]) if len(sub) >= 2 else np.nan
            raw_status = "yes" if np.isfinite(rise) and rise > 0 else "no_or_unresolved"
            evidence = f"Jw day43-53 progress change={rise:.4g}" if np.isfinite(rise) else "Jw catchup interval unavailable"
    elif phenomenon == "H_Jw progress crossing":
        if crossing.empty:
            raw_status = "no_or_unresolved"
            evidence = "no H-to-Jw crossing found"
        else:
            labels = ";".join(crossing["interpretation_label"].astype(str).tolist())
            raw_status = "yes" if "H_retreat_with_Jw_overtake" in labels or "H_to_Jw_progress_crossing" in labels else "partial"
            evidence = labels
    elif phenomenon == "same_departure_candidate":
        hm = observed["whole_markers"].get("H", {})
        jm = observed["whole_markers"].get("Jw", {})
        h_dep = hm.get("departure90", np.nan)
        j_dep = jm.get("departure90", np.nan)
        if np.isfinite(h_dep) and np.isfinite(j_dep):
            lag = float(j_dep - h_dep)
            raw_status = "yes" if abs(lag) <= 1 else "no_or_unresolved"
            evidence = f"departure90 Jw-H={lag:.4g} days"
        else:
            raw_status = "unresolved"
            evidence = "departure90 unavailable for H or Jw"
    elif phenomenon == "global_clean_order":
        raw_status = "no" if bool(h_retreat.get("retreat_detected")) or not crossing.empty else "unresolved"
        evidence = "retreat/crossing evidence prevents clean global H-Jw order" if raw_status == "no" else "no clear contradiction found, but V7-s does not establish global order"
    return {"phenomenon": phenomenon, "raw025_status": raw_status, "raw025_evidence": evidence}


def _read_v7_status(paths: V7SPaths) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for label, p in [("v7_p", paths.v7p_output_dir), ("v7_q", paths.v7q_output_dir), ("v7_r", paths.v7r_output_dir)]:
        meta = p / "run_meta.json"
        status[f"{label}_exists"] = p.exists()
        status[f"{label}_run_meta_exists"] = meta.exists()
        if meta.exists():
            try:
                status[f"{label}_run_meta"] = json.loads(meta.read_text(encoding="utf-8"))
            except Exception as exc:
                status[f"{label}_run_meta_error"] = str(exc)
    return status


def _comparison_table(paths: V7SPaths, observed: dict[str, Any], bootstrap_summary: pd.DataFrame) -> pd.DataFrame:
    phenomena = [
        "H early-frontloaded",
        "H anchor retreat",
        "Jw mid_late_catchup",
        "H_Jw progress crossing",
        "same_departure_candidate",
        "global_clean_order",
    ]
    rows: list[dict[str, Any]] = []
    v7_status = _read_v7_status(paths)
    for ph in phenomena:
        dec = _phenomenon_decision(ph, observed, bootstrap_summary)
        if ph == "H early-frontloaded":
            v7r = "yes_from_feature_relation_v7_r"
            v7q = "feature_level_available"
            v7p = "wholefield_reference_available" if v7_status.get("v7_p_run_meta_exists") else "unknown"
        elif ph == "H anchor retreat":
            v7r = "not_primary_v7_r_metric"
            v7q = "feature_progress_may_contain_signal"
            v7p = "wholefield_curve_reference_available" if v7_status.get("v7_p_run_meta_exists") else "unknown"
        elif ph == "Jw mid_late_catchup":
            v7r = "partial_support_duration_tail_v7_r"
            v7q = "feature_level_available"
            v7p = "wholefield_reference_available" if v7_status.get("v7_p_run_meta_exists") else "unknown"
        elif ph == "H_Jw progress crossing":
            v7r = "phase_specific_relation_not_global_order"
            v7q = "feature_distribution_available"
            v7p = "pairwise_daily_reference_available" if v7_status.get("v7_p_run_meta_exists") else "unknown"
        elif ph == "same_departure_candidate":
            v7r = "yes_near_same_phase_candidate"
            v7q = "feature_marker_available"
            v7p = "marker_reference_available" if v7_status.get("v7_p_run_meta_exists") else "unknown"
        else:
            v7r = "global_clean_order_not_supported"
            v7q = "feature_heterogeneity_present"
            v7p = "clean_global_lead_not_supported_or_unresolved"
        raw = dec["raw025_status"]
        if raw == "yes" and "yes" in str(v7r):
            decision = "robust_across_representation_candidate"
        elif raw == "yes":
            decision = "raw025_supports_phenomenon"
        elif raw == "no":
            decision = "raw025_does_not_support"
        elif "unresolved" in raw:
            decision = "unresolved"
        else:
            decision = "partial_or_representation_sensitive"
        rows.append({
            "phenomenon": ph,
            "v7_p_2deg_status": v7p,
            "v7_q_feature_status": v7q,
            "v7_r_relation_status": v7r,
            "raw025_status": raw,
            "raw025_evidence": dec["raw025_evidence"],
            "decision": decision,
            "interpretation_boundary": "comparison table only; V7-p/q/r are not used to construct raw025 progress",
        })
    df = pd.DataFrame(rows)
    _write_csv(df, paths.output_dir / "w45_H_Jw_raw025_vs_2deg_comparison_v7_s.csv")
    return df


def _plot_progress_curves(whole_curves: dict[str, pd.DataFrame], crossing: pd.DataFrame, h_retreat: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        _write_text(f"Plot skipped because matplotlib import failed: {exc}\n", path.with_suffix(".txt"))
        return
    _ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for field, df in whole_curves.items():
        ax.plot(df["day"], df["progress_raw"], marker="o", linewidth=1.8, label=field)
    ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1, label="anchor_day=45")
    if bool(h_retreat.get("retreat_detected")):
        ax.axvspan(float(h_retreat["retreat_start_day"]), float(h_retreat["retreat_end_day"]), alpha=0.15, label="H retreat candidate")
    if not crossing.empty:
        for _, r in crossing.iterrows():
            ax.axvline(float(r["first_crossing_day"]), linestyle=":", linewidth=1)
    ax.set_xlabel("Day index (Apr 1 = 0)")
    ax.set_ylabel("Raw-field projection progress")
    ax.set_title("W45 H/Jw raw025 whole-field progress (V7-s)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_contribution_map(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except Exception as exc:
        _write_text(
            "Cartopy map skipped. CSV output remains authoritative. "
            f"Import error: {exc}\n",
            path.with_suffix(".txt"),
        )
        return
    _ensure_dir(path.parent)
    piv = df.pivot(index="lat", columns="lon", values="contribution").sort_index()
    lats = piv.index.to_numpy(dtype=float)
    lons = piv.columns.to_numpy(dtype=float)
    vals = piv.to_numpy(dtype=float)
    fig = plt.figure(figsize=(7.5, 4.8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.7)
    mesh = ax.pcolormesh(lons, lats, vals, transform=ccrs.PlateCarree(), shading="auto")
    ax.set_title(title)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3)
    gl.top_labels = False
    gl.right_labels = False
    fig.colorbar(mesh, ax=ax, shrink=0.78, label="progress contribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_region_panel(region_curves: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        _write_text(f"Plot skipped because matplotlib import failed: {exc}\n", path.with_suffix(".txt"))
        return
    if region_curves.empty:
        return
    sub = region_curves[region_curves["region_scheme"].isin(["equal_count_three_region", "contribution_region"])]
    if sub.empty:
        return
    _ensure_dir(path.parent)
    fields = list(FIELDS)
    fig, axes = plt.subplots(len(fields), 1, figsize=(9, 4.2 * len(fields)), sharex=True)
    if len(fields) == 1:
        axes = [axes]
    for ax, field in zip(axes, fields):
        sdf = sub[sub["field"] == field]
        for (scheme, label), g in sdf.groupby(["region_scheme", "region_label"], sort=True):
            ax.plot(g["day"], g["progress_raw"], marker="o", linewidth=1.2, label=f"{scheme}:{label}")
        ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_ylabel(f"{field} progress")
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Day index (Apr 1 = 0)")
    fig.suptitle("W45 H/Jw raw025 regional progress (V7-s)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary_md(
    paths: V7SPaths,
    window: dict[str, Any],
    raw_fields: dict[str, dict[str, Any]],
    observed: dict[str, Any],
    bootstrap_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    h_retreat = observed["h_retreat"]
    crossing = observed["crossing_df"]
    lines: list[str] = []
    lines.append("# W45 H/Jw raw025 process audit V7-s")
    lines.append("")
    lines.append(f"Created: {_now_iso()}")
    lines.append("")
    lines.append("## 1. Input audit")
    lines.append("")
    lines.append(f"- Window: {window}")
    for field in FIELDS:
        fd = raw_fields[field]
        lines.append(f"- {field}: key={fd['field_key']}, shape={list(fd['data'].shape)}, lat={float(np.nanmin(fd['lat'])):.3f}..{float(np.nanmax(fd['lat'])):.3f}, lon={float(np.nanmin(fd['lon'])):.3f}..{float(np.nanmax(fd['lon'])):.3f}")
    lines.append("")
    lines.append("## 2. What is being tested")
    lines.append("")
    lines.append("- H early-frontloaded")
    lines.append("- H anchor-near retreat / non-monotonic segment")
    lines.append("- Jw mid/late catch-up")
    lines.append("- H/Jw progress crossing")
    lines.append("")
    lines.append("## 3. Direct raw025 outputs")
    lines.append("")
    lines.append(f"- H retreat candidate: {h_retreat}")
    if crossing.empty:
        lines.append("- Crossing events: none found")
    else:
        lines.append("- Crossing events:")
        for _, r in crossing.iterrows():
            lines.append(f"  - day {int(r['first_crossing_day'])}: {r['interpretation_label']}")
    lines.append("")
    lines.append("## 4. Comparison with V7-p/q/r")
    lines.append("")
    if comparison.empty:
        lines.append("No comparison table was created.")
    else:
        for _, r in comparison.iterrows():
            lines.append(f"- {r['phenomenon']}: raw025={r['raw025_status']}; decision={r['decision']}; evidence={r['raw025_evidence']}")
    lines.append("")
    lines.append("## 5. Interpretation allowed")
    lines.append("")
    lines.append("- Raw-field support / non-support for H/Jw process phenomena.")
    lines.append("- Region-level candidate patterns when contribution maps and bootstrap stability support them.")
    lines.append("- Representation sensitivity relative to 2-degree profile / feature diagnostics.")
    lines.append("")
    lines.append("## 6. Interpretation forbidden")
    lines.append("")
    lines.append("- Do not infer causal H→Jw or Jw→H from this audit.")
    lines.append("- Do not call near-same departure synchrony without equivalence testing.")
    lines.append("- Do not claim a global clean order if retreat/crossing/phase-specific evidence is present.")
    lines.append("- Do not treat contribution-defined masks as physical regions without independent provenance and spatial-coherence checks.")
    lines.append("")
    lines.append("## 7. Recommended next step")
    lines.append("")
    lines.append("If raw025 supports spatially coherent H retreat and Jw catch-up, continue to a spatial interpretation audit. If it does not, close the V7-r H/Jw relation as representation-sensitive / feature-diagnostic only.")
    _write_text("\n".join(lines) + "\n", paths.output_dir / "w45_H_Jw_raw025_process_audit_summary_v7_s.md")


def run_w45_H_Jw_raw025_process_audit_v7_s(v7_root: Optional[Path] = None) -> None:
    paths = _resolve_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)
    # Clear progress log for this run.
    (paths.log_dir / "run_progress_v7_s.log").write_text("", encoding="utf-8")
    _progress_log(paths, "start")
    settings = _configure_settings(paths)
    smoothed_path = settings.foundation.smoothed_fields_path()
    _progress_log(paths, f"load smoothed raw field: {smoothed_path}")
    smoothed = load_smoothed_fields(smoothed_path)
    raw_fields = {field: _prepare_raw_field(smoothed, field, settings) for field in FIELDS}
    n_days = min(raw_fields[f]["n_days"] for f in FIELDS)
    n_years = min(raw_fields[f]["n_years"] for f in FIELDS)
    window = _load_w45_window(paths, n_days)
    input_audit = {
        "created_at": _now_iso(),
        "status": "success",
        "version": "v7_s",
        "output_tag": OUTPUT_TAG,
        "purpose": "W45 H/Jw raw025 process audit; input-representation sensitivity branch",
        "smoothed_fields_path": str(smoothed_path),
        "smoothed_fields_exists": bool(smoothed_path.exists()),
        "window": window,
        "fields": {field: {
            "field_key": raw_fields[field]["field_key"],
            "shape": list(raw_fields[field]["data"].shape),
            "lat_min": float(np.nanmin(raw_fields[field]["lat"])),
            "lat_max": float(np.nanmax(raw_fields[field]["lat"])),
            "lon_min": float(np.nanmin(raw_fields[field]["lon"])),
            "lon_max": float(np.nanmax(raw_fields[field]["lon"])),
            "lat_order_after_loading": "ascending",
            "lon_order_after_loading": "ascending",
            "lat_indices_original_minmax": [int(np.min(raw_fields[field]["lat_indices_original"])), int(np.max(raw_fields[field]["lat_indices_original"]))],
            "lon_indices_original_minmax": [int(np.min(raw_fields[field]["lon_indices_original"])), int(np.max(raw_fields[field]["lon_indices_original"]))],
        } for field in FIELDS},
        "does_not_use_2deg_interpolation_for_main_computation": True,
        "reads_v7_p_q_r_for_comparison_only": True,
        "reads_v7_p_q_r_for_main_computation": False,
    }
    _write_json(input_audit, paths.output_dir / "input_audit_v7_s.json")
    _progress_log(paths, "observed raw025 progress, contributions, regions")
    observed = _make_observed_outputs(raw_fields, window, settings, paths)
    _progress_log(paths, "bootstrap stability")
    _, bootstrap_summary = _bootstrap_region_summaries(raw_fields, observed, window, settings, paths)
    _progress_log(paths, "comparison with V7-p/q/r status")
    comparison = _comparison_table(paths, observed, bootstrap_summary)
    skip_plots = os.environ.get("V7S_SKIP_PLOTS", "").strip().lower() in {"1", "true", "yes"}
    if skip_plots:
        _progress_log(paths, "figures skipped by V7S_SKIP_PLOTS")
    else:
        _progress_log(paths, "figures")
        _plot_progress_curves(
            observed["whole_curves"],
            observed["crossing_df"],
            observed["h_retreat"],
            paths.figure_dir / "w45_H_Jw_raw025_wholefield_progress_curves_v7_s.png",
        )
        if "H_retreat" in observed["contribution_tables"]:
            _plot_contribution_map(
                observed["contribution_tables"]["H_retreat"],
                paths.figure_dir / "w45_H_raw025_retreat_contribution_map_v7_s.png",
                "W45 H raw025 retreat contribution (V7-s)",
            )
        if "H_early_frontloaded" in observed["contribution_tables"]:
            _plot_contribution_map(
                observed["contribution_tables"]["H_early_frontloaded"],
                paths.figure_dir / "w45_H_raw025_early_contribution_map_v7_s.png",
                "W45 H raw025 early-frontloaded contribution (V7-s)",
            )
        if "H_late_recovery" in observed["contribution_tables"]:
            _plot_contribution_map(
                observed["contribution_tables"]["H_late_recovery"],
                paths.figure_dir / "w45_H_raw025_late_recovery_contribution_map_v7_s.png",
                "W45 H raw025 late-recovery contribution (V7-s)",
            )
        if "Jw_catchup" in observed["contribution_tables"]:
            _plot_contribution_map(
                observed["contribution_tables"]["Jw_catchup"],
                paths.figure_dir / "w45_Jw_raw025_catchup_contribution_map_v7_s.png",
                "W45 Jw raw025 catch-up contribution (V7-s)",
            )
        _plot_region_panel(
            observed["region_curves"],
            paths.figure_dir / "w45_H_Jw_raw025_region_progress_panel_v7_s.png",
        )
    _progress_log(paths, "summary")
    _write_summary_md(paths, window, raw_fields, observed, bootstrap_summary, comparison)
    run_meta = {
        "version": "v7_s",
        "output_tag": OUTPUT_TAG,
        "status": "success",
        "created_at": _now_iso(),
        "input_representation": "raw025_smoothed_field",
        "does_not_use_2deg_interpolation_for_main_computation": True,
        "fields": list(FIELDS),
        "H_field_key": FIELD_SPECS["H"]["field_key"],
        "Jw_field_key": FIELD_SPECS["Jw"]["field_key"],
        "H_domain": {"lon": list(raw_fields["H"]["lon_range_requested"]), "lat": list(raw_fields["H"]["lat_range_requested"])},
        "Jw_domain": {"lon": list(raw_fields["Jw"]["lon_range_requested"]), "lat": list(raw_fields["Jw"]["lat_range_requested"])},
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "analysis_window": [int(window["analysis_window_start"]), int(window["analysis_window_end"])],
        "pre_period": [int(window["pre_period_start"]), int(window["pre_period_end"])],
        "post_period": [int(window["post_period_start"]), int(window["post_period_end"])],
        "n_years": int(n_years),
        "n_days": int(n_days),
        "n_bootstrap": int(settings.bootstrap.effective_n_bootstrap()),
        "reads_v7_p_q_r_for_comparison_only": True,
        "reads_v7_p_q_r_for_main_computation": False,
        "interpretation_boundary": "raw025 audit branch; not final physical mechanism; not global clean-order proof",
        "key_outputs": {
            "wholefield_progress": "w45_H_Jw_raw025_wholefield_progress_curves_v7_s.csv",
            "crossing_events": "w45_H_Jw_raw025_crossing_events_v7_s.csv",
            "region_curves": "w45_H_Jw_raw025_region_progress_curves_v7_s.csv",
            "region_markers": "w45_H_Jw_raw025_region_timing_markers_v7_s.csv",
            "bootstrap_summary": "w45_H_Jw_raw025_region_bootstrap_summary_v7_s.csv",
            "comparison": "w45_H_Jw_raw025_vs_2deg_comparison_v7_s.csv",
            "summary_md": "w45_H_Jw_raw025_process_audit_summary_v7_s.md",
        },
    }
    _write_json(run_meta, paths.output_dir / "run_meta.json")
    _write_text(
        "# W45 H/Jw raw025 process audit V7-s\n\n"
        f"Created: {_now_iso()}\n\n"
        "Status: success\n\n"
        "This branch computes H/Jw process diagnostics from raw025 smoothed fields. "
        "V7-p/q/r are comparison sources only, not main computation inputs.\n",
        paths.log_dir / "w45_H_Jw_raw025_process_audit_v7_s.md",
    )
    _progress_log(paths, "finished success")


__all__ = ["run_w45_H_Jw_raw025_process_audit_v7_s"]
