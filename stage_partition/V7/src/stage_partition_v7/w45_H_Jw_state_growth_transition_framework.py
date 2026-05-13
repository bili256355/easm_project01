from __future__ import annotations

import json
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

OUTPUT_TAG = "w45_H_Jw_state_growth_transition_framework_v7_u"
HOTFIX_ID = "v7_u_hotfix_01_departure_after_pre"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELDS = ("H", "Jw")
FIELD_SPECS = {
    "H": {"field_key": "z500_smoothed", "domain_attr_lon": "h_lon_range", "domain_attr_lat": "h_lat_range"},
    "Jw": {"field_key": "u200_smoothed", "domain_attr_lon": "jw_lon_range", "domain_attr_lat": "jw_lat_range"},
}
EPS = 1.0e-12
STATE_EVENT_NAMES = (
    "departure_from_pre",
    "durable_departure_from_pre_3d",
    "post_dominance_day",
    "durable_post_dominance_2d",
    "durable_post_dominance_3d",
    "durable_post_dominance_4d",
)
GROWTH_EVENT_NAMES = (
    "max_growth_day",
    "postward_growth_peak_day",
    "rapid_growth_start",
    "rapid_growth_center",
    "rapid_growth_end",
)


@dataclass
class V7UPaths:
    v7_root: Path
    project_root: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path
    v7e_output_dir: Path
    v7t_output_dir: Path


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


def _q(values: np.ndarray | list[float], quantile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanquantile(arr, quantile)) if arr.size else np.nan


def _resolve_paths(v7_root: Optional[Path]) -> V7UPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = v7_root.parents[1]
    return V7UPaths(
        v7_root=v7_root,
        project_root=project_root,
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7t_output_dir=v7_root / "outputs" / "w45_H_Jw_transition_definition_audit_v7_t",
    )


def _configure_settings(paths: V7UPaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    env_debug = os.environ.get("V7U_DEBUG_N_BOOTSTRAP", "").strip()
    if env_debug:
        settings.bootstrap.debug_n_bootstrap = int(env_debug)
    return settings


def _progress_log(paths: V7UPaths, message: str) -> None:
    _ensure_dir(paths.log_dir)
    log_path = paths.log_dir / "run_progress_v7_u.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_now_iso()}] {message}\n")
    print(f"[V7-u] {message}", flush=True)


def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(float(lower), float(upper)), max(float(lower), float(upper))
    return (arr >= lo) & (arr <= hi)


def _axis_index(lat_or_lon: np.ndarray, value_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(lat_or_lon, dtype=float)
    mask = _mask_between(values, *value_range)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No coordinate values in requested range {value_range}")
    coord_unsorted = values[idx]
    order = np.argsort(coord_unsorted)
    return idx[order], coord_unsorted[order]


def _load_w45_window(paths: V7UPaths, n_days: int) -> dict[str, Any]:
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


def _days_from_window(window: dict[str, Any]) -> np.ndarray:
    return np.arange(int(window["analysis_window_start"]), int(window["analysis_window_end"]) + 1, dtype=int)


def _period_days(window: dict[str, Any], prefix: str) -> np.ndarray:
    return np.arange(int(window[f"{prefix}_period_start"]), int(window[f"{prefix}_period_end"]) + 1, dtype=int)


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
    lat_idx, lats = _axis_index(lat, lat_range)
    lon_idx, lons = _axis_index(lon, lon_range)
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
    return float(np.sqrt(max(np.nansum(v[good] * v[good] * w[good]) / den, 0.0)))


def _weighted_corr(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if good.sum() < 3:
        return np.nan
    wg = w[good]
    den = float(np.nansum(wg))
    if den <= EPS:
        return np.nan
    xg = x[good]
    yg = y[good]
    mx = float(np.nansum(xg * wg) / den)
    my = float(np.nansum(yg * wg) / den)
    vx = float(np.nansum(((xg - mx) ** 2) * wg))
    vy = float(np.nansum(((yg - my) ** 2) * wg))
    if vx <= EPS or vy <= EPS:
        return np.nan
    cov = float(np.nansum((xg - mx) * (yg - my) * wg))
    return float(cov / np.sqrt(vx * vy))


def _rolling_mean_centered(values: np.ndarray, window: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size == 0:
        return arr.copy()
    out = np.full(arr.shape, np.nan, dtype=float)
    half = window // 2
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        out[i] = _q(arr[lo:hi], 0.5) if np.isfinite(arr[lo:hi]).any() else np.nan
    return out


def _consecutive_true_first(days: np.ndarray, cond: np.ndarray, stable_days: int = 1) -> float:
    days = np.asarray(days, dtype=int)
    cond = np.asarray(cond, dtype=bool)
    stable_days = max(1, int(stable_days))
    if cond.size < stable_days:
        return np.nan
    for i in range(0, cond.size - stable_days + 1):
        if bool(np.all(cond[i : i + stable_days])):
            return float(days[i])
    return np.nan


def _first_true(days: np.ndarray, cond: np.ndarray) -> float:
    days = np.asarray(days, dtype=int)
    cond = np.asarray(cond, dtype=bool)
    idx = np.where(cond)[0]
    return float(days[idx[0]]) if idx.size else np.nan


def _event_status_from_day(day: float, positive_type: str = "confirmed_event") -> str:
    return positive_type if np.isfinite(day) else "not_detected"


def _transition_metrics_for_avg_state(
    avg_state: np.ndarray,
    weights_2d: np.ndarray,
    window: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    state = np.asarray(avg_state, dtype=float)
    if state.ndim != 3:
        raise ValueError(f"Expected avg_state days x lat x lon; got {state.shape}")
    days = _days_from_window(window)
    pre_days = _period_days(window, "pre")
    post_days = _period_days(window, "post")
    if np.any(days < 0) or np.any(days >= state.shape[0]):
        raise ValueError(f"Analysis window outside state length {state.shape[0]}")
    pre = _safe_nanmean(state[pre_days, :, :], axis=0)
    post = _safe_nanmean(state[post_days, :, :], axis=0)
    d = post - pre
    w = np.asarray(weights_2d, dtype=float)
    denom = _weighted_sum(d * d, w)
    transition_norm = _weighted_rms(d, w)
    state_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    prev_x: np.ndarray | None = None
    prev_pdist = np.nan
    prev_rdiff = np.nan
    for day in days:
        x = state[int(day), :, :]
        y = x - pre
        if np.isfinite(denom) and abs(denom) > EPS:
            p_proj = _weighted_sum(y * d, w) / denom
        else:
            p_proj = np.nan
        d_pre = _weighted_rms(x - pre, w)
        d_post = _weighted_rms(x - post, w)
        p_dist = float(d_pre / (d_pre + d_post)) if np.isfinite(d_pre) and np.isfinite(d_post) and (d_pre + d_post) > EPS else np.nan
        r_pre = _weighted_corr(x, pre, w)
        r_post = _weighted_corr(x, post, w)
        r_diff = float(r_post - r_pre) if np.isfinite(r_pre) and np.isfinite(r_post) else np.nan
        state_rows.append({
            "day": int(day),
            "D_pre": float(d_pre) if np.isfinite(d_pre) else np.nan,
            "D_post": float(d_post) if np.isfinite(d_post) else np.nan,
            "P_dist": float(p_dist) if np.isfinite(p_dist) else np.nan,
            "R_pre": float(r_pre) if np.isfinite(r_pre) else np.nan,
            "R_post": float(r_post) if np.isfinite(r_post) else np.nan,
            "R_diff": float(r_diff) if np.isfinite(r_diff) else np.nan,
            "P_proj_reference": float(p_proj) if np.isfinite(p_proj) else np.nan,
        })
        if prev_x is None:
            change_norm = np.nan
            delta_pdist = np.nan
            delta_rdiff = np.nan
        else:
            dx = x - prev_x
            change_norm = _weighted_rms(dx, w)
            delta_pdist = float(p_dist - prev_pdist) if np.isfinite(p_dist) and np.isfinite(prev_pdist) else np.nan
            delta_rdiff = float(r_diff - prev_rdiff) if np.isfinite(r_diff) and np.isfinite(prev_rdiff) else np.nan
        growth_rows.append({
            "day": int(day),
            "field_change_norm": float(change_norm) if np.isfinite(change_norm) else np.nan,
            "delta_P_dist": float(delta_pdist) if np.isfinite(delta_pdist) else np.nan,
            "delta_R_diff": float(delta_rdiff) if np.isfinite(delta_rdiff) else np.nan,
        })
        prev_x = x
        prev_pdist = p_dist
        prev_rdiff = r_diff
    state_df = pd.DataFrame(state_rows)
    growth_df = pd.DataFrame(growth_rows)
    for col in ["field_change_norm", "delta_P_dist", "delta_R_diff"]:
        growth_df[f"{col}_smooth3"] = _rolling_mean_centered(growth_df[col].to_numpy(dtype=float), window=3)
    meta = {
        "transition_norm": transition_norm,
        "projection_denom": denom,
        "pre_mean": float(np.nanmean(pre)),
        "post_mean": float(np.nanmean(post)),
        "pre_days": [int(x) for x in pre_days],
        "post_days": [int(x) for x in post_days],
    }
    return state_df, growth_df, meta


def _state_events(state_df: pd.DataFrame, window: dict[str, Any]) -> tuple[dict[str, float], dict[str, str], dict[str, Any]]:
    days = state_df["day"].to_numpy(dtype=int)
    dpre = state_df["D_pre"].to_numpy(dtype=float)
    pdist = state_df["P_dist"].to_numpy(dtype=float)
    rdiff = state_df["R_diff"].to_numpy(dtype=float)
    pre_start = int(window["pre_period_start"])
    pre_end = int(window["pre_period_end"])
    pre_mask = (days >= pre_start) & (days <= pre_end)
    # Hotfix v7_u_hotfix_01:
    # Departure cannot be detected inside the pre-period used to define the
    # pre-state envelope. Previous V7-u allowed the first crossing to occur
    # at day30--37, which made Jw day30 / H day37 possible and contaminated
    # the departure adjudication. The envelope is still estimated from the
    # pre-period, but the search starts strictly after pre_period_end.
    departure_search_start = pre_end + 1
    search_mask = days >= departure_search_start
    pre_envelope = _q(dpre[pre_mask], 0.95) if pre_mask.any() else _q(dpre, 0.20)
    departure_cond_raw = np.isfinite(dpre) & (dpre > pre_envelope)
    departure_cond = departure_cond_raw & search_mask
    post_cond = np.isfinite(pdist) & np.isfinite(rdiff) & (pdist > 0.5) & (rdiff > 0.0)
    dist_only_cond = np.isfinite(pdist) & (pdist > 0.5)
    pattern_only_cond = np.isfinite(rdiff) & (rdiff > 0.0)
    events = {
        "departure_from_pre": _first_true(days, departure_cond),
        "durable_departure_from_pre_3d": _consecutive_true_first(days, departure_cond, stable_days=3),
        "post_dominance_day": _first_true(days, post_cond),
        "durable_post_dominance_2d": _consecutive_true_first(days, post_cond, stable_days=2),
        "durable_post_dominance_3d": _consecutive_true_first(days, post_cond, stable_days=3),
        "durable_post_dominance_4d": _consecutive_true_first(days, post_cond, stable_days=4),
    }
    # Non-durable departure is retained as a candidate diagnostic only.
    # The main departure adjudication should use durable_departure_from_pre_3d.
    statuses: dict[str, str] = {
        "departure_from_pre": "candidate_non_durable_after_pre" if np.isfinite(events["departure_from_pre"]) else "not_detected",
        "durable_departure_from_pre_3d": _event_status_from_day(events["durable_departure_from_pre_3d"], "confirmed_event"),
    }
    # State post-dominance statuses distinguish confirmed from distance-only / pattern-only possibilities.
    first_dist = _first_true(days, dist_only_cond)
    first_pat = _first_true(days, pattern_only_cond)
    for name in ["post_dominance_day", "durable_post_dominance_2d", "durable_post_dominance_3d", "durable_post_dominance_4d"]:
        if np.isfinite(events[name]):
            statuses[name] = "confirmed_event"
        elif np.isfinite(first_dist) and not np.isfinite(first_pat):
            statuses[name] = "distance_only"
        elif np.isfinite(first_pat) and not np.isfinite(first_dist):
            statuses[name] = "pattern_only"
        elif np.isfinite(first_dist) or np.isfinite(first_pat):
            statuses[name] = "not_durable" if name.startswith("durable") else "not_detected"
        else:
            statuses[name] = "not_detected"
    diag = {
        "pre_envelope_D_pre_q95": pre_envelope,
        "departure_search_start_day": int(departure_search_start),
        "departure_search_rule": "search starts after pre_period_end; pre-period crossings are not allowed as departure events",
        "departure_raw_first_crossing_including_pre": _first_true(days, departure_cond_raw),
        "departure_candidate_after_pre": events["departure_from_pre"],
        "durable_departure_after_pre_3d": events["durable_departure_from_pre_3d"],
        "first_distance_only_post_dominance": first_dist,
        "first_pattern_only_post_dominance": first_pat,
    }
    return events, statuses, diag


def _growth_events(growth_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, str], dict[str, Any]]:
    days = growth_df["day"].to_numpy(dtype=int)
    chg = growth_df["field_change_norm_smooth3"].to_numpy(dtype=float)
    dp = growth_df["delta_P_dist_smooth3"].to_numpy(dtype=float)
    dr = growth_df["delta_R_diff_smooth3"].to_numpy(dtype=float)
    events: dict[str, float] = {}
    statuses: dict[str, str] = {}
    if np.isfinite(chg).any():
        events["max_growth_day"] = float(days[int(np.nanargmax(chg))])
        statuses["max_growth_day"] = "confirmed_event"
        q75 = _q(chg, 0.75)
        cond = np.isfinite(chg) & (chg >= q75)
        # Choose longest contiguous segment; tie-break by largest cumulative change.
        segments: list[tuple[int, int]] = []
        i = 0
        while i < cond.size:
            if not cond[i]:
                i += 1
                continue
            j = i
            while j + 1 < cond.size and cond[j + 1]:
                j += 1
            segments.append((i, j))
            i = j + 1
        if segments:
            def _seg_score(seg: tuple[int, int]) -> tuple[int, float]:
                lo, hi = seg
                return (hi - lo + 1, float(np.nansum(chg[lo : hi + 1])))
            best = max(segments, key=_seg_score)
            lo, hi = best
            weights = chg[lo : hi + 1]
            dseg = days[lo : hi + 1].astype(float)
            if np.isfinite(weights).any() and float(np.nansum(weights)) > EPS:
                center = float(np.nansum(dseg * weights) / np.nansum(weights))
            else:
                center = float(np.nanmean(dseg))
            events["rapid_growth_start"] = float(days[lo])
            events["rapid_growth_center"] = center
            events["rapid_growth_end"] = float(days[hi])
            statuses["rapid_growth_start"] = statuses["rapid_growth_center"] = statuses["rapid_growth_end"] = "confirmed_event"
        else:
            events["rapid_growth_start"] = events["rapid_growth_center"] = events["rapid_growth_end"] = np.nan
            statuses["rapid_growth_start"] = statuses["rapid_growth_center"] = statuses["rapid_growth_end"] = "not_detected"
    else:
        events["max_growth_day"] = np.nan
        statuses["max_growth_day"] = "not_detected"
        events["rapid_growth_start"] = events["rapid_growth_center"] = events["rapid_growth_end"] = np.nan
        statuses["rapid_growth_start"] = statuses["rapid_growth_center"] = statuses["rapid_growth_end"] = "not_detected"
    valid_postward = np.isfinite(dp) & np.isfinite(dr) & (dr >= 0)
    if valid_postward.any():
        idxs = np.where(valid_postward)[0]
        best = idxs[int(np.nanargmax(dp[idxs]))]
        events["postward_growth_peak_day"] = float(days[best])
        statuses["postward_growth_peak_day"] = "confirmed_event"
    elif np.isfinite(dp).any():
        events["postward_growth_peak_day"] = float(days[int(np.nanargmax(dp))])
        statuses["postward_growth_peak_day"] = "distance_growth_only"
    else:
        events["postward_growth_peak_day"] = np.nan
        statuses["postward_growth_peak_day"] = "not_detected"
    diag = {
        "rapid_growth_q75_threshold": _q(chg, 0.75),
        "postward_peak_requires_delta_Rdiff_nonnegative": True,
    }
    return events, statuses, diag


def _projection_reference_events(state_df: pd.DataFrame) -> dict[str, float]:
    days = state_df["day"].to_numpy(dtype=int)
    p = state_df["P_proj_reference"].to_numpy(dtype=float)
    out: dict[str, float] = {}
    for thr in (0.25, 0.50, 0.75):
        cond = np.isfinite(p) & (p >= thr)
        out[f"P_proj_first_t{int(thr * 100):02d}"] = _first_true(days, cond)
        # durable: 3 consecutive days above threshold.
        out[f"P_proj_durable_t{int(thr * 100):02d}_3d"] = _consecutive_true_first(days, cond, stable_days=3)
    return out


def _order_sync_decision(deltas: np.ndarray, margin_days: float = 1.0) -> dict[str, Any]:
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "valid_bootstrap_pairs": 0,
            "delta_median": np.nan,
            "delta_q05": np.nan,
            "delta_q95": np.nan,
            "delta_q025": np.nan,
            "delta_q975": np.nan,
            "P_H_leads": np.nan,
            "P_Jw_leads": np.nan,
            "sync_equivalence_pass": False,
            "final_decision": "invalid_event",
            "decision_level": "method_unclosed",
        }
    q025, q975 = _q(arr, 0.025), _q(arr, 0.975)
    p_h = float(np.mean(arr > float(margin_days)))
    p_j = float(np.mean(arr < -float(margin_days)))
    sync = bool(np.isfinite(q025) and np.isfinite(q975) and q025 >= -float(margin_days) and q975 <= float(margin_days))
    if p_h >= 0.90:
        decision = "H_leads"
        level = "hard_decision"
    elif p_j >= 0.90:
        decision = "Jw_leads"
        level = "hard_decision"
    elif sync:
        decision = "synchronous_equivalent"
        level = "hard_decision"
    else:
        decision = "unresolved"
        if max(p_h, p_j) >= 0.67:
            level = "directional_tendency_only"
        else:
            level = "method_unclosed"
    return {
        "valid_bootstrap_pairs": int(arr.size),
        "delta_median": _q(arr, 0.50),
        "delta_q05": _q(arr, 0.05),
        "delta_q95": _q(arr, 0.95),
        "delta_q025": q025,
        "delta_q975": q975,
        "P_H_leads": p_h,
        "P_Jw_leads": p_j,
        "sync_equivalence_pass": sync,
        "final_decision": decision,
        "decision_level": level,
    }


def _bootstrap_events(field_obj: dict[str, Any], window: dict[str, Any], n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = np.asarray(field_obj["data"], dtype=float)
    weights = np.asarray(field_obj["weights"], dtype=float)
    rng = np.random.default_rng(int(seed))
    n_years = int(data.shape[0])
    state_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for b in range(int(n_boot)):
        idx = rng.integers(0, n_years, size=n_years)
        avg_state = _safe_nanmean(data[idx, :, :, :], axis=0)
        state_df, growth_df, _ = _transition_metrics_for_avg_state(avg_state, weights, window)
        se, ss, _ = _state_events(state_df, window)
        ge, gs, _ = _growth_events(growth_df)
        pe = _projection_reference_events(state_df)
        state_rows.append({"field": field_obj["field"], "bootstrap_id": int(b), **se, **{f"status__{k}": v for k, v in ss.items()}})
        growth_rows.append({"field": field_obj["field"], "bootstrap_id": int(b), **ge, **{f"status__{k}": v for k, v in gs.items()}})
        projection_rows.append({"field": field_obj["field"], "bootstrap_id": int(b), **pe})
    return pd.DataFrame(state_rows), pd.DataFrame(growth_rows), pd.DataFrame(projection_rows)


def _registry_from_events(
    base_events: pd.DataFrame,
    boot_events: pd.DataFrame,
    event_names: tuple[str, ...],
    kind: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in FIELDS:
        base_row = base_events[base_events["field"] == field]
        boot = boot_events[boot_events["field"] == field]
        for event in event_names:
            obs = float(base_row.iloc[0].get(event, np.nan)) if not base_row.empty else np.nan
            status = str(base_row.iloc[0].get(f"status__{event}", "not_detected")) if not base_row.empty else "not_detected"
            vals = boot[event].to_numpy(dtype=float) if event in boot.columns else np.asarray([], dtype=float)
            rows.append({
                "field": field,
                "event_kind": kind,
                "event_name": event,
                "event_basis": _event_basis(event, kind),
                "observed_day": obs,
                "bootstrap_median": _q(vals, 0.50),
                "q05": _q(vals, 0.05),
                "q95": _q(vals, 0.95),
                "q025": _q(vals, 0.025),
                "q975": _q(vals, 0.975),
                "valid_bootstrap_count": int(np.isfinite(vals).sum()),
                "event_status": status,
                "durability_window": _durability_window(event),
            })
    return pd.DataFrame(rows)


def _event_basis(event: str, kind: str) -> str:
    if event.startswith("departure") or event.startswith("durable_departure"):
        return "D_pre departure from pre-envelope"
    if "post_dominance" in event:
        return "P_dist > 0.5 and R_diff > 0"
    if event == "max_growth_day":
        return "field_change_norm_smooth3 maximum"
    if event == "postward_growth_peak_day":
        return "delta_P_dist_smooth3 maximum with delta_R_diff_smooth3 nonnegative"
    if event.startswith("rapid_growth"):
        return "field_change_norm_smooth3 >= q75 rapid-growth window"
    return kind


def _durability_window(event: str) -> Any:
    if event.endswith("_2d"):
        return 2
    if event.endswith("_3d"):
        return 3
    if event.endswith("_4d"):
        return 4
    return "none"


def _compare_events(
    base_events: pd.DataFrame,
    boot_events: pd.DataFrame,
    event_names: tuple[str, ...],
    kind: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    h_base = base_events[base_events["field"] == "H"]
    j_base = base_events[base_events["field"] == "Jw"]
    h_boot = boot_events[boot_events["field"] == "H"].copy()
    j_boot = boot_events[boot_events["field"] == "Jw"].copy()
    for event in event_names:
        h_status = str(h_base.iloc[0].get(f"status__{event}", "not_detected")) if not h_base.empty else "not_detected"
        j_status = str(j_base.iloc[0].get(f"status__{event}", "not_detected")) if not j_base.empty else "not_detected"
        h_obs = float(h_base.iloc[0].get(event, np.nan)) if not h_base.empty else np.nan
        j_obs = float(j_base.iloc[0].get(event, np.nan)) if not j_base.empty else np.nan
        if event in h_boot.columns and event in j_boot.columns:
            merged = h_boot[["bootstrap_id", event]].merge(j_boot[["bootstrap_id", event]], on="bootstrap_id", suffixes=("_H", "_Jw"))
            delta = merged[f"{event}_Jw"].to_numpy(dtype=float) - merged[f"{event}_H"].to_numpy(dtype=float)
        else:
            delta = np.asarray([], dtype=float)
        dec = _order_sync_decision(delta, margin_days=1.0)
        if h_status not in {"confirmed_event"} or j_status not in {"confirmed_event"}:
            # Preserve the numeric tendency if bootstraps exist, but mark the main event invalid if observed event is not confirmed for both fields.
            if dec["final_decision"] != "invalid_event":
                dec["decision_level"] = "method_unclosed" if dec["decision_level"] != "directional_tendency_only" else dec["decision_level"]
            if not np.isfinite(h_obs) or not np.isfinite(j_obs):
                dec["final_decision"] = "invalid_event"
        rows.append({
            "event_kind": kind,
            "event_name": event,
            "H_event_status": h_status,
            "Jw_event_status": j_status,
            "H_day_observed": h_obs,
            "Jw_day_observed": j_obs,
            "observed_delta_Jw_minus_H": float(j_obs - h_obs) if np.isfinite(h_obs) and np.isfinite(j_obs) else np.nan,
            "H_day_median": _q(h_boot[event].to_numpy(dtype=float), 0.50) if event in h_boot.columns else np.nan,
            "Jw_day_median": _q(j_boot[event].to_numpy(dtype=float), 0.50) if event in j_boot.columns else np.nan,
            **dec,
        })
    return pd.DataFrame(rows)


def _projection_comparison(proj_base: pd.DataFrame, proj_boot: pd.DataFrame, state_dec: pd.DataFrame, growth_dec: pd.DataFrame) -> pd.DataFrame:
    # Old projection comparison is intentionally simple: it records projection timing tendencies and whether a comparable state/growth decision supports it.
    rows: list[dict[str, Any]] = []
    proj_events = sorted([c for c in proj_boot.columns if c not in {"field", "bootstrap_id"}])
    h_base = proj_base[proj_base["field"] == "H"]
    j_base = proj_base[proj_base["field"] == "Jw"]
    h_boot = proj_boot[proj_boot["field"] == "H"]
    j_boot = proj_boot[proj_boot["field"] == "Jw"]
    for ev in proj_events:
        if ev not in h_boot.columns or ev not in j_boot.columns:
            continue
        merged = h_boot[["bootstrap_id", ev]].merge(j_boot[["bootstrap_id", ev]], on="bootstrap_id", suffixes=("_H", "_Jw"))
        delta = merged[f"{ev}_Jw"].to_numpy(dtype=float) - merged[f"{ev}_H"].to_numpy(dtype=float)
        dec = _order_sync_decision(delta, margin_days=1.0)
        h_obs = float(h_base.iloc[0].get(ev, np.nan)) if not h_base.empty else np.nan
        j_obs = float(j_base.iloc[0].get(ev, np.nan)) if not j_base.empty else np.nan
        if dec["final_decision"] in {"H_leads", "Jw_leads", "synchronous_equivalent"}:
            old_status = "supported_as_projection_result_only"
        elif dec["decision_level"] == "directional_tendency_only":
            old_status = "downgraded_to_tendency"
        else:
            old_status = "unresolved_projection_reference"
        rows.append({
            "old_projection_event": ev,
            "H_day_observed": h_obs,
            "Jw_day_observed": j_obs,
            "observed_delta_Jw_minus_H": float(j_obs - h_obs) if np.isfinite(h_obs) and np.isfinite(j_obs) else np.nan,
            **dec,
            "state_event_counterpart": _projection_counterpart(ev, "state"),
            "growth_event_counterpart": _projection_counterpart(ev, "growth"),
            "old_conclusion_status": old_status,
            "interpretation_boundary": "P_proj is old-method reference only; it is not a main state/growth decision in V7-u.",
        })
    return pd.DataFrame(rows)


def _projection_counterpart(event: str, target: str) -> str:
    if target == "state":
        if "t25" in event.lower():
            return "departure_from_pre or post_dominance_day depending on event meaning"
        if "t50" in event.lower() or "t75" in event.lower():
            return "post_dominance_day / durable_post_dominance_3d"
    if target == "growth":
        return "no direct counterpart unless event marks rapid-growth timing"
    return "not_comparable"


def _integrated_summary(state_dec: pd.DataFrame, growth_dec: pd.DataFrame, projection_comp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def _find(df: pd.DataFrame, event: str) -> str:
        sub = df[df["event_name"] == event]
        if sub.empty:
            return "missing"
        r = sub.iloc[0]
        return f"{r['final_decision']} ({r['decision_level']}; Δ={r.get('delta_median')})"
    rows.append({
        "question": "who_leaves_pre_first",
        "state_result": _find(state_dec, "durable_departure_from_pre_3d"),
        "growth_result": "not_a_growth_question",
        "interpretation_allowed": "Use durable departure searched after the pre-period only; non-durable departure_from_pre is a candidate diagnostic.",
        "interpretation_forbidden": "Do not infer post-state order from departure alone; do not use pre-period crossings as departure.",
        "method_status": _method_status_from_decision(_find(state_dec, "durable_departure_from_pre_3d")),
    })
    rows.append({
        "question": "who_becomes_post_like_first",
        "state_result": _find(state_dec, "post_dominance_day"),
        "growth_result": "not_a_growth_question",
        "interpretation_allowed": "This is the main state-progress order/synchrony question.",
        "interpretation_forbidden": "Do not replace with P_proj t50 if state event is invalid/unresolved.",
        "method_status": _method_status_from_decision(_find(state_dec, "post_dominance_day")),
    })
    rows.append({
        "question": "who_becomes_durably_post_like_first",
        "state_result": _find(state_dec, "durable_post_dominance_3d"),
        "growth_result": "not_a_growth_question",
        "interpretation_allowed": "Use as stronger state-progress order if hard decision exists.",
        "interpretation_forbidden": "Do not use first-passage as durable conversion.",
        "method_status": _method_status_from_decision(_find(state_dec, "durable_post_dominance_3d")),
    })
    rows.append({
        "question": "who_has_earlier_max_growth",
        "state_result": "not_a_state_question",
        "growth_result": _find(growth_dec, "max_growth_day"),
        "interpretation_allowed": "This is rapid-change timing, not state-transition order.",
        "interpretation_forbidden": "Do not call this state lead.",
        "method_status": _method_status_from_decision(_find(growth_dec, "max_growth_day")),
    })
    rows.append({
        "question": "who_has_earlier_postward_growth",
        "state_result": "not_a_state_question",
        "growth_result": _find(growth_dec, "postward_growth_peak_day"),
        "interpretation_allowed": "This is postward growth timing only.",
        "interpretation_forbidden": "Do not infer durable post-dominance from peak growth.",
        "method_status": _method_status_from_decision(_find(growth_dec, "postward_growth_peak_day")),
    })
    hard_state = int(state_dec["decision_level"].eq("hard_decision").sum()) if not state_dec.empty else 0
    hard_growth = int(growth_dec["decision_level"].eq("hard_decision").sum()) if not growth_dec.empty else 0
    rows.append({
        "question": "whether_state_and_growth_order_agree",
        "state_result": f"hard_state_decisions={hard_state}",
        "growth_result": f"hard_growth_decisions={hard_growth}",
        "interpretation_allowed": "Compare state and growth only after each layer has its own decision.",
        "interpretation_forbidden": "Do not collapse state and growth into one transition day.",
        "method_status": "usable_for_comparison" if hard_state + hard_growth > 0 else "method_unclosed",
    })
    return pd.DataFrame(rows)


def _method_status_from_decision(txt: str) -> str:
    if "H_leads" in txt or "Jw_leads" in txt or "synchronous_equivalent" in txt:
        return "hard_decision_available"
    if "directional_tendency_only" in txt:
        return "tendency_only"
    if "invalid" in txt or "missing" in txt:
        return "invalid_or_missing"
    return "method_unclosed"


def _method_status_table(state_dec: pd.DataFrame, growth_dec: pd.DataFrame, projection_comp: pd.DataFrame) -> pd.DataFrame:
    hard_state = int(state_dec["decision_level"].eq("hard_decision").sum()) if not state_dec.empty else 0
    hard_growth = int(growth_dec["decision_level"].eq("hard_decision").sum()) if not growth_dec.empty else 0
    invalid_state = int(state_dec["final_decision"].eq("invalid_event").sum()) if not state_dec.empty else 0
    invalid_growth = int(growth_dec["final_decision"].eq("invalid_event").sum()) if not growth_dec.empty else 0
    rows = []
    rows.append({
        "method_component": "state_framework_status",
        "status": "usable_for_H_Jw" if hard_state > 0 else ("partially_usable_tendency_only" if (state_dec["decision_level"].eq("directional_tendency_only").sum() if not state_dec.empty else 0) > 0 else "method_unclosed"),
        "evidence": f"hard_state_decisions={hard_state}; invalid_state_events={invalid_state}",
        "can_continue_to_all_fields": bool(hard_state > 0),
        "reason": "State layer has hard decision(s)." if hard_state > 0 else "State layer lacks hard decision for H/Jw.",
        "next_action": "extend_state_framework_after_review" if hard_state > 0 else "do_not_extend_state_framework_yet",
    })
    rows.append({
        "method_component": "growth_framework_status",
        "status": "usable_for_H_Jw" if hard_growth > 0 else ("partially_usable_tendency_only" if (growth_dec["decision_level"].eq("directional_tendency_only").sum() if not growth_dec.empty else 0) > 0 else "method_unclosed"),
        "evidence": f"hard_growth_decisions={hard_growth}; invalid_growth_events={invalid_growth}",
        "can_continue_to_all_fields": bool(hard_growth > 0),
        "reason": "Growth layer has hard decision(s)." if hard_growth > 0 else "Growth layer lacks hard decision for H/Jw.",
        "next_action": "extend_growth_framework_after_review" if hard_growth > 0 else "do_not_extend_growth_framework_yet",
    })
    rows.append({
        "method_component": "projection_method_status",
        "status": "reference_only",
        "evidence": "P_proj outputs are retained only in old projection comparison.",
        "can_continue_to_all_fields": False,
        "reason": "V7-t showed P_proj cannot independently bear order/synchrony adjudication.",
        "next_action": "do_not_use_projection_as_main_decision",
    })
    if hard_state > 0 and hard_growth > 0:
        overall = "usable_for_H_Jw"
        next_action = "review_results_then_consider_all_fields"
    elif hard_state > 0 or hard_growth > 0:
        overall = "partially_usable"
        next_action = "only_extend_the_layer_with_hard_decisions"
    else:
        overall = "method_unclosed"
        next_action = "return_to_transition_event_definition"
    rows.append({
        "method_component": "overall_method_status",
        "status": overall,
        "evidence": f"hard_state={hard_state}; hard_growth={hard_growth}",
        "can_continue_to_all_fields": bool(overall == "usable_for_H_Jw"),
        "reason": "Both state and growth layers have hard decisions." if overall == "usable_for_H_Jw" else "At least one layer lacks hard decisions.",
        "next_action": next_action,
    })
    return pd.DataFrame(rows)


def _write_figures(state_curves: pd.DataFrame, growth_curves: pd.DataFrame, state_dec: pd.DataFrame, growth_dec: pd.DataFrame, paths: V7UPaths) -> None:
    if os.environ.get("V7U_SKIP_FIGURES", "").strip() in {"1", "true", "True", "yes"}:
        _write_text("V7U_SKIP_FIGURES is set; figures skipped.\n", paths.figure_dir / "FIGURE_WARNING.txt")
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_text("matplotlib unavailable; figures skipped.\n", paths.figure_dir / "FIGURE_WARNING.txt")
        return
    _ensure_dir(paths.figure_dir)
    for metric_group, df, metrics, fname, title in [
        ("state", state_curves, ["P_dist", "R_diff", "P_proj_reference"], "w45_H_Jw_state_metric_curves_v7_u.png", "State metric curves"),
        ("growth", growth_curves, ["field_change_norm_smooth3", "delta_P_dist_smooth3", "delta_R_diff_smooth3"], "w45_H_Jw_growth_metric_curves_v7_u.png", "Growth metric curves"),
    ]:
        plt.figure(figsize=(10, 6))
        for field in FIELDS:
            sub = df[df["field"] == field]
            for metric in metrics:
                if metric in sub.columns:
                    plt.plot(sub["day"], sub[metric], marker="o", label=f"{field}:{metric}")
        plt.axvline(ANCHOR_DAY, linestyle="--", linewidth=1, label="anchor_day")
        plt.xlabel("day")
        plt.ylabel("metric value")
        plt.title(f"W45 H/Jw {title}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(paths.figure_dir / fname, dpi=160)
        plt.close()
    decision = pd.concat([
        state_dec.assign(layer="state") if not state_dec.empty else pd.DataFrame(),
        growth_dec.assign(layer="growth") if not growth_dec.empty else pd.DataFrame(),
    ], ignore_index=True)
    if not decision.empty:
        pivot = decision.pivot_table(index="event_name", columns="layer", values="final_decision", aggfunc="first")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(pivot))))
        ax.axis("off")
        table = ax.table(cellText=pivot.fillna("").values, rowLabels=pivot.index, colLabels=pivot.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)
        ax.set_title("W45 H/Jw state-growth decision heatmap")
        plt.tight_layout()
        plt.savefig(paths.figure_dir / "w45_H_Jw_state_growth_decision_heatmap_v7_u.png", dpi=160)
        plt.close()


def _summary_markdown(meta: dict[str, Any], state_dec: pd.DataFrame, growth_dec: pd.DataFrame, integrated: pd.DataFrame, method: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# V7-u W45 H/Jw state–growth separated transition framework")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("This branch separates state-progress information from rapid-growth information. It uses only H/Jw and does not add P/V/Je.")
    lines.append("")
    lines.append("## 2. What changed from V7-t")
    lines.append("- `P_proj` is downgraded to `P_proj_reference`.")
    lines.append("- Main state decisions are based on endpoint-distance and pattern-likeness.")
    lines.append("- Main growth decisions are based on raw-field daily change and postward growth metrics.")
    lines.append("- State lead and growth lead are not collapsed into one transition order.")
    lines.append("")
    lines.append("## 3. State order/synchrony decisions")
    lines.append("Hotfix note: departure events are searched only after the pre-period; `durable_departure_from_pre_3d` is the main departure event, while non-durable `departure_from_pre` is candidate-only.")
    lines.append("")
    if state_dec.empty:
        lines.append("No state decision rows were produced.")
    else:
        for _, row in state_dec.iterrows():
            lines.append(f"- {row['event_name']}: `{row['final_decision']}` / `{row['decision_level']}`; Δmedian={row.get('delta_median')}; P_H={row.get('P_H_leads')}; P_Jw={row.get('P_Jw_leads')}")
    lines.append("")
    lines.append("## 4. Growth order/synchrony decisions")
    if growth_dec.empty:
        lines.append("No growth decision rows were produced.")
    else:
        for _, row in growth_dec.iterrows():
            lines.append(f"- {row['event_name']}: `{row['final_decision']}` / `{row['decision_level']}`; Δmedian={row.get('delta_median')}; P_H={row.get('P_H_leads')}; P_Jw={row.get('P_Jw_leads')}")
    lines.append("")
    lines.append("## 5. Integrated state–growth reading")
    if not integrated.empty:
        for _, row in integrated.iterrows():
            lines.append(f"- {row['question']}: state=`{row['state_result']}`, growth=`{row['growth_result']}`, status=`{row['method_status']}`")
    lines.append("")
    lines.append("## 6. Method status")
    if not method.empty:
        for _, row in method.iterrows():
            lines.append(f"- {row['method_component']}: `{row['status']}`; next={row['next_action']}")
    lines.append("")
    lines.append("## 7. Interpretation boundary")
    lines.append("State-progress lead and growth-rate lead are separate. A growth decision must not be written as a state-transition order. Projection-only tendencies remain downgraded unless state/growth decisions support them.")
    lines.append("")
    lines.append("## 8. run_meta excerpt")
    lines.append("```json")
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2, default=str)[:4000])
    lines.append("```")
    return "\n".join(lines) + "\n"


def run_w45_H_Jw_state_growth_transition_framework_v7_u(v7_root: Optional[Path] = None) -> dict[str, Any]:
    paths = _resolve_paths(v7_root)
    settings = _configure_settings(paths)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)
    _progress_log(paths, "start")
    smoothed_path = settings.foundation.smoothed_fields_path()
    if not smoothed_path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields.npz at {smoothed_path}")
    _progress_log(paths, f"load raw025 smoothed fields: {smoothed_path}")
    smoothed = load_smoothed_fields(smoothed_path)
    raw_fields = {field: _prepare_raw_field(smoothed, field, settings) for field in FIELDS}
    n_days = min(obj["n_days"] for obj in raw_fields.values())
    window = _load_w45_window(paths, n_days=n_days)
    _progress_log(paths, f"window: {window}")
    state_curve_list: list[pd.DataFrame] = []
    growth_curve_list: list[pd.DataFrame] = []
    base_state_rows: list[dict[str, Any]] = []
    base_growth_rows: list[dict[str, Any]] = []
    base_projection_rows: list[dict[str, Any]] = []
    field_meta: dict[str, Any] = {}
    state_diag: dict[str, Any] = {}
    growth_diag: dict[str, Any] = {}
    for field, obj in raw_fields.items():
        _progress_log(paths, f"compute observed state/growth metrics for {field}")
        avg_state = _safe_nanmean(obj["data"], axis=0)
        state_df, growth_df, meta = _transition_metrics_for_avg_state(avg_state, obj["weights"], window)
        se, ss, sd = _state_events(state_df, window)
        ge, gs, gd = _growth_events(growth_df)
        pe = _projection_reference_events(state_df)
        state_df.insert(0, "field", field)
        growth_df.insert(0, "field", field)
        state_curve_list.append(state_df)
        growth_curve_list.append(growth_df)
        base_state_rows.append({"field": field, **se, **{f"status__{k}": v for k, v in ss.items()}})
        base_growth_rows.append({"field": field, **ge, **{f"status__{k}": v for k, v in gs.items()}})
        base_projection_rows.append({"field": field, **pe})
        state_diag[field] = sd
        growth_diag[field] = gd
        field_meta[field] = {
            **meta,
            "field_key": obj["field_key"],
            "n_years": obj["n_years"],
            "n_days": obj["n_days"],
            "lat_min": float(np.nanmin(obj["lat"])),
            "lat_max": float(np.nanmax(obj["lat"])),
            "lon_min": float(np.nanmin(obj["lon"])),
            "lon_max": float(np.nanmax(obj["lon"])),
            "n_lat": int(len(obj["lat"])),
            "n_lon": int(len(obj["lon"])),
        }
    state_curves = pd.concat(state_curve_list, ignore_index=True)
    growth_curves = pd.concat(growth_curve_list, ignore_index=True)
    base_state_events = pd.DataFrame(base_state_rows)
    base_growth_events = pd.DataFrame(base_growth_rows)
    base_projection_events = pd.DataFrame(base_projection_rows)
    _write_csv(state_curves, paths.output_dir / "w45_H_Jw_state_metric_curves_v7_u.csv")
    _write_csv(growth_curves, paths.output_dir / "w45_H_Jw_growth_metric_curves_v7_u.csv")
    _write_csv(base_state_events, paths.output_dir / "w45_H_Jw_state_observed_events_v7_u.csv")
    _write_csv(base_growth_events, paths.output_dir / "w45_H_Jw_growth_observed_events_v7_u.csv")
    _write_csv(base_projection_events, paths.output_dir / "w45_H_Jw_projection_reference_observed_events_v7_u.csv")

    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    _progress_log(paths, f"bootstrap state/growth events: n={n_boot}")
    state_boots: list[pd.DataFrame] = []
    growth_boots: list[pd.DataFrame] = []
    proj_boots: list[pd.DataFrame] = []
    for idx, field in enumerate(FIELDS):
        sboot, gboot, pboot = _bootstrap_events(raw_fields[field], window, n_boot=n_boot, seed=int(settings.bootstrap.random_seed) + 2000 * idx)
        state_boots.append(sboot)
        growth_boots.append(gboot)
        proj_boots.append(pboot)
    state_boot = pd.concat(state_boots, ignore_index=True) if state_boots else pd.DataFrame()
    growth_boot = pd.concat(growth_boots, ignore_index=True) if growth_boots else pd.DataFrame()
    proj_boot = pd.concat(proj_boots, ignore_index=True) if proj_boots else pd.DataFrame()
    _write_csv(state_boot, paths.output_dir / "w45_H_Jw_state_bootstrap_events_v7_u.csv")
    _write_csv(growth_boot, paths.output_dir / "w45_H_Jw_growth_bootstrap_events_v7_u.csv")
    _write_csv(proj_boot, paths.output_dir / "w45_H_Jw_projection_reference_bootstrap_events_v7_u.csv")

    _progress_log(paths, "write registries and decisions")
    state_registry = _registry_from_events(base_state_events, state_boot, STATE_EVENT_NAMES, "state")
    growth_registry = _registry_from_events(base_growth_events, growth_boot, GROWTH_EVENT_NAMES, "growth")
    state_decision = _compare_events(base_state_events, state_boot, STATE_EVENT_NAMES, "state")
    growth_decision = _compare_events(base_growth_events, growth_boot, GROWTH_EVENT_NAMES, "growth")
    projection_comp = _projection_comparison(base_projection_events, proj_boot, state_decision, growth_decision)
    integrated = _integrated_summary(state_decision, growth_decision, projection_comp)
    method_status = _method_status_table(state_decision, growth_decision, projection_comp)
    _write_csv(state_registry, paths.output_dir / "w45_H_Jw_state_event_registry_v7_u.csv")
    _write_csv(growth_registry, paths.output_dir / "w45_H_Jw_growth_event_registry_v7_u.csv")
    _write_csv(state_decision, paths.output_dir / "w45_H_Jw_state_order_sync_decision_v7_u.csv")
    _write_csv(growth_decision, paths.output_dir / "w45_H_Jw_growth_order_sync_decision_v7_u.csv")
    _write_csv(integrated, paths.output_dir / "w45_H_Jw_state_growth_integrated_summary_v7_u.csv")
    _write_csv(projection_comp, paths.output_dir / "w45_H_Jw_projection_vs_state_growth_comparison_v7_u.csv")
    _write_csv(method_status, paths.output_dir / "w45_H_Jw_state_growth_method_status_v7_u.csv")

    input_audit = {
        "smoothed_fields_path": str(smoothed_path),
        "available_keys": sorted(list(smoothed.keys())),
        "field_meta": field_meta,
        "window": window,
        "state_event_diagnostics": state_diag,
        "growth_event_diagnostics": growth_diag,
        "v7t_output_dir_exists_for_context": paths.v7t_output_dir.exists(),
    }
    _write_json(input_audit, paths.output_dir / "input_audit_v7_u.json")

    meta: dict[str, Any] = {
        "version": "v7_u",
        "hotfix_id": HOTFIX_ID,
        "output_tag": OUTPUT_TAG,
        "status": "success",
        "created_at": _now_iso(),
        "primary_goal": "separate state-progress information from rapid-growth information for H/Jw only",
        "main_input_representation": "raw025_smoothed_field",
        "smoothed_fields_path": str(smoothed_path),
        "fields": list(FIELDS),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "window": window,
        "n_bootstrap": n_boot,
        "state_metrics": ["D_pre", "D_post", "P_dist", "R_pre", "R_post", "R_diff"],
        "growth_metrics": ["field_change_norm", "delta_P_dist", "delta_R_diff"],
        "P_proj_role": "reference_only_not_main_decision",
        "state_events": list(STATE_EVENT_NAMES),
        "growth_events": list(GROWTH_EVENT_NAMES),
        "synchrony_requires_positive_equivalence_test": True,
        "order_margin_days": 1.0,
        "hard_decision_probability_threshold": 0.90,
        "no_spatial_pairing": True,
        "no_latband_pairing": True,
        "no_complex_relation_labels": True,
        "extension_rule": "Do not add P/V/Je until H/Jw state-growth framework is reviewed.",
        "departure_hotfix_rule": "departure search starts after pre_period_end; durable_departure_from_pre_3d is the main departure event; non-durable departure_from_pre is candidate-only",
        "field_meta": field_meta,
        "key_outputs": [
            "w45_H_Jw_state_metric_curves_v7_u.csv",
            "w45_H_Jw_growth_metric_curves_v7_u.csv",
            "w45_H_Jw_state_order_sync_decision_v7_u.csv",
            "w45_H_Jw_growth_order_sync_decision_v7_u.csv",
            "w45_H_Jw_projection_vs_state_growth_comparison_v7_u.csv",
            "w45_H_Jw_state_growth_method_status_v7_u.csv",
        ],
    }
    _write_json(meta, paths.output_dir / "run_meta.json")

    _progress_log(paths, "write summary and figures")
    summary = _summary_markdown(meta, state_decision, growth_decision, integrated, method_status)
    _write_text(summary, paths.output_dir / "w45_H_Jw_state_growth_transition_framework_summary_v7_u.md")
    _write_figures(state_curves, growth_curves, state_decision, growth_decision, paths)
    _progress_log(paths, "finished success")
    return meta
