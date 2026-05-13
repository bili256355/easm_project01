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

from .config import StagePartitionV7Settings

OUTPUT_TAG = "w45_H_Jw_baseline_sensitive_state_growth_v7_v"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
ACCEPTED_WINDOW = (40, 48)
FIELDS = ("H", "Jw")
FIELD_SPECS = {
    "H": {"field_key": "z500_smoothed", "domain_attr_lon": "h_lon_range", "domain_attr_lat": "h_lat_range"},
    "Jw": {"field_key": "u200_smoothed", "domain_attr_lon": "jw_lon_range", "domain_attr_lat": "jw_lat_range"},
}
STATE_BRANCHES = ("distance", "pattern")
STATE_EVENT_NAMES = ("S25_day", "S50_day", "S75_day", "durable_S50_day", "durable_S75_day")
GROWTH_EVENT_NAMES = (
    "growth_onset_day",
    "growth_peak_day",
    "growth_window_start",
    "growth_window_center",
    "growth_window_end",
)
NEGATIVE_EVENT_NAMES = ("negative_growth_start", "negative_growth_peak", "negative_growth_end", "negative_growth_center")
EPS = 1.0e-12
PATTERN_DYNAMIC_RANGE_FLOOR = 1.0e-8
MIN_BOOT_VALID_FRACTION = 0.80


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    pre: tuple[int, int]
    post: tuple[int, int]
    search: tuple[int, int]
    diagnostic: tuple[int, int]
    role: str
    notes: str


BASELINE_CONFIGS = (
    BaselineConfig(
        name="C0_full_stage",
        pre=(0, 39),
        post=(49, 74),
        search=(35, 53),
        diagnostic=(0, 74),
        role="main_candidate_full_non_transition_stage_unified_expanded_search",
        notes="Use the full non-accepted-transition intervals before and after W45 as pre/post; use W45 ±5d expanded event-search window to avoid treating the accepted core as the full event search window.",
    ),
    BaselineConfig(
        name="C1_buffered_stage",
        pre=(0, 34),
        post=(54, 69),
        search=(35, 53),
        diagnostic=(0, 74),
        role="buffered_sensitivity_candidate",
        notes="Exclude five-day buffers around W45 and before W003/day81.",
    ),
    BaselineConfig(
        name="C2_immediate_pre",
        pre=(25, 34),
        post=(54, 69),
        search=(35, 53),
        diagnostic=(0, 74),
        role="immediate_pre_sensitivity_candidate",
        notes="Use the near-W45 pre-core to test dependence on immediate background state.",
    ),
)


@dataclass
class V7VPaths:
    v7_root: Path
    project_root: Path
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


def _load_smoothed_fields_npz(path: Path) -> dict[str, Any]:
    """Load raw025 smoothed-field npz without depending on older V6 packages.

    This keeps V7-v self-contained. The expected file contains keys such as
    z500_smoothed, u200_smoothed, lat, lon, and years.
    """
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _resolve_paths(v7_root: Optional[Path]) -> V7VPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = v7_root.parents[1]
    return V7VPaths(
        v7_root=v7_root,
        project_root=project_root,
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _configure_settings(paths: V7VPaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    env_debug = os.environ.get("V7V_DEBUG_N_BOOTSTRAP", "").strip()
    if env_debug:
        settings.bootstrap.debug_n_bootstrap = int(env_debug)
    return settings


def _progress_log(paths: V7VPaths, message: str) -> None:
    _ensure_dir(paths.log_dir)
    with (paths.log_dir / "run_progress_v7_v.log").open("a", encoding="utf-8") as f:
        f.write(f"[{_now_iso()}] {message}\n")
    print(f"[V7-v] {message}", flush=True)


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


def _period_days(period: tuple[int, int], n_days: int) -> np.ndarray:
    start, end = int(period[0]), int(period[1])
    start = max(0, start)
    end = min(n_days - 1, end)
    if end < start:
        raise ValueError(f"Invalid period {period} for n_days={n_days}")
    return np.arange(start, end + 1, dtype=int)


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
        seg = arr[lo:hi]
        out[i] = float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan
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


def _contiguous_segments(cond: np.ndarray) -> list[tuple[int, int]]:
    cond = np.asarray(cond, dtype=bool)
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
    return segments


def _compute_state_curves_for_avg_state(
    avg_state: np.ndarray,
    weights_2d: np.ndarray,
    cfg: BaselineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = np.asarray(avg_state, dtype=float)
    n_days = state.shape[0]
    diag_days = _period_days(cfg.diagnostic, n_days)
    pre_days = _period_days(cfg.pre, n_days)
    post_days = _period_days(cfg.post, n_days)
    pre = _safe_nanmean(state[pre_days, :, :], axis=0)
    post = _safe_nanmean(state[post_days, :, :], axis=0)
    d = post - pre
    w = np.asarray(weights_2d, dtype=float)
    denom = _weighted_sum(d * d, w)
    rows: list[dict[str, Any]] = []
    rdiff_values: list[float] = []
    for day in diag_days:
        x = state[int(day), :, :]
        d_pre = _weighted_rms(x - pre, w)
        d_post = _weighted_rms(x - post, w)
        s_dist = float(d_pre / (d_pre + d_post)) if np.isfinite(d_pre) and np.isfinite(d_post) and (d_pre + d_post) > EPS else np.nan
        r_pre = _weighted_corr(x, pre, w)
        r_post = _weighted_corr(x, post, w)
        r_diff = float(r_post - r_pre) if np.isfinite(r_pre) and np.isfinite(r_post) else np.nan
        p_proj = np.nan
        if np.isfinite(denom) and abs(denom) > EPS:
            p_proj = _weighted_sum((x - pre) * d, w) / denom
        rows.append({
            "day": int(day),
            "D_pre": d_pre,
            "D_post": d_post,
            "S_dist": s_dist,
            "R_pre": r_pre,
            "R_post": r_post,
            "R_diff": r_diff,
            "P_proj_reference": p_proj,
        })
        rdiff_values.append(r_diff)
    df = pd.DataFrame(rows)
    # Pattern progress normalization from pre/post-period R_diff values.
    r_pre_period = df.loc[df["day"].isin(pre_days), "R_diff"].to_numpy(dtype=float)
    r_post_period = df.loc[df["day"].isin(post_days), "R_diff"].to_numpy(dtype=float)
    r0 = float(np.nanmean(r_pre_period)) if np.isfinite(r_pre_period).any() else np.nan
    r1 = float(np.nanmean(r_post_period)) if np.isfinite(r_post_period).any() else np.nan
    dyn = float(r1 - r0) if np.isfinite(r0) and np.isfinite(r1) else np.nan
    if np.isfinite(dyn) and abs(dyn) >= PATTERN_DYNAMIC_RANGE_FLOOR:
        df["S_pattern"] = (df["R_diff"].astype(float) - r0) / dyn
        pattern_valid = True
        pattern_flag = "valid"
    else:
        df["S_pattern"] = np.nan
        pattern_valid = False
        pattern_flag = "low_dynamic_range"
    meta = {
        "pre_days": [int(x) for x in pre_days],
        "post_days": [int(x) for x in post_days],
        "diagnostic_days": [int(x) for x in diag_days],
        "R0_pre_mean": r0,
        "R1_post_mean": r1,
        "S_pattern_dynamic_range": dyn,
        "S_pattern_branch_valid": pattern_valid,
        "S_pattern_branch_validity_flag": pattern_flag,
        "projection_denom": denom,
        "transition_norm": _weighted_rms(d, w),
    }
    return df, meta


def _compute_growth_curves(state_df: pd.DataFrame, cfg: BaselineConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    days = state_df["day"].to_numpy(dtype=int)
    meta: dict[str, Any] = {}
    for branch, s_col in (("distance", "S_dist"), ("pattern", "S_pattern")):
        s = state_df[s_col].to_numpy(dtype=float) if s_col in state_df.columns else np.full(days.shape, np.nan)
        v = np.full(s.shape, np.nan, dtype=float)
        v[1:] = s[1:] - s[:-1]
        vs = _rolling_mean_centered(v, window=3)
        pre_days = _period_days(cfg.pre, int(days.max()) + 1 if days.size else 0)
        pre_mask = np.isin(days, pre_days)
        positive_thr = _q(vs[pre_mask], 0.90) if pre_mask.any() else np.nan
        negative_thr = _q(vs[pre_mask], 0.10) if pre_mask.any() else np.nan
        if not np.isfinite(positive_thr):
            positive_thr = np.nan
        if not np.isfinite(negative_thr):
            negative_thr = np.nan
        meta[f"{branch}_positive_growth_threshold_q90"] = positive_thr
        meta[f"{branch}_negative_growth_threshold_q10"] = negative_thr
        for day, sv, vv, vvs in zip(days, s, v, vs):
            rows.append({
                "day": int(day),
                "branch": branch,
                "S": sv,
                "V": vv,
                "V_smooth3": vvs,
                "positive_growth_threshold": positive_thr,
                "negative_growth_threshold": negative_thr,
                "growth_flag": bool(np.isfinite(vvs) and np.isfinite(positive_thr) and vvs > positive_thr),
                "negative_growth_flag": bool(np.isfinite(vvs) and np.isfinite(negative_thr) and vvs < negative_thr),
            })
    return pd.DataFrame(rows), meta


def _state_event_status_for_threshold(days: np.ndarray, s: np.ndarray, cfg: BaselineConfig, threshold: float, durable: bool = False, stable_days: int = 3) -> tuple[float, str, str]:
    search_start, search_end = cfg.search
    search_mask = (days >= search_start) & (days <= search_end)
    before_mask = days < search_start
    after_mask = days > search_end
    cond = np.isfinite(s) & (s >= float(threshold))
    # If threshold is already reached before search, the event is not observable in this search window.
    if before_mask.any() and bool(np.any(cond[before_mask])):
        first_all = _first_true(days, cond)
        return first_all, "left_censored", "threshold_already_reached_before_search_start"
    if not search_mask.any():
        return np.nan, "invalid_event", "empty_search_window"
    search_days = days[search_mask]
    search_cond = cond[search_mask]
    if durable:
        event_day = _consecutive_true_first(search_days, search_cond, stable_days=stable_days)
        if np.isfinite(event_day):
            return event_day, "valid", ""
        first = _first_true(search_days, search_cond)
        if np.isfinite(first):
            return first, "not_durable", f"first_passage_not_sustained_{stable_days}d"
        if after_mask.any() and bool(np.any(cond[after_mask])):
            return _first_true(days[after_mask], cond[after_mask]), "right_censored", "threshold_reached_after_search_end"
        return np.nan, "not_detected", "threshold_not_reached"
    event_day = _first_true(search_days, search_cond)
    if np.isfinite(event_day):
        return event_day, "valid", ""
    if after_mask.any() and bool(np.any(cond[after_mask])):
        return _first_true(days[after_mask], cond[after_mask]), "right_censored", "threshold_reached_after_search_end"
    return np.nan, "not_detected", "threshold_not_reached"


def _state_events_for_branch(state_df: pd.DataFrame, cfg: BaselineConfig, branch: str, branch_meta: dict[str, Any]) -> pd.DataFrame:
    days = state_df["day"].to_numpy(dtype=int)
    s_col = "S_dist" if branch == "distance" else "S_pattern"
    s = state_df[s_col].to_numpy(dtype=float) if s_col in state_df.columns else np.full(days.shape, np.nan)
    rows: list[dict[str, Any]] = []
    if branch == "pattern" and not bool(branch_meta.get("S_pattern_branch_valid", False)):
        for event in STATE_EVENT_NAMES:
            rows.append({
                "branch": branch,
                "event_family": "state",
                "event_name": event,
                "observed_day": np.nan,
                "event_status": "low_dynamic_range",
                "invalid_reason": "S_pattern_dynamic_range_below_floor",
                "threshold": _event_threshold(event),
                "dynamic_range": branch_meta.get("S_pattern_dynamic_range", np.nan),
            })
        return pd.DataFrame(rows)
    for event in STATE_EVENT_NAMES:
        thr = _event_threshold(event)
        durable = event.startswith("durable")
        day, status, reason = _state_event_status_for_threshold(days, s, cfg, thr, durable=durable, stable_days=3)
        rows.append({
            "branch": branch,
            "event_family": "state",
            "event_name": event,
            "observed_day": day,
            "event_status": status,
            "invalid_reason": reason,
            "threshold": thr,
            "dynamic_range": branch_meta.get("S_pattern_dynamic_range", np.nan) if branch == "pattern" else np.nan,
        })
    return pd.DataFrame(rows)


def _event_threshold(event_name: str) -> float:
    if "S25" in event_name:
        return 0.25
    if "S50" in event_name:
        return 0.50
    if "S75" in event_name:
        return 0.75
    return np.nan


def _growth_events_for_branch(growth_df: pd.DataFrame, cfg: BaselineConfig, branch: str) -> pd.DataFrame:
    g = growth_df[growth_df["branch"] == branch].copy()
    rows: list[dict[str, Any]] = []
    if g.empty:
        for event in GROWTH_EVENT_NAMES + NEGATIVE_EVENT_NAMES:
            rows.append({"branch": branch, "event_family": "growth" if event in GROWTH_EVENT_NAMES else "negative_growth", "event_name": event, "observed_day": np.nan, "event_status": "invalid_branch", "invalid_reason": "missing_growth_branch"})
        return pd.DataFrame(rows)
    days = g["day"].to_numpy(dtype=int)
    v = g["V_smooth3"].to_numpy(dtype=float)
    pos_thr = float(g["positive_growth_threshold"].dropna().iloc[0]) if g["positive_growth_threshold"].notna().any() else np.nan
    neg_thr = float(g["negative_growth_threshold"].dropna().iloc[0]) if g["negative_growth_threshold"].notna().any() else np.nan
    search_start, search_end = cfg.search
    search_mask = (days >= search_start) & (days <= search_end)
    before_mask = days < search_start
    after_mask = days > search_end
    if not np.isfinite(pos_thr):
        for event in GROWTH_EVENT_NAMES:
            rows.append({"branch": branch, "event_family": "growth", "event_name": event, "observed_day": np.nan, "event_status": "invalid_event", "invalid_reason": "invalid_positive_growth_threshold", "threshold": pos_thr})
    else:
        cond = np.isfinite(v) & (v > pos_thr)
        # onset: first 2 consecutive days above threshold within search, with censor check.
        if before_mask.any():
            pre_segments = _contiguous_segments(cond[days < search_start])
            if pre_segments and bool(cond[search_mask][0]) if search_mask.any() else False:
                left_censored_onset = True
            else:
                left_censored_onset = False
        else:
            left_censored_onset = False
        search_days = days[search_mask]
        search_cond = cond[search_mask]
        onset = np.nan
        onset_status = "not_detected"
        onset_reason = "no_2d_positive_growth_run"
        if left_censored_onset:
            onset_status = "left_censored"
            onset_reason = "growth_run_already_active_before_search_start"
        else:
            onset = _consecutive_true_first(search_days, search_cond, stable_days=2) if search_mask.any() else np.nan
            if np.isfinite(onset):
                onset_status = "valid"
                onset_reason = ""
        rows.append({"branch": branch, "event_family": "growth", "event_name": "growth_onset_day", "observed_day": onset, "event_status": onset_status, "invalid_reason": onset_reason, "threshold": pos_thr})
        # peak day.
        if search_mask.any() and np.isfinite(v[search_mask]).any():
            search_idxs = np.where(search_mask)[0]
            peak_idx = search_idxs[int(np.nanargmax(v[search_mask]))]
            peak_day = float(days[peak_idx])
            if int(peak_day) in {search_start, search_end}:
                peak_status = "boundary_peak"
                peak_reason = "peak_on_search_boundary"
            else:
                peak_status = "valid"
                peak_reason = ""
        else:
            peak_day = np.nan
            peak_status = "not_detected"
            peak_reason = "no_finite_speed_in_search"
        rows.append({"branch": branch, "event_family": "growth", "event_name": "growth_peak_day", "observed_day": peak_day, "event_status": peak_status, "invalid_reason": peak_reason, "threshold": pos_thr})
        # positive growth window.
        segments = _contiguous_segments(search_cond) if search_mask.any() else []
        if segments:
            search_indices = np.where(search_mask)[0]
            def score(seg: tuple[int, int]) -> float:
                lo, hi = seg
                idx = search_indices[lo : hi + 1]
                return float(np.nansum(np.maximum(v[idx], 0.0)))
            best = max(segments, key=score)
            lo, hi = best
            idxs = search_indices[lo : hi + 1]
            seg_days = days[idxs].astype(float)
            weights = np.maximum(v[idxs], 0.0)
            center = float(np.nansum(seg_days * weights) / np.nansum(weights)) if np.nansum(weights) > EPS else float(np.nanmean(seg_days))
            start_day, end_day = float(seg_days[0]), float(seg_days[-1])
            # Censor if segment touches search boundary and continues outside.
            win_status = "valid"
            win_reason = ""
            if int(start_day) == search_start:
                prev_idx = np.where(days == search_start - 1)[0]
                if prev_idx.size and bool(cond[prev_idx[0]]):
                    win_status = "left_censored"
                    win_reason = "positive_growth_window_active_before_search_start"
            if int(end_day) == search_end:
                next_idx = np.where(days == search_end + 1)[0]
                if next_idx.size and bool(cond[next_idx[0]]):
                    win_status = "right_censored"
                    win_reason = "positive_growth_window_continues_after_search_end"
            peak_in_win = float(days[idxs[int(np.nanargmax(v[idxs]))]]) if np.isfinite(v[idxs]).any() else np.nan
            rows.extend([
                {"branch": branch, "event_family": "growth", "event_name": "growth_window_start", "observed_day": start_day, "event_status": win_status, "invalid_reason": win_reason, "threshold": pos_thr},
                {"branch": branch, "event_family": "growth", "event_name": "growth_window_center", "observed_day": center, "event_status": win_status, "invalid_reason": win_reason, "threshold": pos_thr},
                {"branch": branch, "event_family": "growth", "event_name": "growth_window_end", "observed_day": end_day, "event_status": win_status, "invalid_reason": win_reason, "threshold": pos_thr},
            ])
        else:
            for event in ("growth_window_start", "growth_window_center", "growth_window_end"):
                rows.append({"branch": branch, "event_family": "growth", "event_name": event, "observed_day": np.nan, "event_status": "not_detected", "invalid_reason": "no_positive_growth_window", "threshold": pos_thr})
    # Negative growth diagnostics, not used for H/Jw order decisions.
    if np.isfinite(neg_thr):
        ncond = np.isfinite(v) & (v < neg_thr)
        search_indices = np.where(search_mask)[0]
        segments = _contiguous_segments(ncond[search_mask]) if search_mask.any() else []
        if segments:
            def nscore(seg: tuple[int, int]) -> float:
                lo, hi = seg
                idx = search_indices[lo : hi + 1]
                return float(np.nansum(np.minimum(v[idx], 0.0)))
            best = min(segments, key=nscore)
            lo, hi = best
            idxs = search_indices[lo : hi + 1]
            seg_days = days[idxs].astype(float)
            neg_weights = np.abs(np.minimum(v[idxs], 0.0))
            center = float(np.nansum(seg_days * neg_weights) / np.nansum(neg_weights)) if np.nansum(neg_weights) > EPS else float(np.nanmean(seg_days))
            peak = float(days[idxs[int(np.nanargmin(v[idxs]))]]) if np.isfinite(v[idxs]).any() else np.nan
            status, reason = "diagnostic_only", "negative_growth_not_used_for_pairwise_order"
            vals = {
                "negative_growth_start": float(seg_days[0]),
                "negative_growth_peak": peak,
                "negative_growth_end": float(seg_days[-1]),
                "negative_growth_center": center,
            }
        else:
            vals = {name: np.nan for name in NEGATIVE_EVENT_NAMES}
            status, reason = "not_detected", "no_negative_growth_window"
    else:
        vals = {name: np.nan for name in NEGATIVE_EVENT_NAMES}
        status, reason = "invalid_event", "invalid_negative_growth_threshold"
    for event, day in vals.items():
        rows.append({"branch": branch, "event_family": "negative_growth", "event_name": event, "observed_day": day, "event_status": status, "invalid_reason": reason, "threshold": neg_thr})
    return pd.DataFrame(rows)


def _compute_all_for_field_config(field_obj: dict[str, Any], cfg: BaselineConfig, avg_state: Optional[np.ndarray] = None) -> dict[str, Any]:
    data = np.asarray(field_obj["data"], dtype=float)
    if avg_state is None:
        avg_state = _safe_nanmean(data, axis=0)
    state_df, meta = _compute_state_curves_for_avg_state(avg_state, field_obj["weights"], cfg)
    growth_df, gmeta = _compute_growth_curves(state_df, cfg)
    state_event_frames = []
    growth_event_frames = []
    for branch in STATE_BRANCHES:
        state_event_frames.append(_state_events_for_branch(state_df, cfg, branch, meta))
        growth_event_frames.append(_growth_events_for_branch(growth_df, cfg, branch))
    state_events = pd.concat(state_event_frames, ignore_index=True)
    growth_events = pd.concat(growth_event_frames, ignore_index=True)
    return {"state_curves": state_df, "growth_curves": growth_df, "state_events": state_events, "growth_events": growth_events, "meta": {**meta, **gmeta}}


def _bootstrap_field_config(field_obj: dict[str, Any], cfg: BaselineConfig, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    data = np.asarray(field_obj["data"], dtype=float)
    n_years = int(data.shape[0])
    state_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    for b in range(int(n_boot)):
        idx = rng.integers(0, n_years, size=n_years)
        avg_state = _safe_nanmean(data[idx, :, :, :], axis=0)
        out = _compute_all_for_field_config(field_obj, cfg, avg_state=avg_state)
        for _, row in out["state_events"].iterrows():
            state_rows.append({
                "bootstrap_id": int(b),
                "field": field_obj["field"],
                "baseline_config": cfg.name,
                "branch": row["branch"],
                "event_name": row["event_name"],
                "event_family": row["event_family"],
                "event_day": row["observed_day"],
                "event_status": row["event_status"],
            })
        for _, row in out["growth_events"].iterrows():
            growth_rows.append({
                "bootstrap_id": int(b),
                "field": field_obj["field"],
                "baseline_config": cfg.name,
                "branch": row["branch"],
                "event_name": row["event_name"],
                "event_family": row["event_family"],
                "event_day": row["observed_day"],
                "event_status": row["event_status"],
            })
    return pd.DataFrame(state_rows), pd.DataFrame(growth_rows)


def _registry_from_long_events(obs: pd.DataFrame, boot: pd.DataFrame, event_family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["baseline_config", "field", "branch", "event_name"]
    for keys, odf in obs[obs["event_family"] == event_family].groupby(group_cols, dropna=False):
        baseline_config, field, branch, event_name = keys
        bdf = boot[(boot["baseline_config"] == baseline_config) & (boot["field"] == field) & (boot["branch"] == branch) & (boot["event_name"] == event_name)]
        vals = bdf["event_day"].to_numpy(dtype=float) if not bdf.empty else np.asarray([], dtype=float)
        valid = bdf[bdf["event_status"] == "valid"]
        valid_vals = valid["event_day"].to_numpy(dtype=float) if not valid.empty else np.asarray([], dtype=float)
        row = odf.iloc[0]
        rows.append({
            "baseline_config": baseline_config,
            "field": field,
            "branch": branch,
            "event_name": event_name,
            "observed_day": row.get("observed_day", np.nan),
            "event_status": row.get("event_status", "not_detected"),
            "invalid_reason": row.get("invalid_reason", ""),
            "bootstrap_median": _q(valid_vals, 0.50),
            "q05": _q(valid_vals, 0.05),
            "q95": _q(valid_vals, 0.95),
            "q025": _q(valid_vals, 0.025),
            "q975": _q(valid_vals, 0.975),
            "valid_bootstrap_count": int(np.isfinite(valid_vals).sum()),
            "total_bootstrap_count": int(len(bdf)),
            "valid_fraction": float(len(valid) / len(bdf)) if len(bdf) else np.nan,
        })
    return pd.DataFrame(rows)


def _order_sync_decision_from_deltas(deltas: np.ndarray, valid_fraction: float, margin_days: float = 1.0) -> dict[str, Any]:
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(valid_fraction) or valid_fraction < MIN_BOOT_VALID_FRACTION:
        return {
            "delta_median": np.nan,
            "delta_q05": np.nan,
            "delta_q95": np.nan,
            "delta_q025": np.nan,
            "delta_q975": np.nan,
            "P_H_leads": np.nan,
            "P_Jw_leads": np.nan,
            "sync_equivalence_pass": False,
            "bootstrap_valid_fraction": valid_fraction,
            "final_decision": "invalid_event",
            "decision_level": "invalid_or_low_valid_fraction",
        }
    q025, q975 = _q(arr, 0.025), _q(arr, 0.975)
    p_h = float(np.mean(arr > float(margin_days)))
    p_j = float(np.mean(arr < -float(margin_days)))
    sync = bool(np.isfinite(q025) and np.isfinite(q975) and q025 >= -float(margin_days) and q975 <= float(margin_days))
    if p_h >= 0.90:
        decision, level = "H_leads", "hard_decision"
    elif p_j >= 0.90:
        decision, level = "Jw_leads", "hard_decision"
    elif sync:
        decision, level = "synchronous_equivalent", "hard_decision"
    else:
        decision = "unresolved"
        level = "directional_tendency_only" if max(p_h, p_j) >= 0.67 else "method_unclosed"
    return {
        "delta_median": _q(arr, 0.50),
        "delta_q05": _q(arr, 0.05),
        "delta_q95": _q(arr, 0.95),
        "delta_q025": q025,
        "delta_q975": q975,
        "P_H_leads": p_h,
        "P_Jw_leads": p_j,
        "sync_equivalence_pass": sync,
        "bootstrap_valid_fraction": valid_fraction,
        "final_decision": decision,
        "decision_level": level,
    }


def _compare_long_events(obs: pd.DataFrame, boot: pd.DataFrame, event_family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sub_obs = obs[obs["event_family"] == event_family].copy()
    keys = sorted(set(zip(sub_obs["baseline_config"], sub_obs["branch"], sub_obs["event_name"])))
    for baseline_config, branch, event_name in keys:
        h_obs = sub_obs[(sub_obs["baseline_config"] == baseline_config) & (sub_obs["branch"] == branch) & (sub_obs["event_name"] == event_name) & (sub_obs["field"] == "H")]
        j_obs = sub_obs[(sub_obs["baseline_config"] == baseline_config) & (sub_obs["branch"] == branch) & (sub_obs["event_name"] == event_name) & (sub_obs["field"] == "Jw")]
        h_status = str(h_obs.iloc[0]["event_status"]) if not h_obs.empty else "missing"
        j_status = str(j_obs.iloc[0]["event_status"]) if not j_obs.empty else "missing"
        h_day = float(h_obs.iloc[0]["observed_day"]) if not h_obs.empty else np.nan
        j_day = float(j_obs.iloc[0]["observed_day"]) if not j_obs.empty else np.nan
        b_h = boot[(boot["baseline_config"] == baseline_config) & (boot["branch"] == branch) & (boot["event_name"] == event_name) & (boot["field"] == "H") & (boot["event_status"] == "valid")]
        b_j = boot[(boot["baseline_config"] == baseline_config) & (boot["branch"] == branch) & (boot["event_name"] == event_name) & (boot["field"] == "Jw") & (boot["event_status"] == "valid")]
        all_h = boot[(boot["baseline_config"] == baseline_config) & (boot["branch"] == branch) & (boot["event_name"] == event_name) & (boot["field"] == "H")]
        all_j = boot[(boot["baseline_config"] == baseline_config) & (boot["branch"] == branch) & (boot["event_name"] == event_name) & (boot["field"] == "Jw")]
        merged = b_h[["bootstrap_id", "event_day"]].merge(b_j[["bootstrap_id", "event_day"]], on="bootstrap_id", suffixes=("_H", "_Jw"))
        deltas = merged["event_day_Jw"].to_numpy(dtype=float) - merged["event_day_H"].to_numpy(dtype=float) if not merged.empty else np.asarray([], dtype=float)
        total_pairs = min(len(all_h), len(all_j))
        valid_fraction = float(len(merged) / total_pairs) if total_pairs else np.nan
        dec = _order_sync_decision_from_deltas(deltas, valid_fraction=valid_fraction, margin_days=1.0)
        if h_status != "valid" or j_status != "valid":
            dec["final_decision"] = "invalid_event"
            dec["decision_level"] = "observed_event_invalid"
        rows.append({
            "baseline_config": baseline_config,
            "branch": branch,
            "event_family": event_family,
            "event_name": event_name,
            "H_event_status": h_status,
            "Jw_event_status": j_status,
            "H_day_observed": h_day,
            "Jw_day_observed": j_day,
            "observed_delta_Jw_minus_H": float(j_day - h_day) if np.isfinite(h_day) and np.isfinite(j_day) else np.nan,
            **dec,
        })
    return pd.DataFrame(rows)


def _baseline_sensitivity_summary(state_dec: pd.DataFrame, growth_dec: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event_layer, df in (("state", state_dec), ("growth", growth_dec)):
        if df.empty:
            continue
        for (branch, event_name), sub in df.groupby(["branch", "event_name"], dropna=False):
            decisions = {str(r["baseline_config"]): str(r["final_decision"]) for _, r in sub.iterrows()}
            validities = {str(r["baseline_config"]): f"H={r['H_event_status']};Jw={r['Jw_event_status']}" for _, r in sub.iterrows()}
            c0 = decisions.get("C0_full_stage", "missing")
            c1 = decisions.get("C1_buffered_stage", "missing")
            c2 = decisions.get("C2_immediate_pre", "missing")
            non_missing = [x for x in (c0, c1, c2) if x != "missing"]
            hard = {"H_leads", "Jw_leads", "synchronous_equivalent"}
            if non_missing and len(set(non_missing)) == 1 and non_missing[0] in hard:
                sens = "stable_across_baselines"
                interp = f"{event_layer}/{branch}/{event_name} is stable as {non_missing[0]} across available baselines."
            elif c0 == c1 and c0 in hard and c2 not in {c0, "missing"}:
                sens = "sensitive_to_immediate_pre"
                interp = f"C0/C1 agree as {c0}; C2 differs or fails."
            elif c0 in hard and c1 not in {c0, "missing"}:
                sens = "sensitive_to_buffering"
                interp = "Full-stage and buffered-stage decisions differ."
            elif all(x == "invalid_event" for x in non_missing):
                sens = "invalid_across_baselines"
                interp = "Event is invalid across available baselines."
            else:
                sens = "mixed_or_unresolved"
                interp = "No stable hard decision across baselines."
            rows.append({
                "event_layer": event_layer,
                "branch": branch,
                "event_name": event_name,
                "C0_decision": c0,
                "C1_decision": c1,
                "C2_decision": c2,
                "C0_validity": validities.get("C0_full_stage", "missing"),
                "C1_validity": validities.get("C1_buffered_stage", "missing"),
                "C2_validity": validities.get("C2_immediate_pre", "missing"),
                "baseline_sensitivity": sens,
                "main_interpretation": interp,
            })
    return pd.DataFrame(rows)


def _final_summary(state_dec: pd.DataFrame, growth_dec: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def summarize(layer: str, branch: str) -> str:
        df = sensitivity[(sensitivity["event_layer"] == layer) & (sensitivity["branch"] == branch)]
        stable = df[df["baseline_sensitivity"] == "stable_across_baselines"]
        if not stable.empty:
            return "; ".join([f"{r.event_name}:{r.C0_decision}" for r in stable.itertuples()])
        inv = df[df["baseline_sensitivity"] == "invalid_across_baselines"]
        if len(inv) == len(df) and len(df) > 0:
            return "invalid_across_baselines"
        return "mixed_or_unresolved"
    rows.append({
        "question": "state_progress_order_distance_branch",
        "distance_branch_result": summarize("state", "distance"),
        "pattern_branch_result": "not_applicable",
        "baseline_sensitivity": "see_baseline_sensitivity_summary",
        "allowed_statement": "Only stable hard decisions across baselines can be written as state-progress order/synchrony.",
        "forbidden_statement": "Do not convert invalid or unresolved events into order/synchrony.",
        "next_action": "review_state_order_sync_decision",
    })
    rows.append({
        "question": "state_progress_order_pattern_branch",
        "distance_branch_result": "not_applicable",
        "pattern_branch_result": summarize("state", "pattern"),
        "baseline_sensitivity": "see_baseline_sensitivity_summary",
        "allowed_statement": "Pattern-state order is separate from distance-state order.",
        "forbidden_statement": "Do not merge pattern and distance state into one transition day.",
        "next_action": "review_state_order_sync_decision",
    })
    rows.append({
        "question": "growth_speed_order_distance_branch",
        "distance_branch_result": summarize("growth", "distance"),
        "pattern_branch_result": "not_applicable",
        "baseline_sensitivity": "see_baseline_sensitivity_summary",
        "allowed_statement": "Growth-speed order is order of dS/dt events, not state-progress order.",
        "forbidden_statement": "Do not write growth lead as conversion-state lead.",
        "next_action": "review_growth_order_sync_decision",
    })
    rows.append({
        "question": "growth_speed_order_pattern_branch",
        "distance_branch_result": "not_applicable",
        "pattern_branch_result": summarize("growth", "pattern"),
        "baseline_sensitivity": "see_baseline_sensitivity_summary",
        "allowed_statement": "Pattern-growth speed is separate from distance-growth speed.",
        "forbidden_statement": "Do not collapse pattern growth and distance growth into one growth event.",
        "next_action": "review_growth_order_sync_decision",
    })
    return pd.DataFrame(rows)


def _method_status(state_dec: pd.DataFrame, growth_dec: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    hard_state = int(state_dec["final_decision"].isin(["H_leads", "Jw_leads", "synchronous_equivalent"]).sum()) if not state_dec.empty else 0
    hard_growth = int(growth_dec["final_decision"].isin(["H_leads", "Jw_leads", "synchronous_equivalent"]).sum()) if not growth_dec.empty else 0
    stable = int((sensitivity["baseline_sensitivity"] == "stable_across_baselines").sum()) if not sensitivity.empty else 0
    branch_conflict = False
    for layer in ["state", "growth"]:
        for cfg in [c.name for c in BASELINE_CONFIGS]:
            sub = pd.concat([state_dec.assign(layer="state"), growth_dec.assign(layer="growth")], ignore_index=True)
            ss = sub[(sub["layer"] == layer) & (sub["baseline_config"] == cfg)]
            hard = ss[ss["final_decision"].isin(["H_leads", "Jw_leads", "synchronous_equivalent"])]
            if len(set(hard["final_decision"].tolist())) > 1:
                branch_conflict = True
    if stable > 0:
        overall = "usable_for_H_Jw" if not branch_conflict else "branch_conflict"
    elif hard_state or hard_growth:
        overall = "baseline_sensitive"
    else:
        overall = "method_unclosed"
    can_all = bool(overall == "usable_for_H_Jw" and stable >= 2)
    rows = [
        {"method_component": "state_framework", "status": "has_hard_decisions" if hard_state else "method_unclosed", "evidence": f"hard_state_decisions={hard_state}", "can_continue_to_all_fields": False, "reason": "H/Jw-only method trial", "next_action": "inspect_state_decisions"},
        {"method_component": "growth_framework", "status": "has_hard_decisions" if hard_growth else "method_unclosed", "evidence": f"hard_growth_decisions={hard_growth}", "can_continue_to_all_fields": False, "reason": "growth=dS/dt trial only", "next_action": "inspect_growth_decisions"},
        {"method_component": "baseline_sensitivity", "status": "stable_available" if stable else "not_stable", "evidence": f"stable_across_baselines={stable}", "can_continue_to_all_fields": can_all, "reason": "baseline sensitivity must be checked before extension", "next_action": "review_baseline_sensitivity_summary"},
        {"method_component": "overall_method_status", "status": overall, "evidence": f"hard_state={hard_state}; hard_growth={hard_growth}; stable={stable}; branch_conflict={branch_conflict}", "can_continue_to_all_fields": can_all, "reason": "Do not extend unless H/Jw state/growth decisions are stable across baseline configs.", "next_action": "extend_only_if_user_accepts_status" if can_all else "do_not_extend_to_P_V_Je"},
    ]
    return pd.DataFrame(rows)


def _plot_outputs(paths: V7VPaths, state_curves: pd.DataFrame, growth_curves: pd.DataFrame, state_dec: pd.DataFrame, growth_dec: pd.DataFrame) -> None:
    if os.environ.get("V7V_SKIP_FIGURES", "").strip() == "1":
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        _write_text(f"Figure generation skipped: matplotlib unavailable: {exc}\n", paths.figure_dir / "FIGURE_SKIPPED.txt")
        return
    _ensure_dir(paths.figure_dir)
    for cfg in [c.name for c in BASELINE_CONFIGS]:
        sub = state_curves[state_curves["baseline_config"] == cfg]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for field in FIELDS:
            f = sub[sub["field"] == field]
            ax.plot(f["day"], f["S_dist"], label=f"{field} S_dist")
            ax.plot(f["day"], f["S_pattern"], linestyle="--", label=f"{field} S_pattern")
        ax.axvspan(ACCEPTED_WINDOW[0], ACCEPTED_WINDOW[1], alpha=0.15)
        ax.set_title(f"W45 H/Jw state progress curves {cfg}")
        ax.set_xlabel("day")
        ax.set_ylabel("state progress")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(paths.figure_dir / f"w45_H_Jw_state_progress_curves_{cfg}_v7_v.png", dpi=160)
        plt.close(fig)
    for cfg in [c.name for c in BASELINE_CONFIGS]:
        sub = growth_curves[growth_curves["baseline_config"] == cfg]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for field in FIELDS:
            for branch in STATE_BRANCHES:
                f = sub[(sub["field"] == field) & (sub["branch"] == branch)]
                ax.plot(f["day"], f["V_smooth3"], label=f"{field} V_{branch}")
        ax.axhline(0.0, linewidth=0.8)
        ax.axvspan(ACCEPTED_WINDOW[0], ACCEPTED_WINDOW[1], alpha=0.15)
        ax.set_title(f"W45 H/Jw growth speed curves {cfg}")
        ax.set_xlabel("day")
        ax.set_ylabel("dS/dt")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(paths.figure_dir / f"w45_H_Jw_growth_speed_curves_{cfg}_v7_v.png", dpi=160)
        plt.close(fig)
    decision = pd.concat([state_dec.assign(layer="state"), growth_dec.assign(layer="growth")], ignore_index=True)
    if not decision.empty:
        pivot = decision.pivot_table(index=["layer", "branch", "event_name"], columns="baseline_config", values="final_decision", aggfunc="first").fillna("")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(pivot))))
        ax.axis("off")
        table = ax.table(cellText=pivot.values, rowLabels=[" | ".join(map(str, idx)) for idx in pivot.index], colLabels=pivot.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.25)
        ax.set_title("W45 H/Jw V7-v decision heatmap")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_Jw_state_growth_decision_heatmap_v7_v.png", dpi=160)
        plt.close(fig)


def _summary_markdown(meta: dict[str, Any], sensitivity: pd.DataFrame, final_summary: pd.DataFrame, method_status: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# V7-v W45 H/Jw baseline-sensitive state-growth framework")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("This branch only tests H/Jw. It separates state progress from growth speed, tests C0/C1/C2 baseline sensitivity, and treats S_dist and S_pattern as parallel state-progress branches.")
    lines.append("")
    lines.append("## 2. Baseline configurations")
    for cfg in BASELINE_CONFIGS:
        lines.append(f"- `{cfg.name}`: pre={cfg.pre}, post={cfg.post}, search={cfg.search}, diagnostic={cfg.diagnostic}. {cfg.notes}")
    lines.append("")
    lines.append("## 3. State and growth definitions")
    lines.append("- State distance branch: `S_dist = D_pre / (D_pre + D_post)`.")
    lines.append("- State pattern branch: `S_pattern = (R_diff - R0) / (R1 - R0)`.")
    lines.append("- Growth speed: first difference of each state branch, `V = dS/dt`, with centered 3-day mean for event detection.")
    lines.append("- `P_proj` is reference only and is not used for main decisions.")
    lines.append("")
    lines.append("## 4. Baseline sensitivity summary")
    if sensitivity.empty:
        lines.append("No sensitivity rows were produced.")
    else:
        for _, row in sensitivity.iterrows():
            lines.append(f"- {row['event_layer']} / {row['branch']} / {row['event_name']}: C0=`{row['C0_decision']}`, C1=`{row['C1_decision']}`, C2=`{row['C2_decision']}`, sensitivity=`{row['baseline_sensitivity']}`")
    lines.append("")
    lines.append("## 5. Final H/Jw state-growth summary")
    if final_summary.empty:
        lines.append("No final summary rows were produced.")
    else:
        for _, row in final_summary.iterrows():
            lines.append(f"- {row['question']}: distance=`{row['distance_branch_result']}`, pattern=`{row['pattern_branch_result']}`; next={row['next_action']}")
    lines.append("")
    lines.append("## 6. Method status")
    for _, row in method_status.iterrows():
        lines.append(f"- {row['method_component']}: `{row['status']}`; evidence={row['evidence']}; next={row['next_action']}")
    lines.append("")
    lines.append("## 7. Interpretation boundary")
    lines.append("Distance-state, pattern-state, distance-growth speed, and pattern-growth speed are separate result layers. A growth-speed lead cannot be written as a state-progress lead. Invalid events cannot be written as unresolved or synchronous.")
    lines.append("")
    lines.append("## 8. run_meta excerpt")
    lines.append("```json")
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2, default=str)[:4000])
    lines.append("```")
    return "\n".join(lines) + "\n"


def run_w45_H_Jw_baseline_sensitive_state_growth_v7_v(v7_root: Optional[Path] = None) -> dict[str, Any]:
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
    smoothed = _load_smoothed_fields_npz(smoothed_path)
    raw_fields = {field: _prepare_raw_field(smoothed, field, settings) for field in FIELDS}
    n_days = min(obj["n_days"] for obj in raw_fields.values())
    # Validate baseline windows before heavy computation.
    for cfg in BASELINE_CONFIGS:
        for label, period in (("pre", cfg.pre), ("post", cfg.post), ("search", cfg.search), ("diagnostic", cfg.diagnostic)):
            _period_days(period, n_days)
    baseline_config_df = pd.DataFrame([
        {
            "baseline_config": cfg.name,
            "pre_start": cfg.pre[0], "pre_end": cfg.pre[1],
            "post_start": cfg.post[0], "post_end": cfg.post[1],
            "search_start": cfg.search[0], "search_end": cfg.search[1],
            "diagnostic_start": cfg.diagnostic[0], "diagnostic_end": cfg.diagnostic[1],
            "role": cfg.role,
            "notes": cfg.notes,
        }
        for cfg in BASELINE_CONFIGS
    ])
    _write_csv(baseline_config_df, paths.output_dir / "w45_H_Jw_baseline_config_table_v7_v.csv")

    state_curve_frames: list[pd.DataFrame] = []
    growth_curve_frames: list[pd.DataFrame] = []
    observed_state_events: list[pd.DataFrame] = []
    observed_growth_events: list[pd.DataFrame] = []
    observed_negative_events: list[pd.DataFrame] = []
    input_meta: dict[str, Any] = {}
    for cfg in BASELINE_CONFIGS:
        for field, obj in raw_fields.items():
            _progress_log(paths, f"observed metrics/events: {cfg.name} {field}")
            avg_state = _safe_nanmean(obj["data"], axis=0)
            out = _compute_all_for_field_config(obj, cfg, avg_state=avg_state)
            sc = out["state_curves"].copy()
            sc.insert(0, "field", field)
            sc.insert(0, "baseline_config", cfg.name)
            sc["S_pattern_dynamic_range"] = out["meta"].get("S_pattern_dynamic_range", np.nan)
            sc["branch_validity_flag"] = out["meta"].get("S_pattern_branch_validity_flag", "unknown")
            state_curve_frames.append(sc)
            gc = out["growth_curves"].copy()
            gc.insert(0, "field", field)
            gc.insert(0, "baseline_config", cfg.name)
            growth_curve_frames.append(gc)
            se = out["state_events"].copy()
            se.insert(0, "field", field)
            se.insert(0, "baseline_config", cfg.name)
            observed_state_events.append(se)
            ge = out["growth_events"].copy()
            ge.insert(0, "field", field)
            ge.insert(0, "baseline_config", cfg.name)
            observed_growth_events.append(ge[ge["event_family"] == "growth"].copy())
            observed_negative_events.append(ge[ge["event_family"] == "negative_growth"].copy())
            input_meta[f"{cfg.name}__{field}"] = out["meta"]
    state_curves = pd.concat(state_curve_frames, ignore_index=True)
    growth_curves = pd.concat(growth_curve_frames, ignore_index=True)
    obs_state = pd.concat(observed_state_events, ignore_index=True)
    obs_growth = pd.concat(observed_growth_events, ignore_index=True)
    obs_negative = pd.concat(observed_negative_events, ignore_index=True)
    _write_csv(state_curves, paths.output_dir / "w45_H_Jw_state_progress_curves_v7_v.csv")
    _write_csv(growth_curves, paths.output_dir / "w45_H_Jw_growth_speed_curves_v7_v.csv")
    event_validity = pd.concat([obs_state, obs_growth, obs_negative], ignore_index=True)
    _write_csv(event_validity, paths.output_dir / "w45_H_Jw_event_validity_table_v7_v.csv")
    _write_csv(obs_state, paths.output_dir / "w45_H_Jw_state_observed_events_v7_v.csv")
    _write_csv(obs_growth, paths.output_dir / "w45_H_Jw_growth_observed_events_v7_v.csv")
    _write_csv(obs_negative, paths.output_dir / "w45_H_Jw_negative_growth_observed_events_v7_v.csv")

    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    _progress_log(paths, f"bootstrap events n={n_boot}")
    state_boot_frames: list[pd.DataFrame] = []
    growth_boot_frames: list[pd.DataFrame] = []
    for cidx, cfg in enumerate(BASELINE_CONFIGS):
        for fidx, field in enumerate(FIELDS):
            _progress_log(paths, f"bootstrap: {cfg.name} {field}")
            sboot, gboot = _bootstrap_field_config(raw_fields[field], cfg, n_boot=n_boot, seed=int(settings.bootstrap.random_seed) + 10000 * cidx + 1000 * fidx)
            state_boot_frames.append(sboot)
            growth_boot_frames.append(gboot[gboot["event_family"] == "growth"].copy())
    state_boot = pd.concat(state_boot_frames, ignore_index=True) if state_boot_frames else pd.DataFrame()
    growth_boot = pd.concat(growth_boot_frames, ignore_index=True) if growth_boot_frames else pd.DataFrame()
    _write_csv(state_boot, paths.output_dir / "w45_H_Jw_state_bootstrap_events_v7_v.csv")
    _write_csv(growth_boot, paths.output_dir / "w45_H_Jw_growth_bootstrap_events_v7_v.csv")

    _progress_log(paths, "registries and decisions")
    state_registry = _registry_from_long_events(obs_state, state_boot, "state")
    growth_registry = _registry_from_long_events(obs_growth, growth_boot, "growth")
    state_decision = _compare_long_events(obs_state, state_boot, "state")
    growth_decision = _compare_long_events(obs_growth, growth_boot, "growth")
    sensitivity = _baseline_sensitivity_summary(state_decision, growth_decision)
    final_summary = _final_summary(state_decision, growth_decision, sensitivity)
    method_status = _method_status(state_decision, growth_decision, sensitivity)
    _write_csv(state_registry, paths.output_dir / "w45_H_Jw_state_event_registry_v7_v.csv")
    _write_csv(growth_registry, paths.output_dir / "w45_H_Jw_growth_event_registry_v7_v.csv")
    _write_csv(state_decision, paths.output_dir / "w45_H_Jw_state_order_sync_decision_v7_v.csv")
    _write_csv(growth_decision, paths.output_dir / "w45_H_Jw_growth_order_sync_decision_v7_v.csv")
    _write_csv(sensitivity, paths.output_dir / "w45_H_Jw_baseline_sensitivity_summary_v7_v.csv")
    _write_csv(final_summary, paths.output_dir / "w45_H_Jw_state_growth_final_summary_v7_v.csv")
    _write_csv(method_status, paths.output_dir / "w45_H_Jw_state_growth_method_status_v7_v.csv")

    field_meta = {
        field: {
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
        for field, obj in raw_fields.items()
    }
    input_audit = {
        "smoothed_fields_path": str(smoothed_path),
        "available_keys": sorted(list(smoothed.keys())),
        "field_meta": field_meta,
        "accepted_window": {"window_id": WINDOW_ID, "anchor_day": ANCHOR_DAY, "accepted_window": list(ACCEPTED_WINDOW)},
        "baseline_configs": [cfg.__dict__ for cfg in BASELINE_CONFIGS],
        "input_meta_by_config_field": input_meta,
    }
    _write_json(input_audit, paths.output_dir / "input_audit_v7_v.json")
    meta = {
        "version": "v7_v",
        "hotfix_id": "v7_v_hotfix_02_unified_expanded_search",
        "output_tag": OUTPUT_TAG,
        "status": "success",
        "finished_at": _now_iso(),
        "primary_goal": "baseline-sensitive H/Jw state-progress and growth-speed adjudication",
        "fields": list(FIELDS),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "accepted_window": list(ACCEPTED_WINDOW),
        "baseline_configs": {cfg.name: {"pre": list(cfg.pre), "post": list(cfg.post), "search": list(cfg.search), "diagnostic": list(cfg.diagnostic)} for cfg in BASELINE_CONFIGS},
        "state_branches": list(STATE_BRANCHES),
        "growth_definition": "first_difference_of_state_progress_dS_dt",
        "P_proj_role": "reference_only_not_used_for_main_decision",
        "no_spatial_pairing": True,
        "no_latband_pairing": True,
        "no_P_V_Je": True,
        "event_validity_required_before_decision": True,
        "bootstrap_valid_fraction_minimum": MIN_BOOT_VALID_FRACTION,
        "n_bootstrap": n_boot,
        "output_files": sorted([p.name for p in paths.output_dir.glob("*.csv")]),
    }
    _write_json(meta, paths.output_dir / "run_meta.json")
    _write_text(_summary_markdown(meta, sensitivity, final_summary, method_status), paths.output_dir / "w45_H_Jw_baseline_sensitive_state_growth_summary_v7_v.md")
    _progress_log(paths, "figures")
    _plot_outputs(paths, state_curves, growth_curves, state_decision, growth_decision)
    _progress_log(paths, "finished success")
    return meta


__all__ = ["run_w45_H_Jw_baseline_sensitive_state_growth_v7_v"]
