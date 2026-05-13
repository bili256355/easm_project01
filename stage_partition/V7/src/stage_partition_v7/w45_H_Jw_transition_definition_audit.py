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

OUTPUT_TAG = "w45_H_Jw_transition_definition_audit_v7_t"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELDS = ("H", "Jw")
FIELD_SPECS = {
    "H": {"field_key": "z500_smoothed", "domain_attr_lon": "h_lon_range", "domain_attr_lat": "h_lat_range"},
    "Jw": {"field_key": "u200_smoothed", "domain_attr_lon": "jw_lon_range", "domain_attr_lat": "jw_lat_range"},
}
THRESHOLDS = (0.25, 0.50, 0.75)
EPS = 1.0e-12


@dataclass
class V7TPaths:
    v7_root: Path
    project_root: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path
    v7e_output_dir: Path
    v7s_output_dir: Path


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


def _resolve_paths(v7_root: Optional[Path]) -> V7TPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = v7_root.parents[1]
    return V7TPaths(
        v7_root=v7_root,
        project_root=project_root,
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7s_output_dir=v7_root / "outputs" / "w45_H_Jw_raw025_process_audit_v7_s",
    )


def _configure_settings(paths: V7TPaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    env_debug = os.environ.get("V7T_DEBUG_N_BOOTSTRAP", "").strip()
    if env_debug:
        settings.bootstrap.debug_n_bootstrap = int(env_debug)
    return settings


def _progress_log(paths: V7TPaths, message: str) -> None:
    _ensure_dir(paths.log_dir)
    log_path = paths.log_dir / "run_progress_v7_t.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{_now_iso()}] {message}\n")
    print(f"[V7-t] {message}", flush=True)


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


def _load_w45_window(paths: V7TPaths, n_days: int) -> dict[str, Any]:
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


def _first_crossing(days: np.ndarray, vals: np.ndarray, threshold: float, stable_days: int = 1) -> float:
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


def _durable_crossing(days: np.ndarray, vals: np.ndarray, threshold: float) -> float:
    # First day after which the curve no longer dips below the threshold.
    days = np.asarray(days, dtype=int)
    vals = np.asarray(vals, dtype=float)
    for i in range(vals.size):
        tail = vals[i:]
        if np.isfinite(vals[i]) and vals[i] >= threshold and np.all(np.isfinite(tail) & (tail >= threshold)):
            return float(days[i])
    return np.nan


def _stable_nonnegative_day(days: np.ndarray, vals: np.ndarray, stable_days: int = 2) -> float:
    return _first_crossing(days, vals, 0.0, stable_days=stable_days)


def _weighted_metrics_for_avg_state(
    avg_state: np.ndarray,
    weights_2d: np.ndarray,
    window: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    rows: list[dict[str, Any]] = []
    previous_x: np.ndarray | None = None
    for day in days:
        x = state[int(day), :, :]
        y = x - pre
        if np.isfinite(denom) and abs(denom) > EPS:
            p_proj = _weighted_sum(y * d, w) / denom
            parallel = p_proj * d
            residual = y - parallel
            orth_norm = _weighted_rms(residual, w)
            orth_ratio = float(orth_norm / transition_norm) if np.isfinite(orth_norm) and np.isfinite(transition_norm) and transition_norm > EPS else np.nan
        else:
            p_proj = np.nan
            orth_norm = np.nan
            orth_ratio = np.nan
        d_pre = _weighted_rms(x - pre, w)
        d_post = _weighted_rms(x - post, w)
        p_dist = float(d_pre / (d_pre + d_post)) if np.isfinite(d_pre) and np.isfinite(d_post) and (d_pre + d_post) > EPS else np.nan
        r_pre = _weighted_corr(x, pre, w)
        r_post = _weighted_corr(x, post, w)
        r_diff = float(r_post - r_pre) if np.isfinite(r_pre) and np.isfinite(r_post) else np.nan
        if previous_x is None:
            daily_cos = np.nan
            daily_change_norm = np.nan
        else:
            dx = x - previous_x
            dx_norm = _weighted_rms(dx, w)
            if np.isfinite(dx_norm) and dx_norm > EPS and np.isfinite(transition_norm) and transition_norm > EPS:
                dot = _weighted_sum(dx * d, w)
                norm_dx = np.sqrt(max(_weighted_sum(dx * dx, w), 0.0))
                norm_d = np.sqrt(max(_weighted_sum(d * d, w), 0.0))
                daily_cos = float(dot / (norm_dx * norm_d)) if norm_dx > EPS and norm_d > EPS else np.nan
            else:
                daily_cos = np.nan
            daily_change_norm = dx_norm
        rows.append({
            "day": int(day),
            "P_proj": float(p_proj) if np.isfinite(p_proj) else np.nan,
            "P_dist": float(p_dist) if np.isfinite(p_dist) else np.nan,
            "D_pre": float(d_pre) if np.isfinite(d_pre) else np.nan,
            "D_post": float(d_post) if np.isfinite(d_post) else np.nan,
            "R_pre": float(r_pre) if np.isfinite(r_pre) else np.nan,
            "R_post": float(r_post) if np.isfinite(r_post) else np.nan,
            "R_diff": float(r_diff) if np.isfinite(r_diff) else np.nan,
            "orthogonal_norm": float(orth_norm) if np.isfinite(orth_norm) else np.nan,
            "orthogonal_ratio": float(orth_ratio) if np.isfinite(orth_ratio) else np.nan,
            "daily_cos_to_post_direction": float(daily_cos) if np.isfinite(daily_cos) else np.nan,
            "daily_change_norm": float(daily_change_norm) if np.isfinite(daily_change_norm) else np.nan,
        })
        previous_x = x
    meta = {
        "transition_norm": transition_norm,
        "projection_denom": denom,
        "pre_mean": float(np.nanmean(pre)),
        "post_mean": float(np.nanmean(post)),
    }
    return pd.DataFrame(rows), meta


def _metric_events(metric_df: pd.DataFrame) -> dict[str, float]:
    days = metric_df["day"].to_numpy(dtype=int)
    out: dict[str, float] = {}
    for metric in ("P_proj", "P_dist"):
        vals = metric_df[metric].to_numpy(dtype=float)
        for thr in THRESHOLDS:
            name = f"{metric}_t{int(round(thr * 100)):02d}"
            out[name] = _first_crossing(days, vals, thr, stable_days=1)
        out[f"{metric}_durable_t50"] = _durable_crossing(days, vals, 0.50)
        out[f"{metric}_durable_t75"] = _durable_crossing(days, vals, 0.75)
    rdiff = metric_df["R_diff"].to_numpy(dtype=float)
    out["Rdiff_post_dominance_day"] = _stable_nonnegative_day(days, rdiff, stable_days=2)
    dpost = metric_df["D_post"].to_numpy(dtype=float)
    if np.isfinite(dpost).any():
        out["Dpost_closest_to_post_day"] = float(days[int(np.nanargmin(dpost))])
    else:
        out["Dpost_closest_to_post_day"] = np.nan
    od = metric_df["orthogonal_ratio"].to_numpy(dtype=float)
    if np.isfinite(od).any():
        out["Orthogonal_max_ratio_day"] = float(days[int(np.nanargmax(od))])
    else:
        out["Orthogonal_max_ratio_day"] = np.nan
    return out


def _detect_h_projection_retreat(metric_df: pd.DataFrame, window: dict[str, Any]) -> dict[str, Any]:
    df = metric_df.sort_values("day").copy()
    days = df["day"].to_numpy(dtype=int)
    vals = df["P_proj"].to_numpy(dtype=float)
    good = np.isfinite(vals)
    if good.sum() < 3:
        return {"retreat_detected": False, "retreat_start_day": np.nan, "retreat_end_day": np.nan, "retreat_drop": np.nan, "retreat_label": "insufficient_curve"}
    anchor_day = int(window["anchor_day"])
    post_start = int(window["post_period_start"])
    eligible_high = good & (days <= anchor_day)
    if not eligible_high.any():
        eligible_high = good & (days <= anchor_day + 2)
    if not eligible_high.any():
        return {"retreat_detected": False, "retreat_start_day": np.nan, "retreat_end_day": np.nan, "retreat_drop": np.nan, "retreat_label": "no_pre_anchor_high"}
    high_pos = np.where(eligible_high)[0][int(np.nanargmax(vals[eligible_high]))]
    eligible_low = good & (np.arange(vals.size) > high_pos) & (days <= post_start)
    if not eligible_low.any():
        return {"retreat_detected": False, "retreat_start_day": float(days[high_pos]), "retreat_end_day": np.nan, "retreat_drop": np.nan, "retreat_label": "no_post_high_low_candidate"}
    low_candidates = np.where(eligible_low)[0]
    low_pos = low_candidates[int(np.nanargmin(vals[low_candidates]))]
    drop = float(vals[low_pos] - vals[high_pos])
    return {
        "retreat_detected": bool(drop < 0),
        "retreat_start_day": float(days[high_pos]),
        "retreat_end_day": float(days[low_pos]),
        "retreat_start_progress": float(vals[high_pos]),
        "retreat_end_progress": float(vals[low_pos]),
        "retreat_drop": drop,
        "retreat_span_days": int(days[low_pos] - days[high_pos]),
        "retreat_label": "projection_retreat_candidate" if drop < 0 else "no_projection_retreat",
    }


def _change_between(metric_df: pd.DataFrame, col: str, start_day: int, end_day: int) -> float:
    sub = metric_df.set_index("day")
    if int(start_day) not in sub.index or int(end_day) not in sub.index:
        return np.nan
    a = float(sub.loc[int(start_day), col])
    b = float(sub.loc[int(end_day), col])
    return float(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan


def _mean_between(metric_df: pd.DataFrame, col: str, start_day: int, end_day: int) -> float:
    sub = metric_df[(metric_df["day"] >= int(start_day)) & (metric_df["day"] <= int(end_day))]
    if sub.empty:
        return np.nan
    return float(np.nanmean(sub[col].to_numpy(dtype=float)))


def _interpret_retreat(row: dict[str, Any]) -> str:
    pproj = row.get("P_proj_change", np.nan)
    pdist = row.get("P_dist_change", np.nan)
    dpre = row.get("D_pre_change", np.nan)
    dpost = row.get("D_post_change", np.nan)
    rdiff = row.get("R_diff_change", np.nan)
    orth = row.get("orthogonal_ratio_change", np.nan)
    cos_mean = row.get("daily_cos_mean", np.nan)
    if not np.isfinite(pproj) or pproj >= 0:
        return "no_projection_retreat"
    true_backward_votes = 0
    off_axis_votes = 0
    if np.isfinite(pdist) and pdist < 0:
        true_backward_votes += 1
    elif np.isfinite(pdist) and pdist >= 0:
        off_axis_votes += 1
    if np.isfinite(dpost) and dpost > 0:
        true_backward_votes += 1
    elif np.isfinite(dpost) and dpost <= 0:
        off_axis_votes += 1
    if np.isfinite(dpre) and dpre < 0:
        true_backward_votes += 1
    elif np.isfinite(dpre) and dpre >= 0:
        off_axis_votes += 1
    if np.isfinite(rdiff) and rdiff < 0:
        true_backward_votes += 1
    elif np.isfinite(rdiff) and rdiff >= 0:
        off_axis_votes += 1
    if np.isfinite(cos_mean) and cos_mean < 0:
        true_backward_votes += 1
    elif np.isfinite(cos_mean) and abs(cos_mean) <= 0.20:
        off_axis_votes += 1
    if np.isfinite(orth) and orth > 0:
        off_axis_votes += 1
    if true_backward_votes >= 4:
        return "true_backward_retreat"
    if off_axis_votes >= 4:
        return "off_axis_reorganization_or_projection_specific_retreat"
    if true_backward_votes >= 2 and off_axis_votes >= 2:
        return "mixed_retreat_and_reorganization"
    return "unresolved"


def _order_sync_decision(deltas: np.ndarray, margin_days: float = 1.0) -> dict[str, Any]:
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "delta_median": np.nan,
            "delta_q05": np.nan,
            "delta_q95": np.nan,
            "delta_q025": np.nan,
            "delta_q975": np.nan,
            "P_H_leads": np.nan,
            "P_Jw_leads": np.nan,
            "sync_equivalence_pass_1d": False,
            "final_decision": "unresolved",
        }
    q025, q975 = _q(arr, 0.025), _q(arr, 0.975)
    p_h = float(np.mean(arr > float(margin_days)))
    p_j = float(np.mean(arr < -float(margin_days)))
    sync = bool(np.isfinite(q025) and np.isfinite(q975) and q025 >= -float(margin_days) and q975 <= float(margin_days))
    if p_h >= 0.90:
        decision = "H_leads"
    elif p_j >= 0.90:
        decision = "Jw_leads"
    elif sync:
        decision = "synchronous_equivalent"
    else:
        decision = "unresolved"
    return {
        "delta_median": _q(arr, 0.50),
        "delta_q05": _q(arr, 0.05),
        "delta_q95": _q(arr, 0.95),
        "delta_q025": q025,
        "delta_q975": q975,
        "P_H_leads": p_h,
        "P_Jw_leads": p_j,
        "sync_equivalence_pass_1d": sync,
        "final_decision": decision,
    }


def _bootstrap_metric_events(field_obj: dict[str, Any], window: dict[str, Any], n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.asarray(field_obj["data"], dtype=float)
    weights = np.asarray(field_obj["weights"], dtype=float)
    rng = np.random.default_rng(int(seed))
    n_years = int(data.shape[0])
    event_rows: list[dict[str, Any]] = []
    # Also keep a small summary of retreat interpretations under bootstrap for H-like fields if needed.
    retreat_rows: list[dict[str, Any]] = []
    for b in range(int(n_boot)):
        idx = rng.integers(0, n_years, size=n_years)
        avg_state = _safe_nanmean(data[idx, :, :, :], axis=0)
        curves, _ = _weighted_metrics_for_avg_state(avg_state, weights, window)
        events = _metric_events(curves)
        events.update({"field": field_obj["field"], "bootstrap_id": int(b)})
        event_rows.append(events)
        if field_obj["field"] == "H":
            ret = _detect_h_projection_retreat(curves, window)
            if ret.get("retreat_detected", False):
                start = int(ret["retreat_start_day"])
                end = int(ret["retreat_end_day"])
                rrow = {
                    "field": "H",
                    "bootstrap_id": int(b),
                    "retreat_start_day": start,
                    "retreat_end_day": end,
                    "P_proj_change": _change_between(curves, "P_proj", start, end),
                    "P_dist_change": _change_between(curves, "P_dist", start, end),
                    "D_pre_change": _change_between(curves, "D_pre", start, end),
                    "D_post_change": _change_between(curves, "D_post", start, end),
                    "R_diff_change": _change_between(curves, "R_diff", start, end),
                    "orthogonal_ratio_change": _change_between(curves, "orthogonal_ratio", start, end),
                    "daily_cos_mean": _mean_between(curves, "daily_cos_to_post_direction", start + 1, end),
                }
                rrow["retreat_interpretation"] = _interpret_retreat(rrow)
                retreat_rows.append(rrow)
    return pd.DataFrame(event_rows), pd.DataFrame(retreat_rows)


def _compare_bootstrap_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    h = events[events["field"] == "H"].copy()
    j = events[events["field"] == "Jw"].copy()
    markers = sorted([c for c in events.columns if c not in {"field", "bootstrap_id"}])
    rows: list[dict[str, Any]] = []
    for marker in markers:
        if marker not in h.columns or marker not in j.columns:
            continue
        merged = h[["bootstrap_id", marker]].merge(j[["bootstrap_id", marker]], on="bootstrap_id", suffixes=("_H", "_Jw"))
        delta = merged[f"{marker}_Jw"].to_numpy(dtype=float) - merged[f"{marker}_H"].to_numpy(dtype=float)
        dec = _order_sync_decision(delta, margin_days=1.0)
        metric = marker.split("_t")[0] if "_t" in marker else marker.split("_")[0]
        rows.append({
            "metric_or_definition": metric,
            "event_marker": marker,
            "valid_bootstrap_pairs": int(np.isfinite(delta).sum()),
            **dec,
        })
    return pd.DataFrame(rows)


def _base_order_table(base_events: pd.DataFrame, boot_compare: pd.DataFrame) -> pd.DataFrame:
    if base_events.empty:
        return boot_compare
    rows = []
    h = base_events[base_events["field"] == "H"].iloc[0].to_dict()
    j = base_events[base_events["field"] == "Jw"].iloc[0].to_dict()
    for _, r in boot_compare.iterrows():
        marker = r["event_marker"]
        hday = h.get(marker, np.nan)
        jday = j.get(marker, np.nan)
        rows.append({
            "metric_or_definition": r["metric_or_definition"],
            "event_marker": marker,
            "H_day_observed": hday,
            "Jw_day_observed": jday,
            "observed_delta_Jw_minus_H": float(jday - hday) if np.isfinite(hday) and np.isfinite(jday) else np.nan,
            **{k: r[k] for k in boot_compare.columns if k not in {"metric_or_definition", "event_marker"}},
        })
    return pd.DataFrame(rows)


def _definition_consistency(order_table: pd.DataFrame, retreat_row: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def _decision(marker: str) -> str:
        sub = order_table[order_table["event_marker"] == marker]
        return str(sub.iloc[0]["final_decision"]) if not sub.empty else "missing"
    for question, markers in [
        ("H early lead", ["P_proj_t25", "P_dist_t25", "P_proj_t50", "P_dist_t50"]),
        ("H/Jw middle transition", ["P_proj_t50", "P_dist_t50", "Rdiff_post_dominance_day"]),
        ("completion relation", ["P_proj_t75", "P_dist_t75", "P_proj_durable_t75", "P_dist_durable_t75", "Dpost_closest_to_post_day"]),
        ("departure/initial post-likeness", ["Rdiff_post_dominance_day", "P_proj_t25", "P_dist_t25"]),
    ]:
        decisions = {m: _decision(m) for m in markers}
        non_missing = [v for v in decisions.values() if v != "missing"]
        unique = sorted(set(non_missing))
        if not non_missing:
            cls = "unresolved"
        elif len(unique) == 1 and unique[0] != "unresolved":
            cls = "robust_across_definitions"
        elif len(unique) == 1 and unique[0] == "unresolved":
            cls = "unresolved"
        elif "unresolved" in unique and len(unique) == 2:
            cls = "partly_definition_supported"
        else:
            cls = "definition_conflict"
        rows.append({
            "question": question,
            "projection_result": "; ".join([f"{m}={decisions[m]}" for m in markers if m.startswith("P_proj")]),
            "distance_result": "; ".join([f"{m}={decisions[m]}" for m in markers if m.startswith("P_dist") or m.startswith("Dpost")]),
            "pattern_result": "; ".join([f"{m}={decisions[m]}" for m in markers if m.startswith("Rdiff")]),
            "orthogonal_diagnostic": "see H_retreat_definition_adjudication_v7_t.csv",
            "daily_direction_result": "see H_retreat_definition_adjudication_v7_t.csv",
            "consistency_class": cls,
            "scientific_decision": "use_with_current_metric" if cls in {"robust_across_definitions", "partly_definition_supported"} else ("return_to_transition_definition_method" if cls == "definition_conflict" else "unresolved"),
        })
    rows.append({
        "question": "H anchor retreat",
        "projection_result": f"P_proj_change={retreat_row.get('P_proj_change', np.nan)}",
        "distance_result": f"P_dist_change={retreat_row.get('P_dist_change', np.nan)}; D_pre_change={retreat_row.get('D_pre_change', np.nan)}; D_post_change={retreat_row.get('D_post_change', np.nan)}",
        "pattern_result": f"R_diff_change={retreat_row.get('R_diff_change', np.nan)}",
        "orthogonal_diagnostic": f"orthogonal_ratio_change={retreat_row.get('orthogonal_ratio_change', np.nan)}",
        "daily_direction_result": f"daily_cos_mean={retreat_row.get('daily_cos_mean', np.nan)}",
        "consistency_class": str(retreat_row.get("retreat_interpretation", "unresolved")),
        "scientific_decision": "projection_retreat_supported_as_true_backward" if retreat_row.get("retreat_interpretation") == "true_backward_retreat" else ("projection_artifact_or_reorganization_risk" if "off_axis" in str(retreat_row.get("retreat_interpretation")) else "unresolved_or_mixed"),
    })
    return pd.DataFrame(rows)


def _write_figures(curves: pd.DataFrame, order_table: pd.DataFrame, retreat_df: pd.DataFrame, paths: V7TPaths) -> None:
    if os.environ.get("V7T_SKIP_FIGURES", "").strip() in {"1", "true", "True", "yes"}:
        _write_text("V7T_SKIP_FIGURES is set; figures skipped.\n", paths.figure_dir / "FIGURE_WARNING.txt")
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_text("matplotlib unavailable; figures skipped.\n", paths.figure_dir / "FIGURE_WARNING.txt")
        return
    _ensure_dir(paths.figure_dir)
    for metric in ["P_proj", "P_dist", "R_diff"]:
        plt.figure(figsize=(9, 5))
        for field in FIELDS:
            sub = curves[curves["field"] == field]
            if not sub.empty:
                plt.plot(sub["day"], sub[metric], marker="o", label=field)
        plt.axvline(ANCHOR_DAY, linestyle="--", linewidth=1, label="anchor_day")
        plt.xlabel("day")
        plt.ylabel(metric)
        plt.title(f"W45 H/Jw transition metric: {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.figure_dir / f"w45_H_Jw_{metric}_curves_v7_t.png", dpi=160)
        plt.close()
    # H retreat diagnostic panel.
    h = curves[curves["field"] == "H"]
    if not h.empty:
        plt.figure(figsize=(10, 6))
        for metric in ["P_proj", "P_dist", "R_diff", "orthogonal_ratio", "daily_cos_to_post_direction"]:
            plt.plot(h["day"], h[metric], marker="o", label=metric)
        plt.axvline(ANCHOR_DAY, linestyle="--", linewidth=1, label="anchor_day")
        plt.xlabel("day")
        plt.ylabel("diagnostic value")
        plt.title("W45 H retreat transition-definition diagnostics")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(paths.figure_dir / "w45_H_retreat_definition_diagnostic_v7_t.png", dpi=160)
        plt.close()
    # Decision heatmap-like table figure: textual image fallback.
    if not order_table.empty:
        pivot = order_table.pivot_table(index="event_marker", columns="metric_or_definition", values="final_decision", aggfunc="first")
        fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(pivot))))
        ax.axis("off")
        table = ax.table(cellText=pivot.fillna("").values, rowLabels=pivot.index, colLabels=pivot.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.25)
        ax.set_title("W45 H/Jw order/synchrony decisions by metric")
        plt.tight_layout()
        plt.savefig(paths.figure_dir / "w45_H_Jw_order_decision_by_metric_v7_t.png", dpi=180)
        plt.close()


def _summary_markdown(meta: dict[str, Any], retreat: pd.DataFrame, order: pd.DataFrame, consistency: pd.DataFrame, failure: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# V7-t W45 H/Jw raw-field transition-definition audit")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("This branch audits whether the previous pre→post projection progress creates artificial H retreat / H-Jw crossing or order ambiguity.")
    lines.append("It does not introduce lat-band pairing, component labels, or complex relation labels as main results.")
    lines.append("")
    lines.append("## 2. Compared transition definitions")
    lines.append("- `P_proj`: weighted pre→post projection progress.")
    lines.append("- `P_dist`: dual-distance progress using distance-to-pre and distance-to-post.")
    lines.append("- `D_pre` / `D_post`: raw weighted distances to endpoint fields.")
    lines.append("- `R_diff`: pattern correlation to post minus pattern correlation to pre.")
    lines.append("- `orthogonal_ratio`: off-axis residual relative to the pre→post transition vector.")
    lines.append("- `daily_cos_to_post_direction`: daily change direction relative to the pre→post vector.")
    lines.append("")
    lines.append("## 3. H retreat adjudication")
    if retreat.empty:
        lines.append("No H retreat adjudication row was produced.")
    else:
        r = retreat.iloc[0].to_dict()
        lines.append(f"- Interval: day {r.get('retreat_start_day')} to {r.get('retreat_end_day')}")
        lines.append(f"- Interpretation: `{r.get('retreat_interpretation')}`")
        lines.append(f"- P_proj change: {r.get('P_proj_change')}")
        lines.append(f"- P_dist change: {r.get('P_dist_change')}")
        lines.append(f"- D_post change: {r.get('D_post_change')}")
        lines.append(f"- R_diff change: {r.get('R_diff_change')}")
        lines.append(f"- Orthogonal ratio change: {r.get('orthogonal_ratio_change')}")
    lines.append("")
    lines.append("## 4. Order / synchrony by metric")
    if order.empty:
        lines.append("No order/synchrony decisions were produced.")
    else:
        counts = order["final_decision"].value_counts(dropna=False).to_dict()
        lines.append(f"Decision counts: `{counts}`")
        high_value = order[order["event_marker"].isin(["P_proj_t25", "P_proj_t50", "P_dist_t25", "P_dist_t50", "Rdiff_post_dominance_day", "Dpost_closest_to_post_day"])]
        if not high_value.empty:
            lines.append("")
            lines.append("Selected decisions:")
            for _, row in high_value.iterrows():
                lines.append(f"- {row['event_marker']}: {row['final_decision']} (observed Δ={row.get('observed_delta_Jw_minus_H')})")
    lines.append("")
    lines.append("## 5. Definition consistency")
    if not consistency.empty:
        for _, row in consistency.iterrows():
            lines.append(f"- {row['question']}: `{row['consistency_class']}` → {row['scientific_decision']}")
    lines.append("")
    lines.append("## 6. Continue / stop decision")
    if not failure.empty:
        for _, row in failure.iterrows():
            lines.append(f"- {row['criterion']}: {row['result']} → {row['if_failed_next_action']}")
    lines.append("")
    lines.append("## 7. Interpretation boundary")
    lines.append("This branch adjudicates metric sensitivity. It does not prove causality, physical mechanism, or region-to-region correspondence between H and Jw.")
    lines.append("If definitions conflict, the next step is transition-definition method redesign, not more spatial post-processing.")
    lines.append("")
    lines.append("## 8. run_meta excerpt")
    lines.append("```json")
    lines.append(json.dumps(meta, ensure_ascii=False, indent=2, default=str)[:4000])
    lines.append("```")
    return "\n".join(lines) + "\n"


def run_w45_H_Jw_transition_definition_audit_v7_t(v7_root: Optional[Path] = None) -> dict[str, Any]:
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
    curves_rows: list[pd.DataFrame] = []
    base_events_rows: list[dict[str, Any]] = []
    field_meta: dict[str, Any] = {}
    for field, obj in raw_fields.items():
        avg_state = _safe_nanmean(obj["data"], axis=0)
        curves, meta = _weighted_metrics_for_avg_state(avg_state, obj["weights"], window)
        curves.insert(0, "field", field)
        curves_rows.append(curves)
        events = _metric_events(curves)
        events.update({"field": field})
        base_events_rows.append(events)
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
    curves_all = pd.concat(curves_rows, ignore_index=True)
    base_events = pd.DataFrame(base_events_rows)
    _write_csv(curves_all, paths.output_dir / "w45_H_Jw_transition_metric_curves_v7_t.csv")
    _write_csv(base_events, paths.output_dir / "w45_H_Jw_transition_metric_observed_events_v7_t.csv")

    _progress_log(paths, "H retreat definition adjudication")
    h_curves = curves_all[curves_all["field"] == "H"].copy()
    ret = _detect_h_projection_retreat(h_curves, window)
    retreat_row: dict[str, Any] = {**ret}
    if ret.get("retreat_detected", False):
        start = int(ret["retreat_start_day"])
        end = int(ret["retreat_end_day"])
        for col in ["P_proj", "P_dist", "D_pre", "D_post", "R_diff", "orthogonal_ratio"]:
            retreat_row[f"{col}_change"] = _change_between(h_curves, col, start, end)
        retreat_row["daily_cos_mean"] = _mean_between(h_curves, "daily_cos_to_post_direction", start + 1, end)
    else:
        for col in ["P_proj", "P_dist", "D_pre", "D_post", "R_diff", "orthogonal_ratio"]:
            retreat_row[f"{col}_change"] = np.nan
        retreat_row["daily_cos_mean"] = np.nan
    retreat_row["retreat_interpretation"] = _interpret_retreat(retreat_row)
    retreat_df = pd.DataFrame([retreat_row])
    _write_csv(retreat_df, paths.output_dir / "w45_H_retreat_definition_adjudication_v7_t.csv")

    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    _progress_log(paths, f"bootstrap metric events: n={n_boot}")
    event_boots: list[pd.DataFrame] = []
    retreat_boots: list[pd.DataFrame] = []
    for idx, field in enumerate(FIELDS):
        eboot, rboot = _bootstrap_metric_events(raw_fields[field], window, n_boot=n_boot, seed=int(settings.bootstrap.random_seed) + 1000 * idx)
        event_boots.append(eboot)
        if not rboot.empty:
            retreat_boots.append(rboot)
    events_boot = pd.concat(event_boots, ignore_index=True) if event_boots else pd.DataFrame()
    retreat_boot = pd.concat(retreat_boots, ignore_index=True) if retreat_boots else pd.DataFrame()
    _write_csv(events_boot, paths.output_dir / "w45_H_Jw_transition_metric_bootstrap_events_v7_t.csv")
    if not retreat_boot.empty:
        _write_csv(retreat_boot, paths.output_dir / "w45_H_retreat_definition_bootstrap_interpretations_v7_t.csv")
    compare = _compare_bootstrap_events(events_boot)
    order_table = _base_order_table(base_events, compare)
    _write_csv(order_table, paths.output_dir / "w45_H_Jw_order_sync_by_metric_v7_t.csv")

    consistency = _definition_consistency(order_table, retreat_row)
    _write_csv(consistency, paths.output_dir / "w45_H_Jw_definition_consistency_summary_v7_t.csv")

    # Method failure / continue decision.
    conflict_count = int((consistency["consistency_class"] == "definition_conflict").sum()) if not consistency.empty else 0
    robust_count = int((consistency["consistency_class"] == "robust_across_definitions").sum()) if not consistency.empty else 0
    projection_risk = str(retreat_row.get("retreat_interpretation", "")).startswith("off_axis") or "projection" in str(retreat_row.get("retreat_interpretation", ""))
    failure_rows = [
        {
            "criterion": "definition_conflict_count",
            "result": conflict_count,
            "does_it_advance_understanding": bool(conflict_count > 0 or robust_count > 0 or retreat_row.get("retreat_interpretation") not in {"unresolved", "no_projection_retreat"}),
            "if_failed_next_action": "return_to_transition_definition_method" if conflict_count > 0 else "continue_if_other_outputs_are_informative",
        },
        {
            "criterion": "H_retreat_projection_risk",
            "result": retreat_row.get("retreat_interpretation"),
            "does_it_advance_understanding": bool(retreat_row.get("retreat_interpretation") not in {"unresolved", "no_projection_retreat"}),
            "if_failed_next_action": "do_not_use_projection_retreat_as_main_result" if projection_risk else "projection_retreat_not_rejected_by_this_audit",
        },
        {
            "criterion": "robust_across_definition_count",
            "result": robust_count,
            "does_it_advance_understanding": bool(robust_count > 0),
            "if_failed_next_action": "if_zero_and_no_other_increment_stop_this_line",
        },
    ]
    failure_df = pd.DataFrame(failure_rows)
    _write_csv(failure_df, paths.output_dir / "w45_H_Jw_method_failure_or_continue_v7_t.csv")

    input_audit = {
        "smoothed_fields_path": str(smoothed_path),
        "available_keys": sorted(list(smoothed.keys())),
        "field_meta": field_meta,
        "window": window,
        "v7s_output_dir_exists_for_comparison": paths.v7s_output_dir.exists(),
    }
    _write_json(input_audit, paths.output_dir / "input_audit_v7_t.json")

    meta: dict[str, Any] = {
        "version": "v7_t",
        "output_tag": OUTPUT_TAG,
        "status": "success",
        "created_at": _now_iso(),
        "primary_goal": "audit whether pre_to_post_projection creates artificial progress retreat or order ambiguity",
        "main_input_representation": "raw025_smoothed_field",
        "smoothed_fields_path": str(smoothed_path),
        "fields": list(FIELDS),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "window": window,
        "n_bootstrap": n_boot,
        "compared_transition_definitions": [
            "pre_to_post_projection_progress",
            "distance_to_pre_post_progress",
            "dual_distance_Dpre_Dpost",
            "pattern_correlation_to_pre_post",
            "orthogonal_residual_ratio",
            "daily_change_direction_cosine",
        ],
        "main_decision": "transition-definition sensitivity before further order/synchrony adjudication",
        "if_definition_sensitive": "return_to_transition_definition_method_design",
        "no_spatial_pairing": True,
        "no_latband_pairing": True,
        "no_complex_relation_labels": True,
        "synchrony_requires_positive_equivalence_test": True,
        "field_meta": field_meta,
        "key_outputs": [
            "w45_H_Jw_transition_metric_curves_v7_t.csv",
            "w45_H_retreat_definition_adjudication_v7_t.csv",
            "w45_H_Jw_order_sync_by_metric_v7_t.csv",
            "w45_H_Jw_definition_consistency_summary_v7_t.csv",
            "w45_H_Jw_method_failure_or_continue_v7_t.csv",
        ],
    }
    _write_json(meta, paths.output_dir / "run_meta.json")

    _progress_log(paths, "write summary and figures")
    summary = _summary_markdown(meta, retreat_df, order_table, consistency, failure_df)
    _write_text(summary, paths.output_dir / "w45_H_Jw_transition_definition_audit_summary_v7_t.md")
    _write_figures(curves_all, order_table, retreat_df, paths)
    _progress_log(paths, "finished success")
    return meta
