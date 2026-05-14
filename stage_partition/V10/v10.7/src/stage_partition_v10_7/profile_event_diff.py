from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .event_content_config import EventContentConfig, EventWindow


def _slice_days(matrix: np.ndarray, day_min: int, day_max: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    lo = max(0, int(day_min))
    hi = min(arr.shape[0] - 1, int(day_max))
    if hi < lo:
        return np.empty((0, arr.shape[1]), dtype=float)
    return arr[lo:hi + 1, :]


def _nanmean_axis0_no_warn(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.full(arr.shape[1] if arr.ndim == 2 else 0, np.nan, dtype=float)
    valid = np.isfinite(arr)
    count = valid.sum(axis=0)
    total = np.nansum(arr, axis=0)
    out = np.full(arr.shape[1], np.nan, dtype=float)
    mask = count > 0
    out[mask] = total[mask] / count[mask]
    return out


def _feature_coord(feature_table: pd.DataFrame, idx: int) -> float | None:
    if feature_table is None or feature_table.empty:
        return None
    if "feature_index" in feature_table.columns:
        rows = feature_table.loc[feature_table["feature_index"].astype(int) == int(idx)]
    else:
        rows = feature_table.iloc[[idx]] if idx < len(feature_table) else pd.DataFrame()
    if rows.empty:
        return None
    for c in ("lat_value", "feature_coord", "latitude"):
        if c in rows.columns:
            val = rows.iloc[0][c]
            try:
                return float(val)
            except Exception:
                return None
    return None


def compute_profile_event_diff(raw_matrix: np.ndarray, feature_table: pd.DataFrame, cfg: EventContentConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    arr = np.asarray(raw_matrix, dtype=float)
    for ev in cfg.event_windows:
        pre = _slice_days(arr, ev.pre_min, ev.pre_max)
        post = _slice_days(arr, ev.post_min, ev.post_max)
        if pre.size == 0 or post.size == 0:
            continue
        pre_mean = _nanmean_axis0_no_warn(pre)
        post_mean = _nanmean_axis0_no_warn(post)
        diff = post_mean - pre_mean
        absdiff = np.abs(diff)
        denom = np.nansum(absdiff)
        if not np.isfinite(denom) or denom <= 0:
            denom = np.nan
        ranks = pd.Series(absdiff).rank(method="min", ascending=False, na_option="bottom").astype(int).to_numpy()
        for j in range(arr.shape[1]):
            feature_name = f"feature_{j:03d}"
            rows.append({
                "event_id": ev.event_id,
                "target_day": int(ev.target_day),
                "pre_day_min": int(ev.pre_min),
                "pre_day_max": int(ev.pre_max),
                "post_day_min": int(ev.post_min),
                "post_day_max": int(ev.post_max),
                "feature_name": feature_name,
                "feature_index": int(j),
                "feature_coord_if_available": _feature_coord(feature_table, j),
                "pre_mean": float(pre_mean[j]) if np.isfinite(pre_mean[j]) else np.nan,
                "post_mean": float(post_mean[j]) if np.isfinite(post_mean[j]) else np.nan,
                "diff": float(diff[j]) if np.isfinite(diff[j]) else np.nan,
                "abs_diff": float(absdiff[j]) if np.isfinite(absdiff[j]) else np.nan,
                "sign": "positive" if diff[j] > 0 else ("negative" if diff[j] < 0 else "zero_or_nan"),
                "contribution_fraction": float(absdiff[j] / denom) if np.isfinite(absdiff[j]) and np.isfinite(denom) else np.nan,
                "rank_by_abs_diff": int(ranks[j]) if np.isfinite(absdiff[j]) else np.nan,
            })
    return pd.DataFrame(rows)


def pivot_event_diff_vectors(profile_diff: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if profile_diff.empty:
        return out
    for ev, g in profile_diff.groupby("event_id"):
        gg = g.sort_values("feature_index")
        out[str(ev)] = gg["diff"].to_numpy(dtype=float)
    return out
