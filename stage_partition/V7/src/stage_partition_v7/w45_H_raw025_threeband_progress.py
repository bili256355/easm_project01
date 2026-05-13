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

from .config import StagePartitionV7Settings

FIELD = "H"
FIELD_KEY = "z500_smoothed"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
OUTPUT_TAG = "w45_H_raw025_threeband_progress_v7_j"


@dataclass
class W45HRaw025ThreeBandPaths:
    v7_root: Path
    project_root: Path
    v7e_output_dir: Path
    v7f_output_dir: Path
    v7i_output_dir: Path
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


def _resolve_paths(v7_root: Optional[Path]) -> W45HRaw025ThreeBandPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = v7_root.parents[1]
    return W45HRaw025ThreeBandPaths(
        v7_root=v7_root,
        project_root=project_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7f_output_dir=v7_root / "outputs" / "w45_H_feature_progress_v7_f",
        v7i_output_dir=v7_root / "outputs" / "w45_H_latbin_profile_progress_v7_i",
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _configure_settings(paths: W45HRaw025ThreeBandPaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    return settings


def _safe_nanmean(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(float(lower), float(upper)), max(float(lower), float(upper))
    return (arr >= lo) & (arr <= hi)


def _weighted_mean(vals: np.ndarray, weights: np.ndarray, axis: int = -1):
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    # Broadcast weights to vals on the selected axis.
    w_shape = [1] * vals.ndim
    w_shape[axis] = weights.size
    w = weights.reshape(w_shape)
    good = np.isfinite(vals) & np.isfinite(w)
    num = np.nansum(np.where(good, vals * w, 0.0), axis=axis)
    den = np.nansum(np.where(good, w, 0.0), axis=axis)
    out = np.full_like(num, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _weighted_dot(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
    if int(good.sum()) == 0:
        return np.nan
    den = np.nansum(w[good])
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return float(np.nansum(a[good] * b[good] * w[good]) / den)


def _weighted_rms(x: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    v = _weighted_dot(x, x, weights)
    if not np.isfinite(v):
        return np.nan
    return float(np.sqrt(max(v, 0.0)))


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


def _count_crossings(vals: np.ndarray, threshold: float) -> int:
    vals = np.asarray(vals, dtype=float)
    good = np.isfinite(vals)
    if int(good.sum()) < 2:
        return 0
    above = vals[good] >= float(threshold)
    return int(np.sum(above[1:] != above[:-1]))


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


def _load_w45_window(paths: W45HRaw025ThreeBandPaths, n_days: int) -> dict:
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


def _prepare_raw025_h_lonmean(smoothed: dict, settings: StagePartitionV7Settings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H lon-mean on raw 0.25-ish latitude points.

    Output:
      raw_profile: years x days x raw_lat_points, sorted by ascending latitude.
      raw_lats: ascending latitude values.
      raw_lat_indices_original: original indices into the smoothed field latitude axis.
    """
    cfg = settings.profile
    lat = np.asarray(smoothed["lat"], dtype=float)
    lon = np.asarray(smoothed["lon"], dtype=float)
    field = np.asarray(smoothed[FIELD_KEY], dtype=float)
    if field.ndim != 4:
        raise ValueError(f"Expected {FIELD_KEY} shape years x days x lat x lon; got {field.shape}")
    lat_mask = _mask_between(lat, *cfg.h_lat_range)
    lon_mask = _mask_between(lon, *cfg.h_lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"No H raw latitude points in range {cfg.h_lat_range}")
    if not np.any(lon_mask):
        raise ValueError(f"No H longitude points in range {cfg.h_lon_range}")
    lat_idx = np.where(lat_mask)[0]
    raw_lats_unsorted = lat[lat_idx]
    order = np.argsort(raw_lats_unsorted)  # user data may be high-to-low; force low-to-high for partitioning.
    raw_lats = raw_lats_unsorted[order]
    raw_idx_sorted = lat_idx[order]
    subset = field[:, :, raw_idx_sorted, :][:, :, :, lon_mask]
    lon_mean = _safe_nanmean(subset, axis=-1)
    return np.asarray(lon_mean, dtype=float), raw_lats, raw_idx_sorted


def _build_equal_count_three_regions(raw_lats: np.ndarray, raw_lat_indices: np.ndarray) -> pd.DataFrame:
    raw_lats = np.asarray(raw_lats, dtype=float)
    raw_lat_indices = np.asarray(raw_lat_indices, dtype=int)
    chunks = np.array_split(np.arange(raw_lats.size), 3)
    labels = ["R1_low", "R2_mid", "R3_high"]
    rows: list[dict] = []
    for ridx, inds in enumerate(chunks):
        vals = raw_lats[inds]
        rows.append({
            "region_id": ridx,
            "region_label": labels[ridx],
            "lat_min": float(np.nanmin(vals)),
            "lat_max": float(np.nanmax(vals)),
            "lat_center": float(np.nanmean(vals)),
            "n_raw_lat_points": int(len(inds)),
            "raw_lat_indices_sorted": ";".join(str(int(i)) for i in inds),
            "raw_lat_indices_original": ";".join(str(int(raw_lat_indices[i])) for i in inds),
            "raw_lat_values": ";".join(f"{x:.6g}" for x in vals),
            "construction_method": "raw025_equal_count_lat_partition_after_lon_mean",
            "uses_2deg_interp": False,
            "uses_2deg_latbin": False,
        })
    return pd.DataFrame(rows)


def _progress_for_region(avg_state: np.ndarray, lats: np.ndarray, region_row: dict, window: dict, settings: StagePartitionV7Settings) -> tuple[dict, pd.DataFrame]:
    cfg = settings.progress_timing
    inds = [int(x) for x in str(region_row["raw_lat_indices_sorted"]).split(";") if str(x).strip()]
    days = np.arange(int(window["analysis_window_start"]), int(window["analysis_window_end"]) + 1, dtype=int)
    pre_days = np.arange(int(window["pre_period_start"]), int(window["pre_period_end"]) + 1, dtype=int)
    post_days = np.arange(int(window["post_period_start"]), int(window["post_period_end"]) + 1, dtype=int)
    region_lats = np.asarray(lats[inds], dtype=float)
    weights = np.cos(np.deg2rad(region_lats))
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    state = np.asarray(avg_state[:, inds], dtype=float)  # days x region_lats
    n_days = state.shape[0]
    if np.any(days < 0) or np.any(days >= n_days):
        raise ValueError(f"Analysis days out of state bounds: {days[0]}-{days[-1]} n_days={n_days}")
    pre = _safe_nanmean(state[pre_days, :], axis=0)
    post = _safe_nanmean(state[post_days, :], axis=0)
    tv = post - pre
    denom = _weighted_dot(tv, tv, weights)
    raw_vals = []
    clipped_vals = []
    for d in days:
        if not np.isfinite(denom) or abs(denom) < float(cfg.min_transition_norm):
            raw = np.nan
        else:
            raw = _weighted_dot(state[d, :] - pre, tv, weights) / denom
        raw_vals.append(raw)
        if np.isfinite(raw):
            clipped_vals.append(float(np.clip(raw, float(cfg.progress_clip_min), float(cfg.progress_clip_max))))
        else:
            clipped_vals.append(np.nan)
    raw_vals = np.asarray(raw_vals, dtype=float)
    progress_vals = np.asarray(clipped_vals, dtype=float)
    onset = _first_stable_crossing(days, progress_vals, float(cfg.threshold_onset), int(cfg.stable_crossing_days))
    midpoint = _first_stable_crossing(days, progress_vals, float(cfg.threshold_midpoint), int(cfg.stable_crossing_days))
    finish = _first_stable_crossing(days, progress_vals, float(cfg.threshold_finish), int(cfg.stable_crossing_days))
    duration = float(finish - onset + 1) if np.isfinite(onset) and np.isfinite(finish) else np.nan
    c25 = _count_crossings(progress_vals, float(cfg.threshold_onset))
    c50 = _count_crossings(progress_vals, float(cfg.threshold_midpoint))
    c75 = _count_crossings(progress_vals, float(cfg.threshold_finish))
    corr = np.nan
    valid = np.isfinite(progress_vals)
    if int(valid.sum()) >= 3 and np.nanstd(progress_vals[valid]) > 1e-12:
        corr = float(np.corrcoef(days[valid], progress_vals[valid])[0, 1])
    # separation
    pre_var_days = [_weighted_rms(state[d, :] - pre, weights) for d in pre_days if 0 <= d < n_days]
    post_var_days = [_weighted_rms(state[d, :] - post, weights) for d in post_days if 0 <= d < n_days]
    pre_var = float(np.nanmean(pre_var_days)) if len(pre_var_days) else np.nan
    post_var = float(np.nanmean(post_var_days)) if len(post_var_days) else np.nan
    dist = _weighted_rms(tv, weights)
    within = pre_var + post_var
    sep_ratio = float(dist / within) if np.isfinite(dist) and np.isfinite(within) and within > 0 else np.nan
    sep_label = _separation_label(sep_ratio, settings)
    excess_crossings = max(c25 - 1, 0) + max(c50 - 1, 0) + max(c75 - 1, 0)
    low_corr = np.isfinite(corr) and corr < float(cfg.monotonic_corr_threshold)
    nonmono = bool(low_corr or excess_crossings >= 1)
    if sep_label == "no_clear_separation":
        quality = "no_clear_prepost_separation"
    elif not np.isfinite(midpoint):
        quality = "partial_progress"
    elif bool(any(np.isfinite(z) and z <= days[0] + int(cfg.boundary_margin_days) for z in [onset, midpoint])) or bool(any(np.isfinite(z) and z >= days[-1] - int(cfg.boundary_margin_days) for z in [midpoint, finish])):
        quality = "boundary_limited_progress"
    elif nonmono:
        quality = "nonmonotonic_progress"
    elif sep_label in {"clear_separation", "moderate_separation"} and np.isfinite(onset) and np.isfinite(finish):
        # This keeps the label simple; broadness is diagnosed by duration/CI rather than by a hidden day threshold.
        quality = "monotonic_clear_progress"
    else:
        quality = "monotonic_broad_progress"
    base = {
        "window_id": str(window.get("window_id", WINDOW_ID)),
        "anchor_day": int(window.get("anchor_day", ANCHOR_DAY)),
        "region_id": int(region_row["region_id"]),
        "region_label": str(region_row["region_label"]),
        "lat_min": float(region_row["lat_min"]),
        "lat_max": float(region_row["lat_max"]),
        "lat_center": float(region_row["lat_center"]),
        "n_raw_lat_points": int(region_row["n_raw_lat_points"]),
        "uses_2deg_interp": False,
        "construction_method": str(region_row["construction_method"]),
        "analysis_window_start": int(window["analysis_window_start"]),
        "analysis_window_end": int(window["analysis_window_end"]),
        "pre_period_start": int(window["pre_period_start"]),
        "pre_period_end": int(window["pre_period_end"]),
        "post_period_start": int(window["post_period_start"]),
        "post_period_end": int(window["post_period_end"]),
    }
    row = {
        **base,
        "observed_onset_day": onset,
        "observed_midpoint_day": midpoint,
        "observed_finish_day": finish,
        "observed_duration": duration,
        "pre_post_distance": dist,
        "within_pre_variability": pre_var,
        "within_post_variability": post_var,
        "separation_ratio": sep_ratio,
        "pre_post_separation_label": sep_label,
        "progress_monotonicity_corr": corr,
        "n_crossings_025": c25,
        "n_crossings_050": c50,
        "n_crossings_075": c75,
        "progress_quality_label": quality,
    }
    curves = pd.DataFrame([
        {**base, "day": int(d), "raw_progress": float(raw) if np.isfinite(raw) else np.nan, "progress": float(p) if np.isfinite(p) else np.nan}
        for d, raw, p in zip(days, raw_vals, progress_vals)
    ])
    return row, curves


def _load_or_make_bootstrap_indices(n_years: int, paths: W45HRaw025ThreeBandPaths, settings: StagePartitionV7Settings) -> list[np.ndarray]:
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


def _summarize_region_bootstrap(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if samples.empty:
        return pd.DataFrame()
    group_cols = ["region_id", "region_label", "lat_min", "lat_max", "lat_center", "n_raw_lat_points"]
    for keys, sub in samples.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {c: v for c, v in zip(group_cols, keys)}
        row = {**base, "n_samples": int(len(sub))}
        for col, outprefix in [("onset_day", "onset"), ("midpoint_day", "midpoint"), ("finish_day", "finish"), ("duration", "duration")]:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"n_valid_{outprefix}"] = int(vals.size)
            row[f"valid_{outprefix}_fraction"] = float(vals.size / len(sub)) if len(sub) else np.nan
            if vals.size:
                row[f"{outprefix}_median"] = float(np.nanmedian(vals))
                row[f"{outprefix}_q05"] = float(np.nanpercentile(vals, 5))
                row[f"{outprefix}_q95"] = float(np.nanpercentile(vals, 95))
                row[f"{outprefix}_q025"] = float(np.nanpercentile(vals, 2.5))
                row[f"{outprefix}_q975"] = float(np.nanpercentile(vals, 97.5))
                row[f"{outprefix}_q90_width"] = float(np.nanpercentile(vals, 95) - np.nanpercentile(vals, 5))
                row[f"{outprefix}_q95_width"] = float(np.nanpercentile(vals, 97.5) - np.nanpercentile(vals, 2.5))
                row[f"{outprefix}_iqr"] = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
            else:
                row[f"{outprefix}_median"] = np.nan
                row[f"{outprefix}_q05"] = np.nan
                row[f"{outprefix}_q95"] = np.nan
                row[f"{outprefix}_q90_width"] = np.nan
        qcounts = sub["progress_quality_label"].astype(str).value_counts().to_dict()
        scounts = sub["pre_post_separation_label"].astype(str).value_counts().to_dict()
        row["dominant_progress_quality_label"] = max(qcounts, key=qcounts.get) if qcounts else "none"
        row["dominant_prepost_separation_label"] = max(scounts, key=scounts.get) if scounts else "none"
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_delta_test(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    regions = samples[["region_id", "region_label"]].drop_duplicates().sort_values("region_id")
    reg_ids = [int(x) for x in regions["region_id"].tolist()]
    labels = dict(zip(regions["region_id"], regions["region_label"]))
    piv = samples.pivot_table(index="bootstrap_id", columns="region_id", values="midpoint_day", aggfunc="first")
    for i in range(len(reg_ids)):
        for j in range(i + 1, len(reg_ids)):
            a, b = reg_ids[i], reg_ids[j]
            if a not in piv.columns or b not in piv.columns:
                continue
            delta = pd.to_numeric(piv[b] - piv[a], errors="coerce").dropna().to_numpy(dtype=float)
            if delta.size == 0:
                stats = {"median_delta_b_minus_a": np.nan, "q05_delta": np.nan, "q95_delta": np.nan, "q025_delta": np.nan, "q975_delta": np.nan, "prob_delta_gt_0": np.nan, "prob_delta_lt_0": np.nan, "pass_90": False, "pass_95": False}
            else:
                q05, q95 = float(np.nanpercentile(delta, 5)), float(np.nanpercentile(delta, 95))
                q025, q975 = float(np.nanpercentile(delta, 2.5)), float(np.nanpercentile(delta, 97.5))
                stats = {
                    "median_delta_b_minus_a": float(np.nanmedian(delta)),
                    "mean_delta_b_minus_a": float(np.nanmean(delta)),
                    "q05_delta": q05,
                    "q95_delta": q95,
                    "q025_delta": q025,
                    "q975_delta": q975,
                    "prob_delta_gt_0": float(np.mean(delta > 0)),
                    "prob_delta_lt_0": float(np.mean(delta < 0)),
                    "prob_delta_eq_0": float(np.mean(delta == 0)),
                    "pass_90": bool(q05 > 0 or q95 < 0),
                    "pass_95": bool(q025 > 0 or q975 < 0),
                }
            med = stats.get("median_delta_b_minus_a", np.nan)
            if np.isfinite(med) and med > 0:
                early, late = labels[a], labels[b]
            elif np.isfinite(med) and med < 0:
                early, late = labels[b], labels[a]
            else:
                early, late = "", ""
            rows.append({
                "region_a": labels[a],
                "region_b": labels[b],
                "region_a_id": a,
                "region_b_id": b,
                **stats,
                "field_early_candidate": early,
                "field_late_candidate": late,
                "test_note": "diagnostic pairwise midpoint delta among three raw025 equal-count H regions; no minimum effective day threshold",
            })
    return pd.DataFrame(rows)


def _load_existing_unit_tables(paths: W45HRaw025ThreeBandPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    v7f = _read_csv(paths.v7f_output_dir / "w45_H_feature_progress_bootstrap_summary_v7_f.csv", required=False)
    v7i = _read_csv(paths.v7i_output_dir / "w45_H_latbin_progress_bootstrap_summary_v7_i.csv", required=False)
    return v7f, v7i


def _make_resolution_unit_comparison(raw_summary: pd.DataFrame, paths: W45HRaw025ThreeBandPaths) -> pd.DataFrame:
    rows: list[dict] = []
    v7f, v7i = _load_existing_unit_tables(paths)
    if not v7f.empty:
        for _, r in v7f.iterrows():
            rows.append({
                "unit_type": "V7_f_single_2deg_interp_feature",
                "unit_label": str(r.get("lat", r.get("feature_lat", r.get("feature_label", "feature")))),
                "n_units_or_points": 1,
                "lat_min": r.get("lat", r.get("feature_lat", np.nan)),
                "lat_max": r.get("lat", r.get("feature_lat", np.nan)),
                "midpoint_median": r.get("midpoint_median", np.nan),
                "midpoint_q05": r.get("midpoint_q05", np.nan),
                "midpoint_q95": r.get("midpoint_q95", np.nan),
                "midpoint_q90_width": r.get("midpoint_q90_width", np.nan),
                "dominant_progress_quality_label": r.get("dominant_progress_quality_label", r.get("progress_quality_label", "")),
                "interpretation": "existing V7-f single 2-degree interpolation-feature diagnostic; not raw025 region",
            })
    if not v7i.empty:
        for _, r in v7i.iterrows():
            rows.append({
                "unit_type": "V7_i_single_2deg_latbin_feature",
                "unit_label": str(r.get("feature_lat_center", r.get("lat", r.get("feature_label", "latbin")))),
                "n_units_or_points": int(r.get("n_raw_lat_points", 1)) if pd.notna(r.get("n_raw_lat_points", np.nan)) else 1,
                "lat_min": r.get("raw_lat_min", r.get("lat", np.nan)),
                "lat_max": r.get("raw_lat_max", r.get("lat", np.nan)),
                "midpoint_median": r.get("midpoint_median", np.nan),
                "midpoint_q05": r.get("midpoint_q05", np.nan),
                "midpoint_q95": r.get("midpoint_q95", np.nan),
                "midpoint_q90_width": r.get("midpoint_q90_width", np.nan),
                "dominant_progress_quality_label": r.get("dominant_progress_quality_label", r.get("progress_quality_label", "")),
                "interpretation": "existing V7-i single 2-degree lat-bin mean feature diagnostic; not three-region result",
            })
    for _, r in raw_summary.iterrows():
        rows.append({
            "unit_type": "V7_j_raw025_threeband_region",
            "unit_label": str(r.get("region_label")),
            "n_units_or_points": int(r.get("n_raw_lat_points", 0)),
            "lat_min": r.get("lat_min", np.nan),
            "lat_max": r.get("lat_max", np.nan),
            "midpoint_median": r.get("midpoint_median", np.nan),
            "midpoint_q05": r.get("midpoint_q05", np.nan),
            "midpoint_q95": r.get("midpoint_q95", np.nan),
            "midpoint_q90_width": r.get("midpoint_q90_width", np.nan),
            "dominant_progress_quality_label": r.get("dominant_progress_quality_label", ""),
            "interpretation": "new raw025 equal-count latitude region-vector progress; does not use 2-degree interpolation",
        })
    return pd.DataFrame(rows)


def _build_upstream_implication(raw_summary: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    raw_width = pd.to_numeric(raw_summary.get("midpoint_q90_width"), errors="coerce")
    raw_nonmono = raw_summary.get("dominant_progress_quality_label", pd.Series(dtype=str)).astype(str).str.contains("nonmonotonic", na=False)
    raw_med_width = float(np.nanmedian(raw_width)) if raw_width.notna().any() else np.nan
    f_width = pd.to_numeric(comparison.loc[comparison["unit_type"] == "V7_f_single_2deg_interp_feature", "midpoint_q90_width"], errors="coerce")
    i_width = pd.to_numeric(comparison.loc[comparison["unit_type"] == "V7_i_single_2deg_latbin_feature", "midpoint_q90_width"], errors="coerce")
    f_med = float(np.nanmedian(f_width)) if f_width.notna().any() else np.nan
    i_med = float(np.nanmedian(i_width)) if i_width.notna().any() else np.nan
    improves_vs_f = bool(np.isfinite(raw_med_width) and np.isfinite(f_med) and raw_med_width < f_med)
    improves_vs_i = bool(np.isfinite(raw_med_width) and np.isfinite(i_med) and raw_med_width < i_med)
    n_nonmono = int(raw_nonmono.sum()) if len(raw_nonmono) else 0
    rows.append({
        "diagnostic_item": "raw025_threeband_improves_midpoint_width",
        "status": "yes" if improves_vs_f and improves_vs_i else ("partial" if improves_vs_f or improves_vs_i else "no"),
        "evidence": f"median_q90_width_raw025_threeband={raw_med_width}; v7f_interp={f_med}; v7i_latbin={i_med}",
        "upstream_implication": "If yes, region progress should be built from raw025 regions rather than post-hoc 2deg features. If no, W45-H instability is not mainly fixed by fair raw025 three-region aggregation.",
        "recommended_followup": "Use this as a diagnosis before modifying broader region-level progress implementation.",
    })
    rows.append({
        "diagnostic_item": "raw025_threeband_progress_quality",
        "status": "limited" if n_nonmono > 0 else "usable",
        "evidence": f"n_nonmonotonic_regions={n_nonmono} of {len(raw_summary)}",
        "upstream_implication": "If nonmonotonic remains common, instability may be driven by W45-H transition shape/window/timing marker rather than 2deg feature construction.",
        "recommended_followup": "If limited, inspect onset/finish/window coverage before further spatial aggregation.",
    })
    return pd.DataFrame(rows)


def _plot_curves(curves: pd.DataFrame, fig_path: Path, window: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _ensure_dir(fig_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, sub in curves.groupby("region_label", sort=False):
        ax.plot(sub["day"], sub["progress"], marker="o", linewidth=1.5, label=str(label))
    ax.axvline(int(window["anchor_day"]), linestyle="--", linewidth=1)
    ax.axvspan(int(window["pre_period_start"]), int(window["pre_period_end"]), alpha=0.12, label="pre")
    ax.axvspan(int(window["post_period_start"]), int(window["post_period_end"]), alpha=0.12, label="post")
    ax.set_title("W45 H raw025 equal-count three-region progress")
    ax.set_xlabel("day index")
    ax.set_ylabel("clipped progress")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _plot_midpoint(summary: pd.DataFrame, fig_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _ensure_dir(fig_path.parent)
    df = summary.sort_values("lat_center")
    x = np.arange(len(df))
    med = pd.to_numeric(df["midpoint_median"], errors="coerce").to_numpy(dtype=float)
    q05 = pd.to_numeric(df["midpoint_q05"], errors="coerce").to_numpy(dtype=float)
    q95 = pd.to_numeric(df["midpoint_q95"], errors="coerce").to_numpy(dtype=float)
    yerr = np.vstack([med - q05, q95 - med])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, med, yerr=yerr, fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["region_label"].astype(str).tolist())
    ax.set_ylabel("midpoint day")
    ax.set_title("W45 H raw025 three-region midpoint bootstrap interval")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _write_summary_md(paths: W45HRaw025ThreeBandPaths, region_def: pd.DataFrame, observed: pd.DataFrame, summary: pd.DataFrame, pairwise: pd.DataFrame, comparison: pd.DataFrame, implication: pd.DataFrame, window: dict) -> None:
    lines = []
    lines.append(f"# W45 H raw025 three-band progress V7-j\n")
    lines.append(f"Created: {_now_iso()}\n")
    lines.append("## Purpose\n")
    lines.append("This audit avoids the current 2-degree interpolation profile and builds three equal-count latitude regions directly from raw latitude points in the H/z500 field. It checks whether W45-H stability improves when the analysis unit is a fair raw-resolution region-vector rather than a post-hoc 2-degree feature.\n")
    lines.append("## Window and construction\n")
    lines.append(f"- Window: {window.get('window_id')} / anchor={window.get('anchor_day')}\n")
    lines.append(f"- Analysis: {window.get('analysis_window_start')}–{window.get('analysis_window_end')}\n")
    lines.append(f"- Pre: {window.get('pre_period_start')}–{window.get('pre_period_end')}\n")
    lines.append(f"- Post: {window.get('post_period_start')}–{window.get('post_period_end')}\n")
    lines.append("- Uses 2-degree interpolation: **False**\n")
    lines.append("- Region construction: raw025_equal_count_lat_partition_after_lon_mean\n")
    lines.append("\n## Region definitions\n")
    for _, r in region_def.iterrows():
        lines.append(f"- {r['region_label']}: {r['lat_min']}–{r['lat_max']}°N, n_raw_lat_points={r['n_raw_lat_points']}\n")
    lines.append("\n## Bootstrap summary\n")
    for _, r in summary.iterrows():
        lines.append(f"- {r['region_label']}: midpoint median={r.get('midpoint_median')}, q05–q95={r.get('midpoint_q05')}–{r.get('midpoint_q95')}, q90_width={r.get('midpoint_q90_width')}, quality={r.get('dominant_progress_quality_label')}\n")
    lines.append("\n## Pairwise diagnostic\n")
    if pairwise.empty:
        lines.append("No pairwise diagnostic rows generated.\n")
    else:
        for _, r in pairwise.iterrows():
            lines.append(f"- {r['region_a']} vs {r['region_b']}: median Δ(B-A)={r.get('median_delta_b_minus_a')}, q05–q95={r.get('q05_delta')}–{r.get('q95_delta')}, pass90={r.get('pass_90')}, pass95={r.get('pass_95')}\n")
    lines.append("\n## Upstream implication\n")
    for _, r in implication.iterrows():
        lines.append(f"- {r['diagnostic_item']}: {r['status']} — {r['evidence']}\n")
    lines.append("\n## Interpretation guardrails\n")
    lines.append("- This audit does not use the 2-degree interpolation profile.\n")
    lines.append("- This audit does not infer causality or pathway.\n")
    lines.append("- Passing or failing here diagnoses W45-H implementation, not other windows or fields.\n")
    lines.append("- If raw025 three-region results remain unstable, the problem is unlikely to be solved by simply replacing interpolation with raw-lat fair regional averaging.\n")
    (paths.output_dir / "w45_H_raw025_threeband_progress_summary_v7_j.md").write_text("".join(lines), encoding="utf-8")


def run_w45_H_raw025_threeband_progress_v7_j(v7_root: Optional[Path] = None) -> None:
    paths = _resolve_paths(v7_root)
    settings = _configure_settings(paths)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)

    smoothed_path = settings.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    raw_profile, raw_lats, raw_lat_indices = _prepare_raw025_h_lonmean(smoothed, settings)
    n_years, n_days, _ = raw_profile.shape
    window = _load_w45_window(paths, n_days)

    input_audit = {
        "created_at": _now_iso(),
        "status": "success",
        "smoothed_fields_path": str(smoothed_path),
        "smoothed_fields_exists": bool(smoothed_path.exists()),
        "raw_profile_shape": list(raw_profile.shape),
        "raw_lat_min": float(np.nanmin(raw_lats)),
        "raw_lat_max": float(np.nanmax(raw_lats)),
        "n_raw_lat_points": int(raw_lats.size),
        "uses_2deg_interp": False,
        "window": window,
        "v7e_output_dir_exists": bool(paths.v7e_output_dir.exists()),
        "v7f_output_dir_exists": bool(paths.v7f_output_dir.exists()),
        "v7i_output_dir_exists": bool(paths.v7i_output_dir.exists()),
    }
    _write_json(input_audit, paths.output_dir / "input_audit_v7_j.json")

    region_def = _build_equal_count_three_regions(raw_lats, raw_lat_indices)
    region_def["lon_min"] = float(min(settings.profile.h_lon_range))
    region_def["lon_max"] = float(max(settings.profile.h_lon_range))
    _write_csv(region_def, paths.output_dir / "w45_H_raw025_threeband_region_definition_v7_j.csv")

    observed_avg = _safe_nanmean(raw_profile, axis=0)  # days x raw_lats
    obs_rows: list[dict] = []
    curve_frames: list[pd.DataFrame] = []
    for _, reg in region_def.iterrows():
        row, curves = _progress_for_region(observed_avg, raw_lats, reg.to_dict(), window, settings)
        obs_rows.append(row)
        curve_frames.append(curves)
    observed_df = pd.DataFrame(obs_rows)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    _write_csv(observed_df, paths.output_dir / "w45_H_raw025_threeband_progress_observed_v7_j.csv")
    _write_csv(curves_df, paths.output_dir / "w45_H_raw025_threeband_progress_curves_long_v7_j.csv")

    resamples = _load_or_make_bootstrap_indices(n_years, paths, settings)
    sample_rows: list[dict] = []
    for bid, idx in enumerate(resamples):
        sample_avg = _safe_nanmean(raw_profile[idx, :, :], axis=0)
        for _, reg in region_def.iterrows():
            row, _ = _progress_for_region(sample_avg, raw_lats, reg.to_dict(), window, settings)
            sample_rows.append({
                "bootstrap_id": int(bid),
                "region_id": row["region_id"],
                "region_label": row["region_label"],
                "lat_min": row["lat_min"],
                "lat_max": row["lat_max"],
                "lat_center": row["lat_center"],
                "n_raw_lat_points": row["n_raw_lat_points"],
                "onset_day": row["observed_onset_day"],
                "midpoint_day": row["observed_midpoint_day"],
                "finish_day": row["observed_finish_day"],
                "duration": row["observed_duration"],
                "pre_post_separation_label": row["pre_post_separation_label"],
                "progress_quality_label": row["progress_quality_label"],
            })
    samples_df = pd.DataFrame(sample_rows)
    _write_csv(samples_df, paths.output_dir / "w45_H_raw025_threeband_progress_bootstrap_samples_v7_j.csv")

    summary_df = _summarize_region_bootstrap(samples_df)
    _write_csv(summary_df, paths.output_dir / "w45_H_raw025_threeband_progress_bootstrap_summary_v7_j.csv")

    pairwise_df = _pairwise_delta_test(samples_df)
    _write_csv(pairwise_df, paths.output_dir / "w45_H_raw025_threeband_pairwise_delta_test_v7_j.csv")

    comparison_df = _make_resolution_unit_comparison(summary_df, paths)
    _write_csv(comparison_df, paths.output_dir / "w45_H_resolution_unit_comparison_v7_j.csv")

    implication_df = _build_upstream_implication(summary_df, comparison_df)
    _write_csv(implication_df, paths.output_dir / "w45_H_raw025_threeband_upstream_implication_v7_j.csv")

    _plot_curves(curves_df, paths.figure_dir / "w45_H_raw025_threeband_progress_curves_v7_j.png", window)
    _plot_midpoint(summary_df, paths.figure_dir / "w45_H_raw025_threeband_midpoint_bootstrap_v7_j.png")

    _write_summary_md(paths, region_def, observed_df, summary_df, pairwise_df, comparison_df, implication_df, window)

    run_meta = {
        "status": "success",
        "created_at": _now_iso(),
        "output_tag": OUTPUT_TAG,
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "does_not_use_2deg_interpolation": True,
        "construction_method": "raw025_equal_count_lat_partition_after_lon_mean",
        "n_years": int(n_years),
        "n_days": int(n_days),
        "n_raw_lat_points": int(raw_lats.size),
        "n_regions": int(len(region_def)),
        "n_bootstrap": int(len(resamples)),
        "input_smoothed_fields_path": str(smoothed_path),
        "outputs": {
            "region_definition": "w45_H_raw025_threeband_region_definition_v7_j.csv",
            "observed": "w45_H_raw025_threeband_progress_observed_v7_j.csv",
            "bootstrap_samples": "w45_H_raw025_threeband_progress_bootstrap_samples_v7_j.csv",
            "bootstrap_summary": "w45_H_raw025_threeband_progress_bootstrap_summary_v7_j.csv",
            "pairwise_delta": "w45_H_raw025_threeband_pairwise_delta_test_v7_j.csv",
            "comparison": "w45_H_resolution_unit_comparison_v7_j.csv",
            "summary_md": "w45_H_raw025_threeband_progress_summary_v7_j.md",
        },
        "notes": [
            "This audit directly builds three equal-count latitude regions from raw H latitudes after longitude averaging.",
            "It does not use V6/V7 2-degree latitude interpolation features.",
            "It is a W45-H implementation diagnostic, not a causal or pathway analysis.",
        ],
    }
    _write_json(run_meta, paths.output_dir / "run_meta.json")

    log_md = paths.log_dir / "w45_H_raw025_threeband_progress_v7_j.md"
    _ensure_dir(log_md.parent)
    log_md.write_text(
        "# W45 H raw025 three-band progress V7-j\n\n"
        f"Created: {_now_iso()}\n\n"
        "This run builds low/mid/high H regions directly from raw latitude points, not from the 2-degree interpolation profile.\n",
        encoding="utf-8",
    )


__all__ = ["run_w45_H_raw025_threeband_progress_v7_j"]
