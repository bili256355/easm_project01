from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import warnings

import numpy as np
import pandas as pd

from stage_partition_v6.io import load_smoothed_fields
from stage_partition_v6.state_builder import build_profiles, build_state_matrix

from .config import StagePartitionV7Settings
from .field_state import build_field_state_matrix_for_year_indices

FIELD = "H"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
OUTPUT_TAG = "w45_H_latband_progress_v7_g"
BAND_SIZE = 3
BAND_STEP = 1


@dataclass
class W45HLatBandPaths:
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


def _resolve_paths(v7_root: Optional[Path]) -> W45HLatBandPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = v7_root.parents[1]
    return W45HLatBandPaths(
        v7_root=v7_root,
        project_root=project_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7f_output_dir=v7_root / "outputs" / "w45_H_feature_progress_v7_f",
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _configure_settings(paths: W45HLatBandPaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    return settings


def _load_w45_window(paths: W45HLatBandPaths, n_days: int) -> dict:
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


def _state_to_full_matrix(matrix: np.ndarray, valid_day_index: np.ndarray, n_days: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    valid_day_index = np.asarray(valid_day_index, dtype=int)
    full = np.full((int(n_days), matrix.shape[1]), np.nan, dtype=float)
    full[valid_day_index, :] = matrix
    return full


def _finite_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if int(good.sum()) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(x[good]))


def _nanmean_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=0)


def _row_norm(row: np.ndarray) -> float:
    row = np.asarray(row, dtype=float)
    good = np.isfinite(row)
    if int(good.sum()) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(row[good] ** 2)))


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


def _build_sliding_lat_bands(lat_values: np.ndarray, band_size: int = BAND_SIZE, step: int = BAND_STEP) -> list[dict]:
    lat_values = np.asarray(lat_values, dtype=float)
    bands: list[dict] = []
    n = int(len(lat_values))
    if n < band_size:
        raise ValueError(f"Need at least {band_size} H features to build sliding latitude bands; got {n}")
    band_id = 0
    for start in range(0, n - band_size + 1, step):
        idx = np.arange(start, start + band_size, dtype=int)
        lats = lat_values[idx]
        lat_min = float(np.nanmin(lats))
        lat_max = float(np.nanmax(lats))
        lat_center = float(np.nanmean(lats))
        label = f"H_band_{lat_min:g}_{lat_max:g}"
        bands.append({
            "band_id": int(band_id),
            "band_label": label,
            "feature_indices": idx,
            "feature_indices_str": ";".join(str(int(i)) for i in idx),
            "lat_values_str": ";".join(f"{float(v):g}" for v in lats),
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lat_center": lat_center,
            "n_features": int(len(idx)),
        })
        band_id += 1
    return bands


def _band_progress(
    full_state: np.ndarray,
    band: dict,
    window: dict,
    settings: StagePartitionV7Settings,
    sample_col: str | None = None,
    sample_value: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    cfg = settings.progress_timing
    idx = np.asarray(band["feature_indices"], dtype=int)
    x = np.asarray(full_state[:, idx], dtype=float)
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
        "band_id": int(band["band_id"]),
        "band_label": str(band["band_label"]),
        "feature_indices": str(band["feature_indices_str"]),
        "lat_values": str(band["lat_values_str"]),
        "lat_min": float(band["lat_min"]),
        "lat_max": float(band["lat_max"]),
        "lat_center": float(band["lat_center"]),
        "n_features": int(band["n_features"]),
        "accepted_window_start": int(window["accepted_window_start"]),
        "accepted_window_end": int(window["accepted_window_end"]),
        "analysis_window_start": astart,
        "analysis_window_end": aend,
        "pre_period_start": pre_start,
        "pre_period_end": pre_end,
        "post_period_start": post_start,
        "post_period_end": post_end,
    }
    if sample_col is not None:
        base[sample_col] = int(sample_value) if sample_value is not None else np.nan

    pre = x[pre_start : pre_end + 1, :]
    post = x[post_start : post_end + 1, :]
    pre_good_days = int(np.all(np.isfinite(pre), axis=1).sum())
    post_good_days = int(np.all(np.isfinite(post), axis=1).sum())
    if pre_good_days < int(cfg.min_period_days) or post_good_days < int(cfg.min_period_days):
        row = {**base, "pre_post_distance": np.nan, "within_pre_variability": np.nan, "within_post_variability": np.nan, "separation_ratio": np.nan, "pre_post_separation_label": "period_too_short", "onset_day": np.nan, "midpoint_day": np.nan, "finish_day": np.nan, "duration": np.nan, "progress_monotonicity_corr": np.nan, "n_crossings_025": np.nan, "n_crossings_050": np.nan, "n_crossings_075": np.nan, "progress_quality_label": "period_too_short", "near_left_boundary": False, "near_right_boundary": False}
        return row, pd.DataFrame()

    pre_proto = _nanmean_vec(pre)
    post_proto = _nanmean_vec(post)
    vec = post_proto - pre_proto
    good_vec = np.isfinite(vec)
    norm2 = float(np.nansum(vec[good_vec] ** 2)) if int(good_vec.sum()) else np.nan
    pre_post_distance = float(np.sqrt(np.nanmean(vec[good_vec] ** 2))) if int(good_vec.sum()) else np.nan
    pre_dists = []
    for rr in pre:
        pre_dists.append(_row_norm(rr - pre_proto))
    post_dists = []
    for rr in post:
        post_dists.append(_row_norm(rr - post_proto))
    pre_var = _finite_mean(np.asarray(pre_dists, dtype=float))
    post_var = _finite_mean(np.asarray(post_dists, dtype=float))
    within = _finite_mean(np.asarray([pre_var, post_var], dtype=float))
    separation_ratio = float(pre_post_distance / (within + 1e-12)) if np.isfinite(within) else np.nan
    sep_label = _separation_label(separation_ratio, settings)

    curve_rows: list[dict] = []
    days: list[int] = []
    vals: list[float] = []
    if np.isfinite(norm2) and norm2 > float(cfg.min_transition_norm):
        for day in range(astart, aend + 1):
            val = x[day, :]
            good = np.isfinite(val) & good_vec
            if int(good.sum()) == 0:
                continue
            denom = float(np.nansum(vec[good] ** 2))
            if denom <= float(cfg.min_transition_norm):
                continue
            raw = float(np.nansum((val[good] - pre_proto[good]) * vec[good]) / denom)
            progress = float(np.clip(raw, float(cfg.progress_clip_min), float(cfg.progress_clip_max)))
            days.append(int(day))
            vals.append(progress)
            curve_rows.append({**base, "day": int(day), "relative_to_anchor": int(day - anchor), "raw_progress": raw, "progress": progress})

    days_arr = np.asarray(days, dtype=int)
    vals_arr = np.asarray(vals, dtype=float)
    near_left = near_right = False
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
        "near_left_boundary": bool(near_left),
        "near_right_boundary": bool(near_right),
    }
    return row, pd.DataFrame(curve_rows)


def _load_or_make_bootstrap_indices(profiles: dict, paths: W45HLatBandPaths, settings: StagePartitionV7Settings) -> list[np.ndarray]:
    n_years = int(np.asarray(profiles[FIELD].raw_cube).shape[0])
    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    src = paths.v7e_output_dir / "bootstrap_resample_year_indices_v7_e.csv"
    out: list[np.ndarray] = []
    if src.exists():
        df = pd.read_csv(src)
        if {"bootstrap_id", "sampled_year_indices"}.issubset(df.columns):
            for _, r in df.sort_values("bootstrap_id").iterrows():
                vals = [int(x) for x in str(r["sampled_year_indices"]).split(";") if str(x).strip() != ""]
                if vals:
                    out.append(np.asarray(vals, dtype=int))
            if len(out) >= n_boot:
                return out[:n_boot]
    rng = np.random.default_rng(int(settings.bootstrap.random_seed))
    return [rng.integers(0, n_years, size=n_years, dtype=int) for _ in range(n_boot)]


def _summarize_band_samples(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if samples.empty:
        return pd.DataFrame()
    group_cols = ["band_id", "band_label", "lat_min", "lat_max", "lat_center", "n_features"]
    for keys, sub in samples.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys))
        mids = pd.to_numeric(sub["midpoint_day"], errors="coerce").dropna().to_numpy(dtype=float)
        onsets = pd.to_numeric(sub["onset_day"], errors="coerce").dropna().to_numpy(dtype=float)
        finishes = pd.to_numeric(sub["finish_day"], errors="coerce").dropna().to_numpy(dtype=float)
        durations = pd.to_numeric(sub["duration"], errors="coerce").dropna().to_numpy(dtype=float)
        qual_counts = sub["progress_quality_label"].astype(str).value_counts().to_dict() if "progress_quality_label" in sub.columns else {}
        sep_counts = sub["pre_post_separation_label"].astype(str).value_counts().to_dict() if "pre_post_separation_label" in sub.columns else {}
        row = {**key, "n_samples": int(len(sub)), "n_valid_midpoints": int(mids.size), "valid_midpoint_fraction": float(mids.size / len(sub)) if len(sub) else np.nan}
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
                "duration_q05": float(np.nanpercentile(durations, 5)) if durations.size else np.nan,
                "duration_q95": float(np.nanpercentile(durations, 95)) if durations.size else np.nan,
                "duration_q25": float(np.nanpercentile(durations, 25)) if durations.size else np.nan,
                "duration_q75": float(np.nanpercentile(durations, 75)) if durations.size else np.nan,
            })
        else:
            row.update({"onset_median": np.nan, "onset_q05": np.nan, "onset_q95": np.nan, "midpoint_median": np.nan, "midpoint_q05": np.nan, "midpoint_q95": np.nan, "midpoint_q25": np.nan, "midpoint_q75": np.nan, "midpoint_iqr": np.nan, "midpoint_q90_width": np.nan, "finish_median": np.nan, "finish_q05": np.nan, "finish_q95": np.nan, "duration_median": np.nan, "duration_q05": np.nan, "duration_q95": np.nan, "duration_q25": np.nan, "duration_q75": np.nan})
        row.update({
            "dominant_progress_quality_label": max(qual_counts, key=qual_counts.get) if qual_counts else "none",
            "dominant_prepost_separation_label": max(sep_counts, key=sep_counts.get) if sep_counts else "none",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _assign_band_timing_classes(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    finite = pd.to_numeric(out.get("midpoint_median"), errors="coerce")
    finite_vals = finite[np.isfinite(finite)].to_numpy(dtype=float)
    if finite_vals.size == 0:
        out["timing_position_label"] = "unstable_band"
        out["latband_stability_label"] = "unstable_band"
        return out
    q33 = float(np.nanpercentile(finite_vals, 33.333))
    q66 = float(np.nanpercentile(finite_vals, 66.667))
    pos_labels = []
    stab_labels = []
    for _, r in out.iterrows():
        qlabel = str(r.get("dominant_progress_quality_label", ""))
        sep = str(r.get("dominant_prepost_separation_label", ""))
        med = pd.to_numeric(pd.Series([r.get("midpoint_median")]), errors="coerce").iloc[0]
        valid_frac = float(r.get("valid_midpoint_fraction", 0.0)) if pd.notna(r.get("valid_midpoint_fraction", np.nan)) else 0.0
        if not np.isfinite(med) or valid_frac < 0.5 or sep in {"no_clear_separation", "unavailable"} or "nonmonotonic" in qlabel:
            pos_labels.append("unstable_band")
            stab_labels.append("unstable_band")
        else:
            if med <= q33:
                pos_labels.append("early_band_candidate")
            elif med >= q66:
                pos_labels.append("late_band_candidate")
            else:
                pos_labels.append("middle_band_candidate")
            stab_labels.append("stable_enough_for_diagnostic")
    out["timing_position_label"] = pos_labels
    out["latband_stability_label"] = stab_labels
    out["timing_class_note"] = "diagnostic tertile class based on band midpoint_median within W45 H lat-bands; not a statistical significance label"
    return out


def _load_v7e_whole_h(paths: W45HLatBandPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv", required=False)
    summary = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_summary_v7_e.csv", required=False)
    if not observed.empty:
        observed = observed[(observed.get("window_id", "").astype(str) == WINDOW_ID) & (observed.get("field", "").astype(str) == FIELD)]
    if not summary.empty:
        summary = summary[(summary.get("window_id", "").astype(str) == WINDOW_ID) & (summary.get("field", "").astype(str) == FIELD)]
    return observed, summary


def _load_v7f_single_feature(paths: W45HLatBandPaths) -> pd.DataFrame:
    df = _read_csv(paths.v7f_output_dir / "w45_H_feature_progress_bootstrap_summary_v7_f.csv", required=False)
    if df.empty:
        return df
    return df.copy()


def _build_comparison(latband_summary: pd.DataFrame, paths: W45HLatBandPaths) -> pd.DataFrame:
    rows: list[dict] = []
    whole_obs, whole_sum = _load_v7e_whole_h(paths)
    if not whole_sum.empty or not whole_obs.empty:
        row = {"unit_type": "whole_field_H", "unit_label": "H_whole_field", "lat_min": np.nan, "lat_max": np.nan, "lat_center": np.nan}
        if not whole_sum.empty:
            r = whole_sum.iloc[0]
            row.update({
                "midpoint_median": r.get("midpoint_median", np.nan),
                "midpoint_q05": r.get("midpoint_q05", r.get("midpoint_q025", np.nan)),
                "midpoint_q95": r.get("midpoint_q95", r.get("midpoint_q975", np.nan)),
                "midpoint_q90_width": (r.get("midpoint_q95", r.get("midpoint_q975", np.nan)) - r.get("midpoint_q05", r.get("midpoint_q025", np.nan))) if pd.notna(r.get("midpoint_q95", r.get("midpoint_q975", np.nan))) and pd.notna(r.get("midpoint_q05", r.get("midpoint_q025", np.nan))) else np.nan,
                "progress_quality_dominant": r.get("dominant_progress_quality_label", np.nan),
                "interpretation": "whole-field H reference from V7-e",
            })
        elif not whole_obs.empty:
            r = whole_obs.iloc[0]
            row.update({"midpoint_median": r.get("midpoint_day", np.nan), "midpoint_q05": np.nan, "midpoint_q95": np.nan, "midpoint_q90_width": np.nan, "progress_quality_dominant": r.get("progress_quality_label", np.nan), "interpretation": "whole-field H observed reference from V7-e"})
        rows.append(row)
    feat = _load_v7f_single_feature(paths)
    if not feat.empty:
        for _, r in feat.iterrows():
            rows.append({
                "unit_type": "single_lat_feature",
                "unit_label": f"H_feature_{r.get('lat_value')}",
                "lat_min": r.get("lat_value", np.nan),
                "lat_max": r.get("lat_value", np.nan),
                "lat_center": r.get("lat_value", np.nan),
                "midpoint_median": r.get("midpoint_median", np.nan),
                "midpoint_q05": r.get("midpoint_q05", np.nan),
                "midpoint_q95": r.get("midpoint_q95", np.nan),
                "midpoint_q90_width": r.get("midpoint_q90_width", np.nan),
                "progress_quality_dominant": r.get("dominant_progress_quality_label", np.nan),
                "interpretation": "single-lat feature reference from V7-f",
            })
    for _, r in latband_summary.iterrows():
        rows.append({
            "unit_type": "local_lat_band",
            "unit_label": r.get("band_label", ""),
            "lat_min": r.get("lat_min", np.nan),
            "lat_max": r.get("lat_max", np.nan),
            "lat_center": r.get("lat_center", np.nan),
            "midpoint_median": r.get("midpoint_median", np.nan),
            "midpoint_q05": r.get("midpoint_q05", np.nan),
            "midpoint_q95": r.get("midpoint_q95", np.nan),
            "midpoint_q90_width": r.get("midpoint_q90_width", np.nan),
            "progress_quality_dominant": r.get("dominant_progress_quality_label", np.nan),
            "interpretation": "local 3-lat-band result from V7-g",
        })
    return pd.DataFrame(rows)


def _build_upstream_implication(latband_summary: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    feat = comparison[comparison["unit_type"] == "single_lat_feature"].copy() if not comparison.empty else pd.DataFrame()
    band = comparison[comparison["unit_type"] == "local_lat_band"].copy() if not comparison.empty else pd.DataFrame()
    feat_width = pd.to_numeric(feat.get("midpoint_q90_width"), errors="coerce").dropna().to_numpy(dtype=float) if not feat.empty else np.array([])
    band_width = pd.to_numeric(band.get("midpoint_q90_width"), errors="coerce").dropna().to_numpy(dtype=float) if not band.empty else np.array([])
    feat_unstable = int(feat["progress_quality_dominant"].astype(str).str.contains("nonmonotonic|unavailable|no_clear", regex=True).sum()) if not feat.empty and "progress_quality_dominant" in feat.columns else 0
    band_unstable = int(latband_summary["latband_stability_label"].astype(str).eq("unstable_band").sum()) if not latband_summary.empty and "latband_stability_label" in latband_summary.columns else 0
    n_feat = int(len(feat))
    n_band = int(len(band))
    med_feat_width = float(np.nanmedian(feat_width)) if feat_width.size else np.nan
    med_band_width = float(np.nanmedian(band_width)) if band_width.size else np.nan
    width_improves = np.isfinite(med_feat_width) and np.isfinite(med_band_width) and med_band_width < med_feat_width
    unstable_improves = n_feat > 0 and n_band > 0 and (band_unstable / max(n_band, 1)) < (feat_unstable / max(n_feat, 1))
    if width_improves and unstable_improves:
        status = "yes"
        evidence = f"median q90 width single-feature={med_feat_width}, lat-band={med_band_width}; unstable fraction single-feature={feat_unstable}/{n_feat}, lat-band={band_unstable}/{n_band}"
        implication = "single-feature unit is likely too noisy; future region-level progress should prefer local band/patch units."
        followup = "design general region-unit progress framework before extending to other objects."
    elif width_improves or unstable_improves:
        status = "partial"
        evidence = f"median q90 width single-feature={med_feat_width}, lat-band={med_band_width}; unstable fraction single-feature={feat_unstable}/{n_feat}, lat-band={band_unstable}/{n_band}"
        implication = "lat-band aggregation partially improves stability, but not enough to treat H regional timing as confirmed."
        followup = "review lat-band quality and consider physically motivated bands only if diagnostics are coherent."
    else:
        status = "no"
        evidence = f"median q90 width single-feature={med_feat_width}, lat-band={med_band_width}; unstable fraction single-feature={feat_unstable}/{n_feat}, lat-band={band_unstable}/{n_band}"
        implication = "W45-H instability is not resolved by local latitude-band aggregation; the issue may be progress definition or W45-H itself."
        followup = "do not upgrade W45-H order; inspect progress definition or retain W45 as candidate only."
    rows.append({"diagnostic_item": "latband_improves_stability", "status": status, "evidence": evidence, "upstream_implication": implication, "recommended_followup": followup})
    rows.append({"diagnostic_item": "single_feature_unit_too_noisy", "status": "yes" if status in {"yes", "partial"} else "unconfirmed", "evidence": evidence, "upstream_implication": "If confirmed, single-lat feature should not be the base unit for future feature/region-level progress.", "recommended_followup": followup})
    rows.append({"diagnostic_item": "region_level_progress_recommended", "status": "yes" if status == "yes" else ("conditional" if status == "partial" else "no"), "evidence": evidence, "upstream_implication": implication, "recommended_followup": followup})
    return pd.DataFrame(rows)


def _build_next_step_decision(latband_summary: pd.DataFrame, comparison: pd.DataFrame, upstream: pd.DataFrame) -> pd.DataFrame:
    status = "unknown"
    if not upstream.empty:
        sub = upstream[upstream["diagnostic_item"] == "latband_improves_stability"]
        if not sub.empty:
            status = str(sub.iloc[0].get("status", "unknown"))
    n_early = int((latband_summary.get("timing_position_label", pd.Series(dtype=str)).astype(str) == "early_band_candidate").sum()) if not latband_summary.empty else 0
    n_late = int((latband_summary.get("timing_position_label", pd.Series(dtype=str)).astype(str) == "late_band_candidate").sum()) if not latband_summary.empty else 0
    n_unstable = int((latband_summary.get("latband_stability_label", pd.Series(dtype=str)).astype(str) == "unstable_band").sum()) if not latband_summary.empty else 0
    if status == "yes" and n_early > 0 and n_late > 0:
        action = "proceed_to_H_region_level_progress_design"
        reason = "Lat-band aggregation improves stability and reveals both early/late diagnostic bands."
        priority = "high"
    elif status == "yes":
        action = "use_latband_as_preferred_H_subfield_unit_but_do_not_claim_regional_order_yet"
        reason = "Lat-band aggregation improves stability, but early/late separation is not yet sufficiently structured."
        priority = "moderate"
    elif status == "partial":
        action = "review_latband_quality_before_generalizing_to_other_objects"
        reason = "Lat-band aggregation partially improves single-feature instability, but evidence is still not ready for confirmed regional timing."
        priority = "diagnostic"
    else:
        action = "do_not_upgrade_W45_H_with_latband; retain_candidate_or_review_progress_definition"
        reason = "Lat-band aggregation does not clearly resolve feature-level instability."
        priority = "low"
    return pd.DataFrame([{
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "band_size": BAND_SIZE,
        "band_step": BAND_STEP,
        "n_latbands": int(len(latband_summary)),
        "n_early_bands": n_early,
        "n_late_bands": n_late,
        "n_unstable_bands": n_unstable,
        "latband_improves_stability": status,
        "recommended_next_action": action,
        "region_level_priority": priority,
        "decision_reason": reason,
    }])


def _plot_heatmap(curves: pd.DataFrame, paths: W45HLatBandPaths) -> None:
    if curves.empty:
        return
    try:
        import matplotlib.pyplot as plt
        piv = curves.pivot_table(index="band_label", columns="day", values="progress", aggfunc="first")
        # Sort labels by lat_center using mapping.
        centers = curves.drop_duplicates("band_label").set_index("band_label")["lat_center"].to_dict()
        ordered = sorted(piv.index, key=lambda x: centers.get(x, np.nan))
        piv = piv.loc[ordered]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", origin="lower", extent=[piv.columns.min(), piv.columns.max(), -0.5, len(piv.index)-0.5])
        ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("day index")
        ax.set_ylabel("H local latitude band")
        ax.set_title("W45 H local latitude-band progress")
        fig.colorbar(im, ax=ax, label="progress")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_latband_progress_heatmap_v7_g.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        _ensure_dir(paths.log_dir)
        (paths.log_dir / "plot_latband_heatmap_error.txt").write_text(str(exc), encoding="utf-8")


def _plot_midpoint_by_band(summary: pd.DataFrame, paths: W45HLatBandPaths) -> None:
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
        df = summary.sort_values("lat_center")
        x = pd.to_numeric(df["lat_center"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["midpoint_median"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(df["midpoint_q05"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df["midpoint_q95"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(y - lo, 0), np.maximum(hi - y, 0)])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-")
        ax.axhline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_xlabel("H latitude-band center")
        ax.set_ylabel("bootstrap median midpoint day")
        ax.set_title("W45 H local latitude-band midpoint")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_latband_midpoint_by_band_v7_g.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        _ensure_dir(paths.log_dir)
        (paths.log_dir / "plot_latband_midpoint_error.txt").write_text(str(exc), encoding="utf-8")


def _write_summary_md(paths: W45HLatBandPaths, latband_summary: pd.DataFrame, comparison: pd.DataFrame, upstream: pd.DataFrame, decision: pd.DataFrame, input_audit: dict) -> None:
    lines = []
    lines.append("# W45 H local latitude-band progress audit v7_g")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- Diagnose whether W45/H single-lat feature instability is reduced by using sliding local 3-latitude bands.")
    lines.append("- This is a W45/H-only implementation diagnostic and does not upgrade W45 order evidence by itself.")
    lines.append("- It does not infer causality and does not alter V7-e/V7-e1/V7-e2 outputs.")
    lines.append("")
    lines.append("## Lat-band construction")
    lines.append(f"- band_size: {BAND_SIZE}")
    lines.append(f"- band_step: {BAND_STEP}")
    lines.append(f"- n_latbands: {len(latband_summary)}")
    lines.append("")
    if not latband_summary.empty:
        lines.append("## Lat-band timing classes")
        for k, v in latband_summary.get("timing_position_label", pd.Series(dtype=str)).astype(str).value_counts().to_dict().items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if not upstream.empty:
        lines.append("## Upstream implication")
        for _, r in upstream.iterrows():
            lines.append(f"- {r.get('diagnostic_item')}: {r.get('status')} | {r.get('upstream_implication')}")
        lines.append("")
    if not decision.empty:
        r = decision.iloc[0]
        lines.append("## Recommended next step")
        for k in ["latband_improves_stability", "recommended_next_action", "region_level_priority", "decision_reason"]:
            lines.append(f"- {k}: {r.get(k)}")
        lines.append("")
    lines.append("## Input audit")
    for k, v in input_audit.get("checks", {}).items():
        lines.append(f"- {k}: {v}")
    (paths.output_dir / "w45_H_latband_progress_summary_v7_g.md").write_text("\n".join(lines), encoding="utf-8")


def _input_audit(paths: W45HLatBandPaths, settings: StagePartitionV7Settings, window: dict) -> dict:
    return {
        "created_at": _now_iso(),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "band_size": BAND_SIZE,
        "band_step": BAND_STEP,
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
        },
    }


def run_w45_H_latband_progress_v7_g(v7_root: Path | None = None) -> None:
    paths = _resolve_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)
    settings = _configure_settings(paths)

    meta: dict = {
        "status": "running",
        "created_at": _now_iso(),
        "output_tag": OUTPUT_TAG,
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "band_size": BAND_SIZE,
        "band_step": BAND_STEP,
        "notes": [
            "W45 H local latitude-band progress diagnostic only.",
            "This run changes only the subfield unit from single-lat feature to sliding local latitude band.",
            "It does not alter V7-e whole-field progress, V7-e1 tests, or V7-e2 failure audit outputs.",
            "No causal/pathway interpretation is made.",
        ],
    }
    _write_json(meta, paths.output_dir / "run_meta.json")

    print("[v7_g] loading smoothed fields and building profiles...")
    smoothed = load_smoothed_fields(settings.foundation.smoothed_fields_path())
    profiles = build_profiles(smoothed, settings.profile)
    n_days = int(np.asarray(profiles[FIELD].raw_cube).shape[1])
    window = _load_w45_window(paths, n_days)
    input_audit = _input_audit(paths, settings, window)
    _write_json(input_audit, paths.output_dir / "input_audit_v7_g.json")

    print("[v7_g] building shared valid-day index...")
    shared_valid_day_index = None
    try:
        joint_state = build_state_matrix(profiles, settings.state)
        if isinstance(joint_state, dict) and "valid_day_index" in joint_state:
            shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)
    except Exception as exc:
        (paths.log_dir / "joint_valid_day_warning.txt").write_text(str(exc), encoding="utf-8")
        shared_valid_day_index = None

    h_lats = np.asarray(profiles[FIELD].lat_grid, dtype=float)
    bands = _build_sliding_lat_bands(h_lats, band_size=BAND_SIZE, step=BAND_STEP)
    print(f"[v7_g] H features: {len(h_lats)}, lat-bands: {len(bands)}")

    observed_matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(
        profiles,
        FIELD,
        None,
        standardize=settings.state.standardize,
        trim_invalid_days=settings.state.trim_invalid_days,
        shared_valid_day_index=shared_valid_day_index,
    )
    observed_full = _state_to_full_matrix(observed_matrix, valid_day_index, n_days)
    observed_rows: list[dict] = []
    curve_frames: list[pd.DataFrame] = []
    for band in bands:
        row, curves = _band_progress(observed_full, band, window, settings)
        observed_rows.append(row)
        if not curves.empty:
            curve_frames.append(curves)
    observed_df = pd.DataFrame(observed_rows)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    _write_csv(observed_df, paths.output_dir / "w45_H_latband_progress_observed_v7_g.csv")
    _write_csv(curves_df, paths.output_dir / "w45_H_latband_progress_curves_long_v7_g.csv")

    print("[v7_g] running bootstrap lat-band progress...")
    bootstrap_indices = _load_or_make_bootstrap_indices(profiles, paths, settings)
    boot_rows: list[dict] = []
    for b, year_idx in enumerate(bootstrap_indices):
        if b % int(settings.bootstrap.progress_every) == 0:
            print(f"[v7_g] bootstrap {b}/{len(bootstrap_indices)}")
        mat, vdi, _ = build_field_state_matrix_for_year_indices(
            profiles,
            FIELD,
            year_idx,
            standardize=settings.state.standardize,
            trim_invalid_days=settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index,
        )
        full = _state_to_full_matrix(mat, vdi, n_days)
        for band in bands:
            row, _ = _band_progress(full, band, window, settings, sample_col="bootstrap_id", sample_value=b)
            boot_rows.append(row)
    boot_df = pd.DataFrame(boot_rows)
    _write_csv(boot_df, paths.output_dir / "w45_H_latband_progress_bootstrap_samples_v7_g.csv")

    print("[v7_g] summarizing lat-band progress...")
    boot_summary = _summarize_band_samples(boot_df)
    boot_summary = _assign_band_timing_classes(boot_summary)
    _write_csv(boot_summary, paths.output_dir / "w45_H_latband_progress_bootstrap_summary_v7_g.csv")
    _write_csv(boot_summary, paths.output_dir / "w45_H_latband_timing_spread_v7_g.csv")

    comparison = _build_comparison(boot_summary, paths)
    _write_csv(comparison, paths.output_dir / "w45_H_whole_feature_latband_comparison_v7_g.csv")
    upstream = _build_upstream_implication(boot_summary, comparison)
    _write_csv(upstream, paths.output_dir / "w45_H_latband_upstream_implication_v7_g.csv")
    decision = _build_next_step_decision(boot_summary, comparison, upstream)
    _write_csv(decision, paths.output_dir / "w45_H_latband_next_step_decision_v7_g.csv")

    print("[v7_g] writing figures...")
    _plot_heatmap(curves_df, paths)
    _plot_midpoint_by_band(boot_summary, paths)

    _write_summary_md(paths, boot_summary, comparison, upstream, decision, input_audit)
    meta.update({
        "status": "success",
        "finished_at": _now_iso(),
        "n_h_features": int(len(h_lats)),
        "n_latbands": int(len(bands)),
        "n_bootstrap": int(len(bootstrap_indices)),
        "output_dir": str(paths.output_dir),
        "log_dir": str(paths.log_dir),
        "core_outputs": [
            "w45_H_latband_progress_observed_v7_g.csv",
            "w45_H_latband_progress_curves_long_v7_g.csv",
            "w45_H_latband_progress_bootstrap_samples_v7_g.csv",
            "w45_H_latband_progress_bootstrap_summary_v7_g.csv",
            "w45_H_latband_timing_spread_v7_g.csv",
            "w45_H_whole_feature_latband_comparison_v7_g.csv",
            "w45_H_latband_upstream_implication_v7_g.csv",
            "w45_H_latband_next_step_decision_v7_g.csv",
            "w45_H_latband_progress_summary_v7_g.md",
        ],
    })
    _write_json(meta, paths.output_dir / "run_meta.json")
    (paths.log_dir / "w45_H_latband_progress_v7_g.md").write_text(
        "\n".join([
            "# W45 H local latitude-band progress v7_g",
            "",
            f"status: {meta['status']}",
            f"created_at: {meta['created_at']}",
            f"finished_at: {meta['finished_at']}",
            f"n_h_features: {meta['n_h_features']}",
            f"n_latbands: {meta['n_latbands']}",
            f"n_bootstrap: {meta['n_bootstrap']}",
            "",
            "This is a W45/H-only local latitude-band progress diagnostic. It does not modify V7-e whole-field results and does not infer causality.",
        ]),
        encoding="utf-8",
    )
    print(f"[v7_g] done: {paths.output_dir}")
