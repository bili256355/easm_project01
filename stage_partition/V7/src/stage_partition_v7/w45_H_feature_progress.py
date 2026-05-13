from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import sys
import warnings

# Hotfix: this V7 diagnostic reuses V6 profile/state helpers. When the
# script is launched from V7, only V7/src is guaranteed to be on sys.path,
# so add the sibling V6/src explicitly before importing stage_partition_v6.
_THIS_FILE = Path(__file__).resolve()
_V7_ROOT = _THIS_FILE.parents[2]
_V6_SRC_ROOT = _V7_ROOT.parent / "V6" / "src"
if _V6_SRC_ROOT.exists() and str(_V6_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_V6_SRC_ROOT))

import numpy as np
import pandas as pd

from stage_partition_v6.io import load_smoothed_fields
from stage_partition_v6.state_builder import build_profiles, build_state_matrix

from .config import StagePartitionV7Settings
from .field_state import build_field_state_matrix_for_year_indices

FIELD = "H"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
OUTPUT_TAG = "w45_H_feature_progress_v7_f"


@dataclass
class W45HFeaturePaths:
    v7_root: Path
    project_root: Path
    v7e_output_dir: Path
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


def _resolve_paths(v7_root: Optional[Path]) -> W45HFeaturePaths:
    if v7_root is None:
        # stage_partition/V7/src/stage_partition_v7/w45_H_feature_progress.py
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    # D:/easm_project01/stage_partition/V7 -> project root is parents[1]
    project_root = v7_root.parents[1]
    return W45HFeaturePaths(
        v7_root=v7_root,
        project_root=project_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _configure_settings(paths: W45HFeaturePaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    return settings


def _load_w45_window(paths: W45HFeaturePaths, settings: StagePartitionV7Settings, n_days: int) -> dict:
    """Read W45 progress window from V7-e if possible; otherwise use documented fallback."""
    win_path = paths.v7e_output_dir / "accepted_windows_used_v7_e.csv"
    if win_path.exists():
        df = pd.read_csv(win_path)
        if "window_id" in df.columns:
            sub = df[df["window_id"].astype(str) == WINDOW_ID]
        else:
            sub = pd.DataFrame()
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
    # Fallback matches the V7-e documented W45 progress window.
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


def _feature_progress(
    full_state: np.ndarray,
    feature_index: int,
    lat_value: float,
    window: dict,
    settings: StagePartitionV7Settings,
    sample_col: str | None = None,
    sample_value: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    cfg = settings.progress_timing
    x = np.asarray(full_state[:, int(feature_index)], dtype=float)
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
        "feature_index": int(feature_index),
        "lat_value": float(lat_value),
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

    pre = x[pre_start : pre_end + 1]
    post = x[post_start : post_end + 1]
    pre_good = int(np.isfinite(pre).sum())
    post_good = int(np.isfinite(post).sum())
    if pre_good < int(cfg.min_period_days) or post_good < int(cfg.min_period_days):
        row = {**base, "pre_post_distance": np.nan, "within_pre_variability": np.nan, "within_post_variability": np.nan, "separation_ratio": np.nan, "pre_post_separation_label": "period_too_short", "onset_day": np.nan, "midpoint_day": np.nan, "finish_day": np.nan, "duration": np.nan, "progress_monotonicity_corr": np.nan, "n_crossings_025": np.nan, "n_crossings_050": np.nan, "n_crossings_075": np.nan, "progress_quality_label": "period_too_short", "near_left_boundary": False, "near_right_boundary": False}
        return row, pd.DataFrame()

    pre_proto = _finite_mean(pre)
    post_proto = _finite_mean(post)
    vec = post_proto - pre_proto
    pre_post_distance = float(abs(vec)) if np.isfinite(vec) else np.nan
    pre_var = _finite_mean(np.abs(pre[np.isfinite(pre)] - pre_proto)) if np.isfinite(pre_proto) else np.nan
    post_var = _finite_mean(np.abs(post[np.isfinite(post)] - post_proto)) if np.isfinite(post_proto) else np.nan
    within = _finite_mean(np.asarray([pre_var, post_var], dtype=float))
    separation_ratio = float(pre_post_distance / (within + 1e-12)) if np.isfinite(within) else np.nan
    sep_label = _separation_label(separation_ratio, settings)

    curve_rows: list[dict] = []
    days: list[int] = []
    vals: list[float] = []
    if np.isfinite(vec) and abs(vec) > float(cfg.min_transition_norm):
        for day in range(astart, aend + 1):
            val = x[day]
            if not np.isfinite(val):
                continue
            raw = float((val - pre_proto) / vec)
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


def _load_or_make_bootstrap_indices(profiles: dict, paths: W45HFeaturePaths, settings: StagePartitionV7Settings) -> list[np.ndarray]:
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


def _summarize_feature_samples(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if samples.empty:
        return pd.DataFrame()
    for (idx, latv), sub in samples.groupby(["feature_index", "lat_value"], sort=True):
        mids = pd.to_numeric(sub["midpoint_day"], errors="coerce").dropna().to_numpy(dtype=float)
        onsets = pd.to_numeric(sub["onset_day"], errors="coerce").dropna().to_numpy(dtype=float)
        finishes = pd.to_numeric(sub["finish_day"], errors="coerce").dropna().to_numpy(dtype=float)
        durations = pd.to_numeric(sub["duration"], errors="coerce").dropna().to_numpy(dtype=float)
        qual_counts = sub["progress_quality_label"].astype(str).value_counts().to_dict() if "progress_quality_label" in sub.columns else {}
        sep_counts = sub["pre_post_separation_label"].astype(str).value_counts().to_dict() if "pre_post_separation_label" in sub.columns else {}
        if mids.size == 0:
            rows.append({"feature_index": int(idx), "lat_value": float(latv), "n_valid_midpoints": 0})
            continue
        rows.append({
            "feature_index": int(idx),
            "lat_value": float(latv),
            "n_samples": int(len(sub)),
            "n_valid_midpoints": int(mids.size),
            "valid_midpoint_fraction": float(mids.size / len(sub)) if len(sub) else np.nan,
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
            "dominant_progress_quality_label": max(qual_counts, key=qual_counts.get) if qual_counts else "none",
            "dominant_prepost_separation_label": max(sep_counts, key=sep_counts.get) if sep_counts else "none",
        })
    return pd.DataFrame(rows)


def _assign_timing_classes(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    labels = []
    finite = pd.to_numeric(out.get("midpoint_median"), errors="coerce")
    finite_vals = finite[np.isfinite(finite)].to_numpy(dtype=float)
    if finite_vals.size == 0:
        out["timing_class"] = "unstable_feature"
        return out
    q33 = float(np.nanpercentile(finite_vals, 33.333))
    q66 = float(np.nanpercentile(finite_vals, 66.667))
    for _, r in out.iterrows():
        qlabel = str(r.get("dominant_progress_quality_label", ""))
        sep = str(r.get("dominant_prepost_separation_label", ""))
        med = pd.to_numeric(pd.Series([r.get("midpoint_median")]), errors="coerce").iloc[0]
        valid_frac = float(r.get("valid_midpoint_fraction", 0.0)) if pd.notna(r.get("valid_midpoint_fraction", np.nan)) else 0.0
        if not np.isfinite(med) or valid_frac < 0.5 or sep in {"no_clear_separation", "unavailable"} or "nonmonotonic" in qlabel:
            labels.append("unstable_feature")
        elif med <= q33:
            labels.append("early_feature")
        elif med >= q66:
            labels.append("late_feature")
        else:
            labels.append("middle_feature")
    out["timing_class"] = labels
    out["timing_class_note"] = "diagnostic tertile class based on feature midpoint_median within W45 H features; not a statistical significance label"
    return out


def _load_v7e_whole_h(paths: W45HFeaturePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv", required=False)
    summ = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_summary_v7_e.csv", required=False)
    if not obs.empty and {"window_id", "field"}.issubset(obs.columns):
        obs = obs[(obs["window_id"].astype(str) == WINDOW_ID) & (obs["field"].astype(str) == FIELD)].copy()
    if not summ.empty and {"window_id", "field"}.issubset(summ.columns):
        summ = summ[(summ["window_id"].astype(str) == WINDOW_ID) & (summ["field"].astype(str) == FIELD)].copy()
    return obs, summ


def _build_whole_vs_feature(summary: pd.DataFrame, observed: pd.DataFrame, paths: W45HFeaturePaths) -> pd.DataFrame:
    whole_obs, whole_summary = _load_v7e_whole_h(paths)
    row: dict = {
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
    }
    if not whole_obs.empty:
        r = whole_obs.iloc[0]
        row.update({
            "whole_observed_onset": r.get("onset_day", np.nan),
            "whole_observed_midpoint": r.get("midpoint_day", np.nan),
            "whole_observed_finish": r.get("finish_day", np.nan),
            "whole_observed_duration": r.get("duration", np.nan),
            "whole_progress_quality_label": r.get("progress_quality_label", "unavailable"),
        })
    if not whole_summary.empty:
        r = whole_summary.iloc[0]
        row.update({
            "whole_bootstrap_midpoint_median": r.get("midpoint_median", np.nan),
            "whole_bootstrap_midpoint_q05": r.get("midpoint_q05", r.get("midpoint_q025", np.nan)),
            "whole_bootstrap_midpoint_q95": r.get("midpoint_q95", r.get("midpoint_q975", np.nan)),
            "whole_bootstrap_midpoint_iqr": r.get("midpoint_iqr", np.nan),
        })
    finite = summary[np.isfinite(pd.to_numeric(summary.get("midpoint_median"), errors="coerce"))].copy()
    if not finite.empty:
        mids = pd.to_numeric(finite["midpoint_median"], errors="coerce").to_numpy(dtype=float)
        row.update({
            "feature_midpoint_min": float(np.nanmin(mids)),
            "feature_midpoint_max": float(np.nanmax(mids)),
            "feature_midpoint_median": float(np.nanmedian(mids)),
            "feature_midpoint_iqr": float(np.nanpercentile(mids, 75) - np.nanpercentile(mids, 25)),
            "n_features": int(len(summary)),
            "n_early_features": int((summary["timing_class"] == "early_feature").sum()),
            "n_middle_features": int((summary["timing_class"] == "middle_feature").sum()),
            "n_late_features": int((summary["timing_class"] == "late_feature").sum()),
            "n_unstable_features": int((summary["timing_class"] == "unstable_feature").sum()),
        })
        if row.get("n_early_features", 0) > 0 and row.get("n_late_features", 0) > 0:
            interp = "internal_timing_heterogeneity: H feature-level progress contains both early and late latitude features; whole-field broad transition may average heterogeneous timing."
        elif row.get("n_unstable_features", 0) >= max(1, int(len(summary) / 2)):
            interp = "feature_level_unstable: many H features have unstable or invalid progress timing; whole-field H early signal needs caution."
        elif row.get("n_early_features", 0) > 0 and row.get("n_late_features", 0) == 0:
            interp = "mostly_early_or_middle_features: H early signal appears more coherent across features, but broadness should still be checked."
        else:
            interp = "feature_level_not_decisive: feature timing does not clearly identify early/late heterogeneity."
    else:
        interp = "feature_midpoint_unavailable"
    row["interpretation"] = interp
    return pd.DataFrame([row])


def _build_decision(summary: pd.DataFrame, whole_vs: pd.DataFrame) -> pd.DataFrame:
    if whole_vs.empty:
        interp = "unavailable"
        early = late = unstable = 0
    else:
        r = whole_vs.iloc[0]
        interp = str(r.get("interpretation", ""))
        early = int(r.get("n_early_features", 0) or 0)
        late = int(r.get("n_late_features", 0) or 0)
        unstable = int(r.get("n_unstable_features", 0) or 0)
    if "internal_timing_heterogeneity" in interp:
        worth = True
        priority = "high"
        action = "derive_H_latitude_band_or_region_progress_before_upgrading_W45_H_order"
        reason = "Feature-level H timing contains both early and late latitude features, consistent with whole-field broad transition caused by internal heterogeneity."
    elif "mostly_early_or_middle" in interp:
        worth = True
        priority = "moderate"
        action = "retain_W45_H_as_early_broad_candidate_and_optionally_define_H_region_progress"
        reason = "H early signal appears relatively coherent, but whole-field broadness remains a limitation."
    elif "feature_level_unstable" in interp:
        worth = False
        priority = "low_until_quality_checked"
        action = "do_not_upgrade_W45_H_order; inspect_H_feature_progress_quality"
        reason = "Many H features are unstable/invalid, so region-level progress should first diagnose quality rather than seek order."
    else:
        worth = False
        priority = "unclear"
        action = "retain_W45_as_candidate_without_method_upgrade_until_feature_diagnostics_are_reviewed"
        reason = "Feature-level diagnostics did not clearly explain W45 H broad transition."
    return pd.DataFrame([{
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "n_early_features": early,
        "n_late_features": late,
        "n_unstable_features": unstable,
        "whole_vs_feature_interpretation": interp,
        "is_worth_region_level_progress": bool(worth),
        "region_level_priority": priority,
        "recommended_next_action": action,
        "decision_reason": reason,
    }])


def _plot_heatmap(curves: pd.DataFrame, paths: W45HFeaturePaths) -> None:
    if curves.empty:
        return
    try:
        import matplotlib.pyplot as plt
        piv = curves.pivot_table(index="lat_value", columns="day", values="progress", aggfunc="first")
        piv = piv.sort_index(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", origin="lower", extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
        ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_xlabel("day index")
        ax.set_ylabel("H feature latitude")
        ax.set_title("W45 H feature-level progress")
        fig.colorbar(im, ax=ax, label="progress")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_feature_progress_heatmap_v7_f.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (paths.log_dir / "plot_heatmap_error.txt").write_text(str(exc), encoding="utf-8")


def _plot_midpoint_by_lat(summary: pd.DataFrame, paths: W45HFeaturePaths) -> None:
    if summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
        df = summary.sort_values("lat_value")
        x = pd.to_numeric(df["lat_value"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["midpoint_median"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(df["midpoint_q05"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df["midpoint_q95"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(y - lo, 0), np.maximum(hi - y, 0)])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-")
        ax.axhline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_xlabel("H feature latitude")
        ax.set_ylabel("bootstrap median midpoint day")
        ax.set_title("W45 H feature midpoint by latitude")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_feature_midpoint_by_lat_v7_f.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (paths.log_dir / "plot_midpoint_error.txt").write_text(str(exc), encoding="utf-8")


def _write_summary_md(paths: W45HFeaturePaths, observed: pd.DataFrame, feature_summary: pd.DataFrame, whole_vs: pd.DataFrame, decision: pd.DataFrame, input_audit: dict) -> None:
    lines = []
    lines.append("# W45 H feature-level progress audit v7_f")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Window: W002 / anchor day 45")
    lines.append("- Field: H / z500 only")
    lines.append("- Purpose: diagnose whether W45 H whole-field early-broad progress is caused by feature/latitude timing heterogeneity")
    lines.append("- This run does not infer causality and does not upgrade W45 whole-field pairwise order evidence.")
    lines.append("")
    if not whole_vs.empty:
        r = whole_vs.iloc[0]
        lines.append("## Whole-field vs feature-level summary")
        for k in ["whole_observed_midpoint", "whole_bootstrap_midpoint_median", "feature_midpoint_min", "feature_midpoint_max", "feature_midpoint_median", "feature_midpoint_iqr", "n_early_features", "n_middle_features", "n_late_features", "n_unstable_features", "interpretation"]:
            if k in r:
                lines.append(f"- {k}: {r.get(k)}")
        lines.append("")
    if not decision.empty:
        r = decision.iloc[0]
        lines.append("## Next-step decision")
        for k in ["is_worth_region_level_progress", "region_level_priority", "recommended_next_action", "decision_reason"]:
            lines.append(f"- {k}: {r.get(k)}")
        lines.append("")
    lines.append("## Field-feature timing classes")
    if not feature_summary.empty:
        counts = feature_summary["timing_class"].astype(str).value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Input audit")
    for k, v in input_audit.get("checks", {}).items():
        lines.append(f"- {k}: {v}")
    (paths.output_dir / "w45_H_feature_progress_summary_v7_f.md").write_text("\n".join(lines), encoding="utf-8")


def _input_audit(paths: W45HFeaturePaths, settings: StagePartitionV7Settings, window: dict) -> dict:
    return {
        "created_at": _now_iso(),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "input_paths": {
            "smoothed_fields": str(settings.foundation.smoothed_fields_path()),
            "v7e_output_dir": str(paths.v7e_output_dir),
        },
        "window": window,
        "checks": {
            "smoothed_fields_exists": settings.foundation.smoothed_fields_path().exists(),
            "v7e_output_dir_exists": paths.v7e_output_dir.exists(),
            "window_source": window.get("source", "unknown"),
        },
    }


def run_w45_H_feature_progress_v7_f(v7_root: Path | None = None) -> None:
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
        "notes": [
            "W45 H feature-level progress audit only.",
            "This run recomputes H feature-level progress from smoothed fields but does not alter V7-e whole-field progress outputs.",
            "No causal/pathway interpretation is made.",
        ],
    }
    _write_json(meta, paths.output_dir / "run_meta.json")

    print("[v7_f] loading smoothed fields and building profiles...")
    smoothed = load_smoothed_fields(settings.foundation.smoothed_fields_path())
    profiles = build_profiles(smoothed, settings.profile)
    n_days = int(np.asarray(profiles[FIELD].raw_cube).shape[1])
    window = _load_w45_window(paths, settings, n_days)
    input_audit = _input_audit(paths, settings, window)
    _write_json(input_audit, paths.output_dir / "input_audit_v7_f.json")

    print("[v7_f] building shared valid-day index...")
    shared_valid_day_index = None
    try:
        joint_state = build_state_matrix(profiles, settings.state)
        if isinstance(joint_state, dict) and "valid_day_index" in joint_state:
            shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)
    except Exception as exc:
        (paths.log_dir / "joint_valid_day_warning.txt").write_text(str(exc), encoding="utf-8")
        shared_valid_day_index = None

    h_lats = np.asarray(profiles[FIELD].lat_grid, dtype=float)
    print(f"[v7_f] H features: {len(h_lats)}")

    observed_matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(
        profiles,
        FIELD,
        None,
        standardize=settings.state.standardize,
        trim_invalid_days=settings.state.trim_invalid_days,
        shared_valid_day_index=shared_valid_day_index,
    )
    observed_full = _state_to_full_matrix(observed_matrix, valid_day_index, n_days)
    observed_rows = []
    curve_frames = []
    for j, latv in enumerate(h_lats):
        row, curves = _feature_progress(observed_full, j, float(latv), window, settings)
        observed_rows.append(row)
        if not curves.empty:
            curve_frames.append(curves)
    observed_df = pd.DataFrame(observed_rows)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    _write_csv(observed_df, paths.output_dir / "w45_H_feature_progress_observed_v7_f.csv")
    _write_csv(curves_df, paths.output_dir / "w45_H_feature_progress_curves_long_v7_f.csv")

    print("[v7_f] running bootstrap feature-level progress...")
    bootstrap_indices = _load_or_make_bootstrap_indices(profiles, paths, settings)
    boot_rows = []
    for b, year_idx in enumerate(bootstrap_indices):
        if b % int(settings.bootstrap.progress_every) == 0:
            print(f"[v7_f] bootstrap {b}/{len(bootstrap_indices)}")
        mat, vdi, _ = build_field_state_matrix_for_year_indices(
            profiles,
            FIELD,
            year_idx,
            standardize=settings.state.standardize,
            trim_invalid_days=settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index,
        )
        full = _state_to_full_matrix(mat, vdi, n_days)
        for j, latv in enumerate(h_lats):
            row, _ = _feature_progress(full, j, float(latv), window, settings, sample_col="bootstrap_id", sample_value=b)
            boot_rows.append(row)
    boot_df = pd.DataFrame(boot_rows)
    _write_csv(boot_df, paths.output_dir / "w45_H_feature_progress_bootstrap_samples_v7_f.csv")

    print("[v7_f] running leave-one-year-out feature-level progress...")
    n_years = int(np.asarray(profiles[FIELD].raw_cube).shape[0])
    loyo_rows = []
    for left_out in range(n_years):
        if left_out % 10 == 0:
            print(f"[v7_f] loyo {left_out}/{n_years}")
        keep = np.asarray([i for i in range(n_years) if i != left_out], dtype=int)
        mat, vdi, _ = build_field_state_matrix_for_year_indices(
            profiles,
            FIELD,
            keep,
            standardize=settings.state.standardize,
            trim_invalid_days=settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index,
        )
        full = _state_to_full_matrix(mat, vdi, n_days)
        for j, latv in enumerate(h_lats):
            row, _ = _feature_progress(full, j, float(latv), window, settings, sample_col="left_out_year_index", sample_value=left_out)
            loyo_rows.append(row)
    loyo_df = pd.DataFrame(loyo_rows)
    _write_csv(loyo_df, paths.output_dir / "w45_H_feature_progress_loyo_samples_v7_f.csv")

    print("[v7_f] summarizing feature-level progress...")
    boot_summary = _summarize_feature_samples(boot_df)
    boot_summary = _assign_timing_classes(boot_summary)
    _write_csv(boot_summary, paths.output_dir / "w45_H_feature_progress_bootstrap_summary_v7_f.csv")
    # Backward-compatible name requested in plan.
    _write_csv(boot_summary, paths.output_dir / "w45_H_feature_timing_spread_v7_f.csv")

    loyo_summary = _summarize_feature_samples(loyo_df)
    if not loyo_summary.empty:
        loyo_summary = _assign_timing_classes(loyo_summary)
    _write_csv(loyo_summary, paths.output_dir / "w45_H_feature_progress_loyo_summary_v7_f.csv")

    whole_vs = _build_whole_vs_feature(boot_summary, observed_df, paths)
    _write_csv(whole_vs, paths.output_dir / "w45_H_whole_vs_feature_progress_v7_f.csv")
    decision = _build_decision(boot_summary, whole_vs)
    _write_csv(decision, paths.output_dir / "w45_H_feature_next_step_decision_v7_f.csv")

    print("[v7_f] writing figures...")
    _plot_heatmap(curves_df, paths)
    _plot_midpoint_by_lat(boot_summary, paths)

    _write_summary_md(paths, observed_df, boot_summary, whole_vs, decision, input_audit)
    meta.update(
        {
            "status": "success",
            "finished_at": _now_iso(),
            "n_h_features": int(len(h_lats)),
            "n_bootstrap": int(len(bootstrap_indices)),
            "n_loyo": int(n_years),
            "output_dir": str(paths.output_dir),
            "log_dir": str(paths.log_dir),
            "core_outputs": [
                "w45_H_feature_progress_observed_v7_f.csv",
                "w45_H_feature_progress_curves_long_v7_f.csv",
                "w45_H_feature_progress_bootstrap_samples_v7_f.csv",
                "w45_H_feature_progress_bootstrap_summary_v7_f.csv",
                "w45_H_feature_timing_spread_v7_f.csv",
                "w45_H_whole_vs_feature_progress_v7_f.csv",
                "w45_H_feature_next_step_decision_v7_f.csv",
            ],
        }
    )
    _write_json(meta, paths.output_dir / "run_meta.json")
    (paths.log_dir / "w45_H_feature_progress_v7_f.md").write_text(
        "\n".join(
            [
                "# W45 H feature progress v7_f",
                "",
                f"status: {meta['status']}",
                f"created_at: {meta['created_at']}",
                f"finished_at: {meta['finished_at']}",
                f"n_h_features: {meta['n_h_features']}",
                f"n_bootstrap: {meta['n_bootstrap']}",
                "",
                "This is a W45/H-only feature-level progress diagnostic. It does not modify V7-e whole-field results and does not infer causality.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[v7_f] done: {paths.output_dir}")
