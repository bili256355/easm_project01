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
OUTPUT_TAG = "w45_H_coherent_region_progress_v7_h"


@dataclass
class W45HCoherentPaths:
    v7_root: Path
    project_root: Path
    v7e_output_dir: Path
    v7f_output_dir: Path
    v7g_output_dir: Path
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


def _resolve_paths(v7_root: Optional[Path]) -> W45HCoherentPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = v7_root.parents[1]
    return W45HCoherentPaths(
        v7_root=v7_root,
        project_root=project_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7f_output_dir=v7_root / "outputs" / "w45_H_feature_progress_v7_f",
        v7g_output_dir=v7_root / "outputs" / "w45_H_latband_progress_v7_g",
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _configure_settings(paths: W45HCoherentPaths) -> StagePartitionV7Settings:
    settings = StagePartitionV7Settings()
    settings.foundation.project_root = paths.project_root
    settings.source.project_root = paths.project_root
    settings.output.output_tag = OUTPUT_TAG
    return settings


def _load_w45_window(paths: W45HCoherentPaths, n_days: int) -> dict:
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


def _nanmean_vec(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.asarray(x, dtype=float), axis=0)


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


def _cfg_value(cfg: object, names: tuple[str, ...], default):
    """Return the first available config attribute from compatible V7-e naming variants."""
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    return default

def _progress_thresholds(cfg: object) -> tuple[float, float, float, int]:
    # V7-e config uses threshold_onset / threshold_midpoint / threshold_finish.
    # Some earlier drafts used onset_threshold / midpoint_threshold / finish_threshold.
    onset = float(_cfg_value(cfg, ("onset_threshold", "threshold_onset"), 0.25))
    midpoint = float(_cfg_value(cfg, ("midpoint_threshold", "threshold_midpoint"), 0.50))
    finish = float(_cfg_value(cfg, ("finish_threshold", "threshold_finish"), 0.75))
    stable_days = int(_cfg_value(cfg, ("stable_crossing_days",), 2))
    return onset, midpoint, finish, stable_days


def _progress_for_indices(
    full_state: np.ndarray,
    indices: np.ndarray,
    window: dict,
    settings: StagePartitionV7Settings,
    unit_meta: dict,
    sample_col: str | None = None,
    sample_value: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    cfg = settings.progress_timing
    idx = np.asarray(indices, dtype=int)
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
        "accepted_window_start": int(window["accepted_window_start"]),
        "accepted_window_end": int(window["accepted_window_end"]),
        "analysis_window_start": astart,
        "analysis_window_end": aend,
        "pre_period_start": pre_start,
        "pre_period_end": pre_end,
        "post_period_start": post_start,
        "post_period_end": post_end,
        **unit_meta,
    }
    if sample_col is not None:
        base[sample_col] = int(sample_value) if sample_value is not None else np.nan

    pre = x[pre_start : pre_end + 1, :]
    post = x[post_start : post_end + 1, :]
    pre_proto = _nanmean_vec(pre)
    post_proto = _nanmean_vec(post)
    transition = post_proto - pre_proto
    denom = float(np.nansum(transition ** 2))

    pre_d = np.array([_row_norm(row - pre_proto) for row in pre], dtype=float)
    post_d = np.array([_row_norm(row - post_proto) for row in post], dtype=float)
    pre_var = float(np.nanmean(pre_d)) if np.isfinite(pre_d).any() else np.nan
    post_var = float(np.nanmean(post_d)) if np.isfinite(post_d).any() else np.nan
    pre_post_distance = _row_norm(post_proto - pre_proto)
    denom_var = pre_var + post_var
    separation_ratio = float(pre_post_distance / denom_var) if np.isfinite(pre_post_distance) and np.isfinite(denom_var) and denom_var > 0 else np.nan
    sep_label = _separation_label(separation_ratio, settings)

    days = np.arange(astart, aend + 1, dtype=int)
    progress_vals = []
    raw_vals = []
    for d in days:
        row = x[d, :]
        if denom <= 0 or not np.isfinite(denom) or not np.all(np.isfinite(row)):
            raw = np.nan
        else:
            raw = float(np.nansum((row - pre_proto) * transition) / denom)
        raw_vals.append(raw)
        progress_vals.append(float(np.clip(raw, 0.0, 1.0)) if np.isfinite(raw) else np.nan)
    raw_vals = np.asarray(raw_vals, dtype=float)
    progress_vals = np.asarray(progress_vals, dtype=float)
    onset_thr, midpoint_thr, finish_thr, stable_days = _progress_thresholds(cfg)
    onset = _first_stable_crossing(days, progress_vals, onset_thr, stable_days)
    mid = _first_stable_crossing(days, progress_vals, midpoint_thr, stable_days)
    finish = _first_stable_crossing(days, progress_vals, finish_thr, stable_days)
    duration = float(finish - onset + 1) if np.isfinite(onset) and np.isfinite(finish) else np.nan

    c25 = _count_crossings(progress_vals, onset_thr)
    c50 = _count_crossings(progress_vals, midpoint_thr)
    c75 = _count_crossings(progress_vals, finish_thr)
    if np.isfinite(progress_vals).sum() >= 3:
        try:
            corr = float(np.corrcoef(days[np.isfinite(progress_vals)], progress_vals[np.isfinite(progress_vals)])[0, 1])
        except Exception:
            corr = np.nan
    else:
        corr = np.nan
    excess_crossings = max(c25 - 1, 0) + max(c50 - 1, 0) + max(c75 - 1, 0)
    low_corr = np.isfinite(corr) and corr < float(cfg.monotonic_corr_threshold)
    if sep_label == "no_clear_separation":
        quality = "no_clear_prepost_separation"
    elif not np.isfinite(mid):
        quality = "partial_progress"
    elif low_corr or excess_crossings >= 1:
        quality = "nonmonotonic_progress"
    elif np.isfinite(duration) and duration >= 0.5 * (aend - astart + 1):
        quality = "monotonic_broad_progress"
    else:
        quality = "monotonic_clear_progress"

    row = {
        **base,
        "pre_post_distance": pre_post_distance,
        "within_pre_variability": pre_var,
        "within_post_variability": post_var,
        "separation_ratio": separation_ratio,
        "pre_post_separation_label": sep_label,
        "observed_onset_day" if sample_col is None else "onset_day": onset,
        "observed_midpoint_day" if sample_col is None else "midpoint_day": mid,
        "observed_finish_day" if sample_col is None else "finish_day": finish,
        "observed_duration" if sample_col is None else "duration": duration,
        "progress_monotonicity_corr": corr,
        "n_crossings_025": c25,
        "n_crossings_050": c50,
        "n_crossings_075": c75,
        "excess_crossings": int(excess_crossings),
        "progress_quality_label": quality,
    }
    curves = pd.DataFrame([{**base, "day": int(d), "raw_progress": raw, "progress": val} for d, raw, val in zip(days, raw_vals, progress_vals)])
    return row, curves


def _load_or_make_bootstrap_indices(profiles: dict, paths: W45HCoherentPaths, settings: StagePartitionV7Settings) -> list[np.ndarray]:
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


def _feature_transition_vector(full_state: np.ndarray, window: dict, lat_values: np.ndarray, sample_col: str | None = None, sample_value: int | None = None) -> pd.DataFrame:
    pre_start = int(window["pre_period_start"])
    pre_end = int(window["pre_period_end"])
    post_start = int(window["post_period_start"])
    post_end = int(window["post_period_end"])
    pre = _nanmean_vec(full_state[pre_start : pre_end + 1, :])
    post = _nanmean_vec(full_state[post_start : post_end + 1, :])
    dH = post - pre
    total_abs = float(np.nansum(np.abs(dH)))
    rows = []
    for i, lat in enumerate(lat_values):
        row = {
            "window_id": WINDOW_ID,
            "anchor_day": ANCHOR_DAY,
            "field": FIELD,
            "feature_index": int(i),
            "lat": float(lat),
            "H_pre": float(pre[i]) if np.isfinite(pre[i]) else np.nan,
            "H_post": float(post[i]) if np.isfinite(post[i]) else np.nan,
            "dH": float(dH[i]) if np.isfinite(dH[i]) else np.nan,
            "abs_dH": float(abs(dH[i])) if np.isfinite(dH[i]) else np.nan,
            "relative_contribution": float(abs(dH[i]) / total_abs) if np.isfinite(dH[i]) and total_abs > 0 else np.nan,
        }
        if sample_col is not None:
            row[sample_col] = int(sample_value) if sample_value is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_transition_vector(obs: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in obs.sort_values("feature_index").iterrows():
        fi = int(r["feature_index"])
        sub = boot[boot["feature_index"] == fi]
        vals = pd.to_numeric(sub["dH"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            q05 = float(np.nanpercentile(vals, 5))
            q95 = float(np.nanpercentile(vals, 95))
            ppos = float(np.nanmean(vals > 0))
            pneg = float(np.nanmean(vals < 0))
        else:
            q05 = q95 = ppos = pneg = np.nan
        if np.isfinite(q05) and q05 > 0:
            cls = "stable_positive_change"
        elif np.isfinite(q95) and q95 < 0:
            cls = "stable_negative_change"
        else:
            cls = "ambiguous_change"
        sign_consistency = float(max(ppos, pneg)) if np.isfinite(ppos) and np.isfinite(pneg) else np.nan
        rows.append({
            **r.to_dict(),
            "bootstrap_dH_q05": q05,
            "bootstrap_dH_q95": q95,
            "prob_dH_gt_0": ppos,
            "prob_dH_lt_0": pneg,
            "sign_consistency": sign_consistency,
            "feature_change_class": cls,
        })
    return pd.DataFrame(rows)


def _build_coherent_regions(vec_summary: pd.DataFrame) -> list[dict]:
    df = vec_summary.sort_values("feature_index").reset_index(drop=True)
    regions: list[dict] = []
    if df.empty:
        return regions
    region_id = 0
    start = 0
    current = str(df.loc[0, "feature_change_class"])
    for pos in range(1, len(df) + 1):
        end_group = pos == len(df) or str(df.loc[pos, "feature_change_class"]) != current
        if end_group:
            sub = df.iloc[start:pos]
            idx = sub["feature_index"].astype(int).to_numpy()
            lats = sub["lat"].astype(float).to_numpy()
            contrib = float(pd.to_numeric(sub["relative_contribution"], errors="coerce").sum())
            lat_min = float(np.nanmin(lats))
            lat_max = float(np.nanmax(lats))
            label = f"H_region_{region_id}_{current}_{lat_min:g}_{lat_max:g}"
            regions.append({
                "region_id": int(region_id),
                "region_label": label,
                "feature_indices": idx,
                "feature_indices_str": ";".join(str(int(i)) for i in idx),
                "lat_list": ";".join(f"{float(v):g}" for v in lats),
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lat_center": float(np.nanmean(lats)),
                "n_features": int(len(idx)),
                "region_change_class": current,
                "region_relative_contribution": contrib,
                "construction_reason": "contiguous latitude features with same bootstrap dH 90% sign class",
            })
            region_id += 1
            if pos < len(df):
                start = pos
                current = str(df.loc[pos, "feature_change_class"])
    return regions


def _region_defs_to_df(regions: list[dict]) -> pd.DataFrame:
    rows = []
    for reg in regions:
        rows.append({k: v for k, v in reg.items() if k != "feature_indices"})
    return pd.DataFrame(rows)


def _summarize_region_samples(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if samples.empty:
        return pd.DataFrame()
    group_cols = ["region_id", "region_label", "lat_min", "lat_max", "lat_center", "n_features", "region_change_class", "region_relative_contribution"]
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
                "midpoint_iqr": float(np.nanpercentile(mids, 75) - np.nanpercentile(mids, 25)),
                "midpoint_q90_width": float(np.nanpercentile(mids, 95) - np.nanpercentile(mids, 5)),
                "finish_median": float(np.nanmedian(finishes)) if finishes.size else np.nan,
                "finish_q05": float(np.nanpercentile(finishes, 5)) if finishes.size else np.nan,
                "finish_q95": float(np.nanpercentile(finishes, 95)) if finishes.size else np.nan,
                "duration_median": float(np.nanmedian(durations)) if durations.size else np.nan,
                "duration_q05": float(np.nanpercentile(durations, 5)) if durations.size else np.nan,
                "duration_q95": float(np.nanpercentile(durations, 95)) if durations.size else np.nan,
            })
        else:
            row.update({"onset_median": np.nan, "onset_q05": np.nan, "onset_q95": np.nan, "midpoint_median": np.nan, "midpoint_q05": np.nan, "midpoint_q95": np.nan, "midpoint_iqr": np.nan, "midpoint_q90_width": np.nan, "finish_median": np.nan, "finish_q05": np.nan, "finish_q95": np.nan, "duration_median": np.nan, "duration_q05": np.nan, "duration_q95": np.nan})
        row.update({
            "dominant_progress_quality_label": max(qual_counts, key=qual_counts.get) if qual_counts else "none",
            "dominant_prepost_separation_label": max(sep_counts, key=sep_counts.get) if sep_counts else "none",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _load_comparison_unit(path: Path, unit_type: str, mapping: dict) -> pd.DataFrame:
    df = _read_csv(path, required=False)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    for out_col, in_col in mapping.items():
        out[out_col] = df[in_col] if in_col in df.columns else np.nan
    out["unit_type"] = unit_type
    return out


def _build_unit_comparison(paths: W45HCoherentPaths, region_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    whole = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_summary_v7_e.csv", required=False)
    if not whole.empty:
        sub = whole[(whole.get("window_id", "").astype(str) == WINDOW_ID) & (whole.get("field", "").astype(str) == FIELD)]
        for _, r in sub.iterrows():
            rows.append({
                "unit_type": "whole_field_H",
                "unit_label": "H_whole_field",
                "n_units": 1,
                "lat_min": np.nan,
                "lat_max": np.nan,
                "lat_center": np.nan,
                "midpoint_median": r.get("midpoint_median", np.nan),
                "midpoint_q05": r.get("midpoint_q05", r.get("midpoint_q025", np.nan)),
                "midpoint_q95": r.get("midpoint_q95", r.get("midpoint_q975", np.nan)),
                "midpoint_q90_width": (r.get("midpoint_q95", r.get("midpoint_q975", np.nan)) - r.get("midpoint_q05", r.get("midpoint_q025", np.nan))) if pd.notna(r.get("midpoint_q95", r.get("midpoint_q975", np.nan))) and pd.notna(r.get("midpoint_q05", r.get("midpoint_q025", np.nan))) else np.nan,
                "progress_quality_dominant": r.get("dominant_progress_quality_label", np.nan),
                "interpretation": "whole-field H reference from V7-e",
            })
    feat = _read_csv(paths.v7f_output_dir / "w45_H_feature_progress_bootstrap_summary_v7_f.csv", required=False)
    if not feat.empty:
        for _, r in feat.iterrows():
            rows.append({
                "unit_type": "single_lat_feature",
                "unit_label": f"H_feature_{r.get('lat_value', r.get('lat', ''))}",
                "n_units": 1,
                "lat_min": r.get("lat_value", r.get("lat", np.nan)),
                "lat_max": r.get("lat_value", r.get("lat", np.nan)),
                "lat_center": r.get("lat_value", r.get("lat", np.nan)),
                "midpoint_median": r.get("midpoint_median", np.nan),
                "midpoint_q05": r.get("midpoint_q05", np.nan),
                "midpoint_q95": r.get("midpoint_q95", np.nan),
                "midpoint_q90_width": r.get("midpoint_q90_width", np.nan),
                "progress_quality_dominant": r.get("dominant_progress_quality_label", np.nan),
                "interpretation": "single-lat feature reference from V7-f",
            })
    latband = _read_csv(paths.v7g_output_dir / "w45_H_latband_progress_bootstrap_summary_v7_g.csv", required=False)
    if not latband.empty:
        for _, r in latband.iterrows():
            rows.append({
                "unit_type": "local_3_lat_band",
                "unit_label": r.get("band_label", ""),
                "n_units": 1,
                "lat_min": r.get("lat_min", np.nan),
                "lat_max": r.get("lat_max", np.nan),
                "lat_center": r.get("lat_center", np.nan),
                "midpoint_median": r.get("midpoint_median", np.nan),
                "midpoint_q05": r.get("midpoint_q05", np.nan),
                "midpoint_q95": r.get("midpoint_q95", np.nan),
                "midpoint_q90_width": r.get("midpoint_q90_width", np.nan),
                "progress_quality_dominant": r.get("dominant_progress_quality_label", np.nan),
                "interpretation": "local 3-lat-band reference from V7-g",
            })
    for _, r in region_summary.iterrows():
        rows.append({
            "unit_type": "coherent_region",
            "unit_label": r.get("region_label", ""),
            "n_units": 1,
            "lat_min": r.get("lat_min", np.nan),
            "lat_max": r.get("lat_max", np.nan),
            "lat_center": r.get("lat_center", np.nan),
            "midpoint_median": r.get("midpoint_median", np.nan),
            "midpoint_q05": r.get("midpoint_q05", np.nan),
            "midpoint_q95": r.get("midpoint_q95", np.nan),
            "midpoint_q90_width": r.get("midpoint_q90_width", np.nan),
            "progress_quality_dominant": r.get("dominant_progress_quality_label", np.nan),
            "interpretation": "transition-coherent region result from V7-h",
        })
    return pd.DataFrame(rows)


def _summarize_unit_types(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if comparison.empty:
        return pd.DataFrame()
    for unit_type, sub in comparison.groupby("unit_type", sort=False):
        widths = pd.to_numeric(sub["midpoint_q90_width"], errors="coerce").dropna().to_numpy(dtype=float)
        qual = sub["progress_quality_dominant"].astype(str) if "progress_quality_dominant" in sub.columns else pd.Series([], dtype=str)
        n_unstable = int(qual.str.contains("nonmonotonic|unavailable|no_clear|unstable", regex=True).sum()) if len(qual) else 0
        rows.append({
            "unit_type": unit_type,
            "n_units": int(len(sub)),
            "median_q90_width": float(np.nanmedian(widths)) if widths.size else np.nan,
            "mean_q90_width": float(np.nanmean(widths)) if widths.size else np.nan,
            "n_unstable_units": n_unstable,
            "n_valid_units": int(len(sub) - n_unstable),
        })
    return pd.DataFrame(rows)


def _build_upstream_implication(unit_summary: pd.DataFrame, region_summary: pd.DataFrame, region_defs: pd.DataFrame, vec_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def val(unit_type: str, col: str) -> float:
        sub = unit_summary[unit_summary["unit_type"] == unit_type]
        if sub.empty or col not in sub.columns:
            return np.nan
        return float(sub.iloc[0][col]) if pd.notna(sub.iloc[0][col]) else np.nan
    feat_width = val("single_lat_feature", "median_q90_width")
    band_width = val("local_3_lat_band", "median_q90_width")
    reg_width = val("coherent_region", "median_q90_width")
    feat_unstable = val("single_lat_feature", "n_unstable_units")
    band_unstable = val("local_3_lat_band", "n_unstable_units")
    reg_unstable = val("coherent_region", "n_unstable_units")
    n_stable_features = int(vec_summary["feature_change_class"].astype(str).str.startswith("stable_").sum()) if not vec_summary.empty else 0
    n_regions = int(len(region_defs))
    coherent_improves = np.isfinite(reg_width) and ((np.isfinite(feat_width) and reg_width < feat_width) or (np.isfinite(band_width) and reg_width < band_width))
    quality_improves = np.isfinite(reg_unstable) and ((np.isfinite(feat_unstable) and reg_unstable < feat_unstable) or (np.isfinite(band_unstable) and reg_unstable < band_unstable))
    if coherent_improves and quality_improves:
        status = "yes"
        implication = "transition-coherent region improves timing stability relative to single-feature/sliding-band units; future region-level progress should first audit transition vectors and construct coherent regions."
        followup = "promote coherent-region unit design to the next region-level progress prototype."
    elif coherent_improves or quality_improves:
        status = "partial"
        implication = "transition-coherent region partially improves stability, but not enough for automatic promotion."
        followup = "inspect region definitions and quality before generalizing to other fields/windows."
    else:
        status = "no"
        implication = "coherent regions do not resolve W45-H instability; W45-H may remain early-broad candidate or progress definition may be inadequate."
        followup = "do not upgrade W45-H regional order; reconsider progress definition or retain candidate only."
    evidence = f"median_q90_width(feature={feat_width}, latband={band_width}, coherent={reg_width}); unstable_units(feature={feat_unstable}, latband={band_unstable}, coherent={reg_unstable}); stable_dH_features={n_stable_features}; n_regions={n_regions}"
    rows.append({"diagnostic_item": "coherent_region_improves_stability", "status": status, "evidence": evidence, "upstream_implication": implication, "recommended_followup": followup})
    rows.append({"diagnostic_item": "transition_vector_has_stable_structure", "status": "yes" if n_stable_features > 0 else "no", "evidence": f"stable_dH_features={n_stable_features}; n_regions={n_regions}", "upstream_implication": "Region construction has data support only where dH sign is stable under bootstrap." if n_stable_features > 0 else "No stable pre-post H feature-change structure was found.", "recommended_followup": "Use stable dH segments as candidate regions only if region progress is also stable."})
    rows.append({"diagnostic_item": "upstream_analysis_unit_should_be_revised", "status": "yes" if status in {"yes", "partial"} else "unconfirmed", "evidence": evidence, "upstream_implication": "If coherent units improve stability, future P/V/H/Je/Jw region progress should not default to single features or mechanical sliding bands.", "recommended_followup": followup})
    return pd.DataFrame(rows)


def _plot_transition_vector(vec_summary: pd.DataFrame, paths: W45HCoherentPaths) -> None:
    if vec_summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
        df = vec_summary.sort_values("lat")
        x = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["dH"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(df["bootstrap_dH_q05"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df["bootstrap_dH_q95"], errors="coerce").to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(y - lo, 0), np.maximum(hi - y, 0)])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("H latitude feature")
        ax.set_ylabel("dH = post - pre")
        ax.set_title("W45 H transition vector by latitude")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_transition_vector_by_lat_v7_h.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (paths.log_dir / "plot_transition_vector_error.txt").write_text(str(exc), encoding="utf-8")


def _plot_region_heatmap(curves: pd.DataFrame, paths: W45HCoherentPaths) -> None:
    if curves.empty:
        return
    try:
        import matplotlib.pyplot as plt
        piv = curves.pivot_table(index="region_label", columns="day", values="progress", aggfunc="first")
        centers = curves.drop_duplicates("region_label").set_index("region_label")["lat_center"].to_dict()
        ordered = sorted(piv.index, key=lambda x: centers.get(x, np.nan))
        piv = piv.loc[ordered]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", origin="lower", extent=[piv.columns.min(), piv.columns.max(), -0.5, len(piv.index)-0.5])
        ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("day index")
        ax.set_ylabel("coherent H region")
        ax.set_title("W45 H coherent-region progress")
        fig.colorbar(im, ax=ax, label="progress")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_coherent_region_progress_heatmap_v7_h.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (paths.log_dir / "plot_region_heatmap_error.txt").write_text(str(exc), encoding="utf-8")


def _plot_unit_stability(unit_summary: pd.DataFrame, paths: W45HCoherentPaths) -> None:
    if unit_summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
        df = unit_summary.copy()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(df))
        y = pd.to_numeric(df["median_q90_width"], errors="coerce").to_numpy(dtype=float)
        ax.bar(x, y)
        ax.set_xticks(x)
        ax.set_xticklabels(df["unit_type"].astype(str).to_list(), rotation=25, ha="right")
        ax.set_ylabel("median q90 width")
        ax.set_title("W45 H unit stability comparison")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_H_unit_stability_comparison_v7_h.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (paths.log_dir / "plot_unit_stability_error.txt").write_text(str(exc), encoding="utf-8")


def _write_summary_md(paths: W45HCoherentPaths, vec_summary: pd.DataFrame, region_defs: pd.DataFrame, region_summary: pd.DataFrame, unit_summary: pd.DataFrame, upstream: pd.DataFrame) -> None:
    lines = []
    lines.append("# W45 H transition-coherent region progress audit v7_h")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- Diagnose whether W45/H whole-field early-broad behavior can be explained by transition-coherent latitude regions.")
    lines.append("- Regions are built from contiguous H latitude features with the same bootstrap 90% dH sign class.")
    lines.append("- This is a W45/H-only implementation diagnostic. It does not infer causality or modify V7-e/V7-e1/V7-e2 outputs.")
    lines.append("")
    lines.append("## Transition-vector classes")
    if not vec_summary.empty:
        for k, v in vec_summary["feature_change_class"].astype(str).value_counts().to_dict().items():
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Coherent regions")
    if not region_defs.empty:
        for _, r in region_defs.iterrows():
            lines.append(f"- {r.get('region_label')}: lat {r.get('lat_min')}–{r.get('lat_max')}, class={r.get('region_change_class')}, n_features={r.get('n_features')}")
    lines.append("")
    lines.append("## Unit comparison")
    if not unit_summary.empty:
        for _, r in unit_summary.iterrows():
            lines.append(f"- {r.get('unit_type')}: n={r.get('n_units')}, median_q90_width={r.get('median_q90_width')}, unstable={r.get('n_unstable_units')}")
    lines.append("")
    lines.append("## Upstream implication")
    if not upstream.empty:
        for _, r in upstream.iterrows():
            lines.append(f"- {r.get('diagnostic_item')}: {r.get('status')} | {r.get('upstream_implication')}")
    lines.append("")
    lines.append("## Interpretation rule")
    lines.append("- Stable dH regions and improved region progress can support revising future region-level progress units.")
    lines.append("- If coherent regions remain unstable, W45-H should remain an early-broad candidate, not a confirmed regional timing result.")
    (paths.output_dir / "w45_H_coherent_region_progress_summary_v7_h.md").write_text("\n".join(lines), encoding="utf-8")


def run_w45_H_coherent_region_progress_v7_h(v7_root: Path | None = None) -> None:
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
        "notes": [
            "W45/H-only transition-coherent region progress diagnostic.",
            "Coherent regions are built from contiguous latitude features with the same bootstrap dH 90% sign class.",
            "This run does not modify V7-e/V7-e1/V7-e2 outputs and does not infer causality.",
        ],
    }
    _write_json(meta, paths.output_dir / "run_meta.json")

    print("[v7_h] loading smoothed fields and H profiles...")
    smoothed = load_smoothed_fields(settings.foundation.smoothed_fields_path())
    profiles = build_profiles(smoothed, settings.profile)
    n_days = int(np.asarray(profiles[FIELD].raw_cube).shape[1])
    window = _load_w45_window(paths, n_days)
    input_audit = {
        "created_at": _now_iso(),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "checks": {
            "smoothed_fields_exists": settings.foundation.smoothed_fields_path().exists(),
            "v7e_output_dir_exists": paths.v7e_output_dir.exists(),
            "v7f_output_dir_exists": paths.v7f_output_dir.exists(),
            "v7g_output_dir_exists": paths.v7g_output_dir.exists(),
            "window_source": window.get("source", "unknown"),
        },
        "window": window,
    }
    _write_json(input_audit, paths.output_dir / "input_audit_v7_h.json")

    print("[v7_h] building shared valid-day index...")
    shared_valid_day_index = None
    try:
        joint_state = build_state_matrix(profiles, settings.state)
        if isinstance(joint_state, dict) and "valid_day_index" in joint_state:
            shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)
    except Exception as exc:
        (paths.log_dir / "joint_valid_day_warning.txt").write_text(str(exc), encoding="utf-8")

    h_lats = np.asarray(profiles[FIELD].lat_grid, dtype=float)
    observed_matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(
        profiles,
        FIELD,
        None,
        standardize=settings.state.standardize,
        trim_invalid_days=settings.state.trim_invalid_days,
        shared_valid_day_index=shared_valid_day_index,
    )
    observed_full = _state_to_full_matrix(observed_matrix, valid_day_index, n_days)
    obs_vec = _feature_transition_vector(observed_full, window, h_lats)
    _write_csv(obs_vec, paths.output_dir / "w45_H_transition_vector_observed_v7_h.csv")

    print("[v7_h] bootstrapping H transition vector...")
    bootstrap_indices = _load_or_make_bootstrap_indices(profiles, paths, settings)
    boot_vec_rows: list[pd.DataFrame] = []
    for b, year_idx in enumerate(bootstrap_indices):
        if b % int(settings.bootstrap.progress_every) == 0:
            print(f"[v7_h] transition-vector bootstrap {b}/{len(bootstrap_indices)}")
        mat, vdi, _ = build_field_state_matrix_for_year_indices(
            profiles,
            FIELD,
            year_idx,
            standardize=settings.state.standardize,
            trim_invalid_days=settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index,
        )
        full = _state_to_full_matrix(mat, vdi, n_days)
        boot_vec_rows.append(_feature_transition_vector(full, window, h_lats, sample_col="bootstrap_id", sample_value=b))
    boot_vec = pd.concat(boot_vec_rows, ignore_index=True) if boot_vec_rows else pd.DataFrame()
    _write_csv(boot_vec, paths.output_dir / "w45_H_transition_vector_bootstrap_samples_v7_h.csv")
    vec_summary = _summarize_transition_vector(obs_vec, boot_vec)
    _write_csv(vec_summary, paths.output_dir / "w45_H_transition_vector_audit_v7_h.csv")

    regions = _build_coherent_regions(vec_summary)
    region_defs = _region_defs_to_df(regions)
    _write_csv(region_defs, paths.output_dir / "w45_H_coherent_regions_v7_h.csv")
    print(f"[v7_h] coherent regions: {len(regions)}")

    print("[v7_h] observed coherent-region progress...")
    obs_rows = []
    curve_frames = []
    for reg in regions:
        meta_reg = {k: v for k, v in reg.items() if k != "feature_indices"}
        row, curves = _progress_for_indices(observed_full, reg["feature_indices"], window, settings, meta_reg)
        obs_rows.append(row)
        curve_frames.append(curves)
    obs_region = pd.DataFrame(obs_rows)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    _write_csv(obs_region, paths.output_dir / "w45_H_coherent_region_progress_observed_v7_h.csv")
    _write_csv(curves_df, paths.output_dir / "w45_H_coherent_region_progress_curves_long_v7_h.csv")

    print("[v7_h] bootstrap coherent-region progress...")
    region_boot_rows = []
    for b, year_idx in enumerate(bootstrap_indices):
        if b % int(settings.bootstrap.progress_every) == 0:
            print(f"[v7_h] region-progress bootstrap {b}/{len(bootstrap_indices)}")
        mat, vdi, _ = build_field_state_matrix_for_year_indices(
            profiles,
            FIELD,
            year_idx,
            standardize=settings.state.standardize,
            trim_invalid_days=settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index,
        )
        full = _state_to_full_matrix(mat, vdi, n_days)
        for reg in regions:
            meta_reg = {k: v for k, v in reg.items() if k != "feature_indices"}
            row, _ = _progress_for_indices(full, reg["feature_indices"], window, settings, meta_reg, sample_col="bootstrap_id", sample_value=b)
            region_boot_rows.append(row)
    region_boot = pd.DataFrame(region_boot_rows)
    _write_csv(region_boot, paths.output_dir / "w45_H_coherent_region_progress_bootstrap_samples_v7_h.csv")
    region_summary = _summarize_region_samples(region_boot)
    _write_csv(region_summary, paths.output_dir / "w45_H_coherent_region_progress_v7_h.csv")

    comparison = _build_unit_comparison(paths, region_summary)
    _write_csv(comparison, paths.output_dir / "w45_H_unit_comparison_v7_h.csv")
    unit_summary = _summarize_unit_types(comparison)
    _write_csv(unit_summary, paths.output_dir / "w45_H_unit_stability_summary_v7_h.csv")
    upstream = _build_upstream_implication(unit_summary, region_summary, region_defs, vec_summary)
    _write_csv(upstream, paths.output_dir / "w45_H_coherent_region_upstream_implication_v7_h.csv")

    print("[v7_h] writing figures...")
    _plot_transition_vector(vec_summary, paths)
    _plot_region_heatmap(curves_df, paths)
    _plot_unit_stability(unit_summary, paths)
    _write_summary_md(paths, vec_summary, region_defs, region_summary, unit_summary, upstream)

    meta.update({
        "status": "success",
        "finished_at": _now_iso(),
        "n_h_features": int(len(h_lats)),
        "n_bootstrap": int(len(bootstrap_indices)),
        "n_coherent_regions": int(len(regions)),
        "output_dir": str(paths.output_dir),
        "log_dir": str(paths.log_dir),
        "core_outputs": [
            "w45_H_transition_vector_audit_v7_h.csv",
            "w45_H_coherent_regions_v7_h.csv",
            "w45_H_coherent_region_progress_v7_h.csv",
            "w45_H_unit_comparison_v7_h.csv",
            "w45_H_unit_stability_summary_v7_h.csv",
            "w45_H_coherent_region_upstream_implication_v7_h.csv",
            "w45_H_coherent_region_progress_summary_v7_h.md",
        ],
    })
    _write_json(meta, paths.output_dir / "run_meta.json")
    (paths.log_dir / "w45_H_coherent_region_progress_v7_h.md").write_text(
        "\n".join([
            "# W45 H coherent-region progress v7_h",
            "",
            f"status: {meta['status']}",
            f"created_at: {meta['created_at']}",
            f"finished_at: {meta['finished_at']}",
            f"n_h_features: {meta['n_h_features']}",
            f"n_coherent_regions: {meta['n_coherent_regions']}",
            f"n_bootstrap: {meta['n_bootstrap']}",
            "",
            "This is a W45/H-only transition-coherent region diagnostic. It does not modify V7-e whole-field results and does not infer causality.",
        ]),
        encoding="utf-8",
    )
    print(f"[v7_h] done: {paths.output_dir}")
