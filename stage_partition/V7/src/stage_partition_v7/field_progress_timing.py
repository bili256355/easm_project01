from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from stage_partition_v6.io import load_smoothed_fields
from stage_partition_v6.state_builder import build_profiles, build_state_matrix
from stage_partition_v6.timeline import day_index_to_month_day

from .config import StagePartitionV7Settings
from .field_state import FIELDS, build_field_state_matrix_for_year_indices
from .field_timing import _audit_accepted_points
from .report import now_utc, write_dataframe, write_json


def _prepare_dirs(settings: StagePartitionV7Settings) -> dict[str, Path]:
    settings.output.output_tag = settings.progress_timing.output_tag
    out = settings.output_root()
    log_dir = settings.log_root()
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"output_root": out, "log_root": log_dir, "figure_root": fig_dir}


def _state_to_full_matrix(matrix: np.ndarray, valid_day_index: np.ndarray, n_days: int) -> np.ndarray:
    full = np.full((int(n_days), matrix.shape[1]), np.nan, dtype=float)
    valid_day_index = np.asarray(valid_day_index, dtype=int)
    full[valid_day_index, :] = np.asarray(matrix, dtype=float)
    return full


def _row_finite_mask(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.all(np.isfinite(x), axis=1)


def _nanmean_rows(x: np.ndarray) -> np.ndarray | None:
    if x.size == 0:
        return None
    good = _row_finite_mask(x)
    if int(good.sum()) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x[good, :], axis=0)


def _norm(x: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.sqrt(np.nanmean(np.square(np.asarray(x, dtype=float)))))


def _build_progress_windows(windows_df: pd.DataFrame, settings: StagePartitionV7Settings, n_days: int) -> pd.DataFrame:
    rows = []
    cfg = settings.progress_timing
    accepted = set(int(x) for x in settings.accepted_windows.accepted_peak_days)
    excluded = [int(x) for x in settings.accepted_windows.excluded_candidate_days]
    for _, r in windows_df.sort_values("main_peak_day").iterrows():
        anchor = int(r["main_peak_day"])
        if anchor not in accepted:
            continue
        astart = int(r["start_day"])
        aend = int(r["end_day"])
        analysis_start = max(0, anchor - int(cfg.analysis_radius_days))
        analysis_end = min(int(n_days) - 1, anchor + int(cfg.analysis_radius_days))
        pre_start = max(0, anchor - int(cfg.pre_offset_start_days))
        pre_end = max(0, anchor - int(cfg.pre_offset_end_days))
        post_start = min(int(n_days) - 1, anchor + int(cfg.post_offset_start_days))
        post_end = min(int(n_days) - 1, anchor + int(cfg.post_offset_end_days))
        nearby = [x for x in excluded if analysis_start - int(cfg.excluded_candidate_margin_days) <= x <= analysis_end + int(cfg.excluded_candidate_margin_days)]
        rows.append(
            {
                "window_id": str(r.get("window_id", f"W{anchor:03d}")),
                "anchor_day": anchor,
                "anchor_month_day": day_index_to_month_day(anchor),
                "accepted_window_start": astart,
                "accepted_window_end": aend,
                "analysis_window_start": analysis_start,
                "analysis_window_end": analysis_end,
                "pre_period_start": pre_start,
                "pre_period_end": pre_end,
                "post_period_start": post_start,
                "post_period_end": post_end,
                "analysis_radius_days": int(cfg.analysis_radius_days),
                "nearby_excluded_candidate_days": ";".join(str(x) for x in nearby) if nearby else "none",
                "n_nearby_excluded_candidates": int(len(nearby)),
                "source_window_id": str(r.get("window_id", f"W{anchor:03d}")),
                "note": "V7-e progress window: accepted anchor +/- radius; excluded candidates flagged only, not reintroduced",
            }
        )
    return pd.DataFrame(rows)


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
    good = np.isfinite(vals)
    if int(good.sum()) < 2:
        return 0
    above = vals[good] >= float(threshold)
    if above.size < 2:
        return 0
    return int(np.sum(above[1:] != above[:-1]))


def _first_stable_crossing(days: np.ndarray, vals: np.ndarray, threshold: float, stable_days: int) -> float:
    stable_days = max(1, int(stable_days))
    days = np.asarray(days, dtype=int)
    vals = np.asarray(vals, dtype=float)
    for i in range(0, len(vals) - stable_days + 1):
        win = vals[i : i + stable_days]
        if np.all(np.isfinite(win)) and np.all(win >= float(threshold)):
            return float(days[i])
    return np.nan


def _near_excluded_days(days: list[float], settings: StagePartitionV7Settings) -> str:
    margin = int(settings.progress_timing.excluded_candidate_margin_days)
    hits = []
    clean = [int(round(x)) for x in days if np.isfinite(x)]
    for c in settings.accepted_windows.excluded_candidate_days:
        c = int(c)
        if any(abs(d - c) <= margin for d in clean):
            hits.append(c)
    return ";".join(str(x) for x in sorted(set(hits))) if hits else "none"


def _compute_progress_for_window(matrix: np.ndarray, valid_day_index: np.ndarray, n_days: int, window: pd.Series | dict, field: str, settings: StagePartitionV7Settings, sample_col: str | None = None, sample_value: int | None = None) -> tuple[dict, pd.DataFrame]:
    cfg = settings.progress_timing
    full = _state_to_full_matrix(matrix, valid_day_index, n_days)
    start = int(window["analysis_window_start"])
    end = int(window["analysis_window_end"])
    pre_start = int(window["pre_period_start"])
    pre_end = int(window["pre_period_end"])
    post_start = int(window["post_period_start"])
    post_end = int(window["post_period_end"])
    anchor = int(window["anchor_day"])
    window_id = str(window["window_id"])
    base = {
        "window_id": window_id,
        "anchor_day": anchor,
        "accepted_window_start": int(window["accepted_window_start"]),
        "accepted_window_end": int(window["accepted_window_end"]),
        "analysis_window_start": start,
        "analysis_window_end": end,
        "pre_period_start": pre_start,
        "pre_period_end": pre_end,
        "post_period_start": post_start,
        "post_period_end": post_end,
        "field": field,
    }
    if sample_col is not None:
        base[sample_col] = int(sample_value) if sample_value is not None else np.nan

    pre = full[pre_start : pre_end + 1, :]
    post = full[post_start : post_end + 1, :]
    pre_good = int(_row_finite_mask(pre).sum())
    post_good = int(_row_finite_mask(post).sum())
    if pre_good < int(cfg.min_period_days) or post_good < int(cfg.min_period_days):
        row = {**base, "pre_post_distance": np.nan, "within_pre_variability": np.nan, "within_post_variability": np.nan, "separation_ratio": np.nan, "pre_post_separation_label": "unavailable", "onset_day": np.nan, "midpoint_day": np.nan, "finish_day": np.nan, "duration": np.nan, "progress_monotonicity_corr": np.nan, "n_crossings_025": np.nan, "n_crossings_050": np.nan, "n_crossings_075": np.nan, "progress_quality_label": "period_too_short", "near_left_boundary": False, "near_right_boundary": False, "near_excluded_candidate_days": "none"}
        return row, pd.DataFrame()

    pre_proto = _nanmean_rows(pre)
    post_proto = _nanmean_rows(post)
    if pre_proto is None or post_proto is None:
        row = {**base, "pre_post_distance": np.nan, "within_pre_variability": np.nan, "within_post_variability": np.nan, "separation_ratio": np.nan, "pre_post_separation_label": "unavailable", "onset_day": np.nan, "midpoint_day": np.nan, "finish_day": np.nan, "duration": np.nan, "progress_monotonicity_corr": np.nan, "n_crossings_025": np.nan, "n_crossings_050": np.nan, "n_crossings_075": np.nan, "progress_quality_label": "prototype_unavailable", "near_left_boundary": False, "near_right_boundary": False, "near_excluded_candidate_days": "none"}
        return row, pd.DataFrame()

    vec = post_proto - pre_proto
    pre_post_distance = _norm(vec)
    pre_var = float(np.nanmean([_norm(x - pre_proto) for x in pre[_row_finite_mask(pre), :]])) if pre_good else np.nan
    post_var = float(np.nanmean([_norm(x - post_proto) for x in post[_row_finite_mask(post), :]])) if post_good else np.nan
    within = np.nanmean([pre_var, post_var])
    separation_ratio = float(pre_post_distance / (within + 1e-12)) if np.isfinite(within) else np.nan
    sep_label = _separation_label(separation_ratio, settings)
    denom = float(np.nansum(np.square(vec)))

    curve_rows = []
    days = []
    vals = []
    if denom > float(cfg.min_transition_norm):
        for day in range(start, end + 1):
            x = full[day, :]
            if not np.all(np.isfinite(x)):
                continue
            raw = float(np.nansum((x - pre_proto) * vec) / denom)
            prog = float(np.clip(raw, float(cfg.progress_clip_min), float(cfg.progress_clip_max)))
            days.append(int(day))
            vals.append(prog)
            curve_rows.append({**base, "day": int(day), "relative_to_anchor": int(day - anchor), "raw_progress": raw, "progress": prog})
    days_arr = np.asarray(days, dtype=int)
    vals_arr = np.asarray(vals, dtype=float)

    if len(vals_arr) == 0 or denom <= float(cfg.min_transition_norm):
        onset = mid = finish = np.nan
        duration = np.nan
        corr = np.nan
        c25 = c50 = c75 = np.nan
        quality = "no_clear_prepost_separation" if sep_label == "no_clear_separation" else "progress_unavailable"
    else:
        onset = _first_stable_crossing(days_arr, vals_arr, float(cfg.threshold_onset), int(cfg.stable_crossing_days))
        mid = _first_stable_crossing(days_arr, vals_arr, float(cfg.threshold_midpoint), int(cfg.stable_crossing_days))
        finish = _first_stable_crossing(days_arr, vals_arr, float(cfg.threshold_finish), int(cfg.stable_crossing_days))
        duration = float(finish - onset + 1) if np.isfinite(onset) and np.isfinite(finish) else np.nan
        corr = np.nan
        good = np.isfinite(vals_arr)
        if int(good.sum()) >= 3 and np.nanstd(vals_arr[good]) > 1e-12:
            corr = float(np.corrcoef(days_arr[good].astype(float), vals_arr[good])[0, 1])
        c25 = _count_crossings(vals_arr, float(cfg.threshold_onset))
        c50 = _count_crossings(vals_arr, float(cfg.threshold_midpoint))
        c75 = _count_crossings(vals_arr, float(cfg.threshold_finish))
        near_left = any(np.isfinite(x) and x <= start + int(cfg.boundary_margin_days) for x in [onset, mid])
        near_right = any(np.isfinite(x) and x >= end - int(cfg.boundary_margin_days) for x in [mid, finish])
        # A monotonic transition is expected to cross each threshold once.
        # V7-e hotfix 01: the previous condition used c25+c50+c75 >= 3,
        # which incorrectly labeled a clean 0->1 progress curve with one
        # crossing at 0.25, 0.50, and 0.75 as nonmonotonic.
        # Only repeated crossings beyond the first crossing are treated as
        # oscillatory/nonmonotonic, or a low day-progress correlation.
        excess_crossings = int(max(c25 - 1, 0) + max(c50 - 1, 0) + max(c75 - 1, 0))
        low_monotonic_corr = bool(np.isfinite(corr) and corr < float(cfg.monotonic_corr_threshold))
        repeated_threshold_crossing = bool(excess_crossings >= 1)
        nonmono = low_monotonic_corr or repeated_threshold_crossing
        if sep_label == "no_clear_separation":
            quality = "no_clear_prepost_separation"
        elif not np.isfinite(mid):
            quality = "partial_progress"
        elif near_left or near_right:
            quality = "boundary_limited_progress"
        elif nonmono:
            quality = "nonmonotonic_progress"
        elif sep_label in ["clear_separation", "moderate_separation"] and np.isfinite(onset) and np.isfinite(finish):
            quality = "monotonic_clear_progress"
        else:
            quality = "monotonic_broad_progress"

    near_left_boundary = bool(any(np.isfinite(x) and x <= start + int(cfg.boundary_margin_days) for x in [onset, mid]))
    near_right_boundary = bool(any(np.isfinite(x) and x >= end - int(cfg.boundary_margin_days) for x in [mid, finish]))
    near_excl = _near_excluded_days([onset, mid, finish], settings)
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
        "near_excluded_candidate_days": near_excl,
    }
    return row, pd.DataFrame(curve_rows)


def _load_or_make_bootstrap_indices(profiles: dict, settings: StagePartitionV7Settings, output_root: Path) -> list[np.ndarray]:
    n_years = int(np.asarray(profiles[FIELDS[0]].raw_cube).shape[0])
    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    if bool(settings.progress_timing.reuse_v7_b_resamples_if_available):
        src = settings.layer_root() / "outputs" / settings.progress_timing.source_v7_b_output_tag / "bootstrap_resample_year_indices_v7_b.csv"
        if src.exists():
            df = pd.read_csv(src)
            out = []
            for _, r in df.sort_values("bootstrap_id").iterrows():
                txt = str(r.get("sampled_year_indices", ""))
                vals = [int(x) for x in txt.split(";") if str(x).strip() != ""]
                if vals:
                    out.append(np.asarray(vals, dtype=int))
            if len(out) >= n_boot:
                out = out[:n_boot]
                write_dataframe(pd.DataFrame({"bootstrap_id": range(len(out)), "sampled_year_indices": [";".join(str(int(x)) for x in arr.tolist()) for arr in out]}), output_root / "bootstrap_resample_year_indices_v7_e.csv")
                return out
    rng = np.random.default_rng(int(settings.bootstrap.random_seed))
    out = [rng.integers(0, n_years, size=n_years, dtype=int) for _ in range(n_boot)]
    write_dataframe(pd.DataFrame({"bootstrap_id": range(len(out)), "sampled_year_indices": [";".join(str(int(x)) for x in arr.tolist()) for arr in out]}), output_root / "bootstrap_resample_year_indices_v7_e.csv")
    return out


def _summarize_progress_samples(samples: pd.DataFrame, id_prefix: str = "bootstrap") -> pd.DataFrame:
    rows = []
    for (window_id, field), sub in samples.groupby(["window_id", "field"], sort=False):
        mids = pd.to_numeric(sub["midpoint_day"], errors="coerce").dropna().to_numpy(dtype=float)
        onsets = pd.to_numeric(sub["onset_day"], errors="coerce").dropna().to_numpy(dtype=float)
        finishes = pd.to_numeric(sub["finish_day"], errors="coerce").dropna().to_numpy(dtype=float)
        durations = pd.to_numeric(sub["duration"], errors="coerce").dropna().to_numpy(dtype=float)
        if mids.size == 0:
            rows.append({"window_id": window_id, "field": field, "n_valid_midpoints": 0})
            continue
        modal = float(pd.Series(np.round(mids).astype(int)).mode().iloc[0])
        qlabels = sub["progress_quality_label"].astype(str).value_counts().to_dict()
        seplabels = sub["pre_post_separation_label"].astype(str).value_counts().to_dict()
        rows.append(
            {
                "window_id": window_id,
                "field": field,
                "n_valid_midpoints": int(mids.size),
                "valid_midpoint_fraction": float(mids.size / len(sub)) if len(sub) else np.nan,
                "onset_median": float(np.nanmedian(onsets)) if onsets.size else np.nan,
                "midpoint_median": float(np.nanmedian(mids)),
                "midpoint_q025": float(np.nanpercentile(mids, 2.5)),
                "midpoint_q975": float(np.nanpercentile(mids, 97.5)),
                "midpoint_iqr": float(np.nanpercentile(mids, 75) - np.nanpercentile(mids, 25)),
                "midpoint_modal_rounded_day": modal,
                "support_midpoint_within_2d_of_modal": float(np.mean(np.abs(mids - modal) <= 2)),
                "support_midpoint_within_3d_of_modal": float(np.mean(np.abs(mids - modal) <= 3)),
                "finish_median": float(np.nanmedian(finishes)) if finishes.size else np.nan,
                "duration_median": float(np.nanmedian(durations)) if durations.size else np.nan,
                "duration_iqr": float(np.nanpercentile(durations, 75) - np.nanpercentile(durations, 25)) if durations.size else np.nan,
                "support_near_boundary": float((sub["near_left_boundary"].astype(bool) | sub["near_right_boundary"].astype(bool)).mean()),
                "dominant_progress_quality_label": max(qlabels, key=qlabels.get) if qlabels else "none",
                "dominant_prepost_separation_label": max(seplabels, key=seplabels.get) if seplabels else "none",
            }
        )
    return pd.DataFrame(rows)


def _progress_timing_confidence(summary: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    out = summary.copy()
    labels = []
    for _, r in out.iterrows():
        if int(r.get("n_valid_midpoints", 0)) <= 0:
            labels.append("progress_unavailable")
            continue
        sep = str(r.get("dominant_prepost_separation_label", ""))
        qual = str(r.get("dominant_progress_quality_label", ""))
        if sep == "no_clear_separation":
            labels.append("no_clear_prepost_separation")
        elif "boundary" in qual or float(r.get("support_near_boundary", 0.0)) >= float(settings.progress_timing.boundary_support_threshold):
            labels.append("boundary_limited_progress")
        elif "nonmonotonic" in qual:
            labels.append("nonmonotonic_progress")
        elif float(r.get("support_midpoint_within_3d_of_modal", 0.0)) >= 0.75 and float(r.get("midpoint_iqr", 999)) <= 5:
            labels.append("progress_timing_moderate")
        else:
            labels.append("progress_timing_uncertain")
    out["progress_timing_confidence_label"] = labels
    return out


def _pivot_progress_samples(samples: pd.DataFrame, id_col: str) -> pd.DataFrame:
    keep = samples[[id_col, "window_id", "field", "onset_day", "midpoint_day", "finish_day"]].copy()
    piv = keep.pivot_table(index=[id_col, "window_id"], columns="field", values=["onset_day", "midpoint_day", "finish_day"], aggfunc="first")
    piv.columns = [f"{field}_{metric.replace('_day','')}" for metric, field in piv.columns]
    piv = piv.reset_index()
    return piv


def _pair_progress_stats(piv: pd.DataFrame, window_id: str, field_a: str, field_b: str) -> dict:
    sub = piv[piv["window_id"].astype(str) == str(window_id)].copy()
    needed = [f"{field_a}_onset", f"{field_a}_midpoint", f"{field_a}_finish", f"{field_b}_onset", f"{field_b}_midpoint", f"{field_b}_finish"]
    if sub.empty or any(c not in sub.columns for c in needed):
        return {"n_valid": 0}
    vals = {c: pd.to_numeric(sub[c], errors="coerce") for c in needed}
    good = np.ones(len(sub), dtype=bool)
    for c in needed:
        good &= vals[c].notna().to_numpy()
    if int(good.sum()) == 0:
        return {"n_valid": 0}
    ao = vals[f"{field_a}_onset"][good].to_numpy(dtype=float)
    am = vals[f"{field_a}_midpoint"][good].to_numpy(dtype=float)
    af = vals[f"{field_a}_finish"][good].to_numpy(dtype=float)
    bo = vals[f"{field_b}_onset"][good].to_numpy(dtype=float)
    bm = vals[f"{field_b}_midpoint"][good].to_numpy(dtype=float)
    bf = vals[f"{field_b}_finish"][good].to_numpy(dtype=float)
    lag_b_minus_a = bm - am
    prob_a_mid = float(np.mean(lag_b_minus_a > 0))
    prob_b_mid = float(np.mean(lag_b_minus_a < 0))
    if prob_a_mid >= prob_b_mid:
        early, late = field_a, field_b
        early_on, early_mid, early_fin = ao, am, af
        late_on, late_mid, late_fin = bo, bm, bf
        major_lag = lag_b_minus_a
        prob_mid = prob_a_mid
    else:
        early, late = field_b, field_a
        early_on, early_mid, early_fin = bo, bm, bf
        late_on, late_mid, late_fin = ao, am, af
        major_lag = -lag_b_minus_a
        prob_mid = prob_b_mid
    prob_onset = float(np.mean(early_on < late_on))
    prob_finish_before_onset = float(np.mean(early_fin < late_on))
    prob_sync = float(np.mean(np.abs(major_lag) <= 2.0))
    return {
        "n_valid": int(good.sum()),
        "field_a": field_a,
        "field_b": field_b,
        "field_early_candidate": early,
        "field_late_candidate": late,
        "prob_a_midpoint_before_b": prob_a_mid,
        "prob_b_midpoint_before_a": prob_b_mid,
        "prob_midpoint_early_before_late": float(prob_mid),
        "prob_onset_early_before_late": prob_onset,
        "prob_finish_before_onset": prob_finish_before_onset,
        "prob_sync_midpoint_2d": prob_sync,
        "median_midpoint_lag_late_minus_early": float(np.nanmedian(major_lag)),
        "q025_midpoint_lag_late_minus_early": float(np.nanpercentile(major_lag, 2.5)),
        "q975_midpoint_lag_late_minus_early": float(np.nanpercentile(major_lag, 97.5)),
        "iqr_midpoint_lag_late_minus_early": float(np.nanpercentile(major_lag, 75) - np.nanpercentile(major_lag, 25)),
    }


def _label_progress_order(row: pd.Series, settings: StagePartitionV7Settings) -> tuple[str, bool, str]:
    cfg = settings.progress_timing
    p = float(row.get("prob_midpoint_early_before_late", np.nan))
    pon = float(row.get("prob_onset_early_before_late", np.nan))
    psep = float(row.get("prob_finish_before_onset", np.nan))
    psync = float(row.get("prob_sync_midpoint_2d", np.nan))
    lag = float(row.get("median_midpoint_lag_late_minus_early", np.nan))
    loyo = float(row.get("loyo_prob_midpoint_early_before_late", np.nan))
    caution = []
    if str(row.get("early_field_progress_label", "")).startswith("no_clear") or str(row.get("late_field_progress_label", "")).startswith("no_clear"):
        return "ambiguous_progress_order", False, "one_or_both_fields_have_no_clear_prepost_separation"
    if psync >= float(cfg.sync_min_prob):
        return "sync_progress", False, "midpoints_overlap_within_sync_radius"
    if p >= float(cfg.separated_min_midpoint_prob) and psep >= float(cfg.separated_min_finish_before_onset_prob) and lag >= float(cfg.separated_min_median_lag_days) and loyo >= float(cfg.separated_min_loyo_midpoint_prob):
        return "separated_progress_order", True, "timing_progress_order_only_not_causality"
    if p >= float(cfg.shifted_min_midpoint_prob) and lag >= float(cfg.shifted_min_median_lag_days) and loyo >= float(cfg.shifted_min_loyo_midpoint_prob):
        return "shifted_progress_order", True, "timing_progress_order_only_not_causality"
    if p >= float(cfg.moderate_min_midpoint_prob) and lag >= float(cfg.moderate_min_median_lag_days) and loyo >= float(cfg.moderate_min_loyo_midpoint_prob):
        return "moderate_progress_order", True, "candidate_progress_order_not_hard_result"
    if p < 0.70:
        caution.append("weak_majority_direction")
    if np.isfinite(loyo) and loyo < 0.60:
        caution.append("loyo_support_weak")
    return "ambiguous_progress_order", False, ";".join(caution) if caution else "ambiguous_midpoint_order"


def _pairwise_progress_orders(boot_df: pd.DataFrame, loyo_df: pd.DataFrame, field_summary: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    boot_piv = _pivot_progress_samples(boot_df, "bootstrap_id")
    loyo_piv = _pivot_progress_samples(loyo_df, "left_out_year_index")
    labels = field_summary.set_index(["window_id", "field"])["progress_timing_confidence_label"].to_dict() if not field_summary.empty else {}
    rows = []
    for window_id in sorted(boot_df["window_id"].astype(str).unique()):
        for a, b in combinations(FIELDS, 2):
            bs = _pair_progress_stats(boot_piv, window_id, a, b)
            if int(bs.get("n_valid", 0)) == 0:
                continue
            ls = _pair_progress_stats(loyo_piv, window_id, a, b)
            early = bs["field_early_candidate"]
            late = bs["field_late_candidate"]
            if int(ls.get("n_valid", 0)) > 0:
                if ls.get("field_early_candidate") == early and ls.get("field_late_candidate") == late:
                    loyo_prob = float(ls.get("prob_midpoint_early_before_late", np.nan))
                elif ls.get("field_early_candidate") == late and ls.get("field_late_candidate") == early:
                    loyo_prob = float(1.0 - float(ls.get("prob_midpoint_early_before_late", np.nan)))
                else:
                    loyo_prob = np.nan
            else:
                loyo_prob = np.nan
            row = {
                "window_id": window_id,
                **bs,
                "loyo_prob_midpoint_early_before_late": loyo_prob,
                "loyo_median_midpoint_lag_late_minus_early": ls.get("median_midpoint_lag_late_minus_early", np.nan),
                "early_field_progress_label": labels.get((window_id, early), "unknown"),
                "late_field_progress_label": labels.get((window_id, late), "unknown"),
            }
            label, usable, caution = _label_progress_order(pd.Series(row), settings)
            row["progress_order_label"] = label
            row["usable_as_progress_order"] = bool(usable)
            row["caution"] = caution
            rows.append(row)
    return pd.DataFrame(rows)


def _field_order_scores(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_id, sub in pair_df.groupby("window_id", sort=False):
        scores = {f: [] for f in FIELDS}
        for _, r in sub.iterrows():
            early = str(r["field_early_candidate"])
            late = str(r["field_late_candidate"])
            p = float(r["prob_midpoint_early_before_late"])
            if early in scores:
                scores[early].append(p)
            if late in scores:
                scores[late].append(1.0 - p)
        for f in FIELDS:
            vals = scores[f]
            score = float(np.nanmean(vals)) if vals else np.nan
            if np.isfinite(score) and score >= 0.65:
                role = "early_tendency"
            elif np.isfinite(score) and score <= 0.35:
                role = "late_tendency"
            elif np.isfinite(score):
                role = "middle_or_uncertain"
            else:
                role = "unavailable"
            rows.append({"window_id": window_id, "field": f, "progress_order_score": score, "dominant_role": role})
    return pd.DataFrame(rows)


def _window_summary(pair_df: pd.DataFrame, field_summary: pd.DataFrame, score_df: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    rows = []
    for window_id, sub in pair_df.groupby("window_id", sort=False):
        n = int(len(sub))
        counts = sub["progress_order_label"].value_counts().to_dict()
        n_sep = int(counts.get("separated_progress_order", 0))
        n_shift = int(counts.get("shifted_progress_order", 0))
        n_mod = int(counts.get("moderate_progress_order", 0))
        n_sync = int(counts.get("sync_progress", 0))
        n_amb = int(counts.get("ambiguous_progress_order", 0))
        fsub = field_summary[field_summary["window_id"].astype(str) == str(window_id)]
        n_no_sep = int((fsub["progress_timing_confidence_label"].astype(str) == "no_clear_prepost_separation").sum()) if not fsub.empty else 0
        n_nonmono = int(fsub["progress_timing_confidence_label"].astype(str).str.contains("nonmonotonic", na=False).sum()) if not fsub.empty else 0
        n_usable = n_sep + n_shift + n_mod
        sync_frac = n_sync / n if n else np.nan
        amb_frac = n_amb / n if n else np.nan
        if n_no_sep >= 3:
            cls = "no_clear_prepost_separation_window"
        elif n_nonmono >= 3:
            cls = "mixed_or_nonmonotonic_window"
        elif n_sep + n_shift >= int(settings.progress_timing.partial_order_min_edges):
            cls = "progress_order_supported"
        elif n_usable >= 2:
            cls = "partial_progress_order"
        elif sync_frac >= float(settings.progress_timing.sync_window_fraction_threshold):
            cls = "sync_progress_window"
        elif amb_frac >= float(settings.progress_timing.ambiguous_fraction_threshold):
            cls = "mixed_or_nonmonotonic_window"
        else:
            cls = "weak_or_inconclusive_progress_window"
        ssub = score_df[score_df["window_id"].astype(str) == str(window_id)]
        early = ";".join(ssub[ssub["dominant_role"] == "early_tendency"]["field"].astype(str).tolist()) or "none"
        late = ";".join(ssub[ssub["dominant_role"] == "late_tendency"]["field"].astype(str).tolist()) or "none"
        middle = ";".join(ssub[ssub["dominant_role"] == "middle_or_uncertain"]["field"].astype(str).tolist()) or "none"
        if cls == "sync_progress_window":
            result = "Most field progress midpoints overlap; do not force field order."
        elif cls == "no_clear_prepost_separation_window":
            result = "Most fields lack clear pre/post separation; progress order is not scientifically interpretable here."
        elif cls == "mixed_or_nonmonotonic_window":
            result = "Progress curves are mixed/nonmonotonic or pairwise orders are largely ambiguous."
        elif cls in ["progress_order_supported", "partial_progress_order"]:
            result = f"Progress timing structure: early={early}, late={late}, middle_or_uncertain={middle}."
        else:
            result = "Only weak or inconclusive progress-order evidence under current criteria."
        rows.append({"window_id": window_id, "n_total_pairs": n, "n_separated_progress_order": n_sep, "n_shifted_progress_order": n_shift, "n_moderate_progress_order": n_mod, "n_sync_progress": n_sync, "n_ambiguous_progress_order": n_amb, "n_fields_no_clear_prepost_separation": n_no_sep, "n_fields_nonmonotonic": n_nonmono, "order_pair_fraction": n_usable / n if n else np.nan, "sync_pair_fraction": sync_frac, "ambiguous_pair_fraction": amb_frac, "early_group": early, "middle_group": middle, "late_group": late, "uncertain_group": middle, "window_progress_order_class": cls, "window_level_result": result, "usable_for_scientific_interpretation": bool(cls not in ["weak_or_inconclusive_progress_window"]), "caution": "progress_order_only_not_causality; whole_field_progress_only; no_spatial_earliest_regions"})
    return pd.DataFrame(rows)


def _try_write_plots(figure_root: Path, curve_df: pd.DataFrame, pair_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        for window_id, sub in curve_df.groupby("window_id"):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for field in FIELDS:
                fs = sub[sub["field"] == field].sort_values("day")
                if fs.empty:
                    continue
                ax.plot(fs["day"], fs["progress"], label=field)
            ax.axhline(0.25, linestyle="--", linewidth=0.8)
            ax.axhline(0.50, linestyle="--", linewidth=0.8)
            ax.axhline(0.75, linestyle="--", linewidth=0.8)
            anchor = int(sub["anchor_day"].iloc[0])
            ax.axvline(anchor, linewidth=1.0)
            ax.set_title(f"{window_id} progress curves\nTiming progress only, not causality")
            ax.set_xlabel("day index")
            ax.set_ylabel("progress toward post state")
            ax.legend(ncol=5, fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_root / f"{window_id}_progress_curves_v7_e.png", dpi=160)
            plt.close(fig)
        for window_id, sub in pair_df.groupby("window_id"):
            mat = pd.DataFrame(np.nan, index=FIELDS, columns=FIELDS, dtype=float)
            for _, r in sub.iterrows():
                early = str(r["field_early_candidate"])
                late = str(r["field_late_candidate"])
                p = float(r["prob_midpoint_early_before_late"])
                if early in mat.index and late in mat.columns:
                    mat.loc[early, late] = p
                    mat.loc[late, early] = 1.0 - p
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat.to_numpy(dtype=float), vmin=0, vmax=1)
            ax.set_xticks(range(len(FIELDS)), FIELDS)
            ax.set_yticks(range(len(FIELDS)), FIELDS)
            ax.set_title(f"{window_id} progress midpoint order probability")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(figure_root / f"{window_id}_pairwise_progress_order_heatmap_v7_e.png", dpi=160)
            plt.close(fig)
    except Exception as exc:
        (figure_root.parent / "plot_warning_v7_e.txt").write_text(str(exc), encoding="utf-8")


def _write_audit_log(log_root: Path, settings: StagePartitionV7Settings, windows: pd.DataFrame, run_meta: dict) -> None:
    txt = [
        "# field_transition_progress_timing_v7_e audit log",
        "",
        f"created_utc: {now_utc()}",
        "",
        "## Scope",
        "- Answers only the first question: field-level timing around accepted transition windows.",
        "- Changes the observation target from change-intensity peaks/intervals to pre/post transition progress.",
        "- Does not infer causality, does not analyze spatial earliest regions, and does not read downstream pathway or lead-lag outputs.",
        "",
        "## Accepted windows",
        windows.to_string(index=False),
        "",
        "## Method notes",
        f"- analysis_radius_days = {settings.progress_timing.analysis_radius_days}",
        f"- pre_period offsets = -{settings.progress_timing.pre_offset_start_days}..-{settings.progress_timing.pre_offset_end_days}",
        f"- post_period offsets = +{settings.progress_timing.post_offset_start_days}..+{settings.progress_timing.post_offset_end_days}",
        f"- n_bootstrap = {settings.bootstrap.effective_n_bootstrap()}",
        "- Progress is projection from pre prototype to post prototype, clipped to [0, 1].",
        "- Excluded candidates are flagged if close but are not reintroduced as main windows.",
        "",
        "## Run meta",
        json.dumps(run_meta, indent=2, default=str),
    ]
    (log_root / "field_transition_progress_timing_v7_e_audit_log.md").write_text("\n".join(txt), encoding="utf-8")


def run_field_transition_progress_timing_v7_e(settings: StagePartitionV7Settings | None = None) -> None:
    settings = settings or StagePartitionV7Settings()
    dirs = _prepare_dirs(settings)
    out = dirs["output_root"]
    log_root = dirs["log_root"]
    fig_root = dirs["figure_root"]
    settings.write_json(log_root / "config_used.json")

    audit = _audit_accepted_points(settings)
    write_json(audit, log_root / "accepted_window_evidence.json")
    windows_source = audit["windows_df"]

    smoothed = load_smoothed_fields(settings.foundation.smoothed_fields_path())
    profiles = build_profiles(smoothed, settings.profile)
    first_cube = np.asarray(profiles[FIELDS[0]].raw_cube)
    n_years = int(first_cube.shape[0])
    n_days = int(first_cube.shape[1])
    years = list(smoothed.get("years", list(range(n_years)))) if isinstance(smoothed, dict) else list(range(n_years))
    progress_windows = _build_progress_windows(windows_source, settings, n_days)
    write_dataframe(progress_windows, out / "accepted_windows_used_v7_e.csv")

    joint_state = build_state_matrix(profiles, settings.state)
    shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)

    observed_rows = []
    curve_rows = []
    for field in FIELDS:
        matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(profiles, field, None, standardize=settings.state.standardize, trim_invalid_days=settings.state.trim_invalid_days, shared_valid_day_index=shared_valid_day_index)
        for _, w in progress_windows.iterrows():
            row, curve = _compute_progress_for_window(matrix, valid_day_index, n_days, w, field, settings)
            observed_rows.append(row)
            if not curve.empty:
                curve_rows.extend(curve.to_dict("records"))
    observed_df = pd.DataFrame(observed_rows)
    curve_df = pd.DataFrame(curve_rows)
    write_dataframe(observed_df, out / "field_transition_progress_observed_v7_e.csv")
    write_dataframe(curve_df, out / "field_transition_progress_observed_curves_long_v7_e.csv")

    boot_indices = _load_or_make_bootstrap_indices(profiles, settings, out)
    boot_rows = []
    for b, year_idx in enumerate(boot_indices):
        if b % int(settings.bootstrap.progress_every) == 0:
            print(f"[V7-e bootstrap] {b}/{len(boot_indices)}")
        for field in FIELDS:
            matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(profiles, field, year_idx, standardize=settings.state.standardize, trim_invalid_days=settings.state.trim_invalid_days, shared_valid_day_index=shared_valid_day_index)
            for _, w in progress_windows.iterrows():
                row, _ = _compute_progress_for_window(matrix, valid_day_index, n_days, w, field, settings, sample_col="bootstrap_id", sample_value=b)
                boot_rows.append(row)
    boot_df = pd.DataFrame(boot_rows)
    if bool(settings.progress_timing.write_sample_long_tables):
        write_dataframe(boot_df, out / "field_transition_progress_bootstrap_samples_v7_e.csv")

    loyo_rows = []
    for left_out in range(n_years):
        if left_out % 10 == 0:
            print(f"[V7-e LOYO] {left_out}/{n_years}")
        keep = np.asarray([i for i in range(n_years) if i != left_out], dtype=int)
        for field in FIELDS:
            matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(profiles, field, keep, standardize=settings.state.standardize, trim_invalid_days=settings.state.trim_invalid_days, shared_valid_day_index=shared_valid_day_index)
            for _, w in progress_windows.iterrows():
                row, _ = _compute_progress_for_window(matrix, valid_day_index, n_days, w, field, settings, sample_col="left_out_year_index", sample_value=left_out)
                row["left_out_year"] = years[left_out] if left_out < len(years) else left_out
                loyo_rows.append(row)
    loyo_df = pd.DataFrame(loyo_rows)
    write_dataframe(loyo_df, out / "field_transition_progress_loyo_samples_v7_e.csv")

    boot_summary = _progress_timing_confidence(_summarize_progress_samples(boot_df), settings)
    loyo_summary = _summarize_progress_samples(loyo_df)
    loyo_summary = loyo_summary.rename(columns={
        "midpoint_median": "loyo_midpoint_median",
        "midpoint_q025": "loyo_midpoint_q025",
        "midpoint_q975": "loyo_midpoint_q975",
        "midpoint_iqr": "loyo_midpoint_iqr",
        "support_midpoint_within_3d_of_modal": "loyo_support_midpoint_within_3d_of_modal",
    })
    write_dataframe(boot_summary, out / "field_transition_progress_bootstrap_summary_v7_e.csv")
    write_dataframe(loyo_summary, out / "field_transition_progress_loyo_summary_v7_e.csv")

    pair_df = _pairwise_progress_orders(boot_df, loyo_df, boot_summary, settings)
    write_dataframe(pair_df, out / "pairwise_progress_order_summary_v7_e.csv")
    score_df = _field_order_scores(pair_df)
    write_dataframe(score_df, out / "field_progress_order_rank_summary_v7_e.csv")
    window_summary = _window_summary(pair_df, boot_summary, score_df, settings)
    write_dataframe(window_summary, out / "window_progress_orderability_summary_v7_e.csv")
    edges = pair_df[pair_df["usable_as_progress_order"].astype(bool)].copy()
    write_dataframe(edges, out / "progress_order_graph_edges_v7_e.csv")

    if bool(settings.progress_timing.write_plots):
        _try_write_plots(fig_root, curve_df, pair_df)

    run_meta = {
        "status": "success",
        "created_utc": now_utc(),
        "run_label": settings.progress_timing.output_tag,
        "method": "pre_post_prototype_projection_progress_timing",
        "accepted_peak_days": [int(x) for x in settings.accepted_windows.accepted_peak_days],
        "excluded_candidate_days": [int(x) for x in settings.accepted_windows.excluded_candidate_days],
        "n_windows": int(len(progress_windows)),
        "n_bootstrap": int(len(boot_indices)),
        "n_years": int(n_years),
        "n_observed_rows": int(len(observed_df)),
        "n_bootstrap_rows": int(len(boot_df)),
        "n_loyo_rows": int(len(loyo_df)),
        "downstream_lead_lag_included": False,
        "pathway_included": False,
        "spatial_earliest_region_included": False,
        "causal_interpretation_included": False,
        "notes": [
            "V7-e answers only field-level transition progress timing around accepted windows.",
            "Progress is projection from pre prototype to post prototype, not a causal path.",
            "18/96/132/135 are excluded from main windows and only flagged if nearby.",
        ],
    }
    write_json(run_meta, out / "run_meta.json")
    write_json(run_meta, log_root / "run_meta.json")
    summary = {
        "run_label": settings.progress_timing.output_tag,
        "window_progress_order_class_counts": window_summary["window_progress_order_class"].value_counts().to_dict() if not window_summary.empty else {},
        "progress_order_label_counts": pair_df["progress_order_label"].value_counts().to_dict() if not pair_df.empty else {},
        "n_usable_progress_order_edges": int(edges.shape[0]),
        "main_output": "window_progress_orderability_summary_v7_e.csv",
    }
    write_json(summary, out / "summary.json")
    _write_audit_log(log_root, settings, progress_windows, run_meta)
    print(f"[V7-e] wrote outputs to {out}")


__all__ = ["run_field_transition_progress_timing_v7_e"]
