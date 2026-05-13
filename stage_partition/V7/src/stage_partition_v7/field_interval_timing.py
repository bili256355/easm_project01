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
from stage_partition_v6.detector_ruptures_window import run_ruptures_window
from stage_partition_v6.timeline import day_index_to_month_day

from .config import StagePartitionV7Settings
from .field_state import FIELDS, build_field_state_matrix_for_year_indices
from .field_timing import _audit_accepted_points
from .report import now_utc, write_dataframe, write_json


METRICS = ["detector_score", "prepost_contrast"]


def _prepare_dirs(settings: StagePartitionV7Settings) -> dict[str, Path]:
    settings.output.output_tag = settings.interval_timing.output_tag
    out = settings.output_root()
    log_dir = settings.log_root()
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"output_root": out, "log_root": log_dir, "figure_root": fig_dir}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": str(exc), "path": str(path)}


def _build_interval_analysis_windows(windows_df: pd.DataFrame, settings: StagePartitionV7Settings, n_days: int) -> pd.DataFrame:
    rows: list[dict] = []
    radius = int(settings.interval_timing.analysis_radius_days)
    accepted = set(int(x) for x in settings.accepted_windows.accepted_peak_days)
    excluded = [int(x) for x in settings.accepted_windows.excluded_candidate_days]
    excluded_margin = int(settings.interval_timing.excluded_candidate_margin_days)
    for _, r in windows_df.sort_values("main_peak_day").iterrows():
        anchor = int(r["main_peak_day"])
        if anchor not in accepted:
            continue
        accepted_start = int(r["start_day"])
        accepted_end = int(r["end_day"])
        analysis_start = max(0, anchor - radius)
        analysis_end = min(int(n_days) - 1, anchor + radius)
        nearby_excluded = [x for x in excluded if analysis_start - excluded_margin <= x <= analysis_end + excluded_margin]
        rows.append(
            {
                "window_id": str(r.get("window_id", f"W{anchor:03d}")),
                "anchor_day": anchor,
                "anchor_month_day": day_index_to_month_day(anchor),
                "accepted_window_start": accepted_start,
                "accepted_window_end": accepted_end,
                "analysis_window_start": analysis_start,
                "analysis_window_end": analysis_end,
                "analysis_radius_days": radius,
                "nearby_excluded_candidate_days": ";".join(str(x) for x in nearby_excluded) if nearby_excluded else "none",
                "n_nearby_excluded_candidates": int(len(nearby_excluded)),
                "source_window_id": str(r.get("window_id", f"W{anchor:03d}")),
                "note": "V7-d interval analysis window: accepted anchor +/- radius; excluded candidates only flagged, not reintroduced",
            }
        )
    return pd.DataFrame(rows)


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


def _compute_prepost_contrast_curve(
    matrix: np.ndarray,
    valid_day_index: np.ndarray,
    *,
    n_days: int,
    k: int,
    min_prepost_days: int,
) -> pd.Series:
    full = _state_to_full_matrix(matrix, valid_day_index, n_days)
    out: dict[int, float] = {}
    for day in range(int(n_days)):
        pre = full[max(0, day - int(k)) : day, :]
        post = full[day + 1 : min(int(n_days), day + 1 + int(k)), :]
        if int(_row_finite_mask(pre).sum()) < int(min_prepost_days) or int(_row_finite_mask(post).sum()) < int(min_prepost_days):
            continue
        pre_mean = _nanmean_rows(pre)
        post_mean = _nanmean_rows(post)
        if pre_mean is None or post_mean is None:
            continue
        diff = post_mean - pre_mean
        if not np.isfinite(diff).any():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            val = float(np.sqrt(np.nanmean(np.square(diff))))
        if np.isfinite(val):
            out[int(day)] = val
    return pd.Series(out, dtype=float).sort_index()


def _normalize_curve_within_window(curve: pd.Series, start: int, end: int, cfg) -> pd.Series:
    sub = curve.sort_index()
    sub = sub[(sub.index.astype(int) >= int(start)) & (sub.index.astype(int) <= int(end))].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    vals = sub.to_numpy(dtype=float)
    if not np.isfinite(vals).any():
        return pd.Series(dtype=float)
    q20 = float(np.nanpercentile(vals, 20))
    shifted = np.maximum(vals - q20, 0.0)
    maxv = float(np.nanmax(shifted)) if shifted.size else np.nan
    if (not np.isfinite(maxv)) or maxv <= float(cfg.no_signal_epsilon):
        return pd.Series(np.zeros_like(vals, dtype=float), index=sub.index.astype(int), dtype=float)
    return pd.Series(shifted / maxv, index=sub.index.astype(int), dtype=float)


def _segments_from_active_days(active_days: list[int]) -> list[list[int]]:
    if not active_days:
        return []
    days = sorted(int(x) for x in active_days)
    segments: list[list[int]] = [[days[0]]]
    for d in days[1:]:
        if d == segments[-1][-1] + 1:
            segments[-1].append(d)
        else:
            segments.append([d])
    return segments


def _near_excluded_segment(seg: list[int], settings: StagePartitionV7Settings) -> str:
    margin = int(settings.interval_timing.excluded_candidate_margin_days)
    hits = []
    for c in settings.accepted_windows.excluded_candidate_days:
        c = int(c)
        if any(abs(int(d) - c) <= margin for d in seg):
            hits.append(c)
    return ";".join(str(x) for x in hits) if hits else "none"


def _extract_active_interval(
    curve: pd.Series,
    window: pd.Series | dict,
    *,
    field: str,
    metric: str,
    settings: StagePartitionV7Settings,
    sample_id_name: str | None = None,
    sample_id_value: int | None = None,
) -> dict:
    cfg = settings.interval_timing
    start = int(window["analysis_window_start"])
    end = int(window["analysis_window_end"])
    astart = int(window["accepted_window_start"])
    aend = int(window["accepted_window_end"])
    anchor = int(window["anchor_day"])
    window_id = str(window["window_id"])
    norm = _normalize_curve_within_window(curve, start, end, cfg)
    base = {
        "window_id": window_id,
        "anchor_day": anchor,
        "accepted_window_start": astart,
        "accepted_window_end": aend,
        "analysis_window_start": start,
        "analysis_window_end": end,
        "field": field,
        "metric": metric,
    }
    if sample_id_name is not None:
        base[sample_id_name] = int(sample_id_value) if sample_id_value is not None else np.nan
    if norm.empty or not np.isfinite(norm.to_numpy(dtype=float)).any() or float(np.nanmax(norm.to_numpy(dtype=float))) <= float(cfg.no_signal_epsilon):
        return {
            **base,
            "peak_day": np.nan,
            "onset_day": np.nan,
            "center_day": np.nan,
            "end_day": np.nan,
            "duration": np.nan,
            "active_threshold": np.nan,
            "n_active_segments": 0,
            "has_multiple_active_segments": False,
            "near_left_boundary": False,
            "near_right_boundary": False,
            "inside_accepted_window_any": False,
            "near_excluded_candidate_days": "none",
            "interval_status": "no_active_signal",
        }
    peak_day = int(norm.idxmax())
    vals = norm.to_numpy(dtype=float)
    qthr = float(np.nanpercentile(vals, float(cfg.active_quantile_threshold) * 100.0))
    thr = max(float(cfg.active_min_norm_threshold), qthr)
    active_days = [int(d) for d, v in norm.items() if np.isfinite(v) and float(v) >= thr]
    segments = _segments_from_active_days(active_days)
    chosen = None
    for seg in segments:
        if peak_day in seg:
            chosen = seg
            break
    if chosen is None and segments:
        chosen = min(segments, key=lambda seg: min(abs(d - peak_day) for d in seg))
    if not chosen:
        return {
            **base,
            "peak_day": peak_day,
            "onset_day": np.nan,
            "center_day": np.nan,
            "end_day": np.nan,
            "duration": np.nan,
            "active_threshold": thr,
            "n_active_segments": 0,
            "has_multiple_active_segments": False,
            "near_left_boundary": False,
            "near_right_boundary": False,
            "inside_accepted_window_any": False,
            "near_excluded_candidate_days": "none",
            "interval_status": "no_active_segment_contains_peak",
        }
    weights = np.asarray([float(norm.loc[d]) for d in chosen], dtype=float)
    days = np.asarray(chosen, dtype=float)
    if np.isfinite(weights).any() and float(np.nansum(weights)) > 0:
        center = float(np.nansum(days * weights) / np.nansum(weights))
    else:
        center = float(np.nanmean(days))
    onset = int(min(chosen))
    endd = int(max(chosen))
    margin = int(cfg.boundary_margin_days)
    return {
        **base,
        "peak_day": peak_day,
        "onset_day": onset,
        "center_day": center,
        "end_day": endd,
        "duration": int(endd - onset + 1),
        "active_threshold": thr,
        "n_active_segments": int(len(segments)),
        "has_multiple_active_segments": bool(len(segments) > 1),
        "near_left_boundary": bool(onset <= start + margin),
        "near_right_boundary": bool(endd >= end - margin),
        "inside_accepted_window_any": bool(any(astart <= int(d) <= aend for d in chosen)),
        "near_excluded_candidate_days": _near_excluded_segment(chosen, settings),
        "interval_status": "ok",
    }


def _curves_for_field(
    profiles: dict,
    field: str,
    year_indices,
    shared_valid_day_index: np.ndarray,
    n_days: int,
    settings: StagePartitionV7Settings,
) -> tuple[pd.Series, pd.Series]:
    matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(
        profiles,
        field,
        year_indices,
        standardize=settings.state.standardize,
        trim_invalid_days=settings.state.trim_invalid_days,
        shared_valid_day_index=shared_valid_day_index if settings.state.use_joint_valid_day_index else None,
    )
    det = run_ruptures_window(matrix, settings.detector, day_index=valid_day_index)
    detector_curve = det["profile"].sort_index()
    contrast_curve = _compute_prepost_contrast_curve(
        matrix,
        valid_day_index,
        n_days=n_days,
        k=int(settings.interval_timing.prepost_k_days),
        min_prepost_days=int(settings.interval_timing.min_prepost_days),
    )
    return detector_curve, contrast_curve


def _load_or_make_bootstrap_indices(profiles: dict, settings: StagePartitionV7Settings, output_root: Path) -> list[np.ndarray]:
    n_years = int(np.asarray(profiles[FIELDS[0]].raw_cube).shape[0])
    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    v7b_root = settings.layer_root() / "outputs" / settings.interval_timing.source_v7_b_output_tag
    resample_path = v7b_root / "bootstrap_resample_year_indices_v7_b.csv"
    if settings.interval_timing.reuse_v7_b_resamples_if_available and resample_path.exists():
        df = pd.read_csv(resample_path)
        candidates = [c for c in df.columns if "sampled" in c and "indices" in c]
        if candidates:
            col = candidates[0]
            out = []
            for _, r in df.sort_values(df.columns[0]).head(n_boot).iterrows():
                txt = str(r[col])
                vals = [int(x) for x in txt.replace(",", ";").split(";") if str(x).strip() != ""]
                if vals:
                    out.append(np.asarray(vals, dtype=int))
            if len(out) == n_boot:
                return out
    rng = np.random.default_rng(int(settings.bootstrap.random_seed))
    out = [rng.integers(0, n_years, size=n_years, dtype=int) for _ in range(n_boot)]
    rows = [{"bootstrap_id": i, "sampled_year_indices": ";".join(str(int(x)) for x in arr.tolist())} for i, arr in enumerate(out)]
    write_dataframe(pd.DataFrame(rows), output_root / "bootstrap_resample_year_indices_v7_d.csv")
    return out


def _summarize_interval_samples(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window_id, field, metric), sub in samples.groupby(["window_id", "field", "metric"], sort=False):
        centers = pd.to_numeric(sub["center_day"], errors="coerce").dropna().to_numpy(dtype=float)
        durations = pd.to_numeric(sub["duration"], errors="coerce").dropna().to_numpy(dtype=float)
        onsets = pd.to_numeric(sub["onset_day"], errors="coerce").dropna().to_numpy(dtype=float)
        ends = pd.to_numeric(sub["end_day"], errors="coerce").dropna().to_numpy(dtype=float)
        if centers.size == 0:
            rows.append({"window_id": window_id, "field": field, "metric": metric, "n_valid_samples": 0})
            continue
        modal_center = float(pd.Series(np.round(centers, 0).astype(int)).mode().iloc[0])
        rows.append(
            {
                "window_id": window_id,
                "field": field,
                "metric": metric,
                "n_valid_samples": int(centers.size),
                "onset_median": float(np.nanmedian(onsets)) if onsets.size else np.nan,
                "onset_q025": float(np.nanpercentile(onsets, 2.5)) if onsets.size else np.nan,
                "onset_q975": float(np.nanpercentile(onsets, 97.5)) if onsets.size else np.nan,
                "center_median": float(np.nanmedian(centers)),
                "center_q025": float(np.nanpercentile(centers, 2.5)),
                "center_q975": float(np.nanpercentile(centers, 97.5)),
                "center_iqr": float(np.nanpercentile(centers, 75) - np.nanpercentile(centers, 25)),
                "center_modal_rounded_day": modal_center,
                "support_center_within_2d_of_modal": float(np.mean(np.abs(centers - modal_center) <= 2)),
                "support_center_within_3d_of_modal": float(np.mean(np.abs(centers - modal_center) <= 3)),
                "end_median": float(np.nanmedian(ends)) if ends.size else np.nan,
                "end_q025": float(np.nanpercentile(ends, 2.5)) if ends.size else np.nan,
                "end_q975": float(np.nanpercentile(ends, 97.5)) if ends.size else np.nan,
                "duration_median": float(np.nanmedian(durations)) if durations.size else np.nan,
                "duration_iqr": float(np.nanpercentile(durations, 75) - np.nanpercentile(durations, 25)) if durations.size else np.nan,
                "support_near_boundary": float(pd.Series(sub["near_left_boundary"]).astype(bool).mean() + pd.Series(sub["near_right_boundary"]).astype(bool).mean()) / 2.0,
                "support_multiple_active_segments": float(pd.Series(sub["has_multiple_active_segments"]).astype(bool).mean()) if "has_multiple_active_segments" in sub else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _metric_agreement(observed_df: pd.DataFrame, boot_df: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    rows = []
    cfg = settings.interval_timing
    for (window_id, field), sub in observed_df.groupby(["window_id", "field"], sort=False):
        det = sub[sub["metric"] == "detector_score"]
        con = sub[sub["metric"] == "prepost_contrast"]
        if det.empty or con.empty:
            continue
        dcenter = float(det["center_day"].iloc[0]) if pd.notna(det["center_day"].iloc[0]) else np.nan
        ccenter = float(con["center_day"].iloc[0]) if pd.notna(con["center_day"].iloc[0]) else np.nan
        diff = abs(dcenter - ccenter) if np.isfinite(dcenter) and np.isfinite(ccenter) else np.nan
        bsub = boot_df[(boot_df["window_id"].astype(str) == str(window_id)) & (boot_df["field"].astype(str) == str(field))]
        piv = bsub.pivot_table(index="bootstrap_id", columns="metric", values="center_day", aggfunc="first")
        corr = np.nan
        if "detector_score" in piv.columns and "prepost_contrast" in piv.columns:
            x = pd.to_numeric(piv["detector_score"], errors="coerce")
            y = pd.to_numeric(piv["prepost_contrast"], errors="coerce")
            good = x.notna() & y.notna()
            if int(good.sum()) >= 3:
                corr = float(np.corrcoef(x[good].to_numpy(dtype=float), y[good].to_numpy(dtype=float))[0, 1])
        if not np.isfinite(diff):
            label = "unavailable"
        elif diff <= float(cfg.metric_good_agreement_days):
            label = "good"
        elif diff <= float(cfg.metric_moderate_agreement_days):
            label = "moderate"
        else:
            label = "poor"
        rows.append(
            {
                "window_id": window_id,
                "field": field,
                "detector_center_observed": dcenter,
                "contrast_center_observed": ccenter,
                "center_difference_days": diff,
                "bootstrap_center_correlation": corr,
                "agreement_label": label,
            }
        )
    return pd.DataFrame(rows)


def _overlap_ratio(a_on, a_end, b_on, b_end) -> np.ndarray:
    inter = np.maximum(0, np.minimum(a_end, b_end) - np.maximum(a_on, b_on) + 1)
    union = np.maximum(a_end, b_end) - np.minimum(a_on, b_on) + 1
    return np.where(union > 0, inter / union, np.nan)


def _pair_interval_stats(piv: pd.DataFrame, window_id: str, metric: str, field_a: str, field_b: str, settings: StagePartitionV7Settings, id_col: str) -> dict:
    cfg = settings.interval_timing
    sub = piv[(piv["window_id"].astype(str) == str(window_id)) & (piv["metric"].astype(str) == str(metric))].copy()
    needed = [f"{field_a}_onset", f"{field_a}_center", f"{field_a}_end", f"{field_b}_onset", f"{field_b}_center", f"{field_b}_end"]
    if sub.empty or any(c not in sub.columns for c in needed):
        return {"n_valid": 0}
    vals = {c: pd.to_numeric(sub[c], errors="coerce") for c in needed}
    good = np.ones(len(sub), dtype=bool)
    for c in needed:
        good &= vals[c].notna().to_numpy()
    if int(good.sum()) == 0:
        return {"n_valid": 0}
    ao = vals[f"{field_a}_onset"][good].to_numpy(dtype=float)
    ac = vals[f"{field_a}_center"][good].to_numpy(dtype=float)
    ae = vals[f"{field_a}_end"][good].to_numpy(dtype=float)
    bo = vals[f"{field_b}_onset"][good].to_numpy(dtype=float)
    bc = vals[f"{field_b}_center"][good].to_numpy(dtype=float)
    be = vals[f"{field_b}_end"][good].to_numpy(dtype=float)
    lag_b_minus_a = bc - ac
    prob_a_center_before = float(np.mean(lag_b_minus_a > 0))
    prob_b_center_before = float(np.mean(lag_b_minus_a < 0))
    if prob_a_center_before >= prob_b_center_before:
        early, late = field_a, field_b
        early_on, early_c, early_end = ao, ac, ae
        late_on, late_c, late_end = bo, bc, be
        major_lag = lag_b_minus_a
        prob_center = prob_a_center_before
    else:
        early, late = field_b, field_a
        early_on, early_c, early_end = bo, bc, be
        late_on, late_c, late_end = ao, ac, ae
        major_lag = -lag_b_minus_a
        prob_center = prob_b_center_before
    overlap = _overlap_ratio(early_on, early_end, late_on, late_end)
    prob_separated = float(np.mean(early_end < late_on))
    prob_shifted = float(np.mean((early_c < late_c) & (overlap < float(cfg.shifted_overlap_max_ratio))))
    prob_sync = float(np.mean((overlap >= float(cfg.sync_overlap_ratio)) | (np.abs(major_lag) <= float(cfg.sync_center_radius_days))))
    return {
        "n_valid": int(good.sum()),
        "field_a": field_a,
        "field_b": field_b,
        "field_early_candidate": early,
        "field_late_candidate": late,
        "prob_a_center_before_b": prob_a_center_before,
        "prob_b_center_before_a": prob_b_center_before,
        "prob_center_early_before_late": float(prob_center),
        "prob_separated_early_before_late": prob_separated,
        "prob_shifted_early_before_late": prob_shifted,
        "prob_sync_or_overlap": prob_sync,
        "median_center_lag_late_minus_early": float(np.nanmedian(major_lag)),
        "q025_center_lag_late_minus_early": float(np.nanpercentile(major_lag, 2.5)),
        "q975_center_lag_late_minus_early": float(np.nanpercentile(major_lag, 97.5)),
        "iqr_center_lag_late_minus_early": float(np.nanpercentile(major_lag, 75) - np.nanpercentile(major_lag, 25)),
        "median_overlap_ratio": float(np.nanmedian(overlap)),
    }


def _pivot_interval_samples(samples: pd.DataFrame, id_col: str) -> pd.DataFrame:
    pieces = []
    idx = ["window_id", "metric", id_col]
    for var in ["onset_day", "center_day", "end_day", "near_left_boundary", "near_right_boundary"]:
        p = samples.pivot_table(index=idx, columns="field", values=var, aggfunc="first")
        p.columns = [f"{c}_{var.replace('_day','')}" for c in p.columns]
        pieces.append(p)
    return pd.concat(pieces, axis=1).reset_index()


def _assign_interval_order_label(row: pd.Series, settings: StagePartitionV7Settings) -> tuple[str, bool, str]:
    cfg = settings.interval_timing
    pc = float(row.get("prob_center_early_before_late", np.nan))
    ps = float(row.get("prob_separated_early_before_late", np.nan))
    psh = float(row.get("prob_shifted_early_before_late", np.nan))
    psync = float(row.get("prob_sync_or_overlap", np.nan))
    lag = float(row.get("median_center_lag_late_minus_early", np.nan))
    loyo = float(row.get("loyo_prob_center_early_before_late", np.nan))
    boundary = str(row.get("boundary_limited_flag", "none")) != "none"
    caution = []
    if boundary:
        caution.append("interval_near_analysis_boundary")
    if np.isfinite(psync) and psync >= float(cfg.sync_overlap_ratio):
        return "sync_or_overlap", True, ";".join(caution + ["active_intervals_overlap_do_not_force_order"])
    if boundary and np.isfinite(pc) and pc >= float(cfg.separated_min_center_prob) and np.isfinite(loyo) and loyo >= float(cfg.separated_min_loyo_center_prob):
        return "boundary_limited_interval_order", True, ";".join(caution + ["relative_direction_supported_but_interval_may_be_censored"])
    if (
        np.isfinite(pc) and pc >= float(cfg.separated_min_center_prob)
        and np.isfinite(ps) and ps >= float(cfg.separated_min_sep_prob)
        and np.isfinite(lag) and lag >= float(cfg.separated_min_median_lag_days)
        and np.isfinite(loyo) and loyo >= float(cfg.separated_min_loyo_center_prob)
    ):
        return "separated_order", True, ";".join(caution)
    if (
        np.isfinite(pc) and pc >= float(cfg.shifted_min_center_prob)
        and np.isfinite(psh) and psh >= float(cfg.shifted_min_shifted_prob)
        and np.isfinite(lag) and lag >= float(cfg.shifted_min_median_lag_days)
        and np.isfinite(loyo) and loyo >= float(cfg.shifted_min_loyo_center_prob)
    ):
        return "shifted_overlap_order", True, ";".join(caution)
    if (
        np.isfinite(pc) and pc >= float(cfg.moderate_min_center_prob)
        and np.isfinite(psh) and psh >= float(cfg.moderate_min_shifted_prob)
        and np.isfinite(lag) and lag >= float(cfg.moderate_min_median_lag_days)
        and np.isfinite(loyo) and loyo >= float(cfg.moderate_min_loyo_center_prob)
    ):
        return "moderate_shifted_order", True, ";".join(caution + ["candidate_interval_order_only"])
    return "ambiguous_interval_order", False, ";".join(caution + ["insufficient_or_conflicting_interval_order_support"])


def _pairwise_interval_orders(boot_df: pd.DataFrame, loyo_df: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    boot_piv = _pivot_interval_samples(boot_df, "bootstrap_id")
    loyo_piv = _pivot_interval_samples(loyo_df, "left_out_year_index")
    rows = []
    for window_id in sorted(boot_df["window_id"].astype(str).unique()):
        for metric in METRICS:
            for a, b in combinations(FIELDS, 2):
                bs = _pair_interval_stats(boot_piv, window_id, metric, a, b, settings, "bootstrap_id")
                if bs.get("n_valid", 0) <= 0:
                    continue
                ls = _pair_interval_stats(loyo_piv, window_id, metric, a, b, settings, "left_out_year_index")
                early = bs["field_early_candidate"]
                late = bs["field_late_candidate"]
                loyo_prob = np.nan
                if ls.get("n_valid", 0) > 0:
                    # Align LOYO direction with bootstrap majority direction.
                    if ls.get("field_early_candidate") == early and ls.get("field_late_candidate") == late:
                        loyo_prob = float(ls.get("prob_center_early_before_late", np.nan))
                    elif ls.get("field_early_candidate") == late and ls.get("field_late_candidate") == early:
                        loyo_prob = float(1.0 - ls.get("prob_center_early_before_late", np.nan))
                    else:
                        loyo_prob = np.nan
                boundary = "none"
                for f in [early, late]:
                    sub = boot_df[(boot_df["window_id"].astype(str) == str(window_id)) & (boot_df["metric"].astype(str) == str(metric)) & (boot_df["field"].astype(str) == str(f))]
                    if not sub.empty:
                        left = pd.Series(sub["near_left_boundary"]).astype(bool).mean()
                        right = pd.Series(sub["near_right_boundary"]).astype(bool).mean()
                        if left >= float(settings.interval_timing.boundary_support_threshold):
                            boundary = f"{f}_left_boundary"
                        if right >= float(settings.interval_timing.boundary_support_threshold):
                            boundary = f"{f}_right_boundary" if boundary == "none" else boundary + f";{f}_right_boundary"
                row = {
                    "window_id": window_id,
                    "metric": metric,
                    **bs,
                    "loyo_prob_center_early_before_late": loyo_prob,
                    "loyo_median_center_lag_late_minus_early": ls.get("median_center_lag_late_minus_early", np.nan),
                    "loyo_prob_sync_or_overlap": ls.get("prob_sync_or_overlap", np.nan),
                    "boundary_limited_flag": boundary,
                }
                label, usable, caution = _assign_interval_order_label(pd.Series(row), settings)
                row["interval_order_label"] = label
                row["usable_as_interval_order"] = bool(usable)
                row["caution"] = caution
                rows.append(row)
    return pd.DataFrame(rows)


def _field_order_scores(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = pair_df[pair_df["metric"] == "detector_score"].copy()
    for window_id, sub in main.groupby("window_id", sort=False):
        scores = {f: [] for f in FIELDS}
        for _, r in sub.iterrows():
            early = str(r["field_early_candidate"])
            late = str(r["field_late_candidate"])
            p = float(r["prob_center_early_before_late"])
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
            rows.append({"window_id": window_id, "field": f, "interval_order_score": score, "dominant_role": role})
    return pd.DataFrame(rows)


def _window_summary(pair_df: pd.DataFrame, score_df: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    rows = []
    main = pair_df[pair_df["metric"] == "detector_score"].copy()
    for window_id, sub in main.groupby("window_id", sort=False):
        n = int(len(sub))
        counts = sub["interval_order_label"].value_counts().to_dict()
        n_sep = int(counts.get("separated_order", 0))
        n_shift = int(counts.get("shifted_overlap_order", 0))
        n_mod = int(counts.get("moderate_shifted_order", 0))
        n_sync = int(counts.get("sync_or_overlap", 0))
        n_amb = int(counts.get("ambiguous_interval_order", 0))
        n_bound = int(counts.get("boundary_limited_interval_order", 0))
        order_pairs = n_sep + n_shift + n_mod + n_bound
        sync_frac = n_sync / n if n else np.nan
        amb_frac = n_amb / n if n else np.nan
        bound_frac = n_bound / n if n else np.nan
        if bound_frac >= 0.40 and order_pairs >= 2:
            cls = "boundary_limited_window"
        elif n_sep + n_shift >= int(settings.interval_timing.partial_order_min_edges):
            cls = "separated_or_shifted_partial_order"
        elif sync_frac >= float(settings.interval_timing.sync_window_fraction_threshold):
            cls = "sync_or_overlap_window"
        elif amb_frac >= float(settings.interval_timing.ambiguous_fraction_threshold) and order_pairs < 4:
            cls = "mixed_or_inseparable_window"
        elif order_pairs > 0:
            cls = "weak_partial_order_window"
        else:
            cls = "no_interval_order_evidence_window"
        ssub = score_df[score_df["window_id"].astype(str) == str(window_id)]
        early = ";".join(ssub[ssub["dominant_role"] == "early_tendency"]["field"].astype(str).tolist()) or "none"
        late = ";".join(ssub[ssub["dominant_role"] == "late_tendency"]["field"].astype(str).tolist()) or "none"
        middle = ";".join(ssub[ssub["dominant_role"] == "middle_or_uncertain"]["field"].astype(str).tolist()) or "none"
        if cls == "sync_or_overlap_window":
            result = "Most field active intervals overlap; do not force field order."
        elif cls == "mixed_or_inseparable_window":
            result = "Current interval timing does not support stable field-order separation."
        elif cls == "boundary_limited_window":
            result = "Some relative order exists, but active intervals are often boundary-limited."
        elif "partial_order" in cls or cls == "weak_partial_order_window":
            result = f"Partial interval timing structure: early={early}, late={late}, middle_or_uncertain={middle}."
        else:
            result = "No usable interval-order evidence under current criteria."
        rows.append(
            {
                "window_id": window_id,
                "n_total_pairs": n,
                "n_separated_order": n_sep,
                "n_shifted_overlap_order": n_shift,
                "n_moderate_shifted_order": n_mod,
                "n_sync_or_overlap": n_sync,
                "n_ambiguous_interval_order": n_amb,
                "n_boundary_limited_interval_order": n_bound,
                "order_pair_fraction": order_pairs / n if n else np.nan,
                "sync_pair_fraction": sync_frac,
                "ambiguous_pair_fraction": amb_frac,
                "boundary_limited_fraction": bound_frac,
                "early_group": early,
                "middle_or_uncertain_group": middle,
                "late_group": late,
                "window_interval_order_class": cls,
                "window_level_result": result,
                "caution": "timing_order_only_not_causality; detector_score_is_main_metric; contrast_metric_is_diagnostic",
            }
        )
    return pd.DataFrame(rows)


def _write_audit_log(log_root: Path, settings: StagePartitionV7Settings, windows: pd.DataFrame, run_meta: dict) -> None:
    txt = [
        "# field_transition_interval_timing_v7_d audit log",
        "",
        f"created_utc: {now_utc()}",
        "",
        "## Scope",
        "- Answers only the first question: for each accepted transition window, when each field is most actively transitioning.",
        "- Changes the observation target from argmax peak_day to active intervals: onset / center / end / duration.",
        "- Does not infer causality, does not analyze spatial earliest regions, and does not read downstream pathway or lead-lag outputs.",
        "",
        "## Accepted windows",
        windows.to_string(index=False),
        "",
        "## Method notes",
        f"- analysis_radius_days = {settings.interval_timing.analysis_radius_days}",
        f"- prepost_k_days = {settings.interval_timing.prepost_k_days}",
        f"- n_bootstrap = {settings.bootstrap.effective_n_bootstrap()}",
        "- detector_score is the main metric inherited from stage_partition; prepost_contrast is an independent diagnostic metric.",
        "- excluded candidates are flagged if close to an active interval but are not reintroduced as main windows.",
        "",
        "## Run meta",
        json.dumps(run_meta, indent=2, default=str),
    ]
    (log_root / "field_transition_interval_timing_v7_d_audit_log.md").write_text("\n".join(txt), encoding="utf-8")


def _try_write_plots(output_root: Path, figure_root: Path, observed_df: pd.DataFrame, pair_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        # Simple order heatmaps for detector_score only.
        for window_id, sub in pair_df[pair_df["metric"] == "detector_score"].groupby("window_id"):
            mat = pd.DataFrame(np.nan, index=FIELDS, columns=FIELDS, dtype=float)
            for _, r in sub.iterrows():
                early = str(r["field_early_candidate"])
                late = str(r["field_late_candidate"])
                p = float(r["prob_center_early_before_late"])
                if early in mat.index and late in mat.columns:
                    mat.loc[early, late] = p
                    mat.loc[late, early] = 1.0 - p
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat.to_numpy(dtype=float), vmin=0, vmax=1)
            ax.set_xticks(range(len(FIELDS)), FIELDS)
            ax.set_yticks(range(len(FIELDS)), FIELDS)
            ax.set_title(f"{window_id} interval center order probability\nTiming order only, not causality")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(figure_root / f"{window_id}_pairwise_interval_order_heatmap_v7_d.png", dpi=160)
            plt.close(fig)
    except Exception as exc:
        (output_root / "plot_warning_v7_d.txt").write_text(str(exc), encoding="utf-8")


def run_field_transition_interval_timing_v7_d(settings: StagePartitionV7Settings | None = None) -> None:
    settings = settings or StagePartitionV7Settings()
    dirs = _prepare_dirs(settings)
    output_root = dirs["output_root"]
    log_root = dirs["log_root"]
    figure_root = dirs["figure_root"]
    settings.write_json(log_root / "config_used.json")

    audit = _audit_accepted_points(settings)
    write_json(audit, log_root / "accepted_window_evidence.json")
    windows_source = audit["windows_df"]

    smoothed = load_smoothed_fields(settings.foundation.smoothed_fields_path())
    profiles = build_profiles(smoothed, settings.profile)
    first_cube = np.asarray(profiles[FIELDS[0]].raw_cube)
    n_years = int(first_cube.shape[0])
    n_days = int(first_cube.shape[1])

    windows = _build_interval_analysis_windows(windows_source, settings, n_days=n_days)
    write_dataframe(windows, output_root / "accepted_windows_used_v7_d.csv")

    joint_state = build_state_matrix(profiles, settings.state)
    shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)

    observed_rows = []
    observed_curves = []
    for field in FIELDS:
        detector_curve, contrast_curve = _curves_for_field(profiles, field, None, shared_valid_day_index, n_days, settings)
        for metric, curve in [("detector_score", detector_curve), ("prepost_contrast", contrast_curve)]:
            for _, w in windows.iterrows():
                rec = _extract_active_interval(curve, w, field=field, metric=metric, settings=settings)
                observed_rows.append(rec)
                sub = curve[(curve.index.astype(int) >= int(w["analysis_window_start"])) & (curve.index.astype(int) <= int(w["analysis_window_end"]))].dropna()
                for d, v in sub.items():
                    observed_curves.append({"window_id": str(w["window_id"]), "anchor_day": int(w["anchor_day"]), "field": field, "metric": metric, "day": int(d), "curve_value": float(v)})
    observed_df = pd.DataFrame(observed_rows)
    observed_curves_df = pd.DataFrame(observed_curves)
    write_dataframe(observed_df, output_root / "field_transition_interval_observed_v7_d.csv")
    write_dataframe(observed_curves_df, output_root / "field_transition_interval_observed_curves_long_v7_d.csv")

    boot_indices = _load_or_make_bootstrap_indices(profiles, settings, output_root)
    boot_rows = []
    for b, year_idx in enumerate(boot_indices):
        if b % int(settings.bootstrap.progress_every) == 0:
            print(f"[V7-d bootstrap] {b}/{len(boot_indices)}")
        for field in FIELDS:
            detector_curve, contrast_curve = _curves_for_field(profiles, field, year_idx, shared_valid_day_index, n_days, settings)
            for metric, curve in [("detector_score", detector_curve), ("prepost_contrast", contrast_curve)]:
                for _, w in windows.iterrows():
                    boot_rows.append(_extract_active_interval(curve, w, field=field, metric=metric, settings=settings, sample_id_name="bootstrap_id", sample_id_value=b))
    boot_df = pd.DataFrame(boot_rows)
    if settings.interval_timing.write_sample_long_tables:
        write_dataframe(boot_df, output_root / "field_transition_interval_bootstrap_samples_v7_d.csv")

    loyo_rows = []
    years = getattr(smoothed, "years", None)
    if years is None and isinstance(smoothed, dict):
        years = smoothed.get("years")
    if years is None:
        years = list(range(n_years))
    years = np.asarray(years)
    for left in range(n_years):
        if left % max(1, int(settings.bootstrap.progress_every // 5)) == 0:
            print(f"[V7-d LOYO] {left}/{n_years}")
        keep = np.asarray([i for i in range(n_years) if i != left], dtype=int)
        for field in FIELDS:
            detector_curve, contrast_curve = _curves_for_field(profiles, field, keep, shared_valid_day_index, n_days, settings)
            for metric, curve in [("detector_score", detector_curve), ("prepost_contrast", contrast_curve)]:
                for _, w in windows.iterrows():
                    rec = _extract_active_interval(curve, w, field=field, metric=metric, settings=settings, sample_id_name="left_out_year_index", sample_id_value=left)
                    rec["left_out_year"] = int(years[left]) if left < len(years) and pd.notna(years[left]) else int(left)
                    loyo_rows.append(rec)
    loyo_df = pd.DataFrame(loyo_rows)
    if settings.interval_timing.write_sample_long_tables:
        write_dataframe(loyo_df, output_root / "field_transition_interval_loyo_samples_v7_d.csv")

    boot_summary = _summarize_interval_samples(boot_df)
    write_dataframe(boot_summary, output_root / "field_transition_interval_bootstrap_summary_v7_d.csv")
    loyo_summary = _summarize_interval_samples(loyo_df).rename(columns={c: f"loyo_{c}" for c in []})
    write_dataframe(loyo_summary, output_root / "field_transition_interval_loyo_summary_v7_d.csv")

    agreement = _metric_agreement(observed_df, boot_df, settings)
    write_dataframe(agreement, output_root / "field_transition_interval_metric_agreement_v7_d.csv")

    pair_df = _pairwise_interval_orders(boot_df, loyo_df, settings)
    write_dataframe(pair_df, output_root / "pairwise_interval_order_summary_v7_d.csv")

    score_df = _field_order_scores(pair_df)
    write_dataframe(score_df, output_root / "field_interval_order_rank_summary_v7_d.csv")

    window_summary = _window_summary(pair_df, score_df, settings)
    write_dataframe(window_summary, output_root / "window_interval_orderability_summary_v7_d.csv")

    detector_pair = pair_df[pair_df["metric"] == "detector_score"].copy()
    edges = detector_pair[detector_pair["usable_as_interval_order"].astype(bool)].copy()
    if not edges.empty:
        edges = edges.rename(columns={"field_early_candidate": "source_field", "field_late_candidate": "target_field", "interval_order_label": "edge_type"})
    write_dataframe(edges, output_root / "interval_order_graph_edges_v7_d.csv")

    summary = {
        "status": "success",
        "created_utc": now_utc(),
        "run_label": settings.interval_timing.output_tag,
        "accepted_peak_days": [int(x) for x in settings.accepted_windows.accepted_peak_days],
        "excluded_candidate_days": [int(x) for x in settings.accepted_windows.excluded_candidate_days],
        "n_windows": int(len(windows)),
        "n_bootstrap": int(settings.bootstrap.effective_n_bootstrap()),
        "n_interval_observed_rows": int(len(observed_df)),
        "n_bootstrap_rows": int(len(boot_df)),
        "n_loyo_rows": int(len(loyo_df)),
        "window_class_counts": window_summary["window_interval_order_class"].value_counts().to_dict() if not window_summary.empty else {},
        "pair_label_counts_detector_score": detector_pair["interval_order_label"].value_counts().to_dict() if not detector_pair.empty else {},
        "method": "active_interval_timing_from_detector_score_and_prepost_contrast; timing order only, not causality",
    }
    write_json(summary, output_root / "summary.json")
    write_json(summary, log_root / "summary.json")
    run_meta = {
        **summary,
        "input_smoothed_fields_path": str(settings.foundation.smoothed_fields_path()),
        "source_logs_checked": [str(settings.source.v6_update_log()), str(settings.source.v6_1_update_log())],
        "source_tables_checked": [str(settings.source.v6_bootstrap_summary_path()), str(settings.source.v6_1_windows_path())],
        "notes": [
            "V7-d is an experiment-level change, not a result-picking layer.",
            "It replaces single-day argmax peak timing with transition active intervals.",
            "Detector score is the main inherited metric; pre/post contrast is an independent diagnostic metric.",
            "No downstream lead-lag/pathway outputs are used.",
        ],
    }
    write_json(run_meta, output_root / "run_meta.json")
    write_json(run_meta, log_root / "run_meta.json")
    _write_audit_log(log_root, settings, windows, run_meta)
    if settings.interval_timing.write_plots:
        _try_write_plots(output_root, figure_root, observed_df, pair_df)

    print(f"[V7-d] done: {output_root}")
