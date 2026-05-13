from __future__ import annotations

from itertools import combinations
from math import exp, lgamma, log
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import StagePartitionV7Settings
FIELDS = ["P", "V", "H", "Je", "Jw"]
from .report import now_utc, write_dataframe, write_json


REQUIRED_V7B_FILES = {
    "bootstrap_samples": "field_transition_peak_days_bootstrap_samples_v7_b.csv",
    "loyo_samples": "field_transition_peak_days_loyo_samples_v7_b.csv",
    "main_table": "field_transition_peak_days_bootstrap_v7_b.csv",
    "accepted_windows": "accepted_windows_used_v7_b.csv",
    "run_meta": "run_meta.json",
    "summary": "summary.json",
}


def _prepare_dirs(settings: StagePartitionV7Settings) -> dict[str, Path]:
    out = settings.pairwise_output_root()
    log_dir = settings.pairwise_log_root()
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return {"output_root": out, "log_root": log_dir}


def _safe_read_json(path: Path) -> dict:
    import json

    if not path.exists():
        return {"missing": True, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": str(exc), "path": str(path)}


def _check_v7b_inputs(v7b_root: Path) -> dict:
    evidence = {"v7b_root": str(v7b_root), "required_files": {}}
    for key, fname in REQUIRED_V7B_FILES.items():
        path = v7b_root / fname
        evidence["required_files"][key] = {"path": str(path), "exists": path.exists()}
    evidence["run_meta"] = _safe_read_json(v7b_root / REQUIRED_V7B_FILES["run_meta"])
    evidence["summary"] = _safe_read_json(v7b_root / REQUIRED_V7B_FILES["summary"])
    return evidence


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required V7-b {label}: {path}")
    return path


def _bh_fdr(pvals: Iterable[float]) -> list[float]:
    arr = np.asarray([np.nan if v is None else float(v) for v in pvals], dtype=float)
    q = np.full(arr.shape, np.nan, dtype=float)
    valid = np.where(np.isfinite(arr))[0]
    if valid.size == 0:
        return q.tolist()
    order = valid[np.argsort(arr[valid])]
    m = float(valid.size)
    prev = 1.0
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = valid.size - rank_rev + 1
        val = min(prev, arr[idx] * m / rank)
        prev = val
        q[idx] = min(1.0, val)
    return q.tolist()


def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P[X >= k] for Binomial(n,p)."""
    k = int(k)
    n = int(n)
    p = float(p)
    if n <= 0:
        return np.nan
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0 if k <= n else 0.0
    try:
        from scipy.stats import binom

        return float(binom.sf(k - 1, n, p))
    except Exception:
        logs = []
        lp = log(p)
        lq = log(1.0 - p)
        for x in range(k, n + 1):
            logs.append(lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1) + x * lp + (n - x) * lq)
        m = max(logs)
        return float(min(1.0, exp(m) * sum(exp(v - m) for v in logs)))


def _direction_p_value(k_major: int, n_strict: int) -> float:
    if n_strict <= 0:
        return 1.0
    # Two-sided exact-ish binomial test for deviation from 50/50 among non-tied samples.
    return float(min(1.0, 2.0 * _binom_sf(int(k_major), int(n_strict), 0.5)))


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _get_field_meta(main_df: pd.DataFrame, window_id: str, field: str) -> dict:
    sub = main_df[(main_df["window_id"].astype(str) == str(window_id)) & (main_df["field"].astype(str) == str(field))]
    if sub.empty:
        return {}
    r = sub.iloc[0].to_dict()
    start = int(r.get("analysis_window_start", np.nan)) if pd.notna(r.get("analysis_window_start", np.nan)) else None
    end = int(r.get("analysis_window_end", np.nan)) if pd.notna(r.get("analysis_window_end", np.nan)) else None
    edge_margin = 1
    obs = r.get("observed_peak_day", np.nan)
    modal = r.get("bootstrap_modal_peak_day", np.nan)
    timing_label = str(r.get("timing_confidence_label", ""))

    side_flags = []
    if start is not None and pd.notna(obs) and float(obs) <= start + edge_margin:
        side_flags.append("observed_left_edge")
    if end is not None and pd.notna(obs) and float(obs) >= end - edge_margin:
        side_flags.append("observed_right_edge")
    if start is not None and pd.notna(modal) and float(modal) <= start + edge_margin:
        side_flags.append("modal_left_edge")
    if end is not None and pd.notna(modal) and float(modal) >= end - edge_margin:
        side_flags.append("modal_right_edge")
    if timing_label == "boundary_truncated" and not side_flags:
        side_flags.append("boundary_truncated_unknown_side")

    return {
        "timing_confidence_label": timing_label,
        "caution": str(r.get("caution", "")),
        "observed_peak_day": r.get("observed_peak_day", np.nan),
        "bootstrap_modal_peak_day": r.get("bootstrap_modal_peak_day", np.nan),
        "support_near_analysis_edge": r.get("support_near_analysis_edge", np.nan),
        "boundary_sides": ";".join(side_flags) if side_flags else "none",
        "has_boundary_censoring": bool(side_flags or timing_label == "boundary_truncated"),
    }


def _pivot_samples(samples: pd.DataFrame, id_col: str) -> pd.DataFrame:
    needed = [id_col, "window_id", "field", "peak_day", "near_analysis_edge"]
    missing = [c for c in needed if c not in samples.columns]
    if missing:
        raise ValueError(f"Missing columns in sample table: {missing}")
    day = samples.pivot_table(index=["window_id", id_col], columns="field", values="peak_day", aggfunc="first")
    edge = samples.pivot_table(index=["window_id", id_col], columns="field", values="near_analysis_edge", aggfunc="first")
    # Flatten with prefixes.
    day.columns = [f"{c}_day" for c in day.columns]
    edge.columns = [f"{c}_near_edge" for c in edge.columns]
    return day.join(edge, how="left").reset_index()


def _pair_stats_from_pivot(pivot: pd.DataFrame, window_id: str, field_a: str, field_b: str) -> dict:
    sub = pivot[pivot["window_id"].astype(str) == str(window_id)].copy()
    a_col = f"{field_a}_day"
    b_col = f"{field_b}_day"
    if a_col not in sub.columns or b_col not in sub.columns or sub.empty:
        return {"n_valid": 0}
    a = pd.to_numeric(sub[a_col], errors="coerce")
    b = pd.to_numeric(sub[b_col], errors="coerce")
    valid = a.notna() & b.notna()
    a = a[valid].astype(float).to_numpy()
    b = b[valid].astype(float).to_numpy()
    if a.size == 0:
        return {"n_valid": 0}
    lag_b_minus_a = b - a
    n = int(a.size)
    n_a_before = int(np.sum(lag_b_minus_a > 0))
    n_b_before = int(np.sum(lag_b_minus_a < 0))
    n_tie = int(np.sum(lag_b_minus_a == 0))
    n_strict = int(n_a_before + n_b_before)
    prob_a_before = float(n_a_before / n)
    prob_b_before = float(n_b_before / n)
    prob_a_not_later = float(np.mean(lag_b_minus_a >= 0))
    prob_b_not_later = float(np.mean(lag_b_minus_a <= 0))
    prob_sync_0 = float(np.mean(np.abs(lag_b_minus_a) <= 0))
    prob_sync_1 = float(np.mean(np.abs(lag_b_minus_a) <= 1))
    prob_sync_2 = float(np.mean(np.abs(lag_b_minus_a) <= 2))

    if prob_a_before >= prob_b_before:
        early = field_a
        late = field_b
        major_lag = lag_b_minus_a
        prob_major = prob_a_before
        prob_major_not_later = prob_a_not_later
    else:
        early = field_b
        late = field_a
        major_lag = -lag_b_minus_a
        prob_major = prob_b_before
        prob_major_not_later = prob_b_not_later

    k_major = max(n_a_before, n_b_before)
    p_val = _direction_p_value(k_major, n_strict)

    return {
        "n_valid": n,
        "n_strict_untied": n_strict,
        "n_tied": n_tie,
        "field_a": field_a,
        "field_b": field_b,
        "field_early_candidate": early,
        "field_late_candidate": late,
        "prob_a_before_b": prob_a_before,
        "prob_b_before_a": prob_b_before,
        "prob_a_not_later_b": prob_a_not_later,
        "prob_b_not_later_a": prob_b_not_later,
        "prob_early_before_late": float(prob_major),
        "prob_early_before_late_1d": float(np.mean(major_lag >= 1)),
        "prob_early_before_late_2d": float(np.mean(major_lag >= 2)),
        "prob_early_not_later_than_late": float(prob_major_not_later),
        "prob_sync_0d": prob_sync_0,
        "prob_sync_1d": prob_sync_1,
        "prob_sync_2d": prob_sync_2,
        "median_lag_late_minus_early": float(np.nanmedian(major_lag)),
        "q025_lag_late_minus_early": float(np.nanpercentile(major_lag, 2.5)),
        "q975_lag_late_minus_early": float(np.nanpercentile(major_lag, 97.5)),
        "iqr_lag_late_minus_early": float(np.nanpercentile(major_lag, 75) - np.nanpercentile(major_lag, 25)),
        "median_lag_b_minus_a": float(np.nanmedian(lag_b_minus_a)),
        "q025_lag_b_minus_a": float(np.nanpercentile(lag_b_minus_a, 2.5)),
        "q975_lag_b_minus_a": float(np.nanpercentile(lag_b_minus_a, 97.5)),
        "order_p_value": p_val,
        "major_direction": f"{early}<{late}",
    }


def _assign_order_label(row: pd.Series, cfg) -> tuple[str, bool, str]:
    p = float(row.get("prob_early_before_late", np.nan))
    p1 = float(row.get("prob_early_before_late_1d", np.nan))
    p2 = float(row.get("prob_early_before_late_2d", np.nan))
    med = float(row.get("median_lag_late_minus_early", np.nan))
    q = float(row.get("order_q_value", np.nan))
    loyo_p = float(row.get("loyo_prob_early_before_late", np.nan))
    sync2 = float(row.get("prob_sync_2d", np.nan))
    boundary = str(row.get("boundary_censoring_flag", "none")) != "none"
    caution = []
    if boundary:
        caution.append("boundary_censoring_affects_exact_peak_day")
    if np.isfinite(sync2) and sync2 >= cfg.sync_min_prob_within_2d and abs(med) <= cfg.sync_max_abs_median_lag_days and p2 < cfg.sync_max_prob_by_2d:
        return "sync_or_overlap", False, ";".join(caution + ["do_not_force_order_overlapping_peaks"])
    if np.isfinite(p) and p >= cfg.conflict_bootstrap_prob_threshold and np.isfinite(loyo_p) and loyo_p < cfg.conflict_loyo_prob_threshold:
        return "conflict_order", False, ";".join(caution + ["bootstrap_and_loyo_order_support_conflict"])

    if boundary:
        if (
            np.isfinite(p) and p >= cfg.robust_censored_min_prob
            and np.isfinite(p1) and p1 >= cfg.robust_censored_min_prob_by_1d
            and np.isfinite(med) and med >= cfg.robust_censored_min_median_abs_lag_days
            and np.isfinite(q) and q < cfg.robust_censored_max_q_value
            and np.isfinite(loyo_p) and loyo_p >= cfg.robust_censored_min_loyo_prob
        ):
            return "robust_censored_order", True, ";".join(caution + ["relative_order_supported_but_exact_day_censored"])
    else:
        if (
            np.isfinite(p) and p >= cfg.robust_min_prob
            and np.isfinite(p1) and p1 >= cfg.robust_min_prob_by_1d
            and np.isfinite(med) and med >= cfg.robust_min_median_abs_lag_days
            and np.isfinite(q) and q < cfg.robust_max_q_value
            and np.isfinite(loyo_p) and loyo_p >= cfg.robust_min_loyo_prob
        ):
            return "robust_order", True, "relative_order_supported_not_causal"

    if (
        np.isfinite(p) and p >= cfg.moderate_min_prob
        and np.isfinite(p1) and p1 >= cfg.moderate_min_prob_by_1d
        and np.isfinite(med) and med >= cfg.moderate_min_median_abs_lag_days
        and np.isfinite(q) and q < cfg.moderate_max_q_value
        and np.isfinite(loyo_p) and loyo_p >= cfg.moderate_min_loyo_prob
    ):
        return "moderate_order", True, ";".join(caution + ["candidate_relative_order_use_with_caution"])

    return "ambiguous_order", False, ";".join(caution + ["relative_order_not_sufficiently_supported"])


def _build_pairwise_tables(
    bootstrap_samples: pd.DataFrame,
    loyo_samples: pd.DataFrame,
    main_df: pd.DataFrame,
    settings: StagePartitionV7Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    boot_pivot = _pivot_samples(bootstrap_samples, "bootstrap_id")
    loyo_pivot = _pivot_samples(loyo_samples, "left_out_year_index")
    rows = []
    all_dir_rows = []
    windows = list(dict.fromkeys(bootstrap_samples["window_id"].astype(str).tolist()))
    for window_id in windows:
        for field_a, field_b in combinations(list(FIELDS), 2):
            boot = _pair_stats_from_pivot(boot_pivot, window_id, field_a, field_b)
            loyo = _pair_stats_from_pivot(loyo_pivot, window_id, field_a, field_b)
            if boot.get("n_valid", 0) <= 0:
                continue
            early = boot["field_early_candidate"]
            late = boot["field_late_candidate"]
            early_meta = _get_field_meta(main_df, window_id, early)
            late_meta = _get_field_meta(main_df, window_id, late)
            boundary_parts = []
            if early_meta.get("has_boundary_censoring", False):
                boundary_parts.append(f"early:{early_meta.get('boundary_sides','boundary')}")
            if late_meta.get("has_boundary_censoring", False):
                boundary_parts.append(f"late:{late_meta.get('boundary_sides','boundary')}")
            boundary_flag = ";".join(boundary_parts) if boundary_parts else "none"

            # LOYO support must be expressed in the bootstrap-majority direction.
            if early == field_a:
                loyo_prob_major = loyo.get("prob_a_before_b", np.nan)
                loyo_prob_major_1d = loyo.get("prob_early_before_late_1d", np.nan) if loyo.get("field_early_candidate") == field_a else np.nan
                loyo_median_lag = loyo.get("median_lag_b_minus_a", np.nan)
            else:
                loyo_prob_major = loyo.get("prob_b_before_a", np.nan)
                # If early==field_b, lag late-early is A-B = -lag_b_minus_a.
                loyo_median_lag = -float(loyo.get("median_lag_b_minus_a", np.nan)) if np.isfinite(loyo.get("median_lag_b_minus_a", np.nan)) else np.nan
                # compute by 1d using loyo majority-oriented fields if it agrees, otherwise derive from raw direction prob unavailable here.
                loyo_prob_major_1d = loyo.get("prob_early_before_late_1d", np.nan) if loyo.get("field_early_candidate") == field_b else np.nan

            row = {
                "window_id": window_id,
                "anchor_day": int(bootstrap_samples[bootstrap_samples["window_id"].astype(str) == str(window_id)]["anchor_day"].iloc[0]),
                "field_a": field_a,
                "field_b": field_b,
                **boot,
                "loyo_prob_early_before_late": float(loyo_prob_major) if np.isfinite(loyo_prob_major) else np.nan,
                "loyo_prob_early_before_late_1d": float(loyo_prob_major_1d) if np.isfinite(loyo_prob_major_1d) else np.nan,
                "loyo_prob_sync_2d": float(loyo.get("prob_sync_2d", np.nan)) if np.isfinite(loyo.get("prob_sync_2d", np.nan)) else np.nan,
                "loyo_median_lag_late_minus_early": float(loyo_median_lag) if np.isfinite(loyo_median_lag) else np.nan,
                "early_field_timing_label_v7_b": early_meta.get("timing_confidence_label", "missing"),
                "late_field_timing_label_v7_b": late_meta.get("timing_confidence_label", "missing"),
                "early_field_boundary_sides": early_meta.get("boundary_sides", "missing"),
                "late_field_boundary_sides": late_meta.get("boundary_sides", "missing"),
                "boundary_censoring_flag": boundary_flag,
            }
            rows.append(row)
            all_dir_rows.append(
                {
                    "window_id": window_id,
                    "anchor_day": row["anchor_day"],
                    "field_a": field_a,
                    "field_b": field_b,
                    "prob_a_before_b": boot.get("prob_a_before_b", np.nan),
                    "prob_b_before_a": boot.get("prob_b_before_a", np.nan),
                    "prob_sync_2d": boot.get("prob_sync_2d", np.nan),
                    "median_lag_b_minus_a": boot.get("median_lag_b_minus_a", np.nan),
                    "q025_lag_b_minus_a": boot.get("q025_lag_b_minus_a", np.nan),
                    "q975_lag_b_minus_a": boot.get("q975_lag_b_minus_a", np.nan),
                    "order_direction_majority": boot.get("major_direction", ""),
                    "n_valid": boot.get("n_valid", 0),
                    "n_strict_untied": boot.get("n_strict_untied", 0),
                    "n_tied": boot.get("n_tied", 0),
                }
            )
    pair_df = pd.DataFrame(rows)
    all_dir_df = pd.DataFrame(all_dir_rows)
    if pair_df.empty:
        return pair_df, all_dir_df
    pair_df["order_q_value"] = _bh_fdr(pair_df["order_p_value"].tolist())
    labels = pair_df.apply(lambda r: _assign_order_label(r, settings.pairwise_order), axis=1)
    pair_df["order_confidence_label"] = [x[0] for x in labels]
    pair_df["usable_as_order_evidence"] = [bool(x[1]) for x in labels]
    pair_df["caution"] = [x[2] for x in labels]
    return pair_df, all_dir_df


def _build_window_partial_order_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pair_df is None or pair_df.empty:
        return pd.DataFrame()

    def fmt_edge(r: pd.Series) -> str:
        return f"{r['field_early_candidate']}<{r['field_late_candidate']} (p={float(r['prob_early_before_late']):.2f}, lag={float(r['median_lag_late_minus_early']):.1f}d)"

    for window_id, sub in pair_df.groupby("window_id", sort=False):
        anchor = int(sub["anchor_day"].iloc[0])
        robust = [fmt_edge(r) for _, r in sub[sub["order_confidence_label"] == "robust_order"].iterrows()]
        cens = [fmt_edge(r) for _, r in sub[sub["order_confidence_label"] == "robust_censored_order"].iterrows()]
        moderate = [fmt_edge(r) for _, r in sub[sub["order_confidence_label"] == "moderate_order"].iterrows()]
        sync = [f"{r['field_a']}~{r['field_b']} (sync2={float(r['prob_sync_2d']):.2f})" for _, r in sub[sub["order_confidence_label"] == "sync_or_overlap"].iterrows()]
        ambiguous_n = int((sub["order_confidence_label"] == "ambiguous_order").sum())
        conflict_n = int((sub["order_confidence_label"] == "conflict_order").sum())
        pieces = []
        if robust:
            pieces.append("robust: " + "; ".join(robust))
        if cens:
            pieces.append("censored: " + "; ".join(cens))
        if moderate:
            pieces.append("moderate: " + "; ".join(moderate))
        if not pieces:
            pieces.append("no robust/moderate pairwise timing order")
        rows.append(
            {
                "window_id": window_id,
                "anchor_day": anchor,
                "robust_orders": "; ".join(robust),
                "robust_censored_orders": "; ".join(cens),
                "moderate_orders": "; ".join(moderate),
                "sync_or_overlap_pairs": "; ".join(sync),
                "n_ambiguous_pairs": ambiguous_n,
                "n_conflict_pairs": conflict_n,
                "interpretable_partial_order": " | ".join(pieces),
                "caution": "timing_order_only_not_causality; censored orders do not locate exact peak day",
            }
        )
    return pd.DataFrame(rows)


def _build_edge_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df is None or pair_df.empty:
        return pd.DataFrame()
    keep = pair_df[pair_df["order_confidence_label"].isin(["robust_order", "robust_censored_order", "moderate_order"])].copy()
    if keep.empty:
        return pd.DataFrame(
            columns=["window_id", "anchor_day", "source_field", "target_field", "edge_type", "prob_source_before_target", "median_lag", "order_confidence_label", "boundary_censoring_flag", "caution"]
        )
    return pd.DataFrame(
        [
            {
                "window_id": r["window_id"],
                "anchor_day": int(r["anchor_day"]),
                "source_field": r["field_early_candidate"],
                "target_field": r["field_late_candidate"],
                "edge_type": "timing_order_not_causal",
                "prob_source_before_target": r["prob_early_before_late"],
                "median_lag": r["median_lag_late_minus_early"],
                "order_confidence_label": r["order_confidence_label"],
                "boundary_censoring_flag": r["boundary_censoring_flag"],
                "caution": r["caution"],
            }
            for _, r in keep.iterrows()
        ]
    )


def _build_rank_summary(all_dir_df: pd.DataFrame) -> pd.DataFrame:
    if all_dir_df is None or all_dir_df.empty:
        return pd.DataFrame()
    rows = []
    fields = list(FIELDS)
    for window_id, sub in all_dir_df.groupby("window_id", sort=False):
        anchor = int(sub["anchor_day"].iloc[0])
        for field in fields:
            probs = []
            for other in fields:
                if other == field:
                    continue
                r1 = sub[(sub["field_a"] == field) & (sub["field_b"] == other)]
                if not r1.empty:
                    probs.append(float(r1["prob_a_before_b"].iloc[0]))
                    continue
                r2 = sub[(sub["field_a"] == other) & (sub["field_b"] == field)]
                if not r2.empty:
                    probs.append(float(r2["prob_b_before_a"].iloc[0]))
            order_score = float(np.nanmean(probs)) if probs else np.nan
            rows.append(
                {
                    "window_id": window_id,
                    "anchor_day": anchor,
                    "field": field,
                    "order_score_mean_prob_earlier_than_others": order_score,
                    "n_pairwise_comparisons": len(probs),
                    "dominant_role_hint": "earlier_tending" if np.isfinite(order_score) and order_score >= 0.65 else ("later_tending" if np.isfinite(order_score) and order_score <= 0.35 else "mixed_or_middle"),
                    "caution": "rank summary is auxiliary; use pairwise edge table for evidence",
                }
            )
    return pd.DataFrame(rows)


def _try_write_plots(pair_df: pd.DataFrame, all_dir_df: pd.DataFrame, out_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fields = list(FIELDS)
    for window_id, sub in all_dir_df.groupby("window_id", sort=False):
        mat = pd.DataFrame(np.nan, index=fields, columns=fields)
        for f in fields:
            mat.loc[f, f] = 0.5
        for _, r in sub.iterrows():
            a = r["field_a"]
            b = r["field_b"]
            pa = float(r["prob_a_before_b"])
            pb = float(r["prob_b_before_a"])
            mat.loc[a, b] = pa
            mat.loc[b, a] = pb
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(111)
        im = ax.imshow(mat.to_numpy(dtype=float), vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(fields)))
        ax.set_yticks(range(len(fields)))
        ax.set_xticklabels(fields)
        ax.set_yticklabels(fields)
        ax.set_title(f"{window_id}: P(row earlier than column)\nTiming order only, not causality")
        for i in range(len(fields)):
            for j in range(len(fields)):
                val = mat.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{window_id}_pairwise_order_probability_heatmap_v7_c.png", dpi=180)
        plt.close(fig)

        edges = pair_df[(pair_df["window_id"].astype(str) == str(window_id)) & (pair_df["order_confidence_label"].isin(["robust_order", "robust_censored_order", "moderate_order"]))]
        if not edges.empty:
            fig = plt.figure(figsize=(8.5, 4.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            lines = [f"{r['field_early_candidate']} → {r['field_late_candidate']}  {r['order_confidence_label']}  p={float(r['prob_early_before_late']):.2f}  lag={float(r['median_lag_late_minus_early']):.1f}d" for _, r in edges.iterrows()]
            ax.text(0.01, 0.98, f"{window_id} partial timing orders\nTiming order only, not causality\n\n" + "\n".join(lines), ha="left", va="top", family="monospace", fontsize=9)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{window_id}_partial_order_edges_v7_c.png", dpi=180)
            plt.close(fig)


def _write_audit_log(path: Path, evidence: dict, settings: StagePartitionV7Settings) -> None:
    text = [
        "# V7-c pairwise field-order audit log",
        "",
        "## Scope",
        "",
        "V7-c reads V7-b bootstrap/LOYO peak-day samples and audits pairwise relative timing order between fields.",
        "It does not rerun detector, does not infer causality, and does not analyze spatial earliest regions.",
        "",
        "## Input evidence checked",
        "",
        f"V7-b root: {evidence.get('v7b_root')}",
        "",
        "Required files:",
    ]
    for key, item in evidence.get("required_files", {}).items():
        text.append(f"- {key}: exists={item.get('exists')} path={item.get('path')}")
    text += [
        "",
        "## V7-b status evidence",
        "",
        f"run_meta.status: {evidence.get('run_meta', {}).get('status')}",
        f"run_meta.accepted_peak_days: {evidence.get('run_meta', {}).get('accepted_peak_days')}",
        f"run_meta.excluded_candidate_days: {evidence.get('run_meta', {}).get('excluded_candidate_days')}",
        f"summary.timing_confidence_counts: {evidence.get('summary', {}).get('timing_confidence_counts')}",
        "",
        "## V7-c interpretation rules",
        "",
        "- robust_order / moderate_order mean relative timing order, not causality.",
        "- robust_censored_order means direction is supported but exact peak day is boundary-censored.",
        "- sync_or_overlap means two fields are too close in timing to force order.",
        "- ambiguous_order means insufficient pairwise support.",
        "",
        "Logs and V7-b run metadata are evidence layers, not automatic truth; they are cross-checked with required output tables.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def run_field_transition_pairwise_order_v7_c(settings: StagePartitionV7Settings | None = None):
    settings = settings or StagePartitionV7Settings()
    settings.pairwise_order.output_tag = "field_transition_pairwise_order_v7_c"
    started_at = now_utc()
    roots = _prepare_dirs(settings)
    output_root = roots["output_root"]
    log_root = roots["log_root"]
    settings.write_json(output_root / "config_used.json")
    settings.write_json(log_root / "config_used.json")

    v7b_root = settings.v7_b_output_root()
    evidence = _check_v7b_inputs(v7b_root)
    write_json(evidence, output_root / "v7_b_input_evidence.json")
    write_json(evidence, log_root / "v7_b_input_evidence.json")
    _write_audit_log(log_root / "field_transition_pairwise_order_v7_c_audit_log.md", evidence, settings)

    boot_path = _require_file(v7b_root / REQUIRED_V7B_FILES["bootstrap_samples"], "bootstrap sample table")
    loyo_path = _require_file(v7b_root / REQUIRED_V7B_FILES["loyo_samples"], "LOYO sample table")
    main_path = _require_file(v7b_root / REQUIRED_V7B_FILES["main_table"], "main V7-b table")
    windows_path = _require_file(v7b_root / REQUIRED_V7B_FILES["accepted_windows"], "accepted windows table")

    bootstrap_samples = pd.read_csv(boot_path)
    loyo_samples = pd.read_csv(loyo_path)
    main_df = pd.read_csv(main_path)
    accepted_windows = pd.read_csv(windows_path)
    write_dataframe(accepted_windows, output_root / "accepted_windows_inherited_from_v7_b_v7_c.csv")

    pair_df, all_dir_df = _build_pairwise_tables(bootstrap_samples, loyo_samples, main_df, settings)
    write_dataframe(all_dir_df, output_root / "pairwise_order_all_directions_v7_c.csv")
    write_dataframe(pair_df, output_root / "pairwise_order_bootstrap_summary_v7_c.csv")

    partial = _build_window_partial_order_summary(pair_df)
    edges = _build_edge_table(pair_df)
    ranks = _build_rank_summary(all_dir_df)
    write_dataframe(partial, output_root / "window_partial_order_summary_v7_c.csv")
    write_dataframe(edges, output_root / "order_graph_edges_v7_c.csv")
    write_dataframe(ranks, output_root / "field_order_rank_summary_v7_c.csv")

    if settings.pairwise_order.write_plots:
        _try_write_plots(pair_df, all_dir_df, output_root)

    label_counts = pair_df["order_confidence_label"].value_counts(dropna=False).to_dict() if not pair_df.empty else {}
    summary = {
        "layer_name": "stage_partition",
        "version_name": "V7",
        "run_label": settings.pairwise_order.output_tag,
        "status": "success",
        "scope": "pairwise field transition timing order from V7-b bootstrap/LOYO samples",
        "source_v7_b_output_tag": settings.pairwise_order.source_v7_b_output_tag,
        "accepted_peak_days": [int(x) for x in settings.accepted_windows.accepted_peak_days],
        "excluded_candidate_days": [int(x) for x in settings.accepted_windows.excluded_candidate_days],
        "n_pairwise_tests": int(pair_df.shape[0]) if pair_df is not None else 0,
        "order_confidence_counts": label_counts,
        "n_order_edges": int(edges.shape[0]) if edges is not None else 0,
        "downstream_lead_lag_included": False,
        "pathway_included": False,
        "spatial_earliest_region_included": False,
        "causal_interpretation_included": False,
    }
    write_json(summary, output_root / "summary.json")
    write_json(summary, log_root / "summary.json")

    run_meta = {
        "status": "success",
        "started_at_utc": started_at,
        "ended_at_utc": now_utc(),
        "layer_name": "stage_partition",
        "version_name": "V7",
        "run_label": settings.pairwise_order.output_tag,
        "source_v7_b_output_root": str(v7b_root),
        "accepted_peak_days": [int(x) for x in settings.accepted_windows.accepted_peak_days],
        "excluded_candidate_days": [int(x) for x in settings.accepted_windows.excluded_candidate_days],
        "method": "pairwise_order_audit_from_v7_b_bootstrap_and_loyo_peak_day_samples",
        "notes": [
            "V7-c does not rerun ruptures.Window; it reuses V7-b bootstrap and LOYO peak-day samples.",
            "V7-b single-field timing labels are retained as metadata, but V7-c evaluates pairwise relative order directly.",
            "robust_censored_order preserves useful relative-order direction while marking exact peak-day boundary censoring.",
            "Timing order does not imply causality.",
            "Downstream lead-lag/pathway outputs are not read.",
            "Spatial earliest-region timing is not included.",
        ],
    }
    write_json(run_meta, output_root / "run_meta.json")
    write_json(run_meta, log_root / "run_meta.json")
    return {"output_root": output_root, "log_root": log_root, "summary": summary}
