from __future__ import annotations

from pathlib import Path
from math import erf, lgamma, log, exp
import numpy as np
import pandas as pd

from stage_partition_v6.io import load_smoothed_fields
from stage_partition_v6.state_builder import build_profiles, build_state_matrix
from stage_partition_v6.detector_ruptures_window import run_ruptures_window
from stage_partition_v6.timeline import day_index_to_month_day

from .config import StagePartitionV7Settings
from .field_state import FIELDS, build_field_state, build_field_state_matrix_for_year_indices
from .field_timing import _audit_accepted_points, _find_second_peak, _label_peak
from .report import now_utc, write_dataframe, write_json


def _prepare_dirs(settings: StagePartitionV7Settings) -> dict[str, Path]:
    out = settings.output_root()
    log_dir = settings.log_root()
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return {"output_root": out, "log_root": log_dir}


def _as_int_day(x) -> int:
    return int(round(float(x)))


def _build_analysis_windows(windows_df: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    rows = []
    buffer_days = int(settings.analysis_window.buffer_days)
    accepted = [int(x) for x in settings.accepted_windows.accepted_peak_days]
    for _, r in windows_df.sort_values("main_peak_day").iterrows():
        anchor = int(r["main_peak_day"])
        if anchor not in accepted:
            continue
        start = int(r["start_day"])
        end = int(r["end_day"])
        rows.append(
            {
                "window_id": str(r.get("window_id", f"W{anchor:03d}")),
                "anchor_day": anchor,
                "anchor_month_day": day_index_to_month_day(anchor),
                "accepted_window_start": start,
                "accepted_window_end": end,
                "analysis_window_start": max(0, start - buffer_days),
                "analysis_window_end": end + buffer_days,
                "buffer_days": buffer_days,
                "source_window_id": str(r.get("window_id", f"W{anchor:03d}")),
                "max_member_bootstrap_match_fraction": float(r.get("max_member_bootstrap_match_fraction", np.nan)),
                "note": "accepted V6/V6_1 window with V7-b expanded analysis window",
            }
        )
    return pd.DataFrame(rows)


def _series_to_wide_frame(field_profiles: dict[str, pd.Series]) -> pd.DataFrame:
    all_days = sorted(set().union(*[set(s.index.astype(int).tolist()) for s in field_profiles.values() if s is not None]))
    out = pd.DataFrame({"day": all_days})
    for field in FIELDS:
        s = field_profiles.get(field, pd.Series(dtype=float)).sort_index()
        out[f"{field}_score"] = out["day"].map(lambda d: float(s.get(int(d), np.nan)))
    return out


def _peak_record_from_profile(
    profile: pd.Series,
    window: pd.Series | dict,
    field: str,
    settings: StagePartitionV7Settings,
    *,
    prefix: str = "observed",
) -> dict:
    start = int(window["analysis_window_start"])
    end = int(window["analysis_window_end"])
    astart = int(window["accepted_window_start"])
    aend = int(window["accepted_window_end"])
    anchor = int(window["anchor_day"])
    window_id = str(window["window_id"])

    profile = profile.sort_index()
    sub = profile[(profile.index.astype(int) >= start) & (profile.index.astype(int) <= end)].dropna()
    if sub.empty:
        return {
            "window_id": window_id,
            "field": field,
            "anchor_day": anchor,
            "accepted_window_start": astart,
            "accepted_window_end": aend,
            "analysis_window_start": start,
            "analysis_window_end": end,
            f"{prefix}_peak_day": np.nan,
            f"{prefix}_relative_to_anchor": np.nan,
            f"{prefix}_peak_score": np.nan,
            f"{prefix}_window_median_score": np.nan,
            f"{prefix}_peak_sharpness": np.nan,
            f"{prefix}_second_peak_day": np.nan,
            f"{prefix}_second_peak_score": np.nan,
            f"{prefix}_second_peak_ratio": np.nan,
            f"{prefix}_inside_accepted_window": False,
            f"{prefix}_inside_buffer_only": False,
            f"{prefix}_near_analysis_edge": False,
            f"{prefix}_peak_position_label": "no_score_in_analysis_window",
            f"{prefix}_morphology_label": "weak_or_unclear",
        }

    peak_day = int(sub.idxmax())
    peak_score = float(sub.loc[peak_day])
    med = float(np.nanmedian(sub.to_numpy(dtype=float)))
    peak_sharpness = peak_score / med if np.isfinite(med) and med > 0 else np.nan
    second_day, second_score = _find_second_peak(sub, peak_day, int(settings.detector.local_peak_min_distance_days))
    second_ratio = float(second_score) / peak_score if second_score is not None and np.isfinite(second_score) and peak_score > 0 else np.nan
    high_cut = peak_score * float(settings.peak_labels.high_plateau_ratio)
    n_high_days = int((sub >= high_cut).sum()) if np.isfinite(high_cut) else 0
    pos, morph = _label_peak(
        peak_day=peak_day,
        window_start=start,
        window_end=end,
        peak_score=peak_score,
        window_median=med,
        second_ratio=second_ratio,
        n_high_days=n_high_days,
        cfg=settings.peak_labels,
    )
    edge_margin = int(settings.analysis_window.edge_margin_days)
    inside_accepted = bool(astart <= peak_day <= aend)
    inside_analysis = bool(start <= peak_day <= end)
    near_edge = bool(peak_day <= start + edge_margin or peak_day >= end - edge_margin)
    return {
        "window_id": window_id,
        "field": field,
        "anchor_day": anchor,
        "accepted_window_start": astart,
        "accepted_window_end": aend,
        "analysis_window_start": start,
        "analysis_window_end": end,
        f"{prefix}_peak_day": peak_day,
        f"{prefix}_relative_to_anchor": int(peak_day - anchor),
        f"{prefix}_peak_score": peak_score,
        f"{prefix}_window_median_score": med,
        f"{prefix}_peak_sharpness": peak_sharpness,
        f"{prefix}_second_peak_day": second_day if second_day is not None else np.nan,
        f"{prefix}_second_peak_score": second_score if second_score is not None else np.nan,
        f"{prefix}_second_peak_ratio": second_ratio,
        f"{prefix}_inside_accepted_window": inside_accepted,
        f"{prefix}_inside_buffer_only": bool(inside_analysis and not inside_accepted),
        f"{prefix}_near_analysis_edge": near_edge,
        f"{prefix}_peak_position_label": pos,
        f"{prefix}_morphology_label": morph,
    }


def _detect_field_profile(profiles: dict, field: str, year_indices, shared_valid_day_index: np.ndarray, settings: StagePartitionV7Settings) -> pd.Series:
    matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(
        profiles,
        field,
        year_indices,
        standardize=settings.state.standardize,
        trim_invalid_days=settings.state.trim_invalid_days,
        shared_valid_day_index=shared_valid_day_index if settings.state.use_joint_valid_day_index else None,
    )
    det = run_ruptures_window(matrix, settings.detector, day_index=valid_day_index)
    return det["profile"].sort_index()


def _peak_long_row_from_profile(profile: pd.Series, window: pd.Series, field: str, settings: StagePartitionV7Settings, *, sample_id_name: str, sample_id_value: int) -> dict:
    rec = _peak_record_from_profile(profile, window, field, settings, prefix="sample")
    out = {
        sample_id_name: int(sample_id_value),
        "window_id": rec["window_id"],
        "field": field,
        "anchor_day": rec["anchor_day"],
        "peak_day": rec["sample_peak_day"],
        "relative_to_anchor": rec["sample_relative_to_anchor"],
        "peak_score": rec["sample_peak_score"],
        "inside_accepted_window": rec["sample_inside_accepted_window"],
        "inside_buffer_only": rec["sample_inside_buffer_only"],
        "near_analysis_edge": rec["sample_near_analysis_edge"],
    }
    return out


def _binomial_sf(k: int, n: int, p: float) -> float:
    """P[X >= k] for Binomial(n,p), using scipy if available, otherwise log-sum-exp."""
    k = int(k)
    n = int(n)
    p = float(p)
    if n <= 0:
        return np.nan
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
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


def _bh_fdr(pvals: list[float]) -> list[float]:
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


def _summarize_bootstrap(samples: pd.DataFrame, observed: pd.DataFrame, analysis_windows: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    rows = []
    radius = int(settings.timing_confidence.support_radius_days)
    for (window_id, field), sub in samples.groupby(["window_id", "field"], sort=False):
        days = pd.to_numeric(sub["peak_day"], errors="coerce").dropna().astype(int)
        obs_sub = observed[(observed["window_id"] == window_id) & (observed["field"] == field)]
        obs_day = int(obs_sub["observed_peak_day"].iloc[0]) if not obs_sub.empty and pd.notna(obs_sub["observed_peak_day"].iloc[0]) else None
        w = analysis_windows[analysis_windows["window_id"] == window_id].iloc[0]
        L = int(w["analysis_window_end"] - w["analysis_window_start"] + 1)
        if days.empty:
            rows.append({"window_id": window_id, "field": field, "n_bootstrap_valid": 0})
            continue
        counts = days.value_counts().sort_values(ascending=False)
        modal_day = int(counts.index[0])
        n = int(days.size)
        in_modal = (np.abs(days.to_numpy(dtype=int) - modal_day) <= radius)
        in_obs = (np.abs(days.to_numpy(dtype=int) - obs_day) <= radius) if obs_day is not None else np.zeros(n, dtype=bool)
        support_modal = float(in_modal.mean())
        support_obs = float(in_obs.mean()) if obs_day is not None else np.nan
        m_days = min(L, 2 * radius + 1)
        p0 = float(m_days) / float(L) if L > 0 else np.nan
        k = int(in_modal.sum())
        loc_p = _binomial_sf(k, n, p0) if np.isfinite(p0) else np.nan
        rows.append(
            {
                "window_id": window_id,
                "field": field,
                "n_bootstrap_valid": n,
                "observed_peak_day": obs_day if obs_day is not None else np.nan,
                "bootstrap_modal_peak_day": modal_day,
                "bootstrap_modal_relative_to_anchor": int(modal_day - int(w["anchor_day"])),
                "bootstrap_modal_fraction": float((days == modal_day).mean()),
                "bootstrap_median_peak_day": float(np.nanmedian(days)),
                "bootstrap_q025_peak_day": float(np.nanpercentile(days, 2.5)),
                "bootstrap_q975_peak_day": float(np.nanpercentile(days, 97.5)),
                "bootstrap_q95_width": float(np.nanpercentile(days, 97.5) - np.nanpercentile(days, 2.5)),
                "bootstrap_iqr": float(np.nanpercentile(days, 75) - np.nanpercentile(days, 25)),
                "support_observed_within_0d": float((days == obs_day).mean()) if obs_day is not None else np.nan,
                "support_observed_within_1d": float((np.abs(days - obs_day) <= 1).mean()) if obs_day is not None else np.nan,
                "support_observed_within_2d": support_obs,
                "support_modal_within_1d": float((np.abs(days - modal_day) <= 1).mean()),
                "support_modal_within_2d": support_modal,
                "support_inside_accepted_window": float(sub["inside_accepted_window"].astype(bool).mean()),
                "support_inside_buffer_only": float(sub["inside_buffer_only"].astype(bool).mean()),
                "support_near_analysis_edge": float(sub["near_analysis_edge"].astype(bool).mean()),
                "localization_p_value": loc_p,
                "localization_null_window_length": L,
                "localization_null_p0_modal_r2": p0,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and "localization_p_value" in out.columns:
        out["localization_q_value"] = _bh_fdr(out["localization_p_value"].tolist())
        out["localization_supported"] = out["localization_q_value"].astype(float) < float(settings.timing_confidence.fdr_alpha)
    return out


def _summarize_loyo(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if samples is None or samples.empty:
        return pd.DataFrame()
    for (window_id, field), sub in samples.groupby(["window_id", "field"], sort=False):
        days = pd.to_numeric(sub["peak_day"], errors="coerce").dropna().astype(int)
        if days.empty:
            rows.append({"window_id": window_id, "field": field, "n_loyo_valid": 0})
            continue
        median = int(round(float(np.nanmedian(days))))
        rows.append(
            {
                "window_id": window_id,
                "field": field,
                "n_loyo_valid": int(days.size),
                "loyo_median_peak_day": median,
                "loyo_min_peak_day": int(np.nanmin(days)),
                "loyo_max_peak_day": int(np.nanmax(days)),
                "loyo_iqr_peak_day": float(np.nanpercentile(days, 75) - np.nanpercentile(days, 25)),
                "loyo_support_median_within_2d": float((np.abs(days - median) <= 2).mean()),
                "loyo_support_near_analysis_edge": float(sub["near_analysis_edge"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def _assign_confidence(row: pd.Series, settings: StagePartitionV7Settings) -> tuple[str, bool, str]:
    cfg = settings.timing_confidence
    cautions = []
    if bool(row.get("observed_near_analysis_edge", False)) or bool(row.get("bootstrap_modal_near_analysis_edge", False)):
        cautions.append("observed_or_modal_peak_near_analysis_edge")
    if float(row.get("support_near_analysis_edge", 0.0) or 0.0) >= float(cfg.boundary_support_threshold):
        cautions.append("bootstrap_often_near_analysis_edge")
    if cautions:
        return "boundary_truncated", False, ";".join(cautions)

    q = float(row.get("localization_q_value", np.nan))
    supported = np.isfinite(q) and q < float(cfg.fdr_alpha)
    modal_support = float(row.get("support_modal_within_2d", np.nan))
    width = float(row.get("bootstrap_q95_width", np.nan))
    loyo_iqr = float(row.get("loyo_iqr_peak_day", np.nan))
    obs = row.get("observed_peak_day", np.nan)
    modal = row.get("bootstrap_modal_peak_day", np.nan)
    obs_modal_gap = abs(float(obs) - float(modal)) if pd.notna(obs) and pd.notna(modal) else np.inf

    if supported and obs_modal_gap <= 2 and modal_support >= cfg.stable_min_support_modal_r2 and width <= cfg.stable_max_q95_width_days and (not np.isfinite(loyo_iqr) or loyo_iqr <= cfg.stable_max_loyo_iqr_days):
        return "timing_stable", True, ""
    if supported and obs_modal_gap <= 3 and modal_support >= cfg.moderate_min_support_modal_r2 and width <= cfg.moderate_max_q95_width_days:
        return "timing_moderate", True, "use_with_caution"
    if not supported and modal_support < cfg.moderate_min_support_modal_r2:
        return "no_clear_transition_peak", False, "bootstrap_peak_days_not_localized"
    return "timing_uncertain", False, "bootstrap_or_loyo_timing_uncertain"


def _merge_main_table(observed: pd.DataFrame, boot_sum: pd.DataFrame, loyo_sum: pd.DataFrame, settings: StagePartitionV7Settings) -> pd.DataFrame:
    df = observed.merge(boot_sum, on=["window_id", "field"], how="left", suffixes=("", "_boot"))
    if loyo_sum is not None and not loyo_sum.empty:
        df = df.merge(loyo_sum, on=["window_id", "field"], how="left")
    # whether modal day itself is close to analysis edge
    edge_margin = int(settings.analysis_window.edge_margin_days)
    df["bootstrap_modal_near_analysis_edge"] = (
        (df["bootstrap_modal_peak_day"] <= df["analysis_window_start"] + edge_margin)
        | (df["bootstrap_modal_peak_day"] >= df["analysis_window_end"] - edge_margin)
    )
    labels = df.apply(lambda r: _assign_confidence(r, settings), axis=1)
    df["timing_confidence_label"] = [x[0] for x in labels]
    df["usable_for_timing_order"] = [bool(x[1]) for x in labels]
    df["caution"] = [x[2] for x in labels]
    return df


def _build_order_tables(main_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_obs = []
    rows_use = []
    for window_id, sub in main_df.groupby("window_id", sort=False):
        sub = sub.copy().sort_values(["observed_peak_day", "field"])
        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            rows_obs.append(
                {
                    "window_id": window_id,
                    "order_rank": rank,
                    "field": r["field"],
                    "observed_peak_day": r.get("observed_peak_day", np.nan),
                    "bootstrap_modal_peak_day": r.get("bootstrap_modal_peak_day", np.nan),
                    "timing_confidence_label": r.get("timing_confidence_label", ""),
                    "usable_for_timing_order": bool(r.get("usable_for_timing_order", False)),
                    "caution": "observed_order_only_not_causal",
                }
            )
        use = sub[sub["usable_for_timing_order"].astype(bool)].copy().sort_values(["bootstrap_modal_peak_day", "field"])
        if use.shape[0] < 2:
            rows_use.append(
                {
                    "window_id": window_id,
                    "order_rank": np.nan,
                    "field": None,
                    "bootstrap_modal_peak_day": np.nan,
                    "timing_confidence_label": "insufficient_usable_fields_for_full_order",
                    "caution": "do_not_build_order_for_this_window",
                }
            )
        else:
            for rank, (_, r) in enumerate(use.iterrows(), start=1):
                rows_use.append(
                    {
                        "window_id": window_id,
                        "order_rank": rank,
                        "field": r["field"],
                        "bootstrap_modal_peak_day": r.get("bootstrap_modal_peak_day", np.nan),
                        "observed_peak_day": r.get("observed_peak_day", np.nan),
                        "timing_confidence_label": r.get("timing_confidence_label", ""),
                        "caution": "usable_timing_order_only_not_causal",
                    }
                )
    return pd.DataFrame(rows_obs), pd.DataFrame(rows_use)


def _write_audit_log(path: Path, evidence: dict, settings: StagePartitionV7Settings) -> None:
    text = [
        "# V7-b field transition timing bootstrap audit log",
        "",
        "## Scope",
        "",
        "V7-b answers only: around the four accepted transition windows, when does each field show its strongest detector-score timing, and how stable is that timing under year bootstrap / LOYO?",
        "",
        "It does not infer causality, downstream pathways, or spatial earliest regions.",
        "",
        "## Accepted windows",
        "",
        f"Accepted peak days: {list(settings.accepted_windows.accepted_peak_days)}",
        f"Excluded candidate days: {list(settings.accepted_windows.excluded_candidate_days)}",
        f"Bootstrap match threshold for inherited windows: {settings.accepted_windows.bootstrap_match_threshold}",
        f"Analysis-window buffer days: {settings.analysis_window.buffer_days}",
        "",
        "## Statistical support added in V7-b",
        "",
        f"Year bootstrap repeats: {settings.bootstrap.effective_n_bootstrap()}",
        "Leave-one-year-out timing checks: enabled",
        "Uniform localization test with BH-FDR across field × window timing tests: enabled",
        "",
        "## Log evidence excerpts",
        "",
        "### V6 UPDATE_LOG excerpt",
        "",
        "```text",
        evidence.get("v6_log_relevant_excerpt", ""),
        "```",
        "",
        "### V6_1 UPDATE_LOG excerpt",
        "",
        "```text",
        evidence.get("v6_1_log_relevant_excerpt", ""),
        "```",
        "",
        "## Interpretation limits",
        "",
        "- timing_stable/moderate means field-specific peak timing is localized under this detector/profile definition.",
        "- It does not mean causal precedence.",
        "- boundary_truncated means the expanded window may still cut off the field's peak.",
        "- no_clear_transition_peak means the field should not be used for timing order in that window.",
        "",
        "Logs are treated as evidence to audit, not as automatic truth.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _try_write_plots(main_df: pd.DataFrame, observed_score_long: pd.DataFrame, boot_samples: pd.DataFrame, output_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if observed_score_long is not None and not observed_score_long.empty:
        for window_id, sub in observed_score_long.groupby("window_id", sort=False):
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            anchor = int(sub["anchor_day"].iloc[0])
            astart = int(sub["accepted_window_start"].iloc[0])
            aend = int(sub["accepted_window_end"].iloc[0])
            s0 = int(sub["analysis_window_start"].iloc[0])
            s1 = int(sub["analysis_window_end"].iloc[0])
            ax.axvspan(s0, s1, alpha=0.06, label="analysis window")
            ax.axvspan(astart, aend, alpha=0.12, label="accepted window")
            ax.axvline(anchor, linestyle="--", linewidth=1.0, label="anchor")
            for field, fs in sub.groupby("field", sort=False):
                fs = fs.sort_values("day")
                ax.plot(fs["day"], fs["score"], marker="o", linewidth=1.3, label=field)
            ax.set_title(f"{window_id}: field score profiles, expanded V7-b window")
            ax.set_xlabel("day index since Apr 1")
            ax.set_ylabel("ruptures.Window score")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{window_id}_field_score_profiles_expanded_v7_b.png", dpi=180)
            plt.close(fig)

    if boot_samples is not None and not boot_samples.empty:
        for window_id, sub in boot_samples.groupby("window_id", sort=False):
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            fields = list(FIELDS)
            for i, field in enumerate(fields):
                days = pd.to_numeric(sub[sub["field"] == field]["peak_day"], errors="coerce").dropna().astype(int)
                if days.empty:
                    continue
                vals, counts = np.unique(days.to_numpy(), return_counts=True)
                # offset tiny amount by field to avoid complete overlap
                ax.plot(vals, counts / counts.sum(), marker="o", linewidth=1.1, label=field)
            ax.set_title(f"{window_id}: bootstrap peak-day distributions")
            ax.set_xlabel("peak day")
            ax.set_ylabel("bootstrap frequency")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{window_id}_bootstrap_peak_day_distribution_v7_b.png", dpi=180)
            plt.close(fig)

    if main_df is not None and not main_df.empty:
        labels = main_df.pivot(index="window_id", columns="field", values="timing_confidence_label")
        # textual heatmap as a CSV-friendly figure alternative
        labels.to_csv(fig_dir / "timing_confidence_summary_matrix_v7_b.csv")


def run_field_transition_timing_v7_b(settings: StagePartitionV7Settings | None = None):
    settings = settings or StagePartitionV7Settings()
    settings.output.output_tag = "field_transition_timing_v7_b"
    started_at = now_utc()
    roots = _prepare_dirs(settings)
    output_root = roots["output_root"]
    log_root = roots["log_root"]
    settings.write_json(output_root / "config_used.json")
    settings.write_json(log_root / "config_used.json")

    audit = _audit_accepted_points(settings)
    source_windows_df = audit["windows_df"]
    analysis_windows = _build_analysis_windows(source_windows_df, settings)
    write_json(audit["evidence"], output_root / "accepted_window_evidence.json")
    write_json(audit["evidence"], log_root / "accepted_window_evidence.json")
    _write_audit_log(log_root / "field_transition_timing_v7_b_audit_log.md", audit["evidence"], settings)
    write_dataframe(analysis_windows, output_root / "accepted_windows_used_v7_b.csv")

    smoothed_path = settings.foundation.smoothed_fields_path()
    if not smoothed_path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields.npz at {smoothed_path}")
    smoothed = load_smoothed_fields(smoothed_path)
    profiles = build_profiles(smoothed, settings.profile)
    years = np.asarray(smoothed.get("years", np.arange(profiles["P"].raw_cube.shape[0])), dtype=int)
    n_years = int(profiles["P"].raw_cube.shape[0])

    joint_state = build_state_matrix(profiles, settings.state)
    shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)

    # Observed field profiles and observed peaks.
    field_profiles = {}
    observed_rows = []
    observed_score_long_rows = []
    field_state_meta = {}
    for field in FIELDS:
        fs_full = build_field_state(
            profiles,
            field,
            standardize=settings.state.standardize,
            trim_invalid_days=settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index if settings.state.use_joint_valid_day_index else None,
        )
        matrix = fs_full.state_matrix[fs_full.valid_day_index, :]
        det = run_ruptures_window(matrix, settings.detector, day_index=fs_full.valid_day_index)
        profile = det["profile"].sort_index()
        field_profiles[field] = profile
        field_state_meta[field] = fs_full.meta
        for _, w in analysis_windows.iterrows():
            observed_rows.append(_peak_record_from_profile(profile, w, field, settings, prefix="observed"))
            sub = profile[(profile.index.astype(int) >= int(w["analysis_window_start"])) & (profile.index.astype(int) <= int(w["analysis_window_end"]))].dropna()
            for day, score in sub.items():
                observed_score_long_rows.append(
                    {
                        "window_id": str(w["window_id"]),
                        "field": field,
                        "day": int(day),
                        "score": float(score),
                        "anchor_day": int(w["anchor_day"]),
                        "relative_to_anchor": int(day) - int(w["anchor_day"]),
                        "accepted_window_start": int(w["accepted_window_start"]),
                        "accepted_window_end": int(w["accepted_window_end"]),
                        "analysis_window_start": int(w["analysis_window_start"]),
                        "analysis_window_end": int(w["analysis_window_end"]),
                    }
                )
    observed_df = pd.DataFrame(observed_rows)
    observed_score_long = pd.DataFrame(observed_score_long_rows)
    write_dataframe(_series_to_wide_frame(field_profiles), output_root / "observed_field_detector_score_profiles_v7_b.csv")
    write_dataframe(observed_df, output_root / "observed_field_transition_peak_days_v7_b.csv")
    write_dataframe(observed_score_long, output_root / "observed_field_window_score_long_v7_b.csv")
    write_json(field_state_meta, output_root / "observed_field_state_meta_v7_b.json")

    # Bootstrap resampling.
    rng = np.random.default_rng(int(settings.bootstrap.random_seed))
    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    resample_rows = []
    bootstrap_rows = []
    print(f"[V7-b] bootstrap start: n_bootstrap={n_boot}, n_years={n_years}")
    for b in range(n_boot):
        if b == 0 or ((b + 1) % int(settings.bootstrap.progress_every) == 0) or (b + 1 == n_boot):
            print(f"[V7-b][bootstrap] {b + 1}/{n_boot}")
        year_idx = rng.integers(0, n_years, size=n_years)
        resample_rows.append({"bootstrap_id": int(b), "sampled_year_indices": ";".join(str(int(x)) for x in year_idx.tolist())})
        for field in FIELDS:
            profile = _detect_field_profile(profiles, field, year_idx, shared_valid_day_index, settings)
            for _, w in analysis_windows.iterrows():
                bootstrap_rows.append(_peak_long_row_from_profile(profile, w, field, settings, sample_id_name="bootstrap_id", sample_id_value=b))
    bootstrap_samples = pd.DataFrame(bootstrap_rows)
    write_dataframe(pd.DataFrame(resample_rows), output_root / "bootstrap_resample_year_indices_v7_b.csv")
    if settings.bootstrap.write_sample_long_tables:
        write_dataframe(bootstrap_samples, output_root / "field_transition_peak_days_bootstrap_samples_v7_b.csv")
    boot_summary = _summarize_bootstrap(bootstrap_samples, observed_df, analysis_windows, settings)
    write_dataframe(boot_summary, output_root / "field_transition_peak_days_bootstrap_summary_v7_b.csv")

    # LOYO.
    print(f"[V7-b] LOYO start: n_years={n_years}")
    loyo_rows = []
    for left_out in range(n_years):
        if left_out == 0 or ((left_out + 1) % 10 == 0) or (left_out + 1 == n_years):
            print(f"[V7-b][loyo] {left_out + 1}/{n_years}")
        keep = np.asarray([i for i in range(n_years) if i != left_out], dtype=int)
        for field in FIELDS:
            profile = _detect_field_profile(profiles, field, keep, shared_valid_day_index, settings)
            for _, w in analysis_windows.iterrows():
                rec = _peak_long_row_from_profile(profile, w, field, settings, sample_id_name="left_out_year_index", sample_id_value=left_out)
                rec["left_out_year"] = int(years[left_out]) if left_out < len(years) else int(left_out)
                loyo_rows.append(rec)
    loyo_samples = pd.DataFrame(loyo_rows)
    write_dataframe(loyo_samples, output_root / "field_transition_peak_days_loyo_samples_v7_b.csv")
    loyo_summary = _summarize_loyo(loyo_samples)
    write_dataframe(loyo_summary, output_root / "field_transition_peak_days_loyo_summary_v7_b.csv")

    main_df = _merge_main_table(observed_df, boot_summary, loyo_summary, settings)
    write_dataframe(main_df, output_root / "field_transition_peak_days_bootstrap_v7_b.csv")
    usable_summary = main_df[[
        "window_id", "field", "observed_peak_day", "bootstrap_modal_peak_day", "bootstrap_q95_width",
        "support_modal_within_2d", "localization_q_value", "timing_confidence_label", "usable_for_timing_order", "caution"
    ]].copy()
    write_dataframe(usable_summary, output_root / "field_transition_timing_usability_summary_v7_b.csv")
    order_obs, order_use = _build_order_tables(main_df)
    write_dataframe(order_obs, output_root / "field_transition_order_observed_v7_b.csv")
    write_dataframe(order_use, output_root / "field_transition_order_usable_only_v7_b.csv")

    if settings.output.write_plots:
        _try_write_plots(main_df, observed_score_long, bootstrap_samples, output_root)

    summary = {
        "layer_name": "stage_partition",
        "version_name": "V7",
        "run_label": settings.output.output_tag,
        "status": "success",
        "scope": "field-specific transition timing with bootstrap/LOYO support around accepted windows only",
        "accepted_peak_days": [int(x) for x in settings.accepted_windows.accepted_peak_days],
        "excluded_candidate_days": [int(x) for x in settings.accepted_windows.excluded_candidate_days],
        "n_bootstrap": n_boot,
        "n_years": n_years,
        "n_field_window_tests": int(main_df.shape[0]),
        "timing_confidence_counts": main_df["timing_confidence_label"].value_counts(dropna=False).to_dict(),
        "n_usable_for_timing_order": int(main_df["usable_for_timing_order"].astype(bool).sum()),
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
        "run_label": settings.output.output_tag,
        "smoothed_fields_path": str(smoothed_path),
        "source_v6_output_tag": settings.source.source_v6_output_tag,
        "source_v6_1_output_tag": settings.source.source_v6_1_output_tag,
        "accepted_peak_days": [int(x) for x in settings.accepted_windows.accepted_peak_days],
        "excluded_candidate_days": [int(x) for x in settings.accepted_windows.excluded_candidate_days],
        "analysis_window_buffer_days": int(settings.analysis_window.buffer_days),
        "n_bootstrap": n_boot,
        "random_seed": int(settings.bootstrap.random_seed),
        "notes": [
            "V7-b adds year-bootstrap, LOYO, expanded-window, and localization support to V7-a field timing.",
            "Only the four V6/V6_1 accepted windows 45, 81, 113, and 160 are used.",
            "Candidate peaks 18, 96, 132, and 135 are excluded from the current main diagnostic.",
            "V6 and V6_1 logs are read as evidence and cross-checked with output tables.",
            "Downstream lead-lag and pathway outputs are not read.",
            "No causal interpretation is made by this run.",
            "Spatial earliest-region timing is not included in this run.",
            "timing_stable/moderate are timing-localization labels, not physical causality labels.",
        ],
    }
    write_json(run_meta, output_root / "run_meta.json")
    write_json(run_meta, log_root / "run_meta.json")
    return {"output_root": output_root, "log_root": log_root, "summary": summary}
