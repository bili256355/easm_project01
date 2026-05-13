from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from stage_partition_v6.io import load_smoothed_fields
from stage_partition_v6.state_builder import build_profiles, build_state_matrix
from stage_partition_v6.detector_ruptures_window import run_ruptures_window, extract_ranked_local_peaks
from stage_partition_v6.timeline import day_index_to_month_day

from .config import StagePartitionV7Settings
from .field_state import FIELDS, build_field_state
from .report import now_utc, write_dataframe, write_json, build_summary


def _prepare_dirs(settings: StagePartitionV7Settings) -> dict[str, Path]:
    out = settings.output_root()
    log = settings.log_root()
    out.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    return {"output_root": out, "log_root": log}


def _read_text_required(path: Path, *, required: bool) -> str:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required log/evidence file: {path}")
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_log_evidence(log_text: str, keywords: list[str], context: int = 2) -> str:
    lines = log_text.splitlines()
    selected: set[int] = set()
    for i, line in enumerate(lines):
        if any(k in line for k in keywords):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                selected.add(j)
    if not selected:
        return ""
    return "\n".join(lines[i] for i in sorted(selected))


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input table: {path}")
    return pd.read_csv(path)


def _audit_accepted_points(settings: StagePartitionV7Settings) -> dict:
    required = bool(settings.accepted_windows.require_logs)
    v6_log = _read_text_required(settings.source.v6_update_log(), required=required)
    v6_1_log = _read_text_required(settings.source.v6_1_update_log(), required=required)

    accepted = [int(x) for x in settings.accepted_windows.accepted_peak_days]
    excluded = [int(x) for x in settings.accepted_windows.excluded_candidate_days]

    bootstrap = _read_csv_required(settings.source.v6_bootstrap_summary_path())
    if "point_day" not in bootstrap.columns or "bootstrap_match_fraction" not in bootstrap.columns:
        raise KeyError(
            "candidate_points_bootstrap_summary.csv must contain point_day and bootstrap_match_fraction"
        )

    bootstrap["point_day"] = bootstrap["point_day"].astype(int)
    accepted_boot = bootstrap[bootstrap["point_day"].isin(accepted)].copy()
    excluded_boot = bootstrap[bootstrap["point_day"].isin(excluded)].copy()
    missing = sorted(set(accepted) - set(accepted_boot["point_day"].astype(int).tolist()))
    if missing:
        raise ValueError(f"Accepted points missing from V6 bootstrap summary: {missing}")

    threshold = float(settings.accepted_windows.bootstrap_match_threshold)
    below = accepted_boot[accepted_boot["bootstrap_match_fraction"].astype(float) < threshold]
    if settings.accepted_windows.require_bootstrap_support and not below.empty:
        raise ValueError(
            "Some accepted points are below bootstrap_match_threshold="
            f"{threshold}: {below[['point_day','bootstrap_match_fraction']].to_dict(orient='records')}"
        )

    windows = _read_csv_required(settings.source.v6_1_windows_path())
    if "main_peak_day" not in windows.columns:
        raise KeyError("derived_windows_registry.csv must contain main_peak_day")
    windows["main_peak_day"] = windows["main_peak_day"].astype(int)
    selected_windows = windows[windows["main_peak_day"].isin(accepted)].copy()
    missing_windows = sorted(set(accepted) - set(selected_windows["main_peak_day"].astype(int).tolist()))
    if settings.accepted_windows.require_windows_from_v6_1 and missing_windows:
        raise ValueError(f"Accepted points missing from V6_1 derived windows: {missing_windows}")
    selected_windows = selected_windows.sort_values("main_peak_day").reset_index(drop=True)

    log_keywords = ["45", "81", "113", "160", "主显著", "后续路径层", "优先", "132", "135"]
    evidence = {
        "v6_update_log_path": str(settings.source.v6_update_log()),
        "v6_1_update_log_path": str(settings.source.v6_1_update_log()),
        "v6_log_relevant_excerpt": _extract_log_evidence(v6_log, log_keywords),
        "v6_1_log_relevant_excerpt": _extract_log_evidence(v6_1_log, log_keywords),
        "accepted_points": accepted,
        "excluded_candidate_days": excluded,
        "bootstrap_match_threshold": threshold,
        "accepted_bootstrap_records": accepted_boot.sort_values("point_day").to_dict(orient="records"),
        "excluded_bootstrap_records": excluded_boot.sort_values("point_day").to_dict(orient="records"),
        "selected_windows": selected_windows.to_dict(orient="records"),
        "source_note": (
            "Logs are audited as evidence and cross-checked with V6 bootstrap summary "
            "and V6_1 derived window registry. They are not treated as automatic truth."
        ),
    }
    return {"windows_df": selected_windows, "bootstrap_df": bootstrap, "evidence": evidence}


def _series_to_frame(profile: pd.Series, field: str) -> pd.DataFrame:
    if profile is None or profile.empty:
        return pd.DataFrame(columns=["day", "field", "score"])
    out = profile.rename_axis("day").reset_index(name="score")
    out["day"] = out["day"].astype(int)
    out["field"] = field
    return out[["day", "field", "score"]]


def _find_second_peak(window_scores: pd.Series, peak_day: int, min_distance: int) -> tuple[int | None, float | None]:
    if window_scores.empty:
        return None, None
    mask = (window_scores.index.astype(int) - int(peak_day)).astype(int)
    keep = np.abs(mask) >= max(1, int(min_distance))
    candidates = window_scores[keep]
    if candidates.empty:
        candidates = window_scores[window_scores.index.astype(int) != int(peak_day)]
    if candidates.empty:
        return None, None
    second_day = int(candidates.idxmax())
    second_score = float(candidates.loc[second_day])
    return second_day, second_score


def _label_peak(
    *,
    peak_day: int,
    window_start: int,
    window_end: int,
    peak_score: float,
    window_median: float,
    second_ratio: float | None,
    n_high_days: int,
    cfg,
) -> tuple[str, str]:
    edge_margin = int(cfg.edge_margin_days)
    near_left = int(peak_day) <= int(window_start) + edge_margin
    near_right = int(peak_day) >= int(window_end) - edge_margin
    if near_left:
        position = "near_left_edge"
    elif near_right:
        position = "near_right_edge"
    else:
        position = "inside"

    sharpness = peak_score / window_median if np.isfinite(window_median) and window_median > 0 else np.nan

    if near_left or near_right:
        return position, "edge_peak"
    if second_ratio is not None and np.isfinite(second_ratio) and second_ratio >= float(cfg.multi_peak_second_ratio):
        return position, "multi_peak"
    if np.isfinite(sharpness) and sharpness < float(cfg.weak_peak_sharpness_threshold):
        return position, "weak_or_unclear"
    if int(n_high_days) >= int(cfg.broad_peak_min_high_days):
        return position, "broad_peak"
    return position, "clear_peak"


def _extract_peak_rows(
    windows_df: pd.DataFrame,
    field_profiles: dict[str, pd.Series],
    local_peaks_by_field: dict[str, pd.DataFrame],
    settings: StagePartitionV7Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak_rows = []
    long_rows = []
    min_distance = int(settings.detector.local_peak_min_distance_days)

    for _, w in windows_df.iterrows():
        window_id = str(w.get("window_id", f"W{int(w['main_peak_day']):03d}"))
        start = int(w["start_day"])
        end = int(w["end_day"])
        anchor = int(w["main_peak_day"])

        for field in FIELDS:
            profile = field_profiles[field].sort_index()
            sub = profile[(profile.index.astype(int) >= start) & (profile.index.astype(int) <= end)].dropna()

            if sub.empty:
                peak_rows.append(
                    {
                        "window_id": window_id,
                        "window_start_day": start,
                        "window_end_day": end,
                        "window_anchor_day": anchor,
                        "window_anchor_month_day": day_index_to_month_day(anchor),
                        "field": field,
                        "field_peak_day": np.nan,
                        "field_peak_month_day": None,
                        "relative_to_anchor": np.nan,
                        "field_peak_score": np.nan,
                        "window_median_score": np.nan,
                        "peak_sharpness": np.nan,
                        "second_peak_day": np.nan,
                        "second_peak_score": np.nan,
                        "second_peak_ratio": np.nan,
                        "n_window_score_days": 0,
                        "n_high_plateau_days": 0,
                        "peak_position_label": "no_score_in_window",
                        "peak_confidence_label": "weak_or_unclear",
                        "method": "field_only_ruptures_window_score_profile_peak_within_accepted_window",
                    }
                )
                continue

            peak_day = int(sub.idxmax())
            peak_score = float(sub.loc[peak_day])
            window_median = float(np.nanmedian(sub.to_numpy(dtype=float)))
            peak_sharpness = peak_score / window_median if np.isfinite(window_median) and window_median > 0 else np.nan
            second_day, second_score = _find_second_peak(sub, peak_day, min_distance)
            second_ratio = (
                float(second_score) / peak_score
                if second_score is not None and np.isfinite(second_score) and peak_score > 0
                else np.nan
            )
            high_threshold = peak_score * float(settings.peak_labels.high_plateau_ratio)
            n_high = int((sub >= high_threshold).sum()) if np.isfinite(high_threshold) else 0
            pos_label, conf_label = _label_peak(
                peak_day=peak_day,
                window_start=start,
                window_end=end,
                peak_score=peak_score,
                window_median=window_median,
                second_ratio=second_ratio,
                n_high_days=n_high,
                cfg=settings.peak_labels,
            )

            local_peaks = local_peaks_by_field.get(field, pd.DataFrame())
            is_local_peak_day = False
            local_peak_prominence = np.nan
            local_peak_rank = np.nan
            if local_peaks is not None and not local_peaks.empty and "peak_day" in local_peaks.columns:
                hit = local_peaks[local_peaks["peak_day"].astype(int) == peak_day]
                if not hit.empty:
                    is_local_peak_day = True
                    local_peak_prominence = float(hit.iloc[0].get("peak_prominence", np.nan))
                    local_peak_rank = int(hit.iloc[0].get("peak_rank", -1))

            peak_rows.append(
                {
                    "window_id": window_id,
                    "window_start_day": start,
                    "window_end_day": end,
                    "window_anchor_day": anchor,
                    "window_anchor_month_day": day_index_to_month_day(anchor),
                    "field": field,
                    "field_peak_day": peak_day,
                    "field_peak_month_day": day_index_to_month_day(peak_day),
                    "relative_to_anchor": int(peak_day - anchor),
                    "field_peak_score": peak_score,
                    "window_median_score": window_median,
                    "peak_sharpness": peak_sharpness,
                    "second_peak_day": second_day,
                    "second_peak_score": second_score,
                    "second_peak_ratio": second_ratio,
                    "n_window_score_days": int(len(sub)),
                    "n_high_plateau_days": n_high,
                    "is_detector_local_peak_day": bool(is_local_peak_day),
                    "local_peak_prominence": local_peak_prominence,
                    "local_peak_rank": local_peak_rank,
                    "peak_position_label": pos_label,
                    "peak_confidence_label": conf_label,
                    "method": "field_only_ruptures_window_score_profile_peak_within_accepted_window",
                }
            )

            for day, score in sub.items():
                long_rows.append(
                    {
                        "window_id": window_id,
                        "window_start_day": start,
                        "window_end_day": end,
                        "window_anchor_day": anchor,
                        "day": int(day),
                        "month_day": day_index_to_month_day(int(day)),
                        "relative_to_anchor": int(day - anchor),
                        "field": field,
                        "score": float(score),
                        "is_peak_day": bool(int(day) == peak_day),
                    }
                )

    peak_df = pd.DataFrame(peak_rows)
    long_df = pd.DataFrame(long_rows)
    if not peak_df.empty:
        peak_df = peak_df.sort_values(["window_anchor_day", "field"]).reset_index(drop=True)
    if not long_df.empty:
        long_df = long_df.sort_values(["window_anchor_day", "field", "day"]).reset_index(drop=True)
    return peak_df, long_df


def _build_order_table(peak_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if peak_df is None or peak_df.empty:
        return pd.DataFrame(columns=["window_id", "window_anchor_day", "order_rank", "field", "field_peak_day"])
    for window_id, sub in peak_df.groupby("window_id", sort=False):
        sub = sub.copy()
        sub = sub[pd.notna(sub["field_peak_day"])].sort_values(["field_peak_day", "field"]).reset_index(drop=True)
        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            rows.append(
                {
                    "window_id": str(window_id),
                    "window_anchor_day": int(r["window_anchor_day"]),
                    "order_rank": int(rank),
                    "field": str(r["field"]),
                    "field_peak_day": int(r["field_peak_day"]),
                    "relative_to_anchor": int(r["relative_to_anchor"]),
                    "peak_confidence_label": str(r["peak_confidence_label"]),
                    "caution": "do_not_interpret_as_causal_order",
                }
            )
    return pd.DataFrame(rows)


def _write_audit_log_md(path: Path, evidence: dict, settings: StagePartitionV7Settings) -> None:
    text = [
        "# V7-a field transition timing audit log",
        "",
        "## Scope",
        "",
        "This run answers only: around accepted transition windows, when does each field show its strongest detector-score transition signal?",
        "",
        "It does not infer causality, downstream pathways, or spatial earliest regions.",
        "",
        "## Accepted windows",
        "",
        f"Accepted peak days: {list(settings.accepted_windows.accepted_peak_days)}",
        f"Excluded candidate days: {list(settings.accepted_windows.excluded_candidate_days)}",
        f"Bootstrap match threshold: {settings.accepted_windows.bootstrap_match_threshold}",
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
        "## Cross-check",
        "",
        "The accepted points are cross-checked against V6 `candidate_points_bootstrap_summary.csv` and V6_1 `derived_windows_registry.csv`.",
        "",
        "Logs are treated as evidence to audit, not as automatic truth.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _try_write_plots(score_long_df: pd.DataFrame, peak_df: pd.DataFrame, output_root: Path) -> None:
    if score_long_df is None or score_long_df.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_dir = output_root / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for window_id, sub in score_long_df.groupby("window_id", sort=False):
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        anchor = int(sub["window_anchor_day"].iloc[0])
        start = int(sub["window_start_day"].iloc[0])
        end = int(sub["window_end_day"].iloc[0])
        for field, fs in sub.groupby("field", sort=False):
            fs = fs.sort_values("day")
            ax.plot(fs["day"], fs["score"], marker="o", linewidth=1.5, label=field)
        ax.axvspan(start, end, alpha=0.08)
        ax.axvline(anchor, linestyle="--", linewidth=1.0)
        psub = peak_df[peak_df["window_id"] == window_id] if peak_df is not None else pd.DataFrame()
        if not psub.empty:
            for _, r in psub.iterrows():
                if pd.notna(r["field_peak_day"]):
                    ax.axvline(int(r["field_peak_day"]), linestyle=":", linewidth=0.8)
        ax.set_title(f"{window_id}: field-specific detector score profiles")
        ax.set_xlabel("day index since Apr 1")
        ax.set_ylabel("ruptures.Window score")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{window_id}_field_score_profiles.png", dpi=180)
        plt.close(fig)


def run_field_transition_timing_v7_a(settings: StagePartitionV7Settings | None = None):
    settings = settings or StagePartitionV7Settings()
    started_at = now_utc()
    roots = _prepare_dirs(settings)
    output_root = roots["output_root"]
    log_root = roots["log_root"]

    settings.write_json(output_root / "config_used.json")
    settings.write_json(log_root / "config_used.json")

    audit = _audit_accepted_points(settings)
    windows_df = audit["windows_df"]
    write_json(audit["evidence"], output_root / "accepted_window_evidence.json")
    write_json(audit["evidence"], log_root / "accepted_window_evidence.json")
    _write_audit_log_md(log_root / "field_transition_timing_audit_log.md", audit["evidence"], settings)

    smoothed_path = settings.foundation.smoothed_fields_path()
    if not smoothed_path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields.npz at {smoothed_path}")
    smoothed = load_smoothed_fields(smoothed_path)

    profiles = build_profiles(smoothed, settings.profile)

    # Build the original joint state only to inherit the shared valid-day index.
    joint_state = build_state_matrix(profiles, settings.state)
    shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)

    field_profiles: dict[str, pd.Series] = {}
    local_peaks_by_field: dict[str, pd.DataFrame] = {}
    feature_tables = []
    scale_tables = []
    empty_tables = []
    field_state_meta = {}

    for field in FIELDS:
        fs = build_field_state(
            profiles,
            field,
            standardize=settings.state.standardize,
            trim_invalid_days=settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index if settings.state.use_joint_valid_day_index else None,
        )
        matrix = fs.state_matrix[fs.valid_day_index, :]
        det = run_ruptures_window(matrix, settings.detector, day_index=fs.valid_day_index)
        profile = det["profile"].sort_index()
        local_peaks = extract_ranked_local_peaks(
            profile,
            min_distance_days=settings.detector.local_peak_min_distance_days,
            prominence_min=0.0,
        )
        field_profiles[field] = profile
        local_peaks_by_field[field] = local_peaks
        feature_tables.append(fs.feature_table)
        scale_tables.append(fs.scale_table)
        empty_tables.append(fs.empty_feature_audit)
        field_state_meta[field] = fs.meta

    score_profile_df = pd.concat(
        [_series_to_frame(field_profiles[field], field) for field in FIELDS],
        ignore_index=True,
    )
    local_peaks_df = pd.concat(
        [df.assign(field=field) for field, df in local_peaks_by_field.items()],
        ignore_index=True,
    ) if local_peaks_by_field else pd.DataFrame()

    peak_df, window_score_long_df = _extract_peak_rows(
        windows_df,
        field_profiles,
        local_peaks_by_field,
        settings,
    )
    order_df = _build_order_table(peak_df)

    write_dataframe(windows_df, output_root / "accepted_windows_used.csv")
    write_dataframe(score_profile_df, output_root / "field_detector_score_profiles.csv")
    write_dataframe(local_peaks_df, output_root / "field_detector_local_peaks.csv")
    write_dataframe(peak_df, output_root / "field_transition_peak_days_by_window.csv")
    write_dataframe(window_score_long_df, output_root / "field_window_score_long.csv")
    write_dataframe(order_df, output_root / "field_transition_order_by_window.csv")
    write_dataframe(pd.concat(feature_tables, ignore_index=True), output_root / "field_feature_table.csv")
    write_dataframe(pd.concat(scale_tables, ignore_index=True), output_root / "field_state_feature_scale.csv")
    write_dataframe(pd.concat(empty_tables, ignore_index=True), output_root / "field_state_empty_feature_audit.csv")
    write_json(field_state_meta, output_root / "field_state_meta.json")

    if settings.output.write_plots:
        _try_write_plots(window_score_long_df, peak_df, output_root)

    summary = build_summary(
        peak_df=peak_df,
        windows_df=windows_df,
        accepted_points=[int(x) for x in settings.accepted_windows.accepted_peak_days],
        settings=settings,
    )
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
        "notes": [
            "V7-a answers only field-level transition peak timing around accepted windows.",
            "Only the four V6/V6_1 accepted windows 45, 81, 113, and 160 are used.",
            "Candidate peaks 18, 96, 132, and 135 are excluded from the current main diagnostic.",
            "V6 and V6_1 logs are read as evidence and cross-checked with output tables.",
            "Field-only ruptures.Window score profiles are used; single-field formal breakpoints are not used as new windows.",
            "Downstream lead-lag and pathway outputs are not read.",
            "No causal interpretation is made by this run.",
            "Spatial earliest-region timing is not included in this run.",
        ],
    }
    write_json(run_meta, output_root / "run_meta.json")
    write_json(run_meta, log_root / "run_meta.json")

    return {"output_root": output_root, "log_root": log_root, "summary": summary}
