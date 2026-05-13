"""
V9.2_a result registry A.

Read-only candidate-result registry for direct-year 2D spatiotemporal MVEOF outputs.
This script does NOT rerun MVEOF, does NOT rerun the V9/V7 peak detector, and does
NOT perform physical interpretation. It only consolidates existing V9.2_a outputs
into window-mode level candidate timing summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import os
from typing import Iterable

import numpy as np
import pandas as pd


OBJECT_ORDER = ["H", "Jw", "Je", "P", "V"]  # only tie-breaking for same/near peak groups
DEFAULT_TAU_SYNC = 3.0


@dataclass
class RegistrySettings:
    version: str = "v9_2_a_result_registry_a"
    source_output_tag: str = "direct_year_2d_meof_peak_audit_v9_2_a"
    output_tag: str = "v9_2_a_result_registry_a"
    shift_days_threshold: float = 3.0
    strong_pair_change_threshold: int = 3
    strong_peak_shift_threshold: float = 7.0
    use_leave_one_year_as_evidence: bool = False
    physical_interpretation_included: bool = False
    does_not_rerun_meof_or_peak: bool = True


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input not found: {path}")
        return pd.DataFrame()
    if path.stat().st_size <= 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _join_years(df: pd.DataFrame, window_id: str, mode: int, phase: str) -> str:
    if df.empty:
        return ""
    sub = df[(df["window_id"] == window_id) & (df["mode"] == mode) & (df["pc_phase"] == phase)]
    if sub.empty:
        return ""
    # Preserve PC-rank order if available.
    if "pc_rank_low_to_high" in sub.columns:
        sub = sub.sort_values("pc_rank_low_to_high")
    else:
        sub = sub.sort_values("year")
    return ";".join(str(int(y)) for y in sub["year"].tolist())


def _phase_peak_df(peaks: pd.DataFrame, window_id: str, mode: int, phase: str) -> pd.DataFrame:
    if peaks.empty:
        return pd.DataFrame()
    sub = peaks[(peaks["window_id"] == window_id) & (peaks["mode"] == mode) & (peaks["pc_phase"] == phase)].copy()
    if sub.empty:
        return sub
    sub["selected_peak_day"] = pd.to_numeric(sub["selected_peak_day"], errors="coerce")
    return sub


def _get_tau(pairwise: pd.DataFrame, window_id: str, mode: int, phase: str) -> float:
    if pairwise.empty or "tau_sync_primary_used" not in pairwise.columns:
        return DEFAULT_TAU_SYNC
    sub = pairwise[(pairwise["window_id"] == window_id) & (pairwise["mode"] == mode) & (pairwise["pc_phase"] == phase)]
    vals = pd.to_numeric(sub.get("tau_sync_primary_used", pd.Series(dtype=float)), errors="coerce").dropna()
    if vals.empty:
        return DEFAULT_TAU_SYNC
    return float(vals.iloc[0])


def _sequence_string(peaks: pd.DataFrame, tau_sync: float = DEFAULT_TAU_SYNC) -> str:
    if peaks.empty or "selected_peak_day" not in peaks.columns:
        return "MISSING"
    sub = peaks[["object", "selected_peak_day"]].copy()
    sub["selected_peak_day"] = pd.to_numeric(sub["selected_peak_day"], errors="coerce")
    sub = sub.dropna(subset=["selected_peak_day"])
    if sub.empty:
        return "MISSING"
    object_rank = {o: i for i, o in enumerate(OBJECT_ORDER)}
    sub["_rank"] = sub["object"].map(object_rank).fillna(999)
    sub = sub.sort_values(["selected_peak_day", "_rank", "object"])

    groups: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = []
    current_center = None
    for _, row in sub.iterrows():
        obj = str(row["object"])
        day = float(row["selected_peak_day"])
        if not current:
            current = [(obj, day)]
            current_center = day
            continue
        # Group if within tau of current group mean; avoids over-separating near peaks.
        if abs(day - float(current_center)) <= tau_sync:
            current.append((obj, day))
            current_center = np.mean([d for _, d in current])
        else:
            groups.append(current)
            current = [(obj, day)]
            current_center = day
    if current:
        groups.append(current)

    parts = []
    for grp in groups:
        grp = sorted(grp, key=lambda od: (od[1], object_rank.get(od[0], 999), od[0]))
        if len(grp) == 1:
            obj, day = grp[0]
            parts.append(f"{obj}({day:g})")
        else:
            parts.append("/".join(f"{obj}({day:g})" for obj, day in grp))
    return " -> ".join(parts)


def _object_peak_days_string(peaks: pd.DataFrame) -> str:
    if peaks.empty:
        return ""
    out = []
    for _, r in peaks.sort_values("object").iterrows():
        day = _safe_float(r.get("selected_peak_day"))
        obj = r.get("object")
        out.append(f"{obj}:{day:g}" if not np.isnan(day) else f"{obj}:NA")
    return ";".join(out)


def _make_sequence_summary(peaks: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if peaks.empty:
        return pd.DataFrame(rows)
    keys = peaks[["window_id", "mode", "pc_phase"]].drop_duplicates().sort_values(["window_id", "mode", "pc_phase"])
    phase_order = {"high": 0, "mid": 1, "low": 2}
    keys["_phase_order"] = keys["pc_phase"].map(phase_order).fillna(99)
    keys = keys.sort_values(["window_id", "mode", "_phase_order"])
    for _, k in keys.iterrows():
        wid, mode, phase = k["window_id"], int(k["mode"]), k["pc_phase"]
        sub = _phase_peak_df(peaks, wid, mode, phase)
        tau = _get_tau(pairwise, wid, mode, phase)
        rows.append({
            "window_id": wid,
            "mode": mode,
            "pc_phase": phase,
            "tau_sync_primary_used": tau,
            "n_objects_with_peak": int(sub["selected_peak_day"].notna().sum()) if not sub.empty else 0,
            "sequence_string": _sequence_string(sub, tau),
            "object_peak_days": _object_peak_days_string(sub),
            "n_years_in_group": _safe_int(sub["n_years_in_group"].iloc[0]) if not sub.empty and "n_years_in_group" in sub.columns else 0,
            "years_in_group": str(sub["years_in_group"].iloc[0]) if not sub.empty and "years_in_group" in sub.columns else "",
        })
    return pd.DataFrame(rows)


def _make_object_shift(peaks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if peaks.empty:
        return pd.DataFrame(rows)
    high = peaks[peaks["pc_phase"] == "high"]
    low = peaks[peaks["pc_phase"] == "low"]
    keys = pd.concat([
        high[["window_id", "mode", "object"]],
        low[["window_id", "mode", "object"]],
    ]).drop_duplicates()
    for _, k in keys.sort_values(["window_id", "mode", "object"]).iterrows():
        wid, mode, obj = k["window_id"], int(k["mode"]), k["object"]
        rh = high[(high["window_id"] == wid) & (high["mode"] == mode) & (high["object"] == obj)]
        rl = low[(low["window_id"] == wid) & (low["mode"] == mode) & (low["object"] == obj)]
        ph = _safe_float(rh["selected_peak_day"].iloc[0]) if not rh.empty else np.nan
        pl = _safe_float(rl["selected_peak_day"].iloc[0]) if not rl.empty else np.nan
        shift = ph - pl if not (np.isnan(ph) or np.isnan(pl)) else np.nan
        rows.append({
            "window_id": wid,
            "mode": mode,
            "object": obj,
            "peak_high": ph,
            "peak_low": pl,
            "shift_high_minus_low": shift,
            "abs_shift": abs(shift) if not np.isnan(shift) else np.nan,
            "high_selected_candidate_id": str(rh["selected_candidate_id"].iloc[0]) if not rh.empty and "selected_candidate_id" in rh.columns else "",
            "low_selected_candidate_id": str(rl["selected_candidate_id"].iloc[0]) if not rl.empty and "selected_candidate_id" in rl.columns else "",
            "high_support_class": str(rh["support_class"].iloc[0]) if not rh.empty and "support_class" in rh.columns else "",
            "low_support_class": str(rl["support_class"].iloc[0]) if not rl.empty and "support_class" in rl.columns else "",
            "high_detector_status": str(rh["detector_status"].iloc[0]) if not rh.empty and "detector_status" in rh.columns else "",
            "low_detector_status": str(rl["detector_status"].iloc[0]) if not rl.empty and "detector_status" in rl.columns else "",
        })
    return pd.DataFrame(rows)


def _make_detector_quality(peaks: pd.DataFrame) -> pd.DataFrame:
    if peaks.empty:
        return pd.DataFrame()
    df = peaks.copy()
    for c in ["excluded_candidates", "early_secondary_candidates", "late_secondary_candidates"]:
        if c not in df.columns:
            df[c] = ""
    df["has_missing_peak"] = pd.to_numeric(df.get("selected_peak_day"), errors="coerce").isna()
    df["has_non_ok_detector_status"] = df.get("detector_status", "").astype(str).str.lower().ne("ok")
    df["has_excluded_candidates"] = df["excluded_candidates"].fillna("").astype(str).str.len().gt(0)
    df["has_early_secondary_candidates"] = df["early_secondary_candidates"].fillna("").astype(str).str.len().gt(0)
    df["has_late_secondary_candidates"] = df["late_secondary_candidates"].fillna("").astype(str).str.len().gt(0)
    df["quality_flag"] = np.where(
        df["has_missing_peak"] | df["has_non_ok_detector_status"],
        "problematic_missing_or_detector_status",
        np.where(
            df["has_excluded_candidates"] | df["has_early_secondary_candidates"] | df["has_late_secondary_candidates"],
            "usable_with_candidate_complexity",
            "usable_clean_selection",
        ),
    )
    keep = [
        "window_id", "mode", "pc_phase", "object", "selected_peak_day", "selected_candidate_id",
        "selected_window_start", "selected_window_end", "selected_role", "support_class",
        "selection_reason", "detector_status", "peak_source", "has_missing_peak",
        "has_non_ok_detector_status", "has_excluded_candidates", "has_early_secondary_candidates",
        "has_late_secondary_candidates", "quality_flag", "excluded_candidates", "early_secondary_candidates",
        "late_secondary_candidates",
    ]
    return df[[c for c in keep if c in df.columns]].sort_values(["window_id", "mode", "pc_phase", "object"])


def _summarize_quality(detector_quality: pd.DataFrame) -> pd.DataFrame:
    if detector_quality.empty:
        return pd.DataFrame()
    rows = []
    for (wid, mode), sub in detector_quality.groupby(["window_id", "mode"], sort=True):
        n = len(sub)
        rows.append({
            "window_id": wid,
            "mode": int(mode),
            "n_peak_records": n,
            "n_missing_peak": int(sub["has_missing_peak"].sum()),
            "n_non_ok_detector_status": int(sub["has_non_ok_detector_status"].sum()),
            "n_records_with_excluded_candidates": int(sub["has_excluded_candidates"].sum()),
            "n_records_with_secondary_candidates": int((sub["has_early_secondary_candidates"] | sub["has_late_secondary_candidates"]).sum()),
            "detector_quality_summary": "problematic" if (sub["has_missing_peak"].any() or sub["has_non_ok_detector_status"].any()) else (
                "usable_with_candidate_complexity" if (sub["has_excluded_candidates"].any() or sub["has_early_secondary_candidates"].any() or sub["has_late_secondary_candidates"].any()) else "usable_clean_selection"
            ),
        })
    return pd.DataFrame(rows)


def _make_order_change_summary(contrast: pd.DataFrame) -> pd.DataFrame:
    if contrast.empty:
        return pd.DataFrame()
    cols = [
        "window_id", "mode", "object_A", "object_B", "peak_A_high", "peak_B_high",
        "delta_high", "order_high", "peak_A_low", "peak_B_low", "delta_low", "order_low",
        "delta_shift_high_minus_low", "order_changed_flag", "order_contrast_type",
    ]
    out = contrast[[c for c in cols if c in contrast.columns]].copy()
    if "order_changed_flag" in out.columns:
        out["order_changed_flag"] = out["order_changed_flag"].astype(bool)
    return out.sort_values(["window_id", "mode", "object_A", "object_B"])


def _candidate_level(n_order: int, n_reversal: int, max_abs_shift: float, n_obj_shift: int, detector_summary: str) -> str:
    if detector_summary == "problematic":
        return "Candidate-Hold_detector_problem"
    if n_order >= 3 or n_reversal >= 2 or max_abs_shift >= 7:
        return "Candidate-A_clear_high_low_timing_difference"
    if n_order >= 1 or n_obj_shift >= 2 or max_abs_shift >= 3:
        return "Candidate-B_moderate_high_low_timing_difference"
    return "Candidate-C_field_mode_with_weak_timing_difference"


def _make_window_mode_registry(
    settings: RegistrySettings,
    mode_summary: pd.DataFrame,
    peak_rel: pd.DataFrame,
    pc_years: pd.DataFrame,
    seq: pd.DataFrame,
    obj_shift: pd.DataFrame,
    order_change: pd.DataFrame,
    quality_summary: pd.DataFrame,
    loo_group_peak: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    # Registry is window-mode at the level with available PC-group peak results.
    # mode_summary may include saved but not peak-audited modes, so do not create
    # candidate result rows for modes without high/low group peak outputs.
    if not seq.empty:
        keys = seq[["window_id", "mode"]].drop_duplicates()
    elif not obj_shift.empty:
        keys = obj_shift[["window_id", "mode"]].drop_duplicates()
    elif not peak_rel.empty:
        keys = peak_rel[["window_id", "mode"]].drop_duplicates()
    else:
        keys = mode_summary[["window_id", "mode"]].drop_duplicates()
    for _, k in keys.sort_values(["window_id", "mode"]).iterrows():
        wid, mode = k["window_id"], int(k["mode"])
        ms = mode_summary[(mode_summary["window_id"] == wid) & (mode_summary["mode"] == mode)]
        pr = peak_rel[(peak_rel["window_id"] == wid) & (peak_rel["mode"] == mode)] if not peak_rel.empty else pd.DataFrame()
        high_seq = seq[(seq["window_id"] == wid) & (seq["mode"] == mode) & (seq["pc_phase"] == "high")]
        low_seq = seq[(seq["window_id"] == wid) & (seq["mode"] == mode) & (seq["pc_phase"] == "low")]
        osub = obj_shift[(obj_shift["window_id"] == wid) & (obj_shift["mode"] == mode)] if not obj_shift.empty else pd.DataFrame()
        csub = order_change[(order_change["window_id"] == wid) & (order_change["mode"] == mode)] if not order_change.empty else pd.DataFrame()
        qsub = quality_summary[(quality_summary["window_id"] == wid) & (quality_summary["mode"] == mode)] if not quality_summary.empty else pd.DataFrame()
        high_years = _join_years(pc_years, wid, mode, "high")
        low_years = _join_years(pc_years, wid, mode, "low")

        n_order = int(csub.get("order_changed_flag", pd.Series(dtype=bool)).astype(bool).sum()) if not csub.empty else 0
        n_reversal = int(csub.get("order_contrast_type", pd.Series(dtype=str)).astype(str).str.contains("reversal", na=False).sum()) if not csub.empty else 0
        max_abs_shift = float(pd.to_numeric(osub.get("abs_shift", pd.Series(dtype=float)), errors="coerce").max()) if not osub.empty else np.nan
        n_obj_shift = int((pd.to_numeric(osub.get("abs_shift", pd.Series(dtype=float)), errors="coerce") >= settings.shift_days_threshold).sum()) if not osub.empty else 0
        detector_summary = str(qsub["detector_quality_summary"].iloc[0]) if not qsub.empty else "not_available"
        candidate = _candidate_level(n_order, n_reversal, 0 if np.isnan(max_abs_shift) else max_abs_shift, n_obj_shift, detector_summary)

        rows.append({
            "window_id": wid,
            "mode": mode,
            "explained_variance_ratio": _safe_float(ms["explained_variance_ratio"].iloc[0]) if not ms.empty and "explained_variance_ratio" in ms.columns else np.nan,
            "cumulative_explained_variance_ratio": _safe_float(ms["cumulative_explained_variance_ratio"].iloc[0]) if not ms.empty and "cumulative_explained_variance_ratio" in ms.columns else np.nan,
            "peak_relevance_class_original": str(pr["peak_relevance_class"].iloc[0]) if not pr.empty and "peak_relevance_class" in pr.columns else "not_available",
            "candidate_level": candidate,
            "evidence_stage": "candidate_pc_group_composite_result_not_final_robust",
            "high_years": high_years,
            "low_years": low_years,
            "high_sequence": str(high_seq["sequence_string"].iloc[0]) if not high_seq.empty else "MISSING",
            "low_sequence": str(low_seq["sequence_string"].iloc[0]) if not low_seq.empty else "MISSING",
            "n_objects_with_peak_shift_ge_3d": n_obj_shift,
            "max_abs_object_peak_shift": max_abs_shift,
            "n_pairs_order_changed": n_order,
            "n_pairs_reversal_like": n_reversal,
            "detector_quality_summary": detector_summary,
            "group_peak_loo_status": "not_run_or_incomplete" if loo_group_peak.empty else "available_but_not_used_for_final_tier",
            "single_year_influence_status": "not_audited_in_registry_a",
            "physical_interpretation_status": "not_started",
            "interpretation_permission": "candidate_timing_pattern_only_not_physical_regime",
            "missing_evidence": "group_peak_leave_one_year_not_run_or_incomplete;physical_interpretation_not_started;single_year_influence_not_audited",
        })
    return pd.DataFrame(rows)


def _write_markdown_summary(out: Path, registry: pd.DataFrame, seq: pd.DataFrame, settings: RegistrySettings) -> None:
    lines = []
    lines.append("# V9.2_a Result Registry A Summary\n")
    lines.append("## 1. Scope\n")
    lines.append("This is a read-only candidate-result registry for V9.2_a direct-year 2D spatiotemporal MVEOF outputs. It does not rerun MVEOF, does not rerun peak detection, and does not perform physical interpretation.\n")
    lines.append("## 2. Evidence status\n")
    lines.append("- Evidence stage: candidate PC-group composite timing result.\n")
    lines.append("- Group-peak leave-one-year stability: marked as not_run_or_incomplete unless a non-empty input table is present.\n")
    lines.append("- Single-year influence audit: not audited in this registry.\n")
    lines.append("- Physical interpretation: not started.\n")
    lines.append("\n## 3. Candidate level definitions\n")
    lines.append("- Candidate-A: clear high/low timing difference by object peak shift or pair order change.\n")
    lines.append("- Candidate-B: moderate high/low timing difference.\n")
    lines.append("- Candidate-C: field mode exists but timing difference is weak.\n")
    lines.append("- Candidate-Hold: detector quality problem.\n")
    if not registry.empty:
        lines.append("\n## 4. Window-mode registry overview\n")
        cols = ["window_id", "mode", "candidate_level", "high_sequence", "low_sequence", "n_pairs_order_changed", "max_abs_object_peak_shift"]
        for _, r in registry[cols].iterrows():
            lines.append(f"- {r['window_id']} mode{int(r['mode'])}: {r['candidate_level']}; high: {r['high_sequence']}; low: {r['low_sequence']}; order_changes={r['n_pairs_order_changed']}; max_object_shift={r['max_abs_object_peak_shift']}.\n")
    lines.append("\n## 5. Prohibited interpretation\n")
    lines.append("- Do not treat PC high/low as physical year types.\n")
    lines.append("- Do not treat candidate levels as final robust tiers.\n")
    lines.append("- Do not treat group-composite peak differences as single-year rules.\n")
    lines.append("- Do not interpret MVEOF modes as physical mechanisms before field/profile audit.\n")
    lines.append("\n## 6. Recommended next use\n")
    lines.append("Use `v9_2_a_window_mode_registry.csv` and `v9_2_a_window_mode_sequence_summary.csv` to extract candidate timing patterns by window-mode. Use detector and missing-evidence columns to avoid overclaiming.\n")
    (out / "V9_2_A_RESULT_REGISTRY_A_SUMMARY.md").write_text("".join(lines), encoding="utf-8")


def run_summarize_v9_2_a_result_registry_a(v92_root: Path | str) -> None:
    v92_root = Path(v92_root)
    settings = RegistrySettings()
    source_root = _env_path(
        "V9_2_A_SOURCE_OUTPUT_DIR",
        v92_root / "outputs" / settings.source_output_tag,
    )
    cross = source_root / "cross_window"
    out_root = v92_root / "outputs" / settings.output_tag
    out = out_root / "cross_window"
    logs = v92_root / "logs" / settings.output_tag
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    print("[1/6] Reading existing V9.2_a outputs")
    mode_summary = _read_csv(cross / "v9_2_meof_mode_summary_all_windows.csv")
    pc_years = _read_csv(cross / "v9_2_pc_phase_years_all_windows.csv")
    peaks = _read_csv(cross / "v9_2_pc_group_object_peak_all_windows.csv")
    pairwise = _read_csv(cross / "v9_2_pc_group_pairwise_peak_order_all_windows.csv")
    contrast = _read_csv(cross / "v9_2_pc_group_high_low_order_contrast_all_windows.csv")
    peak_rel = _read_csv(cross / "v9_2_peak_relevance_summary_all_windows.csv", required=False)
    loo_group_peak = _read_csv(cross / "v9_2_leave_one_year_group_peak_stability_all_windows.csv", required=False)

    print("[2/6] Building sequence and shift summaries")
    seq = _make_sequence_summary(peaks, pairwise)
    obj_shift = _make_object_shift(peaks)
    order_change = _make_order_change_summary(contrast)

    print("[3/6] Building detector quality audit")
    detector_quality = _make_detector_quality(peaks)
    detector_quality_summary = _summarize_quality(detector_quality)

    print("[4/6] Building window-mode candidate registry")
    registry = _make_window_mode_registry(
        settings,
        mode_summary,
        peak_rel,
        pc_years,
        seq,
        obj_shift,
        order_change,
        detector_quality_summary,
        loo_group_peak,
    )

    print("[5/6] Writing registry outputs")
    registry.to_csv(out / "v9_2_a_window_mode_registry.csv", index=False)
    seq.to_csv(out / "v9_2_a_window_mode_sequence_summary.csv", index=False)
    # Alias for convenience; same content but tells the user this is the PC phase year list.
    pc_years.to_csv(out / "v9_2_a_pc_phase_year_list.csv", index=False)
    obj_shift.to_csv(out / "v9_2_a_object_peak_shift_summary.csv", index=False)
    order_change.to_csv(out / "v9_2_a_order_change_summary.csv", index=False)
    detector_quality.to_csv(out / "v9_2_a_detector_quality_audit.csv", index=False)
    detector_quality_summary.to_csv(out / "v9_2_a_detector_quality_summary_by_mode.csv", index=False)

    _write_markdown_summary(out, registry, seq, settings)

    run_meta = {
        "version": settings.version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_output_dir": str(source_root),
        "output_dir": str(out_root),
        **asdict(settings),
        "n_window_mode_rows": int(len(registry)),
        "n_sequence_rows": int(len(seq)),
        "n_object_shift_rows": int(len(obj_shift)),
        "n_order_change_rows": int(len(order_change)),
        "n_detector_quality_rows": int(len(detector_quality)),
        "group_peak_loo_input_rows": int(len(loo_group_peak)),
        "note": "Read-only registry. Candidate timing patterns only; not final robust tiers and not physical interpretation.",
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[6/6] Done")
    (logs / "last_run.txt").write_text(
        f"completed {settings.version} at {run_meta['created_at']}\noutput={out}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run_summarize_v9_2_a_result_registry_a(Path(__file__).resolve().parents[2])
