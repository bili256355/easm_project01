from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELDS = ["P", "V", "H", "Je", "Jw"]
OUTPUT_TAG = "w45_process_curve_foundation_audit_v7_o"
V7M_TAG = "w45_allfield_transition_marker_definition_audit_v7_m"
V7N_TAG = "w45_allfield_process_relation_layer_v7_n"
THRESHOLDS = [0.10, 0.25, 0.50, 0.75]
SENSITIVITY_SHIFT_DAYS = 2


@dataclass
class FoundationAuditPaths:
    v7_root: Path
    v7e_output_dir: Path
    v7m_output_dir: Path
    v7n_output_dir: Path
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


def _write_text(text: str, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_paths(v7_root: Optional[Path]) -> FoundationAuditPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    return FoundationAuditPaths(
        v7_root=v7_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7m_output_dir=v7_root / "outputs" / V7M_TAG,
        v7n_output_dir=v7_root / "outputs" / V7N_TAG,
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _safe_arr(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_median(values) -> float:
    arr = _safe_arr(values)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def _safe_quantile(values, q: float) -> float:
    arr = _safe_arr(values)
    return float(np.nanquantile(arr, q)) if arr.size else np.nan


def _norm(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.sqrt(np.nanmean(np.square(arr))))


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
        return np.nanmean(x[good], axis=0)


def _state_to_full_matrix(matrix: np.ndarray, valid_day_index: np.ndarray, n_days: int) -> np.ndarray:
    full = np.full((int(n_days), matrix.shape[1]), np.nan, dtype=float)
    idx = np.asarray(valid_day_index, dtype=int)
    idx = idx[(idx >= 0) & (idx < n_days)]
    full[idx, :] = np.asarray(matrix, dtype=float)[: idx.size, :]
    return full


def _require_columns(df: pd.DataFrame, cols: Iterable[str], table: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table} missing required columns: {missing}")


def _label_from_ratio(value: float, good: float = 0.70, caution: float = 0.50) -> str:
    if not np.isfinite(value):
        return "unavailable"
    if value >= good:
        return "strong"
    if value >= caution:
        return "caution"
    return "weak"


def _ensure_v7m(paths: FoundationAuditPaths) -> None:
    required = [
        paths.v7m_output_dir / "w45_allfield_progress_observed_curves_long_v7_m.csv",
        paths.v7m_output_dir / "w45_allfield_progress_bootstrap_curves_long_v7_m.csv",
        paths.v7m_output_dir / "w45_allfield_progress_observed_recomputed_rows_v7_m.csv",
        paths.v7m_output_dir / "w45_allfield_progress_bootstrap_recomputed_rows_v7_m.csv",
        paths.v7m_output_dir / "w45_allfield_marker_bootstrap_samples_v7_m.csv",
    ]
    if all(p.exists() for p in required):
        return
    try:
        from stage_partition_v7.w45_allfield_transition_marker_definition_audit import (  # type: ignore
            run_w45_allfield_transition_marker_definition_audit_v7_m,
        )
        print("[V7-o] Required V7-m outputs not found; running V7-m first.")
        run_w45_allfield_transition_marker_definition_audit_v7_m(paths.v7_root)
    except Exception as exc:  # noqa: BLE001
        missing = [str(p) for p in required if not p.exists()]
        raise FileNotFoundError(
            "V7-o requires V7-m outputs. Run scripts/run_w45_allfield_transition_marker_definition_audit_v7_m.py first. "
            f"Missing: {missing}. Original error: {exc}"
        ) from exc


def _ensure_v7n(paths: FoundationAuditPaths) -> None:
    required = [
        paths.v7n_output_dir / "w45_field_transition_curves_v7_n.csv",
        paths.v7n_output_dir / "w45_pairwise_curve_relation_daily_v7_n.csv",
        paths.v7n_output_dir / "w45_pairwise_phase_relation_v7_n.csv",
        paths.v7n_output_dir / "w45_pairwise_relation_type_v7_n.csv",
    ]
    if all(p.exists() for p in required):
        return
    try:
        from stage_partition_v7.w45_allfield_process_relation_layer import (  # type: ignore
            run_w45_allfield_process_relation_layer_v7_n,
        )
        print("[V7-o] Required V7-n outputs not found; running V7-n first.")
        run_w45_allfield_process_relation_layer_v7_n(paths.v7_root)
    except Exception as exc:  # noqa: BLE001
        missing = [str(p) for p in required if not p.exists()]
        raise FileNotFoundError(
            "V7-o requires V7-n process relation outputs. Run scripts/run_w45_allfield_process_relation_layer_v7_n.py first. "
            f"Missing: {missing}. Original error: {exc}"
        ) from exc


def _load_existing_outputs(paths: FoundationAuditPaths) -> dict[str, pd.DataFrame]:
    _ensure_v7m(paths)
    _ensure_v7n(paths)
    tables = {
        "v7m_obs_curves": _read_csv(paths.v7m_output_dir / "w45_allfield_progress_observed_curves_long_v7_m.csv"),
        "v7m_boot_curves": _read_csv(paths.v7m_output_dir / "w45_allfield_progress_bootstrap_curves_long_v7_m.csv"),
        "v7m_obs_rows": _read_csv(paths.v7m_output_dir / "w45_allfield_progress_observed_recomputed_rows_v7_m.csv"),
        "v7m_boot_rows": _read_csv(paths.v7m_output_dir / "w45_allfield_progress_bootstrap_recomputed_rows_v7_m.csv"),
        "v7m_boot_markers": _read_csv(paths.v7m_output_dir / "w45_allfield_marker_bootstrap_samples_v7_m.csv"),
        "v7n_field_curves": _read_csv(paths.v7n_output_dir / "w45_field_transition_curves_v7_n.csv"),
        "v7n_pair_daily": _read_csv(paths.v7n_output_dir / "w45_pairwise_curve_relation_daily_v7_n.csv"),
        "v7n_pair_phase": _read_csv(paths.v7n_output_dir / "w45_pairwise_phase_relation_v7_n.csv"),
        "v7n_pair_marker": _read_csv(paths.v7n_output_dir / "w45_pairwise_marker_family_relation_v7_n.csv"),
        "v7n_pair_type": _read_csv(paths.v7n_output_dir / "w45_pairwise_relation_type_v7_n.csv"),
    }
    return tables


def _get_window_row(paths: FoundationAuditPaths) -> pd.Series:
    windows = _read_csv(paths.v7e_output_dir / "accepted_windows_used_v7_e.csv")
    _require_columns(windows, ["window_id", "anchor_day", "analysis_window_start", "analysis_window_end", "pre_period_start", "pre_period_end", "post_period_start", "post_period_end"], "accepted_windows_used_v7_e")
    sub = windows[windows["window_id"].astype(str) == WINDOW_ID]
    if sub.empty:
        sub = windows[pd.to_numeric(windows["anchor_day"], errors="coerce") == ANCHOR_DAY]
    if sub.empty:
        raise ValueError("Cannot find W002 / anchor=45 in accepted_windows_used_v7_e.csv")
    return sub.iloc[0]


def _try_rebuild_observed_field_matrices(paths: FoundationAuditPaths) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    """Rebuild observed W45 field state matrices in the exact current V7-e representation.

    Returns full day x feature matrices for each field when the installed project has V6/V7 sources and
    foundation smoothed fields. If this cannot be done, returns empty dicts and an audit note.
    """
    try:
        from stage_partition_v6.io import load_smoothed_fields  # type: ignore
        from stage_partition_v6.state_builder import build_profiles, build_state_matrix  # type: ignore
        from stage_partition_v7.config import StagePartitionV7Settings  # type: ignore
        from stage_partition_v7.field_state import build_field_state_matrix_for_year_indices  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {}, {}, {"state_rebuild_status": "unavailable_import_failed", "state_rebuild_error": repr(exc)}
    try:
        settings = StagePartitionV7Settings()
        settings.output.output_tag = "field_transition_progress_timing_v7_e"
        smoothed = load_smoothed_fields(settings.foundation.smoothed_fields_path())
        profiles = build_profiles(smoothed, settings.profile)
        joint_state = build_state_matrix(profiles, settings.state)
        shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)
        n_days = int(np.asarray(profiles[FIELDS[0]].raw_cube).shape[1])
        matrices: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        for field in FIELDS:
            matrix, valid_day_index, _feature_table = build_field_state_matrix_for_year_indices(
                profiles,
                field,
                None,
                standardize=settings.state.standardize,
                trim_invalid_days=settings.state.trim_invalid_days,
                shared_valid_day_index=shared_valid_day_index,
            )
            matrices[field] = _state_to_full_matrix(matrix, valid_day_index, n_days)
            meta[field] = {
                "n_features": int(matrix.shape[1]),
                "n_valid_days": int(len(valid_day_index)),
                "valid_day_index_min": int(np.min(valid_day_index)) if len(valid_day_index) else None,
                "valid_day_index_max": int(np.max(valid_day_index)) if len(valid_day_index) else None,
            }
        return matrices, meta, {"state_rebuild_status": "success", "n_days": n_days, "settings_source": "StagePartitionV7Settings()"}
    except Exception as exc:  # noqa: BLE001
        return {}, {}, {"state_rebuild_status": "unavailable_runtime_failed", "state_rebuild_error": repr(exc)}


def _prototype_metrics(full: np.ndarray, window: pd.Series) -> dict:
    ps, pe = int(window["pre_period_start"]), int(window["pre_period_end"])
    qs, qe = int(window["post_period_start"]), int(window["post_period_end"])
    pre = full[ps : pe + 1]
    post = full[qs : qe + 1]
    pre_proto = _nanmean_rows(pre)
    post_proto = _nanmean_rows(post)
    if pre_proto is None or post_proto is None:
        return {"available": False}
    vec = post_proto - pre_proto
    pre_good = pre[_row_finite_mask(pre)]
    post_good = post[_row_finite_mask(post)]
    pre_var = float(np.nanmean([_norm(x - pre_proto) for x in pre_good])) if len(pre_good) else np.nan
    post_var = float(np.nanmean([_norm(x - post_proto) for x in post_good])) if len(post_good) else np.nan
    sep = _norm(vec)
    within = float(np.nanmean([pre_var, post_var])) if np.isfinite(pre_var) or np.isfinite(post_var) else np.nan
    ratio = float(sep / (within + 1e-12)) if np.isfinite(within) else np.nan
    return {
        "available": True,
        "pre_proto": pre_proto,
        "post_proto": post_proto,
        "vec": vec,
        "pre_post_distance_observed": sep,
        "within_pre_variability_observed": pre_var,
        "within_post_variability_observed": post_var,
        "separation_ratio_observed": ratio,
    }


def build_prepost_foundation(tables: dict[str, pd.DataFrame], matrices: dict[str, np.ndarray], window: pd.Series) -> pd.DataFrame:
    boot_rows = tables["v7m_boot_rows"]
    obs_rows = tables["v7m_obs_rows"]
    rows = []
    for field in FIELDS:
        b = boot_rows[(boot_rows["window_id"].astype(str) == WINDOW_ID) & (boot_rows["field"].astype(str) == field)]
        o = obs_rows[(obs_rows["window_id"].astype(str) == WINDOW_ID) & (obs_rows["field"].astype(str) == field)]
        row: dict[str, object] = {"window_id": WINDOW_ID, "anchor_day": ANCHOR_DAY, "field": field}
        if not o.empty:
            for col in ["pre_post_distance", "within_pre_variability", "within_post_variability", "separation_ratio", "pre_post_separation_label"]:
                if col in o.columns:
                    row[f"v7m_observed_{col}"] = o[col].iloc[0]
        for col in ["pre_post_distance", "within_pre_variability", "within_post_variability", "separation_ratio"]:
            if col in b.columns:
                vals = pd.to_numeric(b[col], errors="coerce")
                row[f"bootstrap_{col}_median"] = _safe_median(vals)
                row[f"bootstrap_{col}_q05"] = _safe_quantile(vals, 0.05)
                row[f"bootstrap_{col}_q95"] = _safe_quantile(vals, 0.95)
        if "pre_post_separation_label" in b.columns and not b.empty:
            row["bootstrap_dominant_pre_post_separation_label"] = b["pre_post_separation_label"].astype(str).value_counts().idxmax()
            row["bootstrap_clear_separation_fraction"] = float((b["pre_post_separation_label"].astype(str) == "clear_separation").mean())
        if field in matrices:
            pm = _prototype_metrics(matrices[field], window)
            if pm.get("available"):
                row["state_rebuild_prepost_available"] = True
                row["state_observed_pre_post_distance"] = pm["pre_post_distance_observed"]
                row["state_observed_within_pre_variability"] = pm["within_pre_variability_observed"]
                row["state_observed_within_post_variability"] = pm["within_post_variability_observed"]
                row["state_observed_separation_ratio"] = pm["separation_ratio_observed"]
            else:
                row["state_rebuild_prepost_available"] = False
        else:
            row["state_rebuild_prepost_available"] = False
        ratio = row.get("bootstrap_separation_ratio_median", row.get("v7m_observed_separation_ratio", np.nan))
        clear_frac = row.get("bootstrap_clear_separation_fraction", np.nan)
        if np.isfinite(ratio) and ratio >= 5 and (not np.isfinite(clear_frac) or clear_frac >= 0.80):
            label = "strong_prepost_foundation"
        elif np.isfinite(ratio) and ratio >= 2:
            label = "usable_prepost_foundation"
        elif np.isfinite(ratio) and ratio >= 1:
            label = "weak_prepost_foundation"
        else:
            label = "prepost_not_established"
        row["prepost_foundation_label"] = label
        row["prepost_foundation_interpretation"] = (
            "pre/post prototypes appear statistically usable in current V7-e representation" if "strong" in label or "usable" in label else
            "pre/post prototypes are weak; downstream progress curve should be downgraded"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_projection_direction_and_validity(matrices: dict[str, np.ndarray], window: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    summary_rows = []
    component_rows = []
    component_summary_rows = []
    start, end = int(window["analysis_window_start"]), int(window["analysis_window_end"])
    for field in FIELDS:
        if field not in matrices:
            summary_rows.append({
                "field": field,
                "state_rebuild_available": False,
                "transition_direction_label": "unavailable_state_rebuild",
                "projection_progress_validity_label": "unavailable_state_rebuild",
                "single_curve_adequacy_label": "unavailable_state_rebuild",
            })
            component_summary_rows.append({"field": field, "single_curve_adequacy_label": "unavailable_state_rebuild"})
            continue
        full = matrices[field]
        pm = _prototype_metrics(full, window)
        if not pm.get("available"):
            summary_rows.append({
                "field": field,
                "state_rebuild_available": True,
                "transition_direction_label": "prototype_unavailable",
                "projection_progress_validity_label": "prototype_unavailable",
                "single_curve_adequacy_label": "prototype_unavailable",
            })
            continue
        pre_proto = pm["pre_proto"]
        post_proto = pm["post_proto"]
        vec = pm["vec"]
        denom = float(np.nansum(np.square(vec)))
        days = []
        proj_fracs = []
        resid_fracs = []
        proj_prog = []
        dist_prog = []
        bounded_bad = []
        same_dir_count = []
        prev_pp = np.nan
        prev_dp = np.nan
        for day in range(start, end + 1):
            x = full[day]
            if not np.all(np.isfinite(x)) or denom <= 1e-12:
                continue
            centered = x - pre_proto
            raw_prog = float(np.nansum(centered * vec) / denom)
            proj_component = raw_prog * vec
            resid = centered - proj_component
            total_norm = _norm(centered)
            proj_norm = _norm(proj_component)
            resid_norm = _norm(resid)
            total_power = float(np.nansum(np.square(centered)))
            proj_power = float(np.nansum(np.square(proj_component)))
            resid_power = float(np.nansum(np.square(resid)))
            projection_fraction = float(proj_power / (total_power + 1e-12)) if np.isfinite(total_power) else np.nan
            residual_fraction = float(resid_power / (total_power + 1e-12)) if np.isfinite(total_power) else np.nan
            dpre = _norm(x - pre_proto)
            dpost = _norm(x - post_proto)
            dprog = float(dpre / (dpre + dpost + 1e-12)) if np.isfinite(dpre) and np.isfinite(dpost) else np.nan
            if np.isfinite(prev_pp) and np.isfinite(prev_dp):
                same_dir = int(np.sign(raw_prog - prev_pp) == np.sign(dprog - prev_dp))
                same_dir_count.append(same_dir)
            prev_pp, prev_dp = raw_prog, dprog
            days.append(day)
            proj_fracs.append(projection_fraction)
            resid_fracs.append(residual_fraction)
            proj_prog.append(raw_prog)
            dist_prog.append(dprog)
            bounded_bad.append(int(raw_prog < -0.05 or raw_prog > 1.05))
            daily_rows.append({
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "field": field,
                "day": day,
                "relative_to_anchor": day - ANCHOR_DAY,
                "projection_progress_raw": raw_prog,
                "projection_progress_clipped_0_1": float(np.clip(raw_prog, 0.0, 1.0)),
                "distance_progress": dprog,
                "dist_to_pre": dpre,
                "dist_to_post": dpost,
                "projection_norm": proj_norm,
                "orthogonal_residual_norm": resid_norm,
                "total_norm_from_pre": total_norm,
                "projection_fraction_power": projection_fraction,
                "residual_fraction_power": residual_fraction,
                "progress_projection_minus_distance": raw_prog - dprog if np.isfinite(dprog) else np.nan,
            })
        corr = np.nan
        if len(proj_prog) >= 3:
            pp = np.asarray(proj_prog, dtype=float)
            dp = np.asarray(dist_prog, dtype=float)
            m = np.isfinite(pp) & np.isfinite(dp)
            if int(m.sum()) >= 3:
                corr = float(np.corrcoef(pp[m], dp[m])[0, 1])
        med_proj = _safe_median(proj_fracs)
        med_resid = _safe_median(resid_fracs)
        sign_same = float(np.nanmean(same_dir_count)) if same_dir_count else np.nan
        out_frac = float(np.nanmean(bounded_bad)) if bounded_bad else np.nan
        transition_label = (
            "dominant_prepost_direction" if np.isfinite(med_proj) and med_proj >= 0.60 else
            "usable_prepost_direction_with_residual" if np.isfinite(med_proj) and med_proj >= 0.35 else
            "multi_direction_transition" if np.isfinite(med_proj) else "unavailable"
        )
        validity_label = (
            "valid_progress_proxy" if np.isfinite(corr) and corr >= 0.90 and (not np.isfinite(out_frac) or out_frac < 0.10) else
            "usable_progress_proxy_with_caution" if np.isfinite(corr) and corr >= 0.70 else
            "projection_distance_inconsistent" if np.isfinite(corr) else "unavailable"
        )
        summary_rows.append({
            "field": field,
            "state_rebuild_available": True,
            "median_projection_fraction_power": med_proj,
            "q05_projection_fraction_power": _safe_quantile(proj_fracs, 0.05),
            "q95_projection_fraction_power": _safe_quantile(proj_fracs, 0.95),
            "median_residual_fraction_power": med_resid,
            "q95_residual_fraction_power": _safe_quantile(resid_fracs, 0.95),
            "corr_projection_vs_distance_progress": corr,
            "median_abs_projection_distance_diff": _safe_median(np.abs(np.asarray(proj_prog) - np.asarray(dist_prog))) if len(proj_prog) else np.nan,
            "fraction_days_projection_distance_same_direction": sign_same,
            "fraction_projection_progress_outside_near_0_1": out_frac,
            "transition_direction_label": transition_label,
            "projection_progress_validity_label": validity_label,
        })
        # Component adequacy: per feature one-dimensional progress against its own dF component.
        dcomp = np.asarray(vec, dtype=float)
        abs_dcomp = np.abs(dcomp)
        total_abs = float(np.nansum(abs_dcomp))
        dom_frac = float(np.nanmax(abs_dcomp) / (total_abs + 1e-12)) if total_abs > 0 else np.nan
        comp_t25 = []
        comp_t50 = []
        comp_t75 = []
        for j in range(full.shape[1]):
            if not np.isfinite(dcomp[j]) or abs(dcomp[j]) <= 1e-12:
                continue
            vals = []
            comp_days = []
            for day in range(start, end + 1):
                x = full[day, j]
                if np.isfinite(x):
                    vals.append(float((x - pre_proto[j]) / dcomp[j]))
                    comp_days.append(day)
            vals_arr = np.asarray(vals, dtype=float)
            days_arr = np.asarray(comp_days, dtype=int)
            def first_cross(q):
                for d, v in zip(days_arr, vals_arr):
                    if np.isfinite(v) and v >= q:
                        return float(d)
                return np.nan
            t25 = first_cross(0.25)
            t50 = first_cross(0.50)
            t75 = first_cross(0.75)
            comp_t25.append(t25); comp_t50.append(t50); comp_t75.append(t75)
            component_rows.append({
                "field": field,
                "feature_index": int(j),
                "dF_component": float(dcomp[j]),
                "abs_dF_component": float(abs_dcomp[j]),
                "relative_abs_dF_contribution": float(abs_dcomp[j] / (total_abs + 1e-12)) if total_abs > 0 else np.nan,
                "component_t25_day": t25,
                "component_t50_day": t50,
                "component_t75_day": t75,
                "component_t25_deviation_from_field_median": t25 - _safe_median(comp_t25) if np.isfinite(t25) else np.nan,
            })
        t25_iqr = _safe_quantile(comp_t25, 0.75) - _safe_quantile(comp_t25, 0.25)
        t50_iqr = _safe_quantile(comp_t50, 0.75) - _safe_quantile(comp_t50, 0.25)
        t75_iqr = _safe_quantile(comp_t75, 0.75) - _safe_quantile(comp_t75, 0.25)
        # Labels are descriptive and deliberately conservative.
        if np.isfinite(t25_iqr) and np.isfinite(t50_iqr) and max(t25_iqr, t50_iqr) <= 3 and np.isfinite(dom_frac) and dom_frac < 0.35:
            adequacy = "single_curve_adequate"
        elif np.isfinite(t25_iqr) and max(t25_iqr, t50_iqr if np.isfinite(t50_iqr) else 0) <= 8:
            adequacy = "single_curve_usable_with_heterogeneity"
        elif np.isfinite(t25_iqr):
            adequacy = "multi_component_transition"
        else:
            adequacy = "single_curve_not_adequate"
        component_summary_rows.append({
            "field": field,
            "n_components_with_nonzero_dF": int(np.isfinite(comp_t25).sum()) if len(comp_t25) else 0,
            "component_t25_iqr": t25_iqr,
            "component_t50_iqr": t50_iqr,
            "component_t75_iqr": t75_iqr,
            "dominant_component_fraction_abs_dF": dom_frac,
            "single_curve_adequacy_label": adequacy,
        })
    return pd.DataFrame(daily_rows), pd.DataFrame(summary_rows), pd.DataFrame(component_rows), pd.DataFrame(component_summary_rows)


def build_pair_phase_comparability(field_decision: pd.DataFrame, v7n_pair_type: pd.DataFrame) -> pd.DataFrame:
    f = field_decision.set_index("field") if not field_decision.empty else pd.DataFrame()
    rows = []
    for a, b in combinations(FIELDS, 2):
        ar = f.loc[a].to_dict() if a in f.index else {}
        br = f.loc[b].to_dict() if b in f.index else {}
        a_label = ar.get("field_curve_foundation_label", "unavailable")
        b_label = br.get("field_curve_foundation_label", "unavailable")
        pair_sub = v7n_pair_type[((v7n_pair_type.get("field_a", "") == a) & (v7n_pair_type.get("field_b", "") == b)) | ((v7n_pair_type.get("field_a", "") == b) & (v7n_pair_type.get("field_b", "") == a))] if not v7n_pair_type.empty else pd.DataFrame()
        relation_type = pair_sub["relation_type"].iloc[0] if (not pair_sub.empty and "relation_type" in pair_sub.columns) else "unknown"
        if a_label.startswith("valid") and b_label.startswith("valid"):
            comp = "phase_comparable"
        elif ("not_valid" in a_label) or ("not_valid" in b_label) or ("unavailable" in a_label) or ("unavailable" in b_label) or ("incomplete" in a_label) or ("incomplete" in b_label):
            comp = "not_comparable_due_to_field_curve_quality"
        else:
            comp = "phase_comparable_with_caution"
        if relation_type in {"phase_crossing_relation", "front_loaded_vs_catchup"} and comp.startswith("phase_comparable"):
            comp_use = "layer_specific_only"
        else:
            comp_use = comp
        rows.append({
            "field_a": a,
            "field_b": b,
            "field_a_curve_foundation_label": a_label,
            "field_b_curve_foundation_label": b_label,
            "v7n_relation_type": relation_type,
            "pair_phase_comparability_label": comp_use,
            "phase_comparability_interpretation": "pair progress phase can be compared only with layer/phase-specific language" if comp_use == "layer_specific_only" else comp_use,
        })
    return pd.DataFrame(rows)


def build_pair_artifact_risk(proj_daily: pd.DataFrame, v7n_pair_daily: pd.DataFrame, v7n_pair_phase: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Use observed projection-vs-distance progress if state rebuild is available.
    proj_piv = pd.DataFrame()
    dist_piv = pd.DataFrame()
    if not proj_daily.empty and {"field", "day", "projection_progress_raw", "distance_progress"}.issubset(proj_daily.columns):
        proj_piv = proj_daily.pivot_table(index="day", columns="field", values="projection_progress_raw", aggfunc="first")
        dist_piv = proj_daily.pivot_table(index="day", columns="field", values="distance_progress", aggfunc="first")
    for a, b in combinations(FIELDS, 2):
        sign_consistency = np.nan
        if not proj_piv.empty and a in proj_piv.columns and b in proj_piv.columns and a in dist_piv.columns and b in dist_piv.columns:
            dp = (proj_piv[a] - proj_piv[b]).to_numpy(dtype=float)
            dd = (dist_piv[a] - dist_piv[b]).to_numpy(dtype=float)
            good = np.isfinite(dp) & np.isfinite(dd)
            if int(good.sum()) > 0:
                sign_consistency = float(np.mean(np.sign(dp[good]) == np.sign(dd[good])))
        ph = v7n_pair_phase[((v7n_pair_phase.get("field_a", "") == a) & (v7n_pair_phase.get("field_b", "") == b)) | ((v7n_pair_phase.get("field_a", "") == b) & (v7n_pair_phase.get("field_b", "") == a))] if not v7n_pair_phase.empty else pd.DataFrame()
        n_phase = int(len(ph))
        has_cross = bool(ph["phase_crossing_flag"].astype(bool).any()) if (not ph.empty and "phase_crossing_flag" in ph.columns) else False
        if np.isfinite(sign_consistency):
            if sign_consistency >= 0.85:
                artifact = "low_artifact_risk"
            elif sign_consistency >= 0.65:
                artifact = "moderate_artifact_risk"
            else:
                artifact = "high_projection_artifact_risk"
        else:
            artifact = "unavailable_no_state_distance_check"
        rows.append({
            "field_a": a,
            "field_b": b,
            "projection_distance_sign_consistency_fraction": sign_consistency,
            "n_v7n_phases": n_phase,
            "v7n_has_crossing_phase": has_cross,
            "relation_artifact_risk_label": artifact,
            "artifact_risk_interpretation": "projection and distance progress agree on pairwise sign for most days" if artifact == "low_artifact_risk" else artifact,
        })
    return pd.DataFrame(rows)


def build_prepost_sensitivity(matrices: dict[str, np.ndarray], window: pd.Series) -> pd.DataFrame:
    if not matrices:
        return pd.DataFrame([{"sensitivity_status": "unavailable_state_rebuild"}])
    n_days = next(iter(matrices.values())).shape[0]
    base = {
        "baseline": (0, 0, 0, 0),
        "inner_minus2": (SENSITIVITY_SHIFT_DAYS, 0, 0, -SENSITIVITY_SHIFT_DAYS),
        "outer_plus2": (-SENSITIVITY_SHIFT_DAYS, 0, 0, SENSITIVITY_SHIFT_DAYS),
        "shift_early2": (-SENSITIVITY_SHIFT_DAYS, -SENSITIVITY_SHIFT_DAYS, -SENSITIVITY_SHIFT_DAYS, -SENSITIVITY_SHIFT_DAYS),
        "shift_late2": (SENSITIVITY_SHIFT_DAYS, SENSITIVITY_SHIFT_DAYS, SENSITIVITY_SHIFT_DAYS, SENSITIVITY_SHIFT_DAYS),
    }
    start, end = int(window["analysis_window_start"]), int(window["analysis_window_end"])
    rows = []
    def curve_for(field: str, case_offsets: tuple[int, int, int, int]) -> pd.Series:
        full = matrices[field]
        ps = max(0, min(n_days - 1, int(window["pre_period_start"]) + case_offsets[0]))
        pe = max(0, min(n_days - 1, int(window["pre_period_end"]) + case_offsets[1]))
        qs = max(0, min(n_days - 1, int(window["post_period_start"]) + case_offsets[2]))
        qe = max(0, min(n_days - 1, int(window["post_period_end"]) + case_offsets[3]))
        if ps > pe or qs > qe:
            return pd.Series(dtype=float)
        pre = _nanmean_rows(full[ps : pe + 1])
        post = _nanmean_rows(full[qs : qe + 1])
        if pre is None or post is None:
            return pd.Series(dtype=float)
        vec = post - pre
        denom = float(np.nansum(np.square(vec)))
        vals = {}
        for day in range(start, end + 1):
            x = full[day]
            if denom > 1e-12 and np.all(np.isfinite(x)):
                vals[day] = float(np.nansum((x - pre) * vec) / denom)
        return pd.Series(vals, dtype=float)
    for case, offs in base.items():
        curves = {f: curve_for(f, offs) for f in FIELDS}
        for a, b in combinations(FIELDS, 2):
            if curves[a].empty or curves[b].empty:
                continue
            idx = sorted(set(curves[a].index).intersection(set(curves[b].index)))
            if not idx:
                continue
            diff = curves[a].loc[idx].to_numpy(dtype=float) - curves[b].loc[idx].to_numpy(dtype=float)
            signs = np.sign(diff)
            n_cross = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
            rows.append({
                "sensitivity_case": case,
                "field_a": a,
                "field_b": b,
                "mean_diff_A_minus_B": float(np.nanmean(diff)),
                "median_diff_A_minus_B": float(np.nanmedian(diff)),
                "n_sign_crossings": n_cross,
                "fraction_A_progress_gt_B": float(np.mean(diff > 0)),
                "fraction_B_progress_gt_A": float(np.mean(diff < 0)),
            })
    return pd.DataFrame(rows)


def build_foundation_decisions(prepost: pd.DataFrame, direction_summary: pd.DataFrame, component_summary: pd.DataFrame, pair_comp: pd.DataFrame, artifact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp_map = component_summary.set_index("field").to_dict("index") if not component_summary.empty and "field" in component_summary.columns else {}
    dir_map = direction_summary.set_index("field").to_dict("index") if not direction_summary.empty and "field" in direction_summary.columns else {}
    rows = []
    for _, r in prepost.iterrows():
        field = str(r["field"])
        pp = str(r.get("prepost_foundation_label", "unavailable"))
        dm = dir_map.get(field, {})
        cm = comp_map.get(field, {})
        transition = str(dm.get("transition_direction_label", "unavailable"))
        proj = str(dm.get("projection_progress_validity_label", "unavailable"))
        single = str(cm.get("single_curve_adequacy_label", "unavailable"))
        if "unavailable" in transition or "unavailable" in proj or "unavailable" in single:
            label = "foundation_incomplete_state_rebuild_unavailable"
        elif pp.startswith("strong") and transition.startswith("dominant") and proj.startswith("valid") and single == "single_curve_adequate":
            label = "valid_single_process_curve"
        elif pp in {"strong_prepost_foundation", "usable_prepost_foundation"} and proj not in {"projection_progress_not_valid", "projection_distance_inconsistent"} and single != "single_curve_not_adequate":
            label = "usable_with_caution"
        elif single in {"multi_component_transition", "single_curve_not_adequate"}:
            label = "multi_component_process"
        elif "not" in pp or "not" in proj:
            label = "not_valid_as_single_curve"
        else:
            label = "usable_only_as_projection_diagnostic"
        rows.append({
            "field": field,
            "prepost_foundation_label": pp,
            "transition_direction_label": transition,
            "projection_progress_validity_label": proj,
            "single_curve_adequacy_label": single,
            "field_curve_foundation_label": label,
            "allowed_uses": "process diagnostic; layer-specific relation if pair comparable" if label != "not_valid_as_single_curve" else "diagnostic only",
            "prohibited_uses": "do not interpret as physical strength or causal lead; do not force single-marker order",
        })
    pair_rows = []
    art_map = {(str(r["field_a"]), str(r["field_b"])): r.to_dict() for _, r in artifact.iterrows()} if not artifact.empty else {}
    for _, r in pair_comp.iterrows():
        a, b = str(r["field_a"]), str(r["field_b"])
        art = art_map.get((a, b), art_map.get((b, a), {}))
        comp = str(r.get("pair_phase_comparability_label", "unavailable"))
        arisk = str(art.get("relation_artifact_risk_label", "unavailable"))
        can_curve = comp in {"phase_comparable", "phase_comparable_with_caution", "layer_specific_only"} and arisk in {"low_artifact_risk", "moderate_artifact_risk"}
        global_ok = comp == "phase_comparable" and arisk == "low_artifact_risk"
        if arisk.startswith("unavailable") and comp in {"phase_comparable", "phase_comparable_with_caution", "layer_specific_only"}:
            label = "foundation_incomplete_artifact_check_unavailable"
        else:
            label = "curve_relation_usable" if global_ok else ("curve_relation_usable_with_caution" if can_curve else "not_comparable")
        pair_rows.append({
            "field_a": a,
            "field_b": b,
            "phase_comparability_label": comp,
            "relation_artifact_risk_label": arisk,
            "can_use_curve_phase_relation": bool(can_curve),
            "can_use_global_lead_lag": bool(global_ok),
            "must_use_layer_specific_language": bool(comp == "layer_specific_only" or not global_ok),
            "can_test_near_equivalence": bool(can_curve),
            "pair_curve_foundation_label": label,
        })
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def _plot_outputs(paths: FoundationAuditPaths, daily: pd.DataFrame, artifact: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _ensure_dir(paths.figure_dir)
    if not daily.empty and {"field", "day", "projection_progress_raw", "distance_progress"}.issubset(daily.columns):
        for field in FIELDS:
            sub = daily[daily["field"].astype(str) == field]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(sub["day"], sub["projection_progress_raw"], label="projection_progress")
            ax.plot(sub["day"], sub["distance_progress"], label="distance_progress")
            ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
            ax.set_title(f"W45 {field}: projection vs distance progress")
            ax.set_xlabel("day index")
            ax.set_ylabel("progress")
            ax.legend()
            fig.tight_layout()
            fig.savefig(paths.figure_dir / f"w45_{field}_projection_vs_distance_progress_v7_o.png", dpi=150)
            plt.close(fig)
    # H/Jw detail.
    if not daily.empty:
        hp = daily[daily["field"].astype(str) == "H"]
        jp = daily[daily["field"].astype(str) == "Jw"]
        if not hp.empty and not jp.empty:
            merged = hp[["day", "projection_progress_raw", "distance_progress"]].merge(
                jp[["day", "projection_progress_raw", "distance_progress"]], on="day", suffixes=("_H", "_Jw")
            )
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(merged["day"], merged["projection_progress_raw_H"] - merged["projection_progress_raw_Jw"], label="projection H-Jw")
            ax.plot(merged["day"], merged["distance_progress_H"] - merged["distance_progress_Jw"], label="distance H-Jw")
            ax.axhline(0, linewidth=1)
            ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
            ax.set_title("W45 H/Jw: projection vs distance progress difference")
            ax.set_xlabel("day index")
            ax.set_ylabel("H - Jw")
            ax.legend()
            fig.tight_layout()
            fig.savefig(paths.figure_dir / "w45_H_Jw_projection_vs_distance_diff_v7_o.png", dpi=150)
            plt.close(fig)


def _write_summary(paths: FoundationAuditPaths, input_audit: dict, field_decision: pd.DataFrame, pair_decision: pd.DataFrame) -> None:
    lines = [
        "# w45_process_curve_foundation_audit_v7_o",
        "",
        "## Purpose",
        "This audit tests whether the process/progress curves used by V7-n can serve as an implementation-layer basis for W45 ordering. It does not claim the curves are absolutely correct.",
        "",
        "## Input coverage",
        f"- state_rebuild_status: {input_audit.get('state_rebuild_status')}",
        f"- input_representation: current V7-e progress representation",
        "",
        "## Six assumptions audited",
        "1. pre/post prototypes are usable statistical state prototypes.",
        "2. pre→post direction explains a meaningful share of the W45 trajectory.",
        "3. projection progress agrees with distance-to-pre/post progress.",
        "4. a high-dimensional field can be summarized by a single curve, or must be downgraded as multi-component.",
        "5. pairwise phase comparison is allowed only where both field curves are valid enough.",
        "6. curve relations are checked against projection/distance agreement and pre/post sensitivity to reduce projection-artifact risk.",
        "",
        "## Field foundation decisions",
    ]
    if not field_decision.empty:
        for _, r in field_decision.iterrows():
            lines.append(f"- {r['field']}: {r.get('field_curve_foundation_label')} | prepost={r.get('prepost_foundation_label')} | projection={r.get('projection_progress_validity_label')} | single_curve={r.get('single_curve_adequacy_label')}")
    lines += ["", "## Pair foundation decisions"]
    if not pair_decision.empty:
        for _, r in pair_decision.iterrows():
            lines.append(f"- {r['field_a']}-{r['field_b']}: {r.get('pair_curve_foundation_label')} | comparable={r.get('phase_comparability_label')} | artifact={r.get('relation_artifact_risk_label')} | global_lead_lag={r.get('can_use_global_lead_lag')}")
    lines += [
        "",
        "## Prohibited interpretations",
        "- Do not treat process curves as absolutely correct.",
        "- Do not treat progress difference as physical strength difference or causality.",
        "- Do not interpret projection-based crossing as a physical crossing without distance/residual support.",
        "- Do not convert order-not-resolved into synchrony; near-equivalence needs an explicit equivalence margin.",
        "- If field_curve_foundation_label is multi_component_process or not_valid_as_single_curve, do not use that field for clean global lead/lag.",
    ]
    _write_text("\n".join(lines), paths.output_dir / "w45_process_curve_foundation_summary_v7_o.md")


def run_w45_process_curve_foundation_audit_v7_o(v7_root: Path | str | None = None) -> None:
    paths = _resolve_paths(Path(v7_root) if v7_root is not None else None)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)

    tables = _load_existing_outputs(paths)
    window = _get_window_row(paths)
    matrices, matrix_meta, rebuild_audit = _try_rebuild_observed_field_matrices(paths)

    input_audit = {
        "status": "checked",
        "created_at": _now_iso(),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "fields": FIELDS,
        "v7e_output_dir": str(paths.v7e_output_dir),
        "v7m_output_dir": str(paths.v7m_output_dir),
        "v7n_output_dir": str(paths.v7n_output_dir),
        "output_dir": str(paths.output_dir),
        "process_curve_status": "candidate_foundation_not_absolute_truth",
        **rebuild_audit,
        "field_matrix_meta": matrix_meta,
    }
    _write_json(input_audit, paths.output_dir / "input_audit_v7_o.json")

    prepost = build_prepost_foundation(tables, matrices, window)
    daily, direction_summary, component_detail, component_summary = build_projection_direction_and_validity(matrices, window)
    pair_comp = build_pair_phase_comparability(direction_summary.merge(component_summary, on="field", how="outer") if not direction_summary.empty else direction_summary, tables["v7n_pair_type"])
    # Build preliminary field decision before pair comparability needs final labels.
    field_decision_tmp, _ = build_foundation_decisions(prepost, direction_summary, component_summary, pd.DataFrame(), pd.DataFrame())
    pair_comp = build_pair_phase_comparability(field_decision_tmp, tables["v7n_pair_type"])
    artifact = build_pair_artifact_risk(daily, tables["v7n_pair_daily"], tables["v7n_pair_phase"])
    sensitivity = build_prepost_sensitivity(matrices, window)
    field_decision, pair_decision = build_foundation_decisions(prepost, direction_summary, component_summary, pair_comp, artifact)

    # H/Jw detail combines the pair-level foundation row, artifact row, and sensitivity rows.
    hjw_rows = []
    if not pair_decision.empty:
        hjw_rows.append(pair_decision[((pair_decision["field_a"] == "H") & (pair_decision["field_b"] == "Jw")) | ((pair_decision["field_a"] == "Jw") & (pair_decision["field_b"] == "H"))])
    if not artifact.empty:
        hjw_rows.append(artifact[((artifact["field_a"] == "H") & (artifact["field_b"] == "Jw")) | ((artifact["field_a"] == "Jw") & (artifact["field_b"] == "H"))])
    hjw_detail = pd.concat([x for x in hjw_rows if not x.empty], axis=0, ignore_index=True) if hjw_rows else pd.DataFrame()

    _write_csv(prepost, paths.output_dir / "w45_field_prepost_foundation_v7_o.csv")
    _write_csv(daily, paths.output_dir / "w45_field_projection_distance_daily_v7_o.csv")
    _write_csv(direction_summary, paths.output_dir / "w45_field_transition_direction_audit_v7_o.csv")
    _write_csv(direction_summary, paths.output_dir / "w45_field_projection_progress_validity_v7_o.csv")
    _write_csv(component_detail, paths.output_dir / "w45_field_component_contribution_v7_o.csv")
    _write_csv(component_summary, paths.output_dir / "w45_field_single_curve_adequacy_v7_o.csv")
    _write_csv(pair_comp, paths.output_dir / "w45_pair_phase_comparability_v7_o.csv")
    _write_csv(artifact, paths.output_dir / "w45_pair_relation_artifact_risk_v7_o.csv")
    _write_csv(sensitivity, paths.output_dir / "w45_prepost_sensitivity_relation_v7_o.csv")
    _write_csv(field_decision, paths.output_dir / "w45_process_curve_foundation_decision_fields_v7_o.csv")
    _write_csv(pair_decision, paths.output_dir / "w45_process_curve_foundation_decision_pairs_v7_o.csv")
    _write_csv(hjw_detail, paths.output_dir / "w45_H_Jw_curve_foundation_detail_v7_o.csv")

    _plot_outputs(paths, daily, artifact, sensitivity)
    _write_summary(paths, input_audit, field_decision, pair_decision)
    _write_json(
        {
            "status": "success",
            "created_at": _now_iso(),
            "output_tag": OUTPUT_TAG,
            "window_id": WINDOW_ID,
            "anchor_day": ANCHOR_DAY,
            "fields": FIELDS,
            "input_representation": "current_v7e_progress_profile",
            "does_not_claim_absolute_correctness": True,
            "state_rebuild_status": input_audit.get("state_rebuild_status"),
            "outputs": {
                "field_prepost_foundation": "w45_field_prepost_foundation_v7_o.csv",
                "field_transition_direction": "w45_field_transition_direction_audit_v7_o.csv",
                "field_projection_validity": "w45_field_projection_progress_validity_v7_o.csv",
                "field_single_curve_adequacy": "w45_field_single_curve_adequacy_v7_o.csv",
                "pair_phase_comparability": "w45_pair_phase_comparability_v7_o.csv",
                "pair_artifact_risk": "w45_pair_relation_artifact_risk_v7_o.csv",
                "field_decision": "w45_process_curve_foundation_decision_fields_v7_o.csv",
                "pair_decision": "w45_process_curve_foundation_decision_pairs_v7_o.csv",
                "summary": "w45_process_curve_foundation_summary_v7_o.md",
            },
        },
        paths.output_dir / "run_meta.json",
    )
    print(f"[V7-o] wrote outputs to {paths.output_dir}")


__all__ = ["run_w45_process_curve_foundation_audit_v7_o"]
