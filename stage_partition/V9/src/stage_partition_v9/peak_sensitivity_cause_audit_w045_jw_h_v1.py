"""
W045 Jw/H peak-sensitivity cause audit v1.

Purpose
-------
This module is a *cause audit* for the high rule sensitivity found in the
existing V9 peak-selection sensitivity outputs.  It deliberately does not create
new accepted transition windows, does not replace V9 peak_all_windows_v9_a, and
does not assign physical sub-peak interpretations by default.

Scope
-----
    window: W045 only
    objects: H, Jw only

Main question
-------------
Why are W045 H/Jw selected peak days sensitive to detection/selection rules?
The diagnosis separates implementation risk, broad-plateau behavior, search
window mixing, rule semantic effects, profile-component mixing, and physically
separable multi-candidate peaks.

Interpretation boundary
-----------------------
Outputs are audit evidence and route-selection diagnostics.  A finding of
multiple clusters is not automatically a physical sub-peak result.  Physical
interpretation is allowed only when candidate clusters are also profile-distinct.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

VERSION = "W045_Jw_H_peak_sensitivity_cause_audit_v1"
OUTPUT_TAG = "peak_sensitivity_cause_audit_w045_jw_h_v1"
TARGET_WINDOW = "W045"
TARGET_OBJECTS = ("H", "Jw")
FACTOR_COLUMNS = ("smoothing", "detector_width", "band_half_width", "search_mode", "selection_rule")


@dataclass(frozen=True)
class W045JwHPeakSensitivityCauseAuditSettings:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    target_window: str = TARGET_WINDOW
    target_objects: Tuple[str, ...] = TARGET_OBJECTS
    cluster_gap_days: int = 4
    near_tie_days: int = 1
    boundary_risk_days: int = 2
    stable_cluster_min_fraction: float = 0.15
    rule_locked_fraction: float = 0.75
    dominant_factor_eta2_threshold: float = 0.35
    plateau_relative_gap_threshold: float = 0.05
    multi_peak_relative_gap_threshold: float = 0.25
    sharp_peak_relative_gap_threshold: float = 0.25
    plateau_min_width_days: int = 5
    profile_window_half_width: int = 4
    centroid_distinct_threshold_deg: float = 1.0
    spread_distinct_threshold_deg: float = 1.0
    amplitude_relative_distinct_threshold: float = 0.15
    shape_distance_distinct_threshold: float = 0.20
    make_figures: bool = True
    recompute_score_landscape_if_possible: bool = True
    physical_interpretation_included: bool = False
    rerun_changepoint_detection: bool = False


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _as_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x, default: Optional[int] = None) -> Optional[int]:
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def _nanmean_no_warning(arr, axis=None):
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    count = finite.sum(axis=axis)
    total = np.where(finite, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    if np.isscalar(mean):
        return np.nan if count == 0 else float(mean)
    mean = np.asarray(mean, dtype=float)
    return np.where(count > 0, mean, np.nan)


def _stage_root_from_v9_root(v9_root: Path) -> Path:
    return v9_root.parent


def _project_root_from_v9_root(v9_root: Path) -> Path:
    return v9_root.parents[1]


def _default_smooth9_path(v9_root: Path) -> Path:
    return _project_root_from_v9_root(v9_root) / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _default_smooth5_path(v9_root: Path) -> Path:
    return _project_root_from_v9_root(v9_root) / "foundation" / "V1" / "outputs" / "baseline_smooth5_a" / "preprocess" / "smoothed_fields.npz"


def _resolve_paths(v9_root: Path) -> Dict[str, Path]:
    v9_root = Path(os.environ.get("W045_JWH_AUDIT_V9_ROOT", str(v9_root))).resolve()
    sensitivity_cross = Path(os.environ.get(
        "W045_JWH_AUDIT_SENSITIVITY_CROSS_DIR",
        str(v9_root / "outputs" / "peak_selection_sensitivity_v9_a" / "cross_window"),
    ))
    baseline_cross = Path(os.environ.get(
        "W045_JWH_AUDIT_BASELINE_CROSS_DIR",
        str(v9_root / "outputs" / "peak_all_windows_v9_a" / "cross_window"),
    ))
    out_dir = Path(os.environ.get(
        "W045_JWH_AUDIT_OUTPUT_DIR",
        str(v9_root / "outputs" / OUTPUT_TAG),
    ))
    smooth9 = Path(os.environ.get("W045_JWH_AUDIT_SMOOTH9_FIELDS", str(_default_smooth9_path(v9_root))))
    smooth5 = Path(os.environ.get("W045_JWH_AUDIT_SMOOTH5_FIELDS", str(_default_smooth5_path(v9_root))))
    return {
        "v9_root": v9_root,
        "stage_root": _stage_root_from_v9_root(v9_root),
        "project_root": _project_root_from_v9_root(v9_root),
        "sensitivity_cross": sensitivity_cross,
        "baseline_cross": baseline_cross,
        "out_dir": out_dir,
        "cross_out": out_dir / "cross_window",
        "fig_out": out_dir / "figures",
        "log_out": v9_root / "logs" / OUTPUT_TAG,
        "smooth9": smooth9,
        "smooth5": smooth5,
    }


def _require_inputs(paths: Dict[str, Path]) -> None:
    required = [
        paths["sensitivity_cross"] / "object_peak_selection_by_config.csv",
        paths["sensitivity_cross"] / "object_profile_build_audit.csv",
        paths["sensitivity_cross"] / "baseline_reproduction_audit.csv",
        paths["baseline_cross"] / "window_scope_registry_v9_peak_all_windows_a.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required audit inputs:\n" + "\n".join(missing))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_input_tables(paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    sens = paths["sensitivity_cross"]
    base = paths["baseline_cross"]
    return {
        "object_peak_selection_by_config": _read_csv(sens / "object_peak_selection_by_config.csv"),
        "object_profile_build_audit": _read_csv(sens / "object_profile_build_audit.csv"),
        "baseline_reproduction_audit": _read_csv(sens / "baseline_reproduction_audit.csv"),
        "object_peak_sensitivity_summary": _read_csv(sens / "object_peak_sensitivity_summary.csv"),
        "sensitivity_config_grid": _read_csv(sens / "sensitivity_config_grid.csv"),
        "scope_registry": _read_csv(base / "window_scope_registry_v9_peak_all_windows_a.csv"),
        "raw_profile_detector_scores": _read_csv(base / "raw_profile_detector_scores_all_windows.csv"),
        "v9_candidate_registry": _read_csv(base / "object_profile_window_registry_all_windows.csv"),
        "v9_object_peak_registry": _read_csv(base / "cross_window_object_peak_registry.csv"),
    }


def _filter_selection(df: pd.DataFrame, settings: W045JwHPeakSensitivityCauseAuditSettings) -> pd.DataFrame:
    out = df[(df["window_id"].astype(str) == settings.target_window) & (df["object"].astype(str).isin(settings.target_objects))].copy()
    out["selected_peak_day"] = pd.to_numeric(out["selected_peak_day"], errors="coerce")
    return out.reset_index(drop=True)


def _cluster_days_for_object(days: Sequence[float], gap_days: int) -> Dict[int, str]:
    vals = sorted({int(round(float(x))) for x in days if np.isfinite(x)})
    if not vals:
        return {}
    groups: List[List[int]] = []
    current = [vals[0]]
    for d in vals[1:]:
        if d - current[-1] <= gap_days:
            current.append(d)
        else:
            groups.append(current)
            current = [d]
    groups.append(current)
    mapping: Dict[int, str] = {}
    for i, group in enumerate(groups, start=1):
        label = f"C{i:02d}_{min(group):03d}_{max(group):03d}"
        for d in group:
            mapping[d] = label
    return mapping


def _add_day_clusters(sel: pd.DataFrame, settings: W045JwHPeakSensitivityCauseAuditSettings) -> pd.DataFrame:
    rows = []
    for obj, sub in sel.groupby("object", sort=True):
        mapping = _cluster_days_for_object(sub["selected_peak_day"].tolist(), settings.cluster_gap_days)
        tmp = sub.copy()
        tmp["selected_peak_day_int"] = tmp["selected_peak_day"].round().astype("Int64")
        tmp["cluster_id"] = tmp["selected_peak_day_int"].map(lambda x: mapping.get(int(x), "NO_CLUSTER") if not pd.isna(x) else "NO_CLUSTER")
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _implementation_consistency_audit(tables: Dict[str, pd.DataFrame], settings: W045JwHPeakSensitivityCauseAuditSettings) -> pd.DataFrame:
    prof = tables["object_profile_build_audit"].copy()
    repro = tables["baseline_reproduction_audit"].copy()
    prof = prof[prof["object"].astype(str).isin(settings.target_objects)].copy() if not prof.empty else prof
    repro = repro[(repro["window_id"].astype(str) == settings.target_window) & (repro["object"].astype(str).isin(settings.target_objects))].copy() if not repro.empty else repro
    rows = []
    for obj in settings.target_objects:
        psub = prof[prof["object"].astype(str) == obj] if not prof.empty else pd.DataFrame()
        rsub = repro[repro["object"].astype(str) == obj] if not repro.empty else pd.DataFrame()
        repro_match = bool(rsub["match_flag"].fillna(False).astype(bool).all()) if not rsub.empty and "match_flag" in rsub.columns else False
        delta_max = float(pd.to_numeric(rsub.get("delta_day", pd.Series(dtype=float)), errors="coerce").abs().max()) if not rsub.empty else np.nan
        helper_values = sorted(psub["used_v7_profile_helper"].dropna().astype(str).unique().tolist()) if not psub.empty and "used_v7_profile_helper" in psub.columns else []
        dims = ";".join(psub.get("profile_shape", pd.Series(dtype=str)).astype(str).unique().tolist()) if not psub.empty else ""
        lat_span = ""
        if not psub.empty and {"lat_min", "lat_max"}.issubset(psub.columns):
            lat_span = ";".join([f"{r.get('lat_min')}..{r.get('lat_max')}" for _, r in psub.iterrows()])
        if rsub.empty or not repro_match:
            status = "FAIL"
            note = "Baseline reproduction for W045 object is missing or mismatched. Do not physically interpret sensitivity."
        elif helper_values and set(helper_values) == {"False"}:
            status = "WARN"
            note = "Baseline day reproduces, but profile build audit reports used_v7_profile_helper=False; keep helper-consistency risk flag."
        else:
            status = "PASS"
            note = "Baseline reproduction and profile helper audit do not show a blocking inconsistency."
        rows.append({
            "window_id": settings.target_window,
            "object": obj,
            "n_profile_rows": int(len(psub)),
            "profile_shapes": dims,
            "profile_lat_spans": lat_span,
            "used_v7_profile_helper_values": ";".join(helper_values),
            "baseline_reproduction_rows": int(len(rsub)),
            "baseline_reproduction_match": repro_match,
            "max_abs_reproduction_delta_day": delta_max,
            "implementation_status": status,
            "risk_note": note,
        })
    return pd.DataFrame(rows)


def _cluster_summary(sel_clustered: pd.DataFrame, settings: W045JwHPeakSensitivityCauseAuditSettings) -> pd.DataFrame:
    rows = []
    total_by_obj = sel_clustered.groupby("object").size().to_dict()
    for (obj, cid), sub in sel_clustered.groupby(["object", "cluster_id"], sort=True):
        valid_days = pd.to_numeric(sub["selected_peak_day"], errors="coerce").dropna()
        n_total = total_by_obj.get(obj, len(sub))
        if valid_days.empty:
            continue
        rule_counts = sub["selection_rule"].astype(str).value_counts(normalize=True).to_dict() if "selection_rule" in sub.columns else {}
        search_counts = sub["search_mode"].astype(str).value_counts(normalize=True).to_dict() if "search_mode" in sub.columns else {}
        smoothing_counts = sub["smoothing"].astype(str).value_counts(normalize=True).to_dict() if "smoothing" in sub.columns else {}
        dom_rule, dom_rule_frac = (max(rule_counts.items(), key=lambda kv: kv[1]) if rule_counts else ("", np.nan))
        dom_search, dom_search_frac = (max(search_counts.items(), key=lambda kv: kv[1]) if search_counts else ("", np.nan))
        frac = len(sub) / max(1, n_total)
        if dom_rule_frac >= settings.rule_locked_fraction or dom_search_frac >= settings.rule_locked_fraction:
            status = "RULE_OR_SEARCH_LOCKED_CLUSTER"
        elif frac >= settings.stable_cluster_min_fraction and all(smoothing_counts.get(s, 0.0) > 0 for s in ["smooth5", "smooth9"]):
            status = "STABLE_CROSS_SMOOTH_CLUSTER"
        elif frac >= settings.stable_cluster_min_fraction:
            status = "STABLE_SINGLE_SCALE_CLUSTER"
        else:
            status = "MINOR_CLUSTER"
        rows.append({
            "window_id": settings.target_window,
            "object": obj,
            "cluster_id": cid,
            "day_min": int(valid_days.min()),
            "day_max": int(valid_days.max()),
            "day_median": float(valid_days.median()),
            "n_configs": int(len(sub)),
            "config_fraction": float(frac),
            "smooth5_fraction": float(smoothing_counts.get("smooth5", 0.0)),
            "smooth9_fraction": float(smoothing_counts.get("smooth9", 0.0)),
            "narrow_fraction": float(search_counts.get("narrow_search", 0.0)),
            "baseline_search_fraction": float(search_counts.get("baseline_search", 0.0)),
            "wide_fraction": float(search_counts.get("wide_search", 0.0)),
            "dominant_selection_rule": dom_rule,
            "dominant_selection_rule_fraction": float(dom_rule_frac),
            "dominant_search_mode": dom_search,
            "dominant_search_mode_fraction": float(dom_search_frac),
            "cluster_status": status,
        })
    return pd.DataFrame(rows).sort_values(["object", "day_median"]).reset_index(drop=True)


def _eta_squared_for_factor(df: pd.DataFrame, factor: str, value_col: str = "selected_peak_day") -> float:
    sub = df[[factor, value_col]].dropna().copy()
    if sub.empty or sub[factor].nunique() <= 1:
        return np.nan
    y = pd.to_numeric(sub[value_col], errors="coerce")
    overall = y.mean()
    ss_total = float(((y - overall) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for _, g in sub.groupby(factor):
        yg = pd.to_numeric(g[value_col], errors="coerce")
        ss_between += len(yg) * float((yg.mean() - overall) ** 2)
    return float(ss_between / ss_total)


def _factor_contribution(sel_clustered: pd.DataFrame, settings: W045JwHPeakSensitivityCauseAuditSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    contrib_rows = []
    cross_rows = []
    for obj, sub in sel_clustered.groupby("object", sort=True):
        for factor in FACTOR_COLUMNS:
            if factor not in sub.columns:
                continue
            eta2 = _eta_squared_for_factor(sub, factor)
            med = sub.groupby(factor)["selected_peak_day"].median()
            iqr = sub.groupby(factor)["selected_peak_day"].apply(lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)) if len(pd.to_numeric(x, errors="coerce").dropna()) else np.nan)
            med_range = float(med.max() - med.min()) if not med.empty else np.nan
            iqr_range = float(iqr.max() - iqr.min()) if not iqr.empty else np.nan
            if eta2 >= settings.dominant_factor_eta2_threshold:
                hint = f"{factor}_dominant"
            elif eta2 >= 0.15:
                hint = f"{factor}_moderate"
            else:
                hint = f"{factor}_weak"
            contrib_rows.append({
                "window_id": settings.target_window,
                "object": obj,
                "factor": factor,
                "day_variance_explained_eta2": eta2,
                "median_day_range_across_levels": med_range,
                "iqr_range_across_levels": iqr_range,
                "interpretation_hint": hint,
            })
            tab = pd.crosstab(sub[factor], sub["cluster_id"])
            for level in tab.index:
                denom_level = max(1, int(tab.loc[level].sum()))
                for cid in tab.columns:
                    n = int(tab.loc[level, cid])
                    denom_cluster = max(1, int(tab[cid].sum()))
                    cross_rows.append({
                        "window_id": settings.target_window,
                        "object": obj,
                        "factor": factor,
                        "factor_level": str(level),
                        "cluster_id": str(cid),
                        "n_configs": n,
                        "fraction_within_factor_level": float(n / denom_level),
                        "fraction_within_cluster": float(n / denom_cluster),
                    })
    contrib = pd.DataFrame(contrib_rows)
    if not contrib.empty:
        contrib["dominant_factor_rank"] = contrib.groupby("object")["day_variance_explained_eta2"].rank(ascending=False, method="first")
    return contrib, pd.DataFrame(cross_rows)


def _search_boundary_audit(sel_clustered: pd.DataFrame, tables: Dict[str, pd.DataFrame], settings: W045JwHPeakSensitivityCauseAuditSettings) -> pd.DataFrame:
    scope = tables["scope_registry"]
    scope = scope[scope["window_id"].astype(str) == settings.target_window].copy()
    if scope.empty:
        core_start = core_end = system_start = system_end = np.nan
    else:
        r = scope.iloc[0]
        core_start, core_end = _as_float(r.get("core_start")), _as_float(r.get("core_end"))
        system_start, system_end = _as_float(r.get("system_window_start")), _as_float(r.get("system_window_end"))
    rows = []
    for _, r in sel_clustered.iterrows():
        d = _as_float(r.get("selected_peak_day"))
        left = _as_float(r.get("search_start_day"))
        right = _as_float(r.get("search_end_day"))
        distance_left = d - left if np.isfinite(d) and np.isfinite(left) else np.nan
        distance_right = right - d if np.isfinite(d) and np.isfinite(right) else np.nan
        boundary_flag = bool((np.isfinite(distance_left) and distance_left <= settings.boundary_risk_days) or (np.isfinite(distance_right) and distance_right <= settings.boundary_risk_days))
        outside_core = bool(np.isfinite(d) and np.isfinite(core_start) and not (core_start <= d <= core_end))
        outside_system = bool(np.isfinite(d) and np.isfinite(system_start) and not (system_start <= d <= system_end))
        rows.append({
            "window_id": settings.target_window,
            "object": r.get("object"),
            "config_id": r.get("config_id"),
            "search_mode": r.get("search_mode"),
            "search_start": left,
            "search_end": right,
            "selected_day": d,
            "cluster_id": r.get("cluster_id"),
            "distance_to_anchor": r.get("distance_to_anchor"),
            "distance_to_left_boundary": distance_left,
            "distance_to_right_boundary": distance_right,
            "boundary_risk_flag": boundary_flag,
            "outside_core_window_flag": outside_core,
            "outside_system_window_flag": outside_system,
        })
    return pd.DataFrame(rows)


def _try_import_sensitivity_module(v9_root: Path):
    src = v9_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from stage_partition_v9 import peak_selection_sensitivity_v9_a as sens
        return sens, "OK"
    except Exception as exc:
        return None, f"IMPORT_FAILED: {exc}"


def _score_landscape_unavailable(reason: str, settings: W045JwHPeakSensitivityCauseAuditSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame([{
        "window_id": settings.target_window,
        "object": obj,
        "config_id": "UNAVAILABLE",
        "n_local_peaks": np.nan,
        "top1_day": np.nan,
        "top1_score": np.nan,
        "top2_day": np.nan,
        "top2_score": np.nan,
        "top1_top2_gap": np.nan,
        "top1_top2_relative_gap": np.nan,
        "score_plateau_width": np.nan,
        "boundary_peak_flag": np.nan,
        "landscape_type": "UNAVAILABLE",
        "unavailable_reason": reason,
    } for obj in settings.target_objects])
    by_day = pd.DataFrame([{
        "window_id": settings.target_window,
        "object": obj,
        "config_id": "UNAVAILABLE",
        "day": np.nan,
        "detector_score": np.nan,
        "score_valid": False,
        "unavailable_reason": reason,
    } for obj in settings.target_objects])
    return by_day, summary


def _local_peak_days(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.sort_values("day").copy()
    vals = pd.to_numeric(df["detector_score"], errors="coerce").to_numpy(dtype=float)
    days = pd.to_numeric(df["day"], errors="coerce").to_numpy(dtype=float)
    rows = []
    for i in range(len(vals)):
        if not np.isfinite(vals[i]):
            continue
        left = vals[i - 1] if i > 0 else -np.inf
        right = vals[i + 1] if i < len(vals) - 1 else -np.inf
        if vals[i] >= left and vals[i] >= right:
            rows.append({"day": int(days[i]), "detector_score": float(vals[i])})
    return pd.DataFrame(rows)


def _classify_landscape(top1_score: float, top2_score: float, plateau_width: int, n_local: int, boundary_flag: bool, settings: W045JwHPeakSensitivityCauseAuditSettings) -> str:
    if boundary_flag:
        return "EDGE_PEAK"
    if not np.isfinite(top1_score):
        return "LOW_CONTRAST_NOISY"
    if not np.isfinite(top2_score):
        return "SHARP_SINGLE_PEAK"
    rel_gap = (top1_score - top2_score) / max(abs(top1_score), 1e-12)
    if plateau_width >= settings.plateau_min_width_days and rel_gap <= settings.plateau_relative_gap_threshold:
        return "FLAT_PLATEAU"
    if n_local >= 2 and rel_gap <= settings.multi_peak_relative_gap_threshold:
        return "MULTI_LOCAL_PEAK"
    if rel_gap >= settings.sharp_peak_relative_gap_threshold:
        return "SHARP_SINGLE_PEAK"
    return "LOW_CONTRAST_NOISY"


def _compute_score_landscape(paths: Dict[str, Path], tables: Dict[str, pd.DataFrame], settings: W045JwHPeakSensitivityCauseAuditSettings) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if not settings.recompute_score_landscape_if_possible:
        by_day, summary = _score_landscape_unavailable("recompute_score_landscape_if_possible=False", settings)
        return by_day, summary, "SKIPPED_BY_SETTINGS"
    missing_fields = [str(paths[k]) for k in ("smooth9", "smooth5") if not paths[k].exists()]
    if missing_fields:
        by_day, summary = _score_landscape_unavailable("Missing smoothed field file(s): " + "; ".join(missing_fields), settings)
        return by_day, summary, "UNAVAILABLE_MISSING_FIELDS"
    sens, status = _try_import_sensitivity_module(paths["v9_root"])
    if sens is None:
        by_day, summary = _score_landscape_unavailable(status, settings)
        return by_day, summary, status
    try:
        v7multi = sens._import_v7_module(paths["stage_root"])
        profiles_by_smooth = {}
        for smooth_name, field_path in [("smooth9", paths["smooth9"]), ("smooth5", paths["smooth5"] )]:
            profiles, _audit = sens._build_profiles_for_smoothing(field_path, v7multi=v7multi)
            profiles_by_smooth[smooth_name] = profiles
        scopes = tables["scope_registry"]
        scope = scopes[scopes["window_id"].astype(str) == settings.target_window].iloc[0]
        configs = sens._config_grid(sens.PeakSelectionSensitivitySettings())
        rows = []
        summary_rows = []
        total = len(configs) * len(settings.target_objects)
        done = 0
        for cfg in configs:
            s_start, s_end = sens._search_range(scope, cfg.search_mode)
            for obj in settings.target_objects:
                done += 1
                if done % 25 == 0:
                    _log(f"    score landscape {done}/{total}")
                prof = profiles_by_smooth[cfg.smoothing][obj]
                comp = sens._composite_profile(prof)
                score_df = sens._compute_detector_score(comp, cfg.detector_width)
                score_df = score_df[(score_df["day"] >= s_start) & (score_df["day"] <= s_end)].copy()
                score_df["window_id"] = settings.target_window
                score_df["object"] = obj
                score_df["config_id"] = cfg.config_id
                score_df["smoothing"] = cfg.smoothing
                score_df["detector_width"] = cfg.detector_width
                score_df["band_half_width"] = cfg.band_half_width
                score_df["search_mode"] = cfg.search_mode
                score_df["selection_rule"] = cfg.selection_rule
                score_df["search_start_day"] = s_start
                score_df["search_end_day"] = s_end
                rows.append(score_df)
                valid = score_df[score_df["score_valid"] == True].copy()
                peaks = _local_peak_days(valid)
                peaks = peaks.sort_values("detector_score", ascending=False) if not peaks.empty else peaks
                top1 = peaks.iloc[0] if len(peaks) >= 1 else None
                top2 = peaks.iloc[1] if len(peaks) >= 2 else None
                top1_score = float(top1["detector_score"]) if top1 is not None else np.nan
                top2_score = float(top2["detector_score"]) if top2 is not None else np.nan
                top1_day = int(top1["day"]) if top1 is not None else np.nan
                top2_day = int(top2["day"]) if top2 is not None else np.nan
                rel_gap = (top1_score - top2_score) / max(abs(top1_score), 1e-12) if np.isfinite(top1_score) and np.isfinite(top2_score) else np.nan
                plateau_threshold = top1_score - abs(top1_score) * settings.plateau_relative_gap_threshold if np.isfinite(top1_score) else np.nan
                plateau_width = int((pd.to_numeric(valid["detector_score"], errors="coerce") >= plateau_threshold).sum()) if np.isfinite(plateau_threshold) else 0
                boundary_flag = bool(np.isfinite(top1_day) and (top1_day - s_start <= settings.boundary_risk_days or s_end - top1_day <= settings.boundary_risk_days))
                landscape_type = _classify_landscape(top1_score, top2_score, plateau_width, len(peaks), boundary_flag, settings)
                summary_rows.append({
                    "window_id": settings.target_window,
                    "object": obj,
                    "config_id": cfg.config_id,
                    "smoothing": cfg.smoothing,
                    "detector_width": cfg.detector_width,
                    "band_half_width": cfg.band_half_width,
                    "search_mode": cfg.search_mode,
                    "selection_rule": cfg.selection_rule,
                    "search_start_day": s_start,
                    "search_end_day": s_end,
                    "n_local_peaks": int(len(peaks)),
                    "top1_day": top1_day,
                    "top1_score": top1_score,
                    "top2_day": top2_day,
                    "top2_score": top2_score,
                    "top1_top2_gap": top1_score - top2_score if np.isfinite(top1_score) and np.isfinite(top2_score) else np.nan,
                    "top1_top2_relative_gap": rel_gap,
                    "score_plateau_width": plateau_width,
                    "boundary_peak_flag": boundary_flag,
                    "landscape_type": landscape_type,
                    "unavailable_reason": "",
                })
        by_day = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return by_day, pd.DataFrame(summary_rows), "OK"
    except Exception as exc:
        by_day, summary = _score_landscape_unavailable(f"Score landscape recomputation failed: {type(exc).__name__}: {exc}", settings)
        return by_day, summary, "FAILED_RECOMPUTE"


def _resolve_npz_key(npz: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> str:
    keys = set(npz.files)
    for key in candidates:
        if key in keys:
            return key
    raise KeyError(f"None of keys {candidates} found. Available keys include {sorted(list(keys))[:20]}")


def _profile_features_unavailable(reason: str, settings: W045JwHPeakSensitivityCauseAuditSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    audit = pd.DataFrame([{
        "window_id": settings.target_window,
        "object": obj,
        "cluster_id": "UNAVAILABLE",
        "smoothing": "UNAVAILABLE",
        "day_median": np.nan,
        "amplitude_l2": np.nan,
        "centroid_lat": np.nan,
        "spread_lat": np.nan,
        "max_lat": np.nan,
        "north_south_contrast": np.nan,
        "profile_shape_change_score": np.nan,
        "component_mixture_status": "UNAVAILABLE",
        "unavailable_reason": reason,
    } for obj in settings.target_objects])
    distinct = pd.DataFrame([{
        "window_id": settings.target_window,
        "object": obj,
        "cluster_pair": "UNAVAILABLE",
        "day_gap": np.nan,
        "amplitude_relative_difference": np.nan,
        "centroid_difference": np.nan,
        "spread_difference": np.nan,
        "shape_distance": np.nan,
        "distinctness_score": np.nan,
        "distinctness_status": "INSUFFICIENT_EVIDENCE",
        "interpretation_allowed": False,
        "unavailable_reason": reason,
    } for obj in settings.target_objects])
    return audit, distinct


def _build_profiles_with_lat(paths: Dict[str, Path], smoothing: str):
    sens, status = _try_import_sensitivity_module(paths["v9_root"])
    if sens is None:
        raise RuntimeError(status)
    field_path = paths[smoothing]
    fields = sens._load_fields(field_path)
    lat, lon = fields["lat"], fields["lon"]
    v7multi = sens._import_v7_module(paths["stage_root"])
    out = {}
    for obj in TARGET_OBJECTS:
        spec = sens.OBJECT_SPECS[obj]
        role = spec["role"]
        arr = sens._to_year_day_lat_lon(fields[role], role, v7multi=v7multi, lat=lat, lon=lon)
        used_helper = False
        if v7multi is not None and hasattr(v7multi, "clean") and hasattr(v7multi.clean, "_build_object_profile"):
            try:
                prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
                used_helper = True
            except Exception:
                prof, target_lat, weights = sens._fallback_object_profile(arr, lat, lon, spec)
        else:
            prof, target_lat, weights = sens._fallback_object_profile(arr, lat, lon, spec)
        out[obj] = {"profile": np.asarray(prof, dtype=float), "lat": np.asarray(target_lat, dtype=float), "used_v7_profile_helper": used_helper}
    return out


def _composite_profile(prof: np.ndarray) -> np.ndarray:
    return _nanmean_no_warning(np.asarray(prof, dtype=float), axis=0)


def _profile_feature_vector(day_coord: np.ndarray, lat: np.ndarray, day: int, half_width: int) -> dict:
    n_days = day_coord.shape[0]
    d = int(np.clip(day, 0, n_days - 1))
    vec = np.asarray(day_coord[d], dtype=float)
    weights = np.abs(vec)
    if not np.isfinite(weights).any() or np.nansum(weights) <= 0:
        centroid = spread = max_lat = ns = np.nan
    else:
        denom = np.nansum(weights)
        centroid = float(np.nansum(lat * weights) / denom)
        spread = float(np.sqrt(np.nansum(((lat - centroid) ** 2) * weights) / denom))
        max_lat = float(lat[int(np.nanargmax(weights))])
        med_lat = float(np.nanmedian(lat))
        north = float(np.nansum(weights[lat >= med_lat]))
        south = float(np.nansum(weights[lat < med_lat]))
        ns = (north - south) / max(north + south, 1e-12)
    amplitude = float(np.sqrt(np.nansum(vec * vec))) if np.isfinite(vec).any() else np.nan
    pre_s = max(0, d - half_width)
    pre_e = max(0, d)
    post_s = min(n_days, d + 1)
    post_e = min(n_days, d + 1 + half_width)
    if pre_e <= pre_s or post_e <= post_s:
        shape_change = np.nan
    else:
        pre = _nanmean_no_warning(day_coord[pre_s:pre_e], axis=0)
        post = _nanmean_no_warning(day_coord[post_s:post_e], axis=0)
        shape_change = float(np.sqrt(np.nansum((post - pre) ** 2))) if np.isfinite(pre).any() and np.isfinite(post).any() else np.nan
    return {
        "amplitude_l2": amplitude,
        "centroid_lat": centroid,
        "spread_lat": spread,
        "max_lat": max_lat,
        "north_south_contrast": ns,
        "profile_shape_change_score": shape_change,
        "profile_vector": vec,
    }


def _profile_component_and_distinctness(paths: Dict[str, Path], cluster_summary: pd.DataFrame, settings: W045JwHPeakSensitivityCauseAuditSettings) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    missing_fields = [str(paths[k]) for k in ("smooth9", "smooth5") if not paths[k].exists()]
    if missing_fields:
        audit, distinct = _profile_features_unavailable("Missing smoothed field file(s): " + "; ".join(missing_fields), settings)
        return audit, distinct, "UNAVAILABLE_MISSING_FIELDS"
    try:
        profile_data = {smooth: _build_profiles_with_lat(paths, smooth) for smooth in ("smooth9", "smooth5")}
        audit_rows = []
        vector_store = {}
        major_clusters = cluster_summary[cluster_summary["config_fraction"] >= settings.stable_cluster_min_fraction].copy()
        if major_clusters.empty:
            major_clusters = cluster_summary.copy()
        for _, c in major_clusters.iterrows():
            obj = str(c["object"])
            cid = str(c["cluster_id"])
            day = int(round(float(c["day_median"])))
            for smoothing in ("smooth9", "smooth5"):
                entry = profile_data[smoothing][obj]
                comp = _composite_profile(entry["profile"])
                lat = entry["lat"]
                feat = _profile_feature_vector(comp, lat, day, settings.profile_window_half_width)
                vector_store[(obj, cid, smoothing)] = feat["profile_vector"]
                status = "COMPONENT_FEATURES_AVAILABLE"
                audit_rows.append({
                    "window_id": settings.target_window,
                    "object": obj,
                    "cluster_id": cid,
                    "smoothing": smoothing,
                    "day_median": day,
                    "amplitude_l2": feat["amplitude_l2"],
                    "centroid_lat": feat["centroid_lat"],
                    "spread_lat": feat["spread_lat"],
                    "max_lat": feat["max_lat"],
                    "north_south_contrast": feat["north_south_contrast"],
                    "profile_shape_change_score": feat["profile_shape_change_score"],
                    "used_v7_profile_helper": bool(entry["used_v7_profile_helper"]),
                    "component_mixture_status": status,
                    "unavailable_reason": "",
                })
        audit = pd.DataFrame(audit_rows)
        distinct_rows = []
        for obj, sub in audit.groupby("object", sort=True):
            clusters = sorted(sub["cluster_id"].unique().tolist())
            for c1, c2 in combinations(clusters, 2):
                # Use smooth9 as primary comparison; fallback to all smooth mean if needed.
                a1 = sub[(sub["cluster_id"] == c1) & (sub["smoothing"] == "smooth9")]
                a2 = sub[(sub["cluster_id"] == c2) & (sub["smoothing"] == "smooth9")]
                if a1.empty or a2.empty:
                    a1 = sub[sub["cluster_id"] == c1].head(1)
                    a2 = sub[sub["cluster_id"] == c2].head(1)
                r1, r2 = a1.iloc[0], a2.iloc[0]
                day_gap = abs(_as_float(r1["day_median"]) - _as_float(r2["day_median"]))
                amp1, amp2 = _as_float(r1["amplitude_l2"]), _as_float(r2["amplitude_l2"])
                amp_rel = abs(amp1 - amp2) / max(abs(amp1), abs(amp2), 1e-12) if np.isfinite(amp1) and np.isfinite(amp2) else np.nan
                cent_diff = abs(_as_float(r1["centroid_lat"]) - _as_float(r2["centroid_lat"]))
                spread_diff = abs(_as_float(r1["spread_lat"]) - _as_float(r2["spread_lat"]))
                v1 = vector_store.get((obj, c1, "smooth9"))
                v2 = vector_store.get((obj, c2, "smooth9"))
                if v1 is not None and v2 is not None:
                    denom = max(float(np.sqrt(np.nansum(v1 * v1))), float(np.sqrt(np.nansum(v2 * v2))), 1e-12)
                    shape_dist = float(np.sqrt(np.nansum((v1 - v2) ** 2)) / denom)
                else:
                    shape_dist = np.nan
                flags = [
                    np.isfinite(amp_rel) and amp_rel >= settings.amplitude_relative_distinct_threshold,
                    np.isfinite(cent_diff) and cent_diff >= settings.centroid_distinct_threshold_deg,
                    np.isfinite(spread_diff) and spread_diff >= settings.spread_distinct_threshold_deg,
                    np.isfinite(shape_dist) and shape_dist >= settings.shape_distance_distinct_threshold,
                ]
                score = int(sum(bool(x) for x in flags))
                if score >= 2:
                    status = "PHYSICAL_DISTINCT"
                    allowed = True
                elif score == 1:
                    status = "WEAKLY_DISTINCT"
                    allowed = False
                else:
                    status = "NOT_DISTINCT"
                    allowed = False
                distinct_rows.append({
                    "window_id": settings.target_window,
                    "object": obj,
                    "cluster_pair": f"{c1}__vs__{c2}",
                    "cluster_1": c1,
                    "cluster_2": c2,
                    "day_gap": day_gap,
                    "amplitude_relative_difference": amp_rel,
                    "centroid_difference": cent_diff,
                    "spread_difference": spread_diff,
                    "shape_distance": shape_dist,
                    "distinctness_score": score,
                    "distinctness_status": status,
                    "interpretation_allowed": bool(allowed),
                    "unavailable_reason": "",
                })
        distinct = pd.DataFrame(distinct_rows) if distinct_rows else pd.DataFrame()
        if distinct.empty:
            distinct = pd.DataFrame([{
                "window_id": settings.target_window,
                "object": obj,
                "cluster_pair": "NO_PAIR",
                "distinctness_status": "INSUFFICIENT_EVIDENCE",
                "interpretation_allowed": False,
                "unavailable_reason": "Fewer than two clusters available for object.",
            } for obj in settings.target_objects])
        return audit, distinct, "OK"
    except Exception as exc:
        audit, distinct = _profile_features_unavailable(f"Profile component audit failed: {type(exc).__name__}: {exc}", settings)
        return audit, distinct, "FAILED_PROFILE_AUDIT"


def _order_sensitivity_decomposition(sel_clustered: pd.DataFrame, settings: W045JwHPeakSensitivityCauseAuditSettings) -> pd.DataFrame:
    h = sel_clustered[sel_clustered["object"].astype(str) == "H"].copy()
    jw = sel_clustered[sel_clustered["object"].astype(str) == "Jw"].copy()
    keys = ["config_id", "smoothing", "detector_width", "band_half_width", "search_mode", "selection_rule"]
    h_cols = keys + ["selected_peak_day", "cluster_id"]
    jw_cols = keys + ["selected_peak_day", "cluster_id"]
    if h.empty or jw.empty:
        return pd.DataFrame()
    merged = jw[jw_cols].merge(h[h_cols], on=keys, suffixes=("_jw", "_h"), how="inner")
    rows = []
    jw_std = float(pd.to_numeric(merged["selected_peak_day_jw"], errors="coerce").std())
    h_std = float(pd.to_numeric(merged["selected_peak_day_h"], errors="coerce").std())
    if jw_std > h_std * 1.25:
        source = "Jw_DOMINATED"
    elif h_std > jw_std * 1.25:
        source = "H_DOMINATED"
    else:
        source = "BOTH_OR_NEAR_BALANCED"
    for _, r in merged.iterrows():
        jw_d = _as_float(r["selected_peak_day_jw"])
        h_d = _as_float(r["selected_peak_day_h"])
        lag = jw_d - h_d
        if not np.isfinite(lag):
            lag_class = "missing"
        elif abs(lag) <= settings.near_tie_days:
            lag_class = "near_tie"
        elif lag < 0:
            lag_class = "Jw_before_H"
        else:
            lag_class = "Jw_after_H"
        rows.append({
            "window_id": settings.target_window,
            "config_id": r["config_id"],
            "smoothing": r["smoothing"],
            "detector_width": r["detector_width"],
            "band_half_width": r["band_half_width"],
            "search_mode": r["search_mode"],
            "selection_rule": r["selection_rule"],
            "jw_selected_day": jw_d,
            "h_selected_day": h_d,
            "jw_cluster_id": r["cluster_id_jw"],
            "h_cluster_id": r["cluster_id_h"],
            "jw_minus_h_lag": lag,
            "lag_class": lag_class,
            "jw_day_std_all_configs": jw_std,
            "h_day_std_all_configs": h_std,
            "dominant_source_of_lag_variation": source,
        })
    return pd.DataFrame(rows)


def _distribution_type(cluster_summary: pd.DataFrame, obj: str) -> str:
    sub = cluster_summary[cluster_summary["object"].astype(str) == obj]
    if sub.empty:
        return "NO_CLEAR_CLUSTER"
    major = sub[sub["config_fraction"] >= 0.15]
    if len(major) >= 2:
        return "TWO_OR_MORE_STABLE_CLUSTERS"
    if len(sub) == 1:
        width = _as_float(sub.iloc[0].get("day_max")) - _as_float(sub.iloc[0].get("day_min"))
        return "SINGLE_BROAD_PLATEAU" if width > 7 else "SHARP_OR_SINGLE_CLUSTER"
    if (sub["cluster_status"].astype(str) == "RULE_OR_SEARCH_LOCKED_CLUSTER").any():
        return "RULE_LOCKED_CLUSTER"
    return "NO_CLEAR_CLUSTER"


def _dominant_factor(factor_contrib: pd.DataFrame, obj: str) -> Tuple[str, float, str]:
    sub = factor_contrib[factor_contrib["object"].astype(str) == obj].copy()
    if sub.empty:
        return "UNKNOWN", np.nan, "NO_FACTOR_TABLE"
    sub = sub.sort_values("day_variance_explained_eta2", ascending=False)
    r = sub.iloc[0]
    return str(r["factor"]), _as_float(r["day_variance_explained_eta2"]), str(r.get("interpretation_hint", ""))


def _modal_landscape(score_summary: pd.DataFrame, obj: str) -> str:
    sub = score_summary[score_summary["object"].astype(str) == obj] if not score_summary.empty else pd.DataFrame()
    if sub.empty or "landscape_type" not in sub.columns:
        return "UNAVAILABLE"
    return str(sub["landscape_type"].astype(str).value_counts().index[0])


def _search_status(boundary: pd.DataFrame, obj: str) -> str:
    sub = boundary[boundary["object"].astype(str) == obj] if not boundary.empty else pd.DataFrame()
    if sub.empty:
        return "UNKNOWN"
    bfrac = float(sub["boundary_risk_flag"].fillna(False).astype(bool).mean())
    outside_frac = float(sub["outside_system_window_flag"].fillna(False).astype(bool).mean())
    if bfrac >= 0.25:
        return "BOUNDARY_DRIVEN"
    if outside_frac >= 0.60:
        return "WIDE_OR_OUTSIDE_SYSTEM_MIXING"
    return "SEARCH_ROBUST_OR_NONDOMINANT"


def _distinctness_status(distinct: pd.DataFrame, obj: str) -> str:
    sub = distinct[distinct["object"].astype(str) == obj] if not distinct.empty else pd.DataFrame()
    if sub.empty:
        return "INSUFFICIENT_EVIDENCE"
    vals = sub["distinctness_status"].astype(str).tolist()
    if "PHYSICAL_DISTINCT" in vals:
        return "PHYSICAL_DISTINCT"
    if "WEAKLY_DISTINCT" in vals:
        return "WEAKLY_DISTINCT"
    if "NOT_DISTINCT" in vals:
        return "NOT_DISTINCT"
    return vals[0] if vals else "INSUFFICIENT_EVIDENCE"


def _implementation_status(impl: pd.DataFrame, obj: str) -> str:
    sub = impl[impl["object"].astype(str) == obj] if not impl.empty else pd.DataFrame()
    if sub.empty:
        return "FAIL"
    return str(sub.iloc[0].get("implementation_status", "FAIL"))


def _diagnose_primary_cause(impl_status: str, dist_type: str, dominant_factor: str, dominant_eta2: float, landscape: str, search_status: str, distinct_status: str) -> Tuple[str, str, float]:
    confidence = 0.5
    if impl_status == "FAIL":
        return "IMPLEMENTATION_INCONSISTENCY", "Fix implementation/helper consistency before physical interpretation.", 0.9
    if search_status in ("BOUNDARY_DRIVEN", "WIDE_OR_OUTSIDE_SYSTEM_MIXING"):
        return "SEARCH_WINDOW_MIXING", "Selected peaks are boundary/outside-system dominated; audit search window before subpeak interpretation.", 0.75
    if landscape == "FLAT_PLATEAU":
        return "FLAT_PEAK_PLATEAU", "Single-day peak is unstable because score landscape is broad/flat; consider peak interval.", 0.75
    if landscape == "LOW_CONTRAST_NOISY":
        return "LOW_CONTRAST_UNINTERPRETABLE", "Score contrast is weak; keep as negative/uncertain result.", 0.65
    if dist_type == "TWO_OR_MORE_STABLE_CLUSTERS" and distinct_status == "PHYSICAL_DISTINCT":
        return "PHYSICAL_MULTICANDIDATE_PEAKS", "Multiple selected-day clusters are also profile-distinct; subpeak interpretation may be considered next.", 0.7
    if dist_type == "TWO_OR_MORE_STABLE_CLUSTERS" and distinct_status in ("WEAKLY_DISTINCT", "NOT_DISTINCT", "INSUFFICIENT_EVIDENCE"):
        return "STATISTICAL_MULTICLUSTER_NOT_YET_PHYSICAL", "Multiple clusters exist but physical distinctness is not strong enough for interpretation.", 0.65
    if dominant_factor == "selection_rule" and np.isfinite(dominant_eta2) and dominant_eta2 >= 0.35:
        return "RULE_SEMANTIC_DIFFERENCE", "Selection rules drive day shifts; separate rule semantics before physical interpretation.", 0.7
    if dominant_factor == "smoothing" and np.isfinite(dominant_eta2) and dominant_eta2 >= 0.35:
        return "SMOOTH_SCALE_DEPENDENCE", "Smooth5/smooth9 scale choice drives peak timing; audit time-scale dependence.", 0.7
    if dominant_factor in ("detector_width", "band_half_width") and np.isfinite(dominant_eta2) and dominant_eta2 >= 0.35:
        return "DETECTOR_SCALE_DEPENDENCE", "Detector/band scale drives peak timing; audit temporal-scale matching.", 0.65
    return "MIXED_OR_INCONCLUSIVE", "No single cause dominates; keep as audit-level uncertainty.", confidence


def _sensitivity_cause_diagnosis(
    impl: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    factor_contrib: pd.DataFrame,
    score_summary: pd.DataFrame,
    boundary: pd.DataFrame,
    profile_audit: pd.DataFrame,
    distinct: pd.DataFrame,
    order_decomp: pd.DataFrame,
    settings: W045JwHPeakSensitivityCauseAuditSettings,
) -> pd.DataFrame:
    rows = []
    for obj in settings.target_objects:
        impl_status = _implementation_status(impl, obj)
        dist_type = _distribution_type(cluster_summary, obj)
        dom_factor, dom_eta2, dom_hint = _dominant_factor(factor_contrib, obj)
        landscape = _modal_landscape(score_summary, obj)
        sstatus = _search_status(boundary, obj)
        dstatus = _distinctness_status(distinct, obj)
        component_status = "UNAVAILABLE"
        if not profile_audit.empty:
            sub = profile_audit[profile_audit["object"].astype(str) == obj]
            if not sub.empty and "component_mixture_status" in sub.columns:
                component_status = str(sub["component_mixture_status"].astype(str).value_counts().index[0])
        cause, next_step, conf = _diagnose_primary_cause(impl_status, dist_type, dom_factor, dom_eta2, landscape, sstatus, dstatus)
        rows.append({
            "window_id": settings.target_window,
            "object_or_pair": obj,
            "record_type": "object",
            "implementation_status": impl_status,
            "day_distribution_type": dist_type,
            "dominant_config_factor": dom_factor,
            "dominant_config_factor_eta2": dom_eta2,
            "dominant_config_hint": dom_hint,
            "landscape_type": landscape,
            "search_window_status": sstatus,
            "component_mixture_status": component_status,
            "physical_distinctness_status": dstatus,
            "order_sensitivity_source": "NA_OBJECT_LEVEL",
            "primary_sensitivity_cause": cause,
            "recommended_next_step": next_step,
            "confidence": conf,
        })
    if not order_decomp.empty:
        order_source = str(order_decomp["dominant_source_of_lag_variation"].dropna().iloc[0]) if order_decomp["dominant_source_of_lag_variation"].notna().any() else "UNKNOWN"
        lag_counts = order_decomp["lag_class"].astype(str).value_counts(normalize=True).to_dict()
        if lag_counts.get("near_tie", 0.0) >= 0.50:
            primary = "NEAR_TIE_DOMINATED_ORDER_SENSITIVITY"
            next_step = "Do not write Jw-before-H or H-before-Jw; treat as near-tie/overlap unless typed subpeaks are later justified."
            conf = 0.75
        elif len([k for k, v in lag_counts.items() if v > 0.15]) >= 2:
            primary = "ORDER_CLASS_SWITCHING"
            next_step = "Order changes across configs; inspect whether switching comes from Jw, H, or both before interpretation."
            conf = 0.65
        else:
            primary = "ORDER_RELATIVELY_STABLE"
            next_step = "Pairwise order is more stable than object peak days; compare with object-level cause audit."
            conf = 0.6
        rows.append({
            "window_id": settings.target_window,
            "object_or_pair": "Jw_minus_H",
            "record_type": "pair",
            "implementation_status": ";".join([_implementation_status(impl, obj) for obj in settings.target_objects]),
            "day_distribution_type": "PAIR_LAG_DISTRIBUTION",
            "dominant_config_factor": "NA_PAIR_LEVEL",
            "dominant_config_factor_eta2": np.nan,
            "dominant_config_hint": json.dumps(lag_counts, ensure_ascii=False),
            "landscape_type": "NA_PAIR_LEVEL",
            "search_window_status": "NA_PAIR_LEVEL",
            "component_mixture_status": "NA_PAIR_LEVEL",
            "physical_distinctness_status": "NA_PAIR_LEVEL",
            "order_sensitivity_source": order_source,
            "primary_sensitivity_cause": primary,
            "recommended_next_step": next_step,
            "confidence": conf,
        })
    return pd.DataFrame(rows)


def _make_figures(paths: Dict[str, Path], sel: pd.DataFrame, cluster_summary: pd.DataFrame, factor_contrib: pd.DataFrame, score_by_day: pd.DataFrame, profile_audit: pd.DataFrame, settings: W045JwHPeakSensitivityCauseAuditSettings) -> str:
    if not settings.make_figures:
        return "SKIPPED_BY_SETTINGS"
    try:
        import matplotlib.pyplot as plt
        fig_dir = _ensure_dir(paths["fig_out"])
        # Fig 1: selected day by config index.
        fig, ax = plt.subplots(figsize=(10, 5))
        for obj, sub in sel.groupby("object", sort=True):
            y = np.arange(len(sub))
            ax.scatter(sub["selected_peak_day"], y, label=obj, s=18, alpha=0.8)
        ax.axvline(45, linestyle="--", linewidth=1)
        ax.set_title("W045 Jw/H selected peak day by sensitivity config")
        ax.set_xlabel("Selected peak day (day 0 = Apr 1)")
        ax.set_ylabel("Config rows within object")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "fig1_selected_day_by_config_W045_Jw_H.png", dpi=180)
        plt.close(fig)
        # Fig 2: histograms.
        fig, ax = plt.subplots(figsize=(10, 5))
        for obj, sub in sel.groupby("object", sort=True):
            ax.hist(pd.to_numeric(sub["selected_peak_day"], errors="coerce").dropna(), bins=20, alpha=0.5, label=obj)
        ax.axvline(45, linestyle="--", linewidth=1)
        ax.set_title("W045 Jw/H selected day distribution")
        ax.set_xlabel("Selected peak day")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "fig2_selected_day_histogram_W045_Jw_H.png", dpi=180)
        plt.close(fig)
        # Fig 3: factor eta2.
        if not factor_contrib.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot = factor_contrib.pivot(index="factor", columns="object", values="day_variance_explained_eta2")
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("W045 Jw/H config-factor contribution to selected-day variance")
            ax.set_ylabel("eta squared")
            fig.tight_layout()
            fig.savefig(fig_dir / "fig3_factor_contribution_W045_Jw_H.png", dpi=180)
            plt.close(fig)
        # Fig 4: representative score landscape, if available.
        if not score_by_day.empty and not (score_by_day.get("config_id", pd.Series()).astype(str) == "UNAVAILABLE").all():
            fig, ax = plt.subplots(figsize=(10, 5))
            for obj, subobj in score_by_day.groupby("object", sort=True):
                # Plot one baseline-width representative per object.
                sub = subobj[(subobj["smoothing"] == "smooth9") & (subobj["detector_width"] == 20) & (subobj["search_mode"] == "baseline_search") & (subobj["selection_rule"] == "baseline_rule")]
                if sub.empty:
                    sub = subobj.head(200)
                ax.plot(sub["day"], sub["detector_score"], label=obj)
            ax.axvline(45, linestyle="--", linewidth=1)
            ax.set_title("Representative W045 Jw/H score landscape")
            ax.set_xlabel("Day")
            ax.set_ylabel("Detector score")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "fig4_score_landscape_W045_Jw_H.png", dpi=180)
            plt.close(fig)
        # Fig 5: profile feature summary if available.
        if not profile_audit.empty and not (profile_audit.get("cluster_id", pd.Series()).astype(str) == "UNAVAILABLE").all():
            fig, ax = plt.subplots(figsize=(10, 5))
            sub = profile_audit[profile_audit["smoothing"].astype(str) == "smooth9"].copy()
            for obj, sobj in sub.groupby("object", sort=True):
                ax.scatter(sobj["day_median"], sobj["centroid_lat"], label=obj, s=50)
            ax.set_title("W045 Jw/H cluster centroid-lat feature audit")
            ax.set_xlabel("Cluster median day")
            ax.set_ylabel("Abs-weighted centroid latitude")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "fig5_cluster_profile_feature_W045_Jw_H.png", dpi=180)
            plt.close(fig)
        return "OK"
    except Exception as exc:
        return f"FAILED_FIGURES: {type(exc).__name__}: {exc}"


def _summary_markdown(
    run_meta: dict,
    impl: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    factor_contrib: pd.DataFrame,
    score_summary: pd.DataFrame,
    distinct: pd.DataFrame,
    order_decomp: pd.DataFrame,
    diagnosis: pd.DataFrame,
) -> str:
    lines = []
    lines.append(f"# {VERSION} summary")
    lines.append("")
    lines.append("## 0. Interpretation boundary")
    lines.append("This output is a cause audit for W045 H/Jw peak-selection sensitivity. It does not assign physical sub-peak interpretations and does not replace V9 peak_all_windows_v9_a.")
    lines.append("")
    lines.append("## 1. Run status")
    for k in ["started_at", "finished_at", "score_landscape_status", "profile_component_status", "figure_status"]:
        lines.append(f"- {k}: {run_meta.get(k)}")
    lines.append("")
    lines.append("## 2. Implementation consistency")
    if impl.empty:
        lines.append("No implementation audit rows were produced.")
    else:
        for _, r in impl.iterrows():
            lines.append(f"- {r['object']}: {r['implementation_status']} — {r['risk_note']}")
    lines.append("")
    lines.append("## 3. Selected-day clusters")
    if cluster_summary.empty:
        lines.append("No cluster summary rows were produced.")
    else:
        for _, r in cluster_summary.iterrows():
            lines.append(f"- {r['object']} {r['cluster_id']}: day {r['day_min']}-{r['day_max']}, n={r['n_configs']}, frac={r['config_fraction']:.2f}, status={r['cluster_status']}")
    lines.append("")
    lines.append("## 4. Dominant configuration factors")
    if factor_contrib.empty:
        lines.append("No factor contribution table was produced.")
    else:
        top = factor_contrib.sort_values(["object", "day_variance_explained_eta2"], ascending=[True, False]).groupby("object").head(1)
        for _, r in top.iterrows():
            lines.append(f"- {r['object']}: {r['factor']} eta2={r['day_variance_explained_eta2']:.3f}, hint={r['interpretation_hint']}")
    lines.append("")
    lines.append("## 5. Landscape / profile evidence")
    if not score_summary.empty and "landscape_type" in score_summary.columns:
        for obj, sub in score_summary.groupby("object", sort=True):
            mode = sub["landscape_type"].astype(str).value_counts().index[0]
            lines.append(f"- {obj} score landscape modal type: {mode}")
    if not distinct.empty and "distinctness_status" in distinct.columns:
        for obj, sub in distinct.groupby("object", sort=True):
            vals = ", ".join(sorted(set(sub["distinctness_status"].astype(str).tolist())))
            lines.append(f"- {obj} cluster distinctness statuses: {vals}")
    lines.append("")
    lines.append("## 6. Jw-H order sensitivity")
    if order_decomp.empty:
        lines.append("No paired Jw-H order decomposition was produced.")
    else:
        counts = order_decomp["lag_class"].astype(str).value_counts().to_dict()
        source = order_decomp["dominant_source_of_lag_variation"].dropna().iloc[0] if order_decomp["dominant_source_of_lag_variation"].notna().any() else "UNKNOWN"
        lines.append(f"- lag class counts: {counts}")
        lines.append(f"- dominant source of lag variation: {source}")
    lines.append("")
    lines.append("## 7. Cause diagnosis")
    if diagnosis.empty:
        lines.append("No diagnosis rows were produced.")
    else:
        for _, r in diagnosis.iterrows():
            lines.append(f"- {r['object_or_pair']}: {r['primary_sensitivity_cause']} | next: {r['recommended_next_step']} | confidence={r['confidence']}")
    lines.append("")
    lines.append("## 8. Output files")
    for rel in run_meta.get("output_files", []):
        lines.append(f"- {rel}")
    lines.append("")
    return "\n".join(lines)


def run_w045_jw_h_peak_sensitivity_cause_audit_v1(v9_root: Path | str, settings: Optional[W045JwHPeakSensitivityCauseAuditSettings] = None) -> dict:
    settings = settings or W045JwHPeakSensitivityCauseAuditSettings()
    paths = _resolve_paths(Path(v9_root))
    _ensure_dir(paths["cross_out"])
    _ensure_dir(paths["fig_out"])
    _ensure_dir(paths["log_out"])
    started = datetime.now().isoformat(timespec="seconds")
    _log(f"Start {settings.version}")
    _log(f"V9 root: {paths['v9_root']}")
    _require_inputs(paths)
    tables = _load_input_tables(paths)

    _log("[1/9] Implementation consistency audit")
    impl = _implementation_consistency_audit(tables, settings)
    _safe_to_csv(impl, paths["cross_out"] / "implementation_consistency_audit_W045_Jw_H.csv")

    _log("[2/9] Selected-day clusters")
    sel = _filter_selection(tables["object_peak_selection_by_config"], settings)
    sel_clustered = _add_day_clusters(sel, settings)
    _safe_to_csv(sel_clustered, paths["cross_out"] / "selected_day_by_config_W045_Jw_H.csv")
    clusters = _cluster_summary(sel_clustered, settings)
    _safe_to_csv(clusters, paths["cross_out"] / "selected_day_cluster_summary_W045_Jw_H.csv")

    _log("[3/9] Configuration-factor contribution")
    factor_contrib, factor_cross = _factor_contribution(sel_clustered, settings)
    _safe_to_csv(factor_contrib, paths["cross_out"] / "factor_contribution_W045_Jw_H.csv")
    _safe_to_csv(factor_cross, paths["cross_out"] / "factor_cluster_crosstab_W045_Jw_H.csv")

    _log("[4/9] Search-window boundary audit")
    boundary = _search_boundary_audit(sel_clustered, tables, settings)
    _safe_to_csv(boundary, paths["cross_out"] / "search_window_boundary_audit_W045_Jw_H.csv")

    _log("[5/9] Score-landscape audit")
    score_by_day, score_summary, score_status = _compute_score_landscape(paths, tables, settings)
    _safe_to_csv(score_by_day, paths["cross_out"] / "score_landscape_by_config_W045_Jw_H.csv")
    _safe_to_csv(score_summary, paths["cross_out"] / "score_landscape_summary_W045_Jw_H.csv")

    _log("[6/9] Profile-component and physical-distinctness audit")
    profile_audit, distinct, profile_status = _profile_component_and_distinctness(paths, clusters, settings)
    _safe_to_csv(profile_audit, paths["cross_out"] / "profile_component_audit_W045_Jw_H.csv")
    _safe_to_csv(distinct, paths["cross_out"] / "cluster_physical_distinctness_audit_W045_Jw_H.csv")

    _log("[7/9] Jw-H order sensitivity decomposition")
    order = _order_sensitivity_decomposition(sel_clustered, settings)
    _safe_to_csv(order, paths["cross_out"] / "jw_h_order_sensitivity_decomposition_W045.csv")

    _log("[8/9] Final cause diagnosis")
    diagnosis = _sensitivity_cause_diagnosis(impl, clusters, factor_contrib, score_summary, boundary, profile_audit, distinct, order, settings)
    _safe_to_csv(diagnosis, paths["cross_out"] / "sensitivity_cause_diagnosis_W045_Jw_H.csv")

    _log("[9/9] Figures and summary")
    figure_status = _make_figures(paths, sel_clustered, clusters, factor_contrib, score_by_day, profile_audit, settings)

    output_files = [
        "cross_window/implementation_consistency_audit_W045_Jw_H.csv",
        "cross_window/selected_day_by_config_W045_Jw_H.csv",
        "cross_window/selected_day_cluster_summary_W045_Jw_H.csv",
        "cross_window/factor_contribution_W045_Jw_H.csv",
        "cross_window/factor_cluster_crosstab_W045_Jw_H.csv",
        "cross_window/search_window_boundary_audit_W045_Jw_H.csv",
        "cross_window/score_landscape_by_config_W045_Jw_H.csv",
        "cross_window/score_landscape_summary_W045_Jw_H.csv",
        "cross_window/profile_component_audit_W045_Jw_H.csv",
        "cross_window/cluster_physical_distinctness_audit_W045_Jw_H.csv",
        "cross_window/jw_h_order_sensitivity_decomposition_W045.csv",
        "cross_window/sensitivity_cause_diagnosis_W045_Jw_H.csv",
        "cross_window/W045_JW_H_PEAK_SENSITIVITY_CAUSE_AUDIT_V1_SUMMARY.md",
        "run_meta.json",
        "summary.json",
    ]
    if settings.make_figures:
        output_files.extend([
            "figures/fig1_selected_day_by_config_W045_Jw_H.png",
            "figures/fig2_selected_day_histogram_W045_Jw_H.png",
            "figures/fig3_factor_contribution_W045_Jw_H.png",
            "figures/fig4_score_landscape_W045_Jw_H.png",
            "figures/fig5_cluster_profile_feature_W045_Jw_H.png",
        ])
    finished = datetime.now().isoformat(timespec="seconds")
    run_meta = {
        "version": settings.version,
        "output_tag": settings.output_tag,
        "started_at": started,
        "finished_at": finished,
        "v9_root": str(paths["v9_root"]),
        "target_window": settings.target_window,
        "target_objects": list(settings.target_objects),
        "input_sensitivity_cross": str(paths["sensitivity_cross"]),
        "input_baseline_cross": str(paths["baseline_cross"]),
        "smooth9_field_path": str(paths["smooth9"]),
        "smooth5_field_path": str(paths["smooth5"]),
        "score_landscape_status": score_status,
        "profile_component_status": profile_status,
        "figure_status": figure_status,
        "settings": asdict(settings),
        "interpretation_boundary": {
            "physical_interpretation_included": settings.physical_interpretation_included,
            "rerun_changepoint_detection": settings.rerun_changepoint_detection,
            "note": "This is a cause audit, not a physical sub-peak classification and not a replacement for V9 baseline.",
        },
        "output_files": output_files,
    }
    _write_json(run_meta, paths["out_dir"] / "run_meta.json")
    summary = {
        "version": settings.version,
        "target_window": settings.target_window,
        "target_objects": list(settings.target_objects),
        "n_selected_rows": int(len(sel_clustered)),
        "n_clusters": int(len(clusters)),
        "score_landscape_status": score_status,
        "profile_component_status": profile_status,
        "figure_status": figure_status,
        "diagnosis_records": diagnosis.to_dict(orient="records"),
    }
    _write_json(summary, paths["out_dir"] / "summary.json")
    md = _summary_markdown(run_meta, impl, clusters, factor_contrib, score_summary, distinct, order, diagnosis)
    _write_text(md, paths["cross_out"] / "W045_JW_H_PEAK_SENSITIVITY_CAUSE_AUDIT_V1_SUMMARY.md")
    _write_text(f"completed_at={finished}\noutput_dir={paths['out_dir']}\n", paths["log_out"] / "last_run.txt")
    _log(f"Completed {settings.version}")
    _log(f"Output: {paths['out_dir']}")
    return run_meta


__all__ = [
    "W045JwHPeakSensitivityCauseAuditSettings",
    "run_w045_jw_h_peak_sensitivity_cause_audit_v1",
]
