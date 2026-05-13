# -*- coding: utf-8 -*-
"""
W045 Jw/H peak sensitivity cause audit v1_1.

Purpose
-------
This patch is a narrow correction layer on top of
`peak_sensitivity_cause_audit_w045_jw_h_v1`.

It does NOT perform physical sub-peak classification. It audits whether the v1
apparent sensitivity/order conclusions are inflated by inadmissible configs,
rule/search locked candidates, outside-system candidates, and whether the Jw
near-window clusters show a minimal core/axis latitude difference.

Main outputs
------------
V9/outputs/peak_sensitivity_cause_audit_w045_jw_h_v1_1/cross_window/
  - config_admissibility_audit_W045_Jw_H.csv
  - jw_h_order_filtered_sensitivity_W045.csv
  - jw_axis_core_shift_audit_W045.csv
  - h_candidate_admissibility_audit_W045.csv
  - cluster_distinctness_revision_W045_Jw_H.csv
  - sensitivity_cause_diagnosis_W045_Jw_H_v1_1.csv
  - W045_JW_H_PEAK_SENSITIVITY_CAUSE_AUDIT_V1_1_SUMMARY.md
  - run_meta.json / summary.json

Notes
-----
`core_lat` is computed as a first-version proxy. If v1 saved full profile vectors,
a weighted core latitude could be computed; v1 currently saves cluster-level
`max_lat`, so v1_1 uses `max_lat` as `core_lat_proxy` and states this explicitly.
"""

from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

WINDOW_ID = "W045"
OBJECTS = ("H", "Jw")
PAIR_TARGET = "Jw_minus_H"
V1_OUTPUT_NAME = "peak_sensitivity_cause_audit_w045_jw_h_v1"
V1_1_OUTPUT_NAME = "peak_sensitivity_cause_audit_w045_jw_h_v1_1"
SYSTEM_WINDOW_FALLBACK = (42.0, 48.0)  # used only when v1 flags are absent
NEAR_TIE_DAYS = 2.0
MIN_PAIR_CONFIGS_FOR_INTERPRETATION = 6


@dataclass
class AuditPaths:
    v9_root: Path
    v1_root: Path
    v1_cross: Path
    out_root: Path
    out_cross: Path
    out_fig: Path
    out_log: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs(paths: AuditPaths) -> None:
    for p in (paths.out_root, paths.out_cross, paths.out_fig, paths.out_log):
        p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input is missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_num(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y")


def _mode(series: pd.Series, default: str = "NA"):
    s = series.dropna()
    if s.empty:
        return default
    return s.value_counts().index[0]


def _frac(series: pd.Series, pred) -> float:
    if len(series) == 0:
        return np.nan
    return float(np.mean([pred(v) for v in series]))


def _dominant_fraction(series: pd.Series) -> Tuple[str, float]:
    s = series.dropna()
    if s.empty:
        return "NA", np.nan
    vc = s.value_counts()
    return str(vc.index[0]), float(vc.iloc[0] / vc.sum())


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_paths(v9_root: Path) -> AuditPaths:
    v9_root = Path(v9_root).resolve()
    v1_root = v9_root / "outputs" / V1_OUTPUT_NAME
    return AuditPaths(
        v9_root=v9_root,
        v1_root=v1_root,
        v1_cross=v1_root / "cross_window",
        out_root=v9_root / "outputs" / V1_1_OUTPUT_NAME,
        out_cross=v9_root / "outputs" / V1_1_OUTPUT_NAME / "cross_window",
        out_fig=v9_root / "outputs" / V1_1_OUTPUT_NAME / "figures",
        out_log=v9_root / "logs" / V1_1_OUTPUT_NAME,
    )


def _infer_inside_system(row: pd.Series) -> bool:
    if "outside_system_window_flag" in row and not pd.isna(row.get("outside_system_window_flag")):
        return not _safe_bool(row.get("outside_system_window_flag"))
    if "relation_to_system_window" in row:
        return str(row.get("relation_to_system_window")) == "within_system_window"
    d = _safe_num(row.get("selected_peak_day", row.get("selected_day", np.nan)))
    return SYSTEM_WINDOW_FALLBACK[0] <= d <= SYSTEM_WINDOW_FALLBACK[1]


def build_config_admissibility(
    selected: pd.DataFrame,
    boundary: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Mark whether each object/config selected-day result is admissible for W045 interpretation."""
    sel = selected.copy()
    if "selected_peak_day" not in sel.columns and "selected_day" in sel.columns:
        sel["selected_peak_day"] = sel["selected_day"]

    # Join boundary flags from v1 if available.
    bcols = [
        c for c in [
            "window_id", "object", "config_id", "selected_day", "distance_to_anchor",
            "distance_to_left_boundary", "distance_to_right_boundary", "boundary_risk_flag",
            "outside_core_window_flag", "outside_system_window_flag", "search_start", "search_end",
        ] if c in boundary.columns
    ]
    if boundary.empty or not {"object", "config_id"}.issubset(boundary.columns):
        merged = sel.copy()
    else:
        merged = sel.merge(boundary[bcols], on=["window_id", "object", "config_id"], how="left", suffixes=("", "_boundary"))

    # Join cluster summary to quantify rule/search lock.
    ccols = [
        c for c in [
            "window_id", "object", "cluster_id", "config_fraction",
            "dominant_selection_rule", "dominant_selection_rule_fraction",
            "dominant_search_mode", "dominant_search_mode_fraction", "cluster_status",
            "n_configs", "narrow_fraction", "baseline_search_fraction", "wide_fraction",
        ] if c in clusters.columns
    ]
    if not clusters.empty and {"object", "cluster_id"}.issubset(clusters.columns):
        merged = merged.merge(clusters[ccols], on=["window_id", "object", "cluster_id"], how="left")

    out_rows = []
    for _, r in merged.iterrows():
        selected_day = _safe_num(r.get("selected_peak_day", r.get("selected_day", np.nan)))
        selection_rule = str(r.get("selection_rule", "NA"))
        search_mode = str(r.get("search_mode", "NA"))
        cluster_status = str(r.get("cluster_status", ""))
        dom_rule_frac = _safe_num(r.get("dominant_selection_rule_fraction"), 0.0)
        dom_search_frac = _safe_num(r.get("dominant_search_mode_fraction"), 0.0)
        rule_locked = (dom_rule_frac >= 0.80) or ("RULE" in cluster_status and "LOCKED" in cluster_status)
        search_locked = (dom_search_frac >= 0.80) or ("SEARCH" in cluster_status and "LOCKED" in cluster_status)
        max_score_flag = selection_rule == "max_score"
        wide_search_flag = search_mode == "wide_search"
        narrow_search_flag = search_mode == "narrow_search"
        baseline_search_flag = search_mode == "baseline_search"
        inside_system = _infer_inside_system(r)
        outside_system = not inside_system
        boundary_risk = _safe_bool(r.get("boundary_risk_flag"))
        # Distance to a fallback system window if not directly available.
        if inside_system:
            distance_to_system = 0.0
        else:
            lo, hi = SYSTEM_WINDOW_FALLBACK
            if math.isnan(selected_day):
                distance_to_system = np.nan
            elif selected_day < lo:
                distance_to_system = lo - selected_day
            else:
                distance_to_system = selected_day - hi

        score = 1.0
        reasons = []
        if outside_system:
            score -= 0.35
            reasons.append("outside_system")
        if boundary_risk:
            score -= 0.20
            reasons.append("boundary_risk")
        if max_score_flag:
            score -= 0.20
            reasons.append("max_score")
        if rule_locked:
            score -= 0.15
            reasons.append("rule_locked_cluster")
        if search_locked:
            score -= 0.10
            reasons.append("search_locked_cluster")
        if outside_system and max_score_flag:
            score -= 0.25
            reasons.append("max_score_outer")
        score = max(0.0, score)

        if outside_system and max_score_flag:
            cls = "EXCLUDE_FROM_ORDER"
        elif outside_system and (wide_search_flag or baseline_search_flag or search_locked):
            cls = "SEARCH_MIXING_RISK"
        elif outside_system:
            cls = "SEARCH_MIXING_RISK"
        elif rule_locked and max_score_flag:
            cls = "RULE_SEMANTIC_ONLY"
        elif inside_system and (not boundary_risk) and (not max_score_flag) and (not rule_locked):
            cls = "CORE_ADMISSIBLE"
        elif inside_system and (boundary_risk or max_score_flag or rule_locked):
            cls = "WEAK_ADMISSIBLE"
        else:
            cls = "WEAK_ADMISSIBLE"

        out_rows.append({
            "window_id": r.get("window_id", WINDOW_ID),
            "object": r.get("object"),
            "config_id": r.get("config_id"),
            "smoothing": r.get("smoothing"),
            "detector_width": r.get("detector_width"),
            "band_half_width": r.get("band_half_width"),
            "search_mode": search_mode,
            "selection_rule": selection_rule,
            "selected_day": selected_day,
            "cluster_id": r.get("cluster_id"),
            "system_window_start": SYSTEM_WINDOW_FALLBACK[0],
            "system_window_end": SYSTEM_WINDOW_FALLBACK[1],
            "inside_system_window": inside_system,
            "distance_to_anchor": _safe_num(r.get("distance_to_anchor"), np.nan),
            "distance_to_system_window": distance_to_system,
            "outside_system_window_flag": outside_system,
            "boundary_risk_flag": boundary_risk,
            "rule_locked_flag": bool(rule_locked),
            "search_locked_flag": bool(search_locked),
            "max_score_flag": bool(max_score_flag),
            "wide_search_flag": bool(wide_search_flag),
            "narrow_search_flag": bool(narrow_search_flag),
            "baseline_search_flag": bool(baseline_search_flag),
            "cluster_config_fraction": _safe_num(r.get("config_fraction"), np.nan),
            "cluster_rule_purity": dom_rule_frac,
            "cluster_search_purity": dom_search_frac,
            "admissibility_score": score,
            "admissibility_class": cls,
            "exclude_reason": ";".join(reasons) if reasons else "NONE",
        })
    return pd.DataFrame(out_rows)


def _summarize_order(df: pd.DataFrame, filter_name: str, excluded_count: int, reason: str) -> dict:
    n = len(df)
    if n == 0:
        return {
            "filter_name": filter_name,
            "n_config_pairs": 0,
            "n_jw_after_h": 0,
            "n_jw_before_h": 0,
            "n_near_tie": 0,
            "frac_jw_after_h": np.nan,
            "frac_jw_before_h": np.nan,
            "frac_near_tie": np.nan,
            "median_lag": np.nan,
            "iqr_lag": np.nan,
            "lag_min": np.nan,
            "lag_max": np.nan,
            "dominant_order_class": "NONE",
            "order_stability_status": "INSUFFICIENT_CORE_CONFIGS",
            "excluded_config_count": excluded_count,
            "main_exclusion_reason": reason,
            "interpretation_allowed": False,
        }
    lag = df["jw_minus_h_lag"].astype(float)
    near = lag.abs() <= NEAR_TIE_DAYS
    after = lag > NEAR_TIE_DAYS
    before = lag < -NEAR_TIE_DAYS
    counts = {
        "Jw_after_H": int(after.sum()),
        "Jw_before_H": int(before.sum()),
        "near_tie": int(near.sum()),
    }
    fracs = {k: v / n for k, v in counts.items()}
    dominant = max(fracs, key=fracs.get)
    q75, q25 = np.nanpercentile(lag, [75, 25]) if n else (np.nan, np.nan)
    iqr = float(q75 - q25) if n else np.nan
    med = float(np.nanmedian(lag))

    if n < MIN_PAIR_CONFIGS_FOR_INTERPRETATION:
        status = "INSUFFICIENT_CORE_CONFIGS"
        allowed = False
    elif fracs["Jw_after_H"] >= 0.75 and fracs["near_tie"] < 0.40:
        status = "ORDER_STABLE_JW_AFTER_H"
        allowed = True
    elif fracs["Jw_before_H"] >= 0.75 and fracs["near_tie"] < 0.40:
        status = "ORDER_STABLE_JW_BEFORE_H"
        allowed = True
    elif fracs["near_tie"] >= 0.40 or (abs(med) <= NEAR_TIE_DAYS and (float(np.nanmin(lag)) <= 0 <= float(np.nanmax(lag)))):
        status = "NEAR_TIE_DOMINATED"
        allowed = False
    else:
        status = "MIXED_ORDER"
        allowed = False

    return {
        "filter_name": filter_name,
        "n_config_pairs": int(n),
        "n_jw_after_h": counts["Jw_after_H"],
        "n_jw_before_h": counts["Jw_before_H"],
        "n_near_tie": counts["near_tie"],
        "frac_jw_after_h": fracs["Jw_after_H"],
        "frac_jw_before_h": fracs["Jw_before_H"],
        "frac_near_tie": fracs["near_tie"],
        "median_lag": med,
        "iqr_lag": iqr,
        "lag_min": float(np.nanmin(lag)),
        "lag_max": float(np.nanmax(lag)),
        "dominant_order_class": dominant,
        "order_stability_status": status,
        "excluded_config_count": int(excluded_count),
        "main_exclusion_reason": reason,
        "interpretation_allowed": bool(allowed),
    }


def build_filtered_order(pair_df: pd.DataFrame, admiss: pd.DataFrame) -> pd.DataFrame:
    adm_jw = admiss[admiss["object"] == "Jw"].add_prefix("jw_")
    adm_h = admiss[admiss["object"] == "H"].add_prefix("h_")
    df = pair_df.copy()
    df = df.merge(adm_jw, left_on="config_id", right_on="jw_config_id", how="left")
    df = df.merge(adm_h, left_on="config_id", right_on="h_config_id", how="left")
    if "jw_minus_h_lag" not in df.columns:
        df["jw_minus_h_lag"] = df["jw_selected_day"] - df["h_selected_day"]

    masks = {}
    masks["F0_all_configs"] = pd.Series(True, index=df.index)
    masks["F1_exclude_outside_system"] = (~df["jw_outside_system_window_flag"].fillna(True)) & (~df["h_outside_system_window_flag"].fillna(True))
    masks["F2_exclude_max_score"] = df["selection_rule"].astype(str) != "max_score"
    masks["F3_exclude_max_score_or_outside"] = masks["F1_exclude_outside_system"] & masks["F2_exclude_max_score"]
    masks["F4_core_configs_only"] = df["jw_admissibility_class"].isin(["CORE_ADMISSIBLE", "WEAK_ADMISSIBLE"]) & df["h_admissibility_class"].isin(["CORE_ADMISSIBLE", "WEAK_ADMISSIBLE"])
    masks["F5_strict_core_configs"] = (
        (df["jw_admissibility_class"] == "CORE_ADMISSIBLE") &
        (df["h_admissibility_class"] == "CORE_ADMISSIBLE") &
        (df["selection_rule"].astype(str) != "max_score") &
        (~df["jw_boundary_risk_flag"].fillna(True)) &
        (~df["h_boundary_risk_flag"].fillna(True))
    )
    rows = []
    total = len(df)
    for name, mask in masks.items():
        rows.append(_summarize_order(df[mask].copy(), name, total - int(mask.sum()), name.replace("F", "filter_", 1)))

    out = pd.DataFrame(rows)
    # Flag filter sensitivity by comparing all-configs against core/strict core.
    all_status = out.loc[out["filter_name"] == "F0_all_configs", "order_stability_status"].iloc[0]
    for idx, row in out.iterrows():
        status = row["order_stability_status"]
        if row["filter_name"] != "F0_all_configs" and status != all_status:
            out.loc[idx, "filter_sensitive_vs_all"] = True
        else:
            out.loc[idx, "filter_sensitive_vs_all"] = False
    return out


def _aggregate_profile_components(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame()
    rows = []
    for (obj, cluster), g in profile.groupby(["object", "cluster_id"], dropna=False):
        rows.append({
            "object": obj,
            "cluster_id": cluster,
            "day_median": float(np.nanmedian(g["day_median"])) if "day_median" in g else np.nan,
            "amplitude_l2": float(np.nanmedian(g["amplitude_l2"])) if "amplitude_l2" in g else np.nan,
            "centroid_lat": float(np.nanmedian(g["centroid_lat"])) if "centroid_lat" in g else np.nan,
            "spread_lat": float(np.nanmedian(g["spread_lat"])) if "spread_lat" in g else np.nan,
            "max_lat": float(np.nanmedian(g["max_lat"])) if "max_lat" in g else np.nan,
            "north_south_contrast": float(np.nanmedian(g["north_south_contrast"])) if "north_south_contrast" in g else np.nan,
            "profile_shape_change_score": float(np.nanmedian(g["profile_shape_change_score"])) if "profile_shape_change_score" in g else np.nan,
            "n_smoothing_rows": int(len(g)),
            "smoothings_available": ";".join(sorted(map(str, g.get("smoothing", pd.Series(dtype=str)).dropna().unique()))),
        })
    return pd.DataFrame(rows)


def build_jw_axis_core_shift(profile: pd.DataFrame) -> pd.DataFrame:
    agg = _aggregate_profile_components(profile)
    jw = agg[agg["object"] == "Jw"].copy()
    rows = []
    clusters = list(jw["cluster_id"].dropna())
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            a = jw[jw["cluster_id"] == c1].iloc[0]
            b = jw[jw["cluster_id"] == c2].iloc[0]
            max_lat_diff = _safe_num(b["max_lat"]) - _safe_num(a["max_lat"])
            centroid_diff = _safe_num(b["centroid_lat"]) - _safe_num(a["centroid_lat"])
            spread_diff = _safe_num(b["spread_lat"]) - _safe_num(a["spread_lat"])
            nsc_diff = _safe_num(b["north_south_contrast"]) - _safe_num(a["north_south_contrast"])
            # v1 did not save full profile vector; use max_lat as first-version core proxy.
            core_lat_a = _safe_num(a["max_lat"])
            core_lat_b = _safe_num(b["max_lat"])
            core_lat_diff = core_lat_b - core_lat_a
            axis_score = max(abs(max_lat_diff), abs(core_lat_diff), abs(centroid_diff))
            if abs(max_lat_diff) >= 2.0 or abs(core_lat_diff) >= 1.5:
                status = "AXIS_SHIFT_CLEAR"
                revised = "AXIS_DISTINCT_ONLY"
                allowed = True
            elif abs(max_lat_diff) >= 1.0 or abs(core_lat_diff) >= 0.75:
                status = "AXIS_SHIFT_WEAK"
                revised = "PHYSICALLY_WEAKLY_DISTINCT"
                allowed = False
            else:
                status = "NO_AXIS_SHIFT"
                revised = "STILL_NOT_DISTINCT"
                allowed = False
            rows.append({
                "window_id": WINDOW_ID,
                "object": "Jw",
                "cluster_pair": f"{c1}__vs__{c2}",
                "cluster_a": c1,
                "cluster_b": c2,
                "day_median_a": a["day_median"],
                "day_median_b": b["day_median"],
                "max_lat_a": a["max_lat"],
                "max_lat_b": b["max_lat"],
                "max_lat_difference": max_lat_diff,
                "centroid_lat_a": a["centroid_lat"],
                "centroid_lat_b": b["centroid_lat"],
                "centroid_lat_difference": centroid_diff,
                "spread_a": a["spread_lat"],
                "spread_b": b["spread_lat"],
                "spread_difference": spread_diff,
                "north_south_contrast_a": a["north_south_contrast"],
                "north_south_contrast_b": b["north_south_contrast"],
                "north_south_contrast_difference": nsc_diff,
                "core_lat_a": core_lat_a,
                "core_lat_b": core_lat_b,
                "core_lat_difference": core_lat_diff,
                "core_lat_proxy_method": "max_lat_from_v1_profile_component_audit",
                "axis_shift_score": axis_score,
                "axis_shift_status": status,
                "physical_distinctness_revision": revised,
                "interpretation_allowed": bool(allowed),
                "threshold_role": "diagnostic_screening_not_final_physical_threshold",
                "note": "core_lat is a proxy because v1 did not save full profile vectors; do not treat as final jet-axis diagnostic.",
            })
    return pd.DataFrame(rows)


def build_h_candidate_admissibility(clusters: pd.DataFrame, admiss: pd.DataFrame, distinct: pd.DataFrame) -> pd.DataFrame:
    hcl = clusters[clusters["object"] == "H"].copy()
    hadm = admiss[admiss["object"] == "H"].copy()
    rows = []
    for _, c in hcl.iterrows():
        cid = c["cluster_id"]
        g = hadm[hadm["cluster_id"] == cid]
        if g.empty:
            continue
        inside_frac = float(g["inside_system_window"].mean())
        outside_frac = float(g["outside_system_window_flag"].mean())
        max_frac = float(g["max_score_flag"].mean())
        narrow_frac = float(g["narrow_search_flag"].mean())
        base_frac = float(g["baseline_search_flag"].mean())
        wide_frac = float(g["wide_search_flag"].mean())
        boundary_frac = float(g["boundary_risk_flag"].mean())
        dist_status = "NA"
        if not distinct.empty and "object" in distinct.columns:
            dg = distinct[(distinct["object"] == "H") & (distinct["cluster_pair"].astype(str).str.contains(str(cid), regex=False))]
            if not dg.empty:
                dist_status = ";".join(sorted(set(dg["distinctness_status"].astype(str))))
        # Legality logic.
        if inside_frac >= 0.70 and max_frac < 0.50 and narrow_frac > 0.0 and boundary_frac < 0.50:
            legality = "ADMISSIBLE_W045_CANDIDATE"
            use = "may_enter_core_order_if_pair_filter_allows"
        elif inside_frac >= 0.20 and narrow_frac > 0.0:
            legality = "TRANSITION_EDGE_CANDIDATE"
            use = "weak_admissible_only"
        elif max_frac >= 0.80 and outside_frac >= 0.80:
            legality = "NOT_ADMISSIBLE_FOR_W045_INTERPRETATION"
            use = "exclude_from_W045_subpeak_and_order_interpretation"
        elif outside_frac >= 0.70 and (base_frac + wide_frac) >= 0.70 and narrow_frac == 0:
            legality = "SEARCH_MIXING_LIKELY"
            use = "exclude_from_core_interpretation"
        elif outside_frac >= 0.70:
            legality = "OUTER_PROCESS_CANDIDATE"
            use = "keep_as_outer_process_audit_only"
        elif max_frac >= 0.80:
            legality = "RULE_ARTIFACT_LIKELY"
            use = "rule_semantic_audit_only"
        else:
            legality = "TRANSITION_EDGE_CANDIDATE"
            use = "weak_admissible_only"
        rows.append({
            "window_id": WINDOW_ID,
            "object": "H",
            "cluster_id": cid,
            "day_min": c.get("day_min"),
            "day_max": c.get("day_max"),
            "day_median": c.get("day_median"),
            "n_configs": c.get("n_configs"),
            "config_fraction": c.get("config_fraction"),
            "inside_system_fraction": inside_frac,
            "outside_system_fraction": outside_frac,
            "max_score_fraction": max_frac,
            "narrow_search_fraction": narrow_frac,
            "baseline_search_fraction": base_frac,
            "wide_search_fraction": wide_frac,
            "boundary_risk_fraction": boundary_frac,
            "profile_distinctness_status": dist_status,
            "distance_to_anchor_median": float(np.nanmedian(g["distance_to_anchor"])),
            "distance_to_system_window_median": float(np.nanmedian(g["distance_to_system_window"])),
            "candidate_legality_status": legality,
            "recommended_use": use,
            "note": "Cluster legality is a W045 interpretation admissibility screen, not a final physical judgment.",
        })
    return pd.DataFrame(rows)


def build_cluster_distinctness_revision(
    v1_distinct: pd.DataFrame,
    profile: pd.DataFrame,
    jw_axis: pd.DataFrame,
    h_legality: pd.DataFrame,
) -> pd.DataFrame:
    prof_agg = _aggregate_profile_components(profile)
    h_leg_map = {r["cluster_id"]: r["candidate_legality_status"] for _, r in h_legality.iterrows()} if not h_legality.empty else {}
    rows = []
    for _, r in v1_distinct.iterrows():
        obj = r.get("object")
        c1 = r.get("cluster_1")
        c2 = r.get("cluster_2")
        max_lat_diff = np.nan
        core_lat_diff = np.nan
        nsc_diff = np.nan
        axis_status = "NA_NOT_JW"
        pair_legality = "ADMISSIBLE_PAIR"
        if obj == "Jw" and not jw_axis.empty:
            pair = jw_axis[jw_axis["cluster_pair"] == f"{c1}__vs__{c2}"]
            if pair.empty:
                pair = jw_axis[jw_axis["cluster_pair"] == f"{c2}__vs__{c1}"]
            if not pair.empty:
                p = pair.iloc[0]
                max_lat_diff = p["max_lat_difference"]
                core_lat_diff = p["core_lat_difference"]
                nsc_diff = p["north_south_contrast_difference"]
                axis_status = p["axis_shift_status"]
        elif obj == "H":
            statuses = [h_leg_map.get(c1, "UNKNOWN"), h_leg_map.get(c2, "UNKNOWN")]
            if any(s in ("NOT_ADMISSIBLE_FOR_W045_INTERPRETATION", "SEARCH_MIXING_LIKELY", "OUTER_PROCESS_CANDIDATE") for s in statuses):
                pair_legality = "NOT_ADMISSIBLE_PAIR"
            # Still report simple max_lat/core proxy difference for traceability.
            g1 = prof_agg[(prof_agg["object"] == obj) & (prof_agg["cluster_id"] == c1)]
            g2 = prof_agg[(prof_agg["object"] == obj) & (prof_agg["cluster_id"] == c2)]
            if not g1.empty and not g2.empty:
                max_lat_diff = _safe_num(g2.iloc[0]["max_lat"]) - _safe_num(g1.iloc[0]["max_lat"])
                core_lat_diff = max_lat_diff
                nsc_diff = _safe_num(g2.iloc[0]["north_south_contrast"]) - _safe_num(g1.iloc[0]["north_south_contrast"])
        v1_status = str(r.get("distinctness_status", "NA"))
        if pair_legality == "NOT_ADMISSIBLE_PAIR":
            revised = "NOT_ADMISSIBLE_PAIR"
            allowed = False
            reason = "at_least_one_H_cluster_is_not_admissible_for_W045_interpretation"
        elif obj == "Jw" and axis_status == "AXIS_SHIFT_CLEAR" and v1_status == "NOT_DISTINCT":
            revised = "AXIS_DISTINCT_ONLY"
            allowed = False
            reason = "v1_not_distinct_but_Jw_max_lat_core_proxy_shift_is_clear"
        elif obj == "Jw" and axis_status in ("AXIS_SHIFT_CLEAR", "AXIS_SHIFT_WEAK"):
            revised = "PHYSICALLY_WEAKLY_DISTINCT"
            allowed = False
            reason = "Jw_axis_or_core_proxy_shift_needs_follow_up"
        elif v1_status in ("PHYSICAL_DISTINCT", "PHYSICALLY_DISTINCT"):
            revised = "PHYSICALLY_DISTINCT_CANDIDATE"
            allowed = True
            reason = "v1_distinctness_already_high"
        else:
            revised = "STILL_NOT_DISTINCT"
            allowed = False
            reason = "additional_v1_1_checks_do_not_raise_distinctness_level"
        rows.append({
            "window_id": WINDOW_ID,
            "object": obj,
            "cluster_pair": r.get("cluster_pair"),
            "cluster_1": c1,
            "cluster_2": c2,
            "v1_distinctness_status": v1_status,
            "amplitude_difference": r.get("amplitude_relative_difference"),
            "centroid_difference": r.get("centroid_difference"),
            "spread_difference": r.get("spread_difference"),
            "shape_distance": r.get("shape_distance"),
            "max_lat_difference": max_lat_diff,
            "core_lat_difference": core_lat_diff,
            "north_south_contrast_difference": nsc_diff,
            "candidate_legality_pair_status": pair_legality,
            "axis_shift_status": axis_status,
            "revised_distinctness_status": revised,
            "revision_reason": reason,
            "interpretation_allowed": bool(allowed),
        })
    return pd.DataFrame(rows)


def build_final_diagnosis(
    v1_diag: pd.DataFrame,
    admiss: pd.DataFrame,
    filtered_order: pd.DataFrame,
    jw_axis: pd.DataFrame,
    h_legality: pd.DataFrame,
    distinct_rev: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    v1_map = {}
    if not v1_diag.empty and "object_or_pair" in v1_diag.columns:
        for _, r in v1_diag.iterrows():
            v1_map[str(r["object_or_pair"])] = r.to_dict()

    # H diagnosis.
    h_leg_counts = h_legality["candidate_legality_status"].value_counts().to_dict() if not h_legality.empty else {}
    h_not_adm = any(k in h_leg_counts for k in ["NOT_ADMISSIBLE_FOR_W045_INTERPRETATION", "SEARCH_MIXING_LIKELY", "OUTER_PROCESS_CANDIDATE", "RULE_ARTIFACT_LIKELY"])
    h_v1 = v1_map.get("H", {})
    if h_not_adm:
        h_primary = "SEARCH_WINDOW_MIXING_OR_OUTER_CANDIDATE"
        h_next = "Exclude non-admissible H clusters from W045 subpeak/order interpretation; audit H core-window candidate only."
        h_level = "Level_1_rule_search_sensitivity"
        h_conf = 0.80
    else:
        h_primary = "H_CANDIDATES_WEAKLY_ADMISSIBLE_BUT_NOT_PHYSICALLY_PROVEN"
        h_next = "Keep H candidates as weak candidates; do not enter full physical subpeak classification."
        h_level = "Level_2_statistical_multicluster_only"
        h_conf = 0.60
    rows.append({
        "window_id": WINDOW_ID,
        "target": "H",
        "record_type": "object",
        "v1_primary_cause": h_v1.get("primary_sensitivity_cause", "NA"),
        "implementation_status": h_v1.get("implementation_status", "NA"),
        "admissibility_summary": json.dumps(h_leg_counts, ensure_ascii=False),
        "filtered_order_summary": "NA_OBJECT_LEVEL",
        "axis_shift_summary": "NA_H_OBJECT_LEVEL",
        "h_candidate_legality_summary": json.dumps(h_leg_counts, ensure_ascii=False),
        "distinctness_revision_summary": ";".join(distinct_rev[distinct_rev["object"] == "H"]["revised_distinctness_status"].astype(str).unique()) if not distinct_rev.empty else "NA",
        "v1_1_primary_cause": h_primary,
        "v1_1_secondary_cause": "RULE_SEMANTIC_DIFFERENCE" if h_v1.get("dominant_config_factor", "") == "selection_rule" else "NA",
        "recommended_next_step": h_next,
        "confidence": h_conf,
        "interpretation_level": h_level,
    })

    # Jw diagnosis.
    jw_v1 = v1_map.get("Jw", {})
    axis_statuses = list(jw_axis["axis_shift_status"].dropna().astype(str)) if not jw_axis.empty else []
    revs = list(distinct_rev[distinct_rev["object"] == "Jw"]["revised_distinctness_status"].dropna().astype(str)) if not distinct_rev.empty else []
    if "AXIS_SHIFT_CLEAR" in axis_statuses:
        jw_primary = "AXIS_SHIFT_CANDIDATE_WITHOUT_FULL_PHYSICAL_PROOF"
        jw_next = "Do a narrow Jw-only axis/core-latitude follow-up before any full subpeak physical classification."
        jw_level = "Level_3_single_physical_dimension_candidate"
        jw_conf = 0.70
    elif "AXIS_SHIFT_WEAK" in axis_statuses:
        jw_primary = "STATISTICAL_MULTICLUSTER_WITH_WEAK_AXIS_HINT"
        jw_next = "Keep Jw clusters as statistical candidates; strengthen axis/profile evidence before interpretation."
        jw_level = "Level_2_statistical_multicluster_only"
        jw_conf = 0.60
    else:
        jw_primary = "STATISTICAL_MULTICLUSTER_NOT_YET_PHYSICAL"
        jw_next = "Do not interpret Jw clusters as physical subpeaks yet."
        jw_level = "Level_2_statistical_multicluster_only"
        jw_conf = 0.60
    rows.append({
        "window_id": WINDOW_ID,
        "target": "Jw",
        "record_type": "object",
        "v1_primary_cause": jw_v1.get("primary_sensitivity_cause", "NA"),
        "implementation_status": jw_v1.get("implementation_status", "NA"),
        "admissibility_summary": json.dumps(admiss[admiss["object"] == "Jw"]["admissibility_class"].value_counts().to_dict(), ensure_ascii=False),
        "filtered_order_summary": "NA_OBJECT_LEVEL",
        "axis_shift_summary": json.dumps({"axis_shift_statuses": axis_statuses, "revision_statuses": revs}, ensure_ascii=False),
        "h_candidate_legality_summary": "NA_JW_OBJECT_LEVEL",
        "distinctness_revision_summary": ";".join(revs) if revs else "NA",
        "v1_1_primary_cause": jw_primary,
        "v1_1_secondary_cause": "RULE_SEMANTIC_DIFFERENCE" if jw_v1.get("dominant_config_factor", "") == "selection_rule" else "NA",
        "recommended_next_step": jw_next,
        "confidence": jw_conf,
        "interpretation_level": jw_level,
    })

    # Pair diagnosis.
    pair_v1 = v1_map.get("Jw_minus_H", {})
    fo = {r["filter_name"]: r.to_dict() for _, r in filtered_order.iterrows()} if not filtered_order.empty else {}
    all_status = fo.get("F0_all_configs", {}).get("order_stability_status", "NA")
    core_status = fo.get("F4_core_configs_only", {}).get("order_stability_status", "NA")
    strict_status = fo.get("F5_strict_core_configs", {}).get("order_stability_status", "NA")
    if all_status != core_status and all_status.startswith("ORDER_STABLE"):
        pair_primary = "ORDER_INFLATED_BY_INADMISSIBLE_OR_RULE_LOCKED_CONFIGS"
        pair_next = "Do not use all-config Jw-H order as result; report filtered order as mixed/near-tie/filter-sensitive."
        pair_level = "Level_1_filter_sensitive_order"
        pair_conf = 0.80
    elif core_status in ("ORDER_STABLE_JW_AFTER_H", "ORDER_STABLE_JW_BEFORE_H") and strict_status in (core_status, "INSUFFICIENT_CORE_CONFIGS"):
        pair_primary = "ORDER_RELATIVELY_STABLE_AFTER_FILTERING"
        pair_next = "Pair order can be retained as filtered candidate evidence, but still cite object-level admissibility."
        pair_level = "Level_3_filtered_order_candidate"
        pair_conf = 0.70
    elif core_status == "NEAR_TIE_DOMINATED" or strict_status == "NEAR_TIE_DOMINATED":
        pair_primary = "NEAR_TIE_OR_MIXED_CORE_ORDER"
        pair_next = "Do not state Jw before/after H; report near-tie/mixed relation inside W045 core."
        pair_level = "Level_1_filter_sensitive_order"
        pair_conf = 0.75
    elif core_status == "MIXED_ORDER" or strict_status == "MIXED_ORDER":
        pair_primary = "NEAR_TIE_OR_MIXED_CORE_ORDER"
        pair_next = "Do not state stable order; retain as mixed core-order evidence."
        pair_level = "Level_1_filter_sensitive_order"
        pair_conf = 0.70
    else:
        pair_primary = "INSUFFICIENT_CORE_EVIDENCE"
        pair_next = "Filtered pair order does not have enough core configs for interpretation."
        pair_level = "Level_0_insufficient_or_uninterpretable"
        pair_conf = 0.60
    rows.append({
        "window_id": WINDOW_ID,
        "target": "Jw-H",
        "record_type": "pair",
        "v1_primary_cause": pair_v1.get("primary_sensitivity_cause", "NA"),
        "implementation_status": pair_v1.get("implementation_status", "NA"),
        "admissibility_summary": "see_config_admissibility_audit",
        "filtered_order_summary": json.dumps({
            "all": all_status,
            "core": core_status,
            "strict": strict_status,
            "all_frac_after": fo.get("F0_all_configs", {}).get("frac_jw_after_h"),
            "core_frac_after": fo.get("F4_core_configs_only", {}).get("frac_jw_after_h"),
            "strict_frac_after": fo.get("F5_strict_core_configs", {}).get("frac_jw_after_h"),
        }, ensure_ascii=False),
        "axis_shift_summary": "NA_PAIR_LEVEL",
        "h_candidate_legality_summary": json.dumps(h_leg_counts, ensure_ascii=False),
        "distinctness_revision_summary": "NA_PAIR_LEVEL",
        "v1_1_primary_cause": pair_primary,
        "v1_1_secondary_cause": "FILTER_SENSITIVE_ORDER" if all_status != core_status else "NA",
        "recommended_next_step": pair_next,
        "confidence": pair_conf,
        "interpretation_level": pair_level,
    })

    return pd.DataFrame(rows)


def _make_figures(paths: AuditPaths, admiss: pd.DataFrame, filtered: pd.DataFrame, jw_axis: pd.DataFrame, h_leg: pd.DataFrame) -> List[str]:
    made = []
    if plt is None:
        return made

    # Figure 1: filtered order comparison.
    if not filtered.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(filtered))
        after = filtered["frac_jw_after_h"].fillna(0).values
        before = filtered["frac_jw_before_h"].fillna(0).values
        near = filtered["frac_near_tie"].fillna(0).values
        ax.bar(x, after, label="Jw_after_H")
        ax.bar(x, before, bottom=after, label="Jw_before_H")
        ax.bar(x, near, bottom=after+before, label="near_tie")
        ax.set_xticks(x)
        ax.set_xticklabels(filtered["filter_name"], rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fraction")
        ax.set_title("W045 Jw-H order fractions by filter")
        ax.legend(loc="upper right")
        fig.tight_layout()
        p = paths.out_fig / "fig1_filtered_jw_h_order_W045.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        made.append(str(p))

    # Figure 2: admissibility strip plot.
    if not admiss.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        obj_y = {"H": 0, "Jw": 1}
        jitter_map = {"baseline_rule": -0.18, "max_score": -0.06, "closest_anchor": 0.06, "max_overlap": 0.18}
        for _, r in admiss.iterrows():
            y = obj_y.get(str(r["object"]), 0) + jitter_map.get(str(r["selection_rule"]), 0.0)
            ax.scatter(r["selected_day"], y, s=28, alpha=0.8)
            # no explicit colors: rely on default cycle
        ax.axvspan(SYSTEM_WINDOW_FALLBACK[0], SYSTEM_WINDOW_FALLBACK[1], alpha=0.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["H", "Jw"])
        ax.set_xlabel("selected day")
        ax.set_title("W045 selected day by object; shaded = fallback/core system window")
        fig.tight_layout()
        p = paths.out_fig / "fig2_config_admissibility_strip_W045_Jw_H.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        made.append(str(p))

    # Figure 3: Jw axis/core proxy.
    if not jw_axis.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = []
        vals = []
        for _, r in jw_axis.iterrows():
            labels.extend([str(r["cluster_a"]), str(r["cluster_b"])])
            vals.extend([r["core_lat_a"], r["core_lat_b"]])
        ax.scatter(range(len(vals)), vals, s=60)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("core latitude proxy (max_lat)")
        ax.set_title("Jw cluster core-latitude proxy comparison")
        fig.tight_layout()
        p = paths.out_fig / "fig3_jw_axis_core_shift_W045.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        made.append(str(p))

    # Figure 4: H legality fractions.
    if not h_leg.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(h_leg))
        inside = h_leg["inside_system_fraction"].fillna(0).values
        outside = h_leg["outside_system_fraction"].fillna(0).values
        maxs = h_leg["max_score_fraction"].fillna(0).values
        ax.bar(x, inside, label="inside_system")
        ax.bar(x, outside, bottom=inside, label="outside_system")
        ax.plot(x, maxs, marker="o", label="max_score_fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(h_leg["cluster_id"], rotation=35, ha="right")
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("fraction")
        ax.set_title("H candidate admissibility summary")
        ax.legend(loc="upper right")
        fig.tight_layout()
        p = paths.out_fig / "fig4_h_candidate_legality_W045.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        made.append(str(p))

    return made


def _write_summary(paths: AuditPaths, diag: pd.DataFrame, filtered: pd.DataFrame, jw_axis: pd.DataFrame, h_leg: pd.DataFrame, distinct_rev: pd.DataFrame) -> str:
    def _fmt_df(df: pd.DataFrame, cols: List[str], max_rows: int = 20) -> str:
        if df.empty:
            return "(empty)"
        use_cols = [c for c in cols if c in df.columns]
        return df[use_cols].head(max_rows).to_markdown(index=False)

    lines = []
    lines.append(f"# W045 Jw/H Peak Sensitivity Cause Audit v1_1 Summary")
    lines.append("")
    lines.append(f"Generated at: {_now()}")
    lines.append("")
    lines.append("## 1. What v1_1 corrects")
    lines.append("")
    lines.append("v1_1 is a correction audit, not a physical sub-peak classification. It corrects three v1 risks: all-config pair-order inflation, missing Jw axis/core-latitude check, and insufficient H candidate admissibility screening.")
    lines.append("")
    lines.append("## 2. Revised diagnosis")
    lines.append("")
    lines.append(_fmt_df(diag, ["target", "v1_primary_cause", "v1_1_primary_cause", "interpretation_level", "recommended_next_step", "confidence"]))
    lines.append("")
    lines.append("## 3. Filtered Jw-H order")
    lines.append("")
    lines.append(_fmt_df(filtered, ["filter_name", "n_config_pairs", "frac_jw_after_h", "frac_jw_before_h", "frac_near_tie", "median_lag", "order_stability_status", "interpretation_allowed"]))
    lines.append("")
    lines.append("## 4. Jw axis/core-latitude proxy audit")
    lines.append("")
    lines.append(_fmt_df(jw_axis, ["cluster_pair", "max_lat_difference", "core_lat_difference", "axis_shift_status", "physical_distinctness_revision", "interpretation_allowed", "core_lat_proxy_method"]))
    lines.append("")
    lines.append("## 5. H candidate admissibility")
    lines.append("")
    lines.append(_fmt_df(h_leg, ["cluster_id", "day_min", "day_max", "inside_system_fraction", "outside_system_fraction", "max_score_fraction", "candidate_legality_status", "recommended_use"]))
    lines.append("")
    lines.append("## 6. Cluster distinctness revision")
    lines.append("")
    lines.append(_fmt_df(distinct_rev, ["object", "cluster_pair", "v1_distinctness_status", "max_lat_difference", "core_lat_difference", "candidate_legality_pair_status", "axis_shift_status", "revised_distinctness_status", "interpretation_allowed"]))
    lines.append("")
    lines.append("## 7. Interpretation constraints")
    lines.append("")
    lines.append("- Do not treat all-config Jw-H order as a final W045 order unless the filtered/core layers support the same order.")
    lines.append("- Do not interpret H outside-system or max-score locked clusters as W045 subpeaks.")
    lines.append("- A Jw `AXIS_SHIFT_CLEAR` flag only supports a follow-up axis/core-latitude audit; it is not by itself a confirmed physical subpeak.")
    lines.append("- `core_lat` in this patch is a proxy based on v1 `max_lat`, because v1 did not save full profile vectors.")
    text = "\n".join(lines)
    out = paths.out_cross / "W045_JW_H_PEAK_SENSITIVITY_CAUSE_AUDIT_V1_1_SUMMARY.md"
    out.write_text(text, encoding="utf-8")
    return str(out)


def run_peak_sensitivity_cause_audit_w045_jw_h_v1_1(v9_root: Path | str) -> dict:
    paths = _prepare_paths(Path(v9_root))
    _ensure_dirs(paths)
    started = _now()
    errors: List[str] = []
    figures: List[str] = []

    # Load v1 outputs.
    selected = _read_csv(paths.v1_cross / "selected_day_by_config_W045_Jw_H.csv")
    clusters = _read_csv(paths.v1_cross / "selected_day_cluster_summary_W045_Jw_H.csv")
    boundary = _read_csv(paths.v1_cross / "search_window_boundary_audit_W045_Jw_H.csv")
    pair = _read_csv(paths.v1_cross / "jw_h_order_sensitivity_decomposition_W045.csv")
    profile = _read_csv(paths.v1_cross / "profile_component_audit_W045_Jw_H.csv", required=False)
    distinct = _read_csv(paths.v1_cross / "cluster_physical_distinctness_audit_W045_Jw_H.csv", required=False)
    v1_diag = _read_csv(paths.v1_cross / "sensitivity_cause_diagnosis_W045_Jw_H.csv", required=False)

    # Build v1_1 outputs.
    admiss = build_config_admissibility(selected, boundary, clusters)
    filtered_order = build_filtered_order(pair, admiss)
    jw_axis = build_jw_axis_core_shift(profile)
    h_leg = build_h_candidate_admissibility(clusters, admiss, distinct)
    distinct_rev = build_cluster_distinctness_revision(distinct, profile, jw_axis, h_leg)
    diag = build_final_diagnosis(v1_diag, admiss, filtered_order, jw_axis, h_leg, distinct_rev)

    # Write tables.
    outputs = {}
    tables = {
        "config_admissibility_audit_W045_Jw_H.csv": admiss,
        "jw_h_order_filtered_sensitivity_W045.csv": filtered_order,
        "jw_axis_core_shift_audit_W045.csv": jw_axis,
        "h_candidate_admissibility_audit_W045.csv": h_leg,
        "cluster_distinctness_revision_W045_Jw_H.csv": distinct_rev,
        "sensitivity_cause_diagnosis_W045_Jw_H_v1_1.csv": diag,
    }
    for name, df in tables.items():
        p = paths.out_cross / name
        df.to_csv(p, index=False, encoding="utf-8-sig")
        outputs[name] = str(p)

    # Figures.
    try:
        figures = _make_figures(paths, admiss, filtered_order, jw_axis, h_leg)
    except Exception:
        errors.append("figure_generation_failed:\n" + traceback.format_exc())

    summary_path = _write_summary(paths, diag, filtered_order, jw_axis, h_leg, distinct_rev)
    outputs[Path(summary_path).name] = summary_path

    # Summary JSON.
    h_primary = diag.loc[diag["target"] == "H", "v1_1_primary_cause"].iloc[0] if not diag.empty else "NA"
    jw_primary = diag.loc[diag["target"] == "Jw", "v1_1_primary_cause"].iloc[0] if not diag.empty else "NA"
    pair_primary = diag.loc[diag["target"] == "Jw-H", "v1_1_primary_cause"].iloc[0] if not diag.empty else "NA"
    summary = {
        "audit_name": V1_1_OUTPUT_NAME,
        "window_id": WINDOW_ID,
        "objects": list(OBJECTS),
        "role": "v1_1 correction audit; not physical subpeak classification",
        "v1_input_root": str(paths.v1_root),
        "n_selected_rows": int(len(selected)),
        "n_pair_rows": int(len(pair)),
        "n_admissibility_rows": int(len(admiss)),
        "n_filtered_order_rows": int(len(filtered_order)),
        "h_v1_1_primary_cause": h_primary,
        "jw_v1_1_primary_cause": jw_primary,
        "jw_h_v1_1_primary_cause": pair_primary,
        "core_lat_proxy_method": "max_lat_from_v1_profile_component_audit",
        "figures": figures,
        "errors": errors,
    }
    _write_json(paths.out_root / "summary.json", summary)

    run_meta = {
        "audit_name": V1_1_OUTPUT_NAME,
        "started_at_utc": started,
        "completed_at_utc": _now(),
        "status": "success_with_warnings" if errors else "success",
        "v9_root": str(paths.v9_root),
        "v1_input_root": str(paths.v1_root),
        "v1_1_output_root": str(paths.out_root),
        "does_not_rerun_v9_peak_detector": True,
        "does_not_rerun_v1_sensitivity": True,
        "does_not_redefine_accepted_windows": True,
        "does_not_perform_physical_subpeak_classification": True,
        "reads_v1_outputs_only": True,
        "core_lat_proxy_method": "max_lat_from_v1_profile_component_audit",
        "system_window_fallback": list(SYSTEM_WINDOW_FALLBACK),
        "near_tie_days": NEAR_TIE_DAYS,
        "outputs": outputs,
        "figures": figures,
        "errors": errors,
    }
    _write_json(paths.out_root / "run_meta.json", run_meta)

    # Last run log.
    log_text = [
        f"[{_now()}] Completed {V1_1_OUTPUT_NAME}",
        f"v9_root={paths.v9_root}",
        f"v1_input_root={paths.v1_root}",
        f"output_root={paths.out_root}",
        f"status={run_meta['status']}",
        f"summary={paths.out_root / 'summary.json'}",
    ]
    if errors:
        log_text.append("Errors/warnings:\n" + "\n".join(errors))
    (paths.out_log / "last_run.txt").write_text("\n".join(log_text), encoding="utf-8")

    return run_meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run W045 Jw/H peak sensitivity cause audit v1_1")
    parser.add_argument("--v9-root", type=str, default=None, help="Path to V9 root. Defaults to two levels above this file when run from installed tree.")
    args = parser.parse_args()
    if args.v9_root:
        root = Path(args.v9_root)
    else:
        root = Path(__file__).resolve().parents[2]
    meta = run_peak_sensitivity_cause_audit_w045_jw_h_v1_1(root)
    print(json.dumps({"status": meta.get("status"), "output_root": meta.get("v1_1_output_root")}, ensure_ascii=False, indent=2))
