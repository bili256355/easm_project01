from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .config import W045PreclusterConfig


def _find_col(cols: list[str], *tokens: str) -> str | None:
    for c in cols:
        cl = c.lower()
        if all(t.lower() in cl for t in tokens):
            return c
    return None


def _filter_w045_h(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    cols = list(out.columns)
    # Object filter.
    obj_col = _find_col(cols, "object") or _find_col(cols, "target")
    if obj_col:
        out = out[out[obj_col].astype(str).str.lower().isin(["h", "z500"])]
    # Window filter.
    win_cols = [c for c in out.columns if "window" in c.lower() or "lineage" in c.lower() or "cluster" in c.lower()]
    if win_cols:
        mask = False
        for c in win_cols:
            s = out[c].astype(str)
            mask = mask | s.str.contains("45|W045|w045", regex=True, na=False)
        out = out[mask]
    # If no rows after filtering but H rows existed, return original H rows for manual/fuzzy audit.
    return out


def _numeric_or_nan(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def audit_h35_role(
    cfg: W045PreclusterConfig,
    metrics: pd.DataFrame,
    competition_df: pd.DataFrame | None,
    family_summary_df: pd.DataFrame | None,
) -> pd.DataFrame:
    h_e2 = metrics[(metrics["cluster_id"] == "E2_second_precluster") & (metrics["object"] == "H")]
    h_m = metrics[(metrics["cluster_id"] == "M_w045_main_cluster") & (metrics["object"] == "H")]
    h_post = metrics[(metrics["cluster_id"] == "H_post_reference") & (metrics["object"] == "H")]

    candidate_h35_inside = bool(h_e2["candidate_inside_cluster_flag"].iloc[0]) if not h_e2.empty else False
    h_has_m_candidate = bool(h_m["candidate_inside_cluster_flag"].iloc[0]) if not h_m.empty else False
    h_post_candidate_day = float(h_post["nearest_candidate_day"].iloc[0]) if not h_post.empty else np.nan

    comp_rows = _filter_w045_h(competition_df) if competition_df is not None else pd.DataFrame()
    energy_days = []
    assigned_days = []
    ratios = []
    top1_fracs = []
    topk_fracs = []
    if not comp_rows.empty:
        cols = list(comp_rows.columns)
        energy_col = _find_col(cols, "energy", "day") or _find_col(cols, "dominant", "day") or _find_col(cols, "top1", "day")
        assigned_col = _find_col(cols, "assigned", "day") or _find_col(cols, "lineage", "day")
        ratio_col = _find_col(cols, "ratio")
        top1_col = _find_col(cols, "top1", "fraction") or _find_col(cols, "top1", "frac")
        topk_col = _find_col(cols, "topk", "fraction") or _find_col(cols, "topk", "frac")
        for _, r in comp_rows.iterrows():
            if energy_col:
                energy_days.append(_numeric_or_nan(r[energy_col]))
            if assigned_col:
                assigned_days.append(_numeric_or_nan(r[assigned_col]))
            if ratio_col:
                ratios.append(_numeric_or_nan(r[ratio_col]))
            if top1_col:
                top1_fracs.append(_numeric_or_nan(r[top1_col]))
            if topk_col:
                topk_fracs.append(_numeric_or_nan(r[topk_col]))

    fam_rows = _filter_w045_h(family_summary_df) if family_summary_df is not None else pd.DataFrame()
    family_status = "missing"
    if not fam_rows.empty:
        status_cols = [c for c in fam_rows.columns if "status" in c.lower() or "support" in c.lower()]
        text = " | ".join(str(v) for c in status_cols for v in fam_rows[c].dropna().astype(str).tolist())
        if "TOP1" in text.upper():
            family_status = "top1_or_topk_status_found: " + text[:250]
        elif text:
            family_status = text[:250]

    not_energy_dominant = bool(energy_days and all(abs(d - 35) > 2 for d in energy_days if not np.isnan(d)))
    not_top1_supported = bool(top1_fracs and np.nanmax(top1_fracs) < 0.5)
    topk_supported = bool(topk_fracs and np.nanmax(topk_fracs) >= 0.5)

    e2_other_active = []
    for obj in ("P", "V", "Je"):
        part = metrics[(metrics["cluster_id"] == "E2_second_precluster") & (metrics["object"] == obj)]
        if not part.empty and part["participation_class"].iloc[0] == "candidate_inside_cluster":
            e2_other_active.append(obj)

    forbid_conditions = {
        "H35_not_energy_dominant": not_energy_dominant,
        "H35_not_top1_supported": not_top1_supported or "TOPK" in family_status.upper(),
        "P_V_Je_also_active_in_E2": len(e2_other_active) >= 2,
        "H_has_no_marker_inside_M": not h_has_m_candidate,
        "yearwise_prediction_not_tested": True,
        "spatial_continuity_not_tested": True,
    }
    n_forbid = sum(bool(v) for v in forbid_conditions.values())
    confirmed_weak_precursor = False
    if candidate_h35_inside and n_forbid <= 1 and topk_supported and not not_energy_dominant:
        confirmed_weak_precursor = True

    if confirmed_weak_precursor:
        role_class = "confirmed_weak_precursor"
    elif candidate_h35_inside and (not_energy_dominant or not_top1_supported) and not h_has_m_candidate:
        role_class = "lineage_assigned_secondary_or_background_conditioning_component"
    elif candidate_h35_inside:
        role_class = "possible_weak_precursor_but_unproven"
    else:
        role_class = "not_supported_as_H35_candidate"

    row = {
        "target_object": "H",
        "target_day": 35,
        "target_cluster": "E2_second_precluster",
        "candidate_inside_E2": candidate_h35_inside,
        "h_has_candidate_in_M_cluster": h_has_m_candidate,
        "h_post_reference_nearest_candidate_day": h_post_candidate_day,
        "energy_dominant_family_days_detected": ";".join(str(int(d)) for d in energy_days if not np.isnan(d)),
        "lineage_assigned_family_days_detected": ";".join(str(int(d)) for d in assigned_days if not np.isnan(d)),
        "assigned_to_energy_score_ratio_min": float(np.nanmin(ratios)) if ratios else np.nan,
        "assigned_to_energy_score_ratio_max": float(np.nanmax(ratios)) if ratios else np.nan,
        "top1_fraction_max": float(np.nanmax(top1_fracs)) if top1_fracs else np.nan,
        "topk_fraction_max": float(np.nanmax(topk_fracs)) if topk_fracs else np.nan,
        "family_summary_status_excerpt": family_status,
        "other_E2_active_objects": ";".join(e2_other_active),
        "forbid_confirmed_weak_precursor_conditions": ";".join(k for k, v in forbid_conditions.items() if v),
        "confirmed_weak_precursor": confirmed_weak_precursor,
        "role_class": role_class,
        "recommended_wording": (
            "H day35 should be described as a W045-context H family inside the E2 second precluster. "
            "Current evidence is insufficient to call it a confirmed weak precursor; it is safer to treat it as "
            "a lineage-assigned secondary/background-conditioning component until yearwise and spatial checks are completed. "
            "HOTFIX02 interpretation rule: E2 marker-supported objects and curve-only/ramp objects must be separated."
        ),
        "forbidden_wording": (
            "Do not write: H day35 is a confirmed/stable weak precursor that triggers W045. "
            "Do not write: H leads W045 as a single clean event."
        ),
    }
    return pd.DataFrame([row])


def build_interpretation_summary(metrics: pd.DataFrame, h_role: pd.DataFrame, morphology: pd.DataFrame) -> pd.DataFrame:
    claims = []
    def add(claim_id: str, text: str, support: str, layer: str, risk: str, usage: str):
        claims.append({
            "claim_id": claim_id,
            "claim_text": text,
            "support_level": support,
            "interpretation_layer": layer,
            "risk": risk,
            "recommended_usage": usage,
        })

    def _objs(cluster_id: str, cls: str) -> list[str]:
        return metrics[(metrics["cluster_id"] == cluster_id) & (metrics["participation_class"] == cls)]["object"].tolist()

    e1_marker = _objs("E1_early_precluster", "candidate_inside_cluster")
    e2_marker = _objs("E2_second_precluster", "candidate_inside_cluster")
    m_marker = _objs("M_w045_main_cluster", "candidate_inside_cluster")
    e1_curve = _objs("E1_early_precluster", "curve_peak_without_marker")
    e2_curve = _objs("E2_second_precluster", "curve_peak_without_marker")
    m_curve = _objs("M_w045_main_cluster", "curve_peak_without_marker")

    add(
        "W045_CLAIM_001",
        f"E1/E2/M candidate-cluster decomposition is detected from fixed windows: marker-supported core objects are E1={e1_marker}, E2={e2_marker}, M={m_marker}; curve-only/ramp objects are E1={e1_curve}, E2={e2_curve}, M={m_curve}.",
        "method_supported_fixed_window_audit",
        "derived_structure",
        "Fixed cluster windows; no yearwise/spatial validation in v10.6_a.",
        "Use as W045 event-semantics audit, not as final physical process proof.",
    )
    if not h_role.empty:
        add(
            "W045_CLAIM_002",
            str(h_role["recommended_wording"].iloc[0]),
            "supported_as_interpretation_boundary",
            "interpretation_constraint",
            "H35 role is not causal; classification depends on V10.5_b/d availability.",
            "Use to forbid over-strong H weak-precursor language.",
        )
    add(
        "W045_CLAIM_003",
        "V10.6_a does not test yearwise prediction, spatial continuity, or causality.",
        "explicit_not_implemented",
        "engineering_state",
        "Do not use V10.6_a to prove physical lead-lag mechanism.",
        "Record as boundary for future V10.6_b/c.",
    )
    return pd.DataFrame(claims)
