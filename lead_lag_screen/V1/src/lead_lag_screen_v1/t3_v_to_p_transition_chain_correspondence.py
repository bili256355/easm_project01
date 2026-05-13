# -*- coding: utf-8 -*-
"""Object-support correspondence and transition-chain diagnosis."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .t3_v_to_p_transition_chain_settings import TransitionChainReportSettings


def _direction(delta: float, eps: float) -> str:
    if not np.isfinite(delta):
        return "unknown"
    if delta > eps:
        return "increase"
    if delta < -eps:
        return "decrease"
    return "near_zero"


def _correspondence(p_dir: str, support_dir: str) -> str:
    if p_dir == "increase" and support_dir == "increase":
        return "P_and_support_increase_together"
    if p_dir == "decrease" and support_dir == "decrease":
        return "P_and_support_decrease_together"
    if p_dir != "increase" and support_dir == "increase":
        return "support_increase_without_P_increase"
    if p_dir == "increase" and support_dir != "increase":
        return "P_increase_without_support_increase"
    if {p_dir, support_dir} == {"increase", "decrease"}:
        return "opposite_change"
    if p_dir == "unknown" or support_dir == "unknown":
        return "unknown"
    return "mixed"


def build_object_support_correspondence(
    object_delta_df: pd.DataFrame,
    support_delta_df: pd.DataFrame,
    settings: TransitionChainReportSettings,
) -> pd.DataFrame:
    obj = object_delta_df.copy()
    sup = support_delta_df.copy()
    merged = sup.merge(obj, on=["comparison", "region"], how="left", suffixes=("_support", "_object"))
    rows: List[dict] = []
    for _, r in merged.iterrows():
        p_delta = float(r.get("P_delta", np.nan))
        v_delta = float(r.get("V850_delta", np.nan))
        support_delta = float(r.get("R2_delta", np.nan))
        p_dir = _direction(p_delta, settings.object_delta_epsilon)
        v_dir = _direction(v_delta, settings.object_delta_epsilon)
        s_dir = _direction(support_delta, settings.r2_delta_epsilon)
        rows.append({
            "comparison": r["comparison"],
            "target_window": r.get("target_window_support", r.get("target_window")),
            "reference_window": r.get("reference_window_support", r.get("reference_window")),
            "v_component": r["v_component"],
            "region": r["region"],
            "P_delta": p_delta,
            "V850_delta": v_delta,
            "support_R2_delta": support_delta,
            "P_direction": p_dir,
            "V850_direction": v_dir,
            "support_direction": s_dir,
            "correspondence_type": _correspondence(p_dir, s_dir),
        })
    return pd.DataFrame(rows)


def _get_delta(support_delta_df: pd.DataFrame, comp: str, region: str, comparison: str) -> float:
    q = support_delta_df[(support_delta_df["v_component"] == comp) & (support_delta_df["region"] == region) & (support_delta_df["comparison"] == comparison)]
    if q.empty:
        return float("nan")
    return float(q.iloc[0]["R2_delta"])


def _support_level_from_delta(delta: float, eps: float) -> str:
    d = _direction(delta, eps)
    if d == "increase":
        return "supported"
    if d == "decrease":
        return "counter_evidence"
    if d == "near_zero":
        return "weak_or_near_zero"
    return "unknown"


def build_transition_chain_diagnosis(
    matrix_df: pd.DataFrame,
    support_delta_df: pd.DataFrame,
    nms_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    settings: TransitionChainReportSettings,
) -> pd.DataFrame:
    rows: List[dict] = []
    eps = settings.r2_delta_epsilon

    # Region-specific change diagnoses with explicit comparison objects.
    for comp in settings.v_components:
        for region, label in [
            (settings.main_region, "main_meiyu_support_change"),
            (settings.south_region, "south_scs_support_change"),
            (settings.north_region, "north_support_change"),
        ]:
            for comparison in ["T3_full_minus_S3", "T3_late_minus_T3_early", "S4_minus_T3_full"]:
                delta = _get_delta(support_delta_df, comp, region, comparison)
                target, reference = settings.comparisons[comparison]
                rows.append({
                    "diagnosis_id": f"{label}__{comp}__{comparison}",
                    "support_level": _support_level_from_delta(delta, eps),
                    "comparison": comparison,
                    "reference_window": reference,
                    "target_window": target,
                    "region": region,
                    "v_component": comp,
                    "primary_evidence_value": delta,
                    "counter_evidence_value": float("nan"),
                    "allowed_statement": f"For {comp}, {region} support { _direction(delta, eps) } in {target} relative to {reference}.",
                    "forbidden_statement": "Do not say enhancement/weakening without naming this comparison; do not infer physical takeover from this single support delta.",
                })

    # Full-window dilution: T3_full lower than max early/late.
    dil = support_delta_df[support_delta_df["comparison"] == "T3_full_minus_max_subwindow"].copy()
    for comp in settings.v_components:
        q = dil[dil["v_component"] == comp]
        if q.empty:
            continue
        n = int(len(q))
        n_dil = int((q["R2_delta"] < -eps).sum())
        level = "supported" if n_dil >= max(1, int(0.6 * n)) else ("mixed" if n_dil > 0 else "not_supported")
        rows.append({
            "diagnosis_id": f"full_window_dilution__{comp}",
            "support_level": level,
            "comparison": "T3_full_minus_max_subwindow",
            "reference_window": "max(T3_early,T3_late)",
            "target_window": "T3_full",
            "region": "all_pre_registered_regions",
            "v_component": comp,
            "primary_evidence_value": n_dil,
            "counter_evidence_value": n - n_dil,
            "allowed_statement": f"For {comp}, {n_dil}/{n} regions have T3_full support lower than the stronger T3 subwindow.",
            "forbidden_statement": "Do not equate full-window dilution with T3_early=S3 or T3_late=S4; that requires separate similarity evidence.",
        })

    # North compensation candidate: must be explicitly comparison-based.
    chain = nms_df.set_index(["v_component", "window"])
    for comp in settings.v_components:
        for comparison, (target, reference) in settings.comparisons.items():
            try:
                t = chain.loc[(comp, target)]
                r = chain.loc[(comp, reference)]
            except KeyError:
                continue
            north_delta = float(t["north_R2"] - r["north_R2"])
            nm_delta = float(t["north_minus_main"] - r["north_minus_main"])
            ns_delta = float(t["north_minus_south"] - r["north_minus_south"])
            supported = north_delta > eps and nm_delta > eps and ns_delta > eps
            rows.append({
                "diagnosis_id": f"north_compensation_candidate__{comp}__{comparison}",
                "support_level": "supported" if supported else "not_supported",
                "comparison": comparison,
                "reference_window": reference,
                "target_window": target,
                "region": settings.north_region,
                "v_component": comp,
                "primary_evidence_value": north_delta,
                "counter_evidence_value": min(nm_delta, ns_delta),
                "allowed_statement": f"For {comp}, north compensation is {'a candidate' if supported else 'not supported'} in {target} relative to {reference}; criteria require north_R2, north-main, and north-south to increase.",
                "forbidden_statement": "Do not call north increase a takeover unless north also becomes the dominant region and object-level evidence supports it.",
            })

    # Dominant-region shift.
    for comp in settings.v_components:
        for comparison, (target, reference) in settings.comparisons.items():
            try:
                t_dom = str(chain.loc[(comp, target)]["dominant_region"])
                r_dom = str(chain.loc[(comp, reference)]["dominant_region"])
            except KeyError:
                continue
            shift = t_dom != r_dom
            rows.append({
                "diagnosis_id": f"dominant_region_shift__{comp}__{comparison}",
                "support_level": "supported" if shift else "not_supported",
                "comparison": comparison,
                "reference_window": reference,
                "target_window": target,
                "region": f"{r_dom}_to_{t_dom}",
                "v_component": comp,
                "primary_evidence_value": 1.0 if shift else 0.0,
                "counter_evidence_value": float("nan"),
                "allowed_statement": f"For {comp}, dominant support region changes from {r_dom} in {reference} to {t_dom} in {target}." if shift else f"For {comp}, dominant support region remains {t_dom} from {reference} to {target}.",
                "forbidden_statement": "Do not claim target shift if dominant region does not change or if the statement lacks the reference/target windows.",
            })

    # Object-support correspondence overview by comparison.
    for comparison in sorted(corr_df["comparison"].unique()):
        q = corr_df[corr_df["comparison"] == comparison]
        together = int(q["correspondence_type"].isin(["P_and_support_increase_together", "P_and_support_decrease_together"]).sum())
        total = int(len(q))
        rows.append({
            "diagnosis_id": f"object_support_correspondence_overview__{comparison}",
            "support_level": "mixed" if total and together < total else "broadly_consistent",
            "comparison": comparison,
            "reference_window": str(q["reference_window"].iloc[0]) if total else "unknown",
            "target_window": str(q["target_window"].iloc[0]) if total else "unknown",
            "region": "all_regions_components",
            "v_component": "all",
            "primary_evidence_value": together,
            "counter_evidence_value": total - together,
            "allowed_statement": f"Object-support correspondence is mixed: {together}/{total} component-region cases have P and support changing together.",
            "forbidden_statement": "Do not infer physical correspondence from support maps alone; consult object_support_correspondence_summary.csv.",
        })

    return pd.DataFrame(rows)
