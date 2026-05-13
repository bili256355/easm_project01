from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .settings import LeadLagScreenSettings


_SEVERITY_ORDER = {
    "none": 0,
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}


def _severity_max(a: object, b: object) -> str:
    a = "none" if pd.isna(a) else str(a)
    b = "none" if pd.isna(b) else str(b)
    return a if _SEVERITY_ORDER.get(a, 0) >= _SEVERITY_ORDER.get(b, 0) else b


def _safe_bool(x: object) -> bool:
    if pd.isna(x):
        return False
    return bool(x)


def _family_map(settings: LeadLagScreenSettings) -> Dict[str, str]:
    return dict(settings.variable_families)


def _attach_phi_risk(
    df: pd.DataFrame,
    ar1_params: pd.DataFrame,
    settings: LeadLagScreenSettings,
) -> pd.DataFrame:
    fam = _family_map(settings)
    phi_cols = [
        "window",
        "variable",
        "raw_phi_before_clip",
        "phi_after_clip",
        "phi_clip_amount",
        "phi_clip_severity",
        "phi_clipped_flag",
    ]
    missing = [c for c in phi_cols if c not in ar1_params.columns]
    if missing:
        out = df.copy()
        out["source_raw_phi"] = np.nan
        out["target_raw_phi"] = np.nan
        out["source_phi_clip_severity"] = "unknown"
        out["target_phi_clip_severity"] = "unknown"
        out["pair_phi_risk"] = "unknown"
        return out

    phi = ar1_params[phi_cols].copy()
    src_phi = phi.rename(columns={
        "variable": "source",
        "raw_phi_before_clip": "source_raw_phi",
        "phi_after_clip": "source_phi_after_clip",
        "phi_clip_amount": "source_phi_clip_amount",
        "phi_clip_severity": "source_phi_clip_severity",
        "phi_clipped_flag": "source_phi_clipped_flag",
    })
    tgt_phi = phi.rename(columns={
        "variable": "target",
        "raw_phi_before_clip": "target_raw_phi",
        "phi_after_clip": "target_phi_after_clip",
        "phi_clip_amount": "target_phi_clip_amount",
        "phi_clip_severity": "target_phi_clip_severity",
        "phi_clipped_flag": "target_phi_clipped_flag",
    })

    out = df.merge(src_phi, on=["window", "source"], how="left")
    out = out.merge(tgt_phi, on=["window", "target"], how="left")
    out["source_phi_clip_severity"] = out["source_phi_clip_severity"].fillna("none")
    out["target_phi_clip_severity"] = out["target_phi_clip_severity"].fillna("none")
    out["pair_phi_risk"] = [
        _severity_max(a, b)
        for a, b in zip(out["source_phi_clip_severity"], out["target_phi_clip_severity"])
    ]
    if "source_family" not in out.columns:
        out["source_family"] = out["source"].map(fam)
    if "target_family" not in out.columns:
        out["target_family"] = out["target"].map(fam)
    return out


def _classify_evidence_tier(row: pd.Series) -> Tuple[str, str, str]:
    group = row.get("lead_lag_group", "")
    label = row.get("lead_lag_label", "")
    direction = row.get("direction_label", "")
    risk = row.get("pair_phi_risk", "none")
    audit_p = row.get("p_pos_audit_surrogate", np.nan)
    audit_q = row.get("q_pos_audit_within_window", np.nan)

    audit_p_pass = bool(np.isfinite(audit_p) and audit_p <= 0.05)
    audit_q_pass = bool(np.isfinite(audit_q) and audit_q <= 0.10)
    severe_phi = risk == "severe"
    moderate_phi = risk == "moderate"

    if group == "lead_lag_yes":
        if audit_q_pass and not severe_phi:
            if label == "lead_lag_yes_clear":
                return (
                    "Tier1a_audit_stable_clear_leadlag",
                    "main yes; audit-FDR pass; clear lead-lag; no severe phi risk",
                    "strict temporal core for pathway candidate screening",
                )
            return (
                "Tier1b_audit_stable_with_same_day",
                "main yes; audit-FDR pass; same-day coupling present; no severe phi risk",
                "usable strict temporal candidate, but must retain same-day coupling warning",
            )

        if audit_p_pass and not severe_phi:
            return (
                "Tier2_main_supported_audit_moderate",
                "main yes; audit p pass but audit-FDR or phi-risk prevents Tier1",
                "usable candidate with audit/moderate-risk flag; not a core-only claim",
            )

        return (
            "Tier3_surrogate_or_persistence_sensitive_yes",
            "main yes but audit p fails and/or severe phi persistence risk",
            "do not use as strict pathway entry; retain in risk pool",
        )

    if group == "lead_lag_ambiguous":
        if "bidirectional" in str(label) or "bidirectional" in str(direction):
            subtype = "Tier4a_ambiguous_bidirectional_close"
            reason = "ambiguous bidirectional/feedback-like temporal structure"
        elif "same_day" in str(label) or "same_day" in str(direction):
            subtype = "Tier4b_ambiguous_same_day_dominant"
            reason = "ambiguous because same-day coupling is dominant or mixed"
        elif "marginal" in str(label):
            subtype = "Tier4d_ambiguous_marginal_statistical_support"
            reason = "ambiguous because statistical support is marginal"
        elif "coupled" in str(label) or "feedback" in str(label):
            subtype = "Tier4e_ambiguous_coupled_or_feedback_like"
            reason = "ambiguous coupled/feedback-like temporal pattern"
        else:
            subtype = "Tier4c_ambiguous_direction_uncertain"
            reason = "ambiguous because direction is uncertain or year-sensitive"
        return (
            subtype,
            reason,
            "expanded temporal risk pool only; useful for feedback/same-day/transition audit",
        )

    if group == "lead_lag_no":
        if "reverse" in str(label) or "reverse" in str(direction):
            return (
                "Tier5b_no_reverse_dominant",
                "original direction not supported; reverse direction is temporally stronger",
                "do not use original direction; inspect suggested reverse if scientifically relevant",
            )
        if "same_day_only" in str(label):
            return (
                "Tier5c_no_same_day_only",
                "same-day coupling exists but no positive-lag support",
                "do not use as lead-lag candidate; retain as same-day coupling evidence",
            )
        return (
            "Tier5a_no_not_supported",
            "positive-lag statistical support not established",
            "exclude from strict and expanded lead-lag candidate pools unless needed as negative evidence",
        )

    if group == "not_evaluable" or "not_evaluable" in str(label):
        return (
            "Tier0_not_evaluable",
            "insufficient sample or non-evaluable configuration",
            "do not interpret as no; inspect sample coverage",
        )

    return (
        "TierX_unclassified",
        "unclassified due to missing or unexpected label fields",
        "manual inspection required",
    )


def build_evidence_tier_outputs(
    pair_summary: pd.DataFrame,
    classified: pd.DataFrame,
    dir_all: pd.DataFrame,
    audit_null: pd.DataFrame,
    ar1_params: pd.DataFrame,
    settings: LeadLagScreenSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build evidence-tier post-processing tables without changing primary labels.
    """

    key = ["window", "source", "target"]

    # Start from classified because it retains the broadest set of null and direction fields.
    tier = classified.copy()

    # Attach audit null fields.
    if audit_null is not None and not audit_null.empty:
        audit_cols = [
            "window", "source", "target",
            "p_pos_audit_surrogate",
            "q_pos_audit_within_window",
            "q_pos_audit_global",
            "audit_support_shift",
            "T_pos_audit_null95",
            "audit_strength_excess",
            "audit_strength_ratio",
            "delta_p_audit_minus_main",
            "delta_null95_audit_minus_main",
        ]
        audit_cols = [c for c in audit_cols if c in audit_null.columns]
        tier = tier.merge(audit_null[audit_cols], on=key, how="left")

    tier = _attach_phi_risk(tier, ar1_params, settings)

    # Apply tier classification.
    records = tier.apply(_classify_evidence_tier, axis=1, result_type="expand")
    tier["evidence_tier"] = records[0]
    tier["evidence_tier_reason"] = records[1]
    tier["recommended_usage"] = records[2]

    # Keep a focused, stable main table column order. Missing columns are skipped.
    preferred_cols = [
        "window", "source", "target", "source_family", "target_family",
        "lead_lag_label", "lead_lag_group", "direction_label",
        "same_day_coupling_flag",
        "p_pos_surrogate", "q_pos_within_window", "q_pos_global",
        "p_pos_audit_surrogate", "q_pos_audit_within_window", "q_pos_audit_global",
        "audit_support_shift",
        "positive_peak_lag", "positive_peak_signed_r", "positive_peak_abs_r",
        "negative_peak_lag", "negative_peak_signed_r", "negative_peak_abs_r",
        "lag0_signed_r", "lag0_abs_r",
        "D_pos_neg", "D_pos_neg_CI90_low", "D_pos_neg_CI90_high",
        "D_pos_neg_CI95_low", "D_pos_neg_CI95_high", "P_D_pos_neg_gt_0",
        "D_pos_0", "D_pos_0_CI90_low", "D_pos_0_CI90_high",
        "D_pos_0_CI95_low", "D_pos_0_CI95_high", "P_D_pos_0_gt_0",
        "source_raw_phi", "target_raw_phi",
        "source_phi_after_clip", "target_phi_after_clip",
        "source_phi_clip_severity", "target_phi_clip_severity", "pair_phi_risk",
        "source_phi_clip_amount", "target_phi_clip_amount",
        "evidence_tier", "evidence_tier_reason", "recommended_usage",
        "failure_reason", "risk_note", "suggested_reverse_direction",
    ]
    tier_out = tier[[c for c in preferred_cols if c in tier.columns]].copy()

    # Rollup by window and family direction.
    grouped = []
    for (window, sfam, tfam), g in tier_out.groupby(["window", "source_family", "target_family"], dropna=False):
        n_total = len(g)
        n_main_yes = int((g["lead_lag_group"] == "lead_lag_yes").sum())
        n_audit_stable_yes = int(g["evidence_tier"].astype(str).str.startswith("Tier1").sum())
        n_tier1 = n_audit_stable_yes
        n_tier2 = int(g["evidence_tier"].astype(str).str.startswith("Tier2").sum())
        n_tier3 = int(g["evidence_tier"].astype(str).str.startswith("Tier3").sum())
        n_ambiguous = int((g["lead_lag_group"] == "lead_lag_ambiguous").sum())
        n_reverse = int(g["lead_lag_label"].astype(str).str.contains("reverse", na=False).sum())
        n_same_day_only = int(g["lead_lag_label"].astype(str).str.contains("same_day_only", na=False).sum())
        n_same_day_flag = int(g.get("same_day_coupling_flag", pd.Series(False, index=g.index)).fillna(False).astype(bool).sum())
        n_high_persist = int(g["pair_phi_risk"].isin(["moderate", "severe"]).sum()) if "pair_phi_risk" in g else 0
        grouped.append({
            "window": window,
            "source_family": sfam,
            "target_family": tfam,
            "n_total_pairs": n_total,
            "n_main_yes": n_main_yes,
            "n_audit_stable_yes": n_audit_stable_yes,
            "n_tier1": n_tier1,
            "n_tier2": n_tier2,
            "n_tier3": n_tier3,
            "n_ambiguous": n_ambiguous,
            "n_reverse": n_reverse,
            "n_same_day_only": n_same_day_only,
            "tier1_fraction": n_tier1 / n_total if n_total else np.nan,
            "audit_stable_fraction_among_main_yes": n_audit_stable_yes / n_main_yes if n_main_yes else np.nan,
            "same_day_coupling_fraction": n_same_day_flag / n_total if n_total else np.nan,
            "high_persistence_risk_fraction": n_high_persist / n_total if n_total else np.nan,
        })
    rollup = pd.DataFrame(grouped)

    # Phi risk audit by window x variable.
    phi = ar1_params.copy()
    if not phi.empty:
        fam = _family_map(settings)
        phi["family"] = phi["variable"].map(fam)
        source_counts = tier_out.groupby(["window", "source"]).agg(
            n_as_source_pairs=("source", "size"),
            n_as_source_yes=("lead_lag_group", lambda s: int((s == "lead_lag_yes").sum())),
            n_as_source_tier1=("evidence_tier", lambda s: int(s.astype(str).str.startswith("Tier1").sum())),
            n_as_source_surrogate_sensitive=("evidence_tier", lambda s: int(s.astype(str).str.startswith("Tier3").sum())),
        ).reset_index().rename(columns={"source": "variable"})
        target_counts = tier_out.groupby(["window", "target"]).agg(
            n_as_target_pairs=("target", "size"),
            n_as_target_yes=("lead_lag_group", lambda s: int((s == "lead_lag_yes").sum())),
            n_as_target_tier1=("evidence_tier", lambda s: int(s.astype(str).str.startswith("Tier1").sum())),
            n_as_target_surrogate_sensitive=("evidence_tier", lambda s: int(s.astype(str).str.startswith("Tier3").sum())),
        ).reset_index().rename(columns={"target": "variable"})
        phi_audit = phi.merge(source_counts, on=["window", "variable"], how="left")
        phi_audit = phi_audit.merge(target_counts, on=["window", "variable"], how="left")
        count_cols = [c for c in phi_audit.columns if c.startswith("n_as_")]
        phi_audit[count_cols] = phi_audit[count_cols].fillna(0).astype(int)
    else:
        phi_audit = pd.DataFrame()

    # Warning flags.
    warnings = []
    severe_phi = phi_audit[phi_audit.get("phi_clip_severity", pd.Series(dtype=str)) == "severe"] if not phi_audit.empty else pd.DataFrame()
    if not severe_phi.empty:
        warnings.append({
            "scope_type": "project",
            "scope_id": "all",
            "warning_type": "high_phi_clipping",
            "warning_level": "high",
            "warning_message": f"{len(severe_phi)} window-variable AR(1) parameter rows have severe phi clipping.",
            "affected_pairs": int(tier_out["pair_phi_risk"].eq("severe").sum()) if "pair_phi_risk" in tier_out else np.nan,
            "suggested_action": "Do not freeze strict pool without reporting high-persistence risk; inspect severe variables/windows.",
        })

    if "audit_support_shift" in tier_out:
        conservative = tier_out["audit_support_shift"].eq("audit_more_conservative").sum()
        if conservative:
            warnings.append({
                "scope_type": "project",
                "scope_id": "all",
                "warning_type": "audit_null_more_conservative",
                "warning_level": "medium",
                "warning_message": f"Audit null is more conservative for {int(conservative)} pairs.",
                "affected_pairs": int(conservative),
                "suggested_action": "Prefer Tier1/Tier2 split instead of treating all main yes equally.",
            })

    n_yes = int((tier_out["lead_lag_group"] == "lead_lag_yes").sum())
    n_yes_same = int(tier_out["lead_lag_label"].eq("lead_lag_yes_with_same_day_coupling").sum())
    if n_yes and n_yes_same / n_yes >= 0.50:
        warnings.append({
            "scope_type": "project",
            "scope_id": "all",
            "warning_type": "many_yes_with_same_day_coupling",
            "warning_level": "medium",
            "warning_message": f"{n_yes_same}/{n_yes} main yes pairs include same-day coupling.",
            "affected_pairs": n_yes_same,
            "suggested_action": "Do not interpret main yes as pure one-way lag propagation.",
        })

    # Window-specific audit collapse warnings.
    for window, g in tier_out.groupby("window"):
        main_yes = int((g["lead_lag_group"] == "lead_lag_yes").sum())
        tier1 = int(g["evidence_tier"].astype(str).str.startswith("Tier1").sum())
        if main_yes > 0 and tier1 == 0:
            warnings.append({
                "scope_type": "window",
                "scope_id": window,
                "warning_type": "audit_stable_yes_collapse",
                "warning_level": "high",
                "warning_message": f"Window {window} has {main_yes} main yes pairs but zero Tier1 audit-stable yes pairs.",
                "affected_pairs": main_yes,
                "suggested_action": "Treat this window's main yes as audit-sensitive until further checked.",
            })

    warning_df = pd.DataFrame(warnings)
    return tier_out, rollup, phi_audit, warning_df
