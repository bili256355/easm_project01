from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StabilityThresholds:
    ci_level: str = "90"
    p_lag_gt_tau0_threshold: float = 0.90
    p_forward_gt_reverse_threshold: float = 0.90
    core_evidence_prefixes: tuple[str, ...] = ("Tier1a", "Tier1b", "Tier2")
    sensitive_evidence_prefixes: tuple[str, ...] = ("Tier3",)


def _get(row: pd.Series, candidates: list[str], default=np.nan):
    lower = {str(k).lower(): k for k in row.index}
    for c in candidates:
        if c in row.index:
            return row[c]
        key = lower.get(c.lower())
        if key is not None:
            return row[key]
    return default


def _float(v) -> float:
    try:
        if pd.isna(v):
            return math.nan
        return float(v)
    except Exception:
        return math.nan


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def evidence_tier_prefix(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split("_", 1)[0]


def _same_day_flag(row: pd.Series) -> bool:
    for c in ["same_day_coupling_flag", "same_day_flag", "has_same_day_coupling", "lag0_significant"]:
        val = _get(row, [c], None)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            if _bool(val):
                return True
    txt = " ".join(str(_get(row, [c], "")) for c in ["evidence_tier", "lead_lag_label", "lead_lag_group"])
    txt = txt.lower()
    return "same" in txt or "lag0" in txt or "tier1b" in txt


def _yes_like(prefix: str, label: str) -> bool:
    if prefix.startswith("Tier1") or prefix in {"Tier2", "Tier3"}:
        return True
    return "lead_lag_yes" in str(label).lower()


def _prob_ge(row: pd.Series, names: list[str], threshold: float) -> bool:
    p = _float(_get(row, names, math.nan))
    return math.isnan(p) or p >= threshold


def _prob_le(row: pd.Series, names: list[str], threshold: float) -> bool:
    p = _float(_get(row, names, math.nan))
    return math.isnan(p) or p <= threshold


def classify_stability(row: pd.Series, th: StabilityThresholds) -> dict[str, object]:
    ci = th.ci_level
    tier_raw = str(_get(row, ["evidence_tier"], "")).strip()
    prefix = evidence_tier_prefix(tier_raw)
    label = str(_get(row, ["lead_lag_label", "lead_lag_group"], "")).strip()

    d_pos_0 = _float(_get(row, ["D_pos_0"]))
    d_pos_0_low = _float(_get(row, [f"D_pos_0_CI{ci}_low"]))
    d_pos_0_high = _float(_get(row, [f"D_pos_0_CI{ci}_high"]))
    p_pos0_gt = _float(_get(row, ["P_D_pos_0_gt_0"]))

    d_pos_neg = _float(_get(row, ["D_pos_neg"]))
    d_pos_neg_low = _float(_get(row, [f"D_pos_neg_CI{ci}_low"]))
    d_pos_neg_high = _float(_get(row, [f"D_pos_neg_CI{ci}_high"]))
    p_posneg_gt = _float(_get(row, ["P_D_pos_neg_gt_0"]))

    same_day = _same_day_flag(row)
    core = prefix in set(th.core_evidence_prefixes)
    sensitive = prefix in set(th.sensitive_evidence_prefixes)
    yes_like = _yes_like(prefix, label)

    missing_pos0 = math.isnan(d_pos_0_low) or math.isnan(d_pos_0_high)
    missing_posneg = math.isnan(d_pos_neg_low) or math.isnan(d_pos_neg_high)

    lag_dom = (
        not missing_pos0
        and d_pos_0_low > 0
        and _prob_ge(row, ["P_D_pos_0_gt_0"], th.p_lag_gt_tau0_threshold)
    )
    tau0_dom = (
        not missing_pos0
        and d_pos_0_high < 0
        and _prob_le(row, ["P_D_pos_0_gt_0"], 1.0 - th.p_lag_gt_tau0_threshold)
    )
    close_tau0 = (not missing_pos0) and d_pos_0_low <= 0 <= d_pos_0_high

    fwd_dom = (
        not missing_posneg
        and d_pos_neg_low > 0
        and _prob_ge(row, ["P_D_pos_neg_gt_0"], th.p_forward_gt_reverse_threshold)
    )
    rev_dom = (
        not missing_posneg
        and d_pos_neg_high < 0
        and _prob_le(row, ["P_D_pos_neg_gt_0"], 1.0 - th.p_forward_gt_reverse_threshold)
    )
    close_reverse = (not missing_posneg) and d_pos_neg_low <= 0 <= d_pos_neg_high

    if not yes_like:
        judgement = "not_supported"
        use_class = "not_candidate"
        guardrail = "No V1 temporal-eligibility support under the existing V1 classification."
    elif sensitive:
        judgement = "audit_sensitive"
        use_class = "sensitive_candidate_keep_separate"
        guardrail = "Tier3/sensitive V1 yes: keep as risk pool, not as a stable V1 core candidate."
    elif missing_pos0 or missing_posneg:
        judgement = "missing_stability_diagnostics"
        use_class = "needs_manual_check"
        guardrail = "Required D_pos_0 or D_pos_neg stability diagnostics were not found."
    elif core and lag_dom and fwd_dom:
        judgement = "stable_lag_dominant"
        use_class = "core_stable_lag_candidate"
        guardrail = "Positive-lag band is stably stronger than both lag0 and reverse. Still temporal eligibility, not causal proof."
    elif core and tau0_dom and fwd_dom:
        judgement = "stable_tau0_dominant_coupling"
        use_class = "same_day_dominant_not_clean_lag"
        guardrail = "Lag0 is stably stronger than the positive-lag band. Interpret as synchronous/rapid adjustment, not stable lead-lag."
    elif core and fwd_dom and (close_tau0 or same_day):
        judgement = "significant_lagged_but_tau0_coupled"
        use_class = "temporal_candidate_with_tau0_coupling"
        guardrail = "Forward direction is stable versus reverse, but positive lag is not stably stronger than lag0 or has same-day coupling."
    elif core and rev_dom:
        judgement = "reverse_competitive"
        use_class = "not_forward_stable"
        guardrail = "Reverse lag band is stably competitive or stronger; do not use this direction as a stable forward candidate."
    elif core and close_reverse:
        judgement = "forward_reverse_competitive_or_uncertain"
        use_class = "direction_uncertain"
        guardrail = "Forward-vs-reverse stability interval crosses zero; direction is not stable."
    elif core:
        judgement = "core_but_unclassified_stability_risk"
        use_class = "needs_manual_check"
        guardrail = "Core evidence tier, but stability diagnostics do not map cleanly to a predefined class."
    else:
        judgement = "not_supported"
        use_class = "not_candidate"
        guardrail = "Not in V1 core/sensitive candidate tiers."

    return {
        "evidence_tier_prefix": prefix,
        "lag_vs_tau0_label": (
            "stable_lag_dominant" if lag_dom else
            "stable_tau0_dominant" if tau0_dom else
            "lag_tau0_close_or_coupled" if close_tau0 or same_day else
            "missing" if missing_pos0 else
            "mixed_or_unstable"
        ),
        "forward_vs_reverse_label": (
            "stable_forward_over_reverse" if fwd_dom else
            "reverse_competitive" if rev_dom else
            "forward_reverse_close_or_uncertain" if close_reverse else
            "missing" if missing_posneg else
            "mixed_or_unstable"
        ),
        "same_day_coupling_detected": same_day,
        "v1_stability_judgement": judgement,
        "v1_stability_use_class": use_class,
        "v1_stability_interpretation_guardrail": guardrail,
        "D_pos_0_used": d_pos_0,
        f"D_pos_0_CI{ci}_low_used": d_pos_0_low,
        f"D_pos_0_CI{ci}_high_used": d_pos_0_high,
        "P_D_pos_0_gt_0_used": p_pos0_gt,
        "D_pos_neg_used": d_pos_neg,
        f"D_pos_neg_CI{ci}_low_used": d_pos_neg_low,
        f"D_pos_neg_CI{ci}_high_used": d_pos_neg_high,
        "P_D_pos_neg_gt_0_used": p_posneg_gt,
    }
