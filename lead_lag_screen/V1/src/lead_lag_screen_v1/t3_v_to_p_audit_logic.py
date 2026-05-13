from __future__ import annotations

import math
import numpy as np
import pandas as pd


def safe_float(v, default=np.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def get_first(row: pd.Series, names: list[str], default=np.nan):
    lower = {str(k).lower(): k for k in row.index}
    for name in names:
        if name in row.index:
            return row[name]
        key = lower.get(name.lower())
        if key is not None:
            return row[key]
    return default


def evidence_prefix(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split("_", 1)[0]


def is_v1_candidate(row: pd.Series) -> bool:
    judgement = str(get_first(row, ["v1_stability_judgement"], "")).strip()
    if judgement in {
        "stable_lag_dominant",
        "significant_lagged_but_tau0_coupled",
        "stable_tau0_dominant_coupling",
        "audit_sensitive",
    }:
        return True
    tier = evidence_prefix(get_first(row, ["evidence_tier"], ""))
    if tier.startswith("Tier1") or tier in {"Tier2", "Tier3"}:
        return True
    label = str(get_first(row, ["lead_lag_label", "lead_lag_group"], "")).lower()
    return "lead_lag_yes" in label


def primary_failure_stage(row: pd.Series) -> tuple[str, str]:
    """Return (stage, human-readable reason) for why a V->P pair did or did not survive.

    The classification is intentionally diagnostic, not a new scientific judgement.
    It traces the row through the existing V1 outputs and the V1 stability judgement layer.
    """
    judgement = str(get_first(row, ["v1_stability_judgement"], "")).strip()
    if judgement == "stable_lag_dominant":
        return "passed_stable_lag", "Entered stable_lag_dominant: positive lag stably exceeds lag0 and reverse."
    if judgement == "significant_lagged_but_tau0_coupled":
        return "passed_tau0_coupled", "Entered temporal candidate pool, but lag0 is close/competitive."
    if judgement == "stable_tau0_dominant_coupling":
        return "tau0_dominant", "Lag0 is stably stronger than positive lag; interpret as synchronous/rapid adjustment."
    if judgement == "audit_sensitive":
        return "audit_sensitive_candidate", "Original V1 candidate is sensitive under audit/persistence checks."

    label = str(get_first(row, ["lead_lag_label", "lead_lag_group"], "")).lower()
    direction_label = str(get_first(row, ["direction_label"], "")).lower()
    tier = evidence_prefix(get_first(row, ["evidence_tier"], ""))

    p_pos = safe_float(get_first(row, ["p_pos_surrogate"]))
    q_pos = safe_float(get_first(row, ["q_pos_within_window"]))
    p_audit = safe_float(get_first(row, ["p_pos_audit_surrogate"]))
    q_audit = safe_float(get_first(row, ["q_pos_audit_within_window"]))
    d_pos_neg_low = safe_float(get_first(row, ["D_pos_neg_CI90_low"]))
    d_pos_neg_high = safe_float(get_first(row, ["D_pos_neg_CI90_high"]))
    d_pos_0_low = safe_float(get_first(row, ["D_pos_0_CI90_low"]))
    d_pos_0_high = safe_float(get_first(row, ["D_pos_0_CI90_high"]))
    same_day = safe_bool(get_first(row, ["same_day_coupling_flag", "same_day_coupling_detected"], False))

    t_pos = safe_float(get_first(row, ["T_pos_obs", "positive_peak_abs_r"]))
    null95 = safe_float(get_first(row, ["T_pos_null95"]))
    margin = t_pos - null95 if not (math.isnan(t_pos) or math.isnan(null95)) else np.nan

    if "reverse" in label or "reverse" in direction_label:
        return "reverse_competitive", "Reverse lag band is stronger or original direction is reverse-dominant."

    if not math.isnan(p_pos) and p_pos > 0.05:
        if not math.isnan(margin) and margin <= 0:
            return "positive_not_surrogate_significant", "Positive-lag max-stat does not exceed the AR(1) surrogate null95."
        return "positive_not_surrogate_significant", "Positive-lag max-stat is not significant under the main AR(1) surrogate null."

    if not math.isnan(q_pos) and q_pos > 0.10 and not is_v1_candidate(row):
        return "positive_fdr_not_supported", "Positive-lag signal exists but does not pass within-window FDR for V1 candidate status."

    if (not math.isnan(p_audit) and p_audit > 0.05) or (not math.isnan(q_audit) and q_audit > 0.10 and tier.startswith("Tier3")):
        return "audit_not_stable", "Main positive-lag support weakens under audit surrogate/null criteria."

    if not math.isnan(d_pos_neg_high) and d_pos_neg_high < 0:
        return "reverse_competitive", "Reverse lag band is stably stronger than positive lag."
    if not math.isnan(d_pos_neg_low) and d_pos_neg_low <= 0 <= d_pos_neg_high:
        return "direction_uncertain", "Forward-vs-reverse interval crosses zero; direction is not stable."

    if not math.isnan(d_pos_0_high) and d_pos_0_high < 0:
        return "tau0_competitive", "Lag0 is stably stronger than positive lag."
    if same_day or (not math.isnan(d_pos_0_low) and d_pos_0_low <= 0 <= d_pos_0_high):
        return "tau0_competitive", "Lag0 is close to or competitive with positive lag."

    if not is_v1_candidate(row):
        return "not_candidate_other", "Did not enter existing V1 candidate pool for reasons not isolated by available diagnostics."

    return "unclassified_manual_review", "Diagnostics do not map cleanly to a predefined audit stage."


def lag_profile_type(summary_row: pd.Series) -> str:
    t_pos = safe_float(get_first(summary_row, ["max_positive_abs_r", "positive_peak_abs_r"]))
    t0 = safe_float(get_first(summary_row, ["lag0_abs_r"]))
    tneg = safe_float(get_first(summary_row, ["max_negative_abs_r", "negative_peak_abs_r"]))
    vals = [v for v in [t_pos, t0, tneg] if not math.isnan(v)]
    if not vals or max(vals) < 0.20:
        return "weak_all_lags"
    mx = max(vals)
    near = 0.03
    if not math.isnan(t0) and abs(t0 - mx) <= near and (math.isnan(t_pos) or abs(t_pos - t0) <= near):
        return "flat_lag0_positive_close"
    if not math.isnan(t0) and t0 == mx:
        return "lag0_peak"
    if not math.isnan(t_pos) and t_pos == mx and (math.isnan(tneg) or t_pos - tneg > near) and (math.isnan(t0) or t_pos - t0 > near):
        return "positive_peak_clear"
    if not math.isnan(tneg) and tneg == mx:
        return "reverse_peak"
    return "multi_peak_or_close"
