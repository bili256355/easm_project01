from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _safe_read(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _v1_rollup(v1_output_dir: Path) -> pd.DataFrame:
    pair_path = v1_output_dir / "lead_lag_pair_summary.csv"
    pair = _safe_read(pair_path)
    if pair is None or pair.empty:
        return pd.DataFrame()
    family_col = "family_direction" if "family_direction" in pair.columns else None
    if family_col is None:
        if {"source_family", "target_family"}.issubset(pair.columns):
            pair["family_direction"] = pair["source_family"].astype(str) + "→" + pair["target_family"].astype(str)
        elif {"source_object", "target_object"}.issubset(pair.columns):
            pair["family_direction"] = pair["source_object"].astype(str) + "→" + pair["target_object"].astype(str)
        else:
            return pd.DataFrame()
    label_col = "lead_lag_label" if "lead_lag_label" in pair.columns else None
    tier_col = "evidence_tier" if "evidence_tier" in pair.columns else None
    same_col = "same_day_coupling_flag" if "same_day_coupling_flag" in pair.columns else None
    rows = []
    for (window, fd), sub in pair.groupby(["window", "family_direction"], sort=False):
        labels = sub[label_col].astype(str) if label_col else pd.Series([], dtype=str)
        tiers = sub[tier_col].astype(str) if tier_col else pd.Series([], dtype=str)
        rows.append({
            "window": window,
            "family_direction": fd,
            "v1_total_pairs": int(len(sub)),
            "v1_yes_count": int(labels.str.contains("lead_lag_yes", na=False).sum()) if label_col else np.nan,
            "v1_tier1a_count": int((tiers == "Tier1a_clear_lead_lag").sum()) if tier_col else np.nan,
            "v1_tier1b_count": int((tiers == "Tier1b_stable_with_same_day").sum()) if tier_col else np.nan,
            "v1_tier2_count": int((tiers == "Tier2_audit_moderate").sum()) if tier_col else np.nan,
            "v1_tier3_count": int((tiers == "Tier3_surrogate_sensitive_yes").sum()) if tier_col else np.nan,
            "v1_same_day_flag_count": int(sub[same_col].astype(bool).sum()) if same_col else np.nan,
        })
    return pd.DataFrame(rows)


def _v3_rollup(v3_output_dir: Path) -> pd.DataFrame:
    pair_path = v3_output_dir / "eof_pc1_pair_summary.csv"
    pair = _safe_read(pair_path)
    if pair is None or pair.empty:
        return pd.DataFrame()
    out = pair.copy()
    if "family_direction" not in out.columns and {"source_object", "target_object"}.issubset(out.columns):
        out["family_direction"] = out["source_object"].astype(str) + "→" + out["target_object"].astype(str)
    needed = [
        "window", "family_direction", "pc1_lead_lag_label", "pc1_evidence_tier",
        "positive_peak_lag", "positive_peak_r", "lag0_r", "same_day_flag",
    ]
    cols = [c for c in needed if c in out.columns]
    out = out[cols].copy()
    rename = {
        "pc1_lead_lag_label": "v3_pc1_label",
        "pc1_evidence_tier": "v3_pc1_tier",
        "positive_peak_lag": "v3_positive_peak_lag",
        "positive_peak_r": "v3_positive_peak_r",
        "lag0_r": "v3_lag0_r",
        "same_day_flag": "v3_same_day_flag",
    }
    return out.rename(columns=rename)


def build_v1_v3_v4_comparison(v1_output_dir: Path, v3_output_dir: Path, cca_summary: pd.DataFrame) -> pd.DataFrame:
    v4 = cca_summary.copy()
    v4 = v4.rename(columns={
        "pair_direction": "family_direction",
        "best_lag_all": "v4_best_lag_all",
        "best_abs_cv_r_all": "v4_best_abs_cv_r_all",
        "best_lagged_lag": "v4_best_lagged_lag",
        "best_lagged_abs_cv_r": "v4_best_lagged_abs_cv_r",
        "tau0_abs_cv_r": "v4_tau0_abs_cv_r",
        "cca_time_structure_label": "v4_cca_time_structure_label",
        "perm_p_max_train_r": "v4_perm_p_max_train_r",
        "cca_evidence_tier": "v4_cca_evidence_tier",
    })
    v1 = _v1_rollup(v1_output_dir)
    v3 = _v3_rollup(v3_output_dir)
    merged = v4.merge(v1, on=["window", "family_direction"], how="left")
    if not v3.empty:
        merged = merged.merge(v3, on=["window", "family_direction"], how="left")
    else:
        merged["v3_pc1_label"] = np.nan

    hints = []
    for row in merged.itertuples(index=False):
        v1_yes = getattr(row, "v1_yes_count", np.nan)
        v4_tau = getattr(row, "v4_tau0_abs_cv_r", np.nan)
        v4_lag = getattr(row, "v4_best_lagged_abs_cv_r", np.nan)
        time_label = getattr(row, "v4_cca_time_structure_label", "")
        if pd.notna(v1_yes) and v1_yes > 0 and pd.notna(v4_tau) and max(v4_tau, v4_lag if pd.notna(v4_lag) else -np.inf) >= 0.3:
            if "tau0" in str(time_label):
                hints.append("v1_signal_cca_tau0_coupling")
            elif "lagged" in str(time_label):
                hints.append("v1_signal_cca_lagged_coupling")
            else:
                hints.append("v1_signal_cca_coupling")
        elif pd.notna(v1_yes) and v1_yes > 0:
            hints.append("v1_signal_cca_weak_or_unstable")
        elif pd.notna(v4_tau) and max(v4_tau, v4_lag if pd.notna(v4_lag) else -np.inf) >= 0.3:
            hints.append("cca_only_field_coupling_candidate")
        else:
            hints.append("both_weak_or_not_available")
    merged["interpretation_hint"] = hints
    return merged


def build_t3_cca_audit(comparison: pd.DataFrame) -> pd.DataFrame:
    t3 = comparison[comparison["window"] == "T3"].copy()
    priority = {"V→P": 0, "H→P": 1, "H→V": 2, "Jw→Je": 3, "Je→H": 4}
    t3["priority_order"] = t3["family_direction"].map(priority).fillna(99)
    return t3.sort_values(["k_eof", "priority_order", "family_direction"]).drop(columns=["priority_order"])
