from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _tier_counts_v1(v1_tier: pd.DataFrame) -> pd.DataFrame:
    if v1_tier.empty:
        return pd.DataFrame()
    df = v1_tier.copy()
    if "family_direction" not in df.columns:
        df["family_direction"] = df["source_family"].astype(str) + "→" + df["target_family"].astype(str)
    df["is_v1_yes"] = df["lead_lag_group"].eq("lead_lag_yes")
    df["is_v1_tier1a"] = df["evidence_tier"].astype(str).str.startswith("Tier1a")
    df["is_v1_tier1b"] = df["evidence_tier"].astype(str).str.startswith("Tier1b")
    df["is_v1_tier2"] = df["evidence_tier"].astype(str).str.startswith("Tier2")
    df["is_v1_tier3"] = df["evidence_tier"].astype(str).str.startswith("Tier3")
    if "same_day_coupling_flag" not in df.columns:
        df["same_day_coupling_flag"] = False
    out = df.groupby(["window", "family_direction"], dropna=False).agg(
        v1_total_pairs=("source", "count"),
        v1_yes_count=("is_v1_yes", "sum"),
        v1_tier1a_count=("is_v1_tier1a", "sum"),
        v1_tier1b_count=("is_v1_tier1b", "sum"),
        v1_tier2_count=("is_v1_tier2", "sum"),
        v1_tier3_count=("is_v1_tier3", "sum"),
        v1_same_day_yes_count=("same_day_coupling_flag", "sum"),
        v1_max_positive_abs_r=("positive_peak_abs_r", "max"),
        v1_max_lag0_abs_r=("lag0_abs_r", "max"),
    ).reset_index()
    return out


def _interpretation_hint(row: pd.Series) -> str:
    v1_yes = int(row.get("v1_yes_count", 0) or 0)
    v1_tier1 = int(row.get("v1_tier1a_count", 0) or 0) + int(row.get("v1_tier1b_count", 0) or 0)
    v3_group = str(row.get("pc1_lead_lag_group", ""))
    v3_label = str(row.get("pc1_lead_lag_label", ""))
    v3_quality_s = str(row.get("source_quality_flag", ""))
    v3_quality_t = str(row.get("target_quality_flag", ""))
    quality_low = ("low_variance" in v3_quality_s) or ("low_variance" in v3_quality_t) or ("pc1_failed" in v3_quality_s) or ("pc1_failed" in v3_quality_t)
    tau0_strong = ("same_day" in v3_label) or bool(row.get("same_day_coupling_flag", False))
    if v1_yes == 0 and v3_group == "PC1_lead_lag_yes":
        return "index_weak_or_no_pc1_strong"
    if v1_tier1 > 0 and v3_group == "PC1_lead_lag_yes":
        return "index_strong_pc1_strong"
    if v1_yes > 0 and v3_group == "PC1_lead_lag_yes":
        return "index_yes_pc1_strong"
    if v1_yes > 0 and v3_group != "PC1_lead_lag_yes" and quality_low:
        return "index_yes_pc1_weak_but_pc1_quality_low"
    if v1_yes > 0 and v3_group != "PC1_lead_lag_yes" and tau0_strong:
        return "index_yes_pc1_lagged_weak_tau0_present"
    if v1_yes > 0 and v3_group != "PC1_lead_lag_yes":
        return "index_yes_pc1_weak"
    if v1_yes == 0 and v3_group != "PC1_lead_lag_yes" and quality_low:
        return "both_weak_pc1_quality_low"
    return "both_weak_or_no_signal"


def build_v1_v3_comparisons(v1_output_dir: Path, pc1_pair_summary: pd.DataFrame, pc1_quality: pd.DataFrame) -> dict[str, pd.DataFrame]:
    v1_tier = _safe_read_csv(v1_output_dir / "lead_lag_evidence_tier_summary.csv")
    v1_pair = _safe_read_csv(v1_output_dir / "lead_lag_pair_summary.csv")
    if v1_tier.empty and not v1_pair.empty:
        # Fallback if evidence tier is unavailable.
        v1_tier = v1_pair.copy()
        v1_tier["evidence_tier"] = "unknown"

    v1_counts = _tier_counts_v1(v1_tier)

    pc = pc1_pair_summary.copy()
    pc["family_direction"] = pc["source_family"].astype(str) + "→" + pc["target_family"].astype(str)
    # Attach source/target quality flags.
    q = pc1_quality[["window", "object", "pc1_explained_variance_ratio", "quality_flag", "sign_reference_corr_abs_after_flip"]].copy()
    q_src = q.rename(columns={
        "object": "source_family",
        "pc1_explained_variance_ratio": "source_pc1_explained_variance_ratio",
        "quality_flag": "source_quality_flag",
        "sign_reference_corr_abs_after_flip": "source_sign_reference_corr_abs",
    })
    q_tgt = q.rename(columns={
        "object": "target_family",
        "pc1_explained_variance_ratio": "target_pc1_explained_variance_ratio",
        "quality_flag": "target_quality_flag",
        "sign_reference_corr_abs_after_flip": "target_sign_reference_corr_abs",
    })
    pc = pc.merge(q_src, on=["window", "source_family"], how="left")
    pc = pc.merge(q_tgt, on=["window", "target_family"], how="left")

    keep_pc_cols = [
        "window", "family_direction", "source", "target", "source_family", "target_family",
        "pc1_lead_lag_label", "pc1_lead_lag_group", "pc1_evidence_tier",
        "positive_peak_lag", "positive_peak_signed_r", "positive_peak_abs_r",
        "lag0_signed_r", "lag0_abs_r", "same_day_coupling_flag",
        "p_pos_surrogate", "q_pos_within_window", "p_pos_audit_surrogate", "q_pos_audit_within_window",
        "source_pc1_explained_variance_ratio", "target_pc1_explained_variance_ratio",
        "source_quality_flag", "target_quality_flag",
        "source_sign_reference_corr_abs", "target_sign_reference_corr_abs",
    ]
    pc = pc[[c for c in keep_pc_cols if c in pc.columns]]
    comp = pc.merge(v1_counts, on=["window", "family_direction"], how="left")
    for col in ["v1_total_pairs", "v1_yes_count", "v1_tier1a_count", "v1_tier1b_count", "v1_tier2_count", "v1_tier3_count", "v1_same_day_yes_count"]:
        if col in comp.columns:
            comp[col] = comp[col].fillna(0).astype(int)
    comp["interpretation_hint"] = comp.apply(_interpretation_hint, axis=1)

    # T3/Meiyu-end focused table.
    priority_dirs = ["V→P", "H→P", "P→V", "P→H", "H→V", "V→H", "Jw→Je", "Je→H"]
    t3 = comp[(comp["window"].isin(["S3", "T3", "S4"])) & (comp["family_direction"].isin(priority_dirs))].copy()
    t3["meiyu_end_focus_note"] = np.where(
        t3["window"].eq("T3"),
        "T3/meiyu-ending primary audit window",
        "neighboring reference window",
    )

    return {
        "v1_index_vs_v3_pc1_family_comparison": comp,
        "t3_meiyu_end_pc1_audit": t3,
        "v1_family_counts_used_for_comparison": v1_counts,
    }
