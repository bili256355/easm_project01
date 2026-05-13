from __future__ import annotations

import numpy as np
import pandas as pd


def _best_pcmci_edge(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    df = edges.copy()
    df["abs_val_matrix"] = df["val_matrix"].abs()
    # Prefer supported links, then lower q, then stronger absolute value.
    df["_support_rank"] = np.where(df["pcmci_plus_supported"].astype(bool), 0, 1)
    df = df.sort_values(
        ["window", "source", "target", "_support_rank", "q_within_window", "abs_val_matrix"],
        ascending=[True, True, True, True, True, False],
    )
    best = df.groupby(["window", "source", "target"], as_index=False).head(1).copy()
    best = best.drop(columns=["_support_rank"])
    return best


def build_v1_overlap(edges: pd.DataFrame, v1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best = _best_pcmci_edge(edges)
    if best.empty:
        return pd.DataFrame(), pd.DataFrame()
    if v1.empty:
        out = best.copy()
        out["v1_available"] = False
        out["v1_lead_lag_yes"] = np.nan
        out["agreement_band"] = "v1_missing"
        return out, pd.DataFrame()

    keep = v1.copy()
    keep["v1_lead_lag_yes"] = keep["lead_lag_group"].astype(str).eq("lead_lag_yes")
    merged = best.merge(
        keep,
        on=["window", "source", "target", "source_family", "target_family"],
        how="left",
        suffixes=("_pcmci", "_v1"),
    )
    merged["v1_available"] = merged["lead_lag_group"].notna()
    merged["pcmci_plus_any_supported_1to5"] = merged["pcmci_plus_supported"].astype(bool)
    merged["pcmci_plus_supported_tau"] = np.where(
        merged["pcmci_plus_supported"].astype(bool), merged["tau"], np.nan
    )
    merged["exact_lag_agreement"] = (
        merged["v1_lead_lag_yes"].fillna(False).astype(bool)
        & merged["pcmci_plus_supported"].astype(bool)
        & (merged["positive_peak_lag"].astype("float") == merged["tau"].astype("float"))
    )
    merged["lag_band_agreement"] = (
        merged["v1_lead_lag_yes"].fillna(False).astype(bool)
        & merged["pcmci_plus_supported"].astype(bool)
    )

    def classify(row):
        v1_yes = bool(row.get("v1_lead_lag_yes", False))
        pc_yes = bool(row.get("pcmci_plus_supported", False))
        if v1_yes and pc_yes:
            return "V1_and_PCMCIplus"
        if v1_yes and not pc_yes:
            return "V1_only"
        if (not v1_yes) and pc_yes:
            return "PCMCIplus_only"
        return "neither_supported"

    merged["agreement_band"] = merged.apply(classify, axis=1)

    summary = (
        merged.groupby(["window", "agreement_band"], dropna=False)
        .size()
        .reset_index(name="n_pairs")
        .sort_values(["window", "agreement_band"])
    )
    return merged, summary
