#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V1 all-window positive_lag_not_supported decomposition audit.

Purpose
-------
Read V1 stability-judged lead-lag summary and decompose why each window has
``positive_lag_not_supported`` pairs, with all windows kept for comparison. This is a read-only diagnostic layer
inside V1/positive_lag_not_supported_decomposition_all_windows_a. It does not rerun lead-lag
screening and does not modify V1 outputs.

Default input
-------------
D:\\easm_project01\\lead_lag_screen\\V1\\outputs\\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\\tables\\lead_lag_pair_summary_stability_judged.csv

Default output
--------------
D:\\easm_project01\\lead_lag_screen\\V1\\positive_lag_not_supported_decomposition_all_windows_a\\outputs

Interpretation note
-------------------
This script localizes statistical/filtering reasons for rows already labelled
``positive_lag_not_supported``. Its labels are diagnostic buckets, not new
physical conclusions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WINDOWS_DEFAULT = ["S1", "T1", "S2", "T2", "S3", "T3", "S4", "T4", "S5"]
AUDIT_WINDOWS_DEFAULT = WINDOWS_DEFAULT
TARGET_REASON = "positive_lag_not_supported"
STRICT_LABEL = "stable_lag_dominant"
TAU0_LABEL = "significant_lagged_but_tau0_coupled"


@dataclass
class ColumnMap:
    window: str
    source_index: Optional[str]
    target_index: Optional[str]
    source_family: Optional[str]
    target_family: Optional[str]
    stability: Optional[str]
    evidence_tier: Optional[str]
    primary_failure_reason: Optional[str]
    lag_tau0: Optional[str]
    forward_reverse: Optional[str]


@dataclass
class Thresholds:
    weak_abs_r: float = 0.10
    moderate_abs_r: float = 0.20
    strong_abs_r: float = 0.30
    surrogate_alpha: float = 0.10
    fdr_alpha: float = 0.10
    audit_surrogate_alpha: float = 0.10
    near_surrogate_alpha: float = 0.15
    near_fdr_alpha: float = 0.20
    dominance_probability: float = 0.60


def norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def find_col(df: pd.DataFrame, candidates: Sequence[str], *, contains: bool = False) -> Optional[str]:
    norm_to_orig = {norm(c): c for c in df.columns}
    for cand in candidates:
        n = norm(cand)
        if n in norm_to_orig:
            return norm_to_orig[n]
    if contains:
        for cand in candidates:
            nc = norm(cand)
            for n, orig in norm_to_orig.items():
                if nc in n:
                    return orig
    return None


def detect_columns(df: pd.DataFrame) -> ColumnMap:
    window = find_col(df, ["window", "window_name", "target_window", "analysis_window"], contains=True)
    if window is None:
        raise ValueError("Cannot identify window column.")
    stability = find_col(df, ["v1_stability_judgement", "stability_judgement", "final_judgement", "classification"], contains=True)
    if stability is None:
        raise ValueError("Cannot identify v1 stability judgement column.")
    return ColumnMap(
        window=window,
        source_index=find_col(df, ["source_variable", "source_index", "source", "x_index", "driver_index"], contains=True),
        target_index=find_col(df, ["target_variable", "target_index", "target", "y_index", "response_index"], contains=True),
        source_family=find_col(df, ["source_family", "source_object", "source_group", "x_family"], contains=True),
        target_family=find_col(df, ["target_family", "target_object", "target_group", "y_family"], contains=True),
        stability=stability,
        evidence_tier=find_col(df, ["evidence_tier", "tier", "old_tier"], contains=True),
        primary_failure_reason=find_col(df, ["primary_failure_reason", "failure_reason", "reason"], contains=True),
        lag_tau0=find_col(df, ["lag_vs_tau0_label", "lag_tau0_label", "tau0_label"], contains=True),
        forward_reverse=find_col(df, ["forward_vs_reverse_label", "forward_reverse_label", "reverse_label"], contains=True),
    )


def infer_family(name: object) -> str:
    if pd.isna(name):
        return "UNKNOWN"
    s = str(name)
    for fam in ["Je", "Jw", "P", "V", "H"]:
        if s == fam or s.startswith(fam + "_") or s.startswith(fam + "-"):
            return fam
    ls = s.lower()
    if ls.startswith("precip") or ls.startswith("rain"):
        return "P"
    if ls.startswith("v850") or ls.startswith("v_"):
        return "V"
    if ls.startswith("z500") or ls.startswith("h_"):
        return "H"
    return "UNKNOWN"


def num(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return find_col(df, candidates, contains=False)


def add_derived_columns(df: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    out = df.copy()
    out[cm.window] = out[cm.window].astype(str)
    out["_stability"] = out[cm.stability].fillna("MISSING").astype(str)
    out["_is_strict_lag"] = out["_stability"].eq(STRICT_LABEL)
    out["_is_tau0_coupled"] = out["_stability"].eq(TAU0_LABEL)
    out["_is_usable_strict_tau0"] = out["_is_strict_lag"] | out["_is_tau0_coupled"]

    if cm.primary_failure_reason is None:
        out["_failure_reason"] = np.where(out["_is_usable_strict_tau0"], "USABLE", "UNKNOWN_NOT_USABLE")
    else:
        out["_failure_reason"] = out[cm.primary_failure_reason].fillna("NONE").astype(str)

    if cm.source_family is not None:
        out["_source_family"] = out[cm.source_family].fillna("UNKNOWN").astype(str)
    elif cm.source_index is not None:
        out["_source_family"] = out[cm.source_index].map(infer_family)
    else:
        out["_source_family"] = "UNKNOWN"

    if cm.target_family is not None:
        out["_target_family"] = out[cm.target_family].fillna("UNKNOWN").astype(str)
    elif cm.target_index is not None:
        out["_target_family"] = out[cm.target_index].map(infer_family)
    else:
        out["_target_family"] = "UNKNOWN"

    out["_family_direction"] = out["_source_family"].astype(str) + "→" + out["_target_family"].astype(str)

    if cm.source_index is not None and cm.target_index is not None:
        out["_pair_id"] = out[cm.source_index].astype(str) + " -> " + out[cm.target_index].astype(str)
    else:
        out["_pair_id"] = np.arange(len(out)).astype(str)

    if cm.evidence_tier is not None:
        out["_evidence_tier"] = out[cm.evidence_tier].fillna("MISSING").astype(str)
    else:
        out["_evidence_tier"] = "MISSING"

    if cm.lag_tau0 is not None:
        out["_lag_tau0_label"] = out[cm.lag_tau0].fillna("MISSING").astype(str)
    else:
        out["_lag_tau0_label"] = "MISSING"

    if cm.forward_reverse is not None:
        out["_forward_reverse_label"] = out[cm.forward_reverse].fillna("MISSING").astype(str)
    else:
        out["_forward_reverse_label"] = "MISSING"

    return out


def attach_metric_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    out = df.copy()
    metric_cols = {
        "positive_peak_abs_r": first_existing(out, ["positive_peak_abs_r", "positive_peak_corr_abs", "best_positive_abs_corr", "best_positive_corr_abs"]),
        "lag0_abs_r": first_existing(out, ["lag0_abs_r", "lag0_abs_corr", "lag0_corr_abs", "tau0_abs_corr"]),
        "negative_peak_abs_r": first_existing(out, ["negative_peak_abs_r", "negative_peak_corr_abs", "best_negative_abs_corr"]),
        "p_pos_surrogate": first_existing(out, ["p_pos_surrogate", "surrogate_p", "positive_surrogate_p"]),
        "q_pos_within_window": first_existing(out, ["q_pos_within_window", "fdr_q", "positive_fdr_q", "q_pos"]),
        "p_pos_audit_surrogate": first_existing(out, ["p_pos_audit_surrogate", "audit_surrogate_p", "positive_audit_surrogate_p"]),
        "D_pos_0": first_existing(out, ["D_pos_0", "lag_minus_tau0", "lag_minus_tau0_abs"]),
        "D_pos_neg": first_existing(out, ["D_pos_neg", "forward_minus_reverse"]),
        "P_D_pos_0_gt_0": first_existing(out, ["P_D_pos_0_gt_0_used", "P_D_pos_0_gt_0_robustness", "P_D_pos_0_gt_0"]),
        "P_D_pos_neg_gt_0": first_existing(out, ["P_D_pos_neg_gt_0_used", "P_D_pos_neg_gt_0_robustness", "P_D_pos_neg_gt_0"]),
        "positive_peak_lag": first_existing(out, ["positive_peak_lag", "best_positive_lag"]),
        "negative_peak_lag": first_existing(out, ["negative_peak_lag", "best_negative_lag"]),
        "positive_peak_signed_r": first_existing(out, ["positive_peak_signed_r", "best_positive_corr"]),
        "negative_peak_signed_r": first_existing(out, ["negative_peak_signed_r", "best_negative_corr"]),
    }
    for canonical, src in metric_cols.items():
        out[f"_m_{canonical}"] = num(out, src)
    return out, metric_cols


def yes_no_missing(series: pd.Series, *, op: str, threshold: float) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    res = pd.Series("missing", index=series.index, dtype=object)
    if op == ">=":
        res.loc[vals.notna() & (vals >= threshold)] = "yes"
        res.loc[vals.notna() & (vals < threshold)] = "no"
    elif op == "<=":
        res.loc[vals.notna() & (vals <= threshold)] = "yes"
        res.loc[vals.notna() & (vals > threshold)] = "no"
    elif op == ">":
        res.loc[vals.notna() & (vals > threshold)] = "yes"
        res.loc[vals.notna() & (vals <= threshold)] = "no"
    else:
        raise ValueError(op)
    return res


def decompose_positive_lag_not_supported(df: pd.DataFrame, cm: ColumnMap, thresholds: Thresholds, audit_windows: Sequence[str]) -> pd.DataFrame:
    sub = df[df[cm.window].isin(audit_windows) & df["_failure_reason"].eq(TARGET_REASON)].copy()

    # Primary signal/support flags.
    sub["effect_size_bin"] = pd.cut(
        sub["_m_positive_peak_abs_r"],
        bins=[-np.inf, thresholds.weak_abs_r, thresholds.moderate_abs_r, thresholds.strong_abs_r, np.inf],
        labels=["weak_abs_r", "moderate_abs_r", "strong_abs_r", "very_strong_abs_r"],
    ).astype(object).where(sub["_m_positive_peak_abs_r"].notna(), "missing_abs_r")

    sub["has_at_least_moderate_positive_peak"] = yes_no_missing(sub["_m_positive_peak_abs_r"], op=">=", threshold=thresholds.moderate_abs_r)
    sub["surrogate_pass"] = yes_no_missing(sub["_m_p_pos_surrogate"], op="<=", threshold=thresholds.surrogate_alpha)
    sub["fdr_pass"] = yes_no_missing(sub["_m_q_pos_within_window"], op="<=", threshold=thresholds.fdr_alpha)
    sub["audit_surrogate_pass"] = yes_no_missing(sub["_m_p_pos_audit_surrogate"], op="<=", threshold=thresholds.audit_surrogate_alpha)

    sub["lag_over_tau0_metric_positive"] = yes_no_missing(sub["_m_D_pos_0"], op=">", threshold=0.0)
    sub["lag_over_tau0_prob_supported"] = yes_no_missing(sub["_m_P_D_pos_0_gt_0"], op=">=", threshold=thresholds.dominance_probability)
    sub["forward_over_reverse_metric_positive"] = yes_no_missing(sub["_m_D_pos_neg"], op=">", threshold=0.0)
    sub["forward_over_reverse_prob_supported"] = yes_no_missing(sub["_m_P_D_pos_neg_gt_0"], op=">=", threshold=thresholds.dominance_probability)

    # Secondary dominance/competition class. This is not the original filter; it tells where a failed positive-lag pair would also have problems later.
    def secondary(row: pd.Series) -> str:
        tau0_bad = (row["lag_over_tau0_metric_positive"] == "no") or (row["lag_over_tau0_prob_supported"] == "no")
        rev_bad = (row["forward_over_reverse_metric_positive"] == "no") or (row["forward_over_reverse_prob_supported"] == "no")
        tau0_missing = (row["lag_over_tau0_metric_positive"] == "missing") and (row["lag_over_tau0_prob_supported"] == "missing")
        rev_missing = (row["forward_over_reverse_metric_positive"] == "missing") and (row["forward_over_reverse_prob_supported"] == "missing")
        if tau0_bad and rev_bad:
            return "would_later_face_tau0_and_reverse_competition"
        if tau0_bad:
            return "would_later_face_tau0_competition"
        if rev_bad:
            return "would_later_face_reverse_competition"
        if tau0_missing and rev_missing:
            return "dominance_metrics_missing"
        return "dominance_metrics_not_main_blocker"

    sub["secondary_dominance_issue"] = sub.apply(secondary, axis=1)

    def blocker(row: pd.Series) -> str:
        r = row["_m_positive_peak_abs_r"]
        p = row["_m_p_pos_surrogate"]
        q = row["_m_q_pos_within_window"]
        audit_p = row["_m_p_pos_audit_surrogate"]
        if pd.isna(r) and pd.isna(p) and pd.isna(q):
            return "missing_core_positive_lag_metrics"
        if pd.notna(r) and r < thresholds.weak_abs_r:
            return "weak_positive_lag_effect_size"
        # If raw surrogate passes but FDR fails, identify multiple-testing loss.
        if pd.notna(p) and p <= thresholds.surrogate_alpha and pd.notna(q) and q > thresholds.fdr_alpha:
            return "passes_surrogate_but_fails_fdr"
        if pd.notna(p) and p > thresholds.surrogate_alpha:
            return "fails_surrogate_support"
        if pd.isna(p) and pd.notna(q) and q > thresholds.fdr_alpha:
            return "fails_fdr_support"
        if pd.notna(audit_p) and audit_p > thresholds.audit_surrogate_alpha:
            return "fails_audit_surrogate_stability"
        if pd.notna(q) and q > thresholds.fdr_alpha:
            return "fails_fdr_support"
        if row["secondary_dominance_issue"] != "dominance_metrics_not_main_blocker":
            return "positive_lag_core_unclear_with_secondary_competition"
        return "positive_lag_not_supported_unresolved_or_threshold_mismatch"

    sub["dominant_positive_lag_blocker"] = sub.apply(blocker, axis=1)

    # Near-pass candidates: useful rows that failed the categorical reason but have nontrivial effect and near-threshold support.
    sub["near_positive_lag_pass_candidate"] = (
        (sub["_m_positive_peak_abs_r"] >= thresholds.moderate_abs_r)
        & (sub["_m_p_pos_surrogate"].fillna(1.0) <= thresholds.near_surrogate_alpha)
        & (sub["_m_q_pos_within_window"].fillna(1.0) <= thresholds.near_fdr_alpha)
    )

    sub["positive_lag_support_profile"] = np.select(
        [
            sub["near_positive_lag_pass_candidate"],
            (sub["_m_positive_peak_abs_r"] >= thresholds.moderate_abs_r) & (sub["_m_p_pos_surrogate"].fillna(1.0) <= thresholds.surrogate_alpha),
            (sub["_m_positive_peak_abs_r"] >= thresholds.moderate_abs_r),
            (sub["_m_positive_peak_abs_r"] < thresholds.weak_abs_r),
        ],
        [
            "near_pass_moderate_effect_and_near_fdr",
            "moderate_effect_surrogate_pass_fdr_failed_or_missing",
            "moderate_effect_but_support_failed",
            "weak_effect",
        ],
        default="low_or_missing_support_profile",
    )
    return sub


def count_table(df: pd.DataFrame, group_cols: Sequence[str], value_col: str, *, total_name: str = "count") -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(list(group_cols) + [value_col, total_name])
    out = df.groupby(list(group_cols) + [value_col], dropna=False).size().reset_index(name=total_name)
    return out.sort_values(list(group_cols) + [total_name], ascending=[True] * len(group_cols) + [False])


def make_window_blocker_summary(decomp: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if len(decomp) == 0:
        return pd.DataFrame()
    for w, sub in decomp.groupby(cm.window):
        total = len(sub)
        rows.append({
            "window": w,
            "positive_lag_not_supported_total": int(total),
            "weak_effect_size": int((sub["dominant_positive_lag_blocker"] == "weak_positive_lag_effect_size").sum()),
            "fails_surrogate_support": int((sub["dominant_positive_lag_blocker"] == "fails_surrogate_support").sum()),
            "passes_surrogate_but_fails_fdr": int((sub["dominant_positive_lag_blocker"] == "passes_surrogate_but_fails_fdr").sum()),
            "fails_fdr_support": int((sub["dominant_positive_lag_blocker"] == "fails_fdr_support").sum()),
            "fails_audit_surrogate_stability": int((sub["dominant_positive_lag_blocker"] == "fails_audit_surrogate_stability").sum()),
            "secondary_tau0_competition": int(sub["secondary_dominance_issue"].str.contains("tau0", na=False).sum()),
            "secondary_reverse_competition": int(sub["secondary_dominance_issue"].str.contains("reverse", na=False).sum()),
            "near_positive_lag_pass_candidates": int(sub["near_positive_lag_pass_candidate"].sum()),
            "moderate_or_stronger_effect": int((sub["_m_positive_peak_abs_r"] >= 0.20).sum()),
            "median_positive_peak_abs_r": float(sub["_m_positive_peak_abs_r"].median(skipna=True)),
            "median_p_pos_surrogate": float(sub["_m_p_pos_surrogate"].median(skipna=True)),
            "median_q_pos_within_window": float(sub["_m_q_pos_within_window"].median(skipna=True)),
        })
    return pd.DataFrame(rows)


def make_family_blocker_summary(decomp: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    if len(decomp) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for (w, fd), sub in decomp.groupby([cm.window, "_family_direction"], dropna=False):
        top_blockers = sub["dominant_positive_lag_blocker"].value_counts().to_dict()
        row: Dict[str, object] = {
            "window": w,
            "family_direction": fd,
            "n_positive_lag_not_supported": int(len(sub)),
            "n_near_positive_lag_pass_candidates": int(sub["near_positive_lag_pass_candidate"].sum()),
            "median_positive_peak_abs_r": float(sub["_m_positive_peak_abs_r"].median(skipna=True)),
            "median_p_pos_surrogate": float(sub["_m_p_pos_surrogate"].median(skipna=True)),
            "median_q_pos_within_window": float(sub["_m_q_pos_within_window"].median(skipna=True)),
        }
        for k, v in top_blockers.items():
            row[f"blocker__{norm(k)}"] = int(v)
        rows.append(row)
    out = pd.DataFrame(rows).fillna(0)
    return out.sort_values(["window", "n_positive_lag_not_supported"], ascending=[True, False])


def make_metric_bins(decomp: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    if len(decomp) == 0:
        return pd.DataFrame()
    rows = []
    for w, sub in decomp.groupby(cm.window):
        for col, label in [
            ("_m_positive_peak_abs_r", "positive_peak_abs_r"),
            ("_m_p_pos_surrogate", "p_pos_surrogate"),
            ("_m_q_pos_within_window", "q_pos_within_window"),
            ("_m_D_pos_0", "D_pos_0"),
            ("_m_D_pos_neg", "D_pos_neg"),
        ]:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().astype(float)
            if len(vals) == 0:
                continue
            rows.append({
                "window": w,
                "metric": label,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "p10": float(vals.quantile(0.10)),
                "p25": float(vals.quantile(0.25)),
                "p75": float(vals.quantile(0.75)),
                "p90": float(vals.quantile(0.90)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })
    return pd.DataFrame(rows)


def make_near_pass_candidates(decomp: pd.DataFrame, cm: ColumnMap, top_n: int = 200) -> pd.DataFrame:
    if len(decomp) == 0:
        return pd.DataFrame()
    cols = [
        cm.window, "_pair_id", "_source_family", "_target_family", "_family_direction",
        "_failure_reason", "_evidence_tier", "dominant_positive_lag_blocker", "positive_lag_support_profile",
        "near_positive_lag_pass_candidate", "effect_size_bin", "secondary_dominance_issue",
    ]
    if cm.source_index:
        cols.append(cm.source_index)
    if cm.target_index:
        cols.append(cm.target_index)
    metric_cols = [c for c in decomp.columns if c.startswith("_m_")]
    cols += metric_cols
    sub = decomp.copy()
    sub["_near_score"] = (
        pd.to_numeric(sub["_m_positive_peak_abs_r"], errors="coerce").fillna(0.0)
        - pd.to_numeric(sub["_m_q_pos_within_window"], errors="coerce").fillna(1.0) * 0.15
        - pd.to_numeric(sub["_m_p_pos_surrogate"], errors="coerce").fillna(1.0) * 0.10
    )
    sub = sub.sort_values(["near_positive_lag_pass_candidate", "_near_score"], ascending=[False, False]).head(top_n)
    return sub[[c for c in cols if c in sub.columns]]


def make_diagnosis(decomp: pd.DataFrame, window_summary: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, r in window_summary.iterrows():
        w = r["window"]
        total = int(r["positive_lag_not_supported_total"])
        weak = int(r.get("weak_effect_size", 0))
        fdr = int(r.get("passes_surrogate_but_fails_fdr", 0)) + int(r.get("fails_fdr_support", 0))
        surg = int(r.get("fails_surrogate_support", 0))
        near = int(r.get("near_positive_lag_pass_candidates", 0))
        mod = int(r.get("moderate_or_stronger_effect", 0))
        if total <= 0:
            support = "not_applicable"
            statement = f"No {TARGET_REASON} rows detected for {w}."
        elif weak / total >= 0.5:
            support = "weak_effect_dominant"
            statement = f"Most {w} {TARGET_REASON} rows have weak positive-lag effect size."
        elif (fdr + surg) / total >= 0.5:
            support = "statistical_support_dominant"
            statement = f"Most {w} {TARGET_REASON} rows have nontrivial positive-lag peaks but fail surrogate/FDR support."
        elif mod / total >= 0.5:
            support = "nonweak_effect_but_unstable_or_unclear"
            statement = f"Many {w} {TARGET_REASON} rows have moderate-or-stronger positive-lag peaks but do not pass support/clarity thresholds."
        else:
            support = "mixed"
            statement = f"{w} {TARGET_REASON} rows show mixed blockers; inspect blocker and metric tables."
        rows.append({
            "diagnosis_id": f"{w}_positive_lag_not_supported_decomposition",
            "support_level": support,
            "primary_evidence": (
                f"total={total}; weak_effect={weak}; surrogate_fail={surg}; fdr_related_fail={fdr}; "
                f"moderate_or_stronger_effect={mod}; near_pass={near}; "
                f"median_abs_r={r.get('median_positive_peak_abs_r', np.nan):.3f}; "
                f"median_p={r.get('median_p_pos_surrogate', np.nan):.3f}; "
                f"median_q={r.get('median_q_pos_within_window', np.nan):.3f}."
            ),
            "allowed_statement": statement,
            "forbidden_statement": (
                "Do not claim this diagnostic proves the physical cause. It only decomposes where rows labelled "
                f"{TARGET_REASON} fail statistically/diagnostically."
            ),
        })
    return pd.DataFrame(rows)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decompose positive_lag_not_supported rows for all windows in V1 stability results.")
    p.add_argument("--project-root", default=r"D:\easm_project01")
    p.add_argument("--input-csv", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--audit-windows", default=",".join(AUDIT_WINDOWS_DEFAULT))
    p.add_argument("--weak-abs-r", type=float, default=0.10)
    p.add_argument("--moderate-abs-r", type=float, default=0.20)
    p.add_argument("--strong-abs-r", type=float, default=0.30)
    p.add_argument("--surrogate-alpha", type=float, default=0.10)
    p.add_argument("--fdr-alpha", type=float, default=0.10)
    p.add_argument("--audit-surrogate-alpha", type=float, default=0.10)
    p.add_argument("--near-surrogate-alpha", type=float, default=0.15)
    p.add_argument("--near-fdr-alpha", type=float, default=0.20)
    p.add_argument("--dominance-probability", type=float, default=0.60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    audit_root = project_root / "lead_lag_screen" / "V1" / "positive_lag_not_supported_decomposition_all_windows_a"
    input_csv = Path(args.input_csv) if args.input_csv else (
        project_root / "lead_lag_screen" / "V1" / "outputs" / "lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b" / "tables" / "lead_lag_pair_summary_stability_judged.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (audit_root / "outputs")
    tables_dir = output_dir / "tables"
    summary_dir = output_dir / "summary"
    logs_dir = output_dir / "logs"
    for d in [tables_dir, summary_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    thresholds = Thresholds(
        weak_abs_r=args.weak_abs_r,
        moderate_abs_r=args.moderate_abs_r,
        strong_abs_r=args.strong_abs_r,
        surrogate_alpha=args.surrogate_alpha,
        fdr_alpha=args.fdr_alpha,
        audit_surrogate_alpha=args.audit_surrogate_alpha,
        near_surrogate_alpha=args.near_surrogate_alpha,
        near_fdr_alpha=args.near_fdr_alpha,
        dominance_probability=args.dominance_probability,
    )
    audit_windows = [w.strip() for w in str(args.audit_windows).split(",") if w.strip()]

    print(f"[positive_lag_not_supported decomposition] input: {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df0 = pd.read_csv(input_csv)
    cm = detect_columns(df0)
    df = add_derived_columns(df0, cm)
    df, metric_cols = attach_metric_columns(df)

    decomp = decompose_positive_lag_not_supported(df, cm, thresholds, audit_windows)
    decomp.to_csv(tables_dir / "positive_lag_not_supported_decomposition_long.csv", index=False, encoding="utf-8-sig")

    window_summary = make_window_blocker_summary(decomp, cm)
    window_summary.to_csv(tables_dir / "positive_lag_not_supported_decomposition_counts_by_window.csv", index=False, encoding="utf-8-sig")

    family_summary = make_family_blocker_summary(decomp, cm)
    family_summary.to_csv(tables_dir / "positive_lag_not_supported_decomposition_counts_by_family_direction.csv", index=False, encoding="utf-8-sig")

    blocker_counts = count_table(decomp, [cm.window], "dominant_positive_lag_blocker")
    blocker_counts.to_csv(tables_dir / "positive_lag_not_supported_blocker_counts.csv", index=False, encoding="utf-8-sig")

    profile_counts = count_table(decomp, [cm.window], "positive_lag_support_profile")
    profile_counts.to_csv(tables_dir / "positive_lag_support_profile_counts.csv", index=False, encoding="utf-8-sig")

    secondary_counts = count_table(decomp, [cm.window], "secondary_dominance_issue")
    secondary_counts.to_csv(tables_dir / "positive_lag_not_supported_secondary_dominance_counts.csv", index=False, encoding="utf-8-sig")

    metric_bins = make_metric_bins(decomp, cm)
    metric_bins.to_csv(tables_dir / "positive_lag_not_supported_metric_distribution_summary.csv", index=False, encoding="utf-8-sig")

    near_pass = make_near_pass_candidates(decomp, cm)
    near_pass.to_csv(tables_dir / "positive_lag_not_supported_near_pass_candidates.csv", index=False, encoding="utf-8-sig")

    diagnosis = make_diagnosis(decomp, window_summary, cm)
    diagnosis.to_csv(tables_dir / "positive_lag_not_supported_decomposition_diagnosis_table.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "audit": "v1_positive_lag_not_supported_decomposition_all_windows_a",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "audit_windows": audit_windows,
        "target_reason": TARGET_REASON,
        "n_rows_total": int(len(df)),
        "n_rows_decomposed": int(len(decomp)),
        "thresholds": thresholds.__dict__,
        "detected_columns": cm.__dict__,
        "metric_source_columns": metric_cols,
        "outputs": {
            "positive_lag_not_supported_decomposition_long": str(tables_dir / "positive_lag_not_supported_decomposition_long.csv"),
            "positive_lag_not_supported_decomposition_counts_by_window": str(tables_dir / "positive_lag_not_supported_decomposition_counts_by_window.csv"),
            "positive_lag_not_supported_decomposition_counts_by_family_direction": str(tables_dir / "positive_lag_not_supported_decomposition_counts_by_family_direction.csv"),
            "positive_lag_not_supported_blocker_counts": str(tables_dir / "positive_lag_not_supported_blocker_counts.csv"),
            "positive_lag_support_profile_counts": str(tables_dir / "positive_lag_support_profile_counts.csv"),
            "positive_lag_not_supported_secondary_dominance_counts": str(tables_dir / "positive_lag_not_supported_secondary_dominance_counts.csv"),
            "positive_lag_not_supported_metric_distribution_summary": str(tables_dir / "positive_lag_not_supported_metric_distribution_summary.csv"),
            "positive_lag_not_supported_near_pass_candidates": str(tables_dir / "positive_lag_not_supported_near_pass_candidates.csv"),
            "positive_lag_not_supported_decomposition_diagnosis_table": str(tables_dir / "positive_lag_not_supported_decomposition_diagnosis_table.csv"),
        },
    }
    write_json(summary_dir / "summary.json", summary)
    write_json(summary_dir / "run_meta.json", summary)

    log_lines = [
        "# V1 all-window positive_lag_not_supported decomposition audit log",
        "",
        f"Input: `{input_csv}`",
        f"Output: `{output_dir}`",
        "",
        "This audit decomposes rows already labelled `positive_lag_not_supported` for all windows, so T3/T4 can be compared against S/T windows under the same diagnostic buckets.",
        "It is a read-only statistical/diagnostic localization layer and does not prove the physical cause of window-level relationship drop.",
        "",
        "Default diagnostic thresholds:",
        f"- weak_abs_r < {thresholds.weak_abs_r}",
        f"- moderate_abs_r >= {thresholds.moderate_abs_r}",
        f"- surrogate_alpha = {thresholds.surrogate_alpha}",
        f"- fdr_alpha = {thresholds.fdr_alpha}",
        f"- audit_surrogate_alpha = {thresholds.audit_surrogate_alpha}",
        "",
        "Interpretation caution: blocker labels are derived diagnostic buckets, not original V1 decisions and not physical explanations.",
    ]
    (logs_dir / "RUN_LOG.md").write_text("\n".join(log_lines), encoding="utf-8")

    print(f"[positive_lag_not_supported decomposition] decomposed rows: {len(decomp)}")
    print(f"[positive_lag_not_supported decomposition] outputs: {output_dir}")


if __name__ == "__main__":
    main()
