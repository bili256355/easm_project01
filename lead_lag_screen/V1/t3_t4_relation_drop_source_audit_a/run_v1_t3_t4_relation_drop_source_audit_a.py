#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V1 T3/T4 relation drop source audit.

Purpose
-------
Read the V1 post-processed stability judgement table and audit where the
T3/T4 relationship-count collapse occurs. This script DOES NOT rerun lead-lag
screening and DOES NOT modify V1 results. It only reads existing V1 outputs and
writes an independent audit output folder inside this audit directory.

Default input
-------------
D:\\easm_project01\\lead_lag_screen\\V1\\outputs\\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\\tables\\lead_lag_pair_summary_stability_judged.csv

Default output
--------------
D:\\easm_project01\\lead_lag_screen\\V1\\t3_t4_relation_drop_source_audit_a\\outputs
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
CHAIN_WINDOWS = ["S3", "T3", "S4", "T4", "S5"]
TARGET_WINDOWS = ["T3", "T4"]
STRICT_LABEL = "stable_lag_dominant"
TAU0_LABEL = "significant_lagged_but_tau0_coupled"
AUDIT_SENSITIVE_LABEL = "audit_sensitive"

OLD_TIER1_PREFIXES = ("Tier1a", "Tier1b")
OLD_TIER2_PREFIXES = ("Tier2",)

FAILURE_REASONS_OF_INTEREST = [
    "positive_lag_not_supported",
    "marginal_statistical_support",
    "same_day_only_without_positive_lag_support",
    "positive_and_negative_supported_but_direction_difference_uncertain",
    "reverse_lag_dominant",
]

EFFECT_COL_CANDIDATES = [
    "positive_peak_abs_r",
    "positive_peak_corr_abs",
    "best_positive_abs_corr",
    "best_positive_corr_abs",
    "best_positive_corr",
    "positive_peak_r2",
    "best_positive_r2",
    "lag0_abs_r",
    "lag0_abs_corr",
    "lag0_corr_abs",
    "lag0_corr",
    "tau0_abs_corr",
    "negative_peak_abs_r",
    "negative_peak_corr_abs",
    "best_negative_abs_corr",
    "D_pos_0",
    "D_pos_neg",
    "lag_minus_tau0",
    "lag_minus_tau0_abs",
    "forward_minus_reverse",
    "surrogate_p",
    "p_pos_surrogate",
    "q_pos_within_window",
    "fdr_q",
]


@dataclass
class ColumnMap:
    window: str
    source_index: Optional[str]
    target_index: Optional[str]
    source_family: Optional[str]
    target_family: Optional[str]
    stability: Optional[str]
    evidence_tier: Optional[str]
    lag_tau0: Optional[str]
    forward_reverse: Optional[str]
    primary_failure_reason: Optional[str]
    lead_lag_label: Optional[str]


def norm(s: str) -> str:
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
        raise ValueError("Cannot identify the window column. Expected a column like 'window'.")
    return ColumnMap(
        window=window,
        source_index=find_col(df, ["source_index", "source", "source_name", "x_index", "driver_index"], contains=True),
        target_index=find_col(df, ["target_index", "target", "target_name", "y_index", "response_index"], contains=True),
        source_family=find_col(df, ["source_family", "source_object", "source_group", "x_family"], contains=True),
        target_family=find_col(df, ["target_family", "target_object", "target_group", "y_family"], contains=True),
        stability=find_col(df, ["v1_stability_judgement", "stability_judgement", "final_judgement", "classification"], contains=True),
        evidence_tier=find_col(df, ["evidence_tier", "tier", "old_tier"], contains=True),
        lag_tau0=find_col(df, ["lag_vs_tau0_label", "lag_tau0_label", "tau0_label"], contains=True),
        forward_reverse=find_col(df, ["forward_vs_reverse_label", "forward_reverse_label", "reverse_label"], contains=True),
        primary_failure_reason=find_col(df, ["primary_failure_reason", "failure_reason", "reason"], contains=True),
        lead_lag_label=find_col(df, ["lead_lag_label", "lead_lag_support", "lead_lag_judgement", "lead_lag_yes"], contains=True),
    )


def infer_family(name: object) -> str:
    if pd.isna(name):
        return "UNKNOWN"
    s = str(name)
    for fam in ["Je", "Jw", "P", "V", "H"]:
        if s == fam or s.startswith(fam + "_") or s.startswith(fam + "-"):
            return fam
    # common lower-case fallback
    ls = s.lower()
    if ls.startswith("precip") or ls.startswith("rain"):
        return "P"
    if ls.startswith("v850") or ls.startswith("v_"):
        return "V"
    if ls.startswith("z500") or ls.startswith("h_"):
        return "H"
    return "UNKNOWN"


def add_derived_columns(df: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    out = df.copy()
    if cm.stability is None:
        raise ValueError("Cannot identify stability judgement column. Expected 'v1_stability_judgement'.")
    out["_stability"] = out[cm.stability].astype(str)
    out["_is_strict_lag"] = out["_stability"].eq(STRICT_LABEL)
    out["_is_tau0_coupled"] = out["_stability"].eq(TAU0_LABEL)
    out["_is_audit_sensitive"] = out["_stability"].eq(AUDIT_SENSITIVE_LABEL)
    out["_is_usable_strict_tau0"] = out["_is_strict_lag"] | out["_is_tau0_coupled"]
    out["_is_usable_with_audit_sensitive"] = out["_is_usable_strict_tau0"] | out["_is_audit_sensitive"]

    if cm.evidence_tier is not None:
        tier = out[cm.evidence_tier].fillna("").astype(str)
        out["_tier1"] = tier.str.startswith(OLD_TIER1_PREFIXES)
        out["_tier2"] = tier.str.startswith(OLD_TIER2_PREFIXES)
        out["_evidence_tier_norm"] = tier
    else:
        out["_tier1"] = False
        out["_tier2"] = False
        out["_evidence_tier_norm"] = "MISSING_EVIDENCE_TIER"

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

    if cm.primary_failure_reason is not None:
        out["_failure_reason"] = out[cm.primary_failure_reason].fillna("NONE").astype(str)
    else:
        out["_failure_reason"] = np.where(out["_is_usable_strict_tau0"], "USABLE", "UNKNOWN_NOT_USABLE")

    if cm.lag_tau0 is not None:
        out["_lag_tau0_label"] = out[cm.lag_tau0].fillna("MISSING").astype(str)
    else:
        out["_lag_tau0_label"] = "MISSING"

    if cm.forward_reverse is not None:
        out["_forward_reverse_label"] = out[cm.forward_reverse].fillna("MISSING").astype(str)
    else:
        out["_forward_reverse_label"] = "MISSING"

    if cm.lead_lag_label is not None:
        label = out[cm.lead_lag_label].fillna("").astype(str).str.lower()
        # conservative: count explicit yes/supported only; avoid counting "not_supported".
        out["_lead_lag_yes"] = label.str.contains("yes|supported|lead_lag", regex=True) & ~label.str.contains("not|no|failed", regex=True)
    else:
        # Fallback: final usable or old Tier1/Tier2 usually means lead-lag support survived.
        out["_lead_lag_yes"] = out["_is_usable_with_audit_sensitive"] | out["_tier1"] | out["_tier2"]

    return out


def ensure_windows(df: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    out = df.copy()
    out[cm.window] = out[cm.window].astype(str)
    return out


def value_counts_prefixed(series: pd.Series, prefix: str) -> Dict[str, int]:
    vc = series.fillna("MISSING").astype(str).value_counts(dropna=False)
    return {f"{prefix}{norm(k)}": int(v) for k, v in vc.items()}


def make_gate_funnel(df: pd.DataFrame, cm: ColumnMap, windows: Sequence[str]) -> pd.DataFrame:
    rows = []
    for w in windows:
        sub = df[df[cm.window] == w]
        row: Dict[str, object] = {
            "window": w,
            "total_pairs": int(len(sub)),
            "lead_lag_yes": int(sub["_lead_lag_yes"].sum()),
            "tier1_count": int(sub["_tier1"].sum()),
            "tier2_count": int(sub["_tier2"].sum()),
            "strict_lag": int(sub["_is_strict_lag"].sum()),
            "tau0_coupled": int(sub["_is_tau0_coupled"].sum()),
            "strict_plus_tau0": int(sub["_is_usable_strict_tau0"].sum()),
            "audit_sensitive": int(sub["_is_audit_sensitive"].sum()),
            "usable_with_audit_sensitive": int(sub["_is_usable_with_audit_sensitive"].sum()),
        }
        for reason in FAILURE_REASONS_OF_INTEREST:
            row[f"reason__{reason}"] = int((sub["_failure_reason"] == reason).sum())
        if cm.lag_tau0 is not None:
            row.update(value_counts_prefixed(sub["_lag_tau0_label"], "lag_tau0__"))
        if cm.forward_reverse is not None:
            row.update(value_counts_prefixed(sub["_forward_reverse_label"], "forward_reverse__"))
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def make_family_direction_gap(df: pd.DataFrame, cm: ColumnMap, windows: Sequence[str]) -> pd.DataFrame:
    counts = (
        df.groupby([cm.window, "_family_direction"], dropna=False)["_is_usable_strict_tau0"]
        .sum()
        .reset_index(name="usable_strict_tau0")
    )
    all_dirs = sorted(df["_family_direction"].unique())
    rows = []
    for target_w in TARGET_WINDOWS:
        if target_w == "T3":
            neighbors = ["S3", "S4"]
        elif target_w == "T4":
            neighbors = ["S4", "S5"]
        else:
            neighbors = []
        other_windows = [w for w in windows if w != target_w]
        for d in all_dirs:
            def get_count(w: str) -> int:
                m = counts[(counts[cm.window] == w) & (counts["_family_direction"] == d)]
                return int(m["usable_strict_tau0"].iloc[0]) if len(m) else 0
            target_count = get_count(target_w)
            neighbor_vals = [get_count(w) for w in neighbors]
            other_vals = [get_count(w) for w in other_windows]
            neighbor_mean = float(np.mean(neighbor_vals)) if neighbor_vals else np.nan
            other_mean = float(np.mean(other_vals)) if other_vals else np.nan
            rows.append({
                "target_window": target_w,
                "family_direction": d,
                "target_usable_strict_tau0": target_count,
                "neighbor_windows": ",".join(neighbors),
                "neighbor_mean_usable": neighbor_mean,
                "gap_to_neighbor_mean": target_count - neighbor_mean if not np.isnan(neighbor_mean) else np.nan,
                "other_window_mean_usable": other_mean,
                "gap_to_other_window_mean": target_count - other_mean if not np.isnan(other_mean) else np.nan,
            })
    out = pd.DataFrame(rows)
    # Contribution among negative gaps per target window.
    frames = []
    for w, sub in out.groupby("target_window"):
        neg = sub["gap_to_other_window_mean"].clip(upper=0).abs()
        total_neg = float(neg.sum())
        part = sub.copy()
        part["negative_gap_contribution_percent"] = np.where(total_neg > 0, neg / total_neg * 100.0, 0.0)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def make_pair_survival_chain(df: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    base_cols = ["_pair_id", "_source_family", "_target_family", "_family_direction"]
    if cm.source_index:
        base_cols.append(cm.source_index)
    if cm.target_index:
        base_cols.append(cm.target_index)
    meta = df[base_cols].drop_duplicates("_pair_id")
    status_frames = []
    for w in CHAIN_WINDOWS:
        sub = df[df[cm.window] == w].copy()
        sub[f"{w}_status"] = np.select(
            [sub["_is_strict_lag"], sub["_is_tau0_coupled"], sub["_is_audit_sensitive"], sub["_tier1"], sub["_tier2"]],
            ["strict_lag", "tau0_coupled", "audit_sensitive", "tier1_not_final", "tier2_not_final"],
            default="not_supported",
        )
        sub[f"{w}_failure_reason"] = sub["_failure_reason"]
        status_frames.append(sub[["_pair_id", f"{w}_status", f"{w}_failure_reason"]])
    out = meta.copy()
    for sf in status_frames:
        out = out.merge(sf, on="_pair_id", how="left")
    for w in CHAIN_WINDOWS:
        out[f"{w}_status"] = out[f"{w}_status"].fillna("missing_pair")
        out[f"{w}_failure_reason"] = out[f"{w}_failure_reason"].fillna("missing_pair")

    def usable(status: object) -> bool:
        return str(status) in {"strict_lag", "tau0_coupled"}

    labels = []
    for _, r in out.iterrows():
        s3, t3, s4, t4, s5 = [usable(r[f"{w}_status"]) for w in CHAIN_WINDOWS]
        if all([s3, t3, s4, t4, s5]):
            lab = "persistent_S3_to_S5"
        elif s3 and not t3:
            lab = "S3_lost_at_T3"
        elif (not s3) and t3:
            lab = "T3_emerged_or_recovered"
        elif s4 and not t4:
            lab = "S4_lost_at_T4"
        elif (not s4) and t4:
            lab = "T4_emerged_or_recovered"
        elif t3 and not t4:
            lab = "T3_lost_at_T4"
        elif (not t3) and t4:
            lab = "T4_recovered_after_T3_absent"
        elif not any([s3, t3, s4, t4, s5]):
            lab = "absent_all_chain_windows"
        else:
            lab = "mixed_chain"
        labels.append(lab)
    out["survival_chain_label"] = labels
    return out


def existing_effect_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    normalized = {norm(c): c for c in df.columns}
    for cand in EFFECT_COL_CANDIDATES:
        n = norm(cand)
        if n in normalized:
            c = normalized[n]
            # Boolean flag columns may be treated as numeric by pandas/numpy,
            # but they are not effect-size metrics and can break quantile.
            if not pd.api.types.is_bool_dtype(df[c]):
                cols.append(c)
    # Also include numeric columns that look like effect/stat columns.
    keywords = ["corr", "r2", "surrogate", "fdr", "lag_minus", "d_pos", "peak", "tau0"]
    for c in df.columns:
        if c in cols:
            continue
        nc = norm(c)
        if (
            any(k in nc for k in keywords)
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
        ):
            cols.append(c)
    return cols


def numeric_effect_values(series: pd.Series) -> pd.Series:
    """Return finite float values for effect-size summaries.

    Boolean flag columns are intentionally excluded. On recent numpy/pandas,
    quantile interpolation on bool arrays can raise ``TypeError: numpy boolean
    subtract``; flags also should not enter effect-size distribution tables.
    """
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(dtype=float)
    vals = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_bool_dtype(vals):
        return pd.Series(dtype=float)
    vals = vals.dropna()
    if len(vals) == 0:
        return pd.Series(dtype=float)
    return vals.astype(float)

def classify_near_miss(reason: str) -> str:
    r = str(reason)
    if "same_day" in r or "tau0" in r:
        return "strong_tau0_or_same_day_but_no_positive_lag"
    if "reverse" in r:
        return "reverse_or_negative_lag_competitive"
    if "direction" in r:
        return "direction_uncertain"
    if "marginal" in r:
        return "marginal_statistical_support"
    if "positive_lag_not_supported" in r:
        return "positive_lag_not_supported"
    return "other_not_usable"


def make_near_miss(df: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    eff_cols = existing_effect_cols(df)
    keep_cols = [cm.window, "_pair_id", "_source_family", "_target_family", "_family_direction", "_failure_reason", "_stability", "_evidence_tier_norm"]
    if cm.source_index:
        keep_cols.append(cm.source_index)
    if cm.target_index:
        keep_cols.append(cm.target_index)
    keep_cols += [c for c in eff_cols if c not in keep_cols]
    sub = df[df[cm.window].isin(TARGET_WINDOWS) & (~df["_is_usable_strict_tau0"])].copy()
    sub["near_miss_class"] = sub["_failure_reason"].map(classify_near_miss)
    cols = keep_cols + ["near_miss_class"]
    return sub[[c for c in cols if c in sub.columns]].sort_values([cm.window, "near_miss_class", "_family_direction", "_pair_id"])


def make_effect_distribution(df: pd.DataFrame, cm: ColumnMap, windows: Sequence[str]) -> pd.DataFrame:
    eff_cols = existing_effect_cols(df)
    rows = []
    for w in windows:
        sub = df[df[cm.window] == w]
        for col in eff_cols:
            vals = numeric_effect_values(sub[col])
            if len(vals) == 0:
                continue
            rows.append({
                "window": w,
                "metric": col,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "p25": float(vals.quantile(0.25)),
                "p75": float(vals.quantile(0.75)),
                "p90": float(vals.quantile(0.90)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })
    return pd.DataFrame(rows)


def make_tier_transition(df: pd.DataFrame, cm: ColumnMap, windows: Sequence[str]) -> pd.DataFrame:
    rows = []
    for w in windows:
        sub = df[df[cm.window] == w]
        if cm.evidence_tier is None:
            continue
        pivot = pd.crosstab(sub["_evidence_tier_norm"], sub["_stability"])
        for tier, rr in pivot.iterrows():
            row = {"window": w, "evidence_tier": tier}
            row.update({str(k): int(v) for k, v in rr.items()})
            rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def make_failure_reason_summary(df: pd.DataFrame, cm: ColumnMap, windows: Sequence[str]) -> pd.DataFrame:
    rows = []
    for w in windows:
        sub = df[df[cm.window] == w]
        vc = sub["_failure_reason"].value_counts(dropna=False)
        for reason, count in vc.items():
            rows.append({"window": w, "failure_reason": str(reason), "count": int(count)})
    return pd.DataFrame(rows)


def make_diagnosis(gate: pd.DataFrame, family_gap: pd.DataFrame, tier_transition: pd.DataFrame, cm: ColumnMap) -> pd.DataFrame:
    def get_gate(w: str, col: str) -> float:
        m = gate[gate["window"] == w]
        if len(m) and col in m.columns:
            return float(m[col].iloc[0])
        return float("nan")

    rows = []
    for w in TARGET_WINDOWS:
        lead_yes = get_gate(w, "lead_lag_yes")
        usable = get_gate(w, "strict_plus_tau0")
        tier1 = get_gate(w, "tier1_count")
        tier2 = get_gate(w, "tier2_count")
        pos_not = get_gate(w, "reason__positive_lag_not_supported")
        total = get_gate(w, "total_pairs")
        rows.append({
            "diagnosis_id": f"{w}_drop_occurs_before_tau0_reverse_filters",
            "support_level": "supported" if math.isfinite(lead_yes) and math.isfinite(usable) and abs(lead_yes - usable) <= max(2, 0.1 * max(lead_yes, 1)) else "needs_manual_review",
            "primary_evidence": f"{w}: lead_lag_yes={lead_yes:.0f}, final strict+tau0={usable:.0f}, positive_lag_not_supported={pos_not:.0f}/{total:.0f}.",
            "allowed_statement": f"{w} low count is primarily located at or before positive-lag support, not mainly created by the later tau0/reverse filters, if lead_lag_yes and final usable are close.",
            "forbidden_statement": f"Do not claim {w} was mostly killed by tau0/reverse filters without checking the gate funnel.",
        })
        rows.append({
            "diagnosis_id": f"{w}_tier_structure",
            "support_level": "supported" if math.isfinite(tier1) else "needs_manual_review",
            "primary_evidence": f"{w}: Tier1={tier1:.0f}, Tier2={tier2:.0f}, final strict+tau0={usable:.0f}.",
            "allowed_statement": f"Describe whether {w} has Tier1 backbone or only Tier2/moderate evidence based on tier counts.",
            "forbidden_statement": f"Do not treat {w}'s low density as a single mechanism before separating Tier1 loss from network narrowing.",
        })
    # family direction biggest gaps
    for w in TARGET_WINDOWS:
        sub = family_gap[family_gap["target_window"] == w].copy()
        if len(sub):
            sub = sub.sort_values("gap_to_other_window_mean")
            top = sub.head(5)
            ev = "; ".join(f"{r.family_direction}: count={r.target_usable_strict_tau0}, gap={r.gap_to_other_window_mean:.2f}" for _, r in top.iterrows())
            rows.append({
                "diagnosis_id": f"{w}_largest_family_direction_gaps",
                "support_level": "descriptive",
                "primary_evidence": ev,
                "allowed_statement": f"Use family-direction gaps to localize which relationship directions contribute most to {w}'s low-density behavior.",
                "forbidden_statement": f"Do not describe {w} as uniformly all-network collapse if the gap is concentrated in a few family directions.",
            })
    return pd.DataFrame(rows)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit where T3/T4 V1 relationship drops occur.")
    p.add_argument("--project-root", default=r"D:\easm_project01", help="Project root path. Default: D:\\easm_project01")
    p.add_argument("--input-csv", default=None, help="Optional explicit V1 stability judged CSV path.")
    p.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    p.add_argument("--windows", default=",".join(WINDOWS_DEFAULT), help="Comma-separated window order.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    audit_root = project_root / "lead_lag_screen" / "V1" / "t3_t4_relation_drop_source_audit_a"
    input_csv = Path(args.input_csv) if args.input_csv else (
        project_root / "lead_lag_screen" / "V1" / "outputs" / "lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b" / "tables" / "lead_lag_pair_summary_stability_judged.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (audit_root / "outputs")
    tables_dir = output_dir / "tables"
    summary_dir = output_dir / "summary"
    logs_dir = output_dir / "logs"
    for d in [tables_dir, summary_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    windows = [w.strip() for w in str(args.windows).split(",") if w.strip()]
    print(f"[V1 T3/T4 drop audit] input: {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df0 = pd.read_csv(input_csv)
    cm = detect_columns(df0)
    df = ensure_windows(add_derived_columns(df0, cm), cm)
    # Only keep requested windows for ordered summaries, but detailed near-miss still uses all rows when appropriate.
    print(f"[V1 T3/T4 drop audit] rows: {len(df)}, columns: {len(df.columns)}")
    print(f"[V1 T3/T4 drop audit] detected columns: {cm}")

    gate = make_gate_funnel(df, cm, windows)
    gate.to_csv(tables_dir / "window_gate_funnel_summary.csv", index=False, encoding="utf-8-sig")

    family_gap = make_family_direction_gap(df[df[cm.window].isin(windows)], cm, windows)
    family_gap.to_csv(tables_dir / "family_direction_T3_T4_gap_contribution.csv", index=False, encoding="utf-8-sig")

    survival = make_pair_survival_chain(df, cm)
    survival.to_csv(tables_dir / "pair_survival_chain_S3_T3_S4_T4_S5.csv", index=False, encoding="utf-8-sig")

    near = make_near_miss(df, cm)
    near.to_csv(tables_dir / "T3_T4_near_miss_pairs.csv", index=False, encoding="utf-8-sig")

    effect = make_effect_distribution(df[df[cm.window].isin(windows)], cm, windows)
    effect.to_csv(tables_dir / "window_effect_size_distribution_summary.csv", index=False, encoding="utf-8-sig")

    tier = make_tier_transition(df[df[cm.window].isin(windows)], cm, windows)
    tier.to_csv(tables_dir / "tier_to_stability_transition_by_window.csv", index=False, encoding="utf-8-sig")

    failure = make_failure_reason_summary(df[df[cm.window].isin(windows)], cm, windows)
    failure.to_csv(tables_dir / "failure_reason_counts_by_window.csv", index=False, encoding="utf-8-sig")

    diag = make_diagnosis(gate, family_gap, tier, cm)
    diag.to_csv(tables_dir / "t3_t4_drop_source_diagnosis_table.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "audit": "v1_t3_t4_relation_drop_source_audit_a",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "windows": windows,
        "target_windows": TARGET_WINDOWS,
        "strict_label": STRICT_LABEL,
        "tau0_label": TAU0_LABEL,
        "detected_columns": cm.__dict__,
        "outputs": {
            "window_gate_funnel_summary": str(tables_dir / "window_gate_funnel_summary.csv"),
            "family_direction_T3_T4_gap_contribution": str(tables_dir / "family_direction_T3_T4_gap_contribution.csv"),
            "pair_survival_chain_S3_T3_S4_T4_S5": str(tables_dir / "pair_survival_chain_S3_T3_S4_T4_S5.csv"),
            "T3_T4_near_miss_pairs": str(tables_dir / "T3_T4_near_miss_pairs.csv"),
            "window_effect_size_distribution_summary": str(tables_dir / "window_effect_size_distribution_summary.csv"),
            "tier_to_stability_transition_by_window": str(tables_dir / "tier_to_stability_transition_by_window.csv"),
            "failure_reason_counts_by_window": str(tables_dir / "failure_reason_counts_by_window.csv"),
            "t3_t4_drop_source_diagnosis_table": str(tables_dir / "t3_t4_drop_source_diagnosis_table.csv"),
        },
    }
    write_json(summary_dir / "summary.json", summary)
    write_json(summary_dir / "run_meta.json", summary)

    log_text = [
        "# V1 T3/T4 relation drop source audit log",
        "",
        f"Input: `{input_csv}`",
        f"Output: `{output_dir}`",
        "",
        "This audit is read-only with respect to V1 main outputs. It localizes where T3/T4 relationship-count collapse occurs by gate funnel, pair survival, family-direction gap, near-miss rows, and effect-size distributions.",
        "",
        "Key caution: this audit localizes *where* pairs are filtered or lost. It does not by itself prove the physical or statistical cause of the drop.",
    ]
    (logs_dir / "RUN_LOG.md").write_text("\n".join(log_text), encoding="utf-8")
    print(f"[V1 T3/T4 drop audit] done: {output_dir}")


if __name__ == "__main__":
    main()
