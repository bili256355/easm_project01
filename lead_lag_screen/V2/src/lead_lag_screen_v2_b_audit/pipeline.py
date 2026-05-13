from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .settings import LeadLagScreenV2BAuditSettings


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required audit input not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _join_unique(values: Iterable[object]) -> str:
    vals: List[str] = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v)
        if s and s not in vals:
            vals.append(s)
    return ";".join(vals)


def _join_ints(values: Iterable[object]) -> str:
    vals: List[int] = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            i = int(float(v))
        except Exception:
            continue
        if i not in vals:
            vals.append(i)
    vals.sort()
    return ";".join(str(v) for v in vals)


def _edge_key_cols() -> List[str]:
    return ["window", "source", "target", "source_family", "target_family"]


def _add_lagged_layer_flags(edges: pd.DataFrame, settings: LeadLagScreenV2BAuditSettings) -> pd.DataFrame:
    df = edges.copy()
    df["p_matrix_num"] = _safe_num(df, "p_matrix")
    df["q_within_window_num"] = _safe_num(df, "q_within_window")
    df["val_matrix_num"] = _safe_num(df, "val_matrix")
    df["abs_val_matrix"] = df["val_matrix_num"].abs()
    df["pcmci_graph_selected_bool"] = _bool_series(df, "pcmci_graph_selected")
    df["pcmci_plus_supported_bool"] = _bool_series(df, "pcmci_plus_supported")
    df["raw_p_le_alpha"] = df["p_matrix_num"].le(float(settings.raw_p_alpha)).fillna(False)
    df["window_fdr_q_le_alpha"] = df["q_within_window_num"].le(float(settings.fdr_alpha)).fillna(False)
    df["graph_selected_and_raw_p"] = df["pcmci_graph_selected_bool"] & df["raw_p_le_alpha"]
    df["graph_selected_raw_p_lost_by_window_fdr"] = (
        df["graph_selected_and_raw_p"] & ~df["window_fdr_q_le_alpha"]
    )
    df["graph_selected_but_raw_p_gt_alpha"] = df["pcmci_graph_selected_bool"] & ~df["raw_p_le_alpha"]
    df["raw_p_sig_but_not_graph_selected"] = df["raw_p_le_alpha"] & ~df["pcmci_graph_selected_bool"]

    def layer(row) -> str:
        if bool(row["pcmci_plus_supported_bool"]):
            return "L4_supported_graph_and_window_fdr"
        if bool(row["graph_selected_raw_p_lost_by_window_fdr"]):
            return "L3_graph_selected_raw_p_lost_by_fdr"
        if bool(row["graph_selected_but_raw_p_gt_alpha"]):
            return "L2_graph_selected_not_raw_p05"
        if bool(row["raw_p_sig_but_not_graph_selected"]):
            return "L1_raw_p05_not_graph_selected"
        return "L0_not_selected_not_raw_p05"

    df["pcmci_lagged_layer"] = df.apply(layer, axis=1)
    return df


def _pair_level_lagged(lagged: pd.DataFrame) -> pd.DataFrame:
    key = _edge_key_cols()
    rows: List[dict] = []
    for key_vals, g in lagged.groupby(key, dropna=False):
        row = dict(zip(key, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        g2 = g.copy()
        # Prefer final support, then graph+raw, then graph, then raw, then lowest q/p/highest abs val.
        g2["_rank_support"] = np.select(
            [
                g2["pcmci_plus_supported_bool"],
                g2["graph_selected_and_raw_p"],
                g2["pcmci_graph_selected_bool"],
                g2["raw_p_le_alpha"],
            ],
            [0, 1, 2, 3],
            default=4,
        )
        g2 = g2.sort_values(
            ["_rank_support", "q_within_window_num", "p_matrix_num", "abs_val_matrix"],
            ascending=[True, True, True, False],
        )
        best = g2.iloc[0]
        row.update({
            "any_lagged_graph_selected": bool(g["pcmci_graph_selected_bool"].any()),
            "any_lagged_raw_p05": bool(g["raw_p_le_alpha"].any()),
            "any_lagged_graph_selected_and_raw_p05": bool(g["graph_selected_and_raw_p"].any()),
            "any_lagged_window_fdr_q05": bool(g["window_fdr_q_le_alpha"].any()),
            "any_lagged_supported": bool(g["pcmci_plus_supported_bool"].any()),
            "supported_taus": _join_ints(g.loc[g["pcmci_plus_supported_bool"], "tau"]),
            "graph_selected_taus": _join_ints(g.loc[g["pcmci_graph_selected_bool"], "tau"]),
            "raw_p05_taus": _join_ints(g.loc[g["raw_p_le_alpha"], "tau"]),
            "graph_raw_p05_taus": _join_ints(g.loc[g["graph_selected_and_raw_p"], "tau"]),
            "best_lagged_tau": int(best["tau"]) if pd.notna(best.get("tau")) else np.nan,
            "best_lagged_layer": str(best["pcmci_lagged_layer"]),
            "best_lagged_val": float(best["val_matrix_num"]) if pd.notna(best["val_matrix_num"]) else np.nan,
            "best_lagged_abs_val": float(best["abs_val_matrix"]) if pd.notna(best["abs_val_matrix"]) else np.nan,
            "best_lagged_p": float(best["p_matrix_num"]) if pd.notna(best["p_matrix_num"]) else np.nan,
            "best_lagged_q": float(best["q_within_window_num"]) if pd.notna(best["q_within_window_num"]) else np.nan,
            "min_lagged_p": float(g["p_matrix_num"].min()) if g["p_matrix_num"].notna().any() else np.nan,
            "min_lagged_q": float(g["q_within_window_num"].min()) if g["q_within_window_num"].notna().any() else np.nan,
            "max_lagged_abs_val": float(g["abs_val_matrix"].max()) if g["abs_val_matrix"].notna().any() else np.nan,
            "n_lagged_tests": int(len(g)),
            "n_lagged_graph_selected": int(g["pcmci_graph_selected_bool"].sum()),
            "n_lagged_raw_p05": int(g["raw_p_le_alpha"].sum()),
            "n_lagged_graph_raw_p05": int(g["graph_selected_and_raw_p"].sum()),
            "n_lagged_supported": int(g["pcmci_plus_supported_bool"].sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _tau0_pair_level(tau0: pd.DataFrame, settings: LeadLagScreenV2BAuditSettings) -> pd.DataFrame:
    if tau0.empty:
        return pd.DataFrame(columns=_edge_key_cols())
    df = tau0.copy()
    df["p_matrix_num"] = _safe_num(df, "p_matrix")
    df["q_within_window_tau0_num"] = _safe_num(df, "q_within_window_tau0")
    df["val_matrix_num"] = _safe_num(df, "val_matrix")
    df["abs_tau0_val"] = df["val_matrix_num"].abs()
    df["tau0_graph_selected_bool"] = _bool_series(df, "pcmci_graph_selected")
    df["tau0_supported_bool"] = _bool_series(df, "tau0_supported")
    df["tau0_raw_p05"] = df["p_matrix_num"].le(float(settings.raw_p_alpha)).fillna(False)
    df["tau0_q05"] = df["q_within_window_tau0_num"].le(float(settings.fdr_alpha)).fillna(False)
    out = df[_edge_key_cols() + [
        "tau0_graph_selected_bool", "tau0_raw_p05", "tau0_q05", "tau0_supported_bool",
        "val_matrix_num", "abs_tau0_val", "p_matrix_num", "q_within_window_tau0_num", "graph_entry",
    ]].copy()
    out = out.rename(columns={
        "val_matrix_num": "tau0_val",
        "p_matrix_num": "tau0_p",
        "q_within_window_tau0_num": "tau0_q",
        "graph_entry": "tau0_graph_entry",
    })
    return out


def _read_v1_or_overlap(settings: LeadLagScreenV2BAuditSettings, v1_overlap: pd.DataFrame) -> pd.DataFrame:
    v1 = _read_csv(settings.v1_evidence_tier_summary, required=False)
    if not v1.empty:
        keep = [
            "window", "source", "target", "source_family", "target_family", "lead_lag_label",
            "lead_lag_group", "direction_label", "same_day_coupling_flag", "positive_peak_lag",
            "positive_peak_signed_r", "positive_peak_abs_r", "lag0_signed_r", "lag0_abs_r",
            "evidence_tier", "recommended_usage", "failure_reason", "risk_note",
            "suggested_reverse_direction", "pair_phi_risk",
        ]
        keep = [c for c in keep if c in v1.columns]
        return v1[keep].copy()
    # Fallback to the V2 overlap table if the V1 table is not available.
    possible = [
        "window", "source", "target", "source_family", "target_family", "lead_lag_label",
        "lead_lag_group", "direction_label", "same_day_coupling_flag", "positive_peak_lag",
        "positive_peak_signed_r", "positive_peak_abs_r", "lag0_signed_r", "lag0_abs_r",
        "evidence_tier", "recommended_usage", "failure_reason", "risk_note",
        "suggested_reverse_direction", "pair_phi_risk",
    ]
    keep = [c for c in possible if c in v1_overlap.columns]
    if not keep:
        return pd.DataFrame()
    return v1_overlap[keep].drop_duplicates().copy()


def _merge_v1(pair_lagged: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    if v1.empty:
        out = pair_lagged.copy()
        out["v1_available"] = False
        out["v1_lead_lag_yes"] = False
        return out
    key = _edge_key_cols()
    v = v1.copy()
    if "lead_lag_group" in v.columns:
        v["v1_lead_lag_yes"] = v["lead_lag_group"].astype(str).eq("lead_lag_yes")
    else:
        v["v1_lead_lag_yes"] = False
    out = pair_lagged.merge(v, on=key, how="left")
    out["v1_available"] = out.get("lead_lag_group", pd.Series(index=out.index)).notna()
    out["v1_lead_lag_yes"] = out["v1_lead_lag_yes"].fillna(False).astype(bool)
    return out


def _add_tau0_and_fate(pair: pd.DataFrame, tau0_pair: pd.DataFrame) -> pd.DataFrame:
    key = _edge_key_cols()
    out = pair.merge(tau0_pair, on=key, how="left") if not tau0_pair.empty else pair.copy()
    for c in ["tau0_graph_selected_bool", "tau0_raw_p05", "tau0_q05", "tau0_supported_bool"]:
        if c not in out.columns:
            out[c] = False
        out[c] = out[c].fillna(False).astype(bool)
    for c in ["tau0_val", "abs_tau0_val", "tau0_p", "tau0_q"]:
        if c not in out.columns:
            out[c] = np.nan

    out["tau0_abs_ge_best_lagged_abs"] = out["abs_tau0_val"].ge(out["best_lagged_abs_val"]).fillna(False)

    def lag_level(row) -> str:
        if bool(row["any_lagged_supported"]):
            return "lagged_L4_supported"
        if bool(row["any_lagged_graph_selected_and_raw_p05"]):
            return "lagged_L3_graph_raw_p05_lost_by_fdr"
        if bool(row["any_lagged_graph_selected"]):
            return "lagged_L2_graph_selected_only"
        if bool(row["any_lagged_raw_p05"]):
            return "lagged_L1_raw_p05_not_graph_selected"
        return "lagged_L0_none"

    def tau0_level(row) -> str:
        if bool(row["tau0_supported_bool"]):
            return "tau0_L4_supported"
        if bool(row["tau0_graph_selected_bool"] and row["tau0_raw_p05"]):
            return "tau0_L3_graph_raw_p05_lost_by_fdr"
        if bool(row["tau0_graph_selected_bool"]):
            return "tau0_L2_graph_selected_only"
        if bool(row["tau0_raw_p05"]):
            return "tau0_L1_raw_p05_not_graph_selected"
        return "tau0_L0_none"

    out["lagged_layer_pair"] = out.apply(lag_level, axis=1)
    out["tau0_layer_pair"] = out.apply(tau0_level, axis=1)

    def joint(row) -> str:
        lag_sup = bool(row["any_lagged_supported"])
        tau_sup = bool(row["tau0_supported_bool"])
        if lag_sup and tau_sup:
            return "both_lagged_and_tau0_supported"
        if lag_sup and not tau_sup:
            return "lagged_supported_only"
        if (not lag_sup) and tau_sup:
            return "tau0_supported_only"
        if bool(row["any_lagged_graph_selected_and_raw_p05"] and row["tau0_graph_selected_bool"]):
            return "both_have_pre_fdr_signal"
        if bool(row["any_lagged_graph_selected_and_raw_p05"]):
            return "lagged_pre_fdr_signal_only"
        if bool(row["tau0_graph_selected_bool"] and row["tau0_raw_p05"]):
            return "tau0_pre_fdr_signal_only"
        if bool(row["any_lagged_graph_selected"] or row["tau0_graph_selected_bool"]):
            return "graph_selected_but_weak_p"
        return "no_pcmci_signal"

    out["tau0_lagged_joint_class"] = out.apply(joint, axis=1)

    def v1_fate(row) -> str:
        if not bool(row.get("v1_lead_lag_yes", False)):
            return "not_v1_lead_lag_yes"
        if bool(row["any_lagged_supported"]):
            return "V1_yes_and_PCMCI_lagged_supported"
        if bool(row["any_lagged_graph_selected_and_raw_p05"]):
            return "V1_yes_graph_raw_p05_lost_by_fdr"
        if bool(row["any_lagged_graph_selected"]):
            return "V1_yes_graph_selected_but_raw_p_gt_alpha"
        if bool(row["any_lagged_raw_p05"]):
            return "V1_yes_raw_p05_not_graph_selected"
        if bool(row["tau0_supported_bool"]):
            return "V1_yes_only_tau0_supported_in_PCMCI"
        if bool(row["tau0_graph_selected_bool"]):
            return "V1_yes_only_tau0_graph_selected_in_PCMCI"
        return "V1_yes_no_PCMCI_lagged_or_tau0_signal"

    out["v1_to_pcmci_fate"] = out.apply(v1_fate, axis=1)
    return out


def _group_counts(df: pd.DataFrame, group_cols: List[str], bool_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols)
    agg: Dict[str, Tuple[str, object]] = {"n_rows": (df.columns[0], "size")}
    for c in bool_cols:
        if c in df.columns:
            agg[f"n_{c}"] = (c, "sum")
    out = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    return out


def _window_threshold(lagged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, g in lagged.groupby("window", dropna=False):
        graph = int(g["pcmci_graph_selected_bool"].sum())
        rawp = int(g["raw_p_le_alpha"].sum())
        graph_raw = int(g["graph_selected_and_raw_p"].sum())
        supported = int(g["pcmci_plus_supported_bool"].sum())
        row = {
            "window": window,
            "n_tests": int(len(g)),
            "n_graph_selected": graph,
            "n_raw_p05": rawp,
            "n_graph_selected_raw_p05": graph_raw,
            "n_window_q05": int(g["window_fdr_q_le_alpha"].sum()),
            "n_supported_graph_and_window_fdr": supported,
            "n_graph_raw_p05_lost_by_window_fdr": int(g["graph_selected_raw_p_lost_by_window_fdr"].sum()),
            "n_raw_p05_not_graph_selected": int(g["raw_p_sig_but_not_graph_selected"].sum()),
            "n_graph_selected_raw_p_gt_alpha": int(g["graph_selected_but_raw_p_gt_alpha"].sum()),
            "min_p": float(g["p_matrix_num"].min()) if g["p_matrix_num"].notna().any() else np.nan,
            "min_q": float(g["q_within_window_num"].min()) if g["q_within_window_num"].notna().any() else np.nan,
            "min_q_among_graph_selected": float(g.loc[g["pcmci_graph_selected_bool"], "q_within_window_num"].min()) if g["pcmci_graph_selected_bool"].any() else np.nan,
            "median_q_among_graph_selected": float(g.loc[g["pcmci_graph_selected_bool"], "q_within_window_num"].median()) if g["pcmci_graph_selected_bool"].any() else np.nan,
            "supported_over_graph_raw_p05": supported / graph_raw if graph_raw else np.nan,
            "supported_over_graph_selected": supported / graph if graph else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _write_guardrails(path: Path) -> None:
    text = """# lead_lag_screen/V2_b audit interpretation guardrails

This audit is a diagnostic layer for `pcmci_plus_smooth5_v2_a`. It does **not** rerun PCMCI+ and does **not** replace V1.

## What this audit can tell you

1. Whether V2_a is narrow because PCMCI+ did not select many graph edges, or because the additional within-window BH-FDR layer removed graph-selected/raw-p signals.
2. Where V1 Tier1/Tier2/Tier3 candidates went under PCMCI+ layers.
3. Whether a relation is more visible as tau=0 contemporaneous coupling than as a tau=1..5 lagged direct edge.
4. Which family directions survive only as a strict direct-edge lower bound.

## What this audit cannot tell you

1. It cannot prove a physical pathway.
2. It cannot prove that V1-only relations are false.
3. It cannot test same-family conditioning sensitivity, because that requires rerunning PCMCI+ with a changed conditioning design.
4. It cannot convert mediator-like or chain-like patterns into established mediation.

## Recommended wording

Use: `under the current strict PCMCI+ direct-edge + window-FDR definition, only a narrow lower-bound set remains`.

Do not use: `PCMCI+ proves only these pathways matter`.
"""
    path.write_text(text, encoding="utf-8")


def _write_conditioning_note(path: Path) -> None:
    text = """# Same-family conditioning sensitivity status

This V2_b audit reads the completed V2_a outputs. It does not rerun PCMCI+.

Therefore, it **does not** test the alternative design in which same-family variables are excluded from the conditioning pool. That test is a separate computational experiment, because it changes the PCMCI+ search/conditioning semantics.

Current V2_a design:

```text
Reported source-target edges: cross-family only
Conditioning pool: all variables, including same-family variables
```

Why this matters:

```text
Allowing same-family controls is statistically cleaner but may partial out object-internal signals.
Excluding same-family controls may recover broader object-level signals but increases redundancy/confounding risk.
```

If this audit shows that key V1 physical expectations disappear before or during graph selection, a next patch can add an explicit `v2_c_same_family_conditioning_sensitivity` rerun rather than silently changing V2_a.
"""
    path.write_text(text, encoding="utf-8")


def run_v2_b_audit(settings: LeadLagScreenV2BAuditSettings | None = None) -> dict:
    settings = settings or LeadLagScreenV2BAuditSettings()
    _ensure_dirs(settings.output_dir, settings.log_dir)

    run_meta = {
        "layer": "lead_lag_screen/V2_b_audit",
        "role": "diagnose why pcmci_plus_smooth5_v2_a is much narrower than V1 smooth5 lead-lag",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings.to_jsonable_dict(),
        "audit_boundaries": [
            "This audit reads existing V2_a outputs; it does not rerun PCMCI+.",
            "It decomposes graph selection, raw p-value, and within-window BH-FDR layers.",
            "It audits tau=0 diagnostics against lagged tau=1..5 edges.",
            "It does not test same-family conditioning sensitivity; that requires a separate rerun.",
            "It does not produce pathway, mediator, or physical mechanism claims.",
        ],
    }
    _write_json(settings.output_dir / "run_meta.json", run_meta)
    _write_json(settings.output_dir / "settings_summary.json", settings.to_jsonable_dict())

    v2dir = settings.v2a_output_dir
    lagged_in = v2dir / "pcmci_plus_edges_long.csv"
    tau0_in = v2dir / "pcmci_plus_tau0_contemporaneous_diagnostic.csv"
    v1_overlap_in = v2dir / "pcmci_plus_v1_overlap.csv"

    edges = _read_csv(lagged_in, required=True)
    tau0 = _read_csv(tau0_in, required=True)
    v1_overlap = _read_csv(v1_overlap_in, required=False)
    v1 = _read_v1_or_overlap(settings, v1_overlap)

    lagged = _add_lagged_layer_flags(edges, settings)
    lagged.to_csv(settings.output_dir / "pcmci_plus_lagged_layer_audit_long.csv", index=False, encoding="utf-8-sig")

    window_threshold = _window_threshold(lagged)
    window_threshold.to_csv(settings.output_dir / "pcmci_plus_window_threshold_audit.csv", index=False, encoding="utf-8-sig")

    family_threshold = _group_counts(
        lagged,
        ["window", "source_family", "target_family"],
        [
            "pcmci_graph_selected_bool", "raw_p_le_alpha", "graph_selected_and_raw_p",
            "window_fdr_q_le_alpha", "pcmci_plus_supported_bool",
            "graph_selected_raw_p_lost_by_window_fdr", "raw_p_sig_but_not_graph_selected",
        ],
    )
    family_threshold.to_csv(settings.output_dir / "pcmci_plus_family_threshold_audit.csv", index=False, encoding="utf-8-sig")

    family_total = _group_counts(
        lagged,
        ["source_family", "target_family"],
        [
            "pcmci_graph_selected_bool", "raw_p_le_alpha", "graph_selected_and_raw_p",
            "window_fdr_q_le_alpha", "pcmci_plus_supported_bool",
            "graph_selected_raw_p_lost_by_window_fdr", "raw_p_sig_but_not_graph_selected",
        ],
    )
    family_total.to_csv(settings.output_dir / "pcmci_plus_family_threshold_audit_all_windows.csv", index=False, encoding="utf-8-sig")

    pair_lagged = _pair_level_lagged(lagged)
    tau0_pair = _tau0_pair_level(tau0, settings)
    pair_with_v1 = _merge_v1(pair_lagged, v1)
    joint = _add_tau0_and_fate(pair_with_v1, tau0_pair)
    joint.to_csv(settings.output_dir / "pcmci_plus_pair_level_tau0_lagged_v1_audit.csv", index=False, encoding="utf-8-sig")

    # V1 lead-lag yes fate table: the main table for diagnosing the user's mismatch.
    v1_yes = joint[joint["v1_lead_lag_yes"].astype(bool)].copy()
    v1_yes.to_csv(settings.output_dir / "v1_lead_lag_yes_fate_under_pcmci_plus.csv", index=False, encoding="utf-8-sig")

    tier_cols = [c for c in ["evidence_tier", "lead_lag_group", "same_day_coupling_flag", "v1_to_pcmci_fate"] if c in joint.columns]
    if tier_cols:
        v1_tier_flow = joint.groupby(tier_cols, dropna=False).size().reset_index(name="n_pairs")
    else:
        v1_tier_flow = pd.DataFrame()
    v1_tier_flow.to_csv(settings.output_dir / "v1_tier_to_pcmci_fate_audit.csv", index=False, encoding="utf-8-sig")

    family_fate = joint.groupby(
        ["window", "source_family", "target_family", "v1_to_pcmci_fate"], dropna=False
    ).size().reset_index(name="n_pairs")
    family_fate.to_csv(settings.output_dir / "v1_family_to_pcmci_fate_audit.csv", index=False, encoding="utf-8-sig")

    tau0_joint = joint.groupby(
        ["window", "source_family", "target_family", "tau0_lagged_joint_class"], dropna=False
    ).size().reset_index(name="n_pairs")
    tau0_joint.to_csv(settings.output_dir / "tau0_vs_lagged_joint_summary.csv", index=False, encoding="utf-8-sig")

    # High-value examples for manual review.
    examples = joint.copy()
    examples["_priority"] = np.select(
        [
            examples["any_lagged_supported"].astype(bool),
            examples["v1_lead_lag_yes"].astype(bool) & examples["any_lagged_graph_selected_and_raw_p05"].astype(bool),
            examples["v1_lead_lag_yes"].astype(bool) & examples["tau0_supported_bool"].astype(bool),
            examples["v1_lead_lag_yes"].astype(bool),
        ],
        [0, 1, 2, 3],
        default=4,
    )
    examples = examples.sort_values(
        ["_priority", "window", "source_family", "target_family", "best_lagged_q", "best_lagged_p", "best_lagged_abs_val"],
        ascending=[True, True, True, True, True, True, False],
    ).head(int(settings.top_n_examples)).drop(columns=["_priority"])
    examples.to_csv(settings.output_dir / "manual_review_priority_examples.csv", index=False, encoding="utf-8-sig")

    # Compact disagreement/retention summaries.
    fate_summary = joint.groupby(["v1_to_pcmci_fate"], dropna=False).size().reset_index(name="n_pairs")
    fate_summary.to_csv(settings.output_dir / "v1_to_pcmci_fate_summary.csv", index=False, encoding="utf-8-sig")

    lagged_layer_summary = lagged.groupby(["window", "pcmci_lagged_layer"], dropna=False).size().reset_index(name="n_tests")
    lagged_layer_summary.to_csv(settings.output_dir / "pcmci_lagged_layer_count_summary.csv", index=False, encoding="utf-8-sig")

    _write_guardrails(settings.output_dir / "INTERPRETATION_GUARDRAILS.md")
    _write_conditioning_note(settings.output_dir / "SAME_FAMILY_CONDITIONING_SENSITIVITY_NOTE.md")

    summary = {
        "status": "completed",
        "layer": "lead_lag_screen/V2_b_audit",
        "input_v2a_output_dir": str(v2dir),
        "output_dir": str(settings.output_dir),
        "n_lagged_tests": int(len(lagged)),
        "n_lagged_graph_selected": int(lagged["pcmci_graph_selected_bool"].sum()),
        "n_lagged_graph_raw_p05": int(lagged["graph_selected_and_raw_p"].sum()),
        "n_lagged_supported_window_fdr": int(lagged["pcmci_plus_supported_bool"].sum()),
        "n_v1_pairs_available": int(joint["v1_available"].sum()) if "v1_available" in joint.columns else 0,
        "n_v1_lead_lag_yes": int(joint["v1_lead_lag_yes"].sum()) if "v1_lead_lag_yes" in joint.columns else 0,
        "n_v1_yes_pcmci_lagged_supported": int((joint.get("v1_to_pcmci_fate", pd.Series(dtype=str)) == "V1_yes_and_PCMCI_lagged_supported").sum()),
        "n_tau0_supported_pairs": int(joint["tau0_supported_bool"].sum()) if "tau0_supported_bool" in joint.columns else 0,
        "same_family_conditioning_sensitivity": "not_tested_in_this_fast_audit_requires_separate_pcmci_rerun",
        "created_outputs": [
            "pcmci_plus_lagged_layer_audit_long.csv",
            "pcmci_plus_window_threshold_audit.csv",
            "pcmci_plus_family_threshold_audit.csv",
            "pcmci_plus_family_threshold_audit_all_windows.csv",
            "pcmci_plus_pair_level_tau0_lagged_v1_audit.csv",
            "v1_lead_lag_yes_fate_under_pcmci_plus.csv",
            "v1_tier_to_pcmci_fate_audit.csv",
            "v1_family_to_pcmci_fate_audit.csv",
            "tau0_vs_lagged_joint_summary.csv",
            "manual_review_priority_examples.csv",
            "v1_to_pcmci_fate_summary.csv",
            "pcmci_lagged_layer_count_summary.csv",
            "INTERPRETATION_GUARDRAILS.md",
            "SAME_FAMILY_CONDITIONING_SENSITIVITY_NOTE.md",
            "summary.json",
            "run_meta.json",
            "settings_summary.json",
        ],
    }
    _write_json(settings.output_dir / "summary.json", summary)
    return summary
