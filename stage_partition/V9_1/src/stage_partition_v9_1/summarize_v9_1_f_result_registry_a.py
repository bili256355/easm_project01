"""
V9.1_f result registry A: read-only statistical evidence freeze layer.

Purpose
-------
This module does NOT rerun V9, V9.1_f, peak detection, bootstrap, or MCA/SVD.
It reads the existing V9 peak outputs and V9.1_f hotfix02 outputs, then builds
clean registry tables for freezing the statistical result layer before physical
interpretation begins.

It is a statistical evidence registry, not a physical interpretation registry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import os
import time

import numpy as np
import pandas as pd

VERSION = "v9_1_f_result_registry_a"
OUTPUT_TAG = "v9_1_f_result_registry_a"
DEFAULT_F_OUTPUT_TAG = "bootstrap_composite_mca_audit_v9_1_f"
DEFAULT_V9_OUTPUT_TAG = "peak_all_windows_v9_a"


@dataclass
class RegistryConfig:
    v91_root: Path
    output_root: Path
    v9_output_root: Path
    f_output_root: Path
    f_cross_root: Path
    v9_cross_root: Path
    include_auxiliary_method_log: bool = True
    freeze_result_methods: bool = True

    @classmethod
    def from_env(cls, v91_root: Path) -> "RegistryConfig":
        v91_root = Path(v91_root)
        v9_root = Path(os.environ.get("V9_1F_REGISTRY_V9_ROOT", v91_root.parent / "V9"))
        v9_output_root = Path(os.environ.get("V9_1F_REGISTRY_V9_OUTPUT_ROOT", v9_root / "outputs" / DEFAULT_V9_OUTPUT_TAG))
        f_output_root = Path(os.environ.get("V9_1F_REGISTRY_F_OUTPUT_ROOT", v91_root / "outputs" / DEFAULT_F_OUTPUT_TAG))
        output_root = Path(os.environ.get("V9_1F_REGISTRY_OUTPUT_ROOT", v91_root / "outputs" / OUTPUT_TAG))
        return cls(
            v91_root=v91_root,
            output_root=output_root,
            v9_output_root=v9_output_root,
            f_output_root=f_output_root,
            f_cross_root=f_output_root / "cross_window",
            v9_cross_root=v9_output_root / "cross_window",
        )


def _log(msg: str) -> None:
    print(f"[V9.1_f registry] {msg}", flush=True)


def _read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        if required:
            raise
        _log(f"WARNING: failed to read {path}: {exc}")
        return pd.DataFrame()


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _safe_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def _num(v, default=np.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _level_from_probability(prob: float) -> str:
    if pd.isna(prob):
        return "missing"
    if prob >= 0.99:
        return "strict_99"
    if prob >= 0.95:
        return "credible_95"
    if prob >= 0.90:
        return "usable_90"
    return "unresolved"


def _canonical_pair(a: str, b: str) -> str:
    return "-".join(sorted([str(a), str(b)]))


def _directional_pair(a: str, b: str) -> str:
    return f"{a}-{b}"


def _find_v9_pair(v9_order: pd.DataFrame, window: str, a: str, b: str) -> Dict[str, object]:
    if v9_order.empty:
        return {}
    df = v9_order.copy()
    exact = df[(df.get("window_id") == window) & (df.get("object_A") == a) & (df.get("object_B") == b)]
    rev = df[(df.get("window_id") == window) & (df.get("object_A") == b) & (df.get("object_B") == a)]
    if not exact.empty:
        r = exact.iloc[0]
        pa = _num(r.get("P_A_earlier"))
        pb = _num(r.get("P_B_earlier"))
        pmax = np.nanmax([pa, pb]) if not (pd.isna(pa) and pd.isna(pb)) else np.nan
        return {
            "V9_pair_match_direction": "exact",
            "V9_peak_A": r.get("A_peak_day", np.nan),
            "V9_peak_B": r.get("B_peak_day", np.nan),
            "V9_delta_B_minus_A": r.get("delta_observed", np.nan),
            "V9_delta_median": r.get("delta_median", np.nan),
            "V9_delta_q025": r.get("delta_q025", np.nan),
            "V9_delta_q975": r.get("delta_q975", np.nan),
            "V9_P_A_earlier": pa,
            "V9_P_B_earlier": pb,
            "V9_P_same_day": r.get("P_same_day", np.nan),
            "V9_peak_order_decision": r.get("peak_order_decision", ""),
            "V9_order_probability_level": _level_from_probability(pmax),
        }
    if not rev.empty:
        r = rev.iloc[0]
        # Original row is B-A relative to requested A-B. Reverse labels and signs.
        pa = _num(r.get("P_B_earlier"))
        pb = _num(r.get("P_A_earlier"))
        pmax = np.nanmax([pa, pb]) if not (pd.isna(pa) and pd.isna(pb)) else np.nan
        return {
            "V9_pair_match_direction": "reversed",
            "V9_peak_A": r.get("B_peak_day", np.nan),
            "V9_peak_B": r.get("A_peak_day", np.nan),
            "V9_delta_B_minus_A": -_num(r.get("delta_observed")),
            "V9_delta_median": -_num(r.get("delta_median")),
            "V9_delta_q025": -_num(r.get("delta_q975")),
            "V9_delta_q975": -_num(r.get("delta_q025")),
            "V9_P_A_earlier": pa,
            "V9_P_B_earlier": pb,
            "V9_P_same_day": r.get("P_same_day", np.nan),
            "V9_peak_order_decision": str(r.get("peak_order_decision", "")) + "__raw_reversed",
            "V9_order_probability_level": _level_from_probability(pmax),
        }
    return {"V9_pair_match_direction": "missing"}


def _object_peak_lookup(obj_reg: pd.DataFrame) -> Dict[Tuple[str, str], object]:
    out = {}
    if obj_reg.empty:
        return out
    for _, r in obj_reg.iterrows():
        out[(str(r.get("window_id")), str(r.get("object")))] = r.get("selected_peak_day", np.nan)
    return out


def _derive_result_tier(evidence_v3_level: str) -> str:
    level = str(evidence_v3_level or "")
    if level in {"robust_pair_specific_reversal", "robust_common_mode_reversal"}:
        return "Tier_1_main_statistical_result"
    if level in {"robust_one_sided_locking", "robust_continuous_gradient"}:
        return "Tier_2_important_auxiliary_result"
    if level.startswith("candidate") or level in {"weak_hint", "weak_bootstrap_coupling_hint"}:
        return "Tier_3_statistical_hint"
    if level in {"not_supported", "replay_mismatch_not_interpretable"}:
        return "Tier_4_negative_or_not_supported"
    return "Tier_3_statistical_hint"


def _interpretation_permission(tier: str, evidence_v3_level: str) -> str:
    if tier.startswith("Tier_1") or tier.startswith("Tier_2"):
        return "ready_for_physical_audit"
    if tier.startswith("Tier_3"):
        return "statistical_result_only"
    return "auxiliary_or_negative_result"


def _result_basis(row: pd.Series) -> str:
    ev = str(row.get("evidence_v3_level", ""))
    scheme = str(row.get("best_quantile_scheme", ""))
    pattern_type = str(row.get("pattern_type", ""))
    if "reversal" in ev:
        if scheme and scheme not in {"tercile", "nan", "None"}:
            return "extreme_quantile_reversal" if scheme in {"decile", "quintile", "quartile"} else "quantile_reversal"
        return "tercile_or_primary_reversal"
    if "one_sided" in ev:
        return "one_sided_locking"
    if "gradient" in ev:
        return "continuous_gradient"
    if pattern_type:
        return pattern_type
    return "unclassified"


def _priority_for_physical_audit(evidence_v3_level: str, tier: str) -> str:
    ev = str(evidence_v3_level or "")
    if ev == "robust_pair_specific_reversal":
        return "priority_1_pair_specific_reversal"
    if ev == "robust_common_mode_reversal":
        return "priority_2_common_mode_reversal"
    if ev in {"robust_one_sided_locking", "robust_continuous_gradient"}:
        return "priority_3_locking_or_gradient"
    if tier.startswith("Tier_3"):
        return "not_priority_statistical_hint_only"
    return "not_ready"


def _dominant_pattern_row(pattern_summary: pd.DataFrame) -> pd.DataFrame:
    if pattern_summary.empty:
        return pd.DataFrame(columns=["window_id", "target_name"])
    df = pattern_summary.copy()
    if "dominant_object_rank" in df.columns:
        df = df.sort_values(["window_id", "target_name", "dominant_object_rank"])
        dom = df.groupby(["window_id", "target_name"], as_index=False).head(1).copy()
    else:
        df = df.sort_values(["window_id", "target_name", "abs_loading_fraction"], ascending=[True, True, False])
        dom = df.groupby(["window_id", "target_name"], as_index=False).head(1).copy()
    keep = [c for c in [
        "window_id", "target_name", "object", "abs_loading_fraction",
        "early_half_abs_loading_fraction", "late_half_abs_loading_fraction",
        "dominant_day_start", "dominant_day_end", "dominant_profile_coord_min", "dominant_profile_coord_max"
    ] if c in dom.columns]
    dom = dom[keep].rename(columns={
        "object": "dominant_object_from_pattern_summary",
        "abs_loading_fraction": "dominant_object_fraction_from_pattern_summary",
    })
    return dom


def _build_registry(cfg: RegistryConfig, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    evidence = tables.get("evidence_v3", pd.DataFrame()).copy()
    if evidence.empty:
        return pd.DataFrame()
    # Normalize key columns.
    if "window" in evidence.columns and "window_id" not in evidence.columns:
        evidence = evidence.rename(columns={"window": "window_id"})

    v9_order = tables.get("v9_pair_order", pd.DataFrame())
    v9_obj = tables.get("v9_object_peak", pd.DataFrame())
    obj_lookup = _object_peak_lookup(v9_obj)

    # Optional dominant pattern fallback.
    dom = _dominant_pattern_row(tables.get("pattern_summary", pd.DataFrame()))
    if not dom.empty:
        evidence = evidence.merge(dom, on=["window_id", "target_name"], how="left")

    rows: List[Dict[str, object]] = []
    for _, r in evidence.iterrows():
        window = str(r.get("window_id"))
        a = str(r.get("object_A"))
        b = str(r.get("object_B"))
        row: Dict[str, object] = {
            "window_id": window,
            "target_name": r.get("target_name", ""),
            "object_A": a,
            "object_B": b,
            "directional_pair": _directional_pair(a, b),
            "canonical_pair": _canonical_pair(a, b),
            "V9_peak_A_from_object_registry": obj_lookup.get((window, a), np.nan),
            "V9_peak_B_from_object_registry": obj_lookup.get((window, b), np.nan),
        }
        row.update(_find_v9_pair(v9_order, window, a, b))
        # Copy key F evidence fields if present.
        copy_cols = [
            "evidence_v3_level", "evidence_v2_level", "evidence_v1_level",
            "pattern_type", "recommended_interpretation", "recommended_interpretation_v3",
            "interpretation_boundary", "permutation_level", "mode_stability_status",
            "extreme_year_dominated_flag", "tercile_high_order_level", "tercile_low_order_level",
            "tercile_high_low_reversal_flag", "best_quantile_scheme", "best_high_order_level",
            "best_low_order_level", "best_high_dominant_probability", "best_low_dominant_probability",
            "best_high_low_reversal_flag", "gradient_status", "delta_gradient_slope",
            "A_earlier_gradient_slope", "B_earlier_gradient_slope", "specificity_status",
            "specificity_own_abs_rank", "specificity_own_minus_max_other", "cross_target_specificity_level",
            "signflip_level", "signflip_percentile", "dominant_object", "dominant_object_fraction",
            "dominant_time_half", "pattern_interpretability_status",
            "dominant_object_from_pattern_summary", "dominant_object_fraction_from_pattern_summary",
            "early_half_abs_loading_fraction", "late_half_abs_loading_fraction",
            "dominant_day_start", "dominant_day_end", "dominant_profile_coord_min", "dominant_profile_coord_max",
        ]
        for c in copy_cols:
            if c in r.index:
                row[c] = r.get(c)
            elif c in evidence.columns:
                row[c] = r.get(c)
        ev = str(row.get("evidence_v3_level", ""))
        tier = _derive_result_tier(ev)
        row["result_tier"] = tier
        row["interpretation_permission"] = _interpretation_permission(tier, ev)
        row["physical_audit_priority"] = _priority_for_physical_audit(ev, tier)
        row["result_basis"] = _result_basis(pd.Series(row))
        row["statistical_interpretation_boundary"] = (
            "statistical evidence registry only; not a physical type, causal pathway, or year-type classification"
        )
        rows.append(row)
    registry = pd.DataFrame(rows)
    preferred = [
        "window_id", "target_name", "object_A", "object_B", "directional_pair", "canonical_pair",
        "V9_peak_A_from_object_registry", "V9_peak_B_from_object_registry", "V9_peak_A", "V9_peak_B",
        "V9_delta_B_minus_A", "V9_delta_median", "V9_delta_q025", "V9_delta_q975",
        "V9_P_A_earlier", "V9_P_B_earlier", "V9_P_same_day", "V9_peak_order_decision", "V9_order_probability_level",
        "evidence_v3_level", "result_tier", "interpretation_permission", "physical_audit_priority", "result_basis",
        "pattern_type", "specificity_status", "cross_target_specificity_level", "signflip_level", "mode_stability_status",
        "extreme_year_dominated_flag", "gradient_status", "best_quantile_scheme", "best_high_order_level", "best_low_order_level",
        "best_high_dominant_probability", "best_low_dominant_probability", "best_high_low_reversal_flag",
        "tercile_high_order_level", "tercile_low_order_level", "tercile_high_low_reversal_flag",
        "dominant_object", "dominant_object_fraction", "dominant_time_half", "pattern_interpretability_status",
        "recommended_interpretation_v3", "statistical_interpretation_boundary",
    ]
    cols = [c for c in preferred if c in registry.columns] + [c for c in registry.columns if c not in preferred]
    return registry[cols]


def _result_tier_summary(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    g = registry.groupby(["result_tier", "evidence_v3_level"], dropna=False).size().reset_index(name="n_results")
    g["tier_definition"] = g["result_tier"].map({
        "Tier_1_main_statistical_result": "main frozen statistical result; eligible for physical-audit queue",
        "Tier_2_important_auxiliary_result": "important auxiliary statistical result; eligible for lower-priority physical audit",
        "Tier_3_statistical_hint": "statistical hint only; do not physically interpret before additional checks",
        "Tier_4_negative_or_not_supported": "negative/not-supported/method-audit result",
    }).fillna("unclassified")
    return g.sort_values(["result_tier", "evidence_v3_level"])


def _join_main_pairs(sub: pd.DataFrame, max_items: int = 6) -> str:
    if sub.empty:
        return ""
    vals = [f"{r.object_A}-{r.object_B}:{r.evidence_v3_level}" for r in sub.itertuples()]
    if len(vals) > max_items:
        vals = vals[:max_items] + [f"...(+{len(vals)-max_items})"]
    return "; ".join(vals)


def _window_summary(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    rows = []
    for window, sub in registry.groupby("window_id"):
        counts = sub["evidence_v3_level"].value_counts().to_dict()
        tier_counts = sub["result_tier"].value_counts().to_dict()
        tier12 = sub[sub["result_tier"].isin(["Tier_1_main_statistical_result", "Tier_2_important_auxiliary_result"])]
        dominant_result_type = sub["evidence_v3_level"].value_counts().idxmax() if not sub.empty else ""
        sentence = (
            f"{window}: {len(sub)} target(s); dominant statistical pattern = {dominant_result_type}; "
            f"Tier1={tier_counts.get('Tier_1_main_statistical_result',0)}, "
            f"Tier2={tier_counts.get('Tier_2_important_auxiliary_result',0)}."
        )
        rows.append({
            "window_id": window,
            "n_targets": int(len(sub)),
            "n_pair_specific_reversal": int(counts.get("robust_pair_specific_reversal", 0)),
            "n_common_mode_reversal": int(counts.get("robust_common_mode_reversal", 0)),
            "n_one_sided_locking": int(counts.get("robust_one_sided_locking", 0)),
            "n_continuous_gradient": int(counts.get("robust_continuous_gradient", 0)),
            "n_tier1": int(tier_counts.get("Tier_1_main_statistical_result", 0)),
            "n_tier2": int(tier_counts.get("Tier_2_important_auxiliary_result", 0)),
            "main_pairs": _join_main_pairs(tier12),
            "dominant_result_type": dominant_result_type,
            "window_level_summary_sentence": sentence,
        })
    return pd.DataFrame(rows).sort_values("window_id")


def _pair_summary(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    rows = []
    for pair, sub in registry.groupby("canonical_pair"):
        ev_counts = sub["evidence_v3_level"].value_counts().to_dict()
        spec_counts = sub["specificity_status"].value_counts().to_dict() if "specificity_status" in sub.columns else {}
        rows.append({
            "canonical_pair": pair,
            "directional_pairs_seen": ";".join(sorted(set(sub["directional_pair"].astype(str)))),
            "windows_present": ";".join(sorted(set(sub["window_id"].astype(str)))),
            "n_windows": int(sub["window_id"].nunique()),
            "n_results": int(len(sub)),
            "n_pair_specific_reversal": int(ev_counts.get("robust_pair_specific_reversal", 0)),
            "n_common_mode_reversal": int(ev_counts.get("robust_common_mode_reversal", 0)),
            "n_one_sided_locking": int(ev_counts.get("robust_one_sided_locking", 0)),
            "n_continuous_gradient": int(ev_counts.get("robust_continuous_gradient", 0)),
            "n_pair_specific_status": int(spec_counts.get("pair_specific", 0)),
            "n_common_mode_status": int(spec_counts.get("common_mode", 0)),
            "recurrent_pair_flag": bool(sub["window_id"].nunique() >= 2),
            "evidence_levels_by_window": "; ".join(f"{r.window_id}:{r.evidence_v3_level}" for r in sub.itertuples()),
        })
    return pd.DataFrame(rows).sort_values(["n_results", "canonical_pair"], ascending=[False, True])


def _method_audit_status(cfg: RegistryConfig, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    def add(item: str, status: str, details: str, decision: str):
        rows.append({"audit_item": item, "status": status, "details": details, "freeze_decision": decision})

    replay = tables.get("replay_audit", pd.DataFrame())
    if replay.empty:
        add("V9 replay audit", "missing", "v9_replay_bootstrap_regression_audit_all_windows.csv not found", "cannot_freeze_without_manual_check")
    else:
        ok = (replay.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "pass").all()
        add("V9 replay audit", "pass" if ok else "fail", f"{int((replay.get('status', pd.Series(dtype=str)).astype(str).str.lower() == 'pass').sum())}/{len(replay)} pass", "freeze_ok" if ok else "do_not_freeze")

    nan = tables.get("nan_audit", pd.DataFrame())
    if nan.empty:
        add("boundary NaN feature audit", "missing", "bootstrap_composite_X_nan_feature_audit_all_windows.csv not found", "manual_check_needed")
    else:
        align_ok = True
        if "alignment_warning" in nan.columns:
            align_ok = nan["alignment_warning"].fillna("none").astype(str).str.lower().isin(["none", "", "nan"]).all()
        add("boundary NaN feature audit", "pass" if align_ok else "warning", f"{len(nan)} object-window blocks audited; all-NaN boundary features explicitly handled", "freeze_ok" if align_ok else "freeze_with_caution")

    ev3 = tables.get("evidence_v3", pd.DataFrame())
    add("evidence_v3", "pass" if not ev3.empty else "missing", f"{len(ev3)} targets in evidence_v3", "freeze_ok" if not ev3.empty else "manual_check_needed")

    if not ev3.empty:
        for col, label, pass_prefix in [
            ("permutation_level", "permutation audit", "strict_99"),
            ("signflip_level", "signflip direction null", "direction_specific"),
            ("mode_stability_status", "mode stability", "stable"),
        ]:
            if col in ev3.columns:
                vals = ev3[col].fillna("").astype(str)
                if label == "mode stability":
                    ok = (vals == pass_prefix).all()
                    detail = f"{int((vals == pass_prefix).sum())}/{len(vals)} stable"
                else:
                    ok = vals.str.startswith(pass_prefix).all()
                    detail = f"{int(vals.str.startswith(pass_prefix).sum())}/{len(vals)} pass-prefix {pass_prefix}"
                add(label, "pass" if ok else "warning", detail, "freeze_ok" if ok else "freeze_with_caution")
        if "extreme_year_dominated_flag" in ev3.columns:
            flags = ev3["extreme_year_dominated_flag"].map(_as_bool)
            ok = not bool(flags.any())
            add("year leverage audit", "pass" if ok else "warning", f"{int(flags.sum())}/{len(flags)} extreme-year dominated", "freeze_ok" if ok else "freeze_with_caution")

    for key, label, filename in [
        ("target_specificity", "target specificity audit", "bootstrap_composite_mca_target_specificity_all_windows.csv"),
        ("cross_target_null", "cross-target null", "bootstrap_composite_mca_cross_target_null_all_windows.csv"),
        ("quantile", "quantile sensitivity", "bootstrap_composite_mca_quantile_sensitivity_all_windows.csv"),
        ("gradient", "score-gradient audit", "bootstrap_composite_mca_score_gradient_summary_all_windows.csv"),
        ("pattern_summary", "pattern summary", "bootstrap_composite_mca_pattern_summary_all_windows.csv"),
    ]:
        df = tables.get(key, pd.DataFrame())
        add(label, "pass" if not df.empty else "missing", f"{len(df)} rows from {filename}", "freeze_ok" if not df.empty else "manual_check_needed")

    return pd.DataFrame(rows)


def _ready_for_physical_audit(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return pd.DataFrame()
    sub = registry[registry["interpretation_permission"] == "ready_for_physical_audit"].copy()
    if sub.empty:
        return sub
    sub["why_ready"] = sub.apply(lambda r: f"{r.get('evidence_v3_level')} with {r.get('signflip_level')} and mode={r.get('mode_stability_status')}; extreme_year_dominated={r.get('extreme_year_dominated_flag')}", axis=1)
    sub["required_next_material"] = "phase_composite_profiles; high_minus_low profiles; object/time/profile pattern physical-audit; no physical naming before audit"
    cols = [
        "window_id", "target_name", "object_A", "object_B", "evidence_v3_level", "result_tier",
        "physical_audit_priority", "why_ready", "required_next_material", "dominant_object", "dominant_time_half",
        "result_basis", "statistical_interpretation_boundary",
    ]
    return sub[[c for c in cols if c in sub.columns]].sort_values(["physical_audit_priority", "window_id", "target_name"])


def _auxiliary_method_results() -> pd.DataFrame:
    rows = [
        {
            "method_branch": "V9",
            "role": "main peak skeleton layer",
            "status": "frozen_candidate",
            "summary": "Multi-window full-sample observed peak skeleton and bootstrap peak/order uncertainty; not a fixed year-by-year script.",
            "interpretation_boundary": "statistical peak-event timing layer; no causal/pathway claim by itself",
        },
        {
            "method_branch": "V9.1_b",
            "role": "unsupervised transition-type mixture audit",
            "status": "auxiliary_negative_result",
            "summary": "Unsupervised behavior-feature clustering did not yield stable, usable year-type groups.",
            "interpretation_boundary": "does not rule out all heterogeneity; only current unsupervised clustering was not sufficient",
        },
        {
            "method_branch": "V9.1_c",
            "role": "bootstrap year-influence / high-leverage year audit",
            "status": "auxiliary_diagnostic_layer",
            "summary": "Identifies years that influence bootstrap peak/order outcomes; useful for leverage checks, not primary type definition.",
            "interpretation_boundary": "single high-influence year indicates leverage/extreme-year candidate, not transition type",
        },
        {
            "method_branch": "V9.1_d",
            "role": "ordinary EOF/MEOF transition-mode audit",
            "status": "auxiliary_negative_result",
            "summary": "Maximum-variance EOF/MEOF modes did not robustly explain peak-order heterogeneity.",
            "interpretation_boundary": "negative result for ordinary EOF, not for all target-guided coupling methods",
        },
        {
            "method_branch": "V9.1_e",
            "role": "C-derived year-influence targeted SVD exploration",
            "status": "exploratory_supporting_result",
            "summary": "Targeted SVD using C-derived year influence showed order-relevant axes but depends on a less conventional target y.",
            "interpretation_boundary": "supporting/exploratory only; main method should rely on V9.1_f",
        },
        {
            "method_branch": "V9.1_f hotfix02",
            "role": "main bootstrap-composite MCA / bootstrap-space targeted SVD diagnostic",
            "status": "frozen_main_statistical_method_candidate",
            "summary": "Directly couples bootstrap composite anomaly X_b with bootstrap peak-order contrast Y_b; includes NaN mask, permutation, signflip, stability, leverage, specificity, gradient, and evidence_v3 audits.",
            "interpretation_boundary": "bootstrap-space statistical coupling diagnostic; not a physical year-type classifier",
        },
    ]
    return pd.DataFrame(rows)


def _write_markdown_summary(out: Path, registry: pd.DataFrame, tier: pd.DataFrame, win: pd.DataFrame, pair: pd.DataFrame, audit: pd.DataFrame, ready: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# V9.1_f Result Registry A Summary\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 1. Scope\n")
    lines.append("This is a statistical evidence registry. It does not provide physical interpretations, physical type names, or causal pathway claims.\n")
    lines.append("## 2. Method freeze status\n")
    if not audit.empty:
        for r in audit.itertuples(index=False):
            lines.append(f"- **{getattr(r, 'audit_item')}**: {getattr(r, 'status')} — {getattr(r, 'details')} ({getattr(r, 'freeze_decision')})")
    lines.append("\n## 3. Result tier counts\n")
    if not tier.empty:
        for r in tier.itertuples(index=False):
            lines.append(f"- {getattr(r, 'result_tier')} / {getattr(r, 'evidence_v3_level')}: {getattr(r, 'n_results')}")
    lines.append("\n## 4. Window summaries\n")
    if not win.empty:
        for r in win.itertuples(index=False):
            lines.append(f"- {getattr(r, 'window_level_summary_sentence')} Main pairs: {getattr(r, 'main_pairs')}")
    lines.append("\n## 5. Pair summaries\n")
    if not pair.empty:
        for r in pair.itertuples(index=False):
            lines.append(f"- {getattr(r, 'canonical_pair')}: windows={getattr(r, 'windows_present')}; evidence={getattr(r, 'evidence_levels_by_window')}")
    lines.append("\n## 6. Ready for physical-audit queue\n")
    if ready.empty:
        lines.append("No result is currently marked ready for physical audit.")
    else:
        for r in ready.itertuples(index=False):
            lines.append(f"- {getattr(r, 'window_id')} {getattr(r, 'object_A')}-{getattr(r, 'object_B')}: {getattr(r, 'evidence_v3_level')} ({getattr(r, 'physical_audit_priority')})")
    lines.append("\n## 7. Interpretation boundary\n")
    lines.append("Allowed terms: robust pair-specific reversal, robust common-mode reversal, robust one-sided locking, robust continuous gradient, bootstrap-composite coupling, order-relevant mode.\n")
    lines.append("Disallowed at this stage: physical type, causal pathway, real year-type, high/low physical naming, object A drives object B.\n")
    (out / "V9_1_F_RESULT_REGISTRY_A_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _load_tables(cfg: RegistryConfig) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    missing_rows = []
    def load(key: str, path: Path, required: bool = False) -> pd.DataFrame:
        df = _read_csv(path, required=False)
        missing_rows.append({
            "key": key,
            "path": str(path),
            "exists": path.exists(),
            "n_rows": int(len(df)) if not df.empty else 0,
            "required": required,
        })
        return df

    tables = {
        "v9_object_peak": load("v9_object_peak", cfg.v9_cross_root / "cross_window_object_peak_registry.csv", False),
        "v9_pair_order": load("v9_pair_order", cfg.v9_cross_root / "cross_window_pairwise_peak_order.csv", False),
        "v9_pair_synchrony": load("v9_pair_synchrony", cfg.v9_cross_root / "cross_window_pairwise_peak_synchrony.csv", False),
        "evidence_v3": load("evidence_v3", cfg.f_cross_root / "bootstrap_composite_mca_evidence_v3_all_windows.csv", True),
        "evidence_v2": load("evidence_v2", cfg.f_cross_root / "bootstrap_composite_mca_evidence_v2_all_windows.csv", False),
        "high_low": load("high_low", cfg.f_cross_root / "bootstrap_composite_mca_high_low_order_all_windows.csv", False),
        "quantile": load("quantile", cfg.f_cross_root / "bootstrap_composite_mca_quantile_sensitivity_all_windows.csv", False),
        "gradient": load("gradient", cfg.f_cross_root / "bootstrap_composite_mca_score_gradient_summary_all_windows.csv", False),
        "target_specificity": load("target_specificity", cfg.f_cross_root / "bootstrap_composite_mca_target_specificity_all_windows.csv", False),
        "signflip": load("signflip", cfg.f_cross_root / "bootstrap_composite_mca_signflip_null_all_windows.csv", False),
        "cross_target_null": load("cross_target_null", cfg.f_cross_root / "bootstrap_composite_mca_cross_target_null_all_windows.csv", False),
        "year_leverage_summary": load("year_leverage_summary", cfg.f_cross_root / "bootstrap_composite_mca_year_leverage_summary_all_windows.csv", False),
        "year_leverage": load("year_leverage", cfg.f_cross_root / "bootstrap_composite_mca_year_leverage_all_windows.csv", False),
        "pattern_summary": load("pattern_summary", cfg.f_cross_root / "bootstrap_composite_mca_pattern_summary_all_windows.csv", False),
        "nan_audit": load("nan_audit", cfg.f_cross_root / "bootstrap_composite_X_nan_feature_audit_all_windows.csv", False),
        "replay_audit": load("replay_audit", cfg.f_cross_root / "v9_replay_bootstrap_regression_audit_all_windows.csv", False),
    }
    return tables, pd.DataFrame(missing_rows)


def run_summarize_v9_1_f_result_registry_a(v91_root: Path) -> None:
    cfg = RegistryConfig.from_env(Path(v91_root))
    out = cfg.output_root
    out.mkdir(parents=True, exist_ok=True)
    _log(f"Output: {out}")
    _log(f"Reading V9 from: {cfg.v9_cross_root}")
    _log(f"Reading V9.1_f from: {cfg.f_cross_root}")

    tables, input_audit = _load_tables(cfg)
    _safe_to_csv(input_audit, out / "v9_1_f_result_registry_input_audit.csv")

    registry = _build_registry(cfg, tables)
    tier = _result_tier_summary(registry)
    win = _window_summary(registry)
    pair = _pair_summary(registry)
    audit = _method_audit_status(cfg, tables)
    ready = _ready_for_physical_audit(registry)
    aux = _auxiliary_method_results() if cfg.include_auxiliary_method_log else pd.DataFrame()

    _safe_to_csv(registry, out / "v9_1_f_statistical_result_registry.csv")
    _safe_to_csv(tier, out / "v9_1_f_result_tier_summary.csv")
    _safe_to_csv(win, out / "v9_1_f_window_summary.csv")
    _safe_to_csv(pair, out / "v9_1_f_pair_summary.csv")
    _safe_to_csv(audit, out / "v9_1_f_method_audit_freeze_status.csv")
    _safe_to_csv(ready, out / "v9_1_f_ready_for_physical_audit.csv")
    _safe_to_csv(aux, out / "v9_1_auxiliary_method_results.csv")
    _write_markdown_summary(out, registry, tier, win, pair, audit, ready)

    run_meta = {
        "version": VERSION,
        "purpose": "read-only statistical evidence registry for V9 + V9.1_f hotfix02",
        "v91_root": str(cfg.v91_root),
        "v9_output_root": str(cfg.v9_output_root),
        "f_output_root": str(cfg.f_output_root),
        "output_root": str(cfg.output_root),
        "does_not_rerun_peak_or_mca": True,
        "physical_interpretation_included": False,
        "n_registry_rows": int(len(registry)),
        "n_ready_for_physical_audit": int(len(ready)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _safe_json(run_meta, out / "run_meta.json")
    _log("Done.")


if __name__ == "__main__":
    run_summarize_v9_1_f_result_registry_a(Path(__file__).resolve().parents[2])
