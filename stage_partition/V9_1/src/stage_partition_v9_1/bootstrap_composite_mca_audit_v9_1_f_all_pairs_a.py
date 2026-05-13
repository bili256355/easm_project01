"""
V9.1_f_all_pairs_a: all-pair coverage extension for V9.1_f.

This patch preserves the V9.1_f bootstrap-composite MCA method and changes only
its target coverage: each accepted window is audited over all 10 unordered
P/V/H/Je/Jw object pairs.  Original V9.1_f priority targets keep their original
A/B direction and are marked as original_priority; newly added pairs are marked
as added_all_pair.

Important boundaries
--------------------
* This is still a bootstrap-space diagnostic, not a direct-year classifier.
* It does not replace the frozen original V9.1_f output directory.
* It audits target-selection/coverage bias and adds multiple-testing summaries.
* It does not perform physical interpretation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import os
import time

import numpy as np
import pandas as pd

VERSION = "v9_1_f_all_pairs_a"
OUTPUT_TAG = "bootstrap_composite_mca_audit_v9_1_f_all_pairs_a"
OBJECTS = ["P", "V", "H", "Je", "Jw"]
DEFAULT_WINDOWS = ["W045", "W081", "W113", "W160"]

# Preserve original V9.1_f target orientations for direct comparison.
ORIGINAL_PRIORITY_TARGETS: Dict[str, List[Tuple[str, str]]] = {
    "W045": [("Je", "Jw"), ("P", "Jw"), ("V", "Jw")],
    "W081": [("P", "V"), ("V", "Jw"), ("H", "Jw")],
    "W113": [("V", "Je"), ("H", "Je"), ("P", "V"), ("Jw", "H"), ("Jw", "V")],
    "W160": [("V", "Je"), ("H", "Jw"), ("P", "V"), ("Jw", "V")],
}


@dataclass
class V91FAllPairsConfig:
    windows: List[str] = field(default_factory=lambda: list(DEFAULT_WINDOWS))
    objects: List[str] = field(default_factory=lambda: list(OBJECTS))
    run_base_method: bool = True
    postprocess_only: bool = False
    physical_interpretation_included: bool = False
    run_multiple_testing_audit: bool = True

    @classmethod
    def from_env(cls) -> "V91FAllPairsConfig":
        cfg = cls()
        if os.environ.get("V9_1_F_ALL_PAIRS_WINDOWS"):
            cfg.windows = [x.strip() for x in os.environ["V9_1_F_ALL_PAIRS_WINDOWS"].replace(";", ",").split(",") if x.strip()]
        if os.environ.get("V9_1_F_ALL_PAIRS_POSTPROCESS_ONLY"):
            cfg.postprocess_only = os.environ["V9_1_F_ALL_PAIRS_POSTPROCESS_ONLY"].strip() not in ("0", "false", "False")
            cfg.run_base_method = not cfg.postprocess_only
        if os.environ.get("V9_1_F_ALL_PAIRS_RUN_BASE_METHOD"):
            cfg.run_base_method = os.environ["V9_1_F_ALL_PAIRS_RUN_BASE_METHOD"].strip() not in ("0", "false", "False")
        return cfg


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _target_name(window_id: str, a: str, b: str) -> str:
    return f"{window_id}_{a}_vs_{b}_delta_peak"


def _unordered_key(a: str, b: str) -> frozenset:
    return frozenset((str(a), str(b)))


def _priority_orientation_map(window_id: str) -> Dict[frozenset, Tuple[str, str]]:
    return {_unordered_key(a, b): (a, b) for a, b in ORIGINAL_PRIORITY_TARGETS.get(window_id, [])}


def build_all_pair_target_registry(window_id: str, objects: Iterable[str] = OBJECTS) -> pd.DataFrame:
    """Build all 10 object pairs, preserving original-priority orientations."""
    objs = list(objects)
    priority_map = _priority_orientation_map(window_id)
    rows: List[dict] = []
    for i, a0 in enumerate(objs):
        for b0 in objs[i + 1:]:
            key = _unordered_key(a0, b0)
            if key in priority_map:
                a, b = priority_map[key]
                target_set = "original_priority"
                was_original = True
                reason = "original V9.1_f priority target; orientation preserved for direct comparison"
            else:
                a, b = a0, b0
                target_set = "added_all_pair"
                was_original = False
                reason = "added by V9.1_f_all_pairs_a to audit target-selection coverage"
            rows.append({
                "window_id": window_id,
                "target_name": _target_name(window_id, a, b),
                "target_pair": f"{a}-{b}",
                "unordered_pair": "-".join(sorted([a, b])),
                "object_A": a,
                "object_B": b,
                "Y_definition": "delta_B_minus_A = peak_B - peak_A; positive means A earlier than B",
                "target_priority": target_set,
                "target_set": target_set,
                "was_in_original_v9_1_f": bool(was_original),
                "target_source_reason": reason,
                "method_role": "bootstrap_composite_MCA_target_not_physical_type",
                "result_scope_note": "all-pair extension result; original priority targets are marked separately",
            })
    return pd.DataFrame(rows)


def _monkey_patch_base_module(base, cfg: V91FAllPairsConfig) -> None:
    """Patch output tag, version, and target registry only; preserve all MCA logic."""
    base.VERSION = VERSION
    base.OUTPUT_TAG = OUTPUT_TAG

    def _targets_for_window_all_pairs(window_id: str, base_cfg) -> pd.DataFrame:
        return build_all_pair_target_registry(window_id, getattr(base_cfg, "objects", OBJECTS))

    base._targets_for_window = _targets_for_window_all_pairs

    # Map all-pair env aliases to original V9.1_f config env variables.
    alias_map = {
        "V9_1_F_ALL_PAIRS_BOOTSTRAP_N": "V9_1F_BOOTSTRAP_N",
        "V9_1_F_ALL_PAIRS_PERM_N": "V9_1F_PERM_N",
        "V9_1_F_ALL_PAIRS_MODE_STABILITY_N": "V9_1F_MODE_STABILITY_N",
        "V9_1_F_ALL_PAIRS_SIGNFLIP_N": "V9_1F_SIGNFLIP_N",
        "V9_1_F_ALL_PAIRS_PERM_BATCH_SIZE": "V9_1F_PERM_BATCH_SIZE",
        "V9_1_F_ALL_PAIRS_MAX_PATTERN_COEFFICIENTS": "V9_1F_MAX_PATTERN_COEFFICIENTS",
        "V9_1_F_ALL_PAIRS_WRITE_PHASE_COMPOSITE_FULL": "V9_1F_WRITE_PHASE_COMPOSITE_FULL",
        "V9_1_F_ALL_PAIRS_ENABLE_PHASE_COMPOSITE": "V9_1F_ENABLE_PHASE_COMPOSITE",
        "V9_1_F_ALL_PAIRS_ENABLE_PATTERN_SUMMARY": "V9_1F_ENABLE_PATTERN_SUMMARY",
        "V9_1_F_ALL_PAIRS_LOG_EVERY_BOOTSTRAP": "V9_1F_LOG_EVERY_BOOTSTRAP",
        "V9_1_F_ALL_PAIRS_WINDOWS": "V9_1F_WINDOWS",
    }
    for src, dst in alias_map.items():
        if os.environ.get(src) is not None:
            os.environ[dst] = os.environ[src]
    os.environ["V9_1F_TARGETS"] = "all"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _annotate_with_target_set(df: pd.DataFrame, target_reg: pd.DataFrame) -> pd.DataFrame:
    if df.empty or target_reg.empty or "target_name" not in df.columns:
        return df.copy()
    keep = [c for c in [
        "window_id", "target_name", "target_pair", "unordered_pair", "object_A", "object_B",
        "target_set", "was_in_original_v9_1_f", "result_scope_note"
    ] if c in target_reg.columns]
    meta = target_reg[keep].drop_duplicates("target_name")
    # Avoid duplicate object columns from target tables that already have object_A/B.
    drop_cols = [c for c in ["target_pair", "unordered_pair", "target_set", "was_in_original_v9_1_f", "result_scope_note"] if c in df.columns]
    out = df.drop(columns=drop_cols, errors="ignore").merge(meta, on=[c for c in ["window_id", "target_name"] if c in df.columns and c in meta.columns], how="left", suffixes=("", "_target"))
    for c in ["object_A", "object_B"]:
        tc = c + "_target"
        if tc in out.columns:
            if c in out.columns:
                out[c] = out[c].where(out[c].notna(), out[tc])
                out = out.drop(columns=[tc])
            else:
                out = out.rename(columns={tc: c})
    return out


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    msk = np.isfinite(p)
    if msk.sum() == 0:
        return q
    vals = p[msk]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(vals)
    q_sorted = ranked * n / (np.arange(n) + 1.0)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    full = np.empty_like(vals)
    full[order] = q_sorted
    q[msk] = full
    return q


def _build_multiple_testing_audit(cross: Path, target_reg: pd.DataFrame) -> pd.DataFrame:
    perm = _read_csv(cross / "bootstrap_composite_mca_permutation_audit_all_windows.csv")
    sign = _read_csv(cross / "bootstrap_composite_mca_signflip_null_all_windows.csv")
    if perm.empty and sign.empty:
        return pd.DataFrame()
    base_cols = ["window_id", "target_name"]
    df = target_reg[["window_id", "target_name", "target_pair", "target_set", "was_in_original_v9_1_f"]].copy()
    if not perm.empty:
        df = df.merge(perm[["window_id", "target_name", "permutation_empirical_p", "permutation_level", "observed_abs_corr"]], on=base_cols, how="left")
    if not sign.empty:
        tmp = sign[["window_id", "target_name", "signflip_percentile", "signflip_level", "observed_abs_corr"]].copy()
        tmp["signflip_empirical_p_from_percentile"] = 1.0 - pd.to_numeric(tmp["signflip_percentile"], errors="coerce")
        df = df.merge(tmp.drop(columns=["observed_abs_corr"], errors="ignore"), on=base_cols, how="left")
    if "permutation_empirical_p" in df.columns:
        df["fdr_q_permutation_all40"] = _bh_fdr(pd.to_numeric(df["permutation_empirical_p"], errors="coerce").to_numpy())
        df["passes_raw_perm_95"] = pd.to_numeric(df["permutation_empirical_p"], errors="coerce") <= 0.05
        df["passes_fdr_perm_10"] = df["fdr_q_permutation_all40"] <= 0.10
        df["passes_fdr_perm_05"] = df["fdr_q_permutation_all40"] <= 0.05
        df["fdr_q_permutation_within_window"] = np.nan
        for wid, idx in df.groupby("window_id").groups.items():
            vals = pd.to_numeric(df.loc[idx, "permutation_empirical_p"], errors="coerce").to_numpy()
            df.loc[idx, "fdr_q_permutation_within_window"] = _bh_fdr(vals)
    if "signflip_empirical_p_from_percentile" in df.columns:
        df["fdr_q_signflip_all40"] = _bh_fdr(pd.to_numeric(df["signflip_empirical_p_from_percentile"], errors="coerce").to_numpy())
        df["passes_raw_signflip_95"] = pd.to_numeric(df["signflip_empirical_p_from_percentile"], errors="coerce") <= 0.05
        df["passes_fdr_signflip_10"] = df["fdr_q_signflip_all40"] <= 0.10
        df["passes_fdr_signflip_05"] = df["fdr_q_signflip_all40"] <= 0.05
        df["fdr_q_signflip_within_window"] = np.nan
        for wid, idx in df.groupby("window_id").groups.items():
            vals = pd.to_numeric(df.loc[idx, "signflip_empirical_p_from_percentile"], errors="coerce").to_numpy()
            df.loc[idx, "fdr_q_signflip_within_window"] = _bh_fdr(vals)
    return df


def _build_statistical_registry(cross: Path, target_reg: pd.DataFrame) -> pd.DataFrame:
    mode = _read_csv(cross / "bootstrap_composite_mca_mode_summary_all_windows.csv")
    ev3 = _read_csv(cross / "bootstrap_composite_mca_evidence_v3_all_windows.csv")
    lev = _read_csv(cross / "bootstrap_composite_mca_year_leverage_summary_all_windows.csv")
    highlow = _read_csv(cross / "bootstrap_composite_mca_high_low_order_all_windows.csv")
    mt = _build_multiple_testing_audit(cross, target_reg)

    df = target_reg.copy()
    if not mode.empty:
        mode_keep = [c for c in ["window_id", "target_name", "n_bootstrap", "n_features", "Y_mean", "Y_std", "fit_status", "covariance_strength", "corr_score_Y", "spearman_score_Y", "permutation_empirical_p", "permutation_level", "mode_pattern_corr_median", "mode_stability_status"] if c in mode.columns]
        df = df.merge(mode[mode_keep], on=["window_id", "target_name"], how="left")
    if not ev3.empty:
        ev_keep = [c for c in ev3.columns if c not in {"object_A", "object_B"}]
        df = df.merge(ev3[ev_keep], on=["window_id", "target_name"], how="left", suffixes=("", "_ev3"))
    if not lev.empty:
        lev_keep = [c for c in ["window_id", "target_name", "top1_year", "top1_fraction_of_total_abs_leverage", "top3_fraction_of_total_abs_leverage", "top5_fraction_of_total_abs_leverage", "extreme_year_dominated_flag"] if c in lev.columns]
        df = df.merge(lev[lev_keep], on=["window_id", "target_name"], how="left", suffixes=("", "_lev"))
    if not mt.empty:
        mt_keep = [c for c in mt.columns if c not in {"target_pair", "target_set", "was_in_original_v9_1_f"}]
        df = df.merge(mt[mt_keep], on=["window_id", "target_name"], how="left", suffixes=("", "_mt"))
    if not highlow.empty:
        # Compact high/low/mid dominant order summary.
        piv = highlow.pivot_table(index=["window_id", "target_name"], columns="score_group", values=["dominant_order_direction", "dominant_order_probability", "order_level", "delta_median"], aggfunc="first")
        piv.columns = [f"{grp}_{metric}" for metric, grp in piv.columns]
        piv = piv.reset_index()
        df = df.merge(piv, on=["window_id", "target_name"], how="left")
    df["interpretation_boundary_all_pairs_a"] = "bootstrap-space all-pair coverage diagnostic; not physical year-type or causal mechanism"
    return df


def _count_if(df: pd.DataFrame, col: str, values: Iterable[str] | None = None) -> int:
    if df.empty or col not in df.columns:
        return 0
    s = df[col]
    if values is None:
        return int(s.notna().sum())
    return int(s.isin(list(values)).sum())


def _build_original_vs_added_summary(reg: pd.DataFrame) -> pd.DataFrame:
    if reg.empty:
        return pd.DataFrame()
    rows = []
    strong_levels = ["strict_99", "direction_specific_99", "credible_95", "usable_90"]
    for wid, g in reg.groupby("window_id"):
        orig = g[g.get("target_set", "") == "original_priority"]
        add = g[g.get("target_set", "") == "added_all_pair"]
        def metrics(x: pd.DataFrame) -> dict:
            ev = x.get("evidence_v3_level", pd.Series(dtype=object)).astype(str)
            return {
                "n_targets": int(len(x)),
                "n_perm_strict_99": _count_if(x, "permutation_level", ["strict_99"]),
                "n_signflip_direction_specific_99": _count_if(x, "signflip_level", ["direction_specific_99"]),
                "n_mode_stable": _count_if(x, "mode_stability_status", ["stable"]),
                "n_extreme_year_dominated": int(pd.Series(x.get("extreme_year_dominated_flag", [])).astype(str).str.lower().isin(["true", "1"]).sum()) if "extreme_year_dominated_flag" in x.columns else 0,
                "n_pair_specific": int(ev.str.contains("pair_specific", na=False).sum()),
                "n_common_mode": int(ev.str.contains("common_mode", na=False).sum()),
                "n_locking": int(ev.str.contains("locking", na=False).sum()),
                "n_gradient": int(ev.str.contains("gradient", na=False).sum()),
            }
        om = metrics(orig); am = metrics(add)
        flag = "likely_undercoverage" if am.get("n_pair_specific", 0) + am.get("n_common_mode", 0) + am.get("n_locking", 0) + am.get("n_gradient", 0) > 0 else "no_strong_added_signal_detected"
        row = {"window_id": wid, "selection_bias_flag": flag}
        row.update({f"original_{k}": v for k, v in om.items()})
        row.update({f"added_{k}": v for k, v in am.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _build_per_window_density(reg: pd.DataFrame) -> pd.DataFrame:
    if reg.empty:
        return pd.DataFrame()
    rows = []
    for wid, g in reg.groupby("window_id"):
        ev = g.get("evidence_v3_level", pd.Series(dtype=object)).astype(str)
        rows.append({
            "window_id": wid,
            "n_total_pairs": int(len(g)),
            "n_original_priority": int((g.get("target_set", "") == "original_priority").sum()),
            "n_added_all_pair": int((g.get("target_set", "") == "added_all_pair").sum()),
            "n_pass_permutation_95": int(pd.to_numeric(g.get("permutation_empirical_p", pd.Series(np.nan, index=g.index)), errors="coerce").le(0.05).sum()),
            "n_pass_permutation_strict_99": _count_if(g, "permutation_level", ["strict_99"]),
            "n_pass_signflip_direction_99": _count_if(g, "signflip_level", ["direction_specific_99"]),
            "n_mode_stable": _count_if(g, "mode_stability_status", ["stable"]),
            "n_not_extreme_year_dominated": int((~pd.Series(g.get("extreme_year_dominated_flag", False)).astype(str).str.lower().isin(["true", "1"])).sum()) if "extreme_year_dominated_flag" in g.columns else np.nan,
            "n_evidence_pair_specific": int(ev.str.contains("pair_specific", na=False).sum()),
            "n_evidence_common_mode": int(ev.str.contains("common_mode", na=False).sum()),
            "n_evidence_locking": int(ev.str.contains("locking", na=False).sum()),
            "n_evidence_gradient": int(ev.str.contains("gradient", na=False).sum()),
            "window_coverage_status": "all_10_pairs_audited" if len(g) == 10 else "coverage_incomplete_check_outputs",
        })
    return pd.DataFrame(rows)


def _build_window_order_heterogeneity_summary(reg: pd.DataFrame) -> pd.DataFrame:
    if reg.empty:
        return pd.DataFrame()
    rows = []
    for wid, g in reg.groupby("window_id"):
        ev = g.get("evidence_v3_level", pd.Series(dtype=object)).astype(str)
        evidence_mask = ev.str.contains("reversal|locking|gradient", regex=True, na=False)
        gg = g[evidence_mask].copy()
        obj_counts = {obj: 0 for obj in OBJECTS}
        for _, r in gg.iterrows():
            for obj in [r.get("object_A"), r.get("object_B")]:
                if obj in obj_counts:
                    obj_counts[obj] += 1
        dominant = sorted(obj_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        dom_txt = ";".join([f"{k}:{v}" for k, v in dominant if v > 0])
        rows.append({
            "window_id": wid,
            "n_pairs_with_order_dependence_labels": int(evidence_mask.sum()),
            "dominant_objects_in_labeled_pairs": dom_txt,
            "n_pairs_involving_P": int(((gg.get("object_A") == "P") | (gg.get("object_B") == "P")).sum()) if not gg.empty else 0,
            "n_pairs_involving_H": int(((gg.get("object_A") == "H") | (gg.get("object_B") == "H")).sum()) if not gg.empty else 0,
            "n_pairs_involving_V": int(((gg.get("object_A") == "V") | (gg.get("object_B") == "V")).sum()) if not gg.empty else 0,
            "n_pairs_involving_Je": int(((gg.get("object_A") == "Je") | (gg.get("object_B") == "Je")).sum()) if not gg.empty else 0,
            "n_pairs_involving_Jw": int(((gg.get("object_A") == "Jw") | (gg.get("object_B") == "Jw")).sum()) if not gg.empty else 0,
            "coverage_warning": "summary is all-pair bootstrap-space evidence, not physical mechanism or direct-year regime",
            "window_level_summary": "Use as coverage registry; inspect target-level evidence before scientific interpretation.",
        })
    return pd.DataFrame(rows)


def _build_pair_summary(reg: pd.DataFrame) -> pd.DataFrame:
    if reg.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "window_id", "target_pair", "unordered_pair", "object_A", "object_B", "target_set",
        "corr_score_Y", "permutation_empirical_p", "permutation_level", "signflip_level",
        "mode_stability_status", "extreme_year_dominated_flag", "evidence_v3_level",
        "high_dominant_order_direction", "low_dominant_order_direction",
        "high_delta_median", "low_delta_median"
    ] if c in reg.columns]
    return reg[cols].copy()


def _write_summary_md(path: Path, reg: pd.DataFrame, original_added: pd.DataFrame, density: pd.DataFrame) -> None:
    lines = [
        "# V9.1_f_all_pairs_a summary",
        "",
        "This is an all-pair coverage extension of V9.1_f. It preserves the bootstrap-composite MCA method and expands the target registry to all 10 P/V/H/Je/Jw object pairs per accepted window.",
        "",
        "## Interpretation boundary",
        "",
        "- Bootstrap samples are resampled composite perturbations, not independent physical years.",
        "- High/low score groups are bootstrap-space score phases, not physical year types.",
        "- New all-pair results must be interpreted with target coverage and multiple-testing audits.",
        "- This patch does not perform physical interpretation.",
        "",
    ]
    if not reg.empty:
        lines.extend([
            "## Coverage",
            f"- total targets: {len(reg)}",
            f"- original priority targets: {int((reg['target_set'] == 'original_priority').sum()) if 'target_set' in reg.columns else 'unknown'}",
            f"- added all-pair targets: {int((reg['target_set'] == 'added_all_pair').sum()) if 'target_set' in reg.columns else 'unknown'}",
            "",
        ])
        if "evidence_v3_level" in reg.columns:
            lines.append("## evidence_v3 counts")
            for k, v in reg["evidence_v3_level"].value_counts(dropna=False).items():
                lines.append(f"- {k}: {int(v)}")
            lines.append("")
    if not density.empty:
        lines.append("## Per-window density")
        for _, r in density.iterrows():
            lines.append(f"- {r.get('window_id')}: {int(r.get('n_total_pairs', 0))} pairs audited; {int(r.get('n_evidence_pair_specific', 0))} pair-specific, {int(r.get('n_evidence_common_mode', 0))} common-mode, {int(r.get('n_evidence_locking', 0))} locking, {int(r.get('n_evidence_gradient', 0))} gradient labels.")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def postprocess_all_pairs_outputs(v91_root: Path, cfg: V91FAllPairsConfig) -> None:
    v91_root = Path(v91_root).resolve()
    out_root = v91_root / "outputs" / OUTPUT_TAG
    cross = out_root / "cross_window"
    cross.mkdir(parents=True, exist_ok=True)

    _log("[V9.1_f_all_pairs_a] Building coverage registry and post-run summaries...")
    target_reg = _read_csv(cross / "bootstrap_composite_mca_target_registry_all_windows.csv")
    if target_reg.empty:
        # Build an expected registry even if the base run failed, so the user can inspect target coverage.
        target_reg = pd.concat([build_all_pair_target_registry(w, cfg.objects) for w in cfg.windows], ignore_index=True)
    else:
        # Re-annotate using the preserved-orientation registry to add target_set columns even if base output is minimal.
        expected = pd.concat([build_all_pair_target_registry(w, cfg.objects) for w in cfg.windows], ignore_index=True)
        target_reg = target_reg.drop(columns=[c for c in ["target_set", "target_pair", "unordered_pair", "was_in_original_v9_1_f", "result_scope_note"] if c in target_reg.columns], errors="ignore")
        meta = expected[["window_id", "target_name", "target_pair", "unordered_pair", "target_set", "was_in_original_v9_1_f", "result_scope_note"]]
        target_reg = target_reg.merge(meta, on=["window_id", "target_name"], how="left")

    _safe_to_csv(target_reg, cross / "v9_1_f_all_pairs_target_registry.csv")

    # Annotated versions of key base outputs.
    for src_name, dst_name in [
        ("bootstrap_composite_mca_evidence_v3_all_windows.csv", "v9_1_f_all_pairs_evidence_v3_annotated.csv"),
        ("bootstrap_composite_mca_mode_summary_all_windows.csv", "v9_1_f_all_pairs_mode_summary_annotated.csv"),
        ("bootstrap_composite_mca_high_low_order_all_windows.csv", "v9_1_f_all_pairs_high_low_order_annotated.csv"),
        ("bootstrap_composite_mca_permutation_audit_all_windows.csv", "v9_1_f_all_pairs_permutation_audit_annotated.csv"),
        ("bootstrap_composite_mca_signflip_null_all_windows.csv", "v9_1_f_all_pairs_signflip_null_annotated.csv"),
    ]:
        df = _read_csv(cross / src_name)
        if not df.empty:
            _safe_to_csv(_annotate_with_target_set(df, target_reg), cross / dst_name)

    registry = _build_statistical_registry(cross, target_reg)
    _safe_to_csv(registry, cross / "v9_1_f_all_pairs_statistical_result_registry.csv")

    mt = _build_multiple_testing_audit(cross, target_reg)
    _safe_to_csv(mt, cross / "v9_1_f_all_pairs_multiple_testing_audit.csv")

    original_added = _build_original_vs_added_summary(registry)
    _safe_to_csv(original_added, cross / "v9_1_f_all_pairs_original_priority_vs_added_summary.csv")

    density = _build_per_window_density(registry)
    _safe_to_csv(density, cross / "v9_1_f_all_pairs_per_window_result_density.csv")

    window_het = _build_window_order_heterogeneity_summary(registry)
    _safe_to_csv(window_het, cross / "v9_1_f_all_pairs_window_order_heterogeneity_summary.csv")

    pair_summary = _build_pair_summary(registry)
    _safe_to_csv(pair_summary, cross / "v9_1_f_all_pairs_pair_summary.csv")

    # Record a compact method audit table.
    method_audit = pd.DataFrame([{
        "version": VERSION,
        "method_base": "v9_1_f_hotfix02_bootstrap_composite_mca",
        "semantic_change": "target coverage expanded from original priority targets to all 10 object pairs per accepted window",
        "n_windows_expected": len(cfg.windows),
        "n_targets_expected": len(cfg.windows) * 10,
        "n_targets_written": int(len(target_reg)),
        "preserves_bootstrap_composite_mca_method": True,
        "does_not_replace_original_v9_1_f": True,
        "multiple_testing_audit_included": True,
        "physical_interpretation_included": False,
    }])
    _safe_to_csv(method_audit, cross / "v9_1_f_all_pairs_method_audit_summary.csv")

    _write_summary_md(cross / "V9_1_F_ALL_PAIRS_A_SUMMARY.md", registry, original_added, density)

    # Merge/augment base run_meta if present.
    meta_path = cross / "run_meta.json"
    base_meta = {}
    if meta_path.exists():
        try:
            base_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            base_meta = {}
    base_meta.update({
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "created_at_postprocess": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method_base": "v9_1_f_hotfix02_bootstrap_composite_mca",
        "semantic_change": "target coverage expanded from priority targets to all 10 object pairs per accepted window",
        "n_windows": len(cfg.windows),
        "n_pairs_per_window": 10,
        "n_targets_total_expected": len(cfg.windows) * 10,
        "n_targets_total_written": int(len(target_reg)),
        "preserves_bootstrap_composite_mca_method": True,
        "uses_bootstrap_resampling": True,
        "uses_all_pair_target_registry": True,
        "does_not_replace_original_v9_1_f": True,
        "multiple_testing_audit_included": True,
        "physical_interpretation_included": False,
        "all_pairs_config": asdict(cfg),
    })
    _write_json(base_meta, meta_path)
    _log(f"[V9.1_f_all_pairs_a] Postprocess complete: {cross}")


def run_bootstrap_composite_mca_audit_v9_1_f_all_pairs_a(v91_root: Path) -> None:
    v91_root = Path(v91_root).resolve()
    cfg = V91FAllPairsConfig.from_env()
    if cfg.run_base_method:
        _log("[V9.1_f_all_pairs_a] Running base V9.1_f method with all-pair target registry...")
        from stage_partition_v9_1 import bootstrap_composite_mca_audit_v9_1_f as base
        _monkey_patch_base_module(base, cfg)
        base.run_bootstrap_composite_mca_audit_v9_1_f(v91_root)
    else:
        _log("[V9.1_f_all_pairs_a] Postprocess-only mode; base MCA run is not executed.")
    postprocess_all_pairs_outputs(v91_root, cfg)


if __name__ == "__main__":
    run_bootstrap_composite_mca_audit_v9_1_f_all_pairs_a(Path(__file__).resolve().parents[2])
