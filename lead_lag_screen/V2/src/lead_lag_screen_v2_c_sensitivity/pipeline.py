from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from lead_lag_screen_v2.data_io import build_window_panel, ensure_dirs, read_index_anomalies, read_v1_evidence
from lead_lag_screen_v2.graph_extract import _graph_entry_selected, extract_pcmci_tables
from lead_lag_screen_v2.logging_utils import setup_logger
from lead_lag_screen_v2.stats_utils import fdr_bh

from .settings import LeadLagScreenV2CSensitivitySettings
from .tigramite_adapter import run_pcmci_plus


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_matrix(results: dict, key: str, n_var: int, tau_max: int) -> np.ndarray:
    arr = results.get(key)
    if arr is None:
        return np.full((n_var, n_var, tau_max + 1), np.nan, dtype=float)
    out = np.asarray(arr)
    if out.shape[0] != n_var or out.shape[1] != n_var:
        raise ValueError(f"Unexpected {key} shape: {out.shape}; expected first dims {n_var}x{n_var}")
    return out


def _extract_single_directed_pair(
    results: dict,
    variables: List[str],
    variable_families: Dict[str, str],
    window: str,
    source: str,
    target: str,
    tau_min: int,
    tau_max: int,
    main_tau_min: int,
    main_tau_max: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph = results.get("graph")
    if graph is None:
        raise ValueError("PCMCI+ results do not contain graph matrix.")
    graph = np.asarray(graph, dtype=object)
    n_var = len(variables)
    val = _safe_matrix(results, "val_matrix", n_var, tau_max)
    pval = _safe_matrix(results, "p_matrix", n_var, tau_max)
    si = variables.index(source)
    ti = variables.index(target)
    sf = variable_families[source]
    tf = variable_families[target]
    main_rows = []
    tau0_rows = []
    for tau in range(tau_min, tau_max + 1):
        try:
            entry = graph[si, ti, tau]
        except IndexError:
            entry = ""
        row = {
            "window": window,
            "source": source,
            "target": target,
            "source_family": sf,
            "target_family": tf,
            "same_family_edge": bool(sf == tf),
            "tau": int(tau),
            "graph_entry": str(entry),
            "pcmci_graph_selected": bool(_graph_entry_selected(entry, tau)),
            "val_matrix": float(val[si, ti, tau]) if np.isfinite(val[si, ti, tau]) else np.nan,
            "p_matrix": float(pval[si, ti, tau]) if np.isfinite(pval[si, ti, tau]) else np.nan,
            "edge_interpretation": "contemporaneous_diagnostic" if tau == 0 else "lagged_direct_edge_control",
            "direction_semantics": "source(t)->target(t), contemporaneous diagnostic only" if tau == 0 else "source(t-tau)->target(t)",
        }
        if tau == 0:
            tau0_rows.append(row)
        elif main_tau_min <= tau <= main_tau_max:
            main_rows.append(row)
    return pd.DataFrame(main_rows), pd.DataFrame(tau0_rows)


def _apply_fdr(edges: pd.DataFrame, q_col: str, support_col: str, group_cols: List[str], alpha: float) -> pd.DataFrame:
    if edges.empty:
        return edges
    out = edges.copy()
    out[q_col] = np.nan
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        out.loc[idx, q_col] = fdr_bh(out.loc[idx, "p_matrix"].to_numpy())
    out[support_col] = out["pcmci_graph_selected"].fillna(False).astype(bool) & out[q_col].le(float(alpha)).fillna(False)
    return out


def _vars_by_family(variable_families: Dict[str, str], family: str) -> List[str]:
    return [v for v, f in variable_families.items() if f == family]


def _controls_excluding_source_target_families(
    variable_families: Dict[str, str],
    source: str,
    target: str,
) -> List[str]:
    sf = variable_families[source]
    tf = variable_families[target]
    return [v for v, fam in variable_families.items() if fam not in {sf, tf} and v not in {source, target}]


def _cache_path(settings: LeadLagScreenV2CSensitivitySettings, variant: str, key: str) -> Path:
    return settings.cache_dir / variant / key


def _load_cached_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return None


def _save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _run_targeted_no_same_family_controls(
    settings: LeadLagScreenV2CSensitivitySettings,
    df: pd.DataFrame,
    years: np.ndarray,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[dict]]:
    variant = "c1_targeted_no_same_family_controls"
    cached_edges = _load_cached_csv(_cache_path(settings, variant, "edges.csv")) if (settings.resume and not settings.force_recompute) else None
    cached_tau0 = _load_cached_csv(_cache_path(settings, variant, "tau0.csv")) if (settings.resume and not settings.force_recompute) else None
    cached_meta = _load_cached_csv(_cache_path(settings, variant, "task_meta.csv")) if (settings.resume and not settings.force_recompute) else None
    if cached_edges is not None and cached_tau0 is not None and cached_meta is not None:
        logger.info("[%s] loaded cached variant", variant)
        return cached_edges, cached_tau0, cached_meta, []

    edge_parts: List[pd.DataFrame] = []
    tau0_parts: List[pd.DataFrame] = []
    meta_rows: List[dict] = []
    failed_rows: List[dict] = []

    total_family_pairs = len(settings.targeted_family_pairs)
    selected_windows = {str(w) for w in settings.c1_targeted_windows}
    invalid_windows = sorted(selected_windows - set(settings.windows.keys()))
    if invalid_windows:
        raise ValueError(f"C1 targeted windows are not defined in settings.windows: {invalid_windows}")
    c1_window_items = [(w, settings.windows[w]) for w in settings.windows.keys() if w in selected_windows]
    logger.info("[%s] targeted C1 windows=%s family_pairs=%s", variant, [w for w, _ in c1_window_items], settings.targeted_family_pairs)
    for w_i, (window, (w_start, w_end)) in enumerate(c1_window_items, start=1):
        for fp_i, (sf, tf) in enumerate(settings.targeted_family_pairs, start=1):
            sources = _vars_by_family(settings.variable_families, sf)
            targets = _vars_by_family(settings.variable_families, tf)
            logger.info("[%s] %s window %s/%s family %s->%s (%s/%s)", variant, window, w_i, len(c1_window_items), sf, tf, fp_i, total_family_pairs)
            for source in sources:
                for target in targets:
                    task_t0 = time.time()
                    try:
                        controls = _controls_excluding_source_target_families(settings.variable_families, source, target)
                        subset_vars = [source, target] + controls
                        subset_fams = {v: settings.variable_families[v] for v in subset_vars}
                        data_dict, mask_dict, ext_days = build_window_panel(
                            df=df,
                            variables=subset_vars,
                            years=years,
                            window_start=w_start,
                            window_end=w_end,
                            tau_max=settings.tau_max,
                        )
                        results = run_pcmci_plus(
                            data_dict=data_dict,
                            mask_dict=mask_dict,
                            variables=subset_vars,
                            tau_min=settings.tau_min,
                            tau_max=settings.tau_max,
                            pc_alpha=settings.pc_alpha,
                            parcorr_significance=settings.parcorr_significance,
                            verbosity=settings.tigramite_verbosity,
                        )
                        edges, tau0 = _extract_single_directed_pair(
                            results=results,
                            variables=subset_vars,
                            variable_families=subset_fams,
                            window=window,
                            source=source,
                            target=target,
                            tau_min=settings.tau_min,
                            tau_max=settings.tau_max,
                            main_tau_min=settings.main_tau_min,
                            main_tau_max=settings.main_tau_max,
                        )
                        for d in (edges, tau0):
                            d["variant_id"] = variant
                            d["variant_role"] = "targeted sensitivity: exclude other source-family and target-family variables from conditioning pool for this concrete source-target test"
                            d["pcmci_variable_scope"] = "source_target_plus_other_families"
                            d["conditioning_family_rule"] = "source variable + target variable + variables from families other than source_family and target_family"
                            d["n_pcmci_variables"] = len(subset_vars)
                            d["conditioning_variables"] = ";".join(controls)
                        edge_parts.append(edges)
                        tau0_parts.append(tau0)
                        meta_rows.append({
                            "variant_id": variant,
                            "window": window,
                            "source": source,
                            "target": target,
                            "source_family": sf,
                            "target_family": tf,
                            "n_pcmci_variables": len(subset_vars),
                            "conditioning_variables": ";".join(controls),
                            "extended_day_start": int(ext_days[0]),
                            "extended_day_end": int(ext_days[-1]),
                            "elapsed_seconds": time.time() - task_t0,
                            "status": "completed",
                        })
                    except Exception as exc:
                        failed_rows.append({
                            "variant_id": variant,
                            "window": window,
                            "source": source,
                            "target": target,
                            "source_family": sf,
                            "target_family": tf,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        })
                        logger.exception("[%s] failed task %s %s->%s", variant, window, source, target)
    edges_all = pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame()
    tau0_all = pd.concat(tau0_parts, ignore_index=True) if tau0_parts else pd.DataFrame()
    meta = pd.DataFrame(meta_rows)
    if not edges_all.empty:
        edges_all = _apply_fdr(edges_all, "q_within_variant_window", "pcmci_plus_supported", ["variant_id", "window"], settings.fdr_alpha)
        edges_all["q_within_family_edge_window"] = np.nan
        for _, idx in edges_all.groupby(["variant_id", "window", "source_family", "target_family"], dropna=False).groups.items():
            idx = list(idx)
            edges_all.loc[idx, "q_within_family_edge_window"] = fdr_bh(edges_all.loc[idx, "p_matrix"].to_numpy())
        edges_all["pcmci_plus_supported_family_fdr"] = edges_all["pcmci_graph_selected"].astype(bool) & edges_all["q_within_family_edge_window"].le(settings.fdr_alpha).fillna(False)
        edges_all["support_rule"] = "graph_selected_and_variant_window_BH_q_le_alpha"
    if not tau0_all.empty:
        tau0_all = _apply_fdr(tau0_all, "q_within_variant_window_tau0", "tau0_supported", ["variant_id", "window"], settings.fdr_alpha)
        tau0_all["support_rule"] = "tau0_graph_selected_and_variant_window_BH_q_le_alpha"
    if settings.write_cache:
        _save_csv(_cache_path(settings, variant, "edges.csv"), edges_all)
        _save_csv(_cache_path(settings, variant, "tau0.csv"), tau0_all)
        _save_csv(_cache_path(settings, variant, "task_meta.csv"), meta)
    return edges_all, tau0_all, meta, failed_rows


def _run_family_representative_standard(
    settings: LeadLagScreenV2CSensitivitySettings,
    df: pd.DataFrame,
    years: np.ndarray,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[dict]]:
    variant = "c2_family_representative_standard"
    cached_edges = _load_cached_csv(_cache_path(settings, variant, "edges.csv")) if (settings.resume and not settings.force_recompute) else None
    cached_tau0 = _load_cached_csv(_cache_path(settings, variant, "tau0.csv")) if (settings.resume and not settings.force_recompute) else None
    cached_meta = _load_cached_csv(_cache_path(settings, variant, "task_meta.csv")) if (settings.resume and not settings.force_recompute) else None
    if cached_edges is not None and cached_tau0 is not None and cached_meta is not None:
        logger.info("[%s] loaded cached variant", variant)
        return cached_edges, cached_tau0, cached_meta, []

    rep_vars = list(settings.representative_variables)
    missing = [v for v in rep_vars if v not in settings.variable_families]
    if missing:
        raise ValueError(f"Representative variables are not in variable_families: {missing}")
    rep_fams = {v: settings.variable_families[v] for v in rep_vars}
    edge_parts: List[pd.DataFrame] = []
    tau0_parts: List[pd.DataFrame] = []
    meta_rows: List[dict] = []
    failed_rows: List[dict] = []
    for w_i, (window, (w_start, w_end)) in enumerate(settings.windows.items(), start=1):
        task_t0 = time.time()
        logger.info("[%s] PCMCI+ window %s (%s/%s) representative variables=%s", variant, window, w_i, len(settings.windows), len(rep_vars))
        try:
            data_dict, mask_dict, ext_days = build_window_panel(
                df=df,
                variables=rep_vars,
                years=years,
                window_start=w_start,
                window_end=w_end,
                tau_max=settings.tau_max,
            )
            n_total = sum(arr.size for arr in data_dict.values())
            n_masked = sum(mask.sum() for mask in mask_dict.values())
            results = run_pcmci_plus(
                data_dict=data_dict,
                mask_dict=mask_dict,
                variables=rep_vars,
                tau_min=settings.tau_min,
                tau_max=settings.tau_max,
                pc_alpha=settings.pc_alpha,
                parcorr_significance=settings.parcorr_significance,
                verbosity=settings.tigramite_verbosity,
            )
            edges, tau0 = extract_pcmci_tables(
                results=results,
                variables=rep_vars,
                variable_families=rep_fams,
                window=window,
                tau_min=settings.tau_min,
                tau_max=settings.tau_max,
                main_tau_min=settings.main_tau_min,
                main_tau_max=settings.main_tau_max,
                fdr_alpha=settings.fdr_alpha,
                include_same_family_reported_edges=settings.include_same_family_reported_edges,
            )
            for d in (edges, tau0):
                d["variant_id"] = variant
                d["variant_role"] = "family representative sensitivity: smaller variable pool to reduce same-field collinearity"
                d["pcmci_variable_scope"] = "representative_variables"
                d["conditioning_family_rule"] = "standard PCMCI+ within representative pool"
                d["n_pcmci_variables"] = len(rep_vars)
                d["conditioning_variables"] = ";".join(rep_vars)
            edge_parts.append(edges)
            tau0_parts.append(tau0)
            meta_rows.append({
                "variant_id": variant,
                "window": window,
                "n_pcmci_variables": len(rep_vars),
                "representative_variables": ";".join(rep_vars),
                "extended_day_start": int(ext_days[0]),
                "extended_day_end": int(ext_days[-1]),
                "n_total_values": int(n_total),
                "n_masked_values": int(n_masked),
                "masked_fraction": float(n_masked / n_total) if n_total else np.nan,
                "elapsed_seconds": time.time() - task_t0,
                "status": "completed",
            })
        except Exception as exc:
            failed_rows.append({
                "variant_id": variant,
                "window": window,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            logger.exception("[%s] failed window %s", variant, window)
    edges_all = pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame()
    tau0_all = pd.concat(tau0_parts, ignore_index=True) if tau0_parts else pd.DataFrame()
    meta = pd.DataFrame(meta_rows)
    # Harmonize column names with C1.
    if not edges_all.empty:
        if "q_within_window" in edges_all.columns:
            edges_all = edges_all.rename(columns={"q_within_window": "q_within_variant_window"})
        if "pcmci_plus_supported" not in edges_all.columns:
            edges_all = _apply_fdr(edges_all, "q_within_variant_window", "pcmci_plus_supported", ["variant_id", "window"], settings.fdr_alpha)
        edges_all["q_within_family_edge_window"] = np.nan
        for _, idx in edges_all.groupby(["variant_id", "window", "source_family", "target_family"], dropna=False).groups.items():
            idx = list(idx)
            edges_all.loc[idx, "q_within_family_edge_window"] = fdr_bh(edges_all.loc[idx, "p_matrix"].to_numpy())
        edges_all["pcmci_plus_supported_family_fdr"] = edges_all["pcmci_graph_selected"].astype(bool) & edges_all["q_within_family_edge_window"].le(settings.fdr_alpha).fillna(False)
    if not tau0_all.empty and "q_within_window_tau0" in tau0_all.columns:
        tau0_all = tau0_all.rename(columns={"q_within_window_tau0": "q_within_variant_window_tau0"})
    if settings.write_cache:
        _save_csv(_cache_path(settings, variant, "edges.csv"), edges_all)
        _save_csv(_cache_path(settings, variant, "tau0.csv"), tau0_all)
        _save_csv(_cache_path(settings, variant, "task_meta.csv"), meta)
    return edges_all, tau0_all, meta, failed_rows


def _read_v2a_edges(settings: LeadLagScreenV2CSensitivitySettings) -> pd.DataFrame:
    p = settings.v2a_output_dir / "pcmci_plus_edges_long.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    keep = ["window", "source", "target", "tau", "pcmci_graph_selected", "pcmci_plus_supported", "val_matrix", "p_matrix", "q_within_window"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].rename(columns={
        "pcmci_graph_selected": "v2a_pcmci_graph_selected",
        "pcmci_plus_supported": "v2a_pcmci_plus_supported",
        "val_matrix": "v2a_val_matrix",
        "p_matrix": "v2a_p_matrix",
        "q_within_window": "v2a_q_within_window",
    })


def _augment_with_v1_v2a(edges: pd.DataFrame, tau0: pd.DataFrame, v1: pd.DataFrame, v2a: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = edges.copy()
    z = tau0.copy()
    if not v2a.empty and not out.empty:
        out = out.merge(v2a, on=["window", "source", "target", "tau"], how="left")
    if not v1.empty:
        v1_keep = v1.copy()
        v1_keep["v1_lead_lag_yes"] = v1_keep.get("lead_lag_group", pd.Series(index=v1_keep.index, dtype=object)).astype(str).eq("lead_lag_yes")
        merge_cols = ["window", "source", "target", "source_family", "target_family"]
        v1_cols = merge_cols + [c for c in [
            "lead_lag_label", "lead_lag_group", "same_day_coupling_flag", "positive_peak_lag",
            "positive_peak_abs_r", "lag0_abs_r", "evidence_tier", "recommended_usage",
            "pair_phi_risk", "v1_lead_lag_yes"
        ] if c in v1_keep.columns]
        v1_keep = v1_keep[v1_cols].drop_duplicates(merge_cols)
        if not out.empty:
            out = out.merge(v1_keep, on=merge_cols, how="left", suffixes=("", "_v1"))
        if not z.empty:
            z = z.merge(v1_keep, on=merge_cols, how="left", suffixes=("", "_v1"))
    return out, z


def _summary_by_window(edges: pd.DataFrame, tau0: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = sorted(set(edges.get("variant_id", pd.Series(dtype=object)).dropna().unique()) | set(tau0.get("variant_id", pd.Series(dtype=object)).dropna().unique()))
    for variant in variants:
        windows = sorted(set(edges.loc[edges["variant_id"] == variant, "window"].unique() if not edges.empty else []) | set(tau0.loc[tau0["variant_id"] == variant, "window"].unique() if not tau0.empty else []))
        for window in windows:
            e = edges[(edges["variant_id"] == variant) & (edges["window"] == window)] if not edges.empty else pd.DataFrame()
            z = tau0[(tau0["variant_id"] == variant) & (tau0["window"] == window)] if not tau0.empty else pd.DataFrame()
            rows.append({
                "variant_id": variant,
                "window": window,
                "n_lagged_tests": int(len(e)),
                "n_lagged_graph_selected": int(e["pcmci_graph_selected"].sum()) if not e.empty else 0,
                "n_lagged_raw_p05": int(e["p_matrix"].le(0.05).fillna(False).sum()) if not e.empty else 0,
                "n_lagged_supported_window_fdr": int(e["pcmci_plus_supported"].sum()) if (not e.empty and "pcmci_plus_supported" in e.columns) else 0,
                "n_lagged_supported_family_fdr": int(e["pcmci_plus_supported_family_fdr"].sum()) if (not e.empty and "pcmci_plus_supported_family_fdr" in e.columns) else 0,
                "n_tau0_tests": int(len(z)),
                "n_tau0_graph_selected": int(z["pcmci_graph_selected"].sum()) if not z.empty else 0,
                "n_tau0_supported": int(z["tau0_supported"].sum()) if (not z.empty and "tau0_supported" in z.columns) else 0,
            })
    return pd.DataFrame(rows)


def _summary_by_family(edges: pd.DataFrame, tau0: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    e = edges.copy()
    e["family_edge"] = e["source_family"] + "->" + e["target_family"]
    roll = e.groupby(["variant_id", "window", "source_family", "target_family", "family_edge"], dropna=False).agg(
        n_lagged_tests=("p_matrix", "size"),
        n_lagged_graph_selected=("pcmci_graph_selected", "sum"),
        n_lagged_raw_p05=("p_matrix", lambda x: int(pd.Series(x).le(0.05).fillna(False).sum())),
        n_lagged_supported_window_fdr=("pcmci_plus_supported", "sum"),
        n_lagged_supported_family_fdr=("pcmci_plus_supported_family_fdr", "sum"),
        min_q_window=("q_within_variant_window", "min"),
        min_q_family=("q_within_family_edge_window", "min"),
        max_abs_val=("val_matrix", lambda x: pd.Series(x).abs().max()),
    ).reset_index()
    if not tau0.empty:
        z = tau0.copy()
        z["family_edge"] = z["source_family"] + "->" + z["target_family"]
        zroll = z.groupby(["variant_id", "window", "source_family", "target_family", "family_edge"], dropna=False).agg(
            n_tau0_tests=("p_matrix", "size"),
            n_tau0_graph_selected=("pcmci_graph_selected", "sum"),
            n_tau0_supported=("tau0_supported", "sum"),
        ).reset_index()
        roll = roll.merge(zroll, on=["variant_id", "window", "source_family", "target_family", "family_edge"], how="left")
    return roll.sort_values(["variant_id", "window", "n_lagged_supported_window_fdr", "n_tau0_supported"], ascending=[True, True, False, False])


def _direction_recovery(edges: pd.DataFrame, tau0: pd.DataFrame, v1: pd.DataFrame, source_family: str, target_family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = edges[(edges["source_family"] == source_family) & (edges["target_family"] == target_family)].copy() if not edges.empty else pd.DataFrame()
    z = tau0[(tau0["source_family"] == source_family) & (tau0["target_family"] == target_family)].copy() if not tau0.empty else pd.DataFrame()
    long = e.copy()
    variants = sorted(set(e.get("variant_id", pd.Series(dtype=object)).dropna().unique()) | set(z.get("variant_id", pd.Series(dtype=object)).dropna().unique()))
    rows = []
    for variant in variants:
        e_v = e[e["variant_id"] == variant] if not e.empty else pd.DataFrame()
        z_v = z[z["variant_id"] == variant] if not z.empty else pd.DataFrame()
        v1_dir = v1[(v1["source_family"] == source_family) & (v1["target_family"] == target_family)].copy() if (not v1.empty and {"source_family", "target_family"}.issubset(v1.columns)) else pd.DataFrame()
        n_v1_yes = int(v1_dir.get("lead_lag_group", pd.Series(dtype=object)).astype(str).eq("lead_lag_yes").sum()) if not v1_dir.empty else 0
        v1_yes_keys = set(map(tuple, v1_dir.loc[v1_dir.get("lead_lag_group", pd.Series(dtype=object)).astype(str).eq("lead_lag_yes"), ["window", "source", "target"]].to_numpy())) if not v1_dir.empty else set()
        supported_keys = set(map(tuple, e_v.loc[e_v.get("pcmci_plus_supported", pd.Series(dtype=bool)).fillna(False).astype(bool), ["window", "source", "target"]].drop_duplicates().to_numpy())) if not e_v.empty else set()
        tau0_keys = set(map(tuple, z_v.loc[z_v.get("tau0_supported", pd.Series(dtype=bool)).fillna(False).astype(bool), ["window", "source", "target"]].drop_duplicates().to_numpy())) if not z_v.empty else set()
        rows.append({
            "variant_id": variant,
            "family_edge": f"{source_family}->{target_family}",
            "n_lagged_tests": int(len(e_v)),
            "n_lagged_graph_selected": int(e_v["pcmci_graph_selected"].sum()) if not e_v.empty else 0,
            "n_lagged_raw_p05": int(e_v["p_matrix"].le(0.05).fillna(False).sum()) if not e_v.empty else 0,
            "n_lagged_supported_window_fdr": int(e_v["pcmci_plus_supported"].sum()) if (not e_v.empty and "pcmci_plus_supported" in e_v.columns) else 0,
            "n_lagged_supported_family_fdr": int(e_v["pcmci_plus_supported_family_fdr"].sum()) if (not e_v.empty and "pcmci_plus_supported_family_fdr" in e_v.columns) else 0,
            "n_tau0_tests": int(len(z_v)),
            "n_tau0_graph_selected": int(z_v["pcmci_graph_selected"].sum()) if not z_v.empty else 0,
            "n_tau0_supported": int(z_v["tau0_supported"].sum()) if (not z_v.empty and "tau0_supported" in z_v.columns) else 0,
            "n_v1_lead_lag_yes": n_v1_yes,
            "n_v1_yes_recovered_by_lagged_window_fdr": len(v1_yes_keys & supported_keys),
            "n_v1_yes_recovered_by_tau0": len(v1_yes_keys & tau0_keys),
            "interpretation": "sensitivity recovery count; not pathway evidence",
        })
    return pd.DataFrame(rows), long


def _write_guardrails(settings: LeadLagScreenV2CSensitivitySettings) -> None:
    text = f"""# V2_c PCMCI+ sensitivity guardrails\n\nThis output is a sensitivity rerun layer, not a final scientific result.\n\n## What V2_c tests\n\n1. Whether V2_a lost physically expected V->P / H->P signals because same-source-family and same-target-family derived indices were allowed to enter the conditioning pool.\n2. Whether the 20-index full system is too collinear for a stable PCMCI+ graph-selection result.\n\n## What V2_c does not prove\n\n- It does not establish causal pathways.\n- It does not replace V1 lead-lag temporal eligibility.\n- It does not make mediator or chain claims.\n- It does not turn PCMCI+ edges into physical mechanism conclusions.\n\n## Variant meanings\n\n- `c1_targeted_no_same_family_controls`: targeted diagnostic for {settings.targeted_family_pairs}. For each concrete source->target variable pair, the PCMCI+ variable set is source + target + variables from other families. This exactly excludes other variables from the source family and target family for that concrete test, but it is not a full-system graph.\n- `c2_family_representative_standard`: standard PCMCI+ on a smaller representative variable pool. This tests whether strong within-family redundancy in the 20-index pool caused graph collapse.\n\n## How to judge V->P\n\nIf V->P remains absent even in C1 and C2, then PCMCI+ is likely not a useful control for this index/window design without a deeper method redesign. If V->P recovers in C1 or C2, then V2_a should be treated as an over-conditioned lower bound, not a scientific contradiction of V1 or physical expectation.\n"""
    (settings.output_dir / "INTERPRETATION_GUARDRAILS_V2_C.md").write_text(text, encoding="utf-8")


def run_v2_c_sensitivity(settings: LeadLagScreenV2CSensitivitySettings | None = None) -> dict:
    settings = settings or LeadLagScreenV2CSensitivitySettings()
    ensure_dirs(settings.output_dir, settings.log_dir, settings.cache_dir)
    logger = setup_logger(settings.log_dir)
    started = datetime.now().isoformat(timespec="seconds")
    run_meta = {
        "layer": "lead_lag_screen/V2_c_sensitivity_targeted",
        "method": "PCMCI+ sensitivity reruns with ParCorr",
        "role": "Targeted audit of V2_a failure mode: V->P disappears in S3/T3; C2 runs all windows",
        "started_at": started,
        "settings": settings.to_jsonable_dict(),
        "important_boundaries": [
            "No fallback if tigramite is unavailable.",
            "V2_c does not overwrite V2_a and does not change V1 labels.",
            "C1 is targeted diagnostic, not a full-system replacement graph.",
            "C2 is representative-pool sensitivity, not final variable selection.",
            "Outputs are direct-edge controls/sensitivities, not pathway results.",
        ],
    }
    _write_json(settings.output_dir / "run_meta.json", run_meta)
    _write_json(settings.output_dir / "settings_summary.json", settings.to_jsonable_dict())
    _write_guardrails(settings)

    logger.info("Starting V2_c PCMCI+ sensitivity reruns")
    logger.info("Input index anomalies: %s", settings.input_index_anomalies)
    df = read_index_anomalies(settings.input_index_anomalies, settings.variables)
    years = np.asarray(sorted(df["year"].dropna().unique()), dtype=int)

    all_edges: List[pd.DataFrame] = []
    all_tau0: List[pd.DataFrame] = []
    all_meta: List[pd.DataFrame] = []
    failed_rows: List[dict] = []

    if settings.run_targeted_no_same_family_controls:
        e, z, m, f = _run_targeted_no_same_family_controls(settings, df, years, logger)
        all_edges.append(e)
        all_tau0.append(z)
        all_meta.append(m)
        failed_rows.extend(f)
    if settings.run_family_representative_standard:
        e, z, m, f = _run_family_representative_standard(settings, df, years, logger)
        all_edges.append(e)
        all_tau0.append(z)
        all_meta.append(m)
        failed_rows.extend(f)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(settings.output_dir / "runtime_failed_tasks.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(f"V2_c sensitivity failed for {len(failed_rows)} task(s). See runtime_failed_tasks.csv")

    edges_all = pd.concat([x for x in all_edges if x is not None and not x.empty], ignore_index=True) if all_edges else pd.DataFrame()
    tau0_all = pd.concat([x for x in all_tau0 if x is not None and not x.empty], ignore_index=True) if all_tau0 else pd.DataFrame()
    task_meta = pd.concat([x for x in all_meta if x is not None and not x.empty], ignore_index=True) if all_meta else pd.DataFrame()

    v1 = read_v1_evidence(settings.v1_evidence_tier_summary)
    v2a = _read_v2a_edges(settings)
    edges_aug, tau0_aug = _augment_with_v1_v2a(edges_all, tau0_all, v1, v2a)

    edges_aug.to_csv(settings.output_dir / "pcmci_plus_v2_c_edges_long.csv", index=False, encoding="utf-8-sig")
    tau0_aug.to_csv(settings.output_dir / "pcmci_plus_v2_c_tau0_contemporaneous.csv", index=False, encoding="utf-8-sig")
    task_meta.to_csv(settings.output_dir / "pcmci_plus_v2_c_task_meta.csv", index=False, encoding="utf-8-sig")

    win_sum = _summary_by_window(edges_aug, tau0_aug)
    fam_sum = _summary_by_family(edges_aug, tau0_aug)
    win_sum.to_csv(settings.output_dir / "pcmci_plus_v2_c_window_summary.csv", index=False, encoding="utf-8-sig")
    fam_sum.to_csv(settings.output_dir / "pcmci_plus_v2_c_family_summary.csv", index=False, encoding="utf-8-sig")

    vp_sum, vp_long = _direction_recovery(edges_aug, tau0_aug, v1, "V", "P")
    hp_sum, hp_long = _direction_recovery(edges_aug, tau0_aug, v1, "H", "P")
    vp_sum.to_csv(settings.output_dir / "v_to_p_recovery_audit.csv", index=False, encoding="utf-8-sig")
    hp_sum.to_csv(settings.output_dir / "h_to_p_recovery_audit.csv", index=False, encoding="utf-8-sig")
    vp_long.to_csv(settings.output_dir / "v_to_p_recovery_long.csv", index=False, encoding="utf-8-sig")
    hp_long.to_csv(settings.output_dir / "h_to_p_recovery_long.csv", index=False, encoding="utf-8-sig")

    summary = {
        "layer": "lead_lag_screen/V2_c_sensitivity_targeted",
        "output_tag": settings.output_tag,
        "c1_targeted_windows": list(settings.c1_targeted_windows),
        "c1_targeted_family_pairs": [list(x) for x in settings.targeted_family_pairs],
        "c2_windows": list(settings.windows.keys()),
        "method": "PCMCI+ with ParCorr sensitivity reruns",
        "input": str(settings.input_index_anomalies),
        "n_windows": int(len(settings.windows)),
        "n_years": int(len(years)),
        "n_variants_with_outputs": int(edges_aug["variant_id"].nunique()) if not edges_aug.empty else 0,
        "n_lagged_tests": int(len(edges_aug)),
        "n_lagged_graph_selected": int(edges_aug["pcmci_graph_selected"].sum()) if not edges_aug.empty else 0,
        "n_lagged_supported_window_fdr": int(edges_aug["pcmci_plus_supported"].sum()) if not edges_aug.empty else 0,
        "n_tau0_tests": int(len(tau0_aug)),
        "n_tau0_supported": int(tau0_aug["tau0_supported"].sum()) if (not tau0_aug.empty and "tau0_supported" in tau0_aug.columns) else 0,
        "v_to_p_summary_file": "v_to_p_recovery_audit.csv",
        "h_to_p_summary_file": "h_to_p_recovery_audit.csv",
        "not_pathway_result": True,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(settings.output_dir / "summary.json", summary)
    logger.info("Finished V2_c sensitivity: %s", summary)
    return summary
