"""
V9 peak-all-windows clean baseline.

Purpose
-------
V9 is a peak-only extraction/generalization of the audited V8/V7 peak layer.
It keeps the original V7-z peak implementation semantics intact while expanding
from the W045-only V8 verification run to the four currently accepted/significant
windows used by the project mainline:

    W045, W081, W113, W160

This module is intentionally not a new detector and not a state/growth method.
It only orchestrates peak-layer outputs for all accepted windows and writes
cross-window peak timing summaries.

Explicitly excluded
-------------------
- S_dist / S_pattern state curves;
- G_dist / G_pattern growth curves;
- pairwise state/growth differences;
- catch-up, rollback, multi-stage, or process_a diagnostics;
- coordinate-meaning audits or final scientific claims.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import os
import sys
import time

import numpy as np
import pandas as pd

V9_VERSION = "v9_peak_all_windows_a"
OUTPUT_TAG = "peak_all_windows_v9_a"
DEFAULT_ACCEPTED_WINDOWS = ["W045", "W081", "W113", "W160"]
DEFAULT_ACCEPTED_WINDOWS_CSV = ",".join(DEFAULT_ACCEPTED_WINDOWS)
EXCLUDED_MAINLINE_WINDOWS = [
    {
        "window_id": "W135",
        "anchor_day": 135,
        "included_in_v9": False,
        "exclusion_reason": "not_in_strict_accepted_95pct_window_set; previously dropped from mainline",
    }
]
V8_PEAK_TAG = "peak_only_v8_a"
V7_HOTFIX06_TAG = "accepted_windows_multi_object_prepost_v7_z_multiwin_a_hotfix06_w45_profile_order"


def _stage_root_from_this_file() -> Path:
    # .../stage_partition/V9/src/stage_partition_v9/peak_all_windows_v9_a.py
    return Path(__file__).resolve().parents[3]


def _import_v7_module(stage_root: Optional[Path] = None):
    stage_root = stage_root or _stage_root_from_this_file()
    v7_src = stage_root / "V7" / "src"
    if not v7_src.exists():
        raise FileNotFoundError(
            f"Cannot find V7 source directory: {v7_src}. "
            "V9 peak-all-windows extraction requires the existing V7 code tree."
        )
    if str(v7_src) not in sys.path:
        sys.path.insert(0, str(v7_src))
    from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as v7multi
    return v7multi


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_v7_cfg_for_v9(v7multi) -> object:
    """Build a V7 MultiWinConfig but force V9 peak-only all-significant-window defaults.

    V9 owns the run-window selection.  V7 helper defaults are intentionally not
    allowed to keep the run at W045.  Method-level V7 semantics are retained;
    only the orchestrated target windows and output boundary are changed.
    """
    cfg = v7multi.MultiWinConfig.from_env()

    # V9 data/source aliases.
    if os.environ.get("V9_PEAK_ACCEPTED_WINDOW_REGISTRY"):
        cfg.accepted_window_registry = os.environ["V9_PEAK_ACCEPTED_WINDOW_REGISTRY"]
        cfg.window_source = "registry"
    if os.environ.get("V9_PEAK_WINDOW_SOURCE"):
        cfg.window_source = os.environ["V9_PEAK_WINDOW_SOURCE"].strip().lower()
    if os.environ.get("V9_PEAK_SMOOTHED_FIELDS"):
        cfg.smoothed_fields_path = os.environ["V9_PEAK_SMOOTHED_FIELDS"]

    # V9 default run selection: the four currently accepted/significant windows.
    cfg.window_mode = "list"
    cfg.target_windows = DEFAULT_ACCEPTED_WINDOWS_CSV
    if os.environ.get("V9_PEAK_WINDOW_MODE"):
        cfg.window_mode = os.environ["V9_PEAK_WINDOW_MODE"].strip().lower()
    if os.environ.get("V9_PEAK_TARGET_WINDOWS"):
        cfg.target_windows = os.environ["V9_PEAK_TARGET_WINDOWS"].strip()

    # V9 bootstrap aliases.
    if os.environ.get("V9_PEAK_N_BOOTSTRAP"):
        cfg.bootstrap_n = int(os.environ["V9_PEAK_N_BOOTSTRAP"])
    if os.environ.get("V9_PEAK_DEBUG_N_BOOTSTRAP"):
        cfg.bootstrap_n = int(os.environ["V9_PEAK_DEBUG_N_BOOTSTRAP"])
    if os.environ.get("V9_PEAK_LOG_EVERY_BOOTSTRAP"):
        cfg.log_every_bootstrap = int(os.environ["V9_PEAK_LOG_EVERY_BOOTSTRAP"])

    # V9 is peak-only.  These flags prevent accidental state/process side paths.
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    return cfg


def _default_smoothed_path(v9_root: Path) -> Path:
    # v9_root = D:/easm_project01/stage_partition/V9 => project root is parents[1]
    return v9_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _build_peak_relation_summary(peak_order: pd.DataFrame, sync: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    """Lightweight merge of original peak-layer outputs; no new scientific gate."""
    if peak_order is None or peak_order.empty:
        return pd.DataFrame()
    out = peak_order.copy()
    keep_sync = [c for c in [
        "window_id", "object_A", "object_B", "tau_sync_primary", "P_within_tau_q50",
        "P_within_tau_q75", "P_within_tau_q90", "synchrony_decision", "tau_quality_flag"
    ] if sync is not None and not sync.empty and c in sync.columns]
    if keep_sync:
        out = out.merge(sync[keep_sync], on=["window_id", "object_A", "object_B"], how="left")
    keep_overlap = [c for c in [
        "window_id", "object_A", "object_B", "overlap_days", "overlap_fraction",
        "A_peak_inside_B_window", "B_peak_inside_A_window", "both_peaks_inside_system_window",
        "window_overlap_decision"
    ] if overlap is not None and not overlap.empty and c in overlap.columns]
    if keep_overlap:
        out = out.merge(overlap[keep_overlap], on=["window_id", "object_A", "object_B"], how="left")
    out["v9_role"] = "peak_layer_summary_no_state_growth"
    out["interpretation_boundary"] = "peak timing only; do not infer state/growth/process relations"
    return out


def _safe_num(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _compare_numeric(a, b, tol: float = 1.0e-9) -> str:
    try:
        aa = float(a); bb = float(b)
    except Exception:
        return "same" if str(a) == str(b) else "different"
    if np.isnan(aa) and np.isnan(bb):
        return "same"
    return "same" if abs(aa - bb) <= tol else "different"


def _window_day_audit_for_profile(prof: np.ndarray, scope, object_name: str) -> List[dict]:
    """Audit finite day domain for analysis and detector windows.

    prof is expected as year x day x lat/profile.  A day is finite if it has at
    least one finite value across years/profile coordinates.  This audit is not
    used to alter original V7 peak logic; it records boundary NaN exposure for
    peak interpretation and debugging.
    """
    arr = np.asarray(prof)
    if arr.ndim < 2:
        day_finite = np.isfinite(arr)
    else:
        axes = tuple(i for i in range(arr.ndim) if i != 1)
        day_finite = np.any(np.isfinite(arr), axis=axes)
    rows = []
    domains = [
        ("analysis", int(scope.analysis_start), int(scope.analysis_end)),
        ("detector", int(scope.detector_search_start), int(scope.detector_search_end)),
        ("system_window", int(scope.system_window_start), int(scope.system_window_end)),
    ]
    n_days = int(day_finite.shape[0]) if hasattr(day_finite, "shape") else 0
    for domain_name, start, end in domains:
        s = max(start, 0)
        e = min(end, n_days - 1)
        if e < s:
            rows.append({
                "window_id": scope.window_id,
                "object": object_name,
                "domain": domain_name,
                "nominal_start_day": start,
                "nominal_end_day": end,
                "finite_start_day": np.nan,
                "finite_end_day": np.nan,
                "n_nominal_days": max(0, end - start + 1),
                "n_finite_days": 0,
                "leading_nan_days": np.nan,
                "trailing_nan_days": np.nan,
                "internal_nan_days": np.nan,
                "valid_day_fraction": 0.0,
                "boundary_nan_warning": "invalid_nominal_domain",
            })
            continue
        sub = np.asarray(day_finite[s:e+1], dtype=bool)
        finite_idx = np.where(sub)[0]
        n_nominal = int(e - s + 1)
        if finite_idx.size == 0:
            rows.append({
                "window_id": scope.window_id,
                "object": object_name,
                "domain": domain_name,
                "nominal_start_day": start,
                "nominal_end_day": end,
                "finite_start_day": np.nan,
                "finite_end_day": np.nan,
                "n_nominal_days": n_nominal,
                "n_finite_days": 0,
                "leading_nan_days": n_nominal,
                "trailing_nan_days": n_nominal,
                "internal_nan_days": 0,
                "valid_day_fraction": 0.0,
                "boundary_nan_warning": "all_nan_domain",
            })
            continue
        finite_start = s + int(finite_idx[0])
        finite_end = s + int(finite_idx[-1])
        leading_nan = int(finite_idx[0])
        trailing_nan = int((n_nominal - 1) - finite_idx[-1])
        internal = sub[finite_idx[0]:finite_idx[-1]+1]
        internal_nan = int((~internal).sum())
        warning = "none"
        if leading_nan or trailing_nan:
            warning = "boundary_nan_present"
        if internal_nan:
            warning = "internal_nan_present" if warning == "none" else warning + "+internal_nan_present"
        rows.append({
            "window_id": scope.window_id,
            "object": object_name,
            "domain": domain_name,
            "nominal_start_day": start,
            "nominal_end_day": end,
            "finite_start_day": finite_start,
            "finite_end_day": finite_end,
            "n_nominal_days": n_nominal,
            "n_finite_days": int(sub.sum()),
            "leading_nan_days": leading_nan,
            "trailing_nan_days": trailing_nan,
            "internal_nan_days": internal_nan,
            "valid_day_fraction": float(sub.mean()),
            "boundary_nan_warning": warning,
        })
    return rows


def _write_v8_w045_peak_regression_audit(v9_root: Path, out_cross: Path, per_window_results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Compare V9 W045 outputs against V8 peak-only W045 outputs when available."""
    stage_root = v9_root.parent
    v8_base = stage_root / "V8" / "outputs" / V8_PEAK_TAG / "per_window" / "W045"
    wid = "W045"
    rows: List[dict] = []
    if wid not in per_window_results:
        _safe_to_csv(pd.DataFrame([{
            "audit_scope": "v9_vs_v8_w045_peak_regression",
            "status": "v9_w045_not_run",
            "v8_reference_dir": str(v8_base),
        }]), out_cross / "v9_vs_v8_w045_peak_regression_audit.csv")
        return
    if not v8_base.exists():
        _safe_to_csv(pd.DataFrame([{
            "audit_scope": "v9_vs_v8_w045_peak_regression",
            "status": "v8_reference_missing",
            "v8_reference_dir": str(v8_base),
        }]), out_cross / "v9_vs_v8_w045_peak_regression_audit.csv")
        return

    file_map = {
        "main_window_selection": "main_window_selection_W045.csv",
        "bootstrap_selected_peak_days": "bootstrap_selected_peak_days_W045.csv",
        "timing_resolution_audit": "timing_resolution_audit_W045.csv",
        "tau_sync_estimate": "tau_sync_estimate_W045.csv",
        "pairwise_peak_order_test": "pairwise_peak_order_test_W045.csv",
        "pairwise_synchrony_equivalence_test": "pairwise_synchrony_equivalence_test_W045.csv",
        "pairwise_window_overlap": "pairwise_window_overlap_W045.csv",
    }
    key_cols = {
        "main_window_selection": ["window_id", "object"],
        "bootstrap_selected_peak_days": ["window_id", "object", "bootstrap_id"],
        "timing_resolution_audit": ["window_id", "object"],
        "tau_sync_estimate": ["window_id"],
        "pairwise_peak_order_test": ["window_id", "object_A", "object_B"],
        "pairwise_synchrony_equivalence_test": ["window_id", "object_A", "object_B"],
        "pairwise_window_overlap": ["window_id", "object_A", "object_B"],
    }
    compare_cols = {
        "main_window_selection": ["selected_peak_day", "selected_window_start", "selected_window_end", "support_class", "selected_role"],
        "bootstrap_selected_peak_days": ["selected_peak_day"],
        "timing_resolution_audit": ["observed_peak_day", "bootstrap_peak_median", "bootstrap_peak_q025", "bootstrap_peak_q975", "abs_error_q75", "support_class"],
        "tau_sync_estimate": ["tau_sync_q50", "tau_sync_q75", "tau_sync_q90", "tau_sync_primary", "tau_quality_flag"],
        "pairwise_peak_order_test": ["delta_observed", "delta_median", "delta_q025", "delta_q975", "P_A_earlier", "P_B_earlier", "P_same_day", "peak_order_decision"],
        "pairwise_synchrony_equivalence_test": ["tau_sync_primary", "delta_observed", "delta_median", "delta_q025", "delta_q975", "P_within_tau_q75", "synchrony_decision", "tau_quality_flag"],
        "pairwise_window_overlap": ["overlap_days", "overlap_fraction", "window_overlap_decision", "both_peaks_inside_system_window"],
    }
    res = per_window_results[wid]
    for logical, fname in file_map.items():
        v9_df = res.get(logical, pd.DataFrame())
        v8_file = v8_base / fname
        if not v8_file.exists():
            rows.append({"table": logical, "status": "v8_file_missing", "v8_file": str(v8_file)})
            continue
        v8_df = pd.read_csv(v8_file)
        if v9_df is None or v9_df.empty:
            rows.append({"table": logical, "status": "v9_table_empty", "v8_file": str(v8_file)})
            continue
        keys = [c for c in key_cols.get(logical, []) if c in v9_df.columns and c in v8_df.columns]
        if not keys:
            rows.append({"table": logical, "status": "no_common_keys", "v8_file": str(v8_file)})
            continue
        merged = v9_df.merge(v8_df, on=keys, how="outer", suffixes=("_v9", "_v8"), indicator=True)
        mismatch = int((merged["_merge"] != "both").sum())
        diff_count = 0
        col_diffs: Dict[str, int] = {}
        for col in compare_cols.get(logical, []):
            c9, c8 = f"{col}_v9", f"{col}_v8"
            if c9 not in merged.columns or c8 not in merged.columns:
                continue
            d = 0
            for _, r in merged[merged["_merge"] == "both"].iterrows():
                if _compare_numeric(r[c9], r[c8]) != "same":
                    d += 1
            if d:
                col_diffs[col] = d
                diff_count += d
        status = "pass" if mismatch == 0 and diff_count == 0 else "difference_found"
        rows.append({
            "window_id": wid,
            "table": logical,
            "status": status,
            "n_v9_rows": int(len(v9_df)),
            "n_v8_rows": int(len(v8_df)),
            "n_key_mismatch_rows": mismatch,
            "n_value_differences": diff_count,
            "columns_with_differences": json.dumps(col_diffs, ensure_ascii=False),
            "v8_file": str(v8_file),
        })
    _safe_to_csv(pd.DataFrame(rows), out_cross / "v9_vs_v8_w045_peak_regression_audit.csv")


def _write_v7_peak_regression_audit(v9_root: Path, out_cross: Path, per_window_results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Compare V9 outputs against existing V7 hotfix06 peak tables, if present."""
    stage_root = v9_root.parent
    v7_base = stage_root / "V7" / "outputs" / V7_HOTFIX06_TAG / "per_window"
    if not v7_base.exists():
        _safe_to_csv(pd.DataFrame([{
            "audit_scope": "v7_peak_regression",
            "status": "v7_reference_missing",
            "v7_reference_dir": str(v7_base),
            "note": "No comparison was made. This does not affect V9 peak outputs.",
        }]), out_cross / "v9_vs_v7_hotfix06_peak_regression_audit.csv")
        return
    rows: List[dict] = []
    # Reuse a compact comparison set; V7 may only have W045 hotfix outputs.
    for wid, res in per_window_results.items():
        v7_dir = v7_base / wid
        for logical, fname in {
            "main_window_selection": f"main_window_selection_{wid}.csv",
            "bootstrap_selected_peak_days": f"bootstrap_selected_peak_days_{wid}.csv",
            "timing_resolution_audit": f"timing_resolution_audit_{wid}.csv",
            "tau_sync_estimate": f"tau_sync_estimate_{wid}.csv",
            "pairwise_peak_order_test": f"pairwise_peak_order_test_{wid}.csv",
            "pairwise_synchrony_equivalence_test": f"pairwise_synchrony_equivalence_test_{wid}.csv",
            "pairwise_window_overlap": f"pairwise_window_overlap_{wid}.csv",
        }.items():
            v7_file = v7_dir / fname
            if not v7_file.exists():
                rows.append({"window_id": wid, "table": logical, "status": "v7_file_missing", "v7_file": str(v7_file)})
                continue
            # Detailed numeric audit is covered by W045 V8 comparison; here flag availability/readability.
            try:
                v7_df = pd.read_csv(v7_file)
                v9_df = res.get(logical, pd.DataFrame())
                rows.append({
                    "window_id": wid,
                    "table": logical,
                    "status": "reference_available",
                    "n_v9_rows": int(len(v9_df)) if v9_df is not None else 0,
                    "n_v7_rows": int(len(v7_df)),
                    "v7_file": str(v7_file),
                })
            except Exception as exc:
                rows.append({"window_id": wid, "table": logical, "status": "v7_read_error", "error": str(exc), "v7_file": str(v7_file)})
    _safe_to_csv(pd.DataFrame(rows), out_cross / "v9_vs_v7_hotfix06_peak_regression_audit.csv")


def _make_accepted_windows_used_table(scopes: List[object], run_scopes: List[object]) -> pd.DataFrame:
    selected_ids = {s.window_id for s in run_scopes}
    rows: List[dict] = []
    for s in scopes:
        rows.append({
            "window_id": s.window_id,
            "anchor_day": s.anchor_day,
            "system_window_start": s.system_window_start,
            "system_window_end": s.system_window_end,
            "detector_search_start": s.detector_search_start,
            "detector_search_end": s.detector_search_end,
            "included_in_v9": s.window_id in selected_ids,
            "source": "v7_accepted_window_scope_registry",
            "exclusion_reason": "" if s.window_id in selected_ids else "not_selected_by_v9_target_window_list",
        })
    existing = {r["window_id"] for r in rows}
    for r in EXCLUDED_MAINLINE_WINDOWS:
        if r["window_id"] not in existing:
            rows.append({
                "window_id": r["window_id"],
                "anchor_day": r["anchor_day"],
                "system_window_start": np.nan,
                "system_window_end": np.nan,
                "detector_search_start": np.nan,
                "detector_search_end": np.nan,
                "included_in_v9": False,
                "source": "mainline_exclusion_record",
                "exclusion_reason": r["exclusion_reason"],
            })
    return pd.DataFrame(rows)


def _cross_window_relation_matrix(peak_order: pd.DataFrame, sync: pd.DataFrame) -> pd.DataFrame:
    if peak_order is None or peak_order.empty:
        return pd.DataFrame()
    out = peak_order.copy()
    keep = [c for c in ["window_id", "object_A", "object_B", "synchrony_decision"] if sync is not None and not sync.empty and c in sync.columns]
    if keep:
        out = out.merge(sync[keep], on=["window_id", "object_A", "object_B"], how="left")
    labels = []
    for _, r in out.iterrows():
        order = str(r.get("peak_order_decision", ""))
        syn = str(r.get("synchrony_decision", ""))
        if "supported" in order or "tendency" in order:
            lab = order
        elif "supported" in syn or "tendency" in syn:
            lab = syn
        else:
            lab = "peak_relation_unresolved"
        labels.append(lab)
    out["peak_relation_label"] = labels
    cols = [c for c in [
        "window_id", "object_A", "object_B", "delta_observed", "peak_order_decision",
        "synchrony_decision", "peak_relation_label"
    ] if c in out.columns]
    return out[cols]


def _cross_window_stage_summary(peak_order: pd.DataFrame, sync: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    windows = sorted(set(list(peak_order.get("window_id", [])) + list(sync.get("window_id", []))))
    for wid in windows:
        po = peak_order[peak_order["window_id"] == wid] if peak_order is not None and not peak_order.empty else pd.DataFrame()
        sy = sync[sync["window_id"] == wid] if sync is not None and not sync.empty else pd.DataFrame()
        row = {"window_id": wid}
        if not po.empty and "peak_order_decision" in po.columns:
            vc = po["peak_order_decision"].value_counts(dropna=False)
            for k, v in vc.items():
                row[f"peak_order__{k}"] = int(v)
        if not sy.empty and "synchrony_decision" in sy.columns:
            vc = sy["synchrony_decision"].value_counts(dropna=False)
            for k, v in vc.items():
                row[f"synchrony__{k}"] = int(v)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_peak_all_windows_summary(path: Path, run_scopes: List[object], concat: Dict[str, pd.DataFrame], cfg: object) -> None:
    lines = [
        "# V9 peak-all-windows summary",
        "",
        f"version: `{V9_VERSION}`",
        f"output_tag: `{OUTPUT_TAG}`",
        f"windows_processed: {len(run_scopes)}",
        f"run_mode: {cfg.window_mode}; targets: {cfg.target_windows}",
        "",
        "## Method boundary",
        "- V9 is peak-only: it extracts object transition-event timing across accepted windows.",
        "- It reuses the audited V7/V8 peak-layer semantics and does not compute state/growth/process outputs.",
        "- It should be interpreted as an event-time skeleton, not a state-process explanation.",
        "",
        "## Windows processed",
    ]
    for s in run_scopes:
        lines.append(f"- {s.window_id}: anchor day {s.anchor_day}; system day{s.system_window_start}-{s.system_window_end}; detector day{s.detector_search_start}-{s.detector_search_end}")
    lines += ["", "## Mainline exclusion", "- W135 is not included in V9 because it is not part of the strict accepted 95% window set.", "", "## Peak relation counts"]
    po = concat.get("pairwise_peak_order_test", pd.DataFrame())
    if not po.empty and "peak_order_decision" in po.columns:
        for k, v in po["peak_order_decision"].value_counts(dropna=False).items():
            lines.append(f"- peak_order_decision={k}: {int(v)}")
    sync = concat.get("pairwise_synchrony_equivalence_test", pd.DataFrame())
    if not sync.empty and "synchrony_decision" in sync.columns:
        for k, v in sync["synchrony_decision"].value_counts(dropna=False).items():
            lines.append(f"- synchrony_decision={k}: {int(v)}")
    lines += [
        "",
        "## Forbidden interpretations",
        "- Do not infer state-front, growth-front, catch-up, rollback, multi-stage, or pre-post process claims from V9 alone.",
        "- Peak order is event timing only; it is not causality.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_peak_all_windows_v9_a(v9_root: Path | str) -> None:
    v9_root = Path(v9_root)
    stage_root = v9_root.parent
    v7_root = stage_root / "V7"
    v7multi = _import_v7_module(stage_root)
    cfg = _make_v7_cfg_for_v9(v7multi)

    out_root = _ensure_dir(v9_root / "outputs" / OUTPUT_TAG)
    out_cross = _ensure_dir(out_root / "cross_window")
    out_per = _ensure_dir(out_root / "per_window")
    log_dir = _ensure_dir(v9_root / "logs" / OUTPUT_TAG)
    t0 = time.time()

    _log("[1/7] Load accepted/significant windows using original V7 helper")
    wins = v7multi._load_accepted_windows(v7_root, out_cross, cfg)
    scopes, validity = v7multi._build_window_scopes(wins, cfg)
    run_scopes, run_scope_audit = v7multi._filter_scopes_for_run(scopes, cfg)
    accepted_used = _make_accepted_windows_used_table(scopes, run_scopes)
    _safe_to_csv(pd.DataFrame([asdict(s) for s in scopes]), out_cross / "window_scope_registry_v9_peak_all_windows_a.csv")
    _safe_to_csv(validity, out_cross / "window_scope_validity_audit_v9_peak_all_windows_a.csv")
    _safe_to_csv(run_scope_audit, out_cross / "run_window_selection_audit_v9_peak_all_windows_a.csv")
    _safe_to_csv(pd.DataFrame([asdict(s) for s in run_scopes]), out_cross / "run_window_scope_registry_v9_peak_all_windows_a.csv")
    _safe_to_csv(accepted_used, out_cross / "accepted_windows_used_v9_a.csv")

    _log("[2/7] Load smoothed fields using original V7 clean loader")
    smoothed = Path(cfg.smoothed_fields_path) if cfg.smoothed_fields_path else _default_smoothed_path(v9_root)
    if not smoothed.exists():
        smoothed = _default_smoothed_path(v9_root)
    if not smoothed.exists():
        raise FileNotFoundError(
            f"smoothed_fields.npz not found: {smoothed}. "
            "Set V9_PEAK_SMOOTHED_FIELDS or V7_MULTI_SMOOTHED_FIELDS to the correct file."
        )
    fields, audit = v7multi.clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    _safe_to_csv(audit, out_cross / "input_key_audit_v9_peak_all_windows_a.csv")

    _write_json({
        "version": V9_VERSION,
        "output_tag": OUTPUT_TAG,
        "source_v7_module_version": getattr(v7multi, "VERSION", "unknown"),
        "source_v7_output_tag": getattr(v7multi, "OUTPUT_TAG", "unknown"),
        "config": asdict(cfg),
        "default_accepted_windows": DEFAULT_ACCEPTED_WINDOWS,
        "excluded_mainline_windows": EXCLUDED_MAINLINE_WINDOWS,
        "smoothed_fields": str(smoothed),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "boundary": "peak-only; no state/growth/prepost/process layers",
    }, out_root / "run_meta.json")

    _log("[3/7] Build object profiles with original V7 clean helpers")
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    object_rows = []
    for spec in v7multi.clean.OBJECT_SPECS:
        arr = v7multi.clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (prof, target_lat, weights)
        object_rows.append({
            **asdict(spec),
            "profile_shape": str(prof.shape),
            "target_lat_min": float(np.nanmin(target_lat)),
            "target_lat_max": float(np.nanmax(target_lat)),
            "v9_role": "peak_detector_profile_input",
        })
    _safe_to_csv(pd.DataFrame(object_rows), out_cross / "object_registry_v9_peak_all_windows_a.csv")

    _log("[4/7] Run original V7 profile detector + peak bootstrap for all selected windows")
    per_window_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    all_parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in [
        "raw_profile_detector_scores", "object_profile_window_registry", "main_window_selection",
        "selected_peak_delta", "bootstrap_selected_peak_days", "timing_resolution_audit",
        "tau_sync_estimate", "pairwise_peak_order_test", "pairwise_synchrony_equivalence_test",
        "pairwise_window_overlap", "peak_relation_summary", "peak_valid_day_audit",
    ]}

    for idx, scope in enumerate(run_scopes, start=1):
        _log(f"  [{idx}/{len(run_scopes)}] {scope.window_id}")
        out_win = _ensure_dir(out_per / scope.window_id)
        _safe_to_csv(pd.DataFrame([asdict(scope)]), out_win / f"window_scope_{scope.window_id}.csv")

        valid_rows: List[dict] = []
        for obj, (prof, _lat, _w) in profiles.items():
            valid_rows.extend(_window_day_audit_for_profile(prof, scope, obj))
        valid_audit_df = pd.DataFrame(valid_rows)
        _safe_to_csv(valid_audit_df, out_win / f"peak_valid_day_audit_{scope.window_id}.csv")

        score_df, cand_df, selection_df, selected_delta_df, boot_peak_days_df = v7multi._run_detector_and_bootstrap(profiles, scope, cfg)
        timing_audit_df, tau_df = v7multi._estimate_timing_resolution(selection_df, boot_peak_days_df, cfg, scope)
        peak_order_df = v7multi._pairwise_peak_order(selection_df, boot_peak_days_df, scope)
        sync_df = v7multi._pairwise_synchrony(peak_order_df, boot_peak_days_df, tau_df, scope)
        overlap_df = v7multi._pairwise_window_overlap(selection_df, scope)
        peak_summary_df = _build_peak_relation_summary(peak_order_df, sync_df, overlap_df)
        result_map = {
            "raw_profile_detector_scores": score_df,
            "object_profile_window_registry": cand_df,
            "main_window_selection": selection_df,
            "selected_peak_delta": selected_delta_df,
            "bootstrap_selected_peak_days": boot_peak_days_df,
            "timing_resolution_audit": timing_audit_df,
            "tau_sync_estimate": tau_df,
            "pairwise_peak_order_test": peak_order_df,
            "pairwise_synchrony_equivalence_test": sync_df,
            "pairwise_window_overlap": overlap_df,
            "peak_relation_summary": peak_summary_df,
            "peak_valid_day_audit": valid_audit_df,
        }
        file_names = {
            "raw_profile_detector_scores": f"raw_profile_detector_scores_{scope.window_id}.csv",
            "object_profile_window_registry": f"object_profile_window_registry_{scope.window_id}.csv",
            "main_window_selection": f"main_window_selection_{scope.window_id}.csv",
            "selected_peak_delta": f"selected_peak_delta_{scope.window_id}.csv",
            "bootstrap_selected_peak_days": f"bootstrap_selected_peak_days_{scope.window_id}.csv",
            "timing_resolution_audit": f"timing_resolution_audit_{scope.window_id}.csv",
            "tau_sync_estimate": f"tau_sync_estimate_{scope.window_id}.csv",
            "pairwise_peak_order_test": f"pairwise_peak_order_test_{scope.window_id}.csv",
            "pairwise_synchrony_equivalence_test": f"pairwise_synchrony_equivalence_test_{scope.window_id}.csv",
            "pairwise_window_overlap": f"pairwise_window_overlap_{scope.window_id}.csv",
            "peak_relation_summary": f"peak_relation_summary_{scope.window_id}.csv",
            "peak_valid_day_audit": f"peak_valid_day_audit_{scope.window_id}.csv",
        }
        for logical, df in result_map.items():
            _safe_to_csv(df, out_win / file_names[logical])
            if df is not None and not df.empty:
                all_parts[logical].append(df)

        # User-facing aliases for V9 peak-only terminology.
        _safe_to_csv(selection_df, out_win / f"object_peak_registry_{scope.window_id}.csv")
        _safe_to_csv(timing_audit_df, out_win / f"object_peak_bootstrap_{scope.window_id}.csv")
        _safe_to_csv(boot_peak_days_df, out_win / f"bootstrap_selected_peak_days_{scope.window_id}.csv")
        _safe_to_csv(peak_order_df, out_win / f"pairwise_peak_order_{scope.window_id}.csv")
        _safe_to_csv(sync_df, out_win / f"pairwise_peak_synchrony_{scope.window_id}.csv")
        _safe_to_csv(peak_summary_df, out_win / f"peak_relation_summary_{scope.window_id}.csv")
        (out_win / f"peak_relation_summary_{scope.window_id}.md").write_text(
            f"# Peak relation summary {scope.window_id}\n\n"
            "This is peak timing only. It excludes state/growth/process interpretation.\n\n"
            f"Objects: {', '.join(sorted(selection_df['object'].astype(str).unique())) if 'object' in selection_df.columns else 'unknown'}\n"
            f"Bootstrap N: {cfg.bootstrap_n}\n",
            encoding="utf-8",
        )
        _write_json({
            "version": V9_VERSION,
            "window_id": scope.window_id,
            "bootstrap_n": int(cfg.bootstrap_n),
            "boundary": "peak-only; no state/growth/process outputs",
            "scope": asdict(scope),
        }, out_win / f"window_run_meta_{scope.window_id}.json")
        per_window_results[scope.window_id] = result_map

    _log("[5/7] Write cross-window peak-only outputs")
    concat = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in all_parts.items()}
    cross_names = {
        "raw_profile_detector_scores": "raw_profile_detector_scores_all_windows.csv",
        "object_profile_window_registry": "object_profile_window_registry_all_windows.csv",
        "main_window_selection": "main_window_selection_all_windows.csv",
        "selected_peak_delta": "selected_peak_delta_all_windows.csv",
        "bootstrap_selected_peak_days": "bootstrap_selected_peak_days_all_windows.csv",
        "timing_resolution_audit": "timing_resolution_audit_all_windows.csv",
        "tau_sync_estimate": "tau_sync_estimate_all_windows.csv",
        "pairwise_peak_order_test": "pairwise_peak_order_test_all_windows.csv",
        "pairwise_synchrony_equivalence_test": "pairwise_synchrony_equivalence_test_all_windows.csv",
        "pairwise_window_overlap": "pairwise_window_overlap_all_windows.csv",
        "peak_relation_summary": "peak_relation_summary_all_windows.csv",
        "peak_valid_day_audit": "peak_valid_day_audit_all_windows.csv",
    }
    for logical, filename in cross_names.items():
        _safe_to_csv(concat[logical], out_cross / filename)
    _safe_to_csv(concat["main_window_selection"], out_cross / "cross_window_object_peak_registry.csv")
    _safe_to_csv(concat["pairwise_peak_order_test"], out_cross / "cross_window_pairwise_peak_order.csv")
    _safe_to_csv(concat["pairwise_synchrony_equivalence_test"], out_cross / "cross_window_pairwise_peak_synchrony.csv")
    _safe_to_csv(_cross_window_relation_matrix(concat["pairwise_peak_order_test"], concat["pairwise_synchrony_equivalence_test"]), out_cross / "cross_window_peak_relation_matrix_by_window.csv")
    _safe_to_csv(_cross_window_stage_summary(concat["pairwise_peak_order_test"], concat["pairwise_synchrony_equivalence_test"]), out_cross / "cross_window_peak_stage_summary.csv")

    _log("[6/7] Write regression audits")
    _write_v8_w045_peak_regression_audit(v9_root, out_cross, per_window_results)
    _write_v7_peak_regression_audit(v9_root, out_cross, per_window_results)
    _write_peak_all_windows_summary(out_cross / "peak_all_windows_v9_a_summary.md", run_scopes, concat, cfg)

    _log("[7/7] Done")
    elapsed = time.time() - t0
    _write_json({
        "version": V9_VERSION,
        "elapsed_seconds": elapsed,
        "n_windows_available": len(scopes),
        "n_windows_processed": len(run_scopes),
        "windows_processed": [s.window_id for s in run_scopes],
        "default_accepted_windows": DEFAULT_ACCEPTED_WINDOWS,
        "excluded_mainline_windows": EXCLUDED_MAINLINE_WINDOWS,
        "bootstrap_n": int(cfg.bootstrap_n),
        "output_root": str(out_root),
        "boundary": "peak-only; state/growth/process layers excluded",
        "source_v7_module_version": getattr(v7multi, "VERSION", "unknown"),
        "source_v7_output_tag": getattr(v7multi, "OUTPUT_TAG", "unknown"),
    }, out_root / "summary.json")
    (log_dir / "last_run.txt").write_text(
        f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')} output={out_root}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    run_peak_all_windows_v9_a(Path(__file__).resolve().parents[2])
