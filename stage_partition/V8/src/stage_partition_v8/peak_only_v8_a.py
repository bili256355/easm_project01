"""
V8 peak-only clean baseline, extracted from the audited V7-z multiwindow peak layer.

Purpose
-------
This module is intentionally *not* a new detector. It is a thin, peak-only
orchestration wrapper around the already-audited V7-z multiwindow implementation:

    V7/src/stage_partition_v7/accepted_windows_multi_object_prepost_v7_z_multiwin_a.py

It keeps the V7 peak-layer implementation semantics intact, while excluding all
state/growth/pre-post/process interpretation layers from V8 outputs.

Included in V8-a
----------------
- hardcoded/registry accepted-window loading through the original V7 helper;
- V7-z raw/profile object-window detector;
- paired-year peak bootstrap;
- timing-resolution audit and tau_sync estimate;
- pairwise peak-order test;
- pairwise peak synchrony/equivalence test;
- pairwise selected-window overlap audit;
- regression audit against the existing V7 hotfix06 peak outputs when available.

Explicitly excluded
-------------------
- S_dist / S_pattern state curves;
- G_dist / G_pattern growth curves;
- pairwise state/growth differences;
- pre-post/process summaries;
- evidence gates / final claims;
- process_a ΔS/ΔG structure diagnostics;
- 2D mirror and Je audit branches.
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

V8_VERSION = "v8_peak_only_a"
OUTPUT_TAG = "peak_only_v8_a"
V7_HOTFIX06_TAG = "accepted_windows_multi_object_prepost_v7_z_multiwin_a_hotfix06_w45_profile_order"


def _stage_root_from_this_file() -> Path:
    # .../stage_partition/V8/src/stage_partition_v8/peak_only_v8_a.py
    return Path(__file__).resolve().parents[3]


def _import_v7_module(stage_root: Optional[Path] = None):
    stage_root = stage_root or _stage_root_from_this_file()
    v7_src = stage_root / "V7" / "src"
    if not v7_src.exists():
        raise FileNotFoundError(
            f"Cannot find V7 source directory: {v7_src}. "
            "V8 peak-only extraction requires the existing V7 code tree."
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


def _make_v7_cfg_for_v8(v7multi) -> object:
    """Build a V7 MultiWinConfig with original defaults and V8 env aliases.

    The detector/peak implementation uses the original V7 config fields. V8 only
    adds environment aliases and forces non-peak layers off by not calling them.
    """
    cfg = v7multi.MultiWinConfig.from_env()
    # V8 aliases. V7 aliases remain honored by MultiWinConfig.from_env().
    if os.environ.get("V8_PEAK_ACCEPTED_WINDOW_REGISTRY"):
        cfg.accepted_window_registry = os.environ["V8_PEAK_ACCEPTED_WINDOW_REGISTRY"]
        cfg.window_source = "registry"
    if os.environ.get("V8_PEAK_WINDOW_SOURCE"):
        cfg.window_source = os.environ["V8_PEAK_WINDOW_SOURCE"].strip().lower()
    if os.environ.get("V8_PEAK_SMOOTHED_FIELDS"):
        cfg.smoothed_fields_path = os.environ["V8_PEAK_SMOOTHED_FIELDS"]
    if os.environ.get("V8_PEAK_N_BOOTSTRAP"):
        cfg.bootstrap_n = int(os.environ["V8_PEAK_N_BOOTSTRAP"])
    if os.environ.get("V8_PEAK_DEBUG_N_BOOTSTRAP"):
        cfg.bootstrap_n = int(os.environ["V8_PEAK_DEBUG_N_BOOTSTRAP"])
    if os.environ.get("V8_PEAK_WINDOW_MODE"):
        cfg.window_mode = os.environ["V8_PEAK_WINDOW_MODE"].strip().lower()
    if os.environ.get("V8_PEAK_TARGET_WINDOWS"):
        cfg.target_windows = os.environ["V8_PEAK_TARGET_WINDOWS"].strip()
    if os.environ.get("V8_PEAK_LOG_EVERY_BOOTSTRAP"):
        cfg.log_every_bootstrap = int(os.environ["V8_PEAK_LOG_EVERY_BOOTSTRAP"])

    # V8-a is peak-only. These flags are recorded but not used to run state/growth.
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    return cfg


def _default_smoothed_path(v8_root: Path) -> Path:
    # v8_root = D:/easm_project01/stage_partition/V8 => project root is parents[1]
    return v8_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _build_peak_relation_summary(peak_order: pd.DataFrame, sync: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    """Lightweight merge of original peak-layer outputs; no new scientific gate."""
    if peak_order is None or peak_order.empty:
        return pd.DataFrame()
    out = peak_order.copy()
    keep_sync = [c for c in [
        "window_id", "object_A", "object_B", "tau_sync_primary", "P_within_tau_q50",
        "P_within_tau_q75", "P_within_tau_q90", "synchrony_decision", "tau_quality_flag"
    ] if c in sync.columns] if sync is not None and not sync.empty else []
    if keep_sync:
        out = out.merge(sync[keep_sync], on=["window_id", "object_A", "object_B"], how="left")
    keep_overlap = [c for c in [
        "window_id", "object_A", "object_B", "overlap_days", "overlap_fraction",
        "A_peak_inside_B_window", "B_peak_inside_A_window", "both_peaks_inside_system_window",
        "window_overlap_decision"
    ] if c in overlap.columns] if overlap is not None and not overlap.empty else []
    if keep_overlap:
        out = out.merge(overlap[keep_overlap], on=["window_id", "object_A", "object_B"], how="left")
    out["v8_role"] = "peak_layer_summary_no_state_growth"
    out["interpretation_boundary"] = "peak timing only; do not infer state/growth/process relations"
    return out


def _compare_numeric(a, b, tol: float = 1.0e-9) -> str:
    try:
        aa = float(a); bb = float(b)
    except Exception:
        return "same" if str(a) == str(b) else "different"
    if np.isnan(aa) and np.isnan(bb):
        return "same"
    return "same" if abs(aa - bb) <= tol else "different"


def _write_v7_peak_regression_audit(v8_root: Path, out_cross: Path, per_window_results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Compare V8 peak-only outputs against existing V7 hotfix06 peak tables, if present.

    This is an implementation regression audit. Missing V7 outputs are reported as
    missing, not treated as failure.
    """
    stage_root = v8_root.parent
    v7_base = stage_root / "V7" / "outputs" / V7_HOTFIX06_TAG / "per_window"
    rows: List[dict] = []
    if not v7_base.exists():
        _safe_to_csv(pd.DataFrame([{
            "audit_scope": "v7_peak_regression",
            "status": "v7_reference_missing",
            "v7_reference_dir": str(v7_base),
            "note": "No comparison was made. This does not affect V8 peak-only run outputs.",
        }]), out_cross / "v8_vs_v7_hotfix06_peak_regression_audit.csv")
        return

    file_map = {
        "main_window_selection": "main_window_selection_{wid}.csv",
        "bootstrap_selected_peak_days": "bootstrap_selected_peak_days_{wid}.csv",
        "timing_resolution_audit": "timing_resolution_audit_{wid}.csv",
        "tau_sync_estimate": "tau_sync_estimate_{wid}.csv",
        "pairwise_peak_order_test": "pairwise_peak_order_test_{wid}.csv",
        "pairwise_synchrony_equivalence_test": "pairwise_synchrony_equivalence_test_{wid}.csv",
        "pairwise_window_overlap": "pairwise_window_overlap_{wid}.csv",
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

    for wid, res in per_window_results.items():
        v7_dir = v7_base / wid
        for logical, pattern in file_map.items():
            v8_df = res.get(logical, pd.DataFrame())
            v7_file = v7_dir / pattern.format(wid=wid)
            if not v7_file.exists():
                rows.append({"window_id": wid, "table": logical, "status": "v7_file_missing", "v7_file": str(v7_file)})
                continue
            try:
                v7_df = pd.read_csv(v7_file)
            except Exception as exc:
                rows.append({"window_id": wid, "table": logical, "status": "v7_read_error", "error": str(exc), "v7_file": str(v7_file)})
                continue
            if v8_df is None or v8_df.empty:
                rows.append({"window_id": wid, "table": logical, "status": "v8_table_empty", "v7_file": str(v7_file)})
                continue
            keys = [c for c in key_cols.get(logical, []) if c in v8_df.columns and c in v7_df.columns]
            if not keys:
                rows.append({"window_id": wid, "table": logical, "status": "no_common_keys", "v7_file": str(v7_file)})
                continue
            merged = v8_df.merge(v7_df, on=keys, how="outer", suffixes=("_v8", "_v7"), indicator=True)
            missing_or_extra = int((merged["_merge"] != "both").sum())
            diff_count = 0
            col_diffs: Dict[str, int] = {}
            for col in compare_cols.get(logical, []):
                c8, c7 = f"{col}_v8", f"{col}_v7"
                if c8 not in merged.columns or c7 not in merged.columns:
                    continue
                d = 0
                for _, r in merged[merged["_merge"] == "both"].iterrows():
                    if _compare_numeric(r[c8], r[c7]) != "same":
                        d += 1
                if d:
                    col_diffs[col] = d
                    diff_count += d
            status = "pass" if missing_or_extra == 0 and diff_count == 0 else "difference_found"
            rows.append({
                "window_id": wid,
                "table": logical,
                "status": status,
                "n_v8_rows": int(len(v8_df)),
                "n_v7_rows": int(len(v7_df)),
                "n_key_mismatch_rows": missing_or_extra,
                "n_value_differences": diff_count,
                "columns_with_differences": json.dumps(col_diffs, ensure_ascii=False),
                "v7_file": str(v7_file),
            })
    _safe_to_csv(pd.DataFrame(rows), out_cross / "v8_vs_v7_hotfix06_peak_regression_audit.csv")


def _write_peak_only_summary(path: Path, run_scopes: List[object], concat: Dict[str, pd.DataFrame], cfg: object) -> None:
    lines = [
        "# V8 peak-only baseline summary",
        "",
        f"version: `{V8_VERSION}`",
        f"output_tag: `{OUTPUT_TAG}`",
        f"windows_processed: {len(run_scopes)}",
        f"run_mode: {cfg.window_mode}; targets: {cfg.target_windows}",
        "",
        "## Method boundary",
        "- This V8 line is a peak-only extraction from the original V7-z multiwindow peak layer.",
        "- It does not compute or write state curves, growth curves, pre-post process summaries, 2D mirror outputs, or final claims.",
        "- Peak-order and peak-synchrony outputs keep the original V7 helper semantics.",
        "",
        "## Windows processed",
    ]
    for s in run_scopes:
        lines.append(f"- {s.window_id}: system day{s.system_window_start}-{s.system_window_end}; detector day{s.detector_search_start}-{s.detector_search_end}")
    lines += ["", "## Peak relation counts"]
    po = concat.get("pairwise_peak_order_test", pd.DataFrame())
    if not po.empty and "peak_order_decision" in po.columns:
        for k, v in po["peak_order_decision"].value_counts(dropna=False).items():
            lines.append(f"- peak_order_decision={k}: {int(v)}")
    sync = concat.get("pairwise_synchrony_equivalence_test", pd.DataFrame())
    if not sync.empty and "synchrony_decision" in sync.columns:
        for k, v in sync["synchrony_decision"].value_counts(dropna=False).items():
            lines.append(f"- synchrony_decision={k}: {int(v)}")
    lines += ["", "## Forbidden interpretations", "- Do not infer state-front, growth-front, catch-up, rollback, multi-stage, or pre-post process claims from this V8 output alone."]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_peak_only_v8_a(v8_root: Path | str) -> None:
    v8_root = Path(v8_root)
    stage_root = v8_root.parent
    v7_root = stage_root / "V7"
    v7multi = _import_v7_module(stage_root)
    cfg = _make_v7_cfg_for_v8(v7multi)

    out_root = _ensure_dir(v8_root / "outputs" / OUTPUT_TAG)
    out_cross = _ensure_dir(out_root / "cross_window")
    out_per = _ensure_dir(out_root / "per_window")
    log_dir = _ensure_dir(v8_root / "logs" / OUTPUT_TAG)
    t0 = time.time()

    _log("[1/6] Load accepted/significant windows using original V7 helper")
    wins = v7multi._load_accepted_windows(v7_root, out_cross, cfg)
    scopes, validity = v7multi._build_window_scopes(wins, cfg)
    run_scopes, run_scope_audit = v7multi._filter_scopes_for_run(scopes, cfg)
    _safe_to_csv(pd.DataFrame([asdict(s) for s in scopes]), out_cross / "window_scope_registry_v8_peak_only_a.csv")
    _safe_to_csv(validity, out_cross / "window_scope_validity_audit_v8_peak_only_a.csv")
    _safe_to_csv(run_scope_audit, out_cross / "run_window_selection_audit_v8_peak_only_a.csv")
    _safe_to_csv(pd.DataFrame([asdict(s) for s in run_scopes]), out_cross / "run_window_scope_registry_v8_peak_only_a.csv")

    _log("[2/6] Load smoothed fields using original V7 clean loader")
    smoothed = Path(cfg.smoothed_fields_path) if cfg.smoothed_fields_path else _default_smoothed_path(v8_root)
    if not smoothed.exists():
        # Keep same default fallback semantics as V7; the explicit error below is clearer for V8.
        smoothed = _default_smoothed_path(v8_root)
    if not smoothed.exists():
        raise FileNotFoundError(
            f"smoothed_fields.npz not found: {smoothed}. "
            "Set V8_PEAK_SMOOTHED_FIELDS or V7_MULTI_SMOOTHED_FIELDS to the correct file."
        )
    fields, audit = v7multi.clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    _safe_to_csv(audit, out_cross / "input_key_audit_v8_peak_only_a.csv")

    _write_json({
        "version": V8_VERSION,
        "output_tag": OUTPUT_TAG,
        "source_v7_module_version": getattr(v7multi, "VERSION", "unknown"),
        "source_v7_output_tag": getattr(v7multi, "OUTPUT_TAG", "unknown"),
        "config": asdict(cfg),
        "smoothed_fields": str(smoothed),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "boundary": "peak-only; no state/growth/prepost/process layers",
    }, out_root / "run_meta.json")

    _log("[3/6] Build object profiles with original V7 clean helpers")
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
            "v8_role": "peak_detector_profile_input",
        })
    _safe_to_csv(pd.DataFrame(object_rows), out_cross / "object_registry_v8_peak_only_a.csv")

    _log("[4/6] Run original V7 profile detector + peak bootstrap, peak-only outputs")
    per_window_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    all_parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in [
        "raw_profile_detector_scores", "object_profile_window_registry", "main_window_selection",
        "selected_peak_delta", "bootstrap_selected_peak_days", "timing_resolution_audit",
        "tau_sync_estimate", "pairwise_peak_order_test", "pairwise_synchrony_equivalence_test",
        "pairwise_window_overlap", "peak_relation_summary",
    ]}

    for idx, scope in enumerate(run_scopes, start=1):
        _log(f"  [{idx}/{len(run_scopes)}] {scope.window_id}")
        out_win = _ensure_dir(out_per / scope.window_id)
        _safe_to_csv(pd.DataFrame([asdict(scope)]), out_win / f"window_scope_{scope.window_id}.csv")
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
        }
        for logical, df in result_map.items():
            _safe_to_csv(df, out_win / file_names[logical])
            if df is not None and not df.empty:
                all_parts[logical].append(df)
        per_window_results[scope.window_id] = result_map

    _log("[5/6] Write cross-window peak-only outputs and regression audit")
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
    }
    for logical, filename in cross_names.items():
        _safe_to_csv(concat[logical], out_cross / filename)
    _write_v7_peak_regression_audit(v8_root, out_cross, per_window_results)
    _write_peak_only_summary(out_cross / "peak_only_v8_a_summary.md", run_scopes, concat, cfg)

    _log("[6/6] Done")
    elapsed = time.time() - t0
    _write_json({
        "version": V8_VERSION,
        "elapsed_seconds": elapsed,
        "n_windows_available": len(scopes),
        "n_windows_processed": len(run_scopes),
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
    run_peak_only_v8_a(Path(__file__).resolve().parents[2])
