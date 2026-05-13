from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .compare_v1 import build_v1_overlap
from .data_io import (
    build_window_panel,
    ensure_dirs,
    make_directed_pairs,
    read_index_anomalies,
    read_v1_evidence,
)
from .graph_extract import extract_pcmci_tables
from .rollups import build_family_rollup, build_window_summary
from .settings import LeadLagScreenV2Settings
from .tigramite_adapter import run_pcmci_plus


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _cache_paths(settings: LeadLagScreenV2Settings, window: str) -> dict:
    safe = window.replace("/", "_")
    return {
        "edge": settings.cache_dir / f"{safe}_pcmci_plus_edges_long.csv",
        "tau0": settings.cache_dir / f"{safe}_pcmci_plus_tau0_contemporaneous.csv",
        "meta": settings.cache_dir / f"{safe}_window_meta.json",
    }


def _load_cached_window(settings: LeadLagScreenV2Settings, window: str):
    p = _cache_paths(settings, window)
    if p["edge"].exists() and p["tau0"].exists() and p["meta"].exists():
        return (
            pd.read_csv(p["edge"], encoding="utf-8-sig"),
            pd.read_csv(p["tau0"], encoding="utf-8-sig"),
            json.loads(p["meta"].read_text(encoding="utf-8")),
        )
    return None


def _save_cached_window(settings: LeadLagScreenV2Settings, window: str, edges: pd.DataFrame, tau0: pd.DataFrame, meta: dict) -> None:
    p = _cache_paths(settings, window)
    p["edge"].parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(p["edge"], index=False, encoding="utf-8-sig")
    tau0.to_csv(p["tau0"], index=False, encoding="utf-8-sig")
    _write_json(p["meta"], meta)


def run_pcmci_plus_smooth5_v2(settings: LeadLagScreenV2Settings | None = None, logger=None) -> dict:
    settings = settings or LeadLagScreenV2Settings()
    ensure_dirs(settings.output_dir, settings.log_dir, settings.cache_dir, settings.runtime_status_dir)

    if logger is None:
        from .logging_utils import setup_logger
        logger = setup_logger(settings.log_dir)

    started = datetime.now().isoformat(timespec="seconds")
    run_meta = {
        "layer": "lead_lag_screen/V2",
        "method": "PCMCI+",
        "role": "smooth5 conditional direct-edge control for V1 lead-lag screen",
        "started_at": started,
        "settings": settings.to_jsonable_dict(),
        "important_boundaries": [
            "No fallback if tigramite is unavailable.",
            "Same-family variables remain in the conditioning pool but are not reported as source-target edges.",
            "Main lagged output uses tau=1..5. tau=0 is written as contemporaneous diagnostic only.",
            "Outputs are direct-edge controls, not pathway or mediator results.",
            "Each year is a separate tigramite dataset; cross-year stitching is forbidden.",
        ],
    }
    _write_json(settings.output_dir / "run_meta.json", run_meta)
    _write_json(settings.output_dir / "settings_summary.json", settings.to_jsonable_dict())

    logger.info("Starting lead_lag_screen/V2 PCMCI+ smooth5 control")
    logger.info("Input index anomalies: %s", settings.input_index_anomalies)
    logger.info("Output directory: %s", settings.output_dir)

    df = read_index_anomalies(settings.input_index_anomalies, settings.variables)
    years = np.asarray(sorted(df["year"].dropna().unique()), dtype=int)
    pairs = make_directed_pairs(
        settings.variable_families,
        include_same_family=settings.include_same_family_reported_edges,
    )
    pairs.to_csv(settings.output_dir / "pcmci_plus_candidate_pairs.csv", index=False, encoding="utf-8-sig")

    all_edges: List[pd.DataFrame] = []
    all_tau0: List[pd.DataFrame] = []
    status_rows = []
    timing_rows = []
    failed_rows = []
    window_meta_rows = []

    for i, (window, (w_start, w_end)) in enumerate(settings.windows.items(), start=1):
        t0 = time.time()
        logger.info("[%s/%s] PCMCI+ window %s days %s-%s", i, len(settings.windows), window, w_start, w_end)
        status = {
            "window": window,
            "status": "started",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "output_tag": settings.output_tag,
        }
        try:
            cached = None if settings.force_recompute else _load_cached_window(settings, window) if settings.resume else None
            if cached is not None:
                edges, tau0, meta = cached
                logger.info("[%s] loaded cached PCMCI+ window result", window)
                status["status"] = "cached"
            else:
                data_dict, mask_dict, ext_days = build_window_panel(
                    df=df,
                    variables=settings.variables,
                    years=years,
                    window_start=w_start,
                    window_end=w_end,
                    tau_max=settings.tau_max,
                )
                n_total = sum(arr.size for arr in data_dict.values())
                n_masked = sum(mask.sum() for mask in mask_dict.values())
                meta = {
                    "window": window,
                    "target_window_start": int(w_start),
                    "target_window_end": int(w_end),
                    "extended_day_start": int(ext_days[0]),
                    "extended_day_end": int(ext_days[-1]),
                    "n_years": int(len(years)),
                    "n_variables": int(len(settings.variables)),
                    "n_extended_days": int(len(ext_days)),
                    "n_total_values": int(n_total),
                    "n_masked_values": int(n_masked),
                    "masked_fraction": float(n_masked / n_total) if n_total else None,
                    "target_side_window_semantics": "Y(t) is in the named window; X(t-tau) can come from the backward tau_max padding; no cross-year stitching.",
                }
                results = run_pcmci_plus(
                    data_dict=data_dict,
                    mask_dict=mask_dict,
                    variables=settings.variables,
                    tau_min=settings.tau_min,
                    tau_max=settings.tau_max,
                    pc_alpha=settings.pc_alpha,
                    parcorr_significance=settings.parcorr_significance,
                    verbosity=settings.tigramite_verbosity,
                )
                edges, tau0 = extract_pcmci_tables(
                    results=results,
                    variables=settings.variables,
                    variable_families=settings.variable_families,
                    window=window,
                    tau_min=settings.tau_min,
                    tau_max=settings.tau_max,
                    main_tau_min=settings.main_tau_min,
                    main_tau_max=settings.main_tau_max,
                    fdr_alpha=settings.fdr_alpha,
                    include_same_family_reported_edges=settings.include_same_family_reported_edges,
                )
                if settings.write_cache:
                    _save_cached_window(settings, window, edges, tau0, meta)
                status["status"] = "completed"

            all_edges.append(edges)
            all_tau0.append(tau0)
            window_meta_rows.append(meta)
            status["n_lagged_tests"] = int(len(edges))
            status["n_lagged_supported"] = int(edges["pcmci_plus_supported"].sum()) if not edges.empty else 0
            status["n_tau0_tests"] = int(len(tau0))
            status["n_tau0_supported"] = int(tau0["tau0_supported"].sum()) if (not tau0.empty and "tau0_supported" in tau0.columns) else 0
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = repr(exc)
            failed_rows.append({
                "window": window,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            logger.exception("[%s] PCMCI+ failed", window)
        finally:
            elapsed = time.time() - t0
            timing_rows.append({"window": window, "elapsed_seconds": elapsed})
            status["elapsed_seconds"] = elapsed
            status["finished_at"] = datetime.now().isoformat(timespec="seconds")
            status_rows.append(status)
            pd.DataFrame(status_rows).to_csv(settings.output_dir / "runtime_task_status.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(timing_rows).to_csv(settings.output_dir / "runtime_task_timing.csv", index=False, encoding="utf-8-sig")
            if failed_rows:
                pd.DataFrame(failed_rows).to_csv(settings.output_dir / "runtime_failed_tasks.csv", index=False, encoding="utf-8-sig")

    if failed_rows:
        raise RuntimeError(
            f"PCMCI+ V2 failed for {len(failed_rows)} window(s). See runtime_failed_tasks.csv in {settings.output_dir}."
        )

    edges_all = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    tau0_all = pd.concat(all_tau0, ignore_index=True) if all_tau0 else pd.DataFrame()
    edges_all.to_csv(settings.output_dir / "pcmci_plus_edges_long.csv", index=False, encoding="utf-8-sig")
    tau0_all.to_csv(settings.output_dir / "pcmci_plus_tau0_contemporaneous_diagnostic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(window_meta_rows).to_csv(settings.output_dir / "pcmci_plus_window_panel_meta.csv", index=False, encoding="utf-8-sig")

    family_rollup = build_family_rollup(edges_all)
    family_rollup.to_csv(settings.output_dir / "pcmci_plus_window_family_rollup.csv", index=False, encoding="utf-8-sig")
    window_summary = build_window_summary(edges_all, tau0_all)
    window_summary.to_csv(settings.output_dir / "pcmci_plus_window_summary.csv", index=False, encoding="utf-8-sig")

    v1 = read_v1_evidence(settings.v1_evidence_tier_summary)
    overlap, overlap_summary = build_v1_overlap(edges_all, v1)
    overlap.to_csv(settings.output_dir / "pcmci_plus_v1_overlap.csv", index=False, encoding="utf-8-sig")
    overlap_summary.to_csv(settings.output_dir / "pcmci_plus_v1_overlap_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "layer": "lead_lag_screen/V2",
        "output_tag": settings.output_tag,
        "method": "PCMCI+ with ParCorr",
        "input": str(settings.input_index_anomalies),
        "n_windows": int(len(settings.windows)),
        "n_years": int(len(years)),
        "n_variables": int(len(settings.variables)),
        "n_reported_directed_pairs": int(len(pairs)),
        "n_lagged_candidate_tests": int(len(edges_all)),
        "n_lagged_graph_selected": int(edges_all["pcmci_graph_selected"].sum()) if not edges_all.empty else 0,
        "n_lagged_pcmci_plus_supported": int(edges_all["pcmci_plus_supported"].sum()) if not edges_all.empty else 0,
        "n_tau0_candidate_tests": int(len(tau0_all)),
        "n_tau0_supported": int(tau0_all["tau0_supported"].sum()) if (not tau0_all.empty and "tau0_supported" in tau0_all.columns) else 0,
        "support_rule": "lagged main output: graph_selected and within-window BH q <= fdr_alpha",
        "not_pathway_result": True,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(settings.output_dir / "summary.json", summary)
    logger.info("Finished lead_lag_screen/V2 PCMCI+ smooth5 control: %s", summary)
    return summary
