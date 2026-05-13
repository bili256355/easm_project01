from __future__ import annotations

import time

import pandas as pd

from .data_io import ensure_dirs, load_fields_for_mode, load_index_table_for_mode, save_csv, save_json
from .logging_utils import setup_logger
from .metadata import FAMILY_ORDER, VARIABLE_ORDER
from .metrics import build_family_guardrail, compute_index_window_metrics, compute_window_family_joint_coverage
from .plotting import plot_selected_composites, select_figure_targets
from .settings import IndexValidityV1BSettings


def _input_file_audit(settings: IndexValidityV1BSettings) -> pd.DataFrame:
    paths = {
        "selected_index_table": settings.selected_index_path,
        "index_values_smoothed": settings.index_values_smoothed_path,
        "index_anomalies": settings.index_anomalies_path,
        "smoothed_fields_bundle": settings.smoothed_fields_bundle,
        "anomaly_fields_bundle": settings.anomaly_fields_bundle,
        "daily_climatology_bundle": settings.climatology_bundle,
    }
    rows = []
    for name, path in paths.items():
        rows.append({
            "input_name": name,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        })
    return pd.DataFrame(rows)


def _timing_row(task: str, seconds: float) -> dict:
    return {"task": task, "window": "__pipeline__", "index_name": "__all__", "seconds": round(seconds, 6)}


def run_index_validity_window_family_guardrail_v1_b(settings: IndexValidityV1BSettings | None = None) -> dict:
    settings = settings or IndexValidityV1BSettings()
    t0 = time.time()
    pipeline_timing = []
    ensure_dirs(settings.output_dir, settings.log_dir, settings.tables_dir, settings.figures_dir, settings.summary_dir)
    logger = setup_logger(settings.log_dir)
    logger.info("Starting index_validity V1_b window-family guardrail")
    logger.info("Output dir: %s", settings.output_dir)
    logger.info("Data mode: %s", settings.data_mode)
    if settings.data_mode != "smoothed":
        logger.warning("Running auxiliary anomaly mode. Main index_validity guardrail should use smoothed mode.")

    t = time.time()
    audit = _input_file_audit(settings)
    save_csv(settings.tables_dir / "input_file_audit.csv", audit)
    if not settings.selected_index_path.exists():
        raise FileNotFoundError(f"Missing required {settings.data_mode} index table: {settings.selected_index_path}")
    if settings.data_mode == "smoothed":
        if not settings.smoothed_fields_bundle.exists():
            raise FileNotFoundError(
                "Missing required smooth5 smoothed fields. Need smoothed_fields.npz. "
                "The uploaded engineering package may omit preprocess files; run on the complete local project."
            )
    elif settings.data_mode == "anomaly":
        if not (settings.anomaly_fields_bundle.exists() or (settings.smoothed_fields_bundle.exists() and settings.climatology_bundle.exists())):
            raise FileNotFoundError(
                "Missing required smooth5 anomaly fields. Need anomaly_fields.npz, or smoothed_fields.npz + daily_climatology.npz. "
                "The uploaded engineering package may omit preprocess files; run on the complete local project."
            )
    else:
        raise ValueError(f"Unsupported data_mode={settings.data_mode!r}; expected 'smoothed' or 'anomaly'.")
    pipeline_timing.append(_timing_row("input_audit", time.time() - t))

    t = time.time()
    logger.info("Loading %s index table", settings.data_mode)
    index_df = load_index_table_for_mode(settings)
    logger.info("Loading smooth5 %s fields", settings.data_mode)
    fields, lat, lon, years, field_input_meta = load_fields_for_mode(settings)
    pipeline_timing.append(_timing_row("load_inputs", time.time() - t))

    t = time.time()
    logger.info("Computing window × index representativeness metrics with equivalent caching/vectorization speedups")
    index_metrics, figure_payloads, metric_timing = compute_index_window_metrics(index_df, fields, lat, lon, years, settings)
    save_csv(settings.tables_dir / "index_window_representativeness.csv", index_metrics)
    pipeline_timing.append(_timing_row("compute_metrics_total", time.time() - t))

    t = time.time()
    logger.info("Building window × family guardrail table")
    family_guardrail = build_family_guardrail(index_metrics, settings)
    save_csv(settings.tables_dir / "window_family_guardrail.csv", family_guardrail)
    t3 = index_metrics[index_metrics["window"].eq("T3")].copy()
    save_csv(settings.tables_dir / "t3_index_representativeness_audit.csv", t3)
    t3_family = family_guardrail[family_guardrail["window"].eq("T3")].copy()
    save_csv(settings.tables_dir / "t3_window_family_guardrail_audit.csv", t3_family)
    pipeline_timing.append(_timing_row("build_guardrail_tables", time.time() - t))

    t = time.time()
    logger.info("Computing window × family mixed/joint field coverage")
    joint_coverage, joint_timing = compute_window_family_joint_coverage(index_df, fields, lat, lon, years, index_metrics, settings)
    save_csv(settings.tables_dir / "window_family_joint_field_coverage.csv", joint_coverage)
    t3_joint = joint_coverage[joint_coverage["window"].eq("T3")].copy()
    save_csv(settings.tables_dir / "t3_window_family_joint_field_coverage.csv", t3_joint)
    pipeline_timing.append(_timing_row("compute_joint_family_coverage", time.time() - t))

    t = time.time()
    logger.info("Selecting and plotting diagnostic composite figures")
    figure_manifest = pd.DataFrame()
    cartopy_status = "figures_disabled"
    if settings.make_figures:
        targets = select_figure_targets(index_metrics, family_guardrail, settings.max_diagnostic_figures)
        save_csv(settings.tables_dir / "figure_target_selection.csv", targets)
        figure_manifest, cartopy_status = plot_selected_composites(
            figure_payloads=figure_payloads,
            targets=targets,
            output_dir=settings.figures_dir / "selected_composite_maps",
            dpi=settings.figure_dpi,
            use_cartopy_if_available=settings.use_cartopy_if_available,
            display_extent=(settings.display_lon_range[0], settings.display_lon_range[1], settings.display_lat_range[0], settings.display_lat_range[1]),
        )
    else:
        save_csv(settings.tables_dir / "figure_target_selection.csv", pd.DataFrame())
    save_csv(settings.tables_dir / "figure_manifest.csv", figure_manifest)
    pipeline_timing.append(_timing_row("plot_figures", time.time() - t))

    runtime_timing = pd.concat([pd.DataFrame(pipeline_timing), metric_timing, joint_timing], ignore_index=True)
    save_csv(settings.tables_dir / "runtime_task_timing.csv", runtime_timing)

    collapse_counts = family_guardrail["family_collapse_risk"].value_counts().to_dict()
    high_risk_rows = family_guardrail[family_guardrail["family_collapse_risk"].eq("high")]
    partial_rows = family_guardrail[family_guardrail["family_collapse_risk"].eq("partial_sensitivity")]
    tier_counts = index_metrics["representativeness_tier"].value_counts().to_dict()

    coverage_tier_counts = joint_coverage["coverage_tier"].value_counts().to_dict()
    coverage_gap_rows = joint_coverage[joint_coverage["collapse_risk_update"].eq("possible_joint_coverage_gap_review_maps")]

    summary = {
        "status": "success",
        "runtime_seconds": round(time.time() - t0, 3),
        "runtime_tag": settings.output_tag,
        "task_scope": "index_validity_window_family_guardrail_no_leadlag_no_pathway",
        "data_mode": settings.data_mode,
        "performance_patch": "v1_b_speedup_equivalent_cache_vectorized_r2_yearlevel_bootstrap",
        "performance_semantics": "No intended change to formulas, thresholds, default samples, or output fields; only computation organization and optional runtime controls changed.",
        "index_input_path": str(settings.selected_index_path),
        "field_input_meta": field_input_meta,
        "n_windows": len(settings.windows),
        "n_families": len(FAMILY_ORDER),
        "n_indices": len(VARIABLE_ORDER),
        "n_index_window_rows": int(len(index_metrics)),
        "n_window_family_rows": int(len(family_guardrail)),
        "n_window_family_joint_coverage_rows": int(len(joint_coverage)),
        "joint_coverage_tier_counts": {str(k): int(v) for k, v in coverage_tier_counts.items()},
        "n_possible_joint_coverage_gap_rows": int(len(coverage_gap_rows)),
        "possible_joint_coverage_gap_rows": coverage_gap_rows[["window", "family", "coverage_tier", "joint_field_R2_year_cv", "joint_eof_coverage_top5_year_cv"]].to_dict("records"),
        "representativeness_tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "family_collapse_risk_counts": {str(k): int(v) for k, v in collapse_counts.items()},
        "n_high_collapse_family_windows": int(len(high_risk_rows)),
        "n_partial_sensitivity_family_windows": int(len(partial_rows)),
        "high_collapse_family_windows": high_risk_rows[["window", "family", "best_index", "best_index_score"]].to_dict("records"),
        "partial_sensitivity_family_windows": partial_rows[["window", "family", "best_index", "best_index_score"]].to_dict("records"),
        "cartopy_status": cartopy_status,
        "outputs": {"tables_dir": str(settings.tables_dir), "figures_dir": str(settings.figures_dir), "summary_dir": str(settings.summary_dir)},
        "optional_runtime_controls": {
            "make_figures": settings.make_figures,
            "use_cartopy_if_available": settings.use_cartopy_if_available,
            "max_diagnostic_figures": settings.max_diagnostic_figures,
        },
        "interpretation_guardrails": [
            "This layer tests index-to-own-field representativeness only.",
            "Main/default mode uses smoothed fields and smoothed index values; anomaly mode is auxiliary only.",
            "It does not test lead-lag, pathway, causality, or downstream effect validity.",
            "family_collapse_risk=low means whole-family index collapse is not supported; it does not mean every index is perfect.",
            "window_family_joint_field_coverage.csv measures mixed family-level field coverage from all same-family indices jointly; it does not alter the original guardrail table.",
            "partial_sensitivity means use index-level flags and avoid over-interpreting sensitive indices.",
            "Map figures are diagnostic aids; table metrics and family guardrails are the primary outputs.",
        ],
    }
    run_meta = {
        "settings": settings.to_jsonable(),
        "input_file_audit": audit.to_dict("records"),
        "runtime_timing_csv": str(settings.tables_dir / "runtime_task_timing.csv"),
        "scope_exclusions": ["lead_lag", "pathway", "causal_discovery", "field_to_field_coupling", "proof_that_all_indices_are_valid_in_all_windows"],
    }
    save_json(settings.summary_dir / "summary.json", summary)
    save_json(settings.summary_dir / "run_meta.json", run_meta)
    logger.info("Finished index_validity V1_b window-family guardrail")
    return summary
