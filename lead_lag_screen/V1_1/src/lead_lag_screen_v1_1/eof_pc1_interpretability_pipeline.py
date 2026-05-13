from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import setup_logger
from .eof_pc1_interpretability_core import (
    build_diagnosis_tables,
    compute_eof_for_field,
    eof_pc_lead_lag_by_window,
    eof_vs_structural_lead_lag,
    load_domain_fields,
    loading_region_summary,
    pc_structural_index_correlation,
    reconstruction_region_change_skill,
)
from .eof_pc1_interpretability_plotting import make_figures
from .eof_pc1_interpretability_settings import EOFPC1InterpretabilitySettings


def _jsonable(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_eof_pc1_interpretability_audit_v1_a(settings: EOFPC1InterpretabilitySettings | None = None, make_figures: bool = True, use_cartopy: bool = True) -> dict:
    settings = settings or EOFPC1InterpretabilitySettings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.table_dir.mkdir(parents=True, exist_ok=True)
    settings.figure_dir.mkdir(parents=True, exist_ok=True)
    settings.summary_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(settings.log_dir)

    logger.info("Starting EOF-PC1 interpretability audit")
    logger.info("Input smoothed fields: %s", settings.input_smoothed_fields)
    logger.info("Output dir: %s", settings.output_dir)
    logger.info("EOF domain lat=%s lon=%s stride=%s", settings.eof_lat_range, settings.eof_lon_range, settings.spatial_stride)

    fields = load_domain_fields(settings)
    lat = fields["lat"]
    lon = fields["lon"]
    years = fields["years"]

    logger.info("Computing P EOF")
    p_eof = compute_eof_for_field("P", fields["precip"], lat, lon, years, settings)
    logger.info("Computing V EOF")
    v_eof = compute_eof_for_field("V", fields["v850"], lat, lon, years, settings)

    # Save PC scores and EOF variance summary.
    p_eof.pcs.to_csv(settings.table_dir / "p_eof_pc_scores.csv", index=False, encoding="utf-8-sig")
    v_eof.pcs.to_csv(settings.table_dir / "v_eof_pc_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "field": ["P"] * len(p_eof.singular_values) + ["V"] * len(v_eof.singular_values),
        "mode": list(range(1, len(p_eof.singular_values) + 1)) + list(range(1, len(v_eof.singular_values) + 1)),
        "singular_value": list(p_eof.singular_values) + list(v_eof.singular_values),
        "explained_variance_ratio": list(p_eof.explained_variance_ratio) + list(v_eof.explained_variance_ratio),
    }).to_csv(settings.table_dir / "eof_variance_summary.csv", index=False, encoding="utf-8-sig")

    p_loading = loading_region_summary(p_eof, settings.regions)
    v_loading = loading_region_summary(v_eof, settings.regions)
    p_loading.to_csv(settings.table_dir / "p_eof_loading_region_summary.csv", index=False, encoding="utf-8-sig")
    v_loading.to_csv(settings.table_dir / "v_eof_loading_region_summary.csv", index=False, encoding="utf-8-sig")

    pc_corr = pc_structural_index_correlation(p_eof, v_eof, settings)
    pc_corr.to_csv(settings.table_dir / "eof_pc_structural_index_correlation.csv", index=False, encoding="utf-8-sig")

    recon = pd.concat([
        reconstruction_region_change_skill(p_eof, settings),
        reconstruction_region_change_skill(v_eof, settings),
    ], ignore_index=True)
    recon.to_csv(settings.table_dir / "eof_reconstruction_region_change_skill.csv", index=False, encoding="utf-8-sig")

    pc_ll = eof_pc_lead_lag_by_window(p_eof, v_eof, settings)
    pc_ll.to_csv(settings.table_dir / "eof_pc_lead_lag_by_window.csv", index=False, encoding="utf-8-sig")

    eof_vs_struct = eof_vs_structural_lead_lag(p_eof, v_eof, settings)
    eof_vs_struct.to_csv(settings.table_dir / "eof_vs_structural_index_lead_lag_comparison.csv", index=False, encoding="utf-8-sig")

    diag = build_diagnosis_tables(p_loading, v_loading, pc_corr, recon, pc_ll)
    diag.to_csv(settings.table_dir / "eof_pc1_interpretability_diagnosis.csv", index=False, encoding="utf-8-sig")

    figure_paths = []
    if make_figures:
        logger.info("Creating figures")
        figure_paths = [str(p) for p in make_figures_fn(p_eof, v_eof, settings, use_cartopy=use_cartopy)]

    summary = {
        "status": "success",
        "audit": "eof_pc1_interpretability_audit_v1_a",
        "primary_question": "Whether EOF-PC1 actually represents the T3 high-latitude/boundary/retreat structures that V1_1 structural indices exposed.",
        "input_smoothed_fields": str(settings.input_smoothed_fields),
        "input_v1_1_structural_indices": str(settings.input_v1_1_structural_indices),
        "domain": {"lat_range": settings.eof_lat_range, "lon_range": settings.eof_lon_range, "spatial_stride": settings.spatial_stride},
        "full_data_extent": {"lat_min": fields["full_lat_min"], "lat_max": fields["full_lat_max"], "lon_min": fields["full_lon_min"], "lon_max": fields["full_lon_max"]},
        "domain_shape": {"n_year": int(fields["precip"].shape[0]), "n_day": int(fields["precip"].shape[1]), "n_lat": int(len(lat)), "n_lon": int(len(lon))},
        "eof_value_mode": settings.eof_value_mode,
        "eof_method": settings.eof_method,
        "n_modes": settings.n_modes,
        "p_evr": [float(x) for x in p_eof.explained_variance_ratio],
        "v_evr": [float(x) for x in v_eof.explained_variance_ratio],
        "diagnosis_counts": diag["support_level"].value_counts(dropna=False).to_dict() if not diag.empty else {},
        "figure_paths": figure_paths,
        "guardrails": [
            "This audit does not replace V1_1 structural V→P screen.",
            "EOF-PC1 continuity cannot refute T3 pair-level weakening unless PC1 reconstructs the T3 structures under dispute.",
            "PC1 is a variance representation, not a V–P coupled-mode optimizer.",
        ],
    }
    (settings.summary_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    run_meta = {
        "settings": _jsonable(asdict(settings)),
        "outputs": {
            "tables": [p.name for p in settings.table_dir.glob("*.csv")],
            "figures": [p.name for p in settings.figure_dir.glob("*.png")],
        },
        "method_notes": {
            "field_value_mode": settings.eof_value_mode,
            "topk_solver": "deterministic subspace iteration; avoids forming full covariance matrix; method recorded to avoid hidden simplification",
            "coslat_weighting": settings.use_coslat_weight,
        },
    }
    (settings.summary_dir / "run_meta.json").write_text(json.dumps(_jsonable(run_meta), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Finished EOF-PC1 interpretability audit")
    return summary


# Avoid shadowing the boolean argument name inside the pipeline.
def make_figures_fn(p_eof, v_eof, settings, use_cartopy=True):
    return make_figures(p_eof, v_eof, settings, use_cartopy=use_cartopy)
