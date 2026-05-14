from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .cluster_definitions import build_cluster_definition_table
from .cluster_similarity import build_cluster_similarity
from .config import W045PreclusterConfig
from .curve_metrics import build_object_cluster_metrics, build_participation_matrix
from .io_utils import (
    find_by_glob,
    read_csv_if_exists,
    require_csv,
    standardize_curve_long,
    standardize_markers,
    write_json,
)
from .morphology_metrics import build_morphology_table
from .plotting import make_all_figures
from .role_classifier import audit_h35_role, build_interpretation_summary
from .summary_writer import write_markdown_summary


def _input_paths(cfg: W045PreclusterConfig) -> dict[str, Path | None]:
    assert cfg.v10_5e_root is not None and cfg.v10_5a_root is not None
    return {
        "main_curve": find_by_glob(cfg.v10_5e_root, ["curves/main_method_continuous_detector_score_curves_v10_5_e.csv", "curves/*main*curve*.csv"]),
        "profile_curve": find_by_glob(cfg.v10_5e_root, ["curves/profile_energy_continuous_curves_fullseason_v10_5_e.csv", "curves/*profile*energy*curve*.csv"]),
        "markers": find_by_glob(cfg.v10_5e_root, ["markers/main_method_candidate_markers_v10_5_e.csv", "markers/*candidate*marker*.csv", "markers/*.csv"]),
        "family_summary_b": cfg.v10_5a_root / "validation_summary_candidate_family_v10_5_b.csv",
        "competition_d": cfg.v10_5a_root / "profile_validation" / "key_competition_cases_v10_5_d.csv",
    }


def run_w045_precluster_audit_v10_6_a(project_root: str | Path) -> dict[str, Any]:
    cfg = W045PreclusterConfig.from_project_root(project_root)
    cfg.ensure_output_dirs()
    paths = _input_paths(cfg)

    if paths["main_curve"] is None:
        raise FileNotFoundError("Cannot find V10.5_e main continuous curve CSV under strength_curve_export_v10_5_e/curves")
    if paths["profile_curve"] is None:
        raise FileNotFoundError("Cannot find V10.5_e profile-energy continuous curve CSV under strength_curve_export_v10_5_e/curves")
    if paths["markers"] is None:
        raise FileNotFoundError("Cannot find V10.5_e candidate marker CSV under strength_curve_export_v10_5_e/markers")

    main_raw = require_csv(paths["main_curve"], "V10.5_e main continuous curve")
    profile_raw = require_csv(paths["profile_curve"], "V10.5_e profile-energy continuous curve")
    marker_raw = require_csv(paths["markers"], "V10.5_e main candidate markers")

    main_curve, main_meta = standardize_curve_long(main_raw, "main_curve", cfg.objects)
    profile_curve, profile_meta = standardize_curve_long(profile_raw, "profile_curve", cfg.objects)
    markers, marker_meta = standardize_markers(marker_raw, cfg.objects)

    family_summary_b = read_csv_if_exists(paths["family_summary_b"]) if paths["family_summary_b"] else None
    competition_d = read_csv_if_exists(paths["competition_d"]) if paths["competition_d"] else None

    # Core tables.
    cluster_def = build_cluster_definition_table(cfg)
    metrics = build_object_cluster_metrics(cfg, main_curve, profile_curve, markers)
    participation = build_participation_matrix(metrics, cfg.objects)
    morphology = build_morphology_table(cfg, main_curve, metrics)
    similarity = build_cluster_similarity(cfg, metrics)
    h_role = audit_h35_role(cfg, metrics, competition_d, family_summary_b)
    interpretation = build_interpretation_summary(metrics, h_role, morphology)

    # Write tables.
    cluster_def.to_csv(cfg.tables_dir / "w045_cluster_definition_v10_6_a.csv", index=False)
    metrics.to_csv(cfg.tables_dir / "w045_object_cluster_metrics_v10_6_a.csv", index=False)
    participation.to_csv(cfg.tables_dir / "w045_cluster_participation_matrix_v10_6_a.csv", index=False)
    morphology.to_csv(cfg.tables_dir / "w045_e1_e2_m_morphology_v10_6_a.csv", index=False)
    similarity.to_csv(cfg.tables_dir / "w045_cluster_similarity_v10_6_a.csv", index=False)
    h_role.to_csv(cfg.tables_dir / "w045_H_day35_role_audit_v10_6_a.csv", index=False)
    interpretation.to_csv(cfg.tables_dir / "w045_interpretation_summary_v10_6_a.csv", index=False)

    # Figures.
    make_all_figures(cfg, main_curve, profile_curve, markers, metrics, similarity)

    summary_path = write_markdown_summary(cfg, metrics, h_role, interpretation)

    run_meta = {
        "version": cfg.version,
        "task": cfg.task_name,
        "scope": "W045 only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(cfg.project_root),
        "output_root": str(cfg.output_root),
        "clusters": [c.__dict__ for c in cfg.clusters],
        "objects": list(cfg.objects),
        "profile_k_values": list(cfg.profile_k_values),
        "input_paths": {k: str(v) if v else None for k, v in paths.items()},
        "standardized_input_meta": {
            "main_curve": main_meta,
            "profile_curve": profile_meta,
            "markers": marker_meta,
        },
        "row_counts": {
            "main_curve": int(len(main_curve)),
            "profile_curve": int(len(profile_curve)),
            "markers": int(len(markers)),
            "metrics": int(len(metrics)),
            "morphology": int(len(morphology)),
        },
        "not_implemented_in_v10_6_a": [
            "yearwise validation",
            "spatial/cartopy validation",
            "causal inference",
            "full-season generalization beyond W045",
            "accepted-window re-detection",
        ],
        "interpretation_boundary": "method-layer and derived-structure audit only; H day35 must not be described as confirmed weak precursor based on v10.6_a alone",
        "summary_path": str(summary_path),
    }
    write_json(cfg.run_meta_dir / "run_meta_v10_6_a.json", run_meta)
    return run_meta
