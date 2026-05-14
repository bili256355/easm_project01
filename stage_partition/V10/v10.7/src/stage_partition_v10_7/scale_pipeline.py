from __future__ import annotations

from pathlib import Path
from typing import Any

from .scale_config import ScaleSettings
from .io_utils import load_smoothed_fields
from .h_profile_builder import build_h_profile, build_h_state_matrix, summarize_h_profile_validity
from .gaussian_scale_space import build_missing_value_audit, build_scale_energy_map
from .ridge_extractor import (
    build_target_day_scale_response,
    extract_scale_local_maxima,
    link_scale_ridges,
    summarize_ridge_families,
)
from .h_w045_scale_classifier import classify_h_w045_scale_identity
from .scale_plotting import (
    plot_derivative_panel,
    plot_scale_energy_map,
    plot_scale_ridge_overlay,
    plot_target_day_scale_response,
)
from .scale_summary_writer import write_scale_summary
from .utils import clean_output_root, now_utc, safe_read_csv, write_dataframe, write_json


def run_h_w045_scale_diagnostic_v10_7_b(project_root: str | Path | None = None) -> dict[str, Any]:
    cfg = ScaleSettings()
    if project_root is not None:
        cfg.with_project_root(Path(project_root))
    output_root = cfg.output.output_root(cfg.project_root)
    clean_output_root(output_root)

    print("[V10.7_b] Loading smoothed fields and rebuilding H object state matrix...")
    smoothed_path = cfg.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    h_profile = build_h_profile(smoothed, cfg.h_profile)
    profile_validity = summarize_h_profile_validity(h_profile)
    state = build_h_state_matrix(h_profile, cfg.state)
    state_matrix = state["state_matrix"]
    valid_day_index = state["valid_day_index"]

    print("[V10.7_b] Building Gaussian derivative scale-energy map...")
    energy_map, feature_norm_audit, scale_meta = build_scale_energy_map(
        state_matrix=state_matrix,
        valid_day_index=valid_day_index,
        sigmas=cfg.scale.sigmas,
        boundary_sigma_multiplier=cfg.scale.boundary_sigma_multiplier,
    )
    missing_audit = build_missing_value_audit(state_matrix, valid_day_index)

    print("[V10.7_b] Extracting local maxima and linking ridge families...")
    local_maxima = extract_scale_local_maxima(
        energy_map=energy_map,
        target_days=cfg.targets.target_days,
        focus_day_min=cfg.scale.focus_day_min,
        focus_day_max=cfg.scale.focus_day_max,
        percentile_threshold=cfg.scale.local_max_percentile_threshold,
        min_prominence_norm=cfg.scale.local_max_min_prominence_norm,
    )
    ridges = link_scale_ridges(local_maxima, ridge_link_radius_days=cfg.scale.ridge_link_radius_days)
    ridge_summary = summarize_ridge_families(ridges, sigmas=cfg.scale.sigmas, target_days=cfg.targets.target_days)
    target_response = build_target_day_scale_response(
        energy_map=energy_map,
        ridges=ridges,
        target_days=cfg.targets.target_days,
        target_radius_days=cfg.scale.target_radius_days,
    )
    decision = classify_h_w045_scale_identity(ridge_summary, target_response, cfg.targets.target_days)

    # Optional references from V10.7_a are copied into run_meta only; they are not used as scale input.
    v10_7a_candidates = safe_read_csv(cfg.v10_7_a_reference.candidate_catalog_path(cfg.project_root))
    v10_7a_atlas = safe_read_csv(cfg.v10_7_a_reference.event_atlas_path(cfg.project_root))

    tables = output_root / "tables"
    write_dataframe(profile_validity, tables / "h_scale_profile_validity_v10_7_b.csv")
    write_dataframe(state["feature_table"], tables / "h_scale_state_feature_table_v10_7_b.csv")
    write_dataframe(feature_norm_audit, tables / "h_scale_feature_normalization_audit_v10_7_b.csv")
    write_dataframe(missing_audit, tables / "h_scale_missing_value_audit_v10_7_b.csv")
    write_dataframe(energy_map, tables / "h_w045_scale_energy_map_v10_7_b.csv")
    write_dataframe(local_maxima, tables / "h_w045_scale_local_maxima_v10_7_b.csv")
    write_dataframe(ridges, tables / "h_w045_scale_ridges_v10_7_b.csv")
    write_dataframe(ridge_summary, tables / "h_w045_ridge_family_summary_v10_7_b.csv")
    write_dataframe(target_response, tables / "h_w045_target_day_scale_response_v10_7_b.csv")
    write_dataframe(decision, tables / "h_w045_scale_interpretation_summary_v10_7_b.csv")

    figures = output_root / "figures"
    print("[V10.7_b] Creating figures...")
    try:
        plot_scale_energy_map(energy_map, cfg.targets.target_days, figures / "h_w045_scale_energy_map_v10_7_b.png")
        plot_scale_ridge_overlay(energy_map, ridges, ridge_summary, cfg.targets.target_days, figures / "h_w045_scale_ridge_overlay_v10_7_b.png")
        plot_target_day_scale_response(target_response, figures / "h_w045_target_day_scale_response_v10_7_b.png")
        plot_derivative_panel(energy_map, cfg.scale.selected_panel_sigmas, cfg.targets.target_days, figures / "h_w045_H19_H35_H45_H57_derivative_panel_v10_7_b.png")
    except Exception as e:
        (figures / "FIGURE_ERROR.txt").write_text(str(e), encoding="utf-8")

    summary_meta = {**scale_meta, "input_smoothed_fields": str(smoothed_path)}
    write_scale_summary(cfg, output_root, ridge_summary, target_response, decision, summary_meta)

    run_meta = {
        "version": "v10.7_b",
        "task": "H W045 Gaussian derivative scale-space diagnostic",
        "created_at_utc": now_utc(),
        "project_root": str(cfg.project_root),
        "input_smoothed_fields": str(smoothed_path),
        "settings": cfg.to_dict(),
        "state_vector_meta": state["state_vector_meta"],
        "scale_backend": scale_meta.get("scale_backend", "unknown"),
        "v10_7_a_reference_status": {
            "candidate_catalog_path": str(cfg.v10_7_a_reference.candidate_catalog_path(cfg.project_root)),
            "candidate_catalog_found": bool(v10_7a_candidates is not None and not v10_7a_candidates.empty),
            "event_atlas_path": str(cfg.v10_7_a_reference.event_atlas_path(cfg.project_root)),
            "event_atlas_found": bool(v10_7a_atlas is not None and not v10_7a_atlas.empty),
            "note": "V10.7_a outputs are optional references only and are not used as scale-diagnostic input.",
        },
        "outputs": {
            "output_root": str(output_root),
            "tables": str(tables),
            "figures": str(figures),
            "summary": str(output_root / "summary_h_w045_scale_diagnostic_v10_7_b.md"),
        },
        "method_boundary": [
            "dedicated scale diagnostic on H object state matrix",
            "not a ruptures.Window rerun",
            "not detector_width sensitivity",
            "not yearwise validation",
            "not cartopy spatial-field validation",
            "not causal or quasi-causal inference",
        ],
        "interpretation_boundary": "heuristic scale-space evidence for selecting follow-up yearwise/spatial targets only",
    }
    write_json(run_meta, output_root / "run_meta" / "run_meta_v10_7_b.json")
    print(f"[V10.7_b] Done. Output root: {output_root}")
    return run_meta
