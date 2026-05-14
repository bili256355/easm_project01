from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .event_content_config import EventContentConfig
from .event_content_io import load_h_content_state, load_optional_v10_7_b_outputs, load_spatial_field
from .feature_contribution import compute_feature_contribution
from .profile_event_diff import compute_profile_event_diff, pivot_event_diff_vectors
from .spatial_composite import compute_spatial_composites, compute_spatial_composites_object_domain
from .event_similarity import compute_event_similarity
from .yearwise_content import compute_yearwise_content
from .event_content_classifier import classify_event_content
from .event_content_plotting import (
    plot_feature_contribution_top,
    plot_profile_diff_panel,
    plot_similarity_heatmap,
    plot_spatial_panel,
)
from .event_content_summary import write_event_content_summary
from .utils import clean_output_root, now_utc, write_dataframe, write_json


def _write_outputs(tables: dict[str, pd.DataFrame], out_tables: Path) -> None:
    for name, df in tables.items():
        if df is None:
            continue
        write_dataframe(df, out_tables / name)


def run_h_w045_event_content_audit_v10_7_c(project_root: Path | str) -> dict[str, Any]:
    project_root = Path(project_root)
    cfg = EventContentConfig()
    output_root = cfg.output_root(project_root)
    clean_output_root(output_root)
    tables_dir = output_root / "tables"
    figs_dir = output_root / "figures"
    meta: dict[str, Any] = {
        "version": cfg.version,
        "task": "H W045 H18/H35/H45 event-content audit",
        "created_utc": now_utc(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "config": cfg.to_dict(),
        "method_boundary": [
            "not influence test",
            "not causal inference",
            "not lead-lag test",
            "not detector rerun",
            "event content audit only",
        ],
        "input_status": {},
    }

    # 1. H profile/state reconstruction.
    try:
        h_state, state_meta = load_h_content_state(project_root)
        meta["input_status"]["profile_status"] = h_state.profile_status
        meta["h_state_meta"] = state_meta
    except Exception as exc:
        # Hard failure: profile content is the minimum required layer.
        meta["input_status"]["profile_status"] = f"failed: {exc}"
        write_json(meta, output_root / "run_meta" / "run_meta_v10_7_c.json")
        raise

    # 2. Optional spatial field.
    spatial, spatial_meta = load_spatial_field(
        project_root,
        cfg.possible_h_field_keys,
        cfg.possible_lat_keys,
        cfg.possible_lon_keys,
        cfg.possible_year_keys,
    )
    meta["input_status"]["spatial_status"] = spatial_meta.get("spatial_status", "unknown")
    meta["spatial_input_meta"] = spatial_meta

    # 3. Optional V10.7_b scale context.
    scale_outputs = load_optional_v10_7_b_outputs(project_root)
    meta["input_status"]["v10_7_b_scale_context"] = "loaded" if any(v is not None and not v.empty for v in scale_outputs.values()) else "missing_or_empty"

    # 4. Profile content diff.
    profile_diff = compute_profile_event_diff(h_state.raw_matrix, h_state.feature_table, cfg)

    # 5. Feature derivative contribution.
    feature_contrib, feature_norm_audit, feat_meta = compute_feature_contribution(
        h_state.state_matrix,
        h_state.valid_day_index,
        h_state.feature_table,
        cfg,
    )
    meta["feature_contribution_meta"] = feat_meta

    # 6. Spatial composite content.
    # Full-domain metrics are retained as context/background reference.
    spatial_metrics, diff_maps, spatial_comp_meta = compute_spatial_composites(spatial, cfg)
    # H-object-domain metrics are the primary spatial evidence for H-content classification.
    spatial_metrics_obj, diff_maps_obj, spatial_comp_meta_obj = compute_spatial_composites_object_domain(spatial, cfg)
    meta["spatial_composite_meta_full_domain"] = {k: v for k, v in spatial_comp_meta.items() if k not in ("lat_values", "lon_values", "yearly_maps")}
    meta["spatial_composite_meta_object_domain"] = {k: v for k, v in spatial_comp_meta_obj.items() if k not in ("lat_values", "lon_values", "yearly_maps")}
    lat_values = spatial_comp_meta.get("lat_values")
    lon_values = spatial_comp_meta.get("lon_values")
    lat_values_obj = spatial_comp_meta_obj.get("lat_values")
    lon_values_obj = spatial_comp_meta_obj.get("lon_values")

    # 7. Similarity.
    profile_vectors = pivot_event_diff_vectors(profile_diff)
    similarity = compute_event_similarity(profile_vectors, diff_maps, profile_diff)
    similarity_obj = compute_event_similarity(profile_vectors, diff_maps_obj, profile_diff)
    if not similarity_obj.empty:
        similarity_obj = similarity_obj.copy()
        similarity_obj["spatial_domain_used"] = "h_object_domain"
    if not similarity.empty:
        similarity = similarity.copy()
        similarity["spatial_domain_used"] = "full_domain_context"

    # 8. Yearwise consistency if possible.
    # Current yearwise audit keeps the legacy full-domain context; object-domain yearwise can be added later if needed.
    yearwise, yearwise_summary, ymeta = compute_yearwise_content(spatial, cfg, diff_maps)
    meta["input_status"]["yearwise_status"] = ymeta.get("yearwise_status", "unknown")

    # 9. Role summary.
    # HOTFIX01: use H-object-domain spatial metrics/similarity for role classification when available.
    primary_spatial_metrics = spatial_metrics_obj if spatial_metrics_obj is not None and not spatial_metrics_obj.empty else spatial_metrics
    primary_similarity = similarity_obj if similarity_obj is not None and not similarity_obj.empty else similarity
    role_summary = classify_event_content(profile_diff, primary_spatial_metrics, primary_similarity, yearwise_summary, scale_outputs)

    # 10. Write tables.
    _write_outputs({
        "h_w045_event_profile_diff_v10_7_c.csv": profile_diff,
        "h_w045_event_feature_contribution_v10_7_c.csv": feature_contrib,
        "h_w045_event_feature_normalization_audit_v10_7_c.csv": feature_norm_audit,
        "h_w045_event_spatial_composite_metrics_v10_7_c.csv": spatial_metrics,
        "h_w045_event_spatial_composite_metrics_object_domain_v10_7_c.csv": spatial_metrics_obj,
        "h_w045_H18_H35_similarity_v10_7_c.csv": similarity,
        "h_w045_H18_H35_similarity_object_domain_v10_7_c.csv": similarity_obj,
        "h_w045_event_yearwise_change_v10_7_c.csv": yearwise,
        "h_w045_event_yearwise_consistency_summary_v10_7_c.csv": yearwise_summary,
        "h_w045_event_content_role_summary_v10_7_c.csv": role_summary,
    }, tables_dir)

    # Save spatial map arrays as npz for exact reuse; tables/figures remain the primary audit outputs.
    if diff_maps:
        np.savez_compressed(output_root / "tables" / "h_w045_event_spatial_diff_maps_v10_7_c.npz", **diff_maps)
    if diff_maps_obj:
        np.savez_compressed(output_root / "tables" / "h_w045_event_spatial_diff_maps_object_domain_v10_7_c.npz", **diff_maps_obj)

    # 11. Figures.
    plot_profile_diff_panel(profile_diff, figs_dir / "h_w045_H18_H35_H45_profile_diff_panel_v10_7_c.png")
    plot_feature_contribution_top(feature_contrib, figs_dir / "h_w045_event_feature_contribution_top_v10_7_c.png")
    map_backend = plot_spatial_panel(diff_maps, lat_values, lon_values, figs_dir / "h_w045_H18_H35_H45_H57_spatial_diff_panel_cartopy_v10_7_c.png")
    map_backend_obj = plot_spatial_panel(diff_maps_obj, lat_values_obj, lon_values_obj, figs_dir / "h_w045_H18_H35_H45_H57_spatial_diff_panel_object_domain_cartopy_v10_7_c.png")
    meta["spatial_plot_backend_full_domain"] = map_backend
    meta["spatial_plot_backend_object_domain"] = map_backend_obj
    plot_similarity_heatmap(similarity, figs_dir / "h_w045_H18_H35_similarity_heatmap_v10_7_c.png")
    plot_similarity_heatmap(similarity_obj, figs_dir / "h_w045_H18_H35_similarity_heatmap_object_domain_v10_7_c.png")

    # 12. Summary and run meta.
    write_event_content_summary(
        output_root / "summary_h_w045_event_content_audit_v10_7_c.md",
        meta,
        profile_diff,
        spatial_metrics,
        spatial_metrics_obj,
        similarity,
        similarity_obj,
        yearwise_summary,
        role_summary,
    )
    write_json(meta, output_root / "run_meta" / "run_meta_v10_7_c.json")
    return meta
