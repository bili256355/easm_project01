# -*- coding: utf-8 -*-
"""Pipeline for P/V850 offset-correspondence audit v1_b."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .t3_p_v_offset_correspondence_settings import PVOffsetCorrespondenceSettings, settings_to_jsonable
from .t3_p_v_offset_correspondence_io import (
    ensure_output_dirs,
    get_field_key,
    get_lat_lon_years,
    load_smoothed_fields,
    resolve_day_mapping,
    write_text,
)
from .t3_p_v_offset_correspondence_core import (
    build_diagnosis_table,
    build_p_change_band_summary,
    build_p_change_peak_to_v_change_structure,
    build_p_clim_band_summary,
    build_p_clim_band_to_v_clim_structure,
    build_p_highlat_v_north_edge_correspondence,
    build_profile_delta,
    compute_lat_profiles,
    compute_window_mean_maps,
    v_change_structure_summary,
    v_clim_structure_summary,
)
from .t3_p_v_offset_correspondence_figures import (
    plot_p_change_vs_v_change,
    plot_p_clim_bands_vs_v_clim_structure,
    plot_p_highlat_v_north_edge_chain,
    plot_p_south_retention_vs_v_retreat,
)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _readme_text() -> str:
    return """# T3 P/V850 Offset-Correspondence Audit v1_b

This output is an object-layer offset-correspondence audit. It separates:

- `P_clim_band`: peaks/bands on window-mean precipitation profiles.
- `P_change_peak` / `P_change_band`: positive/negative centers on window-difference precipitation profiles.
- `V_clim_structure`: V850 positive peak/centroid/edges on window-mean V850 profiles.
- `V_change_structure`: V850 change peak/trough/gradient and V850 positive-edge shifts.

It does not compute V->P support, R2, lag/tau0, pathway, or causality.

The central purpose is to avoid same-region same-sign overinterpretation and to test whether P object changes are more consistent with pre-registered V850 offset, edge, or gradient structures.
"""


def run_t3_p_v_offset_correspondence_audit_v1_b(settings: PVOffsetCorrespondenceSettings) -> Dict[str, object]:
    dirs = ensure_output_dirs(settings)
    data, field_path = load_smoothed_fields(settings)
    lat, lon, years = get_lat_lon_years(data)

    p_key = get_field_key(data, settings.precip_aliases)
    v_key = get_field_key(data, settings.v850_aliases)
    p_field = np.asarray(data[p_key], dtype=float)
    v_field = np.asarray(data[v_key], dtype=float)
    if p_field.shape != v_field.shape:
        raise ValueError(f"P and V850 fields must have same shape. P={p_field.shape}, V={v_field.shape}")

    day_mode, day_map = resolve_day_mapping(p_field, data)

    p_maps = compute_window_mean_maps(p_field, settings.windows, day_map)
    v_maps = compute_window_mean_maps(v_field, settings.windows, day_map)

    p_profile = compute_lat_profiles(p_maps, lat, lon, settings.lon_sectors, settings.window_order)
    v_profile = compute_lat_profiles(v_maps, lat, lon, settings.lon_sectors, settings.window_order)
    p_delta = build_profile_delta(p_profile, settings, "P")
    v_delta = build_profile_delta(v_profile, settings, "V850")

    p_clim_band_df = build_p_clim_band_summary(p_profile, settings)
    p_change_band_df = build_p_change_band_summary(p_delta, settings)
    v_clim_df = v_clim_structure_summary(v_profile)
    v_change_df = v_change_structure_summary(v_delta, v_clim_df, settings)

    p_clim_to_v_df = build_p_clim_band_to_v_clim_structure(p_clim_band_df, v_profile, v_clim_df, settings)
    p_change_to_v_df = build_p_change_peak_to_v_change_structure(p_change_band_df, v_delta, v_change_df, settings)
    highlat_corr_df = build_p_highlat_v_north_edge_correspondence(p_profile, v_clim_df, settings)
    diagnosis_df = build_diagnosis_table(p_clim_to_v_df, p_change_to_v_df, highlat_corr_df, settings)

    # Tables.
    table_paths = {}
    for name, df in {
        "p_lat_profile_long.csv": p_profile,
        "v850_lat_profile_long.csv": v_profile,
        "p_lat_profile_delta_long.csv": p_delta,
        "v850_lat_profile_delta_long.csv": v_delta,
        "p_clim_band_summary.csv": p_clim_band_df,
        "p_change_band_summary.csv": p_change_band_df,
        "v_clim_structure_summary.csv": v_clim_df,
        "v_change_structure_summary.csv": v_change_df,
        "p_clim_band_to_v_clim_structure_summary.csv": p_clim_to_v_df,
        "p_change_peak_to_v_change_structure_summary.csv": p_change_to_v_df,
        "p_highlat_v_north_edge_correspondence.csv": highlat_corr_df,
        "p_v_offset_correspondence_diagnosis_table.csv": diagnosis_df,
    }.items():
        path = settings.tables_dir / name
        _write_csv(df, path)
        table_paths[name] = str(path)

    fig_paths = []
    if settings.make_figures:
        try:
            path = settings.figures_dir / "P_clim_bands_vs_V_clim_structure_chain.png"
            plot_p_clim_bands_vs_v_clim_structure(p_profile, v_profile, p_clim_band_df, v_clim_df, settings, path)
            fig_paths.append(str(path))
            fig_paths.extend([str(p) for p in plot_p_change_vs_v_change(p_delta, v_delta, p_change_band_df, v_change_df, settings, settings.figures_dir)])
            path = settings.figures_dir / "P_highlat_V_north_edge_chain.png"
            plot_p_highlat_v_north_edge_chain(highlat_corr_df, v_clim_df, p_profile, settings, path)
            fig_paths.append(str(path))
            path = settings.figures_dir / "P_south_retention_vs_V_retreat_chain.png"
            plot_p_south_retention_vs_v_retreat(v_clim_df, p_profile, settings, path)
            fig_paths.append(str(path))
        except Exception as exc:  # Keep tables usable if plotting fails.
            fig_paths.append(f"FIGURE_ERROR: {type(exc).__name__}: {exc}")

    summary = {
        "status": "success",
        "audit": "t3_p_v_offset_correspondence_audit_v1_b",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(settings.output_dir),
        "input_smoothed_fields_path": str(field_path),
        "precip_key": p_key,
        "v850_key": v_key,
        "field_shape": list(p_field.shape),
        "lat_min_data": float(np.nanmin(lat)),
        "lat_max_data": float(np.nanmax(lat)),
        "lon_min_data": float(np.nanmin(lon)),
        "lon_max_data": float(np.nanmax(lon)),
        "day_mapping_mode": day_mode,
        "window_definition": settings.windows,
        "lon_sectors": {k: v.as_dict() for k, v in settings.lon_sectors.items()},
        "conceptual_separation": {
            "P_clim_band": "Window-mean precipitation-profile peak/band; not a change peak.",
            "P_change_peak_or_band": "Window-difference precipitation-profile positive/negative center; not a climatological rain band.",
            "V_clim_structure": "Window-mean V850 positive peak/centroid/edge structure.",
            "V_change_structure": "Window-difference V850 peak/trough/gradient and edge shifts.",
        },
        "tables": table_paths,
        "figures": fig_paths,
        "notes": [
            "This audit does not compute V->P support/R2/lag/pathway or causality.",
            "Same-region P/V sign agreement is not treated as the default correspondence rule.",
            "All offset positions are pre-registered fixed offsets or V structure positions, not free post-hoc matches.",
        ],
    }

    run_meta = {
        "settings": settings_to_jsonable(settings),
        "summary": summary,
    }
    write_text(settings.summary_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    write_text(settings.summary_dir / "run_meta.json", json.dumps(run_meta, indent=2, ensure_ascii=False, default=str))
    write_text(settings.summary_dir / "README_T3_P_V_OFFSET_CORRESPONDENCE_AUDIT_V1_B.md", _readme_text())
    return summary
