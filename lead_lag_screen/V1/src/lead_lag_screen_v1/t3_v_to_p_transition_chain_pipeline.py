# -*- coding: utf-8 -*-
"""Pipeline for T3 V->P transition-chain report v1_b."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import traceback

import numpy as np
import pandas as pd

from .t3_v_to_p_transition_chain_settings import TransitionChainReportSettings, settings_to_jsonable
from .t3_v_to_p_transition_chain_io import (
    ensure_output_dirs,
    load_smoothed_fields,
    load_index_values,
    load_previous_region_response,
    get_field_key,
    get_lat_lon_years,
    resolve_day_mapping,
    build_year_index,
    write_text,
)
from .t3_v_to_p_transition_chain_object_state import (
    compute_object_mean_maps,
    build_object_state_region_summary,
    build_object_change_region_delta,
    build_object_delta_maps,
)
from .t3_v_to_p_transition_chain_support import (
    prepare_support_region_rows,
    build_support_transition_matrix,
    build_support_region_delta,
    build_north_main_south_transition_chain,
    compute_positive_lag_max_support_maps,
    build_support_delta_maps,
)
from .t3_v_to_p_transition_chain_correspondence import (
    build_object_support_correspondence,
    build_transition_chain_diagnosis,
)
from .t3_v_to_p_transition_chain_figures import (
    save_chain_maps,
    save_single_map,
    save_transition_panel,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _write_readme(settings: TransitionChainReportSettings) -> None:
    readme = f"""# T3 V→P transition-chain report v1_b

This layer is a reporting / evidence-chain layer. It does **not** rerun V1,
does **not** rerun bootstrap, and does **not** write full-grid NPZ map outputs.

## Purpose

Connect three evidence layers through S3 → T3_early → T3_full → T3_late → S4:

1. precipitation object-state and object-change maps/tables;
2. v850 object-state and object-change maps/tables;
3. V→P support transition tables and support-change maps.

All statements about increase/decrease/shift must be tied to an explicit
comparison, region, and V component.

## Default windows

{settings.windows}

## Output directory

`{settings.output_dir}`
"""
    write_text(settings.summary_dir / "README_T3_V_TO_P_TRANSITION_CHAIN_REPORT_V1_B.md", readme)


def run_t3_v_to_p_transition_chain_report_v1_b(settings: TransitionChainReportSettings) -> Dict[str, Any]:
    ensure_output_dirs(settings)
    run_meta: Dict[str, Any] = {
        "status": "started",
        "layer": "t3_v_to_p_transition_chain_report_v1_b",
        "settings": settings_to_jsonable(settings),
        "full_grid_maps_persisted": False,
        "support_maps_recomputed_in_memory": False,
    }
    try:
        print("[1/9] Loading smooth5 fields and indices...")
        data, field_path = load_smoothed_fields(settings)
        index_df, index_path = load_index_values(settings)
        lat, lon, field_years = get_lat_lon_years(data)
        precip_key = get_field_key(data, settings.precip_aliases)
        v850_key = get_field_key(data, settings.v850_aliases)
        precip = np.asarray(data[precip_key], dtype=float)
        v850 = np.asarray(data[v850_key], dtype=float)
        if precip.shape != v850.shape:
            raise ValueError(f"precip and v850 shapes differ: {precip.shape} vs {v850.shape}")
        day_mapping_mode, day_to_field_i = resolve_day_mapping(precip, data)
        year_mapping_mode, year_to_field_i = build_year_index(field_years, index_df["year"].values, precip.shape[0])
        run_meta.update({
            "input_smoothed_fields_path": str(field_path),
            "input_index_values_path": str(index_path),
            "precip_key": precip_key,
            "v850_key": v850_key,
            "field_shape": list(precip.shape),
            "day_mapping_mode": day_mapping_mode,
            "year_mapping_mode": year_mapping_mode,
            "n_lat": int(len(lat)),
            "n_lon": int(len(lon)),
        })

        print("[2/9] Computing precipitation/v850 object-state maps...")
        p_maps, v_maps = compute_object_mean_maps(precip, v850, settings, day_to_field_i)
        object_state_df = build_object_state_region_summary(p_maps, v_maps, lat, lon, settings)
        object_delta_df = build_object_change_region_delta(object_state_df, settings)
        p_delta_maps, v_delta_maps = build_object_delta_maps(p_maps, v_maps)
        object_state_df.to_csv(settings.tables_dir / "object_state_region_summary.csv", index=False, encoding="utf-8-sig")
        object_delta_df.to_csv(settings.tables_dir / "object_change_region_delta.csv", index=False, encoding="utf-8-sig")

        print("[3/9] Loading previous field-explanation support tables...")
        prev_region_df = load_previous_region_response(settings)
        support_rows = prepare_support_region_rows(prev_region_df, settings)
        support_matrix_df = build_support_transition_matrix(support_rows, settings)
        support_delta_df = build_support_region_delta(support_matrix_df, settings)
        nms_df = build_north_main_south_transition_chain(support_matrix_df, settings)
        support_rows.to_csv(settings.tables_dir / "support_region_long_from_previous_audit.csv", index=False, encoding="utf-8-sig")
        support_matrix_df.to_csv(settings.tables_dir / "support_region_transition_matrix.csv", index=False, encoding="utf-8-sig")
        support_delta_df.to_csv(settings.tables_dir / "support_region_delta.csv", index=False, encoding="utf-8-sig")
        nms_df.to_csv(settings.tables_dir / "north_main_south_transition_chain.csv", index=False, encoding="utf-8-sig")

        print("[4/9] Building object-support correspondence tables...")
        correspondence_df = build_object_support_correspondence(object_delta_df, support_delta_df, settings)
        diagnosis_df = build_transition_chain_diagnosis(support_matrix_df, support_delta_df, nms_df, correspondence_df, settings)
        correspondence_df.to_csv(settings.tables_dir / "object_support_correspondence_summary.csv", index=False, encoding="utf-8-sig")
        diagnosis_df.to_csv(settings.tables_dir / "transition_chain_diagnosis_table.csv", index=False, encoding="utf-8-sig")

        figure_manifest = []
        support_maps = None
        support_delta_maps = None
        if settings.make_figures:
            print("[5/9] Recomputing observed support maps in memory for figures only...")
            support_maps = compute_positive_lag_max_support_maps(index_df, precip, settings, year_to_field_i, day_to_field_i)
            support_delta_maps = build_support_delta_maps(support_maps)
            run_meta["support_maps_recomputed_in_memory"] = True

            print("[6/9] Drawing object-state and object-change maps...")
            path = settings.figures_dir / "P_mean_state_S3_T3early_T3full_T3late_S4_cartopy.png"
            save_chain_maps(p_maps, lat, lon, settings, path, "P mean state", cmap="YlGnBu", diverging=False)
            figure_manifest.append({"figure": str(path), "type": "P_mean_state"})
            path = settings.figures_dir / "V850_mean_state_S3_T3early_T3full_T3late_S4_cartopy.png"
            save_chain_maps(v_maps, lat, lon, settings, path, "V850 mean state", cmap="RdBu_r", diverging=True)
            figure_manifest.append({"figure": str(path), "type": "V850_mean_state"})

            for comp_name, maps, prefix in [("P", p_delta_maps, "P_change"), ("V850", v_delta_maps, "V850_change")]:
                for comparison, arr in maps.items():
                    path = settings.figures_dir / f"{prefix}_{comparison}_cartopy.png"
                    save_single_map(arr, lat, lon, settings, path, f"{comp_name} {comparison}", cmap="RdBu_r", diverging=True)
                    figure_manifest.append({"figure": str(path), "type": prefix, "comparison": comparison})

            print("[7/9] Drawing support-chain and support-delta maps...")
            assert support_maps is not None and support_delta_maps is not None
            for comp, maps in support_maps.items():
                path = settings.figures_dir / f"{comp}_support_chain_S3_T3early_T3full_T3late_S4_cartopy.png"
                save_chain_maps(maps, lat, lon, settings, path, f"{comp} support positive-lag max R²", cmap="viridis", diverging=False)
                figure_manifest.append({"figure": str(path), "type": "support_chain", "v_component": comp})
                for comparison, arr in support_delta_maps[comp].items():
                    path = settings.figures_dir / f"{comp}_support_{comparison}_cartopy.png"
                    save_single_map(arr, lat, lon, settings, path, f"{comp} support R² {comparison}", cmap="RdBu_r", diverging=True)
                    figure_manifest.append({"figure": str(path), "type": "support_delta", "v_component": comp, "comparison": comparison})

            print("[8/9] Drawing P/V/support joint transition panels...")
            for comparison in ["T3_full_minus_S3", "T3_late_minus_T3_early", "S4_minus_T3_full"]:
                path = settings.figures_dir / f"P_V_support_transition_panel_{comparison}.png"
                save_transition_panel(p_delta_maps[comparison], v_delta_maps[comparison], {c: support_delta_maps[c][comparison] for c in settings.v_components}, lat, lon, settings, path, comparison)
                figure_manifest.append({"figure": str(path), "type": "P_V_support_transition_panel", "comparison": comparison})

        pd.DataFrame(figure_manifest).to_csv(settings.tables_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig")

        print("[9/9] Writing summaries...")
        _write_readme(settings)
        summary = {
            "status": "success",
            "layer": "t3_v_to_p_transition_chain_report_v1_b",
            "output_dir": str(settings.output_dir),
            "purpose": "S3→T3→S4 precipitation-v850-support transition-chain evidence report.",
            "n_object_state_rows": int(len(object_state_df)),
            "n_object_delta_rows": int(len(object_delta_df)),
            "n_support_transition_rows": int(len(support_matrix_df)),
            "n_support_delta_rows": int(len(support_delta_df)),
            "n_correspondence_rows": int(len(correspondence_df)),
            "n_diagnosis_rows": int(len(diagnosis_df)),
            "n_figures": int(len(figure_manifest)),
            "full_grid_maps_persisted": False,
            "support_maps_recomputed_in_memory_for_figures_only": bool(run_meta.get("support_maps_recomputed_in_memory", False)),
            "critical_interpretation_rule": "Every increase/decrease/shift statement must name comparison, region, and V component.",
        }
        run_meta["status"] = "success"
        _write_json(settings.summary_dir / "summary.json", summary)
        _write_json(settings.summary_dir / "run_meta.json", run_meta)
        return summary
    except Exception as exc:
        run_meta["status"] = "failed"
        run_meta["error"] = repr(exc)
        run_meta["traceback"] = traceback.format_exc()
        _write_json(settings.summary_dir / "run_meta.json", run_meta)
        raise
