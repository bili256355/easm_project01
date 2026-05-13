# -*- coding: utf-8 -*-
"""Pipeline for P/V850 latitudinal object-change audit v1_a."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .t3_p_v_latitudinal_object_change_settings import (
    PVLatitudinalObjectChangeSettings,
    settings_to_jsonable,
)
from .t3_p_v_latitudinal_object_change_io import (
    ensure_output_dirs,
    get_field_key,
    get_lat_lon_years,
    load_smoothed_fields,
    resolve_day_mapping,
    write_text,
)
from .t3_p_v_latitudinal_object_change_core import (
    build_object_change_region_delta,
    build_p_v_latitudinal_feature_summary,
    build_precip_band_transition_links,
    build_precip_multiband_summary,
    build_profile_delta,
    compute_lat_profiles,
    compute_window_mean_maps,
    precip_feature_summary,
    summarize_latbands,
    summarize_regions,
    v850_feature_summary,
)
from .t3_p_v_latitudinal_object_change_figures import (
    plot_comparison_map,
    plot_p_v_object_panel,
    plot_profile_chain,
    plot_profile_delta_chain,
    plot_window_map_chain,
)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _window_max_subwindow_delta_for_regions(state_df: pd.DataFrame, settings: PVLatitudinalObjectChangeSettings) -> pd.DataFrame:
    rows = []
    idx = {(r.window, r.region): r for r in state_df.itertuples(index=False)}
    for region in settings.regions.keys():
        full = idx[("T3_full", region)]
        early = idx[("T3_early", region)]
        late = idx[("T3_late", region)]
        p_max = max(float(early.P_mean), float(late.P_mean))
        v_max = max(float(early.V850_mean), float(late.V850_mean))
        rows.append({
            "comparison": "T3_full_minus_max_subwindow",
            "target_window": "T3_full",
            "reference_window": "max(T3_early,T3_late)",
            "region": region,
            "P_delta": float(full.P_mean - p_max),
            "P_delta_percent": float((full.P_mean - p_max) / p_max * 100.0) if abs(p_max) > 1e-12 else np.nan,
            "P_direction": "increase" if full.P_mean - p_max > settings.p_delta_epsilon else "decrease" if full.P_mean - p_max < -settings.p_delta_epsilon else "near_zero",
            "V850_delta": float(full.V850_mean - v_max),
            "V850_delta_percent": float((full.V850_mean - v_max) / v_max * 100.0) if abs(v_max) > 1e-12 else np.nan,
            "V850_direction": "increase" if full.V850_mean - v_max > settings.v_delta_epsilon else "decrease" if full.V850_mean - v_max < -settings.v_delta_epsilon else "near_zero",
            "V850_abs_delta": np.nan,
            "V850_abs_delta_percent": np.nan,
        })
    return pd.DataFrame(rows)


def _diagnose_object_changes(
    state_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    p_latband_df: pd.DataFrame,
    v_latband_df: pd.DataFrame,
    band_df: pd.DataFrame,
    link_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    settings: PVLatitudinalObjectChangeSettings,
    lat_min_data: float,
    lat_max_data: float,
) -> pd.DataFrame:
    rows = []

    def add(diagnosis_id, support_level, comparison, region_or_latband, evidence_value, allowed, forbidden):
        target, ref = ("", "")
        if comparison in settings.comparisons:
            target, ref = settings.comparisons[comparison]
        rows.append({
            "diagnosis_id": diagnosis_id,
            "support_level": support_level,
            "comparison": comparison,
            "reference_window": ref,
            "target_window": target,
            "region_or_latband": region_or_latband,
            "evidence_value": evidence_value,
            "allowed_statement": allowed,
            "forbidden_statement": forbidden,
        })

    def delta_lookup(comp: str, region: str, col: str):
        sub = delta_df[(delta_df["comparison"] == comp) & (delta_df["region"] == region)]
        if sub.empty:
            return np.nan
        return float(sub.iloc[0][col])

    # P main/south/north S3 -> T3_full.
    for region, label in [("main_meiyu", "main/Meiyu"), ("south_china_scs", "south/SCS"), ("north_northeast", "north/northeast")]:
        d = delta_lookup("T3_full_minus_S3", region, "P_delta")
        level = "increase" if d > settings.p_delta_epsilon else "decrease" if d < -settings.p_delta_epsilon else "near_zero"
        add(
            f"P_{region}_change_from_S3_to_T3_full",
            level,
            "T3_full_minus_S3",
            region,
            d,
            f"Relative to S3, T3_full {label} precipitation {level} by {d:.6g} in the object-layer mean.",
            f"Do not describe {label} precipitation change without stating the S3 reference window.",
        )

    # North escape: check if fixed north decreases but higher latitude bands increase.
    # Uses 45N+ or 50N+ bands if available.
    s4_vs_t3 = "S4_minus_T3_full"
    north_delta = delta_lookup(s4_vs_t3, "north_northeast", "P_delta")
    high_band_msg = "No higher-latitude band above 50N is available in the data."
    high_increase = False
    if not p_latband_df.empty:
        # Compute high-lat P changes for full_easm_lon between S4 and T3_full.
        high = p_latband_df[(p_latband_df["sector"] == "full_easm_lon") & (p_latband_df["lat_min"] >= 50.0)]
        vals = []
        for band in sorted(high["lat_band"].unique()):
            s4 = high[(high["window"] == "S4") & (high["lat_band"] == band)]
            t3 = high[(high["window"] == "T3_full") & (high["lat_band"] == band)]
            if not s4.empty and not t3.empty:
                dd = float(s4.iloc[0]["P_mean"] - t3.iloc[0]["P_mean"])
                vals.append((band, dd))
                if dd > settings.p_delta_epsilon:
                    high_increase = True
        if vals:
            high_band_msg = "; ".join([f"{b}:S4-T3_full={d:.6g}" for b, d in vals])
    if np.isfinite(north_delta) and north_delta < -settings.p_delta_epsilon and high_increase:
        add(
            "P_north_box_decline_but_higher_latitude_increase",
            "higher_latitude_escape_possible",
            s4_vs_t3,
            "north_northeast_and_latbands_ge_50N",
            f"north_box_delta={north_delta:.6g}; {high_band_msg}",
            "The fixed 35-50N north box declines from T3_full to S4, but higher-latitude precipitation increases in at least one available band.",
            "Do not call the north-side change a true retreat without checking higher-latitude bands.",
        )
    elif np.isfinite(north_delta) and north_delta < -settings.p_delta_epsilon:
        add(
            "P_north_box_decline_without_detected_higher_lat_escape",
            "fixed_box_decline_no_higher_lat_compensation_detected",
            s4_vs_t3,
            "north_northeast_and_latbands_ge_50N",
            f"north_box_delta={north_delta:.6g}; {high_band_msg}",
            "Within available latitude bands, no higher-latitude precipitation compensation is detected for the fixed north-box decline.",
            "Do not generalize beyond the actual data latitude maximum.",
        )
    else:
        add(
            "P_north_continuation_or_escape_check",
            "not_a_fixed_box_decline_case",
            s4_vs_t3,
            "north_northeast_and_latbands_ge_50N",
            f"north_box_delta={north_delta:.6g}; {high_band_msg}",
            "The fixed north-box decline condition is not met; inspect latband tables before using retreat language.",
            "Do not infer north retreat from a non-decline case.",
        )

    # Multiband presence.
    mb = band_df[band_df["multiband"] == True] if not band_df.empty else pd.DataFrame()
    add(
        "P_multiband_structure_present",
        "present" if not mb.empty else "not_detected_by_default_thresholds",
        "all_windows_all_sectors",
        "precip_latitudinal_profiles",
        int(len(mb)),
        "Use multi-band language where n_precip_bands>=2; single-band migration language is not allowed for those window/sector cases.",
        "Do not summarize precipitation evolution with a single peak/centroid when multiband=true.",
    )

    # Single-band northward shift support via continue links.
    cont = link_df[link_df["link_type"] == "continue"] if not link_df.empty else pd.DataFrame()
    north_shifts = cont[cont["peak_lat_shift"] > 0] if not cont.empty else pd.DataFrame()
    add(
        "P_single_band_northward_shift_supported_cases",
        "case_count_only",
        "adjacent_windows",
        "precip_band_links",
        int(len(north_shifts)),
        "Only links marked continue with positive peak_lat_shift can be described as a band northward shift, and only for that specific sector/window pair.",
        "Do not use northward-shift language for emerge/disappear/split-like changes.",
    )

    # V850 northward extension check via positive north edge.
    if not feature_df.empty:
        for comp, (target, ref) in settings.comparisons.items():
            sub_t = feature_df[(feature_df["window"] == target) & (feature_df["sector"] == "full_easm_lon")]
            sub_r = feature_df[(feature_df["window"] == ref) & (feature_df["sector"] == "full_easm_lon")]
            if not sub_t.empty and not sub_r.empty:
                dd = float(sub_t.iloc[0]["V850_positive_north_edge"] - sub_r.iloc[0]["V850_positive_north_edge"])
                level = "north_edge_extends" if dd > 0 else "north_edge_retracts" if dd < 0 else "north_edge_unchanged"
                add(
                    f"V850_positive_north_edge_{comp}",
                    level,
                    comp,
                    "full_easm_lon",
                    dd,
                    f"For v850 positive values, the north edge {level} by {dd:.6g} degrees in {comp}.",
                    "Do not infer precipitation-band movement from v850 north-edge change alone.",
                )

    # Data latitude coverage.
    add(
        "data_latitude_coverage",
        "coverage_metadata",
        "not_a_comparison",
        "data_domain",
        f"lat_min={lat_min_data:.6g}; lat_max={lat_max_data:.6g}",
        "All northward-escape statements are limited by this actual data latitude coverage.",
        "Do not claim absence beyond lat_max_data.",
    )

    return pd.DataFrame(rows)


def _write_readme(settings: PVLatitudinalObjectChangeSettings) -> str:
    return f"""# T3 P/V850 Latitudinal Object-Change Audit v1_a

This layer diagnoses **object-level changes only** for precipitation and v850.
It does not compute or read V->P support/R2/pathway evidence.

## Scope

- P mean state and change maps.
- V850 mean state and change maps.
- P latitudinal profiles, multi-band detection, and cross-window band links.
- V850 raw / positive / absolute latitudinal profiles and positive north-edge diagnostics.
- Lat-band summaries using the actual data latitude range.

## Default windows

{settings.windows}

## Key restriction

Do not use this output to claim V causes/explains P. This output only states how
P and V850 objects themselves change across S3 -> T3 -> S4.
"""


def run_t3_p_v_latitudinal_object_change_audit_v1_a(settings: PVLatitudinalObjectChangeSettings) -> Dict[str, object]:
    started = datetime.now().isoformat(timespec="seconds")
    dirs = ensure_output_dirs(settings)
    print("[1/9] Loading smooth5 fields...")
    data, field_path = load_smoothed_fields(settings)
    p_key = get_field_key(data, settings.precip_aliases)
    v_key = get_field_key(data, settings.v850_aliases)
    lat, lon, years = get_lat_lon_years(data)
    p_field = np.asarray(data[p_key], dtype=float)
    v_field = np.asarray(data[v_key], dtype=float)
    if p_field.shape != v_field.shape:
        raise ValueError(f"P and V fields must have same shape. Got {p_field.shape} vs {v_field.shape}.")
    day_mode, day_map = resolve_day_mapping(p_field, data)
    lat_min_data = float(np.nanmin(lat))
    lat_max_data = float(np.nanmax(lat))

    print("[2/9] Computing P/V850 window mean maps...")
    p_maps = compute_window_mean_maps(p_field, settings.windows, day_map)
    v_maps = compute_window_mean_maps(v_field, settings.windows, day_map)

    print("[3/9] Summarizing regional object states and changes...")
    state_df = summarize_regions(p_maps, v_maps, lat, lon, settings)
    delta_df = build_object_change_region_delta(state_df, settings)
    delta_df = pd.concat([delta_df, _window_max_subwindow_delta_for_regions(state_df, settings)], ignore_index=True)
    _write_csv(state_df, settings.tables_dir / "object_state_region_summary.csv")
    _write_csv(delta_df, settings.tables_dir / "object_change_region_delta.csv")

    print("[4/9] Computing latitudinal profiles and lat-band summaries...")
    p_profile = compute_lat_profiles(p_maps, lat, lon, settings.lon_sectors, settings.window_order)
    v_profile = compute_lat_profiles(v_maps, lat, lon, settings.lon_sectors, settings.window_order)
    _write_csv(p_profile.rename(columns={"value": "P_profile"}), settings.tables_dir / "precip_lat_profile_long.csv")
    _write_csv(v_profile.rename(columns={"value": "V850_profile"}), settings.tables_dir / "v850_lat_profile_long.csv")

    p_delta_prof = build_profile_delta(p_profile, settings, "P")
    v_delta_prof = build_profile_delta(v_profile, settings, "V850")
    _write_csv(p_delta_prof, settings.tables_dir / "precip_lat_profile_delta_long.csv")
    _write_csv(v_delta_prof, settings.tables_dir / "v850_lat_profile_delta_long.csv")

    p_latband, v_latband = summarize_latbands(p_maps, v_maps, lat, lon, settings)
    _write_csv(p_latband, settings.tables_dir / "precip_latband_integrated_summary.csv")
    _write_csv(v_latband, settings.tables_dir / "v850_latband_summary.csv")

    print("[5/9] Detecting precipitation multi-band structures...")
    # For band detection functions, profile column must be named value.
    band_df = build_precip_multiband_summary(p_profile, settings)
    link_df = build_precip_band_transition_links(band_df, settings)
    _write_csv(band_df, settings.tables_dir / "precip_multiband_summary.csv")
    _write_csv(link_df, settings.tables_dir / "precip_band_transition_links.csv")

    print("[6/9] Computing P/V latitudinal feature summaries...")
    p_feat = precip_feature_summary(band_df, p_profile)
    v_feat = v850_feature_summary(v_profile, settings)
    feature_df = build_p_v_latitudinal_feature_summary(p_feat, v_feat)
    _write_csv(v_feat, settings.tables_dir / "v850_latitudinal_feature_summary.csv")
    _write_csv(feature_df, settings.tables_dir / "p_v_latitudinal_feature_summary.csv")

    print("[7/9] Building object-change diagnosis table...")
    diag_df = _diagnose_object_changes(
        state_df=state_df,
        delta_df=delta_df,
        p_latband_df=p_latband,
        v_latband_df=v_latband,
        band_df=band_df,
        link_df=link_df,
        feature_df=feature_df,
        settings=settings,
        lat_min_data=lat_min_data,
        lat_max_data=lat_max_data,
    )
    _write_csv(diag_df, settings.tables_dir / "p_v_object_change_diagnosis_table.csv")

    figure_manifest = []
    if settings.make_figures:
        print("[8/9] Drawing cartopy/profile figures...")
        try:
            path = settings.figures_dir / "P_mean_state_S3_T3early_T3full_T3late_S4_cartopy.png"
            plot_window_map_chain(p_maps, lat, lon, settings, path, "P mean state by window", False)
            figure_manifest.append({"figure": str(path), "description": "P mean state chain"})
            path = settings.figures_dir / "V850_mean_state_S3_T3early_T3full_T3late_S4_cartopy.png"
            plot_window_map_chain(v_maps, lat, lon, settings, path, "V850 mean state by window", True)
            figure_manifest.append({"figure": str(path), "description": "V850 mean state chain"})
            for comp, (target, ref) in settings.comparisons.items():
                p_path = settings.figures_dir / f"P_change_{comp}_cartopy.png"
                v_path = settings.figures_dir / f"V850_change_{comp}_cartopy.png"
                panel_path = settings.figures_dir / f"P_V_object_transition_panel_{comp}.png"
                plot_comparison_map(p_maps[target], p_maps[ref], lat, lon, settings, p_path, f"P change: {target} - {ref}")
                plot_comparison_map(v_maps[target], v_maps[ref], lat, lon, settings, v_path, f"V850 change: {target} - {ref}")
                plot_p_v_object_panel(p_maps, v_maps, lat, lon, settings, comp, panel_path)
                figure_manifest.extend([
                    {"figure": str(p_path), "description": f"P change {comp}"},
                    {"figure": str(v_path), "description": f"V850 change {comp}"},
                    {"figure": str(panel_path), "description": f"P/V object transition panel {comp}"},
                ])
            pp = p_profile.rename(columns={"value": "value"})
            vp = v_profile.rename(columns={"value": "value"})
            pprof_path = settings.figures_dir / "P_lat_profile_chain_by_sector.png"
            vprof_path = settings.figures_dir / "V850_lat_profile_chain_by_sector.png"
            plot_profile_chain(pp, settings, pprof_path, "P latitudinal profile chain by sector", "P")
            plot_profile_chain(vp, settings, vprof_path, "V850 latitudinal profile chain by sector", "V850")
            figure_manifest.extend([
                {"figure": str(pprof_path), "description": "P profile chain by sector"},
                {"figure": str(vprof_path), "description": "V850 profile chain by sector"},
            ])
            pdel_path = settings.figures_dir / "P_lat_profile_delta_chain_by_sector.png"
            vdel_path = settings.figures_dir / "V850_lat_profile_delta_chain_by_sector.png"
            plot_profile_delta_chain(p_delta_prof, settings, "P_delta", pdel_path, "P latitudinal profile deltas by sector", "ΔP")
            plot_profile_delta_chain(v_delta_prof, settings, "V850_delta", vdel_path, "V850 latitudinal profile deltas by sector", "ΔV850")
            figure_manifest.extend([
                {"figure": str(pdel_path), "description": "P profile deltas by sector"},
                {"figure": str(vdel_path), "description": "V850 profile deltas by sector"},
            ])
        except Exception as exc:
            figure_manifest.append({"figure": "FIGURE_GENERATION_FAILED", "description": repr(exc)})
    else:
        print("[8/9] Figure generation skipped (--no-figures).")

    print("[9/9] Writing summary metadata...")
    if figure_manifest:
        _write_csv(pd.DataFrame(figure_manifest), settings.tables_dir / "figure_manifest.csv")
    summary = {
        "status": "success",
        "layer": "t3_p_v_latitudinal_object_change_audit_v1_a",
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(settings.output_dir),
        "input_smoothed_fields_path": str(field_path),
        "precip_key": p_key,
        "v850_key": v_key,
        "field_shape": list(p_field.shape),
        "day_mapping_mode": day_mode,
        "lat_min_data": lat_min_data,
        "lat_max_data": lat_max_data,
        "lon_min_data": float(np.nanmin(lon)),
        "lon_max_data": float(np.nanmax(lon)),
        "windows": settings.windows,
        "lon_sectors": {k: v.as_dict() for k, v in settings.lon_sectors.items()},
        "regions": {k: v.as_dict() for k, v in settings.regions.items()},
        "make_figures": settings.make_figures,
        "use_cartopy": settings.use_cartopy,
        "object_layer_only": True,
        "support_or_r2_computed": False,
        "full_grid_npz_persisted": False,
        "n_precip_band_rows": int(len(band_df)),
        "n_precip_transition_link_rows": int(len(link_df)),
        "n_diagnosis_rows": int(len(diag_df)),
    }
    run_meta = {
        "settings": settings_to_jsonable(settings),
        "summary": summary,
        "notes": [
            "This audit is object-layer only: P and V850 changes, latitudinal profiles, multiband P structures.",
            "It does not compute V->P support/R2/pathway evidence.",
            "Precipitation multi-band detection is a diagnostic aid; conclusions must cite window/sector/band reference explicitly.",
        ],
    }
    (settings.summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (settings.summary_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text(settings.summary_dir / "README_T3_P_V_LATITUDINAL_OBJECT_CHANGE_AUDIT_V1_A.md", _write_readme(settings))
    return summary
