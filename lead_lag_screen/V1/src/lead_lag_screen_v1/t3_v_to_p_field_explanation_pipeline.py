# -*- coding: utf-8 -*-
"""T3 V→P field-explanation hard-evidence audit v1_a.

This module adds a field-level hard-evidence layer. It does not rerun the V1
lead-lag screen and does not change any V1 labels.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import math
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except Exception:  # pragma: no cover
    plt = None
    Rectangle = None

from .t3_v_to_p_field_explanation_settings import FieldExplanationAuditSettings, RegionSpec
from .t3_v_to_p_field_explanation_io import (
    ensure_output_dirs,
    get_field_key,
    get_lat_lon_years,
    load_index_values,
    load_smoothed_fields,
    resolve_day_mapping,
    build_year_index,
    write_text,
)
from .t3_v_to_p_field_explanation_math import (
    corr_r2_beta_map,
    multicomponent_partial_r2_maps,
    r2_1d,
    beta_1d,
    spatial_pattern_similarity,
)
from .t3_v_to_p_field_explanation_regions import (
    weighted_region_mean_2d,
    weighted_region_sum_2d,
    positive_area_fraction,
    sign_consistent_area_fraction,
    weighted_region_series,
    region_mask,
    region_specs_as_dict,
)


WINDOW_ORDER = ["S3", "T3_early", "T3_full", "T3_late", "S4"]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, RegionSpec):
        return asdict(obj)
    return str(obj)


def _ensure_index_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Index table is missing required V-index columns: {missing}")


def _daily_anomaly_index(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        clim = out.groupby("day")[col].transform("mean")
        out[col + "__anom"] = out[col] - clim
    return out


def _daily_anomaly_field(field: np.ndarray) -> np.ndarray:
    """Remove day-of-season climatology across years from field(year, day, lat, lon)."""
    clim = np.nanmean(field, axis=0, keepdims=True)
    return field - clim


def _build_samples(
    index_df: pd.DataFrame,
    field_anom: np.ndarray,
    window: Tuple[int, int],
    lag: int,
    v_col: str,
    year_to_field_i: Dict[int, int],
    day_to_field_i: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build target-side samples for V(t-lag) -> P(t)."""
    start, end = window
    idx_by_year_day = index_df.set_index(["year", "day"])
    x_vals: List[float] = []
    y_vals: List[np.ndarray] = []
    years: List[int] = []
    target_days: List[int] = []
    for target_day in range(start, end + 1):
        source_day = target_day - lag
        if source_day < 1:
            continue
        if target_day not in day_to_field_i:
            continue
        td_i = day_to_field_i[target_day]
        for year, fy_i in year_to_field_i.items():
            key = (int(year), int(source_day))
            if key not in idx_by_year_day.index:
                continue
            try:
                x = float(idx_by_year_day.loc[key, v_col + "__anom"])
            except Exception:
                continue
            if not np.isfinite(x):
                continue
            y = field_anom[fy_i, td_i, :, :]
            if not np.isfinite(y).any():
                continue
            x_vals.append(x)
            y_vals.append(y)
            years.append(int(year))
            target_days.append(int(target_day))
    if not x_vals:
        raise ValueError(f"No valid samples for window={window}, lag={lag}, v={v_col}")
    return (
        np.asarray(x_vals, dtype=float),
        np.stack(y_vals, axis=0),
        np.asarray(years, dtype=int),
        np.asarray(target_days, dtype=int),
    )


def _build_multicomponent_samples(
    index_df: pd.DataFrame,
    field_anom: np.ndarray,
    window: Tuple[int, int],
    lag: int,
    v_cols: List[str],
    year_to_field_i: Dict[int, int],
    day_to_field_i: Dict[int, int],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    start, end = window
    idx_by_year_day = index_df.set_index(["year", "day"])
    xs: Dict[str, List[float]] = {c: [] for c in v_cols}
    y_vals: List[np.ndarray] = []
    years: List[int] = []
    target_days: List[int] = []
    for target_day in range(start, end + 1):
        source_day = target_day - lag
        if source_day < 1:
            continue
        if target_day not in day_to_field_i:
            continue
        td_i = day_to_field_i[target_day]
        for year, fy_i in year_to_field_i.items():
            key = (int(year), int(source_day))
            if key not in idx_by_year_day.index:
                continue
            vals = []
            ok = True
            for c in v_cols:
                try:
                    val = float(idx_by_year_day.loc[key, c + "__anom"])
                except Exception:
                    ok = False
                    break
                if not np.isfinite(val):
                    ok = False
                    break
                vals.append(val)
            if not ok:
                continue
            y = field_anom[fy_i, td_i, :, :]
            if not np.isfinite(y).any():
                continue
            for c, val in zip(v_cols, vals):
                xs[c].append(val)
            y_vals.append(y)
            years.append(int(year))
            target_days.append(int(target_day))
    if not y_vals:
        raise ValueError(f"No valid multicomponent samples for window={window}, lag={lag}")
    return (
        {c: np.asarray(vals, dtype=float) for c, vals in xs.items()},
        np.stack(y_vals, axis=0),
        np.asarray(years, dtype=int),
        np.asarray(target_days, dtype=int),
    )


def _region_summary_rows(
    *,
    map_kind: str,
    window: str,
    v_index: str,
    lag_label: str,
    lag: int | None,
    r2_map: np.ndarray,
    beta_map: np.ndarray | None,
    partial_r2_map: np.ndarray | None,
    lat: np.ndarray,
    lon: np.ndarray,
    regions: Dict[str, RegionSpec],
    n_samples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for region_name, spec in regions.items():
        row: Dict[str, Any] = {
            "map_kind": map_kind,
            "window": window,
            "v_index": v_index,
            "lag_label": lag_label,
            "lag": lag,
            "region": region_name,
            "n_samples": n_samples,
            "region_mean_R2_map": weighted_region_mean_2d(r2_map, lat, lon, spec),
            "region_integrated_R2_map": weighted_region_sum_2d(r2_map, lat, lon, spec),
            "n_valid_grid": int(np.isfinite(r2_map[np.ix_(*region_mask(lat, lon, spec))]).sum()),
        }
        if beta_map is not None:
            row.update({
                "region_mean_beta_map": weighted_region_mean_2d(beta_map, lat, lon, spec),
                "positive_beta_area_fraction": positive_area_fraction(beta_map, lat, lon, spec),
                "sign_consistent_beta_area_fraction": sign_consistent_area_fraction(beta_map, lat, lon, spec),
            })
        else:
            row.update({
                "region_mean_beta_map": np.nan,
                "positive_beta_area_fraction": np.nan,
                "sign_consistent_beta_area_fraction": np.nan,
            })
        if partial_r2_map is not None:
            row["region_mean_partial_R2_map"] = weighted_region_mean_2d(partial_r2_map, lat, lon, spec)
        else:
            row["region_mean_partial_R2_map"] = np.nan
        rows.append(row)
    return rows


def _make_maps_and_tables(
    settings: FieldExplanationAuditSettings,
    index_df: pd.DataFrame,
    field_anom: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    year_to_field_i: Dict[int, int],
    day_to_field_i: Dict[int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps: Dict[str, np.ndarray] = {}
    map_meta: Dict[str, Any] = {"maps": []}
    region_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    positive_best: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    for win_name in WINDOW_ORDER:
        if win_name not in settings.windows:
            continue
        window = settings.windows[win_name]
        for v_index in settings.v_indices:
            # First compute lag0..max_lag simple maps.
            lag_r2_maps: Dict[int, np.ndarray] = {}
            lag_beta_maps: Dict[int, np.ndarray] = {}
            for lag in range(0, settings.max_lag + 1):
                x, y, sample_years, target_days = _build_samples(
                    index_df, field_anom, window, lag, v_index, year_to_field_i, day_to_field_i
                )
                r_map, r2_map, beta_map, n_samples = corr_r2_beta_map(x, y)
                lag_r2_maps[lag] = r2_map
                lag_beta_maps[lag] = beta_map
                key_base = f"{win_name}__{v_index}__lag{lag}"
                maps[key_base + "__r2"] = r2_map
                maps[key_base + "__beta"] = beta_map
                manifest_rows.append({
                    "window": win_name,
                    "v_index": v_index,
                    "lag": lag,
                    "map_r2_key": key_base + "__r2",
                    "map_beta_key": key_base + "__beta",
                    "n_samples": n_samples,
                    "n_years": int(len(set(sample_years.tolist()))),
                    "target_day_min": int(np.min(target_days)),
                    "target_day_max": int(np.max(target_days)),
                    "source_day_min": int(np.min(target_days - lag)),
                    "source_day_max": int(np.max(target_days - lag)),
                })
                map_meta["maps"].append({
                    "key": key_base + "__r2",
                    "kind": "simple_R2",
                    "window": win_name,
                    "v_index": v_index,
                    "lag": lag,
                    "n_samples": n_samples,
                })
                region_rows.extend(_region_summary_rows(
                    map_kind="simple_R2",
                    window=win_name,
                    v_index=v_index,
                    lag_label=f"lag{lag}",
                    lag=lag,
                    r2_map=r2_map,
                    beta_map=beta_map,
                    partial_r2_map=None,
                    lat=lat,
                    lon=lon,
                    regions=settings.regions,
                    n_samples=n_samples,
                ))

            # Positive-lag max maps and best-lag maps.
            positive_stack = np.stack([lag_r2_maps[l] for l in range(1, settings.max_lag + 1)], axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pos_max = np.nanmax(positive_stack, axis=0)
            all_nan = np.all(~np.isfinite(positive_stack), axis=0)
            best_idx = np.nanargmax(np.where(np.isfinite(positive_stack), positive_stack, -np.inf), axis=0) + 1
            best_lag = np.where(all_nan, np.nan, best_idx).astype(float)
            best_beta = np.full_like(pos_max, np.nan, dtype=float)
            for l in range(1, settings.max_lag + 1):
                best_beta = np.where(best_lag == l, lag_beta_maps[l], best_beta)
            base = f"{win_name}__{v_index}__positive_lag_max"
            maps[base + "__r2"] = pos_max
            maps[base + "__best_lag"] = best_lag
            maps[base + "__beta_at_best_lag"] = best_beta
            positive_best[(win_name, v_index)] = {"r2": pos_max, "best_lag": best_lag, "beta": best_beta}
            region_rows.extend(_region_summary_rows(
                map_kind="positive_lag_max_R2",
                window=win_name,
                v_index=v_index,
                lag_label="positive_lag_max",
                lag=None,
                r2_map=pos_max,
                beta_map=best_beta,
                partial_r2_map=None,
                lat=lat,
                lon=lon,
                regions=settings.regions,
                n_samples=int(max(r["n_samples"] for r in manifest_rows if r["window"] == win_name and r["v_index"] == v_index and 1 <= r["lag"] <= settings.max_lag)),
            ))

            # Lag minus tau0 delta.
            delta = pos_max - lag_r2_maps[0]
            maps[base + "__delta_r2_minus_tau0"] = delta
            for region_name, spec in settings.regions.items():
                region_rows.append({
                    "map_kind": "lag_minus_tau0_delta_R2",
                    "window": win_name,
                    "v_index": v_index,
                    "lag_label": "positive_lag_max_minus_tau0",
                    "lag": None,
                    "region": region_name,
                    "n_samples": np.nan,
                    "region_mean_R2_map": weighted_region_mean_2d(delta, lat, lon, spec),
                    "region_integrated_R2_map": weighted_region_sum_2d(delta, lat, lon, spec),
                    "region_mean_partial_R2_map": np.nan,
                    "region_mean_beta_map": np.nan,
                    "positive_beta_area_fraction": np.nan,
                    "sign_consistent_beta_area_fraction": np.nan,
                    "n_valid_grid": int(np.isfinite(delta[np.ix_(*region_mask(lat, lon, spec))]).sum()),
                })

        # Partial R² maps for component competition by lag.
        for lag in range(0, settings.max_lag + 1):
            X_by_comp, y, sample_years, target_days = _build_multicomponent_samples(
                index_df, field_anom, window, lag, settings.v_indices, year_to_field_i, day_to_field_i
            )
            partial_maps = multicomponent_partial_r2_maps(X_by_comp, y)
            for v_index, part_map in partial_maps.items():
                key = f"{win_name}__{v_index}__lag{lag}__partial_r2"
                maps[key] = part_map
                # Also add a compact region summary row for partial R².
                region_rows.extend(_region_summary_rows(
                    map_kind="partial_R2",
                    window=win_name,
                    v_index=v_index,
                    lag_label=f"lag{lag}",
                    lag=lag,
                    r2_map=part_map,
                    beta_map=None,
                    partial_r2_map=part_map,
                    lat=lat,
                    lon=lon,
                    regions=settings.regions,
                    n_samples=int(len(sample_years)),
                ))

    # Component contrast maps: V_NS_diff - V_strength at positive-lag max.
    contrast_rows: List[Dict[str, Any]] = []
    for win_name in WINDOW_ORDER:
        if (win_name, "V_NS_diff") in positive_best and (win_name, "V_strength") in positive_best:
            contrast = positive_best[(win_name, "V_NS_diff")]["r2"] - positive_best[(win_name, "V_strength")]["r2"]
            key = f"{win_name}__V_NS_diff_minus_V_strength__positive_lag_max__r2_contrast"
            maps[key] = contrast
            for region_name, spec in settings.regions.items():
                contrast_rows.append({
                    "window": win_name,
                    "contrast": "V_NS_diff_minus_V_strength",
                    "lag_label": "positive_lag_max",
                    "region": region_name,
                    "region_mean_R2_contrast": weighted_region_mean_2d(contrast, lat, lon, spec),
                    "region_integrated_R2_contrast": weighted_region_sum_2d(contrast, lat, lon, spec),
                    "positive_contrast_area_fraction": positive_area_fraction(contrast, lat, lon, spec),
                    "n_valid_grid": int(np.isfinite(contrast[np.ix_(*region_mask(lat, lon, spec))]).sum()),
                })
                region_rows.append({
                    "map_kind": "component_contrast_R2",
                    "window": win_name,
                    "v_index": "V_NS_diff_minus_V_strength",
                    "lag_label": "positive_lag_max",
                    "lag": None,
                    "region": region_name,
                    "n_samples": np.nan,
                    "region_mean_R2_map": weighted_region_mean_2d(contrast, lat, lon, spec),
                    "region_integrated_R2_map": weighted_region_sum_2d(contrast, lat, lon, spec),
                    "region_mean_partial_R2_map": np.nan,
                    "region_mean_beta_map": np.nan,
                    "positive_beta_area_fraction": positive_area_fraction(contrast, lat, lon, spec),
                    "sign_consistent_beta_area_fraction": sign_consistent_area_fraction(contrast, lat, lon, spec),
                    "n_valid_grid": int(np.isfinite(contrast[np.ix_(*region_mask(lat, lon, spec))]).sum()),
                })

    return maps, map_meta, pd.DataFrame(manifest_rows), pd.DataFrame(region_rows), pd.DataFrame(contrast_rows)


def _bootstrap_region_response(
    settings: FieldExplanationAuditSettings,
    index_df: pd.DataFrame,
    field_anom: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    year_to_field_i: Dict[int, int],
    day_to_field_i: Dict[int, int],
) -> pd.DataFrame:
    """Year-block bootstrap on regional mean P response series.

    This does not bootstrap full grid maps. It estimates uncertainty for the
    pre-registered regional response series, which is the decision layer.
    """
    rng = np.random.default_rng(settings.bootstrap_seed)
    rows: List[Dict[str, Any]] = []
    for win_name in WINDOW_ORDER:
        if win_name not in settings.windows:
            continue
        window = settings.windows[win_name]
        for v_index in settings.v_indices:
            for lag in range(0, settings.max_lag + 1):
                x, y, sample_years, target_days = _build_samples(
                    index_df, field_anom, window, lag, v_index, year_to_field_i, day_to_field_i
                )
                unique_years = np.asarray(sorted(set(sample_years.tolist())), dtype=int)
                if unique_years.size < 5:
                    continue
                for region_name, spec in settings.regions.items():
                    y_reg = weighted_region_series(y, lat, lon, spec)
                    obs_r2 = r2_1d(x, y_reg)
                    obs_beta = beta_1d(x, y_reg)
                    boot_r2: List[float] = []
                    boot_beta: List[float] = []
                    for _ in range(settings.n_bootstrap):
                        draw_years = rng.choice(unique_years, size=unique_years.size, replace=True)
                        # Preserve all target-day samples within each selected year.
                        idx_parts = [np.where(sample_years == yy)[0] for yy in draw_years]
                        if not idx_parts:
                            continue
                        draw_idx = np.concatenate(idx_parts)
                        if draw_idx.size < 5:
                            continue
                        boot_r2.append(r2_1d(x[draw_idx], y_reg[draw_idx]))
                        boot_beta.append(beta_1d(x[draw_idx], y_reg[draw_idx]))
                    arr_r2 = np.asarray([v for v in boot_r2 if np.isfinite(v)], dtype=float)
                    arr_beta = np.asarray([v for v in boot_beta if np.isfinite(v)], dtype=float)
                    rows.append({
                        "window": win_name,
                        "v_index": v_index,
                        "lag": lag,
                        "region": region_name,
                        "n_years": int(unique_years.size),
                        "n_samples": int(len(x)),
                        "observed_region_series_R2": obs_r2,
                        "observed_region_series_beta": obs_beta,
                        "boot_R2_mean": float(np.nanmean(arr_r2)) if arr_r2.size else np.nan,
                        "boot_R2_ci90_low": float(np.nanquantile(arr_r2, 0.05)) if arr_r2.size else np.nan,
                        "boot_R2_ci90_high": float(np.nanquantile(arr_r2, 0.95)) if arr_r2.size else np.nan,
                        "boot_R2_ci95_low": float(np.nanquantile(arr_r2, 0.025)) if arr_r2.size else np.nan,
                        "boot_R2_ci95_high": float(np.nanquantile(arr_r2, 0.975)) if arr_r2.size else np.nan,
                        "boot_beta_mean": float(np.nanmean(arr_beta)) if arr_beta.size else np.nan,
                        "boot_beta_ci90_low": float(np.nanquantile(arr_beta, 0.05)) if arr_beta.size else np.nan,
                        "boot_beta_ci90_high": float(np.nanquantile(arr_beta, 0.95)) if arr_beta.size else np.nan,
                        "boot_beta_ci95_low": float(np.nanquantile(arr_beta, 0.025)) if arr_beta.size else np.nan,
                        "boot_beta_ci95_high": float(np.nanquantile(arr_beta, 0.975)) if arr_beta.size else np.nan,
                        "prob_R2_gt_0_05": float(np.mean(arr_r2 > 0.05)) if arr_r2.size else np.nan,
                        "prob_R2_gt_0_10": float(np.mean(arr_r2 > 0.10)) if arr_r2.size else np.nan,
                        "prob_beta_gt_0": float(np.mean(arr_beta > 0.0)) if arr_beta.size else np.nan,
                        "prob_beta_lt_0": float(np.mean(arr_beta < 0.0)) if arr_beta.size else np.nan,
                    })
    return pd.DataFrame(rows)


def _similarity_tables(
    settings: FieldExplanationAuditSettings,
    maps: Dict[str, np.ndarray],
    region_summary: pd.DataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = [
        ("T3_early", "S3"),
        ("T3_late", "S4"),
        ("T3_full", "T3_early"),
        ("T3_full", "T3_late"),
        ("T3_early", "S4"),
        ("T3_late", "S3"),
    ]
    map_rows: List[Dict[str, Any]] = []
    region_vec_rows: List[Dict[str, Any]] = []
    dilution_rows: List[Dict[str, Any]] = []

    # Whole-domain mask for map similarity.
    domain = settings.regions["main_easm_domain"]
    lat_mask, lon_mask = region_mask(lat, lon, domain)
    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    mask[np.ix_(lat_mask, lon_mask)] = True

    for v_index in settings.v_indices:
        for w1, w2 in pairs:
            k1 = f"{w1}__{v_index}__positive_lag_max__r2"
            k2 = f"{w2}__{v_index}__positive_lag_max__r2"
            if k1 in maps and k2 in maps:
                map_rows.append({
                    "v_index": v_index,
                    "map_kind": "positive_lag_max_R2",
                    "window_a": w1,
                    "window_b": w2,
                    "spatial_pattern_similarity_main_easm_domain": spatial_pattern_similarity(maps[k1], maps[k2], mask=mask),
                })
            # Region vector similarity based on region_mean_R2_map.
            sub = region_summary[
                (region_summary["map_kind"] == "positive_lag_max_R2")
                & (region_summary["v_index"] == v_index)
                & (region_summary["window"].isin([w1, w2]))
            ]
            if not sub.empty:
                piv = sub.pivot_table(index="region", columns="window", values="region_mean_R2_map", aggfunc="mean")
                if w1 in piv.columns and w2 in piv.columns:
                    a = piv[w1].to_numpy(dtype=float)
                    b = piv[w2].to_numpy(dtype=float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    sim = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else np.nan
                    region_vec_rows.append({
                        "v_index": v_index,
                        "map_kind": "positive_lag_max_R2",
                        "window_a": w1,
                        "window_b": w2,
                        "region_vector_similarity": sim,
                    })
        # Full vs subwindow dilution by region.
        sub = region_summary[
            (region_summary["map_kind"] == "positive_lag_max_R2")
            & (region_summary["v_index"] == v_index)
            & (region_summary["window"].isin(["T3_early", "T3_full", "T3_late"]))
        ]
        piv = sub.pivot_table(index="region", columns="window", values="region_mean_R2_map", aggfunc="mean")
        for region in piv.index:
            early = piv.loc[region].get("T3_early", np.nan)
            full = piv.loc[region].get("T3_full", np.nan)
            late = piv.loc[region].get("T3_late", np.nan)
            max_sub = np.nanmax([early, late]) if np.isfinite([early, late]).any() else np.nan
            dilution_rows.append({
                "v_index": v_index,
                "region": region,
                "T3_early_region_mean_R2": early,
                "T3_full_region_mean_R2": full,
                "T3_late_region_mean_R2": late,
                "max_subwindow_region_mean_R2": max_sub,
                "full_minus_max_subwindow_R2": full - max_sub if np.isfinite(full) and np.isfinite(max_sub) else np.nan,
                "dilution_flag": bool(np.isfinite(full) and np.isfinite(max_sub) and full < max_sub),
            })
    return pd.DataFrame(map_rows), pd.DataFrame(region_vec_rows), pd.DataFrame(dilution_rows)


def _diagnosis_table(
    region_summary: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    component_contrast: pd.DataFrame,
    dilution: pd.DataFrame,
) -> pd.DataFrame:
    """Conservative, table-driven diagnosis. Thresholds are explicit diagnostics, not proof."""
    rows: List[Dict[str, Any]] = []

    def _safe_mean(df: pd.DataFrame, col: str) -> float:
        return float(np.nanmean(df[col].to_numpy(dtype=float))) if not df.empty and col in df.columns else np.nan

    # Field-level weakening: T3_full weaker than S3/S4 in main domain across V components.
    rs = region_summary[
        (region_summary["map_kind"] == "positive_lag_max_R2")
        & (region_summary["region"] == "main_easm_domain")
        & (region_summary["v_index"].isin(["V_strength", "V_NS_diff", "V_pos_centroid_lat"]))
    ]
    t3 = _safe_mean(rs[rs["window"] == "T3_full"], "region_mean_R2_map")
    neighbors = _safe_mean(rs[rs["window"].isin(["S3", "S4"])], "region_mean_R2_map")
    if np.isfinite(t3) and np.isfinite(neighbors):
        if t3 < 0.75 * neighbors:
            support = "supported"
        elif t3 > 1.10 * neighbors:
            support = "counter_evidence"
        else:
            support = "mixed_or_weak"
    else:
        support = "insufficient"
    rows.append({
        "diagnosis": "field_level_weakening",
        "support_level": support,
        "primary_evidence": f"main_easm_domain mean positive-lag R2: T3_full={t3:.4g}, S3/S4 mean={neighbors:.4g}",
        "counter_evidence": "If T3 field maps are strong or stronger than S3/S4, V1 index-level contraction may reflect projection/window limitations rather than field-level weakening.",
        "allowed_interpretation": "T3 V-to-P field-level lagged explanatory power is weaker if supported.",
        "forbidden_interpretation": "Do not infer complete disappearance of V physical influence.",
    })

    # tau0 replacement: lag-minus-tau0 delta negative in T3_full.
    delta = region_summary[
        (region_summary["map_kind"] == "lag_minus_tau0_delta_R2")
        & (region_summary["window"] == "T3_full")
        & (region_summary["region"] == "main_easm_domain")
    ]
    dmean = _safe_mean(delta, "region_mean_R2_map")
    if np.isfinite(dmean):
        support = "supported" if dmean < 0 else "not_supported" if dmean > 0.01 else "mixed_or_weak"
    else:
        support = "insufficient"
    rows.append({
        "diagnosis": "tau0_replacement",
        "support_level": support,
        "primary_evidence": f"T3_full main_easm_domain mean delta R2 (positive lag max - tau0)={dmean:.4g}",
        "counter_evidence": "Positive delta indicates lagged map remains stronger than tau0.",
        "allowed_interpretation": "T3 is more tau0-coupled than lag-dominant if supported.",
        "forbidden_interpretation": "Do not write V-P unrelated when tau0 evidence is strong.",
    })

    # Component replacement: V_NS_diff minus V_strength positive in south_china_scs or main domain.
    cc = component_contrast[
        (component_contrast["window"] == "T3_full")
        & (component_contrast["region"].isin(["south_china_scs", "main_easm_domain"]))
    ]
    cmean = _safe_mean(cc, "region_mean_R2_contrast")
    support = "supported" if np.isfinite(cmean) and cmean > 0 else "not_supported" if np.isfinite(cmean) else "insufficient"
    rows.append({
        "diagnosis": "v_component_replacement",
        "support_level": support,
        "primary_evidence": f"T3_full mean R2 contrast V_NS_diff - V_strength over south_china_scs/main_domain={cmean:.4g}",
        "counter_evidence": "Negative contrast means V_strength remains stronger than V_NS_diff.",
        "allowed_interpretation": "V effective component shifts toward NS-difference/structure if supported.",
        "forbidden_interpretation": "Do not merge all V components into a single V-strength story.",
    })

    # P target shift: south/SCS relative to main_meiyu in T3_full.
    tgt = region_summary[
        (region_summary["map_kind"] == "positive_lag_max_R2")
        & (region_summary["window"] == "T3_full")
        & (region_summary["v_index"].isin(["V_strength", "V_NS_diff", "V_pos_centroid_lat"]))
    ]
    piv = tgt.pivot_table(index="v_index", columns="region", values="region_mean_R2_map", aggfunc="mean")
    if "south_china_scs" in piv.columns and "main_meiyu" in piv.columns:
        ratio = float(np.nanmean(piv["south_china_scs"] - piv["main_meiyu"]))
        support = "supported" if ratio > 0 else "not_supported" if ratio < -0.005 else "mixed_or_weak"
    else:
        ratio = np.nan
        support = "insufficient"
    rows.append({
        "diagnosis": "p_target_shift_to_south_scs",
        "support_level": support,
        "primary_evidence": f"T3_full mean regional contrast south_china_scs - main_meiyu={ratio:.4g}",
        "counter_evidence": "Main/Meiyu remains stronger or all regions are weak.",
        "allowed_interpretation": "P response target is relatively south/SCS-shifted if supported.",
        "forbidden_interpretation": "Do not infer South China/SCS is controlled by V without further physical pathway evidence.",
    })

    # Transition mixing: full weaker than early/late in many region/component cases.
    if not dilution.empty:
        frac = float(np.mean(dilution["dilution_flag"].astype(bool)))
        support = "supported" if frac >= 0.60 else "mixed_or_weak" if frac >= 0.35 else "not_supported"
    else:
        frac = np.nan
        support = "insufficient"
    rows.append({
        "diagnosis": "transition_window_mixing",
        "support_level": support,
        "primary_evidence": f"fraction of T3_full region/component cases weaker than max(T3_early,T3_late)={frac:.3g}",
        "counter_evidence": "T3_full is not weaker than subwindows in most cases.",
        "allowed_interpretation": "Full T3 may dilute subwindow structures if supported.",
        "forbidden_interpretation": "Do not call a completed phase transition without S3/S4 similarity evidence.",
    })

    # V1 limitation: inferred only when field evidence is not weak but old V1 context says contraction.
    rows.append({
        "diagnosis": "v1_design_limitation_candidate",
        "support_level": "requires_context_comparison",
        "primary_evidence": "Compare this audit's field/region evidence against old V1 fixed-index pair contraction. This row is intentionally not decided from old V1 results inside the hard-evidence calculation.",
        "counter_evidence": "If field-level evidence is weak, V1 likely captured real weakening rather than leaked signal.",
        "allowed_interpretation": "V1 local T3 limitation may be claimed only if field/region evidence is strong while V1 index-pair layer is weak.",
        "forbidden_interpretation": "Do not declare V1 globally wrong.",
    })
    return pd.DataFrame(rows)


def _cartopy_modules(use_cartopy: bool) -> Tuple[Any | None, Any | None, str]:
    """Return cartopy modules when requested and available.

    Cartopy is the default map backend for this audit. If it is unavailable,
    figures are still written with ordinary lon/lat axes and the fallback is
    recorded in the figure manifest. This prevents a plotting dependency from
    blocking non-figure table production.
    """
    if not use_cartopy:
        return None, None, "plain_lonlat_requested"
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        return ccrs, cfeature, "cartopy"
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"Cartopy requested but unavailable; falling back to plain lon/lat plots: {exc}")
        return None, None, "plain_lonlat_fallback_cartopy_unavailable"


def _lonlat_extent(lon: np.ndarray, lat: np.ndarray) -> List[float]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lon_vals = lon_arr[np.isfinite(lon_arr)]
    lat_vals = lat_arr[np.isfinite(lat_arr)]
    return [float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals)), float(np.nanmin(lat_vals)), float(np.nanmax(lat_vals))]


def _add_region_boxes(ax: Any, regions: Dict[str, RegionSpec], ccrs: Any | None) -> None:
    if Rectangle is None:
        return
    for name, spec in regions.items():
        # Draw only the core interpretive regions to avoid clutter.
        if name not in {"main_meiyu", "south_china", "scs", "north_northeast"}:
            continue
        kwargs: Dict[str, Any] = {"fill": False, "linewidth": 0.8, "alpha": 0.9}
        if ccrs is not None:
            kwargs["transform"] = ccrs.PlateCarree()
        rect = Rectangle(
            (spec.lon_min, spec.lat_min),
            spec.lon_max - spec.lon_min,
            spec.lat_max - spec.lat_min,
            **kwargs,
        )
        try:
            ax.add_patch(rect)
        except Exception:
            pass


def _plot_map_panel(
    ax: Any,
    lon: np.ndarray,
    lat: np.ndarray,
    arr: np.ndarray,
    *,
    title: str,
    ccrs: Any | None,
    cfeature: Any | None,
    regions: Dict[str, RegionSpec],
) -> Any:
    if ccrs is not None:
        transform = ccrs.PlateCarree()
        im = ax.pcolormesh(lon, lat, arr, shading="auto", transform=transform)
        try:
            ax.set_extent(_lonlat_extent(lon, lat), crs=transform)
        except Exception:
            pass
        try:
            ax.coastlines(resolution="110m", linewidth=0.6)
        except Exception:
            pass
        if cfeature is not None:
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            except Exception:
                pass
        try:
            gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
        except Exception:
            pass
    else:
        im = ax.pcolormesh(lon, lat, arr, shading="auto")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    _add_region_boxes(ax, regions, ccrs)
    ax.set_title(title)
    return im


def _new_map_axis(fig: Any, nrows: int, ncols: int, idx: int, ccrs: Any | None) -> Any:
    if ccrs is not None:
        return fig.add_subplot(nrows, ncols, idx, projection=ccrs.PlateCarree())
    return fig.add_subplot(nrows, ncols, idx)


def _write_figures(settings: FieldExplanationAuditSettings, maps: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not settings.make_figures or plt is None:
        return pd.DataFrame(rows)
    settings.output_figures_dir.mkdir(parents=True, exist_ok=True)
    ccrs, cfeature, plot_backend = _cartopy_modules(settings.use_cartopy)

    for v_index in ["V_strength", "V_NS_diff"]:
        fig = plt.figure(figsize=(4.2 * len(WINDOW_ORDER), 3.9), constrained_layout=True)
        axes = []
        last_im = None
        for i, win in enumerate(WINDOW_ORDER, start=1):
            ax = _new_map_axis(fig, 1, len(WINDOW_ORDER), i, ccrs)
            axes.append(ax)
            key = f"{win}__{v_index}__positive_lag_max__r2"
            arr = maps.get(key)
            if arr is None:
                ax.set_axis_off()
                continue
            last_im = _plot_map_panel(
                ax, lon, lat, arr,
                title=f"{win}\n{v_index}",
                ccrs=ccrs,
                cfeature=cfeature,
                regions=settings.regions,
            )
        if last_im is not None:
            fig.colorbar(last_im, ax=axes, shrink=0.72, label="R²")
        fname = f"{v_index}_positive_lag_max_R2_S3_T3early_T3full_T3late_S4_cartopy.png" if plot_backend == "cartopy" else f"{v_index}_positive_lag_max_R2_S3_T3early_T3full_T3late_S4.png"
        path = settings.output_figures_dir / fname
        fig.savefig(path, dpi=180)
        plt.close(fig)
        rows.append({"figure": fname, "description": f"Positive-lag max R2 maps for {v_index}", "plot_backend": plot_backend})

    for v_index in ["V_strength", "V_NS_diff"]:
        key = f"T3_full__{v_index}__positive_lag_max__delta_r2_minus_tau0"
        if key not in maps:
            continue
        fig = plt.figure(figsize=(5.5, 4.6), constrained_layout=True)
        ax = _new_map_axis(fig, 1, 1, 1, ccrs)
        im = _plot_map_panel(
            ax, lon, lat, maps[key],
            title=f"T3_full {v_index}\npositive lag max R² - tau0 R²",
            ccrs=ccrs,
            cfeature=cfeature,
            regions=settings.regions,
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="ΔR²")
        fname = f"T3_full_{v_index}_lag_minus_tau0_delta_R2_cartopy.png" if plot_backend == "cartopy" else f"T3_full_{v_index}_lag_minus_tau0_delta_R2.png"
        path = settings.output_figures_dir / fname
        fig.savefig(path, dpi=180)
        plt.close(fig)
        rows.append({"figure": fname, "description": f"Lag-minus-tau0 delta R2 for {v_index}", "plot_backend": plot_backend})

    key = "T3_full__V_NS_diff_minus_V_strength__positive_lag_max__r2_contrast"
    if key in maps:
        fig = plt.figure(figsize=(5.5, 4.6), constrained_layout=True)
        ax = _new_map_axis(fig, 1, 1, 1, ccrs)
        im = _plot_map_panel(
            ax, lon, lat, maps[key],
            title="T3_full component contrast\nV_NS_diff - V_strength",
            ccrs=ccrs,
            cfeature=cfeature,
            regions=settings.regions,
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="ΔR²")
        fname = "T3_full_component_contrast_VNSdiff_minus_Vstrength_R2_cartopy.png" if plot_backend == "cartopy" else "T3_full_component_contrast_VNSdiff_minus_Vstrength_R2.png"
        path = settings.output_figures_dir / fname
        fig.savefig(path, dpi=180)
        plt.close(fig)
        rows.append({"figure": fname, "description": "Component contrast R2 map for T3_full", "plot_backend": plot_backend})

    return pd.DataFrame(rows)

def _write_readme(settings: FieldExplanationAuditSettings, summary: Dict[str, Any]) -> None:
    text = f"""# T3 V→P Field-Explanation Hard-Evidence Audit v1_a

This output is an independent hard-evidence audit. It does **not** rerun the V1
lead-lag screen and does **not** modify prior T3 physical-hypothesis outputs.

## Purpose

Distinguish whether T3 V→P fixed-index candidate contraction is better explained
as field-level weakening, tau0 replacement, V-component replacement, P-target
regional shift, transition-window mixing, or a local V1 design limitation.

## Window basis

`use_legacy_t3_window = {settings.use_legacy_t3_window}`

Windows used:

```text
{json.dumps(settings.windows, indent=2)}
```

Default is the V6_1 / V1 main-screen basis: S3 87-106, T3 107-117, S4 118-154.
Legacy physical-audit windows are only used when explicitly requested.

## Main evidence layers

1. V-index → P-field positive-lag explained-variance maps.
2. Pre-registered regional response summaries.
3. Positive-lag versus tau0 R² differences.
4. V-component contrast, especially `V_NS_diff - V_strength`.
5. T3 early/full/late and S3/S4 similarity and dilution diagnostics.

## Important interpretation boundaries

- This is not a causal pathway establishment layer.
- Field/region evidence may show stable explanatory structure, but that is not by itself a physical mechanism.
- `v1_design_limitation_candidate` is intentionally left as context-dependent: it requires comparison with old V1 fixed-index results.
- Do not interpret weak T3 maps as complete disappearance of V influence.
- Do not interpret south/SCS response as proof that V controls South China/SCS rainfall.

## Summary

```json
{json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)}
```
"""
    write_text(settings.output_summary_dir / "README_T3_V_TO_P_FIELD_EXPLANATION_AUDIT_V1_A.md", text)


def run_t3_v_to_p_field_explanation_audit(settings: FieldExplanationAuditSettings) -> Dict[str, Any]:
    dirs = ensure_output_dirs(settings)
    index_df, index_path = load_index_values(settings)
    _ensure_index_columns(index_df, settings.v_indices)
    fields, field_path = load_smoothed_fields(settings)
    precip_key = get_field_key(fields, settings.precip_aliases)
    precip = np.asarray(fields[precip_key], dtype=float)
    if precip.ndim != 4:
        raise ValueError(f"Precip field must be year x day x lat x lon; got shape {precip.shape}")
    lat, lon, field_years = get_lat_lon_years(fields)
    day_mapping_mode, day_to_field_i = resolve_day_mapping(precip, fields)
    year_mapping_mode, year_to_field_i_all = build_year_index(field_years, index_df["year"].unique(), precip.shape[0])

    # Restrict to years present in both field and index table.
    index_years = set(index_df["year"].astype(int).unique().tolist())
    year_to_field_i = {y: i for y, i in year_to_field_i_all.items() if y in index_years}
    if not year_to_field_i:
        raise ValueError("No overlapping years between field data and index table.")

    index_df = _daily_anomaly_index(index_df, settings.v_indices)
    precip_anom = _daily_anomaly_field(precip)

    maps, map_meta, manifest, region_summary, component_contrast = _make_maps_and_tables(
        settings, index_df, precip_anom, lat, lon, year_to_field_i, day_to_field_i
    )
    bootstrap_df = _bootstrap_region_response(
        settings, index_df, precip_anom, lat, lon, year_to_field_i, day_to_field_i
    )
    map_sim, region_sim, dilution = _similarity_tables(settings, maps, region_summary, lat, lon)
    diagnosis = _diagnosis_table(region_summary, bootstrap_df, component_contrast, dilution)
    figure_manifest = _write_figures(settings, maps, lat, lon)

    # Write tables.
    manifest.to_csv(settings.output_tables_dir / "field_explained_variance_manifest.csv", index=False, encoding="utf-8-sig")
    region_summary.to_csv(settings.output_tables_dir / "region_response_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap_df.to_csv(settings.output_tables_dir / "region_response_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    # Convenience split files.
    region_summary[region_summary["map_kind"] == "lag_minus_tau0_delta_R2"].to_csv(
        settings.output_tables_dir / "lag_tau0_region_delta_summary.csv", index=False, encoding="utf-8-sig"
    )
    component_contrast.to_csv(settings.output_tables_dir / "component_contrast_region_summary.csv", index=False, encoding="utf-8-sig")
    map_sim.to_csv(settings.output_tables_dir / "window_map_similarity.csv", index=False, encoding="utf-8-sig")
    region_sim.to_csv(settings.output_tables_dir / "window_region_vector_similarity.csv", index=False, encoding="utf-8-sig")
    dilution.to_csv(settings.output_tables_dir / "full_vs_subwindow_dilution_summary.csv", index=False, encoding="utf-8-sig")
    diagnosis.to_csv(settings.output_tables_dir / "hard_evidence_diagnosis_table.csv", index=False, encoding="utf-8-sig")
    if not figure_manifest.empty:
        figure_manifest.to_csv(settings.output_tables_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig")

    # Full-grid map arrays are intentionally not persisted. They are large
    # intermediate objects used to derive the decision-layer tables and optional
    # cartopy figures. The hard-evidence outputs are tables + figures + summary.
    map_meta.update({
        "lat_key": "lat",
        "lon_key": "lon",
        "lat_shape": list(np.asarray(lat).shape),
        "lon_shape": list(np.asarray(lon).shape),
        "regions": region_specs_as_dict(settings.regions),
        "full_grid_map_arrays_persisted": False,
        "note": "Full-grid map arrays are intentionally not written to disk. Hard evidence is carried by region/similarity/diagnosis tables; cartopy figures are visual aids derived from in-memory maps.",
    })

    summary: Dict[str, Any] = {
        "status": "success",
        "layer_name": "t3_v_to_p_field_explanation_audit_v1_a",
        "input_index_path": str(index_path),
        "input_field_path": str(field_path),
        "precip_key": precip_key,
        "window_definition_used": settings.windows,
        "use_legacy_t3_window": settings.use_legacy_t3_window,
        "day_mapping_mode": day_mapping_mode,
        "year_mapping_mode": year_mapping_mode,
        "n_years_overlap": int(len(year_to_field_i)),
        "n_bootstrap": int(settings.n_bootstrap),
        "max_lag": int(settings.max_lag),
        "v_indices": settings.v_indices,
        "regions": region_specs_as_dict(settings.regions),
        "n_observed_map_arrays_computed_in_memory": int(len(maps)),
        "full_grid_map_arrays_persisted": False,
        "figure_plot_backend": figure_manifest["plot_backend"].iloc[0] if (not figure_manifest.empty and "plot_backend" in figure_manifest.columns) else ("none" if not settings.make_figures else "unavailable"),
        "n_region_summary_rows": int(len(region_summary)),
        "n_bootstrap_rows": int(len(bootstrap_df)),
        "diagnosis_counts": diagnosis["support_level"].value_counts(dropna=False).to_dict() if not diagnosis.empty else {},
        "warnings": [
            "This audit uses smooth5 day-of-season anomalies for variability explanation; it is not an index-validity smoothed-field representativeness test.",
            "Bootstrap uncertainty is computed for regional response series and decision-layer summaries, not full gridpoint maps.",
            "Full-grid map arrays are not persisted as NPZ outputs in this revision.",
            "Cartopy is the default figure backend when available; use --no-cartopy to force plain lon/lat figures.",
            "Old V1 pair results are not used inside hard-evidence map/region calculations.",
        ],
    }
    with open(settings.output_summary_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)
    run_meta = {
        "settings": asdict(settings),
        "input_index_path": str(index_path),
        "input_field_path": str(field_path),
        "precip_key": precip_key,
        "day_mapping_mode": day_mapping_mode,
        "year_mapping_mode": year_mapping_mode,
    }
    with open(settings.output_summary_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=_json_default)
    _write_readme(settings, summary)
    return summary
