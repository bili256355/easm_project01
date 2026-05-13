from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .settings import LeadLagScreenSettings


def _field_from_npz(npz, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name])
    raise KeyError(f"None of fields {list(names)} found in npz. Available: {npz.files}")


def _mask_between(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.isfinite(v) & (v >= min(lo, hi)) & (v <= max(lo, hi))


def _cos_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    return w


def _area_mean(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: Tuple[float, float], lon_range: Tuple[float, float]) -> np.ndarray:
    lat_mask = _mask_between(lat, *lat_range)
    lon_mask = _mask_between(lon, *lon_range)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"Empty area mask for lat_range={lat_range}, lon_range={lon_range}")
    weights = _cos_weights(lat[lat_idx])
    if np.nansum(weights) <= 0:
        weights = np.ones(len(lat_idx), dtype=float)
    rows = []
    for li in lat_idx:
        arr = np.asarray(field[:, :, li, :][:, :, lon_idx], dtype=float)
        rows.append(np.nanmean(arr, axis=2))
    stack = np.stack(rows, axis=2)  # year, day, lat
    with np.errstate(invalid="ignore"):
        return np.nansum(stack * weights[None, None, :], axis=2) / np.nansum(weights)


def _lat_profile(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: Tuple[float, float], lon_range: Tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lat_mask = _mask_between(lat, *lat_range)
    lon_mask = _mask_between(lon, *lon_range)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"Empty profile mask for lat_range={lat_range}, lon_range={lon_range}")
    prof = np.empty((field.shape[0], field.shape[1], len(lat_idx)), dtype=float)
    for k, li in enumerate(lat_idx):
        arr = np.asarray(field[:, :, li, :][:, :, lon_idx], dtype=float)
        prof[:, :, k] = np.nanmean(arr, axis=2)
    return lat[lat_idx].astype(float), prof


def _weighted_centroid(lat_sub: np.ndarray, values: np.ndarray) -> np.ndarray:
    val = np.asarray(values, dtype=float)
    pos = np.where(np.isfinite(val) & (val > 0), val, 0.0)
    denom = np.sum(pos, axis=2)
    num = np.sum(pos * lat_sub[None, None, :], axis=2)
    out = num / denom
    out[denom <= 0] = np.nan
    return out


def _positive_edges(lat_sub: np.ndarray, profile: np.ndarray, settings: LeadLagScreenSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_year, n_day, n_lat = profile.shape
    north = np.full((n_year, n_day), np.nan, dtype=float)
    south = np.full((n_year, n_day), np.nan, dtype=float)
    width = np.full((n_year, n_day), np.nan, dtype=float)
    flag = np.full((n_year, n_day), "ok", dtype=object)
    min_run = max(int(settings.v_positive_min_consecutive_grid), 1)

    for yi in range(n_year):
        for di in range(n_day):
            prof = profile[yi, di, :].astype(float)
            finite = np.isfinite(prof)
            if finite.sum() == 0:
                flag[yi, di] = "all_nan_profile"
                continue
            posmax = np.nanmax(np.where(finite, prof, np.nan))
            if (not np.isfinite(posmax)) or posmax <= 0:
                flag[yi, di] = "no_positive_profile"
                continue
            threshold = max(float(settings.v_positive_threshold_abs), float(settings.v_positive_threshold_fraction_of_max) * float(posmax))
            mask = finite & (prof > threshold)
            if mask.sum() < min_run:
                flag[yi, di] = "positive_band_below_min_run"
                continue
            # Keep all threshold-positive latitudes; north/south edge are bounds of the positive support.
            idx = np.where(mask)[0]
            south[yi, di] = float(np.nanmin(lat_sub[idx]))
            north[yi, di] = float(np.nanmax(lat_sub[idx]))
            width[yi, di] = float(north[yi, di] - south[yi, di])
    return north, south, width, flag


def _make_day_anomaly(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    day_means = out.groupby("day", dropna=False)[value_cols].transform("mean")
    out[value_cols] = out[value_cols] - day_means
    return out


def compute_v1_1_structural_indices(settings: LeadLagScreenSettings, logger=None) -> dict[str, object]:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.summary_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("Computing V1_1 structural indices from %s", settings.input_smoothed_fields)

    with np.load(settings.input_smoothed_fields, allow_pickle=False) as npz:
        precip = _field_from_npz(npz, ["precip_smoothed", "precip", "P_smoothed", "P"])
        v850 = _field_from_npz(npz, ["v850_smoothed", "v850", "V850_smoothed", "V850"])
        lat = _field_from_npz(npz, ["lat", "latitude"])
        lon = _field_from_npz(npz, ["lon", "longitude"])
        years = _field_from_npz(npz, ["years", "year"])

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    years = np.asarray(years).astype(int)
    n_year, n_day = int(precip.shape[0]), int(precip.shape[1])
    if len(years) != n_year:
        raise ValueError(f"years length {len(years)} does not match field year dimension {n_year}")

    if logger:
        logger.info("Field shape years=%d days=%d lat=%d lon=%d", n_year, n_day, len(lat), len(lon))

    # P structural indices.
    p_main = _area_mean(precip, lat, lon, settings.p_main_lat_range, settings.p_main_lon_range)
    p_south = _area_mean(precip, lat, lon, settings.p_south_lat_range, settings.p_south_lon_range)
    p_scs = _area_mean(precip, lat, lon, settings.p_scs_lat_range, settings.p_scs_lon_range)
    p_high40 = _area_mean(precip, lat, lon, settings.p_highlat_40_60_lat_range, settings.index_lon_range)
    p_high35 = _area_mean(precip, lat, lon, settings.p_highlat_35_60_lat_range, settings.index_lon_range)

    # V structural indices.
    v_high = _area_mean(v850, lat, lon, settings.v_highlat_lat_range, settings.index_lon_range)
    v_low = _area_mean(v850, lat, lon, settings.v_lowlat_lat_range, settings.index_lon_range)
    lat_prof, v_prof = _lat_profile(v850, lat, lon, (float(np.nanmin(lat)), float(np.nanmax(lat))), settings.index_lon_range)
    v_pos = np.where(np.isfinite(v_prof) & (v_prof > 0), v_prof, 0.0)
    v_cent = _weighted_centroid(lat_prof, v_pos)
    v_north, v_south, v_width, edge_flag = _positive_edges(lat_prof, v_prof, settings)

    grid = pd.MultiIndex.from_product([years, np.arange(1, n_day + 1)], names=["year", "day"]).to_frame(index=False)
    raw = grid.copy()
    def flat(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=float).reshape(-1)

    raw["P_main_28_35_mean"] = flat(p_main)
    raw["P_south_10_25_mean"] = flat(p_south)
    raw["P_scs_10_20_mean"] = flat(p_scs)
    raw["P_highlat_40_60_mean"] = flat(p_high40)
    raw["P_highlat_35_60_mean"] = flat(p_high35)
    raw["P_highlat_minus_main"] = raw["P_highlat_40_60_mean"] - raw["P_main_28_35_mean"]
    raw["P_south_minus_main"] = raw["P_south_10_25_mean"] - raw["P_main_28_35_mean"]
    raw["P_south_plus_highlat_minus_main"] = raw["P_south_10_25_mean"] + raw["P_highlat_40_60_mean"] - raw["P_main_28_35_mean"]

    raw["V_pos_north_edge_lat"] = flat(v_north)
    raw["V_pos_south_edge_lat"] = flat(v_south)
    raw["V_pos_band_width"] = flat(v_width)
    raw["V_pos_centroid_lat_recomputed"] = flat(v_cent)
    raw["V_highlat_35_55_mean"] = flat(v_high)
    raw["V_lowlat_20_30_mean"] = flat(v_low)
    raw["V_high_minus_low_35_55_minus_20_30"] = raw["V_highlat_35_55_mean"] - raw["V_lowlat_20_30_mean"]
    raw["V_lowlat_weakening_proxy_20_30"] = -raw["V_lowlat_20_30_mean"]

    raw.to_csv(settings.generated_index_raw, index=False, encoding="utf-8-sig")

    structural_cols = list(settings.new_v_indices + settings.new_p_indices)
    anomaly_new = _make_day_anomaly(raw[["year", "day", *structural_cols]], structural_cols)

    # Combine with read-only V1 old index anomalies.
    if not settings.input_v1_index_anomalies.exists():
        raise FileNotFoundError(f"V1 old index anomaly file not found: {settings.input_v1_index_anomalies}")
    old = pd.read_csv(settings.input_v1_index_anomalies, encoding="utf-8-sig")
    old["year"] = old["year"].astype(int)
    old["day"] = old["day"].astype(int)
    old_required = list(settings.old_v_indices + settings.old_p_indices)
    missing_old = [c for c in old_required if c not in old.columns]
    if missing_old:
        raise ValueError(f"V1 old anomaly file missing required old P/V columns: {missing_old}")
    old_keep = old[["year", "day", *old_required]].copy()
    combined = old_keep.merge(anomaly_new, on=["year", "day"], how="left", validate="one_to_one")
    combined.to_csv(settings.generated_index_anomalies, index=False, encoding="utf-8-sig")

    # Registry and quality flags.
    reg_rows = []
    for name in settings.old_v_indices:
        reg_rows.append({"index_name": name, "family": "V", "old_or_new": "old_v1", "source_field": "V1_index_anomalies", "definition": "Read-only V1 anomaly index."})
    for name in settings.old_p_indices:
        reg_rows.append({"index_name": name, "family": "P", "old_or_new": "old_v1", "source_field": "V1_index_anomalies", "definition": "Read-only V1 anomaly index."})
    defs = {
        "V_pos_north_edge_lat": "Northern edge latitude of positive v850 band over 100-135E.",
        "V_pos_south_edge_lat": "Southern edge latitude of positive v850 band over 100-135E.",
        "V_pos_band_width": "Positive v850 band width = north_edge - south_edge.",
        "V_pos_centroid_lat_recomputed": "Positive-value weighted centroid latitude of v850 profile over 100-135E.",
        "V_highlat_35_55_mean": "Area-weighted v850 mean over 35-55N,100-135E.",
        "V_lowlat_20_30_mean": "Area-weighted v850 mean over 20-30N,100-135E.",
        "V_high_minus_low_35_55_minus_20_30": "V_highlat_35_55_mean - V_lowlat_20_30_mean.",
        "V_lowlat_weakening_proxy_20_30": "Negative of V_lowlat_20_30_mean; larger means low-lat v850 weakening.",
        "P_main_28_35_mean": "Area-weighted precipitation mean over 28-35N,100-125E.",
        "P_south_10_25_mean": "Area-weighted precipitation mean over 10-25N,105-130E.",
        "P_scs_10_20_mean": "Area-weighted precipitation mean over 10-20N,105-130E.",
        "P_highlat_40_60_mean": "Area-weighted precipitation mean over 40-60N,100-135E.",
        "P_highlat_35_60_mean": "Area-weighted precipitation mean over 35-60N,100-135E.",
        "P_highlat_minus_main": "P_highlat_40_60_mean - P_main_28_35_mean.",
        "P_south_minus_main": "P_south_10_25_mean - P_main_28_35_mean.",
        "P_south_plus_highlat_minus_main": "P_south_10_25_mean + P_highlat_40_60_mean - P_main_28_35_mean.",
    }
    for name in settings.new_v_indices:
        reg_rows.append({"index_name": name, "family": "V", "old_or_new": "new_v1_1", "source_field": "v850_smoothed", "definition": defs[name]})
    for name in settings.new_p_indices:
        reg_rows.append({"index_name": name, "family": "P", "old_or_new": "new_v1_1", "source_field": "precip_smoothed", "definition": defs[name]})
    pd.DataFrame(reg_rows).to_csv(settings.index_dir / "v1_1_index_registry.csv", index=False, encoding="utf-8-sig")

    qf = grid.copy()
    qf["V_positive_edge_flag"] = edge_flag.reshape(-1)
    qf.to_csv(settings.index_dir / "v1_1_index_quality_flags.csv", index=False, encoding="utf-8-sig")

    meta = {
        "status": "success",
        "input_smoothed_fields": str(settings.input_smoothed_fields),
        "input_v1_index_anomalies": str(settings.input_v1_index_anomalies),
        "generated_index_raw": str(settings.generated_index_raw),
        "generated_index_anomalies": str(settings.generated_index_anomalies),
        "field_shape": [int(x) for x in precip.shape],
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
        "index_value_mode_for_screen": "day-of-season anomaly",
        "v_positive_threshold_abs": settings.v_positive_threshold_abs,
        "v_positive_threshold_fraction_of_max": settings.v_positive_threshold_fraction_of_max,
    }
    (settings.summary_dir / "structural_index_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
