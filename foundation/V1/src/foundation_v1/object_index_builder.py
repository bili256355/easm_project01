from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class RegionSpec:
    object_name: str
    source_field: str
    lon_range: Tuple[float, float]
    lat_range: Tuple[float, float]
    note: str


REGION_SPECS: Dict[str, RegionSpec] = {
    "P": RegionSpec(
        "P",
        "precip",
        (105.0, 125.0),
        (10.0, 50.0),
        "P 提取扩展到 10–50N，其中 10–40N 用于原主带/南带竞争组，35–45N 与 10–50N 用于梅雨后北带接替与整体北跳变量。",
    ),
    "V": RegionSpec(
        "V",
        "v850",
        (105.0, 125.0),
        (10.0, 30.0),
        "V 南边界固定为 10N，不再使用 5N。",
    ),
    "H": RegionSpec(
        "H",
        "z500",
        (110.0, 140.0),
        (10.0, 40.0),
        "H core 区域。",
    ),
    "Jw": RegionSpec(
        "Jw",
        "u200",
        (80.0, 110.0),
        (20.0, 50.0),
        "上游急流 core 区域。",
    ),
    "Je": RegionSpec(
        "Je",
        "u200",
        (120.0, 150.0),
        (20.0, 50.0),
        "下游急流 core 区域。",
    ),
}


VARIABLE_ORDER: Tuple[str, ...] = (
    "P_main_band_share",
    "P_south_band_share_18_24",
    "P_main_minus_south",
    "P_spread_lat",
    "P_north_band_share_35_45",
    "P_north_minus_main_35_45",
    "P_total_centroid_lat_10_50",
    "V_strength",
    "V_pos_centroid_lat",
    "V_NS_diff",
    "H_strength",
    "H_centroid_lat",
    "H_west_extent_lon",
    "H_zonal_width",
    "Je_strength",
    "Je_axis_lat",
    "Je_meridional_width",
    "Jw_strength",
    "Jw_axis_lat",
    "Jw_meridional_width",
)


def build_region_table() -> List[dict]:
    return [
        {
            "object_name": spec.object_name,
            "source_field": spec.source_field,
            "lon_range": f"{spec.lon_range[0]:.0f}-{spec.lon_range[1]:.0f}E",
            "lat_range": f"{spec.lat_range[0]:.0f}-{spec.lat_range[1]:.0f}N",
            "note": spec.note,
        }
        for spec in REGION_SPECS.values()
    ]


def build_variable_definition_table() -> List[dict]:
    return [
        {
            "variable_name": "P_main_band_share",
            "object_name": "P",
            "formula": "Main(24-35N) / Total_P(10-40N)",
            "project_meaning": "主雨带份额",
        },
        {
            "variable_name": "P_south_band_share_18_24",
            "object_name": "P",
            "formula": "South(18-24N) / Total_P(10-40N)",
            "project_meaning": "南带份额",
        },
        {
            "variable_name": "P_main_minus_south",
            "object_name": "P",
            "formula": "P_main_band_share - P_south_band_share_18_24",
            "project_meaning": "主带对南带竞争差值",
        },
        {
            "variable_name": "P_spread_lat",
            "object_name": "P",
            "formula": "sqrt(Σ(P(φ)*(φ-μ)^2)/ΣP(φ)), φ∈10-40N",
            "project_meaning": "总带纬向展宽",
        },
        {
            "variable_name": "P_north_band_share_35_45",
            "object_name": "P",
            "formula": "North(35-45N) / Total_P_late(10-50N)",
            "project_meaning": "盛夏北带份额",
        },
        {
            "variable_name": "P_north_minus_main_35_45",
            "object_name": "P",
            "formula": "P_north_band_share_35_45 - Main(24-35N)/Total_P_late(10-50N)",
            "project_meaning": "北带相对主带接替差值",
        },
        {
            "variable_name": "P_total_centroid_lat_10_50",
            "object_name": "P",
            "formula": "Σ(P(φ)*φ)/ΣP(φ), φ∈10-50N",
            "project_meaning": "整体降水带重心纬度",
        },
        {
            "variable_name": "V_strength",
            "object_name": "V",
            "formula": "mean_{φ∈10-30N} V(φ)",
            "project_meaning": "V 整体强度",
        },
        {
            "variable_name": "V_pos_centroid_lat",
            "object_name": "V",
            "formula": "Σ(V+(φ)*φ)/ΣV+(φ), φ∈10-30N",
            "project_meaning": "正向活动带重心纬度",
        },
        {
            "variable_name": "V_NS_diff",
            "object_name": "V",
            "formula": "Σ_{15-30N}V(φ) - Σ_{10-15N}V(φ)",
            "project_meaning": "南北差结构量",
        },
        {
            "variable_name": "H_strength",
            "object_name": "H",
            "formula": "mean(H | H >= q90(core))",
            "project_meaning": "H 高核强度",
        },
        {
            "variable_name": "H_centroid_lat",
            "object_name": "H",
            "formula": "Σ(w*φ)/Σw, w=max(H-q90,0)",
            "project_meaning": "H 重心纬度",
        },
        {
            "variable_name": "H_west_extent_lon",
            "object_name": "H",
            "formula": "weighted q10(lon, Σφw)",
            "project_meaning": "H 西伸位置",
        },
        {
            "variable_name": "H_zonal_width",
            "object_name": "H",
            "formula": "weighted q90 - weighted q10",
            "project_meaning": "H 纬向展宽",
        },
        {
            "variable_name": "Je_strength",
            "object_name": "Je",
            "formula": "mean(profile | profile >= q90(profile))",
            "project_meaning": "Je 强度",
        },
        {
            "variable_name": "Je_axis_lat",
            "object_name": "Je",
            "formula": "Σ(w*φ)/Σw, w=max(profile-q90,0)",
            "project_meaning": "Je 轴线纬度",
        },
        {
            "variable_name": "Je_meridional_width",
            "object_name": "Je",
            "formula": "weighted q90(lat,w) - weighted q10(lat,w)",
            "project_meaning": "Je 纬向宽度",
        },
        {
            "variable_name": "Jw_strength",
            "object_name": "Jw",
            "formula": "mean(profile | profile >= q90(profile))",
            "project_meaning": "Jw 强度",
        },
        {
            "variable_name": "Jw_axis_lat",
            "object_name": "Jw",
            "formula": "Σ(w*φ)/Σw, w=max(profile-q90,0)",
            "project_meaning": "Jw 轴线纬度",
        },
        {
            "variable_name": "Jw_meridional_width",
            "object_name": "Jw",
            "formula": "weighted q90(lat,w) - weighted q10(lat,w)",
            "project_meaning": "Jw 纬向宽度",
        },
    ]


def _mask_between(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.isfinite(arr) & (arr >= lower) & (arr <= upper)


def _safe_nanmean(arr: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(arr)
    count = np.sum(valid, axis=axis)
    summed = np.nansum(arr, axis=axis)
    out = np.full(summed.shape, np.nan, dtype=np.float64)
    np.divide(summed, count, out=out, where=count > 0)
    return out


def _extract_lonmean_profile(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lat_mask = _mask_between(lat, *lat_range)
    lon_mask = _mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"纬度范围 {lat_range} 在坐标中未命中任何点。")
    if not np.any(lon_mask):
        raise ValueError(f"经度范围 {lon_range} 在坐标中未命中任何点。")
    sub = np.asarray(field[:, :, lat_mask, :][:, :, :, lon_mask], dtype=np.float64)
    profile = _safe_nanmean(sub, axis=3)
    return profile, np.asarray(lat[lat_mask], dtype=np.float64)


def _extract_subfield(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_mask = _mask_between(lat, *lat_range)
    lon_mask = _mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"纬度范围 {lat_range} 在坐标中未命中任何点。")
    if not np.any(lon_mask):
        raise ValueError(f"经度范围 {lon_range} 在坐标中未命中任何点。")
    sub = np.asarray(field[:, :, lat_mask, :][:, :, :, lon_mask], dtype=np.float64)
    return sub, np.asarray(lat[lat_mask], dtype=np.float64), np.asarray(lon[lon_mask], dtype=np.float64)


def _weighted_mean_1d(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return np.nan
    v = values[valid]
    w = weights[valid]
    wsum = np.sum(w)
    if wsum <= 0:
        return np.nan
    return float(np.sum(v * w) / wsum)


def _weighted_std_1d(values: np.ndarray, weights: np.ndarray) -> float:
    mu = _weighted_mean_1d(values, weights)
    if not np.isfinite(mu):
        return np.nan
    valid = np.isfinite(values) & np.isfinite(weights)
    v = values[valid]
    w = weights[valid]
    wsum = np.sum(w)
    if wsum <= 0:
        return np.nan
    var = np.sum(w * (v - mu) ** 2) / wsum
    return float(np.sqrt(max(var, 0.0)))


def _weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return np.nan
    v = np.asarray(values[valid], dtype=np.float64)
    w = np.asarray(weights[valid], dtype=np.float64)
    positive = w > 0
    if not np.any(positive):
        return np.nan
    v = v[positive]
    w = w[positive]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cum = np.cumsum(w)
    total = cum[-1]
    if total <= 0:
        return np.nan
    cutoff = q * total
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    idx = max(0, min(idx, len(v) - 1))
    return float(v[idx])


def _row_quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return np.nan
    return float(np.quantile(finite, q))


def _threshold_excess(values: np.ndarray, q: float = 0.9) -> tuple[np.ndarray, float]:
    sq = _row_quantile(values, q)
    if not np.isfinite(sq):
        return np.full_like(values, np.nan, dtype=np.float64), np.nan
    out = np.where(np.isfinite(values), np.maximum(values - sq, 0.0), np.nan)
    return out, sq


def _soft_peak_lat(values: np.ndarray, lats: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    idx = np.nanargmax(values)
    return float(lats[int(idx)])


def _sum_band(profile: np.ndarray, lats: np.ndarray, lat_range: Tuple[float, float]) -> float:
    mask = _mask_between(lats, *lat_range)
    if not np.any(mask):
        return np.nan
    vals = np.asarray(profile[mask], dtype=np.float64)
    if not np.any(np.isfinite(vals)):
        return np.nan
    return float(np.nansum(vals))


def _subset(profile: np.ndarray, lats: np.ndarray, lat_range: Tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    mask = _mask_between(lats, *lat_range)
    return np.asarray(profile[mask], dtype=np.float64), np.asarray(lats[mask], dtype=np.float64)


def _compute_p_indices(profile: np.ndarray, lats: np.ndarray) -> Dict[str, float]:
    total_10_40 = _sum_band(profile, lats, (10.0, 40.0))
    total_10_50 = _sum_band(profile, lats, (10.0, 50.0))
    main = _sum_band(profile, lats, (24.0, 35.0))
    south = _sum_band(profile, lats, (18.0, 24.0))
    north = _sum_band(profile, lats, (35.0, 45.0))

    main_share = np.nan if (not np.isfinite(total_10_40) or total_10_40 == 0) else float(main / total_10_40)
    south_share = np.nan if (not np.isfinite(total_10_40) or total_10_40 == 0) else float(south / total_10_40)
    north_share = np.nan if (not np.isfinite(total_10_50) or total_10_50 == 0) else float(north / total_10_50)
    main_share_late = np.nan if (not np.isfinite(total_10_50) or total_10_50 == 0) else float(main / total_10_50)

    prof_10_40, lats_10_40 = _subset(profile, lats, (10.0, 40.0))
    spread = _weighted_std_1d(lats_10_40, prof_10_40)

    prof_10_50, lats_10_50 = _subset(profile, lats, (10.0, 50.0))
    total_centroid = _weighted_mean_1d(lats_10_50, prof_10_50)

    return {
        "P_main_band_share": main_share,
        "P_south_band_share_18_24": south_share,
        "P_main_minus_south": np.nan if (not np.isfinite(main_share) or not np.isfinite(south_share)) else float(main_share - south_share),
        "P_spread_lat": spread,
        "P_north_band_share_35_45": north_share,
        "P_north_minus_main_35_45": np.nan if (not np.isfinite(north_share) or not np.isfinite(main_share_late)) else float(north_share - main_share_late),
        "P_total_centroid_lat_10_50": total_centroid,
    }


def _compute_v_indices(profile: np.ndarray, lats: np.ndarray) -> Dict[str, float]:
    work_prof, work_lats = _subset(profile, lats, (10.0, 30.0))
    strength = float(np.nanmean(work_prof)) if np.any(np.isfinite(work_prof)) else np.nan
    pos_weights = np.where(np.isfinite(work_prof), np.maximum(work_prof, 0.0), np.nan)
    pos_centroid = _weighted_mean_1d(work_lats, pos_weights)
    north = _sum_band(profile, lats, (15.0, 30.0))
    south = _sum_band(profile, lats, (10.0, 15.0))
    return {
        "V_strength": strength,
        "V_pos_centroid_lat": pos_centroid,
        "V_NS_diff": np.nan if (not np.isfinite(north) or not np.isfinite(south)) else float(north - south),
    }


def _compute_jet_indices(profile: np.ndarray, lats: np.ndarray, prefix: str) -> Dict[str, float]:
    sq = _row_quantile(profile, 0.9)
    if np.isfinite(sq):
        strength_vals = profile[np.isfinite(profile) & (profile >= sq)]
        strength = float(np.nanmean(strength_vals)) if strength_vals.size > 0 else np.nan
    else:
        strength = np.nan
    weights, _ = _threshold_excess(profile, q=0.9)
    axis_lat = _weighted_mean_1d(lats, weights)
    if not np.isfinite(axis_lat):
        axis_lat = _soft_peak_lat(profile, lats)
    width = np.nan
    if np.any(np.isfinite(weights) & (weights > 0)):
        q10 = _weighted_quantile_1d(lats, weights, 0.10)
        q90 = _weighted_quantile_1d(lats, weights, 0.90)
        if np.isfinite(q10) and np.isfinite(q90):
            width = float(q90 - q10)
    return {
        f"{prefix}_strength": strength,
        f"{prefix}_axis_lat": axis_lat,
        f"{prefix}_meridional_width": width,
    }


def _compute_h_indices(subfield: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> Dict[str, float]:
    flat = subfield[np.isfinite(subfield)]
    sq = _row_quantile(flat, 0.9)
    if not np.isfinite(sq):
        return {
            "H_strength": np.nan,
            "H_centroid_lat": np.nan,
            "H_west_extent_lon": np.nan,
            "H_zonal_width": np.nan,
        }
    strength_vals = subfield[np.isfinite(subfield) & (subfield >= sq)]
    strength = float(np.nanmean(strength_vals)) if strength_vals.size > 0 else np.nan
    weights = np.where(np.isfinite(subfield), np.maximum(subfield - sq, 0.0), np.nan)
    lat_weights = np.nansum(weights, axis=1)
    centroid_lat = _weighted_mean_1d(lats, lat_weights)
    lon_weights = np.nansum(weights, axis=0)
    west = _weighted_quantile_1d(lons, lon_weights, 0.10)
    east = _weighted_quantile_1d(lons, lon_weights, 0.90)
    width = np.nan if (not np.isfinite(west) or not np.isfinite(east)) else float(east - west)
    return {
        "H_strength": strength,
        "H_centroid_lat": centroid_lat,
        "H_west_extent_lon": west,
        "H_zonal_width": width,
    }


def compute_indices(smoothed_fields: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    n_years, n_days = smoothed_fields["precip"].shape[:2]
    index_arrays = {name: np.full((n_years, n_days), np.nan, dtype=np.float64) for name in VARIABLE_ORDER}

    p_profile, p_lats = _extract_lonmean_profile(smoothed_fields["precip"], lat, lon, REGION_SPECS["P"].lat_range, REGION_SPECS["P"].lon_range)
    v_profile, v_lats = _extract_lonmean_profile(smoothed_fields["v850"], lat, lon, REGION_SPECS["V"].lat_range, REGION_SPECS["V"].lon_range)
    je_profile, je_lats = _extract_lonmean_profile(smoothed_fields["u200"], lat, lon, REGION_SPECS["Je"].lat_range, REGION_SPECS["Je"].lon_range)
    jw_profile, jw_lats = _extract_lonmean_profile(smoothed_fields["u200"], lat, lon, REGION_SPECS["Jw"].lat_range, REGION_SPECS["Jw"].lon_range)
    h_subfield, h_lats, h_lons = _extract_subfield(smoothed_fields["z500"], lat, lon, REGION_SPECS["H"].lat_range, REGION_SPECS["H"].lon_range)

    for yi in tqdm(range(n_years), desc="compute_indices(year)", leave=False):
        for di in range(n_days):
            result: Dict[str, float] = {}
            result.update(_compute_p_indices(p_profile[yi, di, :], p_lats))
            result.update(_compute_v_indices(v_profile[yi, di, :], v_lats))
            result.update(_compute_h_indices(h_subfield[yi, di, :, :], h_lats, h_lons))
            result.update(_compute_jet_indices(je_profile[yi, di, :], je_lats, "Je"))
            result.update(_compute_jet_indices(jw_profile[yi, di, :], jw_lats, "Jw"))
            for name in VARIABLE_ORDER:
                index_arrays[name][yi, di] = result.get(name, np.nan)

    meta = {
        "source_field_level": "smoothed_fields_only",
        "notes": [
            "所有对象变量均基于 smoothed fields 计算，不直接在 anomaly fields 上定义。",
            "P 变量体系保留 10–40N 的原竞争组，并补入 10–50N 的北带接替/整体北跳组。",
            "V 南边界固定为 10N。",
            "H 在二维 core 区域上做 q90 thresholded-excess 权重结构量。",
            "Je/Jw 在 lon-mean lat profile 上做 q90 thresholded-excess 结构量。",
        ],
        "profile_latitudes": {
            "P": p_lats.tolist(),
            "V": v_lats.tolist(),
            "H": h_lats.tolist(),
            "Je": je_lats.tolist(),
            "Jw": jw_lats.tolist(),
        },
        "profile_longitudes": {
            "H": h_lons.tolist(),
        },
    }
    return index_arrays, meta


def compute_index_daily_climatology(index_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, values in index_arrays.items():
        valid = np.isfinite(values)
        count = np.sum(valid, axis=0)
        summed = np.nansum(values, axis=0)
        clim = np.full(values.shape[1], np.nan, dtype=np.float64)
        np.divide(summed, count, out=clim, where=count > 0)
        out[name] = clim
    return out


def compute_index_anomalies(index_arrays: Dict[str, np.ndarray], daily_climatology: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: index_arrays[name] - daily_climatology[name][None, :] for name in index_arrays}


def build_index_value_table(index_arrays: Dict[str, np.ndarray], years: np.ndarray) -> pd.DataFrame:
    rows = []
    year_values = np.asarray(years)
    sample = next(iter(index_arrays.values()))
    n_years, n_days = sample.shape
    for yi in range(n_years):
        for di in range(n_days):
            row = {"year": int(year_values[yi]), "year_index": int(yi), "day": int(di + 1)}
            for name in VARIABLE_ORDER:
                value = index_arrays[name][yi, di]
                row[name] = float(value) if np.isfinite(value) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def build_daily_climatology_table(daily_climatology: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    n_days = next(iter(daily_climatology.values())).shape[0]
    for di in range(n_days):
        row = {"day": int(di + 1)}
        for name in VARIABLE_ORDER:
            value = daily_climatology[name][di]
            row[name] = float(value) if np.isfinite(value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_index_summary_table(index_arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in VARIABLE_ORDER:
        arr = np.asarray(index_arrays[name], dtype=np.float64)
        valid = np.isfinite(arr)
        rows.append(
            {
                "variable_name": name,
                "valid_count": int(np.sum(valid)),
                "nan_count": int(arr.size - np.sum(valid)),
                "valid_ratio": float(np.mean(valid)),
                "mean": float(np.nanmean(arr)) if np.any(valid) else np.nan,
                "std": float(np.nanstd(arr)) if np.any(valid) else np.nan,
                "min": float(np.nanmin(arr)) if np.any(valid) else np.nan,
                "max": float(np.nanmax(arr)) if np.any(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)
