from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import shutil
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import V10_8_A_Settings


# =============================================================================
# V10.8_a: object-internal transition content audit
# =============================================================================
# Method boundary:
# - This is NOT a new breakpoint detector.
# - This does NOT use joint windows or accepted windows.
# - This does NOT infer precursor/cause/pathway/cross-object coupling.
# - This reconstructs the left/right detector flank around each V10.2 object-native
#   break day using the V10.2 detector width, decomposes the reconstructed
#   object-state change vector, and then translates the dominant content into
#   object-internal semantic metrics.
# =============================================================================

OBJECTS = ("P", "V", "H", "Je", "Jw")

FIELD_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "P": ("precip_smoothed", "precip", "P", "pr", "rain", "tp"),
    "V": ("v850_smoothed", "v850", "V", "va850", "v_component_850", "vwind850"),
    "H": ("z500_smoothed", "z500", "H", "hgt500", "geopotential_500", "zg500"),
    "Je": ("u200_smoothed", "u200", "U200", "u_component_200", "uwind200"),
    "Jw": ("u200_smoothed", "u200", "U200", "u_component_200", "uwind200"),
}

PROFILE_RANGE_KEYS: dict[str, tuple[str, str]] = {
    "P": ("p_lat_range", "p_lon_range"),
    "V": ("v_lat_range", "v_lon_range"),
    "H": ("h_lat_range", "h_lon_range"),
    "Je": ("je_lat_range", "je_lon_range"),
    "Jw": ("jw_lat_range", "jw_lon_range"),
}

SEMANTIC_GROUP_MAP: dict[str, str] = {
    "P_total_strength": "strength",
    "P_centroid_lat": "position",
    "P_spread_lat": "rainband_width",
    "P_main_band_share": "rainband_structure",
    "P_south_band_share_18_24": "rainband_structure",
    "P_main_minus_south": "rainband_structure",
    "V_strength": "strength",
    "V_NS_diff": "latitudinal_structure",
    "V_pos_centroid_lat": "position",
    "H_strength": "strength",
    "H_centroid_lat": "position",
    "H_west_extent_lon": "shape_extent",
    "H_zonal_width": "shape_extent",
    "H_north_edge_lat": "shape_extent",
    "H_south_edge_lat": "shape_extent",
    "Je_strength": "jet_strength",
    "Je_axis_lat": "jet_axis",
    "Je_meridional_width": "jet_width",
    "Jw_strength": "jet_strength",
    "Jw_axis_lat": "jet_axis",
    "Jw_meridional_width": "jet_width",
}

TRANSITION_TYPE_BY_GROUP: dict[str, str] = {
    "profile_structure": "lat_profile_reorganization",
    "strength": "strength_change",
    "position": "position_shift",
    "rainband_width": "rainband_width_change",
    "rainband_structure": "rainband_redistribution",
    "latitudinal_structure": "latitudinal_structure_change",
    "shape_extent": "morphology_extent_change",
    "jet_strength": "jet_strength_change",
    "jet_axis": "jet_axis_shift",
    "jet_width": "jet_width_change",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(settings: V10_8_A_Settings, msg: str) -> None:
    if settings.progress:
        print(f"[V10.8_a] {msg}", flush=True)


def clean_output_root(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for sub in ("tables", "figures", "run_meta", "logs"):
        (path / sub).mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_nanmean(a: np.ndarray, axis=None, keepdims: bool = False):
    arr = np.asarray(a, dtype=float)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis, keepdims=keepdims)
    total = np.nansum(arr, axis=axis, keepdims=keepdims)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return np.where(count > 0, mean, np.nan)


def safe_nanstd(a: np.ndarray, axis=None, keepdims: bool = False):
    arr = np.asarray(a, dtype=float)
    mean = safe_nanmean(arr, axis=axis, keepdims=True)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis, keepdims=True)
    sq = np.where(valid, (arr - mean) ** 2, 0.0)
    var = sq.sum(axis=axis, keepdims=True) / np.where(count > 0, count, 1)
    std = np.sqrt(var)
    std = np.where(count > 0, std, np.nan)
    if not keepdims and axis is not None:
        std = np.squeeze(std, axis=axis)
    return std


def first_key(data: dict[str, np.ndarray], candidates: tuple[str, ...]) -> str | None:
    lower = {str(k).lower(): k for k in data.keys()}
    for c in candidates:
        if c in data:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_windows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing V10.2 object-native windows table: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"object", "window_id", "start_day", "end_day", "center_day", "main_peak_day"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"V10.2 windows table missing columns: {sorted(missing)}")
    df = df[df["object"].isin(OBJECTS)].copy()
    for col in ("start_day", "end_day", "center_day", "main_peak_day"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    return df.sort_values(["object", "main_peak_day", "window_id"]).reset_index(drop=True)


def load_npz(settings: V10_8_A_Settings) -> dict[str, Any]:
    path = settings.smoothed_fields_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields file: {path}")
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    lat_key = first_key(data, ("lat", "latitude", "lats"))
    lon_key = first_key(data, ("lon", "longitude", "lons"))
    year_key = first_key(data, ("year", "years"))
    day_key = first_key(data, ("day", "days", "day_index"))
    if lat_key is None or lon_key is None:
        raise KeyError(f"Could not detect lat/lon keys. Available={sorted(data.keys())}")
    return {
        "path": path,
        "data": data,
        "lat": np.asarray(data[lat_key], dtype=float).ravel(),
        "lon": np.asarray(data[lon_key], dtype=float).ravel(),
        "lat_key": lat_key,
        "lon_key": lon_key,
        "year_key": year_key,
        "day_key": day_key,
    }


def normalize_field_dims(field: np.ndarray, data: dict[str, np.ndarray], year_key: str | None, day_key: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=float)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected year x day x lat x lon or day x lat x lon; got shape={field.shape}")
    if year_key is not None and len(np.asarray(data[year_key]).ravel()) == arr.shape[0]:
        years = np.asarray(data[year_key]).ravel()
    else:
        years = np.arange(arr.shape[0], dtype=int)
    if day_key is not None and len(np.asarray(data[day_key]).ravel()) == arr.shape[1]:
        days = np.asarray(data[day_key]).ravel().astype(int)
    else:
        days = np.arange(arr.shape[1], dtype=int)
    return arr, years, days


def subset_domain(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
    lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError(f"No grid points in requested domain lat={lat_range}, lon={lon_range}")
    sub = field[:, :, lat_mask, :][:, :, :, lon_mask]
    sub_lat = lat[lat_mask]
    sub_lon = lon[lon_mask]
    order_lat = np.argsort(sub_lat)  # always low -> high for metric/profile calculations
    order_lon = np.argsort(sub_lon)
    sub = sub[:, :, order_lat, :][:, :, :, order_lon]
    return sub, sub_lat[order_lat], sub_lon[order_lon]


def select_day_indices(days: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.where((days >= lo) & (days <= hi))[0]


def detector_flank_windows(break_day: int, half_width: int, days: np.ndarray) -> dict[str, Any]:
    # Exclude the break day itself. The goal is to reconstruct left/right detector-side states,
    # not to average the whole derived object-native window.
    left_lo = int(break_day - half_width)
    left_hi = int(break_day - 1)
    right_lo = int(break_day + 1)
    right_hi = int(break_day + half_width)
    li = select_day_indices(days, left_lo, left_hi)
    ri = select_day_indices(days, right_lo, right_hi)
    return {
        "left_start": left_lo,
        "left_end": left_hi,
        "right_start": right_lo,
        "right_end": right_hi,
        "left_indices": li,
        "right_indices": ri,
        "left_n_days": int(li.size),
        "right_n_days": int(ri.size),
        "edge_limited_flag": bool(li.size < half_width or ri.size < half_width),
        "insufficient_flank_flag": bool(li.size < 2 or ri.size < 2),
    }


def climatology_field(field: np.ndarray) -> np.ndarray:
    # V10.2 object-native detection is a seasonal-day state problem. This reconstruction
    # works on the year-mean daily field, so outputs are detector-content diagnostics,
    # not yearwise causality/statistical significance tests.
    return safe_nanmean(field, axis=0)  # day x lat x lon


def field_rms_daily(clim_field: np.ndarray) -> np.ndarray:
    return np.sqrt(safe_nanmean(clim_field ** 2, axis=(1, 2)))


def lat_weighted_centroid_spread_daily(clim_field: np.ndarray, lat: np.ndarray, positive_only: bool) -> tuple[np.ndarray, np.ndarray]:
    prof = safe_nanmean(clim_field, axis=2)  # day x lat
    weights = np.maximum(prof, 0.0) if positive_only else np.abs(prof)
    total = np.nansum(weights, axis=1)
    latv = lat.reshape(1, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cent = np.nansum(weights * latv, axis=1) / total
        spread = np.sqrt(np.nansum(weights * (latv - cent[:, None]) ** 2, axis=1) / total)
    cent = np.where(total > 1e-12, cent, np.nan)
    spread = np.where(total > 1e-12, spread, np.nan)
    return cent, spread


def band_share_daily(clim_field: np.ndarray, lat: np.ndarray, band: tuple[float, float], positive_only: bool = True) -> np.ndarray:
    arr = np.maximum(clim_field, 0.0) if positive_only else np.abs(clim_field)
    total = np.nansum(arr, axis=(1, 2))
    lo, hi = sorted(band)
    mask = (lat >= lo) & (lat <= hi)
    band_sum = np.nansum(arr[:, mask, :], axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = band_sum / total
    return np.where(total > 1e-12, share, np.nan)


def extent_metrics_daily(clim_field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_d = clim_field.shape[0]
    west = np.full(n_d, np.nan)
    width = np.full(n_d, np.nan)
    north_edge = np.full(n_d, np.nan)
    south_edge = np.full(n_d, np.nan)
    for d in range(n_d):
        f = clim_field[d]
        if not np.isfinite(f).any():
            continue
        thr = np.nanpercentile(f, 80)
        mask = f >= thr
        if not np.any(mask):
            continue
        lon_active = lon[np.any(mask, axis=0)]
        lat_active = lat[np.any(mask, axis=1)]
        if lon_active.size:
            west[d] = float(np.nanmin(lon_active))
            width[d] = float(np.nanmax(lon_active) - np.nanmin(lon_active))
        if lat_active.size:
            south_edge[d] = float(np.nanmin(lat_active))
            north_edge[d] = float(np.nanmax(lat_active))
    return west, width, north_edge, south_edge


def lat_profile_daily(clim_field: np.ndarray) -> np.ndarray:
    return safe_nanmean(clim_field, axis=2)  # day x lat


def rebin_profile_to_step(profile: np.ndarray, lat: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    lat_lo = math.floor(float(np.nanmin(lat)) / step) * step
    lat_hi = math.ceil(float(np.nanmax(lat)) / step) * step
    centers = np.arange(lat_lo, lat_hi + 0.5 * step, step)
    out = np.full((profile.shape[0], centers.size), np.nan)
    for i, c in enumerate(centers):
        if i == 0:
            mask = (lat >= c - 0.5 * step) & (lat <= c + 0.5 * step)
        else:
            mask = (lat > c - 0.5 * step) & (lat <= c + 0.5 * step)
        if np.any(mask):
            out[:, i] = safe_nanmean(profile[:, mask], axis=1)
    return out, centers


def build_semantic_metric_series(obj: str, clim_field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    metrics: dict[str, np.ndarray] = {}
    if obj == "P":
        metrics["P_total_strength"] = field_rms_daily(clim_field)
        cent, spread = lat_weighted_centroid_spread_daily(clim_field, lat, positive_only=True)
        metrics["P_centroid_lat"] = cent
        metrics["P_spread_lat"] = spread
        main = band_share_daily(clim_field, lat, (24.0, 35.0), positive_only=True)
        south = band_share_daily(clim_field, lat, (18.0, 24.0), positive_only=True)
        metrics["P_main_band_share"] = main
        metrics["P_south_band_share_18_24"] = south
        metrics["P_main_minus_south"] = main - south
    elif obj == "V":
        metrics["V_strength"] = field_rms_daily(clim_field)
        prof = lat_profile_daily(clim_field)
        mid = float(np.nanmedian(lat))
        south = safe_nanmean(prof[:, lat <= mid], axis=1)
        north = safe_nanmean(prof[:, lat > mid], axis=1)
        metrics["V_NS_diff"] = north - south
        cent, _ = lat_weighted_centroid_spread_daily(clim_field, lat, positive_only=False)
        metrics["V_pos_centroid_lat"] = cent
    elif obj == "H":
        metrics["H_strength"] = field_rms_daily(clim_field)
        cent, _ = lat_weighted_centroid_spread_daily(clim_field, lat, positive_only=False)
        metrics["H_centroid_lat"] = cent
        west, width, north_edge, south_edge = extent_metrics_daily(clim_field, lat, lon)
        metrics["H_west_extent_lon"] = west
        metrics["H_zonal_width"] = width
        metrics["H_north_edge_lat"] = north_edge
        metrics["H_south_edge_lat"] = south_edge
    elif obj in ("Je", "Jw"):
        metrics[f"{obj}_strength"] = field_rms_daily(clim_field)
        cent, spread = lat_weighted_centroid_spread_daily(clim_field, lat, positive_only=False)
        metrics[f"{obj}_axis_lat"] = cent
        metrics[f"{obj}_meridional_width"] = spread
    else:
        raise ValueError(f"Unknown object: {obj}")
    return metrics


def standardize_feature_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = safe_nanmean(x, axis=0)
    std = safe_nanstd(x, axis=0)
    std = np.where((np.isfinite(std)) & (std > 1e-12), std, np.nan)
    z = (x - mean.reshape(1, -1)) / std.reshape(1, -1)
    return z, mean, std


def mean_over_indices(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.full(x.shape[1], np.nan)
    return safe_nanmean(x[indices], axis=0)


def summarize_signed_delta(delta: np.ndarray) -> float:
    if not np.isfinite(delta).any():
        return np.nan
    return float(np.nanmean(delta))


def direction_label(metric: str, delta: float) -> str:
    if not np.isfinite(delta):
        return "not_available"
    if abs(delta) < 1e-12:
        return "near_zero"
    sign_pos = delta > 0
    if metric.endswith("centroid_lat") or metric.endswith("axis_lat") or metric.endswith("pos_centroid_lat"):
        return "northward" if sign_pos else "southward"
    if metric.endswith("spread_lat") or metric.endswith("meridional_width") or metric.endswith("zonal_width"):
        return "widening" if sign_pos else "narrowing"
    if metric.endswith("west_extent_lon"):
        # Smaller west_extent_lon means farther-west active extent under this metric definition.
        return "eastward_retreat_or_less_westward" if sign_pos else "westward_extension_or_more_westward"
    if metric.endswith("north_edge_lat"):
        return "north_edge_northward" if sign_pos else "north_edge_southward"
    if metric.endswith("south_edge_lat"):
        return "south_edge_northward" if sign_pos else "south_edge_southward"
    if "south_band_share" in metric:
        return "south_band_gain" if sign_pos else "south_band_loss"
    if "main_band_share" in metric:
        return "main_band_gain" if sign_pos else "main_band_loss"
    if "main_minus_south" in metric:
        return "main_minus_south_increase" if sign_pos else "main_minus_south_decrease"
    if metric.endswith("strength") or metric.endswith("total_strength") or metric.endswith("NS_diff"):
        return "increase" if sign_pos else "decrease"
    return "increase" if sign_pos else "decrease"


def content_confidence_note(total_norm: float, top_share: float, edge_limited: bool, consistency: str) -> str:
    notes = []
    if edge_limited:
        notes.append("edge_limited")
    if not np.isfinite(total_norm) or total_norm < 1e-9:
        notes.append("weak_detector_change")
    elif top_share >= 0.50:
        notes.append("detector_change_concentrated")
    elif top_share >= 0.30:
        notes.append("detector_change_mixed")
    else:
        notes.append("detector_change_distributed")
    if consistency != "not_evaluated":
        notes.append(f"semantic_{consistency}")
    return ";".join(notes)


def compare_detector_semantic(top_detector_group: str, top_semantic_group: str | None) -> str:
    if top_semantic_group is None or not isinstance(top_semantic_group, str):
        return "semantic_unavailable"
    if top_detector_group == top_semantic_group:
        return "consistent"
    if top_detector_group == "profile_structure" and top_semantic_group in {
        "rainband_structure", "latitudinal_structure", "position", "rainband_width", "jet_axis", "jet_width", "shape_extent"
    }:
        return "partial"
    return "mixed_or_inconsistent"


def get_object_domain(config: dict[str, Any], obj: str) -> tuple[tuple[float, float], tuple[float, float]]:
    profile = config.get("profile", {})
    lat_key, lon_key = PROFILE_RANGE_KEYS[obj]
    if lat_key not in profile or lon_key not in profile:
        raise KeyError(f"V10.2 config missing profile domain for {obj}: {lat_key}, {lon_key}")
    return tuple(profile[lat_key]), tuple(profile[lon_key])  # type: ignore[return-value]


def build_object_feature_tables(
    obj: str,
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_step: float,
) -> dict[str, Any]:
    clim = climatology_field(field)
    profile_raw = lat_profile_daily(clim)
    profile_binned, lat_bins = rebin_profile_to_step(profile_raw, lat, lat_step)
    profile_feature_names = [f"{obj}_profile_lat_{c:.1f}" for c in lat_bins]
    semantic_metrics = build_semantic_metric_series(obj, clim, lat, lon)

    # Detector-content reconstructed state vector: profile bins + semantic metrics.
    # Profile bins are included to expose where the detector-side change concentrates;
    # semantic metrics translate that content into object-readable terms.
    features = []
    feature_names = []
    feature_groups = []
    for j, name in enumerate(profile_feature_names):
        features.append(profile_binned[:, j])
        feature_names.append(name)
        feature_groups.append("profile_structure")
    for name, values in semantic_metrics.items():
        features.append(np.asarray(values, dtype=float))
        feature_names.append(name)
        feature_groups.append(SEMANTIC_GROUP_MAP.get(name, "semantic_metric"))
    feature_matrix = np.vstack(features).T if features else np.empty((clim.shape[0], 0))
    feature_z, feature_mean, feature_std = standardize_feature_matrix(feature_matrix)
    semantic_names = list(semantic_metrics.keys())
    semantic_matrix = np.vstack([semantic_metrics[n] for n in semantic_names]).T if semantic_names else np.empty((clim.shape[0], 0))
    semantic_z, semantic_mean, semantic_std = standardize_feature_matrix(semantic_matrix) if semantic_names else (semantic_matrix, np.array([]), np.array([]))
    return {
        "clim_field": clim,
        "profile": profile_binned,
        "lat_bins": lat_bins,
        "profile_feature_names": profile_feature_names,
        "semantic_metrics": semantic_metrics,
        "semantic_names": semantic_names,
        "semantic_matrix": semantic_matrix,
        "semantic_z": semantic_z,
        "semantic_mean": semantic_mean,
        "semantic_std": semantic_std,
        "feature_matrix": feature_matrix,
        "feature_z": feature_z,
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def max_location(field_delta: np.ndarray, lat: np.ndarray, lon: np.ndarray, positive: bool) -> tuple[float, float, float]:
    arr = np.asarray(field_delta, dtype=float)
    if not np.isfinite(arr).any():
        return np.nan, np.nan, np.nan
    if positive:
        idx = np.nanargmax(arr)
    else:
        idx = np.nanargmin(arr)
    i, j = np.unravel_index(idx, arr.shape)
    return float(arr[i, j]), float(lat[i]), float(lon[j])


def make_metric_panel(seq_df: pd.DataFrame, obj: str, out: Path) -> None:
    sdf = seq_df[seq_df["object"] == obj].copy()
    if sdf.empty:
        return
    x = sdf["main_peak_day"].to_numpy(dtype=float)
    y = sdf["top_detector_group_share"].to_numpy(dtype=float)
    labels = sdf["dominant_change_content"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, y, marker="o")
    for xi, yi, lab in zip(x, y, labels):
        ax.text(xi, yi, lab, rotation=35, ha="left", va="bottom", fontsize=8)
    ax.set_title(f"{obj}: detector-content top group share by object-native breakpoint")
    ax.set_xlabel("object-native break day")
    ax.set_ylabel("top detector feature-group contribution share")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def make_profile_delta_panel(profile_df: pd.DataFrame, obj: str, out: Path) -> None:
    sdf = profile_df[profile_df["object"] == obj].copy()
    if sdf.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for break_day, g in sdf.groupby("main_peak_day"):
        g = g.sort_values("lat")
        ax.plot(g["lat"].to_numpy(dtype=float), g["delta_profile"].to_numpy(dtype=float), label=f"day{int(break_day)}")
    ax.set_title(f"{obj}: reconstructed detector-flank lat-profile delta")
    ax.set_xlabel("latitude")
    ax.set_ylabel("right flank - left flank profile")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def build_root_log(settings: V10_8_A_Settings, output_root: Path, n_windows: int) -> str:
    return f"""# ROOT LOG: V10.8_a object-internal transition content audit

Created: {now_utc()}

## Task type
Engineering + research diagnostic patch.

## Version
`stage_partition/V10/v10.8`

## Entry
`stage_partition/V10/v10.8/scripts/run_object_internal_transition_content_v10_8_a.py`

## Output
`{output_root}`

## Input baseline
- V10.2 object-native breakpoint windows: `{settings.v10_2_windows_path()}`
- V10.2 config: `{settings.v10_2_config_path()}`
- Smoothed fields: `{settings.smoothed_fields_path()}`

## Scope
V10.8_a answers: for each P/V/H/Je/Jw object-native break day from V10.2, what object-internal detector-content changed most strongly?

Number of V10.2 object-native windows processed: {n_windows}

## Method boundary
- Does not rerun breakpoint detection.
- Does not use joint windows.
- Does not align to W045/W081/W113/W160.
- Does not infer synchrony, precursor, pathway, or causality.
- Reconstructs detector left/right flanks using V10.2 detector width and decomposes the reconstructed state-vector change.

## Main tables
- `tables/object_detector_native_change_decomposition_v10_8_a.csv`
- `tables/object_breakpoint_feature_group_contribution_v10_8_a.csv`
- `tables/object_semantic_metric_translation_v10_8_a.csv`
- `tables/object_breakpoint_content_interpretation_v10_8_a.csv`
- `tables/object_internal_breakpoint_content_sequence_v10_8_a.csv`
- `tables/object_breakpoint_lat_profile_delta_v10_8_a.csv`
- `tables/object_field_delta_summary_v10_8_a.csv`

## Forbidden interpretation
- Do not call any object breakpoint a precursor based on V10.8_a alone.
- Do not infer cross-object influence.
- Do not infer H35 -> W45, W33 -> W45, or any causal pathway.
- Do not treat derived content labels as direct detector outputs.
"""


def run_object_internal_transition_content_v10_8_a(settings: V10_8_A_Settings) -> dict[str, Any]:
    started = now_utc()
    output_root = settings.output_root()
    if settings.clean_output:
        clean_output_root(output_root)
    else:
        for sub in ("tables", "figures", "run_meta", "logs"):
            (output_root / sub).mkdir(parents=True, exist_ok=True)

    _log(settings, "[1/8] Load V10.2 object-native windows and config")
    windows = load_windows(settings.v10_2_windows_path())
    config = load_json(settings.v10_2_config_path())
    detector_width = int(settings.detector_width_override or config.get("detector", {}).get("width", 20))
    half_width = int(settings.flank_half_width_override or max(2, detector_width // 2))
    lat_step = float(config.get("profile", {}).get("lat_step_deg", 2.0))

    _log(settings, "[2/8] Load smoothed fields")
    npz = load_npz(settings)
    data = npz["data"]

    object_payloads: dict[str, dict[str, Any]] = {}
    input_audit_rows: list[dict[str, Any]] = []
    days_ref: np.ndarray | None = None

    _log(settings, "[3/8] Build reconstructed object state vectors")
    for obj in OBJECTS:
        key = first_key(data, FIELD_KEY_CANDIDATES[obj])
        if key is None:
            input_audit_rows.append({
                "object": obj,
                "source_field": None,
                "loaded": False,
                "notes": f"No matching field key found. Candidates={FIELD_KEY_CANDIDATES[obj]}",
            })
            warnings.warn(f"Skipping {obj}: no source field found", RuntimeWarning)
            continue
        field_raw, years, days = normalize_field_dims(data[key], data, npz["year_key"], npz["day_key"])
        if days_ref is None:
            days_ref = days
        elif not np.array_equal(days_ref, days):
            raise ValueError(f"Day axis mismatch for {obj}")
        lat_range, lon_range = get_object_domain(config, obj)
        sub, sub_lat, sub_lon = subset_domain(field_raw, npz["lat"], npz["lon"], lat_range, lon_range)
        payload = build_object_feature_tables(obj, sub, sub_lat, sub_lon, lat_step=lat_step)
        payload.update({"source_field": key, "years": years, "days": days, "lat": sub_lat, "lon": sub_lon})
        object_payloads[obj] = payload
        input_audit_rows.append({
            "object": obj,
            "source_field": key,
            "loaded": True,
            "lat_range": list(lat_range),
            "lon_range": list(lon_range),
            "n_lat": int(sub_lat.size),
            "n_lon": int(sub_lon.size),
            "n_days": int(days.size),
            "notes": "reconstructed object-state vector = lat profile bins + semantic object metrics",
        })

    if days_ref is None:
        raise RuntimeError("No object fields could be loaded; aborting V10.8_a")

    inventory_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    interp_rows: list[dict[str, Any]] = []
    seq_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    field_summary_rows: list[dict[str, Any]] = []

    _log(settings, "[4/8] Decompose detector-flank changes for each object-native break")
    for obj in OBJECTS:
        if obj not in object_payloads:
            continue
        payload = object_payloads[obj]
        days = payload["days"]
        lat = payload["lat"]
        lon = payload["lon"]
        fz = payload["feature_z"]
        fraw = payload["feature_matrix"]
        feature_names = payload["feature_names"]
        feature_groups = payload["feature_groups"]
        feature_std = payload["feature_std"]
        sz = payload["semantic_z"]
        sraw = payload["semantic_matrix"]
        semantic_names = payload["semantic_names"]
        clim_field = payload["clim_field"]
        prof = payload["profile"]
        lat_bins = payload["lat_bins"]

        obj_windows = windows[windows["object"] == obj].sort_values("main_peak_day")
        for seq_order, w in enumerate(obj_windows.itertuples(index=False), start=1):
            break_day = int(getattr(w, "main_peak_day"))
            flanks = detector_flank_windows(break_day, half_width, days)
            li = flanks["left_indices"]
            ri = flanks["right_indices"]
            source_support = getattr(w, "max_member_bootstrap_match_fraction", np.nan)

            inventory_rows.append({
                "object": obj,
                "window_id": getattr(w, "window_id"),
                "start_day": int(getattr(w, "start_day")),
                "end_day": int(getattr(w, "end_day")),
                "center_day": int(getattr(w, "center_day")),
                "main_peak_day": break_day,
                "member_candidate_ids": getattr(w, "member_candidate_ids", ""),
                "source_v10_2_support": source_support,
                "detector_width_from_v10_2": detector_width,
                "reconstructed_flank_half_width": half_width,
                "left_start": flanks["left_start"],
                "left_end": flanks["left_end"],
                "right_start": flanks["right_start"],
                "right_end": flanks["right_end"],
                "left_n_days": flanks["left_n_days"],
                "right_n_days": flanks["right_n_days"],
                "edge_limited_flag": flanks["edge_limited_flag"],
                "insufficient_flank_flag": flanks["insufficient_flank_flag"],
            })

            left_z = mean_over_indices(fz, li)
            right_z = mean_over_indices(fz, ri)
            delta_z = right_z - left_z
            left_raw = mean_over_indices(fraw, li)
            right_raw = mean_over_indices(fraw, ri)
            delta_raw = right_raw - left_raw
            contrib = delta_z ** 2
            total_contrib = float(np.nansum(contrib))
            total_norm = float(np.sqrt(total_contrib)) if np.isfinite(total_contrib) else np.nan
            share = contrib / total_contrib if total_contrib > 1e-12 else np.full_like(contrib, np.nan)
            order = np.argsort(-np.nan_to_num(contrib, nan=-np.inf))
            rank_by_idx = np.empty(len(contrib), dtype=int)
            for rank, idx in enumerate(order, start=1):
                rank_by_idx[idx] = rank

            for j, name in enumerate(feature_names):
                decomp_rows.append({
                    "object": obj,
                    "window_id": getattr(w, "window_id"),
                    "break_day": break_day,
                    "main_peak_day": break_day,
                    "left_start": flanks["left_start"],
                    "left_end": flanks["left_end"],
                    "right_start": flanks["right_start"],
                    "right_end": flanks["right_end"],
                    "feature_name": name,
                    "feature_group": feature_groups[j],
                    "left_mean": left_raw[j],
                    "right_mean": right_raw[j],
                    "delta": delta_raw[j],
                    "abs_delta": abs(delta_raw[j]) if np.isfinite(delta_raw[j]) else np.nan,
                    "metric_fullseason_std": feature_std[j],
                    "delta_z": delta_z[j],
                    "contribution_abs": contrib[j],
                    "contribution_share": share[j],
                    "rank_within_break": int(rank_by_idx[j]),
                    "source_v10_2_support": source_support,
                    "edge_limited_flag": flanks["edge_limited_flag"],
                })

            group_contrib: dict[str, float] = {}
            group_signed: dict[str, list[float]] = {}
            for g, dz, c in zip(feature_groups, delta_z, contrib):
                group_contrib[g] = group_contrib.get(g, 0.0) + (float(c) if np.isfinite(c) else 0.0)
                group_signed.setdefault(g, []).append(float(dz) if np.isfinite(dz) else np.nan)
            group_items = sorted(group_contrib.items(), key=lambda kv: kv[1], reverse=True)
            top_group = group_items[0][0] if group_items else "not_available"
            top_group_contrib = group_items[0][1] if group_items else np.nan
            top_group_share = top_group_contrib / total_contrib if total_contrib > 1e-12 else np.nan
            for rank, (g, c) in enumerate(group_items, start=1):
                group_rows.append({
                    "object": obj,
                    "window_id": getattr(w, "window_id"),
                    "break_day": break_day,
                    "main_peak_day": break_day,
                    "feature_group": g,
                    "group_contribution": c,
                    "group_contribution_share": c / total_contrib if total_contrib > 1e-12 else np.nan,
                    "group_signed_delta_mean_z": summarize_signed_delta(np.asarray(group_signed[g], dtype=float)),
                    "rank": rank,
                    "source_v10_2_support": source_support,
                    "edge_limited_flag": flanks["edge_limited_flag"],
                })

            # Semantic translation metrics, ranked separately.
            semantic_top_metric = None
            semantic_top_group = None
            semantic_top_delta_z = np.nan
            if semantic_names:
                left_sem_z = mean_over_indices(sz, li)
                right_sem_z = mean_over_indices(sz, ri)
                delta_sem_z = right_sem_z - left_sem_z
                left_sem_raw = mean_over_indices(sraw, li)
                right_sem_raw = mean_over_indices(sraw, ri)
                delta_sem_raw = right_sem_raw - left_sem_raw
                sem_order = np.argsort(-np.abs(np.nan_to_num(delta_sem_z, nan=0.0)))
                for rank, jj in enumerate(sem_order, start=1):
                    metric = semantic_names[jj]
                    group = SEMANTIC_GROUP_MAP.get(metric, "semantic_metric")
                    if rank == 1:
                        semantic_top_metric = metric
                        semantic_top_group = group
                        semantic_top_delta_z = delta_sem_z[jj]
                    semantic_rows.append({
                        "object": obj,
                        "window_id": getattr(w, "window_id"),
                        "break_day": break_day,
                        "main_peak_day": break_day,
                        "metric": metric,
                        "semantic_group": group,
                        "left_mean": left_sem_raw[jj],
                        "right_mean": right_sem_raw[jj],
                        "delta": delta_sem_raw[jj],
                        "delta_z": delta_sem_z[jj],
                        "direction_label": direction_label(metric, float(delta_sem_raw[jj])),
                        "rank_within_break": rank,
                        "source_v10_2_support": source_support,
                        "edge_limited_flag": flanks["edge_limited_flag"],
                    })

            detector_semantic_consistency = compare_detector_semantic(top_group, semantic_top_group)
            dominant_change_content = TRANSITION_TYPE_BY_GROUP.get(top_group, top_group)
            confidence = content_confidence_note(
                total_norm=total_norm,
                top_share=float(top_group_share) if np.isfinite(top_group_share) else np.nan,
                edge_limited=bool(flanks["edge_limited_flag"]),
                consistency=detector_semantic_consistency,
            )
            interp = {
                "object": obj,
                "window_id": getattr(w, "window_id"),
                "break_day": break_day,
                "main_peak_day": break_day,
                "top_detector_feature_group": top_group,
                "top_detector_group_contribution": top_group_contrib,
                "top_detector_group_share": top_group_share,
                "total_detector_delta_norm": total_norm,
                "top_semantic_metric": semantic_top_metric,
                "top_semantic_metric_group": semantic_top_group,
                "top_semantic_metric_delta_z": semantic_top_delta_z,
                "dominant_change_content": dominant_change_content,
                "secondary_change_content": TRANSITION_TYPE_BY_GROUP.get(group_items[1][0], group_items[1][0]) if len(group_items) > 1 else None,
                "detector_semantic_consistency": detector_semantic_consistency,
                "content_confidence_note": confidence,
                "source_v10_2_support": source_support,
                "edge_limited_flag": flanks["edge_limited_flag"],
                "forbidden_interpretation_note": "Do not infer precursor, causality, pathway, or cross-object influence from V10.8_a alone.",
            }
            interp_rows.append(interp)
            seq_rows.append({
                "object": obj,
                "sequence_order": seq_order,
                "main_peak_day": break_day,
                "window_start": int(getattr(w, "start_day")),
                "window_end": int(getattr(w, "end_day")),
                "dominant_change_content": dominant_change_content,
                "top_detector_feature_group": top_group,
                "top_detector_group_share": top_group_share,
                "top_semantic_metric": semantic_top_metric,
                "top_semantic_metric_delta_z": semantic_top_delta_z,
                "detector_semantic_consistency": detector_semantic_consistency,
                "source_v10_2_support": source_support,
                "content_confidence_note": confidence,
            })

            # Profile delta using the same reconstructed flanks.
            left_prof = mean_over_indices(prof, li)
            right_prof = mean_over_indices(prof, ri)
            delta_prof = right_prof - left_prof
            for c, lp, rp, dp in zip(lat_bins, left_prof, right_prof, delta_prof):
                profile_rows.append({
                    "object": obj,
                    "window_id": getattr(w, "window_id"),
                    "break_day": break_day,
                    "main_peak_day": break_day,
                    "lat": c,
                    "left_profile": lp,
                    "right_profile": rp,
                    "delta_profile": dp,
                    "abs_delta_profile": abs(dp) if np.isfinite(dp) else np.nan,
                    "source_v10_2_support": source_support,
                    "edge_limited_flag": flanks["edge_limited_flag"],
                })

            # Field delta summary on climatological field.
            left_field = safe_nanmean(clim_field[li], axis=0) if li.size else np.full(clim_field.shape[1:], np.nan)
            right_field = safe_nanmean(clim_field[ri], axis=0) if ri.size else np.full(clim_field.shape[1:], np.nan)
            fd = right_field - left_field
            max_pos_val, max_pos_lat, max_pos_lon = max_location(fd, lat, lon, positive=True)
            max_neg_val, max_neg_lat, max_neg_lon = max_location(fd, lat, lon, positive=False)
            field_summary_rows.append({
                "object": obj,
                "window_id": getattr(w, "window_id"),
                "break_day": break_day,
                "main_peak_day": break_day,
                "field_delta_norm": float(np.sqrt(np.nanmean(fd ** 2))) if np.isfinite(fd).any() else np.nan,
                "positive_area_fraction": float(np.nanmean(fd > 0)) if np.isfinite(fd).any() else np.nan,
                "negative_area_fraction": float(np.nanmean(fd < 0)) if np.isfinite(fd).any() else np.nan,
                "max_positive_value": max_pos_val,
                "max_positive_lat": max_pos_lat,
                "max_positive_lon": max_pos_lon,
                "max_negative_value": max_neg_val,
                "max_negative_lat": max_neg_lat,
                "max_negative_lon": max_neg_lon,
                "source_v10_2_support": source_support,
                "edge_limited_flag": flanks["edge_limited_flag"],
            })

    _log(settings, "[5/8] Save tables")
    tables = output_root / "tables"
    input_audit_df = pd.DataFrame(input_audit_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    decomp_df = pd.DataFrame(decomp_rows)
    group_df = pd.DataFrame(group_rows)
    semantic_df = pd.DataFrame(semantic_rows)
    interp_df = pd.DataFrame(interp_rows)
    seq_df = pd.DataFrame(seq_rows)
    profile_df = pd.DataFrame(profile_rows)
    field_df = pd.DataFrame(field_summary_rows)

    write_dataframe(input_audit_df, tables / "input_field_audit_v10_8_a.csv")
    write_dataframe(inventory_df, tables / "object_window_inventory_v10_8_a.csv")
    write_dataframe(decomp_df, tables / "object_detector_native_change_decomposition_v10_8_a.csv")
    write_dataframe(group_df, tables / "object_breakpoint_feature_group_contribution_v10_8_a.csv")
    write_dataframe(semantic_df, tables / "object_semantic_metric_translation_v10_8_a.csv")
    write_dataframe(interp_df, tables / "object_breakpoint_content_interpretation_v10_8_a.csv")
    write_dataframe(seq_df, tables / "object_internal_breakpoint_content_sequence_v10_8_a.csv")
    write_dataframe(profile_df, tables / "object_breakpoint_lat_profile_delta_v10_8_a.csv")
    write_dataframe(field_df, tables / "object_field_delta_summary_v10_8_a.csv")

    _log(settings, "[6/8] Save figures")
    if settings.make_figures:
        figs = output_root / "figures"
        for obj in OBJECTS:
            make_metric_panel(seq_df, obj, figs / f"{obj}_detector_feature_group_contribution_v10_8_a.png")
            make_profile_delta_panel(profile_df, obj, figs / f"{obj}_breakpoint_profile_delta_panel_v10_8_a.png")

    _log(settings, "[7/8] Write run_meta and summary")
    run_meta = {
        "version": settings.version,
        "created_utc": now_utc(),
        "started_utc": started,
        "task": "object-internal transition content audit",
        "method_boundary": {
            "does_not_rerun_breakpoint_detection": True,
            "does_not_use_joint_windows": True,
            "does_not_align_to_accepted_windows": True,
            "does_not_do_cross_object_interpretation": True,
            "does_not_do_precursor_analysis": True,
            "does_not_do_causal_inference": True,
            "detector_native_reconstruction_note": "Uses V10.2 detector width to reconstruct left/right flanks around each object-native break day; this decomposes reconstructed detector-content changes and does not claim exact ruptures internal score attribution.",
        },
        "settings": settings.to_dict(),
        "input_paths": {
            "v10_2_windows": str(settings.v10_2_windows_path()),
            "v10_2_config": str(settings.v10_2_config_path()),
            "smoothed_fields": str(settings.smoothed_fields_path()),
        },
        "detector_width_from_v10_2": detector_width,
        "reconstructed_flank_half_width": half_width,
        "lat_step_deg": lat_step,
        "n_object_windows": int(len(inventory_df)),
        "objects_loaded": sorted(object_payloads.keys()),
        "outputs": {
            "tables": sorted([p.name for p in tables.glob("*.csv")]),
            "figures": sorted([p.name for p in (output_root / "figures").glob("*.png")]) if (output_root / "figures").exists() else [],
        },
    }
    write_json(run_meta, output_root / "run_meta" / "run_meta_v10_8_a.json")

    summary_lines = [
        "# V10.8_a object-internal transition content audit",
        "",
        "## Scope",
        "V10.8_a uses V10.2 object-native breakpoints as fixed input and decomposes each object's reconstructed detector-flank change.",
        "",
        "## Boundary",
        "This run does not use joint windows, accepted windows, cross-object alignment, precursor inference, or causal/pathway interpretation.",
        "",
        "## Processed objects",
        ", ".join(sorted(object_payloads.keys())),
        "",
        "## Main output",
        "The primary table is `object_breakpoint_content_interpretation_v10_8_a.csv`; direct feature-level decomposition is in `object_detector_native_change_decomposition_v10_8_a.csv`.",
    ]
    (output_root / "summary_v10_8_a.md").write_text("\n".join(summary_lines), encoding="utf-8")

    _log(settings, "[8/8] Write root log")
    if settings.write_root_log:
        root_log_text = build_root_log(settings, output_root, n_windows=len(inventory_df))
        settings.root_log_path().write_text(root_log_text, encoding="utf-8")
        append_dir = settings.project_root / "root_log_append"
        append_dir.mkdir(parents=True, exist_ok=True)
        (append_dir / "ROOT_LOG_03_VERSION_STATUS_AND_EXECUTION_CHAIN__V10_8_A_APPEND.md").write_text(root_log_text, encoding="utf-8")
        (append_dir / "ROOT_LOG_05_PENDING_TASKS_AND_FORBIDDEN_INTERPRETATIONS__V10_8_A_APPEND.md").write_text(
            "# V10.8_a pending tasks and forbidden interpretations\n\n"
            "## Pending checks\n"
            "- Inspect `detector_semantic_consistency` before writing any object-content conclusion.\n"
            "- Treat `edge_limited` early/late breakpoints as uncertain, not invalid.\n"
            "- Review feature-group contribution tables before using derived dominant labels.\n\n"
            "## Forbidden interpretations\n"
            "- Do not infer precursor/cause/pathway/cross-object influence.\n"
            "- Do not align results to joint windows in V10.8_a.\n"
            "- Do not treat derived labels as direct detector outputs.\n",
            encoding="utf-8",
        )

    return run_meta
