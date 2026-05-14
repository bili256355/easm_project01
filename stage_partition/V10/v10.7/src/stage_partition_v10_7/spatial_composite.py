from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .event_content_config import EventContentConfig, EventWindow
from .event_content_io import SpatialFieldData


def _mask_between(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    a, b = sorted((float(lo), float(hi)))
    return (arr >= a) & (arr <= b)


def _domain_subset(
    spatial: SpatialFieldData,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    domain_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Subset spatial field by domain and return year x day x lat x lon.

    The user's latitude arrays may be high-to-low. We always sort selected lat/lon
    ascending after masking so downstream pattern metrics and plots are stable.
    """
    lat_mask = _mask_between(spatial.lat, *lat_range)
    lon_mask = _mask_between(spatial.lon, *lon_range)
    meta = {
        "domain_label": domain_label,
        "domain_lat_range": list(lat_range),
        "domain_lon_range": list(lon_range),
        "n_lat_selected": int(lat_mask.sum()),
        "n_lon_selected": int(lon_mask.sum()),
        "lat_sort_applied": False,
        "lon_sort_applied": False,
    }
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(
            f"No spatial grid points selected for domain={domain_label}, "
            f"lat_range={lat_range}, lon_range={lon_range}."
        )
    sub = spatial.field[:, :, lat_mask, :][:, :, :, lon_mask]
    lat_sub = np.asarray(spatial.lat[lat_mask], dtype=float)
    lon_sub = np.asarray(spatial.lon[lon_mask], dtype=float)

    lat_order = np.argsort(lat_sub)
    if not np.all(lat_order == np.arange(len(lat_sub))):
        sub = sub[:, :, lat_order, :]
        lat_sub = lat_sub[lat_order]
        meta["lat_sort_applied"] = True
    lon_order = np.argsort(lon_sub)
    if not np.all(lon_order == np.arange(len(lon_sub))):
        sub = sub[:, :, :, lon_order]
        lon_sub = lon_sub[lon_order]
        meta["lon_sort_applied"] = True
    return sub, lat_sub, lon_sub, meta


def _event_diff(field: np.ndarray, ev: EventWindow) -> np.ndarray:
    # field: year x day x lat x lon
    nday = field.shape[1]
    pre_days = [d for d in range(ev.pre_min, ev.pre_max + 1) if 0 <= d < nday]
    post_days = [d for d in range(ev.post_min, ev.post_max + 1) if 0 <= d < nday]
    if not pre_days or not post_days:
        return np.full((field.shape[0], field.shape[2], field.shape[3]), np.nan)
    pre = np.nanmean(field[:, pre_days, :, :], axis=1)
    post = np.nanmean(field[:, post_days, :, :], axis=1)
    return post - pre


def _dominant_point(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, positive: bool) -> tuple[float, float, float]:
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.nan, np.nan, np.nan
    idx = np.nanargmax(arr) if positive else np.nanargmin(arr)
    iy, ix = np.unravel_index(idx, arr.shape)
    return float(lat[iy]), float(lon[ix]), float(arr[iy, ix])


def _compute_for_domain(
    spatial: SpatialFieldData | None,
    cfg: EventContentConfig,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    domain_label: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    if spatial is None:
        return pd.DataFrame(), {}, {"spatial_composite_status": f"skipped_missing_spatial_field_{domain_label}"}
    sub, lat, lon, domain_meta = _domain_subset(spatial, lat_range, lon_range, domain_label)
    rows: list[dict[str, Any]] = []
    diff_maps: dict[str, np.ndarray] = {}
    yearly_maps: dict[str, np.ndarray] = {}
    for ev in cfg.event_windows:
        diff_year = _event_diff(sub, ev)
        yearly_maps[ev.event_id] = diff_year
        clim = np.nanmean(diff_year, axis=0)
        diff_maps[ev.event_id] = clim
        pos_lat, pos_lon, pos_val = _dominant_point(clim, lat, lon, True)
        neg_lat, neg_lon, neg_val = _dominant_point(clim, lat, lon, False)
        rows.append({
            "domain_label": domain_label,
            "event_id": ev.event_id,
            "field_key": spatial.field_key,
            "pre_days": f"{ev.pre_min}-{ev.pre_max}",
            "post_days": f"{ev.post_min}-{ev.post_max}",
            "domain_lon_min": float(min(lon_range)),
            "domain_lon_max": float(max(lon_range)),
            "domain_lat_min": float(min(lat_range)),
            "domain_lat_max": float(max(lat_range)),
            "n_lat_selected": int(len(lat)),
            "n_lon_selected": int(len(lon)),
            "field_diff_mean": float(np.nanmean(clim)) if np.isfinite(clim).any() else np.nan,
            "field_diff_abs_mean": float(np.nanmean(np.abs(clim))) if np.isfinite(clim).any() else np.nan,
            "field_diff_rms": float(np.sqrt(np.nanmean(clim ** 2))) if np.isfinite(clim).any() else np.nan,
            "field_diff_max": float(np.nanmax(clim)) if np.isfinite(clim).any() else np.nan,
            "field_diff_min": float(np.nanmin(clim)) if np.isfinite(clim).any() else np.nan,
            "dominant_positive_lat": pos_lat,
            "dominant_positive_lon": pos_lon,
            "dominant_positive_value": pos_val,
            "dominant_negative_lat": neg_lat,
            "dominant_negative_lon": neg_lon,
            "dominant_negative_value": neg_val,
        })
    meta = {
        "spatial_composite_status": f"computed_{domain_label}",
        "lat_values": lat.tolist(),
        "lon_values": lon.tolist(),
        **domain_meta,
        "yearly_maps": yearly_maps,
    }
    return pd.DataFrame(rows), diff_maps, meta


def compute_spatial_composites(
    spatial: SpatialFieldData | None,
    cfg: EventContentConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Full-domain composite retained as context/background reference."""
    return _compute_for_domain(spatial, cfg, cfg.lat_range, cfg.lon_range, "full_domain_context")


def compute_spatial_composites_object_domain(
    spatial: SpatialFieldData | None,
    cfg: EventContentConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """H-object-domain composite used as primary H-content evidence."""
    return _compute_for_domain(spatial, cfg, cfg.object_lat_range, cfg.object_lon_range, "h_object_domain")


def flatten_finite(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    return aa[mask], bb[mask]
