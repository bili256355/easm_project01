from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from .config import HProfileConfig, StateBuilderConfig
from .utils import safe_nanmean


@dataclass
class HProfile:
    name: str
    raw_cube: np.ndarray  # year x day x lat_feature
    lat_grid: np.ndarray
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]
    empty_slice_mask: np.ndarray


def _mask_between(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    a, b = sorted((float(lo), float(hi)))
    return (arr >= a) & (arr <= b)


def _ascending_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(x, dtype=float))
    return np.asarray(x, dtype=float)[order], np.asarray(y, dtype=float)[order]


def _interp_profile_to_grid(profile: np.ndarray, src_lats: np.ndarray, dst_lats: np.ndarray) -> np.ndarray:
    # profile: year x day x lat_src
    out = np.full((profile.shape[0], profile.shape[1], len(dst_lats)), np.nan, dtype=float)
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            row = profile[i, j, :]
            valid = np.isfinite(row) & np.isfinite(src_lats)
            if valid.sum() < 2:
                continue
            src, vals = _ascending_pair(src_lats[valid], row[valid])
            out[i, j, :] = np.interp(dst_lats, src, vals, left=np.nan, right=np.nan)
    return out


def build_h_profile(smoothed: dict[str, np.ndarray], cfg: HProfileConfig) -> HProfile:
    required = [cfg.field_key, cfg.lat_key, cfg.lon_key]
    missing = [k for k in required if k not in smoothed]
    if missing:
        raise KeyError(f"Missing keys in smoothed_fields.npz for H profile: {missing}; available={sorted(smoothed.keys())}")

    field = np.asarray(smoothed[cfg.field_key], dtype=float)
    lat = np.asarray(smoothed[cfg.lat_key], dtype=float)
    lon = np.asarray(smoothed[cfg.lon_key], dtype=float)
    if field.ndim != 4:
        raise ValueError(f"Expected {cfg.field_key} shape = year x day x lat x lon; got {field.shape}")

    lat_mask = _mask_between(lat, *cfg.lat_range)
    lon_mask = _mask_between(lon, *cfg.lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points in requested H range {cfg.lat_range}")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points in requested H range {cfg.lon_range}")

    src_lats = lat[lat_mask]
    subset = field[:, :, lat_mask, :][:, :, :, lon_mask]
    prof, valid_count = safe_nanmean(subset, axis=-1, return_valid_count=True)
    empty_slice_mask = valid_count == 0

    lo, hi = sorted((float(cfg.lat_range[0]), float(cfg.lat_range[1])))
    dst_lats = np.arange(lo, hi + 1e-9, float(cfg.lat_step_deg))
    prof_interp = _interp_profile_to_grid(prof, src_lats, dst_lats)
    empty_interp = _interp_profile_to_grid(empty_slice_mask.astype(float), src_lats, dst_lats)
    empty_interp = np.where(np.isfinite(empty_interp), empty_interp >= 0.5, True)

    return HProfile(
        name=cfg.object_name,
        raw_cube=prof_interp,
        lat_grid=dst_lats,
        lon_range=cfg.lon_range,
        lat_range=cfg.lat_range,
        empty_slice_mask=empty_interp,
    )


def summarize_h_profile_validity(profile: HProfile) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cube = profile.raw_cube
    for idx, latv in enumerate(profile.lat_grid):
        col = cube[:, :, idx]
        finite = np.isfinite(col)
        rows.append(
            {
                "object_name": profile.name,
                "lat_feature_index": int(idx),
                "lat_value": float(latv),
                "nan_fraction": float((~finite).mean()),
                "finite_fraction": float(finite.mean()),
                "all_nan_any_day": bool(np.any(np.all(~finite, axis=0))),
                "all_nan_all_days": bool(np.all(~finite)),
            }
        )
    return pd.DataFrame(rows)


def _zscore_along_day(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, _ = safe_nanmean(x, axis=0, return_valid_count=True)
    centered = x - mean[None, :]
    var, _ = safe_nanmean(np.square(centered), axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    z = centered / std[None, :]
    return z, mean, std


def build_h_state_matrix(profile: HProfile, cfg: StateBuilderConfig) -> dict[str, Any]:
    seasonal, _ = safe_nanmean(profile.raw_cube, axis=0, return_valid_count=True)  # day x lat_feature
    raw = np.asarray(seasonal, dtype=float)
    z, feat_mean, feat_std = _zscore_along_day(raw)
    state = z.copy() if cfg.standardize else raw.copy()

    applied_weight = 1.0
    if cfg.block_equal_contribution:
        # Single-object block contribution: preserves timing; score scale changes by a constant.
        applied_weight = 1.0 / np.sqrt(max(1, state.shape[1]))
        state = state * applied_weight

    valid_day_mask = np.all(np.isfinite(state), axis=1) if cfg.trim_invalid_days else np.ones(raw.shape[0], dtype=bool)
    valid_day_index = np.where(valid_day_mask)[0].astype(int)
    feature_table = pd.DataFrame(
        {
            "feature_index": np.arange(len(profile.lat_grid), dtype=int),
            "object_name": profile.name,
            "lat_value": profile.lat_grid.astype(float),
            "raw_mean_day": feat_mean.astype(float),
            "raw_std_day": feat_std.astype(float),
            "applied_weight": float(applied_weight),
        }
    )
    empty_rows = []
    for j in range(raw.shape[1]):
        finite = np.isfinite(state[:, j])
        if not finite.all():
            empty_rows.append(
                {
                    "feature_index": int(j),
                    "object_name": profile.name,
                    "lat_value": float(profile.lat_grid[j]),
                    "n_invalid_days": int((~finite).sum()),
                }
            )

    return {
        "raw_matrix": raw,
        "state_matrix": state[valid_day_mask, :],
        "valid_day_mask": valid_day_mask,
        "valid_day_index": valid_day_index,
        "feature_table": feature_table,
        "state_empty_feature_audit": pd.DataFrame(empty_rows),
        "state_vector_meta": {
            "object_name": profile.name,
            "n_days": int(raw.shape[0]),
            "n_features": int(raw.shape[1]),
            "n_valid_days": int(valid_day_index.size),
            "n_invalid_days": int(raw.shape[0] - valid_day_index.size),
            "valid_day_index": valid_day_index.tolist(),
            "invalid_day_index": np.where(~valid_day_mask)[0].astype(int).tolist(),
            "state_expression_name": "H_raw_smoothed_zscore_single_block_equal",
            "applied_weight": float(applied_weight),
        },
    }
