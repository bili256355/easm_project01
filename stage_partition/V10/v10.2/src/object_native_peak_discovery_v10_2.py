from __future__ import annotations

"""
V10.1 joint main-window reproduction.

This is a self-contained semantic rewrite of the historical V6 -> V6_1
joint-object discovery chain.  It intentionally does not import V6/V6_1/V7/V9
Python modules.  Historical CSV files are read only as regression references.

Scope:
    joint objects -> free-season local peak candidates -> bootstrap support
    -> candidate support bands -> derived windows -> strict accepted lineage.

Out of scope:
    object-native discovery, H/Jw sensitivity, pair order, physical interpretation,
    or any re-decision of the accepted-window set.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, peak_prominences
except Exception:  # pragma: no cover - user environment normally has scipy.
    find_peaks = None
    peak_prominences = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FoundationInputConfig:
    project_root: Path = Path(r"D:\easm_project01")
    foundation_layer: str = "foundation"
    foundation_version: str = "V1"
    preprocess_output_tag: str = "baseline_a"

    def smoothed_fields_path(self) -> Path:
        env = os.environ.get("V10_1_SMOOTHED_FIELDS")
        if env:
            return Path(env)
        return (
            self.project_root
            / self.foundation_layer
            / self.foundation_version
            / "outputs"
            / self.preprocess_output_tag
            / "preprocess"
            / "smoothed_fields.npz"
        )


@dataclass
class ProfileGridConfig:
    lat_step_deg: float = 2.0
    p_lon_range: tuple[float, float] = (105.0, 125.0)
    p_lat_range: tuple[float, float] = (15.0, 39.0)
    v_lon_range: tuple[float, float] = (105.0, 125.0)
    v_lat_range: tuple[float, float] = (10.0, 30.0)
    h_lon_range: tuple[float, float] = (110.0, 140.0)
    h_lat_range: tuple[float, float] = (15.0, 35.0)
    je_lon_range: tuple[float, float] = (120.0, 150.0)
    je_lat_range: tuple[float, float] = (25.0, 45.0)
    jw_lon_range: tuple[float, float] = (80.0, 110.0)
    jw_lat_range: tuple[float, float] = (25.0, 45.0)


@dataclass
class StateBuilderConfig:
    standardize: bool = True
    block_equal_contribution: bool = True
    trim_invalid_days: bool = True


@dataclass
class DetectorConfig:
    width: int = 20
    model: str = "l2"
    min_size: int = 2
    jump: int = 1
    selection_mode: str = "pen"
    fixed_n_bkps: int | None = None
    pen: float | None = 4.0
    epsilon: float | None = None
    local_peak_min_distance_days: int = 3
    nearest_peak_search_radius_days: int = 10


@dataclass
class BootstrapConfig:
    n_bootstrap: int = 1000
    random_seed: int = 42
    progress: bool = True
    strict_match_max_abs_offset_days: int = 2
    match_max_abs_offset_days: int = 5
    near_max_abs_offset_days: int = 8

    def resolved_n_bootstrap(self) -> int:
        if os.environ.get("V10_2_DEBUG_N_BOOTSTRAP"):
            return int(os.environ["V10_2_DEBUG_N_BOOTSTRAP"])
        if os.environ.get("V10_2_N_BOOTSTRAP"):
            return int(os.environ["V10_2_N_BOOTSTRAP"])
        if os.environ.get("V10_1_DEBUG_N_BOOTSTRAP"):
            return int(os.environ["V10_1_DEBUG_N_BOOTSTRAP"])
        if os.environ.get("V10_1_N_BOOTSTRAP"):
            return int(os.environ["V10_1_N_BOOTSTRAP"])
        return int(self.n_bootstrap)


@dataclass
class BandConfig:
    min_band_half_width_days: int = 2
    max_band_half_width_days: int = 10
    peak_floor_quantile: float = 0.35
    prominence_ratio_threshold: float = 0.50
    respect_candidate_boundaries: bool = True
    truncate_at_intervening_candidate: bool = True
    truncate_at_local_valley: bool = True
    allow_band_merge: bool = True
    merge_gap_days: int = 1
    close_neighbor_exemption_days: int = 4
    protect_significant_peaks_from_merge: bool = True
    significant_peak_threshold: float = 0.95


@dataclass
class UncertaintyConfig:
    interval_match_types: tuple[str, ...] = ("strict", "matched", "near")
    emit_width80: bool = True
    emit_width95: bool = True


@dataclass
class ReferenceConfig:
    source_v6_output_tag: str = "mainline_v6_a"
    source_v6_1_output_tag: str = "mainline_v6_1_a"
    strict_accepted_windows: tuple[tuple[str, int, int, int], ...] = (
        ("W045", 45, 40, 48),
        ("W081", 81, 75, 87),
        ("W113", 113, 108, 118),
        ("W160", 160, 155, 165),
    )


@dataclass
class Settings:
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    state: StateBuilderConfig = field(default_factory=StateBuilderConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    band: BandConfig = field(default_factory=BandConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    output_tag: str = "joint_main_window_reproduce_v10_1"

    def to_dict(self) -> dict[str, Any]:
        def convert(x: Any) -> Any:
            if isinstance(x, Path):
                return str(x)
            if isinstance(x, tuple):
                return [convert(v) for v in x]
            if isinstance(x, list):
                return [convert(v) for v in x]
            if isinstance(x, dict):
                return {str(k): convert(v) for k, v in x.items()}
            return x
        return convert(asdict(self))


# =============================================================================
# Generic IO and utilities
# =============================================================================

OBJECT_ORDER = ["P", "V", "H", "Je", "Jw"]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def clean_output_dirs(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for sub in ["joint_point_layer", "joint_window_layer", "lineage", "audit", "figures"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def load_smoothed_fields(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields.npz: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def safe_nanmean(a: np.ndarray, axis=None, return_valid_count: bool = False):
    arr = np.asarray(a, dtype=float)
    valid = np.isfinite(arr)
    valid_count = valid.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = np.nansum(arr, axis=axis)
        mean = total / valid_count
    mean = np.where(valid_count > 0, mean, np.nan)
    if return_valid_count:
        return mean, valid_count
    return mean


def day_index_to_month_day(day_index: int) -> str:
    # Apr-Sep, day 0 = Apr 1.  Stable and dependency-free.
    mdays = [(4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]
    d = int(day_index)
    for month, nday in mdays:
        if d < nday:
            return f"{month:02d}-{d + 1:02d}"
        d -= nday
    return f"overflow+{d}"


# =============================================================================
# Profile and state construction
# =============================================================================

@dataclass
class PointLayerProfile:
    name: str
    raw_cube: np.ndarray  # year x day x lat_feature
    lat_grid: np.ndarray
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]
    empty_slice_mask: np.ndarray


def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(lower, upper), max(lower, upper)
    return (arr >= lo) & (arr <= hi)


def _ascending_pair(lat_vals: np.ndarray, vals: np.ndarray):
    order = np.argsort(lat_vals)
    return lat_vals[order], vals[..., order]


def _interp_profile_to_grid(profile: np.ndarray, src_lats: np.ndarray, dst_lats: np.ndarray) -> np.ndarray:
    out = np.full((profile.shape[0], profile.shape[1], dst_lats.size), np.nan, dtype=float)
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            row = profile[i, j, :]
            valid = np.isfinite(row) & np.isfinite(src_lats)
            if valid.sum() < 2:
                continue
            src = src_lats[valid]
            vals = row[valid]
            src, vals = _ascending_pair(src, vals)
            out[i, j, :] = np.interp(dst_lats, src, vals, left=np.nan, right=np.nan)
    return out


def _build_profile_from_field(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    lat_step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_mask = _mask_between(lat, *lat_range)
    lon_mask = _mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points in requested range {lat_range}")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points in requested range {lon_range}")
    src_lats = lat[lat_mask]
    subset = field[:, :, lat_mask, :][:, :, :, lon_mask]
    prof, valid_count = safe_nanmean(subset, axis=-1, return_valid_count=True)
    empty_slice_mask = valid_count == 0
    lo, hi = min(lat_range), max(lat_range)
    dst_lats = np.arange(lo, hi + 1e-9, lat_step_deg)
    prof_interp = _interp_profile_to_grid(prof, src_lats, dst_lats)
    empty_interp = _interp_profile_to_grid(empty_slice_mask.astype(float), src_lats, dst_lats)
    empty_interp = np.where(np.isfinite(empty_interp), empty_interp >= 0.5, True)
    return prof_interp, dst_lats, empty_interp


def build_profiles(smoothed: dict[str, np.ndarray], cfg: ProfileGridConfig) -> dict[str, PointLayerProfile]:
    lat = smoothed["lat"]
    lon = smoothed["lon"]
    specs = {
        "P": ("precip_smoothed", cfg.p_lon_range, cfg.p_lat_range),
        "V": ("v850_smoothed", cfg.v_lon_range, cfg.v_lat_range),
        "H": ("z500_smoothed", cfg.h_lon_range, cfg.h_lat_range),
        "Je": ("u200_smoothed", cfg.je_lon_range, cfg.je_lat_range),
        "Jw": ("u200_smoothed", cfg.jw_lon_range, cfg.jw_lat_range),
    }
    profs: dict[str, PointLayerProfile] = {}
    for name, (field_key, lon_range, lat_range) in specs.items():
        cube, lat_grid, empty_mask = _build_profile_from_field(
            smoothed[field_key], lat, lon, lon_range, lat_range, cfg.lat_step_deg
        )
        profs[name] = PointLayerProfile(
            name=name,
            raw_cube=cube,
            lat_grid=lat_grid,
            lon_range=lon_range,
            lat_range=lat_range,
            empty_slice_mask=empty_mask,
        )
    return profs


def summarize_profile_validity(profiles: dict[str, PointLayerProfile]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, obj in profiles.items():
        cube = obj.raw_cube
        for idx, latv in enumerate(obj.lat_grid):
            col = cube[:, :, idx]
            finite = np.isfinite(col)
            rows.append(
                {
                    "object_name": name,
                    "lat_feature_index": int(idx),
                    "lat_value": float(latv),
                    "nan_fraction": float((~finite).mean()),
                    "finite_fraction": float(finite.mean()),
                    "all_nan_any_day": bool(np.any(np.all(~finite, axis=0))),
                    "all_nan_all_days": bool(np.all(~finite)),
                }
            )
    return pd.DataFrame(rows)


def summarize_profile_empty_slices(profiles: dict[str, PointLayerProfile]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, obj in profiles.items():
        idx = np.argwhere(obj.empty_slice_mask)
        for year_idx, day_idx, lat_idx in idx:
            rows.append(
                {
                    "object_name": name,
                    "year_index": int(year_idx),
                    "day_index": int(day_idx),
                    "lat_feature_index": int(lat_idx),
                    "lat_value": float(obj.lat_grid[lat_idx]),
                    "reason": "all_nan_over_lon",
                }
            )
    return pd.DataFrame(rows, columns=["object_name", "year_index", "day_index", "lat_feature_index", "lat_value", "reason"])


def _zscore_along_day(x: np.ndarray):
    mean, _ = safe_nanmean(x, axis=0, return_valid_count=True)
    centered = x - mean[None, :]
    sq = np.square(centered)
    var, _ = safe_nanmean(sq, axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    z = centered / std[None, :]
    return z, mean, std


def _apply_equal_block_contribution(x: np.ndarray, block_slices: dict[str, slice]):
    y = x.copy()
    rows: list[dict[str, Any]] = []
    for name, slc in block_slices.items():
        width = slc.stop - slc.start
        factor = 1.0 / np.sqrt(width)
        before = float(np.sqrt(np.nanmean(np.square(y[:, slc]))))
        y[:, slc] *= factor
        after = float(np.sqrt(np.nanmean(np.square(y[:, slc]))))
        rows.append(
            {
                "object_name": name,
                "block_size": int(width),
                "applied_weight": float(factor),
                "weighted_rms_before": before,
                "weighted_rms_after": after,
            }
        )
    return y, pd.DataFrame(rows)


def _build_seasonal(profiles: dict[str, PointLayerProfile], year_indices: np.ndarray | None = None):
    seasonal_blocks: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    start = 0
    for name in OBJECT_ORDER:
        cube = profiles[name].raw_cube if year_indices is None else profiles[name].raw_cube[np.asarray(year_indices, dtype=int), :, :]
        seasonal, _ = safe_nanmean(cube, axis=0, return_valid_count=True)
        seasonal_blocks.append(seasonal)
        width = seasonal.shape[1]
        for j, latv in enumerate(profiles[name].lat_grid):
            feature_rows.append(
                {
                    "feature_index": start + j,
                    "object_name": name,
                    "object_width": width,
                    "feature_in_object": j,
                    "lat_value": float(latv),
                }
            )
        start += width
    return seasonal_blocks, feature_rows


def _build_state_from_seasonal_blocks(seasonal_blocks: list[np.ndarray], feature_rows: list[dict[str, Any]], cfg: StateBuilderConfig):
    raw = np.concatenate(seasonal_blocks, axis=1)
    z, feat_mean, feat_std = _zscore_along_day(raw)
    scaled = z.copy()
    block_slices: dict[str, slice] = {}
    raw_rows = []
    std_rows = []
    empty_rows = []
    start = 0
    for name, block in zip(OBJECT_ORDER, seasonal_blocks):
        width = block.shape[1]
        slc = slice(start, start + width)
        block_slices[name] = slc
        raw_rows.append({"object_name": name, "block_size": int(width), "rms": float(np.sqrt(np.nanmean(np.square(raw[:, slc]))))})
        std_rows.append({"object_name": name, "block_size": int(width), "rms": float(np.sqrt(np.nanmean(np.square(z[:, slc]))))})
        start += width
    if cfg.block_equal_contribution:
        scaled, weights = _apply_equal_block_contribution(scaled, block_slices)
    else:
        weights = pd.DataFrame(
            {"object_name": list(block_slices.keys()), "block_size": [s.stop - s.start for s in block_slices.values()], "applied_weight": 1.0}
        )
    valid_day_mask = np.all(np.isfinite(scaled), axis=1) if cfg.trim_invalid_days else np.ones(raw.shape[0], dtype=bool)
    valid_day_index = np.where(valid_day_mask)[0].astype(int)
    for j in range(raw.shape[1]):
        finite = np.isfinite(scaled[:, j])
        if not finite.all():
            feature = feature_rows[j]
            empty_rows.append(
                {
                    "feature_index": int(j),
                    "object_name": feature["object_name"],
                    "lat_value": float(feature["lat_value"]),
                    "n_invalid_days": int((~finite).sum()),
                }
            )
    meta = {
        "n_days": int(raw.shape[0]),
        "n_features": int(raw.shape[1]),
        "n_valid_days": int(valid_day_index.size),
        "n_invalid_days": int(raw.shape[0] - valid_day_index.size),
        "valid_day_index": valid_day_index.tolist(),
        "invalid_day_index": np.where(~valid_day_mask)[0].astype(int).tolist(),
        "block_slices": {k: [int(v.start), int(v.stop)] for k, v in block_slices.items()},
        "state_expression_name": "raw_smoothed_zscore_block_equal",
    }
    scale_rows = []
    for item in feature_rows:
        idx = int(item["feature_index"])
        scale_rows.append(
            {
                "feature_index": idx,
                "object_name": item["object_name"],
                "lat_value": item["lat_value"],
                "raw_mean_day": float(feat_mean[idx]),
                "raw_std_day": float(feat_std[idx]),
                "z_mean_day": float(np.nanmean(z[:, idx])) if np.isfinite(z[:, idx]).any() else np.nan,
                "z_std_day": float(np.nanstd(z[:, idx])) if np.isfinite(z[:, idx]).any() else np.nan,
            }
        )
    return {
        "raw_matrix": raw,
        "state_matrix": scaled,
        "valid_day_mask": valid_day_mask,
        "valid_day_index": valid_day_index,
        "state_vector_meta": meta,
        "feature_table": pd.DataFrame(feature_rows)[["feature_index", "object_name", "lat_value"]].copy(),
        "state_feature_scale_before_after": pd.DataFrame(scale_rows),
        "state_block_energy_before_after": {"raw": pd.DataFrame(raw_rows), "standardized": pd.DataFrame(std_rows), "weights": weights},
        "state_empty_feature_audit": pd.DataFrame(empty_rows),
    }


def build_state_matrix(profiles: dict[str, PointLayerProfile], cfg: StateBuilderConfig) -> dict[str, Any]:
    seasonal_blocks, feature_rows = _build_seasonal(profiles, None)
    return _build_state_from_seasonal_blocks(seasonal_blocks, feature_rows, cfg)


def build_resampled_state_matrix(profiles: dict[str, PointLayerProfile], sampled_year_indices: np.ndarray, cfg: StateBuilderConfig) -> dict[str, Any]:
    seasonal_blocks, feature_rows = _build_seasonal(profiles, np.asarray(sampled_year_indices, dtype=int))
    out = _build_state_from_seasonal_blocks(seasonal_blocks, feature_rows, cfg)
    out["sampled_year_indices"] = np.asarray(sampled_year_indices, dtype=int)
    return out


# =============================================================================
# Detector and candidate registry
# =============================================================================

def _import_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except Exception as e:  # pragma: no cover
        raise ImportError("ruptures is required for V10.1 joint-window reproduction") from e


def _map_breakpoints_to_days(points_local: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None:
        return points_local.astype(int)
    arr = np.asarray(day_index, dtype=int)
    return pd.Series([int(arr[max(0, min(len(arr) - 1, int(p) - 1))]) for p in points_local.astype(int).tolist()], name="changepoint", dtype=int)


def _map_profile_index(profile: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None or profile.empty:
        return profile
    arr = np.asarray(day_index, dtype=int)
    mapped_idx = []
    for i in profile.index.to_numpy(dtype=int):
        idx = max(0, min(len(arr) - 1, int(i)))
        mapped_idx.append(int(arr[idx]))
    out = profile.copy()
    out.index = np.asarray(mapped_idx, dtype=int)
    return out


def extract_ranked_local_peaks(profile: pd.Series, min_distance_days: int, prominence_min: float = 0.0) -> pd.DataFrame:
    cols = ["peak_id", "peak_day", "peak_score", "peak_prominence", "peak_rank", "source_type"]
    if profile.empty:
        return pd.DataFrame(columns=cols)
    s = profile.sort_index().astype(float).fillna(0.0)
    values = s.to_numpy(dtype=float)
    if find_peaks is None:
        candidates = []
        for i in range(1, len(values) - 1):
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                candidates.append(i)
        peaks = np.asarray(candidates, dtype=int)
        prominences = np.zeros(peaks.shape, dtype=float)
    else:
        peaks, _ = find_peaks(values, distance=max(1, int(min_distance_days)))
        if peak_prominences is not None and peaks.size:
            prominences, _, _ = peak_prominences(values, peaks)
        else:
            prominences = np.zeros(peaks.shape, dtype=float)
    rows = []
    for i, (p, prom) in enumerate(zip(peaks, prominences), start=1):
        if float(prom) < float(prominence_min):
            continue
        rows.append(
            {
                "peak_id": f"LP{i:03d}",
                "peak_day": int(s.index[int(p)]),
                "peak_score": float(values[int(p)]),
                "peak_prominence": float(prom),
                "source_type": "local_peak",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.sort_values(["peak_score", "peak_prominence", "peak_day"], ascending=[False, False, True]).reset_index(drop=True)
    df["peak_rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df[cols]


def _nearest_peak_day(local_peaks_df: pd.DataFrame, target_day: int, radius: int) -> int:
    if local_peaks_df is None or local_peaks_df.empty:
        return int(target_day)
    sub = local_peaks_df.copy()
    sub["abs_offset"] = (sub["peak_day"].astype(int) - int(target_day)).abs()
    sub = sub.sort_values(["abs_offset", "peak_score", "peak_day"], ascending=[True, False, True]).reset_index(drop=True)
    if sub.empty:
        return int(target_day)
    best = sub.iloc[0]
    if int(best["abs_offset"]) <= int(radius):
        return int(best["peak_day"])
    return int(target_day)


def build_primary_points_table(points: pd.Series, profile: pd.Series, local_peaks_df: pd.DataFrame | None, radius: int) -> pd.DataFrame:
    cols = ["point_id", "point_day", "month_day", "peak_score", "source_type", "raw_point_day", "matched_peak_day"]
    if points is None or len(points) == 0:
        return pd.DataFrame(columns=cols)
    rows = []
    for i, raw_point_day in enumerate(sorted(int(x) for x in pd.Series(points).astype(int).tolist()), start=1):
        point_day = _nearest_peak_day(local_peaks_df, raw_point_day, radius) if local_peaks_df is not None else int(raw_point_day)
        rows.append(
            {
                "point_id": f"RP{i:03d}",
                "point_day": int(point_day),
                "month_day": day_index_to_month_day(point_day),
                "peak_score": float(profile.get(int(point_day), np.nan)) if profile is not None and not profile.empty else np.nan,
                "source_type": "formal_primary",
                "raw_point_day": int(raw_point_day),
                "matched_peak_day": int(point_day),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def run_ruptures_window(state_matrix: np.ndarray, cfg: DetectorConfig, day_index: np.ndarray | None = None) -> dict[str, Any]:
    rpt = _import_ruptures()
    signal = np.asarray(state_matrix, dtype=float)
    algo = rpt.Window(width=cfg.width, model=cfg.model, min_size=cfg.min_size, jump=cfg.jump).fit(signal)
    if cfg.selection_mode == "fixed_n_bkps":
        if cfg.fixed_n_bkps is None:
            raise ValueError("fixed_n_bkps must be set")
        bkps = algo.predict(n_bkps=cfg.fixed_n_bkps)
    elif cfg.selection_mode == "pen":
        bkps = algo.predict(pen=cfg.pen)
    elif cfg.selection_mode == "epsilon":
        bkps = algo.predict(epsilon=cfg.epsilon)
    else:
        raise ValueError(f"Unsupported selection_mode={cfg.selection_mode}")
    points_local = pd.Series([int(x) for x in bkps[:-1]], name="changepoint", dtype=int)
    score = getattr(algo, "score", None)
    if score is None:
        profile_raw = pd.Series(dtype=float, name="profile")
    else:
        arr = np.asarray(score, dtype=float).ravel()
        width_half = int(algo.width // 2)
        idx = np.arange(width_half, width_half + len(arr), dtype=int)
        profile_raw = pd.Series(arr, index=idx, name="profile")
    profile = _map_profile_index(profile_raw, day_index)
    points = _map_breakpoints_to_days(points_local, day_index)
    return {"points": points, "profile": profile, "points_local": points_local, "profile_local": profile_raw}


def run_point_detector(state_matrix: np.ndarray, valid_day_index: np.ndarray, cfg: DetectorConfig) -> dict[str, Any]:
    out = run_ruptures_window(state_matrix, cfg, day_index=valid_day_index)
    local_peaks_df = extract_ranked_local_peaks(out["profile"], cfg.local_peak_min_distance_days, prominence_min=0.0)
    primary_points_df = build_primary_points_table(out["points"], out["profile"], local_peaks_df, cfg.nearest_peak_search_radius_days)
    return {**out, "primary_points_df": primary_points_df, "local_peaks_df": local_peaks_df}


def build_candidate_registry(local_peaks_df: pd.DataFrame, primary_points_df: pd.DataFrame | None = None, source_run_tag: str = "joint_main_window_reproduce_v10_1") -> pd.DataFrame:
    cols = [
        "candidate_id",
        "point_day",
        "month_day",
        "registry_rank",
        "source_run_tag",
        "peak_score",
        "peak_prominence",
        "is_formal_primary",
        "nearest_primary_day",
    ]
    if local_peaks_df is None or local_peaks_df.empty:
        return pd.DataFrame(columns=cols)
    peaks = local_peaks_df.copy()
    peaks["peak_day"] = pd.to_numeric(peaks["peak_day"], errors="coerce")
    peaks["peak_score"] = pd.to_numeric(peaks.get("peak_score"), errors="coerce")
    peaks["peak_prominence"] = pd.to_numeric(peaks.get("peak_prominence"), errors="coerce")
    peaks = peaks.dropna(subset=["peak_day"]).sort_values(["peak_day", "peak_score"], ascending=[True, False]).reset_index(drop=True)
    if primary_points_df is not None and not primary_points_df.empty:
        primary_days = set(pd.to_numeric(primary_points_df["point_day"], errors="coerce").dropna().astype(int).tolist())
        primary_list = sorted(primary_days)
    else:
        primary_days = set()
        primary_list = []
    rows = []
    for i, row in peaks.iterrows():
        day = int(row["peak_day"])
        nearest_primary_day = np.nan
        if primary_list:
            nearest_primary_day = int(min(primary_list, key=lambda x: abs(x - day)))
        rows.append(
            {
                "candidate_id": f"CP{i + 1:03d}",
                "point_day": day,
                "month_day": day_index_to_month_day(day),
                "registry_rank": int(i + 1),
                "source_run_tag": source_run_tag,
                "peak_score": float(row["peak_score"]) if pd.notna(row["peak_score"]) else np.nan,
                "peak_prominence": float(row["peak_prominence"]) if pd.notna(row["peak_prominence"]) else np.nan,
                "is_formal_primary": bool(day in primary_days),
                "nearest_primary_day": nearest_primary_day,
            }
        )
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# Bootstrap matching
# =============================================================================

def classify_match_type(abs_offset_days: int | float | None, cfg: BootstrapConfig) -> str:
    if abs_offset_days is None or not np.isfinite(abs_offset_days):
        return "no_match"
    off = int(abs(abs_offset_days))
    if off <= int(cfg.strict_match_max_abs_offset_days):
        return "strict"
    if off <= int(cfg.match_max_abs_offset_days):
        return "matched"
    if off <= int(cfg.near_max_abs_offset_days):
        return "near"
    return "no_match"


def match_candidates_to_local_peaks(
    registry_df: pd.DataFrame,
    local_peaks_df: pd.DataFrame,
    cfg: BootstrapConfig,
    replicate_id: int,
    replicate_kind: str,
    replicate_label: str | int | None = None,
) -> pd.DataFrame:
    cols = [
        "replicate_id",
        "replicate_kind",
        "replicate_label",
        "candidate_id",
        "point_day",
        "month_day",
        "matched_peak_day",
        "matched_peak_score",
        "matched_peak_prominence",
        "offset_days",
        "abs_offset_days",
        "match_type",
    ]
    if registry_df is None or registry_df.empty:
        return pd.DataFrame(columns=cols)
    peaks = local_peaks_df.copy() if local_peaks_df is not None else pd.DataFrame(columns=["peak_day", "peak_score", "peak_prominence"])
    rows = []
    for _, ref in registry_df.iterrows():
        target_day = int(ref["point_day"])
        if peaks.empty:
            matched_peak_day = np.nan
            matched_peak_score = np.nan
            matched_peak_prominence = np.nan
            offset_days = np.nan
            abs_offset_days = np.nan
            match_type = "no_match"
        else:
            sub = peaks.copy()
            sub["abs_offset_days"] = (pd.to_numeric(sub["peak_day"], errors="coerce") - target_day).abs()
            sub = sub.sort_values(["abs_offset_days", "peak_score", "peak_day"], ascending=[True, False, True]).reset_index(drop=True)
            best = sub.iloc[0]
            matched_peak_day = int(best["peak_day"]) if pd.notna(best["peak_day"]) else np.nan
            matched_peak_score = float(best["peak_score"]) if pd.notna(best.get("peak_score")) else np.nan
            matched_peak_prominence = float(best["peak_prominence"]) if pd.notna(best.get("peak_prominence")) else np.nan
            offset_days = int(matched_peak_day - target_day) if pd.notna(matched_peak_day) else np.nan
            abs_offset_days = abs(offset_days) if pd.notna(offset_days) else np.nan
            match_type = classify_match_type(abs_offset_days, cfg)
        rows.append(
            {
                "replicate_id": int(replicate_id),
                "replicate_kind": str(replicate_kind),
                "replicate_label": replicate_label,
                "candidate_id": str(ref["candidate_id"]),
                "point_day": int(ref["point_day"]),
                "month_day": ref.get("month_day"),
                "matched_peak_day": matched_peak_day,
                "matched_peak_score": matched_peak_score,
                "matched_peak_prominence": matched_peak_prominence,
                "offset_days": offset_days,
                "abs_offset_days": abs_offset_days,
                "match_type": match_type,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def summarize_match_records(records_df: pd.DataFrame, group_field: str, point_day_map: dict[str, int]) -> pd.DataFrame:
    cols = [
        "candidate_id",
        "point_day",
        f"{group_field}_strict_fraction",
        f"{group_field}_match_fraction",
        f"{group_field}_near_fraction",
        f"{group_field}_no_match_fraction",
        f"median_{group_field}_matched_peak_score",
        f"median_{group_field}_abs_offset_days",
    ]
    if records_df is None or records_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for cid, sub in records_df.groupby("candidate_id"):
        matched_score = pd.to_numeric(sub["matched_peak_score"], errors="coerce")
        abs_off = pd.to_numeric(sub["abs_offset_days"], errors="coerce")
        rows.append(
            {
                "candidate_id": str(cid),
                "point_day": int(point_day_map.get(str(cid))),
                f"{group_field}_strict_fraction": float((sub["match_type"] == "strict").mean()),
                f"{group_field}_match_fraction": float(sub["match_type"].isin(["strict", "matched"]).mean()),
                f"{group_field}_near_fraction": float((sub["match_type"] == "near").mean()),
                f"{group_field}_no_match_fraction": float((sub["match_type"] == "no_match").mean()),
                f"median_{group_field}_matched_peak_score": float(matched_score.median()) if matched_score.notna().any() else np.nan,
                f"median_{group_field}_abs_offset_days": float(abs_off.median()) if abs_off.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values(["point_day", "candidate_id"]).reset_index(drop=True)


def _progress_iter(items, enabled: bool, desc: str):
    if enabled:
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc)
        except Exception:
            return items
    return items


def run_bootstrap_support(profiles: dict[str, PointLayerProfile], years: np.ndarray, registry_df: pd.DataFrame, settings: Settings) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(settings.bootstrap.random_seed)
    n_years = len(np.asarray(years).astype(int))
    n_boot = settings.bootstrap.resolved_n_bootstrap()
    records: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    point_day_map = {str(r["candidate_id"]): int(r["point_day"]) for _, r in registry_df.iterrows()}
    for rep in _progress_iter(range(n_boot), settings.bootstrap.progress, "V10.1 joint bootstrap"):
        sampled = rng.integers(0, n_years, size=n_years, endpoint=False)
        state = build_resampled_state_matrix(profiles, sampled, settings.state)
        if state["valid_day_index"].size < max(2 * int(settings.detector.width), 3):
            meta_rows.append({"replicate_id": int(rep), "status": "skipped_insufficient_valid_days", "n_valid_days": int(state["valid_day_index"].size), "sampled_year_indices": sampled.tolist()})
            continue
        det = run_point_detector(state["state_matrix"][state["valid_day_mask"], :], state["valid_day_index"], settings.detector)
        rec = match_candidates_to_local_peaks(registry_df, det["local_peaks_df"], settings.bootstrap, int(rep), "bootstrap", int(rep))
        records.append(rec)
        meta_rows.append({"replicate_id": int(rep), "status": "success", "n_valid_days": int(state["valid_day_index"].size), "sampled_year_indices": sampled.tolist()})
    records_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    summary_df = summarize_match_records(records_df, "bootstrap", point_day_map)
    meta_df = pd.DataFrame(meta_rows)
    return {"records_df": records_df, "summary_df": summary_df, "meta_df": meta_df}


# =============================================================================
# Window band and registry layer
# =============================================================================

def _nearest_profile_index(profile_index: np.ndarray, day: int) -> int:
    return int(np.argmin(np.abs(profile_index - int(day))))


def _compute_support_floor(peak_score: float, peak_prominence: float, profile_values: np.ndarray, cfg: BandConfig) -> tuple[float, float]:
    valid = profile_values[np.isfinite(profile_values)]
    global_floor = float(np.quantile(valid, float(cfg.peak_floor_quantile))) if valid.size else 0.0
    if np.isfinite(peak_score) and np.isfinite(peak_prominence):
        relative_floor = float(peak_score - float(cfg.prominence_ratio_threshold) * peak_prominence)
    elif np.isfinite(peak_score):
        relative_floor = float(peak_score)
    else:
        relative_floor = global_floor
    return max(global_floor, relative_floor), global_floor


def _find_local_valley_between(days: np.ndarray, vals: np.ndarray, left_day: int, right_day: int) -> tuple[int | None, float | None]:
    left_idx = _nearest_profile_index(days, left_day)
    right_idx = _nearest_profile_index(days, right_day)
    if left_idx == right_idx:
        return int(days[left_idx]), float(vals[left_idx])
    i0, i1 = sorted((left_idx, right_idx))
    seg = vals[i0 : i1 + 1]
    if seg.size <= 2 or not np.isfinite(seg).any():
        return None, None
    rel = int(np.nanargmin(seg))
    idx = i0 + rel
    return int(days[idx]), float(vals[idx])


def build_candidate_point_bands(registry_df: pd.DataFrame, profile: pd.Series, cfg: BandConfig) -> pd.DataFrame:
    cols = [
        "candidate_id",
        "point_day",
        "month_day",
        "band_start_day",
        "band_end_day",
        "band_center_day",
        "peak_score",
        "peak_prominence",
        "support_floor",
        "global_floor",
        "left_stop_reason",
        "right_stop_reason",
        "is_formal_primary",
    ]
    if registry_df is None or registry_df.empty:
        return pd.DataFrame(columns=cols)
    if profile is None or profile.empty:
        out = registry_df.copy()
        out["band_start_day"] = out["point_day"].astype(int) - int(cfg.min_band_half_width_days)
        out["band_end_day"] = out["point_day"].astype(int) + int(cfg.min_band_half_width_days)
        out["band_center_day"] = out["point_day"].astype(int)
        out["support_floor"] = np.nan
        out["global_floor"] = np.nan
        out["left_stop_reason"] = "min_half_width_only"
        out["right_stop_reason"] = "min_half_width_only"
        return out[cols]
    prof = profile.sort_index().astype(float)
    days = prof.index.to_numpy(dtype=int)
    vals = prof.to_numpy(dtype=float)
    registry = registry_df.sort_values("point_day").reset_index(drop=True)
    registry_days = registry["point_day"].astype(int).tolist()
    min_half = int(cfg.min_band_half_width_days)
    max_half = int(cfg.max_band_half_width_days)
    near_exempt = int(cfg.close_neighbor_exemption_days)
    rows: list[dict[str, Any]] = []
    for _, row in registry.iterrows():
        day = int(row["point_day"])
        i0 = _nearest_profile_index(days, day)
        peak_score = float(row["peak_score"]) if pd.notna(row.get("peak_score")) else float(vals[i0])
        peak_prom = float(row["peak_prominence"]) if pd.notna(row.get("peak_prominence")) else np.nan
        support_floor, global_floor = _compute_support_floor(peak_score, peak_prom, vals, cfg)
        left = i0
        left_stop_reason = "support_floor"
        while left > 0:
            candidate_next_day = int(days[left - 1])
            if abs(day - candidate_next_day) > max_half:
                left_stop_reason = "max_half_width"
                break
            next_val = vals[left - 1]
            if not np.isfinite(next_val) or next_val < support_floor:
                left_stop_reason = "support_floor"
                break
            left -= 1
        right = i0
        right_stop_reason = "support_floor"
        while right < len(vals) - 1:
            candidate_next_day = int(days[right + 1])
            if abs(candidate_next_day - day) > max_half:
                right_stop_reason = "max_half_width"
                break
            next_val = vals[right + 1]
            if not np.isfinite(next_val) or next_val < support_floor:
                right_stop_reason = "support_floor"
                break
            right += 1
        start_day = int(min(days[left], day - min_half))
        end_day = int(max(days[right], day + min_half))
        if bool(cfg.respect_candidate_boundaries):
            other_days = [d for d in registry_days if d != day]
            left_candidates = [d for d in other_days if d < day and (day - d) > near_exempt]
            right_candidates = [d for d in other_days if d > day and (d - day) > near_exempt]
            if left_candidates and bool(cfg.truncate_at_intervening_candidate):
                nearest_left = max(left_candidates)
                if start_day <= nearest_left:
                    stop_day = nearest_left + 1
                    reason = "intervening_candidate"
                    if bool(cfg.truncate_at_local_valley):
                        valley_day, valley_val = _find_local_valley_between(days, vals, nearest_left, day)
                        if valley_day is not None and np.isfinite(valley_val) and valley_val < peak_score:
                            stop_day = max(nearest_left + 1, valley_day)
                            reason = "local_valley"
                    start_day = max(start_day, stop_day)
                    left_stop_reason = reason
            if right_candidates and bool(cfg.truncate_at_intervening_candidate):
                nearest_right = min(right_candidates)
                if end_day >= nearest_right:
                    stop_day = nearest_right - 1
                    reason = "intervening_candidate"
                    if bool(cfg.truncate_at_local_valley):
                        valley_day, valley_val = _find_local_valley_between(days, vals, day, nearest_right)
                        if valley_day is not None and np.isfinite(valley_val) and valley_val < peak_score:
                            stop_day = min(nearest_right - 1, valley_day)
                            reason = "local_valley"
                    end_day = min(end_day, stop_day)
                    right_stop_reason = reason
        start_day = min(start_day, day - min_half)
        end_day = max(end_day, day + min_half)
        start_day = max(start_day, int(days.min()))
        end_day = min(end_day, int(days.max()))
        if end_day < start_day:
            start_day = max(int(days.min()), day - min_half)
            end_day = min(int(days.max()), day + min_half)
            left_stop_reason = "fallback_min_half_width"
            right_stop_reason = "fallback_min_half_width"
        rows.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "point_day": day,
                "month_day": row.get("month_day"),
                "band_start_day": int(start_day),
                "band_end_day": int(end_day),
                "band_center_day": int(round((start_day + end_day) / 2.0)),
                "peak_score": peak_score,
                "peak_prominence": peak_prom,
                "support_floor": float(support_floor),
                "global_floor": float(global_floor),
                "left_stop_reason": left_stop_reason,
                "right_stop_reason": right_stop_reason,
                "is_formal_primary": bool(row.get("is_formal_primary", False)),
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values(["band_start_day", "point_day"]).reset_index(drop=True)


def merge_candidate_bands_into_windows(bands_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame | None, cfg: BandConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    wcols = [
        "window_id",
        "start_day",
        "end_day",
        "center_day",
        "main_peak_day",
        "n_member_points",
        "member_candidate_ids",
        "max_member_bootstrap_match_fraction",
        "merge_reason",
        "protected_split_flag",
    ]
    mcols = ["window_id", "candidate_id", "point_day", "is_main_peak", "bootstrap_match_fraction_5d"]
    if bands_df is None or bands_df.empty:
        return pd.DataFrame(columns=wcols), pd.DataFrame(columns=mcols)
    bands = bands_df.sort_values(["band_start_day", "band_end_day", "point_day"]).reset_index(drop=True)
    boot: dict[str, float] = {}
    if bootstrap_summary_df is not None and not bootstrap_summary_df.empty and "candidate_id" in bootstrap_summary_df.columns:
        boot = bootstrap_summary_df.set_index("candidate_id")["bootstrap_match_fraction"].to_dict()
    sig_thr = float(cfg.significant_peak_threshold)
    near_exempt = int(cfg.close_neighbor_exemption_days)
    gap = int(cfg.merge_gap_days)
    groups: list[tuple[list[int], str, bool]] = []
    current: list[int] = []
    current_group_max_end: int | None = None
    current_group_protected_days: list[int] = []
    current_group_protected_flag = False

    def close_group(reason: str, protected_split_flag: bool = False):
        nonlocal current, current_group_max_end, current_group_protected_days, current_group_protected_flag
        if current:
            groups.append((current.copy(), reason, protected_split_flag))
        current = []
        current_group_max_end = None
        current_group_protected_days = []
        current_group_protected_flag = False

    for i in range(len(bands)):
        row = bands.loc[i]
        day = int(row["point_day"])
        start = int(row["band_start_day"])
        end = int(row["band_end_day"])
        cid = str(row["candidate_id"])
        is_protected = float(boot.get(cid, np.nan)) >= sig_thr if cid in boot else False
        if not current:
            current = [i]
            current_group_max_end = end
            current_group_protected_days = [day] if is_protected else []
            current_group_protected_flag = False
            continue
        overlaps_group = bool(cfg.allow_band_merge) and start <= int(current_group_max_end) + gap
        if not overlaps_group:
            close_group("non_overlap", protected_split_flag=current_group_protected_flag)
            current = [i]
            current_group_max_end = end
            current_group_protected_days = [day] if is_protected else []
            current_group_protected_flag = False
            continue
        protected_conflict = False
        if bool(cfg.protect_significant_peaks_from_merge) and is_protected and current_group_protected_days:
            protected_conflict = all(abs(day - existing_day) > near_exempt for existing_day in current_group_protected_days)
        if protected_conflict:
            close_group("protected_split", protected_split_flag=True)
            current = [i]
            current_group_max_end = end
            current_group_protected_days = [day]
            current_group_protected_flag = False
            continue
        current.append(i)
        current_group_max_end = max(int(current_group_max_end), end)
        if is_protected:
            current_group_protected_days.append(day)
    close_group("final_group", protected_split_flag=current_group_protected_flag)
    wrows = []
    mrows = []
    for gidx, (idxs, merge_reason, protected_split_flag) in enumerate(groups, start=1):
        sub = bands.loc[idxs].copy().reset_index(drop=True)
        start_day = int(sub["band_start_day"].min())
        end_day = int(sub["band_end_day"].max())
        center_day = int(round((start_day + end_day) / 2.0))
        sub["bootstrap_match_fraction_5d"] = sub["candidate_id"].map(lambda x: float(boot.get(str(x), np.nan)))
        sub = sub.sort_values(["bootstrap_match_fraction_5d", "peak_score", "point_day"], ascending=[False, False, True]).reset_index(drop=True)
        main_peak_day = int(sub.iloc[0]["point_day"])
        window_id = f"W{gidx:03d}"
        members = ";".join(sub["candidate_id"].astype(str).tolist())
        max_boot = float(sub["bootstrap_match_fraction_5d"].max()) if sub["bootstrap_match_fraction_5d"].notna().any() else np.nan
        wrows.append(
            {
                "window_id": window_id,
                "start_day": start_day,
                "end_day": end_day,
                "center_day": center_day,
                "main_peak_day": main_peak_day,
                "n_member_points": int(len(sub)),
                "member_candidate_ids": members,
                "max_member_bootstrap_match_fraction": max_boot,
                "merge_reason": merge_reason,
                "protected_split_flag": bool(protected_split_flag),
            }
        )
        for _, subrow in sub.iterrows():
            mrows.append(
                {
                    "window_id": window_id,
                    "candidate_id": str(subrow["candidate_id"]),
                    "point_day": int(subrow["point_day"]),
                    "is_main_peak": bool(int(subrow["point_day"]) == main_peak_day),
                    "bootstrap_match_fraction_5d": float(subrow["bootstrap_match_fraction_5d"]) if pd.notna(subrow["bootstrap_match_fraction_5d"]) else np.nan,
                }
            )
    return pd.DataFrame(wrows, columns=wcols), pd.DataFrame(mrows, columns=mcols)


def summarize_window_uncertainty(
    windows_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    bootstrap_records_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame | None,
    cfg: UncertaintyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scols = [
        "window_id",
        "main_peak_day",
        "main_peak_candidate_id",
        "main_peak_bootstrap_match_fraction",
        "n_returned_replicates_for_interval",
        "return_day_median",
        "return_day_q02_5",
        "return_day_q97_5",
        "return_day_p10",
        "return_day_p90",
        "return_day_iqr",
        "return_day_width95",
        "return_day_width80",
    ]
    dcols = ["window_id", "candidate_id", "replicate_id", "matched_peak_day", "match_type"]
    if windows_df is None or windows_df.empty or membership_df is None or membership_df.empty or bootstrap_records_df is None or bootstrap_records_df.empty:
        return pd.DataFrame(columns=scols), pd.DataFrame(columns=dcols)
    allowed = set(cfg.interval_match_types)
    boot: dict[str, float] = {}
    if bootstrap_summary_df is not None and not bootstrap_summary_df.empty:
        boot = bootstrap_summary_df.set_index("candidate_id")["bootstrap_match_fraction"].to_dict()
    rows = []
    detail_rows = []
    for _, win in windows_df.iterrows():
        wid = str(win["window_id"])
        main = membership_df[(membership_df["window_id"] == wid) & (membership_df["is_main_peak"])]
        if main.empty:
            continue
        main = main.iloc[0]
        cid = str(main["candidate_id"])
        sub = bootstrap_records_df[(bootstrap_records_df["candidate_id"] == cid) & (bootstrap_records_df["match_type"].isin(allowed))].copy()
        for _, rec in sub.iterrows():
            detail_rows.append(
                {
                    "window_id": wid,
                    "candidate_id": cid,
                    "replicate_id": int(rec["replicate_id"]),
                    "matched_peak_day": int(rec["matched_peak_day"]) if pd.notna(rec["matched_peak_day"]) else np.nan,
                    "match_type": str(rec["match_type"]),
                }
            )
        vals = pd.to_numeric(sub["matched_peak_day"], errors="coerce").dropna().astype(float)
        if vals.empty:
            rows.append(
                {
                    "window_id": wid,
                    "main_peak_day": int(win["main_peak_day"]),
                    "main_peak_candidate_id": cid,
                    "main_peak_bootstrap_match_fraction": float(boot.get(cid, np.nan)),
                    "n_returned_replicates_for_interval": 0,
                    "return_day_median": np.nan,
                    "return_day_q02_5": np.nan,
                    "return_day_q97_5": np.nan,
                    "return_day_p10": np.nan,
                    "return_day_p90": np.nan,
                    "return_day_iqr": np.nan,
                    "return_day_width95": np.nan,
                    "return_day_width80": np.nan,
                }
            )
            continue
        q25, q75 = np.quantile(vals, [0.25, 0.75])
        p10, p90 = np.quantile(vals, [0.10, 0.90])
        q025, q975 = np.quantile(vals, [0.025, 0.975])
        rows.append(
            {
                "window_id": wid,
                "main_peak_day": int(win["main_peak_day"]),
                "main_peak_candidate_id": cid,
                "main_peak_bootstrap_match_fraction": float(boot.get(cid, np.nan)),
                "n_returned_replicates_for_interval": int(vals.size),
                "return_day_median": float(np.median(vals)),
                "return_day_q02_5": float(q025),
                "return_day_q97_5": float(q975),
                "return_day_p10": float(p10),
                "return_day_p90": float(p90),
                "return_day_iqr": float(q75 - q25),
                "return_day_width95": float(q975 - q025),
                "return_day_width80": float(p90 - p10),
            }
        )
    return pd.DataFrame(rows, columns=scols), pd.DataFrame(detail_rows, columns=dcols)


# =============================================================================
# Regression audit and lineage
# =============================================================================

def compare_tables(
    name: str,
    v10: pd.DataFrame | None,
    ref: pd.DataFrame | None,
    key_cols: list[str],
    compare_cols: list[str] | None = None,
    numeric_tol: float = 1e-9,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if ref is None:
        return {"component": name, "status": "missing_reference", "n_reference_rows": np.nan, "n_v10_rows": len(v10) if v10 is not None else np.nan, "n_key_mismatch_rows": np.nan, "n_value_differences": np.nan, "note": "reference file missing"}, pd.DataFrame()
    if v10 is None:
        return {"component": name, "status": "missing_v10_output", "n_reference_rows": len(ref), "n_v10_rows": np.nan, "n_key_mismatch_rows": np.nan, "n_value_differences": np.nan, "note": "v10 output missing"}, pd.DataFrame()
    missing = [c for c in key_cols if c not in v10.columns or c not in ref.columns]
    if missing:
        return {"component": name, "status": "missing_key_columns", "n_reference_rows": len(ref), "n_v10_rows": len(v10), "n_key_mismatch_rows": np.nan, "n_value_differences": np.nan, "note": ";".join(missing)}, pd.DataFrame()
    v = v10.copy()
    r = ref.copy()
    for c in key_cols:
        v[c] = v[c].astype(str)
        r[c] = r[c].astype(str)
    v = v.sort_values(key_cols).reset_index(drop=True)
    r = r.sort_values(key_cols).reset_index(drop=True)
    v_keys = set(map(tuple, v[key_cols].to_numpy()))
    r_keys = set(map(tuple, r[key_cols].to_numpy()))
    mismatch = sorted((v_keys ^ r_keys))
    common_keys = sorted(v_keys & r_keys)
    if compare_cols is None:
        compare_cols = [c for c in r.columns if c in v.columns and c not in key_cols]
    diff_rows = []
    if common_keys and compare_cols:
        vv = v.set_index(key_cols)
        rr = r.set_index(key_cols)
        for key in common_keys:
            key_use = key[0] if len(key) == 1 else key
            for col in compare_cols:
                a = vv.loc[key_use, col]
                b = rr.loc[key_use, col]
                if isinstance(a, pd.Series):
                    a = a.iloc[0]
                if isinstance(b, pd.Series):
                    b = b.iloc[0]
                if pd.isna(a) and pd.isna(b):
                    continue
                different = False
                try:
                    af = float(a)
                    bf = float(b)
                    if not (np.isfinite(af) or np.isfinite(bf)):
                        different = False
                    else:
                        different = abs(af - bf) > numeric_tol
                except Exception:
                    different = str(a) != str(b)
                if different:
                    row = {k: key[i] for i, k in enumerate(key_cols)}
                    row.update({"column": col, "v10_value": a, "reference_value": b})
                    diff_rows.append(row)
    status = "pass" if not mismatch and not diff_rows else "difference_found"
    note = ""
    if mismatch:
        note += f"key_mismatch={len(mismatch)}"
    if diff_rows:
        note += (";" if note else "") + f"value_diff={len(diff_rows)}"
    return {
        "component": name,
        "status": status,
        "n_reference_rows": int(len(ref)),
        "n_v10_rows": int(len(v10)),
        "n_key_mismatch_rows": int(len(mismatch)),
        "n_value_differences": int(len(diff_rows)),
        "note": note,
    }, pd.DataFrame(diff_rows)


def build_lineage(
    registry: pd.DataFrame,
    boot_summary: pd.DataFrame,
    windows: pd.DataFrame,
    membership: pd.DataFrame,
    strict_windows: tuple[tuple[str, int, int, int], ...],
) -> pd.DataFrame:
    boot = boot_summary.set_index("candidate_id").to_dict("index") if boot_summary is not None and not boot_summary.empty else {}
    mem = membership.set_index("candidate_id").to_dict("index") if membership is not None and not membership.empty else {}
    win = windows.set_index("window_id").to_dict("index") if windows is not None and not windows.empty else {}
    strict_by_day = {int(anchor): (wid, int(start), int(end)) for wid, anchor, start, end in strict_windows}
    rows = []
    for _, r in registry.sort_values("point_day").iterrows():
        cid = str(r["candidate_id"])
        day = int(r["point_day"])
        m = mem.get(cid, {})
        wid = str(m.get("window_id", "")) if m else ""
        w = win.get(wid, {}) if wid else {}
        strict_id = ""
        strict_flag = False
        strict_reason = ""
        exclusion = ""
        if day in strict_by_day:
            strict_id = strict_by_day[day][0]
            strict_flag = True
            strict_reason = "candidate_day_matches_strict_accepted_anchor"
        elif w and int(w.get("main_peak_day", -999)) in strict_by_day:
            strict_id = strict_by_day[int(w.get("main_peak_day"))][0]
            strict_flag = True
            strict_reason = "derived_window_main_peak_matches_strict_accepted_anchor"
        else:
            exclusion = "not_in_strict_accepted_window_set"
        if strict_flag and bool(m.get("is_main_peak", False)):
            lineage_status = "strict_accepted_main_window"
        elif w:
            lineage_status = "derived_non_strict_candidate_window" if not strict_flag else "strict_window_member"
        else:
            lineage_status = "local_candidate_not_derived_main"
        brow = boot.get(cid, {})
        rows.append(
            {
                "candidate_id": cid,
                "candidate_day": day,
                "month_day": r.get("month_day"),
                "is_formal_primary": bool(r.get("is_formal_primary", False)),
                "v6_peak_score": float(r.get("peak_score")) if pd.notna(r.get("peak_score")) else np.nan,
                "v6_peak_prominence": float(r.get("peak_prominence")) if pd.notna(r.get("peak_prominence")) else np.nan,
                "bootstrap_strict_fraction": float(brow.get("bootstrap_strict_fraction", np.nan)),
                "bootstrap_match_fraction": float(brow.get("bootstrap_match_fraction", np.nan)),
                "bootstrap_near_fraction": float(brow.get("bootstrap_near_fraction", np.nan)),
                "v6_1_window_id": wid,
                "v6_1_window_start": int(w.get("start_day")) if w else np.nan,
                "v6_1_window_end": int(w.get("end_day")) if w else np.nan,
                "v6_1_main_peak_day": int(w.get("main_peak_day")) if w else np.nan,
                "v6_1_is_window_main_peak": bool(m.get("is_main_peak", False)) if m else False,
                "strict_accepted_window_id": strict_id,
                "strict_accepted_flag": bool(strict_flag),
                "strict_accepted_reason": strict_reason,
                "strict_exclusion_reason": exclusion,
                "lineage_status": lineage_status,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Main runner
# =============================================================================

def _reference_paths(bundle_root: Path, settings: Settings) -> dict[str, Path]:
    # Bundle root = .../V10/v10.1.
    v10_root = bundle_root.parent
    stage_root = v10_root.parent
    return {
        "v6": stage_root / "V6" / "outputs" / settings.reference.source_v6_output_tag,
        "v6_1": stage_root / "V6_1" / "outputs" / settings.reference.source_v6_1_output_tag,
        "v10_main": v10_root / "outputs" / "peak_subpeak_reproduce_v10_a",
    }


def _copy_summary_readme(bundle_root: Path, output_root: Path, audit_df: pd.DataFrame, lineage_df: pd.DataFrame, settings: Settings) -> None:
    strict = lineage_df[lineage_df["strict_accepted_flag"] == True] if not lineage_df.empty else pd.DataFrame()
    non_strict = lineage_df[lineage_df["strict_accepted_flag"] == False] if not lineage_df.empty else pd.DataFrame()
    lines = [
        "# V10.1 joint main-window reproduction summary",
        "",
        "This run semantically rewrites the joint-object V6 -> V6_1 discovery chain inside V10.",
        "It does not perform object-native peak discovery, sensitivity testing, pair-order analysis, or physical interpretation.",
        "",
        "## Regression audit status",
    ]
    if audit_df is not None and not audit_df.empty:
        for _, r in audit_df.iterrows():
            lines.append(f"- {r['component']}: {r['status']} (key mismatch={r.get('n_key_mismatch_rows')}, value diff={r.get('n_value_differences')})")
    lines += [
        "",
        "## Lineage counts",
        f"- strict accepted candidates/members: {len(strict)}",
        f"- non-strict / candidate-lineage rows: {len(non_strict)}",
        "",
        "## Important interpretation boundary",
        "day-18-like peaks are tracked as joint free-detection candidate lineage when present; they are not automatically contamination and are not automatically strict accepted windows.",
        "",
        "## Next intended use",
        "Use `lineage/joint_main_window_lineage_v10_1.csv` as the reference coordinate system before object-native peak discovery or sensitivity tests.",
    ]
    (output_root / "JOINT_MAIN_WINDOW_REPRODUCE_V10_1_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_joint_main_window_reproduce_v10_1(bundle_root: Path | str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings()
    bundle_root = Path(bundle_root) if bundle_root is not None else Path(__file__).resolve().parents[1]
    output_root = bundle_root / "outputs" / settings.output_tag
    log_root = bundle_root / "logs"
    started = now_utc()
    clean_output_dirs(output_root)
    log_root.mkdir(parents=True, exist_ok=True)
    ref_paths = _reference_paths(bundle_root, settings)

    write_json(settings.to_dict(), output_root / "config_used.json")

    smoothed_path = settings.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    profiles = build_profiles(smoothed, settings.profile)
    write_dataframe(summarize_profile_validity(profiles), output_root / "joint_point_layer" / "profile_validity_v10_1.csv")
    write_dataframe(summarize_profile_empty_slices(profiles), output_root / "joint_point_layer" / "profile_empty_slice_audit_v10_1.csv")
    meta_profile_rows = []
    for name, p in profiles.items():
        meta_profile_rows.append(
            {
                "object_name": name,
                "cube_shape": "x".join(map(str, p.raw_cube.shape)),
                "lat_grid_min": float(np.nanmin(p.lat_grid)),
                "lat_grid_max": float(np.nanmax(p.lat_grid)),
                "n_lat_features": int(p.lat_grid.size),
                "lon_range": str(p.lon_range),
                "lat_range": str(p.lat_range),
            }
        )
    write_dataframe(pd.DataFrame(meta_profile_rows), output_root / "joint_point_layer" / "object_profile_meta_v10_1.csv")

    state = build_state_matrix(profiles, settings.state)
    write_json(state["state_vector_meta"], output_root / "joint_point_layer" / "joint_state_matrix_meta_v10_1.json")
    write_dataframe(state["feature_table"], output_root / "joint_point_layer" / "joint_state_feature_table_v10_1.csv")
    write_dataframe(state["state_block_energy_before_after"]["weights"], output_root / "joint_point_layer" / "joint_state_block_weight_audit_v10_1.csv")
    write_dataframe(pd.DataFrame({"valid_day_index": state["valid_day_index"]}), output_root / "joint_point_layer" / "joint_state_valid_day_index_v10_1.csv")
    write_dataframe(state["state_feature_scale_before_after"], output_root / "joint_point_layer" / "joint_state_feature_scale_before_after_v10_1.csv")
    write_dataframe(state["state_empty_feature_audit"], output_root / "joint_point_layer" / "joint_state_empty_feature_audit_v10_1.csv")

    det = run_point_detector(state["state_matrix"][state["valid_day_mask"], :], state["valid_day_index"], settings.detector)
    primary = det["primary_points_df"]
    local_peaks = det["local_peaks_df"]
    detector_profile = det["profile"].rename_axis("day").reset_index(name="profile_score")
    write_dataframe(detector_profile, output_root / "joint_point_layer" / "joint_detector_profile_v10_1.csv")
    write_dataframe(primary, output_root / "joint_point_layer" / "joint_ruptures_primary_points_v10_1.csv")
    write_dataframe(local_peaks, output_root / "joint_point_layer" / "joint_detector_local_peaks_all_v10_1.csv")

    registry = build_candidate_registry(local_peaks, primary, source_run_tag=settings.output_tag)
    write_dataframe(registry, output_root / "joint_point_layer" / "joint_candidate_registry_v10_1.csv")

    bootstrap = run_bootstrap_support(profiles, smoothed["years"], registry, settings)
    write_dataframe(bootstrap["records_df"], output_root / "joint_point_layer" / "joint_candidate_bootstrap_match_records_v10_1.csv")
    write_dataframe(bootstrap["summary_df"], output_root / "joint_point_layer" / "joint_candidate_bootstrap_summary_v10_1.csv")
    write_dataframe(bootstrap["meta_df"], output_root / "joint_point_layer" / "joint_candidate_bootstrap_replicates_v10_1.csv")

    bands = build_candidate_point_bands(registry, det["profile"], settings.band)
    write_dataframe(bands, output_root / "joint_window_layer" / "joint_candidate_point_bands_v10_1.csv")
    windows, membership = merge_candidate_bands_into_windows(bands, bootstrap["summary_df"], settings.band)
    write_dataframe(windows, output_root / "joint_window_layer" / "joint_derived_windows_registry_v10_1.csv")
    write_dataframe(membership, output_root / "joint_window_layer" / "joint_window_point_membership_v10_1.csv")
    uncertainty, return_dist = summarize_window_uncertainty(windows, membership, bootstrap["records_df"], bootstrap["summary_df"], settings.uncertainty)
    write_dataframe(uncertainty, output_root / "joint_window_layer" / "joint_window_uncertainty_summary_v10_1.csv")
    write_dataframe(return_dist, output_root / "joint_window_layer" / "joint_window_return_day_distribution_v10_1.csv")

    lineage = build_lineage(registry, bootstrap["summary_df"], windows, membership, settings.reference.strict_accepted_windows)
    write_dataframe(lineage, output_root / "lineage" / "joint_main_window_lineage_v10_1.csv")

    # Regression audit against old CSV outputs.  This is intentionally read-only.
    audit_rows: list[dict[str, Any]] = []
    diff_tables: list[tuple[str, pd.DataFrame]] = []
    v6 = ref_paths["v6"]
    v61 = ref_paths["v6_1"]
    comparisons = [
        ("ruptures_primary_points", primary, safe_read_csv(v6 / "ruptures_primary_points.csv"), ["point_id"], ["point_day", "raw_point_day", "matched_peak_day", "peak_score"]),
        ("detector_local_peaks_all", local_peaks, safe_read_csv(v6 / "detector_local_peaks_all.csv"), ["peak_id"], ["peak_day", "peak_score", "peak_prominence", "peak_rank"]),
        ("baseline_detected_peaks_registry", registry, safe_read_csv(v6 / "baseline_detected_peaks_registry.csv"), ["candidate_id"], ["point_day", "registry_rank", "peak_score", "peak_prominence", "is_formal_primary", "nearest_primary_day"]),
        ("candidate_points_bootstrap_summary", bootstrap["summary_df"], safe_read_csv(v6 / "candidate_points_bootstrap_summary.csv"), ["candidate_id"], ["point_day", "bootstrap_strict_fraction", "bootstrap_match_fraction", "bootstrap_near_fraction", "bootstrap_no_match_fraction"]),
        ("candidate_points_bootstrap_match_records", bootstrap["records_df"], safe_read_csv(v6 / "candidate_points_bootstrap_match_records.csv"), ["replicate_id", "candidate_id"], ["matched_peak_day", "matched_peak_score", "matched_peak_prominence", "offset_days", "abs_offset_days", "match_type"]),
        ("candidate_point_bands", bands, safe_read_csv(v61 / "candidate_point_bands.csv"), ["candidate_id"], ["point_day", "band_start_day", "band_end_day", "band_center_day", "support_floor", "global_floor", "left_stop_reason", "right_stop_reason"]),
        ("derived_windows_registry", windows, safe_read_csv(v61 / "derived_windows_registry.csv"), ["window_id"], ["start_day", "end_day", "center_day", "main_peak_day", "n_member_points", "member_candidate_ids", "max_member_bootstrap_match_fraction"]),
        ("window_point_membership", membership, safe_read_csv(v61 / "window_point_membership.csv"), ["window_id", "candidate_id"], ["point_day", "is_main_peak", "bootstrap_match_fraction_5d"]),
        ("window_uncertainty_summary", uncertainty, safe_read_csv(v61 / "window_uncertainty_summary.csv"), ["window_id"], ["main_peak_day", "main_peak_candidate_id", "main_peak_bootstrap_match_fraction", "n_returned_replicates_for_interval", "return_day_median", "return_day_q02_5", "return_day_q97_5"]),
        ("window_return_day_distribution", return_dist, safe_read_csv(v61 / "window_return_day_distribution.csv"), ["window_id", "candidate_id", "replicate_id"], ["matched_peak_day", "match_type"]),
    ]
    for name, v10df, refdf, keys, cols in comparisons:
        row, diff = compare_tables(name, v10df, refdf, keys, cols, numeric_tol=1e-8)
        audit_rows.append(row)
        if diff is not None and not diff.empty:
            diff_tables.append((name, diff))
            write_dataframe(diff, output_root / "audit" / f"{name}_diff_detail.csv")
        else:
            write_dataframe(pd.DataFrame(columns=[*keys, "column", "v10_value", "reference_value"]), output_root / "audit" / f"{name}_diff_detail.csv")
    audit_df = pd.DataFrame(audit_rows)
    write_dataframe(audit_df, output_root / "audit" / "v10_1_joint_main_window_regression_audit.csv")

    summary = {
        "status": "success",
        "output_tag": settings.output_tag,
        "started_at_utc": started,
        "ended_at_utc": now_utc(),
        "smoothed_fields_path": str(smoothed_path),
        "n_candidates": int(len(registry)),
        "candidate_days": registry["point_day"].astype(int).tolist() if not registry.empty else [],
        "n_derived_windows": int(len(windows)),
        "derived_window_main_peak_days": windows["main_peak_day"].astype(int).tolist() if not windows.empty else [],
        "strict_accepted_days": [int(x[1]) for x in settings.reference.strict_accepted_windows],
        "regression_status_counts": audit_df["status"].value_counts().to_dict() if not audit_df.empty else {},
        "scope": "joint-object main-window discovery chain reproduction only",
        "does_not_import_stage_partition_v6_or_v6_1_or_v7_or_v9": True,
        "does_not_do_object_native_peak_discovery": True,
        "does_not_do_sensitivity_testing": True,
        "does_not_do_physical_interpretation": True,
    }
    write_json(summary, output_root / "summary.json")
    run_meta = {
        **summary,
        "layer_name": "stage_partition",
        "version_name": "V10.1",
        "bundle_root": str(bundle_root),
        "notes": [
            "V10.1 semantically rewrites the joint-object V6->V6_1 discovery lineage inside V10.",
            "Old V6/V6_1 CSV outputs are read only for regression audits; old Python modules are not imported.",
            "Strict accepted windows are mapped as lineage labels, not re-decided here.",
            "This run is a prerequisite for later object-native peak discovery and sensitivity tests.",
        ],
    }
    write_json(run_meta, output_root / "run_meta.json")
    _copy_summary_readme(bundle_root, output_root, audit_df, lineage, settings)
    (log_root / "last_run.txt").write_text(
        f"status=success\nended_at_utc={now_utc()}\noutput_root={output_root}\n", encoding="utf-8"
    )
    return {"output_root": output_root, "summary": summary, "audit": audit_df}


if __name__ == "__main__":
    run_joint_main_window_reproduce_v10_1()


# =============================================================================
# V10.2 object-native peak discovery
# =============================================================================

OBJECT_NATIVE_OUTPUT_TAG = "object_native_peak_discovery_v10_2"


def _build_state_from_scope_blocks(seasonal_blocks: list[np.ndarray], feature_rows: list[dict[str, Any]], object_names: list[str], cfg: StateBuilderConfig):
    """Build a V6-style state matrix for an arbitrary object subset.

    This is intentionally separate from the joint-only helper above because the
    object-native run must not silently relabel a single object as P by zipping
    against the global OBJECT_ORDER list.
    """
    raw = np.concatenate(seasonal_blocks, axis=1)
    z, feat_mean, feat_std = _zscore_along_day(raw)
    scaled = z.copy()
    block_slices: dict[str, slice] = {}
    raw_rows = []
    std_rows = []
    empty_rows = []
    start = 0
    for name, block in zip(object_names, seasonal_blocks):
        width = block.shape[1]
        slc = slice(start, start + width)
        block_slices[name] = slc
        raw_rows.append({"object_name": name, "block_size": int(width), "rms": float(np.sqrt(np.nanmean(np.square(raw[:, slc]))))})
        std_rows.append({"object_name": name, "block_size": int(width), "rms": float(np.sqrt(np.nanmean(np.square(z[:, slc]))))})
        start += width
    if cfg.block_equal_contribution and len(block_slices) > 1:
        scaled, weights = _apply_equal_block_contribution(scaled, block_slices)
    else:
        weights = pd.DataFrame(
            {"object_name": list(block_slices.keys()), "block_size": [s.stop - s.start for s in block_slices.values()], "applied_weight": 1.0}
        )
    valid_day_mask = np.all(np.isfinite(scaled), axis=1) if cfg.trim_invalid_days else np.ones(raw.shape[0], dtype=bool)
    valid_day_index = np.where(valid_day_mask)[0].astype(int)
    for j in range(raw.shape[1]):
        finite = np.isfinite(scaled[:, j])
        if not finite.all():
            feature = feature_rows[j]
            empty_rows.append(
                {
                    "feature_index": int(j),
                    "object_name": feature["object_name"],
                    "lat_value": float(feature["lat_value"]),
                    "n_invalid_days": int((~finite).sum()),
                }
            )
    meta = {
        "n_days": int(raw.shape[0]),
        "n_features": int(raw.shape[1]),
        "n_valid_days": int(valid_day_index.size),
        "n_invalid_days": int(raw.shape[0] - valid_day_index.size),
        "valid_day_index": valid_day_index.tolist(),
        "invalid_day_index": np.where(~valid_day_mask)[0].astype(int).tolist(),
        "block_slices": {k: [int(v.start), int(v.stop)] for k, v in block_slices.items()},
        "state_expression_name": "object_native_raw_smoothed_zscore" if len(object_names) == 1 else "object_subset_raw_smoothed_zscore_block_equal",
        "object_scope": ";".join(object_names),
    }
    scale_rows = []
    for item in feature_rows:
        idx = int(item["feature_index"])
        scale_rows.append(
            {
                "feature_index": idx,
                "object_name": item["object_name"],
                "lat_value": item["lat_value"],
                "raw_mean_day": float(feat_mean[idx]),
                "raw_std_day": float(feat_std[idx]),
                "z_mean_day": float(np.nanmean(z[:, idx])) if np.isfinite(z[:, idx]).any() else np.nan,
                "z_std_day": float(np.nanstd(z[:, idx])) if np.isfinite(z[:, idx]).any() else np.nan,
            }
        )
    return {
        "raw_matrix": raw,
        "state_matrix": scaled,
        "valid_day_mask": valid_day_mask,
        "valid_day_index": valid_day_index,
        "state_vector_meta": meta,
        "feature_table": pd.DataFrame(feature_rows)[["feature_index", "object_name", "lat_value"]].copy(),
        "state_feature_scale_before_after": pd.DataFrame(scale_rows),
        "state_block_energy_before_after": {"raw": pd.DataFrame(raw_rows), "standardized": pd.DataFrame(std_rows), "weights": weights},
        "state_empty_feature_audit": pd.DataFrame(empty_rows),
    }


def _build_scope_seasonal(profiles: dict[str, PointLayerProfile], object_names: list[str], year_indices: np.ndarray | None = None):
    seasonal_blocks: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    start = 0
    for name in object_names:
        cube = profiles[name].raw_cube if year_indices is None else profiles[name].raw_cube[np.asarray(year_indices, dtype=int), :, :]
        seasonal, _ = safe_nanmean(cube, axis=0, return_valid_count=True)
        seasonal_blocks.append(seasonal)
        width = seasonal.shape[1]
        for j, latv in enumerate(profiles[name].lat_grid):
            feature_rows.append(
                {
                    "feature_index": start + j,
                    "object_name": name,
                    "object_width": width,
                    "feature_in_object": j,
                    "lat_value": float(latv),
                }
            )
        start += width
    return seasonal_blocks, feature_rows


def build_scope_state_matrix(profiles: dict[str, PointLayerProfile], object_names: list[str], cfg: StateBuilderConfig, year_indices: np.ndarray | None = None) -> dict[str, Any]:
    seasonal_blocks, feature_rows = _build_scope_seasonal(profiles, object_names, year_indices)
    out = _build_state_from_scope_blocks(seasonal_blocks, feature_rows, object_names, cfg)
    if year_indices is not None:
        out["sampled_year_indices"] = np.asarray(year_indices, dtype=int)
    return out


def run_bootstrap_support_for_scope(
    profiles: dict[str, PointLayerProfile],
    object_names: list[str],
    years: np.ndarray,
    registry_df: pd.DataFrame,
    settings: Settings,
    desc: str,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(settings.bootstrap.random_seed)
    n_years = len(np.asarray(years).astype(int))
    n_boot = settings.bootstrap.resolved_n_bootstrap()
    records: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    point_day_map = {str(r["candidate_id"]): int(r["point_day"]) for _, r in registry_df.iterrows()}
    for rep in _progress_iter(range(n_boot), settings.bootstrap.progress, desc):
        sampled = rng.integers(0, n_years, size=n_years, endpoint=False)
        state = build_scope_state_matrix(profiles, object_names, settings.state, sampled)
        if state["valid_day_index"].size < max(2 * int(settings.detector.width), 3):
            meta_rows.append({"replicate_id": int(rep), "status": "skipped_insufficient_valid_days", "n_valid_days": int(state["valid_day_index"].size), "sampled_year_indices": sampled.tolist()})
            continue
        det = run_point_detector(state["state_matrix"][state["valid_day_mask"], :], state["valid_day_index"], settings.detector)
        rec = match_candidates_to_local_peaks(registry_df, det["local_peaks_df"], settings.bootstrap, int(rep), "bootstrap", int(rep))
        records.append(rec)
        meta_rows.append({"replicate_id": int(rep), "status": "success", "n_valid_days": int(state["valid_day_index"].size), "sampled_year_indices": sampled.tolist()})
    records_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    summary_df = summarize_match_records(records_df, "bootstrap", point_day_map)
    meta_df = pd.DataFrame(meta_rows)
    return {"records_df": records_df, "summary_df": summary_df, "meta_df": meta_df}


def add_object_support_class(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if df.empty:
        df["object_support_class"] = []
        return df
    frac = pd.to_numeric(df["bootstrap_match_fraction"], errors="coerce")
    cls = np.where(frac >= 0.95, "object_strong_candidate",
          np.where(frac >= 0.80, "object_candidate",
          np.where(frac >= 0.50, "object_weak_candidate", "object_unstable_candidate")))
    df["object_support_class"] = cls
    return df


def _read_joint_lineage(bundle_root: Path, settings: Settings) -> pd.DataFrame:
    v10_root = bundle_root.parent
    stage_root = v10_root.parent
    p = v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "lineage" / "joint_main_window_lineage_v10_1.csv"
    if p.exists():
        return pd.read_csv(p)
    # Fallback: derive a minimal lineage from V6/V6_1 references if V10.1 output is absent.
    v6 = stage_root / "V6" / "outputs" / settings.reference.source_v6_output_tag
    v6_1 = stage_root / "V6_1" / "outputs" / settings.reference.source_v6_1_output_tag
    reg = safe_read_csv(v6 / "baseline_detected_peaks_registry.csv")
    boot = safe_read_csv(v6 / "candidate_points_bootstrap_summary.csv")
    win = safe_read_csv(v6_1 / "derived_windows_registry.csv")
    mem = safe_read_csv(v6_1 / "window_point_membership.csv")
    if reg is None:
        return pd.DataFrame()
    if boot is None:
        boot = pd.DataFrame(columns=["candidate_id", "bootstrap_strict_fraction", "bootstrap_match_fraction", "bootstrap_near_fraction"])
    if win is None:
        win = pd.DataFrame()
    if mem is None:
        mem = pd.DataFrame()
    return build_lineage(reg, boot, win, mem, settings.reference.strict_accepted_windows)


def _read_v10_window_conditioned_selection(bundle_root: Path) -> pd.DataFrame:
    v10_root = bundle_root.parent
    p = v10_root / "outputs" / "peak_subpeak_reproduce_v10_a" / "cross_window" / "main_window_selection_all_windows.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def map_candidates_to_joint_lineage(
    object_name: str,
    catalog_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    joint_lineage_df: pd.DataFrame,
    v10_selection_df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    rows = []
    strict = list(settings.reference.strict_accepted_windows)
    if joint_lineage_df is not None and not joint_lineage_df.empty:
        joint_days = pd.to_numeric(joint_lineage_df["candidate_day"], errors="coerce").dropna().astype(int).tolist()
    else:
        joint_days = []
    win_by_candidate = membership_df.set_index("candidate_id").to_dict("index") if membership_df is not None and not membership_df.empty else {}
    win_by_id = windows_df.set_index("window_id").to_dict("index") if windows_df is not None and not windows_df.empty else {}
    sel = v10_selection_df.copy() if v10_selection_df is not None else pd.DataFrame()
    for _, r in catalog_df.sort_values("point_day").iterrows():
        cid = str(r["candidate_id"])
        day = int(r["point_day"])
        nearest_joint_day = np.nan
        nearest_joint_dist = np.nan
        nearest_joint_status = ""
        nearest_joint_window = ""
        if joint_days:
            nearest_joint_day = int(min(joint_days, key=lambda x: abs(x - day)))
            nearest_joint_dist = int(abs(nearest_joint_day - day))
            jrow = joint_lineage_df[pd.to_numeric(joint_lineage_df["candidate_day"], errors="coerce").astype("Int64") == nearest_joint_day]
            if not jrow.empty:
                nearest_joint_status = str(jrow.iloc[0].get("lineage_status", ""))
                nearest_joint_window = str(jrow.iloc[0].get("v6_1_window_id", ""))
        nearest_strict_id = ""
        nearest_strict_anchor = np.nan
        nearest_strict_dist = np.nan
        within_strict_band = False
        if strict:
            wid, anchor, start, end = min(strict, key=lambda x: abs(int(x[1]) - day))
            nearest_strict_id = wid
            nearest_strict_anchor = int(anchor)
            nearest_strict_dist = int(abs(int(anchor) - day))
            within_strict_band = int(start) <= day <= int(end)
        was_selected = False
        selected_windows = []
        if not sel.empty and "object" in sel.columns:
            sub = sel[(sel["object"].astype(str) == object_name) & (pd.to_numeric(sel["selected_peak_day"], errors="coerce").round().astype("Int64") == day)]
            if not sub.empty:
                was_selected = True
                selected_windows = sorted(sub["window_id"].astype(str).unique().tolist())
        mem = win_by_candidate.get(cid, {})
        owid = str(mem.get("window_id", "")) if mem else ""
        ow = win_by_id.get(owid, {}) if owid else {}
        relation = "no_nearby_joint_candidate"
        if np.isfinite(nearest_joint_dist):
            if nearest_joint_dist <= 5:
                relation = "near_joint_strict_accepted" if "strict_accepted" in nearest_joint_status else "near_joint_non_strict_candidate"
            elif nearest_joint_dist <= 10:
                relation = "within_10d_of_joint_candidate"
            else:
                relation = "distant_from_joint_candidate"
        rows.append({
            "object": object_name,
            "candidate_id": cid,
            "candidate_day": day,
            "candidate_date": r.get("month_day"),
            "object_derived_window_id": owid,
            "object_window_start_day": int(ow.get("start_day")) if ow else np.nan,
            "object_window_end_day": int(ow.get("end_day")) if ow else np.nan,
            "is_object_window_main_peak": bool(mem.get("is_main_peak", False)) if mem else False,
            "nearest_joint_candidate_day": nearest_joint_day,
            "distance_to_nearest_joint_candidate": nearest_joint_dist,
            "nearest_joint_lineage_status": nearest_joint_status,
            "nearest_joint_derived_window_id": nearest_joint_window,
            "nearest_strict_accepted_window_id": nearest_strict_id,
            "nearest_strict_anchor_day": nearest_strict_anchor,
            "distance_to_nearest_strict_anchor": nearest_strict_dist,
            "within_strict_accepted_window_band": bool(within_strict_band),
            "was_selected_as_v10_window_conditioned_main_peak": bool(was_selected),
            "v10_window_id_if_selected": ";".join(selected_windows),
            "candidate_relation_to_joint_lineage": relation,
        })
    return pd.DataFrame(rows)


def _write_object_summary(all_catalog: pd.DataFrame, all_windows: pd.DataFrame, all_mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for obj, sub in all_catalog.groupby("object") if not all_catalog.empty else []:
        wins = all_windows[all_windows["object"] == obj] if all_windows is not None and not all_windows.empty else pd.DataFrame()
        mp = wins["main_peak_day"].dropna().astype(int).tolist() if not wins.empty and "main_peak_day" in wins.columns else []
        mapping = all_mapping[all_mapping["object"] == obj] if all_mapping is not None and not all_mapping.empty else pd.DataFrame()
        rows.append({
            "object": obj,
            "n_candidates": int(len(sub)),
            "candidate_days": ";".join(map(str, sub["point_day"].astype(int).tolist())),
            "n_object_derived_windows": int(len(wins)),
            "object_window_main_peak_days": ";".join(map(str, mp)),
            "n_object_strong_candidates": int((sub.get("object_support_class", pd.Series(dtype=str)) == "object_strong_candidate").sum()),
            "n_object_candidate": int((sub.get("object_support_class", pd.Series(dtype=str)) == "object_candidate").sum()),
            "n_object_weak_candidates": int((sub.get("object_support_class", pd.Series(dtype=str)) == "object_weak_candidate").sum()),
            "n_candidates_near_joint_non_strict": int(mapping.get("candidate_relation_to_joint_lineage", pd.Series(dtype=str)).astype(str).str.contains("non_strict", na=False).sum()) if not mapping.empty else 0,
            "n_candidates_near_joint_strict": int(mapping.get("candidate_relation_to_joint_lineage", pd.Series(dtype=str)).astype(str).str.contains("strict_accepted", na=False).sum()) if not mapping.empty else 0,
            "n_v10_window_conditioned_main_matches": int(mapping.get("was_selected_as_v10_window_conditioned_main_peak", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not mapping.empty else 0,
        })
    return pd.DataFrame(rows)


def clean_v10_2_output_dirs(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for sub in ["by_object", "cross_object", "lineage_mapping", "audit", "figures"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def _copy_v10_2_summary(output_root: Path, summary_df: pd.DataFrame, settings: Settings, run_meta: dict[str, Any]) -> None:
    lines = [
        "# V10.2 object-native peak discovery summary",
        "",
        "This run applies the V10.1 free-season peak discovery semantics separately to P, V, H, Je and Jw.",
        "It does not perform sensitivity testing, pair-order analysis, or physical interpretation.",
        "",
        "## Run status",
        f"- status: {run_meta.get('status')}",
        f"- n_bootstrap: {settings.bootstrap.resolved_n_bootstrap()}",
        "",
        "## Object-native candidate summary",
    ]
    if summary_df is not None and not summary_df.empty:
        for _, r in summary_df.iterrows():
            lines.append(f"- {r['object']}: candidates={r['n_candidates']} days={r['candidate_days']} object-window-main-days={r['object_window_main_peak_days']}")
    lines += [
        "",
        "## Interpretation boundary",
        "These object-native candidates are not new accepted joint windows and should not be interpreted as physical sub-peaks by this run alone.",
        "They provide a candidate lineage layer for later sensitivity and physical audits.",
    ]
    (output_root / "OBJECT_NATIVE_PEAK_DISCOVERY_V10_2_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_object_native_peak_discovery_v10_2(bundle_root: Path | str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings()
    settings.output_tag = OBJECT_NATIVE_OUTPUT_TAG
    bundle_root = Path(bundle_root) if bundle_root is not None else Path(__file__).resolve().parents[1]
    output_root = bundle_root / "outputs" / settings.output_tag
    log_root = bundle_root / "logs"
    started = now_utc()
    clean_v10_2_output_dirs(output_root)
    log_root.mkdir(parents=True, exist_ok=True)
    write_json(settings.to_dict(), output_root / "config_used.json")

    smoothed_path = settings.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    profiles = build_profiles(smoothed, settings.profile)
    years = smoothed.get("years", np.arange(next(iter(profiles.values())).raw_cube.shape[0]))

    joint_lineage = _read_joint_lineage(bundle_root, settings)
    v10_sel = _read_v10_window_conditioned_selection(bundle_root)
    write_dataframe(joint_lineage, output_root / "lineage_mapping" / "joint_lineage_reference_used_v10_2.csv")
    write_dataframe(v10_sel, output_root / "lineage_mapping" / "v10_window_conditioned_selection_reference_used_v10_2.csv")

    all_catalogs = []
    all_windows = []
    all_memberships = []
    all_mappings = []
    audit_rows = []

    for obj in OBJECT_ORDER:
        obj_dir = output_root / "by_object" / obj
        obj_dir.mkdir(parents=True, exist_ok=True)
        state = build_scope_state_matrix(profiles, [obj], settings.state)
        write_json(state["state_vector_meta"], obj_dir / f"{obj}_object_state_matrix_meta_v10_2.json")
        write_dataframe(state["feature_table"], obj_dir / f"{obj}_object_state_feature_table_v10_2.csv")
        write_dataframe(state["state_feature_scale_before_after"], obj_dir / f"{obj}_object_state_feature_scale_before_after_v10_2.csv")
        write_dataframe(pd.DataFrame({"valid_day_index": state["valid_day_index"]}), obj_dir / f"{obj}_object_state_valid_day_index_v10_2.csv")

        det = run_point_detector(state["state_matrix"][state["valid_day_mask"], :], state["valid_day_index"], settings.detector)
        detector_profile = det["profile"].rename("detector_score").reset_index().rename(columns={"index": "day"})
        detector_profile.insert(0, "object", obj)
        write_dataframe(detector_profile, obj_dir / f"{obj}_object_detector_scores_v10_2.csv")
        primary = det["primary_points_df"].copy(); primary.insert(0, "object", obj)
        local = det["local_peaks_df"].copy(); local.insert(0, "object", obj)
        write_dataframe(primary, obj_dir / f"{obj}_object_ruptures_primary_points_v10_2.csv")
        write_dataframe(local, obj_dir / f"{obj}_object_detector_local_peaks_all_v10_2.csv")

        registry = build_candidate_registry(det["local_peaks_df"], det["primary_points_df"], source_run_tag=f"{obj}_object_native_peak_discovery_v10_2")
        registry.insert(0, "object", obj)
        # Bootstrap functions expect no object prefix in registry IDs; keep a clean copy.
        registry_clean = registry.drop(columns=["object"]).copy()
        boot = run_bootstrap_support_for_scope(profiles, [obj], years, registry_clean, settings, desc=f"V10.2 {obj} object bootstrap")
        boot_summary = add_object_support_class(boot["summary_df"])
        boot_summary.insert(0, "object", obj)
        boot_records = boot["records_df"].copy(); boot_records.insert(0, "object", obj)
        boot_meta = boot["meta_df"].copy(); boot_meta.insert(0, "object", obj)
        write_dataframe(boot_summary, obj_dir / f"{obj}_object_bootstrap_summary_v10_2.csv")
        write_dataframe(boot_records, obj_dir / f"{obj}_object_bootstrap_match_records_v10_2.csv")
        write_dataframe(boot_meta, obj_dir / f"{obj}_object_bootstrap_replicates_v10_2.csv")

        catalog = registry.merge(boot_summary.drop(columns=["object"], errors="ignore"), on=["candidate_id", "point_day"], how="left")
        catalog["object_support_class"] = catalog.get("object_support_class", "")
        # Bands/windows require registry without object but preserve object later.
        bands = build_candidate_point_bands(registry_clean, det["profile"], settings.band)
        bands.insert(0, "object", obj)
        windows, membership = merge_candidate_bands_into_windows(bands.drop(columns=["object"], errors="ignore"), boot_summary.drop(columns=["object"], errors="ignore"), settings.band)
        windows.insert(0, "object", obj)
        membership.insert(0, "object", obj)
        unc_summary, ret_dist = summarize_window_uncertainty(windows.drop(columns=["object"], errors="ignore"), membership.drop(columns=["object"], errors="ignore"), boot["records_df"], boot_summary.drop(columns=["object"], errors="ignore"), settings.uncertainty)
        unc_summary.insert(0, "object", obj)
        ret_dist.insert(0, "object", obj)
        write_dataframe(catalog, obj_dir / f"{obj}_object_candidate_catalog_v10_2.csv")
        write_dataframe(bands, obj_dir / f"{obj}_object_candidate_point_bands_v10_2.csv")
        write_dataframe(windows, obj_dir / f"{obj}_object_derived_windows_v10_2.csv")
        write_dataframe(membership, obj_dir / f"{obj}_object_window_point_membership_v10_2.csv")
        write_dataframe(unc_summary, obj_dir / f"{obj}_object_window_uncertainty_summary_v10_2.csv")
        write_dataframe(ret_dist, obj_dir / f"{obj}_object_window_return_day_distribution_v10_2.csv")

        mapping = map_candidates_to_joint_lineage(obj, catalog, windows, membership, joint_lineage, v10_sel, settings)
        write_dataframe(mapping, obj_dir / f"{obj}_object_candidate_to_joint_lineage_v10_2.csv")
        all_catalogs.append(catalog)
        all_windows.append(windows)
        all_memberships.append(membership)
        all_mappings.append(mapping)
        audit_rows.append({
            "object": obj,
            "status": "success",
            "n_candidates": int(len(catalog)),
            "candidate_days": ";".join(map(str, catalog["point_day"].astype(int).tolist())) if not catalog.empty else "",
            "n_object_windows": int(len(windows)),
            "n_bootstrap_records": int(len(boot_records)),
        })

    all_catalog = pd.concat(all_catalogs, ignore_index=True) if all_catalogs else pd.DataFrame()
    all_window = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()
    all_membership = pd.concat(all_memberships, ignore_index=True) if all_memberships else pd.DataFrame()
    all_mapping = pd.concat(all_mappings, ignore_index=True) if all_mappings else pd.DataFrame()
    summary_df = _write_object_summary(all_catalog, all_window, all_mapping)
    write_dataframe(all_catalog, output_root / "cross_object" / "object_native_candidate_catalog_all_objects_v10_2.csv")
    write_dataframe(all_window, output_root / "cross_object" / "object_native_derived_windows_all_objects_v10_2.csv")
    write_dataframe(all_membership, output_root / "cross_object" / "object_native_window_point_membership_all_objects_v10_2.csv")
    write_dataframe(all_mapping, output_root / "lineage_mapping" / "object_candidate_to_joint_lineage_v10_2.csv")
    write_dataframe(summary_df, output_root / "cross_object" / "object_native_peak_summary_v10_2.csv")
    write_dataframe(pd.DataFrame(audit_rows), output_root / "audit" / "object_native_run_audit_v10_2.csv")

    run_meta = {
        "status": "success",
        "started_at": started,
        "finished_at": now_utc(),
        "output_tag": settings.output_tag,
        "scope": "P/V/H/Je/Jw object-native full-season peak discovery",
        "does_not_import_stage_partition_v6_v6_1_v7_v9": True,
        "does_not_redefine_strict_accepted_windows": True,
        "does_not_perform_sensitivity_testing": True,
        "does_not_perform_physical_interpretation": True,
        "n_objects": len(OBJECT_ORDER),
        "objects": OBJECT_ORDER,
        "n_bootstrap": int(settings.bootstrap.resolved_n_bootstrap()),
        "smoothed_fields_path": str(smoothed_path),
        "joint_lineage_rows_used": int(len(joint_lineage)) if joint_lineage is not None else 0,
        "v10_window_conditioned_selection_rows_used": int(len(v10_sel)) if v10_sel is not None else 0,
    }
    write_json(run_meta, output_root / "run_meta.json")
    write_json({"object_summary": summary_df.to_dict(orient="records"), "run_meta": run_meta}, output_root / "summary.json")
    _copy_v10_2_summary(output_root, summary_df, settings, run_meta)
    (log_root / "last_run.txt").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_root": str(output_root), "run_meta": run_meta, "object_summary": summary_df.to_dict(orient="records")}
