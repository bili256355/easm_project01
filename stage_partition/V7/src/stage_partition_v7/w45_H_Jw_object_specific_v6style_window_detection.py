from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, peak_prominences
except Exception as exc:  # pragma: no cover
    find_peaks = None
    peak_prominences = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

OUTPUT_TAG = "w45_H_Jw_object_specific_v6style_window_detection_v7_w"
VERSION = "v7_w"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
SYSTEM_W45 = (40, 48)
DETECTOR_DAY_RANGE = (0, 70)
OBJECTS = ("H", "Jw")

EPS = 1.0e-12


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    field_key: str
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]


OBJECT_SPECS: dict[str, ObjectSpec] = {
    "H": ObjectSpec("H", "z500_smoothed", (110.0, 140.0), (15.0, 35.0)),
    "Jw": ObjectSpec("Jw", "u200_smoothed", (80.0, 110.0), (25.0, 45.0)),
}


@dataclass(frozen=True)
class DetectorConfig:
    width: int = 20
    model: str = "l2"
    min_size: int = 2
    jump: int = 1
    selection_mode: str = "pen"
    pen: float = 4.0
    fixed_n_bkps: Optional[int] = None
    epsilon: Optional[float] = None
    local_peak_min_distance_days: int = 3
    nearest_peak_search_radius_days: int = 10


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class BootstrapConfig:
    n_bootstrap: int = 1000
    random_seed: int = 42
    strict_match_max_abs_offset_days: int = 2
    match_max_abs_offset_days: int = 5
    near_max_abs_offset_days: int = 8
    progress: bool = True


@dataclass(frozen=True)
class ProfileConfig:
    lat_step_deg: float = 2.0


@dataclass(frozen=True)
class V7WConfig:
    project_root: str = r"D:\easm_project01"
    smoothed_fields_relpath: str = r"foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz"
    output_tag: str = OUTPUT_TAG
    detector_day_range: tuple[int, int] = DETECTOR_DAY_RANGE
    standardization_scope: str = "full_season"
    objects: tuple[str, str] = OBJECTS
    profile: ProfileConfig = ProfileConfig()
    detector: DetectorConfig = DetectorConfig()
    band: BandConfig = BandConfig()
    bootstrap: BootstrapConfig = BootstrapConfig()
    write_figures: bool = True


@dataclass
class PointLayerProfile:
    name: str
    raw_cube: np.ndarray  # years x days x lat_features
    lat_grid: np.ndarray
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]
    empty_slice_mask: np.ndarray


@dataclass
class V7WPaths:
    v7_root: Path
    project_root: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict[str, Any], path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _load_npz(path: Path) -> dict[str, Any]:
    required = ["z500_smoothed", "u200_smoothed", "lat", "lon", "years"]
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"smoothed_fields.npz is missing required keys for V7-w: {missing}")
        return {k: data[k] for k in data.files}


def _resolve_paths(v7_root: Optional[Path], cfg: V7WConfig) -> V7WPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    project_root = Path(cfg.project_root)
    if not project_root.exists():
        # In a copied project tree, infer project root from V7 root.
        project_root = v7_root.parents[1]
    output_dir = v7_root / "outputs" / cfg.output_tag
    log_dir = v7_root / "logs" / cfg.output_tag
    figure_dir = output_dir / "figures"
    for p in (output_dir, log_dir, figure_dir):
        _ensure_dir(p)
    return V7WPaths(v7_root=v7_root, project_root=project_root, output_dir=output_dir, log_dir=log_dir, figure_dir=figure_dir)


# -----------------------------------------------------------------------------
# Profile/state construction copied from the V6/V6_1 method, but object-scoped.
# This is deliberately self-contained to avoid cross-version import failures.
# -----------------------------------------------------------------------------


def _safe_nanmean(a: np.ndarray, axis=None, return_valid_count: bool = False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(a, axis=axis)
    if return_valid_count:
        count = np.sum(np.isfinite(a), axis=axis)
        return mean, count
    return mean


def _mask_between(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = min(float(lower), float(upper)), max(float(lower), float(upper))
    return (arr >= lo) & (arr <= hi)


def _ascending_pair(lat_vals: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    src_lats = np.asarray(lat)[lat_mask]
    subset = field[:, :, lat_mask, :][:, :, :, lon_mask]
    prof, valid_count = _safe_nanmean(subset, axis=-1, return_valid_count=True)
    empty_slice_mask = valid_count == 0
    lo, hi = min(lat_range), max(lat_range)
    dst_lats = np.arange(lo, hi + 1e-9, lat_step_deg)
    prof_interp = _interp_profile_to_grid(prof, src_lats, dst_lats)
    empty_interp = _interp_profile_to_grid(empty_slice_mask.astype(float), src_lats, dst_lats)
    empty_interp = np.where(np.isfinite(empty_interp), empty_interp >= 0.5, True)
    return prof_interp, dst_lats, empty_interp


def _build_object_profile(smoothed: dict[str, Any], spec: ObjectSpec, cfg: V7WConfig) -> PointLayerProfile:
    cube = np.asarray(smoothed[spec.field_key], dtype=float)
    lat = np.asarray(smoothed["lat"], dtype=float)
    lon = np.asarray(smoothed["lon"], dtype=float)
    prof, lat_grid, empty_mask = _build_profile_from_field(
        cube,
        lat,
        lon,
        spec.lon_range,
        spec.lat_range,
        cfg.profile.lat_step_deg,
    )
    return PointLayerProfile(
        name=spec.name,
        raw_cube=prof,
        lat_grid=lat_grid,
        lon_range=spec.lon_range,
        lat_range=spec.lat_range,
        empty_slice_mask=empty_mask,
    )


def _zscore_along_day(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, _ = _safe_nanmean(x, axis=0, return_valid_count=True)
    centered = x - mean[None, :]
    var, _ = _safe_nanmean(np.square(centered), axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < EPS), 1.0, std)
    z = centered / std[None, :]
    return z, mean, std


def _build_object_state(profile: PointLayerProfile, sampled_year_indices: Optional[np.ndarray] = None) -> dict[str, Any]:
    cube = profile.raw_cube if sampled_year_indices is None else profile.raw_cube[np.asarray(sampled_year_indices, dtype=int), :, :]
    seasonal, valid_count = _safe_nanmean(cube, axis=0, return_valid_count=True)  # day x lat_feature
    raw = np.asarray(seasonal, dtype=float)
    z, feat_mean, feat_std = _zscore_along_day(raw)
    width = z.shape[1]
    factor = 1.0 / np.sqrt(max(width, 1))
    scaled = z * factor
    valid_day_mask = np.all(np.isfinite(scaled), axis=1)
    valid_day_index = np.where(valid_day_mask)[0].astype(int)
    feature_rows = []
    for j, latv in enumerate(profile.lat_grid):
        feature_rows.append(
            {
                "object": profile.name,
                "feature_index": int(j),
                "lat_value": float(latv),
                "raw_mean_day": float(feat_mean[j]) if np.isfinite(feat_mean[j]) else np.nan,
                "raw_std_day": float(feat_std[j]) if np.isfinite(feat_std[j]) else np.nan,
                "block_equal_weight": float(factor),
                "n_valid_days_raw": int(np.sum(np.isfinite(raw[:, j]))),
            }
        )
    return {
        "raw_matrix": raw,
        "state_matrix": scaled,
        "valid_day_mask": valid_day_mask,
        "valid_day_index": valid_day_index,
        "feature_table": pd.DataFrame(feature_rows),
        "sampled_year_indices": sampled_year_indices,
    }


def _subset_state_by_day_range(state: dict[str, Any], day_range: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = int(day_range[0]), int(day_range[1])
    valid_days = np.asarray(state["valid_day_index"], dtype=int)
    mask = (valid_days >= lo) & (valid_days <= hi)
    selected_days = valid_days[mask]
    selected_matrix = np.asarray(state["state_matrix"])[selected_days, :]
    finite_row = np.all(np.isfinite(selected_matrix), axis=1)
    return selected_matrix[finite_row, :], selected_days[finite_row]


# -----------------------------------------------------------------------------
# Detector and peak/band/window machinery copied from V6/V6_1 semantics.
# -----------------------------------------------------------------------------


def _import_ruptures():
    try:
        import ruptures as rpt
        return rpt
    except Exception as exc:  # pragma: no cover
        raise ImportError("V7-w requires the ruptures package in the user's environment.") from exc


def _day_index_to_month_day(day: int) -> str:
    # day 0 = Apr 1, with months Apr-Sep for this project window.
    month_lengths = [(4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]
    d = int(day)
    for month, length in month_lengths:
        if d < length:
            return f"{month:02d}-{d + 1:02d}"
        d -= length
    return f"day{int(day)}"


def _map_breakpoints_to_days(points_local: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None:
        return points_local.astype(int)
    arr = np.asarray(day_index, dtype=int)
    return pd.Series(
        [int(arr[max(0, min(len(arr) - 1, int(p) - 1))]) for p in points_local.astype(int).tolist()],
        name="changepoint",
        dtype=int,
    )


def _map_profile_index(profile: pd.Series, day_index: np.ndarray | None) -> pd.Series:
    if day_index is None or profile.empty:
        return profile
    arr = np.asarray(day_index, dtype=int)
    mapped_idx: list[int] = []
    for i in profile.index.to_numpy(dtype=int):
        idx = max(0, min(len(arr) - 1, int(i)))
        mapped_idx.append(int(arr[idx]))
    out = profile.copy()
    out.index = np.asarray(mapped_idx, dtype=int)
    return out


def _extract_ranked_local_peaks(profile: pd.Series, min_distance_days: int, prominence_min: float = 0.0) -> pd.DataFrame:
    cols = ["peak_id", "peak_day", "peak_score", "peak_prominence", "peak_rank", "source_type"]
    if _SCIPY_IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError("V7-w requires scipy.signal for local peak extraction.") from _SCIPY_IMPORT_ERROR
    if profile is None or profile.empty:
        return pd.DataFrame(columns=cols)
    s = profile.sort_index().astype(float).fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=max(1, int(min_distance_days)))
    if peaks.size == 0:
        return pd.DataFrame(columns=cols)
    prominences, _, _ = peak_prominences(values, peaks)
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


def _run_ruptures_window(state_matrix: np.ndarray, cfg: DetectorConfig, day_index: np.ndarray | None = None) -> dict[str, Any]:
    rpt = _import_ruptures()
    signal = np.asarray(state_matrix, dtype=float)
    if signal.shape[0] < max(2 * int(cfg.width), 3):
        return {
            "points": pd.Series(dtype=int, name="changepoint"),
            "profile": pd.Series(dtype=float, name="profile"),
            "points_local": pd.Series(dtype=int, name="changepoint"),
            "profile_local": pd.Series(dtype=float, name="profile"),
            "status": "skipped_insufficient_valid_days",
        }
    algo = rpt.Window(width=int(cfg.width), model=cfg.model, min_size=int(cfg.min_size), jump=int(cfg.jump)).fit(signal)
    if cfg.selection_mode == "fixed_n_bkps":
        if cfg.fixed_n_bkps is None:
            raise ValueError("fixed_n_bkps must be set when selection_mode=fixed_n_bkps")
        bkps = algo.predict(n_bkps=int(cfg.fixed_n_bkps))
    elif cfg.selection_mode == "pen":
        bkps = algo.predict(pen=float(cfg.pen))
    elif cfg.selection_mode == "epsilon":
        if cfg.epsilon is None:
            raise ValueError("epsilon must be set when selection_mode=epsilon")
        bkps = algo.predict(epsilon=float(cfg.epsilon))
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
    return {"points": points, "profile": profile, "points_local": points_local, "profile_local": profile_raw, "status": "success"}


def _build_primary_points_table(points: pd.Series, profile: pd.Series, local_peaks_df: pd.DataFrame | None, cfg: DetectorConfig) -> pd.DataFrame:
    cols = ["point_id", "point_day", "month_day", "peak_score", "source_type", "raw_point_day", "matched_peak_day"]
    if points is None or len(points) == 0:
        return pd.DataFrame(columns=cols)
    rows = []
    for i, raw_point_day in enumerate(sorted(int(x) for x in pd.Series(points).astype(int).tolist()), start=1):
        point_day = _nearest_peak_day(local_peaks_df, raw_point_day, int(cfg.nearest_peak_search_radius_days)) if local_peaks_df is not None else int(raw_point_day)
        rows.append(
            {
                "point_id": f"RP{i:03d}",
                "point_day": int(point_day),
                "month_day": _day_index_to_month_day(int(point_day)),
                "peak_score": float(profile.get(int(point_day), np.nan)) if profile is not None and not profile.empty else np.nan,
                "source_type": "formal_primary",
                "raw_point_day": int(raw_point_day),
                "matched_peak_day": int(point_day),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _run_object_detector(state: dict[str, Any], cfg: V7WConfig) -> dict[str, Any]:
    matrix, day_index = _subset_state_by_day_range(state, tuple(cfg.detector_day_range))
    out = _run_ruptures_window(matrix, cfg.detector, day_index=day_index)
    local_peaks = _extract_ranked_local_peaks(
        out["profile"],
        min_distance_days=int(cfg.detector.local_peak_min_distance_days),
        prominence_min=0.0,
    )
    primary = _build_primary_points_table(out["points"], out["profile"], local_peaks, cfg.detector)
    return {**out, "local_peaks_df": local_peaks, "primary_points_df": primary, "detector_day_index": day_index}


def _build_candidate_registry(local_peaks_df: pd.DataFrame, primary_points_df: pd.DataFrame | None, source_run_tag: str) -> pd.DataFrame:
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
    primary_days: set[int] = set()
    if primary_points_df is not None and not primary_points_df.empty:
        primary_days = set(pd.to_numeric(primary_points_df["point_day"], errors="coerce").dropna().astype(int).tolist())
    primary_list = sorted(primary_days)
    rows = []
    for i, row in peaks.iterrows():
        day = int(row["peak_day"])
        nearest_primary_day = int(min(primary_list, key=lambda x: abs(x - day))) if primary_list else np.nan
        rows.append(
            {
                "candidate_id": f"CP{i + 1:03d}",
                "point_day": day,
                "month_day": _day_index_to_month_day(day),
                "registry_rank": int(i + 1),
                "source_run_tag": source_run_tag,
                "peak_score": float(row["peak_score"]) if pd.notna(row["peak_score"]) else np.nan,
                "peak_prominence": float(row["peak_prominence"]) if pd.notna(row["peak_prominence"]) else np.nan,
                "is_formal_primary": bool(day in primary_days),
                "nearest_primary_day": nearest_primary_day,
            }
        )
    return pd.DataFrame(rows, columns=cols)


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


def _build_candidate_point_bands(registry_df: pd.DataFrame, profile: pd.Series, cfg: BandConfig) -> pd.DataFrame:
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
    rows = []
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


def _classify_match_type(abs_offset_days: int | float | None, cfg: BootstrapConfig) -> str:
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


def _match_candidates_to_local_peaks(
    registry_df: pd.DataFrame,
    local_peaks_df: pd.DataFrame,
    cfg: BootstrapConfig,
    *,
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
            match_type = _classify_match_type(abs_offset_days, cfg)
        rows.append(
            {
                "replicate_id": int(replicate_id),
                "replicate_kind": str(replicate_kind),
                "replicate_label": replicate_label,
                "candidate_id": str(ref["candidate_id"]),
                "point_day": target_day,
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


def _summarize_match_records(records_df: pd.DataFrame, point_day_map: dict[str, int]) -> pd.DataFrame:
    cols = [
        "candidate_id",
        "point_day",
        "bootstrap_strict_fraction",
        "bootstrap_match_fraction",
        "bootstrap_near_fraction",
        "bootstrap_no_match_fraction",
        "median_bootstrap_matched_peak_score",
        "median_bootstrap_abs_offset_days",
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
                "point_day": int(point_day_map.get(str(cid), -9999)),
                "bootstrap_strict_fraction": float((sub["match_type"] == "strict").mean()),
                "bootstrap_match_fraction": float(sub["match_type"].isin(["strict", "matched"]).mean()),
                "bootstrap_near_fraction": float((sub["match_type"] == "near").mean()),
                "bootstrap_no_match_fraction": float((sub["match_type"] == "no_match").mean()),
                "median_bootstrap_matched_peak_score": float(matched_score.median()) if matched_score.notna().any() else np.nan,
                "median_bootstrap_abs_offset_days": float(abs_off.median()) if abs_off.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values(["point_day", "candidate_id"]).reset_index(drop=True)


def _summarize_return_days(records_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "candidate_id",
        "point_day",
        "n_matched_or_near",
        "return_day_median",
        "return_day_q02_5",
        "return_day_q10",
        "return_day_q90",
        "return_day_q97_5",
        "return_day_width80",
        "return_day_width95",
    ]
    if records_df is None or records_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for cid, sub in records_df.groupby("candidate_id"):
        matched = sub[sub["match_type"].isin(["strict", "matched", "near"])].copy()
        days = pd.to_numeric(matched["matched_peak_day"], errors="coerce").dropna().to_numpy(dtype=float)
        point_day = int(sub["point_day"].iloc[0])
        if days.size:
            q025, q10, q50, q90, q975 = np.nanquantile(days, [0.025, 0.10, 0.50, 0.90, 0.975])
            rows.append(
                {
                    "candidate_id": str(cid),
                    "point_day": point_day,
                    "n_matched_or_near": int(days.size),
                    "return_day_median": float(q50),
                    "return_day_q02_5": float(q025),
                    "return_day_q10": float(q10),
                    "return_day_q90": float(q90),
                    "return_day_q97_5": float(q975),
                    "return_day_width80": float(q90 - q10),
                    "return_day_width95": float(q975 - q025),
                }
            )
        else:
            rows.append(
                {
                    "candidate_id": str(cid),
                    "point_day": point_day,
                    "n_matched_or_near": 0,
                    "return_day_median": np.nan,
                    "return_day_q02_5": np.nan,
                    "return_day_q10": np.nan,
                    "return_day_q90": np.nan,
                    "return_day_q97_5": np.nan,
                    "return_day_width80": np.nan,
                    "return_day_width95": np.nan,
                }
            )
    return pd.DataFrame(rows, columns=cols).sort_values(["point_day", "candidate_id"]).reset_index(drop=True)


def _merge_candidate_bands_into_windows(bands_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame | None, cfg: BandConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    current_group_max_end: Optional[int] = None
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
        is_protected = float(boot.get(row["candidate_id"], np.nan)) >= sig_thr if row["candidate_id"] in boot else False
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
        sub["bootstrap_match_fraction_5d"] = sub["candidate_id"].map(lambda x: float(boot.get(x, np.nan)))
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


# -----------------------------------------------------------------------------
# Run logic
# -----------------------------------------------------------------------------


def _progress_iter(items, enabled: bool, desc: str):
    if enabled:
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc)
        except Exception:
            return items
    return items


def _build_object_outputs(obj: str, profile: PointLayerProfile, years: np.ndarray, cfg: V7WConfig) -> dict[str, pd.DataFrame | dict[str, Any]]:
    state = _build_object_state(profile)
    det = _run_object_detector(state, cfg)
    profile_df = det["profile"].rename_axis("day").reset_index(name="profile_score") if not det["profile"].empty else pd.DataFrame(columns=["day", "profile_score"])
    profile_df.insert(0, "object", obj)
    local_peaks = det["local_peaks_df"].copy()
    if not local_peaks.empty:
        local_peaks.insert(0, "object", obj)
    primary = det["primary_points_df"].copy()
    if not primary.empty:
        primary.insert(0, "object", obj)
    registry = _build_candidate_registry(det["local_peaks_df"], det["primary_points_df"], source_run_tag=f"{OUTPUT_TAG}_{obj}")
    if not registry.empty:
        registry.insert(0, "object", obj)
    feature_table = state["feature_table"].copy()
    # _build_object_state already records the object name in feature_table.
    # Keep this guard so reruns/patches do not fail with duplicate-column inserts.
    if "object" not in feature_table.columns:
        feature_table.insert(0, "object", obj)
    else:
        # Preserve column order with object first for downstream CSV readability.
        feature_table = feature_table[["object"] + [c for c in feature_table.columns if c != "object"]]

    # Bootstrap against observed candidate registry.
    rng = np.random.default_rng(int(cfg.bootstrap.random_seed))
    years_arr = np.asarray(years)
    n_years = int(len(years_arr))
    boot_records = []
    boot_meta_rows = []
    obs_registry_for_match = registry.drop(columns=["object"]) if "object" in registry.columns else registry.copy()
    point_day_map = {str(r["candidate_id"]): int(r["point_day"]) for _, r in obs_registry_for_match.iterrows()}
    debug_n = os.environ.get("V7W_DEBUG_N_BOOTSTRAP")
    n_boot = int(debug_n) if debug_n else int(cfg.bootstrap.n_bootstrap)
    progress_enabled = bool(cfg.bootstrap.progress) and not bool(debug_n)
    for rep in _progress_iter(range(n_boot), progress_enabled, f"V7-w {obj} bootstrap"):
        sampled = rng.integers(0, n_years, size=n_years, endpoint=False)
        try:
            b_state = _build_object_state(profile, sampled)
            b_matrix, b_day_index = _subset_state_by_day_range(b_state, tuple(cfg.detector_day_range))
            if b_matrix.shape[0] < max(2 * int(cfg.detector.width), 3):
                boot_meta_rows.append({"object": obj, "replicate_id": int(rep), "status": "skipped_insufficient_valid_days", "n_valid_days": int(b_matrix.shape[0]), "sampled_year_indices": sampled.tolist()})
                continue
            b_det = _run_object_detector(b_state, cfg)
            rec = _match_candidates_to_local_peaks(
                obs_registry_for_match,
                b_det["local_peaks_df"],
                cfg.bootstrap,
                replicate_id=int(rep),
                replicate_kind="bootstrap",
                replicate_label=int(rep),
            )
            if not rec.empty:
                rec.insert(0, "object", obj)
                boot_records.append(rec)
            boot_meta_rows.append({"object": obj, "replicate_id": int(rep), "status": "success", "n_valid_days": int(b_matrix.shape[0]), "sampled_year_indices": sampled.tolist()})
        except Exception as exc:  # keep a record rather than hiding failed replicates
            boot_meta_rows.append({"object": obj, "replicate_id": int(rep), "status": "error", "error": repr(exc), "sampled_year_indices": sampled.tolist()})

    records_df = pd.concat(boot_records, ignore_index=True) if boot_records else pd.DataFrame(
        columns=["object", "replicate_id", "replicate_kind", "replicate_label", "candidate_id", "point_day", "month_day", "matched_peak_day", "matched_peak_score", "matched_peak_prominence", "offset_days", "abs_offset_days", "match_type"]
    )
    summary = _summarize_match_records(records_df.drop(columns=["object"]) if "object" in records_df.columns else records_df, point_day_map)
    if not summary.empty:
        summary.insert(0, "object", obj)
    dist = _summarize_return_days(records_df.drop(columns=["object"]) if "object" in records_df.columns else records_df)
    if not dist.empty:
        dist.insert(0, "object", obj)
    boot_meta = pd.DataFrame(boot_meta_rows)

    # Bands/windows using bootstrap summary.
    bands = _build_candidate_point_bands(obs_registry_for_match, det["profile"], cfg.band)
    if not bands.empty:
        bands.insert(0, "object", obj)
    windows, membership = _merge_candidate_bands_into_windows(bands.drop(columns=["object"]) if "object" in bands.columns else bands, summary.drop(columns=["object"]) if "object" in summary.columns else summary, cfg.band)
    if not windows.empty:
        windows.insert(0, "object", obj)
        windows["window_status"] = windows["max_member_bootstrap_match_fraction"].map(_classify_window_status)
    if not membership.empty:
        membership.insert(0, "object", obj)

    return {
        "profile_df": profile_df,
        "local_peaks_df": local_peaks,
        "primary_points_df": primary,
        "candidate_registry_df": registry,
        "feature_table_df": feature_table,
        "bootstrap_records_df": records_df,
        "bootstrap_summary_df": summary,
        "bootstrap_return_days_df": dist,
        "bootstrap_meta_df": boot_meta,
        "candidate_bands_df": bands,
        "windows_df": windows,
        "membership_df": membership,
        "detector_meta": {
            "object": obj,
            "n_features": int(state["state_matrix"].shape[1]),
            "n_valid_full_season_days": int(state["valid_day_index"].size),
            "n_detector_days": int(len(det["detector_day_index"])),
            "detector_status": det.get("status"),
            "n_local_peaks": int(len(local_peaks)),
            "n_primary_points": int(len(primary)),
            "n_candidates": int(len(registry)),
        },
    }


def _classify_window_status(x: float) -> str:
    if not np.isfinite(x):
        return "no_bootstrap_summary"
    if float(x) >= 0.95:
        return "accepted_object_window_95"
    if float(x) >= 0.80:
        return "candidate_object_window_80"
    if float(x) >= 0.50:
        return "weak_object_window_50"
    return "unstable_object_window"


def _build_vs_w45_table(windows_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "object",
        "window_id",
        "object_peak_day",
        "object_window_start",
        "object_window_end",
        "overlap_with_system_W45_day40_48",
        "overlap_days_with_W45",
        "is_pre_W45_peak",
        "is_within_W45_peak",
        "is_post_W45_peak",
        "bootstrap_match_fraction",
        "window_status",
        "interpretation",
    ]
    if windows_df is None or windows_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    w45_start, w45_end = SYSTEM_W45
    for _, r in windows_df.iterrows():
        start, end, peak = int(r["start_day"]), int(r["end_day"]), int(r["main_peak_day"])
        overlap = max(0, min(end, w45_end) - max(start, w45_start) + 1)
        if peak < w45_start:
            interp = "object_peak_before_system_W45"
        elif w45_start <= peak <= w45_end:
            interp = "object_peak_within_system_W45"
        else:
            interp = "object_peak_after_system_W45"
        rows.append(
            {
                "object": r["object"],
                "window_id": r["window_id"],
                "object_peak_day": peak,
                "object_window_start": start,
                "object_window_end": end,
                "overlap_with_system_W45_day40_48": bool(overlap > 0),
                "overlap_days_with_W45": int(overlap),
                "is_pre_W45_peak": bool(peak < w45_start),
                "is_within_W45_peak": bool(w45_start <= peak <= w45_end),
                "is_post_W45_peak": bool(peak > w45_end),
                "bootstrap_match_fraction": float(r["max_member_bootstrap_match_fraction"]) if pd.notna(r.get("max_member_bootstrap_match_fraction")) else np.nan,
                "window_status": r.get("window_status"),
                "interpretation": interp,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _build_summary_json(outputs: dict[str, pd.DataFrame], detector_meta: list[dict[str, Any]], cfg: V7WConfig) -> dict[str, Any]:
    windows = outputs.get("windows", pd.DataFrame())
    vs = outputs.get("vs_w45", pd.DataFrame())
    by_obj: dict[str, Any] = {}
    for obj in cfg.objects:
        subw = windows[windows["object"] == obj] if not windows.empty else pd.DataFrame()
        subv = vs[vs["object"] == obj] if not vs.empty else pd.DataFrame()
        by_obj[obj] = {
            "n_windows": int(len(subw)),
            "n_accepted_95_windows": int((subw.get("window_status", pd.Series(dtype=str)) == "accepted_object_window_95").sum()) if not subw.empty else 0,
            "main_peak_days": subw["main_peak_day"].astype(int).tolist() if not subw.empty else [],
            "pre_W45_peak_days": subv.loc[subv["is_pre_W45_peak"], "object_peak_day"].astype(int).tolist() if not subv.empty else [],
            "within_W45_peak_days": subv.loc[subv["is_within_W45_peak"], "object_peak_day"].astype(int).tolist() if not subv.empty else [],
            "post_W45_peak_days": subv.loc[subv["is_post_W45_peak"], "object_peak_day"].astype(int).tolist() if not subv.empty else [],
        }
    return {
        "version": VERSION,
        "output_tag": cfg.output_tag,
        "primary_goal": "reuse V6_1/W45 window detector skeleton with object_scope=H/Jw and detector_day_range=day0-70",
        "system_W45_reference": {"window_id": WINDOW_ID, "anchor_day": ANCHOR_DAY, "accepted_window": list(SYSTEM_W45)},
        "detector_day_range": list(cfg.detector_day_range),
        "objects": list(cfg.objects),
        "object_summaries": by_obj,
        "detector_meta": detector_meta,
    }


def _build_markdown_summary(summary: dict[str, Any], cfg: V7WConfig) -> str:
    lines = []
    lines.append("# V7-w H/Jw object-specific V6_1-style window detection")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Reuse the original W45 detection skeleton while changing only object scope and detector time range.")
    lines.append("")
    lines.append("## What was changed relative to system W45")
    lines.append("- object_scope: all-object system matrix -> H-only and Jw-only runs")
    lines.append("- detector_day_range: day0-70")
    lines.append("")
    lines.append("## What was not changed")
    lines.append("- profile representation: 2-degree latitudinal profiles")
    lines.append("- full-season z-score standardization before day0-70 detection")
    lines.append("- ruptures.Window detector parameters")
    lines.append("- local peak, band, window merge, and bootstrap support logic")
    lines.append("")
    lines.append("## System W45 reference")
    lines.append(f"- {WINDOW_ID}: day{SYSTEM_W45[0]}-day{SYSTEM_W45[1]}, anchor day {ANCHOR_DAY}")
    lines.append("")
    lines.append("## Object summaries")
    for obj, s in summary.get("object_summaries", {}).items():
        lines.append(f"### {obj}")
        lines.append(f"- n_windows: {s.get('n_windows')}")
        lines.append(f"- n_accepted_95_windows: {s.get('n_accepted_95_windows')}")
        lines.append(f"- main_peak_days: {s.get('main_peak_days')}")
        lines.append(f"- pre-W45 peak days: {s.get('pre_W45_peak_days')}")
        lines.append(f"- within-W45 peak days: {s.get('within_W45_peak_days')}")
        lines.append(f"- post-W45 peak days: {s.get('post_W45_peak_days')}")
        lines.append("")
    lines.append("## Interpretation rule")
    lines.append("This run identifies object-level detector peaks/windows. It does not infer causality, physical mechanism, or H/Jw pathway order.")
    return "\n".join(lines)


def _try_write_figures(outputs: dict[str, pd.DataFrame], paths: V7WPaths) -> None:
    if os.environ.get("V7W_SKIP_FIGURES") == "1":
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    profile = outputs.get("profile", pd.DataFrame())
    peaks = outputs.get("local_peaks", pd.DataFrame())
    if profile is None or profile.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for obj, sub in profile.groupby("object"):
        ax.plot(sub["day"], sub["profile_score"], label=f"{obj} detector profile")
    ax.axvspan(SYSTEM_W45[0], SYSTEM_W45[1], alpha=0.15, label="system W45 day40-48")
    if peaks is not None and not peaks.empty:
        for _, r in peaks.iterrows():
            ax.scatter(int(r["peak_day"]), float(r["peak_score"]), s=18)
    ax.set_xlabel("day index")
    ax.set_ylabel("ruptures.Window profile score")
    ax.set_title("H/Jw object-specific V6_1-style detector profiles, day0-70")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(paths.figure_dir / "H_Jw_object_detector_profiles_v7_w.png", dpi=180)
    plt.close(fig)

    windows = outputs.get("windows", pd.DataFrame())
    if windows is not None and not windows.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ymap = {obj: i for i, obj in enumerate(sorted(windows["object"].unique()))}
        for _, r in windows.iterrows():
            y = ymap[r["object"]]
            ax.plot([r["start_day"], r["end_day"]], [y, y], linewidth=6, solid_capstyle="butt")
            ax.scatter([r["main_peak_day"]], [y], s=40)
            ax.text(r["main_peak_day"], y + 0.08, str(r["window_id"]), ha="center", va="bottom", fontsize=8)
        ax.axvspan(SYSTEM_W45[0], SYSTEM_W45[1], alpha=0.15)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(list(ymap.keys()))
        ax.set_xlabel("day index")
        ax.set_title("H/Jw object-specific candidate windows vs system W45")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "H_Jw_object_windows_vs_W45_v7_w.png", dpi=180)
        plt.close(fig)


def run_w45_H_Jw_object_specific_v6style_window_detection_v7_w(v7_root: Optional[Path] = None) -> dict[str, Any]:
    cfg = V7WConfig()
    debug_n = os.environ.get("V7W_DEBUG_N_BOOTSTRAP")
    if debug_n:
        cfg = V7WConfig(bootstrap=BootstrapConfig(n_bootstrap=int(debug_n), progress=False))
    started_at = _now_iso()
    paths = _resolve_paths(v7_root, cfg)
    smoothed_path = paths.project_root / Path(*str(cfg.smoothed_fields_relpath).replace("\\", "/").split("/"))
    if not smoothed_path.exists():
        raise FileNotFoundError(f"Missing smoothed fields file: {smoothed_path}")
    smoothed = _load_npz(smoothed_path)
    years = np.asarray(smoothed["years"])

    all_outputs: dict[str, list[pd.DataFrame]] = {
        "profile": [],
        "local_peaks": [],
        "primary_points": [],
        "candidate_registry": [],
        "feature_table": [],
        "bootstrap_records": [],
        "bootstrap_summary": [],
        "bootstrap_return_days": [],
        "bootstrap_meta": [],
        "candidate_bands": [],
        "windows": [],
        "membership": [],
    }
    detector_meta: list[dict[str, Any]] = []

    for obj in cfg.objects:
        spec = OBJECT_SPECS[obj]
        profile = _build_object_profile(smoothed, spec, cfg)
        out = _build_object_outputs(obj, profile, years, cfg)
        detector_meta.append(out["detector_meta"])
        for key, target_key in [
            ("profile_df", "profile"),
            ("local_peaks_df", "local_peaks"),
            ("primary_points_df", "primary_points"),
            ("candidate_registry_df", "candidate_registry"),
            ("feature_table_df", "feature_table"),
            ("bootstrap_records_df", "bootstrap_records"),
            ("bootstrap_summary_df", "bootstrap_summary"),
            ("bootstrap_return_days_df", "bootstrap_return_days"),
            ("bootstrap_meta_df", "bootstrap_meta"),
            ("candidate_bands_df", "candidate_bands"),
            ("windows_df", "windows"),
            ("membership_df", "membership"),
        ]:
            df = out.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                all_outputs[target_key].append(df)

    combined: dict[str, pd.DataFrame] = {}
    for key, dfs in all_outputs.items():
        combined[key] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    combined["vs_w45"] = _build_vs_w45_table(combined["windows"])

    # Write outputs.
    _write_csv(combined["profile"], paths.output_dir / "H_Jw_object_detector_profile_v7_w.csv")
    _write_csv(combined["local_peaks"], paths.output_dir / "H_Jw_object_local_peaks_v7_w.csv")
    _write_csv(combined["primary_points"], paths.output_dir / "H_Jw_object_primary_points_v7_w.csv")
    _write_csv(combined["candidate_registry"], paths.output_dir / "H_Jw_object_candidate_registry_v7_w.csv")
    _write_csv(combined["candidate_bands"], paths.output_dir / "H_Jw_object_candidate_bands_v7_w.csv")
    _write_csv(combined["windows"], paths.output_dir / "H_Jw_object_windows_registry_v7_w.csv")
    _write_csv(combined["membership"], paths.output_dir / "H_Jw_object_window_point_membership_v7_w.csv")
    _write_csv(combined["bootstrap_summary"], paths.output_dir / "H_Jw_object_peak_bootstrap_summary_v7_w.csv")
    _write_csv(combined["bootstrap_return_days"], paths.output_dir / "H_Jw_object_peak_return_day_distribution_v7_w.csv")
    _write_csv(combined["bootstrap_records"], paths.output_dir / "H_Jw_object_peak_bootstrap_match_records_v7_w.csv")
    _write_csv(combined["bootstrap_meta"], paths.output_dir / "H_Jw_object_bootstrap_meta_v7_w.csv")
    _write_csv(combined["feature_table"], paths.output_dir / "H_Jw_object_feature_table_v7_w.csv")
    _write_csv(combined["vs_w45"], paths.output_dir / "H_Jw_object_windows_vs_system_W45_v7_w.csv")
    _write_csv(pd.DataFrame(detector_meta), paths.output_dir / "H_Jw_object_detector_meta_v7_w.csv")

    config_dict = asdict(cfg)
    _write_json(config_dict, paths.output_dir / "config_used.json")
    summary = _build_summary_json(combined, detector_meta, cfg)
    _write_json(summary, paths.output_dir / "summary.json")
    _write_text(_build_markdown_summary(summary, cfg), paths.output_dir / "H_Jw_object_specific_v6style_window_detection_summary_v7_w.md")
    run_meta = {
        "status": "success",
        "started_at": started_at,
        "ended_at": _now_iso(),
        "version": VERSION,
        "output_tag": cfg.output_tag,
        "primary_goal": "object-specific V6_1-style window detection for H and Jw over day0-70",
        "system_W45_reference": {"window_id": WINDOW_ID, "anchor_day": ANCHOR_DAY, "accepted_window": list(SYSTEM_W45)},
        "method_reuse": "V6_1/W45 profile-state ruptures.Window + local peaks + candidate bands + bootstrap matching",
        "changed_parameters_only": {
            "object_scope": list(cfg.objects),
            "detector_day_range": list(cfg.detector_day_range),
        },
        "unchanged_core_parameters": {
            "profile_lat_step_deg": cfg.profile.lat_step_deg,
            "standardization_scope": cfg.standardization_scope,
            "detector": asdict(cfg.detector),
            "band": asdict(cfg.band),
            "bootstrap": asdict(cfg.bootstrap),
        },
        "notes": [
            "This is not a pre/post progress run.",
            "This is not a raw025 2D field-speed run.",
            "This run intentionally reuses the original W45 detector skeleton and changes only object/time scope.",
            "Object windows are compared to system W45 day40-48 only after detection.",
        ],
    }
    _write_json(run_meta, paths.output_dir / "run_meta.json")
    _try_write_figures(combined, paths)
    return {"output_root": paths.output_dir, "summary": summary}


__all__ = ["run_w45_H_Jw_object_specific_v6style_window_detection_v7_w"]
