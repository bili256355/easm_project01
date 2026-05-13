from __future__ import annotations

"""
V10.5_a field/profile + low-dimensional index validation.

This module reads the established V10 lineage/timing outputs and the foundation
smoothed fields, then performs a first external-validation layer for selected
windows. It does not rerun peak discovery, does not redefine accepted windows,
and does not make physical/causal claims.

Scope:
    - W045, W113, W160 only.
    - Objects: P, V, H, Je, Jw.
    - Profile rolling pre/post contrast validation.
    - Simple low-dimensional profile-metric timing validation.
    - Order-level comparison against V10.4 object-order skeleton.

Out of scope:
    - Cartopy map production in this first patch.
    - Yearwise order support.
    - Alternative detector model comparison.
    - Cross-dataset validation.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import shutil
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


OBJECTS = ["P", "V", "H", "Je", "Jw"]
TARGET_LINEAGES = ["W045", "W113", "W160"]
PRIMARY_K = 9
K_VALUES = (7, 9, 11)
ENERGY_TOP_K = 5
ENERGY_TOPK_MIN_DISTANCE_DAYS = 3
TAU_FOR_ORDER = 2
SUPPORT_DAYS = 5
NEAR_DAYS = 8


@dataclass
class FoundationInputConfig:
    project_root: Path = Path(r"D:\easm_project01")
    foundation_layer: str = "foundation"
    foundation_version: str = "V1"
    preprocess_output_tag: str = "baseline_a"

    def smoothed_fields_path(self) -> Path:
        env = os.environ.get("V10_5_SMOOTHED_FIELDS")
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
class ValidationConfig:
    target_lineages: tuple[str, ...] = tuple(TARGET_LINEAGES)
    window_half_width_days: int = 25
    k_values: tuple[int, ...] = K_VALUES
    primary_k: int = PRIMARY_K
    support_days: int = SUPPORT_DAYS
    near_days: int = NEAR_DAYS
    order_tau_days: int = TAU_FOR_ORDER
    emit_figures: bool = True
    energy_top_k: int = ENERGY_TOP_K
    energy_topk_min_distance_days: int = ENERGY_TOPK_MIN_DISTANCE_DAYS
    # V10.5_d strength-stability audit.
    # DEBUG_N takes precedence when provided, so users can run a quick check first.
    profile_energy_bootstrap_n: int = int(os.environ.get("V10_5_DEBUG_N_BOOTSTRAP", os.environ.get("V10_5_N_BOOTSTRAP", "1000")))
    profile_energy_bootstrap_seed: int = int(os.environ.get("V10_5_BOOTSTRAP_SEED", "42"))
    profile_energy_bootstrap_top_k: int = ENERGY_TOP_K


@dataclass
class Settings:
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output_tag: str = "field_index_validation_v10_5_a"

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


@dataclass
class ObjectProfile:
    object_name: str
    raw_cube: np.ndarray  # year x day x lat
    seasonal: np.ndarray  # day x lat
    z_seasonal: np.ndarray  # day x lat
    lat_grid: np.ndarray
    lon_range: tuple[float, float]
    lat_range: tuple[float, float]


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
    for sub in ["profile_validation", "index_validation", "order_validation", "figures", "audit"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)


def day_index_to_month_day(day_index: int) -> str:
    mdays = [(4, 30), (5, 31), (6, 30), (7, 31), (8, 31), (9, 30)]
    d = int(day_index)
    for month, nday in mdays:
        if d < nday:
            return f"{month:02d}-{d + 1:02d}"
        d -= nday
    return f"overflow+{d}"


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


def load_smoothed_fields(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields.npz: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


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


def _zscore_along_day(x: np.ndarray) -> np.ndarray:
    mean, _ = safe_nanmean(x, axis=0, return_valid_count=True)
    centered = x - mean[None, :]
    var, _ = safe_nanmean(np.square(centered), axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    return centered / std[None, :]


def _build_profile_from_field(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    lat_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat_mask = _mask_between(lat, *lat_range)
    lon_mask = _mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points in requested range {lat_range}")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points in requested range {lon_range}")
    src_lats = lat[lat_mask]
    subset = field[:, :, lat_mask, :][:, :, :, lon_mask]
    prof, _ = safe_nanmean(subset, axis=-1, return_valid_count=True)  # year x day x lat
    lo, hi = min(lat_range), max(lat_range)
    dst_lats = np.arange(lo, hi + 1e-9, lat_step_deg)
    prof_interp = _interp_profile_to_grid(prof, src_lats, dst_lats)
    return prof_interp, dst_lats


def build_profiles(smoothed: dict[str, np.ndarray], cfg: ProfileGridConfig) -> dict[str, ObjectProfile]:
    lat = smoothed["lat"]
    lon = smoothed["lon"]
    specs = {
        "P": ("precip_smoothed", cfg.p_lon_range, cfg.p_lat_range),
        "V": ("v850_smoothed", cfg.v_lon_range, cfg.v_lat_range),
        "H": ("z500_smoothed", cfg.h_lon_range, cfg.h_lat_range),
        "Je": ("u200_smoothed", cfg.je_lon_range, cfg.je_lat_range),
        "Jw": ("u200_smoothed", cfg.jw_lon_range, cfg.jw_lat_range),
    }
    profiles: dict[str, ObjectProfile] = {}
    for obj, (field_key, lon_range, lat_range) in specs.items():
        cube, lat_grid = _build_profile_from_field(smoothed[field_key], lat, lon, lon_range, lat_range, cfg.lat_step_deg)
        seasonal, _ = safe_nanmean(cube, axis=0, return_valid_count=True)
        z = _zscore_along_day(seasonal)
        profiles[obj] = ObjectProfile(obj, cube, seasonal, z, lat_grid, lon_range, lat_range)
    return profiles


def rolling_profile_prepost_energy(profile_day_feature: np.ndarray, k: int) -> pd.Series:
    arr = np.asarray(profile_day_feature, dtype=float)
    n = arr.shape[0]
    vals = np.full(n, np.nan, dtype=float)
    for d in range(n):
        b0, b1 = d - k, d
        a0, a1 = d + 1, d + 1 + k
        if b0 < 0 or a1 > n:
            continue
        before, bc = safe_nanmean(arr[b0:b1, :], axis=0, return_valid_count=True)
        after, ac = safe_nanmean(arr[a0:a1, :], axis=0, return_valid_count=True)
        diff = after - before
        if np.isfinite(diff).sum() == 0:
            continue
        vals[d] = float(np.sqrt(np.nansum(np.square(diff))))
    return pd.Series(vals, index=np.arange(n), name=f"prepost_energy_k{k}")


def rolling_metric_prepost_score(series: pd.Series, k: int) -> pd.Series:
    x = series.astype(float).to_numpy()
    n = x.size
    vals = np.full(n, np.nan, dtype=float)
    for d in range(n):
        b0, b1 = d - k, d
        a0, a1 = d + 1, d + 1 + k
        if b0 < 0 or a1 > n:
            continue
        before = np.nanmean(x[b0:b1])
        after = np.nanmean(x[a0:a1])
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        vals[d] = float(abs(after - before))
    return pd.Series(vals, index=series.index, name=f"metric_prepost_k{k}")


def metric_time_series(profile: ObjectProfile) -> dict[str, pd.Series]:
    z = np.asarray(profile.z_seasonal, dtype=float)
    lats = np.asarray(profile.lat_grid, dtype=float)
    absz = np.abs(z)
    weight_sum = np.nansum(absz, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.nansum(absz * lats[None, :], axis=1) / weight_sum
        spread = np.sqrt(np.nansum(absz * np.square(lats[None, :] - centroid[:, None]), axis=1) / weight_sum)
    max_idx = np.nanargmax(np.where(np.isfinite(absz), absz, -np.inf), axis=1)
    max_lat = lats[max_idx]
    mean_z = np.nanmean(z, axis=1)
    abs_energy = np.sqrt(np.nanmean(np.square(z), axis=1))
    return {
        "profile_mean_z": pd.Series(mean_z, index=np.arange(z.shape[0])),
        "profile_abs_energy": pd.Series(abs_energy, index=np.arange(z.shape[0])),
        "centroid_abs_z_lat": pd.Series(centroid, index=np.arange(z.shape[0])),
        "spread_abs_z_lat": pd.Series(spread, index=np.arange(z.shape[0])),
        "max_abs_z_lat": pd.Series(max_lat, index=np.arange(z.shape[0])),
    }


def relation_to_assigned(candidate_day: float, assigned_day: float, support_days: int, near_days: int) -> str:
    if not (np.isfinite(candidate_day) and np.isfinite(assigned_day)):
        return "NOT_EVALUATED"
    dist = abs(float(candidate_day) - float(assigned_day))
    if dist <= support_days:
        return "SUPPORTS_OBJECT_PEAK_FAMILY"
    if dist <= near_days:
        return "NEAR_SUPPORT"
    return "DIFFERENT_TIMING"


def validation_status_from_counts(n_support: int, n_near: int, n_diff: int, n_total: int) -> str:
    if n_total <= 0:
        return "NOT_EVALUATED"
    if n_support >= max(1, int(np.ceil(n_total * 0.5))):
        return "SUPPORTED"
    if (n_support + n_near) >= max(1, int(np.ceil(n_total * 0.5))):
        return "PARTIALLY_SUPPORTED"
    if n_diff >= max(1, int(np.ceil(n_total * 0.5))):
        return "NOT_SUPPORTED"
    return "AMBIGUOUS"


def order_with_tau(day_a: float, day_b: float, tau: int) -> str:
    if not (np.isfinite(day_a) and np.isfinite(day_b)):
        return "MISSING"
    if abs(float(day_a) - float(day_b)) <= tau:
        return "NEAR_TIE"
    if float(day_a) < float(day_b) - tau:
        return "A_BEFORE_B"
    if float(day_b) < float(day_a) - tau:
        return "B_BEFORE_A"
    return "NEAR_TIE"


def order_validation(v10_order: str, validation_order: str) -> str:
    if validation_order == "MISSING" or pd.isna(validation_order):
        return "ORDER_AMBIGUOUS"
    if v10_order == validation_order:
        if v10_order == "NEAR_TIE":
            return "ORDER_NEAR_TIE_SUPPORTED"
        return "ORDER_SUPPORTED"
    if v10_order == "NEAR_TIE" or validation_order == "NEAR_TIE":
        return "ORDER_AMBIGUOUS"
    return "ORDER_NOT_SUPPORTED"


def select_peak_within_window(score: pd.Series, center_day: int, half_width: int) -> tuple[float, float]:
    lo = max(0, int(center_day) - int(half_width))
    hi = min(int(score.index.max()), int(center_day) + int(half_width))
    sub = score.loc[(score.index >= lo) & (score.index <= hi)].dropna()
    if sub.empty:
        return np.nan, np.nan
    day = int(sub.idxmax())
    return float(day), float(sub.loc[day])




def topk_local_peaks_within_window(
    score: pd.Series,
    center_day: int,
    half_width: int,
    top_k: int,
    min_distance_days: int,
) -> pd.DataFrame:
    """Return top-k local maxima inside a validation window.

    This is a detector-external validation helper. It does not rerun ruptures;
    it only ranks local maxima of the rolling profile-energy curve.
    """
    lo = max(0, int(center_day) - int(half_width))
    hi = min(int(score.index.max()), int(center_day) + int(half_width))
    sub = score.loc[(score.index >= lo) & (score.index <= hi)].dropna().sort_index()
    rows: list[dict[str, Any]] = []
    if sub.empty:
        return pd.DataFrame(columns=["energy_peak_rank", "energy_peak_day", "energy_peak_score"])

    vals = sub.to_dict()
    days = list(sub.index.astype(int))
    local_candidates: list[tuple[int, float]] = []
    for d in days:
        v = vals.get(d, np.nan)
        if not np.isfinite(v):
            continue
        prev_v = vals.get(d - 1, -np.inf)
        next_v = vals.get(d + 1, -np.inf)
        if v >= prev_v and v >= next_v:
            local_candidates.append((int(d), float(v)))
    if not local_candidates:
        d = int(sub.idxmax())
        local_candidates = [(d, float(sub.loc[d]))]

    # Score-ranked local peaks with a simple non-maximum suppression distance.
    selected: list[tuple[int, float]] = []
    for d, v in sorted(local_candidates, key=lambda x: (-x[1], x[0])):
        if all(abs(d - sd) >= int(min_distance_days) for sd, _ in selected):
            selected.append((d, v))
        if len(selected) >= int(top_k):
            break
    top1 = selected[0][1] if selected else np.nan
    for rank, (d, v) in enumerate(selected, start=1):
        rows.append(
            {
                "energy_peak_rank": rank,
                "energy_peak_day": int(d),
                "energy_peak_month_day": day_index_to_month_day(int(d)),
                "energy_peak_score": float(v),
                "score_ratio_to_top1": float(v / top1) if np.isfinite(top1) and top1 != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_object_candidate_reference(object_catalog: pd.DataFrame, object_lineage: pd.DataFrame) -> pd.DataFrame:
    cat = object_catalog.copy()
    if "point_day" in cat.columns and "candidate_day" not in cat.columns:
        cat = cat.rename(columns={"point_day": "candidate_day"})
    lin = object_lineage.copy()
    keep = [
        "object",
        "candidate_id",
        "candidate_day",
        "object_derived_window_id",
        "object_window_start_day",
        "object_window_end_day",
        "is_object_window_main_peak",
        "nearest_joint_candidate_day",
        "distance_to_nearest_joint_candidate",
        "nearest_joint_lineage_status",
        "nearest_joint_derived_window_id",
        "nearest_strict_accepted_window_id",
        "nearest_strict_anchor_day",
        "distance_to_nearest_strict_anchor",
        "within_strict_accepted_window_band",
        "was_selected_as_v10_window_conditioned_main_peak",
        "v10_window_id_if_selected",
        "candidate_relation_to_joint_lineage",
    ]
    keep = [c for c in keep if c in lin.columns]
    ref = cat.merge(lin[keep], on=["object", "candidate_id", "candidate_day"], how="left", suffixes=("", "_lineage"))
    return ref


def nearest_object_candidate(day: float, obj: str, object_ref: pd.DataFrame) -> dict[str, Any]:
    if not np.isfinite(day):
        return {}
    sub = object_ref[object_ref["object"].astype(str) == str(obj)].copy()
    if sub.empty:
        return {}
    sub["_dist"] = (sub["candidate_day"].astype(float) - float(day)).abs()
    row = sub.sort_values(["_dist", "candidate_day"]).iloc[0].to_dict()
    out = {
        "nearest_object_candidate_id": row.get("candidate_id"),
        "nearest_object_candidate_day": row.get("candidate_day"),
        "distance_to_nearest_object_candidate": row.get("_dist"),
        "nearest_object_support_class": row.get("object_support_class"),
        "nearest_object_bootstrap_match_fraction": row.get("bootstrap_match_fraction"),
        "nearest_joint_candidate_day": row.get("nearest_joint_candidate_day"),
        "distance_to_nearest_joint_candidate": row.get("distance_to_nearest_joint_candidate"),
        "nearest_joint_lineage_status": row.get("nearest_joint_lineage_status"),
        "nearest_joint_derived_window_id": row.get("nearest_joint_derived_window_id"),
        "nearest_strict_accepted_window_id": row.get("nearest_strict_accepted_window_id"),
        "nearest_strict_anchor_day": row.get("nearest_strict_anchor_day"),
        "distance_to_nearest_strict_anchor": row.get("distance_to_nearest_strict_anchor"),
        "was_selected_as_v10_window_conditioned_main_peak": row.get("was_selected_as_v10_window_conditioned_main_peak"),
        "v10_window_id_if_selected": row.get("v10_window_id_if_selected"),
        "candidate_relation_to_joint_lineage": row.get("candidate_relation_to_joint_lineage"),
    }
    return out


def same_candidate_family(a: dict[str, Any], b: dict[str, Any], fallback_day_a: float, fallback_day_b: float, support_days: int) -> bool:
    aid = a.get("nearest_object_candidate_id")
    bid = b.get("nearest_object_candidate_id")
    if pd.notna(aid) and pd.notna(bid) and str(aid) == str(bid):
        return True
    if np.isfinite(fallback_day_a) and np.isfinite(fallback_day_b) and abs(float(fallback_day_a) - float(fallback_day_b)) <= support_days:
        return True
    return False


def classify_energy_support(topk: pd.DataFrame, assigned_ref: dict[str, Any], assigned_day: float, support_days: int, near_days: int) -> str:
    if topk.empty:
        return "NO_ENERGY_PEAKS"
    assigned_id = assigned_ref.get("nearest_object_candidate_id")
    for row in topk.itertuples(index=False):
        rref = getattr(row, "nearest_object_candidate_id", None)
        if pd.notna(assigned_id) and pd.notna(rref) and str(assigned_id) == str(rref):
            return "TOP1_SAME_FAMILY" if int(row.energy_peak_rank) == 1 else "TOPK_SAME_FAMILY"
    min_dist = np.inf
    for row in topk.itertuples(index=False):
        if np.isfinite(row.energy_peak_day) and np.isfinite(assigned_day):
            min_dist = min(min_dist, abs(float(row.energy_peak_day) - float(assigned_day)))
    if min_dist <= support_days:
        return "TOPK_NEAR_ASSIGNED"
    if min_dist <= near_days:
        return "NEAR_ENERGY_PEAK_BUT_NOT_TOPK_FAMILY"
    return "NO_ENERGY_PEAK_NEAR_ASSIGNED"


def classify_switch(assigned_ref: dict[str, Any], top1_ref: dict[str, Any], assigned_day: float, top1_day: float, support_days: int) -> str:
    if same_candidate_family(assigned_ref, top1_ref, assigned_day, top1_day, support_days):
        return "SAME_FAMILY"
    status = str(top1_ref.get("nearest_joint_lineage_status", ""))
    if "non_strict" in status or "non-strict" in status:
        return "SWITCH_TO_KNOWN_NON_STRICT_LINEAGE"
    if str(top1_ref.get("nearest_strict_accepted_window_id", "")) not in ("", "nan", "None") and pd.notna(top1_ref.get("nearest_strict_accepted_window_id")):
        return "SWITCH_TO_NEARBY_STRICT_LINEAGE"
    if pd.notna(top1_ref.get("nearest_object_candidate_id")):
        return "SWITCH_TO_OBJECT_NATIVE_SECONDARY"
    return "SWITCH_TO_UNKNOWN_NEW_PEAK"

def load_required_inputs(v10_root: Path, output_root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "v10_4_assignment": v10_root / "v10.4" / "outputs" / "object_order_sensitivity_v10_4" / "assignment" / "object_candidate_assignment_by_lineage_config_v10_4.csv",
        "v10_4_pairwise": v10_root / "v10.4" / "outputs" / "object_order_sensitivity_v10_4" / "order" / "object_pairwise_order_by_lineage_config_v10_4.csv",
        "v10_4_sequence": v10_root / "v10.4" / "outputs" / "object_order_sensitivity_v10_4" / "order" / "object_order_sequence_by_lineage_config_v10_4.csv",
        "v10_2_object_catalog": v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "cross_object" / "object_native_candidate_catalog_all_objects_v10_2.csv",
        "v10_2_object_lineage": v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "lineage_mapping" / "object_candidate_to_joint_lineage_v10_2.csv",
        "v10_1_joint_lineage": v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "lineage" / "joint_main_window_lineage_v10_1.csv",
        "v10_1_joint_candidate_registry": v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "joint_point_layer" / "joint_candidate_registry_v10_1.csv",
        "v10_1_joint_bootstrap_summary": v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "joint_point_layer" / "joint_candidate_bootstrap_summary_v10_1.csv",
        "v10_1_joint_detector_local_peaks": v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "joint_point_layer" / "joint_detector_local_peaks_all_v10_1.csv",
        "v10_1_joint_derived_windows": v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "joint_window_layer" / "joint_derived_windows_registry_v10_1.csv",
    }
    inventory_rows = []
    data = {}
    for key, path in paths.items():
        df = safe_read_csv(path)
        inventory_rows.append({"input_name": key, "path": str(path), "exists": path.exists(), "status": "loaded" if df is not None else "missing_or_unreadable", "n_rows": 0 if df is None else len(df)})
        if df is not None:
            data[key] = df
    write_dataframe(pd.DataFrame(inventory_rows), output_root / "audit" / "input_inventory_v10_5_a.csv")
    missing = [r["input_name"] for r in inventory_rows if r["status"] != "loaded"]
    if missing:
        raise FileNotFoundError(f"Missing required V10.4 inputs: {missing}")
    return data


def get_baseline_assignments(assignment: pd.DataFrame, target_lineages: list[str]) -> pd.DataFrame:
    df = assignment.copy()
    df = df[(df["config_id"].astype(str) == "BASELINE") & (df["lineage_id"].astype(str).isin(target_lineages))]
    df = df[df["object"].astype(str).isin(OBJECTS)]
    if df.empty:
        raise ValueError("No V10.4 BASELINE object assignments found for target lineages")
    return df


def make_profile_energy_outputs(
    profiles: dict[str, ObjectProfile],
    baseline_assign: pd.DataFrame,
    settings: Settings,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    for _, a in baseline_assign.iterrows():
        lineage = str(a["lineage_id"])
        obj = str(a["object"])
        lineage_day = int(a["lineage_day"])
        assigned_day = float(a["assigned_candidate_day"]) if pd.notna(a.get("assigned_candidate_day")) else np.nan
        prof = profiles[obj]
        for k in settings.validation.k_values:
            score = rolling_profile_prepost_energy(prof.z_seasonal, k)
            peak_day, peak_score = select_peak_within_window(score, lineage_day, settings.validation.window_half_width_days)
            relation = relation_to_assigned(peak_day, assigned_day, settings.validation.support_days, settings.validation.near_days)
            peak_rows.append(
                {
                    "window_id": lineage,
                    "lineage_day": lineage_day,
                    "object": obj,
                    "k": int(k),
                    "v10_4_assigned_peak_day": assigned_day,
                    "profile_energy_peak_day": peak_day,
                    "profile_energy_peak_score": peak_score,
                    "distance_to_v10_4_peak": float(abs(peak_day - assigned_day)) if np.isfinite(peak_day) and np.isfinite(assigned_day) else np.nan,
                    "profile_energy_relation": relation,
                }
            )
            if k == settings.validation.primary_k:
                for day, val in score.items():
                    if (lineage_day - settings.validation.window_half_width_days) <= int(day) <= (lineage_day + settings.validation.window_half_width_days):
                        curve_rows.append(
                            {
                                "window_id": lineage,
                                "lineage_day": lineage_day,
                                "object": obj,
                                "k": int(k),
                                "day": int(day),
                                "month_day": day_index_to_month_day(int(day)),
                                "profile_energy_score": float(val) if np.isfinite(val) else np.nan,
                                "v10_4_assigned_peak_day": assigned_day,
                            }
                        )
                primary_rows.append(peak_rows[-1])
    peak_df = pd.DataFrame(peak_rows)
    curve_df = pd.DataFrame(curve_rows)
    primary_df = pd.DataFrame(primary_rows)
    write_dataframe(peak_df, output_root / "profile_validation" / "profile_energy_peak_by_window_object_k_v10_5_a.csv")
    write_dataframe(curve_df, output_root / "profile_validation" / "profile_energy_curves_by_window_object_v10_5_a.csv")
    write_dataframe(primary_df, output_root / "profile_validation" / "profile_energy_primary_k_summary_v10_5_a.csv")
    return peak_df, curve_df, primary_df




def make_profile_energy_topk_family_outputs(
    profiles: dict[str, ObjectProfile],
    baseline_assign: pd.DataFrame,
    object_catalog: pd.DataFrame,
    object_lineage: pd.DataFrame,
    settings: Settings,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    object_ref = build_object_candidate_reference(object_catalog, object_lineage)
    topk_rows: list[dict[str, Any]] = []
    assigned_rank_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for _, a in baseline_assign.iterrows():
        lineage = str(a["lineage_id"])
        obj = str(a["object"])
        lineage_day = int(a["lineage_day"])
        assigned_day = float(a["assigned_candidate_day"]) if pd.notna(a.get("assigned_candidate_day")) else np.nan
        assigned_id = a.get("assigned_candidate_id", np.nan)
        assigned_ref = nearest_object_candidate(assigned_day, obj, object_ref)
        prof = profiles[obj]
        per_k_status: list[str] = []
        per_k_switch: list[str] = []
        for k in settings.validation.k_values:
            score = rolling_profile_prepost_energy(prof.z_seasonal, k)
            topk = topk_local_peaks_within_window(
                score,
                lineage_day,
                settings.validation.window_half_width_days,
                settings.validation.energy_top_k,
                settings.validation.energy_topk_min_distance_days,
            )
            enriched_rows: list[dict[str, Any]] = []
            for _, tr in topk.iterrows():
                eref = nearest_object_candidate(float(tr["energy_peak_day"]), obj, object_ref)
                same_family = same_candidate_family(assigned_ref, eref, assigned_day, float(tr["energy_peak_day"]), settings.validation.support_days)
                row = {
                    "window_id": lineage,
                    "lineage_day": lineage_day,
                    "object": obj,
                    "k": int(k),
                    "v10_4_assigned_peak_day": assigned_day,
                    "v10_4_assigned_candidate_id": assigned_id,
                    **tr.to_dict(),
                    **eref,
                    "same_family_as_v10_4_assigned": bool(same_family),
                    "distance_to_v10_4_assigned_peak": float(abs(float(tr["energy_peak_day"]) - assigned_day)) if np.isfinite(assigned_day) else np.nan,
                }
                enriched_rows.append(row)
                topk_rows.append(row)
            enriched = pd.DataFrame(enriched_rows)
            support_status = classify_energy_support(enriched, assigned_ref, assigned_day, settings.validation.support_days, settings.validation.near_days)
            per_k_status.append(support_status)
            top1 = enriched[enriched["energy_peak_rank"] == 1].iloc[0].to_dict() if not enriched.empty else {}
            top1_day = float(top1.get("energy_peak_day", np.nan)) if top1 else np.nan
            top1_ref = {k2: top1.get(k2) for k2 in [
                "nearest_object_candidate_id", "nearest_object_candidate_day", "nearest_object_support_class", "nearest_joint_candidate_day",
                "nearest_joint_lineage_status", "nearest_joint_derived_window_id", "nearest_strict_accepted_window_id",
                "candidate_relation_to_joint_lineage", "was_selected_as_v10_window_conditioned_main_peak", "v10_window_id_if_selected",
            ]}
            switch_type = classify_switch(assigned_ref, top1_ref, assigned_day, top1_day, settings.validation.support_days)
            per_k_switch.append(switch_type)
            assigned_energy_score = float(score.loc[int(round(assigned_day))]) if np.isfinite(assigned_day) and int(round(assigned_day)) in score.index and np.isfinite(score.loc[int(round(assigned_day))]) else np.nan
            top1_score = float(top1.get("energy_peak_score", np.nan)) if top1 else np.nan
            assigned_rank = np.nan
            nearest_energy_day = np.nan
            nearest_energy_rank = np.nan
            nearest_energy_score = np.nan
            if not enriched.empty and np.isfinite(assigned_day):
                enriched["_dist_to_assigned"] = (enriched["energy_peak_day"].astype(float) - assigned_day).abs()
                near = enriched.sort_values(["_dist_to_assigned", "energy_peak_rank"]).iloc[0]
                nearest_energy_day = float(near["energy_peak_day"])
                nearest_energy_rank = int(near["energy_peak_rank"])
                nearest_energy_score = float(near["energy_peak_score"])
                same_rows = enriched[enriched["same_family_as_v10_4_assigned"] == True]
                if not same_rows.empty:
                    assigned_rank = int(same_rows.sort_values("energy_peak_rank").iloc[0]["energy_peak_rank"])
            assigned_rank_rows.append(
                {
                    "window_id": lineage,
                    "lineage_day": lineage_day,
                    "object": obj,
                    "k": int(k),
                    "v10_4_assigned_peak_day": assigned_day,
                    "v10_4_assigned_candidate_id": assigned_id,
                    "assigned_nearest_object_candidate_id": assigned_ref.get("nearest_object_candidate_id"),
                    "assigned_nearest_object_candidate_day": assigned_ref.get("nearest_object_candidate_day"),
                    "assigned_object_support_class": assigned_ref.get("nearest_object_support_class"),
                    "assigned_nearest_joint_candidate_day": assigned_ref.get("nearest_joint_candidate_day"),
                    "assigned_nearest_joint_lineage_status": assigned_ref.get("nearest_joint_lineage_status"),
                    "assigned_energy_score_at_day": assigned_energy_score,
                    "top1_energy_peak_day": top1_day,
                    "top1_energy_peak_score": top1_score,
                    "assigned_score_ratio_to_top1": float(assigned_energy_score / top1_score) if np.isfinite(assigned_energy_score) and np.isfinite(top1_score) and top1_score != 0 else np.nan,
                    "nearest_energy_peak_day": nearest_energy_day,
                    "nearest_energy_peak_rank": nearest_energy_rank,
                    "nearest_energy_peak_score": nearest_energy_score,
                    "assigned_candidate_energy_rank_if_topk": assigned_rank,
                    "energy_support_status": support_status,
                    "top1_switch_type": switch_type,
                }
            )
            switch_rows.append(
                {
                    "window_id": lineage,
                    "lineage_day": lineage_day,
                    "object": obj,
                    "k": int(k),
                    "v10_4_assigned_peak_day": assigned_day,
                    "v10_4_assigned_candidate_id": assigned_id,
                    "v10_4_candidate_family": f"{obj}:{assigned_ref.get('nearest_object_candidate_id')}@{assigned_ref.get('nearest_object_candidate_day')}|{assigned_ref.get('nearest_joint_lineage_status')}",
                    "profile_energy_top1_day": top1_day,
                    "top1_candidate_family": f"{obj}:{top1_ref.get('nearest_object_candidate_id')}@{top1_ref.get('nearest_object_candidate_day')}|{top1_ref.get('nearest_joint_lineage_status')}",
                    "top1_joint_lineage_status": top1_ref.get("nearest_joint_lineage_status"),
                    "top1_energy_peak_score": top1_score,
                    "assigned_energy_score_at_day": assigned_energy_score,
                    "assigned_score_ratio_to_top1": float(assigned_energy_score / top1_score) if np.isfinite(assigned_energy_score) and np.isfinite(top1_score) and top1_score != 0 else np.nan,
                    "same_family_flag": switch_type == "SAME_FAMILY",
                    "switch_type": switch_type,
                    "interpretation_note": "candidate-family-aware detector-external validation; not physical/causal proof",
                }
            )
        # Primary summary at object/window level.
        primary_status = per_k_status[settings.validation.k_values.index(settings.validation.primary_k)] if settings.validation.primary_k in settings.validation.k_values else per_k_status[0]
        n_top1 = sum(s == "TOP1_SAME_FAMILY" for s in per_k_status)
        n_topk = sum(s in ("TOP1_SAME_FAMILY", "TOPK_SAME_FAMILY") for s in per_k_status)
        n_near = sum(s in ("TOPK_NEAR_ASSIGNED", "NEAR_ENERGY_PEAK_BUT_NOT_TOPK_FAMILY") for s in per_k_status)
        n_switch = sum(sw != "SAME_FAMILY" for sw in per_k_switch)
        if n_top1 > 0:
            family_validation = "FAMILY_TOP1_SUPPORTED"
        elif n_topk > 0:
            family_validation = "FAMILY_TOPK_SUPPORTED"
        elif n_near > 0:
            family_validation = "FAMILY_SECONDARY_WEAK_SUPPORT"
        elif n_switch == len(per_k_switch):
            # If every k switches to another known candidate family, expose that rather than plain no-support.
            family_validation = "FAMILY_SWITCH_TO_KNOWN_CANDIDATE"
        else:
            family_validation = "FAMILY_NOT_DETECTED"
        summary_rows.append(
            {
                "window_id": lineage,
                "lineage_day": lineage_day,
                "object": obj,
                "v10_4_assigned_peak_day": assigned_day,
                "v10_4_assigned_candidate_id": assigned_id,
                "primary_k": settings.validation.primary_k,
                "primary_k_energy_support_status": primary_status,
                "n_k_top1_same_family": n_top1,
                "n_k_topk_same_family": n_topk,
                "n_k_near_assigned": n_near,
                "n_k_family_switch": n_switch,
                "candidate_family_validation_status": family_validation,
            }
        )

    topk_df = pd.DataFrame(topk_rows)
    assigned_rank_df = pd.DataFrame(assigned_rank_rows)
    switch_df = pd.DataFrame(switch_rows)
    summary_df = pd.DataFrame(summary_rows)
    write_dataframe(topk_df, output_root / "profile_validation" / "profile_energy_topk_peaks_by_window_object_v10_5_b.csv")
    write_dataframe(assigned_rank_df, output_root / "profile_validation" / "v10_4_assigned_peak_energy_rank_v10_5_b.csv")
    write_dataframe(switch_df, output_root / "profile_validation" / "candidate_family_switch_inventory_v10_5_b.csv")
    write_dataframe(summary_df, output_root / "validation_summary_candidate_family_v10_5_b.csv")
    return topk_df, assigned_rank_df, switch_df, summary_df



def _is_non_strict_status(x: Any) -> bool:
    s = str(x).lower().replace("-", "_")
    return "non_strict" in s or "derived_non_strict" in s


def _safe_str_nonempty(x: Any) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    sx = str(x)
    if sx.lower() in ("nan", "none", "null"):
        return ""
    return sx


def classify_top1_nonselection_reason(row: pd.Series) -> str:
    """Classify why the energy-top1 candidate family is not the V10.4 assigned family.

    This is a descriptive audit label, not a physical interpretation. It separates
    strong profile-energy candidates that belong to another known lineage from
    cases that may indicate assignment-rule ambiguity.
    """
    switch_type = str(row.get("switch_type", ""))
    if switch_type == "SAME_FAMILY":
        return "ENERGY_TOP1_SAME_AS_V10_4_ASSIGNED_FAMILY"

    top1_status = row.get("top1_nearest_joint_lineage_status", row.get("top1_joint_lineage_status", ""))
    assigned_status = row.get("assigned_nearest_joint_lineage_status", "")
    target_lineage = str(row.get("window_id", ""))
    top1_strict = _safe_str_nonempty(row.get("top1_nearest_strict_accepted_window_id", ""))
    assigned_strict = _safe_str_nonempty(row.get("assigned_nearest_strict_accepted_window_id", ""))

    if _is_non_strict_status(top1_status):
        return "ENERGY_TOP1_NOT_SELECTED_BECAUSE_KNOWN_NON_STRICT_LINEAGE"
    if top1_strict and top1_strict != target_lineage:
        return "ENERGY_TOP1_NOT_SELECTED_BECAUSE_OTHER_STRICT_LINEAGE"
    if top1_strict == target_lineage and assigned_strict == target_lineage:
        return "ENERGY_TOP1_WITHIN_TARGET_STRICT_LINEAGE_BUT_DIFFERENT_CANDIDATE_FAMILY"
    if _safe_str_nonempty(row.get("top1_nearest_object_candidate_id", "")):
        return "ENERGY_TOP1_NOT_SELECTED_BECAUSE_OBJECT_NATIVE_SECONDARY_OR_OTHER_FAMILY"
    return "ENERGY_TOP1_NOT_SELECTED_REASON_UNCLEAR_REVIEW_REQUIRED"


def make_candidate_family_selection_reason_audit(
    topk_df: pd.DataFrame,
    assigned_rank_df: pd.DataFrame,
    switch_df: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit why profile-energy top1 families were not selected by V10.4.

    This closes the current V10.5 gap: a top1 mismatch is not treated as a large
    day error. Instead, it is classified as same-family support, known non-strict
    lineage competition, other strict-lineage competition, or unclear selection
    ambiguity.
    """
    if switch_df.empty:
        empty = pd.DataFrame()
        write_dataframe(empty, output_root / "profile_validation" / "candidate_family_selection_reason_audit_v10_5_c.csv")
        write_dataframe(empty, output_root / "profile_validation" / "candidate_family_role_long_v10_5_c.csv")
        write_dataframe(empty, output_root / "validation_summary_candidate_family_selection_reason_v10_5_c.csv")
        return empty, empty, empty

    # Attach assigned-family metadata and top1 metadata from the top-k table.
    key = ["window_id", "lineage_day", "object", "k"]
    assigned_cols = [
        "assigned_nearest_object_candidate_id",
        "assigned_nearest_object_candidate_day",
        "assigned_object_support_class",
        "assigned_nearest_joint_candidate_day",
        "assigned_nearest_joint_lineage_status",
        "assigned_energy_score_at_day",
        "top1_energy_peak_day",
        "top1_energy_peak_score",
        "assigned_score_ratio_to_top1",
        "energy_support_status",
    ]
    ar = assigned_rank_df.copy()
    ar_keep = [c for c in key + assigned_cols if c in ar.columns]
    if ar_keep:
        sw = switch_df.merge(ar[ar_keep], on=key, how="left", suffixes=("", "_assigned_rank"))
    else:
        sw = switch_df.copy()

    top1 = topk_df[topk_df.get("energy_peak_rank", pd.Series(dtype=float)) == 1].copy() if not topk_df.empty else pd.DataFrame()
    top1_cols = [
        "nearest_object_candidate_id",
        "nearest_object_candidate_day",
        "nearest_object_support_class",
        "nearest_object_bootstrap_match_fraction",
        "nearest_joint_candidate_day",
        "distance_to_nearest_joint_candidate",
        "nearest_joint_lineage_status",
        "nearest_joint_derived_window_id",
        "nearest_strict_accepted_window_id",
        "nearest_strict_anchor_day",
        "distance_to_nearest_strict_anchor",
        "was_selected_as_v10_window_conditioned_main_peak",
        "v10_window_id_if_selected",
        "candidate_relation_to_joint_lineage",
    ]
    if not top1.empty:
        rename = {c: f"top1_{c}" for c in top1_cols if c in top1.columns}
        t1_keep = key + ["energy_peak_day", "energy_peak_score", "score_ratio_to_top1"] + list(rename.keys())
        t1_keep = [c for c in t1_keep if c in top1.columns]
        t1 = top1[t1_keep].rename(columns={**rename, "energy_peak_day": "top1_local_energy_peak_day", "energy_peak_score": "top1_local_energy_peak_score"})
        sw = sw.merge(t1, on=key, how="left", suffixes=("", "_top1row"))

    # Normalize duplicate columns from prior switch output.
    if "top1_nearest_joint_lineage_status" not in sw.columns and "top1_joint_lineage_status" in sw.columns:
        sw["top1_nearest_joint_lineage_status"] = sw["top1_joint_lineage_status"]
    if "top1_local_energy_peak_day" not in sw.columns and "profile_energy_top1_day" in sw.columns:
        sw["top1_local_energy_peak_day"] = sw["profile_energy_top1_day"]
    if "top1_local_energy_peak_score" not in sw.columns and "top1_energy_peak_score" in sw.columns:
        sw["top1_local_energy_peak_score"] = sw["top1_energy_peak_score"]

    sw["top1_nonselection_reason"] = sw.apply(classify_top1_nonselection_reason, axis=1)
    sw["assigned_family_role"] = "LINEAGE_ASSIGNED_FAMILY"
    sw["top1_family_role"] = np.where(sw["switch_type"].astype(str) == "SAME_FAMILY", "ENERGY_DOMINANT_AND_LINEAGE_ASSIGNED_FAMILY", "ENERGY_DOMINANT_COMPETING_FAMILY")
    sw["family_competition_status"] = np.where(
        sw["switch_type"].astype(str) == "SAME_FAMILY",
        "NO_COMPETITION_TOP1_MATCH",
        "TOPK_SUPPORTED_WITH_STRONGER_COMPETING_FAMILY",
    )
    sw["interpretation_boundary"] = "selection-reason audit only; does not make physical or causal claims"

    # Long role table: one row for assigned family and one row for energy-top1 family per k.
    long_rows: list[dict[str, Any]] = []
    for _, r in sw.iterrows():
        base = {
            "window_id": r.get("window_id"),
            "lineage_day": r.get("lineage_day"),
            "object": r.get("object"),
            "k": r.get("k"),
            "switch_type": r.get("switch_type"),
            "top1_nonselection_reason": r.get("top1_nonselection_reason"),
            "family_competition_status": r.get("family_competition_status"),
        }
        long_rows.append({
            **base,
            "candidate_family_role": "LINEAGE_ASSIGNED_FAMILY",
            "candidate_day": r.get("v10_4_assigned_peak_day"),
            "candidate_family_label": r.get("v10_4_candidate_family"),
            "energy_rank_or_role": r.get("energy_support_status"),
            "energy_score": r.get("assigned_energy_score_at_day"),
            "score_ratio_to_top1": r.get("assigned_score_ratio_to_top1"),
            "object_candidate_id": r.get("assigned_nearest_object_candidate_id"),
            "object_candidate_day": r.get("assigned_nearest_object_candidate_day"),
            "object_support_class": r.get("assigned_object_support_class"),
            "nearest_joint_candidate_day": r.get("assigned_nearest_joint_candidate_day"),
            "nearest_joint_lineage_status": r.get("assigned_nearest_joint_lineage_status"),
        })
        long_rows.append({
            **base,
            "candidate_family_role": "ENERGY_DOMINANT_FAMILY",
            "candidate_day": r.get("profile_energy_top1_day"),
            "candidate_family_label": r.get("top1_candidate_family"),
            "energy_rank_or_role": "TOP1_PROFILE_ENERGY",
            "energy_score": r.get("top1_energy_peak_score"),
            "score_ratio_to_top1": 1.0,
            "object_candidate_id": r.get("top1_nearest_object_candidate_id"),
            "object_candidate_day": r.get("top1_nearest_object_candidate_day"),
            "object_support_class": r.get("top1_nearest_object_support_class"),
            "nearest_joint_candidate_day": r.get("top1_nearest_joint_candidate_day"),
            "nearest_joint_lineage_status": r.get("top1_nearest_joint_lineage_status"),
        })
    long_df = pd.DataFrame(long_rows)

    summary = sw.groupby(["window_id", "object"], as_index=False).agg(
        n_k=("k", "count"),
        n_k_same_family=("switch_type", lambda x: int((x.astype(str) == "SAME_FAMILY").sum())),
        n_k_competing_family=("switch_type", lambda x: int((x.astype(str) != "SAME_FAMILY").sum())),
        median_assigned_score_ratio_to_top1=("assigned_score_ratio_to_top1", "median"),
        min_assigned_score_ratio_to_top1=("assigned_score_ratio_to_top1", "min"),
        max_assigned_score_ratio_to_top1=("assigned_score_ratio_to_top1", "max"),
        dominant_nonselection_reason=("top1_nonselection_reason", lambda x: x.value_counts(dropna=False).index[0] if len(x) else ""),
        dominant_competition_status=("family_competition_status", lambda x: x.value_counts(dropna=False).index[0] if len(x) else ""),
    )
    summary["selection_reason_audit_status"] = np.where(
        summary["n_k_same_family"] > 0,
        "ENERGY_TOP1_MATCHES_ASSIGNED_IN_AT_LEAST_ONE_K",
        np.where(
            summary["n_k_competing_family"] > 0,
            "ASSIGNED_FAMILY_TOPK_SUPPORTED_BUT_ENERGY_TOP1_COMPETES",
            "NO_ENERGY_FAMILY_EVIDENCE_REVIEW_REQUIRED",
        ),
    )

    write_dataframe(sw, output_root / "profile_validation" / "candidate_family_selection_reason_audit_v10_5_c.csv")
    write_dataframe(long_df, output_root / "profile_validation" / "candidate_family_role_long_v10_5_c.csv")
    write_dataframe(summary, output_root / "validation_summary_candidate_family_selection_reason_v10_5_c.csv")
    return sw, long_df, summary


def _rank_desc_with_tiebreak(df: pd.DataFrame, score_col: str, day_col: str, rank_col: str) -> pd.Series:
    if df.empty or score_col not in df.columns:
        return pd.Series(dtype=float)
    tmp = df[[score_col, day_col]].copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp[day_col] = pd.to_numeric(tmp[day_col], errors="coerce")
    order = tmp.sort_values([score_col, day_col], ascending=[False, True], na_position="last").index
    ranks = pd.Series(np.nan, index=df.index, dtype=float)
    for i, idx in enumerate(order, start=1):
        ranks.loc[idx] = float(i)
    return ranks


def classify_strength_stability(strength_rank: float, bootstrap_match_fraction: float, n_candidates: int) -> str:
    """Descriptive 2-D class; thresholds are reported but not used as physics."""
    if not np.isfinite(strength_rank):
        strength = "UNKNOWN_SCORE"
    else:
        strength = "HIGH_SCORE" if float(strength_rank) <= max(1, min(3, int(np.ceil(n_candidates * 0.4)))) else "LOWER_SCORE"
    if not np.isfinite(bootstrap_match_fraction):
        stability = "UNKNOWN_BOOTSTRAP"
    else:
        stability = "HIGH_BOOTSTRAP" if float(bootstrap_match_fraction) >= 0.95 else "LOWER_BOOTSTRAP"
    return f"{strength}_{stability}"


def make_main_method_peak_strength_bootstrap_audit(input_data: dict[str, pd.DataFrame], output_root: Path) -> pd.DataFrame:
    """V10.5_d A-line audit: main joint detector peak strength vs year-bootstrap stability.

    This explicitly checks whether non-strict peaks are weak in score, or whether
    they are high-score candidates that did not pass strict bootstrap recurrence.
    """
    reg = input_data.get("v10_1_joint_candidate_registry", pd.DataFrame()).copy()
    boot = input_data.get("v10_1_joint_bootstrap_summary", pd.DataFrame()).copy()
    lineage = input_data.get("v10_1_joint_lineage", pd.DataFrame()).copy()
    local = input_data.get("v10_1_joint_detector_local_peaks", pd.DataFrame()).copy()
    derived = input_data.get("v10_1_joint_derived_windows", pd.DataFrame()).copy()

    if reg.empty:
        out = pd.DataFrame()
        write_dataframe(out, output_root / "profile_validation" / "main_method_peak_strength_vs_bootstrap_v10_5_d.csv")
        return out

    if "candidate_day" not in reg.columns and "point_day" in reg.columns:
        reg = reg.rename(columns={"point_day": "candidate_day"})
    if "candidate_day" not in boot.columns and "point_day" in boot.columns:
        boot = boot.rename(columns={"point_day": "candidate_day"})
    if "candidate_day" not in lineage.columns and "point_day" in lineage.columns:
        lineage = lineage.rename(columns={"point_day": "candidate_day"})
    if "candidate_day" not in local.columns and "peak_day" in local.columns:
        local = local.rename(columns={"peak_day": "candidate_day", "peak_score": "local_peak_score", "peak_rank": "local_peak_rank"})

    df = reg.merge(boot, on=[c for c in ["candidate_id", "candidate_day"] if c in reg.columns and c in boot.columns], how="left", suffixes=("", "_boot"))
    lin_keep = [c for c in ["candidate_id", "candidate_day", "v6_1_window_id", "v6_1_window_start", "v6_1_window_end", "v6_1_main_peak_day", "v6_1_is_window_main_peak", "strict_accepted_window_id", "strict_accepted_flag", "strict_accepted_reason", "strict_exclusion_reason", "lineage_status"] if c in lineage.columns]
    if lin_keep:
        df = df.merge(lineage[lin_keep], on=[c for c in ["candidate_id", "candidate_day"] if c in df.columns and c in lin_keep], how="left")
    local_keep = [c for c in ["candidate_day", "local_peak_score", "local_peak_rank", "peak_prominence"] if c in local.columns]
    if local_keep:
        df = df.merge(local[local_keep], on="candidate_day", how="left", suffixes=("", "_local"))

    if "peak_score" not in df.columns and "local_peak_score" in df.columns:
        df["peak_score"] = df["local_peak_score"]
    if "peak_score" not in df.columns:
        df["peak_score"] = np.nan
    df["joint_detector_score"] = pd.to_numeric(df["peak_score"], errors="coerce")
    df["joint_detector_score_rank"] = _rank_desc_with_tiebreak(df, "joint_detector_score", "candidate_day", "joint_detector_score_rank")
    n = max(1, len(df))
    df["joint_detector_score_percentile_desc"] = 1.0 - (df["joint_detector_score_rank"] - 1.0) / max(1, n - 1)
    if "bootstrap_match_fraction" in df.columns:
        df["bootstrap_match_rank"] = _rank_desc_with_tiebreak(df.rename(columns={"bootstrap_match_fraction": "_bmf"}), "_bmf", "candidate_day", "bootstrap_match_rank")
    else:
        df["bootstrap_match_fraction"] = np.nan
        df["bootstrap_match_rank"] = np.nan
    df["score_rank_minus_bootstrap_rank"] = df["joint_detector_score_rank"] - df["bootstrap_match_rank"]
    df["strength_stability_class"] = [classify_strength_stability(r, b, n) for r, b in zip(df["joint_detector_score_rank"], df["bootstrap_match_fraction"])]

    def _promotion_reason(row: pd.Series) -> str:
        if bool(row.get("strict_accepted_flag", False)):
            return "STRICT_ACCEPTED_MAIN_WINDOW"
        bmf = row.get("bootstrap_match_fraction", np.nan)
        if np.isfinite(bmf) and float(bmf) < 0.95:
            return "DETECTED_BUT_BOOTSTRAP_BELOW_STRICT"
        status = str(row.get("lineage_status", ""))
        if "non_strict" in status:
            return "DERIVED_NON_STRICT_WINDOW"
        return "NON_PROMOTION_REASON_UNCLEAR_REVIEW_REQUIRED"

    df["promotion_failure_reason"] = df.apply(_promotion_reason, axis=1)
    out_cols = [
        "candidate_id", "candidate_day", "month_day", "is_formal_primary",
        "v6_1_window_id", "v6_1_main_peak_day", "v6_1_is_window_main_peak",
        "strict_accepted_flag", "strict_accepted_window_id", "lineage_status",
        "joint_detector_score", "joint_detector_score_rank", "joint_detector_score_percentile_desc",
        "bootstrap_strict_fraction", "bootstrap_match_fraction", "bootstrap_near_fraction", "bootstrap_no_match_fraction",
        "bootstrap_match_rank", "score_rank_minus_bootstrap_rank", "strength_stability_class", "promotion_failure_reason",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    out = df[out_cols].sort_values("candidate_day").reset_index(drop=True)
    write_dataframe(out, output_root / "profile_validation" / "main_method_peak_strength_vs_bootstrap_v10_5_d.csv")
    return out


def _profile_energy_topk_for_resampled_cube(cube: np.ndarray, sample_idx: np.ndarray, k: int, lineage_day: int, settings: Settings) -> pd.DataFrame:
    seasonal, _ = safe_nanmean(cube[sample_idx, :, :], axis=0, return_valid_count=True)
    z = _zscore_along_day(seasonal)
    score = rolling_profile_prepost_energy(z, int(k))
    return topk_local_peaks_within_window(
        score,
        lineage_day,
        settings.validation.window_half_width_days,
        settings.validation.profile_energy_bootstrap_top_k,
        settings.validation.energy_topk_min_distance_days,
    )


def _match_peak_days_to_family(days: list[float], family_day: float, radius: int) -> bool:
    if not np.isfinite(family_day):
        return False
    for d in days:
        if np.isfinite(d) and abs(float(d) - float(family_day)) <= int(radius):
            return True
    return False


def _rank_for_family(topk: pd.DataFrame, family_day: float, radius: int) -> float:
    if topk.empty or not np.isfinite(family_day):
        return np.nan
    matches = topk[(topk["energy_peak_day"].astype(float) - float(family_day)).abs() <= int(radius)]
    if matches.empty:
        return np.nan
    return float(matches.sort_values("energy_peak_rank").iloc[0]["energy_peak_rank"])


def make_profile_energy_family_bootstrap_audit(
    profiles: dict[str, ObjectProfile],
    selection_reason_df: pd.DataFrame,
    family_role_long_df: pd.DataFrame,
    settings: Settings,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """V10.5_d B-line audit: profile-energy family strength vs bootstrap stability.

    Bootstraps year samples and checks whether energy-dominant and lineage-assigned
    families recur as top1/top-k profile-energy peaks. This tests whether the
    validation method itself faces high-strength-but-low-stability candidates.
    """
    if family_role_long_df.empty:
        empty = pd.DataFrame()
        for rel in [
            "profile_energy_family_bootstrap_stability_v10_5_d.csv",
            "profile_energy_family_pair_comparison_v10_5_d.csv",
            "candidate_family_strength_stability_matrix_v10_5_d.csv",
            "key_competition_cases_v10_5_d.csv",
        ]:
            write_dataframe(empty, output_root / "profile_validation" / rel)
        return empty, empty, empty, empty

    n_boot = int(settings.validation.profile_energy_bootstrap_n)
    seed = int(settings.validation.profile_energy_bootstrap_seed)
    rng = np.random.default_rng(seed)
    role = family_role_long_df.copy()
    # Keep only the two role types needed for the strength-stability contrast.
    role = role[role["candidate_family_role"].astype(str).isin(["LINEAGE_ASSIGNED_FAMILY", "ENERGY_DOMINANT_FAMILY"])].copy()
    role["family_day"] = pd.to_numeric(role.get("object_candidate_day", role.get("candidate_day")), errors="coerce")
    role["candidate_day"] = pd.to_numeric(role.get("candidate_day"), errors="coerce")
    role["energy_score"] = pd.to_numeric(role.get("energy_score"), errors="coerce")
    role["score_ratio_to_top1"] = pd.to_numeric(role.get("score_ratio_to_top1"), errors="coerce")

    rows: list[dict[str, Any]] = []
    boot_detail_rows: list[dict[str, Any]] = []
    # Pre-generate bootstrap indices by object to keep paired family roles comparable.
    sample_cache: dict[str, list[np.ndarray]] = {}
    for obj in role["object"].dropna().astype(str).unique():
        cube = profiles[obj].raw_cube
        ny = int(cube.shape[0])
        sample_cache[obj] = [rng.integers(0, ny, size=ny) for _ in range(n_boot)]

    group_cols = ["window_id", "lineage_day", "object", "k"]
    for key, sub in role.groupby(group_cols, dropna=False):
        window_id, lineage_day, obj, k = key
        obj = str(obj)
        lineage_day = int(lineage_day)
        k = int(k)
        cube = profiles[obj].raw_cube
        # Precompute bootstrap top-k lists once for this window/object/k.
        boot_topk: list[pd.DataFrame] = []
        for b, sample_idx in enumerate(sample_cache[obj]):
            tk = _profile_energy_topk_for_resampled_cube(cube, sample_idx, k, lineage_day, settings)
            boot_topk.append(tk)
        for _, r in sub.iterrows():
            family_day = float(r.get("family_day", np.nan))
            role_name = str(r.get("candidate_family_role", ""))
            strict_top1 = match_top1 = near_top1 = 0
            strict_topk = match_topk = near_topk = 0
            ranks: list[float] = []
            missing = 0
            for b, tk in enumerate(boot_topk):
                days = [float(x) for x in tk.get("energy_peak_day", pd.Series(dtype=float)).tolist()]
                top1_day = days[0] if days else np.nan
                if _match_peak_days_to_family([top1_day], family_day, 2):
                    strict_top1 += 1
                if _match_peak_days_to_family([top1_day], family_day, settings.validation.support_days):
                    match_top1 += 1
                if _match_peak_days_to_family([top1_day], family_day, settings.validation.near_days):
                    near_top1 += 1
                if _match_peak_days_to_family(days, family_day, 2):
                    strict_topk += 1
                if _match_peak_days_to_family(days, family_day, settings.validation.support_days):
                    match_topk += 1
                if _match_peak_days_to_family(days, family_day, settings.validation.near_days):
                    near_topk += 1
                rk = _rank_for_family(tk, family_day, settings.validation.support_days)
                if np.isfinite(rk):
                    ranks.append(rk)
                else:
                    missing += 1
            denom = max(1, n_boot)
            mean_rank = float(np.nanmean(ranks)) if ranks else np.nan
            median_rank = float(np.nanmedian(ranks)) if ranks else np.nan
            q25 = float(np.nanpercentile(ranks, 25)) if ranks else np.nan
            q75 = float(np.nanpercentile(ranks, 75)) if ranks else np.nan
            top1_match_fraction = match_top1 / denom
            topk_match_fraction = match_topk / denom
            if top1_match_fraction >= 0.80:
                status = "STABLE_TOP1_FAMILY"
            elif topk_match_fraction >= 0.80:
                status = "STABLE_TOPK_FAMILY"
            elif topk_match_fraction >= 0.50:
                status = "WEAK_TOPK_FAMILY"
            else:
                status = "UNSTABLE_ENERGY_FAMILY"
            rows.append({
                "window_id": window_id,
                "lineage_day": lineage_day,
                "object": obj,
                "k": k,
                "family_role": role_name,
                "family_day": family_day,
                "candidate_day": r.get("candidate_day"),
                "candidate_family_label": r.get("candidate_family_label"),
                "object_candidate_id": r.get("object_candidate_id"),
                "object_candidate_day": r.get("object_candidate_day"),
                "object_support_class": r.get("object_support_class"),
                "nearest_joint_candidate_day": r.get("nearest_joint_candidate_day"),
                "nearest_joint_lineage_status": r.get("nearest_joint_lineage_status"),
                "observed_energy_score": r.get("energy_score"),
                "observed_score_ratio_to_top1": r.get("score_ratio_to_top1"),
                "bootstrap_n": n_boot,
                "bootstrap_top1_strict_fraction": strict_top1 / denom,
                "bootstrap_top1_match_fraction": top1_match_fraction,
                "bootstrap_top1_near_fraction": near_top1 / denom,
                "bootstrap_topk_strict_fraction": strict_topk / denom,
                "bootstrap_topk_match_fraction": topk_match_fraction,
                "bootstrap_topk_near_fraction": near_topk / denom,
                "bootstrap_missing_fraction": missing / denom,
                "mean_bootstrap_rank_if_matched": mean_rank,
                "median_bootstrap_rank_if_matched": median_rank,
                "rank_iqr_if_matched": (q75 - q25) if np.isfinite(q75) and np.isfinite(q25) else np.nan,
                "family_bootstrap_status": status,
            })
    fam = pd.DataFrame(rows)
    write_dataframe(fam, output_root / "profile_validation" / "profile_energy_family_bootstrap_stability_v10_5_d.csv")

    pair_rows: list[dict[str, Any]] = []
    if not fam.empty:
        for key, sub in fam.groupby(group_cols, dropna=False):
            ed = sub[sub["family_role"].astype(str) == "ENERGY_DOMINANT_FAMILY"]
            la = sub[sub["family_role"].astype(str) == "LINEAGE_ASSIGNED_FAMILY"]
            if ed.empty or la.empty:
                continue
            edr = ed.iloc[0]
            lar = la.iloc[0]
            if edr.get("bootstrap_topk_match_fraction", np.nan) > lar.get("bootstrap_topk_match_fraction", np.nan):
                more_stable = "ENERGY_DOMINANT_FAMILY"
            elif edr.get("bootstrap_topk_match_fraction", np.nan) < lar.get("bootstrap_topk_match_fraction", np.nan):
                more_stable = "LINEAGE_ASSIGNED_FAMILY"
            else:
                more_stable = "TIE_OR_UNCLEAR"
            if str(edr.get("family_bootstrap_status")) in ("STABLE_TOP1_FAMILY", "STABLE_TOPK_FAMILY") and str(lar.get("family_bootstrap_status")) in ("STABLE_TOP1_FAMILY", "STABLE_TOPK_FAMILY"):
                interp = "BOTH_STABLE_MULTIFAMILY_STRUCTURE"
            elif str(edr.get("family_bootstrap_status")) in ("STABLE_TOP1_FAMILY", "STABLE_TOPK_FAMILY"):
                interp = "TOP1_STRONG_AND_STABLE"
            elif str(lar.get("family_bootstrap_status")) in ("STABLE_TOP1_FAMILY", "STABLE_TOPK_FAMILY"):
                interp = "ASSIGNED_SECONDARY_BUT_STABLE"
            elif str(edr.get("family_bootstrap_status")) == "WEAK_TOPK_FAMILY":
                interp = "TOP1_STRONG_BUT_WEAKLY_STABLE"
            else:
                interp = "BOTH_UNSTABLE_OR_NEEDS_REVIEW"
            pair_rows.append({
                "window_id": key[0], "lineage_day": key[1], "object": key[2], "k": key[3],
                "energy_dominant_family_day": edr.get("family_day"),
                "lineage_assigned_family_day": lar.get("family_day"),
                "energy_dominant_observed_score": edr.get("observed_energy_score"),
                "lineage_assigned_observed_score": lar.get("observed_energy_score"),
                "observed_assigned_to_top1_score_ratio": lar.get("observed_score_ratio_to_top1"),
                "energy_dominant_bootstrap_top1_fraction": edr.get("bootstrap_top1_match_fraction"),
                "energy_dominant_bootstrap_topk_fraction": edr.get("bootstrap_topk_match_fraction"),
                "lineage_assigned_bootstrap_top1_fraction": lar.get("bootstrap_top1_match_fraction"),
                "lineage_assigned_bootstrap_topk_fraction": lar.get("bootstrap_topk_match_fraction"),
                "energy_dominant_status": edr.get("family_bootstrap_status"),
                "lineage_assigned_status": lar.get("family_bootstrap_status"),
                "which_family_more_stable": more_stable,
                "which_family_more_energy_dominant": "ENERGY_DOMINANT_FAMILY",
                "interpretation_class": interp,
            })
    pair = pd.DataFrame(pair_rows)
    write_dataframe(pair, output_root / "profile_validation" / "profile_energy_family_pair_comparison_v10_5_d.csv")
    return fam, pair, pd.DataFrame(), pd.DataFrame()


def make_strength_stability_matrix(
    main_audit: pd.DataFrame,
    fam_boot: pd.DataFrame,
    pair_comparison: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for _, r in main_audit.iterrows():
        rows.append({
            "method_layer": "MAIN_JOINT_DETECTOR",
            "window_id": r.get("strict_accepted_window_id") if bool(r.get("strict_accepted_flag", False)) else r.get("v6_1_window_id"),
            "object": "joint_all",
            "candidate_day": r.get("candidate_day"),
            "candidate_family_label": f"joint:{r.get('candidate_id')}@{r.get('candidate_day')}|{r.get('lineage_status')}",
            "strength_rank": r.get("joint_detector_score_rank"),
            "strength_score_or_ratio": r.get("joint_detector_score"),
            "bootstrap_match_fraction": r.get("bootstrap_match_fraction"),
            "bootstrap_topk_fraction": np.nan,
            "lineage_status": r.get("lineage_status"),
            "strict_accepted_flag": r.get("strict_accepted_flag"),
            "family_role": "JOINT_CANDIDATE",
            "strength_stability_class": r.get("strength_stability_class"),
            "recommended_usage": "MAIN_STRICT_WINDOW_CANDIDATE" if bool(r.get("strict_accepted_flag", False)) else "KEEP_AS_CANDIDATE_NOT_MAIN",
        })
    for _, r in fam_boot.iterrows():
        st = str(r.get("family_bootstrap_status"))
        if st in ("STABLE_TOP1_FAMILY", "STABLE_TOPK_FAMILY") and str(r.get("family_role")) == "ENERGY_DOMINANT_FAMILY":
            usage = "PROFILE_ENERGY_STRONG_STABLE_COMPETITOR"
        elif st in ("STABLE_TOP1_FAMILY", "STABLE_TOPK_FAMILY") and str(r.get("family_role")) == "LINEAGE_ASSIGNED_FAMILY":
            usage = "LINEAGE_ASSIGNED_SECONDARY_SUPPORTED"
        elif str(r.get("family_role")) == "ENERGY_DOMINANT_FAMILY":
            usage = "PROFILE_ENERGY_STRONG_BUT_UNSTABLE"
        else:
            usage = "KEEP_AS_CANDIDATE_NOT_MAIN"
        rows.append({
            "method_layer": "PROFILE_ENERGY_VALIDATION",
            "window_id": r.get("window_id"),
            "object": r.get("object"),
            "candidate_day": r.get("family_day"),
            "candidate_family_label": r.get("candidate_family_label"),
            "strength_rank": np.nan if str(r.get("family_role")) != "ENERGY_DOMINANT_FAMILY" else 1,
            "strength_score_or_ratio": r.get("observed_score_ratio_to_top1"),
            "bootstrap_match_fraction": np.nan,
            "bootstrap_topk_fraction": r.get("bootstrap_topk_match_fraction"),
            "lineage_status": r.get("nearest_joint_lineage_status"),
            "strict_accepted_flag": np.nan,
            "family_role": r.get("family_role"),
            "strength_stability_class": r.get("family_bootstrap_status"),
            "recommended_usage": usage,
        })
    matrix = pd.DataFrame(rows)
    write_dataframe(matrix, output_root / "profile_validation" / "candidate_family_strength_stability_matrix_v10_5_d.csv")

    key_cases = pair_comparison[pair_comparison["window_id"].astype(str).isin(["W045", "W113", "W160"])].copy() if not pair_comparison.empty else pd.DataFrame()
    if not key_cases.empty:
        key_cases = key_cases[key_cases["object"].astype(str).isin(["H", "P", "Jw"])].copy()
        # Keep the three known candidate-family competition cases plus all k rows.
        keep_mask = (
            ((key_cases["window_id"].astype(str) == "W045") & (key_cases["object"].astype(str) == "H")) |
            ((key_cases["window_id"].astype(str) == "W113") & (key_cases["object"].astype(str) == "P")) |
            ((key_cases["window_id"].astype(str) == "W160") & (key_cases["object"].astype(str) == "Jw"))
        )
        key_cases = key_cases[keep_mask].copy()
        key_cases["case_id"] = key_cases["window_id"].astype(str) + "_" + key_cases["object"].astype(str)
        key_cases["case_interpretation"] = key_cases["interpretation_class"]
    write_dataframe(key_cases, output_root / "profile_validation" / "key_competition_cases_v10_5_d.csv")
    return matrix, key_cases


def write_strength_stability_summary_md(
    output_root: Path,
    main_audit: pd.DataFrame,
    fam_boot: pd.DataFrame,
    pair_comparison: pd.DataFrame,
    key_cases: pd.DataFrame,
    settings: Settings,
) -> None:
    lines = [
        "# V10.5_d strength-stability audit summary",
        "",
        "## Scope",
        "",
        "This audit separates peak strength from year-bootstrap stability.",
        "It asks whether non-strict main-method peaks are high-score but lower-bootstrap, and whether profile-energy top1 families are stable under year resampling.",
        "It does not reinterpret physical mechanisms or redefine accepted windows.",
        "",
        "## Run settings",
        "",
        f"- profile_energy_bootstrap_n: `{settings.validation.profile_energy_bootstrap_n}`",
        f"- profile_energy_bootstrap_seed: `{settings.validation.profile_energy_bootstrap_seed}`",
        f"- energy_k values: `{list(settings.validation.k_values)}`",
        "",
        "## Main-method strength-stability classes",
        "",
        "```json",
        json.dumps(main_audit.get("strength_stability_class", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Profile-energy family bootstrap statuses",
        "",
        "```json",
        json.dumps(fam_boot.get("family_bootstrap_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Pair comparison interpretation classes",
        "",
        "```json",
        json.dumps(pair_comparison.get("interpretation_class", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Key output files",
        "",
        "- `profile_validation/main_method_peak_strength_vs_bootstrap_v10_5_d.csv`",
        "- `profile_validation/profile_energy_family_bootstrap_stability_v10_5_d.csv`",
        "- `profile_validation/profile_energy_family_pair_comparison_v10_5_d.csv`",
        "- `profile_validation/candidate_family_strength_stability_matrix_v10_5_d.csv`",
        "- `profile_validation/key_competition_cases_v10_5_d.csv`",
        "",
        "## Interpretation boundary",
        "",
        "A high profile-energy score is not equivalent to strict accepted status. A high detector score is not equivalent to bootstrap stability. These tables are a method-layer audit only.",
    ]
    (output_root / "FIELD_INDEX_VALIDATION_V10_5_D_STRENGTH_STABILITY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

def make_index_metric_outputs(
    profiles: dict[str, ObjectProfile],
    baseline_assign: pd.DataFrame,
    settings: Settings,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_peak_rows: list[dict[str, Any]] = []
    object_summary_rows: list[dict[str, Any]] = []
    for _, a in baseline_assign.iterrows():
        lineage = str(a["lineage_id"])
        obj = str(a["object"])
        lineage_day = int(a["lineage_day"])
        assigned_day = float(a["assigned_candidate_day"]) if pd.notna(a.get("assigned_candidate_day")) else np.nan
        metrics = metric_time_series(profiles[obj])
        rels_for_object: list[str] = []
        for metric_name, series in metrics.items():
            for k in settings.validation.k_values:
                score = rolling_metric_prepost_score(series, k)
                peak_day, peak_score = select_peak_within_window(score, lineage_day, settings.validation.window_half_width_days)
                relation = relation_to_assigned(peak_day, assigned_day, settings.validation.support_days, settings.validation.near_days)
                metric_peak_rows.append(
                    {
                        "window_id": lineage,
                        "lineage_day": lineage_day,
                        "object": obj,
                        "metric": metric_name,
                        "k": int(k),
                        "v10_4_assigned_peak_day": assigned_day,
                        "metric_peak_day": peak_day,
                        "metric_peak_score": peak_score,
                        "distance_to_v10_4_peak": float(abs(peak_day - assigned_day)) if np.isfinite(peak_day) and np.isfinite(assigned_day) else np.nan,
                        "timing_relation": relation,
                    }
                )
                if k == settings.validation.primary_k:
                    rels_for_object.append(relation)
        n_support = sum(r == "SUPPORTS_OBJECT_PEAK_FAMILY" for r in rels_for_object)
        n_near = sum(r == "NEAR_SUPPORT" for r in rels_for_object)
        n_diff = sum(r == "DIFFERENT_TIMING" for r in rels_for_object)
        n_total = len(rels_for_object)
        object_summary_rows.append(
            {
                "window_id": lineage,
                "lineage_day": lineage_day,
                "object": obj,
                "primary_k": settings.validation.primary_k,
                "n_metrics": n_total,
                "n_support": n_support,
                "n_near_support": n_near,
                "n_different": n_diff,
                "object_index_validation_status": validation_status_from_counts(n_support, n_near, n_diff, n_total),
            }
        )
    metric_df = pd.DataFrame(metric_peak_rows)
    obj_df = pd.DataFrame(object_summary_rows)
    write_dataframe(metric_df, output_root / "index_validation" / "object_metric_timing_by_window_v10_5_a.csv")
    write_dataframe(obj_df, output_root / "index_validation" / "object_index_support_summary_v10_5_a.csv")
    return metric_df, obj_df


def make_validation_summary(
    baseline_assign: pd.DataFrame,
    profile_primary: pd.DataFrame,
    index_summary: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    rows = []
    idx_lookup = {(str(r.window_id), str(r.object)): r for r in index_summary.itertuples(index=False)}
    prof_lookup = {(str(r.window_id), str(r.object)): r for r in profile_primary.itertuples(index=False)}
    for _, a in baseline_assign.iterrows():
        lineage = str(a["lineage_id"])
        obj = str(a["object"])
        key = (lineage, obj)
        prof = prof_lookup.get(key)
        idx = idx_lookup.get(key)
        field_status = "NOT_EVALUATED"
        if prof is not None:
            rel = str(prof.profile_energy_relation)
            field_status = "SUPPORTED" if rel == "SUPPORTS_OBJECT_PEAK_FAMILY" else ("PARTIALLY_SUPPORTED" if rel == "NEAR_SUPPORT" else ("NOT_SUPPORTED" if rel == "DIFFERENT_TIMING" else "NOT_EVALUATED"))
        index_status = getattr(idx, "object_index_validation_status", "NOT_EVALUATED") if idx is not None else "NOT_EVALUATED"
        if field_status == "SUPPORTED" and index_status in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
            overall = "SUPPORTED"
        elif field_status in ("SUPPORTED", "PARTIALLY_SUPPORTED") or index_status in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
            overall = "PARTIALLY_SUPPORTED"
        elif field_status == "NOT_SUPPORTED" and index_status == "NOT_SUPPORTED":
            overall = "NOT_SUPPORTED"
        else:
            overall = "AMBIGUOUS"
        rows.append(
            {
                "window_id": lineage,
                "lineage_day": int(a["lineage_day"]),
                "object": obj,
                "v10_4_assigned_peak_day": a.get("assigned_candidate_day", np.nan),
                "profile_energy_peak_day": getattr(prof, "profile_energy_peak_day", np.nan) if prof is not None else np.nan,
                "profile_energy_relation": getattr(prof, "profile_energy_relation", "NOT_EVALUATED") if prof is not None else "NOT_EVALUATED",
                "n_index_metrics": getattr(idx, "n_metrics", 0) if idx is not None else 0,
                "n_index_support": getattr(idx, "n_support", 0) if idx is not None else 0,
                "n_index_near_support": getattr(idx, "n_near_support", 0) if idx is not None else 0,
                "n_index_different": getattr(idx, "n_different", 0) if idx is not None else 0,
                "field_profile_validation_status": field_status,
                "index_validation_status": index_status,
                "overall_validation_status": overall,
                "note": "method-layer validation only; no physical/causal interpretation",
            }
        )
    df = pd.DataFrame(rows)
    write_dataframe(df, output_root / "validation_summary_v10_5_a.csv")
    return df


def make_order_validation(
    pairwise: pd.DataFrame,
    profile_primary: pd.DataFrame,
    metric_df: pd.DataFrame,
    settings: Settings,
    output_root: Path,
) -> pd.DataFrame:
    base_pairs = pairwise[(pairwise["config_id"].astype(str) == "BASELINE") & (pairwise["lineage_id"].astype(str).isin(settings.validation.target_lineages))].copy()
    profile_day_lookup = {(str(r.window_id), str(r.object)): float(r.profile_energy_peak_day) for r in profile_primary.itertuples(index=False) if np.isfinite(r.profile_energy_peak_day)}
    # Metric consensus: median primary-k metric peak day per window/object.
    msub = metric_df[metric_df["k"] == settings.validation.primary_k].copy()
    metric_cons = msub.groupby(["window_id", "object"], as_index=False).agg(index_consensus_peak_day=("metric_peak_day", "median"))
    metric_lookup = {(str(r.window_id), str(r.object)): float(r.index_consensus_peak_day) for r in metric_cons.itertuples(index=False) if np.isfinite(r.index_consensus_peak_day)}
    rows = []
    for _, r in base_pairs.iterrows():
        lineage = str(r["lineage_id"])
        a = str(r["object_a"])
        b = str(r["object_b"])
        v10_order = str(r.get("order_tau2", "MISSING"))
        pa = profile_day_lookup.get((lineage, a), np.nan)
        pb = profile_day_lookup.get((lineage, b), np.nan)
        prof_order = order_with_tau(pa, pb, settings.validation.order_tau_days)
        ia = metric_lookup.get((lineage, a), np.nan)
        ib = metric_lookup.get((lineage, b), np.nan)
        idx_order = order_with_tau(ia, ib, settings.validation.order_tau_days)
        prof_status = order_validation(v10_order, prof_order)
        idx_status = order_validation(v10_order, idx_order)
        if prof_status in ("ORDER_SUPPORTED", "ORDER_NEAR_TIE_SUPPORTED") and idx_status in ("ORDER_SUPPORTED", "ORDER_NEAR_TIE_SUPPORTED", "ORDER_AMBIGUOUS"):
            combined = prof_status if prof_status == "ORDER_SUPPORTED" else "ORDER_NEAR_TIE_SUPPORTED"
        elif idx_status in ("ORDER_SUPPORTED", "ORDER_NEAR_TIE_SUPPORTED") and prof_status == "ORDER_AMBIGUOUS":
            combined = idx_status
        elif prof_status == "ORDER_NOT_SUPPORTED" and idx_status == "ORDER_NOT_SUPPORTED":
            combined = "ORDER_NOT_SUPPORTED"
        else:
            combined = "ORDER_AMBIGUOUS"
        rows.append(
            {
                "window_id": lineage,
                "lineage_day": int(r["lineage_day"]),
                "object_a": a,
                "object_b": b,
                "v10_4_order_tau2": v10_order,
                "v10_4_day_a": r.get("day_a", np.nan),
                "v10_4_day_b": r.get("day_b", np.nan),
                "profile_energy_day_a": pa,
                "profile_energy_day_b": pb,
                "profile_energy_order_tau2": prof_order,
                "index_consensus_day_a": ia,
                "index_consensus_day_b": ib,
                "index_order_tau2": idx_order,
                "profile_order_validation_status": prof_status,
                "index_order_validation_status": idx_status,
                "order_validation_status": combined,
                "reason": "tau2 comparison; profile/index validation is external support, not causal proof",
            }
        )
    df = pd.DataFrame(rows)
    write_dataframe(df, output_root / "order_validation" / "object_order_validation_by_window_v10_5_a.csv")
    return df


def make_figures(curve_df: pd.DataFrame, output_root: Path, settings: Settings) -> None:
    if not settings.validation.emit_figures or plt is None:
        return
    fig_dir = output_root / "figures"
    for lineage, sub in curve_df.groupby("window_id"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for obj, g in sub.groupby("object"):
            g = g.sort_values("day")
            ax.plot(g["day"], g["profile_energy_score"], label=obj)
        ax.set_title(f"{lineage} profile pre/post energy (k={settings.validation.primary_k})")
        ax.set_xlabel("Day index (Apr 1 = 0)")
        ax.set_ylabel("Profile pre/post contrast norm")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{lineage}_profile_energy_all_objects_v10_5_a.png", dpi=160)
        plt.close(fig)


def write_summary_md(
    output_root: Path,
    validation_summary: pd.DataFrame,
    order_validation: pd.DataFrame,
    run_meta: dict[str, Any],
    family_summary: pd.DataFrame | None = None,
    switch_inventory: pd.DataFrame | None = None,
) -> None:
    status_counts = validation_summary["overall_validation_status"].value_counts(dropna=False).to_dict() if not validation_summary.empty else {}
    order_counts = order_validation["order_validation_status"].value_counts(dropna=False).to_dict() if not order_validation.empty else {}
    family_counts = family_summary["candidate_family_validation_status"].value_counts(dropna=False).to_dict() if family_summary is not None and not family_summary.empty else {}
    switch_counts = switch_inventory["switch_type"].value_counts(dropna=False).to_dict() if switch_inventory is not None and not switch_inventory.empty else {}
    lines = [
        "# V10.5_a field/index validation summary",
        "",
        "## Scope",
        "",
        "This run validates V10.4 object-order timing skeletons for W045, W113, and W160 using profile rolling pre/post energy and simple low-dimensional profile metrics.",
        "It does not rerun peak discovery, redefine accepted windows, perform yearwise validation, or make physical/causal claims.",
        "",
        "## Run status",
        "",
        f"- status: `{run_meta.get('status')}`",
        f"- target_lineages: `{run_meta.get('target_lineages')}`",
        f"- objects: `{run_meta.get('objects')}`",
        f"- primary_k: `{run_meta.get('primary_k')}`",
        "",
        "## Object validation status counts",
        "",
        "```json",
        json.dumps(status_counts, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Object-order validation status counts",
        "",
        "```json",
        json.dumps(order_counts, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Candidate-family top-k validation counts (V10.5_b)",
        "",
        "```json",
        json.dumps(family_counts, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Candidate-family switch counts (V10.5_b)",
        "",
        "```json",
        json.dumps(switch_counts, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Key output files",
        "",
        "- `validation_summary_v10_5_a.csv`",
        "- `profile_validation/profile_energy_peak_by_window_object_k_v10_5_a.csv`",
        "- `profile_validation/profile_energy_topk_peaks_by_window_object_v10_5_b.csv`",
        "- `profile_validation/v10_4_assigned_peak_energy_rank_v10_5_b.csv`",
        "- `profile_validation/candidate_family_switch_inventory_v10_5_b.csv`",
        "- `validation_summary_candidate_family_v10_5_b.csv`",
        "- `profile_validation/candidate_family_selection_reason_audit_v10_5_c.csv`",
        "- `profile_validation/candidate_family_role_long_v10_5_c.csv`",
        "- `validation_summary_candidate_family_selection_reason_v10_5_c.csv`",
        "- `profile_validation/main_method_peak_strength_vs_bootstrap_v10_5_d.csv`",
        "- `profile_validation/profile_energy_family_bootstrap_stability_v10_5_d.csv`",
        "- `profile_validation/profile_energy_family_pair_comparison_v10_5_d.csv`",
        "- `profile_validation/candidate_family_strength_stability_matrix_v10_5_d.csv`",
        "- `profile_validation/key_competition_cases_v10_5_d.csv`",
        "- `FIELD_INDEX_VALIDATION_V10_5_D_STRENGTH_STABILITY_SUMMARY.md`",
        "- `index_validation/object_metric_timing_by_window_v10_5_a.csv`",
        "- `order_validation/object_order_validation_by_window_v10_5_a.csv`",
        "- `figures/*_profile_energy_all_objects_v10_5_a.png`",
        "",
        "## Interpretation boundary",
        "",
        "These outputs are external validation evidence for method-layer timing skeletons. They are not physical mechanism proof, causal evidence, or a re-decision of accepted windows.",
        "V10.5_b top-k outputs distinguish same-family support from candidate-family switches; a top1 mismatch is not automatically a rejection of the V10.4 assigned peak.",
        "V10.5_c selection-reason outputs classify whether the energy-top1 family was not selected because it belongs to a known non-strict lineage, another strict lineage, an object-native secondary family, or requires review.",
    ]
    (output_root / "FIELD_INDEX_VALIDATION_V10_5_A_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_field_index_validation_v10_5_a(bundle_root: Path | str | None = None) -> None:
    bundle_root = Path(bundle_root) if bundle_root is not None else Path(__file__).resolve().parents[1]
    v10_root = bundle_root.parent
    output_root = bundle_root / "outputs" / "field_index_validation_v10_5_a"
    log_root = bundle_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    start = now_utc()
    clean_output_dirs(output_root)

    input_data = load_required_inputs(v10_root, output_root)
    smoothed_path = settings.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    profiles = build_profiles(smoothed, settings.profile)

    baseline_assign = get_baseline_assignments(input_data["v10_4_assignment"], list(settings.validation.target_lineages))
    write_dataframe(baseline_assign, output_root / "audit" / "v10_4_baseline_assignments_used_v10_5_a.csv")

    profile_peak_df, profile_curve_df, profile_primary_df = make_profile_energy_outputs(profiles, baseline_assign, settings, output_root)
    topk_df, assigned_rank_df, switch_df, family_summary_df = make_profile_energy_topk_family_outputs(
        profiles,
        baseline_assign,
        input_data["v10_2_object_catalog"],
        input_data["v10_2_object_lineage"],
        settings,
        output_root,
    )
    selection_reason_df, family_role_long_df, selection_reason_summary_df = make_candidate_family_selection_reason_audit(
        topk_df, assigned_rank_df, switch_df, output_root
    )
    main_strength_df = make_main_method_peak_strength_bootstrap_audit(input_data, output_root)
    profile_family_bootstrap_df, profile_family_pair_df, _, _ = make_profile_energy_family_bootstrap_audit(
        profiles, selection_reason_df, family_role_long_df, settings, output_root
    )
    strength_stability_matrix_df, key_competition_cases_df = make_strength_stability_matrix(
        main_strength_df, profile_family_bootstrap_df, profile_family_pair_df, output_root
    )
    write_strength_stability_summary_md(
        output_root, main_strength_df, profile_family_bootstrap_df, profile_family_pair_df, key_competition_cases_df, settings
    )
    metric_df, index_summary_df = make_index_metric_outputs(profiles, baseline_assign, settings, output_root)
    validation_summary_df = make_validation_summary(baseline_assign, profile_primary_df, index_summary_df, output_root)
    order_validation_df = make_order_validation(input_data["v10_4_pairwise"], profile_primary_df, metric_df, settings, output_root)
    make_figures(profile_curve_df, output_root, settings)

    run_meta = {
        "status": "success",
        "start_time_utc": start,
        "end_time_utc": now_utc(),
        "bundle_root": str(bundle_root),
        "output_root": str(output_root),
        "smoothed_fields_path": str(smoothed_path),
        "target_lineages": list(settings.validation.target_lineages),
        "objects": OBJECTS,
        "primary_k": int(settings.validation.primary_k),
        "n_baseline_assignments": int(len(baseline_assign)),
        "n_validation_summary_rows": int(len(validation_summary_df)),
        "n_profile_peak_rows": int(len(profile_peak_df)),
        "n_metric_peak_rows": int(len(metric_df)),
        "n_order_validation_rows": int(len(order_validation_df)),
        "n_profile_energy_topk_rows_v10_5_b": int(len(topk_df)),
        "n_assigned_peak_energy_rank_rows_v10_5_b": int(len(assigned_rank_df)),
        "n_candidate_family_switch_rows_v10_5_b": int(len(switch_df)),
        "n_candidate_family_summary_rows_v10_5_b": int(len(family_summary_df)),
        "n_candidate_family_selection_reason_rows_v10_5_c": int(len(selection_reason_df)),
        "n_candidate_family_role_long_rows_v10_5_c": int(len(family_role_long_df)),
        "n_candidate_family_selection_reason_summary_rows_v10_5_c": int(len(selection_reason_summary_df)),
        "n_main_method_strength_bootstrap_rows_v10_5_d": int(len(main_strength_df)),
        "n_profile_energy_family_bootstrap_rows_v10_5_d": int(len(profile_family_bootstrap_df)),
        "n_profile_energy_family_pair_rows_v10_5_d": int(len(profile_family_pair_df)),
        "n_strength_stability_matrix_rows_v10_5_d": int(len(strength_stability_matrix_df)),
        "n_key_competition_case_rows_v10_5_d": int(len(key_competition_cases_df)),
        "profile_energy_bootstrap_n_v10_5_d": int(settings.validation.profile_energy_bootstrap_n),
        "does_not_rerun_peak_discovery": True,
        "does_not_redefine_accepted_windows": True,
        "does_not_perform_physical_interpretation": True,
        "settings": settings.to_dict(),
    }
    write_json(run_meta, output_root / "run_meta.json")
    write_json(
        {
            "status": "success",
            "overall_validation_status_counts": validation_summary_df["overall_validation_status"].value_counts(dropna=False).to_dict(),
            "order_validation_status_counts": order_validation_df["order_validation_status"].value_counts(dropna=False).to_dict(),
            "candidate_family_validation_status_counts_v10_5_b": family_summary_df["candidate_family_validation_status"].value_counts(dropna=False).to_dict(),
            "candidate_family_switch_type_counts_v10_5_b": switch_df["switch_type"].value_counts(dropna=False).to_dict(),
            "candidate_family_nonselection_reason_counts_v10_5_c": selection_reason_df["top1_nonselection_reason"].value_counts(dropna=False).to_dict() if not selection_reason_df.empty else {},
            "candidate_family_competition_status_counts_v10_5_c": selection_reason_df["family_competition_status"].value_counts(dropna=False).to_dict() if not selection_reason_df.empty else {},
            "main_method_strength_stability_class_counts_v10_5_d": main_strength_df["strength_stability_class"].value_counts(dropna=False).to_dict() if not main_strength_df.empty else {},
            "profile_energy_family_bootstrap_status_counts_v10_5_d": profile_family_bootstrap_df["family_bootstrap_status"].value_counts(dropna=False).to_dict() if not profile_family_bootstrap_df.empty else {},
            "profile_energy_pair_interpretation_counts_v10_5_d": profile_family_pair_df["interpretation_class"].value_counts(dropna=False).to_dict() if not profile_family_pair_df.empty else {},
        },
        output_root / "summary.json",
    )
    write_summary_md(output_root, validation_summary_df, order_validation_df, run_meta, family_summary_df, switch_df)
    (log_root / "last_run.txt").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    run_field_index_validation_v10_5_a()
