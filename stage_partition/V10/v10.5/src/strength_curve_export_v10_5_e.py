from __future__ import annotations

"""
V10.5_e full-season strength curve export.

Purpose
-------
Export continuous 4–9 month strength/score curves from both layers:

1. Main-method curves:
   - joint_all ruptures.Window detector score from V10.1;
   - object-native ruptures.Window detector scores from V10.2.

2. Detector-external validation curves:
   - full-season rolling pre/post profile-energy curves for P/V/H/Je/Jw;
   - k = 7, 9, 11 by default.

This is a curve-diagnostic export only. It does not rerun peak discovery,
bootstrap, accepted-window selection, object-order analysis, or physical
interpretation. It is meant to expose the full continuous score landscape so
that single-peak vs multi-peak structure can be audited directly.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import shutil

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


OBJECTS = ["P", "V", "H", "Je", "Jw"]
K_VALUES = (7, 9, 11)
PROFILE_ENERGY_GLOBAL_TOP_K = 15
LOCAL_PEAK_MIN_DISTANCE_DAYS = 3


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
class CurveExportConfig:
    k_values: tuple[int, ...] = K_VALUES
    global_top_k: int = PROFILE_ENERGY_GLOBAL_TOP_K
    local_peak_min_distance_days: int = LOCAL_PEAK_MIN_DISTANCE_DAYS
    emit_figures: bool = True


@dataclass
class Settings:
    foundation: FoundationInputConfig = field(default_factory=FoundationInputConfig)
    profile: ProfileGridConfig = field(default_factory=ProfileGridConfig)
    curve_export: CurveExportConfig = field(default_factory=CurveExportConfig)
    output_tag: str = "strength_curve_export_v10_5_e"

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
    for sub in ["curves", "markers", "figures", "audit"]:
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
    out: dict[str, ObjectProfile] = {}
    for obj, (field_key, lon_range, lat_range) in specs.items():
        if field_key not in smoothed:
            raise KeyError(f"Missing field {field_key} in smoothed_fields")
        cube, lat_grid = _build_profile_from_field(smoothed[field_key], lat, lon, lon_range, lat_range, cfg.lat_step_deg)
        seasonal = safe_nanmean(cube, axis=0)
        z = _zscore_along_day(seasonal)
        out[obj] = ObjectProfile(obj, cube, seasonal, z, lat_grid, lon_range, lat_range)
    return out


def rolling_profile_prepost_energy(profile: np.ndarray, k: int) -> pd.Series:
    arr = np.asarray(profile, dtype=float)
    n = arr.shape[0]
    vals = np.full(n, np.nan, dtype=float)
    for d in range(n):
        lo = d - int(k)
        hi = d + 1 + int(k)
        if lo < 0 or hi > n:
            continue
        before = safe_nanmean(arr[lo:d, :], axis=0)
        after = safe_nanmean(arr[d + 1 : hi, :], axis=0)
        diff = after - before
        if np.any(np.isfinite(diff)):
            vals[d] = float(np.sqrt(np.nanmean(np.square(diff))))
    return pd.Series(vals, index=np.arange(n), name=f"profile_energy_k{k}")


def normalize_series(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    z = np.full_like(arr, np.nan, dtype=float)
    mm = np.full_like(arr, np.nan, dtype=float)
    if finite.sum() > 1:
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if np.isfinite(std) and std > 1e-12:
            z[finite] = (arr[finite] - mean) / std
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if np.isfinite(mx - mn) and (mx - mn) > 1e-12:
            mm[finite] = (arr[finite] - mn) / (mx - mn)
    return z, mm


def topk_local_peaks_fullseason(score: pd.Series, top_k: int, min_distance_days: int) -> pd.DataFrame:
    sub = score.dropna().sort_index()
    rows: list[dict[str, Any]] = []
    if sub.empty:
        return pd.DataFrame(columns=["peak_rank", "peak_day", "peak_score"])
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
                "peak_rank": int(rank),
                "peak_day": int(d),
                "peak_month_day": day_index_to_month_day(int(d)),
                "peak_score": float(v),
                "score_ratio_to_top1": float(v / top1) if np.isfinite(top1) and top1 != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def nearest_day_record(day: float, candidates: pd.DataFrame, object_name: str | None = None) -> dict[str, Any]:
    if candidates is None or candidates.empty or not np.isfinite(day):
        return {}
    df = candidates.copy()
    if object_name is not None and "object" in df.columns:
        df = df[df["object"].astype(str) == str(object_name)]
    if df.empty:
        return {}
    day_col = "candidate_day" if "candidate_day" in df.columns else ("point_day" if "point_day" in df.columns else None)
    if day_col is None:
        return {}
    df["_dist"] = (df[day_col].astype(float) - float(day)).abs()
    row = df.sort_values(["_dist", day_col]).iloc[0].to_dict()
    return row


def load_input_inventory(v10_root: Path, output_root: Path) -> dict[str, Any]:
    paths = {
        "v10_1_joint_detector_profile": v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "joint_point_layer" / "joint_detector_profile_v10_1.csv",
        "v10_1_joint_candidate_registry": v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "joint_point_layer" / "joint_candidate_registry_v10_1.csv",
        "v10_1_joint_lineage": v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "lineage" / "joint_main_window_lineage_v10_1.csv",
        "v10_2_object_catalog": v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "cross_object" / "object_native_candidate_catalog_all_objects_v10_2.csv",
        "v10_2_object_lineage": v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "lineage_mapping" / "object_candidate_to_joint_lineage_v10_2.csv",
    }
    for obj in OBJECTS:
        paths[f"v10_2_{obj}_detector_scores"] = v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "by_object" / obj / f"{obj}_object_detector_scores_v10_2.csv"
    inv_rows = []
    data: dict[str, Any] = {}
    for key, path in paths.items():
        df = safe_read_csv(path)
        inv_rows.append({"input_name": key, "path": str(path), "exists": path.exists(), "status": "loaded" if df is not None else "missing_or_unreadable", "n_rows": 0 if df is None else len(df)})
        data[key] = df
    write_dataframe(pd.DataFrame(inv_rows), output_root / "audit" / "input_inventory_v10_5_e.csv")
    return data


def standardize_detector_profile(df: pd.DataFrame, scope: str, object_name: str | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Accept historical column variants.
    if "profile_score" in out.columns and "detector_score" not in out.columns:
        out = out.rename(columns={"profile_score": "detector_score"})
    if "score" in out.columns and "detector_score" not in out.columns:
        out = out.rename(columns={"score": "detector_score"})
    if "day" not in out.columns:
        for c in out.columns:
            if str(c).lower() in ("index", "point_day"):
                out = out.rename(columns={c: "day"})
                break
    if "day" not in out.columns or "detector_score" not in out.columns:
        return pd.DataFrame()
    out = out[["day", "detector_score"]].copy()
    out["day"] = out["day"].astype(int)
    out["month_day"] = out["day"].map(day_index_to_month_day)
    out["scope"] = scope
    out["object"] = object_name if object_name is not None else "joint_all"
    out["method_family"] = "main_method_ruptures_window"
    out["score_name"] = "detector_score"
    z, mm = normalize_series(out["detector_score"])
    out["score_z"] = z
    out["score_minmax"] = mm
    return out[["method_family", "scope", "object", "score_name", "day", "month_day", "detector_score", "score_z", "score_minmax"]]


def export_main_method_curves(v10_root: Path, data: dict[str, Any], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    joint = standardize_detector_profile(data.get("v10_1_joint_detector_profile"), "joint_all", None)
    if not joint.empty:
        rows.append(joint)
    for obj in OBJECTS:
        odf = standardize_detector_profile(data.get(f"v10_2_{obj}_detector_scores"), f"{obj}_only", obj)
        if not odf.empty:
            rows.append(odf)
    curves = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_dataframe(curves, output_root / "curves" / "main_method_continuous_detector_score_curves_v10_5_e.csv")

    # Markers: joint candidates + object-native candidates.
    marker_rows: list[dict[str, Any]] = []
    joint_cat = data.get("v10_1_joint_candidate_registry")
    if joint_cat is not None and not joint_cat.empty:
        jc = joint_cat.copy()
        if "candidate_day" not in jc.columns and "point_day" in jc.columns:
            jc = jc.rename(columns={"point_day": "candidate_day"})
        if "peak_score" in jc.columns and "candidate_score" not in jc.columns:
            jc = jc.rename(columns={"peak_score": "candidate_score"})
        for _, r in jc.iterrows():
            day = r.get("candidate_day", r.get("peak_day", np.nan))
            marker_rows.append({
                "method_family": "main_method_ruptures_window",
                "scope": "joint_all",
                "object": "joint_all",
                "candidate_id": r.get("candidate_id", r.get("peak_id")),
                "candidate_day": day,
                "candidate_month_day": day_index_to_month_day(int(day)) if pd.notna(day) else None,
                "candidate_score": r.get("candidate_score", r.get("peak_score")),
                "candidate_rank": r.get("candidate_rank", r.get("peak_rank")),
                "candidate_role": "joint_candidate",
                "lineage_status": None,
            })
    obj_cat = data.get("v10_2_object_catalog")
    if obj_cat is not None and not obj_cat.empty:
        oc = obj_cat.copy()
        if "candidate_day" not in oc.columns and "point_day" in oc.columns:
            oc = oc.rename(columns={"point_day": "candidate_day"})
        for _, r in oc.iterrows():
            day = r.get("candidate_day", np.nan)
            marker_rows.append({
                "method_family": "main_method_ruptures_window",
                "scope": f"{r.get('object')}_only",
                "object": r.get("object"),
                "candidate_id": r.get("candidate_id"),
                "candidate_day": day,
                "candidate_month_day": day_index_to_month_day(int(day)) if pd.notna(day) else None,
                "candidate_score": r.get("candidate_score", r.get("peak_score")),
                "candidate_rank": r.get("candidate_rank", r.get("peak_rank")),
                "candidate_role": "object_native_candidate",
                "lineage_status": r.get("nearest_joint_lineage_status", None),
            })
    markers = pd.DataFrame(marker_rows)
    write_dataframe(markers, output_root / "markers" / "main_method_candidate_markers_v10_5_e.csv")
    return curves, markers


def export_profile_energy_curves(
    profiles: dict[str, ObjectProfile],
    settings: Settings,
    output_root: Path,
    object_ref: pd.DataFrame,
    joint_lineage: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    for obj, prof in profiles.items():
        for k in settings.curve_export.k_values:
            score = rolling_profile_prepost_energy(prof.z_seasonal, int(k))
            z, mm = normalize_series(score)
            for i, (day, val) in enumerate(score.items()):
                curve_rows.append({
                    "method_family": "detector_external_profile_energy",
                    "scope": f"{obj}_only",
                    "object": obj,
                    "score_name": f"profile_prepost_energy_k{k}",
                    "k": int(k),
                    "day": int(day),
                    "month_day": day_index_to_month_day(int(day)),
                    "profile_energy_score": float(val) if np.isfinite(val) else np.nan,
                    "score_z": float(z[i]) if np.isfinite(z[i]) else np.nan,
                    "score_minmax": float(mm[i]) if np.isfinite(mm[i]) else np.nan,
                })
            top = topk_local_peaks_fullseason(score, settings.curve_export.global_top_k, settings.curve_export.local_peak_min_distance_days)
            for _, r in top.iterrows():
                day = float(r["peak_day"])
                nr = nearest_day_record(day, object_ref, obj)
                jl = None
                if joint_lineage is not None and not joint_lineage.empty:
                    cand_day_col = "candidate_day" if "candidate_day" in joint_lineage.columns else None
                    if cand_day_col is not None:
                        jtmp = joint_lineage.copy()
                        jtmp["_dist"] = (jtmp[cand_day_col].astype(float) - day).abs()
                        jl = jtmp.sort_values(["_dist", cand_day_col]).iloc[0].to_dict()
                peak_rows.append({
                    "method_family": "detector_external_profile_energy",
                    "scope": f"{obj}_only",
                    "object": obj,
                    "k": int(k),
                    "energy_peak_rank_fullseason": int(r["peak_rank"]),
                    "energy_peak_day": int(r["peak_day"]),
                    "energy_peak_month_day": r.get("peak_month_day"),
                    "energy_peak_score": float(r["peak_score"]),
                    "score_ratio_to_top1": r.get("score_ratio_to_top1"),
                    "nearest_object_candidate_id": nr.get("candidate_id"),
                    "nearest_object_candidate_day": nr.get("candidate_day", nr.get("point_day")),
                    "distance_to_nearest_object_candidate": nr.get("_dist"),
                    "nearest_object_support_class": nr.get("object_support_class"),
                    "nearest_object_bootstrap_match_fraction": nr.get("bootstrap_match_fraction"),
                    "nearest_joint_lineage_day": None if jl is None else jl.get("candidate_day"),
                    "nearest_joint_lineage_status": None if jl is None else jl.get("lineage_status"),
                    "distance_to_nearest_joint_lineage": None if jl is None else jl.get("_dist"),
                })
    curves = pd.DataFrame(curve_rows)
    peaks = pd.DataFrame(peak_rows)
    write_dataframe(curves, output_root / "curves" / "profile_energy_continuous_curves_fullseason_v10_5_e.csv")
    write_dataframe(peaks, output_root / "markers" / "profile_energy_global_topk_peaks_v10_5_e.csv")
    return curves, peaks


def make_combined_curves(main_curves: pd.DataFrame, energy_curves: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    if main_curves is not None and not main_curves.empty:
        m = main_curves.rename(columns={"detector_score": "raw_score"}).copy()
        m["parameter_name"] = "detector_width"
        m["parameter_value"] = "baseline"
        rows.append(m[["method_family", "scope", "object", "score_name", "parameter_name", "parameter_value", "day", "month_day", "raw_score", "score_z", "score_minmax"]])
    if energy_curves is not None and not energy_curves.empty:
        e = energy_curves.rename(columns={"profile_energy_score": "raw_score"}).copy()
        e["parameter_name"] = "profile_energy_k"
        e["parameter_value"] = e["k"].astype(str)
        rows.append(e[["method_family", "scope", "object", "score_name", "parameter_name", "parameter_value", "day", "month_day", "raw_score", "score_z", "score_minmax"]])
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_dataframe(combined, output_root / "curves" / "combined_strength_curves_long_v10_5_e.csv")
    return combined


def make_figures(main_curves: pd.DataFrame, energy_curves: pd.DataFrame, output_root: Path, settings: Settings) -> None:
    if not settings.curve_export.emit_figures or plt is None:
        return
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Main method normalized detector curves.
    if main_curves is not None and not main_curves.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        for scope, g in main_curves.groupby("scope"):
            g = g.sort_values("day")
            ax.plot(g["day"], g["score_minmax"], label=scope, linewidth=1.4)
        ax.set_title("Main-method detector score curves, full season (min-max normalized)")
        ax.set_xlabel("Day index (Apr 1 = 0)")
        ax.set_ylabel("Normalized detector score")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "main_method_detector_score_curves_fullseason_v10_5_e.png", dpi=160)
        plt.close(fig)

    # Profile energy k=9 normalized curves.
    if energy_curves is not None and not energy_curves.empty:
        for k in settings.curve_export.k_values:
            sub = energy_curves[energy_curves["k"].astype(int) == int(k)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(11, 5))
            for obj, g in sub.groupby("object"):
                g = g.sort_values("day")
                ax.plot(g["day"], g["score_minmax"], label=obj, linewidth=1.4)
            ax.set_title(f"Profile pre/post energy curves, full season (k={k}, min-max normalized)")
            ax.set_xlabel("Day index (Apr 1 = 0)")
            ax.set_ylabel("Normalized profile-energy")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / f"profile_energy_curves_fullseason_k{k}_v10_5_e.png", dpi=160)
            plt.close(fig)

        # Per-object overlay: main object detector score + profile-energy k variants.
        if main_curves is not None and not main_curves.empty:
            for obj in OBJECTS:
                fig, ax = plt.subplots(figsize=(11, 5))
                mg = main_curves[main_curves["object"].astype(str) == obj].sort_values("day")
                if not mg.empty:
                    ax.plot(mg["day"], mg["score_minmax"], label="main detector score", linewidth=1.8)
                eg = energy_curves[energy_curves["object"].astype(str) == obj]
                for k, g in eg.groupby("k"):
                    g = g.sort_values("day")
                    ax.plot(g["day"], g["score_minmax"], label=f"profile energy k={k}", linewidth=1.1, alpha=0.85)
                ax.set_title(f"{obj}: main-method vs profile-energy strength curves")
                ax.set_xlabel("Day index (Apr 1 = 0)")
                ax.set_ylabel("Min-max normalized score")
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(fig_dir / f"{obj}_main_vs_profile_energy_curves_v10_5_e.png", dpi=160)
                plt.close(fig)


def write_summary(output_root: Path, run_meta: dict[str, Any]) -> None:
    lines = [
        "# V10.5_e full-season strength curve export summary",
        "",
        "## Scope",
        "",
        "This export pulls continuous 4–9 month strength curves from the main-method detector layer and the detector-external profile-energy validation layer.",
        "It does not rerun peak discovery, bootstrap, object-order analysis, accepted-window selection, or physical interpretation.",
        "",
        "## Main outputs",
        "",
        "- `curves/main_method_continuous_detector_score_curves_v10_5_e.csv`",
        "- `curves/profile_energy_continuous_curves_fullseason_v10_5_e.csv`",
        "- `curves/combined_strength_curves_long_v10_5_e.csv`",
        "- `markers/main_method_candidate_markers_v10_5_e.csv`",
        "- `markers/profile_energy_global_topk_peaks_v10_5_e.csv`",
        "- `figures/main_method_detector_score_curves_fullseason_v10_5_e.png`",
        "- `figures/profile_energy_curves_fullseason_k*_v10_5_e.png`",
        "- `figures/*_main_vs_profile_energy_curves_v10_5_e.png`",
        "",
        "## Interpretation boundary",
        "",
        "The exported curves expose score landscapes and multi-peak structure. They are diagnostic evidence only. A high score does not imply strict accepted status, physical mechanism, causal order, or year-bootstrap stability.",
        "",
        "## Run meta",
        "",
        "```json",
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str),
        "```",
    ]
    (output_root / "STRENGTH_CURVE_EXPORT_V10_5_E_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_strength_curve_export_v10_5_e(bundle_root: Path | str | None = None) -> None:
    bundle_root = Path(bundle_root) if bundle_root is not None else Path(__file__).resolve().parents[1]
    v10_root = bundle_root.parent
    output_root = bundle_root / "outputs" / "strength_curve_export_v10_5_e"
    log_root = bundle_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    start = now_utc()
    clean_output_dirs(output_root)

    data = load_input_inventory(v10_root, output_root)
    main_curves, main_markers = export_main_method_curves(v10_root, data, output_root)

    smoothed_path = settings.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    profiles = build_profiles(smoothed, settings.profile)
    object_catalog = data.get("v10_2_object_catalog", pd.DataFrame())
    object_lineage = data.get("v10_2_object_lineage", pd.DataFrame())
    object_ref = object_catalog.copy() if object_catalog is not None else pd.DataFrame()
    if object_ref is not None and not object_ref.empty:
        if "candidate_day" not in object_ref.columns and "point_day" in object_ref.columns:
            object_ref = object_ref.rename(columns={"point_day": "candidate_day"})
        if object_lineage is not None and not object_lineage.empty:
            lin = object_lineage.copy()
            keep = [c for c in ["object", "candidate_id", "candidate_day", "nearest_joint_candidate_day", "distance_to_nearest_joint_candidate", "nearest_joint_lineage_status", "nearest_strict_accepted_window_id", "v10_window_id_if_selected"] if c in lin.columns]
            if keep and set(["object", "candidate_id", "candidate_day"]).issubset(set(keep)):
                object_ref = object_ref.merge(lin[keep], on=["object", "candidate_id", "candidate_day"], how="left", suffixes=("", "_lineage"))

    joint_lineage = data.get("v10_1_joint_lineage", pd.DataFrame())
    if joint_lineage is not None and not joint_lineage.empty:
        if "candidate_day" not in joint_lineage.columns:
            for c in ["point_day", "candidate_point_day"]:
                if c in joint_lineage.columns:
                    joint_lineage = joint_lineage.rename(columns={c: "candidate_day"})
                    break

    energy_curves, energy_peaks = export_profile_energy_curves(profiles, settings, output_root, object_ref, joint_lineage)
    combined = make_combined_curves(main_curves, energy_curves, output_root)
    make_figures(main_curves, energy_curves, output_root, settings)

    run_meta = {
        "status": "success",
        "start_time_utc": start,
        "end_time_utc": now_utc(),
        "bundle_root": str(bundle_root),
        "output_root": str(output_root),
        "smoothed_fields_path": str(smoothed_path),
        "n_main_curve_rows": int(len(main_curves)),
        "n_main_marker_rows": int(len(main_markers)),
        "n_profile_energy_curve_rows": int(len(energy_curves)),
        "n_profile_energy_topk_rows": int(len(energy_peaks)),
        "n_combined_curve_rows": int(len(combined)),
        "does_not_rerun_peak_discovery": True,
        "does_not_rerun_bootstrap": True,
        "does_not_redefine_accepted_windows": True,
        "does_not_perform_physical_interpretation": True,
        "settings": settings.to_dict(),
    }
    write_json(run_meta, output_root / "run_meta.json")
    write_json(
        {
            "status": "success",
            "n_main_curve_rows": int(len(main_curves)),
            "n_profile_energy_curve_rows": int(len(energy_curves)),
            "n_profile_energy_topk_rows": int(len(energy_peaks)),
            "k_values": list(settings.curve_export.k_values),
            "objects": OBJECTS,
        },
        output_root / "summary.json",
    )
    write_summary(output_root, run_meta)
    (log_root / "last_run_strength_curve_export_v10_5_e.txt").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    run_strength_curve_export_v10_5_e()
