from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import shutil
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# V10.7_k: W33-to-W45 structure-transition mapping audit
# =============================================================================
# Method boundary:
# - This is NOT an activity-amplitude/strength-only mapping audit.
# - This is NOT a regression-control experiment that controls away W45 objects.
# - It tests whether E2/W33 structure state or structure change maps to M/W45
#   structure state or structure change.
# - It allows H_E2 structure metrics to map to non-H M metrics (P/V/Je/Jw).
# - It is heuristic route-decision evidence, not causal inference.
# =============================================================================


@dataclass(frozen=True)
class Window:
    name: str
    days: tuple[int, int]


@dataclass(frozen=True)
class ObjectSpec:
    object_name: str
    source_key_candidates: tuple[str, ...]
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    source_note: str = ""


@dataclass
class Settings:
    project_root: Path = Path(r"D:\easm_project01")
    version: str = "v10.7_k"
    output_tag: str = "w45_structure_transition_mapping_v10_7_k"
    smoothed_env_var: str = "V10_7_SMOOTHED_FIELDS"

    lat_key_candidates: tuple[str, ...] = ("lat", "latitude", "lats")
    lon_key_candidates: tuple[str, ...] = ("lon", "longitude", "lons")
    year_key_candidates: tuple[str, ...] = ("year", "years")
    day_key_candidates: tuple[str, ...] = ("day", "days", "day_index")

    object_specs: tuple[ObjectSpec, ...] = (
        ObjectSpec("P", ("precip_smoothed", "precip", "P", "pr", "rain", "tp"), (15.0, 35.0), (110.0, 140.0), "precip object structure domain"),
        ObjectSpec("V", ("v850_smoothed", "v850", "V", "v", "vwind850"), (15.0, 35.0), (110.0, 140.0), "v850 object structure domain"),
        ObjectSpec("H", ("z500_smoothed", "z500", "H", "hgt500", "geopotential_500", "zg500"), (15.0, 35.0), (110.0, 140.0), "H/WPSH-like object structure domain"),
        ObjectSpec("Je", ("u200_smoothed", "u200", "u", "uwnd200"), (25.0, 45.0), (120.0, 150.0), "derived from u200 eastern jet sector"),
        ObjectSpec("Jw", ("u200_smoothed", "u200", "u", "uwnd200"), (25.0, 45.0), (80.0, 110.0), "derived from u200 western jet sector"),
    )

    windows: tuple[Window, ...] = (
        Window("E1", (12, 23)),
        Window("E2", (27, 38)),
        Window("M", (40, 48)),
        Window("E2_pre", (27, 31)),
        Window("E2_post", (34, 38)),
        Window("M_pre", (40, 43)),
        Window("M_post", (45, 48)),
    )

    modes: tuple[str, ...] = ("anomaly", "local_background_removed", "raw")
    primary_modes: tuple[str, ...] = ("anomaly", "local_background_removed")
    object_order: tuple[str, ...] = ("P", "V", "H", "Je", "Jw")
    source_cluster: str = "E2"
    target_cluster: str = "M"
    mapping_types: tuple[str, ...] = (
        "E2_state_to_M_state",
        "E2_state_to_M_transition",
        "E2_transition_to_M_state",
        "E2_transition_to_M_transition",
    )
    local_background_windows: dict[str, tuple[int, int]] | None = None

    n_permutation: int = 10
    n_bootstrap: int = 10
    n_jobs: int = 1
    progress_every: int = 100
    pairwise_scope: str = "all"
    pairwise_bootstrap_policy: str = "all"
    # HOTFIX04: stage 5/6 can be skipped or run in a lighter mode.
    # This is needed because full multioutput ridge permutation is often the real bottleneck.
    multivariate_policy: str = "full"  # full | fast | skip
    multivariate_n_permutation: int | None = None
    object_contribution_policy: str = "full"  # full | fast | skip
    random_seed: int = 20260514
    min_years: int = 8
    pairwise_clear_p: float = 0.10
    pairwise_weak_p: float = 0.20
    pairwise_clear_abs_r: float = 0.30
    pairwise_weak_abs_r: float = 0.20
    mapping_clear_p: float = 0.10
    mapping_weak_p: float = 0.20
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0)
    main_alpha: float = 1.0
    top_pairwise_rows: int = 300

    def __post_init__(self):
        if self.local_background_windows is None:
            self.local_background_windows = {
                "E1": (0, 30),
                "E2": (18, 48),
                "M": (30, 60),
                "E2_transition": (18, 48),
                "M_transition": (30, 60),
            }
        env_perm = os.environ.get("V10_7_K_N_PERM")
        if env_perm:
            self.n_permutation = int(env_perm)
        env_boot = os.environ.get("V10_7_K_N_BOOT")
        if env_boot:
            self.n_bootstrap = int(env_boot)
        env_jobs = os.environ.get("V10_7_K_N_JOBS")
        if env_jobs:
            self.n_jobs = max(1, int(env_jobs))
        env_progress = os.environ.get("V10_7_K_PROGRESS_EVERY")
        if env_progress:
            self.progress_every = max(1, int(env_progress))
        env_scope = os.environ.get("V10_7_K_PAIRWISE_SCOPE")
        if env_scope:
            self.pairwise_scope = str(env_scope)
        env_bootpol = os.environ.get("V10_7_K_PAIRWISE_BOOTSTRAP_POLICY")
        if env_bootpol:
            self.pairwise_bootstrap_policy = str(env_bootpol)
        env_mv = os.environ.get("V10_7_K_MULTIVARIATE_POLICY")
        if env_mv:
            self.multivariate_policy = str(env_mv)
        env_mv_perm = os.environ.get("V10_7_K_MULTIVARIATE_N_PERM")
        if env_mv_perm:
            self.multivariate_n_permutation = int(env_mv_perm)
        env_oc = os.environ.get("V10_7_K_OBJECT_CONTRIBUTION_POLICY")
        if env_oc:
            self.object_contribution_policy = str(env_oc)

    def with_project_root(self, project_root: Path) -> "Settings":
        self.project_root = Path(project_root)
        return self

    def smoothed_fields_path(self) -> Path:
        env = os.environ.get(self.smoothed_env_var)
        if env:
            return Path(env)
        return self.project_root / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"

    def output_root(self) -> Path:
        return self.project_root / "stage_partition" / "V10" / "v10.7" / "outputs" / self.output_tag

    def to_dict(self) -> dict[str, Any]:
        def conv(x: Any) -> Any:
            if isinstance(x, Path):
                return str(x)
            if isinstance(x, tuple):
                return [conv(v) for v in x]
            if isinstance(x, list):
                return [conv(v) for v in x]
            if isinstance(x, dict):
                return {str(k): conv(v) for k, v in x.items()}
            if hasattr(x, "__dataclass_fields__"):
                return conv(asdict(x))
            return x
        return conv(asdict(self))


# ------------------------- IO utilities -------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_output_root(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for sub in ("tables", "figures", "run_meta"):
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
    return np.nanstd(np.asarray(a, dtype=float), axis=axis, keepdims=keepdims)


def zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.full_like(x, np.nan, dtype=float)
    return (x - mu) / sd


def robust_zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    med = np.nanmedian(x)
    q75 = np.nanpercentile(x, 75)
    q25 = np.nanpercentile(x, 25)
    iqr = q75 - q25
    if not np.isfinite(iqr) or abs(iqr) < 1e-12:
        return zscore(x)
    return (x - med) / iqr


def first_key(data: dict[str, np.ndarray], candidates: tuple[str, ...]) -> str | None:
    lower = {k.lower(): k for k in data.keys()}
    for c in candidates:
        if c in data:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_npz(settings: Settings) -> dict[str, Any]:
    path = settings.smoothed_fields_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields file: {path}")
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    lat_key = first_key(data, settings.lat_key_candidates)
    lon_key = first_key(data, settings.lon_key_candidates)
    year_key = first_key(data, settings.year_key_candidates)
    day_key = first_key(data, settings.day_key_candidates)
    if lat_key is None or lon_key is None:
        raise KeyError(f"Could not detect lat/lon keys. Available={sorted(data.keys())}")
    return {
        "path": path,
        "data": data,
        "lat": np.asarray(data[lat_key], dtype=float),
        "lon": np.asarray(data[lon_key], dtype=float),
        "lat_key": lat_key,
        "lon_key": lon_key,
        "year_key": year_key,
        "day_key": day_key,
    }


def normalize_field_dims(field: np.ndarray, data: dict[str, np.ndarray], year_key: str | None, day_key: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    arr = np.asarray(field, dtype=float)
    added_year = False
    if arr.ndim == 3:
        arr = arr[None, ...]
        added_year = True
    if arr.ndim != 4:
        raise ValueError(f"Expected year x day x lat x lon or day x lat x lon, got {field.shape}")
    if year_key is not None and len(np.asarray(data[year_key]).ravel()) == arr.shape[0]:
        years = np.asarray(data[year_key]).ravel()
    else:
        years = np.arange(arr.shape[0], dtype=int)
    if day_key is not None and len(np.asarray(data[day_key]).ravel()) == arr.shape[1]:
        days = np.asarray(data[day_key]).ravel()
    else:
        days = np.arange(arr.shape[1], dtype=int)
    return arr, years, days, added_year


def subset_domain(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]):
    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
    lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError(f"No grid points in lat={lat_range}, lon={lon_range}")
    sub = field[:, :, lat_mask, :][:, :, :, lon_mask]
    sub_lat = lat[lat_mask]
    sub_lon = lon[lon_mask]
    order_lat = np.argsort(sub_lat)
    order_lon = np.argsort(sub_lon)
    sub = sub[:, :, order_lat, :][:, :, :, order_lon]
    return sub, sub_lat[order_lat], sub_lon[order_lon]


def day_mask(days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    lo, hi = win
    return (days >= lo) & (days <= hi)


def window_mean_daily(metric: np.ndarray, days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    m = day_mask(days, win)
    if not np.any(m):
        return np.full(metric.shape[0], np.nan)
    return safe_nanmean(metric[:, m], axis=1)


def window_transition_daily(metric: np.ndarray, days: np.ndarray, pre: tuple[int, int], post: tuple[int, int]) -> np.ndarray:
    return window_mean_daily(metric, days, post) - window_mean_daily(metric, days, pre)


def local_background_state(metric_raw: np.ndarray, days: np.ndarray, cluster_win: tuple[int, int], bg_win: tuple[int, int]) -> np.ndarray:
    cluster = day_mask(days, cluster_win)
    bg = day_mask(days, bg_win) & (~cluster)
    if not np.any(cluster) or not np.any(bg):
        return np.full(metric_raw.shape[0], np.nan)
    return safe_nanmean(metric_raw[:, cluster], axis=1) - safe_nanmean(metric_raw[:, bg], axis=1)


def local_background_transition(metric_raw: np.ndarray, days: np.ndarray, pre: tuple[int, int], post: tuple[int, int], bg_win: tuple[int, int]) -> np.ndarray:
    pre_m = day_mask(days, pre)
    post_m = day_mask(days, post)
    event = pre_m | post_m
    bg = day_mask(days, bg_win) & (~event)
    out = []
    pre_mid = float(np.nanmean(days[pre_m])) if np.any(pre_m) else np.nan
    post_mid = float(np.nanmean(days[post_m])) if np.any(post_m) else np.nan
    for y in range(metric_raw.shape[0]):
        obs = np.nanmean(metric_raw[y, post_m]) - np.nanmean(metric_raw[y, pre_m]) if np.any(pre_m) and np.any(post_m) else np.nan
        if np.sum(bg & np.isfinite(metric_raw[y])) >= 3 and np.isfinite(pre_mid) and np.isfinite(post_mid):
            x = days[bg].astype(float)
            z = metric_raw[y, bg].astype(float)
            ok = np.isfinite(z)
            if np.sum(ok) >= 3:
                slope, intercept = np.polyfit(x[ok], z[ok], 1)
                expected = slope * (post_mid - pre_mid)
                out.append(obs - expected)
            else:
                out.append(np.nan)
        else:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


# ------------------------- metric builders ----------------------------------

def _field_rms(sub: np.ndarray) -> np.ndarray:
    return np.sqrt(safe_nanmean(sub ** 2, axis=(2, 3)))


def _lat_weighted_metrics(sub: np.ndarray, lat: np.ndarray, positive_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    # sub: year x day x lat x lon. Collapse lon first, then compute lat-weighted centroid/spread.
    prof = safe_nanmean(sub, axis=3)  # year x day x lat
    weights = np.maximum(prof, 0.0) if positive_only else np.abs(prof)
    total = np.nansum(weights, axis=2)
    latv = lat.reshape(1, 1, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cent = np.nansum(weights * latv, axis=2) / total
        spread = np.sqrt(np.nansum(weights * (latv - cent[:, :, None]) ** 2, axis=2) / total)
    cent = np.where(total > 1e-12, cent, np.nan)
    spread = np.where(total > 1e-12, spread, np.nan)
    return cent, spread


def _band_share(sub: np.ndarray, lat: np.ndarray, band: tuple[float, float], positive_only: bool = True) -> np.ndarray:
    arr = np.maximum(sub, 0.0) if positive_only else np.abs(sub)
    total = np.nansum(arr, axis=(2, 3))
    lo, hi = sorted(band)
    mask = (lat >= lo) & (lat <= hi)
    band_sum = np.nansum(arr[:, :, mask, :], axis=(2, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = band_sum / total
    return np.where(total > 1e-12, share, np.nan)


def _ns_diff(sub: np.ndarray, lat: np.ndarray, north: tuple[float, float], south: tuple[float, float]) -> np.ndarray:
    nmask = (lat >= min(north)) & (lat <= max(north))
    smask = (lat >= min(south)) & (lat <= max(south))
    nmean = safe_nanmean(sub[:, :, nmask, :], axis=(2, 3))
    smean = safe_nanmean(sub[:, :, smask, :], axis=(2, 3))
    return nmean - smean


def _h_extent_metrics(sub: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Heuristic WPSH-like morphology from each daily field.
    # threshold = 80th percentile within the object domain; west_extent is westernmost active lon.
    n_y, n_d = sub.shape[:2]
    west = np.full((n_y, n_d), np.nan)
    width = np.full((n_y, n_d), np.nan)
    north_edge = np.full((n_y, n_d), np.nan)
    south_edge = np.full((n_y, n_d), np.nan)
    for y in range(n_y):
        for d in range(n_d):
            f = sub[y, d]
            if not np.isfinite(f).any():
                continue
            thr = np.nanpercentile(f, 80)
            mask = f >= thr
            if not np.any(mask):
                continue
            lon_active = lon[np.any(mask, axis=0)]
            lat_active = lat[np.any(mask, axis=1)]
            if lon_active.size:
                west[y, d] = float(np.nanmin(lon_active))
                width[y, d] = float(np.nanmax(lon_active) - np.nanmin(lon_active))
            if lat_active.size:
                south_edge[y, d] = float(np.nanmin(lat_active))
                north_edge[y, d] = float(np.nanmax(lat_active))
    return west, width, north_edge, south_edge


def _jet_metrics(sub: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # sub: year x day x lat x lon. Jet metrics from lon-mean lat profile.
    prof = safe_nanmean(sub, axis=3)  # year x day x lat
    n_y, n_d, _ = prof.shape
    strength = np.full((n_y, n_d), np.nan)
    axis_lat = np.full((n_y, n_d), np.nan)
    width = np.full((n_y, n_d), np.nan)
    for y in range(n_y):
        for d in range(n_d):
            p = prof[y, d]
            ok = np.isfinite(p)
            if np.sum(ok) < 3:
                continue
            pp = p[ok]
            ll = lat[ok]
            imax = int(np.nanargmax(pp))
            axis_lat[y, d] = float(ll[imax])
            q90 = np.nanpercentile(pp, 90)
            top = pp >= q90
            strength[y, d] = float(np.nanmean(pp[top])) if np.any(top) else np.nan
            q80 = np.nanpercentile(pp, 80)
            active_lats = ll[pp >= q80]
            if active_lats.size:
                width[y, d] = float(np.nanmax(active_lats) - np.nanmin(active_lats))
    return strength, axis_lat, width


def build_daily_structure_metrics(settings: Settings, npz: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    data = npz["data"]
    lat = npz["lat"]
    lon = npz["lon"]
    years_ref = None
    days_ref = None
    audit_rows = []
    metric_frames = []

    for spec in settings.object_specs:
        key = first_key(data, spec.source_key_candidates)
        if key is None:
            audit_rows.append({
                "object": spec.object_name, "source_field": None, "loaded": False,
                "lat_range": spec.lat_range, "lon_range": spec.lon_range,
                "metrics": "missing", "notes": f"Missing candidate keys={spec.source_key_candidates}",
            })
            continue
        arr, years, days, added_year = normalize_field_dims(data[key], data, npz["year_key"], npz["day_key"])
        if years_ref is None:
            years_ref = years
            days_ref = days
        elif len(years) != len(years_ref) or len(days) != len(days_ref):
            raise ValueError(f"Dimension mismatch for {spec.object_name}:{key}")
        sub, sub_lat, sub_lon = subset_domain(arr, lat, lon, spec.lat_range, spec.lon_range)
        metrics: dict[str, np.ndarray] = {}
        obj = spec.object_name
        if obj == "P":
            metrics["P_total_strength"] = _field_rms(sub)
            cent, spread = _lat_weighted_metrics(sub, sub_lat, positive_only=True)
            metrics["P_centroid_lat"] = cent
            metrics["P_spread_lat"] = spread
            main = _band_share(sub, sub_lat, (24.0, 35.0), positive_only=True)
            south = _band_share(sub, sub_lat, (18.0, 24.0), positive_only=True)
            metrics["P_main_band_share"] = main
            metrics["P_south_band_share_18_24"] = south
            metrics["P_main_minus_south"] = main - south
        elif obj == "V":
            metrics["V_strength"] = _field_rms(sub)
            metrics["V_NS_diff"] = _ns_diff(sub, sub_lat, north=(25.0, 35.0), south=(15.0, 25.0))
            cent, _ = _lat_weighted_metrics(sub, sub_lat, positive_only=False)
            metrics["V_pos_centroid_lat"] = cent
        elif obj == "H":
            metrics["H_strength"] = _field_rms(sub)
            cent, _ = _lat_weighted_metrics(sub, sub_lat, positive_only=False)
            metrics["H_centroid_lat"] = cent
            west, width, north_edge, south_edge = _h_extent_metrics(sub, sub_lat, sub_lon)
            metrics["H_west_extent_lon"] = west
            metrics["H_zonal_width"] = width
            metrics["H_north_edge_lat"] = north_edge
            metrics["H_south_edge_lat"] = south_edge
        elif obj in ("Je", "Jw"):
            strength, axis_lat, width = _jet_metrics(sub, sub_lat)
            metrics[f"{obj}_strength"] = strength
            metrics[f"{obj}_axis_lat"] = axis_lat
            metrics[f"{obj}_meridional_width"] = width
        else:
            raise ValueError(obj)

        for metric_name, mat in metrics.items():
            df = pd.DataFrame(mat, index=years, columns=days)
            long = df.reset_index().melt(id_vars="index", var_name="day", value_name="value")
            long = long.rename(columns={"index": "year"})
            long["object"] = obj
            long["metric"] = metric_name
            metric_frames.append(long[["year", "day", "object", "metric", "value"]])
        audit_rows.append({
            "object": obj,
            "source_field": key,
            "loaded": True,
            "lat_range": spec.lat_range,
            "lon_range": spec.lon_range,
            "metrics": ",".join(metrics.keys()),
            "notes": spec.source_note + ("; added synthetic year dimension" if added_year else ""),
        })
    if years_ref is None or days_ref is None:
        raise RuntimeError("No object fields could be loaded.")
    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return pd.DataFrame(audit_rows), metrics_df, np.asarray(years_ref), np.asarray(days_ref)


# ------------------------- vector construction ------------------------------

def metric_matrix(metrics_df: pd.DataFrame, object_name: str, metric: str, years: np.ndarray, days: np.ndarray) -> np.ndarray:
    part = metrics_df[(metrics_df["object"] == object_name) & (metrics_df["metric"] == metric)]
    if part.empty:
        return np.full((len(years), len(days)), np.nan)
    piv = part.pivot(index="year", columns="day", values="value")
    piv = piv.reindex(index=years, columns=days)
    return piv.to_numpy(dtype=float)


def anomaly_matrix(mat: np.ndarray) -> np.ndarray:
    clim = safe_nanmean(mat, axis=0, keepdims=True)
    return mat - clim


def build_feature_values(settings: Settings, metrics_df: pd.DataFrame, years: np.ndarray, days: np.ndarray) -> pd.DataFrame:
    rows = []
    windows = {w.name: w.days for w in settings.windows}
    # transitions are named separately to keep state and change explicit.
    transition_defs = {
        "E2_transition": (windows["E2_pre"], windows["E2_post"], settings.local_background_windows["E2_transition"]),
        "M_transition": (windows["M_pre"], windows["M_post"], settings.local_background_windows["M_transition"]),
    }
    state_clusters = {"E1": windows["E1"], "E2": windows["E2"], "M": windows["M"]}

    for obj in settings.object_order:
        metric_names = sorted(metrics_df.loc[metrics_df["object"] == obj, "metric"].unique())
        for metric in metric_names:
            raw = metric_matrix(metrics_df, obj, metric, years, days)
            anom = anomaly_matrix(raw)
            for mode in settings.modes:
                for cluster, win in state_clusters.items():
                    if mode == "raw":
                        vals = window_mean_daily(raw, days, win)
                    elif mode == "anomaly":
                        vals = window_mean_daily(anom, days, win)
                    elif mode == "local_background_removed":
                        bg = settings.local_background_windows[cluster]
                        vals = local_background_state(raw, days, win, bg)
                    else:
                        raise ValueError(mode)
                    for y, val in zip(years, vals):
                        rows.append({
                            "year": y, "mode": mode, "cluster_id": cluster, "vector_kind": "state",
                            "object": obj, "metric": metric, "feature": f"{obj}.{metric}", "value_raw": val,
                        })
                for trans, (pre, post, bg) in transition_defs.items():
                    if mode == "raw":
                        vals = window_transition_daily(raw, days, pre, post)
                    elif mode == "anomaly":
                        vals = window_transition_daily(anom, days, pre, post)
                    elif mode == "local_background_removed":
                        vals = local_background_transition(raw, days, pre, post, bg)
                    else:
                        raise ValueError(mode)
                    for y, val in zip(years, vals):
                        rows.append({
                            "year": y, "mode": mode, "cluster_id": trans, "vector_kind": "transition",
                            "object": obj, "metric": metric, "feature": f"{obj}.{metric}", "value_raw": val,
                        })
    df = pd.DataFrame(rows)
    # z-score every feature within each mode + cluster/vector kind across years.
    zvals = []
    rzvals = []
    for _, sub in df.groupby(["mode", "cluster_id", "vector_kind", "feature"], sort=False):
        zvals.extend(pd.Series(zscore(sub["value_raw"].to_numpy()), index=sub.index).items())
        rzvals.extend(pd.Series(robust_zscore(sub["value_raw"].to_numpy()), index=sub.index).items())
    z_series = pd.Series({i: v for i, v in zvals})
    rz_series = pd.Series({i: v for i, v in rzvals})
    df["value_z"] = z_series.reindex(df.index).to_numpy()
    df["value_robust_z"] = rz_series.reindex(df.index).to_numpy()
    return df


def get_vector_matrix(feature_df: pd.DataFrame, years: np.ndarray, mode: str, cluster_id: str, object_filter: str | None = None) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    part = feature_df[(feature_df["mode"] == mode) & (feature_df["cluster_id"] == cluster_id)]
    if object_filter is not None:
        part = part[part["object"] == object_filter]
    features = sorted(part["feature"].unique())
    objects = []
    metrics = []
    cols = []
    for feat in features:
        fpart = part[part["feature"] == feat]
        piv = fpart.pivot(index="year", columns="feature", values="value_z")
        arr = piv.reindex(index=years, columns=[feat]).to_numpy(dtype=float).ravel()
        cols.append(arr)
        obj = fpart["object"].iloc[0] if not fpart.empty else feat.split(".")[0]
        met = fpart["metric"].iloc[0] if not fpart.empty else feat.split(".")[-1]
        objects.append(obj)
        metrics.append(met)
    if not cols:
        return np.empty((len(years), 0)), [], [], []
    X = np.column_stack(cols)
    return X, features, objects, metrics


def mapping_clusters(mapping_type: str) -> tuple[str, str]:
    if mapping_type == "E2_state_to_M_state":
        return "E2", "M"
    if mapping_type == "E2_state_to_M_transition":
        return "E2", "M_transition"
    if mapping_type == "E2_transition_to_M_state":
        return "E2_transition", "M"
    if mapping_type == "E2_transition_to_M_transition":
        return "E2_transition", "M_transition"
    raise ValueError(mapping_type)


# ------------------------- statistics ---------------------------------------

def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 3:
        return np.nan
    xx = x[ok] - np.mean(x[ok])
    yy = y[ok] - np.mean(y[ok])
    denom = np.sqrt(np.sum(xx ** 2) * np.sum(yy ** 2))
    if denom < 1e-12:
        return np.nan
    return float(np.sum(xx * yy) / denom)


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr_np(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def permutation_p_corr(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int, observed: float) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 4 or not np.isfinite(observed):
        return np.nan
    xx = x[ok]
    yy = y[ok]
    obs = abs(observed)
    count = 1
    total = 1
    for _ in range(n_perm):
        yp = rng.permutation(yy)
        r = spearmanr_np(xx, yp)
        if np.isfinite(r) and abs(r) >= obs:
            count += 1
        total += 1
    return count / total


def bootstrap_ci_corr(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 4:
        return np.nan, np.nan
    xx = x[ok]
    yy = y[ok]
    n = len(xx)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(spearmanr_np(xx[idx], yy[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def support_class_from_pair(r: float, p: float, settings: Settings) -> str:
    if np.isfinite(r) and np.isfinite(p):
        if abs(r) >= settings.pairwise_clear_abs_r and p <= settings.pairwise_clear_p:
            return "clear_structure_mapping_support"
        if abs(r) >= settings.pairwise_weak_abs_r and p <= settings.pairwise_weak_p:
            return "weak_structure_mapping_support"
    return "no_structure_mapping_support"


def _pairwise_support_class_from_task(sr: float, p: float, task: dict[str, Any]) -> str:
    if np.isfinite(sr) and np.isfinite(p):
        if abs(sr) >= float(task["pairwise_clear_abs_r"]) and p <= float(task["pairwise_clear_p"]):
            return "clear_structure_mapping_support"
        if abs(sr) >= float(task["pairwise_weak_abs_r"]) and p <= float(task["pairwise_weak_p"]):
            return "weak_structure_mapping_support"
    return "no_structure_mapping_support"


def _pairwise_corr_task(task: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker for per-pair permutation/bootstrap tests.

    HOTFIX03 adds optional bootstrap policy:
    - all: original behavior, CI for every pair.
    - candidate: CI only for weak/clear pairs after permutation.
    - none: skip pairwise bootstrap CI.
    """
    rng = np.random.default_rng(int(task["seed"]))
    x = np.asarray(task["x"], dtype=float)
    y = np.asarray(task["y"], dtype=float)
    sr = spearmanr_np(x, y)
    pr = pearsonr_np(x, y)
    p = permutation_p_corr(x, y, rng, int(task["n_permutation"]), sr)
    cls = _pairwise_support_class_from_task(sr, p, task)
    policy = str(task.get("pairwise_bootstrap_policy", "all")).lower()
    do_boot = policy == "all" or (policy == "candidate" and cls != "no_structure_mapping_support")
    if do_boot and int(task["n_bootstrap"]) > 0:
        ci_lo, ci_hi = bootstrap_ci_corr(x, y, rng, int(task["n_bootstrap"]))
    else:
        ci_lo, ci_hi = np.nan, np.nan
    out = {k: task[k] for k in (
        "mode", "mapping_type", "source_cluster", "target_cluster",
        "source_object", "source_metric", "source_feature",
        "target_object", "target_metric", "target_feature",
    )}
    out.update({
        "spearman_r": sr,
        "pearson_r": pr,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "permutation_p": p,
        "support_class": cls,
        "pairwise_bootstrap_policy": policy,
    })
    return out


def _include_pairwise_task(scope: str, source_object: str, target_object: str) -> bool:
    scope = str(scope or "all").lower()
    if scope == "all":
        return True
    if scope == "h-source":
        return source_object == "H"
    if scope == "h-related":
        return source_object == "H" or target_object == "H"
    return True


def pairwise_mapping(settings: Settings, feature_df: pd.DataFrame, years: np.ndarray) -> pd.DataFrame:
    tasks: list[dict[str, Any]] = []
    seed_base = settings.random_seed + 110000
    task_id = 0
    for mode_i, mode in enumerate(settings.modes):
        for mt_i, mt in enumerate(settings.mapping_types):
            src_cluster, tgt_cluster = mapping_clusters(mt)
            X, x_feats, x_objs, x_mets = get_vector_matrix(feature_df, years, mode, src_cluster)
            Y, y_feats, y_objs, y_mets = get_vector_matrix(feature_df, years, mode, tgt_cluster)
            for i, sf in enumerate(x_feats):
                for j, tf in enumerate(y_feats):
                    if not _include_pairwise_task(settings.pairwise_scope, x_objs[i], y_objs[j]):
                        continue
                    tasks.append({
                        "seed": seed_base + task_id * 9973,
                        "mode": mode,
                        "mapping_type": mt,
                        "source_cluster": src_cluster,
                        "target_cluster": tgt_cluster,
                        "source_object": x_objs[i],
                        "source_metric": x_mets[i],
                        "source_feature": sf,
                        "target_object": y_objs[j],
                        "target_metric": y_mets[j],
                        "target_feature": tf,
                        "x": X[:, i],
                        "y": Y[:, j],
                        "n_permutation": settings.n_permutation,
                        "n_bootstrap": settings.n_bootstrap,
                        "pairwise_clear_p": settings.pairwise_clear_p,
                        "pairwise_weak_p": settings.pairwise_weak_p,
                        "pairwise_clear_abs_r": settings.pairwise_clear_abs_r,
                        "pairwise_weak_abs_r": settings.pairwise_weak_abs_r,
                        "pairwise_bootstrap_policy": settings.pairwise_bootstrap_policy,
                    })
                    task_id += 1

    if not tasks:
        return pd.DataFrame()

    n_jobs = max(1, int(getattr(settings, "n_jobs", 1)))
    progress_every = max(1, int(getattr(settings, "progress_every", 100)))
    print(
        f"[V10.7_k] pairwise mapping start: tasks={len(tasks)}, "
        f"scope={settings.pairwise_scope}, boot_policy={settings.pairwise_bootstrap_policy}, "
        f"n_perm={settings.n_permutation}, n_boot={settings.n_bootstrap}, n_jobs={n_jobs}",
        flush=True,
    )
    import time
    t0 = time.time()
    rows = []
    if n_jobs <= 1:
        for done, t in enumerate(tasks, start=1):
            rows.append(_pairwise_corr_task(t))
            if done == 1 or done % progress_every == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else float("nan")
                print(f"[V10.7_k] pairwise progress {done}/{len(tasks)} elapsed={elapsed/60:.1f} min rate={rate:.2f}/s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_pairwise_corr_task, t) for t in tasks]
            for done, fut in enumerate(as_completed(futs), start=1):
                rows.append(fut.result())
                if done == 1 or done % progress_every == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else float("nan")
                    print(f"[V10.7_k] pairwise progress {done}/{len(tasks)} elapsed={elapsed/60:.1f} min rate={rate:.2f}/s", flush=True)
    print(f"[V10.7_k] pairwise mapping done: elapsed={(time.time()-t0)/60:.1f} min", flush=True)
    return pd.DataFrame(rows)


# ------------------------- multivariate mapping -----------------------------

def _valid_xy(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    return X[ok], Y[ok], ok


def ridge_fit_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    # Standard intercept handling for already z-scored columns.
    xm = np.nanmean(X_train, axis=0, keepdims=True)
    ym = np.nanmean(Y_train, axis=0, keepdims=True)
    Xc = X_train - xm
    Yc = Y_train - ym
    p = Xc.shape[1]
    beta = np.linalg.pinv(Xc.T @ Xc + alpha * np.eye(p)) @ Xc.T @ Yc
    return (X_test - xm) @ beta + ym


def loo_cv_predict_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, float, float]:
    Xv, Yv, ok = _valid_xy(X, Y)
    n = Xv.shape[0]
    preds_v = np.full_like(Yv, np.nan, dtype=float)
    if n < 4 or Xv.shape[1] < 1 or Yv.shape[1] < 1:
        preds = np.full_like(Y, np.nan, dtype=float)
        return preds, np.nan, np.nan
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        preds_v[i] = ridge_fit_predict(Xv[train], Yv[train], Xv[i:i + 1], alpha)[0]
    sse = np.nansum((Yv - preds_v) ** 2)
    sst = np.nansum((Yv - np.nanmean(Yv, axis=0, keepdims=True)) ** 2)
    cv_r2 = 1.0 - sse / sst if sst > 1e-12 else np.nan
    cv_rmse = float(np.sqrt(sse / np.sum(np.isfinite(Yv - preds_v)))) if np.isfinite(sse) else np.nan
    preds = np.full_like(Y, np.nan, dtype=float)
    preds[ok] = preds_v
    return preds, float(cv_r2), cv_rmse


def permutation_p_mapping(X: np.ndarray, Y: np.ndarray, alpha: float, observed: float, rng: np.random.Generator, n_perm: int) -> float:
    Xv, Yv, ok = _valid_xy(X, Y)
    if Xv.shape[0] < 5 or not np.isfinite(observed):
        return np.nan
    count = 1
    total = 1
    for _ in range(n_perm):
        yp = Yv[rng.permutation(Yv.shape[0])]
        _, r2, _ = loo_cv_predict_ridge(Xv, yp, alpha)
        if np.isfinite(r2) and r2 >= observed:
            count += 1
        total += 1
    return count / total


def mapping_status(cv_r2: float, p: float, settings: Settings) -> str:
    if np.isfinite(cv_r2) and np.isfinite(p):
        if cv_r2 > 0 and p <= settings.mapping_clear_p:
            return "structure_mapping_detected"
        if p <= settings.mapping_weak_p:
            return "weak_structure_mapping"
    return "no_structure_mapping"


def _empty_skill_df(reason: str = "not_evaluated") -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "mode", "mapping_type", "method", "alpha", "n_source_features",
        "n_target_features", "cv_r2", "cv_rmse", "permutation_p",
        "mapping_status", "skip_reason",
    ])


def _empty_contrib_df(reason: str = "not_evaluated") -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "mode", "mapping_type", "method", "alpha", "source_removed",
        "full_skill_cv_r2", "skill_without_source_cv_r2",
        "skill_without_source_cv_rmse", "skill_drop", "null_drop_p90",
        "permutation_p", "contribution_class", "skip_reason",
    ])


def multivariate_mapping(settings: Settings, feature_df: pd.DataFrame, years: np.ndarray) -> tuple[pd.DataFrame, dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray, float]]]:
    if str(settings.multivariate_policy).lower() == "skip":
        print("[V10.7_k]   multivariate_policy=skip: skip ridge mapping and continue with pairwise/H-specific outputs", flush=True)
        return _empty_skill_df("skipped_by_multivariate_policy"), {}

    rng = np.random.default_rng(settings.random_seed + 22)
    rows = []
    cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray, float]] = {}

    if str(settings.multivariate_policy).lower() == "fast":
        modes = tuple(settings.primary_modes)
        mapping_types = tuple(settings.mapping_types)
        alphas = (settings.main_alpha,)
        n_perm = int(settings.multivariate_n_permutation or min(200, settings.n_permutation))
        print(f"[V10.7_k]   multivariate_policy=fast: modes={modes}, alpha={alphas}, n_perm={n_perm}", flush=True)
    else:
        modes = tuple(settings.modes)
        mapping_types = tuple(settings.mapping_types)
        alphas = tuple(settings.ridge_alphas)
        n_perm = int(settings.multivariate_n_permutation or settings.n_permutation)
        print(f"[V10.7_k]   multivariate_policy=full: modes={modes}, alphas={alphas}, n_perm={n_perm}", flush=True)

    total = len(modes) * len(mapping_types) * len(alphas)
    done = 0
    for mode in modes:
        for mt in mapping_types:
            src_cluster, tgt_cluster = mapping_clusters(mt)
            X, x_feats, _, _ = get_vector_matrix(feature_df, years, mode, src_cluster)
            Y, y_feats, _, _ = get_vector_matrix(feature_df, years, mode, tgt_cluster)
            for alpha in alphas:
                done += 1
                print(f"[V10.7_k]   ridge mapping {done}/{total}: mode={mode}, mapping_type={mt}, alpha={alpha}", flush=True)
                preds, cv_r2, cv_rmse = loo_cv_predict_ridge(X, Y, alpha)
                p = permutation_p_mapping(X, Y, alpha, cv_r2, rng, n_perm)
                rows.append({
                    "mode": mode,
                    "mapping_type": mt,
                    "method": "ridge_multioutput",
                    "alpha": alpha,
                    "n_source_features": len(x_feats),
                    "n_target_features": len(y_feats),
                    "cv_r2": cv_r2,
                    "cv_rmse": cv_rmse,
                    "permutation_p": p,
                    "mapping_status": mapping_status(cv_r2, p, settings),
                    "skip_reason": "",
                })
                cache[(mode, mt, alpha)] = (X, Y, cv_r2)
    return pd.DataFrame(rows), cache

def remove_one_source(settings: Settings, feature_df: pd.DataFrame, years: np.ndarray, full_skill_df: pd.DataFrame) -> pd.DataFrame:
    if str(settings.object_contribution_policy).lower() == "skip" or str(settings.multivariate_policy).lower() == "skip":
        print("[V10.7_k]   object_contribution skipped", flush=True)
        return _empty_contrib_df("skipped_by_policy")
    if full_skill_df.empty:
        return _empty_contrib_df("no_multivariate_skill_rows")

    rng = np.random.default_rng(settings.random_seed + 33)
    rows = []
    if str(settings.object_contribution_policy).lower() == "fast":
        modes = tuple(settings.primary_modes)
        mapping_types = tuple(settings.mapping_types)
        n_perm = int(settings.multivariate_n_permutation or min(200, settings.n_permutation))
        print(f"[V10.7_k]   object_contribution_policy=fast: modes={modes}, n_perm={n_perm}", flush=True)
    else:
        modes = tuple(settings.modes)
        mapping_types = tuple(settings.mapping_types)
        n_perm = int(settings.multivariate_n_permutation or settings.n_permutation)
        print(f"[V10.7_k]   object_contribution_policy=full: modes={modes}, n_perm={n_perm}", flush=True)

    total = len(modes) * len(mapping_types) * len(settings.object_order)
    done = 0
    for mode in modes:
        for mt in mapping_types:
            src_cluster, tgt_cluster = mapping_clusters(mt)
            X_full, x_feats, x_objs, _ = get_vector_matrix(feature_df, years, mode, src_cluster)
            Y, _, _, _ = get_vector_matrix(feature_df, years, mode, tgt_cluster)
            full_row = full_skill_df[(full_skill_df["mode"] == mode) & (full_skill_df["mapping_type"] == mt) & (np.isclose(full_skill_df["alpha"], settings.main_alpha))]
            if full_row.empty:
                continue
            full_skill = float(full_row["cv_r2"].iloc[0])
            for obj in settings.object_order:
                done += 1
                print(f"[V10.7_k]   remove-one-source {done}/{total}: mode={mode}, mapping_type={mt}, remove={obj}", flush=True)
                keep = np.array([o != obj for o in x_objs], dtype=bool)
                if not np.any(keep) or np.all(keep):
                    continue
                X_sub = X_full[:, keep]
                _, sub_skill, sub_rmse = loo_cv_predict_ridge(X_sub, Y, settings.main_alpha)
                drop = full_skill - sub_skill if np.isfinite(full_skill) and np.isfinite(sub_skill) else np.nan
                null_drops = []
                Xv, Yv, ok = _valid_xy(X_full, Y)
                if Xv.shape[0] >= 5 and np.isfinite(drop):
                    keep_v = keep
                    for _ in range(n_perm):
                        yp = Yv[rng.permutation(Yv.shape[0])]
                        _, fnull, _ = loo_cv_predict_ridge(Xv, yp, settings.main_alpha)
                        _, snull, _ = loo_cv_predict_ridge(Xv[:, keep_v], yp, settings.main_alpha)
                        if np.isfinite(fnull) and np.isfinite(snull):
                            null_drops.append(fnull - snull)
                null_p90 = float(np.nanpercentile(null_drops, 90)) if null_drops else np.nan
                perm_p = (1 + sum(v >= drop for v in null_drops if np.isfinite(v))) / (1 + len(null_drops)) if null_drops and np.isfinite(drop) else np.nan
                if np.isfinite(drop) and drop > 0 and np.isfinite(null_p90) and drop > null_p90 and np.isfinite(perm_p) and perm_p <= settings.mapping_clear_p:
                    cls = "key_structure_mapping_dimension"
                elif np.isfinite(drop) and drop > 0:
                    cls = "secondary_structure_mapping_dimension"
                elif np.isfinite(drop) and abs(drop) <= 0.01:
                    cls = "nonessential_structure_dimension"
                elif np.isfinite(drop) and drop < 0:
                    cls = "negative_or_unstable_structure_dimension"
                else:
                    cls = "ambiguous"
                rows.append({
                    "mode": mode,
                    "mapping_type": mt,
                    "method": "ridge_multioutput",
                    "alpha": settings.main_alpha,
                    "source_removed": obj,
                    "full_skill_cv_r2": full_skill,
                    "skill_without_source_cv_r2": sub_skill,
                    "skill_without_source_cv_rmse": sub_rmse,
                    "skill_drop": drop,
                    "null_drop_p90": null_p90,
                    "permutation_p": perm_p,
                    "contribution_class": cls,
                    "skip_reason": "",
                })
    return pd.DataFrame(rows)


# ------------------------- route decision -----------------------------------

def build_route_decision(settings: Settings, pair_df: pd.DataFrame, skill_df: pd.DataFrame, contrib_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if skill_df.empty or str(settings.multivariate_policy).lower() == "skip":
        rows.append({
            "decision_item": "E2_to_M_structure_mapping",
            "status": "not_evaluated_multivariate_skipped",
            "evidence": "Multivariate ridge mapping was skipped by runtime policy; inspect pairwise/H-specific target mapping only.",
            "route_implication": "Do not draw any overall E2→M structure-mapping conclusion from this run.",
        })
    else:
        primary = skill_df[skill_df["mode"].isin(settings.primary_modes)]
        detected = primary[primary["mapping_status"].isin(["structure_mapping_detected", "weak_structure_mapping"])]
        if detected.empty:
            rows.append({
                "decision_item": "E2_to_M_structure_mapping",
                "status": "no_structure_mapping_detected",
                "evidence": "No primary-mode state/transition mapping passed ridge/shuffled-year route threshold.",
                "route_implication": "Do not claim W33 connects to W45 through current structure-vector mapping.",
            })
        else:
            best = detected.sort_values(["mapping_status", "cv_r2"], ascending=[True, False]).head(3)
            rows.append({
                "decision_item": "E2_to_M_structure_mapping",
                "status": "candidate_structure_mapping_detected",
                "evidence": "; ".join(f"{r.mode}:{r.mapping_type}:cv_r2={r.cv_r2:.3f},p={r.permutation_p:.3f},{r.mapping_status}" for r in best.itertuples()),
                "route_implication": "Inspect object contribution and H-specific target mapping before physical interpretation.",
            })

    if contrib_df.empty or str(settings.object_contribution_policy).lower() == "skip" or str(settings.multivariate_policy).lower() == "skip":
        rows.append({
            "decision_item": "H_E2_structure_contribution",
            "status": "not_evaluated_object_contribution_skipped",
            "evidence": "Remove-one-source contribution was skipped by runtime policy.",
            "route_implication": "Use H-specific pairwise mapping only; do not claim H is key/non-key from object-contribution tables.",
        })
    else:
        h_rows = contrib_df[(contrib_df["mode"].isin(settings.primary_modes)) & (contrib_df["source_removed"] == "H")]
        key_h = h_rows[h_rows["contribution_class"] == "key_structure_mapping_dimension"]
        if not key_h.empty:
            rows.append({
                "decision_item": "H_E2_structure_contribution",
                "status": "H_key_structure_mapping_dimension",
                "evidence": "; ".join(f"{r.mode}:{r.mapping_type}:drop={r.skill_drop:.3f},p={r.permutation_p:.3f}" for r in key_h.itertuples()),
                "route_implication": "Retain H as a possible W33 structural preconfiguration dimension; require spatial/physical audit.",
            })
        elif not h_rows.empty and (h_rows["skill_drop"] > 0).any():
            pos = h_rows[h_rows["skill_drop"] > 0].sort_values("skill_drop", ascending=False).head(3)
            rows.append({
                "decision_item": "H_E2_structure_contribution",
                "status": "H_secondary_or_unstable_structure_dimension",
                "evidence": "; ".join(f"{r.mode}:{r.mapping_type}:drop={r.skill_drop:.3f},{r.contribution_class}" for r in pos.itertuples()),
                "route_implication": "H structural role remains candidate-level only; do not upgrade without target-specific support.",
            })
        else:
            rows.append({
                "decision_item": "H_E2_structure_contribution",
                "status": "H_not_supported_in_current_structure_mapping",
                "evidence": "Removing H did not reduce primary-mode mapping skill.",
                "route_implication": "Current structure metrics do not support H as a mapping dimension; this is not a causal exclusion.",
            })

    h_pair = pair_df[(pair_df["mode"].isin(settings.primary_modes)) & (pair_df["source_object"] == "H")]
    h_support = h_pair[h_pair["support_class"].isin(["clear_structure_mapping_support", "weak_structure_mapping_support"])]
    if not h_support.empty:
        top = h_support.sort_values(["support_class", "permutation_p"], ascending=[True, True]).head(5)
        rows.append({
            "decision_item": "H_E2_to_M_target_specific_mapping",
            "status": "H_target_specific_candidate_lines",
            "evidence": "; ".join(f"{r.source_metric}->{r.target_object}.{r.target_metric}:r={r.spearman_r:.3f},p={r.permutation_p:.3f},{r.support_class}" for r in top.itertuples()),
            "route_implication": "Only these narrow H-structure target mappings should be considered for follow-up.",
        })
    else:
        rows.append({
            "decision_item": "H_E2_to_M_target_specific_mapping",
            "status": "no_H_target_specific_mapping_support",
            "evidence": "No primary-mode H source structural metric passed pairwise mapping support threshold.",
            "route_implication": "Do not pursue H-specific structure lines from this version unless other evidence exists.",
        })
    return pd.DataFrame(rows)


# ------------------------- plotting / summary -------------------------------

def plot_pairwise_heatmap(pair_df: pd.DataFrame, output: Path, mode: str = "anomaly", mapping_type: str = "E2_state_to_M_state") -> None:
    part = pair_df[(pair_df["mode"] == mode) & (pair_df["mapping_type"] == mapping_type)]
    if part.empty:
        return
    # Collapse metric pairs to object pair max abs spearman for readability.
    mat = part.groupby(["source_object", "target_object"])["spearman_r"].apply(lambda s: s.iloc[np.nanargmax(np.abs(s.to_numpy()))] if np.isfinite(s.to_numpy()).any() else np.nan).unstack()
    rows = [o for o in ("P", "V", "H", "Je", "Jw") if o in mat.index]
    cols = [o for o in ("P", "V", "H", "Je", "Jw") if o in mat.columns]
    arr = mat.reindex(index=rows, columns=cols).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(arr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(f"{mode} {mapping_type}: object-pair max |r|")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mapping_skill(skill_df: pd.DataFrame, output: Path) -> None:
    part = skill_df[np.isclose(skill_df["alpha"], 1.0)]
    if part.empty:
        return
    labels = [f"{r.mode}\n{r.mapping_type.replace('E2_', '').replace('_to_', '→').replace('_', ' ')}" for r in part.itertuples()]
    vals = part["cv_r2"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(vals)), 4))
    ax.bar(range(len(vals)), vals)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(vals)), labels, rotation=90, fontsize=7)
    ax.set_ylabel("LOO CV R²")
    ax.set_title("Structure E2→M mapping skill, ridge alpha=1")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_contribution(contrib_df: pd.DataFrame, output: Path, mode: str = "anomaly", mapping_type: str = "E2_state_to_M_state") -> None:
    part = contrib_df[(contrib_df["mode"] == mode) & (contrib_df["mapping_type"] == mapping_type)]
    if part.empty:
        return
    objs = part["source_removed"].tolist()
    vals = part["skill_drop"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(vals)), vals)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(vals)), objs)
    ax.set_ylabel("CV R² drop when removed")
    ax.set_title(f"Object contribution: {mode} {mapping_type}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_h_specific(h_df: pd.DataFrame, output: Path) -> None:
    part = h_df[(h_df["mode"] == "anomaly") & (h_df["mapping_type"] == "E2_state_to_M_state")]
    if part.empty:
        return
    # Keep top 25 by absolute r.
    part = part.reindex(part["spearman_r"].abs().sort_values(ascending=False).index).head(25)
    labels = [f"{r.source_metric}\n→ {r.target_object}.{r.target_metric}" for r in part.itertuples()]
    vals = part["spearman_r"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(vals)), 4))
    ax.bar(range(len(vals)), vals)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(vals)), labels, rotation=90, fontsize=7)
    ax.set_ylabel("Spearman r")
    ax.set_title("H_E2 structural metrics to M target metrics, anomaly state→state")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summary(settings: Settings, out: Path, route_df: pd.DataFrame, skill_df: pd.DataFrame) -> Path:
    lines = []
    lines.append("# V10.7_k W33→W45 structure-transition mapping audit")
    lines.append("")
    lines.append("## Method boundary")
    lines.append("- This version tests structure/state/transition mapping, not activity-amplitude mapping.")
    lines.append("- It does not control away P/V/Je/Jw as covariates.")
    lines.append("- It allows H_E2 structural metrics to map to non-H M metrics.")
    lines.append("- It does not infer causality.")
    lines.append("")
    lines.append("## Main route decisions")
    for r in route_df.itertuples():
        lines.append(f"- **{r.decision_item}**: `{r.status}` — {r.evidence}  ")
        lines.append(f"  Implication: {r.route_implication}")
    lines.append("")
    lines.append("## Ridge mapping skill snapshot")
    snap = skill_df[np.isclose(skill_df["alpha"], settings.main_alpha)].copy()
    if not snap.empty:
        for r in snap.itertuples():
            lines.append(f"- {r.mode} / {r.mapping_type}: cv_r2={r.cv_r2:.3f}, p={r.permutation_p:.3f}, status={r.mapping_status}")
    lines.append("")
    lines.append("## Forbidden interpretations")
    lines.append("- Do not interpret support as causality.")
    lines.append("- Do not treat a negative result here as proof that H has no W45 role.")
    lines.append("- Do not treat raw-mode results as primary evidence when anomaly/local-background disagree.")
    path = out / "summary_w45_structure_transition_mapping_v10_7_k.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------- pipeline -----------------------------------------

def run_w45_structure_transition_mapping_v10_7_k(
    project_root: str | Path | None = None,
    n_permutation: int | None = None,
    n_bootstrap: int | None = None,
    n_jobs: int | None = None,
    progress_every: int | None = None,
    pairwise_scope: str | None = None,
    pairwise_bootstrap_policy: str | None = None,
    multivariate_policy: str | None = None,
    multivariate_n_permutation: int | None = None,
    object_contribution_policy: str | None = None,
) -> dict[str, Any]:
    settings = Settings()
    # HOTFIX01: allow explicit runtime overrides from the entry script.
    # Environment variables are still supported by Settings.__post_init__, but
    # command-line arguments are more reliable across CMD/PowerShell/IDE launches.
    if n_permutation is not None:
        settings.n_permutation = int(n_permutation)
    if n_bootstrap is not None:
        settings.n_bootstrap = int(n_bootstrap)
    if n_jobs is not None:
        settings.n_jobs = max(1, int(n_jobs))
    if progress_every is not None:
        settings.progress_every = max(1, int(progress_every))
    if pairwise_scope is not None:
        settings.pairwise_scope = str(pairwise_scope)
    if pairwise_bootstrap_policy is not None:
        settings.pairwise_bootstrap_policy = str(pairwise_bootstrap_policy)
    if multivariate_policy is not None:
        settings.multivariate_policy = str(multivariate_policy)
    if multivariate_n_permutation is not None:
        settings.multivariate_n_permutation = int(multivariate_n_permutation)
    if object_contribution_policy is not None:
        settings.object_contribution_policy = str(object_contribution_policy)
    if project_root is not None:
        settings.with_project_root(Path(project_root))
    out = settings.output_root()
    clean_output_root(out)
    print(f"[V10.7_k] output_root = {out}", flush=True)
    print("[V10.7_k] stage 1/9 load smoothed fields", flush=True)
    npz = load_npz(settings)

    print("[V10.7_k] stage 2/9 build daily structure metrics", flush=True)
    input_audit, daily_metrics, years, days = build_daily_structure_metrics(settings, npz)
    if len(years) < settings.min_years:
        warnings.warn(f"Only {len(years)} years detected; mapping may be underpowered.")
    print("[V10.7_k] stage 3/9 build feature vectors", flush=True)
    feature_df = build_feature_values(settings, daily_metrics, years, days)

    print("[V10.7_k] stage 4/9 pairwise structure mapping", flush=True)
    pair_df = pairwise_mapping(settings, feature_df, years)
    print("[V10.7_k] stage 5/9 multivariate ridge mapping", flush=True)
    skill_df, _ = multivariate_mapping(settings, feature_df, years)
    print("[V10.7_k] stage 6/9 remove-one-source contribution", flush=True)
    contrib_df = remove_one_source(settings, feature_df, years, skill_df)
    h_specific_df = pair_df[pair_df["source_object"] == "H"].copy()
    route_df = build_route_decision(settings, pair_df, skill_df, contrib_df)

    print("[V10.7_k] stage 7/9 write tables", flush=True)
    # Write tables.
    write_dataframe(input_audit, out / "tables" / "w45_structure_metric_input_audit_v10_7_k.csv")
    write_dataframe(feature_df, out / "tables" / "w45_structure_vectors_by_year_v10_7_k.csv")
    write_dataframe(pair_df, out / "tables" / "w45_structure_pairwise_mapping_matrix_v10_7_k.csv")
    write_dataframe(skill_df, out / "tables" / "w45_e2_to_m_structure_mapping_skill_v10_7_k.csv")
    write_dataframe(contrib_df, out / "tables" / "w45_e2_structure_object_contribution_v10_7_k.csv")
    write_dataframe(h_specific_df, out / "tables" / "w45_h_e2_structure_to_m_target_mapping_v10_7_k.csv")
    write_dataframe(route_df, out / "tables" / "w45_structure_transition_route_decision_v10_7_k.csv")

    print("[V10.7_k] stage 8/9 write figures", flush=True)
    # Figures.
    plot_pairwise_heatmap(pair_df, out / "figures" / "w45_structure_pairwise_mapping_heatmap_v10_7_k.png", mode="anomaly", mapping_type="E2_state_to_M_state")
    plot_mapping_skill(skill_df, out / "figures" / "w45_structure_mapping_skill_vs_null_v10_7_k.png")
    plot_contribution(contrib_df, out / "figures" / "w45_structure_object_contribution_v10_7_k.png", mode="anomaly", mapping_type="E2_state_to_M_state")
    plot_h_specific(h_specific_df, out / "figures" / "w45_h_e2_structure_target_mapping_v10_7_k.png")

    print("[V10.7_k] stage 9/9 write summary and run_meta", flush=True)
    summary_path = write_summary(settings, out, route_df, skill_df)
    meta = {
        "version": settings.version,
        "task": "W33-to-W45 structure-transition mapping audit",
        "started_finished_utc": now_utc(),
        "project_root": str(settings.project_root),
        "smoothed_fields_path": str(npz["path"]),
        "output_root": str(out),
        "summary_path": str(summary_path),
        "n_years": int(len(years)),
        "n_days": int(len(days)),
        "settings": settings.to_dict(),
        "method_boundary": [
            "not activity-amplitude mapping",
            "not control-regression",
            "not causal inference",
            "tests structure state/transition mapping",
            "allows H_E2 to map to non-H M structure",
            "HOTFIX02: pairwise permutation/bootstrap pairs can run in parallel via n_jobs",
            "HOTFIX03: stage progress, pairwise progress, optional pairwise scope and bootstrap policy",
            "HOTFIX04: optional skip/fast multivariate ridge mapping and object contribution policies",
        ],
        "outputs": {
            "input_audit": "tables/w45_structure_metric_input_audit_v10_7_k.csv",
            "vectors": "tables/w45_structure_vectors_by_year_v10_7_k.csv",
            "pairwise": "tables/w45_structure_pairwise_mapping_matrix_v10_7_k.csv",
            "mapping_skill": "tables/w45_e2_to_m_structure_mapping_skill_v10_7_k.csv",
            "object_contribution": "tables/w45_e2_structure_object_contribution_v10_7_k.csv",
            "h_specific": "tables/w45_h_e2_structure_to_m_target_mapping_v10_7_k.csv",
            "route_decision": "tables/w45_structure_transition_route_decision_v10_7_k.csv",
        },
    }
    write_json(meta, out / "run_meta" / "run_meta_v10_7_k.json")
    return meta


if __name__ == "__main__":
    run_w45_structure_transition_mapping_v10_7_k()
