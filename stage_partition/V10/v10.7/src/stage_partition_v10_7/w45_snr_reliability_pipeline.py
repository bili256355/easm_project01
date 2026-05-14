from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
# V10.7_j: W45 E2-M yearwise signal-to-noise / reliability audit
# =============================================================================
# Method boundary:
# - This is NOT a new H-role / W45 mechanism test.
# - This is NOT a causal, regression-control, or transition-mapping result.
# - It audits whether the yearwise object-window scalar indicators used by
#   V10.7_i have enough signal-to-noise and window stability to support mapping.
# - A negative V10.7_i mapping result should be treated as non-decisive if this
#   audit finds low object-window SNR, unstable year ranks, or high sensitivity
#   to leave-days/window shifts.
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
    reducer: str = "spatial_rms"  # spatial_rms | jet_q90_strength
    source_note: str = ""


@dataclass
class Settings:
    project_root: Path = Path(r"D:\easm_project01")
    version: str = "v10.7_j"
    output_tag: str = "w45_snr_reliability_v10_7_j"
    smoothed_env_var: str = "V10_7_SMOOTHED_FIELDS"

    lat_key_candidates: tuple[str, ...] = ("lat", "latitude", "lats")
    lon_key_candidates: tuple[str, ...] = ("lon", "longitude", "lons")
    year_key_candidates: tuple[str, ...] = ("year", "years")
    day_key_candidates: tuple[str, ...] = ("day", "days", "day_index")

    object_specs: tuple[ObjectSpec, ...] = (
        ObjectSpec("P", ("precip_smoothed", "precip", "P", "pr", "rain", "tp"), (15.0, 35.0), (110.0, 140.0), "spatial_rms", "precip object proxy domain"),
        ObjectSpec("V", ("v850_smoothed", "v850", "V", "v", "vwind850"), (15.0, 35.0), (110.0, 140.0), "spatial_rms", "v850 object proxy domain"),
        ObjectSpec("H", ("z500_smoothed", "z500", "H", "hgt500", "geopotential_500", "zg500"), (15.0, 35.0), (110.0, 140.0), "spatial_rms", "H object domain"),
        ObjectSpec("Je", ("u200_smoothed", "u200", "u", "uwnd200"), (25.0, 45.0), (120.0, 150.0), "jet_q90_strength", "derived from u200 eastern sector: 120-150E, 25-45N"),
        ObjectSpec("Jw", ("u200_smoothed", "u200", "u", "uwnd200"), (25.0, 45.0), (80.0, 110.0), "jet_q90_strength", "derived from u200 western sector: 80-110E, 25-45N"),
    )

    windows: tuple[Window, ...] = (
        Window("E1", (12, 23)),
        Window("E2", (27, 38)),
        Window("M", (40, 48)),
    )

    modes: tuple[str, ...] = ("anomaly", "local_background_removed", "raw")
    primary_modes: tuple[str, ...] = ("anomaly", "local_background_removed")
    object_order: tuple[str, ...] = ("P", "V", "H", "Je", "Jw")

    local_background_windows: dict[str, tuple[int, int]] | None = None
    shift_range: tuple[int, ...] = (-2, -1, 0, 1, 2)
    n_bootstrap: int = 500
    random_seed: int = 20260514
    min_years: int = 8
    snr_usable_threshold: float = 1.0
    snr_marginal_threshold: float = 0.5
    rank_corr_usable_threshold: float = 0.70
    rank_corr_marginal_threshold: float = 0.50
    sign_stability_threshold: float = 0.70

    def __post_init__(self):
        if self.local_background_windows is None:
            self.local_background_windows = {
                "E1": (0, 30),
                "E2": (18, 48),
                "M": (30, 60),
            }
        env_boot = os.environ.get("V10_7_J_N_BOOT")
        if env_boot:
            self.n_bootstrap = int(env_boot)

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


def rankdata_average(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    vals = arr[mask]
    if vals.size == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.size, dtype=float)
    i = 0
    while i < vals.size:
        j = i + 1
        while j < vals.size and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    out[mask] = ranks
    return out


def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m] - np.nanmean(a[m])
    b = b[m] - np.nanmean(b[m])
    den = float(np.sqrt(np.nansum(a * a) * np.nansum(b * b)))
    if den < 1e-12:
        return np.nan
    return float(np.nansum(a * b) / den)


def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr_pearson(rankdata_average(x), rankdata_average(y))


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
    if sub_lat.size > 1 and sub_lat[0] > sub_lat[-1]:
        order = np.argsort(sub_lat)
        sub = sub[:, :, order, :]
        sub_lat = sub_lat[order]
    return sub, sub_lat, sub_lon


def day_indices(days: np.ndarray, day_range: tuple[int, int]) -> np.ndarray:
    lo, hi = day_range
    return np.where((days >= lo) & (days <= hi))[0]


def exclude_indices(base: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    if base.size == 0:
        return base
    return np.array([i for i in base if i not in set(excluded.tolist())], dtype=int)


# ------------------------- object daily strength -----------------------------

@dataclass
class ObjectData:
    spec: ObjectSpec
    source_key: str
    field: np.ndarray  # year x day x lat x lon, subset to object domain
    years: np.ndarray
    days: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    added_year_dimension: bool


def build_object_data(settings: Settings, bundle: dict[str, Any]) -> tuple[dict[str, ObjectData], pd.DataFrame]:
    data = bundle["data"]
    lat = bundle["lat"]
    lon = bundle["lon"]
    audit_rows = []
    objects: dict[str, ObjectData] = {}
    for spec in settings.object_specs:
        key = first_key(data, spec.source_key_candidates)
        if key is None:
            audit_rows.append({
                "object": spec.object_name,
                "loaded": False,
                "source_field": "",
                "derived_from": "",
                "lat_range": spec.lat_range,
                "lon_range": spec.lon_range,
                "strength_method": spec.reducer,
                "notes": f"missing source; candidates={spec.source_key_candidates}",
            })
            continue
        arr, years, days, added_year = normalize_field_dims(data[key], data, bundle["year_key"], bundle["day_key"])
        sub, sub_lat, sub_lon = subset_domain(arr, lat, lon, spec.lat_range, spec.lon_range)
        objects[spec.object_name] = ObjectData(spec, key, sub, years, days, sub_lat, sub_lon, added_year)
        audit_rows.append({
            "object": spec.object_name,
            "loaded": True,
            "source_field": key,
            "derived_from": key if spec.reducer == "spatial_rms" else f"{key} sector profile",
            "lat_range": f"{spec.lat_range[0]}-{spec.lat_range[1]}",
            "lon_range": f"{spec.lon_range[0]}-{spec.lon_range[1]}",
            "strength_method": spec.reducer,
            "n_years": len(years),
            "n_days": len(days),
            "n_lat": len(sub_lat),
            "n_lon": len(sub_lon),
            "added_year_dimension": added_year,
            "notes": spec.source_note,
        })
    return objects, pd.DataFrame(audit_rows)


def reduce_daily_strength(field_ydll: np.ndarray, reducer: str) -> np.ndarray:
    arr = np.asarray(field_ydll, dtype=float)
    # arr: year x selected_day x lat x lon
    if reducer == "spatial_rms":
        return np.sqrt(safe_nanmean(arr * arr, axis=(2, 3)))
    if reducer == "jet_q90_strength":
        # lon-mean lat profile, then mean top 10% lat values per year/day
        prof = safe_nanmean(arr, axis=3)  # year x day x lat
        out = np.full(prof.shape[:2], np.nan, dtype=float)
        for iy in range(prof.shape[0]):
            for iday in range(prof.shape[1]):
                p = prof[iy, iday, :]
                if np.isfinite(p).sum() < 2:
                    continue
                q90 = np.nanpercentile(p, 90.0)
                vals = p[p >= q90]
                out[iy, iday] = np.nanmean(vals) if vals.size else np.nan
        return out
    raise ValueError(f"Unknown reducer={reducer}")


def window_daily_strength(obj: ObjectData, mode: str, cluster: Window, settings: Settings, shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
    days = obj.days
    shifted = (cluster.days[0] + shift, cluster.days[1] + shift)
    idx = day_indices(days, shifted)
    if idx.size == 0:
        return np.full((len(obj.years), 0), np.nan), np.array([], dtype=int)
    raw = obj.field
    if mode == "raw":
        field = raw[:, idx, :, :]
    elif mode == "anomaly":
        clim = safe_nanmean(raw, axis=0, keepdims=True)
        field = raw[:, idx, :, :] - clim[:, idx, :, :]
    elif mode == "local_background_removed":
        bg_range = settings.local_background_windows.get(cluster.name, cluster.days)
        bg_shifted = (bg_range[0] + shift, bg_range[1] + shift)
        bg_idx = exclude_indices(day_indices(days, bg_shifted), idx)
        if bg_idx.size == 0:
            bg_idx = idx
        bg = safe_nanmean(raw[:, bg_idx, :, :], axis=1, keepdims=True)
        field = raw[:, idx, :, :] - bg
    else:
        raise ValueError(f"Unknown mode={mode}")
    return reduce_daily_strength(field, obj.spec.reducer), days[idx]


def build_window_strength_tables(settings: Settings, objects: dict[str, ObjectData]) -> tuple[pd.DataFrame, dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rows = []
    cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for mode in settings.modes:
        for cluster in settings.windows:
            for obj_name in settings.object_order:
                if obj_name not in objects:
                    continue
                obj = objects[obj_name]
                daily, day_vals = window_daily_strength(obj, mode, cluster, settings, shift=0)
                cache[(mode, cluster.name, obj_name)] = (obj.years, day_vals, daily)
                for iy, year in enumerate(obj.years):
                    vals = daily[iy]
                    if vals.size == 0 or not np.isfinite(vals).any():
                        mean = std = cv = peak_val = peak_to_mean = np.nan
                        peak_day = np.nan
                    else:
                        mean = float(np.nanmean(vals))
                        std = float(np.nanstd(vals))
                        cv = float(std / (abs(mean) + 1e-12))
                        imax = int(np.nanargmax(np.abs(vals)))
                        peak_val = float(vals[imax])
                        peak_day = int(day_vals[imax]) if day_vals.size else np.nan
                        peak_to_mean = float(abs(peak_val) / (abs(mean) + 1e-12))
                    rows.append({
                        "year": year,
                        "mode": mode,
                        "cluster_id": cluster.name,
                        "object": obj_name,
                        "window_day_min": cluster.days[0],
                        "window_day_max": cluster.days[1],
                        "n_window_days": int(vals.size),
                        "window_mean": mean,
                        "window_std": std,
                        "window_cv": cv,
                        "peak_day": peak_day,
                        "peak_value": peak_val,
                        "peak_to_mean_ratio": peak_to_mean,
                    })
    return pd.DataFrame(rows), cache


# ------------------------- SNR and reliability -------------------------------

def classify_object_usability(snr: float, rank_corr: float, settings: Settings) -> str:
    if not np.isfinite(snr) or not np.isfinite(rank_corr):
        return "unavailable"
    if snr >= settings.snr_usable_threshold and rank_corr >= settings.rank_corr_usable_threshold:
        return "usable_for_yearwise_mapping"
    if snr >= settings.snr_marginal_threshold and rank_corr >= settings.rank_corr_marginal_threshold:
        return "marginal_for_yearwise_mapping"
    return "low_snr_or_unstable"


def bootstrap_window_means(daily: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    # daily: year x days
    ny, nd = daily.shape
    if nd == 0:
        return np.full((n_boot, ny), np.nan)
    out = np.full((n_boot, ny), np.nan, dtype=float)
    for b in range(n_boot):
        sample_idx = rng.integers(0, nd, size=nd)
        out[b, :] = safe_nanmean(daily[:, sample_idx], axis=1)
    return out


def compute_snr_reliability(settings: Settings, cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(settings.random_seed)
    snr_rows = []
    boot_rows = []
    for key, (years, day_vals, daily) in cache.items():
        mode, cluster, obj = key
        if daily.size == 0:
            continue
        means = safe_nanmean(daily, axis=1)
        stds = safe_nanstd(daily, axis=1)
        between_var = float(np.nanvar(means, ddof=1)) if np.isfinite(means).sum() > 1 else np.nan
        within_noise = float(np.nanmean(stds * stds))
        snr = float(between_var / (within_noise + 1e-12)) if np.isfinite(between_var) else np.nan
        cvs = stds / (np.abs(means) + 1e-12)
        peak_days = []
        for iy in range(daily.shape[0]):
            vals = daily[iy]
            if vals.size and np.isfinite(vals).any():
                peak_days.append(day_vals[int(np.nanargmax(np.abs(vals)))])
        peak_iqr = float(np.nanpercentile(peak_days, 75) - np.nanpercentile(peak_days, 25)) if peak_days else np.nan

        boot = bootstrap_window_means(daily, settings.n_bootstrap, rng)
        base_rank = means
        rank_corrs = np.array([corr_spearman(base_rank, boot[b, :]) for b in range(boot.shape[0])], dtype=float)
        rank_corr_median = float(np.nanmedian(rank_corrs)) if np.isfinite(rank_corrs).any() else np.nan
        rank_corr_p10 = float(np.nanpercentile(rank_corrs[np.isfinite(rank_corrs)], 10)) if np.isfinite(rank_corrs).any() else np.nan
        frac_ge_08 = float(np.nanmean(rank_corrs >= 0.8)) if np.isfinite(rank_corrs).any() else np.nan
        ci_low = np.nanpercentile(boot, 5, axis=0)
        ci_high = np.nanpercentile(boot, 95, axis=0)
        median_ci_width = float(np.nanmedian(ci_high - ci_low))
        signal_sd = float(np.nanstd(means))
        ci_to_signal_ratio = float(median_ci_width / (signal_sd + 1e-12))
        usability = classify_object_usability(snr, rank_corr_median, settings)

        snr_rows.append({
            "mode": mode,
            "cluster_id": cluster,
            "object": obj,
            "n_years": int(len(years)),
            "n_window_days": int(len(day_vals)),
            "between_year_variance": between_var,
            "within_window_noise": within_noise,
            "snr_between_over_within": snr,
            "median_window_cv": float(np.nanmedian(cvs)),
            "mean_window_cv": float(np.nanmean(cvs)),
            "peak_day_iqr": peak_iqr,
            "bootstrap_rank_corr_median": rank_corr_median,
            "bootstrap_rank_corr_p10": rank_corr_p10,
            "bootstrap_fraction_rank_corr_ge_0_8": frac_ge_08,
            "median_bootstrap_ci_width": median_ci_width,
            "ci_width_to_between_year_sd": ci_to_signal_ratio,
            "object_window_usability": usability,
        })

        for iy, year in enumerate(years):
            boot_rows.append({
                "year": year,
                "mode": mode,
                "cluster_id": cluster,
                "object": obj,
                "base_window_mean": means[iy],
                "bootstrap_mean_p05": ci_low[iy],
                "bootstrap_mean_p50": float(np.nanpercentile(boot[:, iy], 50)),
                "bootstrap_mean_p95": ci_high[iy],
                "bootstrap_ci_width": ci_high[iy] - ci_low[iy],
            })
    return pd.DataFrame(snr_rows), pd.DataFrame(boot_rows)


# ------------------------- mapping sensitivity -------------------------------

def window_mean_vector(settings: Settings, objects: dict[str, ObjectData], mode: str, cluster: Window, shift: int = 0) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for obj_name in settings.object_order:
        if obj_name not in objects:
            continue
        daily, _ = window_daily_strength(objects[obj_name], mode, cluster, settings, shift=shift)
        out[obj_name] = safe_nanmean(daily, axis=1)
    return out


def compute_pairwise_mapping(v_source: dict[str, np.ndarray], v_target: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    vals: dict[tuple[str, str], float] = {}
    for s, x in v_source.items():
        for t, y in v_target.items():
            vals[(s, t)] = corr_spearman(zscore(x), zscore(y))
    return vals


def compute_leave_day_sensitivity(settings: Settings, objects: dict[str, ObjectData]) -> pd.DataFrame:
    rows = []
    e2 = next(w for w in settings.windows if w.name == "E2")
    mwin = next(w for w in settings.windows if w.name == "M")
    for mode in settings.primary_modes:
        # Base vectors
        e2_base = window_mean_vector(settings, objects, mode, e2, shift=0)
        m_base = window_mean_vector(settings, objects, mode, mwin, shift=0)
        base_corr = compute_pairwise_mapping(e2_base, m_base)
        # Actual day values for possible leave-one-day runs
        # Source-side leave days
        src_leave_corrs: dict[tuple[str, str], list[float]] = {k: [] for k in base_corr}
        tgt_leave_corrs: dict[tuple[str, str], list[float]] = {k: [] for k in base_corr}
        # Build day lists from any object
        ref_obj = objects[settings.object_order[0]]
        e2_days = ref_obj.days[day_indices(ref_obj.days, e2.days)]
        m_days = ref_obj.days[day_indices(ref_obj.days, mwin.days)]
        for leave_day in e2_days:
            src_vec = {}
            for obj_name in settings.object_order:
                if obj_name not in objects:
                    continue
                obj = objects[obj_name]
                daily, dvals = window_daily_strength(obj, mode, e2, settings, shift=0)
                keep = dvals != leave_day
                src_vec[obj_name] = safe_nanmean(daily[:, keep], axis=1) if keep.any() else safe_nanmean(daily, axis=1)
            cc = compute_pairwise_mapping(src_vec, m_base)
            for k, v in cc.items():
                src_leave_corrs[k].append(v)
        for leave_day in m_days:
            tgt_vec = {}
            for obj_name in settings.object_order:
                if obj_name not in objects:
                    continue
                obj = objects[obj_name]
                daily, dvals = window_daily_strength(obj, mode, mwin, settings, shift=0)
                keep = dvals != leave_day
                tgt_vec[obj_name] = safe_nanmean(daily[:, keep], axis=1) if keep.any() else safe_nanmean(daily, axis=1)
            cc = compute_pairwise_mapping(e2_base, tgt_vec)
            for k, v in cc.items():
                tgt_leave_corrs[k].append(v)
        for (s, t), base in base_corr.items():
            src_arr = np.asarray(src_leave_corrs[(s, t)], dtype=float)
            tgt_arr = np.asarray(tgt_leave_corrs[(s, t)], dtype=float)
            all_arr = np.concatenate([src_arr, tgt_arr])
            def sign_stability(arr):
                if not np.isfinite(base) or not np.isfinite(arr).any() or abs(base) < 1e-12:
                    return np.nan
                return float(np.nanmean(np.sign(arr) == np.sign(base)))
            rows.append({
                "mode": mode,
                "source_object_E2": s,
                "target_object_M": t,
                "base_spearman": base,
                "source_leave_day_median_abs_delta": float(np.nanmedian(np.abs(src_arr - base))) if src_arr.size else np.nan,
                "source_leave_day_max_abs_delta": float(np.nanmax(np.abs(src_arr - base))) if src_arr.size else np.nan,
                "source_leave_day_sign_stability": sign_stability(src_arr),
                "target_leave_day_median_abs_delta": float(np.nanmedian(np.abs(tgt_arr - base))) if tgt_arr.size else np.nan,
                "target_leave_day_max_abs_delta": float(np.nanmax(np.abs(tgt_arr - base))) if tgt_arr.size else np.nan,
                "target_leave_day_sign_stability": sign_stability(tgt_arr),
                "all_leave_day_sign_stability": sign_stability(all_arr),
                "leave_day_reliability_class": "stable" if sign_stability(all_arr) >= settings.sign_stability_threshold else "unstable_or_sensitive",
            })
    return pd.DataFrame(rows)


def compute_window_shift_sensitivity(settings: Settings, objects: dict[str, ObjectData]) -> pd.DataFrame:
    rows = []
    e2 = next(w for w in settings.windows if w.name == "E2")
    mwin = next(w for w in settings.windows if w.name == "M")
    for mode in settings.primary_modes:
        base_e2 = window_mean_vector(settings, objects, mode, e2, shift=0)
        base_m = window_mean_vector(settings, objects, mode, mwin, shift=0)
        base = compute_pairwise_mapping(base_e2, base_m)
        shift_corrs: dict[tuple[str, str], list[float]] = {k: [] for k in base}
        for se in settings.shift_range:
            for sm in settings.shift_range:
                e2_vec = window_mean_vector(settings, objects, mode, e2, shift=se)
                m_vec = window_mean_vector(settings, objects, mode, mwin, shift=sm)
                cc = compute_pairwise_mapping(e2_vec, m_vec)
                for k, v in cc.items():
                    shift_corrs[k].append(v)
                for (s, t), v in cc.items():
                    rows.append({
                        "mode": mode,
                        "source_object_E2": s,
                        "target_object_M": t,
                        "e2_shift": se,
                        "m_shift": sm,
                        "spearman": v,
                        "base_spearman": base.get((s, t), np.nan),
                        "abs_delta_from_base": abs(v - base.get((s, t), np.nan)) if np.isfinite(v) and np.isfinite(base.get((s, t), np.nan)) else np.nan,
                    })
        # add summary rows separately below as dataframe attribute not possible; create second table in wrapper
    return pd.DataFrame(rows)


def summarize_window_shift(shift_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = []
    if shift_df.empty:
        return pd.DataFrame(rows)
    for (mode, s, t), g in shift_df.groupby(["mode", "source_object_E2", "target_object_M"]):
        base = float(g["base_spearman"].iloc[0])
        vals = g["spearman"].to_numpy(dtype=float)
        sign_stab = float(np.nanmean(np.sign(vals) == np.sign(base))) if np.isfinite(base) and abs(base) > 1e-12 else np.nan
        rows.append({
            "mode": mode,
            "source_object_E2": s,
            "target_object_M": t,
            "base_spearman": base,
            "shift_spearman_median": float(np.nanmedian(vals)),
            "shift_spearman_min": float(np.nanmin(vals)),
            "shift_spearman_max": float(np.nanmax(vals)),
            "shift_abs_delta_median": float(np.nanmedian(g["abs_delta_from_base"])),
            "shift_abs_delta_max": float(np.nanmax(g["abs_delta_from_base"])),
            "shift_sign_stability": sign_stab,
            "window_shift_reliability_class": "stable" if sign_stab >= settings.sign_stability_threshold else "unstable_or_shift_sensitive",
        })
    return pd.DataFrame(rows)


# ------------------------- decisions -----------------------------------------

def build_route_decision(settings: Settings, snr_df: pd.DataFrame, leave_df: pd.DataFrame, shift_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = snr_df[snr_df["mode"].isin(settings.primary_modes)].copy()
    if primary.empty:
        rows.append({
            "decision_item": "yearwise_indicator_reliability",
            "status": "unavailable",
            "evidence": "No primary-mode SNR rows available.",
            "route_implication": "V10.7_i cannot be evaluated.",
        })
        return pd.DataFrame(rows)

    em = primary[primary["cluster_id"].isin(["E2", "M"])]
    n_total = len(em)
    n_usable = int((em["object_window_usability"] == "usable_for_yearwise_mapping").sum())
    n_marginal = int((em["object_window_usability"] == "marginal_for_yearwise_mapping").sum())
    frac_usable = n_usable / max(n_total, 1)
    frac_usable_or_marg = (n_usable + n_marginal) / max(n_total, 1)

    if frac_usable >= 0.60:
        status = "sufficient_indicator_snr"
        implication = "V10.7_i scalar mapping non-detection is more interpretable, though still limited to scalar indicators."
    elif frac_usable_or_marg >= 0.60:
        status = "marginal_indicator_snr"
        implication = "V10.7_i non-detection should be treated cautiously; some indicators are usable but many are marginal."
    else:
        status = "low_indicator_snr"
        implication = "V10.7_i negative mapping result is not decisive; low SNR can explain non-detection."
    rows.append({
        "decision_item": "E2_M_yearwise_indicator_snr",
        "status": status,
        "evidence": f"usable={n_usable}/{n_total}; usable_or_marginal={n_usable+n_marginal}/{n_total}",
        "route_implication": implication,
    })

    if leave_df.empty:
        rows.append({
            "decision_item": "leave_days_out_mapping_sensitivity",
            "status": "unavailable",
            "evidence": "No leave-day sensitivity rows.",
            "route_implication": "Cannot evaluate fixed-window day sensitivity.",
        })
    else:
        stable_frac = float((leave_df["leave_day_reliability_class"] == "stable").mean())
        if stable_frac >= 0.60:
            st = "leave_day_stable"
            imp = "Mapping signs are not dominated by individual days for most pairs."
        else:
            st = "leave_day_sensitive"
            imp = "V10.7_i pairwise mapping/non-mapping can be affected by one-day choices; treat as low-power or window-sensitive."
        rows.append({
            "decision_item": "leave_days_out_mapping_sensitivity",
            "status": st,
            "evidence": f"stable_pair_fraction={stable_frac:.3f}",
            "route_implication": imp,
        })

    if shift_summary_df.empty:
        rows.append({
            "decision_item": "window_shift_mapping_sensitivity",
            "status": "unavailable",
            "evidence": "No shift sensitivity rows.",
            "route_implication": "Cannot evaluate fixed-window shift sensitivity.",
        })
    else:
        stable_frac = float((shift_summary_df["window_shift_reliability_class"] == "stable").mean())
        if stable_frac >= 0.60:
            st = "window_shift_stable"
            imp = "Mapping signs are relatively robust to +/-2 day shifts for most pairs."
        else:
            st = "window_shift_sensitive"
            imp = "Fixed E2/M windows may be too rigid; V10.7_i non-detection may reflect timing misalignment."
        rows.append({
            "decision_item": "window_shift_mapping_sensitivity",
            "status": st,
            "evidence": f"stable_pair_fraction={stable_frac:.3f}",
            "route_implication": imp,
        })

    # Object-level usability summary
    for obj in settings.object_order:
        sub = em[em["object"] == obj]
        if sub.empty:
            continue
        usable = int((sub["object_window_usability"] == "usable_for_yearwise_mapping").sum())
        marginal = int((sub["object_window_usability"] == "marginal_for_yearwise_mapping").sum())
        if usable >= 2:
            st = "usable_object_for_yearwise_mapping"
            imp = f"{obj} scalar indicators can support yearwise mapping checks."
        elif usable + marginal >= 2:
            st = "marginal_object_for_yearwise_mapping"
            imp = f"{obj} scalar indicators are marginal; negative object-specific mapping is not strong."
        else:
            st = "low_snr_object"
            imp = f"{obj} scalar indicators are not reliable enough for strong negative conclusions."
        rows.append({
            "decision_item": f"object_usability_{obj}",
            "status": st,
            "evidence": f"usable={usable}/{len(sub)}, marginal={marginal}/{len(sub)} among primary E2/M modes",
            "route_implication": imp,
        })
    return pd.DataFrame(rows)


# ------------------------- plotting ------------------------------------------

def plot_snr_summary(snr_df: pd.DataFrame, out: Path) -> None:
    df = snr_df[(snr_df["mode"].isin(["anomaly", "local_background_removed"])) & (snr_df["cluster_id"].isin(["E2", "M"]))].copy()
    if df.empty:
        return
    df["label"] = df["mode"].str.replace("local_background_removed", "local") + ":" + df["cluster_id"] + ":" + df["object"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(df)), df["snr_between_over_within"].to_numpy())
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.axhline(0.5, linestyle=":", linewidth=1)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["label"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("between-year variance / within-window noise")
    ax.set_title("V10.7_j object-window SNR summary")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_rank_reliability(snr_df: pd.DataFrame, out: Path) -> None:
    df = snr_df[(snr_df["mode"].isin(["anomaly", "local_background_removed"])) & (snr_df["cluster_id"].isin(["E2", "M"]))].copy()
    if df.empty:
        return
    df["label"] = df["mode"].str.replace("local_background_removed", "local") + ":" + df["cluster_id"] + ":" + df["object"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(df)), df["bootstrap_rank_corr_median"].to_numpy())
    ax.axhline(0.7, linestyle="--", linewidth=1)
    ax.axhline(0.5, linestyle=":", linewidth=1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["label"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("median bootstrap rank correlation")
    ax.set_title("V10.7_j yearwise ranking reliability")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_window_shift_summary(shift_summary: pd.DataFrame, out: Path) -> None:
    if shift_summary.empty:
        return
    df = shift_summary[shift_summary["mode"] == "anomaly"].copy()
    if df.empty:
        df = shift_summary.copy()
    objects = sorted(set(df["source_object_E2"]) | set(df["target_object_M"]))
    mat = np.full((len(objects), len(objects)), np.nan)
    obj_i = {o: i for i, o in enumerate(objects)}
    for _, r in df.iterrows():
        mat[obj_i[r["source_object_E2"]], obj_i[r["target_object_M"]]] = r["shift_sign_stability"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(objects)))
    ax.set_yticks(np.arange(len(objects)))
    ax.set_xticklabels([f"M_{o}" for o in objects], rotation=45, ha="right")
    ax.set_yticklabels([f"E2_{o}" for o in objects])
    ax.set_title("Window-shift sign stability (primary shown)")
    fig.colorbar(im, ax=ax, label="sign stability")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# ------------------------- summary -------------------------------------------

def write_summary(settings: Settings, output_root: Path, route_df: pd.DataFrame, snr_df: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# V10.7_j W45 E2–M yearwise SNR / reliability audit")
    lines.append("")
    lines.append("## Method boundary")
    lines.append("")
    lines.append("- This is not a new W33→W45 mechanism test.")
    lines.append("- It audits whether V10.7_i's yearwise scalar indicators are reliable enough for mapping tests.")
    lines.append("- If reliability is low, V10.7_i non-detection should be treated as non-decisive rather than as a structural negative result.")
    lines.append("")
    lines.append("## Key route decisions")
    lines.append("")
    if route_df.empty:
        lines.append("No route decisions were generated.")
    else:
        for _, r in route_df.iterrows():
            lines.append(f"- **{r['decision_item']}**: `{r['status']}` — {r['evidence']}. {r['route_implication']}")
    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("")
    lines.append("- Low SNR does not prove absence of E2→M organization; it means the current yearwise scalar mapping has low power.")
    lines.append("- Good SNR still only supports scalar-indicator mapping, not causality or shape/position mechanisms.")
    lines.append("- Window-shift or leave-day sensitivity indicates fixed-window timing may be too rigid.")
    lines.append("")
    path = output_root / "summary_w45_snr_reliability_v10_7_j.md"
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------- pipeline ------------------------------------------

def run_w45_snr_reliability_v10_7_j(project_root: str | Path | None = None) -> dict[str, Any]:
    settings = Settings()
    if project_root is not None:
        settings.with_project_root(Path(project_root))
    output_root = settings.output_root()
    clean_output_root(output_root)
    meta: dict[str, Any] = {
        "version": settings.version,
        "task": "W45 E2-M yearwise signal-to-noise and reliability audit",
        "started_at": now_utc(),
        "output_root": str(output_root),
        "settings": settings.to_dict(),
    }
    try:
        bundle = load_npz(settings)
        meta["smoothed_fields_path"] = str(bundle["path"])
        objects, input_audit = build_object_data(settings, bundle)
        write_dataframe(input_audit, output_root / "tables" / "w45_snr_input_audit_v10_7_j.csv")
        if len(objects) < 2:
            raise RuntimeError(f"Too few loaded objects: {list(objects)}")
        # Use first object for year count; audit mixed dims if needed.
        first_obj = next(iter(objects.values()))
        if len(first_obj.years) < settings.min_years:
            meta["warning"] = f"n_years={len(first_obj.years)} < min_years={settings.min_years}; yearwise reliability is weak by design."

        daily_metrics, cache = build_window_strength_tables(settings, objects)
        write_dataframe(daily_metrics, output_root / "tables" / "w45_object_window_daily_reliability_by_year_v10_7_j.csv")

        snr_df, boot_df = compute_snr_reliability(settings, cache)
        write_dataframe(snr_df, output_root / "tables" / "w45_object_window_snr_summary_v10_7_j.csv")
        write_dataframe(boot_df, output_root / "tables" / "w45_yearwise_strength_bootstrap_reliability_v10_7_j.csv")

        leave_df = compute_leave_day_sensitivity(settings, objects)
        write_dataframe(leave_df, output_root / "tables" / "w45_mapping_leave_days_out_sensitivity_v10_7_j.csv")

        shift_df = compute_window_shift_sensitivity(settings, objects)
        shift_summary = summarize_window_shift(shift_df, settings)
        write_dataframe(shift_df, output_root / "tables" / "w45_mapping_window_shift_grid_v10_7_j.csv")
        write_dataframe(shift_summary, output_root / "tables" / "w45_mapping_window_shift_sensitivity_v10_7_j.csv")

        route_df = build_route_decision(settings, snr_df, leave_df, shift_summary)
        write_dataframe(route_df, output_root / "tables" / "w45_snr_route_decision_v10_7_j.csv")

        plot_snr_summary(snr_df, output_root / "figures" / "w45_object_window_snr_summary_v10_7_j.png")
        plot_rank_reliability(snr_df, output_root / "figures" / "w45_yearwise_rank_reliability_v10_7_j.png")
        plot_window_shift_summary(shift_summary, output_root / "figures" / "w45_window_shift_sign_stability_v10_7_j.png")

        write_summary(settings, output_root, route_df, snr_df)
        meta.update({
            "status": "completed",
            "finished_at": now_utc(),
            "n_loaded_objects": len(objects),
            "loaded_objects": sorted(objects.keys()),
            "key_outputs": {
                "snr_summary": str(output_root / "tables" / "w45_object_window_snr_summary_v10_7_j.csv"),
                "bootstrap_reliability": str(output_root / "tables" / "w45_yearwise_strength_bootstrap_reliability_v10_7_j.csv"),
                "leave_days_out": str(output_root / "tables" / "w45_mapping_leave_days_out_sensitivity_v10_7_j.csv"),
                "window_shift": str(output_root / "tables" / "w45_mapping_window_shift_sensitivity_v10_7_j.csv"),
                "route_decision": str(output_root / "tables" / "w45_snr_route_decision_v10_7_j.csv"),
            },
        })
    except Exception as exc:
        meta.update({"status": "failed", "finished_at": now_utc(), "error": repr(exc)})
        write_json(meta, output_root / "run_meta" / "run_meta_v10_7_j.json")
        raise
    write_json(meta, output_root / "run_meta" / "run_meta_v10_7_j.json")
    return meta


if __name__ == "__main__":
    run_w45_snr_reliability_v10_7_j()
