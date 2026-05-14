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
# V10.7_h: W45 E1-E2-M multi-object configuration trajectory audit
# =============================================================================
# Method boundary:
# - This is NOT a regression-control experiment.
# - It does NOT control away P/V/Je/Jw as covariates.
# - It treats W45 as a multi-object configuration made by P/V/H/Je/Jw.
# - It tests within-year configuration coupling against shuffled-year null.
# - It is heuristic / route-decision evidence, not causal inference.
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
    version: str = "v10.7_h"
    output_tag: str = "w45_configuration_trajectory_v10_7_h"
    smoothed_env_var: str = "V10_7_SMOOTHED_FIELDS"

    lat_key_candidates: tuple[str, ...] = ("lat", "latitude", "lats")
    lon_key_candidates: tuple[str, ...] = ("lon", "longitude", "lons")
    year_key_candidates: tuple[str, ...] = ("year", "years")
    day_key_candidates: tuple[str, ...] = ("day", "days", "day_index")

    # V10 object-event domains. Je/Jw are derived from u200, not independent fields.
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
    primary_mode: str = "anomaly"
    object_order: tuple[str, ...] = ("P", "V", "H", "Je", "Jw")
    pair_order: tuple[str, ...] = ("E1_E2", "E2_M", "E1_M")
    similarity_metrics: tuple[str, ...] = ("cosine", "pearson", "neg_euclidean")
    primary_similarity_metric: str = "cosine"

    # Local background days are intentionally broad and exclude the active cluster.
    # This is a diagnostic residualization, not a physical anomaly definition.
    local_background_windows: dict[str, tuple[int, int]] = None

    n_permutation: int = 500
    random_seed: int = 20260514
    min_years_for_coupling: int = 8
    coupled_p_threshold: float = 0.10
    strongly_coupled_p_threshold: float = 0.05

    def __post_init__(self):
        if self.local_background_windows is None:
            self.local_background_windows = {
                "E1": (0, 30),
                "E2": (18, 48),
                "M": (30, 60),
            }

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


# ------------------------- basic IO utilities -------------------------------

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
    q75, q25 = np.nanpercentile(x, [75, 25])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr < 1e-12:
        return np.full_like(x, np.nan, dtype=float)
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
    lat = np.asarray(data[lat_key], dtype=float)
    lon = np.asarray(data[lon_key], dtype=float)
    return {"path": path, "data": data, "lat": lat, "lon": lon, "lat_key": lat_key, "lon_key": lon_key, "year_key": year_key, "day_key": day_key}


def normalize_field_dims(field: np.ndarray, data: dict[str, np.ndarray], year_key: str | None, day_key: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    arr = np.asarray(field, dtype=float)
    added_year = False
    if arr.ndim == 3:
        arr = arr[None, ...]
        added_year = True
    if arr.ndim != 4:
        raise ValueError(f"Expected field dims year x day x lat x lon or day x lat x lon, got {field.shape}")
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
    # Make output consistently low-to-high for later interpretation.
    lat_order = np.argsort(sub_lat)
    lon_order = np.argsort(sub_lon)
    return sub[:, :, lat_order, :][:, :, :, lon_order], sub_lat[lat_order], sub_lon[lon_order]


def load_object_fields(settings: Settings, npz: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    data = npz["data"]
    fields: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for spec in settings.object_specs:
        key = first_key(data, spec.source_key_candidates)
        row = {
            "object": spec.object_name,
            "status": "missing",
            "source_key": key,
            "lat_range": f"{spec.lat_range[0]}-{spec.lat_range[1]}",
            "lon_range": f"{spec.lon_range[0]}-{spec.lon_range[1]}",
            "reducer": spec.reducer,
            "source_note": spec.source_note,
            "note": "",
        }
        if key is None:
            row["note"] = "No matching source field key. This object will be skipped."
            rows.append(row)
            continue
        try:
            field, years, days, added_year = normalize_field_dims(data[key], data, npz["year_key"], npz["day_key"])
            if field.shape[2] != len(npz["lat"]) or field.shape[3] != len(npz["lon"]):
                raise ValueError(f"field shape {field.shape} mismatches lat/lon {npz['lat'].shape}/{npz['lon'].shape}")
            sub, sub_lat, sub_lon = subset_domain(field, npz["lat"], npz["lon"], spec.lat_range, spec.lon_range)
            fields[spec.object_name] = {
                "field": sub,
                "lat": sub_lat,
                "lon": sub_lon,
                "years": years,
                "days": days,
                "source_key": key,
                "added_year_axis": added_year,
                "spec": spec,
            }
            row.update({"status": "loaded", "n_years": int(sub.shape[0]), "n_days": int(sub.shape[1]), "n_lat": int(sub.shape[2]), "n_lon": int(sub.shape[3]), "added_year_axis": added_year})
        except Exception as exc:
            row["status"] = "failed"
            row["note"] = repr(exc)
        rows.append(row)
    return fields, pd.DataFrame(rows)


# ------------------------- strength construction ----------------------------

def day_indices(days: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    lo, hi = window
    idx = np.where((days >= lo) & (days <= hi))[0]
    if len(idx) == 0:
        raise ValueError(f"No days in window {window}; available {days[:5]}...{days[-5:]}")
    return idx


def daily_anomaly(field: np.ndarray) -> np.ndarray:
    # year x day x lat x lon -> subtract day climatology over years.
    if field.shape[0] <= 1:
        return np.full_like(field, np.nan, dtype=float)
    clim = safe_nanmean(field, axis=0, keepdims=True)
    return field - clim


def cluster_field_by_mode(field: np.ndarray, days: np.ndarray, cluster: Window, mode: str, settings: Settings) -> np.ndarray:
    cidx = day_indices(days, cluster.days)
    if mode == "raw":
        return safe_nanmean(field[:, cidx, :, :], axis=1)
    if mode == "anomaly":
        anom = daily_anomaly(field)
        return safe_nanmean(anom[:, cidx, :, :], axis=1)
    if mode == "local_background_removed":
        b_lo, b_hi = settings.local_background_windows.get(cluster.name, (max(0, cluster.days[0] - 15), cluster.days[1] + 15))
        bidx_all = day_indices(days, (b_lo, b_hi))
        cset = set(map(int, cidx.tolist()))
        bidx = np.asarray([i for i in bidx_all if int(i) not in cset], dtype=int)
        if len(bidx) == 0:
            # Fallback: use all background, but mark via meta not per-row.
            bidx = bidx_all
        cmean = safe_nanmean(field[:, cidx, :, :], axis=1)
        bmean = safe_nanmean(field[:, bidx, :, :], axis=1)
        return cmean - bmean
    raise ValueError(f"Unknown mode: {mode}")


def reduce_cluster_strength(cluster_field: np.ndarray, spec: ObjectSpec) -> np.ndarray:
    # cluster_field: year x lat x lon
    if spec.reducer == "spatial_rms":
        return np.sqrt(safe_nanmean(cluster_field ** 2, axis=(1, 2)))
    if spec.reducer == "jet_q90_strength":
        # sector field -> lon-mean lat profile, then q90 threshold-excess top mean.
        profile = safe_nanmean(cluster_field, axis=2)  # year x lat
        out = []
        for row in profile:
            valid = row[np.isfinite(row)]
            if valid.size == 0:
                out.append(np.nan)
                continue
            q90 = np.nanpercentile(valid, 90)
            top = valid[valid >= q90]
            out.append(float(np.nanmean(top)) if top.size else np.nan)
        return np.asarray(out, dtype=float)
    raise ValueError(f"Unknown reducer: {spec.reducer}")


def build_object_configuration(settings: Settings, fields: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metric_rows = []
    for obj in settings.object_order:
        if obj not in fields:
            continue
        info = fields[obj]
        spec: ObjectSpec = info["spec"]
        field = info["field"]
        days = info["days"]
        years = info["years"]
        for mode in settings.modes:
            for win in settings.windows:
                try:
                    cf = cluster_field_by_mode(field, days, win, mode, settings)
                    strength = reduce_cluster_strength(cf, spec)
                except Exception as exc:
                    for y in years:
                        rows.append({"year": y, "mode": mode, "cluster_id": win.name, "object": obj, "window_day_min": win.days[0], "window_day_max": win.days[1], "strength_raw": np.nan, "note": repr(exc)})
                    continue
                for y, val in zip(years, strength):
                    rows.append({
                        "year": int(y) if np.issubdtype(np.asarray(years).dtype, np.integer) else y,
                        "mode": mode,
                        "cluster_id": win.name,
                        "object": obj,
                        "window_day_min": win.days[0],
                        "window_day_max": win.days[1],
                        "strength_raw": float(val) if np.isfinite(val) else np.nan,
                        "note": "",
                    })
                metric_rows.append({"object": obj, "mode": mode, "cluster_id": win.name, "reducer": spec.reducer, "source_key": info["source_key"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(metric_rows)

    # Standardize by object across all years and clusters within each mode.
    df["strength_z"] = np.nan
    df["strength_robust_z"] = np.nan
    for (mode, obj), idx in df.groupby(["mode", "object"]).groups.items():
        vals = df.loc[idx, "strength_raw"].to_numpy(dtype=float)
        df.loc[idx, "strength_z"] = zscore(vals)
        df.loc[idx, "strength_robust_z"] = robust_zscore(vals)

    # Rank objects within each year x mode x cluster by z strength.
    df["rank_within_cluster"] = np.nan
    for _, idx in df.groupby(["year", "mode", "cluster_id"]).groups.items():
        vals = df.loc[idx, "strength_z"]
        # descending rank: 1 strongest; keep NaN as NaN.
        df.loc[idx, "rank_within_cluster"] = vals.rank(ascending=False, method="min")
    return df, pd.DataFrame(metric_rows)


# ------------------------- configuration vectors ----------------------------

def build_config_vectors(config_df: pd.DataFrame, settings: Settings, z_col: str = "strength_z") -> tuple[dict[str, dict[Any, dict[str, np.ndarray]]], pd.DataFrame]:
    vectors: dict[str, dict[Any, dict[str, np.ndarray]]] = {}
    rows = []
    for mode in sorted(config_df["mode"].dropna().unique()):
        sub_mode = config_df[config_df["mode"] == mode]
        vectors[mode] = {}
        for year in sorted(sub_mode["year"].dropna().unique()):
            vectors[mode][year] = {}
            sub_y = sub_mode[sub_mode["year"] == year]
            for cluster in [w.name for w in settings.windows]:
                vals = []
                missing = []
                for obj in settings.object_order:
                    cell = sub_y[(sub_y["cluster_id"] == cluster) & (sub_y["object"] == obj)]
                    if cell.empty:
                        vals.append(np.nan)
                        missing.append(obj)
                    else:
                        vals.append(float(cell[z_col].iloc[0]))
                arr = np.asarray(vals, dtype=float)
                vectors[mode][year][cluster] = arr
                rows.append({
                    "mode": mode,
                    "year": year,
                    "cluster_id": cluster,
                    "vector": json.dumps([None if not np.isfinite(v) else float(v) for v in arr], ensure_ascii=False),
                    "n_valid_objects": int(np.isfinite(arr).sum()),
                    "missing_objects": ";".join(missing),
                })
    return vectors, pd.DataFrame(rows)


def vector_similarity(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    x = a[mask]
    y = b[mask]
    if metric == "cosine":
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom < 1e-12:
            return np.nan
        return float(np.dot(x, y) / denom)
    if metric == "pearson":
        x0 = x - np.mean(x)
        y0 = y - np.mean(y)
        denom = np.linalg.norm(x0) * np.linalg.norm(y0)
        if denom < 1e-12:
            return np.nan
        return float(np.dot(x0, y0) / denom)
    if metric == "neg_euclidean":
        return float(-np.linalg.norm(x - y))
    raise ValueError(f"Unknown metric: {metric}")


def parse_pair(pair: str) -> tuple[str, str]:
    a, b = pair.split("_")
    return a, b


def perm_p_greater_or_equal(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    return float((np.sum(null >= obs) + 1.0) / (null.size + 1.0))


def configuration_coupling(settings: Settings, vectors: dict[str, dict[Any, dict[str, np.ndarray]]]) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    rng = np.random.default_rng(settings.random_seed)
    rows = []
    cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    for mode, year_map in vectors.items():
        years = np.asarray(sorted(year_map.keys()))
        if len(years) < settings.min_years_for_coupling:
            for pair in settings.pair_order:
                for metric in settings.similarity_metrics:
                    rows.append({"mode": mode, "pair": pair, "similarity_metric": metric, "status": "skipped_insufficient_years", "n_years": len(years)})
            continue
        for pair in settings.pair_order:
            c1, c2 = parse_pair(pair)
            for metric in settings.similarity_metrics:
                obs_by_year = []
                for y in years:
                    obs_by_year.append(vector_similarity(year_map[y][c1], year_map[y][c2], metric))
                obs_by_year = np.asarray(obs_by_year, dtype=float)
                obs_mean = float(np.nanmean(obs_by_year)) if np.isfinite(obs_by_year).any() else np.nan
                null_means = []
                for _ in range(settings.n_permutation):
                    perm_years = rng.permutation(years)
                    sims = []
                    for y, yp in zip(years, perm_years):
                        sims.append(vector_similarity(year_map[y][c1], year_map[yp][c2], metric))
                    null_means.append(np.nanmean(sims))
                null = np.asarray(null_means, dtype=float)
                null_mean = float(np.nanmean(null)) if np.isfinite(null).any() else np.nan
                null_p90 = float(np.nanpercentile(null, 90)) if np.isfinite(null).any() else np.nan
                null_p95 = float(np.nanpercentile(null, 95)) if np.isfinite(null).any() else np.nan
                pval = perm_p_greater_or_equal(obs_mean, null)
                if np.isfinite(pval) and np.isfinite(obs_mean) and np.isfinite(null_p95) and obs_mean > null_p95 and pval <= settings.strongly_coupled_p_threshold:
                    decision = "strong_coupled_above_null"
                elif np.isfinite(pval) and np.isfinite(obs_mean) and np.isfinite(null_p90) and obs_mean > null_p90 and pval <= settings.coupled_p_threshold:
                    decision = "coupled_above_null"
                else:
                    decision = "not_above_null"
                rows.append({
                    "mode": mode,
                    "pair": pair,
                    "similarity_metric": metric,
                    "status": "ok",
                    "n_years": len(years),
                    "observed_mean_similarity": obs_mean,
                    "null_mean_similarity": null_mean,
                    "null_p90_similarity": null_p90,
                    "null_p95_similarity": null_p95,
                    "observed_minus_null": obs_mean - null_mean if np.isfinite(obs_mean) and np.isfinite(null_mean) else np.nan,
                    "permutation_p": pval,
                    "decision": decision,
                })
                cache[(mode, pair, metric)] = {"years": years, "obs_by_year": obs_by_year, "null": null, "obs_mean": obs_mean, "decision": decision}
    return pd.DataFrame(rows), cache


def remove_object(vec: np.ndarray, obj: str, settings: Settings) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).copy()
    if obj in settings.object_order:
        i = settings.object_order.index(obj)
        arr[i] = np.nan
    return arr


def object_mask_contribution(settings: Settings, vectors: dict[str, dict[Any, dict[str, np.ndarray]]]) -> pd.DataFrame:
    rng = np.random.default_rng(settings.random_seed + 101)
    rows = []
    for mode, year_map in vectors.items():
        years = np.asarray(sorted(year_map.keys()))
        if len(years) < settings.min_years_for_coupling:
            for pair in settings.pair_order:
                for metric in settings.similarity_metrics:
                    for obj in settings.object_order:
                        rows.append({"mode": mode, "pair": pair, "similarity_metric": metric, "object_removed": obj, "status": "skipped_insufficient_years", "n_years": len(years)})
            continue
        for pair in settings.pair_order:
            c1, c2 = parse_pair(pair)
            for metric in settings.similarity_metrics:
                for obj in settings.object_order:
                    drops = []
                    for y in years:
                        sim_all = vector_similarity(year_map[y][c1], year_map[y][c2], metric)
                        sim_wo = vector_similarity(remove_object(year_map[y][c1], obj, settings), remove_object(year_map[y][c2], obj, settings), metric)
                        drops.append(sim_all - sim_wo if np.isfinite(sim_all) and np.isfinite(sim_wo) else np.nan)
                    drops = np.asarray(drops, dtype=float)
                    obs_drop_mean = float(np.nanmean(drops)) if np.isfinite(drops).any() else np.nan

                    null_drop_means = []
                    for _ in range(settings.n_permutation):
                        perm_years = rng.permutation(years)
                        pdrops = []
                        for y, yp in zip(years, perm_years):
                            sim_all = vector_similarity(year_map[y][c1], year_map[yp][c2], metric)
                            sim_wo = vector_similarity(remove_object(year_map[y][c1], obj, settings), remove_object(year_map[yp][c2], obj, settings), metric)
                            pdrops.append(sim_all - sim_wo if np.isfinite(sim_all) and np.isfinite(sim_wo) else np.nan)
                        null_drop_means.append(np.nanmean(pdrops))
                    null = np.asarray(null_drop_means, dtype=float)
                    null_mean = float(np.nanmean(null)) if np.isfinite(null).any() else np.nan
                    null_p90 = float(np.nanpercentile(null, 90)) if np.isfinite(null).any() else np.nan
                    pval = perm_p_greater_or_equal(obs_drop_mean, null)
                    if np.isfinite(obs_drop_mean) and obs_drop_mean < -1e-6:
                        cls = "negative_or_disruptive_dimension"
                    elif np.isfinite(obs_drop_mean) and np.isfinite(null_p90) and obs_drop_mean > null_p90 and np.isfinite(pval) and pval <= settings.coupled_p_threshold:
                        cls = "key_coupling_dimension"
                    elif np.isfinite(obs_drop_mean) and obs_drop_mean > 0:
                        cls = "secondary_dimension"
                    elif np.isfinite(obs_drop_mean) and abs(obs_drop_mean) <= 1e-6:
                        cls = "nonessential_dimension"
                    else:
                        cls = "ambiguous"
                    rows.append({
                        "mode": mode,
                        "pair": pair,
                        "similarity_metric": metric,
                        "object_removed": obj,
                        "status": "ok",
                        "n_years": len(years),
                        "observed_drop_mean": obs_drop_mean,
                        "null_drop_mean": null_mean,
                        "null_drop_p90": null_p90,
                        "observed_drop_minus_null": obs_drop_mean - null_mean if np.isfinite(obs_drop_mean) and np.isfinite(null_mean) else np.nan,
                        "permutation_p": pval,
                        "contribution_class": cls,
                        "interpretation_hint": contribution_hint(obj, pair, cls),
                    })
    return pd.DataFrame(rows)


def contribution_hint(obj: str, pair: str, cls: str) -> str:
    if cls == "key_coupling_dimension":
        return f"{obj} removal substantially weakens {pair} configuration coupling under this metric; {obj} may be a key configuration dimension."
    if cls == "secondary_dimension":
        return f"{obj} contributes positively but not above the shuffled null p90; treat as secondary."
    if cls == "negative_or_disruptive_dimension":
        return f"Removing {obj} increases similarity; {obj} is not aligned with this pair's common configuration direction."
    if cls == "nonessential_dimension":
        return f"Removing {obj} barely changes this configuration coupling."
    return "Ambiguous contribution."


# ------------------------- trajectory clustering ----------------------------

def make_trajectory_matrix(settings: Settings, vectors: dict[Any, dict[str, np.ndarray]]) -> tuple[np.ndarray, list[Any]]:
    years = sorted(vectors.keys())
    mats = []
    for y in years:
        vals = []
        for cluster in [w.name for w in settings.windows]:
            vals.extend(list(vectors[y][cluster]))
        mats.append(vals)
    X = np.asarray(mats, dtype=float)
    # Fill occasional NaN with column means for clustering only.
    for j in range(X.shape[1]):
        col = X[:, j]
        mu = np.nanmean(col)
        if not np.isfinite(mu):
            mu = 0.0
        col[~np.isfinite(col)] = mu
        X[:, j] = col
    return X, years


def trajectory_clustering(settings: Settings, vectors: dict[str, dict[Any, dict[str, np.ndarray]]], config_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        scipy_ok = True
    except Exception:
        scipy_ok = False

    for mode, year_map in vectors.items():
        if len(year_map) < 3:
            continue
        X, years = make_trajectory_matrix(settings, year_map)
        labels_by_k: dict[int, np.ndarray] = {}
        if scipy_ok:
            Z = linkage(X, method="ward")
            for k in (2, 3, 4):
                labels_by_k[k] = fcluster(Z, k, criterion="maxclust")
        else:
            # Fallback: first principal component quantile bins. This is exploratory only.
            Xc = X - X.mean(axis=0, keepdims=True)
            try:
                _, _, vt = np.linalg.svd(Xc, full_matrices=False)
                pc1 = Xc @ vt[0]
            except Exception:
                pc1 = np.arange(len(years), dtype=float)
            for k in (2, 3, 4):
                qs = np.nanpercentile(pc1, np.linspace(0, 100, k + 1)[1:-1])
                labels_by_k[k] = np.digitize(pc1, qs) + 1

        for i, y in enumerate(years):
            sub = config_df[(config_df["mode"] == mode) & (config_df["year"] == y)]
            dom = {}
            for cluster in ("E1", "E2", "M"):
                ss = sub[sub["cluster_id"] == cluster]
                if ss.empty or ss["strength_z"].isna().all():
                    dom[cluster] = "NA"
                else:
                    dom[cluster] = str(ss.sort_values("strength_z", ascending=False)["object"].iloc[0])
            hint = trajectory_hint(dom)
            rows.append({
                "mode": mode,
                "year": y,
                "cluster_k2": int(labels_by_k[2][i]),
                "cluster_k3": int(labels_by_k[3][i]),
                "cluster_k4": int(labels_by_k[4][i]),
                "trajectory_vector_norm": float(np.linalg.norm(X[i])),
                "dominant_E1_object": dom.get("E1", "NA"),
                "dominant_E2_object": dom.get("E2", "NA"),
                "dominant_M_object": dom.get("M", "NA"),
                "trajectory_type_hint": hint,
                "clustering_backend": "scipy_ward" if scipy_ok else "pc1_quantile_fallback",
            })
    return pd.DataFrame(rows)


def trajectory_hint(dom: dict[str, str]) -> str:
    if dom.get("E1") == "H" or dom.get("E2") == "H":
        return "H_pre_strong_candidate"
    if dom.get("M") in ("Je", "Jw"):
        return "jet_entry_or_jet_dominant_candidate"
    if dom.get("M") in ("P", "V"):
        return "P_V_dominated_candidate"
    return "mixed_or_weakly_organized_candidate"


# ------------------------- route decision -----------------------------------

def first_row(df: pd.DataFrame, **conds) -> pd.Series | None:
    if df.empty:
        return None
    sub = df.copy()
    for k, v in conds.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        return None
    return sub.iloc[0]


def route_decision(settings: Settings, coupling_df: pd.DataFrame, contrib_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mode = settings.primary_mode if settings.primary_mode in set(coupling_df.get("mode", [])) else (coupling_df["mode"].iloc[0] if not coupling_df.empty else "NA")
    metric = settings.primary_similarity_metric

    def dec_for_pair(pair: str) -> str:
        r = first_row(coupling_df, mode=mode, pair=pair, similarity_metric=metric)
        return str(r["decision"]) if r is not None and "decision" in r else "unavailable"

    e1e2 = dec_for_pair("E1_E2")
    e2m = dec_for_pair("E2_M")
    e1m = dec_for_pair("E1_M")
    rows.append({
        "decision_item": "E2_to_M_configuration_coupling",
        "status": e2m,
        "evidence": evidence_for_pair(coupling_df, mode, "E2_M", metric),
        "route_implication": implication_for_e2m(e2m),
    })
    rows.append({
        "decision_item": "E1_to_E2_preconfiguration",
        "status": e1e2,
        "evidence": evidence_for_pair(coupling_df, mode, "E1_E2", metric),
        "route_implication": "If coupled, early pre-window multi-object activity may form an organized preconfiguration sequence; if not, E1 should not be over-linked to E2.",
    })
    rows.append({
        "decision_item": "E1_to_M_direct_link",
        "status": e1m,
        "evidence": evidence_for_pair(coupling_df, mode, "E1_M", metric),
        "route_implication": "Direct E1-M coupling would support a longer pre-window trajectory; absence suggests E1 cannot be directly tied to M without E2 evidence.",
    })

    # H contribution to E2-M.
    hrow = first_row(contrib_df, mode=mode, pair="E2_M", similarity_metric=metric, object_removed="H")
    hcls = str(hrow["contribution_class"]) if hrow is not None and "contribution_class" in hrow else "unavailable"
    rows.append({
        "decision_item": "H_contribution_to_E2_M_coupling",
        "status": hcls,
        "evidence": evidence_for_object(contrib_df, mode, "E2_M", metric, "H"),
        "route_implication": implication_for_h_contribution(hcls),
    })

    # Which objects are key in E2-M?
    sub = contrib_df[(contrib_df.get("mode") == mode) & (contrib_df.get("pair") == "E2_M") & (contrib_df.get("similarity_metric") == metric)] if not contrib_df.empty else pd.DataFrame()
    key_objs = []
    if not sub.empty:
        key_objs = sub[sub["contribution_class"].isin(["key_coupling_dimension", "secondary_dimension"])].sort_values("observed_drop_mean", ascending=False)["object_removed"].tolist()
    rows.append({
        "decision_item": "W45_configuration_route",
        "status": route_status(e2m, hcls, key_objs),
        "evidence": f"primary_mode={mode}; primary_metric={metric}; E2_M={e2m}; H_contribution={hcls}; positive_objects={','.join(key_objs)}",
        "route_implication": route_implication(e2m, hcls, key_objs),
    })
    return pd.DataFrame(rows)


def evidence_for_pair(df: pd.DataFrame, mode: str, pair: str, metric: str) -> str:
    r = first_row(df, mode=mode, pair=pair, similarity_metric=metric)
    if r is None:
        return "unavailable"
    return f"obs={r.get('observed_mean_similarity', np.nan):.4g}; null_p90={r.get('null_p90_similarity', np.nan):.4g}; p={r.get('permutation_p', np.nan):.4g}"


def evidence_for_object(df: pd.DataFrame, mode: str, pair: str, metric: str, obj: str) -> str:
    r = first_row(df, mode=mode, pair=pair, similarity_metric=metric, object_removed=obj)
    if r is None:
        return "unavailable"
    return f"drop={r.get('observed_drop_mean', np.nan):.4g}; null_p90={r.get('null_drop_p90', np.nan):.4g}; p={r.get('permutation_p', np.nan):.4g}"


def implication_for_e2m(status: str) -> str:
    if status in ("strong_coupled_above_null", "coupled_above_null"):
        return "E2 is supported as a within-year preconfiguration candidate for M; inspect object contributions instead of single-object leads."
    if status == "not_above_null":
        return "E2 multi-object activity should not be treated as organized W45 preconfiguration under this metric/mode."
    return "Insufficient data or unavailable decision."


def implication_for_h_contribution(status: str) -> str:
    if status == "key_coupling_dimension":
        return "H may participate in E2-M configuration coupling despite no H45 synchronous peak; retain H as a preconfiguration dimension."
    if status == "secondary_dimension":
        return "H contributes weakly/secondarily; retain only as a secondary E2 component, not a standalone precursor."
    if status == "nonessential_dimension":
        return "H does not materially support E2-M coupling under this metric; focus on other configuration dimensions."
    if status == "negative_or_disruptive_dimension":
        return "H is not aligned with E2-M common configuration direction under this metric."
    return "H contribution unresolved."


def route_status(e2m: str, hcls: str, key_objs: list[str]) -> str:
    if e2m not in ("strong_coupled_above_null", "coupled_above_null"):
        return "E2_not_supported_as_M_preconfiguration"
    if hcls == "key_coupling_dimension":
        return "H_possible_preconfiguration_dimension"
    if hcls == "secondary_dimension":
        return "H_secondary_E2_component"
    if key_objs:
        return "E2_M_coupling_without_H_as_key"
    return "E2_M_coupling_object_contribution_unclear"


def route_implication(e2m: str, hcls: str, key_objs: list[str]) -> str:
    if e2m not in ("strong_coupled_above_null", "coupled_above_null"):
        return "Do not interpret E1/E2 activity, including H35, as W45 preconfiguration without further evidence."
    if hcls == "key_coupling_dimension":
        return "H should be retained as part of W45 preconfiguration analysis, but not as a single-object precursor."
    if hcls == "secondary_dimension":
        return "H may remain as a secondary component of E2; main analysis should prioritize stronger coupling dimensions."
    if key_objs:
        return f"Prioritize {','.join(key_objs)} for W45 configuration analysis; H is not the key dimension under this audit."
    return "E2-M coupling exists but object contribution is unclear; inspect heatmaps and spatial fields."


# ------------------------- plotting -----------------------------------------

def save_heatmap_config(config_df: pd.DataFrame, settings: Settings, out: Path) -> None:
    mode = settings.primary_mode if settings.primary_mode in set(config_df["mode"]) else config_df["mode"].iloc[0]
    sub = config_df[config_df["mode"] == mode].copy()
    if sub.empty:
        return
    sub["col"] = sub["cluster_id"] + "_" + sub["object"]
    mat = sub.pivot_table(index="year", columns="col", values="strength_z", aggfunc="mean")
    columns = []
    for cluster in [w.name for w in settings.windows]:
        for obj in settings.object_order:
            c = f"{cluster}_{obj}"
            if c in mat.columns:
                columns.append(c)
    mat = mat[columns]
    fig_w = max(10, 0.45 * len(columns))
    fig_h = max(4, 0.22 * len(mat.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(f"W45 E1/E2/M configuration heatmap ({mode}, z-strength)")
    fig.colorbar(im, ax=ax, label="z-strength")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_coupling_plot(coupling_df: pd.DataFrame, settings: Settings, out: Path) -> None:
    mode = settings.primary_mode if settings.primary_mode in set(coupling_df.get("mode", [])) else (coupling_df["mode"].iloc[0] if not coupling_df.empty else None)
    metric = settings.primary_similarity_metric
    sub = coupling_df[(coupling_df["mode"] == mode) & (coupling_df["similarity_metric"] == metric)] if mode is not None and not coupling_df.empty else pd.DataFrame()
    if sub.empty:
        return
    pairs = settings.pair_order
    x = np.arange(len(pairs))
    obs = [float(sub[sub["pair"] == p]["observed_mean_similarity"].iloc[0]) if not sub[sub["pair"] == p].empty else np.nan for p in pairs]
    null = [float(sub[sub["pair"] == p]["null_mean_similarity"].iloc[0]) if not sub[sub["pair"] == p].empty else np.nan for p in pairs]
    p90 = [float(sub[sub["pair"] == p]["null_p90_similarity"].iloc[0]) if not sub[sub["pair"] == p].empty else np.nan for p in pairs]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.25
    ax.bar(x - width, obs, width, label="observed")
    ax.bar(x, null, width, label="shuffle null mean")
    ax.bar(x + width, p90, width, label="shuffle null p90")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel(metric)
    ax.set_title(f"Configuration coupling vs shuffled-year null ({mode})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_object_contribution_plot(contrib_df: pd.DataFrame, settings: Settings, out: Path) -> None:
    mode = settings.primary_mode if settings.primary_mode in set(contrib_df.get("mode", [])) else (contrib_df["mode"].iloc[0] if not contrib_df.empty else None)
    metric = settings.primary_similarity_metric
    sub = contrib_df[(contrib_df["mode"] == mode) & (contrib_df["pair"] == "E2_M") & (contrib_df["similarity_metric"] == metric)] if mode is not None and not contrib_df.empty else pd.DataFrame()
    if sub.empty:
        return
    sub = sub.set_index("object_removed").reindex(settings.object_order).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(np.arange(len(sub)), sub["observed_drop_mean"].to_numpy(dtype=float), label="observed drop")
    ax.plot(np.arange(len(sub)), sub["null_drop_p90"].to_numpy(dtype=float), marker="o", linestyle="--", label="shuffle null p90")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["object_removed"].tolist())
    ax.set_ylabel("sim_all - sim_without_object")
    ax.set_title(f"Object contribution to E2-M configuration coupling ({mode})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_dendrogram_or_distance(settings: Settings, vectors: dict[str, dict[Any, dict[str, np.ndarray]]], out: Path) -> None:
    mode = settings.primary_mode if settings.primary_mode in vectors else next(iter(vectors.keys()), None)
    if mode is None or len(vectors[mode]) < 3:
        return
    X, years = make_trajectory_matrix(settings, vectors[mode])
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(X, method="ward")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        dendrogram(Z, labels=[str(y) for y in years], ax=ax, leaf_rotation=90)
        ax.set_title(f"W45 configuration trajectory dendrogram ({mode})")
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
    except Exception:
        # Fallback distance matrix.
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(D, interpolation="nearest")
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels(years, rotation=90)
        ax.set_yticks(np.arange(len(years)))
        ax.set_yticklabels(years)
        ax.set_title(f"W45 configuration trajectory distance matrix ({mode})")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)


# ------------------------- summary ------------------------------------------

def write_summary(settings: Settings, out: Path, input_audit: pd.DataFrame, coupling_df: pd.DataFrame, contrib_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    mode = settings.primary_mode
    metric = settings.primary_similarity_metric
    lines = []
    lines.append(f"# V10.7_h W45 configuration trajectory audit")
    lines.append("")
    lines.append("## Method boundary")
    lines.append("- This audit treats W45 as a multi-object configuration made by P/V/H/Je/Jw.")
    lines.append("- It does not control P/V/Je/Jw away as covariates.")
    lines.append("- It tests within-year E1/E2/M configuration coupling against shuffled-year null.")
    lines.append("- It is not causal inference and does not prove physical pathways.")
    lines.append("")
    lines.append("## Input status")
    if not input_audit.empty:
        lines.append(input_audit[["object", "status", "source_key", "reducer", "source_note"]].to_markdown(index=False))
    lines.append("")
    lines.append(f"## Primary coupling decisions ({mode}, {metric})")
    sub = coupling_df[(coupling_df["mode"] == mode) & (coupling_df["similarity_metric"] == metric)] if not coupling_df.empty else pd.DataFrame()
    if sub.empty:
        lines.append("No coupling decisions available for the primary mode/metric.")
    else:
        cols = ["pair", "observed_mean_similarity", "null_p90_similarity", "permutation_p", "decision"]
        lines.append(sub[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## E2-M object contribution")
    subc = contrib_df[(contrib_df["mode"] == mode) & (contrib_df["pair"] == "E2_M") & (contrib_df["similarity_metric"] == metric)] if not contrib_df.empty else pd.DataFrame()
    if subc.empty:
        lines.append("No E2-M object contribution available for the primary mode/metric.")
    else:
        cols = ["object_removed", "observed_drop_mean", "null_drop_p90", "permutation_p", "contribution_class"]
        lines.append(subc[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Route decision")
    if not decision_df.empty:
        lines.append(decision_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Forbidden interpretations")
    lines.append("- Do not interpret object contribution as causality.")
    lines.append("- Do not treat E1/E2 object activity as W45 preconfiguration unless coupling exceeds shuffled-year null.")
    lines.append("- Do not treat H as absent from W45 formation merely because H has no synchronous H45 peak.")
    out.write_text("\n".join(lines), encoding="utf-8")


# ------------------------- pipeline -----------------------------------------

def run_w45_configuration_trajectory_v10_7_h(project_root: str | Path | None = None) -> dict[str, Any]:
    settings = Settings()
    # Optional speed/rigor override, e.g. set V10_7_H_N_PERM=5000 for final runs.
    try:
        _nperm = int(os.environ.get("V10_7_H_N_PERM", str(settings.n_permutation)))
        settings.n_permutation = max(100, _nperm)
    except Exception:
        pass
    if project_root is not None:
        settings.with_project_root(Path(project_root))
    out = settings.output_root()
    clean_output_root(out)

    meta: dict[str, Any] = {
        "version": settings.version,
        "task": "W45 E1-E2-M multi-object configuration trajectory audit",
        "status": "started",
        "created_at_utc": now_utc(),
        "project_root": str(settings.project_root),
        "output_root": str(out),
        "settings": settings.to_dict(),
        "method_boundary": [
            "not a regression-control experiment",
            "not causal inference",
            "does not control away W45 component objects",
            "configuration coupling against shuffled-year null",
        ],
    }
    try:
        print("[V10.7_h] loading smoothed fields...", flush=True)
        npz = load_npz(settings)
        meta["smoothed_fields_path"] = str(npz["path"])
        meta["npz_keys"] = sorted(list(npz["data"].keys()))
        print("[V10.7_h] building object fields...", flush=True)
        fields, input_audit = load_object_fields(settings, npz)
        write_dataframe(input_audit, out / "tables" / "w45_config_input_audit_v10_7_h.csv")

        if len(fields) < 3:
            raise RuntimeError(f"Too few objects loaded for configuration audit: {sorted(fields)}")

        print("[V10.7_h] building E1/E2/M object configuration table...", flush=True)
        config_df, metric_audit = build_object_configuration(settings, fields)
        write_dataframe(config_df, out / "tables" / "w45_e1_e2_m_object_configuration_by_year_v10_7_h.csv")
        write_dataframe(metric_audit, out / "tables" / "w45_object_strength_metric_audit_v10_7_h.csv")

        vectors, vector_df = build_config_vectors(config_df, settings)
        write_dataframe(vector_df, out / "tables" / "w45_configuration_vectors_v10_7_h.csv")

        print(f"[V10.7_h] computing configuration coupling with n_perm={settings.n_permutation}...", flush=True)
        coupling_df, coupling_cache = configuration_coupling(settings, vectors)
        write_dataframe(coupling_df, out / "tables" / "w45_configuration_coupling_v10_7_h.csv")

        print(f"[V10.7_h] computing mask-one-object contributions with n_perm={settings.n_permutation}...", flush=True)
        contrib_df = object_mask_contribution(settings, vectors)
        write_dataframe(contrib_df, out / "tables" / "w45_object_contribution_to_configuration_coupling_v10_7_h.csv")

        print("[V10.7_h] computing exploratory trajectory clusters...", flush=True)
        cluster_df = trajectory_clustering(settings, vectors, config_df)
        write_dataframe(cluster_df, out / "tables" / "w45_configuration_trajectory_year_clusters_v10_7_h.csv")

        decision_df = route_decision(settings, coupling_df, contrib_df)
        write_dataframe(decision_df, out / "tables" / "w45_configuration_route_decision_v10_7_h.csv")

        print("[V10.7_h] writing figures and summary...", flush=True)
        # Figures are not allowed to interrupt table outputs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                save_heatmap_config(config_df, settings, out / "figures" / "w45_e1_e2_m_configuration_heatmap_by_year_v10_7_h.png")
            except Exception as exc:
                meta["figure_heatmap_error"] = repr(exc)
            try:
                save_coupling_plot(coupling_df, settings, out / "figures" / "w45_configuration_coupling_vs_shuffle_null_v10_7_h.png")
            except Exception as exc:
                meta["figure_coupling_error"] = repr(exc)
            try:
                save_object_contribution_plot(contrib_df, settings, out / "figures" / "w45_object_removal_contribution_to_coupling_v10_7_h.png")
            except Exception as exc:
                meta["figure_contribution_error"] = repr(exc)
            try:
                save_dendrogram_or_distance(settings, vectors, out / "figures" / "w45_configuration_trajectory_dendrogram_v10_7_h.png")
            except Exception as exc:
                meta["figure_dendrogram_error"] = repr(exc)

        write_summary(settings, out / "summary_w45_configuration_trajectory_v10_7_h.md", input_audit, coupling_df, contrib_df, decision_df)
        meta.update({
            "status": "completed",
            "n_objects_loaded": len(fields),
            "objects_loaded": sorted(fields.keys()),
            "n_configuration_rows": int(len(config_df)),
            "n_coupling_rows": int(len(coupling_df)),
            "n_contribution_rows": int(len(contrib_df)),
            "completed_at_utc": now_utc(),
        })
    except Exception as exc:
        meta.update({"status": "failed", "error": repr(exc), "failed_at_utc": now_utc()})
        # best effort summary
        (out / "summary_w45_configuration_trajectory_v10_7_h.md").write_text(f"# V10.7_h failed\n\nError: `{repr(exc)}`\n", encoding="utf-8")
        raise
    finally:
        write_json(meta, out / "run_meta" / "run_meta_v10_7_h.json")
    return meta


if __name__ == "__main__":
    run_w45_configuration_trajectory_v10_7_h()
