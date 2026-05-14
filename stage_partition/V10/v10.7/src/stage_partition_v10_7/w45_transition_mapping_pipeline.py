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
# V10.7_i: W33-to-W45 cross-object transition mapping audit
# =============================================================================
# Method boundary:
# - This is NOT same-object configuration similarity (V10.7_h already did that).
# - This is NOT a regression-control experiment that controls away W45 objects.
# - It tests whether the E2/W33 object vector maps to the M/W45 object vector.
# - It allows H_E2 to map to non-H M targets (P/V/Je/Jw), because H need not peak at M.
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
    version: str = "v10.7_i"
    output_tag: str = "w45_transition_mapping_v10_7_i"
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
    primary_mode: str = "anomaly"
    object_order: tuple[str, ...] = ("P", "V", "H", "Je", "Jw")
    source_cluster: str = "E2"
    target_cluster: str = "M"

    local_background_windows: dict[str, tuple[int, int]] = None

    n_permutation: int = 200
    n_bootstrap: int = 200
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

    def __post_init__(self):
        if self.local_background_windows is None:
            self.local_background_windows = {
                "E1": (0, 30),
                "E2": (18, 48),
                "M": (30, 60),
            }
        env_perm = os.environ.get("V10_7_I_N_PERM")
        if env_perm:
            self.n_permutation = int(env_perm)
        env_boot = os.environ.get("V10_7_I_N_BOOT")
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


def zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.full_like(x, np.nan, dtype=float)
    return (x - mu) / sd


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
            row["note"] = "No matching source field key. Object skipped."
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
        raise ValueError(f"No days in window {window}; available range {days[:3]}...{days[-3:]}")
    return idx


def daily_anomaly(field: np.ndarray) -> np.ndarray:
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
            bidx = bidx_all
        cmean = safe_nanmean(field[:, cidx, :, :], axis=1)
        bmean = safe_nanmean(field[:, bidx, :, :], axis=1)
        return cmean - bmean
    raise ValueError(f"Unknown mode: {mode}")


def reduce_cluster_strength(cluster_field: np.ndarray, spec: ObjectSpec) -> np.ndarray:
    if spec.reducer == "spatial_rms":
        return np.sqrt(safe_nanmean(cluster_field ** 2, axis=(1, 2)))
    if spec.reducer == "jet_q90_strength":
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


def build_object_strengths(settings: Settings, fields: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for obj in settings.object_order:
        if obj not in fields:
            continue
        info = fields[obj]
        spec: ObjectSpec = info["spec"]
        for mode in settings.modes:
            for win in settings.windows:
                try:
                    cf = cluster_field_by_mode(info["field"], info["days"], win, mode, settings)
                    strength = reduce_cluster_strength(cf, spec)
                except Exception as exc:
                    strength = np.full(len(info["years"]), np.nan)
                    note = repr(exc)
                else:
                    note = ""
                for y, val in zip(info["years"], strength):
                    rows.append({
                        "year": int(y) if np.issubdtype(np.asarray(info["years"]).dtype, np.integer) else y,
                        "mode": mode,
                        "cluster_id": win.name,
                        "object": obj,
                        "window_day_min": win.days[0],
                        "window_day_max": win.days[1],
                        "strength_raw": float(val) if np.isfinite(val) else np.nan,
                        "note": note,
                    })
                metric_rows.append({"object": obj, "mode": mode, "cluster_id": win.name, "source_key": info["source_key"], "reducer": spec.reducer})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(metric_rows)
    df["strength_z"] = np.nan
    for (mode, obj), idx in df.groupby(["mode", "object"]).groups.items():
        vals = df.loc[idx, "strength_raw"].to_numpy(dtype=float)
        df.loc[idx, "strength_z"] = zscore(vals)
    return df, pd.DataFrame(metric_rows)


def build_vectors(strength_df: pd.DataFrame, settings: Settings) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    vector_data: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for mode in sorted(strength_df["mode"].dropna().unique()):
        sub = strength_df[strength_df["mode"] == mode]
        years = sorted(sub["year"].dropna().unique())
        X_rows = []
        Y_rows = []
        valid_years = []
        for y in years:
            sub_y = sub[sub["year"] == y]
            x = []
            yvec = []
            missing = []
            for obj in settings.object_order:
                cell_x = sub_y[(sub_y["cluster_id"] == settings.source_cluster) & (sub_y["object"] == obj)]
                cell_y = sub_y[(sub_y["cluster_id"] == settings.target_cluster) & (sub_y["object"] == obj)]
                vx = float(cell_x["strength_z"].iloc[0]) if not cell_x.empty else np.nan
                vy = float(cell_y["strength_z"].iloc[0]) if not cell_y.empty else np.nan
                x.append(vx)
                yvec.append(vy)
                if not np.isfinite(vx) or not np.isfinite(vy):
                    missing.append(obj)
            x = np.asarray(x, dtype=float)
            yvec = np.asarray(yvec, dtype=float)
            rows.append({"mode": mode, "year": y, "source_cluster": settings.source_cluster, "target_cluster": settings.target_cluster, "X_E2": json.dumps([None if not np.isfinite(v) else float(v) for v in x]), "Y_M": json.dumps([None if not np.isfinite(v) else float(v) for v in yvec]), "n_valid_source": int(np.isfinite(x).sum()), "n_valid_target": int(np.isfinite(yvec).sum()), "missing_objects": ";".join(missing)})
            if np.isfinite(x).all() and np.isfinite(yvec).all():
                X_rows.append(x)
                Y_rows.append(yvec)
                valid_years.append(y)
        vector_data[mode] = {"years": np.asarray(valid_years), "X": np.asarray(X_rows, dtype=float), "Y": np.asarray(Y_rows, dtype=float)}
    return vector_data, pd.DataFrame(rows)


# ------------------------- statistics ---------------------------------------

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    a = x[mask] - np.nanmean(x[mask])
    b = y[mask] - np.nanmean(y[mask])
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else np.nan


def rank_values(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy(dtype=float)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return pearson_corr(rank_values(x[mask]), rank_values(y[mask]))


def perm_p_abs_corr(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    return float((np.sum(np.abs(null) >= abs(obs)) + 1.0) / (null.size + 1.0))


def perm_p_greater(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        return np.nan
    return float((np.sum(null >= obs) + 1.0) / (null.size + 1.0))


def bootstrap_ci_corr(x: np.ndarray, y: np.ndarray, corr_fn, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(x)
    vals = []
    if n < 5:
        return np.nan, np.nan
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(corr_fn(x[idx], y[idx]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def support_class_pairwise(r: float, p: float, settings: Settings) -> str:
    if np.isfinite(r) and np.isfinite(p) and abs(r) >= settings.pairwise_clear_abs_r and p <= settings.pairwise_clear_p:
        return "clear_mapping_support"
    if np.isfinite(r) and np.isfinite(p) and abs(r) >= settings.pairwise_weak_abs_r and p <= settings.pairwise_weak_p:
        return "weak_mapping_support"
    return "no_mapping_support"


# ------------------------- pairwise transition matrix -----------------------

def pairwise_transition_matrix(settings: Settings, vectors: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(settings.random_seed)
    rows: list[dict[str, Any]] = []
    h_rows: list[dict[str, Any]] = []
    for mode, vd in vectors.items():
        X = vd["X"]
        Y = vd["Y"]
        years = vd["years"]
        if len(years) < settings.min_years:
            rows.append({"mode": mode, "status": "skipped_insufficient_years", "n_years": int(len(years))})
            continue
        for i, src in enumerate(settings.object_order):
            for j, tgt in enumerate(settings.object_order):
                xs = X[:, i]
                yt = Y[:, j]
                pr = pearson_corr(xs, yt)
                sr = spearman_corr(xs, yt)
                null_pr = []
                null_sr = []
                for _ in range(settings.n_permutation):
                    perm = rng.permutation(len(years))
                    null_pr.append(pearson_corr(xs, yt[perm]))
                    null_sr.append(spearman_corr(xs, yt[perm]))
                p_pr = perm_p_abs_corr(pr, np.asarray(null_pr))
                p_sr = perm_p_abs_corr(sr, np.asarray(null_sr))
                ci_lo, ci_hi = bootstrap_ci_corr(xs, yt, spearman_corr, settings.n_bootstrap, rng)
                # Main support class uses Spearman because it is less sensitive to outliers.
                cls = support_class_pairwise(sr, p_sr, settings)
                row = {
                    "mode": mode,
                    "status": "ok",
                    "n_years": int(len(years)),
                    "source_object_E2": src,
                    "target_object_M": tgt,
                    "pearson_r": pr,
                    "pearson_permutation_p": p_pr,
                    "spearman_r": sr,
                    "spearman_bootstrap_ci_low": ci_lo,
                    "spearman_bootstrap_ci_high": ci_hi,
                    "spearman_permutation_p": p_sr,
                    "support_class": cls,
                    "is_diagonal_same_object": bool(src == tgt),
                    "interpretation_hint": pairwise_hint(src, tgt, cls),
                }
                rows.append(row)
                if src == "H":
                    h_rows.append({
                        "mode": mode,
                        "target_object_M": tgt,
                        "spearman_r": sr,
                        "pearson_r": pr,
                        "permutation_p": p_sr,
                        "support_class": cls,
                        "interpretation_hint": h_specific_hint(tgt, cls),
                    })
    return pd.DataFrame(rows), pd.DataFrame(h_rows)


def pairwise_hint(src: str, tgt: str, cls: str) -> str:
    if cls == "no_mapping_support":
        return f"No route-level support for {src}_E2 to {tgt}_M under this scalar mapping."
    if src == tgt:
        return f"Same-object E2-to-M persistence candidate for {src}; not cross-object reorganization evidence."
    return f"Cross-object E2-to-M mapping candidate: {src}_E2 may correspond to {tgt}_M in same-year variability."


def h_specific_hint(tgt: str, cls: str) -> str:
    if cls == "no_mapping_support":
        return f"No evidence that H_E2 maps to M_{tgt} under this scalar diagnostic."
    if tgt == "H":
        return "H_E2 maps to H_M; this is same-object persistence, not the main W45 reorganization hypothesis."
    return f"H_E2 maps to M_{tgt}; candidate evidence for H as a preconfiguration dimension for non-H W45 object."


# ------------------------- ridge mapping utilities --------------------------

def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x_mu = X.mean(axis=0, keepdims=True)
    y_mu = Y.mean(axis=0, keepdims=True)
    Xc = X - x_mu
    Yc = Y - y_mu
    p = X.shape[1]
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ Yc)
    intercept = y_mu - x_mu @ beta
    return {"beta": beta, "intercept": intercept.ravel()}


def predict_ridge(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float) @ model["beta"] + model["intercept"]


def cv_predict_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    n = X.shape[0]
    pred = np.full_like(Y, np.nan, dtype=float)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        if tr.sum() < 3:
            continue
        model = fit_ridge(X[tr], Y[tr], alpha)
        pred[i:i+1] = predict_ridge(model, X[i:i+1])
    return pred


def mapping_skill(Y: np.ndarray, Yhat: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(Y) & np.isfinite(Yhat)
    if mask.sum() == 0:
        return {"cv_r2": np.nan, "cv_rmse": np.nan}
    resid = Y[mask] - Yhat[mask]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    baseline = Y - np.nanmean(Y, axis=0, keepdims=True)
    den = float(np.nansum(baseline ** 2))
    num = float(np.nansum((Y - Yhat) ** 2))
    r2 = 1.0 - num / den if den > 1e-12 else np.nan
    return {"cv_r2": float(r2), "cv_rmse": rmse}


def multivariate_mapping(settings: Settings, vectors: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[tuple[str, str, float], dict[str, Any]]]:
    rng = np.random.default_rng(settings.random_seed + 300)
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, float], dict[str, Any]] = {}
    for mode, vd in vectors.items():
        X = vd["X"]
        Y = vd["Y"]
        years = vd["years"]
        if len(years) < settings.min_years:
            rows.append({"mode": mode, "mapping_method": "ridge", "status": "skipped_insufficient_years", "n_years": int(len(years))})
            continue
        for alpha in settings.ridge_alphas:
            Yhat = cv_predict_ridge(X, Y, alpha)
            skill = mapping_skill(Y, Yhat)
            null_skills = []
            for _ in range(settings.n_permutation):
                perm = rng.permutation(len(years))
                Yp = Y[perm]
                Yhat_p = cv_predict_ridge(X, Yp, alpha)
                null_skills.append(mapping_skill(Yp, Yhat_p)["cv_r2"])
            null = np.asarray(null_skills, dtype=float)
            pval = perm_p_greater(skill["cv_r2"], null)
            null_p90 = float(np.nanpercentile(null, 90)) if np.isfinite(null).any() else np.nan
            null_p95 = float(np.nanpercentile(null, 95)) if np.isfinite(null).any() else np.nan
            if np.isfinite(skill["cv_r2"]) and np.isfinite(pval) and skill["cv_r2"] > null_p90 and pval <= settings.mapping_clear_p:
                status = "mapping_detected"
            elif np.isfinite(skill["cv_r2"]) and np.isfinite(pval) and skill["cv_r2"] > np.nanmean(null) and pval <= settings.mapping_weak_p:
                status = "weak_mapping"
            else:
                status = "no_mapping"
            row = {
                "mode": mode,
                "mapping_method": "ridge",
                "alpha_or_components": alpha,
                "target_set": "P,V,H,Je,Jw_M",
                "status": "ok",
                "n_years": int(len(years)),
                "cv_r2_mean": skill["cv_r2"],
                "cv_rmse_mean": skill["cv_rmse"],
                "null_cv_r2_mean": float(np.nanmean(null)) if np.isfinite(null).any() else np.nan,
                "null_cv_r2_p90": null_p90,
                "null_cv_r2_p95": null_p95,
                "permutation_p": pval,
                "mapping_status": status,
            }
            rows.append(row)
            cache[(mode, "ridge", float(alpha))] = {"X": X, "Y": Y, "Yhat": Yhat, "skill": skill, "null": null, "status": status, "years": years}
    return pd.DataFrame(rows), cache


# ------------------------- remove one source contribution -------------------

def remove_one_source_contribution(settings: Settings, vectors: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rng = np.random.default_rng(settings.random_seed + 400)
    rows: list[dict[str, Any]] = []
    alpha = settings.main_alpha
    for mode, vd in vectors.items():
        X = vd["X"]
        Y = vd["Y"]
        years = vd["years"]
        if len(years) < settings.min_years:
            rows.append({"mode": mode, "mapping_method": "ridge", "status": "skipped_insufficient_years", "n_years": int(len(years))})
            continue
        full_skill = mapping_skill(Y, cv_predict_ridge(X, Y, alpha))["cv_r2"]
        for k, obj in enumerate(settings.object_order):
            keep = [i for i in range(len(settings.object_order)) if i != k]
            X_wo = X[:, keep]
            wo_skill = mapping_skill(Y, cv_predict_ridge(X_wo, Y, alpha))["cv_r2"]
            skill_drop = full_skill - wo_skill if np.isfinite(full_skill) and np.isfinite(wo_skill) else np.nan
            null_drops = []
            for _ in range(settings.n_permutation):
                perm = rng.permutation(len(years))
                Yp = Y[perm]
                fs = mapping_skill(Yp, cv_predict_ridge(X, Yp, alpha))["cv_r2"]
                ws = mapping_skill(Yp, cv_predict_ridge(X_wo, Yp, alpha))["cv_r2"]
                null_drops.append(fs - ws if np.isfinite(fs) and np.isfinite(ws) else np.nan)
            null = np.asarray(null_drops, dtype=float)
            pval = perm_p_greater(skill_drop, null)
            null_p90 = float(np.nanpercentile(null, 90)) if np.isfinite(null).any() else np.nan
            null_p95 = float(np.nanpercentile(null, 95)) if np.isfinite(null).any() else np.nan
            if np.isfinite(skill_drop) and skill_drop < -1e-8:
                cls = "negative_or_unstable_dimension"
            elif np.isfinite(skill_drop) and np.isfinite(null_p90) and skill_drop > null_p90 and np.isfinite(pval) and pval <= settings.mapping_clear_p:
                cls = "key_mapping_dimension"
            elif np.isfinite(skill_drop) and skill_drop > 0:
                cls = "secondary_mapping_dimension"
            elif np.isfinite(skill_drop) and abs(skill_drop) <= 1e-8:
                cls = "nonessential_dimension"
            else:
                cls = "ambiguous"
            rows.append({
                "mode": mode,
                "mapping_method": "ridge",
                "alpha": alpha,
                "source_removed": obj,
                "status": "ok",
                "n_years": int(len(years)),
                "full_skill_cv_r2": full_skill,
                "skill_without_source_cv_r2": wo_skill,
                "skill_drop": skill_drop,
                "null_drop_mean": float(np.nanmean(null)) if np.isfinite(null).any() else np.nan,
                "null_drop_p90": null_p90,
                "null_drop_p95": null_p95,
                "permutation_p": pval,
                "contribution_class": cls,
                "interpretation_hint": contribution_hint(obj, cls),
            })
    return pd.DataFrame(rows)


def contribution_hint(obj: str, cls: str) -> str:
    if cls == "key_mapping_dimension":
        return f"Removing {obj}_E2 substantially weakens E2-to-M mapping; {obj} may be a key preconfiguration dimension."
    if cls == "secondary_mapping_dimension":
        return f"Removing {obj}_E2 weakens mapping but not above null threshold; treat as secondary."
    if cls == "negative_or_unstable_dimension":
        return f"Removing {obj}_E2 improves or destabilizes mapping; {obj} is not a positive mapping dimension under this metric."
    if cls == "nonessential_dimension":
        return f"Removing {obj}_E2 barely changes mapping."
    return "Ambiguous contribution."


# ------------------------- route decision -----------------------------------

def route_decisions(settings: Settings, mapping_df: pd.DataFrame, contrib_df: pd.DataFrame, h_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = settings.primary_mode
    main_map = mapping_df[(mapping_df["mode"] == primary) & (mapping_df["mapping_method"] == "ridge") & (mapping_df["alpha_or_components"] == settings.main_alpha)]
    if main_map.empty:
        map_status = "unavailable"
        evidence = "No primary mapping row."
    else:
        map_status = str(main_map["mapping_status"].iloc[0])
        evidence = f"cv_r2={main_map['cv_r2_mean'].iloc[0]:.3f}, p={main_map['permutation_p'].iloc[0]:.3f}, null_p90={main_map['null_cv_r2_p90'].iloc[0]:.3f}"
    rows.append({"decision_item": "E2_to_M_cross_object_mapping", "status": map_status, "evidence": evidence, "route_implication": mapping_implication(map_status)})

    h_contrib = contrib_df[(contrib_df["mode"] == primary) & (contrib_df["source_removed"] == "H")]
    if h_contrib.empty:
        h_status = "unavailable"
        h_ev = "No H contribution row."
    else:
        h_status = str(h_contrib["contribution_class"].iloc[0])
        h_ev = f"skill_drop={h_contrib['skill_drop'].iloc[0]:.3f}, p={h_contrib['permutation_p'].iloc[0]:.3f}, null_p90={h_contrib['null_drop_p90'].iloc[0]:.3f}"
    rows.append({"decision_item": "H_E2_contribution_to_E2_M_mapping", "status": h_status, "evidence": h_ev, "route_implication": h_implication(h_status, map_status)})

    h_primary = h_df[h_df["mode"] == primary]
    supported = h_primary[h_primary["support_class"].isin(["clear_mapping_support", "weak_mapping_support"])]
    if supported.empty:
        h_target_status = "no_H_E2_target_mapping"
        h_target_ev = "No non-H M target has H_E2 mapping support under primary mode."
    else:
        targets = ";".join([f"M_{r.target_object_M}:{r.support_class},r={r.spearman_r:.2f},p={r.permutation_p:.3f}" for r in supported.itertuples() if r.target_object_M != "H"])
        if not targets:
            targets = ";".join([f"M_{r.target_object_M}:{r.support_class},r={r.spearman_r:.2f},p={r.permutation_p:.3f}" for r in supported.itertuples()])
        h_target_status = "H_E2_maps_to_M_target" if targets else "only_same_H_target_or_none"
        h_target_ev = targets if targets else "Only H_E2 to H_M support, if any."
    rows.append({"decision_item": "H_E2_target_specific_mapping", "status": h_target_status, "evidence": h_target_ev, "route_implication": "If H_E2 maps to non-H M targets, retain H as preconfiguration candidate; otherwise H remains unresolved/secondary."})

    if map_status in ("mapping_detected", "weak_mapping") and h_status in ("key_mapping_dimension", "secondary_mapping_dimension"):
        final = "W33_connected_to_W45_with_H_candidate"
    elif map_status in ("mapping_detected", "weak_mapping"):
        final = "W33_connected_to_W45_without_clear_H_role"
    elif map_status == "no_mapping":
        final = "W33_not_connected_to_W45_by_this_scalar_mapping"
    else:
        final = "unresolved"
    rows.append({"decision_item": "W33_to_W45_route", "status": final, "evidence": f"mapping={map_status}; H_contribution={h_status}; H_target_mapping={h_target_status}", "route_implication": final_implication(final)})
    return pd.DataFrame(rows)


def mapping_implication(status: str) -> str:
    if status == "mapping_detected":
        return "E2/W33 may connect to M/W45 through cross-object transition mapping; inspect source contributions and targets."
    if status == "weak_mapping":
        return "E2/W33 has weak mapping evidence; treat as candidate and require spatial/metric checks."
    if status == "no_mapping":
        return "No scalar cross-object mapping detected; W33 should not yet be treated as W45 preconfiguration."
    return "Mapping unavailable or unresolved."


def h_implication(h_status: str, map_status: str) -> str:
    if map_status not in ("mapping_detected", "weak_mapping"):
        return "Do not interpret H contribution because overall E2-to-M mapping is not established."
    if h_status == "key_mapping_dimension":
        return "H_E2 may be a key dimension in W33-to-W45 reorganization."
    if h_status == "secondary_mapping_dimension":
        return "H_E2 may be a secondary dimension; retain as candidate but not main conclusion."
    return "H_E2 is not supported as a useful mapping dimension under this scalar diagnostic."


def final_implication(final: str) -> str:
    if final == "W33_connected_to_W45_with_H_candidate":
        return "Proceed to spatial/structure checks for H_E2 mapping to non-H M targets, especially Jw/P/V."
    if final == "W33_connected_to_W45_without_clear_H_role":
        return "Focus on non-H E2-to-M transition dimensions; H should not be foregrounded."
    if final == "W33_not_connected_to_W45_by_this_scalar_mapping":
        return "W18-W33 sequence may be pre-window activity not connected to W45 under scalar mapping; consider richer position/shape metrics before final rejection."
    return "Unresolved; inspect input coverage and diagnostics."


# ------------------------- plotting -----------------------------------------

def save_pairwise_heatmap(df: pd.DataFrame, settings: Settings, out: Path) -> None:
    mode = settings.primary_mode
    sub = df[df["mode"] == mode]
    if sub.empty:
        return
    mat = np.full((len(settings.object_order), len(settings.object_order)), np.nan)
    labels = list(settings.object_order)
    for row in sub.itertuples():
        i = labels.index(row.source_object_E2)
        j = labels.index(row.target_object_M)
        mat[i, j] = row.spearman_r
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), [f"M_{x}" for x in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), [f"E2_{x}" for x in labels])
    ax.set_title(f"E2→M pairwise mapping matrix ({mode}, Spearman)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_mapping_skill_plot(df: pd.DataFrame, settings: Settings, out: Path) -> None:
    sub = df[(df["mapping_method"] == "ridge") & (df["alpha_or_components"] == settings.main_alpha)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sub))
    labels = [str(m) for m in sub["mode"]]
    ax.bar(x - 0.15, sub["cv_r2_mean"], width=0.3, label="observed CV R²")
    ax.bar(x + 0.15, sub["null_cv_r2_p90"], width=0.3, label="null p90")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("CV R²")
    ax.set_title("E2→M multivariate mapping skill vs shuffled-year null")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_remove_one_plot(df: pd.DataFrame, settings: Settings, out: Path) -> None:
    sub = df[df["mode"] == settings.primary_mode]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(sub["source_removed"])
    x = np.arange(len(labels))
    ax.bar(x - 0.15, sub["skill_drop"], width=0.3, label="observed drop")
    ax.bar(x + 0.15, sub["null_drop_p90"], width=0.3, label="null p90")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("CV R² drop when source removed")
    ax.set_title(f"Source contribution to E2→M mapping ({settings.primary_mode})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_h_specific_plot(df: pd.DataFrame, settings: Settings, out: Path) -> None:
    sub = df[df["mode"] == settings.primary_mode]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"M_{t}" for t in sub["target_object_M"]]
    vals = sub["spearman_r"].to_numpy(dtype=float)
    ax.bar(np.arange(len(labels)), vals)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Spearman r")
    ax.set_title(f"H_E2 target-specific mapping ({settings.primary_mode})")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# ------------------------- summary ------------------------------------------

def write_summary(settings: Settings, out_root: Path, route_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# V10.7_i W33→W45 cross-object transition mapping audit")
    lines.append("")
    lines.append("## Method boundary")
    lines.append("- This audit does not test same-object E2–M similarity; V10.7_h already handled that.")
    lines.append("- This audit does not control away P/V/Je/Jw as covariates.")
    lines.append("- It asks whether the E2/W33 object vector maps to the M/W45 object vector, allowing cross-object reorganization.")
    lines.append("- It allows H_E2 to map to non-H M objects such as M_Jw, M_P, or M_V.")
    lines.append("- It is not causal inference.")
    lines.append("")
    lines.append("## Route decision")
    for row in route_df.itertuples():
        lines.append(f"- **{row.decision_item}**: `{row.status}`. Evidence: {row.evidence}. Implication: {row.route_implication}")
    lines.append("")
    lines.append("## Forbidden interpretations")
    lines.append("- Do not interpret pairwise or multivariate mapping as causality.")
    lines.append("- Do not require H_E2 to map to H_M; H may matter through non-H M targets.")
    lines.append("- Do not use this scalar mapping to reject position/shape-based H roles; it only tests strength proxies.")
    (out_root / "summary_w45_transition_mapping_v10_7_i.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------- main pipeline ------------------------------------

def run_w45_transition_mapping_v10_7_i(project_root: str | Path | None = None) -> dict[str, Any]:
    settings = Settings()
    if project_root is not None:
        settings.with_project_root(Path(project_root))
    out_root = settings.output_root()
    clean_output_root(out_root)

    meta: dict[str, Any] = {
        "version": settings.version,
        "task": "W33-to-W45 cross-object transition mapping audit",
        "status": "started",
        "started_at_utc": now_utc(),
        "settings": settings.to_dict(),
        "output_root": str(out_root),
        "method_boundary": [
            "not same-object configuration similarity",
            "not regression-control experiment",
            "not causal inference",
            "tests cross-object E2-to-M mapping",
            "allows H_E2 to map to non-H M targets",
        ],
    }
    try:
        npz = load_npz(settings)
        meta["smoothed_fields_path"] = str(npz["path"])
        meta["available_keys"] = sorted(list(npz["data"].keys()))
        fields, audit_df = load_object_fields(settings, npz)
        write_dataframe(audit_df, out_root / "tables" / "w45_transition_mapping_input_audit_v10_7_i.csv")

        strengths_df, metrics_df = build_object_strengths(settings, fields)
        write_dataframe(strengths_df, out_root / "tables" / "w45_transition_mapping_yearwise_vectors_v10_7_i.csv")
        write_dataframe(metrics_df, out_root / "tables" / "w45_transition_mapping_object_metric_audit_v10_7_i.csv")

        vectors, vector_audit_df = build_vectors(strengths_df, settings)
        write_dataframe(vector_audit_df, out_root / "tables" / "w45_transition_mapping_vector_audit_v10_7_i.csv")

        pairwise_df, h_df = pairwise_transition_matrix(settings, vectors)
        write_dataframe(pairwise_df, out_root / "tables" / "w45_e2_to_m_pairwise_transition_matrix_v10_7_i.csv")
        write_dataframe(h_df, out_root / "tables" / "w45_h_e2_to_m_target_mapping_v10_7_i.csv")

        mapping_df, mapping_cache = multivariate_mapping(settings, vectors)
        write_dataframe(mapping_df, out_root / "tables" / "w45_e2_to_m_multivariate_mapping_skill_v10_7_i.csv")

        contrib_df = remove_one_source_contribution(settings, vectors)
        write_dataframe(contrib_df, out_root / "tables" / "w45_e2_source_object_contribution_to_m_mapping_v10_7_i.csv")

        route_df = route_decisions(settings, mapping_df, contrib_df, h_df)
        write_dataframe(route_df, out_root / "tables" / "w45_transition_mapping_route_decision_v10_7_i.csv")

        save_pairwise_heatmap(pairwise_df, settings, out_root / "figures" / "w45_e2_to_m_pairwise_transition_matrix_v10_7_i.png")
        save_mapping_skill_plot(mapping_df, settings, out_root / "figures" / "w45_e2_to_m_mapping_skill_vs_null_v10_7_i.png")
        save_remove_one_plot(contrib_df, settings, out_root / "figures" / "w45_e2_source_object_contribution_to_m_mapping_v10_7_i.png")
        save_h_specific_plot(h_df, settings, out_root / "figures" / "w45_h_e2_to_m_target_mapping_v10_7_i.png")

        write_summary(settings, out_root, route_df)
        meta.update({
            "status": "completed",
            "completed_at_utc": now_utc(),
            "n_loaded_objects": int((audit_df["status"] == "loaded").sum()) if not audit_df.empty else 0,
            "tables": sorted([p.name for p in (out_root / "tables").glob("*.csv")]),
            "figures": sorted([p.name for p in (out_root / "figures").glob("*.png")]),
        })
    except Exception as exc:
        meta.update({"status": "failed", "failed_at_utc": now_utc(), "error": repr(exc)})
        raise
    finally:
        write_json(meta, out_root / "run_meta" / "run_meta_v10_7_i.json")
    return meta


__all__ = ["run_w45_transition_mapping_v10_7_i"]
