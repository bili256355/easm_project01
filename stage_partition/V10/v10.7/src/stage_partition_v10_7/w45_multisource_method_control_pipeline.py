from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    version: str = "v10.7_g"
    output_tag: str = "w45_multisource_method_control_v10_7_g"
    smoothed_env_var: str = "V10_7_SMOOTHED_FIELDS"

    lat_key_candidates: tuple[str, ...] = ("lat", "latitude", "lats")
    lon_key_candidates: tuple[str, ...] = ("lon", "longitude", "lons")
    year_key_candidates: tuple[str, ...] = ("year", "years")
    day_key_candidates: tuple[str, ...] = ("day", "days", "day_index")

    # V10 object-event domains. Je/Jw are derived from u200, not from independent fields.
    object_specs: tuple[ObjectSpec, ...] = (
        ObjectSpec("H", ("z500_smoothed", "z500", "H", "hgt500", "geopotential_500", "zg500"), (15.0, 35.0), (110.0, 140.0), "spatial_rms", "H object domain"),
        ObjectSpec("P", ("precip_smoothed", "precip", "P", "pr", "rain", "tp"), (15.0, 35.0), (110.0, 140.0), "spatial_rms", "precip object proxy domain"),
        ObjectSpec("V", ("v850_smoothed", "v850", "V", "v", "vwind850"), (15.0, 35.0), (110.0, 140.0), "spatial_rms", "v850 object proxy domain"),
        ObjectSpec("Je", ("u200_smoothed", "u200", "u", "uwnd200"), (25.0, 45.0), (120.0, 150.0), "jet_q90_strength", "derived from u200 eastern sector: 120-150E, 25-45N"),
        ObjectSpec("Jw", ("u200_smoothed", "u200", "u", "uwnd200"), (25.0, 45.0), (80.0, 110.0), "jet_q90_strength", "derived from u200 western sector: 80-110E, 25-45N"),
    )

    e1: Window = Window("E1", (12, 23))
    e2: Window = Window("E2", (27, 38))
    h_e1: Window = Window("H_E1", (14, 22))
    h_e2: Window = Window("H_E2", (31, 39))

    target_windows: dict[str, Window] = None  # initialized in __post_init__

    modes: tuple[str, ...] = ("anomaly", "local_background_removed", "raw")
    primary_mode: str = "anomaly"
    min_years_for_regression: int = 8
    n_bootstrap: int = 100
    n_permutation: int = 100
    random_seed: int = 20260514
    clear_delta_r2_threshold: float = 0.05
    weak_delta_r2_threshold: float = 0.02
    sign_stability_threshold: float = 0.70
    permutation_p_threshold: float = 0.10

    def __post_init__(self):
        if self.target_windows is None:
            self.target_windows = {
                "P": Window("P45", (42, 48)),
                "V": Window("V45", (42, 48)),
                "Je": Window("Je46", (43, 48)),
                "Jw": Window("Jw41", (39, 43)),
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


def zscore_series(values: np.ndarray) -> np.ndarray:
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
            fields[spec.object_name] = {"field": sub, "lat": sub_lat, "lon": sub_lon, "years": years, "days": days, "source_key": key, "added_year_axis": added_year, "spec": spec}
            row.update({"status": "loaded", "n_years": int(sub.shape[0]), "n_days": int(sub.shape[1]), "n_lat": int(sub.shape[2]), "n_lon": int(sub.shape[3]), "added_year_axis": added_year})
        except Exception as exc:
            row["status"] = "failed"
            row["note"] = str(exc)
        rows.append(row)
    return fields, pd.DataFrame(rows)


def day_indices(days: np.ndarray, window: Window) -> np.ndarray:
    day_arr = np.asarray(days)
    start, end = window.days
    return np.where((day_arr >= start) & (day_arr <= end))[0]


def compute_daily_climatology(field: np.ndarray) -> np.ndarray:
    return safe_nanmean(field, axis=0, keepdims=True)


def compute_local_background(field: np.ndarray, days: np.ndarray, target_window: Window, pad: int = 8) -> np.ndarray:
    n_years, n_days, n_lat, n_lon = field.shape
    day_arr = np.asarray(days, dtype=float)
    start = max(int(target_window.days[0]) - pad, int(np.nanmin(day_arr)))
    end = min(int(target_window.days[1]) + pad, int(np.nanmax(day_arr)))
    idx = np.where((day_arr >= start) & (day_arr <= end) & ~((day_arr >= target_window.days[0]) & (day_arr <= target_window.days[1])))[0]
    if idx.size < 4:
        return np.full_like(field, np.nan)
    x = day_arr[idx]
    x_mean = np.nanmean(x)
    x_center = x - x_mean
    den = np.sum(x_center ** 2)
    y = field[:, idx, :, :]
    y_mean = safe_nanmean(y, axis=1, keepdims=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.nansum((y - y_mean[:, None, :, :]) * x_center[None, :, None, None], axis=1) / den
    bg = y_mean[:, None, :, :] + slope[:, None, :, :] * (day_arr[None, :, None, None] - x_mean)
    return bg


def prepare_working_field(field: np.ndarray, days: np.ndarray, window: Window, mode: str) -> np.ndarray:
    if mode == "anomaly":
        return field - compute_daily_climatology(field)
    if mode == "local_background_removed":
        return field - compute_local_background(field, days, window)
    if mode == "raw":
        return field.copy()
    raise ValueError(f"Unknown mode {mode}")


def jet_q90_strength(field_window_mean: np.ndarray) -> np.ndarray:
    # field_window_mean: year x lat x lon. Mimic foundation index logic:
    # lon-mean lat profile -> q90 threshold -> mean of values >= q90.
    prof = safe_nanmean(field_window_mean, axis=2)  # year x lat
    q90 = np.nanpercentile(prof, 90, axis=1)
    out = np.full(prof.shape[0], np.nan)
    for i in range(prof.shape[0]):
        mask = np.isfinite(prof[i]) & (prof[i] >= q90[i])
        if np.any(mask):
            out[i] = float(np.nanmean(prof[i, mask]))
    return out


def spatial_rms_strength(field_window_mean: np.ndarray) -> np.ndarray:
    flat = field_window_mean.reshape(field_window_mean.shape[0], -1)
    return np.sqrt(np.nanmean(flat ** 2, axis=1))


def event_strength(info: dict[str, Any], window: Window, mode: str) -> np.ndarray:
    field = info["field"]
    days = info["days"]
    spec: ObjectSpec = info["spec"]
    idx = day_indices(days, window)
    n_years = field.shape[0]
    if idx.size == 0:
        return np.full(n_years, np.nan)
    working = prepare_working_field(field, days, window, mode)
    mean_map = safe_nanmean(working[:, idx, :, :], axis=1)
    if spec.reducer == "jet_q90_strength":
        return jet_q90_strength(mean_map)
    return spatial_rms_strength(mean_map)


def object_windows_for_strength(settings: Settings, obj: str) -> list[Window]:
    windows = [settings.e1, settings.e2]
    if obj == "H":
        windows.extend([settings.h_e1, settings.h_e2])
    if obj in settings.target_windows:
        windows.append(settings.target_windows[obj])
    return windows


def build_yearwise_strength_table(settings: Settings, fields: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in settings.modes:
        for obj, info in fields.items():
            for w in object_windows_for_strength(settings, obj):
                strength = event_strength(info, w, mode)
                for yi, year in enumerate(info["years"]):
                    rows.append({
                        "mode": mode,
                        "year": int(year) if np.issubdtype(np.asarray(year).dtype, np.number) else str(year),
                        "object": obj,
                        "window": w.name,
                        "strength": float(strength[yi]) if np.isfinite(strength[yi]) else np.nan,
                    })
    return pd.DataFrame(rows)


def make_package(wide: pd.DataFrame, mode_mask: pd.Series, cols: list[str]) -> np.ndarray:
    zmat = []
    for c in cols:
        if c in wide:
            zmat.append(zscore_series(wide.loc[mode_mask, c].to_numpy(float)))
    if not zmat:
        return np.full(mode_mask.sum(), np.nan)
    return np.nanmean(np.vstack(zmat), axis=0)


def wide_features_from_strengths(strength_df: pd.DataFrame) -> pd.DataFrame:
    if strength_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (mode, year), g in strength_df.groupby(["mode", "year"], dropna=False):
        row = {"mode": mode, "year": year}
        for _, r in g.iterrows():
            row[f"{r['object']}_{r['window']}_strength"] = r["strength"]
        rows.append(row)
    wide = pd.DataFrame(rows)
    if wide.empty:
        return wide
    for mode in wide["mode"].dropna().unique():
        m = wide["mode"] == mode
        # Source packages.
        pkg_defs = {
            "H_pre_package_strength": ["H_H_E1_strength", "H_H_E2_strength"],
            "P_pre_package_strength": ["P_E1_strength", "P_E2_strength"],
            "V_pre_package_strength": ["V_E1_strength", "V_E2_strength"],
            "Je_pre_package_strength": ["Je_E1_strength", "Je_E2_strength"],
            "Jw_pre_package_strength": ["Jw_E1_strength", "Jw_E2_strength"],
        }
        for out_col, cols in pkg_defs.items():
            if any(c in wide for c in cols):
                wide.loc[m, out_col] = make_package(wide, m, cols)
        # Main-cluster source variables.
        mapping = {"P_M_source_strength": "P_P45_strength", "V_M_source_strength": "V_V45_strength", "Je_M_source_strength": "Je_Je46_strength", "Jw_M_source_strength": "Jw_Jw41_strength"}
        for out_col, in_col in mapping.items():
            if in_col in wide:
                wide.loc[m, out_col] = zscore_series(wide.loc[m, in_col].to_numpy(float))
        target_cols = [c for c in ["P_P45_strength", "V_V45_strength", "Je_Je46_strength", "Jw_Jw41_strength"] if c in wide.columns]
        if target_cols:
            zmat = [zscore_series(wide.loc[m, c].to_numpy(float)) for c in target_cols]
            wide.loc[m, "M_combined_strength"] = np.nanmean(np.vstack(zmat), axis=0)
            wide.loc[m, "joint45_strength_proxy"] = wide.loc[m, "M_combined_strength"]
        # Leave-one-out combined targets.
        for obj, col in {"P": "P_P45_strength", "V": "V_V45_strength", "Je": "Je_Je46_strength", "Jw": "Jw_Jw41_strength"}.items():
            others = [c for c in target_cols if c != col]
            if others:
                wide.loc[m, f"M_minus_{obj}_strength"] = np.nanmean(np.vstack([zscore_series(wide.loc[m, c].to_numpy(float)) for c in others]), axis=0)
    return wide


def finite_mask_for(cols: list[str], df: pd.DataFrame) -> np.ndarray:
    if not cols:
        return np.ones(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= np.isfinite(df[c].to_numpy(float))
    return mask


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata_simple(x), rankdata_simple(y))


def standardize_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = X.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        mu = np.nanmean(col); sd = np.nanstd(col)
        if np.isfinite(sd) and sd > 1e-12:
            out[:, j] = (col - mu) / sd
        else:
            out[:, j] = np.nan
    return out


def ols_fit_predict(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(y)
    if X.shape[1] > 0:
        mask &= np.all(np.isfinite(X), axis=1)
    Xv = X[mask]
    yv = y[mask]
    n_pred = X.shape[1]
    if yv.size < n_pred + 2:
        return mask, np.full_like(y, np.nan, dtype=float), float("nan"), np.full(n_pred + 1, np.nan)
    Xd = np.column_stack([np.ones(Xv.shape[0]), Xv])
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    pred_v = Xd @ beta
    pred = np.full_like(y, np.nan, dtype=float); pred[mask] = pred_v
    ss_res = float(np.sum((yv - pred_v) ** 2)); ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return mask, pred, r2, beta


def loocv_rmse(X: np.ndarray, y: np.ndarray) -> float:
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(y)
    if X.shape[1] > 0:
        mask &= np.all(np.isfinite(X), axis=1)
    Xv = X[mask]
    yv = y[mask]
    n = yv.size
    n_pred = X.shape[1]
    if n < n_pred + 3:
        return float("nan")
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = np.ones(n, dtype=bool); tr[i] = False
        if tr.sum() < n_pred + 1:
            continue
        Xd = np.column_stack([np.ones(tr.sum()), Xv[tr]])
        beta, *_ = np.linalg.lstsq(Xd, yv[tr], rcond=None)
        preds[i] = np.r_[1.0, Xv[i]] @ beta
    return float(np.sqrt(np.nanmean((yv - preds) ** 2)))


def sign_stability_loo(X: np.ndarray, y: np.ndarray, source_col_index: int) -> float:
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv = X[mask]
    yv = y[mask]
    n = yv.size
    if n < X.shape[1] + 3:
        return float("nan")
    vals = []
    for i in range(n):
        tr = np.ones(n, dtype=bool); tr[i] = False
        Xd = np.column_stack([np.ones(tr.sum()), Xv[tr]])
        beta, *_ = np.linalg.lstsq(Xd, yv[tr], rcond=None)
        vals.append(beta[1 + source_col_index])
    vals = np.asarray(vals, dtype=float)
    med = np.nanmedian(vals)
    s = np.sign(med)
    if s == 0:
        return float(np.nanmean(np.sign(vals) == 0))
    return float(np.nanmean(np.sign(vals) == s))


def permutation_delta_r2_p(X_base: np.ndarray, source: np.ndarray, y: np.ndarray, obs_delta: float, n: int, rng: np.random.Generator) -> float:
    if not np.isfinite(obs_delta):
        return float("nan")
    if obs_delta <= 0:
        return 1.0
    _, _, r2b, _ = ols_fit_predict(X_base, y)
    if not np.isfinite(r2b):
        return float("nan")
    count = 0
    for _ in range(n):
        sp = rng.permutation(source)
        _, _, r2e, _ = ols_fit_predict(np.column_stack([X_base, sp]), y)
        delta = r2e - r2b if np.isfinite(r2e) else np.nan
        if np.isfinite(delta) and delta >= obs_delta:
            count += 1
    return float((count + 1) / (n + 1))


def source_definitions() -> list[dict[str, str]]:
    return [
        {"source_object": "H", "package_type": "pre_package", "source_col": "H_pre_package_strength", "interpretation": "pre-W45 H package"},
        {"source_object": "P", "package_type": "pre_package", "source_col": "P_pre_package_strength", "interpretation": "P E1/E2 pre-package"},
        {"source_object": "V", "package_type": "pre_package", "source_col": "V_pre_package_strength", "interpretation": "V E1/E2 pre-package"},
        {"source_object": "Je", "package_type": "pre_package", "source_col": "Je_pre_package_strength", "interpretation": "Je E1/E2 pre-package"},
        {"source_object": "Jw", "package_type": "main_cluster_source", "source_col": "Jw_M_source_strength", "interpretation": "Jw main-cluster source around day41"},
        {"source_object": "P", "package_type": "main_cluster_source", "source_col": "P_M_source_strength", "interpretation": "P main-cluster source"},
        {"source_object": "V", "package_type": "main_cluster_source", "source_col": "V_M_source_strength", "interpretation": "V main-cluster source"},
        {"source_object": "Je", "package_type": "main_cluster_source", "source_col": "Je_M_source_strength", "interpretation": "Je main-cluster source"},
    ]


def target_definitions() -> list[dict[str, str]]:
    return [
        {"target_object": "P", "target_col": "P_P45_strength", "target_type": "individual_main"},
        {"target_object": "V", "target_col": "V_V45_strength", "target_type": "individual_main"},
        {"target_object": "Je", "target_col": "Je_Je46_strength", "target_type": "individual_main"},
        {"target_object": "Jw", "target_col": "Jw_Jw41_strength", "target_type": "individual_main"},
        {"target_object": "M", "target_col": "M_combined_strength", "target_type": "combined_available"},
        {"target_object": "joint_proxy", "target_col": "joint45_strength_proxy", "target_type": "joint_proxy_available"},
        {"target_object": "M_minus_P", "target_col": "M_minus_P_strength", "target_type": "leave_one_object_out"},
        {"target_object": "M_minus_V", "target_col": "M_minus_V_strength", "target_type": "leave_one_object_out"},
        {"target_object": "M_minus_Je", "target_col": "M_minus_Je_strength", "target_type": "leave_one_object_out"},
        {"target_object": "M_minus_Jw", "target_col": "M_minus_Jw_strength", "target_type": "leave_one_object_out"},
    ]


def control_columns_for_source(source_object: str, wide: pd.DataFrame) -> list[str]:
    base = {"P": "P_E2_strength", "V": "V_E2_strength", "Je": "Je_E2_strength"}
    controls = []
    for obj, col in base.items():
        if obj != source_object and col in wide.columns:
            controls.append(col)
    return controls


def build_multisource_incremental_table(settings: Settings, wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(settings.random_seed + 11)
    for mode, g0 in wide.groupby("mode"):
        g = g0.copy().reset_index(drop=True)
        for src in source_definitions():
            source_col = src["source_col"]
            if source_col not in g.columns:
                continue
            for tgt in target_definitions():
                target_col = tgt["target_col"]
                if target_col not in g.columns:
                    continue
                source_obj = src["source_object"]
                target_obj = tgt["target_object"]
                # Self-target for individual targets only.
                is_self = target_obj == source_obj
                # For leave-one-out targets, mark as self-target when source has been excluded from the target by design.
                is_leaveout_for_source = target_obj == f"M_minus_{source_obj}"
                controls = control_columns_for_source(source_obj, g)
                y = zscore_series(g[target_col].to_numpy(float))
                source = zscore_series(g[source_col].to_numpy(float))
                X_base = standardize_matrix(g[controls].to_numpy(float)) if controls else np.empty((len(g), 0))
                X_ext = np.column_stack([X_base, source])
                _, _, base_r2, _ = ols_fit_predict(X_base, y)
                _, _, ext_r2, beta = ols_fit_predict(X_ext, y)
                base_rmse = loocv_rmse(X_base, y)
                ext_rmse = loocv_rmse(X_ext, y)
                delta_r2 = ext_r2 - base_r2 if np.isfinite(ext_r2) and np.isfinite(base_r2) else np.nan
                delta_rmse = base_rmse - ext_rmse if np.isfinite(base_rmse) and np.isfinite(ext_rmse) else np.nan
                source_coef = beta[-1] if beta.size else np.nan
                sign_stab = sign_stability_loo(X_ext, y, source_col_index=X_ext.shape[1] - 1)
                p_delta = permutation_delta_r2_p(X_base, source, y, delta_r2, settings.n_permutation, rng)
                n_valid = int((np.isfinite(y) & np.all(np.isfinite(X_ext), axis=1)).sum())
                if is_self and tgt["target_type"] == "individual_main":
                    decision = "self_target_only"
                elif n_valid < settings.min_years_for_regression:
                    decision = "unstable_or_unresolved"
                elif np.isfinite(delta_r2) and delta_r2 >= settings.clear_delta_r2_threshold and np.isfinite(delta_rmse) and delta_rmse > 0 and np.isfinite(sign_stab) and sign_stab >= settings.sign_stability_threshold and np.isfinite(p_delta) and p_delta <= settings.permutation_p_threshold:
                    decision = "clear_incremental_support"
                elif np.isfinite(delta_r2) and delta_r2 >= settings.weak_delta_r2_threshold and np.isfinite(delta_rmse) and delta_rmse > 0:
                    decision = "weak_incremental_support"
                else:
                    decision = "no_incremental_support"
                rows.append({
                    "mode": mode,
                    "source_object": source_obj,
                    "source_package_type": src["package_type"],
                    "source_col": source_col,
                    "target_object": target_obj,
                    "target_type": tgt["target_type"],
                    "target_col": target_col,
                    "is_self_target": bool(is_self),
                    "is_leaveout_for_source": bool(is_leaveout_for_source),
                    "n_years_valid": n_valid,
                    "base_predictors": ";".join(controls) if controls else "intercept_only",
                    "extended_predictors": ";".join(controls + [source_col]),
                    "base_r2": base_r2,
                    "extended_r2": ext_r2,
                    "delta_r2": delta_r2,
                    "base_cv_rmse": base_rmse,
                    "extended_cv_rmse": ext_rmse,
                    "delta_cv_rmse_positive_means_improvement": delta_rmse,
                    "source_coef": source_coef,
                    "source_coef_loo_sign_stability": sign_stab,
                    "permutation_p_delta_r2": p_delta,
                    "decision": decision,
                })
    return pd.DataFrame(rows)


def build_correlation_table(settings: Settings, wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(settings.random_seed + 12)
    for mode, g0 in wide.groupby("mode"):
        g = g0.copy()
        for src in source_definitions():
            scol = src["source_col"]
            if scol not in g.columns:
                continue
            for tgt in target_definitions():
                tcol = tgt["target_col"]
                if tcol not in g.columns:
                    continue
                x = g[scol].to_numpy(float); y = g[tcol].to_numpy(float)
                mask = np.isfinite(x) & np.isfinite(y)
                xv = x[mask]; yv = y[mask]
                pr = pearson(xv, yv); sr = spearman(xv, yv)
                # simple permutation p for pearson
                p = float("nan")
                if xv.size >= 4 and np.isfinite(pr):
                    count = 0
                    for _ in range(settings.n_permutation):
                        yp = rng.permutation(yv)
                        if abs(pearson(xv, yp)) >= abs(pr):
                            count += 1
                    p = (count + 1) / (settings.n_permutation + 1)
                rows.append({"mode": mode, "source_object": src["source_object"], "source_package_type": src["package_type"], "source_col": scol, "target_object": tgt["target_object"], "target_col": tcol, "is_self_target": bool(tgt["target_object"] == src["source_object"]), "n_years": int(xv.size), "pearson_r": pr, "spearman_r": sr, "permutation_p": p})
    return pd.DataFrame(rows)


def build_method_control_decision(settings: Settings, incr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = incr[incr["mode"] == settings.primary_mode].copy() if not incr.empty else pd.DataFrame()
    if primary.empty:
        rows.append({"decision_item": "method control", "status": "unresolved_no_primary_results", "evidence": "No anomaly-mode incremental records.", "route_implication": "Cannot judge method ability or H negative result."})
        return pd.DataFrame(rows)
    cross = primary[~primary["is_self_target"]].copy()
    non_h_cross = cross[cross["source_object"] != "H"]
    h_cross = cross[cross["source_object"] == "H"]
    clear_non_h = non_h_cross[non_h_cross["decision"] == "clear_incremental_support"]
    weak_non_h = non_h_cross[non_h_cross["decision"] == "weak_incremental_support"]
    clear_h = h_cross[h_cross["decision"] == "clear_incremental_support"]
    weak_h = h_cross[h_cross["decision"] == "weak_incremental_support"]
    self_support = primary[(primary["is_self_target"]) & (primary["decision"].isin(["clear_incremental_support", "weak_incremental_support", "self_target_only"]))]
    if not clear_non_h.empty or not weak_non_h.empty:
        method_status = "method_has_detectable_cross_object_signal"
        evidence = f"Non-H cross-target supports: clear={len(clear_non_h)}, weak={len(weak_non_h)}."
    elif not self_support.empty:
        method_status = "only_self_target_signal_detected"
        evidence = "No non-H cross-target support; self-target diagnostics exist only."
    else:
        method_status = "method_low_power_or_metric_problem"
        evidence = "No cross-target or self-target support detected under current metrics."
    rows.append({"decision_item": "method control", "status": method_status, "evidence": evidence, "route_implication": "Use this to decide whether H negative results are meaningful or whether the audit lacks power."})
    if clear_h.empty and weak_h.empty:
        if method_status == "method_has_detectable_cross_object_signal":
            h_status = "H_negative_result_meaningful"
            implication = "H package can be downgraded relative to sources that show cross-object support."
        else:
            h_status = "H_negative_result_not_decisive"
            implication = "Do not overinterpret H negative result because the method did not detect reliable cross-object signals."
        rows.append({"decision_item": "H package", "status": h_status, "evidence": "No H cross-target incremental support under primary mode.", "route_implication": implication})
    else:
        rows.append({"decision_item": "H package", "status": "H_package_has_cross_object_support", "evidence": f"H clear={len(clear_h)}, weak={len(weak_h)} cross-target supports.", "route_implication": "Keep H package as preconditioning candidate for supported targets only."})
    return pd.DataFrame(rows)


def build_object_route_decision(settings: Settings, incr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = incr[incr["mode"] == settings.primary_mode].copy() if not incr.empty else pd.DataFrame()
    for obj in ["H", "P", "V", "Je", "Jw"]:
        sub = primary[(primary["source_object"] == obj) & (~primary["is_self_target"])]
        clear = sub[sub["decision"] == "clear_incremental_support"]
        weak = sub[sub["decision"] == "weak_incremental_support"]
        if not clear.empty:
            status = "keep_as_cross_object_candidate"
            evidence = "; ".join([f"{r.source_package_type}->{r.target_object}" for r in clear.itertuples()])
            implication = "Eligible for gated spatial high/low composite on supported targets."
        elif not weak.empty:
            status = "keep_as_secondary_candidate"
            evidence = "; ".join([f"{r.source_package_type}->{r.target_object}" for r in weak.itertuples()])
            implication = "Do not make main explanation; retain only as secondary clue."
        else:
            status = "downgrade_or_no_cross_support"
            evidence = "No cross-target incremental support under primary mode."
            implication = "Do not prioritize as W45 cross-object organizer."
        rows.append({"source_object": obj, "status": status, "evidence": evidence, "route_implication": implication})
    return pd.DataFrame(rows)


def plot_delta_r2(incr: pd.DataFrame, settings: Settings, out: Path) -> None:
    if incr.empty:
        return
    g = incr[(incr["mode"] == settings.primary_mode) & (~incr["is_self_target"])].copy()
    if g.empty:
        return
    g["label"] = g["source_object"].astype(str) + ":" + g["source_package_type"].astype(str) + "→" + g["target_object"].astype(str)
    g = g.sort_values("delta_r2", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(g))))
    ax.barh(g["label"], g["delta_r2"].astype(float))
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("ΔR² from adding source package")
    ax.set_title(f"{settings.primary_mode}: cross-target incremental support")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_method_heatmap(incr: pd.DataFrame, settings: Settings, out: Path) -> None:
    if incr.empty:
        return
    g = incr[(incr["mode"] == settings.primary_mode) & (~incr["is_self_target"])].copy()
    if g.empty:
        return
    g["source"] = g["source_object"].astype(str) + "_" + g["source_package_type"].astype(str)
    pivot = g.pivot_table(index="source", columns="target_object", values="delta_r2", aggfunc="max")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title(f"{settings.primary_mode}: ΔR² heatmap")
    fig.colorbar(im, ax=ax, label="ΔR²")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_summary(path: Path, settings: Settings, input_audit: pd.DataFrame, method_decision: pd.DataFrame, object_decision: pd.DataFrame, incr: pd.DataFrame) -> None:
    lines = []
    lines.append("# V10.7_g W45 multisource method-control audit")
    lines.append("")
    lines.append("## Method boundary")
    lines.append("- This is a multisource yearwise incremental-association audit for W45.")
    lines.append("- Je/Jw are derived from u200 sectors, not searched as independent fields.")
    lines.append("- Self-target results are diagnostic only; cross-target results are the main evidence.")
    lines.append("- Positive results indicate yearwise incremental association, not causality.")
    lines.append("")
    lines.append("## Input audit")
    lines.append(input_audit.to_string(index=False) if not input_audit.empty else "No input audit.")
    lines.append("")
    lines.append("## Method-control decision")
    lines.append(method_decision.to_string(index=False) if not method_decision.empty else "No method decision.")
    lines.append("")
    lines.append("## Object route decision")
    lines.append(object_decision.to_string(index=False) if not object_decision.empty else "No object decision.")
    lines.append("")
    lines.append("## Primary-mode top cross-target incremental records")
    if incr.empty:
        lines.append("No incremental records.")
    else:
        primary = incr[(incr["mode"] == settings.primary_mode) & (~incr["is_self_target"])].copy()
        if primary.empty:
            lines.append("No primary-mode cross-target records.")
        else:
            keep = ["source_object", "source_package_type", "target_object", "target_type", "delta_r2", "delta_cv_rmse_positive_means_improvement", "source_coef_loo_sign_stability", "permutation_p_delta_r2", "decision"]
            lines.append(primary.sort_values("delta_r2", ascending=False)[keep].head(30).to_string(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def run_w45_multisource_method_control_v10_7_g(project_root: Path | str | None = None) -> dict[str, Any]:
    settings = Settings()
    if project_root is not None:
        settings.with_project_root(Path(project_root))
    out = settings.output_root()
    clean_output_root(out)
    meta: dict[str, Any] = {"version": settings.version, "task": "W45 multisource method-control audit", "start_time_utc": now_utc(), "settings": settings.to_dict(), "output_root": str(out)}
    try:
        npz = load_npz(settings)
        fields, input_audit = load_object_fields(settings, npz)
        strength = build_yearwise_strength_table(settings, fields)
        wide = wide_features_from_strengths(strength)
        corr = build_correlation_table(settings, wide)
        incr = build_multisource_incremental_table(settings, wide)
        method_decision = build_method_control_decision(settings, incr)
        object_decision = build_object_route_decision(settings, incr)

        write_dataframe(input_audit, out / "tables" / "w45_multisource_input_audit_v10_7_g.csv")
        write_dataframe(strength, out / "tables" / "w45_multisource_yearwise_strength_long_v10_7_g.csv")
        write_dataframe(wide, out / "tables" / "w45_multisource_yearwise_strength_v10_7_g.csv")
        write_dataframe(corr, out / "tables" / "w45_multisource_correlation_v10_7_g.csv")
        write_dataframe(incr, out / "tables" / "w45_multisource_incremental_explanatory_power_v10_7_g.csv")
        write_dataframe(method_decision, out / "tables" / "w45_method_control_decision_v10_7_g.csv")
        write_dataframe(object_decision, out / "tables" / "w45_object_route_decision_v10_7_g.csv")
        plot_delta_r2(incr, settings, out / "figures" / "w45_multisource_delta_r2_top_cross_targets_v10_7_g.png")
        plot_method_heatmap(incr, settings, out / "figures" / "w45_multisource_delta_r2_heatmap_v10_7_g.png")
        write_summary(out / "summary_w45_multisource_method_control_v10_7_g.md", settings, input_audit, method_decision, object_decision, incr)
        meta.update({"status": "success", "input_objects_loaded": sorted(fields.keys()), "n_strength_rows": int(len(strength)), "n_wide_rows": int(len(wide)), "method_decision": method_decision.to_dict(orient="records"), "object_decision": object_decision.to_dict(orient="records")})
    except Exception as exc:
        meta.update({"status": "failed", "error": repr(exc)})
        raise
    finally:
        meta["end_time_utc"] = now_utc()
        write_json(meta, out / "run_meta" / "run_meta_v10_7_g.json")
    return meta


__all__ = ["run_w45_multisource_method_control_v10_7_g"]
