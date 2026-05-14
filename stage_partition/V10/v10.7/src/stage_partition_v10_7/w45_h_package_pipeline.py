from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
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
class FieldSpec:
    object_name: str
    key_candidates: tuple[str, ...]
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]


@dataclass
class Settings:
    project_root: Path = Path(r"D:\easm_project01")
    version: str = "v10.7_f"
    output_tag: str = "w45_h_package_cross_object_audit_v10_7_f"
    smoothed_env_var: str = "V10_7_SMOOTHED_FIELDS"
    lat_key_candidates: tuple[str, ...] = ("lat", "latitude", "lats")
    lon_key_candidates: tuple[str, ...] = ("lon", "longitude", "lons")
    year_key_candidates: tuple[str, ...] = ("year", "years")
    day_key_candidates: tuple[str, ...] = ("day", "days", "day_index")

    # These domains are conservative defaults. They are proxies for event-strength extraction,
    # not final physical object definitions. If project-specific object masks/registries exist,
    # they should replace these defaults in a future hardening version.
    field_specs: tuple[FieldSpec, ...] = (
        FieldSpec("H", ("z500_smoothed", "z500", "H", "hgt500", "geopotential_500", "zg500"), (15.0, 35.0), (110.0, 140.0)),
        FieldSpec("P", ("precip_smoothed", "precip", "P", "pr", "rain", "tp"), (15.0, 35.0), (110.0, 140.0)),
        FieldSpec("V", ("v850_smoothed", "v850", "V", "v", "vwind850"), (15.0, 35.0), (110.0, 140.0)),
        FieldSpec("Je", ("u200_east_smoothed", "Je", "je", "u200_east", "u200_je"), (25.0, 45.0), (110.0, 150.0)),
        FieldSpec("Jw", ("u200_west_smoothed", "Jw", "jw", "u200_west", "u200_jw"), (25.0, 45.0), (60.0, 110.0)),
    )

    # H package windows.
    h_e1: Window = Window("H_E1", (14, 22))
    h_e2: Window = Window("H_E2", (31, 39))
    # E2 controls: kept as the object-layer precluster window.
    e2_control: Window = Window("E2_CONTROL", (27, 38))
    # W45 main-cluster target windows. H is intentionally excluded from main-cluster targets.
    target_windows: tuple[Window, ...] = (
        Window("joint45", (40, 48)),
        Window("P45", (42, 48)),
        Window("V45", (42, 48)),
        Window("Je46", (43, 48)),
        Window("Jw41", (39, 43)),
    )

    modes: tuple[str, ...] = ("anomaly", "local_background_removed", "raw")
    primary_mode: str = "anomaly"
    min_years_for_regression: int = 8
    n_bootstrap: int = 500
    n_permutation: int = 500
    random_seed: int = 20260514
    clear_delta_r2_threshold: float = 0.05
    weak_delta_r2_threshold: float = 0.02
    sign_stability_threshold: float = 0.70
    permutation_p_threshold: float = 0.10

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


def vector_norm(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float).ravel()
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.sum(arr[mask] ** 2)))


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
    lat_lo, lat_hi = sorted(lat_range); lon_lo, lon_hi = sorted(lon_range)
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi)
    lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError(f"No grid points in lat={lat_range}, lon={lon_range}")
    sub = field[:, :, lat_mask, :][:, :, :, lon_mask]
    sub_lat = lat[lat_mask]; sub_lon = lon[lon_mask]
    lat_order = np.argsort(sub_lat); lon_order = np.argsort(sub_lon)
    return sub[:, :, lat_order, :][:, :, :, lon_order], sub_lat[lat_order], sub_lon[lon_order]


def load_object_fields(settings: Settings, npz: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    data = npz["data"]
    fields: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for spec in settings.field_specs:
        key = first_key(data, spec.key_candidates)
        row = {"object": spec.object_name, "status": "missing", "field_key": key, "lat_range": spec.lat_range, "lon_range": spec.lon_range, "note": ""}
        if key is None:
            row["note"] = "No matching field key. This object will be skipped."
            rows.append(row); continue
        try:
            field, years, days, added_year = normalize_field_dims(data[key], data, npz["year_key"], npz["day_key"])
            if field.shape[2] != len(npz["lat"]) or field.shape[3] != len(npz["lon"]):
                raise ValueError(f"field shape {field.shape} mismatches lat/lon {npz['lat'].shape}/{npz['lon'].shape}")
            sub, sub_lat, sub_lon = subset_domain(field, npz["lat"], npz["lon"], spec.lat_range, spec.lon_range)
            fields[spec.object_name] = {"field": sub, "lat": sub_lat, "lon": sub_lon, "years": years, "days": days, "field_key": key, "added_year_axis": added_year, "spec": spec}
            row.update({"status": "loaded", "n_years": sub.shape[0], "n_days": sub.shape[1], "n_lat": sub.shape[2], "n_lon": sub.shape[3], "added_year_axis": added_year})
        except Exception as exc:
            row["status"] = "failed"
            row["note"] = str(exc)
        rows.append(row)
    return fields, pd.DataFrame(rows)


def day_indices(days: np.ndarray, window: Window) -> np.ndarray:
    day_arr = np.asarray(days)
    start, end = window.days
    mask = (day_arr >= start) & (day_arr <= end)
    return np.where(mask)[0]


def compute_daily_climatology(field: np.ndarray) -> np.ndarray:
    # field: year x day x lat x lon
    return safe_nanmean(field, axis=0, keepdims=True)


def compute_local_background(field: np.ndarray, days: np.ndarray, target_window: Window, pad: int = 8) -> np.ndarray:
    # A simple local linear trend across a broader window, excluding the target window.
    # Returns year x day x lat x lon background values for all days.
    n_years, n_days, n_lat, n_lon = field.shape
    start = max(int(target_window.days[0]) - pad, int(np.nanmin(days)))
    end = min(int(target_window.days[1]) + pad, int(np.nanmax(days)))
    day_arr = np.asarray(days, dtype=float)
    idx = np.where((day_arr >= start) & (day_arr <= end) & ~((day_arr >= target_window.days[0]) & (day_arr <= target_window.days[1])))[0]
    if idx.size < 4:
        return np.full_like(field, np.nan)
    x = day_arr[idx]
    x_mean = np.nanmean(x)
    x_center = x - x_mean
    den = np.sum(x_center ** 2)
    y = field[:, idx, :, :]
    y_mean = safe_nanmean(y, axis=1, keepdims=False)
    # slope per year/grid.
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.nansum((y - y_mean[:, None, :, :]) * x_center[None, :, None, None], axis=1) / den
    bg = y_mean[:, None, :, :] + slope[:, None, :, :] * (day_arr[None, :, None, None] - x_mean)
    return bg


def event_strength(field: np.ndarray, days: np.ndarray, window: Window, mode: str) -> tuple[np.ndarray, np.ndarray]:
    # field: year x day x lat x lon. Returns yearly norm strength and yearly mean signed diff.
    idx = day_indices(days, window)
    if idx.size == 0:
        n_years = field.shape[0]
        return np.full(n_years, np.nan), np.full(n_years, np.nan)
    if mode == "anomaly":
        working = field - compute_daily_climatology(field)
    elif mode == "local_background_removed":
        bg = compute_local_background(field, days, window)
        working = field - bg
    elif mode == "raw":
        working = field.copy()
    else:
        raise ValueError(f"Unknown mode {mode}")
    # Strength as RMS spatial magnitude in the target window mean; not an event-diff sign measure.
    mean_map = safe_nanmean(working[:, idx, :, :], axis=1)
    flat = mean_map.reshape(mean_map.shape[0], -1)
    strength = np.sqrt(np.nanmean(flat ** 2, axis=1))
    signed_mean = np.nanmean(flat, axis=1)
    return strength, signed_mean


def build_yearwise_strength_table(settings: Settings, fields: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in settings.modes:
        for obj, info in fields.items():
            f = info["field"]; days = info["days"]; years = info["years"]
            windows = []
            if obj == "H":
                windows.extend([settings.h_e1, settings.h_e2])
            if obj in ("P", "V", "Je"):
                windows.append(settings.e2_control)
            for w in settings.target_windows:
                if (obj == "P" and w.name == "P45") or (obj == "V" and w.name == "V45") or (obj == "Je" and w.name == "Je46") or (obj == "Jw" and w.name == "Jw41"):
                    windows.append(w)
            # The joint45 target is derived later from object targets, not from a field named joint.
            for w in windows:
                strength, signed = event_strength(f, days, w, mode)
                for yi, year in enumerate(years):
                    rows.append({
                        "mode": mode,
                        "year": int(year) if np.issubdtype(np.asarray(year).dtype, np.number) else str(year),
                        "object": obj,
                        "window": w.name,
                        "strength": float(strength[yi]) if np.isfinite(strength[yi]) else np.nan,
                        "signed_mean": float(signed[yi]) if np.isfinite(signed[yi]) else np.nan,
                    })
    return pd.DataFrame(rows)


def wide_features_from_strengths(strength_df: pd.DataFrame) -> pd.DataFrame:
    if strength_df.empty:
        return pd.DataFrame()
    parts = []
    for (mode, year), g in strength_df.groupby(["mode", "year"], dropna=False):
        row = {"mode": mode, "year": year}
        for _, r in g.iterrows():
            key = f"{r['object']}_{r['window']}_strength"
            row[key] = r["strength"]
        parts.append(row)
    wide = pd.DataFrame(parts)
    # Build H package if possible.
    for mode in wide["mode"].dropna().unique():
        m = wide["mode"] == mode
        if "H_H_E1_strength" in wide and "H_H_E2_strength" in wide:
            e1z = zscore_series(wide.loc[m, "H_H_E1_strength"].to_numpy(float))
            e2z = zscore_series(wide.loc[m, "H_H_E2_strength"].to_numpy(float))
            pkg = np.nanmean(np.vstack([e1z, e2z]), axis=0)
            wide.loc[m, "H_package_strength"] = pkg
    # Main cluster combined: use available object targets.
    target_cols = [c for c in ["P_P45_strength", "V_V45_strength", "Je_Je46_strength", "Jw_Jw41_strength"] if c in wide.columns]
    for mode in wide["mode"].dropna().unique():
        m = wide["mode"] == mode
        if target_cols:
            zmat = []
            for c in target_cols:
                zmat.append(zscore_series(wide.loc[m, c].to_numpy(float)))
            wide.loc[m, "M_combined_strength"] = np.nanmean(np.vstack(zmat), axis=0)
            # joint45 proxy is the same combined target in this heuristic version.
            wide.loc[m, "joint45_strength_proxy"] = wide.loc[m, "M_combined_strength"]
    return wide


def finite_xy(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[xcol].to_numpy(float); y = df[ycol].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return s.rank(method="average").to_numpy(float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata_simple(x), rankdata_simple(y))


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    if x.size < 4:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n):
        idx = rng.integers(0, x.size, size=x.size)
        vals.append(pearson(x[idx], y[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def permutation_corr_p(x: np.ndarray, y: np.ndarray, observed: float, n: int, rng: np.random.Generator) -> float:
    if x.size < 4 or not np.isfinite(observed):
        return float("nan")
    count = 0
    for _ in range(n):
        yp = rng.permutation(y)
        val = abs(pearson(x, yp))
        if np.isfinite(val) and val >= abs(observed):
            count += 1
    return float((count + 1) / (n + 1))


def build_correlation_table(settings: Settings, wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(settings.random_seed)
    targets = ["joint45_strength_proxy", "P_P45_strength", "V_V45_strength", "Je_Je46_strength", "Jw_Jw41_strength", "M_combined_strength"]
    for mode, g in wide.groupby("mode"):
        if "H_package_strength" not in g.columns:
            continue
        for target in targets:
            if target not in g.columns:
                continue
            x, y = finite_xy(g, "H_package_strength", target)
            pr = pearson(x, y); sr = spearman(x, y)
            ci_low, ci_high = bootstrap_corr_ci(x, y, settings.n_bootstrap, rng)
            p = permutation_corr_p(x, y, pr, settings.n_permutation, rng)
            rows.append({"mode": mode, "predictor": "H_package_strength", "target": target, "n_years": int(x.size), "pearson_r": pr, "spearman_r": sr, "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high, "permutation_p": p})
    return pd.DataFrame(rows)


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
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv = X[mask]; yv = y[mask]
    if yv.size < X.shape[1] + 2:
        return mask, np.full_like(y, np.nan, dtype=float), float("nan"), np.full(X.shape[1] + 1, np.nan)
    Xd = np.column_stack([np.ones(Xv.shape[0]), Xv])
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    pred_v = Xd @ beta
    pred = np.full_like(y, np.nan, dtype=float); pred[mask] = pred_v
    ss_res = float(np.sum((yv - pred_v) ** 2)); ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return mask, pred, r2, beta


def loocv_rmse(X: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv = X[mask]; yv = y[mask]
    n = yv.size
    if n < X.shape[1] + 3:
        return float("nan")
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = np.ones(n, dtype=bool); tr[i] = False
        if tr.sum() < X.shape[1] + 1:
            continue
        Xd = np.column_stack([np.ones(tr.sum()), Xv[tr]])
        beta, *_ = np.linalg.lstsq(Xd, yv[tr], rcond=None)
        preds[i] = np.r_[1.0, Xv[i]] @ beta
    return float(np.sqrt(np.nanmean((yv - preds) ** 2)))


def sign_stability_loo(X: np.ndarray, y: np.ndarray, h_col_index: int) -> float:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xv = X[mask]; yv = y[mask]
    n = yv.size
    vals = []
    if n < X.shape[1] + 3:
        return float("nan")
    for i in range(n):
        tr = np.ones(n, dtype=bool); tr[i] = False
        Xd = np.column_stack([np.ones(tr.sum()), Xv[tr]])
        beta, *_ = np.linalg.lstsq(Xd, yv[tr], rcond=None)
        vals.append(beta[1 + h_col_index])
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0 or not np.any(np.isfinite(vals)):
        return float("nan")
    med_sign = np.sign(np.nanmedian(vals))
    if med_sign == 0:
        return float(np.nanmean(np.sign(vals) == 0))
    return float(np.nanmean(np.sign(vals) == med_sign))


def permutation_delta_r2_p(X_base: np.ndarray, h: np.ndarray, y: np.ndarray, obs_delta: float, n: int, rng: np.random.Generator) -> float:
    if not np.isfinite(obs_delta):
        return float("nan")
    count = 0
    for _ in range(n):
        hp = rng.permutation(h)
        _, _, r2b, _ = ols_fit_predict(X_base, y)
        _, _, r2e, _ = ols_fit_predict(np.column_stack([X_base, hp]), y)
        delta = r2e - r2b if np.isfinite(r2e) and np.isfinite(r2b) else np.nan
        if np.isfinite(delta) and delta >= obs_delta:
            count += 1
    return float((count + 1) / (n + 1))


def build_incremental_table(settings: Settings, wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(settings.random_seed + 1)
    targets = ["joint45_strength_proxy", "P_P45_strength", "V_V45_strength", "Je_Je46_strength", "Jw_Jw41_strength", "M_combined_strength"]
    controls_all = ["P_E2_CONTROL_strength", "V_E2_CONTROL_strength", "Je_E2_CONTROL_strength"]
    for mode, g0 in wide.groupby("mode"):
        g = g0.copy().reset_index(drop=True)
        if "H_package_strength" not in g.columns:
            continue
        for target in targets:
            if target not in g.columns:
                continue
            controls = [c for c in controls_all if c in g.columns]
            y = zscore_series(g[target].to_numpy(float))
            if len(controls) == 0:
                # Base model is intercept-only; implement as empty X.
                X_base = np.empty((len(g), 0))
            else:
                X_base = standardize_matrix(g[controls].to_numpy(float))
            h = zscore_series(g["H_package_strength"].to_numpy(float))
            X_ext = np.column_stack([X_base, h])
            _, _, base_r2, _ = ols_fit_predict(X_base, y)
            _, _, ext_r2, beta = ols_fit_predict(X_ext, y)
            base_rmse = loocv_rmse(X_base, y)
            ext_rmse = loocv_rmse(X_ext, y)
            delta_r2 = ext_r2 - base_r2 if np.isfinite(ext_r2) and np.isfinite(base_r2) else np.nan
            delta_rmse = base_rmse - ext_rmse if np.isfinite(base_rmse) and np.isfinite(ext_rmse) else np.nan
            h_coef = beta[-1] if beta.size else np.nan
            sign_stab = sign_stability_loo(X_ext, y, h_col_index=X_ext.shape[1] - 1)
            p = permutation_delta_r2_p(X_base, h, y, delta_r2, settings.n_permutation, rng)
            n_valid = int((np.isfinite(y) & np.all(np.isfinite(X_ext), axis=1)).sum())
            if n_valid < settings.min_years_for_regression:
                decision = "unstable_or_unresolved"
            elif np.isfinite(delta_r2) and delta_r2 >= settings.clear_delta_r2_threshold and np.isfinite(delta_rmse) and delta_rmse > 0 and np.isfinite(sign_stab) and sign_stab >= settings.sign_stability_threshold and np.isfinite(p) and p <= settings.permutation_p_threshold:
                decision = "clear_incremental_support"
            elif np.isfinite(delta_r2) and delta_r2 >= settings.weak_delta_r2_threshold and np.isfinite(delta_rmse) and delta_rmse > 0:
                decision = "weak_incremental_support"
            else:
                decision = "no_incremental_support"
            rows.append({
                "mode": mode,
                "target": target,
                "n_years_valid": n_valid,
                "base_model_predictors": ";".join(controls) if controls else "intercept_only",
                "extended_model_predictors": ";".join(controls + ["H_package_strength"]),
                "base_r2": base_r2,
                "extended_r2": ext_r2,
                "delta_r2": delta_r2,
                "base_cv_rmse": base_rmse,
                "extended_cv_rmse": ext_rmse,
                "delta_cv_rmse_positive_means_improvement": delta_rmse,
                "H_package_coef": h_coef,
                "H_package_coef_loo_sign_stability": sign_stab,
                "permutation_p_delta_r2": p,
                "decision": decision,
            })
    return pd.DataFrame(rows)


def build_route_decision(settings: Settings, incr_df: pd.DataFrame, input_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = incr_df[incr_df["mode"] == settings.primary_mode] if not incr_df.empty and "mode" in incr_df else pd.DataFrame()
    if primary.empty:
        rows.append({"decision_item": "H package line", "status": "unresolved_no_primary_mode_results", "evidence": "No anomaly-mode incremental table was produced.", "route_implication": "Cannot judge H package relation to W45 main cluster."})
        return pd.DataFrame(rows)
    clear_targets = primary.loc[primary["decision"] == "clear_incremental_support", "target"].tolist()
    weak_targets = primary.loc[primary["decision"] == "weak_incremental_support", "target"].tolist()
    if clear_targets:
        status = "keep_as_preconditioning_candidate"
        evidence = f"Clear incremental support for targets: {', '.join(clear_targets)}"
        implication = "Proceed to gated high-H vs low-H spatial composites for supported targets only."
    elif weak_targets:
        status = "keep_as_secondary_E2_component"
        evidence = f"Only weak incremental support for targets: {', '.join(weak_targets)}"
        implication = "Do not make H a main explanation; keep as a secondary E2/pre-W45 component and inspect only if needed."
    else:
        status = "downgrade_H_for_W45_main_explanation"
        evidence = "No target shows incremental support from H_package after E2 controls under primary mode."
        implication = "Shift W45 explanation toward P/V/Je/Jw main-cluster structure; do not continue H package as a primary line."
    rows.append({"decision_item": "H package line", "status": status, "evidence": evidence, "route_implication": implication})
    missing = input_audit[input_audit["status"] != "loaded"]
    if not missing.empty:
        rows.append({"decision_item": "input coverage", "status": "partial_inputs", "evidence": "; ".join([f"{r.object}:{r.status}" for r in missing.itertuples()]), "route_implication": "Interpret missing-object targets/controls as unavailable, not as negative evidence."})
    return pd.DataFrame(rows)


def plot_scatter(settings: Settings, wide: pd.DataFrame, out: Path) -> None:
    mode = settings.primary_mode
    g = wide[wide["mode"] == mode].copy()
    if g.empty or "H_package_strength" not in g:
        return
    targets = [c for c in ["joint45_strength_proxy", "P_P45_strength", "V_V45_strength", "Je_Je46_strength", "Jw_Jw41_strength", "M_combined_strength"] if c in g]
    if not targets:
        return
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=(6, max(3, 2.2*n)), squeeze=False)
    for ax, target in zip(axes.ravel(), targets):
        x, y = finite_xy(g, "H_package_strength", target)
        ax.scatter(x, y, s=28)
        if x.size >= 2:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            ax.plot(xx, coef[0] * xx + coef[1], linewidth=1)
        ax.set_xlabel("H_package_strength")
        ax.set_ylabel(target)
        ax.set_title(f"{mode}: H package vs {target}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_delta_r2(incr_df: pd.DataFrame, settings: Settings, out: Path) -> None:
    if incr_df.empty:
        return
    g = incr_df[incr_df["mode"] == settings.primary_mode].copy()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(g["target"].astype(str), g["delta_r2"].astype(float))
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("ΔR² from adding H_package")
    ax.set_title(f"{settings.primary_mode}: H_package incremental explanatory power")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_summary(path: Path, settings: Settings, input_audit: pd.DataFrame, route: pd.DataFrame, incr: pd.DataFrame) -> None:
    lines = []
    lines.append("# V10.7_f W45 H-package to main-cluster cross-object audit")
    lines.append("")
    lines.append("## Method boundary")
    lines.append("- This is a yearwise cross-object association / incremental-explanatory audit.")
    lines.append("- It does not test H35 as a single point; H35 was already closed as a stable-independent target by V10.7_d.")
    lines.append("- It does not infer causality. Positive results mean incremental yearwise association only.")
    lines.append("- Field-domain event strengths are proxy indices; missing object fields are skipped and logged.")
    lines.append("")
    lines.append("## Input coverage")
    if input_audit.empty:
        lines.append("No input audit records.")
    else:
        lines.append(input_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## Route decision")
    if route.empty:
        lines.append("No route decision produced.")
    else:
        lines.append(route.to_markdown(index=False))
    lines.append("")
    lines.append("## Primary-mode incremental table")
    if incr.empty:
        lines.append("No incremental table produced.")
    else:
        primary = incr[incr["mode"] == settings.primary_mode]
        keep_cols = ["target", "n_years_valid", "delta_r2", "delta_cv_rmse_positive_means_improvement", "H_package_coef_loo_sign_stability", "permutation_p_delta_r2", "decision"]
        lines.append(primary[keep_cols].to_markdown(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")


def run_w45_h_package_cross_object_audit_v10_7_f(project_root: Path | str | None = None) -> dict[str, Any]:
    settings = Settings()
    if project_root is not None:
        settings.with_project_root(Path(project_root))
    out = settings.output_root()
    clean_output_root(out)
    meta: dict[str, Any] = {"version": settings.version, "task": "W45 H-package to main-cluster cross-object audit", "start_time_utc": now_utc(), "settings": settings.to_dict(), "output_root": str(out)}
    try:
        npz = load_npz(settings)
        fields, input_audit = load_object_fields(settings, npz)
        strength = build_yearwise_strength_table(settings, fields)
        wide = wide_features_from_strengths(strength)
        corr = build_correlation_table(settings, wide)
        incr = build_incremental_table(settings, wide)
        route = build_route_decision(settings, incr, input_audit)

        write_dataframe(input_audit, out / "tables" / "w45_cross_object_input_audit_v10_7_f.csv")
        write_dataframe(strength, out / "tables" / "w45_yearwise_object_window_strength_long_v10_7_f.csv")
        write_dataframe(wide, out / "tables" / "w45_yearwise_H_package_and_main_cluster_strength_v10_7_f.csv")
        write_dataframe(corr, out / "tables" / "w45_H_package_to_main_cluster_correlation_v10_7_f.csv")
        write_dataframe(incr, out / "tables" / "w45_H_package_incremental_explanatory_power_v10_7_f.csv")
        write_dataframe(route, out / "tables" / "w45_H_package_route_decision_v10_7_f.csv")
        plot_scatter(settings, wide, out / "figures" / "w45_H_package_vs_main_cluster_targets_v10_7_f.png")
        plot_delta_r2(incr, settings, out / "figures" / "w45_H_package_delta_r2_by_target_v10_7_f.png")
        write_summary(out / "summary_w45_h_package_cross_object_audit_v10_7_f.md", settings, input_audit, route, incr)
        meta.update({"status": "success", "input_objects_loaded": sorted(fields.keys()), "n_strength_rows": int(len(strength)), "n_wide_rows": int(len(wide)), "route_decision_records": route.to_dict(orient="records")})
    except Exception as exc:
        meta.update({"status": "failed", "error": repr(exc)})
        raise
    finally:
        meta["end_time_utc"] = now_utc()
        write_json(meta, out / "run_meta" / "run_meta_v10_7_f.json")
    return meta


__all__ = ["run_w45_h_package_cross_object_audit_v10_7_f"]
