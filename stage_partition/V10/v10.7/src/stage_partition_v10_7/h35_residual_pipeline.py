from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
class EventWindow:
    event_id: str
    pre_days: tuple[int, int]
    post_days: tuple[int, int]
    background_range: tuple[int, int]


@dataclass
class Settings:
    project_root: Path = Path(r"D:\easm_project01")
    version: str = "v10.7_d"
    output_tag: str = "h35_residual_independence_v10_7_d"
    smoothed_env_var: str = "V10_7_SMOOTHED_FIELDS"
    field_key_candidates: tuple[str, ...] = ("z500_smoothed", "z500", "H", "hgt500", "geopotential_500", "zg500")
    lat_key_candidates: tuple[str, ...] = ("lat", "latitude", "lats")
    lon_key_candidates: tuple[str, ...] = ("lon", "longitude", "lons")
    year_key_candidates: tuple[str, ...] = ("year", "years")
    day_key_candidates: tuple[str, ...] = ("day", "days", "day_index")
    object_lat_range: tuple[float, float] = (15.0, 35.0)
    object_lon_range: tuple[float, float] = (110.0, 140.0)
    events: tuple[EventWindow, ...] = (
        EventWindow("H18", (14, 17), (19, 22), (8, 28)),
        EventWindow("H35", (31, 34), (36, 39), (25, 43)),
        EventWindow("H45", (40, 43), (45, 48), (36, 52)),
        EventWindow("H57", (52, 55), (57, 60), (48, 66)),
    )
    pseudo_events: tuple[EventWindow, ...] = (
        EventWindow("PSEUDO_22_30", (22, 25), (27, 30), (16, 34)),
        EventWindow("PSEUDO_24_32", (24, 27), (29, 32), (18, 36)),
        EventWindow("PSEUDO_48_56", (48, 51), (53, 56), (42, 62)),
        EventWindow("PSEUDO_60_68", (60, 63), (65, 68), (54, 74)),
    )
    primary_modes_for_decision: tuple[str, ...] = ("anomaly", "local_background_removed")
    null_quantile: float = 0.90
    residual_fraction_moderate: float = 0.35
    residual_fraction_strong: float = 0.50
    n_bootstrap: int = 1000
    n_permutation: int = 2000
    random_seed: int = 20260514

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


def dot_finite(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel(); bb = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() == 0:
        return float("nan")
    return float(np.dot(aa[mask], bb[mask]))


def corr_finite(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel(); bb = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return float("nan")
    aa = aa[mask]; bb = bb[mask]
    if np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def cosine_finite(a: np.ndarray, b: np.ndarray) -> float:
    den = vector_norm(a) * vector_norm(b)
    if not np.isfinite(den) or den < 1e-12:
        return float("nan")
    return float(dot_finite(a, b) / den)


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty_like(arr, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    vals = arr[order]
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = np.mean(np.arange(i + 1, j + 1, dtype=float))
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=float); yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if mask.sum() < 3:
        return float("nan")
    return corr_finite(rankdata_simple(xx[mask]), rankdata_simple(yy[mask]))


def first_key(data: dict[str, np.ndarray], candidates: tuple[str, ...]) -> str | None:
    lower = {k.lower(): k for k in data.keys()}
    for c in candidates:
        if c in data:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_field(settings: Settings) -> dict[str, Any]:
    path = settings.smoothed_fields_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing smoothed field file: {path}")
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    field_key = first_key(data, settings.field_key_candidates)
    lat_key = first_key(data, settings.lat_key_candidates)
    lon_key = first_key(data, settings.lon_key_candidates)
    year_key = first_key(data, settings.year_key_candidates)
    day_key = first_key(data, settings.day_key_candidates)
    if field_key is None or lat_key is None or lon_key is None:
        raise KeyError(f"Could not detect field/lat/lon keys. Available={sorted(data.keys())}")
    field = np.asarray(data[field_key], dtype=float)
    original_shape = tuple(field.shape)
    added_year_axis = False
    if field.ndim == 3:
        field = field[None, ...]
        added_year_axis = True
    if field.ndim != 4:
        raise ValueError(f"Expected field as year x day x lat x lon or day x lat x lon; got {original_shape}")
    lat = np.asarray(data[lat_key], dtype=float); lon = np.asarray(data[lon_key], dtype=float)
    if field.shape[2] != len(lat) or field.shape[3] != len(lon):
        raise ValueError(f"Shape mismatch: field={field.shape}, lat={lat.shape}, lon={lon.shape}")
    if year_key is not None and len(np.asarray(data[year_key]).ravel()) == field.shape[0]:
        years = np.asarray(data[year_key]).ravel()
    else:
        years = np.arange(field.shape[0], dtype=int); year_key = None
    if day_key is not None and len(np.asarray(data[day_key]).ravel()) == field.shape[1]:
        days = np.asarray(data[day_key]).ravel()
    else:
        days = np.arange(field.shape[1], dtype=int); day_key = None
    return {"field": field, "lat": lat, "lon": lon, "years": years, "days": days, "field_key": field_key, "lat_key": lat_key, "lon_key": lon_key, "year_key": year_key, "day_key": day_key, "source_path": path, "original_shape": original_shape, "added_year_axis": added_year_axis}


def subset_domain(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]):
    lat_lo, lat_hi = sorted(lat_range); lon_lo, lon_hi = sorted(lon_range)
    lat_mask = (lat >= lat_lo) & (lat <= lat_hi); lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError(f"No grid points in lat={lat_range}, lon={lon_range}")
    sub = field[:, :, lat_mask, :][:, :, :, lon_mask]
    sub_lat = lat[lat_mask]; sub_lon = lon[lon_mask]
    lat_order = np.argsort(sub_lat); lon_order = np.argsort(sub_lon)
    return sub[:, :, lat_order, :][:, :, :, lon_order], sub_lat[lat_order], sub_lon[lon_order]


def day_range(pair: tuple[int, int], n_days: int) -> np.ndarray:
    s, e = int(pair[0]), int(pair[1])
    s = max(0, s); e = min(n_days - 1, e)
    if e < s:
        return np.array([], dtype=int)
    return np.arange(s, e + 1, dtype=int)


def event_diff(field: np.ndarray, event: EventWindow) -> np.ndarray:
    n_days = field.shape[1]
    pre = day_range(event.pre_days, n_days); post = day_range(event.post_days, n_days)
    if pre.size == 0 or post.size == 0:
        return np.full((field.shape[0],) + field.shape[2:], np.nan)
    return safe_nanmean(field[:, post, ...], axis=1) - safe_nanmean(field[:, pre, ...], axis=1)


def nan_slope(field: np.ndarray, days: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    x = np.asarray(days, dtype=float)
    valid = np.isfinite(arr)
    count = valid.sum(axis=1)
    xb = x[None, :, None, None]
    x_mean = np.where(count > 0, np.sum(np.where(valid, xb, 0.0), axis=1) / count, np.nan)
    y_mean = np.where(count > 0, np.nansum(arr, axis=1) / count, np.nan)
    xc = xb - x_mean[:, None, :, :]
    yc = arr - y_mean[:, None, :, :]
    num = np.nansum(np.where(valid, xc * yc, np.nan), axis=1)
    den = np.nansum(np.where(valid, xc * xc, np.nan), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = num / den
    return np.where((count >= 2) & np.isfinite(slope), slope, np.nan)


def local_background_removed_diff(field: np.ndarray, event: EventWindow) -> np.ndarray:
    n_days = field.shape[1]
    ev_days = set(np.concatenate([day_range(event.pre_days, n_days), day_range(event.post_days, n_days)]).tolist())
    bg = np.array([d for d in day_range(event.background_range, n_days) if d not in ev_days], dtype=int)
    diff = event_diff(field, event)
    if bg.size < 3:
        return diff
    slope = nan_slope(field[:, bg, ...], bg)
    dt = float(np.mean(day_range(event.post_days, n_days)) - np.mean(day_range(event.pre_days, n_days)))
    return diff - slope * dt


def build_modes(field: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    modes = {"raw": field, "local_background_removed": field}
    if field.shape[0] >= 2:
        clim = safe_nanmean(field, axis=0, keepdims=True)
        modes["anomaly"] = field - clim
        anomaly_status = "computed_from_year_mean_daily_climatology"
    else:
        anomaly_status = "skipped_no_year_dimension"
    return modes, {"available_modes": sorted(modes.keys()), "anomaly_status": anomaly_status}


def compute_diffs(modes: dict[str, np.ndarray], events: tuple[EventWindow, ...]) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for mode, field in modes.items():
        for ev in events:
            out[(mode, ev.event_id)] = local_background_removed_diff(field, ev) if mode == "local_background_removed" else event_diff(field, ev)
    return out


def spatial_to_profile(diff: np.ndarray) -> np.ndarray:
    return safe_nanmean(diff, axis=-1)


def project(target: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    den = dot_finite(template, template)
    if not np.isfinite(den) or den < 1e-12:
        fit = np.full_like(target, np.nan); resid = np.full_like(target, np.nan); alpha = np.nan
    else:
        alpha = dot_finite(target, template) / den
        fit = alpha * template
        resid = target - fit
    tnorm = vector_norm(target); fnorm = vector_norm(fit); rnorm = vector_norm(resid)
    with np.errstate(invalid="ignore", divide="ignore"):
        rfrac = (rnorm ** 2) / (tnorm ** 2)
    return fit, resid, {"alpha_h18_template": float(alpha), "target_norm": tnorm, "fitted_norm": fnorm, "residual_norm": rnorm, "residual_fraction": float(rfrac), "explained_fraction": float(1.0 - rfrac) if np.isfinite(rfrac) else np.nan, "cosine_target_template": cosine_finite(target, template), "corr_target_template": corr_finite(target, template)}


def residuals_for_h35(event_diffs: dict[tuple[str, str], np.ndarray], years: np.ndarray):
    rows, cons_rows = [], []
    arrays = {}
    for mode in sorted({m for m, _ in event_diffs}):
        h18s = event_diffs.get((mode, "H18")); h35s = event_diffs.get((mode, "H35"))
        if h18s is None or h35s is None:
            continue
        for domain, h18, h35 in (("object_domain_spatial", h18s, h35s), ("profile", spatial_to_profile(h18s), spatial_to_profile(h35s))):
            fits = np.full_like(h35, np.nan); resids = np.full_like(h35, np.nan); templates = np.full_like(h18, np.nan)
            for i in range(h35.shape[0]):
                template = np.nanmean(np.delete(h18, i, axis=0), axis=0) if h18.shape[0] > 1 else np.nanmean(h18, axis=0)
                fit, resid, metrics = project(h35[i], template)
                fits[i] = fit; resids[i] = resid; templates[i] = template
                rows.append({"year": years[i].item() if hasattr(years[i], "item") else years[i], "mode": mode, "domain": domain, **metrics})
            for i in range(resids.shape[0]):
                tmpl = np.nanmean(np.delete(resids, i, axis=0), axis=0) if resids.shape[0] > 1 else np.nanmean(resids, axis=0)
                sign_mask = np.isfinite(resids[i]) & np.isfinite(tmpl) & (np.abs(tmpl) > 1e-12)
                same_sign = float(np.mean(np.sign(resids[i][sign_mask]) == np.sign(tmpl[sign_mask]))) if sign_mask.sum() else np.nan
                cons_rows.append({"year": years[i].item() if hasattr(years[i], "item") else years[i], "mode": mode, "domain": domain, "pattern_corr_to_loo_residual_template": corr_finite(resids[i], tmpl), "cosine_to_loo_residual_template": cosine_finite(resids[i], tmpl), "same_sign_fraction_to_loo_residual_template": same_sign})
            arrays[(mode, domain)] = {"h18": h18, "h35": h35, "fitted": fits, "residual": resids, "templates": templates}
    return pd.DataFrame(rows), pd.DataFrame(cons_rows), arrays


def pseudo_null(modes: dict[str, np.ndarray], event_diffs: dict[tuple[str, str], np.ndarray], settings: Settings, years: np.ndarray):
    rows, cons_rows = [], []
    for mode, field in modes.items():
        h18s = event_diffs.get((mode, "H18"))
        if h18s is None:
            continue
        for pev in settings.pseudo_events:
            pdiff = local_background_removed_diff(field, pev) if mode == "local_background_removed" else event_diff(field, pev)
            for domain, h18, target in (("object_domain_spatial", h18s, pdiff), ("profile", spatial_to_profile(h18s), spatial_to_profile(pdiff))):
                resids = []
                for i in range(target.shape[0]):
                    template = np.nanmean(np.delete(h18, i, axis=0), axis=0) if h18.shape[0] > 1 else np.nanmean(h18, axis=0)
                    _, resid, metrics = project(target[i], template)
                    resids.append(resid)
                    rows.append({"year": years[i].item() if hasattr(years[i], "item") else years[i], "mode": mode, "domain": domain, "pseudo_event_id": pev.event_id, "pseudo_residual_norm": metrics["residual_norm"], "pseudo_residual_fraction": metrics["residual_fraction"], "pseudo_target_norm": metrics["target_norm"]})
                resids = np.asarray(resids)
                for i in range(resids.shape[0]):
                    tmpl = np.nanmean(np.delete(resids, i, axis=0), axis=0) if resids.shape[0] > 1 else np.nanmean(resids, axis=0)
                    cons_rows.append({"year": years[i].item() if hasattr(years[i], "item") else years[i], "mode": mode, "domain": domain, "pseudo_event_id": pev.event_id, "pseudo_pattern_corr": corr_finite(resids[i], tmpl), "pseudo_pattern_cosine": cosine_finite(resids[i], tmpl)})
    return pd.DataFrame(rows), pd.DataFrame(cons_rows)


def event_rows(event_diffs: dict[tuple[str, str], np.ndarray], years: np.ndarray) -> pd.DataFrame:
    rows = []
    for (mode, event_id), arr in event_diffs.items():
        for domain, data in (("object_domain_spatial", arr), ("profile", spatial_to_profile(arr))):
            for i in range(data.shape[0]):
                rows.append({"year": years[i].item() if hasattr(years[i], "item") else years[i], "mode": mode, "event_id": event_id, "domain": domain, "diff_norm": vector_norm(data[i]), "diff_abs_mean": float(np.nanmean(np.abs(data[i]))) if np.isfinite(data[i]).any() else np.nan, "diff_mean": float(np.nanmean(data[i])) if np.isfinite(data[i]).any() else np.nan, "diff_max": float(np.nanmax(data[i])) if np.isfinite(data[i]).any() else np.nan, "diff_min": float(np.nanmin(data[i])) if np.isfinite(data[i]).any() else np.nan})
    return pd.DataFrame(rows)


def q(series: pd.Series, quantile: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.nanquantile(vals, quantile)) if vals.size else np.nan


def independence_decision(residual: pd.DataFrame, consistency: pd.DataFrame, pseudo: pd.DataFrame, pseudo_cons: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = []
    for (mode, domain), sub in residual.groupby(["mode", "domain"]):
        psub = pseudo[(pseudo["mode"] == mode) & (pseudo["domain"] == domain)]
        csub = consistency[(consistency["mode"] == mode) & (consistency["domain"] == domain)]
        pcsub = pseudo_cons[(pseudo_cons["mode"] == mode) & (pseudo_cons["domain"] == domain)]
        rmed = q(sub["residual_fraction"], 0.5); rnmed = q(sub["residual_norm"], 0.5)
        pqf = q(psub["pseudo_residual_fraction"], settings.null_quantile); pqn = q(psub["pseudo_residual_norm"], settings.null_quantile)
        cmed = q(csub["pattern_corr_to_loo_residual_template"], 0.5); pcq = q(pcsub["pseudo_pattern_corr"], settings.null_quantile)
        above = np.isfinite(rmed) and np.isfinite(pqf) and np.isfinite(rnmed) and np.isfinite(pqn) and rmed > pqf and rnmed > pqn
        cons_above = np.isfinite(cmed) and np.isfinite(pcq) and cmed > pcq
        if not above:
            status = "not_independent"; route = "stop_H35_single_point_line; use H18-H35 package only if H remains relevant"
        elif above and not cons_above:
            status = "weak_or_intermittent_residual" if rmed >= settings.residual_fraction_moderate else "weak_residual_only"
            route = "do_not_prioritize_H35_single_point; keep only as secondary sensitivity target"
        elif rmed >= settings.residual_fraction_strong:
            status = "stable_independent_residual"; route = "H35 can enter dedicated cross-object follow-up"
        elif rmed >= settings.residual_fraction_moderate:
            status = "moderate_independent_residual"; route = "H35 may enter cross-object audit with caution"
        else:
            status = "weak_or_intermittent_residual"; route = "H35 is secondary; do not center next step on it"
        rows.append({"mode": mode, "domain": domain, "h35_residual_fraction_median": rmed, "pseudo_residual_fraction_q90": pqf, "h35_residual_norm_median": rnmed, "pseudo_residual_norm_q90": pqn, "h35_residual_pattern_corr_median": cmed, "pseudo_pattern_corr_q90": pcq, "residual_above_pseudo_null": bool(above), "consistency_above_pseudo_null": bool(cons_above), "h35_independence_status": status, "route_decision": route})
    return pd.DataFrame(rows)


def lin_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) < 1e-12:
        return np.nan
    return float(np.cov(x[mask], y[mask], ddof=0)[0, 1] / np.var(x[mask]))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y); xx = x[mask]; yy = y[mask]
    if len(xx) < 5:
        return np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xx), len(xx)); vals.append(lin_slope(xx[idx], yy[idx]))
    vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def perm_p(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    mask = np.isfinite(x) & np.isfinite(y); xx = x[mask]; yy = y[mask]
    if len(xx) < 5:
        return np.nan
    obs = abs(spearman_corr(xx, yy)); cnt = 0
    for _ in range(n_perm):
        if abs(spearman_corr(xx, rng.permutation(yy))) >= obs:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def loo_sign_stability(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y); xx = x[mask]; yy = y[mask]
    n = len(xx)
    if n < 5:
        return np.nan
    full = lin_slope(xx, yy)
    if not np.isfinite(full) or abs(full) < 1e-12:
        return np.nan
    signs = []
    for i in range(n):
        keep = np.ones(n, dtype=bool); keep[i] = False
        s = lin_slope(xx[keep], yy[keep])
        if np.isfinite(s):
            signs.append(np.sign(s) == np.sign(full))
    return float(np.mean(signs)) if signs else np.nan


def predictive_support(events: pd.DataFrame, residual: pd.DataFrame, indep: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (mode, domain), sub in residual.groupby(["mode", "domain"]):
        dec = indep[(indep["mode"] == mode) & (indep["domain"] == domain)]
        gate = dec["h35_independence_status"].iloc[0] if not dec.empty else "unavailable"
        h18 = events[(events["mode"] == mode) & (events["domain"] == domain) & (events["event_id"] == "H18")][["year", "diff_norm"]].rename(columns={"diff_norm": "h18_strength"})
        h35 = sub[["year", "residual_norm"]].rename(columns={"residual_norm": "h35_residual_strength"})
        merged = h18.merge(h35, on="year", how="inner")
        x = merged["h18_strength"].to_numpy(float); y = merged["h35_residual_strength"].to_numpy(float)
        pear = corr_finite(x, y); spear = spearman_corr(x, y); slope = lin_slope(x, y); lo, hi = bootstrap_ci(x, y, rng, settings.n_bootstrap); p = perm_p(x, y, rng, settings.n_permutation); loo = loo_sign_stability(x, y)
        if gate == "not_independent":
            status = "not_tested_because_H35_not_independent"
        elif np.isfinite(p) and p < 0.05 and np.isfinite(loo) and loo >= 0.80 and np.isfinite(slope) and slope > 0:
            status = "stable_predictive_support"
        elif np.isfinite(p) and p < 0.15 and np.isfinite(slope) and slope > 0:
            status = "weak_predictive_support"
        else:
            status = "no_predictive_support"
        rows.append({"mode": mode, "domain": domain, "n_years": len(merged), "h35_independence_status_gate": gate, "pearson_r": pear, "spearman_r": spear, "slope": slope, "bootstrap_slope_ci_low": lo, "bootstrap_slope_ci_high": hi, "permutation_p_spearman_abs": p, "loo_slope_sign_stability": loo, "h18_precursor_to_h35_status": status})
    return pd.DataFrame(rows)


def final_route(indep: pd.DataFrame, pred: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = []
    cand = indep[indep["mode"].isin(settings.primary_modes_for_decision)].copy()
    if cand.empty: cand = indep.copy()
    pref = cand[cand["domain"] == "object_domain_spatial"]
    if pref.empty: pref = cand
    priority = {"stable_independent_residual":4, "moderate_independent_residual":3, "weak_or_intermittent_residual":2, "weak_residual_only":1, "not_independent":0}
    if pref.empty:
        hstatus = "unresolved_no_decision_rows"; ev = "No independence decision rows."; route = "do_not_interpret"
    else:
        pref = pref.assign(_prio=pref["h35_independence_status"].map(priority).fillna(-1))
        r = pref.sort_values("_prio", ascending=False).iloc[0]
        hstatus = r["h35_independence_status"]
        ev = f"mode={r['mode']}, domain={r['domain']}, residual_fraction_median={r['h35_residual_fraction_median']:.3g}, pseudo_q90={r['pseudo_residual_fraction_q90']:.3g}, consistency_median={r['h35_residual_pattern_corr_median']:.3g}"
        route = r["route_decision"]
    rows.append({"decision_item":"H35 single-point line", "status":hstatus, "evidence":ev, "route_implication":route})
    psub = pred[pred["domain"] == "object_domain_spatial"] if not pred.empty else pd.DataFrame()
    if psub.empty and not pred.empty: psub = pred
    if psub.empty:
        rows.append({"decision_item":"H18 precursor to H35 residual", "status":"not_available", "evidence":"No predictive rows.", "route_implication":"do_not_interpret"})
    else:
        r = psub.iloc[0]
        status = r["h18_precursor_to_h35_status"]
        ev = f"mode={r['mode']}, domain={r['domain']}, spearman_r={r['spearman_r']:.3g}, perm_p={r['permutation_p_spearman_abs']:.3g}, loo_sign={r['loo_slope_sign_stability']:.3g}"
        if str(status).startswith("not_tested"):
            implication = "invalid_question_under_current_evidence"
        elif status == "stable_predictive_support":
            implication = "H18 may be tested as predictive precondition for H35 residual, not causal proof"
        elif status == "weak_predictive_support":
            implication = "weak clue only; do not foreground"
        else:
            implication = "do_not_call_H18_precursor_to_H35"
        rows.append({"decision_item":"H18 precursor to H35 residual", "status":status, "evidence":ev, "route_implication":implication})
    return pd.DataFrame(rows)


def plot_decomp(arrays: dict, lat: np.ndarray, lon: np.ndarray, out: Path):
    mode = next((m for m in ["local_background_removed", "anomaly", "raw"] if (m, "object_domain_spatial") in arrays), None)
    if mode is None: return
    a = arrays[(mode, "object_domain_spatial")]
    h18 = safe_nanmean(a["h18"], axis=0); h35 = safe_nanmean(a["h35"], axis=0); fit = safe_nanmean(a["fitted"], axis=0); res = safe_nanmean(a["residual"], axis=0)
    vals = [np.nanmax(np.abs(x)) for x in [h18,h35,fit,res] if np.isfinite(x).any()]
    vmax = max(vals) if vals else 1.0
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, data, title in zip(axs, [h18,h35,fit,res], ["H18 template","H35 diff","H18-like fitted","H35 residual"]):
        im = ax.pcolormesh(lon, lat, data, shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(title); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.85)
    fig.suptitle(f"H35 residual decomposition ({mode}, object domain)")
    fig.savefig(out / "figures" / "h35_residual_decomposition_object_domain_v10_7_d.png", dpi=180)
    plt.close(fig)
    # profile
    ap = arrays.get((mode, "profile"))
    if ap is not None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for key, label in [("h18","H18 template"),("h35","H35 diff"),("fitted","H18-like fitted"),("residual","H35 residual")]:
            ax.plot(lat, safe_nanmean(ap[key], axis=0), marker="o", label=label)
        ax.axhline(0, linewidth=0.8); ax.set_xlabel("lat"); ax.set_ylabel("profile diff"); ax.legend(); ax.set_title(f"H35 residual profile decomposition ({mode})")
        fig.tight_layout(); fig.savefig(out / "figures" / "h35_residual_decomposition_profile_v10_7_d.png", dpi=180); plt.close(fig)


def plot_null(indep: pd.DataFrame, pseudo: pd.DataFrame, out: Path):
    if indep.empty or pseudo.empty: return
    rowdf = indep[indep["domain"] == "object_domain_spatial"]
    if rowdf.empty: rowdf = indep
    row = rowdf.iloc[0]; mode = row["mode"]; domain = row["domain"]
    vals = pseudo[(pseudo["mode"] == mode) & (pseudo["domain"] == domain)]["pseudo_residual_fraction"].dropna().to_numpy(float)
    if vals.size == 0: return
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.hist(vals, bins=min(20, max(5, vals.size//3)), alpha=0.75)
    ax.axvline(float(row["h35_residual_fraction_median"]), linewidth=2, label="H35 median")
    ax.axvline(float(row["pseudo_residual_fraction_q90"]), linestyle="--", linewidth=2, label="pseudo q90")
    ax.set_xlabel("residual fraction"); ax.set_ylabel("count"); ax.set_title(f"H35 residual vs pseudo-null ({mode}, {domain})"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "figures" / "h35_residual_against_pseudo_null_v10_7_d.png", dpi=180); plt.close(fig)


def write_summary(out: Path, route: pd.DataFrame, indep: pd.DataFrame, pred: pd.DataFrame, input_status: dict):
    lines = ["# V10.7_d H35 residual/anomaly independence audit", "", "## Method boundary", "- Route-decision experiment for H35 independence.", "- Does not test H35 -> W045/P/V/Je/Jw.", "- Does not infer causality.", "- H18 precursor-to-H35 is interpreted only if H35 residual independence passes the gate.", "", "## Input status"]
    for k, v in input_status.items(): lines.append(f"- {k}: {v}")
    lines += ["", "## Route decision"]
    if route.empty: lines.append("No route decision produced.")
    else:
        for _, r in route.iterrows():
            lines += [f"### {r['decision_item']}", f"- status: `{r['status']}`", f"- evidence: {r['evidence']}", f"- route implication: {r['route_implication']}", ""]
    lines += ["## H35 independence rows"]
    if indep.empty: lines.append("No independence rows produced.")
    else:
        for _, r in indep.iterrows():
            lines.append(f"- {r['mode']} / {r['domain']}: status=`{r['h35_independence_status']}`, residual_fraction={r['h35_residual_fraction_median']:.3g}, pseudo_q90={r['pseudo_residual_fraction_q90']:.3g}, consistency={r['h35_residual_pattern_corr_median']:.3g}, pseudo_consistency_q90={r['pseudo_pattern_corr_q90']:.3g}")
    lines += ["", "## Forbidden interpretations", "- Do not call H35 independent unless the route decision supports it.", "- Do not ask H18 -> H35 if H35 residual independence is rejected.", "- Do not treat this as H35 -> W045 evidence; cross-object audit is separate."]
    (out / "summary_h35_residual_independence_v10_7_d.md").write_text("\n".join(lines), encoding="utf-8")


def run_h35_residual_independence_v10_7_d(project_root: Path) -> dict[str, Any]:
    settings = Settings().with_project_root(Path(project_root))
    out = settings.output_root(); clean_output_root(out)
    loaded = load_field(settings)
    obj_field, obj_lat, obj_lon = subset_domain(loaded["field"], loaded["lat"], loaded["lon"], settings.object_lat_range, settings.object_lon_range)
    modes, mode_status = build_modes(obj_field)
    diffs = compute_diffs(modes, settings.events)
    erows = event_rows(diffs, loaded["years"])
    residual, consistency, arrays = residuals_for_h35(diffs, loaded["years"])
    p_diffs = compute_diffs(modes, settings.pseudo_events)
    # pseudo_diffs uses different event ids; H18 template still from main diffs. Build null manually.
    pseudo_rows, pseudo_cons = pseudo_null(modes, diffs, settings, loaded["years"])
    indep = independence_decision(residual, consistency, pseudo_rows, pseudo_cons, settings)
    rng = np.random.default_rng(settings.random_seed)
    pred = predictive_support(erows, residual, indep, settings, rng)
    route = final_route(indep, pred, settings)

    tables = out / "tables"
    write_dataframe(erows, tables / "h35_event_diff_by_year_v10_7_d.csv")
    write_dataframe(residual, tables / "h35_projection_residual_by_year_v10_7_d.csv")
    write_dataframe(consistency, tables / "h35_residual_pattern_consistency_by_year_v10_7_d.csv")
    write_dataframe(pseudo_rows, tables / "h35_pseudo_event_null_v10_7_d.csv")
    write_dataframe(pseudo_cons, tables / "h35_pseudo_event_pattern_consistency_null_v10_7_d.csv")
    write_dataframe(indep, tables / "h35_independence_decision_v10_7_d.csv")
    write_dataframe(pred, tables / "h18_predicts_h35_residual_v10_7_d.csv")
    write_dataframe(route, tables / "h35_route_decision_v10_7_d.csv")
    plot_decomp(arrays, obj_lat, obj_lon, out); plot_null(indep, pseudo_rows, out)

    input_status = {"smoothed_fields_path": str(loaded["source_path"]), "field_key": loaded["field_key"], "original_shape": loaded["original_shape"], "object_domain_shape_year_day_lat_lon": tuple(obj_field.shape), "year_dimension": "detected" if (not loaded["added_year_axis"] and obj_field.shape[0] >= 2) else "not_detected_or_single_year", "object_domain_lat": list(settings.object_lat_range), "object_domain_lon": list(settings.object_lon_range), **mode_status}
    write_summary(out, route, indep, pred, input_status)
    meta = {"version": settings.version, "task": "H35 residual/anomaly independence audit with conditional H18 predictive test", "created_at_utc": now_utc(), "project_root": str(project_root), "output_root": str(out), "settings": settings.to_dict(), "input_status": input_status, "final_route_decision": route.to_dict(orient="records"), "method_boundary": ["not causal inference", "not H35 -> W045 cross-object audit", "not detector rerun", "H18 precursor-to-H35 only interpreted if H35 residual independence passes gate"]}
    write_json(meta, out / "run_meta" / "run_meta_v10_7_d.json")
    return meta
