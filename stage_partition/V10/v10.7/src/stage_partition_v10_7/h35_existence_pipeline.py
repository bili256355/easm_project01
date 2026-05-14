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
class EventWindow:
    event_id: str
    pre_days: tuple[int, int]
    post_days: tuple[int, int]
    background_range: tuple[int, int]
    target_day: int


@dataclass
class Settings:
    project_root: Path = Path(r"D:\easm_project01")
    version: str = "v10.7_e"
    output_tag: str = "h35_existence_attribution_v10_7_e"
    smoothed_env_var: str = "V10_7_SMOOTHED_FIELDS"
    field_key_candidates: tuple[str, ...] = ("z500_smoothed", "z500", "H", "hgt500", "geopotential_500", "zg500")
    lat_key_candidates: tuple[str, ...] = ("lat", "latitude", "lats")
    lon_key_candidates: tuple[str, ...] = ("lon", "longitude", "lons")
    year_key_candidates: tuple[str, ...] = ("year", "years")
    day_key_candidates: tuple[str, ...] = ("day", "days", "day_index")
    object_lat_range: tuple[float, float] = (15.0, 35.0)
    object_lon_range: tuple[float, float] = (110.0, 140.0)
    events: tuple[EventWindow, ...] = (
        EventWindow("H18", (14, 17), (19, 22), (8, 28), 18),
        EventWindow("H35", (31, 34), (36, 39), (25, 43), 35),
        EventWindow("H45", (40, 43), (45, 48), (36, 52), 45),
        EventWindow("H57", (52, 55), (57, 60), (48, 66), 57),
    )
    pseudo_events: tuple[EventWindow, ...] = (
        EventWindow("PSEUDO_22_30", (22, 25), (27, 30), (16, 34), 26),
        EventWindow("PSEUDO_24_32", (24, 27), (29, 32), (18, 36), 28),
        EventWindow("PSEUDO_48_56", (48, 51), (53, 56), (42, 62), 52),
        EventWindow("PSEUDO_60_68", (60, 63), (65, 68), (54, 74), 64),
    )
    # Detector-score proxy: window mean after target minus before target, for attribution only.
    score_half_width: int = 10
    top_year_fraction: float = 0.20
    few_year_top_share_threshold: float = 0.50
    residual_not_special_quantile: float = 0.90
    score_drop_large_threshold: float = 0.50
    contribution_topn: int = 5
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


def score_proxy(field: np.ndarray, center_day: int, half_width: int) -> np.ndarray:
    # Returns per-year vector score using a simple window-difference proxy, not ruptures.Window internals.
    n_days = field.shape[1]
    left = np.arange(max(0, center_day - half_width), min(n_days, center_day), dtype=int)
    right = np.arange(max(0, center_day), min(n_days, center_day + half_width), dtype=int)
    if left.size == 0 or right.size == 0:
        return np.full(field.shape[0], np.nan)
    diff = safe_nanmean(field[:, right, ...], axis=1) - safe_nanmean(field[:, left, ...], axis=1)
    return np.asarray([vector_norm(diff[i]) for i in range(diff.shape[0])], dtype=float)


def remove_h18_like_from_state(field: np.ndarray, h18_diffs: np.ndarray) -> np.ndarray:
    # Per-year leave-one-out H18 template; project each daily field onto template and subtract.
    out = np.array(field, copy=True, dtype=float)
    n_year = field.shape[0]
    for i in range(n_year):
        template = safe_nanmean(np.delete(h18_diffs, i, axis=0), axis=0) if n_year > 1 else safe_nanmean(h18_diffs, axis=0)
        den = dot_finite(template, template)
        if not np.isfinite(den) or den < 1e-12:
            continue
        for t in range(field.shape[1]):
            alpha = dot_finite(field[i, t], template) / den
            out[i, t] = field[i, t] - alpha * template
    return out


def top_contribution_overlap(a: np.ndarray, b: np.ndarray, topn: int) -> float:
    aa = np.abs(np.asarray(a, dtype=float).ravel()); bb = np.abs(np.asarray(b, dtype=float).ravel())
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() == 0:
        return float("nan")
    aa = aa[mask]; bb = bb[mask]
    n = min(topn, len(aa))
    if n == 0:
        return float("nan")
    ia = set(np.argsort(aa)[-n:].tolist()); ib = set(np.argsort(bb)[-n:].tolist())
    return float(len(ia & ib) / n)


def diff_summary_rows(diffs: dict[tuple[str, str], np.ndarray], settings: Settings, years: np.ndarray) -> pd.DataFrame:
    rows = []
    for (mode, event_id), arr in diffs.items():
        for domain, data in (("object_domain_spatial", arr), ("profile", spatial_to_profile(arr))):
            clim = safe_nanmean(data, axis=0)
            rows.append({
                "mode": mode,
                "event_id": event_id,
                "domain": domain,
                "n_years": int(data.shape[0]),
                "clim_norm": vector_norm(clim),
                "clim_abs_mean": float(np.nanmean(np.abs(clim))) if np.isfinite(clim).any() else np.nan,
                "year_median_norm": float(np.nanmedian([vector_norm(data[i]) for i in range(data.shape[0])])),
                "year_mean_norm": float(np.nanmean([vector_norm(data[i]) for i in range(data.shape[0])])),
            })
    return pd.DataFrame(rows)


def feature_contribution_table(diffs: dict[tuple[str, str], np.ndarray], lat: np.ndarray, settings: Settings) -> pd.DataFrame:
    rows = []
    for mode in sorted({m for m, _ in diffs}):
        h18 = diffs.get((mode, "H18")); h35 = diffs.get((mode, "H35"))
        if h18 is None or h35 is None:
            continue
        p18 = safe_nanmean(spatial_to_profile(h18), axis=0)
        p35 = safe_nanmean(spatial_to_profile(h35), axis=0)
        total18 = np.nansum(np.abs(p18)); total35 = np.nansum(np.abs(p35))
        for idx in range(len(p18)):
            rows.append({
                "mode": mode,
                "feature_index": idx,
                "lat": float(lat[idx]) if idx < len(lat) else np.nan,
                "H18_profile_diff": float(p18[idx]),
                "H35_profile_diff": float(p35[idx]),
                "H18_contribution_fraction": float(abs(p18[idx]) / total18) if total18 > 0 else np.nan,
                "H35_contribution_fraction": float(abs(p35[idx]) / total35) if total35 > 0 else np.nan,
                "same_sign": bool(np.sign(p18[idx]) == np.sign(p35[idx])) if np.isfinite(p18[idx]) and np.isfinite(p35[idx]) else False,
            })
    return pd.DataFrame(rows)


def h18_like_residual(diffs: dict[tuple[str, str], np.ndarray], years: np.ndarray, settings: Settings) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, np.ndarray]]]:
    rows = []
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for mode in sorted({m for m, _ in diffs}):
        h18 = diffs.get((mode, "H18")); h35 = diffs.get((mode, "H35"))
        if h18 is None or h35 is None:
            continue
        for domain, a18, a35 in (("object_domain_spatial", h18, h35), ("profile", spatial_to_profile(h18), spatial_to_profile(h35))):
            fits = np.full_like(a35, np.nan); resids = np.full_like(a35, np.nan); templates = np.full_like(a18, np.nan)
            for i in range(a35.shape[0]):
                template = safe_nanmean(np.delete(a18, i, axis=0), axis=0) if a18.shape[0] > 1 else safe_nanmean(a18, axis=0)
                fit, resid, metrics = project(a35[i], template)
                fits[i] = fit; resids[i] = resid; templates[i] = template
                rows.append({"year": years[i].item() if hasattr(years[i], "item") else years[i], "mode": mode, "domain": domain, **metrics})
            arrays[(mode, domain)] = {"h18": a18, "h35": a35, "fitted": fits, "residual": resids, "templates": templates}
    return pd.DataFrame(rows), arrays


def pseudo_null(diffs: dict[tuple[str, str], np.ndarray], pseudo_diffs: dict[tuple[str, str], np.ndarray], years: np.ndarray) -> pd.DataFrame:
    rows = []
    for mode in sorted({m for m, _ in pseudo_diffs}):
        h18 = diffs.get((mode, "H18"))
        if h18 is None:
            continue
        for peid in sorted({e for m, e in pseudo_diffs if m == mode}):
            target = pseudo_diffs.get((mode, peid))
            if target is None:
                continue
            for domain, a18, atarget in (("object_domain_spatial", h18, target), ("profile", spatial_to_profile(h18), spatial_to_profile(target))):
                for i in range(atarget.shape[0]):
                    template = safe_nanmean(np.delete(a18, i, axis=0), axis=0) if a18.shape[0] > 1 else safe_nanmean(a18, axis=0)
                    _, _, metrics = project(atarget[i], template)
                    rows.append({"year": years[i].item() if hasattr(years[i], "item") else years[i], "mode": mode, "domain": domain, "pseudo_event_id": peid, "pseudo_residual_norm": metrics["residual_norm"], "pseudo_residual_fraction": metrics["residual_fraction"], "pseudo_target_norm": metrics["target_norm"]})
    return pd.DataFrame(rows)


def year_contribution(diffs: dict[tuple[str, str], np.ndarray], residual_rows: pd.DataFrame, settings: Settings, years: np.ndarray) -> pd.DataFrame:
    rows = []
    for mode in sorted({m for m, _ in diffs}):
        h35 = diffs.get((mode, "H35"))
        if h35 is None:
            continue
        for domain, data in (("object_domain_spatial", h35), ("profile", spatial_to_profile(h35))):
            scores = np.asarray([vector_norm(data[i]) for i in range(data.shape[0])], dtype=float)
            total = np.nansum(scores)
            if np.isfinite(total) and total > 0:
                order = np.argsort(scores)[::-1]
                top_n = max(1, int(np.ceil(settings.top_year_fraction * len(scores))))
                top_set = set(order[:top_n].tolist())
            else:
                top_set = set()
            rsub = residual_rows[(residual_rows["mode"] == mode) & (residual_rows["domain"] == domain)].copy()
            for i, y in enumerate(years):
                rrow = rsub[rsub["year"] == (y.item() if hasattr(y, "item") else y)]
                rows.append({
                    "year": y.item() if hasattr(y, "item") else y,
                    "mode": mode,
                    "domain": domain,
                    "H35_score": scores[i],
                    "H35_score_share": float(scores[i] / total) if total > 0 else np.nan,
                    "is_top_fraction_year": bool(i in top_set),
                    "H35_residual_norm": float(rrow["residual_norm"].iloc[0]) if not rrow.empty else np.nan,
                    "H35_residual_fraction": float(rrow["residual_fraction"].iloc[0]) if not rrow.empty else np.nan,
                })
    return pd.DataFrame(rows)


def score_ablation_audit(modes: dict[str, np.ndarray], diffs: dict[tuple[str, str], np.ndarray], settings: Settings, years: np.ndarray) -> pd.DataFrame:
    rows = []
    for mode, field in modes.items():
        h18 = diffs.get((mode, "H18"))
        if h18 is None:
            continue
        h18_removed = remove_h18_like_from_state(field, h18)
        for ev in settings.events:
            if ev.event_id != "H35":
                continue
            original = score_proxy(field, ev.target_day, settings.score_half_width)
            after_h18 = score_proxy(h18_removed, ev.target_day, settings.score_half_width)
            # Use local-background event diff norm as a background-removed score proxy.
            lbr = local_background_removed_diff(field, ev)
            lbr_score = np.asarray([vector_norm(lbr[i]) for i in range(lbr.shape[0])], dtype=float)
            # Remove H18-like from local-background diff directly.
            lbr_fit_resid = []
            for i in range(lbr.shape[0]):
                template = safe_nanmean(np.delete(h18, i, axis=0), axis=0) if h18.shape[0] > 1 else safe_nanmean(h18, axis=0)
                _, resid, _ = project(lbr[i], template)
                lbr_fit_resid.append(vector_norm(resid))
            both = np.asarray(lbr_fit_resid, dtype=float)
            for i, y in enumerate(years):
                base = original[i]
                rows.append({
                    "year": y.item() if hasattr(y, "item") else y,
                    "mode": mode,
                    "target_event": ev.event_id,
                    "target_day": ev.target_day,
                    "score_proxy_original": base,
                    "score_proxy_after_H18_like_removal": after_h18[i],
                    "score_proxy_after_background_removal": lbr_score[i],
                    "score_proxy_after_H18_and_background_removal": both[i],
                    "drop_fraction_after_H18_like_removal": float(1.0 - after_h18[i] / base) if np.isfinite(base) and base > 0 else np.nan,
                    "drop_fraction_after_background_removal": float(1.0 - lbr_score[i] / base) if np.isfinite(base) and base > 0 else np.nan,
                    "drop_fraction_after_both": float(1.0 - both[i] / base) if np.isfinite(base) and base > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def attribution_decision(residual_rows: pd.DataFrame, pseudo_rows: pd.DataFrame, year_rows: pd.DataFrame, ablation_rows: pd.DataFrame, feature_rows: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    decisions = []
    modes_priority = ["anomaly", "local_background_removed", "raw"]
    domains_priority = ["object_domain_spatial", "profile"]
    for mode in modes_priority:
        for domain in domains_priority:
            sub = residual_rows[(residual_rows["mode"] == mode) & (residual_rows["domain"] == domain)]
            psub = pseudo_rows[(pseudo_rows["mode"] == mode) & (pseudo_rows["domain"] == domain)]
            if not sub.empty and not psub.empty:
                chosen_mode, chosen_domain = mode, domain
                break
        else:
            continue
        break
    else:
        chosen_mode, chosen_domain = "unavailable", "unavailable"
    status = "unresolved"
    evidence = []
    route = "manual_review"
    if chosen_mode != "unavailable":
        sub = residual_rows[(residual_rows["mode"] == chosen_mode) & (residual_rows["domain"] == chosen_domain)]
        psub = pseudo_rows[(pseudo_rows["mode"] == chosen_mode) & (pseudo_rows["domain"] == chosen_domain)]
        rmed = float(np.nanmedian(sub["residual_fraction"]))
        pq90 = float(np.nanquantile(psub["pseudo_residual_fraction"], settings.residual_not_special_quantile))
        ab = ablation_rows[(ablation_rows["mode"] == chosen_mode)]
        drop_h18 = float(np.nanmedian(ab["drop_fraction_after_H18_like_removal"])) if not ab.empty else np.nan
        drop_bg = float(np.nanmedian(ab["drop_fraction_after_background_removal"])) if not ab.empty else np.nan
        yr = year_rows[(year_rows["mode"] == chosen_mode) & (year_rows["domain"] == chosen_domain)]
        if not yr.empty and np.isfinite(yr["H35_score_share"]).any():
            top_share = float(np.nansum(yr.loc[yr["is_top_fraction_year"], "H35_score_share"]))
        else:
            top_share = np.nan
        fsub = feature_rows[feature_rows["mode"] == chosen_mode]
        overlap = np.nan
        if not fsub.empty:
            # Use top-n overlap from profile contributions.
            h18 = fsub["H18_contribution_fraction"].to_numpy(float)
            h35 = fsub["H35_contribution_fraction"].to_numpy(float)
            overlap = top_contribution_overlap(h18, h35, settings.contribution_topn)
        evidence.append(f"primary={chosen_mode}/{chosen_domain}")
        evidence.append(f"H35 residual_fraction median={rmed:.3g}, pseudo q90={pq90:.3g}")
        evidence.append(f"median score drop after H18-like removal={drop_h18:.3g}")
        evidence.append(f"median score drop after background removal={drop_bg:.3g}")
        evidence.append(f"top {settings.top_year_fraction:.0%} year score share={top_share:.3g}")
        evidence.append(f"top feature contribution overlap={overlap:.3g}")
        if np.isfinite(top_share) and top_share >= settings.few_year_top_share_threshold:
            status = "few_year_driven"
            route = "do_not_use_climatological_H35_as_stable_event; inspect year subsets only"
        elif np.isfinite(drop_bg) and drop_bg >= settings.score_drop_large_threshold:
            status = "seasonal_background_or_local_curvature"
            route = "downgrade_H35; do not pursue single-point H35 unless anomaly-specific residual reappears"
        elif (np.isfinite(rmed) and np.isfinite(pq90) and rmed <= pq90) and (np.isfinite(drop_h18) and drop_h18 >= settings.score_drop_large_threshold):
            status = "H18_like_second_stage"
            route = "stop_H35_single_point; if H remains relevant use H18-H35 package"
        elif (np.isfinite(rmed) and np.isfinite(pq90) and rmed <= pq90):
            status = "method_score_shoulder_or_background"
            route = "H35 exists as method-level candidate but has no stable independent attribution"
        else:
            status = "unresolved_possible_independent_component"
            route = "manual_review; H35 may need targeted follow-up before cross-object use"
    decisions.append({"decision_item": "H35 existence attribution", "status": status, "evidence": "; ".join(evidence), "route_implication": route})
    decisions.append({"decision_item": "E2 multi-object component", "status": "not_tested_in_v10_7_e", "evidence": "V10.7_e is H-only. Cross-object E2 requires P/V/Je/Jw inputs.", "route_implication": "test separately only if H package remains relevant"})
    return pd.DataFrame(decisions)


def plot_ablation(ablation: pd.DataFrame, out: Path):
    if ablation.empty:
        return
    mode = "anomaly" if "anomaly" in set(ablation["mode"]) else ("local_background_removed" if "local_background_removed" in set(ablation["mode"]) else ablation["mode"].iloc[0])
    sub = ablation[ablation["mode"] == mode]
    cols = ["score_proxy_original", "score_proxy_after_H18_like_removal", "score_proxy_after_background_removal", "score_proxy_after_H18_and_background_removal"]
    vals = [float(np.nanmedian(sub[c])) for c in cols]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(np.arange(len(cols)), vals)
    ax.set_xticks(np.arange(len(cols)), ["original", "after H18-like", "after background", "after both"], rotation=25, ha="right")
    ax.set_ylabel("median H35 score proxy")
    ax.set_title(f"H35 score attribution proxy ({mode})")
    fig.tight_layout()
    fig.savefig(out / "figures" / "h35_ablation_score_audit_v10_7_e.png", dpi=180)
    plt.close(fig)


def plot_residual_decomposition(arrays: dict, lat: np.ndarray, lon: np.ndarray, out: Path):
    mode = next((m for m in ["anomaly", "local_background_removed", "raw"] if (m, "object_domain_spatial") in arrays), None)
    if mode is None:
        return
    a = arrays[(mode, "object_domain_spatial")]
    h18 = safe_nanmean(a["h18"], axis=0); h35 = safe_nanmean(a["h35"], axis=0); fit = safe_nanmean(a["fitted"], axis=0); res = safe_nanmean(a["residual"], axis=0)
    vals = [np.nanmax(np.abs(x)) for x in [h18, h35, fit, res] if np.isfinite(x).any()]
    vmax = max(vals) if vals else 1.0
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, data, title in zip(axs, [h18, h35, fit, res], ["H18 template", "H35 diff", "H18-like fitted", "H35 residual"]):
        im = ax.pcolormesh(lon, lat, data, shading="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(title); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.85)
    fig.suptitle(f"H35 existence attribution decomposition ({mode})")
    fig.savefig(out / "figures" / "h35_existence_attribution_decomposition_object_domain_v10_7_e.png", dpi=180)
    plt.close(fig)


def write_summary(out: Path, decision: pd.DataFrame, input_status: dict):
    lines = [
        "# V10.7_e H35 existence attribution audit",
        "",
        "## Method boundary",
        "- This is an attribution audit for why H35 is extracted by the H-only main-method context.",
        "- It does not test H35 -> W045/P/V/Je/Jw.",
        "- It does not prove causality or physical mechanism.",
        "- It distinguishes candidate sources: H18-like second stage, background/local curvature, few-year driven, method shoulder, or unresolved.",
        "",
        "## Input status",
    ]
    for k, v in input_status.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Route decision"]
    if decision.empty:
        lines.append("No decision produced.")
    else:
        for _, r in decision.iterrows():
            lines += [f"### {r['decision_item']}", f"- status: `{r['status']}`", f"- evidence: {r['evidence']}", f"- route implication: {r['route_implication']}", ""]
    lines += [
        "## Required reading boundary",
        "- If the decision is `H18_like_second_stage`, stop H35 single-point line and use H18-H35 package only if H remains relevant.",
        "- If the decision is `seasonal_background_or_local_curvature`, downgrade H35 from event interpretation.",
        "- If the decision is `few_year_driven`, do not use climatological H35 as a stable event; inspect year subsets.",
        "- E2 multi-object attribution is not tested here.",
    ]
    (out / "summary_h35_existence_attribution_v10_7_e.md").write_text("\n".join(lines), encoding="utf-8")


def run_h35_existence_attribution_v10_7_e(project_root: Path) -> dict[str, Any]:
    settings = Settings().with_project_root(Path(project_root))
    out = settings.output_root(); clean_output_root(out)
    loaded = load_field(settings)
    obj_field, obj_lat, obj_lon = subset_domain(loaded["field"], loaded["lat"], loaded["lon"], settings.object_lat_range, settings.object_lon_range)
    modes, mode_status = build_modes(obj_field)
    diffs = compute_diffs(modes, settings.events)
    pseudo_diffs = compute_diffs(modes, settings.pseudo_events)

    diff_summary = diff_summary_rows(diffs, settings, loaded["years"])
    feature_rows = feature_contribution_table(diffs, obj_lat, settings)
    residual_rows, arrays = h18_like_residual(diffs, loaded["years"], settings)
    pseudo_rows = pseudo_null(diffs, pseudo_diffs, loaded["years"])
    year_rows = year_contribution(diffs, residual_rows, settings, loaded["years"])
    ablation_rows = score_ablation_audit(modes, diffs, settings, loaded["years"])
    decision = attribution_decision(residual_rows, pseudo_rows, year_rows, ablation_rows, feature_rows, settings)

    tables = out / "tables"
    write_dataframe(diff_summary, tables / "h35_event_diff_summary_v10_7_e.csv")
    write_dataframe(feature_rows, tables / "h35_score_feature_contribution_v10_7_e.csv")
    write_dataframe(residual_rows, tables / "h35_h18_like_projection_residual_v10_7_e.csv")
    write_dataframe(pseudo_rows, tables / "h35_attribution_pseudo_null_v10_7_e.csv")
    write_dataframe(year_rows, tables / "h35_year_contribution_v10_7_e.csv")
    write_dataframe(ablation_rows, tables / "h35_ablation_score_audit_v10_7_e.csv")
    write_dataframe(decision, tables / "h35_existence_attribution_decision_v10_7_e.csv")

    plot_ablation(ablation_rows, out)
    plot_residual_decomposition(arrays, obj_lat, obj_lon, out)

    input_status = {
        "smoothed_fields_path": str(loaded["source_path"]),
        "field_key": loaded["field_key"],
        "original_shape": loaded["original_shape"],
        "object_domain_shape_year_day_lat_lon": tuple(obj_field.shape),
        "object_domain_lat": list(settings.object_lat_range),
        "object_domain_lon": list(settings.object_lon_range),
        "year_dimension": "detected" if (not loaded["added_year_axis"] and obj_field.shape[0] >= 2) else "not_detected_or_single_year",
        **mode_status,
    }
    write_summary(out, decision, input_status)
    meta = {
        "version": settings.version,
        "task": "H35 existence attribution audit",
        "created_at_utc": now_utc(),
        "project_root": str(project_root),
        "output_root": str(out),
        "settings": settings.to_dict(),
        "input_status": input_status,
        "final_attribution_decision": decision.to_dict(orient="records"),
        "method_boundary": [
            "not causal inference",
            "not H35 -> W045 cross-object audit",
            "not detector rerun",
            "explains why H35 exists as an H-only candidate by testing H18-like, background, year-concentration, and method-shoulder evidence",
        ],
    }
    write_json(meta, out / "run_meta" / "run_meta_v10_7_e.json")
    return meta
