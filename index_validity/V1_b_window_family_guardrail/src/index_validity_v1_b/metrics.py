from __future__ import annotations

import hashlib
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .data_io import samples_from_year_day, subset_field
from .metadata import INDEX_METADATA, REGIONS, VARIABLE_ORDER
from .settings import IndexValidityV1BSettings


def _lat_weights(lat: np.ndarray, n_lon: int, mode: str = "cos") -> np.ndarray:
    if mode == "sqrt_cos":
        w_lat = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    else:
        w_lat = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    return np.repeat(w_lat[:, None], n_lon, axis=1)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return np.nan
    xs = x[m]
    ys = y[m]
    if np.nanstd(xs) <= 0 or np.nanstd(ys) <= 0:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def _weighted_mean_map(arr: np.ndarray, lat: np.ndarray) -> float:
    if arr.size == 0:
        return np.nan
    w = _lat_weights(lat, arr.shape[1])
    m = np.isfinite(arr) & np.isfinite(w)
    if not np.any(m):
        return np.nan
    return float(np.nansum(arr[m] * w[m]) / np.nansum(w[m]))


def _band_mean(diff: np.ndarray, lat: np.ndarray, lower: float, upper: float) -> float:
    mask = (lat >= lower) & (lat <= upper)
    if not np.any(mask):
        return np.nan
    return _weighted_mean_map(diff[mask, :], lat[mask])


def _expected_contrast(index_name: str, diff: np.ndarray, lat: np.ndarray) -> Tuple[float, str, bool, str]:
    meta = INDEX_METADATA[index_name]
    etype = str(meta.get("expected_type", "generic_pattern"))
    note = ""
    contrast = np.nan
    sign_pass = True
    main = _band_mean(diff, lat, 24, 35)
    south = _band_mean(diff, lat, 18, 24)
    north = _band_mean(diff, lat, 35, 45)
    far_south = _band_mean(diff, lat, 10, 18)
    whole = _weighted_mean_map(diff, lat)
    if etype in {"main_gt_other", "main_gt_south"}:
        contrast = main - south
        note = "expected main-band positive relative to south-band"
        sign_pass = bool(np.isfinite(contrast) and contrast > 0)
    elif etype == "south_gt_main":
        contrast = south - main
        note = "expected south-band positive relative to main-band"
        sign_pass = bool(np.isfinite(contrast) and contrast > 0)
    elif etype == "north_gt_main":
        contrast = north - main
        note = "expected north-band positive relative to main-band"
        sign_pass = bool(np.isfinite(contrast) and contrast > 0)
    elif etype in {"north_gt_south", "north_gt_south_v"}:
        contrast = north - np.nanmean([south, far_south])
        note = "expected northern part positive relative to southern part"
        sign_pass = bool(np.isfinite(contrast) and contrast > 0)
    elif etype == "domain_mean_positive":
        contrast = whole
        note = "expected domain-mean high-low composite to be positive"
        sign_pass = bool(np.isfinite(contrast) and contrast > 0)
    elif etype == "spread_like":
        edge = np.nanmean([south, north])
        contrast = abs(edge - main)
        note = "spread-like index; use absolute edge-vs-center contrast"
        sign_pass = bool(np.isfinite(contrast) and contrast > 0)
    else:
        contrast = float(np.nanstd(diff)) if np.any(np.isfinite(diff)) else np.nan
        note = "generic/boundary-sensitive index; use composite amplitude, not sign"
        sign_pass = bool(np.isfinite(contrast) and contrast > 0)
    return float(contrast) if np.isfinite(contrast) else np.nan, etype, sign_pass, note


def _rows_signature(rows: pd.DataFrame) -> str:
    if len(rows) == 0:
        return "empty"
    arr = rows[["year", "day"]].to_numpy(dtype=np.int32, copy=True)
    return hashlib.blake2b(arr.tobytes(), digest_size=12).hexdigest()


class _MetricCache:
    def __init__(self, fields: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray, years: np.ndarray):
        self.fields = fields
        self.lat = lat
        self.lon = lon
        self.years = years
        self.subfield_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.sample_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self.eof_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}

    def subfield(self, family: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if family not in self.subfield_cache:
            region = REGIONS[family]
            field_name = str(region["field"])
            self.subfield_cache[family] = subset_field(
                self.fields[field_name], self.lat, self.lon, region["lat_range"], region["lon_range"]
            )
        return self.subfield_cache[family]

    def samples(self, family: str, rows: pd.DataFrame) -> np.ndarray:
        sig = _rows_signature(rows)
        key = (family, sig)
        if key not in self.sample_cache:
            field, _, _ = self.subfield(family)
            self.sample_cache[key] = samples_from_year_day(field, self.years, rows[["year", "day"]])
        return self.sample_cache[key]

    def eof_scores(self, family: str, rows: pd.DataFrame, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
        sig = _rows_signature(rows)
        key = (f"{family}:{sig}", int(n_modes))
        if key not in self.eof_cache:
            samples = self.samples(family, rows)
            _, sublat, sublon = self.subfield(family)
            self.eof_cache[key] = _eof_scores(samples, sublat, sublon, n_modes=n_modes)
        return self.eof_cache[key]


def _field_r2(field_samples: np.ndarray, index_values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> float:
    X = np.asarray(index_values, dtype=np.float64)
    Y = field_samples.reshape(field_samples.shape[0], -1).astype(np.float64, copy=False)
    mx = np.isfinite(X)
    if int(mx.sum()) < 10 or np.nanstd(X[mx]) <= 0:
        return np.nan
    valid = np.isfinite(Y) & mx[:, None]
    n = valid.sum(axis=0).astype(np.float64)
    enough = n >= 10
    if not np.any(enough):
        return np.nan
    X2d = np.broadcast_to(X[:, None], Y.shape)
    Y0 = np.where(valid, Y, 0.0)
    X0 = np.where(valid, X2d, 0.0)
    sumx = X0.sum(axis=0); sumy = Y0.sum(axis=0)
    sumx2 = (X0 * X0).sum(axis=0); sumy2 = (Y0 * Y0).sum(axis=0)
    sumxy = (X0 * Y0).sum(axis=0)
    safe_n = np.where(n > 0, n, np.nan)
    cov = sumxy - (sumx * sumy / safe_n)
    varx = sumx2 - (sumx * sumx / safe_n)
    vary = sumy2 - (sumy * sumy / safe_n)
    denom = np.sqrt(varx * vary)
    r = np.full(Y.shape[1], np.nan, dtype=np.float64)
    ok = enough & np.isfinite(denom) & (denom > 0)
    r[ok] = cov[ok] / denom[ok]
    r2_grid = (r * r).reshape(len(lat), len(lon))
    w = _lat_weights(lat, len(lon))
    m = np.isfinite(r2_grid) & np.isfinite(w)
    if not np.any(m):
        return np.nan
    return float(np.nansum(r2_grid[m] * w[m]) / np.nansum(w[m]))


def _eof_scores(field_samples: np.ndarray, lat: np.ndarray, lon: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    X = field_samples.reshape(field_samples.shape[0], -1).astype(np.float64, copy=False)
    finite_col = np.isfinite(X).mean(axis=0) >= 0.70
    if int(finite_col.sum()) < 3 or X.shape[0] < 8:
        return np.empty((X.shape[0], 0)), np.empty((0,))
    Xg = X[:, finite_col]
    col_mean = np.nanmean(Xg, axis=0); col_mean[~np.isfinite(col_mean)] = 0.0
    Xc = np.where(np.isfinite(Xg - col_mean[None, :]), Xg - col_mean[None, :], 0.0)
    full_w = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    full_w = np.repeat(full_w[:, None], len(lon), axis=1).reshape(-1)[finite_col]
    full_w = np.where(np.isfinite(full_w) & (full_w > 0), full_w, 1.0)
    try:
        u, s, _ = np.linalg.svd(Xc * full_w[None, :], full_matrices=False)
    except np.linalg.LinAlgError:
        return np.empty((X.shape[0], 0)), np.empty((0,))
    k = int(min(n_modes, len(s)))
    if k <= 0:
        return np.empty((X.shape[0], 0)), np.empty((0,))
    scores = u[:, :k] * s[:k][None, :]
    scores = (scores - np.nanmean(scores, axis=0, keepdims=True)) / np.where(
        np.nanstd(scores, axis=0, keepdims=True) > 0, np.nanstd(scores, axis=0, keepdims=True), 1.0
    )
    evr = (s[:k] ** 2) / np.sum(s ** 2) if np.sum(s ** 2) > 0 else np.full(k, np.nan)
    return scores, evr


def _eof_alignment_cached(cache: _MetricCache, family: str, rows: pd.DataFrame, index_values: np.ndarray, n_modes: int) -> Tuple[float, int, float, str]:
    scores, evr = cache.eof_scores(family, rows, n_modes=n_modes)
    if scores.shape[1] == 0:
        return np.nan, -1, np.nan, "eof_failed"
    corrs = np.asarray([_safe_corr(index_values, scores[:, k]) for k in range(scores.shape[1])], dtype=np.float64)
    if not np.any(np.isfinite(corrs)):
        return np.nan, -1, float(np.nansum(evr)) if evr.size else np.nan, "eof_corr_failed"
    best = int(np.nanargmax(np.abs(corrs)))
    return float(abs(corrs[best])), best + 1, float(np.nansum(evr)), "ok"


def _year_sum_count(field: np.ndarray, years: np.ndarray, rows: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, int]]:
    out: Dict[int, Tuple[np.ndarray, int]] = {}
    for y, sub in rows.groupby("year", sort=False):
        samples = samples_from_year_day(field, years, sub[["year", "day"]])
        if samples.shape[0] == 0:
            continue
        out[int(y)] = (np.nansum(samples, axis=0), int(samples.shape[0]))
    return out


def _sum_count_mean(sum_count: Dict[int, Tuple[np.ndarray, int]], sample_years: np.ndarray) -> np.ndarray | None:
    total = None; count = 0
    for y in sample_years:
        item = sum_count.get(int(y))
        if item is None:
            continue
        s, c = item
        total = s.copy() if total is None else total + s
        count += int(c)
    if total is None or count <= 0:
        return None
    return total / float(count)


def _bootstrap_composite_stability_fast(field: np.ndarray, years: np.ndarray, high_rows: pd.DataFrame, low_rows: pd.DataFrame, original_diff: np.ndarray, rng: np.random.Generator, n_reps: int) -> Tuple[float, float, int]:
    year_values = np.asarray(sorted(set(high_rows["year"]).union(set(low_rows["year"]))), dtype=int)
    if year_values.size < 5 or n_reps <= 0:
        return np.nan, np.nan, 0
    orig_vec = original_diff.reshape(-1)
    high_sc = _year_sum_count(field, years, high_rows)
    low_sc = _year_sum_count(field, years, low_rows)
    pattern_corrs = []; sign_passes = []
    for _ in range(n_reps):
        sample_years = rng.choice(year_values, size=year_values.size, replace=True)
        high_mean = _sum_count_mean(high_sc, sample_years)
        low_mean = _sum_count_mean(low_sc, sample_years)
        if high_mean is None or low_mean is None:
            continue
        diff = high_mean - low_mean
        r = _safe_corr(orig_vec, diff.reshape(-1))
        if np.isfinite(r):
            pattern_corrs.append(r)
        sign_passes.append(bool(np.isfinite(r) and r > 0.20))
    if not pattern_corrs:
        return np.nan, np.nan, 0
    return float(np.nanmedian(pattern_corrs)), float(np.nanmean(sign_passes)) if sign_passes else np.nan, len(pattern_corrs)


def _component_scores(expected_z: float, sign_pass: bool, effect_size: float, field_r2: float, eof_align: float, boot_corr: float) -> Dict[str, float]:
    expected_score = 0.0
    if np.isfinite(expected_z):
        expected_score = min(abs(expected_z) / 1.0, 1.0)
        if not sign_pass:
            expected_score *= 0.25
    amplitude_score = min(effect_size / 0.60, 1.0) if np.isfinite(effect_size) else 0.0
    composite_score = max(expected_score, 0.7 * amplitude_score)
    r2_score = min(field_r2 / 0.12, 1.0) if np.isfinite(field_r2) else 0.0
    eof_score = min(eof_align / 0.55, 1.0) if np.isfinite(eof_align) else 0.0
    boot_score = max(0.0, min((boot_corr + 0.1) / 0.8, 1.0)) if np.isfinite(boot_corr) else 0.0
    overall = 0.35 * composite_score + 0.20 * r2_score + 0.25 * eof_score + 0.20 * boot_score
    return {"composite_score_0_1": float(composite_score), "field_r2_score_0_1": float(r2_score), "eof_alignment_score_0_1": float(eof_score), "bootstrap_score_0_1": float(boot_score), "overall_score": float(overall)}


def _tier(overall: float, scores: Dict[str, float], settings: IndexValidityV1BSettings) -> str:
    supporting = sum(float(scores[k]) >= 0.55 for k in ["composite_score_0_1", "field_r2_score_0_1", "eof_alignment_score_0_1", "bootstrap_score_0_1"])
    if overall >= settings.tier_strong_threshold and supporting >= 3:
        return "strong"
    if overall >= settings.tier_moderate_threshold and supporting >= 2:
        return "moderate"
    if overall >= settings.tier_weak_threshold and supporting >= 1:
        return "weak_but_usable"
    if overall >= 0.25:
        return "high_risk"
    return "not_supported"


def compute_index_window_metrics(index_df: pd.DataFrame, fields: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray, years: np.ndarray, settings: IndexValidityV1BSettings) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, object]], pd.DataFrame]:
    rng = np.random.default_rng(settings.random_seed)
    rows = []; timing_rows = []
    figure_payloads: Dict[Tuple[str, str], Dict[str, object]] = {}
    cache = _MetricCache(fields, lat, lon, years)
    for window, (day0, day1) in settings.windows.items():
        t_window = time.time()
        win_df = index_df[(index_df["day"] >= day0) & (index_df["day"] <= day1)].copy()
        for name in VARIABLE_ORDER:
            t_index = time.time()
            meta = INDEX_METADATA[name]
            family = str(meta["family"]); field_name = str(meta["field"])
            field, sublat, sublon = cache.subfield(family)
            vals = win_df[["year", "day", name]].dropna().copy()
            vals = vals[np.isfinite(vals[name].to_numpy(dtype=float))]
            n_total = int(len(vals))
            if n_total < 20:
                rows.append({"window": window, "family": family, "index_name": name, "representativeness_tier": "not_supported", "risk_flag": "too_few_samples", "n_samples": n_total})
                timing_rows.append({"task": "index_metrics", "window": window, "index_name": name, "seconds": round(time.time() - t_index, 6)})
                continue
            high_thr = float(vals[name].quantile(settings.high_quantile)); low_thr = float(vals[name].quantile(settings.low_quantile))
            high_rows = vals[vals[name] >= high_thr][["year", "day", name]].copy()
            low_rows = vals[vals[name] <= low_thr][["year", "day", name]].copy()
            high_samples = cache.samples(family, high_rows[["year", "day"]])
            low_samples = cache.samples(family, low_rows[["year", "day"]])
            all_samples = cache.samples(family, vals[["year", "day"]])
            if high_samples.shape[0] < 5 or low_samples.shape[0] < 5 or all_samples.shape[0] < 20:
                rows.append({"window": window, "family": family, "index_name": name, "representativeness_tier": "not_supported", "risk_flag": "too_few_composite_samples", "n_samples": n_total})
                timing_rows.append({"task": "index_metrics", "window": window, "index_name": name, "seconds": round(time.time() - t_index, 6)})
                continue
            high_comp = np.nanmean(high_samples, axis=0); low_comp = np.nanmean(low_samples, axis=0); diff = high_comp - low_comp
            temporal_std = float(np.nanmean(np.nanstd(all_samples.reshape(all_samples.shape[0], -1), axis=0)))
            diff_std = float(np.nanstd(diff)) if np.any(np.isfinite(diff)) else np.nan
            effect_size = diff_std / temporal_std if np.isfinite(diff_std) and np.isfinite(temporal_std) and temporal_std > 0 else np.nan
            expected_contrast, expected_type, sign_pass, expected_note = _expected_contrast(name, diff, sublat)
            denom = diff_std if np.isfinite(diff_std) and diff_std > 0 else temporal_std
            expected_z = expected_contrast / denom if np.isfinite(expected_contrast) and np.isfinite(denom) and denom > 0 else np.nan
            r2 = _field_r2(all_samples, vals[name].to_numpy(dtype=float), sublat, sublon)
            eof_align, best_pc, eof_cumvar, eof_status = _eof_alignment_cached(cache, family, vals[["year", "day"]], vals[name].to_numpy(dtype=float), settings.eof_n_modes)
            boot_corr, boot_pass, boot_n = _bootstrap_composite_stability_fast(field, years, high_rows, low_rows, diff, rng, settings.bootstrap_year_reps)
            scores = _component_scores(expected_z, sign_pass, effect_size, r2, eof_align, boot_corr)
            tier = _tier(scores["overall_score"], scores, settings)
            risk_flag = {"strong": "low", "moderate": "low", "weak_but_usable": "usable_but_sensitive", "high_risk": "high_risk", "not_supported": "not_supported"}[tier]
            rows.append({"window": window, "family": family, "index_name": name, "field_name": field_name, "kind": meta.get("kind", ""), "expected_meaning": meta.get("expected_meaning", ""), "expected_type": expected_type, "expected_note": expected_note, "n_samples": n_total, "n_high_samples": int(high_samples.shape[0]), "n_low_samples": int(low_samples.shape[0]), "high_threshold": high_thr, "low_threshold": low_thr, "expected_contrast": expected_contrast, "expected_contrast_z": expected_z, "expected_sign_pass": bool(sign_pass), "composite_effect_size": effect_size, "field_r2_weighted": r2, "eof_alignment_max_abs_corr": eof_align, "eof_alignment_best_pc": int(best_pc), "eof_cumulative_variance_topk": eof_cumvar, "eof_status": eof_status, "bootstrap_pattern_corr_median": boot_corr, "bootstrap_pattern_positive_rate": boot_pass, "bootstrap_valid_reps": int(boot_n), **scores, "representativeness_tier": tier, "risk_flag": risk_flag})
            figure_payloads[(window, name)] = {"window": window, "index_name": name, "family": family, "field_name": field_name, "lat": sublat, "lon": sublon, "high": high_comp, "low": low_comp, "diff": diff, "tier": tier, "overall_score": scores["overall_score"], "band_lines": meta.get("band_lines", []), "expected_meaning": meta.get("expected_meaning", "")}
            timing_rows.append({"task": "index_metrics", "window": window, "index_name": name, "seconds": round(time.time() - t_index, 6)})
        timing_rows.append({"task": "window_metrics_total", "window": window, "index_name": "__all__", "seconds": round(time.time() - t_window, 6)})
    timing_rows.extend([
        {"task": "cache_stats", "window": "__all__", "index_name": "subfield_cache_entries", "seconds": len(cache.subfield_cache)},
        {"task": "cache_stats", "window": "__all__", "index_name": "sample_cache_entries", "seconds": len(cache.sample_cache)},
        {"task": "cache_stats", "window": "__all__", "index_name": "eof_cache_entries", "seconds": len(cache.eof_cache)},
    ])
    return pd.DataFrame(rows), figure_payloads, pd.DataFrame(timing_rows)



# -----------------------------
# Family-level joint coverage
# -----------------------------

def _prepare_design_matrix(X: np.ndarray, mean_: np.ndarray | None = None, std_: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize X and prepend intercept. Missing values are filled with train means."""
    X = np.asarray(X, dtype=np.float64)
    if mean_ is None:
        mean_ = np.nanmean(X, axis=0)
        mean_ = np.where(np.isfinite(mean_), mean_, 0.0)
    Xf = np.where(np.isfinite(X), X, mean_[None, :])
    if std_ is None:
        std_ = np.nanstd(Xf, axis=0)
        std_ = np.where(np.isfinite(std_) & (std_ > 0), std_, 1.0)
    Xs = (Xf - mean_[None, :]) / std_[None, :]
    return np.column_stack([np.ones(Xs.shape[0], dtype=np.float64), Xs]), mean_, std_


def _weighted_r2_flat(y_true: np.ndarray, y_pred: np.ndarray, weights_flat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights_flat = np.asarray(weights_flat, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        return np.nan
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid.sum()) < 10:
        return np.nan
    count = valid.sum(axis=0).astype(np.float64)
    enough = count >= 3
    if not np.any(enough):
        return np.nan
    y0 = np.where(valid, y_true, 0.0)
    mean = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    mean[enough] = y0[:, enough].sum(axis=0) / count[enough]
    resid2 = np.where(valid, (y_true - y_pred) ** 2, 0.0)
    base2 = np.where(valid, (y_true - mean[None, :]) ** 2, 0.0)
    w = np.where(np.isfinite(weights_flat) & (weights_flat > 0), weights_flat, 0.0)
    ok = enough & (w > 0)
    if not np.any(ok):
        return np.nan
    sse = float(np.nansum(resid2[:, ok] * w[None, ok]))
    sst = float(np.nansum(base2[:, ok] * w[None, ok]))
    if not np.isfinite(sst) or sst <= 0:
        return np.nan
    return float(1.0 - sse / sst)


def _fit_predict_multioutput(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    A_train, mean_x, std_x = _prepare_design_matrix(X_train)
    A_test, _, _ = _prepare_design_matrix(X_test, mean_x, std_x)
    col_mean = np.nanmean(Y_train, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    Y_fit = np.where(np.isfinite(Y_train), Y_train, col_mean[None, :])
    coef, *_ = np.linalg.lstsq(A_train, Y_fit, rcond=None)
    return A_test @ coef


def _joint_field_r2(X: np.ndarray, field_samples: np.ndarray, lat: np.ndarray, lon: np.ndarray, sample_years: np.ndarray) -> tuple[float, float, int, int]:
    Y_all = field_samples.reshape(field_samples.shape[0], -1).astype(np.float64, copy=False)
    finite_col = np.isfinite(Y_all).mean(axis=0) >= 0.70
    if int(finite_col.sum()) < 3 or X.shape[0] < max(12, X.shape[1] + 4):
        return np.nan, np.nan, int(finite_col.sum()), 0
    Y = Y_all[:, finite_col]
    w2 = _lat_weights(lat, len(lon)).reshape(-1)[finite_col]
    pred_in = _fit_predict_multioutput(X, Y, X)
    r2_in = _weighted_r2_flat(Y, pred_in, w2)
    pred_cv = np.full_like(Y, np.nan, dtype=np.float64)
    n_folds = 0
    yarr = sample_years.astype(int)
    for y in np.unique(yarr):
        test = yarr == int(y)
        train = ~test
        if int(test.sum()) < 1 or int(train.sum()) < max(10, X.shape[1] + 3):
            continue
        try:
            pred_cv[test, :] = _fit_predict_multioutput(X[train, :], Y[train, :], X[test, :])
            n_folds += 1
        except np.linalg.LinAlgError:
            continue
    r2_cv = _weighted_r2_flat(Y, pred_cv, w2) if n_folds > 0 else np.nan
    return float(r2_in) if np.isfinite(r2_in) else np.nan, float(r2_cv) if np.isfinite(r2_cv) else np.nan, int(finite_col.sum()), int(n_folds)


def _multioutput_r2_vector(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    Y_true = np.asarray(Y_true, dtype=np.float64)
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    out = np.full(Y_true.shape[1], np.nan, dtype=np.float64)
    for j in range(Y_true.shape[1]):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        m = np.isfinite(yt) & np.isfinite(yp)
        if int(m.sum()) < 8:
            continue
        sst = float(np.sum((yt[m] - np.mean(yt[m])) ** 2))
        if sst <= 0:
            continue
        sse = float(np.sum((yt[m] - yp[m]) ** 2))
        out[j] = 1.0 - sse / sst
    return out


def _eof_score_coverage(X: np.ndarray, scores: np.ndarray, evr: np.ndarray, sample_years: np.ndarray, top_k: int) -> tuple[float, float, str]:
    if scores.shape[1] == 0 or evr.size == 0:
        return np.nan, np.nan, "no_eof_scores"
    k = int(min(top_k, scores.shape[1], evr.size))
    if k <= 0 or X.shape[0] < max(12, X.shape[1] + 4):
        return np.nan, np.nan, "too_few_samples"
    Y = scores[:, :k].astype(np.float64, copy=False)
    w = evr[:k].astype(np.float64, copy=True)
    if not np.isfinite(w).any() or np.nansum(w) <= 0:
        w = np.ones(k, dtype=np.float64) / k
    else:
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        w = w / np.sum(w)
    pred_in = _fit_predict_multioutput(X, Y, X)
    r2_in_vec = _multioutput_r2_vector(Y, pred_in)
    pred_cv = np.full_like(Y, np.nan, dtype=np.float64)
    n_folds = 0
    yarr = sample_years.astype(int)
    for y in np.unique(yarr):
        test = yarr == int(y)
        train = ~test
        if int(test.sum()) < 1 or int(train.sum()) < max(10, X.shape[1] + 3):
            continue
        try:
            pred_cv[test, :] = _fit_predict_multioutput(X[train, :], Y[train, :], X[test, :])
            n_folds += 1
        except np.linalg.LinAlgError:
            continue
    r2_cv_vec = _multioutput_r2_vector(Y, pred_cv) if n_folds > 0 else np.full(k, np.nan)
    def _weighted(v: np.ndarray) -> float:
        m = np.isfinite(v) & np.isfinite(w)
        if not np.any(m) or np.nansum(w[m]) <= 0:
            return np.nan
        return float(np.nansum(v[m] * w[m]) / np.nansum(w[m]))
    return _weighted(r2_in_vec), _weighted(r2_cv_vec), "ok"


def _coverage_tier(joint_cv: float, eof5_cv: float) -> str:
    vals = np.asarray([joint_cv, eof5_cv], dtype=np.float64)
    signal = np.nanmax(vals) if np.any(np.isfinite(vals)) else np.nan
    if not np.isfinite(signal):
        return "not_estimated"
    if signal >= 0.60:
        return "very_high_joint_coverage"
    if signal >= 0.40:
        return "high_joint_coverage"
    if signal >= 0.20:
        return "moderate_joint_coverage"
    if signal >= 0.05:
        return "low_but_nonzero_joint_coverage"
    return "weak_or_unstable_joint_coverage"


def compute_window_family_joint_coverage(index_df: pd.DataFrame, fields: Dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray, years: np.ndarray, index_metrics: pd.DataFrame, settings: IndexValidityV1BSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    timing_rows = []
    rows = []
    cache = _MetricCache(fields, lat, lon, years)
    families = ["P", "V", "H", "Je", "Jw"]
    family_to_indices = {fam: [n for n in VARIABLE_ORDER if str(INDEX_METADATA[n]["family"]) == fam] for fam in families}
    for window, (day0, day1) in settings.windows.items():
        win_df = index_df[(index_df["day"] >= day0) & (index_df["day"] <= day1)].copy()
        for family in families:
            t0 = time.time()
            index_names = family_to_indices[family]
            vals = win_df[["year", "day", *index_names]].dropna().copy()
            for name in index_names:
                vals = vals[np.isfinite(vals[name].to_numpy(dtype=float))]
            n_samples = int(len(vals))
            _, sublat, sublon = cache.subfield(family)
            single = index_metrics[(index_metrics["window"].eq(window)) & (index_metrics["family"].eq(family))]
            best_single_r2 = float(single["field_r2_weighted"].max()) if len(single) else np.nan
            mean_single_r2 = float(single["field_r2_weighted"].mean()) if len(single) else np.nan
            if n_samples < max(20, len(index_names) + 8):
                rows.append({"window": window, "family": family, "n_indices": len(index_names), "indices": ";".join(index_names), "n_samples": n_samples, "joint_field_R2_in_sample": np.nan, "joint_field_R2_year_cv": np.nan, "joint_eof_coverage_top5_in_sample": np.nan, "joint_eof_coverage_top5_year_cv": np.nan, "joint_eof_coverage_top3_in_sample": np.nan, "joint_eof_coverage_top3_year_cv": np.nan, "best_single_index_R2": best_single_r2, "mean_single_index_R2": mean_single_r2, "joint_gain_over_best_single": np.nan, "coverage_tier": "not_estimated", "collapse_risk_update": "not_estimated_too_few_samples", "n_valid_gridpoints": 0, "n_cv_year_folds": 0, "eof_coverage_status": "not_run_too_few_samples"})
                timing_rows.append({"task": "family_joint_coverage", "window": window, "index_name": family, "seconds": round(time.time() - t0, 6)})
                continue
            samples = cache.samples(family, vals[["year", "day"]])
            X = vals[index_names].to_numpy(dtype=np.float64)
            sample_years = vals["year"].to_numpy(dtype=int)
            r2_in, r2_cv, n_grid, n_folds = _joint_field_r2(X, samples, sublat, sublon, sample_years)
            scores, evr = cache.eof_scores(family, vals[["year", "day"]], n_modes=max(5, settings.eof_n_modes))
            eof5_in, eof5_cv, eof_status5 = _eof_score_coverage(X, scores, evr, sample_years, top_k=5)
            eof3_in, eof3_cv, eof_status3 = _eof_score_coverage(X, scores, evr, sample_years, top_k=3)
            tier = _coverage_tier(r2_cv, eof5_cv)
            sig_vals = np.asarray([r2_cv, eof5_cv], dtype=np.float64)
            signal = np.nanmax(sig_vals) if np.any(np.isfinite(sig_vals)) else np.nan
            if not np.isfinite(signal):
                update = "not_estimated"
            elif signal < 0.05 and (not np.isfinite(best_single_r2) or best_single_r2 < 0.05):
                update = "possible_joint_coverage_gap_review_maps"
            else:
                update = "no_family_collapse_update_supported"
            rows.append({"window": window, "family": family, "n_indices": len(index_names), "indices": ";".join(index_names), "n_samples": n_samples, "joint_field_R2_in_sample": r2_in, "joint_field_R2_year_cv": r2_cv, "joint_eof_coverage_top5_in_sample": eof5_in, "joint_eof_coverage_top5_year_cv": eof5_cv, "joint_eof_coverage_top3_in_sample": eof3_in, "joint_eof_coverage_top3_year_cv": eof3_cv, "best_single_index_R2": best_single_r2, "mean_single_index_R2": mean_single_r2, "joint_gain_over_best_single": (r2_cv - best_single_r2) if np.isfinite(r2_cv) and np.isfinite(best_single_r2) else np.nan, "coverage_tier": tier, "collapse_risk_update": update, "n_valid_gridpoints": int(n_grid), "n_cv_year_folds": int(n_folds), "eof_coverage_status": eof_status5 if eof_status5 == eof_status3 else f"top5={eof_status5};top3={eof_status3}", "interpretation_guardrail": "Joint coverage measures mixed family-level index indication of the same field; it is not lead-lag, pathway, or causal evidence."})
            timing_rows.append({"task": "family_joint_coverage", "window": window, "index_name": family, "seconds": round(time.time() - t0, 6)})
    out = pd.DataFrame(rows)
    out["_w"] = out["window"].map({w: i for i, w in enumerate(settings.windows.keys())})
    out["_f"] = out["family"].map({f: i for i, f in enumerate(families)})
    return out.sort_values(["_w", "_f"]).drop(columns=["_w", "_f"]), pd.DataFrame(timing_rows)

def build_family_guardrail(index_metrics: pd.DataFrame, settings: IndexValidityV1BSettings) -> pd.DataFrame:
    rows = []
    order = ["strong", "moderate", "weak_but_usable", "high_risk", "not_supported"]
    for (window, family), sub in index_metrics.groupby(["window", "family"], dropna=False):
        counts = {f"n_{k}": int((sub["representativeness_tier"] == k).sum()) for k in order}
        usable = counts["n_strong"] + counts["n_moderate"] + counts["n_weak_but_usable"]
        best_idx = sub.sort_values("overall_score", ascending=False).iloc[0]
        n_indices = int(len(sub)); high_bad = counts["n_high_risk"] + counts["n_not_supported"]
        if usable == 0 or float(best_idx["overall_score"]) < settings.family_best_low_threshold:
            collapse = "high"; recommended = "do_not_use_family_without_manual_map_review"
        elif usable < max(1, int(np.ceil(n_indices / 3))) or high_bad > usable:
            collapse = "partial_sensitivity"; recommended = "use_family_with_index_level_flags_and_prefer_best_indices"
        else:
            collapse = "low"; recommended = "family_has_usable_indices_no_whole_family_collapse_supported"
        rows.append({"window": window, "family": family, "n_indices": n_indices, **counts, "n_usable": int(usable), "best_index": str(best_idx["index_name"]), "best_index_score": float(best_idx["overall_score"]), "best_index_tier": str(best_idx["representativeness_tier"]), "family_collapse_risk": collapse, "recommended_use": recommended, "guardrail_interpretation": ("No evidence of whole-family index collapse." if collapse == "low" else "Some index sensitivity exists, but whole-family collapse is not established." if collapse == "partial_sensitivity" else "Potential whole-family representativeness failure; manual field-map review required before using this family in this window.")})
    win_order = list(settings.windows.keys()); fam_order = ["P", "V", "H", "Je", "Jw"]
    out = pd.DataFrame(rows)
    out["_w"] = out["window"].map({w: i for i, w in enumerate(win_order)})
    out["_f"] = out["family"].map({f: i for i, f in enumerate(fam_order)})
    return out.sort_values(["_w", "_f"]).drop(columns=["_w", "_f"])
