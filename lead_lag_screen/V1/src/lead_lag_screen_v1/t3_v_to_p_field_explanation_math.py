# -*- coding: utf-8 -*-
"""Math helpers for T3 V→P field-explanation audit."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def corr_r2_beta_map(x: np.ndarray, y_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute r, r^2 and OLS beta map for one source vector and P samples.

    Parameters
    ----------
    x : shape (n,)
    y_samples : shape (n, lat, lon)

    Returns
    -------
    r_map, r2_map, beta_map, n_samples
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_samples, dtype=float)
    if y.ndim != 3:
        raise ValueError("y_samples must be samples x lat x lon")
    n = x.shape[0]
    flat = y.reshape(n, -1)
    x_valid = np.isfinite(x)
    valid = x_valid[:, None] & np.isfinite(flat)
    counts = valid.sum(axis=0)

    x2 = np.where(x_valid, x, np.nan)
    x_mean = np.nanmean(np.where(x_valid, x, np.nan))
    # Per-grid y mean due to possible grid missing values.
    y_masked = np.where(valid, flat, np.nan)
    y_mean = np.nanmean(y_masked, axis=0)

    xc = x[:, None] - x_mean
    yc = flat - y_mean[None, :]
    xc = np.where(valid, xc, 0.0)
    yc = np.where(valid, yc, 0.0)

    ssx = np.sum(xc * xc, axis=0)
    ssy = np.sum(yc * yc, axis=0)
    cov = np.sum(xc * yc, axis=0)
    denom = np.sqrt(ssx * ssy)

    r = np.full(flat.shape[1], np.nan, dtype=float)
    beta = np.full(flat.shape[1], np.nan, dtype=float)
    ok = (counts >= 3) & (denom > 0) & (ssx > 0)
    r[ok] = cov[ok] / denom[ok]
    beta[ok] = cov[ok] / ssx[ok]
    r2 = r * r
    return (
        r.reshape(y.shape[1], y.shape[2]),
        r2.reshape(y.shape[1], y.shape[2]),
        beta.reshape(y.shape[1], y.shape[2]),
        int(np.nanmax(counts) if counts.size else 0),
    )


def _r2_from_design(X: np.ndarray, y_flat: np.ndarray) -> np.ndarray:
    """Vectorized R² for X against many grid columns.

    Rows with non-finite X are dropped globally. Grid columns with non-finite y
    anywhere in the retained rows are computed by a slower fallback only for
    those columns.
    """
    X = np.asarray(X, dtype=float)
    y_flat = np.asarray(y_flat, dtype=float)
    x_ok = np.all(np.isfinite(X), axis=1)
    X = X[x_ok]
    Y = y_flat[x_ok]
    n, p = X.shape
    out = np.full(Y.shape[1], np.nan, dtype=float)
    if n <= p + 1:
        return out

    Xd = np.column_stack([np.ones(n), X])
    full_valid = np.all(np.isfinite(Y), axis=0)
    if full_valid.any():
        Yv = Y[:, full_valid]
        beta = np.linalg.pinv(Xd) @ Yv
        pred = Xd @ beta
        resid = Yv - pred
        y_mean = np.mean(Yv, axis=0)
        sst = np.sum((Yv - y_mean[None, :]) ** 2, axis=0)
        sse = np.sum(resid ** 2, axis=0)
        ok = sst > 0
        vals = np.full(Yv.shape[1], np.nan)
        vals[ok] = np.maximum(0.0, 1.0 - sse[ok] / sst[ok])
        out[full_valid] = vals

    # Fallback for columns with partial missingness, expected to be small.
    bad_cols = np.where(~full_valid)[0]
    for j in bad_cols:
        yy = Y[:, j]
        ok_rows = np.isfinite(yy)
        if ok_rows.sum() <= p + 1:
            continue
        Xdj = Xd[ok_rows]
        yyj = yy[ok_rows]
        try:
            bj = np.linalg.lstsq(Xdj, yyj, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        pred = Xdj @ bj
        sst = np.sum((yyj - np.mean(yyj)) ** 2)
        if sst <= 0:
            continue
        sse = np.sum((yyj - pred) ** 2)
        out[j] = max(0.0, 1.0 - sse / sst)
    return out


def multicomponent_partial_r2_maps(X_by_component: Dict[str, np.ndarray], y_samples: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute component partial-R² maps from a same-lag V-component model."""
    names = list(X_by_component.keys())
    X_full = np.column_stack([np.asarray(X_by_component[n], dtype=float) for n in names])
    y = np.asarray(y_samples, dtype=float)
    flat = y.reshape(y.shape[0], -1)
    r2_full = _r2_from_design(X_full, flat)
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        keep = [j for j in range(len(names)) if j != i]
        if keep:
            r2_reduced = _r2_from_design(X_full[:, keep], flat)
        else:
            r2_reduced = np.zeros_like(r2_full)
        part = r2_full - r2_reduced
        part = np.where(np.isfinite(part), np.maximum(0.0, part), np.nan)
        out[name] = part.reshape(y.shape[1], y.shape[2])
    return out


def corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    xx = x[ok]
    yy = y[ok]
    if np.std(xx) <= 0 or np.std(yy) <= 0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def r2_1d(x: np.ndarray, y: np.ndarray) -> float:
    r = corr_1d(x, y)
    return float(r * r) if np.isfinite(r) else float("nan")


def beta_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    xx = x[ok]
    yy = y[ok]
    vx = np.var(xx)
    if vx <= 0:
        return float("nan")
    return float(np.cov(xx, yy, bias=True)[0, 1] / vx)


def spatial_pattern_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if mask is not None:
        ok = mask & np.isfinite(aa) & np.isfinite(bb)
    else:
        ok = np.isfinite(aa) & np.isfinite(bb)
    if ok.sum() < 3:
        return float("nan")
    av = aa[ok]
    bv = bb[ok]
    if np.std(av) <= 0 or np.std(bv) <= 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def bootstrap_ci(values: Iterable[float], q_low: float = 0.025, q_high: float = 0.975) -> Tuple[float, float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(vals)), float(np.nanquantile(vals, q_low)), float(np.nanquantile(vals, q_high))
