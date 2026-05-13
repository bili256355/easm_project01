from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def normal_two_sided_p_from_z(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    cdf = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    return max(0.0, min(1.0, 2.0 * (1.0 - cdf)))


def fisher_effn_p(r: float, n_eff: float) -> float:
    if not np.isfinite(r) or not np.isfinite(n_eff) or n_eff <= 3:
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r) * math.sqrt(max(n_eff - 3.0, 1e-9))
    return normal_two_sided_p_from_z(z)


def safe_corr_1d(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return np.nan, n
    xv = np.asarray(x[mask], dtype=float)
    yv = np.asarray(y[mask], dtype=float)
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    den = math.sqrt(float(np.sum(xv * xv) * np.sum(yv * yv)))
    if den <= 0:
        return np.nan, n
    return float(np.sum(xv * yv) / den), n


def corr_matrix_batch(x: np.ndarray, y: np.ndarray, min_count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx = np.isfinite(x).astype(float)
    my = np.isfinite(y).astype(float)
    xs = np.where(np.isfinite(x), x, 0.0)
    ys = np.where(np.isfinite(y), y, 0.0)

    count = np.einsum("bnv,bnw->bvw", mx, my, optimize=True)
    sumx = np.einsum("bnv,bnw->bvw", xs, my, optimize=True)
    sumy = np.einsum("bnv,bnw->bvw", mx, ys, optimize=True)
    sumxx = np.einsum("bnv,bnw->bvw", xs * xs, my, optimize=True)
    sumyy = np.einsum("bnv,bnw->bvw", mx, ys * ys, optimize=True)
    sumxy = np.einsum("bnv,bnw->bvw", xs, ys, optimize=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sumxy - (sumx * sumy / count)
        vx = sumxx - (sumx * sumx / count)
        vy = sumyy - (sumy * sumy / count)
        den = np.sqrt(vx * vy)
        corr = cov / den
    corr[(count < min_count) | ~np.isfinite(corr)] = np.nan
    return corr, count


def fdr_bh(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    pf = p[finite]
    m = len(pf)
    order = np.argsort(pf)
    ranked = pf[order]
    q_ranked = ranked * m / (np.arange(m) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    qf = np.empty_like(q_ranked)
    qf[order] = q_ranked
    q[finite] = qf
    return q


def estimate_ar1_params_diagnostic(a: np.ndarray, clip_limit: float = 0.95) -> dict:
    arr = np.asarray(a, dtype=float)
    finite_values = arr[np.isfinite(arr)]
    if finite_values.size < 5:
        return {
            "mu": 0.0,
            "raw_phi_before_clip": 0.0,
            "phi_after_clip": 0.0,
            "sigma": 1.0,
            "n_finite": int(finite_values.size),
            "n_pair_for_phi": 0,
            "phi_clip_limit": float(clip_limit),
            "phi_clipped_flag": False,
            "phi_clip_amount": 0.0,
            "phi_clip_direction": "none",
            "phi_clip_severity": "none",
        }
    mu = float(np.nanmean(arr))
    x0 = arr[:, :-1]
    x1 = arr[:, 1:]
    mask = np.isfinite(x0) & np.isfinite(x1)
    n_pair = int(mask.sum())
    if n_pair < 5:
        sigma0 = float(np.nanstd(finite_values))
        if not np.isfinite(sigma0) or sigma0 <= 0:
            sigma0 = 1.0
        return {
            "mu": mu,
            "raw_phi_before_clip": 0.0,
            "phi_after_clip": 0.0,
            "sigma": sigma0,
            "n_finite": int(finite_values.size),
            "n_pair_for_phi": n_pair,
            "phi_clip_limit": float(clip_limit),
            "phi_clipped_flag": False,
            "phi_clip_amount": 0.0,
            "phi_clip_direction": "none",
            "phi_clip_severity": "none",
        }

    z0 = x0[mask] - mu
    z1 = x1[mask] - mu
    den = float(np.sum(z0 * z0))
    raw_phi = 0.0 if den <= 0 else float(np.sum(z0 * z1) / den)
    phi = float(np.clip(raw_phi, -clip_limit, clip_limit))
    clipped = bool(not np.isclose(raw_phi, phi, rtol=0.0, atol=1e-12))
    clip_amount = float(abs(raw_phi - phi)) if clipped else 0.0
    if clipped and raw_phi > clip_limit:
        clip_direction = "upper"
    elif clipped and raw_phi < -clip_limit:
        clip_direction = "lower"
    else:
        clip_direction = "none"
    if not clipped:
        severity = "none"
    elif clip_amount < 0.01:
        severity = "minor"
    elif clip_amount < 0.03:
        severity = "moderate"
    else:
        severity = "severe"
    eps = z1 - phi * z0
    sigma = float(np.nanstd(eps))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(finite_values))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return {
        "mu": mu,
        "raw_phi_before_clip": raw_phi,
        "phi_after_clip": phi,
        "sigma": sigma,
        "n_finite": int(finite_values.size),
        "n_pair_for_phi": n_pair,
        "phi_clip_limit": float(clip_limit),
        "phi_clipped_flag": clipped,
        "phi_clip_amount": clip_amount,
        "phi_clip_direction": clip_direction,
        "phi_clip_severity": severity,
    }
