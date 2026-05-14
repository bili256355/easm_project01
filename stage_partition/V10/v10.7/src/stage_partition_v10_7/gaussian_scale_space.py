from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import pandas as pd

try:  # preferred backend
    from scipy.ndimage import gaussian_filter1d as _scipy_gaussian_filter1d
except Exception:  # pragma: no cover
    _scipy_gaussian_filter1d = None


@dataclass
class ScaleBackendInfo:
    name: str
    scipy_available: bool


def _robust_feature_normalize(state_matrix: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    x = np.asarray(state_matrix, dtype=float)
    rows: list[dict[str, Any]] = []
    out = np.full_like(x, np.nan, dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 5:
            rows.append({
                "feature_index": int(j),
                "median": np.nan,
                "iqr": np.nan,
                "finite_fraction": float(finite.mean()),
                "used_in_energy": False,
                "reason_if_excluded": "too_few_finite_values",
            })
            continue
        q25, med, q75 = np.nanpercentile(col, [25, 50, 75])
        iqr = float(q75 - q25)
        if (not np.isfinite(iqr)) or iqr < 1e-12:
            rows.append({
                "feature_index": int(j),
                "median": float(med) if np.isfinite(med) else np.nan,
                "iqr": float(iqr) if np.isfinite(iqr) else np.nan,
                "finite_fraction": float(finite.mean()),
                "used_in_energy": False,
                "reason_if_excluded": "near_zero_iqr",
            })
            continue
        normalized = (col - med) / iqr
        # Fill rare missing values by nearest finite interpolation for filtering; audit retains missing fraction.
        if not finite.all():
            idx = np.arange(len(col), dtype=float)
            normalized = np.interp(idx, idx[finite], normalized[finite], left=normalized[finite][0], right=normalized[finite][-1])
        out[:, j] = normalized
        rows.append({
            "feature_index": int(j),
            "median": float(med),
            "iqr": float(iqr),
            "finite_fraction": float(finite.mean()),
            "used_in_energy": True,
            "reason_if_excluded": "",
        })
    keep = np.asarray([r["used_in_energy"] for r in rows], dtype=bool)
    if not keep.any():
        raise ValueError("No usable H state features remain after robust normalization; cannot run V10.7_b scale diagnostic.")
    return out[:, keep], pd.DataFrame(rows)


def _manual_gaussian_kernel(sigma: float, order: int = 0) -> np.ndarray:
    radius = max(2, int(math.ceil(4.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-0.5 * (x / float(sigma)) ** 2)
    if order == 0:
        k = g / g.sum()
    elif order == 1:
        # First derivative of Gaussian. Sign convention not important because energy uses squared norm.
        k = -x / (float(sigma) ** 2) * g
        norm = np.sum(np.abs(k))
        if norm > 0:
            k = k / norm
    else:  # pragma: no cover
        raise ValueError("Only order=0 or order=1 is supported by manual Gaussian fallback.")
    return k.astype(float)


def _convolve_reflect_1d(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    padded = np.pad(np.asarray(y, dtype=float), pad_width=radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _manual_gaussian_filter1d(x: np.ndarray, sigma: float, axis: int = 0, order: int = 0) -> np.ndarray:
    if axis != 0:
        raise ValueError("Manual fallback supports axis=0 only.")
    k = _manual_gaussian_kernel(float(sigma), order=order)
    out = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        out[:, j] = _convolve_reflect_1d(x[:, j], k)
    return out


def _gaussian_filter1d(x: np.ndarray, sigma: float, order: int = 0) -> tuple[np.ndarray, ScaleBackendInfo]:
    if _scipy_gaussian_filter1d is not None:
        return (
            _scipy_gaussian_filter1d(x, sigma=float(sigma), axis=0, order=int(order), mode="reflect"),
            ScaleBackendInfo(name="scipy_gaussian_filter1d", scipy_available=True),
        )
    return (
        _manual_gaussian_filter1d(x, sigma=float(sigma), axis=0, order=int(order)),
        ScaleBackendInfo(name="numpy_manual_gaussian_fallback", scipy_available=False),
    )


def _within_sigma_normalize(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(energy, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return np.zeros_like(y, dtype=float), np.zeros_like(y, dtype=float), np.zeros_like(y, dtype=float)
    vals = y[finite]
    p5, p95 = np.nanpercentile(vals, [5, 95])
    denom = p95 - p5
    if not np.isfinite(denom) or denom < 1e-12:
        norm01 = np.zeros_like(y, dtype=float)
    else:
        norm01 = (y - p5) / denom
        norm01 = np.clip(norm01, 0.0, 1.0)
    order = np.argsort(vals)
    ranks = np.empty_like(vals, dtype=float)
    ranks[order] = np.linspace(0.0, 100.0, len(vals), endpoint=True)
    pct = np.zeros_like(y, dtype=float)
    pct[finite] = ranks
    q25, med, q75 = np.nanpercentile(vals, [25, 50, 75])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr < 1e-12:
        rz = np.zeros_like(y, dtype=float)
    else:
        rz = (y - med) / iqr
    return norm01, pct, rz


def build_scale_energy_map(
    state_matrix: np.ndarray,
    valid_day_index: np.ndarray,
    sigmas: tuple[float, ...],
    boundary_sigma_multiplier: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build day x sigma Gaussian derivative energy map from H object state matrix.

    This does not run ruptures.Window and should not be interpreted as breakpoint detection.
    """
    x_norm, feature_audit = _robust_feature_normalize(np.asarray(state_matrix, dtype=float))
    day_index = np.asarray(valid_day_index, dtype=int)
    rows: list[dict[str, Any]] = []
    backend_names = set()
    for sigma in sigmas:
        deriv, backend = _gaussian_filter1d(x_norm, float(sigma), order=1)
        backend_names.add(backend.name)
        # Normalize by sqrt(n_features) so raw energy is less dependent on feature count.
        energy = np.sqrt(np.nansum(np.square(deriv), axis=1)) / np.sqrt(max(1, deriv.shape[1]))
        norm01, pct, rz = _within_sigma_normalize(energy)
        boundary_margin = int(math.ceil(float(boundary_sigma_multiplier) * float(sigma)))
        for i, day in enumerate(day_index):
            rows.append({
                "day": int(day),
                "sigma": float(sigma),
                "energy_raw": float(energy[i]) if np.isfinite(energy[i]) else np.nan,
                "energy_norm_within_sigma": float(norm01[i]) if np.isfinite(norm01[i]) else np.nan,
                "energy_percentile_within_sigma": float(pct[i]) if np.isfinite(pct[i]) else np.nan,
                "energy_robust_z_within_sigma": float(rz[i]) if np.isfinite(rz[i]) else np.nan,
                "boundary_risk_flag": bool(i < boundary_margin or i > (len(day_index) - 1 - boundary_margin)),
            })
    meta = {
        "scale_backend": ";".join(sorted(backend_names)),
        "n_input_days": int(state_matrix.shape[0]),
        "n_input_features": int(state_matrix.shape[1]),
        "n_used_features": int(feature_audit["used_in_energy"].sum()),
        "n_excluded_features": int((~feature_audit["used_in_energy"].astype(bool)).sum()),
    }
    return pd.DataFrame(rows), feature_audit, meta


def build_missing_value_audit(state_matrix: np.ndarray, valid_day_index: np.ndarray) -> pd.DataFrame:
    x = np.asarray(state_matrix, dtype=float)
    rows = []
    for j in range(x.shape[1]):
        finite = np.isfinite(x[:, j])
        bad_days = np.asarray(valid_day_index, dtype=int)[~finite]
        rows.append({
            "feature_index": int(j),
            "finite_fraction": float(finite.mean()),
            "n_missing_days": int((~finite).sum()),
            "missing_day_list": ";".join(str(int(d)) for d in bad_days[:50]),
            "missing_day_list_truncated": bool(len(bad_days) > 50),
        })
    return pd.DataFrame(rows)
