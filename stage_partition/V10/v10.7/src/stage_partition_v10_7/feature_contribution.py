from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

try:
    from scipy.ndimage import gaussian_filter1d as _scipy_gaussian_filter1d
except Exception:  # pragma: no cover
    _scipy_gaussian_filter1d = None

from .event_content_config import EventContentConfig
from .profile_event_diff import _feature_coord


def _robust_normalize(x: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    rows = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 5:
            rows.append({"feature_index": j, "used_in_derivative": False, "reason": "too_few_finite"})
            continue
        q25, med, q75 = np.nanpercentile(col, [25, 50, 75])
        iqr = q75 - q25
        if not np.isfinite(iqr) or iqr < 1e-12:
            rows.append({"feature_index": j, "used_in_derivative": False, "reason": "near_zero_iqr"})
            continue
        z = (col - med) / iqr
        if not finite.all():
            idx = np.arange(len(col), dtype=float)
            z = np.interp(idx, idx[finite], z[finite], left=z[finite][0], right=z[finite][-1])
        out[:, j] = z
        rows.append({"feature_index": j, "used_in_derivative": True, "reason": ""})
    keep = np.array([r["used_in_derivative"] for r in rows], dtype=bool)
    return out[:, keep], pd.DataFrame(rows)


def _manual_derivative(x: np.ndarray, sigma: float) -> np.ndarray:
    # Fallback: smooth by simple moving window roughly proportional to sigma, then gradient.
    win = max(3, int(round(float(sigma) * 2 + 1)))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    kernel = np.ones(win, dtype=float) / win
    sm = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        y = np.pad(x[:, j], pad, mode="reflect")
        sm[:, j] = np.convolve(y, kernel, mode="valid")
    return np.gradient(sm, axis=0)


def _derivative(x: np.ndarray, sigma: float) -> tuple[np.ndarray, str]:
    if _scipy_gaussian_filter1d is not None:
        return _scipy_gaussian_filter1d(x, sigma=float(sigma), axis=0, order=1, mode="reflect"), "scipy_gaussian_filter1d"
    return _manual_derivative(x, sigma), "numpy_moving_gradient_fallback"


def compute_feature_contribution(state_matrix: np.ndarray, valid_day_index: np.ndarray, feature_table: pd.DataFrame, cfg: EventContentConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    z, norm_audit = _robust_normalize(np.asarray(state_matrix, dtype=float))
    keep_features = norm_audit.loc[norm_audit["used_in_derivative"].astype(bool), "feature_index"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    backend_names = set()
    day_index = np.asarray(valid_day_index, dtype=int)
    day_to_pos = {int(d): i for i, d in enumerate(day_index)}
    for ev in cfg.event_windows:
        sigmas = cfg.feature_contribution_sigmas.get(ev.event_id, cfg.representative_sigmas)
        for sigma in sigmas:
            deriv, backend = _derivative(z, float(sigma))
            backend_names.add(backend)
            # Use local max derivative energy within +/-2 days of target_day.
            candidates = [d for d in range(ev.target_day - 2, ev.target_day + 3) if d in day_to_pos]
            if not candidates:
                continue
            energy_by_day = []
            for d in candidates:
                vec = deriv[day_to_pos[d], :]
                energy_by_day.append((d, float(np.sqrt(np.nansum(vec ** 2)))))
            local_day = max(energy_by_day, key=lambda x: x[1])[0]
            pos = day_to_pos[local_day]
            vec = deriv[pos, :]
            absvec = np.abs(vec)
            denom = np.nansum(absvec)
            if not np.isfinite(denom) or denom <= 0:
                denom = np.nan
            ranks = pd.Series(absvec).rank(method="min", ascending=False, na_option="bottom").astype(int).to_numpy()
            for local_j, feature_index in enumerate(keep_features):
                val = vec[local_j]
                rows.append({
                    "event_id": ev.event_id,
                    "target_day": int(ev.target_day),
                    "local_max_day_within_radius": int(local_day),
                    "sigma": float(sigma),
                    "feature_name": f"feature_{int(feature_index):03d}",
                    "feature_index": int(feature_index),
                    "feature_coord_if_available": _feature_coord(feature_table, int(feature_index)),
                    "derivative_value": float(val) if np.isfinite(val) else np.nan,
                    "abs_derivative": float(absvec[local_j]) if np.isfinite(absvec[local_j]) else np.nan,
                    "energy_contribution_fraction": float(absvec[local_j] / denom) if np.isfinite(absvec[local_j]) and np.isfinite(denom) else np.nan,
                    "rank": int(ranks[local_j]) if np.isfinite(absvec[local_j]) else np.nan,
                })
    meta = {"feature_contribution_backend": ";".join(sorted(backend_names)), "n_derivative_features": int(len(keep_features))}
    return pd.DataFrame(rows), norm_audit, meta
