from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_io import extract_object_subfield
from .settings import LeadLagScreenV4Settings


@dataclass
class EOFWindowModel:
    window: str
    object_name: str
    train_days: np.ndarray
    ext_days: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    grid_valid_mask: np.ndarray
    components_weighted: np.ndarray       # shape k x n_valid_grid
    loadings_unweighted: np.ndarray       # shape k x n_lat x n_lon
    col_mean: np.ndarray                  # n_valid_grid
    weights: np.ndarray                   # n_valid_grid
    score_mean: np.ndarray                # k
    score_std: np.ndarray                 # k
    explained_variance_ratio: np.ndarray  # k
    singular_values: np.ndarray           # k
    quality_flag: str
    notes: str


def _day_indices(days: np.ndarray) -> np.ndarray:
    return np.asarray(days, dtype=int) - 1


def _lat_weights(lat: np.ndarray, n_lon: int, mode: str) -> np.ndarray:
    if mode == "none":
        w_lat = np.ones_like(lat, dtype=float)
    elif mode == "cos_lat":
        w_lat = np.cos(np.deg2rad(lat))
        w_lat = np.clip(w_lat, 0.0, None)
    elif mode == "sqrt_cos_lat":
        w_lat = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    else:
        raise ValueError(f"Unsupported EOF weighting mode: {mode}")
    return np.repeat(w_lat[:, None], n_lon, axis=1).reshape(-1)


def _flatten_samples(sub: np.ndarray, days: np.ndarray) -> np.ndarray:
    idx = _day_indices(days)
    if np.any(idx < 0) or np.any(idx >= sub.shape[1]):
        raise IndexError(f"Day index out of bounds for EOF extraction. days={days}, n_day={sub.shape[1]}")
    arr = sub[:, idx, :, :]
    return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2] * arr.shape[3])


def _quality_flag(explained: np.ndarray, settings: LeadLagScreenV4Settings, missing_fraction: float) -> str:
    pc1 = float(explained[0]) if len(explained) and np.isfinite(explained[0]) else np.nan
    flags: List[str] = []
    if not np.isfinite(pc1):
        flags.append("eof_failed")
    elif pc1 >= settings.eof_quality_good_variance_ratio:
        flags.append("pc1_good")
    elif pc1 >= settings.eof_quality_moderate_variance_ratio:
        flags.append("pc1_moderate")
    else:
        flags.append("pc1_low_variance")
    if missing_fraction > 0.20:
        flags.append("missing_risk")
    return ";".join(flags)


def _fit_eof_model(
    subfield: np.ndarray,
    obj_lats: np.ndarray,
    obj_lons: np.ndarray,
    object_name: str,
    window: str,
    train_days: np.ndarray,
    ext_days: np.ndarray,
    years: np.ndarray,
    settings: LeadLagScreenV4Settings,
) -> Tuple[EOFWindowModel, pd.DataFrame, pd.DataFrame]:
    X_train = _flatten_samples(subfield, train_days)
    n_grid_total = X_train.shape[1]
    finite_grid_fraction = np.isfinite(X_train).mean(axis=0)
    grid_valid = finite_grid_fraction >= settings.eof_grid_min_finite_fraction
    if int(grid_valid.sum()) < max(settings.eof_max_modes + 1, 3):
        raise ValueError(f"{window}-{object_name}: EOF 可用格点不足：{int(grid_valid.sum())}/{n_grid_total}")

    notes: List[str] = []
    Xg = X_train[:, grid_valid]
    col_mean = np.nanmean(Xg, axis=0)
    bad_mean = ~np.isfinite(col_mean)
    if np.any(bad_mean):
        col_mean[bad_mean] = 0.0
        notes.append(f"bad_grid_mean_count={int(bad_mean.sum())}")
    Xc = Xg - col_mean[None, :]
    row_frac = np.isfinite(Xc).mean(axis=1)
    row_valid = row_frac >= settings.eof_row_min_finite_fraction
    if int(row_valid.sum()) < max(settings.eof_max_modes + 2, 8):
        raise ValueError(f"{window}-{object_name}: EOF 可用样本不足：{int(row_valid.sum())}/{len(row_valid)}")
    Xfit = np.where(np.isfinite(Xc[row_valid]), Xc[row_valid], 0.0)

    full_weights = _lat_weights(obj_lats, len(obj_lons), settings.eof_weighting)
    weights = full_weights[grid_valid]
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    Xw = Xfit * weights[None, :]
    _, s, vt = np.linalg.svd(Xw, full_matrices=False)
    k = min(settings.eof_max_modes, vt.shape[0])
    comps = vt[:k, :].copy()
    total_var = float(np.sum(s ** 2))
    explained = (s[:k] ** 2) / total_var if total_var > 0 else np.full((k,), np.nan)

    # Stable sign convention: largest-magnitude unweighted loading is positive.
    loadings = np.full((k, len(obj_lats), len(obj_lons)), np.nan, dtype=float)
    for m in range(k):
        unweighted = comps[m, :] / weights
        if unweighted.size and np.isfinite(unweighted).any():
            imax = int(np.nanargmax(np.abs(unweighted)))
            if np.isfinite(unweighted[imax]) and unweighted[imax] < 0:
                comps[m, :] *= -1.0
                unweighted *= -1.0
        grid = np.full((n_grid_total,), np.nan, dtype=float)
        grid[grid_valid] = unweighted
        loadings[m] = grid.reshape(len(obj_lats), len(obj_lons))

    def project(days: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = _flatten_samples(subfield, days)[:, grid_valid]
        rf = np.isfinite(X).mean(axis=1)
        Xc2 = X - col_mean[None, :]
        Xc2 = np.where(np.isfinite(Xc2), Xc2, 0.0)
        scores_raw = np.dot(Xc2 * weights[None, :], comps.T)
        scores_raw[rf < settings.eof_row_min_finite_fraction, :] = np.nan
        return scores_raw, rf

    train_scores_raw, _ = project(train_days)
    score_mean = np.nanmean(train_scores_raw, axis=0)
    score_std = np.nanstd(train_scores_raw, axis=0)
    score_std = np.where(np.isfinite(score_std) & (score_std > 0), score_std, 1.0)

    ext_scores_raw, ext_row_frac = project(ext_days)
    ext_scores = (ext_scores_raw - score_mean[None, :]) / score_std[None, :]

    missing_fraction = 1.0 - float(np.isfinite(X_train[:, grid_valid]).sum() / X_train[:, grid_valid].size)
    model = EOFWindowModel(
        window=window,
        object_name=object_name,
        train_days=np.asarray(train_days, dtype=int),
        ext_days=np.asarray(ext_days, dtype=int),
        lat=np.asarray(obj_lats, dtype=float),
        lon=np.asarray(obj_lons, dtype=float),
        grid_valid_mask=grid_valid.reshape(len(obj_lats), len(obj_lons)),
        components_weighted=comps,
        loadings_unweighted=loadings,
        col_mean=col_mean,
        weights=weights,
        score_mean=score_mean,
        score_std=score_std,
        explained_variance_ratio=explained,
        singular_values=s[:k],
        quality_flag=_quality_flag(explained, settings, missing_fraction),
        notes=";".join(notes),
    )

    score_rows = []
    score_cube = ext_scores.reshape(len(years), len(ext_days), k)
    train_set = set(int(d) for d in train_days)
    for yi, year in enumerate(years):
        for di, day in enumerate(ext_days):
            for mode in range(k):
                score_rows.append({
                    "window": window,
                    "object": object_name,
                    "year": int(year),
                    "day": int(day),
                    "mode": int(mode + 1),
                    "score": float(score_cube[yi, di, mode]) if np.isfinite(score_cube[yi, di, mode]) else np.nan,
                    "is_target_window_day": int(day) in train_set,
                    "is_padding_day": int(day) not in train_set,
                })
    score_df = pd.DataFrame(score_rows)

    quality_rows = []
    for mode in range(k):
        quality_rows.append({
            "window": window,
            "object": object_name,
            "mode": int(mode + 1),
            "explained_variance_ratio": float(explained[mode]) if np.isfinite(explained[mode]) else np.nan,
            "singular_value": float(s[mode]) if mode < len(s) else np.nan,
            "quality_flag_pc1_object": model.quality_flag,
            "n_train_samples": int(X_train.shape[0]),
            "n_valid_gridpoints": int(grid_valid.sum()),
            "n_total_gridpoints": int(n_grid_total),
            "missing_fraction": missing_fraction,
            "notes": model.notes,
        })
    quality_df = pd.DataFrame(quality_rows)
    return model, score_df, quality_df


def build_window_eof_models(
    fields: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    settings: LeadLagScreenV4Settings,
    logger,
) -> Tuple[Dict[Tuple[str, str], EOFWindowModel], pd.DataFrame, pd.DataFrame]:
    models: Dict[Tuple[str, str], EOFWindowModel] = {}
    score_tables: List[pd.DataFrame] = []
    quality_tables: List[pd.DataFrame] = []
    subfields = {}
    coords = {}
    for obj in settings.objects:
        sub, obj_lats, obj_lons = extract_object_subfield(fields, lat, lon, obj)
        subfields[obj] = sub
        coords[obj] = (obj_lats, obj_lons)

    for window, (start, end) in settings.windows.items():
        train_days = np.arange(start, end + 1, dtype=int)
        ext_start = max(1, start - settings.max_lag)
        ext_end = min(183, end + settings.max_lag)
        ext_days = np.arange(ext_start, ext_end + 1, dtype=int)
        logger.info("Fitting EOF models for %s train=%d-%d ext=%d-%d", window, start, end, ext_start, ext_end)
        for obj in settings.objects:
            model, score_df, quality_df = _fit_eof_model(
                subfield=subfields[obj],
                obj_lats=coords[obj][0],
                obj_lons=coords[obj][1],
                object_name=obj,
                window=window,
                train_days=train_days,
                ext_days=ext_days,
                years=years,
                settings=settings,
            )
            models[(window, obj)] = model
            score_tables.append(score_df)
            quality_tables.append(quality_df)
    return models, pd.concat(score_tables, ignore_index=True), pd.concat(quality_tables, ignore_index=True)
