from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_io import build_panel_from_scores, extract_object_subfield
from .settings import LeadLagScreenV3Settings
from .stats_utils import safe_corr_1d


@dataclass
class EOFModel:
    object_name: str
    window: str
    field_name: str
    train_days: np.ndarray
    ext_days: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    grid_valid_mask: np.ndarray
    weighted_eof_vector: np.ndarray
    loading_unweighted: np.ndarray
    col_mean: np.ndarray
    weights: np.ndarray
    score_mean: float
    score_std: float
    explained_variance_ratio: float
    singular_value: float
    sign_reference_variable: str
    sign_reference_corr: float
    sign_flipped: bool
    quality_flag: str
    notes: str


def _day_indices(days: np.ndarray) -> np.ndarray:
    # Project convention: day 1 = Apr 1; array index = day - 1.
    return np.asarray(days, dtype=int) - 1


def _lat_weights(lat: np.ndarray, n_lon: int, mode: str) -> np.ndarray:
    if mode == "none":
        w_lat = np.ones_like(lat, dtype=np.float64)
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
        raise IndexError(f"EOF day index out of bounds. days={days[:3]}..{days[-3:]}, n_day={sub.shape[1]}")
    arr = sub[:, idx, :, :]
    return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2] * arr.shape[3])


def _make_quality_flag(var_ratio: float, sign_corr: float, settings: LeadLagScreenV3Settings, missing_fraction: float) -> str:
    flags = []
    if not np.isfinite(var_ratio):
        flags.append("pc1_failed")
    elif var_ratio >= settings.eof_quality_good_variance_ratio:
        flags.append("pc1_good")
    elif var_ratio >= settings.eof_quality_moderate_variance_ratio:
        flags.append("pc1_moderate")
    else:
        flags.append("pc1_low_variance")
    if (not np.isfinite(sign_corr)) or abs(sign_corr) < settings.sign_reference_min_abs_corr:
        flags.append("sign_reference_weak")
    if missing_fraction > 0.20:
        flags.append("missing_risk")
    return ";".join(flags)


def _fit_pc1_for_object_window(
    subfield: np.ndarray,
    obj_lats: np.ndarray,
    obj_lons: np.ndarray,
    object_name: str,
    window: str,
    train_days: np.ndarray,
    ext_days: np.ndarray,
    years: np.ndarray,
    index_df: pd.DataFrame,
    settings: LeadLagScreenV3Settings,
) -> Tuple[EOFModel, pd.DataFrame, pd.DataFrame]:
    X_train = _flatten_samples(subfield, train_days)
    n_train_samples, n_grid_total = X_train.shape
    finite_grid_fraction = np.isfinite(X_train).mean(axis=0)
    grid_valid = finite_grid_fraction >= settings.eof_grid_min_finite_fraction
    notes = []
    if int(grid_valid.sum()) < 3:
        raise ValueError(f"{window}-{object_name}: EOF 可用格点不足：{int(grid_valid.sum())}/{n_grid_total}")

    Xg = X_train[:, grid_valid]
    col_mean = np.nanmean(Xg, axis=0)
    # Replace all-nan/invalid means with 0 to keep SVD finite, but record risk.
    bad_mean = ~np.isfinite(col_mean)
    if np.any(bad_mean):
        col_mean[bad_mean] = 0.0
        notes.append(f"bad_grid_mean_count={int(bad_mean.sum())}")

    X_centered = Xg - col_mean[None, :]
    row_finite_fraction = np.isfinite(X_centered).mean(axis=1)
    row_valid = row_finite_fraction >= settings.eof_row_min_finite_fraction
    if int(row_valid.sum()) < 5:
        raise ValueError(f"{window}-{object_name}: EOF 可用样本不足：{int(row_valid.sum())}/{len(row_valid)}")
    # Fill remaining missing values by grid mean (zero after centering).
    X_fit = np.where(np.isfinite(X_centered[row_valid]), X_centered[row_valid], 0.0)

    full_weights = _lat_weights(obj_lats, len(obj_lons), settings.eof_weighting)
    weights = full_weights[grid_valid]
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    X_weighted = X_fit * weights[None, :]

    # SVD on weighted anomalies. PC scores are projection onto the first weighted EOF.
    try:
        _, s, vt = np.linalg.svd(X_weighted, full_matrices=False)
    except np.linalg.LinAlgError:
        # A rare fallback to eigen decomposition of covariance, still exact for PC1.
        cov = np.dot(X_weighted.T, X_weighted)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        s = np.sqrt(np.clip(eigvals, 0.0, None))
        vt = eigvecs.T
        notes.append("svd_failed_used_eigh")

    v1 = vt[0, :]
    total_var = float(np.sum(s ** 2))
    explained = float((s[0] ** 2) / total_var) if total_var > 0 else np.nan
    singular_value = float(s[0]) if len(s) else np.nan

    def project(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = _flatten_samples(subfield, days)[:, grid_valid]
        row_frac = np.isfinite(X).mean(axis=1)
        Xc = X - col_mean[None, :]
        Xc = np.where(np.isfinite(Xc), Xc, 0.0)
        scores = np.dot(Xc * weights[None, :], v1)
        scores[row_frac < settings.eof_row_min_finite_fraction] = np.nan
        return scores, row_frac

    train_scores_raw, train_row_frac = project(train_days)
    score_mean = float(np.nanmean(train_scores_raw))
    score_std = float(np.nanstd(train_scores_raw))
    if (not np.isfinite(score_std)) or score_std <= 0:
        score_std = 1.0
        notes.append("score_std_fallback_1")

    ext_scores_raw, ext_row_frac = project(ext_days)
    ext_scores = (ext_scores_raw - score_mean) / score_std
    # ext_scores shape is years*len(ext_days), ordered year-major then day-major.

    # Fix PC1 sign using a representative V1 index anomaly in the target window.
    sign_ref = settings.sign_reference_variables.get(object_name, "")
    sign_corr = np.nan
    sign_flipped = False
    if sign_ref and sign_ref in index_df.columns:
        # Compare target-window PC scores against same-window reference index anomalies.
        target_mask_days = np.isin(ext_days, train_days)
        pc_matrix = ext_scores.reshape(len(years), len(ext_days))[:, target_mask_days]
        ref_sub = index_df[index_df["day"].isin(train_days)][["year", "day", sign_ref]].copy()
        idx = pd.MultiIndex.from_product([years, train_days], names=["year", "day"])
        ref_vals = ref_sub.set_index(["year", "day"]).reindex(idx)[sign_ref].to_numpy(dtype=float)
        r, _ = safe_corr_1d(pc_matrix.reshape(-1), ref_vals)
        sign_corr = r
        if np.isfinite(r) and r < 0:
            ext_scores = -ext_scores
            train_scores_raw = -train_scores_raw
            v1 = -v1
            sign_corr = -r
            sign_flipped = True
    else:
        notes.append("missing_sign_reference")

    loading_valid_unweighted = np.full((n_grid_total,), np.nan, dtype=np.float64)
    loading_valid_unweighted[grid_valid] = v1 / weights
    loading_grid = loading_valid_unweighted.reshape(len(obj_lats), len(obj_lons))

    missing_fraction = 1.0 - float(np.isfinite(X_train[:, grid_valid]).sum() / X_train[:, grid_valid].size)
    qflag = _make_quality_flag(explained, sign_corr, settings, missing_fraction)
    model = EOFModel(
        object_name=object_name,
        window=window,
        field_name="PC1",
        train_days=np.asarray(train_days, dtype=int),
        ext_days=np.asarray(ext_days, dtype=int),
        lat=np.asarray(obj_lats, dtype=float),
        lon=np.asarray(obj_lons, dtype=float),
        grid_valid_mask=grid_valid.reshape(len(obj_lats), len(obj_lons)),
        weighted_eof_vector=v1,
        loading_unweighted=loading_grid,
        col_mean=col_mean,
        weights=weights,
        score_mean=score_mean,
        score_std=score_std,
        explained_variance_ratio=explained,
        singular_value=singular_value,
        sign_reference_variable=sign_ref,
        sign_reference_corr=sign_corr,
        sign_flipped=sign_flipped,
        quality_flag=qflag,
        notes=";".join(notes),
    )

    score_rows = []
    score_mat = ext_scores.reshape(len(years), len(ext_days))
    train_day_set = set(int(d) for d in train_days)
    for yi, year in enumerate(years):
        for di, day in enumerate(ext_days):
            score_rows.append({
                "year": int(year),
                "day": int(day),
                "window": window,
                "object": object_name,
                "variable": f"{object_name}_PC1",
                "pc1_score": float(score_mat[yi, di]) if np.isfinite(score_mat[yi, di]) else np.nan,
                "is_target_window_day": int(day) in train_day_set,
                "is_padding_day": int(day) not in train_day_set,
            })

    load_rows = []
    for li, la in enumerate(obj_lats):
        for lo_i, lo in enumerate(obj_lons):
            load_rows.append({
                "window": window,
                "object": object_name,
                "lat": float(la),
                "lon": float(lo),
                "loading_unweighted": float(loading_grid[li, lo_i]) if np.isfinite(loading_grid[li, lo_i]) else np.nan,
                "grid_valid_for_eof": bool(model.grid_valid_mask[li, lo_i]),
            })

    return model, pd.DataFrame(score_rows), pd.DataFrame(load_rows)


def build_eof_pc1_panels(
    fields: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    index_df: pd.DataFrame,
    settings: LeadLagScreenV3Settings,
    logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Fit window-wise EOF PC1 for each object and return score/quality/loading tables.
    Also returns lead-lag panels per window: panel[window] shape = (year, ext_day, 5 objects).
    """
    score_tables: List[pd.DataFrame] = []
    quality_rows: List[dict] = []
    loading_tables: List[pd.DataFrame] = []
    panels: Dict[str, np.ndarray] = {}

    n_day_total = next(iter(fields.values())).shape[1]
    all_days = np.arange(1, n_day_total + 1, dtype=int)
    variables = [f"{obj}_PC1" for obj in settings.objects]

    # Pre-extract object subfields once.
    subfields = {}
    object_lats = {}
    object_lons = {}
    for obj in settings.objects:
        sub, lats, lons = extract_object_subfield(fields, lat, lon, obj)
        subfields[obj] = sub
        object_lats[obj] = lats
        object_lons[obj] = lons
        logger.info("Object %s subfield shape=%s lat=%s..%s lon=%s..%s", obj, sub.shape, lats.min(), lats.max(), lons.min(), lons.max())

    loadings_npz = {}
    for wi, (window, (w_start, w_end)) in enumerate(settings.windows.items(), start=1):
        logger.info("[%d/%d] Fitting window-wise EOF-PC1: %s day %d-%d", wi, len(settings.windows), window, w_start, w_end)
        train_days = np.arange(w_start, w_end + 1, dtype=int)
        ext_start = max(int(all_days.min()), w_start - settings.diagnostic_max_lag)
        ext_end = min(int(all_days.max()), w_end + settings.diagnostic_max_lag)
        ext_days = np.arange(ext_start, ext_end + 1, dtype=int)
        window_score_tables = []

        for obj in settings.objects:
            model, score_df, loading_df = _fit_pc1_for_object_window(
                subfield=subfields[obj],
                obj_lats=object_lats[obj],
                obj_lons=object_lons[obj],
                object_name=obj,
                window=window,
                train_days=train_days,
                ext_days=ext_days,
                years=years,
                index_df=index_df,
                settings=settings,
            )
            quality_rows.append({
                "window": window,
                "object": obj,
                "variable": f"{obj}_PC1",
                "n_years": int(len(years)),
                "n_days_train": int(len(train_days)),
                "n_days_extended": int(len(ext_days)),
                "n_samples_train": int(len(years) * len(train_days)),
                "n_gridpoints_total": int(model.grid_valid_mask.size),
                "n_gridpoints_used": int(model.grid_valid_mask.sum()),
                "grid_used_fraction": float(model.grid_valid_mask.mean()),
                "pc1_explained_variance_ratio": model.explained_variance_ratio,
                "pc1_singular_value": model.singular_value,
                "eof_weighting": settings.eof_weighting,
                "sign_reference_variable": model.sign_reference_variable,
                "sign_reference_corr_abs_after_flip": model.sign_reference_corr,
                "sign_flipped": model.sign_flipped,
                "pc1_score_train_mean": model.score_mean,
                "pc1_score_train_std": model.score_std,
                "quality_flag": model.quality_flag,
                "notes": model.notes,
            })
            window_score_tables.append(score_df)
            score_tables.append(score_df)
            loading_tables.append(loading_df)
            loadings_npz[f"{window}_{obj}_loading_unweighted"] = model.loading_unweighted
            loadings_npz[f"{window}_{obj}_grid_valid_mask"] = model.grid_valid_mask.astype(np.int8)
            loadings_npz[f"{window}_{obj}_lat"] = model.lat
            loadings_npz[f"{window}_{obj}_lon"] = model.lon

        window_scores = pd.concat(window_score_tables, ignore_index=True)
        panels[window] = build_panel_from_scores(window_scores, variables, years, ext_days)
        # Store ext days separately under a naming convention used by the lead-lag core.
        loadings_npz[f"{window}_ext_days"] = ext_days
        loadings_npz[f"{window}_target_days"] = train_days

    score_all = pd.concat(score_tables, ignore_index=True)
    quality = pd.DataFrame(quality_rows)
    loadings_long = pd.concat(loading_tables, ignore_index=True)
    return score_all, quality, loadings_long, {"panels": panels, "loadings_npz": loadings_npz}
