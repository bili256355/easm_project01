from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .eof_reduce import EOFWindowModel
from .settings import LeadLagScreenV4Settings


@dataclass
class CCAFit:
    train_r: float
    x_weights: np.ndarray
    y_weights: np.ndarray
    x_mean: np.ndarray
    y_mean: np.ndarray
    kx: int
    ky: int
    status: str


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    sx = float(np.nanstd(xv))
    sy = float(np.nanstd(yv))
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def _clean_xy(X: np.ndarray, Y: np.ndarray, min_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    row_ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    Xc = X[row_ok]
    Yc = Y[row_ok]
    if Xc.shape[0] < min_pairs:
        return np.empty((0, X.shape[1])), np.empty((0, Y.shape[1]))
    return Xc, Yc


def fit_first_cca(X: np.ndarray, Y: np.ndarray, settings: LeadLagScreenV4Settings) -> CCAFit:
    Xc, Yc = _clean_xy(X, Y, settings.min_pairs)
    if Xc.shape[0] < settings.min_pairs:
        return CCAFit(np.nan, np.full((X.shape[1],), np.nan), np.full((Y.shape[1],), np.nan), np.zeros(X.shape[1]), np.zeros(Y.shape[1]), X.shape[1], Y.shape[1], "insufficient_pairs")
    x_mean = np.nanmean(Xc, axis=0)
    y_mean = np.nanmean(Yc, axis=0)
    X0 = Xc - x_mean[None, :]
    Y0 = Yc - y_mean[None, :]
    # Drop zero-variance columns after centering.
    x_std = np.nanstd(X0, axis=0)
    y_std = np.nanstd(Y0, axis=0)
    x_keep = np.isfinite(x_std) & (x_std > 1e-12)
    y_keep = np.isfinite(y_std) & (y_std > 1e-12)
    if x_keep.sum() == 0 or y_keep.sum() == 0:
        return CCAFit(np.nan, np.full((X.shape[1],), np.nan), np.full((Y.shape[1],), np.nan), x_mean, y_mean, X.shape[1], Y.shape[1], "zero_variance")
    X1 = X0[:, x_keep]
    Y1 = Y0[:, y_keep]
    n = X1.shape[0]
    ridge = float(settings.ridge)
    Sxx = (X1.T @ X1) / max(n - 1, 1) + ridge * np.eye(X1.shape[1])
    Syy = (Y1.T @ Y1) / max(n - 1, 1) + ridge * np.eye(Y1.shape[1])
    Sxy = (X1.T @ Y1) / max(n - 1, 1)

    def inv_sqrt(S: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(S)
        vals = np.clip(vals, ridge, None)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    try:
        Wx = inv_sqrt(Sxx)
        Wy = inv_sqrt(Syy)
        M = Wx @ Sxy @ Wy
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        a_reduced = Wx @ U[:, 0]
        b_reduced = Wy @ Vt.T[:, 0]
        x_scores = X1 @ a_reduced
        y_scores = Y1 @ b_reduced
        r = _corr(x_scores, y_scores)
        if np.isfinite(r) and r < 0:
            b_reduced = -b_reduced
            r = -r
        a = np.full((X.shape[1],), 0.0)
        b = np.full((Y.shape[1],), 0.0)
        a[x_keep] = a_reduced
        b[y_keep] = b_reduced
        return CCAFit(float(r) if np.isfinite(r) else np.nan, a, b, x_mean, y_mean, X.shape[1], Y.shape[1], "ok")
    except np.linalg.LinAlgError as exc:
        return CCAFit(np.nan, np.full((X.shape[1],), np.nan), np.full((Y.shape[1],), np.nan), x_mean, y_mean, X.shape[1], Y.shape[1], f"linalg_error:{exc}")


def apply_cca(fit: CCAFit, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    u = (X - fit.x_mean[None, :]) @ fit.x_weights
    v = (Y - fit.y_mean[None, :]) @ fit.y_weights
    return u, v


def _lagged_samples_from_models(
    x_model: EOFWindowModel,
    y_model: EOFWindowModel,
    years: np.ndarray,
    lag: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_day_to_i = {int(d): i for i, d in enumerate(x_model.ext_days)}
    y_day_to_i = {int(d): i for i, d in enumerate(y_model.ext_days)}
    x_idx = []
    y_idx = []
    target_days = []
    for t in y_model.train_days:
        sd = int(t) - int(lag)
        if sd in x_day_to_i and int(t) in y_day_to_i:
            x_idx.append(x_day_to_i[sd])
            y_idx.append(y_day_to_i[int(t)])
            target_days.append(int(t))
    if not target_days:
        return np.empty((0, k)), np.empty((0, k)), np.asarray([], dtype=int)

    # Reproject scores from saved model components.
    # This helper relies on score matrices already reconstructed in model auxiliary attributes? Not stored.
    raise RuntimeError("Internal error: _lagged_samples_from_models requires precomputed score matrices; use build_lagged_samples from score_panels.")


def build_score_panels(score_df: pd.DataFrame, years: np.ndarray, max_k: int) -> Dict[Tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Return (score array years x ext_days x max_k, ext_days) by (window, object)."""
    panels: Dict[Tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for (window, obj), sub in score_df.groupby(["window", "object"], sort=False):
        ext_days = np.sort(sub["day"].unique().astype(int))
        arr = np.full((len(years), len(ext_days), max_k), np.nan, dtype=float)
        year_to_i = {int(y): i for i, y in enumerate(years)}
        day_to_i = {int(d): i for i, d in enumerate(ext_days)}
        for row in sub.itertuples(index=False):
            mode = int(row.mode) - 1
            if mode >= max_k:
                continue
            yi = year_to_i.get(int(row.year))
            di = day_to_i.get(int(row.day))
            if yi is None or di is None:
                continue
            arr[yi, di, mode] = float(row.score) if np.isfinite(row.score) else np.nan
        panels[(str(window), str(obj))] = (arr, ext_days)
    return panels


def make_lagged_xy(
    panels: Dict[Tuple[str, str], tuple[np.ndarray, np.ndarray]],
    settings: LeadLagScreenV4Settings,
    window: str,
    source_object: str,
    target_object: str,
    lag: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xpanel, xdays = panels[(window, source_object)]
    Ypanel, ydays = panels[(window, target_object)]
    x_map = {int(d): i for i, d in enumerate(xdays)}
    y_map = {int(d): i for i, d in enumerate(ydays)}
    start, end = settings.windows[window]
    target_days = []
    x_idx = []
    y_idx = []
    for t in range(start, end + 1):
        sd = t - int(lag)
        if sd in x_map and t in y_map:
            target_days.append(t)
            x_idx.append(x_map[sd])
            y_idx.append(y_map[t])
    if not target_days:
        return np.empty((0, k)), np.empty((0, k)), np.asarray([], dtype=int), np.asarray([], dtype=int)
    X = Xpanel[:, x_idx, :k].reshape(-1, k)
    Y = Ypanel[:, y_idx, :k].reshape(-1, k)
    sample_years = np.repeat(np.arange(Xpanel.shape[0]), len(target_days))
    sample_target_days = np.tile(np.asarray(target_days, dtype=int), Xpanel.shape[0])
    return X, Y, sample_years, sample_target_days


def _cv_folds_by_year(n_years: int, n_folds: int) -> List[np.ndarray]:
    folds: List[np.ndarray] = []
    indices = np.arange(n_years)
    for f in range(min(n_folds, n_years)):
        folds.append(indices[f::min(n_folds, n_years)])
    return [fold for fold in folds if len(fold) > 0]


def cross_validated_cca_r(
    X: np.ndarray,
    Y: np.ndarray,
    sample_year_index: np.ndarray,
    settings: LeadLagScreenV4Settings,
) -> tuple[float, float, int, str]:
    n_years = int(np.nanmax(sample_year_index)) + 1 if len(sample_year_index) else 0
    folds = _cv_folds_by_year(n_years, settings.cv_folds)
    u_all = np.full((X.shape[0],), np.nan, dtype=float)
    v_all = np.full((Y.shape[0],), np.nan, dtype=float)
    fold_count = 0
    status_parts = []
    for fold in folds:
        test_mask = np.isin(sample_year_index, fold)
        train_mask = ~test_mask
        Xtr, Ytr = X[train_mask], Y[train_mask]
        Xte, Yte = X[test_mask], Y[test_mask]
        Xtr_clean, Ytr_clean = _clean_xy(Xtr, Ytr, settings.min_pairs)
        Xte_clean, Yte_clean = _clean_xy(Xte, Yte, settings.min_cv_pairs)
        if Xtr_clean.shape[0] < settings.min_pairs or Xte_clean.shape[0] < settings.min_cv_pairs:
            status_parts.append("fold_insufficient")
            continue
        fit = fit_first_cca(Xtr, Ytr, settings)
        if fit.status != "ok":
            status_parts.append(fit.status)
            continue
        u, v = apply_cca(fit, Xte, Yte)
        u_all[test_mask] = u
        v_all[test_mask] = v
        fold_count += 1
    cv_r_signed = _corr(u_all, v_all)
    cv_abs = abs(cv_r_signed) if np.isfinite(cv_r_signed) else np.nan
    status = "ok" if fold_count >= 2 else "insufficient_cv_folds"
    if status_parts:
        status += ";" + ";".join(sorted(set(status_parts)))
    return cv_r_signed, cv_abs, fold_count, status


def run_cca_lags(
    panels: Dict[Tuple[str, str], tuple[np.ndarray, np.ndarray]],
    years: np.ndarray,
    settings: LeadLagScreenV4Settings,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lag_rows: List[dict] = []
    perm_rows: List[dict] = []
    boot_rows: List[dict] = []
    pattern_rows: List[dict] = []
    rng = np.random.default_rng(settings.random_seed)

    for k in settings.eof_k_values:
        logger.info("Running lagged CCA for k=%d", k)
        for window in settings.windows:
            for source_object, target_object in settings.core_pairs:
                pair_label = f"{source_object}→{target_object}"
                observed_by_lag: Dict[int, dict] = {}
                for lag in range(0, settings.max_lag + 1):
                    X, Y, sample_year_idx, sample_days = make_lagged_xy(panels, settings, window, source_object, target_object, lag, k)
                    Xc, Yc = _clean_xy(X, Y, settings.min_pairs)
                    n_samples = int(Xc.shape[0])
                    fit = fit_first_cca(X, Y, settings)
                    if fit.status == "ok":
                        u, v = apply_cca(fit, X, Y)
                        in_sample_r = _corr(u, v)
                    else:
                        in_sample_r = np.nan
                    cv_r, cv_abs, cv_fold_count, cv_status = cross_validated_cca_r(X, Y, sample_year_idx, settings)
                    row = {
                        "window": window,
                        "source_object": source_object,
                        "target_object": target_object,
                        "pair_direction": pair_label,
                        "k_eof": int(k),
                        "lag": int(lag),
                        "n_samples": n_samples,
                        "n_years": int(len(years)),
                        "train_canonical_r": float(in_sample_r) if np.isfinite(in_sample_r) else np.nan,
                        "abs_train_canonical_r": abs(float(in_sample_r)) if np.isfinite(in_sample_r) else np.nan,
                        "cv_canonical_r": float(cv_r) if np.isfinite(cv_r) else np.nan,
                        "abs_cv_canonical_r": float(cv_abs) if np.isfinite(cv_abs) else np.nan,
                        "cv_fold_count": int(cv_fold_count),
                        "cca_status": fit.status,
                        "cv_status": cv_status,
                        "sample_status": "ok" if n_samples >= settings.min_pairs else "insufficient_pairs",
                    }
                    lag_rows.append(row)
                    observed_by_lag[lag] = {"row": row, "fit": fit, "X": X, "Y": Y, "sample_year_idx": sample_year_idx}

                # Pair-window-k permutation audit: max abs in-sample canonical r over lag=0..5.
                obs_max = np.nanmax([observed_by_lag[l]["row"]["abs_train_canonical_r"] for l in observed_by_lag])
                perm_max = np.full((settings.n_permutations,), np.nan, dtype=float)
                for b in range(settings.n_permutations):
                    year_perm = rng.permutation(len(years))
                    vals = []
                    for lag in observed_by_lag:
                        X = observed_by_lag[lag]["X"]
                        Y = observed_by_lag[lag]["Y"].copy()
                        syear = observed_by_lag[lag]["sample_year_idx"]
                        # Permute target blocks by year while preserving within-year day structure.
                        Yp = Y.copy()
                        for yi in range(len(years)):
                            dst = syear == yi
                            src = syear == year_perm[yi]
                            if dst.sum() == src.sum():
                                Yp[dst] = Y[src]
                        fitp = fit_first_cca(X, Yp, settings)
                        vals.append(abs(fitp.train_r) if np.isfinite(fitp.train_r) else np.nan)
                    perm_max[b] = np.nanmax(vals) if vals else np.nan
                p_perm = (1.0 + float(np.nansum(perm_max >= obs_max))) / (1.0 + settings.n_permutations) if np.isfinite(obs_max) else np.nan
                perm_rows.append({
                    "window": window,
                    "source_object": source_object,
                    "target_object": target_object,
                    "pair_direction": pair_label,
                    "k_eof": int(k),
                    "observed_max_abs_train_r_lag0_5": float(obs_max) if np.isfinite(obs_max) else np.nan,
                    "perm_p_max_train_r": p_perm,
                    "n_permutations": int(settings.n_permutations),
                    "permutation_mode": settings.permutation_mode,
                    "note": "Permutation uses max abs in-sample canonical r; cross-validated r is the main effect-size diagnostic.",
                })

                # Bootstrap at best CV lag.
                lag_df = pd.DataFrame([observed_by_lag[l]["row"] for l in observed_by_lag])
                if lag_df["abs_cv_canonical_r"].notna().any():
                    best_lag = int(lag_df.sort_values(["abs_cv_canonical_r", "abs_train_canonical_r"], ascending=False).iloc[0]["lag"])
                else:
                    best_lag = int(lag_df.sort_values(["abs_train_canonical_r"], ascending=False).iloc[0]["lag"])
                Xbest = observed_by_lag[best_lag]["X"]
                Ybest = observed_by_lag[best_lag]["Y"]
                syear = observed_by_lag[best_lag]["sample_year_idx"]
                boot_vals = np.full((settings.n_bootstrap,), np.nan, dtype=float)
                for b in range(settings.n_bootstrap):
                    sampled_years = rng.integers(0, len(years), size=len(years))
                    mask_idx = []
                    for sy in sampled_years:
                        mask_idx.extend(np.where(syear == sy)[0].tolist())
                    mask_idx = np.asarray(mask_idx, dtype=int)
                    fitb = fit_first_cca(Xbest[mask_idx], Ybest[mask_idx], settings)
                    boot_vals[b] = fitb.train_r if np.isfinite(fitb.train_r) else np.nan
                boot_rows.append({
                    "window": window,
                    "source_object": source_object,
                    "target_object": target_object,
                    "pair_direction": pair_label,
                    "k_eof": int(k),
                    "best_lag_for_bootstrap": int(best_lag),
                    "bootstrap_median_train_r": float(np.nanmedian(boot_vals)) if np.isfinite(boot_vals).any() else np.nan,
                    "bootstrap_q025_train_r": float(np.nanquantile(boot_vals, 0.025)) if np.isfinite(boot_vals).any() else np.nan,
                    "bootstrap_q975_train_r": float(np.nanquantile(boot_vals, 0.975)) if np.isfinite(boot_vals).any() else np.nan,
                    "bootstrap_frac_abs_r_ge_threshold": float(np.nanmean(np.abs(boot_vals) >= settings.bootstrap_abs_r_threshold)) if np.isfinite(boot_vals).any() else np.nan,
                    "bootstrap_abs_r_threshold": float(settings.bootstrap_abs_r_threshold),
                    "n_bootstrap": int(settings.n_bootstrap),
                    "bootstrap_mode": settings.bootstrap_mode,
                })

    lag_long = pd.DataFrame(lag_rows)
    perm_df = pd.DataFrame(perm_rows)
    boot_df = pd.DataFrame(boot_rows)
    pattern_df = pd.DataFrame(pattern_rows)
    return lag_long, perm_df, boot_df, pattern_df


def summarize_cca(lag_long: pd.DataFrame, perm_df: pd.DataFrame, boot_df: pd.DataFrame, settings: LeadLagScreenV4Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (window, src, tgt, k), sub in lag_long.groupby(["window", "source_object", "target_object", "k_eof"], sort=False):
        sub = sub.copy()
        sub_non0 = sub[sub["lag"] > 0]
        best_all = sub.sort_values(["abs_cv_canonical_r", "abs_train_canonical_r"], ascending=False).iloc[0]
        best_lagged = sub_non0.sort_values(["abs_cv_canonical_r", "abs_train_canonical_r"], ascending=False).iloc[0] if len(sub_non0) else best_all
        tau0 = sub[sub["lag"] == 0].iloc[0]
        tau0_abs = float(tau0["abs_cv_canonical_r"]) if np.isfinite(tau0["abs_cv_canonical_r"]) else np.nan
        lag_abs = float(best_lagged["abs_cv_canonical_r"]) if np.isfinite(best_lagged["abs_cv_canonical_r"]) else np.nan
        diff = lag_abs - tau0_abs if np.isfinite(lag_abs) and np.isfinite(tau0_abs) else np.nan
        if np.isfinite(lag_abs) and np.isfinite(tau0_abs):
            if lag_abs >= settings.moderate_abs_cv_r and diff > settings.lag_tau0_close_margin:
                dominance = "lagged_dominant"
            elif tau0_abs >= settings.moderate_abs_cv_r and -diff > settings.lag_tau0_close_margin:
                dominance = "tau0_dominant"
            elif max(lag_abs, tau0_abs) >= settings.moderate_abs_cv_r:
                dominance = "lagged_tau0_close"
            else:
                dominance = "weak_or_unstable"
        else:
            dominance = "insufficient"
        perm_match = perm_df[(perm_df["window"] == window) & (perm_df["source_object"] == src) & (perm_df["target_object"] == tgt) & (perm_df["k_eof"] == k)]
        boot_match = boot_df[(boot_df["window"] == window) & (boot_df["source_object"] == src) & (boot_df["target_object"] == tgt) & (boot_df["k_eof"] == k)]
        p_perm = float(perm_match.iloc[0]["perm_p_max_train_r"]) if len(perm_match) else np.nan
        boot_frac = float(boot_match.iloc[0]["bootstrap_frac_abs_r_ge_threshold"]) if len(boot_match) else np.nan
        if (np.isfinite(p_perm) and p_perm <= settings.p_supported and np.isfinite(max(lag_abs, tau0_abs)) and max(lag_abs, tau0_abs) >= settings.strong_abs_cv_r):
            tier = "CCA_strong_coupling"
        elif (np.isfinite(p_perm) and p_perm <= settings.p_supported and np.isfinite(max(lag_abs, tau0_abs)) and max(lag_abs, tau0_abs) >= settings.moderate_abs_cv_r):
            tier = "CCA_moderate_coupling"
        elif np.isfinite(max(lag_abs, tau0_abs)) and max(lag_abs, tau0_abs) >= settings.moderate_abs_cv_r:
            tier = "CCA_effect_without_permutation_support"
        else:
            tier = "CCA_weak_or_not_supported"
        rows.append({
            "window": window,
            "source_object": src,
            "target_object": tgt,
            "pair_direction": f"{src}→{tgt}",
            "k_eof": int(k),
            "best_lag_all": int(best_all["lag"]),
            "best_abs_cv_r_all": float(best_all["abs_cv_canonical_r"]) if np.isfinite(best_all["abs_cv_canonical_r"]) else np.nan,
            "best_signed_cv_r_all": float(best_all["cv_canonical_r"]) if np.isfinite(best_all["cv_canonical_r"]) else np.nan,
            "best_lagged_lag": int(best_lagged["lag"]),
            "best_lagged_abs_cv_r": lag_abs,
            "best_lagged_signed_cv_r": float(best_lagged["cv_canonical_r"]) if np.isfinite(best_lagged["cv_canonical_r"]) else np.nan,
            "tau0_abs_cv_r": tau0_abs,
            "tau0_signed_cv_r": float(tau0["cv_canonical_r"]) if np.isfinite(tau0["cv_canonical_r"]) else np.nan,
            "lagged_minus_tau0_abs_cv_r": diff,
            "cca_time_structure_label": dominance,
            "perm_p_max_train_r": p_perm,
            "bootstrap_frac_abs_r_ge_threshold": boot_frac,
            "cca_evidence_tier": tier,
            "interpretation_boundary": "CCA coupling-mode audit; not pathway proof; direction comes from X(t-lag) vs Y(t) design and remains non-causal.",
        })
    summary = pd.DataFrame(rows)
    main = summary[summary["k_eof"] == max(settings.eof_k_values)].copy()
    if main.empty:
        main = summary.copy()
    return summary, main
