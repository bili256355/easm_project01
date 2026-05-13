from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from lead_lag_screen_v3.data_io import extract_object_subfield, make_directed_object_pairs
from lead_lag_screen_v3.eof_pc1 import _day_indices  # project convention helper
from lead_lag_screen_v3.lead_lag_core import _max_stat_from_corrs, _pair_indices
from lead_lag_screen_v3.settings import LeadLagScreenV3Settings
from lead_lag_screen_v3.stats_utils import safe_corr_1d

from .settings_b import StabilitySettings


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


def _flatten_window_samples(subfield: np.ndarray, days: np.ndarray, year_idx: np.ndarray | None = None) -> np.ndarray:
    day_idx = _day_indices(days)
    arr = subfield if year_idx is None else subfield[year_idx, :, :, :]
    out = arr[:, day_idx, :, :]
    return out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])


def _weighted_pc1_fit_from_matrix(
    X_train: np.ndarray,
    weights_full: np.ndarray,
    observed_grid_mask: np.ndarray | None = None,
    min_grid_fraction: float = 0.70,
    min_row_fraction: float = 0.50,
) -> dict:
    """Fit weighted PC1 on flattened field samples.

    If observed_grid_mask is supplied, use that fixed mask so bootstrap loadings are
    directly comparable with the observed PC1 loading.
    """
    if observed_grid_mask is None:
        finite_grid_fraction = np.isfinite(X_train).mean(axis=0)
        grid_mask = finite_grid_fraction >= min_grid_fraction
    else:
        grid_mask = np.asarray(observed_grid_mask, dtype=bool).reshape(-1)
    if int(grid_mask.sum()) < 3:
        return {"ok": False, "reason": "insufficient_grid"}

    Xg = X_train[:, grid_mask]
    col_mean = np.nanmean(Xg, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    Xc = Xg - col_mean[None, :]
    row_frac = np.isfinite(Xc).mean(axis=1)
    row_mask = row_frac >= min_row_fraction
    if int(row_mask.sum()) < 5:
        return {"ok": False, "reason": "insufficient_rows"}
    Xfit = np.where(np.isfinite(Xc[row_mask]), Xc[row_mask], 0.0)
    weights = weights_full[grid_mask]
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    Xw = Xfit * weights[None, :]
    try:
        _, s, vt = np.linalg.svd(Xw, full_matrices=False)
    except np.linalg.LinAlgError:
        cov = np.dot(Xw.T, Xw)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        s = np.sqrt(np.clip(eigvals, 0.0, None))
        vt = eigvecs.T
    if len(s) == 0 or not np.isfinite(s[0]):
        return {"ok": False, "reason": "svd_failed"}
    total_var = float(np.sum(s ** 2))
    explained = float((s[0] ** 2) / total_var) if total_var > 0 else np.nan
    v1 = vt[0, :]
    loading_full = np.full(X_train.shape[1], np.nan, dtype=float)
    loading_full[grid_mask] = v1 / weights
    return {
        "ok": True,
        "grid_mask": grid_mask,
        "weights": weights,
        "col_mean": col_mean,
        "weighted_eof_vector": v1,
        "loading_full": loading_full,
        "explained_variance_ratio": explained,
        "singular_value": float(s[0]),
    }


def _project_scores_with_fit(X_samples: np.ndarray, fit: dict, score_ref: np.ndarray | None = None) -> np.ndarray:
    grid_mask = fit["grid_mask"]
    Xg = X_samples[:, grid_mask]
    Xc = Xg - fit["col_mean"][None, :]
    Xc = np.where(np.isfinite(Xc), Xc, 0.0)
    scores = np.dot(Xc * fit["weights"][None, :], fit["weighted_eof_vector"])
    if np.isfinite(scores).sum() >= 3:
        scores = (scores - np.nanmean(scores)) / (np.nanstd(scores) if np.nanstd(scores) > 0 else 1.0)
    if score_ref is not None:
        r, _ = safe_corr_1d(scores, score_ref)
        if np.isfinite(r) and r < 0:
            scores = -scores
    return scores


def _mode_label(row: dict, stability: StabilitySettings) -> str:
    med = row.get("bootstrap_loading_abs_corr_median", np.nan)
    p10 = row.get("bootstrap_loading_abs_corr_p10", np.nan)
    sign = row.get("bootstrap_loading_sign_consistency", np.nan)
    score_med = row.get("bootstrap_score_corr_median", np.nan)
    if (
        np.isfinite(med) and med >= stability.pc1_loading_corr_stable_median
        and np.isfinite(p10) and p10 >= stability.pc1_loading_corr_stable_p10
        and np.isfinite(sign) and sign >= stability.pc1_sign_consistency_stable
        and (not np.isfinite(score_med) or score_med >= stability.pc1_score_stability_stable)
    ):
        return "pc1_mode_stable"
    if (
        np.isfinite(med) and med >= stability.pc1_loading_corr_moderate_median
        and np.isfinite(sign) and sign >= stability.pc1_sign_consistency_moderate
        and (not np.isfinite(score_med) or score_med >= stability.pc1_score_stability_moderate)
    ):
        return "pc1_mode_moderate"
    return "pc1_mode_unstable"


def compute_pc1_mode_stability(
    fields: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    years: np.ndarray,
    score_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    loadings_long: pd.DataFrame,
    base_settings: LeadLagScreenV3Settings,
    stability: StabilitySettings,
    rng: np.random.Generator,
    logger,
) -> pd.DataFrame:
    rows: List[dict] = []
    n_year = len(years)
    all_days_total = next(iter(fields.values())).shape[1]

    for window, (w_start, w_end) in base_settings.windows.items():
        train_days = np.arange(w_start, w_end + 1, dtype=int)
        for obj in base_settings.objects:
            logger.info("  PC1 mode stability bootstrap: %s %s n=%d", window, obj, stability.n_pc1_mode_bootstrap)
            subfield, obj_lat, obj_lon = extract_object_subfield(fields, lat, lon, obj)
            obs_loading_rows = loadings_long[(loadings_long["window"] == window) & (loadings_long["object"] == obj)].sort_values(["lat", "lon"])
            obs_loading = obs_loading_rows["loading_unweighted"].to_numpy(dtype=float)
            obs_mask = np.isfinite(obs_loading)
            weights_full = _lat_weights(obj_lat, len(obj_lon), base_settings.eof_weighting)
            X_obs_train = _flatten_window_samples(subfield, train_days)
            fit_obs = _weighted_pc1_fit_from_matrix(
                X_obs_train,
                weights_full=weights_full,
                observed_grid_mask=obs_mask,
                min_grid_fraction=base_settings.eof_grid_min_finite_fraction,
                min_row_fraction=base_settings.eof_row_min_finite_fraction,
            )
            # Use existing scores as the observed reference, because they include the same sign convention as V3_a.
            obs_score_vec = score_df[
                (score_df["window"] == window)
                & (score_df["object"] == obj)
                & (score_df["is_target_window_day"] == True)
            ].sort_values(["year", "day"])["pc1_score"].to_numpy(dtype=float)

            abs_corrs = []
            signed_positive = []
            score_corrs = []
            var_ratios = []
            ok_count = 0
            fail_count = 0
            for _ in range(stability.n_pc1_mode_bootstrap):
                yi = rng.integers(0, n_year, size=n_year)
                X_boot = _flatten_window_samples(subfield, train_days, year_idx=yi)
                fit = _weighted_pc1_fit_from_matrix(
                    X_boot,
                    weights_full=weights_full,
                    observed_grid_mask=obs_mask,
                    min_grid_fraction=base_settings.eof_grid_min_finite_fraction,
                    min_row_fraction=base_settings.eof_row_min_finite_fraction,
                )
                if not fit.get("ok", False):
                    fail_count += 1
                    continue
                boot_loading = fit["loading_full"]
                r, _ = safe_corr_1d(obs_loading, boot_loading)
                if not np.isfinite(r):
                    fail_count += 1
                    continue
                ok_count += 1
                abs_corrs.append(abs(r))
                signed_positive.append(float(r >= 0))
                # Score stability on original target-window samples; sign-align score to observed score.
                boot_scores = _project_scores_with_fit(X_obs_train, fit, score_ref=obs_score_vec)
                sr, _ = safe_corr_1d(obs_score_vec, boot_scores)
                if np.isfinite(sr):
                    score_corrs.append(abs(sr))
                var_ratios.append(fit.get("explained_variance_ratio", np.nan))

            qrow = quality_df[(quality_df["window"] == window) & (quality_df["object"] == obj)]
            qdict = qrow.iloc[0].to_dict() if len(qrow) else {}
            row = {
                "window": window,
                "object": obj,
                "variable": f"{obj}_PC1",
                "n_bootstrap_requested": int(stability.n_pc1_mode_bootstrap),
                "n_bootstrap_success": int(ok_count),
                "n_bootstrap_failed": int(fail_count),
                "pc1_explained_variance_ratio": qdict.get("pc1_explained_variance_ratio", np.nan),
                "pc1_singular_value": qdict.get("pc1_singular_value", np.nan),
                "quality_flag_v3a": qdict.get("quality_flag", ""),
                "sign_reference_corr_abs_after_flip": qdict.get("sign_reference_corr_abs_after_flip", np.nan),
                "bootstrap_loading_abs_corr_median": float(np.nanmedian(abs_corrs)) if abs_corrs else np.nan,
                "bootstrap_loading_abs_corr_p10": float(np.nanpercentile(abs_corrs, 10)) if abs_corrs else np.nan,
                "bootstrap_loading_abs_corr_p25": float(np.nanpercentile(abs_corrs, 25)) if abs_corrs else np.nan,
                "bootstrap_loading_abs_corr_p90": float(np.nanpercentile(abs_corrs, 90)) if abs_corrs else np.nan,
                "bootstrap_loading_sign_consistency": float(np.nanmean(signed_positive)) if signed_positive else np.nan,
                "bootstrap_score_corr_median": float(np.nanmedian(score_corrs)) if score_corrs else np.nan,
                "bootstrap_score_corr_p10": float(np.nanpercentile(score_corrs, 10)) if score_corrs else np.nan,
                "bootstrap_explained_variance_median": float(np.nanmedian(var_ratios)) if var_ratios else np.nan,
            }
            row["pc1_mode_stability_label"] = _mode_label(row, stability)
            rows.append(row)
    return pd.DataFrame(rows)


def _peak_lag_stats_from_panel(panel: np.ndarray, ext_days: np.ndarray, target_days: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray, lags: List[int], min_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_pos, peak_lag, abs_by_lag) for panel or bootstrapped panel."""
    lag_values = []
    for lag in lags:
        vals = _max_stat_from_corrs(panel, ext_days, target_days, [lag], src_idx, tgt_idx, min_pairs)
        if vals.ndim == 1:
            vals = vals[None, :]
        lag_values.append(vals)
    arr = np.stack(lag_values, axis=1)  # rep x lag x pair
    with np.errstate(all="ignore"):
        best_idx = np.nanargmax(np.where(np.isfinite(arr), arr, -np.inf), axis=1)
        best_val = np.take_along_axis(arr, best_idx[:, None, :], axis=1)[:, 0, :]
    peak_lag = np.asarray(lags, dtype=int)[best_idx]
    best_val[~np.isfinite(best_val)] = np.nan
    return best_val, peak_lag, arr


def _mode_and_entropy(values: Iterable[float | int]) -> tuple[float, float, float]:
    arr = [int(v) for v in values if np.isfinite(v)]
    if not arr:
        return np.nan, np.nan, np.nan
    c = Counter(arr)
    mode, count = c.most_common(1)[0]
    frac = count / len(arr)
    probs = np.asarray(list(c.values()), dtype=float) / len(arr)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return float(mode), float(frac), entropy


def _lag_tau0_label(dobs: float, lo: float, hi: float, pgt: float, plt: float, stability: StabilitySettings) -> str:
    if np.isfinite(dobs) and np.isfinite(lo) and lo > 0 and np.isfinite(pgt) and pgt >= stability.stable_probability_cutoff:
        return "stable_lag_dominant"
    if np.isfinite(dobs) and np.isfinite(hi) and hi < 0 and np.isfinite(plt) and plt >= stability.stable_probability_cutoff:
        return "stable_tau0_dominant"
    if np.isfinite(dobs) and abs(dobs) <= stability.lag_tau0_close_margin:
        return "lag_tau0_close_uncertain"
    if np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi:
        return "mixed_lag_tau0_unstable"
    return "lag_tau0_unclassified"


def _forward_reverse_label(dobs: float, lo: float, hi: float, pgt: float, plt: float, stability: StabilitySettings) -> str:
    if np.isfinite(dobs) and np.isfinite(lo) and lo > 0 and np.isfinite(pgt) and pgt >= stability.stable_probability_cutoff:
        return "stable_forward_over_reverse"
    if np.isfinite(dobs) and np.isfinite(hi) and hi < 0 and np.isfinite(plt) and plt >= stability.stable_probability_cutoff:
        return "reverse_competitive"
    if np.isfinite(dobs) and abs(dobs) <= stability.forward_reverse_close_margin:
        return "forward_reverse_close"
    if np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi:
        return "mixed_forward_reverse_unstable"
    return "forward_reverse_unclassified"


def _peak_label(frac: float, stability: StabilitySettings) -> str:
    if not np.isfinite(frac):
        return "peak_lag_not_evaluable"
    if frac >= stability.peak_lag_stable_mode_fraction:
        return "peak_lag_stable"
    if frac >= stability.peak_lag_moderate_mode_fraction:
        return "peak_lag_moderate"
    return "peak_lag_unstable"


def compute_relation_stability(
    panels: Dict[str, np.ndarray],
    years: np.ndarray,
    loadings_npz: Dict[str, np.ndarray],
    base_settings: LeadLagScreenV3Settings,
    stability: StabilitySettings,
    rng: np.random.Generator,
    logger,
) -> Dict[str, pd.DataFrame]:
    variables = [f"{obj}_PC1" for obj in base_settings.objects]
    pair_df = make_directed_object_pairs(base_settings.objects)
    src_idx, tgt_idx = _pair_indices(pair_df, variables)
    pos_lags = list(range(1, base_settings.max_lag + 1))
    neg_lags = list(range(-base_settings.max_lag, 0))
    zero_lags = [0]
    n_pair = len(pair_df)
    n_year = len(years)

    lag_tau0_rows: List[dict] = []
    fwd_rev_rows: List[dict] = []
    peak_rows: List[dict] = []

    for window, panel in panels.items():
        logger.info("Relation stability bootstrap: %s n=%d", window, stability.n_relation_bootstrap)
        ext_days = np.asarray(loadings_npz[f"{window}_ext_days"], dtype=int)
        target_days = np.asarray(loadings_npz[f"{window}_target_days"], dtype=int)
        Tpos_obs = _max_stat_from_corrs(panel, ext_days, target_days, pos_lags, src_idx, tgt_idx, base_settings.min_pairs)
        Tneg_obs = _max_stat_from_corrs(panel, ext_days, target_days, neg_lags, src_idx, tgt_idx, base_settings.min_pairs)
        T0_obs = _max_stat_from_corrs(panel, ext_days, target_days, zero_lags, src_idx, tgt_idx, base_settings.min_pairs)
        _, peak_lag_obs_mat, _ = _peak_lag_stats_from_panel(panel, ext_days, target_days, src_idx, tgt_idx, pos_lags, base_settings.min_pairs)
        peak_lag_obs = peak_lag_obs_mat[0, :]

        Dp0 = np.empty((stability.n_relation_bootstrap, n_pair), dtype=float)
        Dpn = np.empty((stability.n_relation_bootstrap, n_pair), dtype=float)
        peak_lag_boot = np.empty((stability.n_relation_bootstrap, n_pair), dtype=float)

        start = 0
        while start < stability.n_relation_bootstrap:
            stop = min(start + stability.bootstrap_chunk_size, stability.n_relation_bootstrap)
            yi = rng.integers(0, n_year, size=(stop - start, n_year))
            boot = panel[yi, :, :]
            Tpos_b, peak_b, _ = _peak_lag_stats_from_panel(boot, ext_days, target_days, src_idx, tgt_idx, pos_lags, base_settings.min_pairs)
            Tneg_b = _max_stat_from_corrs(boot, ext_days, target_days, neg_lags, src_idx, tgt_idx, base_settings.min_pairs)
            T0_b = _max_stat_from_corrs(boot, ext_days, target_days, zero_lags, src_idx, tgt_idx, base_settings.min_pairs)
            Dp0[start:stop] = Tpos_b - T0_b
            Dpn[start:stop] = Tpos_b - Tneg_b
            peak_lag_boot[start:stop] = peak_b
            start = stop

        for i, pair in pair_df.iterrows():
            d0_obs = float(Tpos_obs[i] - T0_obs[i]) if np.isfinite(Tpos_obs[i]) and np.isfinite(T0_obs[i]) else np.nan
            dn_obs = float(Tpos_obs[i] - Tneg_obs[i]) if np.isfinite(Tpos_obs[i]) and np.isfinite(Tneg_obs[i]) else np.nan
            d0 = Dp0[:, i]
            dn = Dpn[:, i]
            d0_lo90, d0_hi90 = np.nanpercentile(d0, [5, 95]) if np.isfinite(d0).any() else (np.nan, np.nan)
            d0_lo95, d0_hi95 = np.nanpercentile(d0, [2.5, 97.5]) if np.isfinite(d0).any() else (np.nan, np.nan)
            dn_lo90, dn_hi90 = np.nanpercentile(dn, [5, 95]) if np.isfinite(dn).any() else (np.nan, np.nan)
            dn_lo95, dn_hi95 = np.nanpercentile(dn, [2.5, 97.5]) if np.isfinite(dn).any() else (np.nan, np.nan)
            p_d0_gt = float(np.nanmean(d0 > 0)) if np.isfinite(d0).any() else np.nan
            p_d0_lt = float(np.nanmean(d0 < 0)) if np.isfinite(d0).any() else np.nan
            p_dn_gt = float(np.nanmean(dn > 0)) if np.isfinite(dn).any() else np.nan
            p_dn_lt = float(np.nanmean(dn < 0)) if np.isfinite(dn).any() else np.nan
            lag_label = _lag_tau0_label(d0_obs, d0_lo90, d0_hi90, p_d0_gt, p_d0_lt, stability)
            dir_label = _forward_reverse_label(dn_obs, dn_lo90, dn_hi90, p_dn_gt, p_dn_lt, stability)
            mode_lag, mode_frac, entropy = _mode_and_entropy(peak_lag_boot[:, i])
            peak_label = _peak_label(mode_frac, stability)
            base = {
                "window": window,
                "source": pair["source"],
                "target": pair["target"],
                "source_object": pair["source_object"],
                "target_object": pair["target_object"],
                "family_direction": pair["family_direction"],
                "T_pos_obs_raw": float(Tpos_obs[i]) if np.isfinite(Tpos_obs[i]) else np.nan,
                "T_0_obs_raw": float(T0_obs[i]) if np.isfinite(T0_obs[i]) else np.nan,
                "T_neg_obs_raw": float(Tneg_obs[i]) if np.isfinite(Tneg_obs[i]) else np.nan,
                "n_relation_bootstrap": int(stability.n_relation_bootstrap),
            }
            lag_tau0_rows.append({
                **base,
                "D_pos_0_obs_raw": d0_obs,
                "D_pos_0_bootstrap_mean": float(np.nanmean(d0)) if np.isfinite(d0).any() else np.nan,
                "D_pos_0_CI90_low": float(d0_lo90),
                "D_pos_0_CI90_high": float(d0_hi90),
                "D_pos_0_CI95_low": float(d0_lo95),
                "D_pos_0_CI95_high": float(d0_hi95),
                "P_D_pos_0_gt_0": p_d0_gt,
                "P_D_pos_0_lt_0": p_d0_lt,
                "lag_vs_tau0_label": lag_label,
            })
            fwd_rev_rows.append({
                **base,
                "D_pos_neg_obs_raw": dn_obs,
                "D_pos_neg_bootstrap_mean": float(np.nanmean(dn)) if np.isfinite(dn).any() else np.nan,
                "D_pos_neg_CI90_low": float(dn_lo90),
                "D_pos_neg_CI90_high": float(dn_hi90),
                "D_pos_neg_CI95_low": float(dn_lo95),
                "D_pos_neg_CI95_high": float(dn_hi95),
                "P_D_pos_neg_gt_0": p_dn_gt,
                "P_D_pos_neg_lt_0": p_dn_lt,
                "direction_vs_reverse_label": dir_label,
            })
            peak_rows.append({
                **base,
                "positive_peak_lag_obs": float(peak_lag_obs[i]) if np.isfinite(peak_lag_obs[i]) else np.nan,
                "bootstrap_peak_lag_mode": mode_lag,
                "bootstrap_peak_lag_mode_fraction": mode_frac,
                "bootstrap_peak_lag_entropy": entropy,
                "peak_lag_stability_label": peak_label,
            })

    return {
        "eof_pc1_lag_vs_tau0_stability": pd.DataFrame(lag_tau0_rows),
        "eof_pc1_forward_reverse_stability": pd.DataFrame(fwd_rev_rows),
        "eof_pc1_peak_lag_stability": pd.DataFrame(peak_rows),
    }


def _positive_lag_supported(row: pd.Series, stability: StabilitySettings) -> bool:
    p = row.get("p_pos_surrogate", np.nan)
    q = row.get("q_pos_within_window", np.nan)
    ap = row.get("p_pos_audit_surrogate", np.nan)
    aq = row.get("q_pos_audit_within_window", np.nan)
    main_pass = np.isfinite(p) and p <= stability.p_supported and np.isfinite(q) and q <= stability.q_supported
    audit_pass = np.isfinite(ap) and ap <= stability.audit_p_supported
    # If audit q exists, require it; if absent, audit p is still recorded but judgement is less strict.
    if np.isfinite(aq):
        audit_pass = audit_pass and aq <= stability.audit_q_supported
    return bool(main_pass and audit_pass)


def attach_formal_stability_judgement(
    pair_summary: pd.DataFrame,
    mode_stability: pd.DataFrame,
    lag_tau0: pd.DataFrame,
    fwd_rev: pd.DataFrame,
    peak: pd.DataFrame,
    stability: StabilitySettings,
) -> pd.DataFrame:
    out = pair_summary.copy()
    src_mode = mode_stability[["window", "object", "pc1_mode_stability_label"]].rename(
        columns={"object": "source_object", "pc1_mode_stability_label": "pc1_mode_stability_source"}
    )
    tgt_mode = mode_stability[["window", "object", "pc1_mode_stability_label"]].rename(
        columns={"object": "target_object", "pc1_mode_stability_label": "pc1_mode_stability_target"}
    )
    out = out.merge(src_mode, on=["window", "source_object"], how="left")
    out = out.merge(tgt_mode, on=["window", "target_object"], how="left")
    out = out.merge(lag_tau0[["window", "source", "target", "D_pos_0_obs_raw", "D_pos_0_CI90_low", "D_pos_0_CI90_high", "P_D_pos_0_gt_0", "P_D_pos_0_lt_0", "lag_vs_tau0_label"]], on=["window", "source", "target"], how="left")
    out = out.merge(fwd_rev[["window", "source", "target", "D_pos_neg_obs_raw", "D_pos_neg_CI90_low", "D_pos_neg_CI90_high", "P_D_pos_neg_gt_0", "P_D_pos_neg_lt_0", "direction_vs_reverse_label"]], on=["window", "source", "target"], how="left")
    out = out.merge(peak[["window", "source", "target", "bootstrap_peak_lag_mode", "bootstrap_peak_lag_mode_fraction", "bootstrap_peak_lag_entropy", "peak_lag_stability_label"]], on=["window", "source", "target"], how="left")

    judgements = []
    guards = []
    for _, row in out.iterrows():
        src_ok = row.get("pc1_mode_stability_source", "") != "pc1_mode_unstable"
        tgt_ok = row.get("pc1_mode_stability_target", "") != "pc1_mode_unstable"
        pos_supported = _positive_lag_supported(row, stability)
        lag_label = row.get("lag_vs_tau0_label", "")
        dir_label = row.get("direction_vs_reverse_label", "")
        peak_label = row.get("peak_lag_stability_label", "")
        if not (src_ok and tgt_ok):
            judgements.append("pc1_mode_limited")
            guards.append("source_or_target_PC1_mode_unstable; do not use as strong EOF-PC1 lead-lag evidence")
        elif lag_label == "stable_tau0_dominant":
            judgements.append("stable_tau0_dominant_coupling")
            guards.append("tau0 is stably stronger than positive-lag band; not a stable lagged lead")
        elif dir_label in ("reverse_competitive", "mixed_forward_reverse_unstable", "forward_reverse_close"):
            if pos_supported:
                judgements.append("bidirectional_or_reverse_competitive")
                guards.append("positive-lag support exists but forward direction is not stable against reverse")
            else:
                judgements.append("not_supported_or_reverse_competitive")
                guards.append("do not interpret as stable positive-lag lead")
        elif (
            pos_supported
            and lag_label == "stable_lag_dominant"
            and dir_label == "stable_forward_over_reverse"
            and peak_label in ("peak_lag_stable", "peak_lag_moderate")
        ):
            judgements.append("stable_lagged_lead")
            guards.append("formal V3_b stable positive-lag judgement passed")
        elif pos_supported and lag_label in ("lag_tau0_close_uncertain", "mixed_lag_tau0_unstable", "lag_tau0_unclassified"):
            judgements.append("significant_lagged_but_tau0_coupled")
            guards.append("positive-lag significant, but lag is not stably stronger than tau0")
        elif pos_supported:
            judgements.append("significant_lagged_but_not_stably_classified")
            guards.append("positive-lag significant, but stability labels do not support strong lead interpretation")
        else:
            if row.get("same_day_coupling_flag", False):
                judgements.append("same_day_or_coupled_without_stable_lag")
                guards.append("same-day/coupled signal present, but stable lagged lead not supported")
            else:
                judgements.append("not_supported")
                guards.append("positive-lag support and stability gates not passed")
    out["stability_judgement"] = judgements
    out["interpretation_guardrail"] = guards
    return out


def build_v1_v3b_comparison(v1_output_dir, judged: pd.DataFrame) -> pd.DataFrame:
    pair_path = v1_output_dir / "lead_lag_pair_summary.csv"
    if not pair_path.exists():
        return pd.DataFrame()
    v1 = pd.read_csv(pair_path)
    if "evidence_tier" not in v1.columns:
        return pd.DataFrame()
    tier = v1["evidence_tier"].astype(str)
    keep = tier.str.contains("Tier1a|Tier1b|Tier2", regex=True, na=False)
    sub = v1[keep].copy()
    if "family_direction" not in sub.columns and {"source_family", "target_family"}.issubset(sub.columns):
        sub["family_direction"] = sub["source_family"].astype(str) + "→" + sub["target_family"].astype(str)
    def count_contains(s, pat):
        return int(s.astype(str).str.contains(pat, regex=True, na=False).sum())
    v1_roll = sub.groupby(["window", "family_direction"], dropna=False).agg(
        v1_tier1a_count=("evidence_tier", lambda s: count_contains(s, "Tier1a")),
        v1_tier1b_count=("evidence_tier", lambda s: count_contains(s, "Tier1b")),
        v1_tier2_count=("evidence_tier", lambda s: count_contains(s, "Tier2")),
        v1_tier1_2_count=("evidence_tier", "count"),
        v1_max_positive_abs_r=("positive_peak_abs_r", "max") if "positive_peak_abs_r" in sub.columns else ("evidence_tier", "count"),
        v1_max_lag0_abs_r=("lag0_abs_r", "max") if "lag0_abs_r" in sub.columns else ("evidence_tier", "count"),
    ).reset_index()
    v3 = judged[[
        "window", "family_direction", "stability_judgement", "lag_vs_tau0_label", "direction_vs_reverse_label",
        "peak_lag_stability_label", "pc1_mode_stability_source", "pc1_mode_stability_target",
        "positive_peak_lag", "positive_peak_abs_r", "lag0_abs_r"
    ]].copy()
    out = v1_roll.merge(v3, on=["window", "family_direction"], how="outer")
    def hint(row):
        n = row.get("v1_tier1_2_count", 0)
        j = row.get("stability_judgement", "")
        if pd.isna(n):
            n = 0
        if n > 0 and j == "stable_lagged_lead":
            return "v1_tier1_2_and_v3b_stable_lag"
        if n > 0 and j in ("significant_lagged_but_tau0_coupled", "stable_tau0_dominant_coupling"):
            return "v1_tier1_2_but_v3b_tau0_or_close"
        if n > 0 and str(j).startswith("not"):
            return "v1_tier1_2_but_v3b_not_supported"
        if n == 0 and pd.notna(j) and j != "not_supported":
            return "v3b_signal_without_v1_tier1_2"
        return "both_weak_or_not_compared"
    out["comparison_hint"] = out.apply(hint, axis=1)
    return out
