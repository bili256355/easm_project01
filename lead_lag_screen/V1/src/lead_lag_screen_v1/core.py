\
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_io import build_panel, make_directed_pairs, read_index_anomalies, ensure_dirs
from .settings import LeadLagScreenSettings
from .evidence_tier import build_evidence_tier_outputs
from .stats_utils import (
    corr_matrix_batch,
    estimate_ar1_params,
    estimate_ar1_params_diagnostic,
    fdr_bh,
    fisher_effn_p,
    safe_corr_1d,
)


def _lag_arrays(panel: np.ndarray, ext_days: np.ndarray, target_days: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return source and target arrays for a lag.

    panel shape: (year, ext_day, variable).
    output source/target shape: (year, n_valid_target_days, variable).
    """
    day_to_idx = {int(d): i for i, d in enumerate(ext_days)}
    valid_t = []
    src_idx = []
    tgt_idx = []
    for t in target_days:
        sday = int(t - lag)
        if sday in day_to_idx and int(t) in day_to_idx:
            valid_t.append(int(t))
            src_idx.append(day_to_idx[sday])
            tgt_idx.append(day_to_idx[int(t)])
    if not valid_t:
        empty = np.empty((panel.shape[0], 0, panel.shape[2]), dtype=float)
        return empty, empty, np.asarray([], dtype=int)
    return panel[:, src_idx, :], panel[:, tgt_idx, :], np.asarray(valid_t, dtype=int)


def _build_ar1_audit_tables(
    panel: np.ndarray,
    variables: List[str],
    years: np.ndarray,
    window: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    """
    Build explicit AR(1) parameter audit tables.

    Main null:
        pooled_window_variable_ar1
        One pooled AR(1) parameter set is estimated per window x variable.

    Audit null:
        pooled_phi_yearwise_scale_ar1
        Pooled phi is retained, but each year uses its own mean/std scale.
    """
    pooled_rows = []
    year_rows = []
    params: Dict[str, dict] = {}

    for vi, var in enumerate(variables):
        arr = panel[:, :, vi]
        diag = estimate_ar1_params_diagnostic(arr, clip_limit=0.95)
        mu = float(diag["mu"])
        raw_phi = float(diag["raw_phi_before_clip"])
        phi = float(diag["phi_after_clip"])
        sigma = float(diag["sigma"])
        finite = np.isfinite(arr)
        n_finite = int(finite.sum())
        n_total = int(arr.size)
        nan_fraction = 1.0 - (n_finite / n_total if n_total else np.nan)
        stationary_std = sigma / math.sqrt(max(1.0 - phi * phi, 1e-6))
        phi_clipped_flag = bool(diag["phi_clipped_flag"])

        params[var] = {
            "mu": mu,
            "phi": phi,
            "sigma": sigma,
            "stationary_std": stationary_std,
            "raw_phi_before_clip": raw_phi,
            "phi_clip_amount": float(diag["phi_clip_amount"]),
            "phi_clip_direction": diag["phi_clip_direction"],
            "phi_clip_severity": diag["phi_clip_severity"],
        }

        pooled_rows.append({
            "window": window,
            "variable": var,
            "surrogate_mode": "pooled_window_variable_ar1",
            "n_years": int(panel.shape[0]),
            "n_days": int(panel.shape[1]),
            "n_finite": n_finite,
            "n_pair_for_phi": int(diag["n_pair_for_phi"]),
            "nan_fraction": nan_fraction,
            "mu": mu,
            "raw_phi_before_clip": raw_phi,
            "phi": phi,
            "phi_after_clip": phi,
            "phi_clip_limit": float(diag["phi_clip_limit"]),
            "phi_clipped_flag": phi_clipped_flag,
            "phi_clip_amount": float(diag["phi_clip_amount"]),
            "phi_clip_direction": diag["phi_clip_direction"],
            "phi_clip_severity": diag["phi_clip_severity"],
            "sigma": sigma,
            "stationary_std": stationary_std,
            "ar1_estimation_scope": "window_variable_pooled_across_years",
        })

        for yi, year in enumerate(years):
            yarr = arr[yi, :]
            yfinite = np.isfinite(yarr)
            y_n = int(yfinite.sum())
            if y_n > 0:
                y_mean = float(np.nanmean(yarr))
                y_std = float(np.nanstd(yarr))
            else:
                y_mean = np.nan
                y_std = np.nan
            year_rows.append({
                "window": window,
                "year": int(year),
                "variable": var,
                "audit_surrogate_mode": "pooled_phi_yearwise_scale_ar1",
                "year_n_finite": y_n,
                "year_nan_fraction": 1.0 - (y_n / len(yarr) if len(yarr) else np.nan),
                "year_mean": y_mean,
                "year_std": y_std,
                "pooled_raw_phi_before_clip": raw_phi,
                "pooled_phi_used": phi,
                "pooled_phi_clip_amount": float(diag["phi_clip_amount"]),
                "pooled_phi_clip_severity": diag["phi_clip_severity"],
                "fallback_to_pooled_scale": (not np.isfinite(y_std)) or y_std <= 0,
            })

    return pd.DataFrame(pooled_rows), pd.DataFrame(year_rows), params


def _generate_ar1_surrogates(
    panel: np.ndarray,
    variables: List[str],
    ar1_params_by_var: Dict[str, dict],
    rng: np.random.Generator,
    n_rep: int,
    surrogate_mode: str = "pooled_window_variable_ar1",
) -> np.ndarray:
    """
    Generate independent AR(1) surrogates per variable and year.

    Supported modes
    ---------------
    pooled_window_variable_ar1:
        pooled mu/phi/sigma per window x variable.

    pooled_phi_yearwise_scale_ar1:
        pooled phi per window x variable, but each year uses its own mean/std
        scale. This is an audit null to expose sensitivity to yearwise amplitude
        differences without fitting unstable year-specific phi in short windows.

    NaN mask of the observed panel is preserved.
    Returns shape (n_rep, n_year, n_day, n_var).
    """
    n_year, n_day, n_var = panel.shape
    out = np.empty((n_rep, n_year, n_day, n_var), dtype=float)

    for v, var in enumerate(variables):
        pars = ar1_params_by_var[var]
        pooled_mu = float(pars["mu"])
        phi = float(pars["phi"])
        pooled_sigma = float(pars["sigma"])
        pooled_stationary_std = float(pars["stationary_std"])

        x = np.empty((n_rep, n_year, n_day), dtype=float)

        if surrogate_mode == "pooled_window_variable_ar1":
            init_sigma = pooled_stationary_std
            x[:, :, 0] = pooled_mu + rng.normal(0.0, init_sigma, size=(n_rep, n_year))
            eps = rng.normal(0.0, pooled_sigma, size=(n_rep, n_year, max(n_day - 1, 1)))
            for d in range(1, n_day):
                x[:, :, d] = pooled_mu + phi * (x[:, :, d - 1] - pooled_mu) + eps[:, :, d - 1]

        elif surrogate_mode == "pooled_phi_yearwise_scale_ar1":
            # Use pooled phi but year-specific location and amplitude.
            y_mu = np.empty(n_year, dtype=float)
            y_std = np.empty(n_year, dtype=float)
            for yi in range(n_year):
                arr_y = panel[yi, :, v]
                finite = np.isfinite(arr_y)
                if finite.sum() > 0:
                    y_mu[yi] = float(np.nanmean(arr_y))
                    y_std[yi] = float(np.nanstd(arr_y))
                else:
                    y_mu[yi] = pooled_mu
                    y_std[yi] = pooled_stationary_std
                if (not np.isfinite(y_std[yi])) or y_std[yi] <= 0:
                    y_std[yi] = pooled_stationary_std
                if (not np.isfinite(y_mu[yi])):
                    y_mu[yi] = pooled_mu

            eps_sigma = y_std * math.sqrt(max(1.0 - phi * phi, 1e-6))
            x[:, :, 0] = y_mu[None, :] + rng.normal(0.0, 1.0, size=(n_rep, n_year)) * y_std[None, :]
            eps = rng.normal(0.0, 1.0, size=(n_rep, n_year, max(n_day - 1, 1))) * eps_sigma[None, :, None]
            for d in range(1, n_day):
                x[:, :, d] = y_mu[None, :] + phi * (x[:, :, d - 1] - y_mu[None, :]) + eps[:, :, d - 1]

        else:
            raise ValueError(f"Unsupported surrogate_mode: {surrogate_mode}")

        mask = np.isfinite(panel[:, :, v])
        x[:, ~mask] = np.nan
        out[:, :, :, v] = x

    return out



def _pair_indices(pair_df: pd.DataFrame, variables: List[str]) -> tuple[np.ndarray, np.ndarray]:
    var_to_idx = {v: i for i, v in enumerate(variables)}
    src_idx = pair_df["source"].map(var_to_idx).to_numpy()
    tgt_idx = pair_df["target"].map(var_to_idx).to_numpy()
    return src_idx, tgt_idx


def _matrix_pair_values(mat: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray) -> np.ndarray:
    """
    Extract directed pair values from matrix.
    mat shape can be (V,V) or (B,V,V).
    """
    if mat.ndim == 2:
        return mat[src_idx, tgt_idx]
    return mat[:, src_idx, tgt_idx]


def _observed_curve_for_window(
    panel: np.ndarray,
    ext_days: np.ndarray,
    target_days: np.ndarray,
    lags: List[int],
    pair_df: pd.DataFrame,
    variables: List[str],
    phi_by_var: Dict[str, float],
    min_pairs: int,
) -> pd.DataFrame:
    rows = []
    var_to_idx = {v: i for i, v in enumerate(variables)}
    n_total_years = panel.shape[0]

    for lag in lags:
        source_arr, target_arr, valid_t = _lag_arrays(panel, ext_days, target_days, lag)
        for _, pair in pair_df.iterrows():
            src = pair["source"]
            tgt = pair["target"]
            si = var_to_idx[src]
            ti = var_to_idx[tgt]
            x2 = source_arr[:, :, si] if source_arr.shape[1] else np.empty((panel.shape[0], 0))
            y2 = target_arr[:, :, ti] if target_arr.shape[1] else np.empty((panel.shape[0], 0))
            valid = np.isfinite(x2) & np.isfinite(y2)
            n_pairs = int(valid.sum())
            n_years = int(valid.any(axis=1).sum()) if valid.size else 0
            r, _ = safe_corr_1d(x2.reshape(-1), y2.reshape(-1))
            phi_x = phi_by_var.get(src, 0.0)
            phi_y = phi_by_var.get(tgt, 0.0)
            if n_pairs > 0:
                neff = n_pairs * (1.0 - phi_x * phi_y) / max(1.0 + phi_x * phi_y, 1e-6)
                neff = float(np.clip(neff, 3.0, n_pairs))
            else:
                neff = np.nan
            p_effn = fisher_effn_p(r, neff)
            sample_status = "ok" if n_pairs >= min_pairs else "insufficient_pairs"
            rows.append({
                "source": src,
                "target": tgt,
                "lag": lag,
                "n_pairs": n_pairs,
                "n_years_used": n_years,
                "valid_year_fraction": n_years / n_total_years if n_total_years else np.nan,
                "r": r,
                "signed_r": r,
                "abs_r": abs(r) if np.isfinite(r) else np.nan,
                "eff_n": neff,
                "p_effn": p_effn,
                "same_year_only": True,
                "target_window_rule": True,
                "sample_status": sample_status,
            })
    return pd.DataFrame(rows)


def _max_stat_from_corrs(
    panel_or_sur: np.ndarray,
    ext_days: np.ndarray,
    target_days: np.ndarray,
    lags: List[int],
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    min_pairs: int,
) -> np.ndarray:
    """
    Compute max abs correlation across lags for either observed or batched panels.

    Input shape:
        observed: (year, day, var)
        batched:  (B, year, day, var)
    Return:
        observed: (n_pairs,)
        batched:  (B, n_pairs)
    """
    is_batched = panel_or_sur.ndim == 4
    if is_batched:
        B = panel_or_sur.shape[0]
        acc = np.full((B, len(src_idx)), np.nan, dtype=float)
    else:
        acc = np.full((len(src_idx),), np.nan, dtype=float)

    for lag in lags:
        if is_batched:
            # reshape each bootstrap/surrogate replicate as B x N x V
            per_rep = []
            for b in range(panel_or_sur.shape[0]):
                sx, ty, _ = _lag_arrays(panel_or_sur[b], ext_days, target_days, lag)
                per_rep.append((sx.reshape(-1, sx.shape[-1]), ty.reshape(-1, ty.shape[-1])))
            x = np.stack([p[0] for p in per_rep], axis=0)
            y = np.stack([p[1] for p in per_rep], axis=0)
        else:
            sx, ty, _ = _lag_arrays(panel_or_sur, ext_days, target_days, lag)
            x = sx.reshape(1, -1, sx.shape[-1])
            y = ty.reshape(1, -1, ty.shape[-1])

        corr, count = corr_matrix_batch(x, y, min_count=min_pairs)
        vals = np.abs(_matrix_pair_values(corr, src_idx, tgt_idx))
        if is_batched:
            acc = np.fmax(acc, vals)
        else:
            acc = np.fmax(acc, vals[0])
    return acc


def _surrogate_null_for_window(
    panel: np.ndarray,
    ext_days: np.ndarray,
    target_days: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    variables: List[str],
    ar1_params_by_var: Dict[str, dict],
    n_surrogates: int,
    surrogate_mode: str,
    settings: LeadLagScreenSettings,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n_pairs = len(src_idx)
    pos_lags = list(range(1, settings.max_lag + 1))
    neg_lags = list(range(-settings.max_lag, 0))
    zero_lags = [0]

    T_pos = np.empty((n_surrogates, n_pairs), dtype=float)
    T_neg = np.empty((n_surrogates, n_pairs), dtype=float)
    T_0 = np.empty((n_surrogates, n_pairs), dtype=float)

    start = 0
    while start < n_surrogates:
        stop = min(start + settings.surrogate_chunk_size, n_surrogates)
        n_chunk = stop - start
        sur = _generate_ar1_surrogates(
            panel=panel,
            variables=variables,
            ar1_params_by_var=ar1_params_by_var,
            rng=rng,
            n_rep=n_chunk,
            surrogate_mode=surrogate_mode,
        )
        T_pos[start:stop, :] = _max_stat_from_corrs(
            sur, ext_days, target_days, pos_lags, src_idx, tgt_idx, settings.min_pairs
        )
        T_neg[start:stop, :] = _max_stat_from_corrs(
            sur, ext_days, target_days, neg_lags, src_idx, tgt_idx, settings.min_pairs
        )
        T_0[start:stop, :] = _max_stat_from_corrs(
            sur, ext_days, target_days, zero_lags, src_idx, tgt_idx, settings.min_pairs
        )
        start = stop

    return {"T_pos": T_pos, "T_neg": T_neg, "T_0": T_0}



def _bootstrap_direction_for_window(
    panel: np.ndarray,
    ext_days: np.ndarray,
    target_days: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    null95_pos: np.ndarray,
    null95_neg: np.ndarray,
    null95_0: np.ndarray,
    settings: LeadLagScreenSettings,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n_year = panel.shape[0]
    n_pairs = len(src_idx)
    pos_lags = list(range(1, settings.max_lag + 1))
    neg_lags = list(range(-settings.max_lag, 0))
    zero_lags = [0]

    Dpn = np.empty((settings.n_direction_bootstrap, n_pairs), dtype=float)
    Dp0 = np.empty((settings.n_direction_bootstrap, n_pairs), dtype=float)

    start = 0
    while start < settings.n_direction_bootstrap:
        stop = min(start + settings.bootstrap_chunk_size, settings.n_direction_bootstrap)
        n_chunk = stop - start
        year_idx = rng.integers(0, n_year, size=(n_chunk, n_year))
        boot = panel[year_idx, :, :]  # (B, Y, D, V); panel is 3D, advanced year indexing adds bootstrap dim

        T_pos_b = _max_stat_from_corrs(boot, ext_days, target_days, pos_lags, src_idx, tgt_idx, settings.min_pairs)
        T_neg_b = _max_stat_from_corrs(boot, ext_days, target_days, neg_lags, src_idx, tgt_idx, settings.min_pairs)
        T_0_b = _max_stat_from_corrs(boot, ext_days, target_days, zero_lags, src_idx, tgt_idx, settings.min_pairs)

        Spos_b = T_pos_b - null95_pos[None, :]
        Sneg_b = T_neg_b - null95_neg[None, :]
        S0_b = T_0_b - null95_0[None, :]

        Dpn[start:stop, :] = Spos_b - Sneg_b
        Dp0[start:stop, :] = Spos_b - S0_b
        start = stop

    # Leave-one-year-out direction retained fraction.
    loo_retained = np.full((n_pairs,), np.nan, dtype=float)
    loo_flip_counts = np.full((n_pairs,), np.nan, dtype=float)
    if n_year >= 3:
        vals = []
        for omit in range(n_year):
            sub = np.delete(panel, omit, axis=0)
            T_pos_l = _max_stat_from_corrs(sub, ext_days, target_days, pos_lags, src_idx, tgt_idx, settings.min_pairs)
            T_neg_l = _max_stat_from_corrs(sub, ext_days, target_days, neg_lags, src_idx, tgt_idx, settings.min_pairs)
            Spos_l = T_pos_l - null95_pos
            Sneg_l = T_neg_l - null95_neg
            vals.append(Spos_l - Sneg_l)
        loo = np.vstack(vals)
        loo_retained = np.nanmean(loo > 0, axis=0)
        loo_flip_counts = np.sum(loo <= 0, axis=0)

    return {
        "D_pos_neg": Dpn,
        "D_pos_0": Dp0,
        "loo_retained_fraction": loo_retained,
        "loo_flip_years": loo_flip_counts,
    }


def _classify_rows(null_df: pd.DataFrame, dir_df: pd.DataFrame, settings: LeadLagScreenSettings) -> pd.DataFrame:
    df = null_df.merge(dir_df, on=["window", "source", "target"], how="left")

    statistical_support = []
    strength_tier = []
    direction_label = []
    lead_label = []
    lead_group = []
    same_day_flags = []
    failure_reasons = []
    risk_notes = []
    reverse_suggest = []

    for _, r in df.iterrows():
        sample_ok = r.get("sample_status", "ok") == "ok"
        Spos = r["S_pos"]
        Sneg = r["S_neg"]
        S0 = r["S_0"]
        qwin = r["q_pos_within_window"]
        ppos = r["p_pos_surrogate"]

        if not sample_ok:
            stat = "not_evaluable"
        elif np.isfinite(ppos) and ppos <= settings.p_supported and np.isfinite(qwin) and qwin <= settings.q_within_window_supported and Spos > 0:
            stat = "supported"
        elif np.isfinite(ppos) and ppos <= settings.p_marginal and Spos > 0:
            stat = "marginal"
        else:
            stat = "not_supported"

        statistical_support.append(stat)

        # Effect-size characterization only.
        ratio = r["strength_ratio"]
        if stat == "not_supported":
            tier = "not_supported"
        elif np.isfinite(r["T_pos_obs"]) and np.isfinite(r["T_pos_null99"]) and r["T_pos_obs"] >= r["T_pos_null99"]:
            tier = "strong_above_null"
        elif np.isfinite(r["T_pos_obs"]) and np.isfinite(r["T_pos_null95"]) and r["T_pos_obs"] >= r["T_pos_null95"]:
            tier = "moderate_above_null"
        elif stat == "marginal":
            tier = "barely_above_null"
        else:
            tier = "weak_absolute_but_supported"
        strength_tier.append(tier)

        same_day = bool(np.isfinite(S0) and S0 > 0)
        same_day_flags.append(same_day)

        ci90_low = r.get("D_pos_neg_CI90_low", np.nan)
        ci90_high = r.get("D_pos_neg_CI90_high", np.nan)
        ci95_low = r.get("D_pos_neg_CI95_low", np.nan)
        d0_ci90_high = r.get("D_pos_0_CI90_high", np.nan)

        stable_pos = np.isfinite(ci90_low) and ci90_low > 0
        stable_pos_strong = np.isfinite(ci95_low) and ci95_low > 0
        stable_reverse = np.isfinite(ci90_high) and ci90_high < 0
        same_day_dominant = bool(same_day and np.isfinite(d0_ci90_high) and d0_ci90_high < 0)

        if stat == "not_evaluable":
            dlabel = "not_evaluable"
            llabel = "not_evaluable_insufficient_sample"
            group = "not_evaluable"
            reason = "insufficient_sample"
            risk = ""
            rev = ""
        elif stat == "not_supported":
            if same_day and not (np.isfinite(Spos) and Spos > 0):
                dlabel = "same_day_only"
                llabel = "lead_lag_no_same_day_only"
                reason = "same_day_only_without_positive_lag_support"
            else:
                dlabel = "no_directional_signal"
                llabel = "lead_lag_no_not_supported"
                reason = "positive_lag_not_supported"
            group = "lead_lag_no"
            risk = "same_day_coupling" if same_day else ""
            rev = ""
        elif stable_reverse:
            dlabel = "reverse_dominant"
            llabel = "lead_lag_no_reverse_dominant"
            group = "lead_lag_no"
            reason = "reverse_lag_dominant"
            risk = "reverse_direction"
            rev = f"{r['target']} -> {r['source']}"
        elif same_day_dominant:
            dlabel = "same_day_dominant"
            llabel = "lead_lag_ambiguous_same_day_dominant"
            group = "lead_lag_ambiguous"
            reason = "same_day_exceeds_positive_lag"
            risk = "same_day_dominant"
            rev = ""
        elif stat == "marginal":
            dlabel = "direction_not_assessed_as_strict_due_to_marginal_stat_support"
            llabel = "lead_lag_ambiguous_marginal_statistical_support"
            group = "lead_lag_ambiguous"
            reason = "marginal_statistical_support"
            risk = "marginal"
            rev = ""
        elif stable_pos:
            if same_day:
                dlabel = "X_leads_Y_with_same_day_component"
                llabel = "lead_lag_yes_with_same_day_coupling"
                risk = "same_day_coupling_present"
            else:
                dlabel = "clear_X_leads_Y_strong" if stable_pos_strong else "clear_X_leads_Y_moderate"
                llabel = "lead_lag_yes_clear"
                risk = ""
            group = "lead_lag_yes"
            reason = ""
            rev = ""
        else:
            if np.isfinite(Sneg) and Sneg > 0 and np.isfinite(Spos) and Spos > 0:
                dlabel = "bidirectional_close"
                llabel = "lead_lag_ambiguous_bidirectional_close"
                risk = "bidirectional_or_feedback_like"
                reason = "positive_and_negative_supported_but_direction_difference_uncertain"
            elif same_day:
                dlabel = "same_day_short_lag_mixed"
                llabel = "lead_lag_ambiguous_coupled_or_feedback_like"
                risk = "same_day_short_lag_mixed"
                reason = "positive_support_with_same_day_or_uncertain_direction"
            else:
                dlabel = "direction_uncertain"
                llabel = "lead_lag_ambiguous_direction_uncertain"
                risk = "direction_uncertain"
                reason = "direction_difference_not_stable"
            group = "lead_lag_ambiguous"
            rev = ""

        direction_label.append(dlabel)
        lead_label.append(llabel)
        lead_group.append(group)
        failure_reasons.append(reason)
        risk_notes.append(risk)
        reverse_suggest.append(rev)

    df["statistical_support"] = statistical_support
    df["strength_tier"] = strength_tier
    df["same_day_coupling_flag"] = same_day_flags
    df["direction_label"] = direction_label
    df["lead_lag_label"] = lead_label
    df["lead_lag_group"] = lead_group
    df["failure_reason"] = failure_reasons
    df["risk_note"] = risk_notes
    df["suggested_reverse_direction"] = reverse_suggest
    return df


def run_screen(settings: LeadLagScreenSettings, logger) -> dict:
    ensure_dirs(settings.output_dir, settings.log_dir)
    logger.info("Reading input index anomalies: %s", settings.input_index_anomalies)
    df = read_index_anomalies(settings.input_index_anomalies, settings.variables)
    years = np.sort(df["year"].unique())
    all_days = np.sort(df["day"].unique())
    logger.info("Input rows=%d, years=%d, day_range=[%s,%s]", len(df), len(years), all_days.min(), all_days.max())

    pair_df = make_directed_pairs(settings.variable_families, settings.include_same_family_pairs)
    src_idx, tgt_idx = _pair_indices(pair_df, settings.variables)
    logger.info("Directed pairs=%d (same-family included=%s)", len(pair_df), settings.include_same_family_pairs)

    rng = np.random.default_rng(settings.random_seed)

    curve_tables = []
    null_tables = []
    dir_tables = []
    ar1_param_tables = []
    ar1_yearwise_scale_tables = []
    audit_null_tables = []

    main_lags = list(range(-settings.max_lag, settings.max_lag + 1))
    curve_lags = list(range(-settings.diagnostic_max_lag, settings.diagnostic_max_lag + 1)) if settings.write_diagnostic_lags else main_lags
    pos_lags = list(range(1, settings.max_lag + 1))
    neg_lags = list(range(-settings.max_lag, 0))

    for wi, (window, (w_start, w_end)) in enumerate(settings.windows.items(), start=1):
        logger.info("[%d/%d] Processing window %s day %d-%d", wi, len(settings.windows), window, w_start, w_end)
        ext_start = max(int(all_days.min()), w_start - settings.diagnostic_max_lag)
        ext_end = min(int(all_days.max()), w_end + settings.diagnostic_max_lag)
        ext_days = np.arange(ext_start, ext_end + 1)
        target_days = np.arange(w_start, w_end + 1)
        panel = build_panel(df, settings.variables, years, ext_days)

        ar1_param_df, ar1_year_df, ar1_params_by_var = _build_ar1_audit_tables(
            panel=panel,
            variables=settings.variables,
            years=years,
            window=window,
        )
        ar1_param_tables.append(ar1_param_df)
        ar1_yearwise_scale_tables.append(ar1_year_df)
        phi_by_var = {var: pars["phi"] for var, pars in ar1_params_by_var.items()}

        logger.info("  observed lag curves")
        curve = _observed_curve_for_window(
            panel=panel,
            ext_days=ext_days,
            target_days=target_days,
            lags=curve_lags,
            pair_df=pair_df,
            variables=settings.variables,
            phi_by_var=phi_by_var,
            min_pairs=settings.min_pairs,
        )
        curve.insert(0, "window", window)
        curve_tables.append(curve)

        # Observed max stats from curve table.
        rows = []
        for _, pair in pair_df.iterrows():
            src = pair["source"]; tgt = pair["target"]
            sub = curve[(curve["source"] == src) & (curve["target"] == tgt)]
            pos = sub[sub["lag"].isin(pos_lags)].copy()
            neg = sub[sub["lag"].isin(neg_lags)].copy()
            zero = sub[sub["lag"] == 0].copy()

            def peak_info(part):
                if part.empty or part["abs_r"].notna().sum() == 0:
                    return np.nan, np.nan, np.nan, np.nan, 0, 0, "insufficient_pairs"
                idx = part["abs_r"].idxmax()
                row = part.loc[idx]
                return row["abs_r"], row["lag"], row["signed_r"], row["n_pairs"], row["n_years_used"], row["valid_year_fraction"], row["sample_status"]

            Tpos, pos_lag, pos_signed, pos_npairs, pos_nyears, pos_vyf, pos_status = peak_info(pos)
            Tneg, neg_lag, neg_signed, neg_npairs, neg_nyears, neg_vyf, neg_status = peak_info(neg)
            T0, lag0, lag0_signed, lag0_npairs, lag0_nyears, lag0_vyf, lag0_status = peak_info(zero)

            sample_status = "ok"
            if (not np.isfinite(Tpos)) or pos_npairs < settings.min_pairs or pos_vyf < settings.min_valid_year_fraction:
                sample_status = "insufficient_sample"

            rows.append({
                "window": window,
                "source": src,
                "target": tgt,
                "source_family": pair["source_family"],
                "target_family": pair["target_family"],
                "positive_peak_lag": pos_lag,
                "positive_peak_signed_r": pos_signed,
                "positive_peak_abs_r": Tpos,
                "positive_peak_n_pairs": pos_npairs,
                "positive_peak_n_years": pos_nyears,
                "positive_peak_valid_year_fraction": pos_vyf,
                "negative_peak_lag": neg_lag,
                "negative_peak_signed_r": neg_signed,
                "negative_peak_abs_r": Tneg,
                "lag0_signed_r": lag0_signed,
                "lag0_abs_r": T0,
                "sample_status": sample_status,
            })

        obs_summary = pd.DataFrame(rows)

        logger.info("  AR(1) surrogate null: mode=%s, n=%d", settings.surrogate_mode, settings.n_surrogates)
        sur = _surrogate_null_for_window(
            panel=panel,
            ext_days=ext_days,
            target_days=target_days,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            variables=settings.variables,
            ar1_params_by_var=ar1_params_by_var,
            n_surrogates=settings.n_surrogates,
            surrogate_mode=settings.surrogate_mode,
            settings=settings,
            rng=rng,
        )

        Tpos_obs = obs_summary["positive_peak_abs_r"].to_numpy(dtype=float)
        Tneg_obs = obs_summary["negative_peak_abs_r"].to_numpy(dtype=float)
        T0_obs = obs_summary["lag0_abs_r"].to_numpy(dtype=float)

        def p_from_null(null_arr, obs):
            return (1.0 + np.sum(null_arr >= obs[None, :], axis=0)) / (null_arr.shape[0] + 1.0)

        null_df = obs_summary.copy()
        null_df["surrogate_mode"] = settings.surrogate_mode
        for name, obs, null_arr in [
            ("pos", Tpos_obs, sur["T_pos"]),
            ("neg", Tneg_obs, sur["T_neg"]),
            ("0", T0_obs, sur["T_0"]),
        ]:
            null_df[f"T_{name}_obs"] = obs
            null_df[f"T_{name}_null90"] = np.nanpercentile(null_arr, 90, axis=0)
            null_df[f"T_{name}_null95"] = np.nanpercentile(null_arr, 95, axis=0)
            null_df[f"T_{name}_null99"] = np.nanpercentile(null_arr, 99, axis=0)
            null_df[f"p_{name}_surrogate"] = p_from_null(null_arr, obs)

        null_df["S_pos"] = null_df["T_pos_obs"] - null_df["T_pos_null95"]
        null_df["S_neg"] = null_df["T_neg_obs"] - null_df["T_neg_null95"]
        null_df["S_0"] = null_df["T_0_obs"] - null_df["T_0_null95"]
        null_df["strength_excess"] = null_df["S_pos"]
        null_df["strength_ratio"] = null_df["T_pos_obs"] / null_df["T_pos_null95"].replace(0, np.nan)

        if settings.run_audit_surrogate_null:
            logger.info(
                "  audit AR(1) surrogate null: mode=%s, n=%d",
                settings.audit_surrogate_mode,
                settings.n_audit_surrogates,
            )
            audit_sur = _surrogate_null_for_window(
                panel=panel,
                ext_days=ext_days,
                target_days=target_days,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                variables=settings.variables,
                ar1_params_by_var=ar1_params_by_var,
                n_surrogates=settings.n_audit_surrogates,
                surrogate_mode=settings.audit_surrogate_mode,
                settings=settings,
                rng=rng,
            )
            audit_df = obs_summary[["window", "source", "target", "source_family", "target_family", "sample_status"]].copy()
            audit_df["main_surrogate_mode"] = settings.surrogate_mode
            audit_df["audit_surrogate_mode"] = settings.audit_surrogate_mode
            audit_df["T_pos_obs"] = Tpos_obs
            audit_df["T_pos_main_null95"] = null_df["T_pos_null95"].to_numpy(dtype=float)
            audit_df["p_pos_main_surrogate"] = null_df["p_pos_surrogate"].to_numpy(dtype=float)
            audit_df["T_pos_audit_null90"] = np.nanpercentile(audit_sur["T_pos"], 90, axis=0)
            audit_df["T_pos_audit_null95"] = np.nanpercentile(audit_sur["T_pos"], 95, axis=0)
            audit_df["T_pos_audit_null99"] = np.nanpercentile(audit_sur["T_pos"], 99, axis=0)
            audit_df["p_pos_audit_surrogate"] = p_from_null(audit_sur["T_pos"], Tpos_obs)
            audit_df["audit_strength_excess"] = audit_df["T_pos_obs"] - audit_df["T_pos_audit_null95"]
            audit_df["audit_strength_ratio"] = audit_df["T_pos_obs"] / audit_df["T_pos_audit_null95"].replace(0, np.nan)
            audit_df["delta_p_audit_minus_main"] = audit_df["p_pos_audit_surrogate"] - audit_df["p_pos_main_surrogate"]
            audit_df["delta_null95_audit_minus_main"] = audit_df["T_pos_audit_null95"] - audit_df["T_pos_main_null95"]
            audit_null_tables.append(audit_df)

        null_tables.append(null_df)

        logger.info("  directional year-bootstrap: n=%d", settings.n_direction_bootstrap)
        boot = _bootstrap_direction_for_window(
            panel=panel,
            ext_days=ext_days,
            target_days=target_days,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            null95_pos=null_df["T_pos_null95"].to_numpy(dtype=float),
            null95_neg=null_df["T_neg_null95"].to_numpy(dtype=float),
            null95_0=null_df["T_0_null95"].to_numpy(dtype=float),
            settings=settings,
            rng=rng,
        )

        Dpn = boot["D_pos_neg"]
        Dp0 = boot["D_pos_0"]
        dir_df = null_df[["window", "source", "target"]].copy()
        dir_df["D_pos_neg"] = null_df["S_pos"] - null_df["S_neg"]
        dir_df["D_pos_0"] = null_df["S_pos"] - null_df["S_0"]
        for prefix, arr in [("D_pos_neg", Dpn), ("D_pos_0", Dp0)]:
            dir_df[f"{prefix}_bootstrap_mean"] = np.nanmean(arr, axis=0)
            dir_df[f"{prefix}_CI90_low"] = np.nanpercentile(arr, 5, axis=0)
            dir_df[f"{prefix}_CI90_high"] = np.nanpercentile(arr, 95, axis=0)
            dir_df[f"{prefix}_CI95_low"] = np.nanpercentile(arr, 2.5, axis=0)
            dir_df[f"{prefix}_CI95_high"] = np.nanpercentile(arr, 97.5, axis=0)
            dir_df[f"P_{prefix}_gt_0"] = np.nanmean(arr > 0, axis=0)
        dir_df["LOO_direction_retained_fraction"] = boot["loo_retained_fraction"]
        dir_df["LOO_flip_years"] = boot["loo_flip_years"]
        dir_tables.append(dir_df)

    curve_all = pd.concat(curve_tables, ignore_index=True)
    null_all = pd.concat(null_tables, ignore_index=True)
    dir_all = pd.concat(dir_tables, ignore_index=True)
    ar1_params_all = pd.concat(ar1_param_tables, ignore_index=True)
    ar1_yearwise_scale_all = pd.concat(ar1_yearwise_scale_tables, ignore_index=True)
    audit_null_all = pd.concat(audit_null_tables, ignore_index=True) if audit_null_tables else pd.DataFrame()

    logger.info("Applying FDR corrections")
    null_all["q_pos_global"] = fdr_bh(null_all["p_pos_surrogate"])
    null_all["q_pos_within_window"] = np.nan
    for window, sub_idx in null_all.groupby("window").groups.items():
        null_all.loc[sub_idx, "q_pos_within_window"] = fdr_bh(null_all.loc[sub_idx, "p_pos_surrogate"])

    if not audit_null_all.empty:
        audit_null_all["q_pos_audit_global"] = fdr_bh(audit_null_all["p_pos_audit_surrogate"])
        audit_null_all["q_pos_audit_within_window"] = np.nan
        for window, sub_idx in audit_null_all.groupby("window").groups.items():
            audit_null_all.loc[sub_idx, "q_pos_audit_within_window"] = fdr_bh(
                audit_null_all.loc[sub_idx, "p_pos_audit_surrogate"]
            )
        audit_null_all["main_p_supported"] = audit_null_all["p_pos_main_surrogate"] <= settings.p_supported
        audit_null_all["audit_p_supported"] = audit_null_all["p_pos_audit_surrogate"] <= settings.p_supported
        audit_null_all["audit_support_shift"] = np.where(
            audit_null_all["main_p_supported"] == audit_null_all["audit_p_supported"],
            "same_p_support_flag",
            np.where(audit_null_all["audit_p_supported"], "audit_more_permissive", "audit_more_conservative"),
        )

    classified = _classify_rows(null_all, dir_all, settings)

    # Pair summary keeps the most useful columns.
    pair_cols = [
        "window", "source", "target", "source_family", "target_family",
        "positive_peak_lag", "positive_peak_signed_r", "positive_peak_abs_r",
        "negative_peak_lag", "negative_peak_signed_r", "negative_peak_abs_r",
        "lag0_signed_r", "lag0_abs_r",
        "p_pos_surrogate", "q_pos_within_window", "q_pos_global",
        "T_pos_null95", "strength_excess", "strength_ratio",
        "statistical_support", "strength_tier",
        "same_day_coupling_flag", "direction_label",
        "lead_lag_label", "lead_lag_group",
        "failure_reason", "risk_note", "suggested_reverse_direction",
        "sample_status",
    ]
    pair_summary = classified[pair_cols].copy()

    pools = pair_summary[[
        "window", "source", "target", "source_family", "target_family",
        "lead_lag_label", "lead_lag_group", "risk_note",
    ]].copy()
    pools["in_strict_temporal_pool"] = pools["lead_lag_label"].isin([
        "lead_lag_yes_clear",
        "lead_lag_yes_with_same_day_coupling",
    ])
    pools["in_expanded_temporal_risk_pool"] = pools["lead_lag_group"].isin([
        "lead_lag_yes", "lead_lag_ambiguous"
    ])

    logger.info("Building evidence-tier audit outputs")
    evidence_tier, family_rollup, phi_risk_audit, warning_flags = build_evidence_tier_outputs(
        pair_summary=pair_summary,
        classified=classified,
        dir_all=dir_all,
        audit_null=audit_null_all,
        ar1_params=ar1_params_all,
        settings=settings,
    )

    logger.info("Writing outputs to %s", settings.output_dir)
    curve_all.to_csv(settings.output_dir / "lead_lag_curve_long.csv", index=False, encoding="utf-8-sig")
    null_cols = [c for c in null_all.columns if c not in {"positive_peak_n_pairs", "positive_peak_n_years", "positive_peak_valid_year_fraction"}]
    null_all[null_cols].to_csv(settings.output_dir / "lead_lag_null_summary.csv", index=False, encoding="utf-8-sig")
    dir_all.to_csv(settings.output_dir / "lead_lag_directional_robustness.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(settings.output_dir / "lead_lag_pair_summary.csv", index=False, encoding="utf-8-sig")
    pools.to_csv(settings.output_dir / "lead_lag_temporal_pools.csv", index=False, encoding="utf-8-sig")
    ar1_params_all.to_csv(settings.output_dir / "lead_lag_surrogate_ar1_params.csv", index=False, encoding="utf-8-sig")
    ar1_yearwise_scale_all.to_csv(settings.output_dir / "lead_lag_surrogate_yearwise_scale.csv", index=False, encoding="utf-8-sig")
    if not audit_null_all.empty:
        audit_null_all.to_csv(settings.output_dir / "lead_lag_audit_surrogate_null_summary.csv", index=False, encoding="utf-8-sig")
    evidence_tier.to_csv(settings.output_dir / "lead_lag_evidence_tier_summary.csv", index=False, encoding="utf-8-sig")
    family_rollup.to_csv(settings.output_dir / "lead_lag_window_family_tier_rollup.csv", index=False, encoding="utf-8-sig")
    phi_risk_audit.to_csv(settings.output_dir / "lead_lag_phi_risk_audit.csv", index=False, encoding="utf-8-sig")
    warning_flags.to_csv(settings.output_dir / "lead_lag_method_warning_flags.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "output_tag": settings.output_tag,
        "n_years": int(len(years)),
        "n_variables": int(len(settings.variables)),
        "n_directed_pairs": int(len(pair_df)),
        "n_windows": int(len(settings.windows)),
        "n_curve_rows": int(len(curve_all)),
        "n_pair_rows": int(len(pair_summary)),
        "label_counts": pair_summary["lead_lag_label"].value_counts(dropna=False).to_dict(),
        "group_counts": pair_summary["lead_lag_group"].value_counts(dropna=False).to_dict(),
        "strict_temporal_pool_size": int(pools["in_strict_temporal_pool"].sum()),
        "expanded_temporal_risk_pool_size": int(pools["in_expanded_temporal_risk_pool"].sum()),
        "main_surrogate_mode": settings.surrogate_mode,
        "audit_surrogate_mode": settings.audit_surrogate_mode if settings.run_audit_surrogate_null else None,
        "audit_surrogate_enabled": bool(settings.run_audit_surrogate_null),
        "n_ar1_param_rows": int(len(ar1_params_all)),
        "n_ar1_yearwise_scale_rows": int(len(ar1_yearwise_scale_all)),
        "n_audit_null_rows": int(len(audit_null_all)) if not audit_null_all.empty else 0,
        "n_evidence_tier_rows": int(len(evidence_tier)),
        "evidence_tier_counts": evidence_tier["evidence_tier"].value_counts(dropna=False).to_dict() if "evidence_tier" in evidence_tier else {},
        "n_warning_flags": int(len(warning_flags)),
        "n_phi_clipped_total": int(ar1_params_all["phi_clipped_flag"].sum()) if "phi_clipped_flag" in ar1_params_all else None,
        "phi_clip_severity_counts": ar1_params_all["phi_clip_severity"].value_counts(dropna=False).to_dict() if "phi_clip_severity" in ar1_params_all else {},
        "raw_phi_max": float(ar1_params_all["raw_phi_before_clip"].max()) if "raw_phi_before_clip" in ar1_params_all else None,
        "raw_phi_min": float(ar1_params_all["raw_phi_before_clip"].min()) if "raw_phi_before_clip" in ar1_params_all else None,
        "notes": [
            "This layer is a temporal eligibility screen, not pathway establishment.",
            "Strength is represented as null-relative excess and absolute r, not as a fixed-r hard gate.",
            "lag=0 is same-day coupling and does not automatically invalidate positive-lag support.",
            "Old pathway V1-V2 outputs are not read by this pipeline.",
            "Main surrogate mode uses pooled window-variable AR(1), not year-specific phi.",
            "Audit surrogate mode keeps pooled phi but uses yearwise scale; it is diagnostic and does not change main labels.",
            "AR(1) audit table now reports raw_phi_before_clip and clipping severity.",
            "Evidence-tier audit is post-processing only; it does not change primary lead_lag_label.",
        ],
    }
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    run_meta = {
        "settings": {
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items() if k not in {"windows", "variable_families"}},
            "windows": settings.windows,
            "variable_families": settings.variable_families,
        },
        "method_contract": {
            "window_assignment": "target-side Y(t) belongs to W",
            "year_pairing": "same-year only; no cross-year concatenation",
            "statistical_support": "main AR(1) surrogate max-stat for positive lags",
            "main_surrogate_mode": settings.surrogate_mode,
            "audit_surrogate_mode": settings.audit_surrogate_mode if settings.run_audit_surrogate_null else None,
            "audit_surrogate_role": "diagnostic null only; does not alter main lead_lag_label",
            "directional_robustness": "year-block bootstrap of null-relative positive-vs-negative and positive-vs-lag0 evidence",
            "same_day_rule": "lag0 is diagnostic same-day coupling, not lead evidence",
        },
    }
    (settings.output_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary
