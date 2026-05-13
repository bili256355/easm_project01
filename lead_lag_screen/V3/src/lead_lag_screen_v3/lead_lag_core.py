from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_io import make_directed_object_pairs
from .settings import LeadLagScreenV3Settings
from .stats_utils import corr_matrix_batch, estimate_ar1_params_diagnostic, fdr_bh, fisher_effn_p, safe_corr_1d


def _lag_arrays(panel: np.ndarray, ext_days: np.ndarray, target_days: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _pair_indices(pair_df: pd.DataFrame, variables: List[str]) -> tuple[np.ndarray, np.ndarray]:
    var_to_idx = {v: i for i, v in enumerate(variables)}
    return pair_df["source"].map(var_to_idx).to_numpy(), pair_df["target"].map(var_to_idx).to_numpy()


def _matrix_pair_values(mat: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray) -> np.ndarray:
    if mat.ndim == 2:
        return mat[src_idx, tgt_idx]
    return mat[:, src_idx, tgt_idx]


def _build_ar1_audit_tables(panel: np.ndarray, variables: List[str], years: np.ndarray, window: str) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
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
            "surrogate_mode": "pooled_window_object_pc1_ar1",
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
            "phi_clipped_flag": bool(diag["phi_clipped_flag"]),
            "phi_clip_amount": float(diag["phi_clip_amount"]),
            "phi_clip_direction": diag["phi_clip_direction"],
            "phi_clip_severity": diag["phi_clip_severity"],
            "sigma": sigma,
            "stationary_std": stationary_std,
            "ar1_estimation_scope": "window_object_pc1_pooled_across_years",
        })
        for yi, year in enumerate(years):
            yarr = arr[yi, :]
            yfinite = np.isfinite(yarr)
            y_n = int(yfinite.sum())
            year_rows.append({
                "window": window,
                "year": int(year),
                "variable": var,
                "audit_surrogate_mode": "pooled_phi_yearwise_scale_ar1",
                "year_n_finite": y_n,
                "year_nan_fraction": 1.0 - (y_n / len(yarr) if len(yarr) else np.nan),
                "year_mean": float(np.nanmean(yarr)) if y_n else np.nan,
                "year_std": float(np.nanstd(yarr)) if y_n else np.nan,
                "pooled_raw_phi_before_clip": raw_phi,
                "pooled_phi_used": phi,
                "pooled_phi_clip_amount": float(diag["phi_clip_amount"]),
                "pooled_phi_clip_severity": diag["phi_clip_severity"],
            })
    return pd.DataFrame(pooled_rows), pd.DataFrame(year_rows), params


def _generate_ar1_surrogates(panel: np.ndarray, variables: List[str], ar1_params_by_var: Dict[str, dict], rng: np.random.Generator, n_rep: int, surrogate_mode: str) -> np.ndarray:
    n_year, n_day, n_var = panel.shape
    out = np.empty((n_rep, n_year, n_day, n_var), dtype=float)
    for v, var in enumerate(variables):
        pars = ar1_params_by_var[var]
        pooled_mu = float(pars["mu"])
        phi = float(pars["phi"])
        pooled_sigma = float(pars["sigma"])
        pooled_stationary_std = float(pars["stationary_std"])
        x = np.empty((n_rep, n_year, n_day), dtype=float)
        if surrogate_mode == "pooled_window_object_pc1_ar1":
            x[:, :, 0] = pooled_mu + rng.normal(0.0, pooled_stationary_std, size=(n_rep, n_year))
            eps = rng.normal(0.0, pooled_sigma, size=(n_rep, n_year, max(n_day - 1, 1)))
            for d in range(1, n_day):
                x[:, :, d] = pooled_mu + phi * (x[:, :, d - 1] - pooled_mu) + eps[:, :, d - 1]
        elif surrogate_mode == "pooled_phi_yearwise_scale_ar1":
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
                if not np.isfinite(y_mu[yi]):
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


def _observed_curve_for_window(panel: np.ndarray, ext_days: np.ndarray, target_days: np.ndarray, lags: List[int], pair_df: pd.DataFrame, variables: List[str], phi_by_var: Dict[str, float], min_pairs: int) -> pd.DataFrame:
    rows = []
    var_to_idx = {v: i for i, v in enumerate(variables)}
    n_total_years = panel.shape[0]
    for lag in lags:
        source_arr, target_arr, _ = _lag_arrays(panel, ext_days, target_days, lag)
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
            rows.append({
                "source": src,
                "target": tgt,
                "source_object": pair["source_object"],
                "target_object": pair["target_object"],
                "family_direction": pair["family_direction"],
                "lag": lag,
                "n_pairs": n_pairs,
                "n_years_used": n_years,
                "valid_year_fraction": n_years / n_total_years if n_total_years else np.nan,
                "r": r,
                "signed_r": r,
                "abs_r": abs(r) if np.isfinite(r) else np.nan,
                "eff_n": neff,
                "p_effn": fisher_effn_p(r, neff),
                "same_year_only": True,
                "target_window_rule": True,
                "sample_status": "ok" if n_pairs >= min_pairs else "insufficient_pairs",
            })
    return pd.DataFrame(rows)


def _max_stat_from_corrs(panel_or_sur: np.ndarray, ext_days: np.ndarray, target_days: np.ndarray, lags: List[int], src_idx: np.ndarray, tgt_idx: np.ndarray, min_pairs: int) -> np.ndarray:
    is_batched = panel_or_sur.ndim == 4
    if is_batched:
        acc = np.full((panel_or_sur.shape[0], len(src_idx)), np.nan, dtype=float)
    else:
        acc = np.full((len(src_idx),), np.nan, dtype=float)
    for lag in lags:
        if is_batched:
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
        corr, _ = corr_matrix_batch(x, y, min_count=min_pairs)
        vals = np.abs(_matrix_pair_values(corr, src_idx, tgt_idx))
        if is_batched:
            acc = np.fmax(acc, vals)
        else:
            acc = np.fmax(acc, vals[0])
    return acc


def _surrogate_null_for_window(panel: np.ndarray, ext_days: np.ndarray, target_days: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray, variables: List[str], ar1_params_by_var: Dict[str, dict], n_surrogates: int, surrogate_mode: str, settings: LeadLagScreenV3Settings, rng: np.random.Generator) -> dict[str, np.ndarray]:
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
        sur = _generate_ar1_surrogates(panel, variables, ar1_params_by_var, rng, stop - start, surrogate_mode)
        T_pos[start:stop] = _max_stat_from_corrs(sur, ext_days, target_days, pos_lags, src_idx, tgt_idx, settings.min_pairs)
        T_neg[start:stop] = _max_stat_from_corrs(sur, ext_days, target_days, neg_lags, src_idx, tgt_idx, settings.min_pairs)
        T_0[start:stop] = _max_stat_from_corrs(sur, ext_days, target_days, zero_lags, src_idx, tgt_idx, settings.min_pairs)
        start = stop
    return {"T_pos": T_pos, "T_neg": T_neg, "T_0": T_0}


def _bootstrap_direction_for_window(panel: np.ndarray, ext_days: np.ndarray, target_days: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray, null95_pos: np.ndarray, null95_neg: np.ndarray, null95_0: np.ndarray, settings: LeadLagScreenV3Settings, rng: np.random.Generator) -> dict[str, np.ndarray]:
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
        year_idx = rng.integers(0, n_year, size=(stop - start, n_year))
        boot = panel[year_idx, :, :]
        Tpos_b = _max_stat_from_corrs(boot, ext_days, target_days, pos_lags, src_idx, tgt_idx, settings.min_pairs)
        Tneg_b = _max_stat_from_corrs(boot, ext_days, target_days, neg_lags, src_idx, tgt_idx, settings.min_pairs)
        T0_b = _max_stat_from_corrs(boot, ext_days, target_days, zero_lags, src_idx, tgt_idx, settings.min_pairs)
        Dpn[start:stop] = (Tpos_b - null95_pos[None, :]) - (Tneg_b - null95_neg[None, :])
        Dp0[start:stop] = (Tpos_b - null95_pos[None, :]) - (T0_b - null95_0[None, :])
        start = stop
    loo_retained = np.full((n_pairs,), np.nan, dtype=float)
    loo_flip_counts = np.full((n_pairs,), np.nan, dtype=float)
    if n_year >= 3:
        vals = []
        for omit in range(n_year):
            sub = np.delete(panel, omit, axis=0)
            Tpos_l = _max_stat_from_corrs(sub, ext_days, target_days, pos_lags, src_idx, tgt_idx, settings.min_pairs)
            Tneg_l = _max_stat_from_corrs(sub, ext_days, target_days, neg_lags, src_idx, tgt_idx, settings.min_pairs)
            vals.append((Tpos_l - null95_pos) - (Tneg_l - null95_neg))
        loo = np.vstack(vals)
        loo_retained = np.nanmean(loo > 0, axis=0)
        loo_flip_counts = np.sum(loo <= 0, axis=0)
    return {"D_pos_neg": Dpn, "D_pos_0": Dp0, "loo_retained_fraction": loo_retained, "loo_flip_years": loo_flip_counts}


def _classify_rows(null_df: pd.DataFrame, dir_df: pd.DataFrame, settings: LeadLagScreenV3Settings) -> pd.DataFrame:
    df = null_df.merge(dir_df, on=["window", "source", "target"], how="left")
    records = []
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
        if stat == "not_supported":
            tier_strength = "not_supported"
        elif np.isfinite(r["T_pos_obs"]) and np.isfinite(r["T_pos_null99"]) and r["T_pos_obs"] >= r["T_pos_null99"]:
            tier_strength = "strong_above_null"
        elif np.isfinite(r["T_pos_obs"]) and np.isfinite(r["T_pos_null95"]) and r["T_pos_obs"] >= r["T_pos_null95"]:
            tier_strength = "moderate_above_null"
        elif stat == "marginal":
            tier_strength = "barely_above_null"
        else:
            tier_strength = "weak_absolute_but_supported"
        same_day = bool(np.isfinite(S0) and S0 > 0)
        ci90_low = r.get("D_pos_neg_CI90_low", np.nan)
        ci90_high = r.get("D_pos_neg_CI90_high", np.nan)
        ci95_low = r.get("D_pos_neg_CI95_low", np.nan)
        d0_ci90_high = r.get("D_pos_0_CI90_high", np.nan)
        stable_pos = np.isfinite(ci90_low) and ci90_low > 0
        stable_pos_strong = np.isfinite(ci95_low) and ci95_low > 0
        stable_reverse = np.isfinite(ci90_high) and ci90_high < 0
        same_day_dominant = bool(same_day and np.isfinite(d0_ci90_high) and d0_ci90_high < 0)
        if stat == "not_evaluable":
            dlabel = "not_evaluable"; llabel = "PC1_not_evaluable_insufficient_sample"; group = "not_evaluable"; reason = "insufficient_sample"; risk = ""; rev = ""
        elif stat == "not_supported":
            if same_day and not (np.isfinite(Spos) and Spos > 0):
                dlabel = "same_day_only"; llabel = "PC1_no_same_day_only"; reason = "same_day_only_without_positive_lag_support"
            else:
                dlabel = "no_directional_signal"; llabel = "PC1_no_not_supported"; reason = "positive_lag_not_supported"
            group = "PC1_lead_lag_no"; risk = "same_day_coupling" if same_day else ""; rev = ""
        elif stable_reverse:
            dlabel = "reverse_dominant"; llabel = "PC1_no_reverse_dominant"; group = "PC1_lead_lag_no"; reason = "reverse_lag_dominant"; risk = "reverse_direction"; rev = f"{r['target']} -> {r['source']}"
        elif same_day_dominant:
            dlabel = "same_day_dominant"; llabel = "PC1_ambiguous_same_day_dominant"; group = "PC1_lead_lag_ambiguous"; reason = "same_day_exceeds_positive_lag"; risk = "same_day_dominant"; rev = ""
        elif stat == "marginal":
            dlabel = "direction_not_assessed_as_strict_due_to_marginal_stat_support"; llabel = "PC1_ambiguous_marginal_statistical_support"; group = "PC1_lead_lag_ambiguous"; reason = "marginal_statistical_support"; risk = "marginal"; rev = ""
        elif stable_pos:
            if same_day:
                dlabel = "X_leads_Y_with_same_day_component"; llabel = "PC1_yes_with_same_day_coupling"; risk = "same_day_coupling_present"
            else:
                dlabel = "clear_X_leads_Y_strong" if stable_pos_strong else "clear_X_leads_Y_moderate"; llabel = "PC1_yes_clear"; risk = ""
            group = "PC1_lead_lag_yes"; reason = ""; rev = ""
        else:
            if np.isfinite(Sneg) and Sneg > 0 and np.isfinite(Spos) and Spos > 0:
                dlabel = "bidirectional_close"; llabel = "PC1_ambiguous_bidirectional_close"; risk = "bidirectional_or_feedback_like"; reason = "positive_and_negative_supported_but_direction_difference_uncertain"
            elif same_day:
                dlabel = "same_day_short_lag_mixed"; llabel = "PC1_ambiguous_coupled_or_feedback_like"; risk = "same_day_short_lag_mixed"; reason = "positive_support_with_same_day_or_uncertain_direction"
            else:
                dlabel = "direction_uncertain"; llabel = "PC1_ambiguous_direction_uncertain"; risk = "direction_uncertain"; reason = "direction_difference_not_stable"
            group = "PC1_lead_lag_ambiguous"; rev = ""
        records.append((stat, tier_strength, same_day, dlabel, llabel, group, reason, risk, rev))
    rec = pd.DataFrame(records, columns=["statistical_support", "strength_tier", "same_day_coupling_flag", "direction_label", "pc1_lead_lag_label", "pc1_lead_lag_group", "failure_reason", "risk_note", "suggested_reverse_direction"])
    return pd.concat([df.reset_index(drop=True), rec], axis=1)


def _classify_pc1_tier(row: pd.Series) -> tuple[str, str, str]:
    group = row.get("pc1_lead_lag_group", "")
    label = row.get("pc1_lead_lag_label", "")
    audit_p = row.get("p_pos_audit_surrogate", np.nan)
    audit_q = row.get("q_pos_audit_within_window", np.nan)
    risk = row.get("pair_phi_risk", "none")
    audit_p_pass = bool(np.isfinite(audit_p) and audit_p <= 0.05)
    audit_q_pass = bool(np.isfinite(audit_q) and audit_q <= 0.10)
    severe_phi = risk == "severe"
    if group == "PC1_lead_lag_yes":
        if audit_q_pass and not severe_phi:
            if label == "PC1_yes_clear":
                return "PC1_Tier1a_audit_stable_clear_leadlag", "PC1 main yes; audit-FDR pass; clear lead-lag", "strict PC1 field-mode temporal core"
            return "PC1_Tier1b_audit_stable_with_same_day", "PC1 main yes; audit-FDR pass; same-day coupling present", "usable PC1 field-mode candidate with same-day warning"
        if audit_p_pass and not severe_phi:
            return "PC1_Tier2_main_supported_audit_moderate", "PC1 main yes; audit p pass but audit-FDR/phi prevents Tier1", "usable PC1 moderate candidate"
        return "PC1_Tier3_surrogate_or_persistence_sensitive_yes", "PC1 main yes but audit p fails and/or severe phi risk", "risk-pool only"
    if group == "PC1_lead_lag_ambiguous":
        if "same_day" in str(label):
            return "PC1_Tier4b_ambiguous_same_day", "same-day dominant/mixed", "expanded PC1 risk pool"
        if "bidirectional" in str(label):
            return "PC1_Tier4a_ambiguous_bidirectional_close", "bidirectional/feedback-like", "expanded PC1 risk pool"
        if "marginal" in str(label):
            return "PC1_Tier4d_ambiguous_marginal", "marginal statistical support", "expanded PC1 risk pool"
        return "PC1_Tier4c_ambiguous_direction_uncertain", "direction uncertain", "expanded PC1 risk pool"
    if group == "PC1_lead_lag_no":
        if "same_day_only" in str(label):
            return "PC1_Tier5c_no_same_day_only", "same-day coupling but no positive-lag support", "same-day diagnostic only"
        if "reverse" in str(label):
            return "PC1_Tier5b_no_reverse_dominant", "reverse lag is stronger", "inspect reverse direction if relevant"
        return "PC1_Tier5a_no_not_supported", "positive-lag not supported", "exclude from strict PC1 pool"
    return "PC1_Tier0_not_evaluable", "not evaluable or unclassified", "do not interpret as no"


def _attach_phi_risk(df: pd.DataFrame, ar1_params: pd.DataFrame) -> pd.DataFrame:
    phi = ar1_params[["window", "variable", "raw_phi_before_clip", "phi_after_clip", "phi_clip_amount", "phi_clip_severity", "phi_clipped_flag"]].copy()
    src_phi = phi.rename(columns={"variable": "source", "raw_phi_before_clip": "source_raw_phi", "phi_after_clip": "source_phi_after_clip", "phi_clip_amount": "source_phi_clip_amount", "phi_clip_severity": "source_phi_clip_severity", "phi_clipped_flag": "source_phi_clipped_flag"})
    tgt_phi = phi.rename(columns={"variable": "target", "raw_phi_before_clip": "target_raw_phi", "phi_after_clip": "target_phi_after_clip", "phi_clip_amount": "target_phi_clip_amount", "phi_clip_severity": "target_phi_clip_severity", "phi_clipped_flag": "target_phi_clipped_flag"})
    out = df.merge(src_phi, on=["window", "source"], how="left").merge(tgt_phi, on=["window", "target"], how="left")
    order = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    def mx(a, b):
        a = "none" if pd.isna(a) else str(a)
        b = "none" if pd.isna(b) else str(b)
        return a if order.get(a, 0) >= order.get(b, 0) else b
    out["pair_phi_risk"] = [mx(a, b) for a, b in zip(out["source_phi_clip_severity"], out["target_phi_clip_severity"])]
    return out


def run_pc1_lead_lag(panels: Dict[str, np.ndarray], years: np.ndarray, settings: LeadLagScreenV3Settings, logger) -> Dict[str, pd.DataFrame]:
    variables = [f"{obj}_PC1" for obj in settings.objects]
    pair_df = make_directed_object_pairs(settings.objects)
    src_idx, tgt_idx = _pair_indices(pair_df, variables)
    rng = np.random.default_rng(settings.random_seed)
    main_lags = list(range(-settings.max_lag, settings.max_lag + 1))
    curve_lags = list(range(-settings.diagnostic_max_lag, settings.diagnostic_max_lag + 1)) if settings.write_diagnostic_lags else main_lags
    pos_lags = list(range(1, settings.max_lag + 1))
    neg_lags = list(range(-settings.max_lag, 0))

    curve_tables = []
    null_tables = []
    dir_tables = []
    ar1_param_tables = []
    ar1_year_tables = []
    audit_tables = []

    for wi, (window, (w_start, w_end)) in enumerate(settings.windows.items(), start=1):
        logger.info("[%d/%d] PC1 lead-lag window %s", wi, len(settings.windows), window)
        panel = panels[window]
        ext_start = max(1, w_start - settings.diagnostic_max_lag)
        ext_end = min(panel.shape[1] + ext_start - 1, w_end + settings.diagnostic_max_lag)
        # The panel was constructed using the same ext_start/ext_end logic during EOF extraction.
        ext_days = np.arange(ext_start, ext_end + 1, dtype=int)
        target_days = np.arange(w_start, w_end + 1, dtype=int)

        ar1_df, ar1_year_df, ar1_params = _build_ar1_audit_tables(panel, variables, years, window)
        ar1_param_tables.append(ar1_df)
        ar1_year_tables.append(ar1_year_df)
        phi_by_var = {var: pars["phi"] for var, pars in ar1_params.items()}

        curve = _observed_curve_for_window(panel, ext_days, target_days, curve_lags, pair_df, variables, phi_by_var, settings.min_pairs)
        curve.insert(0, "window", window)
        curve_tables.append(curve)

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
            Tpos, pos_lag, pos_signed, pos_npairs, pos_nyears, pos_vyf, _ = peak_info(pos)
            Tneg, neg_lag, neg_signed, _, _, _, _ = peak_info(neg)
            T0, _, lag0_signed, _, _, _, _ = peak_info(zero)
            sample_status = "ok"
            if (not np.isfinite(Tpos)) or pos_npairs < settings.min_pairs or pos_vyf < settings.min_valid_year_fraction:
                sample_status = "insufficient_sample"
            rows.append({
                "window": window,
                "source": src,
                "target": tgt,
                "source_object": pair["source_object"],
                "target_object": pair["target_object"],
                "source_family": pair["source_family"],
                "target_family": pair["target_family"],
                "family_direction": pair["family_direction"],
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

        logger.info("  main AR(1) surrogate null n=%d", settings.n_surrogates)
        sur = _surrogate_null_for_window(panel, ext_days, target_days, src_idx, tgt_idx, variables, ar1_params, settings.n_surrogates, settings.surrogate_mode, settings, rng)
        Tpos_obs = obs_summary["positive_peak_abs_r"].to_numpy(dtype=float)
        Tneg_obs = obs_summary["negative_peak_abs_r"].to_numpy(dtype=float)
        T0_obs = obs_summary["lag0_abs_r"].to_numpy(dtype=float)
        def p_from_null(null_arr, obs):
            return (1.0 + np.sum(null_arr >= obs[None, :], axis=0)) / (null_arr.shape[0] + 1.0)
        null_df = obs_summary.copy()
        null_df["surrogate_mode"] = settings.surrogate_mode
        for name, obs, null_arr in [("pos", Tpos_obs, sur["T_pos"]), ("neg", Tneg_obs, sur["T_neg"]), ("0", T0_obs, sur["T_0"])]:
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
            logger.info("  audit AR(1) surrogate null n=%d", settings.n_audit_surrogates)
            audit_sur = _surrogate_null_for_window(panel, ext_days, target_days, src_idx, tgt_idx, variables, ar1_params, settings.n_audit_surrogates, settings.audit_surrogate_mode, settings, rng)
            audit_df = obs_summary[["window", "source", "target", "source_family", "target_family", "family_direction", "sample_status"]].copy()
            audit_df["main_surrogate_mode"] = settings.surrogate_mode
            audit_df["audit_surrogate_mode"] = settings.audit_surrogate_mode
            audit_df["T_pos_obs"] = Tpos_obs
            audit_df["T_pos_main_null95"] = null_df["T_pos_null95"].to_numpy(dtype=float)
            audit_df["p_pos_main_surrogate"] = null_df["p_pos_surrogate"].to_numpy(dtype=float)
            audit_df["T_pos_audit_null95"] = np.nanpercentile(audit_sur["T_pos"], 95, axis=0)
            audit_df["T_pos_audit_null99"] = np.nanpercentile(audit_sur["T_pos"], 99, axis=0)
            audit_df["p_pos_audit_surrogate"] = p_from_null(audit_sur["T_pos"], Tpos_obs)
            audit_df["audit_strength_excess"] = audit_df["T_pos_obs"] - audit_df["T_pos_audit_null95"]
            audit_df["audit_strength_ratio"] = audit_df["T_pos_obs"] / audit_df["T_pos_audit_null95"].replace(0, np.nan)
            audit_tables.append(audit_df)
        null_tables.append(null_df)

        logger.info("  directional year-bootstrap n=%d", settings.n_direction_bootstrap)
        boot = _bootstrap_direction_for_window(panel, ext_days, target_days, src_idx, tgt_idx, null_df["T_pos_null95"].to_numpy(dtype=float), null_df["T_neg_null95"].to_numpy(dtype=float), null_df["T_0_null95"].to_numpy(dtype=float), settings, rng)
        dir_df = null_df[["window", "source", "target"]].copy()
        dir_df["D_pos_neg"] = null_df["S_pos"] - null_df["S_neg"]
        dir_df["D_pos_0"] = null_df["S_pos"] - null_df["S_0"]
        for prefix, arr in [("D_pos_neg", boot["D_pos_neg"]), ("D_pos_0", boot["D_pos_0"])]:
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
    ar1_year_all = pd.concat(ar1_year_tables, ignore_index=True)
    audit_all = pd.concat(audit_tables, ignore_index=True) if audit_tables else pd.DataFrame()
    null_all["q_pos_global"] = fdr_bh(null_all["p_pos_surrogate"])
    null_all["q_pos_within_window"] = np.nan
    for _, sub_idx in null_all.groupby("window").groups.items():
        null_all.loc[sub_idx, "q_pos_within_window"] = fdr_bh(null_all.loc[sub_idx, "p_pos_surrogate"])
    if not audit_all.empty:
        audit_all["q_pos_audit_global"] = fdr_bh(audit_all["p_pos_audit_surrogate"])
        audit_all["q_pos_audit_within_window"] = np.nan
        for _, sub_idx in audit_all.groupby("window").groups.items():
            audit_all.loc[sub_idx, "q_pos_audit_within_window"] = fdr_bh(audit_all.loc[sub_idx, "p_pos_audit_surrogate"])

    classified = _classify_rows(null_all, dir_all, settings)
    if not audit_all.empty:
        cols = ["window", "source", "target", "p_pos_audit_surrogate", "q_pos_audit_within_window", "q_pos_audit_global", "T_pos_audit_null95", "audit_strength_excess", "audit_strength_ratio"]
        classified = classified.merge(audit_all[cols], on=["window", "source", "target"], how="left")
    classified = _attach_phi_risk(classified, ar1_params_all)
    tier_records = classified.apply(_classify_pc1_tier, axis=1, result_type="expand")
    classified["pc1_evidence_tier"] = tier_records[0]
    classified["pc1_evidence_tier_reason"] = tier_records[1]
    classified["recommended_usage"] = tier_records[2]

    pair_cols = [
        "window", "source", "target", "source_object", "target_object", "source_family", "target_family", "family_direction",
        "positive_peak_lag", "positive_peak_signed_r", "positive_peak_abs_r",
        "negative_peak_lag", "negative_peak_signed_r", "negative_peak_abs_r",
        "lag0_signed_r", "lag0_abs_r", "p_pos_surrogate", "q_pos_within_window", "q_pos_global",
        "p_pos_audit_surrogate", "q_pos_audit_within_window", "T_pos_null95", "T_pos_audit_null95",
        "strength_excess", "strength_ratio", "statistical_support", "strength_tier", "same_day_coupling_flag",
        "direction_label", "pc1_lead_lag_label", "pc1_lead_lag_group", "pc1_evidence_tier", "pc1_evidence_tier_reason",
        "failure_reason", "risk_note", "suggested_reverse_direction", "sample_status", "pair_phi_risk",
    ]
    pair_summary = classified[[c for c in pair_cols if c in classified.columns]].copy()
    family_rollup = pair_summary.groupby(["window", "family_direction"], dropna=False).agg(
        n_pairs=("source", "count"),
        n_pc1_yes=("pc1_lead_lag_group", lambda s: int((s == "PC1_lead_lag_yes").sum())),
        n_pc1_ambiguous=("pc1_lead_lag_group", lambda s: int((s == "PC1_lead_lag_ambiguous").sum())),
        n_same_day=("same_day_coupling_flag", "sum"),
        max_positive_abs_r=("positive_peak_abs_r", "max"),
        max_lag0_abs_r=("lag0_abs_r", "max"),
    ).reset_index()

    return {
        "eof_pc1_lead_lag_curve_long": curve_all,
        "eof_pc1_null_summary": null_all,
        "eof_pc1_directional_robustness": dir_all,
        "eof_pc1_pair_summary": pair_summary,
        "eof_pc1_surrogate_ar1_params": ar1_params_all,
        "eof_pc1_surrogate_yearwise_scale": ar1_year_all,
        "eof_pc1_audit_surrogate_null_summary": audit_all,
        "eof_pc1_family_rollup": family_rollup,
    }
