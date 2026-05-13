from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .eof_pc1_interpretability_settings import EOFPC1InterpretabilitySettings


@dataclass
class EOFResult:
    field_name: str
    lat: np.ndarray
    lon: np.ndarray
    years: np.ndarray
    days: np.ndarray
    anomaly_field: np.ndarray  # year, day, lat, lon, domain subset, unweighted
    sample_mean: np.ndarray  # feature mean before sample centering, unweighted flattened valid features
    valid_feature_mask: np.ndarray
    weights_flat: np.ndarray
    pcs: pd.DataFrame
    eofs_unweighted: np.ndarray  # mode, feature valid, unweighted loading
    eofs_weighted: np.ndarray  # mode, feature valid, weighted loading
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray


def field_from_npz(npz, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name])
    raise KeyError(f"None of {list(names)} found. Available fields: {npz.files}")


def mask_between(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr) & (arr >= min(lo, hi)) & (arr <= max(lo, hi))


def cos_weights(lat: np.ndarray) -> np.ndarray:
    out = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))
    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0
    return out


def load_domain_fields(settings: EOFPC1InterpretabilitySettings) -> dict[str, object]:
    if not settings.input_smoothed_fields.exists():
        raise FileNotFoundError(f"smoothed_fields not found: {settings.input_smoothed_fields}")
    with np.load(settings.input_smoothed_fields, allow_pickle=False) as npz:
        precip = field_from_npz(npz, ["precip_smoothed", "precip", "P_smoothed", "P"])
        v850 = field_from_npz(npz, ["v850_smoothed", "v850", "V850_smoothed", "V850"])
        lat = field_from_npz(npz, ["lat", "latitude"])
        lon = field_from_npz(npz, ["lon", "longitude"])
        years = field_from_npz(npz, ["years", "year"])
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    years = np.asarray(years).astype(int)

    lat_idx = np.where(mask_between(lat, *settings.eof_lat_range))[0]
    lon_idx = np.where(mask_between(lon, *settings.eof_lon_range))[0]
    stride = max(int(settings.spatial_stride), 1)
    lat_idx = lat_idx[::stride]
    lon_idx = lon_idx[::stride]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError("EOF domain mask is empty; check lat/lon ranges.")

    # Copy domain subset to avoid holding unneeded global map in subsequent operations.
    p_sub = np.asarray(precip[:, :, lat_idx, :][:, :, :, lon_idx], dtype=np.float32)
    v_sub = np.asarray(v850[:, :, lat_idx, :][:, :, :, lon_idx], dtype=np.float32)
    return {
        "precip": p_sub,
        "v850": v_sub,
        "lat": lat[lat_idx].astype(float),
        "lon": lon[lon_idx].astype(float),
        "years": years,
        "full_lat_min": float(np.nanmin(lat)),
        "full_lat_max": float(np.nanmax(lat)),
        "full_lon_min": float(np.nanmin(lon)),
        "full_lon_max": float(np.nanmax(lon)),
    }


def make_doy_anomaly(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    clim = np.nanmean(arr, axis=0, keepdims=True)
    return arr - clim


def prepare_field_matrix(field: np.ndarray, lat: np.ndarray, settings: EOFPC1InterpretabilitySettings) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if settings.eof_value_mode == "doy_anomaly":
        anom = make_doy_anomaly(field)
    elif settings.eof_value_mode == "raw_centered":
        anom = np.asarray(field, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported eof_value_mode: {settings.eof_value_mode}")

    n_year, n_day, n_lat, n_lon = anom.shape
    X0 = anom.reshape(n_year * n_day, n_lat * n_lon).astype(np.float32)
    valid = np.isfinite(X0).mean(axis=0) >= 0.80
    X = X0[:, valid]
    feat_mean = np.nanmean(X, axis=0).astype(np.float32)
    # sample-centered EOF matrix; nan filled by feature mean before centering -> zero.
    X = np.where(np.isfinite(X), X, feat_mean[None, :]).astype(np.float32)
    X = X - feat_mean[None, :]

    lat_grid = np.repeat(lat[:, None], n_lon, axis=1).reshape(-1)[valid]
    w = np.sqrt(np.maximum(cos_weights(lat_grid), 1e-6)).astype(np.float32) if settings.use_coslat_weight else np.ones_like(lat_grid, dtype=np.float32)
    Xw = X * w[None, :]
    return Xw, anom, valid, feat_mean, w


def topk_eof_iterative(Xw: np.ndarray, n_modes: int, n_iter: int, random_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PCs, weighted EOFs, singular values for top-k EOFs.

    Deterministic subspace iteration; designed to avoid forming a huge covariance matrix.
    This is a numerical top-k EOF solver, not a scientific approximation of the domain.
    """
    n_samples, n_features = Xw.shape
    k = min(int(n_modes), n_samples - 1, n_features)
    rng = np.random.default_rng(int(random_seed))
    Q = rng.normal(size=(n_features, k)).astype(np.float32)
    Q, _ = np.linalg.qr(Q)
    Q = Q.astype(np.float32)
    for _ in range(max(int(n_iter), 1)):
        Z = Xw @ Q
        Y = Xw.T @ Z
        Q, _ = np.linalg.qr(Y)
        Q = Q.astype(np.float32)
    B = Xw @ Q
    U, S, Vt_small = np.linalg.svd(B, full_matrices=False)
    eofs_weighted = (Q @ Vt_small.T).T.astype(np.float32)  # mode, feature
    pcs = (U[:, :k] * S[None, :k]).astype(np.float32)
    return pcs[:, :k], eofs_weighted[:k, :], S[:k].astype(float)


def compute_eof_for_field(field_name: str, field: np.ndarray, lat: np.ndarray, lon: np.ndarray, years: np.ndarray, settings: EOFPC1InterpretabilitySettings) -> EOFResult:
    Xw, anom, valid, feat_mean, w = prepare_field_matrix(field, lat, settings)
    pcs_arr, eofs_weighted, svals = topk_eof_iterative(Xw, settings.n_modes, settings.n_iter, settings.random_seed + (0 if field_name == "P" else 99))
    total_var = float(np.nansum(Xw * Xw))
    evr = (svals ** 2) / total_var if total_var > 0 else np.full_like(svals, np.nan)
    eofs_unweighted = eofs_weighted / w[None, :]

    n_year, n_day = field.shape[0], field.shape[1]
    grid = pd.MultiIndex.from_product([years.astype(int), np.arange(1, n_day + 1)], names=["year", "day"]).to_frame(index=False)
    pcs = grid.copy()
    for i in range(pcs_arr.shape[1]):
        pcs[f"{field_name}_PC{i+1}"] = pcs_arr[:, i]
    return EOFResult(
        field_name=field_name,
        lat=lat,
        lon=lon,
        years=years.astype(int),
        days=np.arange(1, n_day + 1),
        anomaly_field=anom,
        sample_mean=feat_mean,
        valid_feature_mask=valid,
        weights_flat=w,
        pcs=pcs,
        eofs_unweighted=eofs_unweighted,
        eofs_weighted=eofs_weighted,
        singular_values=svals,
        explained_variance_ratio=evr,
    )


def feature_grid(result: EOFResult, mode: int) -> np.ndarray:
    n_lat, n_lon = len(result.lat), len(result.lon)
    arr = np.full(n_lat * n_lon, np.nan, dtype=float)
    arr[result.valid_feature_mask] = result.eofs_unweighted[mode - 1, :]
    return arr.reshape(n_lat, n_lon)


def region_mask(lat: np.ndarray, lon: np.ndarray, region: Dict[str, Tuple[float, float]]) -> np.ndarray:
    lat_mask = mask_between(lat, *region["lat"])
    lon_mask = mask_between(lon, *region["lon"])
    return lat_mask[:, None] & lon_mask[None, :]


def area_weighted_map_mean(data2d: np.ndarray, lat: np.ndarray, lon: np.ndarray, region: Dict[str, Tuple[float, float]]) -> float:
    mask = region_mask(lat, lon, region)
    if not np.any(mask):
        return float("nan")
    w_lat = cos_weights(lat)[:, None]
    weights = np.where(mask, w_lat, 0.0)
    denom = np.nansum(weights)
    if denom <= 0:
        return float("nan")
    return float(np.nansum(data2d * weights) / denom)


def loading_region_summary(result: EOFResult, regions: Dict[str, Dict[str, Tuple[float, float]]]) -> pd.DataFrame:
    rows = []
    for mode in range(1, result.eofs_unweighted.shape[0] + 1):
        grid = feature_grid(result, mode)
        for region_name, region in regions.items():
            mask = region_mask(result.lat, result.lon, region)
            vals = grid[mask]
            rows.append({
                "field": result.field_name,
                "mode": mode,
                "region": region_name,
                "loading_mean": float(np.nanmean(vals)) if vals.size else np.nan,
                "loading_abs_mean": float(np.nanmean(np.abs(vals))) if vals.size else np.nan,
                "loading_integrated": area_weighted_map_mean(grid, result.lat, result.lon, region),
                "loading_sign": "positive" if np.nanmean(vals) > 0 else ("negative" if np.nanmean(vals) < 0 else "near_zero"),
                "explained_variance_ratio": float(result.explained_variance_ratio[mode - 1]),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_loading_rank_within_mode"] = df.groupby(["field", "mode"])["loading_abs_mean"].rank(ascending=False, method="min")
    return df


def window_mask(df: pd.DataFrame, win: Tuple[int, int]) -> np.ndarray:
    return (df["day"].to_numpy() >= win[0]) & (df["day"].to_numpy() <= win[1])


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) <= 0 or np.nanstd(yy) <= 0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def pc_structural_index_correlation(p_eof: EOFResult, v_eof: EOFResult, settings: EOFPC1InterpretabilitySettings) -> pd.DataFrame:
    if not settings.input_v1_1_structural_indices.exists():
        return pd.DataFrame(columns=["field", "mode", "structural_index", "scope", "corr"])
    idx = pd.read_csv(settings.input_v1_1_structural_indices, encoding="utf-8-sig")
    idx["year"] = idx["year"].astype(int)
    idx["day"] = idx["day"].astype(int)
    rows = []
    for field_name, eof, names in [("P", p_eof, settings.p_structural_indices), ("V", v_eof, settings.v_structural_indices)]:
        merged = eof.pcs.merge(idx, on=["year", "day"], how="left")
        for mode in range(1, settings.n_modes + 1):
            pc_col = f"{field_name}_PC{mode}"
            for name in names:
                if name not in merged.columns:
                    rows.append({"field": field_name, "mode": mode, "structural_index": name, "scope": "missing", "corr": np.nan})
                    continue
                rows.append({"field": field_name, "mode": mode, "structural_index": name, "scope": "full_season", "corr": safe_corr(merged[pc_col].to_numpy(), merged[name].to_numpy())})
                for win_name in ["S3", "T3", "S4"]:
                    m = window_mask(merged, settings.windows[win_name])
                    rows.append({"field": field_name, "mode": mode, "structural_index": name, "scope": win_name, "corr": safe_corr(merged.loc[m, pc_col].to_numpy(), merged.loc[m, name].to_numpy())})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_corr"] = df["corr"].abs()
        df["abs_corr_rank_within_field_index_scope"] = df.groupby(["field", "structural_index", "scope"])["abs_corr"].rank(ascending=False, method="min")
    return df


def window_mean_region_observed(result: EOFResult, window: Tuple[int, int], region: Dict[str, Tuple[float, float]]) -> float:
    day_mask = (result.days >= window[0]) & (result.days <= window[1])
    arr = result.anomaly_field[:, day_mask, :, :]
    mean_map = np.nanmean(arr, axis=(0, 1))
    return area_weighted_map_mean(mean_map, result.lat, result.lon, region)


def eof_region_mean_loading(result: EOFResult, mode: int, region: Dict[str, Tuple[float, float]]) -> float:
    grid = feature_grid(result, mode)
    return area_weighted_map_mean(grid, result.lat, result.lon, region)


def pc_window_mean(result: EOFResult, mode: int, window: Tuple[int, int]) -> float:
    pc_col = f"{result.field_name}_PC{mode}"
    m = window_mask(result.pcs, window)
    return float(np.nanmean(result.pcs.loc[m, pc_col].to_numpy()))


def reconstruction_region_change_skill(result: EOFResult, settings: EOFPC1InterpretabilitySettings) -> pd.DataFrame:
    rows = []
    recon_specs = {"PC1": [1], "PC12": [1, 2], "PC123": [1, 2, 3]}
    for comp_name, (target_win, ref_win) in settings.reconstruction_comparisons.items():
        for region_name, region in settings.regions.items():
            obs_t = window_mean_region_observed(result, settings.windows[target_win], region)
            obs_r = window_mean_region_observed(result, settings.windows[ref_win], region)
            obs_delta = obs_t - obs_r
            for recon_name, modes in recon_specs.items():
                rec_t = 0.0
                rec_r = 0.0
                for mode in modes:
                    load_mean = eof_region_mean_loading(result, mode, region)
                    rec_t += pc_window_mean(result, mode, settings.windows[target_win]) * load_mean
                    rec_r += pc_window_mean(result, mode, settings.windows[ref_win]) * load_mean
                rec_delta = rec_t - rec_r
                same = bool(np.sign(obs_delta) == np.sign(rec_delta)) if np.isfinite(obs_delta) and np.isfinite(rec_delta) and obs_delta != 0 and rec_delta != 0 else False
                amp = abs(rec_delta) / abs(obs_delta) if np.isfinite(obs_delta) and abs(obs_delta) > 1e-12 else np.nan
                rows.append({
                    "field": result.field_name,
                    "reconstruction": recon_name,
                    "comparison": comp_name,
                    "region": region_name,
                    "observed_delta": obs_delta,
                    "reconstructed_delta": rec_delta,
                    "sign_match": same,
                    "amplitude_ratio": amp,
                    "error": rec_delta - obs_delta,
                })
    return pd.DataFrame(rows)


def simple_lead_lag(source_df: pd.DataFrame, target_df: pd.DataFrame, source_col: str, target_col: str, window: Tuple[int, int], max_lag: int, min_pairs: int) -> dict:
    merged = source_df[["year", "day", source_col]].merge(target_df[["year", "day", target_col]], on=["year", "day"], how="inner")
    by_key = merged.set_index(["year", "day"])
    rows = []
    for lag in range(0, max_lag + 1):
        xs = []
        ys = []
        for _, row in merged.iterrows():
            year = int(row["year"])
            tday = int(row["day"])
            if not (window[0] <= tday <= window[1]):
                continue
            sday = tday - lag
            if (year, sday) not in by_key.index:
                continue
            xs.append(float(by_key.loc[(year, sday), source_col]))
            ys.append(float(row[target_col]))
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        r = safe_corr(x, y) if len(x) >= min_pairs else np.nan
        rows.append({"lag": lag, "corr": r, "n_pairs": int(np.isfinite(x * y).sum())})
    out = pd.DataFrame(rows)
    pos = out[out["lag"] > 0].copy()
    if pos["corr"].abs().notna().any():
        best = pos.loc[pos["corr"].abs().idxmax()]
    else:
        best = pd.Series({"lag": np.nan, "corr": np.nan, "n_pairs": 0})
    lag0 = out.loc[out["lag"] == 0, "corr"].iloc[0]
    return {
        "best_positive_lag": int(best["lag"]) if np.isfinite(best["lag"]) else np.nan,
        "best_positive_corr": float(best["corr"]) if np.isfinite(best["corr"]) else np.nan,
        "lag0_corr": float(lag0) if np.isfinite(lag0) else np.nan,
        "lag_minus_tau0_abs_corr": abs(float(best["corr"])) - abs(float(lag0)) if np.isfinite(best["corr"]) and np.isfinite(lag0) else np.nan,
        "n_pairs_best_lag": int(best["n_pairs"]) if np.isfinite(best["n_pairs"]) else 0,
    }


def eof_pc_lead_lag_by_window(p_eof: EOFResult, v_eof: EOFResult, settings: EOFPC1InterpretabilitySettings) -> pd.DataFrame:
    rows = []
    for mode in range(1, min(3, settings.n_modes) + 1):
        src = f"V_PC{mode}"
        tgt = f"P_PC{mode}"
        for win_name, win in settings.windows.items():
            res = simple_lead_lag(v_eof.pcs, p_eof.pcs, src, tgt, win, settings.max_lag, settings.min_pairs)
            margin = res["lag_minus_tau0_abs_corr"]
            if np.isfinite(margin) and margin > settings.lag_tau0_margin:
                cls = "positive_lag_abs_dominant_over_tau0"
            elif np.isfinite(margin) and margin < -settings.lag_tau0_margin:
                cls = "tau0_abs_dominant"
            else:
                cls = "lag_tau0_close"
            rows.append({"window": win_name, "source_pc": src, "target_pc": tgt, **res, "simple_classification": cls})
    return pd.DataFrame(rows)


def eof_vs_structural_lead_lag(p_eof: EOFResult, v_eof: EOFResult, settings: EOFPC1InterpretabilitySettings) -> pd.DataFrame:
    rows = []
    # EOF PC same-mode pairs.
    pc_ll = eof_pc_lead_lag_by_window(p_eof, v_eof, settings)
    for _, r in pc_ll.iterrows():
        rows.append({
            "pair_type": "EOF_PC",
            "source": r["source_pc"],
            "target": r["target_pc"],
            "window": r["window"],
            "best_positive_lag": r["best_positive_lag"],
            "best_positive_corr": r["best_positive_corr"],
            "lag0_corr": r["lag0_corr"],
            "lag_minus_tau0_abs_corr": r["lag_minus_tau0_abs_corr"],
            "simple_classification": r["simple_classification"],
        })
    if settings.input_v1_1_structural_indices.exists():
        idx = pd.read_csv(settings.input_v1_1_structural_indices, encoding="utf-8-sig")
        idx["year"] = idx["year"].astype(int)
        idx["day"] = idx["day"].astype(int)
        for src, tgt in settings.selected_structural_pairs:
            if src not in idx.columns or tgt not in idx.columns:
                continue
            for win_name, win in settings.windows.items():
                res = simple_lead_lag(idx, idx, src, tgt, win, settings.max_lag, settings.min_pairs)
                margin = res["lag_minus_tau0_abs_corr"]
                if np.isfinite(margin) and margin > settings.lag_tau0_margin:
                    cls = "positive_lag_abs_dominant_over_tau0"
                elif np.isfinite(margin) and margin < -settings.lag_tau0_margin:
                    cls = "tau0_abs_dominant"
                else:
                    cls = "lag_tau0_close"
                rows.append({"pair_type": "structural_index", "source": src, "target": tgt, "window": win_name, **res, "simple_classification": cls})
    return pd.DataFrame(rows)


def build_diagnosis_tables(p_loading: pd.DataFrame, v_loading: pd.DataFrame, pc_corr: pd.DataFrame, recon_skill: pd.DataFrame, pc_ll: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def support_from_bool(x: bool) -> str:
        return "supported" if x else "not_supported"

    # PC1 loading highlat emphasis.
    p_pc1 = p_loading[(p_loading["field"] == "P") & (p_loading["mode"] == 1)]
    high_rank = p_pc1.loc[p_pc1["region"].eq("highlat_40_60"), "abs_loading_rank_within_mode"]
    main_rank = p_pc1.loc[p_pc1["region"].eq("main_28_35"), "abs_loading_rank_within_mode"]
    high_rank_val = float(high_rank.iloc[0]) if len(high_rank) else np.nan
    rows.append({
        "diagnosis_id": "pc1_contains_t3_highlat_p_structure",
        "support_level": support_from_bool(np.isfinite(high_rank_val) and high_rank_val <= 2),
        "evidence": f"P PC1 highlat_40_60 abs loading rank={high_rank_val}; main rank={float(main_rank.iloc[0]) if len(main_rank) else np.nan}",
        "allowed_statement": "Use PC1 for T3 high-latitude P interpretation only if high-latitude loading is among dominant PC1 regions.",
        "forbidden_statement": "Do not assume P PC1 represents the T3 high-latitude branch without checking loading regions.",
    })

    v_pc1 = v_loading[(v_loading["field"] == "V") & (v_loading["mode"] == 1)]
    v_high_rank = v_pc1.loc[v_pc1["region"].eq("highlat_40_60"), "abs_loading_rank_within_mode"]
    v_low_rank = v_pc1.loc[v_pc1["region"].eq("lowlat_20_30"), "abs_loading_rank_within_mode"]
    v_high_rank_val = float(v_high_rank.iloc[0]) if len(v_high_rank) else np.nan
    rows.append({
        "diagnosis_id": "pc1_contains_v_highlat_boundary_proxy_structure",
        "support_level": support_from_bool(np.isfinite(v_high_rank_val) and v_high_rank_val <= 2),
        "evidence": f"V PC1 highlat_40_60 abs loading rank={v_high_rank_val}; lowlat_20_30 rank={float(v_low_rank.iloc[0]) if len(v_low_rank) else np.nan}",
        "allowed_statement": "Use V PC1 as a boundary/high-latitude proxy only if high-latitude loading is dominant or structural-index correlations support it.",
        "forbidden_statement": "Do not treat V PC1 as V north-edge/retreat structure by default.",
    })

    # Reconstruction skill.
    for field in ["P", "V"]:
        sub = recon_skill[(recon_skill["field"] == field) & (recon_skill["reconstruction"] == "PC1") & (recon_skill["comparison"].isin(["T3_minus_S3", "S4_minus_T3"]))]
        focus_regions = ["highlat_40_60", "highlat_35_60", "main_28_35"] if field == "P" else ["highlat_40_60", "lowlat_20_30", "main_easm_domain"]
        fsub = sub[sub["region"].isin(focus_regions)]
        sign_rate = float(np.nanmean(fsub["sign_match"].astype(float))) if len(fsub) else np.nan
        amp_med = float(np.nanmedian(fsub["amplitude_ratio"])) if len(fsub) else np.nan
        ok = np.isfinite(sign_rate) and sign_rate >= 0.75 and np.isfinite(amp_med) and amp_med >= 0.35
        rows.append({
            "diagnosis_id": f"pc1_reconstructs_t3_{field.lower()}_changes",
            "support_level": support_from_bool(ok),
            "evidence": f"PC1 reconstruction sign_match_rate={sign_rate:.3f}; median amplitude_ratio={amp_med:.3f}; focus_regions={focus_regions}",
            "allowed_statement": "PC1 lead-lag can be used to discuss T3 changes only if PC1 reconstruction captures the relevant T3 change signs and amplitudes.",
            "forbidden_statement": "Do not use EOF-PC1 lead-lag to refute V1_1 if PC1-only reconstruction misses T3 observed structures.",
        })

    # Is T3 structural signal outside PC1?
    if not pc_corr.empty:
        candidates = pc_corr[(pc_corr["scope"].isin(["T3", "full_season"])) & (pc_corr["structural_index"].isin(["P_highlat_40_60_mean", "P_highlat_35_60_mean", "V_pos_north_edge_lat", "V_high_minus_low_35_55_minus_20_30"]))]
        stronger_nonpc1 = 0
        checked = 0
        for (_, idx, scope), g in candidates.groupby(["field", "structural_index", "scope"]):
            if g["corr"].notna().sum() < 2:
                continue
            checked += 1
            pc1_abs = g.loc[g["mode"] == 1, "corr"].abs().max()
            nonpc1_abs = g.loc[g["mode"].isin([2, 3]), "corr"].abs().max()
            if np.isfinite(nonpc1_abs) and np.isfinite(pc1_abs) and nonpc1_abs > pc1_abs + 0.10:
                stronger_nonpc1 += 1
        rows.append({
            "diagnosis_id": "t3_signal_in_pc2_or_pc3",
            "support_level": "supported" if checked and stronger_nonpc1 / checked >= 0.5 else "mixed_or_not_supported",
            "evidence": f"non-PC1 stronger cases={stronger_nonpc1}/{checked} for selected T3/highlat/boundary structural indices.",
            "allowed_statement": "If PC2/PC3 correlate better with structural indices, EOF-PC1 can miss T3 mode-shift signals.",
            "forbidden_statement": "Do not assume PC1 is sufficient just because it explains the most variance.",
        })

    # Can PC1 refute V1_1? Conservative decision.
    p_rec_ok = any(r["diagnosis_id"] == "pc1_reconstructs_t3_p_changes" and r["support_level"] == "supported" for r in rows)
    v_rec_ok = any(r["diagnosis_id"] == "pc1_reconstructs_t3_v_changes" and r["support_level"] == "supported" for r in rows)
    can_refute = p_rec_ok and v_rec_ok
    rows.append({
        "diagnosis_id": "pc1_result_can_refute_v1_1_t3_weakening",
        "support_level": "not_supported" if not can_refute else "requires_pairwise_comparison",
        "evidence": f"PC1 reconstruction support: P={p_rec_ok}, V={v_rec_ok}. If either fails, EOF-PC1 cannot refute V1_1 T3 weakening.",
        "allowed_statement": "EOF-PC1 can only challenge V1_1 after showing it represents the T3 P and V structures under dispute.",
        "forbidden_statement": "EOF-PC1 lead-lag continuity alone cannot be used to deny T3 pair-level weakening.",
    })

    return pd.DataFrame(rows)
