from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .eof_pc1_v1_adjudication_settings import EOFPC1V1AdjudicationSettings


@dataclass
class LeadLagResult:
    window: str
    source: str
    target: str
    best_positive_lag: float
    best_positive_corr: float
    best_positive_abs_corr: float
    lag0_corr: float
    lag0_abs_corr: float
    lag_minus_tau0_abs: float
    n_pairs_best_positive: int
    n_pairs_lag0: int
    classification: str
    lag_tau0_class: str
    classification_reason: str


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = np.asarray(x[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)
    if np.nanstd(xx) <= 0 or np.nanstd(yy) <= 0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def zscore_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        vals = out[c].to_numpy(dtype=float)
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if (not np.isfinite(sd)) or sd <= 0:
            out[c] = np.nan
        else:
            out[c] = (vals - mu) / sd
    return out


def pca_pc1_scores(df: pd.DataFrame, columns: List[str], out_col: str) -> pd.DataFrame:
    keep = [c for c in columns if c in df.columns]
    base = df[["year", "day"] + keep].copy()
    if not keep:
        base[out_col] = np.nan
        return base[["year", "day", out_col]]
    Xdf = zscore_columns(base, keep)
    X = Xdf[keep].to_numpy(dtype=float)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, col_mean[None, :])
    X = X - np.nanmean(X, axis=0, keepdims=True)
    if X.shape[0] < 3 or X.shape[1] < 1:
        base[out_col] = np.nan
        return base[["year", "day", out_col]]
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        score = U[:, 0] * S[0]
        # Deterministic sign: align with the first available input column.
        ref = X[:, 0]
        if safe_corr(score, ref) < 0:
            score = -score
        base[out_col] = score
    except Exception:
        base[out_col] = np.nan
    return base[["year", "day", out_col]]


def linear_r2(y: np.ndarray, X: np.ndarray) -> float:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(mask.sum()) < max(3, X.shape[1] + 2):
        return float("nan")
    yy = y[mask].astype(float)
    XX = X[mask].astype(float)
    if np.nanstd(yy) <= 0:
        return float("nan")
    XX = np.column_stack([np.ones(len(XX)), XX])
    try:
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        pred = XX @ beta
        ss_res = float(np.sum((yy - pred) ** 2))
        ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    except Exception:
        return float("nan")


def alignment_class(r2: float, settings: EOFPC1V1AdjudicationSettings) -> str:
    if not np.isfinite(r2):
        return "not_evaluable"
    if r2 >= settings.aligned_r2_threshold:
        return "aligned_with_old_index_space"
    if r2 >= settings.partial_r2_threshold:
        return "partially_aligned_with_old_index_space"
    return "not_aligned_with_old_index_space"


def compute_pc1_old_index_alignment(
    p_pc: pd.DataFrame,
    v_pc: pd.DataFrame,
    idx: pd.DataFrame,
    settings: EOFPC1V1AdjudicationSettings,
) -> pd.DataFrame:
    rows = []
    merged_v = v_pc.merge(idx, on=["year", "day"], how="left")
    merged_p = p_pc.merge(idx, on=["year", "day"], how="left")
    specs = [
        ("V", merged_v, "V_PC1", list(settings.old_v_indices)),
        ("P", merged_p, "P_PC1", list(settings.old_p_indices)),
    ]
    for field, df, pc_col, cols in specs:
        for window, (start, end) in settings.windows.items():
            sub = df[(df["day"] >= start) & (df["day"] <= end)].copy()
            available = [c for c in cols if c in sub.columns]
            best_name = ""
            best_corr = np.nan
            best_abs = -np.inf
            for c in available:
                r = safe_corr(sub[pc_col].to_numpy(dtype=float), sub[c].to_numpy(dtype=float))
                if np.isfinite(r) and abs(r) > best_abs:
                    best_name = c
                    best_corr = r
                    best_abs = abs(r)
            X = sub[available].to_numpy(dtype=float) if available else np.empty((len(sub), 0))
            r2 = linear_r2(sub[pc_col].to_numpy(dtype=float), X) if available else np.nan
            rows.append({
                "window": window,
                "field": field,
                "pc_mode": pc_col,
                "n_samples": int(len(sub)),
                "n_old_indices_available": int(len(available)),
                "best_matching_old_index": best_name,
                "best_corr": best_corr,
                "best_abs_corr": abs(best_corr) if np.isfinite(best_corr) else np.nan,
                "multi_index_R2": r2,
                "alignment_class": alignment_class(r2, settings),
            })
    return pd.DataFrame(rows)


def _lag_pair(df: pd.DataFrame, source_col: str, target_col: str, target_days: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    src = df[["year", "day", source_col]].copy()
    tgt = df[["year", "day", target_col]].copy()
    tgt = tgt[tgt["day"].isin(target_days)].copy()
    tgt["source_day"] = tgt["day"] - lag
    src = src.rename(columns={"day": "source_day", source_col: "x"})
    tgt = tgt.rename(columns={target_col: "y"})
    m = tgt.merge(src[["year", "source_day", "x"]], on=["year", "source_day"], how="left")
    return m["x"].to_numpy(dtype=float), m["y"].to_numpy(dtype=float)


def classify_lag(best_abs: float, lag0_abs: float, margin: float, floor: float, n_best: int, min_pairs: int) -> tuple[str, str, str]:
    if n_best < min_pairs or not np.isfinite(best_abs):
        return "not_evaluable", "not_evaluable", "insufficient_pairs"
    if (best_abs >= floor) and np.isfinite(lag0_abs) and (best_abs > lag0_abs + margin):
        return "lead_lag_yes_clear", "stable_lag_dominant", "best positive lag exceeds lag0 by margin"
    if np.isfinite(lag0_abs) and (lag0_abs > best_abs + margin):
        return "lead_lag_ambiguous_same_day_dominant", "tau0_abs_dominant", "lag0 absolute correlation exceeds best positive lag"
    if max(best_abs, lag0_abs if np.isfinite(lag0_abs) else -np.inf) >= floor:
        return "lead_lag_ambiguous_coupled_or_close", "tau0_coupled_or_close", "positive lag and lag0 are close or mixed"
    return "lead_lag_no_not_supported", "weak_or_not_supported", "weak absolute association"


def leadlag_one_pair(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    window: str,
    settings: EOFPC1V1AdjudicationSettings,
) -> LeadLagResult:
    start, end = settings.windows[window]
    target_days = np.arange(start, end + 1)
    rows = []
    for lag in range(0, settings.max_lag + 1):
        x, y = _lag_pair(df, source_col, target_col, target_days, lag)
        mask = np.isfinite(x) & np.isfinite(y)
        r = safe_corr(x, y)
        rows.append({"lag": lag, "corr": r, "abs_corr": abs(r) if np.isfinite(r) else np.nan, "n_pairs": int(mask.sum())})
    res = pd.DataFrame(rows)
    pos = res[res["lag"] >= 1].copy()
    if pos["abs_corr"].notna().any():
        best = pos.loc[pos["abs_corr"].idxmax()]
        best_lag = float(best["lag"])
        best_corr = float(best["corr"])
        best_abs = float(best["abs_corr"])
        n_best = int(best["n_pairs"])
    else:
        best_lag = np.nan; best_corr = np.nan; best_abs = np.nan; n_best = 0
    lag0 = res[res["lag"] == 0].iloc[0]
    lag0_corr = float(lag0["corr"])
    lag0_abs = float(lag0["abs_corr"])
    n0 = int(lag0["n_pairs"])
    label, tau0_class, reason = classify_lag(best_abs, lag0_abs, settings.lag_tau0_margin, settings.stable_abs_corr_floor, n_best, settings.min_pairs)
    return LeadLagResult(
        window=window,
        source=source_col,
        target=target_col,
        best_positive_lag=best_lag,
        best_positive_corr=best_corr,
        best_positive_abs_corr=best_abs,
        lag0_corr=lag0_corr,
        lag0_abs_corr=lag0_abs,
        lag_minus_tau0_abs=best_abs - lag0_abs if np.isfinite(best_abs) and np.isfinite(lag0_abs) else np.nan,
        n_pairs_best_positive=n_best,
        n_pairs_lag0=n0,
        classification=label,
        lag_tau0_class=tau0_class,
        classification_reason=reason,
    )


def leadlag_table(df: pd.DataFrame, source_col: str, target_col: str, settings: EOFPC1V1AdjudicationSettings, source_label: str | None = None, target_label: str | None = None) -> pd.DataFrame:
    out = []
    for window in settings.windows:
        r = leadlag_one_pair(df, source_col, target_col, window, settings)
        out.append({
            "window": r.window,
            "source": source_label or r.source,
            "target": target_label or r.target,
            "best_positive_lag": r.best_positive_lag,
            "best_positive_corr": r.best_positive_corr,
            "best_positive_abs_corr": r.best_positive_abs_corr,
            "lag0_corr": r.lag0_corr,
            "lag0_abs_corr": r.lag0_abs_corr,
            "lag_minus_tau0_abs": r.lag_minus_tau0_abs,
            "n_pairs_best_positive": r.n_pairs_best_positive,
            "n_pairs_lag0": r.n_pairs_lag0,
            "classification": r.classification,
            "lag_tau0_class": r.lag_tau0_class,
            "classification_reason": r.classification_reason,
        })
    return pd.DataFrame(out)


def build_old_index_pc1_leadlag(idx: pd.DataFrame, settings: EOFPC1V1AdjudicationSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    v_pc = pca_pc1_scores(idx, list(settings.old_v_indices), "old_V_index_PC1")
    p_pc = pca_pc1_scores(idx, list(settings.old_p_indices), "old_P_index_PC1")
    merged = v_pc.merge(p_pc, on=["year", "day"], how="inner")
    ll = leadlag_table(merged, "old_V_index_PC1", "old_P_index_PC1", settings)
    var_summary = pd.DataFrame([
        {"pc_name": "old_V_index_PC1", "input_indices": ";".join([c for c in settings.old_v_indices if c in idx.columns])},
        {"pc_name": "old_P_index_PC1", "input_indices": ";".join([c for c in settings.old_p_indices if c in idx.columns])},
    ])
    return ll, var_summary


def make_doy_residual(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    out = df[["year", "day", col]].copy()
    clim = out.groupby("day")[col].transform("mean")
    out[out_col] = out[col] - clim
    return out[["year", "day", out_col]]


def window_centered_leadlag(p_pc: pd.DataFrame, v_pc: pd.DataFrame, settings: EOFPC1V1AdjudicationSettings) -> pd.DataFrame:
    # For each window, center source and target PC1 within the target window context.
    # Source days can extend outside target window due to lag; we center using all available
    # values from the source days actually sampled by lags 0..max_lag for that window.
    rows = []
    base = v_pc[["year", "day", "V_PC1"]].merge(p_pc[["year", "day", "P_PC1"]], on=["year", "day"], how="inner")
    for window, (start, end) in settings.windows.items():
        # Build a temporary dataframe with centered values for this window only.
        target_days = np.arange(start, end + 1)
        source_days = np.unique(np.concatenate([target_days - lag for lag in range(0, settings.max_lag + 1)]))
        tmp = base.copy()
        v_mask = tmp["day"].isin(source_days)
        p_mask = tmp["day"].isin(target_days)
        tmp["V_PC1_window_centered"] = tmp["V_PC1"]
        tmp["P_PC1_window_centered"] = tmp["P_PC1"]
        tmp.loc[v_mask, "V_PC1_window_centered"] = tmp.loc[v_mask, "V_PC1"] - tmp.loc[v_mask, "V_PC1"].mean()
        tmp.loc[p_mask, "P_PC1_window_centered"] = tmp.loc[p_mask, "P_PC1"] - tmp.loc[p_mask, "P_PC1"].mean()
        r = leadlag_one_pair(tmp, "V_PC1_window_centered", "P_PC1_window_centered", window, settings)
        rows.append({
            "window": window,
            "pc1_series_mode": "window_centered",
            "best_positive_lag": r.best_positive_lag,
            "best_positive_corr": r.best_positive_corr,
            "best_positive_abs_corr": r.best_positive_abs_corr,
            "lag0_corr": r.lag0_corr,
            "lag0_abs_corr": r.lag0_abs_corr,
            "lag_minus_tau0_abs": r.lag_minus_tau0_abs,
            "classification": r.classification,
            "lag_tau0_class": r.lag_tau0_class,
            "classification_reason": r.classification_reason,
        })
    return pd.DataFrame(rows)


def build_pc1_seasonal_progression_control(p_pc: pd.DataFrame, v_pc: pd.DataFrame, settings: EOFPC1V1AdjudicationSettings) -> pd.DataFrame:
    base = v_pc[["year", "day", "V_PC1"]].merge(p_pc[["year", "day", "P_PC1"]], on=["year", "day"], how="inner")
    raw = leadlag_table(base, "V_PC1", "P_PC1", settings)
    raw.insert(1, "pc1_series_mode", "raw")

    v_res = make_doy_residual(v_pc, "V_PC1", "V_PC1_doy_residual")
    p_res = make_doy_residual(p_pc, "P_PC1", "P_PC1_doy_residual")
    res = v_res.merge(p_res, on=["year", "day"], how="inner")
    doy = leadlag_table(res, "V_PC1_doy_residual", "P_PC1_doy_residual", settings)
    doy.insert(1, "pc1_series_mode", "doy_residual")

    win = window_centered_leadlag(p_pc, v_pc, settings)
    all_df = pd.concat([raw, doy, win], ignore_index=True)
    raw_ref = raw[["window", "best_positive_abs_corr"]].rename(columns={"best_positive_abs_corr": "raw_best_positive_abs_corr"})
    all_df = all_df.merge(raw_ref, on="window", how="left")
    all_df["corr_drop_from_raw_abs"] = all_df["raw_best_positive_abs_corr"] - all_df["best_positive_abs_corr"]
    return all_df


def load_v1_1_old_pair_counts(settings: EOFPC1V1AdjudicationSettings) -> pd.DataFrame:
    if not settings.v1_1_classified_pairs_path.exists():
        return pd.DataFrame(columns=["window", "old_old_pair_count", "old_old_stable_lag_count"])
    df = pd.read_csv(settings.v1_1_classified_pairs_path, encoding="utf-8-sig")
    src_type = "source_index_type" if "source_index_type" in df.columns else None
    tgt_type = "target_index_type" if "target_index_type" in df.columns else None
    label_col = "lag_vs_tau0_label" if "lag_vs_tau0_label" in df.columns else ("v1_stability_judgement" if "v1_stability_judgement" in df.columns else "classification")
    if src_type and tgt_type:
        old = df[(df[src_type] == "old_v1") & (df[tgt_type] == "old_v1")].copy()
    else:
        old = df.copy()
    rows = []
    for w, sub in old.groupby("window"):
        rows.append({
            "window": w,
            "old_old_pair_count": int(len(sub)),
            "old_old_stable_lag_count": int((sub[label_col] == "stable_lag_dominant").sum()) if label_col in sub.columns else np.nan,
        })
    return pd.DataFrame(rows)


def build_eof_pc1_v1_style_classification(p_pc: pd.DataFrame, v_pc: pd.DataFrame, settings: EOFPC1V1AdjudicationSettings) -> pd.DataFrame:
    base = v_pc[["year", "day", "V_PC1"]].merge(p_pc[["year", "day", "P_PC1"]], on=["year", "day"], how="inner")
    out = leadlag_table(base, "V_PC1", "P_PC1", settings)
    out["source"] = "V_PC1"
    out["target"] = "P_PC1"
    return out


def build_adjudication_diagnosis(
    alignment: pd.DataFrame,
    old_pc_ll: pd.DataFrame,
    seasonal: pd.DataFrame,
    eof_v1_style: pd.DataFrame,
    old_pair_counts: pd.DataFrame,
    settings: EOFPC1V1AdjudicationSettings,
) -> pd.DataFrame:
    rows = []
    # T3 focus evidence.
    a_t3 = alignment[alignment["window"] == "T3"]
    v_align = a_t3[a_t3["field"] == "V"]
    p_align = a_t3[a_t3["field"] == "P"]
    v_class = v_align["alignment_class"].iloc[0] if len(v_align) else "not_evaluable"
    p_class = p_align["alignment_class"].iloc[0] if len(p_align) else "not_evaluable"
    both_aligned = v_class == "aligned_with_old_index_space" and p_class == "aligned_with_old_index_space"
    rows.append({
        "diagnosis_id": "pc1_represents_v1_old_index_space",
        "support_level": "supported" if both_aligned else ("mixed" if "partially" in (v_class + p_class) else "not_supported"),
        "primary_evidence": f"T3 V_PC1 alignment={v_class}; T3 P_PC1 alignment={p_class}.",
        "counter_evidence": "PC1 cannot adjudicate old-index pair weakening if either side is not aligned with the old-index space.",
        "allowed_statement": "EOF-PC1 is in the same problem space as V1 old indices only if both V and P PC1 align with old-index spaces.",
        "forbidden_statement": "Do not use EOF-PC1 to refute V1 T3 weakening when PC1 is not aligned with V1 old-index space.",
    })

    old_t3 = old_pc_ll[old_pc_ll["window"] == "T3"]
    old_pc_class = old_t3["lag_tau0_class"].iloc[0] if len(old_t3) else "not_evaluable"
    pair_t3 = old_pair_counts[old_pair_counts["window"] == "T3"]
    pair_count = int(pair_t3["old_old_stable_lag_count"].iloc[0]) if len(pair_t3) and np.isfinite(pair_t3["old_old_stable_lag_count"].iloc[0]) else np.nan
    rows.append({
        "diagnosis_id": "old_index_aggregate_mode_weakens_in_T3",
        "support_level": "not_supported" if old_pc_class == "stable_lag_dominant" else ("supported" if old_pc_class in {"weak_or_not_supported", "not_evaluable"} else "mixed"),
        "primary_evidence": f"T3 old-index PC1 lag_tau0_class={old_pc_class}; T3 old-old stable pair count={pair_count}.",
        "counter_evidence": "If old-index aggregate PC1 remains stable while individual old pairs drop, the weakening is pair-level rather than aggregate-space decoupling.",
        "allowed_statement": "This diagnostic separates pair-level weakening from old-index aggregate-mode weakening.",
        "forbidden_statement": "Do not equate many failed individual pairs with aggregate old-index space decoupling without checking old-index PC1.",
    })

    s_t3 = seasonal[(seasonal["window"] == "T3") & (seasonal["pc1_series_mode"].isin(["raw", "doy_residual", "window_centered"]))]
    raw_abs = s_t3.loc[s_t3["pc1_series_mode"] == "raw", "best_positive_abs_corr"]
    resid_abs = s_t3.loc[s_t3["pc1_series_mode"] == "doy_residual", "best_positive_abs_corr"]
    win_abs = s_t3.loc[s_t3["pc1_series_mode"] == "window_centered", "best_positive_abs_corr"]
    raw_val = float(raw_abs.iloc[0]) if len(raw_abs) else np.nan
    resid_val = float(resid_abs.iloc[0]) if len(resid_abs) else np.nan
    win_val = float(win_abs.iloc[0]) if len(win_abs) else np.nan
    survives = np.isfinite(raw_val) and np.isfinite(resid_val) and (resid_val >= raw_val - 0.05)
    rows.append({
        "diagnosis_id": "pc1_relation_survives_seasonal_control",
        "support_level": "supported" if survives else "not_supported_or_sensitive",
        "primary_evidence": f"T3 raw abs r={raw_val:.3f}; doy-residual abs r={resid_val:.3f}; window-centered abs r={win_val:.3f}.",
        "counter_evidence": "If PC1 relation drops after day/window controls, EOF-PC1 may mainly reflect seasonal progression/background mode.",
        "allowed_statement": "EOF-PC1 has more adjudication value if its relation survives seasonal/background controls.",
        "forbidden_statement": "Do not treat raw EOF-PC1 correlation as V1-style evidence if it collapses after seasonal controls.",
    })

    e_t3 = eof_v1_style[eof_v1_style["window"] == "T3"]
    eof_class = e_t3["lag_tau0_class"].iloc[0] if len(e_t3) else "not_evaluable"
    rows.append({
        "diagnosis_id": "pc1_has_v1_style_stable_lag_in_T3",
        "support_level": "supported" if eof_class == "stable_lag_dominant" else "not_supported",
        "primary_evidence": f"T3 EOF-PC1 lag_tau0_class={eof_class}.",
        "counter_evidence": "EOF-PC1 cannot refute stable-lag pair weakening if EOF-PC1 itself is tau0-coupled, same-day dominant, or weak under the same lag-vs-tau0 logic.",
        "allowed_statement": "Use V1-style lag-vs-tau0 classification, not raw correlation alone, when comparing EOF-PC1 with V1/V1_1.",
        "forbidden_statement": "Do not say EOF-PC1 contradicts V1 if EOF-PC1 is not stable-lag-dominant in T3.",
    })

    can_adjudicate = both_aligned and survives and eof_class == "stable_lag_dominant"
    rows.append({
        "diagnosis_id": "eof_pc1_can_adjudicate_v1_t3_weakening",
        "support_level": "supported" if can_adjudicate else "not_supported",
        "primary_evidence": f"both_aligned={both_aligned}; seasonal_control_survives={survives}; EOF-PC1_T3_class={eof_class}.",
        "counter_evidence": "EOF-PC1 needs old-index alignment, seasonal-control robustness, and V1-style stable lag in T3 before it can adjudicate V1's T3 weakening.",
        "allowed_statement": "Only if all prerequisites pass can EOF-PC1 be used as a serious counterpoint to V1 T3 weakening.",
        "forbidden_statement": "Do not use EOF-PC1 as a judge of V1 T3 weakening when any prerequisite fails.",
    })
    return pd.DataFrame(rows)
