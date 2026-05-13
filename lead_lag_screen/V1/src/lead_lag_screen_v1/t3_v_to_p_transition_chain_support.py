# -*- coding: utf-8 -*-
"""Support-table and observed support-map helpers for transition-chain report."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .t3_v_to_p_transition_chain_settings import TransitionChainReportSettings

try:
    from .t3_v_to_p_field_explanation_math import corr_r2_beta_map
except Exception:  # pragma: no cover
    corr_r2_beta_map = None


def _classify_delta(delta: float, eps: float) -> str:
    if not np.isfinite(delta):
        return "unknown"
    if delta > eps:
        return "increase"
    if delta < -eps:
        return "decrease"
    return "near_zero"


def prepare_support_region_rows(prev_region_df: pd.DataFrame, settings: TransitionChainReportSettings) -> pd.DataFrame:
    df = prev_region_df.copy()
    # Backward-compatible BOM cleanup.
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    if "v_component" not in df.columns and "v_index" in df.columns:
        df = df.rename(columns={"v_index": "v_component"})
    required = ["window", "v_component", "region", "lag_label", settings.support_metric_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Previous region_response_summary.csv missing columns: {missing}")
    df = df[df["lag_label"].astype(str).eq(settings.support_map_lag_label)].copy()
    df = df[df["window"].isin(settings.window_order)]
    df = df[df["v_component"].isin(settings.v_components)]
    df = df[df["region"].isin(settings.regions.keys())]
    df["support_R2"] = pd.to_numeric(df[settings.support_metric_column], errors="coerce")
    return df[["window", "v_component", "region", "support_R2"]].copy()


def build_support_transition_matrix(support_df: pd.DataFrame, settings: TransitionChainReportSettings) -> pd.DataFrame:
    rows: List[dict] = []
    key = support_df.set_index(["v_component", "region", "window"])["support_R2"]
    for comp in settings.v_components:
        for region in settings.regions.keys():
            row = {"v_component": comp, "region": region}
            vals = []
            for w in settings.window_order:
                val = float(key.get((comp, region, w), np.nan))
                row[f"{w}_R2"] = val
                vals.append(val)
            rows.append(row)
    return pd.DataFrame(rows)


def build_support_region_delta(matrix_df: pd.DataFrame, settings: TransitionChainReportSettings) -> pd.DataFrame:
    rows: List[dict] = []
    comparisons = dict(settings.comparisons)
    for _, row in matrix_df.iterrows():
        comp = row["v_component"]
        region = row["region"]
        for comparison, (target, reference) in comparisons.items():
            t = float(row[f"{target}_R2"])
            r = float(row[f"{reference}_R2"])
            delta = t - r if np.isfinite(t) and np.isfinite(r) else float("nan")
            pct = float(delta / abs(r) * 100.0) if np.isfinite(delta) and abs(r) > 1.0e-12 else float("nan")
            rows.append({
                "v_component": comp,
                "region": region,
                "comparison": comparison,
                "target_window": target,
                "reference_window": reference,
                "R2_target": t,
                "R2_reference": r,
                "R2_delta": delta,
                "R2_delta_percent": pct,
                "direction": _classify_delta(delta, settings.r2_delta_epsilon),
            })
        early = float(row["T3_early_R2"])
        late = float(row["T3_late_R2"])
        full = float(row["T3_full_R2"])
        ref = np.nanmax([early, late])
        delta = full - ref if np.isfinite(full) and np.isfinite(ref) else float("nan")
        pct = float(delta / abs(ref) * 100.0) if np.isfinite(delta) and abs(ref) > 1.0e-12 else float("nan")
        rows.append({
            "v_component": comp,
            "region": region,
            "comparison": "T3_full_minus_max_subwindow",
            "target_window": "T3_full",
            "reference_window": "max(T3_early,T3_late)",
            "R2_target": full,
            "R2_reference": ref,
            "R2_delta": delta,
            "R2_delta_percent": pct,
            "direction": _classify_delta(delta, settings.r2_delta_epsilon),
        })
    return pd.DataFrame(rows)


def build_north_main_south_transition_chain(matrix_df: pd.DataFrame, settings: TransitionChainReportSettings) -> pd.DataFrame:
    rows: List[dict] = []
    key = matrix_df.set_index(["v_component", "region"])
    for comp in settings.v_components:
        for w in settings.window_order:
            north = float(key.loc[(comp, settings.north_region), f"{w}_R2"])
            main = float(key.loc[(comp, settings.main_region), f"{w}_R2"])
            south = float(key.loc[(comp, settings.south_region), f"{w}_R2"])
            vals = {"north": north, "main": main, "south": south}
            finite_vals = {k: v for k, v in vals.items() if np.isfinite(v)}
            dominant = max(finite_vals, key=finite_vals.get) if finite_vals else "unknown"
            rows.append({
                "v_component": comp,
                "window": w,
                "north_R2": north,
                "main_R2": main,
                "south_R2": south,
                "north_minus_main": north - main if np.isfinite(north) and np.isfinite(main) else float("nan"),
                "south_minus_main": south - main if np.isfinite(south) and np.isfinite(main) else float("nan"),
                "north_minus_south": north - south if np.isfinite(north) and np.isfinite(south) else float("nan"),
                "dominant_region": dominant,
            })
    return pd.DataFrame(rows)


def _daily_anomaly_index(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        clim = out.groupby("day")[col].transform("mean")
        out[col + "__anom"] = out[col] - clim
    return out


def _daily_anomaly_field(field: np.ndarray) -> np.ndarray:
    clim = np.nanmean(field, axis=0, keepdims=True)
    return field - clim


def _build_samples(
    index_df: pd.DataFrame,
    field_anom: np.ndarray,
    window: Tuple[int, int],
    lag: int,
    v_col: str,
    year_to_field_i: Dict[int, int],
    day_to_field_i: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    idx_by_year_day = index_df.set_index(["year", "day"])
    xs: List[float] = []
    ys: List[np.ndarray] = []
    start, end = window
    for target_day in range(start, end + 1):
        source_day = target_day - lag
        if source_day < 1 or target_day not in day_to_field_i:
            continue
        td_i = day_to_field_i[target_day]
        for year, fy_i in year_to_field_i.items():
            key = (int(year), int(source_day))
            if key not in idx_by_year_day.index:
                continue
            x = idx_by_year_day.loc[key, v_col + "__anom"]
            if isinstance(x, pd.Series):
                x = x.iloc[0]
            x = float(x)
            if not np.isfinite(x):
                continue
            y = field_anom[fy_i, td_i, :, :]
            if not np.isfinite(y).any():
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"No samples for {v_col} lag={lag} window={window}")
    return np.asarray(xs, dtype=float), np.stack(ys, axis=0)


def compute_positive_lag_max_support_maps(
    index_df: pd.DataFrame,
    precip_field: np.ndarray,
    settings: TransitionChainReportSettings,
    year_to_field_i: Dict[int, int],
    day_to_field_i: Dict[int, int],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Recompute observed positive-lag-max R2 maps in memory for figures only."""
    if corr_r2_beta_map is None:
        raise ImportError("Cannot import corr_r2_beta_map from t3_v_to_p_field_explanation_math. Apply v1_a field-explanation patch first.")
    idx = _daily_anomaly_index(index_df, settings.v_components)
    p_anom = _daily_anomaly_field(precip_field)
    out: Dict[str, Dict[str, np.ndarray]] = {comp: {} for comp in settings.v_components}
    for comp in settings.v_components:
        for w in settings.window_order:
            maps = []
            for lag in range(1, settings.max_lag + 1):
                x, y = _build_samples(idx, p_anom, settings.windows[w], lag, comp, year_to_field_i, day_to_field_i)
                _, r2, _, _ = corr_r2_beta_map(x, y)
                maps.append(r2)
            out[comp][w] = np.nanmax(np.stack(maps, axis=0), axis=0)
    return out


def build_support_delta_maps(support_maps: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for comp, maps in support_maps.items():
        out[comp] = {
            "T3_full_minus_S3": maps["T3_full"] - maps["S3"],
            "T3_late_minus_T3_early": maps["T3_late"] - maps["T3_early"],
            "S4_minus_T3_full": maps["S4"] - maps["T3_full"],
        }
    return out
