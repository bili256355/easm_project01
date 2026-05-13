# -*- coding: utf-8 -*-
"""
T3 V→P physical hypothesis audit.

Purpose
-------
This module does NOT rerun the main V1 lead-lag screening. It performs a
diagnostic audit to evaluate a set of physical hypotheses that may explain why
T3 V→P fixed-index relations contract in V1:

H1: rain-band spatial reorganization
H2: V effect component shifts from strength toward NS-difference/position
H3: T3 internal state mixing / dilution
H4: target P component shift
H5: synchronous multi-family reorganization

The outputs are evidence summaries, not pathway claims.

Implementation notes
--------------------
- Main field/index口径: smoothed field + smoothed index values.
- Diagnostic regions are only for explanation; they do not replace the main
  research indices.
- Figures are optional and can be disabled with --no-figures.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Any
import json
import math
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------
# Settings and constants
# -----------------------------

@dataclass
class PhysicalHypothesisAuditSettings:
    project_root: Path
    output_dir: Path
    make_figures: bool = True
    use_cartopy: bool = True
    max_lag: int = 5
    south_scs_lon_max: float = 130.0

    # Existing outputs
    v1_output_name: str = "lead_lag_screen_v1_smooth5_a"
    v1_stability_output_name: str = "lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b"
    t3_disappearance_output_name: str = "lead_lag_screen_v1_smooth5_a_t3_v_to_p_audit"
    t3_lag0_reduction_output_name: str = "lead_lag_screen_v1_smooth5_a_t3_v_to_p_lag0_reduction_audit"
    index_validity_output_name: str = "window_family_guardrail_v1_b_smoothed_a"

    # Foundation data
    foundation_preprocess_rel: str = "foundation/V1/outputs/baseline_smooth5_a/preprocess"
    foundation_indices_rel: str = "foundation/V1/outputs/baseline_smooth5_a/indices"

    # Figure map domain, per user constraint
    map_lon_min: float = 20.0
    map_lon_max: float = 150.0
    map_lat_min: float = 10.0
    map_lat_max: float = 60.0


WINDOWS: Dict[str, Tuple[int, int]] = {
    "S3": (90, 107),
    "T3_full": (106, 120),
    "T3_early": (106, 113),
    "T3_late": (114, 120),
    "S4": (120, 158),
}

# Used for multi-family stability shift table
STABILITY_WINDOWS = ["S3", "T3", "S4"]

V_INDICES = ["V_strength", "V_pos_centroid_lat", "V_NS_diff"]
P_INDICES = [
    "P_main_band_share",
    "P_south_band_share_18_24",
    "P_main_minus_south",
    "P_spread_lat",
    "P_north_band_share_35_45",
    "P_north_minus_main_35_45",
    "P_total_centroid_lat_10_50",
]

P_GROUPS = {
    "mainband_group": [
        "P_main_band_share",
        "P_main_minus_south",
        "P_north_minus_main_35_45",
    ],
    "spread_centroid_south_group": [
        "P_spread_lat",
        "P_south_band_share_18_24",
        "P_total_centroid_lat_10_50",
    ],
    "north_group": [
        "P_north_band_share_35_45",
    ],
}

# Diagnostic regions: not replacement indices.
def diagnostic_regions(south_scs_lon_max: float) -> Dict[str, Dict[str, float]]:
    return {
        "meiyu_band": {"lat_min": 24.0, "lat_max": 35.0, "lon_min": 100.0, "lon_max": 125.0},
        "northeast_china": {"lat_min": 40.0, "lat_max": 50.0, "lon_min": 110.0, "lon_max": 135.0},
        "south_china_scs": {"lat_min": 10.0, "lat_max": 25.0, "lon_min": 105.0, "lon_max": south_scs_lon_max},
        "main_easm_domain": {"lat_min": 10.0, "lat_max": 50.0, "lon_min": 100.0, "lon_max": 135.0},
    }


# -----------------------------
# I/O helpers
# -----------------------------

def _ensure_dirs(settings: PhysicalHypothesisAuditSettings) -> Dict[str, Path]:
    out = settings.output_dir
    dirs = {
        "root": out,
        "tables": out / "tables",
        "figures": out / "figures",
        "summary": out / "summary",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _v1_outputs(settings: PhysicalHypothesisAuditSettings) -> Dict[str, Path]:
    base = settings.project_root / "lead_lag_screen" / "V1" / "outputs"
    return {
        "v1": base / settings.v1_output_name,
        "stability": base / settings.v1_stability_output_name,
        "t3_audit": base / settings.t3_disappearance_output_name,
        "lag0_audit": base / settings.t3_lag0_reduction_output_name,
    }


def _index_validity_dir(settings: PhysicalHypothesisAuditSettings) -> Path:
    return (
        settings.project_root
        / "index_validity"
        / "V1_b_window_family_guardrail"
        / "outputs"
        / settings.index_validity_output_name
    )


def _find_table(root: Path, candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = root / "tables" / c
        if p.exists():
            return p
        p2 = root / c
        if p2.exists():
            return p2
    return None


def _load_index_values(settings: PhysicalHypothesisAuditSettings) -> Tuple[pd.DataFrame, Path]:
    idx_dir = settings.project_root / Path(settings.foundation_indices_rel)
    candidates = [
        idx_dir / "index_values_smoothed.csv",
        idx_dir / "indices_smoothed.csv",
        idx_dir / "index_values.csv",
        idx_dir / "smoothed_indices.csv",
    ]
    path = _first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            "Cannot find smoothed index CSV. Tried:\n" + "\n".join(map(str, candidates))
        )
    df = pd.read_csv(path)
    df = _normalize_year_day_columns(df)
    return df, path


def _normalize_year_day_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    colmap = {}
    if "year" not in out.columns:
        for c in ["years", "Year", "YEAR"]:
            if c in out.columns:
                colmap[c] = "year"
                break
    if "day" not in out.columns:
        for c in ["day_index", "doy", "DOY", "Day", "dayofseason"]:
            if c in out.columns:
                colmap[c] = "day"
                break
    if colmap:
        out = out.rename(columns=colmap)
    if "year" not in out.columns or "day" not in out.columns:
        raise ValueError("Index CSV must contain year/day columns or recognizable aliases.")
    out["year"] = out["year"].astype(int)
    out["day"] = out["day"].astype(int)
    return out


def _load_smoothed_fields(settings: PhysicalHypothesisAuditSettings) -> Tuple[Dict[str, np.ndarray], Path]:
    pre = settings.project_root / Path(settings.foundation_preprocess_rel)
    candidates = [
        pre / "smoothed_fields.npz",
        pre / "fields_smoothed.npz",
        pre / "smooth5_fields.npz",
    ]
    path = _first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            "Cannot find smoothed_fields npz. Tried:\n" + "\n".join(map(str, candidates))
        )
    npz = np.load(path, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}
    return data, path


def _get_field_key(data: Dict[str, np.ndarray], aliases: List[str]) -> str:
    """Resolve a field key robustly across foundation naming conventions.

    Some preprocess files store fields as short names (``precip``, ``v850``), while
    the smooth5 preprocess generated in this project commonly stores them as
    ``precip_smoothed``, ``v850_smoothed``, ``z500_smoothed`` and
    ``u200_smoothed``. This resolver keeps the scientific logic unchanged and
    only broadens the accepted input naming contract.
    """
    lower = {k.lower(): k for k in data.keys()}

    expanded: List[str] = []
    for a in aliases:
        expanded.extend([a, f"{a}_smoothed", f"{a}_smooth", f"smoothed_{a}", f"smooth_{a}"])

    alias_map = {
        "precip": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "p": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "pr": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "rain": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "precipitation": ["precip_smoothed", "precipitation_smoothed", "rain_smoothed", "pr_smoothed"],
        "v850": ["v850_smoothed"],
        "v": ["v850_smoothed"],
        "v_wind": ["v850_smoothed"],
        "z500": ["z500_smoothed"],
        "h": ["z500_smoothed"],
        "height": ["z500_smoothed"],
        "u200": ["u200_smoothed"],
        "je": ["u200_smoothed"],
        "jw": ["u200_smoothed"],
    }
    for a in aliases:
        expanded.extend(alias_map.get(a.lower(), []))

    seen = set()
    for a in expanded:
        if a in seen:
            continue
        seen.add(a)
        if a in data:
            return a
        if a.lower() in lower:
            return lower[a.lower()]

    raise KeyError(f"Cannot find field key among aliases {aliases}. Available: {list(data.keys())}")


def _get_lat_lon_years(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    lat_key = _get_field_key(data, ["lat", "lats", "latitude"])
    lon_key = _get_field_key(data, ["lon", "lons", "longitude"])
    years = None
    for ykey in ["years", "year", "yrs"]:
        if ykey in data:
            years = np.asarray(data[ykey]).astype(int)
            break
    return np.asarray(data[lat_key]), np.asarray(data[lon_key]), years


def _field_year_day_index(fields: Dict[str, np.ndarray], years_from_npz: Optional[np.ndarray]) -> Dict[int, int]:
    if years_from_npz is None:
        return {}
    return {int(y): i for i, y in enumerate(years_from_npz.tolist())}


# -----------------------------
# Math helpers
# -----------------------------

def _weighted_mean_2d(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, region: Dict[str, float]) -> float:
    lat_mask = (lat >= region["lat_min"]) & (lat <= region["lat_max"])
    lon_mask = (lon >= region["lon_min"]) & (lon <= region["lon_max"])
    if not lat_mask.any() or not lon_mask.any():
        return float("nan")
    sub = field[np.ix_(lat_mask, lon_mask)]
    weights = np.cos(np.deg2rad(lat[lat_mask]))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    w2 = weights[:, None] * np.ones((1, lon_mask.sum()))
    valid = np.isfinite(sub)
    if not valid.any():
        return float("nan")
    denom = np.nansum(w2 * valid)
    if denom <= 0:
        return float("nan")
    return float(np.nansum(sub * w2) / denom)


def _weighted_sum_positive(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, region: Dict[str, float]) -> float:
    lat_mask = (lat >= region["lat_min"]) & (lat <= region["lat_max"])
    lon_mask = (lon >= region["lon_min"]) & (lon <= region["lon_max"])
    if not lat_mask.any() or not lon_mask.any():
        return float("nan")
    sub = field[np.ix_(lat_mask, lon_mask)]
    sub = np.where(np.isfinite(sub), sub, np.nan)
    # For raw precipitation, positive-sum share is meaningful; for possible anomalies,
    # this gracefully becomes a positive-anomaly contribution.
    sub_pos = np.where(sub > 0, sub, 0.0)
    weights = np.cos(np.deg2rad(lat[lat_mask]))[:, None]
    return float(np.nansum(sub_pos * weights))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _lag_arrays_from_index(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    days: List[int],
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for year, g in df.groupby("year", sort=False):
        gd = g.set_index("day")
        for d in days:
            sd = d - lag
            if sd in gd.index and d in gd.index:
                try:
                    x = float(gd.loc[sd, source_col])
                    y = float(gd.loc[d, target_col])
                except Exception:
                    continue
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _lag_profile(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    days: List[int],
    max_lag: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        x, y = _lag_arrays_from_index(df, source_col, target_col, days, lag)
        r = _corr(x, y)
        rows.append({"lag": lag, "r": r, "abs_r": abs(r) if np.isfinite(r) else np.nan, "n": int(len(x))})
    prof = pd.DataFrame(rows)
    pos = prof[prof["lag"] > 0]
    neg = prof[prof["lag"] < 0]
    lag0 = prof[prof["lag"] == 0]
    pos_row = pos.loc[pos["abs_r"].idxmax()] if pos["abs_r"].notna().any() else None
    neg_row = neg.loc[neg["abs_r"].idxmax()] if neg["abs_r"].notna().any() else None
    zero_abs = float(lag0["abs_r"].iloc[0]) if len(lag0) else np.nan

    pos_abs = float(pos_row["abs_r"]) if pos_row is not None else np.nan
    neg_abs = float(neg_row["abs_r"]) if neg_row is not None else np.nan
    profile_type = _profile_type(pos_abs, zero_abs, neg_abs)

    summary = {
        "positive_peak_lag": int(pos_row["lag"]) if pos_row is not None else np.nan,
        "positive_peak_r": float(pos_row["r"]) if pos_row is not None else np.nan,
        "positive_peak_abs_r": pos_abs,
        "lag0_r": float(lag0["r"].iloc[0]) if len(lag0) else np.nan,
        "lag0_abs_r": zero_abs,
        "negative_peak_lag": int(neg_row["lag"]) if neg_row is not None else np.nan,
        "negative_peak_r": float(neg_row["r"]) if neg_row is not None else np.nan,
        "negative_peak_abs_r": neg_abs,
        "profile_type": profile_type,
    }
    return prof, summary


def _profile_type(pos_abs: float, zero_abs: float, neg_abs: float) -> str:
    vals = np.array([pos_abs, zero_abs, neg_abs], dtype=float)
    if not np.isfinite(vals).any():
        return "insufficient_data"
    mx = np.nanmax(vals)
    if mx < 0.20:
        return "weak_all_lags"
    tol = 0.05
    close_count = int(np.nansum(np.abs(vals - mx) <= tol))
    if close_count >= 2:
        if np.isfinite(zero_abs) and abs(zero_abs - mx) <= tol:
            return "flat_lag0_positive_close"
        return "multi_peak_close"
    idx = int(np.nanargmax(vals))
    return ["positive_peak_clear", "lag0_peak", "reverse_peak"][idx]


def _split_days(window: str) -> List[int]:
    start, end = WINDOWS[window]
    return list(range(start, end + 1))


def _subwindow_for_sample(day: int) -> Optional[str]:
    for w, (s, e) in WINDOWS.items():
        if s <= int(day) <= e:
            return w
    return None


# -----------------------------
# Field extraction helpers
# -----------------------------

def _field_sample(
    field: np.ndarray,
    year: int,
    day: int,
    year_index: Dict[int, int],
) -> Optional[np.ndarray]:
    # Field expected shape: year, day, lat, lon. If years unknown, year index
    # may be empty and we assume year is already ordinal impossible; return None.
    if not year_index:
        return None
    yi = year_index.get(int(year))
    if yi is None:
        return None
    if yi < 0 or yi >= field.shape[0] or day < 0 or day >= field.shape[1]:
        return None
    return np.asarray(field[yi, int(day), :, :], dtype=float)


def _field_mean_for_rows(
    field: np.ndarray,
    rows: pd.DataFrame,
    year_index: Dict[int, int],
) -> Optional[np.ndarray]:
    arrs = []
    for _, r in rows.iterrows():
        f = _field_sample(field, int(r["year"]), int(r["day"]), year_index)
        if f is not None:
            arrs.append(f)
    if not arrs:
        return None
    return np.nanmean(np.stack(arrs, axis=0), axis=0)


def _rows_for_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    days = set(_split_days(window))
    return df[df["day"].isin(days)].copy()


def _high_low_rows(df: pd.DataFrame, index_col: str, window: str, q: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = _rows_for_window(df, window)
    if index_col not in sub.columns or sub.empty:
        return sub.iloc[0:0], sub.iloc[0:0]
    vals = sub[index_col].astype(float)
    lo = vals.quantile(q)
    hi = vals.quantile(1 - q)
    return sub[vals >= hi].copy(), sub[vals <= lo].copy()


# -----------------------------
# Module A: subwindow dilution
# -----------------------------

def _compute_subwindow_v_to_p_profiles(index_df: pd.DataFrame, settings: PhysicalHypothesisAuditSettings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows = []
    summary_rows = []
    for win in ["S3", "T3_full", "T3_early", "T3_late", "S4"]:
        days = _split_days(win)
        for src in V_INDICES:
            if src not in index_df.columns:
                continue
            for tgt in P_INDICES:
                if tgt not in index_df.columns:
                    continue
                prof, summ = _lag_profile(index_df, src, tgt, days, settings.max_lag)
                for _, rr in prof.iterrows():
                    long_rows.append({
                        "window_subwindow": win,
                        "source_index": src,
                        "target_index": tgt,
                        "lag": int(rr["lag"]),
                        "r": rr["r"],
                        "abs_r": rr["abs_r"],
                        "n": int(rr["n"]),
                    })
                summary_rows.append({
                    "window_subwindow": win,
                    "source_index": src,
                    "target_index": tgt,
                    **summ,
                })
    long_df = pd.DataFrame(long_rows)
    summ_df = pd.DataFrame(summary_rows)

    # Early/late dilution classification for 21 pairs
    class_rows = []
    early = summ_df[summ_df["window_subwindow"] == "T3_early"]
    late = summ_df[summ_df["window_subwindow"] == "T3_late"]
    merged = early.merge(
        late,
        on=["source_index", "target_index"],
        suffixes=("_early", "_late"),
        how="outer",
    )
    for _, r in merged.iterrows():
        et = r.get("profile_type_early", "")
        lt = r.get("profile_type_late", "")
        epos = r.get("positive_peak_abs_r_early", np.nan)
        lpos = r.get("positive_peak_abs_r_late", np.nan)
        e0 = r.get("lag0_abs_r_early", np.nan)
        l0 = r.get("lag0_abs_r_late", np.nan)
        eneg = r.get("negative_peak_abs_r_early", np.nan)
        lneg = r.get("negative_peak_abs_r_late", np.nan)

        classification = "both_similar_or_unclear"
        if et != lt:
            classification = "profile_type_changed"
        if np.isfinite(epos) and np.isfinite(lpos):
            if epos >= 0.30 and lpos < 0.20:
                classification = "early_strong_late_weak"
            elif epos < 0.20 and lpos >= 0.30:
                classification = "early_weak_late_strong"
        if np.isfinite(epos) and np.isfinite(lneg) and epos >= 0.30 and lneg >= 0.30:
            classification = "early_positive_late_reverse_competition"
        if np.isfinite(eneg) and np.isfinite(lpos) and eneg >= 0.30 and lpos >= 0.30:
            classification = "early_reverse_late_positive_shift"
        if (np.isfinite(epos) and np.isfinite(lpos) and epos < 0.20 and lpos < 0.20 and
                np.isfinite(e0) and np.isfinite(l0) and e0 < 0.20 and l0 < 0.20):
            classification = "both_weak"

        class_rows.append({
            "source_index": r.get("source_index"),
            "target_index": r.get("target_index"),
            "profile_type_early": et,
            "profile_type_late": lt,
            "positive_peak_abs_r_early": epos,
            "positive_peak_abs_r_late": lpos,
            "lag0_abs_r_early": e0,
            "lag0_abs_r_late": l0,
            "negative_peak_abs_r_early": eneg,
            "negative_peak_abs_r_late": lneg,
            "d_positive_late_minus_early": (lpos - epos) if np.isfinite(lpos) and np.isfinite(epos) else np.nan,
            "d_lag0_late_minus_early": (l0 - e0) if np.isfinite(l0) and np.isfinite(e0) else np.nan,
            "d_negative_late_minus_early": (lneg - eneg) if np.isfinite(lneg) and np.isfinite(eneg) else np.nan,
            "dilution_class": classification,
        })
    class_df = pd.DataFrame(class_rows)
    return long_df, summ_df, class_df


def _summarize_dilution(class_df: pd.DataFrame) -> pd.DataFrame:
    if class_df.empty:
        return pd.DataFrame()
    n = len(class_df)
    changed = (class_df["profile_type_early"] != class_df["profile_type_late"]).sum()
    strong_like = class_df["dilution_class"].isin([
        "early_strong_late_weak",
        "early_weak_late_strong",
        "early_positive_late_reverse_competition",
        "early_reverse_late_positive_shift",
        "profile_type_changed",
    ]).sum()
    frac_changed = changed / n if n else np.nan
    support = "weak"
    if frac_changed >= 0.50 and strong_like >= max(6, int(0.33 * n)):
        support = "strong"
    elif frac_changed >= 0.33:
        support = "moderate"
    return pd.DataFrame([{
        "hypothesis": "H3_internal_state_mixing_dilution",
        "n_pairs": n,
        "n_profile_type_changed": int(changed),
        "fraction_profile_type_changed": frac_changed,
        "n_dilution_like_pairs": int(strong_like),
        "evidence_strength": support,
        "interpretation": "Frequent early/late profile changes support T3 internal state mixing if accompanied by spatial/regional shifts.",
    }])


# -----------------------------
# Module B: regional precipitation contribution
# -----------------------------

def _regional_precip_contribution(
    index_df: pd.DataFrame,
    fields: Dict[str, np.ndarray],
    settings: PhysicalHypothesisAuditSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pkey = _get_field_key(fields, ["precip", "P", "pr", "rain", "precipitation"])
    lat, lon, years = _get_lat_lon_years(fields)
    year_index = _field_year_day_index(fields, years)
    pfield = np.asarray(fields[pkey], dtype=float)
    regions = diagnostic_regions(settings.south_scs_lon_max)

    rows = []
    for win in ["S3", "T3_early", "T3_late", "S4"]:
        sub = _rows_for_window(index_df, win)
        fields_list = []
        for _, r in sub.iterrows():
            f = _field_sample(pfield, int(r["year"]), int(r["day"]), year_index)
            if f is not None:
                fields_list.append(f)
        if not fields_list:
            continue
        arr = np.stack(fields_list, axis=0)
        mean_map = np.nanmean(arr, axis=0)
        var_map = np.nanvar(arr, axis=0)
        domain_sum = _weighted_sum_positive(mean_map, lat, lon, regions["main_easm_domain"])
        for name, reg in regions.items():
            rsum = _weighted_sum_positive(mean_map, lat, lon, reg)
            rows.append({
                "window_subwindow": win,
                "region": name,
                "regional_p_mean": _weighted_mean_2d(mean_map, lat, lon, reg),
                "regional_p_variance_mean": _weighted_mean_2d(var_map, lat, lon, reg),
                "regional_positive_weighted_sum": rsum,
                "regional_positive_share_of_main_domain": (rsum / domain_sum) if domain_sum and np.isfinite(domain_sum) and domain_sum != 0 else np.nan,
                "n_samples": int(arr.shape[0]),
            })
    contrib = pd.DataFrame(rows)

    # Early-late shift table
    shift_rows = []
    early = contrib[contrib["window_subwindow"] == "T3_early"]
    late = contrib[contrib["window_subwindow"] == "T3_late"]
    for region in regions:
        er = early[early["region"] == region]
        lr = late[late["region"] == region]
        if er.empty or lr.empty:
            continue
        erow = er.iloc[0]
        lrow = lr.iloc[0]
        shift_rows.append({
            "region": region,
            "share_early": erow["regional_positive_share_of_main_domain"],
            "share_late": lrow["regional_positive_share_of_main_domain"],
            "share_late_minus_early": lrow["regional_positive_share_of_main_domain"] - erow["regional_positive_share_of_main_domain"],
            "mean_early": erow["regional_p_mean"],
            "mean_late": lrow["regional_p_mean"],
            "mean_late_minus_early": lrow["regional_p_mean"] - erow["regional_p_mean"],
            "variance_early": erow["regional_p_variance_mean"],
            "variance_late": lrow["regional_p_variance_mean"],
            "variance_late_minus_early": lrow["regional_p_variance_mean"] - erow["regional_p_variance_mean"],
        })
    shifts = pd.DataFrame(shift_rows)
    return contrib, shifts


# -----------------------------
# Module C: V index regional P response
# -----------------------------

def _v_index_regional_response(
    index_df: pd.DataFrame,
    fields: Dict[str, np.ndarray],
    settings: PhysicalHypothesisAuditSettings,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    pkey = _get_field_key(fields, ["precip", "P", "pr", "rain", "precipitation"])
    vkey = _get_field_key(fields, ["v850", "V", "v", "v_wind"])
    lat, lon, years = _get_lat_lon_years(fields)
    year_index = _field_year_day_index(fields, years)
    pfield = np.asarray(fields[pkey], dtype=float)
    vfield = np.asarray(fields[vkey], dtype=float)
    regions = diagnostic_regions(settings.south_scs_lon_max)

    rows = []
    map_cache: Dict[str, np.ndarray] = {}
    for win in ["T3_full", "T3_early", "T3_late"]:
        for v_index in V_INDICES:
            if v_index not in index_df.columns:
                continue
            high, low = _high_low_rows(index_df, v_index, win)
            if high.empty or low.empty:
                continue
            p_high = _field_mean_for_rows(pfield, high, year_index)
            p_low = _field_mean_for_rows(pfield, low, year_index)
            v_high = _field_mean_for_rows(vfield, high, year_index)
            v_low = _field_mean_for_rows(vfield, low, year_index)
            if p_high is None or p_low is None:
                continue
            p_diff = p_high - p_low
            v_diff = v_high - v_low if v_high is not None and v_low is not None else None

            map_cache[f"{win}_{v_index}_P_diff"] = p_diff
            if v_diff is not None:
                map_cache[f"{win}_{v_index}_V_diff"] = v_diff

            for region_name, reg in regions.items():
                rows.append({
                    "window_subwindow": win,
                    "v_index": v_index,
                    "region": region_name,
                    "p_response_high_minus_low": _weighted_mean_2d(p_diff, lat, lon, reg),
                    "abs_p_response_high_minus_low": abs(_weighted_mean_2d(p_diff, lat, lon, reg)),
                    "v_response_high_minus_low": _weighted_mean_2d(v_diff, lat, lon, reg) if v_diff is not None else np.nan,
                    "p_diff_std_map": float(np.nanstd(p_diff)),
                    "v_diff_std_map": float(np.nanstd(v_diff)) if v_diff is not None else np.nan,
                    "n_high": int(len(high)),
                    "n_low": int(len(low)),
                })
    response = pd.DataFrame(rows)

    # subwindow shift for each V index and region
    shift_rows = []
    early = response[response["window_subwindow"] == "T3_early"]
    late = response[response["window_subwindow"] == "T3_late"]
    for v_index in V_INDICES:
        for region_name in regions:
            er = early[(early["v_index"] == v_index) & (early["region"] == region_name)]
            lr = late[(late["v_index"] == v_index) & (late["region"] == region_name)]
            if er.empty or lr.empty:
                continue
            e = er.iloc[0]
            l = lr.iloc[0]
            shift_rows.append({
                "v_index": v_index,
                "region": region_name,
                "p_response_early": e["p_response_high_minus_low"],
                "p_response_late": l["p_response_high_minus_low"],
                "abs_p_response_early": e["abs_p_response_high_minus_low"],
                "abs_p_response_late": l["abs_p_response_high_minus_low"],
                "abs_response_late_minus_early": l["abs_p_response_high_minus_low"] - e["abs_p_response_high_minus_low"],
            })
    shift = pd.DataFrame(shift_rows)
    return response, shift, map_cache


# -----------------------------
# Module D: P target component shift
# -----------------------------

def _p_target_group_shift(subwindow_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if subwindow_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for win in ["T3_early", "T3_late", "T3_full"]:
        sub = subwindow_summary[subwindow_summary["window_subwindow"] == win]
        for group, targets in P_GROUPS.items():
            g = sub[sub["target_index"].isin(targets)]
            if g.empty:
                continue
            rows.append({
                "window_subwindow": win,
                "p_target_group": group,
                "n_pairs": int(len(g)),
                "mean_positive_peak_abs_r": float(g["positive_peak_abs_r"].mean()),
                "mean_lag0_abs_r": float(g["lag0_abs_r"].mean()),
                "mean_negative_peak_abs_r": float(g["negative_peak_abs_r"].mean()),
                "n_positive_peak_clear": int((g["profile_type"] == "positive_peak_clear").sum()),
                "n_lag0_peak": int((g["profile_type"] == "lag0_peak").sum()),
                "n_weak_all_lags": int((g["profile_type"] == "weak_all_lags").sum()),
                "n_flat_lag0_positive_close": int((g["profile_type"] == "flat_lag0_positive_close").sum()),
            })
    group_summary = pd.DataFrame(rows)

    evidence_rows = []
    if not group_summary.empty:
        for group in P_GROUPS:
            er = group_summary[(group_summary["window_subwindow"] == "T3_early") & (group_summary["p_target_group"] == group)]
            lr = group_summary[(group_summary["window_subwindow"] == "T3_late") & (group_summary["p_target_group"] == group)]
            fr = group_summary[(group_summary["window_subwindow"] == "T3_full") & (group_summary["p_target_group"] == group)]
            if er.empty or lr.empty:
                continue
            e = er.iloc[0]
            l = lr.iloc[0]
            evidence_rows.append({
                "p_target_group": group,
                "early_mean_positive": e["mean_positive_peak_abs_r"],
                "late_mean_positive": l["mean_positive_peak_abs_r"],
                "late_minus_early_positive": l["mean_positive_peak_abs_r"] - e["mean_positive_peak_abs_r"],
                "full_mean_positive": fr.iloc[0]["mean_positive_peak_abs_r"] if not fr.empty else np.nan,
                "early_n_positive_peak_clear": e["n_positive_peak_clear"],
                "late_n_positive_peak_clear": l["n_positive_peak_clear"],
                "early_n_weak_all_lags": e["n_weak_all_lags"],
                "late_n_weak_all_lags": l["n_weak_all_lags"],
            })
    evidence = pd.DataFrame(evidence_rows)
    return group_summary, evidence


# -----------------------------
# Module E: multi-family shift
# -----------------------------

def _family_direction(row: pd.Series) -> str:
    def fam(v: str) -> str:
        if not isinstance(v, str):
            return "UNK"
        return v.split("_", 1)[0]
    if "family_direction" in row and isinstance(row["family_direction"], str):
        return row["family_direction"]
    sv = row.get("source_variable", row.get("source_index", ""))
    tv = row.get("target_variable", row.get("target_index", ""))
    return f"{fam(sv)}→{fam(tv)}"


def _load_stability_table(settings: PhysicalHypothesisAuditSettings) -> pd.DataFrame:
    roots = _v1_outputs(settings)
    stab_dir = roots["stability"]
    path = _find_table(stab_dir, [
        "lead_lag_pair_summary_stability_judged.csv",
        "v1_core_candidate_pool_stability_judged.csv",
    ])
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "family_direction" not in df.columns:
        df["family_direction"] = df.apply(_family_direction, axis=1)
    return df


def _multi_family_stability_shift(settings: PhysicalHypothesisAuditSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _load_stability_table(settings)
    if df.empty or "window" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for win in STABILITY_WINDOWS:
        sub = df[df["window"].astype(str).str.upper().eq(win)]
        for fd, g in sub.groupby("family_direction"):
            n = len(g)
            if n == 0:
                continue
            stable = int((g.get("v1_stability_judgement", pd.Series(index=g.index)).astype(str) == "stable_lag_dominant").sum())
            tau0 = int((g.get("v1_stability_judgement", pd.Series(index=g.index)).astype(str) == "significant_lagged_but_tau0_coupled").sum())
            sensitive = int((g.get("v1_stability_judgement", pd.Series(index=g.index)).astype(str) == "audit_sensitive").sum())
            rows.append({
                "window": win,
                "family_direction": fd,
                "n_rows": int(n),
                "n_stable_lag_dominant": stable,
                "n_tau0_coupled": tau0,
                "n_audit_sensitive": sensitive,
                "stable_fraction": stable / n if n else np.nan,
                "tau0_coupled_fraction": tau0 / n if n else np.nan,
            })
    roll = pd.DataFrame(rows)

    evidence_rows = []
    if not roll.empty:
        for fd, g in roll.groupby("family_direction"):
            s3 = g[g["window"] == "S3"]
            t3 = g[g["window"] == "T3"]
            s4 = g[g["window"] == "S4"]
            if t3.empty:
                continue
            t = t3.iloc[0]
            baseline_stable = np.nanmean([s3.iloc[0]["stable_fraction"] if not s3.empty else np.nan,
                                          s4.iloc[0]["stable_fraction"] if not s4.empty else np.nan])
            baseline_tau0 = np.nanmean([s3.iloc[0]["tau0_coupled_fraction"] if not s3.empty else np.nan,
                                        s4.iloc[0]["tau0_coupled_fraction"] if not s4.empty else np.nan])
            evidence_rows.append({
                "family_direction": fd,
                "t3_stable_fraction": t["stable_fraction"],
                "neighbor_mean_stable_fraction": baseline_stable,
                "t3_minus_neighbor_stable_fraction": t["stable_fraction"] - baseline_stable if np.isfinite(baseline_stable) else np.nan,
                "t3_tau0_coupled_fraction": t["tau0_coupled_fraction"],
                "neighbor_mean_tau0_coupled_fraction": baseline_tau0,
                "t3_minus_neighbor_tau0_fraction": t["tau0_coupled_fraction"] - baseline_tau0 if np.isfinite(baseline_tau0) else np.nan,
            })
    evidence = pd.DataFrame(evidence_rows)
    return roll, evidence


# -----------------------------
# Index validity context
# -----------------------------

def _index_validity_context(settings: PhysicalHypothesisAuditSettings) -> pd.DataFrame:
    root = _index_validity_dir(settings)
    rep_path = _find_table(root, ["index_window_representativeness.csv", "t3_index_representativeness_audit.csv"])
    joint_path = _find_table(root, ["window_family_joint_field_coverage.csv", "t3_window_family_joint_field_coverage.csv"])
    rep = pd.read_csv(rep_path) if rep_path is not None else pd.DataFrame()
    joint = pd.read_csv(joint_path) if joint_path is not None else pd.DataFrame()

    rows = []
    if not rep.empty:
        t3 = rep[rep["window"].astype(str).str.upper().eq("T3")] if "window" in rep.columns else rep
        for idx in V_INDICES + P_INDICES:
            r = t3[t3.get("index_name", pd.Series(dtype=str)).astype(str).eq(idx)]
            if r.empty:
                continue
            rr = r.iloc[0]
            rows.append({
                "index_name": idx,
                "family": idx.split("_", 1)[0],
                "representativeness_tier": rr.get("representativeness_tier", ""),
                "risk_flag": rr.get("risk_flag", ""),
                "field_R2": rr.get("field_r2_weighted", rr.get("field_R2", np.nan)),
                "eof_alignment": rr.get("eof_alignment_max_abs_corr", np.nan),
            })
    ctx = pd.DataFrame(rows)

    if not joint.empty:
        jt3 = joint[joint["window"].astype(str).str.upper().eq("T3")] if "window" in joint.columns else joint
        joint_rows = []
        for fam in ["P", "V"]:
            f = jt3[jt3.get("family", pd.Series(dtype=str)).astype(str).eq(fam)]
            if not f.empty:
                rr = f.iloc[0]
                joint_rows.append({
                    "family": fam,
                    "coverage_tier": rr.get("coverage_tier", ""),
                    "joint_field_R2_year_cv": rr.get("joint_field_R2_year_cv", np.nan),
                    "joint_eof_coverage_top5_year_cv": rr.get("joint_eof_coverage_top5_year_cv", np.nan),
                    "collapse_risk_update": rr.get("collapse_risk_update", ""),
                })
        joint_df = pd.DataFrame(joint_rows)
        if not ctx.empty and not joint_df.empty:
            ctx = ctx.merge(joint_df, on="family", how="left")
    return ctx


# -----------------------------
# Hypothesis evidence summary
# -----------------------------

def _strength_from_share_shift(regional_shift: pd.DataFrame) -> Tuple[str, str]:
    if regional_shift.empty:
        return "weak", "No regional precipitation shift table available."
    # Focus on meiyu down and NE/South up. Use small absolute threshold.
    def get(region: str, col: str) -> float:
        r = regional_shift[regional_shift["region"] == region]
        return float(r[col].iloc[0]) if not r.empty and col in r.columns else np.nan
    meiyu = get("meiyu_band", "share_late_minus_early")
    ne = get("northeast_china", "share_late_minus_early")
    south = get("south_china_scs", "share_late_minus_early")
    support = 0
    notes = []
    if np.isfinite(meiyu) and meiyu < -0.02:
        support += 1
        notes.append(f"Meiyu share decreased ({meiyu:.3f}).")
    if np.isfinite(ne) and ne > 0.02:
        support += 1
        notes.append(f"Northeast share increased ({ne:.3f}).")
    if np.isfinite(south) and south > 0.02:
        support += 1
        notes.append(f"South/SCS share increased ({south:.3f}).")
    if support >= 2:
        return "strong", " ".join(notes)
    if support == 1:
        return "moderate", " ".join(notes)
    return "weak", "No strong Meiyu-to-NE/South share shift detected by default thresholds."


def _strength_h2(v_response_shift: pd.DataFrame) -> Tuple[str, str]:
    if v_response_shift.empty:
        return "weak", "No V-index regional response shift table available."
    # Aggregate absolute response over P regions for early/late.
    agg = v_response_shift.groupby("v_index")[["abs_p_response_early", "abs_p_response_late"]].mean().reset_index()
    row_strength = agg[agg["v_index"] == "V_strength"]
    row_ns = agg[agg["v_index"] == "V_NS_diff"]
    if row_strength.empty or row_ns.empty:
        return "weak", "Required V_strength or V_NS_diff rows missing."
    s_late = float(row_strength["abs_p_response_late"].iloc[0])
    s_early = float(row_strength["abs_p_response_early"].iloc[0])
    ns_late = float(row_ns["abs_p_response_late"].iloc[0])
    ns_early = float(row_ns["abs_p_response_early"].iloc[0])
    notes = f"V_strength abs response early/late={s_early:.3g}/{s_late:.3g}; V_NS_diff={ns_early:.3g}/{ns_late:.3g}."
    if s_late < s_early and ns_late >= 0.8 * ns_early and ns_late > s_late:
        return "strong", notes + " V_strength weakens while V_NS_diff is retained/stronger."
    if ns_late > s_late:
        return "moderate", notes + " V_NS_diff is stronger than V_strength in late T3."
    return "weak", notes + " No clear component transfer from strength to NS-difference."


def _strength_h4(p_group_evidence: pd.DataFrame) -> Tuple[str, str]:
    if p_group_evidence.empty:
        return "weak", "No P target group evidence table available."
    main = p_group_evidence[p_group_evidence["p_target_group"] == "mainband_group"]
    scs = p_group_evidence[p_group_evidence["p_target_group"] == "spread_centroid_south_group"]
    if main.empty or scs.empty:
        return "weak", "Required P target groups missing."
    m = main.iloc[0]
    s = scs.iloc[0]
    m_late = float(m["late_mean_positive"])
    s_late = float(s["late_mean_positive"])
    m_change = float(m["late_minus_early_positive"])
    s_change = float(s["late_minus_early_positive"])
    notes = (
        f"Mainband late positive={m_late:.3f}, change={m_change:.3f}; "
        f"spread/centroid/south late positive={s_late:.3f}, change={s_change:.3f}."
    )
    if s_late > m_late and m_change < 0:
        return "strong", notes + " Target shifts away from mainband toward spread/centroid/south."
    if s_late > m_late:
        return "moderate", notes + " Spread/centroid/south group exceeds mainband in late T3."
    return "weak", notes + " No clear target component shift by default metric."


def _strength_h5(multi_evidence: pd.DataFrame) -> Tuple[str, str]:
    if multi_evidence.empty:
        return "weak", "No multi-family stability shift table available."
    # Count families where T3 has lower stable fraction or higher tau0 fraction than neighbors.
    lower_stable = (multi_evidence["t3_minus_neighbor_stable_fraction"] < -0.10).sum()
    higher_tau0 = (multi_evidence["t3_minus_neighbor_tau0_fraction"] > 0.10).sum()
    n = len(multi_evidence)
    notes = f"{lower_stable}/{n} family-directions lower stable fraction at T3; {higher_tau0}/{n} higher tau0-coupled fraction."
    if lower_stable >= 3 and higher_tau0 >= 2:
        return "strong", notes
    if lower_stable >= 2 or higher_tau0 >= 2:
        return "moderate", notes
    return "weak", notes


def _hypothesis_summary(
    dilution_summary: pd.DataFrame,
    regional_shift: pd.DataFrame,
    v_response_shift: pd.DataFrame,
    p_group_evidence: pd.DataFrame,
    multi_evidence: pd.DataFrame,
    index_context: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # H1
    h1_strength, h1_notes = _strength_from_share_shift(regional_shift)
    rows.append({
        "hypothesis_id": "H1",
        "hypothesis_name": "rain_band_spatial_reorganization",
        "evidence_strength": h1_strength,
        "supporting_evidence": h1_notes,
        "contradicting_evidence": "",
        "missing_evidence": "Manual review of precip maps is still recommended.",
        "interpretation_status": "partially_supported" if h1_strength in ["strong", "moderate"] else "needs_manual_map_review",
    })

    # H2
    h2_strength, h2_notes = _strength_h2(v_response_shift)
    rows.append({
        "hypothesis_id": "H2",
        "hypothesis_name": "V_component_shift_strength_to_NSdiff_or_position",
        "evidence_strength": h2_strength,
        "supporting_evidence": h2_notes,
        "contradicting_evidence": "",
        "missing_evidence": "Needs physical interpretation of V composite patterns.",
        "interpretation_status": "partially_supported" if h2_strength in ["strong", "moderate"] else "weak_support",
    })

    # H3
    if not dilution_summary.empty:
        ds = dilution_summary.iloc[0]
        h3_strength = str(ds.get("evidence_strength", "weak"))
        h3_notes = (
            f"{int(ds.get('n_profile_type_changed', 0))}/{int(ds.get('n_pairs', 0))} "
            f"V→P pairs changed early/late profile type; "
            f"{int(ds.get('n_dilution_like_pairs', 0))} dilution-like pairs."
        )
    else:
        h3_strength = "weak"
        h3_notes = "No dilution summary available."
    rows.append({
        "hypothesis_id": "H3",
        "hypothesis_name": "T3_internal_state_mixing_dilution",
        "evidence_strength": h3_strength,
        "supporting_evidence": h3_notes,
        "contradicting_evidence": "",
        "missing_evidence": "Needs regional/spatial confirmation that early and late T3 correspond to different physical states.",
        "interpretation_status": "partially_supported" if h3_strength in ["strong", "moderate"] else "weak_support",
    })

    # H4
    h4_strength, h4_notes = _strength_h4(p_group_evidence)
    idx_note = ""
    if not index_context.empty:
        n_strong = int((index_context.get("representativeness_tier", pd.Series(dtype=str)).astype(str) == "strong").sum())
        idx_note = f" Index validity context: {n_strong}/{len(index_context)} listed P/V indices are strong where available."
    rows.append({
        "hypothesis_id": "H4",
        "hypothesis_name": "P_target_component_shift",
        "evidence_strength": h4_strength,
        "supporting_evidence": h4_notes + idx_note,
        "contradicting_evidence": "",
        "missing_evidence": "Needs manual map review for P target component composite structure.",
        "interpretation_status": "partially_supported" if h4_strength in ["strong", "moderate"] else "weak_support",
    })

    # H5
    h5_strength, h5_notes = _strength_h5(multi_evidence)
    rows.append({
        "hypothesis_id": "H5",
        "hypothesis_name": "synchronous_multi_family_reorganization",
        "evidence_strength": h5_strength,
        "supporting_evidence": h5_notes,
        "contradicting_evidence": "",
        "missing_evidence": "Needs spatial/object-field confirmation; this table only uses V1 stability counts.",
        "interpretation_status": "partially_supported" if h5_strength in ["strong", "moderate"] else "weak_support",
    })

    return pd.DataFrame(rows)


# -----------------------------
# Figures
# -----------------------------

def _maybe_cartopy(use_cartopy: bool):
    if not use_cartopy:
        return None
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        return ccrs, cfeature
    except Exception:
        return None


def _plot_map_grid(
    maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    out_path: Path,
    settings: PhysicalHypothesisAuditSettings,
    title: str,
) -> None:
    if plt is None or not maps:
        return
    cto = _maybe_cartopy(settings.use_cartopy)
    n = len(maps)
    ncols = min(2, n)
    nrows = int(math.ceil(n / ncols))

    if cto:
        ccrs, cfeature = cto
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
            subplot_kw={"projection": ccrs.PlateCarree()},
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    lon2, lat2 = np.meshgrid(lon, lat)
    vals = np.concatenate([np.ravel(v[np.isfinite(v)]) for v in maps.values() if np.isfinite(v).any()])
    if vals.size:
        lim = np.nanpercentile(np.abs(vals), 98)
        if not np.isfinite(lim) or lim == 0:
            lim = np.nanmax(np.abs(vals)) if vals.size else 1.0
    else:
        lim = 1.0

    for ax, (name, arr) in zip(axes.ravel(), maps.items()):
        if cto:
            ax.set_extent([settings.map_lon_min, settings.map_lon_max, settings.map_lat_min, settings.map_lat_max], crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            im = ax.pcolormesh(lon2, lat2, arr, transform=ccrs.PlateCarree(), shading="auto", vmin=-lim, vmax=lim, cmap="RdBu_r")
        else:
            ax.set_xlim(settings.map_lon_min, settings.map_lon_max)
            ax.set_ylim(settings.map_lat_min, settings.map_lat_max)
            im = ax.pcolormesh(lon2, lat2, arr, shading="auto", vmin=-lim, vmax=lim, cmap="RdBu_r")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.75)

    for ax in axes.ravel()[len(maps):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_lag_heatmap(long_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if plt is None or long_df.empty:
        return
    df = long_df[long_df["window_subwindow"].isin(["T3_early", "T3_late"])]
    if df.empty:
        return
    for win in ["T3_early", "T3_late"]:
        sub = df[df["window_subwindow"] == win].copy()
        sub["pair"] = sub["source_index"] + "→" + sub["target_index"]
        piv = sub.pivot_table(index="pair", columns="lag", values="r", aggfunc="mean")
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(piv))))
        lim = np.nanpercentile(np.abs(piv.values), 98)
        if not np.isfinite(lim) or lim == 0:
            lim = 1
        im = ax.imshow(piv.values, aspect="auto", vmin=-lim, vmax=lim, cmap="RdBu_r")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([str(c) for c in piv.columns])
        ax.set_xlabel("lag")
        ax.set_title(f"{title}: {win}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        p = out_path.with_name(out_path.stem + f"_{win}" + out_path.suffix)
        fig.savefig(p, dpi=160)
        plt.close(fig)


def _plot_regional_share(contrib: pd.DataFrame, out_path: Path) -> None:
    if plt is None or contrib.empty:
        return
    sub = contrib[contrib["region"].isin(["meiyu_band", "northeast_china", "south_china_scs"])]
    if sub.empty:
        return
    piv = sub.pivot(index="window_subwindow", columns="region", values="regional_positive_share_of_main_domain")
    order = [w for w in ["S3", "T3_early", "T3_late", "S4"] if w in piv.index]
    piv = piv.loc[order]
    ax = piv.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("positive precipitation share of main EASM domain")
    ax.set_title("Regional precipitation contribution")
    ax.figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out_path, dpi=160)
    plt.close(ax.figure)


# -----------------------------
# Main pipeline
# -----------------------------

def run_t3_v_to_p_physical_hypothesis_audit(settings: PhysicalHypothesisAuditSettings) -> Dict[str, Any]:
    dirs = _ensure_dirs(settings)
    run_meta: Dict[str, Any] = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
        "missing_optional_inputs": [],
        "warnings": [],
    }

    index_df, index_path = _load_index_values(settings)
    fields, fields_path = _load_smoothed_fields(settings)
    run_meta["index_values_path"] = str(index_path)
    run_meta["smoothed_fields_path"] = str(fields_path)

    # Main derived diagnostics
    lag_long, sub_summary, dilution_class = _compute_subwindow_v_to_p_profiles(index_df, settings)
    dilution_summary = _summarize_dilution(dilution_class)

    regional_contrib, regional_shift = _regional_precip_contribution(index_df, fields, settings)
    v_response, v_response_shift, v_map_cache = _v_index_regional_response(index_df, fields, settings)
    p_group_summary, p_group_evidence = _p_target_group_shift(sub_summary)
    multi_roll, multi_evidence = _multi_family_stability_shift(settings)
    if multi_roll.empty:
        run_meta["missing_optional_inputs"].append("V1 stability judgement table")
    index_context = _index_validity_context(settings)
    if index_context.empty:
        run_meta["missing_optional_inputs"].append("index_validity context tables")

    hypothesis_summary = _hypothesis_summary(
        dilution_summary=dilution_summary,
        regional_shift=regional_shift,
        v_response_shift=v_response_shift,
        p_group_evidence=p_group_evidence,
        multi_evidence=multi_evidence,
        index_context=index_context,
    )

    # Save tables
    tables = {
        "t3_subwindow_v_to_p_lag_profile_long.csv": lag_long,
        "t3_subwindow_v_to_p_pair_profile.csv": sub_summary,
        "t3_subwindow_v_to_p_dilution_classification.csv": dilution_class,
        "t3_subwindow_dilution_summary.csv": dilution_summary,
        "window_subwindow_regional_precip_contribution.csv": regional_contrib,
        "t3_early_late_regional_precip_shift.csv": regional_shift,
        "t3_v_index_to_regional_p_response.csv": v_response,
        "t3_v_index_subwindow_response_shift.csv": v_response_shift,
        "t3_p_target_group_v_to_p_summary.csv": p_group_summary,
        "t3_p_target_component_shift_evidence.csv": p_group_evidence,
        "s3_t3_s4_multi_family_stability_shift.csv": multi_roll,
        "t3_synchronous_reorganization_evidence.csv": multi_evidence,
        "t3_v_to_p_index_validity_context_for_hypotheses.csv": index_context,
        "t3_physical_hypothesis_evidence_summary.csv": hypothesis_summary,
    }
    for name, df in tables.items():
        df.to_csv(dirs["tables"] / name, index=False)

    # Figures
    figure_rows = []
    if settings.make_figures:
        lat, lon, years = _get_lat_lon_years(fields)
        pkey = _get_field_key(fields, ["precip", "P", "pr", "rain", "precipitation"])
        pfield = np.asarray(fields[pkey], dtype=float)
        year_index = _field_year_day_index(fields, years)

        # Precip structure mean maps
        p_maps = {}
        for win in ["S3", "T3_early", "T3_late", "S4"]:
            sub = _rows_for_window(index_df, win)
            m = _field_mean_for_rows(pfield, sub, year_index)
            if m is not None:
                p_maps[win] = m
        pfig = dirs["figures"] / "precip_structure_S3_T3early_T3late_S4.png"
        _plot_map_grid(p_maps, lat, lon, pfig, settings, "Smoothed precipitation structure")
        if pfig.exists():
            figure_rows.append({"figure": str(pfig), "purpose": "P spatial structure H1/H3"})

        # V index -> P composite maps for T3 early/late/full, selected only to avoid explosion
        select_maps = {k: v for k, v in v_map_cache.items() if any(s in k for s in ["V_strength_P_diff", "V_NS_diff_P_diff", "V_pos_centroid_lat_P_diff"])}
        if select_maps:
            vfig = dirs["figures"] / "t3_v_index_to_p_composite_maps.png"
            _plot_map_grid(select_maps, lat, lon, vfig, settings, "V-index high-low -> P-field composite")
            if vfig.exists():
                figure_rows.append({"figure": str(vfig), "purpose": "V component response H2"})

        hfig = dirs["figures"] / "t3_v_to_p_lag_profile_heatmap_early_late.png"
        _plot_lag_heatmap(lag_long, hfig, "T3 V→P lag profile")
        for p in dirs["figures"].glob("t3_v_to_p_lag_profile_heatmap_early_late_*.png"):
            figure_rows.append({"figure": str(p), "purpose": "early/late dilution H3"})

        rfig = dirs["figures"] / "t3_regional_precip_share_barplot.png"
        _plot_regional_share(regional_contrib, rfig)
        if rfig.exists():
            figure_rows.append({"figure": str(rfig), "purpose": "regional P share H1"})

    fig_manifest = pd.DataFrame(figure_rows)
    fig_manifest.to_csv(dirs["tables"] / "figure_manifest.csv", index=False)

    # README
    readme = _build_readme(hypothesis_summary, run_meta)
    (dirs["summary"] / "T3_PHYSICAL_HYPOTHESIS_AUDIT_README.md").write_text(readme, encoding="utf-8")

    summary: Dict[str, Any] = {
        "status": "success",
        "n_v_to_p_pairs": int(len(dilution_class)),
        "dilution_evidence": dilution_summary.to_dict(orient="records"),
        "hypothesis_evidence": hypothesis_summary.to_dict(orient="records"),
        "n_figures": int(len(fig_manifest)),
        "output_dir": str(settings.output_dir),
    }
    (dirs["summary"] / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (dirs["summary"] / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _build_readme(hypothesis_summary: pd.DataFrame, run_meta: Dict[str, Any]) -> str:
    lines = []
    lines.append("# T3 V→P physical hypothesis audit")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This audit does not rerun the V1 lead-lag screen and does not establish pathway causality. It checks whether the T3 V→P contraction is consistent with several physical hypotheses: rain-band reorganization, V-component shift, internal T3 state mixing/dilution, P-target component shift, and synchronous multi-family reorganization.")
    lines.append("")
    lines.append("## Key outputs")
    lines.append("")
    for f in [
        "t3_subwindow_v_to_p_dilution_classification.csv",
        "window_subwindow_regional_precip_contribution.csv",
        "t3_v_index_to_regional_p_response.csv",
        "t3_p_target_group_v_to_p_summary.csv",
        "s3_t3_s4_multi_family_stability_shift.csv",
        "t3_physical_hypothesis_evidence_summary.csv",
    ]:
        lines.append(f"- `tables/{f}`")
    lines.append("")
    lines.append("## Hypothesis evidence summary")
    lines.append("")
    if hypothesis_summary.empty:
        lines.append("No hypothesis summary was generated.")
    else:
        lines.append(hypothesis_summary.to_markdown(index=False))
    lines.append("")
    if run_meta.get("missing_optional_inputs"):
        lines.append("## Missing optional inputs")
        for x in run_meta.get("missing_optional_inputs", []):
            lines.append(f"- {x}")
    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("Evidence strength in this audit is a diagnostic classification, not a physical mechanism proof. Any `strong` or `moderate` hypothesis should be treated as support for further physical interpretation and map review, not as a final causal pathway.")
    return "\n".join(lines)
