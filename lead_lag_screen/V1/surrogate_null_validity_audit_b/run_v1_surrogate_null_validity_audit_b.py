#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V1 surrogate null validity audit B.

Purpose
-------
This standalone audit checks whether the surrogate/null layer that filters out
positive-lag candidates can be trusted and where the null threshold is pushed up.
It does NOT modify V1 results.

Two checks are implemented:
1) Recompute an AR(1)-style surrogate p-value for selected pairs and compare it
   with V1's existing p_pos_surrogate.
2) Decompose null thresholds into single-lag and max-over-lags components.

Important
---------
The script deliberately does NOT perform broad automatic index-file discovery.
By default it uses the smooth5 path only. If your V1 main-screen used a different
index CSV, pass --index-csv explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(r"D:\easm_project01")
DEFAULT_STABILITY_CSV = DEFAULT_PROJECT_ROOT / r"lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv"
DEFAULT_INDEX_CSV = DEFAULT_PROJECT_ROOT / r"foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / r"lead_lag_screen\V1\surrogate_null_validity_audit_b\outputs"

WINDOWS: Dict[str, Tuple[int, int]] = {
    "S1": (1, 39),
    "T1": (40, 48),
    "S2": (49, 74),
    "T2": (75, 86),
    "S3": (87, 106),
    "T3": (107, 117),
    "S4": (118, 154),
    "T4": (155, 164),
    "S5": (165, 183),
    # readable aliases occasionally used in old files
    "stage_1": (1, 39),
    "early_analysis": (40, 48),
    "stage_2": (49, 74),
    "mid_1_analysis": (75, 86),
    "stage_3": (87, 106),
    "mid_2_analysis": (107, 117),
    "stage_4": (118, 154),
    "late_analysis": (155, 164),
    "stage_5": (165, 183),
}

OLD_WINDOW_ORDER = ["S1", "T1", "S2", "T2", "S3", "T3", "S4", "T4", "S5"]


@dataclass
class ColumnMap:
    window: str
    source: str
    target: str
    source_family: Optional[str]
    target_family: Optional[str]
    family_direction: Optional[str]
    evidence_tier: Optional[str]
    failure_reason: Optional[str]
    lead_lag_label: Optional[str]
    stability_judgement: Optional[str]
    p_pos: Optional[str]
    q_pos: Optional[str]
    observed_r: Optional[str]
    observed_lag: Optional[str]
    n_samples: Optional[str]


def _first_existing(columns: Sequence[str], candidates: Sequence[str], required: bool = False, name: str = "") -> Optional[str]:
    lower_to_real = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    if required:
        raise KeyError(f"Cannot find required column {name or candidates[0]!r}. Tried: {candidates}. Available columns: {list(columns)}")
    return None


def infer_column_map(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)
    return ColumnMap(
        window=_first_existing(cols, ["window", "window_name", "analysis_window"], True, "window"),
        source=_first_existing(cols, ["source_index", "source_variable", "source_var", "source", "x_index", "x_variable", "driver_index", "driver_variable", "cause_index", "cause_variable"], True, "source_index"),
        target=_first_existing(cols, ["target_index", "target_variable", "target_var", "target", "y_index", "y_variable", "response_index", "response_variable", "effect_index", "effect_variable"], True, "target_index"),
        source_family=_first_existing(cols, ["source_family", "source_object", "x_family"]),
        target_family=_first_existing(cols, ["target_family", "target_object", "y_family"]),
        family_direction=_first_existing(cols, ["family_direction", "object_direction", "direction_family"]),
        evidence_tier=_first_existing(cols, ["evidence_tier", "tier", "old_tier"]),
        failure_reason=_first_existing(cols, ["failure_reason", "primary_failure_reason", "reason"]),
        lead_lag_label=_first_existing(cols, ["lead_lag_label", "lead_lag_status", "lead_lag_decision"]),
        stability_judgement=_first_existing(cols, ["v1_stability_judgement", "stability_judgement", "final_judgement", "classification"]),
        p_pos=_first_existing(cols, ["p_pos_surrogate", "p_positive_surrogate", "positive_surrogate_p", "surrogate_p", "p_pos"]),
        q_pos=_first_existing(cols, ["q_pos_within_window", "q_pos", "positive_fdr_q", "fdr_q", "q_value"]),
        observed_r=_first_existing(cols, ["positive_peak_abs_r", "positive_peak_abs_corr", "positive_peak_corr", "best_positive_abs_corr", "best_positive_corr", "r_pos", "corr_pos"]),
        observed_lag=_first_existing(cols, ["positive_peak_lag", "best_positive_lag", "lag_pos", "best_lag"]),
        n_samples=_first_existing(cols, ["n_samples", "n", "sample_size"]),
    )


def normalize_window_name(x: object) -> str:
    s = str(x)
    aliases = {
        "stage1": "S1", "stage_1": "S1", "s1": "S1",
        "transition_t1": "T1", "early": "T1", "early_analysis": "T1", "t1": "T1",
        "stage2": "S2", "stage_2": "S2", "s2": "S2",
        "mid1": "T2", "mid_1": "T2", "mid_1_analysis": "T2", "t2": "T2",
        "stage3": "S3", "stage_3": "S3", "s3": "S3",
        "mid2": "T3", "mid_2": "T3", "mid_2_analysis": "T3", "t3": "T3",
        "stage4": "S4", "stage_4": "S4", "s4": "S4",
        "late": "T4", "late_analysis": "T4", "t4": "T4",
        "stage5": "S5", "stage_5": "S5", "s5": "S5",
    }
    return aliases.get(s.strip().lower(), s)


def family_from_index(name: str) -> str:
    s = str(name)
    for fam in ("P", "V", "H", "Je", "Jw"):
        if s == fam or s.startswith(fam + "_"):
            return fam
    return "unknown"


def is_lead_lag_yes(row: pd.Series, cm: ColumnMap) -> bool:
    if cm.lead_lag_label and pd.notna(row.get(cm.lead_lag_label)):
        return str(row.get(cm.lead_lag_label)).startswith("lead_lag_yes")
    if cm.evidence_tier and pd.notna(row.get(cm.evidence_tier)):
        tier = str(row.get(cm.evidence_tier))
        return tier.startswith("Tier1") or tier.startswith("Tier2") or tier.startswith("Tier3")
    if cm.stability_judgement and pd.notna(row.get(cm.stability_judgement)):
        j = str(row.get(cm.stability_judgement))
        return j in {"stable_lag_dominant", "significant_lagged_but_tau0_coupled", "audit_sensitive"}
    return False


def v1_status(row: pd.Series, cm: ColumnMap) -> str:
    if is_lead_lag_yes(row, cm):
        return "lead_lag_yes"
    reason = str(row.get(cm.failure_reason, "")) if cm.failure_reason else ""
    tier = str(row.get(cm.evidence_tier, "")) if cm.evidence_tier else ""
    if "positive_lag_not_supported" in reason or tier.startswith("Tier5a"):
        return "positive_lag_not_supported"
    if tier.startswith("Tier4"):
        return "tier4_ambiguous"
    return "other_not_supported"


def ensure_output_dirs(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": out_dir,
        "tables": out_dir / "tables",
        "summary": out_dir / "summary",
        "logs": out_dir / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def read_index_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)
    year_col = _first_existing(cols, ["year", "years"], True, "year")
    day_col = _first_existing(cols, ["day", "doy", "day_index", "monsoon_day"], True, "day")
    # long format support
    idx_name_col = _first_existing(cols, ["index_name", "variable", "name"])
    val_col = _first_existing(cols, ["value", "index_value", "anomaly"])
    if idx_name_col and val_col and len(cols) <= 6:
        wide = df.pivot_table(index=[year_col, day_col], columns=idx_name_col, values=val_col, aggfunc="mean").reset_index()
        wide.columns.name = None
        df = wide
        year_col, day_col = year_col, day_col
    if year_col != "year":
        df = df.rename(columns={year_col: "year"})
    if day_col != "day":
        df = df.rename(columns={day_col: "day"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "day"]).copy()
    df["year"] = df["year"].astype(int)
    df["day"] = df["day"].astype(int)
    return df


def window_mask(idx_df: pd.DataFrame, window: str) -> pd.Series:
    w = normalize_window_name(window)
    if w not in WINDOWS:
        raise KeyError(f"Unknown window {window!r}; known: {sorted(WINDOWS)}")
    start, end = WINDOWS[w]
    return (idx_df["day"] >= start) & (idx_df["day"] <= end)


def get_aligned_values(idx_df: pd.DataFrame, source: str, target: str, window: str, lag: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    w = normalize_window_name(window)
    start, end = WINDOWS[w]
    if source not in idx_df.columns or target not in idx_df.columns:
        return np.array([]), np.array([]), pd.DataFrame()
    target_df = idx_df.loc[(idx_df["day"] >= start) & (idx_df["day"] <= end), ["year", "day", target]].copy()
    target_df = target_df.rename(columns={"day": "target_day", target: "y"})
    target_df["source_day"] = target_df["target_day"] - int(lag)
    source_df = idx_df[["year", "day", source]].rename(columns={"day": "source_day", source: "x"})
    merged = target_df.merge(source_df, on=["year", "source_day"], how="inner")
    merged = merged.dropna(subset=["x", "y"]).copy()
    x = pd.to_numeric(merged["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(merged["y"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    merged = merged.loc[finite].copy()
    return x[finite], y[finite], merged


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or len(y) < 4:
        return float("nan")
    sx = np.nanstd(x)
    sy = np.nanstd(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ar1_estimate(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    a = x[:-1]
    b = x[1:]
    if np.nanstd(a) <= 0 or np.nanstd(b) <= 0:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return float(np.clip(r, -0.98, 0.98))


def simulate_ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    phi = float(np.clip(phi if np.isfinite(phi) else 0.0, -0.98, 0.98))
    eps = rng.normal(0, 1, size=n)
    out = np.empty(n, dtype=float)
    out[0] = eps[0]
    scale = math.sqrt(max(1e-8, 1.0 - phi * phi))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + scale * eps[i]
    return out


def compute_observed_by_lag(idx_df: pd.DataFrame, source: str, target: str, window: str, lag_range: Sequence[int]) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, pd.DataFrame]]:
    corrs: Dict[int, float] = {}
    ns: Dict[int, int] = {}
    samples: Dict[int, pd.DataFrame] = {}
    for lag in lag_range:
        x, y, merged = get_aligned_values(idx_df, source, target, window, lag)
        corrs[int(lag)] = corr_safe(x, y)
        ns[int(lag)] = int(len(x))
        samples[int(lag)] = merged
    return corrs, ns, samples


def compute_surrogate_null(
    idx_df: pd.DataFrame,
    source: str,
    target: str,
    window: str,
    lag_range: Sequence[int],
    n_surrogates: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, float], Dict[int, int]]:
    """Return signed max, abs max, per-lag abs null arrays, observed corrs, n samples."""
    observed, ns, samples = compute_observed_by_lag(idx_df, source, target, window, lag_range)
    per_lag_abs: Dict[int, List[float]] = {int(l): [] for l in lag_range}
    signed_max: List[float] = []
    abs_max: List[float] = []

    # Estimate per-lag AR1 from aligned arrays, then simulate independent AR1 sequences.
    # This intentionally serves as a V1-p reproduction diagnostic; poor reproduction is possible and informative.
    lag_arrays: Dict[int, Tuple[int, float, float]] = {}
    for lag in lag_range:
        m = samples[int(lag)]
        if m.empty:
            lag_arrays[int(lag)] = (0, 0.0, 0.0)
            continue
        x = m["x"].to_numpy(dtype=float)
        y = m["y"].to_numpy(dtype=float)
        lag_arrays[int(lag)] = (len(x), ar1_estimate(x), ar1_estimate(y))

    for _ in range(int(n_surrogates)):
        lag_corrs: List[float] = []
        for lag in lag_range:
            n, phix, phiy = lag_arrays[int(lag)]
            if n < 4:
                c = float("nan")
            else:
                xs = simulate_ar1(n, phix, rng)
                ys = simulate_ar1(n, phiy, rng)
                c = corr_safe(xs, ys)
            lag_corrs.append(c)
            if np.isfinite(c):
                per_lag_abs[int(lag)].append(abs(c))
        arr = np.asarray([c for c in lag_corrs if np.isfinite(c)], dtype=float)
        if len(arr) == 0:
            signed_max.append(float("nan"))
            abs_max.append(float("nan"))
        else:
            signed_max.append(float(np.max(arr)))
            abs_max.append(float(np.max(np.abs(arr))))

    per_lag_abs_np = {lag: np.asarray(vals, dtype=float) for lag, vals in per_lag_abs.items()}
    return np.asarray(signed_max, dtype=float), np.asarray(abs_max, dtype=float), per_lag_abs_np, observed, ns


def percentile_safe(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(np.percentile(a, q))


def select_pairs(df: pd.DataFrame, cm: ColumnMap, scope: str, rng: np.random.Generator, comparison_sample_per_window: int) -> pd.DataFrame:
    d = df.copy()
    d["_window_norm"] = d[cm.window].map(normalize_window_name)
    d["_v1_status"] = d.apply(lambda r: v1_status(r, cm), axis=1)

    if scope == "all":
        return d.reset_index(drop=True)

    core_parts: List[pd.DataFrame] = []
    # all T3/T4 positive-lag-not-supported and lead-lag-yes
    core = d[(d["_window_norm"].isin(["T3", "T4"])) & (d["_v1_status"].isin(["positive_lag_not_supported", "lead_lag_yes"]))]
    core_parts.append(core)

    # matched comparison sample from other windows
    for w in OLD_WINDOW_ORDER:
        if w in {"T3", "T4"}:
            continue
        sub = d[(d["_window_norm"] == w) & (d["_v1_status"].isin(["positive_lag_not_supported", "lead_lag_yes"]))]
        if len(sub) > comparison_sample_per_window:
            seed = int(rng.integers(0, 2**31 - 1))
            sub = sub.sample(n=comparison_sample_per_window, random_state=seed)
        core_parts.append(sub)
    out = pd.concat(core_parts, ignore_index=True).drop_duplicates(subset=[cm.window, cm.source, cm.target])
    return out.reset_index(drop=True)


def classify_match(existing_p: float, recomputed_p: float) -> str:
    if not np.isfinite(existing_p) or not np.isfinite(recomputed_p):
        return "unusable"
    diff = abs(existing_p - recomputed_p)
    same05 = (existing_p <= 0.05) == (recomputed_p <= 0.05)
    same10 = (existing_p <= 0.10) == (recomputed_p <= 0.10)
    if diff <= 0.05 or same05:
        return "good_match"
    if diff <= 0.10 or same10:
        return "moderate_match"
    return "poor_match"


def safe_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="V1 surrogate null validity audit B")
    parser.add_argument("--stability-csv", default=str(DEFAULT_STABILITY_CSV), help="V1 stability judgement CSV")
    parser.add_argument("--index-csv", default=str(DEFAULT_INDEX_CSV), help="Smooth5 index_anomalies CSV. No broad auto-discovery is performed.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--n-surrogates", type=int, default=1000)
    parser.add_argument("--debug-fast", action="store_true", help="Use fewer surrogates and smaller sample")
    parser.add_argument("--pair-scope", choices=["core", "all"], default="core")
    parser.add_argument("--comparison-sample-per-window", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--lag-range", default="1,2,3,4,5")
    args = parser.parse_args(argv)

    t0 = time.time()
    stability_csv = Path(args.stability_csv)
    index_csv = Path(args.index_csv)
    out_dir = Path(args.output_dir)
    paths = ensure_output_dirs(out_dir)

    if args.debug_fast:
        args.n_surrogates = min(args.n_surrogates, 100)
        args.comparison_sample_per_window = min(args.comparison_sample_per_window, 20)

    if not stability_csv.exists():
        raise FileNotFoundError(f"Missing stability CSV: {stability_csv}")
    if not index_csv.exists():
        raise FileNotFoundError(
            f"Missing index CSV: {index_csv}\n"
            "This audit intentionally avoids broad auto-search to prevent smooth5/smooth9 mix-ups. "
            "Pass --index-csv explicitly if needed."
        )
    if "baseline_smooth5_a" not in str(index_csv):
        print(
            "WARNING: index_csv path does not contain 'baseline_smooth5_a'. "
            "Verify this is the exact V1 smooth5 main-screen index table.",
            file=sys.stderr,
        )

    lag_range = [int(x) for x in str(args.lag_range).split(",") if str(x).strip()]
    rng = np.random.default_rng(args.seed)

    stability = pd.read_csv(stability_csv)
    cm = infer_column_map(stability)
    stability["_window_norm"] = stability[cm.window].map(normalize_window_name)
    stability["_v1_status"] = stability.apply(lambda r: v1_status(r, cm), axis=1)
    stability["_source_family"] = stability[cm.source_family] if cm.source_family else stability[cm.source].map(family_from_index)
    stability["_target_family"] = stability[cm.target_family] if cm.target_family else stability[cm.target].map(family_from_index)
    stability["_family_direction"] = stability[cm.family_direction] if cm.family_direction else (stability["_source_family"].astype(str) + "→" + stability["_target_family"].astype(str))

    idx = read_index_csv(index_csv)
    idx_cols = set(idx.columns)

    selected = select_pairs(stability, cm, args.pair_scope, rng, args.comparison_sample_per_window)
    selected = selected.reset_index(drop=True)

    sample_rows: List[dict] = []
    repro_rows: List[dict] = []
    decomp_rows: List[dict] = []

    for k, row in selected.iterrows():
        window = normalize_window_name(row[cm.window])
        source = str(row[cm.source])
        target = str(row[cm.target])
        if source not in idx_cols or target not in idx_cols or window not in WINDOWS:
            # write an unusable row and continue
            repro_rows.append({
                "window": window,
                "source_index": source,
                "target_index": target,
                "family_direction": row.get("_family_direction", ""),
                "v1_status": row.get("_v1_status", ""),
                "failure_reason": row.get(cm.failure_reason, "") if cm.failure_reason else "",
                "observed_stat_recomputed": np.nan,
                "observed_lag_recomputed": np.nan,
                "p_pos_surrogate_existing": safe_float(row.get(cm.p_pos)) if cm.p_pos else np.nan,
                "p_pos_surrogate_recomputed": np.nan,
                "abs_diff": np.nan,
                "same_decision_at_005": False,
                "same_decision_at_010": False,
                "match_class": "unusable_missing_index_or_window",
            })
            continue

        obs_corrs, ns, samples = compute_observed_by_lag(idx, source, target, window, lag_range)
        for lag in lag_range:
            m = samples[int(lag)]
            same_year_ok = bool((m["year"].notna()).all()) if not m.empty else False
            sample_rows.append({
                "window": window,
                "source_index": source,
                "target_index": target,
                "lag": lag,
                "n_samples_recomputed": ns[int(lag)],
                "n_samples_existing": safe_float(row.get(cm.n_samples)) if cm.n_samples else np.nan,
                "target_day_min": int(m["target_day"].min()) if not m.empty else np.nan,
                "target_day_max": int(m["target_day"].max()) if not m.empty else np.nan,
                "source_day_min": int(m["source_day"].min()) if not m.empty else np.nan,
                "source_day_max": int(m["source_day"].max()) if not m.empty else np.nan,
                "same_year_check": same_year_ok,
                "valid_sample_flag": ns[int(lag)] >= 4,
            })

        signed_max, abs_max, per_lag_abs, observed, ns2 = compute_surrogate_null(
            idx, source, target, window, lag_range, args.n_surrogates, rng
        )
        obs_abs = {lag: abs(v) if np.isfinite(v) else np.nan for lag, v in observed.items()}
        valid_obs = {lag: val for lag, val in obs_abs.items() if np.isfinite(val)}
        if valid_obs:
            obs_lag = max(valid_obs, key=lambda l: valid_obs[l])
            obs_stat = float(valid_obs[obs_lag])
        else:
            obs_lag = np.nan
            obs_stat = np.nan
        null_valid = abs_max[np.isfinite(abs_max)]
        if len(null_valid) > 0 and np.isfinite(obs_stat):
            p_re = float((1 + np.sum(null_valid >= obs_stat)) / (len(null_valid) + 1))
        else:
            p_re = np.nan
        p_existing = safe_float(row.get(cm.p_pos)) if cm.p_pos else np.nan
        q_existing = safe_float(row.get(cm.q_pos)) if cm.q_pos else np.nan
        abs_diff = abs(p_existing - p_re) if np.isfinite(p_existing) and np.isfinite(p_re) else np.nan
        same05 = bool((p_existing <= 0.05) == (p_re <= 0.05)) if np.isfinite(p_existing) and np.isfinite(p_re) else False
        same10 = bool((p_existing <= 0.10) == (p_re <= 0.10)) if np.isfinite(p_existing) and np.isfinite(p_re) else False

        single_p95_vals = [percentile_safe(per_lag_abs[int(lag)], 95) for lag in lag_range]
        single_lag_p95_mean = float(np.nanmean(single_p95_vals)) if np.any(np.isfinite(single_p95_vals)) else np.nan
        single_lag_p95_max = float(np.nanmax(single_p95_vals)) if np.any(np.isfinite(single_p95_vals)) else np.nan
        max_lag_signed_p95 = percentile_safe(signed_max, 95)
        max_lag_abs_p95 = percentile_safe(abs_max, 95)
        max_lag_abs_p99 = percentile_safe(abs_max, 99)
        fdr_pressure = q_existing / p_existing if np.isfinite(q_existing) and np.isfinite(p_existing) and p_existing > 0 else np.nan

        base = {
            "window": window,
            "source_index": source,
            "target_index": target,
            "family_direction": row.get("_family_direction", ""),
            "v1_status": row.get("_v1_status", ""),
            "failure_reason": row.get(cm.failure_reason, "") if cm.failure_reason else "",
            "evidence_tier": row.get(cm.evidence_tier, "") if cm.evidence_tier else "",
            "v1_stability_judgement": row.get(cm.stability_judgement, "") if cm.stability_judgement else "",
        }
        repro_rows.append({
            **base,
            "observed_stat_recomputed": obs_stat,
            "observed_lag_recomputed": obs_lag,
            "p_pos_surrogate_existing": p_existing,
            "p_pos_surrogate_recomputed": p_re,
            "abs_diff": abs_diff,
            "same_decision_at_005": same05,
            "same_decision_at_010": same10,
            "match_class": classify_match(p_existing, p_re),
            "n_surrogate_valid": int(len(null_valid)),
        })
        decomp_rows.append({
            **base,
            "observed_positive_peak_abs_r": obs_stat,
            "observed_positive_peak_lag": obs_lag,
            "single_lag_p95_mean": single_lag_p95_mean,
            "single_lag_p95_max": single_lag_p95_max,
            "max_lag_signed_p95": max_lag_signed_p95,
            "max_lag_abs_p95": max_lag_abs_p95,
            "max_lag_abs_p99": max_lag_abs_p99,
            "observed_minus_single_lag_p95_mean": obs_stat - single_lag_p95_mean if np.isfinite(obs_stat) and np.isfinite(single_lag_p95_mean) else np.nan,
            "observed_minus_max_lag_abs_p95": obs_stat - max_lag_abs_p95 if np.isfinite(obs_stat) and np.isfinite(max_lag_abs_p95) else np.nan,
            "observed_over_max_lag_abs_p95": obs_stat / max_lag_abs_p95 if np.isfinite(obs_stat) and np.isfinite(max_lag_abs_p95) and max_lag_abs_p95 != 0 else np.nan,
            "p_pos_surrogate_existing": p_existing,
            "p_pos_surrogate_recomputed": p_re,
            "q_pos_within_window": q_existing,
            "fdr_pressure_ratio": fdr_pressure,
            "single_to_max_lag_p95_increase": max_lag_abs_p95 - single_lag_p95_mean if np.isfinite(max_lag_abs_p95) and np.isfinite(single_lag_p95_mean) else np.nan,
        })

        if (k + 1) % 50 == 0:
            print(f"Processed {k + 1}/{len(selected)} selected pairs...", flush=True)

    sample_df = pd.DataFrame(sample_rows)
    repro_df = pd.DataFrame(repro_rows)
    decomp_df = pd.DataFrame(decomp_rows)

    sample_df.to_csv(paths["tables"] / "null_input_sample_consistency.csv", index=False)
    repro_df.to_csv(paths["tables"] / "surrogate_p_reproduction_check.csv", index=False)
    decomp_df.to_csv(paths["tables"] / "null_threshold_decomposition_by_pair.csv", index=False)

    # summaries
    if not repro_df.empty:
        reproduction_summary = (
            repro_df.groupby("window", dropna=False)
            .agg(
                n_pairs_checked=("source_index", "count"),
                good_match_rate=("match_class", lambda s: float(np.mean(s == "good_match"))),
                moderate_or_good_match_rate=("match_class", lambda s: float(np.mean(s.isin(["good_match", "moderate_match"])))),
                poor_match_rate=("match_class", lambda s: float(np.mean(s == "poor_match"))),
                unusable_rate=("match_class", lambda s: float(np.mean(s.astype(str).str.startswith("unusable")))),
                existing_fail_but_recomputed_pass=("same_decision_at_005", lambda s: np.nan),
            )
            .reset_index()
        )
        # explicit decision-flip counts at 0.05
        flip_rows = []
        for w, sub in repro_df.groupby("window"):
            ex = pd.to_numeric(sub["p_pos_surrogate_existing"], errors="coerce")
            re = pd.to_numeric(sub["p_pos_surrogate_recomputed"], errors="coerce")
            flip_rows.append({
                "window": w,
                "existing_fail_but_recomputed_pass": int(((ex > 0.05) & (re <= 0.05)).sum()),
                "existing_pass_but_recomputed_fail": int(((ex <= 0.05) & (re > 0.05)).sum()),
            })
        flips = pd.DataFrame(flip_rows)
        reproduction_summary = reproduction_summary.drop(columns=["existing_fail_but_recomputed_pass"]).merge(flips, on="window", how="left")
    else:
        reproduction_summary = pd.DataFrame()
    reproduction_summary.to_csv(paths["tables"] / "surrogate_p_reproduction_summary_by_window.csv", index=False)

    if not decomp_df.empty:
        decomp_summary = (
            decomp_df.groupby("window", dropna=False)
            .agg(
                n_pairs=("source_index", "count"),
                observed_r_median=("observed_positive_peak_abs_r", "median"),
                single_lag_p95_median=("single_lag_p95_mean", "median"),
                max_lag_abs_p95_median=("max_lag_abs_p95", "median"),
                observed_minus_single_lag_p95_median=("observed_minus_single_lag_p95_mean", "median"),
                observed_minus_max_lag_abs_p95_median=("observed_minus_max_lag_abs_p95", "median"),
                single_to_max_lag_p95_increase_median=("single_to_max_lag_p95_increase", "median"),
                fdr_pressure_median=("fdr_pressure_ratio", "median"),
            )
            .reset_index()
        )
    else:
        decomp_summary = pd.DataFrame()
    decomp_summary.to_csv(paths["tables"] / "null_threshold_decomposition_by_window.csv", index=False)

    # diagnosis
    diag: List[dict] = []
    if not reproduction_summary.empty:
        overall_good = float((repro_df["match_class"] == "good_match").mean())
        overall_mod_good = float(repro_df["match_class"].isin(["good_match", "moderate_match"]).mean())
        support = "supported" if overall_mod_good >= 0.70 else ("mixed" if overall_mod_good >= 0.40 else "not_supported")
        diag.append({
            "diagnosis_id": "surrogate_p_reproduces_v1",
            "support_level": support,
            "primary_evidence": f"moderate_or_good_match_rate={overall_mod_good:.3f}; good_match_rate={overall_good:.3f}; n={len(repro_df)}",
            "counter_evidence": "AR(1) surrogate reproduction is diagnostic and may differ from exact V1 implementation.",
            "allowed_statement": "If supported, the recomputed null is close enough to explain V1 surrogate failures at audit level.",
            "forbidden_statement": "Do not treat this diagnostic as a replacement for V1's original p-values.",
        })
    for w in ["T3", "T4"]:
        if not decomp_summary.empty and w in set(decomp_summary["window"]):
            r = decomp_summary.loc[decomp_summary["window"] == w].iloc[0]
            obs_minus_single = safe_float(r.get("observed_minus_single_lag_p95_median"))
            obs_minus_max = safe_float(r.get("observed_minus_max_lag_abs_p95_median"))
            max_increase = safe_float(r.get("single_to_max_lag_p95_increase_median"))
            diag.append({
                "diagnosis_id": f"{w.lower()}_null_high_due_to_single_lag_threshold",
                "support_level": "supported" if np.isfinite(obs_minus_single) and obs_minus_single < 0 else "not_supported",
                "primary_evidence": f"median observed_minus_single_lag_p95={obs_minus_single:.3f}" if np.isfinite(obs_minus_single) else "missing",
                "counter_evidence": "Negative margin can also reflect observed signal weakness, not only a high null threshold.",
                "allowed_statement": f"{w} observed peaks do not exceed single-lag AR(1) null threshold if supported.",
                "forbidden_statement": "Do not infer physical absence of relationship solely from this margin.",
            })
            diag.append({
                "diagnosis_id": f"{w.lower()}_null_high_due_to_max_over_lags",
                "support_level": "supported" if np.isfinite(max_increase) and max_increase > 0.03 else "mixed_or_weak",
                "primary_evidence": f"median max-lag p95 increase over single-lag p95={max_increase:.3f}" if np.isfinite(max_increase) else "missing",
                "counter_evidence": "Max-over-lags is a designed conservative null, not necessarily an implementation error.",
                "allowed_statement": f"{w} null threshold is partly elevated by max-over-lags if supported.",
                "forbidden_statement": "Do not call max-over-lags a bug without sample/p-value reproduction evidence.",
            })
            fdrp = safe_float(r.get("fdr_pressure_median"))
            diag.append({
                "diagnosis_id": f"{w.lower()}_failure_due_to_fdr_pressure",
                "support_level": "supported" if np.isfinite(fdrp) and fdrp > 1.5 else "not_supported_or_unclear",
                "primary_evidence": f"median q/p pressure ratio={fdrp:.3f}" if np.isfinite(fdrp) else "missing",
                "counter_evidence": "q/p pressure is undefined when p is missing or near zero.",
                "allowed_statement": f"{w} failures include an FDR pressure component if supported.",
                "forbidden_statement": "Do not attribute all failures to FDR unless p-value reproduction and threshold margins support it.",
            })
    null_impl_suspect = False
    if not reproduction_summary.empty:
        mod_good = float(repro_df["match_class"].isin(["good_match", "moderate_match"]).mean())
        null_impl_suspect = mod_good < 0.40
    diag.append({
        "diagnosis_id": "null_implementation_suspect",
        "support_level": "supported" if null_impl_suspect else "not_supported_or_unresolved",
        "primary_evidence": "low surrogate p reproduction rate" if null_impl_suspect else "surrogate p reproduction not low enough to flag implementation as suspect",
        "counter_evidence": "This audit uses diagnostic AR(1) surrogates; exact V1 source implementation may still differ.",
        "allowed_statement": "If supported, do not explain V1 surrogate failures until V1 null implementation is reconciled.",
        "forbidden_statement": "Do not label V1 results invalid solely from this diagnostic without inspecting original V1 code.",
    })
    pd.DataFrame(diag).to_csv(paths["tables"] / "surrogate_null_validity_diagnosis.csv", index=False)

    run_meta = {
        "status": "success",
        "audit": "surrogate_null_validity_audit_b",
        "stability_csv": str(stability_csv),
        "index_csv": str(index_csv),
        "output_dir": str(out_dir),
        "column_map": asdict(cm),
        "lag_range": lag_range,
        "n_surrogates": args.n_surrogates,
        "pair_scope": args.pair_scope,
        "comparison_sample_per_window": args.comparison_sample_per_window,
        "seed": args.seed,
        "debug_fast": args.debug_fast,
        "runtime_seconds": round(time.time() - t0, 3),
        "note": "Diagnostic AR(1) surrogate reproduction; does not replace V1 original p-values.",
    }
    with open(paths["summary"] / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    summary = {
        "status": "success",
        "n_selected_pairs": int(len(selected)),
        "n_reproduction_rows": int(len(repro_df)),
        "n_decomposition_rows": int(len(decomp_df)),
        "moderate_or_good_match_rate": float(repro_df["match_class"].isin(["good_match", "moderate_match"]).mean()) if not repro_df.empty else None,
        "good_match_rate": float((repro_df["match_class"] == "good_match").mean()) if not repro_df.empty else None,
    }
    with open(paths["summary"] / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(paths["logs"] / "RUN_LOG.md", "w", encoding="utf-8") as f:
        f.write("# V1 surrogate null validity audit B\n\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")
    print(f"Done. Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
