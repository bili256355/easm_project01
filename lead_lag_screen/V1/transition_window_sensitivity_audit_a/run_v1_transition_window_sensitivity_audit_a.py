#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V1 transition-window sensitivity audit A.

Purpose
-------
Diagnose whether low relationship counts in transition windows (T1/T2/T3/T4)
can be explained by window length / boundary choices / low effective sample size
rather than robust transition-window weakening.

This script is intentionally independent from the V1 main pipeline. It does not
modify V1 results. It reads:
  1) V1 stability judgement summary CSV, for original-window references;
  2) V1 smooth5 anomaly index time series, for recomputed lightweight diagnostics.

Default index path is strict smooth5 to avoid mixing 5-day / 9-day products.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Defaults and window registry
# -----------------------------

DEFAULT_STABILITY_CSV = Path(
    r"D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv"
)
DEFAULT_INDEX_CSV = Path(
    r"D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv"
)

# V1 main-screen windows supplied by user/project context. Inclusive day numbers.
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
}
TRANSITIONS = {
    "T1": ("S1", "S2"),
    "T2": ("S2", "S3"),
    "T3": ("S3", "S4"),
    "T4": ("S4", "S5"),
}

V_FAMILIES = ("P", "V", "H", "Je", "Jw")

# -----------------------------
# Settings / helpers
# -----------------------------

@dataclass
class Settings:
    stability_csv: Path
    index_csv: Path
    output_dir: Path
    n_random: int = 200
    random_seed: int = 20260430
    debug_fast: bool = False
    lag_max: int = 5
    alpha: float = 0.05
    strict_margin: float = 0.02
    max_pairs: Optional[int] = None


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="V1 transition-window sensitivity audit A")
    parser.add_argument("--stability-csv", type=Path, default=DEFAULT_STABILITY_CSV)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260430)
    parser.add_argument("--lag-max", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--strict-margin", type=float, default=0.02)
    parser.add_argument("--debug-fast", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for development only.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    out = args.output_dir or (here / "outputs")
    n_random = 30 if args.debug_fast else args.n_random
    max_pairs = args.max_pairs
    if args.debug_fast and max_pairs is None:
        max_pairs = 80
    return Settings(
        stability_csv=args.stability_csv,
        index_csv=args.index_csv,
        output_dir=out,
        n_random=n_random,
        random_seed=args.random_seed,
        debug_fast=args.debug_fast,
        lag_max=args.lag_max,
        alpha=args.alpha,
        strict_margin=args.strict_margin,
        max_pairs=max_pairs,
    )


def ensure_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "base": base,
        "tables": base / "tables",
        "summary": base / "summary",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def first_existing(cols: Iterable[str], candidates: Sequence[str], required: bool, name: str) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(f"Cannot find required column {name!r}. Tried {list(candidates)}")
    return None


@dataclass
class StabilityCols:
    window: str
    source: str
    target: str
    source_family: Optional[str]
    target_family: Optional[str]
    family_direction: Optional[str]
    evidence_tier: Optional[str]
    failure_reason: Optional[str]
    stability_judgement: Optional[str]
    lead_lag_label: Optional[str]


def infer_stability_cols(df: pd.DataFrame) -> StabilityCols:
    cols = list(df.columns)
    return StabilityCols(
        window=first_existing(cols, ["window", "window_name", "analysis_window"], True, "window"),
        source=first_existing(cols, [
            "source_variable", "source_index", "source", "source_var", "x_variable", "x_index",
            "driver_variable", "driver_index", "cause_variable", "cause_index",
        ], True, "source variable"),
        target=first_existing(cols, [
            "target_variable", "target_index", "target", "target_var", "y_variable", "y_index",
            "response_variable", "effect_variable",
        ], True, "target variable"),
        source_family=first_existing(cols, ["source_family", "source_object", "x_family"], False, "source_family"),
        target_family=first_existing(cols, ["target_family", "target_object", "y_family"], False, "target_family"),
        family_direction=first_existing(cols, ["family_direction", "direction", "object_direction"], False, "family_direction"),
        evidence_tier=first_existing(cols, ["evidence_tier", "tier", "raw_tier"], False, "evidence_tier"),
        failure_reason=first_existing(cols, ["failure_reason", "reason", "primary_failure_reason"], False, "failure_reason"),
        stability_judgement=first_existing(cols, ["v1_stability_judgement", "stability_judgement", "classification"], False, "v1_stability_judgement"),
        lead_lag_label=first_existing(cols, ["lead_lag_label", "lead_lag_status", "lead_lag"], False, "lead_lag_label"),
    )


def infer_family(index_name: str) -> str:
    s = str(index_name)
    if s.startswith("Je"):
        return "Je"
    if s.startswith("Jw"):
        return "Jw"
    if s.startswith("P_") or s == "P":
        return "P"
    if s.startswith("V_") or s == "V":
        return "V"
    if s.startswith("H_") or s == "H":
        return "H"
    # fallback: first token
    return s.split("_")[0]


def family_direction(src: str, tgt: str) -> str:
    return f"{infer_family(src)}→{infer_family(tgt)}"


def is_lead_lag_yes_row(row: pd.Series, cm: StabilityCols) -> bool:
    if cm.lead_lag_label and pd.notna(row.get(cm.lead_lag_label)):
        return "yes" in str(row[cm.lead_lag_label]).lower()
    if cm.evidence_tier and pd.notna(row.get(cm.evidence_tier)):
        tier = str(row[cm.evidence_tier])
        return tier.startswith("Tier1") or tier.startswith("Tier2") or tier.startswith("Tier3")
    if cm.stability_judgement and pd.notna(row.get(cm.stability_judgement)):
        v = str(row[cm.stability_judgement])
        return v in ("stable_lag_dominant", "significant_lagged_but_tau0_coupled")
    return False


def is_strict_or_tau0(row: pd.Series, cm: StabilityCols) -> bool:
    if not cm.stability_judgement:
        return False
    v = str(row.get(cm.stability_judgement, ""))
    return v in ("stable_lag_dominant", "significant_lagged_but_tau0_coupled")

# -----------------------------
# Index table loader
# -----------------------------


def load_index_table(index_csv: Path) -> pd.DataFrame:
    if not index_csv.exists():
        raise FileNotFoundError(
            f"Cannot find smooth5 index CSV: {index_csv}\n"
            "This audit intentionally refuses broad auto-search to avoid mixing 5-day / 9-day products."
        )
    df = pd.read_csv(index_csv)
    cols = list(df.columns)
    year_col = first_existing(cols, ["year", "Year"], True, "year")
    day_col = first_existing(cols, ["day", "doy", "day_index", "monsoon_day"], True, "day")
    if year_col != "year":
        df = df.rename(columns={year_col: "year"})
    if day_col != "day":
        df = df.rename(columns={day_col: "day"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "day"]).copy()
    df["year"] = df["year"].astype(int)
    df["day"] = df["day"].astype(int)
    df = df.sort_values(["year", "day"]).reset_index(drop=True)
    return df


def make_lookup(index_df: pd.DataFrame, index_names: Sequence[str]) -> Dict[str, Dict[Tuple[int, int], float]]:
    lookup: Dict[str, Dict[Tuple[int, int], float]] = {}
    keys = list(zip(index_df["year"].astype(int), index_df["day"].astype(int)))
    for name in index_names:
        if name not in index_df.columns:
            continue
        vals = pd.to_numeric(index_df[name], errors="coerce").to_numpy(dtype=float)
        lookup[name] = {k: v for k, v in zip(keys, vals) if np.isfinite(v)}
    return lookup

# -----------------------------
# Statistics
# -----------------------------


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan
    x = x[m]
    y = y[m]
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def fisher_threshold(neff: float, alpha: float = 0.05, max_lag: int = 1, absolute: bool = True) -> float:
    """Approximate AR(1)-effective-n null threshold for correlation magnitude.

    Uses Fisher-z normal approximation. For max_lag>1, applies Bonferroni-style
    correction. This is a diagnostic approximation, not replacement for V1 p-values.
    """
    if not np.isfinite(neff) or neff <= 4:
        return np.nan
    # two-sided for absolute r; one-sided-ish if absolute=False
    from statistics import NormalDist
    nd = NormalDist()
    if absolute:
        # P(max |Z_lag| <= z) ≈ (P(|Z| <= z))^max_lag = 1-alpha
        single_cdf_abs = (1.0 - alpha) ** (1.0 / max_lag)
        # P(|Z| <= z) = 2Phi(z)-1
        phi = (single_cdf_abs + 1.0) / 2.0
    else:
        phi = (1.0 - alpha) ** (1.0 / max_lag)
    zcrit = nd.inv_cdf(phi)
    se = 1.0 / math.sqrt(max(neff - 3.0, 1.0))
    return float(math.tanh(zcrit * se))


def effective_n_pair(n: int, ar1_x: float, ar1_y: float) -> float:
    if n <= 4:
        return np.nan
    rx = 0.0 if not np.isfinite(ar1_x) else float(np.clip(ar1_x, -0.99, 0.99))
    ry = 0.0 if not np.isfinite(ar1_y) else float(np.clip(ar1_y, -0.99, 0.99))
    rho = rx * ry
    neff = n * (1.0 - rho) / (1.0 + rho)
    return float(max(3.1, min(n, neff)))


def yearwise_ar1(values_by_key: Dict[Tuple[int, int], float], day_start: int, day_end: int) -> Tuple[float, int, float]:
    xs = []
    ys = []
    vals_all = []
    years = sorted({y for (y, d) in values_by_key.keys() if day_start <= d <= day_end})
    for yr in years:
        arr = []
        for d in range(day_start, day_end + 1):
            v = values_by_key.get((yr, d), np.nan)
            if np.isfinite(v):
                arr.append((d, float(v)))
        if len(arr) >= 2:
            arr = sorted(arr)
            for (_, a), (_, b) in zip(arr[:-1], arr[1:]):
                if np.isfinite(a) and np.isfinite(b):
                    xs.append(a)
                    ys.append(b)
            vals_all.extend([v for _, v in arr])
    ar1 = corr_safe(np.array(xs), np.array(ys)) if len(xs) >= 4 else np.nan
    n = len(vals_all)
    var = float(np.nanvar(vals_all)) if vals_all else np.nan
    return ar1, n, var


def paired_samples(
    lookup: Dict[str, Dict[Tuple[int, int], float]],
    source: str,
    target: str,
    day_start: int,
    day_end: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sx = lookup.get(source)
    ty = lookup.get(target)
    if sx is None or ty is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs: List[float] = []
    ys: List[float] = []
    # years available in target domain
    years = sorted({y for (y, d) in ty.keys() if day_start <= d <= day_end})
    for yr in years:
        for td in range(day_start, day_end + 1):
            sd = td - lag
            xv = sx.get((yr, sd), np.nan)
            yv = ty.get((yr, td), np.nan)
            if np.isfinite(xv) and np.isfinite(yv):
                xs.append(float(xv))
                ys.append(float(yv))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def lag0_samples(
    lookup: Dict[str, Dict[Tuple[int, int], float]],
    source: str,
    target: str,
    day_start: int,
    day_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return paired_samples(lookup, source, target, day_start, day_end, lag=0)

@dataclass
class VariantResult:
    variant_name: str
    base_transition: str
    variant_type: str
    day_start: int
    day_end: int
    length: int
    n_pairs_total: int
    n_pairs_valid: int
    lead_lag_like: int
    tau0_coupled_like: int
    strict_lag_like: int
    positive_lag_not_supported_like: int
    median_observed_abs_r: float
    median_null_p95: float
    median_observed_minus_null: float
    median_effective_n: float
    median_source_ar1: float
    median_target_ar1: float
    existing_lead_lag_yes: Optional[int] = None
    existing_strict_tau0: Optional[int] = None


def compute_pair_stats_for_window(
    lookup: Dict[str, Dict[Tuple[int, int], float]],
    pairs: Sequence[Tuple[str, str]],
    day_start: int,
    day_end: int,
    settings: Settings,
) -> pd.DataFrame:
    rows = []
    # Precompute AR1 by index for this target window. This is diagnostic approximate.
    indices = sorted({x for p in pairs for x in p})
    ar_cache: Dict[str, Tuple[float, int, float]] = {}
    for idx in indices:
        if idx in lookup:
            ar_cache[idx] = yearwise_ar1(lookup[idx], day_start, day_end)
        else:
            ar_cache[idx] = (np.nan, 0, np.nan)

    for source, target in pairs:
        if source not in lookup or target not in lookup:
            rows.append({
                "source_index": source, "target_index": target,
                "valid_pair": False, "reason": "missing_index_in_index_csv",
            })
            continue
        lag_corrs = []
        lag_ns = []
        for lag in range(1, settings.lag_max + 1):
            x, y = paired_samples(lookup, source, target, day_start, day_end, lag)
            r = corr_safe(x, y)
            lag_corrs.append(r)
            lag_ns.append(int(np.isfinite(x).sum()))
        abs_corrs = np.abs(np.array(lag_corrs, dtype=float))
        if np.all(~np.isfinite(abs_corrs)):
            best_lag = np.nan
            best_r = np.nan
            observed_abs = np.nan
            n = 0
        else:
            k = int(np.nanargmax(abs_corrs))
            best_lag = k + 1
            best_r = float(lag_corrs[k])
            observed_abs = float(abs_corrs[k])
            n = int(lag_ns[k])
        x0, y0 = lag0_samples(lookup, source, target, day_start, day_end)
        r0 = corr_safe(x0, y0)
        lag0_abs = abs(r0) if np.isfinite(r0) else np.nan
        source_ar1, source_n, source_var = ar_cache.get(source, (np.nan, 0, np.nan))
        target_ar1, target_n, target_var = ar_cache.get(target, (np.nan, 0, np.nan))
        neff = effective_n_pair(n, source_ar1, target_ar1)
        single_p95 = fisher_threshold(neff, settings.alpha, max_lag=1, absolute=True)
        max_p95 = fisher_threshold(neff, settings.alpha, max_lag=settings.lag_max, absolute=True)
        observed_minus = observed_abs - max_p95 if np.isfinite(observed_abs) and np.isfinite(max_p95) else np.nan
        lead_lag_like = bool(np.isfinite(observed_abs) and np.isfinite(max_p95) and observed_abs >= max_p95)
        tau0_coupled_like = bool(lead_lag_like and np.isfinite(lag0_abs) and lag0_abs >= (observed_abs - settings.strict_margin))
        strict_lag_like = bool(lead_lag_like and not tau0_coupled_like)
        rows.append({
            "source_index": source,
            "target_index": target,
            "source_family": infer_family(source),
            "target_family": infer_family(target),
            "family_direction": family_direction(source, target),
            "valid_pair": True,
            "day_start": day_start,
            "day_end": day_end,
            "window_length": day_end - day_start + 1,
            "n_samples_best_lag": n,
            "best_positive_lag": best_lag,
            "best_positive_corr": best_r,
            "best_positive_abs_r": observed_abs,
            "lag0_corr": r0,
            "lag0_abs_r": lag0_abs,
            "source_ar1": source_ar1,
            "target_ar1": target_ar1,
            "source_ar1_n": source_n,
            "target_ar1_n": target_n,
            "pair_effective_n": neff,
            "single_lag_p95_approx": single_p95,
            "max_lag_abs_p95_approx": max_p95,
            "observed_minus_max_null": observed_minus,
            "lead_lag_like": lead_lag_like,
            "tau0_coupled_like": tau0_coupled_like,
            "strict_lag_like": strict_lag_like,
            "positive_lag_not_supported_like": bool(not lead_lag_like),
        })
    return pd.DataFrame(rows)


def summarize_variant(df: pd.DataFrame, variant_name: str, base_t: str, variant_type: str, start: int, end: int,
                      existing_counts: Optional[Tuple[int, int]] = None) -> VariantResult:
    valid = df[df.get("valid_pair", False) == True].copy() if not df.empty else df
    lead = int(valid["lead_lag_like"].sum()) if "lead_lag_like" in valid else 0
    tau0 = int(valid["tau0_coupled_like"].sum()) if "tau0_coupled_like" in valid else 0
    strict = int(valid["strict_lag_like"].sum()) if "strict_lag_like" in valid else 0
    notsup = int(valid["positive_lag_not_supported_like"].sum()) if "positive_lag_not_supported_like" in valid else 0
    ex_lead, ex_st = (existing_counts if existing_counts is not None else (None, None))
    def med(col):
        return float(pd.to_numeric(valid.get(col, pd.Series(dtype=float)), errors="coerce").median()) if not valid.empty and col in valid else np.nan
    return VariantResult(
        variant_name=variant_name,
        base_transition=base_t,
        variant_type=variant_type,
        day_start=start,
        day_end=end,
        length=end - start + 1,
        n_pairs_total=int(len(df)),
        n_pairs_valid=int(len(valid)),
        lead_lag_like=lead,
        tau0_coupled_like=tau0,
        strict_lag_like=strict,
        positive_lag_not_supported_like=notsup,
        median_observed_abs_r=med("best_positive_abs_r"),
        median_null_p95=med("max_lag_abs_p95_approx"),
        median_observed_minus_null=med("observed_minus_max_null"),
        median_effective_n=med("pair_effective_n"),
        median_source_ar1=med("source_ar1"),
        median_target_ar1=med("target_ar1"),
        existing_lead_lag_yes=ex_lead,
        existing_strict_tau0=ex_st,
    )

# -----------------------------
# Variant builders
# -----------------------------


def clamp_window(start: int, end: int, min_day: int = 1, max_day: int = 183) -> Tuple[int, int]:
    start = max(min_day, int(start))
    end = min(max_day, int(end))
    if start > end:
        start, end = end, start
    return start, end


def subwindow_center(stage: Tuple[int, int], length: int) -> Tuple[int, int]:
    s, e = stage
    total = e - s + 1
    if length >= total:
        return s, e
    center = (s + e) // 2
    start = center - length // 2
    end = start + length - 1
    if start < s:
        start = s; end = s + length - 1
    if end > e:
        end = e; start = e - length + 1
    return start, end


def subwindow_early(stage: Tuple[int, int], length: int) -> Tuple[int, int]:
    s, e = stage
    if length >= e - s + 1:
        return s, e
    return s, s + length - 1


def subwindow_late(stage: Tuple[int, int], length: int) -> Tuple[int, int]:
    s, e = stage
    if length >= e - s + 1:
        return s, e
    return e - length + 1, e


def build_variants() -> List[Tuple[str, str, str, int, int]]:
    """Return (variant_name, base_transition, variant_type, start, end)."""
    variants = []
    for t, (prev_s, next_s) in TRANSITIONS.items():
        ts, te = WINDOWS[t]
        length = te - ts + 1
        pad = max(3, int(math.ceil(length / 2.0)))
        # original and transition variants
        for vtype, s, e in [
            ("transition_original", ts, te),
            ("transition_expand_backward", ts - pad, te),
            ("transition_expand_forward", ts, te + pad),
            ("transition_expand_symmetric", ts - pad, te + pad),
            ("transition_shift_left_same_length", ts - pad, te - pad),
            ("transition_shift_right_same_length", ts + pad, te + pad),
        ]:
            s, e = clamp_window(s, e)
            variants.append((f"{t}_{vtype}_{s:03d}_{e:03d}", t, vtype, s, e))
        # adjacent stage equal length fixed windows
        for side, stage_name in [("prev", prev_s), ("next", next_s)]:
            stage = WINDOWS[stage_name]
            for pos, func in [("early", subwindow_early), ("center", subwindow_center), ("late", subwindow_late)]:
                s, e = func(stage, length)
                variants.append((f"{t}_{side}_{stage_name}_equal{length}_{pos}_{s:03d}_{e:03d}", t,
                                 f"equal_length_{side}_{pos}", s, e))
    return variants


def random_subwindows(stage: Tuple[int, int], length: int, n_random: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    s, e = stage
    total = e - s + 1
    if length >= total:
        return [(s, e)]
    starts = np.arange(s, e - length + 2)
    if len(starts) == 0:
        return [(s, e)]
    if n_random <= len(starts):
        chosen = rng.choice(starts, size=n_random, replace=False)
    else:
        chosen = rng.choice(starts, size=n_random, replace=True)
    return [(int(st), int(st + length - 1)) for st in chosen]

# -----------------------------
# Existing counts
# -----------------------------


def existing_counts_from_stability(stability: pd.DataFrame, cm: StabilityCols) -> Dict[str, Tuple[int, int]]:
    out = {}
    for w, g in stability.groupby(cm.window):
        lead = int(g.apply(lambda row: is_lead_lag_yes_row(row, cm), axis=1).sum())
        st = int(g.apply(lambda row: is_strict_or_tau0(row, cm), axis=1).sum())
        out[str(w)] = (lead, st)
    return out


def build_pairs(stability: pd.DataFrame, cm: StabilityCols, index_df: pd.DataFrame, max_pairs: Optional[int]) -> List[Tuple[str, str]]:
    pairs = []
    index_cols = set(index_df.columns) - {"year", "day"}
    for _, row in stability[[cm.source, cm.target]].drop_duplicates().iterrows():
        s = str(row[cm.source])
        t = str(row[cm.target])
        if s in index_cols and t in index_cols:
            pairs.append((s, t))
    pairs = sorted(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError("No usable source/target pairs found in index table. Check stability/index column names.")
    return pairs

# -----------------------------
# Main pipeline
# -----------------------------


def main() -> int:
    settings = parse_args()
    dirs = ensure_dirs(settings.output_dir)
    log_lines = []
    log_lines.append("# V1 transition-window sensitivity audit A")
    log_lines.append("")

    if not settings.stability_csv.exists():
        raise FileNotFoundError(f"Cannot find stability CSV: {settings.stability_csv}")
    if not settings.index_csv.exists():
        raise FileNotFoundError(
            f"Cannot find strict smooth5 index CSV: {settings.index_csv}\n"
            "Pass --index-csv explicitly if the smooth5 path is different."
        )

    stability = pd.read_csv(settings.stability_csv)
    cm = infer_stability_cols(stability)
    index_df = load_index_table(settings.index_csv)
    pairs = build_pairs(stability, cm, index_df, settings.max_pairs)
    lookup = make_lookup(index_df, sorted({x for p in pairs for x in p}))
    existing_counts = existing_counts_from_stability(stability, cm)

    # Performance hotfix: many random subwindow draws repeat the same (start, end)
    # window. Cache full pair diagnostics per unique window so duplicate draws
    # reuse the same computation while preserving the sampled distribution counts.
    window_stats_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

    def get_window_pair_stats(start: int, end: int) -> pd.DataFrame:
        key = (int(start), int(end))
        cached = window_stats_cache.get(key)
        if cached is None:
            cached = compute_pair_stats_for_window(lookup, pairs, key[0], key[1], settings)
            window_stats_cache[key] = cached
        return cached.copy()

    log_lines.append(f"- stability_csv: {settings.stability_csv}")
    log_lines.append(f"- index_csv: {settings.index_csv}")
    log_lines.append(f"- n_pairs: {len(pairs)}")
    log_lines.append(f"- n_random: {settings.n_random}")
    if settings.debug_fast:
        log_lines.append("- debug_fast: true")

    # Run deterministic variants
    variant_rows: List[VariantResult] = []
    per_pair_frames: List[pd.DataFrame] = []
    variants = build_variants()
    for variant_name, base_t, vtype, start, end in variants:
        dfv = get_window_pair_stats(start, end)
        dfv.insert(0, "variant_name", variant_name)
        dfv.insert(1, "base_transition", base_t)
        dfv.insert(2, "variant_type", vtype)
        per_pair_frames.append(dfv)
        # attach existing counts only for exact original window name
        ex = existing_counts.get(base_t) if vtype == "transition_original" else None
        variant_rows.append(summarize_variant(dfv, variant_name, base_t, vtype, start, end, ex))

    variant_summary = pd.DataFrame([asdict(x) for x in variant_rows])
    detail = pd.concat(per_pair_frames, ignore_index=True) if per_pair_frames else pd.DataFrame()
    detail.to_csv(dirs["tables"] / "window_variant_pair_diagnostics.csv", index=False)

    # Split main summaries
    equal_len = variant_summary[variant_summary["variant_type"].str.startswith("equal_length_")].copy()
    expansion = variant_summary[variant_summary["variant_type"].str.startswith("transition_")].copy()
    equal_len.to_csv(dirs["tables"] / "equal_length_window_contrast.csv", index=False)
    expansion.to_csv(dirs["tables"] / "transition_window_expansion_sensitivity.csv", index=False)

    # Null threshold summary for variants
    variant_summary.to_csv(dirs["tables"] / "window_variant_null_threshold_summary.csv", index=False)

    # Random subwindow nulls from adjacent stages
    rng = np.random.default_rng(settings.random_seed)
    random_rows = []
    for t, (prev_s, next_s) in TRANSITIONS.items():
        ts, te = WINDOWS[t]
        length = te - ts + 1
        # transition original summary as benchmark
        trans_row = expansion[(expansion["base_transition"] == t) & (expansion["variant_type"] == "transition_original")]
        transition_count = int(trans_row["lead_lag_like"].iloc[0]) if not trans_row.empty else np.nan
        for side, stage_name in [("prev", prev_s), ("next", next_s)]:
            wins = random_subwindows(WINDOWS[stage_name], length, settings.n_random, rng)
            counts = []
            strict_counts = []
            margins = []
            nulls = []
            effns = []
            for i, (s, e) in enumerate(wins):
                dfv = get_window_pair_stats(s, e)
                counts.append(int(dfv["lead_lag_like"].sum()))
                strict_counts.append(int(dfv["strict_lag_like"].sum()))
                margins.append(float(pd.to_numeric(dfv["observed_minus_max_null"], errors="coerce").median()))
                nulls.append(float(pd.to_numeric(dfv["max_lag_abs_p95_approx"], errors="coerce").median()))
                effns.append(float(pd.to_numeric(dfv["pair_effective_n"], errors="coerce").median()))
            arr = np.array(counts, dtype=float)
            if arr.size:
                percentile = float((arr <= transition_count).mean() * 100.0) if np.isfinite(transition_count) else np.nan
                random_rows.append({
                    "transition_window": t,
                    "adjacent_stage": stage_name,
                    "side": side,
                    "subwindow_length": length,
                    "n_random": len(counts),
                    "transition_lead_lag_like": transition_count,
                    "random_mean_lead_lag_like": float(np.mean(arr)),
                    "random_p05_lead_lag_like": float(np.percentile(arr, 5)),
                    "random_p50_lead_lag_like": float(np.percentile(arr, 50)),
                    "random_p95_lead_lag_like": float(np.percentile(arr, 95)),
                    "transition_percentile_vs_random": percentile,
                    "random_mean_strict_lag_like": float(np.mean(strict_counts)),
                    "random_median_observed_minus_null": float(np.nanmedian(margins)),
                    "random_median_null_p95": float(np.nanmedian(nulls)),
                    "random_median_effective_n": float(np.nanmedian(effns)),
                })
    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(dirs["tables"] / "equal_length_random_subwindow_null.csv", index=False)

    # Diagnosis rules
    diag_rows = []
    for t in TRANSITIONS:
        orig = expansion[(expansion["base_transition"] == t) & (expansion["variant_type"] == "transition_original")]
        if orig.empty:
            continue
        orig = orig.iloc[0]
        eq = equal_len[equal_len["base_transition"] == t]
        exp = expansion[(expansion["base_transition"] == t) & (expansion["variant_type"] != "transition_original")]
        rand = random_df[random_df["transition_window"] == t]
        eq_mean = float(eq["lead_lag_like"].mean()) if not eq.empty else np.nan
        exp_max = int(exp["lead_lag_like"].max()) if not exp.empty else np.nan
        orig_count = int(orig["lead_lag_like"])
        orig_null = float(orig["median_null_p95"])
        orig_margin = float(orig["median_observed_minus_null"])
        # low percentile if transition count is below random p05 for both adjacent stages
        below_random_low = False
        if not rand.empty:
            below_random_low = bool((orig_count < rand["random_p05_lead_lag_like"]).all())
        # length suspected if eq_mean close to original and random percentile not low
        if np.isfinite(eq_mean) and orig_count <= eq_mean * 0.75 and below_random_low:
            robust_length_control = "supported"
        elif np.isfinite(eq_mean) and orig_count <= eq_mean * 0.9:
            robust_length_control = "mixed_or_partial"
        else:
            robust_length_control = "not_supported"
        expansion_gain = exp_max - orig_count if np.isfinite(exp_max) else np.nan
        if np.isfinite(expansion_gain) and expansion_gain >= max(5, orig_count * 0.30):
            boundary_effect = "supported"
        elif np.isfinite(expansion_gain) and expansion_gain > 0:
            boundary_effect = "mixed_or_weak"
        else:
            boundary_effect = "not_supported"
        null_pressure = "supported" if (np.isfinite(orig_null) and orig_null >= 0.22 and np.isfinite(orig_margin) and orig_margin < -0.08) else "mixed_or_not_high"
        diag_rows.append({
            "diagnosis_id": f"{t}_low_density_due_to_short_window",
            "transition_window": t,
            "support_level": "mixed_or_requires_random_control" if robust_length_control != "supported" else "not_primary_or_insufficient",
            "primary_evidence": f"orig_count={orig_count}; equal_len_mean={eq_mean:.2f}; below_adjacent_random_low={below_random_low}",
            "allowed_statement": "Window length may contribute, but compare against equal-length adjacent-stage random subwindows before treating it as the main cause.",
            "forbidden_statement": "The transition-window result is only a window-length artifact.",
        })
        diag_rows.append({
            "diagnosis_id": f"{t}_boundary_cutting_effect",
            "transition_window": t,
            "support_level": boundary_effect,
            "primary_evidence": f"orig_count={orig_count}; max_expanded_count={exp_max}; expansion_gain={expansion_gain}",
            "allowed_statement": "If expansion restores relationships, the original transition boundary is sensitivity-relevant.",
            "forbidden_statement": "Boundary sensitivity alone proves the original window is invalid.",
        })
        diag_rows.append({
            "diagnosis_id": f"{t}_robust_low_density_after_length_control",
            "transition_window": t,
            "support_level": robust_length_control,
            "primary_evidence": f"orig_count={orig_count}; equal_len_mean={eq_mean:.2f}; below_adjacent_random_low={below_random_low}; expansion_gain={expansion_gain}",
            "allowed_statement": "A transition window can be treated as robustly low-density only if it remains low against equal-length and expansion controls.",
            "forbidden_statement": "Treating low density as a robust fact without length/boundary controls.",
        })
        diag_rows.append({
            "diagnosis_id": f"{t}_null_threshold_pressure",
            "transition_window": t,
            "support_level": null_pressure,
            "primary_evidence": f"median_null_p95={orig_null:.3f}; median_observed_minus_null={orig_margin:.3f}",
            "allowed_statement": "Null threshold pressure should be considered when interpreting low counts.",
            "forbidden_statement": "Low counts imply no observed lag peaks.",
        })
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(dirs["tables"] / "transition_window_sensitivity_diagnosis.csv", index=False)

    # Window registry
    reg_rows = []
    for name, (s, e) in WINDOWS.items():
        reg_rows.append({"window": name, "day_start": s, "day_end": e, "length": e - s + 1})
    pd.DataFrame(reg_rows).to_csv(dirs["summary"] / "window_registry_used.csv", index=False)

    summary = {
        "status": "success",
        "audit": "transition_window_sensitivity_audit_a",
        "n_pairs": len(pairs),
        "n_variants": int(len(variant_summary)),
        "n_random_rows": int(len(random_df)),
        "n_unique_windows_computed": int(len(window_stats_cache)),
        "transition_original_counts": expansion[expansion["variant_type"] == "transition_original"][
            ["base_transition", "lead_lag_like", "strict_lag_like", "tau0_coupled_like", "median_null_p95", "median_observed_minus_null"]
        ].to_dict(orient="records"),
        "important_note": (
            "This audit recomputes lightweight AR(1)-effective-n lead-lag diagnostics for window variants. "
            "It is intended for sensitivity comparison, not as a replacement for the V1 main surrogate/FDR results."
        ),
    }
    write_json(dirs["summary"] / "summary.json", summary)
    run_meta = {
        "settings": {**asdict(settings), "stability_csv": str(settings.stability_csv), "index_csv": str(settings.index_csv), "output_dir": str(settings.output_dir)},
        "stability_columns": asdict(cm),
        "windows": {k: {"day_start": v[0], "day_end": v[1], "length": v[1] - v[0] + 1} for k, v in WINDOWS.items()},
        "transition_neighbors": TRANSITIONS,
        "method_note": "Approximate AR(1)-effective-n null threshold with max-over-lags correction. Used only for window sensitivity diagnostics.",
        "performance_hotfix": "Cached pair diagnostics by unique (day_start, day_end) window; duplicate random subwindow draws reuse cached computations without changing sampled counts.",
        "n_unique_windows_computed": int(len(window_stats_cache)),
    }
    write_json(dirs["summary"] / "run_meta.json", run_meta)
    (dirs["logs"] / "RUN_LOG.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[OK] V1 transition-window sensitivity audit written to: {settings.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
