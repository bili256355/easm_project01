# -*- coding: utf-8 -*-
"""
V1 surrogate-failure autocorrelation audit.

Purpose
-------
Read the existing V1 stability-judged pair table and the index time-series used by
V1, then diagnose why many pairs are rejected at the positive-lag surrogate layer.
This script does NOT modify V1 outputs. It creates an independent audit output
folder under lead_lag_screen/V1/surrogate_failure_autocorr_audit_a/outputs.

Core questions
--------------
1. Are T3/T4 indices more autocorrelated than other windows?
2. Do T3/T4 pairs have lower effective sample size?
3. Are rejected positive-lag peaks weak, or are they moderate peaks that fail
   because the AR(1)-aware null threshold is high?
4. Are T3/T4 distinct from other windows in the attribution pattern?

Notes
-----
The threshold used here is an AR(1)-effective-sample-size approximation to a
max-over-lags null threshold. It is an audit diagnostic, not a replacement for
V1's original surrogate test. Existing V1 p/q columns are preserved and used in
parallel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


WINDOWS_DEFAULT: Dict[str, Tuple[int, int]] = {
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


@dataclass
class AuditSettings:
    project_root: str = r"D:\easm_project01"
    stability_csv: str = ""
    index_csv: str = ""
    output_dir: str = ""
    lag_min: int = 1
    lag_max: int = 5
    observed_weak_quantile: float = 0.25
    observed_moderate_quantile: float = 0.50
    null_high_quantile: float = 0.75
    p_threshold: float = 0.05
    q_threshold: float = 0.05
    debug_fast: bool = False


def log(msg: str) -> None:
    print(f"[surrogate_autocorr_audit] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    # utf-8-sig handles BOM from Excel/Windows CSV exports.
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def find_first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def auto_find_stability_csv(project_root: Path) -> Path:
    candidates = [
        project_root / "lead_lag_screen" / "V1" / "outputs" / "lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b" / "tables" / "lead_lag_pair_summary_stability_judged.csv",
        project_root / "lead_lag_screen" / "V1" / "outputs" / "lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b" / "lead_lag_pair_summary_stability_judged.csv",
    ]
    found = find_first_existing(candidates)
    if found:
        return found
    root = project_root / "lead_lag_screen" / "V1" / "outputs"
    matches = sorted(root.glob("**/lead_lag_pair_summary_stability_judged.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Cannot find lead_lag_pair_summary_stability_judged.csv. "
        "Pass --stability-csv explicitly."
    )


def auto_find_index_csv(project_root: Path) -> Path:
    """Backward-compatible broad index CSV discovery.

    The original version only searched a narrow V1 outputs path. Some V1
    packages store the time-series under tables/, daily_indices/, foundation/,
    or a custom run directory. This function returns the first plausible path;
    the validated picker below is preferred when required index names are known.
    """
    roots = [
        project_root / "lead_lag_screen" / "V1" / "outputs",
        project_root / "lead_lag_screen" / "V1",
        project_root / "foundation" / "V1" / "outputs",
        project_root,
    ]
    direct_candidates = []
    for base in roots:
        direct_candidates.extend([
            base / "lead_lag_screen_v1_smooth5_a" / "indices" / "index_anomalies.csv",
            base / "lead_lag_screen_v1_smooth5_a" / "tables" / "index_anomalies.csv",
            base / "lead_lag_screen_v1_smooth5_a" / "index_anomalies.csv",
            base / "lead_lag_screen_v1_smooth5_a" / "indices" / "index_values.csv",
            base / "lead_lag_screen_v1_smooth5_a" / "tables" / "index_values.csv",
            base / "lead_lag_screen_v1_smooth5_a" / "index_values.csv",
            base / "index_anomalies.csv",
            base / "index_values.csv",
        ])
    found = find_first_existing(direct_candidates)
    if found:
        return found

    patterns = [
        "**/index_anomalies.csv",
        "**/*index*anomal*.csv",
        "**/index_values*.csv",
        "**/*index*value*.csv",
        "**/*indices*.csv",
        "**/*index*.csv",
    ]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            matches = sorted(root.glob(pat))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Cannot find V1 index time-series CSV. Pass --index-csv explicitly. "
        "Expected a wide table with year/day and index columns, or a long table "
        "with index_name/value columns."
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def first_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    return None


def load_index_table(path: Path) -> pd.DataFrame:
    df = normalize_columns(read_csv_auto(path))
    year_col = first_col(df, ["year", "Year"])
    day_col = first_col(df, ["day", "doy", "day_index", "season_day", "Day"])
    if year_col is None or day_col is None:
        raise ValueError(
            f"Index table {path} must contain year and day/doy columns. "
            f"Columns: {list(df.columns)}"
        )
    if "index_name" in df.columns and "value" in df.columns:
        # Long format -> wide.
        df = df.pivot_table(index=[year_col, day_col], columns="index_name", values="value", aggfunc="mean").reset_index()
        df.columns = [str(c) for c in df.columns]
        year_col = first_col(df, [year_col, "year"])
        day_col = first_col(df, [day_col, "day", "doy"])
    if year_col != "year":
        df = df.rename(columns={year_col: "year"})
    if day_col != "day":
        df = df.rename(columns={day_col: "day"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "day"]).copy()
    df["year"] = df["year"].astype(int)
    df["day"] = df["day"].astype(int)
    for c in df.columns:
        if c not in ("year", "day"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["year", "day"]).reset_index(drop=True)


def candidate_index_csv_paths(project_root: Path) -> List[Path]:
    """Return a scored-but-unvalidated pool of possible index time-series CSVs."""
    roots = [
        project_root / "lead_lag_screen" / "V1" / "outputs",
        project_root / "lead_lag_screen" / "V1",
        project_root / "foundation" / "V1" / "outputs",
        project_root / "lead_lag_screen" / "V1_1" / "outputs",
        project_root,
    ]
    patterns = [
        "**/index_anomalies.csv",
        "**/*index*anomal*.csv",
        "**/index_values*.csv",
        "**/*index*value*.csv",
        "**/v1_1_index_values_doy_anomaly.csv",
        "**/*indices*.csv",
        "**/*index*.csv",
    ]
    out: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            for m in sorted(root.glob(pat)):
                if not m.is_file():
                    continue
                key = str(m).lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(m)
    return out


def score_index_candidate(path: Path, required_indices: Sequence[str], df: pd.DataFrame) -> Tuple[int, int, int, int]:
    cols = set(map(str, df.columns))
    req = set(map(str, required_indices))
    matched = len(req & cols)
    name = str(path).lower()
    anomaly_bonus = 2 if "anomal" in name else 0
    v1_bonus = 1 if "lead_lag_screen" in name and "v1" in name else 0
    structural_penalty = -2 if "v1_1" in name or "structural" in name else 0
    return (matched, anomaly_bonus, v1_bonus + structural_penalty, -len(cols))


def auto_find_index_csv_validated(project_root: Path, required_indices: Sequence[str]) -> Path:
    """Find the best CSV that actually contains V1 source/target index columns.

    This avoids selecting an unrelated CSV just because its filename contains
    'index'. It accepts both wide and long index tables through load_index_table().
    """
    candidates = candidate_index_csv_paths(project_root)
    scored: List[Tuple[Tuple[int, int, int, int], Path, str]] = []
    diagnostics: List[str] = []
    min_match = max(3, min(10, int(math.ceil(len(set(required_indices)) * 0.25))))
    for path in candidates:
        try:
            df = load_index_table(path)
            score = score_index_candidate(path, required_indices, df)
            matched = score[0]
            diagnostics.append(f"matched={matched:03d} cols={len(df.columns):04d} path={path}")
            if matched >= min_match:
                scored.append((score, path, ""))
        except Exception as exc:
            diagnostics.append(f"invalid path={path} reason={type(exc).__name__}: {exc}")
            continue
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        log(f"auto-selected index CSV: {best} score={scored[0][0]}")
        return best

    preview = "\n".join(diagnostics[:40])
    raise FileNotFoundError(
        "Cannot find a V1 index time-series CSV containing enough source/target "
        "indices from the stability table. Pass --index-csv explicitly.\n"
        f"Required-match minimum: {min_match}; candidates inspected: {len(candidates)}.\n"
        f"Candidate preview:\n{preview}"
    )


def family_from_index(name: str) -> str:
    s = str(name)
    for prefix in ["Jw", "Je", "P", "V", "H"]:
        if s == prefix or s.startswith(prefix + "_"):
            return prefix
    return "unknown"


def window_mask(df: pd.DataFrame, window: str) -> pd.Series:
    lo, hi = WINDOWS_DEFAULT[window]
    return (df["day"] >= lo) & (df["day"] <= hi)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return float("nan")
    x = x[mask]
    y = y[mask]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ar1_within_year(df: pd.DataFrame, index_name: str, window: str) -> Tuple[float, int, float, float]:
    lo, hi = WINDOWS_DEFAULT[window]
    xs: List[float] = []
    ys: List[float] = []
    vals: List[float] = []
    sub = df[(df["day"] >= lo) & (df["day"] <= hi)][["year", "day", index_name]].dropna()
    for _, g in sub.groupby("year", sort=False):
        g = g.sort_values("day")
        day = g["day"].to_numpy()
        val = g[index_name].to_numpy(dtype=float)
        if len(val) < 2:
            continue
        # Only adjacent within-window day pairs; do not cross gaps or years.
        adj = np.diff(day) == 1
        if np.any(adj):
            xs.extend(val[:-1][adj].tolist())
            ys.extend(val[1:][adj].tolist())
        vals.extend(val.tolist())
    ar1 = safe_corr(np.asarray(xs), np.asarray(ys)) if len(xs) >= 4 else float("nan")
    arr = np.asarray(vals, dtype=float)
    n = int(np.isfinite(arr).sum())
    variance = float(np.nanvar(arr, ddof=1)) if n > 1 else float("nan")
    missing_rate = float(1.0 - n / max(1, sub.shape[0])) if sub.shape[0] else float("nan")
    if np.isfinite(ar1):
        ar1_clip = max(min(ar1, 0.98), -0.98)
        eff_n = n * (1 - ar1_clip) / (1 + ar1_clip) if (1 + ar1_clip) > 0 else float("nan")
    else:
        eff_n = float("nan")
    return ar1, n, variance, eff_n


def compute_index_autocorr(index_df: pd.DataFrame, windows: Sequence[str], indices: Sequence[str]) -> pd.DataFrame:
    rows = []
    for w in windows:
        for idx in indices:
            if idx not in index_df.columns:
                continue
            ar1, n, var, eff = ar1_within_year(index_df, idx, w)
            # Basic ar2 from same helper shifted by two days.
            ar2 = ar_k_within_year(index_df, idx, w, k=2)
            rows.append({
                "window": w,
                "index_name": idx,
                "family": family_from_index(idx),
                "n_samples": n,
                "ar1": ar1,
                "ar2": ar2,
                "variance": var,
                "effective_n_ar1": eff,
            })
    return pd.DataFrame(rows)


def ar_k_within_year(df: pd.DataFrame, index_name: str, window: str, k: int = 2) -> float:
    lo, hi = WINDOWS_DEFAULT[window]
    xs: List[float] = []
    ys: List[float] = []
    sub = df[(df["day"] >= lo) & (df["day"] <= hi)][["year", "day", index_name]].dropna()
    for _, g in sub.groupby("year", sort=False):
        g = g.sort_values("day")
        if g.shape[0] <= k:
            continue
        by_day = dict(zip(g["day"].astype(int), g[index_name].astype(float)))
        for d, v in by_day.items():
            if d + k in by_day:
                xs.append(v)
                ys.append(by_day[d + k])
    return safe_corr(np.asarray(xs), np.asarray(ys)) if len(xs) >= 4 else float("nan")


def make_v1_status(row: pd.Series) -> str:
    tier = str(row.get("evidence_tier_prefix", "")) or str(row.get("evidence_tier", ""))
    label = str(row.get("lead_lag_label", ""))
    reason = str(row.get("failure_reason", ""))
    if "lead_lag_yes" in label or tier.startswith("Tier1") or tier.startswith("Tier2") or tier.startswith("Tier3"):
        return "lead_lag_yes"
    if reason == "positive_lag_not_supported" or tier.startswith("Tier5a"):
        return "positive_lag_not_supported"
    if tier.startswith("Tier4"):
        return "tier4_ambiguous"
    return "other_not_supported"


def ar1_pair_effective_n(n: float, ar1_x: float, ar1_y: float) -> float:
    if not np.isfinite(n) or n <= 3:
        return float("nan")
    if not np.isfinite(ar1_x):
        ar1_x = 0.0
    if not np.isfinite(ar1_y):
        ar1_y = 0.0
    prod = max(min(ar1_x * ar1_y, 0.98), -0.98)
    eff = n * (1 - prod) / (1 + prod)
    return float(max(4.0, min(n, eff)))


def max_lag_null_p95(eff_n: float, lag_count: int) -> float:
    if not np.isfinite(eff_n) or eff_n <= 4:
        return float("nan")
    # Approximate two-sided maximum over lag_count correlations under null using
    # Fisher z. P(max |r| <= t) = 0.95 => per-lag alpha.
    per_lag_cdf = 0.95 ** (1.0 / max(1, lag_count))
    alpha_two_sided = 1.0 - per_lag_cdf
    zcrit = NormalDist().inv_cdf(1.0 - alpha_two_sided / 2.0)
    thresh = math.tanh(zcrit / math.sqrt(max(1.0, eff_n - 3.0)))
    return float(thresh)


def max_lag_null_p99(eff_n: float, lag_count: int) -> float:
    if not np.isfinite(eff_n) or eff_n <= 4:
        return float("nan")
    per_lag_cdf = 0.99 ** (1.0 / max(1, lag_count))
    alpha_two_sided = 1.0 - per_lag_cdf
    zcrit = NormalDist().inv_cdf(1.0 - alpha_two_sided / 2.0)
    thresh = math.tanh(zcrit / math.sqrt(max(1.0, eff_n - 3.0)))
    return float(thresh)


def build_pair_context(stab: pd.DataFrame, ac: pd.DataFrame, settings: AuditSettings) -> pd.DataFrame:
    df = stab.copy()
    # Normalize expected column names.
    rename = {}
    for src in ["source", "source_index", "source_variable"]:
        if src in df.columns:
            rename[src] = "source_index"
            break
    for tgt in ["target", "target_index", "target_variable"]:
        if tgt in df.columns:
            rename[tgt] = "target_index"
            break
    df = df.rename(columns=rename)
    if "source_index" not in df.columns or "target_index" not in df.columns:
        raise ValueError("Stability table must contain source_variable/source_index and target_variable/target_index columns.")
    if "source_family" not in df.columns:
        df["source_family"] = df["source_index"].map(family_from_index)
    if "target_family" not in df.columns:
        df["target_family"] = df["target_index"].map(family_from_index)
    if "family_direction" not in df.columns:
        df["family_direction"] = df["source_family"].astype(str) + "→" + df["target_family"].astype(str)
    df["v1_status"] = df.apply(make_v1_status, axis=1)
    # Coerce metrics.
    for c in ["positive_peak_abs_r", "p_pos_surrogate", "q_pos_within_window", "q_pos_global", "p_pos_audit_surrogate", "q_pos_audit_within_window", "lag0_abs_r", "negative_peak_abs_r"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "positive_peak_abs_r" not in df.columns:
        raise ValueError("Stability table missing positive_peak_abs_r column.")

    ac_src = ac.rename(columns={
        "index_name": "source_index",
        "ar1": "source_ar1",
        "ar2": "source_ar2",
        "effective_n_ar1": "source_effective_n",
        "n_samples": "source_n_samples",
        "variance": "source_variance",
    })[["window", "source_index", "source_ar1", "source_ar2", "source_effective_n", "source_n_samples", "source_variance"]]
    ac_tgt = ac.rename(columns={
        "index_name": "target_index",
        "ar1": "target_ar1",
        "ar2": "target_ar2",
        "effective_n_ar1": "target_effective_n",
        "n_samples": "target_n_samples",
        "variance": "target_variance",
    })[["window", "target_index", "target_ar1", "target_ar2", "target_effective_n", "target_n_samples", "target_variance"]]
    out = df.merge(ac_src, on=["window", "source_index"], how="left").merge(ac_tgt, on=["window", "target_index"], how="left")
    out["mean_ar1"] = out[["source_ar1", "target_ar1"]].mean(axis=1)
    out["max_ar1"] = out[["source_ar1", "target_ar1"]].max(axis=1)
    # Use table n if present, otherwise min(source/target n).
    if "n_samples" in out.columns:
        n_for_pair = pd.to_numeric(out["n_samples"], errors="coerce")
    else:
        n_for_pair = out[["source_n_samples", "target_n_samples"]].min(axis=1)
    out["pair_n_proxy"] = n_for_pair
    out["pair_effective_n_proxy"] = [
        ar1_pair_effective_n(n, a, b)
        for n, a, b in zip(out["pair_n_proxy"], out["source_ar1"], out["target_ar1"])
    ]
    lag_count = max(1, settings.lag_max - settings.lag_min + 1)
    out["ar1_null_p95_abs_r"] = out["pair_effective_n_proxy"].map(lambda x: max_lag_null_p95(float(x), lag_count))
    out["ar1_null_p99_abs_r"] = out["pair_effective_n_proxy"].map(lambda x: max_lag_null_p99(float(x), lag_count))
    out["observed_minus_ar1_p95"] = out["positive_peak_abs_r"] - out["ar1_null_p95_abs_r"]
    out["observed_minus_ar1_p99"] = out["positive_peak_abs_r"] - out["ar1_null_p99_abs_r"]
    out["observed_over_ar1_p95"] = out["positive_peak_abs_r"] / out["ar1_null_p95_abs_r"].replace(0, np.nan)
    return out


def summarize_index_autocorr(ac: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (w, fam), g in ac.groupby(["window", "family"], dropna=False):
        rows.append({
            "window": w,
            "family": fam,
            "n_indices": int(g["index_name"].nunique()),
            "ar1_mean": float(g["ar1"].mean(skipna=True)),
            "ar1_median": float(g["ar1"].median(skipna=True)),
            "ar1_p75": float(g["ar1"].quantile(0.75)),
            "ar1_p90": float(g["ar1"].quantile(0.90)),
            "effective_n_mean": float(g["effective_n_ar1"].mean(skipna=True)),
            "effective_n_median": float(g["effective_n_ar1"].median(skipna=True)),
            "variance_median": float(g["variance"].median(skipna=True)),
        })
    # Add all-family summaries.
    for w, g in ac.groupby("window", dropna=False):
        rows.append({
            "window": w,
            "family": "ALL",
            "n_indices": int(g["index_name"].nunique()),
            "ar1_mean": float(g["ar1"].mean(skipna=True)),
            "ar1_median": float(g["ar1"].median(skipna=True)),
            "ar1_p75": float(g["ar1"].quantile(0.75)),
            "ar1_p90": float(g["ar1"].quantile(0.90)),
            "effective_n_mean": float(g["effective_n_ar1"].mean(skipna=True)),
            "effective_n_median": float(g["effective_n_ar1"].median(skipna=True)),
            "variance_median": float(g["variance"].median(skipna=True)),
        })
    return pd.DataFrame(rows)


def summarize_autocorr_by_status(pair_context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (w, status), g in pair_context.groupby(["window", "v1_status"], dropna=False):
        rows.append({
            "window": w,
            "v1_status": status,
            "n_pairs": int(len(g)),
            "mean_ar1_median": float(g["mean_ar1"].median(skipna=True)),
            "max_ar1_median": float(g["max_ar1"].median(skipna=True)),
            "pair_effective_n_proxy_median": float(g["pair_effective_n_proxy"].median(skipna=True)),
            "ar1_null_p95_median": float(g["ar1_null_p95_abs_r"].median(skipna=True)),
            "observed_r_median": float(g["positive_peak_abs_r"].median(skipna=True)),
            "observed_minus_p95_median": float(g["observed_minus_ar1_p95"].median(skipna=True)),
        })
    return pd.DataFrame(rows)


def assign_failure_attribution(pair_context: pd.DataFrame, settings: AuditSettings) -> pd.DataFrame:
    df = pair_context.copy()
    # Thresholds calculated globally across rows with observed positive peak values.
    obs = pd.to_numeric(df["positive_peak_abs_r"], errors="coerce")
    null95 = pd.to_numeric(df["ar1_null_p95_abs_r"], errors="coerce")
    obs_p25 = float(obs.quantile(settings.observed_weak_quantile))
    obs_p50 = float(obs.quantile(settings.observed_moderate_quantile))
    null_p75 = float(null95.quantile(settings.null_high_quantile))

    attributions = []
    for _, r in df.iterrows():
        status = r.get("v1_status")
        observed = float(r.get("positive_peak_abs_r", np.nan))
        thr = float(r.get("ar1_null_p95_abs_r", np.nan))
        obs_minus = float(r.get("observed_minus_ar1_p95", np.nan))
        p = float(r.get("p_pos_surrogate", np.nan)) if "p_pos_surrogate" in df.columns else np.nan
        q = float(r.get("q_pos_within_window", np.nan)) if "q_pos_within_window" in df.columns else np.nan
        if status != "positive_lag_not_supported":
            attr = "not_positive_lag_not_supported_baseline"
        elif np.isfinite(p) and p <= settings.p_threshold and np.isfinite(q) and q > settings.q_threshold:
            attr = "fdr_only_failure"
        elif np.isfinite(observed) and np.isfinite(thr) and observed < obs_p25 and thr > null_p75:
            attr = "both_observed_weak_and_null_high"
        elif np.isfinite(observed) and observed < obs_p25:
            attr = "observed_weak"
        elif np.isfinite(observed) and np.isfinite(thr) and observed >= obs_p50 and thr > null_p75 and obs_minus < 0:
            attr = "observed_moderate_but_null_high"
        elif np.isfinite(thr) and thr > null_p75 and obs_minus < 0:
            attr = "null_threshold_high"
        elif np.isfinite(obs_minus) and obs_minus < 0:
            attr = "does_not_exceed_ar1_null_threshold"
        else:
            attr = "uncertain_or_existing_v1_pq_failure"
        attributions.append(attr)
    df["surrogate_failure_attribution"] = attributions
    df["global_observed_weak_threshold_p25"] = obs_p25
    df["global_observed_moderate_threshold_p50"] = obs_p50
    df["global_null_high_threshold_p75"] = null_p75
    return df


def attribution_counts_by_window(attr: pd.DataFrame) -> pd.DataFrame:
    sub = attr[attr["v1_status"] == "positive_lag_not_supported"].copy()
    if sub.empty:
        return pd.DataFrame()
    tab = pd.crosstab(sub["window"], sub["surrogate_failure_attribution"])
    tab["n_positive_lag_not_supported"] = tab.sum(axis=1)
    return tab.reset_index()


def t3_t4_contrast(attr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = attr[attr["v1_status"] == "positive_lag_not_supported"].copy()
    for w, g in sub.groupby("window"):
        counts = g["surrogate_failure_attribution"].value_counts().to_dict()
        dominant = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "none"
        rows.append({
            "window": w,
            "positive_lag_not_supported_n": int(len(g)),
            "mean_ar1_median": float(g["mean_ar1"].median(skipna=True)),
            "max_ar1_median": float(g["max_ar1"].median(skipna=True)),
            "effective_n_median": float(g["pair_effective_n_proxy"].median(skipna=True)),
            "observed_r_median": float(g["positive_peak_abs_r"].median(skipna=True)),
            "ar1_null_p95_median": float(g["ar1_null_p95_abs_r"].median(skipna=True)),
            "observed_minus_p95_median": float(g["observed_minus_ar1_p95"].median(skipna=True)),
            "existing_p_surrogate_median": float(g["p_pos_surrogate"].median(skipna=True)) if "p_pos_surrogate" in g.columns else np.nan,
            "existing_q_pos_median": float(g["q_pos_within_window"].median(skipna=True)) if "q_pos_within_window" in g.columns else np.nan,
            "dominant_failure_type": dominant,
            "observed_weak_n": int(counts.get("observed_weak", 0)),
            "null_threshold_high_n": int(counts.get("null_threshold_high", 0)),
            "observed_moderate_but_null_high_n": int(counts.get("observed_moderate_but_null_high", 0)),
            "fdr_only_failure_n": int(counts.get("fdr_only_failure", 0)),
            "both_observed_weak_and_null_high_n": int(counts.get("both_observed_weak_and_null_high", 0)),
        })
    return pd.DataFrame(rows)


def build_diagnosis(attr: pd.DataFrame, contrast: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def win_row(w: str) -> Optional[pd.Series]:
        g = contrast[contrast["window"] == w]
        return None if g.empty else g.iloc[0]

    # Use median of non-T3/T4 windows as comparison.
    base = contrast[~contrast["window"].isin(["T3", "T4"])]
    base_null = float(base["ar1_null_p95_median"].median(skipna=True)) if not base.empty else np.nan
    base_ar1 = float(base["mean_ar1_median"].median(skipna=True)) if not base.empty else np.nan
    base_margin = float(base["observed_minus_p95_median"].median(skipna=True)) if not base.empty else np.nan

    for w in ["T3", "T4"]:
        r = win_row(w)
        if r is None:
            continue
        ar1_high = np.isfinite(base_ar1) and r["mean_ar1_median"] > base_ar1
        null_high = np.isfinite(base_null) and r["ar1_null_p95_median"] > base_null
        margin_low = np.isfinite(base_margin) and r["observed_minus_p95_median"] < base_margin
        if ar1_high and null_high and margin_low:
            support = "supported"
        elif null_high or margin_low:
            support = "partially_supported"
        else:
            support = "not_supported"
        rows.append({
            "diagnosis_id": f"{w.lower()}_surrogate_fail_due_to_high_autocorr_null",
            "support_level": support,
            "primary_evidence": (
                f"{w}: mean_ar1_median={r['mean_ar1_median']:.3f}, "
                f"ar1_null_p95_median={r['ar1_null_p95_median']:.3f}, "
                f"observed_minus_p95_median={r['observed_minus_p95_median']:.3f}; "
                f"non-T3/T4 medians: ar1={base_ar1:.3f}, null95={base_null:.3f}, margin={base_margin:.3f}"
            ),
            "counter_evidence": "AR1/null threshold audit is approximate and should not replace V1's original surrogate p-values.",
            "allowed_statement": f"{w} surrogate failures are {'at least partly' if support != 'supported' else ''} consistent with a stronger AR(1)-null / lower effective-n background.",
            "forbidden_statement": f"Do not claim all {w} rejected pairs are physically absent solely from surrogate failure.",
        })

    # Overall whether observed peaks are weak.
    for w in ["T3", "T4"]:
        r = win_row(w)
        if r is None:
            continue
        weak_share = (r.get("observed_weak_n", 0) + r.get("both_observed_weak_and_null_high_n", 0)) / max(1, r["positive_lag_not_supported_n"])
        support = "supported" if weak_share > 0.5 else "mixed_or_not_primary"
        rows.append({
            "diagnosis_id": f"{w.lower()}_observed_lag_peak_is_weak",
            "support_level": support,
            "primary_evidence": f"{w}: weak-related attribution share={weak_share:.3f}; observed_r_median={r['observed_r_median']:.3f}",
            "counter_evidence": "Observed r distribution should be interpreted relative to null thresholds, not alone.",
            "allowed_statement": f"{w} has a measurable weak-observed component among rejected pairs.",
            "forbidden_statement": f"Do not reduce {w} surrogate failure to weak observed correlations unless this share dominates.",
        })

    rows.append({
        "diagnosis_id": "surrogate_failure_explains_lead_lag_yes_drop",
        "support_level": "diagnostic_layer_only",
        "primary_evidence": "This audit decomposes positive_lag_not_supported pairs using AR1/effective-n/null-threshold context and existing V1 p/q columns.",
        "counter_evidence": "It does not rerun the full V1 lead-lag pipeline and does not change V1 classifications.",
        "allowed_statement": "Use this audit to explain where positive-lag candidates fail: observed weakness vs AR1-null threshold vs FDR-only failure.",
        "forbidden_statement": "Do not use this audit to replace V1's original surrogate classifications.",
    })
    return pd.DataFrame(rows)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 surrogate-failure autocorrelation audit")
    parser.add_argument("--project-root", default=r"D:\easm_project01")
    parser.add_argument("--stability-csv", default="")
    parser.add_argument("--index-csv", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--debug-fast", action="store_true", help="Run only a small subset for smoke-style debugging")
    args = parser.parse_args()

    settings = AuditSettings(
        project_root=args.project_root,
        stability_csv=args.stability_csv,
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        debug_fast=bool(args.debug_fast),
    )

    project_root = Path(settings.project_root)
    stability_csv = Path(settings.stability_csv) if settings.stability_csv else auto_find_stability_csv(project_root)
    output_dir = Path(settings.output_dir) if settings.output_dir else project_root / "lead_lag_screen" / "V1" / "surrogate_failure_autocorr_audit_a" / "outputs"
    tables_dir = output_dir / "tables"
    summary_dir = output_dir / "summary"
    logs_dir = output_dir / "logs"
    ensure_dir(tables_dir)
    ensure_dir(summary_dir)
    ensure_dir(logs_dir)

    log(f"stability_csv={stability_csv}")
    log(f"output_dir={output_dir}")

    stab = normalize_columns(read_csv_auto(stability_csv))
    if settings.debug_fast:
        # Keep all rows from T3/T4 plus a tiny random-ish head from other windows.
        stab = pd.concat([
            stab[stab["window"].isin(["T3", "T4"])],
            stab[~stab["window"].isin(["T3", "T4"])].head(200),
        ], ignore_index=True)

    # Determine windows and index set from stability table before selecting the
    # index time-series CSV, so the auto-selector can validate candidates.
    windows = [w for w in WINDOWS_DEFAULT if w in set(stab["window"].astype(str))]
    if not windows:
        windows = list(WINDOWS_DEFAULT.keys())
    src_col = "source_variable" if "source_variable" in stab.columns else "source_index"
    tgt_col = "target_variable" if "target_variable" in stab.columns else "target_index"
    required_index_names = sorted(set(stab[src_col].astype(str)).union(set(stab[tgt_col].astype(str))))

    index_csv = Path(settings.index_csv) if settings.index_csv else auto_find_index_csv_validated(project_root, required_index_names)
    log(f"index_csv={index_csv}")
    index_df = load_index_table(index_csv)

    needed_indices = [x for x in required_index_names if x in index_df.columns]
    missing_indices = sorted(set(required_index_names) - set(needed_indices))
    if not needed_indices:
        raise RuntimeError("No source/target indices from stability table were found in index time-series CSV.")
    if missing_indices:
        log(f"WARNING: {len(missing_indices)} indices from stability table are absent in index CSV. See summary/missing_indices.json")
        write_json(summary_dir / "missing_indices.json", {"missing_indices": missing_indices[:500], "n_missing": len(missing_indices)})

    log("Computing index AR(1)/effective-n by window...")
    index_ac = compute_index_autocorr(index_df, windows, needed_indices)
    index_ac.to_csv(tables_dir / "index_autocorr_by_window.csv", index=False, encoding="utf-8-sig")
    summarize_index_autocorr(index_ac).to_csv(tables_dir / "index_autocorr_summary_by_window.csv", index=False, encoding="utf-8-sig")

    log("Building pair autocorrelation context and AR1-null threshold margins...")
    pair_context = build_pair_context(stab, index_ac, settings)
    # Save focused pair context to keep file readable but complete enough.
    pair_context.to_csv(tables_dir / "pair_autocorr_context.csv", index=False, encoding="utf-8-sig")
    summarize_autocorr_by_status(pair_context).to_csv(tables_dir / "autocorr_by_v1_status_summary.csv", index=False, encoding="utf-8-sig")

    threshold_cols = [
        "window", "source_index", "target_index", "source_family", "target_family", "family_direction", "v1_status", "failure_reason", "evidence_tier", "positive_peak_lag", "positive_peak_abs_r",
        "ar1_null_p95_abs_r", "ar1_null_p99_abs_r", "observed_minus_ar1_p95", "observed_minus_ar1_p99", "observed_over_ar1_p95",
        "p_pos_surrogate", "q_pos_within_window", "q_pos_global", "source_ar1", "target_ar1", "mean_ar1", "max_ar1", "source_effective_n", "target_effective_n", "pair_effective_n_proxy",
    ]
    threshold_cols = [c for c in threshold_cols if c in pair_context.columns]
    pair_context[threshold_cols].to_csv(tables_dir / "surrogate_threshold_margin_by_pair.csv", index=False, encoding="utf-8-sig")

    log("Assigning failure attribution tags...")
    attr = assign_failure_attribution(pair_context, settings)
    attr_cols = threshold_cols + [
        "surrogate_failure_attribution", "global_observed_weak_threshold_p25", "global_observed_moderate_threshold_p50", "global_null_high_threshold_p75"
    ]
    attr_cols = [c for c in attr_cols if c in attr.columns]
    attr[attr_cols].to_csv(tables_dir / "surrogate_failure_attribution_by_pair.csv", index=False, encoding="utf-8-sig")

    counts = attribution_counts_by_window(attr)
    counts.to_csv(tables_dir / "surrogate_failure_attribution_counts_by_window.csv", index=False, encoding="utf-8-sig")
    contrast = t3_t4_contrast(attr)
    contrast.to_csv(tables_dir / "t3_t4_vs_other_window_autocorr_surrogate_contrast.csv", index=False, encoding="utf-8-sig")
    diagnosis = build_diagnosis(attr, contrast)
    diagnosis.to_csv(tables_dir / "surrogate_failure_autocorr_diagnosis_table.csv", index=False, encoding="utf-8-sig")

    # Extra family-direction summary for rejected pairs.
    rejected = attr[attr["v1_status"] == "positive_lag_not_supported"]
    fam = rejected.groupby(["window", "family_direction", "surrogate_failure_attribution"]).size().reset_index(name="n")
    fam.to_csv(tables_dir / "surrogate_failure_attribution_by_family_direction.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "audit": "surrogate_failure_autocorr_audit_a",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stability_csv": str(stability_csv),
        "index_csv": str(index_csv),
        "output_dir": str(output_dir),
        "n_stability_rows": int(len(stab)),
        "n_index_rows": int(len(index_df)),
        "n_indices_used": int(len(needed_indices)),
        "n_missing_indices": int(len(missing_indices)),
        "windows": windows,
        "lag_range": [settings.lag_min, settings.lag_max],
        "threshold_method": "AR1 effective-n approximate max-over-positive-lags p95/p99; diagnostic only, not V1 replacement",
        "key_outputs": [
            "tables/index_autocorr_by_window.csv",
            "tables/index_autocorr_summary_by_window.csv",
            "tables/pair_autocorr_context.csv",
            "tables/autocorr_by_v1_status_summary.csv",
            "tables/surrogate_threshold_margin_by_pair.csv",
            "tables/surrogate_failure_attribution_by_pair.csv",
            "tables/surrogate_failure_attribution_counts_by_window.csv",
            "tables/t3_t4_vs_other_window_autocorr_surrogate_contrast.csv",
            "tables/surrogate_failure_autocorr_diagnosis_table.csv",
        ],
    }
    write_json(summary_dir / "summary.json", summary)
    write_json(summary_dir / "run_meta.json", {"settings": asdict(settings), **summary})
    (logs_dir / "RUN_LOG.md").write_text(
        "# V1 surrogate failure autocorrelation audit\n\n"
        f"- status: success\n"
        f"- stability_csv: `{stability_csv}`\n"
        f"- index_csv: `{index_csv}`\n"
        f"- output_dir: `{output_dir}`\n"
        "\nThis audit is diagnostic. It does not replace V1's original surrogate p-values or classifications.\n",
        encoding="utf-8",
    )
    log("Done.")


if __name__ == "__main__":
    main()
