"""
V8 state_coordinate_meaning_audit_v8_a

Purpose
-------
Audit whether the V8 pre/post state curves S_A(t) can be interpreted as
object-internal post-likeness only, as a shared seasonal-process diagnostic,
or as a calibrated cross-object state-progress coordinate.

This module does NOT modify peak_only_v8_a or state_relation_v8_a outputs.
It reads existing outputs and writes an interpretation-gate layer.

Strict exclusions
-----------------
- no growth diagnostics
- no process_a classification
- no rollback / multi-pulse / non-monotonic mechanism interpretation
- no reclassification of state_relation_v8_a segments or blocks
- no upgrade of raw ΔS to scientific lead/lag language without gate status
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "state_coordinate_meaning_audit_v8_a"
OUTPUT_TAG = "state_coordinate_meaning_audit_v8_a"
STATE_RELATION_TAG = "state_relation_v8_a"
PEAK_ONLY_TAG = "peak_only_v8_a"
WINDOWS = ["W045"]
BRANCHES = ["dist", "pattern"]
Q_LEVELS = [0.25, 0.50, 0.75]
ALT_Q_LEVELS = [0.20, 0.40, 0.60, 0.80]
MIN_HOLD_DAYS_FOR_Q = 3
PEAK_ALIGNMENT_HALF_WINDOW = 5
MIN_COMMON_DAYS = 20
RANDOM_SEED = 42
PERM_N_DEFAULT = 500

OBJECTS_DEFAULT = ["P", "V", "H", "Je", "Jw"]


@dataclass
class AuditConfig:
    windows: List[str]
    branches: List[str]
    q_levels: List[float]
    alt_q_levels: List[float]
    min_hold_days_for_q: int
    peak_alignment_half_window: int
    min_common_days: int
    perm_n: int
    random_seed: int


# -----------------------------
# Generic helpers
# -----------------------------

def _read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(str(path))
    return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _finite_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _iqr(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))


def _nan_corr(x: Sequence[float], y: Sequence[float], method: str = "pearson") -> float:
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    if method == "spearman":
        return float(x[mask].rank().corr(y[mask].rank()))
    return float(x[mask].corr(y[mask]))


def _nan_rmse(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean((x[mask] - y[mask]) ** 2)))


def _linear_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Slope in y ~ a + b*x using finite points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    vx = float(np.var(xv))
    if vx <= 0:
        return float("nan")
    return float(np.cov(xv, yv, ddof=0)[0, 1] / vx)


def _support_from_perm_positive(obs: float, perm_values: np.ndarray) -> float:
    """One-sided support that observed positive association exceeds shuffled null."""
    if not np.isfinite(obs) or len(perm_values) == 0:
        return float("nan")
    perm_values = np.asarray(perm_values, dtype=float)
    perm_values = perm_values[np.isfinite(perm_values)]
    if perm_values.size == 0:
        return float("nan")
    return float(np.mean(obs > perm_values))


def _status_from_support(support: float, supported: float = 0.95, tendency: float = 0.90) -> str:
    if not np.isfinite(support):
        return "not_evaluable"
    if support >= supported:
        return "supported"
    if support >= tendency:
        return "tendency"
    return "not_supported"


def _status_rank_validity(row: pd.Series) -> str:
    """Conservative coordinate health flag. Continuous metrics are the primary output.

    The labels are audit hints, not physical decisions.
    """
    n_finite = row.get("n_finite_days", np.nan)
    dynamic_range = row.get("S_dynamic_range", np.nan)
    overshoot = row.get("overshoot_total_fraction", np.nan)
    repro_q975 = row.get("self_repro_abs_mean_q975", np.nan)
    # Minimal structural failures.
    if not np.isfinite(n_finite) or n_finite < MIN_COMMON_DAYS:
        return "poor_object_internal_coordinate"
    if not np.isfinite(dynamic_range) or dynamic_range <= 0:
        return "poor_object_internal_coordinate"
    if np.isfinite(repro_q975) and repro_q975 >= dynamic_range:
        return "poor_object_internal_coordinate"
    caution_reasons = []
    if np.isfinite(overshoot) and overshoot > 0:
        caution_reasons.append("overshoot_present")
    if np.isfinite(repro_q975) and np.isfinite(dynamic_range) and repro_q975 >= 0.5 * dynamic_range:
        caution_reasons.append("wide_self_reproducibility")
    if row.get("pre_reference_nan_affected", False) or row.get("post_reference_nan_affected", False):
        caution_reasons.append("reference_nan_affected")
    if caution_reasons:
        return "caution_object_internal_coordinate"
    return "valid_object_internal_coordinate"


def _interpret_common_process(row: pd.Series) -> str:
    loo_sup = row.get("LOO_spearman_perm_support", np.nan)
    loo_s = row.get("LOO_spearman", np.nan)
    pc1_ev = row.get("PC1_explained_variance", np.nan)
    pc1_corr = row.get("object_pc1_corr", np.nan)
    # Support-based first; PC1 is complementary. We avoid using correlation magnitudes as hard scientific thresholds.
    if np.isfinite(loo_sup) and loo_sup >= 0.95 and np.isfinite(loo_s) and loo_s > 0:
        return "common_process_consistent"
    if np.isfinite(pc1_ev) and np.isfinite(pc1_corr) and pc1_corr > 0 and np.isfinite(loo_s) and loo_s > 0:
        return "partial_common_process"
    if np.isfinite(loo_s) and loo_s <= 0:
        return "object_specific_or_opposite_to_common_process"
    return "unclear_common_process"


def _days_to_string(days: Sequence[float]) -> str:
    out = []
    for d in days:
        if pd.isna(d):
            out.append("NA")
        else:
            out.append(str(int(d)))
    return ";".join(out)


# -----------------------------
# Data reshaping
# -----------------------------

def _long_state_curves(obj_state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if obj_state.empty:
        return pd.DataFrame()
    for branch, col in [("dist", "S_dist"), ("pattern", "S_pattern")]:
        if col not in obj_state.columns:
            continue
        tmp = obj_state[["window_id", "object", "baseline_config", "day", col]].copy()
        tmp = tmp.rename(columns={col: "S"})
        tmp["branch"] = branch
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    long["S"] = pd.to_numeric(long["S"], errors="coerce")
    long["day"] = pd.to_numeric(long["day"], errors="coerce")
    return long


def _state_series_map(long_state: pd.DataFrame, window: str) -> Dict[Tuple[str, str, str], pd.Series]:
    d = {}
    if long_state.empty:
        return d
    sub = long_state[long_state["window_id"].astype(str) == str(window)]
    for (obj, base, branch), g in sub.groupby(["object", "baseline_config", "branch"]):
        gg = g.sort_values("day")
        d[(str(obj), str(base), str(branch))] = pd.Series(gg["S"].to_numpy(dtype=float), index=gg["day"].astype(int).to_numpy())
    return d


def _get_series(series_map: Dict[Tuple[str, str, str], pd.Series], obj: str, base: str, branch: str) -> pd.Series:
    return series_map.get((obj, base, branch), pd.Series(dtype=float))


# -----------------------------
# Audit layer 1: object-internal coordinate validity
# -----------------------------

def build_object_state_coordinate_validity(
    long_state: pd.DataFrame,
    state_valid_day: pd.DataFrame,
    ref_validity: pd.DataFrame,
    null_seg_summary: pd.DataFrame,
    window: str,
) -> pd.DataFrame:
    rows = []
    if long_state.empty:
        return pd.DataFrame()

    # Reproducibility summarized from segment-level nulls.
    repro = pd.DataFrame()
    if not null_seg_summary.empty and "null_object" in null_seg_summary.columns:
        repro = (
            null_seg_summary.groupby(["null_object", "baseline_config", "branch"], dropna=False)
            .agg(
                self_repro_abs_mean_median=("null_median", "median"),
                self_repro_abs_mean_q025=("null_q025", "median"),
                self_repro_abs_mean_q975=("null_q975", "median"),
                self_repro_null_n=("null_n", "median"),
            )
            .reset_index()
            .rename(columns={"null_object": "object"})
        )

    # Reference effective days: profile reference audit has one input_profile branch, not dist/pattern.
    ref = pd.DataFrame()
    if not ref_validity.empty:
        rr = ref_validity.copy()
        rr = rr[rr.get("window_id", "").astype(str) == str(window)] if "window_id" in rr.columns else rr
        if "range_type" in rr.columns:
            pre = rr[rr["range_type"].astype(str).str.contains("pre_reference", na=False)].copy()
            post = rr[rr["range_type"].astype(str).str.contains("post_reference", na=False)].copy()
            pre = pre[["object", "baseline_config", "n_finite_days", "leading_nan_days", "trailing_nan_days", "internal_nan_days"]].rename(
                columns={
                    "n_finite_days": "pre_reference_effective_days",
                    "leading_nan_days": "pre_reference_leading_nan_days",
                    "trailing_nan_days": "pre_reference_trailing_nan_days",
                    "internal_nan_days": "pre_reference_internal_nan_days",
                }
            )
            post = post[["object", "baseline_config", "n_finite_days", "leading_nan_days", "trailing_nan_days", "internal_nan_days"]].rename(
                columns={
                    "n_finite_days": "post_reference_effective_days",
                    "leading_nan_days": "post_reference_leading_nan_days",
                    "trailing_nan_days": "post_reference_trailing_nan_days",
                    "internal_nan_days": "post_reference_internal_nan_days",
                }
            )
            ref = pre.merge(post, on=["object", "baseline_config"], how="outer")
            ref["pre_reference_nan_affected"] = (
                ref[["pre_reference_leading_nan_days", "pre_reference_trailing_nan_days", "pre_reference_internal_nan_days"]]
                .fillna(0)
                .sum(axis=1)
                > 0
            )
            ref["post_reference_nan_affected"] = (
                ref[["post_reference_leading_nan_days", "post_reference_trailing_nan_days", "post_reference_internal_nan_days"]]
                .fillna(0)
                .sum(axis=1)
                > 0
            )

    # valid day audit keyed by object/base/branch.
    valid = pd.DataFrame()
    if not state_valid_day.empty:
        valid = state_valid_day.copy()
        if "window_id" in valid.columns:
            valid = valid[valid["window_id"].astype(str) == str(window)]
        keep = [c for c in [
            "object", "baseline_config", "branch", "n_finite_days", "finite_start_day", "finite_end_day",
            "leading_nan_days", "trailing_nan_days", "internal_nan_days", "valid_day_fraction", "valid_runs"
        ] if c in valid.columns]
        valid = valid[keep].drop_duplicates()

    sub = long_state[long_state["window_id"].astype(str) == str(window)]
    for (obj, base, branch), g in sub.groupby(["object", "baseline_config", "branch"]):
        vals = pd.to_numeric(g["S"], errors="coerce")
        finite = vals[np.isfinite(vals)]
        row = {
            "window_id": window,
            "object": obj,
            "baseline_config": base,
            "branch": branch,
            "S_min": float(finite.min()) if len(finite) else np.nan,
            "S_max": float(finite.max()) if len(finite) else np.nan,
            "S_median": float(finite.median()) if len(finite) else np.nan,
            "S_iqr": _iqr(finite.to_numpy()) if len(finite) else np.nan,
            "S_dynamic_range": float(finite.max() - finite.min()) if len(finite) else np.nan,
            "overshoot_low_fraction": float(np.mean(finite < 0)) if len(finite) else np.nan,
            "overshoot_high_fraction": float(np.mean(finite > 1)) if len(finite) else np.nan,
        }
        row["overshoot_total_fraction"] = (
            row["overshoot_low_fraction"] + row["overshoot_high_fraction"]
            if np.isfinite(row["overshoot_low_fraction"]) and np.isfinite(row["overshoot_high_fraction"])
            else np.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not valid.empty:
        out = out.merge(valid, on=["object", "baseline_config", "branch"], how="left")
    if not ref.empty:
        out = out.merge(ref, on=["object", "baseline_config"], how="left")
    if not repro.empty:
        out = out.merge(repro, on=["object", "baseline_config", "branch"], how="left")
    if not out.empty:
        out["coordinate_validity_status"] = out.apply(_status_rank_validity, axis=1)
        out["coordinate_validity_notes"] = out.apply(_coordinate_notes, axis=1)
    return out


def _coordinate_notes(row: pd.Series) -> str:
    notes = []
    if row.get("pre_reference_nan_affected", False):
        notes.append("pre_reference_boundary_nan_or_internal_nan_affected")
    if row.get("post_reference_nan_affected", False):
        notes.append("post_reference_boundary_nan_or_internal_nan_affected")
    if np.isfinite(row.get("overshoot_total_fraction", np.nan)) and row.get("overshoot_total_fraction", 0) > 0:
        notes.append("S_overshoot_present")
    if np.isfinite(row.get("self_repro_abs_mean_q975", np.nan)) and np.isfinite(row.get("S_dynamic_range", np.nan)):
        if row["S_dynamic_range"] > 0:
            notes.append(f"self_repro_q975_over_dynamic_range={row['self_repro_abs_mean_q975']/row['S_dynamic_range']:.3f}")
    if not notes:
        return "no_major_coordinate_health_warning"
    return "; ".join(notes)


# -----------------------------
# Audit layer 2: common seasonal process
# -----------------------------

def _compute_pc1_for_matrix(mat: pd.DataFrame) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Return PC1 explained variance, loading per object, correlation with PC1 score."""
    if mat.empty or mat.shape[1] < 2 or mat.shape[0] < 3:
        return float("nan"), {}, {}
    X = mat.astype(float).to_numpy()
    # Fill remaining missing with column medians after filtering common finite days, as a safeguard.
    colmed = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size:
        X[inds] = np.take(colmed, inds[1])
    # Shape-oriented common process: center and scale columns.
    X = X - np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    X = X / sd
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan"), {}, {}
    if S.size == 0:
        return float("nan"), {}, {}
    ev = (S ** 2) / np.sum(S ** 2) if np.sum(S ** 2) > 0 else np.full_like(S, np.nan)
    pc1 = U[:, 0] * S[0]
    loadings = Vt[0, :]
    # Orient loadings mostly positive, if possible.
    if np.nanmedian(loadings) < 0:
        loadings = -loadings
        pc1 = -pc1
    loading_map = {str(col): float(loadings[i]) for i, col in enumerate(mat.columns)}
    corr_map = {str(col): _nan_corr(mat[col].to_numpy(), pc1, method="pearson") for col in mat.columns}
    return float(ev[0]) if ev.size else float("nan"), loading_map, corr_map


def _permutation_support_spearman(x: pd.Series, y: pd.Series, perm_n: int, seed: int) -> float:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    xv = x[mask].astype(float).to_numpy()
    yv = y[mask].astype(float).to_numpy()
    obs = _nan_corr(xv, yv, method="spearman")
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    perms = np.empty(perm_n, dtype=float)
    for i in range(perm_n):
        yp = rng.permutation(yv)
        perms[i] = _nan_corr(xv, yp, method="spearman")
    return _support_from_perm_positive(obs, perms)


def build_common_seasonal_process_audit(
    long_state: pd.DataFrame,
    peak_main: pd.DataFrame,
    object_validity: pd.DataFrame,
    window: str,
    perm_n: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    if long_state.empty:
        return pd.DataFrame()
    sub = long_state[long_state["window_id"].astype(str) == str(window)]
    peak_days = {}
    if not peak_main.empty:
        for _, r in peak_main.iterrows():
            obj = str(r.get("object", ""))
            peak_days[obj] = _safe_float(r.get("selected_peak_day", np.nan))

    for (base, branch), g in sub.groupby(["baseline_config", "branch"]):
        pivot = g.pivot_table(index="day", columns="object", values="S", aggfunc="mean").sort_index()
        objects = [c for c in OBJECTS_DEFAULT if c in pivot.columns]
        if len(objects) < 2:
            continue
        mat = pivot[objects]
        # Use days with at least 3 finite objects for LOO median; PC1 uses common finite days.
        common_mat = mat.dropna(axis=0, how="any")
        pc1_ev, loadings, pc1_corrs = _compute_pc1_for_matrix(common_mat)
        for obj in objects:
            s = mat[obj]
            others = [o for o in objects if o != obj]
            loo = mat[others].median(axis=1, skipna=True)
            corr = _nan_corr(s, loo, method="pearson")
            spear = _nan_corr(s, loo, method="spearman")
            rmse = _nan_rmse(s, loo)
            support = _permutation_support_spearman(s, loo, perm_n=perm_n, seed=seed + hash((window, base, branch, obj)) % 100000)
            peak_day = peak_days.get(obj, np.nan)
            state_at_peak = _state_at_day(s, peak_day)
            local_change = _local_state_change(s, peak_day, PEAK_ALIGNMENT_HALF_WINDOW)
            row = {
                "window_id": window,
                "baseline_config": base,
                "branch": branch,
                "object": obj,
                "n_common_days_for_LOO": int((s.notna() & loo.notna()).sum()),
                "LOO_corr": corr,
                "LOO_spearman": spear,
                "LOO_rmse": rmse,
                "LOO_spearman_perm_support": support,
                "LOO_spearman_support_class": _status_from_support(support),
                "PC1_explained_variance": pc1_ev,
                "object_pc1_loading": loadings.get(obj, np.nan),
                "object_pc1_corr": pc1_corrs.get(obj, np.nan),
                "peak_day": peak_day,
                "state_at_peak": state_at_peak,
                "local_state_change_around_peak": local_change,
                "peak_state_alignment_status": _peak_alignment_status(state_at_peak, local_change),
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        # Merge object coordinate status for context.
        if not object_validity.empty:
            ctx = object_validity[["object", "baseline_config", "branch", "coordinate_validity_status"]].drop_duplicates()
            out = out.merge(ctx, on=["object", "baseline_config", "branch"], how="left")
        out["common_process_status"] = out.apply(_interpret_common_process, axis=1)
        out["common_process_notes"] = out.apply(_common_process_notes, axis=1)
    return out


def _state_at_day(s: pd.Series, day: float) -> float:
    if not np.isfinite(day) or s.empty:
        return float("nan")
    d = int(round(day))
    if d in s.index and np.isfinite(s.loc[d]):
        return float(s.loc[d])
    # nearest finite within ±2 days.
    candidates = []
    for dd in range(d - 2, d + 3):
        if dd in s.index and np.isfinite(s.loc[dd]):
            candidates.append((abs(dd - d), s.loc[dd]))
    if not candidates:
        return float("nan")
    candidates.sort(key=lambda x: x[0])
    return float(candidates[0][1])


def _local_state_change(s: pd.Series, day: float, half_window: int) -> float:
    if not np.isfinite(day) or s.empty:
        return float("nan")
    d = int(round(day))
    before = _state_at_day(s, d - half_window)
    after = _state_at_day(s, d + half_window)
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    return float(after - before)


def _peak_alignment_status(state_at_peak: float, local_change: float) -> str:
    if not np.isfinite(state_at_peak) or not np.isfinite(local_change):
        return "not_evaluable"
    # These are descriptive regions on the 0-1 projection, not scientific thresholds.
    if 0 <= state_at_peak <= 1 and local_change > 0:
        return "peak_near_positive_state_transition"
    if local_change > 0:
        return "positive_state_change_near_peak_with_overshoot"
    if abs(local_change) < 1e-12:
        return "flat_state_near_peak"
    return "negative_state_change_near_peak"


def _common_process_notes(row: pd.Series) -> str:
    notes = []
    if row.get("coordinate_validity_status", "").startswith("poor"):
        notes.append("object_internal_coordinate_poor")
    if row.get("LOO_spearman_support_class") == "supported":
        notes.append("LOO_common_coordinate_supported")
    elif row.get("LOO_spearman_support_class") == "tendency":
        notes.append("LOO_common_coordinate_tendency")
    else:
        notes.append("LOO_common_coordinate_not_supported")
    if np.isfinite(row.get("PC1_explained_variance", np.nan)):
        notes.append(f"PC1_EV={row['PC1_explained_variance']:.3f}")
    return "; ".join(notes)


# -----------------------------
# Audit layer 3: pairwise scale comparability
# -----------------------------

def _first_stable_reach_day(s: pd.Series, q: float, min_hold: int) -> float:
    if s.empty:
        return float("nan")
    idx = sorted([int(i) for i in s.index if np.isfinite(s.loc[i])])
    if not idx:
        return float("nan")
    vals = {int(i): float(s.loc[i]) for i in idx}
    for d in idx:
        ok = True
        for h in range(min_hold):
            dd = d + h
            if dd not in vals or not np.isfinite(vals[dd]) or vals[dd] < q:
                ok = False
                break
        if ok:
            return float(d)
    return float("nan")


def _time_to_q_consistency(deltas: List[float]) -> str:
    finite = [d for d in deltas if np.isfinite(d)]
    if len(finite) == 0:
        return "not_evaluable"
    if len(finite) < len(deltas):
        prefix = "incomplete_"
    else:
        prefix = ""
    # delta = tA - tB. Negative means A earlier.
    neg = sum(d < 0 for d in finite)
    pos = sum(d > 0 for d in finite)
    zero = sum(d == 0 for d in finite)
    if neg > 0 and pos == 0:
        return prefix + "A_earlier_or_same_across_q"
    if pos > 0 and neg == 0:
        return prefix + "B_earlier_or_same_across_q"
    if zero == len(finite):
        return prefix + "same_reach_days_across_q"
    return prefix + "mixed_q_order"


def _block_direction_summary(blocks: pd.DataFrame, block_boot: pd.DataFrame, window: str) -> pd.DataFrame:
    if blocks.empty:
        return pd.DataFrame()
    b = blocks.copy()
    if not block_boot.empty and "block_id" in block_boot.columns:
        boot_cols = ["block_id", "P_block_same_type", "support_class"]
        b = b.merge(block_boot[boot_cols].drop_duplicates("block_id"), on="block_id", how="left")
    else:
        b["P_block_same_type"] = np.nan
        b["support_class"] = "unknown"
    rows = []
    for (A, B, base, branch), g in b.groupby(["object_A", "object_B", "baseline_config", "branch"]):
        dom = []
        for _, r in g.iterrows():
            typ = str(r.get("block_type", ""))
            support = str(r.get("support_class", ""))
            p = _safe_float(r.get("P_block_same_type", np.nan))
            if "dominant" not in typ:
                continue
            direction = None
            if typ.startswith("A_dominant"):
                direction = "A_dominant"
            elif typ.startswith("B_dominant"):
                direction = "B_dominant"
            if direction:
                dom.append(f"{direction}:{int(r.get('start_day'))}-{int(r.get('end_day'))}:{support}:{p:.3f}" if np.isfinite(p) else f"{direction}:{int(r.get('start_day'))}-{int(r.get('end_day'))}:{support}")
        # Determine high-level direction from supported/tendency blocks.
        dir_counts = {"A": 0, "B": 0}
        for s in dom:
            if (":supported" in s) or (":tendency" in s):
                if s.startswith("A_dominant"):
                    dir_counts["A"] += 1
                elif s.startswith("B_dominant"):
                    dir_counts["B"] += 1
        if dir_counts["A"] > 0 and dir_counts["B"] == 0:
            summary_dir = "A_dominant_blocks_only"
        elif dir_counts["B"] > 0 and dir_counts["A"] == 0:
            summary_dir = "B_dominant_blocks_only"
        elif dir_counts["A"] > 0 and dir_counts["B"] > 0:
            summary_dir = "both_A_and_B_dominant_blocks"
        else:
            summary_dir = "no_supported_or_tendency_dominant_blocks"
        rows.append({
            "window_id": window,
            "object_A": A,
            "object_B": B,
            "baseline_config": base,
            "branch": branch,
            "raw_deltaS_block_direction_summary": "; ".join(dom) if dom else "no_dominant_blocks_observed",
            "raw_deltaS_supported_tendency_direction": summary_dir,
        })
    return pd.DataFrame(rows)


def _pair_list_from_objects(objects: Sequence[str]) -> List[Tuple[str, str]]:
    pairs = []
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            pairs.append((a, b))
    return pairs


def build_pairwise_state_scale_comparability(
    long_state: pd.DataFrame,
    object_validity: pd.DataFrame,
    common_audit: pd.DataFrame,
    blocks: pd.DataFrame,
    block_boot: pd.DataFrame,
    window: str,
) -> pd.DataFrame:
    if long_state.empty:
        return pd.DataFrame()
    series_map = _state_series_map(long_state, window)
    objects = sorted(set(long_state["object"].dropna().astype(str)))
    # Prefer default order.
    objects = [o for o in OBJECTS_DEFAULT if o in objects] + [o for o in objects if o not in OBJECTS_DEFAULT]
    baselines = sorted(set(long_state["baseline_config"].dropna().astype(str)))
    branches = sorted(set(long_state["branch"].dropna().astype(str)))
    raw_summary = _block_direction_summary(blocks, block_boot, window)

    rows = []
    for base in baselines:
        for branch in branches:
            # Common coordinate L(t) as median of all available object states.
            mat = []
            days = sorted(set().union(*[set(_get_series(series_map, o, base, branch).index) for o in objects])) if objects else []
            for d in days:
                vals = []
                for o in objects:
                    s = _get_series(series_map, o, base, branch)
                    vals.append(s.loc[d] if d in s.index else np.nan)
                mat.append(vals)
            if days:
                state_mat = pd.DataFrame(mat, index=days, columns=objects, dtype=float)
                L = state_mat.median(axis=1, skipna=True)
            else:
                state_mat = pd.DataFrame()
                L = pd.Series(dtype=float)
            for A, B in _pair_list_from_objects(objects):
                sA = _get_series(series_map, A, base, branch)
                sB = _get_series(series_map, B, base, branch)
                pair_corr = _nan_corr(sA.reindex(days) if days else sA, sB.reindex(days) if days else sB, method="pearson")
                pair_spear = _nan_corr(sA.reindex(days) if days else sA, sB.reindex(days) if days else sB, method="spearman")
                slope_A = _linear_slope(L.reindex(sA.index), sA)
                slope_B = _linear_slope(L.reindex(sB.index), sB)
                slope_ratio = float(slope_A / slope_B) if np.isfinite(slope_A) and np.isfinite(slope_B) and abs(slope_B) > 1e-12 else np.nan

                q_days_A = [_first_stable_reach_day(sA, q, MIN_HOLD_DAYS_FOR_Q) for q in Q_LEVELS]
                q_days_B = [_first_stable_reach_day(sB, q, MIN_HOLD_DAYS_FOR_Q) for q in Q_LEVELS]
                q_deltas = [a - b if np.isfinite(a) and np.isfinite(b) else np.nan for a, b in zip(q_days_A, q_days_B)]
                ttq_status = _time_to_q_consistency(q_deltas)
                # Alternate q levels for sensitivity summary.
                alt_A = [_first_stable_reach_day(sA, q, MIN_HOLD_DAYS_FOR_Q) for q in ALT_Q_LEVELS]
                alt_B = [_first_stable_reach_day(sB, q, MIN_HOLD_DAYS_FOR_Q) for q in ALT_Q_LEVELS]
                alt_delta = [a - b if np.isfinite(a) and np.isfinite(b) else np.nan for a, b in zip(alt_A, alt_B)]
                alt_status = _time_to_q_consistency(alt_delta)

                # Context statuses.
                A_val = _lookup_status(object_validity, A, base, branch, "coordinate_validity_status")
                B_val = _lookup_status(object_validity, B, base, branch, "coordinate_validity_status")
                A_common = _lookup_status(common_audit, A, base, branch, "common_process_status")
                B_common = _lookup_status(common_audit, B, base, branch, "common_process_status")

                raw_dir_summary = "not_available"
                raw_dir_status = "not_available"
                if not raw_summary.empty:
                    rr = raw_summary[
                        (raw_summary["object_A"].astype(str) == A)
                        & (raw_summary["object_B"].astype(str) == B)
                        & (raw_summary["baseline_config"].astype(str) == base)
                        & (raw_summary["branch"].astype(str) == branch)
                    ]
                    if not rr.empty:
                        raw_dir_summary = str(rr.iloc[0].get("raw_deltaS_block_direction_summary", "not_available"))
                        raw_dir_status = str(rr.iloc[0].get("raw_deltaS_supported_tendency_direction", "not_available"))
                deltaS_vs_q = _deltaS_vs_time_to_q(raw_dir_status, ttq_status)
                allowed_level, comp_status, comp_notes = _comparability_level(
                    A_val, B_val, A_common, B_common, ttq_status, alt_status, pair_spear, slope_ratio, deltaS_vs_q
                )
                row = {
                    "window_id": window,
                    "object_A": A,
                    "object_B": B,
                    "baseline_config": base,
                    "branch": branch,
                    "A_internal_coordinate_status": A_val,
                    "B_internal_coordinate_status": B_val,
                    "A_common_process_status": A_common,
                    "B_common_process_status": B_common,
                    "pair_corr": pair_corr,
                    "pair_spearman": pair_spear,
                    "mapping_slope_A": slope_A,
                    "mapping_slope_B": slope_B,
                    "mapping_slope_ratio": slope_ratio,
                    "time_to_q_consistency_status": ttq_status,
                    "alt_q_time_to_q_consistency_status": alt_status,
                    "raw_deltaS_block_direction_summary": raw_dir_summary,
                    "raw_deltaS_supported_tendency_direction": raw_dir_status,
                    "deltaS_vs_time_to_q_consistency": deltaS_vs_q,
                    "scale_comparability_status": comp_status,
                    "allowed_interpretation_level": allowed_level,
                    "comparability_notes": comp_notes,
                }
                for q, a, b, d in zip(Q_LEVELS, q_days_A, q_days_B, q_deltas):
                    qstr = str(q).replace("0.", "q")
                    row[f"tA_{qstr}"] = a
                    row[f"tB_{qstr}"] = b
                    row[f"delta_t_{qstr}"] = d
                rows.append(row)
    return pd.DataFrame(rows)


def _lookup_status(df: pd.DataFrame, obj: str, base: str, branch: str, col: str) -> str:
    if df.empty or col not in df.columns:
        return "not_available"
    m = df[(df["object"].astype(str) == obj) & (df["baseline_config"].astype(str) == base) & (df["branch"].astype(str) == branch)]
    if m.empty:
        return "not_available"
    return str(m.iloc[0].get(col, "not_available"))


def _deltaS_vs_time_to_q(raw_dir_status: str, ttq_status: str) -> str:
    if raw_dir_status == "not_available" or ttq_status == "not_evaluable":
        return "not_evaluable"
    if raw_dir_status == "no_supported_or_tendency_dominant_blocks":
        return "no_raw_dominant_to_compare"
    raw_A = raw_dir_status == "A_dominant_blocks_only"
    raw_B = raw_dir_status == "B_dominant_blocks_only"
    q_A = "A_earlier" in ttq_status
    q_B = "B_earlier" in ttq_status
    if raw_A and q_A:
        return "raw_deltaS_A_direction_consistent_with_time_to_q"
    if raw_B and q_B:
        return "raw_deltaS_B_direction_consistent_with_time_to_q"
    if raw_A and q_B:
        return "raw_deltaS_A_direction_conflicts_with_time_to_q"
    if raw_B and q_A:
        return "raw_deltaS_B_direction_conflicts_with_time_to_q"
    if "mixed" in ttq_status:
        return "time_to_q_mixed_order"
    return "unclear_consistency"


def _comparability_level(
    A_val: str,
    B_val: str,
    A_common: str,
    B_common: str,
    ttq_status: str,
    alt_status: str,
    pair_spearman: float,
    slope_ratio: float,
    deltaS_vs_q: str,
) -> Tuple[int, str, str]:
    notes = []
    if "poor" in A_val or "poor" in B_val or A_val == "not_available" or B_val == "not_available":
        return 0, "not_usable", "at_least_one_object_internal_coordinate_poor_or_missing"
    if not ("common_process_consistent" in A_common or "partial_common_process" in A_common) or not (
        "common_process_consistent" in B_common or "partial_common_process" in B_common
    ):
        return 1, "object_internal_post_likeness_only", "common_seasonal_process_not_established_for_pair"
    # Common process at least partial.
    if "mixed" in ttq_status or ttq_status == "not_evaluable" or "incomplete" in ttq_status:
        return 2, "common_process_diagnostic_raw_deltaS_not_interpretable", "time_to_q_not_consistently_ordered_or_incomplete"
    # Level 3 if time-to-q has coherent order; this allows calibrated comparison, not raw ΔS.
    if "A_earlier" in ttq_status or "B_earlier" in ttq_status or "same_reach_days" in ttq_status:
        notes.append("time_to_q_calibrated_comparison_available")
        if deltaS_vs_q.endswith("consistent_with_time_to_q") and np.isfinite(pair_spearman) and pair_spearman > 0:
            notes.append("raw_deltaS_direction_agrees_with_time_to_q_but_raw_scale_not_auto_licensed")
        elif "conflicts" in deltaS_vs_q:
            notes.append("raw_deltaS_conflicts_with_time_to_q")
        # Level 4 deliberately conservative: not granted automatically in v8_a.
        return 3, "calibrated_time_to_q_progress_comparison_allowed", "; ".join(notes)
    return 2, "common_process_diagnostic_raw_deltaS_not_interpretable", "fallback_unclear_pairwise_scale"


# -----------------------------
# Interpretation gate for existing state relation blocks
# -----------------------------

def build_state_relation_coordinate_gate(
    blocks: pd.DataFrame,
    block_boot: pd.DataFrame,
    pairwise_comp: pd.DataFrame,
    window: str,
) -> pd.DataFrame:
    if blocks.empty:
        return pd.DataFrame()
    b = blocks.copy()
    if not block_boot.empty and "block_id" in block_boot.columns:
        cols = [c for c in ["block_id", "P_block_same_type", "P_block_fragmented", "P_block_contains_opposite_dominant", "P_block_degrades_to_uncertain", "support_class"] if c in block_boot.columns]
        b = b.merge(block_boot[cols].drop_duplicates("block_id"), on="block_id", how="left")
    if not pairwise_comp.empty:
        comp_cols = [
            "object_A", "object_B", "baseline_config", "branch", "scale_comparability_status",
            "allowed_interpretation_level", "comparability_notes", "time_to_q_consistency_status",
            "deltaS_vs_time_to_q_consistency",
        ]
        b = b.merge(pairwise_comp[comp_cols].drop_duplicates(["object_A", "object_B", "baseline_config", "branch"]),
                    on=["object_A", "object_B", "baseline_config", "branch"], how="left")
    else:
        b["scale_comparability_status"] = "not_available"
        b["allowed_interpretation_level"] = 0
        b["comparability_notes"] = "pairwise_comparability_missing"
    b["allowed_statement"] = b.apply(_allowed_statement, axis=1)
    b["forbidden_statement"] = b.apply(_forbidden_statement, axis=1)
    b["coordinate_gate_notes"] = b.apply(_gate_notes, axis=1)
    return b


def _dominant_phrase(row: pd.Series) -> str:
    typ = str(row.get("block_type", ""))
    A = str(row.get("object_A", "A"))
    B = str(row.get("object_B", "B"))
    if typ.startswith("A_dominant"):
        return f"{A} is more post-like than {B} in object-specific pre/post coordinates"
    if typ.startswith("B_dominant"):
        return f"{B} is more post-like than {A} in object-specific pre/post coordinates"
    if typ.startswith("near"):
        return f"{A} and {B} form a within-relation near/low-separation diagnostic block"
    if typ.startswith("uncertain"):
        return f"{A}-{B} state relation is not reliably classifiable in this block"
    return f"{A}-{B} has a {typ} diagnostic block"


def _allowed_statement(row: pd.Series) -> str:
    level = int(row.get("allowed_interpretation_level", 0)) if pd.notna(row.get("allowed_interpretation_level", np.nan)) else 0
    phrase = _dominant_phrase(row)
    if level <= 0:
        return "not usable for interpretation; retain only as failed/blocked diagnostic row"
    if level == 1:
        return phrase + "; do not interpret as cross-object seasonal progress"
    if level == 2:
        return phrase + "; common-process diagnostic only, raw ΔS is not licensed as progress difference"
    if level == 3:
        return "calibrated time-to-q comparison may be discussed; raw ΔS remains diagnostic only unless separately licensed"
    return "raw ΔS comparison tentatively allowed; still not causal or dynamical lead-lag"


def _forbidden_statement(row: pd.Series) -> str:
    A = str(row.get("object_A", "A"))
    B = str(row.get("object_B", "B"))
    return f"Do not write: {A} leads {B}, {B} leads {A}, {A}/{B} state synchrony, catch-up, or causality based only on this raw ΔS block."


def _gate_notes(row: pd.Series) -> str:
    notes = []
    notes.append(f"scale_status={row.get('scale_comparability_status', 'NA')}")
    if pd.notna(row.get("support_class", np.nan)):
        notes.append(f"block_support={row.get('support_class')}")
    if pd.notna(row.get("deltaS_vs_time_to_q_consistency", np.nan)):
        notes.append(f"deltaS_vs_q={row.get('deltaS_vs_time_to_q_consistency')}")
    return "; ".join(notes)


# -----------------------------
# Summary
# -----------------------------

def _write_summary(
    out_md: Path,
    object_validity: pd.DataFrame,
    common_audit: pd.DataFrame,
    pairwise_comp: pd.DataFrame,
    gate: pd.DataFrame,
    input_status: Dict[str, str],
) -> None:
    lines = []
    lines.append(f"# {VERSION} summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("This audit checks whether V8 pre/post state curves can be interpreted as object-internal post-likeness, as common seasonal-process diagnostics, or as calibrated cross-object state-progress coordinates.")
    lines.append("")
    lines.append("## Input status")
    for k, v in input_status.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    if not object_validity.empty:
        lines.append("## Object-internal coordinate validity")
        counts = object_validity["coordinate_validity_status"].value_counts(dropna=False).to_dict()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if not common_audit.empty:
        lines.append("## Common seasonal process audit")
        counts = common_audit["common_process_status"].value_counts(dropna=False).to_dict()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if not pairwise_comp.empty:
        lines.append("## Pairwise scale comparability")
        counts = pairwise_comp["scale_comparability_status"].value_counts(dropna=False).to_dict()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
        lvl = pairwise_comp["allowed_interpretation_level"].value_counts(dropna=False).sort_index().to_dict()
        lines.append("Interpretation levels:")
        for k, v in lvl.items():
            lines.append(f"- level {k}: {v}")
        lines.append("")
    if not gate.empty:
        lines.append("## Existing state-relation block interpretation gate")
        if "allowed_interpretation_level" in gate.columns:
            lvl = gate["allowed_interpretation_level"].value_counts(dropna=False).sort_index().to_dict()
            for k, v in lvl.items():
                lines.append(f"- level {k}: {v} blocks")
        lines.append("")
    lines.append("## Interpretation rule")
    lines.append("Unless a pair reaches a calibrated interpretation level, raw ΔS blocks must be described only as object-specific relative post-likeness diagnostics. They must not be written as lead/lag, synchrony, catch-up, or causality.")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main runner
# -----------------------------

def run_state_coordinate_meaning_audit_v8_a(v8_root: Path | str) -> None:
    v8_root = Path(v8_root)
    outputs_root = v8_root / "outputs"
    state_root = outputs_root / STATE_RELATION_TAG
    peak_root = outputs_root / PEAK_ONLY_TAG
    out_root = outputs_root / OUTPUT_TAG
    out_root.mkdir(parents=True, exist_ok=True)

    perm_n = int(os.environ.get("V8_COORD_PERM_N", "100" if os.environ.get("V8_COORD_DEBUG") else str(PERM_N_DEFAULT)))
    cfg = AuditConfig(
        windows=WINDOWS,
        branches=BRANCHES,
        q_levels=Q_LEVELS,
        alt_q_levels=ALT_Q_LEVELS,
        min_hold_days_for_q=MIN_HOLD_DAYS_FOR_Q,
        peak_alignment_half_window=PEAK_ALIGNMENT_HALF_WINDOW,
        min_common_days=MIN_COMMON_DAYS,
        perm_n=perm_n,
        random_seed=RANDOM_SEED,
    )

    input_status: Dict[str, str] = {}
    cross_rows = []

    for window in WINDOWS:
        state_win = state_root / "per_window" / window
        peak_win = peak_root / "per_window" / window
        out_win = out_root / "per_window" / window
        out_win.mkdir(parents=True, exist_ok=True)

        paths = {
            "object_state_curves": state_win / f"object_state_curves_{window}.csv",
            "state_valid_day_audit": state_win / f"state_valid_day_audit_{window}.csv",
            "state_profile_reference_validity_audit": state_win / f"state_profile_reference_validity_audit_{window}.csv",
            "state_reproducibility_null_segment_summary": state_win / f"state_reproducibility_null_segment_summary_{window}.csv",
            "pairwise_state_relation_blocks": state_win / f"pairwise_state_relation_blocks_{window}.csv",
            "pairwise_state_block_bootstrap": state_win / f"pairwise_state_block_bootstrap_{window}.csv",
            "main_window_selection": peak_win / f"main_window_selection_{window}.csv",
            "pairwise_peak_order_test": peak_win / f"pairwise_peak_order_test_{window}.csv",
            "pairwise_synchrony_equivalence_test": peak_win / f"pairwise_synchrony_equivalence_test_{window}.csv",
        }
        for k, p in paths.items():
            input_status[f"{window}:{k}"] = "found" if p.exists() else f"missing:{p}"

        obj_state = _read_csv(paths["object_state_curves"])
        long_state = _long_state_curves(obj_state)
        state_valid = _read_csv(paths["state_valid_day_audit"])
        ref_valid = _read_csv(paths["state_profile_reference_validity_audit"])
        null_seg = _read_csv(paths["state_reproducibility_null_segment_summary"])
        blocks = _read_csv(paths["pairwise_state_relation_blocks"])
        block_boot = _read_csv(paths["pairwise_state_block_bootstrap"])
        peak_main = _read_csv(paths["main_window_selection"])

        object_validity = build_object_state_coordinate_validity(long_state, state_valid, ref_valid, null_seg, window)
        _write_csv(object_validity, out_win / f"object_state_coordinate_validity_{window}.csv")

        common_audit = build_common_seasonal_process_audit(long_state, peak_main, object_validity, window, perm_n=perm_n, seed=RANDOM_SEED)
        _write_csv(common_audit, out_win / f"common_seasonal_process_audit_{window}.csv")

        pairwise_comp = build_pairwise_state_scale_comparability(long_state, object_validity, common_audit, blocks, block_boot, window)
        _write_csv(pairwise_comp, out_win / f"pairwise_state_scale_comparability_{window}.csv")

        gate = build_state_relation_coordinate_gate(blocks, block_boot, pairwise_comp, window)
        _write_csv(gate, out_win / f"state_relation_coordinate_gate_{window}.csv")

        _write_summary(
            out_win / f"state_coordinate_meaning_summary_{window}.md",
            object_validity,
            common_audit,
            pairwise_comp,
            gate,
            input_status,
        )

        cross_rows.append({
            "window_id": window,
            "n_object_coordinate_rows": len(object_validity),
            "n_common_process_rows": len(common_audit),
            "n_pairwise_comparability_rows": len(pairwise_comp),
            "n_gated_blocks": len(gate),
        })

    _write_csv(pd.DataFrame(cross_rows), out_root / "cross_window" / "state_coordinate_meaning_audit_summary_all_windows.csv")
    meta = {
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "state_relation_input_tag": STATE_RELATION_TAG,
        "peak_only_input_tag": PEAK_ONLY_TAG,
        "config": asdict(cfg),
        "strict_exclusions": {
            "no_growth": True,
            "no_process_a": True,
            "no_state_segment_reclassification": True,
            "raw_deltaS_not_auto_interpreted_as_progress": True,
        },
        "input_status": input_status,
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run_state_coordinate_meaning_audit_v8_a(Path(__file__).resolve().parents[2])
