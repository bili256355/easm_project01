"""
V9.1_f bootstrap-composite MCA / bootstrap-space targeted SVD audit.

Purpose
-------
Read-only V9/V9.1 diagnostic branch.  It replays the V9 paired-year
bootstrap, constructs one multi-object composite anomaly vector X_b for each
bootstrap sample b, pairs it with the same bootstrap sample's V9 peak-order
outcome Y_b = peak_B(b) - peak_A(b), and estimates a single X-Y coupling mode
for each window/pair target.

This is NOT a single-year peak method and NOT a physical year-type classifier.
Bootstrap samples are treated as resampled composite perturbations of the
original multi-year sample, not as independent physical years.

Key outputs
-----------
* replayed bootstrap year-counts and peak/order samples;
* bootstrap-composite X matrix metadata;
* target-wise MCA/SVD mode summary;
* high/mid/low score-group peak-order diagnostics;
* permutation, mode-stability, and year-leverage audits;
* final bootstrap-space coupling evidence table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

VERSION = "v9_1_f_bootstrap_composite_mca_audit"
OUTPUT_TAG = "bootstrap_composite_mca_audit_v9_1_f"
DEFAULT_WINDOWS = ["W045", "W081", "W113", "W160"]
OBJECTS = ["P", "V", "H", "Je", "Jw"]
DEFAULT_TARGETS: Dict[str, List[Tuple[str, str]]] = {
    "W045": [("Je", "Jw"), ("P", "Jw"), ("V", "Jw")],
    "W081": [("P", "V"), ("V", "Jw"), ("H", "Jw")],
    "W113": [("V", "Je"), ("H", "Je"), ("P", "V"), ("Jw", "H"), ("Jw", "V")],
    "W160": [("V", "Je"), ("H", "Jw"), ("P", "V"), ("Jw", "V")],
}


@dataclass
class V91FConfig:
    windows: List[str] = field(default_factory=lambda: list(DEFAULT_WINDOWS))
    objects: List[str] = field(default_factory=lambda: list(OBJECTS))
    bootstrap_n: int = 1000
    debug_bootstrap_n: int = 100
    perm_n: int = 500
    debug_perm_n: int = 50
    mode_stability_n: int = 300
    debug_mode_stability_n: int = 50
    perm_batch_size: int = 128

    target_mode: str = "default"  # default or all
    score_grouping: str = "tercile"

    evidence_usable: float = 0.90
    evidence_credible: float = 0.95
    evidence_strict: float = 0.99
    mode_stability_good: float = 0.70
    mode_stability_caution: float = 0.50
    top1_extreme_fraction: float = 0.35
    top3_extreme_fraction: float = 0.60
    eps: float = 1e-12

    quantile_schemes: List[str] = field(default_factory=lambda: ["tercile", "quartile", "quintile", "decile"])
    score_gradient_bins: int = 5

    # hotfix02 interpretability/specificity audits
    enable_cross_target_null: bool = True
    enable_signflip_null: bool = True
    enable_target_specificity: bool = True
    enable_phase_composite: bool = True
    enable_pattern_summary: bool = True
    signflip_n: int = 300
    debug_signflip_n: int = 50
    specificity_margin: float = 0.05
    write_phase_composite_full: bool = True

    max_pattern_coefficients_per_target: int = 0  # 0 means all; set env to cap huge files.
    log_every_bootstrap: int = 50
    debug: bool = False

    @classmethod
    def from_env(cls) -> "V91FConfig":
        cfg = cls()
        if os.environ.get("V9_1F_DEBUG", "").strip() not in ("", "0", "false", "False"):
            cfg.debug = True
            cfg.bootstrap_n = int(os.environ.get("V9_1F_DEBUG_BOOTSTRAP_N", cfg.debug_bootstrap_n))
            cfg.perm_n = int(os.environ.get("V9_1F_DEBUG_PERM_N", cfg.debug_perm_n))
            cfg.mode_stability_n = int(os.environ.get("V9_1F_DEBUG_MODE_STABILITY_N", cfg.debug_mode_stability_n))
        if os.environ.get("V9_1F_BOOTSTRAP_N"):
            cfg.bootstrap_n = int(os.environ["V9_1F_BOOTSTRAP_N"])
        if os.environ.get("V9_1F_PERM_N"):
            cfg.perm_n = int(os.environ["V9_1F_PERM_N"])
        if os.environ.get("V9_1F_MODE_STABILITY_N"):
            cfg.mode_stability_n = int(os.environ["V9_1F_MODE_STABILITY_N"])
        if os.environ.get("V9_1F_PERM_BATCH_SIZE"):
            cfg.perm_batch_size = int(os.environ["V9_1F_PERM_BATCH_SIZE"])
        if os.environ.get("V9_1F_WINDOWS"):
            cfg.windows = [x.strip() for x in os.environ["V9_1F_WINDOWS"].replace(";", ",").split(",") if x.strip()]
        if os.environ.get("V9_1F_TARGETS"):
            cfg.target_mode = os.environ["V9_1F_TARGETS"].strip().lower()
        if os.environ.get("V9_1F_MAX_PATTERN_COEFFICIENTS"):
            cfg.max_pattern_coefficients_per_target = int(os.environ["V9_1F_MAX_PATTERN_COEFFICIENTS"])
        if os.environ.get("V9_1F_LOG_EVERY_BOOTSTRAP"):
            cfg.log_every_bootstrap = int(os.environ["V9_1F_LOG_EVERY_BOOTSTRAP"])
        if os.environ.get("V9_1F_QUANTILE_SCHEMES"):
            cfg.quantile_schemes = [x.strip().lower() for x in os.environ["V9_1F_QUANTILE_SCHEMES"].replace(";", ",").split(",") if x.strip()]
        if os.environ.get("V9_1F_SCORE_GRADIENT_BINS"):
            cfg.score_gradient_bins = max(2, int(os.environ["V9_1F_SCORE_GRADIENT_BINS"]))
        if os.environ.get("V9_1F_SIGNFLIP_N"):
            cfg.signflip_n = int(os.environ["V9_1F_SIGNFLIP_N"])
        if cfg.debug and not os.environ.get("V9_1F_SIGNFLIP_N"):
            cfg.signflip_n = int(os.environ.get("V9_1F_DEBUG_SIGNFLIP_N", cfg.debug_signflip_n))
        if os.environ.get("V9_1F_ENABLE_CROSS_TARGET_NULL"):
            cfg.enable_cross_target_null = os.environ["V9_1F_ENABLE_CROSS_TARGET_NULL"].strip() not in ("0", "false", "False")
        if os.environ.get("V9_1F_ENABLE_SIGNFLIP_NULL"):
            cfg.enable_signflip_null = os.environ["V9_1F_ENABLE_SIGNFLIP_NULL"].strip() not in ("0", "false", "False")
        if os.environ.get("V9_1F_ENABLE_TARGET_SPECIFICITY"):
            cfg.enable_target_specificity = os.environ["V9_1F_ENABLE_TARGET_SPECIFICITY"].strip() not in ("0", "false", "False")
        if os.environ.get("V9_1F_ENABLE_PHASE_COMPOSITE"):
            cfg.enable_phase_composite = os.environ["V9_1F_ENABLE_PHASE_COMPOSITE"].strip() not in ("0", "false", "False")
        if os.environ.get("V9_1F_ENABLE_PATTERN_SUMMARY"):
            cfg.enable_pattern_summary = os.environ["V9_1F_ENABLE_PATTERN_SUMMARY"].strip() not in ("0", "false", "False")
        if os.environ.get("V9_1F_WRITE_PHASE_COMPOSITE_FULL"):
            cfg.write_phase_composite_full = os.environ["V9_1F_WRITE_PHASE_COMPOSITE_FULL"].strip() not in ("0", "false", "False")
        return cfg


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _stage_root_from_v91(v91_root: Path) -> Path:
    return Path(v91_root).resolve().parent


def _import_c_module(v91_root: Path):
    src = Path(v91_root) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from stage_partition_v9_1 import bootstrap_year_influence_audit_v9_1_c as cmod
    except Exception as exc:
        raise ImportError(
            "V9.1_f requires V9.1_c helper module bootstrap_year_influence_audit_v9_1_c.py. "
            "Apply/run the V9.1_c patch first."
        ) from exc
    return cmod


def _make_c_config(cfg: V91FConfig, cmod) -> object:
    return cmod.V91CConfig(
        windows=list(cfg.windows),
        objects=list(cfg.objects),
        bootstrap_n=int(cfg.bootstrap_n),
        perm_n=0,
        ridge_alpha=1.0,
        log_every_bootstrap=int(cfg.log_every_bootstrap),
        debug=bool(cfg.debug),
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa = a[m] - np.nanmean(a[m])
    bb = b[m] - np.nanmean(b[m])
    da = float(np.sqrt(np.sum(aa * aa)))
    db = float(np.sqrt(np.sum(bb * bb)))
    if da <= 1e-12 or db <= 1e-12:
        return np.nan
    return float(np.sum(aa * bb) / (da * db))


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    vals = x[order]
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return _corr(_rankdata(np.asarray(a)[m]), _rankdata(np.asarray(b)[m]))


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    n = float(np.sqrt(np.sum(v * v)))
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def _prob_level(prob: float, cfg: V91FConfig) -> str:
    if not np.isfinite(prob):
        return "insufficient"
    if prob >= cfg.evidence_strict:
        return "strict_99"
    if prob >= cfg.evidence_credible:
        return "credible_95"
    if prob >= cfg.evidence_usable:
        return "usable_90"
    return "weak_or_none"


def _window_slice(scope, n_days: int) -> Tuple[int, int]:
    start = int(getattr(scope, "analysis_start", getattr(scope, "system_window_start", 0)))
    end = int(getattr(scope, "analysis_end", getattr(scope, "system_window_end", n_days - 1)))
    s = max(0, min(start, n_days - 1))
    e = max(0, min(end, n_days - 1))
    if e < s:
        s, e = 0, n_days - 1
    return s, e


def _targets_for_window(window_id: str, cfg: V91FConfig) -> pd.DataFrame:
    pairs: List[Tuple[str, str]]
    if cfg.target_mode == "all":
        pairs = []
        objs = list(cfg.objects)
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                pairs.append((a, b))
    else:
        pairs = DEFAULT_TARGETS.get(window_id, [])
    rows = []
    for a, b in pairs:
        rows.append({
            "window_id": window_id,
            "target_name": f"{window_id}_{a}_vs_{b}_delta_peak",
            "object_A": a,
            "object_B": b,
            "Y_definition": "delta_B_minus_A = peak_B - peak_A; positive means A earlier than B",
            "target_priority": "default_priority" if cfg.target_mode != "all" else "all_pairs",
            "target_source_reason": "V9/V9.1_e exposed order heterogeneity candidate" if cfg.target_mode != "all" else "all_pair_scan",
            "method_role": "bootstrap_composite_MCA_target_not_physical_type",
        })
    return pd.DataFrame(rows)


def _counts_pivot(year_counts: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[int], List[object]]:
    piv = year_counts.pivot_table(index="bootstrap_id", columns="year_index", values="count_in_bootstrap", aggfunc="sum", fill_value=0).sort_index()
    # Map year_index to actual year label for leverage reporting.
    year_map = year_counts.drop_duplicates("year_index").set_index("year_index")["year"].to_dict() if not year_counts.empty else {}
    year_indices = [int(c) for c in piv.columns]
    years = [year_map.get(i, i) for i in year_indices]
    return piv, piv.to_numpy(dtype=float), list(piv.index.astype(int)), years



def _raw_feature_meta_for_sub(scope, obj: str, s: int, sub: np.ndarray) -> pd.DataFrame:
    """Create feature metadata for a flattened day x profile object block."""
    n_days = int(sub.shape[1]) if sub.ndim >= 2 else 0
    profile_width = int(np.prod(sub.shape[2:])) if sub.ndim > 2 else 1
    rows: List[dict] = []
    for flat_idx in range(n_days * profile_width):
        rel_day = int(flat_idx // profile_width) if profile_width else int(flat_idx)
        prof_coord = int(flat_idx % profile_width) if profile_width else 0
        rows.append({
            "window_id": scope.window_id,
            "feature_index": -1,
            "object": obj,
            "day": int(s + rel_day),
            "profile_coord_index": prof_coord,
            "raw_flat_feature_index": int(flat_idx),
        })
    return pd.DataFrame(rows)


def _prepare_object_block_for_mca(
    raw_block: np.ndarray,
    feature_meta: pd.DataFrame,
    object_name: str,
    window_id: str,
    feature_offset: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    """Mask boundary/all-NaN and zero-variance features before MCA/SVD.

    All-NaN columns typically come from smoothed boundary days and are dropped
    explicitly before any nanmean/nanstd call. Partial-NaN columns are column-
    mean imputed after all-NaN removal. Zero-variance columns are then dropped.
    """
    raw = np.asarray(raw_block, dtype=float)
    n_boot = int(raw.shape[0]) if raw.ndim == 2 else 0
    n_raw = int(raw.shape[1]) if raw.ndim == 2 else 0
    if feature_meta is None or feature_meta.empty:
        meta_all = pd.DataFrame({
            "window_id": [window_id] * n_raw,
            "feature_index": [-1] * n_raw,
            "object": [object_name] * n_raw,
            "raw_flat_feature_index": list(range(n_raw)),
        })
    else:
        meta_all = feature_meta.copy().reset_index(drop=True)
        if len(meta_all) != n_raw:
            meta_all = pd.DataFrame({
                "window_id": [window_id] * n_raw,
                "feature_index": [-1] * n_raw,
                "object": [object_name] * n_raw,
                "raw_flat_feature_index": list(range(n_raw)),
            })
    if raw.ndim != 2 or n_raw == 0 or n_boot == 0:
        meta_all["feature_kept"] = False
        meta_all["drop_reason"] = "empty_object_block"
        audit = {
            "window_id": window_id,
            "object": object_name,
            "n_raw_features": n_raw,
            "n_all_nan_features": 0,
            "n_partial_nan_features": 0,
            "n_zero_variance_features": 0,
            "n_kept_features": 0,
            "kept_feature_fraction": np.nan,
            "partial_nan_max_fraction": np.nan,
            "all_nan_day_min": np.nan,
            "all_nan_day_max": np.nan,
            "nan_warning_status": "empty_object_block",
        }
        return np.empty((n_boot, 0)), meta_all, meta_all.iloc[0:0].copy(), audit

    finite_count = np.isfinite(raw).sum(axis=0)
    all_nan_mask = finite_count == 0
    partial_nan_mask = (finite_count > 0) & (finite_count < n_boot)
    partial_nan_fraction = np.where(n_boot > 0, 1.0 - (finite_count / float(n_boot)), np.nan)

    meta_all["finite_count"] = finite_count.astype(int)
    meta_all["partial_nan_fraction"] = partial_nan_fraction
    meta_all["feature_kept"] = False
    meta_all["drop_reason"] = np.where(all_nan_mask, "all_nan_boundary_feature", "pending")
    meta_all["standardization_mean"] = np.nan
    meta_all["standardization_std"] = np.nan
    meta_all["block_weight"] = np.nan

    non_all_idx = np.where(~all_nan_mask)[0]
    if non_all_idx.size == 0:
        days = meta_all.loc[all_nan_mask, "day"] if "day" in meta_all.columns else pd.Series(dtype=float)
        audit = {
            "window_id": window_id,
            "object": object_name,
            "n_raw_features": n_raw,
            "n_all_nan_features": int(all_nan_mask.sum()),
            "n_partial_nan_features": 0,
            "n_zero_variance_features": 0,
            "n_kept_features": 0,
            "kept_feature_fraction": 0.0,
            "partial_nan_max_fraction": np.nan,
            "all_nan_day_min": float(days.min()) if len(days) else np.nan,
            "all_nan_day_max": float(days.max()) if len(days) else np.nan,
            "nan_warning_status": "all_features_all_nan_dropped",
        }
        return np.empty((n_boot, 0)), meta_all, meta_all.iloc[0:0].copy(), audit

    block1 = raw[:, non_all_idx].copy()
    # Safe: all all-NaN columns have already been removed.
    col_mean1 = np.nanmean(block1, axis=0)
    nan_pos = ~np.isfinite(block1)
    if nan_pos.any():
        _, jj = np.where(nan_pos)
        block1[nan_pos] = col_mean1[jj]
    col_std1 = np.std(block1, axis=0)
    zero_var_local = (~np.isfinite(col_std1)) | (col_std1 <= eps)
    keep_local = ~zero_var_local
    kept_raw_idx = non_all_idx[keep_local]
    zero_raw_idx = non_all_idx[zero_var_local]

    # Mark drop reasons for non-kept non-all-NaN columns.
    if zero_raw_idx.size:
        meta_all.loc[zero_raw_idx, "drop_reason"] = "zero_variance_feature"
    pending_idx = non_all_idx[~zero_var_local]
    if pending_idx.size:
        meta_all.loc[pending_idx, "drop_reason"] = np.where(partial_nan_mask[pending_idx], "partial_nan_imputed", "kept")

    if kept_raw_idx.size == 0:
        days = meta_all.loc[all_nan_mask, "day"] if "day" in meta_all.columns else pd.Series(dtype=float)
        audit = {
            "window_id": window_id,
            "object": object_name,
            "n_raw_features": n_raw,
            "n_all_nan_features": int(all_nan_mask.sum()),
            "n_partial_nan_features": int(partial_nan_mask.sum()),
            "n_zero_variance_features": int(zero_raw_idx.size),
            "n_kept_features": 0,
            "kept_feature_fraction": 0.0,
            "partial_nan_max_fraction": float(np.nanmax(partial_nan_fraction[partial_nan_mask])) if partial_nan_mask.any() else 0.0,
            "all_nan_day_min": float(days.min()) if len(days) else np.nan,
            "all_nan_day_max": float(days.max()) if len(days) else np.nan,
            "nan_warning_status": "all_remaining_features_zero_variance_or_nan",
        }
        return np.empty((n_boot, 0)), meta_all, meta_all.iloc[0:0].copy(), audit

    block2 = block1[:, keep_local]
    mean2 = np.mean(block2, axis=0)
    std2 = np.std(block2, axis=0)
    std_safe = np.where(std2 > eps, std2, 1.0)
    clean = (block2 - mean2[None, :]) / std_safe[None, :]

    block_weight = 1.0 / math.sqrt(max(1, clean.shape[1]))
    kept_meta = meta_all.loc[kept_raw_idx].copy().reset_index(drop=True)
    kept_meta["feature_index"] = np.arange(feature_offset, feature_offset + clean.shape[1], dtype=int)
    kept_meta["feature_kept"] = True
    kept_meta["standardization_mean"] = mean2
    kept_meta["standardization_std"] = std_safe
    kept_meta["block_weight"] = block_weight
    # Put kept feature metadata back into all-feature meta.
    for local_i, raw_i in enumerate(kept_raw_idx):
        meta_all.loc[raw_i, "feature_index"] = int(feature_offset + local_i)
        meta_all.loc[raw_i, "feature_kept"] = True
        meta_all.loc[raw_i, "standardization_mean"] = float(mean2[local_i])
        meta_all.loc[raw_i, "standardization_std"] = float(std_safe[local_i])
        meta_all.loc[raw_i, "block_weight"] = block_weight

    all_nan_days = meta_all.loc[all_nan_mask, "day"] if "day" in meta_all.columns else pd.Series(dtype=float)
    statuses = []
    if all_nan_mask.any():
        statuses.append("boundary_all_nan_features_dropped")
    if partial_nan_mask.any():
        statuses.append("partial_nan_imputed")
    if zero_raw_idx.size:
        statuses.append("zero_variance_features_dropped")
    if not statuses:
        statuses.append("no_nan_issue")
    audit = {
        "window_id": window_id,
        "object": object_name,
        "n_raw_features": n_raw,
        "n_all_nan_features": int(all_nan_mask.sum()),
        "n_partial_nan_features": int(partial_nan_mask.sum()),
        "n_zero_variance_features": int(zero_raw_idx.size),
        "n_kept_features": int(clean.shape[1]),
        "kept_feature_fraction": float(clean.shape[1] / n_raw) if n_raw else np.nan,
        "partial_nan_max_fraction": float(np.nanmax(partial_nan_fraction[partial_nan_mask])) if partial_nan_mask.any() else 0.0,
        "all_nan_day_min": float(all_nan_days.min()) if len(all_nan_days) else np.nan,
        "all_nan_day_max": float(all_nan_days.max()) if len(all_nan_days) else np.nan,
        "nan_warning_status": ";".join(statuses),
    }
    return clean, meta_all, kept_meta, audit


def _build_X_matrix(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    year_counts: pd.DataFrame,
    scope,
    cfg: V91FConfig,
    out_win: Path,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build bootstrap-sample by feature X from weighted composite anomalies.

    Hotfix01: all-NaN boundary features are explicitly dropped before any
    standardization/SVD step; partial NaNs are imputed after that removal; and
    object equal-weighting is applied only after feature masking.
    """
    piv, counts, boot_ids, year_labels = _counts_pivot(year_counts)
    if counts.size == 0:
        empty = pd.DataFrame()
        return np.empty((0, 0)), empty, empty, empty, empty, empty
    weight_sum = counts.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    weights = counts / weight_sum

    block_arrays: List[np.ndarray] = []
    feature_meta_parts: List[pd.DataFrame] = []
    kept_meta_parts: List[pd.DataFrame] = []
    block_rows: List[dict] = []
    raw_rows: List[dict] = []
    nan_audit_rows: List[dict] = []
    feature_offset = 0

    for obj in cfg.objects:
        if obj not in profiles:
            continue
        prof_by_year = np.asarray(profiles[obj][0], dtype=float)
        if prof_by_year.ndim < 2:
            continue
        ny, n_days = prof_by_year.shape[0], prof_by_year.shape[1]
        if weights.shape[1] != ny:
            n = min(weights.shape[1], ny)
            w = weights[:, :n]
            prof = prof_by_year[:n]
            align_warning = f"weights/profile year mismatch; truncated to {n}"
        else:
            w = weights
            prof = prof_by_year
            align_warning = "none"
        s, e = _window_slice(scope, prof.shape[1])
        sub = prof[:, s:e + 1]
        flat_year = sub.reshape(sub.shape[0], -1)
        finite = np.isfinite(flat_year).astype(float)
        full_numer = np.nansum(flat_year, axis=0)
        full_denom = finite.sum(axis=0)
        full_flat = np.divide(full_numer, full_denom, out=np.full(flat_year.shape[1], np.nan), where=full_denom > 0)
        denom = w @ finite
        numer = w @ np.nan_to_num(flat_year, nan=0.0)
        comp = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
        raw = comp - full_flat[None, :]
        raw_meta = _raw_feature_meta_for_sub(scope, obj, s, sub)
        clean_block, meta_all, kept_meta, nan_audit = _prepare_object_block_for_mca(
            raw, raw_meta, obj, scope.window_id, feature_offset, eps=cfg.eps
        )
        feature_meta_parts.append(meta_all)
        nan_audit["alignment_warning"] = align_warning
        nan_audit_rows.append(nan_audit)
        if clean_block.shape[1] == 0:
            block_rows.append({
                "window_id": scope.window_id, "object": obj, "window_start_day": int(s), "window_end_day": int(e),
                "n_raw_features": int(raw.shape[1]), "n_kept_features": 0,
                "block_variance_before_equal_weight": np.nan, "block_variance_after_equal_weight": np.nan,
                "block_scale": np.nan, "alignment_warning": align_warning,
            })
            continue
        block_scale = 1.0 / math.sqrt(max(1, clean_block.shape[1]))
        Xeq = clean_block * block_scale
        # feature metadata already stores block_weight; keep it consistent.
        kept_meta["block_weight"] = block_scale
        feature_meta_parts[-1].loc[feature_meta_parts[-1]["feature_kept"].astype(bool), "block_weight"] = block_scale
        block_arrays.append(Xeq)
        block_var = float(np.nanvar(Xeq))
        block_rows.append({
            "window_id": scope.window_id,
            "object": obj,
            "window_start_day": int(s),
            "window_end_day": int(e),
            "n_raw_features": int(raw.shape[1]),
            "n_kept_features": int(Xeq.shape[1]),
            "block_variance_before_equal_weight": float(np.nanvar(clean_block)),
            "block_variance_after_equal_weight": block_var,
            "block_scale": block_scale,
            "alignment_warning": align_warning,
        })
        kept_meta_parts.append(kept_meta)
        raw_rows.append({
            "window_id": scope.window_id,
            "object": obj,
            "n_kept_features": int(Xeq.shape[1]),
            "note": "full X matrix stored in bootstrap_composite_X_matrix_*.npz",
        })
        feature_offset += Xeq.shape[1]

    X = np.concatenate(block_arrays, axis=1) if block_arrays else np.empty((len(boot_ids), 0))
    feature_meta = pd.concat(feature_meta_parts, ignore_index=True) if feature_meta_parts else pd.DataFrame()
    kept_meta = pd.concat(kept_meta_parts, ignore_index=True) if kept_meta_parts else pd.DataFrame()
    block_audit = pd.DataFrame(block_rows)
    raw_note = pd.DataFrame(raw_rows)
    nan_audit = pd.DataFrame(nan_audit_rows)
    boot_meta = pd.DataFrame({"window_id": scope.window_id, "bootstrap_id": boot_ids})
    np.savez_compressed(out_win / f"bootstrap_composite_X_matrix_{scope.window_id}.npz", X=X, bootstrap_id=np.asarray(boot_ids))
    return X, kept_meta, block_audit, raw_note, boot_meta, nan_audit

def _fit_mode(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray | float | str]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or X.shape[0] != Y.shape[0] or X.shape[0] < 5 or X.shape[1] == 0:
        return {"status": "invalid_shape", "u": np.zeros(X.shape[1] if X.ndim == 2 else 0), "score": np.full(Y.shape, np.nan)}
    valid = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < 5 or np.nanstd(Y[valid]) <= 1e-12:
        return {"status": "insufficient_valid_target", "u": np.zeros(X.shape[1]), "score": np.full(X.shape[0], np.nan)}
    Xc = X - np.nanmean(X[valid], axis=0, keepdims=True)
    yz = np.full_like(Y, np.nan, dtype=float)
    yz[valid] = (Y[valid] - np.nanmean(Y[valid])) / (np.nanstd(Y[valid]) or 1.0)
    c = Xc[valid].T @ yz[valid]
    u = _norm_vec(c)
    score = Xc @ u
    corr = _corr(score, yz)
    spear = _spearman(score, yz)
    cov_strength = float(np.sqrt(np.sum(c * c)))
    return {"status": "ok", "u": u, "score": score, "target_z": yz, "corr": corr, "spearman": spear, "covariance_strength": cov_strength}


def _permutation_audit(X: np.ndarray, Y: np.ndarray, obs_abs_corr: float, cfg: V91FConfig, seed: int) -> Dict[str, float | str]:
    if cfg.perm_n <= 0 or not np.isfinite(obs_abs_corr):
        return {"permutation_percentile": np.nan, "permutation_empirical_p": np.nan, "permutation_level": "insufficient"}
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=float)
    valid = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < 5:
        return {"permutation_percentile": np.nan, "permutation_empirical_p": np.nan, "permutation_level": "insufficient"}
    Xv = X[valid]
    Yv = (Y[valid] - np.nanmean(Y[valid])) / (np.nanstd(Y[valid]) or 1.0)
    Xc = Xv - np.mean(Xv, axis=0, keepdims=True)
    n = len(Yv)
    vals: List[float] = []
    batch = max(1, int(cfg.perm_batch_size))
    remaining = int(cfg.perm_n)
    while remaining > 0:
        k = min(batch, remaining)
        Yp = np.empty((n, k), dtype=float)
        for j in range(k):
            Yp[:, j] = rng.permutation(Yv)
        C = Xc.T @ Yp  # feature x k
        norms = np.sqrt(np.sum(C * C, axis=0))
        norms = np.where(norms > 1e-12, norms, 1.0)
        U = C / norms[None, :]
        S = Xc @ U
        S = S - S.mean(axis=0, keepdims=True)
        Ypc = Yp - Yp.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(S * S, axis=0) * np.sum(Ypc * Ypc, axis=0))
        corr = np.divide(np.sum(S * Ypc, axis=0), denom, out=np.full(k, np.nan), where=denom > 1e-12)
        vals.extend(np.abs(corr[np.isfinite(corr)]).tolist())
        remaining -= k
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"permutation_percentile": np.nan, "permutation_empirical_p": np.nan, "permutation_level": "insufficient"}
    pct = float(np.mean(arr <= abs(obs_abs_corr)))
    p = float(np.mean(arr >= abs(obs_abs_corr)))
    return {"permutation_percentile": pct, "permutation_empirical_p": p, "permutation_level": _prob_level(pct, cfg)}


def _mode_stability(X: np.ndarray, Y: np.ndarray, u_full: np.ndarray, cfg: V91FConfig, seed: int) -> Tuple[pd.DataFrame, Dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    rows: List[dict] = []
    if n < 5 or cfg.mode_stability_n <= 0 or u_full.size == 0:
        return pd.DataFrame(), {"mode_pattern_corr_median": np.nan, "mode_pattern_corr_q025": np.nan, "mode_pattern_corr_q975": np.nan, "mode_stability_status": "insufficient"}
    for rep in range(int(cfg.mode_stability_n)):
        idx = rng.integers(0, n, size=n)
        fit = _fit_mode(X[idx], Y[idx])
        u = np.asarray(fit.get("u", np.zeros_like(u_full)), dtype=float)
        cc = _corr(u, u_full)
        flipped = False
        if np.isfinite(cc) and cc < 0:
            cc = -cc
            flipped = True
        rows.append({"rep_id": rep, "pattern_corr_with_full": cc, "sign_flipped": flipped, "fit_status": fit.get("status", "")})
    df = pd.DataFrame(rows)
    vals = df["pattern_corr_with_full"].to_numpy(float) if not df.empty else np.array([])
    med = float(np.nanmedian(vals)) if vals.size else np.nan
    q025 = float(np.nanquantile(vals, 0.025)) if vals.size and np.isfinite(vals).any() else np.nan
    q975 = float(np.nanquantile(vals, 0.975)) if vals.size and np.isfinite(vals).any() else np.nan
    if np.isfinite(med) and med >= cfg.mode_stability_good:
        status = "stable"
    elif np.isfinite(med) and med >= cfg.mode_stability_caution:
        status = "caution"
    else:
        status = "unstable"
    return df, {"mode_pattern_corr_median": med, "mode_pattern_corr_q025": q025, "mode_pattern_corr_q975": q975, "mode_stability_status": status}


def _assign_score_groups(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    out = np.full(scores.shape, "unassigned", dtype=object)
    valid = np.where(np.isfinite(scores))[0]
    if valid.size < 3:
        return out
    order = valid[np.argsort(scores[valid], kind="mergesort")]
    n = len(order)
    n_low = n // 3
    n_mid = n // 3
    low_idx = order[:n_low]
    mid_idx = order[n_low:n_low + n_mid]
    high_idx = order[n_low + n_mid:]
    out[low_idx] = "low"
    out[mid_idx] = "mid"
    out[high_idx] = "high"
    return out


def _Y_for_pair(peak_samples: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
    a = str(target["object_A"]); b = str(target["object_B"])
    piv = peak_samples.pivot_table(index="bootstrap_id", columns="object", values="selected_peak_day", aggfunc="first").sort_index()
    boot_ids = list(piv.index.astype(int))
    if a not in piv.columns or b not in piv.columns:
        Y = np.full(len(boot_ids), np.nan)
    else:
        Y = piv[b].to_numpy(float) - piv[a].to_numpy(float)
    flags = pd.DataFrame({
        "bootstrap_id": boot_ids,
        "Y_delta_B_minus_A": Y,
        "A_earlier_flag": np.where(np.isfinite(Y), Y > 0, False),
        "B_earlier_flag": np.where(np.isfinite(Y), Y < 0, False),
        "same_day_flag": np.where(np.isfinite(Y), Y == 0, False),
    })
    return Y, boot_ids, flags


def _high_low_order(flags: pd.DataFrame, scores: np.ndarray, target: pd.Series, cfg: V91FConfig) -> pd.DataFrame:
    groups = _assign_score_groups(scores)
    df = flags.copy()
    df["score"] = scores
    df["score_group"] = groups
    rows: List[dict] = []
    a = str(target["object_A"]); b = str(target["object_B"])
    for gname in ["high", "mid", "low"]:
        g = df[df["score_group"].eq(gname)].copy()
        y = g["Y_delta_B_minus_A"].to_numpy(float) if not g.empty else np.array([])
        valid = np.isfinite(y)
        n = int(valid.sum())
        if n:
            pa = float(np.mean(y[valid] > 0)); pb = float(np.mean(y[valid] < 0)); ps = float(np.mean(y[valid] == 0))
            med = float(np.nanmedian(y[valid])); q025 = float(np.nanquantile(y[valid], 0.025)); q975 = float(np.nanquantile(y[valid], 0.975))
        else:
            pa = pb = ps = med = q025 = q975 = np.nan
        best = max(pa if np.isfinite(pa) else -1, pb if np.isfinite(pb) else -1)
        direction = f"{a}_earlier" if np.isfinite(pa) and pa >= pb else f"{b}_earlier"
        rows.append({
            "window_id": target["window_id"],
            "target_name": target["target_name"],
            "score_group": gname,
            "n_bootstrap": int(len(g)),
            "n_valid_delta": n,
            "object_A": a,
            "object_B": b,
            "P_A_earlier": pa,
            "P_B_earlier": pb,
            "P_same_day": ps,
            "dominant_order_direction": direction,
            "dominant_order_probability": best,
            "delta_median": med,
            "delta_q025": q025,
            "delta_q975": q975,
            "order_level": _prob_level(best, cfg),
        })
    return pd.DataFrame(rows)


def _dominant_from_probs(pa: float, pb: float, a: str, b: str, cfg: V91FConfig) -> Tuple[str, float, str]:
    if not np.isfinite(pa) and not np.isfinite(pb):
        return "insufficient", np.nan, "insufficient"
    if (pa if np.isfinite(pa) else -1) >= (pb if np.isfinite(pb) else -1):
        prob = pa
        direction = f"{a}_earlier"
    else:
        prob = pb
        direction = f"{b}_earlier"
    return direction, float(prob), _prob_level(float(prob), cfg)


def _order_stats_for_y(y: np.ndarray, a: str, b: str, cfg: V91FConfig) -> dict:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    n = int(valid.sum())
    if n == 0:
        pa = pb = ps = med = q025 = q975 = np.nan
    else:
        yy = y[valid]
        pa = float(np.mean(yy > 0))
        pb = float(np.mean(yy < 0))
        ps = float(np.mean(yy == 0))
        med = float(np.nanmedian(yy))
        q025 = float(np.nanquantile(yy, 0.025))
        q975 = float(np.nanquantile(yy, 0.975))
    direction, prob, level = _dominant_from_probs(pa, pb, a, b, cfg)
    return {
        "n_valid_delta": n,
        "P_A_earlier": pa,
        "P_B_earlier": pb,
        "P_same_day": ps,
        "dominant_order_direction": direction,
        "dominant_order_probability": prob,
        "delta_median": med,
        "delta_q025": q025,
        "delta_q975": q975,
        "order_level": level,
    }


def _fraction_for_scheme(scheme: str) -> float:
    scheme = str(scheme).lower()
    if scheme == "tercile":
        return 1.0 / 3.0
    if scheme == "quartile":
        return 0.25
    if scheme == "quintile":
        return 0.20
    if scheme == "decile":
        return 0.10
    try:
        val = float(scheme)
        if 0 < val < 0.5:
            return val
    except Exception:
        pass
    return 1.0 / 3.0


def _quantile_sensitivity_order_audit(flags: pd.DataFrame, scores: np.ndarray, target: pd.Series, cfg: V91FConfig) -> pd.DataFrame:
    a = str(target["object_A"]); b = str(target["object_B"])
    df = flags.copy()
    df["score"] = np.asarray(scores, dtype=float)
    valid = df[np.isfinite(df["score"].to_numpy(float)) & np.isfinite(df["Y_delta_B_minus_A"].to_numpy(float))].copy()
    rows: List[dict] = []
    if valid.empty:
        return pd.DataFrame()
    valid = valid.sort_values("score", kind="mergesort")
    n = len(valid)
    for scheme in cfg.quantile_schemes:
        frac = _fraction_for_scheme(scheme)
        k = max(1, int(math.floor(n * frac)))
        if 2 * k > n:
            k = max(1, n // 3)
        low = valid.iloc[:k]
        high = valid.iloc[-k:]
        st_h = _order_stats_for_y(high["Y_delta_B_minus_A"].to_numpy(float), a, b, cfg)
        st_l = _order_stats_for_y(low["Y_delta_B_minus_A"].to_numpy(float), a, b, cfg)
        reversal = bool(st_h["dominant_order_direction"] != st_l["dominant_order_direction"] and "insufficient" not in (st_h["dominant_order_direction"], st_l["dominant_order_direction"]))
        rows.append({
            "window_id": target["window_id"],
            "target_name": target["target_name"],
            "object_A": a,
            "object_B": b,
            "quantile_scheme": scheme,
            "fraction_each_tail": frac,
            "high_n": int(len(high)),
            "low_n": int(len(low)),
            "high_P_A_earlier": st_h["P_A_earlier"],
            "high_P_B_earlier": st_h["P_B_earlier"],
            "low_P_A_earlier": st_l["P_A_earlier"],
            "low_P_B_earlier": st_l["P_B_earlier"],
            "high_delta_median": st_h["delta_median"],
            "low_delta_median": st_l["delta_median"],
            "high_dominant_direction": st_h["dominant_order_direction"],
            "low_dominant_direction": st_l["dominant_order_direction"],
            "high_dominant_probability": st_h["dominant_order_probability"],
            "low_dominant_probability": st_l["dominant_order_probability"],
            "high_order_level": st_h["order_level"],
            "low_order_level": st_l["order_level"],
            "high_low_reversal_flag": reversal,
        })
    return pd.DataFrame(rows)


def _monotonicity_score(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    if arr.size < 3 or not np.isfinite(arr).any():
        return np.nan
    x = np.arange(arr.size, dtype=float)
    m = np.isfinite(arr)
    if m.sum() < 3:
        return np.nan
    return _spearman(x[m], arr[m])


def _slope(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    x = np.arange(arr.size, dtype=float)
    m = np.isfinite(arr)
    if m.sum() < 2:
        return np.nan
    xx = x[m] - np.mean(x[m])
    yy = arr[m] - np.mean(arr[m])
    den = float(np.sum(xx * xx))
    if den <= 1e-12:
        return np.nan
    return float(np.sum(xx * yy) / den)


def _score_gradient_order_audit(flags: pd.DataFrame, scores: np.ndarray, target: pd.Series, cfg: V91FConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = str(target["object_A"]); b = str(target["object_B"])
    df = flags.copy()
    df["score"] = np.asarray(scores, dtype=float)
    valid = df[np.isfinite(df["score"].to_numpy(float)) & np.isfinite(df["Y_delta_B_minus_A"].to_numpy(float))].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid = valid.sort_values("score", kind="mergesort")
    n_bins = max(2, int(cfg.score_gradient_bins))
    chunks = np.array_split(valid.index.to_numpy(), n_bins)
    rows: List[dict] = []
    for i, idx in enumerate(chunks, start=1):
        g = valid.loc[idx]
        st = _order_stats_for_y(g["Y_delta_B_minus_A"].to_numpy(float), a, b, cfg)
        rows.append({
            "window_id": target["window_id"],
            "target_name": target["target_name"],
            "object_A": a,
            "object_B": b,
            "score_bin": f"Q{i}",
            "score_bin_index": i,
            "n_bootstrap": int(len(g)),
            "score_min": float(g["score"].min()) if len(g) else np.nan,
            "score_max": float(g["score"].max()) if len(g) else np.nan,
            "score_mean": float(g["score"].mean()) if len(g) else np.nan,
            "delta_median": st["delta_median"],
            "delta_q025": st["delta_q025"],
            "delta_q975": st["delta_q975"],
            "P_A_earlier": st["P_A_earlier"],
            "P_B_earlier": st["P_B_earlier"],
            "P_same_day": st["P_same_day"],
            "dominant_order_direction": st["dominant_order_direction"],
            "dominant_order_probability": st["dominant_order_probability"],
            "order_level": st["order_level"],
        })
    detail = pd.DataFrame(rows)
    med_vals = detail["delta_median"].to_numpy(float)
    pa_vals = detail["P_A_earlier"].to_numpy(float)
    pb_vals = detail["P_B_earlier"].to_numpy(float)
    delta_mono = _monotonicity_score(med_vals)
    pa_mono = _monotonicity_score(pa_vals)
    pb_mono = _monotonicity_score(pb_vals)
    delta_slope = _slope(med_vals)
    pa_slope = _slope(pa_vals)
    pb_slope = _slope(pb_vals)
    if (np.isfinite(delta_mono) and abs(delta_mono) >= 0.90) or (np.isfinite(pa_mono) and abs(pa_mono) >= 0.90) or (np.isfinite(pb_mono) and abs(pb_mono) >= 0.90):
        status = "clear_continuous_gradient"
    elif (np.isfinite(delta_mono) and abs(delta_mono) >= 0.60) or (np.isfinite(pa_mono) and abs(pa_mono) >= 0.60) or (np.isfinite(pb_mono) and abs(pb_mono) >= 0.60):
        status = "partial_gradient"
    else:
        status = "no_clear_gradient"
    summary = pd.DataFrame([{
        "window_id": target["window_id"],
        "target_name": target["target_name"],
        "object_A": a,
        "object_B": b,
        "n_score_bins": n_bins,
        "delta_gradient_slope": delta_slope,
        "A_earlier_gradient_slope": pa_slope,
        "B_earlier_gradient_slope": pb_slope,
        "delta_monotonicity_score": delta_mono,
        "A_earlier_monotonicity_score": pa_mono,
        "B_earlier_monotonicity_score": pb_mono,
        "gradient_status": status,
    }])
    return detail, summary


def _level_value(level: str) -> int:
    return {"insufficient": -1, "weak_or_none": 0, "usable_90": 1, "credible_95": 2, "strict_99": 3}.get(str(level), 0)


def _evaluate_mca_evidence_v2(mode_row: dict, high_low: pd.DataFrame, quantile: pd.DataFrame, gradient_summary: pd.DataFrame, leverage_summary: dict, evidence_v1: dict, cfg: V91FConfig) -> dict:
    high = high_low[high_low["score_group"].eq("high")].iloc[0].to_dict() if not high_low.empty and high_low["score_group"].eq("high").any() else {}
    low = high_low[high_low["score_group"].eq("low")].iloc[0].to_dict() if not high_low.empty and high_low["score_group"].eq("low").any() else {}
    high_level = str(high.get("order_level", "insufficient"))
    low_level = str(low.get("order_level", "insufficient"))
    high_dir = str(high.get("dominant_order_direction", ""))
    low_dir = str(low.get("dominant_order_direction", ""))
    tercile_reversal = bool(high_dir and low_dir and high_dir != low_dir)
    extreme = bool(leverage_summary.get("extreme_year_dominated_flag", False))
    perm_level = str(mode_row.get("permutation_level", "insufficient"))
    stab = str(mode_row.get("mode_stability_status", "insufficient"))
    # Best quantile row: prefer reversal and high min level, then best max probability.
    best_q = {}
    if quantile is not None and not quantile.empty:
        qdf = quantile.copy()
        qdf["high_level_value"] = qdf["high_order_level"].map(_level_value)
        qdf["low_level_value"] = qdf["low_order_level"].map(_level_value)
        qdf["min_level_value"] = qdf[["high_level_value", "low_level_value"]].min(axis=1)
        qdf["max_level_value"] = qdf[["high_level_value", "low_level_value"]].max(axis=1)
        qdf["best_prob"] = qdf[["high_dominant_probability", "low_dominant_probability"]].max(axis=1)
        qdf["rev_rank"] = qdf["high_low_reversal_flag"].astype(int)
        qdf = qdf.sort_values(["rev_rank", "min_level_value", "max_level_value", "best_prob"], ascending=False)
        best_q = qdf.iloc[0].to_dict()
    grad = gradient_summary.iloc[0].to_dict() if gradient_summary is not None and not gradient_summary.empty else {}
    grad_status = str(grad.get("gradient_status", "no_clear_gradient"))

    best_high_level = str(best_q.get("high_order_level", high_level))
    best_low_level = str(best_q.get("low_order_level", low_level))
    best_reversal = bool(best_q.get("high_low_reversal_flag", tercile_reversal))
    best_quantile_scheme = str(best_q.get("quantile_scheme", "tercile"))
    best_high_prob = float(best_q.get("high_dominant_probability", high.get("dominant_order_probability", np.nan)))
    best_low_prob = float(best_q.get("low_dominant_probability", low.get("dominant_order_probability", np.nan)))
    max_side_level = max(_level_value(best_high_level), _level_value(best_low_level))
    min_side_level = min(_level_value(best_high_level), _level_value(best_low_level))

    if extreme and perm_level in ("usable_90", "credible_95", "strict_99"):
        ev2 = "extreme_year_leverage_only"
        pattern_type = "extreme_year_leverage"
    elif best_reversal and perm_level in ("credible_95", "strict_99") and stab == "stable" and max_side_level >= 2 and min_side_level >= 1:
        ev2 = "strong_reversal_candidate"
        pattern_type = "high_low_reversal"
    elif best_reversal and perm_level in ("credible_95", "strict_99") and max_side_level >= 1:
        ev2 = "moderate_reversal_candidate"
        pattern_type = "high_low_reversal"
    elif max_side_level >= 2 and not best_reversal and perm_level in ("credible_95", "strict_99"):
        ev2 = "one_sided_locking_candidate"
        pattern_type = "one_sided_order_locking"
    elif grad_status == "clear_continuous_gradient" and perm_level in ("credible_95", "strict_99"):
        ev2 = "continuous_gradient_candidate"
        pattern_type = "continuous_score_gradient"
    elif perm_level in ("usable_90", "credible_95", "strict_99") and (max_side_level >= 1 or grad_status in ("partial_gradient", "clear_continuous_gradient")):
        ev2 = "weak_hint"
        pattern_type = "weak_or_partial_structure"
    else:
        ev2 = "not_supported"
        pattern_type = "unsupported"
    interp_map = {
        "strong_reversal_candidate": "high/low or extreme score quantiles show opposite order with usable-to-credible support; bootstrap-space coupling candidate, not physical type",
        "moderate_reversal_candidate": "score quantiles show order reversal but support is moderate or quantile-sensitive; keep as candidate",
        "one_sided_locking_candidate": "one score phase strongly locks an order while the opposite phase is weak/unresolved; indicates order sharpening/suppression rather than full reversal",
        "continuous_gradient_candidate": "score bins show a continuous gradient in peak-time contrast/order probability",
        "weak_hint": "weak or partial bootstrap-space coupling structure; diagnostic only",
        "extreme_year_leverage_only": "coupling is present but high/low contrast is dominated by a few years",
        "not_supported": "no interpretable refined coupling pattern under current gates",
    }
    return {
        "window_id": mode_row.get("window_id"),
        "target_name": mode_row.get("target_name"),
        "object_A": mode_row.get("object_A"),
        "object_B": mode_row.get("object_B"),
        "permutation_level": perm_level,
        "mode_stability_status": stab,
        "extreme_year_dominated_flag": extreme,
        "tercile_high_order_level": high_level,
        "tercile_low_order_level": low_level,
        "tercile_high_low_reversal_flag": tercile_reversal,
        "best_quantile_scheme": best_quantile_scheme,
        "best_high_order_level": best_high_level,
        "best_low_order_level": best_low_level,
        "best_high_dominant_probability": best_high_prob,
        "best_low_dominant_probability": best_low_prob,
        "best_high_low_reversal_flag": best_reversal,
        "gradient_status": grad_status,
        "delta_gradient_slope": grad.get("delta_gradient_slope", np.nan),
        "A_earlier_gradient_slope": grad.get("A_earlier_gradient_slope", np.nan),
        "B_earlier_gradient_slope": grad.get("B_earlier_gradient_slope", np.nan),
        "evidence_v1_level": evidence_v1.get("evidence_level", ""),
        "evidence_v2_level": ev2,
        "pattern_type": pattern_type,
        "recommended_interpretation": interp_map[ev2],
        "interpretation_boundary": "refined bootstrap-space coupling diagnostic; not a physical year-type claim",
    }

def _year_leverage(year_counts: pd.DataFrame, scores: np.ndarray, target: pd.Series, cfg: V91FConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    piv, counts, boot_ids, years = _counts_pivot(year_counts)
    # Align scores to pivot order. In this module replay boot ids are sorted from same samples.
    if len(scores) != len(boot_ids):
        scores_use = np.asarray(scores[:len(boot_ids)], dtype=float)
    else:
        scores_use = np.asarray(scores, dtype=float)
    groups = _assign_score_groups(scores_use)
    high = groups == "high"; low = groups == "low"
    rows: List[dict] = []
    diffs = []
    for j, yr in enumerate(years):
        ch = counts[high, j] if high.any() else np.array([])
        cl = counts[low, j] if low.any() else np.array([])
        ph = ch > 0; pl = cl > 0
        mean_h = float(np.mean(ch)) if ch.size else np.nan
        mean_l = float(np.mean(cl)) if cl.size else np.nan
        pdiff = float(np.mean(ph) - np.mean(pl)) if ph.size and pl.size else np.nan
        cdiff = mean_h - mean_l if np.isfinite(mean_h) and np.isfinite(mean_l) else np.nan
        diffs.append(abs(cdiff) if np.isfinite(cdiff) else 0.0)
        rows.append({
            "window_id": target["window_id"],
            "target_name": target["target_name"],
            "year": yr,
            "mean_count_high": mean_h,
            "mean_count_low": mean_l,
            "count_difference_high_minus_low": cdiff,
            "present_prob_high": float(np.mean(ph)) if ph.size else np.nan,
            "present_prob_low": float(np.mean(pl)) if pl.size else np.nan,
            "present_prob_difference": pdiff,
        })
    arr = np.asarray(diffs, dtype=float)
    total = float(np.sum(arr))
    if total > 0:
        sorted_arr = np.sort(arr)[::-1]
        top1 = float(sorted_arr[0] / total) if len(sorted_arr) >= 1 else np.nan
        top3 = float(np.sum(sorted_arr[:3]) / total) if len(sorted_arr) >= 1 else np.nan
        top5 = float(np.sum(sorted_arr[:5]) / total) if len(sorted_arr) >= 1 else np.nan
    else:
        top1 = top3 = top5 = np.nan
    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_count_difference"] = df["count_difference_high_minus_low"].abs()
        df["year_leverage_rank"] = df["abs_count_difference"].rank(ascending=False, method="min")
        top = df.sort_values("abs_count_difference", ascending=False).head(1)
        top_year = str(top["year"].iloc[0]) if not top.empty else ""
    else:
        top_year = ""
    summary = {
        "top1_year": top_year,
        "top1_fraction_of_total_abs_leverage": top1,
        "top3_fraction_of_total_abs_leverage": top3,
        "top5_fraction_of_total_abs_leverage": top5,
        "extreme_year_dominated_flag": bool((np.isfinite(top1) and top1 >= cfg.top1_extreme_fraction) or (np.isfinite(top3) and top3 >= cfg.top3_extreme_fraction)),
    }
    return df, summary


def _object_contribution(u: np.ndarray, feature_meta: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    rows: List[dict] = []
    if feature_meta.empty or u.size == 0:
        return pd.DataFrame()
    total = float(np.sum(u * u))
    for obj, g in feature_meta.groupby("object"):
        idx = g["feature_index"].to_numpy(int)
        idx = idx[(idx >= 0) & (idx < u.size)]
        e = float(np.sum(u[idx] ** 2)) if idx.size else 0.0
        frac = e / total if total > 0 else np.nan
        rows.append({
            "window_id": target["window_id"],
            "target_name": target["target_name"],
            "object": obj,
            "block_energy": e,
            "block_contribution_fraction": frac,
            "dominant_object_warning": bool(np.isfinite(frac) and frac >= 0.60),
        })
    return pd.DataFrame(rows)


def _pattern_coefficients(u: np.ndarray, feature_meta: pd.DataFrame, target: pd.Series, cfg: V91FConfig) -> pd.DataFrame:
    if feature_meta.empty or u.size == 0:
        return pd.DataFrame()
    fm = feature_meta.copy()
    fm = fm[fm["feature_index"].astype(int).between(0, u.size - 1)].copy()
    fm["coefficient"] = [float(u[int(i)]) for i in fm["feature_index"].astype(int)]
    fm["target_name"] = target["target_name"]
    # Optional cap for very large files: keep largest absolute coefficients per target.
    if cfg.max_pattern_coefficients_per_target and len(fm) > cfg.max_pattern_coefficients_per_target:
        fm["abs_coefficient"] = fm["coefficient"].abs()
        fm = fm.sort_values("abs_coefficient", ascending=False).head(cfg.max_pattern_coefficients_per_target).copy()
    return fm



def _build_all_target_y_table(peak_samples: pd.DataFrame, target_registry: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """Build all target Y vectors for a window so specificity/null audits can compare targets."""
    rows: List[pd.DataFrame] = []
    y_by_target: Dict[str, np.ndarray] = {}
    flags_by_target: Dict[str, pd.DataFrame] = {}
    for _, target in target_registry.iterrows():
        Y, boot_ids, flags = _Y_for_pair(peak_samples, target)
        tname = str(target["target_name"])
        f = flags.copy()
        f["window_id"] = target["window_id"]
        f["target_name"] = tname
        f["object_A"] = target["object_A"]
        f["object_B"] = target["object_B"]
        rows.append(f)
        y_by_target[tname] = np.asarray(Y, dtype=float)
        flags_by_target[tname] = flags
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return table, y_by_target, flags_by_target


def _target_specificity_audit(window_id: str, target_name: str, scores: np.ndarray, y_by_target: Dict[str, np.ndarray], cfg: V91FConfig) -> pd.DataFrame:
    rows: List[dict] = []
    own_corr = _corr(scores, y_by_target.get(target_name, np.array([]))) if target_name in y_by_target else np.nan
    all_abs = []
    for name, y in y_by_target.items():
        c = _corr(scores, y)
        all_abs.append((name, abs(c) if np.isfinite(c) else np.nan, c))
    finite_abs = [x for x in all_abs if np.isfinite(x[1])]
    finite_abs_sorted = sorted(finite_abs, key=lambda x: x[1], reverse=True)
    ranks = {name: i + 1 for i, (name, _a, _c) in enumerate(finite_abs_sorted)}
    max_other = np.nan
    if finite_abs:
        others = [a for name, a, _c in finite_abs if name != target_name]
        max_other = float(np.nanmax(others)) if others else np.nan
    own_abs = abs(own_corr) if np.isfinite(own_corr) else np.nan
    own_minus = float(own_abs - max_other) if np.isfinite(own_abs) and np.isfinite(max_other) else np.nan
    own_rank = int(ranks.get(target_name, 999999)) if ranks else 999999
    if own_rank == 1 and np.isfinite(own_minus) and own_minus >= cfg.specificity_margin:
        status = "pair_specific"
    elif own_rank <= 3:
        status = "common_mode"
    else:
        status = "not_specific"
    for name, abs_c, c in all_abs:
        rows.append({
            "window_id": window_id,
            "mode_target": target_name,
            "tested_y_target": name,
            "corr_score_y": c,
            "abs_corr_score_y": abs_c,
            "own_target_corr": own_corr,
            "max_other_target_corr": max_other,
            "own_minus_max_other": own_minus,
            "own_abs_rank": own_rank,
            "n_targets_tested": len(y_by_target),
            "specificity_margin": cfg.specificity_margin,
            "specificity_status": status,
        })
    return pd.DataFrame(rows)


def _cross_target_null_audit(window_id: str, target_name: str, X: np.ndarray, y_true: np.ndarray, y_by_target: Dict[str, np.ndarray], cfg: V91FConfig) -> pd.DataFrame:
    if not cfg.enable_cross_target_null:
        return pd.DataFrame()
    true_fit = _fit_mode(X, y_true)
    true_corr = abs(float(true_fit.get("corr", np.nan)))
    null_vals: List[Tuple[str, float]] = []
    for name, y in y_by_target.items():
        if name == target_name:
            continue
        fit = _fit_mode(X, y)
        null_vals.append((name, abs(float(fit.get("corr", np.nan)))))
    finite = np.asarray([v for _n, v in null_vals if np.isfinite(v)], dtype=float)
    if finite.size:
        pct = float(np.mean(finite <= true_corr)) if np.isfinite(true_corr) else np.nan
        # rank 1 means true is greater than all nulls.
        rank = int(1 + np.sum(finite > true_corr)) if np.isfinite(true_corr) else 999999
    else:
        pct = np.nan
        rank = 999999
    level = _prob_level(pct, cfg)
    if level == "strict_99":
        spec_level = "specific_99"
    elif level == "credible_95":
        spec_level = "specific_95"
    elif level == "usable_90":
        spec_level = "specific_90"
    else:
        spec_level = "not_specific"
    rows = []
    for name, nc in null_vals:
        rows.append({
            "window_id": window_id,
            "target_name": target_name,
            "null_target_name": name,
            "true_corr": true_corr,
            "null_corr": nc,
            "abs_true_corr": true_corr,
            "abs_null_corr": nc,
            "true_minus_null": true_corr - nc if np.isfinite(true_corr) and np.isfinite(nc) else np.nan,
            "true_rank_among_nulls": rank,
            "specificity_percentile": pct,
            "specificity_level": spec_level,
        })
    if not rows:
        rows.append({"window_id": window_id, "target_name": target_name, "null_target_name": "none", "true_corr": true_corr, "null_corr": np.nan, "abs_true_corr": true_corr, "abs_null_corr": np.nan, "true_minus_null": np.nan, "true_rank_among_nulls": rank, "specificity_percentile": pct, "specificity_level": "insufficient"})
    return pd.DataFrame(rows)


def _signflip_null_audit(window_id: str, target_name: str, X: np.ndarray, y: np.ndarray, observed_abs_corr: float, cfg: V91FConfig, seed: int) -> dict:
    if not cfg.enable_signflip_null or cfg.signflip_n <= 0 or not np.isfinite(observed_abs_corr):
        return {"signflip_percentile": np.nan, "signflip_level": "insufficient", "signflip_corr_q90": np.nan, "signflip_corr_q95": np.nan, "signflip_corr_q99": np.nan, "signflip_n": 0}
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if valid.sum() < 5:
        return {"signflip_percentile": np.nan, "signflip_level": "insufficient", "signflip_corr_q90": np.nan, "signflip_corr_q95": np.nan, "signflip_corr_q99": np.nan, "signflip_n": int(cfg.signflip_n)}
    Xv = X[valid]
    yv = y[valid]
    vals: List[float] = []
    for _ in range(int(cfg.signflip_n)):
        sign = rng.choice(np.array([-1.0, 1.0]), size=len(yv))
        ysf = yv * sign
        fit = _fit_mode(Xv, ysf)
        c = abs(float(fit.get("corr", np.nan)))
        if np.isfinite(c):
            vals.append(c)
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"signflip_percentile": np.nan, "signflip_level": "insufficient", "signflip_corr_q90": np.nan, "signflip_corr_q95": np.nan, "signflip_corr_q99": np.nan, "signflip_n": int(cfg.signflip_n)}
    pct = float(np.mean(arr <= abs(observed_abs_corr)))
    base_level = _prob_level(pct, cfg)
    if base_level == "strict_99":
        level = "direction_specific_99"
    elif base_level == "credible_95":
        level = "direction_specific_95"
    elif base_level == "usable_90":
        level = "direction_specific_90"
    else:
        level = "not_direction_specific"
    return {
        "window_id": window_id,
        "target_name": target_name,
        "observed_abs_corr": abs(observed_abs_corr),
        "signflip_n": int(cfg.signflip_n),
        "signflip_corr_q90": float(np.nanquantile(arr, 0.90)),
        "signflip_corr_q95": float(np.nanquantile(arr, 0.95)),
        "signflip_corr_q99": float(np.nanquantile(arr, 0.99)),
        "signflip_percentile": pct,
        "signflip_level": level,
    }


def _score_group_masks(scores: np.ndarray) -> Dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    valid = np.where(np.isfinite(scores))[0]
    masks: Dict[str, np.ndarray] = {}
    for name in ["low", "mid", "high", "bottom_decile", "top_decile"]:
        m = np.zeros(scores.shape, dtype=bool)
        masks[name] = m
    if valid.size < 3:
        return masks
    order = valid[np.argsort(scores[valid], kind="mergesort")]
    n = len(order)
    n_low = n // 3
    n_mid = n // 3
    masks["low"][order[:n_low]] = True
    masks["mid"][order[n_low:n_low+n_mid]] = True
    masks["high"][order[n_low+n_mid:]] = True
    d = max(1, int(math.floor(0.10 * n)))
    masks["bottom_decile"][order[:d]] = True
    masks["top_decile"][order[-d:]] = True
    return masks


def _phase_composite_profiles(window_id: str, target_name: str, X: np.ndarray, feature_meta: pd.DataFrame, scores: np.ndarray, cfg: V91FConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.enable_phase_composite or X.size == 0 or feature_meta is None or feature_meta.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta = feature_meta.reset_index(drop=True).copy()
    if len(meta) != X.shape[1]:
        meta = meta[meta.get("feature_kept", True).astype(bool)].reset_index(drop=True)
    if len(meta) != X.shape[1]:
        return pd.DataFrame(), pd.DataFrame()
    masks = _score_group_masks(scores)
    prof_rows = []
    # aggregate at object/day/profile_coord level; these are standardized/equal-weight X units.
    key_cols = [c for c in ["object", "day", "profile_coord_index"] if c in meta.columns]
    if not key_cols:
        return pd.DataFrame(), pd.DataFrame()
    # Precompute group means per feature, then join with meta and aggregate duplicate feature coords.
    group_feature_means: Dict[str, pd.DataFrame] = {}
    for gname, mask in masks.items():
        if mask.sum() == 0:
            continue
        vals = np.nanmean(X[mask, :], axis=0)
        tmp = meta[key_cols].copy()
        tmp["value"] = vals
        tmp["n_bootstrap"] = int(mask.sum())
        agg = tmp.groupby(key_cols, dropna=False).agg(composite_anomaly_standardized=("value", "mean"), n_bootstrap=("n_bootstrap", "first")).reset_index()
        agg.insert(0, "score_group", gname)
        agg.insert(0, "target_name", target_name)
        agg.insert(0, "window_id", window_id)
        group_feature_means[gname] = agg
        prof_rows.append(agg)
    profiles = pd.concat(prof_rows, ignore_index=True) if prof_rows else pd.DataFrame()
    diff_rows = []
    def add_diff(g1: str, g2: str, cname: str):
        if g1 not in group_feature_means or g2 not in group_feature_means:
            return
        a = group_feature_means[g1]
        b = group_feature_means[g2]
        m = a.merge(b, on=["window_id", "target_name"] + key_cols, suffixes=("_a", "_b"), how="inner")
        if m.empty:
            return
        out = m[["window_id", "target_name"] + key_cols].copy()
        out.insert(2, "contrast", cname)
        out["difference_value_standardized"] = m["composite_anomaly_standardized_a"] - m["composite_anomaly_standardized_b"]
        diff_rows.append(out)
    add_diff("high", "low", "high_minus_low")
    add_diff("top_decile", "bottom_decile", "top_decile_minus_bottom_decile")
    diffs = pd.concat(diff_rows, ignore_index=True) if diff_rows else pd.DataFrame()
    return profiles, diffs


def _pattern_summary(window_id: str, target_name: str, u: np.ndarray, feature_meta: pd.DataFrame, cfg: V91FConfig) -> pd.DataFrame:
    if not cfg.enable_pattern_summary or u.size == 0 or feature_meta is None or feature_meta.empty:
        return pd.DataFrame()
    meta = feature_meta.reset_index(drop=True).copy()
    if len(meta) != len(u):
        meta = meta[meta.get("feature_kept", True).astype(bool)].reset_index(drop=True)
    if len(meta) != len(u) or "object" not in meta.columns:
        return pd.DataFrame()
    df = meta.copy()
    df["coef"] = u
    df["abs_coef"] = np.abs(u)
    total_abs = float(df["abs_coef"].sum())
    if total_abs <= cfg.eps:
        total_abs = 1.0
    day_mid = float(np.nanmedian(df["day"])) if "day" in df.columns and df["day"].notna().any() else np.nan
    rows = []
    for obj, sub in df.groupby("object"):
        abs_sum = float(sub["abs_coef"].sum())
        pos_sum = float(sub.loc[sub["coef"] > 0, "abs_coef"].sum())
        neg_sum = float(sub.loc[sub["coef"] < 0, "abs_coef"].sum())
        if "day" in sub.columns and np.isfinite(day_mid):
            early = float(sub.loc[sub["day"] <= day_mid, "abs_coef"].sum()) / max(abs_sum, cfg.eps)
            late = float(sub.loc[sub["day"] > day_mid, "abs_coef"].sum()) / max(abs_sum, cfg.eps)
        else:
            early = late = np.nan
        # dominant coordinate range: top 20% abs loading within object.
        thresh = float(np.nanquantile(sub["abs_coef"], 0.80)) if len(sub) else np.nan
        dom = sub[sub["abs_coef"] >= thresh] if np.isfinite(thresh) else sub.iloc[0:0]
        rows.append({
            "window_id": window_id,
            "target_name": target_name,
            "object": obj,
            "abs_loading_fraction": abs_sum / total_abs,
            "positive_loading_fraction": pos_sum / max(abs_sum, cfg.eps),
            "negative_loading_fraction": neg_sum / max(abs_sum, cfg.eps),
            "early_half_abs_loading_fraction": early,
            "late_half_abs_loading_fraction": late,
            "dominant_day_start": float(dom["day"].min()) if "day" in dom.columns and not dom.empty else np.nan,
            "dominant_day_end": float(dom["day"].max()) if "day" in dom.columns and not dom.empty else np.nan,
            "dominant_profile_coord_min": float(dom["profile_coord_index"].min()) if "profile_coord_index" in dom.columns and not dom.empty else np.nan,
            "dominant_profile_coord_max": float(dom["profile_coord_index"].max()) if "profile_coord_index" in dom.columns and not dom.empty else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["dominant_object_rank"] = out["abs_loading_fraction"].rank(ascending=False, method="first").astype(int)
    return out


def _evaluate_evidence_v3(ev2_row: dict, specificity_df: pd.DataFrame, cross_null_df: pd.DataFrame, signflip_row: dict, pattern_summary_df: pd.DataFrame, leverage_summary: dict, cfg: V91FConfig) -> dict:
    v2 = str(ev2_row.get("evidence_v2_level", ev2_row.get("evidence_level", "")))
    tname = ev2_row.get("target_name")
    # Specificity status: all rows for this mode have identical summary fields.
    spec_status = "insufficient"
    own_rank = np.nan
    own_minus = np.nan
    if specificity_df is not None and not specificity_df.empty:
        sub = specificity_df[specificity_df["mode_target"].eq(tname)] if "mode_target" in specificity_df.columns else specificity_df
        if not sub.empty:
            spec_status = str(sub["specificity_status"].iloc[0])
            own_rank = sub["own_abs_rank"].iloc[0] if "own_abs_rank" in sub.columns else np.nan
            own_minus = sub["own_minus_max_other"].iloc[0] if "own_minus_max_other" in sub.columns else np.nan
    cross_level = "insufficient"
    if cross_null_df is not None and not cross_null_df.empty:
        sub = cross_null_df[cross_null_df["target_name"].eq(tname)] if "target_name" in cross_null_df.columns else cross_null_df
        if not sub.empty and "specificity_level" in sub.columns:
            # all rows duplicate specificity level.
            cross_level = str(sub["specificity_level"].iloc[0])
    sign_level = str(signflip_row.get("signflip_level", "insufficient"))
    extreme = bool(leverage_summary.get("extreme_year_dominated_flag", False))
    # Pattern interpretability summary.
    dom_obj = ""
    dom_frac = np.nan
    dom_half = "unclear"
    patt_status = "pattern_summary_missing"
    if pattern_summary_df is not None and not pattern_summary_df.empty:
        ps = pattern_summary_df[pattern_summary_df["target_name"].eq(tname)] if "target_name" in pattern_summary_df.columns else pattern_summary_df
        if not ps.empty:
            top = ps.sort_values("abs_loading_fraction", ascending=False).iloc[0]
            dom_obj = str(top.get("object", ""))
            dom_frac = float(top.get("abs_loading_fraction", np.nan))
            early = float(top.get("early_half_abs_loading_fraction", np.nan))
            late = float(top.get("late_half_abs_loading_fraction", np.nan))
            if np.isfinite(early) and np.isfinite(late):
                dom_half = "early" if early >= late else "late"
            patt_status = "interpretable_summary_available" if np.isfinite(dom_frac) else "pattern_unclear"
    def good_sign(level: str, threshold95: bool = True) -> bool:
        if threshold95:
            return level in ("direction_specific_95", "direction_specific_99")
        return level in ("direction_specific_90", "direction_specific_95", "direction_specific_99")
    if extreme:
        ev3 = "candidate_but_extreme_year_caution"
    elif v2 in ("strong_reversal_candidate", "moderate_reversal_candidate") and spec_status == "pair_specific" and cross_level in ("specific_95", "specific_99") and good_sign(sign_level, True):
        ev3 = "robust_pair_specific_reversal"
    elif v2 in ("strong_reversal_candidate", "moderate_reversal_candidate") and spec_status in ("common_mode", "pair_specific") and good_sign(sign_level, True):
        ev3 = "robust_common_mode_reversal"
    elif v2 == "one_sided_locking_candidate" and good_sign(sign_level, True):
        ev3 = "robust_one_sided_locking"
    elif v2 == "continuous_gradient_candidate" and good_sign(sign_level, False):
        ev3 = "robust_continuous_gradient"
    elif v2 not in ("not_supported", "", "nan") and spec_status == "not_specific":
        ev3 = "candidate_but_not_specific"
    elif v2 not in ("not_supported", "", "nan") and not good_sign(sign_level, False):
        ev3 = "candidate_but_direction_null_weak"
    elif v2 not in ("not_supported", "", "nan") and patt_status == "pattern_unclear":
        ev3 = "candidate_but_pattern_unclear"
    else:
        ev3 = "not_supported"
    interp_map = {
        "robust_pair_specific_reversal": "specific bootstrap-composite mode with robust high/low order reversal; still not a physical year type",
        "robust_common_mode_reversal": "common bootstrap-composite mode modulates multiple targets and shows order reversal",
        "robust_one_sided_locking": "mode robustly locks one order direction in one phase but does not establish symmetric reversal",
        "robust_continuous_gradient": "mode robustly provides continuous peak-time/order gradient without categorical reversal",
        "candidate_but_not_specific": "candidate coupling exists but is not pair-specific; interpret as common mode or downgrade",
        "candidate_but_direction_null_weak": "candidate coupling weakens under sign-flip direction null",
        "candidate_but_pattern_unclear": "candidate coupling lacks interpretable pattern summary",
        "candidate_but_extreme_year_caution": "candidate coupling is likely dominated by high-leverage years",
        "not_supported": "not supported after specificity/null/pattern gates",
    }
    out = dict(ev2_row)
    out.update({
        "specificity_status": spec_status,
        "specificity_own_abs_rank": own_rank,
        "specificity_own_minus_max_other": own_minus,
        "cross_target_specificity_level": cross_level,
        "signflip_level": sign_level,
        "signflip_percentile": signflip_row.get("signflip_percentile", np.nan),
        "dominant_object": dom_obj,
        "dominant_object_fraction": dom_frac,
        "dominant_time_half": dom_half,
        "pattern_interpretability_status": patt_status,
        "evidence_v3_level": ev3,
        "recommended_interpretation_v3": interp_map[ev3],
    })
    return out

def _evidence_for_target(mode_row: dict, high_low: pd.DataFrame, leverage_summary: dict, cfg: V91FConfig) -> dict:
    high = high_low[high_low["score_group"].eq("high")].iloc[0].to_dict() if not high_low.empty and high_low["score_group"].eq("high").any() else {}
    low = high_low[high_low["score_group"].eq("low")].iloc[0].to_dict() if not high_low.empty and high_low["score_group"].eq("low").any() else {}
    high_dir = high.get("dominant_order_direction", "")
    low_dir = low.get("dominant_order_direction", "")
    high_prob = float(high.get("dominant_order_probability", np.nan)) if high else np.nan
    low_prob = float(low.get("dominant_order_probability", np.nan)) if low else np.nan
    reversal = bool(high_dir and low_dir and high_dir != low_dir)
    perm_level = str(mode_row.get("permutation_level", "insufficient"))
    stab = str(mode_row.get("mode_stability_status", "insufficient"))
    extreme = bool(leverage_summary.get("extreme_year_dominated_flag", False))
    best = np.nanmax([high_prob, low_prob]) if np.isfinite([high_prob, low_prob]).any() else np.nan
    high_level = _prob_level(high_prob, cfg)
    low_level = _prob_level(low_prob, cfg)
    if extreme and perm_level in ("usable_90", "credible_95", "strict_99") and reversal:
        ev = "extreme_year_leverage_only"
    elif perm_level in ("credible_95", "strict_99") and stab == "stable" and reversal and ((high_prob >= cfg.evidence_credible and low_prob >= cfg.evidence_usable) or (low_prob >= cfg.evidence_credible and high_prob >= cfg.evidence_usable)) and not extreme:
        ev = "strong_bootstrap_coupling_candidate"
    elif perm_level in ("credible_95", "strict_99") and reversal and best >= cfg.evidence_usable and stab in ("stable", "caution") and not extreme:
        ev = "moderate_bootstrap_coupling_candidate"
    elif perm_level in ("usable_90", "credible_95", "strict_99") and (reversal or best >= cfg.evidence_usable):
        ev = "weak_bootstrap_coupling_hint"
    else:
        ev = "not_supported"
    interp = {
        "strong_bootstrap_coupling_candidate": "bootstrap-space X-Y coupling is stable and high/low score groups show order differentiation; not a physical type yet",
        "moderate_bootstrap_coupling_candidate": "bootstrap-space X-Y coupling has usable/credible order differentiation; needs composite/physical audit",
        "weak_bootstrap_coupling_hint": "weak bootstrap-space coupling hint; keep as diagnostic only",
        "extreme_year_leverage_only": "coupling/order signal is likely dominated by a small number of high-leverage years",
        "not_supported": "no interpretable bootstrap-space coupling under current gates",
    }[ev]
    return {
        "window_id": mode_row.get("window_id"),
        "target_name": mode_row.get("target_name"),
        "object_A": mode_row.get("object_A"),
        "object_B": mode_row.get("object_B"),
        "permutation_level": perm_level,
        "permutation_percentile": mode_row.get("permutation_percentile"),
        "mode_stability_status": stab,
        "mode_pattern_corr_median": mode_row.get("mode_pattern_corr_median"),
        "corr_score_Y": mode_row.get("corr_score_Y"),
        "high_dominant_direction": high_dir,
        "high_dominant_probability": high_prob,
        "high_order_level": high_level,
        "low_dominant_direction": low_dir,
        "low_dominant_probability": low_prob,
        "low_order_level": low_level,
        "high_low_reversal_flag": reversal,
        "best_group_order_level": _prob_level(best, cfg),
        "top1_year": leverage_summary.get("top1_year", ""),
        "top1_fraction_of_total_abs_leverage": leverage_summary.get("top1_fraction_of_total_abs_leverage", np.nan),
        "top3_fraction_of_total_abs_leverage": leverage_summary.get("top3_fraction_of_total_abs_leverage", np.nan),
        "extreme_year_dominated_flag": extreme,
        "evidence_level": ev,
        "recommended_interpretation": interp,
        "interpretation_boundary": "bootstrap-space coupling diagnostic; bootstrap samples are not independent physical years",
    }


def _write_summary(path: Path, evidence: pd.DataFrame) -> None:
    lines = ["# V9.1_f bootstrap-composite MCA summary", ""]
    lines.append("This branch diagnoses X-Y coupling in V9 bootstrap resampling space. It does not identify physical year types.")
    lines.append("")
    if evidence is None or evidence.empty:
        lines.append("No evidence rows generated.")
    else:
        lines.append("## Evidence counts")
        for k, v in evidence["evidence_level"].value_counts(dropna=False).items():
            lines.append(f"- {k}: {int(v)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_bootstrap_composite_mca_audit_v9_1_f(v91_root: Path) -> None:
    v91_root = Path(v91_root).resolve()
    stage_root = _stage_root_from_v91(v91_root)
    cfg = V91FConfig.from_env()
    out_root = _ensure_dir(v91_root / "outputs" / OUTPUT_TAG)
    out_cross = _ensure_dir(out_root / "cross_window")

    _log("[V9.1_f] Loading V9/V9.1_c helpers and V9 context...")
    cmod = _import_c_module(v91_root)
    cfg_c = _make_c_config(cfg, cmod)
    v9peak, v7multi = cmod._import_v9_and_v7(stage_root)
    cfg_v7 = cmod._make_cfg_for_v91c(v9peak, v7multi, v91_root, cfg_c)
    _scopes_all, run_scopes, _validity, run_audit = cmod._load_scopes(v91_root, v9peak, v7multi, cfg_v7, cfg_c)
    profiles, year_values = cmod._load_profiles(v91_root, v9peak, v7multi, cfg_v7)
    v9_root = stage_root / "V9"

    all_replay_audit: List[pd.DataFrame] = []
    all_mode: List[pd.DataFrame] = []
    all_scores: List[pd.DataFrame] = []
    all_obj_contrib: List[pd.DataFrame] = []
    all_coeff: List[pd.DataFrame] = []
    all_perm: List[pd.DataFrame] = []
    all_stability: List[pd.DataFrame] = []
    all_high_low: List[pd.DataFrame] = []
    all_quantile: List[pd.DataFrame] = []
    all_gradient: List[pd.DataFrame] = []
    all_gradient_summary: List[pd.DataFrame] = []
    all_leverage: List[pd.DataFrame] = []
    all_leverage_summary: List[pd.DataFrame] = []
    all_evidence: List[pd.DataFrame] = []
    all_evidence_v2: List[pd.DataFrame] = []
    all_target_y_tables: List[pd.DataFrame] = []
    all_specificity: List[pd.DataFrame] = []
    all_cross_null: List[pd.DataFrame] = []
    all_signflip: List[pd.DataFrame] = []
    all_phase_profiles: List[pd.DataFrame] = []
    all_phase_diffs: List[pd.DataFrame] = []
    all_pattern_summary: List[pd.DataFrame] = []
    all_evidence_v3: List[pd.DataFrame] = []
    all_nan_audit: List[pd.DataFrame] = []
    all_targets: List[pd.DataFrame] = []

    for scope in run_scopes:
        wid = scope.window_id
        if wid not in cfg.windows:
            continue
        _log(f"[V9.1_f] {wid}: replaying V9 bootstrap with year counts...")
        out_win = _ensure_dir(out_root / "per_window" / wid)
        year_counts, peak_samples, pair_samples = cmod._run_v9_bootstrap_replay_with_year_counts(
            v7multi, profiles, year_values, scope, cfg_v7, cfg_c
        )
        _safe_to_csv(year_counts, out_win / f"bootstrap_sample_year_counts_{wid}.csv")
        _safe_to_csv(peak_samples, out_win / f"bootstrap_object_peak_samples_{wid}.csv")
        _safe_to_csv(pair_samples, out_win / f"bootstrap_pairwise_order_samples_{wid}.csv")
        replay_audit = cmod._replay_regression_audit(v9_root, scope, peak_samples, pair_samples)
        _safe_to_csv(replay_audit, out_win / f"v9_replay_bootstrap_regression_audit_{wid}.csv")
        all_replay_audit.append(replay_audit)

        _log(f"[V9.1_f] {wid}: building bootstrap composite X matrix...")
        X, feature_meta, block_audit, raw_note, boot_meta, nan_audit = _build_X_matrix(profiles, year_counts, scope, cfg, out_win)
        _safe_to_csv(feature_meta, out_win / f"bootstrap_composite_X_feature_meta_{wid}.csv")
        _safe_to_csv(block_audit, out_win / f"object_block_variance_contribution_{wid}.csv")
        _safe_to_csv(raw_note, out_win / f"bootstrap_composite_X_samples_{wid}.csv")
        _safe_to_csv(boot_meta, out_win / f"bootstrap_composite_X_bootstrap_meta_{wid}.csv")
        _safe_to_csv(nan_audit, out_win / f"bootstrap_composite_X_nan_feature_audit_{wid}.csv")
        all_nan_audit.append(nan_audit)

        target_registry = _targets_for_window(wid, cfg)
        _safe_to_csv(target_registry, out_win / f"bootstrap_composite_mca_target_registry_{wid}.csv")
        all_targets.append(target_registry)
        target_y_table, y_by_target, _flags_by_target = _build_all_target_y_table(peak_samples, target_registry)
        _safe_to_csv(target_y_table, out_win / f"bootstrap_composite_mca_target_y_table_{wid}.csv")
        all_target_y_tables.append(target_y_table)

        mode_rows: List[dict] = []
        score_rows: List[pd.DataFrame] = []
        obj_contrib_parts: List[pd.DataFrame] = []
        coeff_parts: List[pd.DataFrame] = []
        perm_rows: List[dict] = []
        stability_parts: List[pd.DataFrame] = []
        high_low_parts: List[pd.DataFrame] = []
        quantile_parts: List[pd.DataFrame] = []
        gradient_parts: List[pd.DataFrame] = []
        gradient_summary_parts: List[pd.DataFrame] = []
        leverage_parts: List[pd.DataFrame] = []
        leverage_summary_rows: List[dict] = []
        evidence_rows: List[dict] = []
        evidence_v2_rows: List[dict] = []
        specificity_parts: List[pd.DataFrame] = []
        cross_null_parts: List[pd.DataFrame] = []
        signflip_rows: List[dict] = []
        phase_profile_parts: List[pd.DataFrame] = []
        phase_diff_parts: List[pd.DataFrame] = []
        pattern_summary_parts: List[pd.DataFrame] = []
        evidence_v3_rows: List[dict] = []

        for ti, target in target_registry.iterrows():
            tname = str(target["target_name"])
            _log(f"[V9.1_f] {wid}: fitting bootstrap-composite MCA for {tname}")
            Y, boot_ids, flags = _Y_for_pair(peak_samples, target)
            fit = _fit_mode(X, Y)
            u = np.asarray(fit.get("u", np.zeros(X.shape[1])), dtype=float)
            scores = np.asarray(fit.get("score", np.full(len(Y), np.nan)), dtype=float)
            corr_sy = float(fit.get("corr", np.nan))
            sp_sy = float(fit.get("spearman", np.nan))
            perm = _permutation_audit(X, Y, abs(corr_sy) if np.isfinite(corr_sy) else np.nan, cfg, seed=1000 + (hash((wid, tname)) % 100000))
            stab_df, stab_sum = _mode_stability(X, Y, u, cfg, seed=2000 + (hash((wid, tname)) % 100000))
            high_low = _high_low_order(flags, scores, target, cfg)
            quantile_df = _quantile_sensitivity_order_audit(flags, scores, target, cfg)
            gradient_df, gradient_sum_df = _score_gradient_order_audit(flags, scores, target, cfg)
            lev_df, lev_sum = _year_leverage(year_counts, scores, target, cfg)
            mode_row = {
                "window_id": wid,
                "target_name": tname,
                "object_A": target["object_A"],
                "object_B": target["object_B"],
                "n_bootstrap": int(len(Y)),
                "n_features": int(X.shape[1]),
                "Y_definition": target["Y_definition"],
                "Y_mean": float(np.nanmean(Y)) if np.isfinite(Y).any() else np.nan,
                "Y_std": float(np.nanstd(Y)) if np.isfinite(Y).any() else np.nan,
                "fit_status": fit.get("status", ""),
                "covariance_strength": float(fit.get("covariance_strength", np.nan)),
                "corr_score_Y": corr_sy,
                "spearman_score_Y": sp_sy,
                **perm,
                **stab_sum,
                "method_role": "bootstrap_space_XY_coupling_mode_not_independent_year_type",
            }
            mode_rows.append(mode_row)
            perm_rows.append({"window_id": wid, "target_name": tname, **perm, "observed_abs_corr": abs(corr_sy) if np.isfinite(corr_sy) else np.nan})
            stab_df2 = stab_df.copy(); stab_df2["window_id"] = wid; stab_df2["target_name"] = tname
            stability_parts.append(stab_df2)
            # Scores table.
            sg = _assign_score_groups(scores)
            sdf = pd.DataFrame({
                "window_id": wid,
                "target_name": tname,
                "bootstrap_id": boot_ids,
                "Y_delta_B_minus_A": Y,
                "score": scores,
                "score_group": sg,
                "A_earlier_flag": flags["A_earlier_flag"].to_numpy(bool),
                "B_earlier_flag": flags["B_earlier_flag"].to_numpy(bool),
            })
            score_rows.append(sdf)
            obj_contrib_parts.append(_object_contribution(u, feature_meta, target))
            coeff_parts.append(_pattern_coefficients(u, feature_meta, target, cfg))
            high_low_parts.append(high_low)
            quantile_parts.append(quantile_df)
            gradient_parts.append(gradient_df)
            gradient_summary_parts.append(gradient_sum_df)
            lev_df["target_name"] = tname
            leverage_parts.append(lev_df)
            lev_summary_row = {"window_id": wid, "target_name": tname, **lev_sum}
            leverage_summary_rows.append(lev_summary_row)
            ev1 = _evidence_for_target(mode_row, high_low, lev_sum, cfg)
            evidence_rows.append(ev1)
            ev2 = _evaluate_mca_evidence_v2(mode_row, high_low, quantile_df, gradient_sum_df, lev_sum, ev1, cfg)
            evidence_v2_rows.append(ev2)
            # hotfix02 specificity/null/interpretability audits.
            spec_df = _target_specificity_audit(wid, tname, scores, y_by_target, cfg) if cfg.enable_target_specificity else pd.DataFrame()
            cross_df = _cross_target_null_audit(wid, tname, X, Y, y_by_target, cfg) if cfg.enable_cross_target_null else pd.DataFrame()
            sf = _signflip_null_audit(wid, tname, X, Y, abs(corr_sy) if np.isfinite(corr_sy) else np.nan, cfg, seed=3000 + (hash((wid, tname)) % 100000)) if cfg.enable_signflip_null else {"window_id": wid, "target_name": tname, "signflip_level": "disabled"}
            phase_prof, phase_diff = _phase_composite_profiles(wid, tname, X, feature_meta, scores, cfg) if cfg.enable_phase_composite else (pd.DataFrame(), pd.DataFrame())
            patt_sum = _pattern_summary(wid, tname, u, feature_meta, cfg) if cfg.enable_pattern_summary else pd.DataFrame()
            specificity_parts.append(spec_df)
            cross_null_parts.append(cross_df)
            signflip_rows.append(sf)
            phase_profile_parts.append(phase_prof)
            phase_diff_parts.append(phase_diff)
            pattern_summary_parts.append(patt_sum)
            evidence_v3_rows.append(_evaluate_evidence_v3(ev2, spec_df, cross_df, sf, patt_sum, lev_sum, cfg))

        mode_df = pd.DataFrame(mode_rows)
        score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
        obj_contrib = pd.concat(obj_contrib_parts, ignore_index=True) if obj_contrib_parts else pd.DataFrame()
        coeff = pd.concat(coeff_parts, ignore_index=True) if coeff_parts else pd.DataFrame()
        perm_df = pd.DataFrame(perm_rows)
        stability_df = pd.concat(stability_parts, ignore_index=True) if stability_parts else pd.DataFrame()
        high_low_df = pd.concat(high_low_parts, ignore_index=True) if high_low_parts else pd.DataFrame()
        quantile_df = pd.concat(quantile_parts, ignore_index=True) if quantile_parts else pd.DataFrame()
        gradient_df = pd.concat(gradient_parts, ignore_index=True) if gradient_parts else pd.DataFrame()
        gradient_summary_df = pd.concat(gradient_summary_parts, ignore_index=True) if gradient_summary_parts else pd.DataFrame()
        leverage_df = pd.concat(leverage_parts, ignore_index=True) if leverage_parts else pd.DataFrame()
        leverage_sum_df = pd.DataFrame(leverage_summary_rows)
        evidence_df = pd.DataFrame(evidence_rows)
        evidence_v2_df = pd.DataFrame(evidence_v2_rows)
        specificity_df = pd.concat([p for p in specificity_parts if p is not None and not p.empty], ignore_index=True) if any(p is not None and not p.empty for p in specificity_parts) else pd.DataFrame()
        cross_null_df = pd.concat([p for p in cross_null_parts if p is not None and not p.empty], ignore_index=True) if any(p is not None and not p.empty for p in cross_null_parts) else pd.DataFrame()
        signflip_df = pd.DataFrame(signflip_rows)
        phase_profiles_df = pd.concat([p for p in phase_profile_parts if p is not None and not p.empty], ignore_index=True) if any(p is not None and not p.empty for p in phase_profile_parts) else pd.DataFrame()
        phase_diffs_df = pd.concat([p for p in phase_diff_parts if p is not None and not p.empty], ignore_index=True) if any(p is not None and not p.empty for p in phase_diff_parts) else pd.DataFrame()
        pattern_summary_df = pd.concat([p for p in pattern_summary_parts if p is not None and not p.empty], ignore_index=True) if any(p is not None and not p.empty for p in pattern_summary_parts) else pd.DataFrame()
        evidence_v3_df = pd.DataFrame(evidence_v3_rows)

        _safe_to_csv(mode_df, out_win / f"bootstrap_composite_mca_mode_summary_{wid}.csv")
        _safe_to_csv(score_df, out_win / f"bootstrap_composite_mca_scores_{wid}.csv")
        _safe_to_csv(obj_contrib, out_win / f"bootstrap_composite_mca_object_block_contribution_{wid}.csv")
        _safe_to_csv(coeff, out_win / f"bootstrap_composite_mca_pattern_coefficients_{wid}.csv")
        _safe_to_csv(perm_df, out_win / f"bootstrap_composite_mca_permutation_audit_{wid}.csv")
        _safe_to_csv(stability_df, out_win / f"bootstrap_composite_mca_mode_stability_{wid}.csv")
        _safe_to_csv(high_low_df, out_win / f"bootstrap_composite_mca_high_low_order_{wid}.csv")
        _safe_to_csv(quantile_df, out_win / f"bootstrap_composite_mca_quantile_sensitivity_{wid}.csv")
        _safe_to_csv(gradient_df, out_win / f"bootstrap_composite_mca_score_gradient_{wid}.csv")
        _safe_to_csv(gradient_summary_df, out_win / f"bootstrap_composite_mca_score_gradient_summary_{wid}.csv")
        _safe_to_csv(leverage_df, out_win / f"bootstrap_composite_mca_year_leverage_{wid}.csv")
        _safe_to_csv(leverage_sum_df, out_win / f"bootstrap_composite_mca_year_leverage_summary_{wid}.csv")
        _safe_to_csv(evidence_df, out_win / f"bootstrap_composite_mca_evidence_{wid}.csv")
        _safe_to_csv(evidence_v2_df, out_win / f"bootstrap_composite_mca_evidence_v2_{wid}.csv")
        _safe_to_csv(specificity_df, out_win / f"bootstrap_composite_mca_target_specificity_{wid}.csv")
        _safe_to_csv(cross_null_df, out_win / f"bootstrap_composite_mca_cross_target_null_{wid}.csv")
        _safe_to_csv(signflip_df, out_win / f"bootstrap_composite_mca_signflip_null_{wid}.csv")
        _safe_to_csv(phase_profiles_df, out_win / f"bootstrap_composite_mca_phase_composite_profiles_{wid}.csv")
        _safe_to_csv(phase_diffs_df, out_win / f"bootstrap_composite_mca_phase_composite_difference_{wid}.csv")
        _safe_to_csv(pattern_summary_df, out_win / f"bootstrap_composite_mca_pattern_summary_{wid}.csv")
        _safe_to_csv(evidence_v3_df, out_win / f"bootstrap_composite_mca_evidence_v3_{wid}.csv")
        _write_summary(out_win / f"bootstrap_composite_mca_summary_{wid}.md", evidence_df)

        all_mode.append(mode_df)
        all_scores.append(score_df)
        all_obj_contrib.append(obj_contrib)
        all_coeff.append(coeff)
        all_perm.append(perm_df)
        all_stability.append(stability_df)
        all_high_low.append(high_low_df)
        all_quantile.append(quantile_df)
        all_gradient.append(gradient_df)
        all_gradient_summary.append(gradient_summary_df)
        all_leverage.append(leverage_df)
        all_leverage_summary.append(leverage_sum_df)
        all_evidence.append(evidence_df)
        all_evidence_v2.append(evidence_v2_df)
        all_specificity.append(specificity_df)
        all_cross_null.append(cross_null_df)
        all_signflip.append(signflip_df)
        all_phase_profiles.append(phase_profiles_df)
        all_phase_diffs.append(phase_diffs_df)
        all_pattern_summary.append(pattern_summary_df)
        all_evidence_v3.append(evidence_v3_df)

    def cat(parts: List[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [p for p in parts if p is not None and not p.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

    _safe_to_csv(cat(all_targets), out_cross / "bootstrap_composite_mca_target_registry_all_windows.csv")
    _safe_to_csv(cat(all_replay_audit), out_cross / "v9_replay_bootstrap_regression_audit_all_windows.csv")
    _safe_to_csv(cat(all_mode), out_cross / "bootstrap_composite_mca_mode_summary_all_windows.csv")
    _safe_to_csv(cat(all_scores), out_cross / "bootstrap_composite_mca_scores_all_windows.csv")
    _safe_to_csv(cat(all_obj_contrib), out_cross / "bootstrap_composite_mca_object_block_contribution_all_windows.csv")
    _safe_to_csv(cat(all_coeff), out_cross / "bootstrap_composite_mca_pattern_coefficients_all_windows.csv")
    _safe_to_csv(cat(all_perm), out_cross / "bootstrap_composite_mca_permutation_audit_all_windows.csv")
    _safe_to_csv(cat(all_stability), out_cross / "bootstrap_composite_mca_mode_stability_all_windows.csv")
    _safe_to_csv(cat(all_high_low), out_cross / "bootstrap_composite_mca_high_low_order_all_windows.csv")
    _safe_to_csv(cat(all_quantile), out_cross / "bootstrap_composite_mca_quantile_sensitivity_all_windows.csv")
    _safe_to_csv(cat(all_gradient), out_cross / "bootstrap_composite_mca_score_gradient_all_windows.csv")
    _safe_to_csv(cat(all_gradient_summary), out_cross / "bootstrap_composite_mca_score_gradient_summary_all_windows.csv")
    _safe_to_csv(cat(all_leverage), out_cross / "bootstrap_composite_mca_year_leverage_all_windows.csv")
    _safe_to_csv(cat(all_leverage_summary), out_cross / "bootstrap_composite_mca_year_leverage_summary_all_windows.csv")
    _safe_to_csv(cat(all_nan_audit), out_cross / "bootstrap_composite_X_nan_feature_audit_all_windows.csv")
    evidence_all = cat(all_evidence)
    _safe_to_csv(evidence_all, out_cross / "bootstrap_composite_mca_evidence_all_windows.csv")
    evidence_v2_all = cat(all_evidence_v2)
    _safe_to_csv(evidence_v2_all, out_cross / "bootstrap_composite_mca_evidence_v2_all_windows.csv")
    _safe_to_csv(cat(all_target_y_tables), out_cross / "bootstrap_composite_mca_target_y_table_all_windows.csv")
    _safe_to_csv(cat(all_specificity), out_cross / "bootstrap_composite_mca_target_specificity_all_windows.csv")
    _safe_to_csv(cat(all_cross_null), out_cross / "bootstrap_composite_mca_cross_target_null_all_windows.csv")
    _safe_to_csv(cat(all_signflip), out_cross / "bootstrap_composite_mca_signflip_null_all_windows.csv")
    _safe_to_csv(cat(all_phase_profiles), out_cross / "bootstrap_composite_mca_phase_composite_profiles_all_windows.csv")
    _safe_to_csv(cat(all_phase_diffs), out_cross / "bootstrap_composite_mca_phase_composite_difference_all_windows.csv")
    _safe_to_csv(cat(all_pattern_summary), out_cross / "bootstrap_composite_mca_pattern_summary_all_windows.csv")
    _safe_to_csv(cat(all_evidence_v3), out_cross / "bootstrap_composite_mca_evidence_v3_all_windows.csv")
    _write_summary(out_cross / "bootstrap_composite_mca_summary_all_windows.md", evidence_all)

    _write_json({
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage_root": str(stage_root),
        "windows": cfg.windows,
        "modifies_v9": False,
        "reads_v9_outputs": True,
        "requires_v9_1_c_helpers": True,
        "uses_single_year_peak": False,
        "bootstrap_samples_are_independent_physical_years": False,
        "method_role": "bootstrap-space coupling diagnostic, not physical year-type identification",
        "boundary_nan_handling": "drop all-NaN feature columns before standardization and SVD",
        "partial_nan_handling": "column-mean imputation after all-NaN removal",
        "zero_variance_handling": "drop zero-variance feature columns",
        "object_block_equal_weight_after_mask": True,
        "evidence_refinement_hotfix01": True,
        "hotfix02_specificity_interpretability": {
            "target_specificity_audit": cfg.enable_target_specificity,
            "cross_target_null": cfg.enable_cross_target_null,
            "signflip_null": cfg.enable_signflip_null,
            "phase_composite_profiles": cfg.enable_phase_composite,
            "pattern_summary": cfg.enable_pattern_summary,
            "evidence_v3": True
        },
        "config": asdict(cfg),
    }, out_cross / "run_meta.json")
    _log(f"[V9.1_f] Done. Outputs: {out_root}")
