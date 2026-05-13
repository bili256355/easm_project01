"""
V9.1_e targeted SVD / MCA order-mode audit.

Purpose
-------
Read-only V9/V9.1 audit branch. It uses V9.1_c bootstrap year-influence
scores as target variables and extracts target-guided SVD modes from the same
whole-window, multi-object anomaly feature matrix used by V9.1_d.

This is NOT ordinary EOF and NOT unsupervised year clustering. It asks whether
there is a multi-object window-behaviour anomaly direction that specifically
co-varies with V9 peak/order heterogeneity.

Boundaries
----------
* Does not modify V9 or any previous V9.1 outputs.
* Does not use single-year peak as a target or clustering input.
* Does not assign physical regime names to modes.
* Group-level peak/order checks are multi-year group composites, not single-year peaks.
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

VERSION = "v9_1_e_targeted_svd_order_mode_audit"
OUTPUT_TAG = "targeted_svd_order_mode_audit_v9_1_e"
DEFAULT_WINDOWS = ["W045", "W081", "W113", "W160"]
OBJECTS = ["P", "V", "H", "Je", "Jw"]
PRIORITY_PAIRS = [
    ("P", "V"), ("Jw", "P"), ("Jw", "V"), ("Jw", "H"), ("Jw", "Je"),
    ("H", "Jw"), ("Je", "Jw"), ("V", "Je"), ("H", "Je"), ("P", "H"), ("P", "Je"),
]


@dataclass
class V91EConfig:
    windows: List[str] = field(default_factory=lambda: list(DEFAULT_WINDOWS))
    objects: List[str] = field(default_factory=lambda: list(OBJECTS))
    priority_pairs: List[Tuple[str, str]] = field(default_factory=lambda: list(PRIORITY_PAIRS))

    max_pair_targets_per_window: int = 6
    max_object_targets_per_window: int = 3
    min_pair_abs_effect: float = 0.10
    min_object_peak_day_effect: float = 1.0
    min_object_prob_effect: float = 0.10
    min_target_std: float = 1.0e-9

    group_bootstrap_n: int = 500
    debug_group_bootstrap_n: int = 50
    perm_n: int = 500
    debug_perm_n: int = 50
    min_group_size_for_peak: int = 10

    evidence_usable: float = 0.90
    evidence_credible: float = 0.95
    evidence_strict: float = 0.99
    loo_pattern_corr_caution: float = 0.50
    loo_pattern_corr_good: float = 0.70
    cv_corr_usable: float = 0.30
    cv_corr_credible: float = 0.45

    debug: bool = False

    # Performance-only controls. By default, expensive group-level V9 peak
    # checks are run only for targets whose targeted-SVD screen passes at
    # least the usable mode gate. This does not change the targeted-SVD fit;
    # it only avoids validating already unsupported targets with costly
    # phase-group bootstraps. Set V9_1E_FORCE_ALL_GROUP_PEAK_CHECKS=1 to
    # reproduce the older exhaustive behavior.
    skip_group_peak_for_unsupported_modes: bool = True
    eligible_group_peak_statuses: List[str] = field(default_factory=lambda: [
        "usable_targeted_mode", "credible_targeted_mode", "strict_targeted_mode"
    ])
    perm_batch_size: int = 128

    @classmethod
    def from_env(cls) -> "V91EConfig":
        cfg = cls()
        if os.environ.get("V9_1E_DEBUG"):
            cfg.debug = True
            cfg.group_bootstrap_n = int(os.environ.get("V9_1E_DEBUG_GROUP_BOOTSTRAP_N", cfg.debug_group_bootstrap_n))
            cfg.perm_n = int(os.environ.get("V9_1E_DEBUG_PERM_N", cfg.debug_perm_n))
        if os.environ.get("V9_1E_GROUP_BOOTSTRAP_N"):
            cfg.group_bootstrap_n = int(os.environ["V9_1E_GROUP_BOOTSTRAP_N"])
        if os.environ.get("V9_1E_PERM_N"):
            cfg.perm_n = int(os.environ["V9_1E_PERM_N"])
        if os.environ.get("V9_1E_PERM_BATCH_SIZE"):
            cfg.perm_batch_size = int(os.environ["V9_1E_PERM_BATCH_SIZE"])
        if os.environ.get("V9_1E_FORCE_ALL_GROUP_PEAK_CHECKS"):
            if _env_bool("V9_1E_FORCE_ALL_GROUP_PEAK_CHECKS", False):
                cfg.skip_group_peak_for_unsupported_modes = False
        if os.environ.get("V9_1E_SKIP_GROUP_PEAK_FOR_UNSUPPORTED"):
            cfg.skip_group_peak_for_unsupported_modes = _env_bool("V9_1E_SKIP_GROUP_PEAK_FOR_UNSUPPORTED", cfg.skip_group_peak_for_unsupported_modes)
        if os.environ.get("V9_1E_MIN_GROUP_SIZE"):
            cfg.min_group_size_for_peak = int(os.environ["V9_1E_MIN_GROUP_SIZE"])
        if os.environ.get("V9_1E_WINDOWS"):
            cfg.windows = [x.strip() for x in os.environ["V9_1E_WINDOWS"].replace(";", ",").split(",") if x.strip()]
        if os.environ.get("V9_1E_MAX_PAIR_TARGETS"):
            cfg.max_pair_targets_per_window = int(os.environ["V9_1E_MAX_PAIR_TARGETS"])
        if os.environ.get("V9_1E_MAX_OBJECT_TARGETS"):
            cfg.max_object_targets_per_window = int(os.environ["V9_1E_MAX_OBJECT_TARGETS"])
        return cfg


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _import_d_module(v91_root: Path):
    src = Path(v91_root) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from stage_partition_v9_1 import eof_transition_mode_audit_v9_1_d as dmod
    except Exception as exc:
        raise ImportError(
            "V9.1_e requires V9.1_d helper module eof_transition_mode_audit_v9_1_d.py. "
            "Install/apply V9.1_d first, then rerun V9.1_e."
        ) from exc
    return dmod


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa = a[mask] - np.nanmean(a[mask]); bb = b[mask] - np.nanmean(b[mask])
    da = np.sqrt(np.sum(aa * aa)); db = np.sqrt(np.sum(bb * bb))
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
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return _corr(_rankdata(a[mask]), _rankdata(b[mask]))


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    n = float(np.sqrt(np.sum(v * v)))
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def _evidence_level_from_prob(p: float, cfg: V91EConfig) -> str:
    if not np.isfinite(p):
        return "insufficient"
    if p >= cfg.evidence_strict:
        return "strict_99"
    if p >= cfg.evidence_credible:
        return "credible_95"
    if p >= cfg.evidence_usable:
        return "usable_90"
    return "weak_or_none"


def _order_level(prob: float, cfg: V91EConfig) -> str:
    return _evidence_level_from_prob(prob, cfg)


def _numeric_year_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "year" not in df.columns:
        return df
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out[np.isfinite(out["year"])].copy()
    out["year"] = out["year"].astype(int)
    return out


def _load_v91c_tables(v91_root: Path) -> Dict[str, pd.DataFrame]:
    c_out = v91_root / "outputs" / "bootstrap_year_influence_audit_v9_1_c" / "cross_window"
    out = {}
    for key, fname in {
        "object_influence": "object_peak_year_influence_all_windows.csv",
        "pairwise_influence": "pairwise_order_year_influence_all_windows.csv",
        "summary": "bootstrap_year_influence_summary_all_windows.csv",
    }.items():
        f = c_out / fname
        out[key] = _numeric_year_column(pd.read_csv(f)) if f.exists() else pd.DataFrame()
    return out


def _load_v9_order(stage_root: Path) -> pd.DataFrame:
    f = stage_root / "V9" / "outputs" / "peak_all_windows_v9_a" / "cross_window" / "cross_window_pairwise_peak_order.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([str(a), str(b)], key=lambda x: OBJECTS.index(x) if x in OBJECTS else 99))  # type: ignore


def _pair_y_from_rows(sub: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Return year,y for target = effect_on_a_earlier - effect_on_b_earlier.

    The V9.1_c table stores pair orientation object_A/object_B. If the stored
    orientation is reversed relative to requested target, signs are converted.
    """
    rows = []
    for _, r in sub.iterrows():
        oa, ob = str(r.get("object_A")), str(r.get("object_B"))
        eA = float(r.get("presence_effect_on_A_earlier", np.nan))
        eB = float(r.get("presence_effect_on_B_earlier", np.nan))
        if oa == a and ob == b:
            y = eA - eB
        elif oa == b and ob == a:
            y = eB - eA
        else:
            continue
        rows.append({"year": int(r["year"]), "target_y": y})
    return pd.DataFrame(rows)


def _select_targets(v91c: Dict[str, pd.DataFrame], cfg: V91EConfig) -> pd.DataFrame:
    rows: List[dict] = []
    pair_df = v91c.get("pairwise_influence", pd.DataFrame())
    obj_df = v91c.get("object_influence", pd.DataFrame())

    if pair_df is not None and not pair_df.empty:
        pair_df = _numeric_year_column(pair_df)
        for wid, gw in pair_df.groupby("window_id"):
            if wid not in cfg.windows:
                continue
            cand = []
            for (oa, ob), sub in gw.groupby(["object_A", "object_B"]):
                max_eff = float(np.nanmax(np.abs(np.r_[sub.get("presence_effect_on_A_earlier", pd.Series(dtype=float)).to_numpy(dtype=float), sub.get("presence_effect_on_B_earlier", pd.Series(dtype=float)).to_numpy(dtype=float)])))
                max_pct = float(np.nanmax(sub.get("max_permutation_percentile", pd.Series([np.nan])).to_numpy(dtype=float)))
                priority = 1 if (str(oa), str(ob)) in cfg.priority_pairs or (str(ob), str(oa)) in cfg.priority_pairs else 0
                if max_eff >= cfg.min_pair_abs_effect or priority:
                    cand.append((priority, max_eff, max_pct, str(oa), str(ob)))
            cand = sorted(cand, key=lambda x: (x[0], x[1], x[2]), reverse=True)[:cfg.max_pair_targets_per_window]
            for _priority, max_eff, max_pct, oa, ob in cand:
                rows.append({
                    "window_id": wid,
                    "target_name": f"{wid}_{oa}_earlier_minus_{ob}_earlier",
                    "target_type": "pairwise_order_influence",
                    "object_A": oa,
                    "object_B": ob,
                    "target_definition": f"presence_effect_on_{oa}_earlier - presence_effect_on_{ob}_earlier",
                    "selection_max_abs_effect": max_eff,
                    "selection_max_permutation_percentile": max_pct,
                    "selection_role": "priority_or_high_effect_pair_target_from_v9_1_c",
                })

    if obj_df is not None and not obj_df.empty:
        obj_df = _numeric_year_column(obj_df)
        for wid, gw in obj_df.groupby("window_id"):
            if wid not in cfg.windows:
                continue
            cand = []
            for obj, sub in gw.groupby("object"):
                day_eff = float(np.nanmax(np.abs(sub.get("presence_effect_on_peak_day", pd.Series(dtype=float)).to_numpy(dtype=float))))
                prob_eff = float(np.nanmax(np.abs(np.r_[sub.get("presence_effect_on_late_probability", pd.Series(dtype=float)).to_numpy(dtype=float), sub.get("presence_effect_on_early_probability", pd.Series(dtype=float)).to_numpy(dtype=float)])))
                max_pct = float(np.nanmax(sub.get("max_permutation_percentile", pd.Series([np.nan])).to_numpy(dtype=float)))
                if day_eff >= cfg.min_object_peak_day_effect or prob_eff >= cfg.min_object_prob_effect:
                    cand.append((max(day_eff / max(cfg.min_object_peak_day_effect, 1e-9), prob_eff / max(cfg.min_object_prob_effect, 1e-9)), day_eff, prob_eff, max_pct, str(obj)))
            cand = sorted(cand, key=lambda x: (x[0], x[3]), reverse=True)[:cfg.max_object_targets_per_window]
            for _, day_eff, prob_eff, max_pct, obj in cand:
                rows.append({
                    "window_id": wid,
                    "target_name": f"{wid}_{obj}_late_minus_early_peak_influence",
                    "target_type": "object_peak_shift_influence",
                    "object_A": obj,
                    "object_B": "",
                    "target_definition": "presence_effect_on_late_probability - presence_effect_on_early_probability; fallback=presence_effect_on_peak_day",
                    "selection_max_abs_effect": max(day_eff, prob_eff),
                    "selection_max_permutation_percentile": max_pct,
                    "selection_role": "high_effect_object_peak_target_from_v9_1_c",
                })
    return pd.DataFrame(rows)


def _target_y_for_registry_row(row: pd.Series, v91c: Dict[str, pd.DataFrame], years: np.ndarray, cfg: V91EConfig) -> pd.DataFrame:
    wid = row["window_id"]
    ttype = row["target_type"]
    if ttype == "pairwise_order_influence":
        pair_df = v91c.get("pairwise_influence", pd.DataFrame())
        if pair_df is None or pair_df.empty:
            return pd.DataFrame()
        a, b = str(row["object_A"]), str(row["object_B"])
        sub = pair_df[pair_df["window_id"].astype(str).eq(str(wid))]
        sub = sub[((sub["object_A"].astype(str).eq(a)) & (sub["object_B"].astype(str).eq(b))) |
                  ((sub["object_A"].astype(str).eq(b)) & (sub["object_B"].astype(str).eq(a)))]
        ydf = _pair_y_from_rows(sub, a, b)
    else:
        obj_df = v91c.get("object_influence", pd.DataFrame())
        if obj_df is None or obj_df.empty:
            return pd.DataFrame()
        obj = str(row["object_A"])
        sub = obj_df[obj_df["window_id"].astype(str).eq(str(wid)) & obj_df["object"].astype(str).eq(obj)]
        rows = []
        for _, r in sub.iterrows():
            late = r.get("presence_effect_on_late_probability", np.nan)
            early = r.get("presence_effect_on_early_probability", np.nan)
            peak = r.get("presence_effect_on_peak_day", np.nan)
            if np.isfinite(late) or np.isfinite(early):
                y = float(np.nan_to_num(late, nan=0.0) - np.nan_to_num(early, nan=0.0))
                source = "late_minus_early_probability_effect"
            else:
                y = float(peak)
                source = "peak_day_effect_fallback"
            rows.append({"year": int(r["year"]), "target_y": y, "target_y_source": source})
        ydf = pd.DataFrame(rows)
    # align to MEOF years; missing years get NaN
    full = pd.DataFrame({"year": [int(y) for y in years]})
    if ydf.empty:
        full["target_y"] = np.nan
        return full
    ydf = ydf.groupby("year", as_index=False)["target_y"].mean()
    full = full.merge(ydf, on="year", how="left")
    return full


def _fit_targeted_mode(M: np.ndarray, y: np.ndarray, cfg: V91EConfig) -> Dict[str, object]:
    X = np.asarray(M, dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 8 or X.shape[1] == 0:
        return {"mode_vector": np.zeros(X.shape[1]), "score": np.full(X.shape[0], np.nan), "status": "insufficient_target_years"}
    yc = y.copy()
    yc[mask] = yc[mask] - np.nanmean(yc[mask])
    ysd = np.nanstd(yc[mask])
    if not np.isfinite(ysd) or ysd <= cfg.min_target_std:
        return {"mode_vector": np.zeros(X.shape[1]), "score": np.full(X.shape[0], np.nan), "status": "target_variance_too_small"}
    yz = np.full_like(yc, np.nan, dtype=float)
    yz[mask] = yc[mask] / ysd
    c = X[mask].T @ yz[mask]
    u = _norm_vec(c)
    score = X @ u
    if _corr(score, yz) < 0:
        u = -u
        score = -score
    return {"mode_vector": u, "score": score, "target_z": yz, "status": "ok"}


def _permutation_audit(M: np.ndarray, y: np.ndarray, obs_corr: float, cfg: V91EConfig, seed: int) -> Tuple[float, float]:
    """Permutation audit for targeted-SVD score/y relation.

    Hotfix02 vectorizes permutations in batches. It preserves the same target
    calculation as the previous loop implementation (permute y, refit the
    one-dimensional targeted mode, record |corr(score, y_perm)|), but avoids
    calling _fit_targeted_mode hundreds of times in Python.
    """
    if cfg.perm_n <= 0 or not np.isfinite(obs_corr):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    vals = y[mask].astype(float)
    if vals.size < 8:
        return np.nan, np.nan

    X = np.asarray(M, dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    Xv = X[mask]
    if Xv.size == 0 or Xv.shape[1] == 0:
        return np.nan, np.nan

    null_vals: List[float] = []
    batch_size = max(1, int(getattr(cfg, "perm_batch_size", 128)))
    n_done = 0
    while n_done < int(cfg.perm_n):
        b = min(batch_size, int(cfg.perm_n) - n_done)
        # Build batch of permuted target vectors: n_valid × b.
        Y = np.empty((vals.size, b), dtype=float)
        for j in range(b):
            Y[:, j] = rng.permutation(vals)
        Yc = Y - np.mean(Y, axis=0, keepdims=True)
        Ys = np.std(Yc, axis=0, ddof=0)
        valid_cols = Ys > cfg.min_target_std
        if not np.any(valid_cols):
            n_done += b
            continue
        Yz = Yc[:, valid_cols] / Ys[valid_cols][None, :]

        C = Xv.T @ Yz  # feature × batch
        norms = np.sqrt(np.sum(C * C, axis=0))
        valid_modes = norms > 1.0e-12
        if not np.any(valid_modes):
            n_done += b
            continue
        U = C[:, valid_modes] / norms[valid_modes][None, :]
        Yz2 = Yz[:, valid_modes]
        S = Xv @ U
        S = S - np.mean(S, axis=0, keepdims=True)
        # Yz2 is already zero-mean/std-scaled, but use full denominator for exact corr.
        num = np.sum(S * Yz2, axis=0)
        den = np.sqrt(np.sum(S * S, axis=0) * np.sum(Yz2 * Yz2, axis=0))
        corr = np.full(num.shape, np.nan, dtype=float)
        ok = den > 1.0e-12
        corr[ok] = num[ok] / den[ok]
        null_vals.extend(np.abs(corr[np.isfinite(corr)]).tolist())
        n_done += b

    if not null_vals:
        return np.nan, np.nan
    null = np.asarray(null_vals, dtype=float)
    percentile = float(np.mean(null <= abs(obs_corr)))
    p_emp = float(np.mean(null >= abs(obs_corr)))
    return percentile, p_emp


def _loo_stability(M: np.ndarray, y: np.ndarray, u_full: np.ndarray, cfg: V91EConfig) -> pd.DataFrame:
    rows = []
    n = M.shape[0]
    for i in range(n):
        y2 = y.copy()
        y2[i] = np.nan
        fit = _fit_targeted_mode(M, y2, cfg)
        u = np.asarray(fit.get("mode_vector", np.zeros_like(u_full)), dtype=float)
        pat_corr = _corr(u_full, u)
        if np.isfinite(pat_corr) and pat_corr < 0:
            pat_corr = -pat_corr
        rows.append({"row_index": i, "loo_pattern_corr": pat_corr, "loo_status": fit.get("status", "unknown")})
    return pd.DataFrame(rows)


def _cross_validated_scores(M: np.ndarray, y: np.ndarray, cfg: V91EConfig) -> Tuple[np.ndarray, pd.DataFrame]:
    n = M.shape[0]
    scores = np.full(n, np.nan)
    rows = []
    for i in range(n):
        y2 = y.copy()
        y2[i] = np.nan
        fit = _fit_targeted_mode(M, y2, cfg)
        if fit.get("status") == "ok":
            u = np.asarray(fit["mode_vector"], dtype=float)
            scores[i] = float(np.dot(M[i], u))
        rows.append({"row_index": i, "cv_score": scores[i], "cv_fit_status": fit.get("status", "unknown")})
    return scores, pd.DataFrame(rows)


def _assign_target_phase_groups(window_id: str, target_name: str, years: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    s = np.asarray(score, dtype=float)
    finite = np.isfinite(s)
    rows = []
    if finite.sum() < 6:
        for yi, year in enumerate(years):
            rows.append({"window_id": window_id, "target_name": target_name, "mode": 1, "phase_group": "insufficient", "year": int(year), "mode_score": s[yi] if yi < len(s) else np.nan})
        return pd.DataFrame(rows)
    q1 = float(np.nanquantile(s[finite], 1/3))
    q2 = float(np.nanquantile(s[finite], 2/3))
    for yi, year in enumerate(years):
        val = s[yi]
        if not np.isfinite(val):
            grp = "missing"
        elif val <= q1:
            grp = "target_mode_low"
        elif val >= q2:
            grp = "target_mode_high"
        else:
            grp = "target_mode_mid"
        rows.append({"window_id": window_id, "target_name": target_name, "mode": 1, "phase_group": grp, "year": int(year), "mode_score": float(val) if np.isfinite(val) else np.nan, "q33": q1, "q67": q2})
    return pd.DataFrame(rows)


def _subset_profiles(profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], idx: Sequence[int]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for obj, (prof, lat, weights) in profiles.items():
        out[obj] = (np.asarray(prof)[list(idx)], lat, weights)
    return out


def _add_order_levels(order_df: pd.DataFrame, cfg: V91EConfig) -> pd.DataFrame:
    if order_df is None or order_df.empty:
        return pd.DataFrame()
    out = order_df.copy()
    def best(row):
        pa = row.get("bootstrap_prob_A_earlier", np.nan)
        pb = row.get("bootstrap_prob_B_earlier", np.nan)
        if np.nan_to_num(pa, nan=-1) >= np.nan_to_num(pb, nan=-1):
            return "A_earlier", pa, _order_level(pa, cfg)
        return "B_earlier", pb, _order_level(pb, cfg)
    vals = out.apply(best, axis=1, result_type="expand")
    out["targeted_best_direction"] = vals[0]
    out["targeted_best_prob"] = vals[1]
    out["targeted_order_level"] = vals[2]
    return out


def _run_target_group_peak(dmod, v7multi, profiles, years, scope, phase_groups: pd.DataFrame, cfg_v7, cfg: V91EConfig) -> Dict[str, pd.DataFrame]:
    obj_parts, boot_parts, order_parts, sync_parts, skipped_rows = [], [], [], [], []
    if phase_groups.empty:
        return {"object_peak": pd.DataFrame(), "object_bootstrap": pd.DataFrame(), "order": pd.DataFrame(), "sync": pd.DataFrame(), "skipped": pd.DataFrame()}
    year_list = [int(y) for y in years.tolist()]
    for (target_name, group), sub in phase_groups.groupby(["target_name", "phase_group"]):
        if group not in ["target_mode_high", "target_mode_mid", "target_mode_low"]:
            continue
        if cfg.skip_group_peak_for_unsupported_modes and "phase_group_peak_eligible" in sub.columns:
            eligible = bool(pd.Series(sub["phase_group_peak_eligible"]).fillna(False).astype(bool).any())
            if not eligible:
                skipped_rows.append({
                    "window_id": scope.window_id,
                    "target_name": target_name,
                    "phase_group": group,
                    "phase_group_size": int(len(sub)),
                    "skip_reason": "targeted_mode_not_eligible_for_costly_group_peak_check",
                    "targeted_mode_status": str(sub.get("targeted_mode_status", pd.Series([""])).iloc[0]) if len(sub) else "",
                    "performance_gate": "skip_group_peak_for_unsupported_modes",
                })
                continue
        idx = [int(np.where(years == y)[0][0]) for y in sub["year"].astype(int).to_numpy() if int(y) in year_list]
        group_size = len(idx)
        if group_size < cfg.min_group_size_for_peak:
            skipped_rows.append({
                "window_id": scope.window_id,
                "target_name": target_name,
                "phase_group": group,
                "phase_group_size": int(group_size),
                "skip_reason": "target_phase_group_size_below_min_group_size_for_peak",
                "min_group_size_for_peak": int(cfg.min_group_size_for_peak),
            })
            continue
        _log(f"    Target group peak {scope.window_id}: {target_name} {group} n={group_size}")
        sub_profiles = _subset_profiles(profiles, idx)
        try:
            _score_df, _cand_df, selection_df, _selected_delta_df, boot_peak_days_df = v7multi._run_detector_and_bootstrap(sub_profiles, scope, cfg_v7)
            timing_audit_df, tau_df = v7multi._estimate_timing_resolution(selection_df, boot_peak_days_df, cfg_v7, scope)
            order_df = v7multi._pairwise_peak_order(selection_df, boot_peak_days_df, scope)
            sync_df = v7multi._pairwise_synchrony(order_df, boot_peak_days_df, tau_df, scope)
        except Exception as exc:
            skipped_rows.append({
                "window_id": scope.window_id,
                "target_name": target_name,
                "phase_group": group,
                "phase_group_size": int(group_size),
                "skip_reason": "target_group_peak_runtime_error",
                "error": str(exc),
            })
            continue
        add = {
            "window_id": scope.window_id,
            "target_name": target_name,
            "phase_group": group,
            "phase_group_size": int(group_size),
            "years_in_phase_group": ";".join(str(x) for x in sorted(sub["year"].astype(int).to_numpy())),
            "group_role": "targeted_SVD_phase_group_multi_year_peak_not_single_year_peak",
        }
        for df, parts in [(selection_df, obj_parts), (timing_audit_df, boot_parts), (order_df, order_parts), (sync_df, sync_parts)]:
            d = df.copy()
            for k, v in add.items():
                d[k] = v
            parts.append(d)
    order_all = pd.concat(order_parts, ignore_index=True) if order_parts else pd.DataFrame()
    order_all = _add_order_levels(order_all, cfg)
    return {
        "object_peak": pd.concat(obj_parts, ignore_index=True) if obj_parts else pd.DataFrame(),
        "object_bootstrap": pd.concat(boot_parts, ignore_index=True) if boot_parts else pd.DataFrame(),
        "order": order_all,
        "sync": pd.concat(sync_parts, ignore_index=True) if sync_parts else pd.DataFrame(),
        "skipped": pd.DataFrame(skipped_rows),
    }


def _targeted_mode_status(perm_percentile: float, loo_median: float, cv_corr: float, cfg: V91EConfig) -> str:
    if not np.isfinite(perm_percentile):
        return "not_supported"
    level = _evidence_level_from_prob(perm_percentile, cfg)
    if level == "weak_or_none":
        return "not_supported"
    if np.isfinite(loo_median) and loo_median < cfg.loo_pattern_corr_caution:
        return "overfit_or_unstable_mode_caution"
    if np.isfinite(cv_corr) and cv_corr < 0:
        return "overfit_caution_cv_negative"
    if level == "strict_99" and (not np.isfinite(cv_corr) or cv_corr >= cfg.cv_corr_credible) and (not np.isfinite(loo_median) or loo_median >= cfg.loo_pattern_corr_good):
        return "strict_targeted_mode"
    if level in ["strict_99", "credible_95"]:
        return "credible_targeted_mode"
    return "usable_targeted_mode"


def _mode_evidence_for_targets(reg: pd.DataFrame, mode_summary: pd.DataFrame, group_order: pd.DataFrame, v9_order: pd.DataFrame, cfg: V91EConfig) -> pd.DataFrame:
    rows = []
    if reg.empty:
        return pd.DataFrame()
    for _, r in reg.iterrows():
        wid, tn = r["window_id"], r["target_name"]
        ms = mode_summary[(mode_summary["window_id"].astype(str) == str(wid)) & (mode_summary["target_name"].astype(str) == str(tn))]
        mode_status = ms["mode_status"].iloc[0] if not ms.empty and "mode_status" in ms.columns else "not_supported"
        target_type = r["target_type"]
        oa, ob = str(r.get("object_A", "")), str(r.get("object_B", ""))
        if target_type != "pairwise_order_influence" or group_order.empty:
            rows.append({
                "window_id": wid,
                "target_name": tn,
                "target_type": target_type,
                "target_pair_or_object": oa,
                "targeted_mode_status": mode_status,
                "evidence_level": "object_target_mode_only_no_pair_order_evidence",
                "recommended_interpretation": "Inspect targeted mode/year scores and object-level peak groups; pairwise order evidence not applicable.",
            })
            continue
        go = group_order[(group_order["window_id"].astype(str) == str(wid)) & (group_order["target_name"].astype(str) == str(tn))]
        # Match pair in either orientation.
        go_pair = go[((go["object_A"].astype(str) == oa) & (go["object_B"].astype(str) == ob)) |
                     ((go["object_A"].astype(str) == ob) & (go["object_B"].astype(str) == oa))]
        def best_group(group_name: str):
            gg = go_pair[go_pair["phase_group"].astype(str).eq(group_name)]
            if gg.empty:
                return "missing", np.nan, ""
            # Convert orientation to requested A/B when possible.
            row = gg.iloc[0]
            if str(row["object_A"]) == oa and str(row["object_B"]) == ob:
                pa = row.get("bootstrap_prob_A_earlier", np.nan); pb = row.get("bootstrap_prob_B_earlier", np.nan)
                dir_label = f"{oa}_earlier" if np.nan_to_num(pa, nan=-1) >= np.nan_to_num(pb, nan=-1) else f"{ob}_earlier"
                prob = max(np.nan_to_num(pa, nan=-1), np.nan_to_num(pb, nan=-1))
            else:
                # Stored reversed: object_A is target B.
                pa_stored = row.get("bootstrap_prob_A_earlier", np.nan); pb_stored = row.get("bootstrap_prob_B_earlier", np.nan)
                # A earlier in target orientation corresponds to stored B earlier.
                prob_a = pb_stored; prob_b = pa_stored
                dir_label = f"{oa}_earlier" if np.nan_to_num(prob_a, nan=-1) >= np.nan_to_num(prob_b, nan=-1) else f"{ob}_earlier"
                prob = max(np.nan_to_num(prob_a, nan=-1), np.nan_to_num(prob_b, nan=-1))
            return _order_level(prob, cfg), prob, dir_label
        high_level, high_prob, high_dir = best_group("target_mode_high")
        low_level, low_prob, low_dir = best_group("target_mode_low")
        mid_level, mid_prob, mid_dir = best_group("target_mode_mid")
        opposite = bool(high_dir and low_dir and high_dir != low_dir and high_dir != "" and low_dir != "")
        best_level_rank = {"strict_99": 4, "credible_95": 3, "usable_90": 2, "weak_or_none": 1, "insufficient": 0, "missing": 0}
        best_group_rank = max(best_level_rank.get(high_level, 0), best_level_rank.get(low_level, 0), best_level_rank.get(mid_level, 0))
        if mode_status in ["strict_targeted_mode", "credible_targeted_mode"] and best_group_rank >= 3 and opposite:
            evidence = "strong_targeted_mode_candidate"
        elif mode_status in ["strict_targeted_mode", "credible_targeted_mode", "usable_targeted_mode"] and best_group_rank >= 3:
            evidence = "moderate_targeted_mode_candidate"
        elif mode_status in ["strict_targeted_mode", "credible_targeted_mode", "usable_targeted_mode"] and best_group_rank >= 2:
            evidence = "weak_targeted_mode_hint"
        elif "overfit" in mode_status or "unstable" in mode_status:
            evidence = "overfit_or_extreme_year_caution"
        else:
            evidence = "not_supported"
        rows.append({
            "window_id": wid,
            "target_name": tn,
            "target_type": target_type,
            "target_pair_or_object": f"{oa}-{ob}",
            "full_sample_v9_order_level": "see_V9_cross_window_pairwise_peak_order",
            "targeted_mode_status": mode_status,
            "high_group_order_level": high_level,
            "high_group_best_prob": high_prob,
            "high_group_direction": high_dir,
            "mid_group_order_level": mid_level,
            "mid_group_best_prob": mid_prob,
            "mid_group_direction": mid_dir,
            "low_group_order_level": low_level,
            "low_group_best_prob": low_prob,
            "low_group_direction": low_dir,
            "opposite_order_between_high_low": opposite,
            "group_internal_stabilization": best_group_rank >= 2,
            "does_targeted_mode_explain_order_instability": evidence not in ["not_supported", "overfit_or_extreme_year_caution"],
            "evidence_level": evidence,
            "recommended_interpretation": _recommendation(evidence),
        })
    return pd.DataFrame(rows)


def _recommendation(evidence: str) -> str:
    if evidence == "strong_targeted_mode_candidate":
        return "Target-guided mode is a strong candidate for explaining this peak/order heterogeneity; inspect composites and physical fields before naming any regime."
    if evidence == "moderate_targeted_mode_candidate":
        return "Target-guided mode provides moderate evidence; use as candidate and verify with composites/field diagnostics."
    if evidence == "weak_targeted_mode_hint":
        return "Weak target-guided hint only; do not treat as a resolved mode."
    if evidence == "overfit_or_extreme_year_caution":
        return "Mode may be overfit or high-leverage-year dominated; prioritize influence-year drilldown."
    return "No supported targeted SVD explanation under this feature/target design."


def _write_summary(path: Path, evidence: pd.DataFrame, mode_summary: pd.DataFrame, cfg: V91EConfig) -> None:
    lines = [
        "# V9.1_e targeted SVD order-mode audit summary",
        "",
        "This branch uses V9.1_c year-influence scores as target variables and extracts target-guided SVD/MCA modes from whole-window multi-object anomaly features.",
        "It does not use single-year peak and does not assign physical regime names.",
        "",
        "## Evidence counts",
    ]
    if evidence is not None and not evidence.empty and "evidence_level" in evidence.columns:
        for k, v in evidence["evidence_level"].value_counts(dropna=False).items():
            lines.append(f"- {k}: {int(v)}")
    else:
        lines.append("- No evidence rows generated.")
    lines += ["", "## Mode status counts"]
    if mode_summary is not None and not mode_summary.empty and "mode_status" in mode_summary.columns:
        for k, v in mode_summary["mode_status"].value_counts(dropna=False).items():
            lines.append(f"- {k}: {int(v)}")
    else:
        lines.append("- No mode summary rows generated.")
    lines += [
        "",
        "## Interpretation boundary",
        "Targeted SVD modes are order-relevant statistical candidate axes. They are not physical types unless later composite/field diagnostics support that interpretation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_targeted_svd_order_mode_audit_v9_1_e(v91_root: Path) -> None:
    v91_root = Path(v91_root).resolve()
    stage_root = _stage_root_from_v91(v91_root)
    cfg = V91EConfig.from_env()
    out_root = _ensure_dir(v91_root / "outputs" / OUTPUT_TAG)
    out_cross = _ensure_dir(out_root / "cross_window")

    _log("[V9.1_e] Loading V9.1_d helpers, V9 context, and V9.1_c target tables...")
    dmod = _import_d_module(v91_root)
    v7multi = dmod._import_v7_module(stage_root)
    d_cfg = dmod.V91DConfig.from_env()
    d_cfg.windows = cfg.windows
    d_cfg.group_bootstrap_n = cfg.group_bootstrap_n
    d_cfg.min_group_size_for_peak = cfg.min_group_size_for_peak
    v7_cfg = dmod._make_v7_cfg(v7multi, d_cfg, stage_root)
    scopes = dmod._window_scopes_from_v7(v7multi, stage_root, v7_cfg, cfg.windows)
    profiles, years, profile_audit = dmod._load_profiles(v7multi, stage_root, v7_cfg)
    v91c = _load_v91c_tables(v91_root)
    v9_order = _load_v9_order(stage_root)
    target_registry = _select_targets(v91c, cfg)

    _safe_to_csv(target_registry, out_cross / "targeted_svd_target_registry_all_windows.csv")
    _safe_to_csv(profile_audit, out_cross / "targeted_svd_profile_source_audit.csv")

    all_mode_summary: List[pd.DataFrame] = []
    all_scores: List[pd.DataFrame] = []
    all_obj_contrib: List[pd.DataFrame] = []
    all_coeff: List[pd.DataFrame] = []
    all_perm: List[pd.DataFrame] = []
    all_loo: List[pd.DataFrame] = []
    all_cv: List[pd.DataFrame] = []
    all_group_obj: List[pd.DataFrame] = []
    all_group_order: List[pd.DataFrame] = []
    all_group_sync: List[pd.DataFrame] = []
    all_skipped: List[pd.DataFrame] = []
    all_evidence: List[pd.DataFrame] = []

    for scope in scopes:
        wid = scope.window_id
        if wid not in cfg.windows:
            continue
        _log(f"[V9.1_e] {wid}: building MEOF input X...")
        out_win = _ensure_dir(out_root / "per_window" / wid)
        M, feature_meta, var_contrib, object_feature_indices, _object_matrix_meta = dmod._build_meof_matrix(profiles, years, scope, d_cfg)
        _safe_to_csv(feature_meta, out_win / f"targeted_svd_feature_metadata_{wid}.csv")
        _safe_to_csv(var_contrib, out_win / f"object_block_variance_contribution_{wid}.csv")

        reg_win = target_registry[target_registry["window_id"].astype(str).eq(str(wid))].copy() if not target_registry.empty else pd.DataFrame()
        mode_rows, score_rows, contrib_rows, coeff_rows, perm_rows, loo_rows, cv_rows = [], [], [], [], [], [], []
        phase_group_parts = []
        for ti, tr in reg_win.reset_index(drop=True).iterrows():
            target_name = str(tr["target_name"])
            _log(f"[V9.1_e] {wid}: targeted SVD for {target_name}")
            ydf = _target_y_for_registry_row(tr, v91c, years, cfg)
            y = ydf["target_y"].to_numpy(dtype=float) if not ydf.empty else np.full(len(years), np.nan)
            fit = _fit_targeted_mode(M, y, cfg)
            u = np.asarray(fit.get("mode_vector", np.zeros(M.shape[1])), dtype=float)
            score = np.asarray(fit.get("score", np.full(len(years), np.nan)), dtype=float)
            yz = np.asarray(fit.get("target_z", np.full(len(years), np.nan)), dtype=float)
            fit_status = str(fit.get("status", "unknown"))
            corr_sy = _corr(score, yz)
            sp_sy = _spearman(score, yz)
            perm_pct, perm_p = _permutation_audit(M, y, corr_sy, cfg, seed=7001 + hash((wid, target_name)) % 100000)
            loo_df = _loo_stability(M, y, u, cfg)
            cv_scores, cv_df = _cross_validated_scores(M, y, cfg)
            cv_corr = _corr(cv_scores, y)
            cv_spear = _spearman(cv_scores, y)
            cv_sign = np.nan
            if np.isfinite(cv_scores).sum() >= 3 and np.isfinite(y).sum() >= 3:
                yy = y - np.nanmean(y)
                ss = cv_scores - np.nanmean(cv_scores)
                mask = np.isfinite(yy) & np.isfinite(ss)
                if mask.sum() >= 3:
                    cv_sign = float(np.mean(np.sign(yy[mask]) == np.sign(ss[mask])))
            loo_med = float(np.nanmedian(loo_df["loo_pattern_corr"])) if not loo_df.empty else np.nan
            loo_q025 = float(np.nanquantile(loo_df["loo_pattern_corr"], 0.025)) if not loo_df.empty else np.nan
            loo_q975 = float(np.nanquantile(loo_df["loo_pattern_corr"], 0.975)) if not loo_df.empty else np.nan
            mode_status = _targeted_mode_status(perm_pct, loo_med, cv_corr, cfg) if fit_status == "ok" else fit_status
            mode_rows.append({
                "window_id": wid,
                "target_name": target_name,
                "target_type": tr.get("target_type", ""),
                "object_A": tr.get("object_A", ""),
                "object_B": tr.get("object_B", ""),
                "x_feature_source": "V9.1_d_MEOF_equal_weight_multi_object_window_anomaly_matrix",
                "y_source": "V9.1_c_bootstrap_year_influence",
                "n_years": int(len(years)),
                "n_features": int(M.shape[1]),
                "n_valid_target_years": int(np.isfinite(y).sum()),
                "target_variance": float(np.nanvar(y)),
                "covariance_strength": float(np.sqrt(np.sum((M.T @ np.nan_to_num(y - np.nanmean(y), nan=0.0)) ** 2))) if np.isfinite(y).sum() else np.nan,
                "corr_score_y": corr_sy,
                "spearman_score_y": sp_sy,
                "permutation_percentile": perm_pct,
                "permutation_empirical_p": perm_p,
                "loo_pattern_corr_median": loo_med,
                "loo_pattern_corr_q025": loo_q025,
                "loo_pattern_corr_q975": loo_q975,
                "cv_corr": cv_corr,
                "cv_spearman": cv_spear,
                "cv_sign_accuracy": cv_sign,
                "fit_status": fit_status,
                "mode_status": mode_status,
                "method_role": "target_guided_SVD_order_relevant_candidate_axis_not_physical_type",
            })
            for yi, year in enumerate(years):
                score_rows.append({
                    "window_id": wid,
                    "target_name": target_name,
                    "year": int(year),
                    "target_y": float(y[yi]) if np.isfinite(y[yi]) else np.nan,
                    "target_z": float(yz[yi]) if yi < len(yz) and np.isfinite(yz[yi]) else np.nan,
                    "mode_score": float(score[yi]) if yi < len(score) and np.isfinite(score[yi]) else np.nan,
                    "cv_score": float(cv_scores[yi]) if yi < len(cv_scores) and np.isfinite(cv_scores[yi]) else np.nan,
                })
            # Coefficients and object block contributions.
            total_energy = float(np.sum(u * u)) if u.size else np.nan
            for obj, idx in object_feature_indices.items():
                if idx:
                    e = float(np.sum(u[idx] ** 2))
                    contrib = e / total_energy if total_energy and total_energy > 0 else np.nan
                else:
                    e, contrib = 0.0, np.nan
                contrib_rows.append({"window_id": wid, "target_name": target_name, "object": obj, "block_energy": e, "block_contribution_fraction": contrib, "dominant_object_warning": bool(np.isfinite(contrib) and contrib >= 0.60)})
            for j, val in enumerate(u):
                coeff_rows.append({"window_id": wid, "target_name": target_name, "feature_col": int(j), "targeted_svd_coefficient": float(val)})
            loo_df2 = loo_df.copy(); loo_df2["window_id"] = wid; loo_df2["target_name"] = target_name
            loo_rows.append(loo_df2)
            cv_df2 = cv_df.copy(); cv_df2["window_id"] = wid; cv_df2["target_name"] = target_name
            cv_rows.append(cv_df2)
            perm_rows.append({"window_id": wid, "target_name": target_name, "observed_abs_corr": abs(corr_sy) if np.isfinite(corr_sy) else np.nan, "permutation_percentile": perm_pct, "permutation_empirical_p": perm_p, "perm_n": int(cfg.perm_n)})
            pg = _assign_target_phase_groups(wid, target_name, years, score)
            group_peak_eligible = (not cfg.skip_group_peak_for_unsupported_modes) or (mode_status in set(cfg.eligible_group_peak_statuses))
            pg["targeted_mode_status"] = mode_status
            pg["phase_group_peak_eligible"] = bool(group_peak_eligible)
            pg["performance_gate"] = "eligible_for_group_peak" if group_peak_eligible else "targeted_mode_not_eligible_for_group_peak"
            phase_group_parts.append(pg)

        mode_summary = pd.DataFrame(mode_rows)
        score_df = pd.DataFrame(score_rows)
        contrib_df = pd.DataFrame(contrib_rows)
        coeff_df = pd.DataFrame(coeff_rows)
        perm_df = pd.DataFrame(perm_rows)
        loo_all = pd.concat(loo_rows, ignore_index=True) if loo_rows else pd.DataFrame()
        cv_all = pd.concat(cv_rows, ignore_index=True) if cv_rows else pd.DataFrame()
        phase_groups = pd.concat(phase_group_parts, ignore_index=True) if phase_group_parts else pd.DataFrame()
        _safe_to_csv(mode_summary, out_win / f"targeted_svd_mode_summary_{wid}.csv")
        _safe_to_csv(score_df, out_win / f"targeted_svd_year_scores_{wid}.csv")
        _safe_to_csv(contrib_df, out_win / f"targeted_svd_object_block_contribution_{wid}.csv")
        _safe_to_csv(coeff_df, out_win / f"targeted_svd_pattern_coefficients_{wid}.csv")
        _safe_to_csv(perm_df, out_win / f"targeted_svd_permutation_audit_{wid}.csv")
        _safe_to_csv(loo_all, out_win / f"targeted_svd_loo_stability_{wid}.csv")
        _safe_to_csv(cv_all, out_win / f"targeted_svd_cross_validation_{wid}.csv")
        _safe_to_csv(phase_groups, out_win / f"targeted_svd_phase_groups_{wid}.csv")

        if not phase_groups.empty and "phase_group_peak_eligible" in phase_groups.columns:
            n_targets_total = int(phase_groups["target_name"].nunique())
            n_targets_eligible = int(phase_groups.loc[phase_groups["phase_group_peak_eligible"].astype(bool), "target_name"].nunique())
            _log(f"[V9.1_e] {wid}: group-peak performance gate eligible targets {n_targets_eligible}/{n_targets_total}")
        _log(f"[V9.1_e] {wid}: running targeted-mode phase-group V9 peak checks...")
        group_peak = _run_target_group_peak(dmod, v7multi, profiles, years, scope, phase_groups, v7_cfg, cfg)
        _safe_to_csv(group_peak["object_peak"], out_win / f"targeted_mode_group_object_peak_{wid}.csv")
        _safe_to_csv(group_peak["object_bootstrap"], out_win / f"targeted_mode_group_object_peak_bootstrap_{wid}.csv")
        _safe_to_csv(group_peak["order"], out_win / f"targeted_mode_group_pairwise_peak_order_{wid}.csv")
        _safe_to_csv(group_peak["sync"], out_win / f"targeted_mode_group_pairwise_synchrony_{wid}.csv")
        _safe_to_csv(group_peak["skipped"], out_win / f"targeted_mode_group_skipped_{wid}.csv")

        evidence = _mode_evidence_for_targets(reg_win, mode_summary, group_peak["order"], v9_order, cfg)
        _safe_to_csv(evidence, out_win / f"targeted_svd_order_mode_evidence_{wid}.csv")
        _write_summary(out_win / f"targeted_svd_order_mode_summary_{wid}.md", evidence, mode_summary, cfg)

        all_mode_summary.append(mode_summary)
        all_scores.append(score_df)
        all_obj_contrib.append(contrib_df)
        all_coeff.append(coeff_df)
        all_perm.append(perm_df)
        all_loo.append(loo_all)
        all_cv.append(cv_all)
        all_group_obj.append(group_peak["object_peak"])
        all_group_order.append(group_peak["order"])
        all_group_sync.append(group_peak["sync"])
        all_skipped.append(group_peak["skipped"])
        all_evidence.append(evidence)

    def cat(parts: List[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [p for p in parts if p is not None and not p.empty]
        if not nonempty:
            return pd.DataFrame()
        return pd.concat(nonempty, ignore_index=True)

    mode_all = cat(all_mode_summary)
    evidence_all = cat(all_evidence)
    _safe_to_csv(mode_all, out_cross / "targeted_svd_mode_summary_all_windows.csv")
    _safe_to_csv(cat(all_scores), out_cross / "targeted_svd_year_scores_all_windows.csv")
    _safe_to_csv(cat(all_obj_contrib), out_cross / "targeted_svd_object_block_contribution_all_windows.csv")
    _safe_to_csv(cat(all_coeff), out_cross / "targeted_svd_pattern_coefficients_all_windows.csv")
    _safe_to_csv(cat(all_perm), out_cross / "targeted_svd_permutation_audit_all_windows.csv")
    _safe_to_csv(cat(all_loo), out_cross / "targeted_svd_loo_stability_all_windows.csv")
    _safe_to_csv(cat(all_cv), out_cross / "targeted_svd_cross_validation_all_windows.csv")
    _safe_to_csv(cat(all_group_obj), out_cross / "targeted_mode_group_object_peak_all_windows.csv")
    _safe_to_csv(cat(all_group_order), out_cross / "targeted_mode_group_pairwise_peak_order_all_windows.csv")
    _safe_to_csv(cat(all_group_sync), out_cross / "targeted_mode_group_pairwise_synchrony_all_windows.csv")
    _safe_to_csv(cat(all_skipped), out_cross / "targeted_mode_group_skipped_all_windows.csv")
    _safe_to_csv(evidence_all, out_cross / "targeted_svd_order_mode_evidence_all_windows.csv")
    _write_summary(out_cross / "targeted_svd_order_mode_summary_all_windows.md", evidence_all, mode_all, cfg)

    _write_json({
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage_root": str(stage_root),
        "windows": cfg.windows,
        "modifies_v9": False,
        "reads_v9_outputs": True,
        "reads_v9_1_c_outputs": True,
        "requires_v9_1_d_helpers": True,
        "uses_single_year_peak": False,
        "uses_single_year_peak_for_target": False,
        "state_included": False,
        "growth_included": False,
        "process_a_included": False,
        "method_role": "target_guided_SVD_order_mode_audit_not_physical_type_assignment",
        "config": asdict(cfg),
    }, out_cross / "run_meta.json")
    _log(f"[V9.1_e] Done. Output: {out_root}")
