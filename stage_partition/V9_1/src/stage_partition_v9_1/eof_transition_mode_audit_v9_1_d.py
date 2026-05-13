"""
V9.1_d EOF/MEOF transition-mode audit.

Purpose
-------
Read-only V9 audit branch. It tests whether V9 peak/order instability can be
associated with lower-dimensional year-to-year transition modes estimated from
whole-window, multi-object behaviour anomalies.

Methodological boundary
-----------------------
This module does NOT use single-year peak days as EOF inputs, does NOT modify V9,
does NOT assign physical regime names to EOF phases, and does NOT add state,
growth, or process_a.  EOF/PC phase groups are statistical candidate axes only;
peak/order evidence is evaluated by rerunning V9 peak logic inside PC phase groups.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

VERSION = "v9_1_d_eof_transition_mode_audit"
OUTPUT_TAG = "eof_transition_mode_audit_v9_1_d"
DEFAULT_WINDOWS = ["W045", "W081", "W113", "W160"]
OBJECTS = ["P", "V", "H", "Je", "Jw"]
EXCLUDED_WINDOWS = [
    {"window_id": "W135", "included_in_v9_1_d": False, "reason": "excluded_by_V9_strict_accepted_window_set"}
]


@dataclass
class V91DConfig:
    windows: List[str] = field(default_factory=lambda: list(DEFAULT_WINDOWS))
    objects: List[str] = field(default_factory=lambda: list(OBJECTS))
    n_modes_output: int = 4
    n_modes_main: int = 3
    min_group_size_for_peak: int = 10
    group_bootstrap_n: int = 500
    debug_group_bootstrap_n: int = 50
    eof_stability_bootstrap_n: int = 300
    debug_stability_bootstrap_n: int = 50
    evidence_usable: float = 0.90
    evidence_credible: float = 0.95
    evidence_strict: float = 0.99
    stability_corr_stable: float = 0.70
    stability_corr_caution: float = 0.50
    debug: bool = False

    @classmethod
    def from_env(cls) -> "V91DConfig":
        cfg = cls()
        if os.environ.get("V9_1D_DEBUG"):
            cfg.debug = True
            cfg.group_bootstrap_n = int(os.environ.get("V9_1D_DEBUG_GROUP_BOOTSTRAP_N", cfg.debug_group_bootstrap_n))
            cfg.eof_stability_bootstrap_n = int(os.environ.get("V9_1D_DEBUG_STABILITY_BOOTSTRAP_N", cfg.debug_stability_bootstrap_n))
        if os.environ.get("V9_1D_GROUP_BOOTSTRAP_N"):
            cfg.group_bootstrap_n = int(os.environ["V9_1D_GROUP_BOOTSTRAP_N"])
        if os.environ.get("V9_1D_EOF_STABILITY_BOOTSTRAP_N"):
            cfg.eof_stability_bootstrap_n = int(os.environ["V9_1D_EOF_STABILITY_BOOTSTRAP_N"])
        if os.environ.get("V9_1D_MIN_GROUP_SIZE"):
            cfg.min_group_size_for_peak = int(os.environ["V9_1D_MIN_GROUP_SIZE"])
        if os.environ.get("V9_1D_WINDOWS"):
            cfg.windows = [x.strip() for x in os.environ["V9_1D_WINDOWS"].replace(";", ",").split(",") if x.strip()]
        if os.environ.get("V9_1D_N_MODES_MAIN"):
            cfg.n_modes_main = int(os.environ["V9_1D_N_MODES_MAIN"])
        if os.environ.get("V9_1D_N_MODES_OUTPUT"):
            cfg.n_modes_output = int(os.environ["V9_1D_N_MODES_OUTPUT"])
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


def _import_v7_module(stage_root: Path):
    v7_src = stage_root / "V7" / "src"
    if not v7_src.exists():
        raise FileNotFoundError(f"Cannot find V7 source directory: {v7_src}")
    if str(v7_src) not in sys.path:
        sys.path.insert(0, str(v7_src))
    from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as v7multi
    return v7multi


def _default_smoothed_path(stage_root: Path) -> Path:
    # stage_root is D:/easm_project01/stage_partition; foundation is sibling under project root.
    return stage_root.parent / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _make_v7_cfg(v7multi, cfg91: V91DConfig, stage_root: Path):
    cfg = v7multi.MultiWinConfig.from_env()
    cfg.window_mode = "list"
    cfg.target_windows = ",".join(cfg91.windows)
    if os.environ.get("V9_1D_ACCEPTED_WINDOW_REGISTRY"):
        cfg.accepted_window_registry = os.environ["V9_1D_ACCEPTED_WINDOW_REGISTRY"]
        cfg.window_source = "registry"
    if os.environ.get("V9_1D_SMOOTHED_FIELDS"):
        cfg.smoothed_fields_path = os.environ["V9_1D_SMOOTHED_FIELDS"]
    if not getattr(cfg, "smoothed_fields_path", None):
        default = _default_smoothed_path(stage_root)
        if default.exists():
            cfg.smoothed_fields_path = str(default)
    cfg.bootstrap_n = int(cfg91.group_bootstrap_n)
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    return cfg


def _window_scopes_from_v7(v7multi, stage_root: Path, cfg, target_windows: Sequence[str]) -> List[object]:
    v7_root = stage_root / "V7"
    tmp = _ensure_dir(stage_root / "V9_1" / "outputs" / OUTPUT_TAG / "cross_window" / "_window_registry_tmp")
    wins = v7multi._load_accepted_windows(v7_root, tmp, cfg)
    scopes, _validity = v7multi._build_window_scopes(wins, cfg)
    run_scopes, _audit = v7multi._filter_scopes_for_run(scopes, cfg)
    want = set(target_windows)
    return [s for s in run_scopes if s.window_id in want]


def _load_v9_tables(stage_root: Path) -> Dict[str, pd.DataFrame]:
    v9_out = stage_root / "V9" / "outputs" / "peak_all_windows_v9_a" / "cross_window"
    out = {}
    for key, fname in {
        "accepted_windows": "accepted_windows_used_v9_a.csv",
        "object_peak": "cross_window_object_peak_registry.csv",
        "pairwise_order": "cross_window_pairwise_peak_order.csv",
        "pairwise_sync": "cross_window_pairwise_peak_synchrony.csv",
    }.items():
        f = v9_out / fname
        out[key] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return out


def _load_v91c_tables(v91_root: Path) -> Dict[str, pd.DataFrame]:
    c_out = v91_root / "outputs" / "bootstrap_year_influence_audit_v9_1_c" / "cross_window"
    out = {}
    for key, fname in {
        "object_influence": "object_peak_year_influence_all_windows.csv",
        "pairwise_influence": "pairwise_order_year_influence_all_windows.csv",
        "year_sets": "influential_year_sets_all_windows.csv",
        "summary": "bootstrap_year_influence_summary_all_windows.csv",
    }.items():
        f = c_out / fname
        out[key] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return out


def _load_profiles(v7multi, stage_root: Path, cfg) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, pd.DataFrame]:
    smoothed = Path(cfg.smoothed_fields_path) if getattr(cfg, "smoothed_fields_path", None) else _default_smoothed_path(stage_root)
    if not smoothed.exists():
        raise FileNotFoundError(f"smoothed_fields.npz not found: {smoothed}. Set V9_1D_SMOOTHED_FIELDS if needed.")
    fields, _audit = v7multi.clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for spec in v7multi.clean.OBJECT_SPECS:
        arr = v7multi.clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (np.asarray(prof, dtype=float), target_lat, weights)
        rows.append({
            **getattr(spec, "__dict__", {}),
            "object_name": spec.object_name,
            "profile_shape": str(prof.shape),
            "target_lat_min": float(np.nanmin(target_lat)),
            "target_lat_max": float(np.nanmax(target_lat)),
            "v9_1d_role": "MEOF_input_and_PC_phase_group_peak_input",
        })
    ny = next(iter(profiles.values()))[0].shape[0]
    if years is None:
        years = np.arange(ny)
    years = np.asarray(years)
    if years.shape[0] != ny:
        years = np.arange(ny)
    return profiles, years, pd.DataFrame(rows)


def _zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1.0e-12), sd, 1.0)
    Z = (X - mu) / sd
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z, mu, sd


def _object_window_matrix(prof: np.ndarray, start: int, end: int) -> np.ndarray:
    prof = np.asarray(prof, dtype=float)
    n_years, n_days = prof.shape[0], prof.shape[1]
    s = max(0, int(start))
    e = min(n_days - 1, int(end))
    if e < s:
        return np.zeros((n_years, 0))
    sub = prof[:, s:e + 1, ...]
    return sub.reshape(n_years, -1)


def _build_meof_matrix(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    years: np.ndarray,
    scope: object,
    cfg: V91DConfig,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Dict[str, List[int]], Dict[str, Tuple[np.ndarray, int, int]]]:
    """Return equal-weight MEOF matrix: years x combined features.

    Each object block is converted to year-anomaly/z-score features and divided
    by sqrt(n_features_in_block) so that P/V/H/Je/Jw contribute similarly.
    """
    blocks = []
    feature_rows = []
    var_rows = []
    object_feature_indices: Dict[str, List[int]] = {}
    object_matrix_meta: Dict[str, Tuple[np.ndarray, int, int]] = {}
    col_offset = 0
    for obj in cfg.objects:
        if obj not in profiles:
            continue
        prof = profiles[obj][0]
        X = _object_window_matrix(prof, scope.analysis_start, scope.analysis_end)
        Z, _, _ = _zscore_cols(X)
        if Z.shape[1] > 0:
            Z = Z / math.sqrt(Z.shape[1])
        idx = list(range(col_offset, col_offset + Z.shape[1]))
        object_feature_indices[obj] = idx
        col_offset += Z.shape[1]
        blocks.append(Z)
        total_var = float(np.nanvar(Z, axis=0).sum()) if Z.size else 0.0
        var_rows.append({
            "window_id": scope.window_id,
            "object": obj,
            "n_features": int(Z.shape[1]),
            "equal_weight_scale": f"1/sqrt({Z.shape[1]})" if Z.shape[1] else "empty_block",
            "block_variance_after_scaling": total_var,
            "feature_role": "MEOF_equal_weight_object_block",
        })
        # Save only compact feature row descriptions to avoid a huge metadata table.
        feature_rows.append({
            "window_id": scope.window_id,
            "object": obj,
            "feature_start_col": idx[0] if idx else -1,
            "feature_end_col": idx[-1] if idx else -1,
            "source_days": f"{scope.analysis_start}-{scope.analysis_end}",
            "n_source_features": int(Z.shape[1]),
        })
        object_matrix_meta[obj] = (prof, int(scope.analysis_start), int(scope.analysis_end))
    M = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(years), 0))
    return M, pd.DataFrame(feature_rows), pd.DataFrame(var_rows), object_feature_indices, object_matrix_meta


def _svd_eof(M: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = np.asarray(M, dtype=float)
    M = np.where(np.isfinite(M), M, 0.0)
    n_modes = int(max(1, min(n_modes, M.shape[0] - 1 if M.shape[0] > 1 else 1, M.shape[1] if M.shape[1] else 1)))
    if M.shape[0] <= 1 or M.shape[1] == 0:
        return np.zeros((M.shape[0], n_modes)), np.zeros((n_modes, M.shape[1])), np.zeros(n_modes), np.zeros(n_modes)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    scores = U[:, :n_modes] * s[:n_modes]
    eof = Vt[:n_modes]
    eig = s ** 2
    evr = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    return scores, eof, evr[:n_modes], s[:n_modes]


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
    # simple average ranks for ties
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
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return _corr(_rankdata(a[mask]), _rankdata(b[mask]))


def _eof_mode_tables(
    scope: object,
    years: np.ndarray,
    scores: np.ndarray,
    eof: np.ndarray,
    evr: np.ndarray,
    object_feature_indices: Dict[str, List[int]],
    cfg: V91DConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_rows, score_rows, leverage_rows = [], [], []
    n_modes = min(cfg.n_modes_output, scores.shape[1] if scores.ndim == 2 else 0)
    for m in range(n_modes):
        e = eof[m]
        total_energy = float(np.sum(e * e)) if e.size else np.nan
        mode_rows.append({
            "window_id": scope.window_id,
            "mode": int(m + 1),
            "explained_variance_ratio": float(evr[m]) if m < len(evr) else np.nan,
            "cumulative_variance_ratio": float(np.nansum(evr[:m + 1])) if len(evr) else np.nan,
            "total_eof_energy": total_energy,
            "mode_role": "MEOF_transition_mode_candidate_not_physical_type",
        })
        abs_scores = np.abs(scores[:, m])
        q95 = float(np.nanquantile(abs_scores, 0.95)) if abs_scores.size else np.nan
        for yi, year in enumerate(years):
            score = float(scores[yi, m])
            score_rows.append({"window_id": scope.window_id, "year": year, "mode": int(m + 1), "pc_score": score})
            leverage_rows.append({
                "window_id": scope.window_id,
                "year": year,
                "mode": int(m + 1),
                "pc_score": score,
                "abs_pc_score": abs(score),
                "abs_pc_score_rank_desc": int((-abs_scores).argsort().tolist().index(yi) + 1) if abs_scores.size else np.nan,
                "is_extreme_pc_year_q95": bool(abs(score) >= q95) if np.isfinite(q95) else False,
                "leverage_caution": "extreme_year_leverage_caution" if np.isfinite(q95) and abs(score) >= q95 else "none",
            })
    # object-wise energy contribution in separate rows appended to mode summary
    energy_rows = []
    for m in range(n_modes):
        e = eof[m]
        denom = float(np.sum(e * e)) if e.size else np.nan
        for obj, idx in object_feature_indices.items():
            val = float(np.sum(e[idx] * e[idx])) if idx and np.isfinite(denom) and denom > 0 else np.nan
            energy_rows.append({
                "window_id": scope.window_id,
                "mode": int(m + 1),
                "object": obj,
                "object_eof_energy_fraction": val / denom if np.isfinite(val) and denom > 0 else np.nan,
            })
    mode_df = pd.DataFrame(mode_rows)
    energy_df = pd.DataFrame(energy_rows)
    if not mode_df.empty and not energy_df.empty:
        # Keep energy as a separate file-style table by returning within mode_df with object column rows too.
        pass
    return mode_df, pd.DataFrame(score_rows), pd.DataFrame(leverage_rows)


def _eof_object_energy_table(scope, eof: np.ndarray, evr: np.ndarray, object_feature_indices: Dict[str, List[int]], cfg: V91DConfig) -> pd.DataFrame:
    rows = []
    n_modes = min(cfg.n_modes_output, eof.shape[0] if eof.ndim == 2 else 0)
    for m in range(n_modes):
        e = eof[m]
        denom = float(np.sum(e * e)) if e.size else np.nan
        for obj, idx in object_feature_indices.items():
            val = float(np.sum(e[idx] * e[idx])) if idx and np.isfinite(denom) and denom > 0 else np.nan
            rows.append({
                "window_id": scope.window_id,
                "mode": int(m + 1),
                "object": obj,
                "explained_variance_ratio": float(evr[m]) if m < len(evr) else np.nan,
                "object_eof_energy_fraction": val / denom if np.isfinite(val) and denom > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def _eof_stability_bootstrap(M: np.ndarray, eof: np.ndarray, evr: np.ndarray, scope, cfg: V91DConfig) -> pd.DataFrame:
    n = M.shape[0]
    n_modes = min(cfg.n_modes_main, eof.shape[0] if eof.ndim == 2 else 0)
    rows = []
    if n < 5 or n_modes == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(20260904 + int(str(scope.window_id).replace("W", "")))
    corr_by_mode = {m: [] for m in range(n_modes)}
    for _ in range(cfg.eof_stability_bootstrap_n):
        idx = rng.integers(0, n, size=n)
        Mb = M[idx, :]
        _s, eof_b, _evr_b, _sv_b = _svd_eof(Mb, n_modes)
        for m in range(n_modes):
            if m >= eof_b.shape[0]:
                continue
            c = _corr(eof[m], eof_b[m])
            if np.isfinite(c):
                corr_by_mode[m].append(abs(c))
    for m in range(n_modes):
        vals = np.asarray(corr_by_mode[m], dtype=float)
        med = float(np.nanmedian(vals)) if vals.size else np.nan
        q025 = float(np.nanquantile(vals, 0.025)) if vals.size else np.nan
        q975 = float(np.nanquantile(vals, 0.975)) if vals.size else np.nan
        if np.isfinite(med) and med >= cfg.stability_corr_stable:
            status = "stable_mode"
        elif np.isfinite(med) and med >= cfg.stability_corr_caution:
            status = "caution_mode"
        else:
            status = "unstable_mode"
        rows.append({
            "window_id": scope.window_id,
            "mode": int(m + 1),
            "explained_variance_ratio": float(evr[m]) if m < len(evr) else np.nan,
            "bootstrap_pattern_corr_abs_median": med,
            "bootstrap_pattern_corr_abs_q025": q025,
            "bootstrap_pattern_corr_abs_q975": q975,
            "n_bootstrap": int(cfg.eof_stability_bootstrap_n),
            "mode_stability_status": status,
        })
    return pd.DataFrame(rows)


def _assign_pc_phase_groups(scope, years: np.ndarray, scores: np.ndarray, cfg: V91DConfig) -> pd.DataFrame:
    rows = []
    n_modes = min(cfg.n_modes_main, scores.shape[1] if scores.ndim == 2 else 0)
    for m in range(n_modes):
        pc = scores[:, m]
        q_low, q_high = np.nanquantile(pc, [1/3, 2/3])
        for yi, year in enumerate(years):
            val = pc[yi]
            if val <= q_low:
                group = "PC_low"
            elif val >= q_high:
                group = "PC_high"
            else:
                group = "PC_mid"
            rows.append({
                "window_id": scope.window_id,
                "year": year,
                "mode": int(m + 1),
                "phase_group": group,
                "pc_score": float(val),
                "grouping_method": "tercile_pc_phase_group",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        sizes = df.groupby(["mode", "phase_group"]).size().rename("phase_group_size").reset_index()
        df = df.merge(sizes, on=["mode", "phase_group"], how="left")
    return df


def _subset_profiles(profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], idx: Sequence[int]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    idx = np.asarray(idx, dtype=int)
    out = {}
    for obj, (prof, lat, w) in profiles.items():
        out[obj] = (np.asarray(prof)[idx, ...].copy(), lat, w)
    return out


def _order_level(prob: float, cfg: V91DConfig) -> str:
    if prob >= cfg.evidence_strict:
        return "strict_99"
    if prob >= cfg.evidence_credible:
        return "credible_95"
    if prob >= cfg.evidence_usable:
        return "usable_90"
    return "unresolved"


def _add_order_levels(order_df: pd.DataFrame, cfg: V91DConfig) -> pd.DataFrame:
    if order_df is None or order_df.empty:
        return pd.DataFrame()
    out = order_df.copy()
    dirs, probs, levels = [], [], []
    for _, r in out.iterrows():
        pa = float(r.get("P_A_earlier", np.nan)); pb = float(r.get("P_B_earlier", np.nan))
        if np.nan_to_num(pa, nan=-1) >= np.nan_to_num(pb, nan=-1):
            p = pa; d = "A_earlier"
        else:
            p = pb; d = "B_earlier"
        dirs.append(d); probs.append(p); levels.append(_order_level(p, cfg))
    out["pc_group_order_best_direction"] = dirs
    out["pc_group_order_best_probability"] = probs
    out["pc_group_order_evidence_level"] = levels
    return out


def _run_pc_group_peak(
    v7multi,
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    years: np.ndarray,
    scope: object,
    phase_groups: pd.DataFrame,
    cfg_v7,
    cfg: V91DConfig,
) -> Dict[str, pd.DataFrame]:
    obj_parts, boot_parts, order_parts, sync_parts, skipped_rows = [], [], [], [], []
    if phase_groups.empty:
        return {"object_peak": pd.DataFrame(), "object_bootstrap": pd.DataFrame(), "order": pd.DataFrame(), "sync": pd.DataFrame(), "skipped": pd.DataFrame()}
    year_list = years.tolist()
    for (mode, group), sub in phase_groups.groupby(["mode", "phase_group"]):
        idx = [int(np.where(years == y)[0][0]) for y in sub["year"].to_numpy() if y in year_list]
        group_size = len(idx)
        if group_size < cfg.min_group_size_for_peak:
            skipped_rows.append({
                "window_id": scope.window_id,
                "mode": int(mode),
                "phase_group": group,
                "phase_group_size": int(group_size),
                "skip_reason": "phase_group_size_below_min_group_size_for_peak",
                "min_group_size_for_peak": int(cfg.min_group_size_for_peak),
            })
            continue
        _log(f"    PC group peak {scope.window_id}: mode={mode} group={group} n={group_size}")
        sub_profiles = _subset_profiles(profiles, idx)
        try:
            _score_df, _cand_df, selection_df, _selected_delta_df, boot_peak_days_df = v7multi._run_detector_and_bootstrap(sub_profiles, scope, cfg_v7)
            timing_audit_df, tau_df = v7multi._estimate_timing_resolution(selection_df, boot_peak_days_df, cfg_v7, scope)
            order_df = v7multi._pairwise_peak_order(selection_df, boot_peak_days_df, scope)
            sync_df = v7multi._pairwise_synchrony(order_df, boot_peak_days_df, tau_df, scope)
        except Exception as exc:
            skipped_rows.append({
                "window_id": scope.window_id,
                "mode": int(mode),
                "phase_group": group,
                "phase_group_size": int(group_size),
                "skip_reason": "pc_group_peak_runtime_error",
                "error": str(exc),
            })
            continue
        add = {
            "window_id": scope.window_id,
            "mode": int(mode),
            "phase_group": group,
            "phase_group_size": int(group_size),
            "years_in_phase_group": ";".join(str(x) for x in sorted(sub["year"].to_numpy())),
            "group_role": "PC_phase_group_multi_year_peak_not_single_year_peak",
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


def _v9_order_lookup(v9_order: pd.DataFrame, window_id: str) -> pd.DataFrame:
    if v9_order is None or v9_order.empty or "window_id" not in v9_order.columns:
        return pd.DataFrame()
    return v9_order[v9_order["window_id"].astype(str) == str(window_id)].copy()


def _best_prob_from_v9_row(r: pd.Series, cfg: V91DConfig) -> Tuple[str, float, str]:
    pa = float(r.get("P_A_earlier", np.nan)); pb = float(r.get("P_B_earlier", np.nan))
    if np.nan_to_num(pa, nan=-1) >= np.nan_to_num(pb, nan=-1):
        return "A_earlier", pa, _order_level(pa, cfg)
    return "B_earlier", pb, _order_level(pb, cfg)


def _build_peak_order_evidence(pc_order: pd.DataFrame, stability: pd.DataFrame, v9_order_win: pd.DataFrame, cfg: V91DConfig) -> pd.DataFrame:
    if pc_order.empty:
        return pd.DataFrame()
    rows = []
    for (mode, A, B), sub in pc_order.groupby(["mode", "object_A", "object_B"]):
        v9_dir, v9_prob, v9_level = "missing", np.nan, "missing"
        match = v9_order_win[(v9_order_win.get("object_A", pd.Series(dtype=str)).astype(str) == str(A)) & (v9_order_win.get("object_B", pd.Series(dtype=str)).astype(str) == str(B))]
        if not match.empty:
            v9_dir, v9_prob, v9_level = _best_prob_from_v9_row(match.iloc[0], cfg)
        stab = stability[(stability["mode"].astype(int) == int(mode))] if not stability.empty and "mode" in stability.columns else pd.DataFrame()
        mode_status = str(stab["mode_stability_status"].iloc[0]) if not stab.empty else "missing"
        high = sub[sub["phase_group"].astype(str) == "PC_high"]
        low = sub[sub["phase_group"].astype(str) == "PC_low"]
        mid = sub[sub["phase_group"].astype(str) == "PC_mid"]
        def gl(df, col):
            return str(df[col].iloc[0]) if not df.empty and col in df.columns else "missing"
        def gp(df, col):
            return float(df[col].iloc[0]) if not df.empty and col in df.columns and pd.notna(df[col].iloc[0]) else np.nan
        high_dir, low_dir, mid_dir = gl(high, "pc_group_order_best_direction"), gl(low, "pc_group_order_best_direction"), gl(mid, "pc_group_order_best_direction")
        high_level, low_level, mid_level = gl(high, "pc_group_order_evidence_level"), gl(low, "pc_group_order_evidence_level"), gl(mid, "pc_group_order_evidence_level")
        high_prob, low_prob, mid_prob = gp(high, "pc_group_order_best_probability"), gp(low, "pc_group_order_best_probability"), gp(mid, "pc_group_order_best_probability")
        strong_levels = {"usable_90", "credible_95", "strict_99"}
        credible_levels = {"credible_95", "strict_99"}
        opposite = bool(high_dir in ("A_earlier", "B_earlier") and low_dir in ("A_earlier", "B_earlier") and high_dir != low_dir)
        internal_stab = bool(high_level in strong_levels or low_level in strong_levels or mid_level in strong_levels)
        if mode_status == "unstable_mode":
            evidence = "not_supported_unstable_eof"
        elif opposite and (high_level in credible_levels or low_level in credible_levels):
            evidence = "supported_eof_mode_explanation"
        elif opposite and internal_stab:
            evidence = "partial_eof_mode_hint"
        elif internal_stab and (np.nanmax([high_prob, low_prob, mid_prob]) >= np.nan_to_num(v9_prob, nan=0) + 0.05):
            evidence = "partial_eof_mode_hint"
        else:
            evidence = "not_supported"
        rows.append({
            "window_id": sub["window_id"].iloc[0],
            "mode": int(mode),
            "object_A": A,
            "object_B": B,
            "full_sample_order_direction": v9_dir,
            "full_sample_order_probability": v9_prob,
            "full_sample_order_level": v9_level,
            "pc_high_order_direction": high_dir,
            "pc_high_order_probability": high_prob,
            "pc_high_order_level": high_level,
            "pc_mid_order_direction": mid_dir,
            "pc_mid_order_probability": mid_prob,
            "pc_mid_order_level": mid_level,
            "pc_low_order_direction": low_dir,
            "pc_low_order_probability": low_prob,
            "pc_low_order_level": low_level,
            "opposite_order_between_pc_high_low": opposite,
            "group_internal_stabilization": internal_stab,
            "mode_stability_status": mode_status,
            "eof_explains_peak_order_instability": evidence,
            "recommended_interpretation": _eof_evidence_recommendation(evidence),
        })
    return pd.DataFrame(rows)


def _eof_evidence_recommendation(level: str) -> str:
    if level == "supported_eof_mode_explanation":
        return "PC phase groups stabilize or reverse peak-order evidence; inspect EOF composites before physical naming"
    if level == "partial_eof_mode_hint":
        return "PC phase has some relation to peak/order instability; hypothesis-generating only"
    if level == "not_supported_unstable_eof":
        return "EOF mode itself is unstable; do not use this PC phase to explain V9 instability"
    return "PC phase grouping does not explain V9 peak/order instability in this audit"


def _numeric_year_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
    return out


def _pc_vs_influence(scope, pc_scores: pd.DataFrame, v91c: Dict[str, pd.DataFrame], cfg: V91DConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    object_inf = _numeric_year_column(v91c.get("object_influence", pd.DataFrame()))
    pair_inf = _numeric_year_column(v91c.get("pairwise_influence", pd.DataFrame()))
    pc = _numeric_year_column(pc_scores)
    pc = pc[pc["window_id"].astype(str) == str(scope.window_id)].copy() if not pc.empty else pc
    obj_rows, pair_rows, overlap_rows = [], [], []
    if pc.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Object influence correlations.
    oi = object_inf[object_inf.get("window_id", pd.Series(dtype=str)).astype(str) == str(scope.window_id)].copy() if not object_inf.empty else pd.DataFrame()
    if not oi.empty:
        # Group only by EOF mode here.  Earlier draft attempted to unpack
        # (object, mode) from pc.groupby(["mode"]), but pc_scores does not
        # contain object-specific rows at this stage; object loops are handled
        # by grouping the influence table below.
        for mode, subpc in pc.groupby("mode"):
            for obj, sub in oi.groupby("object"):
                merged = subpc[["year", "pc_score"]].merge(sub, on="year", how="inner")
                if merged.empty:
                    continue
                for target_col in ["presence_effect_on_peak_day", "presence_effect_on_early_probability", "presence_effect_on_late_probability", "high_count_effect_on_peak_day"]:
                    if target_col not in merged.columns:
                        continue
                    obj_rows.append({
                        "window_id": scope.window_id,
                        "mode": int(mode),
                        "target_type": "object_peak_year_influence",
                        "target_name": obj,
                        "target_metric": target_col,
                        "corr_pc_with_influence_score": _corr(merged["pc_score"].to_numpy(), merged[target_col].to_numpy()),
                        "spearman_pc_with_influence_score": _spearman(merged["pc_score"].to_numpy(), merged[target_col].to_numpy()),
                        "n_years": int(len(merged)),
                        "relationship_status": "diagnostic_only_not_physical_type",
                    })
                # high-influence year overlap by direction/evidence if available
                if "influence_evidence_level" in merged.columns:
                    high = merged[merged["influence_evidence_level"].astype(str).isin(["credible_95", "strict_99"])]
                    if not high.empty:
                        for direction, g in high.groupby("influence_direction"):
                            overlap_rows.append({
                                "window_id": scope.window_id,
                                "mode": int(mode),
                                "target": obj,
                                "target_type": "object_peak",
                                "influence_group": direction,
                                "years": ";".join(str(int(y)) for y in sorted(g["year"].dropna().astype(int).tolist())),
                                "n_years": int(len(g)),
                                "pc_score_mean": float(np.nanmean(g["pc_score"])),
                                "pc_score_min": float(np.nanmin(g["pc_score"])),
                                "pc_score_max": float(np.nanmax(g["pc_score"])),
                                "pc_phase_concentration": "candidate_concentrated_positive" if np.nanmean(g["pc_score"]) > 0 else "candidate_concentrated_negative",
                                "overlap_status": "high_influence_years_mapped_to_pc_phase_candidate",
                            })
    # Pairwise influence correlations.
    pi = pair_inf[pair_inf.get("window_id", pd.Series(dtype=str)).astype(str) == str(scope.window_id)].copy() if not pair_inf.empty else pd.DataFrame()
    if not pi.empty:
        for mode, subpc in pc.groupby("mode"):
            for (A, B), sub in pi.groupby(["object_A", "object_B"]):
                merged = subpc[["year", "pc_score"]].merge(sub, on="year", how="inner")
                if merged.empty:
                    continue
                for target_col in ["presence_effect_on_A_earlier", "presence_effect_on_B_earlier", "high_count_effect_on_A_earlier", "high_count_effect_on_B_earlier"]:
                    if target_col not in merged.columns:
                        continue
                    pair_rows.append({
                        "window_id": scope.window_id,
                        "mode": int(mode),
                        "object_A": A,
                        "object_B": B,
                        "target_metric": target_col,
                        "corr_pc_with_order_influence": _corr(merged["pc_score"].to_numpy(), merged[target_col].to_numpy()),
                        "spearman_pc_with_order_influence": _spearman(merged["pc_score"].to_numpy(), merged[target_col].to_numpy()),
                        "n_years": int(len(merged)),
                        "pc_order_relationship_status": "diagnostic_only_not_physical_type",
                    })
                if "influence_evidence_level" in merged.columns:
                    high = merged[merged["influence_evidence_level"].astype(str).isin(["credible_95", "strict_99"])]
                    if not high.empty:
                        for direction, g in high.groupby("influence_direction"):
                            overlap_rows.append({
                                "window_id": scope.window_id,
                                "mode": int(mode),
                                "target": f"{A}-{B}",
                                "target_type": "pairwise_order",
                                "influence_group": direction,
                                "years": ";".join(str(int(y)) for y in sorted(g["year"].dropna().astype(int).tolist())),
                                "n_years": int(len(g)),
                                "pc_score_mean": float(np.nanmean(g["pc_score"])),
                                "pc_score_min": float(np.nanmin(g["pc_score"])),
                                "pc_score_max": float(np.nanmax(g["pc_score"])),
                                "pc_phase_concentration": "candidate_concentrated_positive" if np.nanmean(g["pc_score"]) > 0 else "candidate_concentrated_negative",
                                "overlap_status": "high_influence_years_mapped_to_pc_phase_candidate",
                            })
    return pd.DataFrame(obj_rows), pd.DataFrame(pair_rows), pd.DataFrame(overlap_rows)


def _composite_profiles(scope, profiles, years, phase_groups: pd.DataFrame, cfg: V91DConfig) -> pd.DataFrame:
    rows = []
    if phase_groups.empty:
        return pd.DataFrame()
    years_arr = np.asarray(years)
    year_to_idx = {y: i for i, y in enumerate(years_arr.tolist())}
    for (mode, group), sub in phase_groups.groupby(["mode", "phase_group"]):
        idx = [year_to_idx[y] for y in sub["year"].tolist() if y in year_to_idx]
        if not idx:
            continue
        for obj in cfg.objects:
            if obj not in profiles:
                continue
            prof = profiles[obj][0]
            lat = profiles[obj][1]
            s = max(0, int(scope.analysis_start)); e = min(prof.shape[1] - 1, int(scope.analysis_end))
            comp = np.nanmean(prof[idx, s:e+1, :], axis=0)
            clim = np.nanmean(prof[:, s:e+1, :], axis=0)
            anom = comp - clim
            for di, day in enumerate(range(s, e + 1)):
                for pi in range(anom.shape[1]):
                    coord = float(lat[pi]) if len(lat) > pi else float(pi)
                    rows.append({
                        "window_id": scope.window_id,
                        "mode": int(mode),
                        "phase_group": group,
                        "phase_group_size": int(len(idx)),
                        "object": obj,
                        "day": int(day),
                        "profile_coord": coord,
                        "composite_value": float(comp[di, pi]) if np.isfinite(comp[di, pi]) else np.nan,
                        "composite_anomaly_vs_all_years": float(anom[di, pi]) if np.isfinite(anom[di, pi]) else np.nan,
                        "composite_role": "PC_phase_profile_composite_for_later_physical_inspection_not_named_regime",
                    })
    return pd.DataFrame(rows)


def _write_summary(path: Path, evidence: pd.DataFrame, mode_summary: pd.DataFrame, stability: pd.DataFrame, cfg: V91DConfig) -> None:
    lines = [
        "# V9.1_d EOF transition-mode audit summary",
        "",
        f"version: `{VERSION}`",
        "",
        "## Method boundary",
        "- Read-only relative to V9.",
        "- EOF inputs are whole-window, multi-object year-anomaly profiles, not single-year peak days.",
        "- EOF/PC phases are statistical transition-mode candidates, not physical regime names.",
        "- Peak/order interpretation requires PC phase group reruns of V9 peak logic.",
        "",
        "## Configuration",
        f"- windows: {', '.join(cfg.windows)}",
        f"- n_modes_main: {cfg.n_modes_main}",
        f"- group_bootstrap_n: {cfg.group_bootstrap_n}",
        f"- eof_stability_bootstrap_n: {cfg.eof_stability_bootstrap_n}",
        f"- min_group_size_for_peak: {cfg.min_group_size_for_peak}",
        "",
        "## EOF peak/order evidence counts",
    ]
    if evidence is not None and not evidence.empty and "eof_explains_peak_order_instability" in evidence.columns:
        for k, v in evidence["eof_explains_peak_order_instability"].value_counts(dropna=False).items():
            lines.append(f"- {k}: {int(v)}")
    else:
        lines.append("- no evidence table generated")
    lines.append("")
    lines.append("## Mode stability counts")
    if stability is not None and not stability.empty and "mode_stability_status" in stability.columns:
        for k, v in stability["mode_stability_status"].value_counts(dropna=False).items():
            lines.append(f"- {k}: {int(v)}")
    else:
        lines.append("- no stability table generated")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_eof_transition_mode_audit_v9_1_d(v91_root: Path | str) -> None:
    v91_root = Path(v91_root)
    stage_root = _stage_root_from_v91(v91_root)
    cfg91 = V91DConfig.from_env()
    out_root = _ensure_dir(v91_root / "outputs" / OUTPUT_TAG)
    out_per = _ensure_dir(out_root / "per_window")
    out_cross = _ensure_dir(out_root / "cross_window")
    log_dir = _ensure_dir(v91_root / "logs" / OUTPUT_TAG)
    t0 = time.time()

    _log("[1/8] Load V9/V7 context and profile inputs")
    v7multi = _import_v7_module(stage_root)
    cfg_v7 = _make_v7_cfg(v7multi, cfg91, stage_root)
    v9_tables = _load_v9_tables(stage_root)
    v91c_tables = _load_v91c_tables(v91_root)
    scopes = _window_scopes_from_v7(v7multi, stage_root, cfg_v7, cfg91.windows)
    profiles, years, object_registry = _load_profiles(v7multi, stage_root, cfg_v7)
    _safe_to_csv(object_registry, out_cross / "object_registry_v9_1_d.csv")
    _safe_to_csv(pd.DataFrame(EXCLUDED_WINDOWS), out_cross / "excluded_windows_from_v9_1_d.csv")

    all_mode, all_energy, all_scores, all_stability, all_leverage = [], [], [], [], []
    all_phase, all_obj_pc, all_pair_pc, all_overlap = [], [], [], []
    all_group_obj, all_group_boot, all_group_order, all_group_sync, all_group_skipped = [], [], [], [], []
    all_evidence, all_comp = [], []

    for i, scope in enumerate(scopes, start=1):
        if scope.window_id not in cfg91.windows:
            continue
        _log(f"[2/8] {scope.window_id}: build MEOF input matrix ({i}/{len(scopes)})")
        out_win = _ensure_dir(out_per / scope.window_id)
        M, feat_meta, var_contrib, obj_idx, _obj_meta = _build_meof_matrix(profiles, years, scope, cfg91)
        _safe_to_csv(feat_meta, out_win / f"meof_input_feature_metadata_{scope.window_id}.csv")
        _safe_to_csv(var_contrib, out_win / f"object_block_variance_contribution_{scope.window_id}.csv")
        # Compact feature matrix: years plus first many columns can be huge; still save the actual matrix for audit.
        mat_df = pd.DataFrame(M)
        mat_df.insert(0, "year", years)
        mat_df.insert(0, "window_id", scope.window_id)
        _safe_to_csv(mat_df, out_win / f"meof_input_feature_matrix_{scope.window_id}.csv")

        _log(f"[3/8] {scope.window_id}: compute MEOF/PCs")
        scores, eof, evr, _svals = _svd_eof(M, cfg91.n_modes_output)
        mode_df, score_df, leverage_df = _eof_mode_tables(scope, years, scores, eof, evr, obj_idx, cfg91)
        energy_df = _eof_object_energy_table(scope, eof, evr, obj_idx, cfg91)
        _safe_to_csv(mode_df, out_win / f"eof_mode_summary_{scope.window_id}.csv")
        _safe_to_csv(energy_df, out_win / f"eof_object_energy_contribution_{scope.window_id}.csv")
        _safe_to_csv(score_df, out_win / f"pc_scores_{scope.window_id}.csv")
        _safe_to_csv(leverage_df, out_win / f"pc_year_leverage_{scope.window_id}.csv")
        all_mode.append(mode_df); all_energy.append(energy_df); all_scores.append(score_df); all_leverage.append(leverage_df)

        _log(f"[4/8] {scope.window_id}: EOF bootstrap stability")
        stability = _eof_stability_bootstrap(M, eof, evr, scope, cfg91)
        _safe_to_csv(stability, out_win / f"eof_mode_stability_{scope.window_id}.csv")
        all_stability.append(stability)

        _log(f"[5/8] {scope.window_id}: connect PCs with V9.1_c high-influence years if available")
        obj_pc, pair_pc, overlap = _pc_vs_influence(scope, score_df, v91c_tables, cfg91)
        _safe_to_csv(obj_pc, out_win / f"pc_vs_v9_instability_targets_{scope.window_id}.csv")
        _safe_to_csv(pair_pc, out_win / f"pc_vs_pairwise_order_influence_{scope.window_id}.csv")
        _safe_to_csv(overlap, out_win / f"eof_pc_influence_year_overlap_{scope.window_id}.csv")
        all_obj_pc.append(obj_pc); all_pair_pc.append(pair_pc); all_overlap.append(overlap)

        _log(f"[6/8] {scope.window_id}: PC phase grouping and V9 peak rerun inside groups")
        phase_groups = _assign_pc_phase_groups(scope, years, scores, cfg91)
        _safe_to_csv(phase_groups, out_win / f"pc_phase_group_membership_{scope.window_id}.csv")
        all_phase.append(phase_groups)
        group_res = _run_pc_group_peak(v7multi, profiles, years, scope, phase_groups, cfg_v7, cfg91)
        _safe_to_csv(group_res["object_peak"], out_win / f"pc_phase_group_object_peak_{scope.window_id}.csv")
        _safe_to_csv(group_res["object_bootstrap"], out_win / f"pc_phase_group_object_peak_bootstrap_{scope.window_id}.csv")
        _safe_to_csv(group_res["order"], out_win / f"pc_phase_group_pairwise_order_{scope.window_id}.csv")
        _safe_to_csv(group_res["sync"], out_win / f"pc_phase_group_pairwise_synchrony_{scope.window_id}.csv")
        _safe_to_csv(group_res["skipped"], out_win / f"pc_phase_group_peak_skipped_{scope.window_id}.csv")
        all_group_obj.append(group_res["object_peak"]); all_group_boot.append(group_res["object_bootstrap"])
        all_group_order.append(group_res["order"]); all_group_sync.append(group_res["sync"]); all_group_skipped.append(group_res["skipped"])

        _log(f"[7/8] {scope.window_id}: evaluate EOF/PC phase explanation of V9 peak-order instability")
        v9_order_win = _v9_order_lookup(v9_tables.get("pairwise_order", pd.DataFrame()), scope.window_id)
        evidence = _build_peak_order_evidence(group_res["order"], stability, v9_order_win, cfg91)
        _safe_to_csv(evidence, out_win / f"eof_transition_mode_peak_order_evidence_{scope.window_id}.csv")
        all_evidence.append(evidence)

        _log(f"[8/8] {scope.window_id}: write PC phase composite profiles")
        comp = _composite_profiles(scope, profiles, years, phase_groups, cfg91)
        _safe_to_csv(comp, out_win / f"eof_phase_composite_profiles_{scope.window_id}.csv")
        all_comp.append(comp)

    _log("[cross] Write cross-window outputs")
    cross_tables = {
        "eof_mode_summary_all_windows.csv": pd.concat(all_mode, ignore_index=True) if all_mode else pd.DataFrame(),
        "eof_object_energy_contribution_all_windows.csv": pd.concat(all_energy, ignore_index=True) if all_energy else pd.DataFrame(),
        "pc_scores_all_windows.csv": pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame(),
        "eof_mode_stability_all_windows.csv": pd.concat(all_stability, ignore_index=True) if all_stability else pd.DataFrame(),
        "pc_year_leverage_all_windows.csv": pd.concat(all_leverage, ignore_index=True) if all_leverage else pd.DataFrame(),
        "pc_phase_group_membership_all_windows.csv": pd.concat(all_phase, ignore_index=True) if all_phase else pd.DataFrame(),
        "pc_vs_v9_instability_targets_all_windows.csv": pd.concat(all_obj_pc, ignore_index=True) if all_obj_pc else pd.DataFrame(),
        "pc_vs_pairwise_order_influence_all_windows.csv": pd.concat(all_pair_pc, ignore_index=True) if all_pair_pc else pd.DataFrame(),
        "pc_influence_year_overlap_all_windows.csv": pd.concat(all_overlap, ignore_index=True) if all_overlap else pd.DataFrame(),
        "pc_phase_group_object_peak_all_windows.csv": pd.concat(all_group_obj, ignore_index=True) if all_group_obj else pd.DataFrame(),
        "pc_phase_group_object_peak_bootstrap_all_windows.csv": pd.concat(all_group_boot, ignore_index=True) if all_group_boot else pd.DataFrame(),
        "pc_phase_group_pairwise_order_all_windows.csv": pd.concat(all_group_order, ignore_index=True) if all_group_order else pd.DataFrame(),
        "pc_phase_group_pairwise_synchrony_all_windows.csv": pd.concat(all_group_sync, ignore_index=True) if all_group_sync else pd.DataFrame(),
        "pc_phase_group_peak_skipped_all_windows.csv": pd.concat(all_group_skipped, ignore_index=True) if all_group_skipped else pd.DataFrame(),
        "eof_transition_mode_peak_order_evidence_all_windows.csv": pd.concat(all_evidence, ignore_index=True) if all_evidence else pd.DataFrame(),
        "eof_phase_composite_profiles_all_windows.csv": pd.concat(all_comp, ignore_index=True) if all_comp else pd.DataFrame(),
    }
    for fname, df in cross_tables.items():
        _safe_to_csv(df, out_cross / fname)

    run_meta = {
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "parent_version": "V9 peak_all_windows_v9_a",
        "modifies_v9": False,
        "reads_v9_outputs": True,
        "uses_single_year_peak_for_eof_inputs": False,
        "feature_basis": "whole-window multi-object year-anomaly profiles with object block equal weighting",
        "windows": cfg91.windows,
        "excluded_windows": EXCLUDED_WINDOWS,
        "state_included": False,
        "growth_included": False,
        "process_a_included": False,
        "config": asdict(cfg91),
        "source_v7_module_version": getattr(v7multi, "VERSION", "unknown"),
        "source_v7_output_tag": getattr(v7multi, "OUTPUT_TAG", "unknown"),
        "v9_1c_inputs_found": {k: not v.empty for k, v in v91c_tables.items()},
        "elapsed_seconds": round(time.time() - t0, 3),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(run_meta, out_root / "run_meta.json")
    _write_json(run_meta, out_cross / "run_meta.json")
    _write_summary(out_cross / "eof_transition_mode_audit_summary.md", cross_tables["eof_transition_mode_peak_order_evidence_all_windows.csv"], cross_tables["eof_mode_summary_all_windows.csv"], cross_tables["eof_mode_stability_all_windows.csv"], cfg91)
    _ensure_dir(log_dir).joinpath("run_progress.log").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Done. Outputs written to {out_root}")
