"""
V9.1_c bootstrap year-influence audit.

Purpose
-------
This is a read-only V9 audit branch.  It does NOT use single-year peak finding
as evidence.  Instead, it replays the V9 paired-year bootstrap while recording
which years were sampled in each bootstrap replicate, then asks which years
statistically push object peak days or pairwise peak-order outcomes.

Scientific boundary
-------------------
Outputs identify high-influence year candidates in the bootstrap mechanism.
They do not identify physical year types, causes, or mechanisms.  This module
never modifies V9 source files or V9 outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import os
import sys
import time

import numpy as np
import pandas as pd

VERSION = "bootstrap_year_influence_audit_v9_1_c"
OUTPUT_TAG = "bootstrap_year_influence_audit_v9_1_c"
DEFAULT_WINDOWS = ["W045", "W081", "W113", "W160"]
DEFAULT_OBJECTS = ["P", "V", "H", "Je", "Jw"]
PAIR_OBJECTS = DEFAULT_OBJECTS


@dataclass
class V91CConfig:
    windows: List[str]
    objects: List[str]
    bootstrap_n: int
    perm_n: int
    ridge_alpha: float
    log_every_bootstrap: int
    debug: bool
    v9_peak_output_tag: str = "peak_all_windows_v9_a"
    evidence_usable: float = 0.90
    evidence_credible: float = 0.95
    evidence_strict: float = 0.99


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_csv_env(name: str, default: Sequence[str]) -> List[str]:
    s = os.environ.get(name)
    if not s:
        return list(default)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out or list(default)


def _build_config() -> V91CConfig:
    debug = os.environ.get("V9_1C_DEBUG", "").strip() not in ("", "0", "false", "False")
    bootstrap_n = int(os.environ.get("V9_1C_BOOTSTRAP_N", os.environ.get("V9_PEAK_N_BOOTSTRAP", "1000")))
    if debug:
        bootstrap_n = int(os.environ.get("V9_1C_DEBUG_N_BOOTSTRAP", "80"))
    perm_n = int(os.environ.get("V9_1C_PERM_N", "500"))
    if debug:
        perm_n = int(os.environ.get("V9_1C_DEBUG_PERM_N", "100"))
    return V91CConfig(
        windows=_parse_csv_env("V9_1C_TARGET_WINDOWS", DEFAULT_WINDOWS),
        objects=_parse_csv_env("V9_1C_OBJECTS", DEFAULT_OBJECTS),
        bootstrap_n=bootstrap_n,
        perm_n=perm_n,
        ridge_alpha=float(os.environ.get("V9_1C_RIDGE_ALPHA", "1.0")),
        log_every_bootstrap=int(os.environ.get("V9_1C_LOG_EVERY_BOOTSTRAP", "50")),
        debug=debug,
    )


def _import_v9_and_v7(stage_root: Path):
    v9_src = stage_root / "V9" / "src"
    if not v9_src.exists():
        raise FileNotFoundError(f"Cannot find V9 src directory: {v9_src}")
    if str(v9_src) not in sys.path:
        sys.path.insert(0, str(v9_src))
    from stage_partition_v9 import peak_all_windows_v9_a as v9peak
    v7multi = v9peak._import_v7_module(stage_root)
    return v9peak, v7multi


def _make_cfg_for_v91c(v9peak, v7multi, v91_root: Path, cfg_c: V91CConfig):
    cfg = v9peak._make_v7_cfg_for_v9(v7multi)
    cfg.window_mode = "list"
    cfg.target_windows = ",".join(cfg_c.windows)
    cfg.bootstrap_n = int(cfg_c.bootstrap_n)
    cfg.log_every_bootstrap = int(cfg_c.log_every_bootstrap)
    # Keep V9 peak-only boundary.  We need bootstrap samples here, but not V7
    # curve/process samples.
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    if os.environ.get("V9_1C_SMOOTHED_FIELDS"):
        cfg.smoothed_fields_path = os.environ["V9_1C_SMOOTHED_FIELDS"]
    if os.environ.get("V9_1C_ACCEPTED_WINDOW_REGISTRY"):
        cfg.accepted_window_registry = os.environ["V9_1C_ACCEPTED_WINDOW_REGISTRY"]
        cfg.window_source = "registry"
    return cfg


def _load_scopes(v91_root: Path, v9peak, v7multi, cfg_v7, cfg_c: V91CConfig):
    stage_root = v91_root.parent
    v7_root = stage_root / "V7"
    tmp_audit = _ensure_dir(v91_root / "outputs" / OUTPUT_TAG / "_internal_scope_audit")
    wins = v7multi._load_accepted_windows(v7_root, tmp_audit, cfg_v7)
    scopes, validity = v7multi._build_window_scopes(wins, cfg_v7)
    run_scopes, run_audit = v7multi._filter_scopes_for_run(scopes, cfg_v7)
    # Defensive filter: exactly cfg_c windows, preserving V7 scope objects.
    want = set(cfg_c.windows)
    run_scopes = [s for s in run_scopes if s.window_id in want]
    return scopes, run_scopes, validity, run_audit


def _load_profiles(v91_root: Path, v9peak, v7multi, cfg_v7) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray]:
    stage_root = v91_root.parent
    v9_root = stage_root / "V9"
    smoothed = Path(cfg_v7.smoothed_fields_path) if getattr(cfg_v7, "smoothed_fields_path", None) else v9peak._default_smoothed_path(v9_root)
    if not smoothed.exists():
        smoothed = v9peak._default_smoothed_path(v9_root)
    if not smoothed.exists():
        raise FileNotFoundError(
            f"smoothed_fields.npz not found: {smoothed}. Set V9_1C_SMOOTHED_FIELDS or V9_PEAK_SMOOTHED_FIELDS."
        )
    fields, _audit = v7multi.clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for spec in v7multi.clean.OBJECT_SPECS:
        arr = v7multi.clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (prof, target_lat, weights)
    ny = next(iter(profiles.values()))[0].shape[0]
    if years is None:
        year_values = np.arange(ny, dtype=int)
    else:
        year_values = np.asarray(years)
        if year_values.shape[0] != ny:
            year_values = np.arange(ny, dtype=int)
    return profiles, year_values


def _selected_peak_from_candidates(v7multi, cand: pd.DataFrame, scope) -> Tuple[float, float, str]:
    if cand is None or cand.empty:
        return float("nan"), float("nan"), "no_candidate"
    sel = v7multi._select_main_candidate(cand, scope)
    if sel is None or sel.empty:
        return float("nan"), float("nan"), "no_selected_candidate"
    day = float(sel["selected_peak_day"].iloc[0]) if "selected_peak_day" in sel.columns else float("nan")
    score = float(sel["peak_score"].iloc[0]) if "peak_score" in sel.columns and pd.notna(sel["peak_score"].iloc[0]) else float("nan")
    cid = str(sel["candidate_id"].iloc[0]) if "candidate_id" in sel.columns else ""
    return day, score, cid


def _run_v9_bootstrap_replay_with_year_counts(
    v7multi,
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    year_values: np.ndarray,
    scope,
    cfg_v7,
    cfg_c: V91CConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay V9 peak bootstrap while saving bootstrap sample composition.

    This mirrors V7/V9 _run_detector_and_bootstrap semantics for selected peak
    days, but also stores bootstrap year counts for influence analysis.
    """
    ny = next(iter(profiles.values()))[0].shape[0]
    boot_indices = v7multi._make_bootstrap_indices(ny, scope, cfg_v7)
    year_count_rows: List[dict] = []
    peak_rows: List[dict] = []
    pair_rows: List[dict] = []
    selected_by_boot: Dict[int, Dict[str, float]] = {}

    for ib, idx in enumerate(boot_indices):
        counts = np.bincount(np.asarray(idx, dtype=int), minlength=ny)
        for iy, c in enumerate(counts):
            year_count_rows.append({
                "window_id": scope.window_id,
                "bootstrap_id": int(ib),
                "year_index": int(iy),
                "year": year_values[iy].item() if hasattr(year_values[iy], "item") else year_values[iy],
                "count_in_bootstrap": int(c),
                "present_flag": bool(c > 0),
            })
        selected_by_boot[ib] = {}
        for obj, (prof_by_year, _target_lat, _weights) in profiles.items():
            state = v7multi._raw_state_matrix_v7z_from_year_cube(prof_by_year, idx)
            _bscores, bcand = v7multi._run_original_v7z_detector_for_profile(state, cfg_v7, scope, obj)
            day, score, cid = _selected_peak_from_candidates(v7multi, bcand, scope)
            selected_by_boot[ib][obj] = day
            peak_rows.append({
                "window_id": scope.window_id,
                "bootstrap_id": int(ib),
                "object": obj,
                "selected_peak_day": day,
                "peak_score": score,
                "selected_candidate_id": cid,
            })
        objs = [o for o in PAIR_OBJECTS if o in selected_by_boot[ib]]
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                pa = selected_by_boot[ib].get(a, np.nan)
                pb = selected_by_boot[ib].get(b, np.nan)
                delta = float(pb - pa) if np.isfinite(pa) and np.isfinite(pb) else np.nan
                pair_rows.append({
                    "window_id": scope.window_id,
                    "bootstrap_id": int(ib),
                    "object_A": a,
                    "object_B": b,
                    "peak_A": pa,
                    "peak_B": pb,
                    "delta_B_minus_A": delta,
                    "A_earlier_flag": bool(delta > 0) if np.isfinite(delta) else False,
                    "B_earlier_flag": bool(delta < 0) if np.isfinite(delta) else False,
                    "same_day_flag": bool(delta == 0) if np.isfinite(delta) else False,
                })
        if cfg_c.log_every_bootstrap > 0 and (ib + 1) % cfg_c.log_every_bootstrap == 0:
            _log(f"  replay bootstrap {scope.window_id}: {ib + 1}/{cfg_c.bootstrap_n}")
    return pd.DataFrame(year_count_rows), pd.DataFrame(peak_rows), pd.DataFrame(pair_rows)


def _read_v9_window_table(v9_root: Path, window_id: str, fname: str) -> pd.DataFrame:
    p = v9_root / "outputs" / "peak_all_windows_v9_a" / "per_window" / window_id / fname
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _replay_regression_audit(v9_root: Path, scope, replay_peak: pd.DataFrame, replay_pair: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    wid = scope.window_id
    v9_boot = _read_v9_window_table(v9_root, wid, f"bootstrap_selected_peak_days_{wid}.csv")
    if v9_boot.empty:
        rows.append({"window_id": wid, "table": "bootstrap_selected_peak_days", "status": "v9_reference_missing"})
    else:
        ref = v9_boot.groupby("object")["selected_peak_day"].agg(
            n_v9="count",
            median_v9="median",
            q025_v9=lambda x: float(np.nanquantile(x.to_numpy(float), 0.025)) if np.isfinite(x.to_numpy(float)).any() else np.nan,
            q975_v9=lambda x: float(np.nanquantile(x.to_numpy(float), 0.975)) if np.isfinite(x.to_numpy(float)).any() else np.nan,
        ).reset_index()
        rep = replay_peak.groupby("object")["selected_peak_day"].agg(
            n_replay="count",
            median_replay="median",
            q025_replay=lambda x: float(np.nanquantile(x.to_numpy(float), 0.025)) if np.isfinite(x.to_numpy(float)).any() else np.nan,
            q975_replay=lambda x: float(np.nanquantile(x.to_numpy(float), 0.975)) if np.isfinite(x.to_numpy(float)).any() else np.nan,
        ).reset_index()
        m = ref.merge(rep, on="object", how="outer")
        for _, r in m.iterrows():
            max_diff = np.nanmax(np.abs([
                r.get("median_v9", np.nan) - r.get("median_replay", np.nan),
                r.get("q025_v9", np.nan) - r.get("q025_replay", np.nan),
                r.get("q975_v9", np.nan) - r.get("q975_replay", np.nan),
            ])) if pd.notna(r.get("median_v9", np.nan)) and pd.notna(r.get("median_replay", np.nan)) else np.nan
            status = "pass" if np.isfinite(max_diff) and max_diff <= 1.0e-9 else "difference_or_n_mismatch"
            rows.append({"window_id": wid, "table": "object_bootstrap_distribution", "object": r.get("object"), "status": status, "max_day_summary_diff": max_diff, **r.to_dict()})

    v9_order = _read_v9_window_table(v9_root, wid, f"pairwise_peak_order_test_{wid}.csv")
    if v9_order.empty:
        rows.append({"window_id": wid, "table": "pairwise_peak_order", "status": "v9_reference_missing"})
    else:
        rep_rows = []
        for (a, b), g in replay_pair.groupby(["object_A", "object_B"]):
            valid = np.isfinite(g["delta_B_minus_A"].to_numpy(float))
            n = int(valid.sum())
            if n == 0:
                pa = pb = ps = np.nan
            else:
                d = g.loc[valid, "delta_B_minus_A"].to_numpy(float)
                pa = float(np.mean(d > 0)); pb = float(np.mean(d < 0)); ps = float(np.mean(d == 0))
            rep_rows.append({"object_A": a, "object_B": b, "P_A_earlier_replay": pa, "P_B_earlier_replay": pb, "P_same_day_replay": ps, "n_replay": n})
        rep = pd.DataFrame(rep_rows)
        m = v9_order.merge(rep, on=["object_A", "object_B"], how="outer")
        for _, r in m.iterrows():
            diffs = []
            for c, cr in [("P_A_earlier", "P_A_earlier_replay"), ("P_B_earlier", "P_B_earlier_replay"), ("P_same_day", "P_same_day_replay")]:
                if c in r and cr in r and pd.notna(r[c]) and pd.notna(r[cr]):
                    diffs.append(abs(float(r[c]) - float(r[cr])))
            max_diff = max(diffs) if diffs else np.nan
            status = "pass" if np.isfinite(max_diff) and max_diff <= 1.0e-9 else "difference_or_n_mismatch"
            rows.append({"window_id": wid, "table": "pairwise_peak_order", "object_A": r.get("object_A"), "object_B": r.get("object_B"), "status": status, "max_probability_diff": max_diff})
    return pd.DataFrame(rows)


def _safe_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def _safe_prob(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def _ridge_coefficients(count_matrix: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(count_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 5:
        return np.full(X.shape[1], np.nan)
    X = X[valid]
    y = y[valid]
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    # Standardize columns to comparable scale; zero-variance columns stay zero.
    sd = Xc.std(axis=0, ddof=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Xs = Xc / sd_safe
    XtX = Xs.T @ Xs
    beta = np.linalg.solve(XtX + float(alpha) * np.eye(XtX.shape[0]), Xs.T @ yc)
    return beta / sd_safe


def _effect_one_year(counts: np.ndarray, y: np.ndarray, year_col: int, mode: str = "presence") -> float:
    c = counts[:, year_col]
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if mode == "presence":
        g1 = valid & (c > 0)
        g0 = valid & (c == 0)
    else:
        g1 = valid & (c >= 2)
        g0 = valid & (c == 0)
        if g1.sum() < 5:
            g1 = valid & (c > 0)
    if g1.sum() < 5 or g0.sum() < 5:
        return np.nan
    return float(np.mean(y[g1]) - np.mean(y[g0]))


def _perm_percentiles_for_target(
    counts: np.ndarray,
    y: np.ndarray,
    obs_effects: np.ndarray,
    perm_n: int,
    seed: int,
    mode: str = "presence",
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    valid_y = y[np.isfinite(y)]
    if valid_y.size < 10 or perm_n <= 0:
        return np.full(obs_effects.shape, np.nan)
    rng = np.random.default_rng(seed)
    null_abs: List[np.ndarray] = []
    y_work = y.copy()
    valid_idx = np.where(np.isfinite(y_work))[0]
    for _ in range(int(perm_n)):
        yp = y_work.copy()
        yp[valid_idx] = rng.permutation(yp[valid_idx])
        eff = np.array([_effect_one_year(counts, yp, j, mode=mode) for j in range(counts.shape[1])], dtype=float)
        null_abs.append(np.abs(eff))
    null = np.concatenate([x[np.isfinite(x)] for x in null_abs if np.isfinite(x).any()]) if null_abs else np.array([], dtype=float)
    if null.size == 0:
        return np.full(obs_effects.shape, np.nan)
    return np.array([float(np.mean(null <= abs(e))) if np.isfinite(e) else np.nan for e in obs_effects], dtype=float)


def _evidence_level(percentile: float, cfg: V91CConfig) -> str:
    if not np.isfinite(percentile):
        return "insufficient"
    if percentile >= cfg.evidence_strict:
        return "strict_99"
    if percentile >= cfg.evidence_credible:
        return "credible_95"
    if percentile >= cfg.evidence_usable:
        return "usable_90"
    return "weak_or_none"


def _direction_from_effect(effect: float, positive_label: str, negative_label: str) -> str:
    if not np.isfinite(effect) or abs(effect) < 1.0e-12:
        return "neutral_or_unclear"
    return positive_label if effect > 0 else negative_label


def _counts_matrix(year_counts: pd.DataFrame) -> Tuple[np.ndarray, List[object], List[int]]:
    piv = year_counts.pivot_table(index="bootstrap_id", columns="year", values="count_in_bootstrap", fill_value=0, aggfunc="sum").sort_index()
    return piv.to_numpy(dtype=float), list(piv.columns), list(piv.index)


def _object_peak_year_influence(
    year_counts: pd.DataFrame,
    peak_samples: pd.DataFrame,
    cfg: V91CConfig,
    window_id: str,
) -> pd.DataFrame:
    counts, years, boot_ids = _counts_matrix(year_counts)
    id_to_row = {bid: i for i, bid in enumerate(boot_ids)}
    rows: List[dict] = []
    for obj, g in peak_samples.groupby("object"):
        y_by_id = {int(r["bootstrap_id"]): float(r["selected_peak_day"]) for _, r in g.iterrows()}
        y = np.array([y_by_id.get(bid, np.nan) for bid in boot_ids], dtype=float)
        valid = np.isfinite(y)
        if valid.sum() == 0:
            continue
        q25 = float(np.nanquantile(y, 0.25))
        q75 = float(np.nanquantile(y, 0.75))
        early = np.where(np.isfinite(y), (y <= q25).astype(float), np.nan)
        late = np.where(np.isfinite(y), (y >= q75).astype(float), np.nan)
        ridge_peak = _ridge_coefficients(counts, y, cfg.ridge_alpha)
        effects_peak = np.array([_effect_one_year(counts, y, j, "presence") for j in range(len(years))])
        effects_peak_high = np.array([_effect_one_year(counts, y, j, "high") for j in range(len(years))])
        effects_early = np.array([_effect_one_year(counts, early, j, "presence") for j in range(len(years))])
        effects_late = np.array([_effect_one_year(counts, late, j, "presence") for j in range(len(years))])
        perc_peak = _perm_percentiles_for_target(counts, y, effects_peak, cfg.perm_n, seed=101 + hash((window_id, obj)) % 100000, mode="presence")
        perc_early = _perm_percentiles_for_target(counts, early, effects_early, cfg.perm_n, seed=201 + hash((window_id, obj)) % 100000, mode="presence")
        perc_late = _perm_percentiles_for_target(counts, late, effects_late, cfg.perm_n, seed=301 + hash((window_id, obj)) % 100000, mode="presence")
        for j, yr in enumerate(years):
            c = counts[:, j]
            present = valid & (c > 0)
            absent = valid & (c == 0)
            high = valid & (c >= 2)
            if high.sum() < 5:
                high = valid & (c > 0)
            # Pick strongest among peak-day, early, and late signals for summary level.
            percs = [perc_peak[j], perc_early[j], perc_late[j]]
            max_perc = np.nanmax(percs) if np.isfinite(percs).any() else np.nan
            # Direction by day effect first; if weak, use late/early effects.
            if np.isfinite(effects_peak[j]) and abs(effects_peak[j]) > 1.0e-12:
                direction = _direction_from_effect(effects_peak[j], "push_late_peak", "push_early_peak")
            elif np.isfinite(effects_late[j]) and abs(effects_late[j]) >= abs(effects_early[j] if np.isfinite(effects_early[j]) else 0):
                direction = _direction_from_effect(effects_late[j], "push_late_peak", "reduce_late_peak")
            elif np.isfinite(effects_early[j]):
                direction = _direction_from_effect(effects_early[j], "push_early_peak", "reduce_early_peak")
            else:
                direction = "neutral_or_unclear"
            rows.append({
                "window_id": window_id,
                "object": obj,
                "year": yr,
                "n_bootstrap": int(valid.sum()),
                "n_present": int(present.sum()),
                "n_absent": int(absent.sum()),
                "mean_peak_present": _safe_mean(y[present]),
                "mean_peak_absent": _safe_mean(y[absent]),
                "presence_effect_on_peak_day": effects_peak[j],
                "peak_q25_for_early_definition": q25,
                "peak_q75_for_late_definition": q75,
                "P_early_present": _safe_prob(early[present]),
                "P_early_absent": _safe_prob(early[absent]),
                "presence_effect_on_early_probability": effects_early[j],
                "P_late_present": _safe_prob(late[present]),
                "P_late_absent": _safe_prob(late[absent]),
                "presence_effect_on_late_probability": effects_late[j],
                "n_high_count": int(high.sum()),
                "high_count_effect_on_peak_day": effects_peak_high[j],
                "ridge_coef_peak_day": ridge_peak[j] if j < len(ridge_peak) else np.nan,
                "permutation_percentile_peak_day": perc_peak[j],
                "permutation_percentile_early_probability": perc_early[j],
                "permutation_percentile_late_probability": perc_late[j],
                "max_permutation_percentile": max_perc,
                "influence_direction": direction,
                "influence_evidence_level": _evidence_level(max_perc, cfg),
                "method_role": "bootstrap_year_influence_candidate_not_physical_type",
            })
    return pd.DataFrame(rows)


def _pairwise_order_year_influence(
    year_counts: pd.DataFrame,
    pair_samples: pd.DataFrame,
    cfg: V91CConfig,
    window_id: str,
) -> pd.DataFrame:
    counts, years, boot_ids = _counts_matrix(year_counts)
    rows: List[dict] = []
    for (a, b), g in pair_samples.groupby(["object_A", "object_B"]):
        a_by_id = {int(r["bootstrap_id"]): float(bool(r["A_earlier_flag"])) for _, r in g.iterrows()}
        b_by_id = {int(r["bootstrap_id"]): float(bool(r["B_earlier_flag"])) for _, r in g.iterrows()}
        yA = np.array([a_by_id.get(bid, np.nan) for bid in boot_ids], dtype=float)
        yB = np.array([b_by_id.get(bid, np.nan) for bid in boot_ids], dtype=float)
        ridge_A = _ridge_coefficients(counts, yA, cfg.ridge_alpha)
        ridge_B = _ridge_coefficients(counts, yB, cfg.ridge_alpha)
        eff_A = np.array([_effect_one_year(counts, yA, j, "presence") for j in range(len(years))])
        eff_B = np.array([_effect_one_year(counts, yB, j, "presence") for j in range(len(years))])
        eff_A_high = np.array([_effect_one_year(counts, yA, j, "high") for j in range(len(years))])
        eff_B_high = np.array([_effect_one_year(counts, yB, j, "high") for j in range(len(years))])
        perc_A = _perm_percentiles_for_target(counts, yA, eff_A, cfg.perm_n, seed=401 + hash((window_id, a, b, "A")) % 100000, mode="presence")
        perc_B = _perm_percentiles_for_target(counts, yB, eff_B, cfg.perm_n, seed=501 + hash((window_id, a, b, "B")) % 100000, mode="presence")
        valid = np.isfinite(yA) & np.isfinite(yB)
        for j, yr in enumerate(years):
            c = counts[:, j]
            present = valid & (c > 0)
            absent = valid & (c == 0)
            high = valid & (c >= 2)
            if high.sum() < 5:
                high = valid & (c > 0)
            if np.nan_to_num(perc_A[j], nan=-1) >= np.nan_to_num(perc_B[j], nan=-1):
                max_perc = perc_A[j]
                direction = _direction_from_effect(eff_A[j], f"push_{a}_earlier", f"reduce_{a}_earlier")
            else:
                max_perc = perc_B[j]
                direction = _direction_from_effect(eff_B[j], f"push_{b}_earlier", f"reduce_{b}_earlier")
            # If both positive, choose larger effect direction.
            if np.isfinite(eff_A[j]) and np.isfinite(eff_B[j]) and eff_A[j] > 0 and eff_B[j] > 0:
                direction = f"push_{a}_earlier" if eff_A[j] >= eff_B[j] else f"push_{b}_earlier"
            rows.append({
                "window_id": window_id,
                "object_A": a,
                "object_B": b,
                "year": yr,
                "n_bootstrap": int(valid.sum()),
                "n_present": int(present.sum()),
                "n_absent": int(absent.sum()),
                "P_A_earlier_present": _safe_prob(yA[present]),
                "P_A_earlier_absent": _safe_prob(yA[absent]),
                "presence_effect_on_A_earlier": eff_A[j],
                "P_B_earlier_present": _safe_prob(yB[present]),
                "P_B_earlier_absent": _safe_prob(yB[absent]),
                "presence_effect_on_B_earlier": eff_B[j],
                "n_high_count": int(high.sum()),
                "high_count_effect_on_A_earlier": eff_A_high[j],
                "high_count_effect_on_B_earlier": eff_B_high[j],
                "ridge_coef_A_earlier": ridge_A[j] if j < len(ridge_A) else np.nan,
                "ridge_coef_B_earlier": ridge_B[j] if j < len(ridge_B) else np.nan,
                "permutation_percentile_A_earlier": perc_A[j],
                "permutation_percentile_B_earlier": perc_B[j],
                "max_permutation_percentile": max_perc,
                "influence_direction": direction,
                "influence_evidence_level": _evidence_level(max_perc, cfg),
                "method_role": "bootstrap_year_influence_candidate_not_physical_type",
            })
    return pd.DataFrame(rows)


def _influential_year_sets(obj_inf: pd.DataFrame, pair_inf: pd.DataFrame, cfg: V91CConfig, window_id: str) -> pd.DataFrame:
    rows: List[dict] = []
    if obj_inf is not None and not obj_inf.empty:
        for (obj, direction, level), g in obj_inf[obj_inf["influence_evidence_level"].isin(["usable_90", "credible_95", "strict_99"])].groupby(["object", "influence_direction", "influence_evidence_level"]):
            rows.append({
                "window_id": window_id,
                "target_type": "object_peak",
                "target_name": obj,
                "influence_group": direction,
                "evidence_level": level,
                "years": ";".join(map(str, g.sort_values("max_permutation_percentile", ascending=False)["year"].tolist())),
                "n_years": int(len(g)),
                "mean_effect": float(g["presence_effect_on_peak_day"].mean()),
                "max_percentile": float(g["max_permutation_percentile"].max()),
                "notes": "statistical high-influence year candidates only; not physical year types",
            })
    if pair_inf is not None and not pair_inf.empty:
        for (a, b, direction, level), g in pair_inf[pair_inf["influence_evidence_level"].isin(["usable_90", "credible_95", "strict_99"])].groupby(["object_A", "object_B", "influence_direction", "influence_evidence_level"]):
            rows.append({
                "window_id": window_id,
                "target_type": "pairwise_order",
                "target_name": f"{a}-{b}",
                "influence_group": direction,
                "evidence_level": level,
                "years": ";".join(map(str, g.sort_values("max_permutation_percentile", ascending=False)["year"].tolist())),
                "n_years": int(len(g)),
                "mean_effect": float(np.nanmax([g["presence_effect_on_A_earlier"].abs().mean(), g["presence_effect_on_B_earlier"].abs().mean()])),
                "max_percentile": float(g["max_permutation_percentile"].max()),
                "notes": "statistical high-influence year candidates only; not physical year types",
            })
    return pd.DataFrame(rows)


def _influence_summary(obj_inf: pd.DataFrame, pair_inf: pd.DataFrame, window_id: str) -> pd.DataFrame:
    rows: List[dict] = []
    levels = ["usable_90", "credible_95", "strict_99"]
    if obj_inf is not None and not obj_inf.empty:
        for obj, g in obj_inf.groupby("object"):
            gg = g[g["influence_evidence_level"].isin(levels)].copy()
            top = gg.sort_values("max_permutation_percentile", ascending=False).head(8) if not gg.empty else pd.DataFrame()
            rows.append({
                "window_id": window_id,
                "object_or_pair": obj,
                "target_type": "object_peak",
                "n_usable_or_stronger_years": int((g["influence_evidence_level"] == "usable_90").sum() + (g["influence_evidence_level"] == "credible_95").sum() + (g["influence_evidence_level"] == "strict_99").sum()),
                "n_credible_years": int((g["influence_evidence_level"] == "credible_95").sum()),
                "n_strict_years": int((g["influence_evidence_level"] == "strict_99").sum()),
                "top_influential_years": ";".join(map(str, top["year"].tolist())) if not top.empty else "",
                "top_influence_directions": ";".join(top["influence_direction"].astype(str).tolist()) if not top.empty else "",
                "does_year_influence_explain_instability": _summary_level(top),
                "recommended_next_check": "map high-influence years to field/background anomalies; do not call them physical types yet" if not top.empty else "no strong year influence candidate in this audit",
            })
    if pair_inf is not None and not pair_inf.empty:
        for (a, b), g in pair_inf.groupby(["object_A", "object_B"]):
            gg = g[g["influence_evidence_level"].isin(levels)].copy()
            top = gg.sort_values("max_permutation_percentile", ascending=False).head(8) if not gg.empty else pd.DataFrame()
            rows.append({
                "window_id": window_id,
                "object_or_pair": f"{a}-{b}",
                "target_type": "pairwise_order",
                "n_usable_or_stronger_years": int((g["influence_evidence_level"] == "usable_90").sum() + (g["influence_evidence_level"] == "credible_95").sum() + (g["influence_evidence_level"] == "strict_99").sum()),
                "n_credible_years": int((g["influence_evidence_level"] == "credible_95").sum()),
                "n_strict_years": int((g["influence_evidence_level"] == "strict_99").sum()),
                "top_influential_years": ";".join(map(str, top["year"].tolist())) if not top.empty else "",
                "top_influence_directions": ";".join(top["influence_direction"].astype(str).tolist()) if not top.empty else "",
                "does_year_influence_explain_instability": _summary_level(top),
                "recommended_next_check": "inspect high-influence years before any physical interpretation" if not top.empty else "no strong year influence candidate in this audit",
            })
    return pd.DataFrame(rows)


def _summary_level(top: pd.DataFrame) -> str:
    if top is None or top.empty:
        return "not_supported"
    levels = top["influence_evidence_level"].astype(str).tolist()
    if "strict_99" in levels:
        return "strong_candidate"
    if "credible_95" in levels:
        return "moderate_candidate"
    if "usable_90" in levels:
        return "weak_hint"
    return "not_supported"


def _write_window_summary_md(path: Path, window_id: str, obj_inf: pd.DataFrame, pair_inf: pd.DataFrame, replay_audit: pd.DataFrame, cfg: V91CConfig) -> None:
    n_obj = 0 if obj_inf is None or obj_inf.empty else int(obj_inf["influence_evidence_level"].isin(["usable_90", "credible_95", "strict_99"]).sum())
    n_pair = 0 if pair_inf is None or pair_inf.empty else int(pair_inf["influence_evidence_level"].isin(["usable_90", "credible_95", "strict_99"]).sum())
    replay_status = "unknown"
    if replay_audit is not None and not replay_audit.empty:
        if (replay_audit["status"].astype(str) == "pass").all():
            replay_status = "pass"
        elif replay_audit["status"].astype(str).str.contains("missing").any():
            replay_status = "reference_missing_or_partial"
        else:
            replay_status = "differences_found"
    txt = f"""# V9.1_c bootstrap year-influence audit: {window_id}

This audit replays the V9 paired-year bootstrap while recording sampled-year composition.
It identifies statistical high-influence year candidates for object peak day and pairwise peak-order outcomes.
It does **not** use single-year peak as evidence and does **not** identify physical year types.

- Bootstrap N: {cfg.bootstrap_n}
- Permutation N: {cfg.perm_n}
- Replay audit status: {replay_status}
- Object-level usable-or-stronger influence rows: {n_obj}
- Pairwise-order usable-or-stronger influence rows: {n_pair}

Interpretation boundary: high-influence years are candidates for subsequent field/background checks only.
"""
    path.write_text(txt, encoding="utf-8")


def run_bootstrap_year_influence_audit_v9_1_c(v91_root: Path | str) -> None:
    v91_root = Path(v91_root)
    stage_root = v91_root.parent
    v9_root = stage_root / "V9"
    cfg_c = _build_config()
    t0 = time.time()

    out_root = _ensure_dir(v91_root / "outputs" / OUTPUT_TAG)
    out_per = _ensure_dir(out_root / "per_window")
    out_cross = _ensure_dir(out_root / "cross_window")
    log_dir = _ensure_dir(v91_root / "logs" / OUTPUT_TAG)

    _log("[V9.1_c 1/6] Import V9/V7 and load V9 peak context")
    v9peak, v7multi = _import_v9_and_v7(stage_root)
    cfg_v7 = _make_cfg_for_v91c(v9peak, v7multi, v91_root, cfg_c)
    scopes_all, run_scopes, validity, run_audit = _load_scopes(v91_root, v9peak, v7multi, cfg_v7, cfg_c)
    _safe_to_csv(pd.DataFrame([asdict(s) for s in run_scopes]), out_cross / "run_window_scope_registry_v9_1_c.csv")
    _safe_to_csv(validity, out_cross / "window_scope_validity_audit_v9_1_c.csv")
    _safe_to_csv(run_audit, out_cross / "run_window_selection_audit_v9_1_c.csv")

    _log("[V9.1_c 2/6] Load V9-source profiles")
    profiles, years = _load_profiles(v91_root, v9peak, v7multi, cfg_v7)
    _write_json({
        "version": VERSION,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modifies_v9": False,
        "reads_v9_outputs": True,
        "replays_v9_bootstrap_with_year_counts": True,
        "uses_single_year_peak": False,
        "state_included": False,
        "growth_included": False,
        "process_a_included": False,
        "windows": cfg_c.windows,
        "bootstrap_n": cfg_c.bootstrap_n,
        "perm_n": cfg_c.perm_n,
        "ridge_alpha": cfg_c.ridge_alpha,
        "interpretation_boundary": "statistical year influence candidates only; not physical year types",
    }, out_root / "run_meta.json")

    all_obj: List[pd.DataFrame] = []
    all_pair: List[pd.DataFrame] = []
    all_sets: List[pd.DataFrame] = []
    all_summary: List[pd.DataFrame] = []
    all_replay: List[pd.DataFrame] = []
    all_counts: List[pd.DataFrame] = []

    for iw, scope in enumerate(run_scopes, start=1):
        wid = scope.window_id
        _log(f"[V9.1_c 3/6] {wid} replay V9 bootstrap with year-count recording ({iw}/{len(run_scopes)})")
        out_win = _ensure_dir(out_per / wid)
        year_counts, peak_samples, pair_samples = _run_v9_bootstrap_replay_with_year_counts(v7multi, profiles, years, scope, cfg_v7, cfg_c)
        _safe_to_csv(year_counts, out_win / f"bootstrap_sample_year_counts_{wid}.csv")
        _safe_to_csv(peak_samples, out_win / f"bootstrap_object_peak_samples_{wid}.csv")
        _safe_to_csv(pair_samples, out_win / f"bootstrap_pairwise_order_samples_{wid}.csv")
        all_counts.append(year_counts)

        _log(f"[V9.1_c 4/6] {wid} replay audit against V9")
        replay_audit = _replay_regression_audit(v9_root, scope, peak_samples, pair_samples)
        _safe_to_csv(replay_audit, out_win / f"v9_replay_bootstrap_regression_audit_{wid}.csv")
        all_replay.append(replay_audit)

        _log(f"[V9.1_c 5/6] {wid} compute object/pair year-influence scores")
        obj_inf = _object_peak_year_influence(year_counts, peak_samples, cfg_c, wid)
        pair_inf = _pairwise_order_year_influence(year_counts, pair_samples, cfg_c, wid)
        sets = _influential_year_sets(obj_inf, pair_inf, cfg_c, wid)
        summary = _influence_summary(obj_inf, pair_inf, wid)
        _safe_to_csv(obj_inf, out_win / f"object_peak_year_influence_{wid}.csv")
        _safe_to_csv(pair_inf, out_win / f"pairwise_order_year_influence_{wid}.csv")
        _safe_to_csv(sets, out_win / f"influential_year_sets_{wid}.csv")
        _safe_to_csv(summary, out_win / f"bootstrap_year_influence_summary_{wid}.csv")
        _write_window_summary_md(out_win / f"bootstrap_year_influence_summary_{wid}.md", wid, obj_inf, pair_inf, replay_audit, cfg_c)
        all_obj.append(obj_inf)
        all_pair.append(pair_inf)
        all_sets.append(sets)
        all_summary.append(summary)

    _log("[V9.1_c 6/6] Write cross-window outputs")
    _safe_to_csv(pd.concat(all_obj, ignore_index=True) if all_obj else pd.DataFrame(), out_cross / "object_peak_year_influence_all_windows.csv")
    _safe_to_csv(pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame(), out_cross / "pairwise_order_year_influence_all_windows.csv")
    _safe_to_csv(pd.concat(all_sets, ignore_index=True) if all_sets else pd.DataFrame(), out_cross / "influential_year_sets_all_windows.csv")
    _safe_to_csv(pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame(), out_cross / "bootstrap_year_influence_summary_all_windows.csv")
    _safe_to_csv(pd.concat(all_replay, ignore_index=True) if all_replay else pd.DataFrame(), out_cross / "v9_replay_bootstrap_regression_audit_all_windows.csv")
    # This can be large but is crucial for audit reproducibility.
    _safe_to_csv(pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame(), out_cross / "bootstrap_sample_year_counts_all_windows.csv")

    elapsed = time.time() - t0
    _write_json({
        "version": VERSION,
        "elapsed_seconds": elapsed,
        "windows_processed": [s.window_id for s in run_scopes],
        "bootstrap_n": cfg_c.bootstrap_n,
        "perm_n": cfg_c.perm_n,
        "modifies_v9": False,
        "uses_single_year_peak": False,
        "output_root": str(out_root),
    }, out_root / "summary.json")
    (log_dir / "last_run.txt").write_text(f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')} output={out_root}\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    run_bootstrap_year_influence_audit_v9_1_c(Path(__file__).resolve().parents[2])
