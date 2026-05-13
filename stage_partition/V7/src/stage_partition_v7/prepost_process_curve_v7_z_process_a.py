"""
V7-z-prepost-process-a: curve-structure state/growth process diagnostics for W45.

This module deliberately does NOT change the V7-z detector, accepted-window
registry, C0/C1/C2 baselines, or S/G curve definitions.  It replaces the old
winner-style pre/post middle layer with curve-structure diagnostics based on
paired-year bootstrap classifications of:

    state:  Delta S_AB(t) = S_A(t) - S_B(t)
    growth: Delta G_AB(t) = G_A(t) - G_B(t)

Main boundary
-------------
- W45 only by default.
- Profile only; no 2D mirror.
- early/core/late legacy winner tables are not used for main interpretation.
- 95% bootstrap structure support is required for supported main results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import time

import numpy as np
import pandas as pd

try:
    from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as multi
    from stage_partition_v7 import W45_multi_object_prepost_clean_mainline_v7_z_clean as clean
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "prepost_process_curve_v7_z_process_a requires the V7-z multiwin-a and clean modules."
    ) from exc

VERSION = "v7_z_prepost_process_a"
OUTPUT_TAG = "prepost_process_curve_v7_z_process_a"
EPS = 1.0e-12
OBJECTS = ["P", "V", "H", "Je", "Jw"]
OBJECT_ORDER_PAIRS = multi.OBJECT_ORDER_PAIRS
BRANCHES = ["dist", "pattern"]

STATE_STRUCTURES = [
    "parallel_state_progress",
    "persistent_A_more_postlike",
    "persistent_B_more_postlike",
    "A_front_B_catchup",
    "B_front_A_catchup",
    "state_reversal_A_to_B",
    "state_reversal_B_to_A",
    "multi_crossing_unstable",
    "state_unresolved",
]

OBJECT_GROWTH_STRUCTURES = [
    "single_growth_episode",
    "multi_stage_growth",
    "weak_or_no_positive_growth",
    "growth_unresolved",
]

ROLLBACK_STRUCTURES = [
    "rollback_or_regression",
    "minor_negative_fluctuation",
    "no_meaningful_rollback",
]

PAIR_GROWTH_STRUCTURES = [
    "synchronized_growth",
    "A_growth_front_B_catchup",
    "B_growth_front_A_catchup",
    "A_growth_front_then_synchronized",
    "B_growth_front_then_synchronized",
    "alternating_growth",
    "growth_unresolved",
]


@dataclass
class PrepostProcessConfig:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    window_mode: str = "w45"
    target_windows: str = "W045,45"
    window_source: str = "hardcoded"
    bootstrap_n: int = 1000
    random_seed: int = 42
    log_every_bootstrap: int = 50
    support_supported: float = 0.95
    support_tendency: float = 0.90
    support_exploratory: float = 0.80
    min_episode_len: int = 3
    merge_short_gap_days: int = 1
    main_episode_mass_ratio: float = 0.70
    multi_stage_min_secondary_ratio: float = 0.20
    rollback_min_negative_ratio: float = 0.10
    near_state_sd_factor: float = 1.0
    growth_noise_sd_factor: float = 1.0
    parallel_duration_ratio: float = 0.70
    persistent_duration_ratio: float = 0.70
    opposite_duration_max_ratio: float = 0.10
    sync_growth_mass_ratio: float = 0.70
    pattern_dynamic_low_abs: float = 0.02
    severe_overshoot_margin: float = 0.25
    save_bootstrap_curve_samples: bool = False
    smoothed_fields_path: Optional[str] = None

    @staticmethod
    def from_env() -> "PrepostProcessConfig":
        cfg = PrepostProcessConfig()
        cfg.smoothed_fields_path = os.environ.get("V7_PROCESS_SMOOTHED_FIELDS") or os.environ.get("V7_MULTI_SMOOTHED_FIELDS") or os.environ.get("V7Z_SMOOTHED_FIELDS")
        if os.environ.get("V7_PROCESS_N_BOOTSTRAP"):
            cfg.bootstrap_n = int(os.environ["V7_PROCESS_N_BOOTSTRAP"])
        if os.environ.get("V7_PROCESS_DEBUG_N_BOOTSTRAP"):
            cfg.bootstrap_n = int(os.environ["V7_PROCESS_DEBUG_N_BOOTSTRAP"])
        if os.environ.get("V7_PROCESS_RANDOM_SEED"):
            cfg.random_seed = int(os.environ["V7_PROCESS_RANDOM_SEED"])
        if os.environ.get("V7_PROCESS_LOG_EVERY_BOOTSTRAP"):
            cfg.log_every_bootstrap = int(os.environ["V7_PROCESS_LOG_EVERY_BOOTSTRAP"])
        if os.environ.get("V7_PROCESS_SAVE_BOOTSTRAP_CURVES") == "1":
            cfg.save_bootstrap_curve_samples = True
        return cfg


@dataclass
class Episode:
    label: str
    start: int
    end: int
    length: int
    mass: float = 0.0


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _support_class(support: float, cfg: PrepostProcessConfig) -> str:
    if not np.isfinite(support):
        return "unresolved"
    if support >= cfg.support_supported:
        return "supported"
    if support >= cfg.support_tendency:
        return "tendency"
    if support >= cfg.support_exploratory:
        return "exploratory_signal"
    return "unresolved"


def _quantile(vals: Sequence[float], q: float) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _median(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _build_multi_cfg(cfg: PrepostProcessConfig) -> multi.MultiWinConfig:
    mcfg = multi.MultiWinConfig()
    mcfg.window_mode = cfg.window_mode
    mcfg.target_windows = cfg.target_windows
    mcfg.window_source = cfg.window_source
    mcfg.run_2d = False
    mcfg.run_w45_profile_order_tests = False
    mcfg.bootstrap_n = cfg.bootstrap_n
    mcfg.random_seed = cfg.random_seed
    mcfg.log_every_bootstrap = cfg.log_every_bootstrap
    mcfg.smoothed_fields_path = cfg.smoothed_fields_path
    return mcfg


def _load_w45_profiles(v7_root: Path, cfg: PrepostProcessConfig) -> Tuple[multi.WindowScope, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], int, Path, pd.DataFrame]:
    out_tmp = _ensure_dir(v7_root / "outputs" / OUTPUT_TAG / "_process_a_input_audit")
    mcfg = _build_multi_cfg(cfg)
    wins = multi._load_accepted_windows(v7_root, out_tmp, mcfg)
    scopes, validity = multi._build_window_scopes(wins, mcfg)
    run_scopes, _audit = multi._filter_scopes_for_run(scopes, mcfg)
    if not run_scopes:
        raise RuntimeError("No W45 scope selected. Check hardcoded windows and window_mode.")
    scope = run_scopes[0]
    if scope.window_id != "W045":
        raise RuntimeError(f"process_a expects W045 only, got {scope.window_id}")

    smoothed = Path(cfg.smoothed_fields_path) if cfg.smoothed_fields_path else v7_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"
    if not smoothed.exists():
        smoothed = v7_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"
    fields, input_audit = clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for spec in clean.OBJECT_SPECS:
        arr = clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        prof, target_lat, weights = clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (prof, target_lat, weights)
    ny = next(iter(profiles.values()))[0].shape[0]
    return scope, profiles, ny, smoothed, input_audit


def _curve_column(kind: str, branch: str) -> str:
    if kind == "state":
        return "S_dist" if branch == "dist" else "S_pattern"
    if kind == "growth":
        return "V_dist" if branch == "dist" else "V_pattern"
    raise ValueError(f"unknown kind={kind}")


def _extract_object_curve(df: pd.DataFrame, obj: str, baseline: str, branch: str, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    col = _curve_column(kind, branch)
    g = df[(df["object"] == obj) & (df["baseline_config"] == baseline)].sort_values("day")
    if g.empty or col not in g.columns:
        return np.array([], dtype=int), np.array([], dtype=float)
    return g["day"].to_numpy(dtype=int), g[col].to_numpy(dtype=float)


def _align_arrays(days_a: np.ndarray, vals_a: np.ndarray, days_b: np.ndarray, vals_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if days_a.size == 0 or days_b.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    amap = dict(zip(days_a.tolist(), vals_a.tolist()))
    bmap = dict(zip(days_b.tolist(), vals_b.tolist()))
    days = np.array(sorted(set(amap).intersection(bmap)), dtype=int)
    return days, np.array([amap[int(d)] for d in days], dtype=float), np.array([bmap[int(d)] for d in days], dtype=float)


def _extract_label_episodes(days: np.ndarray, labels: np.ndarray, target_labels: Iterable[str], min_episode_len: int) -> List[Episode]:
    target = set(target_labels)
    out: List[Episode] = []
    cur_label = None
    start_idx = 0
    for i, lab in enumerate(labels.tolist() + ["__END__"]):
        if i < len(labels) and lab in target:
            if cur_label is None:
                cur_label = lab
                start_idx = i
            elif lab != cur_label:
                end_idx = i - 1
                length = end_idx - start_idx + 1
                if length >= min_episode_len:
                    out.append(Episode(str(cur_label), int(days[start_idx]), int(days[end_idx]), int(length), 0.0))
                cur_label = lab
                start_idx = i
        else:
            if cur_label is not None:
                end_idx = i - 1
                length = end_idx - start_idx + 1
                if length >= min_episode_len:
                    out.append(Episode(str(cur_label), int(days[start_idx]), int(days[end_idx]), int(length), 0.0))
                cur_label = None
    return out


def _label_state_days(delta: np.ndarray, eps_s: float) -> np.ndarray:
    labels = np.full(delta.shape, "nan", dtype=object)
    finite = np.isfinite(delta)
    labels[finite & (delta > eps_s)] = "A"
    labels[finite & (delta < -eps_s)] = "B"
    labels[finite & (np.abs(delta) <= eps_s)] = "near"
    return labels


def _classify_state_curve_once(days: np.ndarray, delta_s: np.ndarray, eps_s: float, cfg: PrepostProcessConfig) -> dict:
    valid = np.isfinite(delta_s)
    if days.size == 0 or np.sum(valid) < cfg.min_episode_len:
        return {"state_structure": "state_unresolved"}
    d = days[valid]
    x = delta_s[valid]
    labels = _label_state_days(x, eps_s)
    total = max(len(labels), 1)
    a_duration = int(np.sum(labels == "A"))
    b_duration = int(np.sum(labels == "B"))
    near_duration = int(np.sum(labels == "near"))
    a_ratio = a_duration / total
    b_ratio = b_duration / total
    near_ratio = near_duration / total
    episodes = _extract_label_episodes(d, labels, ["A", "B", "near"], cfg.min_episode_len)
    dom_eps = [e for e in episodes if e.label in ("A", "B")]
    seq = [e.label for e in dom_eps]
    switch_count = int(sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]))
    first = seq[0] if seq else "near"
    last = seq[-1] if seq else "near"

    def _episode_mean_abs(ep: Episode) -> float:
        m = (d >= ep.start) & (d <= ep.end)
        vals = np.abs(x[m])
        return float(np.nanmean(vals)) if vals.size else float("nan")

    initial_abs_gap = _episode_mean_abs(dom_eps[0]) if dom_eps else float(np.nanmean(np.abs(x[: cfg.min_episode_len])))
    final_abs_gap = _episode_mean_abs(dom_eps[-1]) if dom_eps else float(np.nanmean(np.abs(x[-cfg.min_episode_len :])))
    signed_area = float(np.nansum(x))
    near_eps_after_first = False
    if dom_eps:
        first_end = dom_eps[0].end
        near_eps_after_first = any(e.label == "near" and e.start > first_end for e in episodes)
    state_structure = "state_unresolved"
    if near_ratio >= cfg.parallel_duration_ratio and a_ratio < 0.20 and b_ratio < 0.20:
        state_structure = "parallel_state_progress"
    elif a_ratio >= cfg.persistent_duration_ratio and b_ratio <= cfg.opposite_duration_max_ratio and switch_count <= 1:
        state_structure = "persistent_A_more_postlike"
    elif b_ratio >= cfg.persistent_duration_ratio and a_ratio <= cfg.opposite_duration_max_ratio and switch_count <= 1:
        state_structure = "persistent_B_more_postlike"
    elif switch_count >= 2:
        state_structure = "multi_crossing_unstable"
    elif first == "A" and last == "B":
        state_structure = "state_reversal_A_to_B"
    elif first == "B" and last == "A":
        state_structure = "state_reversal_B_to_A"
    elif first == "A" and (near_eps_after_first or last in ("near", "B") or (np.isfinite(final_abs_gap) and final_abs_gap < initial_abs_gap)):
        state_structure = "A_front_B_catchup"
    elif first == "B" and (near_eps_after_first or last in ("near", "A") or (np.isfinite(final_abs_gap) and final_abs_gap < initial_abs_gap)):
        state_structure = "B_front_A_catchup"

    return {
        "state_structure": state_structure,
        "A_total_duration": a_duration,
        "B_total_duration": b_duration,
        "near_total_duration": near_duration,
        "dominant_switch_count": switch_count,
        "first_dominant_label": first,
        "last_dominant_label": last,
        "initial_abs_gap": initial_abs_gap,
        "final_abs_gap": final_abs_gap,
        "signed_area": signed_area,
    }


def _label_growth_days(g: np.ndarray, eps_g: float) -> np.ndarray:
    labels = np.full(g.shape, "nan", dtype=object)
    finite = np.isfinite(g)
    labels[finite & (g > eps_g)] = "pos"
    labels[finite & (g < -eps_g)] = "neg"
    labels[finite & (np.abs(g) <= eps_g)] = "neutral"
    return labels


def _episode_masses(days: np.ndarray, values: np.ndarray, episodes: List[Episode], positive: bool = True) -> List[Episode]:
    out: List[Episode] = []
    for ep in episodes:
        m = (days >= ep.start) & (days <= ep.end)
        vals = values[m]
        mass = float(np.nansum(np.maximum(vals, 0))) if positive else float(np.nansum(np.maximum(-vals, 0)))
        out.append(Episode(ep.label, ep.start, ep.end, ep.length, mass))
    return out


def _classify_object_growth_once(days: np.ndarray, g: np.ndarray, eps_g: float, cfg: PrepostProcessConfig) -> dict:
    valid = np.isfinite(g)
    if days.size == 0 or np.sum(valid) < cfg.min_episode_len:
        return {"growth_structure": "growth_unresolved", "rollback_class": "no_meaningful_rollback"}
    d = days[valid]
    x = g[valid]
    labels = _label_growth_days(x, eps_g)
    pos_eps = _episode_masses(d, x, _extract_label_episodes(d, labels, ["pos"], cfg.min_episode_len), True)
    neg_eps = _episode_masses(d, x, _extract_label_episodes(d, labels, ["neg"], cfg.min_episode_len), False)
    pos_eps = sorted(pos_eps, key=lambda e: e.mass, reverse=True)
    neg_eps = sorted(neg_eps, key=lambda e: e.mass, reverse=True)
    total_pos = float(np.nansum(np.maximum(x, 0)))
    total_neg = float(np.nansum(np.maximum(-x, 0)))
    gross = total_pos + total_neg
    neg_ratio = total_neg / gross if gross > EPS else 0.0
    if total_pos <= EPS or not pos_eps:
        growth_structure = "weak_or_no_positive_growth"
    else:
        main_ratio = pos_eps[0].mass / max(total_pos, EPS)
        secondary_ratio = (pos_eps[1].mass / max(total_pos, EPS)) if len(pos_eps) >= 2 else 0.0
        if len(pos_eps) >= 2 and secondary_ratio >= cfg.multi_stage_min_secondary_ratio:
            growth_structure = "multi_stage_growth"
        elif main_ratio >= cfg.main_episode_mass_ratio:
            growth_structure = "single_growth_episode"
        else:
            growth_structure = "growth_unresolved"
    rollback_class = "no_meaningful_rollback"
    if total_neg > EPS:
        if neg_eps and neg_ratio >= cfg.rollback_min_negative_ratio:
            rollback_class = "rollback_or_regression"
        else:
            rollback_class = "minor_negative_fluctuation"
    main = pos_eps[0] if pos_eps else Episode("pos", -1, -1, 0, 0.0)
    secondary = pos_eps[1] if len(pos_eps) >= 2 else Episode("pos", -1, -1, 0, 0.0)
    return {
        "growth_structure": growth_structure,
        "rollback_class": rollback_class,
        "positive_episode_count": int(len(pos_eps)),
        "main_episode_start": np.nan if main.start < 0 else int(main.start),
        "main_episode_end": np.nan if main.end < 0 else int(main.end),
        "main_episode_mass": float(main.mass),
        "secondary_episode_mass": float(secondary.mass),
        "negative_ratio": float(neg_ratio),
        "negative_episode_count": int(len(neg_eps)),
    }


def _classify_pairwise_growth_once(days: np.ndarray, g_a: np.ndarray, g_b: np.ndarray, delta_g: np.ndarray, eps_ga: float, eps_gb: float, eps_pair: float, cfg: PrepostProcessConfig) -> dict:
    valid = np.isfinite(g_a) & np.isfinite(g_b) & np.isfinite(delta_g)
    if days.size == 0 or np.sum(valid) < cfg.min_episode_len:
        return {"growth_structure": "growth_unresolved"}
    d = days[valid]
    ga = g_a[valid]
    gb = g_b[valid]
    dg = delta_g[valid]
    labels = np.full(d.shape, "neutral", dtype=object)
    posa = ga > eps_ga
    posb = gb > eps_gb
    sync = posa & posb & (np.abs(dg) <= eps_pair)
    labels[sync] = "sync"
    labels[posa & ~sync & (dg > eps_pair)] = "A"
    labels[posb & ~sync & (dg < -eps_pair)] = "B"
    labels[posa & ~posb] = "A"
    labels[posb & ~posa] = "B"
    episodes = _extract_label_episodes(d, labels, ["A", "B", "sync"], cfg.min_episode_len)
    a_eps = [e for e in episodes if e.label == "A"]
    b_eps = [e for e in episodes if e.label == "B"]
    s_eps = [e for e in episodes if e.label == "sync"]
    seq = [e.label for e in episodes if e.label in ("A", "B")]
    switch_count = int(sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]))
    first = episodes[0].label if episodes else "none"
    later_labels = [e.label for e in episodes[1:]] if len(episodes) > 1 else []
    pos_union = posa | posb
    sync_ratio = float(np.sum(sync)) / max(int(np.sum(pos_union)), 1)
    structure = "growth_unresolved"
    if sync_ratio >= cfg.sync_growth_mass_ratio and len(s_eps) > 0:
        structure = "synchronized_growth"
    elif first == "A" and "B" in later_labels:
        structure = "A_growth_front_B_catchup"
    elif first == "B" and "A" in later_labels:
        structure = "B_growth_front_A_catchup"
    elif first == "A" and "sync" in later_labels:
        structure = "A_growth_front_then_synchronized"
    elif first == "B" and "sync" in later_labels:
        structure = "B_growth_front_then_synchronized"
    elif switch_count >= 2 and len(a_eps) > 0 and len(b_eps) > 0:
        structure = "alternating_growth"
    return {
        "growth_structure": structure,
        "first_growth_episode_type": first,
        "last_growth_episode_type": episodes[-1].label if episodes else "none",
        "A_growth_episode_count": int(len(a_eps)),
        "B_growth_episode_count": int(len(b_eps)),
        "sync_growth_episode_count": int(len(s_eps)),
        "growth_switch_count": switch_count,
        "sync_growth_ratio": sync_ratio,
    }


def _summarize_structure_support(records: pd.DataFrame, group_cols: List[str], structure_col: str, all_structures: List[str], prefix: str, cfg: PrepostProcessConfig) -> pd.DataFrame:
    rows: List[dict] = []
    if records is None or records.empty:
        return pd.DataFrame()
    for keys, g in records.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        total = max(len(g), 1)
        probs = {}
        for s in all_structures:
            probs[f"P_{s}"] = float(np.sum(g[structure_col].astype(str) == s) / total)
        ordered = sorted([(s, probs[f"P_{s}"]) for s in all_structures], key=lambda x: x[1], reverse=True)
        primary, ps = ordered[0]
        comp, cs = ordered[1] if len(ordered) > 1 else ("", np.nan)
        rec.update(probs)
        rec[f"{prefix}_primary_structure"] = primary
        rec[f"{prefix}_primary_support"] = ps
        rec[f"{prefix}_support_class"] = _support_class(ps, cfg)
        rec[f"{prefix}_competing_structure"] = comp
        rec[f"{prefix}_competing_support"] = cs
        rows.append(rec)
    return pd.DataFrame(rows)


def _episode_quantile_summary(records: pd.DataFrame, group_cols: List[str], fields: List[str]) -> pd.DataFrame:
    rows = []
    if records is None or records.empty:
        return pd.DataFrame()
    for keys, g in records.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        for f in fields:
            vals = g[f].to_numpy(dtype=float) if f in g.columns else np.array([], dtype=float)
            rec[f"{f}_median"] = _median(vals)
            rec[f"{f}_q025"] = _quantile(vals, 0.025)
            rec[f"{f}_q975"] = _quantile(vals, 0.975)
        rows.append(rec)
    return pd.DataFrame(rows)


def _compute_curve_validity(observed_state: pd.DataFrame, boot_state_samples: Dict[Tuple[str, str, str], np.ndarray], boot_growth_samples: Dict[Tuple[str, str, str], np.ndarray], days_by_key: Dict[Tuple[str, str, str], np.ndarray], cfg: PrepostProcessConfig) -> pd.DataFrame:
    rows = []
    for key in sorted(days_by_key.keys()):
        obj, base, branch = key
        days = days_by_key[key]
        s_samples = boot_state_samples.get(key)
        g_samples = boot_growth_samples.get(key)
        obs_g = observed_state[(observed_state["object"] == obj) & (observed_state["baseline_config"] == base)].sort_values("day")
        state_valid = "valid"
        warnings: List[str] = []
        finite_fraction = float(np.mean(np.isfinite(s_samples))) if s_samples is not None and s_samples.size else np.nan
        overshoot_fraction = np.nan
        severe_fraction = np.nan
        if s_samples is not None and s_samples.size:
            finite = np.isfinite(s_samples)
            overshoot = finite & ((s_samples < 0) | (s_samples > 1))
            severe = finite & ((s_samples < -cfg.severe_overshoot_margin) | (s_samples > 1 + cfg.severe_overshoot_margin))
            overshoot_fraction = float(np.mean(overshoot))
            severe_fraction = float(np.mean(severe))
            if severe_fraction > 0:
                warnings.append("severe_overshoot_warning")
        pattern_dyn_vals = []
        if branch == "pattern" and not obs_g.empty and "pattern_dynamic_range" in obs_g.columns:
            pattern_dyn_vals = obs_g["pattern_dynamic_range"].dropna().astype(float).unique().tolist()
            if pattern_dyn_vals and float(np.nanmedian(pattern_dyn_vals)) < cfg.pattern_dynamic_low_abs:
                warnings.append("low_dynamic_pattern_warning")
        eps_g = np.nan
        positive_detectable = np.nan
        if g_samples is not None and g_samples.size:
            day_sd = np.nanstd(g_samples, axis=0)
            eps_g = float(np.nanmedian(day_sd) * cfg.growth_noise_sd_factor)
            positive_detectable = float(np.mean(np.nanmax(g_samples, axis=1) > eps_g)) if np.isfinite(eps_g) else np.nan
            if positive_detectable < cfg.support_exploratory:
                warnings.append("weak_positive_growth_detectability")
        if finite_fraction < 0.5:
            state_valid = "invalid_or_sparse"
        rows.append({
            "object": obj,
            "baseline": base,
            "branch": branch,
            "state_validity": state_valid,
            "finite_state_fraction": finite_fraction,
            "pattern_dynamic_range_median": _median(pattern_dyn_vals),
            "pattern_dynamic_range_q025": _quantile(pattern_dyn_vals, 0.025),
            "overshoot_fraction_median": overshoot_fraction,
            "severe_overshoot_fraction_median": severe_fraction,
            "growth_validity": "valid" if np.isfinite(eps_g) else "unavailable",
            "eps_G": eps_g,
            "positive_growth_detectable": positive_detectable,
            "curve_warning": ";".join(sorted(set(warnings))),
        })
    return pd.DataFrame(rows)


def _legacy_output_status(v7_root: Path) -> pd.DataFrame:
    legacy_files = [
        "pairwise_state_progress_difference_W045.csv",
        "pairwise_state_catchup_reversal_W045.csv",
        "object_growth_sign_structure_W045.csv",
        "object_growth_pulse_structure_W045.csv",
        "pairwise_growth_process_difference_W045.csv",
        "pairwise_prepost_curve_interpretation_W045.csv",
        "pairwise_order_interpretation_summary_W045.csv",
        "W45_profile_order_test_summary_hotfix06.md",
    ]
    old_dir = v7_root / "outputs" / "accepted_windows_multi_object_prepost_v7_z_multiwin_a_hotfix06_w45_profile_order" / "per_window" / "W045"
    rows = []
    for fn in legacy_files:
        rows.append({
            "legacy_file": fn,
            "exists": (old_dir / fn).exists(),
            "legacy_role": "deprecated_or_legacy_audit",
            "mainline_allowed": False,
            "reason": "process_a replaces winner-style / early-core-late / larger-or-earlier middle layer with curve-structure diagnostics",
        })
    return pd.DataFrame(rows)


def _baseline_consensus(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [c for c in ["window_id", "object_A", "object_B", "branch"] if c in summary.columns]
    if not group_cols:
        return pd.DataFrame()
    struct_col = f"{prefix}_primary_structure"
    class_col = f"{prefix}_support_class"
    support_col = f"{prefix}_primary_support"
    for keys, g in summary.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        by_base = {r["baseline"]: r for _, r in g.iterrows() if "baseline" in r}
        c0 = by_base.get("C0_full_stage")
        c1 = by_base.get("C1_buffered_stage")
        c2 = by_base.get("C2_immediate_pre")
        main_structure = ""
        main_status = "baseline_conflict"
        if c0 is not None and c1 is not None:
            if c0.get(struct_col) == c1.get(struct_col) and c0.get(class_col) == "supported" and c1.get(class_col) == "supported":
                main_structure = c0.get(struct_col)
                main_status = "baseline_main_supported"
            elif c0.get(struct_col) == c1.get(struct_col) and c0.get(class_col) in ["supported", "tendency"] and c1.get(class_col) in ["supported", "tendency"]:
                main_structure = c0.get(struct_col)
                main_status = "baseline_main_tendency"
            else:
                main_structure = f"C0={c0.get(struct_col,'')};C1={c1.get(struct_col,'')}"
                main_status = "baseline_conflict"
        c2_flag = "C2_missing"
        if c2 is not None and main_structure:
            c2_flag = "C2_consistent" if c2.get(struct_col) == main_structure else "C2_sensitive"
        rec.update({
            f"{prefix}_baseline_main_structure": main_structure,
            f"{prefix}_baseline_status": main_status,
            f"{prefix}_C2_structure": c2.get(struct_col, "") if c2 is not None else "",
            f"{prefix}_C2_sensitivity_flag": c2_flag,
            f"{prefix}_C0_support": c0.get(support_col, np.nan) if c0 is not None else np.nan,
            f"{prefix}_C1_support": c1.get(support_col, np.nan) if c1 is not None else np.nan,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_branch_baseline_consensus(state_summary: pd.DataFrame, growth_summary: pd.DataFrame, validity: pd.DataFrame) -> pd.DataFrame:
    state_base = _baseline_consensus(state_summary, "state")
    growth_base = _baseline_consensus(growth_summary, "growth")
    if state_base.empty and growth_base.empty:
        return pd.DataFrame()
    key_cols = ["window_id", "object_A", "object_B", "branch"]
    merged = pd.merge(state_base, growth_base, on=key_cols, how="outer") if not state_base.empty and not growth_base.empty else (state_base if not state_base.empty else growth_base)
    rows = []
    for (wid, a, b), g in merged.groupby(["window_id", "object_A", "object_B"], dropna=False):
        rec = {"window_id": wid, "object_A": a, "object_B": b}
        for branch in BRANCHES:
            gg = g[g["branch"] == branch]
            if gg.empty:
                rec[f"state_{branch}_main"] = ""
                rec[f"growth_{branch}_main"] = ""
                continue
            r = gg.iloc[0]
            rec[f"state_{branch}_main"] = r.get("state_baseline_main_structure", "")
            rec[f"state_{branch}_baseline_status"] = r.get("state_baseline_status", "")
            rec[f"growth_{branch}_main"] = r.get("growth_baseline_main_structure", "")
            rec[f"growth_{branch}_baseline_status"] = r.get("growth_baseline_status", "")
            rec[f"state_{branch}_C2_flag"] = r.get("state_C2_sensitivity_flag", "")
            rec[f"growth_{branch}_C2_flag"] = r.get("growth_C2_sensitivity_flag", "")
        state_branch_status = "branch_unresolved"
        if rec.get("state_dist_main") and rec.get("state_pattern_main"):
            if rec["state_dist_main"] == rec["state_pattern_main"]:
                state_branch_status = "branch_consistent"
            else:
                state_branch_status = "branch_split"
        elif rec.get("state_dist_main"):
            state_branch_status = "dist_primary_pattern_unresolved_or_warning"
        elif rec.get("state_pattern_main"):
            state_branch_status = "pattern_specific_signal"
        growth_branch_status = "branch_unresolved"
        if rec.get("growth_dist_main") and rec.get("growth_pattern_main"):
            if rec["growth_dist_main"] == rec["growth_pattern_main"]:
                growth_branch_status = "branch_consistent"
            else:
                growth_branch_status = "branch_split"
        elif rec.get("growth_dist_main"):
            growth_branch_status = "dist_primary_pattern_unresolved_or_warning"
        elif rec.get("growth_pattern_main"):
            growth_branch_status = "pattern_specific_signal"
        rec["state_branch_status"] = state_branch_status
        rec["growth_branch_status"] = growth_branch_status
        rec["interpretation_readiness"] = "ready_for_supported_interpretation" if (
            rec.get("state_dist_baseline_status") == "baseline_main_supported" or rec.get("growth_dist_baseline_status") == "baseline_main_supported"
        ) else "audit_only_or_unresolved"
        rows.append(rec)
    return pd.DataFrame(rows)


def _interpret_pair(a: str, b: str, row: pd.Series) -> Tuple[str, str, str, str]:
    # Conservative, named interpretations for the known W45 targets; otherwise structural fallback.
    state_dist = str(row.get("state_dist_main", ""))
    growth_dist = str(row.get("growth_dist_main", ""))
    state_pat = str(row.get("state_pattern_main", ""))
    growth_pat = str(row.get("growth_pattern_main", ""))
    risk = []
    if "branch_split" in str(row.get("state_branch_status", "")) or "branch_split" in str(row.get("growth_branch_status", "")):
        risk.append("branch_split")
    if "C2_sensitive" in ";".join(str(row.get(c, "")) for c in row.index if "C2_flag" in c):
        risk.append("C2_sensitive")

    interp = "process_structure_supported_or_audit_only"
    allowed = "State/growth process structure is reported by supported curve-structure diagnostics; inspect state/growth bases and risk flags."
    forbidden = "Do not translate this row into a hard peak lead or causal lead."
    if (a, b) == ("H", "Jw") and ("A_front_B_catchup" in state_dist or "A_growth_front_B_catchup" in growth_dist or "A_growth_front_then_synchronized" in growth_dist):
        interp = "H_distance_state_growth_front_with_Jw_catchup"
        allowed = "H shows distance-state/growth front structure relative to Jw, with Jw catch-up or later/synchronized growth where supported."
        forbidden = "H hard leads Jw by peak timing."
    elif (a, b) == ("P", "V") and ("parallel_state_progress" in state_dist or "synchronized_growth" in growth_dist):
        interp = "P_V_near_parallel_or_synchronized_transition"
        allowed = "P/V show near-parallel state progress or synchronized growth where supported; weak tendencies must remain tendencies."
        forbidden = "P leads V."
    elif "pattern" in state_pat or "pattern" in growth_pat:
        interp = "branch_specific_process_structure"
    return interp, allowed, forbidden, ";".join(risk)


def _build_process_interpretation(consensus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if consensus is None or consensus.empty:
        return pd.DataFrame()
    for _, r in consensus.iterrows():
        a, b = str(r["object_A"]), str(r["object_B"])
        interp, allowed, forbidden, risk = _interpret_pair(a, b, r)
        rows.append({
            "window_id": r.get("window_id", "W045"),
            "object_A": a,
            "object_B": b,
            "process_interpretation": interp,
            "state_basis": f"dist={r.get('state_dist_main','')}; pattern={r.get('state_pattern_main','')}",
            "growth_basis": f"dist={r.get('growth_dist_main','')}; pattern={r.get('growth_pattern_main','')}",
            "branch_basis": f"state={r.get('state_branch_status','')}; growth={r.get('growth_branch_status','')}",
            "baseline_basis": f"state_dist={r.get('state_dist_baseline_status','')}; growth_dist={r.get('growth_dist_baseline_status','')}",
            "support_level": "supported_only_if_basis_supported",
            "allowed_statement": allowed,
            "forbidden_statement": forbidden,
            "risk_notes": risk,
        })
    return pd.DataFrame(rows)


def _prepare_curve_samples(observed_state: pd.DataFrame, observed_growth: pd.DataFrame, boot_state_list: List[pd.DataFrame], boot_growth_list: List[pd.DataFrame], cfg: PrepostProcessConfig):
    baselines = sorted(observed_state["baseline_config"].dropna().unique())
    days_by_key: Dict[Tuple[str, str, str], np.ndarray] = {}
    obs_state_curves: Dict[Tuple[str, str, str], np.ndarray] = {}
    obs_growth_curves: Dict[Tuple[str, str, str], np.ndarray] = {}
    boot_state_samples: Dict[Tuple[str, str, str], np.ndarray] = {}
    boot_growth_samples: Dict[Tuple[str, str, str], np.ndarray] = {}
    for obj in OBJECTS:
        for base in baselines:
            for branch in BRANCHES:
                ds, vs = _extract_object_curve(observed_state, obj, base, branch, "state")
                dg, vg = _extract_object_curve(observed_state, obj, base, branch, "growth")
                days, st, gr = _align_arrays(ds, vs, dg, vg)
                if days.size == 0:
                    continue
                key = (obj, base, branch)
                days_by_key[key] = days
                obs_state_curves[key] = st
                obs_growth_curves[key] = gr
                st_rows = []
                gr_rows = []
                for bs, bg in zip(boot_state_list, boot_growth_list):
                    bds, bvs = _extract_object_curve(bs, obj, base, branch, "state")
                    bdg, bvg = _extract_object_curve(bs, obj, base, branch, "growth")
                    bdays, bst, bgr = _align_arrays(bds, bvs, bdg, bvg)
                    bmap_s = dict(zip(bdays.tolist(), bst.tolist()))
                    bmap_g = dict(zip(bdays.tolist(), bgr.tolist()))
                    st_rows.append([bmap_s.get(int(d), np.nan) for d in days])
                    gr_rows.append([bmap_g.get(int(d), np.nan) for d in days])
                boot_state_samples[key] = np.asarray(st_rows, dtype=float)
                boot_growth_samples[key] = np.asarray(gr_rows, dtype=float)
    return days_by_key, obs_state_curves, obs_growth_curves, boot_state_samples, boot_growth_samples


def _bootstrap_state_structures(days_by_key, boot_state_samples, cfg: PrepostProcessConfig) -> pd.DataFrame:
    records = []
    for a, b in OBJECT_ORDER_PAIRS:
        for base in sorted({k[1] for k in days_by_key.keys()}):
            for branch in BRANCHES:
                ka, kb = (a, base, branch), (b, base, branch)
                if ka not in boot_state_samples or kb not in boot_state_samples:
                    continue
                days = days_by_key[ka]
                sa = boot_state_samples[ka]
                sb = boot_state_samples[kb]
                if sa.shape != sb.shape:
                    continue
                delta = sa - sb
                eps_s = float(np.nanmedian(np.nanstd(delta, axis=0)) * cfg.near_state_sd_factor)
                eps_s = max(eps_s, EPS)
                for ib in range(delta.shape[0]):
                    rec = _classify_state_curve_once(days, delta[ib], eps_s, cfg)
                    rec.update({"window_id": "W045", "object_A": a, "object_B": b, "baseline": base, "branch": branch, "bootstrap_id": ib, "eps_S": eps_s})
                    records.append(rec)
    return pd.DataFrame(records)


def _bootstrap_object_growth(days_by_key, boot_growth_samples, cfg: PrepostProcessConfig) -> pd.DataFrame:
    records = []
    for key, gs in boot_growth_samples.items():
        obj, base, branch = key
        days = days_by_key[key]
        eps_g = float(np.nanmedian(np.nanstd(gs, axis=0)) * cfg.growth_noise_sd_factor)
        eps_g = max(eps_g, EPS)
        for ib in range(gs.shape[0]):
            rec = _classify_object_growth_once(days, gs[ib], eps_g, cfg)
            rec.update({"window_id": "W045", "object": obj, "baseline": base, "branch": branch, "bootstrap_id": ib, "eps_G": eps_g})
            records.append(rec)
    return pd.DataFrame(records)


def _bootstrap_pair_growth(days_by_key, boot_growth_samples, cfg: PrepostProcessConfig) -> pd.DataFrame:
    records = []
    for a, b in OBJECT_ORDER_PAIRS:
        for base in sorted({k[1] for k in days_by_key.keys()}):
            for branch in BRANCHES:
                ka, kb = (a, base, branch), (b, base, branch)
                if ka not in boot_growth_samples or kb not in boot_growth_samples:
                    continue
                days = days_by_key[ka]
                ga = boot_growth_samples[ka]
                gb = boot_growth_samples[kb]
                if ga.shape != gb.shape:
                    continue
                dg = ga - gb
                eps_ga = max(float(np.nanmedian(np.nanstd(ga, axis=0)) * cfg.growth_noise_sd_factor), EPS)
                eps_gb = max(float(np.nanmedian(np.nanstd(gb, axis=0)) * cfg.growth_noise_sd_factor), EPS)
                eps_pair = max(float(np.nanmedian(np.nanstd(dg, axis=0)) * cfg.growth_noise_sd_factor), EPS)
                for ib in range(dg.shape[0]):
                    rec = _classify_pairwise_growth_once(days, ga[ib], gb[ib], dg[ib], eps_ga, eps_gb, eps_pair, cfg)
                    rec.update({"window_id": "W045", "object_A": a, "object_B": b, "baseline": base, "branch": branch, "bootstrap_id": ib, "eps_G_A": eps_ga, "eps_G_B": eps_gb, "eps_G_pair": eps_pair})
                    records.append(rec)
    return pd.DataFrame(records)


def _write_summary_md(path: Path, cfg: PrepostProcessConfig, consensus: pd.DataFrame, interp: pd.DataFrame) -> None:
    lines = [
        "# V7-z prepost process-a summary",
        "",
        "## Method boundary",
        "- This run is W45 profile-only.",
        "- Detector, W45 scope, C0/C1/C2 baselines, and S/G definitions are not modified.",
        "- Main diagnostics use paired-year bootstrap curve-structure classifications of Delta S_AB(t) and Delta G_AB(t).",
        "- early/core/late winner outputs are deprecated for main interpretation.",
        "- Supported threshold is 0.95; 0.90-0.95 is tendency only; 0.80-0.90 is audit-only exploratory signal.",
        "",
        f"bootstrap_n: {cfg.bootstrap_n}",
        "",
        "## Interpretation readiness counts",
    ]
    if consensus is None or consensus.empty:
        lines.append("- No consensus rows produced.")
    else:
        counts = consensus.get("interpretation_readiness", pd.Series(dtype=str)).value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Process interpretation rows")
    if interp is None or interp.empty:
        lines.append("- No interpretation rows produced.")
    else:
        for _, r in interp.iterrows():
            lines.append(f"- {r['object_A']}-{r['object_B']}: {r['process_interpretation']} | risk={r.get('risk_notes','')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_prepost_process_curve_v7_z_process_a(v7_root: Path | str) -> None:
    v7_root = Path(v7_root)
    cfg = PrepostProcessConfig.from_env()
    out_root = _ensure_dir(v7_root / "outputs" / OUTPUT_TAG)
    out_win = _ensure_dir(out_root / "per_window" / "W045")
    out_cross = _ensure_dir(out_root / "cross_window")
    log_dir = _ensure_dir(v7_root / "logs" / OUTPUT_TAG)
    t0 = time.time()
    _write_json({"version": VERSION, "config": asdict(cfg), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, out_root / "run_meta_start.json")

    _log("[process-a 1/7] Load W45 profiles")
    scope, profiles, ny, smoothed, input_audit = _load_w45_profiles(v7_root, cfg)
    _safe_to_csv(input_audit, out_cross / "input_key_audit_v7_z_prepost_process_a.csv")
    _write_json({"smoothed_fields": str(smoothed), "n_years": ny, "scope": asdict(scope)}, out_cross / "input_scope_meta.json")

    mcfg = _build_multi_cfg(cfg)
    _log("[process-a 2/7] Compute observed S/G curves")
    idx_all = np.arange(ny, dtype=int)
    observed_state, observed_growth = multi._state_for_bootstrap_indices(profiles, scope, mcfg, idx_all)
    _safe_to_csv(observed_state, out_win / "observed_state_curves_W045.csv")
    _safe_to_csv(observed_growth, out_win / "observed_growth_curves_W045.csv")

    _log("[process-a 3/7] Paired-year bootstrap S/G curves")
    boot_indices = multi._make_bootstrap_indices(ny, scope, mcfg)
    if len(boot_indices) > cfg.bootstrap_n:
        boot_indices = boot_indices[: cfg.bootstrap_n]
    boot_state_list: List[pd.DataFrame] = []
    boot_growth_list: List[pd.DataFrame] = []
    for ib, idx in enumerate(boot_indices, start=1):
        st, gr = multi._state_for_bootstrap_indices(profiles, scope, mcfg, idx)
        boot_state_list.append(st)
        boot_growth_list.append(gr)
        if cfg.log_every_bootstrap and (ib == 1 or ib == len(boot_indices) or ib % cfg.log_every_bootstrap == 0):
            _log(f"  bootstrap {ib}/{len(boot_indices)}")

    _log("[process-a 4/7] Prepare curve sample arrays")
    days_by_key, obs_state_curves, obs_growth_curves, boot_state_samples, boot_growth_samples = _prepare_curve_samples(observed_state, observed_growth, boot_state_list, boot_growth_list, cfg)

    _log("[process-a 5/7] Classify state/growth curve structures")
    validity = _compute_curve_validity(observed_state, boot_state_samples, boot_growth_samples, days_by_key, cfg)
    validity.insert(0, "window_id", "W045")
    _safe_to_csv(validity, out_win / "object_curve_validity_W045.csv")

    object_growth_records = _bootstrap_object_growth(days_by_key, boot_growth_samples, cfg)
    object_growth_struct = _summarize_structure_support(object_growth_records, ["window_id", "object", "baseline", "branch"], "growth_structure", OBJECT_GROWTH_STRUCTURES, "growth", cfg)
    rollback_struct = _summarize_structure_support(object_growth_records, ["window_id", "object", "baseline", "branch"], "rollback_class", ROLLBACK_STRUCTURES, "rollback", cfg)
    qsum = _episode_quantile_summary(object_growth_records, ["window_id", "object", "baseline", "branch"], ["main_episode_start", "main_episode_end", "negative_ratio", "main_episode_mass"])
    object_growth_out = object_growth_struct.merge(rollback_struct, on=["window_id", "object", "baseline", "branch"], how="outer").merge(qsum, on=["window_id", "object", "baseline", "branch"], how="outer") if not object_growth_struct.empty else pd.DataFrame()
    _safe_to_csv(object_growth_out, out_win / "object_growth_episode_bootstrap_W045.csv")

    state_records = _bootstrap_state_structures(days_by_key, boot_state_samples, cfg)
    state_summary = _summarize_structure_support(state_records, ["window_id", "object_A", "object_B", "baseline", "branch"], "state_structure", STATE_STRUCTURES, "state", cfg)
    _safe_to_csv(state_summary, out_win / "pairwise_state_curve_bootstrap_W045.csv")

    pair_growth_records = _bootstrap_pair_growth(days_by_key, boot_growth_samples, cfg)
    pair_growth_summary = _summarize_structure_support(pair_growth_records, ["window_id", "object_A", "object_B", "baseline", "branch"], "growth_structure", PAIR_GROWTH_STRUCTURES, "growth", cfg)
    _safe_to_csv(pair_growth_summary, out_win / "pairwise_growth_curve_bootstrap_W045.csv")

    _log("[process-a 6/7] Branch/baseline consensus and process interpretation")
    consensus = _build_branch_baseline_consensus(state_summary, pair_growth_summary, validity)
    _safe_to_csv(consensus, out_win / "pairwise_branch_baseline_consensus_W045.csv")
    interp = _build_process_interpretation(consensus)
    _safe_to_csv(interp, out_win / "pairwise_process_interpretation_W045.csv")
    legacy = _legacy_output_status(v7_root)
    _safe_to_csv(legacy, out_win / "legacy_output_status_W045.csv")
    _write_summary_md(out_win / "prepost_process_curve_summary_W045.md", cfg, consensus, interp)

    _log("[process-a 7/7] Write run metadata")
    meta = {
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "elapsed_seconds": time.time() - t0,
        "bootstrap_n": len(boot_indices),
        "n_years": ny,
        "window_id": scope.window_id,
        "output_root": str(out_root),
        "method_boundary": [
            "W45 profile-only",
            "detector unchanged",
            "S/G definitions unchanged",
            "early/core/late legacy outputs not used for main interpretation",
            "Delta S and Delta G curve-structure bootstrap diagnostics are main process layer",
        ],
        "support_rules": {
            "supported": cfg.support_supported,
            "tendency": cfg.support_tendency,
            "exploratory_signal": cfg.support_exploratory,
        },
        "config": asdict(cfg),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(meta, out_root / "run_meta.json")
    _write_json({"n_state_rows": int(len(state_summary)), "n_growth_rows": int(len(pair_growth_summary)), "n_consensus_rows": int(len(consensus)), "n_interpretation_rows": int(len(interp))}, out_root / "summary.json")
    (log_dir / "last_run.txt").write_text(f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')} output={out_root}\n", encoding="utf-8")
    _log(f"[process-a] done: {out_root}")


if __name__ == "__main__":  # pragma: no cover
    run_prepost_process_curve_v7_z_process_a(Path(__file__).resolve().parents[3])
