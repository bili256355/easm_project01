"""
V9.3_a direct-year profile-evolution supervised SVD / PLS1.

Purpose
-------
V9.3_a is a supervised, direct-year control line for the V9/V9.1/V9.2 peak-order
work. It uses the real years 1979-2023 as samples, five-object profile evolution
anomalies as X, and yearly peak_B - peak_A as Y.  It does not use 2D fields and
it does not use bootstrap-resampled year combinations as samples.

Explicit boundaries
-------------------
- X = real-year five-object profile-evolution anomalies, not 2D fields.
- Y = yearly peak_B - peak_A; yearly peak noise is a core risk and is audited.
- The SVD/PLS direction is target-guided; it is not an unsupervised natural mode.
- Result use must go through the usability registry, not raw score-Y correlation.
- Loading summaries such as dominant_object are statistical summaries, not causes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

VERSION = "v9_3_a"
OUTPUT_TAG = "direct_year_profile_supervised_svd_v9_3_a"
DEFAULT_WINDOWS = ("W045", "W081", "W113", "W160")
OBJECT_ORDER = ("P", "V", "H", "Je", "Jw")
ALL_PAIRS = tuple((a, b) for a, b in itertools.combinations(OBJECT_ORDER, 2))
PRIORITY_TARGETS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "W045": (("Je", "Jw"), ("P", "Jw"), ("V", "Jw")),
    "W081": (("P", "V"), ("V", "Jw"), ("H", "Jw")),
    "W113": (("V", "Je"), ("H", "Je"), ("P", "V"), ("Jw", "H"), ("Jw", "V")),
    "W160": (("V", "Je"), ("H", "Jw"), ("P", "V"), ("Jw", "V")),
}


@dataclass
class V93Settings:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    target_windows: Tuple[str, ...] = DEFAULT_WINDOWS
    years_start: int = 1979
    years_end: int = 2023

    # Method boundaries
    use_2d_field_x: bool = False
    use_bootstrap_resampling: bool = False
    use_yearly_peak_y: bool = True
    method_role: str = "direct-year profile-evolution supervised SVD / PLS1"

    # Target selection
    target_mode: str = "priority_and_full_pairs"  # priority_only, full_pairs_only, priority_and_full_pairs
    min_valid_years: int = 30

    # X construction
    standardize_features_across_years: bool = True
    object_block_equal_weight: bool = True
    std_eps: float = 1.0e-12

    # Audits
    perm_n: int = 500
    split_half_n: int = 100
    random_seed: int = 202603
    run_permutation: bool = True
    run_leave_one_year_influence: bool = True
    run_split_half_stability: bool = True
    run_score_phase_separation: bool = True

    # V7 detector compatibility. Default is strict: no bootstrap. If the local V7
    # helper requires at least one bootstrap for API compatibility, users may opt in.
    allow_detector_compat_bootstrap_retry: bool = False
    compat_detector_bootstrap_n: int = 1

    @classmethod
    def from_env(cls) -> "V93Settings":
        s = cls()
        if os.environ.get("V9_3_TARGET_WINDOWS"):
            s.target_windows = tuple(w.strip() for w in os.environ["V9_3_TARGET_WINDOWS"].split(",") if w.strip())
        if os.environ.get("V9_3_TARGET_MODE"):
            s.target_mode = os.environ["V9_3_TARGET_MODE"].strip()
        if os.environ.get("V9_3_MIN_VALID_YEARS"):
            s.min_valid_years = int(os.environ["V9_3_MIN_VALID_YEARS"])
        if os.environ.get("V9_3_PERM_N"):
            s.perm_n = int(os.environ["V9_3_PERM_N"])
        if os.environ.get("V9_3_SPLIT_HALF_N"):
            s.split_half_n = int(os.environ["V9_3_SPLIT_HALF_N"])
        if os.environ.get("V9_3_RANDOM_SEED"):
            s.random_seed = int(os.environ["V9_3_RANDOM_SEED"])
        if os.environ.get("V9_3_RUN_PERMUTATION"):
            s.run_permutation = _env_bool("V9_3_RUN_PERMUTATION", s.run_permutation)
        if os.environ.get("V9_3_RUN_LOO_INFLUENCE"):
            s.run_leave_one_year_influence = _env_bool("V9_3_RUN_LOO_INFLUENCE", s.run_leave_one_year_influence)
        if os.environ.get("V9_3_RUN_SPLIT_HALF"):
            s.run_split_half_stability = _env_bool("V9_3_RUN_SPLIT_HALF", s.run_split_half_stability)
        if os.environ.get("V9_3_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY"):
            s.allow_detector_compat_bootstrap_retry = _env_bool("V9_3_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY", s.allow_detector_compat_bootstrap_retry)
        if os.environ.get("V9_3_COMPAT_DETECTOR_BOOTSTRAP_N"):
            s.compat_detector_bootstrap_n = int(os.environ["V9_3_COMPAT_DETECTOR_BOOTSTRAP_N"])
        return s


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _set_or_insert_front(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    if column in df.columns:
        df[column] = value
        cols = [column] + [c for c in df.columns if c != column]
        return df.loc[:, cols]
    df.insert(0, column, value)
    return df


def _native_scalar(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _stage_root_from_v93(v93_root: Path) -> Path:
    return v93_root.parent


def _default_v9_out_dir(v93_root: Path) -> Path:
    return v93_root.parent / "V9" / "outputs" / "peak_all_windows_v9_a"


def _default_smoothed_path(v93_root: Path) -> Path:
    return v93_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _import_v7_module(v93_root: Path):
    stage_root = _stage_root_from_v93(v93_root)
    v7_src = stage_root / "V7" / "src"
    if not v7_src.exists():
        raise FileNotFoundError(
            f"Cannot find V7 source directory: {v7_src}. V9.3_a needs V7 helpers "
            "to preserve V9/V7 profile and peak semantics."
        )
    if str(v7_src) not in sys.path:
        sys.path.insert(0, str(v7_src))
    from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as v7multi
    return v7multi


def _make_v7_cfg_for_v93(v7multi, settings: V93Settings, v93_root: Path) -> object:
    cfg = v7multi.MultiWinConfig.from_env()
    cfg.smoothed_fields_path = os.environ.get("V9_3_SMOOTHED_FIELDS", str(_default_smoothed_path(v93_root)))
    cfg.window_mode = "list"
    cfg.target_windows = ",".join(settings.target_windows)
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    cfg.bootstrap_n = 0
    return cfg


def _read_v9_scopes(v9_out_dir: Path, settings: V93Settings) -> List[SimpleNamespace]:
    candidates = [
        v9_out_dir / "cross_window" / "run_window_scope_registry_v9_peak_all_windows_a.csv",
        v9_out_dir / "cross_window" / "window_scope_registry_v9_peak_all_windows_a.csv",
    ]
    scope_path = next((p for p in candidates if p.exists()), None)
    if scope_path is None:
        raise FileNotFoundError("Cannot find V9 window scope registry. Expected one of: " + "; ".join(str(p) for p in candidates))
    df = pd.read_csv(scope_path)
    df = df[df["window_id"].astype(str).isin(settings.target_windows)].copy()
    missing = [w for w in settings.target_windows if w not in set(df["window_id"].astype(str))]
    if missing:
        raise ValueError(f"V9 scope registry is missing requested windows: {missing}")
    scopes = []
    for _, r in df.iterrows():
        data = {k: _native_scalar(v) for k, v in r.to_dict().items()}
        scopes.append(SimpleNamespace(**data))
    order = {w: i for i, w in enumerate(settings.target_windows)}
    scopes.sort(key=lambda s: order[str(s.window_id)])
    return scopes


def _year_indices(years: np.ndarray, settings: V93Settings) -> Tuple[np.ndarray, np.ndarray]:
    years = np.asarray(years).astype(int)
    selected_years = np.arange(settings.years_start, settings.years_end + 1, dtype=int)
    index_map = {int(y): i for i, y in enumerate(years)}
    missing = [int(y) for y in selected_years if int(y) not in index_map]
    if missing:
        raise ValueError(f"smoothed_fields years do not cover requested years: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    indices = np.asarray([index_map[int(y)] for y in selected_years], dtype=int)
    return selected_years, indices


def _window_slice(scope: SimpleNamespace) -> slice:
    return slice(int(scope.analysis_start), int(scope.analysis_end) + 1)


def _build_object_profile_cube(v7multi, fields: dict, lat: np.ndarray, lon: np.ndarray, years: np.ndarray, selected_years: np.ndarray, object_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    spec = next(s for s in v7multi.clean.OBJECT_SPECS if str(s.object_name) == object_name)
    arr_all = v7multi.clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
    years_all = np.asarray(years).astype(int)
    idx_map = {int(y): i for i, y in enumerate(years_all)}
    profiles = []
    target_lat_ref = None
    weights_ref = None
    for y in selected_years:
        arr = arr_all[[idx_map[int(y)]], :, :, :]
        prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
        prof = np.asarray(prof, dtype=float)
        if prof.ndim == 3 and prof.shape[0] == 1:
            prof_y = prof[0]
        elif prof.ndim == 2:
            prof_y = prof
        else:
            raise ValueError(f"Unexpected profile shape for {object_name}, year {y}: {prof.shape}")
        profiles.append(prof_y)
        target_lat_ref = np.asarray(target_lat, dtype=float)
        weights_ref = np.asarray(weights, dtype=float)
    cube = np.stack(profiles, axis=0)  # year x day x coord
    audit = {
        "object": object_name,
        "field_role": spec.field_role,
        "n_years": int(cube.shape[0]),
        "n_days_total": int(cube.shape[1]),
        "n_profile_coord": int(cube.shape[2]),
    }
    return cube, target_lat_ref, weights_ref, audit


def _build_all_profile_cubes(v7multi, fields: dict, lat: np.ndarray, lon: np.ndarray, years: np.ndarray, selected_years: np.ndarray) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], pd.DataFrame]:
    cubes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for obj in OBJECT_ORDER:
        cube, coord, weights, audit = _build_object_profile_cube(v7multi, fields, lat, lon, years, selected_years, obj)
        cubes[obj] = (cube, coord, weights)
        rows.append(audit)
    return cubes, pd.DataFrame(rows)


def _build_X_for_window(scope: SimpleNamespace, profile_cubes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], settings: V93Settings) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    wsl = _window_slice(scope)
    blocks: List[np.ndarray] = []
    meta_object: List[str] = []
    meta_day: List[int] = []
    feature_rows: List[dict] = []
    for obj in OBJECT_ORDER:
        cube, coord, _weights = profile_cubes[obj]
        sub = cube[:, wsl, :]  # year x day x coord
        n_year, n_day, n_coord = sub.shape
        raw = sub.reshape(n_year, -1)
        raw_days = np.repeat(np.arange(int(scope.analysis_start), int(scope.analysis_end) + 1), n_coord)
        all_nan = np.all(~np.isfinite(raw), axis=0)
        raw2 = raw[:, ~all_nan]
        days2 = raw_days[~all_nan]
        mean = np.nanmean(raw2, axis=0)
        centered = raw2 - mean[None, :]
        std = np.nanstd(centered, axis=0, ddof=1)
        valid_std = np.isfinite(std) & (std > settings.std_eps)
        centered = centered[:, valid_std]
        days3 = days2[valid_std]
        std = std[valid_std]
        block = centered / std[None, :] if settings.standardize_features_across_years else centered
        partial_nan_count = int(np.isnan(block).sum())
        if partial_nan_count:
            block = np.where(np.isfinite(block), block, 0.0)
        block_var_before = float(np.nanmean(np.nanvar(block, axis=0, ddof=1))) if block.size else float("nan")
        scale = 1.0
        if settings.object_block_equal_weight and block.shape[1] > 0:
            scale = 1.0 / math.sqrt(float(block.shape[1]))
            block = block * scale
        block_var_after = float(np.nanmean(np.nanvar(block, axis=0, ddof=1))) if block.size else float("nan")
        blocks.append(block)
        meta_object.extend([obj] * block.shape[1])
        meta_day.extend([int(d) for d in days3])
        feature_rows.append({
            "window_id": scope.window_id,
            "object": obj,
            "n_years": int(n_year),
            "analysis_start": int(scope.analysis_start),
            "analysis_end": int(scope.analysis_end),
            "n_days": int(n_day),
            "n_profile_coord": int(n_coord),
            "n_features_raw": int(raw.shape[1]),
            "n_all_nan_features_removed": int(all_nan.sum()),
            "n_features_after_nan_filter": int(raw2.shape[1]),
            "n_features_after_std_filter": int(block.shape[1]),
            "n_partial_nan_values_filled": partial_nan_count,
            "block_scale": float(scale),
            "block_variance_before_scale": block_var_before,
            "block_variance_after_scale": block_var_after,
        })
    X = np.concatenate(blocks, axis=1)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    meta = {"object": np.asarray(meta_object, dtype=object), "day": np.asarray(meta_day, dtype=int)}
    return X, pd.DataFrame(feature_rows), meta


def _single_year_profiles(profile_cubes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], year_index: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for obj, (cube, coord, weights) in profile_cubes.items():
        out[obj] = (cube[year_index:year_index + 1, :, :], coord, weights)
    return out


def _run_v7_peak(v7multi, profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], scope: SimpleNamespace, cfg: object, settings: V93Settings) -> Tuple[pd.DataFrame, str, str]:
    local_cfg = copy.deepcopy(cfg)
    local_cfg.bootstrap_n = 0
    try:
        _score_df, _cand_df, selection_df, _selected_delta_df, _boot_df = v7multi._run_detector_and_bootstrap(profiles, scope, local_cfg)
        return selection_df, "v7_detector_zero_bootstrap", "ok"
    except Exception as exc:
        if not settings.allow_detector_compat_bootstrap_retry:
            raise RuntimeError(
                "V7 _run_detector_and_bootstrap failed with bootstrap_n=0. V9.3_a does not use bootstrap. "
                "If your local V7 helper requires one bootstrap for API compatibility, set "
                "V9_3_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY=1. Original error: " + repr(exc)
            )
        local_cfg = copy.deepcopy(cfg)
        local_cfg.bootstrap_n = int(settings.compat_detector_bootstrap_n)
        _score_df, _cand_df, selection_df, _selected_delta_df, _boot_df = v7multi._run_detector_and_bootstrap(profiles, scope, local_cfg)
        return selection_df, f"v7_detector_compat_bootstrap_n_{settings.compat_detector_bootstrap_n}_ignored", "compat_retry_used"


def _quality_from_selection_row(row: pd.Series, detector_status: str) -> str:
    if detector_status not in {"ok", "compat_retry_used"}:
        return "missing_or_detector_failed"
    peak = row.get("selected_peak_day", np.nan)
    if pd.isna(peak):
        return "missing_or_detector_failed"
    support = str(row.get("support_class", ""))
    excluded = str(row.get("excluded_candidates", ""))
    early = str(row.get("early_secondary_candidates", ""))
    late = str(row.get("late_secondary_candidates", ""))
    has_complexity = any(x not in {"", "nan", "None"} for x in [excluded, early, late])
    if support in {"accepted_window", "candidate_window"} and not has_complexity:
        return "ok_clean"
    if support in {"accepted_window", "candidate_window"}:
        return "ok_with_candidate_complexity"
    return "warning_support_class"


def _build_yearly_peak_registry(v7multi, profile_cubes: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], selected_years: np.ndarray, scopes: Sequence[SimpleNamespace], cfg: object, settings: V93Settings) -> pd.DataFrame:
    rows: List[dict] = []
    for wi, scope in enumerate(scopes, start=1):
        wid = str(scope.window_id)
        _log(f"  yearly peaks {wid}: {wi}/{len(scopes)}")
        for yi, year in enumerate(selected_years):
            profiles = _single_year_profiles(profile_cubes, yi)
            selection_df, peak_source, detector_status = _run_v7_peak(v7multi, profiles, scope, cfg, settings)
            if selection_df is None or selection_df.empty:
                for obj in OBJECT_ORDER:
                    rows.append({"window_id": wid, "year": int(year), "object": obj, "peak_day": np.nan, "detector_status": detector_status, "peak_source": peak_source, "missing_flag": True, "quality_flag": "missing_or_detector_failed"})
                continue
            selection_df = selection_df.copy()
            for _, r in selection_df.iterrows():
                obj = str(r.get("object"))
                if obj not in OBJECT_ORDER:
                    continue
                q = _quality_from_selection_row(r, detector_status)
                rows.append({
                    "window_id": wid,
                    "year": int(year),
                    "object": obj,
                    "peak_day": r.get("selected_peak_day", np.nan),
                    "selected_candidate_id": r.get("selected_candidate_id", np.nan),
                    "selected_window_start": r.get("selected_window_start", np.nan),
                    "selected_window_end": r.get("selected_window_end", np.nan),
                    "selected_role": r.get("selected_role", np.nan),
                    "support_class": r.get("support_class", np.nan),
                    "selection_reason": r.get("selection_reason", np.nan),
                    "excluded_candidates": r.get("excluded_candidates", np.nan),
                    "early_secondary_candidates": r.get("early_secondary_candidates", np.nan),
                    "late_secondary_candidates": r.get("late_secondary_candidates", np.nan),
                    "detector_status": detector_status,
                    "peak_source": peak_source,
                    "boundary_flag": False,
                    "missing_flag": pd.isna(r.get("selected_peak_day", np.nan)),
                    "quality_flag": q,
                })
    return pd.DataFrame(rows)


def _order_label(delta: float) -> str:
    if not np.isfinite(delta):
        return "missing"
    if delta > 0:
        return "A_earlier"
    if delta < 0:
        return "B_earlier"
    return "same_day"


def _target_pairs_for_window(wid: str, settings: V93Settings) -> List[Tuple[str, str, str]]:
    priority = list(PRIORITY_TARGETS.get(wid, ()))
    out: List[Tuple[str, str, str]] = []
    if settings.target_mode in {"priority_only", "priority_and_full_pairs"}:
        out.extend([(a, b, "priority") for a, b in priority])
    if settings.target_mode in {"full_pairs_only", "priority_and_full_pairs"}:
        for a, b in ALL_PAIRS:
            if (a, b) in priority:
                label = "priority" if settings.target_mode == "full_pairs_only" else "priority_plus_full_pair_duplicate_removed"
            else:
                label = "exploratory_full_pair"
            out.append((a, b, label))
    # deduplicate, priority wins
    seen = {}
    for a, b, label in out:
        key = (a, b)
        if key not in seen or seen[key] != "priority":
            seen[key] = "priority" if label.startswith("priority") else label
    return [(a, b, seen[(a, b)]) for a, b in sorted(seen.keys(), key=lambda p: (OBJECT_ORDER.index(p[0]), OBJECT_ORDER.index(p[1])))]


def _build_pair_delta_registry(yearly_peak_df: pd.DataFrame, settings: V93Settings) -> pd.DataFrame:
    rows: List[dict] = []
    for wid, gwin in yearly_peak_df.groupby("window_id"):
        pair_targets = _target_pairs_for_window(str(wid), settings)
        for year, gy in gwin.groupby("year"):
            peaks = {str(r["object"]): r for _, r in gy.iterrows()}
            for a, b, target_set in pair_targets:
                ra = peaks.get(a)
                rb = peaks.get(b)
                pa = float(ra["peak_day"]) if ra is not None and pd.notna(ra.get("peak_day")) else np.nan
                pb = float(rb["peak_day"]) if rb is not None and pd.notna(rb.get("peak_day")) else np.nan
                delta = pb - pa if np.isfinite(pa) and np.isfinite(pb) else np.nan
                qa = str(ra.get("quality_flag")) if ra is not None else "missing"
                qb = str(rb.get("quality_flag")) if rb is not None else "missing"
                if not np.isfinite(delta):
                    pq = "missing_peak"
                elif qa.startswith("ok") and qb.startswith("ok"):
                    pq = "ok"
                else:
                    pq = "warning_peak_quality"
                rows.append({
                    "window_id": wid,
                    "year": int(year),
                    "object_A": a,
                    "object_B": b,
                    "target_pair": f"{a}-{b}",
                    "target_set": target_set,
                    "peak_A": pa,
                    "peak_B": pb,
                    "delta_B_minus_A": delta,
                    "order_label": _order_label(delta),
                    "quality_A": qa,
                    "quality_B": qb,
                    "pair_y_quality_flag": pq,
                })
    return pd.DataFrame(rows)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa = a[mask] - np.nanmean(a[mask])
    bb = b[mask] - np.nanmean(b[mask])
    den = float(np.sqrt(np.nansum(aa ** 2) * np.nansum(bb ** 2)))
    if den <= 0:
        return float("nan")
    return float(np.nansum(aa * bb) / den)


def _standardize_y(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=float)
    mean = float(np.nanmean(y))
    sd = float(np.nanstd(y, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(y, np.nan, dtype=float), mean, sd
    return (y - mean) / sd, mean, sd


def _fit_pls1(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    yz, _m, _s = _standardize_y(y)
    if not np.all(np.isfinite(yz)):
        return np.full(X.shape[1], np.nan), np.full(X.shape[0], np.nan), float("nan")
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    c = Xc.T @ yz
    norm = float(np.sqrt(np.nansum(c ** 2)))
    if norm <= 0 or not np.isfinite(norm):
        return np.full(X.shape[1], np.nan), np.full(X.shape[0], np.nan), float("nan")
    u = c / norm
    score = Xc @ u
    if np.nanstd(score, ddof=1) > 0:
        score = (score - np.nanmean(score)) / np.nanstd(score, ddof=1)
    corr = _corr(score, yz)
    if np.isfinite(corr) and corr < 0:
        u = -u
        score = -score
        corr = -corr
    return u, score, float(corr)


def _loading_corr(u1: np.ndarray, u2: np.ndarray) -> float:
    c = _corr(u1, u2)
    if np.isfinite(c) and c < 0:
        c = -c
    return c


def _dominant_loading_summary(u: np.ndarray, meta: dict, scope: SimpleNamespace) -> dict:
    if u is None or not np.all(np.isfinite(u)):
        return {"dominant_object": "unresolved", "dominant_object_loading_fraction": np.nan, "dominant_time_half": "unresolved"}
    obj_arr = np.asarray(meta["object"], dtype=object)
    day_arr = np.asarray(meta["day"], dtype=int)
    total = float(np.nansum(u ** 2))
    if total <= 0:
        return {"dominant_object": "unresolved", "dominant_object_loading_fraction": np.nan, "dominant_time_half": "unresolved"}
    obj_norms = {}
    for obj in OBJECT_ORDER:
        mask = obj_arr == obj
        obj_norms[obj] = float(np.nansum(u[mask] ** 2))
    dom_obj = max(obj_norms, key=obj_norms.get)
    frac = obj_norms[dom_obj] / total
    mid = (int(scope.analysis_start) + int(scope.analysis_end)) / 2.0
    mask_dom = obj_arr == dom_obj
    early_norm = float(np.nansum(u[mask_dom & (day_arr <= mid)] ** 2))
    late_norm = float(np.nansum(u[mask_dom & (day_arr > mid)] ** 2))
    if early_norm > late_norm * 1.2:
        half = "early_half"
    elif late_norm > early_norm * 1.2:
        half = "late_half"
    else:
        half = "balanced_or_full_window"
    return {"dominant_object": dom_obj, "dominant_object_loading_fraction": frac, "dominant_time_half": half}


def _y_quality_for_target(yrows: pd.DataFrame, settings: V93Settings) -> dict:
    y = yrows["delta_B_minus_A"].to_numpy(dtype=float)
    valid = np.isfinite(y)
    n_valid = int(valid.sum())
    n_missing = int((~valid).sum())
    low_quality = int((yrows["pair_y_quality_flag"].astype(str) != "ok").sum())
    y_valid = y[valid]
    y_std = float(np.nanstd(y_valid, ddof=1)) if n_valid >= 2 else np.nan
    y_range = float(np.nanmax(y_valid) - np.nanmin(y_valid)) if n_valid else np.nan
    y_iqr = float(np.nanpercentile(y_valid, 75) - np.nanpercentile(y_valid, 25)) if n_valid else np.nan
    n_near_zero = int(np.sum(np.abs(y_valid) <= 1.0)) if n_valid else 0
    if n_valid < settings.min_valid_years:
        flag = "not_usable_too_few_valid_years"
    elif not np.isfinite(y_std) or y_std <= 0:
        flag = "not_usable_zero_y_variance"
    elif low_quality / max(len(yrows), 1) > 0.35:
        flag = "warning_many_low_quality_peaks"
    elif n_near_zero / max(n_valid, 1) > 0.60:
        flag = "warning_many_near_zero_deltas"
    else:
        flag = "acceptable"
    return {
        "n_valid_years": n_valid,
        "n_missing_years": n_missing,
        "n_low_quality_peak_years": low_quality,
        "Y_std": y_std,
        "Y_range": y_range,
        "Y_iqr": y_iqr,
        "n_near_zero_delta_years": n_near_zero,
        "y_quality_flag": flag,
    }


def _phase_table(wid: str, target_pair: str, target_set: str, years: np.ndarray, score: np.ndarray, y: np.ndarray, order: Sequence[str], peaks_a: np.ndarray, peaks_b: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"year": years.astype(int), "score": score.astype(float), "Y_delta": y.astype(float), "order_label": list(order), "peak_A": peaks_a, "peak_B": peaks_b})
    df = df.sort_values("score", ascending=True).reset_index(drop=True)
    n = len(df)
    n_low = n // 3
    n_high = n // 3
    labels = np.array(["mid"] * n, dtype=object)
    labels[:n_low] = "low"
    labels[n - n_high:] = "high"
    df["score_phase"] = labels
    mean = float(df["score"].mean())
    sd = float(df["score"].std(ddof=1))
    df["score_z"] = (df["score"] - mean) / sd if sd > 0 else 0.0
    df["window_id"] = wid
    df["target_pair"] = target_pair
    df["target_set"] = target_set
    return df[["window_id", "target_pair", "target_set", "score_phase", "year", "score", "score_z", "Y_delta", "order_label", "peak_A", "peak_B"]]


def _phase_y_summary(phase_df: pd.DataFrame) -> dict:
    out = {}
    for phase in ["high", "low"]:
        g = phase_df[phase_df["score_phase"] == phase]
        y = g["Y_delta"].to_numpy(dtype=float)
        out[f"mean_Y_{phase}"] = float(np.nanmean(y)) if len(y) else np.nan
        out[f"median_Y_{phase}"] = float(np.nanmedian(y)) if len(y) else np.nan
        out[f"n_years_{phase}"] = int(len(g))
        out[f"prob_A_earlier_{phase}"] = float(np.mean(g["order_label"].astype(str) == "A_earlier")) if len(g) else np.nan
        out[f"prob_B_earlier_{phase}"] = float(np.mean(g["order_label"].astype(str) == "B_earlier")) if len(g) else np.nan
    out["high_low_Y_shift"] = out.get("mean_Y_high", np.nan) - out.get("mean_Y_low", np.nan)
    out["order_probability_shift_A_earlier"] = out.get("prob_A_earlier_high", np.nan) - out.get("prob_A_earlier_low", np.nan)
    hi = phase_df[phase_df["score_phase"] == "high"]["Y_delta"].to_numpy(dtype=float)
    lo = phase_df[phase_df["score_phase"] == "low"]["Y_delta"].to_numpy(dtype=float)
    pooled = np.sqrt((np.nanvar(hi, ddof=1) + np.nanvar(lo, ddof=1)) / 2.0) if len(hi) >= 2 and len(lo) >= 2 else np.nan
    out["cohens_d_high_low"] = (np.nanmean(hi) - np.nanmean(lo)) / pooled if np.isfinite(pooled) and pooled > 0 else np.nan
    if np.isfinite(out["cohens_d_high_low"]) and abs(out["cohens_d_high_low"]) >= 0.8:
        sep = "clear"
    elif np.isfinite(out["cohens_d_high_low"]) and abs(out["cohens_d_high_low"]) >= 0.5:
        sep = "moderate"
    else:
        sep = "weak_or_unclear"
    out["phase_separation_flag"] = sep
    return out


def _permutation_audit(X: np.ndarray, y: np.ndarray, corr_real: float, rng: np.random.Generator, perm_n: int) -> dict:
    if perm_n <= 0 or not np.isfinite(corr_real):
        return {"perm_p": np.nan, "perm_q90": np.nan, "perm_q95": np.nan, "perm_q99": np.nan, "perm_pass_90": False, "perm_pass_95": False, "perm_pass_99": False}
    vals = np.empty(perm_n, dtype=float)
    for i in range(perm_n):
        yp = rng.permutation(y)
        _u, _s, cp = _fit_pls1(X, yp)
        vals[i] = abs(cp) if np.isfinite(cp) else np.nan
    abs_real = abs(corr_real)
    good = vals[np.isfinite(vals)]
    if len(good) == 0:
        return {"perm_p": np.nan, "perm_q90": np.nan, "perm_q95": np.nan, "perm_q99": np.nan, "perm_pass_90": False, "perm_pass_95": False, "perm_pass_99": False}
    p = float((np.sum(good >= abs_real) + 1.0) / (len(good) + 1.0))
    q90, q95, q99 = np.nanpercentile(good, [90, 95, 99])
    return {"perm_p": p, "perm_q90": float(q90), "perm_q95": float(q95), "perm_q99": float(q99), "perm_pass_90": bool(abs_real > q90), "perm_pass_95": bool(abs_real > q95), "perm_pass_99": bool(abs_real > q99)}


def _loo_audit(X: np.ndarray, y: np.ndarray, years: np.ndarray, u_full: np.ndarray, corr_full: float) -> Tuple[pd.DataFrame, dict]:
    rows = []
    for i, yr in enumerate(years):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        u, score, corr = _fit_pls1(X[mask], y[mask])
        lcorr = _loading_corr(u_full, u)
        rows.append({"left_out_year": int(yr), "corr_without_year": corr, "delta_corr_from_full": corr - corr_full if np.isfinite(corr) and np.isfinite(corr_full) else np.nan, "loading_corr_with_full": lcorr})
    df = pd.DataFrame(rows)
    if df.empty:
        summary = {"min_corr_without_year": np.nan, "max_abs_delta_corr": np.nan, "worst_left_out_year": np.nan, "min_loading_corr": np.nan, "single_year_dominated_flag": False}
    else:
        idx = df["delta_corr_from_full"].abs().idxmax() if df["delta_corr_from_full"].notna().any() else df.index[0]
        max_delta = float(df["delta_corr_from_full"].abs().max()) if df["delta_corr_from_full"].notna().any() else np.nan
        min_corr = float(df["corr_without_year"].min()) if df["corr_without_year"].notna().any() else np.nan
        min_lcorr = float(df["loading_corr_with_full"].min()) if df["loading_corr_with_full"].notna().any() else np.nan
        dominated = bool((np.isfinite(max_delta) and max_delta >= 0.30) or (np.isfinite(min_corr) and min_corr < max(0.0, corr_full - 0.35)) or (np.isfinite(min_lcorr) and min_lcorr < 0.50))
        summary = {"min_corr_without_year": min_corr, "max_abs_delta_corr": max_delta, "worst_left_out_year": int(df.loc[idx, "left_out_year"]), "min_loading_corr": min_lcorr, "single_year_dominated_flag": dominated}
    return df, summary


def _split_half_audit(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, split_n: int) -> Tuple[pd.DataFrame, dict]:
    rows = []
    n = len(y)
    if split_n <= 0 or n < 10:
        return pd.DataFrame(), {"median_test_corr": np.nan, "frac_positive_test_corr": np.nan, "median_loading_corr": np.nan, "split_half_stability_flag": "not_run"}
    for sid in range(split_n):
        idx = rng.permutation(n)
        half = n // 2
        a = idx[:half]
        b = idx[half:]
        u_a, score_a, corr_a = _fit_pls1(X[a], y[a])
        u_b, score_b, corr_b = _fit_pls1(X[b], y[b])
        test_b = _corr(X[b] @ u_a, _standardize_y(y[b])[0]) if np.all(np.isfinite(u_a)) else np.nan
        test_a = _corr(X[a] @ u_b, _standardize_y(y[a])[0]) if np.all(np.isfinite(u_b)) else np.nan
        lc = _loading_corr(u_a, u_b)
        rows.append({"split_id": sid + 1, "train_corr_a": corr_a, "train_corr_b": corr_b, "test_corr_a_to_b": test_b, "test_corr_b_to_a": test_a, "loading_corr_between_halves": lc})
    df = pd.DataFrame(rows)
    tests = pd.concat([df["test_corr_a_to_b"], df["test_corr_b_to_a"]], ignore_index=True).to_numpy(dtype=float)
    tests = tests[np.isfinite(tests)]
    med_test = float(np.nanmedian(tests)) if len(tests) else np.nan
    frac_pos = float(np.mean(tests > 0)) if len(tests) else np.nan
    med_load = float(np.nanmedian(df["loading_corr_between_halves"])) if df["loading_corr_between_halves"].notna().any() else np.nan
    if np.isfinite(med_test) and med_test >= 0.30 and np.isfinite(frac_pos) and frac_pos >= 0.70 and np.isfinite(med_load) and med_load >= 0.40:
        flag = "stable_or_acceptable"
    elif np.isfinite(med_test) and med_test > 0 and np.isfinite(frac_pos) and frac_pos >= 0.60:
        flag = "marginal"
    else:
        flag = "unstable_or_unclear"
    return df, {"median_test_corr": med_test, "frac_positive_test_corr": frac_pos, "median_loading_corr": med_load, "split_half_stability_flag": flag}


def _assign_usability(yq: dict, corr: float, perm: dict, loo: dict, split: dict, sep: dict) -> Tuple[str, str, str]:
    reasons = []
    if not str(yq.get("y_quality_flag", "")).startswith("acceptable"):
        reasons.append(str(yq.get("y_quality_flag")))
    if not np.isfinite(corr) or corr < 0.30:
        reasons.append("weak_score_y_corr")
    if np.isfinite(perm.get("perm_p", np.nan)) and perm["perm_p"] > 0.05:
        reasons.append("permutation_not_95")
    elif not np.isfinite(perm.get("perm_p", np.nan)):
        reasons.append("permutation_not_run_or_invalid")
    if bool(loo.get("single_year_dominated_flag", False)):
        reasons.append("single_year_influence_warning")
    sh = str(split.get("split_half_stability_flag", "not_run"))
    if sh not in {"stable_or_acceptable"}:
        reasons.append(f"split_half_{sh}")
    sepflag = str(sep.get("phase_separation_flag", "weak_or_unclear"))
    if sepflag == "weak_or_unclear":
        reasons.append("weak_score_phase_separation")
    # Levels
    if str(yq.get("y_quality_flag")) in {"not_usable_too_few_valid_years", "not_usable_zero_y_variance"}:
        level = "Level_D_not_usable"
    elif len(reasons) == 0:
        level = "Level_A_usable"
    elif len(reasons) <= 2 and corr >= 0.30 and sepflag in {"clear", "moderate"}:
        level = "Level_B_candidate"
    elif corr >= 0.20:
        level = "Level_C_exploratory"
    else:
        level = "Level_D_not_usable"
    permission = {
        "Level_A_usable": "usable_statistical_target; still not physical mechanism",
        "Level_B_candidate": "candidate_result; requires cautious interpretation",
        "Level_C_exploratory": "exploratory_only; do not use as main result",
        "Level_D_not_usable": "not_usable_for_result_extraction",
    }[level]
    return level, ";".join([r for r in reasons if r and r != "acceptable"]), permission


def _write_summary(path: Path, registry_df: pd.DataFrame, settings: V93Settings) -> None:
    lines = [
        "# V9.3_a result summary",
        "",
        "This is a direct-year profile-evolution supervised SVD / PLS1 audit.",
        "It uses yearly peak_B - peak_A as Y, so yearly peak quality is a core risk.",
        "",
        "## Method boundaries",
        "- No 2D-field X is used.",
        "- No bootstrap year-resampling is used as samples.",
        "- The supervised direction is target-guided, not an unsupervised natural mode.",
        "- Result extraction should use result_usability_level, not raw corr_score_y alone.",
        "",
        "## Settings",
        f"- target_mode: {settings.target_mode}",
        f"- perm_n: {settings.perm_n if settings.run_permutation else 0}",
        f"- split_half_n: {settings.split_half_n if settings.run_split_half_stability else 0}",
        "",
        "## Usability counts",
    ]
    if registry_df.empty or "result_usability_level" not in registry_df.columns:
        lines.append("No registry rows were produced.")
    else:
        counts = registry_df["result_usability_level"].value_counts(dropna=False)
        for k, v in counts.items():
            lines.append(f"- {k}: {int(v)}")
        lines.extend(["", "## Top candidate rows by corr_score_y", ""])
        cols = ["window_id", "target_pair", "target_set", "corr_score_y", "perm_p", "result_usability_level", "downgrade_reasons"]
        top = registry_df.sort_values("corr_score_y", ascending=False).head(20)
        lines.append(top[cols].to_markdown(index=False))
    lines.extend([
        "",
        "## Interpretation restrictions",
        "- Do not call score high/low groups physical year types without separate physical audit.",
        "- Do not interpret dominant_object as a physical driver.",
        "- Do not use Level C/D rows as main results.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_direct_year_profile_supervised_svd_v9_3_a(v93_root: Path | str) -> None:
    v93_root = Path(v93_root)
    settings = V93Settings.from_env()
    out_root = _ensure_dir(v93_root / "outputs" / settings.output_tag)
    out_cross = _ensure_dir(out_root / "cross_window")
    out_per = _ensure_dir(out_root / "per_window")
    log_dir = _ensure_dir(v93_root / "logs" / settings.output_tag)
    t0 = time.time()

    _log("[1/12] Resolve paths and import V7/V9 helpers")
    v7multi = _import_v7_module(v93_root)
    cfg = _make_v7_cfg_for_v93(v7multi, settings, v93_root)
    v9_out_dir = Path(os.environ.get("V9_3_V9_OUTPUT_DIR", str(_default_v9_out_dir(v93_root))))
    smoothed_path = Path(os.environ.get("V9_3_SMOOTHED_FIELDS", cfg.smoothed_fields_path))
    if not v9_out_dir.exists():
        raise FileNotFoundError(f"V9 output directory not found: {v9_out_dir}")
    if not smoothed_path.exists():
        raise FileNotFoundError(f"smoothed_fields.npz not found: {smoothed_path}")

    _log("[2/12] Load V9 window scopes")
    scopes = _read_v9_scopes(v9_out_dir, settings)
    _safe_to_csv(pd.DataFrame([vars(s) for s in scopes]), out_cross / "v9_3_window_scope_used.csv")

    _log("[3/12] Load V7/foundation fields")
    fields, input_audit = v7multi.clean._load_npz_fields(smoothed_path)
    years_raw = np.asarray(fields["years"]).astype(int)
    selected_years, year_idx = _year_indices(years_raw, settings)
    lat = np.asarray(fields["lat"], dtype=float)
    lon = np.asarray(fields["lon"], dtype=float)
    input_rows = [
        {"item": "smoothed_fields_path", "value": str(smoothed_path)},
        {"item": "v9_output_dir", "value": str(v9_out_dir)},
        {"item": "years_start", "value": int(settings.years_start)},
        {"item": "years_end", "value": int(settings.years_end)},
        {"item": "n_years_selected", "value": int(len(selected_years))},
        {"item": "lat_order", "value": "ascending" if lat[0] <= lat[-1] else "descending"},
        {"item": "lon_order", "value": "ascending" if lon[0] <= lon[-1] else "descending"},
    ]
    _safe_to_csv(pd.DataFrame(input_rows), out_cross / "v9_3_input_audit.csv")

    _log("[4/12] Build yearly object profile cubes")
    profile_cubes, profile_audit_df = _build_all_profile_cubes(v7multi, fields, lat, lon, years_raw, selected_years)
    _safe_to_csv(profile_audit_df, out_cross / "v9_3_yearly_profile_cube_audit.csv")

    _log("[5/12] Build yearly object peak registry")
    yearly_peak_df = _build_yearly_peak_registry(v7multi, profile_cubes, selected_years, scopes, cfg, settings)
    _safe_to_csv(yearly_peak_df, out_cross / "v9_3_yearly_object_peak_registry.csv")

    _log("[6/12] Build yearly pair peak delta registry")
    pair_delta_df = _build_pair_delta_registry(yearly_peak_df, settings)
    _safe_to_csv(pair_delta_df, out_cross / "v9_3_yearly_pair_peak_delta_registry.csv")

    all_feature_audits: List[pd.DataFrame] = []
    all_yq: List[dict] = []
    all_results: List[dict] = []
    all_phase_years: List[pd.DataFrame] = []
    all_phase_summaries: List[dict] = []
    all_perm: List[dict] = []
    all_loo_rows: List[pd.DataFrame] = []
    all_loo_summaries: List[dict] = []
    all_split_rows: List[pd.DataFrame] = []
    all_split_summaries: List[dict] = []
    all_sep: List[dict] = []
    registry_rows: List[dict] = []

    rng = np.random.default_rng(settings.random_seed)

    _log("[7/12] Build X matrices and run supervised SVD / audits")
    scope_map = {str(s.window_id): s for s in scopes}
    for si, scope in enumerate(scopes, start=1):
        wid = str(scope.window_id)
        _log(f"  [{si}/{len(scopes)}] {wid}: build profile X")
        X_full, feat_audit, meta = _build_X_for_window(scope, profile_cubes, settings)
        all_feature_audits.append(feat_audit)
        win_dir = _ensure_dir(out_per / wid)
        _safe_to_csv(feat_audit, win_dir / f"feature_audit_{wid}.csv")
        win_targets = pair_delta_df[pair_delta_df["window_id"].astype(str) == wid].copy()
        pairs = _target_pairs_for_window(wid, settings)
        for ti, (a, b, target_set) in enumerate(pairs, start=1):
            target_pair = f"{a}-{b}"
            _log(f"    target {ti}/{len(pairs)}: {wid} {target_pair}")
            yrows = win_targets[(win_targets["object_A"] == a) & (win_targets["object_B"] == b)].copy().sort_values("year")
            yq = _y_quality_for_target(yrows, settings)
            yq.update({"window_id": wid, "target_pair": target_pair, "target_set": target_set})
            all_yq.append(yq)
            valid_mask = np.isfinite(yrows["delta_B_minus_A"].to_numpy(dtype=float))
            years_valid = yrows.loc[valid_mask, "year"].to_numpy(dtype=int)
            # selected_years is sorted and aligned to X_full rows
            idx_map = {int(y): i for i, y in enumerate(selected_years)}
            xidx = np.asarray([idx_map[int(y)] for y in years_valid], dtype=int)
            X = X_full[xidx]
            y = yrows.loc[valid_mask, "delta_B_minus_A"].to_numpy(dtype=float)
            order = yrows.loc[valid_mask, "order_label"].astype(str).tolist()
            peaks_a = yrows.loc[valid_mask, "peak_A"].to_numpy(dtype=float)
            peaks_b = yrows.loc[valid_mask, "peak_B"].to_numpy(dtype=float)
            if len(y) < settings.min_valid_years or not np.isfinite(yq["Y_std"]) or yq["Y_std"] <= 0:
                base = {"window_id": wid, "target_pair": target_pair, "target_set": target_set, **yq}
                base.update({"corr_score_y": np.nan, "result_usability_level": "Level_D_not_usable", "downgrade_reasons": yq["y_quality_flag"], "interpretation_permission": "not_usable_for_result_extraction"})
                registry_rows.append(base)
                continue
            u, score, corr = _fit_pls1(X, y)
            dom = _dominant_loading_summary(u, meta, scope)
            res = {"window_id": wid, "target_pair": target_pair, "target_set": target_set, "n_valid_years": int(len(y)), "corr_score_y": corr, **dom}
            all_results.append(res)
            phase_df = _phase_table(wid, target_pair, target_set, years_valid, score, y, order, peaks_a, peaks_b)
            all_phase_years.append(phase_df)
            phase_sum = _phase_y_summary(phase_df)
            phase_sum.update({"window_id": wid, "target_pair": target_pair, "target_set": target_set})
            all_phase_summaries.append(phase_sum)
            sep = {"window_id": wid, "target_pair": target_pair, "target_set": target_set, **phase_sum}
            all_sep.append(sep)

            perm = {"window_id": wid, "target_pair": target_pair, "target_set": target_set}
            if settings.run_permutation:
                perm.update(_permutation_audit(X, y, corr, rng, settings.perm_n))
            else:
                perm.update({"perm_p": np.nan, "perm_q90": np.nan, "perm_q95": np.nan, "perm_q99": np.nan, "perm_pass_90": False, "perm_pass_95": False, "perm_pass_99": False})
            all_perm.append(perm)

            loo_summary = {"window_id": wid, "target_pair": target_pair, "target_set": target_set, "min_corr_without_year": np.nan, "max_abs_delta_corr": np.nan, "worst_left_out_year": np.nan, "min_loading_corr": np.nan, "single_year_dominated_flag": False}
            if settings.run_leave_one_year_influence:
                loo_df, loo_s = _loo_audit(X, y, years_valid, u, corr)
                if not loo_df.empty:
                    loo_df.insert(0, "target_set", target_set)
                    loo_df.insert(0, "target_pair", target_pair)
                    loo_df.insert(0, "window_id", wid)
                    all_loo_rows.append(loo_df)
                loo_summary.update(loo_s)
            all_loo_summaries.append(loo_summary)

            split_summary = {"window_id": wid, "target_pair": target_pair, "target_set": target_set, "median_test_corr": np.nan, "frac_positive_test_corr": np.nan, "median_loading_corr": np.nan, "split_half_stability_flag": "not_run"}
            if settings.run_split_half_stability:
                split_df, split_s = _split_half_audit(X, y, rng, settings.split_half_n)
                if not split_df.empty:
                    split_df.insert(0, "target_set", target_set)
                    split_df.insert(0, "target_pair", target_pair)
                    split_df.insert(0, "window_id", wid)
                    all_split_rows.append(split_df)
                split_summary.update(split_s)
            all_split_summaries.append(split_summary)

            level, reasons, permission = _assign_usability(yq, corr, perm, loo_summary, split_summary, phase_sum)
            reg = {
                "window_id": wid,
                "target_pair": target_pair,
                "target_set": target_set,
                **{k: v for k, v in yq.items() if k not in {"window_id", "target_pair", "target_set"}},
                "corr_score_y": corr,
                "dominant_object": dom.get("dominant_object"),
                "dominant_object_loading_fraction": dom.get("dominant_object_loading_fraction"),
                "dominant_time_half": dom.get("dominant_time_half"),
                **{k: v for k, v in perm.items() if k not in {"window_id", "target_pair", "target_set"}},
                **{k: v for k, v in loo_summary.items() if k not in {"window_id", "target_pair", "target_set"}},
                **{k: v for k, v in split_summary.items() if k not in {"window_id", "target_pair", "target_set"}},
                "phase_separation_flag": phase_sum.get("phase_separation_flag"),
                "high_low_Y_shift": phase_sum.get("high_low_Y_shift"),
                "cohens_d_high_low": phase_sum.get("cohens_d_high_low"),
                "result_usability_level": level,
                "downgrade_reasons": reasons,
                "interpretation_permission": permission,
            }
            registry_rows.append(reg)

    _log("[8/12] Write cross-window CSV outputs")
    feature_audit_df = pd.concat(all_feature_audits, ignore_index=True) if all_feature_audits else pd.DataFrame()
    yq_df = pd.DataFrame(all_yq)
    result_df = pd.DataFrame(all_results)
    phase_years_df = pd.concat(all_phase_years, ignore_index=True) if all_phase_years else pd.DataFrame()
    phase_summary_df = pd.DataFrame(all_phase_summaries)
    perm_df = pd.DataFrame(all_perm)
    loo_rows_df = pd.concat(all_loo_rows, ignore_index=True) if all_loo_rows else pd.DataFrame()
    loo_summary_df = pd.DataFrame(all_loo_summaries)
    split_rows_df = pd.concat(all_split_rows, ignore_index=True) if all_split_rows else pd.DataFrame()
    split_summary_df = pd.DataFrame(all_split_summaries)
    sep_df = pd.DataFrame(all_sep)
    registry_df = pd.DataFrame(registry_rows)

    _safe_to_csv(feature_audit_df, out_cross / "v9_3_feature_block_audit.csv")
    _safe_to_csv(yq_df, out_cross / "v9_3_y_quality_audit.csv")
    _safe_to_csv(result_df, out_cross / "v9_3_supervised_svd_target_results.csv")
    _safe_to_csv(phase_years_df, out_cross / "v9_3_score_phase_years.csv")
    _safe_to_csv(phase_summary_df, out_cross / "v9_3_score_phase_y_summary.csv")
    _safe_to_csv(perm_df, out_cross / "v9_3_permutation_audit.csv")
    _safe_to_csv(loo_rows_df, out_cross / "v9_3_leave_one_year_influence_audit.csv")
    _safe_to_csv(loo_summary_df, out_cross / "v9_3_single_year_influence_summary.csv")
    _safe_to_csv(split_rows_df, out_cross / "v9_3_split_half_stability_audit.csv")
    _safe_to_csv(split_summary_df, out_cross / "v9_3_split_half_stability_summary.csv")
    _safe_to_csv(sep_df, out_cross / "v9_3_score_phase_separation_audit.csv")
    _safe_to_csv(registry_df, out_cross / "v9_3_result_usability_registry.csv")

    _log("[9/12] Write per-window slices")
    for wid in settings.target_windows:
        wdir = _ensure_dir(out_per / wid)
        for name, df in [
            ("y_quality", yq_df),
            ("supervised_svd_target_results", result_df),
            ("score_phase_years", phase_years_df),
            ("score_phase_y_summary", phase_summary_df),
            ("permutation_audit", perm_df),
            ("single_year_influence_summary", loo_summary_df),
            ("split_half_stability_summary", split_summary_df),
            ("result_usability_registry", registry_df),
        ]:
            if not df.empty and "window_id" in df.columns:
                _safe_to_csv(df[df["window_id"].astype(str) == wid].copy(), wdir / f"{name}_{wid}.csv")

    _log("[10/12] Write summary and run_meta")
    _write_summary(out_cross / "V9_3_A_SUMMARY.md", registry_df, settings)
    run_meta = {
        "version": settings.version,
        "output_tag": settings.output_tag,
        "created_at": _now(),
        "elapsed_seconds": round(time.time() - t0, 3),
        "v9_output_dir": str(v9_out_dir),
        "smoothed_fields_path": str(smoothed_path),
        "windows": list(settings.target_windows),
        "years_start": settings.years_start,
        "years_end": settings.years_end,
        "n_years": int(len(selected_years)),
        "settings": asdict(settings),
        "method_boundaries": {
            "uses_2d_field_x": False,
            "uses_bootstrap_resampling_as_samples": False,
            "uses_yearly_peak_y": True,
            "supervised_target_guided_svd": True,
            "physical_interpretation_included": False,
        },
        "n_registry_rows": int(len(registry_df)),
        "outputs_are_final_physical_results": False,
    }
    _write_json(run_meta, out_cross / "run_meta.json")
    _write_json(run_meta, out_root / "run_meta.json")

    (log_dir / "last_run.txt").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Done. Output: {out_root}")


if __name__ == "__main__":
    run_direct_year_profile_supervised_svd_v9_3_a(Path(__file__).resolve().parents[2])
