"""
V9 peak-selection sensitivity test.

Purpose
-------
This module does not re-detect the transition windows and does not replace the
V9 peak-only baseline.  It fixes the four V7/V9 accepted main windows
(W045, W081, W113, W160) and perturbs only the object-peak detection/selection
layer in order to test whether P/V/H/Je/Jw selected peak days, selected peak
bands, pairwise ordering, and full five-object sequences are stable under
reasonable changes of

    * input smoothing scale: smooth9 vs smooth5;
    * detector temporal scale: 16/8, 20/10, 24/12;
    * search range: anchor±15, V9 baseline detector range, analysis range;
    * candidate selection rule: V9-like baseline, max score, closest anchor,
      max overlap with system window.

Interpretation boundary
-----------------------
The outputs are sensitivity diagnostics only.  They are not physical mechanism
results, not a new changepoint detector, and not a substitute for the original
V9 peak baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

VERSION = "V9_peak_selection_sensitivity_a"
OUTPUT_TAG = "peak_selection_sensitivity_v9_a"
TARGET_WINDOWS = ("W045", "W081", "W113", "W160")
OBJECTS = ("P", "V", "H", "Je", "Jw")
OBJECT_SPECS = {
    "P": {"role": "precip", "key_candidates": ("precip_smoothed", "precip"), "lon_min": 105, "lon_max": 125, "lat_min": 15, "lat_max": 39},
    "V": {"role": "v850", "key_candidates": ("v850_smoothed", "v850"), "lon_min": 105, "lon_max": 125, "lat_min": 10, "lat_max": 30},
    "H": {"role": "z500", "key_candidates": ("z500_smoothed", "z500"), "lon_min": 110, "lon_max": 140, "lat_min": 15, "lat_max": 35},
    "Je": {"role": "u200", "key_candidates": ("u200_smoothed", "u200"), "lon_min": 120, "lon_max": 150, "lat_min": 25, "lat_max": 45},
    "Jw": {"role": "u200", "key_candidates": ("u200_smoothed", "u200"), "lon_min": 80, "lon_max": 110, "lat_min": 25, "lat_max": 45},
}


@dataclass(frozen=True)
class PeakSelectionSensitivitySettings:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    target_windows: Tuple[str, ...] = TARGET_WINDOWS
    objects: Tuple[str, ...] = OBJECTS
    detector_scales: Tuple[Tuple[str, int, int], ...] = (
        ("narrow", 16, 8),
        ("baseline", 20, 10),
        ("wide", 24, 12),
    )
    search_modes: Tuple[str, ...] = ("narrow_search", "baseline_search", "wide_search")
    selection_rules: Tuple[str, ...] = ("baseline_rule", "max_score", "closest_anchor", "max_overlap")
    near_tie_days: int = 1
    stable_peak_day_range_days: int = 3
    moderate_peak_day_range_days: int = 7
    smooth_delta_consistent_days: int = 2
    smooth_delta_moderate_days: int = 5
    max_peaks_per_object: int = 5
    neighbor_buffer: int = 5
    include_bootstrap_consistency_rule: bool = False
    rerun_changepoint_detection: bool = False
    physical_interpretation_included: bool = False


@dataclass(frozen=True)
class SensitivityConfig:
    config_id: str
    smoothing: str
    detector_scale: str
    detector_width: int
    band_half_width: int
    search_mode: str
    selection_rule: str


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


def _day_to_md(day: float) -> str:
    if pd.isna(day):
        return ""
    d = date(2001, 4, 1) + timedelta(days=int(round(float(day))))
    return f"{d.month}月{d.day}日"


def _as_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x, default: Optional[int] = None) -> Optional[int]:
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default



def _nanmean_no_warning(arr, axis=None):
    """NaN-aware mean that keeps all-NaN slices as NaN without warnings.

    Boundary days produced by running smoothers can be entirely NaN.  Those
    slices should remain NaN in downstream peak detection, but they should not
    repeatedly print "Mean of empty slice" warnings during sensitivity runs.
    """
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    count = finite.sum(axis=axis)
    total = np.where(finite, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    if np.isscalar(mean):
        return np.nan if count == 0 else float(mean)
    mean = np.asarray(mean, dtype=float)
    mean = np.where(count > 0, mean, np.nan)
    return mean


def _stage_root_from_v9_root(v9_root: Path) -> Path:
    # D:/easm_project01/stage_partition/V9 -> D:/easm_project01/stage_partition
    return v9_root.parent


def _project_root_from_v9_root(v9_root: Path) -> Path:
    # D:/easm_project01/stage_partition/V9 -> D:/easm_project01
    return v9_root.parents[1]


def _default_smooth9_path(v9_root: Path) -> Path:
    return _project_root_from_v9_root(v9_root) / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _default_smooth5_path(v9_root: Path) -> Path:
    return _project_root_from_v9_root(v9_root) / "foundation" / "V1" / "outputs" / "baseline_smooth5_a" / "preprocess" / "smoothed_fields.npz"


def _resolve_paths(v9_root: Path) -> Dict[str, Path]:
    v9_output_dir = Path(os.environ.get("V9_SENS_V9_OUTPUT_DIR", str(v9_root / "outputs" / "peak_all_windows_v9_a")))
    out_dir = Path(os.environ.get("V9_SENS_OUTPUT_DIR", str(v9_root / "outputs" / OUTPUT_TAG)))
    smooth9 = Path(os.environ.get("V9_SENS_SMOOTH9_FIELDS", str(_default_smooth9_path(v9_root))))
    smooth5 = Path(os.environ.get("V9_SENS_SMOOTH5_FIELDS", str(_default_smooth5_path(v9_root))))
    return {
        "v9_root": v9_root,
        "stage_root": _stage_root_from_v9_root(v9_root),
        "project_root": _project_root_from_v9_root(v9_root),
        "v9_output_dir": v9_output_dir,
        "out_dir": out_dir,
        "smooth9": smooth9,
        "smooth5": smooth5,
    }


def _validate_paths(paths: Dict[str, Path]) -> None:
    missing = []
    for key in ("v9_output_dir", "smooth9", "smooth5"):
        if not paths[key].exists():
            missing.append(f"{key}: {paths[key]}")
    if missing:
        raise FileNotFoundError("Missing required V9 sensitivity inputs:\n" + "\n".join(missing))


def _import_v7_module(stage_root: Path):
    v7_src = stage_root / "V7" / "src"
    if str(v7_src) not in sys.path:
        sys.path.insert(0, str(v7_src))
    try:
        from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as v7multi
        return v7multi
    except Exception as exc:
        _log(f"WARNING: V7 helper import failed; fallback profile builder will be used where possible: {exc}")
        return None


def _load_window_scopes(v9_output_dir: Path) -> pd.DataFrame:
    cross = v9_output_dir / "cross_window"
    p = cross / "run_window_scope_registry_v9_peak_all_windows_a.csv"
    if not p.exists():
        p = cross / "window_scope_registry_v9_peak_all_windows_a.csv"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find V9 scope registry under {cross}")
    df = pd.read_csv(p)
    df = df[df["window_id"].isin(TARGET_WINDOWS)].copy()
    if set(df["window_id"]) != set(TARGET_WINDOWS):
        raise ValueError(f"Scope registry does not contain all target windows {TARGET_WINDOWS}: {sorted(df['window_id'].unique())}")
    required = ["window_id", "anchor_day", "system_window_start", "system_window_end", "detector_search_start", "detector_search_end", "analysis_start", "analysis_end"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Scope registry missing required columns: {miss}")
    for c in required[1:]:
        df[c] = df[c].astype(int)
    return df[required].sort_values("anchor_day").reset_index(drop=True)


def _load_v9_baseline(v9_output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cross = v9_output_dir / "cross_window"
    sel_path = cross / "main_window_selection_all_windows.csv"
    cand_path = cross / "object_profile_window_registry_all_windows.csv"
    if not sel_path.exists() or not cand_path.exists():
        raise FileNotFoundError(f"Cannot find V9 baseline selection/candidate tables under {cross}")
    selection = pd.read_csv(sel_path)
    candidates = pd.read_csv(cand_path)
    selection = selection[selection["window_id"].isin(TARGET_WINDOWS)].copy()
    candidates = candidates[candidates["window_id"].isin(TARGET_WINDOWS)].copy()
    return selection, candidates


def _resolve_npz_key(npz: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> str:
    keys = set(npz.files)
    for k in candidates:
        if k in keys:
            return k
    raise KeyError(f"None of keys {candidates} found in NPZ. Available keys include: {sorted(list(keys))[:20]}")


def _load_fields(path: Path) -> Dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=False)
    out = {
        "lat": np.asarray(npz[_resolve_npz_key(npz, ("lat", "latitude"))], dtype=float),
        "lon": np.asarray(npz[_resolve_npz_key(npz, ("lon", "longitude"))], dtype=float),
    }
    if "years" in npz.files:
        out["years"] = np.asarray(npz["years"])
    for obj, spec in OBJECT_SPECS.items():
        role = spec["role"]
        if role in out:
            continue
        out[role] = np.asarray(npz[_resolve_npz_key(npz, spec["key_candidates"])], dtype=float)
    return out


def _to_year_day_lat_lon(arr: np.ndarray, role: str, v7multi=None, lat=None, lon=None) -> np.ndarray:
    # Prefer V7 normalization when available.  Fallback assumes year/day/lat/lon already.
    if v7multi is not None and hasattr(v7multi, "clean") and hasattr(v7multi.clean, "_as_year_day_lat_lon"):
        try:
            return np.asarray(v7multi.clean._as_year_day_lat_lon(arr, role), dtype=float)
        except TypeError:
            try:
                return np.asarray(v7multi.clean._as_year_day_lat_lon(arr), dtype=float)
            except Exception:
                pass
        except Exception:
            pass
    a = np.asarray(arr, dtype=float)
    if a.ndim != 4:
        raise ValueError(f"Expected {role} array to be 4D year×day×lat×lon; got shape {a.shape}")
    return a


def _fallback_object_profile(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build year × day × latitude-profile by longitude averaging.

    This is a fallback used only if V7's exact object-profile helper is not
    importable.  It preserves object domains and latitude ordering, but V7 helper
    output is preferred for strict reproduction.
    """
    a = np.asarray(arr, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_mask = (lat >= spec["lat_min"]) & (lat <= spec["lat_max"])
    lon_mask = (lon >= spec["lon_min"]) & (lon <= spec["lon_max"])
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(f"Empty lat/lon subset for spec {spec}")
    sub = a[:, :, lat_mask, :][:, :, :, lon_mask]
    # longitude mean, leaving latitude profile.  Do not force lat sorting here;
    # V9/V7 data may use high-to-low lat, which is acceptable for peak timing.
    prof = _nanmean_no_warning(sub, axis=3)
    target_lat = lat[lat_mask]
    weights = np.ones_like(target_lat, dtype=float)
    return prof, target_lat, weights


def _build_profiles_for_smoothing(path: Path, v7multi=None) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    fields = _load_fields(path)
    lat, lon = fields["lat"], fields["lon"]
    profiles: Dict[str, np.ndarray] = {}
    audit_rows: List[dict] = []
    for obj, spec in OBJECT_SPECS.items():
        role = spec["role"]
        arr = _to_year_day_lat_lon(fields[role], role, v7multi=v7multi, lat=lat, lon=lon)
        used_helper = False
        if v7multi is not None and hasattr(v7multi, "clean") and hasattr(v7multi.clean, "_build_object_profile"):
            try:
                prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
                used_helper = True
            except Exception:
                prof, target_lat, weights = _fallback_object_profile(arr, lat, lon, spec)
        else:
            prof, target_lat, weights = _fallback_object_profile(arr, lat, lon, spec)
        profiles[obj] = np.asarray(prof, dtype=float)
        audit_rows.append({
            "object": obj,
            "role": role,
            "profile_shape": "x".join(map(str, profiles[obj].shape)),
            "lat_min": float(np.nanmin(target_lat)) if len(target_lat) else np.nan,
            "lat_max": float(np.nanmax(target_lat)) if len(target_lat) else np.nan,
            "n_profile_lat": int(len(target_lat)),
            "used_v7_profile_helper": bool(used_helper),
        })
    return profiles, pd.DataFrame(audit_rows)


def _composite_profile(prof: np.ndarray) -> np.ndarray:
    # year × day × coord -> day × coord
    return _nanmean_no_warning(np.asarray(prof, dtype=float), axis=0)


def _compute_detector_score(day_coord: np.ndarray, detector_width: int) -> pd.DataFrame:
    """Simple pre/post contrast score for sensitivity testing.

    The original V9/V7 detector is preferred for exact baseline reproduction when
    available.  This local score exists so sensitivity configurations remain
    runnable for changed widths/search ranges without invoking bootstrap.  It is
    used consistently across non-baseline configurations and is marked in run_meta.
    """
    arr = np.asarray(day_coord, dtype=float)
    n_days = arr.shape[0]
    half = max(2, int(detector_width // 2))
    rows = []
    for d in range(n_days):
        if d - half < 0 or d + half >= n_days:
            rows.append({"day": d, "detector_score": np.nan, "score_valid": False})
            continue
        before = arr[d-half:d, :]
        after = arr[d+1:d+1+half, :]
        if not np.isfinite(before).any() or not np.isfinite(after).any():
            rows.append({"day": d, "detector_score": np.nan, "score_valid": False})
            continue
        diff = _nanmean_no_warning(after, axis=0) - _nanmean_no_warning(before, axis=0)
        score = float(np.sqrt(np.nansum(diff * diff)))
        rows.append({"day": d, "detector_score": score, "score_valid": np.isfinite(score)})
    return pd.DataFrame(rows)


def _candidate_relation(peak_day: int, band_start: int, band_end: int, scope: pd.Series) -> Tuple[str, int, float]:
    sys_s = int(scope["system_window_start"]); sys_e = int(scope["system_window_end"])
    overlap = max(0, min(band_end, sys_e) - max(band_start, sys_s) + 1)
    sys_len = max(1, sys_e - sys_s + 1)
    frac = overlap / sys_len
    if sys_s <= peak_day <= sys_e:
        relation = "within_system_window"
    elif peak_day < sys_s:
        relation = "front_or_early" if band_end >= sys_s else "pre_window"
    else:
        relation = "late_or_post" if band_start <= sys_e else "post_window"
    return relation, int(overlap), float(frac)


def _support_class(overlap_fraction: float, relation: str) -> str:
    if relation == "within_system_window" and overlap_fraction >= 0.65:
        return "accepted_window"
    if overlap_fraction > 0.0 or relation in ("front_or_early", "late_or_post"):
        return "candidate_window"
    return "weak_window"


def _find_candidate_peaks(score_df: pd.DataFrame, scope: pd.Series, search_start: int, search_end: int, band_half_width: int, max_peaks: int, neighbor_buffer: int) -> pd.DataFrame:
    df = score_df[(score_df["day"] >= search_start) & (score_df["day"] <= search_end) & (score_df["score_valid"] == True)].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("detector_score", ascending=False)
    selected = []
    used_days: List[int] = []
    for _, r in df.iterrows():
        d = int(r["day"])
        if any(abs(d - u) <= neighbor_buffer for u in used_days):
            continue
        used_days.append(d)
        selected.append(r)
        if len(selected) >= max_peaks:
            break
    rows = []
    analysis_s = int(scope["analysis_start"]); analysis_e = int(scope["analysis_end"])
    anchor = int(scope["anchor_day"])
    for i, r in enumerate(selected, start=1):
        peak_day = int(r["day"])
        bs = max(analysis_s, peak_day - int(band_half_width))
        be = min(analysis_e, peak_day + int(band_half_width))
        relation, overlap, frac = _candidate_relation(peak_day, bs, be, scope)
        rows.append({
            "candidate_id": f"CP{i:03d}",
            "peak_day": peak_day,
            "band_start_day": bs,
            "band_end_day": be,
            "peak_score": float(r["detector_score"]),
            "peak_prominence": np.nan,
            "peak_rank": i,
            "candidate_rank_by_score": i,
            "distance_to_anchor": abs(peak_day - anchor),
            "overlap_with_system_window": overlap,
            "overlap_fraction_with_system_window": frac,
            "support_class": _support_class(frac, relation),
            "relation_to_system_window": relation,
            "detector_source": "local_prepost_contrast_score",
        })
    return pd.DataFrame(rows)


def _v9_candidates_for_baseline(v9_candidates: pd.DataFrame, window_id: str, obj: str) -> pd.DataFrame:
    sub = v9_candidates[(v9_candidates["window_id"] == window_id) & (v9_candidates["object"] == obj)].copy()
    if sub.empty:
        return sub
    # Normalize names for shared selection code.
    if "peak_score" in sub.columns and "detector_score" not in sub.columns:
        sub["detector_score"] = sub["peak_score"]
    if "overlap_days_with_W45" in sub.columns and "overlap_with_system_window" not in sub.columns:
        sub["overlap_with_system_window"] = sub["overlap_days_with_W45"]
    if "overlap_fraction_with_W45" in sub.columns and "overlap_fraction_with_system_window" not in sub.columns:
        sub["overlap_fraction_with_system_window"] = sub["overlap_fraction_with_W45"]
    if "peak_rank" in sub.columns and "candidate_rank_by_score" not in sub.columns:
        sub["candidate_rank_by_score"] = sub["peak_rank"]
    return sub


def _v9_selection_for_baseline(v9_selection: pd.DataFrame, window_id: str, obj: str) -> Optional[pd.Series]:
    sub = v9_selection[(v9_selection["window_id"] == window_id) & (v9_selection["object"] == obj)]
    if sub.empty:
        return None
    return sub.iloc[0]


def _select_candidate(candidates: pd.DataFrame, selection_rule: str, scope: pd.Series, v9_selection_row: Optional[pd.Series] = None) -> Optional[pd.Series]:
    if candidates is None or candidates.empty:
        return None
    cand = candidates.copy()
    # baseline exact: if V9 row provided, choose its candidate id from candidates.
    if selection_rule == "baseline_rule" and v9_selection_row is not None:
        cid = str(v9_selection_row.get("selected_candidate_id", ""))
        sub = cand[cand["candidate_id"].astype(str) == cid]
        if not sub.empty:
            return sub.iloc[0]
    support_priority = {"accepted_window": 0, "candidate_window": 1, "weak_window": 2}
    relation_priority = {"within_system_window": 0, "front_or_early": 1, "late_or_post": 2, "pre_window": 3, "post_window": 4}
    cand["_support_rank"] = cand.get("support_class", "weak_window").map(support_priority).fillna(9)
    cand["_relation_rank"] = cand.get("relation_to_system_window", "post_window").map(relation_priority).fillna(9)
    cand["_score"] = cand.get("detector_score", cand.get("peak_score", np.nan)).astype(float)
    cand["_dist"] = cand.get("distance_to_anchor", np.abs(cand["peak_day"].astype(float) - float(scope["anchor_day"]))).astype(float)
    cand["_overlap"] = cand.get("overlap_with_system_window", 0).fillna(0).astype(float)
    if selection_rule == "baseline_rule":
        sort_cols = ["_support_rank", "_relation_rank", "_dist", "_score"]
        asc = [True, True, True, False]
    elif selection_rule == "max_score":
        sort_cols = ["_score", "_dist"]
        asc = [False, True]
    elif selection_rule == "closest_anchor":
        sort_cols = ["_dist", "_score"]
        asc = [True, False]
    elif selection_rule == "max_overlap":
        sort_cols = ["_overlap", "_score", "_dist"]
        asc = [False, False, True]
    else:
        raise ValueError(f"Unknown selection_rule: {selection_rule}")
    cand = cand.sort_values(sort_cols, ascending=asc)
    return cand.iloc[0]


def _search_range(scope: pd.Series, mode: str) -> Tuple[int, int]:
    anchor = int(scope["anchor_day"])
    analysis_s = int(scope["analysis_start"]); analysis_e = int(scope["analysis_end"])
    if mode == "narrow_search":
        return max(analysis_s, anchor - 15), min(analysis_e, anchor + 15)
    if mode == "baseline_search":
        return int(scope["detector_search_start"]), int(scope["detector_search_end"])
    if mode == "wide_search":
        return analysis_s, analysis_e
    raise ValueError(f"Unknown search_mode: {mode}")


def _config_grid(settings: PeakSelectionSensitivitySettings) -> List[SensitivityConfig]:
    configs: List[SensitivityConfig] = []
    for smoothing in ("smooth9", "smooth5"):
        for scale_name, width, half_width in settings.detector_scales:
            for search_mode in settings.search_modes:
                for rule in settings.selection_rules:
                    cid = f"{smoothing}__{scale_name}_w{width}_b{half_width}__{search_mode}__{rule}"
                    configs.append(SensitivityConfig(cid, smoothing, scale_name, width, half_width, search_mode, rule))
    return configs


def _is_baseline_config(cfg: SensitivityConfig) -> bool:
    return (
        cfg.smoothing == "smooth9" and cfg.detector_width == 20 and cfg.band_half_width == 10
        and cfg.search_mode == "baseline_search" and cfg.selection_rule == "baseline_rule"
    )


def _run_peak_selection_grid(settings: PeakSelectionSensitivitySettings, scopes: pd.DataFrame, profiles_by_smooth: Dict[str, Dict[str, np.ndarray]], v9_selection: pd.DataFrame, v9_candidates: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    configs = _config_grid(settings)
    total = len(configs) * len(scopes) * len(settings.objects)
    done = 0
    for cfg in configs:
        for _, scope in scopes.iterrows():
            wid = str(scope["window_id"])
            s_start, s_end = _search_range(scope, cfg.search_mode)
            for obj in settings.objects:
                done += 1
                if done % 100 == 0:
                    _log(f"    peak selections {done}/{total}")
                v9_sel = _v9_selection_for_baseline(v9_selection, wid, obj)
                baseline_peak = _as_int(v9_sel.get("selected_peak_day")) if v9_sel is not None else None
                if _is_baseline_config(cfg):
                    candidates = _v9_candidates_for_baseline(v9_candidates, wid, obj)
                    selected = _select_candidate(candidates, cfg.selection_rule, scope, v9_selection_row=v9_sel)
                    detector_source = "v9_original_candidates"
                else:
                    prof = profiles_by_smooth[cfg.smoothing][obj]
                    comp = _composite_profile(prof)
                    score_df = _compute_detector_score(comp, cfg.detector_width)
                    candidates = _find_candidate_peaks(score_df, scope, s_start, s_end, cfg.band_half_width, settings.max_peaks_per_object, settings.neighbor_buffer)
                    selected = _select_candidate(candidates, cfg.selection_rule, scope, v9_selection_row=None)
                    detector_source = "local_prepost_contrast_score"
                if selected is None:
                    rows.append({
                        "smoothing": cfg.smoothing, "window_id": wid, "object": obj, "config_id": cfg.config_id,
                        "detector_scale": cfg.detector_scale, "detector_width": cfg.detector_width, "band_half_width": cfg.band_half_width,
                        "search_mode": cfg.search_mode, "search_start_day": s_start, "search_end_day": s_end, "selection_rule": cfg.selection_rule,
                        "selected_peak_day": np.nan, "selected_peak_date": "", "selected_window_start": np.nan, "selected_window_end": np.nan,
                        "selected_candidate_id": "", "candidate_rank_by_score": np.nan, "detector_score": np.nan,
                        "distance_to_anchor": np.nan, "overlap_with_system_window": np.nan, "support_class": "no_candidate",
                        "relation_to_system_window": "no_candidate", "detector_source": detector_source,
                        "is_v9_baseline_peak": False, "v9_baseline_peak_day": baseline_peak, "delta_from_v9_baseline": np.nan,
                    })
                    continue
                peak_day = _as_int(selected.get("peak_day"))
                score = _as_float(selected.get("detector_score", selected.get("peak_score", np.nan)))
                rank = _as_int(selected.get("candidate_rank_by_score", selected.get("peak_rank", np.nan)))
                dist = _as_float(selected.get("distance_to_anchor", abs(peak_day - int(scope["anchor_day"])) if peak_day is not None else np.nan))
                overlap = _as_float(selected.get("overlap_with_system_window", selected.get("overlap_days_with_W45", np.nan)))
                rows.append({
                    "smoothing": cfg.smoothing,
                    "window_id": wid,
                    "object": obj,
                    "config_id": cfg.config_id,
                    "detector_scale": cfg.detector_scale,
                    "detector_width": cfg.detector_width,
                    "band_half_width": cfg.band_half_width,
                    "search_mode": cfg.search_mode,
                    "search_start_day": s_start,
                    "search_end_day": s_end,
                    "selection_rule": cfg.selection_rule,
                    "selected_peak_day": peak_day,
                    "selected_peak_date": _day_to_md(peak_day),
                    "selected_window_start": _as_int(selected.get("band_start_day")),
                    "selected_window_end": _as_int(selected.get("band_end_day")),
                    "selected_candidate_id": str(selected.get("candidate_id", "")),
                    "candidate_rank_by_score": rank,
                    "detector_score": score,
                    "distance_to_anchor": dist,
                    "overlap_with_system_window": overlap,
                    "support_class": str(selected.get("support_class", "")),
                    "relation_to_system_window": str(selected.get("relation_to_system_window", "")),
                    "detector_source": detector_source,
                    "is_v9_baseline_peak": bool(baseline_peak is not None and peak_day == baseline_peak),
                    "v9_baseline_peak_day": baseline_peak,
                    "delta_from_v9_baseline": (peak_day - baseline_peak) if (peak_day is not None and baseline_peak is not None) else np.nan,
                })
    return pd.DataFrame(rows)


def _iqr(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))


def _stability_class(row: pd.Series, settings: PeakSelectionSensitivitySettings) -> str:
    r = _as_float(row.get("peak_day_range"))
    n_cand = _as_int(row.get("n_unique_selected_candidates"), 999)
    smooth_flag = bool(row.get("smooth_sensitive_flag", False))
    if smooth_flag:
        return "smooth_sensitive_peak"
    if r <= settings.stable_peak_day_range_days and n_cand <= 2:
        return "stable_peak_day"
    if r <= settings.moderate_peak_day_range_days:
        return "peak_day_moderately_sensitive"
    return "peak_day_rule_sensitive"


def _band_class(row: pd.Series) -> str:
    start_range = _as_float(row.get("band_start_range"))
    end_range = _as_float(row.get("band_end_range"))
    m = max(start_range if np.isfinite(start_range) else 0, end_range if np.isfinite(end_range) else 0)
    if m <= 3:
        return "band_stable"
    if m <= 7:
        return "band_moderately_sensitive"
    return "band_highly_sensitive"


def _summarize_object_peak(selection_df: pd.DataFrame, settings: PeakSelectionSensitivitySettings) -> pd.DataFrame:
    rows: List[dict] = []
    for (wid, obj), sub in selection_df.groupby(["window_id", "object"], dropna=False):
        x = pd.to_numeric(sub["selected_peak_day"], errors="coerce").dropna()
        base_vals = pd.to_numeric(sub.loc[sub["is_v9_baseline_peak"], "selected_peak_day"], errors="coerce")
        v9_baseline = _as_int(sub["v9_baseline_peak_day"].dropna().iloc[0]) if sub["v9_baseline_peak_day"].notna().any() else None
        cand_counts = sub["selected_candidate_id"].astype(str).value_counts(dropna=True)
        most_cand = cand_counts.index[0] if not cand_counts.empty else ""
        peak_counts = sub["selected_peak_day"].value_counts(dropna=True)
        most_peak = _as_int(peak_counts.index[0]) if not peak_counts.empty else None
        smooth_medians = {}
        for sm in ("smooth9", "smooth5"):
            ss = pd.to_numeric(sub.loc[sub["smoothing"] == sm, "selected_peak_day"], errors="coerce").dropna()
            smooth_medians[sm] = float(np.nanmedian(ss)) if not ss.empty else np.nan
        delta_sm = smooth_medians["smooth5"] - smooth_medians["smooth9"] if np.isfinite(smooth_medians["smooth5"]) and np.isfinite(smooth_medians["smooth9"]) else np.nan
        baseline_candidate = ""
        base_row = sub[(sub["smoothing"] == "smooth9") & (sub["detector_width"] == 20) & (sub["band_half_width"] == 10) & (sub["search_mode"] == "baseline_search") & (sub["selection_rule"] == "baseline_rule")]
        if not base_row.empty:
            baseline_candidate = str(base_row.iloc[0].get("selected_candidate_id", ""))
        baseline_rate = float((sub["selected_candidate_id"].astype(str) == baseline_candidate).mean()) if baseline_candidate else np.nan
        row = {
            "window_id": wid,
            "object": obj,
            "v9_baseline_peak_day": v9_baseline,
            "v9_baseline_peak_date": _day_to_md(v9_baseline),
            "peak_day_min": float(x.min()) if not x.empty else np.nan,
            "peak_day_max": float(x.max()) if not x.empty else np.nan,
            "peak_day_range": float(x.max() - x.min()) if not x.empty else np.nan,
            "peak_day_iqr": _iqr(x),
            "peak_day_std": float(x.std()) if len(x) > 1 else 0.0,
            "n_unique_peak_days": int(x.nunique()) if not x.empty else 0,
            "n_unique_selected_candidates": int(sub["selected_candidate_id"].astype(str).nunique()),
            "most_frequent_peak_day": most_peak,
            "most_frequent_peak_date": _day_to_md(most_peak),
            "most_frequent_candidate_id": most_cand,
            "baseline_candidate_selection_rate": baseline_rate,
            "smooth9_median_peak": smooth_medians["smooth9"],
            "smooth5_median_peak": smooth_medians["smooth5"],
            "delta_smooth5_minus_smooth9": delta_sm,
            "band_start_min": float(pd.to_numeric(sub["selected_window_start"], errors="coerce").min()),
            "band_start_max": float(pd.to_numeric(sub["selected_window_start"], errors="coerce").max()),
            "band_end_min": float(pd.to_numeric(sub["selected_window_end"], errors="coerce").min()),
            "band_end_max": float(pd.to_numeric(sub["selected_window_end"], errors="coerce").max()),
            "candidate_switch_flag": bool(sub["selected_candidate_id"].astype(str).nunique() > 1),
            "early_late_switch_flag": bool(pd.to_numeric(sub["selected_peak_day"], errors="coerce").max() - pd.to_numeric(sub["selected_peak_day"], errors="coerce").min() > settings.moderate_peak_day_range_days),
            "smooth_sensitive_flag": bool(np.isfinite(delta_sm) and abs(delta_sm) > settings.smooth_delta_moderate_days),
        }
        row["band_start_range"] = row["band_start_max"] - row["band_start_min"]
        row["band_end_range"] = row["band_end_max"] - row["band_end_min"]
        row["band_width_range"] = max(row["band_start_range"], row["band_end_range"])
        row["peak_day_stability_class"] = _stability_class(pd.Series(row), settings)
        row["band_stability_class"] = _band_class(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def _order_label(a_day: float, b_day: float, obj_a: str, obj_b: str, near_tie: int) -> str:
    if pd.isna(a_day) or pd.isna(b_day):
        return "missing"
    a = float(a_day); b = float(b_day)
    if abs(a - b) <= near_tie:
        return "same_or_near"
    return f"{obj_a}_earlier" if a < b else f"{obj_b}_earlier"


def _sequence_for_config(group: pd.DataFrame, near_tie: int) -> Tuple[str, str, str, str]:
    vals = {r["object"]: _as_float(r["selected_peak_day"]) for _, r in group.iterrows()}
    items = sorted([(obj, vals.get(obj, np.nan)) for obj in OBJECTS], key=lambda x: (np.inf if pd.isna(x[1]) else x[1], x[0]))
    # Group near-ties by iterative clustering in sorted order.
    groups = []
    for obj, day in items:
        if pd.isna(day):
            continue
        if not groups:
            groups.append([(obj, day)])
        else:
            if abs(day - groups[-1][-1][1]) <= near_tie:
                groups[-1].append((obj, day))
            else:
                groups.append([(obj, day)])
    parts = []
    parts_days = []
    for g in groups:
        objs = "/".join([x[0] for x in g])
        # If group has more than one day in near tie, print median-rounded day.
        d = int(round(float(np.nanmedian([x[1] for x in g]))))
        parts.append(objs)
        parts_days.append(f"{objs}(day{d}/{_day_to_md(d)})")
    seq = " -> ".join(parts)
    seq_days = " -> ".join(parts_days)
    front = parts[0] if parts else ""
    last = parts[-1] if parts else ""
    return seq, seq_days, front, last


def _build_window_sequence(selection_df: pd.DataFrame, settings: PeakSelectionSensitivitySettings) -> pd.DataFrame:
    rows: List[dict] = []
    group_cols = ["window_id", "config_id", "smoothing", "detector_width", "band_half_width", "search_mode", "selection_rule"]
    for keys, sub in selection_df.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))
        seq, seq_days, front, last = _sequence_for_config(sub, settings.near_tie_days)
        row = dict(d)
        for obj in OBJECTS:
            ss = sub[sub["object"] == obj]
            row[f"{obj}_peak_day"] = _as_float(ss.iloc[0]["selected_peak_day"]) if not ss.empty else np.nan
        row.update({"sequence_string": seq, "sequence_with_days": seq_days, "front_object": front, "last_object": last})
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_window_sequence(seq_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wid, sub in seq_df.groupby("window_id"):
        base = sub[(sub["smoothing"] == "smooth9") & (sub["detector_width"] == 20) & (sub["band_half_width"] == 10) & (sub["search_mode"] == "baseline_search") & (sub["selection_rule"] == "baseline_rule")]
        v9_seq = base.iloc[0]["sequence_string"] if not base.empty else ""
        vc = sub["sequence_string"].value_counts(dropna=False)
        most = vc.index[0] if not vc.empty else ""
        freq = float(vc.iloc[0] / len(sub)) if len(sub) else np.nan
        smooth_common = {}
        for sm in ("smooth9", "smooth5"):
            ss = sub[sub["smoothing"] == sm]
            v = ss["sequence_string"].value_counts(dropna=False)
            smooth_common[sm] = v.index[0] if not v.empty else ""
        if freq >= 0.75:
            cls = "sequence_stable"
        elif freq >= 0.50:
            cls = "sequence_moderately_sensitive"
        else:
            cls = "sequence_highly_sensitive"
        rows.append({
            "window_id": wid,
            "v9_baseline_sequence": v9_seq,
            "most_common_sequence": most,
            "most_common_sequence_frequency": freq,
            "n_unique_sequences": int(sub["sequence_string"].nunique()),
            "smooth9_most_common_sequence": smooth_common["smooth9"],
            "smooth5_most_common_sequence": smooth_common["smooth5"],
            "front_object_set": ";".join(sorted(set(sub["front_object"].astype(str)))),
            "front_object_stability": "single_front" if sub["front_object"].nunique() == 1 else "front_sensitive",
            "last_object_set": ";".join(sorted(set(sub["last_object"].astype(str)))),
            "sequence_switch_flag": bool(sub["sequence_string"].nunique() > 1),
            "sequence_stability_class": cls,
        })
    return pd.DataFrame(rows)


def _summarize_pairwise_order(selection_df: pd.DataFrame, settings: PeakSelectionSensitivitySettings) -> pd.DataFrame:
    # Pivot to config-window with object columns.
    piv = selection_df.pivot_table(index=["window_id", "config_id", "smoothing"], columns="object", values="selected_peak_day", aggfunc="first").reset_index()
    rows = []
    for wid, sub in piv.groupby("window_id"):
        base = sub[sub["config_id"].str.contains("smooth9__baseline_w20_b10__baseline_search__baseline_rule", regex=False)]
        for a, b in combinations(OBJECTS, 2):
            labels = sub.apply(lambda r: _order_label(r.get(a), r.get(b), a, b, settings.near_tie_days), axis=1)
            vc = labels.value_counts(normalize=True)
            base_label = ""
            if not base.empty:
                base_label = _order_label(base.iloc[0].get(a), base.iloc[0].get(b), a, b, settings.near_tie_days)
            rows.append({
                "window_id": wid,
                "object_A": a,
                "object_B": b,
                "v9_baseline_order": base_label,
                "A_earlier_rate_across_configs": float(vc.get(f"{a}_earlier", 0.0)),
                "B_earlier_rate_across_configs": float(vc.get(f"{b}_earlier", 0.0)),
                "same_or_near_rate_across_configs": float(vc.get("same_or_near", 0.0)),
                "order_switch_flag": bool(sum(1 for k, v in vc.items() if v > 0 and k not in ("missing",)) > 1),
                "order_stability_class": "order_stable" if vc.max() >= 0.75 else ("order_moderately_sensitive" if vc.max() >= 0.50 else "order_highly_sensitive"),
                "smooth9_A_earlier_rate": float((sub[sub["smoothing"] == "smooth9"].apply(lambda r: _order_label(r.get(a), r.get(b), a, b, settings.near_tie_days), axis=1) == f"{a}_earlier").mean()) if len(sub[sub["smoothing"] == "smooth9"]) else np.nan,
                "smooth5_A_earlier_rate": float((sub[sub["smoothing"] == "smooth5"].apply(lambda r: _order_label(r.get(a), r.get(b), a, b, settings.near_tie_days), axis=1) == f"{a}_earlier").mean()) if len(sub[sub["smoothing"] == "smooth5"]) else np.nan,
            })
    out = pd.DataFrame(rows)
    out["smooth_sensitivity_order_flag"] = (out["smooth9_A_earlier_rate"] - out["smooth5_A_earlier_rate"]).abs() > 0.30
    return out


def _smooth_comparison(selection_df: pd.DataFrame, settings: PeakSelectionSensitivitySettings) -> pd.DataFrame:
    id_cols = ["window_id", "object", "detector_scale", "detector_width", "band_half_width", "search_mode", "selection_rule"]
    s9 = selection_df[selection_df["smoothing"] == "smooth9"][id_cols + ["selected_peak_day"]].rename(columns={"selected_peak_day": "peak_day_smooth9"})
    s5 = selection_df[selection_df["smoothing"] == "smooth5"][id_cols + ["selected_peak_day"]].rename(columns={"selected_peak_day": "peak_day_smooth5"})
    out = s9.merge(s5, on=id_cols, how="outer")
    out["delta_5minus9"] = out["peak_day_smooth5"] - out["peak_day_smooth9"]
    out["same_peak_within_2d"] = out["delta_5minus9"].abs() <= settings.smooth_delta_consistent_days
    out["same_peak_within_5d"] = out["delta_5minus9"].abs() <= settings.smooth_delta_moderate_days
    def cls(d):
        if pd.isna(d):
            return "missing"
        if abs(d) <= settings.smooth_delta_consistent_days:
            return "smooth_consistent"
        if abs(d) <= settings.smooth_delta_moderate_days:
            return "smooth_moderately_sensitive"
        return "smooth_sensitive"
    out["smooth_sensitivity_class"] = out["delta_5minus9"].apply(cls)
    return out


def _baseline_reproduction(selection_df: pd.DataFrame, v9_selection: pd.DataFrame) -> pd.DataFrame:
    base = selection_df[(selection_df["smoothing"] == "smooth9") & (selection_df["detector_width"] == 20) & (selection_df["band_half_width"] == 10) & (selection_df["search_mode"] == "baseline_search") & (selection_df["selection_rule"] == "baseline_rule")]
    rows = []
    for _, r in v9_selection[v9_selection["window_id"].isin(TARGET_WINDOWS)].iterrows():
        wid, obj = r["window_id"], r["object"]
        ss = base[(base["window_id"] == wid) & (base["object"] == obj)]
        sens_day = _as_int(ss.iloc[0]["selected_peak_day"]) if not ss.empty else None
        v9_day = _as_int(r.get("selected_peak_day"))
        rows.append({
            "window_id": wid,
            "object": obj,
            "v9_original_peak_day": v9_day,
            "sensitivity_baseline_peak_day": sens_day,
            "match_flag": bool(v9_day == sens_day),
            "delta_day": (sens_day - v9_day) if (sens_day is not None and v9_day is not None) else np.nan,
            "v9_original_selected_candidate_id": r.get("selected_candidate_id", ""),
            "sensitivity_selected_candidate_id": ss.iloc[0].get("selected_candidate_id", "") if not ss.empty else "",
            "note": "",
        })
    return pd.DataFrame(rows)


def _write_summary_md(path: Path, settings: PeakSelectionSensitivitySettings, baseline_audit: pd.DataFrame, obj_summary: pd.DataFrame, seq_summary: pd.DataFrame) -> None:
    lines = [
        "# V9 peak-selection sensitivity A summary",
        "",
        f"version: `{VERSION}`",
        "",
        "## Purpose",
        "This run fixes the V7/V9 accepted main windows and perturbs only the object-peak selection layer.",
        "It tests whether P/V/H/Je/Jw selected peak days, peak bands, pairwise order, and five-object sequences are stable under detector-scale, search-range, selection-rule, and smoothing-scale changes.",
        "",
        "## Fixed elements",
        "- Main windows are fixed: W045, W081, W113, W160.",
        "- Object definitions are unchanged from V9.",
        "- No changepoint detection is rerun.",
        "- No physical mechanism interpretation is included.",
        "",
        "## Perturbation grid",
        "- smoothing: smooth9, smooth5",
        "- detector scales: 16/8, 20/10, 24/12",
        "- search modes: narrow_search, baseline_search, wide_search",
        "- selection rules: baseline_rule, max_score, closest_anchor, max_overlap",
        "",
        "## Baseline reproduction",
    ]
    if baseline_audit.empty:
        lines.append("- baseline audit table is empty.")
    else:
        n = len(baseline_audit); p = int(baseline_audit.get("match_flag", pd.Series(dtype=bool)).sum())
        lines.append(f"- matched V9 original selected peak day: {p}/{n}")
        if p != n:
            bad = baseline_audit[~baseline_audit["match_flag"]]
            lines.append("- WARNING: baseline reproduction failed for:")
            for _, r in bad.iterrows():
                lines.append(f"  - {r['window_id']} {r['object']}: V9={r['v9_original_peak_day']}, sensitivity={r['sensitivity_baseline_peak_day']}")
    lines += ["", "## Window sequence stability"]
    if not seq_summary.empty:
        for _, r in seq_summary.iterrows():
            lines.append(f"- {r['window_id']}: {r['sequence_stability_class']}; most_common={r['most_common_sequence']} (freq={r['most_common_sequence_frequency']:.2f}); n_unique={r['n_unique_sequences']}")
    lines += ["", "## Most sensitive object peaks"]
    if not obj_summary.empty and "peak_day_range" in obj_summary.columns:
        tmp = obj_summary.sort_values("peak_day_range", ascending=False).head(10)
        for _, r in tmp.iterrows():
            lines.append(f"- {r['window_id']} {r['object']}: range={r['peak_day_range']}, class={r['peak_day_stability_class']}, smooth_delta={r['delta_smooth5_minus_smooth9']}")
    lines += [
        "",
        "## Interpretation boundary",
        "- These outputs are sensitivity diagnostics only.",
        "- A stable peak day can be used as a V9 reference peak; a sensitive peak should be downgraded.",
        "- Selected peak bands should not be interpreted as physical sub-windows unless band stability is also supported.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_peak_selection_sensitivity_v9_a(v9_root: Path) -> None:
    settings = PeakSelectionSensitivitySettings()
    paths = _resolve_paths(Path(v9_root))
    out_dir = paths["out_dir"]
    out_cross = _ensure_dir(out_dir / "cross_window")
    out_per = _ensure_dir(out_dir / "per_window")

    _log("[1/10] Resolve settings and paths")
    _validate_paths(paths)
    v7multi = _import_v7_module(paths["stage_root"])

    _log("[2/10] Load V9 baseline peak outputs")
    v9_selection, v9_candidates = _load_v9_baseline(paths["v9_output_dir"])

    _log("[3/10] Load fixed accepted window scopes")
    scopes = _load_window_scopes(paths["v9_output_dir"])

    _log("[4/10] Load smooth9 and smooth5 fields")
    profiles_by_smooth: Dict[str, Dict[str, np.ndarray]] = {}
    audits = []
    for sm, p in (("smooth9", paths["smooth9"]), ("smooth5", paths["smooth5"])):
        _log(f"    build profiles for {sm}: {p}")
        prof, audit = _build_profiles_for_smoothing(p, v7multi=v7multi)
        audit.insert(0, "smoothing", sm)
        audit.insert(1, "source_file", str(p))
        audits.append(audit)
        profiles_by_smooth[sm] = prof
    _safe_to_csv(pd.concat(audits, ignore_index=True), out_cross / "object_profile_build_audit.csv")

    _log("[5/10] Build sensitivity configuration grid")
    configs = _config_grid(settings)
    _safe_to_csv(pd.DataFrame([asdict(c) for c in configs]), out_cross / "sensitivity_config_grid.csv")

    _log("[6/10] Run candidate peak detection and selection by config")
    selection_by_config = _run_peak_selection_grid(settings, scopes, profiles_by_smooth, v9_selection, v9_candidates)
    _safe_to_csv(selection_by_config, out_cross / "object_peak_selection_by_config.csv")

    _log("[7/10] Run baseline reproduction audit")
    baseline_audit = _baseline_reproduction(selection_by_config, v9_selection)
    _safe_to_csv(baseline_audit, out_cross / "baseline_reproduction_audit.csv")

    _log("[8/10] Summarize object, pairwise, and sequence sensitivity")
    obj_summary = _summarize_object_peak(selection_by_config, settings)
    pair_summary = _summarize_pairwise_order(selection_by_config, settings)
    seq_by_config = _build_window_sequence(selection_by_config, settings)
    seq_summary = _summarize_window_sequence(seq_by_config)
    smooth_cmp = _smooth_comparison(selection_by_config, settings)
    _safe_to_csv(obj_summary, out_cross / "object_peak_sensitivity_summary.csv")
    _safe_to_csv(pair_summary, out_cross / "pairwise_order_sensitivity_summary.csv")
    _safe_to_csv(seq_by_config, out_cross / "window_sequence_by_config.csv")
    _safe_to_csv(seq_summary, out_cross / "window_sequence_sensitivity_summary.csv")
    _safe_to_csv(smooth_cmp, out_cross / "smooth9_vs_smooth5_peak_comparison.csv")

    # Per-window slices for convenience.
    for wid in TARGET_WINDOWS:
        wdir = _ensure_dir(out_per / wid)
        for name, df in [
            ("object_peak_selection_by_config", selection_by_config),
            ("object_peak_sensitivity_summary", obj_summary),
            ("pairwise_order_sensitivity_summary", pair_summary),
            ("window_sequence_by_config", seq_by_config),
        ]:
            if "window_id" in df.columns:
                _safe_to_csv(df[df["window_id"] == wid].copy(), wdir / f"{name}_{wid}.csv")

    _log("[9/10] Write run_meta and summary")
    meta = {
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "main_windows_fixed": True,
        "main_windows_source": "V7 accepted windows / V9 scope registry",
        "rerun_changepoint_detection": False,
        "object_definitions_changed": False,
        "smoothing_inputs": {"smooth9": str(paths["smooth9"]), "smooth5": str(paths["smooth5"])},
        "detector_widths": [x[1] for x in settings.detector_scales],
        "band_half_widths": [x[2] for x in settings.detector_scales],
        "search_modes": list(settings.search_modes),
        "selection_rules": list(settings.selection_rules),
        "bootstrap_consistency_rule_included": False,
        "n_configs": len(configs),
        "n_object_peak_selection_rows": int(len(selection_by_config)),
        "uses_local_prepost_contrast_score_for_nonbaseline_configs": True,
        "baseline_reproduction_required": True,
        "physical_interpretation_included": False,
        "thresholds": {
            "near_tie_days": settings.near_tie_days,
            "stable_peak_day_range_days": settings.stable_peak_day_range_days,
            "moderate_peak_day_range_days": settings.moderate_peak_day_range_days,
            "smooth_delta_consistent_days": settings.smooth_delta_consistent_days,
            "smooth_delta_moderate_days": settings.smooth_delta_moderate_days,
        },
    }
    _write_json(meta, out_cross / "run_meta.json")
    _write_json({"summary": meta}, out_dir / "summary.json")
    _write_summary_md(out_cross / "V9_PEAK_SELECTION_SENSITIVITY_A_SUMMARY.md", settings, baseline_audit, obj_summary, seq_summary)

    _log("[10/10] Done")


if __name__ == "__main__":
    run_peak_selection_sensitivity_v9_a(Path(r"D:\easm_project01\stage_partition\V9"))
