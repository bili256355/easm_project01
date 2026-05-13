"""
V10-a main-folder hotfix02 independent semantic rewrite of the V9 subpeak / peak extraction layer.

Purpose
-------
This module reproduces the V9 peak/subpeak extraction semantics without importing
or calling any V9 or V7 module.  It reimplements the required contracts directly:

1. fixed accepted-window registry used by V9: W045, W081, W113, W160;
2. object profile construction from smoothed_fields.npz;
3. V7-z/V9 detector input: year-mean climatological profile, feature-wise
   z-score along day;
4. ruptures.Window(width=20, model='l2', min_size=2, jump=1), predict(pen=4.0),
   then read algo.score and map local detector indices back to original day;
5. local peak extraction, candidate support band construction, paired-year
   bootstrap support, and original current-window candidate selector;
6. optional read-only regression audit against existing V9 CSV outputs.

Forbidden dependencies
----------------------
- No import from stage_partition_v9.
- No import from stage_partition_v7.
- No call to V9/V7 helper functions.
- Existing V9 CSVs may only be read as reference outputs for regression audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import math
import os
import time
import warnings

import numpy as np
import pandas as pd

VERSION = "v10_peak_subpeak_reproduce_a_hotfix02_main"
OUTPUT_TAG = "peak_subpeak_reproduce_v10_a"
EPS = 1.0e-12

DEFAULT_ACCEPTED_WINDOWS = ["W045", "W081", "W113", "W160"]
DEFAULT_ACCEPTED_WINDOWS_CSV = ",".join(DEFAULT_ACCEPTED_WINDOWS)
EXCLUDED_MAINLINE_WINDOWS = [
    {
        "window_id": "W135",
        "anchor_day": 135,
        "included_in_v10": False,
        "exclusion_reason": "not_in_strict_accepted_95pct_window_set; retained as exclusion record only",
    }
]

FIELD_ALIASES = {
    "precip": [
        "precip_smoothed", "precipitation_smoothed", "pr_smoothed", "P_smoothed",
        "precip", "precipitation", "pr", "P", "tp", "rain", "rainfall",
    ],
    "v850": ["v850_smoothed", "v850", "v_smoothed", "v", "V", "v850_anom"],
    "z500": ["z500_smoothed", "z500", "hgt500_smoothed", "hgt500", "H", "z"],
    "u200": ["u200_smoothed", "u200", "u_smoothed", "u", "U200", "u200_anom"],
    "lat": ["lat", "latitude", "lats", "nav_lat"],
    "lon": ["lon", "longitude", "lons", "nav_lon"],
    "years": ["years", "year", "yrs"],
}

OBJECT_ORDER_PAIRS = [
    ("H", "Je"), ("H", "Jw"), ("H", "P"), ("H", "V"),
    ("Je", "Jw"), ("Je", "P"), ("Je", "V"),
    ("Jw", "P"), ("Jw", "V"),
    ("P", "V"),
]


@dataclass(frozen=True)
class ObjectSpec:
    object_name: str
    field_role: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lat_step: float = 2.0


OBJECT_SPECS: List[ObjectSpec] = [
    ObjectSpec("P", "precip", 105, 125, 15, 39),
    ObjectSpec("V", "v850", 105, 125, 10, 30),
    ObjectSpec("H", "z500", 110, 140, 15, 35),
    ObjectSpec("Je", "u200", 120, 150, 25, 45),
    ObjectSpec("Jw", "u200", 80, 110, 25, 45),
]


@dataclass(frozen=True)
class AcceptedWindow:
    window_id: str
    anchor_day: int
    system_window_start: int
    system_window_end: int
    bootstrap_support: float
    accepted_status: str
    source_file: str


@dataclass(frozen=True)
class WindowScope:
    window_id: str
    anchor_day: int
    system_window_start: int
    system_window_end: int
    bootstrap_support: float
    accepted_status: str
    source_file: str
    detector_search_start: int
    detector_search_end: int
    detector_neighbor_buffer: int
    analysis_start: int
    analysis_end: int
    C0_pre_start: int
    C0_pre_end: int
    C0_post_start: int
    C0_post_end: int
    C1_pre_start: int
    C1_pre_end: int
    C1_post_start: int
    C1_post_end: int
    C2_pre_start: int
    C2_pre_end: int
    C2_post_start: int
    C2_post_end: int
    early_start: int
    early_end: int
    core_start: int
    core_end: int
    late_start: int
    late_end: int
    is_valid_for_prepost: bool
    invalid_reason: str = ""


@dataclass
class V10PeakConfig:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    season_start: int = 0
    season_end: int = 182
    detector_neighbor_buffer: int = 5
    baseline_buffer_days: int = 5
    immediate_pre_days: int = 10
    segment_pre_days: int = 10
    segment_late_extra_days: int = 5
    min_detector_search_days: int = 25
    detector_width: int = 20
    detector_min_size: int = 2
    peak_min_distance: int = 3
    max_peaks_per_object: int = 5
    band_max_half_width: int = 10
    band_min_half_width: int = 2
    band_score_ratio: float = 0.50
    band_floor_quantile: float = 0.35
    bootstrap_n: int = 1000
    random_seed: int = 42
    peak_match_days: int = 5
    low_dynamic_range_eps: float = 1.0e-10
    tau_sync_quantile_primary: float = 0.75
    tau_sync_quantile_low: float = 0.50
    tau_sync_quantile_high: float = 0.90
    window_mode: str = "list"  # all | list | w45
    target_windows: str = DEFAULT_ACCEPTED_WINDOWS_CSV
    smoothed_fields_path: Optional[str] = None
    log_every_bootstrap: int = 50

    @staticmethod
    def from_env() -> "V10PeakConfig":
        cfg = V10PeakConfig()
        if os.environ.get("V10_PEAK_SMOOTHED_FIELDS"):
            cfg.smoothed_fields_path = os.environ["V10_PEAK_SMOOTHED_FIELDS"]
        if os.environ.get("V10_PEAK_N_BOOTSTRAP"):
            cfg.bootstrap_n = int(os.environ["V10_PEAK_N_BOOTSTRAP"])
        if os.environ.get("V10_PEAK_DEBUG_N_BOOTSTRAP"):
            cfg.bootstrap_n = int(os.environ["V10_PEAK_DEBUG_N_BOOTSTRAP"])
        if os.environ.get("V10_PEAK_WINDOW_MODE"):
            cfg.window_mode = os.environ["V10_PEAK_WINDOW_MODE"].strip().lower()
        if os.environ.get("V10_PEAK_TARGET_WINDOWS"):
            cfg.target_windows = os.environ["V10_PEAK_TARGET_WINDOWS"].strip()
        if os.environ.get("V10_PEAK_LOG_EVERY_BOOTSTRAP"):
            cfg.log_every_bootstrap = int(os.environ["V10_PEAK_LOG_EVERY_BOOTSTRAP"])
        return cfg


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _nanmean(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _summarize_samples(samples: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": np.nan, "q025": np.nan, "q975": np.nan, "P_positive": np.nan, "P_negative": np.nan}
    return {
        "median": float(np.median(arr)),
        "q025": float(np.quantile(arr, 0.025)),
        "q975": float(np.quantile(arr, 0.975)),
        "P_positive": float(np.mean(arr > 0)),
        "P_negative": float(np.mean(arr < 0)),
    }


def _decision_from_samples(samples: Sequence[float], positive_name: str, negative_name: str) -> str:
    s = _summarize_samples(samples)
    q025, q975 = s["q025"], s["q975"]
    pp, pn = s["P_positive"], s["P_negative"]
    if np.isfinite(q025) and q025 > 0:
        return positive_name + "_supported"
    if np.isfinite(q975) and q975 < 0:
        return negative_name + "_supported"
    if np.isfinite(pp) and pp >= 0.80:
        return positive_name + "_tendency"
    if np.isfinite(pn) and pn >= 0.80:
        return negative_name + "_tendency"
    return "unresolved"


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> Tuple[int, float]:
    lo, hi = max(a0, b0), min(a1, b1)
    overlap = max(0, hi - lo + 1)
    denom = max(1, min(a1 - a0 + 1, b1 - b0 + 1))
    return int(overlap), float(overlap / denom)


def _stage_root_from_v10_root(v10_root: Path) -> Path:
    return v10_root.parent


def _project_root_from_v10_root(v10_root: Path) -> Path:
    return v10_root.parents[1]


def _default_smoothed_path(v10_root: Path) -> Path:
    return _project_root_from_v10_root(v10_root) / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _hardcoded_accepted_windows() -> List[AcceptedWindow]:
    source = "v10_independent_hardcoded_current_accepted_windows_replicating_v9_inputs"
    return [
        AcceptedWindow("W045", 45, 40, 48, 0.95, "hardcoded_accepted", source),
        AcceptedWindow("W081", 81, 75, 87, 0.95, "hardcoded_accepted", source),
        AcceptedWindow("W113", 113, 108, 118, 0.95, "hardcoded_accepted", source),
        AcceptedWindow("W160", 160, 155, 165, 0.95, "hardcoded_accepted", source),
    ]


def _build_window_scopes(wins: List[AcceptedWindow], cfg: V10PeakConfig) -> Tuple[List[WindowScope], pd.DataFrame]:
    scopes: List[WindowScope] = []
    validity_rows: List[dict] = []
    for i, w in enumerate(wins):
        prev_w = wins[i - 1] if i > 0 else None
        next_w = wins[i + 1] if i < len(wins) - 1 else None
        c0_pre_start = cfg.season_start if prev_w is None else prev_w.system_window_end + 1
        c0_pre_end = w.system_window_start - 1
        c0_post_start = w.system_window_end + 1
        c0_post_end = cfg.season_end if next_w is None else next_w.system_window_start - 1
        c1_pre_start = c0_pre_start if prev_w is None else c0_pre_start + cfg.baseline_buffer_days
        c1_pre_end = w.system_window_start - cfg.baseline_buffer_days - 1
        c1_post_start = w.system_window_end + cfg.baseline_buffer_days + 1
        c1_post_end = c0_post_end if next_w is None else c0_post_end - cfg.baseline_buffer_days
        c2_pre_end = w.system_window_start - cfg.baseline_buffer_days - 1
        c2_pre_start = max(c2_pre_end - cfg.immediate_pre_days + 1, c1_pre_start)
        c2_post_start, c2_post_end = c1_post_start, c1_post_end
        analysis_start, analysis_end = c0_pre_start, c0_post_end
        det_start = cfg.season_start if prev_w is None else prev_w.system_window_end + 1
        det_end = cfg.season_end if next_w is None else next_w.system_window_start - 1 - cfg.detector_neighbor_buffer
        early_start = max(analysis_start, w.system_window_start - cfg.segment_pre_days)
        early_end = w.system_window_start - 1
        core_start = w.system_window_start
        core_end = w.anchor_day
        late_start = w.anchor_day + 1
        late_end = min(analysis_end, w.system_window_end + cfg.segment_late_extra_days)
        invalid = []
        if det_end - det_start + 1 < cfg.min_detector_search_days:
            invalid.append("short_detector_search")
        if c0_pre_end - c0_pre_start + 1 < 8:
            invalid.append("short_C0_pre")
        if c0_post_end - c0_post_start + 1 < 8:
            invalid.append("short_C0_post")
        scope = WindowScope(
            window_id=w.window_id,
            anchor_day=w.anchor_day,
            system_window_start=w.system_window_start,
            system_window_end=w.system_window_end,
            bootstrap_support=w.bootstrap_support,
            accepted_status=w.accepted_status,
            source_file=w.source_file,
            detector_search_start=det_start,
            detector_search_end=det_end,
            detector_neighbor_buffer=cfg.detector_neighbor_buffer,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            C0_pre_start=c0_pre_start,
            C0_pre_end=c0_pre_end,
            C0_post_start=c0_post_start,
            C0_post_end=c0_post_end,
            C1_pre_start=c1_pre_start,
            C1_pre_end=c1_pre_end,
            C1_post_start=c1_post_start,
            C1_post_end=c1_post_end,
            C2_pre_start=c2_pre_start,
            C2_pre_end=c2_pre_end,
            C2_post_start=c2_post_start,
            C2_post_end=c2_post_end,
            early_start=early_start,
            early_end=early_end,
            core_start=core_start,
            core_end=core_end,
            late_start=late_start,
            late_end=late_end,
            is_valid_for_prepost=("short_C0_pre" not in invalid and "short_C0_post" not in invalid),
            invalid_reason=";".join(invalid),
        )
        scopes.append(scope)
        checks = [
            ("detector_search", det_start, det_end, cfg.min_detector_search_days),
            ("analysis_range", analysis_start, analysis_end, 20),
            ("C0_pre", c0_pre_start, c0_pre_end, 8),
            ("C0_post", c0_post_start, c0_post_end, 8),
            ("C1_pre", c1_pre_start, c1_pre_end, 5),
            ("C1_post", c1_post_start, c1_post_end, 5),
            ("C2_pre", c2_pre_start, c2_pre_end, 8),
            ("C2_post", c2_post_start, c2_post_end, 5),
            ("early", early_start, early_end, 5),
            ("core", core_start, core_end, 3),
            ("late", late_start, late_end, 5),
        ]
        for typ, s, e, mn in checks:
            length = e - s + 1
            validity_rows.append({
                "window_id": w.window_id,
                "scope_type": typ,
                "start_day": s,
                "end_day": e,
                "length": length,
                "min_required_length": mn,
                "is_valid": bool(length >= mn),
                "invalid_reason": "" if length >= mn else f"short_{typ}_{length}",
            })
    return scopes, pd.DataFrame(validity_rows)


def _filter_scopes_for_run(scopes: List[WindowScope], cfg: V10PeakConfig) -> Tuple[List[WindowScope], pd.DataFrame]:
    mode = (cfg.window_mode or "list").strip().lower()
    tokens = [x.strip() for x in (cfg.target_windows or "").split(",") if x.strip()]
    if mode == "all":
        rows = [{"window_id": s.window_id, "anchor_day": s.anchor_day, "run_selected": True, "reason": "V10_PEAK_WINDOW_MODE=all"} for s in scopes]
        return scopes, pd.DataFrame(rows)
    if mode == "w45" and not tokens:
        tokens = ["W045", "45"]
    selected: List[WindowScope] = []
    rows: List[dict] = []
    for s in scopes:
        id_norm = str(s.window_id).lower().replace("w", "").lstrip("0")
        match = False
        for tok in tokens:
            t = tok.lower().replace("w", "").lstrip("0")
            if str(s.window_id).lower() == tok.lower() or id_norm == t:
                match = True
            try:
                tv = int(round(float(tok.replace("W", "").replace("w", ""))))
                if tv == int(s.anchor_day) or s.system_window_start <= tv <= s.system_window_end:
                    match = True
            except Exception:
                pass
        rows.append({
            "window_id": s.window_id,
            "anchor_day": s.anchor_day,
            "system_window_start": s.system_window_start,
            "system_window_end": s.system_window_end,
            "run_selected": bool(match),
            "reason": f"mode={mode}; targets={cfg.target_windows}",
        })
        if match:
            selected.append(s)
    if not selected:
        raise RuntimeError(f"No V10 windows selected. mode={mode}, targets={cfg.target_windows}")
    return selected, pd.DataFrame(rows)


def _find_key(npz: np.lib.npyio.NpzFile, aliases: Sequence[str]) -> Optional[str]:
    keys_lower = {k.lower(): k for k in npz.files}
    for a in aliases:
        if a in npz.files:
            return a
        if a.lower() in keys_lower:
            return keys_lower[a.lower()]
    return None


def _load_npz_fields(path: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"smoothed fields not found: {path}")
    npz = np.load(path, allow_pickle=True)
    audit_rows: List[dict] = []
    out: Dict[str, np.ndarray] = {}
    for role, aliases in FIELD_ALIASES.items():
        key = _find_key(npz, aliases)
        audit_rows.append({"role": role, "resolved_key": key, "status": "found" if key else "missing"})
        if key:
            out[role] = np.asarray(npz[key])
    missing = [r["role"] for r in audit_rows if r["status"] == "missing" and r["role"] in ["lat", "lon", "u200", "v850", "z500", "precip"]]
    audit = pd.DataFrame(audit_rows)
    if missing:
        raise KeyError(f"missing required fields in {path}: {missing}; available keys={npz.files}")
    return out, audit


def _as_year_day_lat_lon(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, years: Optional[np.ndarray]) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    nlat, nlon = len(lat), len(lon)
    if arr.ndim == 4:
        if arr.shape[2] == nlat and arr.shape[3] == nlon:
            return arr
        if arr.shape[2] == nlon and arr.shape[3] == nlat:
            return np.transpose(arr, (0, 1, 3, 2))
        if arr.shape[2] == nlat and arr.shape[3] == nlon and years is not None and arr.shape[1] == len(years):
            return np.transpose(arr, (1, 0, 2, 3))
    if arr.ndim == 3:
        if arr.shape[1] == nlat and arr.shape[2] == nlon:
            nt = arr.shape[0]
            if years is not None and len(years) > 0 and nt % len(years) == 0:
                nd = nt // len(years)
                return arr.reshape(len(years), nd, nlat, nlon)
            return arr.reshape(1, nt, nlat, nlon)
        if arr.shape[1] == nlon and arr.shape[2] == nlat:
            arr2 = np.transpose(arr, (0, 2, 1))
            nt = arr2.shape[0]
            if years is not None and len(years) > 0 and nt % len(years) == 0:
                nd = nt // len(years)
                return arr2.reshape(len(years), nd, nlat, nlon)
            return arr2.reshape(1, nt, nlat, nlon)
    raise ValueError(f"Cannot infer field dimensions for array shape {arr.shape}, lat={nlat}, lon={nlon}")


def _target_lats(spec: ObjectSpec) -> np.ndarray:
    lo = min(spec.lat_min, spec.lat_max)
    hi = max(spec.lat_min, spec.lat_max)
    n = int(round((hi - lo) / spec.lat_step))
    vals = lo + np.arange(n + 1) * spec.lat_step
    vals[-1] = hi
    return vals.astype(float)


def _build_object_profile(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: ObjectSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    arr = np.asarray(field, dtype=float)
    lo_lat, hi_lat = min(spec.lat_min, spec.lat_max), max(spec.lat_min, spec.lat_max)
    lo_lon, hi_lon = min(spec.lon_min, spec.lon_max), max(spec.lon_min, spec.lon_max)
    lat_mask = (lat >= lo_lat) & (lat <= hi_lat)
    lon_mask = (lon >= lo_lon) & (lon <= hi_lon)
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points in {lo_lat}-{hi_lat} for {spec.object_name}")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points in {lo_lon}-{hi_lon} for {spec.object_name}")
    region = arr[:, :, lat_mask, :][:, :, :, lon_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        prof_native = np.nanmean(region, axis=-1)
    lat_native = lat[lat_mask]
    order = np.argsort(lat_native)
    lat_sorted = lat_native[order]
    prof_sorted = prof_native[:, :, order]
    target = _target_lats(spec)
    y, d, _ = prof_sorted.shape
    out = np.full((y, d, len(target)), np.nan, dtype=float)
    for iy in range(y):
        for iday in range(d):
            x = prof_sorted[iy, iday, :]
            m = np.isfinite(x) & np.isfinite(lat_sorted)
            if np.sum(m) >= 2:
                out[iy, iday, :] = np.interp(target, lat_sorted[m], x[m], left=np.nan, right=np.nan)
    weights = np.cos(np.deg2rad(target))
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return out, target, weights


def _zscore_features(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected day x feature matrix, got shape={x.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
    sd = np.asarray(sd, dtype=float)
    sd[~np.isfinite(sd) | (sd < EPS)] = 1.0
    return (x - mu) / sd


def _state_matrix_from_year_cube(year_cube: np.ndarray, sampled_year_indices: Optional[np.ndarray] = None) -> np.ndarray:
    cube = np.asarray(year_cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected year x day x feature cube, got shape={cube.shape}")
    if sampled_year_indices is not None:
        cube = cube[np.asarray(sampled_year_indices, dtype=int), :, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clim = np.nanmean(cube, axis=0)
    return _zscore_features(clim)


def _finite_day_subset_matrix(matrix: np.ndarray, start_day: int, end_day: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(matrix, dtype=float)
    n_days = X.shape[0]
    lo = max(0, int(start_day))
    hi = min(n_days - 1, int(end_day))
    if lo > hi:
        return X[0:0], np.asarray([], dtype=int)
    days = np.arange(lo, hi + 1, dtype=int)
    sub = X[days]
    valid = np.any(np.isfinite(sub), axis=1)
    return sub[valid], days[valid]


def _map_detector_score_index(profile_raw: pd.Series, day_index: np.ndarray) -> pd.Series:
    if profile_raw is None or profile_raw.empty:
        return pd.Series(dtype=float, name="detector_score")
    out: Dict[int, float] = {}
    n = len(day_index)
    for local_idx, val in profile_raw.items():
        try:
            li = int(local_idx)
        except Exception:
            continue
        if 0 <= li < n:
            out[int(day_index[li])] = float(val)
    return pd.Series(out, name="detector_score", dtype=float).sort_index()


def _import_ruptures():
    try:
        import ruptures as rpt  # type: ignore
        return rpt
    except Exception as exc:  # pragma: no cover
        raise ImportError("V10-a requires ruptures to reproduce V9 ruptures.Window scores.") from exc


def _run_ruptures_window_score(state_matrix: np.ndarray, cfg: V10PeakConfig, day_index: np.ndarray) -> pd.Series:
    rpt = _import_ruptures()
    signal = np.asarray(state_matrix, dtype=float)
    if signal.shape[0] < max(2 * int(cfg.detector_width), 3):
        return pd.Series(dtype=float, name="detector_score")
    algo = rpt.Window(width=int(cfg.detector_width), model="l2", min_size=int(cfg.detector_min_size), jump=1).fit(signal)
    try:
        _ = algo.predict(pen=4.0)
    except Exception:
        pass
    score = getattr(algo, "score", None)
    if score is None:
        return pd.Series(dtype=float, name="detector_score")
    arr = np.asarray(score, dtype=float).ravel()
    width_half = int(algo.width // 2)
    idx = np.arange(width_half, width_half + len(arr), dtype=int)
    return _map_detector_score_index(pd.Series(arr, index=idx, name="detector_score"), day_index)


def _extract_local_peaks(profile: pd.Series, min_distance_days: int) -> pd.DataFrame:
    cols = ["peak_id", "peak_day", "peak_score", "peak_prominence", "peak_rank"]
    try:
        from scipy.signal import find_peaks, peak_prominences  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("V10-a requires scipy.signal to reproduce V9 local peak extraction.") from exc
    if profile is None or profile.empty:
        return pd.DataFrame(columns=cols)
    s = profile.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=max(1, int(min_distance_days)))
    if peaks.size == 0:
        return pd.DataFrame(columns=cols)
    prominences, _, _ = peak_prominences(values, peaks)
    rows = []
    for pidx, prom in zip(peaks, prominences):
        rows.append({
            "peak_id": "LP000",
            "peak_day": int(s.index[int(pidx)]),
            "peak_score": float(values[int(pidx)]),
            "peak_prominence": float(prom),
        })
    df = pd.DataFrame(rows).sort_values(["peak_score", "peak_prominence", "peak_day"], ascending=[False, False, True]).reset_index(drop=True)
    df["peak_rank"] = np.arange(1, len(df) + 1, dtype=int)
    df["peak_id"] = [f"CP{i:03d}" for i in range(1, len(df) + 1)]
    return df[cols]


def _build_support_band(profile: pd.Series, peak_day: int, cfg: V10PeakConfig) -> Dict[str, object]:
    if profile is None or profile.empty or int(peak_day) not in profile.index:
        return {
            "band_start_day": int(peak_day),
            "band_end_day": int(peak_day),
            "support_floor": np.nan,
            "left_stop_reason": "missing_profile",
            "right_stop_reason": "missing_profile",
        }
    s = profile.sort_index().astype(float)
    peak_score = float(s.loc[int(peak_day)])
    finite = s[np.isfinite(s)]
    if finite.empty:
        floor = np.nan
    else:
        floor = float(max(np.nanquantile(finite.to_numpy(), cfg.band_floor_quantile), peak_score * cfg.band_score_ratio))
    min_lo = int(peak_day) - int(cfg.band_min_half_width)
    min_hi = int(peak_day) + int(cfg.band_min_half_width)
    lo = int(peak_day)
    while lo - 1 in s.index and (int(peak_day) - (lo - 1)) <= int(cfg.band_max_half_width):
        if lo - 1 <= min_lo or float(s.loc[lo - 1]) >= floor:
            lo -= 1
        else:
            break
    hi = int(peak_day)
    while hi + 1 in s.index and ((hi + 1) - int(peak_day)) <= int(cfg.band_max_half_width):
        if hi + 1 >= min_hi or float(s.loc[hi + 1]) >= floor:
            hi += 1
        else:
            break
    return {
        "band_start_day": int(lo),
        "band_end_day": int(hi),
        "support_floor": floor,
        "left_stop_reason": "floor_or_width",
        "right_stop_reason": "floor_or_width",
    }


def _run_detector_for_profile(X: np.ndarray, cfg: V10PeakConfig, scope: WindowScope, object_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub, days = _finite_day_subset_matrix(X, scope.detector_search_start, scope.detector_search_end)
    profile = _run_ruptures_window_score(sub, cfg, days)
    all_days = np.arange(max(0, scope.detector_search_start), min(X.shape[0] - 1, scope.detector_search_end) + 1, dtype=int)
    score_df = pd.DataFrame({"day": all_days})
    if not profile.empty:
        score_df = score_df.merge(profile.rename("detector_score").reset_index().rename(columns={"index": "day"}), on="day", how="left")
    else:
        score_df["detector_score"] = np.nan
    score_df["score_valid"] = np.isfinite(score_df["detector_score"].to_numpy(dtype=float))
    score_df.insert(0, "window_id", scope.window_id)
    score_df["object"] = object_name

    peaks = _extract_local_peaks(profile, cfg.peak_min_distance)
    rows = []
    for _, p in peaks.head(int(cfg.max_peaks_per_object)).iterrows():
        band = _build_support_band(profile, int(p["peak_day"]), cfg)
        ov, ov_frac = _interval_overlap(int(band["band_start_day"]), int(band["band_end_day"]), scope.system_window_start, scope.system_window_end)
        rows.append({
            "object": object_name,
            "candidate_id": p["peak_id"],
            "peak_day": int(p["peak_day"]),
            "band_start_day": int(band["band_start_day"]),
            "band_end_day": int(band["band_end_day"]),
            "peak_score": float(p["peak_score"]),
            "peak_prominence": float(p["peak_prominence"]),
            "peak_rank": int(p["peak_rank"]),
            "overlap_days_with_W45": ov,
            "overlap_fraction_with_W45": ov_frac,
            "left_stop_reason": band.get("left_stop_reason", ""),
            "right_stop_reason": band.get("right_stop_reason", ""),
        })
    return score_df, pd.DataFrame(rows)


def _window_support_class(support: float) -> str:
    if np.isfinite(support) and support >= 0.95:
        return "accepted_window"
    if np.isfinite(support) and support >= 0.80:
        return "candidate_window"
    if np.isfinite(support) and support >= 0.50:
        return "weak_window"
    return "unstable_window"


def _match_candidate_peak(bcand: pd.DataFrame, observed_day: int, radius: int) -> bool:
    if bcand is None or bcand.empty:
        return False
    return bool(np.any(np.abs(bcand["peak_day"].to_numpy(dtype=float) - int(observed_day)) <= int(radius)))


def _relation_to_system(peak_day: int, scope: WindowScope) -> str:
    if peak_day < scope.early_start:
        return "pre_window"
    if scope.early_start <= peak_day <= scope.early_end:
        return "front_or_early"
    if scope.system_window_start <= peak_day <= scope.system_window_end:
        return "within_system_window"
    if scope.late_start <= peak_day <= scope.late_end:
        return "late_or_catchup"
    if peak_day > scope.late_end:
        return "post_window"
    return "near_boundary"


def _is_system_relevant_candidate(row: pd.Series, scope: WindowScope) -> bool:
    peak = int(row["peak_day"])
    return bool(scope.early_start <= peak <= scope.late_end)


def _select_main_candidate(cand: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    base_cols = {
        "object": None,
        "selected_candidate_id": None,
        "selected_peak_day": np.nan,
        "selected_window_start": np.nan,
        "selected_window_end": np.nan,
        "selected_role": "unresolved",
        "support_class": "unavailable",
        "selection_reason": "no_candidates",
        "excluded_candidates": "",
        "early_secondary_candidates": "",
        "late_secondary_candidates": "",
    }
    if cand is None or cand.empty:
        return pd.DataFrame([base_cols])
    c = cand.copy()
    if "support_class" not in c.columns:
        c["support_class"] = "bootstrap_unscored"
    c["support_tier"] = c["support_class"].map({
        "accepted_window": 3,
        "candidate_window": 2,
        "weak_window": 1,
        "unstable_window": 0,
    }).fillna(0)
    if "candidate_id" not in c.columns:
        c["candidate_id"] = [f"CP{i+1:03d}" for i in range(len(c))]
    if "peak_score" not in c.columns:
        c["peak_score"] = np.nan
    if "band_start_day" not in c.columns:
        c["band_start_day"] = c["peak_day"]
    if "band_end_day" not in c.columns:
        c["band_end_day"] = c["peak_day"]
    if "object" not in c.columns:
        c["object"] = None

    c["relation"] = c["peak_day"].apply(lambda x: _relation_to_system(int(x), scope))
    c["system_relevant"] = c.apply(lambda r: _is_system_relevant_candidate(r, scope), axis=1)
    c["overlap_sys"] = c.apply(lambda r: _interval_overlap(int(r["band_start_day"]), int(r["band_end_day"]), scope.system_window_start, scope.system_window_end)[1], axis=1)
    c["distance_to_anchor"] = (c["peak_day"] - scope.anchor_day).abs()
    c["distance_to_system_band"] = c["peak_day"].apply(
        lambda d: 0 if scope.system_window_start <= int(d) <= scope.system_window_end else min(abs(int(d) - scope.system_window_start), abs(int(d) - scope.system_window_end))
    )
    main_pool = c[c["system_relevant"]].copy()
    reason = "system_relevant_first"
    if main_pool.empty:
        main_pool = c.copy()
        reason = "fallback_no_system_relevant_candidate"
    # V10-a hotfix02 main-tree correction: V9 scores the candidate/system-window overlap by overlap fraction, not raw overlap days.
    main_pool["selection_score"] = (
        main_pool["overlap_sys"] * 500
        + main_pool["support_tier"] * 100
        - main_pool["distance_to_anchor"] * 2
        - main_pool["distance_to_system_band"]
        + main_pool["peak_score"].rank(pct=True).fillna(0) * 10
    )
    row = main_pool.sort_values(["selection_score", "support_tier", "peak_score"], ascending=False).iloc[0]

    def _cand_label(r: pd.Series) -> str:
        return f"{r.get('candidate_id', '')}@{int(r['peak_day'])}:{r.get('support_class', 'unavailable')}"

    excluded = ";".join([_cand_label(r) for _, r in c.iterrows() if r.get("candidate_id") != row.get("candidate_id")])
    early = ";".join([_cand_label(r) for _, r in c.iterrows() if int(r["peak_day"]) < scope.early_start])
    late = ";".join([_cand_label(r) for _, r in c.iterrows() if int(r["peak_day"]) > scope.late_end])
    return pd.DataFrame([{
        "object": row.get("object", None),
        "selected_candidate_id": row.get("candidate_id", None),
        "selected_peak_day": int(row["peak_day"]),
        "selected_window_start": int(row["band_start_day"]),
        "selected_window_end": int(row["band_end_day"]),
        "selected_role": row["relation"],
        "support_class": row.get("support_class", "unavailable"),
        "selection_reason": reason,
        "excluded_candidates": excluded,
        "early_secondary_candidates": early,
        "late_secondary_candidates": late,
    }])


def _select_boot_candidate_day(cand: pd.DataFrame, scope: WindowScope) -> float:
    if cand is None or cand.empty:
        return float("nan")
    sel = _select_main_candidate(cand, scope)
    if sel.empty:
        return float("nan")
    return float(sel["selected_peak_day"].iloc[0])


def _make_bootstrap_indices(ny: int, scope: WindowScope, cfg: V10PeakConfig) -> List[np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed + int(scope.anchor_day))
    return [rng.integers(0, ny, size=ny) for _ in range(cfg.bootstrap_n)]


def _run_detector_and_bootstrap(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scope: WindowScope,
    cfg: V10PeakConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: List[pd.DataFrame] = []
    cand_rows: List[pd.DataFrame] = []
    boot_days_by_obj: Dict[str, List[float]] = {obj: [] for obj in profiles}
    boot_peak_rows: List[dict] = []

    for obj, (prof_by_year, _target_lat, _weights) in profiles.items():
        state = _state_matrix_from_year_cube(prof_by_year)
        scores, cand = _run_detector_for_profile(state, cfg, scope, obj)
        cand_rows.append(cand)
        score_rows.append(scores)
    cand_df = pd.concat(cand_rows, ignore_index=True) if cand_rows else pd.DataFrame()
    score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()

    ny = next(iter(profiles.values()))[0].shape[0]
    boot_indices = _make_bootstrap_indices(ny, scope, cfg)
    support_counts = {}
    if not cand_df.empty:
        for _, r in cand_df.iterrows():
            support_counts[(r["object"], r["candidate_id"])] = 0

    for ib, idx in enumerate(boot_indices):
        for obj, (prof_by_year, _target_lat, _weights) in profiles.items():
            state = _state_matrix_from_year_cube(prof_by_year, idx)
            _bscores, bcand = _run_detector_for_profile(state, cfg, scope, obj)
            if bcand.empty:
                selected_boot_day = np.nan
                boot_days_by_obj[obj].append(np.nan)
                boot_peak_rows.append({"window_id": scope.window_id, "bootstrap_id": ib, "object": obj, "selected_peak_day": np.nan})
                continue
            selected_boot_day = _select_boot_candidate_day(bcand, scope)
            boot_days_by_obj[obj].append(selected_boot_day)
            boot_peak_rows.append({"window_id": scope.window_id, "bootstrap_id": ib, "object": obj, "selected_peak_day": selected_boot_day})
            if not cand_df.empty:
                for _, r in cand_df[cand_df["object"] == obj].iterrows():
                    if _match_candidate_peak(bcand, int(r["peak_day"]), cfg.peak_match_days):
                        support_counts[(obj, r["candidate_id"])] = support_counts.get((obj, r["candidate_id"]), 0) + 1
        if cfg.log_every_bootstrap > 0 and (ib + 1) % cfg.log_every_bootstrap == 0:
            _log(f"  bootstrap detector {scope.window_id}: {ib + 1}/{cfg.bootstrap_n}")

    if not cand_df.empty:
        cand_df = cand_df.copy()
        cand_df["bootstrap_support"] = cand_df.apply(lambda r: support_counts.get((r["object"], r["candidate_id"]), 0) / max(1, cfg.bootstrap_n), axis=1)
        cand_df["support_class"] = cand_df["bootstrap_support"].apply(_window_support_class)
        cand_df["relation_to_system_window"] = cand_df["peak_day"].apply(lambda x: _relation_to_system(int(x), scope))
        cand_df.insert(0, "window_id", scope.window_id)

    selections = []
    for obj in profiles:
        sel = _select_main_candidate(cand_df[cand_df["object"] == obj] if not cand_df.empty else pd.DataFrame(), scope)
        sel.insert(0, "window_id", scope.window_id)
        selections.append(sel)
    selection_df = pd.concat(selections, ignore_index=True)

    rows = []
    objs = sorted(profiles.keys())
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            va = np.asarray(boot_days_by_obj[a], dtype=float)
            vb = np.asarray(boot_days_by_obj[b], dtype=float)
            delta = vb - va
            s = _summarize_samples(delta)
            obs_a = selection_df.loc[selection_df["object"] == a, "selected_peak_day"].iloc[0]
            obs_b = selection_df.loc[selection_df["object"] == b, "selected_peak_day"].iloc[0]
            obs_delta = float(obs_b - obs_a) if np.isfinite(obs_a) and np.isfinite(obs_b) else np.nan
            rows.append({
                "window_id": scope.window_id,
                "object_A": a,
                "object_B": b,
                "metric_family": "selected_raw_profile_peak_timing",
                "delta_definition": "B_peak_day - A_peak_day; positive means A earlier",
                "delta_observed": obs_delta,
                **{f"delta_{k}": v for k, v in s.items()},
                "decision": _decision_from_samples(delta, "A_earlier", "B_earlier"),
            })
    selected_delta_df = pd.DataFrame(rows)
    boot_peak_days_df = pd.DataFrame(boot_peak_rows)
    return score_df, cand_df, selection_df, selected_delta_df, boot_peak_days_df


def _estimate_timing_resolution(selection_df: pd.DataFrame, boot_peak_days_df: pd.DataFrame, cfg: V10PeakConfig, scope: WindowScope) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    pooled: List[float] = []
    if selection_df is None or selection_df.empty or boot_peak_days_df is None or boot_peak_days_df.empty:
        return pd.DataFrame(), pd.DataFrame([{"window_id": scope.window_id, "tau_sync_primary": np.nan, "tau_quality_flag": "insufficient_bootstrap"}])
    for _, r in selection_df.iterrows():
        obj = r["object"]
        obs = float(r["selected_peak_day"])
        vals = boot_peak_days_df[boot_peak_days_df["object"] == obj]["selected_peak_day"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        err = vals - obs if vals.size else np.array([], dtype=float)
        abs_err = np.abs(err)
        pooled.extend(abs_err.tolist())
        rows.append({
            "window_id": scope.window_id,
            "object": obj,
            "observed_peak_day": obs,
            "bootstrap_peak_median": float(np.median(vals)) if vals.size else np.nan,
            "bootstrap_peak_q025": float(np.quantile(vals, 0.025)) if vals.size else np.nan,
            "bootstrap_peak_q975": float(np.quantile(vals, 0.975)) if vals.size else np.nan,
            "abs_error_median": float(np.median(abs_err)) if abs_err.size else np.nan,
            "abs_error_q75": float(np.quantile(abs_err, 0.75)) if abs_err.size else np.nan,
            "abs_error_q90": float(np.quantile(abs_err, 0.90)) if abs_err.size else np.nan,
            "abs_error_q975": float(np.quantile(abs_err, 0.975)) if abs_err.size else np.nan,
            "support_class": r.get("support_class", ""),
            "included_in_tau_estimation": bool(abs_err.size > 0),
        })
    arr = np.asarray(pooled, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size:
        q50 = float(np.quantile(arr, cfg.tau_sync_quantile_low))
        q75 = float(np.quantile(arr, cfg.tau_sync_quantile_primary))
        q90 = float(np.quantile(arr, cfg.tau_sync_quantile_high))
        med_width = np.nanmedian(selection_df["selected_window_end"].to_numpy(float) - selection_df["selected_window_start"].to_numpy(float) + 1)
        flag = "broad_resolution_warning" if np.isfinite(med_width) and q75 > 0.5 * med_width else "normal"
        tau = pd.DataFrame([{
            "window_id": scope.window_id,
            "tau_sync_q50": q50,
            "tau_sync_q75": q75,
            "tau_sync_q90": q90,
            "tau_sync_primary": q75,
            "tau_source": "pooled_q75_abs_bootstrap_peak_error",
            "n_objects_used": int(selection_df.shape[0]),
            "n_bootstrap_values_used": int(arr.size),
            "tau_quality_flag": flag,
        }])
    else:
        tau = pd.DataFrame([{"window_id": scope.window_id, "tau_sync_primary": np.nan, "tau_quality_flag": "insufficient_bootstrap"}])
    return pd.DataFrame(rows), tau


def _pairwise_peak_order(selection_df: pd.DataFrame, boot_peak_days_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    by_sel = {r["object"]: r for _, r in selection_df.iterrows()} if selection_df is not None and not selection_df.empty else {}
    pivot = boot_peak_days_df.pivot(index="bootstrap_id", columns="object", values="selected_peak_day") if boot_peak_days_df is not None and not boot_peak_days_df.empty else pd.DataFrame()
    for a, b in OBJECT_ORDER_PAIRS:
        if a not in by_sel or b not in by_sel or a not in pivot.columns or b not in pivot.columns:
            continue
        obs_a = float(by_sel[a]["selected_peak_day"])
        obs_b = float(by_sel[b]["selected_peak_day"])
        delta = pivot[b].to_numpy(dtype=float) - pivot[a].to_numpy(dtype=float)
        s = _summarize_samples(delta)
        valid = delta[np.isfinite(delta)]
        p_same = float(np.mean(valid == 0)) if valid.size else np.nan
        if np.isfinite(s["q025"]) and s["q025"] > 0:
            dec = "A_peak_earlier_supported"
        elif np.isfinite(s["q975"]) and s["q975"] < 0:
            dec = "B_peak_earlier_supported"
        elif np.isfinite(s["median"]) and s["median"] > 0 and s["P_positive"] > s["P_negative"]:
            dec = "A_peak_earlier_tendency"
        elif np.isfinite(s["median"]) and s["median"] < 0 and s["P_negative"] > s["P_positive"]:
            dec = "B_peak_earlier_tendency"
        else:
            dec = "peak_order_unresolved"
        rows.append({
            "window_id": scope.window_id,
            "object_A": a,
            "object_B": b,
            "A_peak_day": obs_a,
            "B_peak_day": obs_b,
            "delta_observed": obs_b - obs_a,
            "delta_median": s["median"],
            "delta_q025": s["q025"],
            "delta_q975": s["q975"],
            "P_A_earlier": s["P_positive"],
            "P_B_earlier": s["P_negative"],
            "P_same_day": p_same,
            "peak_order_decision": dec,
        })
    return pd.DataFrame(rows)


def _pairwise_synchrony(peak_df: pd.DataFrame, boot_peak_days_df: pd.DataFrame, tau_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    tau_row = tau_df.iloc[0] if tau_df is not None and not tau_df.empty else pd.Series(dtype=float)
    tau50 = float(tau_row.get("tau_sync_q50", np.nan))
    tau75 = float(tau_row.get("tau_sync_q75", tau_row.get("tau_sync_primary", np.nan)))
    tau90 = float(tau_row.get("tau_sync_q90", np.nan))
    tau_primary = float(tau_row.get("tau_sync_primary", tau75))
    tau_flag = str(tau_row.get("tau_quality_flag", ""))
    pivot = boot_peak_days_df.pivot(index="bootstrap_id", columns="object", values="selected_peak_day") if boot_peak_days_df is not None and not boot_peak_days_df.empty else pd.DataFrame()
    for _, r in peak_df.iterrows():
        a, b = r["object_A"], r["object_B"]
        if a not in pivot.columns or b not in pivot.columns:
            continue
        delta = pivot[b].to_numpy(dtype=float) - pivot[a].to_numpy(dtype=float)
        valid = delta[np.isfinite(delta)]

        def p_within(tau: float) -> float:
            return float(np.mean(np.abs(valid) <= tau)) if valid.size and np.isfinite(tau) else np.nan

        q025 = float(r["delta_q025"])
        q975 = float(r["delta_q975"])
        p_primary = p_within(tau_primary)
        if np.isfinite(tau_primary) and np.isfinite(q025) and q025 >= -tau_primary and q975 <= tau_primary and tau_flag != "broad_resolution_warning":
            dec = "synchrony_supported"
        elif np.isfinite(p_primary) and p_primary >= 0.80:
            dec = "synchrony_tendency"
        elif np.isfinite(tau_primary) and ((np.isfinite(q025) and q025 > tau_primary) or (np.isfinite(q975) and q975 < -tau_primary)):
            dec = "synchrony_not_supported"
        else:
            dec = "synchrony_indeterminate"
        rows.append({
            "window_id": scope.window_id,
            "object_A": a,
            "object_B": b,
            "tau_sync_primary": tau_primary,
            "tau_sync_q50": tau50,
            "tau_sync_q75": tau75,
            "tau_sync_q90": tau90,
            "delta_observed": r["delta_observed"],
            "delta_median": r["delta_median"],
            "delta_q025": q025,
            "delta_q975": q975,
            "P_within_tau_q50": p_within(tau50),
            "P_within_tau_q75": p_within(tau75),
            "P_within_tau_q90": p_within(tau90),
            "synchrony_decision": dec,
            "tau_quality_flag": tau_flag,
        })
    return pd.DataFrame(rows)


def _pairwise_window_overlap(selection_df: pd.DataFrame, scope: WindowScope) -> pd.DataFrame:
    rows: List[dict] = []
    by_sel = {r["object"]: r for _, r in selection_df.iterrows()} if selection_df is not None and not selection_df.empty else {}
    for a, b in OBJECT_ORDER_PAIRS:
        if a not in by_sel or b not in by_sel:
            continue
        ra, rb = by_sel[a], by_sel[b]
        a0, a1 = int(ra["selected_window_start"]), int(ra["selected_window_end"])
        b0, b1 = int(rb["selected_window_start"]), int(rb["selected_window_end"])
        lo, hi = max(a0, b0), min(a1, b1)
        overlap_days = max(0, hi - lo + 1)
        union_days = max(a1, b1) - min(a0, b0) + 1
        frac = overlap_days / max(1, union_days)
        if overlap_days <= 0:
            dec = "window_separated"
        elif frac >= 0.60:
            dec = "window_overlap_strong"
        elif frac >= 0.25:
            dec = "window_overlap_partial"
        else:
            dec = "window_overlap_weak"
        rows.append({
            "window_id": scope.window_id,
            "object_A": a,
            "object_B": b,
            "A_window_start": a0,
            "A_window_end": a1,
            "B_window_start": b0,
            "B_window_end": b1,
            "overlap_start": lo if overlap_days > 0 else np.nan,
            "overlap_end": hi if overlap_days > 0 else np.nan,
            "overlap_days": overlap_days,
            "union_days": union_days,
            "overlap_fraction": frac,
            "A_peak_inside_B_window": bool(b0 <= int(ra["selected_peak_day"]) <= b1),
            "B_peak_inside_A_window": bool(a0 <= int(rb["selected_peak_day"]) <= a1),
            "both_peaks_inside_system_window": bool(scope.system_window_start <= int(ra["selected_peak_day"]) <= scope.system_window_end and scope.system_window_start <= int(rb["selected_peak_day"]) <= scope.system_window_end),
            "window_overlap_decision": dec,
        })
    return pd.DataFrame(rows)


def _window_day_audit_for_profile(prof: np.ndarray, scope: WindowScope, object_name: str) -> List[dict]:
    arr = np.asarray(prof)
    if arr.ndim < 2:
        day_finite = np.isfinite(arr)
    else:
        axes = tuple(i for i in range(arr.ndim) if i != 1)
        day_finite = np.any(np.isfinite(arr), axis=axes)
    rows: List[dict] = []
    for domain_name, start, end in [
        ("analysis", int(scope.analysis_start), int(scope.analysis_end)),
        ("detector", int(scope.detector_search_start), int(scope.detector_search_end)),
        ("system_window", int(scope.system_window_start), int(scope.system_window_end)),
    ]:
        s = max(start, 0)
        e = min(end, int(day_finite.shape[0]) - 1)
        if e < s:
            rows.append({
                "window_id": scope.window_id,
                "object": object_name,
                "domain": domain_name,
                "nominal_start_day": start,
                "nominal_end_day": end,
                "finite_start_day": np.nan,
                "finite_end_day": np.nan,
                "n_nominal_days": max(0, end - start + 1),
                "n_finite_days": 0,
                "leading_nan_days": np.nan,
                "trailing_nan_days": np.nan,
                "internal_nan_days": np.nan,
                "valid_day_fraction": 0.0,
                "boundary_nan_warning": "invalid_nominal_domain",
            })
            continue
        sub = np.asarray(day_finite[s:e + 1], dtype=bool)
        finite_idx = np.where(sub)[0]
        n_nominal = int(e - s + 1)
        if finite_idx.size == 0:
            rows.append({
                "window_id": scope.window_id,
                "object": object_name,
                "domain": domain_name,
                "nominal_start_day": start,
                "nominal_end_day": end,
                "finite_start_day": np.nan,
                "finite_end_day": np.nan,
                "n_nominal_days": n_nominal,
                "n_finite_days": 0,
                "leading_nan_days": n_nominal,
                "trailing_nan_days": n_nominal,
                "internal_nan_days": 0,
                "valid_day_fraction": 0.0,
                "boundary_nan_warning": "all_nan_domain",
            })
            continue
        finite_start = s + int(finite_idx[0])
        finite_end = s + int(finite_idx[-1])
        leading_nan = int(finite_idx[0])
        trailing_nan = int((n_nominal - 1) - finite_idx[-1])
        internal = sub[finite_idx[0]:finite_idx[-1] + 1]
        internal_nan = int((~internal).sum())
        warning = "none"
        if leading_nan or trailing_nan:
            warning = "boundary_nan_present"
        if internal_nan:
            warning = "internal_nan_present" if warning == "none" else warning + "+internal_nan_present"
        rows.append({
            "window_id": scope.window_id,
            "object": object_name,
            "domain": domain_name,
            "nominal_start_day": start,
            "nominal_end_day": end,
            "finite_start_day": finite_start,
            "finite_end_day": finite_end,
            "n_nominal_days": n_nominal,
            "n_finite_days": int(sub.sum()),
            "leading_nan_days": leading_nan,
            "trailing_nan_days": trailing_nan,
            "internal_nan_days": internal_nan,
            "valid_day_fraction": float(sub.mean()),
            "boundary_nan_warning": warning,
        })
    return rows


def _compare_numeric(a, b, tol: float = 1.0e-9) -> str:
    try:
        aa = float(a)
        bb = float(b)
    except Exception:
        return "same" if str(a) == str(b) else "different"
    if np.isnan(aa) and np.isnan(bb):
        return "same"
    return "same" if abs(aa - bb) <= tol else "different"



def _read_v9_reference_table_for_regression(stage_root: Path, logical: str, fname: str) -> Tuple[Optional[pd.DataFrame], str, str]:
    """Read V9 reference CSVs without importing/calling V9 modules.

    V10-a hotfix02 main-tree correction: the V9 cross-window raw-profile score table may lack
    ``window_id``.  For that logical table, prefer V9 per-window raw score CSVs
    and inject the window id from the folder/name, so the regression key can be
    ``window_id, object, day``.  Other logical tables continue to use the V9
    cross-window CSVs.
    """
    v9_base = stage_root / "V9" / "outputs" / "peak_all_windows_v9_a"
    v9_cross = v9_base / "cross_window"
    cross_ref = v9_cross / fname

    if logical != "raw_profile_detector_scores":
        if not cross_ref.exists():
            return None, str(cross_ref), "missing_cross_window_reference"
        return pd.read_csv(cross_ref), str(cross_ref), "cross_window_reference"

    # raw score: cross-window file may not carry window_id in V9; use per-window files.
    per_root = v9_base / "per_window"
    frames: List[pd.DataFrame] = []
    used_files: List[str] = []
    for wid in DEFAULT_ACCEPTED_WINDOWS:
        f = per_root / wid / f"raw_profile_detector_scores_{wid}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "window_id" not in df.columns:
            df.insert(0, "window_id", wid)
        else:
            df["window_id"] = df["window_id"].fillna(wid)
        frames.append(df)
        used_files.append(str(f))
    if frames:
        return pd.concat(frames, ignore_index=True), ";".join(used_files), "per_window_reference_with_inferred_window_id"

    # Fallback: read the cross-window file if it exists; the caller will still flag
    # missing keys if window_id is unavailable.
    if cross_ref.exists():
        return pd.read_csv(cross_ref), str(cross_ref), "cross_window_reference_fallback"
    return None, str(cross_ref), "missing_raw_reference"

def _regression_audit_against_v9(stage_root: Path, out_audit: Path, concat: Dict[str, pd.DataFrame]) -> None:
    """Read-only V10-vs-V9 regression audit.

    All regression audit tables are written inside the V10-a hotfix02 main-tree output
    subtree.  This function reads existing V9 CSVs only as reference output
    files and never imports/calls V9 or V7 modules.
    """
    out_audit.mkdir(parents=True, exist_ok=True)
    # Main-tree hotfix02: avoid stale diff details from earlier runs.  If a table has
    # no differences after this run, old diff CSVs must not remain and mislead audit.
    for old in out_audit.glob("v10_vs_v9_*_diff_detail.csv"):
        try:
            old.unlink()
        except FileNotFoundError:
            pass
    for old_name in ["bootstrap_selected_peak_days_diff_summary_by_window_object.csv", "v10_vs_v9_all_diff_detail.csv"]:
        old = out_audit / old_name
        if old.exists():
            old.unlink()
    rows: List[dict] = []
    all_detail_frames: List[pd.DataFrame] = []
    pairs = [
        ("main_window_selection", "main_window_selection_all_windows.csv", ["window_id", "object"], ["selected_candidate_id", "selected_peak_day", "selected_window_start", "selected_window_end", "selected_role", "support_class", "selection_reason"]),
        ("object_profile_window_registry", "object_profile_window_registry_all_windows.csv", ["window_id", "object", "candidate_id"], ["peak_day", "band_start_day", "band_end_day", "peak_score", "peak_prominence", "peak_rank", "bootstrap_support", "support_class", "relation_to_system_window"]),
        ("raw_profile_detector_scores", "raw_profile_detector_scores_all_windows.csv", ["window_id", "object", "day"], ["detector_score", "score_valid"]),
        ("bootstrap_selected_peak_days", "bootstrap_selected_peak_days_all_windows.csv", ["window_id", "bootstrap_id", "object"], ["selected_peak_day"]),
    ]
    for logical, fname, keys, compare_cols in pairs:
        v10_df = concat.get(logical, pd.DataFrame())
        try:
            v9_df, ref_file, reference_source = _read_v9_reference_table_for_regression(stage_root, logical, fname)
        except Exception as exc:
            rows.append({"table": logical, "status": "v9_reference_read_error", "error": str(exc), "v9_file": fname})
            continue
        if v9_df is None:
            rows.append({"table": logical, "status": "v9_reference_missing", "v9_file": ref_file, "n_v10_rows": int(len(v10_df)), "reference_source": reference_source})
            continue
        if v10_df is None or v10_df.empty:
            rows.append({"table": logical, "status": "v10_table_empty", "n_v9_rows": int(len(v9_df)), "v9_file": ref_file, "reference_source": reference_source})
            continue
        common_keys = [c for c in keys if c in v10_df.columns and c in v9_df.columns]
        if len(common_keys) != len(keys):
            rows.append({"table": logical, "status": "missing_key_columns", "keys_expected": ",".join(keys), "keys_found": ",".join(common_keys), "v9_file": ref_file, "reference_source": reference_source})
            continue
        merged = v10_df.merge(v9_df, on=common_keys, how="outer", suffixes=("_v10", "_v9"), indicator=True)
        key_mismatch = int((merged["_merge"] != "both").sum())
        diff_count = 0
        col_diffs: Dict[str, int] = {}
        both = merged[merged["_merge"] == "both"].copy()
        table_detail_frames: List[pd.DataFrame] = []
        for col in compare_cols:
            c10, c9 = f"{col}_v10", f"{col}_v9"
            if c10 not in both.columns or c9 not in both.columns:
                continue
            mask = []
            for _, r in both.iterrows():
                mask.append(_compare_numeric(r[c10], r[c9]) != "same")
            mask_arr = np.asarray(mask, dtype=bool)
            d = int(mask_arr.sum())
            if d:
                col_diffs[col] = d
                diff_count += d
                cols = list(keys) + [c10, c9]
                det = both.loc[mask_arr, cols].copy()
                det.insert(0, "table", logical)
                det.insert(len(keys) + 1, "column", col)
                if col == "selected_peak_day":
                    det["delta_day"] = pd.to_numeric(det[c10], errors="coerce") - pd.to_numeric(det[c9], errors="coerce")
                table_detail_frames.append(det)
        if key_mismatch:
            km = merged[merged["_merge"] != "both"].copy()
            keep = [c for c in list(keys) + ["_merge"] if c in km.columns]
            km = km[keep]
            km.insert(0, "table", logical)
            km.insert(len(keys) + 1 if len(keys) + 1 <= len(km.columns) else len(km.columns), "column", "KEY_MISMATCH")
            table_detail_frames.append(km)
        if table_detail_frames:
            detail = pd.concat(table_detail_frames, ignore_index=True)
            all_detail_frames.append(detail)
            detail_name = f"v10_vs_v9_{logical}_diff_detail.csv"
            _safe_to_csv(detail, out_audit / detail_name)
            if logical == "bootstrap_selected_peak_days" and "delta_day" in detail.columns:
                summary = (
                    detail.groupby(["window_id", "object"], dropna=False)
                    .agg(
                        n_diff=("delta_day", "size"),
                        median_delta_day=("delta_day", "median"),
                        min_delta_day=("delta_day", "min"),
                        max_delta_day=("delta_day", "max"),
                    )
                    .reset_index()
                    .sort_values(["window_id", "object"])
                )
                _safe_to_csv(summary, out_audit / "bootstrap_selected_peak_days_diff_summary_by_window_object.csv")
        status = "pass" if key_mismatch == 0 and diff_count == 0 else "difference_found"
        rows.append({
            "table": logical,
            "status": status,
            "n_v10_rows": int(len(v10_df)),
            "n_v9_rows": int(len(v9_df)),
            "n_key_mismatch_rows": key_mismatch,
            "n_value_differences": diff_count,
            "columns_with_differences": json.dumps(col_diffs, ensure_ascii=False),
            "v9_file": ref_file,
            "audit_detail_file": f"v10_vs_v9_{logical}_diff_detail.csv" if table_detail_frames else "",
            "audit_boundary": "read-only CSV comparison; no V9 interface called",
            "reference_source": reference_source,
        })
    if all_detail_frames:
        _safe_to_csv(pd.concat(all_detail_frames, ignore_index=True), out_audit / "v10_vs_v9_all_diff_detail.csv")
    else:
        _safe_to_csv(pd.DataFrame(columns=["table", "column", "note"]), out_audit / "v10_vs_v9_all_diff_detail.csv")
        _safe_to_csv(pd.DataFrame(columns=["window_id", "object", "n_diff", "median_delta_day", "min_delta_day", "max_delta_day"]), out_audit / "bootstrap_selected_peak_days_diff_summary_by_window_object.csv")
    _safe_to_csv(pd.DataFrame(rows), out_audit / "v10_vs_v9_subpeak_regression_audit.csv")
    if all_detail_frames:
        _safe_to_csv(pd.concat(all_detail_frames, ignore_index=True), out_audit / "v10_vs_v9_all_diff_detail.csv")

def _write_summary(path: Path, run_scopes: List[WindowScope], cfg: V10PeakConfig) -> None:
    lines = [
        "# V10-a hotfix02 main-treed independent subpeak/peak reproduction summary",
        "",
        f"version: `{VERSION}`",
        f"output_tag: `{OUTPUT_TAG}`",
        "",
        "## Purpose",
        "V10-a hotfix02 main-treed audit independently reimplements the V9 subpeak/peak extraction layer.",
        "It does not import or call V9 or V7 code. Existing V9 CSVs are read only for regression audit.",
        "",
        "## Windows processed",
    ]
    for s in run_scopes:
        lines.append(f"- {s.window_id}: anchor={s.anchor_day}, system={s.system_window_start}-{s.system_window_end}, detector={s.detector_search_start}-{s.detector_search_end}")
    lines += [
        "",
        "## Method boundary",
        "- This is a reproduction/extraction layer, not a new sensitivity audit.",
        "- This is not a physical subpeak classification layer.",
        "- This does not redefine accepted windows.",
        "- It should first be judged by regression against V9 outputs.",
        "",
        "## Expected core outputs",
        "- object_profile_window_registry_all_windows.csv: all detected candidate subpeaks.",
        "- main_window_selection_all_windows.csv: V9-equivalent selected main candidate per object/window.",
        "- raw_profile_detector_scores_all_windows.csv: detector score landscape.",
        "- bootstrap_selected_peak_days_all_windows.csv: paired-year bootstrap selected days.",
        "- audit/v10_vs_v9_subpeak_regression_audit.csv: read-only comparison against V9 CSVs when available, with diff details stored under the same bundle.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_peak_subpeak_reproduce_v10_a(v10_root: Path | str) -> None:
    v10_root = Path(v10_root)
    stage_root = _stage_root_from_v10_root(v10_root)
    cfg = V10PeakConfig.from_env()
    # Main-folder implementation: code lives under V10/src and outputs/logs are written
    # to the normal V10 main output layout, not to the temporary audit bundle folder.
    out_root = _ensure_dir(v10_root / "outputs" / OUTPUT_TAG)
    out_cross = _ensure_dir(out_root / "cross_window")
    out_per = _ensure_dir(out_root / "per_window")
    out_audit = _ensure_dir(out_root / "audit")
    log_dir = _ensure_dir(v10_root / "logs" / OUTPUT_TAG)
    t0 = time.time()

    _log("[1/7] Build V10 independent fixed accepted-window scopes")
    wins = _hardcoded_accepted_windows()
    scopes, validity = _build_window_scopes(wins, cfg)
    run_scopes, run_scope_audit = _filter_scopes_for_run(scopes, cfg)
    _safe_to_csv(pd.DataFrame([asdict(w) for w in wins]), out_cross / "accepted_windows_used_v10_a.csv")
    _safe_to_csv(pd.DataFrame([asdict(s) for s in scopes]), out_cross / "window_scope_registry_v10_a.csv")
    _safe_to_csv(validity, out_cross / "window_scope_validity_audit_v10_a.csv")
    _safe_to_csv(run_scope_audit, out_cross / "run_window_selection_audit_v10_a.csv")
    _safe_to_csv(pd.DataFrame(EXCLUDED_MAINLINE_WINDOWS), out_cross / "excluded_mainline_windows_v10_a.csv")

    _log("[2/7] Load smoothed fields with V10 independent NPZ resolver")
    smoothed = Path(cfg.smoothed_fields_path) if cfg.smoothed_fields_path else _default_smoothed_path(v10_root)
    if not smoothed.exists():
        raise FileNotFoundError(
            f"smoothed_fields.npz not found: {smoothed}. Set V10_PEAK_SMOOTHED_FIELDS to the correct file."
        )
    fields, input_audit = _load_npz_fields(smoothed)
    lat = fields["lat"]
    lon = fields["lon"]
    years = fields.get("years")
    _safe_to_csv(input_audit, out_cross / "input_key_audit_v10_peak_subpeak_reproduce_a.csv")

    _write_json({
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
        "smoothed_fields": str(smoothed),
        "implementation_boundary": {
            "imports_stage_partition_v9": False,
            "imports_stage_partition_v7": False,
            "calls_v9_or_v7_interfaces": False,
            "reads_v9_csv_for_regression_only": True,
            "does_not_redefine_accepted_windows": True,
            "does_not_perform_sensitivity_or_physical_classification": True,
            "implemented_in_main_v10_tree": True,
            "temporary_audit_bundle_required": False,
        },
    }, out_root / "run_meta.json")

    _log("[3/7] Build object profiles with V10 independent profile constructor")
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    object_rows: List[dict] = []
    for spec in OBJECT_SPECS:
        arr = _as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        prof, target_lat, weights = _build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (prof, target_lat, weights)
        object_rows.append({
            **asdict(spec),
            "profile_shape": str(prof.shape),
            "target_lat_min": float(np.nanmin(target_lat)),
            "target_lat_max": float(np.nanmax(target_lat)),
            "v10_role": "peak_detector_profile_input",
        })
    _safe_to_csv(pd.DataFrame(object_rows), out_cross / "object_registry_v10_peak_subpeak_reproduce_a.csv")

    _log("[4/7] Run V10 independent detector/bootstrap reproduction")
    all_parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in [
        "raw_profile_detector_scores", "object_profile_window_registry", "main_window_selection",
        "selected_peak_delta", "bootstrap_selected_peak_days", "timing_resolution_audit",
        "tau_sync_estimate", "pairwise_peak_order_test", "pairwise_synchrony_equivalence_test",
        "pairwise_window_overlap", "peak_valid_day_audit",
    ]}

    for idx, scope in enumerate(run_scopes, start=1):
        _log(f"  [{idx}/{len(run_scopes)}] {scope.window_id}")
        out_win = _ensure_dir(out_per / scope.window_id)
        _safe_to_csv(pd.DataFrame([asdict(scope)]), out_win / f"window_scope_{scope.window_id}.csv")

        valid_rows: List[dict] = []
        for obj, (prof, _lat, _w) in profiles.items():
            valid_rows.extend(_window_day_audit_for_profile(prof, scope, obj))
        valid_audit_df = pd.DataFrame(valid_rows)

        score_df, cand_df, selection_df, selected_delta_df, boot_peak_days_df = _run_detector_and_bootstrap(profiles, scope, cfg)
        timing_audit_df, tau_df = _estimate_timing_resolution(selection_df, boot_peak_days_df, cfg, scope)
        peak_order_df = _pairwise_peak_order(selection_df, boot_peak_days_df, scope)
        sync_df = _pairwise_synchrony(peak_order_df, boot_peak_days_df, tau_df, scope)
        overlap_df = _pairwise_window_overlap(selection_df, scope)
        result_map = {
            "raw_profile_detector_scores": score_df,
            "object_profile_window_registry": cand_df,
            "main_window_selection": selection_df,
            "selected_peak_delta": selected_delta_df,
            "bootstrap_selected_peak_days": boot_peak_days_df,
            "timing_resolution_audit": timing_audit_df,
            "tau_sync_estimate": tau_df,
            "pairwise_peak_order_test": peak_order_df,
            "pairwise_synchrony_equivalence_test": sync_df,
            "pairwise_window_overlap": overlap_df,
            "peak_valid_day_audit": valid_audit_df,
        }
        for logical, df in result_map.items():
            fname = f"{logical}_{scope.window_id}.csv"
            _safe_to_csv(df, out_win / fname)
            if df is not None and not df.empty:
                all_parts[logical].append(df)
        _safe_to_csv(selection_df, out_win / f"object_peak_registry_{scope.window_id}.csv")
        _safe_to_csv(cand_df, out_win / f"object_subpeak_candidate_registry_{scope.window_id}.csv")
        _safe_to_csv(score_df, out_win / f"raw_profile_detector_scores_{scope.window_id}.csv")
        _write_json({
            "version": VERSION,
            "window_id": scope.window_id,
            "bootstrap_n": int(cfg.bootstrap_n),
            "scope": asdict(scope),
            "boundary": "independent reproduction of V9 subpeak/peak extraction; no V9/V7 interface call",
        }, out_win / f"window_run_meta_{scope.window_id}.json")

    _log("[5/7] Write cross-window V10 reproduction outputs")
    concat = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in all_parts.items()}
    cross_names = {
        "raw_profile_detector_scores": "raw_profile_detector_scores_all_windows.csv",
        "object_profile_window_registry": "object_profile_window_registry_all_windows.csv",
        "main_window_selection": "main_window_selection_all_windows.csv",
        "selected_peak_delta": "selected_peak_delta_all_windows.csv",
        "bootstrap_selected_peak_days": "bootstrap_selected_peak_days_all_windows.csv",
        "timing_resolution_audit": "timing_resolution_audit_all_windows.csv",
        "tau_sync_estimate": "tau_sync_estimate_all_windows.csv",
        "pairwise_peak_order_test": "pairwise_peak_order_test_all_windows.csv",
        "pairwise_synchrony_equivalence_test": "pairwise_synchrony_equivalence_test_all_windows.csv",
        "pairwise_window_overlap": "pairwise_window_overlap_all_windows.csv",
        "peak_valid_day_audit": "peak_valid_day_audit_all_windows.csv",
    }
    for logical, filename in cross_names.items():
        _safe_to_csv(concat[logical], out_cross / filename)
    _safe_to_csv(concat["object_profile_window_registry"], out_cross / "cross_window_subpeak_candidate_registry.csv")
    _safe_to_csv(concat["main_window_selection"], out_cross / "cross_window_object_peak_registry.csv")
    _safe_to_csv(concat["pairwise_peak_order_test"], out_cross / "cross_window_pairwise_peak_order.csv")
    _safe_to_csv(concat["pairwise_synchrony_equivalence_test"], out_cross / "cross_window_pairwise_peak_synchrony.csv")

    _log("[6/7] Write read-only regression audit against existing V9 CSVs")
    _regression_audit_against_v9(stage_root, out_audit, concat)
    audit_main = out_audit / "v10_vs_v9_subpeak_regression_audit.csv"
    if audit_main.exists():
        # Keep the legacy V10-a location available for users who first inspect cross_window.
        pd.read_csv(audit_main).to_csv(out_cross / "v10_vs_v9_subpeak_regression_audit.csv", index=False)
    _write_summary(out_cross / "V10_PEAK_SUBPEAK_REPRODUCE_A_SUMMARY.md", run_scopes, cfg)

    manifest_rows = []
    for p in sorted(out_root.rglob("*")):
        if p.is_file():
            manifest_rows.append({
                "relative_path": str(p.relative_to(out_root)).replace("\\", "/"),
                "size_bytes": int(p.stat().st_size),
                "main_tree_role": "generated_v10_a_hotfix02_main_output",
            })
    _safe_to_csv(pd.DataFrame(manifest_rows), out_root / "manifest_v10_peak_subpeak_reproduce_a.csv")

    _log("[7/7] Done")
    elapsed = time.time() - t0
    _write_json({
        "version": VERSION,
        "elapsed_seconds": elapsed,
        "n_windows_processed": len(run_scopes),
        "windows_processed": [s.window_id for s in run_scopes],
        "bootstrap_n": int(cfg.bootstrap_n),
        "output_root": str(out_root),
        "boundary": "independent semantic rewrite; reproduces V9 subpeak/peak extraction; no V9/V7 imports; main V10 tree implementation",
    }, out_root / "summary.json")
    (log_dir / "last_run.txt").write_text(
        f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')} output={out_root}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    run_peak_subpeak_reproduce_v10_a(Path(__file__).resolve().parents[2])
