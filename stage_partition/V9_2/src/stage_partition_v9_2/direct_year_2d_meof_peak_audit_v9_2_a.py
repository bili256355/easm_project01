"""
V9.2_a direct-year 2D-field MVEOF + PC-year-group V9 peak audit.

Purpose
-------
V9.2_a is a control/diagnostic line for the V9/V9.1 peak-order work.  It does
not use bootstrap-resampled year combinations as samples and does not use
profile features as the SVD input.  Instead it uses the real years 1979-2023 as
samples and 2D object fields as MVEOF features.  PC high/mid/low real-year
groups are then composited and passed through the existing V9/V7 peak semantics.

Explicit boundaries
-------------------
- No bootstrap sample expansion is used as evidence.
- No single-year peak is used as the main result.
- MVEOF modes are field modes; they are not physical regimes by themselves.
- PC high/low groups are real-year score phases, not named physical year types.
- Group-composite peak differences are timing diagnostics, not causality.
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

VERSION = "v9_2_a"
OUTPUT_TAG = "direct_year_2d_meof_peak_audit_v9_2_a"
DEFAULT_WINDOWS = ("W045", "W081", "W113", "W160")
OBJECT_ORDER = ("P", "V", "H", "Je", "Jw")


@dataclass(frozen=True)
class Object2DSpec:
    object_name: str
    field_role: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


OBJECT_2D_SPECS: Tuple[Object2DSpec, ...] = (
    Object2DSpec("P", "precip", 105.0, 125.0, 15.0, 39.0),
    Object2DSpec("V", "v850", 105.0, 125.0, 10.0, 30.0),
    Object2DSpec("H", "z500", 110.0, 140.0, 15.0, 35.0),
    Object2DSpec("Je", "u200", 120.0, 150.0, 25.0, 45.0),
    Object2DSpec("Jw", "u200", 80.0, 110.0, 25.0, 45.0),
)


@dataclass
class V92Settings:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    target_windows: Tuple[str, ...] = DEFAULT_WINDOWS
    years_start: int = 1979
    years_end: int = 2023
    spatial_res_deg: float = 2.0
    n_modes_main: int = 3
    n_modes_save: int = 5
    pc_group_method: str = "tercile"
    pc_group_min_years: int = 10
    use_sqrt_coslat_weight: bool = True
    object_block_equal_weight: bool = True
    standardize_features_across_years: bool = True
    std_eps: float = 1.0e-12
    run_leave_one_year_mode_stability: bool = True
    # hotfix02: group-composite leave-one-year peak stability is expensive because
    # it reruns the V9/V7 detector for every object/year/mode/phase. Keep it
    # available, but do not run it by default.
    run_leave_one_year_group_peak_stability: bool = False
    # hotfix02: the core V9.2 contrast is high-vs-low. Mid-group peaks are
    # optional and disabled by default to reduce detector calls.
    run_pc_mid_group_peak: bool = False
    # hotfix02: saving coarsened input fields is useful for debugging but slow and
    # disk-heavy. Keep audit CSVs by default; enable NPZ dump only when needed.
    save_coarsened_input_fields: bool = False
    allow_detector_compat_bootstrap_retry: bool = False
    compat_detector_bootstrap_n: int = 1
    use_bootstrap_resampling_as_evidence: bool = False
    use_single_year_peak_as_main_result: bool = False

    @classmethod
    def from_env(cls) -> "V92Settings":
        s = cls()
        if os.environ.get("V9_2_TARGET_WINDOWS"):
            s.target_windows = tuple(w.strip() for w in os.environ["V9_2_TARGET_WINDOWS"].split(",") if w.strip())
        if os.environ.get("V9_2_SPATIAL_RES_DEG"):
            s.spatial_res_deg = float(os.environ["V9_2_SPATIAL_RES_DEG"])
        if os.environ.get("V9_2_N_MODES_MAIN"):
            s.n_modes_main = int(os.environ["V9_2_N_MODES_MAIN"])
        if os.environ.get("V9_2_N_MODES_SAVE"):
            s.n_modes_save = int(os.environ["V9_2_N_MODES_SAVE"])
        if os.environ.get("V9_2_RUN_LOO_MODE_STABILITY"):
            s.run_leave_one_year_mode_stability = _env_bool("V9_2_RUN_LOO_MODE_STABILITY", s.run_leave_one_year_mode_stability)
        if os.environ.get("V9_2_RUN_LOO_GROUP_PEAK_STABILITY"):
            s.run_leave_one_year_group_peak_stability = _env_bool("V9_2_RUN_LOO_GROUP_PEAK_STABILITY", s.run_leave_one_year_group_peak_stability)
        if os.environ.get("V9_2_RUN_PC_MID_GROUP_PEAK"):
            s.run_pc_mid_group_peak = _env_bool("V9_2_RUN_PC_MID_GROUP_PEAK", s.run_pc_mid_group_peak)
        if os.environ.get("V9_2_SAVE_COARSENED_INPUT_FIELDS"):
            s.save_coarsened_input_fields = _env_bool("V9_2_SAVE_COARSENED_INPUT_FIELDS", s.save_coarsened_input_fields)
        if os.environ.get("V9_2_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY"):
            s.allow_detector_compat_bootstrap_retry = _env_bool("V9_2_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY", s.allow_detector_compat_bootstrap_retry)
        if os.environ.get("V9_2_COMPAT_DETECTOR_BOOTSTRAP_N"):
            s.compat_detector_bootstrap_n = int(os.environ["V9_2_COMPAT_DETECTOR_BOOTSTRAP_N"])
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


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _set_or_insert_front(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Set a metadata column and keep it near the front without failing on duplicates.

    Some V7 peak-selection helpers already return columns such as ``window_id``.
    V9.2_a normalizes them to the current loop values rather than inserting a
    duplicate column, which would raise ``ValueError: cannot insert ..., already exists``.
    """
    if column in df.columns:
        df[column] = value
        cols = [column] + [c for c in df.columns if c != column]
        return df.loc[:, cols]
    df.insert(0, column, value)
    return df


def _stage_root_from_v92(v92_root: Path) -> Path:
    # Expected: .../stage_partition/V9_2
    return v92_root.parent


def _default_smoothed_path(v92_root: Path) -> Path:
    # v92_root = D:/easm_project01/stage_partition/V9_2
    return v92_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _default_v9_out_dir(v92_root: Path) -> Path:
    return v92_root.parent / "V9" / "outputs" / "peak_all_windows_v9_a"


def _import_v7_module(v92_root: Path):
    stage_root = _stage_root_from_v92(v92_root)
    v7_src = stage_root / "V7" / "src"
    if not v7_src.exists():
        raise FileNotFoundError(
            f"Cannot find V7 source directory: {v7_src}. V9.2_a needs V7 helpers "
            "to preserve V9/V7 peak semantics for PC-group composite peaks."
        )
    if str(v7_src) not in sys.path:
        sys.path.insert(0, str(v7_src))
    from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as v7multi
    return v7multi


def _make_v7_cfg_for_v92(v7multi, settings: V92Settings, v92_root: Path) -> object:
    cfg = v7multi.MultiWinConfig.from_env()
    if os.environ.get("V9_2_SMOOTHED_FIELDS"):
        cfg.smoothed_fields_path = os.environ["V9_2_SMOOTHED_FIELDS"]
    else:
        cfg.smoothed_fields_path = str(_default_smoothed_path(v92_root))
    cfg.window_mode = "list"
    cfg.target_windows = ",".join(settings.target_windows)
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    # V9.2_a does not use bootstrap as evidence.  The call to the V7 detector is
    # only used to preserve observed group-composite peak selection semantics.
    cfg.bootstrap_n = 0
    return cfg


def _read_v9_scopes(v9_out_dir: Path, settings: V92Settings) -> List[SimpleNamespace]:
    candidates = [
        v9_out_dir / "cross_window" / "run_window_scope_registry_v9_peak_all_windows_a.csv",
        v9_out_dir / "cross_window" / "window_scope_registry_v9_peak_all_windows_a.csv",
    ]
    scope_path = next((p for p in candidates if p.exists()), None)
    if scope_path is None:
        raise FileNotFoundError(
            "Cannot find V9 window scope registry. Expected one of: "
            + "; ".join(str(p) for p in candidates)
        )
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


def _native_scalar(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _load_tau_map(v9_out_dir: Path) -> Dict[str, float]:
    path = v9_out_dir / "cross_window" / "tau_sync_estimate_all_windows.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        try:
            out[str(r["window_id"])] = float(r["tau_sync_primary"])
        except Exception:
            pass
    return out


def _lat_lon_prepare(lat: np.ndarray, lon: np.ndarray, arrays_by_field: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], dict]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_order = "ascending" if lat[0] <= lat[-1] else "descending"
    lon_order = "ascending" if lon[0] <= lon[-1] else "descending"
    arrs = dict(arrays_by_field)
    if lat_order == "descending":
        idx = np.argsort(lat)
        lat = lat[idx]
        for k, a in arrs.items():
            arrs[k] = a[:, :, idx, :]
    if lon_order == "descending":
        idx = np.argsort(lon)
        lon = lon[idx]
        for k, a in arrs.items():
            arrs[k] = a[:, :, :, idx]
    audit = {
        "lat_original_order": lat_order,
        "lon_original_order": lon_order,
        "lat_sorted_for_processing": True,
        "lon_sorted_for_processing": True,
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
    }
    return lat, lon, arrs, audit


def _year_indices(years: np.ndarray, settings: V92Settings) -> Tuple[np.ndarray, np.ndarray]:
    years = np.asarray(years).astype(int)
    selected_years = np.arange(settings.years_start, settings.years_end + 1, dtype=int)
    index_map = {int(y): i for i, y in enumerate(years)}
    missing = [int(y) for y in selected_years if int(y) not in index_map]
    if missing:
        raise ValueError(f"smoothed_fields years do not cover requested years: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    indices = np.asarray([index_map[int(y)] for y in selected_years], dtype=int)
    return selected_years, indices


def _crop_field(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, spec: Object2DSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_mask = (lat >= spec.lat_min) & (lat <= spec.lat_max)
    lon_mask = (lon >= spec.lon_min) & (lon <= spec.lon_max)
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points for {spec.object_name}: {spec.lat_min}-{spec.lat_max}")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points for {spec.object_name}: {spec.lon_min}-{spec.lon_max}")
    cropped = arr[:, :, lat_mask, :][:, :, :, lon_mask]
    return cropped, lat[lat_mask], lon[lon_mask]


def _coarsen_2d(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, res: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Bin-average a year x day x lat x lon array to roughly res-degree cells."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_min = math.floor(float(np.nanmin(lat)) / res) * res
    lat_max = math.ceil(float(np.nanmax(lat)) / res) * res
    lon_min = math.floor(float(np.nanmin(lon)) / res) * res
    lon_max = math.ceil(float(np.nanmax(lon)) / res) * res
    lat_edges = np.arange(lat_min, lat_max + res * 1.001, res)
    lon_edges = np.arange(lon_min, lon_max + res * 1.001, res)
    out_blocks: List[np.ndarray] = []
    lat_centers: List[float] = []
    lon_centers: List[float] = []
    # Collect each valid cell as year x day vector, then reshape at the end.
    cell_arrays = []
    cell_lat = []
    cell_lon = []
    for i in range(len(lat_edges) - 1):
        lo, hi = lat_edges[i], lat_edges[i + 1]
        if i == len(lat_edges) - 2:
            lat_mask = (lat >= lo) & (lat <= hi)
        else:
            lat_mask = (lat >= lo) & (lat < hi)
        if not np.any(lat_mask):
            continue
        for j in range(len(lon_edges) - 1):
            lo2, hi2 = lon_edges[j], lon_edges[j + 1]
            if j == len(lon_edges) - 2:
                lon_mask = (lon >= lo2) & (lon <= hi2)
            else:
                lon_mask = (lon >= lo2) & (lon < hi2)
            if not np.any(lon_mask):
                continue
            sub = arr[:, :, lat_mask, :][:, :, :, lon_mask]
            cell = np.nanmean(sub, axis=(2, 3))
            cell_arrays.append(cell)
            cell_lat.append(float(np.nanmean(lat[lat_mask])))
            cell_lon.append(float(np.nanmean(lon[lon_mask])))
    if not cell_arrays:
        raise ValueError("Coarsening produced no valid spatial cells")
    stacked = np.stack(cell_arrays, axis=2)  # year x day x cell
    audit = {
        "n_lat_original": int(len(lat)),
        "n_lon_original": int(len(lon)),
        "n_cells_coarsened": int(stacked.shape[2]),
        "spatial_res_deg": float(res),
    }
    return stacked, np.asarray(cell_lat, dtype=float), np.asarray(cell_lon, dtype=float), audit


def _window_slice(scope: SimpleNamespace) -> slice:
    return slice(int(scope.analysis_start), int(scope.analysis_end) + 1)


def _build_window_matrix(
    scope: SimpleNamespace,
    arrays_by_field: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    selected_years: np.ndarray,
    year_idx: np.ndarray,
    settings: V92Settings,
) -> Tuple[np.ndarray, List[dict], Dict[str, dict], Dict[str, np.ndarray]]:
    """Build direct-year 2D-field MVEOF matrix for a single window."""
    wsl = _window_slice(scope)
    blocks = []
    feature_rows: List[dict] = []
    domain_rows: Dict[str, dict] = {}
    field_npz: Dict[str, np.ndarray] = {}
    for spec in OBJECT_2D_SPECS:
        arr = arrays_by_field[spec.field_role][year_idx, wsl, :, :]
        cropped, lats, lons = _crop_field(arr, lat, lon, spec)
        coarsened, cell_lats, cell_lons, caudit = _coarsen_2d(cropped, lats, lons, settings.spatial_res_deg)
        if settings.use_sqrt_coslat_weight:
            weights = np.sqrt(np.maximum(np.cos(np.deg2rad(cell_lats)), 0.0))
            coarsened = coarsened * weights[None, None, :]
        raw = coarsened.reshape(coarsened.shape[0], -1)
        n_raw = int(raw.shape[1])
        all_nan = np.all(~np.isfinite(raw), axis=0)
        raw2 = raw[:, ~all_nan]
        # Anomaly / z-score across real years for each feature.
        mean = np.nanmean(raw2, axis=0)
        centered = raw2 - mean[None, :]
        std = np.nanstd(centered, axis=0, ddof=1)
        valid_std = np.isfinite(std) & (std > settings.std_eps)
        centered = centered[:, valid_std]
        std = std[valid_std]
        if settings.standardize_features_across_years:
            block = centered / std[None, :]
        else:
            block = centered
        # After centering, remaining NaNs are filled with zero = feature mean anomaly.
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
        feature_rows.append({
            "window_id": scope.window_id,
            "object": spec.object_name,
            "field_role": spec.field_role,
            "n_years": int(len(selected_years)),
            "analysis_start": int(scope.analysis_start),
            "analysis_end": int(scope.analysis_end),
            "n_days": int(coarsened.shape[1]),
            "n_features_raw": n_raw,
            "n_all_nan_features_removed": int(all_nan.sum()),
            "n_features_after_nan_filter": int(raw2.shape[1]),
            "n_features_after_std_filter": int(block.shape[1]),
            "n_partial_nan_values_filled": partial_nan_count,
            "block_scale": float(scale),
            "block_variance_before_scale": block_var_before,
            "block_variance_after_scale": block_var_after,
        })
        domain_rows[spec.object_name] = {
            "object": spec.object_name,
            "field_role": spec.field_role,
            "lon_min": spec.lon_min,
            "lon_max": spec.lon_max,
            "lat_min": spec.lat_min,
            "lat_max": spec.lat_max,
            "n_lat_original": int(cropped.shape[2]),
            "n_lon_original": int(cropped.shape[3]),
            "n_cells_coarsened": int(coarsened.shape[2]),
            **caudit,
        }
        field_npz[f"{scope.window_id}_{spec.object_name}_coarsened"] = coarsened
        field_npz[f"{scope.window_id}_{spec.object_name}_cell_lats"] = cell_lats
        field_npz[f"{scope.window_id}_{spec.object_name}_cell_lons"] = cell_lons
    X = np.concatenate(blocks, axis=1)
    # Recenter after block scaling; SVD is on real-year anomaly matrix.
    X = X - np.nanmean(X, axis=0, keepdims=True)
    return X, feature_rows, domain_rows, field_npz


def _fit_svd(X: np.ndarray, settings: V92Settings) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 years for SVD")
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    evr = var / np.sum(var) if np.sum(var) > 0 else np.full_like(var, np.nan)
    n = min(settings.n_modes_save, len(S))
    # Reproducible sign convention: largest absolute loading is positive.
    for k in range(n):
        vec = Vt[k]
        if vec.size and vec[int(np.nanargmax(np.abs(vec)))] < 0:
            Vt[k] *= -1.0
            U[:, k] *= -1.0
    PC = U[:, :n] * S[:n][None, :]
    return PC, S[:n], Vt[:n], evr[:n]


def _pc_group_table(window_id: str, mode: int, years: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"year": years.astype(int), "pc_score": scores.astype(float)})
    df = df.sort_values("pc_score", ascending=True).reset_index(drop=True)
    n = len(df)
    n_low = n // 3
    n_high = n // 3
    labels = np.array(["mid"] * n, dtype=object)
    labels[:n_low] = "low"
    labels[n - n_high:] = "high"
    df["pc_phase"] = labels
    df["window_id"] = window_id
    df["mode"] = mode
    df["pc_rank_low_to_high"] = np.arange(1, n + 1)
    mean = float(df["pc_score"].mean())
    sd = float(df["pc_score"].std(ddof=1))
    df["pc_score_z"] = (df["pc_score"] - mean) / sd if sd > 0 else 0.0
    return df[["window_id", "mode", "pc_phase", "year", "pc_score", "pc_score_z", "pc_rank_low_to_high"]]


def _build_group_profiles(v7multi, fields: dict, lat: np.ndarray, lon: np.ndarray, years: np.ndarray, selected_years: Sequence[int]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    selected_years = [int(y) for y in selected_years]
    years_all = np.asarray(years).astype(int)
    idx_map = {int(y): i for i, y in enumerate(years_all)}
    missing = [y for y in selected_years if y not in idx_map]
    if missing:
        raise ValueError(f"Years missing from fields for group profile: {missing}")
    idx = np.asarray([idx_map[y] for y in selected_years], dtype=int)
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for spec in v7multi.clean.OBJECT_SPECS:
        arr = v7multi.clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        arr = arr[idx, :, :, :]
        prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (prof, target_lat, weights)
    return profiles


def _run_group_peak(v7multi, profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], scope: SimpleNamespace, cfg: object, settings: V92Settings) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Run existing V7 peak selection on a PC-group composite profile set.

    V9.2_a only uses observed/group-composite selected peaks.  It does not use
    bootstrap outputs as evidence.  By default this tries cfg.bootstrap_n=0.  If
    the local V7 helper cannot handle zero bootstrap, the run fails unless the
    user explicitly enables V9_2_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY=1.
    """
    local_cfg = copy.deepcopy(cfg)
    local_cfg.bootstrap_n = 0
    try:
        _score_df, _cand_df, selection_df, selected_delta_df, _boot_df = v7multi._run_detector_and_bootstrap(profiles, scope, local_cfg)
        return selection_df, selected_delta_df, "v7_detector_zero_bootstrap", "ok"
    except Exception as exc:
        if not settings.allow_detector_compat_bootstrap_retry:
            raise RuntimeError(
                "V7 _run_detector_and_bootstrap failed with bootstrap_n=0. "
                "V9.2_a does not use bootstrap as evidence. If your local V7 helper requires at least one bootstrap "
                "for API compatibility, set V9_2_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY=1; the compatibility bootstrap "
                "will still be ignored by V9.2 outputs. Original error: " + repr(exc)
            )
        local_cfg = copy.deepcopy(cfg)
        local_cfg.bootstrap_n = int(settings.compat_detector_bootstrap_n)
        _score_df, _cand_df, selection_df, selected_delta_df, _boot_df = v7multi._run_detector_and_bootstrap(profiles, scope, local_cfg)
        return selection_df, selected_delta_df, f"v7_detector_compat_bootstrap_n_{settings.compat_detector_bootstrap_n}_ignored", "compat_retry_used"


def _pairwise_from_selection(sel: pd.DataFrame, window_id: str, tau: Optional[float]) -> pd.DataFrame:
    if sel is None or sel.empty or "object" not in sel.columns or "selected_peak_day" not in sel.columns:
        return pd.DataFrame()
    peak = {str(r["object"]): float(r["selected_peak_day"]) for _, r in sel.iterrows() if pd.notna(r.get("selected_peak_day"))}
    rows = []
    for a, b in itertools.combinations(OBJECT_ORDER, 2):
        pa = peak.get(a, np.nan)
        pb = peak.get(b, np.nan)
        delta = pb - pa if np.isfinite(pa) and np.isfinite(pb) else np.nan
        if not np.isfinite(delta):
            order = "unresolved_due_to_missing_peak"
        elif tau is not None and abs(delta) <= tau:
            order = "same_day_or_near"
        elif delta > 0:
            order = "A_earlier"
        else:
            order = "B_earlier"
        rows.append({
            "window_id": window_id,
            "object_A": a,
            "object_B": b,
            "peak_A": pa,
            "peak_B": pb,
            "delta_B_minus_A": delta,
            "tau_sync_primary_used": tau if tau is not None else np.nan,
            "order_label": order,
        })
    return pd.DataFrame(rows)


def _high_low_contrast(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if pair_df is None or pair_df.empty:
        return pd.DataFrame()
    keys = ["window_id", "mode", "object_A", "object_B"]
    for key, g in pair_df.groupby(keys, dropna=False):
        phases = {str(r["pc_phase"]): r for _, r in g.iterrows()}
        if "high" not in phases or "low" not in phases:
            continue
        hi, lo = phases["high"], phases["low"]
        dh = float(hi["delta_B_minus_A"]) if pd.notna(hi["delta_B_minus_A"]) else np.nan
        dl = float(lo["delta_B_minus_A"]) if pd.notna(lo["delta_B_minus_A"]) else np.nan
        order_h = str(hi["order_label"])
        order_l = str(lo["order_label"])
        if not np.isfinite(dh) or not np.isfinite(dl):
            ctype = "missing_or_unresolved"
        elif order_h != order_l and {order_h, order_l} <= {"A_earlier", "B_earlier"}:
            ctype = "reversal_like"
        elif order_h != order_l:
            ctype = "synchrony_or_order_shift"
        elif abs(dh) > abs(dl) + 1.0:
            ctype = "lead_lag_strengthened_in_high"
        elif abs(dl) > abs(dh) + 1.0:
            ctype = "lead_lag_strengthened_in_low"
        else:
            ctype = "no_clear_order_change"
        rows.append({
            "window_id": key[0],
            "mode": int(key[1]),
            "object_A": key[2],
            "object_B": key[3],
            "peak_A_high": hi.get("peak_A", np.nan),
            "peak_B_high": hi.get("peak_B", np.nan),
            "delta_high": dh,
            "order_high": order_h,
            "peak_A_low": lo.get("peak_A", np.nan),
            "peak_B_low": lo.get("peak_B", np.nan),
            "delta_low": dl,
            "order_low": order_l,
            "delta_shift_high_minus_low": dh - dl if np.isfinite(dh) and np.isfinite(dl) else np.nan,
            "order_changed_flag": bool(order_h != order_l),
            "order_contrast_type": ctype,
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


def _mode_stability_for_window(X: np.ndarray, years: np.ndarray, full_pc: np.ndarray, full_vt: np.ndarray, settings: V92Settings) -> pd.DataFrame:
    rows: List[dict] = []
    n_modes = min(settings.n_modes_main, full_vt.shape[0])
    for i, yr in enumerate(years):
        mask = np.ones(len(years), dtype=bool)
        mask[i] = False
        Xd = X[mask, :]
        Xd = Xd - np.nanmean(Xd, axis=0, keepdims=True)
        try:
            Ud, Sd, Vtd = np.linalg.svd(Xd, full_matrices=False)
        except Exception as exc:
            for mode in range(1, n_modes + 1):
                rows.append({"mode": mode, "left_out_year": int(yr), "status": "svd_failed", "error": repr(exc)})
            continue
        for k in range(n_modes):
            eof_full = full_vt[k]
            eof_loo = Vtd[k]
            eof_corr = _corr(eof_full, eof_loo)
            sign = 1.0
            if np.isfinite(eof_corr) and eof_corr < 0:
                sign = -1.0
                eof_corr = -eof_corr
            pc_full_projected = Xd @ eof_full
            pc_loo = (Ud[:, k] * Sd[k]) * sign
            pc_corr = _corr(pc_full_projected, pc_loo)
            rows.append({
                "mode": k + 1,
                "left_out_year": int(yr),
                "eof_loading_abs_corr": eof_corr,
                "pc_score_corr_after_alignment": pc_corr,
                "mode_sign_aligned": bool(sign < 0),
                "stability_flag": "stable" if np.isfinite(eof_corr) and eof_corr >= 0.80 else "unstable_or_mixed",
                "status": "ok",
            })
    return pd.DataFrame(rows)


def _peak_relevance_summary(mode_summary_df: pd.DataFrame, pair_contrast_df: pd.DataFrame, peak_stability_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if mode_summary_df.empty:
        return pd.DataFrame()
    for _, m in mode_summary_df.iterrows():
        wid = str(m["window_id"])
        mode = int(m["mode"])
        sub = pair_contrast_df[(pair_contrast_df["window_id"] == wid) & (pair_contrast_df["mode"] == mode)] if not pair_contrast_df.empty else pd.DataFrame()
        n_order_change = int(sub["order_changed_flag"].sum()) if not sub.empty and "order_changed_flag" in sub.columns else 0
        n_reversal = int((sub["order_contrast_type"] == "reversal_like").sum()) if not sub.empty and "order_contrast_type" in sub.columns else 0
        max_abs_shift = float(np.nanmax(np.abs(sub["delta_shift_high_minus_low"]))) if not sub.empty and "delta_shift_high_minus_low" in sub.columns else np.nan
        st = peak_stability_df[(peak_stability_df["window_id"] == wid) & (peak_stability_df["mode"] == mode)] if not peak_stability_df.empty else pd.DataFrame()
        same_frac = float(np.nanmean(st["same_peak_flag"].astype(float))) if not st.empty and "same_peak_flag" in st.columns else np.nan
        if n_reversal >= 1 and (not np.isfinite(same_frac) or same_frac >= 0.70):
            cls = "peak_relevant_strong"
        elif n_order_change >= 1 or (np.isfinite(max_abs_shift) and max_abs_shift >= 3):
            cls = "peak_relevant_moderate"
        elif np.isfinite(max_abs_shift):
            cls = "field_mode_only_or_weak_peak_relevance"
        else:
            cls = "unresolved_due_to_missing_peak_outputs"
        rows.append({
            "window_id": wid,
            "mode": mode,
            "explained_variance_ratio": m.get("explained_variance_ratio", np.nan),
            "n_pairs_with_order_change": n_order_change,
            "n_pairs_reversal_like": n_reversal,
            "max_abs_high_low_delta_shift": max_abs_shift,
            "mean_leave_one_year_same_peak_fraction": same_frac,
            "peak_relevance_class": cls,
            "interpretation_permission": "statistical_timing_audit_only_not_physical_regime",
        })
    return pd.DataFrame(rows)


def _write_summary(path: Path, settings: V92Settings, mode_summary: pd.DataFrame, phase_years: pd.DataFrame, contrast: pd.DataFrame, relevance: pd.DataFrame) -> None:
    lines = [
        "# V9.2_a direct-year 2D-field MVEOF peak audit summary",
        "",
        f"version: `{VERSION}`",
        f"output_tag: `{OUTPUT_TAG}`",
        "",
        "## Method boundary",
        "- Samples are real years 1979-2023, not bootstrap-resampled year combinations.",
        "- X is built from 2D object fields, not zonal/profile features.",
        "- MVEOF modes are unsupervised field modes; they are not target-guided peak-order modes.",
        "- Peak timing is computed on PC high/mid/low group composites, not on single years.",
        "- PC high/low groups are score phases of real years, not named physical regimes.",
        "",
        "## Configuration",
        f"- target_windows: {', '.join(settings.target_windows)}",
        f"- spatial_res_deg: {settings.spatial_res_deg}",
        f"- n_modes_main: {settings.n_modes_main}; n_modes_save: {settings.n_modes_save}",
        "",
        "## Mode overview",
    ]
    if not mode_summary.empty:
        for _, r in mode_summary.iterrows():
            lines.append(f"- {r['window_id']} mode {int(r['mode'])}: EVR={float(r['explained_variance_ratio']):.4f}")
    lines += ["", "## PC high/low real years"]
    if not phase_years.empty:
        for (wid, mode, phase), g in phase_years.groupby(["window_id", "mode", "pc_phase"]):
            if phase in {"high", "low"}:
                yrs = ", ".join(str(int(y)) for y in g.sort_values("pc_score")["year"].tolist())
                lines.append(f"- {wid} mode {int(mode)} {phase}: {yrs}")
    lines += ["", "## Peak-relevance overview"]
    if not relevance.empty:
        for _, r in relevance.iterrows():
            lines.append(f"- {r['window_id']} mode {int(r['mode'])}: {r['peak_relevance_class']}; order_change_pairs={int(r['n_pairs_with_order_change'])}")
    lines += [
        "",
        "## Forbidden interpretations",
        "- Do not interpret PC high/low as named physical year types without later physical audit.",
        "- Do not interpret MVEOF maximum-variance modes as peak-order mechanisms by default.",
        "- Do not interpret group-composite peak timing as a single-year rule.",
        "- Do not infer causality from high/low order contrasts.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_direct_year_2d_meof_peak_audit_v9_2_a(v92_root: Path | str) -> None:
    v92_root = Path(v92_root)
    settings = V92Settings.from_env()
    out_root = _ensure_dir(v92_root / "outputs" / settings.output_tag)
    out_cross = _ensure_dir(out_root / "cross_window")
    out_per = _ensure_dir(out_root / "per_window")
    log_dir = _ensure_dir(v92_root / "logs" / settings.output_tag)
    t0 = time.time()

    _log("[1/10] Resolve paths and import V7/V9 helpers")
    v7multi = _import_v7_module(v92_root)
    cfg = _make_v7_cfg_for_v92(v7multi, settings, v92_root)
    v9_out_dir = Path(os.environ.get("V9_2_V9_OUTPUT_DIR", str(_default_v9_out_dir(v92_root))))
    smoothed_path = Path(os.environ.get("V9_2_SMOOTHED_FIELDS", cfg.smoothed_fields_path))
    if not v9_out_dir.exists():
        raise FileNotFoundError(f"V9 output directory not found: {v9_out_dir}")
    if not smoothed_path.exists():
        raise FileNotFoundError(f"smoothed_fields.npz not found: {smoothed_path}")

    _log("[2/10] Load V9 accepted window scopes")
    scopes = _read_v9_scopes(v9_out_dir, settings)
    tau_map = _load_tau_map(v9_out_dir)
    _safe_to_csv(pd.DataFrame([vars(s) for s in scopes]), out_cross / "v9_2_window_scope_used.csv")

    _log("[3/10] Load smoothed 2D fields")
    fields, input_audit = v7multi.clean._load_npz_fields(smoothed_path)
    years_raw = np.asarray(fields["years"]).astype(int)
    selected_years, year_idx = _year_indices(years_raw, settings)
    required_fields = sorted(set(spec.field_role for spec in OBJECT_2D_SPECS))
    arrays_by_field = {}
    for role in required_fields:
        arrays_by_field[role] = v7multi.clean._as_year_day_lat_lon(fields[role], fields["lat"], fields["lon"], fields.get("years"))
    lat, lon, arrays_by_field, latlon_audit = _lat_lon_prepare(fields["lat"], fields["lon"], arrays_by_field)
    input_rows = [
        {"item": "smoothed_fields_path", "value": str(smoothed_path)},
        {"item": "v9_output_dir", "value": str(v9_out_dir)},
        {"item": "years_start", "value": int(settings.years_start)},
        {"item": "years_end", "value": int(settings.years_end)},
        {"item": "n_years_selected", "value": int(len(selected_years))},
        *[{"item": k, "value": v} for k, v in latlon_audit.items()],
    ]
    _safe_to_csv(pd.DataFrame(input_rows), out_cross / "v9_2_input_field_audit.csv")

    all_domain_rows: List[dict] = []
    all_feature_rows: List[dict] = []
    all_mode_rows: List[dict] = []
    all_pc_rows: List[pd.DataFrame] = []
    all_phase_rows: List[pd.DataFrame] = []
    all_peak_rows: List[pd.DataFrame] = []
    all_pair_rows: List[pd.DataFrame] = []
    all_contrast_rows: List[pd.DataFrame] = []
    all_mode_stability_rows: List[pd.DataFrame] = []
    all_peak_stability_rows: List[pd.DataFrame] = []

    # Need original lat/lon/fields for V7 profile builders; do not use the sorted/coarsened arrays for that call.
    fields_for_profiles = fields

    _log("[4/10] Build direct-year 2D object matrices and fit MVEOF")
    for w_i, scope in enumerate(scopes, start=1):
        wid = str(scope.window_id)
        _log(f"  [{w_i}/{len(scopes)}] {wid}: build X")
        X, feature_rows, domain_rows, field_npz = _build_window_matrix(scope, arrays_by_field, lat, lon, selected_years, year_idx, settings)
        all_feature_rows.extend(feature_rows)
        all_domain_rows.extend(domain_rows.values())
        win_dir = _ensure_dir(out_per / wid)
        if settings.save_coarsened_input_fields:
            np.savez_compressed(win_dir / f"v9_2_coarsened_input_fields_{wid}.npz", **field_npz, years=selected_years)

        _log(f"  [{w_i}/{len(scopes)}] {wid}: SVD")
        PC, S, Vt, evr = _fit_svd(X, settings)
        np.savez_compressed(win_dir / f"v9_2_eof_patterns_{wid}.npz", Vt=Vt, singular_values=S, explained_variance_ratio=evr, years=selected_years)
        cum = np.cumsum(evr)
        for k in range(len(S)):
            all_mode_rows.append({
                "window_id": wid,
                "mode": k + 1,
                "singular_value": float(S[k]),
                "explained_variance_ratio": float(evr[k]),
                "cumulative_explained_variance_ratio": float(cum[k]),
                "n_years": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_objects": int(len(OBJECT_2D_SPECS)),
                "sign_convention": "largest_abs_loading_positive",
            })
        pc_rows = []
        for k in range(PC.shape[1]):
            scores = PC[:, k]
            mean = float(np.nanmean(scores))
            sd = float(np.nanstd(scores, ddof=1))
            for y, sc in zip(selected_years, scores):
                pc_rows.append({
                    "window_id": wid,
                    "mode": k + 1,
                    "year": int(y),
                    "pc_score": float(sc),
                    "pc_score_z": float((sc - mean) / sd) if sd > 0 else 0.0,
                })
        pc_df = pd.DataFrame(pc_rows)
        all_pc_rows.append(pc_df)
        _safe_to_csv(pc_df, win_dir / f"pc_scores_{wid}.csv")

        _log(f"  [{w_i}/{len(scopes)}] {wid}: PC groups and group-composite peaks")
        phase_tables = []
        peak_tables = []
        pair_tables = []
        for k in range(min(settings.n_modes_main, PC.shape[1])):
            phase_df = _pc_group_table(wid, k + 1, selected_years, PC[:, k])
            phase_tables.append(phase_df)
            all_phase_rows.append(phase_df)
            phases_to_run = ["high", "low"] + (["mid"] if settings.run_pc_mid_group_peak else [])
            for phase in phases_to_run:
                group_years = phase_df.loc[phase_df["pc_phase"] == phase, "year"].astype(int).tolist()
                if len(group_years) < settings.pc_group_min_years:
                    peak_tables.append(pd.DataFrame([{
                        "window_id": wid, "mode": k + 1, "pc_phase": phase,
                        "status": "skipped_too_few_years", "n_years_in_group": len(group_years),
                        "years_in_group": ";".join(map(str, group_years)),
                    }]))
                    continue
                profiles = _build_group_profiles(v7multi, fields_for_profiles, fields["lat"], fields["lon"], years_raw, group_years)
                selection_df, _selected_delta, peak_source, detector_status = _run_group_peak(v7multi, profiles, scope, cfg, settings)
                selection_df = selection_df.copy()
                selection_df = _set_or_insert_front(selection_df, "window_id", wid)
                selection_df = _set_or_insert_front(selection_df, "mode", k + 1)
                selection_df = _set_or_insert_front(selection_df, "pc_phase", phase)
                selection_df["n_years_in_group"] = len(group_years)
                selection_df["years_in_group"] = ";".join(map(str, group_years))
                selection_df["peak_source"] = peak_source
                selection_df["detector_status"] = detector_status
                peak_tables.append(selection_df)
                pair_df = _pairwise_from_selection(selection_df, wid, tau_map.get(wid))
                if not pair_df.empty:
                    pair_df.insert(1, "mode", k + 1)
                    pair_df.insert(2, "pc_phase", phase)
                    pair_df["n_years_in_group"] = len(group_years)
                    pair_df["years_in_group"] = ";".join(map(str, group_years))
                    pair_tables.append(pair_df)
        phase_all = pd.concat(phase_tables, ignore_index=True) if phase_tables else pd.DataFrame()
        peak_all = pd.concat(peak_tables, ignore_index=True) if peak_tables else pd.DataFrame()
        pair_all = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
        contrast_all = _high_low_contrast(pair_all)
        all_peak_rows.append(peak_all)
        all_pair_rows.append(pair_all)
        all_contrast_rows.append(contrast_all)
        _safe_to_csv(phase_all, win_dir / f"pc_phase_years_{wid}.csv")
        _safe_to_csv(peak_all, win_dir / f"pc_group_object_peak_{wid}.csv")
        _safe_to_csv(pair_all, win_dir / f"pc_group_pairwise_peak_order_{wid}.csv")
        _safe_to_csv(contrast_all, win_dir / f"pc_group_high_low_order_contrast_{wid}.csv")

        if settings.run_leave_one_year_mode_stability:
            _log(f"  [{w_i}/{len(scopes)}] {wid}: leave-one-year mode stability")
            ms = _mode_stability_for_window(X, selected_years, PC, Vt, settings)
            ms.insert(0, "window_id", wid)
            all_mode_stability_rows.append(ms)
            _safe_to_csv(ms, win_dir / f"mode_stability_leave_one_year_{wid}.csv")

        if settings.run_leave_one_year_group_peak_stability:
            _log(f"  [{w_i}/{len(scopes)}] {wid}: leave-one-year group peak stability")
            st_rows: List[dict] = []
            if not peak_all.empty and "object" in peak_all.columns:
                for _, pr in peak_all.iterrows():
                    # Only high/low group stability; mid is less important and saves runtime.
                    if str(pr.get("pc_phase")) not in {"high", "low"} or pd.isna(pr.get("selected_peak_day")):
                        continue
                    mode = int(pr["mode"])
                    phase = str(pr["pc_phase"])
                    obj = str(pr["object"])
                    full_peak = float(pr["selected_peak_day"])
                    years_in_group = [int(x) for x in str(pr["years_in_group"]).split(";") if str(x).strip()]
                    for left_out in years_in_group:
                        keep_years = [y for y in years_in_group if y != left_out]
                        if len(keep_years) < settings.pc_group_min_years:
                            st_rows.append({
                                "window_id": wid, "mode": mode, "pc_phase": phase, "object": obj,
                                "left_out_year": left_out, "full_group_peak_day": full_peak,
                                "loo_group_peak_day": np.nan, "peak_day_shift": np.nan,
                                "same_peak_flag": False, "status": "skipped_too_few_years_after_removal",
                            })
                            continue
                        profiles = _build_group_profiles(v7multi, fields_for_profiles, fields["lat"], fields["lon"], years_raw, keep_years)
                        try:
                            sel_loo, _sd, _src, det_status = _run_group_peak(v7multi, profiles, scope, cfg, settings)
                            row = sel_loo[sel_loo["object"].astype(str) == obj]
                            loo_peak = float(row.iloc[0]["selected_peak_day"]) if not row.empty else np.nan
                            shift = loo_peak - full_peak if np.isfinite(loo_peak) else np.nan
                            st_rows.append({
                                "window_id": wid, "mode": mode, "pc_phase": phase, "object": obj,
                                "left_out_year": left_out, "full_group_peak_day": full_peak,
                                "loo_group_peak_day": loo_peak, "peak_day_shift": shift,
                                "same_peak_flag": bool(np.isfinite(shift) and abs(shift) <= 0.0),
                                "status": det_status,
                            })
                        except Exception as exc:
                            st_rows.append({
                                "window_id": wid, "mode": mode, "pc_phase": phase, "object": obj,
                                "left_out_year": left_out, "full_group_peak_day": full_peak,
                                "loo_group_peak_day": np.nan, "peak_day_shift": np.nan,
                                "same_peak_flag": False, "status": "detector_failed", "error": repr(exc),
                            })
            st_df = pd.DataFrame(st_rows)
            all_peak_stability_rows.append(st_df)
            _safe_to_csv(st_df, win_dir / f"leave_one_year_group_peak_stability_{wid}.csv")

    _log("[8/10] Write cross-window outputs")
    domain_df = pd.DataFrame(all_domain_rows).drop_duplicates()
    feature_df = pd.DataFrame(all_feature_rows)
    mode_df = pd.DataFrame(all_mode_rows)
    pc_df = pd.concat(all_pc_rows, ignore_index=True) if all_pc_rows else pd.DataFrame()
    phase_df = pd.concat(all_phase_rows, ignore_index=True) if all_phase_rows else pd.DataFrame()
    peak_df = pd.concat(all_peak_rows, ignore_index=True) if all_peak_rows else pd.DataFrame()
    pair_df = pd.concat(all_pair_rows, ignore_index=True) if all_pair_rows else pd.DataFrame()
    contrast_df = pd.concat(all_contrast_rows, ignore_index=True) if all_contrast_rows else pd.DataFrame()
    mode_stability_df = pd.concat(all_mode_stability_rows, ignore_index=True) if all_mode_stability_rows else pd.DataFrame()
    peak_stability_df = pd.concat(all_peak_stability_rows, ignore_index=True) if all_peak_stability_rows else pd.DataFrame()
    relevance_df = _peak_relevance_summary(mode_df[mode_df["mode"] <= settings.n_modes_main].copy(), contrast_df, peak_stability_df)

    _safe_to_csv(domain_df, out_cross / "v9_2_object_2d_domain_registry.csv")
    _safe_to_csv(feature_df, out_cross / "v9_2_meof_feature_block_audit.csv")
    _safe_to_csv(feature_df[["window_id", "object", "n_days", "n_features_raw", "n_features_after_std_filter"]].copy(), out_cross / "v9_2_spatial_coarsening_audit.csv")
    _safe_to_csv(mode_df, out_cross / "v9_2_meof_mode_summary_all_windows.csv")
    _safe_to_csv(pc_df, out_cross / "v9_2_pc_scores_all_windows.csv")
    _safe_to_csv(phase_df, out_cross / "v9_2_pc_phase_years_all_windows.csv")
    _safe_to_csv(peak_df, out_cross / "v9_2_pc_group_object_peak_all_windows.csv")
    _safe_to_csv(pair_df, out_cross / "v9_2_pc_group_pairwise_peak_order_all_windows.csv")
    _safe_to_csv(contrast_df, out_cross / "v9_2_pc_group_high_low_order_contrast_all_windows.csv")
    _safe_to_csv(mode_stability_df, out_cross / "v9_2_mode_stability_leave_one_year_all_windows.csv")
    _safe_to_csv(peak_stability_df, out_cross / "v9_2_leave_one_year_group_peak_stability_all_windows.csv")
    _safe_to_csv(relevance_df, out_cross / "v9_2_peak_relevance_summary_all_windows.csv")

    _log("[9/10] Write summary and run_meta")
    _write_summary(out_cross / "V9_2_A_SUMMARY.md", settings, mode_df, phase_df, contrast_df, relevance_df)
    run_meta = {
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "started_at": _now(),
        "elapsed_seconds": time.time() - t0,
        "settings": asdict(settings),
        "v9_output_dir": str(v9_out_dir),
        "smoothed_fields": str(smoothed_path),
        "windows_processed": [str(s.window_id) for s in scopes],
        "years_used": [int(y) for y in selected_years.tolist()],
        "method_boundary": {
            "uses_real_year_samples": True,
            "uses_bootstrap_resampling_as_evidence": False,
            "uses_single_year_peak_as_main_result": False,
            "x_input": "direct-year 2D object fields: object x relative_day x lat x lon",
            "peak_input": "PC high/mid/low group-composite profiles using V9/V7 peak semantics",
            "physical_interpretation_included": False,
        },
        "source_v7_module_version": getattr(v7multi, "VERSION", "unknown"),
        "source_v7_output_tag": getattr(v7multi, "OUTPUT_TAG", "unknown"),
    }
    _write_json(run_meta, out_cross / "run_meta.json")
    _write_json(run_meta, out_root / "run_meta.json")

    _log("[10/10] Done")
    (log_dir / "last_run.txt").write_text(
        f"Completed {_now()} output={out_root}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    run_direct_year_2d_meof_peak_audit_v9_2_a(Path(__file__).resolve().parents[2])
