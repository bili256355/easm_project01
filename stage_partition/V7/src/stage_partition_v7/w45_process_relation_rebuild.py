from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Any

import numpy as np
import pandas as pd

# Base-data dependencies. These are intentionally the same low-level inputs used by the
# V7 progress line, but this module does NOT read V7-m/n/o derived outputs.
from stage_partition_v6.io import load_smoothed_fields
from stage_partition_v6.state_builder import build_profiles, build_state_matrix
from stage_partition_v7.config import StagePartitionV7Settings

try:  # Available in the user's full V7 project; absent in some minimal patch bundles.
    from stage_partition_v7.field_state import build_field_state_matrix_for_year_indices as _v7_build_field_state
except Exception:  # noqa: BLE001
    _v7_build_field_state = None

OUTPUT_TAG = "w45_process_relation_rebuild_v7_p"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELDS = ["P", "V", "H", "Je", "Jw"]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75]
EARLY_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
FORBIDDEN_INPUT_DIR_KEYWORDS = [
    "w45_allfield_transition_marker_definition_audit_v7_m",
    "w45_allfield_process_relation_layer_v7_n",
    "w45_process_curve_foundation_audit_v7_o",
]
EPS = 1e-12


@dataclass
class V7PSettings:
    v7_root: Path
    v6_root: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path
    window_id: str = WINDOW_ID
    anchor_day: int = ANCHOR_DAY
    analysis_start: int = 30
    analysis_end: int = 60
    pre_start: int = 30
    pre_end: int = 37
    post_start: int = 53
    post_end: int = 60
    fields: tuple[str, ...] = tuple(FIELDS)
    n_bootstrap: int = 1000
    random_seed: int = 20260445
    stable_crossing_days: int = 2
    progress_clip_min: float = 0.0
    progress_clip_max: float = 1.0
    write_bootstrap_long_table: bool = True
    progress_every: int = 50
    read_previous_derived_results: bool = False


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _safe_arr(values: Iterable[Any] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return arr[np.isfinite(arr)]


def _q(values: Iterable[Any] | np.ndarray, q: float) -> float:
    arr = _safe_arr(values)
    return float(np.nanquantile(arr, q)) if arr.size else np.nan


def _med(values: Iterable[Any] | np.ndarray) -> float:
    arr = _safe_arr(values)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def _mean(values: Iterable[Any] | np.ndarray) -> float:
    arr = _safe_arr(values)
    return float(np.nanmean(arr)) if arr.size else np.nan


def _valid_fraction(values: Iterable[Any] | np.ndarray) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return float(np.isfinite(arr).sum() / arr.size) if arr.size else np.nan


def _norm(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.sqrt(np.nansum(np.square(arr))))


def _rms(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.sqrt(np.nanmean(np.square(arr))))


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() == 0:
        return np.nan
    return float(np.nansum(aa[mask] * bb[mask]))


def _fmt_threshold(qv: float) -> str:
    return f"t{int(round(float(qv) * 100)):02d}"


def _first_stable_crossing(days: np.ndarray, vals: np.ndarray, threshold: float, stable_days: int) -> float:
    stable_days = max(1, int(stable_days))
    days = np.asarray(days, dtype=int)
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return np.nan
    for i in range(0, len(vals) - stable_days + 1):
        win = vals[i : i + stable_days]
        if np.all(np.isfinite(win)) and np.all(win >= float(threshold)):
            return float(days[i])
    return np.nan


def _rolling_mean(vals: np.ndarray, window: int = 3) -> np.ndarray:
    s = pd.Series(np.asarray(vals, dtype=float))
    return s.rolling(int(window), center=True, min_periods=1).mean().to_numpy(dtype=float)


def _stats(values: Iterable[Any] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return {
        "median": _med(arr),
        "q05": _q(arr, 0.05),
        "q25": _q(arr, 0.25),
        "q75": _q(arr, 0.75),
        "q95": _q(arr, 0.95),
        "q025": _q(arr, 0.025),
        "q975": _q(arr, 0.975),
        "q90_width": _q(arr, 0.95) - _q(arr, 0.05),
        "q95_width": _q(arr, 0.975) - _q(arr, 0.025),
        "iqr": _q(arr, 0.75) - _q(arr, 0.25),
        "valid_fraction": _valid_fraction(arr),
    }


def _join(items: Iterable[Any]) -> str:
    vals: list[str] = []
    for it in items:
        s = str(it)
        if s and s.lower() not in {"nan", "none"}:
            vals.append(s)
    return ";".join(vals) if vals else "none"


def _full_matrix(matrix: np.ndarray, valid_day_index: np.ndarray, n_days: int) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float)
    full = np.full((int(n_days), mat.shape[1]), np.nan, dtype=float)
    idx = np.asarray(valid_day_index, dtype=int)
    mask = (idx >= 0) & (idx < n_days)
    idx = idx[mask]
    full[idx, :] = mat[: idx.size, :]
    return full


def _row_finite_mask(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((arr.shape[0] if arr.ndim else 0,), dtype=bool)
    return np.all(np.isfinite(arr), axis=1)


def _nanmean_rows(x: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    good = _row_finite_mask(arr)
    if int(good.sum()) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr[good, :], axis=0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return np.nan
    aa = aa[mask]
    bb = bb[mask]
    if np.nanstd(aa) < EPS or np.nanstd(bb) < EPS:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _resolve_settings(v7_root: Optional[Path]) -> tuple[V7PSettings, StagePartitionV7Settings]:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    v6_root = v7_root.parent / "V6"
    base = StagePartitionV7Settings()
    # Align project root with this installed V7 tree where possible.
    try:
        base.foundation.project_root = v7_root.parents[1]
        base.source.project_root = v7_root.parents[1]
    except Exception:  # noqa: BLE001
        pass
    n_boot = int(base.bootstrap.effective_n_bootstrap()) if hasattr(base.bootstrap, "effective_n_bootstrap") else 1000
    debug_n = getattr(base.bootstrap, "debug_n_bootstrap", None)
    if debug_n is not None:
        n_boot = int(debug_n)
    settings = V7PSettings(
        v7_root=v7_root,
        v6_root=v6_root,
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
        n_bootstrap=n_boot,
        random_seed=int(getattr(base.bootstrap, "random_seed", 20260445)),
        stable_crossing_days=int(getattr(base.progress_timing, "stable_crossing_days", 2)),
        progress_clip_min=float(getattr(base.progress_timing, "progress_clip_min", 0.0)),
        progress_clip_max=float(getattr(base.progress_timing, "progress_clip_max", 1.0)),
        progress_every=int(getattr(base.bootstrap, "progress_every", 50)),
    )
    return settings, base


# -----------------------------------------------------------------------------
# Base rebuild
# -----------------------------------------------------------------------------

def _audit_no_forbidden_inputs(settings: V7PSettings) -> None:
    if settings.read_previous_derived_results:
        raise RuntimeError("V7-p forbids reading previous derived V7-m/n/o outputs as input.")
    for key in FORBIDDEN_INPUT_DIR_KEYWORDS:
        if key in str(settings.output_dir):
            raise RuntimeError(f"V7-p output_dir unexpectedly points to forbidden derived branch: {key}")


def _load_profiles(base_settings: StagePartitionV7Settings):
    smoothed_path = base_settings.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)
    profiles = build_profiles(smoothed, base_settings.profile)
    return smoothed_path, profiles


def _fallback_build_field_state(profiles: dict, field: str, year_indices: Optional[np.ndarray], standardize: bool = True, shared_valid_day_index: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fallback when stage_partition_v7.field_state is unavailable.

    This uses the current 2-degree interpolated profile cube directly. It averages
    resampled years, z-scores each feature along day, and then optionally restricts
    to the joint valid day index. This is not a substitute for the full V7 field_state
    module, but it keeps the new trunk runnable and explicitly records the fallback.
    """
    cube = np.asarray(profiles[field].raw_cube, dtype=float)
    if year_indices is None:
        block = np.nanmean(cube, axis=0)
    else:
        block = np.nanmean(cube[np.asarray(year_indices, dtype=int), :, :], axis=0)
    mat = np.asarray(block, dtype=float)
    if standardize:
        mu = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
        sd = np.where((~np.isfinite(sd)) | (sd < EPS), 1.0, sd)
        mat = (mat - mu[None, :]) / sd[None, :]
    if shared_valid_day_index is None:
        valid = np.where(np.all(np.isfinite(mat), axis=1))[0].astype(int)
    else:
        valid = np.asarray(shared_valid_day_index, dtype=int)
        valid = valid[(valid >= 0) & (valid < mat.shape[0])]
    meta = pd.DataFrame({"feature_index": np.arange(mat.shape[1]), "field": field, "lat_value": getattr(profiles[field], "lat_grid", np.arange(mat.shape[1]))})
    return mat[valid, :], valid, meta


def _build_field_state(profiles: dict, field: str, year_indices: Optional[np.ndarray], base_settings: StagePartitionV7Settings, shared_valid_day_index: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    if _v7_build_field_state is not None:
        matrix, valid_idx, feature_meta = _v7_build_field_state(
            profiles,
            field,
            year_indices,
            standardize=base_settings.state.standardize,
            trim_invalid_days=base_settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index,
        )
        return np.asarray(matrix, dtype=float), np.asarray(valid_idx, dtype=int), pd.DataFrame(feature_meta), "stage_partition_v7.field_state"
    matrix, valid_idx, meta = _fallback_build_field_state(
        profiles,
        field,
        year_indices,
        standardize=bool(getattr(base_settings.state, "standardize", True)),
        shared_valid_day_index=shared_valid_day_index,
    )
    return matrix, valid_idx, meta, "fallback_profile_cube_zscore"


def _bootstrap_indices(settings: V7PSettings, n_years: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(settings.random_seed))
    return [rng.integers(0, int(n_years), size=int(n_years), dtype=int) for _ in range(int(settings.n_bootstrap))]


def _build_observed_states(profiles: dict, settings: V7PSettings, base_settings: StagePartitionV7Settings) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], dict[str, str], np.ndarray, int]:
    first_cube = np.asarray(profiles[FIELDS[0]].raw_cube)
    n_days = int(first_cube.shape[1])
    joint_state = build_state_matrix(profiles, base_settings.state)
    shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)
    states: dict[str, np.ndarray] = {}
    metas: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for field in FIELDS:
        mat, idx, meta, source = _build_field_state(profiles, field, None, base_settings, shared_valid_day_index)
        states[field] = _full_matrix(mat, idx, n_days)
        metas[field] = meta.copy()
        sources[field] = source
    return states, metas, sources, shared_valid_day_index, n_days


def _write_state_summary(profiles: dict, states: dict[str, np.ndarray], metas: dict[str, pd.DataFrame], sources: dict[str, str], settings: V7PSettings, smoothed_path: Path, base_settings: StagePartitionV7Settings) -> pd.DataFrame:
    rows = []
    for field in FIELDS:
        obj = profiles[field]
        arr = states[field]
        lat_grid = getattr(obj, "lat_grid", np.array([]))
        rows.append(
            {
                "field": field,
                "source_variable_profile_key": field,
                "state_builder_source": sources.get(field, "unknown"),
                "n_days": int(arr.shape[0]),
                "n_features": int(arr.shape[1]),
                "state_nan_fraction": float((~np.isfinite(arr)).mean()) if arr.size else np.nan,
                "lat_min": float(np.nanmin(lat_grid)) if len(lat_grid) else np.nan,
                "lat_max": float(np.nanmax(lat_grid)) if len(lat_grid) else np.nan,
                "n_lat_features": int(len(lat_grid)),
                "lon_range": str(getattr(obj, "lon_range", "unknown")),
                "lat_range": str(getattr(obj, "lat_range", "unknown")),
                "lat_step_deg": float(getattr(base_settings.profile, "lat_step_deg", np.nan)),
                "interpolation_method": "lon_mean_then_np_interp_to_2deg_lat_grid",
                "smoothed_fields_path": str(smoothed_path),
            }
        )
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_field_state_rebuild_summary_v7_p.csv")
    # Save observed states for traceability; not used as input by later V7-p steps.
    npz_path = settings.output_dir / "w45_field_state_rebuild_v7_p.npz"
    _ensure_dir(npz_path.parent)
    np.savez_compressed(npz_path, **{f"{f}_state": states[f] for f in FIELDS})
    return df


# -----------------------------------------------------------------------------
# Process curves and foundation
# -----------------------------------------------------------------------------

def _prepost_for_state(full: np.ndarray, settings: V7PSettings) -> tuple[np.ndarray | None, np.ndarray | None]:
    pre = full[settings.pre_start : settings.pre_end + 1, :]
    post = full[settings.post_start : settings.post_end + 1, :]
    return _nanmean_rows(pre), _nanmean_rows(post)


def _within_variability(block: np.ndarray, proto: np.ndarray | None) -> float:
    if proto is None:
        return np.nan
    good = _row_finite_mask(block)
    if good.sum() == 0:
        return np.nan
    vals = [_rms(row - proto) for row in block[good, :]]
    return _mean(vals)


def _compute_curve_for_full_state(full: np.ndarray, field: str, settings: V7PSettings, sample_col: Optional[str] = None, sample_value: Optional[int] = None) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    pre_proto, post_proto = _prepost_for_state(full, settings)
    base = {
        "window_id": settings.window_id,
        "anchor_day": int(settings.anchor_day),
        "analysis_window_start": int(settings.analysis_start),
        "analysis_window_end": int(settings.analysis_end),
        "pre_period_start": int(settings.pre_start),
        "pre_period_end": int(settings.pre_end),
        "post_period_start": int(settings.post_start),
        "post_period_end": int(settings.post_end),
        "field": field,
    }
    if sample_col is not None:
        base[sample_col] = int(sample_value) if sample_value is not None else np.nan
    if pre_proto is None or post_proto is None:
        return {**base, "prepost_status": "prototype_unavailable"}, [], pd.DataFrame()

    d = post_proto - pre_proto
    denom = _dot(d, d)
    d_norm = math.sqrt(denom) if np.isfinite(denom) and denom > 0 else np.nan
    pre_block = full[settings.pre_start : settings.pre_end + 1, :]
    post_block = full[settings.post_start : settings.post_end + 1, :]
    pre_var = _within_variability(pre_block, pre_proto)
    post_var = _within_variability(post_block, post_proto)
    within = _mean([pre_var, post_var])
    sep_ratio = float(d_norm / (within + EPS)) if np.isfinite(d_norm) and np.isfinite(within) else np.nan
    proto_rows = []
    abs_sum = float(np.nansum(np.abs(d)))
    for j, val in enumerate(d):
        proto_rows.append({**base, "feature_id": int(j), "pre_mean": float(pre_proto[j]) if np.isfinite(pre_proto[j]) else np.nan, "post_mean": float(post_proto[j]) if np.isfinite(post_proto[j]) else np.nan, "dF": float(val) if np.isfinite(val) else np.nan, "abs_dF": float(abs(val)) if np.isfinite(val) else np.nan, "relative_abs_dF_contribution": float(abs(val) / (abs_sum + EPS)) if np.isfinite(val) else np.nan})

    curve_rows: list[dict[str, Any]] = []
    prev_prog = np.nan
    prev_raw = np.nan
    for day in range(settings.analysis_start, settings.analysis_end + 1):
        x0 = full[day, :]
        if not np.all(np.isfinite(x0)) or not np.isfinite(denom) or denom <= EPS:
            continue
        x = x0 - pre_proto
        raw_prog = _dot(x, d) / denom
        clipped = float(np.clip(raw_prog, settings.progress_clip_min, settings.progress_clip_max)) if np.isfinite(raw_prog) else np.nan
        proj_comp = raw_prog * d if np.isfinite(raw_prog) else np.full_like(d, np.nan)
        residual = x - proj_comp
        dist_pre = _rms(x0 - pre_proto)
        dist_post = _rms(x0 - post_proto)
        distance_prog = float(dist_pre / (dist_pre + dist_post + EPS)) if np.isfinite(dist_pre) and np.isfinite(dist_post) else np.nan
        proj_norm = _rms(proj_comp)
        x_norm = _rms(x)
        res_norm = _rms(residual)
        projection_fraction = float(proj_norm / (x_norm + EPS)) if np.isfinite(proj_norm) and np.isfinite(x_norm) else np.nan
        residual_fraction = float(res_norm / (x_norm + EPS)) if np.isfinite(res_norm) and np.isfinite(x_norm) else np.nan
        curve_rows.append({
            **base,
            "day": int(day),
            "projection_progress_raw": float(raw_prog) if np.isfinite(raw_prog) else np.nan,
            "projection_progress": clipped,
            "distance_progress": distance_prog,
            "dprojection_progress": float(clipped - prev_prog) if np.isfinite(prev_prog) and np.isfinite(clipped) else np.nan,
            "dprojection_progress_raw": float(raw_prog - prev_raw) if np.isfinite(prev_raw) and np.isfinite(raw_prog) else np.nan,
            "dist_to_pre": dist_pre,
            "dist_to_post": dist_post,
            "projection_fraction": projection_fraction,
            "residual_fraction": residual_fraction,
            "progress_outside_0_1_raw": bool(np.isfinite(raw_prog) and (raw_prog < 0.0 or raw_prog > 1.0)),
        })
        prev_prog = clipped
        prev_raw = raw_prog
    row = {
        **base,
        "prepost_status": "success",
        "pre_norm": _rms(pre_proto),
        "post_norm": _rms(post_proto),
        "dF_norm": d_norm,
        "pre_internal_variability": pre_var,
        "post_internal_variability": post_var,
        "prepost_distance_ratio": sep_ratio,
        "n_curve_days": int(len(curve_rows)),
    }
    return row, curve_rows, pd.DataFrame(proto_rows)


def _label_prepost(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "prepost_not_established"
    if ratio >= 4.0:
        return "strong_prepost_foundation"
    if ratio >= 2.0:
        return "usable_prepost_foundation"
    if ratio >= 1.0:
        return "weak_prepost_foundation"
    return "prepost_not_established"


def _label_direction(proj_frac: float, res_frac: float) -> str:
    if not np.isfinite(proj_frac):
        return "direction_unavailable"
    if proj_frac >= 0.80 and (not np.isfinite(res_frac) or res_frac <= 0.70):
        return "dominant_prepost_direction"
    if proj_frac >= 0.60:
        return "usable_prepost_direction_with_residual"
    if proj_frac >= 0.40:
        return "multi_direction_transition"
    return "prepost_direction_not_representative"


def _label_projection_validity(corr: float, med_abs_diff: float, outside_frac: float) -> str:
    if not np.isfinite(corr):
        return "projection_progress_unavailable"
    if corr >= 0.90 and (not np.isfinite(med_abs_diff) or med_abs_diff <= 0.15) and (not np.isfinite(outside_frac) or outside_frac <= 0.10):
        return "valid_progress_proxy"
    if corr >= 0.80 and (not np.isfinite(med_abs_diff) or med_abs_diff <= 0.25):
        return "usable_progress_proxy_with_caution"
    if corr >= 0.60:
        return "projection_distance_inconsistent"
    return "projection_progress_not_valid"


def _extract_markers_from_curve(curve: pd.DataFrame, settings: V7PSettings) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if curve.empty:
        return {"marker_extract_status": "curve_empty"}
    c = curve.sort_values("day")
    days = pd.to_numeric(c["day"], errors="coerce").to_numpy(dtype=float)
    vals = pd.to_numeric(c["projection_progress"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(days) & np.isfinite(vals)
    days_i = days[good].astype(int)
    vals = vals[good].astype(float)
    if vals.size == 0:
        return {"marker_extract_status": "no_valid_progress"}
    out["marker_extract_status"] = "success"
    for th in THRESHOLDS:
        out[f"{_fmt_threshold(th)}_day"] = _first_stable_crossing(days_i, vals, th, settings.stable_crossing_days)
    pre_mask = (days_i >= settings.pre_start) & (days_i <= settings.pre_end)
    pre_vals = vals[pre_mask]
    pre_vals = pre_vals[np.isfinite(pre_vals)]
    if pre_vals.size >= 2:
        u90 = float(np.nanquantile(pre_vals, 0.90))
        u95 = float(np.nanquantile(pre_vals, 0.95))
        out["pre_progress_upper90"] = u90
        out["pre_progress_upper95"] = u95
        out["departure90_day"] = _first_stable_crossing(days_i, vals, u90, settings.stable_crossing_days)
        out["departure95_day"] = _first_stable_crossing(days_i, vals, u95, settings.stable_crossing_days)
    else:
        out["pre_progress_upper90"] = np.nan
        out["pre_progress_upper95"] = np.nan
        out["departure90_day"] = np.nan
        out["departure95_day"] = np.nan
    if vals.size >= 2:
        dvals = np.diff(vals)
        if np.isfinite(dvals).any():
            idx = int(np.nanargmax(dvals))
            out["peak_change_day_raw"] = float(days_i[idx + 1])
            out["peak_change_value_raw"] = float(dvals[idx])
        smooth = _rolling_mean(vals, window=3)
        ds = np.diff(smooth)
        if np.isfinite(ds).any():
            idx = int(np.nanargmax(ds))
            out["peak_change_day_smooth3"] = float(days_i[idx + 1])
            out["peak_change_value_smooth3"] = float(ds[idx])
    t25 = out.get("t25_day", np.nan)
    t50 = out.get("t50_day", np.nan)
    t75 = out.get("t75_day", np.nan)
    out["duration_25_75"] = float(t75 - t25) if np.isfinite(t25) and np.isfinite(t75) else np.nan
    out["tail_50_75"] = float(t75 - t50) if np.isfinite(t50) and np.isfinite(t75) else np.nan
    out["early_span_25_50"] = float(t50 - t25) if np.isfinite(t25) and np.isfinite(t50) else np.nan
    return out


def _marker_cols() -> list[str]:
    cols = ["departure90_day", "departure95_day"]
    cols += [f"{_fmt_threshold(t)}_day" for t in THRESHOLDS]
    cols += ["peak_change_day_raw", "peak_change_day_smooth3", "duration_25_75", "tail_50_75", "early_span_25_50"]
    return cols


def _marker_display(col: str) -> tuple[str, str]:
    if col.startswith("departure"):
        return "departure", col.replace("_day", "")
    if col.startswith("t") and col.endswith("_day"):
        return "threshold", col.replace("_day", "")
    if col.startswith("peak"):
        return "peak_change", col.replace("_day", "")
    if col.startswith("duration"):
        return "duration", col
    if col.startswith("tail"):
        return "tail", col
    if col.startswith("early_span"):
        return "early_span", col
    return "other", col


# -----------------------------------------------------------------------------
# Build branch data
# -----------------------------------------------------------------------------

def _build_all_from_base(settings: V7PSettings, base_settings: StagePartitionV7Settings):
    print("[1/12] audit base inputs")
    _audit_no_forbidden_inputs(settings)
    _ensure_dir(settings.output_dir)
    _ensure_dir(settings.log_dir)
    _ensure_dir(settings.figure_dir)
    smoothed_path, profiles = _load_profiles(base_settings)
    n_years = int(np.asarray(profiles[FIELDS[0]].raw_cube).shape[0])

    print("[2/12] rebuild field states")
    states, metas, state_sources, shared_valid_day_index, n_days = _build_observed_states(profiles, settings, base_settings)
    state_summary = _write_state_summary(profiles, states, metas, state_sources, settings, smoothed_path, base_settings)

    base_audit = {
        "version": OUTPUT_TAG,
        "created_at": _now_iso(),
        "window_id": settings.window_id,
        "anchor_day": settings.anchor_day,
        "fields": list(settings.fields),
        "input_representation": "current_2deg_interpolated_profile_state",
        "read_previous_derived_results": False,
        "v7_m_outputs_used_as_input": False,
        "v7_n_outputs_used_as_input": False,
        "v7_o_outputs_used_as_input": False,
        "analysis_window": [settings.analysis_start, settings.analysis_end],
        "pre_period": [settings.pre_start, settings.pre_end],
        "post_period": [settings.post_start, settings.post_end],
        "n_years": n_years,
        "n_days": n_days,
        "state_rebuild_status": "success",
        "field_state_builder_sources": state_sources,
        "smoothed_fields_path": str(smoothed_path),
        "forbidden_input_dir_keywords": FORBIDDEN_INPUT_DIR_KEYWORDS,
    }
    _write_json(base_audit, settings.output_dir / "input_base_audit_v7_p.json")

    print("[3/12] build pre/post prototypes and observed process curves")
    observed_rows: list[dict[str, Any]] = []
    observed_curve_rows: list[dict[str, Any]] = []
    proto_tables: list[pd.DataFrame] = []
    for field in FIELDS:
        row, curve_rows, proto = _compute_curve_for_full_state(states[field], field, settings)
        observed_rows.append(row)
        observed_curve_rows.extend(curve_rows)
        if not proto.empty:
            proto_tables.append(proto)
    observed_summary = pd.DataFrame(observed_rows)
    observed_curves = pd.DataFrame(observed_curve_rows)
    prepost_proto = pd.concat(proto_tables, ignore_index=True) if proto_tables else pd.DataFrame()
    _write_csv(prepost_proto, settings.output_dir / "w45_field_prepost_prototypes_v7_p.csv")
    _write_csv(observed_summary, settings.output_dir / "w45_field_prepost_summary_v7_p.csv")

    print("[4/12] bootstrap process curves from base")
    boot_indices = _bootstrap_indices(settings, n_years)
    boot_curve_rows: list[dict[str, Any]] = []
    boot_summary_rows: list[dict[str, Any]] = []
    for b, sample_idx in enumerate(boot_indices):
        if b % max(1, settings.progress_every) == 0:
            print(f"[V7-p bootstrap] {b}/{len(boot_indices)}")
        for field in FIELDS:
            mat, idx, _, _source = _build_field_state(profiles, field, sample_idx, base_settings, shared_valid_day_index)
            full = _full_matrix(mat, idx, n_days)
            row, curve_rows, _proto = _compute_curve_for_full_state(full, field, settings, sample_col="bootstrap_id", sample_value=b)
            boot_summary_rows.append(row)
            boot_curve_rows.extend(curve_rows)
    boot_curves = pd.DataFrame(boot_curve_rows)
    boot_summaries = pd.DataFrame(boot_summary_rows)

    print("[5/12] summarize process curves")
    field_curves = _summarize_field_curves(observed_curves, boot_curves, settings)
    _write_csv(field_curves, settings.output_dir / "w45_field_process_curves_v7_p.csv")
    if settings.write_bootstrap_long_table:
        # This table is useful for reproducibility but may be large; W45 only keeps it manageable.
        _write_csv(boot_curves, settings.output_dir / "w45_field_process_curves_bootstrap_v7_p.csv")
    else:
        np.savez_compressed(settings.output_dir / "w45_field_process_curves_bootstrap_v7_p.npz", data=boot_curves.to_records(index=False))

    print("[6/12] extract marker family")
    obs_markers = _extract_marker_table(observed_curves, settings, sample_col=None)
    boot_markers = _extract_marker_table(boot_curves, settings, sample_col="bootstrap_id")
    marker_family = _summarize_marker_family(obs_markers, boot_markers, settings)
    _write_csv(marker_family, settings.output_dir / "w45_field_marker_family_v7_p.csv")
    _write_csv(obs_markers, settings.output_dir / "w45_field_marker_observed_v7_p.csv")
    _write_csv(boot_markers, settings.output_dir / "w45_field_marker_bootstrap_samples_v7_p.csv")

    print("[7/12] audit field foundation")
    field_foundation = _build_field_foundation(observed_summary, boot_summaries, field_curves, marker_family, prepost_proto, settings)
    _write_csv(field_foundation, settings.output_dir / "w45_field_foundation_audit_v7_p.csv")

    print("[8/12] build pairwise daily relations")
    pair_daily = _build_pairwise_daily(field_curves, boot_curves, settings)
    _write_csv(pair_daily, settings.output_dir / "w45_pairwise_curve_relation_daily_v7_p.csv")

    print("[9/12] segment phase relations")
    pair_phase = _segment_pairwise_phases(pair_daily, settings)
    _write_csv(pair_phase, settings.output_dir / "w45_pairwise_phase_relation_v7_p.csv")

    print("[10/12] build pairwise marker relations")
    pair_marker = _build_pairwise_marker_relation(boot_markers, settings)
    _write_csv(pair_marker, settings.output_dir / "w45_pairwise_marker_relation_v7_p.csv")

    print("[11/12] build field and pair cards")
    field_cards = _build_field_cards(field_foundation, marker_family)
    pair_cards = _build_pair_cards(field_foundation, pair_daily, pair_phase, pair_marker, settings)
    org_layers = _build_window_layers(field_cards, pair_cards, pair_phase, pair_marker)
    decision_counts = _build_relation_decision_counts(pair_cards, field_cards)
    _write_csv(field_cards, settings.output_dir / "w45_field_process_cards_v7_p.csv")
    _write_csv(pair_cards, settings.output_dir / "w45_pair_relation_cards_v7_p.csv")
    _write_csv(org_layers, settings.output_dir / "w45_window_organization_layers_v7_p.csv")
    _write_csv(decision_counts, settings.output_dir / "w45_relation_decision_counts_v7_p.csv")

    print("[12/12] write summary and figures")
    _write_window_summary(settings, state_summary, field_cards, pair_cards, org_layers)
    _try_write_figures(settings, field_curves, pair_daily, pair_phase)
    run_meta = {
        "version": OUTPUT_TAG,
        "created_at": _now_iso(),
        "status": "success",
        "window_id": settings.window_id,
        "anchor_day": settings.anchor_day,
        "fields": list(settings.fields),
        "input_representation": "current_2deg_interpolated_profile_state",
        "read_previous_derived_results": False,
        "v7_m_outputs_used_as_input": False,
        "v7_n_outputs_used_as_input": False,
        "v7_o_outputs_used_as_input": False,
        "n_bootstrap": int(settings.n_bootstrap),
        "outputs": [
            "input_base_audit_v7_p.json",
            "w45_field_process_curves_v7_p.csv",
            "w45_field_foundation_audit_v7_p.csv",
            "w45_pairwise_curve_relation_daily_v7_p.csv",
            "w45_pairwise_phase_relation_v7_p.csv",
            "w45_pairwise_marker_relation_v7_p.csv",
            "w45_field_process_cards_v7_p.csv",
            "w45_pair_relation_cards_v7_p.csv",
            "w45_window_organization_card_v7_p.md",
        ],
    }
    _write_json(run_meta, settings.output_dir / "run_meta.json")
    _write_update_log(settings)


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def _summarize_field_curves(obs: pd.DataFrame, boot: pd.DataFrame, settings: V7PSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if obs.empty:
        return pd.DataFrame()
    boot_group = boot.groupby(["field", "day"], sort=False) if not boot.empty else {}
    for _, r in obs.sort_values(["field", "day"]).iterrows():
        field = str(r["field"])
        day = int(r["day"])
        brow = boot[(boot["field"].astype(str) == field) & (pd.to_numeric(boot["day"], errors="coerce") == day)] if not boot.empty else pd.DataFrame()
        base = {
            "window_id": settings.window_id,
            "anchor_day": settings.anchor_day,
            "field": field,
            "day": day,
            "projection_progress_observed": float(r.get("projection_progress", np.nan)),
            "projection_progress_raw_observed": float(r.get("projection_progress_raw", np.nan)),
            "distance_progress_observed": float(r.get("distance_progress", np.nan)),
            "dprojection_progress_observed": float(r.get("dprojection_progress", np.nan)),
            "dist_to_pre_observed": float(r.get("dist_to_pre", np.nan)),
            "dist_to_post_observed": float(r.get("dist_to_post", np.nan)),
            "projection_fraction_observed": float(r.get("projection_fraction", np.nan)),
            "residual_fraction_observed": float(r.get("residual_fraction", np.nan)),
        }
        for col, out in [
            ("projection_progress", "projection_progress"),
            ("distance_progress", "distance_progress"),
            ("dprojection_progress", "dprojection_progress"),
            ("residual_fraction", "residual_fraction"),
            ("projection_fraction", "projection_fraction"),
        ]:
            vals = pd.to_numeric(brow[col], errors="coerce") if col in brow.columns else pd.Series(dtype=float)
            st = _stats(vals.to_numpy())
            base[f"{out}_bootstrap_median"] = st["median"]
            base[f"{out}_q05"] = st["q05"]
            base[f"{out}_q25"] = st["q25"]
            base[f"{out}_q75"] = st["q75"]
            base[f"{out}_q95"] = st["q95"]
        for th in [0.10, 0.25, 0.50, 0.75]:
            vals = pd.to_numeric(brow.get("projection_progress", pd.Series(dtype=float)), errors="coerce")
            base[f"prob_progress_above_{int(th*100):03d}"] = float(np.nanmean(vals >= th)) if len(vals) else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def _extract_marker_table(curves: pd.DataFrame, settings: V7PSettings, sample_col: Optional[str]) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    group_cols = ["field"] + ([sample_col] if sample_col else [])
    rows: list[dict[str, Any]] = []
    for key, sub in curves.groupby(group_cols, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_tuple))
        markers = _extract_markers_from_curve(sub, settings)
        rows.append({"window_id": settings.window_id, "anchor_day": settings.anchor_day, **base, **markers})
    return pd.DataFrame(rows)


def _summarize_marker_family(obs_markers: pd.DataFrame, boot_markers: pd.DataFrame, settings: V7PSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in FIELDS:
        obs = obs_markers[obs_markers["field"].astype(str) == field]
        boot = boot_markers[boot_markers["field"].astype(str) == field]
        for col in _marker_cols():
            family, name = _marker_display(col)
            obs_val = float(obs[col].iloc[0]) if (not obs.empty and col in obs.columns and np.isfinite(obs[col].iloc[0])) else np.nan
            vals = pd.to_numeric(boot[col], errors="coerce") if col in boot.columns else pd.Series(dtype=float)
            st = _stats(vals.to_numpy())
            edge_hit = np.nan
            if col.endswith("_day") and vals.notna().any():
                edge_hit = float(((vals == settings.analysis_start) | (vals == settings.analysis_end)).mean())
            label = "usable" if st["valid_fraction"] >= 0.90 and st["q90_width"] <= 6 else "usable_with_caution" if st["valid_fraction"] >= 0.80 and st["q90_width"] <= 12 else "broad_uncertain" if st["valid_fraction"] >= 0.50 else "not_reliable"
            rows.append({
                "window_id": settings.window_id,
                "field": field,
                "marker_family": family,
                "marker_name": name,
                "source_column": col,
                "observed_value": obs_val,
                **st,
                "edge_hit_fraction": edge_hit,
                "marker_reliability_label": label,
                "marker_interpretation": _marker_interpretation(family, name),
            })
    return pd.DataFrame(rows)


def _marker_interpretation(family: str, name: str) -> str:
    if family == "departure":
        return "departure from pre-state background; not physical onset"
    if family == "threshold":
        return f"{name} progress crossing; threshold diagnostic only"
    if family == "peak_change":
        return "fastest progress-change day; not onset"
    if family in {"duration", "tail", "early_span"}:
        return "process-shape duration/tail diagnostic"
    return "marker diagnostic"


def _build_field_foundation(obs_summary: pd.DataFrame, boot_summary: pd.DataFrame, curves: pd.DataFrame, markers: pd.DataFrame, proto: pd.DataFrame, settings: V7PSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in FIELDS:
        obs = obs_summary[obs_summary["field"].astype(str) == field]
        cv = curves[curves["field"].astype(str) == field].sort_values("day")
        mk = markers[markers["field"].astype(str) == field]
        if obs.empty:
            rows.append({"field": field, "field_curve_foundation_label": "not_valid_as_single_curve", "allowed_uses": "none", "prohibited_uses": "all"})
            continue
        obs0 = obs.iloc[0]
        ratio = float(obs0.get("prepost_distance_ratio", np.nan))
        prepost_label = _label_prepost(ratio)
        proj_frac = _med(cv["projection_fraction_bootstrap_median"].to_numpy()) if "projection_fraction_bootstrap_median" in cv.columns else _med(cv["projection_fraction_observed"].to_numpy())
        res_frac = _med(cv["residual_fraction_bootstrap_median"].to_numpy()) if "residual_fraction_bootstrap_median" in cv.columns else _med(cv["residual_fraction_observed"].to_numpy())
        direction_label = _label_direction(proj_frac, res_frac)
        corr = _corr(cv["projection_progress_observed"].to_numpy(), cv["distance_progress_observed"].to_numpy()) if not cv.empty else np.nan
        mad = _med(np.abs(cv["projection_progress_observed"].to_numpy() - cv["distance_progress_observed"].to_numpy())) if not cv.empty else np.nan
        outside_frac = float(np.nanmean((cv["projection_progress_raw_observed"] < 0) | (cv["projection_progress_raw_observed"] > 1))) if "projection_progress_raw_observed" in cv.columns and not cv.empty else np.nan
        prog_label = _label_projection_validity(corr, mad, outside_frac)
        # Component adequacy from feature-wise pre/post contributions and approximate component timing dispersion.
        comp = proto[proto["field"].astype(str) == field]
        dominant = float(comp["relative_abs_dF_contribution"].max()) if not comp.empty else np.nan
        comp_t_iqr = _component_timing_iqr(comp, settings)
        single_label = _label_single_curve(dominant, comp_t_iqr)
        if prepost_label.startswith("strong") and direction_label.startswith("dominant") and prog_label in {"valid_progress_proxy", "usable_progress_proxy_with_caution"} and single_label in {"single_curve_adequate", "single_curve_usable_with_heterogeneity"}:
            foundation = "valid_single_process_curve" if single_label == "single_curve_adequate" and prog_label == "valid_progress_proxy" else "usable_with_caution"
        elif prepost_label == "prepost_not_established" or prog_label == "projection_progress_not_valid":
            foundation = "not_valid_as_single_curve"
        elif single_label in {"multi_component_transition", "single_curve_not_adequate"}:
            foundation = "multi_component_process"
        else:
            foundation = "usable_only_as_projection_diagnostic"
        rows.append({
            "field": field,
            "prepost_foundation_label": prepost_label,
            "transition_direction_label": direction_label,
            "projection_progress_validity_label": prog_label,
            "single_curve_adequacy_label": single_label,
            "field_curve_foundation_label": foundation,
            "prepost_distance_ratio": ratio,
            "projection_distance_corr": corr,
            "projection_distance_median_abs_diff": mad,
            "fraction_progress_outside_0_1": outside_frac,
            "median_projection_fraction": proj_frac,
            "median_residual_fraction": res_frac,
            "dominant_component_fraction": dominant,
            "component_timing_iqr_proxy": comp_t_iqr,
            "allowed_uses": _allowed_uses(foundation),
            "prohibited_uses": _prohibited_uses(foundation),
        })
    return pd.DataFrame(rows)


def _component_timing_iqr(comp: pd.DataFrame, settings: V7PSettings) -> float:
    # This is a lightweight proxy in V7-p trunk: component spread is inferred from contribution dominance.
    # Full feature-wise curves should be added only if this branch is promoted beyond W45.
    if comp.empty or "relative_abs_dF_contribution" not in comp.columns:
        return np.nan
    vals = comp["relative_abs_dF_contribution"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    # Higher entropy means more components contribute; not a day-IQR, but a heterogeneity proxy.
    p = vals / (vals.sum() + EPS)
    entropy = -float(np.nansum(p * np.log(p + EPS))) / math.log(len(p) + EPS) if len(p) > 1 else 0.0
    return entropy


def _label_single_curve(dominant: float, heterogeneity_proxy: float) -> str:
    if not np.isfinite(dominant):
        return "single_curve_not_adequate"
    if dominant <= 0.35 and np.isfinite(heterogeneity_proxy) and heterogeneity_proxy >= 0.65:
        return "single_curve_adequate"
    if dominant <= 0.50:
        return "single_curve_usable_with_heterogeneity"
    if dominant <= 0.70:
        return "multi_component_transition"
    return "single_curve_not_adequate"


def _allowed_uses(label: str) -> str:
    if label == "valid_single_process_curve":
        return "field_process_diagnostic;layer_specific_pair_relation;marker_family_summary"
    if label == "usable_with_caution":
        return "field_process_diagnostic;layer_specific_pair_relation_with_caution;marker_family_summary"
    if label == "multi_component_process":
        return "diagnostic_only;needs_component_or_spatial_support"
    if label == "usable_only_as_projection_diagnostic":
        return "projection_diagnostic_only;no_global_lead_lag"
    return "none"


def _prohibited_uses(label: str) -> str:
    common = "causality;physical_strength;global_clean_order_without_validation"
    if label in {"valid_single_process_curve", "usable_with_caution"}:
        return common
    if label == "multi_component_process":
        return common + ";single_curve_final_interpretation"
    return common + ";pairwise_phase_interpretation"


# -----------------------------------------------------------------------------
# Pairwise relations and cards
# -----------------------------------------------------------------------------

def _build_pairwise_daily(field_curves: pd.DataFrame, boot_curves: pd.DataFrame, settings: V7PSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    obs_p = field_curves.pivot_table(index="day", columns="field", values="projection_progress_observed", aggfunc="mean")
    obs_d = field_curves.pivot_table(index="day", columns="field", values="distance_progress_observed", aggfunc="mean")
    boot_p = boot_curves.pivot_table(index=["bootstrap_id", "day"], columns="field", values="projection_progress", aggfunc="mean") if not boot_curves.empty else pd.DataFrame()
    boot_d = boot_curves.pivot_table(index=["bootstrap_id", "day"], columns="field", values="distance_progress", aggfunc="mean") if not boot_curves.empty else pd.DataFrame()
    for a, b in combinations(FIELDS, 2):
        for day in range(settings.analysis_start, settings.analysis_end + 1):
            obs_diff = float(obs_p.loc[day, a] - obs_p.loc[day, b]) if day in obs_p.index and a in obs_p.columns and b in obs_p.columns else np.nan
            dist_diff = float(obs_d.loc[day, a] - obs_d.loc[day, b]) if day in obs_d.index and a in obs_d.columns and b in obs_d.columns else np.nan
            if not boot_p.empty and a in boot_p.columns and b in boot_p.columns:
                try:
                    vals = (boot_p.xs(day, level="day")[a] - boot_p.xs(day, level="day")[b]).to_numpy(dtype=float)
                except Exception:  # noqa: BLE001
                    vals = np.asarray([], dtype=float)
            else:
                vals = np.asarray([], dtype=float)
            if not boot_d.empty and a in boot_d.columns and b in boot_d.columns:
                try:
                    dvals = (boot_d.xs(day, level="day")[a] - boot_d.xs(day, level="day")[b]).to_numpy(dtype=float)
                except Exception:  # noqa: BLE001
                    dvals = np.asarray([], dtype=float)
            else:
                dvals = np.asarray([], dtype=float)
            st = _stats(vals)
            prob_a = float(np.nanmean(vals > 0)) if vals.size else np.nan
            prob_b = float(np.nanmean(vals < 0)) if vals.size else np.nan
            row = {
                "window_id": settings.window_id,
                "field_a": a,
                "field_b": b,
                "day": int(day),
                "projection_diff_A_minus_B_observed": obs_diff,
                "distance_diff_A_minus_B_observed": dist_diff,
                "projection_diff_median": st["median"],
                "projection_diff_q05": st["q05"],
                "projection_diff_q95": st["q95"],
                "projection_diff_q025": st["q025"],
                "projection_diff_q975": st["q975"],
                "prob_A_progress_gt_B": prob_a,
                "prob_B_progress_gt_A": prob_b,
                "prob_near_equal_delta_1": float(np.nanmean(np.abs(vals) <= 0.01)) if vals.size else np.nan,
                "prob_near_equal_delta_2": float(np.nanmean(np.abs(vals) <= 0.02)) if vals.size else np.nan,
                "prob_near_equal_delta_5": float(np.nanmean(np.abs(vals) <= 0.05)) if vals.size else np.nan,
                "projection_distance_sign_consistent": bool(np.isfinite(obs_diff) and np.isfinite(dist_diff) and np.sign(obs_diff) == np.sign(dist_diff)),
            }
            row["relation_day_label"] = _day_relation_label(row)
            row["relation_day_evidence_level"] = _day_evidence_level(row)
            rows.append(row)
    return pd.DataFrame(rows)


def _day_relation_label(row: dict[str, Any]) -> str:
    q05 = row.get("projection_diff_q05", np.nan)
    q95 = row.get("projection_diff_q95", np.nan)
    med = row.get("projection_diff_median", np.nan)
    pa = row.get("prob_A_progress_gt_B", np.nan)
    pb = row.get("prob_B_progress_gt_A", np.nan)
    pnear = row.get("prob_near_equal_delta_5", np.nan)
    if np.isfinite(q05) and q05 > 0:
        return "A_ahead_supported"
    if np.isfinite(q95) and q95 < 0:
        return "B_ahead_supported"
    if np.isfinite(med) and med > 0 and np.isfinite(pa) and pa >= 0.75:
        return "A_ahead_tendency"
    if np.isfinite(med) and med < 0 and np.isfinite(pb) and pb >= 0.75:
        return "B_ahead_tendency"
    if np.isfinite(pnear) and pnear >= 0.60:
        return "near_equal_candidate"
    return "not_resolved"


def _day_evidence_level(row: dict[str, Any]) -> str:
    label = str(row.get("relation_day_label"))
    if label.endswith("supported"):
        return "q05_q95_excludes_zero"
    if label.endswith("tendency"):
        return "probability_tendency"
    if label == "near_equal_candidate":
        return "near_equal_probability_only"
    return "unresolved"


def _segment_pairwise_phases(pair_daily: pd.DataFrame, settings: V7PSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (a, b), sub in pair_daily.sort_values(["field_a", "field_b", "day"]).groupby(["field_a", "field_b"], sort=False):
        current_label = None
        current: list[pd.Series] = []
        phase_id = 0
        def flush(chunk: list[pd.Series], label: str, pid: int) -> None:
            if not chunk:
                return
            df = pd.DataFrame(chunk)
            rows.append({
                "window_id": settings.window_id,
                "field_a": a,
                "field_b": b,
                "phase_id": int(pid),
                "day_start": int(df["day"].min()),
                "day_end": int(df["day"].max()),
                "n_days": int(len(df)),
                "phase_relation_label": _phase_label(label),
                "phase_evidence_level": _phase_evidence(df),
                "mean_projection_diff": _mean(df["projection_diff_median"].to_numpy()),
                "median_projection_diff": _med(df["projection_diff_median"].to_numpy()),
                "mean_prob_A_gt_B": _mean(df["prob_A_progress_gt_B"].to_numpy()),
                "mean_prob_B_gt_A": _mean(df["prob_B_progress_gt_A"].to_numpy()),
                "projection_distance_sign_consistency_fraction": float(np.nanmean(df["projection_distance_sign_consistent"].astype(float))) if "projection_distance_sign_consistent" in df.columns else np.nan,
                "phase_crossing_flag": bool(label == "crossing"),
                "phase_interpretation": _phase_interpretation(a, b, label, df),
            })
        prev_sign = None
        for _, r in sub.iterrows():
            lab0 = str(r["relation_day_label"])
            sign = 1 if lab0.startswith("A_") else -1 if lab0.startswith("B_") else 0
            lab = lab0
            # Preserve tendency/support direction. Crossings are represented by adjacent phase changes.
            if current_label is None:
                current_label = lab
                current = [r]
                prev_sign = sign
            elif lab == current_label:
                current.append(r)
                prev_sign = sign
            else:
                flush(current, current_label, phase_id)
                phase_id += 1
                current_label = lab
                current = [r]
                prev_sign = sign
        if current_label is not None:
            flush(current, current_label, phase_id)
    # Mark phases whose neighbours imply a sign crossing.
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["phase_crossing_context"] = False
    for (a, b), idxs in out.groupby(["field_a", "field_b"]).groups.items():
        idx_list = list(idxs)
        for i in range(1, len(idx_list)):
            prev = out.loc[idx_list[i - 1], "phase_relation_label"]
            cur = out.loc[idx_list[i], "phase_relation_label"]
            if (str(prev).startswith("A_") and str(cur).startswith("B_")) or (str(prev).startswith("B_") and str(cur).startswith("A_")):
                out.loc[idx_list[i - 1], "phase_crossing_context"] = True
                out.loc[idx_list[i], "phase_crossing_context"] = True
    return out


def _phase_label(day_label: str) -> str:
    if day_label == "A_ahead_supported":
        return "A_ahead_phase"
    if day_label == "B_ahead_supported":
        return "B_ahead_phase"
    if day_label == "A_ahead_tendency":
        return "A_tendency_phase"
    if day_label == "B_ahead_tendency":
        return "B_tendency_phase"
    if day_label == "near_equal_candidate":
        return "near_equal_phase_candidate"
    return "not_resolved_phase"


def _phase_evidence(df: pd.DataFrame) -> str:
    labels = set(df["relation_day_label"].astype(str))
    if any(x.endswith("supported") for x in labels):
        return "contains_q05_q95_supported_days"
    if any(x.endswith("tendency") for x in labels):
        return "contains_probability_tendency_days"
    if "near_equal_candidate" in labels:
        return "near_equal_candidate_only"
    return "unresolved"


def _phase_interpretation(a: str, b: str, label: str, df: pd.DataFrame) -> str:
    p = _phase_label(label)
    days = f"day {int(df['day'].min())}-{int(df['day'].max())}"
    if p.startswith("A_ahead"):
        return f"{a} progress exceeds {b} in {days}; relative transition phase only."
    if p.startswith("B_ahead"):
        return f"{b} progress exceeds {a} in {days}; relative transition phase only."
    if p.startswith("A_tendency"):
        return f"{a} tends ahead of {b} in {days}; not confirmed."
    if p.startswith("B_tendency"):
        return f"{b} tends ahead of {a} in {days}; not confirmed."
    if p.startswith("near_equal"):
        return f"{a}/{b} are near-equal candidates in {days}; not equivalence proof."
    return f"{a}/{b} relation unresolved in {days}."


def _build_pairwise_marker_relation(boot_markers: pd.DataFrame, settings: V7PSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for a, b in combinations(FIELDS, 2):
        am = boot_markers[boot_markers["field"].astype(str) == a]
        bm = boot_markers[boot_markers["field"].astype(str) == b]
        if am.empty or bm.empty:
            continue
        merged = am.merge(bm, on="bootstrap_id", suffixes=("_A", "_B"))
        for col in _marker_cols():
            ca = f"{col}_A"
            cb = f"{col}_B"
            if ca not in merged.columns or cb not in merged.columns:
                continue
            delta = pd.to_numeric(merged[cb], errors="coerce") - pd.to_numeric(merged[ca], errors="coerce")
            st = _stats(delta.to_numpy())
            prob_a_earlier = float(np.nanmean(delta > 0)) if len(delta) else np.nan
            prob_b_earlier = float(np.nanmean(delta < 0)) if len(delta) else np.nan
            prob_same = float(np.nanmean(delta == 0)) if len(delta) else np.nan
            family, name = _marker_display(col)
            row = {
                "window_id": settings.window_id,
                "field_a": a,
                "field_b": b,
                "marker_family": family,
                "marker_name": name,
                "source_column": col,
                "median_delta_B_minus_A": st["median"],
                "q05_delta": st["q05"],
                "q95_delta": st["q95"],
                "q025_delta": st["q025"],
                "q975_delta": st["q975"],
                "prob_A_earlier": prob_a_earlier,
                "prob_B_earlier": prob_b_earlier,
                "prob_same_day": prob_same,
                "pass90_A_leads": bool(np.isfinite(st["q05"]) and st["q05"] > 0),
                "pass95_A_leads": bool(np.isfinite(st["q025"]) and st["q025"] > 0),
                "pass90_B_leads": bool(np.isfinite(st["q95"]) and st["q95"] < 0),
                "pass95_B_leads": bool(np.isfinite(st["q975"]) and st["q975"] < 0),
                "minimum_equivalence_margin_90": float(max(abs(st["q05"]), abs(st["q95"]))) if np.isfinite(st["q05"]) and np.isfinite(st["q95"]) else np.nan,
                "minimum_equivalence_margin_95": float(max(abs(st["q025"]), abs(st["q975"]))) if np.isfinite(st["q025"]) and np.isfinite(st["q975"]) else np.nan,
            }
            row["marker_relation_label"] = _marker_relation_label(row)
            row["marker_relation_interpretation"] = _marker_relation_interpretation(a, b, row)
            rows.append(row)
    return pd.DataFrame(rows)


def _marker_relation_label(row: dict[str, Any]) -> str:
    if row.get("pass90_A_leads"):
        return "A_leads_90"
    if row.get("pass90_B_leads"):
        return "B_leads_90"
    med = row.get("median_delta_B_minus_A", np.nan)
    prob_a = row.get("prob_A_earlier", np.nan)
    prob_b = row.get("prob_B_earlier", np.nan)
    if np.isfinite(med) and med > 0 and np.isfinite(prob_a) and prob_a >= 0.70:
        return "A_tendency"
    if np.isfinite(med) and med < 0 and np.isfinite(prob_b) and prob_b >= 0.70:
        return "B_tendency"
    if np.isfinite(row.get("prob_same_day", np.nan)) and row["prob_same_day"] >= 0.60:
        return "same_day_candidate"
    return "not_resolved"


def _marker_relation_interpretation(a: str, b: str, row: dict[str, Any]) -> str:
    label = row.get("marker_relation_label")
    marker = row.get("marker_name")
    if label == "A_leads_90":
        return f"{a} is earlier than {b} for {marker} at 90%; marker-specific only."
    if label == "B_leads_90":
        return f"{b} is earlier than {a} for {marker} at 90%; marker-specific only."
    if label == "A_tendency":
        return f"{a} tends earlier than {b} for {marker}, not 90%-confirmed."
    if label == "B_tendency":
        return f"{b} tends earlier than {a} for {marker}, not 90%-confirmed."
    if label == "same_day_candidate":
        return f"{a}/{b} same-day candidate for {marker}; not equivalence proof."
    return f"{a}/{b} not resolved for {marker}."



def _field_caution_level(row: pd.Series) -> str:
    """Conservative field-card caution layer for V7-p hotfix_01.

    This does not recompute states. It prevents over-reading V7-p labels such as
    single_curve_adequate by combining the existing foundation fields with
    progress-bound and contribution diagnostics.
    """
    foundation = str(row.get("field_curve_foundation_label", "unknown"))
    single = str(row.get("single_curve_adequacy_label", "unknown"))
    outside = float(row.get("fraction_progress_outside_0_1", np.nan)) if pd.notna(row.get("fraction_progress_outside_0_1", np.nan)) else np.nan
    dominant = float(row.get("dominant_component_fraction", np.nan)) if pd.notna(row.get("dominant_component_fraction", np.nan)) else np.nan
    hetero = float(row.get("component_timing_iqr_proxy", np.nan)) if pd.notna(row.get("component_timing_iqr_proxy", np.nan)) else np.nan
    if foundation in {"not_valid_as_single_curve", "multi_component_process"} or single in {"multi_component_transition", "single_curve_not_adequate"}:
        return "high_component_or_curve_caution"
    if foundation != "valid_single_process_curve":
        return "moderate_component_or_curve_caution"
    if np.isfinite(outside) and outside > 0.20:
        return "moderate_component_or_curve_caution"
    if np.isfinite(dominant) and dominant > 0.50:
        return "moderate_component_or_curve_caution"
    if np.isfinite(hetero) and hetero < 0.50:
        # Low entropy means a few components dominate; this is only a proxy.
        return "moderate_component_or_curve_caution"
    return "low_caution"


def _field_caution_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    foundation = str(row.get("field_curve_foundation_label", "unknown"))
    single = str(row.get("single_curve_adequacy_label", "unknown"))
    outside = row.get("fraction_progress_outside_0_1", np.nan)
    dominant = row.get("dominant_component_fraction", np.nan)
    hetero = row.get("component_timing_iqr_proxy", np.nan)
    if foundation != "valid_single_process_curve":
        reasons.append(f"foundation={foundation}")
    if single != "single_curve_adequate":
        reasons.append(f"single_curve={single}")
    if pd.notna(outside) and float(outside) > 0.20:
        reasons.append(f"progress_outside_0_1={float(outside):.3f}")
    if pd.notna(dominant) and float(dominant) > 0.50:
        reasons.append(f"dominant_component_fraction={float(dominant):.3f}")
    if pd.notna(hetero) and float(hetero) < 0.50:
        reasons.append(f"component_entropy_proxy={float(hetero):.3f}")
    if not reasons:
        reasons.append("no_major_caution_flag_from_available_V7p_diagnostics")
    return ";".join(reasons)


def _build_field_cards(field_foundation: pd.DataFrame, marker_family: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in field_foundation.iterrows():
        field = str(r["field"])
        fmk = marker_family[marker_family["field"].astype(str) == field]
        usable = fmk[fmk["marker_reliability_label"].isin(["usable", "usable_with_caution"])]
        unusable = fmk[~fmk["marker_reliability_label"].isin(["usable", "usable_with_caution"])]
        label = str(r.get("field_curve_foundation_label", "unknown"))
        caution_level = _field_caution_level(r)
        caution_reason = _field_caution_reason(r)
        if label == "valid_single_process_curve" and caution_level == "low_caution":
            transition_type = "single_process_transition_candidate"
        elif label in {"valid_single_process_curve", "usable_with_caution"}:
            transition_type = "process_transition_usable_with_caution"
        elif label == "multi_component_process":
            transition_type = "multi_component_transition_candidate"
        elif label == "usable_only_as_projection_diagnostic":
            transition_type = "projection_diagnostic_only"
        else:
            transition_type = "transition_not_established"
        text = (
            f"{field}: {transition_type}. Foundation={label}; "
            f"prepost={r.get('prepost_foundation_label')}, direction={r.get('transition_direction_label')}, "
            f"projection={r.get('projection_progress_validity_label')}, single_curve={r.get('single_curve_adequacy_label')}; "
            f"caution={caution_level} ({caution_reason})."
        )
        rows.append({
            "field": field,
            "field_transition_type": transition_type,
            "process_curve_status": label,
            "component_heterogeneity_caution": caution_level,
            "component_heterogeneity_caution_reason": caution_reason,
            "usable_marker_family": _join(usable["marker_name"].astype(str).tolist()) if "marker_name" in usable.columns else "none",
            "unusable_marker_family": _join(unusable["marker_name"].astype(str).tolist()) if "marker_name" in unusable.columns else "none",
            "single_curve_warning": "yes" if caution_level != "low_caution" else "no",
            "needs_component_or_spatial_support": bool(caution_level != "low_caution"),
            "process_card_text": text,
            "do_not_overinterpret": "no_causality;no_physical_strength;no_single_marker_order_without_pair_card;component_caution_must_be_retained",
        })
    return pd.DataFrame(rows)


def _has_crossing(pphase: pd.DataFrame) -> bool:
    labels = list(pphase["phase_relation_label"].astype(str)) if not pphase.empty else []
    has_a = any(x.startswith("A_") for x in labels)
    has_b = any(x.startswith("B_") for x in labels)
    return bool(has_a and has_b)


def _marker_family_summary(pmk: pd.DataFrame, fam: str) -> str:
    sub = pmk[pmk["marker_family"].astype(str) == fam] if not pmk.empty and "marker_family" in pmk.columns else pd.DataFrame()
    if sub.empty:
        return "unavailable"
    labs = sub["marker_relation_label"].astype(str).value_counts().to_dict()
    return ";".join(f"{k}:{v}" for k, v in labs.items())


def _marker_name_summary(pmk: pd.DataFrame, name: str) -> str:
    sub = pmk[pmk["marker_name"].astype(str) == name] if not pmk.empty and "marker_name" in pmk.columns else pd.DataFrame()
    if sub.empty:
        return "unavailable"
    r = sub.iloc[0]
    return f"{r.get('marker_relation_label')} median_delta={r.get('median_delta_B_minus_A')} q05={r.get('q05_delta')} q95={r.get('q95_delta')}"


def _early_marker_summary(pmk: pd.DataFrame) -> str:
    sub = pmk[(pmk["marker_family"].astype(str) == "threshold") & (pmk["marker_name"].astype(str).isin(["t10", "t15", "t20", "t25", "t30", "t35", "t40"]))] if not pmk.empty else pd.DataFrame()
    if sub.empty:
        return "unavailable"
    labs = sub["marker_relation_label"].astype(str).value_counts().to_dict()
    return ";".join(f"{k}:{v}" for k, v in labs.items())


def _curve_phase_summary(pphase: pd.DataFrame, a: str, b: str) -> str:
    if pphase.empty:
        return "unavailable"
    parts = []
    for _, r in pphase.iterrows():
        parts.append(f"{int(r['day_start'])}-{int(r['day_end'])}:{r['phase_relation_label']}")
    return ";".join(parts)


def _layer_specific_evidence_summary(pmk: pd.DataFrame) -> dict[str, Any]:
    """Summarize marker-family evidence without upgrading it to global lead.

    This hotfix keeps global clean-lead strict, but makes layer-specific evidence
    visible in cards and organization counts. A pair can have confirmed
    departure/early-progress evidence and still not be a clean global lead.
    """
    if pmk.empty:
        return {
            "has_confirmed_layer_specific_evidence": False,
            "has_tendency_layer_specific_evidence": False,
            "n_confirmed_layer_markers": 0,
            "n_tendency_layer_markers": 0,
            "confirmed_layer_evidence_summary": "none",
            "tendency_layer_evidence_summary": "none",
            "layer_specific_evidence_summary": "none",
        }
    confirmed = pmk[pmk["marker_relation_label"].astype(str).isin(["A_leads_90", "B_leads_90"])]
    tendency = pmk[pmk["marker_relation_label"].astype(str).isin(["A_tendency", "B_tendency"])]

    def summarize(sub: pd.DataFrame) -> str:
        if sub.empty:
            return "none"
        parts: list[str] = []
        for _, r in sub.iterrows():
            direction = "A" if str(r.get("marker_relation_label")).startswith("A_") else "B"
            parts.append(f"{r.get('marker_family')}:{r.get('marker_name')}:{r.get('marker_relation_label')}:{direction}")
        return ";".join(parts)

    csum = summarize(confirmed)
    tsum = summarize(tendency)
    combined = ";".join(x for x in [csum if csum != "none" else "", tsum if tsum != "none" else ""] if x) or "none"
    return {
        "has_confirmed_layer_specific_evidence": bool(not confirmed.empty),
        "has_tendency_layer_specific_evidence": bool(not tendency.empty),
        "n_confirmed_layer_markers": int(len(confirmed)),
        "n_tendency_layer_markers": int(len(tendency)),
        "confirmed_layer_evidence_summary": csum,
        "tendency_layer_evidence_summary": tsum,
        "layer_specific_evidence_summary": combined,
    }


def _build_pair_cards(field_foundation: pd.DataFrame, pair_daily: pd.DataFrame, pair_phase: pd.DataFrame, pair_marker: pd.DataFrame, settings: V7PSettings) -> pd.DataFrame:
    fnd = {str(r["field"]): str(r["field_curve_foundation_label"]) for _, r in field_foundation.iterrows()}
    rows: list[dict[str, Any]] = []
    for a, b in combinations(FIELDS, 2):
        pphase = pair_phase[(pair_phase["field_a"].astype(str) == a) & (pair_phase["field_b"].astype(str) == b)]
        pmk = pair_marker[(pair_marker["field_a"].astype(str) == a) & (pair_marker["field_b"].astype(str) == b)]
        labels = set(pphase["phase_relation_label"].astype(str)) if not pphase.empty else set()
        marker_labels = set(pmk["marker_relation_label"].astype(str)) if not pmk.empty else set()
        a_supported = any(str(x).startswith("A_ahead") for x in labels) or "A_leads_90" in marker_labels
        b_supported = any(str(x).startswith("B_ahead") for x in labels) or "B_leads_90" in marker_labels
        crossing = _has_crossing(pphase)
        same_phase = any("near_equal" in str(x) for x in labels) or "same_day_candidate" in marker_labels
        foundation_ok = all(fnd.get(f, "not_valid") in {"valid_single_process_curve", "usable_with_caution"} for f in [a, b])
        evidence = _layer_specific_evidence_summary(pmk)
        # Global clean lead remains intentionally strict. Layer-specific evidence is
        # reported separately and should not be upgraded to global lead.
        if not foundation_ok:
            global_type = "not_comparable"
        elif crossing and a_supported and b_supported:
            global_type = "front_loaded_vs_catchup"
        elif crossing:
            global_type = "phase_crossing"
        elif same_phase and not a_supported and not b_supported:
            global_type = "same_phase_candidate"
        elif evidence["has_confirmed_layer_specific_evidence"]:
            global_type = "layer_specific_lead"
        elif evidence["has_tendency_layer_specific_evidence"]:
            global_type = "weak_layer_specific_tendency"
        elif marker_labels and len(marker_labels) > 1:
            global_type = "marker_conflict"
        else:
            global_type = "order_not_resolved"
        can_a = bool(global_type == "clean_lead" and a_supported and not b_supported)
        can_b = bool(global_type == "clean_lead" and b_supported and not a_supported)
        dep_rel = _marker_family_summary(pmk, "departure")
        early_rel = _early_marker_summary(pmk)
        peak_rel = _marker_family_summary(pmk, "peak_change")
        mid_rel = _marker_name_summary(pmk, "t50")
        finish_rel = _marker_name_summary(pmk, "t75")
        curve_rel = _curve_phase_summary(pphase, a, b)
        text = (
            f"{a}-{b}: {global_type}. {curve_rel} "
            f"Layer evidence={evidence['layer_specific_evidence_summary']}. "
            f"Departure={dep_rel}; early_progress={early_rel}; peak={peak_rel}; midpoint={mid_rel}; finish={finish_rel}."
        )
        rows.append({
            "field_a": a,
            "field_b": b,
            "can_say_A_leads_B": can_a,
            "can_say_B_leads_A": can_b,
            "must_use_layer_specific_language": bool(global_type != "clean_lead"),
            "can_test_near_equivalence": bool(same_phase or global_type in {"order_not_resolved", "same_phase_candidate"}),
            "global_relation_type": global_type,
            **evidence,
            "departure_relation": dep_rel,
            "early_progress_relation": early_rel,
            "curve_phase_relation": curve_rel,
            "peak_relation": peak_rel,
            "midpoint_relation": mid_rel,
            "finish_tail_relation": finish_rel,
            "artifact_risk_label": "not_evaluated_in_v7_p_clean_trunk",
            "comparability_label": "phase_comparable_with_caution" if foundation_ok else "not_comparable_due_to_field_foundation",
            "relation_card_text": text,
            "do_not_overinterpret": "no_global_lead_lag_unless_clean_lead;layer_specific_evidence_is_not_global_order;no_synchrony_without_equivalence;no_causality",
        })
    return pd.DataFrame(rows)


def _build_window_layers(field_cards: pd.DataFrame, pair_cards: pd.DataFrame, pair_phase: pd.DataFrame, pair_marker: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in field_cards.iterrows():
        rows.append({"layer_name": "field_process_layer", "field_or_pair": r["field"], "organization_label": r["field_transition_type"], "evidence_summary": r["process_card_text"], "interpretation": "field-level transition process card; caution fields must be retained"})
    for _, r in pair_cards.iterrows():
        pair = f"{r['field_a']}-{r['field_b']}"
        rows.append({"layer_name": "pair_relation_layer", "field_or_pair": pair, "organization_label": r["global_relation_type"], "evidence_summary": r["relation_card_text"], "interpretation": "pair relation card; layer-specific unless clean_lead"})
    confirmed = pair_cards[pair_cards.get("has_confirmed_layer_specific_evidence", False) == True] if not pair_cards.empty and "has_confirmed_layer_specific_evidence" in pair_cards.columns else pd.DataFrame()
    tendency = pair_cards[pair_cards.get("has_tendency_layer_specific_evidence", False) == True] if not pair_cards.empty and "has_tendency_layer_specific_evidence" in pair_cards.columns else pd.DataFrame()
    rows.append({
        "layer_name": "confirmed_layer_specific_evidence_layer",
        "field_or_pair": "all_pairs",
        "organization_label": "confirmed_layer_specific_summary",
        "evidence_summary": str({f"{r.field_a}-{r.field_b}": r.confirmed_layer_evidence_summary for _, r in confirmed.iterrows()}),
        "interpretation": "Confirmed marker-family layer evidence; does not imply global lead.",
    })
    rows.append({
        "layer_name": "tendency_layer_specific_evidence_layer",
        "field_or_pair": "all_pairs",
        "organization_label": "tendency_layer_specific_summary",
        "evidence_summary": str({f"{r.field_a}-{r.field_b}": r.tendency_layer_evidence_summary for _, r in tendency.iterrows()}),
        "interpretation": "Tendency marker-family layer evidence; diagnostic only.",
    })
    for layer, fam in [("departure_layer", "departure"), ("peak_change_layer", "peak_change")]:
        sub = pair_marker[pair_marker["marker_family"].astype(str) == fam] if not pair_marker.empty else pd.DataFrame()
        if not sub.empty:
            rows.append({"layer_name": layer, "field_or_pair": "all_pairs", "organization_label": "marker_family_summary", "evidence_summary": str(sub["marker_relation_label"].value_counts().to_dict()), "interpretation": f"{fam} marker-family pairwise relation counts"})
    sub = pair_marker[(pair_marker["marker_family"].astype(str) == "threshold") & (pair_marker["marker_name"].astype(str).isin(["t10", "t15", "t20", "t25", "t30", "t35", "t40"]))] if not pair_marker.empty else pd.DataFrame()
    if not sub.empty:
        rows.append({"layer_name": "early_progress_layer", "field_or_pair": "all_pairs", "organization_label": "early_threshold_summary", "evidence_summary": str(sub["marker_relation_label"].value_counts().to_dict()), "interpretation": "early-progress threshold pairwise relation counts"})
    return pd.DataFrame(rows)


def _build_relation_decision_counts(pair_cards: pd.DataFrame, field_cards: pd.DataFrame) -> pd.DataFrame:
    if pair_cards.empty:
        return pd.DataFrame()
    rows = [{
        "clean_global_lead_pairs": int((pair_cards["global_relation_type"] == "clean_lead").sum()),
        "confirmed_layer_specific_pairs": int(pair_cards.get("has_confirmed_layer_specific_evidence", pd.Series(False, index=pair_cards.index)).sum()),
        "tendency_layer_specific_pairs": int(pair_cards.get("has_tendency_layer_specific_evidence", pd.Series(False, index=pair_cards.index)).sum()),
        "phase_crossing_or_frontloaded_pairs": int(pair_cards["global_relation_type"].isin(["phase_crossing", "front_loaded_vs_catchup"]).sum()),
        "same_phase_candidate_pairs": int((pair_cards["global_relation_type"] == "same_phase_candidate").sum()),
        "not_comparable_pairs": int((pair_cards["global_relation_type"] == "not_comparable").sum()),
        "field_low_caution_count": int((field_cards.get("component_heterogeneity_caution", pd.Series(index=field_cards.index, dtype=object)) == "low_caution").sum()) if not field_cards.empty else 0,
        "field_moderate_or_high_caution_count": int((field_cards.get("component_heterogeneity_caution", pd.Series(index=field_cards.index, dtype=object)) != "low_caution").sum()) if not field_cards.empty else 0,
        "interpretation": "Clean global lead is strict. Confirmed/tendency layer-specific pairs must use layer-specific language and cannot be counted as global order.",
    }]
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Summary / figures
# -----------------------------------------------------------------------------


def _write_window_summary(settings: V7PSettings, state_summary: pd.DataFrame, field_cards: pd.DataFrame, pair_cards: pd.DataFrame, org_layers: pd.DataFrame) -> None:
    clean = pair_cards[pair_cards["global_relation_type"] == "clean_lead"] if not pair_cards.empty else pd.DataFrame()
    confirmed_layer = pair_cards[pair_cards.get("has_confirmed_layer_specific_evidence", pd.Series(False, index=pair_cards.index)) == True] if not pair_cards.empty else pd.DataFrame()
    tendency_layer = pair_cards[pair_cards.get("has_tendency_layer_specific_evidence", pd.Series(False, index=pair_cards.index)) == True] if not pair_cards.empty else pd.DataFrame()
    crossing = pair_cards[pair_cards["global_relation_type"].isin(["phase_crossing", "front_loaded_vs_catchup"])] if not pair_cards.empty else pd.DataFrame()
    same = pair_cards[pair_cards["global_relation_type"] == "same_phase_candidate"] if not pair_cards.empty else pd.DataFrame()
    notcomp = pair_cards[pair_cards["global_relation_type"] == "not_comparable"] if not pair_cards.empty else pd.DataFrame()
    caution = field_cards[field_cards.get("component_heterogeneity_caution", pd.Series("", index=field_cards.index)) != "low_caution"] if not field_cards.empty else pd.DataFrame()
    lines = [
        "# W45 process-relation rebuild v7_p hotfix_01",
        "",
        "## Purpose",
        "This branch rebuilds W45 field processes and pair relations from the current 2-degree interpolated base representation. It does not read V7-m/V7-n/V7-o derived outputs as input.",
        "",
        "## Hotfix 01 changes",
        "- Separates clean global lead from confirmed/tendency layer-specific evidence.",
        "- Adds layer-specific evidence summaries to pair cards.",
        "- Adds component/curve caution fields to field process cards so single-curve labels are not over-read.",
        "",
        "## Input base",
        state_summary.to_markdown(index=False) if not state_summary.empty else "State summary unavailable.",
        "",
        "## Field process cards",
        field_cards.to_markdown(index=False) if not field_cards.empty else "No field cards generated.",
        "",
        "## Field caution summary",
        caution[["field", "component_heterogeneity_caution", "component_heterogeneity_caution_reason"]].to_markdown(index=False) if not caution.empty and all(c in caution.columns for c in ["field", "component_heterogeneity_caution", "component_heterogeneity_caution_reason"]) else "No moderate/high component caution fields from available V7-p diagnostics.",
        "",
        "## Pair relation cards",
        pair_cards.to_markdown(index=False) if not pair_cards.empty else "No pair cards generated.",
        "",
        "## Organization counts",
        f"clean_global_lead_pairs: {len(clean)}",
        f"confirmed_layer_specific_pairs: {len(confirmed_layer)}",
        f"tendency_layer_specific_pairs: {len(tendency_layer)}",
        f"phase_crossing_or_frontloaded_pairs: {len(crossing)}",
        f"same_phase_candidate_pairs: {len(same)}",
        f"not_comparable_pairs: {len(notcomp)}",
        "",
        "## Layer-specific evidence guardrail",
        "Confirmed layer-specific evidence is preserved, but it is not a clean global lead. It must be reported as departure-layer, early-progress-layer, peak-layer, midpoint-layer, or finish/tail-layer evidence.",
        "",
        "## What W45 can support under V7-p hotfix_01",
        "- Field process descriptions and pair relation cards from a clean base rebuild.",
        "- Layer-specific / phase-specific relation language where the card allows it.",
        "- Explicit marking of non-comparable, phase-crossing, marker-conflict, and layer-specific relations.",
        "",
        "## What W45 cannot support under V7-p hotfix_01",
        "- A global clean lead-lag chain unless a pair card explicitly says clean_lead.",
        "- Treating layer-specific evidence as global order.",
        "- Treating t25 as physical onset.",
        "- Treating progress difference as physical strength, causality, or raw variable magnitude.",
        "- Treating not-resolved as synchrony without equivalence testing.",
    ]
    _write_text("\n".join(lines), settings.output_dir / "w45_window_organization_card_v7_p.md")


def _try_write_figures(settings: V7PSettings, field_curves: pd.DataFrame, pair_daily: pd.DataFrame, pair_phase: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        _ensure_dir(settings.figure_dir)
        # Field progress curves.
        fig, ax = plt.subplots(figsize=(9, 5))
        for field in FIELDS:
            sub = field_curves[field_curves["field"].astype(str) == field].sort_values("day")
            if sub.empty:
                continue
            ax.plot(sub["day"], sub["projection_progress_observed"], label=field)
        for th in [0.25, 0.50, 0.75]:
            ax.axhline(th, linestyle="--", linewidth=0.8)
        ax.axvline(settings.anchor_day, linewidth=1.0)
        ax.set_title("W45 field projection progress curves (V7-p rebuild)")
        ax.set_xlabel("day index")
        ax.set_ylabel("projection progress toward post state")
        ax.legend(ncol=5)
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_field_projection_progress_curves_v7_p.png", dpi=160)
        plt.close(fig)

        # Projection vs distance progress.
        fig, ax = plt.subplots(figsize=(9, 5))
        for field in FIELDS:
            sub = field_curves[field_curves["field"].astype(str) == field].sort_values("day")
            if sub.empty:
                continue
            ax.plot(sub["day"], sub["distance_progress_observed"], label=f"{field} distance")
        ax.axvline(settings.anchor_day, linewidth=1.0)
        ax.set_title("W45 distance-progress curves (V7-p rebuild)")
        ax.set_xlabel("day index")
        ax.set_ylabel("distance progress")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_field_projection_vs_distance_progress_v7_p.png", dpi=160)
        plt.close(fig)

        # Pairwise heatmap median diff.
        piv = pair_daily.pivot_table(index=["field_a", "field_b"], columns="day", values="projection_diff_median", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto")
        ax.set_yticks(range(len(piv.index)), [f"{a}-{b}" for a, b in piv.index])
        ax.set_xticks(range(len(piv.columns)), [str(int(x)) for x in piv.columns], rotation=90, fontsize=7)
        ax.set_title("W45 pairwise projection-progress difference median")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_pairwise_curve_diff_heatmap_v7_p.png", dpi=160)
        plt.close(fig)

        # H/Jw detail.
        fig, ax = plt.subplots(figsize=(9, 5))
        for field in ["H", "Jw"]:
            sub = field_curves[field_curves["field"].astype(str) == field].sort_values("day")
            if sub.empty:
                continue
            ax.plot(sub["day"], sub["projection_progress_observed"], label=f"{field} projection")
        hjs = pair_daily[((pair_daily["field_a"] == "H") & (pair_daily["field_b"] == "Jw")) | ((pair_daily["field_a"] == "Jw") & (pair_daily["field_b"] == "H"))].copy()
        if not hjs.empty:
            hjs = hjs.sort_values("day")
            sign = 1.0 if str(hjs["field_a"].iloc[0]) == "H" else -1.0
            ax2 = ax.twinx()
            ax2.plot(hjs["day"], sign * hjs["projection_diff_median"], linestyle=":", label="H-Jw diff median")
            ax2.axhline(0, linewidth=0.8)
            ax2.set_ylabel("H-Jw progress difference")
        ax.axvline(settings.anchor_day, linewidth=1.0)
        ax.set_title("W45 H/Jw process relation detail")
        ax.set_xlabel("day index")
        ax.set_ylabel("projection progress")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_H_Jw_process_relation_detail_v7_p.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        _write_text(f"Plot warning: {exc}", settings.output_dir / "plot_warning_v7_p.txt")


def _write_update_log(settings: V7PSettings) -> None:
    txt = f"""# UPDATE_LOG_V7_P

created_at: {_now_iso()}

## Patch
- Added `w45_process_relation_rebuild_v7_p` as a clean W45 process-relation trunk.
- This branch rebuilds from the current 2-degree interpolated base representation.
- It does not read V7-m / V7-n / V7-o derived outputs as input.

## Scope
- Window: W002 / anchor day 45.
- Fields: P, V, H, Je, Jw.
- Outputs field process cards, pair relation cards, and a W45 organization card.

## Interpretation guardrails
- t25 is early_progress_day_25, not physical onset.
- Progress is relative pre-to-post transition progress, not physical strength or causality.
- Clean lead is allowed only if a pair card explicitly says clean_lead.
- Phase-crossing, marker conflict, and not-comparable relations must be retained.
"""
    _write_text(txt, settings.v7_root / "UPDATE_LOG_V7_P.md")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def run_w45_process_relation_rebuild_v7_p(v7_root: Optional[Path] = None) -> None:
    settings, base_settings = _resolve_settings(v7_root)
    _ensure_dir(settings.output_dir)
    _ensure_dir(settings.log_dir)
    try:
        _build_all_from_base(settings, base_settings)
    except Exception as exc:  # noqa: BLE001
        error = {
            "version": OUTPUT_TAG,
            "created_at": _now_iso(),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "read_previous_derived_results": False,
            "v7_m_outputs_used_as_input": False,
            "v7_n_outputs_used_as_input": False,
            "v7_o_outputs_used_as_input": False,
        }
        _write_json(error, settings.output_dir / "run_error_meta.json")
        _write_text(str(exc), settings.log_dir / "run_errors_v7_p.log")
        raise


if __name__ == "__main__":
    run_w45_process_relation_rebuild_v7_p()
