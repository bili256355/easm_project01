"""
V8 state-relation layer, built on the V8 peak-only clean baseline.

Boundary
--------
This module adds *state only* to V8. It does not compute or interpret growth,
rollback, multi-stage growth, process_a labels, or final causal/process claims.

Core design
-----------
- Reuse the original V7 pre-post state definitions for S_dist and S_pattern.
- Use pairwise ΔS_AB(t) as the state-relation object.
- Split ΔS_AB(t) into atomic same-sign segments, then classify segments as
  A-dominant, B-dominant, near, or uncertain.
- Do not use fixed |ΔS| strength thresholds. Segment strength is compared against
  a same-object reproducibility null from same-size bootstrap replicate pairs.
- Near blocks and uncertain blocks are aggregated from classified segments; they
  are not proven by signed cancellation over mixed positive/negative segments.
- Paired-year bootstrap is used for segment/block stability, with overlap-based
  matching rather than exact boundary matching.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

V8_STATE_VERSION = "v8_state_relation_a_hotfix01"
OUTPUT_TAG = "state_relation_v8_a"
V8_PEAK_TAG = "peak_only_v8_a"
V7_HOTFIX06_TAG = "accepted_windows_multi_object_prepost_v7_z_multiwin_a_hotfix06_w45_profile_order"
BRANCH_COLS = {"dist": "S_dist", "pattern": "S_pattern"}
OBJECT_ORDER = ["P", "V", "H", "Je", "Jw"]
EPS = 1.0e-12


def _stage_root_from_this_file() -> Path:
    # .../stage_partition/V8/src/stage_partition_v8/state_relation_v8_a.py
    return Path(__file__).resolve().parents[3]


def _import_v7_module(stage_root: Optional[Path] = None):
    stage_root = stage_root or _stage_root_from_this_file()
    v7_src = stage_root / "V7" / "src"
    if not v7_src.exists():
        raise FileNotFoundError(
            f"Cannot find V7 source directory: {v7_src}. "
            "V8 state relation extraction requires the existing V7 code tree."
        )
    if str(v7_src) not in sys.path:
        sys.path.insert(0, str(v7_src))
    from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as v7multi
    return v7multi


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


def _default_smoothed_path(v8_root: Path) -> Path:
    return v8_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _make_v7_cfg_for_v8_state(v7multi) -> object:
    cfg = v7multi.MultiWinConfig.from_env()
    if os.environ.get("V8_STATE_ACCEPTED_WINDOW_REGISTRY"):
        cfg.accepted_window_registry = os.environ["V8_STATE_ACCEPTED_WINDOW_REGISTRY"]
        cfg.window_source = "registry"
    if os.environ.get("V8_STATE_WINDOW_SOURCE"):
        cfg.window_source = os.environ["V8_STATE_WINDOW_SOURCE"].strip().lower()
    if os.environ.get("V8_STATE_SMOOTHED_FIELDS"):
        cfg.smoothed_fields_path = os.environ["V8_STATE_SMOOTHED_FIELDS"]
    if os.environ.get("V8_STATE_N_BOOTSTRAP"):
        cfg.bootstrap_n = int(os.environ["V8_STATE_N_BOOTSTRAP"])
    if os.environ.get("V8_STATE_DEBUG_N_BOOTSTRAP"):
        cfg.bootstrap_n = int(os.environ["V8_STATE_DEBUG_N_BOOTSTRAP"])
    if os.environ.get("V8_STATE_WINDOW_MODE"):
        cfg.window_mode = os.environ["V8_STATE_WINDOW_MODE"].strip().lower()
    if os.environ.get("V8_STATE_TARGET_WINDOWS"):
        cfg.target_windows = os.environ["V8_STATE_TARGET_WINDOWS"].strip()
    if os.environ.get("V8_STATE_LOG_EVERY_BOOTSTRAP"):
        cfg.log_every_bootstrap = int(os.environ["V8_STATE_LOG_EVERY_BOOTSTRAP"])

    # State-relation v8-a is intentionally profile-state only.
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    return cfg


def _state_settings_from_env() -> dict:
    def _int(name: str, default: int) -> int:
        return int(os.environ.get(name, default))

    def _float(name: str, default: float) -> float:
        return float(os.environ.get(name, default))

    return {
        "null_n": _int("V8_STATE_NULL_N", _int("V8_STATE_DEBUG_N_NULL", 0) or _int("V8_STATE_N_BOOTSTRAP", _int("V8_STATE_DEBUG_N_BOOTSTRAP", 1000))),
        "min_atomic_segment_len": _int("V8_STATE_MIN_ATOMIC_SEGMENT_LEN", 3),
        "min_near_block_len": _int("V8_STATE_MIN_NEAR_BLOCK_LEN", 3),
        "max_short_gap_for_merge": _int("V8_STATE_MAX_SHORT_GAP_FOR_MERGE", 1),
        "min_overlap_ratio": _float("V8_STATE_MIN_OVERLAP_RATIO", 0.50),
        "support_supported": _float("V8_STATE_SUPPORT_SUPPORTED", 0.95),
        "support_tendency": _float("V8_STATE_SUPPORT_TENDENCY", 0.90),
        "support_exploratory": _float("V8_STATE_SUPPORT_EXPLORATORY", 0.80),
        "random_seed_offset": _int("V8_STATE_NULL_RANDOM_SEED_OFFSET", 9137),
        "write_null_daily": os.environ.get("V8_STATE_WRITE_NULL_DAILY", "0") == "1",
    }


def _support_class(p: float, settings: dict) -> str:
    if not np.isfinite(p):
        return "unavailable"
    if p >= settings["support_supported"]:
        return "supported"
    if p >= settings["support_tendency"]:
        return "tendency"
    if p >= settings["support_exploratory"]:
        return "exploratory_signal"
    return "unresolved"


def _nanmean(a, axis=None):
    """nanmean that suppresses all-NaN RuntimeWarning.

    Boundary-smoothed fields legitimately contain all-NaN days at the season
    start/end. Those days are audited explicitly later; they should not flood
    the run with warnings during intermediate state reconstruction.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            return np.nanmean(a, axis=axis)


def _nanmax_abs_safe(a) -> float:
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.max(np.abs(x))) if x.size else float("nan")


def _safe_quantile(arr: Sequence[float], q: float) -> float:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def _curve_matrix_from_state(state_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str, str], np.ndarray], np.ndarray]:
    """Return mapping (object, baseline, branch) -> array over season days."""
    if state_df.empty:
        return {}, np.arange(0, 183)
    max_day = int(np.nanmax(state_df["day"].to_numpy(float)))
    days = np.arange(0, max_day + 1, dtype=int)
    out: Dict[Tuple[str, str, str], np.ndarray] = {}
    for (obj, base), g in state_df.groupby(["object", "baseline_config"]):
        gg = g.sort_values("day")
        day_vals = gg["day"].to_numpy(int)
        for branch, col in BRANCH_COLS.items():
            arr = np.full(days.shape, np.nan, dtype=float)
            if col in gg.columns:
                vals = gg[col].to_numpy(float)
                valid = (day_vals >= 0) & (day_vals < len(arr))
                arr[day_vals[valid]] = vals[valid]
            out[(str(obj), str(base), branch)] = arr
    return out, days


def _compute_state_for_profiles(v7multi, profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], scope, cfg, idx: Optional[np.ndarray] = None) -> pd.DataFrame:
    ccfg = v7multi._clean_cfg_for_window(scope, cfg)
    valid_bases = [b.to_clean() for b in scope.baselines() if b.is_valid]
    rows = []
    for obj, (prof_by_year, _lat, weights) in profiles.items():
        arr = prof_by_year[idx] if idx is not None else prof_by_year
        clim = _nanmean(arr, axis=0)
        # V7 state code may call nanmean internally; suppress boundary all-NaN
        # warnings here and expose those days through explicit validity audits.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            st, _gr, _br = v7multi.clean._compute_state_growth_for_object(obj, clim, weights, valid_bases, ccfg)
        rows.append(st)
    state = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not state.empty:
        state.insert(0, "window_id", scope.window_id)
    return state


def _make_profiles(v7multi, fields: dict, lat: np.ndarray, lon: np.ndarray, years) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], pd.DataFrame]:
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for spec in v7multi.clean.OBJECT_SPECS:
        arr = v7multi.clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (prof, target_lat, weights)
        rows.append({
            **asdict(spec),
            "profile_shape": str(prof.shape),
            "target_lat_min": float(np.nanmin(target_lat)),
            "target_lat_max": float(np.nanmax(target_lat)),
            "v8_role": "state_relation_profile_input",
        })
    return profiles, pd.DataFrame(rows)


def _pairwise_delta_curves(state_df: pd.DataFrame) -> pd.DataFrame:
    mat, days = _curve_matrix_from_state(state_df)
    objects = [o for o in OBJECT_ORDER if any(k[0] == o for k in mat)]
    bases = sorted({k[1] for k in mat})
    rows = []
    for base in bases:
        for branch in BRANCH_COLS:
            for i, a in enumerate(objects):
                for b in objects[i + 1:]:
                    va = mat.get((a, base, branch))
                    vb = mat.get((b, base, branch))
                    if va is None or vb is None:
                        continue
                    delta = va - vb
                    for d, x in zip(days, delta):
                        rows.append({
                            "object_A": a,
                            "object_B": b,
                            "baseline_config": base,
                            "branch": branch,
                            "day": int(d),
                            "delta_S": float(x) if np.isfinite(x) else np.nan,
                        })
    return pd.DataFrame(rows)




def _run_lengths_from_mask(days: np.ndarray, mask: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return finite/valid runs as (start_day, end_day, length)."""
    runs: List[Tuple[int, int, int]] = []
    start: Optional[int] = None
    for i, ok in enumerate(mask.astype(bool)):
        if ok and start is None:
            start = i
        elif (not ok) and start is not None:
            runs.append((int(days[start]), int(days[i - 1]), int(i - start)))
            start = None
    if start is not None:
        runs.append((int(days[start]), int(days[len(mask) - 1]), int(len(mask) - start)))
    return runs


def _validity_counts(days: np.ndarray, finite: np.ndarray, nominal_start: int, nominal_end: int) -> dict:
    sel = (days >= int(nominal_start)) & (days <= int(nominal_end))
    d = days[sel]
    f = finite[sel].astype(bool)
    if d.size == 0:
        return {
            "nominal_start_day": int(nominal_start),
            "nominal_end_day": int(nominal_end),
            "n_nominal_days": 0,
            "n_finite_days": 0,
            "finite_start_day": np.nan,
            "finite_end_day": np.nan,
            "leading_nan_days": 0,
            "trailing_nan_days": 0,
            "internal_nan_days": 0,
            "valid_day_fraction": np.nan,
            "valid_runs": "",
        }
    finite_idx = np.where(f)[0]
    if finite_idx.size:
        first = int(finite_idx[0]); last = int(finite_idx[-1])
        finite_start = int(d[first]); finite_end = int(d[last])
        leading = first
        trailing = int(d.size - 1 - last)
        internal = int((~f[first:last + 1]).sum())
    else:
        finite_start = np.nan; finite_end = np.nan
        leading = int(d.size); trailing = int(d.size); internal = 0
    runs = _run_lengths_from_mask(d, f)
    return {
        "nominal_start_day": int(nominal_start),
        "nominal_end_day": int(nominal_end),
        "n_nominal_days": int(d.size),
        "n_finite_days": int(f.sum()),
        "finite_start_day": finite_start,
        "finite_end_day": finite_end,
        "leading_nan_days": int(leading),
        "trailing_nan_days": int(trailing),
        "internal_nan_days": int(internal),
        "valid_day_fraction": float(f.mean()) if d.size else np.nan,
        "valid_runs": ";".join([f"{a}-{b}({n})" for a, b, n in runs]),
    }


def _profile_validity_audit(profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], scope) -> pd.DataFrame:
    """Audit boundary NaNs in profile inputs and effective pre/post reference days."""
    rows: List[dict] = []
    for obj, (prof_by_year, _lat, _weights) in profiles.items():
        # A day is usable if at least one year/lat element is finite. Boundary
        # smoothing usually makes the first/last few days all-NaN for all years.
        day_finite = np.isfinite(prof_by_year).any(axis=(0, 2))
        days = np.arange(day_finite.size, dtype=int)
        base_row = {"object": obj, "source": "profile_by_year", "branch": "input_profile"}
        rows.append({**base_row, "range_type": "analysis", **_validity_counts(days, day_finite, int(scope.analysis_start), int(scope.analysis_end))})
        for b in scope.baselines():
            rows.append({**base_row, "baseline_config": b.name, "range_type": "pre_reference", **_validity_counts(days, day_finite, int(b.pre_start), int(b.pre_end))})
            rows.append({**base_row, "baseline_config": b.name, "range_type": "post_reference", **_validity_counts(days, day_finite, int(b.post_start), int(b.post_end))})
    return pd.DataFrame(rows)


def _state_valid_day_audit(state_df: pd.DataFrame, scope) -> pd.DataFrame:
    rows: List[dict] = []
    if state_df.empty:
        return pd.DataFrame()
    for (obj, base), g in state_df.groupby(["object", "baseline_config"]):
        gg = g.sort_values("day")
        days = gg["day"].to_numpy(int)
        for branch, col in BRANCH_COLS.items():
            finite = pd.to_numeric(gg[col], errors="coerce").notna().to_numpy(bool) if col in gg.columns else np.zeros(days.shape, dtype=bool)
            rows.append({
                "object": obj,
                "baseline_config": base,
                "branch": branch,
                "range_type": "analysis_state_curve",
                **_validity_counts(days, finite, int(scope.analysis_start), int(scope.analysis_end)),
            })
    return pd.DataFrame(rows)


def _pairwise_delta_valid_domain_audit(delta_df: pd.DataFrame, scope) -> pd.DataFrame:
    rows: List[dict] = []
    if delta_df.empty:
        return pd.DataFrame()
    for (a, b, base, branch), g in delta_df.groupby(["object_A", "object_B", "baseline_config", "branch"]):
        gg = g.sort_values("day")
        days = gg["day"].to_numpy(int)
        finite = pd.to_numeric(gg["delta_S"], errors="coerce").notna().to_numpy(bool)
        rows.append({
            "object_A": a,
            "object_B": b,
            "baseline_config": base,
            "branch": branch,
            "range_type": "pairwise_common_valid_domain",
            **_validity_counts(days, finite, int(scope.analysis_start), int(scope.analysis_end)),
            "note": "segments are built only over finite runs; boundary NaNs are audited, not interpreted as relation gaps",
        })
    return pd.DataFrame(rows)

def _segment_sign(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"


def _raw_segments_for_curve(days: np.ndarray, delta: np.ndarray) -> List[dict]:
    segs: List[dict] = []
    current_sign: Optional[str] = None
    start_idx: Optional[int] = None
    for i, (d, x) in enumerate(zip(days, delta)):
        s = _segment_sign(float(x))
        if s == "nan":
            if current_sign is not None and start_idx is not None:
                segs.append(_segment_record(days, delta, start_idx, i - 1, current_sign))
            current_sign, start_idx = None, None
            continue
        if current_sign is None:
            current_sign, start_idx = s, i
        elif s != current_sign:
            segs.append(_segment_record(days, delta, start_idx, i - 1, current_sign))
            current_sign, start_idx = s, i
    if current_sign is not None and start_idx is not None:
        segs.append(_segment_record(days, delta, start_idx, len(days) - 1, current_sign))
    return segs


def _segment_record(days: np.ndarray, delta: np.ndarray, i0: int, i1: int, sign: str) -> dict:
    vals = delta[i0:i1 + 1].astype(float)
    finite = vals[np.isfinite(vals)]
    D = float(np.nanmean(finite)) if finite.size else float("nan")
    E = float(abs(D)) if np.isfinite(D) else float("nan")
    return {
        "raw_start_idx": int(i0),
        "raw_end_idx": int(i1),
        "start_day": int(days[i0]),
        "end_day": int(days[i1]),
        "duration": int(i1 - i0 + 1),
        "raw_sign_type": sign,
        "D_mean": D,
        "E_abs_mean": E,
        "signed_integral": float(np.nansum(vals)),
        "abs_integral": float(np.nansum(np.abs(vals))),
        "peak_abs_delta": float(np.nanmax(np.abs(vals))) if finite.size else float("nan"),
    }


def _segment_null_values(null_arrays: Dict[Tuple[str, str, str], np.ndarray], obj: str, base: str, branch: str, i0: int, i1: int) -> np.ndarray:
    arr = null_arrays.get((obj, base, branch))
    if arr is None:
        return np.asarray([], dtype=float)
    vals = arr[:, i0:i1 + 1]
    m = _nanmean(vals, axis=1)
    return np.abs(m[np.isfinite(m)])


def _pair_null_values(null_arrays: Dict[Tuple[str, str, str], np.ndarray], a: str, b: str, base: str, branch: str, i0: int, i1: int) -> np.ndarray:
    na = _segment_null_values(null_arrays, a, base, branch, i0, i1)
    nb = _segment_null_values(null_arrays, b, base, branch, i0, i1)
    if na.size and nb.size:
        return np.concatenate([na, nb])
    if na.size:
        return na
    return nb


def _classify_segment(row: dict, null_values: np.ndarray, settings: dict) -> dict:
    sign = row["raw_sign_type"]
    E = float(row["E_abs_mean"])
    duration = int(row["duration"])
    valid_null = null_values[np.isfinite(null_values)]
    if duration < settings["min_atomic_segment_len"]:
        base = "short_uncertain"
        bias = sign
        return {**row, "classified_type": base, "directional_bias": bias, "P_near_null": np.nan, "P_separation_gt_null": np.nan, "support_basis": "too_short_for_primary_segment"}
    if not np.isfinite(E) or valid_null.size == 0:
        return {**row, "classified_type": "uncertain", "directional_bias": sign, "P_near_null": np.nan, "P_separation_gt_null": np.nan, "support_basis": "missing_null_or_metric"}
    # Monte Carlo comparison: Is this segment's average separation within same-object reproducibility?
    p_near = float(np.mean(E <= valid_null))
    p_sep = float(np.mean(E > valid_null))
    if p_near >= settings["support_supported"]:
        ctype = "near_segment"
    elif sign == "positive" and p_sep >= settings["support_supported"]:
        ctype = "A_dominant_segment"
    elif sign == "negative" and p_sep >= settings["support_supported"]:
        ctype = "B_dominant_segment"
    elif p_near >= settings["support_tendency"]:
        ctype = "near_segment_tendency"
    elif sign == "positive" and p_sep >= settings["support_tendency"]:
        ctype = "A_dominant_segment_tendency"
    elif sign == "negative" and p_sep >= settings["support_tendency"]:
        ctype = "B_dominant_segment_tendency"
    else:
        ctype = "uncertain_segment"
    return {**row, "classified_type": ctype, "directional_bias": sign, "P_near_null": p_near, "P_separation_gt_null": p_sep, "support_basis": "same_object_reproducibility_null"}


def _general_type(classified_type: str) -> str:
    if str(classified_type).startswith("A_dominant"):
        return "A_dominant"
    if str(classified_type).startswith("B_dominant"):
        return "B_dominant"
    if str(classified_type).startswith("near"):
        return "near"
    return "uncertain"


def _observed_segments(delta_df: pd.DataFrame, null_arrays: Dict[Tuple[str, str, str], np.ndarray], settings: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: List[dict] = []
    cls_rows: List[dict] = []
    seg_counter = 0
    for (a, b, base, branch), g in delta_df.groupby(["object_A", "object_B", "baseline_config", "branch"]):
        gg = g.sort_values("day")
        days = gg["day"].to_numpy(int)
        delta = gg["delta_S"].to_numpy(float)
        raw = _raw_segments_for_curve(days, delta)
        for r in raw:
            seg_counter += 1
            rr = {
                "segment_id": f"SEG{seg_counter:05d}",
                "object_A": a,
                "object_B": b,
                "baseline_config": base,
                "branch": branch,
                **r,
            }
            raw_rows.append(rr)
            nv = _pair_null_values(null_arrays, a, b, base, branch, int(r["raw_start_idx"]), int(r["raw_end_idx"]))
            cr = _classify_segment(rr, nv, settings)
            cr.update({
                "null_n": int(nv.size),
                "null_E_median": _safe_quantile(nv, 0.50),
                "null_E_q025": _safe_quantile(nv, 0.025),
                "null_E_q975": _safe_quantile(nv, 0.975),
            })
            cls_rows.append(cr)
    return pd.DataFrame(raw_rows), pd.DataFrame(cls_rows)


def _compute_null_arrays(v7multi, profiles, scope, cfg, settings: dict) -> Dict[Tuple[str, str, str], np.ndarray]:
    ny = next(iter(profiles.values()))[0].shape[0]
    n_null = int(settings["null_n"])
    rng = np.random.default_rng(int(cfg.random_seed) + int(scope.anchor_day) + int(settings["random_seed_offset"]))
    example_state = _compute_state_for_profiles(v7multi, profiles, scope, cfg)
    mat0, days = _curve_matrix_from_state(example_state)
    null_arrays: Dict[Tuple[str, str, str], np.ndarray] = {k: np.full((n_null, len(days)), np.nan, dtype=float) for k in mat0}
    for ir in range(n_null):
        idx1 = rng.integers(0, ny, size=ny)
        idx2 = rng.integers(0, ny, size=ny)
        st1 = _compute_state_for_profiles(v7multi, profiles, scope, cfg, idx1)
        st2 = _compute_state_for_profiles(v7multi, profiles, scope, cfg, idx2)
        m1, _ = _curve_matrix_from_state(st1)
        m2, _ = _curve_matrix_from_state(st2)
        for k in null_arrays:
            if k in m1 and k in m2:
                null_arrays[k][ir, :] = m1[k] - m2[k]
        if cfg.log_every_bootstrap > 0 and (ir + 1) % cfg.log_every_bootstrap == 0:
            _log(f"    same-object reproducibility null {scope.window_id}: {ir + 1}/{n_null}")
    return null_arrays


def _null_segment_summary(obs_segments: pd.DataFrame, null_arrays: Dict[Tuple[str, str, str], np.ndarray]) -> pd.DataFrame:
    rows = []
    for _, r in obs_segments.iterrows():
        a, b, base, branch = r["object_A"], r["object_B"], r["baseline_config"], r["branch"]
        i0, i1 = int(r["raw_start_idx"]), int(r["raw_end_idx"])
        for obj in [a, b]:
            vals = _segment_null_values(null_arrays, obj, base, branch, i0, i1)
            rows.append({
                "segment_id": r["segment_id"],
                "object_A": a,
                "object_B": b,
                "null_object": obj,
                "baseline_config": base,
                "branch": branch,
                "start_day": int(r["start_day"]),
                "end_day": int(r["end_day"]),
                "duration": int(r["duration"]),
                "null_metric": "abs_mean_self_replicate_delta",
                "null_n": int(vals.size),
                "null_median": _safe_quantile(vals, 0.50),
                "null_q025": _safe_quantile(vals, 0.025),
                "null_q975": _safe_quantile(vals, 0.975),
            })
    return pd.DataFrame(rows)


def _aggregate_blocks(segments: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    block_id = 0
    for keys, g in segments.sort_values(["object_A", "object_B", "baseline_config", "branch", "start_day"]).groupby(["object_A", "object_B", "baseline_config", "branch"]):
        a, b, base, branch = keys
        current: List[pd.Series] = []
        current_type: Optional[str] = None

        def flush():
            nonlocal block_id, current, current_type
            if not current:
                return
            block_id += 1
            start = int(current[0]["start_day"]); end = int(current[-1]["end_day"])
            comp_ids = [str(x["segment_id"]) for x in current]
            rows.append({
                "block_id": f"BLK{block_id:05d}",
                "object_A": a,
                "object_B": b,
                "baseline_config": base,
                "branch": branch,
                "block_type": f"{current_type}_block" if current_type in ["near", "uncertain"] else f"{current_type}_block",
                "component_segment_ids": ";".join(comp_ids),
                "start_day": start,
                "end_day": end,
                "duration": end - start + 1,
                "n_components": len(current),
                "block_notes": "aggregated_from_classified_segments; no block-level signed cancellation used",
            })
            current = []
            current_type = None

        for _, r in g.iterrows():
            gt = _general_type(str(r["classified_type"]))
            if current_type is None:
                current = [r]
                current_type = gt
                continue
            prev_end = int(current[-1]["end_day"])
            gap = int(r["start_day"]) - prev_end - 1
            if gt == current_type and gap <= settings["max_short_gap_for_merge"]:
                current.append(r)
            else:
                flush()
                current = [r]
                current_type = gt
        flush()
    return pd.DataFrame(rows)


def _bootstrap_segment_type_counts(observed_segments: pd.DataFrame, boot_segments: pd.DataFrame, settings: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if observed_segments.empty:
        return out
    for _, obs in observed_segments.iterrows():
        obs_id = str(obs["segment_id"])
        obs_type = _general_type(str(obs["classified_type"]))
        obs_start, obs_end = int(obs["start_day"]), int(obs["end_day"])
        obs_len = max(1, obs_end - obs_start + 1)
        g = boot_segments[
            (boot_segments["object_A"] == obs["object_A"]) &
            (boot_segments["object_B"] == obs["object_B"]) &
            (boot_segments["baseline_config"] == obs["baseline_config"]) &
            (boot_segments["branch"] == obs["branch"])
        ]
        counts = {"same_type": 0, "near_instead": 0, "uncertain_instead": 0, "opposite_dominant": 0, "absent_or_fragmented": 0, "matched_start_days": [], "matched_end_days": []}
        if g.empty:
            counts["absent_or_fragmented"] = 1
            out[obs_id] = counts
            continue
        overlaps = []
        for _, br in g.iterrows():
            s, e = int(br["start_day"]), int(br["end_day"])
            ov = max(0, min(obs_end, e) - max(obs_start, s) + 1)
            ratio = ov / obs_len
            overlaps.append((ratio, br))
        overlaps.sort(key=lambda x: x[0], reverse=True)
        best_ratio, best = overlaps[0]
        # split-match: total same-type overlap across split pieces.
        same_overlap = 0
        opposite_overlap = 0
        near_overlap = 0
        uncertain_overlap = 0
        for ratio, br in overlaps:
            if ratio <= 0:
                continue
            gt = _general_type(str(br["classified_type"]))
            ov_days = ratio * obs_len
            if gt == obs_type:
                same_overlap += ov_days
            elif gt == "near":
                near_overlap += ov_days
            elif gt == "uncertain":
                uncertain_overlap += ov_days
            elif (obs_type == "A_dominant" and gt == "B_dominant") or (obs_type == "B_dominant" and gt == "A_dominant"):
                opposite_overlap += ov_days
        same_ratio = same_overlap / obs_len
        if same_ratio >= settings["min_overlap_ratio"] and opposite_overlap == 0:
            counts["same_type"] = 1
            # record best same-type boundaries
            for ratio, br in overlaps:
                if _general_type(str(br["classified_type"])) == obs_type and ratio > 0:
                    counts["matched_start_days"].append(int(br["start_day"]))
                    counts["matched_end_days"].append(int(br["end_day"]))
                    break
        elif opposite_overlap / obs_len >= settings["min_overlap_ratio"]:
            counts["opposite_dominant"] = 1
        elif near_overlap / obs_len >= settings["min_overlap_ratio"]:
            counts["near_instead"] = 1
        elif uncertain_overlap / obs_len >= settings["min_overlap_ratio"]:
            counts["uncertain_instead"] = 1
        elif best_ratio >= settings["min_overlap_ratio"]:
            gt = _general_type(str(best["classified_type"]))
            if gt == "near":
                counts["near_instead"] = 1
            elif gt == "uncertain":
                counts["uncertain_instead"] = 1
            else:
                counts["absent_or_fragmented"] = 1
        else:
            counts["absent_or_fragmented"] = 1
        out[obs_id] = counts
    return out


def _classify_delta_dataframe(delta_df: pd.DataFrame, null_arrays: Dict[Tuple[str, str, str], np.ndarray], settings: dict) -> pd.DataFrame:
    _raw, cls = _observed_segments(delta_df, null_arrays, settings)
    return cls


def _bootstrap_state_segments(v7multi, profiles, scope, cfg, observed_segments: pd.DataFrame, null_arrays, settings: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ny = next(iter(profiles.values()))[0].shape[0]
    boot_indices = v7multi._make_bootstrap_indices(ny, scope, cfg)
    counters: Dict[str, dict] = {
        str(r["segment_id"]): {"same_type": 0, "near_instead": 0, "uncertain_instead": 0, "opposite_dominant": 0, "absent_or_fragmented": 0, "matched_start_days": [], "matched_end_days": []}
        for _, r in observed_segments.iterrows()
    }
    block_rows_all: List[pd.DataFrame] = []
    for ib, idx in enumerate(boot_indices):
        bst = _compute_state_for_profiles(v7multi, profiles, scope, cfg, idx)
        bdelta = _pairwise_delta_curves(bst)
        bcls = _classify_delta_dataframe(bdelta, null_arrays, settings)
        bblocks = _aggregate_blocks(bcls, settings)
        if not bblocks.empty:
            bblocks = bblocks.copy()
            bblocks.insert(0, "bootstrap_id", ib)
            block_rows_all.append(bblocks)
        c = _bootstrap_segment_type_counts(observed_segments, bcls, settings)
        for sid, vals in c.items():
            if sid not in counters:
                continue
            for k in ["same_type", "near_instead", "uncertain_instead", "opposite_dominant", "absent_or_fragmented"]:
                counters[sid][k] += int(vals.get(k, 0))
            counters[sid]["matched_start_days"].extend(vals.get("matched_start_days", []))
            counters[sid]["matched_end_days"].extend(vals.get("matched_end_days", []))
        if cfg.log_every_bootstrap > 0 and (ib + 1) % cfg.log_every_bootstrap == 0:
            _log(f"    paired-year state segment bootstrap {scope.window_id}: {ib + 1}/{cfg.bootstrap_n}")
    rows = []
    n = max(1, len(boot_indices))
    obs_by_id = {str(r["segment_id"]): r for _, r in observed_segments.iterrows()}
    for sid, c in counters.items():
        obs = obs_by_id.get(sid)
        p_same = c["same_type"] / n
        p_near = c["near_instead"] / n
        p_unc = c["uncertain_instead"] / n
        p_opp = c["opposite_dominant"] / n
        p_abs = c["absent_or_fragmented"] / n
        rows.append({
            "segment_id": sid,
            "object_A": obs.get("object_A", "") if obs is not None else "",
            "object_B": obs.get("object_B", "") if obs is not None else "",
            "baseline_config": obs.get("baseline_config", "") if obs is not None else "",
            "branch": obs.get("branch", "") if obs is not None else "",
            "observed_classified_type": obs.get("classified_type", "") if obs is not None else "",
            "start_day": int(obs.get("start_day", -1)) if obs is not None else np.nan,
            "end_day": int(obs.get("end_day", -1)) if obs is not None else np.nan,
            "duration": int(obs.get("duration", -1)) if obs is not None else np.nan,
            "P_same_type_matched": p_same,
            "P_near_instead": p_near,
            "P_uncertain_instead": p_unc,
            "P_opposite_dominant": p_opp,
            "P_absent_or_fragmented": p_abs,
            "support_class": _support_class(p_same, settings),
            "matched_start_day_median": _safe_quantile(c["matched_start_days"], 0.50),
            "matched_start_day_q025": _safe_quantile(c["matched_start_days"], 0.025),
            "matched_start_day_q975": _safe_quantile(c["matched_start_days"], 0.975),
            "matched_end_day_median": _safe_quantile(c["matched_end_days"], 0.50),
            "matched_end_day_q025": _safe_quantile(c["matched_end_days"], 0.025),
            "matched_end_day_q975": _safe_quantile(c["matched_end_days"], 0.975),
            "matching_rule": "same_type_overlap_or_split_match; no exact boundary requirement",
        })
    boot_blocks = pd.concat(block_rows_all, ignore_index=True) if block_rows_all else pd.DataFrame()
    return pd.DataFrame(rows), boot_blocks


def _block_bootstrap_support(observed_blocks: pd.DataFrame, boot_blocks: pd.DataFrame, settings: dict, n_boot: int) -> pd.DataFrame:
    if observed_blocks.empty:
        return pd.DataFrame()
    rows = []
    for _, obs in observed_blocks.iterrows():
        oid = str(obs["block_id"])
        obs_type = str(obs["block_type"])
        start, end = int(obs["start_day"]), int(obs["end_day"])
        length = max(1, end - start + 1)
        gg = boot_blocks[
            (boot_blocks.get("object_A", pd.Series(dtype=object)) == obs["object_A"]) &
            (boot_blocks.get("object_B", pd.Series(dtype=object)) == obs["object_B"]) &
            (boot_blocks.get("baseline_config", pd.Series(dtype=object)) == obs["baseline_config"]) &
            (boot_blocks.get("branch", pd.Series(dtype=object)) == obs["branch"])
        ] if not boot_blocks.empty else pd.DataFrame()
        same = fragmented = opp = uncertain = 0
        for ib in range(n_boot):
            b = gg[gg["bootstrap_id"] == ib] if not gg.empty else pd.DataFrame()
            if b.empty:
                fragmented += 1
                continue
            best_same = 0.0
            best_unc = 0.0
            best_opp = 0.0
            for _, br in b.iterrows():
                ov = max(0, min(end, int(br["end_day"])) - max(start, int(br["start_day"])) + 1) / length
                if ov <= 0:
                    continue
                bt = str(br["block_type"])
                if bt == obs_type:
                    best_same = max(best_same, ov)
                elif "uncertain" in bt:
                    best_unc = max(best_unc, ov)
                elif ("A_dominant" in obs_type and "B_dominant" in bt) or ("B_dominant" in obs_type and "A_dominant" in bt):
                    best_opp = max(best_opp, ov)
            if best_same >= settings["min_overlap_ratio"]:
                same += 1
            elif best_opp >= settings["min_overlap_ratio"]:
                opp += 1
            elif best_unc >= settings["min_overlap_ratio"]:
                uncertain += 1
            else:
                fragmented += 1
        n = max(1, n_boot)
        p_same = same / n
        rows.append({
            "block_id": oid,
            "object_A": obs["object_A"],
            "object_B": obs["object_B"],
            "baseline_config": obs["baseline_config"],
            "branch": obs["branch"],
            "observed_block_type": obs_type,
            "start_day": start,
            "end_day": end,
            "duration": int(obs["duration"]),
            "P_block_same_type": p_same,
            "P_block_fragmented": fragmented / n,
            "P_block_contains_opposite_dominant": opp / n,
            "P_block_degrades_to_uncertain": uncertain / n,
            "support_class": _support_class(p_same, settings),
            "matching_rule": "overlap-based block matching after bootstrap segment aggregation",
        })
    return pd.DataFrame(rows)


def _state_curve_regression_audit(v8_root: Path, state_df: pd.DataFrame, out_cross: Path, window_id: str) -> None:
    stage_root = v8_root.parent
    v7_file = stage_root / "V7" / "outputs" / V7_HOTFIX06_TAG / "per_window" / window_id / f"profile_state_progress_curves_{window_id}.csv"
    if not v7_file.exists():
        _safe_to_csv(pd.DataFrame([{
            "window_id": window_id,
            "status": "v7_state_curve_file_missing",
            "v7_file": str(v7_file),
            "note": "State curve regression audit could not be performed.",
        }]), out_cross / "v8_vs_v7_hotfix06_state_curve_regression_audit.csv")
        return
    try:
        v7 = pd.read_csv(v7_file)
    except Exception as exc:
        _safe_to_csv(pd.DataFrame([{"window_id": window_id, "status": "v7_read_error", "error": str(exc), "v7_file": str(v7_file)}]), out_cross / "v8_vs_v7_hotfix06_state_curve_regression_audit.csv")
        return
    v8 = state_df.copy()
    if "window_id" in v8.columns:
        v8 = v8.drop(columns=["window_id"])
    keys = ["object", "baseline_config", "day"]
    merged = v8.merge(v7, on=keys, how="outer", suffixes=("_v8", "_v7"), indicator=True)
    rows = []
    for col in ["S_dist", "S_pattern", "D_pre", "D_post", "R_diff"]:
        c8, c7 = f"{col}_v8", f"{col}_v7"
        if c8 not in merged.columns or c7 not in merged.columns:
            rows.append({"window_id": window_id, "metric": col, "status": "column_missing"})
            continue
        diff = pd.to_numeric(merged[c8], errors="coerce") - pd.to_numeric(merged[c7], errors="coerce")
        mad = float(np.nanmax(np.abs(diff.to_numpy(float)))) if diff.notna().any() else 0.0
        rows.append({
            "window_id": window_id,
            "metric": col,
            "status": "pass" if mad <= 1.0e-9 and int((merged["_merge"] != "both").sum()) == 0 else "difference_found",
            "max_abs_diff": mad,
            "n_key_mismatch_rows": int((merged["_merge"] != "both").sum()),
            "v7_file": str(v7_file),
        })
    _safe_to_csv(pd.DataFrame(rows), out_cross / "v8_vs_v7_hotfix06_state_curve_regression_audit.csv")


def _multi_object_network(blocks: pd.DataFrame, window_id: str) -> pd.DataFrame:
    if blocks.empty:
        return pd.DataFrame()
    rows = []
    for (base, branch), g in blocks.groupby(["baseline_config", "branch"]):
        near_edges = []
        dom_edges = []
        uncertain_edges = []
        for _, r in g.iterrows():
            edge = f"{r['object_A']}-{r['object_B']}:{int(r['start_day'])}-{int(r['end_day'])}"
            bt = str(r["block_type"])
            if bt.startswith("near"):
                near_edges.append(edge)
            elif bt.startswith("A_dominant"):
                dom_edges.append(f"{r['object_A']}->{r['object_B']}:{int(r['start_day'])}-{int(r['end_day'])}")
            elif bt.startswith("B_dominant"):
                dom_edges.append(f"{r['object_B']}->{r['object_A']}:{int(r['start_day'])}-{int(r['end_day'])}")
            elif "uncertain" in bt:
                uncertain_edges.append(edge)
        rows.append({
            "window_id": window_id,
            "baseline_config": base,
            "branch": branch,
            "near_edges": ";".join(near_edges),
            "dominance_edges": ";".join(dom_edges),
            "uncertain_edges": ";".join(uncertain_edges),
            "network_notes": "observed pairwise block network only; group closure not promoted to scientific claim in v8_state_relation_a",
        })
    return pd.DataFrame(rows)


def _peak_state_comparison(v8_root: Path, window_id: str, blocks: pd.DataFrame) -> pd.DataFrame:
    peak_dir = v8_root / "outputs" / V8_PEAK_TAG / "per_window" / window_id
    peak_order = peak_dir / f"pairwise_peak_order_test_{window_id}.csv"
    sync = peak_dir / f"pairwise_synchrony_equivalence_test_{window_id}.csv"
    if not peak_order.exists():
        return pd.DataFrame([{"window_id": window_id, "status": "peak_only_output_missing", "peak_dir": str(peak_dir)}])
    po = pd.read_csv(peak_order)
    sy = pd.read_csv(sync) if sync.exists() else pd.DataFrame()
    rows = []
    for _, r in po.iterrows():
        a, b = r["object_A"], r["object_B"]
        bg = blocks[(blocks["object_A"] == a) & (blocks["object_B"] == b)] if not blocks.empty else pd.DataFrame()
        state_summary = ";".join([f"{x['baseline_config']}/{x['branch']}:{x['block_type']}@{int(x['start_day'])}-{int(x['end_day'])}" for _, x in bg.iterrows()])
        srow = sy[(sy.get("object_A", pd.Series(dtype=object)) == a) & (sy.get("object_B", pd.Series(dtype=object)) == b)] if not sy.empty else pd.DataFrame()
        rows.append({
            "window_id": window_id,
            "object_A": a,
            "object_B": b,
            "peak_order_decision": r.get("peak_order_decision", r.get("decision", "")),
            "peak_synchrony_decision": srow["synchrony_decision"].iloc[0] if not srow.empty and "synchrony_decision" in srow.columns else "unavailable",
            "state_block_observed_summary": state_summary,
            "allowed_statement": "Peak and state are separate layers; state blocks describe state-progress relation only.",
            "forbidden_statement": "Do not infer growth, causality, or process_a-style catch-up from this table.",
        })
    return pd.DataFrame(rows)


def _write_summary(path: Path, window_id: str, seg_boot: pd.DataFrame, blocks: pd.DataFrame, settings: dict) -> None:
    lines = [
        "# V8 state relation summary",
        "",
        f"version: `{V8_STATE_VERSION}`",
        f"window: `{window_id}`",
        "",
        "## Method boundary",
        "- State only: S_dist / S_pattern and pairwise ΔS_AB(t).",
        "- No growth, rollback, multi-stage growth, process_a labels, or final causal/process claims.",
        "- No fixed |ΔS| strength thresholds are used for near/dominant classification.",
        "- Strength is compared against same-object same-size bootstrap reproducibility nulls.",
        "- Segment duration and matching parameters are construction parameters and must be sensitivity-audited.",
        "- Smoothed-field boundary NaNs are audited explicitly; finite pairwise domains are used for segmentation.",
        "",
        "## Segment construction parameters",
    ]
    for k, v in settings.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Segment bootstrap support counts")
    if not seg_boot.empty and "support_class" in seg_boot.columns:
        for k, v in seg_boot["support_class"].value_counts(dropna=False).items():
            lines.append(f"- {k}: {int(v)}")
    lines.append("")
    lines.append("## Observed block counts")
    if not blocks.empty and "block_type" in blocks.columns:
        for k, v in blocks["block_type"].value_counts(dropna=False).items():
            lines.append(f"- {k}: {int(v)}")
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("- A-dominant/B-dominant/near/uncertain are state-relation segment classes, not peak timing claims.")
    lines.append("- near means within same-object state reconstruction resolution, not perfect physical synchrony.")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_state_relation_v8_a(v8_root: Path | str) -> None:
    v8_root = Path(v8_root)
    stage_root = v8_root.parent
    v7_root = stage_root / "V7"
    v7multi = _import_v7_module(stage_root)
    cfg = _make_v7_cfg_for_v8_state(v7multi)
    settings = _state_settings_from_env()
    if os.environ.get("V8_STATE_DEBUG_N_BOOTSTRAP") and not os.environ.get("V8_STATE_DEBUG_N_NULL") and not os.environ.get("V8_STATE_NULL_N"):
        settings["null_n"] = int(os.environ["V8_STATE_DEBUG_N_BOOTSTRAP"])

    out_root = _ensure_dir(v8_root / "outputs" / OUTPUT_TAG)
    out_cross = _ensure_dir(out_root / "cross_window")
    out_per = _ensure_dir(out_root / "per_window")
    log_dir = _ensure_dir(v8_root / "logs" / OUTPUT_TAG)
    t0 = time.time()

    _log("[1/8] Load windows using original V7 helper")
    wins = v7multi._load_accepted_windows(v7_root, out_cross, cfg)
    scopes, validity = v7multi._build_window_scopes(wins, cfg)
    run_scopes, run_scope_audit = v7multi._filter_scopes_for_run(scopes, cfg)
    _safe_to_csv(pd.DataFrame([asdict(s) for s in scopes]), out_cross / "window_scope_registry_v8_state_relation_a.csv")
    _safe_to_csv(validity, out_cross / "window_scope_validity_audit_v8_state_relation_a.csv")
    _safe_to_csv(run_scope_audit, out_cross / "run_window_selection_audit_v8_state_relation_a.csv")
    _safe_to_csv(pd.DataFrame([asdict(s) for s in run_scopes]), out_cross / "run_window_scope_registry_v8_state_relation_a.csv")

    _log("[2/8] Load smoothed fields and build object profiles")
    smoothed = Path(cfg.smoothed_fields_path) if cfg.smoothed_fields_path else _default_smoothed_path(v8_root)
    if not smoothed.exists():
        raise FileNotFoundError(f"smoothed_fields.npz not found: {smoothed}. Set V8_STATE_SMOOTHED_FIELDS.")
    fields, input_audit = v7multi.clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    _safe_to_csv(input_audit, out_cross / "input_key_audit_v8_state_relation_a.csv")
    profiles, object_registry = _make_profiles(v7multi, fields, lat, lon, years)
    _safe_to_csv(object_registry, out_cross / "object_registry_v8_state_relation_a.csv")

    _write_json({
        "version": V8_STATE_VERSION,
        "output_tag": OUTPUT_TAG,
        "source_v7_module_version": getattr(v7multi, "VERSION", "unknown"),
        "source_v7_output_tag": getattr(v7multi, "OUTPUT_TAG", "unknown"),
        "config": asdict(cfg),
        "state_settings": settings,
        "smoothed_fields": str(smoothed),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "boundary": "state relation only; no growth/process_a/nonmonotonic interpretation",
        "strength_threshold_policy": "no fixed |DeltaS| thresholds; same-object reproducibility null used",
        "boundary_nan_policy": "smoothed-field leading/trailing NaNs are audited explicitly; segments are built only on finite pairwise common valid days",
    }, out_root / "run_meta.json")

    all_seg_boot: List[pd.DataFrame] = []
    all_blocks: List[pd.DataFrame] = []
    processed = 0
    for iw, scope in enumerate(run_scopes, start=1):
        _log(f"[3/8] Process state curves {scope.window_id} ({iw}/{len(run_scopes)})")
        out_win = _ensure_dir(out_per / scope.window_id)
        _safe_to_csv(pd.DataFrame([asdict(scope)]), out_win / f"window_scope_{scope.window_id}.csv")

        _log("  profile/reference boundary-NaN validity audit")
        profile_validity = _profile_validity_audit(profiles, scope)
        profile_validity.insert(0, "window_id", scope.window_id)
        _safe_to_csv(profile_validity, out_win / f"state_profile_reference_validity_audit_{scope.window_id}.csv")

        _log("  observed state curves")
        state = _compute_state_for_profiles(v7multi, profiles, scope, cfg)
        _safe_to_csv(state, out_win / f"object_state_curves_{scope.window_id}.csv")
        state_validity = _state_valid_day_audit(state, scope)
        if not state_validity.empty:
            state_validity.insert(0, "window_id", scope.window_id)
        _safe_to_csv(state_validity, out_win / f"state_valid_day_audit_{scope.window_id}.csv")
        _state_curve_regression_audit(v8_root, state, out_cross, scope.window_id)

        delta = _pairwise_delta_curves(state)
        delta.insert(0, "window_id", scope.window_id)
        _safe_to_csv(delta, out_win / f"pairwise_delta_state_curves_{scope.window_id}.csv")
        delta_no_win = delta.drop(columns=["window_id"])
        pairwise_validity = _pairwise_delta_valid_domain_audit(delta_no_win, scope)
        if not pairwise_validity.empty:
            pairwise_validity.insert(0, "window_id", scope.window_id)
        _safe_to_csv(pairwise_validity, out_win / f"pairwise_delta_state_valid_domain_audit_{scope.window_id}.csv")

        _log("  same-object reproducibility null")
        null_arrays = _compute_null_arrays(v7multi, profiles, scope, cfg, settings)

        _log("  observed segmentation and segment classification")
        raw_segments, cls_segments = _observed_segments(delta_no_win, null_arrays, settings)
        raw_segments.insert(0, "window_id", scope.window_id)
        cls_segments.insert(0, "window_id", scope.window_id)
        _safe_to_csv(raw_segments, out_win / f"pairwise_state_raw_segments_observed_{scope.window_id}.csv")
        _safe_to_csv(cls_segments, out_win / f"pairwise_state_segments_observed_{scope.window_id}.csv")
        null_summary = _null_segment_summary(cls_segments.drop(columns=["window_id"]), null_arrays)
        null_summary.insert(0, "window_id", scope.window_id)
        _safe_to_csv(null_summary, out_win / f"state_reproducibility_null_segment_summary_{scope.window_id}.csv")

        _log("  relation block aggregation")
        blocks = _aggregate_blocks(cls_segments.drop(columns=["window_id"]), settings)
        if not blocks.empty:
            blocks.insert(0, "window_id", scope.window_id)
        _safe_to_csv(blocks, out_win / f"pairwise_state_relation_blocks_{scope.window_id}.csv")

        _log("  paired-year bootstrap segment/block matching")
        seg_boot, boot_blocks = _bootstrap_state_segments(v7multi, profiles, scope, cfg, cls_segments.drop(columns=["window_id"]), null_arrays, settings)
        if not seg_boot.empty:
            seg_boot.insert(0, "window_id", scope.window_id)
        _safe_to_csv(seg_boot, out_win / f"pairwise_state_segment_bootstrap_{scope.window_id}.csv")
        block_boot = _block_bootstrap_support(blocks.drop(columns=["window_id"]) if not blocks.empty else blocks, boot_blocks, settings, int(cfg.bootstrap_n))
        if not block_boot.empty:
            block_boot.insert(0, "window_id", scope.window_id)
        _safe_to_csv(block_boot, out_win / f"pairwise_state_block_bootstrap_{scope.window_id}.csv")

        _log("  multi-object network and peak-state comparison")
        net = _multi_object_network(blocks.drop(columns=["window_id"]) if not blocks.empty else blocks, scope.window_id)
        _safe_to_csv(net, out_win / f"multi_object_state_relation_network_{scope.window_id}.csv")
        ps = _peak_state_comparison(v8_root, scope.window_id, blocks.drop(columns=["window_id"]) if not blocks.empty else blocks)
        _safe_to_csv(ps, out_win / f"peak_state_relation_comparison_{scope.window_id}.csv")
        _write_summary(out_win / f"state_relation_summary_{scope.window_id}.md", scope.window_id, seg_boot, blocks, settings)

        if not seg_boot.empty:
            all_seg_boot.append(seg_boot)
        if not blocks.empty:
            all_blocks.append(blocks)
        processed += 1

    _log("[8/8] Write cross-window summaries")
    seg_all = pd.concat(all_seg_boot, ignore_index=True) if all_seg_boot else pd.DataFrame()
    blk_all = pd.concat(all_blocks, ignore_index=True) if all_blocks else pd.DataFrame()
    _safe_to_csv(seg_all, out_cross / "pairwise_state_segment_bootstrap_all_windows.csv")
    _safe_to_csv(blk_all, out_cross / "pairwise_state_relation_blocks_all_windows.csv")
    _write_json({
        "version": V8_STATE_VERSION,
        "elapsed_seconds": time.time() - t0,
        "n_windows_processed": processed,
        "bootstrap_n": int(cfg.bootstrap_n),
        "null_n": int(settings["null_n"]),
        "output_root": str(out_root),
        "boundary": "state relation only; no growth/process_a/nonmonotonic interpretation",
    }, out_root / "summary.json")
    (log_dir / "last_run.txt").write_text(f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')} output={out_root}\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    run_state_relation_v8_a(Path(__file__).resolve().parents[2])
