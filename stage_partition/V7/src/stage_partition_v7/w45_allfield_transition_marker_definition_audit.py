from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from stage_partition_v6.io import load_smoothed_fields
from stage_partition_v6.state_builder import build_profiles, build_state_matrix

from stage_partition_v7.config import StagePartitionV7Settings
from stage_partition_v7.field_state import FIELDS as V7_FIELDS, build_field_state_matrix_for_year_indices
from stage_partition_v7.field_progress_timing import _compute_progress_for_window

WINDOW_ID = "W002"
ANCHOR_DAY = 45
OUTPUT_TAG = "w45_allfield_transition_marker_definition_audit_v7_m"
FIELDS = ["P", "V", "H", "Je", "Jw"]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75]
EARLY_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


@dataclass
class MarkerDefinitionPaths:
    v7_root: Path
    v7e_output_dir: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path


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


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, cols: Iterable[str], table_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _resolve_paths(v7_root: Optional[Path]) -> MarkerDefinitionPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    return MarkerDefinitionPaths(
        v7_root=v7_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _safe_arr(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_quantile(values: pd.Series | np.ndarray, q: float) -> float:
    arr = _safe_arr(values)
    if arr.size == 0:
        return np.nan
    return float(np.nanquantile(arr, q))


def _safe_median(values: pd.Series | np.ndarray) -> float:
    arr = _safe_arr(values)
    if arr.size == 0:
        return np.nan
    return float(np.nanmedian(arr))


def _valid_fraction(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.isfinite(arr).sum() / arr.size)


def _first_stable_crossing(days: np.ndarray, vals: np.ndarray, threshold: float, stable_days: int) -> float:
    stable_days = max(1, int(stable_days))
    days = np.asarray(days, dtype=int)
    vals = np.asarray(vals, dtype=float)
    for i in range(0, len(vals) - stable_days + 1):
        win = vals[i : i + stable_days]
        if np.all(np.isfinite(win)) and np.all(win >= float(threshold)):
            return float(days[i])
    return np.nan


def _rolling_mean(values: np.ndarray, window: int = 3) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals
    s = pd.Series(vals)
    return s.rolling(window=int(window), center=True, min_periods=1).mean().to_numpy(dtype=float)


def _join(vals: Iterable[object]) -> str:
    out = []
    for v in vals:
        if v is None:
            continue
        s = str(v)
        if not s or s.lower() in {"nan", "none"}:
            continue
        out.append(s)
    return ";".join(out) if out else "none"


def _fmt_threshold(q: float) -> str:
    return f"t{int(round(float(q) * 100)):02d}"


def _marker_stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "median": _safe_median(values),
        "q05": _safe_quantile(values, 0.05),
        "q25": _safe_quantile(values, 0.25),
        "q75": _safe_quantile(values, 0.75),
        "q95": _safe_quantile(values, 0.95),
        "q025": _safe_quantile(values, 0.025),
        "q975": _safe_quantile(values, 0.975),
        "q90_width": _safe_quantile(values, 0.95) - _safe_quantile(values, 0.05),
        "q95_width": _safe_quantile(values, 0.975) - _safe_quantile(values, 0.025),
        "iqr": _safe_quantile(values, 0.75) - _safe_quantile(values, 0.25),
        "valid_fraction": _valid_fraction(values),
    }


def _direction_from_delta(delta: float, field_a: str, field_b: str) -> tuple[str, str, str]:
    if not np.isfinite(delta) or abs(delta) < 1e-12:
        return "tie_or_zero", "", ""
    if delta > 0:
        return "field_a_before_field_b", field_a, field_b
    return "field_b_before_field_a", field_b, field_a


def _load_bootstrap_indices(paths: MarkerDefinitionPaths, settings: StagePartitionV7Settings, profiles: dict) -> list[np.ndarray]:
    src = paths.v7e_output_dir / "bootstrap_resample_year_indices_v7_e.csv"
    if src.exists():
        df = pd.read_csv(src)
        rows = []
        for _, r in df.sort_values("bootstrap_id").iterrows():
            txt = str(r.get("sampled_year_indices", ""))
            vals = [int(x) for x in txt.split(";") if str(x).strip()]
            if vals:
                rows.append(np.asarray(vals, dtype=int))
        if rows:
            return rows
    n_years = int(np.asarray(profiles[FIELDS[0]].raw_cube).shape[0])
    n_boot = int(settings.bootstrap.effective_n_bootstrap())
    rng = np.random.default_rng(int(settings.bootstrap.random_seed))
    return [rng.integers(0, n_years, size=n_years, dtype=int) for _ in range(n_boot)]


def _recompute_w45_progress_curves(paths: MarkerDefinitionPaths, settings: StagePartitionV7Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute W45 observed and bootstrap progress curves in the current V7-e representation.

    This deliberately does not change the spatial representation, accepted window, or progress definition.
    It only reconstructs curves because V7-e stores bootstrap marker samples but not bootstrap curves.
    """
    windows = _read_csv(paths.v7e_output_dir / "accepted_windows_used_v7_e.csv")
    _require_columns(windows, ["window_id", "anchor_day", "analysis_window_start", "analysis_window_end", "pre_period_start", "pre_period_end", "post_period_start", "post_period_end"], "accepted_windows_used_v7_e")
    wsub = windows[windows["window_id"].astype(str) == WINDOW_ID]
    if wsub.empty:
        wsub = windows[pd.to_numeric(windows["anchor_day"], errors="coerce") == ANCHOR_DAY]
    if wsub.empty:
        raise ValueError("Cannot find W002 / anchor=45 in accepted_windows_used_v7_e.csv")
    window = wsub.iloc[0]

    smoothed = load_smoothed_fields(settings.foundation.smoothed_fields_path())
    profiles = build_profiles(smoothed, settings.profile)
    first_cube = np.asarray(profiles[FIELDS[0]].raw_cube)
    n_days = int(first_cube.shape[1])
    joint_state = build_state_matrix(profiles, settings.state)
    shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)

    observed_rows: list[dict] = []
    observed_curve_rows: list[dict] = []
    for field in FIELDS:
        matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(
            profiles,
            field,
            None,
            standardize=settings.state.standardize,
            trim_invalid_days=settings.state.trim_invalid_days,
            shared_valid_day_index=shared_valid_day_index,
        )
        row, curve = _compute_progress_for_window(matrix, valid_day_index, n_days, window, field, settings)
        observed_rows.append(row)
        if not curve.empty:
            observed_curve_rows.extend(curve.to_dict("records"))

    boot_indices = _load_bootstrap_indices(paths, settings, profiles)
    boot_rows: list[dict] = []
    boot_curve_rows: list[dict] = []
    for b, year_idx in enumerate(boot_indices):
        if b % max(1, int(getattr(settings.bootstrap, "progress_every", 100))) == 0:
            print(f"[V7-m bootstrap curves] {b}/{len(boot_indices)}")
        for field in FIELDS:
            matrix, valid_day_index, _ = build_field_state_matrix_for_year_indices(
                profiles,
                field,
                year_idx,
                standardize=settings.state.standardize,
                trim_invalid_days=settings.state.trim_invalid_days,
                shared_valid_day_index=shared_valid_day_index,
            )
            row, curve = _compute_progress_for_window(
                matrix,
                valid_day_index,
                n_days,
                window,
                field,
                settings,
                sample_col="bootstrap_id",
                sample_value=b,
            )
            boot_rows.append(row)
            if not curve.empty:
                boot_curve_rows.extend(curve.to_dict("records"))

    return pd.DataFrame(observed_rows), pd.DataFrame(observed_curve_rows), pd.DataFrame(boot_rows), pd.DataFrame(boot_curve_rows)


def _extract_markers_for_curve(curve: pd.DataFrame, stable_days: int) -> dict:
    if curve.empty:
        return {"marker_extract_status": "curve_empty"}
    c = curve.sort_values("day").copy()
    days = pd.to_numeric(c["day"], errors="coerce").to_numpy(dtype=float)
    progress = pd.to_numeric(c["progress"], errors="coerce").to_numpy(dtype=float)
    raw_progress = pd.to_numeric(c.get("raw_progress", c["progress"]), errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(days) & np.isfinite(progress)
    days_i = days[good].astype(int)
    vals = progress[good].astype(float)
    raw_vals = raw_progress[good].astype(float)
    out: dict[str, object] = {"marker_extract_status": "success" if len(vals) else "no_valid_progress"}
    if len(vals) == 0:
        for q in THRESHOLDS:
            out[f"{_fmt_threshold(q)}_day"] = np.nan
        for name in ["departure90_day", "departure95_day", "peak_change_day_raw", "peak_change_day_smooth3", "duration_25_75", "tail_50_75", "early_span_25_50"]:
            out[name] = np.nan
        return out

    # Threshold sweep.
    for q in THRESHOLDS:
        out[f"{_fmt_threshold(q)}_day"] = _first_stable_crossing(days_i, vals, q, stable_days)

    # Departure from pre-period background. Uses the pre period carried in the V7-e curve rows.
    pre_start = int(c["pre_period_start"].iloc[0]) if "pre_period_start" in c.columns else int(np.nanmin(days_i))
    pre_end = int(c["pre_period_end"].iloc[0]) if "pre_period_end" in c.columns else int(np.nanmin(days_i))
    pre_mask = (days_i >= pre_start) & (days_i <= pre_end)
    pre_vals = vals[pre_mask]
    pre_vals = pre_vals[np.isfinite(pre_vals)]
    if pre_vals.size >= 2:
        pre_upper90 = float(np.nanquantile(pre_vals, 0.90))
        pre_upper95 = float(np.nanquantile(pre_vals, 0.95))
        out["pre_progress_upper90"] = pre_upper90
        out["pre_progress_upper95"] = pre_upper95
        out["departure90_day"] = _first_stable_crossing(days_i, vals, pre_upper90, stable_days)
        out["departure95_day"] = _first_stable_crossing(days_i, vals, pre_upper95, stable_days)
    else:
        out["pre_progress_upper90"] = np.nan
        out["pre_progress_upper95"] = np.nan
        out["departure90_day"] = np.nan
        out["departure95_day"] = np.nan

    # Progress derivative peak. The day is assigned to the later day in a daily increment.
    if len(vals) >= 2:
        d_raw = np.diff(vals)
        if np.any(np.isfinite(d_raw)):
            idx = int(np.nanargmax(d_raw))
            out["peak_change_day_raw"] = float(days_i[idx + 1])
            out["peak_change_value_raw"] = float(d_raw[idx])
        else:
            out["peak_change_day_raw"] = np.nan
            out["peak_change_value_raw"] = np.nan
        smooth_vals = _rolling_mean(vals, window=3)
        d_smooth = np.diff(smooth_vals)
        if np.any(np.isfinite(d_smooth)):
            idx2 = int(np.nanargmax(d_smooth))
            out["peak_change_day_smooth3"] = float(days_i[idx2 + 1])
            out["peak_change_value_smooth3"] = float(d_smooth[idx2])
        else:
            out["peak_change_day_smooth3"] = np.nan
            out["peak_change_value_smooth3"] = np.nan
    else:
        out["peak_change_day_raw"] = np.nan
        out["peak_change_value_raw"] = np.nan
        out["peak_change_day_smooth3"] = np.nan
        out["peak_change_value_smooth3"] = np.nan

    t25 = out.get("t25_day", np.nan)
    t50 = out.get("t50_day", np.nan)
    t75 = out.get("t75_day", np.nan)
    out["duration_25_75"] = float(t75 - t25 + 1) if np.isfinite(t25) and np.isfinite(t75) else np.nan
    out["tail_50_75"] = float(t75 - t50) if np.isfinite(t50) and np.isfinite(t75) else np.nan
    out["early_span_25_50"] = float(t50 - t25) if np.isfinite(t25) and np.isfinite(t50) else np.nan
    out["analysis_start"] = int(np.nanmin(days_i)) if len(days_i) else np.nan
    out["analysis_end"] = int(np.nanmax(days_i)) if len(days_i) else np.nan
    out["progress_at_analysis_start"] = float(vals[0]) if len(vals) else np.nan
    out["progress_at_anchor"] = float(vals[np.argmin(np.abs(days_i - ANCHOR_DAY))]) if len(vals) else np.nan
    out["progress_at_analysis_end"] = float(vals[-1]) if len(vals) else np.nan
    return out


def _extract_markers(curves: pd.DataFrame, row_table: pd.DataFrame, sample_col: str | None, stable_days: int) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    group_cols = ["window_id", "field"] + ([sample_col] if sample_col else [])
    rows = []
    row_index_cols = group_cols.copy()
    if not row_table.empty:
        row_lookup = {tuple(r[c] for c in row_index_cols): r for _, r in row_table.iterrows() if all(c in row_table.columns for c in row_index_cols)}
    else:
        row_lookup = {}
    for key, sub in curves.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key_tuple = (key,)
        else:
            key_tuple = key
        base = dict(zip(group_cols, key_tuple))
        if str(base.get("window_id")) != WINDOW_ID or str(base.get("field")) not in FIELDS:
            continue
        markers = _extract_markers_for_curve(sub, stable_days=stable_days)
        lookup_key = tuple(base[c] for c in row_index_cols)
        r = row_lookup.get(lookup_key)
        if r is not None:
            for c in [
                "anchor_day",
                "onset_day",
                "midpoint_day",
                "finish_day",
                "duration",
                "pre_post_separation_label",
                "progress_quality_label",
                "progress_monotonicity_corr",
                "n_crossings_025",
                "n_crossings_050",
                "n_crossings_075",
            ]:
                if c in r.index:
                    base[f"v7e_{c}"] = r[c]
        rows.append({**base, **markers})
    return pd.DataFrame(rows)


def _load_or_build_markers(paths: MarkerDefinitionPaths, settings: StagePartitionV7Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    observed_curve_path = paths.v7e_output_dir / "field_transition_progress_observed_curves_long_v7_e.csv"
    observed_path = paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv"
    boot_path = paths.v7e_output_dir / "field_transition_progress_bootstrap_samples_v7_e.csv"
    cached_boot_curve_path = paths.output_dir / "w45_allfield_progress_bootstrap_curves_long_v7_m.csv"
    cached_obs_curve_path = paths.output_dir / "w45_allfield_progress_observed_curves_long_v7_m.csv"

    build_notes = []
    obs_rows = _read_csv(observed_path)
    boot_rows_existing = _read_csv(boot_path)
    obs_curves_existing = _read_csv(observed_curve_path, required=False)

    if cached_boot_curve_path.exists() and cached_obs_curve_path.exists():
        build_notes.append("used_cached_v7m_progress_curves")
        obs_curves = _read_csv(cached_obs_curve_path)
        boot_curves = _read_csv(cached_boot_curve_path)
        # Need bootstrap row metadata for quality labels. Existing V7-e sample table is enough.
        boot_rows = boot_rows_existing
    else:
        build_notes.append("recomputed_w45_progress_curves_from_v7e_representation")
        obs_rows_re, obs_curves, boot_rows, boot_curves = _recompute_w45_progress_curves(paths, settings)
        _write_csv(obs_rows_re, paths.output_dir / "w45_allfield_progress_observed_recomputed_rows_v7_m.csv")
        _write_csv(obs_curves, cached_obs_curve_path)
        _write_csv(boot_rows, paths.output_dir / "w45_allfield_progress_bootstrap_recomputed_rows_v7_m.csv")
        _write_csv(boot_curves, cached_boot_curve_path)
        obs_rows = obs_rows_re

    # If observed curves are already present and no recompute happened, use them for observed marker extraction.
    if not obs_curves_existing.empty and "recomputed_w45_progress_curves_from_v7e_representation" not in build_notes:
        obs_curves = obs_curves_existing

    stable_days = int(settings.progress_timing.stable_crossing_days)
    obs_markers = _extract_markers(obs_curves, obs_rows, sample_col=None, stable_days=stable_days)
    boot_markers = _extract_markers(boot_curves, boot_rows, sample_col="bootstrap_id", stable_days=stable_days)
    meta = {
        "build_notes": build_notes,
        "observed_curve_rows": int(len(obs_curves)),
        "bootstrap_curve_rows": int(len(boot_curves)),
        "observed_marker_rows": int(len(obs_markers)),
        "bootstrap_marker_rows": int(len(boot_markers)),
        "stable_crossing_days": stable_days,
    }
    return obs_markers, boot_markers, obs_rows, boot_rows_existing, meta


def _validate_inputs(obs_markers: pd.DataFrame, boot_markers: pd.DataFrame, paths: MarkerDefinitionPaths, meta: dict) -> dict:
    required_marker_cols = ["window_id", "field", "t25_day", "t50_day", "t75_day", "departure90_day", "peak_change_day_smooth3"]
    _require_columns(obs_markers, required_marker_cols, "observed markers")
    _require_columns(boot_markers, ["bootstrap_id"] + required_marker_cols, "bootstrap markers")
    obs_fields = set(obs_markers[obs_markers["window_id"].astype(str) == WINDOW_ID]["field"].astype(str))
    boot_fields = set(boot_markers[boot_markers["window_id"].astype(str) == WINDOW_ID]["field"].astype(str))
    audit = {
        "status": "checked",
        "created_at": _now_iso(),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "fields_required": FIELDS,
        "observed_fields_found": sorted(obs_fields),
        "bootstrap_fields_found": sorted(boot_fields),
        "observed_fields_complete": all(f in obs_fields for f in FIELDS),
        "bootstrap_fields_complete": all(f in boot_fields for f in FIELDS),
        "v7e_output_dir": str(paths.v7e_output_dir),
        "output_dir": str(paths.output_dir),
        **meta,
    }
    if not audit["observed_fields_complete"]:
        raise ValueError(f"Observed marker rows missing required fields. Found={sorted(obs_fields)}, required={FIELDS}")
    if not audit["bootstrap_fields_complete"]:
        raise ValueError(f"Bootstrap marker rows missing required fields. Found={sorted(boot_fields)}, required={FIELDS}")
    return audit


def _summarize_marker_stability(boot_markers: pd.DataFrame) -> pd.DataFrame:
    marker_cols = ["departure90_day", "departure95_day"]
    marker_cols += [f"{_fmt_threshold(q)}_day" for q in THRESHOLDS]
    marker_cols += ["peak_change_day_raw", "peak_change_day_smooth3", "duration_25_75", "tail_50_75", "early_span_25_50"]
    rows = []
    for field, sub in boot_markers[boot_markers["window_id"].astype(str) == WINDOW_ID].groupby("field", sort=False):
        row: dict[str, object] = {"window_id": WINDOW_ID, "anchor_day": ANCHOR_DAY, "field": field, "n_bootstrap_rows": int(len(sub))}
        widths = {}
        for col in marker_cols:
            stats = _marker_stats(pd.to_numeric(sub[col], errors="coerce") if col in sub.columns else np.asarray([], dtype=float))
            prefix = col.replace("_day", "")
            for k, v in stats.items():
                row[f"{prefix}_{k}"] = v
            if col.endswith("_day"):
                widths[prefix] = stats["q90_width"]
        finite_widths = {k: v for k, v in widths.items() if np.isfinite(v)}
        if finite_widths:
            min_w = min(finite_widths.values())
            max_w = max(finite_widths.values())
            most = [k for k, v in finite_widths.items() if np.isclose(v, min_w)]
            least = [k for k, v in finite_widths.items() if np.isclose(v, max_w)]
            row["most_stable_marker_group"] = ";".join(most)
            row["least_stable_marker_group"] = ";".join(least)
        else:
            row["most_stable_marker_group"] = "none"
            row["least_stable_marker_group"] = "none"
        row["dominant_progress_quality_label"] = sub.get("v7e_progress_quality_label", pd.Series(dtype=str)).astype(str).value_counts().idxmax() if "v7e_progress_quality_label" in sub.columns and not sub.empty else "unknown"
        row["dominant_prepost_separation_label"] = sub.get("v7e_pre_post_separation_label", pd.Series(dtype=str)).astype(str).value_counts().idxmax() if "v7e_pre_post_separation_label" in sub.columns and not sub.empty else "unknown"
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_delta_from_marker(boot_markers: pd.DataFrame, marker_col: str, extra_cols: dict | None = None) -> pd.DataFrame:
    sub = boot_markers[boot_markers["window_id"].astype(str) == WINDOW_ID].copy()
    if marker_col not in sub.columns:
        return pd.DataFrame()
    piv = sub.pivot_table(index="bootstrap_id", columns="field", values=marker_col, aggfunc="first")
    rows = []
    for a, b in combinations(FIELDS, 2):
        if a not in piv.columns or b not in piv.columns:
            continue
        av = pd.to_numeric(piv[a], errors="coerce")
        bv = pd.to_numeric(piv[b], errors="coerce")
        good = av.notna() & bv.notna()
        delta = (bv[good] - av[good]).to_numpy(dtype=float)
        if delta.size == 0:
            row = {"field_a": a, "field_b": b, "n_valid": 0}
        else:
            med = _safe_median(delta)
            q05 = _safe_quantile(delta, 0.05)
            q95 = _safe_quantile(delta, 0.95)
            q025 = _safe_quantile(delta, 0.025)
            q975 = _safe_quantile(delta, 0.975)
            direction, early, late = _direction_from_delta(med, a, b)
            pass90 = bool((q05 > 0) or (q95 < 0))
            pass95 = bool((q025 > 0) or (q975 < 0))
            row = {
                "field_a": a,
                "field_b": b,
                "n_valid": int(delta.size),
                "median_delta": med,
                "q05_delta": q05,
                "q95_delta": q95,
                "q025_delta": q025,
                "q975_delta": q975,
                "prob_delta_gt_0": float(np.mean(delta > 0)),
                "prob_delta_lt_0": float(np.mean(delta < 0)),
                "pass90": pass90,
                "pass95": pass95,
                "direction_label": direction,
                "early_field_candidate": early,
                "late_field_candidate": late,
            }
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_sweep_pairwise(boot_markers: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for q in THRESHOLDS:
        marker_col = f"{_fmt_threshold(q)}_day"
        df = _pairwise_delta_from_marker(boot_markers, marker_col, {"threshold": float(q), "marker_col": marker_col})
        if not df.empty:
            dfs.append(df)
    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not out.empty:
        out["order_label"] = np.where(out["pass95"], "threshold_order_pass95", np.where(out["pass90"], "threshold_order_pass90", "threshold_order_not_confirmed"))
    return out


def _departure_pairwise(boot_markers: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for level, col in [("departure90", "departure90_day"), ("departure95", "departure95_day")]:
        df = _pairwise_delta_from_marker(boot_markers, col, {"departure_level": level, "marker_col": col})
        if not df.empty:
            dfs.append(df)
    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not out.empty:
        out["departure_order_label"] = np.where(out["pass95"], "departure_order_pass95", np.where(out["pass90"], "departure_order_pass90", "departure_order_not_confirmed"))
    return out


def _peak_pairwise(boot_markers: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for typ, col in [("raw", "peak_change_day_raw"), ("smooth3", "peak_change_day_smooth3")]:
        df = _pairwise_delta_from_marker(boot_markers, col, {"peak_marker_type": typ, "marker_col": col})
        if not df.empty:
            dfs.append(df)
    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not out.empty:
        out["peak_order_label"] = np.where(out["pass95"], "peak_order_pass95", np.where(out["pass90"], "peak_order_pass90", "peak_order_not_confirmed"))
    return out


def _same_direction(directions: Iterable[str]) -> str:
    vals = [d for d in directions if d and d != "tie_or_zero" and str(d).lower() != "nan"]
    if not vals:
        return "none"
    return vals[0] if all(v == vals[0] for v in vals) else "mixed"


def _build_marker_consistency_ledger(thr: pd.DataFrame, dep: pd.DataFrame, peak: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(FIELDS, 2):
        tsub = thr[(thr["field_a"] == a) & (thr["field_b"] == b)].copy() if not thr.empty else pd.DataFrame()
        dsub = dep[(dep["field_a"] == a) & (dep["field_b"] == b)].copy() if not dep.empty else pd.DataFrame()
        psub = peak[(peak["field_a"] == a) & (peak["field_b"] == b)].copy() if not peak.empty else pd.DataFrame()
        t90 = tsub[tsub.get("pass90", False).astype(bool)] if not tsub.empty else pd.DataFrame()
        t95 = tsub[tsub.get("pass95", False).astype(bool)] if not tsub.empty else pd.DataFrame()
        d90 = dsub[dsub.get("pass90", False).astype(bool)] if not dsub.empty else pd.DataFrame()
        d95 = dsub[dsub.get("pass95", False).astype(bool)] if not dsub.empty else pd.DataFrame()
        p90 = psub[psub.get("pass90", False).astype(bool)] if not psub.empty else pd.DataFrame()
        p95 = psub[psub.get("pass95", False).astype(bool)] if not psub.empty else pd.DataFrame()
        dirs = []
        for df in [t90, d90, p90]:
            if not df.empty:
                dirs.extend(df["direction_label"].astype(str).tolist())
        dir_cons = _same_direction(dirs)
        thresholds_pass90 = _join([f"{float(r['threshold']):.2f}" for _, r in t90.iterrows()]) if not t90.empty else "none"
        thresholds_pass95 = _join([f"{float(r['threshold']):.2f}" for _, r in t95.iterrows()]) if not t95.empty else "none"
        # Determine supported direction candidate from majority of pass90 rows if available.
        early_candidates = []
        late_candidates = []
        for df in [t90, d90, p90]:
            if not df.empty:
                early_candidates.extend(df["early_field_candidate"].astype(str).tolist())
                late_candidates.extend(df["late_field_candidate"].astype(str).tolist())
        early_mode = pd.Series([x for x in early_candidates if x]).mode().iloc[0] if early_candidates and pd.Series([x for x in early_candidates if x]).size else ""
        late_mode = pd.Series([x for x in late_candidates if x]).mode().iloc[0] if late_candidates and pd.Series([x for x in late_candidates if x]).size else ""
        n_t90 = int(len(t90))
        n_t95 = int(len(t95))
        dep90_any = bool(len(d90) > 0)
        dep95_any = bool(len(d95) > 0)
        peak90_any = bool(len(p90) > 0)
        peak95_any = bool(len(p95) > 0)
        midpoint50_pass90 = bool((not tsub.empty) and bool(tsub[np.isclose(pd.to_numeric(tsub["threshold"], errors="coerce"), 0.50)].get("pass90", pd.Series(dtype=bool)).any()))
        finish75_pass90 = bool((not tsub.empty) and bool(tsub[np.isclose(pd.to_numeric(tsub["threshold"], errors="coerce"), 0.75)].get("pass90", pd.Series(dtype=bool)).any()))
        if dir_cons == "mixed":
            label = "marker_inconsistent"
            interp = "Different marker families support conflicting directions; do not simplify into an order."
        elif n_t90 >= 3 and (dep90_any or peak90_any):
            label = "robust_early_transition_order"
            interp = "Same direction appears across several threshold levels and at least one non-threshold marker family."
        elif dep90_any and peak90_any:
            label = "departure_and_peak_supported_order"
            interp = "Departure-from-pre and peak-change both support the same direction."
        elif dep90_any:
            label = "departure_supported_order"
            interp = "Departure-from-pre supports the order; threshold/peak support should be checked."
        elif peak90_any:
            label = "peak_supported_order"
            interp = "Progress-derivative peak supports the order; this is strongest-change timing, not onset."
        elif n_t90 >= 2:
            label = "threshold_range_order"
            interp = "A range of progress thresholds supports this direction, but non-threshold markers do not."
        elif n_t90 == 1:
            label = "threshold_only_order"
            interp = "Only one threshold supports this direction; treat as threshold-sensitive candidate."
        else:
            label = "not_distinguishable_across_markers"
            interp = "No marker family gives a 90% confirmed direction."
        rows.append(
            {
                "field_a": a,
                "field_b": b,
                "thresholds_pass90": thresholds_pass90,
                "thresholds_pass95": thresholds_pass95,
                "n_thresholds_pass90": n_t90,
                "n_thresholds_pass95": n_t95,
                "departure90_pass": dep90_any,
                "departure95_pass": dep95_any,
                "peak_raw_pass90": bool((not p90.empty) and (p90["peak_marker_type"].astype(str) == "raw").any()),
                "peak_smooth3_pass90": bool((not p90.empty) and (p90["peak_marker_type"].astype(str) == "smooth3").any()),
                "peak_any_pass90": peak90_any,
                "peak_any_pass95": peak95_any,
                "midpoint50_pass90": midpoint50_pass90,
                "finish75_pass90": finish75_pass90,
                "pass90_direction_consistency": dir_cons,
                "early_field_candidate": early_mode,
                "late_field_candidate": late_mode,
                "overall_marker_consistency_label": label,
                "interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


def _classify_field_transition_types(summary: pd.DataFrame, obs_markers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    obs_by_field = {str(r["field"]): r for _, r in obs_markers.iterrows() if str(r.get("window_id")) == WINDOW_ID}
    for _, r in summary.iterrows():
        field = str(r["field"])
        obs = obs_by_field.get(field)
        if obs is None:
            obs = pd.Series(dtype=object)
        # Marker family widths.
        dep_w = min([x for x in [r.get("departure90_q90_width", np.nan), r.get("departure95_q90_width", np.nan)] if np.isfinite(x)] or [np.nan])
        early_ws = [r.get(f"{_fmt_threshold(q)}_q90_width", np.nan) for q in EARLY_THRESHOLDS]
        early_w = float(np.nanmedian([x for x in early_ws if np.isfinite(x)])) if any(np.isfinite(x) for x in early_ws) else np.nan
        mid_w = r.get("t50_q90_width", np.nan)
        fin_w = r.get("t75_q90_width", np.nan)
        peak_w = min([x for x in [r.get("peak_change_raw_q90_width", np.nan), r.get("peak_change_smooth3_q90_width", np.nan)] if np.isfinite(x)] or [np.nan])
        dur_w = r.get("duration_25_75_q90_width", np.nan)
        # Label by relative marker-family stability. Do not use fixed day thresholds.
        widths = {"departure": dep_w, "early_threshold": early_w, "midpoint": mid_w, "finish": fin_w, "peak_change": peak_w}
        finite = {k: v for k, v in widths.items() if np.isfinite(v)}
        if not finite:
            rec_family = "none"
            label = "transition_not_established"
        else:
            min_w = min(finite.values())
            best = [k for k, v in finite.items() if np.isclose(v, min_w)]
            rec_family = ";".join(best)
            if "midpoint" in best and len(best) == 1:
                label = "compact_midpoint_transition"
            elif "departure" in best or "early_threshold" in best:
                # If finish/duration are wide relative to departure/early, interpret as broad candidate.
                label = "early_departure_broad_transition" if ((np.isfinite(fin_w) and np.isfinite(min_w) and fin_w > min_w) or (np.isfinite(dur_w) and np.isfinite(min_w) and dur_w > min_w)) else "early_departure_transition"
            elif "peak_change" in best:
                label = "peak_change_defined_transition"
            else:
                label = "marker_mixed_transition"
        quality = str(obs.get("v7e_progress_quality_label", obs.get("progress_quality_label", "unknown")))
        if "nonmonotonic" in quality.lower() and label not in ["transition_not_established"]:
            label = f"{label}_with_nonmonotonic_caution"
        rows.append(
            {
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "field": field,
                "pre_post_separation_label": obs.get("v7e_pre_post_separation_label", obs.get("pre_post_separation_label", "unknown")),
                "progress_quality_label": quality,
                "departure_q90_width_best": dep_w,
                "early_threshold_q90_width_median": early_w,
                "peak_change_q90_width_best": peak_w,
                "midpoint50_q90_width": mid_w,
                "finish75_q90_width": fin_w,
                "duration_25_75_q90_width": dur_w,
                "transition_type_label": label,
                "recommended_marker_family": rec_family,
                "can_use_midpoint_order": bool(label == "compact_midpoint_transition"),
                "can_use_departure_order": bool("departure" in rec_family or "early_threshold" in rec_family),
                "can_use_peak_order": bool("peak_change" in rec_family),
                "can_use_threshold_order": bool("early_threshold" in rec_family),
                "field_transition_interpretation": _interpret_field_type(field, label, rec_family),
            }
        )
    return pd.DataFrame(rows)


def _interpret_field_type(field: str, label: str, rec_family: str) -> str:
    if "compact_midpoint" in label:
        return f"{field}: compact transition; midpoint may be representative under current V7-e representation."
    if "early_departure" in label:
        return f"{field}: early-departure/early-progress family is more stable than midpoint/finish; avoid default midpoint-order."
    if "peak_change" in label:
        return f"{field}: strongest-change timing is the clearest marker; peak-change is not onset or completion."
    if "not_established" in label:
        return f"{field}: transition marker is not established under current inputs."
    return f"{field}: marker family is mixed or unresolved; do not force a single transition day."


def _build_window_summary(field_types: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    counts = field_types["transition_type_label"].astype(str).value_counts().to_dict() if not field_types.empty else {}
    robust = ledger[ledger["overall_marker_consistency_label"] == "robust_early_transition_order"] if not ledger.empty else pd.DataFrame()
    threshold_only = ledger[ledger["overall_marker_consistency_label"] == "threshold_only_order"] if not ledger.empty else pd.DataFrame()
    dep_supported = ledger[ledger["overall_marker_consistency_label"].astype(str).str.contains("departure", na=False)] if not ledger.empty else pd.DataFrame()
    peak_supported = ledger[ledger["overall_marker_consistency_label"].astype(str).str.contains("peak", na=False)] if not ledger.empty else pd.DataFrame()
    inconsistent = ledger[ledger["overall_marker_consistency_label"] == "marker_inconsistent"] if not ledger.empty else pd.DataFrame()
    not_dist = ledger[ledger["overall_marker_consistency_label"] == "not_distinguishable_across_markers"] if not ledger.empty else pd.DataFrame()
    def edge_list(df: pd.DataFrame) -> str:
        if df.empty:
            return "none"
        vals = []
        for _, r in df.iterrows():
            e = str(r.get("early_field_candidate", ""))
            l = str(r.get("late_field_candidate", ""))
            if e and l:
                vals.append(f"{e}<{l}")
            else:
                vals.append(f"{r.get('field_a')}~{r.get('field_b')}")
        return ";".join(vals)
    n_mid = int(sum("compact_midpoint" in str(x) for x in field_types.get("transition_type_label", pd.Series(dtype=str)))) if not field_types.empty else 0
    n_early = int(sum("early_departure" in str(x) for x in field_types.get("transition_type_label", pd.Series(dtype=str)))) if not field_types.empty else 0
    n_peak = int(sum("peak_change" in str(x) for x in field_types.get("transition_type_label", pd.Series(dtype=str)))) if not field_types.empty else 0
    n_unres = int(sum("not_established" in str(x) or "mixed" in str(x) for x in field_types.get("transition_type_label", pd.Series(dtype=str)))) if not field_types.empty else 0
    if len(robust) > 0:
        wtype = "early_transition_order_window"
        interp = "At least one pair is supported across multiple marker families; interpret only as marker-consistent early-transition order."
    elif n_early >= 2 and len(dep_supported) + len(peak_supported) > 0:
        wtype = "marker_mixed_with_early_transition_evidence"
        interp = "Several fields favor early/departure markers and some pairwise marker support exists; do not force midpoint-order."
    elif n_mid >= 3:
        wtype = "midpoint_order_candidate_window"
        interp = "Several fields have compact midpoint markers; midpoint-order may be considered after pairwise confirmation."
    elif n_unres >= 3:
        wtype = "marker_mixed_or_unresolved_window"
        interp = "Most field transition markers are mixed or unresolved; use descriptive marker ledger only."
    else:
        wtype = "transition_shape_window"
        interp = "Window is better described by field transition shapes than by a single order family."
    return pd.DataFrame(
        [
            {
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "n_fields_compact_midpoint": n_mid,
                "n_fields_early_departure_broad": n_early,
                "n_fields_peak_defined": n_peak,
                "n_fields_marker_mixed_or_unresolved": n_unres,
                "robust_pairwise_orders": edge_list(robust),
                "threshold_only_pairwise_orders": edge_list(threshold_only),
                "departure_supported_orders": edge_list(dep_supported),
                "peak_supported_orders": edge_list(peak_supported),
                "marker_inconsistent_pairs": edge_list(inconsistent),
                "not_distinguishable_pairs": edge_list(not_dist),
                "window_transition_type": wtype,
                "recommended_window_interpretation": interp,
                "recommended_next_action": "Use marker-consistency ledger to decide which local claims survive; do not call t25 a physical onset without consistency support.",
                "transition_type_counts": json.dumps(counts, ensure_ascii=False),
            }
        ]
    )


def _try_write_figures(paths: MarkerDefinitionPaths, boot_markers: pd.DataFrame, threshold_pairwise: pd.DataFrame, summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        _ensure_dir(paths.figure_dir)
        # 1. Threshold sweep heatmap: use H/Je, P/Je, H/Jw etc. all pairs as rows, thresholds as columns with median delta.
        if not threshold_pairwise.empty:
            df = threshold_pairwise.copy()
            df["pair"] = df["field_a"].astype(str) + "-" + df["field_b"].astype(str)
            mat = df.pivot_table(index="pair", columns="threshold", values="median_delta", aggfunc="first").sort_index()
            fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(mat))))
            im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto")
            ax.set_yticks(range(len(mat.index)), mat.index)
            ax.set_xticks(range(len(mat.columns)), [f"{c:.2f}" for c in mat.columns])
            ax.set_title("W45 threshold-sweep pairwise median Δ = t_B - t_A")
            ax.set_xlabel("progress threshold")
            ax.set_ylabel("field pair")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(paths.figure_dir / "w45_threshold_sweep_pairwise_heatmap_v7_m.png", dpi=160)
            plt.close(fig)
        # 2. Marker interval plot.
        if not summary.empty:
            markers = ["departure90", "t25", "t50", "t75", "peak_change_smooth3"]
            rows = []
            for _, r in summary.iterrows():
                for m in markers:
                    med = r.get(f"{m}_median", np.nan)
                    q05 = r.get(f"{m}_q05", np.nan)
                    q95 = r.get(f"{m}_q95", np.nan)
                    if np.isfinite(med):
                        rows.append({"field": r["field"], "marker": m, "median": med, "q05": q05, "q95": q95})
            pdf = pd.DataFrame(rows)
            if not pdf.empty:
                fig, ax = plt.subplots(figsize=(9, 5.5))
                ylabels = []
                y = 0
                for field in FIELDS:
                    fs = pdf[pdf["field"] == field]
                    for m in markers:
                        rr = fs[fs["marker"] == m]
                        if rr.empty:
                            continue
                        r = rr.iloc[0]
                        ax.plot([r["q05"], r["q95"]], [y, y], linewidth=2)
                        ax.scatter([r["median"]], [y], s=20)
                        ylabels.append(f"{field}:{m}")
                        y += 1
                ax.set_yticks(range(len(ylabels)), ylabels)
                ax.axvline(ANCHOR_DAY, linewidth=1.0)
                ax.set_xlabel("day index")
                ax.set_title("W45 marker bootstrap intervals (q05-q95)")
                fig.tight_layout()
                fig.savefig(paths.figure_dir / "w45_field_marker_intervals_v7_m.png", dpi=160)
                plt.close(fig)
        # 3. Mean progress derivative curves from bootstrap marker curves are expensive to reload here; use bootstrap marker peak distribution instead.
        if not boot_markers.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            data = []
            labels = []
            for f in FIELDS:
                vals = pd.to_numeric(boot_markers[boot_markers["field"] == f]["peak_change_day_smooth3"], errors="coerce").dropna().to_numpy(dtype=float)
                if vals.size:
                    data.append(vals)
                    labels.append(f)
            if data:
                ax.boxplot(data, labels=labels, showfliers=False)
                ax.axhline(ANCHOR_DAY, linewidth=1.0)
                ax.set_ylabel("peak-change day (smooth3)")
                ax.set_title("W45 progress-derivative peak day by field")
                fig.tight_layout()
                fig.savefig(paths.figure_dir / "w45_progress_derivative_peak_boxplot_v7_m.png", dpi=160)
            plt.close(fig)
    except Exception as exc:
        _write_text(f"Plotting failed: {exc}", paths.output_dir / "plot_warning_v7_m.txt")


def _write_markdown_summary(paths: MarkerDefinitionPaths, audit: dict, field_types: pd.DataFrame, ledger: pd.DataFrame, window_summary: pd.DataFrame) -> None:
    lines = [
        "# W45 all-field transition marker definition audit v7_m",
        "",
        "## Purpose",
        "This audit checks whether the former t(0.25) early-progress crossing is robust to alternative field-transition marker definitions.",
        "It keeps the current V7-e progress/profile representation fixed and does not switch to raw025 inputs.",
        "",
        "## Input representation",
        f"- V7-e output dir: `{audit.get('v7e_output_dir')}`",
        f"- Bootstrap marker rows: {audit.get('bootstrap_marker_rows')}",
        f"- Build notes: {audit.get('build_notes')}",
        "",
        "## Field transition types",
    ]
    if not field_types.empty:
        for _, r in field_types.iterrows():
            lines.append(f"- {r['field']}: {r['transition_type_label']} | recommended_marker_family={r['recommended_marker_family']}")
    lines += ["", "## Marker consistency ledger"]
    if not ledger.empty:
        for _, r in ledger.iterrows():
            lines.append(f"- {r['field_a']}-{r['field_b']}: {r['overall_marker_consistency_label']} | {r['interpretation']}")
    lines += ["", "## Window-level interpretation"]
    if not window_summary.empty:
        r = window_summary.iloc[0]
        lines.append(f"- window_transition_type: {r['window_transition_type']}")
        lines.append(f"- interpretation: {r['recommended_window_interpretation']}")
    lines += [
        "",
        "## Prohibited interpretations",
        "- Do not call t25 a physical onset unless marker-consistency support is present.",
        "- Do not treat threshold-only results as robust transition order.",
        "- Do not treat peak-change day as departure or completion timing.",
        "- Do not force marker-inconsistent pairs into an order.",
        "- Do not call not-distinguishable pairs synchronous without an equivalence test.",
        "- Do not omit P/V/H/Je/Jw from W45 summaries.",
    ]
    _write_text("\n".join(lines), paths.output_dir / "w45_transition_marker_definition_audit_summary_v7_m.md")
    _write_text("\n".join(lines), paths.log_dir / "w45_transition_marker_definition_audit_summary_v7_m.md")


def run_w45_allfield_transition_marker_definition_audit_v7_m(v7_root: Optional[Path] = None) -> None:
    paths = _resolve_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)

    settings = StagePartitionV7Settings()
    # Keep V7-e's output tag untouched. This branch only reads/recomputes W45 curves under V7-e representation.
    obs_markers, boot_markers, obs_rows, boot_rows, build_meta = _load_or_build_markers(paths, settings)
    audit = _validate_inputs(obs_markers, boot_markers, paths, build_meta)
    _write_json(audit, paths.output_dir / "input_audit_v7_m.json")
    _write_json(audit, paths.log_dir / "input_audit_v7_m.json")

    _write_csv(obs_markers, paths.output_dir / "w45_allfield_marker_observed_v7_m.csv")
    _write_csv(boot_markers, paths.output_dir / "w45_allfield_marker_bootstrap_samples_v7_m.csv")

    marker_summary = _summarize_marker_stability(boot_markers)
    _write_csv(marker_summary, paths.output_dir / "w45_allfield_marker_stability_summary_v7_m.csv")

    threshold_pairwise = _threshold_sweep_pairwise(boot_markers)
    _write_csv(threshold_pairwise, paths.output_dir / "w45_allfield_threshold_sweep_pairwise_v7_m.csv")

    departure_pairwise = _departure_pairwise(boot_markers)
    _write_csv(departure_pairwise, paths.output_dir / "w45_allfield_departure_pairwise_v7_m.csv")

    peak_pairwise = _peak_pairwise(boot_markers)
    _write_csv(peak_pairwise, paths.output_dir / "w45_allfield_peak_change_pairwise_v7_m.csv")

    ledger = _build_marker_consistency_ledger(threshold_pairwise, departure_pairwise, peak_pairwise)
    _write_csv(ledger, paths.output_dir / "w45_allfield_marker_consistency_ledger_v7_m.csv")

    field_types = _classify_field_transition_types(marker_summary, obs_markers)
    _write_csv(field_types, paths.output_dir / "w45_allfield_transition_type_classification_v7_m.csv")

    window_summary = _build_window_summary(field_types, ledger)
    _write_csv(window_summary, paths.output_dir / "w45_transition_marker_definition_window_summary_v7_m.csv")

    _try_write_figures(paths, boot_markers, threshold_pairwise, marker_summary)

    run_meta = {
        "status": "success",
        "created_at": _now_iso(),
        "output_tag": OUTPUT_TAG,
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "fields": FIELDS,
        "input_representation": "current_v7e_progress_profile",
        "does_not_change_spatial_unit": True,
        "does_not_switch_to_raw025": True,
        "does_not_change_stage_window": True,
        "marker_definitions": ["threshold_sweep", "departure_from_pre", "progress_derivative_peak", "midpoint", "finish", "duration"],
        "n_observed_marker_rows": int(len(obs_markers)),
        "n_bootstrap_marker_rows": int(len(boot_markers)),
        "n_threshold_pairwise_rows": int(len(threshold_pairwise)),
        "n_departure_pairwise_rows": int(len(departure_pairwise)),
        "n_peak_pairwise_rows": int(len(peak_pairwise)),
        "notes": [
            "t25 is treated as early_progress_day_25, not as a physical onset by default.",
            "Pairwise order is only considered robust if multiple marker families agree.",
            "Progress curves are recomputed only for W45 and only under the current V7-e representation when bootstrap curves are not available.",
        ],
    }
    _write_json(run_meta, paths.output_dir / "run_meta.json")
    _write_json(run_meta, paths.log_dir / "run_meta.json")
    _write_markdown_summary(paths, audit, field_types, ledger, window_summary)
    print(f"[V7-m] wrote outputs to {paths.output_dir}")


__all__ = ["run_w45_allfield_transition_marker_definition_audit_v7_m"]
