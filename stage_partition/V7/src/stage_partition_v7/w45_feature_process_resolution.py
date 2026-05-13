from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# Reuse base-state rebuilding utilities from the clean V7-p trunk.  This module
# does NOT read V7-p result tables; it only reuses low-level code paths/functions.
from stage_partition_v7.w45_process_relation_rebuild import (  # noqa: WPS450
    _audit_no_forbidden_inputs as _audit_no_forbidden_inputs_p,
    _build_field_state,
    _bootstrap_indices,
    _ensure_dir,
    _first_stable_crossing,
    _fmt_threshold,
    _full_matrix,
    _load_profiles,
    _q,
    _resolve_settings as _resolve_p_settings,
    _rolling_mean,
    _safe_arr,
    _write_csv,
    _write_json,
    _write_text,
)

OUTPUT_TAG = "w45_feature_process_resolution_v7_q"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELDS = ["P", "V", "H", "Je", "Jw"]
EARLY_GROUP_FIELDS = ["P", "V", "H", "Jw"]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75]
MARKER_NAMES = ["departure90", "departure95"] + [_fmt_threshold(t) for t in THRESHOLDS] + ["peak_raw", "peak_smooth3", "duration_25_75", "tail_50_75", "early_span_25_50"]
TIMING_MARKERS = ["departure90", "departure95"] + [_fmt_threshold(t) for t in THRESHOLDS] + ["peak_raw", "peak_smooth3"]
FORBIDDEN_INPUT_DIR_KEYWORDS = [
    "w45_allfield_transition_marker_definition_audit_v7_m",
    "w45_allfield_process_relation_layer_v7_n",
    "w45_process_curve_foundation_audit_v7_o",
    "w45_process_relation_rebuild_v7_p",
]
EPS = 1e-12


@dataclass
class V7QSettings:
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
    early_group_fields: tuple[str, ...] = tuple(EARLY_GROUP_FIELDS)
    n_bootstrap: int = 1000
    random_seed: int = 20260445
    stable_crossing_days: int = 2
    progress_every: int = 50
    near_equal_days: float = 1.0
    strong_contribution_quantile: float = 0.75
    moderate_contribution_quantile: float = 0.50
    read_previous_derived_results: bool = False


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _mean(values: Iterable[Any] | np.ndarray) -> float:
    arr = _safe_arr(values)
    return float(np.nanmean(arr)) if arr.size else np.nan


def _median(values: Iterable[Any] | np.ndarray) -> float:
    arr = _safe_arr(values)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def _valid_fraction(values: Iterable[Any] | np.ndarray) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return float(np.isfinite(arr).sum() / arr.size) if arr.size else np.nan


def _norm(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nansum(np.square(arr)))) if arr.size else np.nan


def _stats(values: Iterable[Any] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return {
        "median": _median(arr),
        "q05": _q(arr, 0.05),
        "q10": _q(arr, 0.10),
        "q25": _q(arr, 0.25),
        "q75": _q(arr, 0.75),
        "q90": _q(arr, 0.90),
        "q95": _q(arr, 0.95),
        "q025": _q(arr, 0.025),
        "q975": _q(arr, 0.975),
        "iqr": _q(arr, 0.75) - _q(arr, 0.25),
        "q90_width": _q(arr, 0.95) - _q(arr, 0.05),
        "q95_width": _q(arr, 0.975) - _q(arr, 0.025),
        "valid_fraction": _valid_fraction(arr),
    }


def _join(values: Iterable[Any]) -> str:
    out: list[str] = []
    for v in values:
        s = str(v)
        if s and s.lower() not in {"nan", "none"}:
            out.append(s)
    return ";".join(out) if out else "none"


def _resolve_settings(v7_root: Optional[Path]) -> tuple[V7QSettings, Any]:
    p_settings, base_settings = _resolve_p_settings(v7_root)
    settings = V7QSettings(
        v7_root=p_settings.v7_root,
        v6_root=p_settings.v6_root,
        output_dir=p_settings.v7_root / "outputs" / OUTPUT_TAG,
        log_dir=p_settings.v7_root / "logs" / OUTPUT_TAG,
        figure_dir=p_settings.v7_root / "outputs" / OUTPUT_TAG / "figures",
        n_bootstrap=int(p_settings.n_bootstrap),
        random_seed=int(p_settings.random_seed),
        stable_crossing_days=int(p_settings.stable_crossing_days),
        progress_every=int(p_settings.progress_every),
    )
    return settings, base_settings


def _audit_no_forbidden_inputs(settings: V7QSettings) -> None:
    if settings.read_previous_derived_results:
        raise RuntimeError("V7-q forbids reading previous derived V7-m/n/o/p outputs as input.")
    # This branch may output to its own directory only.  The forbidden keywords apply to input use,
    # but checking the current output_dir helps catch accidental misconfiguration.
    for key in FORBIDDEN_INPUT_DIR_KEYWORDS:
        if key in str(settings.output_dir):
            raise RuntimeError(f"V7-q output_dir unexpectedly points to forbidden branch: {key}")


def _prepost_vectors(full: np.ndarray, settings: V7QSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = np.arange(full.shape[0], dtype=int)
    pre_mask = (days >= settings.pre_start) & (days <= settings.pre_end)
    post_mask = (days >= settings.post_start) & (days <= settings.post_end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pre = np.nanmean(full[pre_mask, :], axis=0)
        post = np.nanmean(full[post_mask, :], axis=0)
    return pre, post, post - pre


def _feature_meta_rows(meta: pd.DataFrame, field: str, n_features: int) -> pd.DataFrame:
    if meta is None or meta.empty:
        return pd.DataFrame({"feature_id": np.arange(n_features, dtype=int), "field": field, "feature_coordinate": np.arange(n_features, dtype=float), "feature_type": "feature_index", "metadata_status": "missing"})
    df = meta.copy().reset_index(drop=True)
    if "feature_index" in df.columns:
        df = df.rename(columns={"feature_index": "feature_id"})
    if "feature_id" not in df.columns:
        df["feature_id"] = np.arange(len(df), dtype=int)
    if "lat_value" in df.columns:
        df["feature_coordinate"] = df["lat_value"]
        df["feature_type"] = "lat_value"
    elif "coordinate" in df.columns:
        df["feature_coordinate"] = df["coordinate"]
        df["feature_type"] = "coordinate"
    else:
        df["feature_coordinate"] = df["feature_id"]
        df["feature_type"] = "feature_index"
    df["field"] = field
    df["metadata_status"] = "available"
    keep = ["field", "feature_id", "feature_coordinate", "feature_type", "metadata_status"]
    return df[keep].iloc[:n_features].copy()


def _feature_progress_matrix(full: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    d = post - pre
    progress = np.full_like(full, np.nan, dtype=float)
    good = np.isfinite(d) & (np.abs(d) > EPS)
    if good.any():
        progress[:, good] = (full[:, good] - pre[good][None, :]) / d[good][None, :]
    return progress


def _feature_markers_from_progress(days: np.ndarray, progress: np.ndarray, settings: V7QSettings) -> dict[str, np.ndarray]:
    n_features = progress.shape[1]
    out: dict[str, np.ndarray] = {}
    # Departure uses the feature's own pre-period progress distribution.
    pre_mask = (days >= settings.pre_start) & (days <= settings.pre_end)
    dep90 = np.full(n_features, np.nan)
    dep95 = np.full(n_features, np.nan)
    for j in range(n_features):
        vals = progress[:, j]
        pre_vals = vals[pre_mask]
        if not np.isfinite(pre_vals).any():
            continue
        upper90 = _q(pre_vals, 0.90)
        upper95 = _q(pre_vals, 0.95)
        dep90[j] = _first_stable_crossing(days, vals, upper90, settings.stable_crossing_days)
        dep95[j] = _first_stable_crossing(days, vals, upper95, settings.stable_crossing_days)
    out["departure90"] = dep90
    out["departure95"] = dep95
    for thr in THRESHOLDS:
        name = _fmt_threshold(thr)
        arr = np.full(n_features, np.nan)
        for j in range(n_features):
            arr[j] = _first_stable_crossing(days, progress[:, j], thr, settings.stable_crossing_days)
        out[name] = arr
    # Peak markers.
    peak_raw = np.full(n_features, np.nan)
    peak_sm = np.full(n_features, np.nan)
    for j in range(n_features):
        vals = progress[:, j]
        if np.isfinite(vals).sum() < 3:
            continue
        d_raw = np.diff(vals, prepend=np.nan)
        if np.isfinite(d_raw).any():
            peak_raw[j] = float(days[int(np.nanargmax(d_raw))])
        sm = _rolling_mean(vals, 3)
        d_sm = np.diff(sm, prepend=np.nan)
        if np.isfinite(d_sm).any():
            peak_sm[j] = float(days[int(np.nanargmax(d_sm))])
    out["peak_raw"] = peak_raw
    out["peak_smooth3"] = peak_sm
    # Spans.
    out["duration_25_75"] = out["t75"] - out["t25"]
    out["tail_50_75"] = out["t75"] - out["t50"]
    out["early_span_25_50"] = out["t50"] - out["t25"]
    return out


def _label_amplitude(abs_d: float, rel: float, snr: float) -> str:
    if not np.isfinite(abs_d) or abs_d < EPS:
        return "low_contribution_feature"
    if np.isfinite(snr) and snr < 0.5:
        return "unstable_or_noisy_feature"
    if np.isfinite(rel) and rel >= 0.05:
        return "strong_transition_feature"
    if np.isfinite(rel) and rel >= 0.02:
        return "moderate_transition_feature"
    if np.isfinite(snr) and snr >= 1.0:
        return "weak_transition_feature"
    return "low_contribution_feature"


def _label_bootstrap_marker(st: dict[str, float]) -> str:
    if st.get("valid_fraction", 0.0) < 0.5:
        return "invalid_feature_marker"
    width = st.get("q90_width", np.nan)
    if not np.isfinite(width):
        return "invalid_feature_marker"
    if width <= 3:
        return "stable_feature_marker"
    if width <= 8:
        return "moderately_stable_feature_marker"
    return "unstable_feature_marker"


def _label_feature_process(amplitude_label: str, reliable_markers: int) -> str:
    if amplitude_label in {"strong_transition_feature", "moderate_transition_feature"} and reliable_markers >= 4:
        return "resolved_transition_feature"
    if amplitude_label in {"strong_transition_feature", "moderate_transition_feature"}:
        return "transition_feature_with_uncertain_timing"
    if amplitude_label == "unstable_or_noisy_feature":
        return "noisy_transition_feature"
    return "weak_or_low_contribution_feature"


def _build_states_from_base(settings: V7QSettings, base_settings: Any):
    smoothed_path, profiles = _load_profiles(base_settings)
    n_years = int(np.asarray(profiles[FIELDS[0]].raw_cube).shape[0])
    first_cube = np.asarray(profiles[FIELDS[0]].raw_cube)
    n_days = int(first_cube.shape[1])
    # Reuse the same joint valid day index logic from V7-p.
    from stage_partition_v6.state_builder import build_state_matrix

    joint_state = build_state_matrix(profiles, base_settings.state)
    shared_valid_day_index = np.asarray(joint_state["valid_day_index"], dtype=int)
    states: dict[str, np.ndarray] = {}
    metas: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for field in FIELDS:
        mat, idx, meta, source = _build_field_state(profiles, field, None, base_settings, shared_valid_day_index)
        states[field] = _full_matrix(mat, idx, n_days)
        metas[field] = _feature_meta_rows(pd.DataFrame(meta), field, states[field].shape[1])
        sources[field] = source
    return smoothed_path, profiles, states, metas, sources, shared_valid_day_index, n_years, n_days


def _write_state_summary(settings: V7QSettings, states: dict[str, np.ndarray], metas: dict[str, pd.DataFrame], sources: dict[str, str], smoothed_path: Path, base_settings: Any) -> pd.DataFrame:
    rows = []
    for field, arr in states.items():
        meta = metas[field]
        coord = pd.to_numeric(meta.get("feature_coordinate", pd.Series(dtype=float)), errors="coerce")
        rows.append({
            "field": field,
            "n_days": int(arr.shape[0]),
            "n_features": int(arr.shape[1]),
            "feature_metadata_status": str(meta.get("metadata_status", pd.Series(["missing"])).iloc[0]) if not meta.empty else "missing",
            "feature_axis_type": str(meta.get("feature_type", pd.Series(["feature_index"])).iloc[0]) if not meta.empty else "feature_index",
            "feature_coordinate_min": float(np.nanmin(coord)) if coord.size and np.isfinite(coord).any() else np.nan,
            "feature_coordinate_max": float(np.nanmax(coord)) if coord.size and np.isfinite(coord).any() else np.nan,
            "state_nan_fraction": float((~np.isfinite(arr)).mean()) if arr.size else np.nan,
            "state_builder_source": sources.get(field, "unknown"),
            "smoothed_fields_path": str(smoothed_path),
            "state_rebuild_status": "success",
        })
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_feature_state_rebuild_summary_v7_q.csv")
    npz_path = settings.output_dir / "w45_feature_state_rebuild_v7_q.npz"
    _ensure_dir(npz_path.parent)
    payload = {f"{field}_state": states[field] for field in FIELDS}
    payload["days"] = np.arange(next(iter(states.values())).shape[0], dtype=int)
    np.savez_compressed(npz_path, **payload)
    return df


def _build_feature_prepost(states: dict[str, np.ndarray], metas: dict[str, pd.DataFrame], settings: V7QSettings) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    prepost: dict[str, dict[str, np.ndarray]] = {}
    for field, full in states.items():
        pre, post, d = _prepost_vectors(full, settings)
        prepost[field] = {"pre": pre, "post": post, "d": d}
        abs_d = np.abs(d)
        total_abs = np.nansum(abs_d)
        days = np.arange(full.shape[0])
        pre_mask = (days >= settings.pre_start) & (days <= settings.pre_end)
        post_mask = (days >= settings.post_start) & (days <= settings.post_end)
        meta = metas[field].reset_index(drop=True)
        for j in range(full.shape[1]):
            pre_std = float(np.nanstd(full[pre_mask, j]))
            post_std = float(np.nanstd(full[post_mask, j]))
            noise = np.nanmean([pre_std, post_std])
            rel = float(abs_d[j] / total_abs) if total_abs > EPS and np.isfinite(abs_d[j]) else np.nan
            snr = float(abs_d[j] / (noise + EPS)) if np.isfinite(abs_d[j]) else np.nan
            rows.append({
                "field": field,
                "feature_id": int(meta.loc[j, "feature_id"]) if j < len(meta) else j,
                "feature_coordinate": meta.loc[j, "feature_coordinate"] if j < len(meta) else j,
                "feature_type": meta.loc[j, "feature_type"] if j < len(meta) else "feature_index",
                "pre_mean": pre[j],
                "post_mean": post[j],
                "dF": d[j],
                "abs_dF": abs_d[j],
                "relative_abs_dF_contribution": rel,
                "signed_contribution_fraction": float(d[j] / (np.nansum(np.abs(d)) + EPS)) if np.isfinite(d[j]) else np.nan,
                "pre_internal_std": pre_std,
                "post_internal_std": post_std,
                "feature_signal_to_noise": snr,
                "feature_transition_amplitude_label": _label_amplitude(abs_d[j], rel, snr),
            })
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_feature_prepost_contribution_v7_q.csv")
    return df, prepost


def _observed_feature_markers(states: dict[str, np.ndarray], prepost: dict[str, dict[str, np.ndarray]], prepost_df: pd.DataFrame, metas: dict[str, pd.DataFrame], settings: V7QSettings) -> pd.DataFrame:
    rows = []
    days = np.arange(settings.analysis_start, settings.analysis_end + 1, dtype=int)
    for field in FIELDS:
        full = states[field]
        progress = _feature_progress_matrix(full, prepost[field]["pre"], prepost[field]["post"])
        sub = progress[days, :]
        markers = _feature_markers_from_progress(days, sub, settings)
        meta = metas[field].reset_index(drop=True)
        pp = prepost_df[prepost_df["field"] == field].reset_index(drop=True)
        for j in range(full.shape[1]):
            n_reliable = sum(np.isfinite(markers.get(m, np.array([]))[j]) for m in ["departure90", "t10", "t25", "t50", "t75", "peak_smooth3"] if m in markers)
            row = {
                "field": field,
                "feature_id": int(meta.loc[j, "feature_id"]) if j < len(meta) else j,
                "feature_coordinate": meta.loc[j, "feature_coordinate"] if j < len(meta) else j,
                "feature_type": meta.loc[j, "feature_type"] if j < len(meta) else "feature_index",
                "dF": pp.loc[j, "dF"] if j < len(pp) else np.nan,
                "abs_dF": pp.loc[j, "abs_dF"] if j < len(pp) else np.nan,
                "relative_abs_dF_contribution": pp.loc[j, "relative_abs_dF_contribution"] if j < len(pp) else np.nan,
                "feature_signal_to_noise": pp.loc[j, "feature_signal_to_noise"] if j < len(pp) else np.nan,
                "feature_transition_amplitude_label": pp.loc[j, "feature_transition_amplitude_label"] if j < len(pp) else "unknown",
            }
            for name in MARKER_NAMES:
                row[name] = markers.get(name, np.full(full.shape[1], np.nan))[j]
            row["marker_valid_fraction"] = np.nanmean([np.isfinite(row.get(m, np.nan)) for m in TIMING_MARKERS])
            row["edge_hit_fraction"] = np.nanmean([row.get(m) in {settings.analysis_start, settings.analysis_end} for m in TIMING_MARKERS if np.isfinite(row.get(m, np.nan))]) if TIMING_MARKERS else np.nan
            row["feature_marker_reliability"] = "observed_only_pending_bootstrap"
            row["feature_process_type"] = _label_feature_process(row["feature_transition_amplitude_label"], int(n_reliable))
            row["feature_interpretation"] = "Observed feature-level timing marker; use bootstrap summary for reliability."
            rows.append(row)
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_feature_process_markers_v7_q.csv")
    return df


def _bootstrap_feature_marker_samples(profiles: dict, shared_valid_day_index: np.ndarray, n_days: int, settings: V7QSettings, base_settings: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_years = int(np.asarray(profiles[FIELDS[0]].raw_cube).shape[0])
    boot_indices = _bootstrap_indices(settings, n_years)
    sample_rows = []
    for b, sample_idx in enumerate(boot_indices):
        if b % max(1, settings.progress_every) == 0:
            print(f"[V7-q bootstrap feature markers] {b}/{len(boot_indices)}")
        for field in FIELDS:
            mat, idx, meta, _source = _build_field_state(profiles, field, sample_idx, base_settings, shared_valid_day_index)
            full = _full_matrix(mat, idx, n_days)
            pre, post, _d = _prepost_vectors(full, settings)
            progress = _feature_progress_matrix(full, pre, post)
            days = np.arange(settings.analysis_start, settings.analysis_end + 1, dtype=int)
            markers = _feature_markers_from_progress(days, progress[days, :], settings)
            fmeta = _feature_meta_rows(pd.DataFrame(meta), field, full.shape[1]).reset_index(drop=True)
            for j in range(full.shape[1]):
                row = {"bootstrap_id": int(b), "field": field, "feature_id": int(fmeta.loc[j, "feature_id"]) if j < len(fmeta) else j}
                for name in MARKER_NAMES:
                    row[name] = markers.get(name, np.full(full.shape[1], np.nan))[j]
                sample_rows.append(row)
    samples = pd.DataFrame(sample_rows)
    _write_csv(samples, settings.output_dir / "w45_feature_marker_bootstrap_samples_v7_q.csv")
    # Long summary per feature x marker.
    rows = []
    for (field, fid), g in samples.groupby(["field", "feature_id"], dropna=False):
        for marker in MARKER_NAMES:
            if marker not in g.columns:
                continue
            st = _stats(g[marker].to_numpy(dtype=float))
            rows.append({
                "field": field,
                "feature_id": fid,
                "marker": marker,
                **st,
                "same_day_fraction_if_relevant": float((g[marker] == _median(g[marker])).mean()) if g[marker].notna().any() else np.nan,
                "edge_hit_fraction": float(g[marker].isin([settings.analysis_start, settings.analysis_end]).mean()) if g[marker].notna().any() else np.nan,
                "bootstrap_stability_label": _label_bootstrap_marker(st),
            })
    summary = pd.DataFrame(rows)
    _write_csv(summary, settings.output_dir / "w45_feature_marker_bootstrap_summary_v7_q.csv")
    return samples, summary


def _merge_marker_reliability(obs: pd.DataFrame, boot_summary: pd.DataFrame) -> pd.DataFrame:
    # Attach a compact reliability label per feature using key markers.
    key = boot_summary[boot_summary["marker"].isin(["departure90", "t25", "t50", "t75", "peak_smooth3"])].copy()
    if key.empty:
        obs["feature_marker_reliability"] = "bootstrap_unavailable"
        return obs
    counts = key.groupby(["field", "feature_id"]) ["bootstrap_stability_label"].apply(lambda s: int(s.isin(["stable_feature_marker", "moderately_stable_feature_marker"]).sum())).reset_index(name="n_stable_or_moderate_key_markers")
    out = obs.merge(counts, on=["field", "feature_id"], how="left")
    out["n_stable_or_moderate_key_markers"] = out["n_stable_or_moderate_key_markers"].fillna(0).astype(int)
    out["feature_marker_reliability"] = np.where(out["n_stable_or_moderate_key_markers"] >= 3, "usable_feature_marker_family", np.where(out["n_stable_or_moderate_key_markers"] >= 1, "usable_with_caution_feature_marker_family", "unreliable_feature_marker_family"))
    out["feature_process_type"] = [
        _label_feature_process(a, int(n))
        for a, n in zip(out["feature_transition_amplitude_label"], out["n_stable_or_moderate_key_markers"])
    ]
    out["feature_interpretation"] = out["feature_process_type"].map({
        "resolved_transition_feature": "Feature has nontrivial pre/post amplitude and multiple stable timing markers.",
        "transition_feature_with_uncertain_timing": "Feature changes but timing is not strongly resolved.",
        "noisy_transition_feature": "Feature may change but signal-to-noise is weak.",
        "weak_or_low_contribution_feature": "Feature is retained but should not drive strong timing conclusions.",
    }).fillna("Feature retained with unresolved status.")
    return out


def _build_field_timing_distribution(obs: pd.DataFrame, boot_summary: pd.DataFrame, settings: V7QSettings) -> pd.DataFrame:
    rows = []
    merged = obs.copy()
    for field, fg in merged.groupby("field"):
        total_abs = float(np.nansum(np.abs(fg["dF"].to_numpy(dtype=float))))
        weights = np.abs(fg["dF"].to_numpy(dtype=float))
        if not np.isfinite(weights).any() or np.nansum(weights) <= EPS:
            weights = np.ones(len(fg), dtype=float)
        weights = weights / (np.nansum(weights) + EPS)
        strong_mask = fg["feature_transition_amplitude_label"].isin(["strong_transition_feature", "moderate_transition_feature"])
        usable_mask = fg["feature_marker_reliability"].isin(["usable_feature_marker_family", "usable_with_caution_feature_marker_family"])
        for marker in TIMING_MARKERS + ["duration_25_75", "tail_50_75", "early_span_25_50"]:
            vals = pd.to_numeric(fg[marker], errors="coerce").to_numpy(dtype=float) if marker in fg.columns else np.full(len(fg), np.nan)
            valid = np.isfinite(vals)
            valid_strong = valid & strong_mask.to_numpy(dtype=bool)
            st = _stats(vals[valid_strong] if valid_strong.any() else vals[valid])
            # Weighted quantiles are approximated by sorting and cumulative weights.
            wmed = np.nan
            wq25 = np.nan
            wq75 = np.nan
            if valid.any():
                vv = vals[valid]
                ww = weights[valid]
                order = np.argsort(vv)
                vv = vv[order]
                ww = ww[order]
                cdf = np.cumsum(ww) / (np.sum(ww) + EPS)
                wq25 = float(vv[np.searchsorted(cdf, 0.25, side="left").clip(0, len(vv)-1)])
                wmed = float(vv[np.searchsorted(cdf, 0.50, side="left").clip(0, len(vv)-1)])
                wq75 = float(vv[np.searchsorted(cdf, 0.75, side="left").clip(0, len(vv)-1)])
            iqr = st["iqr"]
            if not np.isfinite(iqr):
                hetero = "feature_timing_unresolved"
            elif iqr <= 2:
                hetero = "coherent_feature_transition"
            elif iqr <= 6:
                hetero = "moderately_heterogeneous_transition"
            elif iqr <= 12:
                hetero = "strongly_heterogeneous_transition"
            else:
                hetero = "multi_component_transition"
            rows.append({
                "field": field,
                "marker": marker,
                "n_features": int(len(fg)),
                "n_strong_transition_features": int(strong_mask.sum()),
                "n_moderate_transition_features": int((fg["feature_transition_amplitude_label"] == "moderate_transition_feature").sum()),
                "n_weak_or_unreliable_features": int((~usable_mask).sum()),
                "feature_timing_median": st["median"],
                "feature_timing_q10": st["q10"],
                "feature_timing_q25": st["q25"],
                "feature_timing_q75": st["q75"],
                "feature_timing_q90": st["q90"],
                "feature_timing_iqr": st["iqr"],
                "weighted_timing_median": wmed,
                "weighted_timing_q25": wq25,
                "weighted_timing_q75": wq75,
                "early_feature_fraction": float(np.nanmean(vals < st["median"])) if valid.any() and np.isfinite(st["median"]) else np.nan,
                "late_feature_fraction": float(np.nanmean(vals > st["median"])) if valid.any() and np.isfinite(st["median"]) else np.nan,
                "near_whole_field_fraction": np.nan,  # Whole-field result intentionally not used as input in V7-q.
                "dominant_feature_contribution_fraction": float(np.nanmax(weights)) if weights.size else np.nan,
                "top5_feature_contribution_fraction": float(np.nansum(np.sort(weights)[-5:])) if weights.size else np.nan,
                "heterogeneity_label": hetero,
                "field_internal_process_label": hetero,
                "interpretation": f"{field} {marker}: feature timing distribution classified as {hetero}; weak/noisy features retained but down-weighted in interpretation.",
            })
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_field_feature_timing_distribution_v7_q.csv")
    return df


def _build_field_support_summary(obs: pd.DataFrame, dist: pd.DataFrame, settings: V7QSettings) -> pd.DataFrame:
    rows = []
    for field, fg in obs.groupby("field"):
        # Support lists: highest contribution features among those early in t25 / late in t75 / long tail.
        def support_list(marker: str, mode: str) -> str:
            if marker not in fg.columns:
                return "none"
            g = fg.copy()
            g[marker] = pd.to_numeric(g[marker], errors="coerce")
            if mode == "early":
                cut = np.nanquantile(g[marker], 0.25) if g[marker].notna().any() else np.nan
                sub = g[g[marker] <= cut]
            elif mode == "late":
                cut = np.nanquantile(g[marker], 0.75) if g[marker].notna().any() else np.nan
                sub = g[g[marker] >= cut]
            else:
                sub = g.sort_values("relative_abs_dF_contribution", ascending=False)
            sub = sub.sort_values("relative_abs_dF_contribution", ascending=False).head(8)
            labels = []
            for _, r in sub.iterrows():
                labels.append(f"feature={r['feature_id']} coord={r.get('feature_coordinate', np.nan)} contrib={r.get('relative_abs_dF_contribution', np.nan):.3g}")
            return " | ".join(labels) if labels else "none"
        metadata_status = fg.get("feature_type", pd.Series(["feature_index"])).iloc[0]
        rows.append({
            "field": field,
            "early_support_features": support_list("t25", "early"),
            "late_support_features": support_list("t75", "late"),
            "tail_support_features": support_list("tail_50_75", "late"),
            "dominant_support_features": support_list("t25", "dominant"),
            "early_support_fraction": np.nan,
            "tail_support_fraction": np.nan,
            "feature_metadata_status_for_physical_region": "available" if metadata_status != "feature_index" else "feature_metadata_insufficient_for_physical_region_interpretation",
            "feature_process_summary": "Feature support summary ranks components by timing and dF contribution; it is not a physical-region conclusion unless metadata are sufficient.",
            "needs_spatial_or_component_followup": True,
            "do_not_overinterpret": "Do not infer physical region/causality from feature IDs without metadata and follow-up maps/components.",
        })
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_field_feature_support_summary_v7_q.csv")
    return df


def _overlap_score(a_q25: float, a_q75: float, b_q25: float, b_q75: float) -> float:
    if not all(np.isfinite([a_q25, a_q75, b_q25, b_q75])):
        return np.nan
    inter = max(0.0, min(a_q75, b_q75) - max(a_q25, b_q25))
    union = max(a_q75, b_q75) - min(a_q25, b_q25)
    return float(inter / (union + EPS))


def _label_pair_feature_relation(delta: float, overlap: float, frac_a_earlier: float, frac_b_earlier: float) -> str:
    if not np.isfinite(delta):
        return "feature_relation_unresolved"
    if np.isfinite(overlap) and overlap >= 0.50:
        return "feature_distributions_overlap"
    if delta > 0 and frac_a_earlier >= 0.65:
        return "A_feature_distribution_earlier"
    if delta < 0 and frac_b_earlier >= 0.65:
        return "B_feature_distribution_earlier"
    if delta > 0:
        return "A_partial_feature_advantage"
    if delta < 0:
        return "B_partial_feature_advantage"
    return "feature_relation_mixed"


def _build_pair_feature_distribution_relation(dist: pd.DataFrame, obs: pd.DataFrame, settings: V7QSettings) -> pd.DataFrame:
    rows = []
    obs_key = obs.set_index(["field", "feature_id"])
    for a, b in combinations(FIELDS, 2):
        da = dist[dist["field"] == a].set_index("marker")
        db = dist[dist["field"] == b].set_index("marker")
        fa = obs[obs["field"] == a]
        fb = obs[obs["field"] == b]
        for marker in TIMING_MARKERS + ["duration_25_75", "tail_50_75", "early_span_25_50"]:
            if marker not in da.index or marker not in db.index:
                continue
            ar = da.loc[marker]
            br = db.loc[marker]
            a_vals = pd.to_numeric(fa.get(marker), errors="coerce").to_numpy(dtype=float)
            b_vals = pd.to_numeric(fb.get(marker), errors="coerce").to_numpy(dtype=float)
            a_med = ar["feature_timing_median"]
            b_med = br["feature_timing_median"]
            delta = b_med - a_med if np.isfinite(a_med) and np.isfinite(b_med) else np.nan
            overlap = _overlap_score(ar["feature_timing_q25"], ar["feature_timing_q75"], br["feature_timing_q25"], br["feature_timing_q75"])
            frac_a_earlier = float(np.nanmean(a_vals < b_med)) if np.isfinite(b_med) and np.isfinite(a_vals).any() else np.nan
            frac_b_earlier = float(np.nanmean(b_vals < a_med)) if np.isfinite(a_med) and np.isfinite(b_vals).any() else np.nan
            frac_near = np.nan
            if np.isfinite(a_vals).any() and np.isfinite(b_vals).any():
                # Distribution-level near equality using pairwise combinations; feature counts are small in 2-degree profile space.
                aa = a_vals[np.isfinite(a_vals)]
                bb = b_vals[np.isfinite(b_vals)]
                if aa.size and bb.size:
                    diff = bb[None, :] - aa[:, None]
                    frac_near = float(np.mean(np.abs(diff) <= settings.near_equal_days))
            label = _label_pair_feature_relation(delta, overlap, frac_a_earlier, frac_b_earlier)
            rows.append({
                "field_a": a,
                "field_b": b,
                "marker": marker,
                "A_feature_timing_median": a_med,
                "B_feature_timing_median": b_med,
                "median_delta_B_minus_A": delta,
                "A_feature_q25": ar["feature_timing_q25"],
                "A_feature_q75": ar["feature_timing_q75"],
                "B_feature_q25": br["feature_timing_q25"],
                "B_feature_q75": br["feature_timing_q75"],
                "distribution_overlap_score": overlap,
                "fraction_A_features_earlier_than_B_median": frac_a_earlier,
                "fraction_B_features_earlier_than_A_median": frac_b_earlier,
                "fraction_near_equal_features": frac_near,
                "weighted_fraction_A_earlier": np.nan,
                "weighted_fraction_B_earlier": np.nan,
                "weighted_fraction_near_equal": np.nan,
                "feature_relation_label": label,
                "interpretation": f"{a}-{b} {marker}: {label}; compares feature timing distributions, not a one-to-one physical feature pairing.",
            })
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_pair_feature_distribution_relation_v7_q.csv")
    return df


def _build_h_jw_detail(pair_rel: pd.DataFrame, obs: pd.DataFrame, settings: V7QSettings) -> pd.DataFrame:
    rows = []
    sub = pair_rel[((pair_rel["field_a"] == "H") & (pair_rel["field_b"] == "Jw")) | ((pair_rel["field_a"] == "Jw") & (pair_rel["field_b"] == "H"))].copy()
    if sub.empty:
        df = pd.DataFrame()
        _write_csv(df, settings.output_dir / "w45_H_Jw_feature_relation_detail_v7_q.csv")
        return df
    for _, r in sub.iterrows():
        marker = r["marker"]
        h = obs[obs["field"] == "H"].copy()
        jw = obs[obs["field"] == "Jw"].copy()
        def support(field_df: pd.DataFrame, mode: str) -> str:
            g = field_df.copy()
            if marker not in g.columns:
                return "none"
            g[marker] = pd.to_numeric(g[marker], errors="coerce")
            if mode == "early":
                cut = np.nanquantile(g[marker], 0.25) if g[marker].notna().any() else np.nan
                g = g[g[marker] <= cut]
            else:
                cut = np.nanquantile(g[marker], 0.75) if g[marker].notna().any() else np.nan
                g = g[g[marker] >= cut]
            g = g.sort_values("relative_abs_dF_contribution", ascending=False).head(6)
            labels = [f"feature={x.feature_id} coord={getattr(x, 'feature_coordinate', np.nan)} contrib={getattr(x, 'relative_abs_dF_contribution', np.nan):.3g}" for x in g.itertuples()]
            return " | ".join(labels) if labels else "none"
        rows.append({
            "marker": marker,
            "H_feature_median": r["A_feature_timing_median"] if r["field_a"] == "H" else r["B_feature_timing_median"],
            "Jw_feature_median": r["B_feature_timing_median"] if r["field_a"] == "H" else r["A_feature_timing_median"],
            "median_delta_Jw_minus_H": r["median_delta_B_minus_A"] if r["field_a"] == "H" else -r["median_delta_B_minus_A"],
            "H_q25": r["A_feature_q25"] if r["field_a"] == "H" else r["B_feature_q25"],
            "H_q75": r["A_feature_q75"] if r["field_a"] == "H" else r["B_feature_q75"],
            "Jw_q25": r["B_feature_q25"] if r["field_a"] == "H" else r["A_feature_q25"],
            "Jw_q75": r["B_feature_q75"] if r["field_a"] == "H" else r["A_feature_q75"],
            "overlap_score": r["distribution_overlap_score"],
            "fraction_H_features_earlier": r["fraction_A_features_earlier_than_B_median"] if r["field_a"] == "H" else r["fraction_B_features_earlier_than_A_median"],
            "fraction_Jw_features_earlier": r["fraction_B_features_earlier_than_A_median"] if r["field_a"] == "H" else r["fraction_A_features_earlier_than_B_median"],
            "fraction_near_equal": r["fraction_near_equal_features"],
            "weighted_fraction_H_earlier": np.nan,
            "weighted_fraction_Jw_earlier": np.nan,
            "weighted_fraction_near_equal": np.nan,
            "H_support_features": support(h, "early"),
            "Jw_support_features": support(jw, "early"),
            "relation_label": r["feature_relation_label"],
            "interpretation": f"H/Jw {marker}: {r['feature_relation_label']}; feature distribution comparison, not global lead-lag.",
        })
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_H_Jw_feature_relation_detail_v7_q.csv")
    return df


def _build_early_group_org(dist: pd.DataFrame, pair_rel: pd.DataFrame, settings: V7QSettings) -> pd.DataFrame:
    rows = []
    for field in settings.early_group_fields:
        for marker in ["departure90", "t25", "t50", "t75", "peak_smooth3"]:
            fd = dist[(dist["field"] == field) & (dist["marker"] == marker)]
            if fd.empty:
                continue
            overlaps_h = pair_rel[((pair_rel["marker"] == marker) & (((pair_rel["field_a"] == field) & (pair_rel["field_b"] == "H")) | ((pair_rel["field_b"] == field) & (pair_rel["field_a"] == "H"))))]
            overlaps_jw = pair_rel[((pair_rel["marker"] == marker) & (((pair_rel["field_a"] == field) & (pair_rel["field_b"] == "Jw")) | ((pair_rel["field_b"] == field) & (pair_rel["field_a"] == "Jw"))))]
            h_overlap = float(overlaps_h["distribution_overlap_score"].iloc[0]) if not overlaps_h.empty else (1.0 if field == "H" else np.nan)
            jw_overlap = float(overlaps_jw["distribution_overlap_score"].iloc[0]) if not overlaps_jw.empty else (1.0 if field == "Jw" else np.nan)
            if (np.isfinite(h_overlap) and h_overlap >= 0.5) or (np.isfinite(jw_overlap) and jw_overlap >= 0.5):
                role = "early_group_overlap"
            elif field in {"H", "Jw"}:
                role = "early_core_candidate"
            else:
                role = "broad_or_mixed_feature_role"
            rows.append({
                "field": field,
                "marker": marker,
                "feature_timing_rank_distribution": "computed_by_feature_timing_distribution",
                "prob_field_feature_distribution_earliest": np.nan,
                "prob_field_feature_distribution_latest": np.nan,
                "overlap_with_H": h_overlap,
                "overlap_with_Jw": jw_overlap,
                "early_group_role_label": role,
                "interpretation": f"{field} {marker}: {role}; based on feature distribution overlap with H/Jw.",
            })
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_early_group_feature_organization_v7_q.csv")
    return df


def _build_je_consistency(dist: pd.DataFrame, obs: pd.DataFrame, settings: V7QSettings) -> pd.DataFrame:
    rows = []
    je = obs[obs["field"] == "Je"]
    for marker in ["departure90", "t25", "t50", "t75", "peak_smooth3"]:
        d = dist[(dist["field"] == "Je") & (dist["marker"] == marker)]
        if d.empty:
            continue
        vals = pd.to_numeric(je.get(marker), errors="coerce").to_numpy(dtype=float) if marker in je.columns else np.array([])
        med = float(d["feature_timing_median"].iloc[0])
        iqr = float(d["feature_timing_iqr"].iloc[0])
        late_fraction = float(np.nanmean(vals > med)) if np.isfinite(vals).any() and np.isfinite(med) else np.nan
        label = "Je_feature_late_consistent" if np.isfinite(iqr) and iqr <= 4 else ("Je_feature_mixed_late" if np.isfinite(iqr) else "Je_feature_consistency_unresolved")
        rows.append({
            "marker": marker,
            "Je_feature_median": med,
            "Je_iqr": iqr,
            "late_feature_fraction": late_fraction,
            "dominant_late_feature_contribution": np.nan,
            "Je_late_consistency_label": label,
            "interpretation": f"Je {marker}: {label}; this is a consistency check, not the main research target.",
        })
    df = pd.DataFrame(rows)
    _write_csv(df, settings.output_dir / "w45_Je_feature_consistency_v7_q.csv")
    return df


def _write_summary(settings: V7QSettings, state_summary: pd.DataFrame, dist: pd.DataFrame, pair_rel: pd.DataFrame, h_jw: pd.DataFrame, early_org: pd.DataFrame, je: pd.DataFrame) -> None:
    lines = []
    lines.append(f"# W45 feature process resolution V7-q\n")
    lines.append(f"Created: {_now_iso()}\n")
    lines.append("## Purpose\n")
    lines.append("V7-q raises W45 from whole-field scalar process curves to field x feature/component-level process ensembles. It rebuilds from the current 2-degree interpolated profile/state base and does not read V7-m/n/o/p derived result tables as input.\n")
    lines.append("## Base rebuild\n")
    lines.append(state_summary.to_markdown(index=False) if not state_summary.empty else "State summary unavailable.")
    lines.append("\n## Field feature timing distributions\n")
    core = dist[dist["marker"].isin(["departure90", "t25", "t50", "t75", "peak_smooth3"])]
    if not core.empty:
        cols = ["field", "marker", "feature_timing_median", "feature_timing_iqr", "heterogeneity_label", "field_internal_process_label"]
        lines.append(core[cols].to_markdown(index=False))
    lines.append("\n## H/Jw feature relation detail\n")
    if not h_jw.empty:
        cols = ["marker", "median_delta_Jw_minus_H", "overlap_score", "fraction_H_features_earlier", "fraction_Jw_features_earlier", "fraction_near_equal", "relation_label"]
        lines.append(h_jw[cols].to_markdown(index=False))
    else:
        lines.append("H/Jw feature relation detail unavailable.")
    lines.append("\n## Early group feature organization\n")
    if not early_org.empty:
        lines.append(early_org.head(50).to_markdown(index=False))
    lines.append("\n## Je feature consistency as background reference\n")
    if not je.empty:
        lines.append(je.to_markdown(index=False))
    lines.append("\n## Prohibited interpretations\n")
    lines.append("- Feature-level relation is not causality.\n- Feature timing overlap is not synchronization unless a separate equivalence test is defined.\n- Low-contribution or noisy features are retained and labelled, not deleted.\n- Weighted and unweighted distributions must not be mixed.\n- If feature metadata are insufficient, do not write physical-region conclusions.\n")
    _write_text("\n".join(lines), settings.output_dir / "w45_feature_level_organization_summary_v7_q.md")


def _try_write_figures(settings: V7QSettings, obs: pd.DataFrame, dist: pd.DataFrame, pair_rel: pd.DataFrame, h_jw: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        _ensure_dir(settings.figure_dir)
        # Field feature timing distributions for selected markers.
        for marker in ["departure90", "t25", "t50", "t75"]:
            plt.figure(figsize=(9, 5))
            data = []
            labels = []
            for field in FIELDS:
                vals = pd.to_numeric(obs[(obs["field"] == field)][marker], errors="coerce").dropna().to_numpy(dtype=float) if marker in obs.columns else np.array([])
                if vals.size:
                    data.append(vals)
                    labels.append(field)
            if data:
                plt.boxplot(data, labels=labels, showfliers=False)
                plt.title(f"W45 feature timing distributions: {marker}")
                plt.xlabel("field")
                plt.ylabel("day")
                plt.tight_layout()
                plt.savefig(settings.figure_dir / f"w45_field_feature_timing_{marker}_v7_q.png", dpi=160)
            plt.close()
        # H/Jw overlap summary.
        if not h_jw.empty:
            plt.figure(figsize=(9, 5))
            x = np.arange(len(h_jw))
            plt.bar(x, h_jw["median_delta_Jw_minus_H"].to_numpy(dtype=float))
            plt.xticks(x, h_jw["marker"].astype(str).tolist(), rotation=45, ha="right")
            plt.axhline(0, linewidth=1)
            plt.title("W45 H/Jw feature timing median delta (Jw - H)")
            plt.ylabel("day")
            plt.tight_layout()
            plt.savefig(settings.figure_dir / "w45_H_Jw_feature_timing_overlap_v7_q.png", dpi=160)
            plt.close()
        # Pair feature distribution heatmap for t25.
        t25 = pair_rel[pair_rel["marker"] == "t25"].copy()
        if not t25.empty:
            fields = FIELDS
            mat = np.full((len(fields), len(fields)), np.nan)
            for _, r in t25.iterrows():
                i = fields.index(r["field_a"])
                j = fields.index(r["field_b"])
                mat[i, j] = r["median_delta_B_minus_A"]
                mat[j, i] = -r["median_delta_B_minus_A"]
            plt.figure(figsize=(7, 6))
            im = plt.imshow(mat)
            plt.colorbar(im, label="median delta (col - row), day")
            plt.xticks(np.arange(len(fields)), fields)
            plt.yticks(np.arange(len(fields)), fields)
            plt.title("W45 pair feature timing distribution delta, t25")
            plt.tight_layout()
            plt.savefig(settings.figure_dir / "w45_pair_feature_distribution_heatmap_v7_q.png", dpi=160)
            plt.close()
    except Exception as exc:  # noqa: BLE001
        _write_text(f"Figure generation failed: {exc}\n", settings.log_dir / "figure_generation_error_v7_q.log")


def _write_run_meta(settings: V7QSettings, status: str = "success") -> None:
    meta = {
        "version": OUTPUT_TAG,
        "created_at": _now_iso(),
        "status": status,
        "window_id": settings.window_id,
        "anchor_day": settings.anchor_day,
        "fields": list(settings.fields),
        "input_representation": "current_2deg_interpolated_profile_state",
        "v7_m_outputs_used_as_input": False,
        "v7_n_outputs_used_as_input": False,
        "v7_o_outputs_used_as_input": False,
        "v7_p_outputs_used_as_input": False,
        "feature_level_resolution": True,
        "whole_field_relation_cards_used_as_input": False,
        "n_bootstrap": int(settings.n_bootstrap),
        "outputs": [
            "input_base_audit_v7_q.json",
            "w45_feature_process_markers_v7_q.csv",
            "w45_field_feature_timing_distribution_v7_q.csv",
            "w45_pair_feature_distribution_relation_v7_q.csv",
            "w45_H_Jw_feature_relation_detail_v7_q.csv",
            "w45_feature_level_organization_summary_v7_q.md",
        ],
    }
    _write_json(meta, settings.output_dir / "run_meta.json")


def run_w45_feature_process_resolution_v7_q(v7_root: Optional[Path] = None) -> None:
    settings, base_settings = _resolve_settings(v7_root)
    _ensure_dir(settings.output_dir)
    _ensure_dir(settings.log_dir)
    _ensure_dir(settings.figure_dir)
    print("[1/11] audit base inputs and rebuild feature states")
    _audit_no_forbidden_inputs(settings)
    smoothed_path, profiles, states, metas, sources, shared_valid_day_index, n_years, n_days = _build_states_from_base(settings, base_settings)
    state_summary = _write_state_summary(settings, states, metas, sources, smoothed_path, base_settings)
    audit = {
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
        "v7_p_outputs_used_as_input": False,
        "analysis_window": [settings.analysis_start, settings.analysis_end],
        "pre_period": [settings.pre_start, settings.pre_end],
        "post_period": [settings.post_start, settings.post_end],
        "n_years": int(n_years),
        "n_days": int(n_days),
        "state_rebuild_status": "success",
        "field_state_builder_sources": sources,
        "smoothed_fields_path": str(smoothed_path),
    }
    _write_json(audit, settings.output_dir / "input_base_audit_v7_q.json")

    print("[2/11] compute feature pre/post contributions")
    prepost_df, prepost = _build_feature_prepost(states, metas, settings)

    print("[3/11] compute observed feature markers")
    obs_markers = _observed_feature_markers(states, prepost, prepost_df, metas, settings)

    print("[4/11] bootstrap feature marker stability")
    boot_samples, boot_summary = _bootstrap_feature_marker_samples(profiles, shared_valid_day_index, n_days, settings, base_settings)

    print("[5/11] merge feature reliability")
    obs_markers = _merge_marker_reliability(obs_markers, boot_summary)
    _write_csv(obs_markers, settings.output_dir / "w45_feature_process_markers_v7_q.csv")

    print("[6/11] build field feature timing distributions")
    field_dist = _build_field_timing_distribution(obs_markers, boot_summary, settings)

    print("[7/11] build field feature support summaries")
    field_support = _build_field_support_summary(obs_markers, field_dist, settings)

    print("[8/11] build pair feature distribution relations")
    pair_rel = _build_pair_feature_distribution_relation(field_dist, obs_markers, settings)

    print("[9/11] build H/Jw, early-group, and Je detail tables")
    h_jw_detail = _build_h_jw_detail(pair_rel, obs_markers, settings)
    early_org = _build_early_group_org(field_dist, pair_rel, settings)
    je_consistency = _build_je_consistency(field_dist, obs_markers, settings)

    print("[10/11] write summary and figures")
    _write_summary(settings, state_summary, field_dist, pair_rel, h_jw_detail, early_org, je_consistency)
    _try_write_figures(settings, obs_markers, field_dist, pair_rel, h_jw_detail)

    print("[11/11] write run metadata")
    _write_run_meta(settings, status="success")
    print(f"[done] V7-q outputs written to {settings.output_dir}")
