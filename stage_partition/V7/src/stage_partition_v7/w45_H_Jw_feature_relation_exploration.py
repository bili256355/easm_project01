from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

OUTPUT_TAG = "w45_H_Jw_feature_relation_exploration_v7_r"
INPUT_Q_TAG = "w45_feature_process_resolution_v7_q"
WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELDS = ("H", "Jw")
TIMING_MARKERS = [
    "departure90", "departure95",
    "t10", "t15", "t20", "t25", "t30", "t35", "t40", "t45", "t50", "t75",
    "peak_raw", "peak_smooth3",
]
EARLY_MARKERS = ["t10", "t15", "t20", "t25", "t30", "t35", "t40", "t45", "t50"]
DEPARTURE_MARKERS = ["departure90", "departure95", "t10"]
CATCHUP_MARKERS = ["t50", "t75", "peak_smooth3", "duration_25_75", "tail_50_75", "early_span_25_50"]
SPAN_MARKERS = ["duration_25_75", "tail_50_75", "early_span_25_50"]
ALL_MARKERS = DEPARTURE_MARKERS + [m for m in EARLY_MARKERS if m not in DEPARTURE_MARKERS] + [m for m in CATCHUP_MARKERS if m not in DEPARTURE_MARKERS + EARLY_MARKERS]
EPS = 1e-12


@dataclass
class V7RSettings:
    v7_root: Path
    input_q_dir: Path
    output_dir: Path
    log_dir: Path
    figure_dir: Path
    window_id: str = WINDOW_ID
    anchor_day: int = ANCHOR_DAY
    near_equal_days: float = 1.0
    high_overlap_threshold: float = 0.60
    broad_support_threshold: float = 0.60
    moderate_support_threshold: float = 0.40
    read_v7q_outputs_as_diagnostic_input: bool = True
    diagnostic_branch: bool = True


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict[str, Any], path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _safe_arr(values: Iterable[Any] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return arr[np.isfinite(arr)]


def _q(values: Iterable[Any] | np.ndarray, q: float) -> float:
    arr = _safe_arr(values)
    return float(np.nanquantile(arr, q)) if arr.size else np.nan


def _median(values: Iterable[Any] | np.ndarray) -> float:
    arr = _safe_arr(values)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def _mean(values: Iterable[Any] | np.ndarray) -> float:
    arr = _safe_arr(values)
    return float(np.nanmean(arr)) if arr.size else np.nan


def _valid_fraction(values: Iterable[Any] | np.ndarray) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return float(np.isfinite(arr).sum() / arr.size) if arr.size else np.nan


def _weighted_median(values: Iterable[Any] | np.ndarray, weights: Iterable[Any] | np.ndarray) -> float:
    v = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    w = np.asarray(list(weights) if not isinstance(weights, np.ndarray) else weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    c = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(c, 0.5, side="left")])


def _weighted_fraction(mask_values: Iterable[Any] | np.ndarray, weights: Iterable[Any] | np.ndarray) -> float:
    m = np.asarray(list(mask_values) if not isinstance(mask_values, np.ndarray) else mask_values)
    w = np.asarray(list(weights) if not isinstance(weights, np.ndarray) else weights, dtype=float)
    good = np.isfinite(w) & (w > 0)
    if not good.any():
        return np.nan
    # mask may contain nan-like values; cast finite bools only.
    mb = np.asarray(m, dtype=object)[good]
    ww = w[good]
    bool_mask = np.array([bool(x) if x is not None and str(x).lower() != "nan" else False for x in mb], dtype=bool)
    return float(np.sum(ww[bool_mask]) / np.sum(ww))


def _interval_overlap(a: np.ndarray, b: np.ndarray) -> float:
    a = _safe_arr(a)
    b = _safe_arr(b)
    if not a.size or not b.size:
        return np.nan
    a1, a2 = np.nanquantile(a, [0.25, 0.75])
    b1, b2 = np.nanquantile(b, [0.25, 0.75])
    if abs(a2 - a1) < EPS and abs(b2 - b1) < EPS:
        return 1.0 if abs(a1 - b1) <= EPS else 0.0
    lo = max(a1, b1)
    hi = min(a2, b2)
    inter = max(0.0, hi - lo)
    union = max(a2, b2) - min(a1, b1)
    if union <= EPS:
        return 1.0 if inter <= EPS else 0.0
    return float(inter / union)


def _support_label(fraction: float, weighted_fraction: float, *, broad_label: str, core_label: str, limited_label: str, none_label: str) -> str:
    candidates = [x for x in [fraction, weighted_fraction] if np.isfinite(x)]
    if not candidates:
        return "insufficient_feature_evidence"
    best = max(candidates)
    if best >= 0.60:
        return broad_label
    if best >= 0.40:
        return core_label
    if best > 0.0:
        return limited_label
    return none_label


def _list_support_features(df: pd.DataFrame, marker: str, support_mask: pd.Series, max_n: int = 8) -> str:
    if marker not in df.columns:
        return "none"
    sub = df.loc[support_mask].copy()
    if sub.empty:
        return "none"
    sub = sub.sort_values("relative_abs_dF_contribution", ascending=False, na_position="last").head(max_n)
    parts: list[str] = []
    for _, r in sub.iterrows():
        coord = r.get("feature_coordinate", np.nan)
        contrib = r.get("relative_abs_dF_contribution", np.nan)
        role = r.get("feature_use_role", r.get("feature_transition_amplitude_label", ""))
        parts.append(f"feature={int(r['feature_id'])} coord={coord} contrib={contrib:.4g} role={role}")
    return " | ".join(parts) if parts else "none"


def _resolve_settings(v7_root: Optional[Path]) -> V7RSettings:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    input_q_dir = v7_root / "outputs" / INPUT_Q_TAG
    output_dir = v7_root / "outputs" / OUTPUT_TAG
    return V7RSettings(
        v7_root=v7_root,
        input_q_dir=input_q_dir,
        output_dir=output_dir,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=output_dir / "figures",
    )


def _load_inputs(settings: V7RSettings) -> dict[str, pd.DataFrame]:
    required = {
        "prepost": "w45_feature_prepost_contribution_v7_q.csv",
        "markers": "w45_feature_process_markers_v7_q.csv",
        "boot_summary": "w45_feature_marker_bootstrap_summary_v7_q.csv",
        "timing_dist": "w45_field_feature_timing_distribution_v7_q.csv",
        "hjw_detail_q": "w45_H_Jw_feature_relation_detail_v7_q.csv",
        "state_summary": "w45_feature_state_rebuild_summary_v7_q.csv",
    }
    missing = [name for name in required.values() if not (settings.input_q_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "V7-r expects V7-q feature-level outputs as diagnostic input. Missing: " + ", ".join(missing)
        )
    return {k: pd.read_csv(settings.input_q_dir / v) for k, v in required.items()}


def _input_audit(settings: V7RSettings, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    fields_available = sorted(data["markers"]["field"].dropna().unique().tolist())
    hjw_ok = all(f in fields_available for f in FIELDS)
    return {
        "version": OUTPUT_TAG,
        "created_at": _now_iso(),
        "window_id": settings.window_id,
        "anchor_day": settings.anchor_day,
        "purpose": "H/Jw diagnostic exploration from V7-q feature-level outputs",
        "diagnostic_branch": settings.diagnostic_branch,
        "input_q_dir": str(settings.input_q_dir),
        "read_v7q_outputs_as_diagnostic_input": settings.read_v7q_outputs_as_diagnostic_input,
        "uses_v7_m_outputs": False,
        "uses_v7_n_outputs": False,
        "uses_v7_o_outputs": False,
        "uses_v7_p_outputs": False,
        "fields_required": list(FIELDS),
        "fields_available": fields_available,
        "H_Jw_available": bool(hjw_ok),
        "n_marker_rows": int(len(data["markers"])),
        "n_bootstrap_summary_rows": int(len(data["boot_summary"])),
        "interpretation_boundary": "This is an H/Jw exploration branch. It may read V7-q feature outputs but must not be treated as a clean trunk rebuild.",
    }


def _provenance_audit(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    state = data["state_summary"].copy()
    pre = data["prepost"].copy()
    rows = []
    for field in FIELDS:
        s = state[state["field"] == field]
        p = pre[pre["field"] == field]
        feature_axis = s["feature_axis_type"].iloc[0] if not s.empty and "feature_axis_type" in s else "unknown"
        meta_status = s["feature_metadata_status"].iloc[0] if not s.empty and "feature_metadata_status" in s else "unknown"
        for _, r in p.iterrows():
            ftype = str(r.get("feature_type", "unknown"))
            physical_coordinate = ftype in {"lat_value", "latitude", "coordinate"} and ftype != "feature_index"
            if physical_coordinate:
                label = "physical_coordinate_available"
                limit = "Coordinate-level interpretation may be possible after verifying profile builder semantics."
            elif meta_status == "available":
                label = "feature_index_only"
                limit = "Feature index is available, but physical region interpretation is not allowed without provenance mapping."
            else:
                label = "metadata_insufficient"
                limit = "Feature metadata insufficient; only feature/component-level interpretation is allowed."
            rows.append({
                "field": field,
                "feature_id": int(r["feature_id"]),
                "feature_coordinate": r.get("feature_coordinate", np.nan),
                "feature_axis_type": feature_axis,
                "feature_type": ftype,
                "feature_physical_meaning": "latitude_or_coordinate_candidate" if physical_coordinate else "feature_index_not_physical_region",
                "is_latitude_feature": bool(ftype in {"lat_value", "latitude"}),
                "is_component_feature": bool(ftype not in {"lat_value", "latitude"} and ftype != "feature_index"),
                "can_map_to_physical_region": bool(physical_coordinate),
                "provenance_label": label,
                "interpretation_limit": limit,
            })
    return pd.DataFrame(rows)


def _eligibility_audit(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    markers = data["markers"].copy()
    boot = data["boot_summary"].copy()
    # Summarise marker instability for each feature across key markers.
    key = boot[boot["marker"].isin(["departure90", "departure95", "t25", "t50", "t75", "peak_smooth3"])].copy()
    agg = key.groupby(["field", "feature_id"], as_index=False).agg(
        marker_q90_width_mean=("q90_width", "mean"),
        marker_valid_fraction=("valid_fraction", "mean"),
        edge_hit_fraction=("edge_hit_fraction", "mean"),
        n_stable_markers=("bootstrap_stability_label", lambda x: int(sum(str(v).startswith("stable") for v in x))),
    )
    base = markers[markers["field"].isin(FIELDS)].copy()
    base = base.merge(agg, on=["field", "feature_id"], how="left")
    rows = []
    for field, sub in base.groupby("field"):
        contrib = sub["relative_abs_dF_contribution"].astype(float)
        q75 = np.nanquantile(contrib, 0.75) if np.isfinite(contrib).any() else np.nan
        q50 = np.nanquantile(contrib, 0.50) if np.isfinite(contrib).any() else np.nan
        for _, r in sub.iterrows():
            label = str(r.get("feature_transition_amplitude_label", ""))
            reliability = str(r.get("feature_marker_reliability", ""))
            c = float(r.get("relative_abs_dF_contribution", np.nan))
            valid = float(r.get("marker_valid_fraction", np.nan))
            width = float(r.get("marker_q90_width_mean", np.nan))
            if label.startswith("strong") and np.isfinite(c) and c >= q75 and "unusable" not in reliability:
                role = "core_support_feature"
            elif label.startswith(("strong", "moderate")) and np.isfinite(c) and c >= q50 and "unusable" not in reliability:
                role = "secondary_support_feature"
            elif "low" in label:
                role = "low_contribution_feature"
            elif "noisy" in label or "unusable" in reliability or (np.isfinite(valid) and valid < 0.6):
                role = "noisy_unreliable_feature"
            else:
                role = "weak_but_retained_feature"
            rows.append({
                "field": field,
                "feature_id": int(r["feature_id"]),
                "feature_coordinate": r.get("feature_coordinate", np.nan),
                "dF": r.get("dF", np.nan),
                "abs_dF": r.get("abs_dF", np.nan),
                "relative_abs_dF_contribution": c,
                "feature_signal_to_noise": r.get("feature_signal_to_noise", np.nan),
                "marker_valid_fraction": valid,
                "marker_q90_width_mean": width,
                "edge_hit_fraction": r.get("edge_hit_fraction", np.nan),
                "n_stable_markers": r.get("n_stable_markers", np.nan),
                "feature_transition_amplitude_label": label,
                "feature_marker_reliability": reliability,
                "feature_use_role": role,
            })
    return pd.DataFrame(rows)


def _merge_eligible(data: dict[str, pd.DataFrame], eligibility: pd.DataFrame) -> pd.DataFrame:
    markers = data["markers"].copy()
    cols = ["field", "feature_id", "feature_use_role", "marker_q90_width_mean", "n_stable_markers"]
    return markers.merge(eligibility[cols], on=["field", "feature_id"], how="left")


def _field_marker_values(df: pd.DataFrame, field: str, marker: str) -> pd.DataFrame:
    sub = df[df["field"] == field].copy()
    if marker not in sub.columns:
        sub[marker] = np.nan
    return sub


def _dist_compare(df: pd.DataFrame, marker: str, near_equal_days: float) -> dict[str, Any]:
    h = _field_marker_values(df, "H", marker)
    j = _field_marker_values(df, "Jw", marker)
    hv = h[marker].to_numpy(dtype=float)
    jv = j[marker].to_numpy(dtype=float)
    hw = h["relative_abs_dF_contribution"].to_numpy(dtype=float)
    jw = j["relative_abs_dF_contribution"].to_numpy(dtype=float)
    h_med = _median(hv)
    j_med = _median(jv)
    h_earlier = hv < j_med
    j_earlier = jv < h_med
    h_near = np.abs(hv - j_med) <= near_equal_days
    j_near = np.abs(jv - h_med) <= near_equal_days
    # Report both field-relative fractions.  Combined near-equal is mean of valid H/Jw near fractions.
    h_frac = float(np.nanmean(h_earlier)) if h_earlier.size else np.nan
    j_frac = float(np.nanmean(j_earlier)) if j_earlier.size else np.nan
    h_near_frac = float(np.nanmean(h_near)) if h_near.size else np.nan
    j_near_frac = float(np.nanmean(j_near)) if j_near.size else np.nan
    return {
        "marker": marker,
        "H_median": h_med,
        "Jw_median": j_med,
        "median_delta_Jw_minus_H": j_med - h_med if np.isfinite(h_med) and np.isfinite(j_med) else np.nan,
        "H_q25": _q(hv, 0.25),
        "H_q75": _q(hv, 0.75),
        "Jw_q25": _q(jv, 0.25),
        "Jw_q75": _q(jv, 0.75),
        "overlap_score": _interval_overlap(hv, jv),
        "fraction_H_features_earlier": h_frac,
        "fraction_Jw_features_earlier": j_frac,
        "fraction_near_equal": np.nanmean([h_near_frac, j_near_frac]),
        "weighted_fraction_H_earlier": _weighted_fraction(h_earlier, hw),
        "weighted_fraction_Jw_earlier": _weighted_fraction(j_earlier, jw),
        "weighted_fraction_near_equal": np.nanmean([_weighted_fraction(h_near, hw), _weighted_fraction(j_near, jw)]),
        "weighted_H_median": _weighted_median(hv, hw),
        "weighted_Jw_median": _weighted_median(jv, jw),
    }


def _departure_same_phase(df: pd.DataFrame, settings: V7RSettings) -> pd.DataFrame:
    rows = []
    for marker in DEPARTURE_MARKERS:
        d = _dist_compare(df, marker, settings.near_equal_days)
        near = d["weighted_fraction_near_equal"] if np.isfinite(d["weighted_fraction_near_equal"]) else d["fraction_near_equal"]
        if np.isfinite(near) and near >= settings.high_overlap_threshold and abs(d["median_delta_Jw_minus_H"]) <= settings.near_equal_days:
            label = "strong_same_phase_candidate"
        elif np.isfinite(near) and near >= settings.moderate_support_threshold:
            label = "moderate_same_phase_candidate"
        else:
            label = "same_phase_not_established"
        d["same_phase_candidate_label"] = label
        d["interpretation"] = f"H/Jw {marker}: {label}; near-equal means |feature timing - opposite-field median| <= {settings.near_equal_days} day, not synchrony."
        rows.append(d)
    return pd.DataFrame(rows)


def _H_early_support(df: pd.DataFrame, settings: V7RSettings) -> pd.DataFrame:
    rows = []
    for marker in EARLY_MARKERS:
        d = _dist_compare(df, marker, settings.near_equal_days)
        h = _field_marker_values(df, "H", marker)
        j_med = d["Jw_median"]
        support_mask = h[marker].astype(float) < j_med if np.isfinite(j_med) else pd.Series(False, index=h.index)
        core_mask = support_mask & h["feature_use_role"].isin(["core_support_feature", "secondary_support_feature"])
        low_mask = support_mask & h["feature_use_role"].isin(["low_contribution_feature", "noisy_unreliable_feature", "weak_but_retained_feature"])
        frac_core = float(core_mask.sum() / max(1, h["feature_use_role"].isin(["core_support_feature", "secondary_support_feature"]).sum()))
        frac_low = float(low_mask.sum() / max(1, h["feature_use_role"].isin(["low_contribution_feature", "noisy_unreliable_feature", "weak_but_retained_feature"]).sum()))
        label = _support_label(
            d["fraction_H_features_earlier"],
            d["weighted_fraction_H_earlier"],
            broad_label="broad_feature_support",
            core_label="core_feature_supported",
            limited_label="limited_feature_driven",
            none_label="not_feature_supported",
        )
        if label in {"broad_feature_support", "core_feature_supported"} and frac_core < 0.4:
            label = "support_not_core_feature_dominated"
        d.update({
            "fraction_core_H_features_earlier": frac_core,
            "fraction_secondary_or_low_H_features_earlier": frac_low,
            "H_early_support_feature_ids": _list_support_features(h, marker, support_mask),
            "H_early_support_contribution_sum": float(np.nansum(h.loc[support_mask, "relative_abs_dF_contribution"].astype(float))) if support_mask.any() else 0.0,
            "support_pattern_label": label,
            "interpretation": f"H early-progress vs Jw at {marker}: {label}; compares H feature timing to Jw feature median, not global lead-lag.",
        })
        rows.append(d)
    return pd.DataFrame(rows)


def _Jw_catchup_support(df: pd.DataFrame, settings: V7RSettings) -> pd.DataFrame:
    rows = []
    for marker in CATCHUP_MARKERS:
        d = _dist_compare(df, marker, settings.near_equal_days)
        h_med = d["H_median"]
        jw = _field_marker_values(df, "Jw", marker)
        if marker in SPAN_MARKERS:
            # For duration/tail/span, smaller Jw value means more compact / catch-up-like relative to H.
            support_mask = jw[marker].astype(float) < h_med if np.isfinite(h_med) else pd.Series(False, index=jw.index)
        else:
            # For timing markers, earlier-or-near Jw relative to H median supports catch-up.
            support_mask = jw[marker].astype(float) <= (h_med + settings.near_equal_days) if np.isfinite(h_med) else pd.Series(False, index=jw.index)
        fraction = float(support_mask.mean()) if len(support_mask) else np.nan
        weighted = _weighted_fraction(support_mask.to_numpy(), jw["relative_abs_dF_contribution"].to_numpy(dtype=float))
        label = _support_label(
            fraction,
            weighted,
            broad_label="broad_feature_catchup_support",
            core_label="core_feature_catchup_support",
            limited_label="limited_feature_catchup",
            none_label="no_feature_catchup_support",
        )
        d.update({
            "fraction_Jw_features_earlier_or_more_compact": fraction,
            "weighted_fraction_Jw_features_earlier_or_more_compact": weighted,
            "Jw_catchup_support_feature_ids": _list_support_features(jw, marker, support_mask),
            "Jw_catchup_support_contribution_sum": float(np.nansum(jw.loc[support_mask, "relative_abs_dF_contribution"].astype(float))) if support_mask.any() else 0.0,
            "catchup_support_label": label,
            "interpretation": f"Jw catch-up/finish support at {marker}: {label}; timing markers use earlier-or-near, span markers use smaller duration/tail.",
        })
        rows.append(d)
    return pd.DataFrame(rows)


def _overlap_by_phase(df: pd.DataFrame, settings: V7RSettings) -> pd.DataFrame:
    phase_map = {
        "departure_phase": ["departure90", "departure95"],
        "early_progress_phase": ["t10", "t15", "t20", "t25", "t30"],
        "mid_progress_phase": ["t35", "t40", "t45", "t50"],
        "finish_phase": ["t75", "duration_25_75", "tail_50_75"],
        "peak_phase": ["peak_smooth3"],
    }
    rows = []
    for phase, markers in phase_map.items():
        comps = [_dist_compare(df, m, settings.near_equal_days) for m in markers if m in df.columns]
        if not comps:
            continue
        median_delta = _median([c["median_delta_Jw_minus_H"] for c in comps])
        overlap = _mean([c["overlap_score"] for c in comps])
        hfrac = _mean([c["fraction_H_features_earlier"] for c in comps])
        jwfrac = _mean([c["fraction_Jw_features_earlier"] for c in comps])
        near = _mean([c["fraction_near_equal"] for c in comps])
        wh = _mean([c["weighted_fraction_H_earlier"] for c in comps])
        wj = _mean([c["weighted_fraction_Jw_earlier"] for c in comps])
        wnear = _mean([c["weighted_fraction_near_equal"] for c in comps])
        if np.isfinite(near) and near >= 0.60 and abs(median_delta) <= settings.near_equal_days:
            label = "strong_overlap"
        elif np.isfinite(hfrac) and hfrac >= 0.60:
            label = "H_shifted_earlier"
        elif np.isfinite(jwfrac) and jwfrac >= 0.60:
            label = "Jw_shifted_earlier"
        elif np.isfinite(overlap) and overlap >= 0.40:
            label = "moderate_overlap"
        else:
            label = "mixed_distribution"
        rows.append({
            "phase": phase,
            "marker_set": ";".join(markers),
            "H_distribution_median": _median([c["H_median"] for c in comps]),
            "Jw_distribution_median": _median([c["Jw_median"] for c in comps]),
            "median_delta_Jw_minus_H": median_delta,
            "overlap_score": overlap,
            "near_equal_fraction": near,
            "H_earlier_fraction": hfrac,
            "Jw_earlier_fraction": jwfrac,
            "weighted_overlap_score": np.nan,  # weighted distribution overlap is not well-defined without resampling; fractions retained below.
            "weighted_H_earlier_fraction": wh,
            "weighted_Jw_earlier_fraction": wj,
            "weighted_near_equal_fraction": wnear,
            "separation_label": label,
            "interpretation": f"H/Jw {phase}: {label}; phase summary from feature timing distributions."
        })
    return pd.DataFrame(rows)


def _weighted_unweighted_contrast(df: pd.DataFrame, settings: V7RSettings) -> pd.DataFrame:
    rows = []
    for marker in ALL_MARKERS:
        if marker not in df.columns:
            continue
        d = _dist_compare(df, marker, settings.near_equal_days)
        unweighted_delta = d["median_delta_Jw_minus_H"]
        weighted_delta = d["weighted_Jw_median"] - d["weighted_H_median"] if np.isfinite(d["weighted_Jw_median"]) and np.isfinite(d["weighted_H_median"]) else np.nan
        if not np.isfinite(weighted_delta) or not np.isfinite(unweighted_delta):
            label = "weight_unavailable"
        elif abs(weighted_delta - unweighted_delta) <= settings.near_equal_days:
            label = "weight_robust"
        elif abs(weighted_delta) > abs(unweighted_delta):
            label = "weighted_stronger_than_unweighted"
        elif abs(unweighted_delta) > abs(weighted_delta):
            label = "unweighted_only_signal"
        else:
            label = "weight_sensitive_unstable"
        rows.append({
            "marker": marker,
            "unweighted_delta_Jw_minus_H": unweighted_delta,
            "weighted_delta_Jw_minus_H": weighted_delta,
            "unweighted_H_earlier_fraction": d["fraction_H_features_earlier"],
            "weighted_H_earlier_fraction": d["weighted_fraction_H_earlier"],
            "unweighted_Jw_earlier_fraction": d["fraction_Jw_features_earlier"],
            "weighted_Jw_earlier_fraction": d["weighted_fraction_Jw_earlier"],
            "unweighted_overlap_score": d["overlap_score"],
            "weighted_near_equal_fraction": d["weighted_fraction_near_equal"],
            "weight_sensitivity_label": label,
            "interpretation": f"H/Jw {marker}: {label}; weighted values use |dF| contribution.",
        })
    return pd.DataFrame(rows)


def _coordinate_concentration(prov: pd.DataFrame, early: pd.DataFrame, catchup: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Helper: support features based on label tables.
    support_specs = []
    for _, r in early.iterrows():
        marker = r["marker"]
        if str(r.get("support_pattern_label", "")).startswith(("broad", "core", "support")) or r.get("fraction_H_features_earlier", 0) > 0:
            support_specs.append(("H", "H_early_support", marker, "H", marker))
    for _, r in catchup.iterrows():
        marker = r["marker"]
        if r.get("fraction_Jw_features_earlier_or_more_compact", 0) > 0:
            support_specs.append(("Jw", "Jw_catchup_support", marker, "Jw", marker))
    for field, support_type, phase_marker, data_field, marker in support_specs:
        sub = _field_marker_values(df, data_field, marker)
        if sub.empty or marker not in sub.columns:
            continue
        # Use earlier-than opposite median for H and catchup logic for Jw.
        compare = _dist_compare(df, marker, 1.0)
        if field == "H":
            threshold = compare["Jw_median"]
            mask = sub[marker].astype(float) < threshold if np.isfinite(threshold) else pd.Series(False, index=sub.index)
        else:
            threshold = compare["H_median"]
            if marker in SPAN_MARKERS:
                mask = sub[marker].astype(float) < threshold if np.isfinite(threshold) else pd.Series(False, index=sub.index)
            else:
                mask = sub[marker].astype(float) <= threshold + 1.0 if np.isfinite(threshold) else pd.Series(False, index=sub.index)
        coords = _safe_arr(sub.loc[mask, "feature_coordinate"].to_numpy(dtype=float))
        psub = prov[(prov["field"] == field)]
        allowed = bool(psub["can_map_to_physical_region"].all()) if not psub.empty else False
        if coords.size:
            iqr = _q(coords, 0.75) - _q(coords, 0.25)
            label = "coordinate_cluster_candidate" if coords.size >= 2 and np.isfinite(iqr) and iqr <= 2 else "coordinate_spread_or_insufficient"
        else:
            iqr = np.nan
            label = "no_support_features"
        if not allowed:
            interp = "Feature metadata insufficient for physical-region interpretation; coordinate concentration is feature-index diagnostic only."
        else:
            interp = "Coordinate concentration may support physical-region follow-up, pending map/profile verification."
        rows.append({
            "field": field,
            "support_type": support_type,
            "marker_phase": phase_marker,
            "support_feature_ids": ";".join(map(str, sub.loc[mask, "feature_id"].astype(int).tolist())) if mask.any() else "none",
            "support_feature_coordinates": ";".join(map(str, sub.loc[mask, "feature_coordinate"].tolist())) if mask.any() else "none",
            "coordinate_min": float(np.nanmin(coords)) if coords.size else np.nan,
            "coordinate_max": float(np.nanmax(coords)) if coords.size else np.nan,
            "coordinate_median": _median(coords),
            "coordinate_iqr": iqr,
            "coordinate_concentration_label": label,
            "physical_region_interpretation_allowed": allowed,
            "interpretation": interp,
        })
    return pd.DataFrame(rows)


def _special_relation_card(departure: pd.DataFrame, early: pd.DataFrame, catchup: pd.DataFrame, overlap: pd.DataFrame, weight: pd.DataFrame, coord: pd.DataFrame) -> pd.DataFrame:
    def count_label(df: pd.DataFrame, col: str, contains: str) -> int:
        return int(df[col].fillna("").astype(str).str.contains(contains, regex=False).sum()) if col in df else 0

    same_phase = count_label(departure, "same_phase_candidate_label", "same_phase")
    h_broad = count_label(early, "support_pattern_label", "broad") + count_label(early, "support_pattern_label", "core")
    jw_catch = count_label(catchup, "catchup_support_label", "broad") + count_label(catchup, "catchup_support_label", "core")
    weight_robust = count_label(weight, "weight_sensitivity_label", "weight_robust")
    region_allowed = bool(coord["physical_region_interpretation_allowed"].any()) if not coord.empty and "physical_region_interpretation_allowed" in coord else False

    if same_phase >= 2 and h_broad >= 4 and jw_catch >= 1:
        overall = "same_departure_with_H_frontloaded_and_Jw_catchup_support"
    elif same_phase >= 2 and h_broad >= 4:
        overall = "same_departure_with_H_frontloaded_support"
    elif h_broad > 0:
        overall = "H_frontloaded_limited_feature_support"
    else:
        overall = "feature_level_unresolved"

    rows = [
        {
            "relation_dimension": "departure_same_phase",
            "evidence_summary": f"{same_phase} departure markers show same/near-phase candidate labels.",
            "support_level": "supported" if same_phase >= 2 else "weak_or_unresolved",
            "support_feature_basis": "departure90/departure95/t10 feature distributions",
            "spatial_or_component_basis": "available" if region_allowed else "feature-index only; no physical-region claim",
            "interpretation": "Use as near-same-phase candidate only, not synchrony.",
            "do_not_overinterpret": "Do not infer synchrony without equivalence test.",
        },
        {
            "relation_dimension": "H_early_progress_support",
            "evidence_summary": f"{h_broad} early-progress markers show broad/core H feature support.",
            "support_level": "supported" if h_broad >= 4 else ("limited" if h_broad > 0 else "not_supported"),
            "support_feature_basis": "t10-t50 H feature timing compared to Jw median/distribution",
            "spatial_or_component_basis": "available" if region_allowed else "feature-index only; no physical-region claim",
            "interpretation": "H front-loaded early-progress is supported only if broad/core support labels dominate.",
            "do_not_overinterpret": "Do not write H globally leads Jw.",
        },
        {
            "relation_dimension": "Jw_catchup_support",
            "evidence_summary": f"{jw_catch} catch-up/finish markers show broad/core Jw feature support.",
            "support_level": "supported" if jw_catch >= 2 else ("limited" if jw_catch > 0 else "not_supported"),
            "support_feature_basis": "t50/t75/peak/duration/tail Jw features compared to H distribution",
            "spatial_or_component_basis": "available" if region_allowed else "feature-index only; no physical-region claim",
            "interpretation": "Jw catch-up is a separate finish/compactness question, not the inverse of H early-progress.",
            "do_not_overinterpret": "Do not infer Jw physically catches up without spatial/component follow-up.",
        },
        {
            "relation_dimension": "weighted_unweighted_robustness",
            "evidence_summary": f"{weight_robust} markers are robust to |dF|-contribution weighting.",
            "support_level": "supported" if weight_robust >= 5 else "sensitive_or_mixed",
            "support_feature_basis": "weighted vs unweighted feature timing summaries",
            "spatial_or_component_basis": "not applicable",
            "interpretation": "If weight-sensitive, evidence may depend on low/high contribution features.",
            "do_not_overinterpret": "Separate weighted and unweighted statements.",
        },
        {
            "relation_dimension": "overall_H_Jw_relation",
            "evidence_summary": overall,
            "support_level": "diagnostic_candidate",
            "support_feature_basis": "all V7-r H/Jw feature exploration tables",
            "spatial_or_component_basis": "available" if region_allowed else "feature-index only; no physical-region claim",
            "interpretation": overall,
            "do_not_overinterpret": "This is an exploratory diagnostic card based on V7-q outputs; it is not a clean trunk result.",
        },
    ]
    return pd.DataFrame(rows)


def _make_figures(settings: V7RSettings, early: pd.DataFrame, catchup: pd.DataFrame, overlap: pd.DataFrame, weight: pd.DataFrame) -> None:
    _ensure_dir(settings.figure_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        _write_text(f"matplotlib unavailable: {exc}\n", settings.log_dir / "figure_warning_v7_r.log")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1. H/Jw feature timing by marker.
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(early))
        ax.plot(x, early["H_median"], marker="o", label="H median")
        ax.plot(x, early["Jw_median"], marker="o", label="Jw median")
        ax.set_xticks(x)
        ax.set_xticklabels(early["marker"], rotation=45, ha="right")
        ax.set_ylabel("Feature timing day")
        ax.set_title("W45 H/Jw early-progress feature timing")
        ax.legend()
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_H_Jw_feature_timing_by_marker_v7_r.png", dpi=160)
        plt.close(fig)

        # 2. overlap by phase.
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(overlap))
        ax.bar(x, overlap["overlap_score"].fillna(0))
        ax.set_xticks(x)
        ax.set_xticklabels(overlap["phase"], rotation=30, ha="right")
        ax.set_ylabel("IQR overlap score")
        ax.set_title("H/Jw feature distribution overlap by phase")
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_H_Jw_distribution_overlap_by_phase_v7_r.png", dpi=160)
        plt.close(fig)

        # 3. H early support fractions.
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(early))
        ax.plot(x, early["fraction_H_features_earlier"], marker="o", label="unweighted")
        ax.plot(x, early["weighted_fraction_H_earlier"], marker="o", label="weighted")
        ax.set_xticks(x)
        ax.set_xticklabels(early["marker"], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction H features earlier")
        ax.set_title("H early support features vs Jw")
        ax.legend()
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_H_early_support_features_v7_r.png", dpi=160)
        plt.close(fig)

        # 4. Jw catch-up support fractions.
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(catchup))
        ax.plot(x, catchup["fraction_Jw_features_earlier_or_more_compact"], marker="o", label="unweighted")
        ax.plot(x, catchup["weighted_fraction_Jw_features_earlier_or_more_compact"], marker="o", label="weighted")
        ax.set_xticks(x)
        ax.set_xticklabels(catchup["marker"], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction Jw support features")
        ax.set_title("Jw catch-up/finish support vs H")
        ax.legend()
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_Jw_catchup_support_features_v7_r.png", dpi=160)
        plt.close(fig)

        # 5. weighted vs unweighted.
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(weight))
        ax.plot(x, weight["unweighted_delta_Jw_minus_H"], marker="o", label="unweighted delta")
        ax.plot(x, weight["weighted_delta_Jw_minus_H"], marker="o", label="weighted delta")
        ax.axhline(0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(weight["marker"], rotation=45, ha="right")
        ax.set_ylabel("Jw - H timing delta")
        ax.set_title("H/Jw weighted vs unweighted feature timing contrast")
        ax.legend()
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "w45_H_Jw_weighted_vs_unweighted_v7_r.png", dpi=160)
        plt.close(fig)


def _summary_markdown(audit: dict[str, Any], provenance: pd.DataFrame, departure: pd.DataFrame, early: pd.DataFrame, catchup: pd.DataFrame, overlap: pd.DataFrame, weight: pd.DataFrame, card: pd.DataFrame) -> str:
    region_allowed = bool(provenance["can_map_to_physical_region"].any()) if not provenance.empty else False
    overall = card.loc[card["relation_dimension"] == "overall_H_Jw_relation", "interpretation"].iloc[0] if not card.empty else "not_available"
    lines = [
        f"# W45 H/Jw feature relation exploration V7-r",
        "",
        "## 1. Purpose",
        "This diagnostic branch reduces the scope to H and Jw and explores feature-level evidence for their W45 relation.",
        "It reads V7-q feature-level outputs and is not a clean trunk rebuild.",
        "",
        "## 2. Input boundary",
        f"- Input V7-q dir: `{audit.get('input_q_dir')}`",
        "- Uses V7-m/n/o/p outputs as input: false",
        "- Uses V7-q outputs as diagnostic input: true",
        "",
        "## 3. Feature provenance",
        f"- Physical region interpretation allowed: {region_allowed}",
        "- If false, all feature-coordinate patterns are feature-index diagnostics only, not regional conclusions.",
        "",
        "## 4. Departure same-phase evidence",
    ]
    if not departure.empty:
        for _, r in departure.iterrows():
            lines.append(f"- {r['marker']}: delta Jw-H={r['median_delta_Jw_minus_H']:.3g}, near_equal={r['fraction_near_equal']:.3g}, label={r['same_phase_candidate_label']}")
    lines += ["", "## 5. H early-progress support"]
    if not early.empty:
        for _, r in early.iterrows():
            lines.append(f"- {r['marker']}: H earlier fraction={r['fraction_H_features_earlier']:.3g}, weighted={r['weighted_fraction_H_earlier']:.3g}, label={r['support_pattern_label']}")
    lines += ["", "## 6. Jw catch-up / finish support"]
    if not catchup.empty:
        for _, r in catchup.iterrows():
            lines.append(f"- {r['marker']}: Jw support fraction={r['fraction_Jw_features_earlier_or_more_compact']:.3g}, weighted={r['weighted_fraction_Jw_features_earlier_or_more_compact']:.3g}, label={r['catchup_support_label']}")
    lines += ["", "## 7. Distribution overlap by phase"]
    if not overlap.empty:
        for _, r in overlap.iterrows():
            lines.append(f"- {r['phase']}: delta={r['median_delta_Jw_minus_H']:.3g}, overlap={r['overlap_score']:.3g}, label={r['separation_label']}")
    lines += [
        "",
        "## 8. Overall diagnostic card",
        f"- Overall relation candidate: {overall}",
        "",
        "## 9. Prohibited interpretations",
        "- Do not write H globally leads Jw from this branch.",
        "- Do not write H/Jw are synchronous; near-same-phase is only a candidate without an equivalence test.",
        "- Do not infer physical regions from feature IDs unless provenance allows coordinate interpretation.",
        "- Do not delete weak/noisy features; they are retained but down-weighted.",
    ]
    return "\n".join(lines) + "\n"


def run_w45_H_Jw_feature_relation_exploration_v7_r(v7_root: Optional[Path] = None) -> None:
    settings = _resolve_settings(v7_root)
    for d in [settings.output_dir, settings.log_dir, settings.figure_dir]:
        _ensure_dir(d)
    progress_log = settings.log_dir / "run_progress_v7_r.log"
    progress_log.write_text(f"[{_now_iso()}] start {OUTPUT_TAG}\n", encoding="utf-8")

    data = _load_inputs(settings)
    audit = _input_audit(settings, data)
    if not audit["H_Jw_available"]:
        raise RuntimeError("V7-r requires both H and Jw feature-level rows in V7-q outputs.")
    _write_json(audit, settings.output_dir / "input_audit_v7_r.json")

    progress_log.write_text(progress_log.read_text(encoding="utf-8") + f"[{_now_iso()}] provenance and eligibility\n", encoding="utf-8")
    provenance = _provenance_audit(data)
    eligibility = _eligibility_audit(data)
    feature_df = _merge_eligible(data, eligibility)
    feature_df = feature_df[feature_df["field"].isin(FIELDS)].copy()

    progress_log.write_text(progress_log.read_text(encoding="utf-8") + f"[{_now_iso()}] H/Jw relation diagnostics\n", encoding="utf-8")
    departure = _departure_same_phase(feature_df, settings)
    early = _H_early_support(feature_df, settings)
    catchup = _Jw_catchup_support(feature_df, settings)
    overlap = _overlap_by_phase(feature_df, settings)
    weight = _weighted_unweighted_contrast(feature_df, settings)
    coord = _coordinate_concentration(provenance, early, catchup, feature_df)
    card = _special_relation_card(departure, early, catchup, overlap, weight, coord)

    progress_log.write_text(progress_log.read_text(encoding="utf-8") + f"[{_now_iso()}] write tables\n", encoding="utf-8")
    _write_csv(provenance, settings.output_dir / "w45_H_Jw_feature_provenance_audit_v7_r.csv")
    _write_csv(eligibility[eligibility["field"].isin(FIELDS)].copy(), settings.output_dir / "w45_H_Jw_feature_eligibility_v7_r.csv")
    _write_csv(departure, settings.output_dir / "w45_H_Jw_departure_same_phase_v7_r.csv")
    _write_csv(early, settings.output_dir / "w45_H_early_progress_support_vs_Jw_v7_r.csv")
    _write_csv(catchup, settings.output_dir / "w45_Jw_catchup_finish_support_vs_H_v7_r.csv")
    _write_csv(overlap, settings.output_dir / "w45_H_Jw_feature_distribution_overlap_by_phase_v7_r.csv")
    _write_csv(weight, settings.output_dir / "w45_H_Jw_weighted_unweighted_contrast_v7_r.csv")
    _write_csv(coord, settings.output_dir / "w45_H_Jw_feature_coordinate_concentration_v7_r.csv")
    _write_csv(card, settings.output_dir / "w45_H_Jw_special_relation_card_v7_r.csv")

    summary = _summary_markdown(audit, provenance, departure, early, catchup, overlap, weight, card)
    _write_text(summary, settings.output_dir / "w45_H_Jw_special_relation_summary_v7_r.md")

    progress_log.write_text(progress_log.read_text(encoding="utf-8") + f"[{_now_iso()}] figures\n", encoding="utf-8")
    _make_figures(settings, early, catchup, overlap, weight)

    run_meta = {
        "version": OUTPUT_TAG,
        "status": "success",
        "created_at": _now_iso(),
        "window_id": settings.window_id,
        "anchor_day": settings.anchor_day,
        "fields": list(FIELDS),
        "diagnostic_branch": True,
        "input_branch": INPUT_Q_TAG,
        "read_v7q_outputs_as_diagnostic_input": True,
        "v7_m_outputs_used_as_input": False,
        "v7_n_outputs_used_as_input": False,
        "v7_o_outputs_used_as_input": False,
        "v7_p_outputs_used_as_input": False,
        "n_provenance_rows": int(len(provenance)),
        "n_eligibility_rows": int(len(eligibility[eligibility["field"].isin(FIELDS)])),
        "n_early_markers": int(len(early)),
        "n_catchup_markers": int(len(catchup)),
        "overall_relation_candidate": card.loc[card["relation_dimension"] == "overall_H_Jw_relation", "interpretation"].iloc[0] if not card.empty else "not_available",
    }
    _write_json(run_meta, settings.output_dir / "run_meta.json")
    progress_log.write_text(progress_log.read_text(encoding="utf-8") + f"[{_now_iso()}] finished success\n", encoding="utf-8")
