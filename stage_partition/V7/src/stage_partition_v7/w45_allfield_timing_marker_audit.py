from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

WINDOW_ID = "W002"
ANCHOR_DAY = 45
OUTPUT_TAG = "w45_allfield_timing_marker_audit_v7_l"
FIELDS = ["P", "V", "H", "Je", "Jw"]
MARKERS = ["onset", "midpoint", "finish", "duration"]
ORDER_MARKERS = ["onset", "midpoint", "finish"]


@dataclass
class W45AllFieldMarkerPaths:
    v7_root: Path
    v7e_output_dir: Path
    v7e1_output_dir: Path
    v7e2_output_dir: Path
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


def _resolve_paths(v7_root: Optional[Path]) -> W45AllFieldMarkerPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    return W45AllFieldMarkerPaths(
        v7_root=v7_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7e1_output_dir=v7_root / "outputs" / "progress_order_significance_audit_v7_e1",
        v7e2_output_dir=v7_root / "outputs" / "progress_order_failure_audit_v7_e2",
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _safe_quantile(values: pd.Series | np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanquantile(arr, q))


def _safe_median(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanmedian(arr))


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _valid_fraction(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.isfinite(arr).sum() / arr.size)


def _join(values: Iterable[object]) -> str:
    vals = []
    for x in values:
        if x is None:
            continue
        s = str(x)
        if not s or s.lower() == "nan":
            continue
        vals.append(s)
    return ";".join(vals) if vals else "none"


def _direction_from_delta(delta: float, a: str, b: str) -> tuple[str, str, str]:
    if not np.isfinite(delta) or delta == 0:
        return "tie_or_zero", "", ""
    if delta > 0:
        return "field_a_before_field_b", a, b
    return "field_b_before_field_a", b, a


def _load_inputs(paths: W45AllFieldMarkerPaths) -> dict[str, pd.DataFrame]:
    return {
        "v7e_obs": _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv"),
        "v7e_boot": _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_samples_v7_e.csv"),
        "v7e_sum": _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_summary_v7_e.csv", required=False),
        "v7e1_pair": _read_csv(paths.v7e1_output_dir / "pairwise_progress_delta_test_v7_e1.csv", required=False),
        "v7e2_fail": _read_csv(paths.v7e2_output_dir / "pairwise_progress_failure_audit_v7_e2.csv", required=False),
    }


def _validate_inputs(tables: dict[str, pd.DataFrame], paths: W45AllFieldMarkerPaths) -> dict:
    _require_columns(
        tables["v7e_obs"],
        ["window_id", "anchor_day", "field", "onset_day", "midpoint_day", "finish_day", "duration"],
        "V7-e observed",
    )
    _require_columns(
        tables["v7e_boot"],
        ["window_id", "anchor_day", "field", "bootstrap_id", "onset_day", "midpoint_day", "finish_day", "duration"],
        "V7-e bootstrap samples",
    )

    obs = tables["v7e_obs"][tables["v7e_obs"]["window_id"].astype(str) == WINDOW_ID]
    boot = tables["v7e_boot"][tables["v7e_boot"]["window_id"].astype(str) == WINDOW_ID]
    obs_fields = set(obs["field"].astype(str))
    boot_fields = set(boot["field"].astype(str))

    audit = {
        "status": "checked",
        "created_at": _now_iso(),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "v7e_output_dir": str(paths.v7e_output_dir),
        "v7e1_output_dir": str(paths.v7e1_output_dir),
        "v7e2_output_dir": str(paths.v7e2_output_dir),
        "v7e_observed_fields_found": sorted(obs_fields),
        "v7e_bootstrap_fields_found": sorted(boot_fields),
        "required_fields": FIELDS,
        "observed_fields_complete": all(f in obs_fields for f in FIELDS),
        "bootstrap_fields_complete": all(f in boot_fields for f in FIELDS),
        "v7e1_pairwise_available": bool(not tables["v7e1_pair"].empty),
        "v7e2_failure_available": bool(not tables["v7e2_fail"].empty),
    }
    if not audit["observed_fields_complete"]:
        raise ValueError(f"V7-e observed W45 rows are missing fields. Found={sorted(obs_fields)} required={FIELDS}")
    if not audit["bootstrap_fields_complete"]:
        raise ValueError(f"V7-e bootstrap W45 rows are missing fields. Found={sorted(boot_fields)} required={FIELDS}")
    if not tables["v7e1_pair"].empty:
        _require_columns(tables["v7e1_pair"], ["window_id", "field_a", "field_b", "final_evidence_label"], "V7-e1 pairwise")
        audit["v7e1_w45_pair_count"] = int((tables["v7e1_pair"]["window_id"].astype(str) == WINDOW_ID).sum())
    if not tables["v7e2_fail"].empty:
        _require_columns(tables["v7e2_fail"], ["window_id", "field_a", "field_b", "primary_failure_type"], "V7-e2 failure")
        audit["v7e2_w45_pair_count"] = int((tables["v7e2_fail"]["window_id"].astype(str) == WINDOW_ID).sum())
    return audit


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
        "iqr": _safe_quantile(values, 0.75) - _safe_quantile(values, 0.25),
        "valid_fraction": _valid_fraction(values),
    }


def _field_quality(obs_row: pd.Series) -> str:
    return str(obs_row.get("progress_quality_label", "unknown"))


def _quality_bad_or_limited(q: str) -> bool:
    q = str(q).lower()
    return ("no_clear" in q) or ("invalid" in q) or ("boundary" in q)


def _choose_marker(row: dict) -> tuple[str, str, str, str, str, str]:
    """Choose timing marker without silently promoting ties to onset.

    Earlier V7-l preferred onset whenever onset tied for the smallest q90 width.
    That made P/V/Je appear to be clean onset-type fields when their onset and
    midpoint widths were actually tied or nearly tied.  This hotfix keeps ties
    explicit.
    """
    widths = {m: row.get(f"{m}_q90_width", np.nan) for m in ORDER_MARKERS}
    finite = {m: w for m, w in widths.items() if np.isfinite(w)}
    quality = str(row.get("progress_quality_label", "unknown"))
    if not finite:
        return "none", "marker_unresolved", "No finite onset/midpoint/finish bootstrap widths.", "unresolved", "none", "none"
    if _quality_bad_or_limited(quality):
        return "none", "marker_unresolved", f"Progress quality is limited ({quality}); no marker recommended.", "unresolved", "none", "none"
    min_w = min(finite.values())
    best = [m for m, w in finite.items() if np.isclose(w, min_w)]
    tie_markers = _join(best)
    if len(best) == 1:
        recommended = best[0]
        label = f"{recommended}_unique_best"
        reason = f"{recommended} is the unique smallest-q90 marker among onset/midpoint/finish: {finite}."
        return recommended, label, reason, "unique_best", tie_markers, recommended

    # Do not force an onset recommendation when onset ties with midpoint/finish.
    recommended = "marker_tie"
    label = "marker_tie"
    reason = f"No unique recommended marker: tied smallest-q90 markers are {tie_markers}; widths={finite}."
    return recommended, label, reason, "tie", tie_markers, "none"


def _shape_label_from_row(obs: pd.Series, rec_marker: str, stability_row: dict) -> tuple[str, str]:
    onset = float(obs.get("onset_day", np.nan))
    mid = float(obs.get("midpoint_day", np.nan))
    fin = float(obs.get("finish_day", np.nan))
    dur = float(obs.get("duration", np.nan))
    q = str(obs.get("progress_quality_label", "unknown"))
    onset_to_mid = mid - onset if np.isfinite(mid) and np.isfinite(onset) else np.nan
    mid_to_finish = fin - mid if np.isfinite(fin) and np.isfinite(mid) else np.nan
    onset_w = stability_row.get("onset_q90_width", np.nan)
    mid_w = stability_row.get("midpoint_q90_width", np.nan)
    fin_w = stability_row.get("finish_q90_width", np.nan)
    dur_w = stability_row.get("duration_q90_width", np.nan)

    if _quality_bad_or_limited(q):
        return "nonmonotonic_or_uncertain_shape", f"Progress quality is {q}; observed shape is not promoted."
    if rec_marker == "marker_tie":
        return "marker_tie_transition", "No unique marker is preferred; do not promote tied onset/midpoint/finish widths to a field-order marker."
    if rec_marker == "onset" and (
        (np.isfinite(fin_w) and np.isfinite(onset_w) and fin_w > onset_w)
        or (np.isfinite(dur_w) and np.isfinite(onset_w) and dur_w > onset_w)
        or (np.isfinite(mid_to_finish) and np.isfinite(onset_to_mid) and mid_to_finish > onset_to_mid)
    ):
        return "early_onset_broad_transition", "Onset is the unique preferred/stablest marker and the completion or post-midpoint phase is broader."
    if rec_marker == "midpoint" and str(q).lower() not in {"nonmonotonic_progress"}:
        return "midpoint_representative_transition", "Midpoint is the unique preferred/stablest marker and progress quality is not flagged as nonmonotonic."
    if rec_marker == "finish":
        return "finish_marker_transition", "Finish is the unique preferred/stablest marker; interpret as completion timing, not onset or midpoint order."
    return "marker_unresolved_transition", "No clean transition-shape label can be assigned without overinterpretation."


def _summarize_marker_stability(obs: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    obs_w = obs[obs["window_id"].astype(str) == WINDOW_ID].copy()
    boot_w = boot[boot["window_id"].astype(str) == WINDOW_ID].copy()
    for field in FIELDS:
        sub = boot_w[boot_w["field"].astype(str) == field]
        obs_sub = obs_w[obs_w["field"].astype(str) == field]
        if obs_sub.empty:
            raise ValueError(f"Missing observed row for field={field} window={WINDOW_ID}")
        obs_row = obs_sub.iloc[0]
        row = {"window_id": WINDOW_ID, "anchor_day": ANCHOR_DAY, "field": field}
        for marker, col in [("onset", "onset_day"), ("midpoint", "midpoint_day"), ("finish", "finish_day"), ("duration", "duration")]:
            stats = _marker_stats(pd.to_numeric(sub[col], errors="coerce"))
            for k, v in stats.items():
                row[f"{marker}_{k}"] = v
        row["progress_quality_label"] = obs_row.get("progress_quality_label", "unknown")
        row["pre_post_separation_label"] = obs_row.get("pre_post_separation_label", "unknown")
        widths = {m: row.get(f"{m}_q90_width", np.nan) for m in MARKERS}
        finite_widths = {m: w for m, w in widths.items() if np.isfinite(w)}
        if finite_widths:
            min_w = min(finite_widths.values())
            max_w = max(finite_widths.values())
            row["most_stable_marker"] = _join([m for m, w in finite_widths.items() if np.isclose(w, min_w)])
            row["least_stable_marker"] = _join([m for m, w in finite_widths.items() if np.isclose(w, max_w)])
        else:
            row["most_stable_marker"] = "none"
            row["least_stable_marker"] = "none"
        rec, label, reason, selection_type, tie_markers, unique_best = _choose_marker(row)
        row["recommended_marker"] = rec
        row["marker_reliability_label"] = label
        row["marker_selection_type"] = selection_type
        row["tie_markers"] = tie_markers
        row["unique_best_marker"] = unique_best
        row["marker_interpretation"] = reason
        shape, shape_reason = _shape_label_from_row(obs_row, rec, row)
        row["transition_shape_label"] = shape
        row["transition_shape_interpretation"] = shape_reason
        rows.append(row)
    return pd.DataFrame(rows)


def _observed_shape(obs: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    obs_w = obs[obs["window_id"].astype(str) == WINDOW_ID].copy()
    for field in FIELDS:
        o = obs_w[obs_w["field"].astype(str) == field].iloc[0]
        s = stability[stability["field"].astype(str) == field].iloc[0].to_dict()
        onset = float(o.get("onset_day", np.nan))
        mid = float(o.get("midpoint_day", np.nan))
        fin = float(o.get("finish_day", np.nan))
        onset_to_mid = mid - onset if np.isfinite(onset) and np.isfinite(mid) else np.nan
        mid_to_finish = fin - mid if np.isfinite(mid) and np.isfinite(fin) else np.nan
        shape, interp = _shape_label_from_row(o, str(s.get("recommended_marker", "none")), s)
        rows.append(
            {
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "field": field,
                "observed_onset": onset,
                "observed_midpoint": mid,
                "observed_finish": fin,
                "observed_duration": float(o.get("duration", np.nan)),
                "observed_onset_to_midpoint": onset_to_mid,
                "observed_midpoint_to_finish": mid_to_finish,
                "pre_post_separation_label": o.get("pre_post_separation_label", "unknown"),
                "progress_quality_label": o.get("progress_quality_label", "unknown"),
                "progress_monotonicity_corr": o.get("progress_monotonicity_corr", np.nan),
                "n_crossings_025": o.get("n_crossings_025", np.nan),
                "n_crossings_050": o.get("n_crossings_050", np.nan),
                "n_crossings_075": o.get("n_crossings_075", np.nan),
                "observed_shape_label": shape,
                "observed_shape_interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


def _marker_comparability(stability: pd.DataFrame) -> pd.DataFrame:
    rec = dict(zip(stability["field"].astype(str), stability["recommended_marker"].astype(str)))
    ties = dict(zip(stability["field"].astype(str), stability.get("tie_markers", pd.Series(["none"] * len(stability))).astype(str)))
    rows = []
    for a, b in combinations(FIELDS, 2):
        ma, mb = rec.get(a, "none"), rec.get(b, "none")
        same = ma == mb and ma in ORDER_MARKERS
        if same:
            allowed = True
            ctype = f"{ma}_order_comparison"
            reason = f"Both fields have unique recommended marker {ma}; same-marker comparison is allowed as {ma} order."
        elif ma == "marker_tie" or mb == "marker_tie":
            allowed = False
            ctype = "not_allowed_marker_tie"
            reason = f"At least one field has tied best markers ({a}:{ties.get(a, 'none')}, {b}:{ties.get(b, 'none')}); do not force an onset/midpoint comparison."
        elif ma == "none" or mb == "none":
            allowed = False
            ctype = "not_allowed_marker_unresolved"
            reason = "At least one field has no reliable recommended timing marker."
        else:
            allowed = False
            ctype = "not_allowed_mixed_marker"
            reason = f"Mixed unique recommended markers ({a}:{ma}, {b}:{mb}); do not interpret as same-stage field order."
        rows.append(
            {
                "field_a": a,
                "field_b": b,
                "field_a_recommended_marker": ma,
                "field_b_recommended_marker": mb,
                "field_a_tie_markers": ties.get(a, "none"),
                "field_b_tie_markers": ties.get(b, "none"),
                "same_marker_type": same,
                "marker_comparison_allowed": allowed,
                "comparison_type": ctype,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _delta_stats(delta: np.ndarray) -> dict:
    return {
        "median": _safe_median(delta),
        "mean": _safe_mean(delta),
        "q05": _safe_quantile(delta, 0.05),
        "q95": _safe_quantile(delta, 0.95),
        "q025": _safe_quantile(delta, 0.025),
        "q975": _safe_quantile(delta, 0.975),
        "prob_gt_0": _safe_mean(np.asarray(delta, dtype=float) > 0),
        "prob_lt_0": _safe_mean(np.asarray(delta, dtype=float) < 0),
        "prob_eq_0": _safe_mean(np.asarray(delta, dtype=float) == 0),
        "n_valid": int(np.isfinite(np.asarray(delta, dtype=float)).sum()),
    }


def _pairwise_marker_delta_test(boot: pd.DataFrame, marker_col: str, out_prefix: str) -> pd.DataFrame:
    boot_w = boot[boot["window_id"].astype(str) == WINDOW_ID].copy()
    pivot = boot_w.pivot_table(index="bootstrap_id", columns="field", values=marker_col, aggfunc="first")
    rows = []
    for a, b in combinations(FIELDS, 2):
        if a not in pivot.columns or b not in pivot.columns:
            continue
        delta = pd.to_numeric(pivot[b], errors="coerce") - pd.to_numeric(pivot[a], errors="coerce")
        st = _delta_stats(delta.to_numpy())
        direction, early, late = _direction_from_delta(st["median"], a, b)
        pass90 = bool((np.isfinite(st["q05"]) and st["q05"] > 0) or (np.isfinite(st["q95"]) and st["q95"] < 0))
        pass95 = bool((np.isfinite(st["q025"]) and st["q025"] > 0) or (np.isfinite(st["q975"]) and st["q975"] < 0))
        if pass95:
            label = f"confirmed_{out_prefix}_order_95"
        elif pass90:
            label = f"confirmed_{out_prefix}_order_90"
        elif direction != "tie_or_zero":
            label = f"{out_prefix}_direction_tendency"
        else:
            label = f"{out_prefix}_not_distinguishable"
        rows.append(
            {
                "field_a": a,
                "field_b": b,
                f"median_delta_{out_prefix}_b_minus_a": st["median"],
                f"mean_delta_{out_prefix}_b_minus_a": st["mean"],
                f"q05_delta_{out_prefix}": st["q05"],
                f"q95_delta_{out_prefix}": st["q95"],
                f"q025_delta_{out_prefix}": st["q025"],
                f"q975_delta_{out_prefix}": st["q975"],
                f"prob_delta_{out_prefix}_gt_0": st["prob_gt_0"],
                f"prob_delta_{out_prefix}_lt_0": st["prob_lt_0"],
                f"prob_delta_{out_prefix}_eq_0": st["prob_eq_0"],
                "n_bootstrap_valid": st["n_valid"],
                "pass_90": pass90,
                "pass_95": pass95,
                "early_field_candidate": early,
                "late_field_candidate": late,
                f"{out_prefix}_order_label": label,
                f"{out_prefix}_order_interpretation": f"This is {out_prefix} timing only; do not interpret as full transition order or causality.",
            }
        )
    return pd.DataFrame(rows)


def _duration_finish_tail_audit(boot: pd.DataFrame) -> pd.DataFrame:
    boot_w = boot[boot["window_id"].astype(str) == WINDOW_ID].copy()
    boot_w = boot_w.copy()
    boot_w["post_midpoint_tail"] = pd.to_numeric(boot_w["finish_day"], errors="coerce") - pd.to_numeric(boot_w["midpoint_day"], errors="coerce")
    piv_duration = boot_w.pivot_table(index="bootstrap_id", columns="field", values="duration", aggfunc="first")
    piv_finish = boot_w.pivot_table(index="bootstrap_id", columns="field", values="finish_day", aggfunc="first")
    piv_tail = boot_w.pivot_table(index="bootstrap_id", columns="field", values="post_midpoint_tail", aggfunc="first")
    rows = []
    for a, b in combinations(FIELDS, 2):
        if a not in piv_duration.columns or b not in piv_duration.columns:
            continue
        d_dur = pd.to_numeric(piv_duration[b], errors="coerce") - pd.to_numeric(piv_duration[a], errors="coerce")
        d_fin = pd.to_numeric(piv_finish[b], errors="coerce") - pd.to_numeric(piv_finish[a], errors="coerce")
        d_tail = pd.to_numeric(piv_tail[b], errors="coerce") - pd.to_numeric(piv_tail[a], errors="coerce")
        sd = _delta_stats(d_dur.to_numpy())
        sf = _delta_stats(d_fin.to_numpy())
        st = _delta_stats(d_tail.to_numpy())
        pass90_dur = bool((np.isfinite(sd["q05"]) and sd["q05"] > 0) or (np.isfinite(sd["q95"]) and sd["q95"] < 0))
        pass90_fin = bool((np.isfinite(sf["q05"]) and sf["q05"] > 0) or (np.isfinite(sf["q95"]) and sf["q95"] < 0))
        pass90_tail = bool((np.isfinite(st["q05"]) and st["q05"] > 0) or (np.isfinite(st["q95"]) and st["q95"] < 0))
        involves_h = a == "H" or b == "H"
        h_broad_bits = []
        h_minus_other_dur = h_minus_other_dur_q05 = h_minus_other_dur_q95 = np.nan
        h_minus_other_tail = h_minus_other_tail_q05 = h_minus_other_tail_q95 = np.nan
        h_minus_other_finish = h_minus_other_finish_q05 = h_minus_other_finish_q95 = np.nan
        h_duration_longer_pass90 = False
        h_tail_longer_pass90 = False
        h_finish_later_pass90 = False
        if involves_h:
            # Convert pair deltas (B-A) to H-other for clearer broadness interpretation.
            if a == "H":
                h_minus_other_dur = -sd["median"]
                h_minus_other_dur_q05 = -sd["q95"]
                h_minus_other_dur_q95 = -sd["q05"]
                h_minus_other_tail = -st["median"]
                h_minus_other_tail_q05 = -st["q95"]
                h_minus_other_tail_q95 = -st["q05"]
                h_minus_other_finish = -sf["median"]
                h_minus_other_finish_q05 = -sf["q95"]
                h_minus_other_finish_q95 = -sf["q05"]
            else:
                h_minus_other_dur = sd["median"]
                h_minus_other_dur_q05 = sd["q05"]
                h_minus_other_dur_q95 = sd["q95"]
                h_minus_other_tail = st["median"]
                h_minus_other_tail_q05 = st["q05"]
                h_minus_other_tail_q95 = st["q95"]
                h_minus_other_finish = sf["median"]
                h_minus_other_finish_q05 = sf["q05"]
                h_minus_other_finish_q95 = sf["q95"]
            h_duration_longer_pass90 = bool(np.isfinite(h_minus_other_dur_q05) and h_minus_other_dur_q05 > 0)
            h_tail_longer_pass90 = bool(np.isfinite(h_minus_other_tail_q05) and h_minus_other_tail_q05 > 0)
            h_finish_later_pass90 = bool(np.isfinite(h_minus_other_finish_q05) and h_minus_other_finish_q05 > 0)
            if np.isfinite(h_minus_other_dur) and h_minus_other_dur > 0:
                h_broad_bits.append("H_duration_longer_median")
            if np.isfinite(h_minus_other_tail) and h_minus_other_tail > 0:
                h_broad_bits.append("H_post_midpoint_tail_longer_median")
            if np.isfinite(h_minus_other_finish) and h_minus_other_finish > 0:
                h_broad_bits.append("H_finish_later_median")
            if h_duration_longer_pass90:
                h_broad_bits.append("H_duration_longer_pass90")
            if h_tail_longer_pass90:
                h_broad_bits.append("H_tail_longer_pass90")
            if h_finish_later_pass90:
                h_broad_bits.append("H_finish_later_pass90")
        broad_label = _join(h_broad_bits) if h_broad_bits else "no_H_specific_broad_label"
        rows.append(
            {
                "field_a": a,
                "field_b": b,
                "involves_H": involves_h,
                "median_delta_duration_b_minus_a": sd["median"],
                "q05_delta_duration": sd["q05"],
                "q95_delta_duration": sd["q95"],
                "pass90_duration_difference": pass90_dur,
                "median_delta_finish_b_minus_a": sf["median"],
                "q05_delta_finish": sf["q05"],
                "q95_delta_finish": sf["q95"],
                "pass90_finish_difference": pass90_fin,
                "median_delta_midpoint_to_finish_b_minus_a": st["median"],
                "q05_delta_midpoint_to_finish": st["q05"],
                "q95_delta_midpoint_to_finish": st["q95"],
                "pass90_tail_difference": pass90_tail,
                "H_minus_other_duration_median": h_minus_other_dur,
                "H_minus_other_duration_q05": h_minus_other_dur_q05,
                "H_minus_other_duration_q95": h_minus_other_dur_q95,
                "H_duration_longer_pass90": h_duration_longer_pass90,
                "H_minus_other_finish_median": h_minus_other_finish,
                "H_minus_other_finish_q05": h_minus_other_finish_q05,
                "H_minus_other_finish_q95": h_minus_other_finish_q95,
                "H_finish_later_pass90": h_finish_later_pass90,
                "H_minus_other_tail_median": h_minus_other_tail,
                "H_minus_other_tail_q05": h_minus_other_tail_q05,
                "H_minus_other_tail_q95": h_minus_other_tail_q95,
                "H_tail_longer_pass90": h_tail_longer_pass90,
                "broad_transition_label": broad_label,
                "finish_tail_label": "finish_tail_difference_pass90" if pass90_tail else "finish_tail_difference_not_confirmed",
                "interpretation": "Duration/finish-tail shape comparison only; not a field-order or causal test. H-specific pass90 columns require H-other 90% interval to be positive.",
            }
        )
    return pd.DataFrame(rows)


def _collect_edges_for_field(df: pd.DataFrame, field: str, label_col: str, allowed_labels: list[str]) -> str:
    if df.empty or label_col not in df.columns:
        return "none"
    rows = []
    for _, r in df.iterrows():
        if r.get(label_col) not in allowed_labels:
            continue
        a, b = str(r.get("field_a", "")), str(r.get("field_b", ""))
        if field not in {a, b}:
            continue
        early = str(r.get("early_field_candidate", ""))
        late = str(r.get("late_field_candidate", ""))
        if early and late and early != "nan" and late != "nan":
            rows.append(f"{early}<{late}")
        else:
            rows.append(f"{a}-{b}")
    return _join(rows)


def _build_role_reinterpretation(
    stability: pd.DataFrame,
    onset_order: pd.DataFrame,
    duration_tail: pd.DataFrame,
    e1_pair: pd.DataFrame,
    e2_fail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    e1w = e1_pair[e1_pair.get("window_id", pd.Series(dtype=str)).astype(str) == WINDOW_ID].copy() if not e1_pair.empty else pd.DataFrame()
    e2w = e2_fail[e2_fail.get("window_id", pd.Series(dtype=str)).astype(str) == WINDOW_ID].copy() if not e2_fail.empty else pd.DataFrame()
    for _, s in stability.iterrows():
        field = str(s["field"])
        rec = str(s["recommended_marker"])
        shape = str(s["transition_shape_label"])
        midpoint_edges = _collect_edges_for_field(
            e1w,
            field,
            "final_evidence_label",
            ["confirmed_directional_order", "supported_directional_tendency"],
        )
        onset_edges_90 = _collect_edges_for_field(
            onset_order,
            field,
            "onset_order_label",
            ["confirmed_onset_order_95", "confirmed_onset_order_90"],
        )
        onset_edges_tendency = _collect_edges_for_field(
            onset_order,
            field,
            "onset_order_label",
            ["onset_direction_tendency"],
        )
        failure_types = []
        if not e2w.empty:
            for _, r in e2w.iterrows():
                if field in {str(r.get("field_a", "")), str(r.get("field_b", ""))}:
                    failure_types.append(str(r.get("primary_failure_type", "unknown")))
        duration_role = "none"
        observed_broad_candidate = bool("broad" in shape or "early_onset" in shape)
        statistically_confirmed_broad = False
        if field == "H":
            hrows = duration_tail[duration_tail["involves_H"] == True]
            labels = [x for x in hrows["broad_transition_label"].astype(str).tolist() if x != "no_H_specific_broad_label"]
            duration_role = _join(labels)
            if not hrows.empty:
                statistically_confirmed_broad = bool(
                    hrows.get("H_duration_longer_pass90", pd.Series(dtype=bool)).fillna(False).any()
                    or hrows.get("H_tail_longer_pass90", pd.Series(dtype=bool)).fillna(False).any()
                    or hrows.get("H_finish_later_pass90", pd.Series(dtype=bool)).fillna(False).any()
                )
        usable_midpoint = rec == "midpoint" and "midpoint_representative" in shape
        usable_onset = rec == "onset"
        usable_shape = observed_broad_candidate or rec in {"onset", "finish"}
        if field == "H" and "early_onset_broad" in shape:
            final = "early_onset_broad_transition_candidate"
        elif rec == "marker_tie":
            final = "marker_tie_unresolved"
        elif rec == "onset":
            final = "onset_marker_candidate"
        elif rec == "midpoint":
            final = "midpoint_marker_candidate"
        elif rec == "finish":
            final = "finish_marker_candidate"
        else:
            final = "marker_unresolved"
        rows.append(
            {
                "field": field,
                "previous_role_from_v7e": midpoint_edges,
                "recommended_marker": rec,
                "marker_selection_type": str(s.get("marker_selection_type", "unknown")),
                "tie_markers": str(s.get("tie_markers", "none")),
                "transition_shape_label": shape,
                "observed_broad_transition_candidate": observed_broad_candidate,
                "statistically_confirmed_broad_transition": statistically_confirmed_broad,
                "onset_order_role_confirmed90_or95": onset_edges_90,
                "onset_order_role_tendency": onset_edges_tendency,
                "midpoint_order_role_from_v7e1": midpoint_edges,
                "duration_shape_role": duration_role,
                "dominant_failure_types_from_v7e2": _join(sorted(set(failure_types))),
                "usable_for_midpoint_order": bool(usable_midpoint),
                "usable_for_onset_order": bool(usable_onset),
                "usable_for_shape_interpretation": bool(usable_shape),
                "final_role_label": final,
                "final_role_interpretation": "Role is based on marker suitability first; do not hide unresolved fields or promote tendencies to confirmed order.",
                "do_not_overinterpret": "onset_order_not_full_transition_order; broad_transition_not_causality; not_distinguishable_not_synchrony",
            }
        )
    return pd.DataFrame(rows)


def _build_window_summary(stability: pd.DataFrame, onset_order: pd.DataFrame, duration_tail: pd.DataFrame, roles: pd.DataFrame, e1_pair: pd.DataFrame) -> pd.DataFrame:
    rec_counts = stability["recommended_marker"].value_counts().to_dict()
    onset_pass90 = int(onset_order["pass_90"].sum()) if "pass_90" in onset_order.columns else 0
    onset_pass95 = int(onset_order["pass_95"].sum()) if "pass_95" in onset_order.columns else 0
    mid_existing = 0
    if not e1_pair.empty:
        e1w = e1_pair[e1_pair["window_id"].astype(str) == WINDOW_ID]
        mid_existing = int((e1w.get("final_evidence_label", pd.Series(dtype=str)).astype(str) == "confirmed_directional_order").sum())
    dur_tail_pass90 = int((duration_tail.get("pass90_duration_difference", pd.Series(dtype=bool)) | duration_tail.get("pass90_tail_difference", pd.Series(dtype=bool))).sum())

    n_onset = int(rec_counts.get("onset", 0))
    n_midpoint = int(rec_counts.get("midpoint", 0))
    n_finish = int(rec_counts.get("finish", 0))
    n_none = int(rec_counts.get("none", 0))
    n_tie = int((stability.get("marker_selection_type", pd.Series(dtype=str)).astype(str) == "tie").sum())
    n_onset_unique = int(((stability["recommended_marker"].astype(str) == "onset") & (stability.get("marker_selection_type", pd.Series(dtype=str)).astype(str) == "unique_best")).sum())
    n_midpoint_unique = int(((stability["recommended_marker"].astype(str) == "midpoint") & (stability.get("marker_selection_type", pd.Series(dtype=str)).astype(str) == "unique_best")).sum())
    n_finish_unique = int(((stability["recommended_marker"].astype(str) == "finish") & (stability.get("marker_selection_type", pd.Series(dtype=str)).astype(str) == "unique_best")).sum())
    n_unresolved = int(((stability["recommended_marker"].astype(str) == "none") | (stability.get("marker_selection_type", pd.Series(dtype=str)).astype(str) == "unresolved")).sum())
    h_role = roles[roles["field"] == "H"]["final_role_label"].iloc[0] if not roles[roles["field"] == "H"].empty else "unknown"
    h_broad_confirmed = False
    if not roles[roles["field"] == "H"].empty and "statistically_confirmed_broad_transition" in roles.columns:
        h_broad_confirmed = bool(roles[roles["field"] == "H"]["statistically_confirmed_broad_transition"].iloc[0])
    if "early_onset_broad" in h_role:
        wtype = "onset_layer_with_H_early_broad_candidate"
        interp = "W45 has robust onset-layer information for H, but H broad-transition is a candidate shape unless duration/finish-tail differences are statistically confirmed. Midpoint-order should not be the default interpretation."
    elif n_onset_unique >= 3 and onset_pass90 > 0:
        wtype = "onset_staggered_window"
        interp = "Multiple fields uniquely prefer onset and some onset-order differences pass 90%; interpret as onset timing, not full order."
    elif n_midpoint_unique >= 3 and mid_existing > 0:
        wtype = "midpoint_order_window"
        interp = "Several fields uniquely prefer midpoint and there is at least one existing midpoint confirmed edge."
    elif n_tie > 0 or n_unresolved > 0:
        wtype = "marker_mixed_or_tied_window"
        interp = "Several fields have tied or unresolved timing markers; do not force a single onset or midpoint skeleton."
    else:
        wtype = "marker_mixed_window"
        interp = "Fields have mixed marker types; do not force a single midpoint-order skeleton."
    return pd.DataFrame(
        [
            {
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "n_fields": len(stability),
                "n_fields_onset_recommended_raw": n_onset,
                "n_fields_midpoint_recommended_raw": n_midpoint,
                "n_fields_finish_recommended_raw": n_finish,
                "n_fields_marker_unresolved_raw": n_none,
                "n_fields_onset_unique_best": n_onset_unique,
                "n_fields_midpoint_unique_best": n_midpoint_unique,
                "n_fields_finish_unique_best": n_finish_unique,
                "n_fields_marker_tie": n_tie,
                "n_fields_marker_unresolved": n_unresolved,
                "H_statistically_confirmed_broad_transition": h_broad_confirmed,
                "n_onset_order_pass90": onset_pass90,
                "n_onset_order_pass95": onset_pass95,
                "n_midpoint_order_pass95_existing": mid_existing,
                "n_duration_or_tail_pass90": dur_tail_pass90,
                "window_timing_type": wtype,
                "window_interpretation": interp,
                "recommended_next_action": "Use marker-specific interpretation: onset-order only where onset is comparable; describe H-like broad transitions as shape results, not midpoint-order results.",
            }
        ]
    )


def _make_figures(stability: pd.DataFrame, onset_order: pd.DataFrame, duration_tail: pd.DataFrame, paths: W45AllFieldMarkerPaths) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _ensure_dir(paths.figure_dir)
    # Marker q90 width grouped bar.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(stability))
    width = 0.18
    for idx, marker in enumerate(["onset", "midpoint", "finish", "duration"]):
        ax.bar(x + (idx - 1.5) * width, pd.to_numeric(stability[f"{marker}_q90_width"], errors="coerce"), width=width, label=marker)
    ax.set_xticks(x)
    ax.set_xticklabels(stability["field"].astype(str).tolist())
    ax.set_ylabel("q90 width (days)")
    ax.set_title("W45 all-field timing marker q90 width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figure_dir / "w45_allfield_marker_q90_width_v7_l.png", dpi=180)
    plt.close(fig)

    # Onset intervals.
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(stability))
    med = pd.to_numeric(stability["onset_median"], errors="coerce").to_numpy()
    q05 = pd.to_numeric(stability["onset_q05"], errors="coerce").to_numpy()
    q95 = pd.to_numeric(stability["onset_q95"], errors="coerce").to_numpy()
    q025 = pd.to_numeric(stability["onset_q025"], errors="coerce").to_numpy()
    q975 = pd.to_numeric(stability["onset_q975"], errors="coerce").to_numpy()
    ax.hlines(y, q025, q975, linewidth=1)
    ax.hlines(y, q05, q95, linewidth=3)
    ax.plot(med, y, "o")
    ax.set_yticks(y)
    ax.set_yticklabels(stability["field"].astype(str).tolist())
    ax.set_xlabel("onset day")
    ax.set_title("W45 all-field onset bootstrap intervals")
    fig.tight_layout()
    fig.savefig(paths.figure_dir / "w45_allfield_onset_intervals_v7_l.png", dpi=180)
    plt.close(fig)

    # Duration/tail medians.
    fig, ax = plt.subplots(figsize=(8, 5))
    fields = stability["field"].astype(str).tolist()
    x = np.arange(len(fields))
    dur = pd.to_numeric(stability["duration_median"], errors="coerce").to_numpy()
    # post-midpoint tail median from observed is not in stability; approximate with finish median - midpoint median
    tail = pd.to_numeric(stability["finish_median"], errors="coerce").to_numpy() - pd.to_numeric(stability["midpoint_median"], errors="coerce").to_numpy()
    ax.bar(x - 0.18, dur, width=0.36, label="duration median")
    ax.bar(x + 0.18, tail, width=0.36, label="finish-midpoint median")
    ax.set_xticks(x)
    ax.set_xticklabels(fields)
    ax.set_ylabel("days")
    ax.set_title("W45 duration and post-midpoint tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figure_dir / "w45_allfield_duration_tail_v7_l.png", dpi=180)
    plt.close(fig)


def _write_markdown(
    paths: W45AllFieldMarkerPaths,
    stability: pd.DataFrame,
    onset_order: pd.DataFrame,
    duration_tail: pd.DataFrame,
    roles: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> None:
    ws = window_summary.iloc[0].to_dict() if not window_summary.empty else {}
    lines = []
    lines.append("# W45 all-field timing-marker audit v7_l")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Audit whether W45 P/V/H/Je/Jw should be compared by midpoint, onset, finish, or treated as marker-mixed / broad-transition objects.")
    lines.append("")
    lines.append("## Window-level result")
    lines.append(f"- window_timing_type: `{ws.get('window_timing_type', 'unknown')}`")
    lines.append(f"- interpretation: {ws.get('window_interpretation', '')}")
    lines.append(f"- unique onset-best fields: {ws.get('n_fields_onset_unique_best', 'NA')}")
    lines.append(f"- marker-tie fields: {ws.get('n_fields_marker_tie', 'NA')}")
    lines.append(f"- H statistically confirmed broad transition: {ws.get('H_statistically_confirmed_broad_transition', 'NA')}")
    lines.append("")
    lines.append("## Field recommended markers")
    for _, r in stability.iterrows():
        lines.append(f"- {r['field']}: recommended_marker=`{r['recommended_marker']}`, selection=`{r.get('marker_selection_type', 'unknown')}`, tie_markers=`{r.get('tie_markers', 'none')}`, shape=`{r['transition_shape_label']}`, onset_q90={r.get('onset_q90_width')}, midpoint_q90={r.get('midpoint_q90_width')}, finish_q90={r.get('finish_q90_width')}")
    lines.append("")
    lines.append("## Onset-order results")
    pass_rows = onset_order[onset_order["pass_90"] == True] if not onset_order.empty else pd.DataFrame()
    if pass_rows.empty:
        lines.append("- No onset-order pair passed the 90% CI criterion.")
    else:
        for _, r in pass_rows.iterrows():
            lines.append(f"- {r.get('early_field_candidate')} < {r.get('late_field_candidate')}: {r.get('onset_order_label')}, median_delta={r.get('median_delta_onset_b_minus_a')}, q05={r.get('q05_delta_onset')}, q95={r.get('q95_delta_onset')}")
    lines.append("")
    lines.append("## Duration / finish-tail notes")
    h_rows = duration_tail[duration_tail["involves_H"] == True] if not duration_tail.empty else pd.DataFrame()
    for _, r in h_rows.iterrows():
        lines.append(f"- {r['field_a']}-{r['field_b']}: {r['broad_transition_label']}; pass90_duration={r['pass90_duration_difference']}; pass90_tail={r['pass90_tail_difference']}")
    lines.append("")
    lines.append("## Field role reinterpretation")
    for _, r in roles.iterrows():
        lines.append(f"- {r['field']}: {r['final_role_label']}; marker={r['recommended_marker']}; usable_midpoint={r['usable_for_midpoint_order']}; usable_onset={r['usable_for_onset_order']}")
    lines.append("")
    lines.append("## Prohibited interpretations")
    lines.append("- Onset order is not full transition order.")
    lines.append("- Early-onset does not imply causal upstream status.")
    lines.append("- Broad-transition is not a statistical failure to be hidden.")
    lines.append("- Midpoint instability must not be rewritten as synchrony without an equivalence test.")
    lines.append("- Do not omit P/V/H/Je/Jw when reporting W45.")
    lines.append("- Do not promote marker ties to onset or midpoint without explicitly reporting the tie.")
    lines.append("- Observed broad-transition shape is not a statistically confirmed broadness difference unless the duration/tail test passes.")
    _ensure_dir(paths.output_dir)
    (paths.output_dir / "w45_allfield_timing_marker_audit_summary_v7_l.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _ensure_dir(paths.log_dir)
    (paths.log_dir / "w45_allfield_timing_marker_audit_v7_l.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_w45_allfield_timing_marker_audit_v7_l(v7_root: Optional[Path] = None) -> dict:
    paths = _resolve_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)

    tables = _load_inputs(paths)
    audit = _validate_inputs(tables, paths)
    _write_json(audit, paths.output_dir / "input_audit_v7_l.json")

    obs = tables["v7e_obs"]
    boot = tables["v7e_boot"]
    stability = _summarize_marker_stability(obs, boot)
    observed_shape = _observed_shape(obs, stability)
    comparability = _marker_comparability(stability)
    onset_order = _pairwise_marker_delta_test(boot, marker_col="onset_day", out_prefix="onset")
    duration_tail = _duration_finish_tail_audit(boot)
    roles = _build_role_reinterpretation(stability, onset_order, duration_tail, tables["v7e1_pair"], tables["v7e2_fail"])
    window_summary = _build_window_summary(stability, onset_order, duration_tail, roles, tables["v7e1_pair"])

    _write_csv(stability, paths.output_dir / "w45_allfield_timing_marker_stability_v7_l.csv")
    _write_csv(observed_shape, paths.output_dir / "w45_allfield_observed_timing_shape_v7_l.csv")
    _write_csv(comparability, paths.output_dir / "w45_allfield_marker_comparability_v7_l.csv")
    _write_csv(onset_order, paths.output_dir / "w45_allfield_onset_order_test_v7_l.csv")
    _write_csv(duration_tail, paths.output_dir / "w45_allfield_duration_finish_tail_audit_v7_l.csv")
    _write_csv(roles, paths.output_dir / "w45_allfield_role_reinterpretation_v7_l.csv")
    _write_csv(window_summary, paths.output_dir / "w45_allfield_timing_marker_window_summary_v7_l.csv")

    _make_figures(stability, onset_order, duration_tail, paths)
    _write_markdown(paths, stability, onset_order, duration_tail, roles, window_summary)

    run_meta = {
        "status": "success",
        "created_at": _now_iso(),
        "output_tag": OUTPUT_TAG,
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "fields": FIELDS,
        "input_v7e_dir": str(paths.v7e_output_dir),
        "input_v7e1_dir": str(paths.v7e1_output_dir),
        "input_v7e2_dir": str(paths.v7e2_output_dir),
        "does_not_recompute_progress": True,
        "does_not_change_region": True,
        "does_not_change_threshold": True,
        "does_not_infer_causality": True,
        "n_fields": int(len(stability)),
        "n_onset_pair_tests": int(len(onset_order)),
        "window_timing_type": window_summary.iloc[0]["window_timing_type"] if not window_summary.empty else "unknown",
        "notes": [
            "This hotfix keeps tied marker widths explicit instead of promoting tied markers to onset.",
            "This audit tests timing marker suitability for W45 all fields.",
            "Onset-order is not full transition order.",
            "Broad-transition shape is separated from statistically confirmed duration/finish-tail differences.",
        ],
    }
    _write_json(run_meta, paths.output_dir / "run_meta.json")
    return run_meta


if __name__ == "__main__":
    run_w45_allfield_timing_marker_audit_v7_l()
