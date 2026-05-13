from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELD = "H"
OUTPUT_TAG = "w45_H_timing_marker_audit_v7_k"
MARKERS = ["onset", "midpoint", "finish", "duration"]
ORDER_MARKERS = ["onset", "midpoint", "finish"]


@dataclass
class W45HTimingMarkerPaths:
    v7_root: Path
    v7e_output_dir: Path
    v7j_output_dir: Path
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


def _resolve_paths(v7_root: Optional[Path]) -> W45HTimingMarkerPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    return W45HTimingMarkerPaths(
        v7_root=v7_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7j_output_dir=v7_root / "outputs" / "w45_H_raw025_threeband_progress_v7_j",
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _alias_columns(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    """Return a copy with canonical columns added from the first available alias."""
    out = df.copy()
    for canonical, aliases in mapping.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out[canonical] = out[alias]
                break
    return out


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


def _safe_fraction(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.isfinite(arr).sum() / arr.size)


def _join_labels(labels: Iterable[str]) -> str:
    vals = [str(x) for x in labels if str(x) and str(x) != "nan"]
    return ";".join(vals) if vals else "none"


def _load_inputs(paths: W45HTimingMarkerPaths) -> dict[str, pd.DataFrame]:
    v7e_obs = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv")
    v7e_boot = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_samples_v7_e.csv")
    v7e_sum = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_summary_v7_e.csv", required=False)
    v7e_curves = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_curves_long_v7_e.csv", required=False)

    v7j_regions = _read_csv(paths.v7j_output_dir / "w45_H_raw025_threeband_region_definition_v7_j.csv")
    v7j_obs = _read_csv(paths.v7j_output_dir / "w45_H_raw025_threeband_progress_observed_v7_j.csv")
    v7j_boot = _read_csv(paths.v7j_output_dir / "w45_H_raw025_threeband_progress_bootstrap_samples_v7_j.csv")
    v7j_sum = _read_csv(paths.v7j_output_dir / "w45_H_raw025_threeband_progress_bootstrap_summary_v7_j.csv", required=False)
    v7j_curves = _read_csv(paths.v7j_output_dir / "w45_H_raw025_threeband_progress_curves_long_v7_j.csv", required=False)

    return {
        "v7e_obs": v7e_obs,
        "v7e_boot": v7e_boot,
        "v7e_sum": v7e_sum,
        "v7e_curves": v7e_curves,
        "v7j_regions": v7j_regions,
        "v7j_obs": v7j_obs,
        "v7j_boot": v7j_boot,
        "v7j_sum": v7j_sum,
        "v7j_curves": v7j_curves,
    }


def _validate_inputs(tables: dict[str, pd.DataFrame], paths: W45HTimingMarkerPaths) -> dict:
    _require_columns(tables["v7e_obs"], ["window_id", "field", "onset_day", "midpoint_day", "finish_day", "duration"], "V7-e observed")
    _require_columns(tables["v7e_boot"], ["window_id", "field", "bootstrap_id", "onset_day", "midpoint_day", "finish_day", "duration"], "V7-e bootstrap")
    _require_columns(tables["v7j_regions"], ["region_id", "region_label", "lat_min", "lat_max", "n_raw_lat_points"], "V7-j region definition")
    _require_columns(tables["v7j_obs"], ["region_id", "region_label", "observed_onset_day", "observed_midpoint_day", "observed_finish_day", "observed_duration"], "V7-j observed")
    _require_columns(tables["v7j_boot"], ["bootstrap_id", "region_id", "region_label", "onset_day", "midpoint_day", "finish_day", "duration"], "V7-j bootstrap")

    e_obs = tables["v7e_obs"][(tables["v7e_obs"]["window_id"].astype(str) == WINDOW_ID) & (tables["v7e_obs"]["field"].astype(str) == FIELD)]
    e_boot = tables["v7e_boot"][(tables["v7e_boot"]["window_id"].astype(str) == WINDOW_ID) & (tables["v7e_boot"]["field"].astype(str) == FIELD)]
    j_regions = tables["v7j_regions"]
    j_obs = tables["v7j_obs"]
    j_boot = tables["v7j_boot"]

    expected_regions = {"R1_low", "R2_mid", "R3_high"}
    found_regions = set(j_regions["region_label"].astype(str))

    audit = {
        "status": "checked",
        "created_at": _now_iso(),
        "v7e_output_dir": str(paths.v7e_output_dir),
        "v7j_output_dir": str(paths.v7j_output_dir),
        "v7e_observed_has_w45_H": bool(len(e_obs) == 1),
        "v7e_bootstrap_has_w45_H": bool(len(e_boot) > 0),
        "v7j_regions_found": sorted(found_regions),
        "v7j_regions_expected": sorted(expected_regions),
        "v7j_regions_complete": bool(expected_regions.issubset(found_regions)),
        "v7j_observed_rows": int(len(j_obs)),
        "v7j_bootstrap_rows": int(len(j_boot)),
        "v7e_curves_available": bool(not tables["v7e_curves"].empty),
        "v7j_curves_available": bool(not tables["v7j_curves"].empty),
    }
    if not audit["v7e_observed_has_w45_H"]:
        raise ValueError("V7-e observed table must contain exactly one W002/H row.")
    if not audit["v7e_bootstrap_has_w45_H"]:
        raise ValueError("V7-e bootstrap table has no W002/H samples.")
    if not audit["v7j_regions_complete"]:
        raise ValueError(f"V7-j region definitions are incomplete. Found={sorted(found_regions)} expected={sorted(expected_regions)}")
    return audit


def _extract_progress_at(curves: pd.DataFrame, unit_type: str, unit_label: str, day: int) -> float:
    if curves is None or curves.empty:
        return np.nan
    df = curves.copy()
    if unit_type == "whole_field":
        if "field" not in df.columns:
            return np.nan
        df = df[(df.get("window_id", "").astype(str) == WINDOW_ID) & (df["field"].astype(str) == FIELD)]
    else:
        if "region_label" not in df.columns:
            return np.nan
        df = df[df["region_label"].astype(str) == unit_label]
    if "day" not in df.columns or "progress" not in df.columns:
        return np.nan
    sub = df[pd.to_numeric(df["day"], errors="coerce") == int(day)]
    if sub.empty:
        return np.nan
    return float(pd.to_numeric(sub.iloc[0]["progress"], errors="coerce"))


def _build_units(tables: dict[str, pd.DataFrame]) -> list[dict]:
    units: list[dict] = []
    e_obs = tables["v7e_obs"][(tables["v7e_obs"]["window_id"].astype(str) == WINDOW_ID) & (tables["v7e_obs"]["field"].astype(str) == FIELD)].iloc[0]
    e_boot = tables["v7e_boot"][(tables["v7e_boot"]["window_id"].astype(str) == WINDOW_ID) & (tables["v7e_boot"]["field"].astype(str) == FIELD)].copy()
    units.append(
        {
            "unit_type": "whole_field",
            "unit_label": "whole_field_H",
            "lat_min": np.nan,
            "lat_max": np.nan,
            "lat_center": np.nan,
            "n_raw_lat_points": np.nan,
            "observed": {
                "onset": e_obs.get("onset_day", np.nan),
                "midpoint": e_obs.get("midpoint_day", np.nan),
                "finish": e_obs.get("finish_day", np.nan),
                "duration": e_obs.get("duration", np.nan),
                "pre_post_separation_label": e_obs.get("pre_post_separation_label", "unknown"),
                "progress_quality_label": e_obs.get("progress_quality_label", "unknown"),
                "progress_monotonicity_corr": e_obs.get("progress_monotonicity_corr", np.nan),
                "n_crossings_025": e_obs.get("n_crossings_025", np.nan),
                "n_crossings_050": e_obs.get("n_crossings_050", np.nan),
                "n_crossings_075": e_obs.get("n_crossings_075", np.nan),
                "analysis_window_start": e_obs.get("analysis_window_start", 30),
                "analysis_window_end": e_obs.get("analysis_window_end", 60),
            },
            "bootstrap": e_boot,
            "curves": tables.get("v7e_curves", pd.DataFrame()),
        }
    )

    regions = tables["v7j_regions"].copy()
    obs_j = tables["v7j_obs"].copy()
    boot_j = tables["v7j_boot"].copy()
    for _, reg in regions.sort_values("region_id").iterrows():
        label = str(reg["region_label"])
        sub_obs = obs_j[obs_j["region_label"].astype(str) == label]
        if sub_obs.empty:
            raise ValueError(f"Missing V7-j observed row for {label}")
        row = sub_obs.iloc[0]
        sub_boot = boot_j[boot_j["region_label"].astype(str) == label].copy()
        if sub_boot.empty:
            raise ValueError(f"Missing V7-j bootstrap rows for {label}")
        units.append(
            {
                "unit_type": "raw025_threeband",
                "unit_label": label,
                "lat_min": reg.get("lat_min", np.nan),
                "lat_max": reg.get("lat_max", np.nan),
                "lat_center": reg.get("lat_center", np.nan),
                "n_raw_lat_points": reg.get("n_raw_lat_points", np.nan),
                "observed": {
                    "onset": row.get("observed_onset_day", np.nan),
                    "midpoint": row.get("observed_midpoint_day", np.nan),
                    "finish": row.get("observed_finish_day", np.nan),
                    "duration": row.get("observed_duration", np.nan),
                    "pre_post_separation_label": row.get("pre_post_separation_label", "unknown"),
                    "progress_quality_label": row.get("progress_quality_label", "unknown"),
                    "progress_monotonicity_corr": row.get("progress_monotonicity_corr", np.nan),
                    "n_crossings_025": row.get("n_crossings_025", np.nan),
                    "n_crossings_050": row.get("n_crossings_050", np.nan),
                    "n_crossings_075": row.get("n_crossings_075", np.nan),
                    "analysis_window_start": row.get("analysis_window_start", 30),
                    "analysis_window_end": row.get("analysis_window_end", 60),
                },
                "bootstrap": sub_boot,
                "curves": tables.get("v7j_curves", pd.DataFrame()),
            }
        )
    return units


def _marker_stats(values: pd.Series | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "median": np.nan,
            "q05": np.nan,
            "q95": np.nan,
            "q90_width": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "valid_fraction": _safe_fraction(arr),
        }
    q05, q25, q50, q75, q95 = np.nanquantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "median": float(q50),
        "q05": float(q05),
        "q95": float(q95),
        "q90_width": float(q95 - q05),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "valid_fraction": _safe_fraction(arr),
    }


def _min_labels(widths: dict[str, float]) -> str:
    vals = {k: v for k, v in widths.items() if np.isfinite(v)}
    if not vals:
        return "none"
    m = min(vals.values())
    return _join_labels(k for k, v in vals.items() if np.isclose(v, m))


def _max_labels(widths: dict[str, float]) -> str:
    vals = {k: v for k, v in widths.items() if np.isfinite(v)}
    if not vals:
        return "none"
    m = max(vals.values())
    return _join_labels(k for k, v in vals.items() if np.isclose(v, m))


def _timing_marker_label(widths: dict[str, float], progress_quality: str) -> tuple[str, str]:
    main_widths = {k: widths.get(k, np.nan) for k in ORDER_MARKERS}
    most = _min_labels(main_widths)
    least = _max_labels(main_widths)
    if most == "none":
        return "all_markers_unavailable", "No valid onset/midpoint/finish samples are available."
    if "nonmonotonic" in str(progress_quality):
        caution = "Progress quality is nonmonotonic or uncertain; marker stability is diagnostic, not a confirmed timing result."
    else:
        caution = "Progress quality is not flagged as nonmonotonic in the observed row."
    if "onset" in most and "midpoint" not in most:
        return "onset_most_stable_candidate", f"Onset has the smallest q90 width among onset/midpoint/finish. {caution}"
    if "midpoint" in most:
        return "midpoint_representative_candidate", f"Midpoint is among the most stable markers by q90 width. {caution}"
    if "finish" in most:
        return "finish_most_stable_candidate", f"Finish has the smallest q90 width; this describes completion timing, not necessarily onset/order. {caution}"
    return "timing_marker_unresolved", f"Most stable marker={most}; least stable marker={least}. {caution}"


def _build_marker_stability(units: list[dict]) -> pd.DataFrame:
    rows = []
    for unit in units:
        boot = unit["bootstrap"].copy()
        # Canonical marker column names.
        boot = _alias_columns(
            boot,
            {
                "onset_day": ["progress_onset_day", "onset"],
                "midpoint_day": ["progress_midpoint_day", "midpoint"],
                "finish_day": ["progress_finish_day", "finish"],
                "duration": ["progress_duration"],
            },
        )
        marker_cols = {
            "onset": "onset_day",
            "midpoint": "midpoint_day",
            "finish": "finish_day",
            "duration": "duration",
        }
        row = {
            "unit_type": unit["unit_type"],
            "unit_label": unit["unit_label"],
            "lat_min": unit["lat_min"],
            "lat_max": unit["lat_max"],
            "lat_center": unit.get("lat_center", np.nan),
            "n_raw_lat_points": unit["n_raw_lat_points"],
        }
        widths = {}
        for marker, col in marker_cols.items():
            if col not in boot.columns:
                vals = np.asarray([], dtype=float)
            else:
                vals = _numeric_series(boot[col]).to_numpy(dtype=float)
            stats = _marker_stats(vals)
            for key, val in stats.items():
                row[f"{marker}_{key}"] = val
            widths[marker] = stats["q90_width"]
        main_widths = {k: widths[k] for k in ORDER_MARKERS}
        row["most_stable_marker"] = _min_labels(widths)
        row["least_stable_marker"] = _max_labels(widths)
        row["most_stable_order_marker"] = _min_labels(main_widths)
        row["least_stable_order_marker"] = _max_labels(main_widths)
        label, interp = _timing_marker_label(widths, unit["observed"].get("progress_quality_label", "unknown"))
        row["timing_marker_label"] = label
        row["marker_stability_interpretation"] = interp
        rows.append(row)
    return pd.DataFrame(rows)


def _shape_label(obs: dict) -> tuple[str, str]:
    quality = str(obs.get("progress_quality_label", "unknown"))
    onset = float(pd.to_numeric(obs.get("onset", np.nan), errors="coerce"))
    midpoint = float(pd.to_numeric(obs.get("midpoint", np.nan), errors="coerce"))
    finish = float(pd.to_numeric(obs.get("finish", np.nan), errors="coerce"))
    duration = float(pd.to_numeric(obs.get("duration", np.nan), errors="coerce"))
    if "nonmonotonic" in quality:
        return "nonmonotonic_or_uncertain_shape", "Observed progress quality is nonmonotonic; shape should be treated as diagnostic."
    if not all(np.isfinite([onset, midpoint, finish, duration])):
        return "shape_unavailable", "At least one observed timing marker is unavailable."
    left_span = midpoint - onset
    right_span = finish - midpoint
    if right_span > left_span:
        return "early_onset_broad_finish", "Observed transition reaches onset earlier than completion; finish side is broader than onset-to-midpoint side."
    if left_span > right_span:
        return "late_midpoint_or_fast_finish", "Observed transition takes longer to reach midpoint than to finish."
    return "balanced_or_compact_transition", "Observed onset-to-midpoint and midpoint-to-finish spans are comparable."


def _build_observed_shape(units: list[dict]) -> pd.DataFrame:
    rows = []
    for unit in units:
        obs = unit["observed"]
        onset = pd.to_numeric(obs.get("onset", np.nan), errors="coerce")
        midpoint = pd.to_numeric(obs.get("midpoint", np.nan), errors="coerce")
        finish = pd.to_numeric(obs.get("finish", np.nan), errors="coerce")
        label, interp = _shape_label(obs)
        rows.append(
            {
                "unit_type": unit["unit_type"],
                "unit_label": unit["unit_label"],
                "lat_min": unit["lat_min"],
                "lat_max": unit["lat_max"],
                "lat_center": unit.get("lat_center", np.nan),
                "n_raw_lat_points": unit["n_raw_lat_points"],
                "observed_onset": onset,
                "observed_midpoint": midpoint,
                "observed_finish": finish,
                "observed_duration": pd.to_numeric(obs.get("duration", np.nan), errors="coerce"),
                "observed_onset_to_midpoint": midpoint - onset if np.isfinite(onset) and np.isfinite(midpoint) else np.nan,
                "observed_midpoint_to_finish": finish - midpoint if np.isfinite(finish) and np.isfinite(midpoint) else np.nan,
                "pre_post_separation_label": obs.get("pre_post_separation_label", "unknown"),
                "progress_quality_label": obs.get("progress_quality_label", "unknown"),
                "progress_monotonicity_corr": obs.get("progress_monotonicity_corr", np.nan),
                "n_crossings_025": obs.get("n_crossings_025", np.nan),
                "n_crossings_050": obs.get("n_crossings_050", np.nan),
                "n_crossings_075": obs.get("n_crossings_075", np.nan),
                "observed_shape_label": label,
                "observed_shape_interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


def _build_censoring(units: list[dict]) -> pd.DataFrame:
    rows = []
    for unit in units:
        obs = unit["observed"]
        boot = unit["bootstrap"].copy()
        boot = _alias_columns(
            boot,
            {
                "onset_day": ["progress_onset_day", "onset"],
                "finish_day": ["progress_finish_day", "finish"],
            },
        )
        start = int(pd.to_numeric(obs.get("analysis_window_start", 30), errors="coerce"))
        end = int(pd.to_numeric(obs.get("analysis_window_end", 60), errors="coerce"))
        onset = _numeric_series(boot["onset_day"]) if "onset_day" in boot.columns else pd.Series(dtype=float)
        finish = _numeric_series(boot["finish_day"]) if "finish_day" in boot.columns else pd.Series(dtype=float)
        finite_onset = onset[np.isfinite(onset)]
        finite_finish = finish[np.isfinite(finish)]
        frac_left = float((finite_onset == start).mean()) if len(finite_onset) else np.nan
        frac_right = float((finite_finish == end).mean()) if len(finite_finish) else np.nan
        labels = []
        if np.isfinite(frac_left) and frac_left > 0:
            labels.append("possible_left_censoring")
        if np.isfinite(frac_right) and frac_right > 0:
            labels.append("possible_right_censoring")
        if not labels:
            labels.append("no_edge_hit_in_bootstrap")
        curves = unit["curves"]
        rows.append(
            {
                "unit_type": unit["unit_type"],
                "unit_label": unit["unit_label"],
                "lat_min": unit["lat_min"],
                "lat_max": unit["lat_max"],
                "analysis_start": start,
                "analysis_end": end,
                "observed_onset": obs.get("onset", np.nan),
                "observed_midpoint": obs.get("midpoint", np.nan),
                "observed_finish": obs.get("finish", np.nan),
                "observed_duration": obs.get("duration", np.nan),
                "fraction_onset_at_left_edge": frac_left,
                "fraction_finish_at_right_edge": frac_right,
                "onset_q05": _safe_quantile(onset, 0.05),
                "onset_q95": _safe_quantile(onset, 0.95),
                "finish_q05": _safe_quantile(finish, 0.05),
                "finish_q95": _safe_quantile(finish, 0.95),
                "progress_at_day30": _extract_progress_at(curves, unit["unit_type"], unit["unit_label"], 30),
                "progress_at_day45": _extract_progress_at(curves, unit["unit_type"], unit["unit_label"], 45),
                "progress_at_day60": _extract_progress_at(curves, unit["unit_type"], unit["unit_label"], 60),
                "window_censoring_label": _join_labels(labels),
                "censoring_interpretation": "Edge-hit fractions are reported as censoring risk diagnostics; they do not prove synchrony or out-of-window timing by themselves.",
            }
        )
    return pd.DataFrame(rows)


def _lookup(df: pd.DataFrame, unit_label: str) -> pd.Series:
    sub = df[df["unit_label"].astype(str) == unit_label]
    if sub.empty:
        raise ValueError(f"Missing unit {unit_label} in table")
    return sub.iloc[0]


def _build_decision(stab: pd.DataFrame, shape: pd.DataFrame, censor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in stab.iterrows():
        label = str(s["unit_label"])
        sh = _lookup(shape, label)
        ce = _lookup(censor, label)
        quality = str(sh.get("progress_quality_label", "unknown"))
        most_order = str(s.get("most_stable_order_marker", "none"))
        # Recommendation is based on the narrowest q90 width among onset/midpoint/finish.
        recommended_marker = most_order if most_order != "none" else "none"
        midpoint_ok = ("midpoint" in recommended_marker) and ("nonmonotonic" not in quality)
        early_onset = "onset" in recommended_marker
        broad_transition = False
        if np.isfinite(s.get("finish_q90_width", np.nan)) and np.isfinite(s.get("onset_q90_width", np.nan)):
            broad_transition = bool(s["finish_q90_width"] > s["onset_q90_width"])
        if np.isfinite(s.get("duration_q90_width", np.nan)) and np.isfinite(s.get("onset_q90_width", np.nan)):
            broad_transition = broad_transition or bool(s["duration_q90_width"] > s["onset_q90_width"])
        needs_extension = "possible" in str(ce.get("window_censoring_label", ""))
        if midpoint_ok:
            decision = "midpoint_can_be_used_diagnostically"
            interpretation = "Midpoint is among the most stable order markers and observed progress is not flagged nonmonotonic. This is still a timing diagnostic, not a causal claim."
        elif early_onset:
            decision = "prefer_onset_over_midpoint"
            interpretation = "Onset is more stable than midpoint/finish by q90 width; this unit is better described as early-onset candidate than midpoint-order object."
        elif recommended_marker == "none":
            decision = "no_reliable_marker"
            interpretation = "No valid timing marker can be recommended from the available bootstrap samples."
        else:
            decision = "marker_unresolved_or_completion_focused"
            interpretation = f"Recommended marker={recommended_marker}; avoid forcing midpoint-order interpretation."
        rows.append(
            {
                "unit_type": s["unit_type"],
                "unit_label": label,
                "lat_min": s.get("lat_min", np.nan),
                "lat_max": s.get("lat_max", np.nan),
                "recommended_marker": recommended_marker,
                "can_use_midpoint_for_order": bool(midpoint_ok),
                "can_describe_as_early_onset": bool(early_onset),
                "can_describe_as_broad_transition": bool(broad_transition),
                "needs_window_extension_audit": bool(needs_extension),
                "timing_marker_decision": decision,
                "final_timing_interpretation": interpretation,
                "do_not_overinterpret": "Do not treat not-distinguishable as synchrony; do not convert early-onset/broad-transition into confirmed field order.",
            }
        )
    return pd.DataFrame(rows)


def _build_window_summary(decision: pd.DataFrame) -> pd.DataFrame:
    whole = decision[decision["unit_label"].astype(str) == "whole_field_H"]
    whole_marker = str(whole.iloc[0]["recommended_marker"]) if not whole.empty else "unknown"
    counts = decision["recommended_marker"].value_counts().to_dict()
    n_onset = sum(1 for x in decision["recommended_marker"].astype(str) if "onset" in x)
    n_midpoint = sum(1 for x in decision["recommended_marker"].astype(str) if "midpoint" in x)
    n_none = sum(1 for x in decision["recommended_marker"].astype(str) if x == "none")
    n_extension = int(decision["needs_window_extension_audit"].astype(bool).sum())
    if n_onset >= max(n_midpoint, 1):
        wtype = "early_onset_broad_transition_candidate"
        action = "interpret_H_as_timing-shape_candidate_before_using_midpoint_order"
        interp = "Onset is frequently the preferred marker; W45-H should be assessed as early-onset/broad-transition before any midpoint-order claim."
    elif n_midpoint > 0:
        wtype = "midpoint_order_candidate"
        action = "midpoint_order_can_be_revisited_for_units_where_midpoint_is_recommended"
        interp = "At least one unit supports midpoint as a diagnostic marker; use only those units and keep evidence-level labels."
    else:
        wtype = "timing_marker_unresolved"
        action = "do_not_use_W45_H_for_order_until_marker_issue_is_resolved"
        interp = "No stable marker consensus is found."
    if n_extension > 0:
        interp += " Censoring risk is present in at least one unit; an extension audit may be needed before final interpretation."
    return pd.DataFrame(
        [
            {
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "field": FIELD,
                "n_units": int(len(decision)),
                "n_units_onset_recommended": int(n_onset),
                "n_units_midpoint_recommended": int(n_midpoint),
                "n_units_no_marker": int(n_none),
                "n_units_needing_window_extension_audit": int(n_extension),
                "whole_field_recommended_marker": whole_marker,
                "marker_recommendation_counts": json.dumps(counts, ensure_ascii=False),
                "window_level_timing_type": wtype,
                "window_level_interpretation": interp,
                "recommended_next_action": action,
            }
        ]
    )


def _plot_marker_widths(stab: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _ensure_dir(out_path.parent)
    labels = stab["unit_label"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    for i, marker in enumerate(MARKERS):
        vals = pd.to_numeric(stab[f"{marker}_q90_width"], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (i - 1.5) * width, vals, width, label=marker)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("q90 width (days)")
    ax.set_title("W45-H timing marker q90 width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_marker_intervals(stab: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    _ensure_dir(out_path.parent)
    rows = []
    for _, r in stab.iterrows():
        for marker in ORDER_MARKERS:
            rows.append(
                {
                    "unit_label": str(r["unit_label"]),
                    "marker": marker,
                    "median": r.get(f"{marker}_median", np.nan),
                    "q05": r.get(f"{marker}_q05", np.nan),
                    "q95": r.get(f"{marker}_q95", np.nan),
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(plot_df))))
    y = np.arange(len(plot_df))
    med = pd.to_numeric(plot_df["median"], errors="coerce").to_numpy(dtype=float)
    q05 = pd.to_numeric(plot_df["q05"], errors="coerce").to_numpy(dtype=float)
    q95 = pd.to_numeric(plot_df["q95"], errors="coerce").to_numpy(dtype=float)
    xerr = np.vstack([med - q05, q95 - med])
    ax.errorbar(med, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels((plot_df["unit_label"] + " / " + plot_df["marker"]).tolist())
    ax.set_xlabel("day index")
    ax.set_title("W45-H onset/midpoint/finish bootstrap intervals")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_markdown_summary(
    path: Path,
    audit: dict,
    stab: pd.DataFrame,
    shape: pd.DataFrame,
    censor: pd.DataFrame,
    decision: pd.DataFrame,
    window_summary: pd.DataFrame,
) -> None:
    _ensure_dir(path.parent)
    ws = window_summary.iloc[0].to_dict()
    lines = [
        f"# W45-H timing marker audit v7_k",
        "",
        "## Purpose",
        "Assess whether W45-H should continue to be represented by midpoint timing, or whether it is better described as early-onset / broad-transition.",
        "",
        "## Inputs",
        f"- V7-e whole-field H available: {audit.get('v7e_observed_has_w45_H')}",
        f"- V7-j raw025 three regions complete: {audit.get('v7j_regions_complete')}",
        "",
        "## Window-level decision",
        f"- window_level_timing_type: `{ws.get('window_level_timing_type')}`",
        f"- whole_field_recommended_marker: `{ws.get('whole_field_recommended_marker')}`",
        f"- recommended_next_action: `{ws.get('recommended_next_action')}`",
        "",
        str(ws.get("window_level_interpretation")),
        "",
        "## Per-unit marker decisions",
    ]
    for _, r in decision.iterrows():
        lines.append(f"- `{r['unit_label']}`: recommended_marker=`{r['recommended_marker']}`, decision=`{r['timing_marker_decision']}`. {r['final_timing_interpretation']}")
    lines += [
        "",
        "## Prohibited interpretations",
        "- Do not treat order-not-resolved as synchrony; synchrony requires a separate equivalence test.",
        "- Do not convert early-onset / broad-transition into confirmed field order.",
        "- Do not infer causality or pathway direction from this timing-marker audit.",
        "- Do not hide whole-field, R1_low, R2_mid, or R3_high; each unit remains part of the result state.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_w45_H_timing_marker_audit_v7_k(v7_root: Optional[Path] = None) -> dict:
    paths = _resolve_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)

    tables = _load_inputs(paths)
    audit = _validate_inputs(tables, paths)
    _write_json(audit, paths.output_dir / "input_audit_v7_k.json")

    units = _build_units(tables)
    marker_stability = _build_marker_stability(units)
    observed_shape = _build_observed_shape(units)
    censoring = _build_censoring(units)
    decision = _build_decision(marker_stability, observed_shape, censoring)
    window_summary = _build_window_summary(decision)

    _write_csv(marker_stability, paths.output_dir / "w45_H_timing_marker_stability_v7_k.csv")
    _write_csv(observed_shape, paths.output_dir / "w45_H_observed_timing_shape_v7_k.csv")
    _write_csv(censoring, paths.output_dir / "w45_H_window_censoring_audit_v7_k.csv")
    _write_csv(decision, paths.output_dir / "w45_H_timing_marker_decision_v7_k.csv")
    _write_csv(window_summary, paths.output_dir / "w45_H_timing_marker_window_summary_v7_k.csv")

    _plot_marker_widths(marker_stability, paths.figure_dir / "w45_H_timing_marker_q90_width_v7_k.png")
    _plot_marker_intervals(marker_stability, paths.figure_dir / "w45_H_timing_marker_intervals_v7_k.png")

    _write_markdown_summary(
        paths.output_dir / "w45_H_timing_marker_audit_summary_v7_k.md",
        audit,
        marker_stability,
        observed_shape,
        censoring,
        decision,
        window_summary,
    )

    run_meta = {
        "status": "success",
        "created_at": _now_iso(),
        "output_tag": OUTPUT_TAG,
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field": FIELD,
        "v7_root": str(paths.v7_root),
        "input_v7e_dir": str(paths.v7e_output_dir),
        "input_v7j_dir": str(paths.v7j_output_dir),
        "audit_type": "timing_marker_stability",
        "does_not_change_region": True,
        "does_not_recompute_progress": True,
        "does_not_change_threshold": True,
        "n_units": int(len(marker_stability)),
        "outputs": {
            "input_audit": str(paths.output_dir / "input_audit_v7_k.json"),
            "marker_stability": str(paths.output_dir / "w45_H_timing_marker_stability_v7_k.csv"),
            "observed_shape": str(paths.output_dir / "w45_H_observed_timing_shape_v7_k.csv"),
            "censoring_audit": str(paths.output_dir / "w45_H_window_censoring_audit_v7_k.csv"),
            "marker_decision": str(paths.output_dir / "w45_H_timing_marker_decision_v7_k.csv"),
            "window_summary": str(paths.output_dir / "w45_H_timing_marker_window_summary_v7_k.csv"),
            "summary_md": str(paths.output_dir / "w45_H_timing_marker_audit_summary_v7_k.md"),
        },
    }
    _write_json(run_meta, paths.output_dir / "run_meta.json")

    log_text = (
        "# W45-H timing marker audit v7_k\n\n"
        f"Created: {run_meta['created_at']}\n\n"
        "This audit reads V7-e whole-field H and V7-j raw025 three-region H results. "
        "It does not recompute progress, change regions, or change thresholds.\n\n"
        "It compares onset, midpoint, finish, and duration stability to decide whether W45-H should be interpreted as a midpoint-order object or as an early-onset / broad-transition candidate.\n"
    )
    (paths.log_dir / "w45_H_timing_marker_audit_v7_k.md").write_text(log_text, encoding="utf-8")
    return run_meta


if __name__ == "__main__":
    run_w45_H_timing_marker_audit_v7_k()
