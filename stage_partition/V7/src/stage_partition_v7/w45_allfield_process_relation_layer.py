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

WINDOW_ID = "W002"
ANCHOR_DAY = 45
FIELDS = ["P", "V", "H", "Je", "Jw"]
OUTPUT_TAG = "w45_allfield_process_relation_layer_v7_n"
V7M_TAG = "w45_allfield_transition_marker_definition_audit_v7_m"
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75]
EARLY_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
PROGRESS_NEAR_MARGINS = [0.05, 0.10, 0.20]


@dataclass
class ProcessRelationPaths:
    v7_root: Path
    v7e_output_dir: Path
    v7m_output_dir: Path
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


def _resolve_paths(v7_root: Optional[Path]) -> ProcessRelationPaths:
    if v7_root is None:
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    return ProcessRelationPaths(
        v7_root=v7_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        v7m_output_dir=v7_root / "outputs" / V7M_TAG,
        output_dir=v7_root / "outputs" / OUTPUT_TAG,
        log_dir=v7_root / "logs" / OUTPUT_TAG,
        figure_dir=v7_root / "outputs" / OUTPUT_TAG / "figures",
    )


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _safe_arr(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe_quantile(values, q: float) -> float:
    arr = _safe_arr(values)
    if arr.size == 0:
        return np.nan
    return float(np.nanquantile(arr, q))


def _safe_median(values) -> float:
    arr = _safe_arr(values)
    if arr.size == 0:
        return np.nan
    return float(np.nanmedian(arr))


def _valid_fraction(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.isfinite(arr).sum() / arr.size)


def _fmt_threshold(q: float) -> str:
    return f"t{int(round(float(q) * 100)):02d}"


def _marker_cols() -> list[str]:
    cols = ["departure90_day", "departure95_day"]
    cols += [f"{_fmt_threshold(q)}_day" for q in THRESHOLDS]
    cols += ["peak_change_day_raw", "peak_change_day_smooth3", "duration_25_75", "tail_50_75", "early_span_25_50"]
    return cols


def _marker_display(marker_col: str) -> tuple[str, str]:
    if marker_col.startswith("departure"):
        return "departure", marker_col.replace("_day", "")
    if marker_col.startswith("t") and marker_col.endswith("_day"):
        return "threshold", marker_col.replace("_day", "")
    if marker_col.startswith("peak"):
        return "peak_change", marker_col.replace("_day", "")
    if marker_col.startswith("duration"):
        return "duration", marker_col
    if marker_col.startswith("tail"):
        return "tail", marker_col
    if marker_col.startswith("early_span"):
        return "early_span", marker_col
    return "other", marker_col.replace("_day", "")


def _stats(values) -> dict[str, float]:
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


def _ensure_v7m_outputs(paths: ProcessRelationPaths) -> None:
    required = [
        paths.v7m_output_dir / "w45_allfield_progress_observed_curves_long_v7_m.csv",
        paths.v7m_output_dir / "w45_allfield_progress_bootstrap_curves_long_v7_m.csv",
        paths.v7m_output_dir / "w45_allfield_marker_observed_v7_m.csv",
        paths.v7m_output_dir / "w45_allfield_marker_bootstrap_samples_v7_m.csv",
    ]
    if all(p.exists() for p in required):
        return
    # Try to build them by invoking the V7-m runner if the patch is already installed.
    try:
        from stage_partition_v7.w45_allfield_transition_marker_definition_audit import (  # type: ignore
            run_w45_allfield_transition_marker_definition_audit_v7_m,
        )
        print("[V7-n] Required V7-m curve/marker outputs not found; running V7-m first.")
        run_w45_allfield_transition_marker_definition_audit_v7_m(paths.v7_root)
    except Exception as exc:  # noqa: BLE001
        missing = [str(p) for p in required if not p.exists()]
        raise FileNotFoundError(
            "V7-n requires V7-m W45 progress curves and marker outputs. "
            "Run scripts/run_w45_allfield_transition_marker_definition_audit_v7_m.py first, "
            f"or install the V7-m patch. Missing: {missing}. Original error: {exc}"
        ) from exc


def _load_base_tables(paths: ProcessRelationPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_v7m_outputs(paths)
    obs_curves = _read_csv(paths.v7m_output_dir / "w45_allfield_progress_observed_curves_long_v7_m.csv")
    boot_curves = _read_csv(paths.v7m_output_dir / "w45_allfield_progress_bootstrap_curves_long_v7_m.csv")
    obs_markers = _read_csv(paths.v7m_output_dir / "w45_allfield_marker_observed_v7_m.csv")
    boot_markers = _read_csv(paths.v7m_output_dir / "w45_allfield_marker_bootstrap_samples_v7_m.csv")
    v7e_obs = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv")
    return obs_curves, boot_curves, obs_markers, boot_markers, v7e_obs


def _validate_inputs(obs_curves: pd.DataFrame, boot_curves: pd.DataFrame, obs_markers: pd.DataFrame, boot_markers: pd.DataFrame, paths: ProcessRelationPaths) -> dict:
    _require_columns(obs_curves, ["window_id", "field", "day", "progress"], "observed curves")
    _require_columns(boot_curves, ["window_id", "field", "bootstrap_id", "day", "progress"], "bootstrap curves")
    _require_columns(obs_markers, ["window_id", "field"] + _marker_cols()[:4], "observed markers")
    _require_columns(boot_markers, ["window_id", "field", "bootstrap_id"] + _marker_cols()[:4], "bootstrap markers")
    obs_w = obs_curves[obs_curves["window_id"].astype(str) == WINDOW_ID]
    boot_w = boot_curves[boot_curves["window_id"].astype(str) == WINDOW_ID]
    marker_w = boot_markers[boot_markers["window_id"].astype(str) == WINDOW_ID]
    obs_fields = sorted(set(obs_w["field"].astype(str)))
    boot_fields = sorted(set(boot_w["field"].astype(str)))
    marker_fields = sorted(set(marker_w["field"].astype(str)))
    audit = {
        "status": "checked",
        "created_at": _now_iso(),
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "fields_required": FIELDS,
        "observed_curve_fields_found": obs_fields,
        "bootstrap_curve_fields_found": boot_fields,
        "bootstrap_marker_fields_found": marker_fields,
        "observed_curves_complete": all(f in obs_fields for f in FIELDS),
        "bootstrap_curves_complete": all(f in boot_fields for f in FIELDS),
        "bootstrap_markers_complete": all(f in marker_fields for f in FIELDS),
        "observed_curve_rows": int(len(obs_w)),
        "bootstrap_curve_rows": int(len(boot_w)),
        "bootstrap_marker_rows": int(len(marker_w)),
        "input_representation": "current_v7e_progress_profile_via_v7m_curves",
        "v7e_output_dir": str(paths.v7e_output_dir),
        "v7m_output_dir": str(paths.v7m_output_dir),
        "output_dir": str(paths.output_dir),
    }
    if not (audit["observed_curves_complete"] and audit["bootstrap_curves_complete"] and audit["bootstrap_markers_complete"]):
        raise ValueError(f"V7-n input audit failed: {audit}")
    return audit


def _add_dprogress(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["day"] = pd.to_numeric(out["day"], errors="coerce").astype(int)
    out["progress"] = pd.to_numeric(out["progress"], errors="coerce")
    out = out.sort_values(group_cols + ["day"])
    out["dprogress"] = out.groupby(group_cols, sort=False)["progress"].diff()
    return out


def build_field_transition_curves(obs_curves: pd.DataFrame, boot_curves: pd.DataFrame) -> pd.DataFrame:
    obs = obs_curves[(obs_curves["window_id"].astype(str) == WINDOW_ID) & (obs_curves["field"].astype(str).isin(FIELDS))].copy()
    boot = boot_curves[(boot_curves["window_id"].astype(str) == WINDOW_ID) & (boot_curves["field"].astype(str).isin(FIELDS))].copy()
    obs = _add_dprogress(obs, ["field"])
    boot = _add_dprogress(boot, ["field", "bootstrap_id"])
    rows = []
    for (field, day), bsub in boot.groupby(["field", "day"], sort=True):
        p = pd.to_numeric(bsub["progress"], errors="coerce")
        dp = pd.to_numeric(bsub["dprogress"], errors="coerce")
        osub = obs[(obs["field"].astype(str) == str(field)) & (pd.to_numeric(obs["day"], errors="coerce") == int(day))]
        oprog = float(osub["progress"].iloc[0]) if not osub.empty else np.nan
        odprog = float(osub["dprogress"].iloc[0]) if not osub.empty else np.nan
        row = {
            "window_id": WINDOW_ID,
            "anchor_day": ANCHOR_DAY,
            "field": field,
            "day": int(day),
            "relative_to_anchor": int(day) - ANCHOR_DAY,
            "progress_observed": oprog,
            "dprogress_observed": odprog,
            "progress_bootstrap_median": _safe_median(p),
            "progress_bootstrap_q05": _safe_quantile(p, 0.05),
            "progress_bootstrap_q25": _safe_quantile(p, 0.25),
            "progress_bootstrap_q75": _safe_quantile(p, 0.75),
            "progress_bootstrap_q95": _safe_quantile(p, 0.95),
            "dprogress_bootstrap_median": _safe_median(dp),
            "dprogress_bootstrap_q05": _safe_quantile(dp, 0.05),
            "dprogress_bootstrap_q95": _safe_quantile(dp, 0.95),
            "curve_source": "current_v7e_progress_profile_via_v7m_curves",
        }
        for q in [0.10, 0.25, 0.50, 0.75]:
            row[f"prob_progress_above_{int(q*100):03d}"] = float(np.nanmean(np.asarray(p, dtype=float) >= q)) if len(p) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _shape_label_from_row(field: str, obs_row: pd.Series | None, marker_stats: dict[str, float]) -> tuple[str, str, str]:
    if obs_row is None:
        return "transition_not_established", "single_marker_not_recommended", "No observed row available."
    sep = str(obs_row.get("pre_post_separation_label", "unknown"))
    quality = str(obs_row.get("progress_quality_label", "unknown"))
    t25_w = marker_stats.get("t25_q90_width", np.nan)
    t50_w = marker_stats.get("t50_q90_width", np.nan)
    t75_w = marker_stats.get("t75_q90_width", np.nan)
    dur_w = marker_stats.get("duration_25_75_q90_width", np.nan)
    onset = float(obs_row.get("onset_day", np.nan)) if "onset_day" in obs_row.index else np.nan
    mid = float(obs_row.get("midpoint_day", np.nan)) if "midpoint_day" in obs_row.index else np.nan
    finish = float(obs_row.get("finish_day", np.nan)) if "finish_day" in obs_row.index else np.nan
    if sep not in {"clear_separation", "moderate_separation"}:
        return "transition_not_established", "single_marker_not_recommended", "pre/post separation is not sufficiently clear."
    if "nonmonotonic" in quality and not np.isfinite(t25_w):
        return "nonmonotonic_unresolved", "single_marker_not_recommended", "progress curve is nonmonotonic and markers are not reliable."
    if np.isfinite(onset) and np.isfinite(mid) and np.isfinite(finish):
        early_span = mid - onset
        tail = finish - mid
        if np.isfinite(t25_w) and np.isfinite(t50_w) and np.isfinite(t75_w) and t25_w < min(t50_w, t75_w) and tail > early_span:
            return "early_departure_broad_finish", "marker_family_required", "early progress is more stable than midpoint/finish and finish tail is broader."
        if abs(tail - early_span) <= 3 and np.isfinite(t50_w) and t50_w <= max(t25_w, 1):
            return "compact_transition", "midpoint_or_peak_may_be_usable", "compact transition candidate; midpoint may be representative."
    if "monotonic" in quality:
        return "broad_or_marker_mixed_transition", "marker_family_required", "monotonic but no single marker is clearly sufficient."
    return "nonmonotonic_unresolved", "marker_family_required", "nonmonotonic or marker-mixed transition; keep full process."


def build_field_process_quality(field_curves: pd.DataFrame, v7e_obs: pd.DataFrame, boot_markers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    v7e_w = v7e_obs[(v7e_obs["window_id"].astype(str) == WINDOW_ID) & (v7e_obs["field"].astype(str).isin(FIELDS))].copy()
    bm_w = boot_markers[(boot_markers["window_id"].astype(str) == WINDOW_ID) & (boot_markers["field"].astype(str).isin(FIELDS))]
    for field in FIELDS:
        fc = field_curves[field_curves["field"].astype(str) == field].copy()
        obs_row = None
        ov = v7e_w[v7e_w["field"].astype(str) == field]
        if not ov.empty:
            obs_row = ov.iloc[0]
        bm = bm_w[bm_w["field"].astype(str) == field]
        marker_stat = {}
        for col in ["departure90_day", "departure95_day", "t25_day", "t50_day", "t75_day", "peak_change_day_smooth3", "duration_25_75", "tail_50_75"]:
            if col in bm.columns:
                s = _stats(pd.to_numeric(bm[col], errors="coerce"))
                prefix = col.replace("_day", "")
                marker_stat[f"{prefix}_median"] = s["median"]
                marker_stat[f"{prefix}_q90_width"] = s["q90_width"]
        start_prog = _safe_median(fc[fc["day"] == fc["day"].min()]["progress_bootstrap_median"]) if not fc.empty else np.nan
        anchor_prog = _safe_median(fc[fc["day"] == ANCHOR_DAY]["progress_bootstrap_median"]) if not fc.empty else np.nan
        end_prog = _safe_median(fc[fc["day"] == fc["day"].max()]["progress_bootstrap_median"]) if not fc.empty else np.nan
        shape, suitability, interp = _shape_label_from_row(field, obs_row, marker_stat)
        row = {
            "window_id": WINDOW_ID,
            "anchor_day": ANCHOR_DAY,
            "field": field,
            "pre_post_separation_label": obs_row.get("pre_post_separation_label", "unknown") if obs_row is not None else "unknown",
            "progress_quality_label": obs_row.get("progress_quality_label", "unknown") if obs_row is not None else "unknown",
            "observed_start_progress": float(fc.sort_values("day")["progress_observed"].iloc[0]) if not fc.empty else np.nan,
            "observed_anchor_progress": float(fc[fc["day"] == ANCHOR_DAY]["progress_observed"].iloc[0]) if not fc[fc["day"] == ANCHOR_DAY].empty else np.nan,
            "observed_end_progress": float(fc.sort_values("day")["progress_observed"].iloc[-1]) if not fc.empty else np.nan,
            "bootstrap_median_start_progress": start_prog,
            "bootstrap_median_anchor_progress": anchor_prog,
            "bootstrap_median_end_progress": end_prog,
            "progress_range_observed": float(fc["progress_observed"].max() - fc["progress_observed"].min()) if not fc.empty else np.nan,
            "n_crossings_025": obs_row.get("n_crossings_025", np.nan) if obs_row is not None else np.nan,
            "n_crossings_050": obs_row.get("n_crossings_050", np.nan) if obs_row is not None else np.nan,
            "n_crossings_075": obs_row.get("n_crossings_075", np.nan) if obs_row is not None else np.nan,
            **marker_stat,
            "curve_shape_label": shape,
            "marker_suitability_label": suitability,
            "single_marker_recommended": suitability in {"midpoint_or_peak_may_be_usable"},
            "single_marker_warning": "marker family required; do not reduce field transition to one day" if suitability == "marker_family_required" else "none",
            "field_process_interpretation": interp,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_field_marker_family(obs_markers: pd.DataFrame, boot_markers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    obs_w = obs_markers[(obs_markers["window_id"].astype(str) == WINDOW_ID) & (obs_markers["field"].astype(str).isin(FIELDS))]
    boot_w = boot_markers[(boot_markers["window_id"].astype(str) == WINDOW_ID) & (boot_markers["field"].astype(str).isin(FIELDS))]
    for field in FIELDS:
        osub = obs_w[obs_w["field"].astype(str) == field]
        bsub = boot_w[boot_w["field"].astype(str) == field]
        for col in _marker_cols():
            if col not in bsub.columns:
                continue
            fam, name = _marker_display(col)
            vals = pd.to_numeric(bsub[col], errors="coerce")
            st = _stats(vals)
            obs_val = pd.to_numeric(osub[col], errors="coerce").iloc[0] if (not osub.empty and col in osub.columns) else np.nan
            edge_hit = np.nan
            if "analysis_start" in bsub.columns and col.endswith("_day"):
                starts = pd.to_numeric(bsub["analysis_start"], errors="coerce")
                ends = pd.to_numeric(bsub["analysis_end"], errors="coerce") if "analysis_end" in bsub.columns else np.nan
                edge_hit = float(np.nanmean((vals == starts) | (vals == ends))) if len(vals) else np.nan
            if st["valid_fraction"] < 0.8:
                rel = "not_reliable"
            elif np.isfinite(st["q90_width"]) and st["q90_width"] <= 5:
                rel = "usable"
            elif np.isfinite(st["q90_width"]) and st["q90_width"] <= 10:
                rel = "usable_with_caution"
            else:
                rel = "broad_uncertain"
            rows.append({
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "field": field,
                "marker_family": fam,
                "marker_name": name,
                "marker_column": col,
                "observed_value": obs_val,
                "median": st["median"],
                "q05": st["q05"],
                "q95": st["q95"],
                "q90_width": st["q90_width"],
                "q025": st["q025"],
                "q975": st["q975"],
                "q95_width": st["q95_width"],
                "valid_fraction": st["valid_fraction"],
                "edge_hit_fraction": edge_hit,
                "marker_reliability_label": rel,
                "marker_interpretation": "process summary marker; not the transition process itself",
            })
    return pd.DataFrame(rows)


def _day_relation_label(q05: float, q95: float, med: float, prob_a: float, prob_b: float, near010: float) -> tuple[str, str]:
    if np.isfinite(q05) and q05 > 0:
        return "A_ahead_supported", "supported_90"
    if np.isfinite(q95) and q95 < 0:
        return "B_ahead_supported", "supported_90"
    if np.isfinite(med) and med > 0 and prob_a >= 0.75:
        return "A_ahead_tendency", "tendency"
    if np.isfinite(med) and med < 0 and prob_b >= 0.75:
        return "B_ahead_tendency", "tendency"
    if np.isfinite(near010) and near010 >= 0.50:
        return "near_equal_candidate", "near_equal_candidate"
    return "not_resolved", "unresolved"


def build_pairwise_curve_relation_daily(obs_curves: pd.DataFrame, boot_curves: pd.DataFrame) -> pd.DataFrame:
    obs = obs_curves[(obs_curves["window_id"].astype(str) == WINDOW_ID) & (obs_curves["field"].astype(str).isin(FIELDS))].copy()
    boot = boot_curves[(boot_curves["window_id"].astype(str) == WINDOW_ID) & (boot_curves["field"].astype(str).isin(FIELDS))].copy()
    obs_piv = obs.pivot_table(index="day", columns="field", values="progress", aggfunc="first")
    boot_piv = boot.pivot_table(index=["bootstrap_id", "day"], columns="field", values="progress", aggfunc="first")
    rows = []
    for a, b in combinations(FIELDS, 2):
        if a not in boot_piv.columns or b not in boot_piv.columns:
            continue
        delta_boot = boot_piv[a] - boot_piv[b]
        tmp = delta_boot.reset_index(name="diff")
        for day, sub in tmp.groupby("day", sort=True):
            vals = pd.to_numeric(sub["diff"], errors="coerce").to_numpy(dtype=float)
            med = _safe_median(vals)
            q05 = _safe_quantile(vals, 0.05)
            q95 = _safe_quantile(vals, 0.95)
            q025 = _safe_quantile(vals, 0.025)
            q975 = _safe_quantile(vals, 0.975)
            prob_a = float(np.nanmean(vals > 0)) if vals.size else np.nan
            prob_b = float(np.nanmean(vals < 0)) if vals.size else np.nan
            near = {margin: float(np.nanmean(np.abs(vals) <= margin)) if vals.size else np.nan for margin in PROGRESS_NEAR_MARGINS}
            odiff = np.nan
            if int(day) in obs_piv.index and a in obs_piv.columns and b in obs_piv.columns:
                odiff = float(obs_piv.loc[int(day), a] - obs_piv.loc[int(day), b])
            label, level = _day_relation_label(q05, q95, med, prob_a, prob_b, near.get(0.10, np.nan))
            rows.append({
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "field_a": a,
                "field_b": b,
                "day": int(day),
                "relative_to_anchor": int(day) - ANCHOR_DAY,
                "diff_observed_A_minus_B": odiff,
                "diff_bootstrap_median": med,
                "diff_q05": q05,
                "diff_q95": q95,
                "diff_q025": q025,
                "diff_q975": q975,
                "prob_A_progress_gt_B": prob_a,
                "prob_B_progress_gt_A": prob_b,
                "prob_near_equal_progress_margin_005": near.get(0.05, np.nan),
                "prob_near_equal_progress_margin_010": near.get(0.10, np.nan),
                "prob_near_equal_progress_margin_020": near.get(0.20, np.nan),
                "relation_day_label": label,
                "relation_day_evidence_level": level,
                "progress_difference_interpretation": "A_minus_B is relative pre-to-post progress difference, not physical magnitude or causality.",
            })
    return pd.DataFrame(rows)


def _phase_category(label: str) -> str:
    if str(label).startswith("A_ahead"):
        return "A_ahead"
    if str(label).startswith("B_ahead"):
        return "B_ahead"
    if str(label).startswith("near_equal"):
        return "near_equal"
    return "not_resolved"


def build_pairwise_phase_relation(pair_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (a, b), sub in pair_daily.groupby(["field_a", "field_b"], sort=False):
        ss = sub.sort_values("day").copy()
        ss["phase_category"] = ss["relation_day_label"].map(_phase_category)
        phase_id = 0
        current = None
        group_rows = []
        for _, r in ss.iterrows():
            cat = r["phase_category"]
            if current is None or cat != current:
                if group_rows:
                    phase_id += 1
                    rows.append(_summarize_phase(a, b, phase_id, group_rows))
                current = cat
                group_rows = [r]
            else:
                group_rows.append(r)
        if group_rows:
            phase_id += 1
            rows.append(_summarize_phase(a, b, phase_id, group_rows))
    return pd.DataFrame(rows)


def _summarize_phase(a: str, b: str, phase_id: int, group_rows: list[pd.Series]) -> dict:
    df = pd.DataFrame(group_rows)
    cat = _phase_category(str(df["relation_day_label"].iloc[0]))
    label_map = {
        "A_ahead": "A_ahead_phase",
        "B_ahead": "B_ahead_phase",
        "near_equal": "near_equal_phase_candidate",
        "not_resolved": "not_resolved_phase",
    }
    if cat == "A_ahead":
        interp = f"{a} has higher relative transition progress than {b} in this day segment."
    elif cat == "B_ahead":
        interp = f"{b} has higher relative transition progress than {a} in this day segment."
    elif cat == "near_equal":
        interp = "Near-equal progress candidate; this is not an equivalence conclusion without a chosen margin."
    else:
        interp = "No resolved curve-level progress relation in this day segment."
    return {
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "field_a": a,
        "field_b": b,
        "phase_id": phase_id,
        "day_start": int(df["day"].min()),
        "day_end": int(df["day"].max()),
        "n_days": int(len(df)),
        "phase_relation_label": label_map[cat],
        "phase_evidence_level": str(df["relation_day_evidence_level"].mode().iloc[0]) if not df["relation_day_evidence_level"].mode().empty else "unknown",
        "mean_diff_observed": float(pd.to_numeric(df["diff_observed_A_minus_B"], errors="coerce").mean()),
        "median_diff_bootstrap": float(pd.to_numeric(df["diff_bootstrap_median"], errors="coerce").median()),
        "mean_prob_A_gt_B": float(pd.to_numeric(df["prob_A_progress_gt_B"], errors="coerce").mean()),
        "mean_prob_B_gt_A": float(pd.to_numeric(df["prob_B_progress_gt_A"], errors="coerce").mean()),
        "phase_crossing_flag": False,
        "phase_interpretation": interp,
    }


def build_pairwise_marker_family_relation(boot_markers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bm = boot_markers[(boot_markers["window_id"].astype(str) == WINDOW_ID) & (boot_markers["field"].astype(str).isin(FIELDS))].copy()
    for col in _marker_cols():
        if col not in bm.columns:
            continue
        fam, name = _marker_display(col)
        piv = bm.pivot_table(index="bootstrap_id", columns="field", values=col, aggfunc="first")
        for a, b in combinations(FIELDS, 2):
            if a not in piv.columns or b not in piv.columns:
                continue
            av = pd.to_numeric(piv[a], errors="coerce")
            bv = pd.to_numeric(piv[b], errors="coerce")
            good = av.notna() & bv.notna()
            delta = (bv[good] - av[good]).to_numpy(dtype=float)
            if delta.size == 0:
                continue
            med = _safe_median(delta)
            q05 = _safe_quantile(delta, 0.05)
            q95 = _safe_quantile(delta, 0.95)
            q025 = _safe_quantile(delta, 0.025)
            q975 = _safe_quantile(delta, 0.975)
            pass90_a = bool(q05 > 0)  # B-A > 0 => A earlier/smaller.
            pass90_b = bool(q95 < 0)
            pass95_a = bool(q025 > 0)
            pass95_b = bool(q975 < 0)
            if pass95_a or pass90_a:
                rel = "A_leads" if fam not in {"duration", "tail", "early_span"} else "A_smaller_or_shorter"
            elif pass95_b or pass90_b:
                rel = "B_leads" if fam not in {"duration", "tail", "early_span"} else "B_smaller_or_shorter"
            elif med > 0:
                rel = "A_tendency" if fam not in {"duration", "tail", "early_span"} else "A_smaller_tendency"
            elif med < 0:
                rel = "B_tendency" if fam not in {"duration", "tail", "early_span"} else "B_smaller_tendency"
            else:
                rel = "not_resolved"
            rows.append({
                "window_id": WINDOW_ID,
                "anchor_day": ANCHOR_DAY,
                "field_a": a,
                "field_b": b,
                "marker_family": fam,
                "marker_name": name,
                "marker_column": col,
                "median_delta_B_minus_A": med,
                "q05_delta": q05,
                "q95_delta": q95,
                "q025_delta": q025,
                "q975_delta": q975,
                "prob_A_earlier_or_smaller": float(np.nanmean(delta > 0)),
                "prob_B_earlier_or_smaller": float(np.nanmean(delta < 0)),
                "prob_same_day_or_equal": float(np.nanmean(np.isclose(delta, 0))),
                "pass90_A_leads_or_smaller": pass90_a,
                "pass95_A_leads_or_smaller": pass95_a,
                "pass90_B_leads_or_smaller": pass90_b,
                "pass95_B_leads_or_smaller": pass95_b,
                "minimum_equivalence_margin_90": max(abs(q05), abs(q95)) if np.isfinite(q05) and np.isfinite(q95) else np.nan,
                "minimum_equivalence_margin_95": max(abs(q025), abs(q975)) if np.isfinite(q025) and np.isfinite(q975) else np.nan,
                "marker_relation_label": rel,
                "marker_relation_interpretation": "For timing markers, positive delta means field_a earlier than field_b. For duration/tail markers, positive delta means field_a is smaller/shorter.",
            })
    return pd.DataFrame(rows)


def _marker_summary_for_pair(mrel: pd.DataFrame, a: str, b: str, families: set[str] | None = None, names: set[str] | None = None) -> dict:
    sub = mrel[(mrel["field_a"] == a) & (mrel["field_b"] == b)].copy()
    if families is not None:
        sub = sub[sub["marker_family"].isin(families)]
    if names is not None:
        sub = sub[sub["marker_name"].isin(names)]
    if sub.empty:
        return {"labels": "none", "n_A_pass90": 0, "n_B_pass90": 0, "n_A_tendency": 0, "n_B_tendency": 0}
    labels = sub["marker_relation_label"].astype(str).tolist()
    return {
        "labels": ";".join(labels),
        "n_A_pass90": int(sub["pass90_A_leads_or_smaller"].sum()),
        "n_B_pass90": int(sub["pass90_B_leads_or_smaller"].sum()),
        "n_A_tendency": int(sum(l.startswith("A_tendency") or l == "A_leads" for l in labels)),
        "n_B_tendency": int(sum(l.startswith("B_tendency") or l == "B_leads" for l in labels)),
    }


def build_pairwise_relation_type(pair_daily: pd.DataFrame, phases: pd.DataFrame, mrel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(FIELDS, 2):
        ph = phases[(phases["field_a"] == a) & (phases["field_b"] == b)]
        phase_labels = ph["phase_relation_label"].astype(str).tolist() if not ph.empty else []
        has_a_phase = any(l == "A_ahead_phase" for l in phase_labels)
        has_b_phase = any(l == "B_ahead_phase" for l in phase_labels)
        has_cross = has_a_phase and has_b_phase
        dep = _marker_summary_for_pair(mrel, a, b, {"departure"})
        early = _marker_summary_for_pair(mrel, a, b, {"threshold"}, {"t10", "t15", "t20", "t25", "t30", "t35", "t40", "t45"})
        midpoint = _marker_summary_for_pair(mrel, a, b, {"threshold"}, {"t50"})
        finish = _marker_summary_for_pair(mrel, a, b, {"threshold"}, {"t75"})
        peak = _marker_summary_for_pair(mrel, a, b, {"peak_change"})
        dur = _marker_summary_for_pair(mrel, a, b, {"duration", "tail"})
        a_evidence = dep["n_A_pass90"] + early["n_A_pass90"] + midpoint["n_A_pass90"] + peak["n_A_pass90"]
        b_evidence = dep["n_B_pass90"] + early["n_B_pass90"] + midpoint["n_B_pass90"] + peak["n_B_pass90"]
        a_tend = dep["n_A_tendency"] + early["n_A_tendency"] + midpoint["n_A_tendency"] + peak["n_A_tendency"]
        b_tend = dep["n_B_tendency"] + early["n_B_tendency"] + midpoint["n_B_tendency"] + peak["n_B_tendency"]
        if has_cross:
            if {a, b} == {"H", "Jw"}:
                rtype = "front_loaded_vs_catchup"
                interp = "Curve-level evidence suggests phase-dependent relation: early-progress advantage and later catch-up; do not reduce to a single lead."
            else:
                rtype = "phase_crossing_relation"
                interp = "Curve relation changes sign across phases; single order is not adequate."
        elif a_evidence >= 2 and b_evidence == 0:
            rtype = "unidirectional_lead"
            interp = f"Multiple marker families support {a} before {b}; still interpret as transition-progress order, not causality."
        elif b_evidence >= 2 and a_evidence == 0:
            rtype = "unidirectional_lead"
            interp = f"Multiple marker families support {b} before {a}; still interpret as transition-progress order, not causality."
        elif (a_evidence + b_evidence) == 0 and (a_tend > 0 or b_tend > 0):
            rtype = "weak_lead_tendency"
            interp = "Only tendency-level marker support; no 90% directional order."
        elif dep["n_A_pass90"] == 0 and dep["n_B_pass90"] == 0 and "near_equal_phase_candidate" in phase_labels:
            rtype = "same_phase_candidate"
            interp = "Near-same phase candidate; requires equivalence margin before saying synchronous."
        elif a_evidence > 0 and b_evidence > 0:
            rtype = "marker_inconsistent_relation"
            interp = "Different marker families support conflicting directions; do not force an order."
        else:
            rtype = "order_not_resolved"
            interp = "No stable directional or phase relation resolved by current implementation layer."
        rows.append({
            "window_id": WINDOW_ID,
            "anchor_day": ANCHOR_DAY,
            "field_a": a,
            "field_b": b,
            "departure_relation": dep["labels"],
            "early_progress_relation": early["labels"],
            "midpoint_relation": midpoint["labels"],
            "finish_relation": finish["labels"],
            "peak_change_relation": peak["labels"],
            "duration_tail_relation": dur["labels"],
            "curve_phase_relation": ";".join(phase_labels) if phase_labels else "none",
            "relation_type": rtype,
            "relation_confidence": "implementation_layer_relation_type_not_final_statistical_conclusion",
            "dominant_evidence_layer": "curve_phase" if has_cross else ("marker_family" if a_evidence + b_evidence > 0 else "unresolved"),
            "final_relation_interpretation": interp,
            "do_not_overinterpret": "No causality, no physical strength comparison, no synchrony without equivalence margin.",
        })
    return pd.DataFrame(rows)


def build_allfield_organization_layers(pair_types: pd.DataFrame, mrel: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    layer_specs = [
        ("departure_layer", "departure", {"departure"}),
        ("early_progress_layer", "threshold_sweep_early", {"threshold"}),
        ("peak_change_layer", "peak_change", {"peak_change"}),
        ("midpoint_layer", "t50_midpoint", {"threshold"}),
        ("finish_tail_layer", "t75_duration_tail", {"duration", "tail"}),
    ]
    for lname, ltype, fams in layer_specs:
        sub = mrel[mrel["marker_family"].isin(fams)].copy()
        if lname == "early_progress_layer":
            sub = sub[sub["marker_name"].isin({"t10", "t15", "t20", "t25", "t30", "t35", "t40", "t45"})]
        if lname == "midpoint_layer":
            sub = sub[sub["marker_name"] == "t50"]
        if lname == "finish_tail_layer":
            # Include t75 as completion marker plus duration/tail.
            sub2 = mrel[(mrel["marker_family"] == "threshold") & (mrel["marker_name"] == "t75")]
            sub = pd.concat([sub, sub2], ignore_index=True)
        pass_rows = sub[(sub["pass90_A_leads_or_smaller"] == True) | (sub["pass90_B_leads_or_smaller"] == True)] if not sub.empty else pd.DataFrame()
        tendency_rows = sub[sub["marker_relation_label"].astype(str).str.contains("tendency", na=False)] if not sub.empty else pd.DataFrame()
        rows.append({
            "window_id": WINDOW_ID,
            "anchor_day": ANCHOR_DAY,
            "layer_name": lname,
            "layer_type": ltype,
            "n_pair_marker_rows": int(len(sub)),
            "n_pass90_relations": int(len(pass_rows)),
            "n_tendency_relations": int(len(tendency_rows)),
            "organization_label": "has_confirmed_relations" if len(pass_rows) else ("tendency_only" if len(tendency_rows) else "not_resolved"),
            "evidence_summary": "; ".join((pass_rows["field_a"] + "-" + pass_rows["field_b"] + ":" + pass_rows["marker_name"] + ":" + pass_rows["marker_relation_label"]).astype(str).head(20).tolist()) if len(pass_rows) else "no pass90 relations in this layer",
            "confidence_level": "implementation_layer_summary",
            "interpretation": "Layer summary only; detailed pairwise relation table controls interpretation.",
        })
    # Curve phase layer from relation type table.
    phase_cross = pair_types[pair_types["relation_type"].isin(["phase_crossing_relation", "front_loaded_vs_catchup"])]
    rows.append({
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "layer_name": "curve_phase_layer",
        "layer_type": "daily_progress_phase_relation",
        "n_pair_marker_rows": int(len(pair_types)),
        "n_pass90_relations": int(len(phase_cross)),
        "n_tendency_relations": int((pair_types["relation_type"] == "weak_lead_tendency").sum()),
        "organization_label": "has_phase_crossing_or_catchup" if len(phase_cross) else "no_phase_crossing_detected",
        "evidence_summary": "; ".join((phase_cross["field_a"] + "-" + phase_cross["field_b"] + ":" + phase_cross["relation_type"]).astype(str).tolist()) if len(phase_cross) else "no phase crossing relation classified",
        "confidence_level": "implementation_layer_summary",
        "interpretation": "Curve phase layer describes process relation, not causal order.",
    })
    return pd.DataFrame(rows)


def build_relation_complexity(pair_daily: pd.DataFrame, phases: pd.DataFrame, mrel: pd.DataFrame, pair_types: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pair_types.iterrows():
        a, b = r["field_a"], r["field_b"]
        ph = phases[(phases["field_a"] == a) & (phases["field_b"] == b)]
        labels = ph["phase_relation_label"].astype(str).tolist() if not ph.empty else []
        has_a = any(x == "A_ahead_phase" for x in labels)
        has_b = any(x == "B_ahead_phase" for x in labels)
        mr = mrel[(mrel["field_a"] == a) & (mrel["field_b"] == b)]
        has_marker_conflict = bool((mr["pass90_A_leads_or_smaller"].sum() > 0) and (mr["pass90_B_leads_or_smaller"].sum() > 0)) if not mr.empty else False
        has_near = any(x == "near_equal_phase_candidate" for x in labels)
        has_directional = bool((mr["pass90_A_leads_or_smaller"].sum() + mr["pass90_B_leads_or_smaller"].sum()) > 0) if not mr.empty else False
        if has_a and has_b:
            comp = "phase_crossing"
        elif has_marker_conflict:
            comp = "marker_conflict"
        elif r["relation_type"] == "unidirectional_lead":
            comp = "simple_order"
        elif r["relation_type"] == "weak_lead_tendency":
            comp = "weak_order"
        elif has_near:
            comp = "same_phase_unresolved"
        else:
            comp = "highly_complex_unresolved" if len(ph) > 4 else "order_not_resolved"
        rows.append({
            "window_id": WINDOW_ID,
            "anchor_day": ANCHOR_DAY,
            "field_a": a,
            "field_b": b,
            "n_relation_phases": int(len(ph)),
            "has_crossing": bool(has_a and has_b),
            "has_marker_conflict": has_marker_conflict,
            "has_near_equivalence_candidate": bool(has_near),
            "has_directional_lead": has_directional,
            "has_curve_phase_reversal": bool(has_a and has_b),
            "complexity_label": comp,
            "complexity_interpretation": "Complexity is part of the implementation-layer result; do not force a clean order if labels are complex.",
        })
    return pd.DataFrame(rows)


def _make_figures(paths: ProcessRelationPaths, field_curves: pd.DataFrame, pair_daily: pd.DataFrame, pair_types: pd.DataFrame) -> None:
    _ensure_dir(paths.figure_dir)
    try:
        import matplotlib.pyplot as plt
        # Field progress curves.
        fig, ax = plt.subplots(figsize=(10, 6))
        for field in FIELDS:
            sub = field_curves[field_curves["field"].astype(str) == field].sort_values("day")
            ax.plot(sub["day"], sub["progress_observed"], label=field)
        ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
        ax.set_xlabel("Day index")
        ax.set_ylabel("Observed relative pre-to-post progress")
        ax.set_title("W45 field progress curves (V7-n)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_field_progress_curves_v7_n.png", dpi=180)
        plt.close(fig)

        # Pairwise heatmap.
        pvt = pair_daily.pivot_table(index=["field_a", "field_b"], columns="day", values="diff_bootstrap_median", aggfunc="first")
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pvt.to_numpy(dtype=float), aspect="auto")
        ax.set_yticks(np.arange(len(pvt.index)))
        ax.set_yticklabels([f"{a}-{b}" for a, b in pvt.index])
        ax.set_xticks(np.arange(len(pvt.columns))[:: max(1, len(pvt.columns)//10)])
        ax.set_xticklabels([str(int(x)) for x in pvt.columns[:: max(1, len(pvt.columns)//10)]])
        ax.set_xlabel("Day index")
        ax.set_title("W45 pairwise median progress difference A-B")
        fig.colorbar(im, ax=ax, label="median progress_A - progress_B")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_pairwise_curve_diff_heatmap_v7_n.png", dpi=180)
        plt.close(fig)

        # H-Jw detail.
        hj = pair_daily[((pair_daily["field_a"] == "H") & (pair_daily["field_b"] == "Jw")) | ((pair_daily["field_a"] == "Jw") & (pair_daily["field_b"] == "H"))].copy()
        if not hj.empty:
            # Ensure difference is H-Jw.
            if hj["field_a"].iloc[0] == "Jw":
                for c in ["diff_observed_A_minus_B", "diff_bootstrap_median", "diff_q05", "diff_q95"]:
                    hj[c] = -pd.to_numeric(hj[c], errors="coerce")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hj["day"], hj["diff_observed_A_minus_B"], label="observed H-Jw progress")
            ax.plot(hj["day"], hj["diff_bootstrap_median"], label="bootstrap median H-Jw")
            ax.fill_between(hj["day"].to_numpy(dtype=float), hj["diff_q05"].to_numpy(dtype=float), hj["diff_q95"].to_numpy(dtype=float), alpha=0.2)
            ax.axhline(0, linewidth=1)
            ax.axvline(ANCHOR_DAY, linestyle="--", linewidth=1)
            ax.set_xlabel("Day index")
            ax.set_ylabel("H - Jw relative progress")
            ax.set_title("W45 H/Jw curve relation (V7-n)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(paths.figure_dir / "w45_H_Jw_progress_relation_v7_n.png", dpi=180)
            plt.close(fig)

        # Relation type counts.
        fig, ax = plt.subplots(figsize=(10, 4))
        counts = pair_types["relation_type"].value_counts()
        counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Pair count")
        ax.set_title("W45 relation type counts (V7-n)")
        fig.tight_layout()
        fig.savefig(paths.figure_dir / "w45_allfield_relation_layers_v7_n.png", dpi=180)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        _write_text(f"Figure generation failed: {exc}\n", paths.log_dir / "figure_generation_error_v7_n.txt")


def _write_summary(paths: ProcessRelationPaths, audit: dict, field_quality: pd.DataFrame, pair_types: pd.DataFrame, org_layers: pd.DataFrame, complexity: pd.DataFrame) -> None:
    hq = field_quality[field_quality["field"] == "H"]
    jwq = field_quality[field_quality["field"] == "Jw"]
    hj = pair_types[((pair_types["field_a"] == "H") & (pair_types["field_b"] == "Jw")) | ((pair_types["field_a"] == "Jw") & (pair_types["field_b"] == "H"))]
    lines = []
    lines.append("# W45 all-field process relation layer V7-n")
    lines.append("")
    lines.append("## Purpose")
    lines.append("This is an implementation-layer audit. It does not try to force W45 into a clean field-order chain. It represents each field as a pre-to-post progress process and each pair as a curve/phase/marker relation object.")
    lines.append("")
    lines.append("## Input representation")
    lines.append(f"- Input representation: {audit.get('input_representation')}")
    lines.append("- Fields: P, V, H, Je, Jw. All five fields participate in calculations.")
    lines.append("- Progress is relative pre-to-post progress, not physical magnitude and not causality.")
    lines.append("")
    lines.append("## Field process quality")
    for _, r in field_quality.iterrows():
        lines.append(f"- {r['field']}: {r['curve_shape_label']} / {r['marker_suitability_label']} — {r['field_process_interpretation']}")
    lines.append("")
    lines.append("## Pairwise relation types")
    for _, r in pair_types.iterrows():
        lines.append(f"- {r['field_a']}-{r['field_b']}: {r['relation_type']} — {r['final_relation_interpretation']}")
    lines.append("")
    lines.append("## H/Jw focus")
    if not hj.empty:
        r = hj.iloc[0]
        lines.append(f"H/Jw relation type: {r['relation_type']}")
        lines.append(f"Interpretation: {r['final_relation_interpretation']}")
        lines.append("Use `w45_pairwise_curve_relation_daily_v7_n.csv` and `w45_pairwise_phase_relation_v7_n.csv` to inspect the day-by-day and phase-level evidence. This summary must not replace those tables.")
    else:
        lines.append("H/Jw row not found in pairwise relation type table.")
    lines.append("")
    lines.append("## Organization layers")
    for _, r in org_layers.iterrows():
        lines.append(f"- {r['layer_name']}: {r['organization_label']} — {r['evidence_summary']}")
    lines.append("")
    lines.append("## Relation complexity")
    counts = complexity["complexity_label"].value_counts().to_dict() if not complexity.empty else {}
    lines.append(f"Complexity counts: {counts}")
    lines.append("A complex or mixed result is not an implementation failure. It means W45 should not be forced into a clean sequence unless later evidence supports it.")
    lines.append("")
    lines.append("## Prohibited interpretations")
    lines.append("- Do not interpret progress difference as physical strength difference.")
    lines.append("- Do not interpret progress relation as causality.")
    lines.append("- Do not convert not_resolved into synchrony; near-equivalence needs an explicit margin.")
    lines.append("- Do not hide any of P/V/H/Je/Jw from computation.")
    lines.append("- Do not force phase-crossing relations into one lead/lag edge.")
    _write_text("\n".join(lines) + "\n", paths.output_dir / "w45_allfield_organization_summary_v7_n.md")


def run_w45_allfield_process_relation_layer_v7_n(v7_root: Path | None = None) -> dict:
    paths = _resolve_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)
    _ensure_dir(paths.figure_dir)

    obs_curves, boot_curves, obs_markers, boot_markers, v7e_obs = _load_base_tables(paths)
    audit = _validate_inputs(obs_curves, boot_curves, obs_markers, boot_markers, paths)
    _write_json(audit, paths.output_dir / "input_audit_v7_n.json")

    print("[V7-n] building field transition curves")
    field_curves = build_field_transition_curves(obs_curves, boot_curves)
    _write_csv(field_curves, paths.output_dir / "w45_field_transition_curves_v7_n.csv")

    print("[V7-n] auditing field process quality")
    field_quality = build_field_process_quality(field_curves, v7e_obs, boot_markers)
    _write_csv(field_quality, paths.output_dir / "w45_field_process_quality_v7_n.csv")

    print("[V7-n] summarizing marker families")
    field_markers = build_field_marker_family(obs_markers, boot_markers)
    _write_csv(field_markers, paths.output_dir / "w45_field_marker_family_v7_n.csv")

    print("[V7-n] building pairwise daily curve relations")
    pair_daily = build_pairwise_curve_relation_daily(obs_curves, boot_curves)
    _write_csv(pair_daily, paths.output_dir / "w45_pairwise_curve_relation_daily_v7_n.csv")

    print("[V7-n] segmenting pairwise phases")
    phases = build_pairwise_phase_relation(pair_daily)
    _write_csv(phases, paths.output_dir / "w45_pairwise_phase_relation_v7_n.csv")

    print("[V7-n] building pairwise marker-family relations")
    marker_rel = build_pairwise_marker_family_relation(boot_markers)
    _write_csv(marker_rel, paths.output_dir / "w45_pairwise_marker_family_relation_v7_n.csv")

    print("[V7-n] classifying pairwise relation types")
    pair_types = build_pairwise_relation_type(pair_daily, phases, marker_rel)
    _write_csv(pair_types, paths.output_dir / "w45_pairwise_relation_type_v7_n.csv")

    print("[V7-n] building organization layers")
    org_layers = build_allfield_organization_layers(pair_types, marker_rel, phases)
    _write_csv(org_layers, paths.output_dir / "w45_allfield_organization_layers_v7_n.csv")

    print("[V7-n] building relation complexity audit")
    complexity = build_relation_complexity(pair_daily, phases, marker_rel, pair_types)
    _write_csv(complexity, paths.output_dir / "w45_relation_complexity_audit_v7_n.csv")

    _make_figures(paths, field_curves, pair_daily, pair_types)
    _write_summary(paths, audit, field_quality, pair_types, org_layers, complexity)

    run_meta = {
        "status": "success",
        "created_at": _now_iso(),
        "output_tag": OUTPUT_TAG,
        "window_id": WINDOW_ID,
        "anchor_day": ANCHOR_DAY,
        "fields": FIELDS,
        "input_representation": "current_v7e_progress_profile_via_v7m_curves",
        "does_not_change_spatial_unit": True,
        "does_not_switch_to_raw025": True,
        "does_not_change_stage_window": True,
        "does_not_infer_causality": True,
        "outputs": {
            "field_curves": "w45_field_transition_curves_v7_n.csv",
            "field_quality": "w45_field_process_quality_v7_n.csv",
            "field_marker_family": "w45_field_marker_family_v7_n.csv",
            "pairwise_daily": "w45_pairwise_curve_relation_daily_v7_n.csv",
            "pairwise_phase": "w45_pairwise_phase_relation_v7_n.csv",
            "pairwise_marker": "w45_pairwise_marker_family_relation_v7_n.csv",
            "pairwise_type": "w45_pairwise_relation_type_v7_n.csv",
            "organization_layers": "w45_allfield_organization_layers_v7_n.csv",
            "complexity": "w45_relation_complexity_audit_v7_n.csv",
        },
    }
    _write_json(run_meta, paths.output_dir / "run_meta.json")
    _write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), paths.log_dir / "run_meta_v7_n.json")
    print(f"[V7-n] done: {paths.output_dir}")
    return run_meta


if __name__ == "__main__":
    run_w45_allfield_process_relation_layer_v7_n()
