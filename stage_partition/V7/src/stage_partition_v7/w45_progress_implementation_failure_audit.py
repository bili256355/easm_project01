from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

FIELDS: List[str] = ["P", "V", "H", "Je", "Jw"]
W45_ID = "W002"
W45_ANCHOR = 45

ACCEPTABLE_SEPARATION = {"clear_separation", "moderate_separation"}
WEAK_SEPARATION = {"weak_separation"}
INVALID_SEPARATION = {"no_clear_separation"}

USABLE_PROGRESS = {"monotonic_clear_progress"}
CAUTION_PROGRESS = {"monotonic_broad_progress", "partial_progress", "boundary_limited_progress"}
LIMITED_PROGRESS = {"nonmonotonic_progress"}
INVALID_PROGRESS = {"no_clear_prepost_separation"}

FOCUS_H_PAIRS = [("H", "P"), ("H", "V"), ("H", "Je"), ("H", "Jw")]
AUXILIARY_PAIRS = [("P", "Je"), ("V", "Je"), ("Jw", "Je"), ("P", "Jw")]
TAIL_AUDIT_PAIRS = FOCUS_H_PAIRS + AUXILIARY_PAIRS


@dataclass
class W45AuditPaths:
    v7_root: Path
    v7e_output_dir: Path
    e1_output_dir: Path
    e2_output_dir: Path
    output_dir: Path
    log_dir: Path


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path, required: bool = False) -> dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_columns(df: pd.DataFrame, required_cols: Iterable[str], table_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
        return out
    except Exception:
        return default


def _get_paths(v7_root: Optional[Path]) -> W45AuditPaths:
    if v7_root is None:
        # stage_partition/V7/src/stage_partition_v7/w45_progress_implementation_failure_audit.py
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    return W45AuditPaths(
        v7_root=v7_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        e1_output_dir=v7_root / "outputs" / "progress_order_significance_audit_v7_e1",
        e2_output_dir=v7_root / "outputs" / "progress_order_failure_audit_v7_e2",
        output_dir=v7_root / "outputs" / "w45_progress_implementation_failure_audit_v7_e3",
        log_dir=v7_root / "logs" / "w45_progress_implementation_failure_audit_v7_e3",
    )


def _load_inputs(paths: W45AuditPaths) -> dict:
    observed = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv")
    bootstrap_samples = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_samples_v7_e.csv")
    bootstrap_summary = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_summary_v7_e.csv")
    loyo_samples = _read_csv(paths.v7e_output_dir / "field_transition_progress_loyo_samples_v7_e.csv", required=False)
    loyo_summary = _read_csv(paths.v7e_output_dir / "field_transition_progress_loyo_summary_v7_e.csv", required=False)
    observed_curves = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_curves_long_v7_e.csv", required=False)
    e1_pairwise = _read_csv(paths.e1_output_dir / "pairwise_progress_delta_test_v7_e1.csv")
    e2_pairwise = _read_csv(paths.e2_output_dir / "pairwise_progress_failure_audit_v7_e2.csv")
    e2_window = _read_csv(paths.e2_output_dir / "window_progress_failure_audit_v7_e2.csv", required=False)
    v7e_meta = _read_json(paths.v7e_output_dir / "run_meta.json", required=False)
    e1_meta = _read_json(paths.e1_output_dir / "run_meta.json", required=False)
    e2_meta = _read_json(paths.e2_output_dir / "run_meta.json", required=False)

    _require_columns(
        observed,
        [
            "window_id",
            "field",
            "pre_post_separation_label",
            "progress_quality_label",
            "onset_day",
            "midpoint_day",
            "finish_day",
            "duration",
        ],
        "field_transition_progress_observed_v7_e.csv",
    )
    _require_columns(
        bootstrap_samples,
        ["window_id", "field", "bootstrap_id", "onset_day", "midpoint_day", "finish_day", "duration"],
        "field_transition_progress_bootstrap_samples_v7_e.csv",
    )
    _require_columns(
        bootstrap_summary,
        ["window_id", "field", "midpoint_median", "midpoint_q025", "midpoint_q975", "midpoint_iqr"],
        "field_transition_progress_bootstrap_summary_v7_e.csv",
    )
    _require_columns(
        e1_pairwise,
        ["window_id", "field_a", "field_b", "median_delta_b_minus_a", "final_evidence_label"],
        "pairwise_progress_delta_test_v7_e1.csv",
    )
    _require_columns(
        e2_pairwise,
        ["window_id", "field_a", "field_b", "pass_90", "primary_failure_type"],
        "pairwise_progress_failure_audit_v7_e2.csv",
    )

    return {
        "observed": observed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_summary": bootstrap_summary,
        "loyo_samples": loyo_samples,
        "loyo_summary": loyo_summary,
        "observed_curves": observed_curves,
        "e1_pairwise": e1_pairwise,
        "e2_pairwise": e2_pairwise,
        "e2_window": e2_window,
        "v7e_meta": v7e_meta,
        "e1_meta": e1_meta,
        "e2_meta": e2_meta,
    }


def validate_w45_inputs(inputs: dict, paths: W45AuditPaths) -> dict:
    audit = {
        "created_at": _now_iso(),
        "window_id": W45_ID,
        "anchor_day": W45_ANCHOR,
        "inputs": {
            "v7e_output_dir": str(paths.v7e_output_dir),
            "e1_output_dir": str(paths.e1_output_dir),
            "e2_output_dir": str(paths.e2_output_dir),
        },
        "checks": {},
        "notes": [],
    }
    observed = inputs["observed"]
    bootstrap_samples = inputs["bootstrap_samples"]
    bootstrap_summary = inputs["bootstrap_summary"]
    e1_pairwise = inputs["e1_pairwise"]
    e2_pairwise = inputs["e2_pairwise"]
    observed_curves = inputs["observed_curves"]

    for name, df in [
        ("observed", observed),
        ("bootstrap_samples", bootstrap_samples),
        ("bootstrap_summary", bootstrap_summary),
        ("e1_pairwise", e1_pairwise),
        ("e2_pairwise", e2_pairwise),
    ]:
        audit["checks"][f"{name}_has_w45"] = bool((df["window_id"].astype(str) == W45_ID).any())

    obs_fields = set(observed.loc[observed["window_id"].astype(str) == W45_ID, "field"].astype(str))
    boot_fields = set(bootstrap_samples.loc[bootstrap_samples["window_id"].astype(str) == W45_ID, "field"].astype(str))
    summary_fields = set(bootstrap_summary.loc[bootstrap_summary["window_id"].astype(str) == W45_ID, "field"].astype(str))
    audit["checks"]["observed_fields_complete"] = sorted(obs_fields) == sorted(FIELDS)
    audit["checks"]["bootstrap_fields_complete"] = sorted(boot_fields) == sorted(FIELDS)
    audit["checks"]["bootstrap_summary_fields_complete"] = sorted(summary_fields) == sorted(FIELDS)

    e1_w45_pairs = e1_pairwise.loc[e1_pairwise["window_id"].astype(str) == W45_ID]
    e2_w45_pairs = e2_pairwise.loc[e2_pairwise["window_id"].astype(str) == W45_ID]
    audit["checks"]["e1_w45_pair_count"] = int(len(e1_w45_pairs))
    audit["checks"]["e2_w45_pair_count"] = int(len(e2_w45_pairs))
    audit["checks"]["e1_w45_has_10_pairs"] = int(len(e1_w45_pairs)) == 10
    audit["checks"]["e2_w45_has_10_pairs"] = int(len(e2_w45_pairs)) == 10
    audit["checks"]["observed_curves_available"] = not observed_curves.empty

    failed = [k for k, v in audit["checks"].items() if isinstance(v, bool) and not v]
    if failed:
        audit["notes"].append(f"Some checks failed: {failed}")
    if observed_curves.empty:
        audit["notes"].append("field_transition_progress_observed_curves_long_v7_e.csv is unavailable; curve-shape details are limited to observed summary columns.")
    return audit


def _w45(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "window_id" not in df.columns:
        return pd.DataFrame()
    return df.loc[df["window_id"].astype(str) == W45_ID].copy()


def _field_lookup(df: pd.DataFrame) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if df.empty:
        return out
    for _, row in df.iterrows():
        out[str(row["field"])] = row.to_dict()
    return out


def _progress_validity_label(progress_quality: str) -> str:
    if progress_quality in USABLE_PROGRESS:
        return "usable"
    if progress_quality in CAUTION_PROGRESS:
        return "usable_with_caution"
    if progress_quality in LIMITED_PROGRESS:
        return "limited"
    if progress_quality in INVALID_PROGRESS:
        return "invalid"
    return "unknown"


def _prepost_validity_label(separation_label: str) -> str:
    if separation_label in ACCEPTABLE_SEPARATION:
        return "usable"
    if separation_label in WEAK_SEPARATION:
        return "weak_with_caution"
    if separation_label in INVALID_SEPARATION:
        return "invalid"
    return "unknown"


def build_implementation_validity(inputs: dict) -> pd.DataFrame:
    obs = _w45(inputs["observed"])
    bs = _w45(inputs["bootstrap_samples"])
    bsum = _w45(inputs["bootstrap_summary"])
    obs_lookup = _field_lookup(obs)
    bsum_lookup = _field_lookup(bsum)

    # Compute per-field bootstrap quantiles that are not necessarily in V7-e summary.
    boot_stats: Dict[str, dict] = {}
    for field, sub in bs.groupby("field"):
        vals_mid = pd.to_numeric(sub["midpoint_day"], errors="coerce").dropna().to_numpy(float)
        vals_on = pd.to_numeric(sub["onset_day"], errors="coerce").dropna().to_numpy(float)
        vals_fin = pd.to_numeric(sub["finish_day"], errors="coerce").dropna().to_numpy(float)
        vals_dur = pd.to_numeric(sub["duration"], errors="coerce").dropna().to_numpy(float)
        def q(vals, p):
            return float(np.nanpercentile(vals, p)) if vals.size else np.nan
        boot_stats[str(field)] = {
            "bootstrap_onset_median": q(vals_on, 50),
            "bootstrap_onset_q25": q(vals_on, 25),
            "bootstrap_onset_q75": q(vals_on, 75),
            "bootstrap_midpoint_median": q(vals_mid, 50),
            "bootstrap_midpoint_q05": q(vals_mid, 5),
            "bootstrap_midpoint_q25": q(vals_mid, 25),
            "bootstrap_midpoint_q75": q(vals_mid, 75),
            "bootstrap_midpoint_q95": q(vals_mid, 95),
            "bootstrap_midpoint_iqr": q(vals_mid, 75) - q(vals_mid, 25) if vals_mid.size else np.nan,
            "bootstrap_midpoint_q90_width": q(vals_mid, 95) - q(vals_mid, 5) if vals_mid.size else np.nan,
            "bootstrap_finish_median": q(vals_fin, 50),
            "bootstrap_finish_q25": q(vals_fin, 25),
            "bootstrap_finish_q75": q(vals_fin, 75),
            "bootstrap_duration_median": q(vals_dur, 50),
            "bootstrap_duration_q25": q(vals_dur, 25),
            "bootstrap_duration_q75": q(vals_dur, 75),
        }

    base_rows: List[dict] = []
    for field in FIELDS:
        obs_row = obs_lookup.get(field, {})
        sum_row = bsum_lookup.get(field, {})
        stats = boot_stats.get(field, {})
        prepost_label = str(obs_row.get("pre_post_separation_label", "unknown"))
        progress_label = str(obs_row.get("progress_quality_label", "unknown"))
        prepost_valid = _prepost_validity_label(prepost_label)
        progress_valid = _progress_validity_label(progress_label)
        duration = _safe_float(obs_row.get("duration", np.nan))
        midpoint_iqr = _safe_float(stats.get("bootstrap_midpoint_iqr", sum_row.get("midpoint_iqr", np.nan)))
        q90_width = _safe_float(stats.get("bootstrap_midpoint_q90_width", np.nan))
        base_rows.append({
            "window_id": W45_ID,
            "anchor_day": W45_ANCHOR,
            "field": field,
            "pre_post_separation_label": prepost_label,
            "pre_post_prototype_validity": prepost_valid,
            "progress_quality_label": progress_label,
            "progress_curve_validity": progress_valid,
            "observed_onset_day": obs_row.get("onset_day", np.nan),
            "observed_midpoint_day": obs_row.get("midpoint_day", np.nan),
            "observed_finish_day": obs_row.get("finish_day", np.nan),
            "observed_duration": duration,
            "bootstrap_midpoint_iqr": midpoint_iqr,
            "bootstrap_midpoint_q90_width": q90_width,
            **stats,
        })
    df = pd.DataFrame(base_rows)

    # Relative ranks within W45. rank=1 means broadest / widest.
    df["duration_rank_within_w45"] = df["observed_duration"].rank(method="min", ascending=False)
    df["midpoint_iqr_rank_within_w45"] = df["bootstrap_midpoint_iqr"].rank(method="min", ascending=False)
    df["q90_width_rank_within_w45"] = df["bootstrap_midpoint_q90_width"].rank(method="min", ascending=False)
    df["is_broadest_duration_field"] = df["duration_rank_within_w45"] == 1
    df["is_widest_midpoint_uncertainty_field"] = df["q90_width_rank_within_w45"] == 1

    midpoint_labels: List[str] = []
    impl_labels: List[str] = []
    reasons: List[str] = []
    interpretable: List[bool] = []
    for _, r in df.iterrows():
        prog_valid = r["progress_curve_validity"]
        pre_valid = r["pre_post_prototype_validity"]
        if prog_valid == "invalid" or pre_valid == "invalid":
            midpoint_label = "midpoint_not_reliable"
            impl = "implementation_invalid"
            reason = "pre/post prototype or progress curve is invalid; pairwise statistics should not be interpreted."
            ok = False
        elif prog_valid == "limited":
            midpoint_label = "midpoint_not_reliable"
            impl = "implementation_limited"
            reason = "progress curve is limited, so midpoint timing is not a reliable field-level timing observable."
            ok = False
        elif bool(r["is_broadest_duration_field"]) and (r["midpoint_iqr_rank_within_w45"] <= 2 or r["q90_width_rank_within_w45"] <= 2):
            midpoint_label = "broad_transition_midpoint_limited"
            impl = "partly_valid_with_caution"
            reason = "field has the broadest W45 transition and high midpoint uncertainty rank; midpoint is interpretable but not a sharp timing marker."
            ok = True
        elif r["duration_rank_within_w45"] <= 2 and r["q90_width_rank_within_w45"] <= 2:
            midpoint_label = "broad_or_uncertain_midpoint_caution"
            impl = "partly_valid_with_caution"
            reason = "field has broad duration and/or wide midpoint uncertainty relative to other W45 fields."
            ok = True
        elif prog_valid == "usable_with_caution" or pre_valid == "weak_with_caution":
            midpoint_label = "midpoint_reasonably_representative_with_caution"
            impl = "partly_valid_with_caution"
            reason = "progress/prepost layer is usable with caution."
            ok = True
        else:
            midpoint_label = "midpoint_reasonably_representative"
            impl = "valid_for_statistical_audit"
            reason = "pre/post separation and progress quality are usable, and midpoint is not unusually broad within W45."
            ok = True
        midpoint_labels.append(midpoint_label)
        impl_labels.append(impl)
        reasons.append(reason)
        interpretable.append(ok)

    df["midpoint_representativeness_label"] = midpoint_labels
    df["implementation_validity_label"] = impl_labels
    df["implementation_validity_reason"] = reasons
    df["statistical_tests_interpretable"] = interpretable
    return df


def _make_w45_bootstrap_pivot(bs: pd.DataFrame, value: str = "midpoint_day") -> pd.DataFrame:
    w = _w45(bs)
    pivot = w.pivot_table(index="bootstrap_id", columns="field", values=value, aggfunc="first")
    return pivot


def _pair_row(df: pd.DataFrame, field_a: str, field_b: str) -> Optional[pd.Series]:
    sub = df.loc[
        (df["window_id"].astype(str) == W45_ID)
        & (df["field_a"].astype(str) == field_a)
        & (df["field_b"].astype(str) == field_b)
    ]
    if not sub.empty:
        return sub.iloc[0]
    sub = df.loc[
        (df["window_id"].astype(str) == W45_ID)
        & (df["field_a"].astype(str) == field_b)
        & (df["field_b"].astype(str) == field_a)
    ]
    if not sub.empty:
        return sub.iloc[0]
    return None


def _delta_for_pair(pivot: pd.DataFrame, field_a: str, field_b: str) -> pd.Series:
    return pivot[field_b].astype(float) - pivot[field_a].astype(float)


def _early_late_from_e1(row: pd.Series) -> Tuple[str, str, float]:
    fa = str(row["field_a"])
    fb = str(row["field_b"])
    med = _safe_float(row.get("median_delta_b_minus_a", np.nan))
    if med > 0:
        return fa, fb, med
    if med < 0:
        return fb, fa, med
    early = str(row.get("field_early_candidate", ""))
    late = str(row.get("field_late_candidate", ""))
    if early and late and early != "nan" and late != "nan":
        return early, late, med
    return "", "", med


def build_tail_audit(inputs: dict, validity_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bs = _w45(inputs["bootstrap_samples"])
    e1 = _w45(inputs["e1_pairwise"])
    midpoint_pivot = _make_w45_bootstrap_pivot(bs, "midpoint_day")
    onset_pivot = _make_w45_bootstrap_pivot(bs, "onset_day")
    finish_pivot = _make_w45_bootstrap_pivot(bs, "finish_day")
    duration_pivot = _make_w45_bootstrap_pivot(bs, "duration")

    q25: Dict[str, float] = {}
    q75: Dict[str, float] = {}
    for field in FIELDS:
        vals = midpoint_pivot[field].dropna().to_numpy(float) if field in midpoint_pivot.columns else np.array([])
        q25[field] = float(np.nanpercentile(vals, 25)) if vals.size else np.nan
        q75[field] = float(np.nanpercentile(vals, 75)) if vals.size else np.nan

    field_ranges = midpoint_pivot[FIELDS].max(axis=1) - midpoint_pivot[FIELDS].min(axis=1)
    range_q25 = float(np.nanpercentile(field_ranges.dropna().to_numpy(float), 25)) if not field_ranges.dropna().empty else np.nan

    sample_rows: List[dict] = []
    summary_rows: List[dict] = []

    for field_x, field_y in TAIL_AUDIT_PAIRS:
        row = _pair_row(e1, field_x, field_y)
        if row is None:
            continue
        early, late, med = _early_late_from_e1(row)
        if not early or not late or med == 0 or pd.isna(med):
            # No directional tail to audit.
            summary_rows.append({
                "pair": f"{field_x}-{field_y}",
                "field_early_candidate": early or "none",
                "field_late_candidate": late or "none",
                "median_delta_for_e1_pair": med,
                "n_tail_samples": 0,
                "tail_sample_fraction": 0.0,
                "n_early_field_late_tail": 0,
                "n_late_field_early_tail": 0,
                "n_mixed_tail": 0,
                "n_all_fields_clustered": 0,
                "n_tail_near_center_mixed": 0,
                "dominant_tail_type": "no_direction_assigned",
                "tail_failure_interpretation": "Pair has no assigned majority direction in V7-e1; tail audit is not applicable.",
            })
            continue
        if early not in midpoint_pivot.columns or late not in midpoint_pivot.columns:
            continue
        delta_late_minus_early = midpoint_pivot[late].astype(float) - midpoint_pivot[early].astype(float)
        tail_mask = delta_late_minus_early <= 0
        tail_ids = list(delta_late_minus_early.index[tail_mask])
        counts = {
            "early_field_late_tail": 0,
            "late_field_early_tail": 0,
            "mixed_tail": 0,
            "all_fields_clustered": 0,
            "tail_near_center_mixed": 0,
        }
        for bid in tail_ids:
            early_mid = _safe_float(midpoint_pivot.loc[bid, early])
            late_mid = _safe_float(midpoint_pivot.loc[bid, late])
            all_range = _safe_float(field_ranges.loc[bid]) if bid in field_ranges.index else np.nan
            clustered = bool(np.isfinite(all_range) and np.isfinite(range_q25) and all_range <= range_q25)
            early_late_side = bool(np.isfinite(early_mid) and np.isfinite(q75.get(early, np.nan)) and early_mid >= q75[early])
            late_early_side = bool(np.isfinite(late_mid) and np.isfinite(q25.get(late, np.nan)) and late_mid <= q25[late])
            if clustered:
                tail_type = "all_fields_clustered"
                interp = "All five W45 midpoints are in a relatively clustered bootstrap state; ordering is hard to resolve in this sample."
            elif early_late_side and not late_early_side:
                tail_type = "early_field_late_tail"
                interp = f"{early} shifts to the late side of its own bootstrap distribution, weakening its lead over {late}."
            elif late_early_side and not early_late_side:
                tail_type = "late_field_early_tail"
                interp = f"{late} shifts to the early side of its own bootstrap distribution, weakening {early}'s lead."
            elif early_late_side and late_early_side:
                tail_type = "mixed_tail"
                interp = f"Both {early} late-side and {late} early-side effects occur in this tail sample."
            else:
                tail_type = "tail_near_center_mixed"
                interp = "Tail sample is not explained by a simple field-specific quartile-side shift."
            counts[tail_type] += 1
            sample_rows.append({
                "pair": f"{field_x}-{field_y}",
                "field_early_candidate": early,
                "field_late_candidate": late,
                "bootstrap_id": bid,
                "delta_late_minus_early": _safe_float(delta_late_minus_early.loc[bid]),
                "H_midpoint": _safe_float(midpoint_pivot.loc[bid, "H"]) if "H" in midpoint_pivot.columns else np.nan,
                "P_midpoint": _safe_float(midpoint_pivot.loc[bid, "P"]) if "P" in midpoint_pivot.columns else np.nan,
                "V_midpoint": _safe_float(midpoint_pivot.loc[bid, "V"]) if "V" in midpoint_pivot.columns else np.nan,
                "Je_midpoint": _safe_float(midpoint_pivot.loc[bid, "Je"]) if "Je" in midpoint_pivot.columns else np.nan,
                "Jw_midpoint": _safe_float(midpoint_pivot.loc[bid, "Jw"]) if "Jw" in midpoint_pivot.columns else np.nan,
                "H_onset": _safe_float(onset_pivot.loc[bid, "H"]) if "H" in onset_pivot.columns else np.nan,
                "H_finish": _safe_float(finish_pivot.loc[bid, "H"]) if "H" in finish_pivot.columns else np.nan,
                "H_duration": _safe_float(duration_pivot.loc[bid, "H"]) if "H" in duration_pivot.columns else np.nan,
                "five_field_midpoint_range": all_range,
                "tail_type": tail_type,
                "tail_interpretation": interp,
            })
        n_tail = int(len(tail_ids))
        n_valid = int(delta_late_minus_early.notna().sum())
        nonzero_counts = {k: v for k, v in counts.items() if v > 0}
        dominant = max(nonzero_counts, key=nonzero_counts.get) if nonzero_counts else "none"
        if dominant == "early_field_late_tail":
            interp = f"Tail failures are mostly caused by {early} moving late within its own bootstrap distribution."
        elif dominant == "late_field_early_tail":
            interp = f"Tail failures are mostly caused by {late} moving early within its own bootstrap distribution."
        elif dominant == "all_fields_clustered":
            interp = "Tail failures often occur when all fields are clustered; W45 is locally hard to separate in those samples."
        elif dominant == "mixed_tail":
            interp = "Tail failures often combine early-field late shifts and late-field early shifts."
        elif dominant == "tail_near_center_mixed":
            interp = "Tail failures are not explained by simple field-specific quartile shifts."
        else:
            interp = "No tail samples found for this directed pair."
        summary_rows.append({
            "pair": f"{field_x}-{field_y}",
            "field_early_candidate": early,
            "field_late_candidate": late,
            "median_delta_for_e1_pair": med,
            "n_tail_samples": n_tail,
            "tail_sample_fraction": float(n_tail / n_valid) if n_valid else np.nan,
            "n_early_field_late_tail": counts["early_field_late_tail"],
            "n_late_field_early_tail": counts["late_field_early_tail"],
            "n_mixed_tail": counts["mixed_tail"],
            "n_all_fields_clustered": counts["all_fields_clustered"],
            "n_tail_near_center_mixed": counts["tail_near_center_mixed"],
            "dominant_tail_type": dominant,
            "tail_failure_interpretation": interp,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def _edge_from_row(row: pd.Series) -> str:
    early = str(row.get("field_early_candidate", ""))
    late = str(row.get("field_late_candidate", ""))
    if early and late and early not in {"nan", "none"} and late not in {"nan", "none"}:
        return f"{early}<{late}"
    med = _safe_float(row.get("median_delta_b_minus_a", np.nan))
    fa = str(row.get("field_a", ""))
    fb = str(row.get("field_b", ""))
    if med > 0:
        return f"{fa}<{fb}"
    if med < 0:
        return f"{fb}<{fa}"
    return f"{fa}/{fb}"


def _edges_for_field(e1: pd.DataFrame, field: str, label: str) -> Tuple[List[str], int, int]:
    sub = e1[(e1["field_a"].astype(str) == field) | (e1["field_b"].astype(str) == field)]
    if label:
        sub = sub[sub["final_evidence_label"].astype(str) == label]
    edges: List[str] = []
    early_count = 0
    late_count = 0
    for _, r in sub.iterrows():
        edge = _edge_from_row(r)
        edges.append(edge)
        if edge.startswith(f"{field}<"):
            early_count += 1
        elif edge.endswith(f"<{field}"):
            late_count += 1
    return edges, early_count, late_count


def build_field_role_audit(inputs: dict, validity_df: pd.DataFrame, e2_w45: pd.DataFrame) -> pd.DataFrame:
    e1 = _w45(inputs["e1_pairwise"])
    validity = {str(r["field"]): r.to_dict() for _, r in validity_df.iterrows()}
    rows: List[dict] = []
    for field in FIELDS:
        confirmed95 = []
        confirmed90 = []
        failed90 = []
        e2_sub = e2_w45[(e2_w45["field_a"].astype(str) == field) | (e2_w45["field_b"].astype(str) == field)]
        for _, r in e2_sub.iterrows():
            edge = _edge_from_row(r)
            if _as_bool(r.get("pass_95", False)):
                confirmed95.append(edge)
            elif _as_bool(r.get("pass_90", False)):
                confirmed90.append(edge)
            else:
                failed90.append(edge)
        supported_edges, sup_early, sup_late = _edges_for_field(e1, field, "supported_directional_tendency")
        notdist_edges, _, _ = _edges_for_field(e1, field, "not_distinguishable")
        validity_label = str(validity.get(field, {}).get("implementation_validity_label", "unknown"))
        midpoint_rep = str(validity.get(field, {}).get("midpoint_representativeness_label", "unknown"))
        progress_quality = str(validity.get(field, {}).get("progress_quality_label", "unknown"))
        prepost = str(validity.get(field, {}).get("pre_post_separation_label", "unknown"))

        if validity_label in {"implementation_limited", "implementation_invalid"}:
            role_label = "quality_limited"
            evidence_level = "implementation_limited"
        elif confirmed95:
            role_label = "confirmed_role"
            evidence_level = "confirmed_95"
        elif confirmed90:
            role_label = "confirmed_90_role"
            evidence_level = "confirmed_90"
        elif sup_early >= 3 and sup_late == 0:
            role_label = "early_broad_candidate" if "broad" in midpoint_rep else "early_candidate"
            evidence_level = "supported_tendency_only"
        elif sup_late >= 3 and sup_early == 0:
            role_label = "late_candidate"
            evidence_level = "supported_tendency_only"
        elif sup_early > sup_late:
            role_label = "middle_early_unresolved"
            evidence_level = "supported_tendency_only"
        elif sup_late > sup_early:
            role_label = "middle_late_unresolved"
            evidence_level = "supported_tendency_only"
        elif notdist_edges:
            role_label = "middle_unresolved"
            evidence_level = "not_distinguishable"
        else:
            role_label = "role_not_resolved"
            evidence_level = "role_not_resolved"

        main_failure_type = "none"
        if not e2_sub.empty:
            vc = e2_sub.loc[~e2_sub["pass_90"].map(_as_bool), "primary_failure_type"].value_counts()
            main_failure_type = str(vc.index[0]) if not vc.empty else "passed_90_or_none"
        quality_flags = []
        if progress_quality not in USABLE_PROGRESS:
            quality_flags.append(f"progress_quality={progress_quality}")
        if prepost not in ACCEPTABLE_SEPARATION:
            quality_flags.append(f"prepost={prepost}")
        if "broad" in midpoint_rep or "uncertain" in midpoint_rep:
            quality_flags.append(f"midpoint={midpoint_rep}")

        interp = (
            f"{field}: role_label={role_label}; evidence={evidence_level}; "
            f"supported_early_edges={sup_early}; supported_late_edges={sup_late}; "
            f"not_distinguishable_edges={len(notdist_edges)}."
        )
        if evidence_level == "supported_tendency_only":
            over = "Do not treat supported tendencies as 90% confirmed field order."
        elif evidence_level == "not_distinguishable":
            over = "Do not interpret not-distinguishable relations as synchrony without an equivalence test."
        elif evidence_level == "implementation_limited":
            over = "Do not interpret pairwise statistics as scientific timing evidence before resolving implementation limits."
        else:
            over = "Interpret only within progress-timing order, not causality."
        rows.append({
            "window_id": W45_ID,
            "anchor_day": W45_ANCHOR,
            "field": field,
            "role_label": role_label,
            "evidence_level": evidence_level,
            "confirmed_edges_95": "; ".join(confirmed95) if confirmed95 else "none",
            "confirmed_edges_90": "; ".join(confirmed90) if confirmed90 else "none",
            "supported_edges": "; ".join(supported_edges) if supported_edges else "none",
            "not_distinguishable_edges": "; ".join(notdist_edges) if notdist_edges else "none",
            "failed_90_edges": "; ".join(failed90) if failed90 else "none",
            "implementation_validity_label": validity_label,
            "main_failure_type": main_failure_type,
            "quality_flags": ";".join(quality_flags) if quality_flags else "none",
            "role_interpretation": interp,
            "do_not_overinterpret": over,
        })
    return pd.DataFrame(rows)


def build_next_step_decision(
    validity_df: pd.DataFrame,
    tail_summary: pd.DataFrame,
    role_df: pd.DataFrame,
    e2_w45: pd.DataFrame,
) -> pd.DataFrame:
    validity_counts = validity_df["implementation_validity_label"].value_counts().to_dict()
    if validity_counts.get("implementation_invalid", 0) or validity_counts.get("implementation_limited", 0):
        overall = "implementation_limited"
    elif validity_counts.get("partly_valid_with_caution", 0):
        overall = "partly_valid_with_caution"
    else:
        overall = "valid_for_statistical_audit"

    h_row = validity_df[validity_df["field"] == "H"]
    h_morph = h_row["midpoint_representativeness_label"].iloc[0] if not h_row.empty else "unknown"
    h_impl = h_row["implementation_validity_label"].iloc[0] if not h_row.empty else "unknown"

    h_pairs = tail_summary[tail_summary["field_early_candidate"].astype(str) == "H"]
    tail_type_counts: Dict[str, int] = {}
    for _, r in h_pairs.iterrows():
        dt = str(r.get("dominant_tail_type", "none"))
        tail_type_counts[dt] = tail_type_counts.get(dt, 0) + 1
    dominant_h_tail = max(tail_type_counts, key=tail_type_counts.get) if tail_type_counts else "none"

    p_jw = e2_w45[
        (((e2_w45["field_a"].astype(str) == "P") & (e2_w45["field_b"].astype(str) == "Jw"))
         | ((e2_w45["field_a"].astype(str) == "Jw") & (e2_w45["field_b"].astype(str) == "P")))
    ]
    if not p_jw.empty:
        middle_status = str(p_jw["primary_failure_type"].iloc[0])
    else:
        middle_status = "unknown"

    if overall == "implementation_limited":
        worth_region = True
        priority = "high"
        action = "inspect_or_revise_progress_implementation_before_scientific_interpretation"
        reason = "At least one W45 field has implementation-limited progress, so statistical order evidence cannot be treated as scientific result."
    elif dominant_h_tail in {"early_field_late_tail", "mixed_tail"} or "broad" in h_morph:
        worth_region = True
        priority = "high_for_H"
        action = "run_H_focused_region_or_feature_level_progress_before_upgrading_W45_order"
        reason = "H shows broad/uncertain midpoint behavior or H-tail failures; W45 H early tendency may be spatially heterogeneous."
    elif dominant_h_tail == "all_fields_clustered" or middle_status == "central_overlap":
        worth_region = False
        priority = "low_to_moderate"
        action = "retain_W45_as_candidate_tendency_and_accept_current_whole_field_nonconfirmation"
        reason = "Tail failures often reflect field clustering or middle-field overlap; whole-field order may be intrinsically hard to separate."
    else:
        worth_region = True
        priority = "moderate"
        action = "inspect_tail_samples_then_decide_region_level_progress"
        reason = "W45 has supported tendencies but no 90% confirmation; tail audit should guide whether region-level progress is warranted."

    return pd.DataFrame([{
        "window_id": W45_ID,
        "anchor_day": W45_ANCHOR,
        "dominant_failure_type": "tail_uncertainty" if not e2_w45.empty and (e2_w45["primary_failure_type"] == "tail_uncertainty").sum() >= 1 else "unknown",
        "implementation_validity_overall": overall,
        "h_morphology_label": h_morph,
        "h_implementation_validity_label": h_impl,
        "dominant_h_tail_type": dominant_h_tail,
        "middle_fields_overlap_status": middle_status,
        "is_worth_region_level_progress": bool(worth_region),
        "region_level_priority": priority,
        "recommended_next_action": action,
        "decision_reason": reason,
    }])


def _write_markdown_summary(paths: W45AuditPaths, validity_df: pd.DataFrame, tail_summary: pd.DataFrame, role_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.extend([
        "# W45 progress implementation/failure audit v7_e3",
        "",
        f"Created at: {_now_iso()}",
        "",
        "## Purpose",
        "Audit whether the V7-e progress-midpoint implementation is interpretable for W45, and explain why W45 directional tendencies do not pass 90% confirmation.",
        "",
        "This audit does not rerun progress timing, does not change thresholds, and does not upgrade supported tendencies to confirmed results.",
        "",
        "## Implementation validity by field",
        "",
    ])
    for _, r in validity_df.iterrows():
        lines.append(f"- {r['field']}: {r['implementation_validity_label']} — {r['implementation_validity_reason']}")
    lines.extend(["", "## Tail failure summary", ""])
    for _, r in tail_summary.iterrows():
        lines.append(f"- {r['pair']} ({r['field_early_candidate']}<{r['field_late_candidate']}): dominant_tail_type={r['dominant_tail_type']}; tail_fraction={r['tail_sample_fraction']:.3f}")
    lines.extend(["", "## Five-field role audit", ""])
    for _, r in role_df.iterrows():
        lines.append(f"- {r['field']}: role={r['role_label']}; evidence={r['evidence_level']}; main_failure={r['main_failure_type']}; caution={r['do_not_overinterpret']}")
    lines.extend(["", "## Next-step decision", ""])
    if not decision_df.empty:
        r = decision_df.iloc[0]
        lines.append(f"- implementation_validity_overall: {r['implementation_validity_overall']}")
        lines.append(f"- h_morphology_label: {r['h_morphology_label']}")
        lines.append(f"- dominant_h_tail_type: {r['dominant_h_tail_type']}")
        lines.append(f"- recommended_next_action: {r['recommended_next_action']}")
        lines.append(f"- reason: {r['decision_reason']}")
    lines.extend([
        "",
        "## Interpretation boundaries",
        "",
        "- Do not interpret not-distinguishable pairs as synchrony without an equivalence test.",
        "- Do not interpret supported tendencies as 90% confirmed orders.",
        "- Do not infer causality or pathway direction from this timing audit.",
        "- Do not omit any of P/V/H/Je/Jw from the W45 interpretation; unresolved roles are still results.",
    ])
    text = "\n".join(lines) + "\n"
    for path in [paths.output_dir / "w45_progress_implementation_failure_audit_summary_v7_e3.md", paths.log_dir / "w45_progress_implementation_failure_audit_summary_v7_e3.md"]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def _write_run_meta(paths: W45AuditPaths, input_audit: dict, validity_df: pd.DataFrame, tail_summary: pd.DataFrame, role_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    meta = {
        "status": "success",
        "created_at": _now_iso(),
        "output_tag": "w45_progress_implementation_failure_audit_v7_e3",
        "window_id": W45_ID,
        "anchor_day": W45_ANCHOR,
        "input_dirs": {
            "v7e_output_dir": str(paths.v7e_output_dir),
            "e1_output_dir": str(paths.e1_output_dir),
            "e2_output_dir": str(paths.e2_output_dir),
        },
        "n_fields": int(len(validity_df)),
        "n_tail_pairs": int(len(tail_summary)),
        "n_role_rows": int(len(role_df)),
        "implementation_validity_counts": validity_df["implementation_validity_label"].value_counts().to_dict() if not validity_df.empty else {},
        "field_role_counts": role_df["role_label"].value_counts().to_dict() if not role_df.empty else {},
        "decision": decision_df.iloc[0].to_dict() if not decision_df.empty else {},
        "input_audit_checks": input_audit.get("checks", {}),
        "notes": [
            "This audit reads V7-e, V7-e1, and V7-e2 outputs only.",
            "It does not rerun progress timing.",
            "It does not modify statistical thresholds.",
            "Implementation validity is checked before interpreting pairwise statistics.",
            "All five fields P/V/H/Je/Jw are retained in the role audit.",
            "not_distinguishable is not interpreted as synchrony.",
        ],
    }
    for path in [paths.output_dir / "run_meta.json", paths.log_dir / "run_meta.json"]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def run_w45_progress_implementation_failure_audit_v7_e3(v7_root: Optional[Path] = None) -> None:
    paths = _get_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)

    print("[V7-e3/W45] Loading V7-e, V7-e1, and V7-e2 outputs...")
    inputs = _load_inputs(paths)

    print("[V7-e3/W45] Validating W45 input completeness...")
    input_audit = validate_w45_inputs(inputs, paths)
    with open(paths.output_dir / "input_audit_v7_e3.json", "w", encoding="utf-8") as f:
        json.dump(input_audit, f, ensure_ascii=False, indent=2)
    with open(paths.log_dir / "input_audit_v7_e3.json", "w", encoding="utf-8") as f:
        json.dump(input_audit, f, ensure_ascii=False, indent=2)

    print("[V7-e3/W45] Building implementation-validity table...")
    validity_df = build_implementation_validity(inputs)
    validity_df.to_csv(paths.output_dir / "w45_implementation_validity_by_field_v7_e3.csv", index=False)

    print("[V7-e3/W45] Building tail-failure audit...")
    tail_summary, tail_samples = build_tail_audit(inputs, validity_df)
    tail_summary.to_csv(paths.output_dir / "w45_tail_failure_by_pair_v7_e3.csv", index=False)
    tail_samples.to_csv(paths.output_dir / "w45_tail_samples_by_pair_v7_e3.csv", index=False)

    print("[V7-e3/W45] Building complete five-field role audit...")
    e2_w45 = _w45(inputs["e2_pairwise"])
    role_df = build_field_role_audit(inputs, validity_df, e2_w45)
    role_df.to_csv(paths.output_dir / "w45_field_role_audit_v7_e3.csv", index=False)

    print("[V7-e3/W45] Building next-step decision table...")
    decision_df = build_next_step_decision(validity_df, tail_summary, role_df, e2_w45)
    decision_df.to_csv(paths.output_dir / "w45_next_step_decision_v7_e3.csv", index=False)

    # Source snapshots for audit reproducibility.
    _w45(inputs["e1_pairwise"]).to_csv(paths.output_dir / "source_w45_pairwise_progress_delta_test_v7_e1_copy.csv", index=False)
    _w45(inputs["e2_pairwise"]).to_csv(paths.output_dir / "source_w45_pairwise_progress_failure_audit_v7_e2_copy.csv", index=False)

    _write_markdown_summary(paths, validity_df, tail_summary, role_df, decision_df)
    _write_run_meta(paths, input_audit, validity_df, tail_summary, role_df, decision_df)
    print(f"[V7-e3/W45] Done. Outputs written to: {paths.output_dir}")
