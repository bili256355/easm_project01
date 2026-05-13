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
PROBLEM_QUALITY_LABELS = {
    "nonmonotonic_progress",
    "boundary_limited_progress",
    "no_clear_prepost_separation",
    "partial_progress",
}
ACCEPTABLE_SEPARATION = {"clear_separation", "moderate_separation"}


@dataclass
class FailureAuditPaths:
    v7_root: Path
    v7e_output_dir: Path
    e1_output_dir: Path
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
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _direction_from_delta(delta: float) -> str:
    if pd.isna(delta):
        return "invalid"
    if delta > 0:
        return "field_a_before_field_b"
    if delta < 0:
        return "field_b_before_field_a"
    return "tie"


def _format_edge(early: str, late: str) -> str:
    if not early or not late or pd.isna(early) or pd.isna(late):
        return "none"
    return f"{early}<{late}"


def _get_paths(v7_root: Optional[Path]) -> FailureAuditPaths:
    if v7_root is None:
        # Module path: stage_partition/V7/src/stage_partition_v7/progress_order_failure_audit.py
        v7_root = Path(__file__).resolve().parents[2]
    v7_root = Path(v7_root).resolve()
    return FailureAuditPaths(
        v7_root=v7_root,
        v7e_output_dir=v7_root / "outputs" / "field_transition_progress_timing_v7_e",
        e1_output_dir=v7_root / "outputs" / "progress_order_significance_audit_v7_e1",
        output_dir=v7_root / "outputs" / "progress_order_failure_audit_v7_e2",
        log_dir=v7_root / "logs" / "progress_order_failure_audit_v7_e2",
    )


def _load_inputs(paths: FailureAuditPaths) -> dict:
    bootstrap_samples = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_samples_v7_e.csv")
    bootstrap_summary = _read_csv(paths.v7e_output_dir / "field_transition_progress_bootstrap_summary_v7_e.csv")
    observed = _read_csv(paths.v7e_output_dir / "field_transition_progress_observed_v7_e.csv")
    loyo_samples = _read_csv(paths.v7e_output_dir / "field_transition_progress_loyo_samples_v7_e.csv", required=False)
    loyo_summary = _read_csv(paths.v7e_output_dir / "field_transition_progress_loyo_summary_v7_e.csv", required=False)
    e1_pairwise = _read_csv(paths.e1_output_dir / "pairwise_progress_delta_test_v7_e1.csv")
    e1_window = _read_csv(paths.e1_output_dir / "window_progress_delta_test_summary_v7_e1.csv", required=False)
    v7e_meta = _read_json(paths.v7e_output_dir / "run_meta.json", required=False)
    e1_meta = _read_json(paths.e1_output_dir / "run_meta.json", required=False)

    _require_columns(
        bootstrap_samples,
        ["window_id", "field", "bootstrap_id", "midpoint_day"],
        "field_transition_progress_bootstrap_samples_v7_e.csv",
    )
    _require_columns(
        e1_pairwise,
        [
            "window_id",
            "field_a",
            "field_b",
            "median_delta_b_minus_a",
            "q025_delta_b_minus_a",
            "q975_delta_b_minus_a",
            "final_evidence_label",
        ],
        "pairwise_progress_delta_test_v7_e1.csv",
    )
    _require_columns(
        bootstrap_summary,
        ["window_id", "field", "midpoint_iqr", "midpoint_q025", "midpoint_q975"],
        "field_transition_progress_bootstrap_summary_v7_e.csv",
    )
    _require_columns(
        observed,
        ["window_id", "field", "pre_post_separation_label", "progress_quality_label"],
        "field_transition_progress_observed_v7_e.csv",
    )
    return {
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_summary": bootstrap_summary,
        "observed": observed,
        "loyo_samples": loyo_samples,
        "loyo_summary": loyo_summary,
        "e1_pairwise": e1_pairwise,
        "e1_window": e1_window,
        "v7e_meta": v7e_meta,
        "e1_meta": e1_meta,
    }


def _make_bootstrap_delta_lookup(bootstrap_samples: pd.DataFrame) -> Dict[Tuple[str, str, str], np.ndarray]:
    lookup: Dict[Tuple[str, str, str], np.ndarray] = {}
    # Pivot per window. Keep only valid numeric midpoint days.
    for window_id, sub in bootstrap_samples.groupby("window_id"):
        pivot = sub.pivot_table(index="bootstrap_id", columns="field", values="midpoint_day", aggfunc="first")
        for field_a, field_b in combinations(FIELDS, 2):
            if field_a not in pivot.columns or field_b not in pivot.columns:
                continue
            delta = pivot[field_b].astype(float) - pivot[field_a].astype(float)
            lookup[(str(window_id), field_a, field_b)] = delta.to_numpy(dtype=float)
    return lookup


def _make_observed_lookup(observed: pd.DataFrame) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    for _, row in observed.iterrows():
        key = (str(row["window_id"]), str(row["field"]))
        out[key] = row.to_dict()
    return out


def _make_field_summary_lookup(bootstrap_summary: pd.DataFrame) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    for _, row in bootstrap_summary.iterrows():
        key = (str(row["window_id"]), str(row["field"]))
        d = row.to_dict()
        q025 = float(d.get("midpoint_q025", np.nan))
        q975 = float(d.get("midpoint_q975", np.nan))
        d["midpoint_q95_width"] = q975 - q025 if np.isfinite(q025) and np.isfinite(q975) else np.nan
        out[key] = d
    return out


def _percentiles(delta: np.ndarray) -> dict:
    valid = delta[np.isfinite(delta)]
    if valid.size == 0:
        return {k: np.nan for k in ["q025", "q05", "q25", "q50", "q75", "q95", "q975", "mean"]}
    return {
        "q025": float(np.nanpercentile(valid, 2.5)),
        "q05": float(np.nanpercentile(valid, 5.0)),
        "q25": float(np.nanpercentile(valid, 25.0)),
        "q50": float(np.nanpercentile(valid, 50.0)),
        "q75": float(np.nanpercentile(valid, 75.0)),
        "q95": float(np.nanpercentile(valid, 95.0)),
        "q975": float(np.nanpercentile(valid, 97.5)),
        "mean": float(np.nanmean(valid)),
    }


def _field_quality_flags(obs_a: dict, obs_b: dict) -> Tuple[str, List[str]]:
    flags: List[str] = []
    qa = str(obs_a.get("progress_quality_label", "unknown"))
    qb = str(obs_b.get("progress_quality_label", "unknown"))
    sa = str(obs_a.get("pre_post_separation_label", "unknown"))
    sb = str(obs_b.get("pre_post_separation_label", "unknown"))

    if qa in PROBLEM_QUALITY_LABELS:
        flags.append(f"field_a_quality={qa}")
    if qb in PROBLEM_QUALITY_LABELS:
        flags.append(f"field_b_quality={qb}")
    if sa not in ACCEPTABLE_SEPARATION:
        flags.append(f"field_a_separation={sa}")
    if sb not in ACCEPTABLE_SEPARATION:
        flags.append(f"field_b_separation={sb}")

    label = "quality_ok" if not flags else "quality_bottleneck"
    return label, flags


def _larger_spread_field(field_a: str, field_b: str, sum_a: dict, sum_b: dict) -> str:
    ia = float(sum_a.get("midpoint_iqr", np.nan))
    ib = float(sum_b.get("midpoint_iqr", np.nan))
    if not np.isfinite(ia) or not np.isfinite(ib):
        return "unknown"
    if ia > ib:
        return field_a
    if ib > ia:
        return field_b
    return "equal"


def _classify_pair_failure(
    pass_90: bool,
    e1_label: str,
    zero_in_iqr: bool,
    zero_in_90ci: bool,
    loyo_conflict: bool,
    quality_label: str,
    e1_significant: bool,
) -> Tuple[str, List[str], str]:
    secondary: List[str] = []

    if pass_90:
        return "passed_90", secondary, "usable_confirmed_or_90_supported_edge"

    if loyo_conflict:
        secondary.append("loyo_conflict")
        return "loyo_conflict", secondary, "do_not_use_for_order"

    if quality_label == "quality_bottleneck":
        secondary.append("quality_bottleneck")
        # quality can coexist with overlap/tail; keep as primary because it limits interpretability.
        return "quality_bottleneck", secondary, "inspect_field_progress_quality"

    if zero_in_iqr:
        secondary.append("zero_inside_iqr")
        return "central_overlap", secondary, "accept_as_indistinguishable_under_current_whole_field_progress"

    if zero_in_90ci:
        secondary.append("zero_inside_90ci_tail")
        if e1_significant or e1_label == "supported_directional_tendency":
            secondary.append("directional_signal_but_tail_uncertain")
            # Clean fields + directional significance + tail uncertainty is a likely method-resolution issue.
            secondary.append("method_resolution_limit_risk")
            return "tail_uncertainty", secondary, "retain_as_candidate_tendency_and_consider_region_level_progress"
        return "tail_uncertainty", secondary, "retain_as_descriptive_only"

    # Fallback: if it failed 90 but no specific pattern is captured.
    return "unclassified_failure", secondary, "manual_review_needed"


def build_pairwise_failure_audit(inputs: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    boot_lookup = _make_bootstrap_delta_lookup(inputs["bootstrap_samples"])
    obs_lookup = _make_observed_lookup(inputs["observed"])
    sum_lookup = _make_field_summary_lookup(inputs["bootstrap_summary"])
    e1 = inputs["e1_pairwise"].copy()

    rows: List[dict] = []
    tail_rows: List[dict] = []

    for _, row in e1.iterrows():
        window_id = str(row["window_id"])
        field_a = str(row["field_a"])
        field_b = str(row["field_b"])
        key = (window_id, field_a, field_b)
        delta = boot_lookup.get(key)
        if delta is None:
            raise ValueError(f"Could not find bootstrap deltas for {key}; check field/window IDs.")
        valid = delta[np.isfinite(delta)]
        pct = _percentiles(valid)
        zero_in_iqr = bool(pct["q25"] <= 0 <= pct["q75"]) if np.isfinite(pct["q25"]) else False
        zero_in_90ci = bool(pct["q05"] <= 0 <= pct["q95"]) if np.isfinite(pct["q05"]) else False
        zero_in_95ci = bool(pct["q025"] <= 0 <= pct["q975"]) if np.isfinite(pct["q025"]) else False
        pass_95 = bool((pct["q025"] > 0) or (pct["q975"] < 0)) if np.isfinite(pct["q025"]) else False
        pass_90 = bool((pct["q05"] > 0) or (pct["q95"] < 0)) if np.isfinite(pct["q05"]) else False

        obs_a = obs_lookup.get((window_id, field_a), {})
        obs_b = obs_lookup.get((window_id, field_b), {})
        sum_a = sum_lookup.get((window_id, field_a), {})
        sum_b = sum_lookup.get((window_id, field_b), {})
        quality_label, quality_flags = _field_quality_flags(obs_a, obs_b)

        final_e1 = str(row.get("final_evidence_label", "unknown"))
        sign_label = str(row.get("significance_label", ""))
        e1_significant = sign_label == "direction_significant" or final_e1 in {
            "confirmed_directional_order",
            "supported_directional_tendency",
        }
        loyo_conflict = _as_bool(row.get("loyo_conflict_flag", False))

        primary, secondary, action = _classify_pair_failure(
            pass_90=pass_90,
            e1_label=final_e1,
            zero_in_iqr=zero_in_iqr,
            zero_in_90ci=zero_in_90ci,
            loyo_conflict=loyo_conflict,
            quality_label=quality_label,
            e1_significant=e1_significant,
        )
        secondary.extend(quality_flags)

        larger_spread = _larger_spread_field(field_a, field_b, sum_a, sum_b)
        if larger_spread not in {"unknown", "equal"}:
            secondary.append(f"larger_spread_field={larger_spread}")

        # Interpret the failure/result in audit language.
        if primary == "passed_90":
            interp = "This pair passes the 90% bootstrap CI criterion and is not treated as a failure."
        elif primary == "central_overlap":
            interp = "Zero lies inside the central 50% bootstrap interval; the two fields are not separable under current whole-field progress timing."
        elif primary == "tail_uncertainty":
            interp = "The central distribution has a direction, but the 90% bootstrap tail crosses zero; this is a candidate tendency, not accepted evidence."
        elif primary == "quality_bottleneck":
            interp = "At least one field has a progress-quality or pre/post-separation issue that limits order interpretation."
        elif primary == "loyo_conflict":
            interp = "Bootstrap and LOYO median directions conflict; the order is sensitive to year composition."
        else:
            interp = "Failure mode is not classified by the current audit rules; manual inspection is recommended."

        out = {
            "window_id": window_id,
            "anchor_day": row.get("anchor_day", np.nan),
            "field_a": field_a,
            "field_b": field_b,
            "median_delta_b_minus_a": pct["q50"],
            "mean_delta_b_minus_a": pct["mean"],
            "q05_delta_b_minus_a": pct["q05"],
            "q95_delta_b_minus_a": pct["q95"],
            "q25_delta_b_minus_a": pct["q25"],
            "q75_delta_b_minus_a": pct["q75"],
            "q025_delta_b_minus_a": pct["q025"],
            "q975_delta_b_minus_a": pct["q975"],
            "pass_95": pass_95,
            "pass_90": pass_90,
            "final_evidence_label_v7_e1": final_e1,
            "significance_label_v7_e1": sign_label,
            "primary_failure_type": primary,
            "secondary_failure_flags": ";".join(secondary) if secondary else "none",
            "zero_in_iqr": zero_in_iqr,
            "zero_in_90ci": zero_in_90ci,
            "zero_in_95ci": zero_in_95ci,
            "field_a_progress_quality": obs_a.get("progress_quality_label", "unknown"),
            "field_b_progress_quality": obs_b.get("progress_quality_label", "unknown"),
            "field_a_prepost_separation": obs_a.get("pre_post_separation_label", "unknown"),
            "field_b_prepost_separation": obs_b.get("pre_post_separation_label", "unknown"),
            "field_a_midpoint_iqr": sum_a.get("midpoint_iqr", np.nan),
            "field_b_midpoint_iqr": sum_b.get("midpoint_iqr", np.nan),
            "field_a_midpoint_q95_width": (sum_a.get("midpoint_q975", np.nan) - sum_a.get("midpoint_q025", np.nan)) if sum_a else np.nan,
            "field_b_midpoint_q95_width": (sum_b.get("midpoint_q975", np.nan) - sum_b.get("midpoint_q025", np.nan)) if sum_b else np.nan,
            "larger_spread_field": larger_spread,
            "loyo_direction": row.get("loyo_direction", "unknown"),
            "loyo_conflict_flag": loyo_conflict,
            "failure_interpretation": interp,
            "recommended_next_action": action,
        }
        rows.append(out)

        if (not pass_90) and primary == "tail_uncertainty" and valid.size > 0:
            med = pct["q50"]
            # Tail samples are samples on the non-dominant side of zero or exactly zero.
            if med > 0:
                mask = delta <= 0
                tail_type = "non_positive_tail_against_A_before_B"
            elif med < 0:
                mask = delta >= 0
                tail_type = "non_negative_tail_against_B_before_A"
            else:
                mask = np.isfinite(delta)
                tail_type = "median_zero_all_samples_for_review"
            # Need bootstrap IDs for tail rows.
            sub = inputs["bootstrap_samples"][inputs["bootstrap_samples"]["window_id"].astype(str) == window_id]
            pivot = sub.pivot_table(index="bootstrap_id", columns="field", values="midpoint_day", aggfunc="first")
            if field_a in pivot.columns and field_b in pivot.columns:
                dseries = pivot[field_b].astype(float) - pivot[field_a].astype(float)
                for bid, dval in dseries[mask].items():
                    if pd.isna(dval):
                        continue
                    tail_rows.append({
                        "window_id": window_id,
                        "field_a": field_a,
                        "field_b": field_b,
                        "bootstrap_id": bid,
                        "delta_b_minus_a": float(dval),
                        "tail_type": tail_type,
                    })

    pair_df = pd.DataFrame(rows)
    tail_df = pd.DataFrame(tail_rows)
    return pair_df, tail_df


def _join_edges(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return "none"
    edges: List[str] = []
    for _, r in df.iterrows():
        # Use median delta sign for edge direction.
        med = r.get("median_delta_b_minus_a", np.nan)
        if pd.isna(med) or med == 0:
            edges.append(f"{r['field_a']}/{r['field_b']}")
        elif med > 0:
            edges.append(f"{r['field_a']}<{r['field_b']}")
        else:
            edges.append(f"{r['field_b']}<{r['field_a']}")
    return "; ".join(edges) if edges else "none"


def build_window_failure_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for window_id, sub in pair_df.groupby("window_id", sort=True):
        n = len(sub)
        counts = sub["primary_failure_type"].value_counts().to_dict()
        n_pass_95 = int(sub["pass_95"].sum())
        n_pass_90 = int(sub["pass_90"].sum())
        n_central = int(counts.get("central_overlap", 0))
        n_tail = int(counts.get("tail_uncertainty", 0))
        n_field_instability = int(sub["secondary_failure_flags"].astype(str).str.contains("larger_spread_field", regex=False).sum())
        n_quality = int(counts.get("quality_bottleneck", 0))
        n_loyo = int(counts.get("loyo_conflict", 0))
        n_method = int(sub["secondary_failure_flags"].astype(str).str.contains("method_resolution_limit_risk", regex=False).sum())
        fail_counts = {k: v for k, v in counts.items() if k != "passed_90"}
        dominant_failure_type = max(fail_counts, key=fail_counts.get) if fail_counts else "none_or_passed_90"

        pass95 = sub[sub["pass_95"]]
        pass90 = sub[(sub["pass_90"]) & (~sub["pass_95"])]
        candidate = sub[sub["primary_failure_type"] == "tail_uncertainty"]
        indist = sub[sub["primary_failure_type"] == "central_overlap"]

        if n_pass_90 > 0 and n_tail > 0:
            interp = "Window has accepted 90% edges plus tail-uncertain candidate relations."
            action = "keep_confirmed_edges_and_audit_tail_uncertainty"
        elif n_pass_90 > 0:
            interp = "Window has at least one accepted 90% edge; non-passing pairs should be interpreted separately."
            action = "keep_confirmed_edges"
        elif n_loyo > 0 and n_loyo >= max(n_central, n_tail, n_quality):
            interp = "LOYO conflicts are present; direction is sensitive to year composition."
            action = "do_not_use_conflicting_edges"
        elif n_quality > 0 and n_quality >= max(n_central, n_tail):
            interp = "Progress-quality bottlenecks dominate or tie for the dominant limitation; field progress curves should be inspected before interpretation."
            action = "inspect_field_progress_quality"
        elif n_central >= max(n_tail, n_quality, n_loyo):
            interp = "Most failures are central-overlap failures; current whole-field progress does not separate many fields in this window."
            action = "accept_indistinguishable_pairs_or_move_to_region_level_only_if_scientifically_needed"
        elif n_tail >= max(n_central, n_quality, n_loyo):
            interp = "Most failures are tail-uncertainty failures; direction exists in the distribution center but does not pass 90%."
            action = "inspect_tail_samples_and_consider_region_level_progress"
        else:
            interp = "No dominant failure mode was identified."
            action = "manual_review_needed"

        rows.append({
            "window_id": window_id,
            "anchor_day": sub["anchor_day"].iloc[0] if "anchor_day" in sub.columns else np.nan,
            "n_pairs": n,
            "n_pass_95": n_pass_95,
            "n_pass_90": n_pass_90,
            "n_central_overlap": n_central,
            "n_tail_uncertainty": n_tail,
            "n_field_timing_instability_flags": n_field_instability,
            "n_quality_bottleneck": n_quality,
            "n_loyo_conflict": n_loyo,
            "n_method_resolution_limit_risk": n_method,
            "dominant_failure_type": dominant_failure_type,
            "usable_confirmed_edges_95": _join_edges(pass95, "pass_95"),
            "usable_edges_90_only": _join_edges(pass90, "pass_90"),
            "candidate_edges_tail_uncertainty": _join_edges(candidate, "tail_uncertainty"),
            "indistinguishable_pairs_central_overlap": _join_edges(indist, "central_overlap"),
            "window_failure_interpretation": interp,
            "recommended_window_next_action": action,
        })
    return pd.DataFrame(rows)


def _write_run_meta(paths: FailureAuditPaths, pair_df: pd.DataFrame, window_df: pd.DataFrame, inputs: dict) -> None:
    meta = {
        "status": "success",
        "created_at": _now_iso(),
        "output_tag": "progress_order_failure_audit_v7_e2",
        "input_v7e_output_dir": str(paths.v7e_output_dir),
        "input_e1_output_dir": str(paths.e1_output_dir),
        "n_pairs": int(len(pair_df)),
        "n_windows": int(window_df["window_id"].nunique()) if not window_df.empty else 0,
        "n_pass_95": int(pair_df["pass_95"].sum()) if not pair_df.empty else 0,
        "n_pass_90": int(pair_df["pass_90"].sum()) if not pair_df.empty else 0,
        "failure_type_counts": pair_df["primary_failure_type"].value_counts().to_dict() if not pair_df.empty else {},
        "notes": [
            "This audit does not recompute progress timing.",
            "It explains why pairwise progress orders do or do not pass the 90% bootstrap CI criterion.",
            "No minimum effective day threshold is introduced.",
            "central_overlap means zero lies in the central 50% delta interval, not merely in a tail.",
            "tail_uncertainty means the distribution center has direction but the 90% interval crosses zero.",
            "method_resolution_limit_risk is a caution flag, not a confirmed diagnosis.",
        ],
    }
    with open(paths.output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(paths.log_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _write_audit_log(paths: FailureAuditPaths, pair_df: pd.DataFrame, window_df: pd.DataFrame) -> None:
    lines = [
        "# progress_order_failure_audit_v7_e2",
        "",
        f"Created at: {_now_iso()}",
        "",
        "## Purpose",
        "Explain why most V7-e / V7-e1 pairwise progress orders did not pass the 90% bootstrap CI criterion.",
        "This is a failure decomposition audit, not a new timing method and not a result-picking layer.",
        "",
        "## Inputs",
        f"- V7-e output: `{paths.v7e_output_dir}`",
        f"- V7-e1 output: `{paths.e1_output_dir}`",
        "",
        "## Failure categories",
        "- `passed_90`: pair passes 90% CI and is not treated as a failure.",
        "- `central_overlap`: zero lies inside the 25–75% interval of Δ; current whole-field progress does not separate the pair.",
        "- `tail_uncertainty`: zero is outside the IQR but inside the 90% CI; direction exists in the center but fails in tails.",
        "- `quality_bottleneck`: at least one field has problematic progress quality or pre/post separation.",
        "- `loyo_conflict`: bootstrap median direction conflicts with LOYO median direction.",
        "",
        "## Counts",
        "",
    ]
    if not pair_df.empty:
        counts = pair_df["primary_failure_type"].value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
    lines.extend([
        "",
        "## Window summary",
        "",
    ])
    if not window_df.empty:
        for _, r in window_df.iterrows():
            lines.append(f"### {r['window_id']}")
            lines.append(f"- pass95/pass90: {r['n_pass_95']} / {r['n_pass_90']}")
            lines.append(f"- dominant_failure_type: {r['dominant_failure_type']}")
            lines.append(f"- interpretation: {r['window_failure_interpretation']}")
            lines.append(f"- recommended_next_action: {r['recommended_window_next_action']}")
            lines.append("")
    lines.extend([
        "## Interpretation boundaries",
        "",
        "- This audit does not prove causality.",
        "- This audit does not create new progress-order evidence.",
        "- Pairs that fail 90% should not be upgraded by narrative wording.",
        "- `method_resolution_limit_risk` only indicates where region/component-level progress may be worth checking.",
    ])
    path = paths.log_dir / "progress_order_failure_audit_v7_e2.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(paths.output_dir / "progress_order_failure_audit_v7_e2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_progress_order_failure_audit_v7_e2(v7_root: Optional[Path] = None) -> None:
    paths = _get_paths(v7_root)
    _ensure_dir(paths.output_dir)
    _ensure_dir(paths.log_dir)

    print("[V7-e2] Loading V7-e / V7-e1 outputs...")
    inputs = _load_inputs(paths)

    print("[V7-e2] Building pairwise failure audit...")
    pair_df, tail_df = build_pairwise_failure_audit(inputs)
    pair_path = paths.output_dir / "pairwise_progress_failure_audit_v7_e2.csv"
    pair_df.to_csv(pair_path, index=False)

    print("[V7-e2] Building window-level failure audit...")
    window_df = build_window_failure_summary(pair_df)
    window_path = paths.output_dir / "window_progress_failure_audit_v7_e2.csv"
    window_df.to_csv(window_path, index=False)

    if not tail_df.empty:
        tail_df.to_csv(paths.output_dir / "pairwise_progress_tail_samples_v7_e2.csv", index=False)
    else:
        pd.DataFrame(columns=["window_id", "field_a", "field_b", "bootstrap_id", "delta_b_minus_a", "tail_type"]).to_csv(
            paths.output_dir / "pairwise_progress_tail_samples_v7_e2.csv", index=False
        )

    # Keep source snapshots for reproducibility.
    inputs["e1_pairwise"].to_csv(paths.output_dir / "source_pairwise_progress_delta_test_v7_e1_copy.csv", index=False)

    _write_run_meta(paths, pair_df, window_df, inputs)
    _write_audit_log(paths, pair_df, window_df)
    print(f"[V7-e2] Done. Outputs written to: {paths.output_dir}")
