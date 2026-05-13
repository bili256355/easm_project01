"""
V9.1_f_all_pairs_sequence_registry_a

Read-only summarizer for V9.1_f_all_pairs_a outputs.
It does NOT rerun bootstrap, MCA/SVD, or the peak detector.

Purpose:
For each window-target score phase (high/mid/low), summarize the full five-object
bootstrap peak sequence using already-computed bootstrap object peaks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import os
import itertools
import math
import shutil
from typing import Iterable

import numpy as np
import pandas as pd


OBJECTS = ("P", "V", "H", "Je", "Jw")
PAIR_LIST = tuple(itertools.combinations(OBJECTS, 2))


@dataclass
class SequenceRegistrySettings:
    version: str = "v9_1_f_all_pairs_sequence_registry_a"
    input_tag: str = "bootstrap_composite_mca_audit_v9_1_f_all_pairs_a"
    output_tag: str = "v9_1_f_all_pairs_sequence_registry_a"
    target_windows: tuple[str, ...] = ("W045", "W081", "W113", "W160")
    score_groups: tuple[str, ...] = ("high", "mid", "low")
    high_low_groups: tuple[str, str] = ("high", "low")
    near_day_threshold: float = 0.0
    object_shift_report_threshold_days: float = 3.0
    does_not_rerun_bootstrap: bool = True
    does_not_rerun_mca_or_svd: bool = True
    does_not_rerun_peak_detector: bool = True
    physical_interpretation_included: bool = False


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name, "").strip()
    return Path(val) if val else default


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _move_cols_front(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.loc[:, cols + [c for c in df.columns if c not in cols]]


def _target_pair_from_name(target_name: str) -> str:
    # Expected examples: W045_P_vs_V_delta_peak
    parts = str(target_name).split("_")
    if len(parts) >= 4 and parts[2] == "vs":
        return f"{parts[1]}-{parts[3]}"
    return target_name


def _order_label(delta_b_minus_a: float, near: float) -> str:
    if pd.isna(delta_b_minus_a):
        return "missing"
    if abs(float(delta_b_minus_a)) <= near:
        return "same_or_near"
    if delta_b_minus_a > 0:
        return "A_earlier"
    return "B_earlier"


def _sequence_string(stats_phase: pd.DataFrame, near: float) -> str:
    """Build sequence like H(36.0) -> Jw/P(42.0/42.5) -> V(45.0)."""
    df = stats_phase[["object", "median_peak_day"]].copy()
    df = df.dropna(subset=["median_peak_day"])
    if df.empty:
        return "missing"
    df = df.sort_values(["median_peak_day", "object"]).reset_index(drop=True)
    groups: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = []
    current_ref = None
    for _, row in df.iterrows():
        obj = str(row["object"])
        val = float(row["median_peak_day"])
        if current_ref is None:
            current = [(obj, val)]
            current_ref = val
        elif abs(val - current_ref) <= near:
            current.append((obj, val))
            current_ref = float(np.nanmean([v for _, v in current]))
        else:
            groups.append(current)
            current = [(obj, val)]
            current_ref = val
    if current:
        groups.append(current)
    chunks = []
    for g in groups:
        objs = "/".join(o for o, _ in g)
        vals = "/".join(f"{v:.1f}" for _, v in g)
        chunks.append(f"{objs}({vals})")
    return " -> ".join(chunks)


def _dominant_order_label(pair_df: pd.DataFrame) -> str:
    # Dominant order by highest probability among A_earlier, B_earlier, same_or_near.
    vals = {
        "A_earlier": float(pair_df.get("prob_A_earlier", np.nan)),
        "B_earlier": float(pair_df.get("prob_B_earlier", np.nan)),
        "same_or_near": float(pair_df.get("prob_same_or_near", np.nan)),
    }
    vals = {k: v for k, v in vals.items() if not pd.isna(v)}
    if not vals:
        return "missing"
    return max(vals, key=vals.get)


def _load_inputs(base_in: Path, settings: SequenceRegistrySettings) -> dict[str, pd.DataFrame]:
    cross = base_in / "cross_window"
    inputs = {
        "scores": _read_csv(cross / "bootstrap_composite_mca_scores_all_windows.csv"),
        "target_registry": _read_csv(cross / "v9_1_f_all_pairs_target_registry.csv"),
        "stat_registry": _read_csv(cross / "v9_1_f_all_pairs_statistical_result_registry.csv", required=False),
        "evidence_v3": _read_csv(cross / "bootstrap_composite_mca_evidence_v3_all_windows.csv", required=False),
        "high_low_order": _read_csv(cross / "bootstrap_composite_mca_high_low_order_all_windows.csv", required=False),
    }
    return inputs


def _load_object_peaks(base_in: Path, window_id: str) -> pd.DataFrame:
    p = base_in / "per_window" / window_id / f"bootstrap_object_peak_samples_{window_id}.csv"
    df = _read_csv(p)
    needed = {"window_id", "bootstrap_id", "object", "selected_peak_day"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns: {missing}")
    return df


def _build_long_phase_peaks(scores: pd.DataFrame, object_peaks: pd.DataFrame, window_id: str, settings: SequenceRegistrySettings) -> pd.DataFrame:
    s = scores.loc[scores["window_id"].astype(str) == str(window_id)].copy()
    s = s.loc[s["score_group"].isin(settings.score_groups)].copy()
    p = object_peaks.loc[object_peaks["window_id"].astype(str) == str(window_id)].copy()
    keep_s = ["window_id", "target_name", "bootstrap_id", "score", "score_group", "Y_delta_B_minus_A"]
    keep_s = [c for c in keep_s if c in s.columns]
    merged = s[keep_s].merge(
        p[["window_id", "bootstrap_id", "object", "selected_peak_day"]],
        on=["window_id", "bootstrap_id"],
        how="left",
        validate="many_to_many",
    )
    merged["target_pair"] = merged["target_name"].map(_target_pair_from_name)
    return merged


def _object_peak_stats(long_df: pd.DataFrame, target_registry: pd.DataFrame, stat_registry: pd.DataFrame, settings: SequenceRegistrySettings) -> pd.DataFrame:
    rows = []
    group_cols = ["window_id", "target_name", "target_pair", "score_group", "object"]
    for keys, g in long_df.groupby(group_cols, dropna=False):
        window_id, target_name, target_pair, score_group, obj = keys
        vals = pd.to_numeric(g["selected_peak_day"], errors="coerce")
        rows.append({
            "window_id": window_id,
            "target_name": target_name,
            "target_pair": target_pair,
            "score_group": score_group,
            "object": obj,
            "n_bootstrap_group": int(g["bootstrap_id"].nunique()),
            "n_valid_peak": int(vals.notna().sum()),
            "missing_peak_fraction": float(vals.isna().mean()) if len(vals) else np.nan,
            "mean_peak_day": float(vals.mean()) if vals.notna().any() else np.nan,
            "median_peak_day": float(vals.median()) if vals.notna().any() else np.nan,
            "q25_peak_day": float(vals.quantile(0.25)) if vals.notna().any() else np.nan,
            "q75_peak_day": float(vals.quantile(0.75)) if vals.notna().any() else np.nan,
            "std_peak_day": float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan,
            "min_peak_day": float(vals.min()) if vals.notna().any() else np.nan,
            "max_peak_day": float(vals.max()) if vals.notna().any() else np.nan,
        })
    out = pd.DataFrame(rows)
    out = _annotate_targets(out, target_registry, stat_registry)
    return _move_cols_front(out, ["window_id", "target_name", "target_pair", "target_set", "was_in_original_v9_1_f", "score_group", "object"])


def _sequence_summary(object_stats: pd.DataFrame, target_registry: pd.DataFrame, stat_registry: pd.DataFrame, settings: SequenceRegistrySettings) -> pd.DataFrame:
    rows = []
    for (window_id, target_name, target_pair, score_group), g in object_stats.groupby(["window_id", "target_name", "target_pair", "score_group"], dropna=False):
        seq = _sequence_string(g, settings.near_day_threshold)
        peak_dict = {str(r["object"]): r["median_peak_day"] for _, r in g.iterrows()}
        rows.append({
            "window_id": window_id,
            "target_name": target_name,
            "target_pair": target_pair,
            "score_group": score_group,
            "sequence_string": seq,
            "object_peak_median_json": json.dumps(peak_dict, ensure_ascii=False, sort_keys=True),
            "n_objects_present": int(pd.Series(peak_dict).notna().sum()),
            "near_day_threshold": settings.near_day_threshold,
            "evidence_stage": "target_conditioned_sequence_from_precomputed_bootstrap_peaks",
            "does_not_rerun_peak_detector": True,
        })
    out = pd.DataFrame(rows)
    out = _annotate_targets(out, target_registry, stat_registry)
    return _move_cols_front(out, ["window_id", "target_name", "target_pair", "target_set", "was_in_original_v9_1_f", "score_group", "sequence_string"])


def _pair_order_matrix(long_df: pd.DataFrame, target_registry: pd.DataFrame, stat_registry: pd.DataFrame, settings: SequenceRegistrySettings) -> pd.DataFrame:
    # Pivot once per window-target-phase-bootstrap into object columns.
    pivot = long_df.pivot_table(
        index=["window_id", "target_name", "target_pair", "score_group", "bootstrap_id"],
        columns="object",
        values="selected_peak_day",
        aggfunc="first",
    ).reset_index()
    rows = []
    for (window_id, target_name, target_pair, score_group), g in pivot.groupby(["window_id", "target_name", "target_pair", "score_group"], dropna=False):
        for A, B in PAIR_LIST:
            if A not in g.columns or B not in g.columns:
                continue
            delta = pd.to_numeric(g[B], errors="coerce") - pd.to_numeric(g[A], errors="coerce")
            valid = delta.dropna()
            labels = valid.map(lambda x: _order_label(x, settings.near_day_threshold))
            n = int(len(valid))
            if n == 0:
                prob_A = prob_B = prob_same = np.nan
            else:
                prob_A = float((labels == "A_earlier").mean())
                prob_B = float((labels == "B_earlier").mean())
                prob_same = float((labels == "same_or_near").mean())
            rows.append({
                "window_id": window_id,
                "target_name": target_name,
                "target_pair": target_pair,
                "score_group": score_group,
                "object_A": A,
                "object_B": B,
                "pair_name": f"{A}-{B}",
                "n_valid_bootstrap": n,
                "median_delta_B_minus_A": float(valid.median()) if n else np.nan,
                "mean_delta_B_minus_A": float(valid.mean()) if n else np.nan,
                "q25_delta_B_minus_A": float(valid.quantile(0.25)) if n else np.nan,
                "q75_delta_B_minus_A": float(valid.quantile(0.75)) if n else np.nan,
                "prob_A_earlier": prob_A,
                "prob_B_earlier": prob_B,
                "prob_same_or_near": prob_same,
                "dominant_order_label": max({"A_earlier": prob_A, "B_earlier": prob_B, "same_or_near": prob_same}, key=lambda k: -1 if pd.isna({"A_earlier": prob_A, "B_earlier": prob_B, "same_or_near": prob_same}[k]) else {"A_earlier": prob_A, "B_earlier": prob_B, "same_or_near": prob_same}[k]),
                "near_day_threshold": settings.near_day_threshold,
            })
    out = pd.DataFrame(rows)
    out = _annotate_targets(out, target_registry, stat_registry)
    return _move_cols_front(out, ["window_id", "target_name", "target_pair", "target_set", "was_in_original_v9_1_f", "score_group", "pair_name", "object_A", "object_B"])


def _high_low_sequence_shift(object_stats: pd.DataFrame, seq: pd.DataFrame, pair_order: pd.DataFrame, target_registry: pd.DataFrame, stat_registry: pd.DataFrame, settings: SequenceRegistrySettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    hi, lo = settings.high_low_groups
    obj_rows = []
    summary_rows = []
    # object shift
    keycols = ["window_id", "target_name", "target_pair", "object"]
    high = object_stats.loc[object_stats["score_group"] == hi, keycols + ["median_peak_day", "q25_peak_day", "q75_peak_day"]].rename(columns={
        "median_peak_day": "median_peak_day_high",
        "q25_peak_day": "q25_peak_day_high",
        "q75_peak_day": "q75_peak_day_high",
    })
    low = object_stats.loc[object_stats["score_group"] == lo, keycols + ["median_peak_day", "q25_peak_day", "q75_peak_day"]].rename(columns={
        "median_peak_day": "median_peak_day_low",
        "q25_peak_day": "q25_peak_day_low",
        "q75_peak_day": "q75_peak_day_low",
    })
    obj_shift = high.merge(low, on=keycols, how="outer")
    obj_shift["shift_high_minus_low"] = obj_shift["median_peak_day_high"] - obj_shift["median_peak_day_low"]
    obj_shift["abs_shift_high_low"] = obj_shift["shift_high_minus_low"].abs()
    obj_shift["shift_ge_threshold_flag"] = obj_shift["abs_shift_high_low"] >= settings.object_shift_report_threshold_days
    obj_shift = _annotate_targets(obj_shift, target_registry, stat_registry)
    obj_shift = _move_cols_front(obj_shift, ["window_id", "target_name", "target_pair", "target_set", "was_in_original_v9_1_f", "object"])

    # pair order high-low changes
    keyp = ["window_id", "target_name", "target_pair", "pair_name", "object_A", "object_B"]
    ph = pair_order.loc[pair_order["score_group"] == hi, keyp + ["dominant_order_label", "median_delta_B_minus_A", "prob_A_earlier", "prob_B_earlier", "prob_same_or_near"]].rename(columns={
        "dominant_order_label": "dominant_order_high",
        "median_delta_B_minus_A": "median_delta_high",
        "prob_A_earlier": "prob_A_earlier_high",
        "prob_B_earlier": "prob_B_earlier_high",
        "prob_same_or_near": "prob_same_or_near_high",
    })
    pl = pair_order.loc[pair_order["score_group"] == lo, keyp + ["dominant_order_label", "median_delta_B_minus_A", "prob_A_earlier", "prob_B_earlier", "prob_same_or_near"]].rename(columns={
        "dominant_order_label": "dominant_order_low",
        "median_delta_B_minus_A": "median_delta_low",
        "prob_A_earlier": "prob_A_earlier_low",
        "prob_B_earlier": "prob_B_earlier_low",
        "prob_same_or_near": "prob_same_or_near_low",
    })
    pair_shift = ph.merge(pl, on=keyp, how="outer")
    pair_shift["median_delta_shift_high_minus_low"] = pair_shift["median_delta_high"] - pair_shift["median_delta_low"]
    pair_shift["dominant_order_changed_flag"] = pair_shift["dominant_order_high"] != pair_shift["dominant_order_low"]
    pair_shift = _annotate_targets(pair_shift, target_registry, stat_registry)
    pair_shift = _move_cols_front(pair_shift, ["window_id", "target_name", "target_pair", "target_set", "was_in_original_v9_1_f", "pair_name", "object_A", "object_B"])

    # target-level summary
    seqh = seq.loc[seq["score_group"] == hi, ["window_id", "target_name", "target_pair", "sequence_string"]].rename(columns={"sequence_string": "sequence_high"})
    seql = seq.loc[seq["score_group"] == lo, ["window_id", "target_name", "target_pair", "sequence_string"]].rename(columns={"sequence_string": "sequence_low"})
    target_keys = ["window_id", "target_name", "target_pair"]
    base = seqh.merge(seql, on=target_keys, how="outer")
    for keys, g in obj_shift.groupby(target_keys, dropna=False):
        row = dict(zip(target_keys, keys))
        shifts = g.dropna(subset=["abs_shift_high_low"])
        row.update({
            "n_objects_with_shift_ge_threshold": int(shifts["shift_ge_threshold_flag"].sum()) if not shifts.empty else 0,
            "max_abs_object_shift_high_low": float(shifts["abs_shift_high_low"].max()) if not shifts.empty else np.nan,
            "object_with_max_abs_shift": str(shifts.loc[shifts["abs_shift_high_low"].idxmax(), "object"]) if not shifts.empty else "missing",
        })
        # pair changes
        gp = pair_shift
        gp = gp[(gp["window_id"] == row["window_id"]) & (gp["target_name"] == row["target_name"])]
        row.update({
            "n_pairs_with_dominant_order_change": int(gp["dominant_order_changed_flag"].fillna(False).sum()) if not gp.empty else 0,
            "n_pairs_total": int(len(gp)) if not gp.empty else 0,
            "sequence_registry_scope": "target_score_phase_conditioned_window_level_sequence",
            "evidence_stage": "read_only_derived_from_v9_1_f_all_pairs_a_outputs",
            "does_not_rerun_bootstrap": True,
            "does_not_rerun_peak_detector": True,
            "physical_interpretation_status": "not_started",
        })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary = base.merge(summary, on=target_keys, how="outer") if not base.empty else summary
    summary = _annotate_targets(summary, target_registry, stat_registry)
    summary = _move_cols_front(summary, ["window_id", "target_name", "target_pair", "target_set", "was_in_original_v9_1_f", "sequence_high", "sequence_low"])
    return obj_shift, pair_shift, summary


def _annotate_targets(df: pd.DataFrame, target_registry: pd.DataFrame, stat_registry: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    cols = [c for c in ["window_id", "target_name", "target_pair", "target_set", "was_in_original_v9_1_f"] if c in target_registry.columns]
    anno = target_registry[cols].drop_duplicates()
    merge_cols = ["window_id", "target_name"] if "target_name" in out.columns else []
    if merge_cols:
        out = out.merge(anno.drop(columns=[c for c in ["target_pair"] if c in anno.columns and c in out.columns], errors="ignore"), on=merge_cols, how="left")
        # If target_pair got not retained due to dedupe, leave existing.
    if not stat_registry.empty and "target_name" in stat_registry.columns:
        stat_cols = [c for c in ["window_id", "target_name", "evidence_v3", "evidence_family", "result_basis", "corr_score_y", "permutation_level", "signflip_level", "mode_stability_status", "extreme_year_dominated_flag"] if c in stat_registry.columns]
        if len(stat_cols) >= 2:
            out = out.merge(stat_registry[stat_cols].drop_duplicates(), on=["window_id", "target_name"], how="left")
    # repair target_pair if missing
    if "target_pair" not in out.columns or out["target_pair"].isna().all():
        if "target_name" in out.columns:
            out["target_pair"] = out["target_name"].map(_target_pair_from_name)
    return out


def _window_level_summary(sequence_shift: pd.DataFrame, target_registry: pd.DataFrame, settings: SequenceRegistrySettings) -> pd.DataFrame:
    rows = []
    for wid, g in sequence_shift.groupby("window_id", dropna=False):
        n_targets = int(g["target_name"].nunique())
        n_orig = int(g.loc[g.get("target_set", pd.Series(index=g.index)) == "original_priority", "target_name"].nunique()) if "target_set" in g.columns else np.nan
        n_added = int(g.loc[g.get("target_set", pd.Series(index=g.index)) == "added_all_pair", "target_name"].nunique()) if "target_set" in g.columns else np.nan
        rows.append({
            "window_id": wid,
            "n_targets_summarized": n_targets,
            "n_original_priority_targets": n_orig,
            "n_added_all_pair_targets": n_added,
            "mean_n_pairs_order_changed_high_low": float(g["n_pairs_with_dominant_order_change"].mean()) if "n_pairs_with_dominant_order_change" in g else np.nan,
            "max_n_pairs_order_changed_high_low": int(g["n_pairs_with_dominant_order_change"].max()) if "n_pairs_with_dominant_order_change" in g and len(g) else np.nan,
            "mean_max_abs_object_shift": float(g["max_abs_object_shift_high_low"].mean()) if "max_abs_object_shift_high_low" in g else np.nan,
            "max_abs_object_shift": float(g["max_abs_object_shift_high_low"].max()) if "max_abs_object_shift_high_low" in g and len(g) else np.nan,
            "summary_scope": "target-conditioned high-low window-level peak sequence, not physical interpretation",
        })
    return pd.DataFrame(rows)


def _write_summary_md(path: Path, run_meta: dict, window_summary: pd.DataFrame, sequence_shift: pd.DataFrame) -> None:
    lines = []
    lines.append("# V9.1_f all-pairs sequence registry A\n")
    lines.append("This is a read-only derived registry. It does not rerun bootstrap, MCA/SVD, or the peak detector.\n")
    lines.append("## Purpose\n")
    lines.append("For each V9.1_f_all_pairs target, use the target score high/mid/low groups and the precomputed bootstrap object peaks to summarize the full five-object peak sequence under each score phase.\n")
    lines.append("## Key boundaries\n")
    lines.append("- The sequence is target-conditioned: each target has its own score direction and therefore its own high/low grouping.\n")
    lines.append("- These outputs are not physical regimes and not independent year types.\n")
    lines.append("- Results are derived from bootstrap-composite samples, not direct-year samples.\n")
    lines.append("- This layer exposes window-level sequence information that was already latent in the bootstrap peaks but not previously summarized.\n")
    lines.append("## Output overview\n")
    lines.append(f"- targets summarized: {int(sequence_shift['target_name'].nunique()) if not sequence_shift.empty else 0}\n")
    lines.append(f"- windows summarized: {int(sequence_shift['window_id'].nunique()) if not sequence_shift.empty else 0}\n")
    lines.append("\n## Window summary\n")
    if not window_summary.empty:
        lines.append(window_summary.to_markdown(index=False))
        lines.append("\n")
    lines.append("\n## Run metadata\n")
    lines.append("```json\n")
    lines.append(json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("\n```\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_summarize_v9_1_f_all_pairs_sequence_registry_a(v91_root: Path | str) -> None:
    v91_root = Path(v91_root)
    settings = SequenceRegistrySettings()
    base_in = _env_path("V9_1_F_ALL_PAIRS_OUTPUT_DIR", v91_root / "outputs" / settings.input_tag)
    base_out = _env_path("V9_1_F_ALL_PAIRS_SEQUENCE_OUTPUT_DIR", v91_root / "outputs" / settings.output_tag)
    cross_out = base_out / "cross_window"
    per_out = base_out / "per_window"
    _ensure_dir(cross_out)
    _ensure_dir(per_out)

    print(f"[1/6] Load V9.1_f all-pair outputs from {base_in}")
    inputs = _load_inputs(base_in, settings)
    scores = inputs["scores"]
    target_registry = inputs["target_registry"]
    stat_registry = inputs["stat_registry"]
    if stat_registry.empty and not inputs["evidence_v3"].empty:
        stat_registry = inputs["evidence_v3"]

    all_long = []
    all_object_stats = []
    all_seq = []
    all_pair_order = []

    windows = tuple(str(w) for w in settings.target_windows)
    for idx, wid in enumerate(windows, start=1):
        print(f"[2/6] Window {idx}/{len(windows)} {wid}: load bootstrap object peaks and merge score phases")
        object_peaks = _load_object_peaks(base_in, wid)
        long_df = _build_long_phase_peaks(scores, object_peaks, wid, settings)
        all_long.append(long_df)
        object_stats = _object_peak_stats(long_df, target_registry, stat_registry, settings)
        seq = _sequence_summary(object_stats, target_registry, stat_registry, settings)
        pair_order = _pair_order_matrix(long_df, target_registry, stat_registry, settings)
        all_object_stats.append(object_stats)
        all_seq.append(seq)
        all_pair_order.append(pair_order)

        wout = per_out / wid
        _ensure_dir(wout)
        object_stats.to_csv(wout / f"target_conditioned_object_peak_stats_{wid}.csv", index=False)
        seq.to_csv(wout / f"target_conditioned_sequence_summary_{wid}.csv", index=False)
        pair_order.to_csv(wout / f"target_conditioned_pair_order_matrix_{wid}.csv", index=False)

    print("[3/6] Build high-low sequence shift summaries")
    long_all = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    object_stats_all = pd.concat(all_object_stats, ignore_index=True) if all_object_stats else pd.DataFrame()
    seq_all = pd.concat(all_seq, ignore_index=True) if all_seq else pd.DataFrame()
    pair_order_all = pd.concat(all_pair_order, ignore_index=True) if all_pair_order else pd.DataFrame()
    obj_shift, pair_shift, sequence_shift = _high_low_sequence_shift(object_stats_all, seq_all, pair_order_all, target_registry, stat_registry, settings)
    window_summary = _window_level_summary(sequence_shift, target_registry, settings)

    print("[4/6] Write cross-window outputs")
    # Long merged sample table can be huge but useful; write only a lightweight version if requested.
    save_long = os.environ.get("V9_1_F_ALL_PAIRS_SEQUENCE_SAVE_LONG", "0").strip() == "1"
    if save_long:
        long_all.to_csv(cross_out / "target_conditioned_bootstrap_object_peak_long_all_windows.csv", index=False)
    object_stats_all.to_csv(cross_out / "target_conditioned_object_peak_stats_all_windows.csv", index=False)
    seq_all.to_csv(cross_out / "target_conditioned_sequence_summary_all_windows.csv", index=False)
    pair_order_all.to_csv(cross_out / "target_conditioned_pair_order_matrix_all_windows.csv", index=False)
    obj_shift.to_csv(cross_out / "target_conditioned_object_peak_shift_high_low_all_windows.csv", index=False)
    pair_shift.to_csv(cross_out / "target_conditioned_pair_order_shift_high_low_all_windows.csv", index=False)
    sequence_shift.to_csv(cross_out / "target_conditioned_sequence_shift_summary_all_windows.csv", index=False)
    window_summary.to_csv(cross_out / "target_conditioned_window_sequence_summary.csv", index=False)

    print("[5/6] Write run_meta and summary")
    run_meta = {
        "version": settings.version,
        "input_tag": settings.input_tag,
        "output_tag": settings.output_tag,
        "input_dir": str(base_in),
        "output_dir": str(base_out),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_windows": int(sequence_shift["window_id"].nunique()) if not sequence_shift.empty else 0,
        "n_targets": int(sequence_shift["target_name"].nunique()) if not sequence_shift.empty else 0,
        "n_sequence_rows": int(len(seq_all)),
        "n_object_peak_stat_rows": int(len(object_stats_all)),
        "n_pair_order_rows": int(len(pair_order_all)),
        "does_not_rerun_bootstrap": settings.does_not_rerun_bootstrap,
        "does_not_rerun_mca_or_svd": settings.does_not_rerun_mca_or_svd,
        "does_not_rerun_peak_detector": settings.does_not_rerun_peak_detector,
        "physical_interpretation_included": settings.physical_interpretation_included,
        "sequence_scope": "target-conditioned high/mid/low score phases using precomputed bootstrap object peaks",
        "near_day_threshold": settings.near_day_threshold,
        "object_shift_report_threshold_days": settings.object_shift_report_threshold_days,
        "long_bootstrap_object_peak_table_saved": save_long,
    }
    (cross_out / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_md(cross_out / "V9_1_F_ALL_PAIRS_SEQUENCE_REGISTRY_A_SUMMARY.md", run_meta, window_summary, sequence_shift)

    print("[6/6] Update log")
    update_log = v91_root / "UPDATE_LOG_V9_1_F_ALL_PAIRS_SEQUENCE_REGISTRY_A.md"
    update_log.write_text(
        "# UPDATE_LOG V9.1_f_all_pairs_sequence_registry_a\n\n"
        "Purpose: add a read-only target-conditioned window-level sequence registry for V9.1_f_all_pairs_a.\n\n"
        "This layer does not rerun bootstrap, MCA/SVD, or peak detection. It merges each target's score phase with precomputed bootstrap object peaks, so each target high/low phase can be summarized as a full P/V/H/Je/Jw peak sequence rather than only its target-pair order.\n\n"
        "Interpretation boundary: sequences are target-conditioned bootstrap-space summaries; they are not physical regimes, direct-year types, or causal mechanisms.\n",
        encoding="utf-8",
    )
    print(f"Done. Outputs written to {base_out}")


if __name__ == "__main__":
    run_summarize_v9_1_f_all_pairs_sequence_registry_a(Path(__file__).resolve().parents[2])
