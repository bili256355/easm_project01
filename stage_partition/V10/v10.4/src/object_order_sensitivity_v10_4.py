from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

OBJECTS = ["P", "V", "H", "Je", "Jw"]
TAUS = [0, 2, 5]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _to_int_or_none(x: Any) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(round(float(x)))
    except Exception:
        return None


def _bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


def _month_day_from_day(day: Any) -> str:
    d = _to_int_or_none(day)
    if d is None:
        return ""
    # day 0 = Apr 1
    import datetime as _dt
    base = _dt.date(2001, 4, 1)
    return (base + _dt.timedelta(days=d)).strftime("%m-%d")


@dataclass
class Paths:
    bundle_root: Path
    v10_root: Path
    out_root: Path
    v10_1_lineage: Path
    v10_2_catalog: Path
    v10_2_mapping: Path
    v10_3_registry: Path
    v10_3_config: Path
    v10_3_lineage: Path
    v10_window_ref: Path


def build_paths(bundle_root: Path) -> Paths:
    v10_root = bundle_root.parent
    out_root = bundle_root / "outputs" / "object_order_sensitivity_v10_4"
    return Paths(
        bundle_root=bundle_root,
        v10_root=v10_root,
        out_root=out_root,
        v10_1_lineage=v10_root / "v10.1" / "outputs" / "joint_main_window_reproduce_v10_1" / "lineage" / "joint_main_window_lineage_v10_1.csv",
        v10_2_catalog=v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "cross_object" / "object_native_candidate_catalog_all_objects_v10_2.csv",
        v10_2_mapping=v10_root / "v10.2" / "outputs" / "object_native_peak_discovery_v10_2" / "lineage_mapping" / "object_candidate_to_joint_lineage_v10_2.csv",
        v10_3_registry=v10_root / "v10.3" / "outputs" / "peak_discovery_sensitivity_v10_3" / "cross_scope" / "candidate_registry_all_scopes_configs_v10_3.csv",
        v10_3_config=v10_root / "v10.3" / "outputs" / "peak_discovery_sensitivity_v10_3" / "cross_scope" / "sensitivity_config_grid_v10_3.csv",
        v10_3_lineage=v10_root / "v10.3" / "outputs" / "peak_discovery_sensitivity_v10_3" / "lineage_mapping" / "candidate_lineage_sensitivity_mapping_v10_3.csv",
        v10_window_ref=v10_root / "v10.3" / "outputs" / "peak_discovery_sensitivity_v10_3" / "lineage_mapping" / "v10_window_conditioned_selection_reference_used_v10_3.csv",
    )


def _lineage_id(row: pd.Series) -> str:
    strict_id = row.get("strict_accepted_window_id")
    if isinstance(strict_id, str) and strict_id.strip():
        return strict_id.strip()
    day = _to_int_or_none(row.get("candidate_day"))
    win = row.get("v6_1_window_id")
    win_s = str(win) if not pd.isna(win) else "WNA"
    return f"J{day:03d}_{win_s}" if day is not None else f"JUNK_{win_s}"


def prepare_lineages(lineage: pd.DataFrame) -> pd.DataFrame:
    out = lineage.copy()
    out["lineage_id"] = out.apply(_lineage_id, axis=1)
    out["lineage_day"] = out["candidate_day"].apply(_to_int_or_none)
    out["lineage_month_day"] = out["lineage_day"].apply(_month_day_from_day)
    out["lineage_status"] = out.get("lineage_status", "")
    out["lineage_strict_accepted_flag"] = out.get("strict_accepted_flag", False).apply(_bool)
    out["lineage_strict_accepted_window_id"] = out.get("strict_accepted_window_id", np.nan)
    keep = [
        "lineage_id", "lineage_day", "lineage_month_day", "lineage_status",
        "lineage_strict_accepted_flag", "lineage_strict_accepted_window_id",
        "v6_1_window_id", "v6_1_window_start", "v6_1_window_end", "v6_1_main_peak_day",
        "bootstrap_match_fraction", "bootstrap_strict_fraction", "bootstrap_near_fraction",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values("lineage_day").reset_index(drop=True)


def _candidate_priority(row: pd.Series, lineage_day: int, strict: bool) -> Tuple[int, float, float]:
    """Lower is better. Priority uses known V10 selected main when available, then distance, then support."""
    was_main = _bool(row.get("was_selected_as_v10_window_conditioned_main_peak"))
    dist_lineage = abs(_to_float(row.get("candidate_day"), 9999) - float(lineage_day))
    dist_strict = abs(_to_float(row.get("distance_to_nearest_strict_anchor"), 9999))
    match = -_to_float(row.get("bootstrap_match_fraction"), 0.0)
    if strict and was_main:
        return (0, dist_strict, match)
    if _to_int_or_none(row.get("nearest_joint_candidate_day")) == lineage_day:
        return (1, dist_lineage, match)
    return (2, dist_lineage, match)


def build_baseline_candidate_reference(
    lineages: pd.DataFrame,
    v10_2_catalog: pd.DataFrame,
    v10_2_mapping: pd.DataFrame,
    v10_window_ref: pd.DataFrame,
) -> pd.DataFrame:
    # merge support columns from catalog into mapping if absent
    cat = v10_2_catalog.rename(columns={"point_day": "candidate_day"}).copy()
    support_cols = ["object", "candidate_id", "candidate_day", "bootstrap_match_fraction", "bootstrap_strict_fraction", "bootstrap_near_fraction", "object_support_class"]
    cat_small = cat[[c for c in support_cols if c in cat.columns]].drop_duplicates()
    m = v10_2_mapping.copy()
    if "bootstrap_match_fraction" not in m.columns:
        m = m.merge(cat_small, on=["object", "candidate_id", "candidate_day"], how="left")
    refs: List[Dict[str, Any]] = []
    for _, lin in lineages.iterrows():
        lineage_id = lin["lineage_id"]
        lineage_day = int(lin["lineage_day"])
        strict = _bool(lin["lineage_strict_accepted_flag"])
        strict_win = lin.get("lineage_strict_accepted_window_id")
        for obj in OBJECTS:
            chosen: Optional[pd.Series] = None
            source = ""
            if strict and isinstance(strict_win, str) and strict_win.strip():
                wr = v10_window_ref[(v10_window_ref["window_id"] == strict_win) & (v10_window_ref["object"] == obj)]
                if not wr.empty:
                    sel_day = _to_int_or_none(wr.iloc[0].get("selected_peak_day"))
                    mm = m[(m["object"] == obj) & (_to_int_or_none_series(m["candidate_day"]) == sel_day)] if sel_day is not None else pd.DataFrame()
                    if not mm.empty:
                        chosen = mm.iloc[0]
                    else:
                        # fallback create pseudo from v10 window ref
                        row = wr.iloc[0].to_dict()
                        row.update({
                            "object": obj,
                            "candidate_id": row.get("selected_candidate_id"),
                            "candidate_day": sel_day,
                            "candidate_date": _month_day_from_day(sel_day),
                            "was_selected_as_v10_window_conditioned_main_peak": True,
                            "v10_window_id_if_selected": strict_win,
                            "object_support_class": row.get("support_class", ""),
                        })
                        chosen = pd.Series(row)
                    source = "V10_WINDOW_CONDITIONED_MAIN"
            if chosen is None:
                cand = m[m["object"] == obj].copy()
                if not cand.empty:
                    cand["__priority"] = cand.apply(lambda r: _candidate_priority(r, lineage_day, strict=False), axis=1)
                    cand = cand.sort_values("__priority")
                    chosen = cand.iloc[0]
                    source = "OBJECT_NATIVE_NEAREST_TO_LINEAGE"
            if chosen is None:
                refs.append({
                    "lineage_id": lineage_id, "lineage_day": lineage_day, "object": obj,
                    "baseline_candidate_available": False,
                    "baseline_candidate_source": "NO_BASELINE_OBJECT_CANDIDATE",
                })
            else:
                cand_day = _to_int_or_none(chosen.get("candidate_day"))
                refs.append({
                    "lineage_id": lineage_id,
                    "lineage_day": lineage_day,
                    "lineage_status": lin.get("lineage_status"),
                    "lineage_strict_accepted_flag": strict,
                    "lineage_strict_accepted_window_id": strict_win,
                    "object": obj,
                    "baseline_candidate_available": cand_day is not None,
                    "baseline_candidate_source": source,
                    "baseline_candidate_id": chosen.get("candidate_id"),
                    "baseline_candidate_day": cand_day,
                    "baseline_candidate_date": _month_day_from_day(cand_day),
                    "baseline_object_support_class": chosen.get("object_support_class", ""),
                    "baseline_nearest_joint_candidate_day": chosen.get("nearest_joint_candidate_day"),
                    "baseline_nearest_joint_lineage_status": chosen.get("nearest_joint_lineage_status"),
                    "baseline_v10_window_id_if_selected": chosen.get("v10_window_id_if_selected"),
                    "baseline_was_v10_window_conditioned_main_peak": _bool(chosen.get("was_selected_as_v10_window_conditioned_main_peak")),
                    "baseline_distance_to_lineage_day": (abs(cand_day - lineage_day) if cand_day is not None else np.nan),
                })
    return pd.DataFrame(refs)


def _to_int_or_none_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def _scope_for_object(obj: str) -> str:
    return f"{obj}_only"


def _assignment_class_from_distance(config_id: str, source: str, dist_base: Optional[float], dist_lineage: Optional[float], chosen_source: str) -> str:
    if chosen_source == "none":
        return "NO_ASSIGNED_OBJECT_CANDIDATE"
    if config_id == "BASELINE":
        if source == "V10_WINDOW_CONDITIONED_MAIN":
            return "BASELINE_MAIN_CANDIDATE"
        return "BASELINE_OBJECT_NATIVE_CANDIDATE"
    if chosen_source == "baseline":
        if dist_base is not None and dist_base == 0:
            return "PERTURBED_SAME_AS_BASELINE_CANDIDATE_DAY"
        if dist_base is not None and dist_base <= 2:
            return "PERTURBED_MATCH_TO_BASELINE_CANDIDATE_STRICT"
        if dist_base is not None and dist_base <= 5:
            return "PERTURBED_MATCH_TO_BASELINE_CANDIDATE_MATCH"
        return "PERTURBED_MATCH_TO_BASELINE_CANDIDATE_NEAR"
    if chosen_source == "lineage":
        return "CANDIDATE_REASSIGNMENT_TO_LINEAGE_NEAREST"
    return "NEW_CANDIDATE_NEAR_LINEAGE"


def build_assignment_table(
    lineages: pd.DataFrame,
    configs: pd.DataFrame,
    registry: pd.DataFrame,
    lineage_map: pd.DataFrame,
    baseline_ref: pd.DataFrame,
) -> pd.DataFrame:
    # enrich registry with candidate provenance map
    prov_cols = [
        "scope", "object", "config_id", "candidate_id", "perturbed_candidate_day",
        "nearest_joint_candidate_day", "distance_to_nearest_joint_candidate",
        "nearest_joint_lineage_status", "nearest_joint_derived_window_id",
        "nearest_strict_accepted_window_id", "distance_to_nearest_strict_anchor",
        "matched_object_native_baseline_candidate_id", "matched_object_native_baseline_candidate_day",
        "object_support_class", "was_v10_window_conditioned_main_peak", "v10_window_id_if_main",
        "candidate_provenance_class",
    ]
    lm = lineage_map[[c for c in prov_cols if c in lineage_map.columns]].copy()
    cand_all = registry.copy()
    cand_all = cand_all.merge(
        lm,
        on=["scope", "object", "config_id", "candidate_id"],
        how="left",
        suffixes=("", "_lm"),
    )
    if "point_day" in cand_all.columns:
        cand_all["candidate_day"] = cand_all["point_day"].apply(_to_int_or_none)
    elif "perturbed_candidate_day" in cand_all.columns:
        cand_all["candidate_day"] = cand_all["perturbed_candidate_day"].apply(_to_int_or_none)
    records: List[Dict[str, Any]] = []
    for _, cfg in configs.iterrows():
        cfg_id = str(cfg["config_id"])
        ref_cfg = str(cfg.get("reference_config_id", "")) if "reference_config_id" in cfg.index else ""
        if not ref_cfg or ref_cfg == "nan":
            # V10.3 grid lacks reference_config_id in older versions: smooth5 internal uses SMOOTH_INPUT_5D, others BASELINE.
            ref_cfg = "SMOOTH_INPUT_5D" if cfg_id.startswith("SMOOTH5_") else "BASELINE"
        for _, lin in lineages.iterrows():
            lineage_id = lin["lineage_id"]
            lineage_day = int(lin["lineage_day"])
            bref_sub = baseline_ref[baseline_ref["lineage_id"] == lineage_id]
            for obj in OBJECTS:
                bref = bref_sub[bref_sub["object"] == obj]
                bref_row = bref.iloc[0] if not bref.empty else pd.Series(dtype=object)
                base_day = _to_int_or_none(bref_row.get("baseline_candidate_day"))
                base_source = str(bref_row.get("baseline_candidate_source", ""))
                scope = _scope_for_object(obj)
                cc = cand_all[(cand_all["scope"] == scope) & (cand_all["object"] == obj) & (cand_all["config_id"] == cfg_id)].copy()
                assigned = None
                chosen_source = "none"
                dist_base = None
                dist_lineage = None
                if not cc.empty:
                    cc["__dist_lineage"] = (pd.to_numeric(cc["candidate_day"], errors="coerce") - lineage_day).abs()
                    if base_day is not None:
                        cc["__dist_base"] = (pd.to_numeric(cc["candidate_day"], errors="coerce") - base_day).abs()
                        nearest_base = cc.sort_values(["__dist_base", "__dist_lineage", "registry_rank"]).iloc[0]
                        db = _to_float(nearest_base["__dist_base"])
                        if db <= 8:
                            assigned = nearest_base
                            chosen_source = "baseline"
                            dist_base = db
                            dist_lineage = _to_float(nearest_base["__dist_lineage"])
                    if assigned is None:
                        nearest_lineage = cc.sort_values(["__dist_lineage", "registry_rank"]).iloc[0]
                        dl = _to_float(nearest_lineage["__dist_lineage"])
                        # lineages such as W045 can legitimately have front candidates ~10d earlier; use 12d as broad assignment radius.
                        if dl <= 12:
                            assigned = nearest_lineage
                            chosen_source = "lineage"
                            dist_lineage = dl
                            dist_base = (abs(_to_float(nearest_lineage["candidate_day"]) - base_day) if base_day is not None else np.nan)
                assignment_class = _assignment_class_from_distance(cfg_id, base_source, dist_base, dist_lineage, chosen_source)
                rec = {
                    "lineage_id": lineage_id,
                    "lineage_day": lineage_day,
                    "lineage_month_day": _month_day_from_day(lineage_day),
                    "lineage_status": lin.get("lineage_status"),
                    "lineage_strict_accepted_flag": _bool(lin.get("lineage_strict_accepted_flag")),
                    "lineage_strict_accepted_window_id": lin.get("lineage_strict_accepted_window_id"),
                    "config_id": cfg_id,
                    "reference_config_id": ref_cfg,
                    "sensitivity_group": cfg.get("sensitivity_group"),
                    "changed_factor": cfg.get("changed_factor"),
                    "changed_value": cfg.get("changed_value"),
                    "input_source": cfg.get("input_source"),
                    "object": obj,
                    "baseline_candidate_id": bref_row.get("baseline_candidate_id"),
                    "baseline_candidate_day": base_day,
                    "baseline_candidate_source": base_source,
                    "baseline_object_support_class": bref_row.get("baseline_object_support_class"),
                    "assigned_candidate_available": assigned is not None,
                    "assignment_class": assignment_class,
                    "candidate_selection_basis": chosen_source,
                }
                if assigned is not None:
                    cand_day = _to_int_or_none(assigned.get("candidate_day"))
                    rec.update({
                        "assigned_candidate_id": assigned.get("candidate_id"),
                        "assigned_candidate_day": cand_day,
                        "assigned_candidate_date": _month_day_from_day(cand_day),
                        "assigned_peak_score": assigned.get("peak_score"),
                        "assigned_registry_rank": assigned.get("registry_rank"),
                        "object_support_class": assigned.get("object_support_class"),
                        "nearest_joint_candidate_day": assigned.get("nearest_joint_candidate_day"),
                        "nearest_joint_lineage_status": assigned.get("nearest_joint_lineage_status"),
                        "nearest_joint_derived_window_id": assigned.get("nearest_joint_derived_window_id"),
                        "nearest_strict_accepted_window_id": assigned.get("nearest_strict_accepted_window_id"),
                        "was_v10_window_conditioned_main_peak": _bool(assigned.get("was_v10_window_conditioned_main_peak")),
                        "v10_window_id_if_main": assigned.get("v10_window_id_if_main"),
                        "candidate_provenance_class": assigned.get("candidate_provenance_class"),
                        "distance_to_lineage_day": dist_lineage,
                        "distance_to_baseline_object_candidate": dist_base,
                    })
                    if base_day is None:
                        shift_class = "NO_BASELINE_CANDIDATE"
                    else:
                        d = abs(cand_day - base_day)
                        if d == 0:
                            shift_class = "SAME_DAY"
                        elif d <= 2:
                            shift_class = "STRICT_SHIFT_LE_2D"
                        elif d <= 5:
                            shift_class = "MATCH_SHIFT_LE_5D"
                        elif d <= 8:
                            shift_class = "NEAR_SHIFT_LE_8D"
                        else:
                            shift_class = "REASSIGNED_GT_8D"
                    rec["candidate_shift_class"] = shift_class
                else:
                    rec.update({
                        "assigned_candidate_id": np.nan,
                        "assigned_candidate_day": np.nan,
                        "assigned_candidate_date": "",
                        "assigned_peak_score": np.nan,
                        "assigned_registry_rank": np.nan,
                        "object_support_class": "",
                        "nearest_joint_candidate_day": np.nan,
                        "nearest_joint_lineage_status": "",
                        "nearest_joint_derived_window_id": "",
                        "nearest_strict_accepted_window_id": "",
                        "was_v10_window_conditioned_main_peak": False,
                        "v10_window_id_if_main": "",
                        "candidate_provenance_class": "NO_ASSIGNED_CANDIDATE",
                        "distance_to_lineage_day": np.nan,
                        "distance_to_baseline_object_candidate": np.nan,
                        "candidate_shift_class": "MISSING_ASSIGNED_CANDIDATE",
                    })
                records.append(rec)
    return pd.DataFrame(records)


def _order(day_a: Any, day_b: Any, tau: int) -> str:
    da = _to_float(day_a)
    db = _to_float(day_b)
    if np.isnan(da) or np.isnan(db):
        return "MISSING"
    if da < db - tau:
        return "A_BEFORE_B"
    if db < da - tau:
        return "B_BEFORE_A"
    return "NEAR_TIE"


def build_pairwise_orders(assign: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["lineage_id", "lineage_day", "config_id"]
    ref_lookup = assign.set_index(["lineage_id", "config_id", "object"])
    rows: List[Dict[str, Any]] = []
    for (lineage_id, lineage_day, cfg_id), sub in assign.groupby(key_cols, dropna=False):
        meta = sub.iloc[0]
        ref_cfg = str(meta.get("reference_config_id", "BASELINE"))
        for a, b in itertools.combinations(OBJECTS, 2):
            ra = sub[sub["object"] == a]
            rb = sub[sub["object"] == b]
            if ra.empty or rb.empty:
                continue
            ra = ra.iloc[0]
            rb = rb.iloc[0]
            ref_a = None
            ref_b = None
            try:
                ref_a = ref_lookup.loc[(lineage_id, ref_cfg, a)]
                ref_b = ref_lookup.loc[(lineage_id, ref_cfg, b)]
            except Exception:
                try:
                    ref_a = ref_lookup.loc[(lineage_id, "BASELINE", a)]
                    ref_b = ref_lookup.loc[(lineage_id, "BASELINE", b)]
                except Exception:
                    ref_a, ref_b = None, None
            rec = {
                "lineage_id": lineage_id,
                "lineage_day": lineage_day,
                "lineage_status": meta.get("lineage_status"),
                "lineage_strict_accepted_flag": meta.get("lineage_strict_accepted_flag"),
                "lineage_strict_accepted_window_id": meta.get("lineage_strict_accepted_window_id"),
                "config_id": cfg_id,
                "reference_config_id": ref_cfg,
                "sensitivity_group": meta.get("sensitivity_group"),
                "changed_factor": meta.get("changed_factor"),
                "changed_value": meta.get("changed_value"),
                "input_source": meta.get("input_source"),
                "object_a": a,
                "object_b": b,
                "day_a": ra.get("assigned_candidate_day"),
                "day_b": rb.get("assigned_candidate_day"),
                "candidate_id_a": ra.get("assigned_candidate_id"),
                "candidate_id_b": rb.get("assigned_candidate_id"),
                "assignment_class_a": ra.get("assignment_class"),
                "assignment_class_b": rb.get("assignment_class"),
                "support_class_a": ra.get("object_support_class"),
                "support_class_b": rb.get("object_support_class"),
                "candidate_provenance_class_a": ra.get("candidate_provenance_class"),
                "candidate_provenance_class_b": rb.get("candidate_provenance_class"),
                "missing_flag": (not _bool(ra.get("assigned_candidate_available"))) or (not _bool(rb.get("assigned_candidate_available"))),
            }
            da = _to_float(rec["day_a"])
            db = _to_float(rec["day_b"])
            rec["day_gap_a_minus_b"] = (da - db) if not (np.isnan(da) or np.isnan(db)) else np.nan
            for tau in TAUS:
                o = _order(rec["day_a"], rec["day_b"], tau)
                rec[f"order_tau{tau}"] = o
                if ref_a is not None and ref_b is not None:
                    bo = _order(ref_a.get("assigned_candidate_day"), ref_b.get("assigned_candidate_day"), tau)
                    rec[f"reference_order_tau{tau}"] = bo
                    rec[f"order_changed_tau{tau}"] = (o != bo) and o != "MISSING" and bo != "MISSING"
                    rec[f"near_tie_flag_tau{tau}"] = o == "NEAR_TIE"
                else:
                    rec[f"reference_order_tau{tau}"] = "MISSING_REFERENCE"
                    rec[f"order_changed_tau{tau}"] = False
                    rec[f"near_tie_flag_tau{tau}"] = False
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["lineage_id", "lineage_day", "lineage_status", "lineage_strict_accepted_flag", "object_a", "object_b", "sensitivity_group"]
    for keys, sub in pairwise.groupby(group_cols, dropna=False):
        keyd = dict(zip(group_cols, keys))
        n = len(sub)
        valid = sub[~sub["missing_flag"].astype(bool)].copy()
        n_valid = len(valid)
        baseline_orders = valid[valid["config_id"] == valid["reference_config_id"]]["order_tau2"].dropna().unique().tolist()
        baseline_order = baseline_orders[0] if baseline_orders else (valid["reference_order_tau2"].dropna().iloc[0] if n_valid else "MISSING")
        same = ((valid["order_tau2"] == valid["reference_order_tau2"]) & (valid["order_tau2"] != "MISSING")).sum() if n_valid else 0
        changed = valid["order_changed_tau2"].astype(bool).sum() if n_valid else 0
        near = (valid["order_tau2"] == "NEAR_TIE").sum() if n_valid else 0
        rev = 0
        if n_valid:
            # reversal only if reference and current are opposite before/after, not near-tie.
            rev = (((valid["reference_order_tau2"] == "A_BEFORE_B") & (valid["order_tau2"] == "B_BEFORE_A")) |
                   ((valid["reference_order_tau2"] == "B_BEFORE_A") & (valid["order_tau2"] == "A_BEFORE_B"))).sum()
        if n_valid == 0:
            status = "INSUFFICIENT_CANDIDATES"
        elif rev > 0:
            status = "ORDER_REVERSAL_DETECTED"
        elif near / max(n_valid, 1) >= 0.5:
            status = "ORDER_NEAR_TIE_DOMINATED"
        elif same / max(n_valid, 1) >= 0.8:
            status = "ORDER_STABLE"
        else:
            status = "ORDER_CONFIG_SENSITIVE"
        rows.append({
            **keyd,
            "n_configs": n,
            "n_valid_configs": n_valid,
            "baseline_order_tau2": baseline_order,
            "n_same_as_reference_tau2": int(same),
            "n_order_changed_tau2": int(changed),
            "n_reversed_order_tau2": int(rev),
            "n_near_tie_tau2": int(near),
            "frac_same_order_tau2": same / n_valid if n_valid else np.nan,
            "frac_reversed_order_tau2": rev / n_valid if n_valid else np.nan,
            "frac_near_tie_tau2": near / n_valid if n_valid else np.nan,
            "frac_missing": (n - n_valid) / n if n else np.nan,
            "order_stability_class_tau2": status,
        })
    return pd.DataFrame(rows)


def _sequence_for_group(sub: pd.DataFrame, tau: int = 2) -> Tuple[str, str, int, int]:
    ok = sub[sub["assigned_candidate_available"].astype(bool)].copy()
    if ok.empty:
        return "", "", 0, 0
    ok["assigned_candidate_day"] = pd.to_numeric(ok["assigned_candidate_day"], errors="coerce")
    ok = ok.sort_values(["assigned_candidate_day", "object"])
    seq = " < ".join(f"{r.object}@{int(r.assigned_candidate_day)}" for r in ok.itertuples())
    groups: List[List[str]] = []
    current: List[str] = []
    current_start = None
    for r in ok.itertuples():
        day = int(r.assigned_candidate_day)
        token = f"{r.object}@{day}"
        if current_start is None or abs(day - current_start) <= tau:
            current.append(token)
            if current_start is None:
                current_start = day
        else:
            groups.append(current)
            current = [token]
            current_start = day
    if current:
        groups.append(current)
    grouped = " < ".join("≈".join(g) for g in groups)
    return seq, grouped, len(ok), len(groups)


def build_sequences(assign: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ref_seq_lookup: Dict[Tuple[str, str], str] = {}
    # first pass computes all sequences
    tmp = []
    for (lineage_id, lineage_day, cfg_id), sub in assign.groupby(["lineage_id", "lineage_day", "config_id"], dropna=False):
        meta = sub.iloc[0]
        seq, grouped, nobj, ng = _sequence_for_group(sub, tau=2)
        rec = {
            "lineage_id": lineage_id,
            "lineage_day": lineage_day,
            "lineage_status": meta.get("lineage_status"),
            "lineage_strict_accepted_flag": meta.get("lineage_strict_accepted_flag"),
            "config_id": cfg_id,
            "reference_config_id": meta.get("reference_config_id"),
            "sensitivity_group": meta.get("sensitivity_group"),
            "changed_factor": meta.get("changed_factor"),
            "changed_value": meta.get("changed_value"),
            "input_source": meta.get("input_source"),
            "assigned_objects": ";".join(sub[sub["assigned_candidate_available"].astype(bool)]["object"].tolist()),
            "n_assigned_objects": nobj,
            "object_order_sequence": seq,
            "object_order_sequence_tau2_grouped": grouped,
            "n_near_tie_groups_tau2": ng,
        }
        tmp.append(rec)
        ref_seq_lookup[(lineage_id, cfg_id)] = grouped
    for rec in tmp:
        ref_cfg = str(rec.get("reference_config_id", "BASELINE"))
        ref_grouped = ref_seq_lookup.get((rec["lineage_id"], ref_cfg), ref_seq_lookup.get((rec["lineage_id"], "BASELINE"), ""))
        rec["reference_order_sequence_tau2_grouped"] = ref_grouped
        rec["sequence_changed_from_reference_tau2"] = rec["object_order_sequence_tau2_grouped"] != ref_grouped
        if rec["n_assigned_objects"] < 2:
            change_type = "INSUFFICIENT_ASSIGNED_OBJECTS"
        elif not rec["sequence_changed_from_reference_tau2"]:
            change_type = "UNCHANGED"
        elif rec["n_assigned_objects"] != (ref_grouped.count("@") if ref_grouped else 0):
            change_type = "ASSIGNED_OBJECT_SET_CHANGED"
        else:
            change_type = "ORDER_OR_NEAR_TIE_GROUP_CHANGED"
        rec["sequence_change_type"] = change_type
        rows.append(rec)
    return pd.DataFrame(rows)


def build_reversal_inventory(pairwise: pd.DataFrame) -> pd.DataFrame:
    rev = pairwise[
        (((pairwise["reference_order_tau2"] == "A_BEFORE_B") & (pairwise["order_tau2"] == "B_BEFORE_A")) |
         ((pairwise["reference_order_tau2"] == "B_BEFORE_A") & (pairwise["order_tau2"] == "A_BEFORE_B")))
    ].copy()
    if rev.empty:
        cols = list(pairwise.columns) + ["reason_hint"]
        return pd.DataFrame(columns=cols)
    def reason(row: pd.Series) -> str:
        if "REASSIGNMENT" in str(row.get("assignment_class_a")) or "REASSIGNMENT" in str(row.get("assignment_class_b")):
            return "candidate_reassignment"
        if str(row.get("sensitivity_group")) in {"smooth_input", "smooth5_detector_width"}:
            return str(row.get("sensitivity_group"))
        if str(row.get("sensitivity_group")) == "detector_width":
            return "detector_width_candidate_shift"
        return "candidate_shift_or_near_tie_change"
    rev["reason_hint"] = rev.apply(reason, axis=1)
    return rev


def write_summary_md(out_root: Path, meta: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("# V10.4 Object-to-Object Order Sensitivity Summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("V10.4 audits object-to-object peak timing order near each joint lineage / accepted window. It reads V10.1, V10.2, and V10.3 outputs and does not rerun peak discovery.")
    lines.append("")
    lines.append("## Run status")
    for k in ["status", "n_lineages", "n_configs", "n_assignment_rows", "n_pairwise_rows", "n_reversal_rows"]:
        lines.append(f"- {k}: {meta.get(k)}")
    lines.append("")
    lines.append("## Key counts")
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("This is a timing/order stability audit only. It does not establish causality, physical mechanism, or whether a non-strict candidate should enter the main result.")
    lines.append("")
    (out_root / "OBJECT_ORDER_SENSITIVITY_V10_4_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_object_order_sensitivity_v10_4(bundle_root: Path) -> None:
    paths = build_paths(bundle_root)
    for sub in ["assignment", "order", "audit"]:
        _ensure_dir(paths.out_root / sub)
    _ensure_dir(bundle_root / "logs")
    started = _now()

    lineage_raw = _safe_read_csv(paths.v10_1_lineage)
    v10_2_catalog = _safe_read_csv(paths.v10_2_catalog)
    v10_2_mapping = _safe_read_csv(paths.v10_2_mapping)
    registry = _safe_read_csv(paths.v10_3_registry)
    configs = _safe_read_csv(paths.v10_3_config)
    v10_3_lineage = _safe_read_csv(paths.v10_3_lineage)
    v10_window_ref = _safe_read_csv(paths.v10_window_ref)

    # If the grid lacks reference_config_id, add it here using V10.3 convention.
    if "reference_config_id" not in configs.columns:
        configs["reference_config_id"] = configs["config_id"].apply(lambda c: "SMOOTH_INPUT_5D" if str(c).startswith("SMOOTH5_") else "BASELINE")

    lineages = prepare_lineages(lineage_raw)
    baseline_ref = build_baseline_candidate_reference(lineages, v10_2_catalog, v10_2_mapping, v10_window_ref)
    assignment = build_assignment_table(lineages, configs, registry, v10_3_lineage, baseline_ref)
    pairwise = build_pairwise_orders(assignment)
    pair_summary = summarize_pairwise(pairwise)
    sequences = build_sequences(assignment)
    reversals = build_reversal_inventory(pairwise)

    # Write outputs.
    baseline_ref.to_csv(paths.out_root / "assignment" / "object_baseline_candidate_reference_by_lineage_v10_4.csv", index=False)
    assignment.to_csv(paths.out_root / "assignment" / "object_candidate_assignment_by_lineage_config_v10_4.csv", index=False)
    pairwise.to_csv(paths.out_root / "order" / "object_pairwise_order_by_lineage_config_v10_4.csv", index=False)
    pair_summary.to_csv(paths.out_root / "order" / "object_pairwise_order_stability_summary_v10_4.csv", index=False)
    sequences.to_csv(paths.out_root / "order" / "object_order_sequence_by_lineage_config_v10_4.csv", index=False)
    reversals.to_csv(paths.out_root / "order" / "object_order_reversal_inventory_v10_4.csv", index=False)

    input_inventory = pd.DataFrame([
        {"input_name": "v10_1_lineage", "path": str(paths.v10_1_lineage), "exists": paths.v10_1_lineage.exists(), "n_rows": len(lineage_raw)},
        {"input_name": "v10_2_catalog", "path": str(paths.v10_2_catalog), "exists": paths.v10_2_catalog.exists(), "n_rows": len(v10_2_catalog)},
        {"input_name": "v10_2_mapping", "path": str(paths.v10_2_mapping), "exists": paths.v10_2_mapping.exists(), "n_rows": len(v10_2_mapping)},
        {"input_name": "v10_3_registry", "path": str(paths.v10_3_registry), "exists": paths.v10_3_registry.exists(), "n_rows": len(registry)},
        {"input_name": "v10_3_config", "path": str(paths.v10_3_config), "exists": paths.v10_3_config.exists(), "n_rows": len(configs)},
        {"input_name": "v10_3_lineage", "path": str(paths.v10_3_lineage), "exists": paths.v10_3_lineage.exists(), "n_rows": len(v10_3_lineage)},
        {"input_name": "v10_window_ref", "path": str(paths.v10_window_ref), "exists": paths.v10_window_ref.exists(), "n_rows": len(v10_window_ref)},
    ])
    input_inventory.to_csv(paths.out_root / "audit" / "input_inventory_v10_4.csv", index=False)

    status_counts = pair_summary["order_stability_class_tau2"].value_counts(dropna=False).to_dict() if not pair_summary.empty else {}
    summary = {
        "n_lineages": int(len(lineages)),
        "n_configs": int(configs["config_id"].nunique()),
        "n_assignment_rows": int(len(assignment)),
        "n_pairwise_rows": int(len(pairwise)),
        "n_pairwise_summary_rows": int(len(pair_summary)),
        "n_sequence_rows": int(len(sequences)),
        "n_reversal_rows_tau2": int(len(reversals)),
        "pairwise_order_stability_class_counts_tau2": status_counts,
        "n_missing_assignments": int((~assignment["assigned_candidate_available"].astype(bool)).sum()),
        "n_candidate_reassignments": int(assignment["assignment_class"].astype(str).str.contains("REASSIGNMENT", na=False).sum()),
        "n_sequence_changed_from_reference_tau2": int(sequences["sequence_changed_from_reference_tau2"].astype(bool).sum()) if not sequences.empty else 0,
    }
    meta = {
        "status": "success",
        "started_at": started,
        "finished_at": _now(),
        "bundle_root": str(bundle_root),
        "v10_root": str(paths.v10_root),
        "does_not_rerun_peak_discovery": True,
        "does_not_perform_physical_interpretation": True,
        "uses_v10_1_v10_2_v10_3_outputs": True,
        **summary,
    }
    (paths.out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (paths.out_root / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(paths.out_root, meta, summary)
    (bundle_root / "logs" / "last_run.txt").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run_object_order_sensitivity_v10_4(Path(__file__).resolve().parents[1])
