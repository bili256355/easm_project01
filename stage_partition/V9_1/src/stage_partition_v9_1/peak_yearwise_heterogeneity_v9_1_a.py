"""
V9.1 peak yearwise heterogeneity audit.

Purpose
-------
V9.1 is a read-only audit branch for the V9 peak-only results.  It tests the
hypothesis that weak/unstable V9 peak order can arise because the multi-year
sample contains different year-types, different candidate-event modes, or
object-specific weak/flat peak responses.

Method boundary
---------------
- V9.1 does not modify V9 source files.
- V9.1 does not overwrite V9 outputs.
- V9.1 does not redefine the V9/V7 peak detector.
- V9.1 does not include state, growth, pre-post process, or physical mechanism
  interpretation.
- V9.1 outputs heterogeneity-audit evidence only.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

V91_VERSION = "v9_1_peak_yearwise_heterogeneity_a"
OUTPUT_TAG = "peak_yearwise_heterogeneity_v9_1_a"
PARENT_V9_TAG = "peak_all_windows_v9_a"
DEFAULT_WINDOWS = ["W045", "W081", "W113", "W160"]
EXCLUDED_WINDOWS = ["W135"]
OBJECTS = ["P", "V", "H", "Je", "Jw"]
PAIR_OBJECTS_ORDERED = sorted(OBJECTS)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _v91_stage_root(v91_root: Path) -> Path:
    return v91_root.parent


def _import_v7_module(stage_root: Path):
    v7_src = stage_root / "V7" / "src"
    if not v7_src.exists():
        raise FileNotFoundError(
            f"Cannot find V7 source directory: {v7_src}. V9.1 requires the existing V7 code tree."
        )
    if str(v7_src) not in sys.path:
        sys.path.insert(0, str(v7_src))
    from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as v7multi
    return v7multi


def _make_v7_cfg_for_v91(v7multi):
    cfg = v7multi.MultiWinConfig.from_env()
    # Match V9 all-significant-window selection unless user overrides.
    cfg.window_mode = "list"
    cfg.target_windows = ",".join(DEFAULT_WINDOWS)
    if os.environ.get("V9_1_TARGET_WINDOWS"):
        cfg.target_windows = os.environ["V9_1_TARGET_WINDOWS"].strip()
    if os.environ.get("V9_1_WINDOW_MODE"):
        cfg.window_mode = os.environ["V9_1_WINDOW_MODE"].strip().lower()
    if os.environ.get("V9_1_SMOOTHED_FIELDS"):
        cfg.smoothed_fields_path = os.environ["V9_1_SMOOTHED_FIELDS"]
    if os.environ.get("V9_1_PEAK_ACCEPTED_WINDOW_REGISTRY"):
        cfg.accepted_window_registry = os.environ["V9_1_PEAK_ACCEPTED_WINDOW_REGISTRY"]
        cfg.window_source = "registry"
    if os.environ.get("V9_1_LOG_EVERY"):
        cfg.log_every_bootstrap = int(os.environ["V9_1_LOG_EVERY"])
    # V9.1 does not need V7 bootstrap inside helper calls.
    cfg.bootstrap_n = int(os.environ.get("V9_1_INTERNAL_BOOTSTRAP_N", "1"))
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    return cfg


def _default_smoothed_path(v91_root: Path) -> Path:
    # v91_root = D:/easm_project01/stage_partition/V9_1 => project root is parents[1]
    return v91_root.parents[1] / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _load_v9_reference_tables(stage_root: Path) -> Dict[str, pd.DataFrame]:
    base = stage_root / "V9" / "outputs" / PARENT_V9_TAG
    cross = base / "cross_window"
    out = {
        "accepted_windows": _read_csv_if_exists(cross / "accepted_windows_used_v9_a.csv"),
        "object_peak_registry": _read_csv_if_exists(cross / "cross_window_object_peak_registry.csv"),
        "pairwise_peak_order": _read_csv_if_exists(cross / "cross_window_pairwise_peak_order.csv"),
        "pairwise_peak_synchrony": _read_csv_if_exists(cross / "cross_window_pairwise_peak_synchrony.csv"),
        "bootstrap_all": _read_csv_if_exists(cross / "bootstrap_selected_peak_days_all_windows.csv"),
        "timing_resolution": _read_csv_if_exists(cross / "timing_resolution_audit_all_windows.csv"),
    }
    return out


def _build_profiles(v91_root: Path, stage_root: Path, v7multi, cfg) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], Optional[np.ndarray], pd.DataFrame, Path]:
    smoothed = Path(cfg.smoothed_fields_path) if getattr(cfg, "smoothed_fields_path", "") else _default_smoothed_path(v91_root)
    if not smoothed.exists():
        smoothed = _default_smoothed_path(v91_root)
    if not smoothed.exists():
        raise FileNotFoundError(
            f"smoothed_fields.npz not found: {smoothed}. Set V9_1_SMOOTHED_FIELDS to the correct file."
        )
    fields, audit = v7multi.clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for spec in v7multi.clean.OBJECT_SPECS:
        arr = v7multi.clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (prof, target_lat, weights)
    return profiles, years, audit, smoothed


def _select_peak_for_profile(prof_by_year: np.ndarray, scope, cfg, v7multi, object_name: str) -> Tuple[float, float, pd.DataFrame]:
    """Run the original V7 detector on an arbitrary year subset and select main peak."""
    state = v7multi._raw_state_matrix_v7z_from_year_cube(prof_by_year)
    _scores, cand = v7multi._run_original_v7z_detector_for_profile(state, cfg, scope, object_name)
    if cand is None or cand.empty:
        return np.nan, np.nan, cand if cand is not None else pd.DataFrame()
    try:
        sel = v7multi._select_main_candidate(cand.copy(), scope)
        peak_day = float(sel.get("selected_peak_day", sel.get("peak_day", np.nan)).iloc[0])
        score = float(sel.get("selected_peak_score", sel.get("peak_score", np.nan)).iloc[0])
    except Exception:
        # Conservative fallback to V7 boot selector semantics.
        peak_day = float(v7multi._select_boot_candidate_day(cand.copy(), scope))
        row = cand.loc[cand["peak_day"].astype(float) == float(peak_day)].head(1)
        score = float(row["peak_score"].iloc[0]) if not row.empty and "peak_score" in row.columns else np.nan
    return peak_day, score, cand.copy()


def _top_candidate_info(cand: pd.DataFrame, rank: int) -> Tuple[float, float]:
    if cand is None or cand.empty:
        return np.nan, np.nan
    c = cand.copy()
    score_col = "peak_score" if "peak_score" in c.columns else ("score" if "score" in c.columns else None)
    if score_col is None or "peak_day" not in c.columns:
        return np.nan, np.nan
    c = c.sort_values(score_col, ascending=False).reset_index(drop=True)
    if len(c) < rank:
        return np.nan, np.nan
    r = c.iloc[rank - 1]
    return float(r["peak_day"]), float(r[score_col])


def _make_yearwise_registry(profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], years: Optional[np.ndarray], scope, cfg, v7multi) -> pd.DataFrame:
    rows: List[dict] = []
    ny = next(iter(profiles.values()))[0].shape[0]
    year_vals = list(years) if years is not None and len(years) == ny else list(range(ny))
    for iy in range(ny):
        for obj, (prof, _lat, _w) in profiles.items():
            one = prof[[iy], ...]
            peak_day, peak_score, cand = _select_peak_for_profile(one, scope, cfg, v7multi, obj)
            r1_day, r1_score = _top_candidate_info(cand, 1)
            r2_day, r2_score = _top_candidate_info(cand, 2)
            gap = r1_score - r2_score if np.isfinite(r1_score) and np.isfinite(r2_score) else np.nan
            weak = False
            ambiguous = False
            if cand is None or cand.empty or not np.isfinite(peak_day):
                weak = True
                ambiguous = True
            elif np.isfinite(gap) and abs(gap) <= 1.0e-9:
                ambiguous = True
            rows.append({
                "window_id": scope.window_id,
                "year_index": iy,
                "year": year_vals[iy],
                "object": obj,
                "yearwise_peak_day": peak_day,
                "yearwise_peak_score": peak_score,
                "candidate_rank1_day": r1_day,
                "candidate_rank1_score": r1_score,
                "candidate_rank2_day": r2_day,
                "candidate_rank2_score": r2_score,
                "rank1_rank2_score_gap": gap,
                "weak_year_flag": bool(weak),
                "ambiguous_year_peak_flag": bool(ambiguous),
                "method_role": "single_year_peak_diagnostic_not_main_result",
            })
    return pd.DataFrame(rows)


def _jackknife_influence(profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], years: Optional[np.ndarray], scope, cfg, v7multi, v9_selection: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    ny = next(iter(profiles.values()))[0].shape[0]
    year_vals = list(years) if years is not None and len(years) == ny else list(range(ny))
    v9_lookup = {}
    if v9_selection is not None and not v9_selection.empty:
        for _, r in v9_selection.iterrows():
            obj = str(r.get("object", ""))
            day = r.get("selected_peak_day", r.get("peak_day", np.nan))
            v9_lookup[obj] = float(day) if pd.notna(day) else np.nan
    for iy in range(ny):
        keep = np.asarray([j for j in range(ny) if j != iy], dtype=int)
        for obj, (prof, _lat, _w) in profiles.items():
            sub = prof[keep, ...]
            peak_day, peak_score, _cand = _select_peak_for_profile(sub, scope, cfg, v7multi, obj)
            obs = v9_lookup.get(obj, np.nan)
            shift = peak_day - obs if np.isfinite(peak_day) and np.isfinite(obs) else np.nan
            rows.append({
                "window_id": scope.window_id,
                "left_out_year_index": iy,
                "left_out_year": year_vals[iy],
                "object": obj,
                "v9_observed_peak_day": obs,
                "loo_peak_day": peak_day,
                "loo_peak_shift": shift,
                "loo_peak_score": peak_score,
                "influence_flag": "shift_ge_5_days" if np.isfinite(shift) and abs(shift) >= 5 else ("shift_ge_3_days" if np.isfinite(shift) and abs(shift) >= 3 else "low_shift"),
                "method_role": "leave_one_year_out_influence_diagnostic",
            })
    return pd.DataFrame(rows)


def _merge_day_modes(days: np.ndarray, merge_gap_days: int = 2) -> List[dict]:
    finite = np.asarray([int(round(x)) for x in days if np.isfinite(x)], dtype=int)
    if finite.size == 0:
        return []
    vc = pd.Series(finite).value_counts().sort_index()
    unique_days = vc.index.to_numpy(int)
    counts = vc.to_numpy(int)
    groups: List[Tuple[List[int], List[int]]] = []
    cur_days = [int(unique_days[0])]
    cur_counts = [int(counts[0])]
    for d, c in zip(unique_days[1:], counts[1:]):
        if int(d) - cur_days[-1] <= merge_gap_days:
            cur_days.append(int(d)); cur_counts.append(int(c))
        else:
            groups.append((cur_days, cur_counts))
            cur_days = [int(d)]; cur_counts = [int(c)]
    groups.append((cur_days, cur_counts))
    total = int(counts.sum())
    modes = []
    for ds, cs in groups:
        prob = float(sum(cs) / max(total, 1))
        weighted_day = float(np.average(ds, weights=cs)) if sum(cs) > 0 else np.nan
        modes.append({
            "day_start": int(min(ds)),
            "day_end": int(max(ds)),
            "day_weighted_mean": weighted_day,
            "count": int(sum(cs)),
            "probability": prob,
        })
    modes.sort(key=lambda r: r["probability"], reverse=True)
    return modes


def _bootstrap_mode_audit(v9_bootstrap: pd.DataFrame, v9_selection: pd.DataFrame, scope, mode_merge_gap_days: int) -> pd.DataFrame:
    rows: List[dict] = []
    if v9_bootstrap is None or v9_bootstrap.empty:
        return pd.DataFrame([{"window_id": scope.window_id, "status": "v9_bootstrap_missing"}])
    df = v9_bootstrap[v9_bootstrap.get("window_id", "") == scope.window_id].copy()
    sel = v9_selection if v9_selection is not None else pd.DataFrame()
    obs_lookup = {}
    if not sel.empty:
        for _, r in sel.iterrows():
            obj = str(r.get("object", ""))
            day = r.get("selected_peak_day", r.get("peak_day", np.nan))
            obs_lookup[obj] = float(day) if pd.notna(day) else np.nan
    for obj, g in df.groupby("object"):
        days = g["selected_peak_day"].to_numpy(float) if "selected_peak_day" in g.columns else np.array([])
        finite_days = days[np.isfinite(days)]
        modes = _merge_day_modes(finite_days, merge_gap_days=mode_merge_gap_days)
        row = {
            "window_id": scope.window_id,
            "object": obj,
            "observed_peak_day": obs_lookup.get(str(obj), np.nan),
            "bootstrap_n": int(len(days)),
            "finite_bootstrap_n": int(len(finite_days)),
            "bootstrap_median": float(np.nanmedian(finite_days)) if finite_days.size else np.nan,
            "bootstrap_q025": float(np.nanquantile(finite_days, 0.025)) if finite_days.size else np.nan,
            "bootstrap_q975": float(np.nanquantile(finite_days, 0.975)) if finite_days.size else np.nan,
            "bootstrap_width_q025_q975": float(np.nanquantile(finite_days, 0.975) - np.nanquantile(finite_days, 0.025)) if finite_days.size else np.nan,
            "mode_count": int(len(modes)),
            "mode_merge_gap_days": int(mode_merge_gap_days),
        }
        for i in range(3):
            if i < len(modes):
                m = modes[i]
                row[f"mode{i+1}_day_range"] = f"{m['day_start']}-{m['day_end']}"
                row[f"mode{i+1}_day_weighted_mean"] = m["day_weighted_mean"]
                row[f"mode{i+1}_probability"] = m["probability"]
            else:
                row[f"mode{i+1}_day_range"] = ""
                row[f"mode{i+1}_day_weighted_mean"] = np.nan
                row[f"mode{i+1}_probability"] = np.nan
        if len(modes) == 0:
            dtype = "missing"
        elif len(modes) >= 2 and modes[1]["probability"] >= 0.20:
            dtype = "multi_mode"
        elif row["bootstrap_width_q025_q975"] >= 15:
            dtype = "single_broad_mode"
        elif modes[0]["probability"] < 0.15:
            dtype = "flat_or_weak"
        else:
            dtype = "single_compact_mode"
        row["distribution_type"] = dtype
        rows.append(row)
    return pd.DataFrame(rows)


def _pivot_yearwise(yearwise: pd.DataFrame) -> pd.DataFrame:
    if yearwise.empty:
        return pd.DataFrame()
    piv = yearwise.pivot_table(index=["window_id", "year_index", "year"], columns="object", values="yearwise_peak_day", aggfunc="first").reset_index()
    for obj in OBJECTS:
        if obj not in piv.columns:
            piv[obj] = np.nan
    return piv[["window_id", "year_index", "year"] + OBJECTS]


def _kmeans_numpy(X: np.ndarray, k: int, n_iter: int = 100) -> np.ndarray:
    # Deterministic initialization: choose rows near equally spaced sorted first PC proxy.
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n < k:
        return np.arange(n, dtype=int)
    scores = np.nanmean(X, axis=1)
    order = np.argsort(scores)
    init_idx = [order[int(round(i * (n - 1) / max(k - 1, 1)))] for i in range(k)]
    centers = X[init_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dist = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            if np.any(new_labels == j):
                new_centers[j] = np.mean(X[new_labels == j], axis=0)
        if np.array_equal(new_labels, labels):
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    # Renumber clusters by mean peak day for interpretability.
    cluster_means = [(j, float(np.mean(X[labels == j]))) for j in range(k) if np.any(labels == j)]
    cluster_means.sort(key=lambda x: x[1])
    mapper = {old: new for new, (old, _m) in enumerate(cluster_means, start=1)}
    return np.asarray([mapper.get(int(l), int(l)+1) for l in labels], dtype=int)


def _year_type_cluster_audit(yearwise: pd.DataFrame, scope, k_values: Iterable[int]) -> pd.DataFrame:
    piv = _pivot_yearwise(yearwise)
    if piv.empty:
        return pd.DataFrame()
    piv = piv[piv["window_id"] == scope.window_id].copy()
    if piv.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    X_raw = piv[OBJECTS].to_numpy(float)
    # Fill missing with object median. Keep original in output.
    X = X_raw.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col) if np.any(np.isfinite(col)) else float(scope.anchor_day)
        col[~np.isfinite(col)] = med
        X[:, j] = col
    # Standardize by object spread to avoid high-variance object domination.
    Xz = X.copy()
    for j in range(Xz.shape[1]):
        sd = np.nanstd(Xz[:, j])
        if not np.isfinite(sd) or sd <= 1e-9:
            sd = 1.0
        Xz[:, j] = (Xz[:, j] - np.nanmean(Xz[:, j])) / sd
    for k in k_values:
        if len(piv) < k:
            continue
        labels = _kmeans_numpy(Xz, int(k))
        counts = pd.Series(labels).value_counts().to_dict()
        for idx, r in piv.reset_index(drop=True).iterrows():
            cl = int(labels[idx])
            row = {
                "window_id": scope.window_id,
                "k": int(k),
                "year_index": int(r["year_index"]),
                "year": r["year"],
                "cluster_id": cl,
                "cluster_size": int(counts.get(cl, 0)),
                "cluster_label_provisional": f"k{k}_cluster{cl}",
                "cluster_stability": "not_tested_first_pass",
            }
            for obj in OBJECTS:
                row[f"peak_{obj}"] = r[obj]
                row[f"relative_peak_{obj}"] = r[obj] - scope.anchor_day if pd.notna(r[obj]) else np.nan
            row["method_role"] = "provisional_year_type_cluster_not_physical_type"
            rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_order_status(prob_a: float, prob_b: float) -> str:
    # User semantics: 90 usable, 95 credible, 99 strict.
    top = max(prob_a, prob_b)
    side = "A_earlier" if prob_a >= prob_b else "B_earlier"
    if top >= 0.99:
        return f"{side}_strict_99"
    if top >= 0.95:
        return f"{side}_credible_95"
    if top >= 0.90:
        return f"{side}_usable_90"
    return "order_unresolved_lt90"


def _pairwise_order_by_year_type(cluster_df: pd.DataFrame, scope) -> pd.DataFrame:
    if cluster_df is None or cluster_df.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    df = cluster_df[cluster_df["window_id"] == scope.window_id].copy()
    for (k, cl), g in df.groupby(["k", "cluster_id"]):
        n = len(g)
        for i, a in enumerate(PAIR_OBJECTS_ORDERED):
            for b in PAIR_OBJECTS_ORDERED[i+1:]:
                va = g[f"peak_{a}"].to_numpy(float)
                vb = g[f"peak_{b}"].to_numpy(float)
                m = np.isfinite(va) & np.isfinite(vb)
                if m.sum() == 0:
                    continue
                delta = vb[m] - va[m]
                p_a = float(np.mean(delta > 0))
                p_b = float(np.mean(delta < 0))
                p_same = float(np.mean(delta == 0))
                status = _pairwise_order_status(p_a, p_b)
                rows.append({
                    "window_id": scope.window_id,
                    "k": int(k),
                    "cluster_id": int(cl),
                    "cluster_size": int(n),
                    "object_A": a,
                    "object_B": b,
                    "delta_definition": "B_peak_day - A_peak_day; positive means A earlier",
                    "P_A_earlier_within_type": p_a,
                    "P_B_earlier_within_type": p_b,
                    "P_same_day_within_type": p_same,
                    "median_delta_within_type": float(np.nanmedian(delta)),
                    "q025_delta_within_type": float(np.nanquantile(delta, 0.025)) if len(delta) > 1 else float(delta[0]),
                    "q975_delta_within_type": float(np.nanquantile(delta, 0.975)) if len(delta) > 1 else float(delta[0]),
                    "within_type_order_status": status,
                    "small_cluster_warning": bool(n < 5),
                    "method_role": "within_provisional_year_type_peak_order_audit",
                })
    return pd.DataFrame(rows)


def _instability_summary(scope, v9_tables: Dict[str, pd.DataFrame], mode_df: pd.DataFrame, jack_df: pd.DataFrame, yearwise_df: pd.DataFrame, cluster_df: pd.DataFrame, order_by_type_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    timing = v9_tables.get("timing_resolution", pd.DataFrame())
    po = v9_tables.get("pairwise_peak_order", pd.DataFrame())
    timing_w = timing[timing.get("window_id", "") == scope.window_id].copy() if not timing.empty else pd.DataFrame()
    po_w = po[po.get("window_id", "") == scope.window_id].copy() if not po.empty else pd.DataFrame()
    # Object-level instability source.
    for obj in OBJECTS:
        mode_row = mode_df[mode_df["object"].astype(str) == obj].head(1) if not mode_df.empty and "object" in mode_df.columns else pd.DataFrame()
        jack_obj = jack_df[jack_df["object"].astype(str) == obj] if not jack_df.empty and "object" in jack_df.columns else pd.DataFrame()
        yw_obj = yearwise_df[yearwise_df["object"].astype(str) == obj] if not yearwise_df.empty and "object" in yearwise_df.columns else pd.DataFrame()
        trow = timing_w[timing_w["object"].astype(str) == obj].head(1) if not timing_w.empty and "object" in timing_w.columns else pd.DataFrame()
        width = np.nan
        if not trow.empty:
            q025 = trow.get("bootstrap_peak_q025", pd.Series([np.nan])).iloc[0]
            q975 = trow.get("bootstrap_peak_q975", pd.Series([np.nan])).iloc[0]
            try:
                width = float(q975) - float(q025)
            except Exception:
                width = np.nan
        dtype = str(mode_row.get("distribution_type", pd.Series([""])).iloc[0]) if not mode_row.empty else "mode_missing"
        jack_max = float(np.nanmax(np.abs(jack_obj["loo_peak_shift"].to_numpy(float)))) if not jack_obj.empty and "loo_peak_shift" in jack_obj.columns else np.nan
        yw_spread = float(np.nanquantile(yw_obj["yearwise_peak_day"].to_numpy(float), 0.975) - np.nanquantile(yw_obj["yearwise_peak_day"].to_numpy(float), 0.025)) if not yw_obj.empty and np.isfinite(yw_obj["yearwise_peak_day"].to_numpy(float)).sum() > 1 else np.nan
        if dtype == "multi_mode":
            source = "multi_candidate_event_candidate"
        elif np.isfinite(jack_max) and jack_max >= 5:
            source = "high_jackknife_year_influence_candidate"
        elif np.isfinite(yw_spread) and yw_spread >= 15:
            source = "yearwise_peak_spread_candidate"
        elif dtype in ["single_broad_mode", "flat_or_weak"]:
            source = "flat_or_weak_peak_candidate"
        else:
            source = "insufficient_or_low_instability_evidence"
        rows.append({
            "window_id": scope.window_id,
            "scope": "object",
            "object_or_pair": obj,
            "v9_bootstrap_width_q025_q975": width,
            "bootstrap_mode_type": dtype,
            "jackknife_max_abs_shift": jack_max,
            "yearwise_peak_spread_q025_q975": yw_spread,
            "likely_instability_source": source,
            "recommended_interpretation": "audit-only; do not replace V9 peak result",
        })
    # Pair-level possible year-type stabilization.
    for _, r in po_w.iterrows() if not po_w.empty else []:
        a = str(r.get("object_A", "")); b = str(r.get("object_B", ""))
        pair = f"{a}-{b}"
        full_status = str(r.get("peak_order_decision", r.get("decision", "")))
        ob = order_by_type_df[(order_by_type_df["object_A"] == a) & (order_by_type_df["object_B"] == b)] if not order_by_type_df.empty else pd.DataFrame()
        n_stable = 0
        if not ob.empty and "within_type_order_status" in ob.columns:
            n_stable = int(ob["within_type_order_status"].astype(str).str.contains("usable_90|credible_95|strict_99", regex=True).sum())
        source = "year_type_mixture_candidate" if n_stable >= 2 and "unresolved" in full_status else ("year_type_may_refine_order" if n_stable >= 1 else "no_clear_year_type_stabilization")
        rows.append({
            "window_id": scope.window_id,
            "scope": "pair",
            "object_or_pair": pair,
            "v9_order_status": full_status,
            "n_within_type_orders_ge90": n_stable,
            "likely_instability_source": source,
            "recommended_interpretation": "audit-only; compare with V9 full-sample order before scientific use",
        })
    return pd.DataFrame(rows)


def _write_summary(path: Path, run_scopes: List[object], concat_summary: pd.DataFrame) -> None:
    lines = [
        "# V9.1 peak yearwise heterogeneity audit summary",
        "",
        f"version: `{V91_VERSION}`",
        "",
        "## Purpose",
        "V9.1 tests whether V9 peak instability is associated with yearwise event-type heterogeneity, multi-mode bootstrap peak distributions, jackknife-sensitive years, or weak/flat object responses.",
        "",
        "## Boundary",
        "- Reads V9 outputs but does not modify V9.",
        "- Does not redefine the V9/V7 peak method.",
        "- Does not include state/growth/process diagnostics.",
        "- Year-type clusters are provisional audit structures, not physical climate regimes.",
        "",
        "## Windows processed",
    ]
    for s in run_scopes:
        lines.append(f"- {s.window_id}: anchor day {s.anchor_day}")
    if concat_summary is not None and not concat_summary.empty and "likely_instability_source" in concat_summary.columns:
        lines += ["", "## Likely instability source counts"]
        vc = concat_summary["likely_instability_source"].value_counts(dropna=False)
        for k, v in vc.items():
            lines.append(f"- {k}: {int(v)}")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_peak_yearwise_heterogeneity_v9_1_a(v91_root: Path | str) -> None:
    v91_root = Path(v91_root)
    stage_root = _v91_stage_root(v91_root)
    v9_root = stage_root / "V9"
    v7_root = stage_root / "V7"
    v7multi = _import_v7_module(stage_root)
    cfg = _make_v7_cfg_for_v91(v7multi)

    out_root = _ensure_dir(v91_root / "outputs" / OUTPUT_TAG)
    out_cross = _ensure_dir(out_root / "cross_window")
    out_per = _ensure_dir(out_root / "per_window")
    log_dir = _ensure_dir(v91_root / "logs" / OUTPUT_TAG)
    t0 = time.time()

    mode_merge_gap_days = int(os.environ.get("V9_1_MODE_MERGE_GAP_DAYS", "2"))
    k_values = [int(x) for x in os.environ.get("V9_1_CLUSTER_K_VALUES", "2,3").split(",") if x.strip()]

    _log("[1/7] Read V9 reference outputs")
    v9_tables = _load_v9_reference_tables(stage_root)
    v9_output_root = v9_root / "outputs" / PARENT_V9_TAG
    _write_json({
        "version": V91_VERSION,
        "output_tag": OUTPUT_TAG,
        "parent_version": "V9 peak_all_windows_v9_a",
        "modifies_v9": False,
        "reads_v9_outputs": True,
        "v9_output_root": str(v9_output_root),
        "windows": DEFAULT_WINDOWS,
        "excluded_windows": EXCLUDED_WINDOWS,
        "state_included": False,
        "growth_included": False,
        "process_a_included": False,
        "mode_merge_gap_days": mode_merge_gap_days,
        "cluster_k_values": k_values,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_root / "run_meta.json")

    _log("[2/7] Load V7 scopes and profiles without modifying V9")
    wins = v7multi._load_accepted_windows(v7_root, out_cross, cfg)
    scopes, validity = v7multi._build_window_scopes(wins, cfg)
    run_scopes, run_scope_audit = v7multi._filter_scopes_for_run(scopes, cfg)
    _safe_to_csv(pd.DataFrame([asdict(s) for s in scopes]), out_cross / "window_scope_registry_v9_1_a.csv")
    _safe_to_csv(validity, out_cross / "window_scope_validity_audit_v9_1_a.csv")
    _safe_to_csv(run_scope_audit, out_cross / "run_window_selection_audit_v9_1_a.csv")

    profiles, years, input_audit, smoothed = _build_profiles(v91_root, stage_root, v7multi, cfg)
    _safe_to_csv(input_audit, out_cross / "input_key_audit_v9_1_a.csv")

    all_yearwise: List[pd.DataFrame] = []
    all_jack: List[pd.DataFrame] = []
    all_mode: List[pd.DataFrame] = []
    all_cluster: List[pd.DataFrame] = []
    all_order_type: List[pd.DataFrame] = []
    all_summary: List[pd.DataFrame] = []

    _log("[3/7] Run yearwise, jackknife, bootstrap-mode, and year-type audits")
    for idx, scope in enumerate(run_scopes, start=1):
        _log(f"  [{idx}/{len(run_scopes)}] {scope.window_id}")
        out_win = _ensure_dir(out_per / scope.window_id)
        _safe_to_csv(pd.DataFrame([asdict(scope)]), out_win / f"window_scope_{scope.window_id}.csv")
        v9_win = v9_output_root / "per_window" / scope.window_id
        v9_selection = _read_csv_if_exists(v9_win / f"object_peak_registry_{scope.window_id}.csv")
        if v9_selection.empty:
            v9_selection = _read_csv_if_exists(v9_win / f"main_window_selection_{scope.window_id}.csv")

        yearwise = _make_yearwise_registry(profiles, years, scope, cfg, v7multi)
        jack = _jackknife_influence(profiles, years, scope, cfg, v7multi, v9_selection)
        boot_df = v9_tables.get("bootstrap_all", pd.DataFrame())
        mode = _bootstrap_mode_audit(boot_df, v9_selection, scope, mode_merge_gap_days)
        cluster = _year_type_cluster_audit(yearwise, scope, k_values)
        order_type = _pairwise_order_by_year_type(cluster, scope)
        summary = _instability_summary(scope, v9_tables, mode, jack, yearwise, cluster, order_type)

        _safe_to_csv(yearwise, out_win / f"yearwise_object_peak_registry_{scope.window_id}.csv")
        _safe_to_csv(jack, out_win / f"jackknife_object_peak_influence_{scope.window_id}.csv")
        _safe_to_csv(mode, out_win / f"bootstrap_peak_mode_audit_{scope.window_id}.csv")
        _safe_to_csv(cluster, out_win / f"year_type_cluster_audit_{scope.window_id}.csv")
        _safe_to_csv(order_type, out_win / f"pairwise_order_by_year_type_{scope.window_id}.csv")
        _safe_to_csv(summary, out_win / f"peak_instability_source_summary_{scope.window_id}.csv")
        (out_win / f"peak_yearwise_heterogeneity_summary_{scope.window_id}.md").write_text(
            "# Peak yearwise heterogeneity audit\n\n"
            f"window: {scope.window_id}\n\n"
            "This is an audit-only result. It does not replace V9 peak outputs.\n",
            encoding="utf-8",
        )
        _write_json({
            "version": V91_VERSION,
            "window_id": scope.window_id,
            "modifies_v9": False,
            "mode_merge_gap_days": mode_merge_gap_days,
            "cluster_k_values": k_values,
            "boundary": "heterogeneity audit only; no state/growth/process interpretation",
        }, out_win / f"window_run_meta_{scope.window_id}.json")
        all_yearwise.append(yearwise)
        all_jack.append(jack)
        all_mode.append(mode)
        all_cluster.append(cluster)
        all_order_type.append(order_type)
        all_summary.append(summary)

    _log("[4/7] Write cross-window concatenated audit tables")
    concat_yearwise = pd.concat(all_yearwise, ignore_index=True) if all_yearwise else pd.DataFrame()
    concat_jack = pd.concat(all_jack, ignore_index=True) if all_jack else pd.DataFrame()
    concat_mode = pd.concat(all_mode, ignore_index=True) if all_mode else pd.DataFrame()
    concat_cluster = pd.concat(all_cluster, ignore_index=True) if all_cluster else pd.DataFrame()
    concat_order_type = pd.concat(all_order_type, ignore_index=True) if all_order_type else pd.DataFrame()
    concat_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    _safe_to_csv(concat_yearwise, out_cross / "yearwise_object_peak_registry_all_windows.csv")
    _safe_to_csv(concat_jack, out_cross / "jackknife_object_peak_influence_all_windows.csv")
    _safe_to_csv(concat_mode, out_cross / "bootstrap_peak_mode_audit_all_windows.csv")
    _safe_to_csv(concat_cluster, out_cross / "year_type_cluster_audit_all_windows.csv")
    _safe_to_csv(concat_order_type, out_cross / "pairwise_order_by_year_type_all_windows.csv")
    _safe_to_csv(concat_summary, out_cross / "v9_peak_instability_source_summary.csv")
    _safe_to_csv(pd.DataFrame([{
        "window_id": w,
        "included_in_v9_1": False,
        "exclusion_reason": "inherits V9 strict window set; W135 excluded in V9",
    } for w in EXCLUDED_WINDOWS]), out_cross / "excluded_windows_v9_1_a.csv")

    _log("[5/7] Write summary")
    _write_summary(out_cross / "peak_yearwise_heterogeneity_v9_1_a_summary.md", run_scopes, concat_summary)

    _log("[6/7] Write run metadata")
    elapsed = time.time() - t0
    _write_json({
        "version": V91_VERSION,
        "elapsed_seconds": elapsed,
        "output_root": str(out_root),
        "v9_output_root": str(v9_output_root),
        "smoothed_fields": str(smoothed),
        "windows_processed": [s.window_id for s in run_scopes],
        "n_years": int(next(iter(profiles.values()))[0].shape[0]) if profiles else 0,
        "mode_merge_gap_days": mode_merge_gap_days,
        "cluster_k_values": k_values,
        "modifies_v9": False,
        "boundary": "audit-only; no state/growth/prepost/process outputs",
        "source_v7_module_version": getattr(v7multi, "VERSION", "unknown"),
    }, out_root / "summary.json")
    (log_dir / "last_run.txt").write_text(
        f"Completed {time.strftime('%Y-%m-%d %H:%M:%S')} output={out_root}\n",
        encoding="utf-8",
    )
    _log("[7/7] Done")


if __name__ == "__main__":  # pragma: no cover
    run_peak_yearwise_heterogeneity_v9_1_a(Path(__file__).resolve().parents[2])
