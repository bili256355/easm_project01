"""
V9.1_b transition-type mixture audit.

Purpose
-------
This module is a read-only audit branch for V9 peak_all_windows_v9_a.  It tests
whether V9 peak/order instability can plausibly be explained by mixtures of
multiple year-level transition-behaviour types.

Critical methodological boundary
--------------------------------
V9.1_b does NOT cluster single-year peak days.  The user explicitly rejected
single-year peak detection as the primary evidence because single-year samples
are too noisy and change-point/peak detectors can amplify noise.  Instead,
V9.1_b clusters each year by low-dimensional, whole-window, multi-object
behaviour features, then returns to the V9 multi-year peak detector inside each
candidate type group.

This module does not modify V9 code or V9 outputs, and it does not add state,
growth, process_a, pre-post state, or physical year-type interpretation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

VERSION = "v9_1_b_transition_type_mixture_audit"
OUTPUT_TAG = "transition_type_mixture_audit_v9_1_b"
DEFAULT_WINDOWS = ["W045", "W081", "W113", "W160"]
OBJECTS = ["P", "V", "H", "Je", "Jw"]
EXCLUDED_WINDOWS = [
    {
        "window_id": "W135",
        "included_in_v9_1_b": False,
        "reason": "excluded_by_V9_strict_accepted_window_set",
    }
]


@dataclass
class V91BConfig:
    windows: List[str] = field(default_factory=lambda: list(DEFAULT_WINDOWS))
    objects: List[str] = field(default_factory=lambda: list(OBJECTS))
    k_values: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    som_grids: List[Tuple[int, int]] = field(default_factory=lambda: [(2, 2), (2, 3), (3, 3)])
    profile_pcs_per_object: int = 3
    feature_pca_max_components: int = 8
    change_half_window_days: int = 3
    min_cluster_size_for_type_peak: int = 8
    type_bootstrap_n: int = 500
    debug_type_bootstrap_n: int = 50
    kmedoids_seeds: List[int] = field(default_factory=lambda: [11, 23, 37, 51, 73])
    kmedoids_max_iter: int = 80
    som_iter: int = 800
    debug: bool = False
    evidence_usable: float = 0.90
    evidence_credible: float = 0.95
    evidence_strict: float = 0.99

    @classmethod
    def from_env(cls) -> "V91BConfig":
        cfg = cls()
        if os.environ.get("V9_1B_DEBUG"):
            cfg.debug = True
            cfg.type_bootstrap_n = int(os.environ.get("V9_1B_DEBUG_BOOTSTRAP_N", cfg.debug_type_bootstrap_n))
        if os.environ.get("V9_1B_TYPE_BOOTSTRAP_N"):
            cfg.type_bootstrap_n = int(os.environ["V9_1B_TYPE_BOOTSTRAP_N"])
        if os.environ.get("V9_1B_K_VALUES"):
            cfg.k_values = [int(x) for x in os.environ["V9_1B_K_VALUES"].replace(";", ",").split(",") if x.strip()]
        if os.environ.get("V9_1B_MIN_CLUSTER_SIZE"):
            cfg.min_cluster_size_for_type_peak = int(os.environ["V9_1B_MIN_CLUSTER_SIZE"])
        if os.environ.get("V9_1B_PROFILE_PCS_PER_OBJECT"):
            cfg.profile_pcs_per_object = int(os.environ["V9_1B_PROFILE_PCS_PER_OBJECT"])
        if os.environ.get("V9_1B_CHANGE_HALF_WINDOW_DAYS"):
            cfg.change_half_window_days = int(os.environ["V9_1B_CHANGE_HALF_WINDOW_DAYS"])
        if os.environ.get("V9_1B_SOM_ITER"):
            cfg.som_iter = int(os.environ["V9_1B_SOM_ITER"])
        return cfg


def _stage_root_from_v91(v91_root: Path) -> Path:
    return Path(v91_root).resolve().parent


def _import_v7_module(stage_root: Path):
    v7_src = stage_root / "V7" / "src"
    if not v7_src.exists():
        raise FileNotFoundError(f"Cannot find V7 source directory: {v7_src}")
    if str(v7_src) not in sys.path:
        sys.path.insert(0, str(v7_src))
    from stage_partition_v7 import accepted_windows_multi_object_prepost_v7_z_multiwin_a as v7multi
    return v7multi


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _default_smoothed_path(stage_root: Path) -> Path:
    return stage_root.parent / "foundation" / "V1" / "outputs" / "baseline_a" / "preprocess" / "smoothed_fields.npz"


def _make_v7_cfg_for_v91(v7multi, cfg91: V91BConfig, stage_root: Path):
    cfg = v7multi.MultiWinConfig.from_env()
    # Keep V9/V7 peak semantics; V9.1_b controls windows and bootstrap budget.
    cfg.window_mode = "list"
    cfg.target_windows = ",".join(cfg91.windows)
    if os.environ.get("V9_1B_PEAK_ACCEPTED_WINDOW_REGISTRY"):
        cfg.accepted_window_registry = os.environ["V9_1B_PEAK_ACCEPTED_WINDOW_REGISTRY"]
        cfg.window_source = "registry"
    if os.environ.get("V9_1B_SMOOTHED_FIELDS"):
        cfg.smoothed_fields_path = os.environ["V9_1B_SMOOTHED_FIELDS"]
    if not getattr(cfg, "smoothed_fields_path", None):
        default = _default_smoothed_path(stage_root)
        if default.exists():
            cfg.smoothed_fields_path = str(default)
    cfg.bootstrap_n = int(cfg91.type_bootstrap_n)
    # V9.1_b is peak-only; keep side paths closed.
    cfg.run_2d = False
    cfg.run_w45_profile_order_tests = False
    cfg.save_daily_curves = False
    cfg.save_bootstrap_samples = False
    cfg.save_bootstrap_curves = False
    return cfg


def _load_v9_tables(stage_root: Path) -> Dict[str, pd.DataFrame]:
    v9_out = stage_root / "V9" / "outputs" / "peak_all_windows_v9_a" / "cross_window"
    tables = {}
    for key, fname in {
        "accepted_windows": "accepted_windows_used_v9_a.csv",
        "object_peak": "cross_window_object_peak_registry.csv",
        "pairwise_order": "cross_window_pairwise_peak_order.csv",
        "pairwise_sync": "cross_window_pairwise_peak_synchrony.csv",
    }.items():
        f = v9_out / fname
        tables[key] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return tables


def _window_scopes_from_v7(v7multi, stage_root: Path, cfg) -> List[object]:
    v7_root = stage_root / "V7"
    tmp = _ensure_dir(stage_root / "V9_1" / "outputs" / OUTPUT_TAG / "cross_window" / "_window_registry_tmp")
    wins = v7multi._load_accepted_windows(v7_root, tmp, cfg)
    scopes, _validity = v7multi._build_window_scopes(wins, cfg)
    run_scopes, _audit = v7multi._filter_scopes_for_run(scopes, cfg)
    return [s for s in run_scopes if s.window_id in set(DEFAULT_WINDOWS)]


def _load_fields_and_profiles(v7multi, stage_root: Path, cfg) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, pd.DataFrame]:
    smoothed = Path(cfg.smoothed_fields_path) if getattr(cfg, "smoothed_fields_path", None) else _default_smoothed_path(stage_root)
    if not smoothed.exists():
        raise FileNotFoundError(f"smoothed_fields.npz not found: {smoothed}. Set V9_1B_SMOOTHED_FIELDS if needed.")
    fields, audit = v7multi.clean._load_npz_fields(smoothed)
    lat, lon = fields["lat"], fields["lon"]
    years = fields.get("years")
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for spec in v7multi.clean.OBJECT_SPECS:
        arr = v7multi.clean._as_year_day_lat_lon(fields[spec.field_role], lat, lon, years)
        prof, target_lat, weights = v7multi.clean._build_object_profile(arr, lat, lon, spec)
        profiles[spec.object_name] = (np.asarray(prof, dtype=float), target_lat, weights)
        rows.append({
            **asdict(spec),
            "profile_shape": str(prof.shape),
            "target_lat_min": float(np.nanmin(target_lat)),
            "target_lat_max": float(np.nanmax(target_lat)),
            "v9_1b_role": "window_behavior_feature_input_and_type_level_peak_input",
        })
    n_years = next(iter(profiles.values()))[0].shape[0]
    if years is None:
        years = np.arange(n_years)
    years = np.asarray(years)
    return profiles, years, pd.DataFrame(rows)


def _nanmean_safe(arr: np.ndarray, axis=None):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.nan
    with np.errstate(all="ignore"):
        return np.nanmean(arr, axis=axis)


def _zscore_cols(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1.0e-12), sd, 1.0)
    Z = (X - mu) / sd
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z, mu, sd


def _pca_scores(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    Z, _, _ = _zscore_cols(X)
    n_components = int(max(1, min(n_components, Z.shape[0] - 1, Z.shape[1]))) if Z.shape[0] > 1 and Z.shape[1] > 0 else 1
    if Z.shape[0] <= 1 or Z.shape[1] == 0:
        return np.zeros((Z.shape[0], n_components)), np.zeros(n_components)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    scores = U[:, :n_components] * s[:n_components]
    ev = s ** 2
    evr = ev / ev.sum() if ev.sum() > 0 else np.zeros_like(ev)
    return scores[:, :n_components], evr[:n_components]


def _object_window_matrix(prof: np.ndarray, start: int, end: int) -> np.ndarray:
    prof = np.asarray(prof, dtype=float)
    n_years, n_days = prof.shape[0], prof.shape[1]
    s = max(0, int(start)); e = min(n_days - 1, int(end))
    if e < s:
        return np.zeros((n_years, 0))
    sub = prof[:, s:e+1, ...]
    return sub.reshape(n_years, -1)


def _mean_profile(prof: np.ndarray, idx: slice) -> np.ndarray:
    sub = prof[:, idx, ...]
    return _nanmean_safe(sub, axis=1).reshape(prof.shape[0], -1)


def _change_curve_for_object(prof: np.ndarray, start: int, end: int, half_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return years x center_days change score curve, and center day labels."""
    prof = np.asarray(prof, dtype=float)
    n_years, n_days = prof.shape[0], prof.shape[1]
    centers = []
    curves = []
    for t in range(max(start, half_window), min(end, n_days - 1 - half_window) + 1):
        left = _nanmean_safe(prof[:, t-half_window:t, ...], axis=1).reshape(n_years, -1)
        right = _nanmean_safe(prof[:, t+1:t+1+half_window, ...], axis=1).reshape(n_years, -1)
        diff = right - left
        # Normalise by available finite dimensions to avoid profile-length dominance.
        finite = np.isfinite(diff)
        diff = np.where(finite, diff, 0.0)
        denom = np.sqrt(np.maximum(finite.sum(axis=1), 1))
        dist = np.sqrt(np.sum(diff * diff, axis=1)) / denom
        centers.append(t)
        curves.append(dist)
    if not centers:
        return np.zeros((n_years, 0)), np.asarray([], dtype=int)
    return np.vstack(curves).T, np.asarray(centers, dtype=int)


def _change_summaries(curve: np.ndarray, days: np.ndarray) -> pd.DataFrame:
    rows = []
    if curve.shape[1] == 0:
        for i in range(curve.shape[0]):
            rows.append({"change_area": np.nan, "change_centroid": np.nan, "change_width": np.nan,
                         "change_skew": np.nan, "early_change_area": np.nan, "late_change_area": np.nan})
        return pd.DataFrame(rows)
    mid = np.nanmedian(days)
    for y in range(curve.shape[0]):
        c = np.asarray(curve[y], dtype=float)
        c = np.where(np.isfinite(c) & (c > 0), c, 0.0)
        area = float(np.sum(c))
        if area <= 1.0e-12:
            centroid = np.nan; width = np.nan; skew = np.nan
        else:
            centroid = float(np.sum(days * c) / area)
            var = float(np.sum(((days - centroid) ** 2) * c) / area)
            width = float(np.sqrt(max(var, 0.0)))
            if width > 1.0e-12:
                skew = float(np.sum(((days - centroid) ** 3) * c) / area / (width ** 3))
            else:
                skew = 0.0
        rows.append({
            "change_area": area,
            "change_centroid": centroid,
            "change_width": width,
            "change_skew": skew,
            "early_change_area": float(np.sum(c[days <= mid])),
            "late_change_area": float(np.sum(c[days > mid])),
        })
    return pd.DataFrame(rows)


def _build_window_year_behavior_features(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    years: np.ndarray,
    scope: object,
    cfg: V91BConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame({"window_id": scope.window_id, "year": years})
    ev_rows = []
    object_blocks = []
    rel_centroids = {}
    rel_areas = {}
    for obj in cfg.objects:
        prof = profiles[obj][0]
        # Whole-window profile shape PCs.
        X_flat = _object_window_matrix(prof, scope.analysis_start, scope.analysis_end)
        scores, evr = _pca_scores(X_flat, cfg.profile_pcs_per_object)
        obj_cols = []
        for j in range(scores.shape[1]):
            col = f"{obj}__profile_pc{j+1}"
            rows[col] = scores[:, j]
            obj_cols.append(col)
            ev_rows.append({"window_id": scope.window_id, "object": obj, "feature_kind": "profile_pca", "component": j+1, "explained_variance_ratio": float(evr[j]) if j < len(evr) else np.nan})
        # Change curve summaries.
        curve, days = _change_curve_for_object(prof, scope.analysis_start, scope.analysis_end, cfg.change_half_window_days)
        summ = _change_summaries(curve, days)
        for c in summ.columns:
            col = f"{obj}__{c}"
            rows[col] = summ[c].values
            obj_cols.append(col)
        rel_centroids[obj] = rows[f"{obj}__change_centroid"].to_numpy(dtype=float)
        rel_areas[obj] = rows[f"{obj}__change_area"].to_numpy(dtype=float)
        object_blocks.append((obj, obj_cols))
    # A few joint relative features; these are inputs only, not peak conclusions.
    for a, b in [("H", "Jw"), ("P", "V"), ("Je", "Jw"), ("H", "V"), ("Jw", "V")]:
        if a in rel_centroids and b in rel_centroids:
            rows[f"REL__centroid_{a}_minus_{b}"] = rel_centroids[a] - rel_centroids[b]
            rows[f"REL__area_{a}_minus_{b}"] = rel_areas[a] - rel_areas[b]
    # Build equalized feature matrix columns.
    feature_cols = [c for c in rows.columns if c not in ("window_id", "year")]
    X = rows[feature_cols].to_numpy(dtype=float)
    Z, _, _ = _zscore_cols(X)
    # Equalize object blocks by block size; relative features are another block.
    scaled = np.zeros_like(Z)
    col_index = {c: i for i, c in enumerate(feature_cols)}
    for _obj, cols in object_blocks:
        idx = [col_index[c] for c in cols if c in col_index]
        if idx:
            scaled[:, idx] = Z[:, idx] / math.sqrt(len(idx))
    rel_idx = [i for c, i in col_index.items() if c.startswith("REL__")]
    if rel_idx:
        scaled[:, rel_idx] = Z[:, rel_idx] / math.sqrt(len(rel_idx))
    for i, c in enumerate(feature_cols):
        rows[f"STD__{c}"] = scaled[:, i]
    rows["v9_1b_feature_role"] = "whole_window_multi_object_behavior_not_single_year_peak"
    return rows, pd.DataFrame(ev_rows)


def _feature_matrix_for_clustering(features: pd.DataFrame) -> Tuple[np.ndarray, List[str], np.ndarray]:
    cols = [c for c in features.columns if c.startswith("STD__")]
    X = features[cols].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    scores, evr = _pca_scores(X, min(8, X.shape[1] if X.ndim == 2 else 1))
    return scores, cols, evr


def _euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    ss = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.maximum(ss + ss.T - 2 * X @ X.T, 0.0)
    return np.sqrt(D2)


def _kmedoids_once(X: np.ndarray, k: int, seed: int, max_iter: int = 80) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    D = _euclidean_distance_matrix(X)
    medoids = rng.choice(n, size=k, replace=False)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist_to = D[:, medoids]
        labels = np.argmin(dist_to, axis=1)
        new_medoids = medoids.copy()
        for c in range(k):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                # Re-seed an empty cluster with the point farthest from its current medoid.
                current_min = np.min(dist_to, axis=1)
                new_medoids[c] = int(np.argmax(current_min))
                continue
            subD = D[np.ix_(idx, idx)]
            new_medoids[c] = int(idx[np.argmin(np.sum(subD, axis=1))])
        if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            medoids = new_medoids
            break
        medoids = new_medoids
    dist_to = D[:, medoids]
    labels = np.argmin(dist_to, axis=1)
    cost = float(np.sum(np.min(dist_to, axis=1)))
    return labels, medoids, cost


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    n = labels_a.size
    if n < 2:
        return 1.0
    def comb2(x): return x * (x - 1) / 2.0
    classes_a = {v: np.where(labels_a == v)[0] for v in np.unique(labels_a)}
    classes_b = {v: np.where(labels_b == v)[0] for v in np.unique(labels_b)}
    sum_nij = 0.0
    for ia in classes_a.values():
        seta = set(ia.tolist())
        for ib in classes_b.values():
            nij = len(seta.intersection(ib.tolist()))
            sum_nij += comb2(nij)
    sum_ai = sum(comb2(len(v)) for v in classes_a.values())
    sum_bj = sum(comb2(len(v)) for v in classes_b.values())
    total = comb2(n)
    expected = sum_ai * sum_bj / total if total else 0.0
    max_index = 0.5 * (sum_ai + sum_bj)
    denom = max_index - expected
    if abs(denom) < 1.0e-12:
        return 1.0
    return float((sum_nij - expected) / denom)


def _run_kmedoids_suite(features: pd.DataFrame, cfg: V91BConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, feature_cols, evr = _feature_matrix_for_clustering(features)
    years = features["year"].to_numpy()
    window_id = str(features["window_id"].iloc[0])
    membership_rows, stability_rows = [], []
    pca_rows = []
    for j, v in enumerate(evr, start=1):
        pca_rows.append({"window_id": window_id, "component": j, "explained_variance_ratio": float(v)})
    for k in cfg.k_values:
        if k >= len(years):
            continue
        run_labels, run_medoids, run_costs = [], [], []
        for seed in cfg.kmedoids_seeds:
            labels, medoids, cost = _kmedoids_once(X, k, seed, cfg.kmedoids_max_iter)
            run_labels.append(labels); run_medoids.append(medoids); run_costs.append(cost)
        best_idx = int(np.argmin(run_costs))
        labels = run_labels[best_idx]; medoids = run_medoids[best_idx]
        # Stabilty across seed runs.
        aris = []
        for i in range(len(run_labels)):
            for j in range(i + 1, len(run_labels)):
                aris.append(_adjusted_rand_index(run_labels[i], run_labels[j]))
        stability = float(np.nanmean(aris)) if aris else 1.0
        sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
        D = _euclidean_distance_matrix(X)
        within = []
        for c, medoid in enumerate(medoids):
            idx = np.where(labels == c)[0]
            if idx.size:
                within.extend(D[idx, medoid].tolist())
        between = []
        for i in range(len(medoids)):
            for j in range(i + 1, len(medoids)):
                between.append(float(D[medoids[i], medoids[j]]))
        mean_within = float(np.nanmean(within)) if within else np.nan
        mean_between = float(np.nanmean(between)) if between else np.nan
        min_size = min(sizes.values()) if sizes else 0
        status = "usable_for_type_peak"
        if min_size < cfg.min_cluster_size_for_type_peak:
            status = "caution_small_cluster"
        if stability < 0.50:
            status = "unstable_clustering"
        stability_rows.append({
            "window_id": window_id,
            "method": "kmedoids_behavior_pca",
            "k": int(k),
            "n_years": int(len(years)),
            "min_cluster_size": int(min_size),
            "max_cluster_size": int(max(sizes.values()) if sizes else 0),
            "mean_within_distance": mean_within,
            "mean_between_distance": mean_between,
            "between_within_ratio": mean_between / mean_within if np.isfinite(mean_between) and np.isfinite(mean_within) and mean_within > 0 else np.nan,
            "stability_score_ari_across_seeds": stability,
            "cluster_quality_status": status,
            "best_seed_index": best_idx,
            "best_cost": float(run_costs[best_idx]),
            "feature_source": "whole_window_behavior_features_not_single_year_peak",
        })
        for n, year in enumerate(years):
            cid = int(labels[n])
            membership_rows.append({
                "window_id": window_id,
                "year": year,
                "method": "kmedoids_behavior_pca",
                "k": int(k),
                "cluster_id": cid,
                "cluster_size": sizes.get(cid, 0),
                "medoid_year": years[int(medoids[cid])] if cid < len(medoids) else np.nan,
                "medoid_year_flag": bool(n == int(medoids[cid])) if cid < len(medoids) else False,
                "distance_to_medoid": float(D[n, int(medoids[cid])]) if cid < len(medoids) else np.nan,
                "small_cluster_warning": sizes.get(cid, 0) < cfg.min_cluster_size_for_type_peak,
                "feature_source": "whole_window_behavior_features_not_single_year_peak",
            })
    return pd.DataFrame(membership_rows), pd.DataFrame(stability_rows), pd.DataFrame(pca_rows)


def _run_mini_som(features: pd.DataFrame, cfg: V91BConfig) -> pd.DataFrame:
    X, _cols, _evr = _feature_matrix_for_clustering(features)
    X = np.asarray(X, dtype=float)
    years = features["year"].to_numpy()
    window_id = str(features["window_id"].iloc[0])
    rows = []
    if X.shape[0] == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(20260901)
    for gx, gy in cfg.som_grids:
        n_nodes = gx * gy
        if n_nodes > X.shape[0]:
            continue
        # initialize with random observations
        init_idx = rng.choice(X.shape[0], n_nodes, replace=False)
        W = X[init_idx].copy()
        coords = np.array([(i, j) for i in range(gx) for j in range(gy)], dtype=float)
        for it in range(max(1, cfg.som_iter)):
            x = X[int(rng.integers(0, X.shape[0]))]
            d = np.sum((W - x) ** 2, axis=1)
            bmu = int(np.argmin(d))
            frac = 1.0 - it / max(cfg.som_iter, 1)
            lr = 0.4 * frac + 0.03
            sigma = max(gx, gy) * 0.5 * frac + 0.3
            grid_d2 = np.sum((coords - coords[bmu]) ** 2, axis=1)
            h = np.exp(-grid_d2 / (2 * sigma * sigma))[:, None]
            W += lr * h * (x - W)
        d_all = ((X[:, None, :] - W[None, :, :]) ** 2).sum(axis=2)
        node = np.argmin(d_all, axis=1)
        counts = {int(i): int((node == i).sum()) for i in np.unique(node)}
        for n, year in enumerate(years):
            nd = int(node[n]); x, y = coords[nd].astype(int).tolist()
            rows.append({
                "window_id": window_id,
                "year": year,
                "som_grid": f"{gx}x{gy}",
                "node_id": nd,
                "node_x": x,
                "node_y": y,
                "node_count": counts.get(nd, 0),
                "distance_to_node_center": float(np.sqrt(d_all[n, nd])),
                "som_role": "topology_audit_only_not_primary_type_evidence",
            })
    return pd.DataFrame(rows)


def _subset_profiles(profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], idx: Sequence[int]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    idx = np.asarray(idx, dtype=int)
    out = {}
    for obj, (prof, lat, w) in profiles.items():
        out[obj] = (np.asarray(prof)[idx, ...].copy(), lat, w)
    return out


def _order_level(prob: float, cfg: V91BConfig) -> str:
    if prob >= cfg.evidence_strict:
        return "strict_99"
    if prob >= cfg.evidence_credible:
        return "credible_95"
    if prob >= cfg.evidence_usable:
        return "usable_90"
    return "unresolved"


def _add_order_levels(order_df: pd.DataFrame, cfg: V91BConfig) -> pd.DataFrame:
    if order_df is None or order_df.empty:
        return pd.DataFrame()
    out = order_df.copy()
    levels = []
    dirs = []
    probs = []
    for _, r in out.iterrows():
        pa = float(r.get("P_A_earlier", np.nan))
        pb = float(r.get("P_B_earlier", np.nan))
        if np.nan_to_num(pa, nan=-1) >= np.nan_to_num(pb, nan=-1):
            p = pa; direction = "A_earlier"
        else:
            p = pb; direction = "B_earlier"
        probs.append(p); dirs.append(direction); levels.append(_order_level(p, cfg))
    out["type_order_best_direction"] = dirs
    out["type_order_best_probability"] = probs
    out["type_order_evidence_level"] = levels
    return out


def _run_type_level_peak_for_window(
    v7multi,
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    years: np.ndarray,
    scope: object,
    membership: pd.DataFrame,
    stability: pd.DataFrame,
    cfg_v7,
    cfg91: V91BConfig,
    out_win: Path,
) -> Dict[str, pd.DataFrame]:
    obj_peak_parts, obj_boot_parts, order_parts, sync_parts, skipped_rows = [], [], [], [], []
    if membership.empty:
        return {"object_peak": pd.DataFrame(), "object_bootstrap": pd.DataFrame(), "order": pd.DataFrame(), "sync": pd.DataFrame(), "skipped": pd.DataFrame()}
    # Only run primary kmedoids method. SOM remains auxiliary and is not type-peak evidence in this version.
    mm = membership[membership["method"] == "kmedoids_behavior_pca"].copy()
    for (method, k, cluster_id), sub in mm.groupby(["method", "k", "cluster_id"]):
        cluster_years = sub["year"].to_numpy()
        idx = [int(np.where(years == y)[0][0]) for y in cluster_years if y in set(years.tolist())]
        cluster_size = len(idx)
        stab = stability[(stability["method"] == method) & (stability["k"] == k)]
        qual = str(stab["cluster_quality_status"].iloc[0]) if not stab.empty else "unknown"
        if cluster_size < cfg91.min_cluster_size_for_type_peak:
            skipped_rows.append({
                "window_id": scope.window_id, "method": method, "k": int(k), "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_size), "skip_reason": "cluster_size_below_min_cluster_size_for_type_peak",
                "min_cluster_size_for_type_peak": int(cfg91.min_cluster_size_for_type_peak),
            })
            continue
        _log(f"    type peak {scope.window_id}: {method} k={k} cluster={cluster_id} n={cluster_size}")
        sub_profiles = _subset_profiles(profiles, idx)
        try:
            score_df, cand_df, selection_df, selected_delta_df, boot_peak_days_df = v7multi._run_detector_and_bootstrap(sub_profiles, scope, cfg_v7)
            timing_audit_df, tau_df = v7multi._estimate_timing_resolution(selection_df, boot_peak_days_df, cfg_v7, scope)
            order_df = v7multi._pairwise_peak_order(selection_df, boot_peak_days_df, scope)
            sync_df = v7multi._pairwise_synchrony(order_df, boot_peak_days_df, tau_df, scope)
        except Exception as exc:
            skipped_rows.append({
                "window_id": scope.window_id, "method": method, "k": int(k), "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_size), "skip_reason": "type_peak_runtime_error", "error": str(exc),
            })
            continue
        add = {
            "window_id": scope.window_id,
            "method": method,
            "k": int(k),
            "cluster_id": int(cluster_id),
            "cluster_size": int(cluster_size),
            "cluster_quality_status": qual,
            "years_in_type": ";".join(str(x) for x in sorted(cluster_years)),
            "v9_1b_role": "type_group_multi_year_peak_not_single_year_peak",
        }
        for df, parts in [(selection_df, obj_peak_parts), (timing_audit_df, obj_boot_parts), (order_df, order_parts), (sync_df, sync_parts)]:
            d = df.copy()
            for kk, vv in add.items():
                d[kk] = vv
            parts.append(d)
    order_all = pd.concat(order_parts, ignore_index=True) if order_parts else pd.DataFrame()
    order_all = _add_order_levels(order_all, cfg91)
    return {
        "object_peak": pd.concat(obj_peak_parts, ignore_index=True) if obj_peak_parts else pd.DataFrame(),
        "object_bootstrap": pd.concat(obj_boot_parts, ignore_index=True) if obj_boot_parts else pd.DataFrame(),
        "order": order_all,
        "sync": pd.concat(sync_parts, ignore_index=True) if sync_parts else pd.DataFrame(),
        "skipped": pd.DataFrame(skipped_rows),
    }


def _v9_order_lookup(v9_order: pd.DataFrame, window_id: str) -> pd.DataFrame:
    if v9_order is None or v9_order.empty:
        return pd.DataFrame()
    return v9_order[v9_order["window_id"].astype(str) == str(window_id)].copy()


def _best_prob_from_v9_row(r: pd.Series) -> Tuple[str, float, str]:
    pa = float(r.get("P_A_earlier", np.nan)); pb = float(r.get("P_B_earlier", np.nan))
    if np.nan_to_num(pa, nan=-1) >= np.nan_to_num(pb, nan=-1):
        return "A_earlier", pa, str(r.get("peak_order_decision", ""))
    return "B_earlier", pb, str(r.get("peak_order_decision", ""))


def _build_ordinary_vs_type_audit(type_order: pd.DataFrame, v9_order_win: pd.DataFrame, cfg: V91BConfig) -> pd.DataFrame:
    if type_order.empty:
        return pd.DataFrame()
    rows = []
    for _, r in type_order.iterrows():
        A, B = r.get("object_A"), r.get("object_B")
        match = v9_order_win[(v9_order_win["object_A"].astype(str) == str(A)) & (v9_order_win["object_B"].astype(str) == str(B))]
        v9_dir = "missing"; v9_prob = np.nan; v9_dec = "missing"
        if not match.empty:
            v9_dir, v9_prob, v9_dec = _best_prob_from_v9_row(match.iloc[0])
        type_dir = str(r.get("type_order_best_direction", ""))
        type_prob = float(r.get("type_order_best_probability", np.nan))
        type_level = str(r.get("type_order_evidence_level", "unresolved"))
        v9_level = _order_level(v9_prob, cfg) if np.isfinite(v9_prob) else "missing"
        rows.append({
            "window_id": r.get("window_id"),
            "method": r.get("method"),
            "k": r.get("k"),
            "cluster_id": r.get("cluster_id"),
            "cluster_size": r.get("cluster_size"),
            "object_A": A,
            "object_B": B,
            "v9_best_direction": v9_dir,
            "v9_best_probability": v9_prob,
            "v9_order_level": v9_level,
            "v9_order_decision_original": v9_dec,
            "type_best_direction": type_dir,
            "type_best_probability": type_prob,
            "type_order_level": type_level,
            "type_improves_over_v9": bool(np.isfinite(type_prob) and np.isfinite(v9_prob) and type_prob >= v9_prob + 0.05),
            "opposite_order_vs_v9": bool(v9_dir in ("A_earlier", "B_earlier") and type_dir in ("A_earlier", "B_earlier") and v9_dir != type_dir),
            "type_internal_stabilization": bool(type_level in ("usable_90", "credible_95", "strict_99") and v9_level not in ("credible_95", "strict_99")),
        })
    return pd.DataFrame(rows)


def _build_mixture_evidence(ordinary_vs: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    if ordinary_vs.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["window_id", "method", "k", "object_A", "object_B"]
    for keys, sub in ordinary_vs.groupby(group_cols):
        window_id, method, k, A, B = keys
        stab = stability[(stability["window_id"].astype(str) == str(window_id)) & (stability["method"].astype(str) == str(method)) & (stability["k"].astype(int) == int(k))]
        cluster_status = str(stab["cluster_quality_status"].iloc[0]) if not stab.empty else "unknown"
        n_usable = int(sub["type_order_level"].isin(["usable_90", "credible_95", "strict_99"]).sum())
        n_credible = int(sub["type_order_level"].isin(["credible_95", "strict_99"]).sum())
        n_strict = int((sub["type_order_level"] == "strict_99").sum())
        dirs = set(sub.loc[sub["type_order_level"].isin(["usable_90", "credible_95", "strict_99"]), "type_best_direction"].astype(str).tolist())
        opposite_across = len(dirs.intersection({"A_earlier", "B_earlier"})) >= 2
        any_improve = bool(sub["type_internal_stabilization"].any())
        small = bool("small" in cluster_status)
        unstable = bool("unstable" in cluster_status)
        if small:
            level = "insufficient_group_size"
        elif unstable:
            level = "unstable_clustering"
        elif n_credible >= 2 and opposite_across and any_improve:
            level = "strong_mixture_candidate"
        elif (n_credible >= 1 and any_improve) or (n_usable >= 2 and opposite_across):
            level = "moderate_mixture_candidate"
        elif n_usable >= 1:
            level = "weak_mixture_hint"
        else:
            level = "not_supported"
        rows.append({
            "window_id": window_id,
            "method": method,
            "k": int(k),
            "object_A": A,
            "object_B": B,
            "object_or_pair": f"{A}-{B}",
            "cluster_quality_status": cluster_status,
            "n_types_tested": int(sub["cluster_id"].nunique()),
            "n_types_with_usable_order": n_usable,
            "n_types_with_credible_order": n_credible,
            "n_types_with_strict_order": n_strict,
            "opposite_order_across_types": bool(opposite_across),
            "type_internal_stabilization": bool(any_improve),
            "mixture_evidence_level": level,
            "recommended_interpretation": _mixture_interpretation(level),
        })
    return pd.DataFrame(rows)


def _mixture_interpretation(level: str) -> str:
    if level == "strong_mixture_candidate":
        return "candidate transition types may explain full-sample peak/order instability; inspect clusters before any physical interpretation"
    if level == "moderate_mixture_candidate":
        return "some type-level stabilization is present; treat as hypothesis-generating evidence"
    if level == "weak_mixture_hint":
        return "weak within-type order signal; not enough to explain full-sample instability"
    if level == "insufficient_group_size":
        return "candidate types are too small for reliable type-level peak bootstrap"
    if level == "unstable_clustering":
        return "year grouping is unstable; do not interpret type-level peak results as robust"
    return "no evidence that this clustering explains V9 peak/order instability"


def _write_summary(path: Path, all_evidence: pd.DataFrame, cfg: V91BConfig) -> None:
    lines = [
        "# V9.1_b transition-type mixture audit summary",
        "",
        f"version: `{VERSION}`",
        "",
        "## Method boundary",
        "- This audit is read-only relative to V9 and does not modify V9 outputs.",
        "- It does not cluster single-year peak days.",
        "- It clusters whole-window multi-object behaviour features, then reruns V9 peak logic inside type groups.",
        "- It does not assign physical meanings to clusters.",
        "",
        "## Configuration",
        f"- windows: {', '.join(cfg.windows)}",
        f"- k_values: {cfg.k_values}",
        f"- type_bootstrap_n: {cfg.type_bootstrap_n}",
        f"- min_cluster_size_for_type_peak: {cfg.min_cluster_size_for_type_peak}",
        "",
        "## Evidence counts",
    ]
    if all_evidence is not None and not all_evidence.empty and "mixture_evidence_level" in all_evidence.columns:
        for k, v in all_evidence["mixture_evidence_level"].value_counts(dropna=False).items():
            lines.append(f"- {k}: {int(v)}")
    else:
        lines.append("- no evidence table generated")
    lines += [
        "",
        "## Interpretation rules",
        "- strong/moderate mixture candidates are statistical transition-type hypotheses, not physical regimes.",
        "- no V9 peak result is replaced by V9.1_b.",
        "- If type-level peak/order does not stabilize, the mixture hypothesis is not supported in the tested feature space.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_transition_type_mixture_audit_v9_1_b(v91_root: Path | str) -> None:
    v91_root = Path(v91_root)
    stage_root = _stage_root_from_v91(v91_root)
    cfg91 = V91BConfig.from_env()
    v7multi = _import_v7_module(stage_root)
    cfg_v7 = _make_v7_cfg_for_v91(v7multi, cfg91, stage_root)

    out_root = _ensure_dir(v91_root / "outputs" / OUTPUT_TAG)
    out_per = _ensure_dir(out_root / "per_window")
    out_cross = _ensure_dir(out_root / "cross_window")
    log_dir = _ensure_dir(v91_root / "logs" / OUTPUT_TAG)
    t0 = time.time()

    _log("[1/8] Load V9 context and V7/V9 peak-compatible inputs")
    v9_tables = _load_v9_tables(stage_root)
    scopes = _window_scopes_from_v7(v7multi, stage_root, cfg_v7)
    profiles, years, object_registry = _load_fields_and_profiles(v7multi, stage_root, cfg_v7)
    _safe_to_csv(object_registry, out_cross / "object_registry_v9_1_b.csv")
    _safe_to_csv(pd.DataFrame(EXCLUDED_WINDOWS), out_cross / "excluded_windows_from_v9_1_b.csv")

    all_feature_parts, all_evr_parts, all_membership_parts, all_stability_parts = [], [], [], []
    all_som_parts, all_type_peak_parts, all_type_boot_parts, all_type_order_parts, all_type_sync_parts = [], [], [], [], []
    all_skipped_parts, all_ordinary_parts, all_evidence_parts = [], [], []

    for wi, scope in enumerate(scopes, start=1):
        if scope.window_id not in cfg91.windows:
            continue
        _log(f"[2/8] {scope.window_id}: build whole-window multi-object behaviour features ({wi}/{len(scopes)})")
        out_win = _ensure_dir(out_per / scope.window_id)
        features, evr = _build_window_year_behavior_features(profiles, years, scope, cfg91)
        _safe_to_csv(features, out_win / f"window_year_behavior_features_{scope.window_id}.csv")
        _safe_to_csv(evr, out_win / f"feature_pca_explained_variance_by_object_{scope.window_id}.csv")
        all_feature_parts.append(features); all_evr_parts.append(evr)

        _log(f"[3/8] {scope.window_id}: cluster behaviour features k={cfg91.k_values}")
        membership, stability, feature_evr = _run_kmedoids_suite(features, cfg91)
        _safe_to_csv(feature_evr, out_win / f"feature_pca_explained_variance_{scope.window_id}.csv")
        _safe_to_csv(membership, out_win / f"transition_type_membership_{scope.window_id}.csv")
        _safe_to_csv(stability, out_win / f"transition_type_cluster_stability_{scope.window_id}.csv")
        all_membership_parts.append(membership); all_stability_parts.append(stability)

        _log(f"[4/8] {scope.window_id}: run auxiliary SOM topology audit")
        som = _run_mini_som(features, cfg91)
        _safe_to_csv(som, out_win / f"som_node_assignment_{scope.window_id}.csv")
        all_som_parts.append(som)

        _log(f"[5/8] {scope.window_id}: rerun V9 peak inside candidate type groups")
        type_res = _run_type_level_peak_for_window(v7multi, profiles, years, scope, membership, stability, cfg_v7, cfg91, out_win)
        _safe_to_csv(type_res["object_peak"], out_win / f"type_level_object_peak_registry_{scope.window_id}.csv")
        _safe_to_csv(type_res["object_bootstrap"], out_win / f"type_level_object_peak_bootstrap_{scope.window_id}.csv")
        _safe_to_csv(type_res["order"], out_win / f"type_level_pairwise_peak_order_{scope.window_id}.csv")
        _safe_to_csv(type_res["sync"], out_win / f"type_level_pairwise_peak_synchrony_{scope.window_id}.csv")
        _safe_to_csv(type_res["skipped"], out_win / f"type_level_peak_skipped_clusters_{scope.window_id}.csv")
        all_type_peak_parts.append(type_res["object_peak"]); all_type_boot_parts.append(type_res["object_bootstrap"])
        all_type_order_parts.append(type_res["order"]); all_type_sync_parts.append(type_res["sync"]); all_skipped_parts.append(type_res["skipped"])

        _log(f"[6/8] {scope.window_id}: compare type-level peak/order against V9 full sample")
        v9_order_win = _v9_order_lookup(v9_tables.get("pairwise_order", pd.DataFrame()), scope.window_id)
        ordinary = _build_ordinary_vs_type_audit(type_res["order"], v9_order_win, cfg91)
        evidence = _build_mixture_evidence(ordinary, stability)
        _safe_to_csv(ordinary, out_win / f"ordinary_vs_type_level_peak_audit_{scope.window_id}.csv")
        _safe_to_csv(evidence, out_win / f"transition_type_mixture_evidence_{scope.window_id}.csv")
        all_ordinary_parts.append(ordinary); all_evidence_parts.append(evidence)

    _log("[7/8] Write cross-window outputs")
    cross = {
        "window_year_behavior_features_all_windows.csv": pd.concat(all_feature_parts, ignore_index=True) if all_feature_parts else pd.DataFrame(),
        "feature_pca_explained_variance_by_object_all_windows.csv": pd.concat(all_evr_parts, ignore_index=True) if all_evr_parts else pd.DataFrame(),
        "transition_type_membership_all_windows.csv": pd.concat(all_membership_parts, ignore_index=True) if all_membership_parts else pd.DataFrame(),
        "cluster_stability_all_windows.csv": pd.concat(all_stability_parts, ignore_index=True) if all_stability_parts else pd.DataFrame(),
        "som_node_assignment_all_windows.csv": pd.concat(all_som_parts, ignore_index=True) if all_som_parts else pd.DataFrame(),
        "type_level_object_peak_registry_all_windows.csv": pd.concat(all_type_peak_parts, ignore_index=True) if all_type_peak_parts else pd.DataFrame(),
        "type_level_object_peak_bootstrap_all_windows.csv": pd.concat(all_type_boot_parts, ignore_index=True) if all_type_boot_parts else pd.DataFrame(),
        "type_level_pairwise_peak_order_all_windows.csv": pd.concat(all_type_order_parts, ignore_index=True) if all_type_order_parts else pd.DataFrame(),
        "type_level_pairwise_peak_synchrony_all_windows.csv": pd.concat(all_type_sync_parts, ignore_index=True) if all_type_sync_parts else pd.DataFrame(),
        "type_level_peak_skipped_clusters_all_windows.csv": pd.concat(all_skipped_parts, ignore_index=True) if all_skipped_parts else pd.DataFrame(),
        "ordinary_vs_type_level_peak_audit_all_windows.csv": pd.concat(all_ordinary_parts, ignore_index=True) if all_ordinary_parts else pd.DataFrame(),
        "transition_type_mixture_evidence_all_windows.csv": pd.concat(all_evidence_parts, ignore_index=True) if all_evidence_parts else pd.DataFrame(),
    }
    for fname, df in cross.items():
        _safe_to_csv(df, out_cross / fname)

    _log("[8/8] Write run metadata and summary")
    run_meta = {
        "version": VERSION,
        "output_tag": OUTPUT_TAG,
        "parent_version": "V9 peak_all_windows_v9_a",
        "modifies_v9": False,
        "reads_v9_outputs": True,
        "uses_single_year_peak_for_clustering": False,
        "feature_basis": "whole-window multi-object behaviour features",
        "windows": cfg91.windows,
        "excluded_windows": EXCLUDED_WINDOWS,
        "state_included": False,
        "growth_included": False,
        "process_a_included": False,
        "config": asdict(cfg91),
        "source_v7_module_version": getattr(v7multi, "VERSION", "unknown"),
        "source_v7_output_tag": getattr(v7multi, "OUTPUT_TAG", "unknown"),
        "elapsed_seconds": round(time.time() - t0, 3),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(run_meta, out_root / "run_meta.json")
    _write_json(run_meta, out_cross / "run_meta.json")
    _write_summary(out_cross / "transition_type_mixture_audit_summary.md", cross["transition_type_mixture_evidence_all_windows.csv"], cfg91)
    (log_dir / "run_progress.log").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Done. Outputs written to {out_root}")
