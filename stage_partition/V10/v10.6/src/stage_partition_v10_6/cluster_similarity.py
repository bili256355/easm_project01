from __future__ import annotations

import numpy as np
import pandas as pd

from .config import W045PreclusterConfig


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def build_cluster_similarity(cfg: W045PreclusterConfig, metrics: pd.DataFrame) -> pd.DataFrame:
    # Use relative strength vector for all objects, excluding H_post for main E1/E2/M comparisons.
    main_clusters = [c.cluster_id for c in cfg.clusters if c.included_in_order_test]
    vecs: dict[str, np.ndarray] = {}
    active: dict[str, set[str]] = {}
    for cid in main_clusters:
        vals = []
        acts = set()
        for obj in cfg.objects:
            part = metrics[(metrics["cluster_id"] == cid) & (metrics["object"] == obj)]
            if part.empty:
                vals.append(0.0)
                continue
            v = part["relative_main_strength_to_object_fullseason_max"].iloc[0]
            vals.append(float(v) if not pd.isna(v) else 0.0)
            if part["participation_class"].iloc[0] in {"candidate_inside_cluster", "curve_peak_without_marker"}:
                acts.add(obj)
        vecs[cid] = np.asarray(vals, dtype=float)
        active[cid] = acts
    rows = []
    for i, a in enumerate(main_clusters):
        for b in main_clusters[i + 1:]:
            shared = sorted(active[a].intersection(active[b]))
            different = sorted(active[a].symmetric_difference(active[b]))
            cos = _safe_cosine(vecs[a], vecs[b])
            corr = _safe_corr(vecs[a], vecs[b])
            hint = ""
            if cos >= 0.75:
                hint = "high_vector_similarity_possible_continuity"
            elif cos <= 0.35:
                hint = "low_vector_similarity_possible_distinct_events"
            else:
                hint = "moderate_similarity_needs_morphology_or_yearwise_check"
            rows.append({
                "cluster_a": a,
                "cluster_b": b,
                "metric_source": "relative_main_strength_to_object_fullseason_max",
                "cosine_similarity": cos,
                "pearson_similarity": corr,
                "shared_active_objects": ";".join(shared),
                "different_active_objects": ";".join(different),
                "interpretation_hint": hint,
            })
    return pd.DataFrame(rows)
