from __future__ import annotations

import numpy as np
import pandas as pd

from .config import W045PreclusterConfig, ClusterDef


def _cluster_peak(curve: pd.DataFrame, obj: str, cluster: ClusterDef) -> tuple[float, float]:
    part = curve[(curve["object"] == obj) & (curve["day"] >= cluster.day_min) & (curve["day"] <= cluster.day_max)].copy()
    if part.empty:
        return np.nan, np.nan
    idx = part["value"].idxmax()
    return float(part.loc[idx, "day"]), float(part.loc[idx, "value"])


def _valley(curve: pd.DataFrame, obj: str, left: ClusterDef, right: ClusterDef) -> tuple[float, float]:
    part = curve[(curve["object"] == obj) & (curve["day"] > left.day_max) & (curve["day"] < right.day_min)].copy()
    if part.empty:
        return np.nan, np.nan
    idx = part["value"].idxmin()
    return float(part.loc[idx, "day"]), float(part.loc[idx, "value"])


def _classify(has_left_marker: bool, has_right_marker: bool, rel_depth: float, sep_days: float) -> str:
    if not has_left_marker and not has_right_marker:
        return "weak_or_unclear"
    if has_left_marker and not has_right_marker:
        return "left_peak_only"
    if has_right_marker and not has_left_marker:
        return "right_peak_only"
    if np.isnan(rel_depth):
        return "weak_or_unclear"
    if rel_depth >= 0.30 and sep_days >= 5:
        return "separated_double_peak"
    if rel_depth <= 0.15:
        return "broad_continuous_adjustment"
    return "partially_separated_or_unclear"


def build_morphology_table(cfg: W045PreclusterConfig, main_curve: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    cluster_map = {c.cluster_id: c for c in cfg.clusters}
    pairs = [
        (cluster_map["E1_early_precluster"], cluster_map["E2_second_precluster"]),
        (cluster_map["E2_second_precluster"], cluster_map["M_w045_main_cluster"]),
        (cluster_map["M_w045_main_cluster"], cluster_map["H_post_reference"]),
    ]
    rows = []
    for obj in cfg.objects:
        for left, right in pairs:
            ld, lv = _cluster_peak(main_curve, obj, left)
            rd, rv = _cluster_peak(main_curve, obj, right)
            vd, vv = _valley(main_curve, obj, left, right)
            min_peak = np.nanmin([lv, rv]) if not (np.isnan(lv) and np.isnan(rv)) else np.nan
            absolute_valley_depth = float(min_peak - vv) if not np.isnan(min_peak) and not np.isnan(vv) else np.nan
            relative_valley_depth = float(absolute_valley_depth / min_peak) if min_peak and not np.isnan(min_peak) else np.nan
            sep = float(abs(rd - ld)) if not np.isnan(ld) and not np.isnan(rd) else np.nan
            left_marker = bool(metrics[(metrics["cluster_id"] == left.cluster_id) & (metrics["object"] == obj)]["candidate_inside_cluster_flag"].iloc[0])
            right_marker = bool(metrics[(metrics["cluster_id"] == right.cluster_id) & (metrics["object"] == obj)]["candidate_inside_cluster_flag"].iloc[0])
            rows.append({
                "object": obj,
                "left_cluster": left.cluster_id,
                "right_cluster": right.cluster_id,
                "left_peak_day": ld,
                "right_peak_day": rd,
                "left_peak_score": lv,
                "right_peak_score": rv,
                "valley_day": vd,
                "valley_score": vv,
                "absolute_valley_depth": absolute_valley_depth,
                "relative_valley_depth": relative_valley_depth,
                "peak_separation_days": sep,
                "left_candidate_inside": left_marker,
                "right_candidate_inside": right_marker,
                "morphology_class": _classify(left_marker, right_marker, relative_valley_depth, sep),
            })
    return pd.DataFrame(rows)
