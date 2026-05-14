from __future__ import annotations

import numpy as np
import pandas as pd

from .config import W045PreclusterConfig, ClusterDef


MARKER_SUPPORTED_CLASS = "candidate_inside_cluster"
CURVE_ONLY_CLASS = "curve_peak_without_marker"
WEAK_CURVE_CLASS = "weak_curve_signal"
NO_SIGNAL_CLASS = "no_signal"
MISSING_CLASS = "missing_input"


def _subset_cluster(curve: pd.DataFrame, obj: str, cluster: ClusterDef) -> pd.DataFrame:
    part = curve[(curve["object"] == obj) & (curve["day"] >= cluster.day_min) & (curve["day"] <= cluster.day_max)].copy()
    return part.sort_values("day")


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper for NumPy 2.x and older NumPy releases.

    NumPy 2.x removed ``np.trapz`` in favor of ``np.trapezoid``.
    Some older environments may still expose only ``np.trapz``.
    Keep the integration semantics unchanged while avoiding environment-specific crashes.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    # Last-resort manual trapezoidal integration. This should rarely be used,
    # but makes the audit independent of NumPy API churn.
    if len(y) < 2:
        return float(y[0]) if len(y) == 1 else 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _max_metrics(part: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if part.empty:
        return None, None, None
    idx = part["value"].idxmax()
    max_day = float(part.loc[idx, "day"])
    max_value = float(part.loc[idx, "value"])
    if len(part) >= 2:
        auc = _integrate_trapezoid(part["value"].to_numpy(), part["day"].to_numpy())
    else:
        auc = float(max_value)
    return max_day, max_value, auc


def _nearest_marker(markers: pd.DataFrame, obj: str, cluster: ClusterDef) -> dict:
    rows = markers[markers["object"] == obj].copy()
    if rows.empty:
        return {
            "nearest_candidate_day": np.nan,
            "nearest_candidate_score": np.nan,
            "nearest_candidate_distance": np.nan,
            "candidate_inside_cluster_flag": False,
        }
    rows["distance_to_center"] = (rows["candidate_day"] - cluster.center_day).abs()
    nearest = rows.sort_values("distance_to_center").iloc[0]
    inside_rows = rows[(rows["candidate_day"] >= cluster.day_min) & (rows["candidate_day"] <= cluster.day_max)]
    if not inside_rows.empty:
        inside_rows = inside_rows.assign(distance_to_center=(inside_rows["candidate_day"] - cluster.center_day).abs())
        inside = inside_rows.sort_values("distance_to_center").iloc[0]
        return {
            "nearest_candidate_day": float(inside["candidate_day"]),
            "nearest_candidate_score": float(inside.get("candidate_score", np.nan)),
            "nearest_candidate_distance": float(abs(inside["candidate_day"] - cluster.center_day)),
            "candidate_inside_cluster_flag": True,
        }
    return {
        "nearest_candidate_day": float(nearest["candidate_day"]),
        "nearest_candidate_score": float(nearest.get("candidate_score", np.nan)),
        "nearest_candidate_distance": float(abs(nearest["candidate_day"] - cluster.center_day)),
        "candidate_inside_cluster_flag": False,
    }


def _profile_cluster_metrics(profile_curve: pd.DataFrame, obj: str, cluster: ClusterDef, k_values: tuple[int, ...]) -> dict:
    out: dict[str, float | None] = {}
    if profile_curve.empty or "k" not in profile_curve.columns:
        for k in k_values:
            out[f"profile_energy_max_day_k{k}"] = np.nan
            out[f"profile_energy_max_score_k{k}"] = np.nan
        return out
    for k in k_values:
        part = profile_curve[
            (profile_curve["object"] == obj)
            & (profile_curve["k"] == k)
            & (profile_curve["day"] >= cluster.day_min)
            & (profile_curve["day"] <= cluster.day_max)
        ].copy()
        d, v, _ = _max_metrics(part)
        out[f"profile_energy_max_day_k{k}"] = d if d is not None else np.nan
        out[f"profile_energy_max_score_k{k}"] = v if v is not None else np.nan
    return out


def classify_participation(candidate_inside: bool, rel_strength: float | None) -> str:
    """Classify raw curve/marker evidence inside a cluster.

    HOTFIX02 keeps the original class names for backward compatibility but
    changes downstream interpretation: ``curve_peak_without_marker`` is no
    longer treated as equal to marker-supported activity. It is interpreted as
    curve-only ramp/shoulder evidence.
    """
    if candidate_inside:
        return MARKER_SUPPORTED_CLASS
    if rel_strength is None or np.isnan(rel_strength):
        return MISSING_CLASS
    if rel_strength >= 0.35:
        return CURVE_ONLY_CLASS
    if rel_strength >= 0.15:
        return WEAK_CURVE_CLASS
    return NO_SIGNAL_CLASS


def participation_tier(participation_class: str) -> str:
    """Translate raw participation class into interpretation-safe evidence tiers."""
    if participation_class == MARKER_SUPPORTED_CLASS:
        return "marker_supported_active"
    if participation_class == CURVE_ONLY_CLASS:
        return "curve_only_ramp_or_shoulder"
    if participation_class == WEAK_CURVE_CLASS:
        return "weak_curve_signal"
    if participation_class in {NO_SIGNAL_CLASS, MISSING_CLASS}:
        return "absent_or_missing"
    return "unknown"


def build_object_cluster_metrics(
    cfg: W045PreclusterConfig,
    main_curve: pd.DataFrame,
    profile_curve: pd.DataFrame,
    markers: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    # Object-level normalization, used only for descriptive participation class.
    full_max = main_curve.groupby("object")["value"].max().to_dict() if not main_curve.empty else {}
    for cluster in cfg.clusters:
        for obj in cfg.objects:
            part = _subset_cluster(main_curve, obj, cluster)
            max_day, max_score, auc = _max_metrics(part)
            marker_info = _nearest_marker(markers, obj, cluster)
            denom = full_max.get(obj, np.nan)
            rel = float(max_score / denom) if max_score is not None and denom and not np.isnan(denom) else np.nan
            pclass = classify_participation(bool(marker_info["candidate_inside_cluster_flag"]), rel)
            row = {
                "cluster_id": cluster.cluster_id,
                "object": obj,
                "day_min": cluster.day_min,
                "day_max": cluster.day_max,
                "center_day": cluster.center_day,
                "max_day_main": max_day if max_day is not None else np.nan,
                "max_score_main": max_score if max_score is not None else np.nan,
                "auc_main": auc if auc is not None else np.nan,
                "relative_main_strength_to_object_fullseason_max": rel,
                **marker_info,
            }
            row.update(_profile_cluster_metrics(profile_curve, obj, cluster, cfg.profile_k_values))
            row["participation_class"] = pclass
            row["participation_tier"] = participation_tier(pclass)
            row["marker_supported_active_flag"] = pclass == MARKER_SUPPORTED_CLASS
            row["curve_only_ramp_or_shoulder_flag"] = pclass == CURVE_ONLY_CLASS
            row["weak_curve_signal_flag"] = pclass == WEAK_CURVE_CLASS
            row["absent_or_missing_flag"] = pclass in {NO_SIGNAL_CLASS, MISSING_CLASS}
            rows.append(row)
    return pd.DataFrame(rows)


def build_participation_matrix(metrics: pd.DataFrame, objects: tuple[str, ...]) -> pd.DataFrame:
    """Build interpretation-safe participation summary.

    HOTFIX02: do not merge marker-supported evidence and curve-only/ramp evidence.
    The previous ``dominant_or_active_objects`` field was too broad and could
    make E2 look as if joint_all/Jw were candidate-supported participants.
    """
    rows = []
    for cluster, g in metrics.groupby("cluster_id", sort=False):
        row = {"cluster_id": cluster}
        marker_supported: list[str] = []
        curve_only: list[str] = []
        weak_curve: list[str] = []
        absent: list[str] = []
        for obj in objects:
            part = g[g["object"] == obj]
            cls = part["participation_class"].iloc[0] if not part.empty else MISSING_CLASS
            tier = participation_tier(cls)
            row[obj] = cls
            row[f"{obj}_tier"] = tier
            if tier == "marker_supported_active":
                marker_supported.append(obj)
            elif tier == "curve_only_ramp_or_shoulder":
                curve_only.append(obj)
            elif tier == "weak_curve_signal":
                weak_curve.append(obj)
            elif tier == "absent_or_missing":
                absent.append(obj)
        row["marker_supported_active_objects"] = ";".join(marker_supported)
        row["curve_only_ramp_or_shoulder_objects"] = ";".join(curve_only)
        row["weak_curve_signal_objects"] = ";".join(weak_curve)
        row["absent_or_missing_objects"] = ";".join(absent)
        row["event_semantics_core_objects"] = ";".join(marker_supported)
        row["interpretation_note"] = (
            "marker_supported_active is the event-semantics core. "
            "curve_only_ramp_or_shoulder indicates curve elevation without an in-cluster candidate marker and should not be treated as equal active participation."
        )
        rows.append(row)
    return pd.DataFrame(rows)
