from __future__ import annotations

import numpy as np
import pandas as pd


def _event_strength_class(profile_diff: pd.DataFrame, event_id: str) -> str:
    g = profile_diff.loc[profile_diff["event_id"] == event_id]
    if g.empty:
        return "profile_not_available"
    total = float(np.nansum(g["abs_diff"].to_numpy(dtype=float)))
    if not np.isfinite(total):
        return "profile_not_available"
    # Relative classes are refined later against all events.
    return "computed"


def classify_event_content(profile_diff: pd.DataFrame, spatial_metrics: pd.DataFrame, similarity: pd.DataFrame, yearwise_summary: pd.DataFrame, scale_outputs: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    events = ["H18", "H35", "H45", "H57"]
    profile_norm = {}
    if not profile_diff.empty:
        for ev, g in profile_diff.groupby("event_id"):
            profile_norm[str(ev)] = float(np.nansum(g["abs_diff"].to_numpy(dtype=float)))
    spatial_abs = {}
    if spatial_metrics is not None and not spatial_metrics.empty:
        for _, r in spatial_metrics.iterrows():
            spatial_abs[str(r["event_id"])] = float(r.get("field_diff_abs_mean", np.nan))
    ridge_context = _ridge_context(scale_outputs.get("ridge_summary"))
    rows = []
    pvals = np.array([v for v in profile_norm.values() if np.isfinite(v)], dtype=float)
    svals = np.array([v for v in spatial_abs.values() if np.isfinite(v)], dtype=float)
    for ev in events:
        p = profile_norm.get(ev, np.nan)
        s = spatial_abs.get(ev, np.nan)
        pcls = _relative_class(p, pvals, "profile")
        scls = _relative_class(s, svals, "spatial")
        sim_to_h18 = _get_similarity(similarity, "H18", ev)
        ycls = _get_yearwise_class(yearwise_summary, ev)
        ridge = ridge_context.get(ev, "no_scale_context_available")
        role = _content_role(ev, pcls, scls, sim_to_h18, ycls, ridge)
        rows.append({
            "event_id": ev,
            "profile_strength_class": pcls,
            "profile_change_norm_proxy_sum_absdiff": p,
            "spatial_strength_class": scls,
            "spatial_abs_mean": s,
            "profile_similarity_to_H18": sim_to_h18.get("profile_pearson_correlation", np.nan) if isinstance(sim_to_h18, dict) else np.nan,
            "spatial_similarity_to_H18": sim_to_h18.get("spatial_pattern_correlation", np.nan) if isinstance(sim_to_h18, dict) else np.nan,
            "yearwise_consistency_class": ycls,
            "scale_ridge_context_from_v10_7_b": ridge,
            "content_role_class": role,
            "recommended_next_test_target": _next_target(ev, role),
            "forbidden_interpretation": _forbidden(ev),
        })
    return pd.DataFrame(rows)


def _relative_class(v: float, vals: np.ndarray, prefix: str) -> str:
    if not np.isfinite(v) or vals.size == 0:
        return f"{prefix}_not_available"
    q33, q67 = np.nanpercentile(vals, [33, 67]) if vals.size >= 3 else (np.nanmin(vals), np.nanmax(vals))
    if vals.size <= 1:
        return f"{prefix}_computed_single_event"
    if v >= q67:
        return f"{prefix}_strong_relative"
    if v <= q33:
        return f"{prefix}_weak_relative"
    return f"{prefix}_moderate_relative"


def _ridge_context(ridge_df: pd.DataFrame | None) -> dict[str, str]:
    """Map V10.7_b scale-ridge labels into V10.7_c event IDs.

    HOTFIX01: V10.7_b labels the early stable ridge as H19, while
    V10.7_c event-content windows use H18 for the same early event.
    We therefore treat H18 and H19 as aliases for the early-H scale context.
    """
    out = {
        "H18": "no_clear_scale_ridge_near_target",
        "H35": "no_clear_scale_ridge_near_target",
        "H45": "no_clear_scale_ridge_near_target",
        "H57": "no_clear_scale_ridge_near_target",
    }
    if ridge_df is None or ridge_df.empty:
        return {k: "scale_context_not_available" for k in out}

    aliases = {
        "H18": {"labels": {"H18", "H19"}, "days": (18, 19)},
        "H35": {"labels": {"H35"}, "days": (35,)},
        "H45": {"labels": {"H45"}, "days": (45,)},
        "H57": {"labels": {"H57"}, "days": (57,)},
    }
    center_col = None
    for c in ("day_center_weighted", "max_energy_day", "day_center", "day"):
        if c in ridge_df.columns:
            center_col = c
            break

    for label, spec in aliases.items():
        parts = []
        if "nearest_target_label" in ridge_df.columns:
            parts.append(ridge_df.loc[ridge_df["nearest_target_label"].astype(str).isin(spec["labels"])])
        if center_col is not None:
            center = pd.to_numeric(ridge_df[center_col], errors="coerce")
            mask = np.zeros(len(ridge_df), dtype=bool)
            for d in spec["days"]:
                mask |= np.abs(center - float(d)) <= 4
            parts.append(ridge_df.loc[mask])
        if not parts:
            continue
        g = pd.concat(parts, ignore_index=False).drop_duplicates()
        if g.empty:
            continue
        sort_cols = [c for c in ("persistence_fraction", "max_energy_norm", "mean_energy_norm") if c in g.columns]
        if sort_cols:
            best = g.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
        else:
            best = g.iloc[0]
        ridge_target = best.get("nearest_target_label", label)
        out[label] = (
            f"ridge_near_{label};v10_7_b_target={ridge_target};"
            f"role_hint={best.get('role_hint','')};"
            f"persistence={best.get('persistence_fraction', np.nan)}"
        )
    return out


def _get_similarity(df: pd.DataFrame, a: str, b: str):
    if df is None or df.empty or a == b:
        return {"profile_pearson_correlation": 1.0 if a == b else np.nan, "spatial_pattern_correlation": 1.0 if a == b else np.nan}
    key1 = f"{a}_vs_{b}"
    key2 = f"{b}_vs_{a}"
    g = df.loc[df["comparison"].astype(str).isin([key1, key2])]
    if g.empty:
        return {"profile_pearson_correlation": np.nan, "spatial_pattern_correlation": np.nan}
    return g.iloc[0].to_dict()


def _get_yearwise_class(df: pd.DataFrame, ev: str) -> str:
    if df is None or df.empty:
        return "yearwise_not_available"
    g = df.loc[df["event_id"].astype(str) == ev]
    if g.empty:
        return "yearwise_not_available"
    return str(g.iloc[0].get("yearwise_consistency_class", "yearwise_not_available"))


def _content_role(ev: str, pcls: str, scls: str, sim: dict, ycls: str, ridge: str) -> str:
    prof_corr = sim.get("profile_pearson_correlation", np.nan) if isinstance(sim, dict) else np.nan
    spat_corr = sim.get("spatial_pattern_correlation", np.nan) if isinstance(sim, dict) else np.nan
    if ev == "H18":
        if "stable_cross_scale" in ridge or "strong" in pcls:
            return "strong_early_H_adjustment"
        return "early_H_adjustment_content_candidate"
    if ev == "H35":
        if np.isfinite(prof_corr) and prof_corr >= 0.6 and (not np.isfinite(spat_corr) or spat_corr >= 0.4):
            return "same_type_second_H_adjustment"
        if "medium_scale" in ridge or "local_bump" in ridge:
            return "short_scale_local_H_bump"
        if "weak" in pcls and "weak" in scls:
            return "weak_or_unclear_H_change"
        return "different_type_or_unclear_H35_change"
    if ev == "H45":
        if "weak" in pcls and "weak" in scls and "no_clear" in ridge:
            return "H_absent_in_W045_main_cluster"
        return "weak_background_change_or_unclear_H45_control"
    if ev == "H57":
        if "strong" in pcls or "strong" in scls:
            return "post_window_reference_change"
        return "weak_or_unclear_post_window_reference"
    return "unclassified"


def _next_target(ev: str, role: str) -> str:
    if ev == "H18":
        return "Use H18/H18-26 package as primary early-H target for later yearwise/spatial relationship tests."
    if ev == "H35":
        if "same_type" in role:
            return "Test H18-H35 package rather than H35 alone."
        if "local" in role or "different" in role:
            return "If pursued, test H35 as separate E2 local bump; do not assume precursor role."
        return "Use only as secondary target unless yearwise/spatial evidence strengthens it."
    if ev == "H45":
        return "Use as negative/control target for H absence in W045 main cluster."
    if ev == "H57":
        return "Use only as post-window reference target."
    return ""


def _forbidden(ev: str) -> str:
    return (
        f"Do not cite {ev} content audit as evidence for H18→H35, H35→W045, causality, or confirmed precursor role."
    )
