from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import W045PreclusterConfig


def _objects_by_class(g: pd.DataFrame, cls: str) -> list[str]:
    if g.empty or "participation_class" not in g.columns:
        return []
    return g[g["participation_class"] == cls]["object"].tolist()


def write_markdown_summary(
    cfg: W045PreclusterConfig,
    metrics: pd.DataFrame,
    h_role: pd.DataFrame,
    interpretation: pd.DataFrame,
) -> Path:
    path = cfg.output_root / "summary_w045_precluster_audit_v10_6_a.md"
    lines: list[str] = []
    lines.append("# V10.6_a W045 precluster audit summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("This output audits W045 only. It decomposes day16-19, day30-35, and day41-46 into fixed candidate clusters E1/E2/M, plus H_post_reference around day57.")
    lines.append("It does not perform yearwise prediction, spatial/cartopy validation, or causal inference.")
    lines.append("")
    lines.append("## HOTFIX02 interpretation rule")
    lines.append("`candidate_inside_cluster` is marker-supported activity and is the event-semantics core.")
    lines.append("`curve_peak_without_marker` is curve-only/ramp/shoulder evidence. It must not be treated as equal to marker-supported activity.")
    lines.append("")
    lines.append("## Cluster participation snapshot")
    for cid, g in metrics.groupby("cluster_id", sort=False):
        marker_supported = _objects_by_class(g, "candidate_inside_cluster")
        curve_only = _objects_by_class(g, "curve_peak_without_marker")
        weak_curve = _objects_by_class(g, "weak_curve_signal")
        absent_or_missing = g[g["participation_class"].isin(["no_signal", "missing_input"])] ["object"].tolist() if "participation_class" in g.columns else []
        lines.append(f"- {cid}:")
        lines.append(f"  - marker_supported_core_objects = {marker_supported}")
        lines.append(f"  - curve_only_ramp_or_shoulder_objects = {curve_only}")
        lines.append(f"  - weak_curve_signal_objects = {weak_curve}")
        lines.append(f"  - absent_or_missing_objects = {absent_or_missing}")
    lines.append("")
    lines.append("## H day35 role audit")
    if h_role.empty:
        lines.append("H day35 role audit is missing because required inputs could not be read.")
    else:
        r = h_role.iloc[0]
        lines.append(f"- role_class: {r['role_class']}")
        lines.append(f"- confirmed_weak_precursor: {r['confirmed_weak_precursor']}")
        lines.append(f"- recommended_wording: {r['recommended_wording']}")
        lines.append(f"- forbidden_wording: {r['forbidden_wording']}")
    lines.append("")
    lines.append("## Interpretation claims")
    for _, r in interpretation.iterrows():
        lines.append(f"- {r['claim_id']} [{r['support_level']}]: {r['claim_text']}")
    lines.append("")
    lines.append("## Boundary")
    lines.append("V10.6_a is a method-layer / derived-structure audit. It should not be used to claim that H day35 is a confirmed weak precursor or causal trigger of W045.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
