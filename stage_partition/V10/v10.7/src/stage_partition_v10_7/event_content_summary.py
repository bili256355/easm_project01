from __future__ import annotations

from pathlib import Path
import pandas as pd


def _safe_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df is None or df.empty:
        return "No rows available.\n"
    use = [c for c in cols if c in df.columns]
    if not use:
        return df.head(n).to_markdown(index=False)
    return df[use].head(n).to_markdown(index=False)


def write_event_content_summary(
    out_path: Path,
    meta: dict,
    profile_diff: pd.DataFrame,
    spatial_metrics: pd.DataFrame,
    spatial_metrics_object: pd.DataFrame,
    similarity: pd.DataFrame,
    similarity_object: pd.DataFrame,
    yearwise_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# V10.7_c H W045 Event-Content Audit Summary\n")
    lines.append("## 1. Method boundary\n")
    lines.append("This run audits what H changes occur around H18/H35/H45/H57. It is not an influence test, not a causal test, not a lead-lag test, and not a detector rerun.\n")
    lines.append("It should be used to decide what to test next, not to claim H18→H35 or H35→W045.\n")
    lines.append("\nHOTFIX01 note: full-domain spatial composites are retained as background/context. H-object-domain spatial metrics and similarity are added and used as the primary spatial evidence for role classification. H18/H19 scale-context aliasing is also fixed.\n")
    lines.append("## 2. Input status\n")
    istat = meta.get("input_status", {})
    for k, v in istat.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## 3. Event-content role summary\n")
    lines.append(_safe_table(role_summary, [
        "event_id", "profile_strength_class", "spatial_strength_class", "yearwise_consistency_class",
        "scale_ridge_context_from_v10_7_b", "content_role_class", "recommended_next_test_target"
    ], n=20))
    lines.append("\n\n## 4. H18/H35/H45/H57 similarity — H-object domain primary\n")
    lines.append(_safe_table(similarity_object, [
        "comparison", "spatial_domain_used", "profile_pearson_correlation", "spatial_pattern_correlation", "profile_top_feature_overlap_top5", "interpretation_hint"
    ], n=20))
    lines.append("\n\n## 5. H18/H35/H45/H57 similarity — full-domain context\n")
    lines.append(_safe_table(similarity, [
        "comparison", "spatial_domain_used", "profile_pearson_correlation", "spatial_pattern_correlation", "profile_top_feature_overlap_top5", "interpretation_hint"
    ], n=20))
    lines.append("\n\n## 6. Spatial metrics — H-object domain primary\n")
    lines.append(_safe_table(spatial_metrics_object, [
        "event_id", "domain_label", "domain_lat_min", "domain_lat_max", "domain_lon_min", "domain_lon_max",
        "field_diff_abs_mean", "field_diff_max", "field_diff_min", "dominant_positive_lat", "dominant_positive_lon", "dominant_negative_lat", "dominant_negative_lon"
    ], n=20))
    lines.append("\n\n## 7. Spatial metrics — full-domain background/context\n")
    lines.append(_safe_table(spatial_metrics, [
        "event_id", "domain_label", "domain_lat_min", "domain_lat_max", "domain_lon_min", "domain_lon_max",
        "field_diff_abs_mean", "field_diff_max", "field_diff_min", "dominant_positive_lat", "dominant_positive_lon", "dominant_negative_lat", "dominant_negative_lon"
    ], n=20))
    lines.append("\n\n## 8. Yearwise consistency\n")
    lines.append(_safe_table(yearwise_summary, [
        "event_id", "n_years", "median_pattern_corr", "fraction_positive_pattern_corr", "yearwise_consistency_class"
    ], n=20))
    lines.append("\n\n## 9. Forbidden interpretations\n")
    lines.append("- Do not claim H18 influences H35 from this run.\n")
    lines.append("- Do not claim H35 influences W045 from this run.\n")
    lines.append("- Do not call H35 a confirmed weak precursor based on this run alone.\n")
    lines.append("- Do not infer causality or a physical pathway from profile/spatial composite content alone.\n")
    lines.append("- Do not treat full-domain z500 composites as H-object evidence without checking the H-object-domain metrics.\n")
    lines.append("\n## 10. Output notes\n")
    lines.append("Profile diffs are computed from H object profile/state reconstruction. Spatial maps are computed only if H/z500 field and lat/lon are detected in smoothed_fields.npz. Yearwise results are computed only if a year dimension is detected.\n")
    out_path.write_text("".join(lines), encoding="utf-8")
