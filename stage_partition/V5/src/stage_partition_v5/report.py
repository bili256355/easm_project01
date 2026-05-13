from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_summary(reference_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame, yearwise_summary_df: pd.DataFrame, settings) -> dict:
    top_bootstrap = []
    if bootstrap_summary_df is not None and not bootstrap_summary_df.empty:
        top = bootstrap_summary_df.sort_values('bootstrap_local_peak_match_fraction', ascending=False).head(5)
        for _, row in top.iterrows():
            top_bootstrap.append({
                'point_id': str(row['point_id']),
                'point_day': int(row['point_day']),
                'bootstrap_local_peak_match_fraction': float(row['bootstrap_local_peak_match_fraction']),
                'bootstrap_local_peak_strict_fraction': float(row['bootstrap_local_peak_strict_fraction']),
            })
    return {
        'layer_name': 'stage_partition',
        'version_name': 'V5',
        'run_scope': 'paper_metric_local_peak_stability_only',
        'headline_bootstrap_metric': settings.contract.headline_metric_mode,
        'year_support_metric': settings.contract.year_support_mode,
        'object_aware_support_included': bool(settings.contract.include_object_aware_support),
        'competition_included': bool(settings.contract.include_competition),
        'parameter_path_included': bool(settings.contract.include_parameter_path),
        'final_judgement_included': bool(settings.contract.include_final_judgement),
        'n_reference_points': int(len(reference_df)) if reference_df is not None else 0,
        'n_points_bootstrap_summarized': int(len(bootstrap_summary_df)) if bootstrap_summary_df is not None else 0,
        'n_points_yearwise_summarized': int(len(yearwise_summary_df)) if yearwise_summary_df is not None else 0,
        'top_bootstrap_points': top_bootstrap,
    }
