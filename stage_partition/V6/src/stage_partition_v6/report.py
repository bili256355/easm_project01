from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_summary(registry_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame, yearwise_summary_df: pd.DataFrame, settings) -> dict:
    top_bootstrap = []
    if bootstrap_summary_df is not None and not bootstrap_summary_df.empty:
        top = bootstrap_summary_df.sort_values('bootstrap_match_fraction', ascending=False).head(10)
        registry_idx = registry_df.set_index('candidate_id') if registry_df is not None and not registry_df.empty else None
        for _, row in top.iterrows():
            is_primary = bool(registry_idx.loc[row['candidate_id'], 'is_formal_primary']) if registry_idx is not None and row['candidate_id'] in registry_idx.index else False
            top_bootstrap.append({
                'candidate_id': str(row['candidate_id']),
                'point_day': int(row['point_day']),
                'is_formal_primary': is_primary,
                'bootstrap_match_fraction': float(row['bootstrap_match_fraction']),
                'bootstrap_strict_fraction': float(row['bootstrap_strict_fraction']),
            })
    return {
        'layer_name': 'stage_partition',
        'version_name': 'V6',
        'run_scope': 'baseline_detected_peaks_bootstrap_screening_only',
        'candidate_scope': settings.contract.candidate_scope,
        'headline_metric': settings.contract.headline_metric_mode,
        'bootstrap_n_resamples': int(settings.bootstrap.n_bootstrap),
        'bootstrap_strict_window_days': int(settings.bootstrap.strict_match_max_abs_offset_days),
        'bootstrap_match_window_days': int(settings.bootstrap.match_max_abs_offset_days),
        'bootstrap_near_window_days': int(settings.bootstrap.near_max_abs_offset_days),
        'year_support_mode': settings.contract.year_support_mode,
        'year_support_is_auxiliary': True,
        'yearwise_match_window_days': int(settings.yearwise.match_max_abs_offset_days),
        'window_judgement_included': bool(settings.contract.include_window_judgement),
        'competition_included': bool(settings.contract.include_competition),
        'parameter_path_included': bool(settings.contract.include_parameter_path),
        'final_judgement_included': bool(settings.contract.include_final_judgement),
        'n_candidates': int(len(registry_df)) if registry_df is not None else 0,
        'n_candidates_bootstrap_summarized': int(len(bootstrap_summary_df)) if bootstrap_summary_df is not None else 0,
        'n_candidates_yearwise_summarized': int(len(yearwise_summary_df)) if yearwise_summary_df is not None else 0,
        'top_bootstrap_candidates': top_bootstrap,
    }
