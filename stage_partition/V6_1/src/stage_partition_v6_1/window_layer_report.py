from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_summary(windows_df: pd.DataFrame, membership_df: pd.DataFrame, uncertainty_df: pd.DataFrame, settings) -> dict:
    top_windows = []
    if windows_df is not None and not windows_df.empty:
        top = windows_df.sort_values(['max_member_bootstrap_match_fraction','start_day'], ascending=[False,True]).head(10)
        for _, row in top.iterrows():
            top_windows.append({
                'window_id': str(row['window_id']),
                'start_day': int(row['start_day']),
                'end_day': int(row['end_day']),
                'main_peak_day': int(row['main_peak_day']),
                'n_member_points': int(row['n_member_points']),
                'max_member_bootstrap_match_fraction': float(row['max_member_bootstrap_match_fraction']) if pd.notna(row['max_member_bootstrap_match_fraction']) else None,
            })
    return {
        'layer_name': 'stage_partition',
        'version_name': 'V6_1',
        'run_scope': 'lightweight_derived_window_layer_from_v6_points',
        'source_v6_output_tag': settings.source_v6.source_v6_output_tag,
        'n_windows': int(len(windows_df)) if windows_df is not None else 0,
        'n_memberships': int(len(membership_df)) if membership_df is not None else 0,
        'n_uncertainty_rows': int(len(uncertainty_df)) if uncertainty_df is not None else 0,
        'competition_included': False,
        'parameter_path_included': False,
        'final_judgement_included': False,
        'yearwise_gating_included': False,
        'uncertainty_is_window_attribute_only': True,
        'top_windows': top_windows,
    }
