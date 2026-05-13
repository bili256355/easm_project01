from __future__ import annotations
import numpy as np
import pandas as pd


def build_formal_reference_points(primary_points_df: pd.DataFrame, *, source_run_tag: str) -> pd.DataFrame:
    cols = [
        'point_id','point_day','month_day','reference_rank','point_role_group','point_role_detail',
        'is_formal_primary','is_headline_primary','neighbor_group_id','source_run_tag',
        'raw_point_day','matched_peak_day','peak_score','peak_prominence'
    ]
    if primary_points_df is None or primary_points_df.empty:
        return pd.DataFrame(columns=cols)
    df = primary_points_df.copy().sort_values(['point_day','point_id']).reset_index(drop=True)
    df['reference_rank'] = np.arange(1, len(df) + 1, dtype=int)
    df['point_role_group'] = 'formal_primary'
    df['point_role_detail'] = 'headline_primary'
    df['is_formal_primary'] = True
    df['is_headline_primary'] = True
    df['neighbor_group_id'] = None
    df['source_run_tag'] = source_run_tag
    if 'raw_point_day' not in df.columns:
        df['raw_point_day'] = df['point_day']
    if 'matched_peak_day' not in df.columns:
        df['matched_peak_day'] = df['point_day']

    if 'peak_score' not in df.columns:
        if 'peak_value' in df.columns:
            df['peak_score'] = pd.to_numeric(df['peak_value'], errors='coerce')
        else:
            df['peak_score'] = np.nan
    else:
        df['peak_score'] = pd.to_numeric(df['peak_score'], errors='coerce')

    if 'peak_prominence' not in df.columns:
        df['peak_prominence'] = np.nan
    else:
        df['peak_prominence'] = pd.to_numeric(df['peak_prominence'], errors='coerce')

    return df[cols]
