from __future__ import annotations
import numpy as np
import pandas as pd


def build_reference_points(primary_points_df: pd.DataFrame, *, source_run_tag: str) -> pd.DataFrame:
    cols = [
        'point_id','point_day','month_day','reference_rank','source_run_tag',
        'raw_point_day','matched_peak_day','peak_score','peak_prominence'
    ]
    if primary_points_df is None or primary_points_df.empty:
        return pd.DataFrame(columns=cols)
    df = primary_points_df.copy().sort_values(['point_day', 'point_id']).reset_index(drop=True)
    df['reference_rank'] = np.arange(1, len(df) + 1, dtype=int)
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
