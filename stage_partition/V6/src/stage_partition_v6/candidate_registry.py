from __future__ import annotations
import numpy as np
import pandas as pd
from .timeline import day_index_to_month_day


def build_candidate_registry(local_peaks_df: pd.DataFrame, primary_points_df: pd.DataFrame | None = None, *, source_run_tag: str) -> pd.DataFrame:
    cols = [
        'candidate_id','point_day','month_day','registry_rank','source_run_tag',
        'peak_score','peak_prominence','is_formal_primary','nearest_primary_day'
    ]
    if local_peaks_df is None or local_peaks_df.empty:
        return pd.DataFrame(columns=cols)
    peaks = local_peaks_df.copy()
    peaks['peak_day'] = pd.to_numeric(peaks['peak_day'], errors='coerce')
    peaks['peak_score'] = pd.to_numeric(peaks.get('peak_score'), errors='coerce')
    peaks['peak_prominence'] = pd.to_numeric(peaks.get('peak_prominence'), errors='coerce')
    peaks = peaks.dropna(subset=['peak_day']).sort_values(['peak_day','peak_score'], ascending=[True, False]).reset_index(drop=True)
    primary_days = set()
    if primary_points_df is not None and not primary_points_df.empty:
        primary_days = set(pd.to_numeric(primary_points_df['point_day'], errors='coerce').dropna().astype(int).tolist())
        primary_list = sorted(primary_days)
    else:
        primary_list = []
    rows = []
    for i, row in peaks.iterrows():
        day = int(row['peak_day'])
        nearest_primary_day = np.nan
        if primary_list:
            nearest_primary_day = int(min(primary_list, key=lambda x: abs(x - day)))
        rows.append({
            'candidate_id': f'CP{i+1:03d}',
            'point_day': day,
            'month_day': day_index_to_month_day(day),
            'registry_rank': int(i + 1),
            'source_run_tag': source_run_tag,
            'peak_score': float(row['peak_score']) if pd.notna(row['peak_score']) else np.nan,
            'peak_prominence': float(row['peak_prominence']) if pd.notna(row['peak_prominence']) else np.nan,
            'is_formal_primary': bool(day in primary_days),
            'nearest_primary_day': nearest_primary_day,
        })
    return pd.DataFrame(rows, columns=cols)
