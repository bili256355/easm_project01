from __future__ import annotations

import numpy as np
import pandas as pd


def build_point_yearwise_support_summary(reference_points_df: pd.DataFrame, yearwise_point_relation_df: pd.DataFrame):
    summary_cols = ['point_id','n_years_total','n_exact_hit_years','n_near_hit_years','n_missing_years','yearwise_exact_hit_fraction','yearwise_near_hit_fraction','yearwise_missing_fraction','median_yearwise_peak_score']
    long_cols = ['year','point_id','point_day','nearest_yearwise_peak_day','nearest_yearwise_peak_score','day_offset','yearwise_support_type']
    if reference_points_df.empty:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=long_cols)
    if yearwise_point_relation_df is None or yearwise_point_relation_df.empty:
        rows = [{'point_id': str(ref['point_id']),'n_years_total': 0,'n_exact_hit_years': 0,'n_near_hit_years': 0,'n_missing_years': 0,'yearwise_exact_hit_fraction': np.nan,'yearwise_near_hit_fraction': np.nan,'yearwise_missing_fraction': np.nan,'median_yearwise_peak_score': np.nan} for _, ref in reference_points_df.iterrows()]
        return pd.DataFrame(rows, columns=summary_cols), pd.DataFrame(columns=long_cols)
    long_df = yearwise_point_relation_df.copy()
    long_df['yearwise_support_type'] = np.where(long_df['strict_point_flag'].fillna(False), 'exact_hit', np.where(long_df['near_point_flag'].fillna(False), 'near_hit', 'missing'))
    long_df = long_df.rename(columns={'formal_point_day': 'point_day'})
    long_df = long_df[long_cols].sort_values(['point_id','year']).reset_index(drop=True)
    rows = []
    for _, ref in reference_points_df.iterrows():
        pid = str(ref['point_id'])
        sub = long_df[long_df['point_id']==pid].copy()
        n = len(sub); n_exact = int((sub['yearwise_support_type']=='exact_hit').sum()); n_near = int((sub['yearwise_support_type']=='near_hit').sum()); n_missing = int((sub['yearwise_support_type']=='missing').sum())
        rows.append({'point_id': pid,'n_years_total': int(n),'n_exact_hit_years': n_exact,'n_near_hit_years': n_near,'n_missing_years': n_missing,'yearwise_exact_hit_fraction': float(n_exact/n) if n else np.nan,'yearwise_near_hit_fraction': float(n_near/n) if n else np.nan,'yearwise_missing_fraction': float(n_missing/n) if n else np.nan,'median_yearwise_peak_score': float(sub['nearest_yearwise_peak_score'].dropna().median()) if not sub.empty else np.nan})
    return pd.DataFrame(rows, columns=summary_cols).sort_values('point_id').reset_index(drop=True), long_df
