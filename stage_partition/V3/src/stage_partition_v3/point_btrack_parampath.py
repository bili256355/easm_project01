from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV3Settings
from .point_btrack_matching import annotate_pair_ambiguity, classify_match_type


def build_point_parameter_path_support(reference_points_df: pd.DataFrame, point_param_df: pd.DataFrame, pair_df: pd.DataFrame, settings: StagePartitionV3Settings) -> pd.DataFrame:
    cols = ['point_id','path_presence_rate','path_strict_match_rate','path_near_support_rate','path_ambiguous_rate','path_offset_min','path_offset_max','n_path_configs']
    if reference_points_df.empty or point_param_df is None or point_param_df.empty:
        return pd.DataFrame(columns=cols)
    df = point_param_df.copy().rename(columns={'width':'path_width','pen':'path_pen'})
    df['replicate_kind'] = 'param_path'
    combos = df[['path_width','path_pen']].drop_duplicates().reset_index(drop=True)
    combos['replicate_id'] = range(len(combos))
    df = df.merge(combos, on=['path_width','path_pen'], how='left')
    cfg = settings.btrack_point
    df['match_type'] = [classify_match_type(r['day_offset'], r['nearest_peak_day'], strict_radius_days=cfg.strict_match_radius_days, match_radius_days=cfg.match_radius_days, near_radius_days=cfg.near_radius_days) for _, r in df.iterrows()]
    df['matched_flag'] = df['match_type'].isin(['strict_match','matched_point'])
    df['strict_match_flag'] = df['match_type'].eq('strict_match')
    df['near_support_flag'] = df['match_type'].eq('near_support_only')
    df = annotate_pair_ambiguity(df, pair_df, matched_types=('strict_match','matched_point'), score_tie_tol=float(settings.point_significance.competition_tie_tolerance_score), peak_day_tie_tol=0)
    rows = []
    for _, ref in reference_points_df.iterrows():
        pid = str(ref['point_id']); sub = df[df['point_id']==pid].copy(); matched_offsets = sub.loc[sub['matched_flag']==True,'day_offset'].dropna().to_numpy(dtype=float)
        rows.append({'point_id': pid,'path_presence_rate': float(sub['matched_flag'].mean()) if not sub.empty else np.nan,'path_strict_match_rate': float(sub['strict_match_flag'].mean()) if not sub.empty else np.nan,'path_near_support_rate': float(sub['near_support_flag'].mean()) if not sub.empty else np.nan,'path_ambiguous_rate': float(sub['ambiguous_match'].fillna(False).mean()) if not sub.empty else np.nan,'path_offset_min': float(np.nanmin(matched_offsets)) if matched_offsets.size else np.nan,'path_offset_max': float(np.nanmax(matched_offsets)) if matched_offsets.size else np.nan,'n_path_configs': int(sub[['path_width','path_pen']].drop_duplicates().shape[0]) if not sub.empty else 0})
    return pd.DataFrame(rows, columns=cols).sort_values('point_id').reset_index(drop=True)
