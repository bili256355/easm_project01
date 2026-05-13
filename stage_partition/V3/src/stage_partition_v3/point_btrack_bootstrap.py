from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV3Settings
from .point_btrack_matching import (
    annotate_pair_ambiguity,
    assign_match_confidence_class,
    build_neighbor_group_map,
    build_neighbor_partner_map,
    classify_match_type,
    day_index_to_month_day,
)


def build_point_btrack_reference_points(point_candidates_df: pd.DataFrame, pair_df: pd.DataFrame, *, source_run_tag: str) -> pd.DataFrame:
    cols = [
        'point_id','point_day','month_day','reference_rank','point_role','point_role_group',
        'point_role_detail','is_formal_primary','is_headline_primary','related_window_id',
        'neighbor_group_id','source_run_tag'
    ]
    if point_candidates_df is None or point_candidates_df.empty:
        return pd.DataFrame(columns=cols)
    df = point_candidates_df.copy().sort_values(['point_day','point_role','point_id']).reset_index(drop=True)
    group_map = build_neighbor_group_map(pair_df, df['point_id'].astype(str).tolist())
    df['month_day'] = df['point_day'].apply(day_index_to_month_day)
    df['reference_rank'] = np.arange(1, len(df) + 1, dtype=int)
    df['point_role_detail'] = df['point_role'].astype(str)
    df['is_formal_primary'] = df['point_role_detail'].eq('formal_primary_point')
    df['is_headline_primary'] = df['is_formal_primary']
    df['point_role_group'] = np.where(df['is_formal_primary'], 'formal_primary', 'neighbor_competition')
    df['neighbor_group_id'] = df['point_id'].astype(str).map(group_map)
    df['source_run_tag'] = source_run_tag
    return df[cols]


def build_point_bootstrap_match_records(reference_points_df: pd.DataFrame, bootstrap_df: pd.DataFrame, pair_df: pd.DataFrame, settings: StagePartitionV3Settings) -> pd.DataFrame:
    cols = [
        'replicate_id','replicate_kind','point_id','point_day','month_day','point_role','point_role_group','point_role_detail',
        'is_headline_primary','related_window_id','neighbor_group_id','nearest_peak_day','nearest_peak_score','day_offset',
        'match_type','matched_flag','strict_match_flag','near_support_flag','no_match_flag','ambiguous_match',
        'matched_to_neighbor_id','ambiguous_with_neighbor_id','match_confidence_class'
    ]
    if bootstrap_df is None or bootstrap_df.empty or reference_points_df.empty:
        return pd.DataFrame(columns=cols)
    ref = reference_points_df.copy()
    df = bootstrap_df.copy().rename(columns={'rep':'replicate_id'})
    df['replicate_kind'] = 'bootstrap'
    df = df.merge(
        ref[['point_id','point_day','month_day','point_role','point_role_group','point_role_detail','is_headline_primary','related_window_id','neighbor_group_id']],
        on='point_id', how='left'
    )
    cfg = settings.btrack_point
    df['match_type'] = [
        classify_match_type(r['day_offset'], r['nearest_peak_day'], strict_radius_days=cfg.strict_match_radius_days, match_radius_days=cfg.match_radius_days, near_radius_days=cfg.near_radius_days)
        for _, r in df.iterrows()
    ]
    df['matched_flag'] = df['match_type'].isin(['strict_match','matched_point'])
    df['strict_match_flag'] = df['match_type'].eq('strict_match')
    df['near_support_flag'] = df['match_type'].eq('near_support_only')
    df['no_match_flag'] = df['match_type'].eq('no_match')
    df = annotate_pair_ambiguity(
        df,
        pair_df,
        matched_types=('strict_match','matched_point'),
        score_tie_tol=float(settings.point_significance.competition_tie_tolerance_score),
        peak_day_tie_tol=0,
    )
    partner_map = build_neighbor_partner_map(pair_df)
    df['matched_to_neighbor_id'] = df['point_id'].astype(str).map(partner_map)
    df['ambiguous_with_neighbor_id'] = np.where(df['ambiguous_match'].fillna(False), df['matched_to_neighbor_id'], None)
    df['match_confidence_class'] = [assign_match_confidence_class(mt, amb) for mt, amb in zip(df['match_type'], df['ambiguous_match'].fillna(False))]
    return df[cols].sort_values(['replicate_id','point_day','point_id']).reset_index(drop=True)


def summarize_point_bootstrap_matches(reference_points_df: pd.DataFrame, match_records_df: pd.DataFrame, pair_df: pd.DataFrame):
    summary_cols = [
        'point_id','bootstrap_match_rate','bootstrap_strict_match_rate','bootstrap_near_support_rate','bootstrap_no_match_rate',
        'offset_mean','offset_sd','offset_q05','offset_q50','offset_q95','offset_iqr','ambiguous_match_rate',
        'dominant_over_neighbor_rate','neighbor_tie_rate','n_bootstrap_reps'
    ]
    offset_cols = ['point_id','replicate_id','replicate_kind','point_day','nearest_peak_day','day_offset']
    if reference_points_df.empty or match_records_df.empty:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=offset_cols)
    matched_offsets = match_records_df.loc[match_records_df['matched_flag'] == True, offset_cols].copy()
    rows = []
    for _, ref in reference_points_df.iterrows():
        pid = str(ref['point_id'])
        sub = match_records_df[match_records_df['point_id'] == pid].copy()
        n = max(len(sub), 1)
        matched = sub[sub['matched_flag'] == True]
        offsets = matched['day_offset'].dropna().to_numpy(dtype=float)
        if offsets.size:
            q05, q50, q95 = np.nanquantile(offsets, [0.05, 0.50, 0.95])
            q75, q25 = np.nanquantile(offsets, [0.75, 0.25])
            offset_iqr = float(q75 - q25)
        else:
            q05 = q50 = q95 = offset_iqr = np.nan
        rows.append({
            'point_id': pid,
            'bootstrap_match_rate': float(sub['matched_flag'].mean()),
            'bootstrap_strict_match_rate': float(sub['strict_match_flag'].mean()),
            'bootstrap_near_support_rate': float(sub['near_support_flag'].mean()),
            'bootstrap_no_match_rate': float(sub['no_match_flag'].mean()),
            'offset_mean': float(np.nanmean(offsets)) if offsets.size else np.nan,
            'offset_sd': float(np.nanstd(offsets)) if offsets.size else np.nan,
            'offset_q05': float(q05) if np.isfinite(q05) else np.nan,
            'offset_q50': float(q50) if np.isfinite(q50) else np.nan,
            'offset_q95': float(q95) if np.isfinite(q95) else np.nan,
            'offset_iqr': offset_iqr,
            'ambiguous_match_rate': float(sub['ambiguous_match'].fillna(False).mean()),
            'dominant_over_neighbor_rate': np.nan,
            'neighbor_tie_rate': np.nan,
            'n_bootstrap_reps': int(n),
        })
    summary_df = pd.DataFrame(rows, columns=summary_cols)
    if pair_df is not None and not pair_df.empty:
        dom = {pid: [] for pid in reference_points_df['point_id'].astype(str).tolist()}
        tie = {pid: [] for pid in reference_points_df['point_id'].astype(str).tolist()}
        score_df = match_records_df[['replicate_id','replicate_kind','point_id','nearest_peak_score','matched_flag','ambiguous_match']].copy()
        for _, pair in pair_df.iterrows():
            a = str(pair['point_a_id'])
            b = str(pair['point_b_id'])
            a_df = score_df[score_df['point_id'] == a].copy().rename(columns={'nearest_peak_score':'a_score','matched_flag':'a_matched','ambiguous_match':'a_amb'})
            b_df = score_df[score_df['point_id'] == b].copy().rename(columns={'nearest_peak_score':'b_score','matched_flag':'b_matched','ambiguous_match':'b_amb'})
            merged = a_df.merge(b_df, on=['replicate_id','replicate_kind'], how='inner')
            for _, row in merged.iterrows():
                if bool(row['a_amb']) or bool(row['b_amb']):
                    tie[a].append(1.0); tie[b].append(1.0); dom[a].append(0.0); dom[b].append(0.0)
                    continue
                if bool(row['a_matched']) and bool(row['b_matched']):
                    a_score = float(row['a_score']) if pd.notna(row['a_score']) else np.nan
                    b_score = float(row['b_score']) if pd.notna(row['b_score']) else np.nan
                    if np.isfinite(a_score) and np.isfinite(b_score) and abs(a_score - b_score) <= 1e-8:
                        tie[a].append(1.0); tie[b].append(1.0); dom[a].append(0.0); dom[b].append(0.0)
                    elif np.isfinite(a_score) and np.isfinite(b_score) and a_score > b_score:
                        tie[a].append(0.0); tie[b].append(0.0); dom[a].append(1.0); dom[b].append(0.0)
                    elif np.isfinite(a_score) and np.isfinite(b_score) and b_score > a_score:
                        tie[a].append(0.0); tie[b].append(0.0); dom[a].append(0.0); dom[b].append(1.0)
                elif bool(row['a_matched']) and not bool(row['b_matched']):
                    tie[a].append(0.0); tie[b].append(0.0); dom[a].append(1.0); dom[b].append(0.0)
                elif bool(row['b_matched']) and not bool(row['a_matched']):
                    tie[a].append(0.0); tie[b].append(0.0); dom[a].append(0.0); dom[b].append(1.0)
        summary_df['dominant_over_neighbor_rate'] = summary_df['point_id'].map(lambda x: float(np.nanmean(dom.get(str(x), [np.nan]))) if dom.get(str(x)) else np.nan)
        summary_df['neighbor_tie_rate'] = summary_df['point_id'].map(lambda x: float(np.nanmean(tie.get(str(x), [np.nan]))) if tie.get(str(x)) else np.nan)
    return summary_df.sort_values('point_id').reset_index(drop=True), matched_offsets.sort_values(['point_id','replicate_id']).reset_index(drop=True)
