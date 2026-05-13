from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV3Settings


def _flag_to_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def _role_group(row: pd.Series) -> str:
    role_group = str(row.get('point_role_group') or '').strip()
    if role_group:
        return role_group
    if bool(row.get('is_formal_primary')):
        return 'formal_primary'
    return 'neighbor_competition'


def build_point_robust_support_audit(reference_points_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame, yearwise_summary_df: pd.DataFrame, parampath_summary_df: pd.DataFrame, atrack_df: pd.DataFrame, competition_df: pd.DataFrame, settings: StagePartitionV3Settings) -> pd.DataFrame:
    cols = [
        'point_id','point_day','month_day','reference_rank','point_role','point_role_group','point_role_detail',
        'is_formal_primary','is_headline_primary','related_window_id','bootstrap_match_rate','bootstrap_strict_match_rate',
        'bootstrap_near_support_rate','bootstrap_no_match_rate','offset_mean','offset_sd','offset_q05','offset_q50',
        'offset_q95','offset_iqr','ambiguous_match_rate','dominant_over_neighbor_rate','neighbor_tie_rate',
        'yearwise_exact_hit_fraction','yearwise_near_hit_fraction','yearwise_missing_fraction','median_yearwise_peak_score',
        'path_presence_rate','path_strict_match_rate','path_near_support_rate','path_ambiguous_rate','atrack_matched_stat',
        'atrack_rank','atrack_warning_flag','competition_status','competition_reason','judgement','caution_flag','caution_note'
    ]
    if reference_points_df.empty:
        return pd.DataFrame(columns=cols)
    out = reference_points_df.copy()
    for df in [bootstrap_summary_df, yearwise_summary_df, parampath_summary_df]:
        if df is not None and not df.empty:
            out = out.merge(df, on='point_id', how='left')
    if atrack_df is not None and not atrack_df.empty:
        atr = atrack_df.copy()
        atr['atrack_warning_flag'] = atr['notes'].astype(str).str.contains('matched_scale_warning', case=False, na=False)
        atr['atrack_rank'] = atr['observed_matched_stat'].rank(ascending=False, method='min')
        atr = atr.rename(columns={'observed_matched_stat':'atrack_matched_stat','neighbor_competition_status':'competition_status','notes':'competition_reason'})
        out = out.merge(atr[['point_id','atrack_matched_stat','atrack_rank','atrack_warning_flag','competition_status','competition_reason']], on='point_id', how='left')
    else:
        out['atrack_matched_stat'] = np.nan
        out['atrack_rank'] = np.nan
        out['atrack_warning_flag'] = False
        out['competition_status'] = np.nan
        out['competition_reason'] = np.nan
    if competition_df is not None and not competition_df.empty:
        status_map = {}
        reason_map = {}
        for _, row in competition_df.iterrows():
            a = str(row['point_a_id'])
            b = str(row['point_b_id'])
            outcome = str(row['competition_outcome'])
            reason = ';'.join([str(row.get('competition_reason_primary', '')), str(row.get('competition_reason_secondary', ''))]).strip(';')
            if outcome == 'tie':
                status_map[a] = 'neighbor_competition_tie'
                status_map[b] = 'neighbor_competition_tie'
            elif outcome == 'point_a_wins':
                status_map[a] = 'neighbor_competition_winner'
                status_map[b] = 'neighbor_competition_loser'
            elif outcome == 'point_b_wins':
                status_map[a] = 'neighbor_competition_loser'
                status_map[b] = 'neighbor_competition_winner'
            reason_map[a] = reason
            reason_map[b] = reason
        out['competition_status'] = out['point_id'].map(lambda x: status_map.get(str(x))).combine_first(out['competition_status'])
        out['competition_reason'] = out['point_id'].map(lambda x: reason_map.get(str(x))).combine_first(out['competition_reason'])

    cfg = settings.btrack_point_judgement
    judgements = []
    caution_flags = []
    caution_notes = []
    for _, row in out.iterrows():
        role_group = _role_group(row)
        out_role_detail = str(row.get('point_role_detail') or row.get('point_role') or '')
        is_headline = bool(row.get('is_headline_primary'))
        match_rate = row.get('bootstrap_match_rate')
        ambiguous_rate = row.get('ambiguous_match_rate')
        offset_iqr = row.get('offset_iqr')
        path_rate = row.get('path_presence_rate')
        year_exact = row.get('yearwise_exact_hit_fraction')
        tie_rate = row.get('neighbor_tie_rate')
        comp_status = str(row.get('competition_status')) if pd.notna(row.get('competition_status')) else 'no_neighbor'
        atr_warn = _flag_to_bool(row.get('atrack_warning_flag'))
        notes = []
        if not is_headline:
            notes.append('not_headline_primary')
        if pd.notna(ambiguous_rate) and float(ambiguous_rate) > float(cfg.formal_supported_ambiguous_rate_max):
            notes.append('high_neighbor_ambiguity')
        if pd.notna(offset_iqr) and float(offset_iqr) > float(cfg.formal_supported_offset_iqr_max):
            notes.append('wide_center_spread')
        if atr_warn:
            notes.append('atrack_warning')
        if comp_status == 'neighbor_competition_tie' or (pd.notna(tie_rate) and float(tie_rate) >= float(cfg.neighbor_high_tie_rate_min)):
            notes.append('neighbor_tie')
        if pd.notna(year_exact) and float(year_exact) < float(cfg.caution_yearwise_exact_min):
            notes.append('yearwise_exact_support_limited')
        if pd.notna(path_rate) and float(path_rate) < float(cfg.caution_path_presence_min):
            notes.append('path_support_only_coarse')

        if role_group != 'formal_primary':
            if comp_status == 'neighbor_competition_tie' or (pd.notna(ambiguous_rate) and float(ambiguous_rate) >= float(cfg.neighbor_ambiguous_rate_min)) or (pd.notna(tie_rate) and float(tie_rate) >= float(cfg.neighbor_high_tie_rate_min)):
                lab = 'ambiguous_neighbor_pair'
            elif pd.notna(match_rate) and float(match_rate) <= float(cfg.neighbor_weak_match_rate_max):
                lab = 'weak_neighbor_peak'
            else:
                lab = 'neighbor_candidate_with_support'
        else:
            robust_cond = (
                pd.notna(match_rate) and float(match_rate) >= float(cfg.formal_robust_match_rate_min)
                and (pd.isna(ambiguous_rate) or float(ambiguous_rate) <= float(cfg.formal_robust_ambiguous_rate_max))
                and (pd.isna(offset_iqr) or float(offset_iqr) <= float(cfg.formal_robust_offset_iqr_max))
                and not atr_warn
                and comp_status != 'neighbor_competition_loser'
                and comp_status != 'neighbor_competition_tie'
                and (pd.isna(year_exact) or float(year_exact) >= float(cfg.caution_yearwise_exact_min))
            )
            supported_cond = (
                pd.notna(match_rate) and float(match_rate) >= float(cfg.formal_supported_match_rate_min)
                and (pd.isna(ambiguous_rate) or float(ambiguous_rate) <= float(cfg.formal_supported_ambiguous_rate_max))
                and (pd.isna(offset_iqr) or float(offset_iqr) <= float(cfg.formal_supported_offset_iqr_max))
                and comp_status != 'neighbor_competition_loser'
                and comp_status != 'neighbor_competition_tie'
            )
            caution_cond = (
                pd.notna(match_rate) and float(match_rate) >= float(cfg.formal_caution_match_rate_min)
            ) or atr_warn or (pd.notna(year_exact) and float(year_exact) > 0.0)
            if comp_status == 'neighbor_competition_tie' or (pd.notna(ambiguous_rate) and float(ambiguous_rate) >= float(cfg.neighbor_ambiguous_rate_min)):
                lab = 'ambiguous_neighbor_pair'
            elif robust_cond:
                lab = 'robust_primary_point'
            elif supported_cond and not (atr_warn or (pd.notna(year_exact) and float(year_exact) < float(cfg.caution_yearwise_exact_min)) or (pd.notna(path_rate) and float(path_rate) < float(cfg.caution_path_presence_min))):
                lab = 'supported_primary_point'
            elif supported_cond or caution_cond:
                lab = 'primary_point_with_caution'
            else:
                lab = 'weak_neighbor_peak'
        judgements.append(lab)
        caution_flags.append(bool(notes))
        caution_notes.append(';'.join(dict.fromkeys(notes)))
    out['judgement'] = judgements
    out['caution_flag'] = caution_flags
    out['caution_note'] = caution_notes
    if 'point_role_detail' not in out.columns:
        out['point_role_detail'] = out['point_role']
    if 'point_role_group' not in out.columns:
        out['point_role_group'] = out.apply(_role_group, axis=1)
    if 'is_headline_primary' not in out.columns:
        out['is_headline_primary'] = out['is_formal_primary']
    return out[cols].sort_values(['point_day', 'point_id']).reset_index(drop=True)


def build_point_btrack_role_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['point_id','point_day','month_day','point_role_group','point_role_detail','is_headline_primary','judgement','caution_flag','caution_note','competition_status']
    if audit_df is None or audit_df.empty:
        return pd.DataFrame(columns=cols)
    out = audit_df.copy()
    return out[cols].sort_values(['point_day','point_id']).reset_index(drop=True)


def build_point_btrack_summary_json(audit_df: pd.DataFrame, meta: dict) -> dict:
    if audit_df is None or audit_df.empty:
        return {'n_reference_points': 0, **meta}
    formal_df = audit_df[audit_df['point_role_group'] == 'formal_primary'].copy()
    neighbor_df = audit_df[audit_df['point_role_group'] == 'neighbor_competition'].copy()
    summary = {
        'n_reference_points': int(len(audit_df)),
        'n_formal_primary_points': int((audit_df['is_formal_primary'].fillna(False)).sum()),
        'n_headline_primary_points': int((audit_df['is_headline_primary'].fillna(False)).sum()),
        'n_robust_primary_points': int((audit_df['judgement'] == 'robust_primary_point').sum()),
        'n_supported_primary_points': int((audit_df['judgement'] == 'supported_primary_point').sum()),
        'n_primary_points_with_caution': int((audit_df['judgement'] == 'primary_point_with_caution').sum()),
        'n_ambiguous_neighbor_pairs': int((audit_df['judgement'] == 'ambiguous_neighbor_pair').sum()),
        'n_neighbor_candidates_with_support': int((audit_df['judgement'] == 'neighbor_candidate_with_support').sum()),
        'n_weak_neighbor_peaks': int((audit_df['judgement'] == 'weak_neighbor_peak').sum()),
        'n_caution_points': int(audit_df['caution_flag'].fillna(False).sum()),
        'formal_primary_counts': formal_df['judgement'].value_counts(dropna=False).to_dict(),
        'neighbor_competition_counts': neighbor_df['judgement'].value_counts(dropna=False).to_dict(),
        'judgement_by_point': {
            str(row['point_id']): {
                'point_day': int(row['point_day']),
                'month_day': row.get('month_day'),
                'point_role_group': row.get('point_role_group'),
                'point_role_detail': row.get('point_role_detail'),
                'is_headline_primary': bool(row.get('is_headline_primary')),
                'judgement': row.get('judgement'),
                'bootstrap_match_rate': float(row['bootstrap_match_rate']) if pd.notna(row['bootstrap_match_rate']) else None,
                'yearwise_exact_hit_fraction': float(row['yearwise_exact_hit_fraction']) if pd.notna(row['yearwise_exact_hit_fraction']) else None,
                'path_presence_rate': float(row['path_presence_rate']) if pd.notna(row['path_presence_rate']) else None,
                'caution_note': row.get('caution_note') or None,
            }
            for _, row in audit_df.iterrows()
        },
    }
    summary.update(meta)
    return summary
