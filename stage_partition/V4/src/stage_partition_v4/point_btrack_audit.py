from __future__ import annotations
import pandas as pd
import numpy as np


def _safe_float(value):
    return float(value) if pd.notna(value) else np.nan


def _classify_core_stability(row, cfg) -> str:
    match_rate = _safe_float(row.get('bootstrap_match_rate'))
    strict_boot = _safe_float(row.get('bootstrap_strict_match_rate'))
    ambiguous_rate = _safe_float(row.get('ambiguous_match_rate'))
    offset_iqr = _safe_float(row.get('offset_iqr'))
    path_rate = _safe_float(row.get('path_presence_rate'))

    if (
        pd.notna(match_rate) and match_rate >= float(cfg.formal_robust_match_rate_min)
        and (pd.isna(strict_boot) or strict_boot >= 0.85)
        and (pd.isna(ambiguous_rate) or ambiguous_rate <= float(cfg.formal_supported_ambiguous_rate_max))
        and (pd.isna(offset_iqr) or offset_iqr <= float(cfg.formal_supported_offset_iqr_max))
        and (pd.isna(path_rate) or path_rate >= float(cfg.caution_path_presence_min))
    ):
        return 'core_strong'
    if (
        pd.notna(match_rate) and match_rate >= float(cfg.formal_supported_match_rate_min)
        and (pd.isna(ambiguous_rate) or ambiguous_rate <= max(float(cfg.formal_supported_ambiguous_rate_max), 0.30))
        and (pd.isna(offset_iqr) or offset_iqr <= max(float(cfg.formal_supported_offset_iqr_max), 8.0))
        and (pd.isna(path_rate) or path_rate >= 0.70)
    ):
        return 'core_moderate'
    if (
        pd.notna(match_rate) and match_rate >= float(cfg.formal_caution_match_rate_min)
    ):
        return 'core_weak'
    return 'core_insufficient'


def _classify_yearwise_support(row, cfg) -> tuple[str, float]:
    strict_frac = _safe_float(row.get('strict_match_fraction'))
    matched_frac = _safe_float(row.get('matched_point_fraction'))
    near_frac = _safe_float(row.get('near_support_only_fraction'))
    total = (0.0 if pd.isna(strict_frac) else strict_frac) + (0.0 if pd.isna(matched_frac) else matched_frac)

    if (
        (pd.notna(strict_frac) and strict_frac >= float(cfg.formal_robust_yearwise_strict_min))
        and total >= float(cfg.formal_supported_yearwise_total_support_min)
    ):
        return 'yearwise_strong', float(total)
    if total >= max(float(cfg.formal_caution_yearwise_total_support_min), float(cfg.formal_supported_yearwise_total_support_min) - 0.05):
        return 'yearwise_moderate', float(total)
    if pd.notna(strict_frac) and strict_frac >= max(float(cfg.formal_robust_yearwise_strict_min) - 0.025, 0.0):
        return 'yearwise_moderate', float(total)
    if total >= float(cfg.formal_caution_yearwise_total_support_min) or (pd.notna(near_frac) and near_frac > 0.0):
        return 'yearwise_weak', float(total)
    return 'yearwise_insufficient', float(total)


def _classify_pair_status(row) -> str:
    comp_status = str(row.get('competition_status') or 'no_neighbor')
    if comp_status == 'neighbor_competition_loser':
        return 'pair_loser'
    if comp_status in ('neighbor_competition_tie', 'neighbor_competition_unresolved') or bool(row.get('has_neighbor_ambiguity')):
        return 'pair_ambiguous'
    if comp_status == 'neighbor_competition_winner':
        return 'pair_clear'
    if comp_status == 'no_neighbor':
        return 'pair_clear'
    return 'pair_unresolved'


def build_point_robust_support_audit(reference_points_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame, yearwise_summary_df: pd.DataFrame, parampath_summary_df: pd.DataFrame, competition_df: pd.DataFrame, settings) -> pd.DataFrame:
    cols = [
        'point_id', 'point_day', 'month_day', 'reference_rank', 'point_role_group', 'point_role_detail', 'is_formal_primary', 'is_headline_primary', 'neighbor_group_id',
        'bootstrap_match_rate', 'bootstrap_strict_match_rate', 'bootstrap_near_support_rate', 'bootstrap_no_match_rate', 'offset_mean', 'offset_sd', 'offset_q05', 'offset_q50', 'offset_q95', 'offset_iqr', 'ambiguous_match_rate', 'dominant_over_neighbor_rate', 'neighbor_tie_rate',
        'yearwise_exact_hit_fraction', 'yearwise_near_hit_fraction', 'yearwise_missing_fraction', 'median_yearwise_peak_score',
        'strict_match_fraction', 'matched_point_fraction', 'near_support_only_fraction', 'no_match_fraction',
        'path_presence_rate', 'path_strict_match_rate', 'path_near_support_rate', 'path_ambiguous_rate',
        'competition_status', 'competition_reason', 'neighbor_pair_status', 'has_neighbor_ambiguity', 'headline_primary_with_neighbor_ambiguity',
        'core_stability_class', 'yearwise_support_class', 'pair_status_class', 'yearwise_total_support',
        'judgement', 'caution_flag', 'caution_note',
    ]
    if reference_points_df is None or reference_points_df.empty:
        return pd.DataFrame(columns=cols)
    out = reference_points_df.copy()
    for df in [bootstrap_summary_df, yearwise_summary_df, parampath_summary_df]:
        if df is not None and not df.empty:
            out = out.merge(df, on='point_id', how='left')

    status_map = {}
    reason_map = {}
    pair_status_map = {}
    has_ambiguity_map = {}
    headline_ambiguity_map = {}
    if competition_df is not None and not competition_df.empty:
        for _, row in competition_df.iterrows():
            a = str(row['point_a_id'])
            b = str(row['point_b_id'])
            outcome = str(row['competition_outcome'])
            reason = str(row.get('competition_reason') or '')
            if outcome == 'tie':
                status_map[a] = 'neighbor_competition_tie'
                status_map[b] = 'neighbor_competition_tie'
                pair_status_map[a] = 'pair_tie'
                pair_status_map[b] = 'pair_tie'
                has_ambiguity_map[a] = True
                has_ambiguity_map[b] = True
                headline_ambiguity_map[a] = True
            elif outcome == 'point_a_wins':
                status_map[a] = 'neighbor_competition_winner'
                status_map[b] = 'neighbor_competition_loser'
                pair_status_map[a] = 'formal_wins_pair'
                pair_status_map[b] = 'formal_wins_pair'
                has_ambiguity_map[a] = False
                has_ambiguity_map[b] = False
                headline_ambiguity_map[a] = False
            elif outcome == 'point_b_wins':
                status_map[a] = 'neighbor_competition_loser'
                status_map[b] = 'neighbor_competition_winner'
                pair_status_map[a] = 'neighbor_wins_pair'
                pair_status_map[b] = 'neighbor_wins_pair'
                has_ambiguity_map[a] = False
                has_ambiguity_map[b] = False
                headline_ambiguity_map[a] = False
            else:
                status_map[a] = 'neighbor_competition_unresolved'
                status_map[b] = 'neighbor_competition_unresolved'
                pair_status_map[a] = 'pair_unresolved'
                pair_status_map[b] = 'pair_unresolved'
                has_ambiguity_map[a] = True
                has_ambiguity_map[b] = True
                headline_ambiguity_map[a] = True
            reason_map[a] = reason
            reason_map[b] = reason
    out['competition_status'] = out['point_id'].map(lambda x: status_map.get(str(x), 'no_neighbor'))
    out['competition_reason'] = out['point_id'].map(lambda x: reason_map.get(str(x), ''))
    out['neighbor_pair_status'] = out['point_id'].map(lambda x: pair_status_map.get(str(x), 'no_neighbor'))
    out['has_neighbor_ambiguity'] = out['point_id'].map(lambda x: bool(has_ambiguity_map.get(str(x), False)))
    out['headline_primary_with_neighbor_ambiguity'] = out['point_id'].map(lambda x: bool(headline_ambiguity_map.get(str(x), False)))

    cfg = settings.btrack_judgement
    judgements = []
    caution_flags = []
    caution_notes = []
    core_classes = []
    yearwise_classes = []
    pair_classes = []
    total_supports = []
    for _, row in out.iterrows():
        role_group = str(row.get('point_role_group') or 'neighbor_competition')
        match_rate = _safe_float(row.get('bootstrap_match_rate'))
        ambiguous_rate = _safe_float(row.get('ambiguous_match_rate'))
        offset_iqr = _safe_float(row.get('offset_iqr'))
        path_rate = _safe_float(row.get('path_presence_rate'))
        tie_rate = _safe_float(row.get('neighbor_tie_rate'))
        comp_status = str(row.get('competition_status') or 'no_neighbor')

        core_class = _classify_core_stability(row, cfg)
        yearwise_class, total_year_support = _classify_yearwise_support(row, cfg)
        pair_class = _classify_pair_status(row)
        core_classes.append(core_class)
        yearwise_classes.append(yearwise_class)
        pair_classes.append(pair_class)
        total_supports.append(total_year_support)

        notes = []
        if not bool(row.get('is_headline_primary')):
            notes.append('not_headline_primary')
        if pair_class == 'pair_ambiguous':
            notes.append('headline_primary_with_neighbor_ambiguity')
        if pd.notna(ambiguous_rate) and ambiguous_rate > float(cfg.formal_supported_ambiguous_rate_max):
            notes.append('high_neighbor_ambiguity')
        if pd.notna(offset_iqr) and offset_iqr > float(cfg.formal_supported_offset_iqr_max):
            notes.append('wide_center_spread')
        if comp_status == 'neighbor_competition_tie' or (pd.notna(tie_rate) and tie_rate >= float(cfg.neighbor_high_tie_rate_min)):
            notes.append('neighbor_tie')
        if yearwise_class in ('yearwise_weak', 'yearwise_insufficient'):
            notes.append('yearwise_total_support_limited')
        if pd.notna(path_rate) and path_rate < float(cfg.caution_path_presence_min):
            notes.append('path_support_only_coarse')

        if role_group != 'formal_primary':
            if bool(row.get('has_neighbor_ambiguity')) or pair_class == 'pair_ambiguous' or (pd.notna(ambiguous_rate) and ambiguous_rate >= float(cfg.neighbor_ambiguous_rate_min)) or (pd.notna(tie_rate) and tie_rate >= float(cfg.neighbor_high_tie_rate_min)):
                lab = 'ambiguous_neighbor_pair'
            elif pd.notna(match_rate) and match_rate <= float(cfg.neighbor_weak_match_rate_max):
                lab = 'weak_neighbor_peak'
            else:
                lab = 'neighbor_candidate_with_support'
        else:
            if pair_class == 'pair_loser':
                lab = 'weak_neighbor_peak'
                notes.append('lost_neighbor_competition')
            elif pair_class == 'pair_ambiguous':
                lab = 'primary_point_with_caution'
            elif core_class == 'core_strong' and yearwise_class == 'yearwise_strong':
                lab = 'robust_primary_point'
            elif (
                (core_class == 'core_strong' and yearwise_class == 'yearwise_moderate')
                or (core_class == 'core_moderate' and yearwise_class == 'yearwise_strong')
                or (core_class == 'core_moderate' and yearwise_class == 'yearwise_moderate' and pair_class == 'pair_clear')
            ):
                lab = 'supported_primary_point'
            elif core_class in ('core_strong', 'core_moderate') and yearwise_class in ('yearwise_weak', 'yearwise_moderate'):
                lab = 'primary_point_with_caution'
            elif core_class == 'core_weak' and (yearwise_class != 'yearwise_insufficient' or (pd.notna(path_rate) and path_rate >= float(cfg.caution_path_presence_min))):
                lab = 'primary_point_with_caution'
            else:
                lab = 'weak_neighbor_peak'
        judgements.append(lab)
        caution_flags.append(bool(notes) or lab in ('primary_point_with_caution', 'ambiguous_neighbor_pair'))
        caution_notes.append(';'.join(dict.fromkeys(notes)))
    out['has_neighbor_ambiguity'] = out['has_neighbor_ambiguity'].fillna(False)
    out['headline_primary_with_neighbor_ambiguity'] = out['headline_primary_with_neighbor_ambiguity'].fillna(False)
    out['core_stability_class'] = core_classes
    out['yearwise_support_class'] = yearwise_classes
    out['pair_status_class'] = pair_classes
    out['yearwise_total_support'] = total_supports
    out['judgement'] = judgements
    out['caution_flag'] = caution_flags
    out['caution_note'] = caution_notes
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols].sort_values(['point_day', 'point_id']).reset_index(drop=True)


def build_point_btrack_role_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'point_id', 'point_day', 'month_day', 'point_role_group', 'point_role_detail', 'is_headline_primary', 'neighbor_group_id',
        'core_stability_class', 'yearwise_support_class', 'pair_status_class',
        'judgement', 'caution_flag', 'caution_note', 'competition_status', 'neighbor_pair_status', 'has_neighbor_ambiguity',
    ]
    if audit_df is None or audit_df.empty:
        return pd.DataFrame(columns=cols)
    return audit_df[cols].copy().sort_values(['point_day', 'point_id']).reset_index(drop=True)


def build_point_btrack_summary_json(audit_df: pd.DataFrame) -> dict:
    if audit_df is None or audit_df.empty:
        return {'n_reference_points': 0, 'btrack_backend_mode': 'independent_recompute', 'btrack_backend_independence': 'full'}
    formal_df = audit_df[audit_df['point_role_group'] == 'formal_primary'].copy()
    neighbor_df = audit_df[audit_df['point_role_group'] == 'neighbor_competition'].copy()
    n_formal_caution = int((formal_df['judgement'] == 'primary_point_with_caution').sum())
    n_neighbor_ambiguous = int((neighbor_df['judgement'] == 'ambiguous_neighbor_pair').sum())
    return {
        'n_reference_points': int(len(audit_df)),
        'n_formal_primary_points': int(len(formal_df)),
        'n_neighbor_competition_points': int(len(neighbor_df)),
        'formal_primary_counts': {
            'robust_primary_point': int((formal_df['judgement'] == 'robust_primary_point').sum()),
            'supported_primary_point': int((formal_df['judgement'] == 'supported_primary_point').sum()),
            'primary_point_with_caution': int((formal_df['judgement'] == 'primary_point_with_caution').sum()),
            'weak_neighbor_peak': int((formal_df['judgement'] == 'weak_neighbor_peak').sum()),
        },
        'neighbor_competition_counts': {
            'ambiguous_neighbor_pair': int((neighbor_df['judgement'] == 'ambiguous_neighbor_pair').sum()),
            'neighbor_candidate_with_support': int((neighbor_df['judgement'] == 'neighbor_candidate_with_support').sum()),
            'weak_neighbor_peak': int((neighbor_df['judgement'] == 'weak_neighbor_peak').sum()),
        },
        'n_formal_caution_points': n_formal_caution,
        'n_neighbor_ambiguous_points': n_neighbor_ambiguous,
        'n_caution_points': int(n_formal_caution + n_neighbor_ambiguous),
        'btrack_backend_mode': 'independent_recompute',
        'btrack_backend_independence': 'full',
        'judgement_by_point': {
            str(row['point_id']): {
                'point_day': int(row['point_day']),
                'month_day': row.get('month_day'),
                'point_role_group': row.get('point_role_group'),
                'point_role_detail': row.get('point_role_detail'),
                'is_headline_primary': bool(row.get('is_headline_primary')),
                'core_stability_class': row.get('core_stability_class'),
                'yearwise_support_class': row.get('yearwise_support_class'),
                'pair_status_class': row.get('pair_status_class'),
                'judgement': row.get('judgement'),
                'bootstrap_match_rate': float(row['bootstrap_match_rate']) if pd.notna(row['bootstrap_match_rate']) else None,
                'strict_match_fraction': float(row['strict_match_fraction']) if pd.notna(row['strict_match_fraction']) else None,
                'matched_point_fraction': float(row['matched_point_fraction']) if pd.notna(row['matched_point_fraction']) else None,
                'near_support_only_fraction': float(row['near_support_only_fraction']) if pd.notna(row['near_support_only_fraction']) else None,
                'yearwise_total_support': float(row['yearwise_total_support']) if pd.notna(row['yearwise_total_support']) else None,
                'path_presence_rate': float(row['path_presence_rate']) if pd.notna(row['path_presence_rate']) else None,
                'has_neighbor_ambiguity': bool(row.get('has_neighbor_ambiguity')),
                'neighbor_pair_status': row.get('neighbor_pair_status'),
                'caution_note': row.get('caution_note') or None,
            }
            for _, row in audit_df.iterrows()
        },
    }
