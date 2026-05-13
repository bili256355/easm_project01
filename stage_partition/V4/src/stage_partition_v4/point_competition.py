from __future__ import annotations
import numpy as np
import pandas as pd


def build_point_neighbor_pairs(formal_reference_df: pd.DataFrame, audit_universe_df: pd.DataFrame, neighbor_radius_days: int, *, source_run_tag: str) -> dict:
    pair_cols = ['neighbor_group_id', 'pair_id', 'point_a_id', 'point_b_id', 'point_a_day', 'point_b_day', 'point_a_score', 'point_b_score', 'day_distance', 'score_diff']
    if formal_reference_df is None or formal_reference_df.empty:
        empty = pd.DataFrame(columns=pair_cols)
        return {'pairs_df': empty, 'augmented_reference_df': formal_reference_df.copy() if formal_reference_df is not None else pd.DataFrame()}
    formal = formal_reference_df.copy().sort_values(['point_day', 'point_id']).reset_index(drop=True)
    candidates = audit_universe_df.copy() if audit_universe_df is not None else pd.DataFrame()
    if candidates.empty:
        empty = pd.DataFrame(columns=pair_cols)
        return {'pairs_df': empty, 'augmented_reference_df': formal}
    if 'candidate_day' in candidates.columns and 'point_day' not in candidates.columns:
        candidates = candidates.rename(columns={'candidate_day': 'point_day', 'candidate_id': 'point_id', 'candidate_score': 'peak_score'})
    neighbor = candidates[candidates.get('role_group', '').astype(str) == 'neighbor_competition'].copy()
    if neighbor.empty:
        empty = pd.DataFrame(columns=pair_cols)
        return {'pairs_df': empty, 'augmented_reference_df': formal}
    pairs = []
    formal_aug = formal.copy()
    formal_aug['neighbor_group_id'] = formal_aug.get('neighbor_group_id')
    neighbor_rows = []
    used_neighbors = set()
    gid = 0
    for _, frow in formal.iterrows():
        sub = neighbor[(neighbor['point_day'].astype(int) - int(frow['point_day'])).abs() <= int(neighbor_radius_days)].copy()
        if sub.empty:
            continue
        sub['abs_offset'] = (sub['point_day'].astype(int) - int(frow['point_day'])).abs()
        sub = sub.sort_values(['abs_offset', 'peak_score', 'point_day'], ascending=[True, False, True]).reset_index(drop=True)
        cand = sub.iloc[0]
        if str(cand['point_id']) in used_neighbors:
            continue
        used_neighbors.add(str(cand['point_id']))
        gid += 1
        a_score = float(frow.get('peak_score', np.nan)) if pd.notna(frow.get('peak_score', np.nan)) else np.nan
        b_score = float(cand.get('peak_score', np.nan)) if pd.notna(cand.get('peak_score', np.nan)) else np.nan
        ng = f'NG{gid:03d}'
        pid = f'NP{gid:03d}'
        pairs.append(
            {
                'neighbor_group_id': ng,
                'pair_id': pid,
                'point_a_id': str(frow['point_id']),
                'point_b_id': str(cand['point_id']),
                'point_a_day': int(frow['point_day']),
                'point_b_day': int(cand['point_day']),
                'point_a_score': a_score,
                'point_b_score': b_score,
                'day_distance': int(abs(int(frow['point_day']) - int(cand['point_day']))),
                'score_diff': float(a_score - b_score) if np.isfinite(a_score) and np.isfinite(b_score) else np.nan,
            }
        )
        formal_aug.loc[formal_aug['point_id'] == frow['point_id'], 'neighbor_group_id'] = ng
        nrow = cand.copy()
        nrow['neighbor_group_id'] = ng
        nrow['source_run_tag'] = source_run_tag
        neighbor_rows.append(nrow)
    pairs_df = pd.DataFrame(pairs, columns=pair_cols)
    neighbor_df = pd.DataFrame(neighbor_rows)
    keep_cols = ['point_id', 'point_day', 'month_day', 'reference_rank', 'point_role_group', 'point_role_detail', 'is_formal_primary', 'is_headline_primary', 'neighbor_group_id', 'source_run_tag', 'raw_point_day', 'matched_peak_day', 'peak_score', 'peak_prominence']
    if neighbor_df.empty:
        augmented = formal_aug[keep_cols].copy()
    else:
        defaults = {
            'is_formal_primary': False,
            'is_headline_primary': False,
            'point_role_group': 'neighbor_competition',
            'point_role_detail': 'neighbor_local_peak',
            'raw_point_day': np.nan,
            'matched_peak_day': np.nan,
            'peak_score': np.nan,
            'peak_prominence': np.nan,
        }
        for col in keep_cols:
            if col not in neighbor_df.columns:
                neighbor_df[col] = defaults.get(col, np.nan)
        augmented = pd.concat([formal_aug[keep_cols], neighbor_df[keep_cols]], ignore_index=True, sort=False).sort_values(['point_day', 'is_headline_primary', 'point_id'], ascending=[True, False, True]).reset_index(drop=True)
    return {'pairs_df': pairs_df, 'augmented_reference_df': augmented}


def _support_relation(a_rate, b_rate, tie_rate, tie_tol):
    if pd.isna(a_rate) or pd.isna(b_rate):
        return 'missing'
    if pd.notna(tie_rate) and float(tie_rate) >= 0.5:
        return 'close'
    if abs(float(a_rate) - float(b_rate)) <= float(tie_tol):
        return 'close'
    return 'a' if float(a_rate) > float(b_rate) else 'b'


def _support_asymmetry(score_relation: str, boot_relation: str, year_relation: str) -> tuple[bool, str | None, str]:
    supports_a = []
    supports_b = []
    if score_relation == 'a':
        supports_a.append('score')
    elif score_relation == 'b':
        supports_b.append('score')
    if boot_relation == 'a':
        supports_a.append('bootstrap')
    elif boot_relation == 'b':
        supports_b.append('bootstrap')
    if year_relation == 'a':
        supports_a.append('yearwise')
    elif year_relation == 'b':
        supports_b.append('yearwise')
    if supports_a and not supports_b:
        return True, 'formal', ';'.join([f'{src}_leans_formal' for src in supports_a])
    if supports_b and not supports_a:
        return True, 'neighbor', ';'.join([f'{src}_leans_neighbor' for src in supports_b])
    return False, None, ''


def finalize_point_competition(pairs_df: pd.DataFrame, bootstrap_pair_df: pd.DataFrame | None, yearwise_pair_summary_df: pd.DataFrame | None, competition_cfg, *, source_run_tag: str) -> pd.DataFrame:
    cols = [
        'neighbor_group_id', 'pair_id', 'point_a_id', 'point_b_id', 'point_a_day', 'point_b_day', 'point_a_score', 'point_b_score', 'day_distance', 'score_diff',
        'point_a_bootstrap_dominant_rate', 'point_b_bootstrap_dominant_rate', 'pair_bootstrap_tie_rate',
        'point_a_yearwise_dominant_rate', 'point_b_yearwise_dominant_rate', 'pair_yearwise_tie_rate', 'n_years_compared',
        'competition_outcome', 'competition_reason_primary', 'competition_reason_secondary', 'competition_reason',
        'support_asymmetry_flag', 'support_asymmetry_direction', 'support_asymmetry_note', 'source_run_tag',
    ]
    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame(columns=cols)
    out = pairs_df.copy()
    if bootstrap_pair_df is not None and not bootstrap_pair_df.empty:
        tmp = bootstrap_pair_df.rename(columns={
            'pair_ambiguous_rate': 'pair_bootstrap_tie_rate',
            'point_a_dominant_rate': 'point_a_bootstrap_dominant_rate',
            'point_b_dominant_rate': 'point_b_bootstrap_dominant_rate',
        })
        keep = ['pair_id', 'point_a_bootstrap_dominant_rate', 'point_b_bootstrap_dominant_rate', 'pair_bootstrap_tie_rate']
        out = out.merge(tmp[keep], on='pair_id', how='left')
    else:
        out['point_a_bootstrap_dominant_rate'] = np.nan
        out['point_b_bootstrap_dominant_rate'] = np.nan
        out['pair_bootstrap_tie_rate'] = np.nan
    if yearwise_pair_summary_df is not None and not yearwise_pair_summary_df.empty:
        keep = ['pair_id', 'point_a_yearwise_dominant_rate', 'point_b_yearwise_dominant_rate', 'pair_yearwise_tie_rate', 'n_years_compared']
        out = out.merge(yearwise_pair_summary_df[keep], on='pair_id', how='left')
    else:
        out['point_a_yearwise_dominant_rate'] = np.nan
        out['point_b_yearwise_dominant_rate'] = np.nan
        out['pair_yearwise_tie_rate'] = np.nan
        out['n_years_compared'] = np.nan

    primaries = []
    secondaries = []
    reasons = []
    outcomes = []
    asym_flags = []
    asym_dirs = []
    asym_notes = []
    for _, row in out.iterrows():
        a_score = row.get('point_a_score')
        b_score = row.get('point_b_score')
        if pd.notna(a_score) and pd.notna(b_score):
            if abs(float(a_score) - float(b_score)) <= float(competition_cfg.score_tie_tolerance):
                score_relation = 'close'
            else:
                score_relation = 'a' if float(a_score) > float(b_score) else 'b'
        else:
            score_relation = 'missing'
        boot_relation = _support_relation(row.get('point_a_bootstrap_dominant_rate'), row.get('point_b_bootstrap_dominant_rate'), row.get('pair_bootstrap_tie_rate'), competition_cfg.bootstrap_tie_tolerance)
        year_relation = _support_relation(row.get('point_a_yearwise_dominant_rate'), row.get('point_b_yearwise_dominant_rate'), row.get('pair_yearwise_tie_rate'), competition_cfg.yearwise_tie_tolerance)
        tokens = []
        if score_relation == 'close':
            tokens.append('score_close')
        elif score_relation == 'a':
            tokens.append('formal_peak_higher_score')
        elif score_relation == 'b':
            tokens.append('neighbor_peak_higher_score')
        else:
            tokens.append('score_missing')
        if boot_relation == 'close':
            tokens.append('bootstrap_close')
        elif boot_relation == 'a':
            tokens.append('bootstrap_supports_formal')
        elif boot_relation == 'b':
            tokens.append('bootstrap_supports_neighbor')
        else:
            tokens.append('bootstrap_missing')
        if year_relation == 'close':
            tokens.append('yearwise_close')
        elif year_relation == 'a':
            tokens.append('yearwise_supports_formal')
        elif year_relation == 'b':
            tokens.append('yearwise_supports_neighbor')
        else:
            tokens.append('yearwise_missing')

        if score_relation == 'close' and boot_relation in ('close', 'missing') and year_relation in ('close', 'missing'):
            outcome = 'tie'
            primary = 'near_equal_score_and_support'
        elif score_relation == 'missing':
            outcome = 'unresolved_score_missing'
            primary = 'score_missing_requires_manual_check'
        elif score_relation == 'a':
            if not bool(competition_cfg.require_support_for_non_tie) or boot_relation == 'a' or year_relation == 'a':
                outcome = 'point_a_wins'
                primary = 'formal_peak_supported'
            elif boot_relation == 'close' and year_relation == 'close':
                outcome = 'tie'
                primary = 'support_levels_close'
            else:
                outcome = 'unresolved_support_insufficient'
                primary = 'score_advantage_without_support'
        else:
            if not bool(competition_cfg.require_support_for_non_tie) or boot_relation == 'b' or year_relation == 'b':
                outcome = 'point_b_wins'
                primary = 'neighbor_peak_supported'
            elif boot_relation == 'close' and year_relation == 'close':
                outcome = 'tie'
                primary = 'support_levels_close'
            else:
                outcome = 'unresolved_support_insufficient'
                primary = 'score_advantage_without_support'
        asym_flag, asym_dir, asym_note = _support_asymmetry(score_relation, boot_relation, year_relation)
        primaries.append(primary)
        secondaries.append(';'.join(tokens))
        reasons.append(primary + ';' + ';'.join(tokens))
        outcomes.append(outcome)
        asym_flags.append(bool(asym_flag))
        asym_dirs.append(asym_dir)
        asym_notes.append(asym_note)
    out['competition_outcome'] = outcomes
    out['competition_reason_primary'] = primaries
    out['competition_reason_secondary'] = secondaries
    out['competition_reason'] = reasons
    out['support_asymmetry_flag'] = asym_flags
    out['support_asymmetry_direction'] = asym_dirs
    out['support_asymmetry_note'] = asym_notes
    out['source_run_tag'] = source_run_tag
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols].sort_values(['point_a_day', 'pair_id']).reset_index(drop=True)
