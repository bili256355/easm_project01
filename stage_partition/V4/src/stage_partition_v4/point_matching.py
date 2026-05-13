from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from .timeline import day_index_to_month_day


def build_neighbor_group_map(pair_df: pd.DataFrame, point_ids: Iterable[str]) -> dict[str, str | None]:
    point_ids = [str(pid) for pid in point_ids]
    if pair_df is None or pair_df.empty:
        return {pid: None for pid in point_ids}
    out = {pid: None for pid in point_ids}
    for _, row in pair_df.iterrows():
        out[str(row['point_a_id'])] = row.get('neighbor_group_id')
        out[str(row['point_b_id'])] = row.get('neighbor_group_id')
    return out


def classify_match_type(day_offset, nearest_peak_day, *, strict_radius_days: int, match_radius_days: int, near_radius_days: int) -> str:
    if nearest_peak_day is None or not np.isfinite(nearest_peak_day) or day_offset is None or not np.isfinite(day_offset):
        return 'no_match'
    off = abs(int(day_offset))
    if off <= int(strict_radius_days):
        return 'strict_match'
    if off <= int(match_radius_days):
        return 'matched_point'
    if off <= int(near_radius_days):
        return 'near_support_only'
    return 'no_match'


def _build_candidate_frames(primary_points_df: pd.DataFrame, local_peaks_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['candidate_id', 'candidate_day', 'candidate_score', 'candidate_kind']
    frames: list[pd.DataFrame] = []
    if primary_points_df is not None and not primary_points_df.empty:
        p = primary_points_df[['point_id', 'point_day', 'peak_score']].copy()
        p['candidate_id'] = p['point_id'].astype(str)
        p['candidate_day'] = pd.to_numeric(p['point_day'], errors='coerce').astype('Int64')
        p['candidate_score'] = pd.to_numeric(p['peak_score'], errors='coerce')
        p['candidate_kind'] = 'formal_primary'
        frames.append(p[cols])
    if local_peaks_df is not None and not local_peaks_df.empty:
        l = local_peaks_df[['peak_id', 'peak_day', 'peak_score']].copy()
        l['candidate_id'] = l['peak_id'].astype(str)
        l['candidate_day'] = pd.to_numeric(l['peak_day'], errors='coerce').astype('Int64')
        l['candidate_score'] = pd.to_numeric(l['peak_score'], errors='coerce')
        l['candidate_kind'] = 'local_peak'
        frames.append(l[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def _prepare_candidate_universe(primary_points_df: pd.DataFrame, local_peaks_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _build_candidate_frames(primary_points_df, local_peaks_df)
    debug_cols = [
        'candidate_id', 'candidate_day', 'candidate_score', 'candidate_kind',
        'physical_peak_key', 'merged_candidate_id', 'merged_candidate_kind',
        'is_alias_of_formal_primary', 'alias_candidate_ids', 'alias_candidate_kinds'
    ]
    universe_cols = [
        'candidate_id', 'candidate_day', 'candidate_score', 'candidate_kind',
        'physical_peak_key', 'alias_candidate_ids', 'alias_candidate_kinds',
        'has_formal_primary_alias', 'has_local_peak_alias'
    ]
    if raw.empty:
        return pd.DataFrame(columns=universe_cols), pd.DataFrame(columns=debug_cols)

    raw = raw.dropna(subset=['candidate_day']).copy()
    raw['candidate_day'] = raw['candidate_day'].astype(int)
    raw['kind_priority'] = raw['candidate_kind'].map({'formal_primary': 0, 'local_peak': 1}).fillna(9)
    raw['physical_peak_key'] = raw['candidate_day'].map(lambda d: f'D{int(d):03d}')

    universe_rows = []
    debug_rows = []
    for day, grp in raw.groupby('candidate_day', sort=True):
        grp = grp.sort_values(['kind_priority', 'candidate_score', 'candidate_id'], ascending=[True, False, True]).reset_index(drop=True)
        best = grp.iloc[0]
        alias_ids = '|'.join(grp['candidate_id'].astype(str).tolist())
        alias_kinds = '|'.join(grp['candidate_kind'].astype(str).tolist())
        has_formal = bool((grp['candidate_kind'] == 'formal_primary').any())
        has_local = bool((grp['candidate_kind'] == 'local_peak').any())
        universe_rows.append({
            'candidate_id': str(best['candidate_id']),
            'candidate_day': int(day),
            'candidate_score': float(best['candidate_score']) if pd.notna(best['candidate_score']) else np.nan,
            'candidate_kind': str(best['candidate_kind']),
            'physical_peak_key': str(best['physical_peak_key']),
            'alias_candidate_ids': alias_ids,
            'alias_candidate_kinds': alias_kinds,
            'has_formal_primary_alias': has_formal,
            'has_local_peak_alias': has_local,
        })
        for _, row in grp.iterrows():
            debug_rows.append({
                'candidate_id': str(row['candidate_id']),
                'candidate_day': int(day),
                'candidate_score': float(row['candidate_score']) if pd.notna(row['candidate_score']) else np.nan,
                'candidate_kind': str(row['candidate_kind']),
                'physical_peak_key': str(best['physical_peak_key']),
                'merged_candidate_id': str(best['candidate_id']),
                'merged_candidate_kind': str(best['candidate_kind']),
                'is_alias_of_formal_primary': bool(has_formal and str(row['candidate_kind']) == 'local_peak'),
                'alias_candidate_ids': alias_ids,
                'alias_candidate_kinds': alias_kinds,
            })
    universe_df = pd.DataFrame(universe_rows, columns=universe_cols).sort_values(['candidate_day', 'candidate_id']).reset_index(drop=True)
    debug_df = pd.DataFrame(debug_rows, columns=debug_cols).sort_values(['candidate_day', 'candidate_kind', 'candidate_id']).reset_index(drop=True)
    return universe_df, debug_df


def _prepare_candidate_universe_from_df(candidate_universe_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_cols = [
        'candidate_id', 'candidate_day', 'candidate_score', 'candidate_kind',
        'physical_peak_key', 'alias_candidate_ids', 'alias_candidate_kinds',
        'has_formal_primary_alias', 'has_local_peak_alias'
    ]
    debug_cols = [
        'candidate_id', 'candidate_day', 'candidate_score', 'candidate_kind',
        'physical_peak_key', 'merged_candidate_id', 'merged_candidate_kind',
        'is_alias_of_formal_primary', 'alias_candidate_ids', 'alias_candidate_kinds'
    ]
    if candidate_universe_df is None or candidate_universe_df.empty:
        return pd.DataFrame(columns=universe_cols), pd.DataFrame(columns=debug_cols)
    df = candidate_universe_df.copy()
    rename_map = {}
    if 'point_id' in df.columns and 'candidate_id' not in df.columns:
        rename_map['point_id'] = 'candidate_id'
    if 'point_day' in df.columns and 'candidate_day' not in df.columns:
        rename_map['point_day'] = 'candidate_day'
    if 'peak_score' in df.columns and 'candidate_score' not in df.columns:
        rename_map['peak_score'] = 'candidate_score'
    if 'source_type' in df.columns and 'candidate_kind' not in df.columns:
        rename_map['source_type'] = 'candidate_kind'
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in ['candidate_id', 'candidate_day', 'candidate_score', 'candidate_kind']:
        if col not in df.columns:
            df[col] = np.nan if col != 'candidate_id' and col != 'candidate_kind' else None
    df = df.dropna(subset=['candidate_day']).copy()
    if df.empty:
        return pd.DataFrame(columns=universe_cols), pd.DataFrame(columns=debug_cols)
    df['candidate_day'] = pd.to_numeric(df['candidate_day'], errors='coerce').astype(int)
    df['candidate_score'] = pd.to_numeric(df['candidate_score'], errors='coerce')
    if 'candidate_kind' not in df.columns:
        df['candidate_kind'] = 'candidate'
    df['candidate_kind'] = df['candidate_kind'].fillna('candidate').astype(str)
    if 'physical_peak_key' not in df.columns:
        df['physical_peak_key'] = df['candidate_day'].map(lambda d: f'D{int(d):03d}')
    if 'alias_candidate_ids' not in df.columns:
        df['alias_candidate_ids'] = df['candidate_id'].astype(str)
    if 'alias_candidate_kinds' not in df.columns:
        df['alias_candidate_kinds'] = df['candidate_kind'].astype(str)
    if 'has_formal_primary_alias' not in df.columns:
        df['has_formal_primary_alias'] = df['candidate_kind'].eq('formal_primary')
    if 'has_local_peak_alias' not in df.columns:
        df['has_local_peak_alias'] = df['candidate_kind'].str.contains('local_peak', na=False)
    debug_df = pd.DataFrame([{
        'candidate_id': str(r['candidate_id']),
        'candidate_day': int(r['candidate_day']),
        'candidate_score': float(r['candidate_score']) if pd.notna(r['candidate_score']) else np.nan,
        'candidate_kind': str(r['candidate_kind']),
        'physical_peak_key': str(r['physical_peak_key']),
        'merged_candidate_id': str(r['candidate_id']),
        'merged_candidate_kind': str(r['candidate_kind']),
        'is_alias_of_formal_primary': bool(r.get('has_formal_primary_alias', False) and str(r['candidate_kind']) != 'formal_primary'),
        'alias_candidate_ids': str(r.get('alias_candidate_ids') or r['candidate_id']),
        'alias_candidate_kinds': str(r.get('alias_candidate_kinds') or r['candidate_kind']),
    } for _, r in df.iterrows()], columns=debug_cols)
    out = df[universe_cols].copy().sort_values(['candidate_day', 'candidate_id']).reset_index(drop=True)
    return out, debug_df


def build_candidate_universe_debug(primary_points_df: pd.DataFrame = None, local_peaks_df: pd.DataFrame = None, candidate_universe_df: pd.DataFrame = None) -> pd.DataFrame:
    if candidate_universe_df is not None:
        _, debug_df = _prepare_candidate_universe_from_df(candidate_universe_df)
    else:
        _, debug_df = _prepare_candidate_universe(primary_points_df, local_peaks_df)
    return debug_df


def match_reference_points(reference_points_df: pd.DataFrame, primary_points_df: pd.DataFrame = None, local_peaks_df: pd.DataFrame = None, *, candidate_universe_df: pd.DataFrame | None = None, replicate_id: int, replicate_kind: str, match_cfg, pair_df: pd.DataFrame | None = None) -> pd.DataFrame:
    cols = ['replicate_id', 'replicate_kind', 'point_id', 'point_day', 'month_day', 'point_role_group', 'point_role_detail', 'is_headline_primary', 'neighbor_group_id', 'matched_candidate_id', 'matched_candidate_kind', 'nearest_peak_day', 'nearest_peak_score', 'day_offset', 'match_type', 'matched_flag', 'strict_match_flag', 'near_support_flag', 'no_match_flag', 'ambiguous_match', 'match_confidence_class']
    if reference_points_df is None or reference_points_df.empty:
        return pd.DataFrame(columns=cols)
    if candidate_universe_df is not None:
        universe, _ = _prepare_candidate_universe_from_df(candidate_universe_df)
    else:
        universe, _ = _prepare_candidate_universe(primary_points_df, local_peaks_df)
    group_map = build_neighbor_group_map(pair_df, reference_points_df['point_id'].astype(str).tolist()) if pair_df is not None else {pid: None for pid in reference_points_df['point_id'].astype(str).tolist()}
    rows = []
    for _, ref in reference_points_df.iterrows():
        ref_day = int(ref['point_day'])
        sub = universe[(universe['candidate_day'] - ref_day).abs() <= int(match_cfg.near_radius_days)].copy()
        if sub.empty:
            rows.append({'replicate_id': replicate_id, 'replicate_kind': replicate_kind, 'point_id': str(ref['point_id']), 'point_day': ref_day, 'month_day': ref.get('month_day') or day_index_to_month_day(ref_day), 'point_role_group': ref.get('point_role_group'), 'point_role_detail': ref.get('point_role_detail'), 'is_headline_primary': bool(ref.get('is_headline_primary')), 'neighbor_group_id': group_map.get(str(ref['point_id'])), 'matched_candidate_id': None, 'matched_candidate_kind': None, 'nearest_peak_day': np.nan, 'nearest_peak_score': np.nan, 'day_offset': np.nan, 'match_type': 'no_match', 'matched_flag': False, 'strict_match_flag': False, 'near_support_flag': False, 'no_match_flag': True, 'ambiguous_match': False, 'match_confidence_class': 'no_match'})
            continue
        sub['abs_offset'] = (sub['candidate_day'] - ref_day).abs()
        sub = sub.sort_values(['abs_offset', 'candidate_score', 'candidate_day'], ascending=[True, False, True]).reset_index(drop=True)
        best = sub.iloc[0]
        offset = int(best['candidate_day']) - ref_day
        match_type = classify_match_type(offset, best['candidate_day'], strict_radius_days=match_cfg.strict_match_radius_days, match_radius_days=match_cfg.match_radius_days, near_radius_days=match_cfg.near_radius_days)
        ambiguous = False
        eligible = sub[sub['abs_offset'] <= int(match_cfg.match_radius_days)].copy()
        if len(eligible) > 1:
            score1 = float(eligible.iloc[0]['candidate_score']) if pd.notna(eligible.iloc[0]['candidate_score']) else np.nan
            score2 = float(eligible.iloc[1]['candidate_score']) if pd.notna(eligible.iloc[1]['candidate_score']) else np.nan
            day1, day2 = int(eligible.iloc[0]['candidate_day']), int(eligible.iloc[1]['candidate_day'])
            key1 = str(eligible.iloc[0].get('physical_peak_key'))
            key2 = str(eligible.iloc[1].get('physical_peak_key'))
            same_alias_peak = key1 == key2
            ambiguous = (not same_alias_peak) and (((np.isfinite(score1) and np.isfinite(score2) and abs(score1 - score2) <= float(match_cfg.ambiguous_score_tie_tol))) or abs(day1 - day2) <= int(match_cfg.ambiguous_day_tie_tol))
        conf = 'high_confidence' if match_type == 'strict_match' and not ambiguous else ('moderate_confidence' if match_type == 'matched_point' and not ambiguous else ('ambiguous' if ambiguous else ('near_only' if match_type == 'near_support_only' else 'no_match')))
        rows.append({'replicate_id': replicate_id, 'replicate_kind': replicate_kind, 'point_id': str(ref['point_id']), 'point_day': ref_day, 'month_day': ref.get('month_day') or day_index_to_month_day(ref_day), 'point_role_group': ref.get('point_role_group'), 'point_role_detail': ref.get('point_role_detail'), 'is_headline_primary': bool(ref.get('is_headline_primary')), 'neighbor_group_id': group_map.get(str(ref['point_id'])), 'matched_candidate_id': str(best['candidate_id']) if match_type != 'no_match' else None, 'matched_candidate_kind': str(best['candidate_kind']) if match_type != 'no_match' else None, 'nearest_peak_day': int(best['candidate_day']) if match_type != 'no_match' else np.nan, 'nearest_peak_score': float(best['candidate_score']) if match_type != 'no_match' and pd.notna(best['candidate_score']) else np.nan, 'day_offset': int(offset) if match_type != 'no_match' else np.nan, 'match_type': match_type, 'matched_flag': match_type in ('strict_match', 'matched_point'), 'strict_match_flag': match_type == 'strict_match', 'near_support_flag': match_type == 'near_support_only', 'no_match_flag': match_type == 'no_match', 'ambiguous_match': bool(ambiguous), 'match_confidence_class': conf})
    return annotate_pair_ambiguity(pd.DataFrame(rows, columns=cols), pair_df)


def annotate_pair_ambiguity(records_df: pd.DataFrame, pair_df: pd.DataFrame | None, *, matched_types=('strict_match', 'matched_point')) -> pd.DataFrame:
    if records_df is None or records_df.empty:
        out = records_df.copy() if records_df is not None else pd.DataFrame()
        out['ambiguous_match'] = out.get('ambiguous_match', pd.Series(dtype=bool))
        return out
    out = records_df.copy()
    if pair_df is None or pair_df.empty:
        return out
    for _, pair in pair_df.iterrows():
        a, b = str(pair['point_a_id']), str(pair['point_b_id'])
        a_df = out[out['point_id'] == a][['replicate_id', 'replicate_kind', 'nearest_peak_day', 'nearest_peak_score', 'match_type']].copy().rename(columns={'nearest_peak_day': 'a_day', 'nearest_peak_score': 'a_score', 'match_type': 'a_type'})
        b_df = out[out['point_id'] == b][['replicate_id', 'replicate_kind', 'nearest_peak_day', 'nearest_peak_score', 'match_type']].copy().rename(columns={'nearest_peak_day': 'b_day', 'nearest_peak_score': 'b_score', 'match_type': 'b_type'})
        merged = a_df.merge(b_df, on=['replicate_id', 'replicate_kind'], how='inner')
        if merged.empty:
            continue
        cond = merged['a_type'].isin(matched_types) & merged['b_type'].isin(matched_types)
        same_peak = pd.to_numeric(merged['a_day'], errors='coerce').sub(pd.to_numeric(merged['b_day'], errors='coerce')).abs() <= 0
        score_close = pd.to_numeric(merged['a_score'], errors='coerce').sub(pd.to_numeric(merged['b_score'], errors='coerce')).abs() <= 1e-8
        amb = merged.loc[cond & (same_peak | score_close), ['replicate_id', 'replicate_kind']]
        if amb.empty:
            continue
        key = set((int(r.replicate_id), str(r.replicate_kind)) for r in amb.itertuples())
        mask = [(int(rid), str(kind)) in key and str(pid) in {a, b} for rid, kind, pid in zip(out['replicate_id'], out['replicate_kind'], out['point_id'])]
        out.loc[mask, 'ambiguous_match'] = True
        out.loc[mask, 'match_confidence_class'] = 'ambiguous'
    return out


def summarize_match_records(reference_points_df: pd.DataFrame, match_records_df: pd.DataFrame, pair_df: pd.DataFrame | None = None):
    summary_cols = ['point_id', 'bootstrap_match_rate', 'bootstrap_strict_match_rate', 'bootstrap_near_support_rate', 'bootstrap_no_match_rate', 'offset_mean', 'offset_sd', 'offset_q05', 'offset_q50', 'offset_q95', 'offset_iqr', 'ambiguous_match_rate', 'dominant_over_neighbor_rate', 'neighbor_tie_rate', 'n_replicates']
    offset_cols = ['point_id', 'replicate_id', 'replicate_kind', 'point_day', 'nearest_peak_day', 'day_offset']
    if reference_points_df is None or reference_points_df.empty or match_records_df is None or match_records_df.empty:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=offset_cols)
    matched_offsets = match_records_df.loc[match_records_df['matched_flag'] == True, offset_cols].copy()
    rows = []
    for pid, sub in match_records_df.groupby('point_id'):
        n = max(len(sub), 1)
        offsets = pd.to_numeric(sub.loc[sub['matched_flag'] == True, 'day_offset'], errors='coerce').dropna().to_numpy(dtype=float)
        if offsets.size:
            q05, q50, q95 = np.nanquantile(offsets, [0.05, 0.5, 0.95])
            q25, q75 = np.nanquantile(offsets, [0.25, 0.75])
            iqr = float(q75 - q25)
        else:
            q05 = q50 = q95 = iqr = np.nan
        rows.append({'point_id': pid, 'bootstrap_match_rate': float(sub['matched_flag'].mean()), 'bootstrap_strict_match_rate': float(sub['strict_match_flag'].mean()), 'bootstrap_near_support_rate': float(sub['near_support_flag'].mean()), 'bootstrap_no_match_rate': float(sub['no_match_flag'].mean()), 'offset_mean': float(np.nanmean(offsets)) if offsets.size else np.nan, 'offset_sd': float(np.nanstd(offsets)) if offsets.size else np.nan, 'offset_q05': float(q05) if np.isfinite(q05) else np.nan, 'offset_q50': float(q50) if np.isfinite(q50) else np.nan, 'offset_q95': float(q95) if np.isfinite(q95) else np.nan, 'offset_iqr': iqr, 'ambiguous_match_rate': float(sub['ambiguous_match'].fillna(False).mean()), 'dominant_over_neighbor_rate': np.nan, 'neighbor_tie_rate': np.nan, 'n_replicates': int(n)})
    summary_df = pd.DataFrame(rows, columns=summary_cols)
    if pair_df is not None and not pair_df.empty:
        dom = {pid: [] for pid in reference_points_df['point_id'].astype(str).tolist()}
        tie = {pid: [] for pid in reference_points_df['point_id'].astype(str).tolist()}
        score_df = match_records_df[['replicate_id', 'replicate_kind', 'point_id', 'nearest_peak_score', 'matched_flag', 'ambiguous_match']].copy()
        for _, pair in pair_df.iterrows():
            a, b = str(pair['point_a_id']), str(pair['point_b_id'])
            a_df = score_df[score_df['point_id'] == a].copy().rename(columns={'nearest_peak_score': 'a_score', 'matched_flag': 'a_matched', 'ambiguous_match': 'a_amb'})
            b_df = score_df[score_df['point_id'] == b].copy().rename(columns={'nearest_peak_score': 'b_score', 'matched_flag': 'b_matched', 'ambiguous_match': 'b_amb'})
            merged = a_df.merge(b_df, on=['replicate_id', 'replicate_kind'], how='inner')
            for _, row in merged.iterrows():
                if bool(row['a_amb']) or bool(row['b_amb']):
                    tie[a].append(1.0); tie[b].append(1.0); dom[a].append(0.0); dom[b].append(0.0); continue
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
    return summary_df.sort_values('point_id').reset_index(drop=True), matched_offsets.sort_values(['point_id', 'replicate_id']).reset_index(drop=True)


def build_neighbor_match_audit(match_records_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['neighbor_group_id', 'pair_id', 'point_a_id', 'point_b_id', 'n_records', 'pair_ambiguous_rate', 'point_a_dominant_rate', 'point_b_dominant_rate', 'both_matched_rate']
    if pair_df is None or pair_df.empty or match_records_df is None or match_records_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    score_df = match_records_df[['replicate_id', 'replicate_kind', 'point_id', 'nearest_peak_score', 'matched_flag', 'ambiguous_match']].copy()
    for _, pair in pair_df.iterrows():
        a, b = str(pair['point_a_id']), str(pair['point_b_id'])
        a_df = score_df[score_df['point_id'] == a].copy().rename(columns={'nearest_peak_score': 'a_score', 'matched_flag': 'a_matched', 'ambiguous_match': 'a_amb'})
        b_df = score_df[score_df['point_id'] == b].copy().rename(columns={'nearest_peak_score': 'b_score', 'matched_flag': 'b_matched', 'ambiguous_match': 'b_amb'})
        merged = a_df.merge(b_df, on=['replicate_id', 'replicate_kind'], how='inner')
        if merged.empty:
            continue
        amb = (merged['a_amb'].fillna(False) | merged['b_amb'].fillna(False)).astype(float)
        both = (merged['a_matched'].fillna(False) & merged['b_matched'].fillna(False)).astype(float)
        a_dom, b_dom = [], []
        for _, row in merged.iterrows():
            if bool(row['a_amb']) or bool(row['b_amb']):
                a_dom.append(0.0); b_dom.append(0.0); continue
            if bool(row['a_matched']) and bool(row['b_matched']):
                a_score = float(row['a_score']) if pd.notna(row['a_score']) else np.nan
                b_score = float(row['b_score']) if pd.notna(row['b_score']) else np.nan
                if np.isfinite(a_score) and np.isfinite(b_score) and a_score > b_score:
                    a_dom.append(1.0); b_dom.append(0.0)
                elif np.isfinite(a_score) and np.isfinite(b_score) and b_score > a_score:
                    a_dom.append(0.0); b_dom.append(1.0)
                else:
                    a_dom.append(0.0); b_dom.append(0.0)
            elif bool(row['a_matched']) and not bool(row['b_matched']):
                a_dom.append(1.0); b_dom.append(0.0)
            elif bool(row['b_matched']) and not bool(row['a_matched']):
                a_dom.append(0.0); b_dom.append(1.0)
            else:
                a_dom.append(0.0); b_dom.append(0.0)
        rows.append({'neighbor_group_id': pair.get('neighbor_group_id'), 'pair_id': pair.get('pair_id'), 'point_a_id': a, 'point_b_id': b, 'n_records': int(len(merged)), 'pair_ambiguous_rate': float(np.nanmean(amb)), 'point_a_dominant_rate': float(np.nanmean(a_dom)), 'point_b_dominant_rate': float(np.nanmean(b_dom)), 'both_matched_rate': float(np.nanmean(both))})
    return pd.DataFrame(rows, columns=cols)
