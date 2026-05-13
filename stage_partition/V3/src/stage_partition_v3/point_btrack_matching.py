from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


def day_index_to_month_day(day_index: int | float | None) -> str | None:
    if day_index is None or not np.isfinite(day_index):
        return None
    base = date(2001, 4, 1)
    dt = base + timedelta(days=int(day_index))
    return dt.strftime('%m-%d')


def build_neighbor_group_map(pair_df: pd.DataFrame, point_ids: Iterable[str]) -> dict[str, str | None]:
    point_ids = [str(pid) for pid in point_ids]
    if pair_df is None or pair_df.empty:
        return {pid: None for pid in point_ids}
    graph: dict[str, set[str]] = {pid: set() for pid in point_ids}
    for _, row in pair_df.iterrows():
        a = str(row['point_a_id'])
        b = str(row['point_b_id'])
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    visited: set[str] = set()
    out: dict[str, str | None] = {pid: None for pid in graph.keys()}
    group_idx = 1
    for pid in sorted(graph.keys()):
        if pid in visited:
            continue
        stack = [pid]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            stack.extend(sorted(graph[cur] - visited))
        if len(comp) <= 1:
            continue
        gid = f'NG{group_idx:03d}'
        for member in comp:
            out[member] = gid
        group_idx += 1
    return {str(pid): out.get(str(pid)) for pid in point_ids}


def build_neighbor_partner_map(pair_df: pd.DataFrame) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if pair_df is None or pair_df.empty:
        return out
    for _, row in pair_df.iterrows():
        a = str(row['point_a_id'])
        b = str(row['point_b_id'])
        out[a] = b
        out[b] = a
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


def assign_match_confidence_class(match_type: str, ambiguous_match: bool) -> str:
    if bool(ambiguous_match):
        return 'ambiguous'
    if match_type == 'strict_match':
        return 'high_confidence'
    if match_type == 'matched_point':
        return 'moderate_confidence'
    if match_type == 'near_support_only':
        return 'near_only'
    return 'no_match'


def annotate_pair_ambiguity(records_df: pd.DataFrame, pair_df: pd.DataFrame, *, matched_types=('strict_match','matched_point'), score_tie_tol: float = 1e-6, peak_day_tie_tol: int = 0) -> pd.DataFrame:
    if records_df.empty:
        out = records_df.copy()
        out['ambiguous_match'] = pd.Series(dtype=bool)
        return out
    out = records_df.copy()
    out['ambiguous_match'] = False
    if pair_df is None or pair_df.empty:
        return out
    wide = out[['point_id','replicate_id','replicate_kind','nearest_peak_day','nearest_peak_score','match_type']].copy()
    for _, pair in pair_df.iterrows():
        a = str(pair['point_a_id'])
        b = str(pair['point_b_id'])
        a_df = wide[wide['point_id'] == a].copy().rename(columns={'nearest_peak_day':'a_nearest_peak_day','nearest_peak_score':'a_nearest_peak_score','match_type':'a_match_type'})
        b_df = wide[wide['point_id'] == b].copy().rename(columns={'nearest_peak_day':'b_nearest_peak_day','nearest_peak_score':'b_nearest_peak_score','match_type':'b_match_type'})
        merged = a_df.merge(b_df, on=['replicate_id','replicate_kind'], how='inner')
        if merged.empty:
            continue
        cond = merged['a_match_type'].isin(matched_types) & merged['b_match_type'].isin(matched_types)
        same_peak = pd.to_numeric(merged['a_nearest_peak_day'], errors='coerce').sub(pd.to_numeric(merged['b_nearest_peak_day'], errors='coerce')).abs() <= int(peak_day_tie_tol)
        score_close = pd.to_numeric(merged['a_nearest_peak_score'], errors='coerce').sub(pd.to_numeric(merged['b_nearest_peak_score'], errors='coerce')).abs() <= float(score_tie_tol)
        amb = merged.loc[cond & (same_peak | score_close), ['replicate_id','replicate_kind']]
        if amb.empty:
            continue
        mask = out['replicate_id'].isin(amb['replicate_id']) & out['replicate_kind'].isin(amb['replicate_kind']) & out['point_id'].isin([a, b])
        out.loc[mask, 'ambiguous_match'] = True
    return out


def build_point_neighbor_match_audit(match_records_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'point_a_id','point_b_id','neighbor_group_id','n_pair_replicates',
        'pair_ambiguous_rate','point_a_dominant_rate','point_b_dominant_rate',
        'both_matched_rate','both_near_only_rate','both_no_match_rate',
        'point_a_mean_offset','point_b_mean_offset'
    ]
    if match_records_df is None or match_records_df.empty or pair_df is None or pair_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, pair in pair_df.iterrows():
        a = str(pair['point_a_id'])
        b = str(pair['point_b_id'])
        gid = pair.get('neighbor_group_id')
        a_df = match_records_df[match_records_df['point_id'] == a].copy().rename(columns={
            'match_type':'a_match_type','matched_flag':'a_matched','near_support_flag':'a_near',
            'no_match_flag':'a_no_match','ambiguous_match':'a_amb','nearest_peak_score':'a_score','day_offset':'a_offset'
        })
        b_df = match_records_df[match_records_df['point_id'] == b].copy().rename(columns={
            'match_type':'b_match_type','matched_flag':'b_matched','near_support_flag':'b_near',
            'no_match_flag':'b_no_match','ambiguous_match':'b_amb','nearest_peak_score':'b_score','day_offset':'b_offset'
        })
        merged = a_df.merge(b_df, on=['replicate_id','replicate_kind'], how='inner')
        if merged.empty:
            rows.append({
                'point_a_id': a, 'point_b_id': b, 'neighbor_group_id': gid, 'n_pair_replicates': 0,
                'pair_ambiguous_rate': np.nan, 'point_a_dominant_rate': np.nan, 'point_b_dominant_rate': np.nan,
                'both_matched_rate': np.nan, 'both_near_only_rate': np.nan, 'both_no_match_rate': np.nan,
                'point_a_mean_offset': np.nan, 'point_b_mean_offset': np.nan,
            })
            continue
        both_matched = merged['a_matched'].fillna(False) & merged['b_matched'].fillna(False)
        both_near = merged['a_near'].fillna(False) & merged['b_near'].fillna(False)
        both_none = merged['a_no_match'].fillna(False) & merged['b_no_match'].fillna(False)
        pair_amb = merged['a_amb'].fillna(False) | merged['b_amb'].fillna(False)
        a_dom = []
        b_dom = []
        for _, row in merged.iterrows():
            if bool(row.get('a_amb', False)) or bool(row.get('b_amb', False)):
                a_dom.append(0.0); b_dom.append(0.0)
                continue
            a_matched = bool(row.get('a_matched', False))
            b_matched = bool(row.get('b_matched', False))
            if a_matched and b_matched:
                a_score = float(row['a_score']) if pd.notna(row['a_score']) else np.nan
                b_score = float(row['b_score']) if pd.notna(row['b_score']) else np.nan
                if np.isfinite(a_score) and np.isfinite(b_score) and abs(a_score - b_score) <= 1e-8:
                    a_dom.append(0.0); b_dom.append(0.0)
                elif np.isfinite(a_score) and np.isfinite(b_score) and a_score > b_score:
                    a_dom.append(1.0); b_dom.append(0.0)
                elif np.isfinite(a_score) and np.isfinite(b_score) and b_score > a_score:
                    a_dom.append(0.0); b_dom.append(1.0)
                else:
                    a_dom.append(0.0); b_dom.append(0.0)
            elif a_matched and not b_matched:
                a_dom.append(1.0); b_dom.append(0.0)
            elif b_matched and not a_matched:
                a_dom.append(0.0); b_dom.append(1.0)
            else:
                a_dom.append(0.0); b_dom.append(0.0)
        rows.append({
            'point_a_id': a,
            'point_b_id': b,
            'neighbor_group_id': gid,
            'n_pair_replicates': int(len(merged)),
            'pair_ambiguous_rate': float(pair_amb.mean()),
            'point_a_dominant_rate': float(np.mean(a_dom)) if a_dom else np.nan,
            'point_b_dominant_rate': float(np.mean(b_dom)) if b_dom else np.nan,
            'both_matched_rate': float(both_matched.mean()),
            'both_near_only_rate': float(both_near.mean()),
            'both_no_match_rate': float(both_none.mean()),
            'point_a_mean_offset': float(pd.to_numeric(merged['a_offset'], errors='coerce').mean()) if not merged.empty else np.nan,
            'point_b_mean_offset': float(pd.to_numeric(merged['b_offset'], errors='coerce').mean()) if not merged.empty else np.nan,
        })
    return pd.DataFrame(rows, columns=cols).sort_values(['point_a_id','point_b_id']).reset_index(drop=True)
