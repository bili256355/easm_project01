from __future__ import annotations
import numpy as np
import pandas as pd
from .state_builder import build_year_state_matrix
from .detector_ruptures_window import run_point_detector
from .point_reference import build_formal_reference_points
from .point_audit_universe import build_point_audit_universe
from .point_matching import match_reference_points


def _progress_iter(items, enabled: bool, desc: str):
    if enabled:
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc)
        except Exception:
            return items
    return items


def _build_pair_summary(long_df: pd.DataFrame, pair_df: pd.DataFrame | None, settings) -> pd.DataFrame:
    cols = [
        'pair_id',
        'point_a_id',
        'point_b_id',
        'point_a_yearwise_dominant_rate',
        'point_b_yearwise_dominant_rate',
        'pair_yearwise_tie_rate',
        'n_years_compared',
        'pair_yearwise_method',
        'pair_yearwise_requires_both_detected',
        'pair_yearwise_is_conservative',
    ]
    if pair_df is None or pair_df.empty or long_df is None or long_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    base = long_df[['year_index', 'point_id', 'match_type', 'peak_score']].copy()
    matched_types = {'strict_match', 'matched_point'}
    requires_both = bool(getattr(settings.yearwise, 'pair_requires_both_detected', True))
    method = str(getattr(settings.yearwise, 'pair_competition_mode', 'conservative_comparable_years_only'))
    conservative = 'conservative' in method
    for _, pair in pair_df.iterrows():
        a = str(pair['point_a_id'])
        b = str(pair['point_b_id'])
        a_df = base[base['point_id'] == a].rename(columns={'match_type': 'a_match_type', 'peak_score': 'a_peak_score'})
        b_df = base[base['point_id'] == b].rename(columns={'match_type': 'b_match_type', 'peak_score': 'b_peak_score'})
        merged = a_df.merge(b_df, on='year_index', how='inner')
        if merged.empty:
            rows.append({
                'pair_id': str(pair['pair_id']),
                'point_a_id': a,
                'point_b_id': b,
                'point_a_yearwise_dominant_rate': np.nan,
                'point_b_yearwise_dominant_rate': np.nan,
                'pair_yearwise_tie_rate': np.nan,
                'n_years_compared': 0,
                'pair_yearwise_method': method,
                'pair_yearwise_requires_both_detected': requires_both,
                'pair_yearwise_is_conservative': conservative,
            })
            continue
        a_dom = []
        b_dom = []
        tie = []
        for _, row in merged.iterrows():
            a_ok = str(row['a_match_type']) in matched_types and pd.notna(row['a_peak_score'])
            b_ok = str(row['b_match_type']) in matched_types and pd.notna(row['b_peak_score'])
            if requires_both and not (a_ok and b_ok):
                continue
            if not a_ok and not b_ok:
                continue
            if not a_ok or not b_ok:
                # Current method contract stays conservative: skip asymmetric years.
                continue
            a_score = float(row['a_peak_score'])
            b_score = float(row['b_peak_score'])
            if abs(a_score - b_score) <= float(getattr(settings.matching, 'ambiguous_score_tie_tol', 1e-8)):
                tie.append(1.0)
                a_dom.append(0.0)
                b_dom.append(0.0)
            elif a_score > b_score:
                tie.append(0.0)
                a_dom.append(1.0)
                b_dom.append(0.0)
            else:
                tie.append(0.0)
                a_dom.append(0.0)
                b_dom.append(1.0)
        n = len(tie)
        rows.append({
            'pair_id': str(pair['pair_id']),
            'point_a_id': a,
            'point_b_id': b,
            'point_a_yearwise_dominant_rate': float(np.mean(a_dom)) if n else np.nan,
            'point_b_yearwise_dominant_rate': float(np.mean(b_dom)) if n else np.nan,
            'pair_yearwise_tie_rate': float(np.mean(tie)) if n else np.nan,
            'n_years_compared': int(n),
            'pair_yearwise_method': method,
            'pair_yearwise_requires_both_detected': requires_both,
            'pair_yearwise_is_conservative': conservative,
        })
    return pd.DataFrame(rows, columns=cols)


def run_point_yearwise_support(profiles: dict, years: np.ndarray, reference_points_df: pd.DataFrame, settings, pair_df: pd.DataFrame | None = None) -> dict:
    long_rows = []
    years_arr = np.asarray(years).astype(int)
    for yi in _progress_iter(range(len(years_arr)), settings.yearwise.progress, 'V4 yearwise'):
        year_state = build_year_state_matrix(profiles, yi, settings.state)
        if year_state['valid_day_index'].size < max(2 * int(settings.detector.width), 3):
            for _, ref in reference_points_df.iterrows():
                long_rows.append(
                    {
                        'year': int(years_arr[yi]),
                        'year_index': int(yi),
                        'point_id': str(ref['point_id']),
                        'reference_day': int(ref['point_day']),
                        'detected_day': np.nan,
                        'offset': np.nan,
                        'match_type': 'no_match',
                        'hit': 'missing',
                        'peak_score': np.nan,
                    }
                )
            continue
        det = run_point_detector(
            year_state['state_matrix'][year_state['valid_day_mask'], :],
            year_state['valid_day_index'],
            settings.detector,
            local_peak_distance_days=settings.detector.local_peak_min_distance_days,
        )
        formal_ref = build_formal_reference_points(det['primary_points_df'], source_run_tag=settings.output.output_tag)
        audit_universe_df = build_point_audit_universe(
            formal_ref,
            det['local_peaks_df'],
            det['profile'],
            radius_days=settings.point_audit_universe.radius_days,
            weak_peak_min_prominence=settings.point_audit_universe.weak_peak_min_prominence,
            source_run_tag=settings.output.output_tag,
        )
        rec = match_reference_points(
            reference_points_df,
            candidate_universe_df=audit_universe_df,
            replicate_id=int(yi),
            replicate_kind='yearwise',
            match_cfg=settings.matching,
            pair_df=pair_df,
        )
        for _, row in rec.iterrows():
            if str(row['match_type']) == 'strict_match' and (
                bool(getattr(settings.yearwise, 'strict_exact_equivalent', True))
                or abs(int(row['day_offset'])) <= int(settings.yearwise.exact_hit_max_abs_offset_days)
            ):
                hit = 'exact_hit'
            elif str(row['match_type']) in ('matched_point', 'near_support_only'):
                hit = 'near_hit'
            else:
                hit = 'missing'
            long_rows.append(
                {
                    'year': int(years_arr[yi]),
                    'year_index': int(yi),
                    'point_id': str(row['point_id']),
                    'reference_day': int(row['point_day']),
                    'detected_day': int(row['nearest_peak_day']) if pd.notna(row['nearest_peak_day']) else np.nan,
                    'offset': int(row['day_offset']) if pd.notna(row['day_offset']) else np.nan,
                    'match_type': row['match_type'],
                    'hit': hit,
                    'peak_score': float(row['nearest_peak_score']) if pd.notna(row['nearest_peak_score']) else np.nan,
                }
            )
    long_df = pd.DataFrame(
        long_rows,
        columns=['year', 'year_index', 'point_id', 'reference_day', 'detected_day', 'offset', 'match_type', 'hit', 'peak_score'],
    )
    if long_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                'point_id', 'yearwise_exact_hit_fraction', 'yearwise_near_hit_fraction', 'yearwise_missing_fraction',
                'n_years_total', 'n_exact_hit_years', 'n_near_hit_years', 'n_missing_years',
                'median_yearwise_peak_score', 'exact_definition',
                'strict_match_fraction', 'matched_point_fraction', 'near_support_only_fraction', 'no_match_fraction',
            ]
        )
        matchtype_summary_df = pd.DataFrame(columns=['point_id', 'strict_match_fraction', 'matched_point_fraction', 'near_support_only_fraction', 'no_match_fraction'])
        pair_summary_df = pd.DataFrame(columns=[
            'pair_id', 'point_a_id', 'point_b_id', 'point_a_yearwise_dominant_rate', 'point_b_yearwise_dominant_rate',
            'pair_yearwise_tie_rate', 'n_years_compared', 'pair_yearwise_method', 'pair_yearwise_requires_both_detected', 'pair_yearwise_is_conservative',
        ])
        return {'long_df': long_df, 'summary_df': summary_df, 'matchtype_summary_df': matchtype_summary_df, 'pair_summary_df': pair_summary_df}
    summary_rows = []
    matchtype_rows = []
    for pid, sub in long_df.groupby('point_id'):
        n = len(sub)
        summary_rows.append(
            {
                'point_id': pid,
                'yearwise_exact_hit_fraction': float((sub['hit'] == 'exact_hit').mean()),
                'yearwise_near_hit_fraction': float((sub['hit'] == 'near_hit').mean()),
                'yearwise_missing_fraction': float((sub['hit'] == 'missing').mean()),
                'n_years_total': int(n),
                'n_exact_hit_years': int((sub['hit'] == 'exact_hit').sum()),
                'n_near_hit_years': int((sub['hit'] == 'near_hit').sum()),
                'n_missing_years': int((sub['hit'] == 'missing').sum()),
                'median_yearwise_peak_score': float(sub['peak_score'].median()) if sub['peak_score'].notna().any() else np.nan,
                'exact_definition': 'strict_match_only' if bool(getattr(settings.yearwise, 'strict_exact_equivalent', True)) else f"strict_match_with_abs_offset_le_{int(settings.yearwise.exact_hit_max_abs_offset_days)}",
            }
        )
        matchtype_rows.append(
            {
                'point_id': pid,
                'strict_match_fraction': float((sub['match_type'] == 'strict_match').mean()),
                'matched_point_fraction': float((sub['match_type'] == 'matched_point').mean()),
                'near_support_only_fraction': float((sub['match_type'] == 'near_support_only').mean()),
                'no_match_fraction': float((sub['match_type'] == 'no_match').mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    matchtype_summary_df = pd.DataFrame(matchtype_rows)
    summary_df = summary_df.merge(matchtype_summary_df, on='point_id', how='left')
    pair_summary_df = _build_pair_summary(long_df, pair_df, settings)
    return {
        'long_df': long_df.sort_values(['point_id', 'year']).reset_index(drop=True),
        'summary_df': summary_df.sort_values('point_id').reset_index(drop=True),
        'matchtype_summary_df': matchtype_summary_df.sort_values('point_id').reset_index(drop=True),
        'pair_summary_df': pair_summary_df.sort_values(['pair_id']).reset_index(drop=True),
    }
