from __future__ import annotations
import numpy as np
import pandas as pd
from .detector_ruptures_window import extract_ranked_local_peaks
from .timeline import day_index_to_month_day


def _profile_value_at_day(profile: pd.Series, day: int) -> float:
    if profile is None or len(profile) == 0:
        return np.nan
    try:
        if day in profile.index:
            val = profile.loc[day]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            return float(val) if pd.notna(val) else np.nan
    except Exception:
        pass
    return np.nan


def _best_observed_peak(local_peaks_df: pd.DataFrame, point_day: int, tolerance_days: int = 2):
    if local_peaks_df is None or local_peaks_df.empty:
        return np.nan, np.nan, np.nan, np.nan
    peaks = local_peaks_df.copy()
    if 'peak_day' not in peaks.columns:
        return np.nan, np.nan, np.nan, np.nan
    peaks = peaks.dropna(subset=['peak_day']).copy()
    if peaks.empty:
        return np.nan, np.nan, np.nan, np.nan
    peaks['peak_day'] = peaks['peak_day'].astype(int)
    peaks['peak_score'] = pd.to_numeric(peaks.get('peak_score'), errors='coerce')
    peaks['peak_prominence'] = pd.to_numeric(peaks.get('peak_prominence'), errors='coerce')
    peaks['abs_offset'] = (peaks['peak_day'] - int(point_day)).abs()
    peaks = peaks[peaks['abs_offset'] <= int(tolerance_days)].copy()
    if peaks.empty:
        return np.nan, np.nan, np.nan, np.nan
    peaks = peaks.sort_values(['abs_offset', 'peak_score', 'peak_day'], ascending=[True, False, True]).reset_index(drop=True)
    best = peaks.iloc[0]
    return (
        float(best['peak_score']) if pd.notna(best['peak_score']) else np.nan,
        float(best['peak_prominence']) if pd.notna(best['peak_prominence']) else np.nan,
        int(best['peak_day']) if pd.notna(best['peak_day']) else np.nan,
        1,
    )


def _normalize_formal_reference(formal_reference_df: pd.DataFrame, local_peaks_df: pd.DataFrame, profile: pd.Series) -> pd.DataFrame:
    if formal_reference_df is None or formal_reference_df.empty:
        return pd.DataFrame(columns=['candidate_id','candidate_day','month_day','candidate_score','candidate_prominence','candidate_kind','source_type','role_group','role_detail','is_headline_primary','nearest_formal_point_id','distance_to_formal','keep_reason','matched_observed_peak_day','matched_observed_peak_rank'])
    rows = []
    for _, row in formal_reference_df.iterrows():
        point_day = int(row['point_day'])
        score, prom, matched_day, matched_rank = _best_observed_peak(local_peaks_df, point_day, tolerance_days=2)
        keep_reason = 'formal_primary_bound_to_observed_peak'
        if not np.isfinite(score):
            score = float(row.get('peak_score')) if pd.notna(row.get('peak_score')) else np.nan
            prom = float(row.get('peak_prominence')) if pd.notna(row.get('peak_prominence')) else np.nan
            matched_day = int(row.get('matched_peak_day')) if pd.notna(row.get('matched_peak_day')) else np.nan
            matched_rank = np.nan
            keep_reason = 'formal_primary_fallback_to_reference_score'
        if not np.isfinite(score):
            score = _profile_value_at_day(profile, point_day)
            prom = prom if pd.notna(prom) else np.nan
            matched_day = point_day if pd.notna(point_day) else np.nan
            matched_rank = matched_rank if pd.notna(matched_rank) else np.nan
            keep_reason = 'formal_primary_fallback_to_profile_value'
        rows.append({
            'candidate_id': str(row['point_id']),
            'candidate_day': point_day,
            'month_day': row.get('month_day') or day_index_to_month_day(point_day),
            'candidate_score': float(score) if pd.notna(score) else np.nan,
            'candidate_prominence': float(prom) if pd.notna(prom) else np.nan,
            'candidate_kind': 'formal_primary',
            'source_type': 'formal_primary',
            'role_group': 'formal_primary',
            'role_detail': 'headline_primary',
            'is_headline_primary': True,
            'nearest_formal_point_id': str(row['point_id']),
            'distance_to_formal': 0,
            'keep_reason': keep_reason,
            'matched_observed_peak_day': int(matched_day) if pd.notna(matched_day) else np.nan,
            'matched_observed_peak_rank': matched_rank,
        })
    return pd.DataFrame(rows)


def build_point_audit_universe(
    formal_reference_df: pd.DataFrame,
    local_peaks_df: pd.DataFrame,
    profile: pd.Series,
    *,
    radius_days: int,
    weak_peak_min_prominence: float,
    source_run_tag: str,
) -> pd.DataFrame:
    cols = ['candidate_id','candidate_day','month_day','candidate_score','candidate_prominence','candidate_kind','source_type','role_group','role_detail','is_headline_primary','nearest_formal_point_id','distance_to_formal','keep_reason','matched_observed_peak_day','matched_observed_peak_rank','source_run_tag']
    formal_rows = _normalize_formal_reference(formal_reference_df, local_peaks_df, profile)
    if formal_rows.empty:
        return pd.DataFrame(columns=cols)

    formal_days = set(formal_rows['candidate_day'].astype(int).tolist())
    formal_points = formal_reference_df[['point_id','point_day']].copy().sort_values('point_day').reset_index(drop=True)

    surviving = local_peaks_df.copy() if local_peaks_df is not None else pd.DataFrame(columns=['peak_id','peak_day','peak_score','peak_prominence'])
    for c in ['peak_id','peak_day','peak_score','peak_prominence']:
        if c not in surviving.columns:
            surviving[c] = np.nan if c != 'peak_id' else None
    all_peaks = extract_ranked_local_peaks(profile, min_distance_days=1, prominence_min=0.0)
    if not all_peaks.empty:
        all_peaks = all_peaks.copy()
        all_peaks['source_type'] = 'profile_local_peak'
    if not surviving.empty:
        surviving = surviving.copy()
        surviving['source_type'] = 'detector_local_peak'

    peak_frames = []
    if not surviving.empty:
        peak_frames.append(surviving[['peak_id','peak_day','peak_score','peak_prominence','source_type']])
    if not all_peaks.empty:
        peak_frames.append(all_peaks[['peak_id','peak_day','peak_score','peak_prominence','source_type']])
    if peak_frames:
        peaks = pd.concat(peak_frames, ignore_index=True, sort=False)
    else:
        peaks = pd.DataFrame(columns=['peak_id','peak_day','peak_score','peak_prominence','source_type'])
    if peaks.empty:
        out = formal_rows.copy()
        out['source_run_tag'] = source_run_tag
        return out[cols].sort_values(['candidate_day','is_headline_primary','candidate_id'], ascending=[True,False,True]).reset_index(drop=True)

    peaks = peaks.dropna(subset=['peak_day']).copy()
    peaks['peak_day'] = peaks['peak_day'].astype(int)
    peaks['peak_score'] = pd.to_numeric(peaks['peak_score'], errors='coerce')
    peaks['peak_prominence'] = pd.to_numeric(peaks['peak_prominence'], errors='coerce').fillna(0.0)
    peaks = peaks.sort_values(['peak_day','source_type','peak_score'], ascending=[True,True,False]).reset_index(drop=True)

    merged_peak_rows = []
    for day, grp in peaks.groupby('peak_day', sort=True):
        grp = grp.copy().reset_index(drop=True)
        detector_grp = grp[grp['source_type'] == 'detector_local_peak']
        best = detector_grp.iloc[0] if not detector_grp.empty else grp.iloc[0]
        merged_peak_rows.append({
            'peak_id': str(best['peak_id']) if pd.notna(best['peak_id']) else f'LPD{int(day):03d}',
            'peak_day': int(day),
            'peak_score': float(best['peak_score']) if pd.notna(best['peak_score']) else np.nan,
            'peak_prominence': float(best['peak_prominence']) if pd.notna(best['peak_prominence']) else np.nan,
            'source_type': str(best['source_type']),
            'has_detector_local_peak': bool((grp['source_type'] == 'detector_local_peak').any()),
            'has_profile_local_peak': bool((grp['source_type'] == 'profile_local_peak').any()),
        })
    merged_peaks = pd.DataFrame(merged_peak_rows)

    weak_rows = []
    for _, peak in merged_peaks.iterrows():
        peak_day = int(peak['peak_day'])
        if peak_day in formal_days:
            continue
        distances = (formal_points['point_day'].astype(int) - peak_day).abs()
        nearest_idx = int(distances.idxmin())
        nearest_formal = formal_points.loc[nearest_idx]
        distance = int(abs(int(nearest_formal['point_day']) - peak_day))
        if distance > int(radius_days):
            continue
        prominence = float(peak['peak_prominence']) if pd.notna(peak['peak_prominence']) else np.nan
        if np.isfinite(prominence) and prominence < float(weak_peak_min_prominence):
            continue
        weak_rows.append({
            'candidate_id': str(peak['peak_id']) if pd.notna(peak['peak_id']) else f'LPD{peak_day:03d}',
            'candidate_day': peak_day,
            'month_day': day_index_to_month_day(peak_day),
            'candidate_score': float(peak['peak_score']) if pd.notna(peak['peak_score']) else np.nan,
            'candidate_prominence': prominence,
            'candidate_kind': 'local_peak',
            'source_type': str(peak['source_type']),
            'role_group': 'neighbor_competition',
            'role_detail': 'neighbor_local_peak',
            'is_headline_primary': False,
            'nearest_formal_point_id': str(nearest_formal['point_id']),
            'distance_to_formal': distance,
            'keep_reason': 'weak_neighbor_peak_retained_for_audit',
            'matched_observed_peak_day': peak_day,
            'matched_observed_peak_rank': np.nan,
        })

    weak_df = pd.DataFrame(weak_rows)
    out = pd.concat([formal_rows, weak_df], ignore_index=True, sort=False)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out['kind_priority'] = out['candidate_kind'].map({'formal_primary': 0, 'local_peak': 1}).fillna(9)
    out = out.sort_values(['candidate_day','kind_priority','candidate_score','candidate_id'], ascending=[True,True,False,True]).drop_duplicates(subset=['candidate_day'], keep='first').reset_index(drop=True)
    out['source_run_tag'] = source_run_tag
    return out[cols].sort_values(['candidate_day','is_headline_primary','candidate_id'], ascending=[True,False,True]).reset_index(drop=True)
