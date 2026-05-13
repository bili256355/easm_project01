from __future__ import annotations
import numpy as np
import pandas as pd


def classify_match_type(abs_offset_days: int | float | None, cfg) -> str:
    if abs_offset_days is None or not np.isfinite(abs_offset_days):
        return 'no_match'
    off = int(abs(abs_offset_days))
    if off <= int(cfg.strict_match_max_abs_offset_days):
        return 'strict'
    if off <= int(cfg.match_max_abs_offset_days):
        return 'matched'
    if off <= int(cfg.near_max_abs_offset_days):
        return 'near'
    return 'no_match'


def match_reference_to_local_peaks(reference_df: pd.DataFrame, local_peaks_df: pd.DataFrame, cfg, *, replicate_id: int, replicate_kind: str, replicate_label: str | int | None = None) -> pd.DataFrame:
    cols = [
        'replicate_id','replicate_kind','replicate_label','point_id','point_day','month_day',
        'matched_peak_day','matched_peak_score','matched_peak_prominence',
        'offset_days','abs_offset_days','match_type'
    ]
    if reference_df is None or reference_df.empty:
        return pd.DataFrame(columns=cols)
    peaks = local_peaks_df.copy() if local_peaks_df is not None else pd.DataFrame(columns=['peak_day','peak_score','peak_prominence'])
    rows = []
    for _, ref in reference_df.iterrows():
        target_day = int(ref['point_day'])
        if peaks.empty:
            matched_peak_day = np.nan
            matched_peak_score = np.nan
            matched_peak_prominence = np.nan
            offset_days = np.nan
            abs_offset_days = np.nan
            match_type = 'no_match'
        else:
            sub = peaks.copy()
            sub['abs_offset_days'] = (pd.to_numeric(sub['peak_day'], errors='coerce') - target_day).abs()
            sub = sub.sort_values(['abs_offset_days', 'peak_score', 'peak_day'], ascending=[True, False, True]).reset_index(drop=True)
            best = sub.iloc[0]
            matched_peak_day = int(best['peak_day']) if pd.notna(best['peak_day']) else np.nan
            matched_peak_score = float(best['peak_score']) if pd.notna(best.get('peak_score')) else np.nan
            matched_peak_prominence = float(best['peak_prominence']) if pd.notna(best.get('peak_prominence')) else np.nan
            offset_days = int(matched_peak_day - target_day) if pd.notna(matched_peak_day) else np.nan
            abs_offset_days = abs(offset_days) if pd.notna(offset_days) else np.nan
            match_type = classify_match_type(abs_offset_days, cfg)
        rows.append({
            'replicate_id': int(replicate_id),
            'replicate_kind': str(replicate_kind),
            'replicate_label': replicate_label,
            'point_id': str(ref['point_id']),
            'point_day': int(ref['point_day']),
            'month_day': ref.get('month_day'),
            'matched_peak_day': matched_peak_day,
            'matched_peak_score': matched_peak_score,
            'matched_peak_prominence': matched_peak_prominence,
            'offset_days': offset_days,
            'abs_offset_days': abs_offset_days,
            'match_type': match_type,
        })
    return pd.DataFrame(rows, columns=cols)


def summarize_match_records(records_df: pd.DataFrame, *, group_field: str, point_day_map: dict[str, int]) -> pd.DataFrame:
    cols = [
        'point_id','point_day',
        f'{group_field}_strict_fraction',
        f'{group_field}_match_fraction',
        f'{group_field}_near_fraction',
        f'{group_field}_no_match_fraction',
        f'median_{group_field}_matched_peak_score',
        f'median_{group_field}_abs_offset_days',
    ]
    if records_df is None or records_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for pid, sub in records_df.groupby('point_id'):
        matched_score = pd.to_numeric(sub['matched_peak_score'], errors='coerce')
        abs_off = pd.to_numeric(sub['abs_offset_days'], errors='coerce')
        rows.append({
            'point_id': str(pid),
            'point_day': int(point_day_map.get(str(pid))),
            f'{group_field}_strict_fraction': float((sub['match_type'] == 'strict').mean()),
            f'{group_field}_match_fraction': float(sub['match_type'].isin(['strict', 'matched']).mean()),
            f'{group_field}_near_fraction': float((sub['match_type'] == 'near').mean()),
            f'{group_field}_no_match_fraction': float((sub['match_type'] == 'no_match').mean()),
            f'median_{group_field}_matched_peak_score': float(matched_score.median()) if matched_score.notna().any() else np.nan,
            f'median_{group_field}_abs_offset_days': float(abs_off.median()) if abs_off.notna().any() else np.nan,
        })
    return pd.DataFrame(rows, columns=cols).sort_values(['point_day', 'point_id']).reset_index(drop=True)
