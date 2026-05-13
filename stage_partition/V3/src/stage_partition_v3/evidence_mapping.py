from __future__ import annotations

import pandas as pd


def build_window_evidence_mapping(main_windows_df: pd.DataFrame, point_df: pd.DataFrame, band_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'window_id', 'legacy_window_id', 'evidence_object_type', 'evidence_object_id',
        'point_id', 'point_day', 'band_id', 'band_start_day', 'band_end_day',
        'band_window_overlap_days', 'band_window_overlap_ratio',
        'used_in_window_consolidation', 'evidence_status', 'warning_flag'
    ]
    rows = []
    if main_windows_df.empty:
        return pd.DataFrame(columns=cols)

    mapped_point_ids = set()
    mapped_band_ids = set()
    for _, win in main_windows_df.iterrows():
        win_id = str(win['window_id'])
        legacy_id = str(win['legacy_window_id'])
        start_day = int(win['start_day'])
        end_day = int(win['end_day'])

        if not point_df.empty:
            point_mask = (point_df['assigned_window_id_legacy'] == legacy_id)
            if not point_mask.any():
                point_mask = point_df['point_day'].between(start_day, end_day, inclusive='both')
            for _, p in point_df.loc[point_mask].iterrows():
                mapped_point_ids.add(str(p['point_id']))
                rows.append({
                    'window_id': win_id,
                    'legacy_window_id': legacy_id,
                    'evidence_object_type': 'point',
                    'evidence_object_id': p['point_id'],
                    'point_id': p['point_id'],
                    'point_day': int(p['point_day']),
                    'band_id': None,
                    'band_start_day': None,
                    'band_end_day': None,
                    'band_window_overlap_days': None,
                    'band_window_overlap_ratio': None,
                    'used_in_window_consolidation': True,
                    'evidence_status': 'mapped_to_main_window',
                    'warning_flag': '',
                })

        if not band_df.empty:
            band_mask = (band_df['source_window_id_legacy'] == legacy_id)
            if not band_mask.any():
                band_mask = (band_df['band_start_day'] <= end_day) & (band_df['band_end_day'] >= start_day)
            for _, b in band_df.loc[band_mask].iterrows():
                overlap_days = max(0, min(end_day, int(b['band_end_day'])) - max(start_day, int(b['band_start_day'])) + 1)
                overlap_ratio = overlap_days / max(1, int(b['band_end_day']) - int(b['band_start_day']) + 1)
                mapped_band_ids.add(str(b['band_id']))
                rows.append({
                    'window_id': win_id,
                    'legacy_window_id': legacy_id,
                    'evidence_object_type': 'band',
                    'evidence_object_id': b['band_id'],
                    'point_id': None,
                    'point_day': None,
                    'band_id': b['band_id'],
                    'band_start_day': int(b['band_start_day']),
                    'band_end_day': int(b['band_end_day']),
                    'band_window_overlap_days': int(overlap_days),
                    'band_window_overlap_ratio': float(overlap_ratio),
                    'used_in_window_consolidation': True,
                    'evidence_status': 'mapped_to_main_window',
                    'warning_flag': '' if overlap_ratio >= 0.5 else 'partial_overlap_only',
                })

    if not point_df.empty:
        for _, p in point_df.iterrows():
            pid = str(p['point_id'])
            if pid in mapped_point_ids:
                continue
            rows.append({
                'window_id': None,
                'legacy_window_id': None,
                'evidence_object_type': 'point',
                'evidence_object_id': pid,
                'point_id': pid,
                'point_day': int(p['point_day']),
                'band_id': None,
                'band_start_day': None,
                'band_end_day': None,
                'band_window_overlap_days': None,
                'band_window_overlap_ratio': None,
                'used_in_window_consolidation': False,
                'evidence_status': 'orphan_point',
                'warning_flag': 'point_not_mapped_to_any_main_window',
            })

    if not band_df.empty:
        for _, b in band_df.iterrows():
            bid = str(b['band_id'])
            if bid in mapped_band_ids:
                continue
            rows.append({
                'window_id': None,
                'legacy_window_id': None,
                'evidence_object_type': 'band',
                'evidence_object_id': bid,
                'point_id': None,
                'point_day': None,
                'band_id': bid,
                'band_start_day': int(b['band_start_day']),
                'band_end_day': int(b['band_end_day']),
                'band_window_overlap_days': 0,
                'band_window_overlap_ratio': 0.0,
                'used_in_window_consolidation': False,
                'evidence_status': 'orphan_band',
                'warning_flag': 'band_not_mapped_to_any_main_window',
            })

    return pd.DataFrame(rows, columns=cols)
