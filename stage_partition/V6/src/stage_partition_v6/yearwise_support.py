from __future__ import annotations
import numpy as np
import pandas as pd
from .state_builder import build_year_state_matrix
from .detector_ruptures_window import run_point_detector
from .point_matching import match_candidates_to_local_peaks, summarize_match_records


def _progress_iter(items, enabled: bool, desc: str):
    if enabled:
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc)
        except Exception:
            return items
    return items


def run_yearwise_support(profiles: dict, years: np.ndarray, registry_df: pd.DataFrame, settings) -> dict:
    records = []
    years_arr = np.asarray(years).astype(int)
    point_day_map = {str(r['candidate_id']): int(r['point_day']) for _, r in registry_df.iterrows()}
    for yi in _progress_iter(range(len(years_arr)), settings.yearwise.progress, 'V6 yearwise'):
        year_state = build_year_state_matrix(profiles, yi, settings.state)
        if year_state['valid_day_index'].size < max(2 * int(settings.detector.width), 3):
            rec = match_candidates_to_local_peaks(registry_df, pd.DataFrame(columns=['peak_day','peak_score','peak_prominence']), settings.yearwise, replicate_id=int(yi), replicate_kind='yearwise', replicate_label=int(years_arr[yi]))
        else:
            det = run_point_detector(
                year_state['state_matrix'][year_state['valid_day_mask'], :],
                year_state['valid_day_index'],
                settings.detector,
                local_peak_distance_days=settings.detector.local_peak_min_distance_days,
            )
            rec = match_candidates_to_local_peaks(registry_df, det['local_peaks_df'], settings.yearwise, replicate_id=int(yi), replicate_kind='yearwise', replicate_label=int(years_arr[yi]))
        rec['year'] = int(years_arr[yi])
        records.append(rec)
    records_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=['replicate_id','replicate_kind','replicate_label','candidate_id','point_day','month_day','matched_peak_day','matched_peak_score','matched_peak_prominence','offset_days','abs_offset_days','match_type','year'])
    summary_df = summarize_match_records(records_df, group_field='yearwise', point_day_map=point_day_map)
    return {'records_df': records_df, 'summary_df': summary_df}
