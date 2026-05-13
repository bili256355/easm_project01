from __future__ import annotations
import numpy as np
import pandas as pd
from .state_builder import build_resampled_state_matrix
from .detector_ruptures_window import run_point_detector
from .point_matching import match_reference_to_local_peaks, summarize_match_records


def _progress_iter(items, enabled: bool, desc: str):
    if enabled:
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc)
        except Exception:
            return items
    return items


def run_point_bootstrap_local_peak(profiles: dict, years: np.ndarray, reference_points_df: pd.DataFrame, settings) -> dict:
    rng = np.random.default_rng(settings.bootstrap.random_seed)
    n_years = len(np.asarray(years).astype(int))
    records = []
    meta_rows = []
    point_day_map = {str(r['point_id']): int(r['point_day']) for _, r in reference_points_df.iterrows()}
    for rep in _progress_iter(range(int(settings.bootstrap.n_bootstrap)), settings.bootstrap.progress, 'V5 bootstrap'):
        sampled = rng.integers(0, n_years, size=n_years, endpoint=False)
        state = build_resampled_state_matrix(profiles, sampled, settings.state)
        if state['valid_day_index'].size < max(2 * int(settings.detector.width), 3):
            meta_rows.append({'replicate_id': int(rep), 'status': 'skipped_insufficient_valid_days', 'n_valid_days': int(state['valid_day_index'].size)})
            continue
        det = run_point_detector(
            state['state_matrix'][state['valid_day_mask'], :],
            state['valid_day_index'],
            settings.detector,
            local_peak_distance_days=settings.detector.local_peak_min_distance_days,
        )
        rec = match_reference_to_local_peaks(
            reference_points_df,
            det['local_peaks_df'],
            settings.bootstrap,
            replicate_id=int(rep),
            replicate_kind='bootstrap',
            replicate_label=int(rep),
        )
        records.append(rec)
        meta_rows.append({'replicate_id': int(rep), 'status': 'success', 'n_valid_days': int(state['valid_day_index'].size), 'sampled_year_indices': sampled.tolist()})
    records_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=['replicate_id','replicate_kind','replicate_label','point_id','point_day','month_day','matched_peak_day','matched_peak_score','matched_peak_prominence','offset_days','abs_offset_days','match_type'])
    summary_df = summarize_match_records(records_df, group_field='bootstrap_local_peak', point_day_map=point_day_map)
    return {'records_df': records_df, 'summary_df': summary_df, 'meta_df': pd.DataFrame(meta_rows)}
