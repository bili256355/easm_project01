from __future__ import annotations
import numpy as np
import pandas as pd
from .state_builder import build_resampled_state_matrix
from .detector_ruptures_window import run_point_detector
from .point_reference import build_formal_reference_points
from .point_audit_universe import build_point_audit_universe
from .point_matching import match_reference_points, summarize_match_records


def _progress_iter(items, enabled: bool, desc: str):
    if enabled:
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc)
        except Exception:
            return items
    return items


def run_point_bootstrap_support(profiles: dict, years: np.ndarray, reference_points_df: pd.DataFrame, settings, pair_df: pd.DataFrame | None = None) -> dict:
    rng = np.random.default_rng(settings.bootstrap.random_seed)
    n_years = len(np.asarray(years).astype(int))
    records = []
    meta = []
    for rep in _progress_iter(range(int(settings.bootstrap.n_bootstrap)), settings.bootstrap.progress, 'V4 bootstrap'):
        sampled = rng.integers(0, n_years, size=n_years, endpoint=False)
        state = build_resampled_state_matrix(profiles, sampled, settings.state)
        if state['valid_day_index'].size < max(2 * int(settings.detector.width), 3):
            meta.append({'replicate_id': int(rep), 'status': 'skipped_insufficient_valid_days', 'n_valid_days': int(state['valid_day_index'].size)})
            continue
        det = run_point_detector(state['state_matrix'][state['valid_day_mask'], :], state['valid_day_index'], settings.detector, local_peak_distance_days=settings.detector.local_peak_min_distance_days)
        formal_ref = build_formal_reference_points(det['primary_points_df'], source_run_tag=settings.output.output_tag)
        audit_universe_df = build_point_audit_universe(
            formal_ref,
            det['local_peaks_df'],
            det['profile'],
            radius_days=settings.point_audit_universe.radius_days,
            weak_peak_min_prominence=settings.point_audit_universe.weak_peak_min_prominence,
            source_run_tag=settings.output.output_tag,
        )
        records.append(match_reference_points(reference_points_df, candidate_universe_df=audit_universe_df, replicate_id=int(rep), replicate_kind='bootstrap', match_cfg=settings.matching, pair_df=pair_df))
        meta.append({'replicate_id': int(rep), 'status': 'success', 'n_valid_days': int(state['valid_day_index'].size), 'sampled_year_indices': sampled.tolist()})
    records_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    summary_df, offsets_df = summarize_match_records(reference_points_df, records_df, pair_df=pair_df)
    return {'records_df': records_df, 'summary_df': summary_df, 'offsets_df': offsets_df, 'meta_df': pd.DataFrame(meta)}
