from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV3Settings
from .profiles import ObjectProfile
from .safe_stats import safe_nanmean
from .detector_ruptures_window import run_ruptures_window
from .window_builder import build_primary_windows_from_ruptures
from .matching_rules import classify_window_relation


_RELATION_PRIORITY = {
    'strict_overlap': 3,
    'overlap_any': 2,
    'near_peak_only': 1,
    'none': 0,
}


def _build_year_state_matrix(profile_dict: dict[str, ObjectProfile], year_idx: int) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    widths = []
    for name in ['P', 'V', 'H', 'Je', 'Jw']:
        seasonal = profile_dict[name].raw_cube[year_idx, :, :].astype(float)
        blocks.append(seasonal)
        widths.append(seasonal.shape[1])
    raw = np.concatenate(blocks, axis=1)
    mean, _ = safe_nanmean(raw, axis=0, return_valid_count=True)
    centered = raw - mean[None, :]
    var, _ = safe_nanmean(np.square(centered), axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    state = centered / std[None, :]
    start = 0
    for width in widths:
        state[:, start:start + width] *= 1.0 / np.sqrt(width)
        start += width
    valid_day_mask = np.all(np.isfinite(state), axis=1)
    return state, valid_day_mask


def _nearest_window(day: int, main_windows_df: pd.DataFrame, tolerance_days: int) -> tuple[str | None, int | None, int | None, bool]:
    if main_windows_df.empty:
        return None, None, None, False
    tmp = main_windows_df.copy()
    tmp['abs_dist'] = (tmp['center_day'].astype(int) - int(day)).abs()
    row = tmp.sort_values(['abs_dist', 'center_day']).iloc[0]
    offset = int(day - int(row['center_day']))
    return str(row['window_id']), int(row['center_day']), offset, bool(abs(offset) <= int(tolerance_days))


def _select_best_relation(relations: list[dict]) -> dict | None:
    if not relations:
        return None
    rel_df = pd.DataFrame(relations)
    rel_df['priority'] = rel_df['relation_tier'].map(_RELATION_PRIORITY).fillna(-1).astype(int)
    rel_df = rel_df.sort_values(
        ['priority', 'overlap_ratio', 'overlap_days', 'center_distance_days', 'candidate_center_day'],
        ascending=[False, False, False, True, True],
    )
    return rel_df.iloc[0].to_dict()


def build_yearwise_diagnostics(profile_dict: dict[str, ObjectProfile], years: np.ndarray, main_windows_df: pd.DataFrame, settings: StagePartitionV3Settings) -> dict[str, pd.DataFrame]:
    point_rows = []
    candidate_rows = []
    membership_rows = []
    peak_rows = []
    relation_rows = []
    years_arr = np.asarray(years).astype(int)
    min_required = max(2 * int(settings.ruptures_window.width), 3)

    for year_idx, year_value in enumerate(years_arr.tolist()):
        state, valid_day_mask = _build_year_state_matrix(profile_dict, year_idx)
        valid_day_index = np.where(valid_day_mask)[0].astype(int)
        if valid_day_index.size < min_required:
            continue
        state_valid = state[valid_day_mask, :]
        out = run_ruptures_window(state_valid, settings.ruptures_window, day_index=valid_day_index)
        cand_windows, _, _, _ = build_primary_windows_from_ruptures(out['points'], out['profile'], settings.edge, settings.window_construction)

        for point_day in out['points'].astype(int).tolist():
            nearest_id, nearest_center, day_offset, near_flag = _nearest_window(
                point_day, main_windows_df, settings.yearwise_support.near_peak_tolerance_days
            )
            point_rows.append({
                'year': int(year_value),
                'raw_peak_day': int(point_day),
                'mapped_peak_day': int(point_day),
                'peak_score': float(out['profile'].get(point_day, np.nan)) if not out['profile'].empty else np.nan,
                'is_near_main_window': bool(near_flag),
                'nearest_window_id': nearest_id,
                'nearest_window_center_day': nearest_center,
                'day_offset_to_window_center': day_offset,
            })

        if not cand_windows.empty:
            for cand_idx, row in cand_windows.sort_values(['center_day', 'start_day']).reset_index(drop=True).iterrows():
                nearest_id, nearest_center, offset, near_flag = _nearest_window(
                    int(row['center_day']), main_windows_df, settings.yearwise_support.near_peak_tolerance_days
                )
                overlap_days = 0
                overlap_ratio = 0.0
                relation_tier = 'none'
                if nearest_id is not None:
                    ref = main_windows_df[main_windows_df['window_id'] == nearest_id].iloc[0]
                    relation = classify_window_relation(
                        int(row['start_day']), int(row['end_day']), int(row['center_day']),
                        int(ref['start_day']), int(ref['end_day']), int(ref['center_day']),
                        near_peak_tolerance_days=settings.yearwise_support.near_peak_tolerance_days,
                        overlap_days_min=settings.yearwise_support.overlap_days_min,
                        overlap_ratio_min=settings.yearwise_support.overlap_ratio_min,
                    )
                    overlap_days = relation.overlap_days
                    overlap_ratio = relation.overlap_ratio
                    relation_tier = relation.relation_tier
                candidate_rows.append({
                    'year': int(year_value),
                    'candidate_id': f'Y{int(year_value)}_C{cand_idx + 1:02d}',
                    'start_day': int(row['start_day']),
                    'end_day': int(row['end_day']),
                    'center_day': int(row['center_day']),
                    'width_days': int(row['end_day'] - row['start_day'] + 1),
                    'nearest_window_id': nearest_id,
                    'nearest_window_center_day': nearest_center,
                    'nearest_window_distance_days': offset,
                    'overlap_ratio_with_main_window': float(overlap_ratio),
                    'overlap_days_with_main_window': int(overlap_days),
                    'relation_tier': relation_tier,
                })

        for _, win in main_windows_df.iterrows():
            win_id = str(win['window_id'])
            per_relations = []
            if not cand_windows.empty:
                for cand_idx, row in cand_windows.sort_values(['center_day', 'start_day']).reset_index(drop=True).iterrows():
                    relation = classify_window_relation(
                        int(row['start_day']), int(row['end_day']), int(row['center_day']),
                        int(win['start_day']), int(win['end_day']), int(win['center_day']),
                        near_peak_tolerance_days=settings.yearwise_support.near_peak_tolerance_days,
                        overlap_days_min=settings.yearwise_support.overlap_days_min,
                        overlap_ratio_min=settings.yearwise_support.overlap_ratio_min,
                    )
                    per_relations.append({
                        'year': int(year_value),
                        'window_id': win_id,
                        'candidate_id': f'Y{int(year_value)}_C{cand_idx + 1:02d}',
                        'candidate_start_day': int(row['start_day']),
                        'candidate_end_day': int(row['end_day']),
                        'candidate_center_day': int(row['center_day']),
                        'main_start_day': int(win['start_day']),
                        'main_end_day': int(win['end_day']),
                        'main_center_day': int(win['center_day']),
                        'overlap_days': int(relation.overlap_days),
                        'overlap_ratio': float(relation.overlap_ratio),
                        'center_distance_days': int(relation.center_distance_days),
                        'near_peak_flag': bool(relation.near_peak_flag),
                        'overlap_any_flag': bool(relation.overlap_any_flag),
                        'strict_overlap_flag': bool(relation.strict_overlap_flag),
                        'relation_tier': relation.relation_tier,
                    })
            best = _select_best_relation(per_relations)
            if best is None:
                best = {
                    'year': int(year_value),
                    'window_id': win_id,
                    'candidate_id': None,
                    'candidate_start_day': np.nan,
                    'candidate_end_day': np.nan,
                    'candidate_center_day': np.nan,
                    'main_start_day': int(win['start_day']),
                    'main_end_day': int(win['end_day']),
                    'main_center_day': int(win['center_day']),
                    'overlap_days': 0,
                    'overlap_ratio': 0.0,
                    'center_distance_days': np.nan,
                    'near_peak_flag': False,
                    'overlap_any_flag': False,
                    'strict_overlap_flag': False,
                    'relation_tier': 'none',
                }
            best['is_best_relation_for_year_window'] = True
            relation_rows.append(best)

            local_peak_day = np.nan
            local_peak_score = np.nan
            is_peak_detected = False
            peak_rank = np.nan
            if not out['profile'].empty:
                local = out['profile'].sort_values(ascending=False)
                if not local.empty:
                    for rank, (peak_day, peak_score) in enumerate(local.items(), start=1):
                        if abs(int(peak_day) - int(win['center_day'])) <= settings.yearwise_support.near_peak_tolerance_days:
                            local_peak_day = int(peak_day)
                            local_peak_score = float(peak_score)
                            is_peak_detected = True
                            peak_rank = int(rank)
                            break
            best_overlap_days = int(best['overlap_days'])
            best_relation_tier = str(best['relation_tier'])
            for day in range(int(win['start_day']), int(win['end_day']) + 1):
                membership_rows.append({
                    'year': int(year_value),
                    'window_id': win_id,
                    'day': int(day),
                    'is_member': bool(best_overlap_days > 0),
                    'membership_source': 'yearwise_candidate_overlap',
                    'relation_tier': best_relation_tier,
                    'strict_overlap_flag': bool(best.get('strict_overlap_flag', False)),
                    'overlap_any_flag': bool(best.get('overlap_any_flag', False)),
                })
            peak_rows.append({
                'year': int(year_value),
                'window_id': win_id,
                'local_peak_day': local_peak_day,
                'local_peak_score': local_peak_score,
                'peak_rank_within_year': peak_rank,
                'is_peak_detected_near_window': bool(is_peak_detected),
                'best_relation_tier': best_relation_tier,
                'overlap_ratio_with_window': float(best['overlap_ratio']),
                'overlap_days_with_window': best_overlap_days,
                'near_peak_flag': bool(best.get('near_peak_flag', False)),
                'strict_overlap_flag': bool(best.get('strict_overlap_flag', False)),
            })

    relation_df = pd.DataFrame(relation_rows)
    if not relation_df.empty:
        relation_df['is_best_relation_for_year_window'] = relation_df['is_best_relation_for_year_window'].astype(bool)

    support_summary_df = pd.DataFrame()
    if not relation_df.empty:
        rows = []
        for window_id, sub in relation_df.groupby('window_id'):
            sub = sub.copy()
            overlap_only = sub[sub['overlap_any_flag'] == True]
            near_only = sub[sub['near_peak_flag'] == True]
            rows.append({
                'window_id': str(window_id),
                'n_years_total': int(sub['year'].nunique()),
                'n_years_near_peak': int(near_only['year'].nunique()),
                'n_years_overlap_any': int(overlap_only['year'].nunique()),
                'n_years_strict_overlap': int(sub[sub['strict_overlap_flag'] == True]['year'].nunique()),
                'frac_years_near_peak': float(near_only['year'].nunique() / max(1, sub['year'].nunique())),
                'frac_years_overlap_any': float(overlap_only['year'].nunique() / max(1, sub['year'].nunique())),
                'frac_years_strict_overlap': float(sub[sub['strict_overlap_flag'] == True]['year'].nunique() / max(1, sub['year'].nunique())),
                'median_overlap_ratio_when_overlap': float(overlap_only['overlap_ratio'].median()) if not overlap_only.empty else np.nan,
                'median_center_distance_when_near': float(near_only['center_distance_days'].median()) if not near_only.empty else np.nan,
                'headline_support_mode': settings.yearwise_support.headline_support_mode,
                'headline_support_count': int(sub[sub['strict_overlap_flag'] == True]['year'].nunique()) if settings.yearwise_support.headline_support_mode == 'strict_overlap' else int(overlap_only['year'].nunique()),
                'headline_support_fraction': float(sub[sub['strict_overlap_flag'] == True]['year'].nunique() / max(1, sub['year'].nunique())) if settings.yearwise_support.headline_support_mode == 'strict_overlap' else float(overlap_only['year'].nunique() / max(1, sub['year'].nunique())),
            })
        support_summary_df = pd.DataFrame(rows).sort_values('window_id').reset_index(drop=True)

    return {
        'yearwise_primary_points': pd.DataFrame(point_rows),
        'yearwise_window_candidates': pd.DataFrame(candidate_rows),
        'yearwise_window_membership': pd.DataFrame(membership_rows),
        'yearwise_detector_peak_summary': pd.DataFrame(peak_rows),
        'yearwise_window_relation_audit': relation_df,
        'yearwise_window_support_summary': support_summary_df,
    }
