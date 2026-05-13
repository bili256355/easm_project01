from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV3Settings
from .profiles import ObjectProfile
from .safe_stats import safe_nanmean
from .detector_ruptures_window import run_ruptures_window, extract_ranked_local_peaks
from .null_significance_utils import generate_yearwise_circular_shift_replicate
from .point_local_match import (
    select_best_matched_peak,
    compute_local_matched_statistic,
    build_matched_stat_record,
)


def _profile_value_at_day(profile: pd.Series, day: int) -> float:
    if profile.empty:
        return float('nan')
    s = profile.sort_index()
    if int(day) in s.index:
        return float(s.loc[int(day)])
    idx = int(np.argmin(np.abs(s.index.to_numpy(dtype=int) - int(day))))
    return float(s.iloc[idx])


def _best_observed_peak(peaks_df: pd.DataFrame, target_day: int, tolerance_days: int) -> tuple[float, float, int | None, int | None]:
    if peaks_df.empty:
        return float('nan'), float('nan'), None, None
    sub = peaks_df.loc[peaks_df['peak_day'].astype(int).between(int(target_day) - int(tolerance_days), int(target_day) + int(tolerance_days))].copy()
    if sub.empty:
        return float('nan'), float('nan'), None, None
    sub['day_offset_abs'] = (sub['peak_day'].astype(int) - int(target_day)).abs()
    sub = sub.sort_values(['day_offset_abs', 'peak_score', 'peak_prominence'], ascending=[True, False, False]).reset_index(drop=True)
    row = sub.iloc[0]
    return float(row['peak_score']), float(row['peak_prominence']), int(row['peak_day']), int(row['peak_rank'])


def build_point_audit_universe(point_df: pd.DataFrame, main_windows_df: pd.DataFrame, profile: pd.Series, settings: StagePartitionV3Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_peaks_df = extract_ranked_local_peaks(
        profile,
        min_distance_days=settings.point_significance.peak_distance_min_days,
        prominence_min=settings.point_significance.peak_prominence_min,
    )
    point_to_window: dict[str, str | None] = {}
    if not main_windows_df.empty:
        for _, row in main_windows_df.iterrows():
            for pid in str(row.get('source_primary_point_ids', '')).split(','):
                pid = pid.strip()
                if pid:
                    point_to_window[pid] = str(row['window_id'])
    formal_rows = []
    tolerance_days = int(settings.point_significance.null_local_match_tolerance_days)
    if not point_df.empty:
        for _, row in point_df.sort_values('point_day').iterrows():
            pid = str(row['point_id'])
            score, prom, matched_day, matched_rank = _best_observed_peak(local_peaks_df, int(row['point_day']), tolerance_days)
            if not np.isfinite(score):
                score = float(row['peak_value']) if pd.notna(row.get('peak_value')) else _profile_value_at_day(profile, int(row['point_day']))
            if not np.isfinite(prom):
                prom = float(row['peak_prominence']) if pd.notna(row.get('peak_prominence')) else np.nan
            formal_rows.append({
                'point_id': pid,
                'point_day': int(row['point_day']),
                'point_role': 'formal_primary_point',
                'related_window_id': point_to_window.get(pid),
                'observed_peak_score': float(score),
                'observed_peak_prominence': float(prom),
                'matched_observed_peak_day': matched_day,
                'matched_observed_peak_rank': matched_rank,
            })
    formal_df = pd.DataFrame(formal_rows)
    radius = int(settings.point_significance.neighbor_competition_radius_days)
    formal_days = formal_df['point_day'].astype(int).tolist() if not formal_df.empty else []
    neighbor_rows = []
    if not local_peaks_df.empty and formal_days:
        for _, row in local_peaks_df.iterrows():
            peak_day = int(row['peak_day'])
            if peak_day in formal_days:
                continue
            nearest_formal_day = min(formal_days, key=lambda x: abs(x - peak_day))
            if abs(peak_day - nearest_formal_day) <= radius:
                nearest_window = None
                if not main_windows_df.empty:
                    tmp = main_windows_df.copy()
                    tmp['abs_dist'] = (tmp['center_day'].astype(int) - peak_day).abs()
                    nearest_window = str(tmp.sort_values(['abs_dist', 'center_day']).iloc[0]['window_id'])
                neighbor_rows.append({
                    'point_id': str(row['peak_id']),
                    'point_day': peak_day,
                    'point_role': 'neighbor_local_peak',
                    'related_window_id': nearest_window,
                    'observed_peak_score': float(row['peak_score']),
                    'observed_peak_prominence': float(row['peak_prominence']),
                    'matched_observed_peak_day': int(row['peak_day']),
                    'matched_observed_peak_rank': int(row['peak_rank']),
                })
    universe_df = pd.concat([formal_df, pd.DataFrame(neighbor_rows)], ignore_index=True)
    if universe_df.empty:
        universe_df = pd.DataFrame(columns=['point_id','point_day','point_role','related_window_id','observed_peak_score','observed_peak_prominence','matched_observed_peak_day','matched_observed_peak_rank'])
    universe_df = universe_df.sort_values(['point_day', 'point_role', 'point_id']).reset_index(drop=True)
    return universe_df, local_peaks_df


def _build_state_matrix_from_sample(profile_dict: dict[str, ObjectProfile], sample_year_indices: np.ndarray, sample_shifts: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = []
    widths = []
    n_sample = len(sample_year_indices)
    for name in ['P', 'V', 'H', 'Je', 'Jw']:
        years_data = []
        raw_cube = profile_dict[name].raw_cube
        for i in range(n_sample):
            yr_idx = int(sample_year_indices[i])
            arr = raw_cube[yr_idx, :, :].astype(float).copy()
            if sample_shifts is not None:
                shift = int(sample_shifts[i])
                if shift != 0:
                    arr = np.roll(arr, shift=shift, axis=0)
            years_data.append(arr)
        cube = np.stack(years_data, axis=0)
        seasonal, _ = safe_nanmean(cube, axis=0, return_valid_count=True)
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
    valid_day_index = np.where(valid_day_mask)[0].astype(int)
    return state, valid_day_mask, valid_day_index


def run_point_null_significance(profile_dict: dict[str, ObjectProfile], years: np.ndarray, point_candidates_df: pd.DataFrame, observed_profile: pd.Series, settings: StagePartitionV3Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs_cols = [
        'point_id','point_day','point_role','related_window_id',
        'observed_peak_score','observed_peak_prominence','observed_peak_rank',
        'observed_matched_peak_day','observed_matched_stat',
        'local_window_start','local_window_end'
    ]
    raw_cols = [
        'replicate_type','replicate_id','point_id','point_day',
        'matched_peak_exists','matched_peak_day','center_offset_days',
        'matched_peak_score','matched_peak_prominence','matched_peak_rank',
        'matched_stat','local_background_median','local_background_mad',
        'match_quality_flag','local_window_start','local_window_end'
    ]
    sig_cols = [
        'point_id','point_day','observed_peak_score','observed_matched_stat',
        'p_global_fwer','p_local','p_match_exist','p_match_exceed_conditional',
        'local_joint_challenge_score','n_null_reps','n_null_reps_with_matched_peak'
    ]
    detail_cols = ['replicate_id','null_global_peak_score']
    scale_cols = [
        'point_id','point_day','observed_peak_score','observed_matched_stat',
        'null_global_peak_median','null_global_peak_p95',
        'null_matched_stat_median','null_matched_stat_p95',
        'matched_stat_ratio','matched_peak_exists_fraction',
        'scale_warning_flag','local_test_mode'
    ]
    if point_candidates_df.empty:
        return (
            pd.DataFrame(columns=obs_cols), pd.DataFrame(columns=raw_cols),
            pd.DataFrame(columns=sig_cols), pd.DataFrame(columns=detail_cols),
            pd.DataFrame(columns=scale_cols)
        )

    point_candidates_df = point_candidates_df.copy()
    observed_profile = observed_profile.copy().sort_index()
    # For observed matching, use aggregated detector local peaks around each formal point.
    observed_peaks_df = extract_ranked_local_peaks(
        observed_profile,
        min_distance_days=settings.point_significance.peak_distance_min_days,
        prominence_min=settings.point_significance.peak_prominence_min,
    )
    observed_rows = []
    raw_rows = []
    for _, cand in point_candidates_df.iterrows():
        target_day = int(cand['point_day'])
        matched = select_best_matched_peak(
            observed_peaks_df,
            target_day=target_day,
            max_center_offset_days=int(getattr(settings.point_significance, 'matched_peak_max_center_offset_days', settings.point_significance.null_local_match_tolerance_days)),
            prominence_min=float(getattr(settings.point_significance, 'matched_peak_min_prominence', 0.0)),
            select_mode=str(getattr(settings.point_significance, 'matched_peak_select_mode', 'closest_then_prominence_then_score')),
        )
        if matched is None:
            obs_score = float(cand.get('observed_peak_score', np.nan))
            obs_prom = float(cand.get('observed_peak_prominence', np.nan))
            matched_day = np.nan
            matched_rank = np.nan
            obs_stat = np.nan
            bg_median = np.nan
            bg_mad = np.nan
            quality_flag = 'no_matched_peak'
        else:
            obs_score = float(matched['peak_score'])
            obs_prom = float(matched.get('peak_prominence', np.nan))
            matched_day = int(matched['peak_day'])
            matched_rank = int(matched['peak_rank']) if pd.notna(matched.get('peak_rank')) else np.nan
            obs_stat, bg_median, bg_mad = compute_local_matched_statistic(
                observed_profile,
                matched_peak_day=matched_day,
                matched_peak_score=obs_score,
                target_day=target_day,
                background_window_days=int(getattr(settings.point_significance, 'local_background_window_days', 7)),
                exclude_core_days=int(getattr(settings.point_significance, 'local_background_exclude_peak_core_days', 1)),
            )
            quality_flag = 'matched_peak' if np.isfinite(obs_stat) else 'matched_peak_bad_background'
        observed_rows.append({
            'point_id': str(cand['point_id']),
            'point_day': target_day,
            'point_role': str(cand['point_role']),
            'related_window_id': cand.get('related_window_id'),
            'observed_peak_score': float(obs_score),
            'observed_peak_prominence': float(obs_prom),
            'observed_peak_rank': matched_rank,
            'observed_matched_peak_day': matched_day,
            'observed_matched_stat': obs_stat,
            'local_window_start': target_day - int(settings.point_significance.local_window_days),
            'local_window_end': target_day + int(settings.point_significance.local_window_days),
        })
        raw_rows.append({
            'replicate_type': 'observed',
            'replicate_id': -1,
            'point_id': str(cand['point_id']),
            'point_day': target_day,
            'matched_peak_exists': matched is not None,
            'matched_peak_day': matched_day,
            'center_offset_days': (matched_day - target_day) if np.isfinite(matched_day) else np.nan,
            'matched_peak_score': obs_score,
            'matched_peak_prominence': obs_prom,
            'matched_peak_rank': matched_rank,
            'matched_stat': obs_stat,
            'local_background_median': bg_median,
            'local_background_mad': bg_mad,
            'match_quality_flag': quality_flag,
            'local_window_start': target_day - int(settings.point_significance.local_window_days),
            'local_window_end': target_day + int(settings.point_significance.local_window_days),
        })
    observed_df = pd.DataFrame(observed_rows, columns=obs_cols)

    rng = np.random.default_rng(settings.support.random_seed + 303)
    years_arr = np.arange(len(np.asarray(years)))
    n_days = int(profile_dict['P'].raw_cube.shape[1])
    global_rows = []
    min_required = max(2 * int(settings.ruptures_window.width), 3)
    for rep in range(int(settings.point_significance.null_reps)):
        shifts = generate_yearwise_circular_shift_replicate(rng, len(years_arr), n_days)
        state, valid_mask, valid_day_index = _build_state_matrix_from_sample(profile_dict, years_arr, shifts)
        if valid_day_index.size < min_required:
            global_rows.append({'replicate_id': rep, 'null_global_peak_score': np.nan})
            for _, cand in point_candidates_df.iterrows():
                target_day = int(cand['point_day'])
                raw_rows.append({
                    'replicate_type': 'null', 'replicate_id': rep, 'point_id': str(cand['point_id']), 'point_day': target_day,
                    'matched_peak_exists': False, 'matched_peak_day': np.nan, 'center_offset_days': np.nan,
                    'matched_peak_score': np.nan, 'matched_peak_prominence': np.nan, 'matched_peak_rank': np.nan,
                    'matched_stat': np.nan, 'local_background_median': np.nan, 'local_background_mad': np.nan,
                    'match_quality_flag': 'insufficient_valid_days',
                    'local_window_start': target_day - int(settings.point_significance.local_window_days),
                    'local_window_end': target_day + int(settings.point_significance.local_window_days),
                })
            continue
        out = run_ruptures_window(state[valid_mask, :], settings.ruptures_window, day_index=valid_day_index)
        profile = out['profile']
        null_peaks_df = extract_ranked_local_peaks(
            profile,
            min_distance_days=settings.point_significance.peak_distance_min_days,
            prominence_min=settings.point_significance.peak_prominence_min,
        )
        global_score = float(null_peaks_df['peak_score'].max()) if not null_peaks_df.empty else np.nan
        global_rows.append({'replicate_id': rep, 'null_global_peak_score': global_score})
        for _, cand in point_candidates_df.iterrows():
            target_day = int(cand['point_day'])
            matched = select_best_matched_peak(
                null_peaks_df,
                target_day=target_day,
                max_center_offset_days=int(getattr(settings.point_significance, 'matched_peak_max_center_offset_days', settings.point_significance.null_local_match_tolerance_days)),
                prominence_min=float(getattr(settings.point_significance, 'matched_peak_min_prominence', 0.0)),
                select_mode=str(getattr(settings.point_significance, 'matched_peak_select_mode', 'closest_then_prominence_then_score')),
            )
            if matched is None:
                rec = build_matched_stat_record(
                    replicate_type='null', replicate_id=rep, point_id=str(cand['point_id']), point_day=target_day,
                    matched_peak_exists=False, matched_peak_day=np.nan, center_offset_days=np.nan,
                    matched_peak_score=np.nan, matched_peak_prominence=np.nan, matched_peak_rank=np.nan,
                    matched_stat=np.nan, local_background_median=np.nan, local_background_mad=np.nan,
                    match_quality_flag='no_matched_peak', local_window_days=int(settings.point_significance.local_window_days),
                )
            else:
                score = float(matched['peak_score'])
                prom = float(matched.get('peak_prominence', np.nan))
                matched_day = int(matched['peak_day'])
                rank = int(matched['peak_rank']) if pd.notna(matched.get('peak_rank')) else np.nan
                stat, bg_median, bg_mad = compute_local_matched_statistic(
                    profile,
                    matched_peak_day=matched_day,
                    matched_peak_score=score,
                    target_day=target_day,
                    background_window_days=int(getattr(settings.point_significance, 'local_background_window_days', 7)),
                    exclude_core_days=int(getattr(settings.point_significance, 'local_background_exclude_peak_core_days', 1)),
                )
                qflag = 'matched_peak' if np.isfinite(stat) else 'matched_peak_bad_background'
                rec = build_matched_stat_record(
                    replicate_type='null', replicate_id=rep, point_id=str(cand['point_id']), point_day=target_day,
                    matched_peak_exists=True, matched_peak_day=matched_day, center_offset_days=matched_day - target_day,
                    matched_peak_score=score, matched_peak_prominence=prom, matched_peak_rank=rank,
                    matched_stat=stat, local_background_median=bg_median, local_background_mad=bg_mad,
                    match_quality_flag=qflag, local_window_days=int(settings.point_significance.local_window_days),
                )
            raw_rows.append(rec)

    raw_df = pd.DataFrame(raw_rows, columns=raw_cols)
    detail_df = pd.DataFrame(global_rows, columns=detail_cols)

    sig_rows = []
    scale_rows = []
    global_scores = detail_df['null_global_peak_score'].dropna().to_numpy(dtype=float)
    n_reps = int(settings.point_significance.null_reps)
    for _, row in observed_df.iterrows():
        pid = str(row['point_id'])
        obs_score = float(row['observed_peak_score'])
        obs_stat = float(row['observed_matched_stat']) if pd.notna(row['observed_matched_stat']) else np.nan
        sub = raw_df[(raw_df['replicate_type'] == 'null') & (raw_df['point_id'] == pid)].copy()
        exists_mask = sub['matched_peak_exists'].fillna(False).astype(bool).to_numpy()
        matched_stats = sub.loc[sub['matched_peak_exists'] == True, 'matched_stat'].dropna().to_numpy(dtype=float)
        p_global = float((np.sum(global_scores >= obs_score) + 1) / (len(global_scores) + 1)) if global_scores.size else np.nan
        p_match_exist = float(np.mean(exists_mask)) if exists_mask.size else np.nan
        p_match_exceed_cond = float((np.sum(matched_stats >= obs_stat) + 1) / (len(matched_stats) + 1)) if (matched_stats.size and np.isfinite(obs_stat)) else np.nan
        joint = float(p_match_exist * p_match_exceed_cond) if np.isfinite(p_match_exist) and np.isfinite(p_match_exceed_cond) else np.nan
        sig_rows.append({
            'point_id': pid,
            'point_day': int(row['point_day']),
            'observed_peak_score': obs_score,
            'observed_matched_stat': obs_stat,
            'p_global_fwer': p_global,
            'p_local': p_match_exceed_cond,
            'p_match_exist': p_match_exist,
            'p_match_exceed_conditional': p_match_exceed_cond,
            'local_joint_challenge_score': joint,
            'n_null_reps': n_reps,
            'n_null_reps_with_matched_peak': int(len(matched_stats)),
        })
        gmed = float(np.nanmedian(global_scores)) if global_scores.size else np.nan
        gp95 = float(np.nanquantile(global_scores, 0.95)) if global_scores.size else np.nan
        lmed = float(np.nanmedian(matched_stats)) if matched_stats.size else np.nan
        lp95 = float(np.nanquantile(matched_stats, 0.95)) if matched_stats.size else np.nan
        ratio = float(obs_stat / lmed) if np.isfinite(obs_stat) and np.isfinite(lmed) and lmed > 0 else np.nan
        scale_warn = bool((np.isfinite(ratio) and ratio < 0.75 and np.isfinite(p_match_exist) and p_match_exist > 0.2) or (np.isfinite(obs_stat) and not np.isfinite(ratio)))
        scale_rows.append({
            'point_id': pid,
            'point_day': int(row['point_day']),
            'observed_peak_score': obs_score,
            'observed_matched_stat': obs_stat,
            'null_global_peak_median': gmed,
            'null_global_peak_p95': gp95,
            'null_matched_stat_median': lmed,
            'null_matched_stat_p95': lp95,
            'matched_stat_ratio': ratio,
            'matched_peak_exists_fraction': p_match_exist,
            'scale_warning_flag': scale_warn,
            'local_test_mode': 'matched_peak_same_scale',
        })
    null_df = pd.DataFrame(sig_rows, columns=sig_cols)
    scale_df = pd.DataFrame(scale_rows, columns=scale_cols)
    return observed_df, raw_df, null_df, detail_df, scale_df

def run_point_stability_audit(profile_dict: dict[str, ObjectProfile], years: np.ndarray, point_candidates_df: pd.DataFrame, state_matrix: np.ndarray, valid_day_mask: np.ndarray, yearwise_primary_points_df: pd.DataFrame, settings: StagePartitionV3Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = ['point_id','bootstrap_presence_fraction','param_presence_fraction','yearwise_near_fraction','yearwise_strict_fraction','median_yearwise_peak_score','stability_warning']
    if point_candidates_df.empty:
        empty = pd.DataFrame(columns=cols)
        blank = pd.DataFrame(columns=['rep','point_id','present_flag','nearest_peak_day','nearest_peak_score','day_offset'])
        year_blank = pd.DataFrame(columns=['year','point_id','formal_point_day','nearest_yearwise_peak_day','nearest_yearwise_peak_score','best_match_peak_rank_within_year','day_offset','best_match_exists','near_point_flag','strict_point_flag'])
        return empty, blank, blank.copy(), year_blank
    rng = np.random.default_rng(settings.support.random_seed + 202)
    n_years = len(np.asarray(years))
    year_indices = np.arange(n_years)
    tolerance = int(settings.point_significance.bootstrap_presence_tolerance_days)
    bootstrap_rows = []
    for rep in range(int(settings.support.bootstrap_reps)):
        sample = rng.integers(0, n_years, size=n_years, endpoint=False)
        state_sample, valid_mask_sample, valid_idx = _build_state_matrix_from_sample(profile_dict, sample, None)
        if valid_idx.size < max(2 * int(settings.ruptures_window.width), 3):
            for _, cand in point_candidates_df.iterrows():
                bootstrap_rows.append({'rep': rep, 'point_id': str(cand['point_id']), 'present_flag': False, 'nearest_peak_day': np.nan, 'nearest_peak_score': np.nan, 'day_offset': np.nan})
            continue
        out = run_ruptures_window(state_sample[valid_mask_sample, :], settings.ruptures_window, day_index=valid_idx)
        peaks = extract_ranked_local_peaks(out['profile'], settings.point_significance.peak_distance_min_days, settings.point_significance.peak_prominence_min)
        for _, cand in point_candidates_df.iterrows():
            if peaks.empty:
                bootstrap_rows.append({'rep': rep, 'point_id': str(cand['point_id']), 'present_flag': False, 'nearest_peak_day': np.nan, 'nearest_peak_score': np.nan, 'day_offset': np.nan})
                continue
            peaks2 = peaks.copy()
            peaks2['abs_dist'] = (peaks2['peak_day'].astype(int) - int(cand['point_day'])).abs()
            row = peaks2.sort_values(['abs_dist','peak_score','peak_prominence'], ascending=[True,False,False]).iloc[0]
            day_offset = int(row['peak_day']) - int(cand['point_day'])
            bootstrap_rows.append({'rep': rep, 'point_id': str(cand['point_id']), 'present_flag': bool(abs(day_offset) <= tolerance), 'nearest_peak_day': int(row['peak_day']), 'nearest_peak_score': float(row['peak_score']), 'day_offset': day_offset})
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    param_rows = []
    width_grid = list(settings.support.ruptures_width_grid)
    pen_grid = list(settings.support.ruptures_pen_grid)
    for width in width_grid:
        for pen in pen_grid:
            cfg = settings.ruptures_window
            local_cfg = type(cfg)(enabled=cfg.enabled, width=int(width), model=cfg.model, min_size=cfg.min_size, jump=cfg.jump, selection_mode='pen', fixed_n_bkps=cfg.fixed_n_bkps, pen=float(pen), epsilon=cfg.epsilon)
            out = run_ruptures_window(state_matrix[valid_day_mask, :], local_cfg, day_index=np.where(valid_day_mask)[0].astype(int))
            peaks = extract_ranked_local_peaks(out['profile'], settings.point_significance.peak_distance_min_days, settings.point_significance.peak_prominence_min)
            for _, cand in point_candidates_df.iterrows():
                if peaks.empty:
                    param_rows.append({'width': width, 'pen': pen, 'point_id': str(cand['point_id']), 'present_flag': False, 'nearest_peak_day': np.nan, 'nearest_peak_score': np.nan, 'day_offset': np.nan})
                    continue
                peaks2 = peaks.copy()
                peaks2['abs_dist'] = (peaks2['peak_day'].astype(int) - int(cand['point_day'])).abs()
                row = peaks2.sort_values(['abs_dist','peak_score','peak_prominence'], ascending=[True,False,False]).iloc[0]
                day_offset = int(row['peak_day']) - int(cand['point_day'])
                param_rows.append({'width': width, 'pen': pen, 'point_id': str(cand['point_id']), 'present_flag': bool(abs(day_offset) <= tolerance), 'nearest_peak_day': int(row['peak_day']), 'nearest_peak_score': float(row['peak_score']), 'day_offset': day_offset})
    param_df = pd.DataFrame(param_rows)

    year_rows = []
    years_arr = np.asarray(years).astype(int)
    grouped = yearwise_primary_points_df.groupby('year') if not yearwise_primary_points_df.empty else []
    yearly_dict = {int(year): df.copy() for year, df in grouped} if not yearwise_primary_points_df.empty else {}
    for year in years_arr.tolist():
        peaks = yearly_dict.get(int(year), pd.DataFrame())
        for _, cand in point_candidates_df.iterrows():
            if peaks.empty:
                year_rows.append({'year': int(year), 'point_id': str(cand['point_id']), 'formal_point_day': int(cand['point_day']), 'nearest_yearwise_peak_day': np.nan, 'nearest_yearwise_peak_score': np.nan, 'best_match_peak_rank_within_year': np.nan, 'day_offset': np.nan, 'best_match_exists': False, 'near_point_flag': False, 'strict_point_flag': False})
                continue
            peaks2 = peaks.copy()
            peaks2['abs_dist'] = (peaks2['mapped_peak_day'].astype(int) - int(cand['point_day'])).abs()
            row = peaks2.sort_values(['abs_dist','peak_score'], ascending=[True,False]).iloc[0]
            day_offset = int(row['mapped_peak_day']) - int(cand['point_day'])
            year_rows.append({'year': int(year), 'point_id': str(cand['point_id']), 'formal_point_day': int(cand['point_day']), 'nearest_yearwise_peak_day': int(row['mapped_peak_day']), 'nearest_yearwise_peak_score': float(row['peak_score']), 'best_match_peak_rank_within_year': np.nan, 'day_offset': day_offset, 'best_match_exists': True, 'near_point_flag': bool(abs(day_offset) <= settings.yearwise_support.near_peak_tolerance_days), 'strict_point_flag': bool(abs(day_offset) <= settings.point_significance.local_window_days)})
    yearwise_df = pd.DataFrame(year_rows)

    rows = []
    for _, cand in point_candidates_df.iterrows():
        pid = str(cand['point_id'])
        boot_sub = bootstrap_df[bootstrap_df['point_id'] == pid]
        param_sub = param_df[param_df['point_id'] == pid]
        year_sub = yearwise_df[yearwise_df['point_id'] == pid]
        boot_frac = float(boot_sub['present_flag'].mean()) if not boot_sub.empty else np.nan
        param_frac = float(param_sub['present_flag'].mean()) if not param_sub.empty else np.nan
        year_near = float(year_sub['near_point_flag'].mean()) if not year_sub.empty else np.nan
        year_strict = float(year_sub['strict_point_flag'].mean()) if not year_sub.empty else np.nan
        med_score = float(year_sub['nearest_yearwise_peak_score'].median()) if not year_sub.empty else np.nan
        warning = bool((np.isfinite(boot_frac) and boot_frac < 0.2) or (np.isfinite(param_frac) and param_frac < 0.2))
        rows.append({'point_id': pid, 'bootstrap_presence_fraction': boot_frac, 'param_presence_fraction': param_frac, 'yearwise_near_fraction': year_near, 'yearwise_strict_fraction': year_strict, 'median_yearwise_peak_score': med_score, 'stability_warning': warning})
    return pd.DataFrame(rows, columns=cols), bootstrap_df, param_df, yearwise_df


def build_neighbor_pairs(point_candidates_df: pd.DataFrame, settings: StagePartitionV3Settings) -> pd.DataFrame:
    cols = ['pair_id','point_a_id','point_b_id','point_a_day','point_b_day','distance_days']
    if point_candidates_df.empty:
        return pd.DataFrame(columns=cols)
    radius = int(settings.point_significance.neighbor_competition_radius_days)
    rows = []
    pair_idx = 1
    pts = point_candidates_df[['point_id', 'point_day']].copy().sort_values('point_day')
    vals = pts.to_dict('records')
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            a = vals[i]
            b = vals[j]
            dist = abs(int(a['point_day']) - int(b['point_day']))
            if dist <= radius:
                rows.append({'pair_id': f'PAIR{pair_idx:03d}', 'point_a_id': str(a['point_id']), 'point_b_id': str(b['point_id']), 'point_a_day': int(a['point_day']), 'point_b_day': int(b['point_day']), 'distance_days': int(dist)})
                pair_idx += 1
    return pd.DataFrame(rows, columns=cols)


def run_point_neighbor_competition(point_candidates_df: pd.DataFrame, pair_df: pd.DataFrame, bootstrap_df: pd.DataFrame, yearwise_rel_df: pd.DataFrame, settings: StagePartitionV3Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ['pair_id','point_a_id','point_b_id','point_a_score','point_b_score','point_a_bootstrap_win_fraction','point_b_bootstrap_win_fraction','point_a_yearwise_win_fraction','point_b_yearwise_win_fraction','competition_outcome','competition_reason_primary','competition_reason_secondary']
    if pair_df.empty:
        point_status = point_candidates_df[['point_id']].copy()
        point_status['neighbor_competition_status'] = 'no_neighbor'
        return pd.DataFrame(columns=cols), point_status
    score_tol = float(settings.point_significance.competition_tie_tolerance_score)
    boot_tol = float(getattr(settings.point_significance, 'competition_tie_tolerance_bootstrap', settings.point_significance.competition_tie_tolerance_winfrac))
    year_tol = float(getattr(settings.point_significance, 'competition_tie_tolerance_yearwise', settings.point_significance.competition_tie_tolerance_winfrac))
    score_map = point_candidates_df.set_index('point_id')['observed_peak_score'].to_dict()
    rows = []
    for _, pair in pair_df.iterrows():
        a_id = str(pair['point_a_id'])
        b_id = str(pair['point_b_id'])
        a_score = float(score_map.get(a_id, np.nan))
        b_score = float(score_map.get(b_id, np.nan))
        if not bootstrap_df.empty:
            wide = bootstrap_df[bootstrap_df['point_id'].isin([a_id, b_id])].pivot(index='rep', columns='point_id', values='nearest_peak_score')
            if a_id in wide.columns and b_id in wide.columns and not wide.empty:
                valid = wide[[a_id, b_id]].dropna(how='all')
                a_boot = float((valid[a_id].fillna(-np.inf) > valid[b_id].fillna(-np.inf)).mean()) if not valid.empty else np.nan
                b_boot = float((valid[b_id].fillna(-np.inf) > valid[a_id].fillna(-np.inf)).mean()) if not valid.empty else np.nan
            else:
                a_boot = np.nan; b_boot = np.nan
        else:
            a_boot = np.nan; b_boot = np.nan
        if not yearwise_rel_df.empty:
            wide = yearwise_rel_df[yearwise_rel_df['point_id'].isin([a_id, b_id])].pivot(index='year', columns='point_id', values='nearest_yearwise_peak_score')
            if a_id in wide.columns and b_id in wide.columns and not wide.empty:
                valid = wide[[a_id, b_id]].dropna(how='all')
                a_year = float((valid[a_id].fillna(-np.inf) > valid[b_id].fillna(-np.inf)).mean()) if not valid.empty else np.nan
                b_year = float((valid[b_id].fillna(-np.inf) > valid[a_id].fillna(-np.inf)).mean()) if not valid.empty else np.nan
            else:
                a_year = np.nan; b_year = np.nan
        else:
            a_year = np.nan; b_year = np.nan

        score_close = np.isfinite(a_score) and np.isfinite(b_score) and abs(a_score - b_score) <= score_tol * max(a_score, b_score, 1.0)
        boot_close = np.isfinite(a_boot) and np.isfinite(b_boot) and abs(a_boot - b_boot) <= boot_tol
        year_close = np.isfinite(a_year) and np.isfinite(b_year) and abs(a_year - b_year) <= year_tol

        if score_close and (not np.isfinite(a_boot) or boot_close) and (not np.isfinite(a_year) or year_close):
            outcome = 'tie'
            reason_primary = 'near_equal_score_and_winfractions'
            reason_secondary = 'score_close;bootstrap_close;yearwise_close'
        elif np.isfinite(a_score) and np.isfinite(b_score) and a_score > b_score:
            outcome = 'point_a_wins'
            reason_primary = 'higher_observed_peak_score'
            reason_secondary = 'bootstrap_supports' if np.isfinite(a_boot) and np.isfinite(b_boot) and a_boot > b_boot else ('yearwise_supports' if np.isfinite(a_year) and np.isfinite(b_year) and a_year > b_year else 'score_primary')
        elif np.isfinite(a_score) and np.isfinite(b_score) and b_score > a_score:
            outcome = 'point_b_wins'
            reason_primary = 'higher_observed_peak_score'
            reason_secondary = 'bootstrap_supports' if np.isfinite(a_boot) and np.isfinite(b_boot) and b_boot > a_boot else ('yearwise_supports' if np.isfinite(a_year) and np.isfinite(b_year) and b_year > a_year else 'score_primary')
        else:
            outcome = 'tie'
            reason_primary = 'insufficient_information'
            reason_secondary = ''
        rows.append({'pair_id': str(pair['pair_id']), 'point_a_id': a_id, 'point_b_id': b_id, 'point_a_score': a_score, 'point_b_score': b_score, 'point_a_bootstrap_win_fraction': a_boot, 'point_b_bootstrap_win_fraction': b_boot, 'point_a_yearwise_win_fraction': a_year, 'point_b_yearwise_win_fraction': b_year, 'competition_outcome': outcome, 'competition_reason_primary': reason_primary, 'competition_reason_secondary': reason_secondary})
    comp_df = pd.DataFrame(rows, columns=cols)

    status_map = {pid: [] for pid in point_candidates_df['point_id'].astype(str).tolist()}
    for _, row in comp_df.iterrows():
        if row['competition_outcome'] == 'point_a_wins':
            status_map[row['point_a_id']].append('winner')
            status_map[row['point_b_id']].append('loser')
        elif row['competition_outcome'] == 'point_b_wins':
            status_map[row['point_a_id']].append('loser')
            status_map[row['point_b_id']].append('winner')
        else:
            status_map[row['point_a_id']].append('tie')
            status_map[row['point_b_id']].append('tie')
    point_status_rows = []
    for pid in point_candidates_df['point_id'].astype(str).tolist():
        flags = status_map.get(pid, [])
        if not flags:
            status = 'no_neighbor'
        elif 'loser' in flags:
            status = 'neighbor_competition_loser'
        elif 'winner' in flags:
            status = 'neighbor_competition_winner'
        elif 'tie' in flags:
            status = 'neighbor_competition_tie'
        else:
            status = 'no_neighbor'
        point_status_rows.append({'point_id': pid, 'neighbor_competition_status': status})
    return comp_df, pd.DataFrame(point_status_rows)


def summarize_point_significance(point_candidates_df: pd.DataFrame, null_df: pd.DataFrame, scale_df: pd.DataFrame, stability_df: pd.DataFrame, point_status_df: pd.DataFrame, settings: StagePartitionV3Settings) -> pd.DataFrame:
    cols = ['point_id','point_day','point_role','related_window_id','observed_peak_score','observed_matched_stat','p_global_fwer','p_local','p_match_exist','p_match_exceed_conditional','local_joint_challenge_score','bootstrap_presence_fraction','param_presence_fraction','yearwise_near_fraction','yearwise_strict_fraction','neighbor_competition_status','point_significance_tier','strict_local_interpretation','notes']
    if point_candidates_df.empty:
        return pd.DataFrame(columns=cols)
    out = point_candidates_df.copy()
    merge_cols = ['point_id','point_day','observed_peak_score']
    out = out.merge(null_df, on=merge_cols, how='left')
    out = out.merge(scale_df[['point_id','scale_warning_flag','matched_stat_ratio','matched_peak_exists_fraction','local_test_mode']], on='point_id', how='left')
    out = out.merge(stability_df, on='point_id', how='left')
    out = out.merge(point_status_df, on='point_id', how='left')
    tiers, interps, notes = [], [], []
    for _, row in out.iterrows():
        comp_status = row.get('neighbor_competition_status', 'not_applicable')
        p_global = row.get('p_global_fwer')
        p_match_exist = row.get('p_match_exist')
        p_match_cond = row.get('p_match_exceed_conditional')
        ratio = row.get('matched_stat_ratio')
        scale_warn = bool(row.get('scale_warning_flag')) if pd.notna(row.get('scale_warning_flag')) else False
        boot = row.get('bootstrap_presence_fraction')
        param = row.get('param_presence_fraction')
        year_near = row.get('yearwise_near_fraction')
        note_parts = []
        if comp_status == 'neighbor_competition_loser':
            tier = 'strict_weak_or_unresolved'
            interp = 'competition_loser'
            note_parts.append('loses_closed_neighbor_competition')
        elif pd.notna(p_match_cond) and float(p_match_cond) <= settings.point_significance.local_alpha and pd.notna(ratio) and float(ratio) >= 1.0 and not scale_warn:
            if pd.notna(p_match_exist) and float(p_match_exist) <= settings.point_significance.local_exist_alpha:
                interp = 'rare_matched_peak_and_observed_strong'
            else:
                interp = 'common_matched_peak_but_observed_still_strong'
            tier = 'strict_local_significant'
            note_parts.append('passes_matched_local_test')
        elif pd.notna(p_global) and float(p_global) <= settings.point_significance.global_alpha and not scale_warn:
            tier = 'strict_core_significant'
            interp = 'globally_rare_peak_and_strong'
            note_parts.append('passes_global_null_test')
        elif pd.notna(boot) and float(boot) >= 0.35 and pd.notna(param) and float(param) >= 0.35 and pd.notna(year_near) and float(year_near) >= max(0.25, settings.point_significance.yearwise_near_min - 0.10):
            tier = 'strict_borderline'
            if pd.notna(p_match_exist) and float(p_match_exist) <= settings.point_significance.local_exist_alpha and pd.notna(p_match_cond):
                interp = 'rare_matched_peak_and_observed_not_outstanding'
            elif pd.notna(p_match_cond):
                interp = 'common_matched_peak_and_observed_not_strong'
            else:
                interp = 'insufficient_matched_local_evidence'
        else:
            tier = 'strict_weak_or_unresolved'
            if pd.notna(p_match_cond):
                interp = 'common_matched_peak_and_observed_not_strong'
            else:
                interp = 'insufficient_matched_local_evidence'
        if row.get('point_role') == 'neighbor_local_peak':
            note_parts.append('neighbor_peak_candidate')
        if comp_status == 'neighbor_competition_tie':
            note_parts.append('neighbor_competition_tie')
        if scale_warn:
            note_parts.append('matched_scale_warning')
        tiers.append(tier)
        interps.append(interp)
        notes.append(';'.join(note_parts))
    out['neighbor_competition_status'] = out['neighbor_competition_status'].fillna('not_applicable')
    out['point_significance_tier'] = tiers
    out['strict_local_interpretation'] = interps
    out['notes'] = notes
    return out[cols]

def build_point_significance_summary(audit_df: pd.DataFrame, scale_df: pd.DataFrame | None = None, competition_df: pd.DataFrame | None = None) -> dict:
    if audit_df is None or audit_df.empty:
        return {'n_points_audited': 0}
    strong_cond = audit_df[audit_df['point_day'].isin([81,113,160])]['p_match_exceed_conditional'].dropna().to_numpy(dtype=float)
    weak_cond = audit_df[audit_df['point_day'].isin([132,135])]['p_match_exceed_conditional'].dropna().to_numpy(dtype=float)
    strong_ratio = audit_df[audit_df['point_day'].isin([81,113,160])]['observed_matched_stat'].dropna().to_numpy(dtype=float)
    weak_ratio = audit_df[audit_df['point_day'].isin([132,135])]['observed_matched_stat'].dropna().to_numpy(dtype=float)
    gradient_present_matched = bool(strong_cond.size and weak_cond.size and np.nanmedian(strong_cond) < np.nanmedian(weak_cond) and strong_ratio.size and weak_ratio.size and np.nanmedian(strong_ratio) > np.nanmedian(weak_ratio))
    summary = {
        'n_points_audited': int(len(audit_df)),
        'n_strict_core_significant': int((audit_df['point_significance_tier'] == 'strict_core_significant').sum()),
        'n_strict_local_significant': int((audit_df['point_significance_tier'] == 'strict_local_significant').sum()),
        'n_strict_borderline': int((audit_df['point_significance_tier'] == 'strict_borderline').sum()),
        'n_strict_weak_or_unresolved': int((audit_df['point_significance_tier'] == 'strict_weak_or_unresolved').sum()),
        'n_significant_points_global': int((audit_df['p_global_fwer'] <= 0.05).fillna(False).sum()),
        'n_significant_points_local': int((audit_df['p_match_exceed_conditional'] <= 0.10).fillna(False).sum()),
        'n_points_match_exist_frequent': int((audit_df['p_match_exist'] >= 0.10).fillna(False).sum()),
        'n_points_match_exceed_supportive': int((audit_df['p_match_exceed_conditional'] <= 0.10).fillna(False).sum()),
        'strong_vs_weak_gradient_present': gradient_present_matched,
        'strong_vs_weak_gradient_present_conditional': gradient_present_matched,
        'strong_vs_weak_gradient_present_matched': gradient_present_matched,
        'local_match_by_point': {
            str(row['point_id']): {
                'point_day': int(row['point_day']),
                'p_match_exist': float(row['p_match_exist']) if pd.notna(row['p_match_exist']) else None,
                'p_match_exceed_conditional': float(row['p_match_exceed_conditional']) if pd.notna(row['p_match_exceed_conditional']) else None,
                'local_joint_challenge_score': float(row['local_joint_challenge_score']) if pd.notna(row['local_joint_challenge_score']) else None,
                'observed_matched_stat': float(row['observed_matched_stat']) if pd.notna(row['observed_matched_stat']) else None,
                'strict_local_interpretation': row.get('strict_local_interpretation'),
            }
            for _, row in audit_df.iterrows()
        },
    }
    if scale_df is not None and not scale_df.empty:
        summary['point_null_scale_warning_count'] = int(scale_df['scale_warning_flag'].fillna(False).sum())
    if competition_df is not None and not competition_df.empty:
        summary['neighbor_competition_closed_pairs'] = int(len(competition_df))
        summary['neighbor_competition_ties'] = int((competition_df['competition_outcome'] == 'tie').sum())
    return summary
