from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd

from .config import StagePartitionV3Settings
from .profiles import ObjectProfile
from .safe_stats import safe_nanmean
from .detector_ruptures_window import run_ruptures_window
from .window_builder import build_primary_windows_from_ruptures


def compute_window_overlap(ref_start: int, ref_end: int, cand_start: int, cand_end: int) -> tuple[int, int, float]:
    overlap_days = max(0, min(ref_end, cand_end) - max(ref_start, cand_start) + 1)
    union_days = max(ref_end, cand_end) - min(ref_start, cand_start) + 1
    overlap_ratio = overlap_days / float(max(1, union_days))
    return overlap_days, union_days, overlap_ratio


def _legacy_match_reference_window(ref_row: pd.Series, cand_windows_df: pd.DataFrame, settings: StagePartitionV3Settings) -> dict:
    if cand_windows_df.empty:
        return {'legacy_match_flag': False, 'legacy_matched_window_id': None, 'legacy_matched_center_day': None, 'legacy_overlap_ratio': 0.0}
    ref_start = int(ref_row['start_day'])
    ref_end = int(ref_row['end_day'])
    ref_center = int(ref_row['center_day'])
    best = None
    best_ratio = -1.0
    for _, cand in cand_windows_df.iterrows():
        cand_start = int(cand['start_day'])
        cand_end = int(cand['end_day'])
        cand_center = int(cand['center_day'])
        _, _, ratio = compute_window_overlap(ref_start, ref_end, cand_start, cand_end)
        close = abs(cand_center - ref_center) <= settings.support.center_tolerance_days
        if ratio >= settings.support.overlap_ratio_threshold or close:
            if ratio > best_ratio:
                best_ratio = ratio
                best = cand
    if best is None:
        return {'legacy_match_flag': False, 'legacy_matched_window_id': None, 'legacy_matched_center_day': None, 'legacy_overlap_ratio': 0.0}
    return {
        'legacy_match_flag': True,
        'legacy_matched_window_id': str(best['window_id']),
        'legacy_matched_center_day': int(best['center_day']),
        'legacy_overlap_ratio': float(max(best_ratio, 0.0)),
    }


def _strict_match_reference_window(ref_row: pd.Series, cand_windows_df: pd.DataFrame, settings: StagePartitionV3Settings) -> dict:
    if cand_windows_df.empty:
        return {
            'strict_match_flag': False,
            'strict_matched_window_id': None,
            'strict_matched_center_day': None,
            'overlap_days': 0,
            'union_days': 0,
            'overlap_ratio': 0.0,
            'center_close_flag': False,
            'match_decision_source': 'no_candidate_windows',
        }
    ref_start = int(ref_row['start_day'])
    ref_end = int(ref_row['end_day'])
    ref_center = int(ref_row['center_day'])
    best = None
    best_key = None
    for _, cand in cand_windows_df.iterrows():
        cand_start = int(cand['start_day'])
        cand_end = int(cand['end_day'])
        cand_center = int(cand['center_day'])
        overlap_days, union_days, overlap_ratio = compute_window_overlap(ref_start, ref_end, cand_start, cand_end)
        center_close = abs(cand_center - ref_center) <= settings.support.center_tolerance_days
        strict_match = overlap_days >= settings.support.support_overlap_days_min and overlap_ratio >= settings.support.support_overlap_ratio_min
        key = (1 if strict_match else 0, overlap_ratio, overlap_days, -abs(cand_center - ref_center))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                'strict_match_flag': bool(strict_match),
                'strict_matched_window_id': str(cand['window_id']),
                'strict_matched_center_day': cand_center,
                'overlap_days': int(overlap_days),
                'union_days': int(union_days),
                'overlap_ratio': float(overlap_ratio),
                'center_close_flag': bool(center_close),
                'match_decision_source': 'strict_overlap' if strict_match else ('center_close_but_not_match' if center_close else 'best_nonmatch_candidate'),
            }
    return best


def _build_bootstrap_state_from_profiles(profile_dict: dict[str, ObjectProfile], sample_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    feature_widths = []
    for name in ['P', 'V', 'H', 'Je', 'Jw']:
        seasonal, _ = safe_nanmean(profile_dict[name].raw_cube[sample_idx, :, :], axis=0, return_valid_count=True)
        blocks.append(seasonal)
        feature_widths.append(seasonal.shape[1])
    raw = np.concatenate(blocks, axis=1)
    mean, _ = safe_nanmean(raw, axis=0, return_valid_count=True)
    centered = raw - mean[None, :]
    var, _ = safe_nanmean(np.square(centered), axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    state = centered / std[None, :]
    start = 0
    for width in feature_widths:
        state[:, start:start + width] *= 1.0 / np.sqrt(width)
        start += width
    valid_day_mask = np.all(np.isfinite(state), axis=1)
    return state, valid_day_mask


def run_bootstrap_window_matches(profile_dict: dict[str, ObjectProfile], main_windows_df: pd.DataFrame, settings: StagePartitionV3Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(settings.support.random_seed)
    n_years = next(iter(profile_dict.values())).raw_cube.shape[0]
    years_idx = np.arange(n_years)
    match_rows = []
    validity_rows = []
    min_required = max(2 * int(settings.ruptures_window.width), 3)
    for rep in range(settings.support.bootstrap_reps):
        sample_idx = rng.choice(years_idx, size=n_years, replace=True)
        state, valid_day_mask = _build_bootstrap_state_from_profiles(profile_dict, sample_idx)
        valid_day_index = np.where(valid_day_mask)[0].astype(int)
        validity_rows.append({
            'rep': rep,
            'n_total_days': int(state.shape[0]),
            'n_valid_days': int(valid_day_index.size),
            'n_invalid_days': int(state.shape[0] - valid_day_index.size),
            'invalid_fraction': float((state.shape[0] - valid_day_index.size) / state.shape[0]),
            'sample_valid_flag': bool(valid_day_index.size >= min_required),
            'skip_reason': '' if valid_day_index.size >= min_required else 'insufficient_valid_days',
        })
        if valid_day_index.size < min_required:
            for _, ref in main_windows_df.iterrows():
                match_rows.append({
                    'rep': rep,
                    'window_id': ref['window_id'],
                    'legacy_match_flag': False,
                    'legacy_matched_window_id': None,
                    'legacy_matched_center_day': None,
                    'legacy_overlap_ratio': 0.0,
                    'strict_match_flag': False,
                    'strict_matched_window_id': None,
                    'strict_matched_center_day': None,
                    'overlap_days': 0,
                    'union_days': 0,
                    'overlap_ratio': 0.0,
                    'center_close_flag': False,
                    'match_decision_source': 'invalid_bootstrap_sample',
                })
            continue
        state_valid = state[valid_day_mask, :]
        out = run_ruptures_window(state_valid, settings.ruptures_window, day_index=valid_day_index)
        rep_windows, _, _, _ = build_primary_windows_from_ruptures(out['points'], out['profile'], settings.edge, settings.window_construction)
        for _, ref in main_windows_df.iterrows():
            legacy = _legacy_match_reference_window(ref, rep_windows, settings)
            strict = _strict_match_reference_window(ref, rep_windows, settings)
            match_rows.append({'rep': rep, 'window_id': ref['window_id'], **legacy, **strict})
    return pd.DataFrame(match_rows), pd.DataFrame(validity_rows)


def run_param_path_window_matches(state_matrix: np.ndarray, valid_day_mask: np.ndarray, main_windows_df: pd.DataFrame, settings: StagePartitionV3Settings) -> pd.DataFrame:
    state_valid = state_matrix[valid_day_mask, :]
    day_index = np.where(valid_day_mask)[0].astype(int)
    rows = []
    for width in settings.support.ruptures_width_grid:
        for pen in settings.support.ruptures_pen_grid:
            cfg_local = replace(settings.ruptures_window, width=int(width), selection_mode='pen', pen=float(pen))
            out = run_ruptures_window(state_valid, cfg_local, day_index=day_index)
            cand_windows, _, _, _ = build_primary_windows_from_ruptures(out['points'], out['profile'], settings.edge, settings.window_construction)
            for _, ref in main_windows_df.iterrows():
                legacy = _legacy_match_reference_window(ref, cand_windows, settings)
                strict = _strict_match_reference_window(ref, cand_windows, settings)
                rows.append({'window_id': ref['window_id'], 'width': int(width), 'pen': float(pen), **legacy, **strict})
    return pd.DataFrame(rows)


def run_global_permutation_audit(state_matrix: np.ndarray, valid_day_mask: np.ndarray, settings: StagePartitionV3Settings) -> pd.DataFrame:
    state_valid = state_matrix[valid_day_mask, :]
    day_index = np.where(valid_day_mask)[0].astype(int)
    rng = np.random.default_rng(settings.support.random_seed)
    reps = settings.support.permutation_reps
    out = run_ruptures_window(state_valid, settings.ruptures_window, day_index=day_index)
    observed = float(np.nanmax(out['profile'].to_numpy(dtype=float))) if not out['profile'].empty else np.nan
    null_vals = []
    for _ in range(reps):
        perm = rng.permutation(state_valid.shape[0])
        perm_state = state_valid[perm, :]
        perm_days = day_index[perm]
        perm_out = run_ruptures_window(perm_state, settings.ruptures_window, day_index=perm_days)
        null_vals.append(float(np.nanmax(perm_out['profile'].to_numpy(dtype=float))) if not perm_out['profile'].empty else np.nan)
    null_arr = np.asarray(null_vals, dtype=float)
    pvalue = float((np.sum(null_arr >= observed) + 1) / (np.sum(np.isfinite(null_arr)) + 1)) if np.isfinite(observed) else np.nan
    return pd.DataFrame([{'backend': 'ruptures_window', 'observed_max_score': observed, 'empirical_pvalue': pvalue, 'permutation_reps': int(reps)}])


def summarize_window_support(main_windows_df: pd.DataFrame, bootstrap_matches_df: pd.DataFrame, validity_df: pd.DataFrame, param_df: pd.DataFrame, perm_df: pd.DataFrame, settings: StagePartitionV3Settings) -> pd.DataFrame:
    cols = [
        'window_id', 'n_total_samples', 'n_valid_samples', 'n_invalid_samples', 'invalid_ratio', 'sample_scope',
        'n_bootstrap_requested', 'n_bootstrap_effective', 'support_score', 'support_score_source',
        'legacy_bootstrap_match_fraction', 'strict_bootstrap_match_fraction', 'legacy_param_path_hit_fraction',
        'param_path_hits', 'param_path_trials', 'param_path_hit_fraction', 'permutation_empirical_pvalue_global',
        'support_status', 'support_reliability_flag', 'support_warning'
    ]
    if main_windows_df.empty:
        return pd.DataFrame(columns=cols)
    n_total = int(len(validity_df))
    n_valid = int(validity_df['sample_valid_flag'].sum()) if not validity_df.empty else 0
    n_invalid = n_total - n_valid
    invalid_ratio = float(n_invalid / n_total) if n_total > 0 else np.nan
    global_perm = float(perm_df.iloc[0]['empirical_pvalue']) if not perm_df.empty else np.nan
    rows = []
    for _, win in main_windows_df.iterrows():
        win_id = str(win['window_id'])
        win_boot = bootstrap_matches_df[bootstrap_matches_df['window_id'] == win_id].copy() if not bootstrap_matches_df.empty else pd.DataFrame()
        legacy_boot_hits = int(win_boot['legacy_match_flag'].sum()) if not win_boot.empty else 0
        strict_boot_hits = int(win_boot['strict_match_flag'].sum()) if not win_boot.empty else 0
        legacy_boot_frac = float(legacy_boot_hits / n_valid) if n_valid > 0 else np.nan
        strict_boot_frac = float(strict_boot_hits / n_valid) if n_valid > 0 else np.nan
        param_rows = param_df[param_df['window_id'] == win_id].copy() if not param_df.empty else pd.DataFrame()
        legacy_param_hits = int(param_rows['legacy_match_flag'].sum()) if not param_rows.empty else 0
        strict_param_hits = int(param_rows['strict_match_flag'].sum()) if not param_rows.empty else 0
        param_trials = int(len(param_rows)) if not param_rows.empty else 0
        legacy_param_frac = float(legacy_param_hits / param_trials) if param_trials > 0 else np.nan
        strict_param_frac = float(strict_param_hits / param_trials) if param_trials > 0 else np.nan
        support_score = strict_boot_frac
        warnings = []
        if n_valid == 0:
            warnings.append('no_valid_bootstrap_samples')
        if np.isfinite(legacy_boot_frac) and np.isfinite(strict_boot_frac) and strict_boot_frac < legacy_boot_frac:
            warnings.append('strict_support_below_legacy')
        if param_trials == 0:
            warnings.append('no_param_trials')
        if np.isfinite(strict_boot_frac) and strict_boot_frac < settings.retention.min_support_score:
            warnings.append('strict_support_below_retention_floor')
        reliability = 'high'
        if n_valid < settings.retention.min_bootstrap_effective or (np.isfinite(strict_param_frac) and strict_param_frac < settings.retention.min_param_path_hit_fraction):
            reliability = 'limited'
        if n_valid == 0:
            reliability = 'low'
        rows.append({
            'window_id': win_id,
            'n_total_samples': n_total,
            'n_valid_samples': n_valid,
            'n_invalid_samples': n_invalid,
            'invalid_ratio': invalid_ratio,
            'sample_scope': 'bootstrap_rep_validity',
            'n_bootstrap_requested': int(len(validity_df)),
            'n_bootstrap_effective': n_valid,
            'support_score': support_score,
            'support_score_source': 'strict_bootstrap_match_fraction',
            'legacy_bootstrap_match_fraction': legacy_boot_frac,
            'strict_bootstrap_match_fraction': strict_boot_frac,
            'legacy_param_path_hit_fraction': legacy_param_frac,
            'param_path_hits': strict_param_hits,
            'param_path_trials': param_trials,
            'param_path_hit_fraction': strict_param_frac,
            'permutation_empirical_pvalue_global': global_perm,
            'support_status': 'resolved_window_level',
            'support_reliability_flag': reliability,
            'support_warning': ';'.join(warnings),
        })
    return pd.DataFrame(rows, columns=cols)


def build_support_rule_comparison(support_audit_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'window_id', 'legacy_bootstrap_match_fraction', 'strict_bootstrap_match_fraction',
        'legacy_param_hit_fraction', 'strict_param_hit_fraction',
        'support_score_legacy', 'support_score_strict', 'delta_support_score', 'support_rule_change_flag'
    ]
    if support_audit_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, row in support_audit_df.iterrows():
        legacy_boot = float(row['legacy_bootstrap_match_fraction']) if pd.notna(row['legacy_bootstrap_match_fraction']) else np.nan
        strict_boot = float(row['strict_bootstrap_match_fraction']) if pd.notna(row['strict_bootstrap_match_fraction']) else np.nan
        legacy_param = float(row['legacy_param_path_hit_fraction']) if pd.notna(row['legacy_param_path_hit_fraction']) else np.nan
        strict_param = float(row['param_path_hit_fraction']) if pd.notna(row['param_path_hit_fraction']) else np.nan
        delta = strict_boot - legacy_boot if np.isfinite(strict_boot) and np.isfinite(legacy_boot) else np.nan
        rows.append({
            'window_id': str(row['window_id']),
            'legacy_bootstrap_match_fraction': legacy_boot,
            'strict_bootstrap_match_fraction': strict_boot,
            'legacy_param_hit_fraction': legacy_param,
            'strict_param_hit_fraction': strict_param,
            'support_score_legacy': legacy_boot,
            'support_score_strict': strict_boot,
            'delta_support_score': delta,
            'support_rule_change_flag': bool(np.isfinite(delta) and abs(delta) > 1e-12),
        })
    return pd.DataFrame(rows, columns=cols)
