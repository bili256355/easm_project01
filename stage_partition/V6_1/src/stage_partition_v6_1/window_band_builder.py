
from __future__ import annotations
import numpy as np
import pandas as pd


def _nearest_profile_index(profile_index: np.ndarray, day: int) -> int:
    return int(np.argmin(np.abs(profile_index - int(day))))


def _compute_support_floor(peak_score: float, peak_prominence: float, profile_values: np.ndarray, cfg) -> tuple[float, float]:
    valid = profile_values[np.isfinite(profile_values)]
    global_floor = float(np.quantile(valid, float(cfg.peak_floor_quantile))) if valid.size else 0.0
    if np.isfinite(peak_score) and np.isfinite(peak_prominence):
        relative_floor = float(peak_score - float(cfg.prominence_ratio_threshold) * peak_prominence)
    elif np.isfinite(peak_score):
        relative_floor = float(peak_score)
    else:
        relative_floor = global_floor
    return max(global_floor, relative_floor), global_floor


def _find_local_valley_between(days: np.ndarray, vals: np.ndarray, left_day: int, right_day: int) -> tuple[int | None, float | None]:
    left_idx = _nearest_profile_index(days, left_day)
    right_idx = _nearest_profile_index(days, right_day)
    if left_idx == right_idx:
        return int(days[left_idx]), float(vals[left_idx])
    i0, i1 = sorted((left_idx, right_idx))
    seg = vals[i0:i1 + 1]
    if seg.size <= 2 or not np.isfinite(seg).any():
        return None, None
    rel = int(np.nanargmin(seg))
    idx = i0 + rel
    return int(days[idx]), float(vals[idx])


def build_candidate_point_bands(registry_df: pd.DataFrame, profile: pd.Series, cfg) -> pd.DataFrame:
    cols = [
        'candidate_id', 'point_day', 'month_day',
        'band_start_day', 'band_end_day', 'band_center_day',
        'peak_score', 'peak_prominence', 'support_floor', 'global_floor',
        'left_stop_reason', 'right_stop_reason', 'is_formal_primary'
    ]
    if registry_df is None or registry_df.empty:
        return pd.DataFrame(columns=cols)

    if profile is None or profile.empty:
        out = registry_df.copy()
        out['band_start_day'] = out['point_day'].astype(int) - int(cfg.min_band_half_width_days)
        out['band_end_day'] = out['point_day'].astype(int) + int(cfg.min_band_half_width_days)
        out['band_center_day'] = out['point_day'].astype(int)
        out['support_floor'] = np.nan
        out['global_floor'] = np.nan
        out['left_stop_reason'] = 'min_half_width_only'
        out['right_stop_reason'] = 'min_half_width_only'
        return out[cols]

    prof = profile.sort_index().astype(float)
    days = prof.index.to_numpy(dtype=int)
    vals = prof.to_numpy(dtype=float)
    registry = registry_df.sort_values('point_day').reset_index(drop=True)
    registry_days = registry['point_day'].astype(int).tolist()
    min_half = int(cfg.min_band_half_width_days)
    max_half = int(cfg.max_band_half_width_days)
    near_exempt = int(cfg.close_neighbor_exemption_days)

    rows = []
    for _, row in registry.iterrows():
        day = int(row['point_day'])
        i0 = _nearest_profile_index(days, day)
        peak_score = float(row['peak_score']) if pd.notna(row.get('peak_score')) else float(vals[i0])
        peak_prom = float(row['peak_prominence']) if pd.notna(row.get('peak_prominence')) else np.nan
        support_floor, global_floor = _compute_support_floor(peak_score, peak_prom, vals, cfg)

        left = i0
        left_stop_reason = 'support_floor'
        while left > 0:
            candidate_next_day = int(days[left - 1])
            if abs(day - candidate_next_day) > max_half:
                left_stop_reason = 'max_half_width'
                break
            next_val = vals[left - 1]
            if not np.isfinite(next_val) or next_val < support_floor:
                left_stop_reason = 'support_floor'
                break
            left = left - 1

        right = i0
        right_stop_reason = 'support_floor'
        while right < len(vals) - 1:
            candidate_next_day = int(days[right + 1])
            if abs(candidate_next_day - day) > max_half:
                right_stop_reason = 'max_half_width'
                break
            next_val = vals[right + 1]
            if not np.isfinite(next_val) or next_val < support_floor:
                right_stop_reason = 'support_floor'
                break
            right = right + 1

        start_day = int(min(days[left], day - min_half))
        end_day = int(max(days[right], day + min_half))

        if bool(cfg.respect_candidate_boundaries):
            other_days = [d for d in registry_days if d != day]
            left_candidates = [d for d in other_days if d < day and (day - d) > near_exempt]
            right_candidates = [d for d in other_days if d > day and (d - day) > near_exempt]

            if left_candidates and bool(cfg.truncate_at_intervening_candidate):
                nearest_left = max(left_candidates)
                if start_day <= nearest_left:
                    stop_day = nearest_left + 1
                    reason = 'intervening_candidate'
                    if bool(cfg.truncate_at_local_valley):
                        valley_day, valley_val = _find_local_valley_between(days, vals, nearest_left, day)
                        if valley_day is not None and np.isfinite(valley_val) and valley_val < peak_score:
                            stop_day = max(nearest_left + 1, valley_day)
                            reason = 'local_valley'
                    start_day = max(start_day, stop_day)
                    left_stop_reason = reason

            if right_candidates and bool(cfg.truncate_at_intervening_candidate):
                nearest_right = min(right_candidates)
                if end_day >= nearest_right:
                    stop_day = nearest_right - 1
                    reason = 'intervening_candidate'
                    if bool(cfg.truncate_at_local_valley):
                        valley_day, valley_val = _find_local_valley_between(days, vals, day, nearest_right)
                        if valley_day is not None and np.isfinite(valley_val) and valley_val < peak_score:
                            stop_day = min(nearest_right - 1, valley_day)
                            reason = 'local_valley'
                    end_day = min(end_day, stop_day)
                    right_stop_reason = reason

        # Final safety to preserve minimum width while respecting profile index bounds.
        start_day = min(start_day, day - min_half)
        end_day = max(end_day, day + min_half)
        start_day = max(start_day, int(days.min()))
        end_day = min(end_day, int(days.max()))
        if end_day < start_day:
            start_day = max(int(days.min()), day - min_half)
            end_day = min(int(days.max()), day + min_half)
            left_stop_reason = 'fallback_min_half_width'
            right_stop_reason = 'fallback_min_half_width'

        rows.append({
            'candidate_id': str(row['candidate_id']),
            'point_day': day,
            'month_day': row.get('month_day'),
            'band_start_day': int(start_day),
            'band_end_day': int(end_day),
            'band_center_day': int(round((start_day + end_day) / 2.0)),
            'peak_score': peak_score,
            'peak_prominence': peak_prom,
            'support_floor': float(support_floor),
            'global_floor': float(global_floor),
            'left_stop_reason': left_stop_reason,
            'right_stop_reason': right_stop_reason,
            'is_formal_primary': bool(row.get('is_formal_primary', False)),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(['band_start_day', 'point_day']).reset_index(drop=True)
