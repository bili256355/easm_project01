from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings
from .utils import nanmean_no_warn

_STRUCT_COLS = ['window_id', 'struct_test_stat', 'struct_test_pvalue', 'struct_test_pass']
_CONT_COLS = ['window_id', 'continuity_score', 'continuity_pass']
_YEARWISE_DETAIL_COLS = ['window_id', 'year_index', 'anchor_day', 'hit', 'search_start_day', 'search_end_day']
_YEARWISE_SUMMARY_COLS = ['window_id', 'occurrence_rate', 'anchor_dispersion', 'yearwise_pass', 'yearwise_status']


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype='object') for c in columns})


def _ensure_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if df is None or df.empty and len(df.columns) == 0:
        return _empty_df(columns)
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    return out[columns]


def _window_slices(peak_day: int, half_window: int, n_day: int) -> tuple[slice, slice]:
    left_start = max(0, peak_day - half_window)
    left_end = peak_day
    right_start = peak_day + 1
    right_end = min(n_day, peak_day + half_window + 1)
    return slice(left_start, left_end), slice(right_start, right_end)


def run_struct_difference_test(state_cube: np.ndarray, candidate_df: pd.DataFrame, settings: StagePartitionV1Settings) -> pd.DataFrame:
    if candidate_df.empty:
        return _empty_df(_STRUCT_COLS)
    rows = []
    rng = np.random.default_rng(42)
    for _, row in candidate_df.iterrows():
        peak = int(row['peak_day'])
        left_sl, right_sl = _window_slices(peak, settings.score.half_window, state_cube.shape[1])
        year_diffs = []
        for y in range(state_cube.shape[0]):
            left = nanmean_no_warn(state_cube[y, left_sl, :], axis=0)
            right = nanmean_no_warn(state_cube[y, right_sl, :], axis=0)
            diff = np.nan_to_num(right - left, nan=0.0)
            year_diffs.append(diff)
        year_diffs = np.asarray(year_diffs, dtype=np.float64)
        mean_diff = nanmean_no_warn(year_diffs, axis=0)
        stat = float(np.linalg.norm(mean_diff))
        null = np.zeros(settings.tests.struct_test_reps, dtype=np.float64)
        for i in range(settings.tests.struct_test_reps):
            signs = rng.choice(np.array([-1.0, 1.0]), size=year_diffs.shape[0])[:, None]
            null[i] = float(np.linalg.norm(nanmean_no_warn(year_diffs * signs, axis=0)))
        pvalue = float((1 + np.sum(null >= stat)) / (1 + null.size))
        rows.append({
            'window_id': row['window_id'],
            'struct_test_stat': stat,
            'struct_test_pvalue': pvalue,
            'struct_test_pass': bool(pvalue <= 0.10),
        })
    return _ensure_columns(pd.DataFrame(rows), _STRUCT_COLS)


def run_continuity_test(score_df: pd.DataFrame, candidate_df: pd.DataFrame, settings: StagePartitionV1Settings) -> pd.DataFrame:
    if candidate_df.empty:
        return _empty_df(_CONT_COLS)
    rows = []
    valid = score_df['score_smooth'].dropna()
    median_val = float(valid.median()) if not valid.empty else np.nan
    for _, row in candidate_df.iterrows():
        sub = score_df[(score_df['day_index'] >= row['start_day']) & (score_df['day_index'] <= row['end_day'])]
        vals = sub['score_smooth'].to_numpy(dtype=np.float64)
        if vals.size == 0 or not np.isfinite(vals).any():
            score = np.nan
        else:
            frac_high = float(np.mean(vals >= median_val)) if np.isfinite(median_val) else 0.0
            peak_pos = int(np.nanargmax(vals))
            left_monotone = np.mean(np.diff(vals[:peak_pos + 1]) >= -1e-8) if peak_pos > 0 else 1.0
            right_monotone = np.mean(np.diff(vals[peak_pos:]) <= 1e-8) if peak_pos < vals.size - 1 else 1.0
            score = float(0.5 * frac_high + 0.25 * left_monotone + 0.25 * right_monotone)
        rows.append({'window_id': row['window_id'], 'continuity_score': score, 'continuity_pass': bool(np.isfinite(score) and score >= settings.tests.continuity_min_score)})
    return _ensure_columns(pd.DataFrame(rows), _CONT_COLS)


def run_yearwise_occurrence_test(score_cube: np.ndarray, candidate_df: pd.DataFrame, settings: StagePartitionV1Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_df.empty:
        return _empty_df(_YEARWISE_DETAIL_COLS), _empty_df(_YEARWISE_SUMMARY_COLS)
    year_rows = []
    summary_rows = []
    for _, row in candidate_df.iterrows():
        start = max(0, int(row['start_day']) - settings.score.half_window)
        end = min(score_cube.shape[1] - 1, int(row['end_day']) + settings.score.half_window)
        anchors = []
        hits = 0
        for y in range(score_cube.shape[0]):
            year_vals = score_cube[y]
            valid = year_vals[np.isfinite(year_vals)]
            threshold = float(np.quantile(valid, settings.tests.yearwise_peak_quantile)) if valid.size else np.nan
            local = year_vals[start:end + 1]
            if local.size == 0 or not np.isfinite(local).any():
                anchor = np.nan
                hit = False
            else:
                local_idx = int(np.nanargmax(local))
                local_peak = float(local[local_idx])
                anchor = float(start + local_idx)
                hit = bool(np.isfinite(threshold) and local_peak >= threshold)
            anchors.append(anchor)
            hits += int(hit)
            year_rows.append({'window_id': row['window_id'], 'year_index': y, 'anchor_day': anchor, 'hit': hit, 'search_start_day': start, 'search_end_day': end})
        anchors_arr = np.asarray([a for a in anchors if np.isfinite(a)], dtype=np.float64)
        occ = float(hits / score_cube.shape[0])
        disp = float(np.nanstd(anchors_arr)) if anchors_arr.size else np.nan
        summary_rows.append({'window_id': row['window_id'], 'occurrence_rate': occ, 'anchor_dispersion': disp, 'yearwise_pass': bool(occ >= settings.tests.occurrence_min), 'yearwise_status': 'yearwise_pass' if occ >= settings.tests.occurrence_min else 'yearwise_too_sparse'})
    return _ensure_columns(pd.DataFrame(year_rows), _YEARWISE_DETAIL_COLS), _ensure_columns(pd.DataFrame(summary_rows), _YEARWISE_SUMMARY_COLS)


def assemble_general_test_table(struct_df: pd.DataFrame, continuity_df: pd.DataFrame, yearwise_summary_df: pd.DataFrame) -> pd.DataFrame:
    struct_df = _ensure_columns(struct_df, _STRUCT_COLS)
    continuity_df = _ensure_columns(continuity_df, _CONT_COLS)
    yearwise_summary_df = _ensure_columns(yearwise_summary_df, _YEARWISE_SUMMARY_COLS)

    if struct_df.empty and continuity_df.empty and yearwise_summary_df.empty:
        return pd.DataFrame(columns=['window_id', 'struct_test_stat', 'struct_test_pvalue', 'struct_test_pass', 'continuity_score', 'continuity_pass', 'occurrence_rate', 'anchor_dispersion', 'yearwise_pass', 'yearwise_status'])

    merged = struct_df.merge(continuity_df, on='window_id', how='outer').merge(yearwise_summary_df, on='window_id', how='outer')
    return merged.sort_values('window_id').reset_index(drop=True)
