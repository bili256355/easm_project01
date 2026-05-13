from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .config import EdgeConfig, RupturesWindowConstructionConfig


@dataclass
class _PeakInfo:
    position: int
    day: int
    value: float
    prominence: float
    width: float
    left_base: int
    right_base: int


def _find_profile_peaks(series: pd.Series, cfg: EdgeConfig) -> dict[int, _PeakInfo]:
    s = series.sort_index().fillna(0.0)
    values = s.to_numpy(dtype=float)
    peaks, _ = find_peaks(values, distance=cfg.min_distance_days)
    if peaks.size == 0:
        return {}
    prominences, left_bases, right_bases = peak_prominences(values, peaks)
    widths, _, _, _ = peak_widths(values, peaks, rel_height=0.5)
    out: dict[int, _PeakInfo] = {}
    for p, prom, width, lb, rb in zip(peaks, prominences, widths, left_bases, right_bases):
        out[int(p)] = _PeakInfo(
            position=int(p),
            day=int(s.index[int(p)]),
            value=float(values[int(p)]),
            prominence=float(prom),
            width=float(width),
            left_base=int(lb),
            right_base=int(rb),
        )
    return out


def _nearest_peak_position(peaks: np.ndarray, target_day: int, day_index: np.ndarray, radius: int) -> int | None:
    if peaks.size == 0:
        return None
    days = day_index[peaks]
    diffs = np.abs(days - int(target_day))
    k = int(np.argmin(diffs))
    if diffs[k] <= radius:
        return int(peaks[k])
    return None


def _build_support_band(values: np.ndarray, peak_info: _PeakInfo, cfg: RupturesWindowConstructionConfig) -> tuple[int, int, float]:
    n = values.size
    floor_q = float(np.nanquantile(values, cfg.support_floor_quantile))
    floor_by_prom = peak_info.value - cfg.support_rel_prominence * peak_info.prominence
    floor_by_ratio = peak_info.value * cfg.support_floor_ratio
    threshold = max(floor_q, floor_by_prom, floor_by_ratio)

    left = peak_info.position
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    right = peak_info.position
    while right < n - 1 and values[right + 1] >= threshold:
        right += 1

    left = max(left, peak_info.left_base)
    right = min(right, peak_info.right_base)

    if right < left:
        left = peak_info.position
        right = peak_info.position

    if (right - left + 1) < cfg.min_band_width_days:
        needed = cfg.min_band_width_days - (right - left + 1)
        expand_left = needed // 2
        expand_right = needed - expand_left
        left = max(0, left - expand_left)
        right = min(n - 1, right + expand_right)

    return int(left), int(right), float(threshold)


def _consolidate_bands(bands_df: pd.DataFrame, profile: pd.Series, cfg: RupturesWindowConstructionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bands_df.empty:
        cols = ['window_id', 'backend', 'window_type', 'start_day', 'end_day', 'center_day', 'main_peak_day', 'native_support', 'support_points', 'band_strength', 'n_support_points']
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=['merged_window_id', 'source_point_days', 'source_band_starts', 'source_band_ends'])

    s = profile.sort_index().fillna(0.0)
    values = s.to_numpy(dtype=float)
    days = s.index.to_numpy(dtype=int)
    ordered = bands_df.sort_values(['band_start_day', 'band_end_day', 'peak_day']).reset_index(drop=True)
    groups: list[list[dict]] = []
    current: list[dict] = []

    for rec in ordered.to_dict('records'):
        if not current:
            current = [rec]
            continue
        prev_end = max(int(x['band_end_day']) for x in current)
        if int(rec['band_start_day']) <= prev_end + cfg.merge_gap_days:
            current.append(rec)
        else:
            groups.append(current)
            current = [rec]
    if current:
        groups.append(current)

    windows = []
    merge_rows = []
    for idx, group in enumerate(groups, start=1):
        start_day = min(int(x['band_start_day']) for x in group)
        end_day = max(int(x['band_end_day']) for x in group)
        mask = (days >= start_day) & (days <= end_day)
        local_vals = values[mask]
        local_days = days[mask]
        if local_vals.size:
            local_peak_idx = int(np.nanargmax(local_vals))
            center_day = int(local_days[local_peak_idx])
            band_strength = float(np.nanmax(local_vals))
        else:
            center_day = int(np.mean([start_day, end_day]))
            band_strength = np.nan
        group_sorted = sorted(group, key=lambda x: (-float(x['peak_value']), abs(int(x['peak_day']) - center_day)))
        main_peak_day = int(group_sorted[0]['peak_day'])
        support_points = [int(x['peak_day']) for x in sorted(group, key=lambda x: int(x['peak_day']))]
        windows.append({
            'window_id': f'RW{idx:02d}',
            'backend': 'ruptures_window',
            'window_type': 'primary',
            'start_day': start_day,
            'end_day': end_day,
            'center_day': center_day,
            'main_peak_day': main_peak_day,
            'native_support': band_strength,
            'support_points': ','.join(str(x) for x in support_points),
            'band_strength': band_strength,
            'n_support_points': int(len(support_points)),
        })
        merge_rows.append({
            'merged_window_id': f'RW{idx:02d}',
            'source_point_days': ','.join(str(int(x['peak_day'])) for x in group),
            'source_band_starts': ','.join(str(int(x['band_start_day'])) for x in group),
            'source_band_ends': ','.join(str(int(x['band_end_day'])) for x in group),
            'n_merged_points': int(len(group)),
        })

    return pd.DataFrame(windows), pd.DataFrame(merge_rows)


def build_primary_windows_from_ruptures(points: pd.Series, profile: pd.Series, edge_cfg: EdgeConfig, cfg: RupturesWindowConstructionConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols_points = ['peak_day', 'peak_value', 'peak_prominence', 'peak_width', 'peak_width_zero', 'band_start_day', 'band_end_day', 'band_threshold', 'assigned_window_id']
    if profile.empty:
        return (
            pd.DataFrame(columns=['window_id', 'backend', 'window_type', 'start_day', 'end_day', 'center_day', 'main_peak_day', 'native_support', 'support_points', 'band_strength', 'n_support_points']),
            pd.DataFrame(columns=cols_points),
            pd.DataFrame(columns=['merged_window_id', 'source_point_days', 'source_band_starts', 'source_band_ends', 'n_merged_points']),
            pd.DataFrame(columns=['peak_day', 'peak_value', 'peak_prominence', 'peak_width', 'peak_width_zero', 'absorbed_into_window_id']),
        )

    s = profile.sort_index().fillna(0.0)
    values = s.to_numpy(dtype=float)
    days = s.index.to_numpy(dtype=int)
    peak_map = _find_profile_peaks(s, edge_cfg)
    peaks = np.array(sorted(peak_map.keys()), dtype=int) if peak_map else np.array([], dtype=int)

    point_rows = []
    width_rows = []
    for point_day in points.astype(int).tolist():
        peak_pos = _nearest_peak_position(peaks, int(point_day), days, cfg.nearest_peak_search_radius)
        if peak_pos is None:
            peak_pos = int(np.argmin(np.abs(days - int(point_day))))
            peak_info = _PeakInfo(
                position=peak_pos,
                day=int(days[peak_pos]),
                value=float(values[peak_pos]),
                prominence=0.0,
                width=0.0,
                left_base=max(0, peak_pos - 1),
                right_base=min(len(values) - 1, peak_pos + 1),
            )
        else:
            peak_info = peak_map[int(peak_pos)]
        left, right, threshold = _build_support_band(values, peak_info, cfg)
        point_rows.append({
            'peak_day': int(peak_info.day),
            'peak_value': float(peak_info.value),
            'peak_prominence': float(peak_info.prominence),
            'peak_width': float(peak_info.width),
            'peak_width_zero': bool(float(peak_info.width) <= 0.0),
            'band_start_day': int(days[left]),
            'band_end_day': int(days[right]),
            'band_threshold': float(threshold),
        })
        width_rows.append({
            'peak_day': int(peak_info.day),
            'peak_value': float(peak_info.value),
            'peak_prominence': float(peak_info.prominence),
            'peak_width': float(peak_info.width),
            'peak_width_zero': bool(float(peak_info.width) <= 0.0),
        })

    point_df = pd.DataFrame(point_rows)
    primary_windows, merge_audit = _consolidate_bands(point_df, s, cfg)

    point_to_window = []
    if not point_df.empty and not primary_windows.empty:
        for _, row in point_df.iterrows():
            matched = primary_windows[
                (primary_windows['start_day'] <= int(row['peak_day'])) &
                (primary_windows['end_day'] >= int(row['peak_day']))
            ]
            if matched.empty:
                assigned = None
            else:
                assigned = str(matched.iloc[0]['window_id'])
            r = row.to_dict()
            r['assigned_window_id'] = assigned
            point_to_window.append(r)
    point_to_window_df = pd.DataFrame(point_to_window) if point_to_window else pd.DataFrame(columns=cols_points)

    width_df = pd.DataFrame(width_rows)
    if not width_df.empty and not primary_windows.empty:
        assigns = []
        for _, row in width_df.iterrows():
            matched = primary_windows[
                (primary_windows['start_day'] <= int(row['peak_day'])) &
                (primary_windows['end_day'] >= int(row['peak_day']))
            ]
            assigns.append(str(matched.iloc[0]['window_id']) if not matched.empty else None)
        width_df['absorbed_into_window_id'] = assigns
    else:
        width_df['absorbed_into_window_id'] = pd.Series(dtype=object)

    return primary_windows, point_to_window_df, merge_audit, width_df


def build_edge_windows_from_ruptures(profile: pd.Series, primary_windows: pd.DataFrame, edge_cfg: EdgeConfig, construction_cfg: RupturesWindowConstructionConfig) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame(columns=['window_id', 'backend', 'window_type', 'start_day', 'end_day', 'center_day', 'main_peak_day', 'native_support', 'support_points', 'band_strength', 'n_support_points'])

    s = profile.sort_index().fillna(0.0)
    values = s.to_numpy(dtype=float)
    days = s.index.to_numpy(dtype=int)
    peak_map = _find_profile_peaks(s, edge_cfg)
    if not peak_map:
        return pd.DataFrame(columns=['window_id', 'backend', 'window_type', 'start_day', 'end_day', 'center_day', 'main_peak_day', 'native_support', 'support_points', 'band_strength', 'n_support_points'])

    peaks = np.array(sorted(peak_map.keys()), dtype=int)
    prominences = np.array([peak_map[int(p)].prominence for p in peaks], dtype=float)
    prom_threshold = float(np.quantile(prominences, edge_cfg.prominence_quantile)) if prominences.size else np.inf

    used_mask = np.zeros(peaks.size, dtype=bool)
    if not primary_windows.empty:
        for i, p in enumerate(peaks):
            d = int(days[p])
            if np.any((primary_windows['start_day'].to_numpy(dtype=int) <= d) & (primary_windows['end_day'].to_numpy(dtype=int) >= d)):
                used_mask[i] = True

    rows = []
    edge_idx = 0
    for i, p in enumerate(peaks):
        if used_mask[i]:
            continue
        info = peak_map[int(p)]
        if info.prominence < prom_threshold:
            continue
        left, right, _ = _build_support_band(values, info, construction_cfg)
        edge_idx += 1
        rows.append({
            'window_id': f'RWE{edge_idx:02d}',
            'backend': 'ruptures_window',
            'window_type': 'edge',
            'start_day': int(days[left]),
            'end_day': int(days[right]),
            'center_day': int(info.day),
            'main_peak_day': int(info.day),
            'native_support': float(info.value),
            'support_points': str(int(info.day)),
            'band_strength': float(info.value),
            'n_support_points': 1,
        })
    return pd.DataFrame(rows)
