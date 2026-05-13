
from __future__ import annotations
import numpy as np
import pandas as pd


def merge_candidate_bands_into_windows(bands_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame | None, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    wcols = [
        'window_id', 'start_day', 'end_day', 'center_day', 'main_peak_day',
        'n_member_points', 'member_candidate_ids', 'max_member_bootstrap_match_fraction',
        'merge_reason', 'protected_split_flag'
    ]
    mcols = ['window_id', 'candidate_id', 'point_day', 'is_main_peak', 'bootstrap_match_fraction_5d']
    if bands_df is None or bands_df.empty:
        return pd.DataFrame(columns=wcols), pd.DataFrame(columns=mcols)

    bands = bands_df.sort_values(['band_start_day', 'band_end_day', 'point_day']).reset_index(drop=True)
    boot = {}
    if bootstrap_summary_df is not None and not bootstrap_summary_df.empty and 'candidate_id' in bootstrap_summary_df.columns:
        metric_col = 'bootstrap_match_fraction'
        boot = bootstrap_summary_df.set_index('candidate_id')[metric_col].to_dict()

    sig_thr = float(cfg.significant_peak_threshold)
    near_exempt = int(cfg.close_neighbor_exemption_days)
    gap = int(cfg.merge_gap_days)

    groups = []
    current = []
    current_group_max_end = None
    current_group_protected_days: list[int] = []
    current_group_protected_flag = False

    def close_group(reason: str, protected_split_flag: bool = False):
        nonlocal current, current_group_max_end, current_group_protected_days, current_group_protected_flag
        if current:
            groups.append((current.copy(), reason, protected_split_flag))
        current = []
        current_group_max_end = None
        current_group_protected_days = []
        current_group_protected_flag = False

    for i in range(len(bands)):
        row = bands.loc[i]
        day = int(row['point_day'])
        start = int(row['band_start_day'])
        end = int(row['band_end_day'])
        is_protected = float(boot.get(row['candidate_id'], np.nan)) >= sig_thr if row['candidate_id'] in boot else False

        if not current:
            current = [i]
            current_group_max_end = end
            current_group_protected_days = [day] if is_protected else []
            current_group_protected_flag = False
            continue

        overlaps_group = bool(cfg.allow_band_merge) and start <= int(current_group_max_end) + gap
        if not overlaps_group:
            close_group('non_overlap', protected_split_flag=current_group_protected_flag)
            current = [i]
            current_group_max_end = end
            current_group_protected_days = [day] if is_protected else []
            current_group_protected_flag = False
            continue

        protected_conflict = False
        if bool(cfg.protect_significant_peaks_from_merge) and is_protected and current_group_protected_days:
            # If this protected peak is not a close-neighbor exception of any existing protected peak,
            # keep it separate even if bands overlap.
            protected_conflict = all(abs(day - existing_day) > near_exempt for existing_day in current_group_protected_days)

        if protected_conflict:
            close_group('protected_split', protected_split_flag=True)
            current = [i]
            current_group_max_end = end
            current_group_protected_days = [day]
            current_group_protected_flag = False
            continue

        current.append(i)
        current_group_max_end = max(int(current_group_max_end), end)
        if is_protected:
            current_group_protected_days.append(day)

    close_group('final_group', protected_split_flag=current_group_protected_flag)

    wrows = []
    mrows = []
    for gidx, (idxs, merge_reason, protected_split_flag) in enumerate(groups, start=1):
        sub = bands.loc[idxs].copy().reset_index(drop=True)
        start_day = int(sub['band_start_day'].min())
        end_day = int(sub['band_end_day'].max())
        center_day = int(round((start_day + end_day) / 2.0))
        sub['bootstrap_match_fraction_5d'] = sub['candidate_id'].map(lambda x: float(boot.get(x, np.nan)))
        sub = sub.sort_values(['bootstrap_match_fraction_5d', 'peak_score', 'point_day'], ascending=[False, False, True]).reset_index(drop=True)
        main_peak_day = int(sub.iloc[0]['point_day'])
        window_id = f'W{gidx:03d}'
        members = ';'.join(sub['candidate_id'].astype(str).tolist())
        max_boot = float(sub['bootstrap_match_fraction_5d'].max()) if sub['bootstrap_match_fraction_5d'].notna().any() else np.nan
        wrows.append({
            'window_id': window_id,
            'start_day': start_day,
            'end_day': end_day,
            'center_day': center_day,
            'main_peak_day': main_peak_day,
            'n_member_points': int(len(sub)),
            'member_candidate_ids': members,
            'max_member_bootstrap_match_fraction': max_boot,
            'merge_reason': merge_reason,
            'protected_split_flag': bool(protected_split_flag),
        })
        for _, subrow in sub.iterrows():
            mrows.append({
                'window_id': window_id,
                'candidate_id': str(subrow['candidate_id']),
                'point_day': int(subrow['point_day']),
                'is_main_peak': bool(int(subrow['point_day']) == main_peak_day),
                'bootstrap_match_fraction_5d': float(subrow['bootstrap_match_fraction_5d']) if pd.notna(subrow['bootstrap_match_fraction_5d']) else np.nan,
            })
    windows_df = pd.DataFrame(wrows, columns=wcols)
    membership_df = pd.DataFrame(mrows, columns=mcols)
    return windows_df, membership_df
