from __future__ import annotations
import numpy as np
import pandas as pd

from .selection_frequency_config import StagePartitionV6SelectionFrequencySettings
from .io import prepare_output_dirs, resolve_smoothed_fields_path, load_smoothed_fields, write_dataframe, write_json
from .state_builder import build_profiles, build_state_matrix, build_resampled_state_matrix
from .detector_ruptures_window import run_point_detector
from .selection_frequency import (
    build_selection_frequency_by_day,
    smooth_selection_frequency,
    extract_selection_local_maxima,
    build_baseline_peak_context_table,
    plot_selection_frequency_curve,
)
from .selection_frequency_report import now_utc, build_selection_frequency_summary


def _progress_iter(items, enabled: bool, desc: str):
    if enabled:
        try:
            from tqdm import tqdm
            return tqdm(items, desc=desc)
        except Exception:
            return items
    return items


def run_stage_partition_v6_b_selection_frequency(settings: StagePartitionV6SelectionFrequencySettings | None = None):
    settings = settings or StagePartitionV6SelectionFrequencySettings()
    started_at = now_utc()
    roots = prepare_output_dirs(settings)
    output_root = roots['output_root']
    smoothed_path = resolve_smoothed_fields_path(settings)
    smoothed = load_smoothed_fields(smoothed_path)
    settings.write_json(output_root / 'config_used.json')

    profiles = build_profiles(smoothed, settings.profile)
    state = build_state_matrix(profiles, settings.state)
    det = run_point_detector(
        state['state_matrix'][state['valid_day_mask'], :],
        state['valid_day_index'],
        settings.detector,
        local_peak_distance_days=settings.detector.local_peak_min_distance_days,
    )
    write_dataframe(det['primary_points_df'], output_root / 'ruptures_primary_points.csv')
    write_dataframe(det['local_peaks_df'], output_root / 'baseline_detector_local_peaks_all.csv')

    rng = np.random.default_rng(settings.bootstrap.random_seed)
    years = np.asarray(smoothed['years']).astype(int)
    n_years = len(years)
    peak_records = []
    peak_counts = []

    for rep in _progress_iter(range(int(settings.bootstrap.n_bootstrap)), settings.bootstrap.progress, 'V6-b selection frequency'):
        sampled = rng.integers(0, n_years, size=n_years, endpoint=False)
        rs = build_resampled_state_matrix(profiles, sampled, settings.state)
        if rs['valid_day_index'].size < max(2 * int(settings.detector.width), 3):
            peak_counts.append({
                'replicate_id': int(rep),
                'status': 'skipped_insufficient_valid_days',
                'n_valid_days': int(rs['valid_day_index'].size),
                'n_detected_local_peaks': 0,
            })
            continue
        rep_det = run_point_detector(
            rs['state_matrix'][rs['valid_day_mask'], :],
            rs['valid_day_index'],
            settings.detector,
            local_peak_distance_days=settings.detector.local_peak_min_distance_days,
        )
        lp = rep_det['local_peaks_df'].copy()
        if lp is None or lp.empty:
            peak_counts.append({
                'replicate_id': int(rep),
                'status': 'success',
                'n_valid_days': int(rs['valid_day_index'].size),
                'n_detected_local_peaks': 0,
            })
            continue
        lp['replicate_id'] = int(rep)
        peak_records.append(lp[['replicate_id', 'peak_id', 'peak_day', 'peak_score', 'peak_prominence', 'peak_rank', 'source_type']].copy())
        peak_counts.append({
            'replicate_id': int(rep),
            'status': 'success',
            'n_valid_days': int(rs['valid_day_index'].size),
            'n_detected_local_peaks': int(len(lp)),
        })

    peak_records_df = pd.concat(peak_records, ignore_index=True) if peak_records else pd.DataFrame(columns=['replicate_id','peak_id','peak_day','peak_score','peak_prominence','peak_rank','source_type'])
    peak_counts_df = pd.DataFrame(peak_counts)
    if bool(settings.experiment.write_replicate_peak_counts):
        write_dataframe(peak_counts_df, output_root / 'bootstrap_replicate_peak_counts.csv')

    max_day_index = int(max(pd.to_numeric(det['local_peaks_df']['peak_day'], errors='coerce').max() if det['local_peaks_df'] is not None and not det['local_peaks_df'].empty else 0, np.max(state['valid_day_index']) if state['valid_day_index'].size else 0))
    freq_df = build_selection_frequency_by_day(
        peak_records_df,
        max_day_index=max_day_index,
        n_total_replicates=int(settings.bootstrap.n_bootstrap),
    )
    if bool(settings.experiment.use_smoothing):
        freq_df = smooth_selection_frequency(freq_df, window_days=int(settings.experiment.smoothing_window_days))
    else:
        freq_df['selection_frequency_smooth'] = np.nan
    write_dataframe(freq_df, output_root / 'bootstrap_selection_frequency_by_day.csv')

    maxima_df = pd.DataFrame(columns=['peak_id','peak_day','peak_month_day','selection_frequency_raw','selection_frequency_smooth','source_column'])
    if bool(settings.experiment.emit_local_maxima):
        maxima_df = extract_selection_local_maxima(
            freq_df,
            min_frequency=float(settings.experiment.local_maxima_min_frequency),
            use_smoothed=bool(settings.experiment.use_smoothing),
        )
        write_dataframe(maxima_df, output_root / 'bootstrap_selection_local_maxima.csv')

    baseline_context_df = build_baseline_peak_context_table(freq_df, det['local_peaks_df'])
    write_dataframe(baseline_context_df, output_root / 'baseline_detector_local_peaks_with_frequency.csv')

    plot_meta = {
        'status': 'disabled',
        'output_path': str(output_root / str(settings.experiment.curve_png_filename)),
    }
    if bool(settings.experiment.write_curve_png):
        plot_meta = plot_selection_frequency_curve(
            freq_df,
            det['local_peaks_df'],
            output_root / str(settings.experiment.curve_png_filename),
            annotate_baseline_peaks=bool(settings.experiment.annotate_baseline_peaks),
            dpi=int(settings.experiment.curve_dpi),
        )

    summary = build_selection_frequency_summary(freq_df, peak_counts_df, maxima_df, settings, plot_meta=plot_meta)
    write_json(summary, output_root / 'summary.json')
    run_meta = {
        'status': 'success',
        'started_at_utc': started_at,
        'ended_at_utc': now_utc(),
        'layer_name': 'stage_partition',
        'version_name': 'V6',
        'run_label': settings.experiment.output_tag,
        'smoothed_fields_path': str(smoothed_path),
        'notes': [
            'This is an isolated V6-b selection-frequency experiment layer.',
            'It does not replace the current V6-a point-level bootstrap main table.',
            'It summarizes where detector local peaks are repeatedly selected across bootstrap replicates.',
            'It does not include window judgement, competition, parameter-path, or final judgement.',
            'Default bootstrap replicate count for this experiment is 1000.',
            'Selection-frequency curve PNG is written by default when matplotlib is available.',
        ],
    }
    write_json(run_meta, output_root / 'run_meta.json')
    return {'output_root': output_root, 'summary': summary}
