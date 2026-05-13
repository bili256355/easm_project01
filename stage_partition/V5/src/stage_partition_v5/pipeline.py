from __future__ import annotations
from .config import StagePartitionV5Settings
from .io import prepare_output_dirs, resolve_smoothed_fields_path, load_smoothed_fields, write_dataframe, write_json
from .state_builder import build_profiles, build_state_matrix
from .detector_ruptures_window import run_point_detector
from .point_reference import build_reference_points
from .point_bootstrap_local_peak import run_point_bootstrap_local_peak
from .point_yearwise_local_peak import run_point_yearwise_local_peak
from .report import now_utc, build_summary


def run_stage_partition_v5(settings: StagePartitionV5Settings | None = None):
    settings = settings or StagePartitionV5Settings()
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
    write_dataframe(det['local_peaks_df'], output_root / 'detector_local_peaks_all.csv')

    reference_df = build_reference_points(det['primary_points_df'], source_run_tag=settings.output.output_tag)
    write_dataframe(reference_df, output_root / 'point_reference_points.csv')

    bootstrap = run_point_bootstrap_local_peak(profiles, smoothed['years'], reference_df, settings)
    write_dataframe(bootstrap['records_df'], output_root / 'point_bootstrap_local_peak_match_records.csv')
    write_dataframe(bootstrap['summary_df'], output_root / 'point_bootstrap_local_peak_summary.csv')
    write_dataframe(bootstrap['meta_df'], output_root / 'point_bootstrap_local_peak_replicates.csv')

    yearwise = run_point_yearwise_local_peak(profiles, smoothed['years'], reference_df, settings)
    write_dataframe(yearwise['records_df'], output_root / 'point_yearwise_local_peak_match_records.csv')
    write_dataframe(yearwise['summary_df'], output_root / 'point_yearwise_local_peak_summary.csv')

    summary = build_summary(reference_df, bootstrap['summary_df'], yearwise['summary_df'], settings)
    write_json(summary, output_root / 'summary.json')

    run_meta = {
        'status': 'success',
        'started_at_utc': started_at,
        'ended_at_utc': now_utc(),
        'layer_name': 'stage_partition',
        'version_name': 'V5',
        'run_label': settings.output.output_tag,
        'smoothed_fields_path': str(smoothed_path),
        'notes': [
            'V5-a is a paper-metric local-peak stability branch.',
            'Headline bootstrap metric uses nearest local-peak recurrence on all detector local peaks.',
            'Yearwise support uses nearest local-peak matching on all yearly detector local peaks.',
            'Object-aware support is not included.',
            'Competition is not included.',
            'Parameter-path is not included.',
            'Final judgement is not included.',
        ],
    }
    write_json(run_meta, output_root / 'run_meta.json')
    return {'output_root': output_root, 'summary': summary}
