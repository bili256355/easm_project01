from __future__ import annotations
from .config import StagePartitionV6Settings
from .io import prepare_output_dirs, resolve_smoothed_fields_path, load_smoothed_fields, write_dataframe, write_json
from .state_builder import build_profiles, build_state_matrix
from .detector_ruptures_window import run_point_detector
from .candidate_registry import build_candidate_registry
from .bootstrap_support import run_bootstrap_support
from .yearwise_support import run_yearwise_support
from .report import now_utc, build_summary


def run_stage_partition_v6(settings: StagePartitionV6Settings | None = None):
    settings = settings or StagePartitionV6Settings()
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

    registry_df = build_candidate_registry(det['local_peaks_df'], det['primary_points_df'], source_run_tag=settings.output.output_tag)
    write_dataframe(registry_df, output_root / 'baseline_detected_peaks_registry.csv')

    bootstrap = run_bootstrap_support(profiles, smoothed['years'], registry_df, settings)
    write_dataframe(bootstrap['records_df'], output_root / 'candidate_points_bootstrap_match_records.csv')
    write_dataframe(bootstrap['summary_df'], output_root / 'candidate_points_bootstrap_summary.csv')
    write_dataframe(bootstrap['meta_df'], output_root / 'candidate_points_bootstrap_replicates.csv')

    yearwise = run_yearwise_support(profiles, smoothed['years'], registry_df, settings)
    write_dataframe(yearwise['records_df'], output_root / 'candidate_points_yearwise_match_records.csv')
    write_dataframe(yearwise['summary_df'], output_root / 'candidate_points_yearwise_support.csv')

    summary = build_summary(registry_df, bootstrap['summary_df'], yearwise['summary_df'], settings)
    write_json(summary, output_root / 'summary.json')

    run_meta = {
        'status': 'success',
        'started_at_utc': started_at,
        'ended_at_utc': now_utc(),
        'layer_name': 'stage_partition',
        'version_name': 'V6',
        'run_label': settings.output.output_tag,
        'smoothed_fields_path': str(smoothed_path),
        'notes': [
            'V6-a is a baseline-detected-peaks bootstrap screening branch.',
            'All baseline detector local peaks enter the candidate registry.',
            'Headline significance metric is bootstrap local-peak recurrence for all candidates.',
            'Default bootstrap contract is 1000 replicates with strict<=2 days, match<=5 days, near<=8 days.',
            'Yearwise local-peak support is auxiliary only and does not enter headline significance decisions.',
            'Auxiliary yearwise support uses the same match<=5 day contract for consistency, but remains non-decisive.',
            'Window judgement is not included.',
            'Competition is not included.',
            'Parameter-path is not included.',
            'Final judgement is not included.',
        ],
    }
    write_json(run_meta, output_root / 'run_meta.json')
    return {'output_root': output_root, 'summary': summary}
