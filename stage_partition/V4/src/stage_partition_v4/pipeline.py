from __future__ import annotations
from .config import StagePartitionV4Settings
from .io import prepare_output_dirs, resolve_smoothed_fields_path, load_smoothed_fields, write_dataframe, write_json
from .timeline import build_valid_day_metadata
from .state_builder import build_profiles, summarize_profile_validity, summarize_profile_empty_slices, build_state_matrix
from .detector_ruptures_window import run_point_detector
from .point_reference import build_formal_reference_points
from .point_audit_universe import build_point_audit_universe
from .point_competition import build_point_neighbor_pairs, finalize_point_competition
from .point_yearwise import run_point_yearwise_support
from .point_bootstrap import run_point_bootstrap_support
from .point_matching import build_neighbor_match_audit, build_candidate_universe_debug
from .point_parampath import run_point_parameter_path_support
from .point_btrack_audit import build_point_robust_support_audit, build_point_btrack_role_summary, build_point_btrack_summary_json
from .report import now_utc, build_summary


def run_stage_partition_v4(settings: StagePartitionV4Settings | None = None):
    settings = settings or StagePartitionV4Settings()
    started_at = now_utc()
    roots = prepare_output_dirs(settings)
    output_root = roots['output_root']
    smoothed_path = resolve_smoothed_fields_path(settings)
    smoothed = load_smoothed_fields(smoothed_path)
    settings.write_json(output_root / 'config_used.json')

    profiles = build_profiles(smoothed, settings.profile)
    write_dataframe(summarize_profile_validity(profiles), output_root / 'profile_validity_summary.csv')
    write_dataframe(summarize_profile_empty_slices(profiles), output_root / 'profile_empty_slice_audit.csv')

    state = build_state_matrix(profiles, settings.state)
    write_dataframe(state['feature_table'], output_root / 'feature_table.csv')
    write_json(state['state_vector_meta'], output_root / 'state_vector_meta.json')
    write_dataframe(state['state_feature_scale_before_after'], output_root / 'state_feature_scale_before_after.csv')
    write_dataframe(state['state_block_energy_before_after']['raw'], output_root / 'state_block_energy_raw.csv')
    write_dataframe(state['state_block_energy_before_after']['standardized'], output_root / 'state_block_energy_standardized.csv')
    write_dataframe(state['state_block_energy_before_after']['weights'], output_root / 'state_block_weight_effect.csv')
    write_dataframe(state['state_empty_feature_audit'], output_root / 'state_empty_feature_audit.csv')
    write_dataframe(build_valid_day_metadata(state['valid_day_mask']), output_root / 'detector_valid_day_mask.csv')

    det = run_point_detector(
        state['state_matrix'][state['valid_day_mask'], :],
        state['valid_day_index'],
        settings.detector,
        local_peak_distance_days=settings.detector.local_peak_min_distance_days,
    )
    write_dataframe(det['primary_points_df'], output_root / 'ruptures_primary_points.csv')
    raw_points_df = det.get('raw_points_df')
    if raw_points_df is not None and len(raw_points_df) > 0:
        write_dataframe(raw_points_df, output_root / 'ruptures_primary_points_raw.csv')
    else:
        write_json(
            {
                'warning': 'raw_points_df_missing',
                'message': 'Detector did not return raw_points_df; raw breakpoint audit table was skipped.'
            },
            output_root / 'raw_points_df_warning.json',
        )
    write_dataframe(det['local_peaks_df'], output_root / 'detector_local_peaks_all.csv')
    write_dataframe(det['profile'].rename('profile').reset_index().rename(columns={'index': 'day'}), output_root / 'ruptures_window_profile.csv')

    reference_df = build_formal_reference_points(det['primary_points_df'], source_run_tag=settings.output.output_tag)
    write_dataframe(reference_df, output_root / 'point_reference_points.csv')

    audit_universe_df = build_point_audit_universe(
        reference_df,
        det['local_peaks_df'],
        det['profile'],
        radius_days=settings.point_audit_universe.radius_days,
        weak_peak_min_prominence=settings.point_audit_universe.weak_peak_min_prominence,
        source_run_tag=settings.output.output_tag,
    )
    write_dataframe(audit_universe_df, output_root / 'point_audit_universe.csv')
    write_dataframe(build_candidate_universe_debug(candidate_universe_df=audit_universe_df), output_root / 'point_candidate_universe_debug.csv')

    pair_stage = build_point_neighbor_pairs(
        reference_df,
        audit_universe_df,
        settings.competition.neighbor_radius_days,
        source_run_tag=settings.output.output_tag,
    )
    write_dataframe(pair_stage['pairs_df'], output_root / 'point_neighbor_pairs.csv')
    write_dataframe(pair_stage['augmented_reference_df'], output_root / 'point_reference_points_augmented.csv')

    yearwise = run_point_yearwise_support(profiles, smoothed['years'], pair_stage['augmented_reference_df'], settings, pair_df=pair_stage['pairs_df'])
    write_dataframe(yearwise['long_df'], output_root / 'point_yearwise_hits_long.csv')
    write_dataframe(yearwise['summary_df'], output_root / 'point_yearwise_support_summary.csv')
    write_dataframe(yearwise['matchtype_summary_df'], output_root / 'point_yearwise_matchtype_support.csv')
    write_dataframe(yearwise['pair_summary_df'], output_root / 'point_yearwise_pair_competition.csv')

    bootstrap = run_point_bootstrap_support(profiles, smoothed['years'], pair_stage['augmented_reference_df'], settings, pair_df=pair_stage['pairs_df'])
    write_dataframe(bootstrap['records_df'], output_root / 'point_bootstrap_match_records.csv')
    write_dataframe(bootstrap['summary_df'], output_root / 'point_bootstrap_match_summary.csv')
    write_dataframe(bootstrap['offsets_df'], output_root / 'point_bootstrap_offsets.csv')
    write_dataframe(bootstrap['meta_df'], output_root / 'point_bootstrap_replicates.csv')

    bootstrap_pair_df = build_neighbor_match_audit(bootstrap['records_df'], pair_stage['pairs_df'])
    write_dataframe(bootstrap_pair_df, output_root / 'point_neighbor_match_audit.csv')

    param = run_point_parameter_path_support(
        state['state_matrix'],
        state['valid_day_mask'],
        state['valid_day_index'],
        pair_stage['augmented_reference_df'],
        settings,
        pair_df=pair_stage['pairs_df'],
    )
    write_dataframe(param['records_df'], output_root / 'point_parameter_path_replicates.csv')
    write_dataframe(param['summary_df'], output_root / 'point_parameter_path_support.csv')

    competition_audit_df = finalize_point_competition(
        pair_stage['pairs_df'],
        bootstrap_pair_df,
        yearwise['pair_summary_df'],
        settings.competition,
        source_run_tag=settings.output.output_tag,
    )
    write_dataframe(competition_audit_df, output_root / 'point_neighbor_competition_audit.csv')

    audit_df = build_point_robust_support_audit(
        pair_stage['augmented_reference_df'],
        bootstrap['summary_df'],
        yearwise['summary_df'],
        param['summary_df'],
        competition_audit_df,
        settings,
    )
    role_summary_df = build_point_btrack_role_summary(audit_df)
    btrack_summary = build_point_btrack_summary_json(audit_df)
    btrack_summary['ambiguity_source_cleaned'] = True
    btrack_summary['yearwise_exact_strict_only'] = True
    btrack_summary['point_audit_universe_enabled'] = True
    btrack_summary['point_audit_universe_count'] = int(len(audit_universe_df))
    btrack_summary['audit_universe_mode'] = settings.point_audit_universe.audit_universe_mode
    btrack_summary['pair_yearwise_method'] = settings.yearwise.pair_competition_mode
    btrack_summary['pair_yearwise_requires_both_detected'] = bool(settings.yearwise.pair_requires_both_detected)
    btrack_summary['parameter_path_scope'] = settings.parameter_path.parameter_path_scope
    btrack_summary['judgement_sensitivity_not_included'] = True
    write_dataframe(audit_df, output_root / 'point_robust_support_audit.csv')
    write_dataframe(role_summary_df, output_root / 'point_btrack_role_summary.csv')
    write_json(btrack_summary, output_root / 'point_btrack_summary.json')

    summary = build_summary(audit_df, role_summary_df, pair_stage['augmented_reference_df'], competition_audit_df, bootstrap['meta_df'])
    summary['audit_universe_mode'] = settings.point_audit_universe.audit_universe_mode
    summary['pair_yearwise_method'] = settings.yearwise.pair_competition_mode
    summary['pair_yearwise_requires_both_detected'] = bool(settings.yearwise.pair_requires_both_detected)
    summary['parameter_path_scope'] = settings.parameter_path.parameter_path_scope
    summary['judgement_sensitivity_not_included'] = True
    write_json(summary, output_root / 'summary.json')

    run_meta = {
        'status': 'success',
        'started_at_utc': started_at,
        'ended_at_utc': now_utc(),
        'layer_name': 'stage_partition',
        'version_name': 'V4',
        'run_label': settings.output.output_tag,
        'foundation_dependency': 'V1',
        'smoothed_fields_path': str(smoothed_path),
        'notes': [
            'V4 mainline_v4_d keeps the clean point-layer B-track-only branch.',
            'A-track does not enter V4.',
            'Window-layer and pathway-layer logic do not enter V4 stage one.',
            'B-track backend mode is independent_recompute and does not read V3 outputs.',
            'Candidate universe now deduplicates formal-primary/local-peak aliases.',
            'Yearwise exact_hit now counts strict_match only.',
            'Parameter-path configs now use detector-effective width and pen combinations only.',
            'Neighbor competition final verdict now uses score plus bootstrap/yearwise support.',
            'Point audit universe remains headline-centered and local-neighbor only.',
            'Pair-level yearwise remains conservative and comparable-years only.',
            'Current parameter-path scope is detector-effective only; judgement sensitivity is not yet included.',
            'Judgement mapping now promotes pair-clear moderate+moderate formal-primary cases to supported rather than caution.',
        ],
        'method_contract': {
            'audit_universe_mode': settings.point_audit_universe.audit_universe_mode,
            'pair_yearwise_method': settings.yearwise.pair_competition_mode,
            'pair_yearwise_requires_both_detected': bool(settings.yearwise.pair_requires_both_detected),
            'parameter_path_scope': settings.parameter_path.parameter_path_scope,
            'judgement_sensitivity_not_included': True,
        },
    }
    write_json(run_meta, output_root / 'run_meta.json')
    return {'output_root': output_root, 'summary': summary, 'btrack_summary': btrack_summary}
