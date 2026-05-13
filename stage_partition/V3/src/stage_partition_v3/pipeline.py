from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd

from .config import StagePartitionV3Settings
from .io import prepare_output_dirs, resolve_smoothed_fields_path, load_smoothed_fields, write_dataframe, write_json
from .profiles import build_profiles, summarize_profile_validity, summarize_profile_empty_slices
from .state_vector import build_state_matrix
from .detector_ruptures_window import run_ruptures_window
from .window_builder import (
    build_primary_windows_from_ruptures,
    build_edge_windows_from_ruptures,
    build_point_table,
    build_band_table,
    build_main_window_table,
    build_window_catalog,
)
from .evidence_mapping import build_window_evidence_mapping
from .yearwise_diagnostics import build_yearwise_diagnostics
from .point_significance import (
    build_point_audit_universe,
    run_point_null_significance,
    run_point_stability_audit,
    build_neighbor_pairs,
    run_point_neighbor_competition,
    summarize_point_significance,
    build_point_significance_summary,
)
from .support_audit_window import (
    run_bootstrap_window_matches,
    run_param_path_window_matches,
    run_global_permutation_audit,
    summarize_window_support,
    build_support_rule_comparison,
)
from .retention_audit_window import build_window_retention_audit
from .trust_tiers import build_audit_trust_tiers
from .point_btrack_bootstrap import (
    build_point_btrack_reference_points,
    build_point_bootstrap_match_records,
    summarize_point_bootstrap_matches,
)
from .point_btrack_yearwise import build_point_yearwise_support_summary
from .point_btrack_parampath import build_point_parameter_path_support
from .point_btrack_audit import (
    build_point_robust_support_audit,
    build_point_btrack_role_summary,
    build_point_btrack_summary_json,
)
from .point_btrack_matching import build_point_neighbor_match_audit
from .report import plot_detector_profile, plot_main_windows, build_summary


def _now_utc() -> str:
    return datetime.utcnow().isoformat() + 'Z'


def run_stage_partition_v3(settings: StagePartitionV3Settings | None = None) -> dict:
    settings = settings or StagePartitionV3Settings()
    started_at = _now_utc()
    dirs = prepare_output_dirs(settings)
    output_root = dirs['output_root']
    settings.write_json(output_root / 'config_used.json')

    smoothed_path = resolve_smoothed_fields_path(settings)
    smoothed = load_smoothed_fields(smoothed_path)

    profiles = build_profiles(smoothed, settings.profile)
    profile_validity = summarize_profile_validity(profiles)
    profile_empty_audit = summarize_profile_empty_slices(profiles)
    write_dataframe(profile_validity, output_root / 'profile_validity_summary.csv')
    write_dataframe(profile_empty_audit, output_root / 'profile_empty_slice_audit.csv')

    state = build_state_matrix(profiles, settings.state)
    np.save(output_root / 'state_matrix.npy', state['state_matrix'])
    write_json(state['state_vector_meta'], output_root / 'state_vector_meta.json')
    write_dataframe(state['feature_table'], output_root / 'feature_table.csv')
    write_dataframe(state['state_feature_scale_before_after'], output_root / 'state_feature_scale_before_after.csv')
    write_dataframe(state['state_block_energy_before_after']['raw'], output_root / 'state_block_energy_raw.csv')
    write_dataframe(state['state_block_energy_before_after']['standardized'], output_root / 'state_block_energy_standardized.csv')
    write_dataframe(state['state_block_energy_before_after']['weights'], output_root / 'state_block_weight_effect.csv')
    write_dataframe(state['state_empty_feature_audit'], output_root / 'state_empty_feature_audit.csv')
    write_dataframe(pd.DataFrame({'day_index': np.arange(state['state_matrix'].shape[0], dtype=int), 'detector_valid': state['valid_day_mask'].astype(int)}), output_root / 'detector_valid_day_mask.csv')

    state_valid = state['state_matrix'][state['valid_day_mask'], :]
    valid_day_index = state['valid_day_index']

    rw_out = run_ruptures_window(state_valid, settings.ruptures_window, day_index=valid_day_index)
    write_dataframe(pd.DataFrame({'changepoint': rw_out['points']}), output_root / 'ruptures_primary_points_raw.csv')
    write_dataframe(pd.DataFrame({'day': rw_out['profile'].index, 'profile': rw_out['profile'].to_numpy()}), output_root / 'ruptures_window_profile.csv')

    primary_windows_legacy, point_to_window_audit, band_merge_audit, peak_width_zero_audit = build_primary_windows_from_ruptures(rw_out['points'], rw_out['profile'], settings.edge, settings.window_construction)
    edge_windows_legacy = build_edge_windows_from_ruptures(rw_out['profile'], primary_windows_legacy, settings.edge, settings.window_construction)

    point_df = build_point_table(rw_out['points'], point_to_window_audit)
    band_df = build_band_table(point_to_window_audit)
    main_windows_df = build_main_window_table(primary_windows_legacy, point_df, band_df)
    evidence_df = build_window_evidence_mapping(main_windows_df, point_df, band_df)
    window_catalog_df = build_window_catalog(primary_windows_legacy, edge_windows_legacy)

    if settings.output.emit_yearwise_outputs:
        yearwise = build_yearwise_diagnostics(profiles, smoothed['years'], main_windows_df, settings)
    else:
        yearwise = {
            'yearwise_primary_points': pd.DataFrame(),
            'yearwise_window_candidates': pd.DataFrame(),
            'yearwise_window_membership': pd.DataFrame(),
            'yearwise_detector_peak_summary': pd.DataFrame(),
            'yearwise_window_relation_audit': pd.DataFrame(),
            'yearwise_window_support_summary': pd.DataFrame(),
        }

    point_candidates_df, detector_local_peaks_df = build_point_audit_universe(point_df, main_windows_df, rw_out['profile'], settings)
    point_observed_stats_df, point_null_raw_df, point_null_df, point_null_detail_df, point_null_scale_df = run_point_null_significance(profiles, smoothed['years'], point_candidates_df, rw_out['profile'], settings)
    point_stability_df, point_bootstrap_df, point_param_df, yearwise_point_relation_df = run_point_stability_audit(
        profiles, smoothed['years'], point_candidates_df, state['state_matrix'], state['valid_day_mask'], yearwise['yearwise_primary_points'], settings
    )
    point_neighbor_pairs_df = build_neighbor_pairs(point_candidates_df, settings)
    point_competition_df, point_status_df = run_point_neighbor_competition(point_candidates_df, point_neighbor_pairs_df, point_bootstrap_df, yearwise_point_relation_df, settings)
    point_audit_df = summarize_point_significance(point_candidates_df, point_null_df, point_null_scale_df, point_stability_df, point_status_df, settings)
    point_summary = build_point_significance_summary(point_audit_df, point_null_scale_df, point_competition_df)

    point_btrack_reference_df = build_point_btrack_reference_points(
        point_candidates_df,
        point_neighbor_pairs_df,
        source_run_tag=settings.output.output_tag,
    )
    point_bootstrap_match_records_df = build_point_bootstrap_match_records(
        point_btrack_reference_df,
        point_bootstrap_df,
        point_neighbor_pairs_df,
        settings,
    )
    point_bootstrap_match_summary_df, point_bootstrap_offsets_df = summarize_point_bootstrap_matches(
        point_btrack_reference_df,
        point_bootstrap_match_records_df,
        point_neighbor_pairs_df,
    )
    point_neighbor_match_audit_df = build_point_neighbor_match_audit(
        point_bootstrap_match_records_df,
        point_neighbor_pairs_df,
    )
    point_yearwise_support_summary_df, point_yearwise_hits_long_df = build_point_yearwise_support_summary(
        point_btrack_reference_df,
        yearwise_point_relation_df,
    )
    point_parameter_path_support_df = build_point_parameter_path_support(
        point_btrack_reference_df,
        point_param_df,
        point_neighbor_pairs_df,
        settings,
    )
    point_robust_support_audit_df = build_point_robust_support_audit(
        point_btrack_reference_df,
        point_bootstrap_match_summary_df,
        point_yearwise_support_summary_df,
        point_parameter_path_support_df,
        point_audit_df,
        point_competition_df,
        settings,
    )
    point_btrack_role_summary_df = build_point_btrack_role_summary(point_robust_support_audit_df)
    point_btrack_summary = build_point_btrack_summary_json(
        point_robust_support_audit_df,
        meta={
            'source_run_tag': settings.output.output_tag,
            'uses_existing_point_stability_outputs': bool(settings.btrack_point.use_existing_point_stability_outputs),
            'point_layer_scope': 'formal_primary_points_plus_neighbor_competition_points',
            'btrack_backend_mode': 'reuse_existing_outputs' if settings.btrack_point.use_existing_point_stability_outputs else 'independent_recompute',
            'btrack_backend_independence': 'partial' if settings.btrack_point.use_existing_point_stability_outputs else 'full',
        },
    )

    bootstrap_matches_df, bootstrap_validity_df = run_bootstrap_window_matches(profiles, main_windows_df, settings)
    param_path_df = run_param_path_window_matches(state['state_matrix'], state['valid_day_mask'], main_windows_df, settings)
    permutation_df = run_global_permutation_audit(state['state_matrix'], state['valid_day_mask'], settings)
    support_audit_df = summarize_window_support(main_windows_df, bootstrap_matches_df, bootstrap_validity_df, param_path_df, permutation_df, settings)
    support_rule_comparison_df = build_support_rule_comparison(support_audit_df)
    retention_audit_df = build_window_retention_audit(main_windows_df, evidence_df, support_audit_df, settings)
    trust_tiers_df = build_audit_trust_tiers()
    summary = build_summary(
        main_windows_df,
        evidence_df,
        support_audit_df,
        retention_audit_df,
        support_rule_comparison_df,
        yearwise['yearwise_detector_peak_summary'],
        yearwise['yearwise_window_support_summary'],
        point_summary,
        point_btrack_summary,
    )

    write_dataframe(point_df, output_root / 'ruptures_primary_points.csv')
    write_dataframe(band_df, output_root / 'ruptures_support_bands.csv')
    write_dataframe(primary_windows_legacy, output_root / 'ruptures_primary_windows_legacy.csv')
    write_dataframe(edge_windows_legacy, output_root / 'ruptures_edge_windows_legacy.csv')
    write_dataframe(point_to_window_audit, output_root / 'ruptures_point_to_window_audit.csv')
    write_dataframe(band_merge_audit, output_root / 'ruptures_band_merge_audit.csv')
    write_dataframe(peak_width_zero_audit, output_root / 'peak_width_zero_audit.csv')

    write_dataframe(main_windows_df, output_root / 'stage_partition_main_windows.csv')
    write_dataframe(window_catalog_df, output_root / 'window_catalog.csv')
    write_dataframe(evidence_df, output_root / 'window_evidence_mapping.csv')
    write_dataframe(bootstrap_matches_df, output_root / 'window_support_bootstrap_matches.csv')
    write_dataframe(bootstrap_validity_df, output_root / 'window_support_sample_validity.csv')
    write_dataframe(param_path_df, output_root / 'window_support_param_path_matches.csv')
    write_dataframe(permutation_df, output_root / 'window_support_permutation_global.csv')
    write_dataframe(support_audit_df, output_root / 'window_support_audit.csv')
    write_dataframe(support_rule_comparison_df, output_root / 'support_rule_comparison.csv')
    write_dataframe(retention_audit_df, output_root / 'window_retention_audit.csv')
    write_dataframe(trust_tiers_df, output_root / 'audit_trust_tiers.csv')
    write_json(summary, output_root / 'summary.json')

    write_dataframe(yearwise['yearwise_primary_points'], output_root / 'yearwise_primary_points.csv')
    write_dataframe(yearwise['yearwise_window_candidates'], output_root / 'yearwise_window_candidates.csv')
    write_dataframe(yearwise['yearwise_window_membership'], output_root / 'yearwise_window_membership.csv')
    write_dataframe(yearwise['yearwise_detector_peak_summary'], output_root / 'yearwise_detector_peak_summary.csv')
    write_dataframe(yearwise['yearwise_window_relation_audit'], output_root / 'yearwise_window_relation_audit.csv')
    write_dataframe(yearwise['yearwise_window_support_summary'], output_root / 'yearwise_window_support_summary.csv')

    write_dataframe(detector_local_peaks_df, output_root / 'detector_local_peaks_all.csv')
    write_dataframe(point_candidates_df, output_root / 'point_candidates_audit.csv')
    write_dataframe(point_observed_stats_df, output_root / 'point_observed_statistics.csv')
    write_dataframe(point_null_raw_df, output_root / 'point_null_statistics_raw.csv')
    write_dataframe(point_null_df, output_root / 'point_null_significance.csv')
    write_dataframe(point_null_detail_df, output_root / 'point_null_detail.csv')
    write_dataframe(point_null_scale_df, output_root / 'point_null_scale_diagnostic.csv')
    write_dataframe(point_stability_df, output_root / 'point_stability_audit.csv')
    write_dataframe(point_bootstrap_df, output_root / 'point_bootstrap_presence.csv')
    write_dataframe(point_param_df, output_root / 'point_param_presence.csv')
    write_dataframe(yearwise_point_relation_df, output_root / 'yearwise_point_relation_audit.csv')
    write_dataframe(point_neighbor_pairs_df, output_root / 'point_neighbor_pairs.csv')
    write_dataframe(point_competition_df, output_root / 'point_neighbor_competition_audit.csv')
    write_dataframe(point_audit_df, output_root / 'point_significance_audit.csv')
    write_json(point_summary, output_root / 'point_significance_summary.json')
    write_json(point_summary, output_root / 'point_local_null_decomposition_summary.json')

    write_dataframe(point_btrack_reference_df, output_root / 'point_btrack_reference_points.csv')
    write_dataframe(point_bootstrap_match_records_df, output_root / 'point_bootstrap_match_records.csv')
    write_dataframe(point_bootstrap_match_summary_df, output_root / 'point_bootstrap_match_summary.csv')
    write_dataframe(point_neighbor_match_audit_df, output_root / 'point_neighbor_match_audit.csv')
    write_dataframe(point_bootstrap_offsets_df, output_root / 'point_bootstrap_offsets.csv')
    write_dataframe(point_yearwise_support_summary_df, output_root / 'point_yearwise_support_summary.csv')
    write_dataframe(point_yearwise_hits_long_df, output_root / 'point_yearwise_hits_long.csv')
    write_dataframe(point_parameter_path_support_df, output_root / 'point_parameter_path_support.csv')
    write_dataframe(point_robust_support_audit_df, output_root / 'point_robust_support_audit.csv')
    write_dataframe(point_btrack_role_summary_df, output_root / 'point_btrack_role_summary.csv')
    write_json(point_btrack_summary, output_root / 'point_btrack_summary.json')

    if settings.output.write_plots:
        plot_detector_profile(rw_out['profile'], output_root)
        plot_main_windows(main_windows_df, output_root)

    run_meta = {
        'status': 'success',
        'started_at_utc': started_at,
        'ended_at_utc': _now_utc(),
        'layer_name': 'stage_partition',
        'version_name': 'V3',
        'run_label': settings.output.output_tag,
        'foundation_dependency': 'V1',
        'smoothed_fields_path': str(smoothed_path),
        'notes': [
            'V3 patch D-1 repairs point-level null-significance by comparing observed and null peaks on the same extracted local-peak scale.',
            'E-1 keeps D-1 closed competition and further aligns observed/null point statistics on the same extracted peak scale.',
            'E-2 decomposes local null testing into peak-existence frequency and conditional peak-strength comparison.',
            'J patch refines the point-layer B-track by hard-separating headline formal primary points from neighbor competition points, downgrading parameter-path to an auxiliary role, and making backend reuse status explicit.',
            'stage_partition_main_windows.csv remains the only formal downstream window-object table.',
        ],
    }
    write_json(run_meta, output_root / 'run_meta.json')
    return {'output_root': output_root, 'summary': summary, 'point_summary': point_summary}
