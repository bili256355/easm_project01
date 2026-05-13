from __future__ import annotations
import itertools
import pandas as pd
from .config import RupturesWindowConfig
from .detector_ruptures_window import run_point_detector
from .point_reference import build_formal_reference_points
from .point_audit_universe import build_point_audit_universe
from .point_matching import match_reference_points, summarize_match_records


def _clone_detector_cfg(base_cfg: RupturesWindowConfig, width: int, pen: float | None = None) -> RupturesWindowConfig:
    return RupturesWindowConfig(
        width=width,
        model=base_cfg.model,
        min_size=base_cfg.min_size,
        jump=base_cfg.jump,
        selection_mode=base_cfg.selection_mode,
        fixed_n_bkps=base_cfg.fixed_n_bkps,
        pen=base_cfg.pen if pen is None else float(pen),
        epsilon=base_cfg.epsilon,
        local_peak_min_distance_days=base_cfg.local_peak_min_distance_days,
        nearest_peak_search_radius_days=base_cfg.nearest_peak_search_radius_days,
    )


def run_point_parameter_path_support(state_matrix, valid_day_mask, valid_day_index, reference_points_df, settings, pair_df=None):
    records = []
    grid = list(itertools.product(settings.parameter_path.bandwidth_values, settings.parameter_path.pen_values))
    state_valid = state_matrix[valid_day_mask, :]
    for rep_id, (width, pen) in enumerate(grid):
        det = run_point_detector(
            state_valid,
            valid_day_index,
            _clone_detector_cfg(settings.detector, int(width), float(pen)),
            local_peak_distance_days=settings.detector.local_peak_min_distance_days,
        )
        formal_ref = build_formal_reference_points(det['primary_points_df'], source_run_tag=settings.output.output_tag)
        audit_universe_df = build_point_audit_universe(
            formal_ref,
            det['local_peaks_df'],
            det['profile'],
            radius_days=settings.point_audit_universe.radius_days,
            weak_peak_min_prominence=settings.point_audit_universe.weak_peak_min_prominence,
            source_run_tag=settings.output.output_tag,
        )
        rec = match_reference_points(
            reference_points_df,
            candidate_universe_df=audit_universe_df,
            replicate_id=int(rep_id),
            replicate_kind='param_path',
            match_cfg=settings.matching,
            pair_df=pair_df,
        )
        rec['path_width'] = int(width)
        rec['path_pen'] = float(pen)
        rec['path_config_signature'] = f"w{int(width)}_pen{float(pen):.2f}"
        records.append(rec)
    records_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    summary_df, _ = summarize_match_records(reference_points_df, records_df, pair_df=pair_df)
    if not summary_df.empty:
        summary_df = summary_df.rename(
            columns={
                'bootstrap_match_rate': 'path_presence_rate',
                'bootstrap_strict_match_rate': 'path_strict_match_rate',
                'bootstrap_near_support_rate': 'path_near_support_rate',
                'ambiguous_match_rate': 'path_ambiguous_rate',
                'offset_q05': 'path_offset_min',
                'offset_q95': 'path_offset_max',
                'n_replicates': 'n_path_configs',
            }
        )
        summary_df = summary_df[
            [
                'point_id',
                'path_presence_rate',
                'path_strict_match_rate',
                'path_near_support_rate',
                'path_ambiguous_rate',
                'path_offset_min',
                'path_offset_max',
                'n_path_configs',
            ]
        ]
    return {'records_df': records_df, 'summary_df': summary_df}
