from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_summary(audit_df: pd.DataFrame, role_summary_df: pd.DataFrame, reference_df: pd.DataFrame, competition_df: pd.DataFrame, bootstrap_meta_df: pd.DataFrame | None = None) -> dict:
    formal_df = audit_df[audit_df['point_role_group'] == 'formal_primary'].copy() if audit_df is not None and not audit_df.empty else pd.DataFrame()
    neighbor_df = audit_df[audit_df['point_role_group'] == 'neighbor_competition'].copy() if audit_df is not None and not audit_df.empty else pd.DataFrame()
    return {
        'layer_name': 'stage_partition',
        'version_name': 'V4',
        'run_scope': 'point_layer_btrack_only',
        'btrack_backend_mode': 'independent_recompute',
        'btrack_backend_independence': 'full',
        'no_atrack': True,
        'no_window_layer': True,
        'n_reference_points': int(len(reference_df)) if reference_df is not None else 0,
        'n_formal_primary_points': int(len(formal_df)),
        'n_neighbor_competition_points': int(len(neighbor_df)),
        'n_neighbor_pairs': int(len(competition_df)) if competition_df is not None else 0,
        'formal_primary_counts': {
            'robust_primary_point': int((formal_df['judgement'] == 'robust_primary_point').sum()) if not formal_df.empty else 0,
            'supported_primary_point': int((formal_df['judgement'] == 'supported_primary_point').sum()) if not formal_df.empty else 0,
            'primary_point_with_caution': int((formal_df['judgement'] == 'primary_point_with_caution').sum()) if not formal_df.empty else 0,
            'weak_neighbor_peak': int((formal_df['judgement'] == 'weak_neighbor_peak').sum()) if not formal_df.empty else 0,
        },
        'neighbor_competition_counts': {
            'ambiguous_neighbor_pair': int((neighbor_df['judgement'] == 'ambiguous_neighbor_pair').sum()) if not neighbor_df.empty else 0,
            'neighbor_candidate_with_support': int((neighbor_df['judgement'] == 'neighbor_candidate_with_support').sum()) if not neighbor_df.empty else 0,
            'weak_neighbor_peak': int((neighbor_df['judgement'] == 'weak_neighbor_peak').sum()) if not neighbor_df.empty else 0,
        },
        'bootstrap_replicates_effective': int((bootstrap_meta_df['status'] == 'success').sum()) if bootstrap_meta_df is not None and not bootstrap_meta_df.empty and 'status' in bootstrap_meta_df.columns else None,
        'candidate_universe_deduplicated': True,
        'yearwise_exact_definition': 'strict_only',
    }
