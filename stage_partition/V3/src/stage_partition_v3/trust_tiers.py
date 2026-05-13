from __future__ import annotations

import pandas as pd


def build_audit_trust_tiers() -> pd.DataFrame:
    rows = [
        {
            'artifact_name': 'stage_partition_main_windows.csv',
            'artifact_type': 'main_window_table',
            'scope': 'window_object',
            'trust_tier': 'A_main_basis',
            'can_be_used_as_main_basis': True,
            'must_be_qualified_when_cited': False,
            'cannot_be_used_for': 'window_internal_order_or_pathway',
            'notes': '唯一主窗口对象主表',
        },
        {
            'artifact_name': 'window_evidence_mapping.csv',
            'artifact_type': 'evidence_mapping',
            'scope': 'point_and_band_to_window',
            'trust_tier': 'B_supporting_evidence',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'replace_main_window_table',
            'notes': 'point和band只作窗口支撑证据',
        },
        {
            'artifact_name': 'window_support_audit.csv',
            'artifact_type': 'support_audit',
            'scope': 'window_level_audit',
            'trust_tier': 'B_supporting_evidence',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'direct_physical_interpretation',
            'notes': 'support分数必须与样本口径一起引用',
        },
        {
            'artifact_name': 'window_retention_audit.csv',
            'artifact_type': 'retention_audit',
            'scope': 'window_level_audit',
            'trust_tier': 'C_restricted_audit_only',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'defining_main_windows',
            'notes': 'V3 retention按window-native解释，但仍属审计层',
        },
        {
            'artifact_name': 'V2 legacy outputs',
            'artifact_type': 'legacy_outputs',
            'scope': 'historical_only',
            'trust_tier': 'D_legacy_do_not_use',
            'can_be_used_as_main_basis': False,
            'must_be_qualified_when_cited': True,
            'cannot_be_used_for': 'V3 mainline conclusions',
            'notes': 'V2旧结果表在V3中不再作为主结果引用',
        },
    ]
    return pd.DataFrame(rows)
