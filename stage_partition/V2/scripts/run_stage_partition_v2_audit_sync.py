from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v2.audit_sync_config import AuditSyncSettings
from stage_partition_v2.audit_sync_pipeline import run_stage_partition_v2_audit_sync


if __name__ == '__main__':
    settings = AuditSyncSettings()
    print(f'[audit_sync] source_results_dir = {settings.source_results_dir()}')
    print(f'[audit_sync] output_root       = {settings.output_root()}')
    result = run_stage_partition_v2_audit_sync(settings)
    print(f'[audit_sync] done: {result["output_root"]}')
