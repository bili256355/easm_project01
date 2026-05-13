from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = V7_ROOT.parent
for p in [V7_ROOT / "src", STAGE_ROOT / "V6" / "src"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from stage_partition_v7.w45_process_curve_foundation_audit import run_w45_process_curve_foundation_audit_v7_o

if __name__ == "__main__":
    run_w45_process_curve_foundation_audit_v7_o(V7_ROOT)
