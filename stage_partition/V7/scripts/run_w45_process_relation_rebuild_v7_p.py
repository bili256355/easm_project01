from __future__ import annotations

import sys
from pathlib import Path

# Expected location: D:/easm_project01/stage_partition/V7/scripts/run_*.py
V7_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = V7_ROOT.parent
V6_ROOT = STAGE_ROOT / "V6"

for p in [V7_ROOT / "src", V6_ROOT / "src"]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from stage_partition_v7.w45_process_relation_rebuild import run_w45_process_relation_rebuild_v7_p


if __name__ == "__main__":
    run_w45_process_relation_rebuild_v7_p(V7_ROOT)
