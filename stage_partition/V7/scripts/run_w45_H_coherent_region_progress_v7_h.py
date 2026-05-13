from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V7_ROOT = THIS_FILE.parents[1]
STAGE_ROOT = V7_ROOT.parent
for p in [V7_ROOT / "src", STAGE_ROOT / "V6" / "src"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from stage_partition_v7.w45_H_coherent_region_progress import run_w45_H_coherent_region_progress_v7_h


if __name__ == "__main__":
    run_w45_H_coherent_region_progress_v7_h(V7_ROOT)
