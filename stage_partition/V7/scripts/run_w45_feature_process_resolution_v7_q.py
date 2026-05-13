from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
V6_ROOT = V7_ROOT.parent / "V6"
for _p in (V7_ROOT / "src", V6_ROOT / "src"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from stage_partition_v7.w45_feature_process_resolution import run_w45_feature_process_resolution_v7_q


if __name__ == "__main__":
    run_w45_feature_process_resolution_v7_q(V7_ROOT)
