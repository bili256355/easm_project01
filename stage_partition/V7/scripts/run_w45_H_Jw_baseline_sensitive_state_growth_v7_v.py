from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = V7_ROOT / "src"
V6_ROOT = V7_ROOT.parents[0] / "V6_1"
V6_SRC_ROOT = V6_ROOT / "src"
for p in (SRC_ROOT, V6_SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stage_partition_v7.w45_H_Jw_baseline_sensitive_state_growth import (  # noqa: E402
    run_w45_H_Jw_baseline_sensitive_state_growth_v7_v,
)


if __name__ == "__main__":
    run_w45_H_Jw_baseline_sensitive_state_growth_v7_v(V7_ROOT)
