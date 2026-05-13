from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = V7_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v7.H_Jw_pattern_only_detector_v7_y import (  # noqa: E402
    run_H_Jw_pattern_only_detector_v7_y,
)


if __name__ == "__main__":
    run_H_Jw_pattern_only_detector_v7_y(V7_ROOT)
