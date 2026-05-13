from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = V7_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v7.w45_H_Jw_object_specific_v6style_window_detection import (  # noqa: E402
    run_w45_H_Jw_object_specific_v6style_window_detection_v7_w,
)


if __name__ == "__main__":
    run_w45_H_Jw_object_specific_v6style_window_detection_v7_w(V7_ROOT)
