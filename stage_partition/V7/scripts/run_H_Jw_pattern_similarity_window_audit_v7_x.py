from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
V7_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = V7_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v7.H_Jw_pattern_similarity_window_audit import (  # noqa: E402
    run_H_Jw_pattern_similarity_window_audit_v7_x,
)


if __name__ == "__main__":
    run_H_Jw_pattern_similarity_window_audit_v7_x(V7_ROOT)
