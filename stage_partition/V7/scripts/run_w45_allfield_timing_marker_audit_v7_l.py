from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
V7_SRC = V7_ROOT / "src"
if str(V7_SRC) not in sys.path:
    sys.path.insert(0, str(V7_SRC))

from stage_partition_v7.w45_allfield_timing_marker_audit import run_w45_allfield_timing_marker_audit_v7_l


if __name__ == "__main__":
    run_w45_allfield_timing_marker_audit_v7_l(V7_ROOT)
