from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V7_ROOT = THIS_FILE.parents[1]
V6_SRC = V7_ROOT.parent / "V6" / "src"
V7_SRC = V7_ROOT / "src"
for p in [V6_SRC, V7_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stage_partition_v7.field_progress_timing import run_field_transition_progress_timing_v7_e


if __name__ == "__main__":
    run_field_transition_progress_timing_v7_e()
