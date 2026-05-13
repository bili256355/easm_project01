from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SIBLING_V6_SRC = ROOT.parent / "V6" / "src"

for p in (SRC, SIBLING_V6_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stage_partition_v7.field_timing import run_field_transition_timing_v7_a


if __name__ == "__main__":
    run_field_transition_timing_v7_a()
