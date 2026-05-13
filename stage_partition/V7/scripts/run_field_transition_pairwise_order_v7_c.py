from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V7_ROOT = THIS_FILE.parents[1]
SRC = V7_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v7.field_order_pairwise import run_field_transition_pairwise_order_v7_c


if __name__ == "__main__":
    run_field_transition_pairwise_order_v7_c()
