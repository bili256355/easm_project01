from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
V7_SRC = V7_ROOT / "src"
V6_SRC = V7_ROOT.parent / "V6" / "src"
for p in [V7_SRC, V6_SRC]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stage_partition_v7.w45_allfield_process_relation_layer import (
    run_w45_allfield_process_relation_layer_v7_n,
)


if __name__ == "__main__":
    run_w45_allfield_process_relation_layer_v7_n(V7_ROOT)
