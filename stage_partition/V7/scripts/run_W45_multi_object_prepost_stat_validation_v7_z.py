from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = V7_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stage_partition_v7.W45_multi_object_prepost_stat_validation_v7_z import (  # noqa: E402
    run_W45_multi_object_prepost_stat_validation_v7_z,
)


if __name__ == "__main__":
    run_W45_multi_object_prepost_stat_validation_v7_z(V7_ROOT)
