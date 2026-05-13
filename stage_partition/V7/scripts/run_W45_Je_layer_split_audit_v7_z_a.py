from __future__ import annotations

import sys
from pathlib import Path

V7_ROOT = Path(__file__).resolve().parents[1]
SRC = V7_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v7.W45_Je_layer_split_audit_v7_z_a import (  # noqa: E402
    run_W45_Je_layer_split_audit_v7_z_a,
)

if __name__ == "__main__":
    run_W45_Je_layer_split_audit_v7_z_a(V7_ROOT)
