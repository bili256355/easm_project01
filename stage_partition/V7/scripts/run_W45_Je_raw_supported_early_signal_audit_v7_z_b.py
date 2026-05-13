from pathlib import Path
import sys

V7_ROOT = Path(__file__).resolve().parents[1]
SRC = V7_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v7.W45_Je_raw_supported_early_signal_audit_v7_z_b import (  # noqa: E402
    run_W45_Je_raw_supported_early_signal_audit_v7_z_b,
)

if __name__ == "__main__":
    run_W45_Je_raw_supported_early_signal_audit_v7_z_b(V7_ROOT)
