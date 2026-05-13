from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
V7_ROOT = SCRIPT_PATH.parents[1]
SRC = V7_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v7.prepost_process_curve_v7_z_process_a import (  # noqa: E402
    run_prepost_process_curve_v7_z_process_a,
)

if __name__ == "__main__":
    run_prepost_process_curve_v7_z_process_a(V7_ROOT)
