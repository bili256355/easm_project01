from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
V7_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = V7_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v7.progress_order_failure_audit import run_progress_order_failure_audit_v7_e2


if __name__ == "__main__":
    run_progress_order_failure_audit_v7_e2(v7_root=V7_ROOT)
