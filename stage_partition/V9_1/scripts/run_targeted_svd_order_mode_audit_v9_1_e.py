from pathlib import Path
import sys

V91_ROOT = Path(__file__).resolve().parents[1]
SRC = V91_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v9_1.targeted_svd_order_mode_audit_v9_1_e import run_targeted_svd_order_mode_audit_v9_1_e

if __name__ == "__main__":
    run_targeted_svd_order_mode_audit_v9_1_e(V91_ROOT)
