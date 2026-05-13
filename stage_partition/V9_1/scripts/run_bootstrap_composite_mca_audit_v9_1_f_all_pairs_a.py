from pathlib import Path
import sys

V91_ROOT = Path(__file__).resolve().parents[1]
SRC = V91_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v9_1.bootstrap_composite_mca_audit_v9_1_f_all_pairs_a import (
    run_bootstrap_composite_mca_audit_v9_1_f_all_pairs_a,
)

if __name__ == "__main__":
    run_bootstrap_composite_mca_audit_v9_1_f_all_pairs_a(V91_ROOT)
