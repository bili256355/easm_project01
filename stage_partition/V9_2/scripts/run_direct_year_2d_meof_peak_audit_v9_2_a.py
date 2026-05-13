from pathlib import Path
import sys

V92_ROOT = Path(__file__).resolve().parents[1]
SRC = V92_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v9_2.direct_year_2d_meof_peak_audit_v9_2_a import (
    run_direct_year_2d_meof_peak_audit_v9_2_a,
)

if __name__ == "__main__":
    run_direct_year_2d_meof_peak_audit_v9_2_a(V92_ROOT)
