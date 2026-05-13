from pathlib import Path
import sys

V7_ROOT = Path(__file__).resolve().parents[1]
SRC = V7_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v7.W45_Je_physical_variance_audit_v7_z_c import run_W45_Je_physical_variance_audit_v7_z_c

if __name__ == '__main__':
    run_W45_Je_physical_variance_audit_v7_z_c(V7_ROOT)
