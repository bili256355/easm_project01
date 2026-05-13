from pathlib import Path
import sys

V91_ROOT = Path(__file__).resolve().parents[1]
SRC = V91_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v9_1.eof_transition_mode_audit_v9_1_d import run_eof_transition_mode_audit_v9_1_d

if __name__ == "__main__":
    run_eof_transition_mode_audit_v9_1_d(V91_ROOT)
