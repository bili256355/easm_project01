from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
V7_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = V7_ROOT / "src"

if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v7.w45_H_timing_marker_audit import run_w45_H_timing_marker_audit_v7_k


if __name__ == "__main__":
    run_w45_H_timing_marker_audit_v7_k(v7_root=V7_ROOT)
