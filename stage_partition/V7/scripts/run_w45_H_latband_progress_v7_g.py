from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
V7_ROOT = SCRIPT_PATH.parents[1]
STAGE_PARTITION_ROOT = V7_ROOT.parent
SRC_ROOT = V7_ROOT / "src"
V6_SRC_ROOT = STAGE_PARTITION_ROOT / "V6" / "src"

for path in (SRC_ROOT, V6_SRC_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage_partition_v7.w45_H_latband_progress import run_w45_H_latband_progress_v7_g


if __name__ == "__main__":
    run_w45_H_latband_progress_v7_g(v7_root=V7_ROOT)
