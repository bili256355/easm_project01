from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
_V4_ROOT = _THIS_FILE.parents[1]
_SRC_DIR = _V4_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


from stage_partition_v4.pipeline import run_stage_partition_v4

if __name__ == "__main__":
    run_stage_partition_v4()
