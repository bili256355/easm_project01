from pathlib import Path
import sys

V9_ROOT = Path(__file__).resolve().parents[1]
SRC = V9_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v9.peak_all_windows_v9_a import run_peak_all_windows_v9_a


if __name__ == "__main__":
    run_peak_all_windows_v9_a(V9_ROOT)
