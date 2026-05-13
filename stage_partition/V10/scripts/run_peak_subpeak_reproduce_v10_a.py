from pathlib import Path
import sys

V10_ROOT = Path(__file__).resolve().parents[1]
SRC = V10_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v10 import run_peak_subpeak_reproduce_v10_a

if __name__ == "__main__":
    run_peak_subpeak_reproduce_v10_a(V10_ROOT)
