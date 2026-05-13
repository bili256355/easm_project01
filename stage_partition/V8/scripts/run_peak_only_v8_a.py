from pathlib import Path
import sys

V8_ROOT = Path(__file__).resolve().parents[1]
SRC = V8_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v8.peak_only_v8_a import run_peak_only_v8_a


if __name__ == "__main__":
    run_peak_only_v8_a(V8_ROOT)
