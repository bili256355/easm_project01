from pathlib import Path
import sys

V91_ROOT = Path(__file__).resolve().parents[1]
SRC = V91_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v9_1.peak_yearwise_heterogeneity_v9_1_a import run_peak_yearwise_heterogeneity_v9_1_a


if __name__ == "__main__":
    run_peak_yearwise_heterogeneity_v9_1_a(V91_ROOT)
