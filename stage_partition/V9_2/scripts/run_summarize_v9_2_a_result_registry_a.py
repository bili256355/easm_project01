from pathlib import Path
import sys

V92_ROOT = Path(__file__).resolve().parents[1]
SRC = V92_ROOT / "src"
sys.path.insert(0, str(SRC))

from stage_partition_v9_2.summarize_v9_2_a_result_registry_a import (
    run_summarize_v9_2_a_result_registry_a,
)

if __name__ == "__main__":
    run_summarize_v9_2_a_result_registry_a(V92_ROOT)
