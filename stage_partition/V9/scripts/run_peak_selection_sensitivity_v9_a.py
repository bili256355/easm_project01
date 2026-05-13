from pathlib import Path
import sys

V9_ROOT = Path(r"D:\easm_project01\stage_partition\V9")
SRC = V9_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v9.peak_selection_sensitivity_v9_a import run_peak_selection_sensitivity_v9_a

if __name__ == "__main__":
    run_peak_selection_sensitivity_v9_a(V9_ROOT)
