from pathlib import Path
import sys

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = BUNDLE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from object_native_peak_discovery_v10_2 import run_object_native_peak_discovery_v10_2

if __name__ == "__main__":
    result = run_object_native_peak_discovery_v10_2(BUNDLE_ROOT)
    print("[V10.2] object-native peak discovery completed")
    print(f"[V10.2] output_root = {result['output_root']}")
