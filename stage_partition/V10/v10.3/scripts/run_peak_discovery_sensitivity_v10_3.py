from pathlib import Path
import importlib.util
import sys

SCRIPT = Path(__file__).resolve()
BUNDLE_ROOT = SCRIPT.parents[1]
SRC = BUNDLE_ROOT / "src" / "peak_discovery_sensitivity_v10_3.py"

spec = importlib.util.spec_from_file_location("peak_discovery_sensitivity_v10_3", SRC)
mod = importlib.util.module_from_spec(spec)
# Register before exec_module so @dataclass in the dynamically loaded module can resolve __module__.
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

if __name__ == "__main__":
    mod.run_peak_discovery_sensitivity_v10_3(BUNDLE_ROOT)
