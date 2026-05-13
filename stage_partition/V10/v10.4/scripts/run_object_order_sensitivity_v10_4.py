from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = BUNDLE_ROOT / "src" / "object_order_sensitivity_v10_4.py"
MODULE_NAME = "object_order_sensitivity_v10_4"

spec = importlib.util.spec_from_file_location(MODULE_NAME, SRC_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SRC_PATH}")
mod = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = mod
spec.loader.exec_module(mod)
mod.run_object_order_sensitivity_v10_4(BUNDLE_ROOT)
