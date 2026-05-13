from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SRC = BUNDLE_ROOT / "src" / "strength_curve_export_v10_5_e.py"
MODULE_NAME = "strength_curve_export_v10_5_e"

spec = importlib.util.spec_from_file_location(MODULE_NAME, SRC)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SRC}")
mod = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = mod
spec.loader.exec_module(mod)
mod.run_strength_curve_export_v10_5_e(BUNDLE_ROOT)
