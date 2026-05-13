from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = BUNDLE_ROOT / "src" / "field_index_validation_v10_5_a.py"
MODULE_NAME = "field_index_validation_v10_5_a"

spec = importlib.util.spec_from_file_location(MODULE_NAME, SRC_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load module from {SRC_PATH}")
mod = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = mod
spec.loader.exec_module(mod)
mod.run_field_index_validation_v10_5_a(BUNDLE_ROOT)
