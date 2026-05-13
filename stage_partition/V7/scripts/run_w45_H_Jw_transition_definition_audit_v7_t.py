from __future__ import annotations

import sys
from pathlib import Path


def _find_v7_root() -> Path:
    here = Path(__file__).resolve()
    # Expected: .../stage_partition/V7/scripts/this_file.py
    return here.parents[1]


V7_ROOT = _find_v7_root()
V6_ROOT = V7_ROOT.parent / "V6"
for p in [V7_ROOT / "src", V6_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stage_partition_v7.w45_H_Jw_transition_definition_audit import (  # noqa: E402
    run_w45_H_Jw_transition_definition_audit_v7_t,
)


if __name__ == "__main__":
    run_w45_H_Jw_transition_definition_audit_v7_t(V7_ROOT)
