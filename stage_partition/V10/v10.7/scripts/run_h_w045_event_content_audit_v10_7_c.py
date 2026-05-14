from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    # .../stage_partition/V10/v10.7/scripts/run_*.py -> project root parents[4]
    return here.parents[4]


def main() -> None:
    project_root = _resolve_project_root()
    src_root = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from stage_partition_v10_7.event_content_pipeline import run_h_w045_event_content_audit_v10_7_c

    meta = run_h_w045_event_content_audit_v10_7_c(project_root)
    print("V10.7_c H W045 event-content audit completed.")
    print(f"Output root: {meta.get('output_root')}")
    print(f"Profile status: {meta.get('input_status', {}).get('profile_status')}")
    print(f"Spatial status: {meta.get('input_status', {}).get('spatial_status')}")
    print(f"Yearwise status: {meta.get('input_status', {}).get('yearwise_status')}")


if __name__ == "__main__":
    main()
