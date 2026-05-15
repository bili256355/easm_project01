from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _default_project_root() -> Path:
    # script: <root>/stage_partition/V10/v10.8/scripts/run_*.py
    return Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V10.8_a object-internal transition content audit: "
            "reconstruct detector flanks for V10.2 object-native breakpoints and "
            "decompose what object-internal content changed."
        )
    )
    parser.add_argument("--project-root", type=Path, default=_default_project_root())
    parser.add_argument("--smoothed-fields", type=Path, default=None)
    parser.add_argument("--detector-width", type=int, default=None, help="Override V10.2 detector.width")
    parser.add_argument("--flank-half-width", type=int, default=None, help="Override reconstructed left/right half-width")
    parser.add_argument("--no-clean-output", action="store_true", help="Do not delete the existing V10.8_a output directory before running")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG figure generation")
    parser.add_argument("--no-root-log", action="store_true", help="Do not write root log files")
    parser.add_argument("--progress", action="store_true", default=True, help="Print progress messages")
    args = parser.parse_args()

    src_root = args.project_root / "stage_partition" / "V10" / "v10.8" / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from stage_partition_v10_8.config import V10_8_A_Settings
    from stage_partition_v10_8.pipeline import run_object_internal_transition_content_v10_8_a

    settings = V10_8_A_Settings(
        project_root=args.project_root,
        progress=bool(args.progress),
        smoothed_fields_path_override=args.smoothed_fields,
        detector_width_override=args.detector_width,
        flank_half_width_override=args.flank_half_width,
        clean_output=not args.no_clean_output,
        write_root_log=not args.no_root_log,
        make_figures=not args.no_figures,
    )
    run_meta = run_object_internal_transition_content_v10_8_a(settings)
    print(f"[V10.8_a] Done. Output: {run_meta['settings']['project_root']}/stage_partition/V10/v10.8/outputs/object_internal_transition_content_v10_8_a", flush=True)


if __name__ == "__main__":
    main()
