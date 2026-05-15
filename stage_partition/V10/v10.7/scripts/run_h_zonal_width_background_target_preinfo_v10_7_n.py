from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
V10_7_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = V10_7_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v10_7.h_zonal_width_background_target_preinfo_pipeline import (  # noqa: E402
    Settings,
    run_h_zonal_width_background_target_preinfo_v10_7_n,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V10.7_n: H_zonal_width local-vs-background, P target "
            "redefinition, and weak preinformation-value audit."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\easm_project01"))
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--n-random-windows", type=int, default=1000)
    parser.add_argument("--group-frac", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=20260515)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--experiment-a-policy",
        choices=["full", "screen", "skip_heavy"],
        default="full",
        help=(
            "HOTFIX01 performance control for experiment A. full preserves formal resampling; "
            "screen uses screen_n_perm/screen_n_boot for experiment-A screens and skips formal delta resampling; "
            "skip_heavy skips expensive incremental-CV and sliding-window screens."
        ),
    )
    parser.add_argument("--screen-n-perm", type=int, default=199)
    parser.add_argument("--screen-n-boot", type=int, default=199)
    parser.add_argument(
        "--smoothed-fields",
        type=Path,
        default=None,
        help="Optional override for smoothed_fields.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        project_root=args.project_root,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        n_random_windows=args.n_random_windows,
        group_frac=args.group_frac,
        random_seed=args.random_seed,
        progress=args.progress,
        experiment_a_policy=args.experiment_a_policy,
        screen_n_perm=args.screen_n_perm,
        screen_n_boot=args.screen_n_boot,
        smoothed_fields_path_override=args.smoothed_fields,
    )
    run_h_zonal_width_background_target_preinfo_v10_7_n(settings)


if __name__ == "__main__":
    main()
