from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
LAYER_ROOT = THIS_FILE.parents[1]
SRC_ROOT = LAYER_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from index_validity_v1_b import IndexValidityV1BSettings, run_index_validity_window_family_guardrail_v1_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Run index_validity V1_b window-family guardrail.")
    parser.add_argument(
        "--data-mode",
        choices=["smoothed", "anomaly"],
        default="smoothed",
        help="Main/default is smoothed: smoothed_fields + index_values_smoothed. Anomaly is auxiliary only.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional output tag. Defaults to mode-aware tag from settings.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Optional speed/debug mode: compute all CSV metrics but skip diagnostic figures. Default remains unchanged.",
    )
    parser.add_argument(
        "--no-cartopy",
        action="store_true",
        help="Optional speed/debug mode: draw plain lat-lon figures instead of cartopy maps. Default remains unchanged.",
    )
    parser.add_argument(
        "--max-figures",
        type=int,
        default=None,
        help="Optional cap for diagnostic figures. Default remains settings.max_diagnostic_figures.",
    )
    args = parser.parse_args()

    default = IndexValidityV1BSettings()
    output_tag = args.output_tag
    if output_tag is None:
        output_tag = default.output_tag if args.data_mode == "smoothed" else "window_family_guardrail_v1_b_anomaly_aux_a"

    settings_kwargs = {
        "data_mode": args.data_mode,
        "output_tag": output_tag,
        "make_figures": not args.tables_only,
        "use_cartopy_if_available": not args.no_cartopy,
    }
    if args.max_figures is not None:
        settings_kwargs["max_diagnostic_figures"] = int(args.max_figures)
    settings = IndexValidityV1BSettings(**settings_kwargs)
    summary = run_index_validity_window_family_guardrail_v1_b(settings)
    print("index_validity V1_b window-family guardrail completed")
    print(f"status={summary.get('status')}")
    print(f"data_mode={settings.data_mode}")
    print(f"output_dir={settings.output_dir}")
    print(f"family_collapse_risk_counts={summary.get('family_collapse_risk_counts')}")
    print(f"runtime_task_timing={settings.tables_dir / 'runtime_task_timing.csv'}")


if __name__ == "__main__":
    main()
