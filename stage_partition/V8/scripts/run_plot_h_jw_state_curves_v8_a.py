from pathlib import Path
import sys

V8_ROOT = Path(__file__).resolve().parents[1]
SRC = V8_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v8.plot_h_jw_state_curves_v8_a import run_plot_h_jw_state_curves_v8_a

if __name__ == '__main__':
    out = run_plot_h_jw_state_curves_v8_a(V8_ROOT)
    print(f'Plot output written to: {out}')
