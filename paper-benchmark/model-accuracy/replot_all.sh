#!/usr/bin/env bash
#===============================================================================
# replot_all.sh -- redraw every figure in figures/new-workloads/nn-v2 from the
# finished full-size runs (eval_v2_workloads.FULL_WORKLOADS):
#   v2_<ds>_fixed_vs_learn.png   per workload (best single codec vs learning)
#   v2_spectrum.png              opportunity, cost, measured time, ratio
#   v2_prediction_error.png      predictions vs the exhaustive measurements
#   v2_learning_over_time.png    20-pass learning replay (replay_learning.py,
#                                model as trained and Clio's rule)
# A workload whose runs are not finished yet is skipped.
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
for ds in $(python3 -c "
import os, eval_v2_workloads as ev
print(' '.join(d for d in ev.FULL_WORKLOADS if all(ev.run_finished(
    os.path.join(ev.RUNS, f'{d}_{m}_nolog')) for m in ('learn', 'fixed'))))"); do
  best=$(python3 baseline_store.py best "$ds") || continue
  python3 plot_fixed_vs_learn.py "$ds" --codec "$best" | tail -1
done
python3 plot_spectrum.py | head -1
python3 pred_error_real.py | tail -1
OPENBLAS_NUM_THREADS=4 ~/np-venv/bin/python replay_learning.py compare --passes 20 \
    --rules "frozen,nlms lr 0.5 (Clio)" 2>/dev/null | tail -2
python3 plot_learning_over_time.py
