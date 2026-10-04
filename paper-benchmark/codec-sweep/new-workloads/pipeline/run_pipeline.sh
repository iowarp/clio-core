#!/bin/bash
# Full producer-consumer runs: the four workloads, three policies (none / best
# single / per-chunk), 3 reps each, after every GPU sweep has finished so
# nothing else shares the GPU or the disk.
#   run_pipeline.sh STORE_DIR [OUT_DIR] [workloads...]
# STORE_DIR must be on the storage device under test and have room for the
# largest uncompressed run (~21 GB for VPIC and LAMMPS); its *.bin files are
# deleted between runs. Needs passwordless sudo to drop the page cache.
set -u
STORE=${1:?usage: run_pipeline.sh STORE_DIR [OUT_DIR] [workloads...]}
OUT=${2:-$HOME/np-pipeline}
shift $(( $# >= 2 ? 2 : 1 ))
WORKLOADS=${*:-fem nanoaod vpic lammps}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$HOME/builds/logs/pipeline.log
until grep -q ASTRO_SWEEP_DONE "$HOME"/builds/logs/astro_after*.log 2>/dev/null \
      && ! pgrep -x corpus_sweep >/dev/null; do sleep 120; done
mkdir -p "$STORE" "$OUT"
export OPENBLAS_NUM_THREADS=16
for w in $WORKLOADS; do
  echo "== $w start $(date +%T)" | tee -a "$LOG"
  ~/np-venv/bin/python "$HERE/pipeline.py" --workload "$w" --store "$STORE/$w" \
      --out "$OUT/$w" --reps 3 > "$OUT/$w.log" 2>&1
  echo "   rc=$? end $(date +%T)" | tee -a "$LOG"
done
~/np-venv/bin/python "$HERE/summarize.py" --out "$OUT" | tee "$OUT/summary.txt"
echo PIPELINE_DONE | tee -a "$LOG"
