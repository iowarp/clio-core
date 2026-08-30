#!/usr/bin/env bash
#SBATCH --job-name=np-nyx-qual
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bekn/imuradli/np-qual/slurm-%j.out
#
# The smoke cell under run_benchmark.sh's NEW defaults: K=31, balance only,
# lossy only, mandatory error bound. One cell, not four.
#
# Called through nyx/run_config.sh rather than run_benchmark.sh because
# config_for() prefers run_config_insitu.sh whenever it is executable, and the
# nyx in-situ route needs libclio_nyx_insitu.so (not built) plus a Nyx carrying
# the in-situ patch -- this Nyx has only the raw-field-dump one. The arguments
# below are exactly what run_benchmark.sh --smoke printed under --dry-run,
# minus the three (--ncell/--steps/--int) that only the in-situ route accepts;
# the replay route takes its size from the dumps on disk instead.
#
# 10 minutes, not 5: K=31 measures 32 actions per chunk instead of 4, so the
# compression work is ~8x the K=3 cell that ran in 9 s. 126 chunks x 32
# actions = ~4,032 measured candidates.
set -uo pipefail

REPO=/u/imuradli/clio-core
PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

FIELDS=/scratch/bekn/imuradli/np-smoke/fields/nyx   # already generated
ROOT=/projects/bekn/imuradli/np-qual
RESULTS=$ROOT/results
mkdir -p "$RESULTS"

say() { echo "[$(date +%H:%M:%S)] $*"; }
say "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | sed 's/^/       gpu: /'
[ -d "$FIELDS" ] || { say "no field dumps at $FIELDS"; exit 1; }
say "fields: $(find "$FIELDS" -name '*.f32' | wc -l) files, $(du -sh "$FIELDS" | cut -f1)"
echo

tag=nyx_lossy_balance_nvme_5GBs
say "== $tag  (K=31, THRESH=-1, eb=1e-3, balance, 5 GB/s)"
env CLIO_NEUROPRESS_STAGE_H2D=1 \
    "$PB/nyx/run_config.sh" explore-balance \
      --fields "$FIELDS" --bw 5e6 --eb 1e-3 \
      --explore-k 31 --explore-thresh -1 --chunk 8388608 \
      --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
rc=${PIPESTATUS[0]}
rm -f "$RESULTS/$tag/chi_bdev.dat" "$RESULTS/$tag"/cte_tier.dat*
echo; say "rc=$rc"

# What the new settings were supposed to buy: the WHOLE action space, measured.
if [ -f "$RESULTS/$tag/explore.csv" ]; then
  say "== measured coverage"
  python3 - "$RESULTS/$tag/explore.csv" <<'PY'
import csv, sys
from collections import Counter
r = list(csv.DictReader(open(sys.argv[1])))
acts = {(x['lib_name'], x['shuffle'], x['quantize']) for x in r}
per = Counter(x['blob'] for x in r)
q = Counter(x['quantize'] for x in r)
print(f"   rows={len(r)}  chunks={len(per)}  rows/chunk={dict(Counter(per.values()))}")
print(f"   distinct actions measured: {len(acts)}/32")
print(f"   quantize mix: q=0 {q.get('0',0)}   q=1 {q.get('1',0)}")
codecs = Counter(x['lib_name'] for x in r)
print(f"   codecs measured: {len(codecs)} -> {' '.join(sorted(codecs))}")
PY
fi
[ -x "$PB/collect.py" ] && { echo; say "== collect"; "$PB/collect.py" "$RESULTS" 2>&1 | tail -14; }
exit $rc
