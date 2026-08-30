#!/usr/bin/env bash
#SBATCH --job-name=np-nyx-30g
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=/projects/bekn/imuradli/np-nyx-30g/slurm-%j.out
#
# The 30 GiB Nyx cell. Volume comes from TIME, not from the grid: 128^3 is the
# resolution the evolving-configuration study validated, and 6400 steps at
# plot-int 10 gives 641 frames x 6 fields = 3846 files = 30.0 GiB. run_
# benchmark.sh's `full` profile reaches a similar size by going to 256^3
# instead, which is more voxels of the same instants rather than more instants.
#
#   payload      30.0 GiB      chunks @8 MiB   3846
#   explore rows 123,072       (3846 x 32, K=31)
#   peak disk    ~92 GiB       fields 30 + tier 31 + bdev 61, the last two
#                              dropped when the cell finishes
#
# ONE cell, not the 15-cell ladder. Everything validated so far ran at 126
# chunks; this is 30x that, and the first thing to learn is whether it holds at
# scale. The bandwidth ladder and repeats can follow once it does -- and under
# the new defaults bandwidth cannot change a MEASURED value anyway, only the
# cost the model computes from it.
#
# eb=0.05, matching run_benchmark.sh's LOSSY_EB, and it is the value this
# workload wants. The quantizer relaxes when
#   eb - max|v|*2.4e-7 - 0.05*eb <= 0
# evaluated per CHUNK, so the relaxation ceiling is max|v| > 0.95*eb/2.4e-7:
# ~197900 at 0.05 against ~39600 at 0.01. Nyx rho_E reaches 2.1e5, so at 0.05
# only chunks holding the very peak relax, where at 0.01 a large share did --
# that run reported an effective bound of 0.00504203.
#
# Measured, not assumed. Three points on the curve:
#   1e-3   rho_E/rho_e relaxed to 5.042e-3; four of six fields got the bound
#          asked for, two did not
#   0.01   relaxation warning fired, effective bound 0.00504203; it relaxed
#          TIGHTER so nothing exceeded 0.01, but the ratios are those of 0.005
#   0.05   0 of 1898 measured rows over the bound, worst 4.750e-02
#
# Still check meta.json and the relax warnings in runtime.log before quoting
# ratios: a relaxed bound makes the ratio belong to a bound nobody asked for.
set -uo pipefail
REPO=/u/imuradli/clio-core; PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export NYX_BIN=/u/imuradli/np-build/nyx/Exec/HydroTests/nyx_HydroTests
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

ROOT=/projects/bekn/imuradli/np-nyx-30g
FIELDS=$ROOT/fields; RESULTS=$ROOT/results
tag=nyx_lossy_balance_nvme_5GBs; S=$RESULTS/$tag
mkdir -p "$RESULTS"
say(){ echo; echo "[$(date +%H:%M:%S)] $*"; }

say "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/       gpu: /'
df -BG --output=avail "$ROOT" | tail -1 | sed 's/^/       free: /;s/$/ GiB/'

# ---- phase 1 --------------------------------------------------------------
if [ "$(find "$FIELDS" -name '*.f32' 2>/dev/null | wc -l)" -ge 3846 ]; then
  say "phase 1: dumps already complete, skipping Nyx"
else
  say "phase 1: Nyx 128^3, 6400 steps, dump every 10 -> 641 frames, 30 GiB"
  t0=$(date +%s)
  "$PB/nyx/gen_fields.sh" --ncell 128 --steps 6400 --plot-int 10 --out "$FIELDS" 2>&1 | sed 's/^/       /'
  [ "${PIPESTATUS[0]}" -eq 0 ] || { say "gen_fields FAILED"; exit 1; }
  say "   phase 1 took $(( $(date +%s) - t0 )) s"
fi
NF=$(find "$FIELDS" -name '*.f32' | wc -l)
say "   $NF field files, $(du -sh "$FIELDS" | cut -f1)"
[ "$NF" -gt 0 ] || { say "no dumps"; exit 1; }

# ---- phase 2 --------------------------------------------------------------
say "phase 2: exploration replay (K=31, thresh -1, eb=0.05, balance, quality on)"
t0=$(date +%s)
env CLIO_NEUROPRESS_STAGE_H2D=1 \
    "$PB/nyx/run_config.sh" explore-balance --fields "$FIELDS" \
      --bw 5e6 --eb 0.05 --explore-k 31 --explore-thresh -1 --chunk 8388608 \
      --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
rc=${PIPESTATUS[0]}
say "   phase 2 took $(( $(date +%s) - t0 )) s, rc=$rc"

# The tier and bdev are 92 GiB between them; the measurements are in the CSVs.
rm -f "$S/chi_bdev.dat" "$S"/cte_tier.dat*

# ---- checks, the same gate the smoke test passed --------------------------
say "checks"
NB=0; [ -f "$S/blobs.csv" ] && NB=$(( $(wc -l < "$S/blobs.csv") - 1 ))
echo "   blobs on the tier : $NB"
for f in blobs.csv selection.csv explore.csv meta.json; do
  printf "   %-14s %s\n" "$f" "$([ -s "$S/$f" ] && echo "$(du -h "$S/$f"|cut -f1)" || echo MISSING)"
done
E=$(grep -E "\bERROR\b|\bFATAL\b|terminate called|Segmentation fault" "$S/runtime.log" 2>/dev/null \
     | grep -cv "HANGWATCH"); E=${E:-0}
echo "   real errors       : $E"
[ -f "$S/explore.csv" ] && python3 "$PB/analysis/smoke/validate_exploration.py" "$S/explore.csv" 31 2>&1 | sed 's/^/   /'
say "collect"; "$PB/collect.py" "$RESULTS" 2>&1 | tail -10
say "done rc=$rc"; exit $rc
