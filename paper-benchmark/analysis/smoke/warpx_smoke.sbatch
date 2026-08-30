#!/usr/bin/env bash
#SBATCH --job-name=np-warpx-smoke
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bekn/imuradli/np-warpx/slurm-%j.out
#
# WarpX, the fourth workload and the only one Clio reaches through the HDF5 VOL
# with ZERO lines changed in the application. WarpX writes openPMD-HDF5 exactly
# as it always does; HDF5 dlopens libclio_hdf5_vol.so because
# HDF5_VOL_CONNECTOR=clio says so, and the VOL chunks and compresses on the way
# past. (Upstream NeuroPress reaches WarpX through a 636-line patch adding a
# whole FlushFormatGPUCompress backend.)
#
# THE FAILURE MODE THIS JOB CHECKS FOR EXPLICITLY: if the connector does not
# load, HDF5 silently falls back to writing a native file. The simulation
# succeeds, the exit code is 0, the diagnostics are all there -- and nothing
# was compressed. Neither rc nor the WarpX log can tell you. So the run is only
# believed if blobs reached the tier.
#
# Settings are the new benchmark defaults: K=31 (whole 32-action space per
# chunk), threshold -1 (every chunk explores, not just those the predictor got
# wrong), eb=1e-3 lossy, balance, 5 GB/s -- plus MEASURE_QUALITY, which
# run_config.sh now turns on by default.
set -uo pipefail

REPO=/u/imuradli/clio-core
PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export WARPX_BIN=/projects/bekn/imuradli/np-src/warpx/build-clio/bin/warpx.3d
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

ROOT=/projects/bekn/imuradli/np-warpx
RESULTS=$ROOT/results
mkdir -p "$RESULTS"

say() { echo "[$(date +%H:%M:%S)] $*"; }
say "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | sed 's/^/       gpu: /'
say "warpx : $WARPX_BIN"
say "vol   : $BUILD/bin/libclio_hdf5_vol.so"
ldd "$BUILD/bin/libclio_hdf5_vol.so" 2>/dev/null | grep -i hdf5 | sed 's/^/       vol links: /'
ldd "$WARPX_BIN" 2>/dev/null | grep -i hdf5 | sed 's/^/       warpx links: /'
echo

tag=warpx_lossy_balance_nvme_5GBs
say "== $tag  (64x64x512, 40 steps, diag every 10, chunk 1 MiB, K=31, thresh -1)"
"$PB/warpx/run_config.sh" explore-balance \
    --ncell "64 64 512" --steps 40 --interval 10 --stage-h2d \
    --deck /projects/bekn/imuradli/np-src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d \
    --chunk 1048576 --bw 5e6 --eb 1e-3 \
    --explore-k 31 --explore-thresh -1 \
    --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
rc=${PIPESTATUS[0]}
echo; say "rc=$rc"

S=$RESULTS/$tag
# WarpX writes a full native openPMD set on top of the compressed tier.
rm -rf "$S/run/diags"
rm -f "$S/chi_bdev.dat" "$S"/cte_tier.dat*

say "== did the VOL actually intercept?"
nb=0; [ -f "$S/blobs.csv" ] && nb=$(( $(wc -l < "$S/blobs.csv") - 1 ))
echo "   blobs stored through the compressor : $nb"
echo "   selection.csv rows : $([ -f "$S/selection.csv" ] && echo $(( $(wc -l < "$S/selection.csv") - 1 )) || echo MISSING)"
echo "   explore.csv rows   : $([ -f "$S/explore.csv" ] && echo $(( $(wc -l < "$S/explore.csv") - 1 )) || echo MISSING)"
if [ "$nb" -le 0 ]; then
  echo "   VERDICT: NO BLOBS -- the VOL did not intercept. HDF5 wrote a native"
  echo "            file and nothing was compressed, whatever rc says."
  grep -iE "vol|connector|plugin" "$S/runtime.log" 2>/dev/null | head -5 | sed 's/^/            /'
else
  echo "   VERDICT: the VOL intercepted $nb chunk(s)."
fi

if [ -f "$S/explore.csv" ]; then
  echo; say "== exploration coverage"
  python3 "$PB/analysis/smoke/validate_exploration.py" "$S/explore.csv" 31 2>&1 | sed 's/^/   /'
fi
[ -x "$PB/collect.py" ] && { echo; say "== collect"; "$PB/collect.py" "$RESULTS" 2>&1 | tail -12; }
exit $rc
