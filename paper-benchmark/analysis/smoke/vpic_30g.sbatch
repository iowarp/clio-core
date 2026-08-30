#!/usr/bin/env bash
#SBATCH --job-name=np-vpic-30g
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=03:00:00
#SBATCH --output=/projects/bekn/imuradli/np-vpic-30g/slurm-%j.out
#
# The 30 GiB VPIC cell, matching the Nyx one in scale and settings.
#
#   126^3 (+ghost = 128^3), 6000 steps, dump every 25 -> 240 frames x 16 vars
#   = 30.0 GiB, 3840 chunks @ 8 MiB, 122,880 measured candidates at K=31.
#
# NO PHASE 1. VPIC runs IN SITU: the deck hands the compressor its Kokkos field
# views from begin_diagnostics, so the simulation and the compression are the
# same process and no file is ever written. Profiling confirmed this is real --
# zero payload-sized HtoD copies, against Nyx's two per chunk.
#
# eb=0.05, matching run_benchmark.sh's LOSSY_EB. READ THE RESULTS OF THIS CELL
# WITH THAT IN MIND: 0.05 is chosen for Nyx, whose fields reach 2.1e5, and it
# is far too coarse for VPIC, whose fields are order 0.1-1 (MAD 7.9e-09 ..
# 2.0e-01, cby amplitude +/-0.169).
#
#   0.05 is ~30% of cby's full dynamic range. The grid step is 2*eb = 0.1
#   against a range of ~0.338, so roughly 3 quantization levels survive across
#   the entire field, and every VPIC field whose range is under ~0.05 collapses
#   to a single level.
#
# For scale, at 1e-3 -- fifty times tighter -- this run already had 720 of 3840
# chunks (18.8%) whose ENTIRE data range sat below the bound: div_e_err ~1e-6,
# div_b_err ~8e-8, rhob ~2e-6, all flattened to a constant with
# max_error == data_range, SSIM ~0 and PSNR ~5 dB. At 0.05 those stay flattened
# and many more join them.
#
# So a ratio from this cell is largely a measure of the bound, not of the
# compressor. Before quoting anything from it, check meas_ssim and the
# data_range column in selection.csv.quality and report how many chunks were
# flattened. If VPIC needs a bound its own data can carry, pass --eb explicitly
# (1e-3 was the previous value) rather than taking the shared default -- one
# absolute number cannot serve both workloads.
#
# clean_div stays at run_config_insitu.sh's default 10, the value the
# 1000-timestep hyperparameter study selected -- at upstream's 0, four of the
# sixteen variables are never recomputed and are dumped bit-identical forever.
set -uo pipefail
REPO=/u/imuradli/clio-core; PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

ROOT=/projects/bekn/imuradli/np-vpic-30g
RESULTS=$ROOT/results; tag=vpic_lossy_balance_nvme_5GBs; S=$RESULTS/$tag
mkdir -p "$RESULTS"
say(){ echo; echo "[$(date +%H:%M:%S)] $*"; }

say "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/       gpu: /'
df -BG --output=avail "$ROOT" | tail -1 | sed 's/^/       free: /;s/$/ GiB/'
[ -x "$PB/vpic/weibel_clio.Linux" ] || { say "no deck binary"; exit 1; }

say "in-situ run: 126^3, 6000 steps, dump every 25 (K=31, thresh -1, eb=0.05)"
t0=$(date +%s)
"$PB/vpic/run_config_insitu.sh" explore-balance \
    --ncell 126 --steps 6000 --int 25 \
    --chunk 8388608 --bw 5e6 --eb 0.05 \
    --explore-k 31 --explore-thresh -1 \
    --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
rc=${PIPESTATUS[0]}
say "   took $(( $(date +%s) - t0 )) s, rc=$rc"

rm -f "$S/chi_bdev.dat" "$S"/cte_tier.dat*
rm -rf "$S"/rank*/chi_bdev.dat "$S"/rank*/cte_tier.dat*

say "checks"
NB=0; [ -f "$S/blobs.csv" ] && NB=$(( $(wc -l < "$S/blobs.csv") - 1 ))
echo "   blobs on the tier : $NB"
for f in blobs.csv selection.csv explore.csv meta.json; do
  printf "   %-14s %s\n" "$f" "$([ -s "$S/$f" ] && du -h "$S/$f"|cut -f1 || echo MISSING)"
done
H=$(grep -c "REQUIRE_DEVICE is set" "$S/runtime.log" 2>/dev/null); H=${H:-0}
echo "   host-resident chunks REFUSED : $H   (0 = every chunk stayed on the GPU)"
E=$(grep -cE "\bERROR\b|\bFATAL\b|terminate called|Segmentation fault" "$S/runtime.log" 2>/dev/null); E=${E:-0}
echo "   real errors       : $E"
[ -f "$S/explore.csv" ] && python3 "$PB/analysis/smoke/validate_exploration.py" "$S/explore.csv" 31 2>&1 | sed 's/^/   /'
say "collect"; "$PB/collect.py" "$RESULTS" 2>&1 | tail -10
say "done rc=$rc"; exit $rc
