#!/usr/bin/env bash
#SBATCH --job-name=np-qual-na
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:25:00
#SBATCH --output=/projects/bekn/imuradli/np-qual-na-log/slurm-%j.out
#
# Gate for the explore-log quality columns after the NA/deferred-primary
# change. Small on purpose -- 6 frames, 36 chunks, 1152 rows -- because what
# is being checked is a FORMAT invariant that holds per row, not a scale
# property. The 30 GiB cells re-check it at scale for free.
#
# What must hold, and what each would mean if it did not:
#   quantize=1 -> five real numbers. A -1 or NA means the primary is still
#                 being written before its quality exists, i.e. the deferral
#                 did not take.
#   quantize=0 -> NA in all five and in quality_measured. A number there means
#                 the lossless arm is still reporting a codec-vs-its-own-input
#                 tautology as if it measured the data.
#   rows == chunks*32 -> no row lost. The deferral moves a write; a dropped
#                 row would be the way that goes wrong.

set -uo pipefail
REPO=/u/imuradli/clio-core; PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

export NYX_BIN=/u/imuradli/np-build/nyx/Exec/HydroTests/nyx_HydroTests
ROOT=/projects/bekn/imuradli/np-qual-na
FIELDS=$ROOT/fields; RESULTS=$ROOT/results
tag=nyx_lossy_balance_nvme_5GBs; S=$RESULTS/$tag
say(){ echo; echo "[$(date +%H:%M:%S)] $*"; }

# NOT `rm -rf $ROOT`: SLURM is writing this job's stdout into a sibling of it,
# and an earlier revision of this script deleted the file out from under the
# job -- it died at 5 s with no log at all. Clear only what we rewrite.
rm -rf "$FIELDS" "$RESULTS"; mkdir -p "$FIELDS" "$RESULTS"
say "node=$(hostname) job=${SLURM_JOB_ID:-none}"

# Generated here rather than borrowed from another run's tree: this gate is
# about a per-row format invariant, so it should not depend on whatever
# happens to be left on scratch. 60 steps at plot-int 10 is 6 frames.
say "generating 6 frames (128^3, 60 steps)"
"$PB/nyx/gen_fields.sh" --ncell 128 --steps 60 --plot-int 10 --out "$FIELDS" 2>&1 | sed 's/^/       /'
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "gen_fields FAILED"; exit 1; }
NF=$(find -L "$FIELDS" -name '*.f32' | wc -l)
say "fields: $(ls -d "$FIELDS"/plt* 2>/dev/null | wc -l) frames, $NF files"
[ "$NF" -gt 0 ] || { say "no field files"; exit 1; }

say "replay: K=31, thresh -1, eb=0.05, balance, quality on"
env CLIO_NEUROPRESS_STAGE_H2D=1 \
    "$PB/nyx/run_config.sh" explore-balance --fields "$FIELDS" \
      --bw 5e6 --eb 0.05 --explore-k 31 --explore-thresh -1 --chunk 8388608 \
      --results "$RESULTS" --tag "$tag" 2>&1 | sed 's/^/       /'
rc=${PIPESTATUS[0]}
say "replay rc=$rc"

say "checks"
python3 "$PB/analysis/smoke/check_quality_na.py" "$S/explore.csv"
gate=$?
say "done rc=$rc gate=$gate"; exit $(( rc != 0 ? rc : gate ))
