#!/usr/bin/env bash
#SBATCH --job-name=np-warpx-replay
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bekn/imuradli/np-warpx-replay/slurm-%j.out
#
# Phase 2 of the WarpX REPLAY route: replay the flat float32 dumps through
# neuropress_field_replay -- the SAME driver Nyx and VPIC use -- with
# CLIO_NEUROPRESS_STAGE_H2D=1, so every chunk is device-resident at the
# compressor. See warpx_gen_fields.sh for why the VOL route cannot be.
#
# The compose file comes from nyx/common.sh's bench_compose, which is generic:
# it sizes the tier from $FIELDS and knows nothing about Nyx. bench_setup from
# the same file points BIN at neuropress_field_replay. Only meta.json is
# written here rather than by nyx/run_config.sh, so the run is labelled
# warpx-lwfa and collect.py does not fold it in with the Nyx cells.
set -uo pipefail

REPO=/u/imuradli/clio-core
PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

ROOT=/projects/bekn/imuradli/np-warpx-replay
FIELDS=$ROOT/fields
RESULTS=$ROOT/results
mkdir -p "$RESULTS"

say() { echo "[$(date +%H:%M:%S)] $*"; }
say "node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | sed 's/^/       gpu: /'

# ---- phase 1: dumps (kept across jobs; regenerating costs a WarpX run) ----
if [ -f "$FIELDS/gen.json" ] && [ "$(find "$FIELDS" -name '*.f32' | wc -l)" -gt 0 ]; then
  say "field dumps already present ($(du -sh "$FIELDS" | cut -f1)), skipping WarpX"
else
  say "== phase 1: WarpX -> flat float32"
  "$PB/analysis/smoke/warpx_gen_fields.sh" --out "$FIELDS" 2>&1 | sed 's/^/       /'
  [ "${PIPESTATUS[0]}" -eq 0 ] || { say "gen_fields FAILED"; exit 1; }
fi
echo

# ---- phase 2: replay ------------------------------------------------------
export FIELDS
. "$PB/nyx/common.sh"
bench_setup || exit 1

tag=warpx_lossy_balance_nvme_5GBs
STORE=$RESULTS/$tag
rm -rf "$STORE"; mkdir -p "$STORE"

CHUNK=1048576            # 1 MiB, the chunk size the WarpX VOL route uses
EB=1e-3 BW=5e6 K=31 THRESH=-1

# BEFORE bench_compose, not on the driver's env line. bench_compose reads these
# as SHELL variables and bakes them into compose.yaml; setting them later got a
# YAML with exploration_enabled=false, k=0, threshold=0.5, so the run was
# inference-only and wrote no explore.csv at all.
export NP_LEARN=true NP_EXPLORE=true EXPLORE_K="$K" THRESH="$THRESH" BEST=false
bench_compose "$STORE"
NFILES=$(find "$FIELDS" -name '*.f32' | wc -l)
PAYLOAD=$(du -sb "$FIELDS" | cut -f1)
say "== $tag: $NFILES field file(s), $(awk -v p=$PAYLOAD 'BEGIN{printf "%.1f", p/1048576}') MiB, chunk $CHUNK, K=$K, thresh=$THRESH, eb=$EB"

START=$(date +%s.%N)
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_STAGE_H2D=1 \
    CLIO_NEUROPRESS_REQUIRE_DEVICE=1 \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv" \
    CLIO_NEUROPRESS_ERROR_BOUND="$EB" \
    CLIO_NEUROPRESS_COST_BW="$BW" \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=1 \
    CLIO_NEUROPRESS_MEASURE_QUALITY=1 \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    "$BIN" --dir "$FIELDS" --ext .f32 --chunk "$CHUNK" \
           --tag "warpx_$tag" --report "$STORE/blobs.csv" \
    > "$STORE/stdout.log" 2> "$STORE/runtime.log"
rc=$?
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
grep -E "^stored|VERIFIED|FAILED:|  time:" "$STORE/stdout.log" | sed 's/^/       /' || true
[ "$rc" -ne 0 ] && { say "FAILED rc=$rc"; grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -8; }

# NOT `$(grep -c ... || echo 0)`: grep -c PRINTS the count and then exits 1
# when it is zero, so the fallback appended a SECOND "0" and meta.json
# came out as `"host_refusals":0\n0,` -- invalid JSON that took collect.py
# down with a decode error two steps later.
HOSTED=$(grep -c "REQUIRE_DEVICE is set" "$STORE/runtime.log" 2>/dev/null); HOSTED=${HOSTED:-0}
NCHUNKS=0; [ -f "$STORE/blobs.csv" ] && NCHUNKS=$(( $(wc -l < "$STORE/blobs.csv") - 1 ))
cat > "$STORE/meta.json" <<JSON
{"config":"explore-balance","tag":"$tag","rc":$rc,"mode":"lossy",
 "cost_model":"balance","bw_bytes_per_ms":$BW,"error_bound":$EB,
 "explore_k":$K,"explore_thresh":$THRESH,"device":"gpu","workload":"warpx-lwfa",
 "residency":"device (staged H2D by the replay driver)","host_refusals":$HOSTED,
 "atoms":0,"steps":40,"gap":10,"frames":5,"chunk":$CHUNK,
 "files":$NCHUNKS,"payload_bytes":$PAYLOAD,"wall_s":$WALL,
 "verified":0,"physics":{"problem":"laser-wakefield","precision":"float32","route":"replay"}}
JSON

echo; say "== residency"
echo "   host-resident chunks REFUSED at the compressor: $HOSTED   (0 = every chunk was device memory)"
rm -f "$STORE/chi_bdev.dat" "$STORE"/cte_tier.dat*

if [ -f "$STORE/explore.csv" ]; then
  echo; say "== exploration coverage"
  python3 "$PB/analysis/smoke/validate_exploration.py" "$STORE/explore.csv" 31 2>&1 | sed 's/^/   /'
fi
[ -x "$PB/collect.py" ] && { echo; say "== collect"; "$PB/collect.py" "$RESULTS" 2>&1 | tail -12; }
exit $rc
