#!/usr/bin/env bash
#SBATCH --job-name=np-profile2
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bekn/imuradli/np-profile/slurm2-%j.out
#
# Round 2: the two routes round 1 missed, plus a DIRECT TEST of the claim that
# LAMMPS' per-chunk D2H is caused by --report rather than by --verify.
# run_config.sh always passes --report, so the only way to test it is to call
# the driver itself, once with and once without, changing nothing else.
set -uo pipefail
REPO=/u/imuradli/clio-core; PB=$REPO/paper-benchmark
export BUILD=/u/imuradli/np-build/clio
export LD_LIBRARY_PATH=/u/imuradli/np-env/np/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MEASURE_DT=0 MEASURE_QUALITY=0
ROOT=/projects/bekn/imuradli/np-profile; mkdir -p "$ROOT"; cd "$ROOT"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
prof(){ local name=$1; shift; say "== $name"; rm -f "$ROOT/$name.nsys-rep" "$ROOT/$name.sqlite"
  nsys profile -t cuda --force-overwrite true -o "$ROOT/$name" --sample none --cpuctxsw none \
    "$@" > "$ROOT/$name.stdout" 2> "$ROOT/$name.stderr" || say "   rc=$?"
  nsys export --type sqlite --force-overwrite true -o "$ROOT/$name.sqlite" "$ROOT/$name.nsys-rep" >/dev/null 2>&1; }

# --- WarpX REPLAY: same driver as nyx, dumps already on disk ---------------
WF=/projects/bekn/imuradli/np-warpx-replay/fields
if [ -d "$WF" ]; then
  export FIELDS=$WF
  . "$PB/nyx/common.sh"; bench_setup
  S=$ROOT/r/wxr; rm -rf "$S"; mkdir -p "$S"
  export NP_LEARN=true NP_EXPLORE=true EXPLORE_K=0 THRESH=0.5 BEST=false
  bench_compose "$S"
  prof warpx_replay env CLIO_SERVER_CONF="$S/compose.yaml" CLIO_WITH_RUNTIME=1 \
      CLIO_REPLAY_COMPRESSOR_POOL=512.0 CLIO_NEUROPRESS_STAGE_H2D=1 \
      CLIO_NEUROPRESS_REQUIRE_DEVICE=1 CLIO_NEUROPRESS_ERROR_BOUND=1e-3 \
      CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=0 CLIO_NEUROPRESS_MEASURE_QUALITY=0 \
      CTP_LOG_LEVEL=warning \
      "$BIN" --dir "$WF" --ext .f32 --chunk 1048576 --max-files 6 \
             --tag wxr --report "$S/blobs.csv"
fi

# --- LAMMPS: the --report hypothesis, driver called directly ---------------
LB=$BUILD/bin/neuropress_lammps_lib
if [ -x "$LB" ]; then
  for variant in with_report without_report; do
    S=$ROOT/r/lmp_$variant; rm -rf "$S"; mkdir -p "$S"
    natoms=$(( 4*20*20*20 )); dump_mb=$(( natoms*3*8*3/1048576 )); [ $dump_mb -lt 1 ] && dump_mb=1
    tier=$(( dump_mb*3 + 512 )); bdev=$(( tier*2 ))
    port=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
    cat > "$S/compose.yaml" <<YAML
networking:
  port: $port
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "$S/chi_bdev.dat"
    pool_query: local
    pool_id: "301.0"
    bdev_type: file
    path: "$S/chi_bdev.dat"
    capacity: "${bdev}MB"
  - mod_name: clio_cte_compressor
    pool_name: cte_compressor
    pool_query: local
    pool_id: "512.0"
    next_pool_id: "513.0"
    neuropress_model_path: "$REPO/context-transport-primitives/src/compress/model/weights"
    neuropress_online_learning_enabled: true
    neuropress_exploration_enabled: true
    neuropress_exploration_k: 0
    neuropress_exploration_threshold: 0.5
    neuropress_best_mode: false
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "513.0"
    storage:
      - path: "$S/cte_tier.dat"
        bdev_type: "file"
        capacity_limit: "${tier}MB"
        score: 1.0
        persistence_level: "temporary"
    performance:
      metadata_log_path: "$S/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
YAML
    ARGS=(--deck "$PB/lammps/in.melt" --box 20 --steps 50 --gap 25
          --chunk 1048576 --order device --kokkos --log "$S/log.lammps" --quiet)
    # THE ONLY DIFFERENCE between the two runs.
    [ "$variant" = with_report ] && ARGS+=(--report "$S/blobs.csv")
    prof "lammps_$variant" env CLIO_SERVER_CONF="$S/compose.yaml" CLIO_WITH_RUNTIME=1 \
        CLIO_LMP_COMPRESSOR_POOL=512.0 CLIO_NEUROPRESS_REQUIRE_DEVICE=1 \
        CLIO_NEUROPRESS_ERROR_BOUND=1e-3 CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=0 \
        CLIO_NEUROPRESS_MEASURE_QUALITY=0 CTP_LOG_LEVEL=warning \
        "$LB" "${ARGS[@]}"
  done
fi
say "done"
