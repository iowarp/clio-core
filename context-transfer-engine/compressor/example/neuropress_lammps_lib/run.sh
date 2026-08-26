#!/usr/bin/env bash
# LAMMPS as a library, Clio in the same process.
#
#   ./run.sh [--box N] [--steps N] [--gap N] [--chunk BYTES] [--store DIR]
#            [--order id|local|device] [--kokkos] [--verify] [--restart]
#            [--learn] [--explore K] [--threshold X] [--best]
#            [--static LIB] [--static-shuffle N] [--port N]
#
#   --order device gathers into atom-ID order ON THE GPU and hands the
#   compressor a CUDA-IPC-registered device pointer, so the payload is never
#   host bytes. Needs --kokkos; the driver refuses without it rather than
#   quietly reading the host mirror. Same ordering as --order id.
#
# One process, one binary (bin/neuropress_lammps_lib): liblammps runs the LJ
# melt, Clio's runtime is composed from $STORE/compose.yaml and hosted in that
# same process (CLIO_WITH_RUNTIME=1), and the driver moves Atom::x/v/f from
# one to the other between `run` segments. The compose file and every
# NeuroPress knob below are the same as ../neuropress_lammps_h5/lmp_common.sh,
# so numbers are comparable across the three LAMMPS examples.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-/home/cc/clio-core}
BUILD=${BUILD:-$REPO/build}
BIN=$BUILD/bin/neuropress_lammps_lib
WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights

BOX=10 STEPS=100 GAP=50 CHUNK=4194304 ORDER=id
PORT=${PORT:-9413}   # runtime port; every Clio runtime on the box needs its own
STORE=${STORE:-$HERE/store}
KOKKOS=false VERIFY=false RESTART=false
LEARN=false EXPLORE_K=0 THRESH=0.5 BEST=false STATIC_LIB= STATIC_SHUF=0
EXTRA=()
usage() { sed -n '2,17p' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --box) BOX=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --store) STORE=$2; shift 2;;
    --order) ORDER=$2; shift 2;;
    --kokkos) KOKKOS=true; shift;;
    --verify) VERIFY=true; shift;;
    --restart) RESTART=true; shift;;       # keep the store, replay its log
    --learn) LEARN=true; shift;;
    --explore) EXPLORE_K=$2; shift 2;;
    --threshold) THRESH=$2; shift 2;;
    --best) BEST=true; shift;;
    --static) STATIC_LIB=$2; shift 2;;
    --static-shuffle) STATIC_SHUF=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) EXTRA+=("$1"); shift;;              # passed to the driver verbatim
  esac
done

[ -x "$BIN" ] || { echo "missing $BIN -- configure with LAMMPS_SRC_DIR/LAMMPS_BUILD_DIR and build target neuropress_lammps_lib" >&2; exit 1; }
[ -d "$WEIGHTS" ] || { echo "missing weights: $WEIGHTS" >&2; exit 1; }

# A fresh store unless asked to continue: a stale tier would let a later
# --verify pass on data this run never produced.
if [ "$RESTART" != true ]; then rm -rf "$STORE"; fi
mkdir -p "$STORE"

# Sizing, same arithmetic as the siblings: float64, natoms*3*8 per field.
NATOMS=$(( 4 * BOX * BOX * BOX ))
DUMP_MB=$(( NATOMS * 3 * 8 * 3 / 1048576 )); [ "$DUMP_MB" -lt 1 ] && DUMP_MB=1
NDUMP=$(( STEPS / GAP + 1 ))
TIER_MB=$(( DUMP_MB * NDUMP + 512 ))
BDEV_MB=$(( TIER_MB * 2 ))

if [ -n "$STATIC_LIB" ]; then NP_LEARN=false; NP_EXPLORE=false
elif [ "$EXPLORE_K" -gt 0 ]; then NP_LEARN=true; NP_EXPLORE=true
elif [ "$LEARN" = true ]; then NP_LEARN=true; NP_EXPLORE=false
else NP_LEARN=false; NP_EXPLORE=false; fi

if [ "$RESTART" != true ]; then
cat > "$STORE/compose.yaml" <<YAML
networking:
  port: $PORT
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "$STORE/chi_bdev.dat"
    pool_query: local
    pool_id: "301.0"
    bdev_type: file
    path: "$STORE/chi_bdev.dat"
    capacity: "${BDEV_MB}MB"
  - mod_name: clio_cte_compressor
    pool_name: cte_compressor
    pool_query: local
    pool_id: "512.0"
    next_pool_id: "513.0"
    neuropress_model_path: "$WEIGHTS"
    neuropress_online_learning_enabled: $NP_LEARN
    neuropress_exploration_enabled: $NP_EXPLORE
    neuropress_exploration_k: $EXPLORE_K
    neuropress_exploration_threshold: $THRESH
    neuropress_best_mode: $BEST
${STATIC_LIB:+    neuropress_static_lib: "$STATIC_LIB"}
${STATIC_LIB:+    neuropress_static_shuffle: $STATIC_SHUF}
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "513.0"
    storage:
      - path: "$STORE/cte_tier.dat"
        bdev_type: "file"
        capacity_limit: "${TIER_MB}MB"
        score: 1.0
        persistence_level: "temporary"
    performance:
      metadata_log_path: "$STORE/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
YAML
fi

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
ARGS=(--deck "$HERE/in.melt_lib" --box "$BOX" --steps "$STEPS" --gap "$GAP"
      --chunk "$CHUNK" --order "$ORDER" --log "$STORE/log.lammps"
      --report "$STORE/blobs.csv" --quiet)
[ "$KOKKOS" = true ] && ARGS+=(--kokkos)
[ "$VERIFY" = true ] && ARGS+=(--verify)

echo "== LAMMPS(lib) + Clio in one process: box=$BOX ($NATOMS atoms) steps=$STEPS gap=$GAP chunk=$CHUNK order=$ORDER device=$([ "$KOKKOS" = true ] && echo GPU || echo CPU)"
if [ -n "$STATIC_LIB" ]; then echo "   STATIC codec=$STATIC_LIB shuffle=$STATIC_SHUF (NeuroPress bypassed)"
elif [ "$BEST" = true ]; then echo "   BEST mode (exhaustive, ratio-only)"
else echo "   learn=$NP_LEARN explore=$NP_EXPLORE k=$EXPLORE_K"; fi
echo "   store=$STORE  port=$PORT"

NP_ENV=(env CLIO_SERVER_CONF="$STORE/compose.yaml"
        CLIO_WITH_RUNTIME=1
        CLIO_LMP_COMPRESSOR_POOL=512.0
        # "warning", not "warn". Logger::Logger (clio_ctp/util/logging.h:143-163)
        # matches the full names only and then falls through to std::stoi, which
        # throws on "warn" and leaves the COMPILE-TIME default in place -- kDebug in
        # this build. Measured on this example: "warn" gives 386 DEBUG lines out
        # of 540, "warning" gives 0 out of 50. Every run was silently at debug,
        # which both bloats the log and buries the warnings it should surface.
        CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}")
# CLIO_RESTART=1 replays the metadata log, so a second process can see the
# blobs a first one stored. Same switch the siblings' read phases use.
[ "$RESTART" = true ] && NP_ENV+=(CLIO_RESTART=1)

# PROFILE=<command> wraps the driver, so the transfer claims in the README can
# be re-measured rather than believed. It goes INSIDE the env prefix: nsys must
# see the environment the driver runs with, and mpirun-style argument rules do
# not apply here.
#   PROFILE="nsys profile --trace=cuda --force-overwrite=true -o /tmp/x" ./run.sh ...
PROFILE=${PROFILE:-}
set +e
"${NP_ENV[@]}" ${PROFILE} "$BIN" "${ARGS[@]}" "${EXTRA[@]}" 2> "$STORE/runtime.log"
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "-- driver failed (rc=$RC); last lines of $STORE/runtime.log:"
  grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -12
  if grep -q "Address already in use" "$STORE/runtime.log"; then
    echo "-- port $PORT is held by another Clio runtime on this machine (ss -ltnp | grep $PORT); rerun with --port N"
  fi
fi
echo "-- blobs: $STORE/blobs.csv   runtime log: $STORE/runtime.log   exit=$RC"
exit $RC
