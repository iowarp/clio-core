# Shared setup for the LAMMPS paper benchmark. Sourced, not executed.
#
# The workload is a Lennard-Jones melt run by LAMMPS as a LIBRARY, with Clio's
# runtime hosted in the same process and NeuroPress choosing a codec per chunk
# (DYNAMIC compression). Nothing is written to a file: the driver reads
# Atom::x/v/f between `run` segments and hands them to the compressor.
#
# Every configuration in run_sweep.sh differs ONLY in how the codec is chosen,
# so the payload is held constant and the selection policy is the variable.

bench_setup() {
  REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
  BUILD=${BUILD:-$REPO/build}
  BIN=${BIN:-$BUILD/bin/neuropress_lammps_lib}
  WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights

  if [ ! -x "$BIN" ]; then
    cat >&2 <<MSG
missing driver: $BIN

Build it with a LAMMPS tree available:
  cmake -S $REPO -B $BUILD -DLAMMPS_SRC_DIR=\$HOME/src/lammps -DLAMMPS_BUILD_DIR=\$HOME/src/lammps/build
  cmake --build $BUILD --target neuropress_lammps_lib
MSG
    return 1
  fi
  [ -d "$WEIGHTS" ] || { echo "missing NeuroPress weights: $WEIGHTS" >&2; return 1; }

  # Every Clio runtime binds a TCP port, and these runs host one in-process.
  # A fixed port collides with any other Clio process on the box -- which is a
  # silent, confusing failure (the client dies before doing anything), so pick
  # a free one unless the caller pinned it.
  if [ -z "${PORT:-}" ]; then
    PORT=$(python3 - <<'PY'
import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()
PY
)
  fi
  export PORT
  return 0
}

# bench_compose <store> -- write the three-pool compose file.
# clio_bdev(301.0) + clio_cte_compressor(512.0) -> clio_cte_core(513.0).
bench_compose() {
  local store=$1
  local natoms=$(( 4 * BOX * BOX * BOX ))
  local dump_mb=$(( natoms * 3 * 8 * 3 / 1048576 )); [ "$dump_mb" -lt 1 ] && dump_mb=1
  local ndump=$(( STEPS / GAP + 1 ))
  local tier_mb=$(( dump_mb * ndump + 512 ))
  local bdev_mb=$(( tier_mb * 2 ))

  mkdir -p "$store"
  cat > "$store/compose.yaml" <<YAML
networking:
  port: $PORT
runtime:
  num_threads: ${THREADS:-4}
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "$store/chi_bdev.dat"
    pool_query: local
    pool_id: "301.0"
    bdev_type: file
    path: "$store/chi_bdev.dat"
    capacity: "${bdev_mb}MB"
  - mod_name: clio_cte_compressor
    pool_name: cte_compressor
    pool_query: local
    pool_id: "512.0"
    next_pool_id: "513.0"
    neuropress_model_path: "$WEIGHTS"
    neuropress_online_learning_enabled: ${NP_LEARN:-false}
    neuropress_exploration_enabled: ${NP_EXPLORE:-false}
    neuropress_exploration_k: ${EXPLORE_K:-0}
    neuropress_exploration_threshold: ${THRESH:-0.5}
    neuropress_best_mode: ${BEST:-false}
${STATIC_LIB:+    neuropress_static_lib: "$STATIC_LIB"}
${STATIC_LIB:+    neuropress_static_shuffle: ${STATIC_SHUF:-0}}
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "513.0"
    storage:
      - path: "$store/cte_tier.dat"
        bdev_type: "file"
        capacity_limit: "${tier_mb}MB"
        score: 1.0
        persistence_level: "temporary"
    performance:
      metadata_log_path: "$store/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
YAML
}
