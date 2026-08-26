#!/usr/bin/env bash
# Shared setup for the Gray-Scott sweep.
#
# Gray-Scott is the fourth workload and the only SYNTHETIC one: a
# reaction-diffusion PDE whose (F, k) pair selects the pattern regime, and the
# regime is what moves compressibility. That makes it the one workload here
# where the data's compressibility is a dial rather than a property of a
# physics code, which is why upstream NeuroPress uses it too
# (benchmarks/grayscott/grayscott-benchmark-pm.cu).
#
# The application is bin/neuropress_grayscott_h5: it steps the simulation on
# the GPU and writes HDF5. It links NOTHING from Clio -- `ldd` shows zero Clio
# libraries -- and Clio enters through the HDF5 VOL connector, exactly as in
# the WarpX workload. So this sweep is IN SITU: the field never lands in a
# file before it is compressed, and there is no replay phase.
bench_setup() {
  REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
  BUILD=${BUILD:-$REPO/build}
  BIN=${BIN:-$BUILD/bin/neuropress_grayscott_h5}
  VOL=${VOL:-$BUILD/bin/libclio_hdf5_vol.so}
  WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights
  if [ ! -x "$BIN" ]; then
    cat >&2 <<MSG
missing $BIN
  cmake --build $BUILD --target neuropress_grayscott_h5
MSG
    return 1
  fi
  [ -f "$VOL" ] || { echo "missing $VOL (target clio_hdf5_vol)" >&2; return 1; }
  [ -d "$WEIGHTS" ] || { echo "missing weights: $WEIGHTS" >&2; return 1; }
  return 0
}

# A free TCP port. Every Clio runtime binds more than the one it is told about
# (a peer socket at port+1), so probe a small range and leave room.
bench_port() {
  python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close()
print(p)
PY
}

# The three-pool compose file every workload in this directory uses.
bench_compose() {
  local store=$1 port=$2 tier_mb=$3
  cat > "$store/compose.yaml" <<YAML
networking:
  port: $port
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "$store/chi_bdev.dat"
    pool_query: local
    pool_id: "301.0"
    bdev_type: file
    path: "$store/chi_bdev.dat"
    capacity: "$(( tier_mb * 2 ))MB"
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
