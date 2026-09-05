# Shared setup for the VPIC paper benchmark. Sourced, not executed.
#
# The workload is a Weibel instability computed by VPIC-Kokkos (CUDA, float32)
# and dumped to flat field files by gen_fields.sh. This phase replays those
# files through Clio with NeuroPress choosing a codec per chunk.
#
# VPIC completes the picture the other two workloads leave incomplete. LAMMPS
# is float64 and high-entropy; Nyx is float32 and highly structured. VPIC is
# float32 and high-entropy -- particle-in-cell field arrays are close to noise
# -- which separates "float32" from "compressible" in every claim that rests on
# the element width.
#
# Every configuration in run_sweep.sh differs ONLY in how the codec is chosen.
# Unlike the LAMMPS benchmark -- whose GPU trajectory is not bit-reproducible,
# so each policy sees slightly different bytes -- every policy here replays the
# IDENTICAL files, which makes the comparison exact rather than statistical.

bench_setup() {
  REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
  BUILD=${BUILD:-$REPO/build}
  BIN=${BIN:-$BUILD/bin/neuropress_field_replay}
  WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights

  if [ ! -x "$BIN" ]; then
    cat >&2 <<MSG
missing driver: $BIN

Build it (no simulation dependency at all):
  cmake --build $BUILD --target neuropress_field_replay
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
  # Tier is sized from the payload actually on disk plus headroom, since the
  # replay's size is a property of the dump rather than of any parameter here.
  local payload_mb=$(( $(du -sm "$FIELDS" 2>/dev/null | cut -f1) ))
  [ "$payload_mb" -lt 1 ] && payload_mb=1
  local tier_mb=$(( payload_mb + 512 ))
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
${STATIC_LIB:+    neuropress_static_quantize: ${STATIC_QUANT:-false}}
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
