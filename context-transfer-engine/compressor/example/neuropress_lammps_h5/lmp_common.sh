# Shared setup for the LAMMPS -> HDF5 -> Clio -> NeuroPress runs.
# Sourced, not executed. Mirrors neuropress_grayscott_h5/np_common.sh, but the
# application here is a STOCK LAMMPS binary: no patch, no Clio link, no
# gpucompress_* call. Clio arrives purely through HDF5_VOL_CONNECTOR.

lmp_setup() {
  REPO=${REPO:-/home/cc/clio-core}
  BUILD=${BUILD:-$REPO/build}
  LMP=${LMP:-$HOME/src/lammps/build/lmp}
  VOL=$BUILD/bin/libclio_hdf5_vol.so
  WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights
  for f in "$LMP" "$VOL"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; return 1; }
  done
  [ -d "$WEIGHTS" ] || { echo "missing weights: $WEIGHTS" >&2; return 1; }

  mkdir -p "$STORE"
  ln -sf "$VOL" "$STORE/"

  # Sizing. h5md writes float64: natoms*3*8 bytes per field, 3 fields per dump.
  NATOMS=$(( 4 * BOX * BOX * BOX ))
  DUMP_MB=$(( NATOMS * 3 * 8 * 3 / 1048576 )); [ "$DUMP_MB" -lt 1 ] && DUMP_MB=1
  NDUMP=$(( STEPS / GAP + 1 )); [ "$NDUMP" -lt 1 ] && NDUMP=1
  TIER_MB=$(( DUMP_MB * NDUMP + 512 )); [ "$TIER_MB" -lt 512 ] && TIER_MB=512
  BDEV_MB=$(( TIER_MB * 2 ))

  # 4 MiB. Every published measurement for this example used it, and h5md's
  # frames (natoms*3*8 bytes) do not divide it -- which is the point: chunk
  # boundaries fall INSIDE frames, so the append path has to assemble chunks
  # across successive H5Dwrite calls rather than getting whole ones handed to it.
  CHUNK=${CHUNK:-4194304}
  BEST=${BEST:-false}
  # Static codec control. Non-empty pins every chunk to one library and takes
  # NeuroPress out of the loop entirely -- the baseline a selector's ratio has
  # to be compared against. "nvcomp-zstd", not "zstd": upstream's
  # GPUCOMPRESS_ALGO_ZSTD builds an nvcomp::ZstdManager, so the CPU zstd is
  # not its counterpart.
  STATIC_LIB=${STATIC_LIB:-}
  EXPLORE_K=${EXPLORE_K:-0}
  LEARN=${LEARN:-false}
  THRESH=${THRESH:-0.5}
  if [ -n "$STATIC_LIB" ]; then
    NP_LEARN=false; NP_EXPLORE=false
  elif [ "${EXPLORE_K:-0}" -gt 0 ]; then
    NP_LEARN=true;  NP_EXPLORE=true
  elif [ "$LEARN" = true ]; then
    NP_LEARN=true;  NP_EXPLORE=false
  else
    NP_LEARN=false; NP_EXPLORE=false
  fi

  cat > "$STORE/compose.yaml" <<YAML
networking:
  port: 9413
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

  export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  NP_ENV=(env CLIO_SERVER_CONF="$STORE/compose.yaml"
          CLIO_WITH_RUNTIME=1
          HDF5_VOL_CONNECTOR=clio
          HDF5_PLUGIN_PATH="$STORE"
          CLIO_VOL_COMPRESSOR_POOL=512.0
          CLIO_VOL_CHUNK_SIZE="$CHUNK"
          CLIO_VOL_STAMP_GRANULARITY_NS=0
          CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-debug}")
  NP_ENV_RESTART=("${NP_ENV[@]}" CLIO_RESTART=1)
  return 0
}
