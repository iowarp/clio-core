# Shared setup for np_write.sh / np_read.sh. Sourced, not executed.
#
# The two scripts are separate processes with separate runtimes. They agree on
# exactly one thing: STORE, the directory holding the HDF5 file, the file-backed
# tier, and the metadata log. Everything else is rebuilt from scratch on each
# side -- which is the point.

np_setup() {
  HERE=$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)
  REPO=$(cd "$HERE/../../../.." && pwd)
  BUILD=${BUILD:-$REPO/build}
  BIN=$BUILD/bin/neuropress_grayscott_h5
  VOL=$BUILD/bin/libclio_hdf5_vol.so
  WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights
  for f in "$BIN" "$VOL"; do
    [ -e "$f" ] || { echo "missing: $f
build first:  make -C $BUILD neuropress_grayscott_h5 clio_hdf5_vol" >&2; return 1; }
  done
  [ -d "$WEIGHTS" ] || { echo "missing NeuroPress weights: $WEIGHTS" >&2; return 1; }

  mkdir -p "$STORE"
  ln -sf "$VOL" "$STORE/"

  SNAP_MB=$(( L * L * L * 4 / 1048576 )); [ "$SNAP_MB" -lt 1 ] && SNAP_MB=1
  NSNAP=$(( STEPS / GAP )); [ "$NSNAP" -lt 1 ] && NSNAP=1
  TIER_MB=$(( SNAP_MB * NSNAP + 512 )); [ "$TIER_MB" -lt 512 ] && TIER_MB=512
  BDEV_MB=$(( TIER_MB * 2 ))

  CHUNK=${CHUNK:-1048576}
  # Exploration needs online learning: the point of trying K candidates is to
  # feed their measured outcomes back into the model, and with learning off the
  # extra compressions are paid for and then discarded.
  EXPLORE_K=${EXPLORE_K:-0}
  if [ "${EXPLORE_K:-0}" -gt 0 ]; then
    NP_LEARN=true;  NP_EXPLORE=true
  else
    NP_LEARN=false; NP_EXPLORE=false
  fi

  # A FILE-backed tier plus a metadata log, not the RAM tier run_standalone.sh
  # uses. RAM dies with the process that allocated it, and without the log the
  # tier's bytes would survive while nothing remembered which blob they belong
  # to -- either one alone leaves the reader with an empty cache.
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

  # Chunk size must match between the phases: the reader derives its chunk
  # count from it, and a mismatch turns every lookup into a miss that native
  # HDF5 then answers correctly -- a silent, passing, meaningless read.

  export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  # Each script hosts its OWN runtime. Load-bearing: a compressor task sent to
  # a runtime in another process arrives default-constructed and is rejected,
  # so the compressor has to live in whichever process is calling it.
  NP_ENV=(env CLIO_SERVER_CONF="$STORE/compose.yaml"
          CLIO_WITH_RUNTIME=1
          HDF5_VOL_CONNECTOR=clio
          HDF5_PLUGIN_PATH="$STORE"
          CLIO_VOL_COMPRESSOR_POOL=512.0
          CLIO_VOL_CHUNK_SIZE="$CHUNK"
          CLIO_VOL_STAMP_GRANULARITY_NS=0
          CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-debug}")
  # Reader-side variant: CLIO_RESTART=1 makes this runtime replay the metadata
  # WAL, which is the only way a fresh process learns which blobs the previous
  # one wrote. Without it the tier's bytes are on disk but nothing remembers
  # what they are, so every read misses and is served by native HDF5.
  NP_ENV_RESTART=("${NP_ENV[@]}" CLIO_RESTART=1)
  return 0
}

np_trace_on() {
  # env -u LD_LIBRARY_PATH: np_setup prepends the build's bin/ to it, and the
  # system `strings` then resolves its own dependencies against Clio's shipped
  # libraries and fails. The probe reported "trace off" for a build that had it
  # on, which turned a real PASS into a false INCONCLUSIVE.
  # Counted, not `grep -q`: under `set -o pipefail` grep -q closes the pipe on
  # its first match, `strings` dies of SIGPIPE, and the pipeline reports failure
  # for a build that DOES have the trace -- which showed up as a false
  # INCONCLUSIVE on a run that had already proved itself.
  local n
  n=$(env -u LD_LIBRARY_PATH strings \
        "$BUILD/bin/libclio_cte_compressor_runtime.so" 2>/dev/null \
      | grep -c 'np-path' || true)
  [ "${n:-0}" -gt 0 ]
}
