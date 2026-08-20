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
  # Best mode brings its own exploration settings (it forces exploration on and
  # pins K to 31) and ranks on ratio alone, so it does not take an explicit K.
  BEST=${BEST:-false}
  # Static codec control -- see the LAMMPS example's lmp_common.sh. Non-empty
  # pins every chunk to one library and takes NeuroPress out of the loop, the
  # fixed-codec baseline a selector has to be measured against.
  STATIC_LIB=${STATIC_LIB:-}
  EXPLORE_K=${EXPLORE_K:-0}
  LEARN=${LEARN:-false}
  # Exploration only fires on chunks whose cost error exceeds this. 0 means
  # every chunk, which is what best mode does via its own bypass -- but best
  # mode ALSO forces ratio-only adoption (kCostW0/W1 are hardcoded off it), so
  # it cannot be used to study the cost model. Setting the threshold directly
  # gets exhaustive exploration while leaving the weights alone.
  THRESH=${THRESH:-0.5}
  # Three distinct modes, and they are not a ladder:
  #   inference   predict and pick, nothing is measured back
  #   learning    SGD updates the model from each chunk's real outcome
  #   exploration learning PLUS actually compressing alternatives to compare
  # Exploration implies learning (its samples are what SGD trains on), but
  # learning is useful on its own and used to be unreachable here.
  if [ -n "$STATIC_LIB" ]; then
    NP_LEARN=false; NP_EXPLORE=false
  elif [ "${EXPLORE_K:-0}" -gt 0 ]; then
    NP_LEARN=true;  NP_EXPLORE=true
  elif [ "$LEARN" = true ]; then
    NP_LEARN=true;  NP_EXPLORE=false
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

# Stored size, read from the CTEC headers in the tier rather than by summing
# "Compression:" log lines. Those lines are one per CODEC RUN, and exploration
# runs several candidates per chunk, so summing them counted alternatives that
# were measured and thrown away -- it reported 19.82 MiB for a store that holds
# 2.43 MiB. The header carries the codec and the payload length of the bytes
# that were actually kept, so it cannot disagree with what is on disk.
#
# $1 = store dir, $2 = expected chunk count (0 to skip the cross-check)
np_stored_stats() {
  python3 - "$1" "${2:-0}" "$CHUNK" <<'PYEOF'
import struct, sys
from collections import Counter
store, expect, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
NAMES = {11:"lz4",12:"snappy",13:"zstd",14:"gdeflate",15:"deflate",16:"ans",
         21:"cascaded",22:"bitcomp"}
try:
    raw = open(f"{store}/cte_tier.dat_node0","rb").read()
except OSError:
    print("  (no tier file -- nothing stored)"); raise SystemExit(1)
needle = struct.pack("<I", 0x43544543)   # "CTEC"
off = tot = orig = n = 0
codecs = Counter()
while True:
    i = raw.find(needle, off)
    if i < 0: break
    _, lib, _, csz, osz = struct.unpack_from("<IIIIQ", raw, i)
    # A real header: known codec, payload smaller than the original, and an
    # original no larger than one chunk. Keeps stray "CTEC" bytes inside
    # compressed payloads from being counted as records.
    if lib in NAMES and 0 < csz < osz <= chunk:
        tot += csz; orig += osz; n += 1; codecs[NAMES[lib]] += 1
    off = i + 4
if not n:
    print("  NO COMPRESSED CHUNKS FOUND in the tier."); raise SystemExit(1)
# Two different quantities, and under exploration they diverge sharply.
# CODEC OUTPUT is what the winning codec produced. STORAGE OCCUPIED is what
# the blob actually takes in the tier -- and when exploration adopts a winner
# smaller than the primary, the blob keeps the PRIMARY's size and the surplus
# is dead space. Measured: without exploration physical == payload + 24 (the
# header) exactly; with it, physical ran 4-9x the payload. Reporting only the
# payload made exploration look far better than it stores, so print both and
# let the gap show.
print(f"  codec output    : {tot/1048576:.2f} MiB from {orig/1048576:.1f} MiB "
      f"({orig/tot:.1f}x)")
print(f"  storage occupied: run np_read.sh -- it reports the real footprint "
      f"from GetBlobSize (physical), which is the number to quote")
print(f"  codecs: " + " ".join(f"{k}:{v}" for k,v in codecs.most_common()))
if expect and n != expect:
    print(f"  WARNING: found {n} chunk headers, expected {expect} -- the tier "
          f"may hold chunks from an earlier run, or the scan missed some.")
PYEOF
}
