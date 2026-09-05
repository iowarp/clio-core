# Shared setup for the WarpX paper benchmark. Sourced, not executed.
#
# This is the only IN-SITU workload here. The other three either embed a
# simulation (LAMMPS) or replay files it wrote (Nyx, VPIC). WarpX is a STOCK,
# UNPATCHED binary: it writes openPMD-HDF5 the way it always does, HDF5 dlopens
# Clio's VOL connector because HDF5_VOL_CONNECTOR says so, and the VOL chunks
# and compresses on the way past. Clio's runtime is hosted inside the WarpX
# process (CLIO_WITH_RUNTIME=1).
#
# Contrast with upstream NeuroPress, which reaches WarpX through a 636-line
# patch adding a whole FlushFormatGPUCompress diagnostic backend. Clio needs
# zero lines in WarpX.
#
# CHUNK SIZE IS LOAD-BEARING HERE in a way it is not for the other workloads --
# see the note on CLIO_VOL_CHUNK_SIZE in run_config.sh.

bench_setup() {
  REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
  BUILD=${BUILD:-$REPO/build}
  BIN=${BIN:-$WARPX_BIN}
  WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights

  if [ ! -x "$BIN" ]; then
    cat >&2 <<MSG
missing WarpX binary: $BIN

Build a STOCK WarpX with openPMD (no patch of any kind):
  git clone --depth 1 https://github.com/BLAST-WarpX/warpx.git ~/src/warpx
  cd ~/src/warpx
  cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release -DWarpX_COMPUTE=CUDA \\
        -DWarpX_DIMS=3 -DWarpX_MPI=OFF -DWarpX_OPENPMD=ON \\
        -DAMReX_CUDA_ARCH=8.0 -DWarpX_PRECISION=SINGLE
  cmake --build build-clio -j
Then point at it with WARPX_BIN=<path>.
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

  VOL_SO=$BUILD/bin/libclio_hdf5_vol.so
  [ -e "$VOL_SO" ] || { echo "missing VOL connector: $VOL_SO" >&2; return 1; }
  return 0
}

# bench_compose <store> -- write the three-pool compose file.
# clio_bdev(301.0) + clio_cte_compressor(512.0) -> clio_cte_core(513.0).
bench_compose() {
  local store=$1
  # Tier is sized from the payload actually on disk plus headroom, since the
  # replay's size is a property of the dump rather than of any parameter here.
  # Sized from what the run is expected to write rather than from files on
  # disk: in-situ, nothing exists yet when the compose file is generated.
  local tier_mb=${TIER_MB:-4096}
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
