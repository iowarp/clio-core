#!/usr/bin/env bash
# VPIC in situ: the deck hands its field arrays to Clio, no file in between.
#
#   ./run.sh [--ncell N] [--steps N] [--int N] [--nppc N] [--chunk BYTES]
#            [--store DIR] [--port N] [--verify] [--restart] [--ranks N]
#            [--learn] [--explore K] [--threshold X] [--best]
#            [--static LIB] [--static-shuffle N] [--bin PATH]
#
# The deck is paper-benchmark/vpic/weibel_clio.cxx built with --insitu, so it
# links libclio_vpic_insitu.so and calls it from begin_diagnostics. Clio's
# runtime is hosted in the VPIC process itself (CLIO_WITH_RUNTIME=1): the
# compressor's tasks have no cross-process wire format, so a runtime elsewhere
# would receive every DynamicSchedule empty. The adapter refuses rather than
# let that look like success.
#
# MPI: --ranks N runs under mpirun with ONE Clio runtime per rank -- its own
# process, its own port, its own store under $STORE/rank%04d. Every rank then
# takes the same in-process path that was verified single-rank. VPIC splits the
# domain along x by nproc(), so each rank hands over its own subdomain and the
# blob names are identical across ranks; the stores are what keep them apart.
#
# Same three pools and the same NeuroPress knobs as every sibling example, so
# numbers are comparable with the offline sweep in paper-benchmark/vpic.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
DECK_DIR=${DECK_DIR:-$REPO/paper-benchmark/vpic}
BIN=${BIN:-$DECK_DIR/weibel_clio.Linux}
WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights

NCELL=126 STEPS=200 INT=25 NPPC=8 CHUNK=4194304
PORT=${PORT:-9413}
STORE=${STORE:-$HERE/store}
VERIFY=false RESTART=false RANKS=1
LEARN=false EXPLORE_K=0 THRESH=0.5 BEST=false STATIC_LIB= STATIC_SHUF=0
usage() { sed -n '2,17p' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int) INT=$2; shift 2;;
    --nppc) NPPC=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --store) STORE=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --verify) VERIFY=true; shift;;
    --restart) RESTART=true; shift;;
    --ranks) RANKS=$2; shift 2;;
    --learn) LEARN=true; shift;;
    --explore) EXPLORE_K=$2; shift 2;;
    --threshold) THRESH=$2; shift 2;;
    --best) BEST=true; shift;;
    --static) STATIC_LIB=$2; shift 2;;
    --static-shuffle) STATIC_SHUF=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -x "$BIN" ] || { echo "missing $BIN -- build it with:
  cmake --build $BUILD --target clio_vpic_insitu
  cd $DECK_DIR && ./build_deck.sh --insitu" >&2; exit 1; }
ldd "$BIN" | grep -q libclio_vpic_insitu || {
  echo "$BIN is not linked against libclio_vpic_insitu.so -- rebuild the deck
with ./build_deck.sh --insitu (without it the deck writes files instead)." >&2
  exit 1; }
[ -d "$WEIGHTS" ] || { echo "missing weights: $WEIGHTS" >&2; exit 1; }
if [ "$RANKS" -lt 1 ]; then echo "--ranks must be >= 1" >&2; exit 2; fi
if [ "$RANKS" -gt 1 ]; then
  command -v mpirun >/dev/null || { echo "--ranks needs mpirun on PATH" >&2; exit 1; }
fi

# One runtime per rank means $RANKS ports -- and a runtime binds more than the
# one the compose file names (a peer socket at port+1, at least), so they are
# spaced rather than consecutive. Measured the hard way: two ranks at 9413 and
# 9414 collide on 9414 and the second runtime fails to start.
PORT_STRIDE=${PORT_STRIDE:-8}
PORTS=""
p=$PORT
for ((r=0; r<RANKS; r++)); do
  while ss -ltn 2>/dev/null | grep -qE ":($p|$((p+1))|$((p+2))|$((p+3))) "; do
    p=$((p+PORT_STRIDE))
  done
  PORTS="$PORTS $p"; p=$((p+PORT_STRIDE))
done
PORTS=${PORTS# }

if [ "$RESTART" != true ]; then rm -rf "$STORE"; fi
mkdir -p "$STORE"

# Sizing: VPIC's field array carries a ghost layer, so the extent is (N+2)^3
# voxels; 16 float32 variables per frame.
VOX=$(( (NCELL+2) * (NCELL+2) * (NCELL+2) ))
FRAME_MB=$(( VOX * 4 * 16 / 1048576 )); [ "$FRAME_MB" -lt 1 ] && FRAME_MB=1
NFRAMES=$(( STEPS / INT ))
TIER_MB=$(( FRAME_MB * NFRAMES + 512 ))
BDEV_MB=$(( TIER_MB * 2 ))

if [ "$LEARN" = true ] || [ "$EXPLORE_K" -gt 0 ]; then NP_LEARN=true; else NP_LEARN=false; fi
if [ "$EXPLORE_K" -gt 0 ]; then NP_EXPLORE=true; else NP_EXPLORE=false; fi
if [ -n "$STATIC_LIB" ]; then NP_LEARN=false; NP_EXPLORE=false; fi

RANK_DIRS=()
r=0
for RPORT in $PORTS; do
  if [ "$RANKS" -eq 1 ]; then RD=$STORE; else RD=$(printf '%s/rank%04d' "$STORE" "$r"); fi
  RANK_DIRS+=("$RD")
  mkdir -p "$RD"
  if [ "$RESTART" != true ]; then
cat > "$RD/compose.yaml" <<YAML
networking:
  port: $RPORT
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "$RD/chi_bdev.dat"
    pool_query: local
    pool_id: "301.0"
    bdev_type: file
    path: "$RD/chi_bdev.dat"
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
      - path: "$RD/cte_tier.dat"
        bdev_type: "file"
        capacity_limit: "${TIER_MB}MB"
        score: 1.0
        persistence_level: "temporary"
    performance:
      metadata_log_path: "$RD/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
YAML
  fi
  r=$((r+1))
done

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "== VPIC in situ + Clio in one process: ${NCELL}^3 cells ($VOX voxels with ghosts)"
echo "   steps=$STEPS every $INT -> $NFRAMES frame(s) x 16 vars, ~${FRAME_MB} MiB/frame, chunk=$CHUNK"
if [ -n "$STATIC_LIB" ]; then echo "   STATIC codec=$STATIC_LIB shuffle=$STATIC_SHUF (NeuroPress bypassed)"
elif [ "$BEST" = true ]; then echo "   BEST mode (exhaustive, ratio-only)"
else echo "   learn=$NP_LEARN explore=$NP_EXPLORE k=$EXPLORE_K"; fi
if [ "$RANKS" -gt 1 ]; then
  echo "   ranks=$RANKS (mpirun), one Clio runtime PER RANK"
  echo "   stores=$STORE/rank%04d  ports=$PORTS"
else
  echo "   store=$STORE  port=$PORTS"
fi

# The knobs every rank shares. The per-rank ones (compose file, report path)
# are resolved inside the shim below, because mpirun's -x can only carry one
# value per variable.
# Leading `env`: mpirun treats a bare KEY=VALUE as the executable name.
COMMON_ENV=(env CLIO_WITH_RUNTIME=1
            CLIO_VPIC_TAG=vpic_insitu
            CLIO_VPIC_POOL=512.0
            CLIO_VPIC_CHUNK="$CHUNK"
            CLIO_VPIC_VERIFY=$([ "$VERIFY" = true ] && echo 1 || echo 0)
            CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}"
            VPIC_NX="$NCELL" VPIC_NY="$NCELL" VPIC_NZ="$NCELL" VPIC_NPPC="$NPPC"
            VPIC_STEPS="$STEPS" VPIC_INSITU=1 VPIC_INSITU_INT="$INT")

set +e
if [ "$RANKS" -eq 1 ]; then
  "${COMMON_ENV[@]}" CLIO_SERVER_CONF="$STORE/compose.yaml" \
      CLIO_VPIC_REPORT="$STORE/blobs.csv" \
      "$BIN" > "$STORE/vpic.log" 2> "$STORE/runtime.log"
  RC=$?
else
  # VPIC splits the domain along x by nproc(), so every rank has its own
  # subdomain but the SAME blob names; the per-rank stores are what keep them
  # apart, and the adapter refuses (exit 6) if two ranks are ever handed one.
  SHIM=$STORE/rank_launch.sh
  cat > "$SHIM" <<'SHIMEOF'
#!/usr/bin/env bash
R=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${MV2_COMM_WORLD_RANK:-0}}}
RD=$(printf '%s/rank%04d' "$CLIO_VPIC_STORE_ROOT" "$R")
export CLIO_SERVER_CONF="$RD/compose.yaml"
export CLIO_VPIC_REPORT="$RD/blobs.csv"
exec "$@" > "$RD/vpic.log" 2> "$RD/runtime.log"
SHIMEOF
  chmod +x "$SHIM"
  mpirun -np "$RANKS" --oversubscribe ${MPIRUN_EXTRA:-} \
      "${COMMON_ENV[@]}" CLIO_VPIC_STORE_ROOT="$STORE" \
      "$SHIM" "$BIN"
  RC=$?
fi
set -e

for RD in "${RANK_DIRS[@]}"; do
  [ -f "$RD/vpic.log" ] || continue
  grep -E "^\[clio-(vpic-)?insitu\]|VERIFIED|FAILED|REFUSING" "$RD/vpic.log" "$RD/runtime.log" 2>/dev/null | sed 's/^[^:]*://' || true
  if [ $RC -ne 0 ]; then
    echo "-- rank dir $RD, last lines of runtime.log:" >&2
    grep -vE "DEBUG|INFO" "$RD/runtime.log" | tail -6 >&2
  fi
done
if [ "$RANKS" -eq 1 ]; then
  echo "-- blobs: $STORE/blobs.csv   vpic: $STORE/vpic.log   runtime: $STORE/runtime.log   exit=$RC"
else
  echo "-- blobs: $STORE/rank%04d/blobs.csv ($RANKS of them)   logs under $STORE   exit=$RC"
fi
exit $RC
