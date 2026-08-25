#!/usr/bin/env bash
# Nyx in situ: the simulation hands its GPU-resident hydro state to Clio.
#
#   ./run.sh [--ncell N] [--steps N] [--int N] [--chunk BYTES] [--store DIR]
#            [--hook insitu|plotfile|both] [--ghosts] [--verify] [--restart]
#            [--learn] [--explore K] [--threshold X] [--best]
#            [--static LIB] [--static-shuffle N] [--port N] [--bin PATH]
#
# One process: Nyx runs the Sedov blast on the GPU, Clio's runtime is composed
# from $STORE/compose.yaml and hosted in that same process
# (CLIO_WITH_RUNTIME=1), and a patched Nyx::updateInSitu() hands
# fab.dataPtr(comp) -- AMReX device memory -- to libclio_nyx_insitu.so, which
# it dlopens. No plotfile is written: with --hook insitu the compressed tier
# is the only copy of the data in existence.
#
# Needs a Nyx built with BOTH patches applied (see README.md):
#   paper-benchmark/nyx/patches/nyx-raw-field-dump.patch   (route B, first)
#   patches/nyx-clio-insitu.patch                          (this one)
#
# The compose file and every NeuroPress knob are the same as
# ../neuropress_lammps_lib/run.sh and ../../../../paper-benchmark/nyx, so the
# ratios are comparable across all of them.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
LIB=$BUILD/bin/libclio_nyx_insitu.so
WEIGHTS=$REPO/context-transport-primitives/src/compress/model/weights
NYX_BIN=${NYX_BIN:-$HOME/src/Nyx/build-clio/Exec/HydroTests/nyx_HydroTests}

NCELL=128 STEPS=200 INT=10 CHUNK=4194304 HOOK=insitu GHOSTS=false
STORE=${STORE:-$HERE/store}
PORT=${PORT:-}
VERIFY=false RESTART=false
LEARN=false EXPLORE_K=0 THRESH=0.5 BEST=false STATIC_LIB= STATIC_SHUF=0
EXTRA=()
usage() { sed -n '2,22p' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int) INT=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --store) STORE=$2; shift 2;;
    --hook) HOOK=$2; shift 2;;
    --ghosts) GHOSTS=true; shift;;
    --verify) VERIFY=true; shift;;
    --restart) RESTART=true; shift;;
    --learn) LEARN=true; shift;;
    --explore) EXPLORE_K=$2; shift 2;;
    --threshold) THRESH=$2; shift 2;;
    --best) BEST=true; shift;;
    --static) STATIC_LIB=$2; shift 2;;
    --static-shuffle) STATIC_SHUF=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --bin) NYX_BIN=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) EXTRA+=("$1"); shift;;              # passed to Nyx verbatim
  esac
done

[ -f "$LIB" ] || { echo "missing $LIB -- cmake --build $BUILD --target clio_nyx_insitu" >&2; exit 1; }
[ -x "$NYX_BIN" ] || { echo "missing Nyx: $NYX_BIN (see README.md for the build)" >&2; exit 1; }
[ -d "$WEIGHTS" ] || { echo "missing weights: $WEIGHTS" >&2; exit 1; }
DECK=$(dirname "$NYX_BIN")/inputs.3d.sph.sedov
[ -f "$DECK" ] || { echo "missing deck: $DECK" >&2; exit 1; }

# Every Clio runtime binds a TCP port and one process per port is the limit.
# A collision is silent from the client's side -- it dies before writing
# anything -- so pick a free one unless the caller pinned it.
if [ -z "$PORT" ]; then
  PORT=$(python3 - <<'PY'
import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()
PY
)
fi

if [ "$RESTART" != true ]; then rm -rf "$STORE"; fi
mkdir -p "$STORE"

# Sizing: float32 (AMReX_PRECISION=SINGLE), 6 hydro components, ncell^3 cells.
FRAME_MB=$(( NCELL * NCELL * NCELL * 6 * 4 / 1048576 )); [ "$FRAME_MB" -lt 1 ] && FRAME_MB=1
NFRAMES=$(( STEPS / INT )); [ "$NFRAMES" -lt 1 ] && NFRAMES=1
TIER_MB=$(( FRAME_MB * NFRAMES + 512 ))
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
  num_threads: ${THREADS:-4}
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

NYX_ARGS=(amr.n_cell="$NCELL $NCELL $NCELL" amr.max_grid_size="$NCELL"
          max_step="$STEPS" amr.check_int=-1 nyx.v=0 amr.v=0)
case "$HOOK" in
  # plot_int=-1 alone is NOT enough to stop Nyx writing: nyx_main.cpp writes a
  # final checkpoint AND a final plotfile after the loop, unconditionally
  # (nyx_main.cpp:165-171) -- 360 MB at 128^3. plot_files_output=0 /
  # checkpoint_files_output=0 gate Amr::writePlotFile and Amr::checkPoint
  # themselves, which is what makes "the compressed tier is the only copy"
  # actually true for this hook.
  insitu)   NYX_ARGS+=(insitu.int="$INT" insitu.start=0 amr.plot_int=-1
                       amr.plot_files_output=0 amr.checkpoint_files_output=0);;
  # The plotfile hook needs the plotfile: it is the reference the blobs are
  # compared against (crosscheck_plotfile.sh).
  plotfile) NYX_ARGS+=(amr.plot_int="$INT" amr.checkpoint_files_output=0);;
  both)     NYX_ARGS+=(insitu.int="$INT" insitu.start=0 amr.plot_int="$INT"
                       amr.checkpoint_files_output=0);;
  *) echo "bad --hook $HOOK" >&2; exit 2;;
esac

echo "== Nyx in situ + Clio in one process: ${NCELL}^3, $STEPS steps, every $INT -> ~$NFRAMES frames"
echo "   hook=$HOOK chunk=$CHUNK ghosts=$GHOSTS  (~${FRAME_MB} MiB/frame)"
if [ -n "$STATIC_LIB" ]; then echo "   STATIC codec=$STATIC_LIB shuffle=$STATIC_SHUF (NeuroPress bypassed)"
elif [ "$BEST" = true ]; then echo "   BEST mode (exhaustive, ratio-only)"
else echo "   learn=$NP_LEARN explore=$NP_EXPLORE k=$EXPLORE_K"; fi
echo "   store=$STORE  port=$PORT"

NP_ENV=(env CLIO_SERVER_CONF="$STORE/compose.yaml"
        CLIO_WITH_RUNTIME=1
        # "warning", not "warn": Logger::Logger matches the full level names
        # and then falls through to std::stoi, which throws on "warn" and
        # leaves the compile-time default (kDebug) in place.
        CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}"
        NYX_CLIO_INSITU=1
        NYX_CLIO_INSITU_LIB="$LIB"
        NYX_CLIO_INSITU_HOOK="$HOOK"
        CLIO_NYX_TAG="${CLIO_NYX_TAG:-nyx_insitu}"
        CLIO_NYX_POOL=512.0
        CLIO_NYX_CHUNK="$CHUNK"
        CLIO_NYX_REPORT="$STORE/blobs.csv")
[ "$GHOSTS" = true ] && NP_ENV+=(NYX_CLIO_INSITU_GHOSTS=1)
[ "$VERIFY" = true ] && NP_ENV+=(CLIO_NYX_VERIFY=1)
[ -n "${RAW_DIR:-}" ] && { mkdir -p "$RAW_DIR"; NP_ENV+=(CLIO_NYX_RAW_DIR="$RAW_DIR"); }
[ -n "${NYX_DUMP_DIR:-}" ] && NP_ENV+=(NYX_DUMP_FIELDS=1 NYX_DUMP_DIR="$NYX_DUMP_DIR")
[ "$RESTART" = true ] && NP_ENV+=(CLIO_RESTART=1)

# Nyx writes plotfiles/checkpoints into the working directory; keep them out
# of the source tree. WORK=DIR keeps them (what crosscheck_plotfile.sh needs).
if [ -n "${WORK:-}" ]; then mkdir -p "$WORK"; else
  WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
fi
cd "$WORK"

set +e
"${NP_ENV[@]}" "$NYX_BIN" "$DECK" "${NYX_ARGS[@]}" "${EXTRA[@]}" \
    > "$STORE/nyx.log" 2> "$STORE/runtime.log"
RC=$?
set -e
grep -E "^\[clio-nyx-insitu\]|^\[ptr-probe\]" "$STORE/nyx.log" || true
if [ $RC -ne 0 ]; then
  echo "-- Nyx failed (rc=$RC); last lines of $STORE/runtime.log:"
  grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -15
  if grep -q "Address already in use" "$STORE/runtime.log"; then
    echo "-- port $PORT is held by another Clio runtime (ss -ltnp | grep $PORT); rerun with --port N"
  fi
fi
echo "-- blobs: $STORE/blobs.csv   nyx: $STORE/nyx.log   runtime: $STORE/runtime.log   exit=$RC"
exit $RC
