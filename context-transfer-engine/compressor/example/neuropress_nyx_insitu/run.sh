#!/usr/bin/env bash
# Nyx in situ: the simulation hands its GPU-resident hydro state to Clio.
#
#   ./run.sh [--ncell N] [--steps N] [--int N] [--chunk BYTES] [--store DIR]
#            [--hook insitu|plotfile|both] [--ghosts] [--verify] [--check-bound]
#            [--restart]
#            [--learn] [--explore K] [--threshold X] [--best]
#            [--static LIB] [--static-shuffle N] [--port N] [--bin PATH]
#            [--mpi] [--ranks N] [--max-grid N]
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
# MPI: --mpi runs the MPI build under mpirun; --ranks N implies it. EVERY RANK
# HOSTS ITS OWN CLIO RUNTIME, in its own process, with its own store directory
# and its own TCP port ($STORE/rank0000, rank0001, ...). That is not a
# preference: the compressor chimod has no cross-process wire format, so a
# rank that talked to a runtime in another process would store nothing while
# reporting success. See the README's "Q5: MPI". The adapter refuses rather
# than letting that happen.
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
RANKS=1 USE_MPI=false MAXGRID=
NYX_BIN_MPI=${NYX_BIN_MPI:-$HOME/src/Nyx/build-clio-mpi/Exec/HydroTests/nyx_HydroTests}
NYX_BIN_SET=false
VERIFY=false CHECK_BOUND=false RESTART=false
LEARN=false EXPLORE_K=0 THRESH=0.5 BEST=false STATIC_LIB= STATIC_SHUF=0
EXTRA=()
usage() { sed -n '2,30p' "$0"; }
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
    # LOSSY verification: the values, not a digest. A lossy round trip must
    # fail a digest by construction, so this is the only check that says
    # anything about correctness on a run with an error bound.
    --check-bound) CHECK_BOUND=true; shift;;
    --restart) RESTART=true; shift;;
    --learn) LEARN=true; shift;;
    --explore) EXPLORE_K=$2; shift 2;;
    --threshold) THRESH=$2; shift 2;;
    --best) BEST=true; shift;;
    --static) STATIC_LIB=$2; shift 2;;
    --static-shuffle) STATIC_SHUF=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --mpi) USE_MPI=true; shift;;
    --ranks) RANKS=$2; USE_MPI=true; shift 2;;
    --max-grid) MAXGRID=$2; shift 2;;
    --bin) NYX_BIN=$2; NYX_BIN_SET=true; shift 2;;
    -h|--help) usage; exit 0;;
    *) EXTRA+=("$1"); shift;;              # passed to Nyx verbatim
  esac
done

# The MPI and non-MPI builds are separate trees on purpose: build-clio is what
# every single-rank number in the README was measured on and must keep working.
if [ "$USE_MPI" = true ] && [ "$NYX_BIN_SET" != true ]; then NYX_BIN=$NYX_BIN_MPI; fi
[ -f "$LIB" ] || { echo "missing $LIB -- cmake --build $BUILD --target clio_nyx_insitu" >&2; exit 1; }
[ -x "$NYX_BIN" ] || { echo "missing Nyx: $NYX_BIN (see README.md for the build)" >&2; exit 1; }
if [ "$USE_MPI" = true ]; then
  command -v mpirun >/dev/null || { echo "--mpi needs mpirun on PATH" >&2; exit 1; }
fi
[ -d "$WEIGHTS" ] || { echo "missing weights: $WEIGHTS" >&2; exit 1; }
DECK=$(dirname "$NYX_BIN")/inputs.3d.sph.sedov
[ -f "$DECK" ] || { echo "missing deck: $DECK" >&2; exit 1; }

# Every Clio runtime binds a TCP port and one process per port is the limit.
# A collision is silent from the client's side -- it dies before writing
# anything -- so pick free ones unless the caller pinned a base.
# Under MPI there is one runtime PER RANK, so there are $RANKS ports to pick.
PORTS=$(python3 - "$RANKS" "${PORT:-0}" <<'PY'
import socket, sys
n = int(sys.argv[1]); base = int(sys.argv[2])
if base:
    # Pinned base: rank r gets base+r. Reported, not probed -- the caller
    # asked for these, and a collision produces the named runtime error.
    print(" ".join(str(base + r) for r in range(n)))
else:
    ss = [socket.socket() for _ in range(n)]
    for x in ss: x.bind(("", 0))
    print(" ".join(str(x.getsockname()[1]) for x in ss))
    for x in ss: x.close()
PY
)
PORT=$(echo "$PORTS" | awk '{print $1}')

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

write_compose() {   # $1 = store dir, $2 = port
local SDIR=$1 SPORT=$2
cat > "$SDIR/compose.yaml" <<YAML
networking:
  port: $SPORT
runtime:
  num_threads: ${THREADS:-4}
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "$SDIR/chi_bdev.dat"
    pool_query: local
    pool_id: "301.0"
    bdev_type: file
    path: "$SDIR/chi_bdev.dat"
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
      - path: "$SDIR/cte_tier.dat"
        bdev_type: "file"
        capacity_limit: "${TIER_MB}MB"
        score: 1.0
        persistence_level: "temporary"
    performance:
      metadata_log_path: "$SDIR/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
YAML
}

# One store, one compose file, one port PER RANK. Under MPI each rank hosts
# its own Clio runtime; sharing a store would mean sharing a bdev file, a tier
# file, a metadata log and a TCP port between processes, and the loser of the
# bind race stores nothing without saying so.
RANK_DIRS=()
for ((r=0; r<RANKS; r++)); do
  if [ "$RANKS" -eq 1 ]; then RD=$STORE; else RD=$(printf '%s/rank%04d' "$STORE" "$r"); fi
  RANK_DIRS+=("$RD")
  mkdir -p "$RD"
  if [ "$RESTART" != true ]; then
    write_compose "$RD" "$(echo "$PORTS" | awk -v i=$((r+1)) '{print $i}')"
  fi
done

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# max_grid_size decides how many FArrayBoxes the domain is cut into, and
# therefore how many of the ranks own any data at all. The default is the
# whole domain in ONE box (what every single-rank number was measured with);
# under MPI that would leave ranks 1..N-1 with nothing to store, so halve it
# until there are at least as many boxes as ranks.
if [ -z "$MAXGRID" ]; then
  MAXGRID=$NCELL
  if [ "$RANKS" -gt 1 ]; then
    NBOX=1
    while [ "$NBOX" -lt "$RANKS" ] && [ $((MAXGRID % 2)) -eq 0 ] && [ "$MAXGRID" -gt 8 ]; do
      MAXGRID=$((MAXGRID / 2))
      NB=$(( NCELL / MAXGRID )); NBOX=$(( NB * NB * NB ))
    done
  fi
fi
NYX_ARGS=(amr.n_cell="$NCELL $NCELL $NCELL" amr.max_grid_size="$MAXGRID"
          max_step="$STEPS" amr.check_int=-1 nyx.v=0 amr.v=0)

# AMReX's device arena reserves 3/4 of the GPU up front, divided among the
# ranks sharing that GPU -- and it reserves it whether it needs it or not
# (measured at 128^3/4 ranks: 7,583 MB reserved per rank, 474 MB actually
# used, 162 MB left free on a 40 GB A100). With one Clio runtime PER RANK,
# and nvcomp needing scratch of its own, that is what runs out: the codec
# starts failing, blobs fall back to being stored raw, and some of them then
# fail to read back. So under MPI, cap the reservation at something the
# working set fits in and leave the rest of the card to the compressor.
# Single-rank runs are untouched: they are the configuration every existing
# number in the README was measured on.
if [ "$RANKS" -gt 1 ] && [[ ! " ${EXTRA[*]:-} " =~ the_arena_init_size ]]; then
  ARENA_MB=$(( FRAME_MB * 16 )); [ "$ARENA_MB" -lt 2048 ] && ARENA_MB=2048
  NYX_ARGS+=(amrex.the_arena_init_size=$(( ARENA_MB * 1048576 )))
fi
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
echo "   hook=$HOOK chunk=$CHUNK ghosts=$GHOSTS max_grid=$MAXGRID  (~${FRAME_MB} MiB/frame)"
if [ -n "$STATIC_LIB" ]; then echo "   STATIC codec=$STATIC_LIB shuffle=$STATIC_SHUF (NeuroPress bypassed)"
elif [ "$BEST" = true ]; then echo "   BEST mode (exhaustive, ratio-only)"
else echo "   learn=$NP_LEARN explore=$NP_EXPLORE k=$EXPLORE_K"; fi
if [ "$RANKS" -gt 1 ]; then
  echo "   ranks=$RANKS (mpirun), one Clio runtime PER RANK, arena=${ARENA_MB:-default} MB/rank"
  echo "   stores=$STORE/rank%04d  ports=$PORTS"
else
  echo "   store=$STORE  port=$PORT${USE_MPI:+  mpi=$USE_MPI}"
fi

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
[ "$CHECK_BOUND" = true ] && NP_ENV+=(CLIO_NYX_CHECK_BOUND=1)
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
if [ "$USE_MPI" != true ]; then
  "${NP_ENV[@]}" "$NYX_BIN" "$DECK" "${NYX_ARGS[@]}" "${EXTRA[@]}" \
      > "$STORE/nyx.log" 2> "$STORE/runtime.log"
  RC=$?
else
  # Each rank needs a DIFFERENT CLIO_SERVER_CONF (its own store and port) and
  # a different report file, and mpirun's -x can only pass one value for a
  # variable. So the per-rank values are resolved inside a shim, from the
  # rank id the launcher exports. Rank stdout/stderr are interleaved into one
  # log for rank 0 only (AMReX's own convention) and per rank otherwise, so a
  # refusal on rank 3 is still readable.
  SHIM=$STORE/rank_launch.sh
  cat > "$SHIM" <<'SHIMEOF'
#!/usr/bin/env bash
R=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${MV2_COMM_WORLD_RANK:-0}}}
N=${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-1}}
if [ "$N" -eq 1 ] && [ "$CLIO_NYX_MPI_STORE_PER_RANK" != 1 ]; then
  RD=$CLIO_NYX_STORE_ROOT
else
  RD=$(printf '%s/rank%04d' "$CLIO_NYX_STORE_ROOT" "$R")
fi
export CLIO_SERVER_CONF="$RD/compose.yaml"
export CLIO_NYX_REPORT="$RD/blobs.csv"
exec "$@" > "$RD/nyx.log" 2> "$RD/runtime.log"
SHIMEOF
  chmod +x "$SHIM"
  MPI_ENV=("${NP_ENV[@]}" CLIO_NYX_STORE_ROOT="$STORE"
           CLIO_NYX_MPI_STORE_PER_RANK=$([ "$RANKS" -gt 1 ] && echo 1 || echo 0))
  mpirun -np "$RANKS" --oversubscribe ${MPIRUN_EXTRA:-} \
      "${MPI_ENV[@]}" "$SHIM" "$NYX_BIN" "$DECK" "${NYX_ARGS[@]}" "${EXTRA[@]}"
  RC=$?
fi
set -e
for RD in "${RANK_DIRS[@]}"; do
  [ -f "$RD/nyx.log" ] || continue
  grep -E "^\[clio-nyx-insitu\]|^\[ptr-probe\]" "$RD/nyx.log" || true
  if [ $RC -ne 0 ]; then
    grep -E "REFUSING|Abort|amrex::Abort" "$RD/nyx.log" "$RD/runtime.log" 2>/dev/null | head -5 || true
  fi
done
if [ $RC -ne 0 ]; then
  echo "-- Nyx failed (rc=$RC); last lines of ${RANK_DIRS[0]}/runtime.log:"
  grep -vE "DEBUG|INFO" "${RANK_DIRS[0]}/runtime.log" 2>/dev/null | tail -15
  for RD in "${RANK_DIRS[@]}"; do
    if grep -q "Address already in use" "$RD/runtime.log" 2>/dev/null; then
      echo "-- a port in [$PORTS] is held by another Clio runtime (ss -ltnp); rerun with --port N"
      break
    fi
  done
fi
if [ "$RANKS" -gt 1 ]; then
  echo "-- blobs: $STORE/rank%04d/blobs.rank%04d.csv ($RANKS of them)   logs under $STORE   exit=$RC"
else
  echo "-- blobs: $STORE/blobs.csv   nyx: $STORE/nyx.log   runtime: $STORE/runtime.log   exit=$RC"
fi
exit $RC
