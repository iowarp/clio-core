#!/usr/bin/env bash
# Phase 1 of 2: run a STOCK LAMMPS through Clio + NeuroPress into a PERSISTENT
# store, then exit. Run lmp_read.sh afterwards, as a separate process with its
# own runtime, to read it back.
#
# The LAMMPS binary here is unmodified and links nothing from Clio
# (`ldd lmp | grep -c clio` is 0). Compression arrives entirely through
# HDF5_VOL_CONNECTOR -- see README.md for how to build it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BOX=80 STEPS=300 GAP=50 CHUNK=4194304
STORE=${STORE:-$HERE/store}
LEARN=false EXPLORE_K=0 THRESH=0.5 BEST=false STATIC_LIB= CPU=false
usage() {
  cat <<USAGE
usage: $0 [options]
  --box N          lattice cells per side (default 80 -> 4*N^3 = 2,048,000 atoms)
  --steps N        timesteps (default 300)
  --gap N          dump interval (default 50; frames = steps/gap + 1)
  --chunk BYTES    CTE chunk size (default 4194304)
  --store DIR      persistent store (default ./store)
  --learn          online learning (SGD from measured outcomes)
  --explore K      exploration with K candidates (implies --learn)
  --threshold X    cost error above which exploration fires (default 0.5; 0 = every chunk)
  --best           best mode: exhaustive, ratio-only ranking, ~32x slower
  --static LIB     pin every chunk to LIB and bypass NeuroPress entirely
                   (e.g. nvcomp-zstd) -- the fixed-codec control
  --cpu            run LAMMPS on the CPU instead of KOKKOS/GPU. Slower, but
                   BIT-REPRODUCIBLE: the GPU run is not, so two GPU runs
                   produce different chunk bytes and cannot be compared
                   per-chunk. Required for any A/B across configurations.
Modes: default is inference (predict and store, nothing measured back).
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --box) BOX=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --store) STORE=$2; shift 2;;
    --learn) LEARN=true; shift;;
    --explore) EXPLORE_K=$2; shift 2;;
    --threshold) THRESH=$2; shift 2;;
    --best) BEST=true; shift;;
    --static) STATIC_LIB=$2; shift 2;;
    --cpu) CPU=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 2;;
  esac
done
export BOX STEPS GAP CHUNK STORE LEARN EXPLORE_K THRESH BEST STATIC_LIB CPU

# A fresh store every write: a stale tier from an earlier run would let the
# reader "pass" on data this run never produced.
rm -rf "$STORE"
source "$HERE/lmp_common.sh"
lmp_setup || exit 1

NATOMS=$(( 4 * BOX * BOX * BOX ))
FRAMES=$(( STEPS / GAP + 1 ))
MIB=$(( NATOMS * 3 * 8 * 3 * FRAMES / 1048576 ))
echo "LAMMPS -> HDF5 -> Clio -> NeuroPress"
echo "  atoms=$NATOMS frames=$FRAMES  ~${MIB} MiB across 3 fields (float64)"
if [ -n "$STATIC_LIB" ]; then
  echo "  chunk=$CHUNK  STATIC codec=$STATIC_LIB (NeuroPress bypassed)"
elif [ "$BEST" = true ]; then
  echo "  chunk=$CHUNK  BEST mode: explore=true k=31 (forced by the runtime), SGD off"
else
  echo "  chunk=$CHUNK  learn=$NP_LEARN explore=$NP_EXPLORE k=$EXPLORE_K best=$BEST"
fi
[ "$CPU" = true ] && echo "  device=CPU (bit-reproducible)" || echo "  device=GPU/KOKKOS (NOT bit-reproducible)"
echo "  store=$STORE"

# -log into the store: LAMMPS writes log.lammps into the working directory by
# default, which drops a build artefact in the source tree when this is run
# from where it lives.
if [ "$CPU" = true ]; then KOKKOS=(); else KOKKOS=(-k on g 1 -sf kk); fi
"${NP_ENV[@]}" "$LMP" "${KOKKOS[@]}" \
  -log "$STORE/log.lammps" \
  -in "$HERE/in.melt_clio" \
  -var BOX "$BOX" -var GAP "$GAP" -var STEPS "$STEPS" \
  -var OUT "$STORE/melt.h5" > "$STORE/write.log" 2>&1
RC=$?
echo
grep -E 'Total wall time' "$STORE/write.log" | tail -1 || true
if [ $RC -ne 0 ]; then echo "LAMMPS FAILED (rc=$RC); see $STORE/write.log" >&2; exit $RC; fi

# Chunk-level outcome, when the connector was built with the path trace on.
KEPT=$(grep -c 'kept=1' "$STORE/write.log" 2>/dev/null || true)
RAW=$(grep -c 'kept=0' "$STORE/write.log" 2>/dev/null || true)
if [ "${KEPT:-0}" -gt 0 ] || [ "${RAW:-0}" -gt 0 ]; then
  echo "  chunks compressed: $KEPT   stored raw (codec did not shrink them): $RAW"
fi
echo
echo "now read it back:  ./lmp_read.sh --store $STORE --chunk $CHUNK"
