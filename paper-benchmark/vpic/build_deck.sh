#!/usr/bin/env bash
# Compile the Clio VPIC deck.
#
#   ./build_deck.sh [--vpic DIR] [--deck FILE]
#
# VPIC compiles a deck by textually including it into deck/wrapper.cc and
# linking against libvpic + Kokkos, which its generated bin/vpic script does.
# That script is used as-is here, with ONE addition: -lcuda.
#
# Why: Kokkos' CUDA backend calls the CUDA DRIVER API (cuStreamGetCtx,
# cuCtxPushCurrent_v2, cuCtxGetDevice) from Kokkos_Cuda_Instance.cpp, but the
# stock link line carries only the runtime (-lcudart via nvcc_wrapper). Any
# deck therefore fails at link with undefined references to those three
# symbols. Upstream NeuroPress's own VPIC build script passes -lcuda for the
# same reason.
#
# The stock script reads its Kokkos flags from generated files, so there is no
# environment hook to inject a library through -- hence rewriting the command
# rather than setting a variable. VPIC's tree is left untouched.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VPIC_DIR=${VPIC_DIR:-$HOME/src/vpic-kokkos}
VPIC_BUILD=${VPIC_BUILD:-$VPIC_DIR/build-clio}
DECK=${DECK:-$HERE/weibel_clio.cxx}
while [ $# -gt 0 ]; do
  case "$1" in
    --vpic) VPIC_DIR=$2; VPIC_BUILD=$2/build-clio; shift 2;;
    --build) VPIC_BUILD=$2; shift 2;;
    --deck) DECK=$2; shift 2;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

STOCK=$VPIC_BUILD/bin/vpic
if [ ! -x "$STOCK" ]; then
  cat >&2 <<MSG
missing VPIC deck compiler: $STOCK

Build VPIC first:
  git clone --depth 1 --recursive https://github.com/lanl/vpic-kokkos.git ~/src/vpic-kokkos
  cd ~/src/vpic-kokkos
  cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release \\
        -DENABLE_KOKKOS_CUDA=ON -DENABLE_KOKKOS_OPENMP=OFF \\
        -DBUILD_INTERNAL_KOKKOS=ON -DKokkos_ARCH_AMPERE80=ON \\
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \\
        -DCMAKE_CXX_COMPILER=\$PWD/kokkos/bin/nvcc_wrapper
  cmake --build build-clio -j
MSG
  exit 1
fi
[ -f "$DECK" ] || { echo "missing deck: $DECK" >&2; exit 1; }

# Find libcuda: the driver library, not the CUDA-toolkit stub. Linking the stub
# builds but cannot run.
CUDA_DRIVER_DIR=""
for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /lib/x86_64-linux-gnu; do
  [ -e "$d/libcuda.so.1" ] && { CUDA_DRIVER_DIR=$d; break; }
done
[ -n "$CUDA_DRIVER_DIR" ] || { echo "cannot find libcuda.so.1 (NVIDIA driver)" >&2; exit 1; }

PATCHED=$(mktemp /tmp/vpic_build_XXXXXX.sh)
trap 'rm -f "$PATCHED"' EXIT
sed "s|-lpthread -ldl|-lpthread -ldl -L$CUDA_DRIVER_DIR -lcuda|" "$STOCK" > "$PATCHED"
grep -q -- "-lcuda" "$PATCHED" || { echo "failed to inject -lcuda into $STOCK" >&2; exit 1; }
chmod +x "$PATCHED"

cd "$HERE"
echo "== compiling $(basename "$DECK") (vpic $VPIC_BUILD, +$CUDA_DRIVER_DIR/libcuda)"
"$PATCHED" "$DECK" > "$HERE/build_deck.log" 2>&1 || {
  echo "deck compile FAILED; last lines of build_deck.log:" >&2
  grep -iE "error" "$HERE/build_deck.log" | head -10 >&2
  exit 1; }

BIN="$HERE/$(basename "${DECK%.*}").Linux"
[ -x "$BIN" ] || { echo "no binary produced at $BIN" >&2; exit 1; }
echo "   built: $BIN"
