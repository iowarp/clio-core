#!/usr/bin/env bash
# Build -- and optionally run -- a ported paged benchmark through SYCL.
#
#   ./run_newcoro_sycl.sh <dir> [name] [run args...]
#
# ONE SOURCE, TWO BACKENDS. The transpile is backend-independent, so this
# script does not repeat it: it calls build_newcoro.sh with
# NEWCORO_STAGE=transpile and compiles the SAME generated source that the
# CUDA edition compiles. A divergence between the two is therefore a real
# backend difference rather than two differently-transpiled programs.
#
# The binary lands in $CLIO_SYCL_BUILD_DIR/bin/clio_<name>_paged_newcoro_sycl,
# beside the libraries it links, so the distributed docker harness can mount
# and run it exactly as it does the CUDA one.
#
#   CLIO_BUILD_DIR       a CUDA-enabled clio build, for the transpile's -I/-D
#                        set (see build_newcoro.sh)     [<root>/build-gv|build]
#   CLIO_SYCL_BUILD_DIR  the SYCL build tree whose bin/ holds the libraries
#                        and receives the binary. DPC++ and nvcc cannot share
#                        a build tree, hence a second one.  [<root>/build-syclreg]
#   SYCL_CXX             the DPC++ driver                             [clang++]
#   SYCL_TARGET          the SYCL target triple             [nvptx64-nvidia-cuda]
#   GPU_ARCH             sm_XX, for an nvptx64 target   [from the CUDA build's cache]
#   CUDA_HOME            the toolkit the nvptx64 backend uses
#   NEWCORO_RUN          0 to stop after the link (what the docker harness
#                        wants: it runs the binary itself, in a container)
set -u

DIR="${1:?usage: run_newcoro_sycl.sh <dir> [name] [run args...]}"; shift
NAME="$DIR"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then NAME="$1"; shift; fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(cd "${HERE}/../../../.." && pwd)"

B="${CLIO_BUILD_DIR:-}"
if [ -z "$B" ]; then
  for cand in "${W}/build-gv" "${W}/build"; do
    [ -f "${cand}/compile_commands.json" ] && { B="$cand"; break; }
  done
fi
test -n "$B" && test -d "$B" || { echo "set CLIO_BUILD_DIR to a CUDA-enabled clio build" >&2; exit 1; }
B="$(cd "$B" && pwd)"

SB="${CLIO_SYCL_BUILD_DIR:-${W}/build-syclreg}"
test -d "${SB}/bin" || { echo "no SYCL build at ${SB} -- set CLIO_SYCL_BUILD_DIR" >&2; exit 1; }
SB="$(cd "$SB" && pwd)"

GVI=context-transfer-engine/adapter/gpu_vector/include
BD=context-transfer-engine/adapter/gpu_vector/benchmark/$DIR
G="${NEWCORO_GEN_DIR:-${B}/newcoro/gen_${NAME}}"
O="${NEWCORO_OVERLAY_DIR:-${B}/newcoro/ov_${NAME}}"
SRC="$G/$BD/clio_${NAME}_paged_newcoro.cc"
LOG="${SB}/newcoro/log"
mkdir -p "$LOG" "$SB/bin"

# Transpile on demand. Reusing an existing tree would silently compile a
# stale lowering after an edit to the benchmark or to the transpiler.
CLIO_BUILD_DIR="$B" NEWCORO_STAGE=transpile \
  "$HERE/build_newcoro.sh" "$DIR" "$NAME" > "$LOG/tp_$NAME.log" 2>&1 || {
  echo "TRANSPILE FAILED (see $LOG/tp_$NAME.log)"; tail -15 "$LOG/tp_$NAME.log"; exit 1; }
test -f "$SRC" || { echo "no transpiled source at $SRC" >&2; exit 1; }

SYCL_CXX="${SYCL_CXX:-clang++}"
command -v "$SYCL_CXX" > /dev/null || { echo "no SYCL compiler '$SYCL_CXX' -- set SYCL_CXX" >&2; exit 1; }
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)")")")}"
GPU_ARCH="${GPU_ARCH:-sm_$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES[^=]*=\([0-9]*\).*/\1/p' \
    "${B}/CMakeCache.txt" 2>/dev/null | head -1)}"
[ "${GPU_ARCH}" = "sm_" ] && GPU_ARCH=sm_80
SYCL_TARGET="${SYCL_TARGET:-nvptx64-nvidia-cuda}"

EXTRA="-I$W/context-runtime/include -I$W/context-runtime/modules/admin/include
       -I$W/context-runtime/modules/bdev/include
       -I$W/context-transfer-engine/core/include
       -I$W/context-transport-primitives/include
       -I$B/context-transport-primitives/src/include
       -I$W/context-transfer-engine/checkpoint/include
       -I$W/context-transfer-engine/compressor/include"
DEFS="-DCTP_ENABLE_SYCL=1 -DCTP_ENABLE_CUDA=0 -DCLIO_COROC -DCLIO_RUNTIME=1
      -DCLIO_RUN_HAS_POCO=0
      -DCTP_DEFAULT_THREAD_MODEL=ctp::thread::Pthread
      -DCTP_DEFAULT_THREAD_MODEL_GPU=ctp::thread::StdThread
      -DCTP_ENABLE_PTHREADS=1 -DCTP_ENABLE_CEREAL=1 -DCTP_LOG_LEVEL=1
      -DCTP_ENABLE_ZMQ=1 -DCTP_ENABLE_LIGHTBEAM=1 -DYAML_CPP_STATIC_DEFINE
      -DCTP_ENABLE_COMPRESS=0 -DCTP_ENABLE_NVCOMP=0 -DCTP_ENABLE_CUSZ=0
      -DCTP_ENABLE_CUSZP=0 -DCTP_ENABLE_LIBPRESSIO=0 -DCTP_ENABLE_NDZIP=0
      -DCTP_ENABLE_ZFP_SYCL=0 -DCTP_ENABLE_BLOSC2=0"

# ONE DEVICE IMAGE. yield_stack.h keeps its smem base in a device_global with
# device_image_scope, and DPC++ splits a TU with many kernels into several
# images -- which that property forbids. lammps_md has enough kernels to trip
# it; the others do not, so the split is disabled for all of them rather than
# left as a per-benchmark surprise.
SYCLT="-fsycl -fsycl-device-code-split=off -fsycl-targets=${SYCL_TARGET}"
case "$SYCL_TARGET" in
  nvptx64*) SYCLT="$SYCLT -Xsycl-target-backend --cuda-gpu-arch=${GPU_ARCH}
                  --cuda-path=${CUDA_HOME} -Wno-unknown-cuda-version" ;;
esac

echo "### [$NAME] SYCL compile+link for ${SYCL_TARGET} (${GPU_ARCH})"
# $ORIGIN, not the build path: the docker harness mounts this tree somewhere
# else entirely, and an absolute host rpath resolves to nothing there.
"$SYCL_CXX" $SYCLT -std=c++20 -O2 "$SRC" -o "$SB/bin/clio_${NAME}_paged_newcoro_sycl" \
  -I"$W/$BD" -I"$O/gv_include" -I"$O/gv_include/clio_cte/gpu_vector" \
  $EXTRA $DEFS \
  -L"$SB/bin" -Xlinker -rpath -Xlinker '$ORIGIN' \
  -lclio_cte_core_client -lclio_cte_core_runtime \
  -lclio_admin_client -lclio_bdev_client -lclio_run_cxx -lclio_ctp_host \
  -lzmq -lyaml-cpp -lpthread -ldl -lrt > "$LOG/sy_$NAME.log" 2>&1
test -x "$SB/bin/clio_${NAME}_paged_newcoro_sycl" || { echo "BUILD FAILED (see $LOG/sy_$NAME.log)";
  grep -E "error|undefined reference" "$LOG/sy_$NAME.log" | head -15; exit 1; }
echo "  LINKED OK: $SB/bin/clio_${NAME}_paged_newcoro_sycl"

if [ "${NEWCORO_RUN:-1}" = "0" ]; then
  echo "### [$NAME] NEWCORO_RUN=0 -- stopping before the run"
  exit 0
fi

echo "### [$NAME] run on SYCL (${ONEAPI_DEVICE_SELECTOR:-cuda:gpu}): $*"
RUNDIR="${NEWCORO_RUN_DIR:-$SB/newcoro/run_$NAME}"
mkdir -p "$RUNDIR" && cd "$RUNDIR" && rm -f ./*.yaml
ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-cuda:gpu}" \
  timeout "${NEWCORO_TIMEOUT:-1200}" "$SB/bin/clio_${NAME}_paged_newcoro_sycl" "$@" \
  > "$LOG/syrun_$NAME.log" 2>&1
rc=$?
grep -viE "HANGWATCH|PDF|INFO |WARNING" "$LOG/syrun_$NAME.log" | tail -12
echo "### [$NAME] exit=$rc"
exit $rc
