#!/usr/bin/env bash
# SYCL DEVICE COMPILE of a ported paged benchmark -- spir64 by default.
#
#   ./build_newcoro_sycl.sh <dir> [name]
#
# COMPILE ONLY, deliberately. Linking would need the whole CTE/runtime stack
# rebuilt with CTP_ENABLE_SYCL, which is a separate build tree and a separate
# job (run_newcoro_sycl.sh, which links and runs against one). What this
# answers is the codegen question on its own -- does the transpiled state
# machine, and the parallel_for that enters it, survive a SYCL device
# compile -- and it answers it from a plain CUDA build tree, with no DPC++
# runtime and no GPU.
#
#   CLIO_BUILD_DIR   a CUDA-enabled clio build, for the -I/-D set  [build-gv|build]
#   SYCL_CXX         the DPC++ driver                                   [clang++]
#   SYCL_TARGET      the device target                                   [spir64]
set -u

DIR="${1:?usage: build_newcoro_sycl.sh <dir> [name]}"; shift
NAME="${1:-$DIR}"

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

GVI=context-transfer-engine/adapter/gpu_vector/include
BD=context-transfer-engine/adapter/gpu_vector/benchmark/$DIR
G="${NEWCORO_GEN_DIR:-${B}/newcoro/gen_${NAME}}"
O="${NEWCORO_OVERLAY_DIR:-${B}/newcoro/ov_${NAME}}"
SRC="$G/$BD/clio_${NAME}_paged_newcoro.cc"
LOG="${B}/newcoro/log"
mkdir -p "$LOG"

# Transpile on demand, through the one script that knows how: reusing an
# existing tree would silently compile a stale lowering after an edit.
CLIO_BUILD_DIR="$B" NEWCORO_STAGE=transpile \
  "$HERE/build_newcoro.sh" "$DIR" "$NAME" > "$LOG/tps_$NAME.log" 2>&1 || {
  echo "TRANSPILE FAILED (see $LOG/tps_$NAME.log)"; tail -15 "$LOG/tps_$NAME.log"; exit 1; }
test -f "$SRC" || { echo "no transpiled source at $SRC" >&2; exit 1; }

SYCL_CXX="${SYCL_CXX:-clang++}"
SYCL_TARGET="${SYCL_TARGET:-spir64}"

EXTRA="-I$W/context-runtime/include -I$W/context-runtime/modules/admin/include
       -I$W/context-runtime/modules/bdev/include
       -I$W/context-transfer-engine/core/include
       -I$W/context-transport-primitives/include
       -I$B/context-transport-primitives/src/include
       -I$W/context-transfer-engine/checkpoint/include
       -I$W/context-transfer-engine/compressor/include"
DEFS="-DCTP_ENABLE_SYCL=1 -DCTP_ENABLE_CUDA=0 -DCLIO_COROC -DCLIO_RUNTIME=1
      -DCTP_DEFAULT_THREAD_MODEL=ctp::thread::Pthread
      -DCTP_DEFAULT_THREAD_MODEL_GPU=ctp::thread::StdThread
      -DCTP_ENABLE_PTHREADS=1 -DCTP_ENABLE_CEREAL=1 -DCTP_LOG_LEVEL=1
      -DCTP_ENABLE_ZMQ=1 -DCTP_ENABLE_LIGHTBEAM=1 -DYAML_CPP_STATIC_DEFINE
      -DCTP_ENABLE_COMPRESS=0 -DCTP_ENABLE_NVCOMP=0 -DCTP_ENABLE_CUSZ=0
      -DCTP_ENABLE_CUSZP=0 -DCTP_ENABLE_LIBPRESSIO=0 -DCTP_ENABLE_NDZIP=0
      -DCTP_ENABLE_ZFP_SYCL=0 -DCTP_ENABLE_BLOSC2=0"

echo "### [$NAME] SYCL device compile (${SYCL_TARGET})"
"$SYCL_CXX" -fsycl -fsycl-targets="${SYCL_TARGET}" -fsycl-device-code-split=off \
  -std=c++20 -O2 -c "$SRC" -o "$B/newcoro/${NAME}_sycl.o" \
  -I"$W/$BD" -I"$O/gv_include" -I"$O/gv_include/clio_cte/gpu_vector" \
  $EXTRA $DEFS > "$LOG/syc_$NAME.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "SYCL COMPILE FAILED (see $LOG/syc_$NAME.log)"
  grep -E "error" "$LOG/syc_$NAME.log" | head -20
  exit $rc
fi
echo "  SYCL OK: $(stat -c%s "$B/newcoro/${NAME}_sycl.o") bytes"
