#!/usr/bin/env bash
# Transpile, compile with NVCC, link and (optionally) run one ported paged
# benchmark -- the clio-coroc edition, whose coroutines are lowered by the
# source-to-source pass instead of by the compiler's co_await.
#
#   ./build_newcoro.sh <dir> [name] [run args...]
#   ./build_newcoro.sh kmeans --data-mb 64 --blocks 8
#
# `name` defaults to `dir`; the source is
# <dir>/clio_<name>_paged_newcoro.cc and the binary lands in
# $CLIO_BUILD_DIR/bin/clio_<name>_paged_newcoro, beside the libraries it
# links, so the distributed docker harness can mount and run it.
#
# PORTABLE BY DISCOVERY, not by hard-coded paths: everything below is either
# derived from this script's own location or overridable from the
# environment, because the pipeline has to run on a developer box, in the
# devcontainer, under ctest and on Aurora, and those agree on nothing.
#
#   CLIO_BUILD_DIR   an existing CUDA-enabled clio build. Its
#                    compile_commands.json supplies the CTE -I/-D set, its
#                    CMakeCache the GPU arch, its bin/ the libraries and the
#                    output directory.        [<root>/build-gv, else build]
#   COROC            the transpiler                  [<root>/build-coroc/clio-coroc]
#   CUDA_HOME        the toolkit nvcc comes from     [from `which nvcc`]
#   COROC_CUDA_HOME  the toolkit the TRANSPILER's clang parses against. It
#                    often has to be OLDER than CUDA_HOME: clang refuses a
#                    CUDA it does not know, and clang 18 cannot read the
#                    CUDA 13 headers at all. Only the parse is affected --
#                    nvcc below still builds against CUDA_HOME.   [CUDA_HOME]
#   GPU_ARCH         sm_XX for nvcc             [from CMAKE_CUDA_ARCHITECTURES]
#   NEWCORO_MAXREG   the per-thread register ceiling                      [64]
#   NEWCORO_STAGE    transpile, to stop after the source-to-source pass and
#                    the include overlay. run_newcoro_sycl.sh uses this: the
#                    transpile is backend-independent, so the SYCL edition
#                    reuses this script's output rather than repeating it.
#   NEWCORO_RUN      0 to stop after the link (what CI and the docker harness
#                    want: they run the binary themselves, elsewhere)
set -u

DIR="${1:?usage: build_newcoro.sh <dir> [name] [run args...]}"; shift
NAME="$DIR"
# A bare second word that is not a flag is the benchmark name; anything
# starting with '-' is already a run argument.
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then NAME="$1"; shift; fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(cd "${HERE}/../../../.." && pwd)"
test -f "${W}/CMakeLists.txt" || { echo "not a clio checkout: ${W}" >&2; exit 1; }

# The build tree: whatever CLIO_BUILD_DIR names, else the first of the usual
# suspects that actually carries a compile_commands.json.
B="${CLIO_BUILD_DIR:-}"
if [ -z "$B" ]; then
  for cand in "${W}/build-gv" "${W}/build"; do
    [ -f "${cand}/compile_commands.json" ] && { B="$cand"; break; }
  done
fi
test -n "$B" && test -f "${B}/compile_commands.json" || {
  echo "no compile_commands.json in '${B:-<none found>}' -- set CLIO_BUILD_DIR" \
       "to a CUDA-enabled clio build" >&2; exit 1; }
B="$(cd "$B" && pwd)"

COROC="${COROC:-${W}/build-coroc/clio-coroc}"
test -x "${COROC}" || { echo "no transpiler at ${COROC} -- run tools/coroc/build.sh" >&2; exit 1; }

GVI=context-transfer-engine/adapter/gpu_vector/include
BD=context-transfer-engine/adapter/gpu_vector/benchmark/$DIR
SRC="${W}/${BD}/clio_${NAME}_paged_newcoro.cc"
test -f "${SRC}" || { echo "no such benchmark source: ${SRC}" >&2; exit 1; }

# Scratch trees, one pair per benchmark so several can be built in parallel.
G="${NEWCORO_GEN_DIR:-${B}/newcoro/gen_${NAME}}"
O="${NEWCORO_OVERLAY_DIR:-${B}/newcoro/ov_${NAME}}"
LOG="${B}/newcoro/log"
mkdir -p "$LOG" "$B/bin"

CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)")")")}"
NVCC="${CUDA_HOME}/bin/nvcc"
test -x "${NVCC}" || { echo "no nvcc at ${NVCC} -- set CUDA_HOME" >&2; exit 1; }
COROC_CUDA_HOME="${COROC_CUDA_HOME:-${CUDA_HOME}}"

# sm_XX: whatever the build was configured for, first entry, suffix stripped
# ("89-real" -> 89). sm_80 only if the cache says nothing at all.
GPU_ARCH="${GPU_ARCH:-sm_$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES[^=]*=\([0-9]*\).*/\1/p' \
    "${B}/CMakeCache.txt" 2>/dev/null | head -1)}"
[ "${GPU_ARCH}" = "sm_" ] && GPU_ARCH=sm_80

# THE REGISTER CEILING, and why nvcc needs one stated outright.
#
# Every paged kernel carries GV_LAUNCH_BOUNDS -- __launch_bounds__(256, 4) --
# which on a 65536-register SM asks ptxas for 65536/(256*4) = 64 registers.
# In the CMake build that budget is met by cmake/ClioCoroRegCap.cmake, an LLVM
# pass that stamps nvvm.maxnreg on the coroutine kernels; it is a clang plugin
# and there is no way to run it from nvcc.
#
# Without it, -rdc=true is enough to break the build outright. Separate
# compilation leaves DeviceVector::CoFetch as a real call rather than inlining
# it, the callee is allocated on its own at 190 registers, and ptxas refuses
# the mismatch:
#
#   ptxas error : Entry function '...StepKernel...' with max regcount of 64
#                 calls function '...CoFetch...' with regcount of 190
#
# (Measured: grayscott fails this way on sm_89 while the other five link. The
# difference is which calls survive inlining, not anything about the kernels,
# so a ceiling only grayscott gets would be a coincidence waiting to move.)
#
# -maxrregcount is the TU-wide blunt version of that pass -- the very thing
# ClioCoroRegCap.cmake exists to avoid -- and it is the right tool HERE, where
# the TU is one benchmark whose kernels all want the same budget anyway. The
# number is derived from the same three inputs as the launch bounds and the
# cmake module, so the three cannot drift apart.
NEWCORO_REGS_PER_SM="${NEWCORO_REGS_PER_SM:-65536}"
NEWCORO_LB_THREADS="${NEWCORO_LB_THREADS:-256}"
NEWCORO_LB_BLOCKS="${NEWCORO_LB_BLOCKS:-4}"
NEWCORO_MAXREG="${NEWCORO_MAXREG:-$((NEWCORO_REGS_PER_SM / (NEWCORO_LB_THREADS * NEWCORO_LB_BLOCKS)))}"

echo "### [$NAME] root=$W build=$B arch=$GPU_ARCH maxrreg=$NEWCORO_MAXREG"
echo "### [$NAME] nvcc=$NVCC   coroc parses against $COROC_CUDA_HOME"

# Flags: take a real CTE TU's -I/-D set from this build, drop the ones that
# belong to that TU's own target, add the gmx/gv include set.
FLAGS=$(python3 - "${B}/compile_commands.json" <<'PY'
import json, shlex, sys
d = json.load(open(sys.argv[1]))
pick = None
for e in d:
    if "core_runtime.cc" in e["file"] and "context-transfer-engine" in e["file"]:
        pick = e
        break
if pick is None:
    sys.exit("no CTE core_runtime.cc entry in compile_commands.json")
a = shlex.split(pick["command"] if "command" in pick else " ".join(pick["arguments"]))[1:]
keep, i = [], 0
while i < len(a):
    x = a[i]
    if x == "-isystem" and i + 1 < len(a):
        keep += [x, a[i + 1]]; i += 2; continue
    if x.startswith("-I") or x.startswith("-D"):
        keep.append(x)
    i += 1
keep = [x for x in keep if "CLIO_YIELD_CORO" not in x]
keep = [x for x in keep if not (x.startswith("-I") and "gpu_vector" in x)]
keep.append("-DCLIO_COROC")
print(" ".join(keep))
PY
) || exit 1

EXTRA="-I$W/context-runtime/include -I$W/context-runtime/modules/admin/include
       -I$W/context-runtime/modules/bdev/include
       -I$W/context-transfer-engine/core/include
       -I$W/context-transport-primitives/include
       -I$B/context-transport-primitives/src/include
       -I$W/context-transfer-engine/checkpoint/include
       -I$W/context-transfer-engine/compressor/include"

# THE TOOL AND NVCC DO NOT PARSE THE SAME TOOLKIT. clang hard-errors on a
# CUDA it does not know, so the transpile runs against COROC_CUDA_HOME -- and
# every include path inherited from the build has to move with it, or
# crt/math_functions.hpp arrives from the wrong toolkit and the two disagree
# about what is already declared.
TPFLAGS=$(echo "$FLAGS $EXTRA" \
  | sed -e "s#${CUDA_HOME}#${COROC_CUDA_HOME}#g" \
        -e "s#/usr/local/cuda\(-[0-9.]*\)\?/#${COROC_CUDA_HOME}/#g")

echo "### [$NAME] 1. transpile"
rm -rf "$G"
"${COROC}" "$SRC" --verbose --rewrite-root="$W" --mirror-to="$G" \
  -- -x cuda --cuda-path="${COROC_CUDA_HOME}" --cuda-gpu-arch=sm_80 \
     --cuda-device-only -Wno-unknown-cuda-version -std=c++20 -Wno-everything \
     -ferror-limit=0 \
     -I"$W/$BD" -I"$W/$GVI" $TPFLAGS > "$LOG/tp_$NAME.log" 2>&1
grep -E "clio-coroc" "$LOG/tp_$NAME.log" | head -5
if ! test -f "$G/$BD/clio_${NAME}_paged_newcoro.cc"; then
  echo "TRANSPILE FAILED (see $LOG/tp_$NAME.log)"
  grep -E "error" "$LOG/tp_$NAME.log" | head -10; exit 1
fi

echo "### [$NAME] 2. overlay"
# The transpiler rewrote device_vector.h too; the overlay is the gv include
# tree with that one file replaced, so the benchmark and the header it awaits
# across agree on the lowered form. Both backends consume this tree, which is
# why it is produced before the stage gate below.
rm -rf "$O"; mkdir -p "$O"
cp -r "$W/$GVI" "$O/gv_include"
cp "$G/$GVI/clio_cte/gpu_vector/device_vector.h" \
   "$O/gv_include/clio_cte/gpu_vector/device_vector.h" 2>/dev/null || true

if [ "${NEWCORO_STAGE:-all}" = "transpile" ]; then
  echo "### [$NAME] NEWCORO_STAGE=transpile -- stopping before nvcc"
  echo "  gen=$G"
  echo "  overlay=$O"
  exit 0
fi

echo "### [$NAME] 3. compile with NVCC (the compiler co_await cannot use)"
# -rdc=true is not optional: yield_stack.h declares an inline __device__
# variable with external linkage, which whole-program mode rejects.
"${NVCC}" -x cu "$G/$BD/clio_${NAME}_paged_newcoro.cc" \
  -c -o "$B/newcoro/${NAME}.o" -arch=${GPU_ARCH} -std=c++20 -O2 -rdc=true \
  -maxrregcount=${NEWCORO_MAXREG} \
  -Xcompiler -fPIC --expt-relaxed-constexpr --extended-lambda \
  -I"$W/$BD" -I"$O/gv_include" -I"$O/gv_include/clio_cte/gpu_vector" \
  $EXTRA $FLAGS > "$LOG/nv_$NAME.log" 2>&1
if ! test -f "$B/newcoro/${NAME}.o"; then
  echo "NVCC FAILED (see $LOG/nv_$NAME.log)"
  grep -E "error" "$LOG/nv_$NAME.log" | head -15; exit 1
fi
echo "  object $(stat -c%s "$B/newcoro/${NAME}.o") bytes"

echo "### [$NAME] 4. link"
# RPATH IS $ORIGIN, NOT THE BUILD PATH. The binary sits in the same bin/ as
# the libraries it links, and the distributed harness mounts that tree at a
# DIFFERENT path inside the container (/workspace/<build>/bin). An absolute
# host rpath resolves to nothing there; $ORIGIN resolves to the right place
# on both sides. It is also what the rest of the build system does.
"${NVCC}" -arch=${GPU_ARCH} -rdc=true "$B/newcoro/${NAME}.o" \
  -o "$B/bin/clio_${NAME}_paged_newcoro" \
  -L"$B/bin" -Xlinker -rpath -Xlinker '$ORIGIN' \
  -lclio_cte_core_client -lclio_cte_core_runtime \
  -lclio_admin_client -lclio_bdev_client \
  -lclio_run_cxx -lclio_run_cxx_gpu -lclio_ctp_host -lclio_ctp_cuda \
  -L"${CUDA_HOME}/lib64" -lcudart -lzmq -lyaml-cpp -lpthread -ldl -lrt \
  > "$LOG/ln_$NAME.log" 2>&1
test -x "$B/bin/clio_${NAME}_paged_newcoro" || { echo "LINK FAILED (see $LOG/ln_$NAME.log)";
  grep -E "error|undefined" "$LOG/ln_$NAME.log" | head -10; exit 1; }
echo "  LINKED OK: $B/bin/clio_${NAME}_paged_newcoro"

if [ "${NEWCORO_RUN:-1}" = "0" ]; then
  echo "### [$NAME] NEWCORO_RUN=0 -- stopping before the run"
  exit 0
fi

echo "### [$NAME] 5. run: $*"
RUNDIR="${NEWCORO_RUN_DIR:-$B/newcoro/run_$NAME}"
mkdir -p "$RUNDIR" && cd "$RUNDIR" && rm -f ./*.yaml
timeout "${NEWCORO_TIMEOUT:-1200}" "$B/bin/clio_${NAME}_paged_newcoro" "$@" 2>&1 \
  | grep -viE "INFO|DEBUG" | grep -E "GATE|PASS|FAIL|paging:|faults|ERROR|error" | head -20
rc=${PIPESTATUS[0]}
echo "### [$NAME] exit=$rc"
exit $rc
