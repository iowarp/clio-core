#!/usr/bin/env bash
# Transpile a ported paged benchmark and DEVICE-COMPILE it for SYCL on Aurora.
#
#   build_newcoro_aurora.sh <dir> <name>
#
# COMPILE ONLY, deliberately -- the same question build_newcoro_sycl.sh asks in
# the container. Linking would need the whole CTE/runtime stack rebuilt with
# CTP_ENABLE_SYCL, which is a separate job; what this answers is the codegen
# question: does the transpiled state machine, and the parallel_for that enters
# it, survive a real SYCL device compile.
#
# Differs from build_newcoro_sycl.sh only in where things live. That script is
# written for the devcontainer (/workspace, /home/iowarp/bnv2) and pulls its
# flags out of an existing build's compile_commands.json; there is no such build
# here, so the include set is spelled out.
set -u

DIR=${1:?usage: build_newcoro_aurora.sh <dir> <name>}
NAME=${2:?usage: build_newcoro_aurora.sh <dir> <name>}

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
GVI=context-transfer-engine/adapter/gpu_vector/include
BD=context-transfer-engine/adapter/gpu_vector/benchmark/$DIR
SRC="$W/$BD/clio_${NAME}_paged_newcoro.cc"
G="${BUILD_GEN:-$W}/build-gen-$NAME"
OUT="${BUILD_OUT:-$W/build-spike}"
COROC="$W/build-coroc/clio-coroc"

test -f "$SRC" || { echo "no source: $SRC" >&2; exit 1; }
test -x "$COROC" || { echo "build the transpiler first: tools/coroc/build.sh" >&2; exit 1; }
mkdir -p "$OUT"

INCS=(-I"$W/$BD"
      -I"$W/$GVI"
      # device_vector.h's siblings (page.h and friends) are reached by QUOTED
      # include, which resolved by same-directory lookup until the generated
      # copy moved to the mirror. The original's directory has to be on the
      # path so those siblings still resolve.
      -I"$W/$GVI/clio_cte/gpu_vector"
      -I"$W/context-runtime/include"
      -I"$W/context-runtime/modules/admin/include"
      -I"$W/context-runtime/modules/bdev/include"
      -I"$W/context-transfer-engine/core/include"
      -I"$W/context-transfer-engine/checkpoint/include"
      -I"$W/context-transfer-engine/compressor/include"
      -I"$W/context-transport-primitives/include")
# Third-party headers. In the devcontainer these come out of the build tree's
# _deps; on Aurora they are spack prefixes, so discover them rather than
# hardcoding a hash-suffixed path that will rot at the next module update.
SPACK_ROOT=${SPACK_ROOT:-$(ls -d /opt/aurora/*/spack/unified/*/install/linux-x86_64 2>/dev/null | head -1)}
# Aurora's unified spack has boost but not yaml-cpp/cereal/libzmq; those are
# in the user's own spack (the one build-fresh was configured against), so
# search both. Without them the transpile dies on 'yaml-cpp/yaml.h'.
for pkg in yaml-cpp cereal boost libzmq zeromq; do
  d=$(ls -d "$SPACK_ROOT"/${pkg}-*/include "$HOME"/spack/opt/spack/linux-*/${pkg}-*/include 2>/dev/null | head -1)
  [ -n "$d" ] && INCS+=(-I"$d")
done
# The SYCL headers, for the PARSE only. macros.h pulls in <sycl/sycl.hpp>
# whenever CTP_ENABLE_SYCL=1, and the tool's clang is a stock LLVM that has
# never heard of oneAPI. Without this the parse dies on a fatal include error
# and hands the tool a TRUNCATED AST -- which it used to transpile anyway.
ONEAPI=${ONEAPI:-$(ls -d /opt/aurora/*/oneapi/compiler/latest 2>/dev/null | head -1)}
[ -n "${ONEAPI}" ] && INCS+=(-I"$ONEAPI/include")
for extra in ${CLIO_EXTRA_INCLUDES:-}; do INCS+=(-I"$extra"); done

DEFS=(-DCTP_ENABLE_SYCL=1 -DCTP_ENABLE_CUDA=0 -DCLIO_COROC -DCLIO_RUNTIME=1
      -DCTP_DEFAULT_THREAD_MODEL=ctp::thread::Pthread
      -DCTP_DEFAULT_THREAD_MODEL_GPU=ctp::thread::StdThread
      -DCTP_ENABLE_PTHREADS=1 -DCTP_ENABLE_CEREAL=1 -DCTP_LOG_LEVEL=1
      -DCTP_ENABLE_ZMQ=1 -DCTP_ENABLE_LIGHTBEAM=1 -DYAML_CPP_STATIC_DEFINE
      -DCTP_ENABLE_COMPRESS=0 -DCTP_ENABLE_NVCOMP=0 -DCTP_ENABLE_CUSZ=0
      -DCTP_ENABLE_CUSZP=0 -DCTP_ENABLE_LIBPRESSIO=0 -DCTP_ENABLE_NDZIP=0
      -DCTP_ENABLE_ZFP_SYCL=0 -DCTP_ENABLE_BLOSC2=0)

# The parse must take the SAME preprocessor branches the SYCL build takes, or
# the tool reads a different program than the compiler will. CTP_IS_SYCL_COMPILER
# is gated on SYCL_LANGUAGE_VERSION, which only -fsycl defines -- and the tool's
# clang is a stock LLVM, so it is defined by hand here. Without it the SYCL
# branches of YieldableView and the IPC manager simply are not there.
TPDEFS=("${DEFS[@]}" -DSYCL_LANGUAGE_VERSION=202001)

echo "### [$NAME] 1. transpile"
rm -rf "$G"
# Parsed as ordinary C++ with the SYCL flags, NOT as CUDA: the suspending code
# is backend-neutral, and the tool only has to answer questions about it.
"$COROC" "$SRC" --verbose --rewrite-root="$W" --mirror-to="$G" \
  -- -x c++ -std=c++20 -Wno-everything -ferror-limit=0 \
     "${INCS[@]}" "${TPDEFS[@]}" > "$OUT/tp_$NAME.log" 2>&1
rc=$?
grep -E "^clio-coroc" "$OUT/tp_$NAME.log" | head -5
if grep -q "fatal error\|refusing to write" "$OUT/tp_$NAME.log"; then
  echo "TRANSPILE PARSE FAILED -- output would be silently wrong:"
  grep -E "fatal error|refusing to write" "$OUT/tp_$NAME.log" | head -5
  exit 1
fi
if ! test -f "$G/$BD/clio_${NAME}_paged_newcoro.cc"; then
  echo "TRANSPILE FAILED (rc=$rc), first errors:"
  grep -E "error" "$OUT/tp_$NAME.log" | head -15
  exit 1
fi

# The generated device_vector.h must shadow the original, so the overlay goes
# FIRST on the include path -- the mirror-dir trick, exactly as designed.
# AOT=1 compiles the device code through IGC for PVC HERE, on the login node.
# It reproduces a device-side JIT compiler crash for free, in minutes, instead
# of in a queued job -- which is how lammps_md's IGC segfault gets iterated on
# without spending allocation. The object and executable get an _aot suffix so
# they never shadow the JIT build.
AOTFLAGS=()
SUFFIX=""
if [ "${AOT:-0}" = "1" ]; then
  AOTFLAGS=(-fsycl-targets=spir64_gen -Xs "-device pvc")
  # GRF=large compiles for PVC's 256-register mode. The coroutine kernels
  # spill >10 KB per thread at the default 128 (IGC dump: LaunchAssignTiled
  # 11328 B, SIMD16), and every access to spilled state goes to scratch.
  [ "${GRF:-}" = large ] &&
    AOTFLAGS=(-fsycl-targets=spir64_gen -Xs "-device pvc -options -ze-opt-large-register-file")
  # NOL2G=1 promises IGC that no generic pointer ever holds a LOCAL address,
  # so every generic access resolves to global at compile time. The paging
  # path reads everything through h_ (a generic pointer to the device header),
  # and each of those loads was a runtime local-window check plus a divergent
  # branch. Only valid for a kernel that never routes SLM through a generic
  # pointer -- grayscott; NOT the kmeans local tile.
  # GRF=small forces 128 registers: with subroutine calls (IGC_FC=2) IGC's
  # auto-selection picked 256, which halved occupancy at high block counts.
  [ "${GRF:-}" = small ] &&
    AOTFLAGS=(-fsycl-targets=spir64_gen -Xs "-device pvc -options -ze-intel-128-GRF-per-thread")
  [ "${NOL2G:-}" = 1 ] &&
    AOTFLAGS=(-fsycl-targets=spir64_gen -Xs "-device pvc -options -cl-intel-no-local-to-generic")
  SUFFIX="_aot"
fi
[ "${SPLIT:-off}" != "off" ] && SUFFIX="${SUFFIX}_split"
[ -n "${EXTRA_CXX:-}" ] && SUFFIX="${SUFFIX}_x"
# IGC_FC overrides the stack-call setting below (default 3). Only lammps_md
# needs 3; a build with another value gets its own suffix so it never
# replaces the default binary.
[ -n "${IGC_FC:-}" ] && SUFFIX="${SUFFIX}_fc${IGC_FC}"
[ "${GRF:-}" = large ] && SUFFIX="${SUFFIX}_grf"
[ "${GRF:-}" = small ] && SUFFIX="${SUFFIX}_sgrf"
[ "${NOL2G:-}" = 1 ] && SUFFIX="${SUFFIX}_nol2g"
# TAG names a build variant without changing how it compiles.
[ -n "${TAG:-}" ] && SUFFIX="${SUFFIX}_${TAG}"

echo "### [$NAME] 2. SYCL device compile (${SUFFIX:+AOT pvc}${SUFFIX:-spir64 JIT})"
ICPX=${ICPX:-$(command -v icpx 2>/dev/null || ls /opt/aurora/*/oneapi/compiler/latest/bin/icpx 2>/dev/null | head -1)}
test -n "$ICPX" || { echo "no icpx on PATH" >&2; exit 1; }

DGDEF=()
[ "${SPLIT:-off}" != "off" ] && DGDEF=(-DCLIO_SYCL_DG_USM)
# EXTRA_CXX: experiment hook, e.g. -fno-inline to test whether clang-level
# inlining is what produces the single 6000-block function IGC cannot take.
# PRECISE FLOATING POINT. icpx defaults to -fp-model=fast on both passes,
# which nvcc and clang do not: explicit __fmaf_rn/sycl::fma need not stay
# fused-and-correctly-rounded, and operations may be reassociated. The
# benchmarks gate device results bitwise against host references computed
# in the same TU with explicit fmaf, and lammps_md's ballistic gate failed
# at 3-4 ulp (x=687, v=11694 mismatches, identical with -ffp-contract=off
# alone) while its double closed form and thermo agreed to 3e-14.
"$ICPX" -fsycl "${AOTFLAGS[@]}" "${DGDEF[@]}" ${EXTRA_CXX:-} -std=c++20 -O2 -fp-model=precise -ffp-contract=off -c "$G/$BD/clio_${NAME}_paged_newcoro.cc" \
  -o "$OUT/${NAME}_sycl${SUFFIX}.o" \
  -I"$G/$GVI" -I"$G/$GVI/clio_cte/gpu_vector" \
  "${INCS[@]}" "${DEFS[@]}" > "$OUT/sy_$NAME.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "  rc=$rc"
  grep -E "error" "$OUT/sy_$NAME.log" | head -20
  exit 1
fi
echo "  SYCL OK: $(stat -c%s "$OUT/${NAME}_sycl${SUFFIX}.o") bytes"

# --------------------------------------------------------------------------
# 3. link, against the SYCL build of the stack that already exists.
#
# Skipped unless CLIO_SYCL_BUILD points at one. The library-side delta on this
# branch is header-only (co/coro.h, co/driver.h, co/yield_backend.h), so a
# stack built before them still links.
# --------------------------------------------------------------------------
# THIS TREE'S BUILD BY DEFAULT, never the main checkout's. The runpath is baked
# in at link time, and one rebuild without the override linked every
# benchmark against /home/llogan/clio-core/build/sycl -- a runtime from before
# the device-ring port. Its kernels took the legacy host-memory queue and
# died on its first atomic (AtomicAccessViolation at the 16 MB queue base),
# which read exactly like the pre-ring failure and cost a night of bisecting.
BUILD=${CLIO_SYCL_BUILD:-$W/build-fresh}
if [ ! -d "$BUILD/bin" ]; then
  echo "  (no CLIO_SYCL_BUILD -- compile only)"
  exit 0
fi

# ONE DEVICE IMAGE. yield_backend.h keeps its smem base in a device_global with
# device_image_scope, and DPC++ splits a TU with many kernels into several
# images -- which that property forbids. lammps_md has enough kernels to trip
# it; the others do not.
# SPLIT=per_kernel is the other half of the lammps_md fix: with the
# device_global on the USM form (CLIO_SYCL_DG_USM, above) the TU can be split
# per kernel, and IGC gets ~32 small images instead of one it cannot compile.
SPLIT=${SPLIT:-off}
SYCLT=(-fsycl -fsycl-device-code-split="${SPLIT}")

LIBDIRS=(-L"$BUILD/bin")
for pkg in yaml-cpp libzmq zeromq; do
  d=$(ls -d "$SPACK_ROOT"/${pkg}-*/lib64 "$SPACK_ROOT"/${pkg}-*/lib 2>/dev/null | head -1)
  [ -n "$d" ] && LIBDIRS+=(-L"$d" -Wl,-rpath,"$d")
done

# IGC MUST NOT INLINE THE COROUTINES. Measured on lammps_md: IGC takes the
# 27 functions of a coroutine kernel (11.8k lines) and inlines them into one
# 6,475-block function (55k lines), then segfaults in its backend on 13 of the
# 20 per-kernel images. IGC_FunctionControl=3 (stack calls) keeps them as
# calls and every image compiles. Set for the link, where ocloc runs; the JIT
# path needs the same variable in the job environment (pbs_newcoro_aurora.sh).
export IGC_FunctionControl=${IGC_FC:-3}

echo "### [$NAME] 3. link"
"$ICPX" "${SYCLT[@]}" "${AOTFLAGS[@]}" -std=c++20 -O2 \
  "$OUT/${NAME}_sycl${SUFFIX}.o" -o "$OUT/clio_${NAME}_paged_newcoro${SUFFIX}" \
  "${LIBDIRS[@]}" -Wl,-rpath,"$BUILD/bin" \
  -lclio_cte_core_client -lclio_cte_core_runtime \
  -lclio_admin_client -lclio_bdev_client -lclio_run_cxx -lclio_ctp_host \
  -lzmq -lyaml-cpp -lpthread -ldl -lrt > "$OUT/ln_$NAME.log" 2>&1
if [ ! -x "$OUT/clio_${NAME}_paged_newcoro${SUFFIX}" ]; then
  echo "  LINK FAILED"
  grep -E "error|undefined reference" "$OUT/ln_$NAME.log" | head -12
  exit 1
fi
echo "  linked: $OUT/clio_${NAME}_paged_newcoro${SUFFIX}"
# PROVENANCE. One line per build in $OUT/MANIFEST.tsv: when, binary, commit
# (+dirty when the tree had uncommitted changes), and the knobs that shaped
# it. Binaries used to be identified by suffix alone, and their build flags
# had to be recovered from old transcripts.
{
  printf '%s\t%s\t%s%s\tEXTRA_CXX=%s\tIGC_FC=%s\tTAG=%s\tGRF=%s\tSPLIT=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "clio_${NAME}_paged_newcoro${SUFFIX}" \
    "$(git -C "$W" rev-parse --short HEAD 2>/dev/null || echo nogit)" \
    "$(git -C "$W" diff --quiet HEAD -- 2>/dev/null || echo +dirty)" \
    "${EXTRA_CXX:-}" "${IGC_FC:-}" "${TAG:-}" "${GRF:-}" "${SPLIT:-off}"
} >> "$OUT/MANIFEST.tsv"
