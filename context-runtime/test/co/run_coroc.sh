#!/usr/bin/env bash
# End-to-end: transpile the demo workload, then prove the generated state
# machine computes exactly what the source computes.
#
# This is the build integration in miniature -- one extra parse per source, the
# rewritten file mirrored at the same relative path, and the consumer's include
# path pointing at the mirror FIRST so that #include resolves to the transpiled
# copy with no source edit anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COROC="${ROOT}/build-coroc/clio-coroc"
GEN="${ROOT}/build-gen"
OUT="${ROOT}/build-spike"
INC="${ROOT}/context-runtime/include"
SRC="${ROOT}/context-runtime/test/co"
CXXFLAGS=(-std=c++20 -O1 -g -Wall -Wextra -pthread)

mkdir -p "${OUT}"
rm -rf "${GEN}"

echo "== transpile =="
"${COROC}" "${SRC}/coro_workload.h" \
  --rewrite-root="${ROOT}/context-runtime" \
  --mirror-to="${GEN}/context-runtime" \
  -- -x c++ -std=c++20 -I"${INC}" -I"${SRC}"

GENH="${GEN}/context-runtime/test/co/coro_workload.h"
test -f "${GENH}" || { echo "transpiler produced no output" >&2; exit 1; }
# The marker must not survive into the output -- excluding the file's own
# doc comment, which mentions it in prose.
if grep -v "^ \*" "${GENH}" | grep -q "CO_AWAIT("; then
  echo "CO_AWAIT survived into output" >&2
  exit 1
fi
echo "   ${GENH}"

echo "== build =="
# Same two source files, two include paths: coro_ref.cc resolves
# coro_workload.h to the ORIGINAL, coro_gen.cc to the TRANSPILED copy.
g++ "${CXXFLAGS[@]}" "${SRC}/coro_ref.cc" -I"${INC}" -I"${SRC}" \
    -o "${OUT}/coro_ref"
g++ "${CXXFLAGS[@]}" "${SRC}/coro_gen.cc" \
    -I"${INC}" -I"${GEN}/context-runtime/test/co" -I"${SRC}" \
    -o "${OUT}/coro_gen"

echo "== run =="
"${OUT}/coro_ref" > "${OUT}/ref.txt"
"${OUT}/coro_gen" > "${OUT}/gen.txt"

fail=0
check() {  # check <name> <candidate-file>
  if diff -q "${OUT}/ref.txt" "$2" > /dev/null; then
    echo "PASS  $1"
  else
    echo "FAIL  $1 -- outputs differ:"
    diff "${OUT}/ref.txt" "$2" | head -10
    fail=1
  fi
}
check "host: transpiled == source ($(wc -l < "${OUT}/ref.txt") values)" "${OUT}/gen.txt"

# ---------------------------------------------------------------------------
# SYCL: the SAME generated header, no port and no variant.
# ---------------------------------------------------------------------------
ICPX="${ICPX:-$(command -v icpx || true)}"
if [[ -n "${ICPX}" ]]; then
  echo "== sycl =="
  "${ICPX}" -fsycl -std=c++20 -O2 -Wall \
      -I"${INC}" -I"${GEN}/context-runtime/test/co" -I"${SRC}" \
      "${SRC}/coro_sycl.cc" -o "${OUT}/coro_sycl"
  # Forced, not inherited: Aurora login nodes preset ONEAPI_DEVICE_SELECTOR to
  # level_zero:gpu, and there is no GPU here. Override with CORO_SYCL_DEVICE.
  ONEAPI_DEVICE_SELECTOR="${CORO_SYCL_DEVICE:-opencl:cpu}" \
      "${OUT}/coro_sycl" > "${OUT}/sycl.txt"
  check "sycl: transpiled == source, on device" "${OUT}/sycl.txt"

  # AOT through IGC for Aurora's GPU. Compile-only: it proves the generated
  # shape survives the backend that matters, without needing a GPU here.
  echo "== sycl AOT (spir64_gen, pvc) =="
  if "${ICPX}" -fsycl -std=c++20 -O2 -fsycl-targets=spir64_gen -Xs "-device pvc" \
        -I"${INC}" -I"${GEN}/context-runtime/test/co" -I"${SRC}" \
        "${SRC}/coro_sycl.cc" -o "${OUT}/coro_sycl_pvc" 2>&1 | tail -2; then
    echo "PASS  sycl: IGC accepts the generated shape for pvc"
  else
    echo "FAIL  sycl: IGC rejected the generated shape"
    fail=1
  fi
else
  echo "SKIP  sycl (no icpx on PATH)"
fi

# ---------------------------------------------------------------------------
# Negative test: the check that does not exist in a hand-decorated world.
# ---------------------------------------------------------------------------
echo "== diagnostics =="
if "${COROC}" "${SRC}/bad_missing_await.h" \
     --rewrite-root="${ROOT}/context-runtime" \
     --mirror-to="${OUT}/badout" \
     -- -x c++ -std=c++20 -I"${INC}" -I"${SRC}" > /dev/null 2>&1; then
  echo "FAIL  R1: an unwrapped call to a suspending function was accepted"
  fail=1
else
  echo "PASS  R1: unwrapped call to a suspending function is an error"
fi

exit "${fail}"
