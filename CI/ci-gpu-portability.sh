#!/usr/bin/env bash
# GPU-portability gates that need no GPU. Runs inside iowarp/deps-cpu (CI) or
# the devcontainer (locally); the same script, so a local run is the CI run.
#
#   CI/ci-gpu-portability.sh sycl-compile
#       Install Intel oneAPI DPC++ (if missing), configure a SYCL build for
#       spir64 (Intel GPUs), and build gpu_vector_all -- every gpu_vector
#       target the SYCL configure creates, including the macro-form editions
#       of the paged benchmarks.
#
#   CI/ci-gpu-portability.sh cpu-vector-matrix
#       Build with the compressor on and run the cpu_vector matrix: every CPU
#       codec the build found x {one RAM tier, RAM tier spilling to a file
#       tier}, out of core and resident, byte-exact. This is the CTE storage
#       path gpu_vector uses (compressor module, tier placement, generations,
#       pool layout), exercised from the host.
#
# Environment:
#   SRC         source tree        (default: the repo containing this script)
#   BUILD_ROOT  build directories  (default: $SRC/build-gpu-portability)
#   ONEAPI_PKG  oneAPI compiler package (default: 2025.3, Aurora's version)
set -euo pipefail

MODE="${1:?usage: $0 sycl-compile|cpu-vector-matrix}"
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_ROOT="${BUILD_ROOT:-${SRC}/build-gpu-portability}"
ONEAPI_PKG="${ONEAPI_PKG:-intel-oneapi-compiler-dpcpp-cpp-2025.3}"
JOBS="${JOBS:-$(nproc)}"
# Ninja when the image has it, Makefiles otherwise.
if command -v ninja >/dev/null 2>&1; then GEN=Ninja; else GEN="Unix Makefiles"; fi

# Loopback only: no runtime started here ever listens on 0.0.0.0.
export CLIO_BIND_ADDR=127.0.0.1 CLIO_SERVER_ADDR=127.0.0.1

as_root() { if [[ $(id -u) -eq 0 ]]; then "$@"; else sudo "$@"; fi; }

install_oneapi() {
  if [[ -x /opt/intel/oneapi/compiler/latest/bin/icpx ]]; then
    return
  fi
  echo "### installing ${ONEAPI_PKG}"
  as_root apt-get update -qq
  as_root apt-get install -y -qq gpg wget ca-certificates
  wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | as_root tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | as_root tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
  as_root apt-get update -qq
  as_root apt-get install -y -qq "${ONEAPI_PKG}"
}

sycl_compile() {
  install_oneapi
  # setvars.sh reads unset variables; it must run before `set -u` applies.
  set +u
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  set -u
  local b="${BUILD_ROOT}/sycl"
  local cfg=(-S "${SRC}" -B "${b}" -G "${GEN}" -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
    -DCLIO_CORE_ENABLE_SYCL=ON -DSYCL_TARGET=spir64
    -DCLIO_CORE_ENABLE_TESTS=ON -DCLIO_CORE_ENABLE_BENCHMARKS=ON
    -DCLIO_CTE_ENABLE_COMPRESS=ON -DCLIO_CORE_ENABLE_IO_URING=OFF
    -DCLIO_CORE_ENABLE_CAE=OFF -DCLIO_CORE_ENABLE_CEE=OFF
    -DCMAKE_DISABLE_FIND_PACKAGE_Poco=ON)
  # Configured twice: CLIO_SYCL_INCLUDE_DIR is only in the cache after the
  # first pass, and a single pass builds without the SYCL include path.
  cmake "${cfg[@]}"
  cmake "${cfg[@]}" | grep -E "gpu_vector_all|paged bench \(SYCL" || true
  cmake --build "${b}" -j "${JOBS}" --target gpu_vector_all
}

cpu_vector_matrix() {
  local b="${BUILD_ROOT}/cpu"
  cmake -S "${SRC}" -B "${b}" -G "${GEN}" -DCMAKE_BUILD_TYPE=Release \
    -DCLIO_CORE_ENABLE_TESTS=ON -DCLIO_CTE_ENABLE_COMPRESS=ON \
    -DCLIO_CORE_ENABLE_CAE=OFF -DCLIO_CORE_ENABLE_CEE=OFF \
    -DCLIO_CORE_ENABLE_IO_URING=OFF
  cmake --build "${b}" -j "${JOBS}" --target \
    test_cpu_vector test_cpu_vector_matrix clio_kmeans_cpu_bench \
    clio_kmeans_cpu_ref clio_cte_compressor_runtime clio_cte_compressor_client
  # Every entry that exists is expected to pass: codecs the build did not find
  # are not registered (see adapter/cpu_vector/test/CMakeLists.txt).
  ctest --test-dir "${b}" -L cpu_vector --output-on-failure --timeout 600
}

case "${MODE}" in
  sycl-compile) sycl_compile ;;
  cpu-vector-matrix) cpu_vector_matrix ;;
  *) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
