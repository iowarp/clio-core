#!/bin/bash
# install_gpu_toolchain.sh - GPU compiler/runtime for a GPU wheel variant.
#
# Runs inside the manylinux_2_28 image (AlmaLinux 8) from CIBW_BEFORE_ALL,
# before build_deps_manylinux.sh. Variant comes from $1 or CLIO_WHEEL_GPU.
# Versions match what the rest of the tree is tested with: CUDA 12.9 (13 breaks
# ROCm 6.4's HIP headers and drops nothing we need), ROCm 6.4, oneAPI 2025.
set -euo pipefail

VARIANT="${1:-${CLIO_WHEEL_GPU:-}}"
case "$VARIANT" in
  cuda12)
    dnf config-manager --add-repo \
      https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    # Compiler + headers/libs to link against. The driver (libcuda.so.1) is
    # not installed: the wheel uses the user's driver.
    dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 cuda-driver-devel-12-9 \
      cuda-nvrtc-devel-12-9 cuda-cccl-12-9
    /usr/local/cuda-12.9/bin/nvcc --version | tail -1
    # libclio_ctp_gpu_linker links -lcuda (the driver), which a build image
    # has no driver for; link against CUDA's stub (LIBRARY_PATH in
    # build-pip.yml). The wheel then needs the user's real libcuda.so.1. The
    # .so.1 link lets the GPU-less import test load the wheel; never shipped.
    ln -sf libcuda.so /usr/local/cuda-12.9/lib64/stubs/libcuda.so.1
    ;;
  rocm6)
    cat > /etc/yum.repos.d/rocm.repo <<'REPO'
[rocm]
name=ROCm 6.4
baseurl=https://repo.radeon.com/rocm/rhel8/6.4/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
    # HIP compiler + runtime dev files only (no math libraries).
    dnf install -y rocm-hip-runtime-devel hipcc
    /opt/rocm/bin/hipconfig --version
    ;;
  sycl)
    cat > /etc/yum.repos.d/oneAPI.repo <<'REPO'
[oneAPI]
name=Intel oneAPI
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
REPO
    # 2025.x: libsycl.so.8, matching the intel-sycl-rt>=2025,<2026 dependency.
    dnf install -y intel-oneapi-compiler-dpcpp-cpp-2025.2
    ;;
  *)
    echo "usage: $0 {cuda12|rocm6|sycl} (or set CLIO_WHEEL_GPU)" >&2
    exit 2
    ;;
esac
