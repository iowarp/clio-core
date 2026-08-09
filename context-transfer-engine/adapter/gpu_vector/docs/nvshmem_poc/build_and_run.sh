#!/bin/bash
# NVSHMEM PoC build + run. Validated on an RTX 5060 (sm_120) via sm_90 PTX-JIT,
# CUDA 12.6, Open MPI 4.1.6, NVSHMEM 3.7.2 (pip: nvidia-nvshmem-cu12).
#
# On ONE GPU: only `-np 1` works (NVSHMEM requires 1 PE per GPU; 2 PEs on the same
# physical GPU fail at cudaIpcOpenMemHandle, and MPS is unavailable under
# Docker-on-Windows). On DELTA (N A100s): `-np N` runs the real N-PE cluster with
# no code changes.
set -e
NVSH=$(python3 -c "import nvidia.nvshmem,os;print(os.path.dirname(nvidia.nvshmem.__file__))")
export PATH=/usr/local/cuda-12.6/bin:$PATH
ln -sf "$NVSH/lib/libnvshmem_host.so.3" "$NVSH/lib/libnvshmem_host.so" 2>/dev/null || true

echo "=== compile (sm_90 PTX -> JIT to sm_120; use -arch=sm_80 native on Delta A100) ==="
nvcc -rdc=true -arch=sm_90 -ccbin g++ \
  -I"$NVSH/include" $(mpicc --showme:compile) \
  poc_nvshmem.cu -o poc_nvshmem \
  -L"$NVSH/lib" -lnvshmem_host -lnvshmem_device \
  $(mpicc --showme:link) -lcuda

export LD_LIBRARY_PATH="$NVSH/lib":/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/lib64/stubs:$LD_LIBRARY_PATH
export NVSHMEM_REMOTE_TRANSPORT=none   # intra-node only; drop on Delta if using IB

NP=${1:-1}   # PEs = GPUs. 1 here; N on a real N-GPU node/cluster.
echo "=== run $NP PE(s) ==="
mpirun -np "$NP" -x NVSHMEM_REMOTE_TRANSPORT -x LD_LIBRARY_PATH ./poc_nvshmem
