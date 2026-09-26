#!/bin/bash
# Build the CTE-free MPI and NCCL baselines against CRAY-MPICH.
#
# WHY THIS EXISTS RATHER THAN CMAKE DOING IT. The workload CMakeLists build
# these with find_package(MPI), which on Delta resolves to the HPC-X OpenMPI
# inside nvhpc -- the MPI that NVSHMEM's bootstrap plugin needs and that
# cannot cross Slingshot ("ucp_ep_create failed: Destination is
# unreachable"). Both MPIs cannot be the answer for one build tree, so the
# multi-node baselines are produced here, into <build>/bin-cray, and the
# in-tree bin/ keeps the HPC-X build that the single-node ctest entries use.
#
#   bin/       HPC-X OpenMPI -- single node, mpirun, the bench_*_2rank tests
#   bin-cray/  cray-mpich    -- MULTI-NODE, srun --mpi=cray_shasta
#
# Folding this into CMake as a second flavour is worth doing; until then the
# rule is that any multi-node MPI or NCCL number comes from bin-cray.
#
# Usage:  ./build_baselines_cray.sh [build-dir]      (default: build-clang)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

BUILD="${1:-build-clang}"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/context-transfer-engine/adapter/gpu_vector/benchmark"
OUT="$ROOT/$BUILD/bin-cray"
MPI="$CLIO_DELTA_CRAY_MPI"
NCCL="$NCCL_HOME"
CUDA="$CLIO_DELTA_CUDA_HOME"

# nvcc 12.6 REFUSES gcc > 13 ("unsupported GNU version! gcc versions later
# than 13 are not supported"), and Delta's default gcc is 14.2.1. env.sh
# already picks gcc-toolset-13 for exactly this; reuse it rather than
# -allow-unsupported-compiler.
CCBIN="$CLIO_DELTA_NVCC_CCBIN"

[ -d "$MPI/include" ] || { echo "no cray-mpich at $MPI"; exit 1; }
mkdir -p "$OUT"
echo "cray-mpich: $MPI"
echo "nccl:       $NCCL"
echo "out:        $OUT"

rc=0
for wl in kmeans weights gmx grayscott lbann lammps_md; do
  for kind in mpi nccl; do
    src="$SRC/$wl/clio_${wl}_${kind}_bench.cc"
    [ -f "$src" ] || { printf '%-10s %-5s SKIP (no source)\n' "$wl" "$kind"; continue; }
    # NOTE the expansion below is ${extra[@]+"${extra[@]}"}, not
    # "${extra[@]:-}". Under `set -u` the latter expands an EMPTY array to a
    # single empty-string argument, which nvcc accepts as a filename and then
    # dies on with "invalid filename" / "Broken module found, compilation
    # aborted!" -- so the mpi half (empty extra) failed while the nccl half
    # (non-empty) built fine.
    extra=()
    [ "$kind" = nccl ] && extra=(-I"$NCCL/include" -L"$NCCL/lib" -lnccl
                                 -Xlinker -rpath -Xlinker "$NCCL/lib")
    printf '%-10s %-5s ' "$wl" "$kind"
    if "$CUDA/bin/nvcc" -ccbin "$CCBIN" -x cu -std=c++17 -O3 -arch=sm_80 \
         -I"$MPI/include" -I"$SRC/$wl" ${extra[@]+"${extra[@]}"} \
         "$src" -o "$OUT/clio_${wl}_${kind}_bench" \
         -L"$MPI/lib" -lmpi_gnu_112 > "/tmp/cray_${wl}_${kind}.log" 2>&1; then
      echo OK
    else
      echo "FAIL (/tmp/cray_${wl}_${kind}.log)"; rc=1
    fi
  done
done

cat <<'NOTE'

Run them with the RIGHT launcher -- cray-mpich under --mpi=pmi2 does a
singleton MPI_Init and every task becomes its own 1-rank job, which passes
every gate while proving nothing:

  srun --mpi=cray_shasta -n 2 --ntasks-per-node=1 bin-cray/clio_<wl>_mpi_bench ...

and for NCCL, with the OFI plugin already exported by env.sh:

  LD_PRELOAD=$CLIO_DELTA_NCCL_PRELOAD \
  srun --mpi=cray_shasta -n 2 --ntasks-per-node=1 bin-cray/clio_<wl>_nccl_bench ...

Always assert the printed rank count. The six benches say it five different
ways: "ranks=2", "2 ranks", "PEs=2", "2 PEs", "2 ranks (model-parallel)".
NOTE
exit $rc
