#!/usr/bin/env bash
# Build one Aurora baseline edition: a SYCL workload source from
# sycl_baseline/ against one data plane.
#
#   build_baselines_aurora.sh <workload> <mpi|ccl|ishmem>
#
# Output: build-spike/clio_<workload>_<sub>_bench. Login-node build with
# icpx, AOT for PVC (the same -device pvc the newcoro builds use), MPICH's
# wrappers for the MPI half, oneCCL / Intel SHMEM from the oneAPI tree the
# login environment already exports (CCL_ROOT, ISHMEM_ROOT).
#
# Nothing from clio is linked. The math headers each edition includes are
# the CUDA-free ones the paged bench shares (kmeans_math.h and friends).
set -eu
WL=${1:?usage: build_baselines_aurora.sh <workload> <mpi|ccl|ishmem>}
SUB=${2:?usage: build_baselines_aurora.sh <workload> <mpi|ccl|ishmem>}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(cd "$HERE/../../../.." && pwd)"
OUT="$W/build-spike"
mkdir -p "$OUT"
SRC="$HERE/sycl_baseline/clio_${WL}_sycl_bench.cc"
test -f "$SRC" || { echo "no source: $SRC"; exit 2; }
ICPX=${ICPX:-$(command -v icpx || ls /opt/aurora/*/oneapi/compiler/latest/bin/icpx | head -1)}
MPICH_ROOT=${MPICH_ROOT:-$(dirname "$(dirname "$(command -v mpicxx)")")}
CCL_ROOT=${CCL_ROOT:-/opt/aurora/26.26.0/oneapi/ccl/latest}
ISHMEM_ROOT=${ISHMEM_ROOT:-/opt/aurora/26.26.0/oneapi/ishmem/latest}
ZE_LIB=${ZE_LIB:-/usr/lib64}

# -fno-sycl-id-queries-fit-in-int: a 32 GB shard is 8.6 G elements, past
# the 32-bit index range SYCL assumes by default (the kmeans anchor deck
# threw "range does not fit in int" at launch).
# PRECISE FLOATING POINT, as build_newcoro_aurora.sh: icpx defaults to
# -fp-model=fast, and the gates compare against host references computed
# with explicit rounding.
CXX=("$ICPX" -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc"
     -std=c++20 -O2 -fp-model=precise -ffp-contract=off
     -fno-sycl-id-queries-fit-in-int
     -I"$MPICH_ROOT/include" -I"$HERE/sycl_baseline")
LD=(-L"$MPICH_ROOT/lib" -Wl,-rpath,"$MPICH_ROOT/lib" -lmpi)
case "$SUB" in
  mpi)    DEF=(-DGV_COMM_MPI) ;;
  ccl)    DEF=(-DGV_COMM_CCL -I"$CCL_ROOT/include")
          LD+=(-L"$CCL_ROOT/lib" -Wl,-rpath,"$CCL_ROOT/lib" -lccl) ;;
  ishmem) DEF=(-DGV_COMM_ISHMEM -I"$ISHMEM_ROOT/include")
          LD+=("$ISHMEM_ROOT/lib/libishmem.a" -L"$ZE_LIB" -lze_loader -lpthread) ;;
  *) echo "substrate must be mpi, ccl or ishmem"; exit 2 ;;
esac
BIN="$OUT/clio_${WL}_${SUB}_bench"
echo "### [$WL/$SUB] $(date +%H:%M:%S) compile+link -> $BIN"
"${CXX[@]}" "${DEF[@]}" "$SRC" -o "$BIN" "${LD[@]}" > "$OUT/bl_${WL}_${SUB}.log" 2>&1 \
  || { echo "BUILD FAILED; see $OUT/bl_${WL}_${SUB}.log"; grep -m5 "error" "$OUT/bl_${WL}_${SUB}.log"; exit 1; }
echo "  linked: $BIN ($(stat -c %s "$BIN") bytes)"
