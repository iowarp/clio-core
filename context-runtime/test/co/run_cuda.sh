#!/usr/bin/env bash
set -u
R=/workspace
OUT=$R/build-spike-dc
mkdir -p $OUT
CXX=clang++
CU="-x cuda --cuda-path=/usr/local/cuda-13.2 --cuda-gpu-arch=sm_120 -Wno-unknown-cuda-version -std=c++20 -O2 -Xcuda-ptxas -v"
INC="-I$R/context-runtime/include -I$R/context-runtime/test/co"
GENINC="-I$R/context-runtime/include -I$R/build-gen/context-runtime/test/co -I$R/context-runtime/test/co"
LIBS="-L/usr/local/cuda-13.2/lib64 -lcudart"

build () { # name, extra-defines, includes, source
  echo "### $1"
  $CXX $CU $2 $3 "$4" $LIBS -o "$OUT/$1" 2>&1 | grep -E "registers|spill|stack frame|error|Error" | sed 's/^/    /'
}

build A_sync   '-DCORO_CUDA_REF=1 -DCORO_CUDA_TAG="\"cuda-sync\""'  "$INC"    $R/context-runtime/test/co/coro_cuda.cu
build B_coroc  '-DCORO_CUDA_REF=0 -DCORO_CUDA_TAG="\"cuda-coroc\""' "$GENINC" $R/context-runtime/test/co/coro_cuda.cu
build C_coroc_lb '-DCORO_CUDA_REF=0 -DCORO_CUDA_TAG="\"cuda-coroc-lb\"" -DCORO_CUDA_LB_THREADS=256 -DCORO_CUDA_LB_BLOCKS=4' "$GENINC" $R/context-runtime/test/co/coro_cuda.cu
build D_c20    ''  "$INC" $R/context-runtime/test/co/coro_cuda_c20.cu
build E_c20_lb '-DCORO_CUDA_LB_THREADS=256 -DCORO_CUDA_LB_BLOCKS=4' "$INC" $R/context-runtime/test/co/coro_cuda_c20.cu
echo "### built:"; ls -1 $OUT
build F_coroc_noinline '-DCORO_CUDA_REF=0 -DCORO_CUDA_TAG="\"cuda-coroc-ni\"" -DCLIO_CO_NO_FORCEINLINE=1' "$GENINC" $R/context-runtime/test/co/coro_cuda.cu

# --- run and check -----------------------------------------------------------
cd $OUT
for e in A_sync B_coroc C_coroc_lb D_c20 E_c20_lb F_coroc_noinline; do
  [ -x "./$e" ] || continue
  echo "=== $e ==="
  timeout 180 ./$e > $e.out 2> $e.err; echo "exit=$?"
  cat $e.err
  if diff -q A_sync.out $e.out >/dev/null 2>&1; then
    echo "PASS  $e == synchronous source ($(wc -l < A_sync.out) values)"
  elif [ "$e" != A_sync ]; then
    echo "FAIL  $e differs"; diff A_sync.out $e.out | head -4
  fi
done
