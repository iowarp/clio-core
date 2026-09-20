set -u
# Compile a ported benchmark's transpiled source for SYCL (spir64).
#   sycl_bench.sh <dir> <name>
#
# COMPILE ONLY, deliberately. Linking would need the whole CTE/runtime stack
# rebuilt with CTP_ENABLE_SYCL, which is a separate job; what this answers is
# the codegen question -- does the transpiled state machine, and the
# parallel_for that enters it, survive a SYCL device compile. That is the
# same question the design doc's AOT check asks.
DIR=$1; NAME=$2
W=/workspace
B=/home/iowarp/bnv2
G=/home/iowarp/gen_$NAME
O=/home/iowarp/ov_$NAME
GVI=context-transfer-engine/adapter/gpu_vector/include
BD=context-transfer-engine/adapter/gpu_vector/benchmark/$DIR

test -f "$G/$BD/clio_${NAME}_paged_newcoro.cc" || {
  echo "no transpiled source for $NAME -- run bench.sh first"; exit 1; }

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

echo "### [$NAME] SYCL device compile (spir64)"
clang++ -fsycl -std=c++20 -O2 -c "$G/$BD/clio_${NAME}_paged_newcoro.cc" \
  -o /tmp/${NAME}_sycl.o \
  -I$W/$BD -I"$O/gv_include" -I"$O/gv_include/clio_cte/gpu_vector" \
  $EXTRA $DEFS > /tmp/sy_$NAME.log 2>&1
rc=$?
echo "rc=$rc"
if [ $rc -ne 0 ]; then
  grep -E "error" /tmp/sy_$NAME.log | head -20
else
  echo "SYCL OK: $(stat -c%s /tmp/${NAME}_sycl.o) bytes"
fi
