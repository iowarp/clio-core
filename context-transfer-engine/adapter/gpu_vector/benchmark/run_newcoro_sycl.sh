set -u
# Build and RUN a ported benchmark through SYCL, on the NVIDIA backend.
#   sycl_run.sh <dir> <name> [run args...]
#
# The only SYCL device in this container is cuda:gpu (DPC++'s CUDA backend
# on the RTX 5080), so the device target is nvptx64 rather than spir64.
# That is still the SYCL path -- parallel_for, nd_item, the SYCL runtime --
# and it is the one that can actually execute here.
DIR=$1; NAME=$2; shift 2
W=/workspace
B=/home/iowarp/bsy2
G=/home/iowarp/gen_$NAME
O=/home/iowarp/ov_$NAME
BD=context-transfer-engine/adapter/gpu_vector/benchmark/$DIR
SRC=$G/$BD/clio_${NAME}_paged_newcoro.cc
test -f "$SRC" || { echo "no transpiled source -- run bench.sh first"; exit 1; }

EXTRA="-I$W/context-runtime/include -I$W/context-runtime/modules/admin/include
       -I$W/context-runtime/modules/bdev/include
       -I$W/context-transfer-engine/core/include
       -I$W/context-transport-primitives/include
       -I$B/context-transport-primitives/src/include
       -I$W/context-transfer-engine/checkpoint/include
       -I$W/context-transfer-engine/compressor/include"
DEFS="-DCTP_ENABLE_SYCL=1 -DCTP_ENABLE_CUDA=0 -DCLIO_COROC -DCLIO_RUNTIME=1
      -DCLIO_RUN_HAS_POCO=0
      -DCTP_DEFAULT_THREAD_MODEL=ctp::thread::Pthread
      -DCTP_DEFAULT_THREAD_MODEL_GPU=ctp::thread::StdThread
      -DCTP_ENABLE_PTHREADS=1 -DCTP_ENABLE_CEREAL=1 -DCTP_LOG_LEVEL=1
      -DCTP_ENABLE_ZMQ=1 -DCTP_ENABLE_LIGHTBEAM=1 -DYAML_CPP_STATIC_DEFINE
      -DCTP_ENABLE_COMPRESS=0 -DCTP_ENABLE_NVCOMP=0 -DCTP_ENABLE_CUSZ=0
      -DCTP_ENABLE_CUSZP=0 -DCTP_ENABLE_LIBPRESSIO=0 -DCTP_ENABLE_NDZIP=0
      -DCTP_ENABLE_ZFP_SYCL=0 -DCTP_ENABLE_BLOSC2=0"
SYCLT="-fsycl -fsycl-targets=nvptx64-nvidia-cuda
       -Xsycl-target-backend --cuda-gpu-arch=sm_120
       --cuda-path=/usr/local/cuda-13.2 -Wno-unknown-cuda-version"

echo "### [$NAME] SYCL compile+link for nvptx64"
clang++ $SYCLT -std=c++20 -O2 "$SRC" -o /tmp/clio_${NAME}_sycl \
  -I$W/$BD -I"$O/gv_include" -I"$O/gv_include/clio_cte/gpu_vector" \
  $EXTRA $DEFS \
  -L$B/bin -Xlinker -rpath -Xlinker $B/bin \
  -lclio_cte_core_client -lclio_cte_core_runtime \
  -lclio_admin_client -lclio_bdev_client -lclio_run_cxx -lclio_ctp_host \
  -lzmq -lyaml-cpp -lpthread -ldl -lrt > /tmp/syr_$NAME.log 2>&1
test -x /tmp/clio_${NAME}_sycl || { echo "BUILD FAILED";
  grep -E "error|undefined reference" /tmp/syr_$NAME.log | head -15; exit 1; }
echo "built"

echo "### [$NAME] run on SYCL (cuda:gpu)"
mkdir -p /home/iowarp/rundir_sycl && cd /home/iowarp/rundir_sycl && rm -f *.yaml
ONEAPI_DEVICE_SELECTOR=cuda:gpu timeout 1200 /tmp/clio_${NAME}_sycl "$@" \
  > /tmp/syrun_$NAME.log 2>&1
echo "exit=$?"
grep -viE "HANGWATCH|PDF|INFO |WARNING" /tmp/syrun_$NAME.log | tail -12
