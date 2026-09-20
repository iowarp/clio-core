set -u
W=/workspace
B=/home/iowarp/bnv2

cat > /tmp/sycl_probe.cc <<'EOF'
// Is the PRODUCTION coroc backend SYCL-clean?
//
// yield_backend.h is written in CUDA spellings (threadIdx, __syncthreads_or,
// __threadfence_system, __trap). yield_stack.h includes sycl_cuda_compat.h,
// which defines every one of them for SYCL -- so the question is whether the
// backend compiles for a SYCL device unchanged, or whether it needs an arm of
// its own. This TU answers exactly that and nothing else: it instantiates
// Ctx and Frame inside a SYCL kernel and exercises the five operations the
// generated code actually emits.
#include <sycl/sycl.hpp>
#include <clio_runtime/co/yield_backend.h>

namespace co = clio::co;

void Probe(sycl::queue &q, clio::run::gpu::YieldStackView sv,
           clio::run::gpu::YieldBlockState *bs, unsigned *out) {
  q.parallel_for(sycl::nd_range<1>{sycl::range<1>(64), sycl::range<1>(64)},
                 [=](sycl::nd_item<1> it) {
                   clio::run::gpu::YieldTlsPublish(sv, bs, 0);
                   clio::run::gpu::YieldLane()->cur_depth_ = 0;
                   co::Ctx cy;
                   co::Frame f(cy, 32);
                   const unsigned r = f.Resume();
                   if (cy.Any(r == 0)) {
                     unsigned long long a = 7;
                     f.Push(1, a);
                     cy.SetWaitTag(3ull);
                   }
                   if (f.Replaying()) {
                     unsigned long long a = 0;
                     f.Pop(a);
                     out[0] = static_cast<unsigned>(a);
                   }
                   f.Done();
                   out[1] = cy.Parked() ? 1u : 0u;
                 });
}
EOF

FLAGS="-I$W/context-runtime/include -I$W/context-transport-primitives/include
       -I$B/context-transport-primitives/src/include
       -I$W/context-runtime/modules/admin/include
       -I$W/context-runtime/modules/bdev/include"
DEFS="-DCTP_ENABLE_SYCL=1 -DCTP_ENABLE_CUDA=0 -DCLIO_COROC
      -DCTP_DEFAULT_THREAD_MODEL=ctp::thread::Pthread
      -DCTP_ENABLE_PTHREADS=1 -DCTP_LOG_LEVEL=1
      -DCTP_DEFAULT_THREAD_MODEL_GPU=ctp::thread::StdThread"

echo "### compiling the coroc backend for SYCL (spir64)"
clang++ -fsycl -std=c++20 -O2 -c /tmp/sycl_probe.cc -o /tmp/sycl_probe.o \
  $DEFS $FLAGS > /tmp/syclprobe.log 2>&1
rc=$?
echo "rc=$rc"
if [ $rc -ne 0 ]; then
  grep -E "error" /tmp/syclprobe.log | head -20
else
  echo "SYCL BACKEND COMPILES: $(stat -c%s /tmp/sycl_probe.o) bytes"
fi
