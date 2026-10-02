// Register/size attribution probe for DeviceVector.
//
// One kernel per public verb, selected by -DGVP=<n>. COMPILE-ONLY: these
// kernels are never launched -- they exist so `cuobjdump -res-usage` can price
// each verb separately.
//
// EACH RUNG MUST BE COMPILED IN ITS OWN TRANSLATION UNIT. Compiled together,
// every coroutine kernel in a module inherits the worst one (an indirect
// resume call makes ptxas allocate for the largest address-taken function),
// and all rungs report the same number.
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

#define GVP_CORO_KERNEL(name, coro_expr)                                  \
  __global__ void name(clio::run::IpcManagerGpuInfo info,                 \
                       gv::DeviceVector<float> v, u64 arg, u64 *out,      \
                       gy::YieldableView<> yv, gy::YieldStackView ys) {   \
    CLIO_GPU_INIT(info, nullptr);                                         \
    v.Init(yv.Block());                                                   \
    (void) arg;                                                           \
    (void) out;                                                           \
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());                          \
    __syncthreads();                                                      \
    CLIO_YCORO_RUN(coro_expr);                                            \
  }

#if GVP == 0
// BASELINE: the kernel shell and CLIO_GPU_INIT, no vector call at all.
__global__ void K_Baseline(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> v, u64 arg, u64 *out) {
  CLIO_GPU_INIT(info, nullptr);
  (void) v;
  out[threadIdx.x] = arg;
}

#elif GVP == 1
// The yield driver and an empty coroutine: the coroutine machinery alone.
__device__ gy::YCoroMain C_Empty() { co_return; }
GVP_CORO_KERNEL(K_EmptyCoro, C_Empty())

#elif GVP == 2
__device__ gy::YCoroMain C_Hold(gv::DeviceVector<float> v, u64 off, u64 *out) {
  auto h = co_await v.HoldPage(off, 1);
  out[threadIdx.x] = (u64) h.ptr();
}
GVP_CORO_KERNEL(K_HoldPage, C_Hold(v, arg, out))

#elif GVP == 6
__device__ gy::YCoroMain C_BeginFetch(gv::DeviceVector<float> v, u64 off) {
  co_await v.BeginFetch(0, off, off + 1);
}
GVP_CORO_KERNEL(K_BeginFetch, C_BeginFetch(v, arg))

#elif GVP == 7
__device__ gy::YCoroMain C_AwaitFetch(gv::DeviceVector<float> v) {
  co_await v.AwaitFetch();
}
GVP_CORO_KERNEL(K_AwaitFetch, C_AwaitFetch(v))

#elif GVP == 3
__device__ gy::YCoroMain C_BeginFlush(gv::DeviceVector<float> v, u64 off) {
  co_await v.BeginFlush(0, off, 1);
}
GVP_CORO_KERNEL(K_BeginFlush, C_BeginFlush(v, arg))

#elif GVP == 4
__device__ gy::YCoroMain C_EndFlush(gv::DeviceVector<float> v) {
  co_await v.EndFlush();
}
GVP_CORO_KERNEL(K_EndFlush, C_EndFlush(v))

#elif GVP == 5
// Hold, write, flush, wait -- a whole realistic use of the vector.
__device__ gy::YCoroMain C_All(gv::DeviceVector<float> v, u64 off, u64 *out) {
  co_await v.Fetch(0, off, off + 1);
  {
    auto h = co_await v.HoldPage(off, 1, /*write=*/true);
    if (h.run() != 0) h[0] = 1.0f;
    out[threadIdx.x] = (u64) h.ptr();
  }
  co_await v.Flush(0, off, 1);
}
GVP_CORO_KERNEL(K_All, C_All(v, arg, out))
#endif
