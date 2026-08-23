// Register/size attribution probe for DeviceVector.
//
// One kernel per internal function, selected by -DGVP=<n>. COMPILE-ONLY: these
// kernels are never launched and are not correct -- they exist so that
// `cuobjdump -res-usage` and a SASS instruction count can price each piece of
// the page cache separately.
//
// EACH RUNG MUST BE COMPILED IN ITS OWN TRANSLATION UNIT. Compiled together,
// every coroutine kernel in a module inherits the worst one (an indirect
// resume call makes ptxas allocate for the largest address-taken function),
// and all rungs report the same number -- which is how an earlier attempt at
// this measurement concluded, wrongly, that an empty coroutine costs 106
// registers.
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

using A = gv::DeviceVectorTestAccess;

// A sink every rung stores through, so nothing is optimized away.
#define GVP_KERNEL(name, expr)                                          \
  __global__ void name(clio::run::IpcManagerGpuInfo info,               \
                       gv::DeviceVector<float> v, u64 arg, u64 *out) {  \
    CLIO_GPU_INIT(info, nullptr);                                       \
    v.block_override_ = 0;                                              \
    __syncthreads();                                                    \
    out[threadIdx.x] = (u64) (expr);                                    \
  }
#define GVP_KERNEL_V(name, stmt)                                        \
  __global__ void name(clio::run::IpcManagerGpuInfo info,               \
                       gv::DeviceVector<float> v, u64 arg, u64 *out) {  \
    CLIO_GPU_INIT(info, nullptr);                                       \
    v.block_override_ = 0;                                              \
    __syncthreads();                                                    \
    stmt;                                                               \
    out[threadIdx.x] = arg;                                             \
  }

#if GVP == 0
// BASELINE: the kernel shell and CLIO_GPU_INIT, no page-cache call at all.
// Every other rung's cost is (rung - this).
GVP_KERNEL_V(K_Baseline, (void) v)
#elif GVP == 1
GVP_KERNEL(K_IsResident, A::IsResident(v, arg))
#elif GVP == 2
GVP_KERNEL(K_Find, A::Find(v, arg))
#elif GVP == 7
GVP_KERNEL_V(K_ReapFlushed, A::ReapFlushed(v))
#elif GVP == 14
GVP_KERNEL_V(K_FlushBatched, A::FlushRangeLocked(v, arg))

// --- coroutine rungs: need the yield driver -------------------------------
#elif GVP == 20
__device__ gy::YCoroMain C_Empty() { co_return; }
__global__ void K_EmptyCoro(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> v, u64 arg, u64 *out,
                            gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  (void) v; (void) arg; (void) out;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(C_Empty());
}
#elif GVP == 21
__device__ gy::YCoroMain C_Hold(gv::DeviceVector<float> v, u64 off, u64 *out) {
  auto h = co_await v.HoldPage(off, 1);
  out[threadIdx.x] = (u64) h.ptr();
}
__global__ void K_HoldPage(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> v, u64 arg, u64 *out,
                           gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(C_Hold(v, arg, out));
}
#elif GVP == 22
__device__ gy::YCoroMain C_Flush(gv::DeviceVector<float> v) {
  co_await v.EndFlush();
}
__global__ void K_AwaitFlush(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> v, u64 arg, u64 *out,
                             gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = 0; (void) arg; (void) out;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(C_Flush(v));
}
#elif GVP == 23
__device__ gy::YCoroMain C_HoldSet(gv::DeviceVector<float> v) {
  co_await v.EnterHoldSet(4);
}
__global__ void K_EnterHoldSet(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> v, u64 arg, u64 *out,
                               gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = 0; (void) arg; (void) out;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(C_HoldSet(v));
}
#endif
