// REGISTER ATTRIBUTION LADDER. Each rung adds exactly ONE ingredient of a real
// paged kernel; `cuobjdump -res-usage` then prices that ingredient. Same
// compiler, same flags, same headers as the real build. Compile-only.
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
static constexpr u64 kStride = 4;



// ONE RUNG PER COMPILE (-DRUNG=n): each rung must be measured in its own
// translation unit, or it inherits the module's worst coroutine -- the very
// effect under study. Measuring them together reported 106 for ALL rungs.

#if RUNG == 0
// R0: raw pointer loop. No clio, no coroutine. The floor.
__global__ void R0_Raw(float *p, u64 n) {
  for (u64 s = threadIdx.x; s < n; s += blockDim.x) p[s * kStride + 3] = -1.0f;
}
#endif

#if RUNG == 1
// R1: + runtime init and yield TLS publish, still NO coroutine.
__global__ void R1_Init(clio::run::IpcManagerGpuInfo info, float *p, u64 n,
                        gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  for (u64 s = threadIdx.x; s < n; s += blockDim.x) p[s * kStride + 3] = -1.0f;
}
#endif

#if RUNG == 2
// R2: + an EMPTY coroutine (co_return only). Prices the lowering itself.
__device__ gy::YCoroMain EmptyCoro() { co_return; }
__global__ void R2_EmptyCoro(clio::run::IpcManagerGpuInfo info,
                             gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(EmptyCoro());
}
#endif

#if RUNG == 3
// R3: + a coroutine doing the R0 loop on a RAW pointer (no HoldPage).
//     Isolates "being a coroutine" from "paging".
__device__ gy::YCoroMain RawLoopCoro(float *p, u64 n) {
  for (u64 s = threadIdx.x; s < n; s += blockDim.x) p[s * kStride + 3] = -1.0f;
  co_return;
}
__global__ void R3_RawLoopCoro(clio::run::IpcManagerGpuInfo info, float *p,
                               u64 n, gy::YieldableView<> yv,
                               gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RawLoopCoro(p, n));
}
#endif

#if RUNG == 4
// R4: + ONE HoldPage. This is the paging machinery entering the kernel.
__device__ gy::YCoroMain OneHoldCoro(gv::DeviceVector<float> dst) {
  const u64 epp = dst.h_->elems_per_page_;
  auto h = co_await dst.HoldPage(0, epp, /*write=*/true);
  float *const p = h.ptr();
  const u64 nslots = epp / kStride;
  for (u64 s = threadIdx.x; s < nslots; s += blockDim.x)
    p[s * kStride + 3] = -1.0f;
  __syncthreads();
}
__global__ void R4_OneHold(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<float> dst,
                           gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  dst.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(OneHoldCoro(dst));
}
#endif

#if RUNG == 5
// R5: the real SentinelCoro -- HoldPage in a LOOP. What the MD bench ships.
__device__ gy::YCoroMain SentinelLikeCoro(gv::DeviceVector<float> dst,
                                          u32 nblocks, u32 block) {
  const u64 epp = dst.h_->elems_per_page_;
  const u64 npages = (dst.h_->size_ + epp - 1) / epp;
  for (u64 pg = block; pg < npages; pg += nblocks) {
    auto h = co_await dst.HoldPage(pg * epp, epp, /*write=*/true);
    float *const p = h.ptr();
    const u64 nslots = epp / kStride;
    for (u64 s = threadIdx.x; s < nslots; s += blockDim.x)
      p[s * kStride + 3] = -1.0f;
    __syncthreads();
  }
}
__global__ void R5_Sentinel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> dst, u32 nblocks,
                            gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  dst.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SentinelLikeCoro(dst, nblocks, yv.Block()));
}
#endif

#if RUNG == 6
// R6: the FAST PATH ONLY. TryHoldFast is a plain device function (probe +
// lock-free table scan, no fetch, no eviction, no flush). If the slow path is
// what costs the registers, this rung is cheap and the fix is architectural
// rather than compiler-side.
__device__ gy::YCoroMain FastOnlyCoro(gv::DeviceVector<float> dst) {
  const u64 epp = dst.h_->elems_per_page_;
  volatile u64 run = dst.TryHoldFast(0, epp, /*write=*/true);
  // Consume `run` without touching the page: this rung prices the PROBE,
  // not the access.
  if (run != 0 && threadIdx.x == 0) __threadfence_block();  // consume run
  co_return;
}
__global__ void R6_FastOnly(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> dst,
                            gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  dst.block_override_ = 0;
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(FastOnlyCoro(dst));
}
#endif
