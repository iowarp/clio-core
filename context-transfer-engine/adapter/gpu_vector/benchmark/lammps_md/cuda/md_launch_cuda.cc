/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * MD launches, CUDA. Compiled by clang-CUDA (the vector's device API is C++20
 * coroutines, which nvcc cannot compile).
 *
 * Nothing here is workload: every coroutine body lives in ../md_kernels.h.
 * These trampolines are what the source carried inline before the seam split.
 */

// Plain includes, not "../": clio_coro_regcap stages every source by BASENAME
// into a flat probe directory, where a relative parent path does not resolve.
// The benchmark's own source dir is on the include path for both builds.
#include "md_launch.h"

#include "md_kernels.h"

#include <clio_ctp/util/gpu_api.h>


/* x and v are created with ONE shared table (vector nblocks = 1): a single
 * copy of every page, held by all GPU blocks. block_override_ = 0 points
 * every block at that table; blocks write DISJOINT pages, so the only
 * sharing is the lock-free probes. */

__global__ MD_LAUNCH_BOUNDS void ReadProbeKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x, u64 passes,
                                u32 nblocks, gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ReadProbeCoro(x, passes, nblocks, yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void IntegrateKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<float> v,
                                gv::DeviceVector<float> third, int use_third,
                                float dt, float gx, float gy_, float gz,
                                int drift, u32 nblocks,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  third.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(IntegrateCoro(x, v, third, use_third, dt, gx, gy_, gz,
                               drift, nblocks, yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void PublishSlabKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x,
    gv::DeviceVector<float> v, u32 nb, u32 cap, u32 z0, u32 z1, u64 gen,
    u32 halo_first, u32 nblocks, u32 tbl_base, gy::YieldableView<> yv,
    gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  // tbl_base separates the ASYNC publish's task tables from the compute
  // kernels' 0..nblocks-1, so the overlap never shares a task slot.
  x.Init(tbl_base + yv.Block());
  v.Init(tbl_base + yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(PublishSlabCoro(x, v, nb, cap, z0, z1, gen, halo_first,
                                 nblocks, yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void HaloPinKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x, u32 nb,
    u32 cap, u32 z0, u32 z1, u64 gen, u32 tbl_base, gy::YieldableView<> yv,
    gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(tbl_base + yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(HaloPinCoro(x, nb, cap, z0, z1, gen, yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void RefaultWriteKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x, u64 pg_lo,
    u64 pg_hi, u64 round, u64 ppp, u32 nblocks, gy::YieldableView<> yv,
    gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RefaultWriteCoro(x, pg_lo, pg_hi, round, ppp, nblocks,
                                  yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void RefaultVerifyKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x, u64 pg_lo,
    u64 pg_hi, u64 round, u64 ppp, u64 below_pg, u64 above_pg, u32 nblocks,
    unsigned long long *d_out, gy::YieldableView<> yv,
    gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RefaultVerifyCoro(x, pg_lo, pg_hi, round, ppp, below_pg,
                                   above_pg, nblocks, yv.Block(), d_out));
}


__global__ MD_LAUNCH_BOUNDS void HaloUnpinKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x, u32 nb,
    u32 cap, u32 z0, u32 z1, u32 tbl_base, gy::YieldableView<> yv,
    gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(tbl_base + yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(HaloUnpinCoro(x, nb, cap, z0, z1, yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void ThermoKernel(clio::run::IpcManagerGpuInfo info,
                             gv::DeviceVector<float> x,
                             gv::DeviceVector<float> v, double *out,
                             u32 nb, u32 cap, u32 z0, u32 z1,
                             u32 nblocks, gy::YieldableView<> yv,
                             gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ThermoCoro(x, v, out, nb, cap, z0, z1, nblocks, yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void ForceKernel(clio::run::IpcManagerGpuInfo info,
                            gv::DeviceVector<float> x,
                            gv::DeviceVector<float> f, u32 nb, u32 cap,
                            float box, float cutoff, int eflag, double *acc,
                            u32 z0, u32 z1, u32 nblocks, u64 hgen,
                            bool force_all, gy::YieldableView<> yv,
                            gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  f.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ForceCoro(x, f, nb, cap, box, cutoff, eflag, acc, z0, z1, nblocks,
                           yv.Block(), hgen, force_all));
}


__global__ MD_LAUNCH_BOUNDS void BuildListKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<int> nl, u32 nb, u32 cap,
                                float box, float rlist, u32 maxneigh,
                                u32 *d_cnt, int *d_err, u32 rowchunk,
                                u32 z0, u32 z1, u32 nblocks, u64 hgen,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  nl.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(BuildListCoro(x, nl, nb, cap, box, rlist, maxneigh, d_cnt,
                               d_err, rowchunk, z0, z1, nblocks,
                               yv.Block(), hgen));
}


__global__ MD_LAUNCH_BOUNDS void ListForceKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<float> x,
                                gv::DeviceVector<float> f,
                                gv::DeviceVector<int> nl, u32 nb, u32 cap,
                                float box, float cutoff, u32 maxneigh,
                                const u32 *d_cnt, int eflag, double *acc,
                                int nocompute, u32 rowchunk, u32 z0, u32 z1, u32 nblocks,
                                u64 hgen, bool force_all, u32 band,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  f.Init(yv.Block());
  nl.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ListForceCoro(x, f, nl, nb, cap, box, cutoff, maxneigh,
                               d_cnt, eflag, acc, nocompute, rowchunk,
                               z0, z1, nblocks, yv.Block(), hgen, force_all,
                               band));
}


__global__ MD_LAUNCH_BOUNDS void RebinWrapKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x, u32 nb,
    u32 cap, float box, u32 z0, u32 z1, u32 nblocks, gy::YieldableView<> yv,
    gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RebinWrapCoro(x, nb, cap, box, z0, z1, nblocks, yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void RebinAssignKernel(
    clio::run::IpcManagerGpuInfo info, gv::DeviceVector<float> x, u32 nb,
    u32 cap, float box, u32 *bincnt, u32 *d_dest, int *d_err, u32 z0, u32 z1,
    u32 nblocks, u64 hgen, gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(RebinAssignCoro(x, nb, cap, box, bincnt, d_dest, d_err, z0,
                                 z1, nblocks, yv.Block(), hgen));
}


__global__ MD_LAUNCH_BOUNDS void GatherKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<float> src,
                              gv::DeviceVector<float> srcx,
                              gv::DeviceVector<float> dst, u32 nb, u32 cap,
                              const u32 *d_dest, int keep_w, u32 z0, u32 z1,
                              u32 nblocks, u64 hgen,
                              gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  src.Init(yv.Block());
  srcx.Init(yv.Block());
  dst.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(GatherCoro(src, srcx, dst, nb, cap, d_dest, keep_w, z0, z1,
                            nblocks, yv.Block(), hgen));
}


__global__ MD_LAUNCH_BOUNDS void SentinelKernel(clio::run::IpcManagerGpuInfo info,
                               gv::DeviceVector<float> dst, u32 nb, u32 cap,
                               u32 z0, u32 z1, u32 nblocks,
                               gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  dst.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(SentinelCoro(dst, nb, cap, z0, z1, nblocks, yv.Block()));
}


__global__ MD_LAUNCH_BOUNDS void MDIntegrateKernel(clio::run::IpcManagerGpuInfo info,
                                  gv::DeviceVector<float> x,
                                  gv::DeviceVector<float> v,
                                  gv::DeviceVector<float> f, float dt,
                                  int drift, u32 nb, u32 cap, u32 z0, u32 z1,
                                  u32 nblocks,
                                  gy::YieldableView<> yv,
                                  gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  x.Init(yv.Block());
  v.Init(yv.Block());
  f.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(MDIntegrateCoro(x, v, f, dt, drift, nb, cap, z0, z1, nblocks,
                                 yv.Block()));
}

namespace clio::gv_bench::md {

void InitBackend(u32 max_blocks, const clio::run::IpcManagerGpuInfo &info) {
  // CUDA's per-block IpcManager is __shared__ storage, born fresh at every
  // launch and initialized by CLIO_GPU_INIT; nothing to preallocate.
  (void)max_blocks;
  (void)info;
}

bool DevMemInfo(size_t *free_bytes, size_t *total_bytes) {
  return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess;
}

const char *LaunchError() {
  // PeekAt, not Get: getting CLEARS the sticky error, and the caller reads the
  // device fatal channel next -- which only says anything while the error
  // stands.
  const cudaError_t e = cudaPeekAtLastError();
  return (e == cudaSuccess) ? nullptr : cudaGetErrorString(e);
}

u32 DeviceClockKHz() {
  int khz = 0;
  cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
  return static_cast<u32>(khz);
}

void SymbolWrite(MdSym sym, const void *src, size_t bytes) {
  switch (sym) {
    case MdSym::kYieldFatal:
      cudaMemcpyToSymbol(gy::g_yield_fatal, src, bytes);
      return;
    case MdSym::kReadBad:
      cudaMemcpyToSymbol(g_read_bad, src, bytes);
      return;
    case MdSym::kReadSample:
      cudaMemcpyToSymbol(g_read_sample, src, bytes);
      return;
    case MdSym::kProbeLdcg:
      cudaMemcpyToSymbol(kProbeLdcg, src, bytes);
      return;
    case MdSym::kReadGeom:
      cudaMemcpyToSymbol(g_read_geom, src, bytes);
      return;
    case MdSym::kGatherWrote:
      cudaMemcpyToSymbol(g_gather_wrote, src, bytes);
      return;
    case MdSym::kPublish:
      cudaMemcpyToSymbol(g_publish, src, bytes);
      return;
    case MdSym::kPubInterior:
      cudaMemcpyToSymbol(g_pub_interior, src, bytes);
      return;
    case MdSym::kMdFlush:
      cudaMemcpyToSymbol(g_md_flush, src, bytes);
      return;
    case MdSym::kXMask:
      cudaMemcpyToSymbol(g_xmask, src, bytes);
      return;
    case MdSym::kNlMask:
      cudaMemcpyToSymbol(g_nlmask, src, bytes);
      return;
    case MdSym::kPubFlushCyc:
      cudaMemcpyToSymbol(g_pub_flush_cyc, src, bytes);
      return;
    case MdSym::kPubFetchCyc:
      cudaMemcpyToSymbol(g_pub_fetch_cyc, src, bytes);
      return;
    case MdSym::kMdCyc:
      cudaMemcpyToSymbol(g_md_cyc, src, bytes);
      return;
    case MdSym::kPagesDone:
      cudaMemcpyToSymbol(g_pages_done, src, bytes);
      return;
    case MdSym::kBlkDone:
      cudaMemcpyToSymbol(g_blk_done, src, bytes);
      return;
    case MdSym::kBlkLast:
      cudaMemcpyToSymbol(g_blk_last, src, bytes);
      return;
  }
}

void SymbolRead(void *dst, MdSym sym, size_t bytes) {
  switch (sym) {
    case MdSym::kYieldFatal:
      cudaMemcpyFromSymbol(dst, gy::g_yield_fatal, bytes);
      return;
    case MdSym::kReadBad:
      cudaMemcpyFromSymbol(dst, g_read_bad, bytes);
      return;
    case MdSym::kReadSample:
      cudaMemcpyFromSymbol(dst, g_read_sample, bytes);
      return;
    case MdSym::kProbeLdcg:
      cudaMemcpyFromSymbol(dst, kProbeLdcg, bytes);
      return;
    case MdSym::kReadGeom:
      cudaMemcpyFromSymbol(dst, g_read_geom, bytes);
      return;
    case MdSym::kGatherWrote:
      cudaMemcpyFromSymbol(dst, g_gather_wrote, bytes);
      return;
    case MdSym::kPublish:
      cudaMemcpyFromSymbol(dst, g_publish, bytes);
      return;
    case MdSym::kPubInterior:
      cudaMemcpyFromSymbol(dst, g_pub_interior, bytes);
      return;
    case MdSym::kMdFlush:
      cudaMemcpyFromSymbol(dst, g_md_flush, bytes);
      return;
    case MdSym::kXMask:
      cudaMemcpyFromSymbol(dst, g_xmask, bytes);
      return;
    case MdSym::kNlMask:
      cudaMemcpyFromSymbol(dst, g_nlmask, bytes);
      return;
    case MdSym::kPubFlushCyc:
      cudaMemcpyFromSymbol(dst, g_pub_flush_cyc, bytes);
      return;
    case MdSym::kPubFetchCyc:
      cudaMemcpyFromSymbol(dst, g_pub_fetch_cyc, bytes);
      return;
    case MdSym::kMdCyc:
      cudaMemcpyFromSymbol(dst, g_md_cyc, bytes);
      return;
    case MdSym::kPagesDone:
      cudaMemcpyFromSymbol(dst, g_pages_done, bytes);
      return;
    case MdSym::kBlkDone:
      cudaMemcpyFromSymbol(dst, g_blk_done, bytes);
      return;
    case MdSym::kBlkLast:
      cudaMemcpyFromSymbol(dst, g_blk_last, bytes);
      return;
  }
}

void LaunchReadProbe(dim3 grid,
                     dim3 block,
                     size_t smem,
                     clio::run::IpcManagerGpuInfo info,
                     gv::DeviceVector<float> x,
                     u64 passes,
                     u32 nblocks,
                     gy::YieldableView<> yv,
                     gy::YieldStackView ys) {
  ReadProbeKernel<<<grid, block, smem>>>(
      info, x, passes, nblocks, yv, ys);
}

void LaunchIntegrate(dim3 grid,
                     dim3 block,
                     size_t smem,
                     clio::run::IpcManagerGpuInfo info,
                     gv::DeviceVector<float> x,
                     gv::DeviceVector<float> v,
                     gv::DeviceVector<float> third,
                     int use_third,
                     float dt,
                     float gx,
                     float gy_,
                     float gz,
                     int drift,
                     u32 nblocks,
                     gy::YieldableView<> yv,
                     gy::YieldStackView ys) {
  IntegrateKernel<<<grid, block, smem>>>(
      info, x, v, third, use_third, dt, gx, gy_, gz, drift, nblocks, yv, ys);
}

void LaunchPublishSlab(dim3 grid,
                       dim3 block,
                       size_t smem,
                       clio::run::IpcManagerGpuInfo info,
                       gv::DeviceVector<float> x,
                       gv::DeviceVector<float> v,
                       u32 nb,
                       u32 cap,
                       u32 z0,
                       u32 z1,
                       u64 gen,
                       u32 halo_first,
                       u32 nblocks,
                       u32 tbl_base,
                       gy::YieldableView<> yv,
                       gy::YieldStackView ys) {
  PublishSlabKernel<<<grid, block, smem>>>(
      info, x, v, nb, cap, z0, z1, gen, halo_first, nblocks, tbl_base, yv, ys);
}

void LaunchHaloPin(dim3 grid,
                   dim3 block,
                   size_t smem,
                   clio::run::IpcManagerGpuInfo info,
                   gv::DeviceVector<float> x,
                   u32 nb,
                   u32 cap,
                   u32 z0,
                   u32 z1,
                   u64 gen,
                   u32 tbl_base,
                   gy::YieldableView<> yv,
                   gy::YieldStackView ys) {
  HaloPinKernel<<<grid, block, smem>>>(
      info, x, nb, cap, z0, z1, gen, tbl_base, yv, ys);
}

void LaunchRefaultWrite(dim3 grid,
                        dim3 block,
                        size_t smem,
                        clio::run::IpcManagerGpuInfo info,
                        gv::DeviceVector<float> x,
                        u64 pg_lo,
                        u64 pg_hi,
                        u64 round,
                        u64 ppp,
                        u32 nblocks,
                        gy::YieldableView<> yv,
                        gy::YieldStackView ys) {
  RefaultWriteKernel<<<grid, block, smem>>>(
      info, x, pg_lo, pg_hi, round, ppp, nblocks, yv, ys);
}

void LaunchRefaultVerify(dim3 grid,
                         dim3 block,
                         size_t smem,
                         clio::run::IpcManagerGpuInfo info,
                         gv::DeviceVector<float> x,
                         u64 pg_lo,
                         u64 pg_hi,
                         u64 round,
                         u64 ppp,
                         u64 below_pg,
                         u64 above_pg,
                         u32 nblocks,
                         unsigned long long *d_out,
                         gy::YieldableView<> yv,
                         gy::YieldStackView ys) {
  RefaultVerifyKernel<<<grid, block, smem>>>(
      info, x, pg_lo, pg_hi, round, ppp, below_pg, above_pg, nblocks, d_out, yv, ys);
}

void LaunchHaloUnpin(dim3 grid,
                     dim3 block,
                     size_t smem,
                     clio::run::IpcManagerGpuInfo info,
                     gv::DeviceVector<float> x,
                     u32 nb,
                     u32 cap,
                     u32 z0,
                     u32 z1,
                     u32 tbl_base,
                     gy::YieldableView<> yv,
                     gy::YieldStackView ys) {
  HaloUnpinKernel<<<grid, block, smem>>>(
      info, x, nb, cap, z0, z1, tbl_base, yv, ys);
}

void LaunchThermo(dim3 grid,
                  dim3 block,
                  size_t smem,
                  clio::run::IpcManagerGpuInfo info,
                  gv::DeviceVector<float> x,
                  gv::DeviceVector<float> v,
                  double *out,
                  u32 nb,
                  u32 cap,
                  u32 z0,
                  u32 z1,
                  u32 nblocks,
                  gy::YieldableView<> yv,
                  gy::YieldStackView ys) {
  ThermoKernel<<<grid, block, smem>>>(
      info, x, v, out, nb, cap, z0, z1, nblocks, yv, ys);
}

void LaunchForce(dim3 grid,
                 dim3 block,
                 size_t smem,
                 clio::run::IpcManagerGpuInfo info,
                 gv::DeviceVector<float> x,
                 gv::DeviceVector<float> f,
                 u32 nb,
                 u32 cap,
                 float box,
                 float cutoff,
                 int eflag,
                 double *acc,
                 u32 z0,
                 u32 z1,
                 u32 nblocks,
                 u64 hgen,
                 bool force_all,
                 gy::YieldableView<> yv,
                 gy::YieldStackView ys) {
  ForceKernel<<<grid, block, smem>>>(
      info, x, f, nb, cap, box, cutoff, eflag, acc, z0, z1, nblocks, hgen, force_all, yv, ys);
}

void LaunchBuildList(dim3 grid,
                     dim3 block,
                     size_t smem,
                     clio::run::IpcManagerGpuInfo info,
                     gv::DeviceVector<float> x,
                     gv::DeviceVector<int> nl,
                     u32 nb,
                     u32 cap,
                     float box,
                     float rlist,
                     u32 maxneigh,
                     u32 *d_cnt,
                     int *d_err,
                     u32 rowchunk,
                     u32 z0,
                     u32 z1,
                     u32 nblocks,
                     u64 hgen,
                     gy::YieldableView<> yv,
                     gy::YieldStackView ys) {
  BuildListKernel<<<grid, block, smem>>>(
      info, x, nl, nb, cap, box, rlist, maxneigh, d_cnt, d_err, rowchunk, z0, z1, nblocks, hgen, yv, ys);
}

void LaunchListForce(dim3 grid,
                     dim3 block,
                     size_t smem,
                     clio::run::IpcManagerGpuInfo info,
                     gv::DeviceVector<float> x,
                     gv::DeviceVector<float> f,
                     gv::DeviceVector<int> nl,
                     u32 nb,
                     u32 cap,
                     float box,
                     float cutoff,
                     u32 maxneigh,
                     const u32 *d_cnt,
                     int eflag,
                     double *acc,
                     int nocompute,
                     u32 rowchunk,
                     u32 z0,
                     u32 z1,
                     u32 nblocks,
                     u64 hgen,
                     bool force_all,
                     u32 band,
                     gy::YieldableView<> yv,
                     gy::YieldStackView ys) {
  ListForceKernel<<<grid, block, smem>>>(
      info, x, f, nl, nb, cap, box, cutoff, maxneigh, d_cnt, eflag, acc, nocompute, rowchunk, z0, z1, nblocks, hgen, force_all, band, yv, ys);
}

void LaunchRebinWrap(dim3 grid,
                     dim3 block,
                     size_t smem,
                     clio::run::IpcManagerGpuInfo info,
                     gv::DeviceVector<float> x,
                     u32 nb,
                     u32 cap,
                     float box,
                     u32 z0,
                     u32 z1,
                     u32 nblocks,
                     gy::YieldableView<> yv,
                     gy::YieldStackView ys) {
  RebinWrapKernel<<<grid, block, smem>>>(
      info, x, nb, cap, box, z0, z1, nblocks, yv, ys);
}

void LaunchRebinAssign(dim3 grid,
                       dim3 block,
                       size_t smem,
                       clio::run::IpcManagerGpuInfo info,
                       gv::DeviceVector<float> x,
                       u32 nb,
                       u32 cap,
                       float box,
                       u32 *bincnt,
                       u32 *d_dest,
                       int *d_err,
                       u32 z0,
                       u32 z1,
                       u32 nblocks,
                       u64 hgen,
                       gy::YieldableView<> yv,
                       gy::YieldStackView ys) {
  RebinAssignKernel<<<grid, block, smem>>>(
      info, x, nb, cap, box, bincnt, d_dest, d_err, z0, z1, nblocks, hgen, yv, ys);
}

void LaunchGather(dim3 grid,
                  dim3 block,
                  size_t smem,
                  clio::run::IpcManagerGpuInfo info,
                  gv::DeviceVector<float> src,
                  gv::DeviceVector<float> srcx,
                  gv::DeviceVector<float> dst,
                  u32 nb,
                  u32 cap,
                  const u32 *d_dest,
                  int keep_w,
                  u32 z0,
                  u32 z1,
                  u32 nblocks,
                  u64 hgen,
                  gy::YieldableView<> yv,
                  gy::YieldStackView ys) {
  GatherKernel<<<grid, block, smem>>>(
      info, src, srcx, dst, nb, cap, d_dest, keep_w, z0, z1, nblocks, hgen, yv, ys);
}

void LaunchSentinel(dim3 grid,
                    dim3 block,
                    size_t smem,
                    clio::run::IpcManagerGpuInfo info,
                    gv::DeviceVector<float> dst,
                    u32 nb,
                    u32 cap,
                    u32 z0,
                    u32 z1,
                    u32 nblocks,
                    gy::YieldableView<> yv,
                    gy::YieldStackView ys) {
  SentinelKernel<<<grid, block, smem>>>(
      info, dst, nb, cap, z0, z1, nblocks, yv, ys);
}

void LaunchMDIntegrate(dim3 grid,
                       dim3 block,
                       size_t smem,
                       clio::run::IpcManagerGpuInfo info,
                       gv::DeviceVector<float> x,
                       gv::DeviceVector<float> v,
                       gv::DeviceVector<float> f,
                       float dt,
                       int drift,
                       u32 nb,
                       u32 cap,
                       u32 z0,
                       u32 z1,
                       u32 nblocks,
                       gy::YieldableView<> yv,
                       gy::YieldStackView ys) {
  MDIntegrateKernel<<<grid, block, smem>>>(
      info, x, v, f, dt, drift, nb, cap, z0, z1, nblocks, yv, ys);
}

}  // namespace clio::gv_bench::md
