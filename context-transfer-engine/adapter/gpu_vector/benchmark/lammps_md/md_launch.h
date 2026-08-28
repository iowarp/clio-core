/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The sixteen launches the paged MD bench needs, as a backend-neutral
 * interface -- the same seam kmeans, grayscott, gmx, weights and lbann use.
 *
 * A CUDA kernel is a __global__ at namespace scope; a SYCL kernel is a LAMBDA
 * INSIDE ITS LAUNCHER, and DPC++ member-checks the whole translation unit, so
 * a TU holding both the kernels and gpu_vector.h's host-only Vector cannot
 * compile. The workload itself is in md_kernels.h, once.
 */
#ifndef CLIO_GV_BENCH_MD_LAUNCH_H_
#define CLIO_GV_BENCH_MD_LAUNCH_H_

#include "md_common.h"

#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>

#include <cstddef>

namespace clio::gv_bench::md {

/** Per-block device state the SYCL backend allocates once; no-op on CUDA. */
void InitBackend(u32 max_blocks, const clio::run::IpcManagerGpuInfo &info);

/**
 * Device globals the host driver reads or writes. Named through an enum
 * rather than directly because under SYCL they are `device_global`s owned by
 * the launch TU, which the host cannot address.
 */
enum class MdSym {
  kYieldFatal,
  kReadBad,
  kReadSample,
  kProbeLdcg,
  kReadGeom,
  kGatherWrote,
  kPublish,
  kPubInterior,
  kMdFlush,
  kXMask,
  kNlMask,
  kPubFlushCyc,
  kPubFetchCyc,
  kMdCyc,
  kPagesDone,
  kBlkDone,
  kBlkLast,
};

/**
 * The three remaining backend-specific host verbs. Pinned host memory is not
 * among them -- ctp::GpuApi::MallocHost is already portable.
 */
/** Free/total device memory. False if the backend cannot report it. */
bool DevMemInfo(size_t *free_bytes, size_t *total_bytes);
/** Message for a failed launch, or nullptr if the last launch was clean.
 *  Must NOT clear a sticky error: the driver reads the device fatal channel
 *  after this and a clear would turn the story into a bare error code. */
const char *LaunchError();
/** Device clock in kHz, for turning clock64() deltas into microseconds. */
u32 DeviceClockKHz();

void SymbolWrite(MdSym sym, const void *src, size_t bytes);
void SymbolRead(void *dst, MdSym sym, size_t bytes);

void LaunchReadProbe(dim3 grid,
                     dim3 block,
                     size_t smem,
                     clio::run::IpcManagerGpuInfo info,
                     gv::DeviceVector<float> x,
                     u64 passes,
                     u32 nblocks,
                     gy::YieldableView<> yv,
                     gy::YieldStackView ys);

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
                     gy::YieldStackView ys);

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
                       gy::YieldStackView ys);

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
                   gy::YieldStackView ys);

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
                        gy::YieldStackView ys);

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
                         gy::YieldStackView ys);

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
                     gy::YieldStackView ys);

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
                  gy::YieldStackView ys);

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
                 gy::YieldStackView ys);

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
                     gy::YieldStackView ys);

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
                     gy::YieldStackView ys);

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
                     gy::YieldStackView ys);

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
                       gy::YieldStackView ys);

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
                  gy::YieldStackView ys);

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
                    gy::YieldStackView ys);

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
                       gy::YieldStackView ys);

}  // namespace clio::gv_bench::md

#endif  // CLIO_GV_BENCH_MD_LAUNCH_H_
