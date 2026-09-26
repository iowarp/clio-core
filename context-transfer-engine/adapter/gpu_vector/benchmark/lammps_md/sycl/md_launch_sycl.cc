/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * MD launches, SYCL. Compiled with -fsycl; the ONLY file in this benchmark
 * that is.
 *
 * Nothing here is workload: every coroutine body lives in ../md_kernels.h,
 * shared verbatim with the CUDA side. This TU also owns the yield device
 * globals and the MD globals block (CLIO_SYCL_KERNEL_TU) -- exactly one
 * -fsycl TU per linked program may define a device_global, and each benchmark
 * is its own program.
 *
 * GENERATED FROM THE CUDA TRAMPOLINES rather than hand-copied. All sixteen
 * have the identical shape (init the vectors, publish the yield TLS, run the
 * coroutine), and hand-copying sixteen bodies is how two backends drift apart
 * one argument at a time.
 */

#define CLIO_SYCL_KERNEL_TU 1

#include "md_launch.h"

#include "md_kernels.h"

#include <clio_ctp/util/gpu_api.h>

#include <sycl/sycl.hpp>

namespace clio::gv_bench::md {

namespace {

template <typename BodyT>
void Submit(dim3 grid, dim3 block, BodyT body) {
  auto &q = ctp::GpuApi::SyclQueue();
  const size_t global = static_cast<size_t>(grid.x) * block.x;
  q.parallel_for(
       sycl::nd_range<1>{sycl::range<1>(global), sycl::range<1>(block.x)},
       [=](sycl::nd_item<1>) { body(); })
      .wait();
}

/** The MD globals block, device-side. Allocated once by InitBackend and
 *  installed in the device_global that md_kernels.h reads through MdG(). */
MdGlobals *g_md_host_view = nullptr;

}  // namespace

void InitBackend(u32 max_blocks, const clio::run::IpcManagerGpuInfo &info) {
  ::clio::run::gpu::SyclInitBlockIpcManagers(max_blocks, info);
  if (g_md_host_view != nullptr) return;
  auto &q = ctp::GpuApi::SyclQueue();
  g_md_host_view = sycl::malloc_device<MdGlobals>(1, q);
  // Zero first, then set the two flags whose CUDA declarations defaulted to 1.
  // Getting this wrong would not crash: publish would simply be off, and the
  // physics would be quietly wrong in the way MD_NO_PUBLISH exists to measure.
  q.memset(g_md_host_view, 0, sizeof(MdGlobals)).wait();
  const u32 one = 1u;
  q.copy(&one, reinterpret_cast<u32 *>(
                   reinterpret_cast<char *>(g_md_host_view) +
                   offsetof(MdGlobals, publish)), 1).wait();
  q.copy(&one, reinterpret_cast<u32 *>(
                   reinterpret_cast<char *>(g_md_host_view) +
                   offsetof(MdGlobals, pub_interior)), 1).wait();
  // INSTALL, not read: queue::copy has BOTH directions as overloads
  // (device_global -> host and host -> device_global), so writing the
  // arguments the wrong way round compiles clean and merely leaves the
  // device_global null -- MdG() then dereferences null in every kernel, and
  // this also clobbers g_md_host_view with the uninitialised device value so
  // every later SymbolWrite/SymbolRead targets garbage too. Source first.
  q.copy(&g_md_host_view, g_md_dg, 1).wait();
}

namespace {

struct SymLoc { size_t off, size; };

SymLoc Locate(MdSym sym) {
  switch (sym) {
    case MdSym::kYieldFatal:  return {0, 0};
    case MdSym::kMdCyc:       return {offsetof(MdGlobals, md_cyc), 0};
    case MdSym::kPublish:     return {offsetof(MdGlobals, publish), 0};
    case MdSym::kPubInterior: return {offsetof(MdGlobals, pub_interior), 0};
    case MdSym::kMdFlush:     return {offsetof(MdGlobals, md_flush), 0};
    case MdSym::kGatherWrote: return {offsetof(MdGlobals, gather_wrote), 0};
    case MdSym::kPubFlushCyc: return {offsetof(MdGlobals, pub_flush_cyc), 0};
    case MdSym::kPubFetchCyc: return {offsetof(MdGlobals, pub_fetch_cyc), 0};
    case MdSym::kXMask:       return {offsetof(MdGlobals, xmask), 0};
    case MdSym::kNlMask:      return {offsetof(MdGlobals, nlmask), 0};
    case MdSym::kPagesDone:   return {offsetof(MdGlobals, pages_done), 0};
    case MdSym::kReadBad:     return {offsetof(MdGlobals, read_bad), 0};
    case MdSym::kReadSample:  return {offsetof(MdGlobals, read_sample), 0};
    case MdSym::kProbeLdcg:   return {offsetof(MdGlobals, probe_ldcg), 0};
    case MdSym::kReadGeom:    return {offsetof(MdGlobals, read_geom), 0};
    case MdSym::kBlkDone:     return {offsetof(MdGlobals, blk_done), 0};
    case MdSym::kBlkLast:     return {offsetof(MdGlobals, blk_last), 0};
  }
  return {0, 0};
}

}  // namespace

void SymbolWrite(MdSym sym, const void *src, size_t bytes) {
  auto &q = ctp::GpuApi::SyclQueue();
  if (sym == MdSym::kYieldFatal) {
    q.copy(static_cast<unsigned long long *const *>(src),
           ::clio::run::gpu::g_yield_fatal_dg, 1).wait();
    return;
  }
  q.memcpy(reinterpret_cast<char *>(g_md_host_view) + Locate(sym).off, src,
           bytes).wait();
}

void SymbolRead(void *dst, MdSym sym, size_t bytes) {
  auto &q = ctp::GpuApi::SyclQueue();
  if (sym == MdSym::kYieldFatal) {
    q.copy(::clio::run::gpu::g_yield_fatal_dg,
           static_cast<unsigned long long **>(dst), 1).wait();
    return;
  }
  q.memcpy(dst, reinterpret_cast<char *>(g_md_host_view) + Locate(sym).off,
           bytes).wait();
}

bool DevMemInfo(size_t *free_bytes, size_t *total_bytes) {
  auto &q = ctp::GpuApi::SyclQueue();
  const auto dev = q.get_device();
  *total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
  // SYCL exposes no free-memory query in the core spec. Reporting the total
  // as "free" would make the bench's VRAM ledger silently wrong, so this says
  // "cannot report" and the caller prints nothing.
  *free_bytes = 0;
  return false;
}

const char *LaunchError() {
  // SYCL reports launch failures as exceptions from the queue, not as a
  // sticky flag to peek at; anything fatal has already thrown by here.
  return nullptr;
}

const char *DeviceSyncCheck() {
  // SYCL surfaces failures as exceptions from the queue rather than a sticky
  // status, so a clean return here means "nothing threw", not "nothing was
  // checked" -- the wait is still what pins an async failure to its launch.
  ctp::GpuApi::SyclQueue().wait_and_throw();
  return nullptr;
}

u32 DeviceClockKHz() {
  auto &q = ctp::GpuApi::SyclQueue();
  return static_cast<u32>(
      q.get_device().get_info<sycl::info::device::max_clock_frequency>() *
      1000u);
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(ReadProbeCoro(x_, passes, nblocks, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gv::DeviceVector<float> v_ = v;
    v_.Init(yv.Block());
    gv::DeviceVector<float> third_ = third;
    third_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(IntegrateCoro(x_, v_, third_, use_third, dt, gx, gy_, gz, drift, nblocks, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(tbl_base + yv.Block());
    gv::DeviceVector<float> v_ = v;
    v_.Init(tbl_base + yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(PublishSlabCoro(x_, v_, nb, cap, z0, z1, gen, halo_first, nblocks, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(tbl_base + yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(HaloPinCoro(x_, nb, cap, z0, z1, gen, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(RefaultWriteCoro(x_, pg_lo, pg_hi, round, ppp, nblocks, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(RefaultVerifyCoro(x_, pg_lo, pg_hi, round, ppp, below_pg, above_pg, nblocks, yv.Block(), d_out));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(tbl_base + yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(HaloUnpinCoro(x_, nb, cap, z0, z1, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gv::DeviceVector<float> v_ = v;
    v_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(ThermoCoro(x_, v_, out, nb, cap, z0, z1, nblocks, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gv::DeviceVector<float> f_ = f;
    f_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(ForceCoro(x_, f_, nb, cap, box, cutoff, eflag, acc, z0, z1, nblocks, yv.Block(), hgen, force_all));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gv::DeviceVector<int> nl_ = nl;
    nl_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(BuildListCoro(x_, nl_, nb, cap, box, rlist, maxneigh, d_cnt, d_err, rowchunk, z0, z1, nblocks, yv.Block(), hgen));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gv::DeviceVector<float> f_ = f;
    f_.Init(yv.Block());
    gv::DeviceVector<int> nl_ = nl;
    nl_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(ListForceCoro(x_, f_, nl_, nb, cap, box, cutoff, maxneigh, d_cnt, eflag, acc, nocompute, rowchunk, z0, z1, nblocks, yv.Block(), hgen, force_all, band));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(RebinWrapCoro(x_, nb, cap, box, z0, z1, nblocks, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(RebinAssignCoro(x_, nb, cap, box, bincnt, d_dest, d_err, z0, z1, nblocks, yv.Block(), hgen));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> src_ = src;
    src_.Init(yv.Block());
    gv::DeviceVector<float> srcx_ = srcx;
    srcx_.Init(yv.Block());
    gv::DeviceVector<float> dst_ = dst;
    dst_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(GatherCoro(src_, srcx_, dst_, nb, cap, d_dest, keep_w, z0, z1, nblocks, yv.Block(), hgen));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> dst_ = dst;
    dst_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(SentinelCoro(dst_, nb, cap, z0, z1, nblocks, yv.Block()));
  });
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
  // info is stamped once by InitBackend, not per launch; SYCL sizes its own
  // scratch, so the CUDA dynamic-shared byte count is not used here.
  (void)info;
  (void)smem;
  Submit(grid, block, [=]() {
    gv::DeviceVector<float> x_ = x;
    x_.Init(yv.Block());
    gv::DeviceVector<float> v_ = v;
    v_.Init(yv.Block());
    gv::DeviceVector<float> f_ = f;
    f_.Init(yv.Block());
    gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
    __syncthreads();
    CLIO_YCORO_RUN(MDIntegrateCoro(x_, v_, f_, dt, drift, nb, cap, z0, z1, nblocks, yv.Block()));
  });
}

}  // namespace clio::gv_bench::md

namespace clio::run::gpu {

/**
 * The out-of-line half of YieldStack::Reset; see yield_stack.h. Exactly one
 * -fsycl TU per linked program may define this, for the same reason it owns
 * the device_globals: every -fsycl TU produces its own device image, and the
 * yield smem base has to be installed in the image the kernels actually run
 * from.
 */
void SyclYieldStackReset(const YieldStackView &view, clio::run::u32 nlanes,
                         char *smem_base) {
  auto &q = ctp::GpuApi::SyclQueue();
  YieldStackView v = view;
  q.parallel_for(sycl::range<1>(nlanes), [=](sycl::id<1> i) {
     auto *h = reinterpret_cast<YieldLaneHeader *>(
         v.base_ + static_cast<clio::run::u64>(i[0]) * v.bytes_per_lane_);
     h->sp_ = sizeof(YieldLaneHeader);   // the header is not frame space
     h->live_depth_ = 0;
     h->cur_depth_ = 0;
     h->error_ = kYieldErrNone;
     h->coro_resume_ = 0;
     h->coro_top_ = 0;
     h->coro_park_ = 0;
   }).wait();
  char *base = smem_base;
  q.copy(&base, g_yield_smem_dg, 1).wait();
}

}  // namespace clio::run::gpu
