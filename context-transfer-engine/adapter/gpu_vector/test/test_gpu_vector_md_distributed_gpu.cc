/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Distributed paged halo exchange -- the coherence case MD needs.
 *
 * WHAT THIS IS. An MD domain is split into slabs of rows, one slab per node,
 * and every step each node needs its neighbour's BOUNDARY row: the nine-row
 * stencil reaches one row past the slab edge. So the same page is resident on
 * two nodes, written by one and read by the other, EVERY STEP.
 *
 * WHAT IT IS NOT. It is not the LJ kernel. The physics is a stand-in -- what
 * is under test is the exchange, not the force loop, and dragging the pair
 * loop in would only make a coherence failure harder to attribute.
 *
 * WHY THE EXISTING DISTRIBUTED TEST DOES NOT COVER THIS. That one is explicit
 * that it "never re-reads a region an earlier round already pulled in ...
 * every round therefore works in its own fresh region", because a private
 * per-node cache with no invalidation would serve a reader its own stale copy.
 * MD cannot do that: it re-reads the same halo every step. Reuse of one fixed
 * region across rounds is the entire point here.
 *
 * WHAT MAKES IT SAFE. Generations. The halo is published as generation r and
 * read AS generation r, so the reader's get is not served until the writer's
 * put has landed -- and a resident copy from round r-1 is refused rather than
 * returned. Note what is deliberately absent: THERE IS NO BARRIER BETWEEN THE
 * PUBLISH AND THE READ. If generations did not work, the reader would race
 * ahead and see round r-1's bytes (or zeros through create_on_get_), and the
 * round-salted pattern would catch it. The generation IS the barrier, and
 * that claim is what this test exists to check.
 *
 * The start barrier remains: that one is cluster formation, not coherence.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

namespace {
constexpr u64 kPageBytes = 64ull << 10;
constexpr u64 kPageElems = kPageBytes / sizeof(u32);
/** One "row" is one page: the halo is then exactly one page, which is the
 *  granularity the vector transfers and the smallest thing that can be
 *  resident on two nodes at once. */
constexpr u64 kRowElems = kPageElems;
constexpr u32 kCheck = 512;   // elements of the halo verified per round

/** Round-salted so a copy from an earlier round fails loudly rather than
 *  looking like plausible data. */
CTP_INLINE_CROSS_FUN u32 Pattern(u64 row, u64 i, u32 round) {
  return static_cast<u32>(row * 2654435761ull + i * 40503ull +
                          round * 0x9E3779B9ull + 1ull);
}
}  // namespace

/** Write every row this node owns, then publish the slab AS `gen`. */
__device__ gy::YCoroMain PublishCoro(gv::DeviceVector<u32> v, u64 first_row,
                                     u64 nrows, u32 round, u64 gen,
                                     u32 nblocks, u32 block) {
  for (u64 r = block; r < nrows; r += nblocks) {
    const u64 off = (first_row + r) * kRowElems;
    {
      auto h = co_await v.UpdateRange(off, kRowElems, block, /*write=*/true);
      for (u64 i = threadIdx.x; i < kRowElems; i += blockDim.x) {
        // Element 0 carries the round that wrote this row. A reader cannot
        // assume WHICH round it will see -- only that it is not an older one
        // -- so it has to be told, in the data, which snapshot it got.
        h[off + i] = (i == 0) ? round : Pattern(first_row + r, i, round);
      }
    }
    // Publish AS this generation: a reader naming it is released only now.
    co_await v.Flush(gen, off, kRowElems);
    // UpdateRange fetches, and a fetch pins: release it after the flush.
    v.UnpinRange(v.PageLo(off), v.PageSpan(off, kRowElems));
  }
}

/**
 * Read the neighbour's boundary row, naming the generation.
 *
 * WHAT IS ASSERTED, and why it is not "equals round r". A generational get
 * promises AT LEAST the generation asked for; the nodes run without a
 * per-round barrier, so the writer can be a round or two ahead and returning
 * NEWER bytes is correct. The property that matters -- the one a stale cache
 * would break -- is that the reader never sees an OLDER round, and that what
 * it sees is one coherent snapshot rather than a mix of two.
 *
 * This block already holds this page from the previous round at the previous
 * generation. Serving that copy is the failure mode.
 */
__device__ gy::YCoroMain HaloCoro(gv::DeviceVector<u32> v, u64 halo_row,
                                  u32 round, u64 gen, unsigned long long *bad,
                                  unsigned long long *older, u32 block) {
  if (block != 0) co_return;   // one block does the halo; the rest are idle
  const u64 off = halo_row * kRowElems;
  co_await v.Fetch(gen, off, kRowElems);
  auto h = co_await v.HoldPage(off, kRowElems);
  const u32 seen = h[off];     // the round these bytes were written in
  if (threadIdx.x == 0 && seen < round) {
    // STALE: older than the generation this get named.
    atomicAdd(older, 1ull);
    printf("[halo] row=%llu STALE: asked round %u, got round %u\n",
           (unsigned long long) halo_row, round, seen);
    // WHICH FRAME DID WE READ? If more than one frame in this block's table
    // holds the page, Find() picks whichever comes first -- which may be the
    // older copy. Dump every frame claiming it, with what it holds.
    const gv::Page *tbl = v.TableForDebug();
    const u64 pn = v.PageOf(off);
    u32 holders = 0;
    // The whole cache, not a block's share of it: there are no shares.
    for (u64 i = 0; i < v.NumFrames(); ++i) {
      if (tbl[i].page_num == pn) {
        ++holders;
        printf("[halo]   frame %u holds page %llu gen=%llu valid=[%u,%u) "
               "pins=%u fetching=%u flushing=%u\n",
               i, (unsigned long long) pn,
               (unsigned long long) tbl[i].generation, tbl[i].valid_lo,
               tbl[i].valid_hi, tbl[i].pins, tbl[i].fetching,
               tbl[i].flushing);
      }
    }
    printf("[halo]   %u frame(s) hold page %llu; asked gen %llu\n", holders,
           (unsigned long long) pn, (unsigned long long) gen);
  }
  // Whatever snapshot it is, it must be a WHOLE one.
  for (u64 i = threadIdx.x + 1; i < kCheck; i += blockDim.x) {
    if (h[off + i] != Pattern(halo_row, i, seen)) {
      if (atomicAdd(bad, 1ull) == 0) {
        printf("[halo] row=%llu TORN at i=%llu: got=%u want=%u (round %u)\n",
               (unsigned long long) halo_row, (unsigned long long) i,
               h[off + i], Pattern(halo_row, i, seen), seen);
      }
    }
  }
  v.UnpinRange(v.PageLo(off), v.PageSpan(off, kRowElems));
}

__global__ void PublishKernel(clio::run::IpcManagerGpuInfo info,
                              gv::DeviceVector<u32> v, u64 first_row,
                              u64 nrows, u32 round, u64 gen, u32 nblocks,
                              gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(
      PublishCoro(v, first_row, nrows, round, gen, nblocks, yv.Block()));
}

__global__ void HaloKernel(clio::run::IpcManagerGpuInfo info,
                           gv::DeviceVector<u32> v, u64 halo_row, u32 round,
                           u64 gen, unsigned long long *bad,
                           unsigned long long *older, gy::YieldableView<> yv,
                           gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(HaloCoro(v, halo_row, round, gen, bad, older, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS
namespace {

int EnvInt(const char *name, int dflt) {
  const char *e = std::getenv(name);
  return e != nullptr ? std::atoi(e) : dflt;
}

void Step(int node, const char *what) {
  std::fprintf(stderr, "[md-node%d] STEP %s\n", node, what);
  std::fflush(stderr);
}

template <typename LaunchT>
u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  gy::YieldStack stack(nblocks, 32, 8192);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000, gv::ResumeWhenComplete);
}

/** Cluster formation only -- NOT used between publish and read. */
bool StartBarrier(clio::cte::core::Client &cte,
                  const clio::cte::core::TagId &tag, int node, int nodes,
                  int timeout_s) {
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    u32 val = 1;
    auto f = cte.AsyncPutBlob(tag, "mdbar_" + std::to_string(node), 0,
                              sizeof(val), reinterpret_cast<char *>(&val),
                              1.0f);
    f.Wait();
    if (f->GetReturnCode() == 0) break;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() > timeout_s) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  for (int peer = 1; peer <= nodes;) {
    u32 val = 0;
    auto f = cte.AsyncGetBlob(tag, "mdbar_" + std::to_string(peer), 0,
                              sizeof(val), 0u,
                              reinterpret_cast<char *>(&val));
    f.Wait();
    if (f->GetReturnCode() == 0 && val == 1) {
      ++peer;
      continue;
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() > timeout_s) {
      std::fprintf(stderr, "[md-node%d] start barrier timeout at peer %d\n",
                   node, peer);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return true;
}

}  // namespace

int main() {
  const int node = EnvInt("NODE_ID", 1);            // 1-based
  const int nodes = EnvInt("CLIO_NUM_CONTAINERS", 2);
  const u64 rows_per_node = static_cast<u64>(EnvInt("GVMD_ROWS", 8));
  const int rounds = EnvInt("GVMD_ROUNDS", 4);
  const int blocks = EnvInt("GVMD_BLOCKS", 4);
  const int timeout_s = EnvInt("GVMD_BARRIER_TIMEOUT", 180);
  // Deliberately smaller than a slab, so publishing evicts and the halo page
  // is a real transfer rather than something that never left the cache.
  const u32 slots = static_cast<u32>(EnvInt("GVMD_SLOTS", 4));

  Step(node, "runtime-init");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "[md-node%d] runtime init failed\n", node);
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[md-node%d] cte client init failed\n", node);
    return 1;
  }
  clio::run::IpcManagerGpuInfo gpu =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  if (EnvInt("GVMD_REGISTER_TARGET", 1) != 0) {
    Step(node, "register-target");
    clio::cte::core::Client reg(clio::cte::core::kCtePoolId);
    auto f = reg.AsyncRegisterTarget(
        "ram::gvmd_ram_" + std::to_string(node),
        clio::run::bdev::BdevType::kRam, 256ull << 20,
        clio::run::PoolQuery::Local(),
        clio::run::PoolId(610, static_cast<u32>(node)),
        clio::run::PoolQuery::Local());
    f.Wait();
    if (f->GetReturnCode() != 0) {
      std::fprintf(stderr, "[md-node%d] RegisterTarget rc=%d\n", node,
                   f->GetReturnCode());
      return 1;
    }
  }

  // ONE FIXED REGION for every round -- the whole point. Rows are laid out
  // slab by slab, so node i owns [i*rows_per_node, (i+1)*rows_per_node) and
  // its halo is the first row of the next slab, in ring order.
  const u64 total_rows = rows_per_node * static_cast<u64>(nodes);
  const u64 total_elems = total_rows * kRowElems;
  const u64 my_first_row = static_cast<u64>(node - 1) * rows_per_node;
  const u64 halo_row =
      (static_cast<u64>(node % nodes)) * rows_per_node;   // next slab's first

  Step(node, "vector-ctor");
  gv::Vector<u32> vec("gvmd_shared", {0}, kPageBytes, blocks, slots,
                      total_elems);
  vec.EnableStats();
  clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
  const clio::cte::core::TagId tag = vec.TagId();

  unsigned long long *d_bad =
      ctp::GpuApi::Malloc<unsigned long long>(sizeof(unsigned long long));
  unsigned long long *d_older =
      ctp::GpuApi::Malloc<unsigned long long>(sizeof(unsigned long long));

  Step(node, "barrier-start");
  if (!StartBarrier(cte, tag, node, nodes, timeout_s)) return 1;

  int failures = 0;
  u64 total_faults = 0;
  for (int r = 0; r < rounds; ++r) {
    const u64 gen = static_cast<u64>(r) + 1;   // 0 means "no generation"

    Step(node, "publish");
    RunYieldable(blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                             gy::YieldStackView sv) {
      PublishKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, vec.GetDevice(0), my_first_row, rows_per_node,
          static_cast<u32>(r), gen, static_cast<u32>(blocks), vw, sv);
    });
    ctp::GpuApi::Synchronize();

    // NO BARRIER HERE. The generation is the barrier -- see the header.
    const auto before = vec.ReadStats(0);
    ctp::GpuApi::Memset(d_bad, 0, sizeof(unsigned long long));
    ctp::GpuApi::Memset(d_older, 0, sizeof(unsigned long long));
    Step(node, "halo");
    RunYieldable(blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                             gy::YieldStackView sv) {
      HaloKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu, vec.GetDevice(0), halo_row, static_cast<u32>(r), gen, d_bad,
          d_older, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    const auto after = vec.ReadStats(0);

    unsigned long long bad = 0, older = 0;
    ctp::GpuApi::Memcpy(&bad, d_bad, sizeof(bad));
    ctp::GpuApi::Memcpy(&older, d_older, sizeof(older));
    const u64 faults = after.faults - before.faults;
    total_faults += faults;
    std::fprintf(stderr,
                 "[md-node%d] round %d gen %llu: halo row %llu torn=%llu "
                 "stale=%llu faults=%llu get_err=%llu\n",
                 node, r, (unsigned long long)gen,
                 (unsigned long long)halo_row, bad, older,
                 (unsigned long long)faults,
                 (unsigned long long)(after.get_errors - before.get_errors));
    if (bad != 0 || older != 0) ++failures;
  }

  // A round after the first MUST have refetched the halo: it is the same page
  // every time, so a zero here means the reader served its own stale copy and
  // the round-salted pattern only happened not to catch it.
  if (total_faults == 0) {
    std::fprintf(stderr,
                 "[md-node%d] FAIL: no halo fault in %d rounds -- the reader "
                 "never went back to the owner\n",
                 node, rounds);
    ++failures;
  }

  std::fprintf(stderr, "[md-node%d] %s (%d failures, %llu halo faults)\n",
               node, failures == 0 ? "PASS" : "FAIL", failures,
               (unsigned long long)total_faults);
  clio::run::CLIO_RUNTIME_FINALIZE();
  return failures == 0 ? 0 : 1;
}
#endif  // !CTP_IS_DEVICE_PASS
