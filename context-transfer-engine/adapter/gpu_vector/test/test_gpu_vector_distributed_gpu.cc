/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Distributed gpu_vector stress test: 4 nodes, one shared tag.
 *
 * Every node runs this SAME binary with an in-process runtime that joins the
 * cluster (hostfile). Per round:
 *
 *   1. WRITE  — node i's GPU writes pages [i*P, (i+1)*P) of this round's
 *               region through the paged cache. The cache is deliberately
 *               SMALLER than the range (slots < P), so the write phase
 *               continuously evicts: dirty victims flush mid-phase, then
 *               FlushAsync/AwaitFlush drains the rest. Each flushed page is a
 *               PodPutBlob whose blob HASHES to some node -- so the writes
 *               land across the whole cluster, not on the writer.
 *   2. BARRIER — a sentinel blob per (node, round), polled via the ordinary
 *               hashed CPU client. No side channels: the barrier is CTE data.
 *   3. READ   — node i's GPU reads the NEXT node's range (ring order).
 *               Nothing is resident, so every page FAULTS; the fault's
 *               PodGetBlob hashes to the owning node and the bytes cross the
 *               wire CPU-side (the cross-node page fault). slots < P keeps
 *               eviction churning through the read too.
 *   4. VERIFY — every element must equal Pattern(global_index, round),
 *               checked ON the GPU through the paged reads.
 *
 * Two things this test deliberately does NOT do, both learned the hard way:
 *
 *   - It never faults through the BLOCKING HoldPage. That path deadlocks
 *     against itself the moment a page needs a writeback: the kernel holds
 *     the SM waiting for it, and a resident kernel blocks every later launch
 *     in its context, including the one that would service the writeback.
 *     The kernels here yield.
 *   - It never re-reads a region an earlier round already pulled in. Each
 *     node's cache is private and there is no cross-node invalidation, so a
 *     reader would serve its own stale copy of a page the peer has since
 *     rewritten. Every round therefore works in its own fresh region.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/gpu_vector/gpu_vector.h>

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace gv = clio::cte::gpu_vector;
namespace gy = clio::run::gpu;
using clio::run::u32;
using clio::run::u64;

static constexpr u64 kPageBytes = 64 * 1024;
static constexpr u64 kPageElems = kPageBytes / sizeof(u32);

/** Deterministic per-element pattern; round-salted so re-reads of a stale
 *  page from an earlier round fail loudly. */
__host__ __device__ inline u32 Pattern(u64 i, u32 round) {
  u64 x = i * 2654435761ull + round * 0x9e3779b9ull;
  x ^= x >> 33;
  return static_cast<u32>(x);
}

/**
 * Write this node's range.
 *
 * A COROUTINE, and that is not a style choice. The blocking HoldPage
 * deadlocked against itself as soon as a page had to be written back: the
 * kernel sat on the SM waiting for the writeback, and a resident kernel
 * blocks every later launch in its context -- including the one that would
 * service it. co_await suspends the block so the fault can be served.
 *
 * The slice is keyed off the driver-assigned block id, not blockIdx.x: the
 * driver relaunches a COMPACTED grid of the still-pending blocks, so
 * blockIdx.x is not stable across resumes. Every lane runs the loop and they
 * split each resolved run; the fault path is block-collective, so no lane
 * may return early.
 */
__device__ gy::YCoroMain WriteRangeCoro(gv::DeviceVector<u32> v,
                                        u64 first_elem, u64 elems_per_block,
                                        u32 round, clio::run::u32 block) {
  const u64 base = first_elem + static_cast<u64>(block) * elems_per_block;
  for (u64 i = 0; i < elems_per_block;) {
    u64 run = 0;
    {
      auto h = co_await v.HoldPage(base + i, elems_per_block - i, /*write=*/true);
      run = h.run();
      for (u64 k = threadIdx.x; k < run; k += blockDim.x) {
        h[base + i + k] = Pattern(base + i + k, round);
      }
    }
    // Flush as we go: the vector never writes back on its own, so a
    // dirty page is unevictable until the caller flushes it. Async, so
    // it overlaps the next page; the servicer retires it once it lands.
    co_await v.BeginFlush();
    i += run;
  }
  co_await v.EndFlush();
}

__global__ void WriteRangeKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<u32> v, u64 first_elem,
                                 u64 elems_per_block, u32 round,
                                 gy::YieldableView<> yv,
                                 gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(
      WriteRangeCoro(v, first_elem, elems_per_block, round, yv.Block()));
}

/** Read + verify a range; accumulates mismatches and a checksum. Every page
 *  here is a fault, and most of them are owned by another node. */
__device__ gy::YCoroMain ReadRangeCoro(gv::DeviceVector<u32> v, u64 first_elem,
                                       u64 elems_per_block, u32 round,
                                       unsigned long long *sum,
                                       unsigned long long *mismatches,
                                       clio::run::u32 block) {
  const u64 base = first_elem + static_cast<u64>(block) * elems_per_block;
  for (u64 i = 0; i < elems_per_block;) {
    auto h = co_await v.HoldPage(base + i, elems_per_block - i);
    for (u64 k = threadIdx.x; k < h.run(); k += blockDim.x) {
      // Read-only sweep: the hold stays write=false, so every page is dropped
      // clean and nothing is written back. THE HOLD IS THE PIN: eviction
      // cannot re-tenant the slot under a lane that is still reading.
      const u32 got = h[base + i + k];
      atomicAdd(sum, static_cast<unsigned long long>(got));
      if (got != Pattern(base + i + k, round)) atomicAdd(mismatches, 1ull);
    }
    i += h.run();
  }
}

__global__ void ReadRangeKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<u32> v, u64 first_elem,
                                u64 elems_per_block, u32 round,
                                unsigned long long *sum,
                                unsigned long long *mismatches,
                                gy::YieldableView<> yv,
                                gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.block_override_ = yv.Block();
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());
  __syncthreads();
  CLIO_YCORO_RUN(ReadRangeCoro(v, first_elem, elems_per_block, round, sum,
                               mismatches, yv.Block()));
}

#if !CTP_IS_DEVICE_PASS

namespace {

int EnvInt(const char *name, int dflt) {
  const char *e = std::getenv(name);
  return e != nullptr ? std::atoi(e) : dflt;
}

/** Phase marker. A hang in this test is only diagnosable if the log says which
 *  phase it died in -- every other line in the container log belongs to the
 *  runtime. Unbuffered so a killed container still shows its last phase. */
void Step(int node, const char *what) {
  std::fprintf(stderr, "[node%d] STEP %s\n", node, what);
  std::fflush(stderr);
}

/** Run a yieldable kernel to completion: a yielding kernel is not launched
 *  once but re-launched until no block is left suspended. */
template <typename LaunchT>
clio::run::u32 RunYieldable(unsigned nblocks, LaunchT &&launch) {
  gy::Yieldable<> drv(nblocks, 32);
  // 8192, not the macro-era 256: the kernels are C++20 coroutines now and
  // their frames spill into the yield lane -- a page-thin lane overflows on
  // the first co_await.
  gy::YieldStack stack(nblocks, 32, 8192);
  return drv.RunToCompletion(
      [&](dim3 g, dim3 b, gy::YieldableView<> view) {
        launch(g, b, view, stack.View());
      },
      [] {}, /*max_rounds=*/200000);
}

/** Cluster barrier over CTE data: publish a sentinel blob for (node, phase),
 *  then poll until every node's sentinel exists. Routed with the ordinary
 *  hashed query, so the sentinels themselves spread across the cluster. */
bool Barrier(clio::cte::core::Client &cte, const clio::cte::core::TagId &tag,
             int node, int nodes, const std::string &phase, int timeout_s) {
  const auto t0 = std::chrono::steady_clock::now();
  // The put can land on a node that hasn't registered its storage target yet
  // (starts are staggered), failing with "no targets". Retry until timeout.
  for (;;) {
    u32 val = 1;
    auto f = cte.AsyncPutBlob(tag, "bar_" + phase + "_" + std::to_string(node),
                              0, sizeof(val),
                              reinterpret_cast<char *>(&val), 1.0f);
    f.Wait();
    if (f->GetReturnCode() == 0) break;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() > timeout_s) {
      std::fprintf(stderr, "[node%d] barrier put failed rc=%d\n", node,
                   f->GetReturnCode());
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  for (int peer = 1; peer <= nodes;) {
    auto f = cte.AsyncGetBlobSize(
        tag, "bar_" + phase + "_" + std::to_string(peer));
    f.Wait();
    if (f->GetReturnCode() == 0 && f->size_ == sizeof(u32)) {
      ++peer;
      continue;
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() > timeout_s) {
      std::fprintf(stderr, "[node%d] barrier '%s' timed out waiting for %d\n",
                   node, phase.c_str(), peer);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return true;
}

}  // namespace

int main() {
  const int node = EnvInt("NODE_ID", 1);           // 1-based, like the harness
  const int nodes = EnvInt("CLIO_NUM_CONTAINERS", 4);
  const u64 pages_per_node = static_cast<u64>(EnvInt("GVD_PAGES", 64));
  const u32 slots = static_cast<u32>(
      EnvInt("GVD_SLOTS", static_cast<int>(pages_per_node / 4)));
  const int rounds = EnvInt("GVD_ROUNDS", 3);
  const int blocks = EnvInt("GVD_BLOCKS", 4);
  const int barrier_timeout_s = EnvInt("GVD_BARRIER_TIMEOUT", 120);

  Step(node, "runtime-init");
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "[node%d] runtime init failed\n", node);
    return 1;
  }
  // Let SWIM/membership settle before creating cluster-spanning pools.
  std::this_thread::sleep_for(std::chrono::seconds(2));
  Step(node, "cte-client-init");
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[node%d] cte client init failed\n", node);
    return 1;
  }
  Step(node, "gpu-info");
  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  // Optional client-driven storage registration. The harness config composes
  // a RAM tier per node, so this is OFF there; it exists for running the
  // binary against a runtime whose config has no compose section. Local() on
  // both queries: the registration runs on this node's container, for a bdev
  // on this node.
  if (EnvInt("GVD_REGISTER_TARGET", 1) != 0) {
    Step(node, "register-target");
    clio::cte::core::Client reg(clio::cte::core::kCtePoolId);
    // Pool creation rejects a null PoolId (EINVAL): supply an explicit id per
    // node, off in (600,*) to stay clear of the compose driver's 512+idx range.
    auto f = reg.AsyncRegisterTarget(
        "ram::gvd_ram_" + std::to_string(node),
        clio::run::bdev::BdevType::kRam, 256ull << 20,
        clio::run::PoolQuery::Local(),
        clio::run::PoolId(600, static_cast<u32>(node)),
        clio::run::PoolQuery::Local());
    f.Wait();
    if (f->GetReturnCode() != 0) {
      std::fprintf(stderr, "[node%d] RegisterTarget failed rc=%d\n", node,
                   f->GetReturnCode());
      return 1;
    }
  }

  const u64 elems_per_node = pages_per_node * kPageElems;
  // Each round gets its OWN region of the vector. The cache is private per
  // node with no cross-node invalidation -- by design; see the header. If
  // round r re-read the region round r-1 already pulled in, the reader would
  // serve its own resident copy and never see the peer's rewrite. That is not
  // a bug to detect, it is the stated model, so the test must not depend on
  // it: a fresh region per round means every page is a real fault every time.
  const u64 elems_per_round = elems_per_node * nodes;
  const u64 total_elems = elems_per_round * static_cast<u64>(rounds);

  Step(node, "vector-ctor");
  gv::Vector<u32> vec("gvd_shared", {0}, kPageBytes, blocks, slots,
                      total_elems);
  vec.EnableStats();
  clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
  const clio::cte::core::TagId tag = vec.TagId();

  const u64 elems_per_block = elems_per_node / blocks;
  unsigned long long *d_out = nullptr;
  d_out = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(d_out)>>(2 * sizeof(unsigned long long));

  Step(node, "barrier-start");
  if (!Barrier(cte, tag, node, nodes, "start", barrier_timeout_s)) return 1;
  Step(node, "barrier-start-done");

  int failures = 0;
  for (int r = 0; r < rounds; ++r) {
    // --- WRITE my range (evicting throughout: slots < pages_per_node) ---
    const u64 round_base = static_cast<u64>(r) * elems_per_round;
    const u64 my_first =
        round_base + static_cast<u64>(node - 1) * elems_per_node;
    Step(node, "write-kernel");
    RunYieldable(blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                             gy::YieldStackView sv) {
      WriteRangeKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu_info, vec.GetDevice(0), my_first, elems_per_block,
          static_cast<u32>(r), vw, sv);
    });
    ctp::GpuApi::Synchronize();
    Step(node, "barrier-write");
    if (!Barrier(cte, tag, node, nodes, "w" + std::to_string(r),
                 barrier_timeout_s)) {
      return 1;
    }

    // --- READ the NEXT node's range: all faults, all (likely) remote ----
    const int peer = (node % nodes) + 1;
    const u64 peer_first =
        round_base + static_cast<u64>(peer - 1) * elems_per_node;
    ctp::GpuApi::Memset(d_out, 0, 2 * sizeof(unsigned long long));
    Step(node, "read-kernel");
    RunYieldable(blocks, [&](dim3 g, dim3 b, gy::YieldableView<> vw,
                             gy::YieldStackView sv) {
      ReadRangeKernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(
          gpu_info, vec.GetDevice(0), peer_first, elems_per_block,
          static_cast<u32>(r), d_out, d_out + 1, vw, sv);
    });
    ctp::GpuApi::Synchronize();
    unsigned long long h[2] = {0, 0};
    ctp::GpuApi::Memcpy(h, d_out, sizeof(h));
    const auto st = vec.ReadStats(0);
    std::fprintf(stderr,
                 "[node%d] round %d: read node%d's %llu pages, sum=%llx "
                 "mismatches=%llu | faults=%llu evicts=%llu puts=%llu "
                 "put_errors=%llu get_errors=%llu\n",
                 node, r, peer, (unsigned long long) pages_per_node, h[0],
                 h[1], (unsigned long long) st.faults,
                 (unsigned long long) st.evicts, (unsigned long long) st.puts,
                 (unsigned long long) st.put_errors,
                 (unsigned long long) st.get_errors);
    if (h[1] != 0 || st.put_errors != 0) ++failures;

    if (!Barrier(cte, tag, node, nodes, "r" + std::to_string(r),
                 barrier_timeout_s)) {
      return 1;
    }
  }

  std::fprintf(stderr, "[node%d] GVD %s: %d rounds, %d failures\n", node,
               failures == 0 ? "PASS" : "FAIL", rounds, failures);
  return failures == 0 ? 0 : 1;
}
#endif  // !CTP_IS_DEVICE_PASS
