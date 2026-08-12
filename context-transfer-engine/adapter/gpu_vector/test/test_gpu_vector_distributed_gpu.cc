/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Distributed gpu_vector stress test: 4 nodes, one shared tag.
 *
 * Every node runs this SAME binary with an in-process runtime that joins the
 * cluster (hostfile). The flow, per round:
 *
 *   1. WRITE  — node i's GPU writes pages [i*P, (i+1)*P) of the shared
 *               vector through the paged cache. The cache is deliberately
 *               SMALLER than the range (slots = P/4), so the write phase
 *               continuously evicts: dirty victims flush mid-phase, then
 *               BeginFlush/WaitFlush drains the rest. Each flushed page is a
 *               PodPutBlob whose blob HASHES to some node -- so the writes
 *               land across the whole cluster, not on the writer.
 *   2. BARRIER — a sentinel blob per (node, round), polled via the ordinary
 *               hashed CPU client. No side channels: the barrier is CTE data.
 *   3. READ   — node i's GPU reads the NEXT node's range (ring order).
 *               Nothing is resident, so every page FAULTS; the fault's
 *               PodGetBlob hashes to the owning node and the bytes cross the
 *               wire CPU-side (the cross-node page fault). slots < P keeps
 *               eviction churning through the read too.
 *   4. VERIFY — every element must equal Pattern(global_index, round). The
 *               checksum is computed ON the GPU from the paged reads.
 *
 * Rounds > 1 re-write with a different salt, so stale pages from the prior
 * round are a detectable corruption, not silently-correct leftovers.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
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

/** Write this node's range. Single-threaded per block on purpose: the
 *  blocking HoldPage fault path is only correct from the faulting lane. */
__global__ void WriteRangeKernel(clio::run::IpcManagerGpuInfo info,
                                 gv::DeviceVector<u32> v, u64 first_elem,
                                 u64 elems_per_block, u32 round) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  const u64 base = first_elem + static_cast<u64>(blockIdx.x) * elems_per_block;
  for (u64 i = 0; i < elems_per_block;) {
    const u64 run = v.HoldPage(base + i, elems_per_block - i);
    for (u64 k = 0; k < run; ++k, ++i) {
      v[base + i] = Pattern(base + i, round);
    }
  }
  v.BeginFlush(base, elems_per_block);
  v.WaitFlush(base, elems_per_block);
}

/** Read + verify a range; accumulates mismatches and a checksum. */
__global__ void ReadRangeKernel(clio::run::IpcManagerGpuInfo info,
                                gv::DeviceVector<u32> v, u64 first_elem,
                                u64 elems_per_block, u32 round,
                                unsigned long long *sum,
                                unsigned long long *mismatches) {
  CLIO_GPU_INIT(info, nullptr);
  if (threadIdx.x != 0) return;
  const u64 base = first_elem + static_cast<u64>(blockIdx.x) * elems_per_block;
  unsigned long long acc = 0, bad = 0;
  for (u64 i = 0; i < elems_per_block;) {
    const u64 run = v.HoldPage(base + i, elems_per_block - i);
    for (u64 k = 0; k < run; ++k, ++i) {
      const u32 got = v.at(base + i);
      acc += got;
      if (got != Pattern(base + i, round)) ++bad;
    }
  }
  atomicAdd(sum, acc);
  atomicAdd(mismatches, bad);
}

#if !CTP_IS_DEVICE_PASS

namespace {

int EnvInt(const char *name, int dflt) {
  const char *e = std::getenv(name);
  return e != nullptr ? std::atoi(e) : dflt;
}

/** Cluster barrier over CTE data: publish a sentinel blob for (node, phase),
 *  then poll until every node's sentinel exists. Routed with the ordinary
 *  hashed query, so the sentinels themselves spread across the cluster. */
bool Barrier(clio::cte::core::Client &cte, const clio::cte::core::TagId &tag,
             int node, int nodes, const std::string &phase, int timeout_s) {
  {
    u32 val = 1;
    auto f = cte.AsyncPutBlob(tag, "bar_" + phase + "_" + std::to_string(node),
                              0, sizeof(val),
                              reinterpret_cast<char *>(&val), 1.0f);
    f.Wait();
    if (f->GetReturnCode() != 0) {
      std::fprintf(stderr, "[node%d] barrier put failed rc=%d\n", node,
                   f->GetReturnCode());
      return false;
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
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

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::fprintf(stderr, "[node%d] runtime init failed\n", node);
    return 1;
  }
  // Let SWIM/membership settle before creating cluster-spanning pools.
  std::this_thread::sleep_for(std::chrono::seconds(2));
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    std::fprintf(stderr, "[node%d] cte client init failed\n", node);
    return 1;
  }
  clio::run::IpcManagerGpuInfo gpu_info =
      CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);

  const u64 elems_per_node = pages_per_node * kPageElems;
  const u64 total_elems = elems_per_node * nodes;

  gv::Vector<u32> vec("gvd_shared", {0}, kPageBytes, blocks, slots,
                      total_elems);
  vec.EnableStats();
  clio::cte::core::Client cte(clio::cte::core::kCtePoolId);
  const clio::cte::core::TagId tag = vec.TagId();

  const u64 elems_per_block = elems_per_node / blocks;
  unsigned long long *d_out = nullptr;
  cudaMalloc(&d_out, 2 * sizeof(unsigned long long));

  if (!Barrier(cte, tag, node, nodes, "start", barrier_timeout_s)) return 1;

  int failures = 0;
  for (int r = 0; r < rounds; ++r) {
    // --- WRITE my range (evicting throughout: slots < pages_per_node) ---
    const u64 my_first = static_cast<u64>(node - 1) * elems_per_node;
    WriteRangeKernel<<<blocks, 32>>>(gpu_info, vec.GetDevice(0), my_first,
                                     elems_per_block, static_cast<u32>(r));
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "[node%d] write kernel failed (round %d)\n", node,
                   r);
      return 1;
    }
    if (!Barrier(cte, tag, node, nodes, "w" + std::to_string(r),
                 barrier_timeout_s)) {
      return 1;
    }

    // --- READ the NEXT node's range: all faults, all (likely) remote ----
    const int peer = (node % nodes) + 1;
    const u64 peer_first = static_cast<u64>(peer - 1) * elems_per_node;
    cudaMemset(d_out, 0, 2 * sizeof(unsigned long long));
    ReadRangeKernel<<<blocks, 32>>>(gpu_info, vec.GetDevice(0), peer_first,
                                    elems_per_block, static_cast<u32>(r),
                                    d_out, d_out + 1);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "[node%d] read kernel failed (round %d)\n", node,
                   r);
      return 1;
    }
    unsigned long long h[2] = {0, 0};
    cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost);
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
