/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * Multi-node GPU blob-exchange test — one rank per node, one GPU per rank.
 *
 * Every other GPU test in this repo is single-process and single-node (they
 * bind CLIO_BIND_ADDR=127.0.0.1 and never leave the box). This one scales the
 * GPU path across a real N-node Clio cluster: the payload is PRODUCED on rank
 * i's GPU, travels through CTE to another node, and is VERIFIED on the
 * consuming rank's GPU. Every byte that is checked is checked by a CUDA
 * kernel, so all N GPUs do real work.
 *
 * Ring exchange (N ranks, N nodes, rank i on node_of[i]):
 *
 *   Phase 1  LOCAL round trip, all ranks concurrently.
 *            fill kernel -> D2H -> PutBlob -> GetBlob -> H2D -> verify kernel.
 *            Proves the per-node GPU+CTE path works under N-way concurrency
 *            before any cross-node claim is made. A Phase-1 failure means the
 *            cluster or the build is broken, not the network path.
 *
 *   Phase 2  CROSS-NODE exchange. Rank i publishes blob "xn_b<i>" holding its
 *            own GPU-generated pattern, barrier, then rank i reads back
 *            "xn_b<peer>" where peer = (i + N - 1) % N — a blob it never
 *            wrote, produced on a different node's GPU — and verifies the
 *            PEER's pattern on its own GPU. This is the authoritative
 *            assertion: the bytes must have crossed the wire.
 *
 *   Phase 3  (optional, XNODE_FORCE_PHYSICAL=1) repeats the exchange with
 *            PoolQuery::Physical(next_node) so the remote hop is pinned to a
 *            named node instead of left to Dynamic placement. Off by default
 *            because Physical routing of blob ops is not what the existing
 *            cross-node tests exercise; turn it on to probe it deliberately.
 *
 * MPI is used ONLY for coordination — rank/node discovery, sharing the tag
 * handle, barriers, and reducing results. It never carries the payload under
 * test; that is CTE's job.
 *
 * NOTE ON SCOPE: the cross-node hop moves the payload through host memory.
 * The runtime has no GPUDirect/NCCL path today (no source file in the tree
 * combines MPI and CUDA, and nothing references GPUDirect or gdrcopy), so
 * device-to-device across nodes is not something this test can assert. What
 * it does assert is GPU-produced/GPU-verified data surviving a cross-node CTE
 * round trip on N nodes at once.
 *
 * Requires a cluster that is ALREADY RUNNING: one `clio_run runtime start`
 * daemon per node, all sharing one hostfile. Each rank attaches to its own
 * local daemon with CLIO_INIT(kClient). See clio-core-xnode-gpu-8node.slurm.
 *
 * Exit code 0 == every rank passed every enabled phase.
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

// macros.h first: CTP_IS_DEVICE_PASS gates everything below. The CTE client
// API lives inside `#if CTP_IS_HOST`, so in nvcc's device pass those members
// do not exist — host code that calls them has to be preprocessed away, not
// merely left uncalled.
#include <clio_ctp/constants/macros.h>
#include <clio_ctp/util/gpu_api.h>

namespace {

/**
 * Per-rank payload pattern. Distinct per rank so a consumer that verifies the
 * PEER's seed cannot pass on its own leftover bytes.
 * @param seed Producer's seed (derived from its rank)
 * @param i Byte index within the blob
 * @return Expected byte value
 */
CTP_CROSS_FUN unsigned char PayloadByte(unsigned int seed,
                                        unsigned long long i) {
  return static_cast<unsigned char>(((i * 131ull + 7ull) ^ seed) & 0xFFull);
}

/**
 * Seed for a given producer rank.
 * @param rank Producer's MPI rank
 * @return Seed feeding PayloadByte
 */
CTP_CROSS_FUN unsigned int SeedForRank(int rank) {
  return 0x5Au + static_cast<unsigned int>(rank) * 31u;
}

}  // namespace

// ============================================================================
// GPU kernels — the payload is produced and checked on the device.
// ============================================================================

/**
 * Write the rank's pattern across the device buffer.
 * @param buf Device buffer
 * @param size Bytes to write
 * @param seed Producer seed
 */
__global__ void XnFillKernel(unsigned char *buf, unsigned long long size,
                             unsigned int seed) {
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (; i < size; i += stride) {
    buf[i] = PayloadByte(seed, i);
  }
}

/**
 * Count bytes that do not match the expected pattern.
 * @param buf Device buffer to check
 * @param size Bytes to check
 * @param seed Expected producer seed
 * @param mismatches Device accumulator, incremented per bad byte
 */
__global__ void XnVerifyKernel(const unsigned char *buf,
                               unsigned long long size, unsigned int seed,
                               unsigned int *mismatches) {
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  unsigned int local = 0;
  for (; i < size; i += stride) {
    if (buf[i] != PayloadByte(seed, i)) ++local;
  }
  if (local != 0) atomicAdd(mismatches, local);
}

#if !CTP_IS_DEVICE_PASS

#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <clio_cte/core/core_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/ipc_manager.h>

namespace {

/** CTE core pool composed on every node (pool_id 512.0 in the cluster yaml). */
constexpr clio::run::u32 kCorePoolMajor = 512;
constexpr const char *kTagName = "/xnode_gpu/shared_tag";

/** Exit codes, distinct per failure site so a batch log pinpoints the phase. */
enum XnodeRc : int {
  kOk = 0,
  kRcAttach = 2,
  kRcTag = 3,
  kRcTopology = 4,
  kRcLocalPut = 5,
  kRcLocalGet = 6,
  kRcLocalVerify = 7,
  kRcRemotePut = 8,
  kRcRemoteGet = 9,
  kRcRemoteVerify = 10,
  kRcPhysical = 11,
};

/**
 * Read a positive integer from the environment.
 * @param name Variable name
 * @param def Value to use when unset or non-positive
 * @return Parsed value or def
 */
int EnvInt(const char *name, int def) {
  const char *e = std::getenv(name);
  if (!e || !*e) return def;
  int v = std::atoi(e);
  return v > 0 ? v : def;
}

/**
 * Test a boolean environment flag ("0" and empty count as false).
 * @param name Variable name
 * @return true when set to anything other than "0"
 */
bool EnvFlag(const char *name) {
  const char *e = std::getenv(name);
  return e && *e && std::strcmp(e, "0") != 0;
}

/**
 * Emit a rank-tagged line to stderr, flushed so interleaved MPI output is
 * still readable in a batch log.
 * @param rank MPI rank
 * @param msg Message
 */
void Log(int rank, const std::string &msg) {
  std::fprintf(stderr, "[xnode-gpu rank%d] %s\n", rank, msg.c_str());
  std::fflush(stderr);
}

/**
 * Elapsed milliseconds between two steady_clock stamps (project rule: all
 * timing output is in ms).
 * @param a Start
 * @param b End
 * @return Milliseconds
 */
double Ms(std::chrono::steady_clock::time_point a,
          std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

/**
 * Launch geometry: enough blocks to saturate, capped so tiny blobs stay cheap.
 * @param size Element count
 * @param blocks Out: grid size
 * @param threads Out: block size
 */
void FillGeometry(unsigned long long size, unsigned int *blocks,
                  unsigned int *threads) {
  *threads = 256;
  unsigned long long want = (size + *threads - 1) / *threads;
  *blocks = static_cast<unsigned int>(want > 1024ull ? 1024ull : want);
  if (*blocks == 0) *blocks = 1;
}

/** Device-side scratch shared by every phase on this rank. */
struct GpuScratch {
  unsigned char *dev_buf = nullptr;      /**< payload staging buffer */
  unsigned int *dev_mismatch = nullptr;  /**< verify kernel accumulator */
};

/**
 * Run the verify kernel over device memory and return the mismatch count.
 * @param g Device scratch (buffer already holds the bytes to check)
 * @param size Bytes to check
 * @param seed Expected producer seed
 * @return Number of mismatching bytes (0 == clean)
 */
unsigned int VerifyOnGpu(const GpuScratch &g, unsigned long long size,
                         unsigned int seed) {
  unsigned int zero = 0;
  ctp::GpuApi::Memcpy(g.dev_mismatch, &zero, sizeof(unsigned int));
  unsigned int blocks = 0, threads = 0;
  FillGeometry(size, &blocks, &threads);
  XnVerifyKernel<<<blocks, threads>>>(g.dev_buf, size, seed, g.dev_mismatch);
  ctp::GpuApi::Synchronize();
  unsigned int out = 0;
  ctp::GpuApi::Memcpy(&out, g.dev_mismatch, sizeof(unsigned int));
  return out;
}

/** Fill the device buffer with this rank's pattern. */
void FillOnGpu(const GpuScratch &g, unsigned long long size,
               unsigned int seed) {
  unsigned int blocks = 0, threads = 0;
  FillGeometry(size, &blocks, &threads);
  XnFillKernel<<<blocks, threads>>>(g.dev_buf, size, seed);
  ctp::GpuApi::Synchronize();
}

/**
 * PutBlob whose bytes were just produced on the GPU: D2H into a buffer the
 * runtime owns, then hand that buffer to CTE.
 * @param core CTE core client
 * @param ipc This rank's IPC manager
 * @param tag_id Shared tag
 * @param blob_name Blob to write
 * @param g Device scratch holding the payload
 * @param size Bytes
 * @param pq Routing for the put
 * @return CTE return code (0 == success)
 */
clio::run::u32 PutFromGpu(clio::cte::core::Client *core,
                          clio::run::IpcManager *ipc,
                          const clio::cte::core::TagId &tag_id,
                          const std::string &blob_name, const GpuScratch &g,
                          unsigned long long size,
                          const clio::run::PoolQuery &pq) {
  ctp::ipc::FullPtr<char> wbuf = ipc->AllocateBuffer(size);
  ctp::GpuApi::Memcpy(reinterpret_cast<unsigned char *>(wbuf.ptr_), g.dev_buf,
                      size);
  auto pb = core->AsyncPutBlob(tag_id, blob_name, 0, size,
                               wbuf.shm_.template Cast<void>(), -1.0f,
                               clio::cte::core::Context(), 0u, pq);
  pb.Wait();
  clio::run::u32 rc = pb->GetReturnCode();
  ipc->FreeBuffer(wbuf);
  return rc;
}

/**
 * GetBlob straight into GPU memory: CTE fills the runtime buffer, we H2D it
 * into the device scratch so the verify kernel can read it.
 * @param core CTE core client
 * @param ipc This rank's IPC manager
 * @param tag_id Shared tag
 * @param blob_name Blob to read
 * @param g Device scratch to receive the payload
 * @param size Bytes
 * @param pq Routing for the get
 * @return CTE return code (0 == success)
 */
clio::run::u32 GetToGpu(clio::cte::core::Client *core,
                        clio::run::IpcManager *ipc,
                        const clio::cte::core::TagId &tag_id,
                        const std::string &blob_name, const GpuScratch &g,
                        unsigned long long size,
                        const clio::run::PoolQuery &pq) {
  ctp::ipc::FullPtr<char> rbuf = ipc->AllocateBuffer(size);
  std::memset(rbuf.ptr_, 0, size);
  auto gb = core->AsyncGetBlob(tag_id, blob_name, 0, size, 0u,
                               rbuf.shm_.template Cast<void>(), pq);
  gb.Wait();
  clio::run::u32 rc = gb->GetReturnCode();
  if (rc == 0) {
    // Clear the device buffer first so a short read cannot pass by leaving
    // the previous (correct) contents in place.
    ctp::GpuApi::Memset(g.dev_buf, 0, size);
    ctp::GpuApi::Memcpy(g.dev_buf,
                        reinterpret_cast<const unsigned char *>(rbuf.ptr_),
                        size);
  }
  ipc->FreeBuffer(rbuf);
  return rc;
}

}  // namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  const unsigned long long kBlobSize =
      static_cast<unsigned long long>(EnvInt("XNODE_BLOB_MB", 1)) << 20;
  const bool force_physical = EnvFlag("XNODE_FORCE_PHYSICAL");

  // ---- Pin this rank to a GPU. One rank per node is the intended layout,
  //      but split by shared memory so a >1-rank-per-node run still spreads
  //      across the node's devices instead of piling onto device 0.
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shmcomm);
  int local_rank = 0;
  MPI_Comm_rank(shmcomm, &local_rank);
  MPI_Comm_free(&shmcomm);
  int ndev = ctp::GpuApi::GetDeviceCount();
  if (ndev <= 0) {
    Log(rank, "FAIL: no GPU visible on this node");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  ctp::GpuApi::SetDevice(local_rank % ndev);

  // ---- Attach to THIS node's daemon. Never spawn a runtime: the cluster is
  //      already up (one daemon per node, shared hostfile).
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  int local_rc = kOk;
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    Log(rank, "FAIL: CLIO_INIT(kClient) — is the local daemon running?");
    local_rc = kRcAttach;
  }

  clio::cte::core::Client core;
  clio::run::IpcManager *ipc = nullptr;
  clio::cte::core::TagId tag_id = clio::cte::core::TagId::GetNull();
  std::uint64_t my_node = 0;
  std::vector<std::uint64_t> node_of(nranks, 0);

  if (local_rc == kOk) {
    core.Init(clio::run::PoolId(kCorePoolMajor, 0));
    ipc = CLIO_CPU_IPC;
    // Informational only. A CLIENT does not load the hostfile unless
    // CLIO_CLIENT_TRY_NEW_SERVERS is set (see IpcManager::ClientInit), so
    // GetNodeId()/GetNumHosts() normally report 0/0 here even on a healthy
    // 8-node cluster. The cross-node precondition is therefore taken from the
    // MPI processor names below, which are ground truth about where each rank
    // actually runs, rather than from a server-side notion the client lacks.
    my_node = ipc->GetNodeId();
  }
  MPI_Allgather(&my_node, 1, MPI_UINT64_T, node_of.data(), 1, MPI_UINT64_T,
                MPI_COMM_WORLD);

  // ---- Topology: prove the ranks really are on distinct hosts.
  char myhost[MPI_MAX_PROCESSOR_NAME];
  std::memset(myhost, 0, sizeof(myhost));
  int hostlen = 0;
  MPI_Get_processor_name(myhost, &hostlen);
  std::vector<char> allhosts(static_cast<size_t>(nranks) *
                             MPI_MAX_PROCESSOR_NAME);
  MPI_Allgather(myhost, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allhosts.data(),
                MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
  // Short host name for a rank (domain stripped) — used in every phase
  // message so the log shows where work actually ran. The Clio node id is
  // useless for this: it is 0 on every client.
  auto host_of = [&allhosts](int r) {
    std::string h(&allhosts[static_cast<size_t>(r) * MPI_MAX_PROCESSOR_NAME]);
    size_t dot = h.find('.');
    return dot == std::string::npos ? h : h.substr(0, dot);
  };
  int distinct_hosts = 0;
  for (int i = 0; i < nranks; ++i) {
    const char *hi = &allhosts[static_cast<size_t>(i) * MPI_MAX_PROCESSOR_NAME];
    bool first = true;
    for (int j = 0; j < i; ++j) {
      const char *hj =
          &allhosts[static_cast<size_t>(j) * MPI_MAX_PROCESSOR_NAME];
      if (std::strcmp(hi, hj) == 0) first = false;
    }
    if (first) ++distinct_hosts;
  }
  if (rank == 0) {
    std::string map = "topology: nranks=" + std::to_string(nranks) +
                      " distinct hosts=" + std::to_string(distinct_hosts) +
                      " rank->host(clio node id)";
    for (int i = 0; i < nranks; ++i) {
      map += " " + std::to_string(i) + ":" +
             std::string(&allhosts[static_cast<size_t>(i) *
                                   MPI_MAX_PROCESSOR_NAME]) +
             "(" + std::to_string(node_of[i]) + ")";
    }
    Log(0, map);
  }
  if (distinct_hosts < nranks && local_rc == kOk) {
    Log(rank, "FAIL: ranks share hosts — " + std::to_string(distinct_hosts) +
                  " distinct hosts for " + std::to_string(nranks) +
                  " ranks; nothing would cross a node boundary");
    local_rc = kRcTopology;
  }

  // ---- Shared tag: rank 0 creates it, everyone else takes the handle. A
  //      GetOrCreateTag race across N ranks would be legal but noisy; this
  //      keeps the tag id unambiguous.
  std::uint64_t tag_u64 = 0;
  if (rank == 0 && local_rc == kOk) {
    auto mk = core.AsyncGetOrCreateTag(kTagName,
                                       clio::cte::core::TagId::GetNull(),
                                       clio::run::PoolQuery::Dynamic());
    mk.Wait();
    if (mk->GetReturnCode() != 0 || mk->tag_id_.IsNull()) {
      Log(0, "FAIL: GetOrCreateTag rc=" +
                 std::to_string(mk->GetReturnCode()));
      local_rc = kRcTag;
    } else {
      tag_u64 = mk->tag_id_.ToU64();
    }
  }
  MPI_Bcast(&tag_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (tag_u64 == 0 && local_rc == kOk) local_rc = kRcTag;
  if (local_rc == kOk) tag_id = clio::cte::core::TagId::FromU64(tag_u64);

  // ---- GPU scratch.
  GpuScratch g;
  if (local_rc == kOk) {
    g.dev_buf = ctp::GpuApi::Malloc<unsigned char>(kBlobSize);
    g.dev_mismatch = ctp::GpuApi::Malloc<unsigned int>(sizeof(unsigned int));
    if (g.dev_buf == nullptr || g.dev_mismatch == nullptr) {
      Log(rank, "FAIL: device allocation");
      local_rc = kRcAttach;
    }
  }

  const unsigned int my_seed = SeedForRank(rank);
  const std::string my_blob = "xn_b" + std::to_string(rank);
  const int peer = (rank + nranks - 1) % nranks;  // producer we consume from
  const std::string peer_blob = "xn_b" + std::to_string(peer);

  // ---- Produce the payload on the GPU and confirm the kernel itself is sane
  //      before any of it is attributed to the transport.
  if (local_rc == kOk) {
    FillOnGpu(g, kBlobSize, my_seed);
    if (VerifyOnGpu(g, kBlobSize, my_seed) != 0) {
      Log(rank, "FAIL: fill kernel did not produce its own pattern");
      local_rc = kRcLocalVerify;
    }
  }

  // ---- Phase 1: node-local round trip, all ranks at once. ------------------
  if (local_rc == kOk) {
    const std::string local_blob = my_blob + "L";
    auto t0 = std::chrono::steady_clock::now();
    clio::run::u32 rc = PutFromGpu(&core, ipc, tag_id, local_blob, g,
                                   kBlobSize,
                                   clio::run::PoolQuery::Dynamic());
    auto t1 = std::chrono::steady_clock::now();
    if (rc != 0) {
      Log(rank, "FAIL: phase1 PutBlob rc=" + std::to_string(rc));
      local_rc = kRcLocalPut;
    } else {
      rc = GetToGpu(&core, ipc, tag_id, local_blob, g, kBlobSize,
                    clio::run::PoolQuery::Dynamic());
      auto t2 = std::chrono::steady_clock::now();
      if (rc != 0) {
        Log(rank, "FAIL: phase1 GetBlob rc=" + std::to_string(rc));
        local_rc = kRcLocalGet;
      } else {
        unsigned int bad = VerifyOnGpu(g, kBlobSize, my_seed);
        if (bad != 0) {
          Log(rank, "FAIL: phase1 verify — " + std::to_string(bad) +
                        " mismatching bytes");
          local_rc = kRcLocalVerify;
        } else {
          char buf[192];
          std::snprintf(buf, sizeof(buf),
                        "phase1 local round trip OK: put=%.1f ms get=%.1f ms "
                        "(%llu bytes on %s)",
                        Ms(t0, t1), Ms(t1, t2), kBlobSize,
                        host_of(rank).c_str());
          Log(rank, buf);
        }
      }
    }
  }

  // ---- Phase 2: cross-node ring exchange. ----------------------------------
  // Publish first, barrier, then read the PEER's blob. Every rank reaches the
  // barrier regardless of its own status so one failure cannot hang the job;
  // ranks that already failed simply skip the work.
  if (local_rc == kOk) {
    // Re-fill: phase 1's GetBlob overwrote the device buffer (with the same
    // bytes, but do not depend on that).
    FillOnGpu(g, kBlobSize, my_seed);
    clio::run::u32 rc = PutFromGpu(&core, ipc, tag_id, my_blob, g, kBlobSize,
                                   clio::run::PoolQuery::Dynamic());
    if (rc != 0) {
      Log(rank, "FAIL: phase2 PutBlob rc=" + std::to_string(rc));
      local_rc = kRcRemotePut;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (local_rc == kOk) {
    auto t0 = std::chrono::steady_clock::now();
    clio::run::u32 rc = GetToGpu(&core, ipc, tag_id, peer_blob, g, kBlobSize,
                                 clio::run::PoolQuery::Dynamic());
    auto t1 = std::chrono::steady_clock::now();
    if (rc != 0) {
      Log(rank, "FAIL: phase2 cross-node GetBlob of " + peer_blob + " rc=" +
                    std::to_string(rc) + " (produced by rank " +
                    std::to_string(peer) + " on " + host_of(peer) + ")");
      local_rc = kRcRemoteGet;
    } else {
      unsigned int bad = VerifyOnGpu(g, kBlobSize, SeedForRank(peer));
      if (bad != 0) {
        Log(rank, "FAIL: phase2 verify of " + peer_blob + " — " +
                      std::to_string(bad) + " mismatching bytes");
        local_rc = kRcRemoteVerify;
      } else {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "phase2 cross-node OK: read %s — produced by rank %d on "
                      "%s, consumed here on %s in %.1f ms, verified on GPU",
                      peer_blob.c_str(), peer, host_of(peer).c_str(),
                      host_of(rank).c_str(), Ms(t0, t1));
        Log(rank, buf);
      }
    }
  }

  // ---- Phase 3 (opt-in): pin the remote hop to a named node. ---------------
  // Needs real Clio node ids, which a client only has when the hostfile was
  // loaded (CLIO_CLIENT_TRY_NEW_SERVERS>0). Skip loudly rather than routing
  // every rank's put to node 0 and calling the result a cross-node test.
  bool node_ids_usable = false;
  for (int i = 1; i < nranks; ++i) {
    if (node_of[i] != node_of[0]) node_ids_usable = true;
  }
  if (force_physical && !node_ids_usable && rank == 0) {
    Log(0,
        "SKIP phase3: every rank reports Clio node id " +
            std::to_string(node_of[0]) +
            " — set CLIO_CLIENT_TRY_NEW_SERVERS=1 so the client loads the "
            "hostfile, otherwise Physical() routing has nothing to aim at");
  }
  if (force_physical && node_ids_usable && local_rc == kOk) {
    const int next = (rank + 1) % nranks;
    const auto target = clio::run::PoolQuery::Physical(
        static_cast<clio::run::u32>(node_of[next]));
    const std::string phys_blob = my_blob + "P";
    FillOnGpu(g, kBlobSize, my_seed);
    clio::run::u32 rc =
        PutFromGpu(&core, ipc, tag_id, phys_blob, g, kBlobSize, target);
    if (rc != 0) {
      Log(rank, "FAIL: phase3 Physical PutBlob to node " +
                    std::to_string(node_of[next]) + " rc=" +
                    std::to_string(rc));
      local_rc = kRcPhysical;
    } else {
      rc = GetToGpu(&core, ipc, tag_id, phys_blob, g, kBlobSize, target);
      if (rc != 0 || VerifyOnGpu(g, kBlobSize, my_seed) != 0) {
        Log(rank, "FAIL: phase3 Physical round trip via node " +
                      std::to_string(node_of[next]));
        local_rc = kRcPhysical;
      } else {
        Log(rank, "phase3 Physical round trip via node " +
                      std::to_string(node_of[next]) + " OK");
      }
    }
  }

  if (g.dev_buf) ctp::GpuApi::Free(g.dev_buf);
  if (g.dev_mismatch) ctp::GpuApi::Free(g.dev_mismatch);

  // ---- Reduce: the job passes only if every rank passed. -------------------
  int global_rc = 0;
  MPI_Allreduce(&local_rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    if (global_rc == 0) {
      Log(0, "PASS: " + std::to_string(nranks) +
                 " ranks / GPUs completed the local + cross-node exchange (" +
                 std::to_string(kBlobSize >> 20) + " MiB each)");
    } else {
      Log(0, "FAIL: worst rank exit code " + std::to_string(global_rc) +
                 " (see per-rank lines above)");
    }
  }
  MPI_Finalize();
  return global_rc;
}

#endif  // !CTP_IS_DEVICE_PASS

#else

#include <cstdio>
int main() {
  std::fprintf(stderr,
               "test_cte_xnode_gpu: built without CUDA/ROCm — nothing to do\n");
  return 0;
}

#endif  // (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL
