/*
 * Cross-node bdev throughput benchmark.
 *
 * Bypasses CTE / FUSE / IOR. One client per MPI rank, each rank issues
 * bdev writes targeted at a SPECIFIC peer container so every task
 * crosses the network deterministically:
 *
 *   rank on node N  ->  DirectHash((N + 1) % num_nodes)
 *
 * i.e. node 0 writes into bdev container 1 (on node 1), node 1 into
 * container 2, ... node N-1 into container 0 — a perfect ring of
 * cross-node forwards. No local short-circuit, no FUSE plumbing, no
 * blob hashing through clio_cte_core. Pure stress of:
 *   client::AsyncWrite  ->  net_queue  ->  SendIn  ->  ZMQ DEALER
 *   ZMQ ROUTER  ->  RecvIn  ->  worker dispatch  ->  bdev WriteToRam
 *   bdev returns  ->  SendOut  ->  ROUTER -> client wake
 *
 * Compose-side: each node owns one bdev container at hash N (so we
 * already have a bdev container per node and the ring routing
 * trivially matches a single neighbor).
 *
 * Output: aggregate MiB/s + per-rank ops + per-rank wall time.
 */
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <clio_ctp/util/logging.h>
#include <mpi.h>

#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"

namespace {

struct Args {
  size_t io_size = 1 << 20;     // bytes per write (default 1 MiB)
  size_t num_ops = 256;         // writes per rank
  size_t depth = 16;            // outstanding writes per rank (pipeline depth)
  std::string pool_name = "bench_bdev";
  clio::run::PoolId pool_id{777, 0};  // unique id reserved for this bench
  size_t tier_bytes = 4ULL << 30;  // 4 GiB per node — fits cluster total in node-local RAM
  bool create_pool = true;      // first rank creates, others find
};

bool Parse(int argc, char **argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto take = [&](const char *key, auto &slot) {
      if (s == key && i + 1 < argc) {
        slot = std::stoull(argv[++i]);
        return true;
      }
      return false;
    };
    if (take("--io-size", a.io_size)) continue;
    if (take("--num-ops", a.num_ops)) continue;
    if (take("--depth", a.depth)) continue;
    if (s == "--no-create") {
      a.create_pool = false;
      continue;
    }
    if (s == "--help") {
      std::cerr << "usage: clio_xnode_bdev_bench --io-size <bytes> "
                   "--num-ops <N> --depth <D> [--no-create]\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  Args args;
  if (!Parse(argc, argv, args)) {
    MPI_Finalize();
    return 1;
  }

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "rank={} CLIO_INIT failed", world_rank);
    MPI_Finalize();
    return 2;
  }

  auto *ipc = CLIO_IPC;
  const clio::run::u64 num_nodes = ipc->GetNumHosts();
  const clio::run::u64 my_node = ipc->GetNodeId();
  // Ring pattern: every rank's writes target the NEXT node's bdev
  // container. With one bdev container per node (DirectHash(i) -> node i)
  // this guarantees every task crosses the network.
  const clio::run::u32 target_container =
      static_cast<clio::run::u32>((my_node + 1) % num_nodes);

  HLOG(kInfo,
       "rank={}/{} node={} num_nodes={} target_container={} "
       "io_size={} num_ops={} depth={}",
       world_rank, world_size, my_node, num_nodes, target_container,
       args.io_size, args.num_ops, args.depth);

  // -------- Pool setup --------
  // Rank 0 (globally) creates the pool with N containers (one per node).
  // Everyone else waits on MPI_Barrier and reuses the same pool_id.
  clio::run::bdev::Client bdev_client;
  if (world_rank == 0 && args.create_pool) {
    clio::run::admin::Client admin_client;
    clio::run::bdev::CreateParams params;
    params.bdev_type_ = clio::run::bdev::BdevType::kRam;
    params.total_size_ = args.tier_bytes;
    params.alignment_ = 4096;

    auto create_task = bdev_client.AsyncCreate(
        clio::run::PoolQuery::Broadcast(), args.pool_name, args.pool_id,
        clio::run::bdev::BdevType::kRam, args.tier_bytes);
    create_task.Wait();
    if (create_task->return_code_ != 0) {
      HLOG(kError, "AsyncCreate failed rc={}",
           create_task->return_code_.load());
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
    bdev_client.pool_id_ = create_task->new_pool_id_;
    HLOG(kInfo, "Pool created: ({},{}) tier={} GiB",
         bdev_client.pool_id_.major_, bdev_client.pool_id_.minor_,
         args.tier_bytes >> 30);
  } else {
    bdev_client.pool_id_ = args.pool_id;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // -------- Buffer setup --------
  auto write_buf = CLIO_IPC->AllocateBuffer(args.io_size);
  if (write_buf.IsNull()) {
    HLOG(kError, "rank={} AllocateBuffer failed", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }
  std::memset(write_buf.ptr_, static_cast<int>(world_rank & 0xff),
              args.io_size);

  // Pre-build a Block list that points at "the whole io_size on the
  // peer's bdev". We don't go through AllocateBlocks (that uses the
  // bdev's heap allocator, which is sequential per-rank-on-same-node);
  // we just synthesize Block{offset_, size_} the same shape AsyncWrite
  // accepts, with offset varying per rank so writes don't trample.
  // bdev WriteToRam paths use the offset to index into ram_pages_.
  const clio::run::u64 rank_offset =
      static_cast<clio::run::u64>(world_rank) * args.num_ops * args.io_size;

  // -------- Pipeline --------
  // Each rank keeps `depth` outstanding AsyncWrites at a time. Wait on
  // the oldest before issuing a new one. This is how you actually
  // saturate a cross-node DEALER without the client-side Wait-after-
  // every-send anti-pattern we see in the FUSE adapter.
  std::vector<clio::run::Future<clio::run::bdev::WriteTask>> inflight;
  inflight.reserve(args.depth);

  MPI_Barrier(MPI_COMM_WORLD);
  auto t_start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < args.num_ops; ++i) {
    clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
    clio::run::bdev::Block blk;
    blk.offset_ = rank_offset + i * args.io_size;
    blk.size_ = args.io_size;
    blocks.push_back(blk);

    auto fut = bdev_client.AsyncWrite(
        clio::run::PoolQuery::DirectHash(target_container), blocks,
        write_buf.shm_.template Cast<void>(), args.io_size);
    inflight.push_back(std::move(fut));

    if (inflight.size() >= args.depth) {
      inflight.front().Wait();
      if (inflight.front()->return_code_ != 0) {
        HLOG(kError, "rank={} write failed rc={}",
             world_rank, inflight.front()->return_code_.load());
        MPI_Abort(MPI_COMM_WORLD, 5);
      }
      inflight.erase(inflight.begin());
    }
  }
  // Drain the tail.
  for (auto &f : inflight) {
    f.Wait();
    if (f->return_code_ != 0) {
      HLOG(kError, "rank={} drain write failed rc={}",
           world_rank, f->return_code_.load());
      MPI_Abort(MPI_COMM_WORLD, 6);
    }
  }
  inflight.clear();

  auto t_end = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t_end - t_start).count();
  double per_rank_mibs =
      static_cast<double>(args.num_ops) * args.io_size / (1ULL << 20) / secs;
  double per_rank_iops = args.num_ops / secs;

  // Aggregate across ranks (sum bytes, max wall time)
  double total_bytes =
      static_cast<double>(world_size) * args.num_ops * args.io_size;
  double max_secs = secs;
  MPI_Allreduce(MPI_IN_PLACE, &max_secs, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  double cluster_mibs = total_bytes / (1ULL << 20) / max_secs;

  HLOG(kInfo,
       "rank={} per_rank: ops={} bytes={} t={:.3f}s "
       "rate={:.1f} MiB/s iops={:.1f}",
       world_rank, args.num_ops, args.num_ops * args.io_size, secs,
       per_rank_mibs, per_rank_iops);
  if (world_rank == 0) {
    HLOG(kInfo,
         "CLUSTER ranks={} nodes={} io_size={} num_ops_per_rank={} depth={} "
         "total_bytes={} GiB wall={:.3f}s aggregate={:.1f} MiB/s",
         world_size, num_nodes, args.io_size, args.num_ops, args.depth,
         total_bytes / (1ULL << 30), max_secs, cluster_mibs);
  }

  CLIO_IPC->FreeBuffer(write_buf);
  MPI_Finalize();
  return 0;
}
