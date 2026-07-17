#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <random>

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/safe_bdev/safe_bdev_client.h>
#include <clio_runtime/safe_bdev/safe_bdev_tasks.h>
#include <clio_runtime/admin/admin_client.h>

using namespace std::chrono_literals;

constexpr clio::run::u64 kMemberSize = 64ULL * 1024ULL * 1024ULL; // 64 MiB
constexpr clio::run::u64 kChunkLen = 4096;
constexpr clio::run::u64 kPoolSize = 32ULL * 1024ULL * 1024ULL; // 32 MiB logical

std::string member_file(int i) {
  return "mig_bench_member_" + std::to_string(i) + ".dat";
}

bool CreateFileMember(clio::run::bdev::Client &client, const std::string &path, clio::run::PoolId id) {
  FILE *f = fopen(path.c_str(), "w");
  if (f) {
    fclose(f);
  }
  auto fut = client.AsyncCreate(clio::run::PoolQuery::Dynamic(), "bdev_mem_" + std::to_string(id.major_), id,
                                clio::run::bdev::BdevType::kFile, kMemberSize);
  fut.Wait();
  return fut->return_code_ == 0;
}

std::atomic<bool> g_io_run{true};
std::atomic<uint64_t> g_bytes_rw{0};
std::atomic<uint64_t> g_io_ops{0};

void IOWorker(clio::run::PoolId safe_id) {
  clio::run::safe_bdev::Client safe(safe_id);
  std::vector<char> buf(kChunkLen, 'x');
  ctp::ipc::FullPtr<char> ipc_buf = CLIO_IPC->AllocateBuffer(kChunkLen);
  memcpy(ipc_buf.ptr_, buf.data(), kChunkLen);

  // use simple rand since mt19937 may need <random> included properly
  unsigned int seed = 42;
  while (g_io_run) {
    clio::run::u64 off = (rand_r(&seed) % (kPoolSize / kChunkLen)) * kChunkLen;
    clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
    blocks.push_back(clio::run::bdev::Block(off, kChunkLen, 0));

    auto w_fut = safe.AsyncWrite(clio::run::PoolQuery::Local(), 
                                 blocks,
                                 ipc_buf.shm_.Cast<void>(), kChunkLen);
    w_fut.Wait();
    
    if (w_fut->return_code_ == 0) {
      g_bytes_rw += kChunkLen;
      g_io_ops++;
    }

    auto r_fut = safe.AsyncRead(clio::run::PoolQuery::Local(), 
                                blocks,
                                ipc_buf.shm_.Cast<void>(), kChunkLen);
    r_fut.Wait();

    if (r_fut->return_code_ == 0) {
      g_bytes_rw += kChunkLen;
      g_io_ops++;
    }
  }
  CLIO_IPC->FreeBuffer(ipc_buf);
}

int main(int argc, char** argv) {
  bool graceful = false;
  if (argc > 1 && std::string(argv[1]) == "--graceful") {
    graceful = true;
  }
  
  std::cout << "Starting Safe-BDEV Migration Benchmark (Mode: " 
            << (graceful ? "Graceful Migration" : "Hard Failure Recovery") << ")\n";

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    std::cerr << "CLIO_INIT failed\n";
    return 1;
  }

  // Create 5 bdev members
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < 5; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(20000 + c), 0);
    clio::run::bdev::Client client(id);
    if (!CreateFileMember(client, member_file(c), id)) {
      std::cerr << "Failed to create member " << c << "\n";
      return 1;
    }
    if (c < 3) {
      members.emplace_back(member_file(c), 0, client.pool_id_);
    }
  }

  clio::run::PoolId safe_id(21000, 0);
  clio::run::safe_bdev::Client safe(safe_id);
  auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(), "safe_bench_pool", safe_id, 1, members, "mig_bench_alog");
  create_task.Wait();
  if (create_task->return_code_ != 0) {
    std::cerr << "AsyncCreate failed\n";
    return 1;
  }
  safe.pool_id_ = create_task->new_pool_id_;

  // Add parity (member 3)
  auto p_task = safe.AsyncAddBdev(clio::run::PoolQuery::Local(), "safe_bench_pool", 0, clio::run::PoolId(20003, 0), 1);
  p_task.Wait();

  std::cout << "Array created. Starting background IO...\n";
  std::thread io_thread(IOWorker, safe_id);

  std::this_thread::sleep_for(2s);
  
  uint64_t bytes_before = g_bytes_rw.exchange(0);
  std::cout << "Baseline Throughput: " << (bytes_before / 2.0 / 1024.0 / 1024.0) << " MiB/s\n";

  std::cout << "Triggering Migration/Recovery...\n";
  auto start_time = std::chrono::steady_clock::now();
  
  if (graceful) {
    // Simulate graceful migration: paced copy from member 0 to member 4.
    clio::run::bdev::Client b0(clio::run::PoolId(20000, 0));
    clio::run::bdev::Client b4(clio::run::PoolId(20004, 0));
    ctp::ipc::FullPtr<char> tmp = CLIO_IPC->AllocateBuffer(kChunkLen);
    
    int chunks = (16 * 1024 * 1024) / kChunkLen;
    for (int i = 0; i < chunks; ++i) {
      clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
      blocks.push_back(clio::run::bdev::Block(static_cast<clio::run::u64>(i*kChunkLen), kChunkLen, 0));

      auto r = b0.AsyncRead(clio::run::PoolQuery::Local(), blocks, tmp.shm_.Cast<void>(), kChunkLen);
      r.Wait();
      auto w = b4.AsyncWrite(clio::run::PoolQuery::Local(), blocks, tmp.shm_.Cast<void>(), kChunkLen);
      w.Wait();
      std::this_thread::sleep_for(2s / chunks);
    }
    CLIO_IPC->FreeBuffer(tmp);
  } else {
    // Hard failure recovery: use the runtime's full-speed AsyncRecoverBdev
    auto r_task = safe.AsyncRecoverBdev(clio::run::PoolQuery::Local(), clio::run::PoolId(20000, 0), "safe_bench_pool", 0, clio::run::PoolId(20004, 0));
    r_task.Wait();
  }

  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;
  
  uint64_t bytes_during = g_bytes_rw.exchange(0);
  double mibs_during = (bytes_during / diff.count()) / 1024.0 / 1024.0;
  
  std::cout << "Migration/Recovery took " << diff.count() << " seconds.\n";
  std::cout << "Throughput During Event: " << mibs_during << " MiB/s\n";

  g_io_run = false;
  io_thread.join();

  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}
