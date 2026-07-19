/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

/**
 * GPU mem-bdev async-vs-sync copy benchmark.
 *
 * Drives *device* buffers through the bdev client so the mem transport takes
 * its GPU copy path (`ctp::IsDevicePointer(data)` is true), then measures how
 * many concurrent write tasks overlap. The change under test replaces
 * "worker blocks on cudaStreamSynchronize" with "worker yields and polls the
 * stream", so many in-flight D2H copies overlap instead of serializing one per
 * worker.
 *
 *   async (default) : env CLIO_BDEV_FORCE_SYNC unset -> yield-poll path
 *   sync  (old)     : env CLIO_BDEV_FORCE_SYNC=1     -> Synchronize per task
 *
 * Run the same binary both ways and diff (see run_gpu_bdev_bench.sh).
 *
 * How a device pointer reaches the transport without GPU-backend registration:
 * a ShmPtr with a NULL alloc_id carries the raw address in `off_`, and the
 * worker's IpcManager::ToFullPtr resolves that directly (Case 1). So a plain
 * cudaMalloc buffer + a hand-built ShmPtr is enough to exercise the GPU path.
 *
 * Metrics, per concurrency depth:
 *   - aggregate bandwidth (GB/s) and IOPS over a fixed batch of writes
 *   - avg per-op latency
 *   - total wall-time for the batch (the "100 x 39 MB" headline)
 */

#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM) && !CTP_ENABLE_SYCL

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>

namespace {

using clio::run::u64;
using Block = clio::run::bdev::Block;

struct Args {
  size_t io_size = 39ull * 1024 * 1024;  // bytes per write (user scenario: 39 MB)
  size_t num_ops = 200;                  // writes per depth (the fixed batch)
  std::vector<size_t> depths = {1, 4, 16, 64};  // outstanding writes per wave
  size_t tier_bytes = 24ull << 30;       // bdev capacity (lazily pinned)
};

bool ParseSizeArg(const char *s, size_t &out) {
  char *end = nullptr;
  double v = std::strtod(s, &end);
  if (end == s) return false;
  size_t mult = 1;
  if (*end == 'k' || *end == 'K') mult = 1ull << 10;
  else if (*end == 'm' || *end == 'M') mult = 1ull << 20;
  else if (*end == 'g' || *end == 'G') mult = 1ull << 30;
  out = static_cast<size_t>(v * mult);
  return out > 0;
}

bool ParseDepths(const char *csv, std::vector<size_t> &out) {
  out.clear();
  std::string s = csv;
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string tok = s.substr(start, comma - start);
    if (!tok.empty()) out.push_back(std::stoull(tok));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return !out.empty();
}

bool Parse(int argc, char **argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--io-size" && i + 1 < argc) {
      if (!ParseSizeArg(argv[++i], a.io_size)) return false;
    } else if (s == "--num-ops" && i + 1 < argc) {
      a.num_ops = std::stoull(argv[++i]);
    } else if (s == "--depths" && i + 1 < argc) {
      if (!ParseDepths(argv[++i], a.depths)) return false;
    } else if (s == "--tier-bytes" && i + 1 < argc) {
      if (!ParseSizeArg(argv[++i], a.tier_bytes)) return false;
    } else if (s == "--help" || s == "-h") {
      std::printf(
          "Usage: %s [--io-size 39M] [--num-ops 200] [--depths 1,4,16,64] "
          "[--tier-bytes 24G]\n"
          "Set env CLIO_BDEV_FORCE_SYNC=1 for the synchronous (old) path.\n",
          argv[0]);
      return false;
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", s.c_str());
      return false;
    }
  }
  return true;
}

// A ShmPtr whose NULL alloc_id makes ToFullPtr treat off_ as the raw address.
ctp::ipc::ShmPtr<> DeviceShm(void *dev) {
  ctp::ipc::ShmPtr<> s;
  s.alloc_id_.SetNull();
  s.off_ = reinterpret_cast<u64>(dev);
  return s;
}

}  // namespace

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  Args a;
  if (!Parse(argc, argv, a)) return 1;

  // In-process co-located runtime so the benchmark is self-contained.
  if (!std::getenv("CLIO_BIND_ADDR")) setenv("CLIO_BIND_ADDR", "127.0.0.1", 0);
  const bool force_sync = std::getenv("CLIO_BDEV_FORCE_SYNC") != nullptr;

  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kServer)) {
    std::fprintf(stderr, "ERROR: CLIO_INIT(kServer) failed\n");
    return 1;
  }

  const size_t max_depth =
      *std::max_element(a.depths.begin(), a.depths.end());

  std::printf("=== GPU mem-bdev copy benchmark ===\n");
  std::printf("mode        : %s\n", force_sync ? "SYNC (old, worker blocks)"
                                               : "ASYNC (yield-poll)");
  std::printf("io-size     : %.2f MB\n", a.io_size / (1024.0 * 1024.0));
  std::printf("num-ops     : %zu per depth\n", a.num_ops);
  std::printf("max-depth   : %zu\n", max_depth);
  std::printf("tier        : %.1f GB (kPinned)\n",
              a.tier_bytes / (1024.0 * 1024.0 * 1024.0));

  // Create a kPinned bdev pool (pinned host pages -> async copies truly
  // overlap; pageable pages would silently serialize).
  clio::run::PoolId pool_id(7100, 0);
  clio::run::bdev::Client bdev(pool_id);
  {
    auto ct = bdev.AsyncCreate(clio::run::PoolQuery::Broadcast(),
                               "gpu_bdev_bench", pool_id,
                               clio::run::bdev::BdevType::kPinned, a.tier_bytes,
                               /*io_depth=*/32, /*alignment=*/4096);
    ct.Wait();
    if (ct->GetReturnCode() != 0) {
      std::fprintf(stderr, "ERROR: create pool rc=%d\n", ct->GetReturnCode());
      return 1;
    }
    bdev.pool_id_ = ct->new_pool_id_;
  }

  // Pre-allocate max_depth device source buffers + distinct bdev block regions
  // so concurrent in-flight writes never alias. Reused across every wave.
  std::vector<char *> dev_bufs(max_depth, nullptr);
  std::vector<clio::run::priv::vector<Block>> regions;
  regions.reserve(max_depth);
  for (size_t i = 0; i < max_depth; ++i) {
    dev_bufs[i] = ctp::GpuApi::Malloc<char>(a.io_size);
    if (!dev_bufs[i]) {
      std::fprintf(stderr, "ERROR: cudaMalloc slot %zu failed\n", i);
      return 1;
    }
    ctp::GpuApi::Memset(dev_bufs[i], static_cast<int>(i + 1), a.io_size);

    auto at = bdev.AsyncAllocateBlocks(clio::run::PoolQuery::Local(), a.io_size);
    at.Wait();
    if (at->GetReturnCode() != 0 || at->blocks_.empty()) {
      std::fprintf(stderr, "ERROR: allocate slot %zu rc=%d\n", i,
                   at->GetReturnCode());
      return 1;
    }
    clio::run::priv::vector<Block> blks(CLIO_PRIV_ALLOC);
    for (size_t b = 0; b < at->blocks_.size(); ++b) blks.push_back(at->blocks_[b]);
    regions.push_back(std::move(blks));
  }

  std::printf("\n%-8s %12s %14s %14s %14s\n", "depth", "batch_ms", "GB/s",
              "kIOPS", "us/op");
  std::printf("-------------------------------------------------------------\n");

  for (size_t depth : a.depths) {
    // Warm up one wave (page allocation / stream setup excluded from timing).
    {
      std::vector<clio::run::Future<clio::run::bdev::WriteTask>> f;
      for (size_t i = 0; i < depth; ++i)
        f.push_back(bdev.AsyncWrite(clio::run::PoolQuery::Local(), regions[i],
                                    DeviceShm(dev_bufs[i]), a.io_size));
      for (auto &x : f) x.Wait();
    }

    auto t0 = std::chrono::steady_clock::now();
    size_t done = 0;
    while (done < a.num_ops) {
      size_t wave = std::min(depth, a.num_ops - done);
      std::vector<clio::run::Future<clio::run::bdev::WriteTask>> futs;
      futs.reserve(wave);
      for (size_t i = 0; i < wave; ++i)
        futs.push_back(bdev.AsyncWrite(clio::run::PoolQuery::Local(),
                                       regions[i], DeviceShm(dev_bufs[i]),
                                       a.io_size));
      for (auto &x : futs) x.Wait();
      done += wave;
    }
    auto t1 = std::chrono::steady_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double gb = (double)a.num_ops * a.io_size / 1e9;
    double gbps = gb / sec;
    double kiops = (a.num_ops / sec) / 1e3;
    double us_op = sec * 1e6 / a.num_ops;
    std::printf("%-8zu %12.2f %14.2f %14.2f %14.2f\n", depth, sec * 1e3, gbps,
                kiops, us_op);
  }

  std::printf("\nDone (%s).\n", force_sync ? "SYNC" : "ASYNC");
  ::ctp::SystemInfo::TerminateProcessNow(0);
  return 0;
}

#else  // no GPU backend

#include <cstdio>
int main() {
  std::printf("clio_run_gpu_bdev_bench requires a CUDA or ROCm build.\n");
  return 0;
}

#endif
