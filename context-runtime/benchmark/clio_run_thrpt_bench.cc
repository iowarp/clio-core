/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Task Throughput and Latency Benchmark
 *
 * Benchmarks different aspects of the CLIO Runtime runtime:
 * - BDev I/O throughput (allocate/write/free)
 * - BDev allocation throughput (allocate/free only)
 * - Round-trip latency using MOD_NAME Custom function
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include <clio_ctp/util/logging.h>
#include <clio_ctp/introspect/system_info.h>

// Client fast-exit (repo pattern, mirrors SIMPLE_TEST_PROCESS_EXIT). On Windows
// TerminateProcessNow skips the static-destructor chain that aborts in libzmq's
// signaler ("Successful WSASTARTUP not yet performed") once this client has a
// ZMQ ROUTER response listener; on POSIX it is a no-op so the normal teardown
// (return below) runs. Flush first so results/output are not lost.
#define BENCH_EXIT(code)                                                       \
  do {                                                                         \
    std::cout.flush();                                                         \
    std::cerr.flush();                                                         \
    ::ctp::SystemInfo::TerminateProcessNow((code));                            \
    return (code);                                                             \
  } while (0)

#include "clio_runtime/MOD_NAME/MOD_NAME_client.h"
#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/bdev/bdev_client.h"
#include "clio_runtime/clio_runtime.h"

/**
 * Benchmark test cases
 */
enum class TestCase {
  kBDevIO,         // Full I/O (Allocate -> Write -> Free)
  kBDevAllocation, // Allocation only (Allocate -> Free)
  kBDevTaskAlloc,  // Task allocation/deletion (NewTask -> DelTask)
  kLatency,        // Round-trip latency using MOD_NAME Custom
  kSchedVariety    // issue #781: mixed 1us..1s compute; p99 by class (starvation)
};

/**
 * Benchmark configuration
 */
struct BenchmarkConfig {
  TestCase test_case = TestCase::kBDevIO; // Test case to run
  size_t num_threads = 4;                 // Number of client threads
  double duration_seconds = 10.0;         // Duration to run benchmark (seconds)
  size_t max_file_size = 1ULL << 30;      // Maximum file size (default: 1GB)
  size_t io_size = 4096; // I/O size per operation (default: 4KB)
  bool verbose = false;  // Print detailed output
  std::string lane_policy =
      ""; // Lane mapping policy override (empty = use config)
  std::string output_dir =
      "/tmp/clio_benchmark"; // Output directory for benchmark files
  // Attach to an already-composed block-device pool instead of creating a
  // private bdev. safe_bdev reuses bdev's data-plane task types and method
  // ids (10..14), so the bdev client drives either module unchanged -- only
  // the target pool differs. Empty => create a private bdev (default).
  bool attach_pool = false;
  clio::run::PoolId attach_pool_id;
};

/**
 * Parse a "major.minor" pool id (the same form compose files use, e.g.
 * "7000.0"). Returns false on malformed input.
 */
bool ParsePoolId(const std::string &str, clio::run::PoolId &pool_id) {
  size_t dot = str.find('.');
  if (dot == std::string::npos || dot == 0 || dot + 1 >= str.size()) {
    return false;
  }
  try {
    clio::run::u32 major = static_cast<clio::run::u32>(
        std::stoul(str.substr(0, dot)));
    clio::run::u32 minor = static_cast<clio::run::u32>(
        std::stoul(str.substr(dot + 1)));
    pool_id = clio::run::PoolId(major, minor);
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

/**
 * Parse test case from string
 */
bool ParseTestCase(const std::string &str, TestCase &test_case) {
  if (str == "bdev_io") {
    test_case = TestCase::kBDevIO;
    return true;
  } else if (str == "bdev_allocation") {
    test_case = TestCase::kBDevAllocation;
    return true;
  } else if (str == "bdev_task_alloc") {
    test_case = TestCase::kBDevTaskAlloc;
    return true;
  } else if (str == "latency") {
    test_case = TestCase::kLatency;
    return true;
  } else if (str == "sched_variety") {
    test_case = TestCase::kSchedVariety;
    return true;
  }
  return false;
}

/**
 * Parse command line arguments
 */
bool ParseArgs(int argc, char **argv, BenchmarkConfig &config) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--test-case" && i + 1 < argc) {
      if (!ParseTestCase(argv[++i], config.test_case)) {
        HLOG(kError, "ERROR: Invalid test case. Valid options: bdev_io, bdev_allocation, bdev_task_alloc, latency, sched_variety");
        return false;
      }
    } else if (arg == "--threads" && i + 1 < argc) {
      config.num_threads = std::stoull(argv[++i]);
    } else if (arg == "--duration" && i + 1 < argc) {
      config.duration_seconds = std::stod(argv[++i]);
    } else if (arg == "--max-file-size" && i + 1 < argc) {
      config.max_file_size = ctp::ConfigParse::ParseSize(argv[++i]);
    } else if (arg == "--io-size" && i + 1 < argc) {
      config.io_size = ctp::ConfigParse::ParseSize(argv[++i]);
    } else if (arg == "--lane-policy" && i + 1 < argc) {
      config.lane_policy = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else if (arg == "--pool-id" && i + 1 < argc) {
      if (!ParsePoolId(argv[++i], config.attach_pool_id)) {
        HLOG(kError, "ERROR: --pool-id expects <major>.<minor> (e.g. 7000.0)");
        return false;
      }
      config.attach_pool = true;
    } else if (arg == "--verbose" || arg == "-v") {
      config.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      HIPRINT("Usage: {} [options]", argv[0]);
      HIPRINT("Options:");
      HIPRINT("  --test-case <case>      Test case: bdev_io, bdev_allocation, bdev_task_alloc, latency, sched_variety (default: bdev_io)");
      HIPRINT("  --threads <N>           Number of client threads (default: 4)");
      HIPRINT("  --duration <seconds>    Duration to run benchmark in seconds (default: 10.0)");
      HIPRINT("  --max-file-size <size>  Maximum file size with suffix: k, m, g (default: 1g)");
      HIPRINT("  --io-size <size>        I/O size per operation with suffix: k, m, g (default: 4k)");
      HIPRINT("  --lane-policy <P>       Lane policy: map_by_pid_tid, round_robin, random (default: from config)");
      HIPRINT("  --output-dir <dir>      Output directory for benchmark files (default: /tmp/clio_benchmark)");
      HIPRINT("  --pool-id <major.minor> Attach to an existing block-device pool (e.g. a composed");
      HIPRINT("                          safe_bdev) instead of creating a private bdev. bdev-only tests.");
      HIPRINT("  --verbose, -v           Verbose output");
      HIPRINT("  --help, -h              Show this help");
      HIPRINT("");
      HIPRINT("Test Cases:");
      HIPRINT("  bdev_io          - BDev I/O throughput (Allocate -> Write -> Free)");
      HIPRINT("  bdev_allocation  - BDev allocation throughput (Allocate -> Free)");
      HIPRINT("  bdev_task_alloc  - BDev task allocation (NewTask -> DelTask)");
      HIPRINT("  latency          - Round-trip task latency using MOD_NAME Custom");
      return false;
    } else {
      HLOG(kError, "Unknown argument: {}", arg);
      return false;
    }
  }

  return true;
}

/**
 * Allocation-only worker thread function - benchmarks AllocateBlocks/FreeBlocks
 * Runs Allocate -> Free loop until stop flag is set (no I/O operations)
 */
void AllocationWorkerThread(size_t thread_id, const BenchmarkConfig &config,
                            clio::run::PoolId pool_id, std::atomic<bool> &stop_flag,
                            std::atomic<size_t> &completed_ops,
                            std::chrono::nanoseconds &elapsed_time) {
  // Create BDev client for this thread
  clio::run::bdev::Client bdev_client(pool_id);

  // Use io_size for allocation-only benchmark
  size_t alloc_size = config.io_size;
  HLOG(kInfo, "Allocate size: {}", alloc_size);

  size_t local_ops = 0;
  const size_t WARMUP_OPS = 5; // Ignore first 5 operations
  auto start_time = std::chrono::high_resolution_clock::now();

  // Continuously perform allocate/free operations until stop signal
  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Allocate blocks
    auto alloc_task = bdev_client.AsyncAllocateBlocks(clio::run::PoolQuery::Local(),
                                                       alloc_size);
    alloc_task.Wait();
    if (alloc_task->GetReturnCode() != 0 || alloc_task->blocks_.empty()) {
      HLOG(kError, "Thread {}: AllocateBlocks failed (rc={}, blocks={})",
           thread_id, alloc_task->GetReturnCode(), alloc_task->blocks_.size());
      stop_flag.store(true, std::memory_order_relaxed);
      return;
    }
    std::vector<clio::run::bdev::Block> blocks;
    for (size_t i = 0; i < alloc_task->blocks_.size(); ++i) {
      blocks.push_back(alloc_task->blocks_[i]);
    }

    // Free blocks immediately
    auto free_task = bdev_client.AsyncFreeBlocks(clio::run::PoolQuery::Local(), blocks);
    free_task.Wait();

    local_ops++;

    // Start timer after warmup operations
    if (local_ops == WARMUP_OPS) {
      start_time = std::chrono::high_resolution_clock::now();
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);

  // Update global counters
  completed_ops.fetch_add(local_ops, std::memory_order_relaxed);

  if (config.verbose) {
    double thread_throughput = (local_ops * 1e9) / elapsed_time.count();
    double avg_latency_us = elapsed_time.count() / (local_ops * 1e3);
    HIPRINT("Thread {}: {} alloc/free ops in {} ms, {} ops/sec, {} us/op",
            thread_id, local_ops, (elapsed_time.count() / 1e6), thread_throughput, avg_latency_us);
  }
}

/**
 * Task allocation worker thread - benchmarks NewTask/DelTask overhead
 * Creates AllocateBlocksTask and FreeBlocksTask, then immediately deletes them
 */
void TaskAllocationWorkerThread(size_t thread_id, const BenchmarkConfig &config,
                                clio::run::PoolId pool_id,
                                std::atomic<bool> &stop_flag,
                                std::atomic<size_t> &completed_ops,
                                std::chrono::nanoseconds &elapsed_time) {
  // Get IPC manager
  auto *ipc_manager = CLIO_IPC;

  // Use io_size for task allocation benchmark
  size_t alloc_size = config.io_size;

  // Create dummy blocks vector for FreeBlocksTask
  std::vector<clio::run::bdev::Block> dummy_blocks(2);
  dummy_blocks[0].offset_ = 0;
  dummy_blocks[0].size_ = 1024;
  dummy_blocks[0].block_type_ = 0;
  dummy_blocks[1].offset_ = 1024;
  dummy_blocks[1].size_ = 2048;
  dummy_blocks[1].block_type_ = 1;

  size_t local_ops = 0;
  const size_t WARMUP_OPS = 5; // Ignore first 5 operations
  auto start_time = std::chrono::high_resolution_clock::now();

  // Continuously perform task allocation/deletion until stop signal
  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Create and delete AllocateBlocksTask
    auto alloc_task = ipc_manager->NewTask<clio::run::bdev::AllocateBlocksTask>(
        clio::run::CreateTaskId(), pool_id, clio::run::PoolQuery::Local(), alloc_size);
    alloc_task.reset();

    // Create and delete FreeBlocksTask
    auto free_task = ipc_manager->NewTask<clio::run::bdev::FreeBlocksTask>(
        clio::run::CreateTaskId(), pool_id, clio::run::PoolQuery::Local(), dummy_blocks);
    free_task.reset();

    local_ops += 2; // Count both allocate and free task creations

    // Start timer after warmup operations
    if (local_ops == WARMUP_OPS * 2) {
      start_time = std::chrono::high_resolution_clock::now();
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);

  // Update global counters
  completed_ops.fetch_add(local_ops, std::memory_order_relaxed);

  if (config.verbose) {
    double thread_throughput = (local_ops * 1e9) / elapsed_time.count();
    double avg_latency_us = elapsed_time.count() / (local_ops * 1e3);
    HIPRINT("Thread {}: {} task allocs, {} ms, {} ops/sec, {} us/op",
            thread_id, local_ops, (elapsed_time.count() / 1e6), thread_throughput, avg_latency_us);
  }
}

/**
 * I/O worker thread function - continuously performs BDev I/O operations
 * Runs Allocate -> Write -> Free loop until stop flag is set
 */
void IOWorkerThread(size_t thread_id, const BenchmarkConfig &config,
                    clio::run::PoolId pool_id, std::atomic<bool> &stop_flag,
                    std::atomic<size_t> &completed_ops,
                    std::atomic<size_t> &total_bytes,
                    std::chrono::nanoseconds &elapsed_time) {
  // Create BDev client for this thread
  clio::run::bdev::Client bdev_client(pool_id);

  // Allocate data buffer in shared memory for writes (full io_size)
  auto write_buffer = CLIO_IPC->AllocateBuffer(config.io_size);
  std::memset(write_buffer.ptr_, static_cast<int>(thread_id), config.io_size);
  HLOG(kInfo, "Allocate write buffer for thread {}", config.io_size);

  size_t local_ops = 0;
  size_t local_bytes = 0;
  const size_t WARMUP_OPS = 5; // Ignore first 5 operations
  auto start_time = std::chrono::high_resolution_clock::now();

  // Continuously perform I/O operations until stop signal
  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Allocate blocks for the requested I/O size
    auto alloc_task = bdev_client.AsyncAllocateBlocks(clio::run::PoolQuery::Local(),
                                                       config.io_size);
    alloc_task.Wait();
    if (alloc_task->GetReturnCode() != 0 || alloc_task->blocks_.empty()) {
      HLOG(kError, "Thread {}: AllocateBlocks failed (rc={}, blocks={})",
           thread_id, alloc_task->GetReturnCode(), alloc_task->blocks_.size());
      stop_flag.store(true, std::memory_order_relaxed);
      CLIO_IPC->FreeBuffer(write_buffer);
      return;
    }
    std::vector<clio::run::bdev::Block> blocks;
    for (size_t i = 0; i < alloc_task->blocks_.size(); ++i) {
      blocks.push_back(alloc_task->blocks_[i]);
    }

    // Write data across all allocated blocks. Each block gets up to its
    // own size of payload; previously this was hard-capped at 4 KiB which
    // made the reported "Bandwidth" pure fiction for any io_size > 4 KiB.
    size_t bytes_written = 0;
    for (size_t block_idx = 0; block_idx < blocks.size(); block_idx++) {
      size_t bytes_remaining = config.io_size - bytes_written;
      size_t block_capacity = blocks[block_idx].size_;
      size_t bytes_to_write = std::min(bytes_remaining, block_capacity);

      // Create clio::run::priv::vector with single block for Write operation
      clio::run::priv::vector<clio::run::bdev::Block> single_block(CTP_MALLOC);
      single_block.push_back(blocks[block_idx]);

      auto write_task = bdev_client.AsyncWrite(clio::run::PoolQuery::Local(),
                                                single_block, write_buffer.shm_.template Cast<void>(), bytes_to_write);
      write_task.Wait();
      clio::run::u64 ret = write_task->bytes_written_;
      if (ret != bytes_to_write) {
        HLOG(kError, "ERROR: Thread {} failed to write data to block {}", thread_id, block_idx);
        stop_flag.store(true, std::memory_order_relaxed);
        return;
      }
      bytes_written += ret;
    }

    // Free blocks
    auto free_task = bdev_client.AsyncFreeBlocks(clio::run::PoolQuery::Local(), blocks);
    free_task.Wait();

    local_ops++;
    local_bytes += config.io_size;

    // Start timer after warmup operations
    if (local_ops == WARMUP_OPS) {
      start_time = std::chrono::high_resolution_clock::now();
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);

  // Free the allocated write buffer
  CLIO_IPC->FreeBuffer(write_buffer);

  // Update global counters
  completed_ops.fetch_add(local_ops, std::memory_order_relaxed);
  total_bytes.fetch_add(local_bytes, std::memory_order_relaxed);

  if (config.verbose) {
    double thread_throughput = (local_ops * 1e9) / elapsed_time.count();
    double bandwidth_mbps =
        (local_bytes * 1e9) / (elapsed_time.count() * 1024 * 1024);
    HIPRINT("Thread {}: {} I/O ops in {} ms, {} ops/sec, {} MB/s",
            thread_id, local_ops, (elapsed_time.count() / 1e6), thread_throughput, bandwidth_mbps);
  }
}

/**
 * Latency worker thread function - measures round-trip task latency
 * Uses MOD_NAME Custom function for pure task overhead measurement
 */
void LatencyWorkerThread(size_t thread_id, const BenchmarkConfig &config,
                         clio::run::PoolId pool_id, std::atomic<bool> &stop_flag,
                         std::atomic<size_t> &completed_ops,
                         std::chrono::nanoseconds &elapsed_time) {
  // Create MOD_NAME client for this thread
  clio::run::MOD_NAME::Client mod_client(pool_id);

  size_t local_ops = 0;
  const size_t WARMUP_OPS = 5; // Ignore first 5 operations
  auto start_time = std::chrono::high_resolution_clock::now();

  // Continuously perform Custom operations until stop signal
  std::string input_data = "test";
  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Call Custom with simple operation (operation_id = 0)
    auto task = mod_client.AsyncCustom(clio::run::PoolQuery::Broadcast(),
                                        input_data, 0);
    task.Wait();
    clio::run::u32 result = task->return_code_;

    // Verify result (should echo back input_data)
    if (result != 0) {
      HLOG(kError, "ERROR: Thread {} received unexpected result: {}", thread_id, result);
      stop_flag.store(true, std::memory_order_relaxed);
      return;
    }

    local_ops++;

    // Start timer after warmup operations
    if (local_ops == WARMUP_OPS) {
      start_time = std::chrono::high_resolution_clock::now();
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
      end_time - start_time);

  // Update global counters
  completed_ops.fetch_add(local_ops, std::memory_order_relaxed);

  if (config.verbose) {
    double thread_throughput = (local_ops * 1e9) / elapsed_time.count();
    double avg_latency_us = elapsed_time.count() / (local_ops * 1e3);
    HIPRINT("Thread {}: {} Custom ops in {} ms, {} ops/sec, {} us/op",
            thread_id, local_ops, (elapsed_time.count() / 1e6), thread_throughput, avg_latency_us);
  }
}

// ===========================================================================
// issue #781: scheduler-variety benchmark. Each client thread submits a stream
// of Custom tasks whose compute time (spin_us) is drawn from a mixed 1us..1s
// distribution (mostly quick, a few heavy/mislabeled). We record every request's
// round-trip latency and its class, then report avg / p50 / p99 / max PER CLASS.
// The headline metric is QUICK-class p99: with a scheduler that funnels quick
// tasks behind heavy ones it explodes; the anti-deadlock scheduler keeps it low.
// ===========================================================================

struct VarietySample {
  clio::run::u32 spin_us;
  double lat_us;
};

enum VarietyClass { kQuick = 0, kMedium, kHeavy, kNumVarietyClasses };
static VarietyClass ClassOf(clio::run::u32 spin_us) {
  if (spin_us < 100) return kQuick;      // < 100 us
  if (spin_us < 10000) return kMedium;   // 100 us .. 10 ms
  return kHeavy;                         // >= 10 ms
}
static const char *ClassName(int c) {
  switch (c) {
  case kQuick: return "quick  (<100us)   ";
  case kMedium: return "medium (100us-10ms)";
  default: return "heavy  (>=10ms)   ";
  }
}
// v must be pre-sorted ascending.
static double Pctl(const std::vector<double> &v, double p) {
  if (v.empty()) return 0.0;
  size_t idx = static_cast<size_t>(p * (v.size() - 1) + 0.5);
  if (idx >= v.size()) idx = v.size() - 1;
  return v[idx];
}

// Log-uniform draw within a mixed distribution: 90% quick, 9% medium, 1% heavy.
static clio::run::u32 DrawSpinUs(std::mt19937 &rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng), lo, hi;
  if (r < 0.01) { lo = 10000; hi = 2000000; }        // heavy  10ms..2s (some >1s
                                                      // to exercise stall detect)
  else if (r < 0.10) { lo = 100; hi = 10000; }       // medium 100us..10ms
  else { lo = 1; hi = 100; }                          // quick  1us..100us
  double e = std::log(lo) + u(rng) * (std::log(hi) - std::log(lo));
  clio::run::u32 s = static_cast<clio::run::u32>(std::exp(e));
  return s == 0 ? 1 : s;
}

void SchedVarietyWorkerThread(size_t thread_id, const BenchmarkConfig &config,
                              clio::run::PoolId pool_id,
                              std::atomic<bool> &stop_flag,
                              std::atomic<size_t> &completed_ops,
                              std::vector<VarietySample> &out) {
  clio::run::MOD_NAME::Client mod_client(pool_id);
  std::mt19937 rng(static_cast<uint32_t>(0x9e3779b9u ^
                                         (thread_id * 2654435761u)));
  std::string input_data = "v";
  out.reserve(1 << 16);
  size_t local_ops = 0;
  while (!stop_flag.load(std::memory_order_relaxed)) {
    clio::run::u32 spin_us = DrawSpinUs(rng);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto task = mod_client.AsyncCustom(clio::run::PoolQuery::Broadcast(),
                                       input_data, 0, spin_us);
    task.Wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    if (task->return_code_ != 0) {
      HLOG(kError, "variety: thread {} rc={}", thread_id, task->return_code_);
      stop_flag.store(true, std::memory_order_relaxed);
      break;
    }
    double lat_us =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() /
        1e3;
    out.push_back({spin_us, lat_us});
    local_ops++;
  }
  completed_ops.fetch_add(local_ops, std::memory_order_relaxed);
  (void)config;
}

int main(int argc, char **argv) {
  BenchmarkConfig config;

  // Unbuffered stdout so results survive even if process teardown aborts
  // (libzmq's static-destructor signaler assertion on Windows). std::cout is
  // synced with stdio by default, so this makes HIPRINT output flush eagerly.
  setvbuf(stdout, nullptr, _IONBF, 0);

  // Parse command line arguments
  if (!ParseArgs(argc, argv, config)) {
    return 1;
  }

  // Print benchmark header
  HIPRINT("=== Clio Task Throughput Benchmark ===");
  switch (config.test_case) {
  case TestCase::kBDevIO:
    HIPRINT("Test case: BDev I/O (Allocate -> Write -> Free)");
    HIPRINT("I/O size per operation: {} bytes", config.io_size);
    break;
  case TestCase::kBDevAllocation:
    HIPRINT("Test case: BDev Allocation (Allocate -> Free)");
    HIPRINT("Allocation size per operation: {} bytes", config.io_size);
    break;
  case TestCase::kBDevTaskAlloc:
    HIPRINT("Test case: BDev Task Allocation (NewTask -> DelTask)");
    HIPRINT("Task size: AllocateBlocksTask + FreeBlocksTask");
    break;
  case TestCase::kLatency:
    HIPRINT("Test case: Round-trip Latency (MOD_NAME Custom)");
    break;
  }
  HIPRINT("Threads: {}", config.num_threads);
  HIPRINT("Duration: {} seconds", config.duration_seconds);
  // max_file_size only sizes a bdev this run creates; when attaching to an
  // existing pool the capacity is whatever that pool was composed with.
  if (config.test_case != TestCase::kLatency && !config.attach_pool) {
    HIPRINT("Max file size: {} bytes", config.max_file_size);
  }

  // Initialize CLIO Runtime client
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "ERROR: Failed to initialize Clio client");
    BENCH_EXIT(1);
  }

  // Lane mapping always uses PID+TID hash
  HIPRINT("Lane policy: map_by_pid_tid (default)");

  // Create pool based on test case
  clio::run::PoolId test_pool_id;
  if (config.test_case == TestCase::kLatency ||
      config.test_case == TestCase::kSchedVariety) {
    // Create MOD_NAME container for latency / scheduler-variety tests (#781)
    test_pool_id = clio::run::PoolId(8000, 0);
    clio::run::MOD_NAME::Client mod_client(test_pool_id);
    auto create_task = mod_client.AsyncCreate(clio::run::PoolQuery::Broadcast(),
                      "latency_test_pool", test_pool_id);
    create_task.Wait();
    mod_client.pool_id_ = create_task->new_pool_id_;
    mod_client.return_code_ = create_task->return_code_;
    if (create_task->GetReturnCode() != 0) {
      HLOG(kError, "ERROR: Failed to create MOD_NAME container (return code: {})",
           create_task->GetReturnCode());
      BENCH_EXIT(1);
    }
  } else if (config.attach_pool) {
    // Attach to a pool that already exists in the runtime (composed via
    // `clio_run compose start`). This is how the bdev tests are pointed at a
    // safe_bdev: it serves the same AllocateBlocks/Write/FreeBlocks methods
    // (ids 10..12) over its member bdevs, so no client-side change is needed
    // beyond targeting its pool id.
    test_pool_id = config.attach_pool_id;
    clio::run::bdev::Client probe(test_pool_id);
    auto stats_task = probe.AsyncGetStats(clio::run::PoolQuery::Local());
    stats_task.Wait();
    if (stats_task->GetReturnCode() != 0) {
      HLOG(kError,
           "ERROR: pool {} is not reachable as a block device (return code: "
           "{}). Is it composed?",
           test_pool_id, stats_task->GetReturnCode());
      BENCH_EXIT(1);
    }
    HIPRINT("Attached to existing pool: {}", test_pool_id);
  } else {
    // Create BDev container for I/O and allocation tests
    test_pool_id = clio::run::PoolId(7000, 0);
    clio::run::bdev::Client bdev_client(test_pool_id);

    // Determine BDev type and pool name based on output directory
    clio::run::bdev::BdevType bdev_type;
    std::string pool_name;

    // Check if output_dir begins with "ram" (case-insensitive)
    bool is_ram_bdev = false;
    if (config.output_dir.size() >= 3) {
      std::string prefix = config.output_dir.substr(0, 3);
      // Convert to lowercase for comparison
      for (auto &c : prefix) {
        c = std::tolower(static_cast<unsigned char>(c));
      }
      is_ram_bdev = (prefix == "ram");
    }

    if (is_ram_bdev) {
      // Use RAM-based BDev
      bdev_type = clio::run::bdev::BdevType::kRam;
      pool_name = "benchmark_ram_bdev";
      HIPRINT("Using RAM-based BDev");
    } else {
      // Use file-based BDev
      bdev_type = clio::run::bdev::BdevType::kFile;
      std::filesystem::create_directories(config.output_dir);
      pool_name = config.output_dir + "/benchmark_bdev.dat";
      HIPRINT("Using file-based BDev: {}", pool_name);
    }

    auto create_task = bdev_client.AsyncCreate(clio::run::PoolQuery::Broadcast(), pool_name,
                                                test_pool_id, bdev_type, config.max_file_size, 32, 4096);
    create_task.Wait();

    // Update client pool_id_ with the actual pool ID from the task
    bdev_client.pool_id_ = create_task->new_pool_id_;
    bdev_client.return_code_ = create_task->return_code_;

    if (create_task->GetReturnCode() != 0) {
      HLOG(kError, "ERROR: Failed to create BDev container (return code: {})",
           create_task->GetReturnCode());
      BENCH_EXIT(1);
    }
  }

  HIPRINT("\nStarting benchmark...");

  // Atomic counters and control flag
  std::atomic<bool> stop_flag{false};
  std::atomic<size_t> completed_ops{0};
  std::atomic<size_t> total_bytes{0};

  // Storage for per-thread elapsed times
  std::vector<std::chrono::nanoseconds> thread_times(config.num_threads);

  // issue #781: per-thread latency samples for the variety benchmark
  std::vector<std::vector<VarietySample>> variety_samples(config.num_threads);

  // Spawn worker threads
  std::vector<std::thread> threads;
  threads.reserve(config.num_threads);

  auto benchmark_start = std::chrono::high_resolution_clock::now();

  switch (config.test_case) {
  case TestCase::kBDevAllocation:
    // Spawn allocation-only worker threads
    for (size_t i = 0; i < config.num_threads; i++) {
      threads.emplace_back(AllocationWorkerThread, i, std::ref(config),
                           test_pool_id, std::ref(stop_flag),
                           std::ref(completed_ops), std::ref(thread_times[i]));
    }
    break;

  case TestCase::kBDevTaskAlloc:
    // Spawn task allocation worker threads
    for (size_t i = 0; i < config.num_threads; i++) {
      threads.emplace_back(TaskAllocationWorkerThread, i, std::ref(config),
                           test_pool_id, std::ref(stop_flag),
                           std::ref(completed_ops), std::ref(thread_times[i]));
    }
    break;

  case TestCase::kBDevIO:
    // Spawn I/O worker threads
    for (size_t i = 0; i < config.num_threads; i++) {
      threads.emplace_back(IOWorkerThread, i, std::ref(config), test_pool_id,
                           std::ref(stop_flag), std::ref(completed_ops),
                           std::ref(total_bytes), std::ref(thread_times[i]));
    }
    break;

  case TestCase::kLatency:
    // Spawn latency worker threads
    for (size_t i = 0; i < config.num_threads; i++) {
      threads.emplace_back(LatencyWorkerThread, i, std::ref(config),
                           test_pool_id, std::ref(stop_flag),
                           std::ref(completed_ops), std::ref(thread_times[i]));
    }
    break;

  case TestCase::kSchedVariety:
    // issue #781: mixed-compute workers, each recording per-request samples
    for (size_t i = 0; i < config.num_threads; i++) {
      threads.emplace_back(SchedVarietyWorkerThread, i, std::ref(config),
                           test_pool_id, std::ref(stop_flag),
                           std::ref(completed_ops),
                           std::ref(variety_samples[i]));
    }
    break;
  }

  // Sleep for the specified duration
  std::this_thread::sleep_for(std::chrono::milliseconds(
      static_cast<long long>(config.duration_seconds * 1000)));

  // Signal threads to stop
  stop_flag.store(true, std::memory_order_relaxed);

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  auto benchmark_end = std::chrono::high_resolution_clock::now();
  auto total_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      benchmark_end - benchmark_start);

  // Get final counters
  size_t final_ops = completed_ops.load();
  size_t final_bytes = total_bytes.load();

  // Calculate statistics
  double total_seconds = total_elapsed.count() / 1e9;
  double throughput = final_ops / total_seconds;
  double avg_latency_us = (total_elapsed.count() / final_ops) / 1e3;

  // Calculate average per-thread time
  std::chrono::nanoseconds avg_thread_time{0};
  for (const auto &t : thread_times) {
    avg_thread_time += t;
  }
  avg_thread_time /= config.num_threads;

  // Print results
  HIPRINT("\n=== Results ===");
  HIPRINT("Total operations: {}", final_ops);
  HIPRINT("Total time: {} seconds", total_seconds);
  HIPRINT("Avg thread time: {} seconds", (avg_thread_time.count() / 1e9));

  switch (config.test_case) {
  case TestCase::kBDevAllocation:
    // Allocation-only mode results
    HIPRINT("Throughput: {} alloc/free ops/sec", throughput);
    HIPRINT("Avg latency: {} us/op", avg_latency_us);
    break;

  case TestCase::kBDevTaskAlloc:
    // Task allocation mode results
    HIPRINT("Throughput: {} task allocs/sec", throughput);
    HIPRINT("Avg latency: {} us/task", avg_latency_us);
    break;

  case TestCase::kBDevIO:
    // I/O mode results
    {
      double bandwidth_mbps = (final_bytes / total_seconds) / (1024 * 1024);
      HIPRINT("Total bytes written: {} ({} MB)", final_bytes, (final_bytes / (1024.0 * 1024.0)));
      HIPRINT("IOPS: {} ops/sec", throughput);
      HIPRINT("Bandwidth: {} MB/s", bandwidth_mbps);
      HIPRINT("Avg latency: {} us/op", avg_latency_us);
    }
    break;

  case TestCase::kLatency:
    // Latency mode results
    HIPRINT("Throughput: {} Custom ops/sec", throughput);
    HIPRINT("Avg round-trip latency: {} us/op", avg_latency_us);
    break;

  case TestCase::kSchedVariety: {
    // issue #781: bucket every request by its requested compute class and report
    // avg / p50 / p99 / max round-trip latency per class. QUICK p99 is the
    // starvation metric — how badly quick tasks are delayed by heavy ones.
    std::vector<double> lat[kNumVarietyClasses];
    for (auto &thread_vec : variety_samples) {
      for (const auto &s : thread_vec) {
        lat[ClassOf(s.spin_us)].push_back(s.lat_us);
      }
    }
    // ctp::Formatter supports only plain {} (stream insertion), no specs — round
    // to 1 decimal so the columns stay readable.
    auto r1 = [](double x) { return std::round(x * 10.0) / 10.0; };
    HIPRINT("Throughput: {} Custom ops/sec", throughput);
    HIPRINT("\n  class                  count   avg_us   p50_us   p99_us   max_us");
    for (int c = 0; c < kNumVarietyClasses; ++c) {
      auto &v = lat[c];
      if (v.empty()) continue;
      std::sort(v.begin(), v.end());
      double sum = 0.0;
      for (double x : v) sum += x;
      HIPRINT("  {}   {}   {}   {}   {}   {}", ClassName(c), v.size(),
              r1(sum / v.size()), r1(Pctl(v, 0.50)), r1(Pctl(v, 0.99)),
              r1(v.back()));
    }
    if (!lat[kQuick].empty()) {
      HIPRINT("\n  >>> STARVATION METRIC: quick-task p99 = {} us (p50 = {} us)",
              r1(Pctl(lat[kQuick], 0.99)), r1(Pctl(lat[kQuick], 0.50)));
    }
    break;
  }
  }

  BENCH_EXIT(0);
}
