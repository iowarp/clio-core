/**
 * Task Throughput and Latency Benchmark
 *
 * Benchmarks different aspects of the Chimaera runtime:
 * - BDev I/O throughput (allocate/write/free)
 * - BDev allocation throughput (allocate/free only)
 * - Round-trip latency using MOD_NAME Custom function
 */

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "chimaera/MOD_NAME/MOD_NAME_client.h"
#include "chimaera/admin/admin_client.h"
#include "chimaera/bdev/bdev_client.h"
#include "chimaera/chimaera.h"

/**
 * Benchmark test cases
 */
enum class TestCase {
  kBDevIO,         // Full I/O (Allocate -> Write -> Free)
  kBDevAllocation, // Allocation only (Allocate -> Free)
  kBDevTaskAlloc,  // Task allocation/deletion (NewTask -> DelTask)
  kLatency         // Round-trip latency using MOD_NAME Custom
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
      "/tmp/wrp_benchmark"; // Output directory for benchmark files
};

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
        std::cerr << "ERROR: Invalid test case. Valid options: bdev_io, "
                     "bdev_allocation, bdev_task_alloc, latency\n";
        return false;
      }
    } else if (arg == "--threads" && i + 1 < argc) {
      config.num_threads = std::stoull(argv[++i]);
    } else if (arg == "--duration" && i + 1 < argc) {
      config.duration_seconds = std::stod(argv[++i]);
    } else if (arg == "--max-file-size" && i + 1 < argc) {
      config.max_file_size = hshm::ConfigParse::ParseSize(argv[++i]);
    } else if (arg == "--io-size" && i + 1 < argc) {
      config.io_size = hshm::ConfigParse::ParseSize(argv[++i]);
    } else if (arg == "--lane-policy" && i + 1 < argc) {
      config.lane_policy = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else if (arg == "--verbose" || arg == "-v") {
      config.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "Options:\n"
          << "  --test-case <case>      Test case: bdev_io, bdev_allocation, "
             "bdev_task_alloc, latency (default: bdev_io)\n"
          << "  --threads <N>           Number of client threads (default: 4)\n"
          << "  --duration <seconds>    Duration to run benchmark in seconds "
             "(default: 10.0)\n"
          << "  --max-file-size <size>  Maximum file size with suffix: k, m, g "
             "(default: 1g)\n"
          << "  --io-size <size>        I/O size per operation with suffix: k, "
             "m, g (default: 4k)\n"
          << "  --lane-policy <P>       Lane policy: map_by_pid_tid, "
             "round_robin, random (default: from config)\n"
          << "  --output-dir <dir>      Output directory for benchmark files "
             "(default: /tmp/wrp_benchmark)\n"
          << "  --verbose, -v           Verbose output\n"
          << "  --help, -h              Show this help\n\n"
          << "Test Cases:\n"
          << "  bdev_io          - BDev I/O throughput (Allocate -> Write -> "
             "Free)\n"
          << "  bdev_allocation  - BDev allocation throughput (Allocate -> "
             "Free)\n"
          << "  bdev_task_alloc  - BDev task allocation (NewTask -> DelTask)\n"
          << "  latency          - Round-trip task latency using MOD_NAME "
             "Custom\n";
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
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
                            chi::PoolId pool_id, std::atomic<bool> &stop_flag,
                            std::atomic<size_t> &completed_ops,
                            std::chrono::nanoseconds &elapsed_time) {
  // Create BDev client for this thread
  chimaera::bdev::Client bdev_client(pool_id);

  // Use io_size for allocation-only benchmark
  size_t alloc_size = config.io_size;
  HILOG(kInfo, "Allocate size: {}", alloc_size);

  size_t local_ops = 0;
  const size_t WARMUP_OPS = 5; // Ignore first 5 operations
  auto start_time = std::chrono::high_resolution_clock::now();

  // Continuously perform allocate/free operations until stop signal
  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Allocate blocks
    auto blocks = bdev_client.AllocateBlocks(HSHM_MCTX, chi::PoolQuery::Local(),
                                             alloc_size);

    // Free blocks immediately
    bdev_client.FreeBlocks(HSHM_MCTX, chi::PoolQuery::Local(), blocks);

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
    std::cout << "Thread " << thread_id << ": " << local_ops
              << " alloc/free ops in " << std::fixed << std::setprecision(3)
              << (elapsed_time.count() / 1e6) << " ms, " << std::setprecision(0)
              << thread_throughput << " ops/sec, " << std::setprecision(3)
              << avg_latency_us << " us/op\n";
  }
}

/**
 * Task allocation worker thread - benchmarks NewTask/DelTask overhead
 * Creates AllocateBlocksTask and FreeBlocksTask, then immediately deletes them
 */
void TaskAllocationWorkerThread(size_t thread_id, const BenchmarkConfig &config,
                                chi::PoolId pool_id,
                                std::atomic<bool> &stop_flag,
                                std::atomic<size_t> &completed_ops,
                                std::chrono::nanoseconds &elapsed_time) {
  // Get IPC manager
  auto *ipc_manager = CHI_IPC;

  // Use io_size for task allocation benchmark
  size_t alloc_size = config.io_size;

  // Create dummy blocks vector for FreeBlocksTask
  std::vector<chimaera::bdev::Block> dummy_blocks(2);
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
    auto alloc_task = ipc_manager->NewTask<chimaera::bdev::AllocateBlocksTask>(
        chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(), alloc_size);
    ipc_manager->DelTask(alloc_task);

    // Create and delete FreeBlocksTask
    auto free_task = ipc_manager->NewTask<chimaera::bdev::FreeBlocksTask>(
        chi::CreateTaskId(), pool_id, chi::PoolQuery::Local(), dummy_blocks);
    ipc_manager->DelTask(free_task);

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
    std::cout << "Thread " << thread_id << ": " << local_ops << " task allocs, "
              << std::setprecision(2) << (elapsed_time.count() / 1e6) << " ms, "
              << std::setprecision(0) << thread_throughput << " ops/sec, "
              << std::setprecision(3) << avg_latency_us << " us/op\n";
  }
}

/**
 * I/O worker thread function - continuously performs BDev I/O operations
 * Runs Allocate -> Write -> Free loop until stop flag is set
 */
void IOWorkerThread(size_t thread_id, const BenchmarkConfig &config,
                    chi::PoolId pool_id, std::atomic<bool> &stop_flag,
                    std::atomic<size_t> &completed_ops,
                    std::atomic<size_t> &total_bytes,
                    std::chrono::nanoseconds &elapsed_time) {
  // Create BDev client for this thread
  chimaera::bdev::Client bdev_client(pool_id);

  // Allocate data buffer in shared memory for writes (full io_size)
  auto write_buffer = CHI_IPC->AllocateBuffer(config.io_size);
  std::memset(write_buffer.ptr_, static_cast<int>(thread_id), config.io_size);
  HILOG(kInfo, "Allocate write buffer for thread {}", config.io_size);

  size_t local_ops = 0;
  size_t local_bytes = 0;
  const size_t WARMUP_OPS = 5; // Ignore first 5 operations
  auto start_time = std::chrono::high_resolution_clock::now();

  // Continuously perform I/O operations until stop signal
  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Allocate blocks for the requested I/O size
    auto blocks = bdev_client.AllocateBlocks(HSHM_MCTX, chi::PoolQuery::Local(),
                                             config.io_size);

    // Write data across all allocated blocks
    size_t bytes_written = 0;
    for (size_t block_idx = 0; block_idx < blocks.size(); block_idx++) {
      size_t bytes_remaining = config.io_size - bytes_written;
      size_t bytes_to_write = std::min(bytes_remaining, size_t(4096));

      // Create pointer offset into write buffer
      hipc::Pointer write_ptr = write_buffer.shm_;
      write_ptr.off_ += bytes_written;

      // Create ArrayVector with single block for Write operation
      chimaera::bdev::ArrayVector<chimaera::bdev::Block, 16> single_block;
      single_block.push_back(blocks[block_idx]);

      chi::u64 ret =
          bdev_client.Write(HSHM_MCTX, chi::PoolQuery::Local(),
                            single_block, write_ptr, bytes_to_write);
      if (ret != bytes_to_write) {
        std::cerr << "ERROR: Thread " << thread_id
                  << " failed to write data to block " << block_idx << "\n";
        stop_flag.store(true, std::memory_order_relaxed);
        return;
      }
      bytes_written += ret;
    }

    // Free blocks
    bdev_client.FreeBlocks(HSHM_MCTX, chi::PoolQuery::Local(), blocks);

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
  CHI_IPC->FreeBuffer(write_buffer);

  // Update global counters
  completed_ops.fetch_add(local_ops, std::memory_order_relaxed);
  total_bytes.fetch_add(local_bytes, std::memory_order_relaxed);

  if (config.verbose) {
    double thread_throughput = (local_ops * 1e9) / elapsed_time.count();
    double bandwidth_mbps =
        (local_bytes * 1e9) / (elapsed_time.count() * 1024 * 1024);
    std::cout << "Thread " << thread_id << ": " << local_ops << " I/O ops in "
              << std::fixed << std::setprecision(3)
              << (elapsed_time.count() / 1e6) << " ms, " << std::setprecision(0)
              << thread_throughput << " ops/sec, " << std::setprecision(2)
              << bandwidth_mbps << " MB/s\n";
  }
}

/**
 * Latency worker thread function - measures round-trip task latency
 * Uses MOD_NAME Custom function for pure task overhead measurement
 */
void LatencyWorkerThread(size_t thread_id, const BenchmarkConfig &config,
                         chi::PoolId pool_id, std::atomic<bool> &stop_flag,
                         std::atomic<size_t> &completed_ops,
                         std::chrono::nanoseconds &elapsed_time) {
  // Create MOD_NAME client for this thread
  chimaera::MOD_NAME::Client mod_client(pool_id);

  size_t local_ops = 0;
  const size_t WARMUP_OPS = 5; // Ignore first 5 operations
  auto start_time = std::chrono::high_resolution_clock::now();

  // Continuously perform Custom operations until stop signal
  std::string input_data = "test";
  std::string output_data;
  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Call Custom with simple operation (operation_id = 0)
    chi::u32 result = mod_client.Custom(HSHM_MCTX, chi::PoolQuery::Broadcast(),
                                        input_data, 0, output_data);

    // Verify result (should echo back input_data)
    if (result != 0) {
      std::cerr << "ERROR: Thread " << thread_id
                << " received unexpected result: " << result << "\n";
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
    std::cout << "Thread " << thread_id << ": " << local_ops
              << " Custom ops in " << std::fixed << std::setprecision(3)
              << (elapsed_time.count() / 1e6) << " ms, " << std::setprecision(0)
              << thread_throughput << " ops/sec, " << std::setprecision(3)
              << avg_latency_us << " us/op\n";
  }
}

int main(int argc, char **argv) {
  BenchmarkConfig config;

  // Parse command line arguments
  if (!ParseArgs(argc, argv, config)) {
    return 1;
  }

  // Print benchmark header
  std::cout << "=== Chimaera Task Throughput Benchmark ===\n";
  switch (config.test_case) {
  case TestCase::kBDevIO:
    std::cout << "Test case: BDev I/O (Allocate -> Write -> Free)\n";
    std::cout << "I/O size per operation: " << config.io_size << " bytes\n";
    break;
  case TestCase::kBDevAllocation:
    std::cout << "Test case: BDev Allocation (Allocate -> Free)\n";
    std::cout << "Allocation size per operation: " << config.io_size
              << " bytes\n";
    break;
  case TestCase::kBDevTaskAlloc:
    std::cout << "Test case: BDev Task Allocation (NewTask -> DelTask)\n";
    std::cout << "Task size: AllocateBlocksTask + FreeBlocksTask\n";
    break;
  case TestCase::kLatency:
    std::cout << "Test case: Round-trip Latency (MOD_NAME Custom)\n";
    break;
  }
  std::cout << "Threads: " << config.num_threads << "\n";
  std::cout << "Duration: " << config.duration_seconds << " seconds\n";
  if (config.test_case != TestCase::kLatency) {
    std::cout << "Max file size: " << config.max_file_size << " bytes\n";
  }

  // Initialize Chimaera client
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    std::cerr << "ERROR: Failed to initialize Chimaera client\n";
    return 1;
  }

  // Apply lane policy override if specified
  if (!config.lane_policy.empty()) {
    auto *ipc_manager = CHI_IPC;
    if (ipc_manager) {
      chi::LaneMapPolicy policy;
      if (config.lane_policy == "map_by_pid_tid") {
        policy = chi::LaneMapPolicy::kMapByPidTid;
      } else if (config.lane_policy == "round_robin") {
        policy = chi::LaneMapPolicy::kRoundRobin;
      } else if (config.lane_policy == "random") {
        policy = chi::LaneMapPolicy::kRandom;
      } else {
        std::cerr << "ERROR: Unknown lane policy: " << config.lane_policy
                  << "\n";
        return 1;
      }
      ipc_manager->SetLaneMapPolicy(policy);
      std::cout << "Lane policy: " << config.lane_policy << " (override)\n";
    }
  } else {
    auto *ipc_manager = CHI_IPC;
    if (ipc_manager) {
      auto policy = ipc_manager->GetLaneMapPolicy();
      const char *policy_name = "unknown";
      switch (policy) {
      case chi::LaneMapPolicy::kMapByPidTid:
        policy_name = "map_by_pid_tid";
        break;
      case chi::LaneMapPolicy::kRoundRobin:
        policy_name = "round_robin";
        break;
      case chi::LaneMapPolicy::kRandom:
        policy_name = "random";
        break;
      }
      std::cout << "Lane policy: " << policy_name << " (from config)\n";
    }
  }

  // Create pool based on test case
  chi::PoolId test_pool_id;
  if (config.test_case == TestCase::kLatency) {
    // Create MOD_NAME container for latency test
    test_pool_id = chi::PoolId(8000, 0);
    chimaera::MOD_NAME::Client mod_client(test_pool_id);
    mod_client.Create(HSHM_MCTX, chi::PoolQuery::Broadcast(),
                      "latency_test_pool", test_pool_id);
    if (mod_client.GetReturnCode() != 0) {
      std::cerr << "ERROR: Failed to create MOD_NAME container (return code: "
                << mod_client.GetReturnCode() << ")\n";
      return 1;
    }
  } else {
    // Create BDev container for I/O and allocation tests
    test_pool_id = chi::PoolId(7000, 0);
    chimaera::bdev::Client bdev_client(test_pool_id);

    // Determine BDev type and pool name based on output directory
    chimaera::bdev::BdevType bdev_type;
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
      bdev_type = chimaera::bdev::BdevType::kRam;
      pool_name = "benchmark_ram_bdev";
      std::cout << "Using RAM-based BDev\n";
    } else {
      // Use file-based BDev
      bdev_type = chimaera::bdev::BdevType::kFile;
      pool_name = config.output_dir + "/benchmark_bdev.dat";
      std::cout << "Using file-based BDev: " << pool_name << "\n";
    }

    bdev_client.Create(HSHM_MCTX, chi::PoolQuery::Broadcast(), pool_name,
                       test_pool_id, bdev_type, config.max_file_size, 32, 4096);
    if (bdev_client.GetReturnCode() != 0) {
      std::cerr << "ERROR: Failed to create BDev container (return code: "
                << bdev_client.GetReturnCode() << ")\n";
      return 1;
    }
  }

  std::cout << "\nStarting benchmark...\n";

  // Atomic counters and control flag
  std::atomic<bool> stop_flag{false};
  std::atomic<size_t> completed_ops{0};
  std::atomic<size_t> total_bytes{0};

  // Storage for per-thread elapsed times
  std::vector<std::chrono::nanoseconds> thread_times(config.num_threads);

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
  std::cout << "\n=== Results ===\n";
  std::cout << "Total operations: " << final_ops << "\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Total time: " << total_seconds << " seconds\n";
  std::cout << "Avg thread time: " << (avg_thread_time.count() / 1e9)
            << " seconds\n";

  switch (config.test_case) {
  case TestCase::kBDevAllocation:
    // Allocation-only mode results
    std::cout << std::setprecision(0);
    std::cout << "Throughput: " << throughput << " alloc/free ops/sec\n";
    std::cout << std::setprecision(3);
    std::cout << "Avg latency: " << avg_latency_us << " us/op\n";
    break;

  case TestCase::kBDevTaskAlloc:
    // Task allocation mode results
    std::cout << std::setprecision(0);
    std::cout << "Throughput: " << throughput << " task allocs/sec\n";
    std::cout << std::setprecision(3);
    std::cout << "Avg latency: " << avg_latency_us << " us/task\n";
    break;

  case TestCase::kBDevIO:
    // I/O mode results
    {
      double bandwidth_mbps = (final_bytes / total_seconds) / (1024 * 1024);
      std::cout << "Total bytes written: " << final_bytes << " ("
                << std::setprecision(2) << (final_bytes / (1024.0 * 1024.0))
                << " MB)\n";
      std::cout << std::setprecision(0);
      std::cout << "IOPS: " << throughput << " ops/sec\n";
      std::cout << std::setprecision(2);
      std::cout << "Bandwidth: " << bandwidth_mbps << " MB/s\n";
      std::cout << std::setprecision(3);
      std::cout << "Avg latency: " << avg_latency_us << " us/op\n";
    }
    break;

  case TestCase::kLatency:
    // Latency mode results
    std::cout << std::setprecision(0);
    std::cout << "Throughput: " << throughput << " Custom ops/sec\n";
    std::cout << std::setprecision(3);
    std::cout << "Avg round-trip latency: " << avg_latency_us << " us/op\n";
    break;
  }

  return 0;
}
