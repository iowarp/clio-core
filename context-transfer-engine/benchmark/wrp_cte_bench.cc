/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * CTE Core Benchmark Application
 *
 * This benchmark measures the performance of Put, Get, and GetTagSize
 * operations in the Content Transfer Engine (CTE) with MPI support for parallel
 * I/O.
 *
 * Usage:
 *   mpirun -n <num_procs> wrp_cte_bench <test_case> <depth> <io_size>
 * <io_count>
 *
 * Parameters:
 *   test_case: Benchmark to conduct (Put, Get, PutGet)
 *   depth: Number of async requests to generate
 *   io_size: Size of I/O operations in bytes (supports k/K, m/M, g/G suffixes)
 *   io_count: Number of I/O operations to generate per node
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <thread>
#include <vector>

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>

using namespace std::chrono;

namespace {

/**
 * Parse size string with k/K, m/M, g/G suffixes
 */
chi::u64 ParseSize(const std::string &size_str) {
  chi::u64 size = 0;
  chi::u64 multiplier = 1;

  std::string num_str;
  char suffix = 0;

  for (char c : size_str) {
    if (std::isdigit(c)) {
      num_str += c;
    } else if (c == 'k' || c == 'K' || c == 'm' || c == 'M' || c == 'g' ||
               c == 'G') {
      suffix = std::tolower(c);
      break;
    }
  }

  if (num_str.empty()) {
    std::cerr << "Error: Invalid size format: " << size_str << std::endl;
    return 0;
  }

  size = std::stoull(num_str);

  switch (suffix) {
  case 'k':
    multiplier = 1024;
    break;
  case 'm':
    multiplier = 1024 * 1024;
    break;
  case 'g':
    multiplier = 1024 * 1024 * 1024;
    break;
  default:
    multiplier = 1;
    break;
  }

  return size * multiplier;
}

/**
 * Convert bytes to human-readable string with units
 */
std::string FormatSize(chi::u64 bytes) {
  if (bytes >= 1024ULL * 1024 * 1024) {
    return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
  } else if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + " MB";
  } else if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + " KB";
  } else {
    return std::to_string(bytes) + " B";
  }
}

/**
 * Convert milliseconds to appropriate unit
 */
std::string FormatTime(double milliseconds) {
  if (milliseconds >= 1000.0) {
    return std::to_string(milliseconds / 1000.0) + " s";
  } else {
    return std::to_string(milliseconds) + " ms";
  }
}

/**
 * Calculate bandwidth in MB/s
 */
double CalcBandwidth(chi::u64 total_bytes, double milliseconds) {
  if (milliseconds <= 0.0)
    return 0.0;
  double seconds = milliseconds / 1000.0;
  double megabytes = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  return megabytes / seconds;
}

/**
 * Helper function to check if runtime should be initialized
 * Reads CHIMAERA_WITH_RUNTIME environment variable
 * Returns true if unset or set to any value except "0", "false", "no", "off"
 */
bool ShouldInitializeRuntime() {
  const char *env_val = std::getenv("CHIMAERA_WITH_RUNTIME");
  if (env_val == nullptr) {
    return false; // Default for benchmark: assume runtime already initialized
  }
  std::string val(env_val);
  // Convert to lowercase for case-insensitive comparison
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  return !(val == "0" || val == "false" || val == "no" || val == "off");
}

} // namespace

/**
 * Main benchmark class
 */
class CTEBenchmark {
public:
  CTEBenchmark(int rank, int size, const std::string &test_case, int depth,
               chi::u64 io_size, int io_count)
      : rank_(rank), size_(size), test_case_(test_case), depth_(depth),
        io_size_(io_size), io_count_(io_count) {}

  ~CTEBenchmark() = default;

  /**
   * Run the benchmark
   */
  void Run() {
    if (rank_ == 0) {
      PrintBenchmarkInfo();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (test_case_ == "Put") {
      RunPutBenchmark();
    } else if (test_case_ == "Get") {
      RunGetBenchmark();
    } else if (test_case_ == "PutGet") {
      RunPutGetBenchmark();
    } else {
      if (rank_ == 0) {
        std::cerr << "Error: Unknown test case: " << test_case_ << std::endl;
        std::cerr << "Valid options: Put, Get, PutGet" << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

private:
  void PrintBenchmarkInfo() {
    std::cout << "=== CTE Core Benchmark ===" << std::endl;
    std::cout << "Test case: " << test_case_ << std::endl;
    std::cout << "MPI ranks: " << size_ << std::endl;
    std::cout << "Async depth: " << depth_ << std::endl;
    std::cout << "I/O size: " << FormatSize(io_size_) << std::endl;
    std::cout << "I/O count per rank: " << io_count_ << std::endl;
    std::cout << "Total I/O per rank: " << FormatSize(io_size_ * io_count_)
              << std::endl;
    std::cout << "Total I/O (all ranks): "
              << FormatSize(io_size_ * io_count_ * size_) << std::endl;
    std::cout << "===========================" << std::endl << std::endl;
  }

  void RunPutBenchmark() {
    // Allocate data buffer
    std::vector<char> data(io_size_);
    std::memset(data.data(), rank_ & 0xFF, io_size_);

    // Allocate shared memory buffer for async operations
    auto shm_buffer = CHI_IPC->AllocateBuffer(io_size_);
    std::memcpy(shm_buffer.ptr_, data.data(), io_size_);
    hipc::Pointer shm_ptr = shm_buffer.shm_;

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      int batch_size = std::min(depth_, io_count_ - i);
      std::vector<hipc::FullPtr<wrp_cte::core::PutBlobTask>> tasks;
      tasks.reserve(batch_size);

      // Generate async Put operations
      for (int j = 0; j < batch_size; ++j) {
        std::string tag_name =
            "tag_r" + std::to_string(rank_) + "_i" + std::to_string(i + j);
        wrp_cte::core::Tag tag(tag_name);
        std::string blob_name = "blob_0";

        auto task = tag.AsyncPutBlob(blob_name, shm_ptr, io_size_, 0, 0.8f);
        tasks.push_back(task);
      }

      // Wait for all async operations to complete
      for (auto &task : tasks) {
        task->Wait();
        CHI_IPC->DelTask(task);
      }
    }

    auto end_time = high_resolution_clock::now();
    auto duration_ms =
        duration_cast<milliseconds>(end_time - start_time).count();

    PrintResults("Put", duration_ms);
  }

  void RunGetBenchmark() {
    // Allocate data buffers
    std::vector<char> put_data(io_size_);
    std::vector<char> get_data(io_size_);

    // Allocate shared memory buffer
    auto shm_buffer = CHI_IPC->AllocateBuffer(io_size_);
    hipc::Pointer shm_ptr = shm_buffer.shm_;

    // First populate data using Put operations
    if (rank_ == 0) {
      std::cout << "Populating data for Get benchmark..." << std::endl;
    }

    for (int i = 0; i < io_count_; ++i) {
      std::string tag_name =
          "tag_r" + std::to_string(rank_) + "_i" + std::to_string(i);
      wrp_cte::core::Tag tag(tag_name);
      std::string blob_name = "blob_0";

      std::memset(put_data.data(), (rank_ + i) & 0xFF, io_size_);
      tag.PutBlob(blob_name, put_data.data(), io_size_);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank_ == 0) {
      std::cout << "Starting Get benchmark..." << std::endl;
    }

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      int batch_size = std::min(depth_, io_count_ - i);

      // For Get operations, use synchronous API in batches
      for (int j = 0; j < batch_size; ++j) {
        std::string tag_name =
            "tag_r" + std::to_string(rank_) + "_i" + std::to_string(i + j);
        wrp_cte::core::Tag tag(tag_name);
        std::string blob_name = "blob_0";

        tag.GetBlob(blob_name, get_data.data(), io_size_);
      }
    }

    auto end_time = high_resolution_clock::now();
    auto duration_ms =
        duration_cast<milliseconds>(end_time - start_time).count();

    PrintResults("Get", duration_ms);
  }

  void RunPutGetBenchmark() {
    // Allocate data buffers
    std::vector<char> put_data(io_size_);
    std::vector<char> get_data(io_size_);

    // Fill put data with pattern
    std::memset(put_data.data(), rank_ & 0xFF, io_size_);

    // Allocate shared memory buffer for async Put
    auto shm_buffer = CHI_IPC->AllocateBuffer(io_size_);
    std::memcpy(shm_buffer.ptr_, put_data.data(), io_size_);
    hipc::Pointer shm_ptr = shm_buffer.shm_;

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      int batch_size = std::min(depth_, io_count_ - i);
      std::vector<hipc::FullPtr<wrp_cte::core::PutBlobTask>> put_tasks;
      put_tasks.reserve(batch_size);

      // Generate async Put operations
      for (int j = 0; j < batch_size; ++j) {
        std::string tag_name =
            "tag_r" + std::to_string(rank_) + "_i" + std::to_string(i + j);
        wrp_cte::core::Tag tag(tag_name);
        std::string blob_name = "blob_0";

        auto task = tag.AsyncPutBlob(blob_name, shm_ptr, io_size_, 0, 0.8f);
        put_tasks.push_back(task);
      }

      // Wait for Put operations
      for (auto &task : put_tasks) {
        task->Wait();
        CHI_IPC->DelTask(task);
      }

      // Perform Get operations synchronously
      for (int j = 0; j < batch_size; ++j) {
        std::string tag_name =
            "tag_r" + std::to_string(rank_) + "_i" + std::to_string(i + j);
        wrp_cte::core::Tag tag(tag_name);
        std::string blob_name = "blob_0";

        tag.GetBlob(blob_name, get_data.data(), io_size_);
      }
    }

    auto end_time = high_resolution_clock::now();
    auto duration_ms =
        duration_cast<milliseconds>(end_time - start_time).count();

    PrintResults("PutGet", duration_ms);
  }

  void PrintResults(const std::string &operation, long long duration_ms) {
    // Gather timing results from all ranks
    std::vector<long long> all_times;
    if (rank_ == 0) {
      all_times.resize(size_);
    }

    MPI_Gather(&duration_ms, 1, MPI_LONG_LONG, all_times.data(), 1,
               MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank_ == 0) {
      // Calculate statistics
      long long min_time =
          *std::min_element(all_times.begin(), all_times.end());
      long long max_time =
          *std::max_element(all_times.begin(), all_times.end());
      long long sum_time = 0;
      for (auto t : all_times) {
        sum_time += t;
      }
      double avg_time = static_cast<double>(sum_time) / size_;

      chi::u64 total_bytes = io_size_ * io_count_;
      chi::u64 aggregate_bytes = total_bytes * size_;

      double min_bw = CalcBandwidth(total_bytes, min_time);
      double max_bw = CalcBandwidth(total_bytes, max_time);
      double avg_bw = CalcBandwidth(total_bytes, avg_time);
      double agg_bw = CalcBandwidth(aggregate_bytes, avg_time);

      std::cout << std::endl;
      std::cout << "=== " << operation << " Benchmark Results ===" << std::endl;
      std::cout << "Time (min): " << FormatTime(min_time) << std::endl;
      std::cout << "Time (max): " << FormatTime(max_time) << std::endl;
      std::cout << "Time (avg): " << FormatTime(avg_time) << std::endl;
      std::cout << std::endl;
      std::cout << "Bandwidth per rank (min): " << min_bw << " MB/s"
                << std::endl;
      std::cout << "Bandwidth per rank (max): " << max_bw << " MB/s"
                << std::endl;
      std::cout << "Bandwidth per rank (avg): " << avg_bw << " MB/s"
                << std::endl;
      std::cout << "Aggregate bandwidth: " << agg_bw << " MB/s" << std::endl;
      std::cout << "===========================" << std::endl;
    }
  }

  int rank_;
  int size_;
  std::string test_case_;
  int depth_;
  chi::u64 io_size_;
  int io_count_;
};

int main(int argc, char **argv) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Check arguments
  if (argc != 5) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0]
                << " <test_case> <depth> <io_size> <io_count>" << std::endl;
      std::cerr << "  test_case: Put, Get, or PutGet" << std::endl;
      std::cerr << "  depth: Number of async requests (e.g., 4)" << std::endl;
      std::cerr << "  io_size: Size of I/O operations (e.g., 1m, 4k, 1g)"
                << std::endl;
      std::cerr << "  io_count: Number of I/O operations per rank (e.g., 100)"
                << std::endl;
      std::cerr << std::endl;
      std::cerr << "Environment variables:" << std::endl;
      std::cerr << "  CHIMAERA_WITH_RUNTIME: Set to '1', 'true', 'yes', or 'on' to "
                   "initialize runtime"
                << std::endl;
      std::cerr
          << "                         Default: assumes runtime already initialized"
          << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // Initialize Chimaera client and optionally runtime
  bool should_init_runtime = ShouldInitializeRuntime();

  if (should_init_runtime) {
    if (rank == 0) {
      std::cout << "Initializing Chimaera runtime (CHIMAERA_WITH_RUNTIME="
                << std::getenv("CHIMAERA_WITH_RUNTIME") << ")..." << std::endl;
    }

    // Initialize runtime
    if (!chi::CHIMAERA_RUNTIME_INIT()) {
      if (rank == 0) {
        std::cerr << "Error: Failed to initialize Chimaera runtime"
                  << std::endl;
      }
      MPI_Finalize();
      return 1;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Initialize CTE client
    if (!wrp_cte::core::WRP_CTE_CLIENT_INIT()) {
      if (rank == 0) {
        std::cerr << "Error: Failed to initialize CTE client" << std::endl;
      }
      MPI_Finalize();
      return 1;
    }

    if (rank == 0) {
      std::cout << "Runtime and client initialized successfully" << std::endl;
    }
  } else {
    if (rank == 0) {
      std::cout
          << "Initializing CTE client only (runtime assumed initialized)..."
          << std::endl;
    }

    // Initialize client only
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
      if (rank == 0) {
        std::cerr << "Error: Failed to initialize Chimaera client" << std::endl;
      }
      MPI_Finalize();
      return 1;
    }

    if (!wrp_cte::core::WRP_CTE_CLIENT_INIT()) {
      if (rank == 0) {
        std::cerr << "Error: Failed to initialize CTE client" << std::endl;
      }
      MPI_Finalize();
      return 1;
    }

    if (rank == 0) {
      std::cout << "Client initialized successfully" << std::endl;
    }
  }

  // Small delay to ensure all ranks are synchronized
  MPI_Barrier(MPI_COMM_WORLD);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  std::string test_case = argv[1];
  int depth = std::atoi(argv[2]);
  chi::u64 io_size = ParseSize(argv[3]);
  int io_count = std::atoi(argv[4]);

  // Validate parameters
  if (depth <= 0 || io_size == 0 || io_count <= 0) {
    if (rank == 0) {
      std::cerr << "Error: Invalid parameters" << std::endl;
      std::cerr << "  depth must be > 0" << std::endl;
      std::cerr << "  io_size must be > 0" << std::endl;
      std::cerr << "  io_count must be > 0" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // Run benchmark
  CTEBenchmark benchmark(rank, size, test_case, depth, io_size, io_count);
  benchmark.Run();

  MPI_Finalize();
  return 0;
}
