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
 * operations in the Content Transfer Engine (CTE) with multi-threaded support.
 *
 * Usage:
 *   wrp_cte_bench <test_case> <num_threads> <depth> <io_size> <io_count>
 *
 * Parameters:
 *   test_case: Benchmark to conduct (Put, Get, PutGet)
 *   num_threads: Number of worker threads to spawn (e.g., 4)
 *   depth: Number of async requests to generate per thread
 *   io_size: Size of I/O operations in bytes (supports k/K, m/M, g/G suffixes)
 *   io_count: Number of I/O operations to generate per thread
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
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
 * Convert microseconds to appropriate unit
 */
std::string FormatTime(double microseconds) {
  if (microseconds >= 1000000.0) {
    return std::to_string(microseconds / 1000000.0) + " s";
  } else if (microseconds >= 1000.0) {
    return std::to_string(microseconds / 1000.0) + " ms";
  } else {
    return std::to_string(microseconds) + " us";
  }
}

/**
 * Calculate bandwidth in MB/s
 */
double CalcBandwidth(chi::u64 total_bytes, double microseconds) {
  if (microseconds <= 0.0)
    return 0.0;
  double seconds = microseconds / 1000000.0;
  double megabytes = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  return megabytes / seconds;
}

} // namespace

/**
 * Main benchmark class
 */
class CTEBenchmark {
public:
  CTEBenchmark(size_t num_threads, const std::string &test_case, int depth,
               chi::u64 io_size, int io_count)
      : num_threads_(num_threads), test_case_(test_case), depth_(depth),
        io_size_(io_size), io_count_(io_count) {}

  ~CTEBenchmark() = default;

  /**
   * Run the benchmark
   */
  void Run() {
    PrintBenchmarkInfo();

    if (test_case_ == "Put") {
      RunPutBenchmark();
    } else if (test_case_ == "Get") {
      RunGetBenchmark();
    } else if (test_case_ == "PutGet") {
      RunPutGetBenchmark();
    } else {
      std::cerr << "Error: Unknown test case: " << test_case_ << std::endl;
      std::cerr << "Valid options: Put, Get, PutGet" << std::endl;
    }
  }

private:
  void PrintBenchmarkInfo() {
    std::cout << "=== CTE Core Benchmark ===" << std::endl;
    std::cout << "Test case: " << test_case_ << std::endl;
    std::cout << "Worker threads: " << num_threads_ << std::endl;
    std::cout << "Async depth per thread: " << depth_ << std::endl;
    std::cout << "I/O size: " << FormatSize(io_size_) << std::endl;
    std::cout << "I/O count per thread: " << io_count_ << std::endl;
    std::cout << "Total I/O per thread: " << FormatSize(io_size_ * io_count_)
              << std::endl;
    std::cout << "Total I/O (all threads): "
              << FormatSize(io_size_ * io_count_ * num_threads_) << std::endl;
    std::cout << "===========================" << std::endl << std::endl;
  }

  /**
   * Worker thread for Put benchmark
   */
  void PutWorkerThread(size_t thread_id, std::atomic<bool> &error_flag,
                       std::vector<long long> &thread_times) {
    // Allocate data buffer
    std::vector<char> data(io_size_);
    std::memset(data.data(), thread_id & 0xFF, io_size_);

    // Allocate shared memory buffer for async operations
    auto shm_buffer = CHI_IPC->AllocateBuffer(io_size_);
    std::memcpy(shm_buffer.ptr_, data.data(), io_size_);
    hipc::Pointer shm_ptr = shm_buffer.shm_;

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      int batch_size = std::min(depth_, io_count_ - i);
      std::vector<hipc::FullPtr<wrp_cte::core::PutBlobTask>> tasks;
      tasks.reserve(batch_size);

      // Generate async Put operations
      for (int j = 0; j < batch_size; ++j) {
        std::string tag_name =
            "tag_t" + std::to_string(thread_id) + "_i" + std::to_string(i + j);
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
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();

    // Free shared memory buffer
    CHI_IPC->FreeBuffer(shm_buffer);
  }

  void RunPutBenchmark() {
    std::vector<std::thread> threads;
    std::vector<long long> thread_times(num_threads_);
    std::atomic<bool> error_flag{false};

    // Spawn worker threads
    for (size_t i = 0; i < num_threads_; ++i) {
      threads.emplace_back(&CTEBenchmark::PutWorkerThread, this, i,
                           std::ref(error_flag), std::ref(thread_times));
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
      thread.join();
    }

    PrintResults("Put", thread_times);
  }

  /**
   * Worker thread for Get benchmark
   */
  void GetWorkerThread(size_t thread_id, std::atomic<bool> &error_flag,
                       std::vector<long long> &thread_times) {
    // Allocate data buffers
    std::vector<char> put_data(io_size_);
    std::vector<char> get_data(io_size_);

    // First populate data using Put operations
    for (int i = 0; i < io_count_; ++i) {
      std::string tag_name =
          "tag_t" + std::to_string(thread_id) + "_i" + std::to_string(i);
      wrp_cte::core::Tag tag(tag_name);
      std::string blob_name = "blob_0";

      std::memset(put_data.data(), (thread_id + i) & 0xFF, io_size_);
      tag.PutBlob(blob_name, put_data.data(), io_size_);
    }

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      int batch_size = std::min(depth_, io_count_ - i);

      // For Get operations, use synchronous API in batches
      for (int j = 0; j < batch_size; ++j) {
        std::string tag_name =
            "tag_t" + std::to_string(thread_id) + "_i" + std::to_string(i + j);
        wrp_cte::core::Tag tag(tag_name);
        std::string blob_name = "blob_0";

        tag.GetBlob(blob_name, get_data.data(), io_size_);
      }
    }

    auto end_time = high_resolution_clock::now();
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();
  }

  void RunGetBenchmark() {
    std::cout << "Populating data for Get benchmark..." << std::endl;

    std::vector<std::thread> threads;
    std::vector<long long> thread_times(num_threads_);
    std::atomic<bool> error_flag{false};

    // Spawn worker threads
    for (size_t i = 0; i < num_threads_; ++i) {
      threads.emplace_back(&CTEBenchmark::GetWorkerThread, this, i,
                           std::ref(error_flag), std::ref(thread_times));
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
      thread.join();
    }

    PrintResults("Get", thread_times);
  }

  /**
   * Worker thread for PutGet benchmark
   */
  void PutGetWorkerThread(size_t thread_id, std::atomic<bool> &error_flag,
                          std::vector<long long> &thread_times) {
    // Allocate data buffers
    std::vector<char> put_data(io_size_);
    std::vector<char> get_data(io_size_);

    // Fill put data with pattern
    std::memset(put_data.data(), thread_id & 0xFF, io_size_);

    // Allocate shared memory buffer for async Put
    auto shm_buffer = CHI_IPC->AllocateBuffer(io_size_);
    std::memcpy(shm_buffer.ptr_, put_data.data(), io_size_);
    hipc::Pointer shm_ptr = shm_buffer.shm_;

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      int batch_size = std::min(depth_, io_count_ - i);
      std::vector<hipc::FullPtr<wrp_cte::core::PutBlobTask>> put_tasks;
      put_tasks.reserve(batch_size);

      // Generate async Put operations
      for (int j = 0; j < batch_size; ++j) {
        std::string tag_name =
            "tag_t" + std::to_string(thread_id) + "_i" + std::to_string(i + j);
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
            "tag_t" + std::to_string(thread_id) + "_i" + std::to_string(i + j);
        wrp_cte::core::Tag tag(tag_name);
        std::string blob_name = "blob_0";

        tag.GetBlob(blob_name, get_data.data(), io_size_);
      }
    }

    auto end_time = high_resolution_clock::now();
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();

    // Free shared memory buffer
    CHI_IPC->FreeBuffer(shm_buffer);
  }

  void RunPutGetBenchmark() {
    std::vector<std::thread> threads;
    std::vector<long long> thread_times(num_threads_);
    std::atomic<bool> error_flag{false};

    // Spawn worker threads
    for (size_t i = 0; i < num_threads_; ++i) {
      threads.emplace_back(&CTEBenchmark::PutGetWorkerThread, this, i,
                           std::ref(error_flag), std::ref(thread_times));
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
      thread.join();
    }

    PrintResults("PutGet", thread_times);
  }

  void PrintResults(const std::string &operation,
                    const std::vector<long long> &thread_times) {
    // Calculate statistics
    long long min_time = *std::min_element(thread_times.begin(), thread_times.end());
    long long max_time = *std::max_element(thread_times.begin(), thread_times.end());
    long long sum_time = 0;
    for (auto t : thread_times) {
      sum_time += t;
    }
    double avg_time = static_cast<double>(sum_time) / num_threads_;

    chi::u64 total_bytes = io_size_ * io_count_;
    chi::u64 aggregate_bytes = total_bytes * num_threads_;

    double min_bw = CalcBandwidth(total_bytes, min_time);
    double max_bw = CalcBandwidth(total_bytes, max_time);
    double avg_bw = CalcBandwidth(total_bytes, avg_time);
    double agg_bw = CalcBandwidth(aggregate_bytes, avg_time);

    // Calculate bandwidth in bytes/sec for finer granularity
    double min_bw_bytes = min_time > 0 ? (static_cast<double>(total_bytes) / (min_time / 1000000.0)) : 0.0;
    double max_bw_bytes = max_time > 0 ? (static_cast<double>(total_bytes) / (max_time / 1000000.0)) : 0.0;
    double avg_bw_bytes = avg_time > 0 ? (static_cast<double>(total_bytes) / (avg_time / 1000000.0)) : 0.0;
    double agg_bw_bytes = avg_time > 0 ? (static_cast<double>(aggregate_bytes) / (avg_time / 1000000.0)) : 0.0;

    std::cout << std::endl;
    std::cout << "=== " << operation << " Benchmark Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time (min): " << min_time << " us (" << (min_time / 1000.0) << " ms)" << std::endl;
    std::cout << "Time (max): " << max_time << " us (" << (max_time / 1000.0) << " ms)" << std::endl;
    std::cout << "Time (avg): " << avg_time << " us (" << (avg_time / 1000.0) << " ms)" << std::endl;
    std::cout << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Bandwidth per thread (min): " << min_bw << " MB/s (" << min_bw_bytes << " bytes/s)"
              << std::endl;
    std::cout << "Bandwidth per thread (max): " << max_bw << " MB/s (" << max_bw_bytes << " bytes/s)"
              << std::endl;
    std::cout << "Bandwidth per thread (avg): " << avg_bw << " MB/s (" << avg_bw_bytes << " bytes/s)"
              << std::endl;
    std::cout << "Aggregate bandwidth: " << agg_bw << " MB/s (" << agg_bw_bytes << " bytes/s)" << std::endl;
    std::cout << "===========================" << std::endl;
  }

  size_t num_threads_;
  std::string test_case_;
  int depth_;
  chi::u64 io_size_;
  int io_count_;
};

int main(int argc, char **argv) {
  // Check arguments
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <test_case> <num_threads> <depth> <io_size> <io_count>"
              << std::endl;
    std::cerr << "  test_case: Put, Get, or PutGet" << std::endl;
    std::cerr << "  num_threads: Number of worker threads (e.g., 4)" << std::endl;
    std::cerr << "  depth: Number of async requests per thread (e.g., 4)" << std::endl;
    std::cerr << "  io_size: Size of I/O operations (e.g., 1m, 4k, 1g)"
              << std::endl;
    std::cerr << "  io_count: Number of I/O operations per thread (e.g., 100)"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "Environment variables:" << std::endl;
    std::cerr << "  CHIMAERA_WITH_RUNTIME: Set to '1', 'true', 'yes', or 'on' to "
                 "initialize runtime"
              << std::endl;
    std::cerr
        << "                         Default: assumes runtime already initialized"
        << std::endl;
    return 1;
  }

  // Initialize Chimaera runtime and client
  std::cout << "Initializing Chimaera runtime..." << std::endl;

  // Initialize Chimaera (client with embedded runtime)
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
    std::cerr << "Error: Failed to initialize Chimaera runtime" << std::endl;
    return 1;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Initialize CTE client
  if (!wrp_cte::core::WRP_CTE_CLIENT_INIT()) {
    std::cerr << "Error: Failed to initialize CTE client" << std::endl;
    return 1;
  }

  std::cout << "Runtime and client initialized successfully" << std::endl;

  // Small delay to ensure initialization is complete
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  std::string test_case = argv[1];
  size_t num_threads = std::stoull(argv[2]);
  int depth = std::atoi(argv[3]);
  chi::u64 io_size = ParseSize(argv[4]);
  int io_count = std::atoi(argv[5]);

  // Validate parameters
  if (num_threads == 0 || depth <= 0 || io_size == 0 || io_count <= 0) {
    std::cerr << "Error: Invalid parameters" << std::endl;
    std::cerr << "  num_threads must be > 0" << std::endl;
    std::cerr << "  depth must be > 0" << std::endl;
    std::cerr << "  io_size must be > 0" << std::endl;
    std::cerr << "  io_count must be > 0" << std::endl;
    return 1;
  }

  // Run benchmark
  CTEBenchmark benchmark(num_threads, test_case, depth, io_size, io_count);
  benchmark.Run();

  return 0;
}
