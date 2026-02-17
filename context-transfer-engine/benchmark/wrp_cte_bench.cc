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

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <hermes_shm/util/logging.h>

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

using namespace std::chrono;

namespace {

/**
 * Parse size string with k/K, m/M, g/G suffixes
 * Supports decimal numbers (e.g., 0.5g, 1.5m, 2.5k)
 */
chi::u64 ParseSize(const std::string &size_str) {
  double size = 0.0;
  chi::u64 multiplier = 1;

  std::string num_str;
  char suffix = 0;

  for (char c : size_str) {
    if (std::isdigit(c) || c == '.') {
      num_str += c;
    } else if (c == 'k' || c == 'K' || c == 'm' || c == 'M' || c == 'g' ||
               c == 'G') {
      suffix = std::tolower(c);
      break;
    }
  }

  if (num_str.empty()) {
    HLOG(kError, "Invalid size format: {}", size_str);
    return 0;
  }

  size = std::stod(num_str);

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

  return static_cast<chi::u64>(size * multiplier);
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
  if (microseconds <= 0.0) return 0.0;
  double seconds = microseconds / 1000000.0;
  double megabytes = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  return megabytes / seconds;
}

}  // namespace

/**
 * Main benchmark class
 */
class CTEBenchmark {
 public:
  CTEBenchmark(size_t num_threads, const std::string &test_case, int depth,
               chi::u64 io_size, int io_count)
      : num_threads_(num_threads),
        test_case_(test_case),
        depth_(depth),
        io_size_(io_size),
        io_count_(io_count) {}

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
      HLOG(kError, "Unknown test case: {}", test_case_);
      HLOG(kError, "Valid options: Put, Get, PutGet");
    }
  }

 private:
  void PrintBenchmarkInfo() {
    HLOG(kInfo, "=== CTE Core Benchmark ===");
    HLOG(kInfo, "Test case: {}", test_case_);
    HLOG(kInfo, "Worker threads: {}", num_threads_);
    HLOG(kInfo, "Async depth per thread: {}", depth_);
    HLOG(kInfo, "I/O size: {}", FormatSize(io_size_));
    HLOG(kInfo, "I/O count per thread: {}", io_count_);
    HLOG(kInfo, "Total I/O per thread: {}", FormatSize(io_size_ * io_count_));
    HLOG(kInfo, "Total I/O (all threads): {}",
         FormatSize(io_size_ * io_count_ * num_threads_));
    HLOG(kInfo, "===========================");
  }

  /**
   * Worker thread for Put benchmark
   */
  void PutWorkerThread(size_t thread_id, std::atomic<bool> &error_flag,
                       std::vector<long long> &thread_times) {
    auto *cte_client = WRP_CTE_CLIENT;

    // Allocate shared memory buffer
    auto shm_buffer = CHI_IPC->AllocateBuffer(io_size_);
    std::memset(shm_buffer.ptr_, thread_id & 0xFF, io_size_);
    hipc::ShmPtr<> shm_ptr = shm_buffer.shm_.template Cast<void>();

    // Create one tag per thread
    std::string tag_name = "tag_t" + std::to_string(thread_id);
    auto tag_task = cte_client->AsyncGetOrCreateTag(tag_name);
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      int batch_size = std::min(depth_, io_count_ - i);
      std::vector<chi::Future<wrp_cte::core::PutBlobTask>> tasks;
      tasks.reserve(batch_size);

      for (int j = 0; j < batch_size; ++j) {
        std::string blob_name = "blob_" + std::to_string(i + j);
        auto task = cte_client->AsyncPutBlob(tag_id, blob_name, 0, io_size_,
                                             shm_ptr, 0.8f);
        tasks.push_back(task);
      }

      for (auto &task : tasks) {
        task.Wait();
      }
    }

    auto end_time = high_resolution_clock::now();
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();

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
    auto *cte_client = WRP_CTE_CLIENT;

    // Allocate shared memory buffers
    auto put_shm = CHI_IPC->AllocateBuffer(io_size_);
    auto get_shm = CHI_IPC->AllocateBuffer(io_size_);
    hipc::ShmPtr<> put_ptr = put_shm.shm_.template Cast<void>();
    hipc::ShmPtr<> get_ptr = get_shm.shm_.template Cast<void>();

    // Create one tag per thread
    std::string tag_name = "tag_t" + std::to_string(thread_id);
    auto tag_task = cte_client->AsyncGetOrCreateTag(tag_name);
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    // Populate data using Put operations
    for (int i = 0; i < io_count_; ++i) {
      std::memset(put_shm.ptr_, (thread_id + i) & 0xFF, io_size_);
      std::string blob_name = "blob_" + std::to_string(i);
      auto task = cte_client->AsyncPutBlob(tag_id, blob_name, 0, io_size_,
                                           put_ptr, 0.8f);
      task.Wait();
    }

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      int batch_size = std::min(depth_, io_count_ - i);

      for (int j = 0; j < batch_size; ++j) {
        std::string blob_name = "blob_" + std::to_string(i + j);
        auto task = cte_client->AsyncGetBlob(tag_id, blob_name, 0, io_size_, 0,
                                             get_ptr);
        task.Wait();
      }
    }

    auto end_time = high_resolution_clock::now();
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();

    CHI_IPC->FreeBuffer(put_shm);
    CHI_IPC->FreeBuffer(get_shm);
  }

  void RunGetBenchmark() {
    HLOG(kInfo, "Populating data for Get benchmark...");

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
    auto *cte_client = WRP_CTE_CLIENT;

    // Allocate shared memory buffers
    auto put_shm = CHI_IPC->AllocateBuffer(io_size_);
    auto get_shm = CHI_IPC->AllocateBuffer(io_size_);
    std::memset(put_shm.ptr_, thread_id & 0xFF, io_size_);
    hipc::ShmPtr<> put_ptr = put_shm.shm_.template Cast<void>();
    hipc::ShmPtr<> get_ptr = get_shm.shm_.template Cast<void>();

    // Create one tag per thread
    std::string tag_name = "tag_t" + std::to_string(thread_id);
    auto tag_task = cte_client->AsyncGetOrCreateTag(tag_name);
    tag_task.Wait();
    wrp_cte::core::TagId tag_id = tag_task->tag_id_;

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < io_count_; i += depth_) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      int batch_size = std::min(depth_, io_count_ - i);
      std::vector<chi::Future<wrp_cte::core::PutBlobTask>> put_tasks;
      put_tasks.reserve(batch_size);

      for (int j = 0; j < batch_size; ++j) {
        std::string blob_name = "blob_" + std::to_string(i + j);
        auto task = cte_client->AsyncPutBlob(tag_id, blob_name, 0, io_size_,
                                             put_ptr, 0.8f);
        put_tasks.push_back(task);
      }

      for (auto &task : put_tasks) {
        task.Wait();
      }

      for (int j = 0; j < batch_size; ++j) {
        std::string blob_name = "blob_" + std::to_string(i + j);
        auto task = cte_client->AsyncGetBlob(tag_id, blob_name, 0, io_size_, 0,
                                             get_ptr);
        task.Wait();
      }
    }

    auto end_time = high_resolution_clock::now();
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();

    CHI_IPC->FreeBuffer(put_shm);
    CHI_IPC->FreeBuffer(get_shm);
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
    long long min_time =
        *std::min_element(thread_times.begin(), thread_times.end());
    long long max_time =
        *std::max_element(thread_times.begin(), thread_times.end());
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
    double min_bw_bytes =
        min_time > 0
            ? (static_cast<double>(total_bytes) / (min_time / 1000000.0))
            : 0.0;
    double max_bw_bytes =
        max_time > 0
            ? (static_cast<double>(total_bytes) / (max_time / 1000000.0))
            : 0.0;
    double avg_bw_bytes =
        avg_time > 0
            ? (static_cast<double>(total_bytes) / (avg_time / 1000000.0))
            : 0.0;
    double agg_bw_bytes =
        avg_time > 0
            ? (static_cast<double>(aggregate_bytes) / (avg_time / 1000000.0))
            : 0.0;

    HLOG(kInfo, "");
    HLOG(kInfo, "=== {} Benchmark Results ===", operation);
    HLOG(kInfo, "Time (min): {} us ({} ms)", min_time, min_time / 1000.0);
    HLOG(kInfo, "Time (max): {} us ({} ms)", max_time, max_time / 1000.0);
    HLOG(kInfo, "Time (avg): {} us ({} ms)", avg_time, avg_time / 1000.0);
    HLOG(kInfo, "");
    HLOG(kInfo, "Bandwidth per thread (min): {} MB/s ({} bytes/s)", min_bw,
         min_bw_bytes);
    HLOG(kInfo, "Bandwidth per thread (max): {} MB/s ({} bytes/s)", max_bw,
         max_bw_bytes);
    HLOG(kInfo, "Bandwidth per thread (avg): {} MB/s ({} bytes/s)", avg_bw,
         avg_bw_bytes);
    HLOG(kInfo, "Aggregate bandwidth: {} MB/s ({} bytes/s)", agg_bw,
         agg_bw_bytes);
    HLOG(kInfo, "===========================");
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
    HLOG(kError, "Usage: {} <test_case> <num_threads> <depth> <io_size> <io_count>",
         argv[0]);
    HLOG(kError, "  test_case: Put, Get, or PutGet");
    HLOG(kError, "  num_threads: Number of worker threads (e.g., 4)");
    HLOG(kError, "  depth: Number of async requests per thread (e.g., 4)");
    HLOG(kError, "  io_size: Size of I/O operations (e.g., 1m, 4k, 1g)");
    HLOG(kError, "  io_count: Number of I/O operations per thread (e.g., 100)");
    HLOG(kError, "");
    HLOG(kError, "Environment variables:");
    HLOG(kError,
         "  CHIMAERA_WITH_RUNTIME: Set to '1', 'true', 'yes', or 'on' to "
         "initialize runtime");
    HLOG(kError, "                         Default: assumes runtime already initialized");
    return 1;
  }

  // Initialize Chimaera runtime and client
  HLOG(kInfo, "Initializing Chimaera runtime...");

  // Initialize Chimaera (client with embedded runtime)
  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Initialize CTE client
  if (!wrp_cte::core::WRP_CTE_CLIENT_INIT()) {
    HLOG(kError, "Failed to initialize CTE client");
    return 1;
  }

  HLOG(kInfo, "Runtime and client initialized successfully");

  // Small delay to ensure initialization is complete
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  std::string test_case = argv[1];
  size_t num_threads = std::stoull(argv[2]);
  int depth = std::atoi(argv[3]);
  chi::u64 io_size = ParseSize(argv[4]);
  int io_count = std::atoi(argv[5]);

  // Validate parameters
  if (num_threads == 0 || depth <= 0 || io_size == 0 || io_count <= 0) {
    HLOG(kError, "Invalid parameters");
    HLOG(kError, "  num_threads must be > 0");
    HLOG(kError, "  depth must be > 0");
    HLOG(kError, "  io_size must be > 0");
    HLOG(kError, "  io_count must be > 0");
    return 1;
  }

  // Run benchmark
  CTEBenchmark benchmark(num_threads, test_case, depth, io_size, io_count);
  benchmark.Run();

  return 0;
}
