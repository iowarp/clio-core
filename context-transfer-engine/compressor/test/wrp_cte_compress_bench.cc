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
 * CTE Compression Benchmark Application
 *
 * This benchmark measures the performance of Put, Get, and PutGet operations
 * with compression in the Content Transfer Engine (CTE). It emulates a
 * scientific workflow where compute phases generate data that is then
 * checkpointed via compression.
 *
 * Usage:
 *   wrp_cte_compress_bench <compress_type> <num_threads> <data_per_thread>
 *                          <transfer_size> <data_type> <distribution>
 *                          <compressibility> <test_case> <compute_phase_sec>
 *                          <checkpoint_interval>
 *
 * Parameters:
 *   compress_type: Compression algorithm (dynamic, zstd, lz4, zlib, snappy,
 *                  brotli, blosc2, bzip2, lzma, none)
 *   num_threads: Number of worker threads to spawn
 *   data_per_thread: Total data to generate per thread (supports k/m/g suffix)
 *   transfer_size: Size of each transfer/blob (supports k/m/g suffix)
 *   data_type: Data element type (char, int, float, double)
 *   distribution: Data distribution (normal, gamma, exponential, uniform)
 *   compressibility: Target compressibility (low, medium, high)
 *   test_case: Benchmark to conduct (put, get, putget)
 *   compute_phase_sec: Seconds to busy-wait simulating computation
 *   checkpoint_interval: Number of compute phases before blocking for Puts
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#ifndef _WIN32
#include <sched.h>
#endif
#include <string>
#include <thread>
#include <vector>

#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <hermes_shm/util/logging.h>

using namespace std::chrono;

namespace {

// Compression library IDs (from core_runtime.cc)
enum CompressionLib {
  kBrotli = 0,
  kBzip2 = 1,
  kBlosc2 = 2,
  kFpzip = 3,
  kLz4 = 4,
  kLzma = 5,
  kSnappy = 6,
  kSz3 = 7,
  kZfp = 8,
  kZlib = 9,
  kZstd = 10,
  kNone = -1
};

/**
 * Parse compression type string to library ID and dynamic mode
 * @param type_str Compression type string
 * @param dynamic_compress Output: 0=none, 1=static, 2=dynamic
 * @param compress_lib Output: Library ID for static compression
 * @return true if valid compression type
 */
bool ParseCompressionType(const std::string &type_str, int &dynamic_compress,
                          int &compress_lib) {
  std::string lower = type_str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower == "dynamic") {
    dynamic_compress = 2;
    compress_lib = kZstd;  // Default for dynamic
    return true;
  } else if (lower == "none") {
    dynamic_compress = 0;
    compress_lib = kNone;
    return true;
  } else if (lower == "zstd") {
    dynamic_compress = 1;
    compress_lib = kZstd;
    return true;
  } else if (lower == "lz4") {
    dynamic_compress = 1;
    compress_lib = kLz4;
    return true;
  } else if (lower == "zlib") {
    dynamic_compress = 1;
    compress_lib = kZlib;
    return true;
  } else if (lower == "snappy") {
    dynamic_compress = 1;
    compress_lib = kSnappy;
    return true;
  } else if (lower == "brotli") {
    dynamic_compress = 1;
    compress_lib = kBrotli;
    return true;
  } else if (lower == "blosc2") {
    dynamic_compress = 1;
    compress_lib = kBlosc2;
    return true;
  } else if (lower == "bzip2") {
    dynamic_compress = 1;
    compress_lib = kBzip2;
    return true;
  } else if (lower == "lzma") {
    dynamic_compress = 1;
    compress_lib = kLzma;
    return true;
  }

  return false;
}

/**
 * Get compression type name for display
 */
std::string GetCompressionName(int dynamic_compress, int compress_lib) {
  if (dynamic_compress == 0) return "none";
  if (dynamic_compress == 2) return "dynamic";

  switch (compress_lib) {
    case kZstd: return "zstd";
    case kLz4: return "lz4";
    case kZlib: return "zlib";
    case kSnappy: return "snappy";
    case kBrotli: return "brotli";
    case kBlosc2: return "blosc2";
    case kBzip2: return "bzip2";
    case kLzma: return "lzma";
    default: return "unknown";
  }
}

/**
 * Parse size string with k/K, m/M, g/G suffixes
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
    case 'k': multiplier = 1024; break;
    case 'm': multiplier = 1024 * 1024; break;
    case 'g': multiplier = 1024ULL * 1024 * 1024; break;
    default: multiplier = 1; break;
  }

  return static_cast<chi::u64>(size * multiplier);
}

/**
 * Convert bytes to human-readable string
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
 * Calculate bandwidth in MB/s
 */
double CalcBandwidth(chi::u64 total_bytes, double microseconds) {
  if (microseconds <= 0.0) return 0.0;
  double seconds = microseconds / 1000000.0;
  double megabytes = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  return megabytes / seconds;
}

/**
 * Pin current thread to a specific CPU core
 * @param core_id CPU core to pin to
 * @return true if successful
 */
bool PinThreadToCore(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  return result == 0;
}

/**
 * Busy wait for specified duration (simulates computation)
 * @param seconds Duration to busy wait
 */
void BusyWait(double seconds) {
  if (seconds <= 0.0) return;

  auto start = high_resolution_clock::now();
  auto target_duration = duration<double>(seconds);

  volatile double dummy = 0.0;
  while (high_resolution_clock::now() - start < target_duration) {
    // Perform some computation to prevent optimization
    for (int i = 0; i < 1000; ++i) {
      dummy += std::sin(static_cast<double>(i)) * std::cos(static_cast<double>(i));
    }
  }
}

}  // namespace

/**
 * Data generator with various distributions and compressibility levels
 */
class DataGenerator {
 public:
  enum class DataType { kChar, kInt, kFloat, kDouble };
  enum class Distribution { kNormal, kGamma, kExponential, kUniform };
  enum class Compressibility { kLow, kMedium, kHigh };

  DataGenerator(DataType data_type, Distribution distribution,
                Compressibility compressibility, unsigned int seed)
      : data_type_(data_type),
        distribution_(distribution),
        compressibility_(compressibility),
        gen_(seed) {}

  /**
   * Get data element size in bytes
   */
  size_t GetElementSize() const {
    switch (data_type_) {
      case DataType::kChar: return sizeof(char);
      case DataType::kInt: return sizeof(int);
      case DataType::kFloat: return sizeof(float);
      case DataType::kDouble: return sizeof(double);
      default: return 1;
    }
  }

  /**
   * Fill buffer with generated data
   * @param buffer Pointer to buffer
   * @param size Size in bytes
   */
  void FillBuffer(void *buffer, size_t size) {
    switch (data_type_) {
      case DataType::kChar:
        FillTypedBuffer<char>(static_cast<char *>(buffer), size);
        break;
      case DataType::kInt:
        FillTypedBuffer<int>(static_cast<int *>(buffer), size / sizeof(int));
        break;
      case DataType::kFloat:
        FillTypedBuffer<float>(static_cast<float *>(buffer), size / sizeof(float));
        break;
      case DataType::kDouble:
        FillTypedBuffer<double>(static_cast<double *>(buffer), size / sizeof(double));
        break;
    }
  }

 private:
  template <typename T>
  void FillTypedBuffer(T *buffer, size_t count) {
    // Adjust distribution parameters based on compressibility
    double repeat_factor = GetRepeatFactor();

    for (size_t i = 0; i < count; ++i) {
      double value = GenerateValue();

      // Apply repetition for compressibility
      if (compressibility_ != Compressibility::kLow) {
        // Quantize values to create repetition
        int quantize_levels = (compressibility_ == Compressibility::kHigh) ? 4 : 16;
        value = std::round(value * quantize_levels) / quantize_levels;
      }

      // Add periodic repetition for high compressibility
      if (compressibility_ == Compressibility::kHigh && i > 0) {
        size_t repeat_period = static_cast<size_t>(count * 0.1);  // 10% period
        if (repeat_period > 0 && (i % repeat_period) < repeat_period / 2) {
          buffer[i] = buffer[i % (repeat_period / 2)];
          continue;
        }
      }

      buffer[i] = static_cast<T>(value);
    }
  }

  double GetRepeatFactor() const {
    switch (compressibility_) {
      case Compressibility::kLow: return 0.0;
      case Compressibility::kMedium: return 0.3;
      case Compressibility::kHigh: return 0.7;
      default: return 0.0;
    }
  }

  double GenerateValue() {
    switch (distribution_) {
      case Distribution::kNormal: {
        std::normal_distribution<double> dist(0.0, 1.0);
        return dist(gen_);
      }
      case Distribution::kGamma: {
        std::gamma_distribution<double> dist(2.0, 2.0);
        return dist(gen_);
      }
      case Distribution::kExponential: {
        std::exponential_distribution<double> dist(1.0);
        return dist(gen_);
      }
      case Distribution::kUniform:
      default: {
        std::uniform_real_distribution<double> dist(-100.0, 100.0);
        return dist(gen_);
      }
    }
  }

  DataType data_type_;
  Distribution distribution_;
  Compressibility compressibility_;
  std::mt19937 gen_;
};

/**
 * Parse data type string
 */
DataGenerator::DataType ParseDataType(const std::string &type_str) {
  std::string lower = type_str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower == "char") return DataGenerator::DataType::kChar;
  if (lower == "int") return DataGenerator::DataType::kInt;
  if (lower == "float") return DataGenerator::DataType::kFloat;
  if (lower == "double") return DataGenerator::DataType::kDouble;

  HLOG(kWarning, "Unknown data type '{}', using float", type_str);
  return DataGenerator::DataType::kFloat;
}

/**
 * Parse distribution string
 */
DataGenerator::Distribution ParseDistribution(const std::string &dist_str) {
  std::string lower = dist_str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower == "normal") return DataGenerator::Distribution::kNormal;
  if (lower == "gamma") return DataGenerator::Distribution::kGamma;
  if (lower == "exponential") return DataGenerator::Distribution::kExponential;
  if (lower == "uniform") return DataGenerator::Distribution::kUniform;

  HLOG(kWarning, "Unknown distribution '{}', using normal", dist_str);
  return DataGenerator::Distribution::kNormal;
}

/**
 * Parse compressibility string
 */
DataGenerator::Compressibility ParseCompressibility(const std::string &comp_str) {
  std::string lower = comp_str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower == "low") return DataGenerator::Compressibility::kLow;
  if (lower == "medium") return DataGenerator::Compressibility::kMedium;
  if (lower == "high") return DataGenerator::Compressibility::kHigh;

  HLOG(kWarning, "Unknown compressibility '{}', using medium", comp_str);
  return DataGenerator::Compressibility::kMedium;
}

/**
 * Get data type name for display
 */
std::string GetDataTypeName(DataGenerator::DataType type) {
  switch (type) {
    case DataGenerator::DataType::kChar: return "char";
    case DataGenerator::DataType::kInt: return "int";
    case DataGenerator::DataType::kFloat: return "float";
    case DataGenerator::DataType::kDouble: return "double";
    default: return "unknown";
  }
}

/**
 * Get distribution name for display
 */
std::string GetDistributionName(DataGenerator::Distribution dist) {
  switch (dist) {
    case DataGenerator::Distribution::kNormal: return "normal";
    case DataGenerator::Distribution::kGamma: return "gamma";
    case DataGenerator::Distribution::kExponential: return "exponential";
    case DataGenerator::Distribution::kUniform: return "uniform";
    default: return "unknown";
  }
}

/**
 * Get compressibility name for display
 */
std::string GetCompressibilityName(DataGenerator::Compressibility comp) {
  switch (comp) {
    case DataGenerator::Compressibility::kLow: return "low";
    case DataGenerator::Compressibility::kMedium: return "medium";
    case DataGenerator::Compressibility::kHigh: return "high";
    default: return "unknown";
  }
}

/**
 * Main compression benchmark class
 */
class CTECompressBenchmark {
 public:
  CTECompressBenchmark(int dynamic_compress, int compress_lib,
                       size_t num_threads, chi::u64 data_per_thread,
                       chi::u64 transfer_size, DataGenerator::DataType data_type,
                       DataGenerator::Distribution distribution,
                       DataGenerator::Compressibility compressibility,
                       const std::string &test_case, double compute_phase_sec,
                       int checkpoint_interval)
      : dynamic_compress_(dynamic_compress),
        compress_lib_(compress_lib),
        num_threads_(num_threads),
        data_per_thread_(data_per_thread),
        transfer_size_(transfer_size),
        data_type_(data_type),
        distribution_(distribution),
        compressibility_(compressibility),
        test_case_(test_case),
        compute_phase_sec_(compute_phase_sec),
        checkpoint_interval_(checkpoint_interval) {
    // Calculate number of transfers per thread
    transfers_per_thread_ = data_per_thread_ / transfer_size_;
    if (transfers_per_thread_ == 0) transfers_per_thread_ = 1;
  }

  /**
   * Run the benchmark
   */
  void Run() {
    PrintBenchmarkInfo();

    std::string lower_case = test_case_;
    std::transform(lower_case.begin(), lower_case.end(), lower_case.begin(),
                   ::tolower);

    if (lower_case == "put") {
      RunPutBenchmark();
    } else if (lower_case == "get") {
      RunGetBenchmark();
    } else if (lower_case == "putget") {
      RunPutGetBenchmark();
    } else {
      HLOG(kError, "Unknown test case: {}", test_case_);
      HLOG(kError, "Valid options: put, get, putget");
    }
  }

 private:
  void PrintBenchmarkInfo() {
    HLOG(kInfo, "=== CTE Compression Benchmark ===");
    HLOG(kInfo, "Compression: {}", GetCompressionName(dynamic_compress_, compress_lib_));
    HLOG(kInfo, "Worker threads: {}", num_threads_);
    HLOG(kInfo, "Data per thread: {}", FormatSize(data_per_thread_));
    HLOG(kInfo, "Transfer size: {}", FormatSize(transfer_size_));
    HLOG(kInfo, "Transfers per thread: {}", transfers_per_thread_);
    HLOG(kInfo, "Data type: {}", GetDataTypeName(data_type_));
    HLOG(kInfo, "Distribution: {}", GetDistributionName(distribution_));
    HLOG(kInfo, "Compressibility: {}", GetCompressibilityName(compressibility_));
    HLOG(kInfo, "Test case: {}", test_case_);
    HLOG(kInfo, "Compute phase: {} seconds", compute_phase_sec_);
    HLOG(kInfo, "Checkpoint interval: {} phases", checkpoint_interval_);
    HLOG(kInfo, "Total data (all threads): {}",
         FormatSize(data_per_thread_ * num_threads_));
    HLOG(kInfo, "===================================");
  }

  /**
   * Create compression context with current settings
   */
  wrp_cte::core::Context CreateCompressionContext() {
    wrp_cte::core::Context ctx;
    ctx.dynamic_compress_ = dynamic_compress_;
    ctx.compress_lib_ = compress_lib_;
    ctx.max_performance_ = false;  // Optimize for ratio
    return ctx;
  }

  /**
   * Worker thread for Put benchmark with compute phases
   */
  void PutWorkerThread(size_t thread_id, std::atomic<bool> &error_flag,
                       std::vector<long long> &thread_times,
                       std::vector<double> &compression_ratios) {
    // Pin thread to core
    if (!PinThreadToCore(static_cast<int>(thread_id))) {
      HLOG(kWarning, "Failed to pin thread {} to core", thread_id);
    }

    // Create data generator with thread-specific seed
    DataGenerator generator(data_type_, distribution_, compressibility_,
                            static_cast<unsigned int>(thread_id * 12345 + 67890));

    // Allocate data buffer
    std::vector<char> data(transfer_size_);

    // Allocate shared memory buffer
    auto shm_buffer = CHI_IPC->AllocateBuffer(transfer_size_);
    if (shm_buffer.IsNull()) {
      HLOG(kError, "Failed to allocate shared memory for thread {}", thread_id);
      error_flag.store(true);
      return;
    }
    hipc::ShmPtr<> shm_ptr = shm_buffer.shm_.template Cast<void>();

    // Create compression context
    wrp_cte::core::Context ctx = CreateCompressionContext();

    // Track pending tasks for checkpointing
    std::vector<chi::Future<wrp_cte::core::PutBlobTask>> pending_tasks;
    int phases_since_checkpoint = 0;

    chi::u64 total_original_size = 0;
    chi::u64 total_compressed_size = 0;

    auto start_time = high_resolution_clock::now();

    for (size_t i = 0; i < transfers_per_thread_; ++i) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      // Simulate compute phase
      BusyWait(compute_phase_sec_);
      phases_since_checkpoint++;

      // Generate data
      generator.FillBuffer(data.data(), transfer_size_);
      std::memcpy(shm_buffer.ptr_, data.data(), transfer_size_);

      // Create tag and blob names
      std::string tag_name = "compress_tag_t" + std::to_string(thread_id) +
                             "_i" + std::to_string(i);
      wrp_cte::core::Tag tag(tag_name);
      std::string blob_name = "blob_0";

      // Submit async Put with compression
      auto task = tag.AsyncPutBlob(blob_name, shm_ptr, transfer_size_, 0, 0.8f, ctx);
      pending_tasks.push_back(task);

      total_original_size += transfer_size_;

      // Checkpoint: wait for all pending Puts to complete
      if (phases_since_checkpoint >= checkpoint_interval_ ||
          i == transfers_per_thread_ - 1) {
        for (auto &t : pending_tasks) {
          t.Wait();
          // Track compressed size if available (approximate)
          total_compressed_size += transfer_size_;  // Placeholder
        }
        pending_tasks.clear();
        phases_since_checkpoint = 0;
      }
    }

    auto end_time = high_resolution_clock::now();
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();

    // Calculate compression ratio (approximate)
    compression_ratios[thread_id] =
        (total_compressed_size > 0)
            ? static_cast<double>(total_original_size) / total_compressed_size
            : 1.0;

    // Free shared memory buffer
    CHI_IPC->FreeBuffer(shm_buffer);
  }

  void RunPutBenchmark() {
    std::vector<std::thread> threads;
    std::vector<long long> thread_times(num_threads_);
    std::vector<double> compression_ratios(num_threads_);
    std::atomic<bool> error_flag{false};

    // Spawn worker threads
    for (size_t i = 0; i < num_threads_; ++i) {
      threads.emplace_back(&CTECompressBenchmark::PutWorkerThread, this, i,
                           std::ref(error_flag), std::ref(thread_times),
                           std::ref(compression_ratios));
    }

    // Wait for all threads
    for (auto &thread : threads) {
      thread.join();
    }

    PrintResults("Put", thread_times, compression_ratios);
  }

  /**
   * Worker thread for Get benchmark
   */
  void GetWorkerThread(size_t thread_id, std::atomic<bool> &error_flag,
                       std::vector<long long> &thread_times,
                       std::vector<double> &compression_ratios) {
    // Pin thread to core
    if (!PinThreadToCore(static_cast<int>(thread_id))) {
      HLOG(kWarning, "Failed to pin thread {} to core", thread_id);
    }

    // Create data generator
    DataGenerator generator(data_type_, distribution_, compressibility_,
                            static_cast<unsigned int>(thread_id * 12345 + 67890));

    // Allocate buffers
    std::vector<char> put_data(transfer_size_);
    std::vector<char> get_data(transfer_size_);

    // Create compression context
    wrp_cte::core::Context ctx = CreateCompressionContext();

    // First, populate data with Put operations
    HLOG(kInfo, "Thread {}: Populating data for Get benchmark...", thread_id);

    for (size_t i = 0; i < transfers_per_thread_; ++i) {
      generator.FillBuffer(put_data.data(), transfer_size_);

      std::string tag_name = "compress_get_tag_t" + std::to_string(thread_id) +
                             "_i" + std::to_string(i);
      wrp_cte::core::Tag tag(tag_name);
      std::string blob_name = "blob_0";

      tag.PutBlob(blob_name, put_data.data(), transfer_size_, 0, 0.8f, ctx);
    }

    // Now benchmark Get operations
    auto start_time = high_resolution_clock::now();

    for (size_t i = 0; i < transfers_per_thread_; ++i) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      // Simulate compute phase
      BusyWait(compute_phase_sec_);

      std::string tag_name = "compress_get_tag_t" + std::to_string(thread_id) +
                             "_i" + std::to_string(i);
      wrp_cte::core::Tag tag(tag_name);
      std::string blob_name = "blob_0";

      tag.GetBlob(blob_name, get_data.data(), transfer_size_);
    }

    auto end_time = high_resolution_clock::now();
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();
    compression_ratios[thread_id] = 1.0;  // Decompression ratio is 1:1
  }

  void RunGetBenchmark() {
    HLOG(kInfo, "Populating data for Get benchmark...");

    std::vector<std::thread> threads;
    std::vector<long long> thread_times(num_threads_);
    std::vector<double> compression_ratios(num_threads_);
    std::atomic<bool> error_flag{false};

    // Spawn worker threads
    for (size_t i = 0; i < num_threads_; ++i) {
      threads.emplace_back(&CTECompressBenchmark::GetWorkerThread, this, i,
                           std::ref(error_flag), std::ref(thread_times),
                           std::ref(compression_ratios));
    }

    // Wait for all threads
    for (auto &thread : threads) {
      thread.join();
    }

    PrintResults("Get", thread_times, compression_ratios);
  }

  /**
   * Worker thread for PutGet benchmark
   */
  void PutGetWorkerThread(size_t thread_id, std::atomic<bool> &error_flag,
                          std::vector<long long> &thread_times,
                          std::vector<double> &compression_ratios) {
    // Pin thread to core
    if (!PinThreadToCore(static_cast<int>(thread_id))) {
      HLOG(kWarning, "Failed to pin thread {} to core", thread_id);
    }

    // Create data generator
    DataGenerator generator(data_type_, distribution_, compressibility_,
                            static_cast<unsigned int>(thread_id * 12345 + 67890));

    // Allocate buffers
    std::vector<char> data(transfer_size_);
    std::vector<char> get_data(transfer_size_);

    // Allocate shared memory buffer
    auto shm_buffer = CHI_IPC->AllocateBuffer(transfer_size_);
    if (shm_buffer.IsNull()) {
      HLOG(kError, "Failed to allocate shared memory for thread {}", thread_id);
      error_flag.store(true);
      return;
    }
    hipc::ShmPtr<> shm_ptr = shm_buffer.shm_.template Cast<void>();

    // Create compression context
    wrp_cte::core::Context ctx = CreateCompressionContext();

    // Track pending tasks
    std::vector<chi::Future<wrp_cte::core::PutBlobTask>> pending_tasks;
    int phases_since_checkpoint = 0;

    auto start_time = high_resolution_clock::now();

    for (size_t i = 0; i < transfers_per_thread_; ++i) {
      if (error_flag.load(std::memory_order_relaxed)) {
        break;
      }

      // Simulate compute phase
      BusyWait(compute_phase_sec_);
      phases_since_checkpoint++;

      // Generate data
      generator.FillBuffer(data.data(), transfer_size_);
      std::memcpy(shm_buffer.ptr_, data.data(), transfer_size_);

      std::string tag_name = "compress_putget_tag_t" + std::to_string(thread_id) +
                             "_i" + std::to_string(i);
      wrp_cte::core::Tag tag(tag_name);
      std::string blob_name = "blob_0";

      // Async Put
      auto task = tag.AsyncPutBlob(blob_name, shm_ptr, transfer_size_, 0, 0.8f, ctx);
      pending_tasks.push_back(task);

      // Checkpoint
      if (phases_since_checkpoint >= checkpoint_interval_ ||
          i == transfers_per_thread_ - 1) {
        for (auto &t : pending_tasks) {
          t.Wait();
        }

        // Read back all checkpointed data
        for (size_t j = i - pending_tasks.size() + 1; j <= i; ++j) {
          std::string read_tag_name = "compress_putget_tag_t" +
                                      std::to_string(thread_id) + "_i" +
                                      std::to_string(j);
          wrp_cte::core::Tag read_tag(read_tag_name);
          read_tag.GetBlob(blob_name, get_data.data(), transfer_size_);
        }

        pending_tasks.clear();
        phases_since_checkpoint = 0;
      }
    }

    auto end_time = high_resolution_clock::now();
    thread_times[thread_id] =
        duration_cast<microseconds>(end_time - start_time).count();
    compression_ratios[thread_id] = 1.0;

    // Free shared memory buffer
    CHI_IPC->FreeBuffer(shm_buffer);
  }

  void RunPutGetBenchmark() {
    std::vector<std::thread> threads;
    std::vector<long long> thread_times(num_threads_);
    std::vector<double> compression_ratios(num_threads_);
    std::atomic<bool> error_flag{false};

    // Spawn worker threads
    for (size_t i = 0; i < num_threads_; ++i) {
      threads.emplace_back(&CTECompressBenchmark::PutGetWorkerThread, this, i,
                           std::ref(error_flag), std::ref(thread_times),
                           std::ref(compression_ratios));
    }

    // Wait for all threads
    for (auto &thread : threads) {
      thread.join();
    }

    PrintResults("PutGet", thread_times, compression_ratios);
  }

  void PrintResults(const std::string &operation,
                    const std::vector<long long> &thread_times,
                    const std::vector<double> &compression_ratios) {
    // Calculate time statistics
    long long min_time = *std::min_element(thread_times.begin(), thread_times.end());
    long long max_time = *std::max_element(thread_times.begin(), thread_times.end());
    long long sum_time = 0;
    for (auto t : thread_times) {
      sum_time += t;
    }
    double avg_time = static_cast<double>(sum_time) / num_threads_;

    // Calculate compression ratio statistics
    double avg_ratio = 0.0;
    for (auto r : compression_ratios) {
      avg_ratio += r;
    }
    avg_ratio /= num_threads_;

    chi::u64 total_bytes = data_per_thread_;
    chi::u64 aggregate_bytes = total_bytes * num_threads_;

    double min_bw = CalcBandwidth(total_bytes, min_time);
    double max_bw = CalcBandwidth(total_bytes, max_time);
    double avg_bw = CalcBandwidth(total_bytes, avg_time);
    double agg_bw = CalcBandwidth(aggregate_bytes, avg_time);

    HLOG(kInfo, "");
    HLOG(kInfo, "=== {} Compression Benchmark Results ===", operation);
    HLOG(kInfo, "Time (min): {} ms", min_time / 1000.0);
    HLOG(kInfo, "Time (max): {} ms", max_time / 1000.0);
    HLOG(kInfo, "Time (avg): {} ms", avg_time / 1000.0);
    HLOG(kInfo, "");
    HLOG(kInfo, "Bandwidth per thread (min): {} MB/s", min_bw);
    HLOG(kInfo, "Bandwidth per thread (max): {} MB/s", max_bw);
    HLOG(kInfo, "Bandwidth per thread (avg): {} MB/s", avg_bw);
    HLOG(kInfo, "Aggregate bandwidth: {} MB/s", agg_bw);
    HLOG(kInfo, "");
    HLOG(kInfo, "Compression ratio (avg): {}:1", avg_ratio);
    HLOG(kInfo, "=============================================");
  }

  int dynamic_compress_;
  int compress_lib_;
  size_t num_threads_;
  chi::u64 data_per_thread_;
  chi::u64 transfer_size_;
  size_t transfers_per_thread_;
  DataGenerator::DataType data_type_;
  DataGenerator::Distribution distribution_;
  DataGenerator::Compressibility compressibility_;
  std::string test_case_;
  double compute_phase_sec_;
  int checkpoint_interval_;
};

void PrintUsage(const char *program) {
  HLOG(kError, "Usage: {} <compress_type> <num_threads> <data_per_thread> <transfer_size> <data_type> <distribution> <compressibility> <test_case> <compute_phase_sec> <checkpoint_interval>",
       program);
  HLOG(kError, "");
  HLOG(kError, "Parameters:");
  HLOG(kError, "  compress_type: Compression algorithm");
  HLOG(kError, "                 (dynamic, zstd, lz4, zlib, snappy, brotli, blosc2, bzip2, lzma, none)");
  HLOG(kError, "  num_threads: Number of worker threads (e.g., 4)");
  HLOG(kError, "  data_per_thread: Total data per thread (e.g., 1g, 100m)");
  HLOG(kError, "  transfer_size: Size of each transfer (e.g., 1m, 4k)");
  HLOG(kError, "  data_type: Data element type (char, int, float, double)");
  HLOG(kError, "  distribution: Data distribution (normal, gamma, exponential, uniform)");
  HLOG(kError, "  compressibility: Target compressibility (low, medium, high)");
  HLOG(kError, "  test_case: Benchmark to run (put, get, putget)");
  HLOG(kError, "  compute_phase_sec: Seconds to busy-wait per compute phase");
  HLOG(kError, "  checkpoint_interval: Phases between checkpoints");
  HLOG(kError, "");
  HLOG(kError, "Example:");
  HLOG(kError, "  {} zstd 4 1g 1m float normal medium put 0.1 10", program);
}

int main(int argc, char **argv) {
  if (argc != 11) {
    PrintUsage(argv[0]);
    return 1;
  }

  // Parse compression type
  int dynamic_compress, compress_lib;
  if (!ParseCompressionType(argv[1], dynamic_compress, compress_lib)) {
    HLOG(kError, "Invalid compression type: {}", argv[1]);
    PrintUsage(argv[0]);
    return 1;
  }

  // Parse other arguments
  size_t num_threads = std::stoull(argv[2]);
  chi::u64 data_per_thread = ParseSize(argv[3]);
  chi::u64 transfer_size = ParseSize(argv[4]);
  DataGenerator::DataType data_type = ParseDataType(argv[5]);
  DataGenerator::Distribution distribution = ParseDistribution(argv[6]);
  DataGenerator::Compressibility compressibility = ParseCompressibility(argv[7]);
  std::string test_case = argv[8];
  double compute_phase_sec = std::stod(argv[9]);
  int checkpoint_interval = std::atoi(argv[10]);

  // Validate parameters
  if (num_threads == 0 || data_per_thread == 0 || transfer_size == 0 ||
      checkpoint_interval <= 0) {
    HLOG(kError, "Invalid parameters");
    HLOG(kError, "  num_threads must be > 0");
    HLOG(kError, "  data_per_thread must be > 0");
    HLOG(kError, "  transfer_size must be > 0");
    HLOG(kError, "  checkpoint_interval must be > 0");
    return 1;
  }

  // Initialize Chimaera runtime
  HLOG(kInfo, "Initializing Chimaera runtime...");

  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true)) {
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
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Run benchmark
  CTECompressBenchmark benchmark(dynamic_compress, compress_lib, num_threads,
                                 data_per_thread, transfer_size, data_type,
                                 distribution, compressibility, test_case,
                                 compute_phase_sec, checkpoint_interval);
  benchmark.Run();

  return 0;
}
