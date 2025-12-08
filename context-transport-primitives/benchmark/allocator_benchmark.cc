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
 * Allocator Benchmark Suite
 *
 * This benchmark suite measures the performance of different allocator implementations
 * under various workload patterns. It supports:
 * - MultiProcessAllocator (mp): Thread-safe, multi-process allocator
 * - BuddyAllocator (buddy): Buddy system allocator
 * - malloc (malloc): Standard C library malloc
 *
 * Usage:
 *   allocator_benchmark <allocator_type> <num_threads> <min_size> <max_size> [duration_sec]
 *
 * Parameters:
 *   allocator_type: "mp", "buddy", or "malloc"
 *   num_threads: Number of concurrent threads (1-64)
 *   min_size: Minimum allocation size (supports K, M, G suffixes, e.g., "1K", "4K")
 *   max_size: Maximum allocation size (supports K, M, G suffixes, e.g., "1M", "16M")
 *   duration_sec: Duration to run benchmark in seconds (default: 10)
 *
 * Example:
 *   allocator_benchmark mp 8 4K 1M 30
 *   This runs the MultiProcessAllocator with 8 threads, allocation sizes between
 *   4KB and 1MB, for 30 seconds.
 */

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <cstring>
#include <iomanip>

#include "hermes_shm/memory/allocator/mp_allocator.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/util/config_parse.h"

namespace bench {

/**
 * Strong type for minimum allocation size
 */
struct MinAllocSize {
  size_t value;
  explicit MinAllocSize(size_t val) : value(val) {}
};

/**
 * Strong type for maximum allocation size
 */
struct MaxAllocSize {
  size_t value;
  explicit MaxAllocSize(size_t val) : value(val) {}
};

/**
 * Size range configuration to prevent parameter swapping
 */
struct SizeRange {
  size_t min_size;
  size_t max_size;

  /**
   * Constructor using strong types
   * @param min Minimum allocation size
   * @param max Maximum allocation size
   */
  SizeRange(const MinAllocSize& min, const MaxAllocSize& max)
      : min_size(min.value), max_size(max.value) {}

  /**
   * Constructor for convenience (direct size values)
   * @param min Minimum allocation size
   * @param max Maximum allocation size
   */
  SizeRange(size_t min, size_t max) : min_size(min), max_size(max) {}
};

/**
 * Print usage information and exit
 */
void PrintUsage(const char* program_name) {
  std::cerr << "Usage: " << program_name
            << " <allocator_type> <num_threads> <min_size> <max_size> [duration_sec]\n\n"
            << "Parameters:\n"
            << "  allocator_type: Allocator to benchmark\n"
            << "                  - mp: MultiProcessAllocator\n"
            << "                  - buddy: BuddyAllocator\n"
            << "                  - malloc: Standard C malloc\n"
            << "  num_threads:    Number of concurrent threads (1-64)\n"
            << "  min_size:       Minimum allocation size (supports K, M, G suffixes)\n"
            << "  max_size:       Maximum allocation size (supports K, M, G suffixes)\n"
            << "  duration_sec:   Duration to run benchmark in seconds (default: 10)\n\n"
            << "Examples:\n"
            << "  " << program_name << " mp 8 4K 1M 30\n"
            << "  " << program_name << " buddy 4 1K 64K\n"
            << "  " << program_name << " malloc 16 128 16K 60\n";
  exit(1);
}

/**
 * Thread-safe counter for tracking operations
 */
class OperationCounter {
 private:
  std::atomic<uint64_t> alloc_count_;
  std::atomic<uint64_t> free_count_;

 public:
  OperationCounter() : alloc_count_(0), free_count_(0) {}

  void RecordAlloc() { alloc_count_.fetch_add(1, std::memory_order_relaxed); }
  void RecordFree() { free_count_.fetch_add(1, std::memory_order_relaxed); }

  uint64_t GetAllocCount() const { return alloc_count_.load(std::memory_order_relaxed); }
  uint64_t GetFreeCount() const { return free_count_.load(std::memory_order_relaxed); }
  uint64_t GetTotalOps() const { return GetAllocCount() + GetFreeCount(); }
};

/**
 * Benchmark worker for MultiProcessAllocator or BuddyAllocator
 *
 * This worker performs random allocations/deallocations for a specified duration.
 * It maintains a pool of allocated pointers and randomly chooses to either:
 * - Allocate new memory (if pool not full)
 * - Free existing memory (if pool not empty)
 */
template<typename AllocT>
class AllocatorBenchmarkWorker {
 private:
  AllocT* alloc_;
  size_t min_size_;
  size_t max_size_;
  int duration_sec_;
  OperationCounter* counter_;
  std::mt19937 rng_;

 public:
  AllocatorBenchmarkWorker(AllocT* alloc, size_t min_size, size_t max_size,
                          int duration_sec, OperationCounter* counter)
    : alloc_(alloc), min_size_(min_size), max_size_(max_size),
      duration_sec_(duration_sec), counter_(counter),
      rng_(std::random_device{}()) {}

  /**
   * Run the benchmark workload
   *
   * Performs random allocations and deallocations until the duration expires.
   * Maintains a pool of up to 1000 allocations.
   */
  void Run() {
    const size_t kMaxAllocations = 1000;
    std::vector<hipc::FullPtr<char>> ptrs;
    ptrs.reserve(kMaxAllocations);

    std::uniform_int_distribution<size_t> size_dist(min_size_, max_size_);
    std::uniform_int_distribution<int> action_dist(0, 1);

    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_sec_);

    while (std::chrono::steady_clock::now() < end_time) {
      // Decide whether to allocate or free
      bool should_allocate = ptrs.empty() ||
                            (ptrs.size() < kMaxAllocations && action_dist(rng_) == 0);

      if (should_allocate) {
        // Allocate random size
        size_t alloc_size = size_dist(rng_);
        auto ptr = alloc_->template Allocate<char>(alloc_size);

        if (!ptr.IsNull()) {
          // Write to the allocation to ensure it's valid
          std::memset(ptr.ptr_, static_cast<unsigned char>(ptrs.size() & 0xFF), alloc_size);
          ptrs.push_back(ptr);
          counter_->RecordAlloc();
        }
      } else {
        // Free a random allocation
        std::uniform_int_distribution<size_t> index_dist(0, ptrs.size() - 1);
        size_t index = index_dist(rng_);

        alloc_->Free(ptrs[index]);
        ptrs[index] = ptrs.back();
        ptrs.pop_back();
        counter_->RecordFree();
      }
    }

    // Clean up remaining allocations
    for (auto& ptr : ptrs) {
      alloc_->Free(ptr);
      counter_->RecordFree();
    }
  }
};

/**
 * Benchmark worker for standard malloc
 *
 * Similar to AllocatorBenchmarkWorker but uses malloc/free instead of
 * HSHM allocator APIs.
 */
class MallocBenchmarkWorker {
 private:
  size_t min_size_;
  size_t max_size_;
  int duration_sec_;
  OperationCounter* counter_;
  std::mt19937 rng_;

 public:
  MallocBenchmarkWorker(size_t min_size, size_t max_size,
                       int duration_sec, OperationCounter* counter)
    : min_size_(min_size), max_size_(max_size),
      duration_sec_(duration_sec), counter_(counter),
      rng_(std::random_device{}()) {}

  /**
   * Run the benchmark workload using malloc/free
   */
  void Run() {
    const size_t kMaxAllocations = 1000;
    std::vector<void*> ptrs;
    ptrs.reserve(kMaxAllocations);

    std::uniform_int_distribution<size_t> size_dist(min_size_, max_size_);
    std::uniform_int_distribution<int> action_dist(0, 1);

    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration_sec_);

    while (std::chrono::steady_clock::now() < end_time) {
      // Decide whether to allocate or free
      bool should_allocate = ptrs.empty() ||
                            (ptrs.size() < kMaxAllocations && action_dist(rng_) == 0);

      if (should_allocate) {
        // Allocate random size
        size_t alloc_size = size_dist(rng_);
        void* ptr = malloc(alloc_size);

        if (ptr != nullptr) {
          // Write to the allocation to ensure it's valid
          std::memset(ptr, static_cast<unsigned char>(ptrs.size() & 0xFF), alloc_size);
          ptrs.push_back(ptr);
          counter_->RecordAlloc();
        }
      } else {
        // Free a random allocation
        std::uniform_int_distribution<size_t> index_dist(0, ptrs.size() - 1);
        size_t index = index_dist(rng_);

        free(ptrs[index]);
        ptrs[index] = ptrs.back();
        ptrs.pop_back();
        counter_->RecordFree();
      }
    }

    // Clean up remaining allocations
    for (auto* ptr : ptrs) {
      free(ptr);
      counter_->RecordFree();
    }
  }
};

/**
 * Format a number with thousands separators
 */
std::string FormatNumber(uint64_t num) {
  std::string str = std::to_string(num);
  std::string result;
  int count = 0;

  for (auto it = str.rbegin(); it != str.rend(); ++it) {
    if (count == 3) {
      result = ',' + result;
      count = 0;
    }
    result = *it + result;
    count++;
  }

  return result;
}

/**
 * Format a size with appropriate units (B, KB, MB, GB)
 */
std::string FormatSize(size_t size) {
  const size_t KB = 1024UL;
  const size_t MB = 1024UL * 1024UL;
  const size_t GB = 1024UL * 1024UL * 1024UL;

  if (size < KB) {
    return std::to_string(size) + "B";
  }
  if (size < MB) {
    return std::to_string(size / KB) + "KB";
  }
  if (size < GB) {
    return std::to_string(size / MB) + "MB";
  }
  return std::to_string(size / GB) + "GB";
}

/**
 * Run benchmark with MultiProcessAllocator
 * @param num_threads Number of threads to use
 * @param size_range Size range configuration (min_size, max_size)
 * @param duration_sec Duration to run benchmark in seconds
 */
void BenchmarkMultiProcessAllocator(int num_threads,
                                   const SizeRange& size_range,
                                   int duration_sec) {
  size_t min_size = size_range.min_size;
  size_t max_size = size_range.max_size;
  // Initialize backend with sufficient memory
  hipc::PosixShmMmap backend;
  size_t heap_size = 2ULL * 1024 * 1024 * 1024;  // 2 GB heap
  std::string shm_url = "/allocator_benchmark_mp";

  if (!backend.shm_init(hipc::MemoryBackendId(0, 0), heap_size, shm_url)) {
    std::cerr << "Error: Failed to initialize PosixShmMmap backend\n";
    return;
  }

  // Initialize allocator
  auto* alloc = backend.MakeAlloc<hipc::MultiProcessAllocator>();
  if (alloc == nullptr) {
    std::cerr << "Error: Failed to initialize MultiProcessAllocator\n";
    backend.shm_destroy();
    return;
  }

  // Create operation counter
  OperationCounter counter;

  // Launch worker threads
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([alloc, min_size, max_size, duration_sec, &counter]() {
      AllocatorBenchmarkWorker<hipc::MultiProcessAllocator> worker(
          alloc, min_size, max_size, duration_sec, &counter);
      worker.Run();
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();

  // Calculate and print results
  uint64_t total_ops = counter.GetTotalOps();
  double ops_per_sec = (static_cast<double>(total_ops) * 1000.0) /
                       static_cast<double>(elapsed_ms);

  std::cout << "\n=== MultiProcessAllocator Benchmark Results ===\n";
  std::cout << "Threads:          " << num_threads << "\n";
  std::cout << "Size Range:       " << FormatSize(min_size) << " - " << FormatSize(max_size) << "\n";
  std::cout << "Duration:         " << duration_sec << " seconds (actual: "
            << elapsed_ms << " ms)\n";
  std::cout << "Total Allocs:     " << FormatNumber(counter.GetAllocCount()) << "\n";
  std::cout << "Total Frees:      " << FormatNumber(counter.GetFreeCount()) << "\n";
  std::cout << "Total Operations: " << FormatNumber(total_ops) << "\n";
  std::cout << "Operations/sec:   " << std::fixed << std::setprecision(2)
            << FormatNumber(static_cast<uint64_t>(ops_per_sec)) << "\n";
  std::cout << "============================================\n";

  // Cleanup
  if (alloc != nullptr) {
    alloc->shm_detach();
  }
  backend.shm_destroy();
}

/**
 * Run benchmark with BuddyAllocator
 * @param num_threads Number of threads to use
 * @param size_range Size range configuration (min_size, max_size)
 * @param duration_sec Duration to run benchmark in seconds
 */
void BenchmarkBuddyAllocator(int num_threads, const SizeRange& size_range,
                            int duration_sec) {
  size_t min_size = size_range.min_size;
  size_t max_size = size_range.max_size;
  // Initialize backend with sufficient memory
  hipc::MallocBackend backend;
  size_t heap_size = 2ULL * 1024 * 1024 * 1024;  // 2 GB heap
  size_t alloc_size = sizeof(hipc::BuddyAllocator);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  // Initialize allocator
  auto* alloc = backend.MakeAlloc<hipc::BuddyAllocator>();

  // Create operation counter
  OperationCounter counter;

  // Launch worker threads
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([alloc, min_size, max_size, duration_sec, &counter]() {
      AllocatorBenchmarkWorker<hipc::BuddyAllocator> worker(
          alloc, min_size, max_size, duration_sec, &counter);
      worker.Run();
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();

  // Calculate and print results
  uint64_t total_ops = counter.GetTotalOps();
  double ops_per_sec = (static_cast<double>(total_ops) * 1000.0) /
                       static_cast<double>(elapsed_ms);

  std::cout << "\n=== BuddyAllocator Benchmark Results ===\n";
  std::cout << "Threads:          " << num_threads << "\n";
  std::cout << "Size Range:       " << FormatSize(min_size) << " - " << FormatSize(max_size) << "\n";
  std::cout << "Duration:         " << duration_sec << " seconds (actual: "
            << elapsed_ms << " ms)\n";
  std::cout << "Total Allocs:     " << FormatNumber(counter.GetAllocCount()) << "\n";
  std::cout << "Total Frees:      " << FormatNumber(counter.GetFreeCount()) << "\n";
  std::cout << "Total Operations: " << FormatNumber(total_ops) << "\n";
  std::cout << "Operations/sec:   " << std::fixed << std::setprecision(2)
            << FormatNumber(static_cast<uint64_t>(ops_per_sec)) << "\n";
  std::cout << "========================================\n";

  // Cleanup
  backend.shm_destroy();
}

/**
 * Run benchmark with standard malloc
 * @param num_threads Number of threads to use
 * @param size_range Size range configuration (min_size, max_size)
 * @param duration_sec Duration to run benchmark in seconds
 */
void BenchmarkMalloc(int num_threads, const SizeRange& size_range,
                    int duration_sec) {
  size_t min_size = size_range.min_size;
  size_t max_size = size_range.max_size;
  // Create operation counter
  OperationCounter counter;

  // Launch worker threads
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([min_size, max_size, duration_sec, &counter]() {
      MallocBenchmarkWorker worker(min_size, max_size, duration_sec, &counter);
      worker.Run();
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();

  // Calculate and print results
  uint64_t total_ops = counter.GetTotalOps();
  double ops_per_sec = (static_cast<double>(total_ops) * 1000.0) /
                       static_cast<double>(elapsed_ms);

  std::cout << "\n=== Standard malloc Benchmark Results ===\n";
  std::cout << "Threads:          " << num_threads << "\n";
  std::cout << "Size Range:       " << FormatSize(min_size) << " - " << FormatSize(max_size) << "\n";
  std::cout << "Duration:         " << duration_sec << " seconds (actual: "
            << elapsed_ms << " ms)\n";
  std::cout << "Total Allocs:     " << FormatNumber(counter.GetAllocCount()) << "\n";
  std::cout << "Total Frees:      " << FormatNumber(counter.GetFreeCount()) << "\n";
  std::cout << "Total Operations: " << FormatNumber(total_ops) << "\n";
  std::cout << "Operations/sec:   " << std::fixed << std::setprecision(2)
            << FormatNumber(static_cast<uint64_t>(ops_per_sec)) << "\n";
  std::cout << "==========================================\n";
}

}  // namespace bench

int main(int argc, char** argv) {
  // Parse command-line arguments
  if (argc < 5 || argc > 6) {
    bench::PrintUsage(argv[0]);
  }

  std::string alloc_type = argv[1];
  int num_threads = std::atoi(argv[2]);
  size_t min_size = hshm::ConfigParse::ParseSize(argv[3]);
  size_t max_size = hshm::ConfigParse::ParseSize(argv[4]);
  int duration_sec = (argc == 6) ? std::atoi(argv[5]) : 10;

  // Validate parameters
  if (num_threads < 1 || num_threads > 64) {
    std::cerr << "Error: num_threads must be between 1 and 64\n";
    bench::PrintUsage(argv[0]);
  }

  if (min_size == 0 || max_size == 0 || min_size > max_size) {
    std::cerr << "Error: Invalid size range (min_size must be > 0 and <= max_size)\n";
    bench::PrintUsage(argv[0]);
  }

  if (duration_sec < 1 || duration_sec > 3600) {
    std::cerr << "Error: duration_sec must be between 1 and 3600\n";
    bench::PrintUsage(argv[0]);
  }

  // Create size range configuration
  bench::SizeRange size_range(min_size, max_size);

  // Run the appropriate benchmark
  try {
    if (alloc_type == "mp") {
      bench::BenchmarkMultiProcessAllocator(num_threads, size_range, duration_sec);
    } else if (alloc_type == "buddy") {
      bench::BenchmarkBuddyAllocator(num_threads, size_range, duration_sec);
    } else if (alloc_type == "malloc") {
      bench::BenchmarkMalloc(num_threads, size_range, duration_sec);
    } else {
      std::cerr << "Error: Unknown allocator type '" << alloc_type << "'\n";
      std::cerr << "Valid types: mp, buddy, malloc\n";
      bench::PrintUsage(argv[0]);
    }
  } catch (const std::exception& e) {
    std::cerr << "Benchmark failed with exception: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Benchmark failed with unknown exception\n";
    return 1;
  }

  return 0;
}
