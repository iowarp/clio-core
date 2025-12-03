/**
 * Multi-process unit test for MultiProcessAllocator
 *
 * Usage: test_mp_allocator_multiprocess <rank> <duration_sec> <nthreads>
 *
 * rank 0: Initializes shared memory and optionally runs for duration_sec
 * rank 1+: Attaches to shared memory and runs for duration_sec
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>

#include "hermes_shm/memory/allocator/mp_allocator.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"

using namespace hshm::ipc;

// Shared memory configuration
constexpr size_t kShmSize = 512 * 1024 * 1024;  // 512 MB
const std::string kShmUrl = "/mp_allocator_multiprocess_test";

// Test statistics
struct ThreadStats {
  std::atomic<size_t> allocations{0};
  std::atomic<size_t> frees{0};
  std::atomic<size_t> bytes_allocated{0};
  std::atomic<size_t> errors{0};
};

// Worker thread function
void worker_thread(MultiProcessAllocator *allocator, int rank, int thread_id,
                   int duration_sec, ThreadStats &stats) {
  std::random_device rd;
  std::mt19937 gen(rd() + rank * 1000 + thread_id);
  std::uniform_int_distribution<size_t> size_dist(64, 4096);

  auto start = std::chrono::steady_clock::now();
  auto end = start + std::chrono::seconds(duration_sec);

  std::vector<OffsetPtr<>> active_allocations;
  active_allocations.reserve(100);

  while (std::chrono::steady_clock::now() < end) {
    // Decide: allocate or free
    bool should_allocate = active_allocations.empty() ||
                          (active_allocations.size() < 50 && (gen() % 2 == 0));

    if (should_allocate) {
      // Allocate
      size_t size = size_dist(gen);
      OffsetPtr<> ptr = allocator->AllocateOffset(size);

      if (ptr.IsNull()) {
        stats.errors++;
        continue;
      }

      // Write pattern to verify data integrity
      char *data = reinterpret_cast<char*>(allocator->alloc_.GetBackend().data_ + ptr.load());
      unsigned char pattern = static_cast<unsigned char>((rank * 100 + thread_id) & 0xFF);
      memset(data, pattern, size);

      // Verify immediately
      for (size_t i = 0; i < size; ++i) {
        if (static_cast<unsigned char>(data[i]) != pattern) {
          std::cerr << "Rank " << rank << " Thread " << thread_id
                    << ": Data corruption detected!" << std::endl;
          stats.errors++;
          break;
        }
      }

      active_allocations.push_back(ptr);
      stats.allocations++;
      stats.bytes_allocated += size;

    } else if (!active_allocations.empty()) {
      // Free random allocation
      std::uniform_int_distribution<size_t> idx_dist(0, active_allocations.size() - 1);
      size_t idx = idx_dist(gen);

      OffsetPtr<> ptr = active_allocations[idx];
      allocator->FreeOffset(ptr);

      active_allocations[idx] = active_allocations.back();
      active_allocations.pop_back();

      stats.frees++;
    }

    // Occasional yield to let other threads/processes run
    if (gen() % 100 == 0) {
      std::this_thread::yield();
    }
  }

  // Clean up remaining allocations
  for (const auto &ptr : active_allocations) {
    allocator->FreeOffset(ptr);
    stats.frees++;
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <rank> <duration_sec> <nthreads>" << std::endl;
    return 1;
  }

  int rank = std::atoi(argv[1]);
  int duration_sec = std::atoi(argv[2]);
  int nthreads = std::atoi(argv[3]);

  std::cout << "Rank " << rank << ": Starting test with " << nthreads
            << " threads for " << duration_sec << " seconds" << std::endl;

  // Create or attach to shared memory
  PosixShmMmap backend;
  bool success = false;

  if (rank == 0) {
    // Rank 0 initializes
    std::cout << "Rank 0: Initializing shared memory" << std::endl;
    success = backend.shm_init(MemoryBackendId(0, 0), kShmSize, kShmUrl);
    if (!success) {
      std::cerr << "Rank 0: Failed to initialize shared memory" << std::endl;
      return 1;
    }
  } else {
    // Other ranks attach to existing shared memory
    std::cout << "Rank " << rank << ": Attaching to shared memory" << std::endl;
    success = backend.shm_attach(kShmUrl);
    if (!success) {
      std::cerr << "Rank " << rank << ": Failed to attach to shared memory" << std::endl;
      return 1;
    }
  }

  // Initialize or attach allocator
  MultiProcessAllocator allocator;

  if (rank == 0) {
    std::cout << "Rank 0: Initializing allocator" << std::endl;
    allocator.shm_init(backend, 0);
    std::cout << "Rank 0: Allocator initialized successfully" << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Attaching to allocator" << std::endl;

    // Give rank 0 time to fully initialize before we try to attach
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Attach to existing allocator without reinitializing
    allocator.shm_attach(backend);
    std::cout << "Rank " << rank << ": Attached to allocator successfully" << std::endl;
  }

  // Run test if duration > 0
  if (duration_sec > 0) {
    std::vector<std::thread> threads;
    std::vector<ThreadStats> thread_stats(nthreads);

    std::cout << "Rank " << rank << ": Starting " << nthreads << " worker threads" << std::endl;

    for (int i = 0; i < nthreads; ++i) {
      threads.emplace_back(worker_thread, &allocator, rank, i, duration_sec,
                          std::ref(thread_stats[i]));
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
      thread.join();
    }

    // Aggregate statistics
    size_t total_allocs = 0;
    size_t total_frees = 0;
    size_t total_bytes = 0;
    size_t total_errors = 0;

    for (const auto &stats : thread_stats) {
      total_allocs += stats.allocations.load();
      total_frees += stats.frees.load();
      total_bytes += stats.bytes_allocated.load();
      total_errors += stats.errors.load();
    }

    std::cout << "Rank " << rank << " Results:" << std::endl;
    std::cout << "  Total allocations: " << total_allocs << std::endl;
    std::cout << "  Total frees: " << total_frees << std::endl;
    std::cout << "  Total bytes allocated: " << total_bytes << std::endl;
    std::cout << "  Total errors: " << total_errors << std::endl;

    if (total_errors > 0) {
      std::cerr << "Rank " << rank << ": TEST FAILED with " << total_errors << " errors" << std::endl;
      return 1;
    }

    std::cout << "Rank " << rank << ": TEST PASSED" << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Initialization complete, exiting" << std::endl;
  }

  // Only rank 0 should clean up shared memory, and only if it ran the test
  // (if duration was 0, other ranks may still be using it)
  if (rank == 0 && duration_sec > 0) {
    std::cout << "Rank 0: Cleaning up shared memory" << std::endl;
    backend.shm_destroy();
  }

  return 0;
}
